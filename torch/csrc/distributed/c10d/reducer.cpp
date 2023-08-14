#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>

#include <functional>

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/utils/grad_layout_contract.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/utils/memory.h>

namespace c10d {
namespace {

constexpr int kUnsetDivFactor = -1;

// Macro that wraps TORCH_CHECK with DDP logging.
#define REDUCER_CHECK(cond, logger_, ...)             \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {               \
    if (!logger_.expired()) {                         \
      logger_.lock()->set_error_and_log(__VA_ARGS__); \
    }                                                 \
    TORCH_CHECK(false, ##__VA_ARGS__);                \
  }

} // namespace

C10_DEFINE_TYPED_REGISTRY( // NOLINT
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device);

namespace {

class CpuTimer : public Timer {
 public:
  explicit CpuTimer(c10::Device /* unused */) {}

  c10::optional<int64_t> measureDifference(Event start, Event end) override {
    int64_t start_time = getTimeRef(start);
    int64_t end_time = getTimeRef(end);
    // If cpu_end_time is not recorded in this iteration,
    // avg_time will return invalid value.
    // For some cases like DDP runs on non-sync mode, backward compute
    // end time can not be recorded in this iteration and thus can not
    // calculate the valid avg_time.
    // In this case, skip calculating the avg_time and return.
    if (end_time < start_time) {
      return c10::nullopt;
    }
    return end_time - start_time;
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCPU, CpuTimer);

std::vector<at::Tensor> extractTensors(const c10::IValue& result) {
  if (result.isPyObject()) {
    return result.toPyObjectHolder()->extractTensors();
  }
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList found ",
      result.tagKind());

  if (result.isTensor()) {
    return {result.toTensor()};
  }

  return result.toTensorVector();
}

} // namespace

//其次，在 Reducer 构建函数之中，会把进程组配置给 Reducer 的成员变量 process_group_ 之上。
Reducer::Reducer(
    std::vector<at::Tensor> params,
    std::vector<std::vector<size_t>> bucket_indices, // 桶信息
    std::vector<size_t> per_bucket_size_limits,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    std::vector<bool> expect_sparse_gradients,
    int64_t bucket_bytes_cap,
    bool find_unused_parameters,
    bool gradient_as_bucket_view,
    std::unordered_map<size_t, std::string> param_names,
    int64_t first_bucket_bytes_cap)
    : params_(std::move(params)),
      process_group_(std::move(process_group)), // 在这里
      expect_sparse_gradients_(std::move(expect_sparse_gradients)),
      expect_autograd_hooks_(false),
      require_finalize_(false),
      next_bucket_(0),
      has_marked_unused_parameters_(false),
      find_unused_parameters_(find_unused_parameters),
      gradient_as_bucket_view_(gradient_as_bucket_view),
      local_used_map_reduced_(false),
      num_iterations_(0),
      num_buckets_ready_(0),
      has_rebuilt_bucket_(false),
      bucket_bytes_cap_(bucket_bytes_cap),
      div_factor_(kUnsetDivFactor),
      static_graph_(false),
      comm_hook_(nullptr),
      ddp_debug_level_(debug_level()),
      param_names_(std::move(param_names)),
      first_bucket_bytes_cap_(first_bucket_bytes_cap) {
      /*具体逻辑如下：

看看本模块是不是多设备模块，具体是: 遍历张量，得到张量的设备，把设备插入到一个set结构之中，如果set内的设备多于一个，是多设备
如果 expect_sparse_gradients没有设置，就把expect_sparse_gradients_初始化为false。
调用 initialize_buckets 初始化 buckets 并尽可能按照逆序将 parameters 分配到 buckets 之中，这样按桶通信就可以提高效率。后续在运行时候也可能再次重新初始化桶。
为每个 parameter 加上 grad_accumulator，它们在 backward 时负责梯度同步。
因为这些variables是autograd图的叶子张量，所以它们的grad_fn都被设置为 gradient accumulation function。
Reducer保存了指向这些functions的指针，这样Reducer就可以知道它们在autograd传播之中是否被使用，如果没有使用，那么就把这些functions的梯度张量（grad tensors）设置为规约就绪状态。
遍历张量，为每个张量生成一个类型为VariableIndex的变量index。
得到Variable::AutogradMeta的grad_accumulator_，即用于累加叶子 Variable 的梯度累加器。
把reducer的autograd_hook函数添加进去每个grad_accumulator_之中，变量index是hook的参数。这个 hook 挂在 autograd graph 之上，在 backward 时负责梯度同步。grad_accumulator 执行完后，autograd_hook 就会运行。
gradAccToVariableMap_ 存了grad_accumulator & index 的对应关系（函数指针和参数张量的对应关系），这样以后在 autograd graph 遍历寻找 unused parameters 就方便了。
初始化 backward_stats_。
调用 initialize_local_used_map 初始化各种 unused map。
    */
  C10_LOG_API_USAGE_ONCE("torch.distributed.ddp.reducer");
  TORCH_INTERNAL_ASSERT(!params_.empty(), "Expected at least one parameter.");

  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    LOG(INFO) << "Reducer initialized with bucket_bytes_cap: "
              << bucket_bytes_cap_
              << " first_bucket_bytes_cap: " << first_bucket_bytes_cap;
  }
  // Check whether the module is multi_device_module
  //// 看看本模块是不是多设备模块
  {
    std::set<int> unique_devices;
    for (const auto& v : params_) { // 遍历张量
      auto device_idx = int(v.device().index()); // 得到张量的设备
      if (unique_devices.find(device_idx) == unique_devices.end()) {
        unique_devices.insert(device_idx);  // 把设备插入到一个set结构之中
        if (unique_devices.size() > 1) {  // 如果set内的设备多于一个，是多设备
          is_multi_device_module_ = true;
          break;
        }
      }
    }
  }

  // For CUDA, record events only for single device module.
  c10::Device device = params_[0].device();
  if (!(device.is_cuda() && is_multi_device_module_)) {
    timer_ = TimerRegistry()->Create(device.type(), device);
  }

  // If `expect_sparse_gradients` is not specified, initialize it such that
  // we do not expect sparse gradients for any parameter.
  if (expect_sparse_gradients_.empty()) {
    expect_sparse_gradients_ = std::vector<bool>(params_.size(), false);
  }
  TORCH_INTERNAL_ASSERT(expect_sparse_gradients_.size() == params_.size());

  // Initialize variable bucketing.
  // This can be reinitialized later after capturing runtime information.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    initialize_buckets(std::move(bucket_indices));  //初始化桶
  }

  // All variables are expected to have their `grad_fn` set to the gradient
  // accumulation function (since they are leafs in the autograd graph).
  // We store pointers to these functions such that we can check if they are
  // used in an autograd pass. If they are not, we know their grad tensors
  // can be marked as ready for reduction.
  {
    const auto variable_count = params_.size();
    grad_accumulators_.resize(variable_count);

    // 以下两个for循环会遍历所有的张量
    for (const auto variable_index : c10::irange(variable_count)) { // 只有replicas_[0]有意义
      auto& variable = params_[variable_index];

      // The gradient accumulator function is lazily initialized once.
      // Therefore we can use its presence in the autograd graph as
      // evidence that the parameter has participated in an iteration.
      // // 得到一个张量的grad_accumulator
      //// 得到Variable::AutogradMeta的grad_accumulator_，即，用于累加叶子 Variable 的梯度累加器
      auto grad_accumulator = torch::autograd::impl::grad_accumulator(variable); // 给grad_accumulators_分配内存

#ifndef _WIN32
      using torch::distributed::autograd::ThreadLocalDistAutogradContext;
#endif
      // Hook to execute after the gradient accumulator has executed.
      hooks_.emplace_back(
          // 累加器添加hook,这个 hook 挂在 autograd graph 之上，在 backward 时负责梯度同步。
          // grad_accumulator 执行完后，autograd_hook 就会运行
          grad_accumulator->add_post_hook(
              torch::make_unique<torch::autograd::utils::LambdaPostHook>(
                  [=](const torch::autograd::variable_list& outputs,
                      const torch::autograd::variable_list& /* unused */) {
#ifndef _WIN32
                    this->rpc_context_.set(
                        ThreadLocalDistAutogradContext::getContextPtr());
#endif
                    this->autograd_hook(variable_index);  // 把reducer的autograd_hook函数添加进去
                    return outputs;
                  })),
          grad_accumulator);

      // Map raw function pointer to parameter index.
      // This is used later on when the autograd graph is traversed
      // to check for parameters for which no gradient is computed, if
      // find_unused_parameters=True.
      // Note that the mapping of gradient accumulator to variable should be
      // one to one as we deduplicate shared parameters before constructing
      // Reducer.
      // gradAccToVariableMap_ 存了grad_accumulator & index 的对应关系（函数指针和参数张量的对应关系），这样以后在 autograd graph 遍历寻找 unused parameters 就方便了
      if (find_unused_parameters_) {
        gradAccToVariableMap_[grad_accumulator.get()] = variable_index;
      }

      numGradHooksTriggeredMap_[variable_index] = 0;

      // The gradient accumulator is stored as weak_ptr in the autograd
      // metadata of the variable, so we have to keep it alive here for
      // the raw pointer to be valid.
      REDUCER_CHECK(
          grad_accumulators_[variable_index] == nullptr,
          logger_,
          c10::str(
              "Reducer tried to register duplicate grad accumulator for variable ",
              variable_index));

 // 把这个张量的 grad_accumulator 复制到 grad_accumulators_
      grad_accumulators_[variable_index] = std::move(grad_accumulator);
    }
  }

  // Initialize backward stats vector.
  {
    const auto variable_count = params_.size();
    backward_stats_.resize(variable_count);
  }

  // See Note [Skip allreducing local_used_map_dev]
  if (find_unused_parameters_) {
    initialize_local_used_map();
  }
}

// Note [Skip allreducing local_used_map_dev]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If find_unused_parameters_ is set to false, there is no need to allreduce
// local_used_map_dev_, because all parameters will be reduced anyway.
// Therefore, we can avoid allocating memory for local_used_map and
// local_used_map_dev_ if find_unused_parameters_ is false.

/*
5.7.3 comm_hook_
5.7.3.1 概念
我们通过 [DDP Communication Hook] 来看看概念。

DDP通信钩子是一种增强功能，它提供了一个钩子，其可用于覆盖DDP来进行跨rank梯度通信，这可用于梯度压缩/GossipGrad等算法。可以使用Python API register_comm_hook来注册钩子函数。

如果未注册DDP通信钩子（DDP communication hook），则reducer只需调用allreduce即可对桶进行规约。如果注册了，则会调用钩子并使用future work handle来处理。如果注册，reducer也会跳过"将梯度除以世界大小（world size）" 这个步骤。这样做的目的是：通信钩子可以完全覆盖我们执行通信的方式，用户可以完全控制如何处理梯度。

PythonCommHook是CommHookInterface的子类，其可以注册一个 Python 钩子。此外，还有一些内置的C++钩子实现，可以通过调用Python API register_builtin_comm_hook来指定。
*/
// Note [DDP Communication Hook]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If DDP communication hook is not registered, the reducer reduces the buckets
// by just calling allreduce. If registered, it calls the hook and uses future
// work handle. If registered, reducer also skips dividing grads by world size.
// The reason for this is that the communication hook is expected to completely
// override how we perform communication and the user should have complete
// control over how the grads are handled.
//
// DDP communication hook is an enhancement that provides a hook which can be
// used to override how DDP communicates gradients across ranks, this can be
// used for algorithms like Gradient Compression/GossipGrad. This hook can be
// registered from Python API using `register_comm_hook`. `PythonCommHook`
// enables registering a Python hook and is a subclass of `CommHookInterface`.
// Additionally, there are also some built-in C++ hook implementations that can
// be specified by calling `register_builtin_comm_hook` from Python API.

Reducer::~Reducer() noexcept(false) {
  // Remove all hooks on variables registered by this Reducer. This is necessary
  // to make DDP failure recoverable. Otherwise, multiple Reducer instances
  // (from recoveries) will add their hooks to the original model, and those
  // hooks will try to invoke methods on a deleted Reducer objects.
  for (auto& hook : hooks_) {
    auto& key = hook.first;
    auto& grad_accumulator = hook.second;

    TORCH_INTERNAL_ASSERT(
        grad_accumulator->del_post_hook(key),
        "Reducer attempts to delete a non-existing hook.");
  }
}

bool Reducer::dynamic_graph_find_unused() {
  return !static_graph_ && find_unused_parameters_;
}

bool Reducer::static_graph_first_iteration() {
  return static_graph_ && num_iterations_ == 1;
}

bool Reducer::static_graph_after_first_iteration() {
  return static_graph_ && num_iterations_ > 1;
}

bool Reducer::ddp_graph_static() {
  std::lock_guard<std::mutex> lock(mutex_);
  return ddp_graph_static_;
}

/*
5.6.1 论文
此处可以结合论文看看。

全局未使用参数（Globally Unused Parameters）的梯度在向前和向后过程中应保持不变。检测未使用的参数需要全局信息，因为在一个DDP过程中，一个参数可能在一次操作中不存在，但可能在另一个过程的同一次迭代中参与训练。因此DDP在位图中维护本地未使用的参数信息，并启动额外的AllReduce以收集全局位图。由于位图比张量尺寸小得多，因此模型中的所有参数共享同一位图，而不是创建每桶位图（per-bucket bitmaps）。位图位于CPU上，以避免为每次更新启动专用CUDA内核。但是，某些ProcessGroup后端可能无法在CPU 张量上运行AllReduce。例如，ProcessGroupNCCL仅支持CUDA张量。此外，由于DDP应该与任何定制的ProcessGroup后端一起工作，它不能假设所有后端都支持CPU张量。为了解决这个问题，DDP在同一设备上维护另一个位图作为第一个模型参数，并调用非阻塞拷贝操作（non-blocking copy）将CPU位图移动到设备位图以进行集合通信。

5.6.2 初始化
初始化函数如下：

2.3.1 BucketReplica成员变量
我们先回忆一下BucketReplica的几个成员变量。

at::Tensor contents ：把桶的内容展平的结果，即Flattened (1 dimensional) 之后的结果。
std::vector<at::Tensor> bucket_views_in ：提供了从输入角度在 contents 之中查看具体梯度的方法。
std::vector<at::Tensor> bucket_views_out ：提供了从输入角度在 contents 之中查看具体梯度的方法。
关于 std::vector<at::Tensor> bucket_views_in 和 std::vector<at::Tensor> bucket_views_out 的进一步说明：

这两个变量提供在 contents 之中操作具体梯度的方法，或者说，它们提供了视图（views），该视图可以操作contents 之中每个张量的梯度。用户把这两个变量作为入口点来把每个梯度的数据从 content 之中移入和移出。
在 PyTorch 之中，视图是指创建一个方便查看的东西，视图与原数据共享内存，它只是将原有的数据进行整理，直接显示其中部分内容或者进行重排序后再显示出来。
也需要对几个 PyTorch 函数进行说明。

as_strided ：依据现有tensor以及给定的步长来创建一个视图（类型仍然为tensor），需要注意，这里的结果是视图，所以这个张量依然和原始张量共享内存。
narrow ：返回一个新的张量，其是原来张量的缩小版，但是这个张量依然和原始张量共享内存。
BucketReplica 逻辑具体如下图：

+------------------------------------------+
| BucketReplica                            |
|                                          |
|       vector<Tensor> bucket_views_in +--------------------+
|                                          |                |
|                                          |                |
|       vector<Tensor> bucket_views_out +--------------+    |
|                                          |           |    |
|                                          |           |    |
|                                          |           v    v
|                                          |     +-----+----+--------------------------+
|       Tensor contents  +---------------------> |Flattened (Tensor1, Tensor2, Tensor3)|
|                                          |     +-------------------------------------+
|                                          |
|                                          |
|       vector<Tensor> variables  +------------>  [Tensor1,Tensor2,Tensor3]
|                                          |
|                                          |
|                                          |
+------------------------------------------+

2.3.2 调用
如何调用？如果gradient_as_bucket_view_设置为true，则有两种情况需要处理：

rebuild_buckets 之中可以在initialize_bucket内调用initialize_bucket_view，如果grad在上一次迭代中已经定义/计算过，则需要将旧的grad复制到新的bucket_view中，并让grad指向新的bucket_view，
在构造过程中，也可以在initialize_bucket中调用initialize_bucket_views。在构造期间不会定义梯度，在这种情况下，不要让梯度指向bucket_view，因为对于全局未使用的参数，梯度应保持为未定义。
2.4 初始化本地使用变量
initialize_local_used_map此处是初始化 local_used_maps_，我们回忆一下论文内容，local_used_maps_ 就是用来查找全局未使用参数（Globally Unused Parameters）：

全局未使用参数（Globally Unused Parameters）的梯度在向前和向后过程中应保持不变。检测未使用的参数需要全局信息，因为在一个DDP过程中，一个参数可能在一次操作中不存在，但可能在另一个过程的同一次迭代中参与训练。因此DDP在位图中维护本地未使用的参数信息，并启动额外的AllReduce以收集全局位图。由于位图比张量尺寸小得多，因此模型中的所有参数共享同一位图，而不是创建每桶位图（per-bucket bitmaps）。位图位于CPU上，以避免为每次更新启动专用CUDA内核。但是，某些ProcessGroup后端可能无法在CPU 张量上运行AllReduce。例如，ProcessGroupNCCL仅支持CUDA张量。此外，由于DDP应该与任何定制的ProcessGroup后端一起工作，它不能假设所有后端都支持CPU张量。为了解决这个问题，DDP在同一设备上维护另一个位图作为第一个模型参数，并调用非阻塞拷贝操作（non-blocking copy）将CPU位图移动到设备位图以进行集合通信。




初始化流程大致如下：

                                    +
                                    |
                                    |
                                    v
                  rpc_context_ = ThreadLocalDistAutogradContext
                                    +
                                    |
                                    |
                                    v
                  buckets_ & variable_locators_ (clear & resize)
                                    +
                                    |
                                    |
                                    v
+----------------------->  from 0 ~ bucket_count :  +--------------------------->
|                                                                                +
|                                                                                |
|      +-------------------------------------------------------------------+     |
|      | init Bucket          set bucket_indices                           |     |
|      |                            +                                      |     |
|      |                            |                                      |     |
|      |                            |                                      |     |
|      |                            v                                      |     |
|      |   ^ +------------> from 0 ~ replica_count : +----------------->   |     |
|      |   |                                                           |   |     |
|      |   |  +---------------------------------------------------+    |   |     |
|      |   |  | init BucketReplica                                |    |   |     |
|      |   |  |                                                   |    |   |     |
<----+ |   +--+                                                   | <--+   | <---+
       |      |    bucket.replicas.push_back(std::move(replica))  |        |
       |      |                                                   |        |
       |      +----------------------+----------------------------+        |
       |                             |                                     |
       |                             |                                     |
       |                             v                                     |
       |             buckets_.push_back(std::move(bucket))                 |
       |                             +                                     |
       +-------------------------------------------------------------------+
                                     |
                                     v

得到的 Reducer 大致如下，这里需要注意的是 ，BucketReplica 每个桶只有一个：

            +----------------------------------------+                 +------------------+
            |tensor index 4, tensor index 5, tensor 6| <------+        | index 2, index 3 |
            +----------------------------------------+        |        +--------------+---+
                                                              |                       ^
                                                              |                       |
+---------------------------+   +---------------------------------------------------------+
| Reducer                   |   | +----------------------------------+     +------------+ |
|                           |   | |Bucket                     |      |     |Bucket    | | |
|                           |   | |                           +      |     |          | | |
| vector<Bucket> buckets_ +---> | | vector<size_t> variable_indices  |     | indices ++ | |
|                           |   | |                                  |     |            | |
|                           |   | |  vector<BucketReplica> replicas  | ... | replicas   | |
|                           |   | |                         +        |     |   +        | |
|                           |   | |                         |        |     |   |        | |
|                           |   | +----------------------------------+     +------------+ |
|                           |   |                           |                  |          |
+---------------------------+   +---------------------------------------------------------+
                                                            |                  |
                                                            |                  |
                                                            v                  v
                          +---------------------------------------+   +-------------------+
                          |  +----------------------------------+ |   | +---------------+ |
                          |  | BucketReplica                    | |   | | BucketReplica | |
                          |  |                                  | |   | |               | |
                          |  |                                  | |   | |               | |
                          |  |  vector<Tensor> bucket_views_in  | |   | |   views_in    | |
                          |  |                                  | |   | |               | |
                          |  |  vector<Tensor> bucket_views_out | |   | |   views_out   | |
                          |  |                                  | |   | |               | |
                          |  |  Tensor contents                 | |   | |   contents    | |
                          |  |                                  | |   | |               | |
                          |  |  vector<Tensor> variables        | |   | |   variables   | |
                          |  |                     +            | |   | |      +        | |
                          |  +----------------------------------+ |   | +---------------+ |
                          +---------------------------------------+   +-------------------+
                                                   |                           |
                                                   |                           |
                                                   v                           v
                                   +---------------+------------+    +---------+----------+
                                   |Tensor 4, Tensor 5, Tensor 6|    | Tensor 2, Tensor 3 |
                                   +----------------------------+    +--------------------+
*/
void Reducer::initialize_local_used_map() {
  const auto variable_count = params_.size();
  at::TensorOptions options;
  options = options.dtype(at::kInt);

  // Deliberately don't pin the memory even if local_used_map_dev_ will
  // be cuda. See Note [local_used_map_ -> local_used_map_dev copying]
  local_used_map_ = at::zeros({static_cast<long>(variable_count)}, options);

  // This tensor needs to be on the same device as the replica params because
  // backend such as NCCL may not support CPU tensors, and hence it might not
  // work if we always put it on CPU.
  options = options.device(params_[0].device());
  local_used_map_dev_ = at::empty({static_cast<long>(variable_count)}, options);
}

void Reducer::check_grad_layout(
    const at::Tensor& grad,
    const at::Tensor& bucket_view) {
  // Ensure that the gradient type matches the bucket type.
  REDUCER_CHECK(
      grad.options().type_equal(bucket_view.options()),
      logger_,
      c10::str("Expected ", bucket_view.toString(), ", got ", grad.toString()));

  TORCH_INTERNAL_ASSERT(grad.device() == bucket_view.device());
  TORCH_INTERNAL_ASSERT(grad.numel() == bucket_view.numel());
  // AccumulateGrad doesn't HAVE to obey the grad layout contract.
  // The penalty for disobedience is reduced performance, not numerical
  // death. Warnings here help diagnose poor DDP performance.
  if (grad.strides() != bucket_view.strides()) {
    TORCH_WARN_ONCE(
        "Grad strides do not match bucket view strides. "
        "This may indicate grad was not created according to the "
        "gradient layout contract, or that the param's strides "
        "changed since DDP was constructed.  This is not an error, "
        "but may impair performance.\n"
        "grad.sizes() = ",
        grad.sizes(),
        ", strides() = ",
        grad.strides(),
        "\n",
        "bucket_view.sizes() = ",
        bucket_view.sizes(),
        ", strides() = ",
        bucket_view.strides());
  }
  if (!gradient_as_bucket_view_) {
    TORCH_INTERNAL_ASSERT(!grad.is_alias_of(bucket_view));
  }
}

//autograd_hook 最终调用到 mark_variable_ready_dense，这里进而通过 variable_locators_ 来确定桶，然后进行后续操作。
void Reducer::mark_variable_ready_dense(size_t variable_index) {
  const auto& bucket_index = variable_locators_[variable_index];  // 找到张量对应的桶index
  auto& bucket = buckets_[bucket_index.bucket_index]; // 找到桶
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];  // 找到了张量
  auto& bucket_view = bucket.bucket_views_in[bucket_index.intra_bucket_index];

  // Copy the contents of the gradient tensor to the corresponding part of the
  // bucket's flattened gradient tensor.
  // If the gradient is not set, we assume it wasn't computed as part of the
  // current backwards pass, and we zero the part of the bucket it would
  // otherwise hold.
  runGradCallbackForVariable(variable, [&](auto& grad) {
    if (grad.defined()) {
      this->check_grad_layout(grad, bucket_view);
      // When gradient_as_bucket_view_ is false, or even when
      // gradient_as_bucket_view_ is true, in rare cases users may set grad to
      // be None after every iteration. In these cases, grad and bucket_view are
      // pointing to different storages and thus need to copy grads to
      // bucket_view. If gradient_as_bucket_view_ is set as true, let grad point
      // to bucket_view. If grad has already been set as views of buckets in
      // previous iterations, no copy is needed.
      if (!grad.is_alias_of(bucket_view)) {
        if (comm_hook_ == nullptr) {
          auto wrapped =
              at::native::wrapped_scalar_tensor(double(1.) / div_factor_);
          if (!grad.requires_grad()) {
            // Divides while copying into the bucket view to save one scan over
            // all the input parameters.
            at::mul_out(bucket_view, grad, wrapped);
          } else {
            // If DDP is running with create_graph=True, gradients require_grad
            // themselves in order to compute higher order derivatives. However,
            // DDP will not sync up these gradients currently (see
            // https://github.com/pytorch/pytorch/issues/63812).
            C10_LOG_EVERY_N(WARNING, 1000)
                << "Using DistributedDataParallel with create_graph=True "
                << " is not well-supported. The higher-order gradient will "
                << " not be synchronized across ranks, and backpropagation "
                << " through all_reduce operations will not occur. If you require "
                << " DDP to work with higher-order gradients for your use case, "
                << " please ping https://github.com/pytorch/pytorch/issues/63929";
            auto div_result = at::mul(grad, wrapped);
            bucket_view.copy_(div_result);
          }
        } else {
          bucket_view.copy_(grad);
        }

        if (gradient_as_bucket_view_) {
          // Let grad point to bucket_view buffer.
          grad = bucket_view;
          // The grad is modified and need to be written back.
          return true;
        }
      } else {
        // If grad and bucket view point to the same storage, no need to copy.
        if (comm_hook_ == nullptr) {
          bucket_view.div_(div_factor_);
        }
      }
    } else {
      // Gradient is undefined. When find_unused_parameters=True, ensure it is
      // not marked as locally used, otherwise we will be allreducing zero's
      // instead of not touching .grad field of parameter.
      if (this->dynamic_graph_find_unused() ||
          this->static_graph_first_iteration()) {
        REDUCER_CHECK(
            local_used_map_[variable_index].item<int>() == 0,
            logger_,
            "Encountered gradient which is undefined, but still allreduced by "
            "DDP reducer. This indicates a bug in DDP implementation, please "
            "report a bug with a repro to PyTorch.");
      }
      bucket_view.zero_();
    }
    // The grad is not modified and doesn't need to be written back.
    return false;
  });
}

void Reducer::mark_variable_ready_sparse(size_t variable_index) {
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];

  runGradCallbackForVariable(variable, [&](auto& grad) {
    REDUCER_CHECK(
        grad.defined(), logger_, "Expected sparse gradient to be defined.");
    REDUCER_CHECK(
        grad.options().layout() == c10::kSparse,
        logger_,
        "Expected variable to have sparse gradient.");

    // Sparse tensors cannot be grouped together with other sparse tensors in a
    // single reduction operation like we can for dense tensors. Therefore, the
    // `offsets` and `lengths` vectors in the bucket struct are empty, and
    // there is no pre-existing accumulation tensor.
    // Directly assign the sparse tensor to the `gradients` field.
    bucket.gradients = grad;
    // If no DDP comm hook is registered, the allreduce only sums up the
    // value, and a separate division is required.
    if (comm_hook_ == nullptr) {
      bucket.gradients.div_(div_factor_);
    }
    // The grad is modified in place and needs to be written back.
    return true;
  });
}

std::vector<c10d::GradBucket> Reducer::get_grad_buckets(
    bool return_zero_tensors) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<c10d::GradBucket> gradBuckets;
  gradBuckets.reserve(buckets_.size());
  for (const auto i : c10::irange(buckets_.size())) {
    auto& bucket = buckets_[i];
    auto variables_for_bucket = get_variables_for_bucket(i, bucket);
    gradBuckets.emplace_back(
        i,
        buckets_.size(),
        return_zero_tensors ? at::zeros_like(bucket.gradients)
                            : bucket.gradients,
        bucket.offsets,
        bucket.lengths,
        bucket.sizes_vec,
        variables_for_bucket);
  }
  return gradBuckets;
}

void Reducer::set_forward_pass_work_handle(
    c10::intrusive_ptr<c10d::Work> forwardPassWorkHandle,
    bool useStaticWorldSize) {
  std::lock_guard<std::mutex> lock(mutex_);
  forwardPassWorkHandle_.workHandle = std::move(forwardPassWorkHandle);
  forwardPassWorkHandle_.useStaticWorldSize = useStaticWorldSize;
}

at::Tensor Reducer::get_local_used_map_on_device() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_used_map_dev_;
}

void Reducer::push_rebuilt_params_for_all_indices() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!should_rebuild_buckets() || !rebuilt_param_indices_.empty()) {
    return;
  }
  const auto variable_count = params_.size();
  for (const auto variable_index : c10::irange(variable_count)) {
    push_rebuilt_params(variable_index);
  }
}

//其次，push_rebuilt_params_for_all_indices 会遍历每个 replica，针对 replica 之中的每个 variable 进行设置。
void Reducer::push_rebuilt_params(const size_t& index) {
  rebuilt_params_.push_back(params_[index]);
  rebuilt_param_indices_.push_back(index);
}

void Reducer::set_divide_factor() {
  // If it was scheduled, wait on allreduce in forward pass that tells us
  // division factor based on no. of currently participating processes.
  if (div_factor_ == kUnsetDivFactor) {
    div_factor_ = process_group_->getSize();
    auto& workHandle = forwardPassWorkHandle_.workHandle;
    if (workHandle && !forwardPassWorkHandle_.useStaticWorldSize) {
      workHandle->wait();
      // PyProcessGroup::PyWork doesn't expose value, so fetch it from the
      // future
      auto results = extractTensors(workHandle->getFuture()->value());

      // Guard against the results being empty
      TORCH_INTERNAL_ASSERT(!results.empty());
      at::Tensor& res = results.front();
      div_factor_ = res.item().to<int>();
    }
  }
}

// Right now delay_all_reduce is only called when static_graph_=true and
// num_iterations_==1.
void Reducer::delay_all_reduce() {
  std::lock_guard<std::mutex> lock(this->mutex_);

  if (should_collect_runtime_stats()) {
    record_backward_compute_end_time();
    record_backward_comm_start_time();
  }

  // launch all reduce local used map
  all_reduce_local_used_map();

  // prepare to set unused_parameters_, if it is static graph,
  // unused_parameters_ will not change after 1st iteration.
  unused_parameters_.clear();

  // copy all gradients to buckets
  for (const auto variable_index : c10::irange(params_.size())) {
    // set unused_parameters_
    if (numGradHooksTriggeredMap_[variable_index] == 0) { // 如果为0，则插入unused_parameters_
      unused_parameters_.push_back(variable_index);
    }
    require_finalize_ = true;
    set_divide_factor();
    if (expect_sparse_gradients_[variable_index]) {
      mark_variable_ready_sparse(variable_index);
    } else {
      mark_variable_ready_dense(variable_index);
    }
  }

  // To avoid confusion around why static graph is picking up
  // some parameters as unused on a rank vs not, we log
  // unused parameter names for each rank for better
  // debugability when TORCH_DISTRIBUTED_DEBUG is set to
  // INFO or DETAIL
  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    // construct one string to output
    std::ostringstream unused_params_stream;

    for (const auto& unused_index : unused_parameters_) {
      auto param_name = param_names_.find(unused_index);
      TORCH_INTERNAL_ASSERT(
          param_name != param_names_.end(),
          "Expected to find parameter name from unused parameters map in debug mode.");
      // Add the param_name
      unused_params_stream << "{" << param_name->second << "," << unused_index
                           << "}";
    }

    // Each rank prints out all the unused parameters detected
    if (!unused_parameters_.empty()) {
      LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
                << "Parameter(s) (in the format of {param_name, index}): "
                << unused_params_stream.str()
                << " is(are) unused during first iteration. Since"
                << " static_graph=True is enabled for DDP, we expect"
                << " this set of unused parameters to remain consistent"
                << " on this rank throughout the training.";
    }
  }

  // launch all reduces for all buckets
  for (auto& bucket : buckets_) {
    all_reduce_bucket(bucket);
  }

  finalize_backward();
}

void Reducer::set_logger(std::weak_ptr<c10d::Logger> logger) {
  logger_ = logger;
}

// The function `autograd_hook` is called after the gradient for a
// model parameter has been accumulated into its gradient tensor.
// This function is only to be called from the autograd thread.
//如何使用？在静态图情况下，如果不是第一次迭代（此时刚刚产生梯度），就会把 numGradHooksTriggeredMapPerIteration_[index] 递减，
//如果为0，就说明该变量就绪，可以进行集合操作梯度规约了。

//finalize_bucket_dense 和 finalize_backward 都会重置。
void Reducer::autograd_hook(size_t index) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  // Ignore if we don't expect to be called.
  // This may be the case if the user wants to accumulate gradients
  // for number of iterations before reducing them.
  if (!expect_autograd_hooks_) {
    return;
  }

  grad_ready_order_indices_.push_back(index);

  // See Note [Skip allreducing local_used_map_dev]
  //在这里会记录，已经使用了。
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // Since it gets here, this param has been used for this iteration. We want
    // to mark it in local_used_map_. During no_sync session, the same var can
    // be set multiple times, which is OK as does not affect correctness. As
    // long as it is used once during no_sync session, it is marked as used.
    // Only set it as locally used if the grad is defined. Otherwise, hooks can
    // be fired  with undefined grads, such as when not all outputs are used in
    // DDP when computing loss. In this case, we don't want to mark it as
    // locally used to ensure we don't touch the parameter's .grad field.
    auto& variable = get_param_from_index(index);
    runGradCallbackForVariable(variable, [&](auto& grad) {
      if (grad.defined()) {
        local_used_map_[index] = 1;
      }
      // The gradient is never modified.
      return false;
    });
  }

  if (static_graph_first_iteration()) {
    numGradHooksTriggeredMap_[index] += 1;  // 静态图第一次迭代时候，这里会增加1
    return;  // 然后直接返回，注意！
  }

  // If `find_unused_parameters_` is true there may be model parameters that
  // went unused when computing the model output, they won't be part of the
  // autograd graph, and won't receive gradients. These parameters are
  // discovered in the `prepare_for_backward` function and their indexes stored
  // in the `unused_parameters_` vector.
  if (!has_marked_unused_parameters_) {
    has_marked_unused_parameters_ = true;
    for (const auto& unused_index : unused_parameters_) {
      mark_variable_ready(unused_index);
    }
  }

  // Rebuild bucket only if 1) it is the first time to rebuild bucket 2)
  // static_graph_ is true or find_unused_parameters_ is false,
  // 3) this backward pass needs to run allreduce.
  // Here, we just dump tensors and their parameter indices into
  // rebuilt_params_ and rebuilt_param_indices_ based on gradient arriving
  // order, and then at the end of finalize_backward(), buckets will be
  // rebuilt based on rebuilt_params_ and rebuilt_param_indices_, and then
  // will be broadcasted and initialized.
  // If it is static graph, after 1st iteration, check if a variable
  // is ready for communication based on numGradHooksTriggeredMap_.
  if (static_graph_after_first_iteration()) {
    REDUCER_CHECK(
        numGradHooksTriggeredMapPerIteration_[index] > 0,
        logger_,
        "Your training graph has changed in this iteration, ",
        "e.g., one parameter is unused in first iteration, but ",
        "then got used in the second iteration. this is not ",
        "compatible with static_graph set to True.");
    if (--numGradHooksTriggeredMapPerIteration_[index] == 0) {
      if (should_rebuild_buckets()) {
        push_rebuilt_params(index); // 插入列表
      }
      // Finally mark variable for which this function was originally called.
      mark_variable_ready(index);
    }
  } else {
    if (should_rebuild_buckets()) {
      push_rebuilt_params(index);
    }
    // Finally mark variable for which this function was originally called.
    mark_variable_ready(index);
  }
}

/*
5.6.5 使用
在 mark_variable_ready 时候会调用到 all_reduce_local_used_map，如果需要同步，这里进行同步。我们还是翻译一下注释：

DDP 用异步H2D来避免阻塞开销。异步复制和allreduce 会着眼于当前流，因此将正确排序。

关于主机操作的正确顺序也很重要。H2D copy_ 是按流排序的，而主机对 local_used_maps_ 的更改是按主机排序的。

如果大量积压的cuda流工作将 copy_ 操作推迟到将来，并且如果从现在到finalize_backward 之间没有发生阻塞调用，那么finalize_backward 会在流执行复制之前将主机上使用的本地映射重新归零，在这种情况下，copy_会读取到这些零，而不是我们在这里告诉它读取的值。

将 local_used_maps_[i] 复制到pinned临时内存（固定的缓存分配器应该异步提供）可以避免这种恶劣的、罕见的争用情况。

在希望使用所有参数的情况下，从现在到重新调零，DDP本身不会做任何阻塞工作，因此这种危险情况是真实存在的。

所以，Reducer 采用防御性操作，以确保 local_used_maps_tmp 与local_used_maps_[i] 不同。

*/
void Reducer::all_reduce_local_used_map() {
  // See Note [Skip allreducing local_used_map_dev]
  // H2D from local_used_map_ to local_used_map_dev_
  if (local_used_map_dev_.is_cuda()) {
    // Note [local_used_map_ -> local_used_map_dev copying]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We do async H2D to avoid the blocking overhead. The async copy and
    // allreduce respect the current stream, so will be sequenced
    // correctly.
    //
    // Correct sequencing with respect to host operations is also
    // essential. The H2D copy_ is stream ordered, while the host's
    // changes to local_used_map_ are host ordered. If a large backlog of
    // cuda-stream work pushes the copy_ far into the future, and if no
    // blocking calls occur between now and finalize_backward()** such
    // that finalize_backward() re-zeroes local_used_map_ on the host
    // before the stream executes the copy_, copy_ will read those zeros
    // instead of the values we thought we told it to read here. Copying
    // local_used_map_ to a pinned temporary (which the pinned caching
    // allocator should supply asynchronously) avoids this nasty, rare
    // race condition.
    //
    // ** In the hoped-for case where all params are used, DDP itself
    // won't do any blocking work between now and the re-zeroing, so the
    // danger is real.
    //
    // Defensively ensures local_used_map_tmp is distinct from
    // local_used_map_
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt(),
        true /* pinned_memory */);
    // Paranoid asserts here because in some workloads, the pinned
    // allocator behaves in a way we don't understand, and may be bugged.
    // See https://github.com/pytorch/pytorch/pull/54474
    TORCH_INTERNAL_ASSERT(local_used_map_tmp.is_pinned());
    TORCH_INTERNAL_ASSERT(
        local_used_map_tmp.data_ptr() != local_used_map_.data_ptr());
    local_used_map_tmp.copy_(local_used_map_);
    local_used_map_dev_.copy_(local_used_map_tmp, true);
  } else {
    local_used_map_dev_.copy_(local_used_map_, true);
  }
  std::vector<at::Tensor> temp_local_used_map_dev_vec_ = {local_used_map_dev_};
  local_used_work_ = process_group_->allreduce(temp_local_used_map_dev_vec_);
}

at::Tensor& Reducer::get_param_from_index(size_t index) {
  const auto& bucket_index = variable_locators_[index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  // Cannot simply access variable via `bucket.variables[variable_index]` since
  // return value is used in `runGradCallbackForVariable()` which does not
  // accept const tensors.
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  return variable;
}

void Reducer::checkAndRaiseMarkedTwiceError(size_t index) {
  // Something is wrong if all variables contained in this bucket have
  // already been marked as ready.
  // We don't expect the same variable to be marked ready twice.
  bool marked_twice =
      perIterationReadyParams_.find(index) != perIterationReadyParams_.end();

  if (marked_twice) {
    // Report index of param that has been marked twice. In debug mode, also
    // report fully qualified parameter name.
    auto param_name = param_names_.find(index);
    const bool found_param_name = param_name != param_names_.end();
    TORCH_INTERNAL_ASSERT(
        ddp_debug_level_ == c10d::DebugLevel::Off || found_param_name,
        "Expected to find parameter name in debug mode.");
    std::string paramInfo = c10::str(
        "Parameter at index ",
        index,
        found_param_name ? c10::str(" with name ", param_name->second) : "",
        " has been marked as ready twice. This means that multiple autograd engine ",
        " hooks have fired for this particular parameter during this iteration.");
    // param_names_ is empty in debug mode.
    if (!found_param_name) {
      paramInfo += c10::str(
          " You can set the environment variable TORCH_DISTRIBUTED_DEBUG to either",
          " INFO or DETAIL to print parameter names for further debugging.");
    }
    std::string common_error = c10::str(
        "Expected to mark a variable ready only once. ",
        "",
        "This error is caused by one of the following reasons: ",
        "1) Use of a module parameter outside the `forward` function. ",
        "Please make sure model parameters are not shared across multiple ",
        "concurrent forward-backward passes. or try to use _set_static_graph() ",
        "as a workaround if this module graph does not change ",
        "during training loop.",
        "2) Reused parameters in multiple reentrant backward passes. For ",
        "example, if you use multiple `checkpoint` functions to wrap the ",
        "same part of your model, it would result in the same set of ",
        "parameters been used by different reentrant backward passes ",
        "multiple times, and hence marking a variable ready multiple times. ",
        "DDP does not support such use cases in default. You can try to ",
        "use _set_static_graph() as a workaround if your module graph ",
        "does not change over iterations.");

    common_error += c10::str("\n", paramInfo);

    REDUCER_CHECK(
        has_marked_unused_parameters_,
        logger_,
        common_error,
        "3) Incorrect unused parameter detection. The return value of the ",
        "`forward` function is inspected by the distributed data parallel ",
        "wrapper to figure out if any of the module's parameters went ",
        "unused. For unused parameters, DDP would not expect gradients from ",
        "then. However, if an unused parameter becomes part of the autograd ",
        "graph at a later point in time (e.g., in a reentrant backward when ",
        "using `checkpoint`), the gradient will show up unexpectedly. If all ",
        "parameters in the model participate in the backward pass, you can ",
        "disable unused parameter detection by passing the keyword argument ",
        "`find_unused_parameters=False` to ",
        "`torch.nn.parallel.DistributedDataParallel`. If unused parameters ",
        "in the model do not change over iterations, You can try to use ",
        "_set_static_graph() as a workaround if this module graph does not ",
        "change during training loop.");
    REDUCER_CHECK(!has_marked_unused_parameters_, logger_, common_error);
  }
}

//就是如果某个variable是就绪状态，就插入到 perIterationReadyParams_。
/*
4.4 何时设定重建
重建仅在以下情况进行设定：

第一次重建存储桶

static_graph_ is true 或 find_unused_parameters_ is false

此反向传播过程需要运行allreduce。

在这里，我们只需基于梯度到达顺序将张量及其参数索引转储到rebuilt_params_和 rebuilt_param_indices_。然后在finalize_backward() 结束时，将基于rebuilt_params_和 rebuilt_param_indices_重建存储桶，然后广播和初始化存储桶。

此外，我们只需要转储一个副本的张量和参数索引。

以 mark_variable_ready 为例，其中就会调用 push_rebuilt_params(index) 来插入列表。
*/
void Reducer::mark_variable_ready(size_t variable_index) {
  REDUCER_CHECK(
      variable_index < variable_locators_.size(),
      logger_,
      "Out of range variable index.");

  checkAndRaiseMarkedTwiceError(variable_index);
  perIterationReadyParams_.insert(variable_index);
  backward_stats_[variable_index] =
      current_time_in_nanos() - backward_compute_start_time_;

  // Any time we mark a variable ready (be it in line due to unused parameters,
  // or via an autograd hook), we require a call to the finalize function. If
  // this doesn't happen before the next iteration (or call to
  // `prepare_for_backwards`), we know something is wrong.
  require_finalize_ = true;

  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];

  set_divide_factor();

  if (bucket.expect_sparse_gradient) {
    mark_variable_ready_sparse(variable_index);
  } else {
    mark_variable_ready_dense(variable_index);
  }

  // TODO(@pietern): Make this work for both CPU/CUDA tensors.
  // When using CPU tensors we don't need to do this.
  // Record event so that we can wait for all of them.
  // auto& event = bucket.events[bucket_index.intra_bucket_index];
  // event.record();

  // Check if this was the final gradient for this bucket.
  if (--bucket.pending == 0) {
    mark_bucket_ready(bucket_index.bucket_index);
  }

  // Run finalizer function and kick off reduction for local_used_map once the
  // final bucket was marked ready.
  if (next_bucket_ == buckets_.size()) {
    if (dynamic_graph_find_unused()) {
      all_reduce_local_used_map();
    }

    torch::autograd::Engine::get_default_engine().queue_callback([=] {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (should_collect_runtime_stats()) {
        record_backward_compute_end_time();
      }
      // Check that all buckets were completed and had their work kicked off.
      TORCH_INTERNAL_ASSERT(next_bucket_ == buckets_.size());
      if (static_graph_after_first_iteration() && should_rebuild_buckets()) {
        for (const auto& unused_index : unused_parameters_) {
          push_rebuilt_params(unused_index);
        }
      }
      this->finalize_backward();
    });
  }
}

c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_comm_hook(
    GradBucket& grad_bucket) {
  if (comm_hook_ == nullptr) {
    return run_allreduce_hook(grad_bucket);
  } else {
    return comm_hook_->runHook(grad_bucket);
  }
}

c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_allreduce_hook(
    GradBucket& grad_bucket) {
  _AllReduceBySumCommHook allreduce_hook(process_group_);
  return allreduce_hook.runHook(grad_bucket);
}

/*
最后，当需要对梯度做 all-reduce 时候，则会调用 process_group_->allreduce(tensors) 进行处理。

现在，我们就知道如何使用进程组了
*/
void Reducer::all_reduce_bucket(Bucket& bucket) {
  auto variables_for_bucket = get_variables_for_bucket(next_bucket_, bucket);
  // TODO(@pietern): Ensure proper synchronization with the CUDA events
  // that recorded copies into this `gradients` tensor. If these copies are
  // executed on non-default streams, the current stream for the device
  // that holds the `gradients` tensor must wait on these events.
  //
  // As long as autograd uses the default stream for every device,
  // these operations are implicitly sequenced, and we don't need to
  // do any extra synchronization here.
  const auto& tensor = bucket.gradients;

  GradBucket grad_bucket(
      next_bucket_,
      buckets_.size(),
      tensor,
      bucket.offsets,
      bucket.lengths,
      bucket.sizes_vec,
      variables_for_bucket);
  bucket.future_work = run_comm_hook(grad_bucket);
}

std::vector<at::Tensor> Reducer::get_variables_for_bucket(
    size_t bucket_index,
    const Bucket& bucket) const {
  // Check if we have cached mapping previously.
  if (has_rebuilt_bucket_ &&
      cached_variables_for_bucket_.find(bucket_index) !=
          cached_variables_for_bucket_.end()) {
    return cached_variables_for_bucket_[bucket_index];
  }
  std::vector<at::Tensor> variables_for_bucket;
  variables_for_bucket.reserve(bucket.variable_indices.size());
  for (const auto& variable_index : bucket.variable_indices) {
    // Grab bucket index where gradient is located using variable_locators_.
    auto& bucket_index_for_variable = variable_locators_[variable_index];
    // Grab the actual model parameter.
    auto& variable =
        bucket.variables[bucket_index_for_variable.intra_bucket_index];
    variables_for_bucket.emplace_back(variable);
  }

  if (has_rebuilt_bucket_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        cached_variables_for_bucket_.find(bucket_index) ==
        cached_variables_for_bucket_.end());
    cached_variables_for_bucket_.insert(
        {bucket_index, std::move(variables_for_bucket)});
    return cached_variables_for_bucket_[bucket_index];
  } else {
    return variables_for_bucket;
  }
}

// Called when the bucket at the specified index is ready to be reduced.
void Reducer::mark_bucket_ready(size_t bucket_index) {
  TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);

  // Buckets are reduced in sequence. Ignore this bucket if
  // it's not its turn to be reduced.
  if (bucket_index > next_bucket_) {
    return;
  }

  // Keep going, until we either:
  // - have kicked off reduction for all buckets, or
  // - found a bucket that's not yet ready for reduction.
  for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
       next_bucket_++) {
    num_buckets_ready_++;
    if (num_buckets_ready_ == 1 && should_collect_runtime_stats()) {
      record_backward_comm_start_time();
    }
    auto& bucket = buckets_[next_bucket_];
    all_reduce_bucket(bucket);
  }
}

void Reducer::install_futures(
    c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futs) {
  // Append instead of overwrite so that this method can be called multiple
  // times in one iteration.
  if (!installed_futures_) {
    installed_futures_ = std::move(futs);
  } else {
    installed_futures_->append(futs);
  }
}
/*

在初始化函数中有如何初始化 buckets_，核心是：

    找到本bucket在 bucket_indices 之中的 index。
    在 parameters 之中找到 index 对应的张量。
    在 BucketReplica 之中配置这些张量，就是本bucket应该规约的张量。


用图例表示如下，这里假设 bucket index 是 1，即第 2 个桶，所以 variable_indices 对应了 bucket_indices 中的相应部分。比如 BucketReplica[0] 里面是 Tensor 4,5,6，而variable_indices就是 Tensor 4,5,6 分别的 index。

下图中的 bucket_indices 是 Reducer 构造函数的参数之一。

+--------------------------------+   +------------------------------------+
|Reducer                         |   |                                    |
|                                |   |bucket 0, bucket 1, ...... bucket n |
|      vector<Bucket> buckets_ +---> |    +                               |
|                                |   |    |                               |
+--------------------------------+   +------------------------------------+
                                          |
                          +---------------+              +------------------------------+
                          |                         +--> | Tensor 4, Tensor 5, Tensor 6 |
                          |                         |    +------------------------------+
                          |                         |
                          v                   +-----------------------------------------+
+-------------------------+-----------+       |     |                                   |
| Bucket                              |       | +---+-----------+     +---------------+ |
|                                     |       | | BucketReplica |     | BucketReplica | |
|                                     |       | |               | ... |               | |
|   vector<BucketReplica> replicas +--------> | +---------------+     +---------------+ |
|                                     |       +-----------------------------------------+
|                                     |
|   vector<size_t> variable_indices +------->  <tensor index 4, tensor index 5, tensor 6>
|                                     |
+-------------------------------------+





bucket_indices    +-----------------------------------------------------------------------+
     +            |                                                                       |
     |            |  <tensor index 0, tensor index 1, tensor index 2, tensor index 3>     |
     |            |                                                                       |
     +----------> |                                                                       |
                  |  <tensor index 4, tensor index 5, tensor 6>                           |
                  |                                                                       |
                  |                                                                       |
                  |  ......                                                               |
                  |                                                                       |
                  |                                                                       |
                  |  <tensor index 16, tensor index 17, tensor index 18, tensor index 19> |
                  |                                                                       |
                  +-----------------------------------------------------------------------+

2.2 初始化桶
initialize_buckets方法用来初始化桶，具体逻辑是对于每一个桶，添加其模型副本，对于每一个模型副本，添加张量列表：

用分布式上下文设置 rpc_context_。

如果在DDP构造函数内调用initialize_bucket，则 rpc上下文指针（rpc context ptr）是否为null 无关紧要，因为grad不会发生变化。
如果在训练循环期间调用initialize_bucket，例如在rebuild_bucket 内部，因为grad可能会发生改变并指向bucket_view，那么它需要检查rpc context ptr是否为null。
如果rpc context ptr是null，则改变 variable.grad()，否则，在rpc上下文中改变梯度。
清空buckets_ 和 variable_locators_。

重置variable_locators_的尺寸，这样每个variable都有一个bucket index。

利用如下得到所有桶的个数和每个桶中副本个数：bucket_count = bucket_indices.size(); replica_count = replicas_.size();

从0开始递增到 bucket_count，逐一初始化 Bucket。

生成一个 Bucket bucket
如果bucket_indices[bucket_index].size() == 1，说明这个桶期待一个single sparse gradient，则设置 bucket.expect_sparse_gradient = true。
从0开始递增到replica_count，逐一初始化 BucketReplica。
生成一个 BucketReplica replica
如果这个桶期待一个single sparse gradient，则
利用bucket_indices[bucket_index].front()取出向量第一个元素，设置为 variable_index。
利用 variable_index 得到副本之中对应的variable。
设置副本replica的变量列表，代码为replica.variables = {variable}，这个副本只包括一个variable。
否则说明是dense gradient，则
遍历桶的variable，即利用 replicas_[replica_index][variable_index] 得到variable。
设置variable的设备和数据类型
给副本设置其variables，代码为：replica.variables.push_back(variable)。
设置replica 的一些关于variable的元信息，这些元信息是flat contents相关的，比如offsets存储了各个张量在flat bucket contents中的offset。
给relica.contents分配内存
利用 initialize_bucket_views(replica, replica.contents) 初始化 cotnents 和 views。
利用 bucket.replicas.push_back(std::move(replica)) 把这个 replica 加入到 bucket。
遍历桶中的variable，代码为 bucket_indices[bucket_index]。
设置 Reducer.variable_locators_，这样 Reducer 就知道如何在 bucket 之中确定一个varaible。bucket_index 是buckets_列表的位置，表示 buckets_ 之上的一个bucket。intra_bucket_index 是在 bucket replica 之中 vector 域的 variable index。
设置桶的变量，bucket.variable_indices = std::move(bucket_indices[bucket_index]);
利用 buckets_.push_back(std::move(bucket)) 把bucket这个桶加入到 Reducer之中。
*/

void Reducer::initialize_buckets(
    std::vector<std::vector<size_t>> bucket_indices) {
  // If initialize_buckets is called inside DDP constructor, then
  // it does not matter rpc context ptr is nullptr or not, as grad
  // will not be mutated.
  // If initialize_buckets is called during training loop, e.g, inside
  // rebuild_buckets(), since grad could be mutated and be pointed to
  // bucket_view, then it needs to check rpc context ptr is nullptr or not,
  // If rpc context ptr is nullptr, mutate variable.grad(); otherwise,
  // mutate grad in rpc context.
#ifndef _WIN32
  using torch::distributed::autograd::ThreadLocalDistAutogradContext;
  this->rpc_context_.set(ThreadLocalDistAutogradContext::getContextPtr());
#endif

  // This shouldn't be called if we're expecting autograd hooks to fire.
  REDUCER_CHECK(
      !expect_autograd_hooks_,
      logger_,
      "`initialize_buckets` must NOT be called during autograd execution.");

  // Clear current bucket assignment.
  buckets_.clear();
  variable_locators_.clear();

  // Ensure we have a bucket index for every variable.
  variable_locators_.resize(params_.size());

  // Iterate over buckets.
  const auto bucket_count = bucket_indices.size();
  buckets_.reserve(bucket_count);

  // 从0开始递增到bucket_count 从0开始递增到replica_count，遍历模型副本数目，为每一个模型副本都要做同样设置
  for (const auto bucket_index : c10::irange(bucket_count)) {  // 遍历桶
    Bucket bucket; // 生成一个桶

    // TODO(@pietern): Validate indices.
    // Must be non-empty, unique, and unique across buckets.
    REDUCER_CHECK(
        !bucket_indices[bucket_index].empty(),
        logger_,
        "Empty bucket specified.");

    // Variables that expect sparse gradients must have their own bucket.
    if (bucket_indices[bucket_index].size() == 1) {
      // 说明这个桶期待一个single sparse gradient
      const auto variable_index = bucket_indices[bucket_index].front();
      bucket.expect_sparse_gradient = expect_sparse_gradients_[variable_index];
    } else {
      for (const auto variable_index : bucket_indices[bucket_index]) {  // 遍历桶里面的张量，所有桶里每个张量index 都是唯一的
        REDUCER_CHECK(
            !expect_sparse_gradients_[variable_index],
            logger_,
            "Buckets with more than one variable cannot include variables ",
            "that expect a sparse gradient.");
      }
    }

    if (bucket.expect_sparse_gradient) {
      const auto variable_index = bucket_indices[bucket_index].front(); // 得到张量的index
      const auto& variable = params_[variable_index];  // 得到张量
      TORCH_INTERNAL_ASSERT(bucket_indices[bucket_index].size() == 1);
      bucket.variables = {variable};  // 这个副本只包括一个variable
    } else {
      at::TensorOptions options;
      // The start index of the variable in the flattened tensor.
      size_t offset = 0;

      // Reserve enough space for the per-variable fields stored in the bucket
      // for efficiency.
      const size_t num_variables = bucket_indices[bucket_index].size();
      bucket.variables.reserve(num_variables);
      bucket.offsets.reserve(num_variables);
      bucket.lengths.reserve(num_variables);
      bucket.sizes_vec.reserve(num_variables);

      // Iterate over bucket variables.
      for (const auto variable_index : bucket_indices[bucket_index]) { //遍历桶中的variable
        TORCH_INTERNAL_ASSERT(
            variable_index < params_.size(),
            "Out of range variable index specified.");
        const auto& variable = params_[variable_index];
        if (!options.has_device()) {
          options = options.device(variable.device());
        } else {
          REDUCER_CHECK(
              variable.device() == options.device(),
              logger_,
              "All parameters in a bucket must be ",
              "placed on the same device.");
        }
        if (!options.has_dtype()) {
          options = options.dtype(variable.dtype());
        } else {
          REDUCER_CHECK(
              variable.dtype() == options.dtype(),
              logger_,
              "All parameters in a bucket must have the same dtype.");
        }
        const auto length = variable.numel();
        bucket.variables.push_back(variable);
        bucket.offsets.push_back(offset);
        bucket.lengths.push_back(length);
        bucket.sizes_vec.push_back(variable.sizes());
        offset += length;
      }

      // Allocate the bucket's flattened `gradients` tensor.
      bucket.gradients = at::empty({static_cast<long>(offset)}, options);


/*
0x03 BucketReplica
如前面讨论的，一个 BucketReplica 代表了 [1..N] 个需要被规约的梯度，这些梯度拥有同样的 dtype，位于同样的设备上。是一个模型待求梯度参数的一部分，具体是哪些，由 bucket 的 variable_indices 决定。

其关键成员变量为：

std::vector<at::Tensor> variables 是构成此bucket副本的variable。我们在这里使用refcounted value，这样我们就可以在完成规约之后，轻松地将bucket内容 unflatten 到参与变量中。
at::Tensor contents ：把桶的内容展平的结果，即Flattened (1 dimensional) 之后的结果。
std::vector<at::Tensor> bucket_views_in ：提供了从输入角度在 contents 之中查看具体梯度的方法。
std::vector<at::Tensor> bucket_views_out ：提供了从输出角度在 contents 之中查看具体梯度的方法。
*/
      // Note:  "Gradient Layout Contract"
      //
      // Here, create views into the `gradients` tensor for each variable's
      // grad. Views serve as entry points to `copy_()` each grad's data in/out
      // of the flattened `gradients` tensor.
      //
      // Gradients may have dense memory but non-row-major-contiguous strides
      // (e.g. channels_last or channels_last_3d). For coalesced accesses
      // during copy_s, it's beneficial for each view's layout to match its
      // grad's layout.
      //
      // Specifically, we expect torch/csrc/autograd/functions/accumulate_grad.h
      // produces grads that obey the "Gradient Layout Contract":
      //   (1) if variable.is_non_overlapping_and_dense(), the stashed grad's
      //       strides match variable.
      //   (2) else, stashed grad is rowmajor contiguous.
      // and create views to match.
      //
      // If AccumulateGrad breaks the contract, and produces a grad with an
      // unexpected layout, performance will degrade due to poor memory access
      // patterns when copy_ing grad data in and out of its bucket view.
      // However, numerics remain correct, because the bucket view is the same
      // on either end of the raw allreduce.  bucket_view_in.copy(grad)
      // tranposes
      // (+ densifies) to the bucket view's layout, the data is allreduced,
      // then grad.copy_(bucket_view_out) transposes it back to grad's layout.
      //
      // The only way the numerics can go haywire is if the bucket views
      // themselves have different layouts across processes.
      // Bucket views' sizes and strides are set based on param layouts, using
      // the same logic that (we expect) AccumulateGrad uses for their grads.
      // Therefore, the only way a bucket view could have different layouts in
      // different processes is if its param has a different layout in
      // different processes. We can check that param layouts match across
      // processes in Reducer's constructor by allreducing some metadata.
      // Checking just once won't catch if someone messes with
      // param layouts over time, but not messing with params after DDP
      // construction is already a documented constraint.
      initialize_bucket_views(bucket);
    }

    // Map participating variables to this bucket.
    size_t intra_bucket_index = 0;
    for (const auto variable_index : bucket_indices[bucket_index]) {  // 遍历桶里面的张量，所有桶里每个张量index 都是唯一的
      TORCH_INTERNAL_ASSERT(
          variable_index < variable_locators_.size(),
          "Out of range variable index specified.");
      variable_locators_[variable_index] =
          VariableLocator(bucket_index, intra_bucket_index++);  // intra_bucket_index 就是递加
    }
    bucket.variable_indices = std::move(bucket_indices[bucket_index]);

    buckets_.push_back(std::move(bucket)); //插入桶列表
  }
}

/*
3.3 初始化
部分初始化的代码在 Reducer::initialize_buckets 之中。

// Allocate bucket contents tensor. 分配内存
replica.contents = at::empty({static_cast<long>(offset)}, options);

initialize_bucket_views(replica, replica.contents);
initialize_bucket_views 具体代码如下，这里需要对几个 PyTorch 函数进行说明。

as_strided ：依据现有tensor以及给定的步长来创建一个视图（类型仍然为tensor），与原数据共享内存，不存储诗句，所以两个view都不是真实的存储，只是视图。
narrow ：返回一个新的张量，其是原来张量的缩小版。
initialize_bucket_views 主要逻辑是：

遍历replica的张量，针对每一个张量，依据其是dense还是sparse进行不同处理，最后插入到replica.bucket_views_in之中。

把 replica.bucket_views_out 设置为 replica.bucket_views_in，正常应该是相等的。

如果gradient_as_bucket_view_设置为true，则需要处理两种情况：

当调用 rebuild_buckets 重建 bucket时，initialize_bucket_view 可以在initialize_bucket内调用，如果grad在上一次迭代中已经定义/计算过，则需要将旧的grad复制到新的bucket_view中，并让grad指向新的bucket_view。

initialize_bucket_view 也可以在构建时候在 initialize_bucket 内调用。在构建时间内不会定义 Grad，

在这种情况下，不要让梯度指向bucket_view，因为对于全局未使用的参数，梯度应保持为未定义。





具体如下图：

+------------------------------------------+
| BucketReplica                            |
|                                          |
|       vector<Tensor> bucket_views_in +--------------------+
|                                          |                |
|                                          |                |
|       vector<Tensor> bucket_views_out +--------------+    |
|                                          |           |    |
|                                          |           |    |
|                                          |           v    v
|                                          |     +-----+----+--------------------------+
|       Tensor contents  +---------------------> |Flattened (Tensor1, Tensor2, Tensor3)|
|                                          |     +-------------------------------------+
|                                          |
|                                          |
|       vector<Tensor> variables  +------------>  [Tensor1,Tensor2,Tensor3]
|                                          |
|                                          |
|                                          |
+------------------------------------------+

另外，mark_variable_ready_sparse, mark_variable_ready_dense， finalize_backward 都有对 contents 赋值。


*/

//initialize_bucket_views 这里是设置 Replica 的contents 和 views。
// (see Note:  "Gradient Layout Contract" in initialize_buckets).
void Reducer::initialize_bucket_views(Reducer::Bucket& bucket) {
  const auto& gradients = bucket.gradients;
  for (const auto i : c10::irange(bucket.variables.size())) { // 遍历replica的张量
    auto& v = bucket.variables[i];
    const auto offset = bucket.offsets[i];
    const auto length = bucket.lengths[i];
    if (v.is_non_overlapping_and_dense()) {  // Dense类型的张量
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      bucket.bucket_views_in.push_back(  // dense类型  // replica.bucket_views_in里面都是视图
          gradients.as_strided(v.sizes(), v.strides(), offset));
    } else { // Sparse类型的张量
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      bucket.bucket_views_in.push_back( // sparse类型  // replica.bucket_views_in里面都是视图
          gradients.narrow(0, offset, length).view(v.sizes()));
    }
    // By default `bucket_views_out` and `bucket_views_in` are
    // essentially the same thing.
    bucket.bucket_views_out = bucket.bucket_views_in;  // out也是视图

    // If gradient_as_bucket_view_ is set as true, then there are two cases to
    // handle: initialize_bucket_views could be called inside initialize_buckets
    // when rebuild_buckets, if grad has already been defined/calculated in
    // previous iteration, old grad needs to be copied into new bucket_view and
    // let grad point to the new bucket_view, initialize_bucket_views could also
    // be called inside initialize_buckets during construction. Grads are not
    // defined during construction time, in this case, do not let grad point to
    // bucket_view, because grads should be kept as being undefined for globally
    // unused parameters.
    if (gradient_as_bucket_view_) {
      auto& bucket_view = bucket.bucket_views_in.back();
      runGradCallbackForVariable(v, [&](auto& grad) {
        if (grad.defined() && !grad.is_alias_of(bucket_view)) {
          bucket_view.copy_(grad);
          grad = bucket_view;
          // 梯度被修改，需要写回去
          // The grad is modefied and needs to be written back.
          return true;
        }
        // 梯度没有被修改，不需要回写
        // The grad is not modified and does not need to be written back.
        return false;
      });
    }
  }
}

// (see Note:  "Gradient Layout Contract" in initialize_buckets).
void Reducer::populate_bucket_views_out(
    Reducer::Bucket& bucket,
    at::Tensor& tensor) {
  bucket.bucket_views_out.clear();
  for (const auto i : c10::irange(bucket.variables.size())) {
    const auto& v = bucket.variables[i];
    const auto offset = bucket.offsets[i];
    const auto length = bucket.lengths[i];
    if (v.is_non_overlapping_and_dense()) {
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      bucket.bucket_views_out.push_back(
          tensor.as_strided(v.sizes(), v.strides(), offset));
    } else {
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      bucket.bucket_views_out.push_back(
          tensor.narrow(0, offset, length).view(v.sizes()));
    }
  }
}

//这里把 num_iterations_ 增加，并且记录时间。
void Reducer::prepare_for_forward() {
  std::lock_guard<std::mutex> lock(mutex_);
  num_iterations_++; // 这里会递增
  if (should_collect_runtime_stats()) {
    record_forward_compute_start_time();
  }
}

/*
当新一次迭代时候，会重置这个值，prepare_for_backward 会调用到 reset_bucket_counting。

而且是使用 numGradHooksTriggeredMap_ 来重置。


具体逻辑我们展示一下：

对于 张量 2，就没有使用过，所以 delay_all_reduce 方法 之中直接放入到未使用参数。
对于 张量 1：
numGradHooksTriggeredMap_ 初始化是 0。
第一次迭代之后变成 1。
后向传播时候，调用 prepare_for_backward 和 reset_bucket_counting，把 numGradHooksTriggeredMap_赋值给 numGradHooksTriggeredMapPerIteration_。
autograd_hook 之中会递减，然后如果是 0，就设置此变量为 ready，可以规约了。
   Variable 2

                                     delay_all_reduce

   numGradHooksTriggeredMap_[2] = 0  +---------------> unused_parameters_.push_back(0)


+----------------------------------------------------------------------------------------+

   Variable 1



    numGradHooksTriggeredMap_[1] = 0

                   +
                   |
                   |  first_iteration
                   |
                   v

    numGradHooksTriggeredMap_[1] = 1

                   +
                   |  prepare_for_backward
                   |
                   |  reset_bucket_counting
                   v

 numGradHooksTriggeredMapPerIteration_ = numGradHooksTriggeredMap_
                   +
                   |
                   |
                   | backward
                   |
                   | autograd_hook
                   v
                                                               YES
 if (++numGradHooksTriggeredMapPerIteration_[index]=== 0)?? +------->  mark_variable_ready(1)
                   +
                   |  NO
                   |
                   v



这里会遍历桶，对于每个桶，重置其副本的pending状态，某一个模型副本pending状态是由这个模型副本中对应桶的变量数目决定。

如果是静态图，则重置numGradHooksTriggeredMapPerIteration_。

*/
void Reducer::reset_bucket_counting() {
  next_bucket_ = 0;
  // Reset num_buckets_ready_ at the beginning of backward computation
  // in each iteration.
  num_buckets_ready_ = 0;

  for (auto& bucket : buckets_) {   // 遍历桶
    bucket.pending = bucket.variables.size(); //对于每个桶，重置其副本的pending状态，某一个模型副本pending，是由这个模型副本中，本桶的变量数目决定
  }

  if (static_graph_) {
    numGradHooksTriggeredMapPerIteration_ = numGradHooksTriggeredMap_;
  }
}

/*
3.3.2 查找未使用的参数
search_unused_parameters 完成了 "查找未使用的参数" 功能。

我们首先要看看 Reducer 的 find_unused_parameters_ 成员变量。如果 find_unused_parameters_ 被设置为 true，则 DDP 会在前向传播结束时候，从指定的输出进行回溯，遍历autograd计算图来找到所有没有使用过的参数，并且一一标记为就绪 ready。

对于所有参数，DDP 都有一个指向它们的梯度累积函数的指针，但对于那些autograd图中不存在的参数，它们将在第一次调用autograd钩子时就被标记为准备就绪。

因为模型输出可能会被忽略，所以这个操作不是立即完成的，我们只是像在torch.autograd.backward()这里开始执行规约操作。

大家可以发现，这么做开销会很大，为什么要这么做？这是因为计算动态图会改变。

训练时候，某次迭代可能只用到模型的一个子图，而且因为PyTorch 是动态计算，所以子图会在迭代期间改变，就是说，某些参数可能在下一次迭代训练时候被跳过。
同时，因为所有参数在一开始就已经被分好桶，而 hook 又规定了只有整个桶 ready （即，pending == 0）之后才会进行通信，所以如果我们不将未使用参数标记为 ready，整个通信过程就会没法进行。
*/
// Traverse the autograd graph starting at the specified output.
// All parameters for which we have a pointer to their gradient accumulation
// functions, but don't show up in the autograd graph will be marked ready for
// for reduction as soon as the first autograd hook is called. This is not
// done immediately because the model output may be ignored, and we only
// want to start performing reductions on `torch.autograd.backward()`.
/*
gradAccToVariableMap_ 的使用如下，search_unused_parameters 就是遍历查找 gradAccToVariableMap_，如果某一个accumulator 函数没有在 gradAccToVariableMap_ 里面，就说明不用计算梯度。
*/
void Reducer::search_unused_parameters(
    const std::vector<torch::autograd::Variable>& outputs) {
  std::unordered_set<torch::autograd::Node*> seen;
  std::vector<torch::autograd::Node*> queue;

  RECORD_FUNCTION(
      "torch.distributed.ddp.reducer::search_unused_parameters",
      std::vector<c10::IValue>());

  // Seed queue with the grad functions of all outputs.
  for (const auto& output : outputs) {
    const auto& grad_fn = output.grad_fn();
    if (grad_fn) {
      queue.push_back(grad_fn.get()); // 把所有输出节点的梯度函数插入到queue
    }
  }

  // Traverse the autograd graph starting at the specified output.
  // 遍历这个queue中的元素，对于每一个函数，找到其后向图之中的后续边，然后把后续边指向的节点再插入queue，然后继续循环，最终 seen 里面是所有从output出发，所有节点的梯度函数
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        const bool was_inserted = seen.insert(next_ptr).second;
        if (was_inserted) {
          queue.push_back(next_ptr);
        }
      }
    }
  }

  // Find accumulator functions that don't show up in this graph.
  // 遍历查找，如果某一个accumulator 函数没有在这图里面，就说明不用计算梯度
  // gradAccToVariableMap_ 里面是所有需要被规约的variable
  // 遍历gradAccToVariableMap_，如果 seen 之中没有，就说明这个参数没有被使用，插入到unused_parameters_
  /*
  至此，前向传播已经结束，我们得到了如下：

需要计算梯度的参数已经分桶。
桶已经重建完毕。
前向传播已经完成。
从指定的输出进行回溯，遍历autograd计算图来找到所有没有使用过的参数，并且一一标记为就绪 ready。
我们在下一篇就分析后向传播。
  */
  for (const auto& it : gradAccToVariableMap_) {
    // If the accumulator function is present in the graph, we know
    // a gradient will be computed for the corresponding parameter.
    if (seen.count(it.first) == 0) {
      if (ddp_debug_level_ == c10d::DebugLevel::Detail) {
        const auto param_info = param_names_.find(it.second);
        TORCH_INTERNAL_ASSERT(
            param_info != param_names_.end(),
            "Did not find variable index ",
            it.second,
            " in DDP parameter name mapping!");
        const auto param_name = param_info->second;
        LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
                  << "Parameter " << param_name << " at index " << it.second
                  << " is marked as unused.";
      }
      unused_parameters_.push_back(it.second);
    }
  }

  // Warn user about unnecessary perf hit if all parameters were used in
  // forward.
  if (unused_parameters_.empty()) {
    TORCH_WARN_ONCE(
        "find_unused_parameters=True was specified in DDP constructor, "
        "but did not find any unused parameters in the forward pass. This flag "
        "results in an extra traversal of the autograd graph every iteration, "
        " which can adversely affect performance. If your model indeed never "
        "has any unused parameters in the forward pass, consider turning this "
        "flag off. Note that this warning may be a false positive if your model "
        "has flow control causing later iterations to have unused parameters.");
  }
  if (!static_graph_ && ddp_graph_static_) {
    if (num_iterations_ > 1) {
      // Graph is still static if the set of unused parameters did not change.
      ddp_graph_static_ =
          prev_iteration_unused_parameters_ == unused_parameters_;

      if (!ddp_graph_static_) {
        // Log graph is not static. Logger takes care of ensuring this is done
        // only once to avoid overhead.
        logger_.lock()->log_if_graph_static(false);
      }
    }
    prev_iteration_unused_parameters_ = unused_parameters_;
  }
}

/*
3.2.3 初始化桶
同步之后就是初始化桶，本部分代码在前文已经分析过，故此省略。

3.3 准备后向传播
前向传播完成之后，调用 prepare_for_backward 完成了后向传播的准备。

具体大致分为两步：重置，查找未使用的参数。
*/
void Reducer::prepare_for_backward(
    const std::vector<torch::autograd::Variable>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);

  // 记录开始时间
  backward_compute_start_time_ = current_time_in_nanos();
  if (should_collect_runtime_stats()) {
    record_backward_compute_start_time();
  }

  // Reset accounting.
  expect_autograd_hooks_ = true;
  // Clear gradient ready order as it can be different in the next iteration.
  grad_ready_order_indices_.clear();

  reset_bucket_counting();

  // Reset unused parameter accounting.
  has_marked_unused_parameters_ = false;
  // Reset per iteration marked ready parameters.
  //在反向传播之前，会重置这个变量。
  perIterationReadyParams_.clear();

  // If static graph is not set, search graph to detect unused parameters.
  // When static graph is set, unused_parameters_ will be detected and will
  // not change after 1st iteration.
  // If static_graph_ = false and find_unused_parameters_ is false,
  // we assume that autograd hooks for ALL variables will be called,
  // and we don't have to search the autograd graph for presence of these hooks.
  if (dynamic_graph_find_unused()) {
    unused_parameters_.clear();
    search_unused_parameters(outputs);  // 查找没有使用的参数
  }
}

void Reducer::copy_bucket_to_grad(
    at::Tensor& variable,
    Reducer::Bucket& bucket,
    size_t intra_bucket_index,
    bool global_unused) {
  const auto& bucket_view = bucket.bucket_views_out[intra_bucket_index];
  runGradCallbackForVariable(variable, [&](auto& grad) {
    // If a parameter is globally unused, we keep its grad untouched.
    if (!global_unused) {
      if (!grad.defined()) {
        // Creates grad according to the "Gradient Layout Contract"
        // (see torch/csrc/autograd/functions/accumulate_grad.h)
        grad =
            torch::autograd::utils::clone_obey_contract(bucket_view, variable);
      } else {
        grad.copy_(bucket_view);
      }
      // The grad is modified and needs to be written back.
      return true;
    }
    // The grad is not modified.
    return false;
  });
}

/*
就是遍历perIterationReadyParams_，如果没找到，就返回。

在 rebuild_buckets 方法中会调用 ensure_prior_reduction_finished，里面会调用这两个方法来校验。
*/
std::vector<std::string> Reducer::getUnmarkedParamsForIteration() {
  std::vector<std::string> unMarkedParamNames;
  for (const auto& it : param_names_) {
    if (perIterationReadyParams_.find(it.first) ==
        perIterationReadyParams_.end()) {
      unMarkedParamNames.push_back(it.second);
    }
  }
  return unMarkedParamNames;
}

std::vector<size_t> Reducer::getUnmarkedParamIndicesForIteration() {
  std::vector<size_t> unmarked_param_indices;
  const auto variable_count = params_.size();
  for (const auto variable_index : c10::irange(variable_count)) {
    if (perIterationReadyParams_.find(variable_index) ==
        perIterationReadyParams_.end()) {
      unmarked_param_indices.push_back(variable_index);
    }
  }
  return unmarked_param_indices;
}

// A bucket with one or more dense tensors needs to be unflattened.
void Reducer::finalize_bucket_dense(Bucket& bucket) {
  for (const auto intra_bucket_index : c10::irange(bucket.variables.size())) {
    auto& variable = bucket.variables[intra_bucket_index];

    bool global_unused = false;
    // See Note [Skip allreducing local_used_map_dev]
    if (static_graph_ || find_unused_parameters_) {
      // Determine if this param has been used globally or not.
      //
      // If the variable was used locally, it is also used globally and then
      // we don't need to wait for the reduction. Otherwise we lazily wait for
      // the reduction to complete, only when we see a variable that was
      // unused locally. Then we end up delaying the synchronization point
      // that local_used_work_->wait() implies. If we don't have any unused
      // parameters at all, we can skip waiting for the work to complete
      // altogether, and cause negligible performance overhead for models
      // where all parameters are used. Such lazily waiting means minimizing
      // performance impact for the big majority of models where all
      // parameters are always used. Then we only pay the overhead cost if
      // there is indeed a parameter that is locally unused, because we need
      // to check if it's also globally unused.
      size_t variable_index = bucket.variable_indices[intra_bucket_index];
      // Note: global_unused might not be global yet. As we lazily wait for
      // the reduction to complete, it becomes really global only if we get to
      // the point as below where we wait for the reduction work, make D2H
      // copy, and update global_unused with the real global consensus, i.e.
      // local_used_map_reduced_ is true.
      global_unused = local_used_map_[variable_index].item<int>() == 0;
      if (global_unused && !local_used_map_reduced_) {
        // Wait for local_used_map reduction to complete.
        local_used_work_->wait();
        // D2H from local_used_map_dev_ to local_used_map_
        // Blocking copy, if local_used_map_dev_ is cuda
        local_used_map_.copy_(local_used_map_dev_);

        global_unused = local_used_map_[variable_index].item<int>() == 0;
        local_used_map_reduced_ = true;
      }
    }

    if (!gradient_as_bucket_view_) {
      if (set_grads_to_none_) {
        runGradCallbackForVariable(variable, [&](auto& grad) {
          grad.reset();
          return true;
        });
      } else {
        RECORD_FUNCTION(
            "torch.distributed.ddp.reducer::copy_bucket_to_grad",
            std::vector<c10::IValue>({variable}));
        copy_bucket_to_grad(
            variable, bucket, intra_bucket_index, global_unused);
      }
    } else {
      const auto& bucket_view_out = bucket.bucket_views_out[intra_bucket_index];
      auto& bucket_view_in = bucket.bucket_views_in[intra_bucket_index];
      // If a communication hook is registered, then `bucket_view_out` stores
      // the allreduced results in a newly allocated tensor, so we copy
      // `bucket_view_out` back to `bucket_view_in` for this gradient.
      if (!bucket_view_in.is_alias_of(bucket_view_out)) {
        bucket_view_in.copy_(bucket_view_out);
      }
      runGradCallbackForVariable(variable, [&](auto& grad) {
        if (set_grads_to_none_) {
          grad.reset();
          return true;
        }
        // If a parameter is globally unused, we keep its grad untouched.
        if (!global_unused) {
          // If grad is globally used but locally unused, let grad point to
          // bucket_view_in
          if (!grad.defined()) {
            grad = bucket_view_in;
          } else {
            if (!grad.is_alias_of(bucket_view_in)) {
              REDUCER_CHECK(
                  false,
                  logger_,
                  "Detected at least one parameter gradient is not the "
                  "expected DDP bucket view with gradient_as_bucket_view=True. "
                  "This may happen (for example) if multiple allreduce hooks "
                  "were registered onto the same parameter. If you hit this error, "
                  "please file an issue with a minimal repro.");
            }
          }
          // The grad is modified and needs to be written back.
          return true;
        }
        // The grad is not modified.
        return false;
      });
    }
  }
}

void Reducer::finalize_backward() {
  // No longer expect autograd hooks to fire after this function returns.
  TORCH_INTERNAL_ASSERT(expect_autograd_hooks_);
  expect_autograd_hooks_ = false;

  // No longer require call to finalize after this function returns.
  TORCH_INTERNAL_ASSERT(require_finalize_);
  require_finalize_ = false;

  // Wait for asynchronous reduction to complete, and unflatten the bucket's
  // flattened `gradients` tensor.
  for (auto& bucket : buckets_) {
    // See Note [DDP Communication Hook]
    TORCH_INTERNAL_ASSERT(
        bucket.future_work,
        "Expected bucket.future_work not to be null. "
        "This may indicate that communication hook was not properly installed.");
    bucket.future_work->wait();
    auto future_result = comm_hook_ == nullptr
        ? detail::parseCppCommHookResult(bucket.future_work->value())
        : comm_hook_->parseHookResult(bucket.future_work->value());
    if (bucket.expect_sparse_gradient) {
      bucket.gradients.copy_(future_result);
    } else {
      // Reinitialize only `bucket_views_out` with the future_result by
      // following the same logic in `initialize_buckets`.
      populate_bucket_views_out(bucket, future_result);
    }

    // Unset allreduce division factor, as it may change in next backwards pass
    // when running with DDP join mode.
    div_factor_ = kUnsetDivFactor;

    if (!bucket.expect_sparse_gradient) {
      // We don't need to finalize the sparse bucket since the sparse grad and
      // the bucket essentially point to the same storage. As a result, once
      // the allreduce is done, the sparse grads are automatically updated.
      finalize_bucket_dense(bucket);
    }
  }

  if (installed_futures_ != c10::nullopt) {
    c10::collectAll(*installed_futures_)->wait();
    installed_futures_ = c10::nullopt;
  }

  // See Note [Skip allreducing local_used_maps_dev]
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // Due to the lazy wait, it is possible that reduction of the current
    // iteration is still going when the one for next iteration gets kicked off.
    // For such case, we want to wait explicitly to make sure the reduction does
    // complete before kicking off next one. Otherwise the previous one may
    // interfere, write to the device-side memory and clobber the content of
    // local_unused_maps_dev_.
    if (!local_used_map_reduced_) {
      local_used_work_->wait();
    }
  }

  if (dynamic_graph_find_unused()) {
    // Reset unused parameter accounting.
    // See Note [local_used_map_ -> local_used_map_dev copying]
    local_used_map_.fill_(0);
    local_used_map_reduced_ = false;
  }

  if (should_collect_runtime_stats()) {
    record_backward_comm_end_time();
  }
}

/*
5.7.4 runGradCallbackForVariable
mark_variable_ready_dense 函数会调用到 runGradCallbackForVariable。

5.7.4.1 Reducer
Reducer的runGradCallbackForVariable如下，其调用 distributed::autograd::ContextPtr.runGradCallbackForVariable 来处理。
*/
void Reducer::runGradCallbackForVariable(
    at::Tensor& variable,
    GradCallback&& cb) {
#ifdef _WIN32
  cb(variable.mutable_grad());
#else
  // 加载rpc context
  auto context_ptr = rpc_context_.context_ptr.load();
  if (context_ptr == nullptr) {
    cb(variable.mutable_grad());
  } else {
    // Under distributed autograd
    // 下面分析
    context_ptr->runGradCallbackForVariable(variable, std::move(cb));
  }
#endif
}

#ifndef _WIN32
void Reducer::RpcContext::set(ContextPtr&& new_context_ptr) {
  // We should set 'new_context_ptr' even if it's nullptr. That means the
  // reducer is under a local backward run.
  const auto new_context_raw_ptr = new_context_ptr.get();
  if (context_ptr.exchange(new_context_raw_ptr) != new_context_raw_ptr) {
    // Set the shared ptr to the context only if it's set first time.
    // All call sites should use the same context ptr.
    // Use an atomic to avoid data race from multiple threads.
    context_ptr_holder = std::move(new_context_ptr);
  }
}
#endif
/*
3.2.2 同步桶indices
产生尺寸之后，就使用 sync_bucket_indices 同步桶的indices，其逻辑如下：

遍历桶，把桶的大小都记录到bucket_sizes。
配置TensorOptions。
把桶对应的indices和桶数目放入indices_tensor，这里是通过 PyTorch accessor来对张量进行读写，accessor就像是一个张量，但它将张量的维度和 dtype 硬编码为了模板参数，可以高效的访问元素。
因为 NCCL这样的 ProcessGroup 只支持device之间的操作，所以把indices_tensor拷贝到indices_tensor_device。
对 indices_tensor_device 进行广播。
类似，对桶尺寸进行广播。
广播结束之后，遍历桶，使用从rank 0收到的num_buckets, bucket_sizes_tensor 和 indices_tensor 更新传进来的参数bucket_indices。
*/
void Reducer::sync_bucket_indices(
    std::vector<std::vector<size_t>>& bucket_indices) {
  auto num_buckets = bucket_indices.size();
  std::vector<size_t> bucket_sizes;
  bucket_sizes.reserve(num_buckets);
  int64_t total_size = 0;

  // 遍历桶，把桶的大小都记录到bucket_sizes
  for (const auto i : c10::irange(num_buckets)) {
    auto bucket_size = bucket_indices.at(i).size();
    bucket_sizes.push_back(bucket_size);
    total_size += bucket_size;
  }

  // 配置TensorOptions
  at::TensorOptions options;
  options = options.dtype(at::kInt);
  options = options.device(params_[0].device());

  // Group indices and num_bucket together into indices_tensor
  // Broadcast this tensor first, as its size is equal among all processes
  // 把桶对应的indices和桶数目放入indices_tensor，这里是通过 PyTorch accessor来对张量进行读写，accessor就像是一个张量，但它将张量的维度和 dtype 硬编码为了模板参数，可以高效的访问元素
  auto indices_tensor = at::empty({total_size + 1}, at::kInt);
  auto indices_accessor = indices_tensor.accessor<int, 1>();
  auto indices_accessor_Index = 0;
  for (const auto i : c10::irange(num_buckets)) {
    const auto& bucket_size = bucket_indices.at(i).size();
    for (const auto j : c10::irange(bucket_size)) {
      indices_accessor[indices_accessor_Index++] = bucket_indices[i][j];
    }
  }
  indices_accessor[indices_accessor_Index] = num_buckets;

  // Copy CPU tensor to device tensor, as the process_group_ could be NCCL and
  // it can only broadcast device tensors.
  auto indices_tensor_device = at::empty({total_size + 1}, options);

  //// 因为 NCCL这样的 ProcessGroup 只支持device之间的操作，所以把indices_tensor拷贝到indices_tensor_device
  indices_tensor_device.copy_(indices_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> indices_tensor_list = {indices_tensor_device};

  //// 对 indices_tensor_device 进行广播
  process_group_->broadcast(indices_tensor_list)->wait();
  indices_tensor.copy_(indices_tensor_list.front(), /*non_blocking=*/false);

  // Update num_buckets after receiving it from rank 0
  num_buckets = indices_accessor[indices_accessor_Index];

  // Broadcast bucket_sizes // 类似，对桶尺寸进行广播
  auto bucket_sizes_tensor = at::empty({(int64_t)num_buckets}, at::kInt);
  auto bucket_sizes_accessor = bucket_sizes_tensor.accessor<int, 1>();
  for (const auto i : c10::irange(num_buckets)) {
    // For rank != 0, it is possible that local num buckets bucket_sizes.size()
    // is smaller than broadcasted num_buckets
    bucket_sizes_accessor[i] =
        bucket_sizes.at(std::min(i, (bucket_sizes.size() - 1)));
  }
  auto bucket_sizes_tensor_device = at::empty({(int64_t)num_buckets}, options);
  bucket_sizes_tensor_device.copy_(bucket_sizes_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> bucket_sizes_tensor_list = {
      bucket_sizes_tensor_device};
  process_group_->broadcast(bucket_sizes_tensor_list)->wait();
  bucket_sizes_tensor.copy_(
      bucket_sizes_tensor_list.front(), /*non_blocking=*/false);

  // Clear bucket_indices first, and then update bucket_indices using received
  // num_buckets, bucket_sizes_tensor and indices_tensor from rank 0
  bucket_indices.clear();
  bucket_indices.reserve(num_buckets);
  indices_accessor_Index = 0;
  // 遍历桶，使用从rank 0收到的num_buckets, bucket_sizes_tensor 和 indices_tensor 更新传进来的参数bucket_indices
  for (const auto i : c10::irange(num_buckets)) {
    const auto& bucket_size = bucket_sizes_accessor[i];
    std::vector<size_t> bucket;
    bucket.reserve(bucket_size);
    for (const auto j : c10::irange(bucket_size)) {
      (void)j;
      bucket.push_back(indices_accessor[indices_accessor_Index++]);
    }
    bucket_indices.emplace_back(std::move(bucket));
  }
}

/*
4.3 重建
我们接下来看看重建机制。

DDP 根据张量在后向传播中接收梯度的时间，使用 rebuilt_params_ 和 rebuilt_param_indices_ 来重建存储桶。

rebuild_buckets 函数进行广播通信调用，并且可以与下一个forward()调用重叠，因此它可以是异步的。

在find_unused_parameters=true情况下重建bucket 就是异步操作，因为我们可以多次重建bucket，其中子图经过训练，参数索引顺序可能会更频繁地更改。
对于find_unused_parameters=false的情况，bucket只重建一次，性能成本可以忽略不计。如果已重建存储桶， rebuild_buckets 则返回true。

接下来进行重建桶，具体分为：

配置各种尺寸限制。
计算桶的尺寸。
同步桶indices。
初始化桶。
*/
bool Reducer::rebuild_buckets() {
  // Ensure reduction for previous backwards pass is finished. If user's model
  // has unused parameters for example, this will raise an error recommending to
  // run with find_unused_parameters=True, instead of the size mismatch
  // exception below.
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_prior_reduction_finished();
  if (!should_rebuild_buckets() || rebuilt_params_.empty()) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      rebuilt_params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter tensors size is not same as rebuilt parameter indices size: ",
          rebuilt_params_.size(),
          " versus ",
          rebuilt_param_indices_.size()));
  TORCH_INTERNAL_ASSERT(
      params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter indices size is not same as original model parameters size.",
          "Original model param size is: ",
          params_.size(),
          " versus rebuilt params size of: ",
          rebuilt_param_indices_.size()));
  std::vector<std::vector<size_t>> rebuilt_bucket_indices;

  // 配置各种尺寸限制
  std::vector<size_t> bucket_size_limits;
  bucket_size_limits.push_back(first_bucket_bytes_cap_);
  bucket_size_limits.push_back(bucket_bytes_cap_);

  std::vector<size_t> per_bucket_size_limits;
  auto ddp_set_last_bucket_as_small =
      (parse_env("DDP_SET_LAST_BUCKET_CAP") == "1");

  if (ddp_set_last_bucket_as_small) {
    // Reverse so that first_bucket_bytes_cap_ (smaller bucket) becomes the last
    // bucket. We cannot simply pass in {bucket_bytes_cap_,
    // first_bucket_bytes_cap} as the bucket order as we would immediately
    // advance to the 2nd element after the first bucket, whereas we only want
    // the last bucket to have a smaller size.
    std::reverse(rebuilt_params_.begin(), rebuilt_params_.end());
    std::reverse(rebuilt_param_indices_.begin(), rebuilt_param_indices_.end());
  }
  // 计算桶的尺寸
  std::tie(rebuilt_bucket_indices, per_bucket_size_limits) =
      compute_bucket_assignment_by_size(
          rebuilt_params_,
          bucket_size_limits,
          expect_sparse_gradients_,
          rebuilt_param_indices_,
          logger_);

  if (ddp_set_last_bucket_as_small) {
    // Reverse again because buckets were rebuilt in the opposite of gradient
    // ready order.
    std::reverse(rebuilt_bucket_indices.begin(), rebuilt_bucket_indices.end());
    std::reverse(per_bucket_size_limits.begin(), per_bucket_size_limits.end());
  }

  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    TORCH_INTERNAL_ASSERT(
        rebuilt_bucket_indices.size() == per_bucket_size_limits.size())
    LOG(INFO) << rebuilt_bucket_indices.size()
              << " buckets rebuilt with size limits: "
              << c10::Join(", ", per_bucket_size_limits) << " bytes.";
  }

  // For rebuilt bucket indices, it needs to be synced across all ranks.
  // Broadcast the newly rebuilt bucket indices from rank 0 in default.
  // After syncing up rebuilt bucket indices, initialize buckets for reducer.
  // 同步桶indices
  sync_bucket_indices(rebuilt_bucket_indices);

  has_rebuilt_bucket_ = true; // 只重建一次
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();

  // 初始化桶
  initialize_buckets(std::move(rebuilt_bucket_indices));

  return true;
}

// See Note [DDP Communication Hook]
void Reducer::register_comm_hook(std::unique_ptr<CommHookInterface> iface) {
  REDUCER_CHECK(
      comm_hook_ == nullptr,
      logger_,
      "register_comm_hook or register_builtin_comm_hook can only be called once.");

  comm_hook_ = std::move(iface);
}

// See Note [DDP Communication Hook]
void Reducer::register_builtin_comm_hook(
    c10d::BuiltinCommHookType comm_hook_type) {
  REDUCER_CHECK(
      comm_hook_ == nullptr,
      logger_,
      "register_builtin_comm_hook or register_comm_hook can only be called once.");

  switch (comm_hook_type) {
    case c10d::BuiltinCommHookType::ALLREDUCE:
      comm_hook_ = std::make_unique<c10d::AllReduceCommHook>(process_group_);
      LOG(INFO) << "Built-in communication hook ALLREDUCE is registered.";
      break;
    case c10d::BuiltinCommHookType::FP16_COMPRESS:
      comm_hook_ = std::make_unique<c10d::FP16CompressCommHook>(process_group_);
      LOG(INFO) << "Built-in communication hook FP16_COMPRESS is registered.";
      break;
    default:
      TORCH_WARN_ONCE(
          "Unknown built-in DDP comm hook type is provided. No comm hook will be used.");
  }
}

void Reducer::set_grads_to_none(bool set_to_none) {
  set_grads_to_none_ = set_to_none;
}

void Reducer::ensure_prior_reduction_finished() {
  // Check that any prior reduction has finished.
  // The variable `require_finalize_` is true until all gradients
  // have been computed and reduction of all buckets has been kicked off.
  if (require_finalize_) {
    // Collect unmarked parameter indices, additionally, in debug mode retrieve
    // parameter names.
    auto unmarked_param_indices = getUnmarkedParamIndicesForIteration();
    // We should have some unmarked parameter indices, otherwise we would not
    // have run into this error branch.
    TORCH_INTERNAL_ASSERT(!unmarked_param_indices.empty());

    std::string kBaseErrorMsg =
        "Expected to have finished reduction in the prior iteration before "
        "starting a new one. "
        ""
        "This error indicates that your module has parameters that were "
        "not used in producing loss. ";
    std::string kOutputsNotUsedInLossErrorMsg =
        "making sure all "
        "`forward` function outputs participate in calculating loss. ";
    std::string kDDPBugErrorMsg =
        "\nIf you already have done the above, then the distributed "
        "data parallel module wasn't able to locate the output tensors in the "
        "return value of your module's `forward` function. "
        "Please include the loss function and the structure of the return "
        "value of `forward` of your module when reporting this issue (e.g. "
        "list, dict, iterable).";

    if (static_graph_) {
      kBaseErrorMsg =
          "Expected to have finished reduction in the prior iteration before "
          "starting a new one. "
          "This error indicates that your training graph has changed "
          "in this iteration, e.g., one parameter is used in first "
          "iteration, but then got unused in the second iteration. "
          "this is not compatible with static_graph set to True.";
    } else if (!find_unused_parameters_) {
      // Parameters may have been unused in forward pass, or not all outputs
      // were used in producing loss.
      kBaseErrorMsg +=
          "You can enable unused parameter detection by passing the "
          "keyword argument `find_unused_parameters=True` to "
          "`torch.nn.parallel.DistributedDataParallel`, and by \n";
      kBaseErrorMsg += kOutputsNotUsedInLossErrorMsg;
      kBaseErrorMsg += kDDPBugErrorMsg;
    } else {
      // Note that it does not really matter whether unused_parameters_.empty(),
      // since user may have enabled detection but this particular iteration
      // could have used or not used all parameters.
      kBaseErrorMsg +=
          "Since `find_unused_parameters=True` is enabled, this likely "
          " means that not all `forward` outputs participate in computing loss. You can fix this by ";
      kBaseErrorMsg += kOutputsNotUsedInLossErrorMsg;
      kBaseErrorMsg += kDDPBugErrorMsg;
    }

    const std::string unmarked_param_indices_info = c10::str(
        "\n",
        "Parameter indices which did not receive grad for rank ",
        process_group_->getRank(),
        ": ",
        unmarked_param_indices);

    if (ddp_debug_level_ == DebugLevel::Off) {
      // Without debug mode, log unmarked_param_indices, as well as
      // recommendation to use debug mode to print parameter names.
      kBaseErrorMsg += unmarked_param_indices_info;
      kBaseErrorMsg +=
          "\n In addition, you can set the environment variable "
          "TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information "
          "about which particular parameters did not receive gradient on this rank "
          "as part of this error";
    } else {
      // Retrieve set of parameter names that did not receive gradient.
      auto unmarkedParams = getUnmarkedParamsForIteration();
      TORCH_INTERNAL_ASSERT(!unmarkedParams.empty());
      for (const auto& s : unmarkedParams) {
        LOG(INFO) << "[Rank " << process_group_->getRank() << "] "
                  << "Parameter: " << s
                  << " did not get gradient in backwards pass.";
      }
      const std::string unmarkedParamInfo = c10::Join(", ", unmarkedParams);
      // In debug mode, log param names and indices that went unused.
      kBaseErrorMsg += c10::str(
          "\n",
          "Parameters which did not receive grad for rank ",
          process_group_->getRank(),
          ": ",
          unmarkedParamInfo);
      kBaseErrorMsg += unmarked_param_indices_info;
    }
    REDUCER_CHECK(false, logger_, kBaseErrorMsg);
  }
}

void Reducer::set_ddp_runtime_logging_sample_rate(int sample_rate) {
  ddp_runtime_logging_sample_rate_ = sample_rate;
}

int Reducer::get_ddp_runtime_logging_sample_rate() {
  return ddp_runtime_logging_sample_rate_;
}

bool Reducer::should_collect_runtime_stats() {
  if (num_iterations_ > 0 &&
      (num_iterations_ <= 10 ||
       num_iterations_ % get_ddp_runtime_logging_sample_rate() == 0)) {
    return true;
  }
  return false;
}

void Reducer::record_forward_compute_start_time() {
  if (timer_) {
    timer_->record(Timer::Event::kForwardStart);
  }
}

void Reducer::record_backward_compute_start_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardComputeStart);
  }
}

void Reducer::record_backward_compute_end_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardComputeEnd);
  }
}

void Reducer::record_backward_comm_start_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardCommStart);
  }
}

void Reducer::record_backward_comm_end_time() {
  if (timer_) {
    timer_->record(Timer::Event::kBackwardCommEnd);
  }
}

//Reducer 只有在第一次迭代之后才能生成静态图，因为毕竟PyTorch还是动态的，无论如何也得走一步动态生成
void Reducer::set_static_graph() {
  std::lock_guard<std::mutex> lock(mutex_);
  REDUCER_CHECK(
      num_iterations_ == 0,
      logger_,
      "set_static_graph() should be called before training loop starts "
      "and after DistributedDataParallel is constructed.");
  static_graph_ = true;
  // when static_graph_ is set as true, always initialize_local_used_map
  // and detect the global unused parameters in the first iteration.
  initialize_local_used_map();
}

namespace {
/*
2.5.2.1 论文内容
我们接下来就要结合论文内容来分析。

梯度bucketing的思想是基于这样一个观察，即集合通信在大张量上更有效。

实验表明，如果DDP在短时间内等待并将多个梯度存储到一个AllReduce操作中，它可以实现更高的吞吐量和更低的延迟，而不是在每个梯度存储可用时立即启动专用的AllReduce。这对于具有许多小参数的模型尤其有用。但是，DDP不应在一个AllReduce中传输所有数据，否则，在计算结束之前无法启动任何通信。

参数到桶映射（Parameter-to-Bucket Mapping）对DDP速度有相当大的影响。在每次向后传播中，将所有参数梯度中的张量复制到桶中，并在AllReduce之后将平均梯度复制回桶中。为了加速复制操作，存储桶始终与参数在同一设备上创建。如果模型跨越多个设备，DDP会考虑设备关联性，以确保同一存储桶中的所有参数都位于同一设备上。AllReduce的顺序也会对结果产生影响，因为它决定了多少通信可以与计算重叠。DDP按model.parameters()的相反顺序启动AllReduce。

所以，为了提高通信效率，DDP 将Reducer参数梯度组织成为桶，一次规约一个桶。从参数梯度到桶的映射是在构建时根据桶大小限制和参数大小确定的，。用户可以通过设置bucket_cap_mb来配置桶的大小。

模型参数以（大致）Model.parameters()与给定模型相反的顺序分配到桶中 。使用相反顺序的原因是：

反向传播的次序是前向传播计算的反序。
DDP 期望梯度在反向传递期间以前向传播的大致顺序来就绪。
2.5.2.2 分组依据
DDP 按照类型和设备作为key来分组，因为不同设备上的tensor不应该分在一组上，同类型张量应该分在一桶。用类型和设备作为key 就可以保证同设备上同类型张量分配在同一个桶里。
*/
// Tensors may be coalesced into buckets. Buckets must contain tensors of
// the same type, on the same device, so a bucket can identified by a
// composite key of a tensor's type identifier and its device.
struct BucketKey {
  BucketKey(c10::ScalarType type, c10::Device device)
      : type(type), device(device) {}

  const c10::ScalarType type;
  const c10::Device device;

  // See torch/csrc/utils/hash.h for dispatch code.
  static size_t hash(const BucketKey& key) {
    return c10::get_hash(key.type, key.device); // 用类型和设备作为key
  }
};

inline bool operator==(const BucketKey& lhs, const BucketKey& rhs) {
  return lhs.type == rhs.type && lhs.device == rhs.device;
}

} // namespace
/*
我们来看看 compute_bucket_assignment_by_size的具体逻辑：

定义了桶大小限制列表。bucket_size_limit_iterators。
定义了所有桶的列表 buckets，每一个实际桶可以认为是 BucketAccumulator。
遍历传入的所有张量：
给所有的tensor一个index，从0开始递增，一直到 tensors.size()，如果已经传入了 indices，就拿到张量的index。
如果配置了期待sparse gradient，则把这个张量自己放入一个桶，因为没法和其他张量放在一起。
使用张量信息构建桶的key，找到对应的桶。
拿到BucketAccumulator，往该桶的张量列表里面插入新张量的index，indices 是 tensor index list。
增加对应桶大小。
如果需要，就设定成大小限制的初始值。
拿到当前最小值限制。
如果桶的尺寸大于最小值限制，就是说目前桶的尺寸已经达到了桶的最大限制，按说需要转移到新桶了。
实际上确实转移到了逻辑上的新桶，但是实际还是在现有桶内执行，因为 type, device 还是同样的，还是应该在原有桶内继续累积，不过原有桶的indice已经转移到了result之中，就相当于清空了。
把桶内容插入到返回result，就是说，当桶尺寸过大的时候，就先插入到result之中。
重新生成桶，bucket是个引用，所以直接赋值，就相当于清空原有的桶，就是原来桶继续用，但是桶内原有的indices已经转移到了result之中。
前进到下一个尺寸限制。
把剩余的桶内indices插入到返回值，因为之前已经有些直接插入到了result之中。
对result 进行排序：
如果 tensor_indices 非空，说明张量的顺序已经是梯度准备好的顺序，不需要再排序了。
如果 tensor_indices 是空的，依据最小张量index来排序，这里假定张量的顺序是他们使用的顺序（或者说是他们梯度产生次序的反序）。这种排序可保证桶是按照连续不断的顺序准备好。
注意，这里就是正序排列，等到创建Reducer的时候，才反序传入：list(reversed(bucket_indices))。
最后返回 result，result 最终如下，里面每个vector 都对应了一个bucket，里面是都是 tensor 的 index，这里都是从小到大顺序排序。



result 最终如下，里面每个vector 都对应了一个bucket，里面是都是 tensor 的 index，这里都是从小到大顺序排序。

这里注意的是：因为 传入参数 tensors就是 parameters[0]，而 parameters[0] 是按照 parametes() 的返回结果来的，即，模型参数以（大致）Model.parameters()与给定模型相反的顺序分配到桶中 。使用相反顺序的原因是因为 DDP 期望梯度在反向传递期间以大约该顺序准备就绪。最终 DDP 是按model.parameters()的相反顺序启动AllReduce。

+-----------------------------------------------------------------------+
|                                                                       |
|  <tensor index 1, tensor index 2, tensor index 3, tensor index 4>     |
|                                                                       |
|                                                                       |
|  <tensor index 5, tensor index 6, tensor 7>                           |
|                                                                       |
|                                                                       |
|  ......                                                               |
|                                                                       |
|                                                                       |
|  <tensor index 8, tensor index 9, tensor index 10, tensor index 11>   |
|                                                                       |
+-----------------------------------------------------------------------+




其次，我们来看看 compute_bucket_assignment_by_size的具体逻辑：

生成一个计算结果 result，并且使用参数tensors的大小来为result预留出空间。

生成一个buckets，这是所有桶的列表，每一个实际桶可以认为是 BucketAccumulator

遍历传入的所有张量，对于每一个张量：

如果有index，就拿到张量的index。
如果配置了期待sparse gradient，则把这个张量自己放入一个桶，因为没法和其他张量放在一起。
使用张量信息构建桶的key。
使用 key 找到对应的桶, 拿到BucketAccumulator。
向该桶的张量列表 indices 里面插入新张量的index，indices 是 tensor index list。
增加对应桶大小。
如果需要，就设定成大小限制的初始值。
如果桶的尺寸大于最小值限制，就是说目前桶的尺寸已经达到了桶的最大限制，按说需要转移到新桶了（实际上确实转移到了逻辑的新桶，但是实际还是在现有桶内执行，因为 type, device 还是同样的，还是应该在原有桶内继续累积，不过原有桶的indice已经转移到了result之中，就相当于清空了）。
把桶内容插入到返回result，就是说，当桶尺寸过大的时候，就先插入到result之中。
利用 BucketAccumulator() 重新生成桶，bucket是个引用，所以直接赋值，就相当于清空原有的桶，就是原来桶继续用，但是桶内原有的indices已经转移到了result之中。
把剩余的桶内indices插入到返回值result。之前已经有些直接插入到了result之中。

对 result 进行排序：

如果 tensor_indices 非空，说明张量的顺序已经是梯度准备好的顺序，不需要再排序了。
如果 tensor_indices 是空的，依据最小张量index来排序，这里假定张量的顺序是他们使用的顺序（或者说是他们梯度产生次序的反序）。这种排序可保证桶是按照连续不断的顺序准备好。
注意，这里就是正序排列，等到创建Reducer的时候，才反序传入：list(reversed(bucket_indices))
另外需要注意的是：因为 tensors就是 Python 代码中的参数 parameters[0]，而 parameters[0] 是按照 parametes() 的返回结果来的，所以DDP最终是按model.parameters()的相反顺序启动AllReduce。


*/
std::tuple<std::vector<std::vector<size_t>>, std::vector<size_t>>
compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    const std::vector<size_t>& bucket_size_limits,  // 桶大小限制
    const std::vector<bool>& expect_sparse_gradient,
    const std::vector<int64_t>& tensor_indices, //实际上，初始化时候没有传入 tensor_indices
    const c10::optional<std::weak_ptr<c10d::Logger>>& logger) {
  // Either expect_sparse_gradient is not specified or it has as many elements
  // as the vector with tensors.
  TORCH_INTERNAL_ASSERT(
      expect_sparse_gradient.empty() ||
      (tensors.size() == expect_sparse_gradient.size()));
  TORCH_INTERNAL_ASSERT(!tensors.empty());
  // Store bucket indices and their sizes together, because we later sort the
  // resulting indices by minimum tensor index and want to keep sizes
  // consistent.
  std::vector<std::tuple<std::vector<size_t>, size_t>> result;
  // Sparse tensors go in their own bucket, so they do not have an enforced size
  // limit.
  size_t kNoSizeLimit = 0;
  result.reserve(tensors.size());  // 预留大小

  // Keep iterator into the size_limit vector by tensor type and device.
  // This is done so that we can use the consecutive bucket limits per type.
  std::unordered_map<
      BucketKey,
      std::vector<size_t>::const_iterator,
      c10::hash<BucketKey>>
      bucket_size_limit_iterators;

  // Keep vector of indices and size accumulator by tensor type and device.
  std::unordered_map<BucketKey, BucketAccumulator, c10::hash<BucketKey>>
      buckets;  // 所有桶的列表，每一个实际桶可以认为是 BucketAccumulator

  for (const auto i : c10::irange(tensors.size())) {  // 遍历传入的所有张量
    const auto& tensor = tensors[i];  //拿到张量
    auto msg = std::string("No support for sparse tensors.");
    if (logger.has_value()) {
      REDUCER_CHECK(!tensor.is_sparse(), logger.value(), msg);
    } else {
      TORCH_CHECK(!tensor.is_sparse(), msg);
    }

    // when tensor_indices is empty, the index of tensors[i] assigned to
    // bucket is i, otherwise the tensor index is tensor_indices[i].
    auto tensor_index = i; // 就是给所有的tensor一个index，从0开始递增，一直到 tensors.size()
    if (!tensor_indices.empty()) {
      tensor_index = tensor_indices[i];  // 如果有index，就拿到张量的index
    }
    // If we expect a sparse gradient to be produced for this tensor, it cannot
    // be grouped together with other gradients and gets its own bucket.
    // 如果配置了期待sparse gradient，则把这个张量自己放入一个桶，因为没法和其他张量放在一起
    if (!expect_sparse_gradient.empty() &&
        expect_sparse_gradient[tensor_index]) {
      result.emplace_back(std::vector<size_t>({tensor_index}), kNoSizeLimit);
      continue;
    }

    auto key = BucketKey(tensor.scalar_type(), tensor.device()); //使用张量信息构建桶的key
    auto& bucket = buckets[key];  // 找到对应的桶, 拿到BucketAccumulator
    bucket.indices.push_back(tensor_index); // 往该桶的张量列表里面插入新张量的index，indices 是 tensor index list
    bucket.size += tensor.numel() * tensor.element_size();  // 增加对应桶大小

    // Initialize bucket size limit iterator if necessary.
    // 如果需要，就设定成大小限制的初始值
    if (bucket_size_limit_iterators.count(key) == 0) {
      bucket_size_limit_iterators[key] = bucket_size_limits.begin();
    }
    // bucket_size_limit_iterator 就是桶大小的范围, 即 [_DEFAULT_FIRST_BUCKET_BYTES, int(bucket_cap_mb * 1024 * 1024)]
    auto& bucket_size_limit_iterator = bucket_size_limit_iterators[key];
    const auto bucket_size_limit = *bucket_size_limit_iterator;  // 当前最小值限制
    bucket.size_limit = bucket_size_limit;
    if (bucket.size >= bucket_size_limit) {
    //      // 如果桶的尺寸大于最小值限制，就是说目前桶的尺寸已经达到了桶的最大限制，按说需要转移到新桶了（实际上确实转移到了逻辑上的新桶，但是实际还是在现有桶内执行，因为 type, device 还是同样的，还是应该在原有桶内继续累积，不过原有桶的indice已经转移到了result之中，就相当于清空了）
      result.emplace_back(std::move(bucket.indices), bucket.size_limit); // 把桶内容插入到返回result，就是说，当桶尺寸过大的时候，就先插入到result之中
      bucket = BucketAccumulator(); // 重新生成桶，bucket是个引用，所以直接赋值，就相当于清空原有的桶，就是原来桶继续用，但是桶内原有的indices已经转移到了result之中。

      // Advance to the next bucket size limit for this type/device.
      // 前进到下一个尺寸限制
      auto next = bucket_size_limit_iterator + 1;
      if (next != bucket_size_limits.end()) {
        bucket_size_limit_iterator = next;
      }
    }
  }

  // Add remaining buckets. 把剩余的桶内indices插入到返回值，因为之前已经有些直接插入到了result之中
  for (auto& it : buckets) {
    auto& bucket = it.second;
    if (!bucket.indices.empty()) {
      result.emplace_back(std::move(bucket.indices), bucket.size_limit);
    }
  }

  // If tensor_indices is not empty, the order of the tensors is in the gradient
  // ready order, so no need to sort.
  // If tensor_indices is empty, sort resulting buckets by the minimum tensor
  // index they include. We assume that the order of the tensors is the order in
  // which they are used (or the reverse order in which their gradients are
  // produced). This sorting step ensures that the buckets are ready in
  // consecutive order.
  // 如果 tensor_indices 非空，说明张量的顺序已经是梯度准备好的顺序，不需要再排序了
  // 如果 tensor_indices 是空的，依据最小张量index来排序，这里假定张量的顺序是他们使用的顺序（或者说是他们梯度产生次序的反序）。这种排序可保证桶是按照连续不断的顺序准备好。
  // 注意，这里就是正序排列，等到创建Reducer的时候，才反序传入：list(reversed(bucket_indices))
  if (tensor_indices.empty()) {
    std::sort(
        result.begin(),
        result.end(),
        [](const std::tuple<std::vector<size_t>, size_t>& a,
           const std::tuple<std::vector<size_t>, size_t>& b) {
           // 对于任意两个vector，排序的依据是：用这两个vector之中最小index来排序
          auto indices_a = std::get<0>(a);
          auto indices_b = std::get<0>(b);
          const auto amin =
              std::min_element(indices_a.begin(), indices_a.end());  // a中的最小index
          const auto bmin =
              std::min_element(indices_b.begin(), indices_b.end());  // b中的最小index
          return *amin < *bmin;
        });
  }

  // Return bucket indices and size limits as separate entries in tuple, as some
  // APIs only need to consume bucket indices.
  std::vector<std::vector<size_t>> bucket_indices;
  bucket_indices.reserve(result.size());
  std::vector<size_t> per_bucket_size_limits;
  per_bucket_size_limits.reserve(result.size());
  for (const auto& bucket_indices_with_size : result) {
    bucket_indices.emplace_back(std::get<0>(bucket_indices_with_size));
    per_bucket_size_limits.emplace_back(std::get<1>(bucket_indices_with_size));
  }
  return std::make_tuple(bucket_indices, per_bucket_size_limits);
}

/*
通过下面代码我们可知，_verify_model_across_ranks 实际调用到verify_replica0_across_processes。

module.def(
    "_verify_model_across_ranks",
    &::c10d::verify_replica0_across_processes,
    py::arg("process_group"),
    py::arg("replicas"),
    py::call_guard<py::gil_scoped_release>());
verify_replica0_across_processes 之中，参数model_replicas 就是前面的 parameters，其逻辑如下：

首先，从 model_replicas 得到 metadata。
然后把metadata克隆到metadata_dev。
然后，把 process 0 的 metadata_dev 广播到对应的设备。
每个进程都会运行同样的代码，但是 process_group->broadcast 之中，只有 rank 0 会设置为 root_rank，这样就只广播 rank 0 的数据。
广播之后，所有进程的 metadata_dev 都一样，就是 process 0 内的数据。
然后把 metadata_dev 拷贝回 control，把 control 和 model_replicas[0]比较，看看是否和原来相等。
检查 control 是否和 model_replicas 的尺寸一样。
这里使用了 accessor，LibTorch 使用 accessor 快速访问 Tensor，如果 tensor 在CPU上，使用 accessor，如果在 GPU上，使用 packed_accessor 访问，这部分在 "核心开发者全面解读PyTorch 内部机制" 有相关提及。

*/
// Verifies corresponding params in the model replica have the same
// sizes/strides across processes.
void verify_params_across_processes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<at::Tensor>& params,
    const c10::optional<std::weak_ptr<c10d::Logger>>& logger) {
  // First verify number of parameters to avoid inconsistent inputs into
  // broadcast which can cause a crash.
  // See https://github.com/pytorch/pytorch/issues/73547

  at::TensorOptions param_size_options;
  param_size_options = param_size_options.dtype(at::kLong);
  param_size_options = param_size_options.device(params[0].device());
  // Note: Not using tensor building API because of
  // https://github.com/pytorch/pytorch/issues/74114
  at::Tensor param_size_tensor =
      at::tensor({static_cast<int64_t>(params.size())}, param_size_options);

  // Allgather and verify parameter size.
  std::vector<std::vector<at::Tensor>> param_size_output_tensors;
  param_size_output_tensors.emplace_back();
  auto world_size = process_group->getSize();
  for (size_t i = 0; i < world_size; ++i) {
    param_size_output_tensors.front().emplace_back(
        at::empty_like(param_size_tensor));
  }

  std::vector<at::Tensor> param_size_vec{param_size_tensor};
  process_group->allgather(param_size_output_tensors, param_size_vec)->wait();
  auto result_size_tensors = param_size_output_tensors.front();
  for (size_t i = 0; i < world_size; ++i) {
    auto param_size_for_rank = result_size_tensors[i][0].item<int>();
    TORCH_CHECK(
        param_size_for_rank == params.size(),
        c10::str(
            "DDP expects same model across all ranks, but Rank ",
            process_group->getRank(),
            " has ",
            params.size(),
            " params, while rank ",
            i,
            " has inconsistent ",
            param_size_for_rank,
            " params."));
  }

  // Continue with parameter shape verification.
  size_t i = 0;
  for (const auto& t : params) {
    i += 2 * t.dim();
  }
  at::TensorOptions options;
  options = options.dtype(at::kLong);
  auto metadata = at::empty({static_cast<long>(i)}, options);

  // Technically, process 0 is the broadcast source, so only process 0 needs
  // to populate metadata.  But no harm keeping work aligned across processes.
  auto metadata_accessor = metadata.accessor<int64_t, 1>();
  i = 0;
  // 把model_replicas[0]拷贝到metadata_accessor，其实就是metadata
  for (const auto& t : params) {
    for (const auto& sz : t.sizes()) {
      metadata_accessor[i++] = sz;
    }
    for (const auto& str : t.strides()) {
      metadata_accessor[i++] = str;
    }
  }

  // 然后把metadata克隆到metadata_dev
  auto metadata_dev = metadata.clone().to(params[0].device());
  std::vector<at::Tensor> vec{metadata_dev};

  ////  广播metadata_dev
  /*因为后续用到了如下代码，所以我们首先看看背景知识 broadcast。不熟悉这部分的朋友会有疑问是：为什么 broadcast 可以从 rank 0 广播到其他rank，明明所有rank都调用到了同样的 broadcast 代码。*/
  process_group->broadcast(vec)->wait();  // 把 rank 0 的 meta 广播到对应的设备

  // 这之后，metadata_dev 就是所有进程的结果大家都一样了
  // Technically, process 0 doesn't need to double-check metadata, because it
  // was the source.  But no harm keeping work aligned.
  auto control = at::empty({static_cast<long>(i)}, options);

  // 把 metadata_dev 拷贝回 control
  control.copy_(metadata_dev, /*non_blocking=*/false);

  // 然后把 control 和 model_replicas[0]比较，看看是否和原来相等
  auto control_accessor = control.accessor<int64_t, 1>();
  i = 0;
  for (const auto p : c10::irange(params.size())) {
    const auto& t = params[p];
    for (const auto& sz : t.sizes()) {
      auto msg = c10::str(
          "[",
          process_group->getRank(),
          "]: params[",
          p,
          "] in this process",
          " with sizes ",
          t.sizes(),
          " appears not to match sizes of the same param in process 0.");
      if (logger.has_value()) {
        REDUCER_CHECK(sz == control_accessor[i++], logger.value(), msg)
      } else {
        TORCH_CHECK(sz == control_accessor[i++], msg)
      }
    }
    for (const auto& str : t.strides()) {
      auto msg = c10::str(
          "params[",
          p,
          "] in this process",
          " with sizes ",
          t.sizes(),
          " appears not to match strides of the same param in process 0.");
      if (logger.has_value()) {
        REDUCER_CHECK(str == control_accessor[i++], logger.value(), msg)
      } else {
        TORCH_CHECK(str == control_accessor[i++], msg)
      }
    }
  }
}

} // namespace c10d
