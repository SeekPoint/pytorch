#pragma once

#include <atomic>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#endif
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/comm.hpp>
#include <c10d/Utils.hpp>
#include <c10d/default_comm_hooks.hpp>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/autograd/context/context.h>

namespace c10d {

constexpr int kDefaultFirstBucketBytes = int(1024 * 1024);
constexpr int kDefaultBucketBytesCap = int(25 * 1024 * 1024);
// Collect runtime stats once for every kDDPRuntimeLoggingSampleRate iterations.
constexpr int kDDPRuntimeLoggingSampleRate = 100;

/*
0x04 查询类
以下两个类用来让 autograd hook 函数确定张量对应桶。

4.1 VariableIndex
VariableIndex 就是确定某个 tensor 在某个桶中的位置。这个对于 autograd hook 有用。
对于autograd hook 回调，回调函数所在进程只是知道自己的梯度张量，
但是回调函数需要知道这个张量位于哪个replica，以及位于replica之中哪个位置，这样才能进一步规约。

4.1.1 成员变量
Reducer 等类的实例之中，只有一个 VariableIndex 的成员变量，这个独立成员变量是：

    std::vector<VariableIndex> unused_parameters_

VariableIndex 更多是作为其他成员变量的一部分或者参数存在，比如在 Reducer 之中，gradAccToVariableMap_ 就是使用了 VaribaleIndex。

    std::unordered_map<torch::autograd::Node*, VariableIndex>
          gradAccToVariableMap_;
          // 存了grad_accumulator & index 的对应关系，这样以后在 autograd graph 寻找 unused parameters 就方便了

4.1.2 定义
VariableIndex 定义如下：

*/
// Locates a specific variable by replica index and variable index.
struct VariableIndex {
  size_t replica_index; // 位于哪个replica
  // variable index，注意，不是"位于replica之中哪个位置"，而是所有 varibale的index，
  //比如一共有10个参数，variable_index 的取值是从0～9。那么"位于replica之中哪个位置"由什么来确定？
  //由下面的 VariableLocator 确定。
  size_t variable_index;

  VariableIndex() = default;

  VariableIndex(size_t replica_index_, size_t variable_index_) {
    replica_index = replica_index_;
    variable_index = variable_index_;
  }

  static size_t hash(const VariableIndex& key) {
    return c10::get_hash(key.replica_index, key.variable_index);
  }
};

inline bool operator==(const VariableIndex& lhs, const VariableIndex& rhs) {
  return lhs.replica_index == rhs.replica_index
    && lhs.variable_index == rhs.variable_index;
}

//0x02 Reducer 定义
//Reducer提供了反向传播中梯度同步的核心实现，其定义相当复杂，我们甚至需要去掉一些不重要的成员变量以便展示：
class Reducer {
 public:
  // The constructor takes a list of variables for every model replica.
  // The bucket assignment for this reducer is specified as a list of
  // buckets, each of which is specified as a list of indices into the
  // variables list for **a single replica** (i.e. `variables[0]`).
  explicit Reducer(
      std::vector<std::vector<at::Tensor>> replicas,
      std::vector<std::vector<size_t>> bucket_indices,
      c10::intrusive_ptr<c10d::ProcessGroup> process_group,
      std::vector<std::vector<bool>> expect_sparse_gradients,
      int64_t bucket_bytes_cap,
      bool find_unused_parameters,
      bool gradient_as_bucket_view,
      std::unordered_map<size_t, std::string>
          paramNames);

  ~Reducer() noexcept(false);

  // To (re-)initialize bucket assignment, pass a list of buckets, each
  // of which is specified by a list of indices in the variables list.
  // This function performs validation that the variables within a bucket
  // all live on the same device and have the same dimensionality.
  void initialize_buckets(std::vector<std::vector<size_t>> bucket_indices);

  // This function is called when the forward function has produced an output,
  // and the user wishes to reduce gradients in the backwards pass.
  // If they don't, and wish to accumulate gradients before reducing them,
  // a call to this function can simply be omitted.
  void prepare_for_backward(const std::vector<at::Tensor>& outputs);

  // Called at the begginning of forward() inside DistributedDataParallel,
  // right now it caputures the starting time of forward in each iteration.
  void prepare_for_forward();

  // Returns the relative time in nanoseconds when gradients were ready,
  // with respect to the time `prepare_for_backward` was called. The outer
  // vector is for model replicas and the inner vector is for parameters.
  std::vector<std::vector<int64_t>> get_backward_stats() const {
    return backward_stats_;
  }

  // Registers a hook to the reducer. The hook is `CommHookInterface`
  // type to allow both Python and CPP hooks. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_builtin_comm_hook`.
  void register_comm_hook(std::unique_ptr<CommHookInterface> iface);

  // Registers a built-in C++ comm hook to the reducer. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_comm_hook`.
  void register_builtin_comm_hook(c10d::BuiltinCommHookType comm_hook_type);

  // Returns a vector of tensors in each bucket in sequential order.
  std::vector<std::vector<at::Tensor>> get_bucket_tensors() const;

  // Rebuild buckets based on rebuilt_params_ and rebuilt_param_indices_
  // according to when tensors received grads in the backward pass.
  // TODO this function makes broadcast communication call and
  // could be overlapped with next forward() call, thus
  // it could be async. Will make it async when rebuilding buckets for
  // find_unused_parameters = true case, as we could rebuild buckets more than
  // once for find_unused_parameters = true case, where subgraphs are trained
  // and parameter indices order may change more frequently.
  // For find_unused_parameters = false case, buckets are only rebuilt once,
  // the performance cost is negligible. Returns true if the buckets were
  // rebuilt.
  bool rebuild_buckets();

  // Returns true if we should rebuild buckets, else false. We only rebuild
  // buckets once after the first iteration and never rebuild them if
  // find_unused_parameters_.
  inline bool should_rebuild_buckets() const {
    return (static_graph_ || !find_unused_parameters_) && !has_rebuilt_bucket_;
  }

  // Pushes all parameters to be rebuilt.
  void push_rebuilt_params_for_all_indices();

  // Creates and sets ForwardPassWorkHandle given a ProcessGroup::Work and the
  // corresponding tensor being reduced.
  void set_forward_pass_work_handle(
      c10::intrusive_ptr<c10d::ProcessGroup::Work> forwardPassWorkHandle,
      bool useStaticWorldSize);

  // Retrieve on-device tensors used to track locally unused parameters. For
  // each replica, it is a tensor where index i = 1 if the Variable with that
  // index has been used.
  std::vector<at::Tensor> get_local_used_maps_on_device() const;

  // Saves thread local state to be used by autograd engine callbacks.
  void save_thread_local_state();

  // An function for users to set sample_rate of collecting
  // runtime stats. The time stats will be recorded for the
  // first 10 iterations, after 10 iteratons time stats will be
  // recorded once every "sample_rate" training iterations.
  void set_ddp_runtime_logging_sample_rate(int sample_rate);

  // Specify the training graph is static.
  void set_static_graph();

  // Delay all reduce to be after all gradients' calculation is complete.
  void delay_all_reduce();

 protected:
  // Forward declaration.
  struct Bucket;

  void push_rebuilt_params(const VariableIndex& index);

  mutable std::mutex mutex_;
  const std::vector<std::vector<at::Tensor>> replicas_;  // 传入的张量
  const c10::intrusive_ptr<::c10d::ProcessGroup> process_group_;  // 进程组
  std::vector<std::vector<bool>> expect_sparse_gradients_;

  std::vector<std::vector<std::shared_ptr<torch::autograd::Node>>>
      grad_accumulators_; // 对应的 index 存了相应的 grad_accumulator，就是 tensor index对应的grad_accumulator
  std::unordered_map<torch::autograd::Node*, VariableIndex>
      gradAccToVariableMap_; // 存了grad_accumulator & index 的对应关系，这样以后在 autograd graph 寻找 unused parameters 就方便了

  /*
5.7.2 hooks_
其作用就是保持了 autograd hook，也是起到了bookkeeping 作用。
  */
  std::vector<std::pair<uintptr_t, std::shared_ptr<torch::autograd::Node>>>
      hooks_;

  bool expect_autograd_hooks_;
  bool require_finalize_;
  size_t next_bucket_;

  bool has_marked_unused_parameters_;
  const bool find_unused_parameters_;
  const bool gradient_as_bucket_view_;
  std::vector<VariableIndex> unused_parameters_; // 如果没有用到，直接设置为就绪，第一次迭代之后久不会改变了
  // Locally used parameter maps indicating if parameters are used locally
  // during the current iteration or no_sync session if no_sync is on. One
  // tensor for each model replica and each tensor is one-dim int32 tensor of
  // number of parameters. These tensors are marked in autograd_hook to indicate
  // the corresponding param has been used, and get allreduced in the end of
  // backward of current iteration or no_sync session for figuring out the
  // globally unused parameters.
  //
  // local_used_maps_:     CPU tensors for bookkeeping locally used params
  // local_used_maps_dev_: dev tensors for reducing globally unused params
  std::vector<at::Tensor> local_used_maps_; // autograd_hook中会设置，对应论文中的
  std::vector<at::Tensor> local_used_maps_dev_; // GPU
  // Indicate that reduction is done and D2H copy is done as well.
  bool local_used_maps_reduced_;

  // Work handle for allreduce on local_used_maps_
  c10::intrusive_ptr<c10d::ProcessGroup::Work> local_used_work_;

  void mark_variable_ready_dense(VariableIndex index);

  void mark_variable_ready_sparse(VariableIndex index);

  void mark_variable_ready(VariableIndex index);

  void autograd_hook(VariableIndex index);

  void mark_bucket_ready(size_t bucket_index);

  void finalize_bucket_dense(Bucket& replica);

  void finalize_backward();

  // Asserts that the reduction for the previous iteration has finished before
  // rebuilding buckets or kicking off the next one.
  void ensure_prior_reduction_finished();

  // Broadcast rebuilt buckets from rank 0 to other ranks before initializing
  // the buckets
  void sync_bucket_indices(std::vector<std::vector<size_t>>& bucket_indices);

  using GradCallback =
      torch::distributed::autograd::DistAutogradContext::GradCallback;
  void runGradCallbackForVariable(at::Tensor& variable, GradCallback&& cb);

  // A bucket replica represents [1..N] gradients to be reduced,
  // with the same dtype, on the same device.
  //
  // Batching gradients together before reducing them can result in lower
  // overhead and/or faster time to completion. Only gradients of the same type
  // and on the same device can be batched. The tensor that represents the
  // flattened gradient uses the same type and is placed on the same device.
  // Buckets are filled as the gradients they hold are computed (triggered by
  // autograd hooks). Buckets are reduced in a predetermined order that is
  // identical across processes.
/*
3.2 定义
BucketReplica 具体定义为：
目前为止，逻辑如下，如前所述，每个bucket只有 replicas[0] 有意义。

                                    +-----------------------------------------------------+
+----------------------------+      | +-------+      +----------------------------------+ |
| Reducer                    |      | |Bucket |      |Bucket                            | |
|                            |      | |       |      |                                  | |
|                            |      | |       |      |            Future  future_work   | |
|  vector<Bucket> buckets_ +------> | |       | ...  |                                  | |
|                            |      | |       |      |       ProcessGroup::Work  work   | |
|                            |      | |       |      |                                  | |
|                            |      | |       |      | vector<size_t> variable_indices  | |
|                            |      | |       |      |                                  | |
|                            |      | |       |      |  vector<BucketReplica> replicas  | |
|                            |      | |       |      |                          +       | |
|                            |      | |       |      |                          |       | |
|                            |      | |       |      |                          |       | |
+----------------------------+      | +-------+      +----------------------------------+ |
                                    +-----------------------------------------------------+
                                                                                |
                                                                                |
                                                                                v
                           +--------------------------------------------------------------+
                           | +---------------+       +----------------------------------+ |
                           | |BucketReplica  |       | BucketReplica                    | |
                           | |               |       |                                  | |
                           | |               |       |                                  | |
                           | |               |       |  vector<Tensor> bucket_views_in  | |
                           | |               |  ...  |                                  | |
                           | |               |       |  vector<Tensor> bucket_views_out | |
                           | |               |       |                                  | |
                           | |               |       |  Tensor contents                 | |
                           | |               |       |                                  | |
                           | |               |       |  vector<Tensor> variables        | |
                           | |               |       |                                  | |
                           | |               |       |                                  | |
                           | +---------------+       +----------------------------------+ |
                           +--------------------------------------------------------------+

*/
  struct BucketReplica {
    // Flattened (1 dimensional) contents of bucket.
    at::Tensor contents; // 这里打平了

    // Views into contents for each grad.  Each view will be created with
    // layout (sizes + strides) matching the grad's expected layout
    // ("Gradient Layout Contract" in torch/csrc/autograd/AccumulateGrad.h).
    // `bucket_views_in[i].copy_(grad)` and
    // `grad.copy_(bucket_views_out[i])`
    // provide convenient ways to move grad data in/out of contents.
    // The reason we keep two states for bucket_views is that if DDP
    // communication hook was registered, `bucket_views_out` could be
    // re-initialized with the value of hook's `future_work`. We still need to
    // keep a separate view reference to replica's original contents for
    // `bucket_views_in[i].copy_(grad)` call.
    std::vector<at::Tensor> bucket_views_in; // 怎么从contents 之中查找
    std::vector<at::Tensor> bucket_views_out; // 一个输出视图

    // Variables that contribute to this bucket replica. Use refcounted value
    // here so that we can easily unflatten the bucket contents into the
    // participating variables after reduction has completed.
    std::vector<at::Tensor> variables;

    // Per-variable offset/length into the flat bucket contents tensor and grad
    // bucket.
    std::vector<size_t> offsets;
    std::vector<size_t> lengths;

    // Per-variable sizes into the grad bucekt.
    std::vector<c10::IntArrayRef> sizes_vec;

    // Number of tensors to be added before this bucket is complete.
    // This is reset to `variables.size()` every iteration.
    size_t pending;

    // TODO(@pietern)
    // Memory copies from gradient tensors into the bucket are potentially
    // done on different CUDA streams. We record an event for every copy
    // so that we can synchronize with them prior to kicking off the reduction.
    // std::vector<at::cuda::CUDAEvent> events;
  };

  // This function is called inside `initialize_buckets`, it initializes both
  // bucket_views_in and bucket_views_out into the contents tensor for each
  // variable's grad. Views serve as entry points to copy_ each grad's data
  // in/out of the flat contents tensor.
  void initialize_bucket_views(BucketReplica& replica, at::Tensor& contents);

  // This function is called inside `finalize_backward`, it happens only if
  // DDP communication hook was registered to recreate just bucket_views_out
  // with the result of `future_work`.
  void populate_bucket_views_out(BucketReplica& replica, at::Tensor& tensor);

  // If gradient_as_bucket_view_ is false, after allreduce buckets,
  // copy bucket results back to grads.
  void copy_bucket_to_grad(
      at::Tensor& variable,
      Reducer::BucketReplica& replica,
      size_t intra_bucket_index,
      bool global_unused);
  // Check layout of grad and bucket_view before calling copy_grad_to_bucket
  void check_grad_layout(const at::Tensor& grad, const at::Tensor& bucket_view);
  // If gradient_as_bucket_view_ is false, before allreduce buckets,
  // copy grads to buckets.
  void copy_grad_to_bucket(const at::Tensor& grad, at::Tensor& bucket_view);

  // A bucket holds N bucket replicas (1 per model replica).
  //
  // If every bucket in this struct is ready, the reduction can be kicked off.
  // One bucket per replica. Reduction is kicked off when every bucket is ready.
  //
  struct Bucket {
    std::vector<BucketReplica> replicas;  // 每个模型副本对应一个桶

    // Global indices of participating variables in the bucket
    std::vector<size_t> variable_indices; // 具体每个桶里面有哪些 variable。

    // Number of replicas to be marked done before this bucket is ready.
    size_t pending;  // 计数，

    // Keep work handle around when this set of buckets is being reduced.
    c10::intrusive_ptr<c10d::ProcessGroup::Work> work;

    // Keep future work handle around if DDP comm hook is registered.
    c10::intrusive_ptr<torch::jit::Future> future_work;

    // If this bucket should expect a single sparse gradient.
    // Implies: replicas[i].variables.size() == 1.
    bool expect_sparse_gradient = false;
  };

  std::vector<Bucket> buckets_;
/*
4.2 VariableLocator
4.2.1 定义
VariableLocator 用来在 bucket 之中确定一个varaible。为了找到一个张量位置，我们需要知道在哪个桶，在桶的张量之中的哪个位置。

    哪个桶 : bucket_index 是Reducer.buckets_列表的位置，表示 buckets_ 之上的一个bucket。
    桶副本的哪个位置 : intra_bucket_index 是在 bucket.replica 之中 vector 域的 variable index。

*/
  // A variable locator locates a particular variable in the bucket
  // structure. The `bucket_index` field points to the bucket in the `buckets_`
  // vector. The `intra_bucket_index` field points to the index of the variable
  // in any of the vector fields in the bucket replica.
  struct VariableLocator {
    // Index into the `buckets_` variable.
    size_t bucket_index; // 哪个桶
    // Index of parameter in single bucket replica.
    size_t intra_bucket_index; // 在桶副本的哪个位置

    VariableLocator() = default;

    VariableLocator(size_t bucket_index_, size_t intra_bucket_index_) {
      bucket_index = bucket_index_;
      intra_bucket_index = intra_bucket_index_;
    }
  };
//4.2.2 成员变量
//Reducer 的成员变量为：
  // Map the index of a variable to its location in the bucket structure.
  std::vector<VariableLocator> variable_locators_;

  // track the number of iterations to synchronize grads in training so far.
  long num_iterations_;
  // track the number of buckets that have been ready for
  // communication calls like allReduce or communication hooks.
  int num_buckets_ready_;

  // CPU timestamp to record event start and end time.
  struct CPUTimer {
    // The timestamp of forward call start time in each iteration.
    int64_t forward_start_time;
    // The timestamp of backward computation start and end time in each
    // iteration.
    int64_t backward_compute_start_time;
    int64_t backward_compute_end_time;
    // The timestamp of first communication call start time in each iteration.
    int64_t backward_comm_start_time;
    // The timestamp of last communication call end time in each iteration.
    int64_t backward_comm_end_time;
  };

  CPUTimer cpu_timer_{};

#ifdef USE_CUDA
  // GPU events to record event start and end time.
  struct GPUTimer {
    at::cuda::CUDAEvent forward_start = at::cuda::CUDAEvent(cudaEventDefault);
    at::cuda::CUDAEvent backward_compute_start =
        at::cuda::CUDAEvent(cudaEventDefault);
    at::cuda::CUDAEvent backward_compute_end =
        at::cuda::CUDAEvent(cudaEventDefault);
    at::cuda::CUDAEvent backward_comm_start =
        at::cuda::CUDAEvent(cudaEventDefault);
    at::cuda::CUDAEvent backward_comm_end =
        at::cuda::CUDAEvent(cudaEventDefault);
  };
  GPUTimer gpu_timer_;
#endif

  // We collect the relative timestamp of every gradient being ready
  // when executing autograd. This can be used to derive a timeline of
  // the point in time buckets were ready, or ideal bucket assignment/ordering.
  std::vector<std::vector<int64_t>> backward_stats_;

  bool should_collect_runtime_stats();
  void record_forward_compute_start_time();
  void record_backward_compute_start_time();
  void record_backward_compute_end_time();
  void record_backward_comm_start_time();
  void record_backward_comm_end_time();

  int get_ddp_runtime_logging_sample_rate();
  int ddp_runtime_logging_sample_rate_ = kDDPRuntimeLoggingSampleRate;

  bool is_multi_device_module_ = false;

  // Following variables are to help build dynamic bucket order
  bool has_rebuilt_bucket_;
  std::vector<at::Tensor> rebuilt_params_;
  std::vector<int64_t> rebuilt_param_indices_;
  const int64_t bucket_bytes_cap_;

/*
5.7 计算梯度支撑类
我们接下来分析一些计算梯度所涉及到的基本函数和支撑类。

5.7.1 RpcContext
该类用来封装 distributed::autograd::ContextPtr。
*/
  struct RpcContext {
    using ContextPtr = torch::distributed::autograd::ContextPtr;
    // The shared_ptr is to hold the context instance.
    ContextPtr context_ptr_holder;
    std::atomic<ContextPtr::element_type*> context_ptr{nullptr};

    void set(ContextPtr&& new_context_ptr);
  };
  RpcContext rpc_context_;

  // A struct containing work handle and tensor for allreduce scheduled in
  // forward pass, if applicable.
  struct ForwardPassAllreduceWork {
    c10::intrusive_ptr<c10d::ProcessGroup::Work> workHandle;
    at::Tensor resultTensor;
    // whether we should divide by the initial world_size or the no. of
    // remaining DDP ranks.
    bool useStaticWorldSize;
  };

  // Handle for the currently scheduled allreduce in the forward pass, if
  // applicable.
  ForwardPassAllreduceWork forwardPassWorkHandle_;

  // Division factor for reduction of gradients.
  int divFactor_;

  bool static_graph_;
/*
5.3 numGradHooksTriggeredMap_
记录在本张量的梯度就绪之前，该张量的 autograd_hook 应该被调用几次。第一次迭代之后，不再增加，所以这个数值应该就是1或者0。
用来设置 unused_parameters_ 和 配置 numGradHooksTriggeredMapPerIteration_。
5.3.1 初始化
如何初始化？在构建函数之中有：
    numGradHooksTriggeredMap_[index] = 0;
第一次迭代之后，后续调用 autogrid_hook 就递增加一。
*/
  // Key: VariableIndex, Value: the number of times that a variable's autograd_hook()
  // should be triggered before marking this variable's grad as ready for communication.
  // Map will not change after 1st iteration.
  std::unordered_map<VariableIndex, int, c10::hash<VariableIndex>> numGradHooksTriggeredMap_;

/*
5.4 numGradHooksTriggeredMapPerIteration_
在本张量的梯度就绪之前，该张量的 autograd_hook 还需要被调用几次。如果为0，就说明这个桶应该整体就绪了。
本成员变量是使用 numGradHooksTriggeredMap_ 来重置。
*/
  // Key: VariableIndex, Value: the number of times that a variable's autograd_hook()
  // are left to be triggered before marking this variable's grad as ready for communication.
  // Map will change after 1st iteration to track a grad is ready for communication or not.
  std::unordered_map<VariableIndex, int, c10::hash<VariableIndex>> numGradHooksTriggeredMapPerIteration_;

 private:
  // reset counting for buckets before backward starts
  void reset_bucket_counting();
  // search unused parameters beore backward starts
  void search_unused_parameters(
      const std::vector<torch::autograd::Variable>& outputs);
  void set_divide_factor();
  // kick off all reduce for the ready bucket
  void all_reduce_bucket(Bucket& bucket);
  // kick off all reduce to local used map, it can help find global unused parameters
  void all_reduce_local_used_map();
  // initialize locally used parameter maps
  void initialize_local_used_map();
  // get current cuda stream
  const c10::Stream get_current_stream();
  bool dynamic_graph_find_unused();
  bool static_graph_first_iteration();
  bool static_graph_after_first_iteration();

  // comm_hook_ is used to access the DDP communication hook if registered.
  std::unique_ptr<CommHookInterface> comm_hook_;
  // Current thread local state
  at::ThreadLocalState thread_local_state_;
  // Debug level setting. It is parsed once when Reducer is constructed, and
  // remains the same across a single invocation of DDP training.
  DistributedDebugLevel ddp_debug_level_;
  // Mapping of variable index to fully qualified name of model to notify users
  // about errors when certain parameters do not get gradient.
  std::unordered_map<size_t, std::string> param_names_;
  // Per iteration set of parameter indices that have been marked ready.

  /*
5.5 perIterationReadyParams_
每个迭代之中，perIterationReadyParams_ 表示就绪的参数。
  */
  std::unordered_set<size_t> perIterationReadyParams_;
  // Retrieves parameter names that have not been marked as ready as part of
  // previous iteration.
  std::vector<std::string> getUnmarkedParamsForIteration();
  // Retrives parameter indices that have not been marked as ready as part of
  // previous iteration.
  std::vector<size_t> getUnmarkedParamIndicesForIteration();
  // Raises appropriate error if mark_variable_ready is called on the same
  // variable twice, which is unexpected.
  void checkAndRaiseMarkedTwiceError(size_t curVariableIndex);

  friend class Logger;
};

// This is equivalent to take_tensors but returns indices into the
// tensor list argument for bucket assignment. Also, it is aware
// of device placement and will not allow buckets to span devices.
// The index of tensors[i] assigned to bucket is tensor_indices[i],
// when tensor_indices is empty, the index of tensors[i] assigned to
// bucket is i.
std::vector<std::vector<size_t>> compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    const std::vector<size_t>& bucket_size,
    const std::vector<bool>& expect_sparse_gradient = {},
    const std::vector<int64_t>& tensor_indices = {});

// Verify models across all processes are the same as model on rank 0 with
// respect to no. of params and matching dtype/size/layout.
void verify_replica0_across_processes(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    std::vector<std::vector<at::Tensor>> model_replicas);
} // namespace c10d
