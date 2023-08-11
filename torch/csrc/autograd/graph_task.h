#pragma once
#include <ATen/ThreadLocalState.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/utils/warnings.h>
#include <vector>

namespace torch {
namespace autograd {

using edge_list = std::vector<Edge>;
struct ReadyQueue;
/*
因为线程部分比较复杂，所以我们从启动部分提出来单独分析。

因为大型模型往往梯度数目太多，所以PyTorch 采用了多线程处理。
为了应对各种情况，PyTorch 定义了一个线程变量 worker_device。
引擎生成的线程都被分配一个 "worker_device"，指定它们为哪个设备处理工作。此变量在以下位置初始化：
    在CUDA，XLA设备线程的创建时间初始化，因为他们正等待在自己的设备上工作。
    在CPU线程的图任务执行之前初始化，因为对于每个后向调用，我们使用调用线程（caller thread）来驱动引擎执行。

    除了上述线程之外，引擎还有一个主线程。这里使用 NO_DEVICE 来标识。如前所示，也会用 CPU_DEVICE 来临时做重入判别，但是依然是主线程。
*/
static constexpr int NO_DEVICE = -2;
static constexpr int CPU_DEVICE = -1;

namespace {
std::atomic<uint64_t> graph_task_id{0};
}
/*
GraphTask 其主要成员变量如下：
    outstanding_tasks_ ：用来记录当前任务数目，如果数目为0，则说明任务结束了。 如果这个数量不为0，则此GraphTask依然需要运行。
    dependencies_ ：用来判断后续节点是否已经可以被执行。
    not_ready_ ：存储没有完成的function和其输入。
    grad_mode_ ：是否需要进行梯度计算。反向计算期间执行的代码逻辑依靠AutoGradMode::is_enabled() 来判断当前是否是要计算grad。
    owner : GraphTask 所属线程的Device 数值。GraphTask是在哪个线程中创建的，该值就是那个线程中的worker_device的值。
    cpu_ready_queue_ ：
    CPU线程专用于处理反向传播之中的CPU相关工作。因此所有Graph task都会维护自己的cpu_ready_queue_，CPU相关任务应该将发送到该队列。
    对于每个GraphTask，我们维护cpu_ready_queue_，这样在设备线程（即GPU）上执行时，如果是下一个NodeTask 应该在CPU上运行，我们就知道应该推送 NodeTask 到哪个就绪队列。
    mutex_ ：保护如下变量：not_ready_, dependencies_, captured_vars，has_error_, future_result_, cpu_ready_queue_, and leaf_streams。
    keep_graph ：用来指定一次反向计算后是否释放资源。
    具体定义如下，这里只给出成员变量：
*/
// GraphTask holds metadata needed for a single execution of backward()
struct GraphTask : std::enable_shared_from_this<GraphTask> {
//outstanding_tasks_是待处理 NodeTask的数量，用来判断该GrapTask是否还需要执行，其数值总是先加再减，如果数目为0，则说明任务结束了。
//当 GraphTask 被创建出来时候，此数值为0。
//如果有一个NodeTask被送入到 ReadyQueue，则outstanding_tasks_ 增加 1。
//如果在工作线程作执行一次 evaluate_function(task)后，outstanding_tasks的值减1。
//如果这个数量不为0，则此GraphTask依然需要运行。
  std::atomic<uint64_t> outstanding_tasks_{0};

  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error_{false};
  std::atomic_bool future_completed_{false};
  // It is safe to read keep_graph_ without synchronization
  bool keep_graph_;

  // To protect reads/writes to not_ready_, dependencies_, captured_vars_,
  // has_error_, future_result_, cpu_ready_queue_, and leaf_streams.
  std::mutex mutex_;

  /*not_ready_
  是针对未就绪节点和其输入的map，假设某节点 A 在反向传播路径上有两个输入，
  当第一个输入完成时候，因为第二个输入没有完成反向计算，
  所以需要有一个地方暂存这个 A 和 其第一个输入以备后续处理。not_ready_ 就是用来做这个的。
    not_ready_ 的 key 是未就绪节点，value 是这个节点目前就绪的输入列表。
    第一次遇到某节点的一个输入之后，就把 (节点 A, A 的输入信息 ) 放入 not_ready_ 这里，得到 (节点 A, [A 的输入信息 1 ] )
    后续遇到 A 的其他输入，就继续调整这里，把 A 的其他输入加入到 "A 的输入信息" 之中，比如得到 (节点 A, [A 的输入信息 1，A的输入信息 2 ] )
    如果 此时 A 已经 ready，就把 A 和其输入信息 放入 一个 Ready Queue，然后从 not_ready_ 移除 节点 A。
    如果 A 还没有 ready（A还需要其他输出），就继续维持 not_ready_ 的状态，把目前 A 输入都加入到 not_ready_ 里面。
  */
  std::unordered_map<Node*, InputBuffer> not_ready_;

  std::unordered_map<Node*, int> dependencies_;
  //dependencies 用来判断后续节点是否已经可以被执行
    //  dependencies成员在compute_dependencies调用中被初始化，
    //  只要一个grad_fn函数在别人的next_edges()中出现过一次，那么dependencies[this_grad_fn] 就自增1。
    //  如果dependencies[this_grad_fn]大于0，说明this_grad_fn有一个后向传播的依赖，即this_grad_fn需要等被依赖者完成，才能进行反向传播。

  // Records the nodes that are in the graph
  std::unordered_set<Node*> nodes_in_graph_;
  c10::SmallVector<Node*, 4> graph_roots_;
  // Note [Exec info]
  // Exec info is created for each GraphTask, which allows filtering paths on
  // the graph that are not needed. It has a bit complicated semantics. If it's
  // empty, it means the task is run in a "default" mode, which means that all
  // next_edges we encounter should get executed. If it's not empty, only
  // functions that have an entry and this entry has needed == True should be
  // executed. exec_info is only empty when the graph is executed via
  // .backward() and the inputs parameter is not passed. Otherwise, when
  // executed through .grad(), or when inputs arg is specified for .backward(),
  // exec_info will be non-empty.
  // ExecInfo 主要作用就是判断是否需要执行，并且注册了一个hook，用来在计算梯度时候做调用。
  /*exec_info_ 的作用就是给 GraphTask 的每一个Node配置一个ExecInfo，就是执行信息。
    如果exec_info_为空，说明该task运行在默认模式，即，所有遇到的 next_edges 都需要执行。
    如果 exec_info_ 非空，说明只有特定 functions 才会被执行，这些 Functions 的特点是：拥有 entry，并且这个 entry 的 “has needed == True”。
    exec_info_ 何时为空？何时非空？
    当图被用 .backward() 执行，并且没有传递输入参数，则 exec_info 为空，就是全部执行。
    如果只是使用用 .grad() 执行，或者使用.backward() 执行时候并且给定输入参数，那么 exec_info_ 非空。
    所以，exec 和 captured_vars_ 就是针对 grad() 和指定参数的 backward()，就是标注在这种情况下需要计算哪些梯度。在这种情况下，只有某些节点需要执行，从这些节点开始，有一条路径通向 outpus。
    */
  struct ExecInfo {
    struct Capture {
      Capture(const Capture&) = delete;
      Capture(Capture&&) = default;

      Capture(int input_idx, int output_idx)
          : input_idx_(input_idx), output_idx_(output_idx) {}
      int input_idx_; // within Node inputs
      int output_idx_; // within the output vector of a GraphTask

      // This hook will be executed after a grad is captured. The captured
      // grad will be replaced by the return value of the hook.
      struct GradCaptureHook {
        virtual ~GradCaptureHook() = default;
        virtual at::Tensor operator()(const at::Tensor& grad) = 0;
      };
      // NOTE [Deprecated capture hooks]
      //
      // The current status of capture hooks is that we continue to support
      // the single usage of it by distributed in the dist_engine. If anyone
      // else needs to use it for other purposes, they should file an issue.
      //
      // Capture hooks were originally created because there did not exist
      // any way to register pre/post hooks to grad_fn in a way such that it
      // would still be executed even if that is the grad_fn of a Tensor
      // passed as input= of .grad. As far as I know, only dist_engine uses
      // this hook.
      //
      // However, there are other alternatives today like tensor hooks that can
      // replace the usage that originally motivated its creation. Also,
      // Captures hooks are an outlier in terms of the types of hook that
      // autograd offers in how it is registered and behaves, e.g. it is a hook
      // registered not to the graph, but to a particular graph_task! This makes
      // it a burden to maintain.
      //
      // It would be very nice to clean up/do a migration from pre/post
      // hooks used in distributed to use tensor hooks, but for now we just
      // mark this method as deprecated to prevent additional usage.
      //
      // If you still think you really need to capture hooks, please file an
      // issue (and tag autograd).
      const std::vector<std::unique_ptr<GradCaptureHook>>&
      DO_NOT_USE_DEPRECATED_get_capture_hooks() const {
        return hooks_;
      }
      // See NOTE [deprecated capture hooks]
      void DO_NOT_USE_DEPRECATED_register_capture_hook(
          std::unique_ptr<GradCaptureHook> hook) {
        hooks_.push_back(std::move(hook));
      }

     private:
      // The hooks will be called one by one in the order as they were added.
      // The input grad of a hook will be the output of its preceding hook. The
      // first hook will take the captured grad as the input. The output of the
      // last hook will replace the captured grad.
      std::vector<std::unique_ptr<GradCaptureHook>> hooks_;
    };

    bool should_execute() const {
      return needed_ || captures_;
    }

    bool needed_ = false;
    std::unique_ptr<std::vector<Capture>> captures_;
  };
  // exec_info_ is safe to read without synchronization
  std::unordered_map<Node*, ExecInfo> exec_info_;
  // Captures variables are grads captured that we return to the user. After
  // execution of the GraphTask is completed, the captured_vars_ are moved
  // out of the GraphTask and are no longer valid.
  //Captures variables是我们返回给用户的捕获梯度。GraphTask执行完成后，Captures variables 将移出GraphTask，不再有效。
  //这个 captured_vars_ 是可以进行后续处理，就是使用上面提到的GradCaptureHook 在 evaluate_function 进行处理，
  //具体赋值也是在 evaluate_function 其中，参见前面代码之中的注释，我们后文详细对函数也会有分析。
  std::vector<Variable> captured_vars_;

  // Note: this field is not ready to be used until the proper
  // `thread_locals_.set_grad_mode()` call in the constructor.
  at::ThreadLocalState thread_locals_ = at::ThreadLocalState();

  std::unordered_set<c10::Stream> leaf_streams;

  // Per-device current streams of the execute() that called this GraphTask.
  // These will be synced with leaf_streams in exec_post_processing.
  std::vector<c10::optional<c10::Stream>> caller_current_streams_;

  // Collects caller_current_streams_
  void stash_current_streams();

  void init_to_execute(
      Node& graph_root,
      const edge_list& outputs,
      bool accumulate_grad,
      uint64_t min_topo_nr);

  // The value of worker_device in the thread that created this task.
  // See Note [Reentrant backwards]
  // Safe to read owner_ and reentrant_depth_ without synchronizaton
  int owner_;
  // The number of parent graph tasks for this graph task
  const int reentrant_depth_;

  bool can_checkpoint() const {
    return exec_info_.empty();
  }

  // check if the GraphTask is completed or not
  bool completed();
  // mark the graph task as completed and trigger post processing
  void mark_as_completed_and_run_post_processing();

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function.
  void set_exception(std::exception_ptr eptr, const std::shared_ptr<Node>& fn);

  // Set an appropriate exception on this graph_task which was encountered while
  // running the provided function. But doesn't signal completion on
  // 'future_result_' right away. The user needs to explicitly mark
  // 'future_result_' completed with an appropriate exception.
  void set_exception_without_signal(const std::shared_ptr<Node>& fn);

  // Whether or not to stop execution for this GraphTask when an error is
  // encountered. When set to true, this would cause Engine::execute() to throw
  // an exception as soon as the autograd engine receives an exception.
  bool exit_on_error_;

  // CPU threads are dedicated to processing CPU work for the backward they
  // invoked. So any given graph task maintains its own cpu_ready_queue_ where
  // you should send work for it to be done. We memoize the cpu_ready_queue_ per
  // GraphTask so that we know which ready queue we should push to if we are on
  // device thread (i.e. GPU) and but next NodeTask should be run on CPU.
  /*这就是 CPU queue。我们把 CPU queue 和工作线程的queue做比较。
    设备 ReadyQueue 的数目 与 worker线程数目相同，每个worker有一个对应的 ReadyQueue。
    CUDA、XLA线程在所有反向传播调用之间通过 device_ready_queues_进行信息共享。
    而CPU线程专用于处理它们调用的反向传播相关CPU工作。
    因此，任何给定的graph任务都会维护自己的cpu_ready_queue_，
    用户应该向其发送要完成的工作。
    CPU queue 就是GraphTask 的成员变量 cpu_ready_queue_。

    注意，CPU就绪队列为每个GraphTask独有，但CUDA设备就绪队列在所有GraphTask中共享。

所以，引擎之中就绪队列数目是：设备数目 + GraphTask 数目。
    */
  std::shared_ptr<ReadyQueue> cpu_ready_queue_;

  // Future representing the completion of the graph task. Notified when all
  // tasks are done.
  c10::intrusive_ptr<at::ivalue::Future> future_result_;

  // Final callbacks installed during execution of this GraphTask
  std::vector<std::function<void()>> final_callbacks_;
  // To protect reads and writes to final_callbacks_. Intentionally no reusing
  // mutex_ as the two are protecting different data structures.
  std::mutex final_callbacks_lock_;

  utils::DelayWarningHandler warning_handler_;

  uint64_t id_;
//在构建函数内部，就把 local_ready_queue 赋值给了 GraphTask 的成员变量 cpu_ready_queue_。
  GraphTask(
      bool keep_graph,
      bool grad_mode,
      int reentrant_depth,
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      c10::SmallVector<Node*, 4> graph_roots,
      bool exit_on_error = false)
      : keep_graph_(keep_graph),
        graph_roots_(std::move(graph_roots)),
        owner_(NO_DEVICE),
        reentrant_depth_(reentrant_depth),
        exit_on_error_(exit_on_error),
        cpu_ready_queue_(std::move(cpu_ready_queue)),
        future_result_(c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()))),
        id_(graph_task_id.fetch_add(1, std::memory_order_relaxed)) {
    thread_locals_.set_grad_mode(grad_mode);
  }

 private:
  // run GraphTask post processing
  void exec_post_processing();
};

// The guard that sets and restores current_graph_task.
class GraphTaskGuard {
 public:
  explicit GraphTaskGuard(std::shared_ptr<GraphTask> graph_task);
  ~GraphTaskGuard();

  void restore_current_graph_task();

 private:
  std::shared_ptr<GraphTask> last_graph_task_;
};

TORCH_API const std::unordered_map<Node*, GraphTask::ExecInfo>*
get_current_graph_task_exec_info();
TORCH_API const std::unordered_set<Node*>*
get_current_graph_task_nodes_in_graph();
TORCH_API bool get_current_graph_task_keep_graph();
TORCH_API std::vector<Node*> get_current_graph_task_execution_order();
TORCH_API int get_current_graph_task_id();
void add_node_to_current_graph_task_exec_info(Node* fn);

} // namespace autograd
} // namespace torch
