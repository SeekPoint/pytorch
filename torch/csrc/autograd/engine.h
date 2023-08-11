#pragma once

// Engine implements backpropagation from output variables and their gradients
// to "root" variables (variables created by the user with requires_grad=True).

#include <ATen/Tensor.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/autograd/utils/warnings.h>

#include <c10/util/CallOnce.h>

#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace autograd {
struct ReadyQueue;
}
} // namespace torch

namespace torch {
namespace autograd {

// Maximum reentrant backward depth before switching to a new thread
// This limit is based on the TSAN's deadlock detector, where it will
// fail if a program hold more than 65 locks in one thread at once.
// As we hold mutex in every of our custom C++ autograd Node, we would
// like to avoid TSAN complains on this when doing reentrant backwards
// For reference, see https://github.com/google/sanitizers/issues/950
static constexpr int MAX_DEPTH = 60;

void set_device(int device);
void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error);
/*
对于NodeTask，我们有一个疑问：为什么要再增加一个新类型？而不是继续使用 GraphTask。

因为 GraphTask 只是包括本计算图的总体信息，但是具体某一个节点如何计算梯度，GraphTask 是不知道的，
所以引入了一个新类型 NodeTask 来处理。NodeTask 这个类的对象正是在queue中传输的东西，就是一个可以被执行的求导函数。
从下面的定义可以看到，我们使用GraphTask、Node、InputBuffer来构建一个NodeTask实例，
可以认为，生产者不停的向 ReadyQueue 插入 NodeTask，消费者则从 ReadyQueue 之中提取 NodeTask 进行处理。
*/
struct NodeTask {
  std::weak_ptr<GraphTask> base_; // 所属的GraphTask
  std::shared_ptr<Node> fn_; // 需要执行的Node，比如 PowBackward0
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  InputBuffer inputs_; // fn_的输入
  // When worker receives a task with isShutdownTask = true, it will immediately
  // exit. The engine sends a shutdown task to every queue upon its destruction.
  bool isShutdownTask_;

  int getReentrantDepth() const;

  NodeTask(
      std::weak_ptr<GraphTask> base,
      std::shared_ptr<Node> fn,
      InputBuffer inputs,
      bool isShutdownTask = false)
      : base_(std::move(base)),
        fn_(std::move(fn)),
        inputs_(std::move(inputs)),
        isShutdownTask_(isShutdownTask) {}
};

// Guard that sets and restores checkpoint_valid
class CheckpointValidGuard {
 public:
  explicit CheckpointValidGuard(
      const std::shared_ptr<const GraphTask>& graph_task);
  ~CheckpointValidGuard();

 private:
  bool prev_checkpoint_valid_state;
};
/*
ReadyQueue 用来在主线程和worker线程之间、以及worker线程和worker线程之间传输任务（NodeTask对象）。
为什么要传递 NodeTask？是因为NodeTask 包含了求导函数，
逐一运行NodeTask 就是在反向计算图路径上逐一运行求导函数，
最后往输出节点输出最终梯度。
ReadyQueue就指定了worker线程要执行的工作流。
*/
struct ReadyQueue {
 private:
  // Returns true when t2 should be (weakly) BEFORE t1 in the queue.
  // Shutdown tasks are first and then empty NodeTask are next.
  struct CompareNodeTaskTime {
    bool operator()(NodeTask const& t1, NodeTask const& t2) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (t2.isShutdownTask_) {
        return true;
      } else if (!t1.fn_ || t1.isShutdownTask_) {
        return false;
      } else if (!t2.fn_) {
        return true;
      } else if (t1.getReentrantDepth() == t2.getReentrantDepth()) {
        return t1.fn_->sequence_nr() < t2.fn_->sequence_nr();
      } else {
        return t1.getReentrantDepth() < t2.getReentrantDepth();
      }
    }
  };

  // To notify threads waiting on the ReadyQueue of available tasks on the heap_
  std::condition_variable not_empty_;
  // To protect read and writes to heap_
  mutable std::mutex mutex_;

  std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime>
      heap_;

 public:
  // incrementOutstandingTasks indicates whether or not we should increment
  // 'outstanding_tasks_' for the associated GraphTask. This should mostly
  // always be true and is only set false in certain cases (see docs for
  // DistEngine.execute_graph_task_until_ready_queue_empty)
  /*
  ReadyQueue 主要成员函数/成员变量如下:
        std::condition_variable not_empty_ 其作用是在线程之间同步。
        Push 是生成者行为，使用 not_empty_.notify_one() 来通知消费者，这样就可以解锁一个消费者。
        Pop 是消费者行为，使用 not_empty_.wait(lock, [this]{ return !heap_.empty(); }) 来阻塞等待生产。
        std::priority_queue heap_，使用 CompareNodeTaskTime 来做比较。
        每次 pop 时会取出 CompareNodeTaskTime 最小的 NodeTask。
        CompareNodeTaskTime 依据 ReentrantDepth 和 sequence_nr 做比较，哪一个小就消费哪一个。
        因此消费的顺序不等同于生产的顺序，这里生产的意思是往queue之中插入NodeTask。
    */
  void push(NodeTask item, bool incrementOutstandingTasks = true);
  void pushShutdownTask();
  NodeTask pop();
  bool empty() const;
  size_t size() const;
};

// A single instance of this struct should be created through the whole process
// lifetime. The worker thread creation logic and Engine's destructor rely on
// this.
//Engine 是autograd的核心，其实现了后向传播。后向传播方向是从根节点（就是正向传播的输出）到输出（就是正向传播的输入），
//在后向传播过程之中依据前向传播过程中设置的依赖关系生成了动态计算图。
//Engine 入口 是execute函数，其逻辑如下：
//根据根节点 roots 构建GraphRoot。
//根据 roots 之中的Node实例 metadata 以及各层之间的关系来构建计算图。
//通过next_edge不断的找到指向的下一个Edge，最终完成整个计算图的计算。
//利用 Queue 来多线程完成反向计算的工作。
//引擎定义在：torch/csrc/autograd/engine.cpp，这里只给出成员变量，最主要的变量是：
//
//device_ready_queues_ ：ReadyQueue 列表 device_ready_queues_ 之中的每一个ReadyQueue都启动了一个工作线程。各个线程之间通过 device_ready_queues_ 来进行交互。注意，因为CPU线程会处理其调用的反向传播的CPU相关工作，所以每个 GraphTask 拥有自己的 cpu_ready_queue_，用户可以向这些 cpu_ready_queue_ 发送待处理的工作。
//thread_pool_shared_ ：线程池，用来多线程处理后向传播。
struct TORCH_API Engine {
  /// Returns a reference to a static `Engine` instance.
  static Engine& get_default_engine();

  static Engine& get_base_engine();

  Engine(const Engine&) = delete;
  Engine(Engine&&) = delete;
  virtual ~Engine();

  // Given a list of (Node, input number) pairs computes the value of the graph
  // by following next_edge references.
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      bool accumulate_grad,
      const edge_list& outputs = {});

  // Given a pre-populated GraphTask and GraphRoot, computes the backward pass
  // for the graph.
  //
  // NB: This API should only be used by internal autograd specific
  // machinery and shouldn't be exposed to users in anyway.
  virtual c10::intrusive_ptr<at::ivalue::Future> execute_with_graph_task(
      const std::shared_ptr<GraphTask>& graph_task,
      std::shared_ptr<Node> graph_root,
      InputBuffer&& input_buffer);

  virtual std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() {
    return std::make_unique<AnomalyMetadata>();
  }

  virtual std::unique_ptr<SavedVariableHooks> get_default_saved_variable_hooks() {
    return nullptr;
  }

  // We pass cpu_ready_queue to evaluate_function, so that it knows
  // the correct ready queue to push to after a NodeTask is ready
  void evaluate_function(
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs,
      const std::shared_ptr<ReadyQueue>& cpu_ready_queue);

  void initialize_device_threads_pool();
  virtual void thread_on_exception(
      std::shared_ptr<GraphTask> graph_task,
      const std::shared_ptr<Node>& fn,
      std::exception& e);

  void queue_callback(std::function<void()> callback);

  bool is_checkpoint_valid();

  // Should be called after fork to notify that worker threads are gone
  void release_workers();

  // Must be called by subclass before destructing to avoid a data-race-on-vptr.
  void stop();

  // Initializes a device thread for the autograd engine.
  virtual void thread_init(
      int device,
      const std::shared_ptr<ReadyQueue>& ready_queue,
      bool should_increment = true);

 protected:
  Engine();
  void compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr);

  // initialize the thread local ready queue with the ready queue that is
  // created elsewhere (i.e. thread_init, Engine::execute, etc), or create a new
  // ready queue if ready_queue is not provided.
  void init_local_ready_queue(
      std::shared_ptr<ReadyQueue> ready_queue = nullptr);

  std::shared_ptr<ReadyQueue> ready_queue(
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      at::Device device);
  std::shared_ptr<ReadyQueue> ready_queue_by_index(
      std::shared_ptr<ReadyQueue> cpu_ready_queue,
      int device_index);
  // start device threads (CUDA, XLA, etc.) in Engine,
  // note that it does NOT start CPU thread.
  void start_device_threads();
  void increment_non_reentrant_thread_count();
  void decrement_non_reentrant_thread_count();
  virtual void thread_main(const std::shared_ptr<GraphTask>& task);
  void reentrant_thread_init();
  void add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task);

  // Ensures device_ready_queues_ are initialized only once
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  c10::once_flag start_device_threads_flag_;
  // Safe to read device_ready_queues_ without synchronization after
  // initialization
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  //在引擎之中，线程数量和ReadyQueue 的数量是由据设备的数量来决定的。
  //有多少个设备，就启动多少个工作线程，也生成与线程一一对应的ReadyQueue。
  //所以，引擎有如下成员变量，使用 vector 来统一管理 queue。
//  在引擎之中，工作线程的数目是依据设备数量决定的。如果有n个设备，就会启动n个设备工作线程。比如，如果有2个GPU，则启动2个设备工作线程。但是每一个GraphTask都有自己的CPU工作线程（我们接下来马上介绍）。
//
//GPU工作线程对应的 ReadyTask 是 Engine 之中的 成员变量。
  std::vector<std::shared_ptr<ReadyQueue>> device_ready_queues_;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::function<void()>> final_callbacks_;
  // To protect reads and writes to final_callbacks_
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex post_callbacks_lock_;

  // How many nested reentrant calls are allowed until a new thread is used
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  int max_recursion_depth_;

//可重入向后 相关数据结构如下：
  struct ThreadPoolShared {
    // Data structures used by the threads for executing reentrant backwards
    // tasks. See Note [Reentrant backwards]
    // Number of available threads for processing new GraphTasks.
    unsigned int num_workers_{0};
    // The threads will wait on work_ to be notified of GraphTasks
    std::condition_variable work_;
    // To protect reads and writes to graphtask_queue_ and num_workers_
    // and for synchronizing creating new threads when needed
    std::mutex mutex_;
    // Workers will process the GraphTasks added to this queue. A GraphTask is
    // allocated inside Engine::execute and lives for the duration of execute
    std::queue<std::weak_ptr<GraphTask>> graphtasks_queue_;

    ThreadPoolShared() = default;
  };

  // Temporary workaround until shutting down threads is done
  // We need shared ownership of all these objects because the threads are
  // leaked when Engine shuts down, so there may be threads waiting on work_ for
  // the graphtasks_queue_ to be nonempty.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<ThreadPoolShared> thread_pool_shared_;

 private:
  // Number of non-reentrant threads
  std::atomic<uint32_t> non_reentrant_device_thread_count_;
  // Destructor will wait for non-reentrant threads to finish
  std::condition_variable non_reentrant_device_thread_condvar_;
  std::mutex non_reentrant_device_thread_mutex_;
  // stop() must be called before the destruction path goes down to the base
  // class, in order to avoid a data-race-on-vptr. Use this boolean to guard
  // whether stop() has already been called, so we can call this in every
  // destructor of the class hierarchy.
  bool stopped_{false};
};

// allow python_engine to override the default engine when it loads
using EngineStub = Engine& (*)();
TORCH_API void set_default_engine_stub(EngineStub stub);

} // namespace autograd
} // namespace torch
