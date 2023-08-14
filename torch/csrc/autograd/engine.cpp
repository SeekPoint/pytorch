#include <torch/csrc/autograd/engine.h>

#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/memory.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/isnan.h>
#endif

#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/ThreadLocal.h>
#include <c10/util/irange.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_set>
#include <utility>

namespace torch {
namespace autograd {

namespace {
static bool in_bad_autograd_fork =
    false; // True for children forked after engine's thread pool init

// Called in the forked child if engine's thread pool has already been
// initialized
static void forked_autograd_child() {
  in_bad_autograd_fork = true;
}

// Should be called before unsafe for forks (thread pool) calls
static void track_bad_autograd_forks() {
#if !defined(WIN32)
  static c10::once_flag flag;
  c10::call_once(
      flag, [&] { pthread_atfork(nullptr, nullptr, forked_autograd_child); });
#endif
}

inline bool should_run_in_cpu_ready_queue(c10::DeviceType device) {
  if (device == c10::kCPU || device == c10::kMeta || device == c10::kLazy) {
    return true;
  } else {
    return false;
  }
}
} // namespace

// Threads spawned by the engine are assigned a 'worker_device' specifying
// what device they process work for. This variable is initialized at:
// 1. thread creation time for CUDA, XLA device threads, as they are
//    spinning threads waiting for works on their device.
// 2. before the graph task execution for CPU threads, as for each
//    backward call we use the caller thread to drive engine execution.
// This is used when handling reentrant backwards calls;
// See Note [Reentrant backwards]
static thread_local int worker_device = NO_DEVICE;

// This variable is true if ALL invocations in the stack of re-entrant engine
// invocations are imperative backwards. This special variable is needed for the
// gradient checkpointing feature only.
static thread_local bool checkpoint_valid = true;

// Number of nested reentrant backwards calls currently on this thread
static thread_local int current_depth = 0;

// For all device threads (i.e. CUDA, XLA), total_depth represents the total
// nested
//   reentrant backwards depths over all device threads.
// For CPU devices, it is the total depth associated with the original backward
// call.
static thread_local int total_depth = 0;

// The current GraphTask being executed by this thread. This helps
// queue_callback() to find the target GraphTask to append final callbacks.
C10_DEFINE_TLS_static(std::shared_ptr<GraphTask>, tls_current_graph_task);
#define current_graph_task (tls_current_graph_task.get())

// Every autograd worker thread is associated with a ready queue, which
// specifies the stream of work of this thread to do. This shared_ptr is a
// thread_local pointer to each thread's ready_queue, and it should be
// initialized via the Engine::init_local_ready_queue() call in each
// corresponding thread before execution.
//
// The CUDA, XLA threads are shared among all invocations of backwards via
// device_ready_queues_, while the caller thread is dedicated to processing work
// for devices returning true in should_run_in_cpu_ready_queue (most notably the
// CPU device). So any given graph task maintains its own cpu_ready_queue_ where
// you should send work for it to be done.
//
// For reentrant backward calls, if we spawn new thread from the current thread
// because we reached the maximum depth, the new thread will just reuse the same
// ReadyQueue with the parent thread for performance improvement.
// see Note [Reentrant backwards] for more details.
C10_DEFINE_TLS_static(std::shared_ptr<ReadyQueue>, tls_local_ready_queue);
//每个autogard 工作线程都与一个就绪队列相关联，该队列指定该线程要执行的工作流，这个队列定义如下。
//local_ready_queue 使用 std::shared_ptr 来作为 本地线程指针。
#define local_ready_queue (tls_local_ready_queue.get())

// Note [Reentrant backwards]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// To understand the reentrant backwards problem, we have to notice two
// aspects of how the autograd engine is implemented today:
//
//  1. When you call Engine::execute(), you want to block until
//  differentiation finishes so that you can get the final result variables
//  of the backwards pass.
//
//  2. The engine operates by having a single worker thread per work queue,
//  and every work queue is pinned to a specific device where the
//  operation is executed.
//
// The problem is, suppose that you call backward() inside of a worker
// thread.  By property (1), we're supposed to block until the nested task
// finishes.  However, by property (2), this worker thread is on the
// hook for processing the tasks assigned to it; we better not block,
// because then all of our backward executions (including the one we
// just started) will deadlock!
//
// We maintain a pool of threads waiting for work to do
// When a reentrant backwards call occurs, the current thread blocks
// and a thread from the pool is woken up to complete the blocking tasks and an
// any other tasks that would have been assigned to that worker. If there are no
// threads available, a new thread is spawned. The new thread will continue
// processing tasks from the same ReadyQueue as the parent worker
//
// When the GraphTask is finished, the parent worker thread that is waiting on
// the task is notified and the current thread returns to the pool.

// Note [Streaming backwards]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// On CUDA devices the autograd engine's device operations are run on the
// same stream that ran them in forward. This requires automatically
// syncing the streams so that function A finishes producing its
// output before function B consumes it.
//
// This synchronization occurs when outputs are placed into input buffers.
// The functions corresponding to input buffer positions have metadata
// recording their streams from forward, and during backward this
// data is used to sync the producer's stream with the consumer's.
//
// When a CUDA function is run either all its inputs were accumulated on the
// stream used to run the function OR the inputs are on different devices
// and the function is responsible for properly acquiring them.
//
// User-facing stream semantics of a backward() (or torch.autograd.grad())
// call with respect to surrounding ops are the same as for any other call.
// See "Stream semantics of backward passes" on
// https://pytorch.org/docs/stable/notes/cuda.html
//
// Internally, backward() runs ops (including leaf nodes) on side threads.
// And streams are thread local. So GraphTask achieves the above semantics by
//  1. remembering the current streams on all active CUDA devices
//     in the user-facing thread (aka, the thread that called execute() to
//     launch the GraphTask)
//  2. remembering the "leaf streams" (streams each backward leaf node ran on)
//  3. during exec_post_processing, for each leaf stream, sync the remembered
//     current streams (on the leaf stream's device) with that
//     leaf stream.

int NodeTask::getReentrantDepth() const {
  std::shared_ptr<GraphTask> graph_task = base_.lock();
  if (graph_task) {
    return graph_task->reentrant_depth_;
  } else {
    // The graph task is no longer valid indicating an error. As a result, we
    // try to move this to the front of the queue to ensure the autograd
    // engine threads pick up this error soon.
    return std::numeric_limits<int>::max();
  }
}

CheckpointValidGuard::CheckpointValidGuard(
    const std::shared_ptr<const GraphTask>& graph_task) {
  prev_checkpoint_valid_state = checkpoint_valid;
  checkpoint_valid =
      graph_task->can_checkpoint() && prev_checkpoint_valid_state;
}

CheckpointValidGuard::~CheckpointValidGuard() {
  checkpoint_valid = prev_checkpoint_valid_state;
}

//NodeTask任务增加时 outstanding_tasks_ 就加一。
//即，往某一个 ReadyQueue 之中插入一个 NodeTask 时候， NodeTask 对应的GraphTask 就会把其 outstanding_tasks_ 增加一。
auto ReadyQueue::push(NodeTask item, bool incrementOutstandingTasks) -> void {
  {
    // Lock mutex for writing to heap_
    std::lock_guard<std::mutex> lock(mutex_);
    if (incrementOutstandingTasks) {
      std::shared_ptr<GraphTask> graph_task = item.base_.lock();
      TORCH_INTERNAL_ASSERT(graph_task, "GraphTask is no longer valid!");
      ++graph_task->outstanding_tasks_; // 增加
    }
    heap_.push(std::move(item));
  }
  not_empty_.notify_one();
}

auto ReadyQueue::pushShutdownTask() -> void {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    heap_.push(NodeTask({}, nullptr, InputBuffer(0), true));
  }
  not_empty_.notify_one();
}

size_t ReadyQueue::size() const {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  return heap_.size();
}

auto ReadyQueue::pop() -> NodeTask {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [this] { return !heap_.empty(); });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto task = std::move(const_cast<NodeTask&>(heap_.top()));
  heap_.pop();
  return task;
}

bool ReadyQueue::empty() const {
  // Lock mutex for accesses to heap_
  std::unique_lock<std::mutex> lock(mutex_);
  return heap_.empty();
}

Engine::Engine()
    : max_recursion_depth_(MAX_DEPTH), non_reentrant_device_thread_count_(0) {}

Engine::~Engine() {
  stop();
}

// Send shutdown tasks to all device_ready_queues_ if no backward tasks are
// running Even though readyQueue should be empty, shutdown tasks have the
// highest priority
void Engine::stop() {
  if (stopped_) {
    return;
  }
  stopped_ = true;
  // Under some conditions, autograd threads can hang on shutdown
  // Do not wait for them to shutdown indefinitely but rely on timeout
  auto wait_duration_str = getenv("TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT");
  auto wait_duration = wait_duration_str ? std::atof(wait_duration_str) : 10.0;
  bool noBackward = true;
  for (auto& queue : device_ready_queues_) {
    noBackward = noBackward && queue->empty();
  }
  if (noBackward && wait_duration > 0.0f) {
    for (auto& queue : device_ready_queues_) {
      queue->pushShutdownTask();
    }
    // Do not wait for termination of global threads on Windows
    // Because CRT terminates DLL threads before calling
    // global object destructors
#if !defined(_WIN32) || defined(C10_USE_MSVC_STATIC_RUNTIME)

    using namespace std::chrono_literals;
    // Set a deadline for how long it is OK to wait device threads to shutdown
    auto wait_deadline =
        std::chrono::steady_clock::now() + wait_duration * 1.0s;
    std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
    while (non_reentrant_device_thread_count_.load() != 0) {
      if (non_reentrant_device_thread_condvar_.wait_until(lk, wait_deadline) ==
          std::cv_status::timeout) {
        break;
      }
    }
#endif
  }
  // Otherwise threads are leaked
}

void Engine::release_workers() {
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  non_reentrant_device_thread_count_.store(0);
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::increment_non_reentrant_thread_count() {
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  non_reentrant_device_thread_count_.fetch_add(1);
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::decrement_non_reentrant_thread_count() {
  std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
  non_reentrant_device_thread_count_.fetch_sub(1);
  non_reentrant_device_thread_condvar_.notify_one();
}

void Engine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  if (should_increment) {
    increment_non_reentrant_thread_count();
  }

  at::init_num_threads();

  // Note [Allocating GPUs to autograd threads]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // What's our strategy here?  Originally, the autograd engine was written
  // with only CUDA in mind.  We allocate one thread to handle all CPU
  // operations, and a thread per CUDA device.
  //
  // But what if we have OTHER devices?  There are two plausible
  // strategies:
  //
  //  - We can allocate threads equal to max(num_cuda_devices, num_xla_devices,
  //    ...) and colocate cuda device 0 with xla device 0
  //  - We can allocate threads equal to sum(num_cuda_devices, num_xla_devices,
  //    ...) keeping everyone separate.
  //
  // We don't have any good reason to prefer one or the other, so we've
  // arbitrarily picked to colocate devices.  Maybe the other approach is
  // better.

#if defined(USE_CUDA)
  if (at::detail::getCUDAHooks().hasPrimaryContext(device)) {
    set_device(device);
  }
#else
  set_device(device);
#endif

  // initialize each device thread's thread local ready queue with the ready
  // queue that is created before the thread initialization
  init_local_ready_queue(ready_queue);

  std::shared_ptr<GraphTask> graph_task = nullptr;
  thread_main(graph_task);
  if (should_increment) {
    // Decrement the count during shutdown if we incremented earlier.
    decrement_non_reentrant_thread_count();
  }
}

GraphTaskGuard::GraphTaskGuard(std::shared_ptr<GraphTask> graph_task) {
  last_graph_task_ = std::move(current_graph_task);
  current_graph_task = std::move(graph_task);
}
GraphTaskGuard::~GraphTaskGuard() {
  restore_current_graph_task();
}

void GraphTaskGuard::restore_current_graph_task() {
  current_graph_task = std::move(last_graph_task_);
}

// The current graph task's exec_info is being used to trim unnecessary edegs
// during node evaluation, see `Node.task_should_compute_output()` function.
const std::unordered_map<Node*, GraphTask::ExecInfo>*
get_current_graph_task_exec_info() {
  return current_graph_task ? &current_graph_task->exec_info_ : nullptr;
}

const std::unordered_set<Node*>* get_current_graph_task_nodes_in_graph() {
  return current_graph_task ? &current_graph_task->nodes_in_graph_ : nullptr;
}

int get_current_graph_task_id() {
  return current_graph_task ? current_graph_task->id_ : -1;
}

bool get_current_graph_task_keep_graph() {
  return current_graph_task ? current_graph_task->keep_graph_ : true;
}

void add_node_to_current_graph_task_exec_info(Node* fn) {
  current_graph_task->exec_info_[fn].needed_ = true;
}

// NB: The engine itself does not use the outputs of this function.
std::vector<Node*> get_current_graph_task_execution_order() {
  std::shared_ptr<GraphTask> task = current_graph_task;
  if (!task) {
    return {};
  }

  // We could potentially check if there is only a single device here
  // but explicitly require this context doens't seem bad either
  TORCH_CHECK(
      !c10::AutogradState::get_tls_state().get_multithreading_enabled(),
      "get_current_graph_task_execution_order expects the current backward to be "
      "executed with multithreading disabled, e.g. by running:\n\n"
      ">>> with torch.autograd.set_multithreading_enabled(False):\n"
      "...     torch.autograd.grad(...)\n");

  const bool check_exec_info = !task->exec_info_.empty();
  std::vector<Node*> out{};
  std::unordered_set<Node*> seen{};

  auto compare_seq_nr = [](Node* n1, Node* n2) {
    return n1->sequence_nr() < n2->sequence_nr();
  };
  std::priority_queue<Node*, std::vector<Node*>, decltype(compare_seq_nr)> heap(
      compare_seq_nr);

  for (Node* ptr : task->graph_roots_) {
    heap.push(ptr);
  }

  // Implementation notes:
  // - Don't need to count dependencies because we have sequence_nr
  // - Don't need to check topological_nr because we have exec_info
  while (!heap.empty()) {
    Node* fn = heap.top();
    heap.pop();

    const bool was_inserted = seen.insert(fn).second;
    if (!was_inserted) {
      continue;
    }

    out.push_back(fn);
    for (const auto& edge : fn->next_edges()) {
      Node* next_ptr = edge.function.get();
      if (!next_ptr) {
        continue;
      }
      if (check_exec_info) {
        auto it = task->exec_info_.find(next_ptr);
        if (it == task->exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
      }
      heap.push(next_ptr);
    }
  }
  return out;
}


/*
前文中我们介绍了反向传播引擎的动态逻辑，因为具体反向传播算法是在设备线程中完成的，所以我们单独用一章来讲解。
0x01 工作线程主体
thread_main是工作线程的主体函数，主要逻辑就是围绕着 ReadyQueue 执行一个 while 循环，
工作线程阻塞在 ReadyQueue -> pop 这里，如果主线程或者其他线程插入了一个 NodeTask，
则 pop 会返回取出一个 NodeTask，工作线程处理这个 NodeTask，完成后向计算的一个环节，
如果有需要就继续往某一ReadyQueue插入新的 NodeTask，驱动引擎继续执行后向计算其他环节。
thread_main 从如下途径被调用：
    CUDA, XLA 设备的 autograd threads 会调用。
    CPU 之上的反向传播主线程会调用。
    前两个case 进行可重入反向传播，也会调用。
1.1 线程主体代码
工作线程的计算始于动态图的GraphRoot函数，反向传播就以 Node 的edge为纽带，层层从前向后计算，直到来到了leaf节点，最终完成了反向计算，具体如下：

local_graph_task表示我们从队列中检索的graph_task。外部graph_ 任务表示我们需要执行的可重入执行的总体 graph_任务。
从自己的ReadyQueue之中取出NodeTask实例，使用 local_graph_task 为参数来执行evaluate_function（反向传播函数）。
outstanding_tasks 自减 1。
如果本 local_graph_task 已经结束（可重入反向传播会运行多个 GraphTask），即：
执行后续操作 exec_post_processing，然后使用 future_result_->markCompleted。
如果这个task是来自其它worker thread，即 worker_device != base_owner，则向那个worker thread的queue发送一个dummy function task，让那个工作线程也执行起来。
*/
// NOTE: graph_tasks do not necessarily form a stack. Imagine this
// case:
//
//    +----> Eval1
//  Root
//    +----> Eval2
//
// Once Root is executed, both Eval1 and Eval2 are added to the ready queue.
// Next, Eval1 is run and this causes the worker to enter thread_main again.
// Then, it pops the next task from the queue, but at this point it is Eval2.
// It enters thread_main once again, but now with graph_task of Eval2, which is
// completely unrelated to that of Eval1 (it's not a recursive call).
// It's all ok and is handled right now, but it should be accounted for
// in case this code is to be changed.
//
// thread_main is used by:
// 1). autograd threads for devices (i.e. CUDA, XLA)
// 2). the caller/owning thread of the backward call on CPU (sync mode)
// 3). Renetrant backward that invoked by either 1) or 2)
// The exit conditions are different for the above three cases.
// For 1), we are spinning on running the thread_main on device autograd
//         threads throughout the Engine lifetime, thread_main will get
//         terminated during Engine destruction by pushing shutdown tasks
// For 2), the owning thread of the backward call drives the thread_main
//         synchronously until the graph_task of that owning thread is
//         completed and exit the thread_main to continue executing the
//         result of caller's code.
// For 3), the reentrant backward that invokes
//         thread_main, either from 1) or 2), will not spin and will exit as
//         long as graph_task is completed and notify the owning thread as
//         needed.
//NodeTask 任务结束时候就减一，我们用简化代码看看。
auto Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) -> void { //运行 GraphTask
  // When graph_task is nullptr, this is a long running thread that processes
  // tasks (ex: device threads). When graph_task is non-null (ex: reentrant
  // backwards, user thread), this function is expected to exit once that
  // graph_task complete.

  // local_ready_queue should already been initialized when we get into
  // thread_main
  TORCH_INTERNAL_ASSERT(local_ready_queue != nullptr);
  while (graph_task == nullptr || !graph_task->future_result_->completed()) {
    // local_graph_task represents the graph_task we retrieve from the queue.
    // The outer graph_task represents the overall graph_task we need to execute
    // for reentrant execution.
    std::shared_ptr<GraphTask> local_graph_task;
    {
      // Scope this block of execution since NodeTask is not needed after this
      // block and can be deallocated (release any references to grad tensors
      // as part of inputs_).
      /*
        工作线程之中如何消费 NodeTask
        当worker线程刚被创建出来的时候，该线程被阻塞在queue->pop()，就是等待生产者往这个queue里插入一个task。当主线程往 ReadyQueue 发送了 NodeTask 实例之后，消费端的工作线程在 thread_main 的 pop 结束阻塞被唤醒。
        于是worker线程获取 到了NodeTask。worker线程 然后：
            通过task.base来访问到这个GraphTask实例。
            通过 task.fn_ 访问到这个roots实例，也就是该NodeTask需要执行的后向计算方法，比如 MulBackward0。
            通过task.inputs_ 来访问这个InputBuffer实例，就是 MulBackward0 的输入。
            后把NodeTask 的 fn_, inputs 传给evaluate_function。进行反向计算。
            具体代码如下：
       */
      NodeTask task = local_ready_queue->pop(); // 阻塞等待
      // This will only work if the worker is running a non backward task
      // TODO Needs to be fixed this to work in all cases
      if (task.isShutdownTask_) {
        C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
        break;
      }

      if (!(local_graph_task = task.base_.lock())) {
        // GraphTask for function is no longer valid, skipping further
        // execution.
        continue;
      }

      if (task.fn_ && !local_graph_task->has_error_.load()) {
        // Set the ThreadLocalState before calling the function.
        // NB: The ThreadLocalStateGuard doesn't set the grad_mode because
        // GraphTask always saves ThreadLocalState without grad_mode.
        //利用grad_mode_来配置AutoGradMode，整个反向计算期间的代码都靠GradMode::is_enabled()来判断当前是否是要计算grad
        at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);
        c10::WarningUtils::WarningHandlerGuard warnings_guard(
            &local_graph_task->warning_handler_);

        try {
          // The guard sets the thread_local current_graph_task on construction
          // and restores it on exit. The current_graph_task variable helps
          // queue_callback() to find the target GraphTask to append final
          // callbacks.
          GraphTaskGuard guard(local_graph_task);
          NodeGuard ndguard(task.fn_);
          {
            RECORD_FUNCTION(
                c10::str(
                    "autograd::engine::evaluate_function: ",
                    task.fn_.get()->name()),
                c10::ArrayRef<const c10::IValue>());
            // 运行 NodeTask  // 后向计算
            evaluate_function(
                local_graph_task,
                task.fn_.get(),
                task.inputs_,
                local_graph_task->cpu_ready_queue_);
          }
        } catch (std::exception& e) {
          thread_on_exception(local_graph_task, task.fn_, e);
        }
      }
    }

    // Decrement the outstanding tasks.
    --local_graph_task->outstanding_tasks_;  // 运行 NodeTask完毕，这里减一

    //在 thread_main 之中，有一个 work around。
    //就是：当前工作线程完成 graph_task，但此时，拥有graph_task的线程可能正在pop()上等待休眠。
    //因此，我们需要向所属线程发送一个仿造的函数任务，以唤醒它，这样我们可以退出thread_main。
    //在 thread_main 之中，如果本task已经结束，即做后续操作
    // Check if we've completed execution.
    if (local_graph_task->completed()) { // 判断 GraphTask是否结束。  // 已经结束了，进行后续处理   // 判断是否结束

      // 如果结束了，就进行后续操作
      local_graph_task->mark_as_completed_and_run_post_processing();

      auto base_owner = local_graph_task->owner_;  // 后续是需要在 GraphTask 的 owner_ 处理   // 当前设备
      // The current worker thread finish the graph_task, but the owning thread
      // of the graph_task might be sleeping on pop() if it does not have work.
      // So we need to send a dummy function task to the owning thread just to
      // ensure that it's not sleeping, so that we can exit the thread_main.
      // If it has work, it might see that graph_task->outstanding_tasks_ == 0
      // before it gets to the task, but it's a no-op anyway.
      //
      // NB: This is not necessary if the current thread is the owning thread.
      if (worker_device != base_owner) {  // 不是同一个设备
        // Synchronize outstanding_tasks_ with queue mutex
        std::atomic_thread_fence(std::memory_order_release);
        // 获取后续工作的queue
        ready_queue_by_index(local_graph_task->cpu_ready_queue_, base_owner)
            ->push(NodeTask(local_graph_task, nullptr, InputBuffer(0)));
      }
    }
    /*
    其他线程当收到了 dummy task 之后，不会处理，因为 function 是 nullptr，然后就调用 local_ready_queue->pop() 继续从自己的queue 中读取下一个 task。
    具体如下：
        主线程等待。
        如果工作线程发现GraphTask 已经结束，就通知主线程。
        如果需要唤醒其他线程，就向该线程对应的 queue 插入 NodeTask。
        对应线程取出 NodeTask 进行执行。
   */
  }
}

/*
新线程执行函数 reentrant_thread_init 如下：
    与graph_task's 原线程共享 cpu_ready_queue。
    其从 graphtasks_queue_ 获取 GraphTask，赋值给 graph_task。
    然后用 thread_main(graph_task) 来执行。
*/
// Reentrant call will re-use the graph_task's owner thread ready_queue for
// queueing tasks (NOTE: this is not true in the async_mode of the engine).
// While we can create separate ready queue for each new reentrant
// thread, but sharing the same cpu_ready_queue with parent thread is a
// performance improvement and cuda thread still have to do the same thing.
void Engine::reentrant_thread_init() {
  at::init_num_threads();
  auto tp_shared = thread_pool_shared_;
  while (true) {
    std::unique_lock<std::mutex> lk(tp_shared->mutex_);
    ++thread_pool_shared_->num_workers_;
    tp_shared->work_.wait(
        lk, [&tp_shared] { return !tp_shared->graphtasks_queue_.empty(); });
    --thread_pool_shared_->num_workers_;
    auto task = tp_shared->graphtasks_queue_.front();
    tp_shared->graphtasks_queue_.pop();
    lk.unlock();
    std::shared_ptr<GraphTask> graph_task;
    if (!(graph_task = task.lock())) {
      LOG(INFO) << "GraphTask has expired, skipping reentrant execution";
      continue;
    }
    set_device(graph_task->owner_);
    // set the local_ready_queue to the ready queue on the graph_task->owner_
    // device
    local_ready_queue =
        ready_queue_by_index(graph_task->cpu_ready_queue_, graph_task->owner_);
    total_depth = graph_task->reentrant_depth_;
    thread_main(graph_task); // 这里调用了线程函数
  }
}

void Engine::thread_on_exception(
    std::shared_ptr<GraphTask> graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  graph_task->set_exception(std::current_exception(), fn);
}

/*
以下代码用来判断本 GraphTask是否结束，其实就是 ReadyQueue 之中是否还有待运行的 NodeTask。
outstanding_tasks_ 是待处理 NodeTask的数量，用来判断该GrapTask是否还需要执行，其数值总是先加再减，如果数目为0，则说明任务结束了。
    当 GraphTask 被创建出来时候，此数值为0。
    如果有一个NodeTask被送入到 ReadyQueue，则outstanding_tasks_ 增加 1。
    如果在工作线程作执行一次 evaluate_function(task)后，outstanding_tasks的值减 1。
    如果这个数量不为0，则此GraphTask依然需要运行。
*/
bool GraphTask::completed() {
  // outstanding_tasks在evaluate_function中可能会被改变
  return outstanding_tasks_.load() == 0 ||
      (exit_on_error_ && has_error_.load());
}

//mark_as_completed_and_run_post_processing 就是进行后续处理。
//执行后续操作 exec_post_processing，然后使用 future_result_->markCompleted 通知主线程。
void GraphTask::mark_as_completed_and_run_post_processing() {
  // Allow only one thread one attempt to process this logic.
  if (future_completed_.exchange(true)) {
    // Future is already marked complete, or being marked as such.
    // In case the marking complete is only in progress, we add a
    // wait() to guarantee the future is marked complete on exit.
    future_result_->wait();
    return;
  }

  try {
    // Run post processing, before marking the future as complete.
    // Drop lock prior to completing, to avoid holding across callbacks.
    std::unique_lock<std::mutex> lock(mutex_);

    exec_post_processing(); // 进行后续操作
    std::vector<Variable> vars = std::move(captured_vars_);   //最后返回的输出

    // Need to unlock before we call markCompleted to avoid holding locks
    // when the callbacks are called.
    lock.unlock();

    //在 mark_as_completed_and_run_post_processing 会用如下代码来通知主线程。
    future_result_->markCompleted(std::move(vars)); // 反向传播最后的返回输出  // 通知主线程
  } catch (std::exception& e) {
    future_result_->setErrorIfNeeded(std::current_exception());
  }
}

//后续操作，如果之前有注册了 callback，则进行调用。也会进行流同步。
//对于 final_callbacks_ 处理，在 exec_post_processing 之中，就是当 engine 全部完成 backward 的时候会调用 callback。
/*
于是逻辑拓展如下：

Reduer 会注册autograd_hook到AccumulateGrad的post_hooks之上。
Autograd Engine 在反向传播过程中，如果发现某个参数ready，就调用autograd_hook。
autograd_hook 之中继续处理。
会注册一个 finalize_backward到 engine。
在 GraphTask::exec_post_processing 之中会调用 finalize_backward。
          Engine        AccumulateGrad                Reducer

            +                  +                         +
            |                  |                         |
            |                  |           1             |
            |                  | <-----------------------+
            |                  |
            |                  |
            |                  |
            |                  v
            |                              2
            |             post_hooks  +-------->  autograd_hook
            |                                            +
            |                                            |
            |                                            |  3
            |                                            v
            |                         +------------------+---------------------------+
            |                         | mark_variable_ready                          |
            |                         |                                              |
            |                         |                                              |
            |                         |     All variable in replica are ready?       |
            |                         |                   +                          |
            |                         |                   | YES                      |
            |                         |                   v                          |
            |                         |     All replica in bucket are ready?         |
            |                         |                   +                          |
            |                         |                   | YES                      |
            |                         |                   v                          |
            |                         |            mark_bucket_ready                 |
            |                         |                                              |
            |                         |                                              |
            |                         |                                              |
            |                         |                   +                          |
            |                         |                   |                          |
            |                         |                   |                          |
            |                         |                   v                          |
            |                         |          All buckets are ready?              |
            |                         |                   +                          |
            |                         |                   | YES                      |
            |                         |                   v                          |
            |   queue_back    4       |          all_reduce_local_used_map           |
            | <----------------------------+  queue_callback(finalize_backward)      |
            |                         |                                              |
            |                         |                                              |
            |                         +-------------------+--------------------------+
            v                                             |
                                                          |
GraphTask::exec_post_processing                           |
            +                                             |
            |                                             |
            |                 5                           v
            +--------------------------------->   finalize_backward
            |                                             +
            |                                             |
            |                                             |
            v                                             v


*/
void GraphTask::exec_post_processing() {
  if (!not_ready_.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }

  // set the thread_local current_graph_task_ as more callbacks can be installed
  // by existing final callbacks.
  GraphTaskGuard guard(shared_from_this());
  // Lock mutex during each iteration for accessing final_callbacks.size()
  // Unlocking is necessary, because the callback can register
  // more callbacks (or they can be registered from other threads
  // while it's waiting.
  std::unique_lock<std::mutex> cb_lock(final_callbacks_lock_);

  // caller_current_streams_ with nullopt entries removed
  std::vector<c10::Stream> caller_current_streams_filtered;

  // See Note [Streaming backwards].
  // Syncs caller_current_stream with leaf streams, so final_callbacks may use
  // any grad on its device's current stream.
  if (!leaf_streams.empty()) {
    for (const auto& leaf_stream : leaf_streams) {
      // stash_current_streams() stashed streams for all device IDs that already
      // had a CUDA context before the GraphTask executed. For inactive devices,
      // it stashed a c10::nullopt. I don't expect GraphTask's backward pass ran
      // leaf nodes on any new devices, so the stashed streams should be enough.
      // If leaf_stream.device_index() happens to be for a new device,
      // operator* on the c10::nullopt should throw an error.
      const auto caller_current_stream =
          *caller_current_streams_[leaf_stream.device_index()];

      if (caller_current_stream != leaf_stream) {
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(leaf_stream);
        caller_current_stream.wait(event);
      }
    }

    caller_current_streams_filtered.reserve(caller_current_streams_.size());
    for (const auto& opt_stream : caller_current_streams_) {
      if (opt_stream.has_value()) {
        caller_current_streams_filtered.push_back(*opt_stream);
      }
    }
  }

  {
    // final_callbacks run on the per-device caller_current_streams (the ambient
    // streams surrounding the user's call to backward()). This has two
    // benefits:
    //  1. caller_current_streams have been synced with leaf_streams, so
    //  callbacks may
    //     safely access any grad.
    //  2. The callback's results can safely be used on (user-facing)
    //  caller_current_streams
    //     after backward().
    c10::MultiStreamGuard g(caller_current_streams_filtered);

    // Set the ThreadLocalState before calling the function.
    // NB: The ThreadLocalStateGuard doesn't set the grad_mode because GraphTask
    // always saves ThreadLocalState without grad_mode.
    at::ThreadLocalStateGuard tls_guard(this->thread_locals_);

    // WARNING: Don't use a range-for loop here because more callbacks may be
    // added in between callback calls, so iterators may become invalidated.
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (size_t i = 0; i < final_callbacks_.size(); ++i) {
      cb_lock.unlock();
      final_callbacks_[i]();  // 调用了callback
      cb_lock.lock();
    }
  }
}

void GraphTask::set_exception_without_signal(const std::shared_ptr<Node>& fn) {
  if (!has_error_.exchange(true)) {
    if (AnomalyMode::is_enabled() && fn) {
      fn->metadata()->print_stack(fn->name());
    }
  }
}

void GraphTask::set_exception(
    std::exception_ptr eptr,
    const std::shared_ptr<Node>& fn) {
  set_exception_without_signal(fn);
  if (!future_completed_.exchange(true)) {
    future_result_->setError(std::move(eptr));
  }
}

static variable_list call_pre_hooks(Node& fn, variable_list inputs) {
  for (const auto& hook : fn.pre_hooks()) {
    inputs = (*hook)(inputs);
  }
  return inputs;
}

static variable_list call_tensor_pre_hooks(Node& fn, variable_list inputs) {
  for (const auto& hook : fn.tensor_pre_hooks()) {
    inputs = (*hook)(inputs);
  }
  for (const auto& pair : fn.retains_grad_hooks()) {
    inputs = (*pair.second)(inputs);
  }
  return inputs;
}

static variable_list call_post_hooks(
    Node& fn,
    variable_list outputs,
    const variable_list& inputs) {
  for (const auto& hook : fn.post_hooks()) {
    outputs = (*hook)(outputs, inputs);
  }
  return outputs;
}

//上节代码之中，有如下代码进行配置设备
//可以看到，在 set_device 时候，如果不是 CPU_DEVICE，就设置 impl->setDevice，
void set_device(int device) {
  // NB: We MUST NOT construct the guard for device CPU,
  // as in some settings we compile with cuda, but
  // have lazy stubs for CUDA functionality (so actually
  // attempting to setup a guard(CPU_DEVICE) will cause an
  // error, because it will still query cudaGetDevice).
  //
  // Don't use DeviceGuard here because its destructor may be called before the
  // device is reset. This is fine because the device is thread local.
  if (device != CPU_DEVICE) {
    for (const auto i : c10::irange(static_cast<size_t>(
             c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES))) {
      auto* impl = c10::impl::device_guard_impl_registry[i].load();
      if (impl && device < impl->deviceCount()) {
        impl->setDevice(at::Device(static_cast<c10::DeviceType>(i), device));
      }
    }
  }
  worker_device = device;
}

/*
其定义在 torch/csrc/autograd/engine.cpp，原生引擎和分布式引擎都会调用。validate_outputs 之中包含了大量的验证代码。

    如果梯度数量与边数目不同，则退出。
    遍历梯度，对于每个梯度：
        获取对应的边，如果边无效，则去下一个梯度。
        使用input_metadata 获取输入信息。
        如果梯度没有定义，也去下一个梯度。
        如果梯度尺寸与输入形状不同，则退出。
        对梯度的设备，元数据的设备进行一系列判断。

*/
void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error) {
  if (grads.size() != edges.size()) {
    std::stringstream ss;
    ss << "invalid number of gradients - expected ";
    ss << edges.size() << ", but got " << grads.size();
    AT_ERROR(format_error(ss.str()));
  }
  for (const auto i : c10::irange(grads.size())) {
    const auto& edge = edges[i];
    if (!edge.is_valid())
      continue;

    const auto& metadata = edge.function->input_metadata(edge.input_nr);
    auto& grad = grads[i];
    if (!grad.defined()) {
      // FIXME: TestJit.test_ge_optimized fails this assertion.
      // std::stringstream ss;
      // ss << "undefined gradient at index " << i;
      // AT_ERROR(format_error(ss.str()));
      continue;
    }

    if (!metadata.is_same_shape(grad)) {
      if (metadata.is_expandable_to_shape(grad)) {
        grad = metadata.reduce_grad(grad);
      } else {
        const auto message = metadata.incompatible_shape_error_message(i, grad);
        AT_ERROR(format_error(message.str()));
      }
    }

    bool input_is_complex =
        isComplexType(c10::typeMetaToScalarType(metadata.options().dtype()));
    bool grad_is_complex = isComplexType(grad.scalar_type());

    TORCH_CHECK(
        isFloatingType(grad.scalar_type()) ||
        (input_is_complex == grad_is_complex));
    if (c10::typeMetaToScalarType(metadata.options().dtype()) !=
        grad.scalar_type()) {
      grad = grad.to(c10::typeMetaToScalarType(metadata.options().dtype()));
    }

    // 如果梯度尺寸与输入形状不同，则退出
    if (grad.dtype() != metadata.dtype()) {
      std::stringstream ss;
      ss << "invalid gradient at index " << i << " - expected dtype ";
      ss << metadata.dtype() << " but got " << grad.dtype();
      AT_ERROR(format_error(ss.str()));
    }
    if (grad.layout() != metadata.layout()) {
      // TODO: Currently we only support (*, Sparse) combination for
      // (tensor.layout(), tensor.grad.layout()) In future, there will be an
      // oppportunity to support more combinations of layouts if they are
      // composable (example., operations like addition etc., are well defined
      // between tensors of different layouts.), as well as all parts of
      // autograd like AccumulateGrad correctly handle this. We allow grad to be
      // Strided when metadata is SparseCsr
      if (!grad.is_sparse() &&
          !(grad.layout() == at::kStrided &&
            (at::sparse_csr::is_sparse_compressed(metadata.layout()) ||
             metadata.layout() == at::kSparse))) {
        std::stringstream ss;
        ss << "invalid gradient at index " << i << " - expected layout ";
        ss << metadata.layout() << " but got " << grad.layout();
        AT_ERROR(format_error(ss.str()));
      }
    }

    if (grad.device() != metadata.device()) {
      // quick hack for: https://github.com/pytorch/pytorch/issues/65016 but
      // should be eventually removed
      if (!(metadata.is_tensor_subclass() ||
            grad.unsafeGetTensorImpl()->is_python_dispatch())) {
        if (grad.dim() == 0) {
          grad = grad.to(metadata.device());
        } else {
          std::stringstream ss;
          ss << "invalid gradient at index " << i << " - expected device ";
          ss << metadata.device() << " but got " << grad.device();
          AT_ERROR(format_error(ss.str()));
        }
      }
    }
    // We should not build graph for Tensors that are not differentiable
    TORCH_INTERNAL_ASSERT(isDifferentiableType(grad.scalar_type()));
  }
}

/*
0x04 核心逻辑
call_function是反向传播中计算相关的核心逻辑。
    调用注册在本 node上的pre_hooks；
    调用node本身，比如MeanBackward0、MulBackward0等。
        输入是InputBuffer::variables(std::move(inputBuffer))，一组Variable的实例。当动态图刚开始进行反向计算时，引擎首先执行的是图的根节点——graph_root，它的输入是task.inputs——InputBuffer(0)。
        调用的是fn的apply()，apply是多态实现，针对不同的operation会dispatch到operation对应的apply实现上。
        输出也是一组Variable的实例 outputs = fn(std::move(inputs_copy))，outputs 要作为下一个fn的输入。
    调用注册在node上的post hooks。
    返回当前节点对应的导数，这是一个variable_list。

*/
static variable_list call_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputBuffer) {
  CheckpointValidGuard cpvguard(graph_task);
  auto& fn = *func;
  auto inputs =
      call_tensor_pre_hooks(fn, InputBuffer::variables(std::move(inputBuffer)));
  inputs = call_pre_hooks(fn, std::move(inputs));
  if (!graph_task->keep_graph_) {
    fn.will_release_variables(); // 如果不需要保持图，就调用释放。
  }

  const auto has_post_hooks = !fn.post_hooks().empty();
  variable_list outputs;

  if (has_post_hooks) {
    // In functions/accumulate_grad.cpp, there is some logic to check the
    // conditions under which the incoming gradient can be stolen directly
    // (which elides a deep copy) instead of cloned. One of these conditions
    // is that the incoming gradient's refcount must be 1 (nothing else is
    // referencing the same data).  Stashing inputs_copy here bumps the
    // refcount, so if post hooks are employed, it's actually still ok for
    // accumulate_grad.cpp to steal the gradient if the refcount is 2.
    //
    // "new_grad.use_count() <= 1 + !post_hooks().empty()" in
    // accumulate_grad.cpp accounts for this, but also creates a silent
    // dependency between engine.cpp (ie, this particular engine
    // implementation) and accumulate_grad.cpp.
    //
    // If you change the logic here, make sure it's compatible with
    // accumulate_grad.cpp.
    auto inputs_copy = inputs;
    outputs = fn(std::move(inputs_copy));
  } else {
    outputs = fn(std::move(inputs));
  }

  validate_outputs(fn.next_edges(), outputs, [&](const std::string& msg) {
    std::ostringstream ss;
    ss << "Function " << fn.name() << " returned an " << msg;
    return ss.str();
  });

  if (has_post_hooks) {
    // NOLINTNEXTLINE(bugprone-use-after-move)
    return call_post_hooks(fn, std::move(outputs), inputs);
  }
  return outputs;
}
/*
evaluate_function 方法完成了反向计算的逻辑，总体逻辑如下：

准备工作：如果exec_info需要处理，则处理 captured_vars_。
反向计算：调用 call_function(graph_task, func, inputs)，这是反向传播中计算相关的核心逻辑：
    调用pre hooks。
    调用fn进行计算。
    调用post hooks。
扫尾工作：
    如果不需要keep graph，则fn.release_variables();
    依据 call_function的输出 outputs，进行计算 num_outputs = outputs.size()，得到 num_outputs的元素数量（该数量等同于当前fn的next_edge()返回的list中的元素数量）。
准备下一步工作，具体就是查找后续需要计算的NodeTask，num_outputs 就是在这里被用到。这部分比较复杂。


准备工作，具体如下：
    取出当前 Node 的 ExecInfo。
    取出其 captures_，遍历其中每一个 Capture。
    遍历Capture 的 hooks，链式调用hook进行计算。
    captured_grad 不停的作为输入和输出在流水线中流淌，针对 captured_vars_[capture.output_idx_]陆续计算。
    最终结果保存在 captured_vars_[capture.output_idx_] 之中。




2.3 evaluate_function
上面代码之中，实际上会调用原生引擎的 evaluate_function 来完成操作。

我们看看如何使用 exec_info_，如果没有设置为需要执行，则就不处理。在此处，我们可以看到 上文提到的recvBackwardEdges 如何与 exec_info_ 交互。

遍历 recvBackwardEdges，对于每个 recvBackward，在 GraphTask.exec_info_ 之中对应项之上设止为需要执行。

具体代码如下，这里会：

针对 AccumulateGrad 来累积梯度。
调用 RecvRpcBackward 来向反向传播下游发送消息。

具体结合到分布式引擎，就是当引擎发现某一个 Node 是 RecvRpcBackward，就调用其 apply 函数。
*/
void Engine::evaluate_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func, // 导数计算方法
    InputBuffer& inputs,  // 当前Node的输入梯度
    const std::shared_ptr<ReadyQueue>& cpu_ready_queue) {
  // The InputBuffer::adds that supplied incoming grads took pains to
  // ensure they're safe to consume in the context of the present
  // func's stream (if applicable). So we guard onto that stream
  // before working with the grads in any capacity.
  const auto opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);
  c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};

  // 进行准备工作
  // If exec_info_ is not empty, we have to instrument the execution
  auto& exec_info_ = graph_task->exec_info_;
  if (!exec_info_.empty()) {
    auto& fn_info = exec_info_.at(func); // 取出当前的进行处理
    variable_list new_inputs = inputs.buffer;
    if (!fn_info.needed_) {
      // We always want to call tensor pre-hooks, but want to avoid calling it
      // twice. needed_ = True indicates that we will call tensor pre-hooks
      // later.
      //
      // See NOTE [Hooks ordering] for more context.
      new_inputs = call_tensor_pre_hooks(
          *func, InputBuffer::variables(std::move(inputs)));
    }
    if (auto* capture_vec = fn_info.captures_.get()) {
      const auto opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);
      // Lock mutex for writing to graph_task->captured_vars_.
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      for (const auto& capture : *capture_vec) {
        // 这里是引用操作，所以 captured_grad 的赋值实际就是往 graph_task->captured_vars_ 赋值

        // captured_grad 就是临时存储下，每次node计算都会更新，最终输出给调用者，相当于引用
        // 1. captured_grad 引用了captured_vars_[capture.output_idx_]，
        auto& captured_grad = graph_task->captured_vars_[capture.output_idx_];

        // 这里是引用操作，所以 captured_grad 的赋值实际就是往 graph_task->captured_vars_ 赋值
        // 2. 给 captured_vars_[capture.output_idx_] 赋值 inputs[capture.input_idx_]
        captured_grad = new_inputs[capture.input_idx_];
        // NOTE [Deprecated capture hooks]

        // 遍历hooks，链式调用hook进行计算，captured_grad 不停的作为输入和输出在流水线中流淌
        // 就是针对 captured_vars_[capture.output_idx_]不停的计算，最终结果还是在 captured_vars_[capture.output_idx_] 之中。
        for (const auto& hook :
             capture.DO_NOT_USE_DEPRECATED_get_capture_hooks()) {
          // 这里使用了 hook 进行后置操作
          ////这里调用 hook，就是 DistAccumulateGradCaptureHook 的 operator()，captured_grad 就是累积的梯度
          captured_grad = (*hook)(captured_grad);
        }
        if (opt_parent_stream) {
          // No need to take graph_task->mutex_ here, we already hold it
          graph_task->leaf_streams.emplace(*opt_parent_stream);
        }
      }
    }
    if (!fn_info.needed_) {
      // Skip execution if we don't need to execute the function.
      // 如果没有设置需要执行，则直接返回。recvBackward 会设置需要执行
      return;
    }
  }

// 执行后向计算 // 进行反向计算
//这部分是反向传播的复杂之处。
//现在调用 call_function，得到了后向传播的输出，记录到了 outputs 之中。
// 这里就是调用 recvBackward  // 这里就是调用 recvBackward.apply 函数
  auto outputs = call_function(graph_task, func, inputs);

// 如果不需要保持计算图，则本节点释放变量
  auto& fn = *func;
  if (!graph_task->keep_graph_) {
    fn.release_variables();
  }

// 得到 num_outputs的元素数量（该数量等同于当前fn的next_edge()返回的list中的元素数量），后续遍历本节点输出时候会用到
  int num_outputs = outputs.size();
  if (num_outputs == 0) { // Note: doesn't acquire the mutex
    // Records leaf stream (if applicable)
    // See Note [Streaming backwards]
    if (opt_parent_stream) {
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      graph_task->leaf_streams.emplace(*opt_parent_stream);
    }
    return;
  }

  if (AnomalyMode::is_enabled() && AnomalyMode::should_check_nan()) {
    AutoGradMode grad_mode(false);
    for (const auto i : c10::irange(num_outputs)) {
      auto& output = outputs[i];
      at::OptionalDeviceGuard guard(device_of(output));
      if (output.defined() && isnan(output)._is_any_true().item<bool>()) {
        std::stringstream ss;
        ss << "Function '" << fn.name() << "' returned nan values in its " << i
           << "th output.";
        throw std::runtime_error(ss.str());
      }
    }
  }

// 准备下一步工作
  // Lock mutex for the accesses to GraphTask dependencies_, not_ready_ and
  // cpu_ready_queue_ below
  std::lock_guard<std::mutex> lock(graph_task->mutex_);

  /*总体思路就是：遍历后向传播的输出节点（就是该节点在前向计算图中的入边连接的节点），逐一衡量输出节点。遍历循环中分为两段代码，对于每一个输出节点做如下操作：
        第一段是依据依赖排查这个节点，得到这个节点是否就绪。核心就是看看这个输出节点在GraphTask的dependencies的计数是否降为0。
            如果是0，就说明这个节点就绪了，说明这个node不会被未来的计算所依赖了。
            如果非0，就说明这个节点有多个输入，即，被多个node连接，而且有的输入还没有计算完成梯度。
        第二段是依据是否就绪来处理这个节点，比如放入哪一个queue。
        5.1 依据依赖排查节点
            第一段代码功能是依据依赖关系来 排查节点，得到这个节点是否就绪，具体如下：
            假定某一个节点是 output，我们得到对应的边，遍历输出边。
                每次把一个输出边记录为 next，func 是 NodeTask 之中的函数。
                利用 dependencies_ 的信息，next 是否可以计算。dependencies_ 里面记录的是图中所有节点的依赖。
                从 dependencies_ 之中找到 next 对应的依赖数目，把依赖数目减一（通常因为有多个 input）。
                    如果--it->second == 0，说明该前置节点计算梯度所依赖的其他节点梯度都已经完成计算。则
                        把该前置节点对应的信息GraphTask中移除，即从GraphTask的dependencies中移除（后续也会从GraphTask的 not_ready 成员变量之中移除）。
                        将is_ready 置为true，后续会依据这个 is_ready 的数值进行操作。
                从 not_ready_ 之中得到 next 对应的输入buffer（后续代码就是对此进行操作）；
                std::unordered_map<Node*, InputBuffer> not_ready_;
        代码如下：
    */
  for (const auto i : c10::irange(num_outputs)) {  // 遍历自己的输出
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);  // 找到第i个输出  // 查找下一个可以计算的节点 // next_edge是该node在前向传播图中的输入，在反向传播时候就是本节点的输出，所以next就是下一个可能运算的节点

    if (!next.is_valid())
      continue;

    // Check if the next function is ready to be computed
    bool is_ready = false;
    auto& dependencies = graph_task->dependencies_;  // 拿到GraphTask的依赖关系
    auto it = dependencies.find(next.function.get()); // 找到第i个输出的依赖关系 // 找到下一个节点的依赖  // 找到输出节点的依赖项

    if (it == dependencies.end()) {
      auto name = next.function->name(); // 没找到
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) { // 因为本节点的后向计算已经完成，所以第i个输出的依赖数目减一
      dependencies.erase(it);   // 如果为0，说明没有依赖了，就从依赖关系之中删除   // 找到了，并且已经计算完毕
      is_ready = true;  // true 代表没有依赖关系，可以构建一个 NodeTask 进行下一步反向计算了  // 下一个节点没有入度了，那么说明计算该节点梯度依赖的其他节点梯度都已经计算完成
    }

    // 得到依赖关系  // 要去 not_ready里面看看，是否已经存储了
    auto& not_ready = graph_task->not_ready_;
    auto not_ready_it = not_ready.find(next.function.get());  // 找到输入buffer   现在已经找到了某一个输出节点，也知道其是否计算完毕（依据有没有依赖项），也拿到了其存在"未就绪队列"的输入buffer（如果存在的话）。


    /*5.2 处理这个节点
    第二段是依据是否就绪来处理这个节点，比如放入哪一个queue，是就绪队列？还是未就绪队列？核心是：
        如果就绪，就放到该节点对应的 ReadyQueue 去处理。
        如果没有就绪，就新建立一个NodeTask放到 GraphTask的 not_ready 等待后续处理。需要注意的是，这个新的NodeTask 是在 worker thread 之中创建的。
        如何找到 ReadyQueue？需要看这个 Node 节点的 input_buffer.device() ，即，这个新 NodeTask 应该发送到 input_buffer.device() 那个 device 对应的 ReadyQueue。
    我们具体看看如何依据 is_ready 的数值来对 not_ready 进行操作。

        如果在 未就绪队列 not_ready 之中 没有找到 next_edge 对应的元素，则:
            如果 exec_info_ 不为空，则在 exec_info_ 之中查找 next_edge 对应的元素，如果有元素且注明了不需要执行，就跳到for循环的下一个。
            用 next_edge 的流，inut_nr 等信息构建一个 input_buffer。
            如果 is_ready 是 True，就用 本 GraphTask，next.function，input_buffer构建一个NodeTask，
            放入 ReadyQueue（利用 input_buffer.device() 来得到对应的 queue）。这就要唤醒下一个 worker 线程。
            如果 is_ready 是 False，这通常表明这个node有多个输入（被更多的node连接，使用num_inputs()可以获得数量），也说明此次处理的是这个node的第一个输入，后续还需要使用这个 next_edge，
            所以这个 next_edge 需要被放到 not_ready 之中。则把 next.function，input_buffer 放入到 not_ready 之中，这个input_buffer 就是 next_edge 后续执行时候需要的各种输入。
        如果在 未就绪队列 not_ready 之中找到了 next_edge 对应的元素，则:
            拿出来该元素对应的 input_buffer，把信息累积到 input_buffer 之中。此次累积的是该节点的其他输入。
            input_buffer.add(next.input_nr, std::move(output), opt_parent_stream, opt_next_stream) 完成了累积操作，
            next.input_nr 就表明当前的node是反向传播中要流向的node（next）的第几个输入。
            如果is_ready 是 True，就用 本 GraphTask，next.function，input_buffer构建一个NodeTask，放入 ReadyQueue。这就要唤醒下一个 worker 线程。
            从 not_ready 之中移除此元素，就是从 GraphTask 的依赖关系之中去除。


    具体逻辑图如下：

        func 指向了目前正在进行反向计算的 Node。
        func 调用自己的 apply 方法进行计算，得出了 outputs，假设有3个输出，遍历，我们选择第三个为 output。
        func 的边是 next_edges_ 成员变量，遍历，我们选择第三个边为next。
        用 next 和 GraphTask 的 dependencies_ 来判断 next 是不是就绪。
        如果就绪，把 output 构建一个 input_buffer，然后生成一个 NodeTask，插入到对应的 ReadyQuieue。
        如果没就绪，把 output 构建一个 input_buffer，和 next 一起放入 GraphTask 的 not_ready_，后续会使用。
    */
    if (not_ready_it == not_ready.end()) {  // 如果未就绪队列之中没有next节点
      // Skip functions that aren't supposed to be executed
      // 跳过不需要计算的节点
      //// 下一个节点的梯度还没有进行计算
      if (!exec_info_.empty()) {
        auto it = exec_info_.find(next.function.get());
        if (it == exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
      }
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next.function->num_inputs());  // 整理 next 节点的输入参数信息 // 下一个节点前置梯度的buffer，就是下一个节点的输入梯度

      // Accumulates into buffer
      // 下一个节点的输入梯度就是当前节点的输出，所以要拷贝过去
      const auto opt_next_stream = next.function->stream(c10::DeviceType::CUDA);
      input_buffer.add(
          next.input_nr, // 插入 next 节点的输入参数信息
          std::move(output),
          opt_parent_stream,
          opt_next_stream);

      if (is_ready) {
        // is_ready 是前面小节之中，通过依赖关系计算出来的，true表示可以进行反向计算了
        auto queue = ready_queue(cpu_ready_queue, input_buffer.device());
        //// 插入下一个需要计算的NodeTask  // 既然依赖全部完成，就插入到ReadyQueue 之中
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
      } else {
        // 还有依赖关系，不能进行反向计算，只能放入未就绪队列 not_ready_  // 下一个节点的输入依赖还没有完成，就放到not_ready之中。
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      // 如果未就绪队列之中已经有next节点
      // 如果下一个节点已经开始计算，但是没有完成（就是依赖梯度还有），此时应该在not_ready之中
      // The function already has a buffer
      auto& input_buffer = not_ready_it->second;

      // Accumulates into buffer
      const auto opt_next_stream = next.function->stream(c10::DeviceType::CUDA);
      input_buffer.add(
          next.input_nr, std::move(output), opt_parent_stream, opt_next_stream);

      // Graph中每一个node（fn）的输出是下一个node（fn）的输入，下面4句代码来将前一个fn的输出转化为下一个fn的输入
      if (is_ready) {
        // 如果可以计算，就放入ready 队列
        // 如果此时已经没有输入依赖，就放入新的NodeTask，就是下一个需要计算梯度的NodeTask
        // 找出了下一个Node的queue
        auto queue = ready_queue(cpu_ready_queue, input_buffer.device());
        queue->push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
        //已经完成下一个节点前置梯度计算，从not_ready中移除相应的buffer
        not_ready.erase(not_ready_it);   // 同时从未就绪队列之中移除
      }
    }
  }
}

//compute_min_topological_nr 的作用是：遍历出边，找到最小拓扑数 min_topo_nr。min_topo_nr 接下来会用在计算依赖函数之中。
inline static uint64_t compute_min_topological_nr(const edge_list& outputs) {
  // Computes the mininum topological number among all the outputs
  if (outputs.empty()) {
    return 0;
  }
  auto min_topo_nr = std::numeric_limits<uint64_t>::max();
  for (auto& output_edge : outputs) {
    auto topo_nr = output_edge.function.get()->topological_nr();
    min_topo_nr = (min_topo_nr < topo_nr) ? min_topo_nr : topo_nr;
  }
  return min_topo_nr;
}

/*
dependencies 的 key 数目就是微分图中Node数目，dependencies 计算的就是每一个Node的入度。
dependencies成员在compute_dependencies调用中被初始化，
只要某一个grad_fn函数在其他人的 next_edges() 中出现过一次，那么dependencies[this_grad_fn] 就自增1。
如果dependencies[this_grad_fn]大于0，
说明this_grad_fn有一个后向传播的依赖，
即 this_grad_fn 需要等 被依赖者 完成，才能进行自己的反向传播相关计算。
compute_dependencies 就是计算GraphTask的dependencies_。
其逻辑是：从 graph_root 开始，对微分图中每个node的依赖进行计算，计算从根节点开始，通过广度优先的算法进行。
如果一个grad_fn函数在别人的next_edges()中出现过一次，那么dependencies[grad_fn] 就自增1。具体代码如下：
*/
auto Engine::compute_dependencies(
    Node* root,
    GraphTask& task,
    uint64_t min_topo_nr) -> void {
  // Computes the number of dependencies for each function which requires grad
  std::vector<Node*> queue{root};
  bool might_use_cuda = at::globalContext().hasCUDA();
  bool will_use_cuda = false;

  // Queue contains all nodes that will start propagating gradients.
  // We no longer have to expand functions that don't require grad.
  auto& dependencies = task.dependencies_;
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    if (fn->topological_nr() < min_topo_nr) {
      continue;
    }
    if (might_use_cuda && !will_use_cuda) {
      will_use_cuda = fn->stream(c10::DeviceType::CUDA).has_value();
    }
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        dependencies[next_ptr] += 1;
        const bool was_inserted = task.nodes_in_graph_.insert(next_ptr).second;
        if (was_inserted)
          queue.push_back(next_ptr);
      }
    }
  }

  if (will_use_cuda) {
    // Collects current streams for devices where this process has a context,
    // so GraphTask::exec_post_processing can sync them with leaf_streams.
    task.stash_current_streams();
  }
}

// 之前在 execute 之中会用 fut->wait() 来等待任务完成。下面我们省略了部分代码。
auto Engine::execute(
    const edge_list& root_edges, // 反向传播的根节点
    const variable_list& inputs, // 根节点的梯度
    bool keep_graph,  // 计算图是否需要保留
    bool create_graph, // 是否需要构建微分图以进行高阶求导
    bool accumulate_grad,
    const edge_list& outputs // 需要输出梯度的节点
    ) -> variable_list {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  validate_outputs(
      root_edges,
      const_cast<variable_list&>(inputs),
      [](const std::string& msg) { return msg; });
  if (accumulate_grad && create_graph) {
    TORCH_WARN_ONCE(
        "Using backward() with create_graph=True will create a reference cycle "
        "between the parameter and its gradient which can cause a memory leak. "
        "We recommend using autograd.grad when creating the graph to avoid this. "
        "If you have to use this function, make sure to reset the .grad fields of "
        "your parameters to None after use to break the cycle and avoid the leak.");
  }

  // accumulate_grad is true if and only if the frontend call was to
  // grad(), not backward(). grad() returns the sum of the gradients
  // w.r.t. the inputs and thus needs the inputs to be present.
  TORCH_CHECK_VALUE(
      accumulate_grad || !outputs.empty(), "grad requires non-empty inputs.");

  // A fresh first time Engine::execute call should start on the CPU device,
  // initialize a new thread local ready queue on CPU or reuse the existing one
  // (if there is one allocated already, i.e. consecutive backward calls,
  // re-entrant backward calls), then memoize the local_ready_queue in GraphTask
  init_local_ready_queue(); // 初始化local ready_queue
  bool not_reentrant_backward_call = worker_device == NO_DEVICE;

  // Store root nodes so we can traverse through the graph later
  // e.g., for get_current_graph_task_execution_order
  c10::SmallVector<Node*, 4> temp_roots{root_edges.size()};
  for (const auto i : c10::irange(root_edges.size())) {
    temp_roots[i] = root_edges[i].function.get();
  }

// 构建一个GraphTask
//对于普通引擎也会设置一个 cpu 专用 queue。
/*
2.5 小结
对于分布式引擎，与普通引擎在计算部分主要不同之处为：

如果是 RecvRpcBackward 则会给对应的下游节点发送 RPC 消息。

如果是 AccumulateGrad，则在上下文累积梯度。

所以我们接下来看看具体这两部分如何处理。

*/
  auto graph_task = std::make_shared<GraphTask>(
      /* keep_graph */ keep_graph,
      //用来指定一次反向计算后是否释放资源。资源就是在前向过程中建立起来的资源。
      //keep_graph如果是False的话，则会在 fn 执行完毕后调用 fn 的 will_release_variables 方法来释放该资源。

      /* create_graph */ create_graph,
      /* depth */ not_reentrant_backward_call ? 0 : total_depth + 1,
      /* cpu_ready_queue */ local_ready_queue,
      /* graph_roots */ std::move(temp_roots));

    // If we receive a single root, skip creating extra root node
    //构建 GraphRoot。结合 execute 的调用方式，我们知道是使用 反向传播的根节点（起始点）和 根节点的梯度 inputs 来构建 GraphRoot。
    //  GraphRoot 的作用是:
    //GraphRoot 就是后向传播的输入，就是根节点。
    //在构造 graph_root 时候：如果只有一个root节点，则就直接使用root作为 GraphRoot 。
    //如果多个root，就构造一个GraphRoot（可以认为是虚拟根节点），把这些 root 作为参数构建一个GraphRoot，这个 GraphRoot 作为真正的根节点。
    //root 就是 Node 的边。即，把这些根对应的 edge_list 转换为 Node 里面的 next_edges_，这个GraphRoot可以认为是一个虚拟Root。

    //从初始化函数可以看出来，引擎的输入inputs（反向传播的输入梯度）就是GraphRoot的输出 outputs。
    //Function 的灵魂是 apply 方法，对于 GraphRoot 来说，其apply函数仅仅返回它的输入，这样，原始输入 input 就直接被 GraphRoot 透传给反向传播的下一阶段。
    //后续计算 compute_dependencies 会用这个 GraphRoot 来得到计算图的依赖关系，就是利用 GraphRoot 的 next_edges_ 来得到计算图的依赖关系。
    //// 构建GraphRoot
  bool skip_dummy_node = root_edges.size() == 1;
  auto graph_root = skip_dummy_node
      ? root_edges.at(0).function  // 如果只有一个root，就直接使用root作为 GraphRoot
      : std::make_shared<GraphRoot>(root_edges, inputs); // 如果多个root，就构造一个GraphRoot
      /*graph_root 由roots和inputs构建，roots就是最终输出节点的gradient_edge()，
      比如 [ (MulBackward0实例，0），(PowerBackward0, 0) ]。
      inputs 如果用户没有指定，就是默认的 tensor(1.)，
      如果指定了，就是起始梯度。*/

  auto min_topo_nr = compute_min_topological_nr(outputs);
  // Now compute the dependencies for all executable functions
  compute_dependencies(graph_root.get(), *graph_task, min_topo_nr);

  //在 Engine::execute 之中会调用 init_to_execute 生成ExecInfo。
  //如果出边不为空，则会调用 init_to_execute 对GraphTask.exec_info_进行初始化。
  if (!outputs.empty()) {
    graph_task->init_to_execute(
        *graph_root, outputs, accumulate_grad, min_topo_nr);
  }

  // Queue the root
  if (skip_dummy_node) {
    //// 配置工作进程的各种输入
    //// 如果是单节点，则直接使用 CUDA queue
    InputBuffer input_buffer(root_edges.at(0).function->num_inputs());
    auto input = inputs.at(0);
    // 构建InputMetadata
    const auto input_stream = InputMetadata(input).stream();
    const auto opt_next_stream =
        root_edges.at(0).function->stream(c10::DeviceType::CUDA);
    input_buffer.add(
        root_edges.at(0).input_nr,
        std::move(input),
        input_stream,
        opt_next_stream);

     // 启动工作进程
    execute_with_graph_task(
        graph_task, std::move(graph_root), std::move(input_buffer));
  } else {
    // 启动工作进程
    //// 如果是多输入根节点，之前构建了虚拟根节点，后续就对应了 CPU queue
    execute_with_graph_task(
        graph_task, std::move(graph_root), InputBuffer(variable_list()));
  }
  // Avoid a refcount bump for the Future, since we check for refcount in
  // DistEngine (see TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1)
  // in dist_engine.cpp).
  // 主进程进行阻塞等待，等待 graph_task->future_result_。
  //  这里使用 std::shared_ptr<at::ivalue::Future> future_result_; 来进行线程间通信。
  //主线程用wait等待，工作线程用markCompleted来通知主进程结束。
  auto& fut = graph_task->future_result_;
  fut->wait();
  graph_task->warning_handler_.replay_warnings();
  return fut->value().toTensorVector();
}

//这里使用std::call_once来确保在整个进程周期之内，start_device_threads成员函数只被调用了一次，即设备线程只生成一次。
void Engine::initialize_device_threads_pool() {
  TORCH_CHECK(
      !in_bad_autograd_fork,
      "Unable to handle autograd's threading in combination with fork-based multiprocessing. "
      "See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork");
  c10::call_once(
      start_device_threads_flag_, &Engine::start_device_threads, this);
}

/*
接下来会调用 execute_with_graph_task。execute_with_graph_task具体做就是启动后续各种线程，关于 线程 具体我们后续详解。
这里就是知道，根据当前设备来走不同路径，具体逻辑如下：
如果 worker_device == NO_DEVICE，说明这是主线程，则：
    获取到相关queue，具体是利用 input_buffer.device() 获得的，这就用到了上节的 InputBuffer。
    如果获取到的设备是 CPU，就获取 GraphTask.cpu_ready_queue_，如果是 GPU，就用到了对应的GPU设备对应的 queue，具体我们后文详述。
    在 queue 之中插入 NodeTask。调用 thread_main 运行 GraphTask。
否则是可重入反向传播情况下的主线程，则：
    利用 graph_task->owner_ = worker_device 指定当前设备是哪个设备，GPU 或者 CPU。
    如果已经达到了最大递归深度，则采用add_thread_pool_task 启动 GPU线程 或者 CPU线程。
    否则运行 thread_main。
*/
c10::intrusive_ptr<at::ivalue::Future> Engine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  initialize_device_threads_pool(); // 启动设备工作线程 // 这里首先会启动工作线程
  // Lock mutex for GraphTask.
  std::unique_lock<std::mutex> lock(graph_task->mutex_);

  // 获取到相关queue，具体是利用 input_buffer.device() 获得的。
  // 这里指定了后续究竟是GPU还是CPU上运行，
  // 因为 input_buffer.device() 里面指定了运行的设备，所以依据这个设备，获取到了对应的 queue
  auto queue = ready_queue(graph_task->cpu_ready_queue_, input_buffer.device());

  // worker_device == NO_DEVICE it's a CPU thread and it's trying to drive the
  // autograd engine with corresponding GraphTask, and its NOT a re-entrant call
  if (worker_device == NO_DEVICE) { // 判断是否已经运行了反向传播
    // 如果到了这里，必然是没有活跃工作设备
    // 主线程
    // We set the worker_device to CPU_DEVICE only if worker_device was
    // previously NO_DEVICE. Setting it to CPU afterwards allow us to detect
    // whether this is a re-entrant call or not.
    // 在 execute_with_graph_task 之中，当有重入的反向传播时候，也会插入 NodeTask：
    set_device(CPU_DEVICE);

    // set the graph_task owner to the current device
    graph_task->owner_ = worker_device; // 就是 CPU 设备

    // Now that all the non-thread safe fields of the graph_task have been
    // populated, we can enqueue it.
    //// 给 queue 之上插入 NodeTask，这样就会唤醒对应线程开始工作
    // 主线程之中
    //刚启动时候，在 execute_with_graph_task 之中，
    //主线程就是往 index = -1 的 CPU 工作线程的queue 发送一个 NodeTask。
    queue->push(
        NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    // The owning thread start to drive the engine execution for any CPU task
    // that was just pushed or will be added later from other worker threads
    lock.unlock();
    thread_main(graph_task); // 在主线程运行，这里会在queue之上阻塞  // thread_main 依然是被主线程执行，内部通过 pop 阻塞等待
    TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    // reset the worker_device after the completion of the graph_task, this is
    // so that the initial state of the engine remains the same across every
    // backward() or grad() call, we don't need to reset local_ready_queue as we
    // could possibly reuse it for new backward calls.
    //// 主线程
    worker_device = NO_DEVICE;
    // 如果到了这里，必然是没有活跃工作设备，就是所有 GraphTask都结束了，如果没有结束，就是reentrant，必须走下面的case
    // 主线程
  } else {
    // 主线程，可重入的反向传播
    // // 重入后向传播状况下的主线程
    // If worker_device is any devices (i.e. CPU, CUDA): this is a re-entrant
    //    backward call from that device.

    /*
    如果这个task是来自其它work thread，即 worker_device != base_owner，则向那个worker thread的queue发送一个dummy function task，让那个工作线程也执行起来。
    local_graph_task 表示我们从队列中检索的 graph_task。外部graph_ 任务表示我们需要执行的可重入执行的总体graph_任务。
    在 thread_main 之中，有一个 work around。就是：当前工作线程完成 graph_task，但此时，拥有graph_task的线程可能正在pop()上等待休眠。
    因此，我们需要向所属线程发送一个仿造的函数任务，以唤醒它，这样我们可以退出thread_main。
    这种情况发生在可重入反向传播的情形。
    */
    graph_task->owner_ = worker_device; // 指定是哪个设备，是 GPU 或者 CPU

    // Now that all the non-thread safe fields of the graph_task have been
    // populated, we can enqueue it.
    //// 向 queue 插入第一个NodeTrask，就是 graph_root
    queue->push(
        NodeTask(graph_task, std::move(graph_root), std::move(input_buffer)));

    if (current_depth >= max_recursion_depth_) {
      //当发现是可重入后向传播时，而且超出最大递归深度，Engine::execute_with_graph_task 会调用如下代码向线程池加入一个新线程。
      // See Note [Reentrant backwards]
      // If reached the max depth, switch to a different thread
      // 达到最大重入深度，这里会启动一个新的线程
      add_thread_pool_task(graph_task);
    } else {
      // Total depth needs to be updated only in this codepath, since it is
      // not used in the block above (when we call add_thread_pool_task).
      // In the codepath above, GraphTask.reentrant_depth_ is used to
      // bootstrap total_depth in the other thread.
      ++total_depth;

      // Get back to work while we wait for our new graph_task to
      // complete!
      ++current_depth;
      lock.unlock();
      thread_main(graph_task);  // 在主线程运行，这里会在queue之上阻塞 // thread_main 依然是被主线程执行，内部通过 pop 阻塞等待
      --current_depth;
      --total_depth;

      // The graph task should have completed and the associated future should
      // be marked completed as well since 'thread_main' above is a call
      // blocking an autograd engine thread.
      TORCH_INTERNAL_ASSERT(graph_task->future_result_->completed());
    }
    // 重入后向传播状况下的主线程
  }
  // graph_task_exec_post_processing is done when the Future is marked as
  // completed in mark_as_completed_and_run_post_processing.
  return graph_task->future_result_;
}

// note that when python is present, this base engine will be overriden
// with a PythonEngine. Because this typically happens before get_default_engine
// is called, this base engine will never be created.
Engine& Engine::get_base_engine() {
  static Engine engine;
  return engine;
}

std::atomic<EngineStub> engine_stub(Engine::get_base_engine);

void set_default_engine_stub(EngineStub stub) {
  engine_stub.store(stub);
}

Engine& Engine::get_default_engine() {
  return engine_stub.load()();
}

/*
3.1.2 注册callback
上面代码之中，使用了 torch::autograd::Engine::get_default_engine().queue_callback 来注册了一个回调函数。我们就来分析一下。

在engine之中有定义，就是往 final_callbacks_ 插入callback：
*/
void Engine::queue_callback(std::function<void()> callback) {
  TORCH_CHECK(
      current_graph_task,
      "Final callbacks can only be installed during backward pass.");

  std::lock_guard<std::mutex> lock(current_graph_task->final_callbacks_lock_);
  current_graph_task->final_callbacks_.emplace_back(std::move(callback));
}

bool Engine::is_checkpoint_valid() {
  return checkpoint_valid;
}
/*
这个shared_ptr是一个thread_local指针，其指向每个线程的ready_queue，在执行之前，
应该通过每个对应线程中的 Engine::init_local_ready_queue() 调用对其进行初始化。
另外，GraphTask 也有一个 CPU queue 成员变量 cpu_ready_queue_，
专用于处理反向传播相关CPU工作。
init_local_ready_queue 代码有两个执行路径：
    主线程执行路径 ：参数 ready_queue 没有配置，则此时 Engine::execute 是全新调用，则应该在CPU设备上启动。所以需要在CPU上初始化一个新的线程本地就绪队列或重用现有的线程本地就绪队列（比如可重入的后向传播），然后在工作线程的 local_ready_queue 之上保存。这就通过如下代码完成，此时 local_ready_queue 是主线程的线程本地变量。
    工作线程执行路径 ：参数 ready_queue 有配置，是通过 std::thread t(&Engine::thread_init, this, i, device_ready_queues_[i], true)，这时候 local_ready_queue 就是工作线程的本地变量。
    我们这个阶段介绍的就是主线程执行路径，init_local_ready_queue 调用没有参数，生成工作线程的 local_ready_queue。
*/

//对于主线程，则调用 init_local_ready_queue() 来 初始化local ready_queue。
//因为 init_local_ready_queue 没有传入参数，所以新生成一个queue。
//这个shared_ptr是一个thread_local指针，其指向每个线程的ready_queue，在执行之前，
//应该通过每个对应线程中的 Engine::init_local_ready_queue() 调用对其进行初始化。

//线程会调用 thread_init 进行初始化，这里会：
//配置线程的设备。
//调用 init_local_ready_queue 来初始化局部queue。
//调用 thread_main 作为线程主体函数来执行。
void Engine::init_local_ready_queue(std::shared_ptr<ReadyQueue> ready_queue) {
  if (ready_queue) {
    // if ready_queue provided in the caller, use the caller's ready_queue to
    // initialize local_ready_queue
    // // 工作线程执行路径
    local_ready_queue = std::move(ready_queue);
  } else if (!local_ready_queue) {
    // otherwise if local_ready_queue not allocated, allocate a new ready_queue
    // // 主线程执行路径。
    local_ready_queue = std::make_shared<ReadyQueue>();
  }
}

/*
我们具体看看是如何获取当前queue的，这是根据本GraphTask的 CPU queue 和 配置的输入device 一起计算得出的，即：
调用 InputBuffer::device() 从输入获取设定的设备，如果设定了，就使用这个设备，否则使用 CPU。
如果是CPU，就使用 cpu_ready_queue，否则使用 device_ready_queues_。
进一步解析如下：
每个GraphTask都有自己的CPU queue，但是 GPU Queues 被所有GraphTask共享。
*/
// CPU ready queue is per GraphTask, but CUDA device ready queues are shared
// across all graph tasks
auto Engine::ready_queue(
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    at::Device device) -> std::shared_ptr<ReadyQueue> {
  bool multithreading_disabled =
      !c10::AutogradState::get_tls_state().get_multithreading_enabled();
  if (multithreading_disabled || should_run_in_cpu_ready_queue(device.type())) {
    // return the cpu ready queue passed in
    TORCH_INTERNAL_ASSERT(cpu_ready_queue);
    return cpu_ready_queue;
  } else {
    TORCH_INTERNAL_ASSERT(
        0 <= device.index() &&
        device.index() <
            static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
    // See Note [Allocating GPUs to autograd threads]
    return device_ready_queues_.at(device.index());
  }
}

/*
如何获取Ready Queue？具体策略是：
    如果下一个 需要执行的设备是 CPU，则选用cpu_ready_queue。
    否则从device_ready_queues_选取一个GPU对应的 ReadyQueue。
*/
//因为是使用 vector 来管理queue，所以可以使用设备号（device index）去vector里得到每个device专属的ReadyQueue。
auto Engine::ready_queue_by_index(
    std::shared_ptr<ReadyQueue> cpu_ready_queue,
    int device_index) -> std::shared_ptr<ReadyQueue> {
  if (device_index == CPU_DEVICE) {
    // return the cpu ready queue passed in
    TORCH_INTERNAL_ASSERT(cpu_ready_queue);
    return cpu_ready_queue;
  } else {
    TORCH_INTERNAL_ASSERT(
        0 <= device_index &&
        device_index <
            static_cast<c10::DeviceIndex>(device_ready_queues_.size()));
    // See Note [Allocating GPUs to autograd threads]
    // NB: This function would become obsolete if we truly allocated a CPU
    // thread per device, rather than colocate.
    return device_ready_queues_.at(device_index);
  }
}

/*
对于可重入的向后调用，如果由于达到最大深度而从当前线程生成新线程，
则新线程将与父线程重用相同的ReadyQueue以提高性能。
对于工作线程，其对应的 ReadyQueue 是 device_ready_queues_ 之中对应的 queue，
比如下面是用 std::thread t(&Engine::thread_init, this, i, device_ready_queues_[i], true) 来初始化。
除了初始化时候调用 set_device(CPU_DEVICE)，在Engine::thread_init也会调用，
就是启动设备线程时候用到的，设置了设备序列号。
这个序列号就可以和 ReadyQueue 对应。

start_device_threads 用来启动设备线程，设备线程数目与设备数目相关，这样 NUM_DEVICES 个线程在后台一起处理 GraphTask 中的任务。
    使用deviceCount得到设备数量 num_devices。
    然后根据设备的数量来决定要启动的设备线程数量。
    创建多个ReadyQueue，ReadyQueue数目和工作线程数目一样。这些ReadyQueue 在 Engine 的 device_ready_queues_ 之上被管理。
    device_ready_queues_ 就是管理GPU。
    创建设备线程。每个线程都是std::thread，构建时候，把对应的ReadyQueue，就是device_ready_queues_[i] 和 Engine（整个进程生命周期只有一个Engine实例）都传递进去。
    这样Queue可以依靠Engine对于device_ready_queues_的共享来实现线程间工作对象传输。
    作为对比，GraphTask 专门有一个ReadyQueue（cpu_ready_queue_）是用来跑CPU相关工作线程。
    因为CPU工作线程专门用来处理反向传播的CPU工作。
    当 GraphTask 在某一个GPU之上的工作结束了，下一个 NodeTask 应该切换到 CPU 之上，
    所以GraphTask需要记住自己的cpu_ready_queue_ ，从而给 cpu_ready_queue_ 发送消息。
    注意，cpu_ready_queue_ 这是 GraphTask 专有的 ReadyQueue。

    设备线程启动之后，都使用 wait 阻塞在自己对应的 ReadyQueue 之中，
    主线程或者其他worker线程通过对 某一个设备线程的ReadyQueue 采用 push(NodeTask) 操作来唤醒该设备线程进行工作。
*/
auto Engine::start_device_threads() -> void {
  // First always initialize the thread pool for re-entrant threads
  thread_pool_shared_ = std::make_shared<ThreadPoolShared>();

  // Second, create special threads for each non-CPU device
  // See Note [Allocating GPUs to autograd threads]
  // // 使用deviceCount得到 设备数量 num_devices。
  c10::DeviceIndex num_devices = 0;
  // 得到设备数量
  for (const auto& impl_atomic : c10::impl::device_guard_impl_registry) {
    auto* impl = impl_atomic.load();
    // Only record the number of devices for device that don't run on the
    // cpu ready queue.
    if (impl && !should_run_in_cpu_ready_queue(impl->type())) {
      num_devices = std::max(num_devices, impl->deviceCount());
    }
  }

  // If there are no device except cpu, no need to create worker threads
  if (num_devices == 0) {
    return;
  }

  // Since we're about to create threads, forking is not possible anymore
  track_bad_autograd_forks();

  // 确定queue数量，并且生成queue
  // allocate one thread for every GPU device (but colocate GPUs of different
  // types), and pre-allocate the device_ready_queues_ to ensure safe reading on
  // it.
  //// 创建多个ReadyQueue，ReadyQueue数目和工作线程数目一样
  device_ready_queues_ = std::vector<std::shared_ptr<ReadyQueue>>(num_devices);
  for (auto& queue : device_ready_queues_) {
    queue = std::make_shared<ReadyQueue>();
  }

  // 生成线程 // 创建设备线程
  for (const auto i : c10::irange(num_devices)) {
    std::thread t(&Engine::thread_init, this, i, device_ready_queues_[i], true);
    t.detach(); // 让工作线程独立运行
  }
  // Wait for the threads to start
  {
    std::unique_lock<std::mutex> lk(non_reentrant_device_thread_mutex_);
    while (non_reentrant_device_thread_count_.load() !=
           static_cast<uint32_t>(num_devices)) {
      non_reentrant_device_thread_condvar_.wait(lk);
    }
  }
}
/*
add_thread_pool_task 代码如下。
    这里判断是否 graphtask 队列达到最大值，如果没有达到，则建立一个新线程。
    把 graph_task 放入队列 graphtasks_queue_。
    新线程的执行函数是 reentrant_thread_init，其会等待在 thread_pool_shared_->work_ 之上。
    这里会 thread_pool_shared_->work_.notify_one() 让新线程运行。
*/
void Engine::add_thread_pool_task(const std::weak_ptr<GraphTask>& graph_task) {
  std::unique_lock<std::mutex> lck(thread_pool_shared_->mutex_);
  // There may already be some items on the graphtasks_queue_ added by other
  // threads but not enough workers to get to the new task that will be
  // added
  bool create_thread =
      (thread_pool_shared_->num_workers_ <=
       thread_pool_shared_->graphtasks_queue_.size());
  thread_pool_shared_->graphtasks_queue_.push(graph_task);
  // Don't need to be holding the lock while actually creating the thread
  lck.unlock();
  if (create_thread) {
    // If we're creating a new thread, forking is not allowed anymore
    track_bad_autograd_forks();
    std::thread t(&Engine::reentrant_thread_init, this);
    t.detach();
  }
  // This works even if new thread is created because wait() will test the
  // predicate before waiting
  thread_pool_shared_->work_.notify_one();
}

// Remembers current streams on all devices where a context has been created.
// Only called if Engine::execute detects at least one node runs on a cuda
// stream.
void GraphTask::stash_current_streams() {
  const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
  auto num_gpus = guard.deviceCount();
  caller_current_streams_.resize(num_gpus);
  if (num_gpus > 0) {
    for (c10::DeviceIndex idx = 0; idx < num_gpus; idx++) {
#if defined(USE_ROCM) && (ROCM_VERSION < 50000)
      // If the build targets ROCM, stash streams for all visible devices
      // unconditionally, to work around
      // https://github.com/pytorch/pytorch/issues/59750.
      // TODO: Remove ROCM-specific behavior when
      // https://github.com/pytorch/pytorch/issues/59750 is fixed.
      if (true) {
#else
      if (at::detail::getCUDAHooks().hasPrimaryContext(idx)) {
#endif
        caller_current_streams_[idx] =
            guard.getStream({c10::DeviceType::CUDA, idx});
      } else {
        caller_current_streams_[idx] = c10::nullopt;
      }
    }
  }
}

/*
GraphTask.exec_info_ 的作用就是给 GraphTask 的每一个 Node 配置一个 ExecInfo，就是该 Node 的执行信息。
如果exec_info_为空，说明该task运行在默认模式，即，所有遇到的 next_edges 都需要执行。
如果 exec_info_ 非空，说明只有特定 functions 才会被执行，这些 Functions 的特点是：拥有 entry，并且这个 entry 的 “has needed == True”。
exec_info_ 何时为空？何时非空？
当图被用 .backward() 执行，并且没有传递输入参数，则 exec_info 为空。
如果只是使用用 .grad() 执行，或者使用.backward() 执行时候并且给定输入参数，那么 exec_info_ 非空。
所以，exec 和 captured_vars_ 就是针对 grad() 和指定参数的 backward()，就是标注在这种情况下需要计算哪些梯度。
在这种情况下，只有某些节点需要执行，从这些节点开始，有一条路径通向 outpus。
init_to_execute 的作用是给 exec_info 填充数据，目的是对于应该执行的节点，设置其成员变量exec_info[node].needed_ = true。
只有某些特定节点会得到执行，这些节点有一条出边，出边的另一端在“outputs”之中。
其主要算法逻辑为：使用递归思路来填充 exec_info，但是对于实际代码是使用iterative方式进行。
在iterative版本中，当你操作当前节点时候，在你所有孩子节点被更新之后，你需要更新你父亲节点。
*/
void GraphTask::init_to_execute(
    Node& graph_root,
    const edge_list& outputs,
    bool accumulate_grad,
    uint64_t min_topo_nr) {
  // Populates exec_info so nodes that should be executed have
  // `exec_info[node].needed_ = true` Only nodes that have a path to any edge in
  // `outputs` should be executed. The code below populates exec_info using
  // recursion, but the actual code does this iteratively. Refer to the
  // numbering to see how the actual code corresponds. A difference to note is
  // that in the iterative version, when you are working with the current Node,
  // you are reponsible to update your parent's is_needed after all your
  // children have been updated.
  //
  // is_needed = {fn: True for fn in outputs}             # (0)
  // seen = {}
  // def compute_is_needed(fn):
  //   for next_edge in fn.next_edges:
  //     child_fn = next_edge.fn
  //     if child_fn in seen and is_needed[child_fn]:     # (1)
  //       is_needed[fn] = true
  //     else:
  //       seen.add(child_fn)
  //       if compute_is_needed(child_fn):
  //         is_needed[fn] = true                         # (2)
  //                                                      # (3) exit for-loop
  //   return is_needed[fn]
  // compute_is_needed(graph_root)
  //
  // NB: you might be wondering why we don't populate `seen` with outputs. We
  // cannot because in the case where two outputs lie on the same path, we still
  // need to explore past the first output or we would miss the nodes that are
  // required to compute the second output.
  /*从其注释可知，其作用是：填充exec_info，以便应执行的节点具有exec_info[node].needed_ = true 。
    只具特定节点才应该执行，这些节点的性质是：节点拥有一条路径，这路径可以通往outputs的任何一条边。
    下面的代码使用递归填充exec_info，但实际代码以迭代方式执行此操作。
    关键代码如下，就是插入ExecInfo信息 exec_info_.emplace(stack.back().fn_, ExecInfo());。具体删减版代码如下：
  */

  // 这一段就是针对 grad() API 进行处理，只有在所求梯度的张量路径上的其他张量才会被计算梯度
  int output_idx = 0;
  for (auto& output_edge : outputs) {  // 遍历输出边
    // (0) `is_needed` above corresponds to `exec_info_[fn].needed_`
    Node* output = output_edge.function.get();
    auto& info = exec_info_[output];
    if (accumulate_grad) {
      // if called through `.backward()` we directly set `needed_` for all the
      // outputs to true
      info.needed_ = true;
    } else {
      // otherwise it is `.grad()` and we set exec_info[fn].captures_ instead
      // In terms of populating the rest of exec_info though, you can basically
      // think of this as the same as setting `needed_` is true directly.
      if (!info.captures_) {
        info.captures_ = make_unique<std::vector<ExecInfo::Capture>>();
      }
      // 第 i 个输入对应的输出
      info.captures_->emplace_back(output_edge.input_nr, output_idx++);
    }
  }
  captured_vars_.resize(output_idx);

  struct Frame {
    Frame(Node* fn) : fn_(fn) {}
    Node* fn_{};
    size_t next_next_fn_{};

    Node* get_next_fn() {
      const auto& next = fn_->next_edges();
      auto num_next = next.size();
      while (next_next_fn_ < num_next) {
        auto fn = next[next_next_fn_++].function.get();
        if (fn)
          return fn;
      }
      return nullptr;
    }
  };

  auto nodeShouldExecute = [this](Node* fn) {
    auto it = exec_info_.find(fn);
    return it != exec_info_.end() && it->second.should_execute();
  };

  std::vector<Frame> stack;
  std::unordered_set<Node*> seen;
  stack.emplace_back(&graph_root);
  exec_info_.emplace(stack.back().fn_, ExecInfo()); // 这里会初始化 exec_info_，有多个 exec_info

  while (!stack.empty()) {
    auto& frame = stack.back();
    const auto fn = frame.fn_;

    Node* child_fn = nullptr;
    while ((child_fn = frame.get_next_fn()) && !seen.emplace(child_fn).second) {
      // (1) next child exists AND has already been seen
      if (nodeShouldExecute(child_fn)) {
        exec_info_[fn].needed_ = true;
      }
    }

    if (child_fn) {
      // (2) next child exists but has not been seen
      if (child_fn->topological_nr() < min_topo_nr) {
        // child created before the first output means this child cannot have
        // an edge to output
        continue;
      }
      stack.emplace_back(child_fn);
    } else {
      // (3) no next child exists for `fn` means its `needed` has already been
      // finalized. pop stack and update parent
      stack.pop_back();
      if (nodeShouldExecute(fn) && !stack.empty()) {
        exec_info_[stack.back().fn_].needed_ = true;
      }
    }
  }
}

} // namespace autograd
} // namespace torch
