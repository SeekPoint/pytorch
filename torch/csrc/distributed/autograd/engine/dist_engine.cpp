#include <queue>

#include <ATen/Parallel.h>
#include <c10/core/Event.h>
#include <c10/util/DeadlockDetection.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;
using torch::autograd::edge_list;
using torch::autograd::Engine;
using torch::autograd::GraphRoot;
using torch::autograd::GraphTask;
using torch::autograd::GraphTaskGuard;
using torch::autograd::InputBuffer;
using torch::autograd::Node;
using torch::autograd::NodeTask;
using torch::autograd::ReadyQueue;
using torch::autograd::validate_outputs;
using torch::autograd::variable_list;

static constexpr char* kNumBackwardPasses = "num_current_backward_passes";
static constexpr char* kNumAutogradContexts = "num_autograd_contexts";

// This hook does 3 things:
//   1. Call pre hooks of the original AccumulateGrad to modify the input grad.
//   2. Accumuate the guard to RPC context.
//   3. Call post hooks of the original AccumulateGrad.
class DistAccumulateGradCaptureHook
    : public GraphTask::ExecInfo::Capture::GradCaptureHook {
 public:
  DistAccumulateGradCaptureHook(
      std::shared_ptr<AccumulateGrad> accumulateGrad,
      ContextPtr autogradContext)
      : accumulateGrad_(std::move(accumulateGrad)),
        autogradContext_(std::move(autogradContext)) {}

  at::Tensor operator()(const at::Tensor& grad) override {
    ThreadLocalDistAutogradContext contextGuard{ContextPtr(autogradContext_)};
    variable_list inputGrads = {grad};
    // It's intended that pre/post hooks are still called even if the grad is
    // undefined here.
    for (const auto& hook : accumulateGrad_->pre_hooks()) {
      inputGrads = (*hook)(inputGrads);
    }
    // It is possible that the grad is not defined since a separate
    // invocation of the autograd engine on the same node might actually
    // compute this gradient.
    if (inputGrads[0].defined()) {
      // There are 3 internal references to 'inputGrads[0]' at this moment:
      //   1. 'inputGrads[0]' in this function.
      //   2. 'graph_task->captured_vars_' on the callsite in the local engine.
      //   3. 'InputBuffer& inputs' on the callsite as the inputs of the
      //   function node.
      autogradContext_->accumulateGrad(
          accumulateGrad_->variable, inputGrads[0], 3 /* num_expected_refs */);
    }
    const variable_list kEmptyOuput;
    for (const auto& hook : accumulateGrad_->post_hooks()) {
      (*hook)(kEmptyOuput, inputGrads);
    }
    return inputGrads[0];
  }

 private:
  std::shared_ptr<AccumulateGrad> accumulateGrad_;
  ContextPtr autogradContext_;
};

//globalCpuThread 是工作线程，其就是从 ready queue 里面弹出 NodeTask，然后执行。
void DistEngine::globalCpuThread(
    const std::shared_ptr<ReadyQueue>& ready_queue) {
  while (true) {
    NodeTask task = ready_queue->pop();
    if (task.isShutdownTask_) {
      // Need to shutdown this thread.
      C10_LOG_API_USAGE_ONCE("torch.autograd.thread_shutdown");
      break;
    }

    auto graphTask = task.base_.lock();
    if (graphTask == nullptr) {
      // GraphTask has expired, ignore and continue processing.
      continue;
    }

    // Launch the execution on a JIT thread.
    at::launch([this,
                graphTask,
                graphRoot = task.fn_,
                variables =
                    InputBuffer::variables(std::move(task.inputs_))]() mutable {
      InputBuffer inputs(variables.size());
      for (const auto i : c10::irange(variables.size())) {
        inputs.add(i, std::move(variables[i]), c10::nullopt, c10::nullopt);
      }
      execute_graph_task_until_ready_queue_empty(
          /*node_task*/ NodeTask(graphTask, graphRoot, std::move(inputs)),
          /*incrementOutstandingTasks*/ false);
    });
  }
}

DistEngine::DistEngine()
    : initializedContextIds_(),
      engine_(Engine::get_default_engine()),
      global_cpu_ready_queue_(std::make_shared<ReadyQueue>()),  // 这里构建了
      global_cpu_thread_(  // 这里构建了
          &DistEngine::globalCpuThread,
          this,
          global_cpu_ready_queue_) {
  // Note [GPU to CPU continuations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  以下是 GPU to CPU continuations 的翻译和理解。
//
//Continuations 最初应该是在schema语言里面接触过的，后来也看过不少语言用到，这个概念没有找到一个很好的延续概念，暂时使用"延续"这个翻译。
//
//为了执行GPU任务的延续（continuations），所以需要初始化一个单独的CPU线程来处理。分布式引擎的多线程结构仅适用于CPU任务。如果我们有CPU->GPU->CPU这样的任务顺序，分布式 autograd 就没有线程来执行最后一个CPU任务。为了解决这个问题，我们引入了一个全局CPU线程来处理这种情况，它将负责执行这些CPU任务。
//
//CPU线程有自己的就绪队列（ready_queue），它用作DistEngine的所有GraphTask的CPU就绪队列（cpu_ready_queue）。这确保所有GPU到CPU的延续（continuations）都在此线程上排队。全局CPU线程只需将任务从全局队列中取出，并在JIT线程上调用"execute_graph_task_until_ready_queue_empty"，以执行相应的任务。
  // Initialize a single CPU thread to execute continuations from GPU
  // tasks. The multithreaded structure for the distributed engine works
  // well only for CPU tasks. If we have an order of tasks like
  // CPU->GPU->CPU, distributed autograd has no thread to execute the last
  // CPU task on. To fix this, we introduce a global CPU thread to handle
  // such situations and it will be responsible for executing these CPU
  // tasks. The CPU thread has its own ready_queue which is used as the
  // cpu_ready_queue for all GraphTasks for DistEngine. This ensures all GPU
  // to CPU continuations are enqueued on this thread. The global CPU thread
  // simply dequeues tasks from the global queue and calls
  // "execute_graph_task_until_ready_queue_empty" on a JIT thread to execute the
  // appropriate task.
  global_cpu_thread_.detach();  // detach之后就独立运行了
}

//析构函数之中有如下，就是为了引擎结束而做对这两个成员变量做了相关操作。
DistEngine::~DistEngine() {
  // Ensure we shutdown the CPU thread.
  TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP();
  global_cpu_ready_queue_->pushShutdownTask();
  global_cpu_thread_.join();
}

//引擎使用了单例模式，这样每个 worker 之中就只有一个单例在运行。
DistEngine& DistEngine::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static DistEngine* engine = new DistEngine();
  return *engine;
}

/*
我们接下来看看如何做验证工作。

validateRootsAndRetrieveEdges 被用来验证节点和边的有效性，具体逻辑是：

    验证根节点的有效性，获取根节点的边。
    看看根节点是否为空。
    根节点是否需要计算梯度。
    根节点是否有梯度函数。
    计算梯度的边，生成相应的梯度。
    调用 validate_outputs 来验证输出。
*/
void DistEngine::validateRootsAndRetrieveEdges(
    const variable_list& roots,
    edge_list& rootEdges,
    variable_list& grads) {
  TORCH_CHECK(!roots.empty(), "No tensors provided for gradient computation.");
  TORCH_INTERNAL_ASSERT(rootEdges.empty());
  TORCH_INTERNAL_ASSERT(grads.empty());

  // Verify roots are all scalar and require gradients.
  for (const auto& root : roots) {
    TORCH_CHECK(root.requires_grad(), "requires_grad not set on root");
    TORCH_CHECK(
        root.numel() == 1,   // python numel()函数：返回数组中元素的个数
        root.name(),
        " is not a scalar, all roots need to be scalar");
    TORCH_CHECK(
        root.grad_fn(),
        root.name(),
        " does not have a valid gradient function.");

    // Compute the root edges and generate the appropriate gradients.
    rootEdges.push_back(torch::autograd::impl::gradient_edge(root));
    grads.push_back(at::ones_like(root, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  // Validate rootEdges and grads.
  validate_outputs(
      rootEdges, grads, [](const std::string& msg) { return msg; });
}

/*
ExecInfo.Capture.GradCaptureHook 是要对梯度再做后续处理。
但是这个使用却是主要在分布式状态下，是因为分布式引擎有一个累积梯度的需要，
这个必须在正常梯度操作之后的后置处理中完成。

在哪里往 global_cpu_ready_queue_ 插入？
在 DistEngine::computeDependencies 里面会有插入。
首先，每个 GraphTask 都把 global_cpu_ready_queue_ 设置为 cpu_ready_queue。GraphTask构造函数这里参数在调用时候传入的是 global_cpu_ready_queue_。


所以，如果 GraphTask 最后返回需要 CPU 运行时候，就统一用这个。


计算依赖分为两大部分，第一部分是做准备工作，第二部分是计算依赖关系，第三部分是根据依赖关系来得到需要计算哪些函数。
*/
void DistEngine::computeDependencies(
    const ContextPtr& autogradContext,
    const edge_list& rootEdges,
    const variable_list& grads,
    const std::shared_ptr<Node>& graphRoot,
    edge_list& outputEdges,
    bool retainGraph) {
  TORCH_INTERNAL_ASSERT(graphRoot, "graphRoot is null!");

  // Store root nodes so we can traverse through the graph later
  // e.g., for get_current_graph_task_execution_order
  c10::SmallVector<Node*, 4> temp_roots{rootEdges.size()};
  for (const auto i : c10::irange(rootEdges.size())) {
    temp_roots[i] = rootEdges[i].function.get();
  }

  // 第一部分，准备工作
  // 1. 生成一个GraphTask
  // Build the graph task and graph root.
  // NOTE: we don't need to build and pass a cpu_ready_queue to GraphTask
  // as we use execute_graph_task_until_ready_queue_empty, which will build
  // a separate ReadyQueue for each call.
  // 不需要给 GraphTask 传一个cpu_ready_queue，因为我们后面使用execute_graph_task_until_ready_queue_empty，在那里会给每一个调用建立一个独立的ReadyQueue
  auto graphTask = std::make_shared<GraphTask>(
      /* keep_graph */ retainGraph,
      /* create_graph */ false,
      /* depth */ 0,
      /* cpu_ready_queue */ global_cpu_ready_queue_,
      /* graph_roots */ temp_roots,
      /* exit_on_error */ true);

  // Run BFS to traverse the graph locally. The roots of the graph are
  // GraphRoot and all send functions for this autograd context.
  std::unordered_set<Node*> seen; // 记录已经访问过的节点
  std::queue<Node*> queue; // 一个 Node 类型的 queue
  queue.push(static_cast<Node*>(graphRoot.get())); // 插入根对应的Node

  auto sendFunctions = autogradContext->sendFunctions();

  // 2. 获取出边列表
  // Add all the send functions to the queue as roots.
  // 普通状态下，root节点内在反向传播时候，已经有了next edges，但是分布式模式下，出边是在sendFunctions之中
  // Add all the send functions to the queue as roots.
  for (const auto& mapEntry : sendFunctions) {   // sendFunctions就是出边，之前在 addSendFunction之中被添加
    // Increment 'outstanding_tasks_' for GraphTask for each send_function
    // since we want the local autograd engine to wait for all of them.
    graphTask->outstanding_tasks_++;  // 出边增加
    queue.push(mapEntry.second.get());  // 后续用queue来处理，插入的是 SendRpcBackward
  }

  bool might_use_cuda = at::globalContext().hasCUDA();
  bool will_use_cuda = false;

// 第二部分，遍历图，计算依赖关系，此时 queue 里面是 root 和 若干 SendRpcBackward
  edge_list recvBackwardEdges;
  // Traverse the graph.
  auto& dependencies = graphTask->dependencies_;  // 获取依赖关系
  while (!queue.empty()) {  // 遍历所有发送边
    auto fn = queue.front();  // 得到发送边
    queue.pop();

    if (might_use_cuda && !will_use_cuda) {
      will_use_cuda = fn->stream(c10::DeviceType::CUDA).has_value();
    }

    for (const auto& edge : fn->next_edges()) {  // 遍历Node（根节点或者SendRpcBackward）的next_edges
      if (auto nextFn = edge.function.get()) {  // 得到一个边
        dependencies[nextFn] += 1;  // 对应的节点依赖度加一
        const bool wasInserted = seen.insert(nextFn).second;  // 是否已经访问过
        if (wasInserted) {  // 如果true，是插入了，就说明之前没有访问过，否则插不进去，是false
          // Seeing this function for the first time.
          queue.push(nextFn);  // 既然之前没有访问过，就插入到queue

          if (nextFn->next_edges().empty()) {  // 如果这个边本身没有输出边，说明是叶子节点
            TORCH_INTERNAL_ASSERT(
                dynamic_cast<AccumulateGrad*>(nextFn) ||
                dynamic_cast<RecvRpcBackward*>(nextFn));  // 叶子节点有两种
            // We have found a leaf node which should be either AccumulateGrad
            // or RecvRpcBackward. Record the function
            // to ensure we don't execute it and instead accumulate the grads on
            // the autograd context. These functions would be passed in as the
            // 'outputs' parameter of the vanilla autograd engine.

            // We don't accumulate any grads in the context for RecvRpcBackward.
            // RecvRpcBackward is added as an output edge to indicate it is a
            // leaf node and this helps in properly computing dependencies for
            // the local autograd graph. Putting RecvRpcBackward in
            // 'outputEdges' means that this function needs to be executed
            // (inline with our assumption for FAST mode that all send/recv
            // functions are valid in the backward pass), and as a result all of
            //  its ancestors need to be executed as well.
            if (dynamic_cast<RecvRpcBackward*>(nextFn)) {
              recvBackwardEdges.emplace_back(edge);   // 特殊处理
            }
            outputEdges.emplace_back(edge);  // 最终输出边
          }
        }
      }
    }
  }

  if (will_use_cuda) {
    // Collects current streams for devices where this process has a context,
    // so graphTask::exec_post_processing can sync them with leaf_streams.
    graphTask->stash_current_streams();
  }

  // 此时，recvBackwardEdges 里面是RecvRpcBackward，outputEdges 里面是 AccumulateGrad

  // 以下是第三部分，根据依赖关系找到需要计算那些functions

  // Now lets compute which functions need to be executed. The algorithm is as
  // follows:
  // 1. Create a dummy GraphRoot which points to all 'send' functions for this
  //    context and the original graphRoot. Run 'init_to_execute' with the
  //    outputEdges and the dummy GraphRoot. This ensures we mark
  //    appropriate functions as needed if they are reachable only from a
  //    specific 'send' function locally and not necessarily from the provided
  //    roots.
  // 2. For all edges in 'outputEdges' which point to 'RecvRpcBackward', mark
  //    those functions as needed for execution. The reason for this is that
  //    'init_to_execute', will mark these as not needed. But 'RecvRpcBackward'
  //    is unique in the sense that we use it as a leaf node in graph to compute
  //    needed execution accurately, but unlike AccumulateGrad, we do need to
  //    execute this function.
  if (!outputEdges.empty()) {
    // Compute 'needed execution' starting from all 'send' functions and the
    // original graphRoot.
    edge_list edges;
    // Create some dummy edges (input_nr not important for init_to_execute).
    for (const auto& mapEntry : sendFunctions) {  // 遍历
      edges.emplace_back(mapEntry.second, 0);   // 得到出边列表
    }

    // Add the original graphRoot as an edge.
    edges.emplace_back(graphRoot, 0);  // root也加入出边列表

    // Create a dummy GraphRoot and run init_to_execute with it.
    GraphRoot dummyRoot(edges, {});  // 建立一个虚拟Root

    // 如果出边不为空，则会调用 init_to_execute  对GraphTask进行初始化
    graphTask->init_to_execute(
        dummyRoot, outputEdges, /*accumulate_grad=*/false, /*min_topo_nr=*/0);

    // exec_info_ 的数据结构是std::unordered_map<Node*, ExecInfo>
    for (auto& mapEntry : graphTask->exec_info_) {
      auto& execInfo = mapEntry.second;
      if (!execInfo.captures_) {  // 看看此张量是否在所求梯度的张量路径上
        continue;  // 如果不在路径之上，就跳到下一个张量
      }
      auto fn = mapEntry.first;
      // There may be nodes other than 'AccumulateGrad', e.g. RecvRPCBackward,
      // to be captured.
      if (auto accumulateGradFn = dynamic_cast<AccumulateGrad*>(fn)) {
        // 如果是叶子节点
        for (auto& capture : *execInfo.captures_) {  // 遍历张量路径上的节点
          // Capture hooks are technically deprecated, but as an exception below
          // is the single and only instance of capture hooks usage that we
          // support. See NOTE [Deprecated capture hooks] for more context.
          // 在这里添加 hook
          capture.DO_NOT_USE_DEPRECATED_register_capture_hook(
              std::make_unique<DistAccumulateGradCaptureHook>(  // 给张量插入Hook
                  std::dynamic_pointer_cast<AccumulateGrad>(
                      accumulateGradFn->shared_from_this()),
                  autogradContext));
        }
      }
    }

    // Mark all 'RecvRPCBackward' as needing execution.
    // RecvRPCBackward需要执行
    for (const auto& recvBackwardEdge : recvBackwardEdges) {
      graphTask->exec_info_[recvBackwardEdge.function.get()].needed_ = true;
    }
  }

  // Set graph task owner in a single thread since concurrent access to
  // 'owner_' field is not permitted.
  graphTask->owner_ = torch::autograd::CPU_DEVICE;

  // Let autograd context take ownership of the GraphTask.
  // 上下文里面设置了 GraphTask  // 设定在上下文之中
  autogradContext->setGraphTask(std::move(graphTask));
}

void DistEngine::execute_graph_task_until_ready_queue_empty(
    NodeTask&& node_task,
    bool incrementOutstandingTasks) {
  engine_.initialize_device_threads_pool();
  // Create a ready queue per call to traverse the graph_task from
  // root_to_execute This allow concurrent execution of the same GraphTask from
  // different threads
  std::shared_ptr<ReadyQueue> cpu_ready_queue = std::make_shared<ReadyQueue>();
  auto graph_task = node_task.base_.lock();
  if (graph_task == nullptr) {
    LOG(ERROR) << "GraphTask has expired for NodeTask: "
               << node_task.fn_->name() << ", skipping execution.";
    return;
  }

  cpu_ready_queue->push(std::move(node_task), incrementOutstandingTasks);

  torch::autograd::set_device(torch::autograd::CPU_DEVICE);
  while (!cpu_ready_queue->empty()) {
    std::shared_ptr<GraphTask> local_graph_task;
    {
      // Scope this block of execution since NodeTask is not needed after this
      // block and can be deallocated (release any references to grad tensors
      // as part of inputs_)
      NodeTask task = cpu_ready_queue->pop();
      if (!(local_graph_task = task.base_.lock())) {
        continue;
      }
      if (task.fn_ && !local_graph_task->has_error_.load()) {
        at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);
        try {
          GraphTaskGuard guard(local_graph_task);
          engine_.evaluate_function(
              local_graph_task, task.fn_.get(), task.inputs_, cpu_ready_queue);
        } catch (std::exception& e) {
          engine_.thread_on_exception(local_graph_task, task.fn_, e);
          // break the loop in error so that we immediately stop the execution
          // of this GraphTask, mark it completed if necessary and return the
          // future with proper ErrorMessage
          break;
        }
      }
    }
    // Decrement the outstanding task.
    --local_graph_task->outstanding_tasks_;
  }
  // Check if we've completed execution.
  if (graph_task->completed()) {
    // We don't need to explicitly notify the owner thread, since
    // 'mark_as_completed_and_run_post_processing' would mark the Future as
    // completed and this would notify the owner thread that the task has been
    // completed.
    graph_task->mark_as_completed_and_run_post_processing();
  }
}

c10::intrusive_ptr<c10::ivalue::Future> DistEngine::
    runEngineAndAccumulateGradients(
        const ContextPtr& autogradContext,
        const std::shared_ptr<Node>& graphRoot,
        const edge_list& outputEdges,
        bool incrementOutstandingTasks) {
  // Cleanup previous state for outstanding RPCs. Outstanding RPCs could be
  // lingering if we're running backward multiple times and some of the
  // passes ran into errors.
  autogradContext->clearOutstandingRpcs();
  auto graphTask = autogradContext->retrieveGraphTask();
  at::launch([this, graphTask, graphRoot, incrementOutstandingTasks]() {
    execute_graph_task_until_ready_queue_empty(
        /*node_task*/ NodeTask(graphTask, graphRoot, InputBuffer(0)),
        /*incrementOutstandingTasks*/ incrementOutstandingTasks);
  });
  // Use a reference here to avoid refcount bump on futureGrads.
  auto& futureGrads = graphTask->future_result_;

  // Build a future that waits for the callbacks to execute (since callbacks
  // execute after the original future is completed). This ensures we return a
  // future that waits for all gradient accumulation to finish.
  auto accumulateGradFuture =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

  futureGrads->addCallback([autogradContext, outputEdges, accumulateGradFuture](
                               c10::ivalue::Future& futureGrads) {
    if (futureGrads.hasError()) {
      // Don't accumulate gradients if we receive an error.
      // We must add the node information here since DistEngine::execute
      // waits on accumulateGradFuture and will throw an exception once we
      // set the error below.
      std::string errorMsg = c10::str(
          "Error on Node ",
          DistAutogradContainer::getInstance().getWorkerId(),
          ": ",
          futureGrads.tryRetrieveErrorMessage());
      accumulateGradFuture->setError(std::make_exception_ptr(
          c10::ivalue::Future::FutureError(std::move(errorMsg))));
      return;
    }

    try {
      const variable_list& grads = futureGrads.constValue().toTensorVector();
      TORCH_INTERNAL_ASSERT(grads.size() == outputEdges.size());
      accumulateGradFuture->markCompleted(c10::IValue());
    } catch (std::exception& e) {
      accumulateGradFuture->setErrorIfNeeded(std::current_exception());
    }
  });

  return accumulateGradFuture;
}

//executeSendFunctionAsync 就会用 sendFunction->getGrads() 提取梯度，进行操作。
c10::intrusive_ptr<c10::ivalue::Future> DistEngine::executeSendFunctionAsync(
    const ContextPtr& autogradContext,
    const std::shared_ptr<SendRpcBackward>& sendFunction,
    bool retainGraph) {
  // Typically the local autograd engine ensures stream synchronizations between
  // nodes in the graph. However, for distributed autograd the sendFunction
  // inputs might have been retrieved over the wire on a separate stream and the
  // sendFunction itself runs on a different stream. As a result, we need to
  // manually synchronize those two streams here.
  const auto& send_backward_stream =
      sendFunction->stream(c10::DeviceType::CUDA);
  if (send_backward_stream) {
    for (const auto& grad : sendFunction->getGrads()) {   // 这里有获取
      const auto guard = c10::impl::VirtualGuardImpl{c10::DeviceType::CUDA};
      const auto default_stream = guard.getStream(grad.device());
      if (send_backward_stream != default_stream) {
        auto event = c10::Event{c10::DeviceType::CUDA};
        event.record(default_stream);
        send_backward_stream->wait(event);
      }
    }
  }

  std::unique_lock<std::mutex> lock(initializedContextIdsLock_);
  if (initializedContextIds_.find(autogradContext->contextId()) ==
      initializedContextIds_.end()) {
    edge_list outputEdges;
    // Pass in a dummy graphRoot since all send functions are the roots.
    auto dummyRoot = std::make_shared<GraphRoot>(edge_list(), variable_list());
    computeDependencies(
        autogradContext, {}, {}, dummyRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized and unlock.
    initializedContextIds_.insert(autogradContext->contextId());
    lock.unlock();

    // Enqueue the current send function.
    auto graphTask = autogradContext->retrieveGraphTask();
    // Run the autograd engine.
    auto accumulateGradFuture = runEngineAndAccumulateGradients(
        autogradContext,
        sendFunction,
        outputEdges,
        /*incrementOutstandingTasks=*/false);

    // Build the 'uber' future that waits for everything.
    auto callbackFuture =
        c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

    accumulateGradFuture->addCallback(
        [autogradContext,
         callbackFuture](c10::ivalue::Future& accumulateGradFuture) {
          try {
            if (accumulateGradFuture.hasError()) {
              // Perform cleanup at the end of the backward pass (before we mark
              // the future as completed).
              DistEngine::getInstance().cleanupBackwardPass(autogradContext);

              // Skip any further processing on errors.
              callbackFuture->setError(accumulateGradFuture.exception_ptr());
              return;
            }

            // Wait for all RPCs after the autograd engine is done.
            auto rpcFuture =
                autogradContext->clearAndWaitForOutstandingRpcsAsync();
            rpcFuture->addCallback([callbackFuture, autogradContext](
                                       c10::ivalue::Future& rpcFuture) {
              try {
                // Perform cleanup at the end of the backward pass (before
                // we mark the future as completed).
                DistEngine::getInstance().cleanupBackwardPass(autogradContext);
              } catch (std::exception& e) {
                callbackFuture->setErrorIfNeeded(std::current_exception());
                return;
              }

              // Finally mark the 'uber' future as completed.
              if (!rpcFuture.hasError()) {
                callbackFuture->markCompleted(c10::IValue());
              } else {
                callbackFuture->setError(rpcFuture.exception_ptr());
              }
            });
          } catch (std::exception& e) {
            callbackFuture->setErrorIfNeeded(std::current_exception());
          }
        });

    // Return the future which waits for all async processing to be done.
    return callbackFuture;
  } else {
    lock.unlock();
    auto graphTask = autogradContext->retrieveGraphTask();
    at::launch([this, graphTask, sendFunction]() {
      execute_graph_task_until_ready_queue_empty(
          /*node_task*/ NodeTask(graphTask, sendFunction, InputBuffer(0)),
          /*incrementOutstandingTasks*/ false);
    });
    auto fut = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    fut->markCompleted(c10::IValue());
    return fut;
  }
}
/*
总体执行是在 DistEngine::execute 之中完成，具体分为如下步骤：

    使用 contextId 得到前向的上下文。
    使用 validateRootsAndRetrieveEdges 进行验证。
    构造一个GraphRoot，用它来驱动后向传播，可以认为是一个虚拟根。
    使用 computeDependencies 计算依赖。
    使用 runEngineAndAccumulateGradients 进行反向传播计算。
    使用 clearAndWaitForOutstandingRpcsAsync 等待 RPC 完成。
可以看到，与普通引擎相比较，分布式多了一个计算root边和生成边上梯度信息的过程。因为在普通前向传播过程之中，这些是已经配置好的，但是在分布式计算之中，前向传播是没有计算这些，所以需要在反向传播之前计算出来。


4.3 VS 普通 engine
我们和普通引擎进行对比一下校验部分。

普通Engine 之中只调用了 validate_outputs。

因此，对于校验部分，DistEngine 可以总结为：

做校验。
根据 roots 来计算root对应的边和生成对应梯度。
再用validate_outputs验证输出。
*/
void DistEngine::execute(
    int64_t contextId,
    const variable_list& roots,
    bool retainGraph) {
  // Retrieve the context for the given context_id. This will throw if the
  // context_id is invalid.
  auto autogradContext =
      DistAutogradContainer::getInstance().retrieveContext(contextId);

  // Perform initial pre-processing.
  edge_list rootEdges;
  variable_list grads;
  validateRootsAndRetrieveEdges(roots, rootEdges, grads);

  //// 构造一个GraphRoot，用它来驱动后向传播，可以认为是一个虚拟根
  std::shared_ptr<Node> graphRoot =
      std::make_shared<GraphRoot>(rootEdges, grads);
  edge_list outputEdges;
  // Compute dependencies locally, starting from all roots and all 'send'
  // functions.
  {
    std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
    // Context should not have been initialized already.
    TORCH_INTERNAL_ASSERT(
        initializedContextIds_.find(autogradContext->contextId()) ==
        initializedContextIds_.end());
    // 计算依赖
    computeDependencies(
        autogradContext, rootEdges, grads, graphRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized.
    initializedContextIds_.insert(autogradContext->contextId());
  }

  BackwardPassCleanupGuard guard(autogradContext);

  // This needs to be blocking and as a result we wait for the future to
  // complete.
  runEngineAndAccumulateGradients(autogradContext, graphRoot, outputEdges)
      ->waitAndThrow(); // 反向传播计算

  // Wait for all of the outstanding rpcs to complete.
  autogradContext->clearAndWaitForOutstandingRpcsAsync()->waitAndThrow();
}

void DistEngine::cleanupBackwardPass(const ContextPtr& autogradContext) {
  // Validate only the GraphTask is holding a reference to the Future
  // which holds gradients for the backward pass. This ensures that
  // after 'resetGraphTask' is called below, there are no remaining
  // references left to the gradients for the backward pass.
  //
  // This ensures our 'use_count' checks in
  // AccumulateGrad::accumulateGrad are correct and we're
  // not leaking any references to the gradients anywhere else.
  const auto& futureGrads =
      autogradContext->retrieveGraphTask()->future_result_;
  TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1);

  // Reset the graph task once we're done with all processing.
  autogradContext->resetGraphTask();

  // Clear any outstanding rpcs.
  autogradContext->clearOutstandingRpcs();

  // Clear the context id once we're done with the autograd engine
  // processing.
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  initializedContextIds_.erase(autogradContext->contextId());
}

size_t DistEngine::numBackwardPasses() const {
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  return initializedContextIds_.size();
}

std::unordered_map<std::string, int> DistEngine::getDebugInfo() const {
  std::unordered_map<std::string, int> debugInfo;
  debugInfo[kNumBackwardPasses] = numBackwardPasses();
  debugInfo[kNumAutogradContexts] =
      DistAutogradContainer::getInstance().numAutogradContexts();
  return debugInfo;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
