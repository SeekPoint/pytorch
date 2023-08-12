#include <ATen/ThreadLocalState.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::distributed::autograd::AutogradMetadata;
using torch::distributed::autograd::RpcWithAutograd;
using torch::distributed::rpc::JitFuture;
using torch::distributed::rpc::Message;
using torch::distributed::rpc::MessageType;
using torch::distributed::rpc::RpcAgent;
using torch::distributed::rpc::WorkerInfo;

// addSendRpcBackward 就被传入当前上下文之中，后续反向传播时候，会取出这个 addSendRpcBackward。
/*
0x05 前向传播交互过程
前面的分享过程还是简略，我们接下来把完整的发送/接受过程详细分析一下。

5.1 发送
这里对应设计中的如下文字：

在前向传播期间，我们在上下文中存储每个 autograd 传播的send和recv函数。这确保我们在 autograd 图中保存对适当节点的引用以使其保持活动状态。除此之外，这也使得在后向传播期间很容易查找到对应的send和recv函数。

5.1.1 发送逻辑
代码逻辑如下：

生成一个 grad_fn，其类型是 SendRpcBackward。
调用 collect_next_edges 和 set_next_edges 为 SendRpcBackward 添加后续边，这些函数我们在前面系列中有分析。
调用 add_input_metadata 添加输入元数据。
调用 addSendFunction 往上下文添加 grad_fn。
*/
void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors) {
  // Attach autograd information only for tensors requiring grad.
  std::vector<torch::Tensor> tensors_with_grad;
  std::copy_if(
      tensors.begin(),
      tensors.end(),
      std::back_inserter(tensors_with_grad),
      [](const torch::Tensor& t) { return t.requires_grad(); });

  // Attach the appropriate autograd edges.
  auto grad_fn = std::make_shared<SendRpcBackward>();
  grad_fn->set_next_edges(  // 这里会设置其输出边
      torch::autograd::collect_next_edges(tensors_with_grad));

  // Add the appropriate input metadata for the grad_fn.
  for (const auto& tensor : tensors_with_grad) {
    grad_fn->add_input_metadata(tensor);
  }

  // Record the send autograd function in our current context.
  autogradContext->addSendFunction(grad_fn, autogradMetadata.autogradMessageId);
}

// 在 addRecvRpcBackward 之中，会依据传递过来的 autogradMetadata.autogradContextId 来构建一个上下文。
/*这样，发送方和接收方就共享了一个上下文，而且这个上下文的id是全局唯一的。

具体逻辑如下，上方是发送端，下方是接收端。

发送端
利用本地 context_id 构建了 AutogradMetadata，AutogradMetadata含有 ctx_id, msg_id。
利用 AutogradMetadata 构建了 Message。
利用 agent.send 发送了 Message。
接收端：
收到了 Message。
从 Message 之中解析出 AutogradMetadata。
从 AutogradMetadata 提取出 context_id。
利用 context_id 构建了本地的 DistAutogradContext。
发送方和接收方就共享了一个上下文（这个上下文的id是全局唯一的）。
+----------------------------------------------------------------------------------+
| sendMessageWithAutograd                                                          |
|                                                                                  |
|  +----------------------------------------------------------------------------+  |
|  | addSendRpcBackward                                                         |  |
|  |                                                                            |  |
|  |                                                                            |  |
|  |               autogradMetadata = AutogradMetadata(context_id, message_id)  |  |
|  |                          +                                                 |  |
|  |                          |                                                 |  |
|  +----------------------------------------------------------------------------+  |
|                             |                                                    |
|                             v                                                    |
|        agent.send(message(autogradMetadata)                                      |
|                             +                                                    |
|                             |                                                    |
+----------------------------------------------------------------------------------+
                              |
                              |
                              |
                              |                                             Sender
+-----------------------------------------------------------------------------------+
                              |                                             Receiver
                              | message
                              v
                              |
+----------------------------------------------------------------------------------+
| processForwardAutogradReq   |                                                    |
|                             |                                                    |
|                             | message.autogradMetadata                           |
|                             v                                                    |
|  +----------------------------------------------------------------------------+  |
|  | addSendRpcBackward       |                                                 |  |
|  |                          |                                                 |  |
|  |                          +--------------------+                            |  |
|  |                                               |                            |  |
|  |                                               v                            |  |
|  |   autogradContext = getOrCreateContext(autogradMetadata.autogradContextId) |  |
|  |                                                                            |  |
|  |                                                                            |  |
|  +----------------------------------------------------------------------------+  |
|                                                                                  |
+----------------------------------------------------------------------------------+


5.2.3 上下文交互
torch/csrc/distributed/autograd/utils.cpp 之中，addRecvRpcBackward 函数会对上下文进行处理。

这里对应设计中的：

在前向传播期间，我们在上下文中存储每个 autograd 传播的send和recv函数。这确保我们在 autograd 图中保存对适当节点的引用以使其保持活动状态。除此之外，这也使得在向后传播期间很容易查找到对应的send和recv函数。

其具体逻辑是：

根据 rpc信息中的 autogradContextId 拿到本地的上下文。
生成一个 RecvRpcBackward。
用 rpc 信息中的张量来对 RecvRpcBackward 进行配置，包括torch::autograd::set_history(tensor, grad_fn)。
调用 addRecvFunction 把 RecvRpcBackward 加入到上下文。

*/
ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const rpc::DeviceMap& deviceMap) {
  // Initialize autograd context if necessary.
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // 生成或者得到一个上下文，把发送方的 autogradContextId 传入，即利用 autogradContextId 作为key后续可以查找到这个上下文
  auto autogradContext =
      autogradContainer.getOrCreateContext(autogradMetadata.autogradContextId);

  if (!tensors.empty() && torch::autograd::compute_requires_grad(tensors)) {
    // Attach the tensors as inputs to the autograd function.
    auto grad_fn = std::make_shared<RecvRpcBackward>(
        autogradMetadata, autogradContext, fromWorkerId, deviceMap);
    for (auto& tensor : tensors) {
      if (tensor.requires_grad()) {
        torch::autograd::set_history(tensor, grad_fn);
      }
    }

    // Now update the autograd context with the necessary information.
    autogradContext->addRecvFunction(
        grad_fn, autogradMetadata.autogradMessageId);
  }

  return autogradContext;
}

c10::intrusive_ptr<Message> getMessageWithProfiling(
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMessage,
    MessageType msgType,
    torch::autograd::profiler::ProfilerConfig&& profilerConfig) {
  auto& remoteProfilerManager =
      torch::distributed::rpc::RemoteProfilerManager::getInstance();

  auto key = remoteProfilerManager.getCurrentProfilingKey();
  // generate a globally unique Id
  auto globallyUniqueProfilingId = remoteProfilerManager.getNextProfilerId();
  // Save a mapping of ID -> RPC profiling key and unset the current TLS key.
  remoteProfilerManager.saveRPCKey(globallyUniqueProfilingId, key);
  remoteProfilerManager.unsetCurrentKey();
  auto wrappedProfilingMsg = RpcWithProfilingReq(
      msgType,
      std::move(wrappedRpcMessage),
      std::move(profilerConfig),
      globallyUniqueProfilingId);

  return std::move(wrappedProfilingMsg).toMessage();
}

//而 getMessageWithAutograd 会与上下文交互，其代码位于 torch/csrc/distributed/autograd/utils.cpp。
/*
4.4 如何共享上下文
具体使用中，在with语句中生成的context_id可以用作在所有 worker 之上唯一标识一个分布式后向传播（包括前向传播和后向传播）。每个worker存储与此 context_id关联的元数据，这是正确执行分布式自动加载过程所必需的。

因为需要在多个 worker 之中都存储这个 context_id关联的元数据，所以就需要一个 封装/发送/接受的机制来在 worker 之间传递这个元数据，封装机制就是我们前面提到的 AutogradMetadata。我们接下来看看如何发送/接受上下文元信息。

4.4.1 发送方
当发送消息时候，getMessageWithAutograd 会使用 autogradContainer.currentContext() 获取当前上下文，进行发送。
*/
c10::intrusive_ptr<Message> getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMsg,
    MessageType msgType,
    bool forceGradRecording,
    const rpc::DeviceMap& deviceMap) {
  // 获取到 DistAutogradContainer
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // If there is no valid context and no tensor requires grads, send original
  // rpc message. otherwise, attach grad info and grad functions and send
  // rpcWithAutograd message.
  auto tensorsRequireGrad =
      torch::autograd::compute_requires_grad(wrappedRpcMsg->tensors());
  if (!autogradContainer.hasValidContext() ||
      (!forceGradRecording && !tensorsRequireGrad)) {
    return wrappedRpcMsg;
  }

  // Retrieve the appropriate context to modify.
  // 获取到上下文，每个worker都有自己的上下文
  auto autogradContext = autogradContainer.currentContext(); // 获取当前上下文

  // Wrap the original rpc with autograd information.
  // newAutogradMessageId 会生成一个messageID
  AutogradMetadata autogradMetadata(  // 构建了 AutogradMetadata  // 使用上下文id和消息id来构建元数据
      autogradContext->contextId(), autogradContainer.newAutogradMessageId());
  auto rpcWithAutograd = std::make_unique<RpcWithAutograd>(
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_,
      msgType,
      autogradMetadata,
      std::move(wrappedRpcMsg),
      deviceMap);

  if (tensorsRequireGrad) {
    // Record autograd information for 'send'.
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd->tensors());
  }
  // Record the workerID
  autogradContext->addKnownWorkerId(dstId);

  return std::move(*rpcWithAutograd).toMessage();  // 最终构建了一个message
}


'''
我们之前的图现在可以拓展，加入了上下文ID。

+----------------------------------------------------------------------------------------+
| worker                                                                                 |
|                  +------------------------------------------+                          |
|                  |DistAutogradContainer                     |                          |
|          init()  |                                          |                          |
|  rank +-------------+----> worker_id_                       |                          |
|                  |  |                                       |                          |
|                  |  +----> next_context_id_+-------------+  |                          |
|                  |  |                                    |  |                          |
|                  |  +----> next_autograd_message_id_ +----------------------+          |
|                  |                                       |  |               |          |
|                  |                                       |  |               |          |
|                  +------------------------------------------+               |          |
|                                                          |                  |          |
|                                                          |                  |          |
|                                                          |                  |          |
|                  +------------------------------------------------------------------+  |
|                  |getMessageWithAutograd                 |                  |       |  |
|                  |                                       |                  |       |  |
|                  |                                       v                  v       |  |
|                  |                                                                  |  |
|                  |    AutogradMetadata autogradMetadata(contextId(), MessageId())   |  |
|                  |                                                                  |  |
|                  |                                                                  |  |
|                  +------------------------------------------------------------------+  |
|                                                                                        |
+----------------------------------------------------------------------------------------+
'''

/*
在 torch/csrc/distributed/autograd/utils.cpp 这里利用 agent 来进行发送 FORWARD_AUTOGRAD_REQ。

后面在接收方，我们将会看到处理 FORWARD_AUTOGRAD_REQ 消息，因此发送和接受大致可以联系起来。



发送流程如下，其中 sendMessageWithAutograd 会使用 RpcAgent::getCurrentRpcAgent() 得到 RpcAgent::currentRpcAgent_，就是得到了全局设置的代理，然后通过代理进行发送。

  rpc.rpc_sync
         +
         |
         |
         v
  _invoke_rpc_builtin
         +
         |                                               Python
+---------------------------------------------------------------+
         |                                               C++
         |
         v

    pyRpcBuiltin
         +
         |
         |
         v

 sendMessageWithAutograd(RpcAgent::getCurrentRpcAgent())
         +
         |
         |
         |   RpcAgent::currentRpcAgent_
         |           +
         |           |
         |           |
         |           v
         |     +-----+-----------+
         |     | TensorPipeAgent |        +-----------------------+
         |     |                 |        | RequestCallbackImpl   |
         |     |       cb_ +------------> |                       |
         |     |                 |        +-----------------------+
         |     |                 |
         |     |                 |
         +-----------> send +-----------> Will send message to other worker
               |                 |
               |                 |
               +-----------------+

*/
c10::intrusive_ptr<JitFuture> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMsg,
    bool forceGradRecording,
    const float rpcTimeoutSeconds,
    bool forceDisableProfiling) {
  auto msg = getMessageWithAutograd(  // 这里会与上下文交互，构建了 FORWARD_AUTOGRAD_REQ
      dst.id_,
      std::move(wrappedRpcMsg),
      MessageType::FORWARD_AUTOGRAD_REQ,
      forceGradRecording,
      agent.getDeviceMap(dst));

  // If profiler is enabled, wrap this message with profiling metadata that will
  // tell the remote end to process this request with the profiler enabled.
  if (!forceDisableProfiling) {
    switch (torch::profiler::impl::profilerType()) {
      case torch::profiler::impl::ActiveProfilerType::LEGACY: {
        auto profilerConfig = torch::autograd::profiler::getProfilerConfig();
        auto msgWithProfiling = getMessageWithProfiling(
            std::move(msg),
            rpc::MessageType::RUN_WITH_PROFILING_REQ,  //构建消息
            std::move(profilerConfig));
        // 发送消息
        return agent.send(dst, std::move(msgWithProfiling), rpcTimeoutSeconds);
      }
      case torch::profiler::impl::ActiveProfilerType::KINETO:
        TORCH_WARN_ONCE(
            "Profiling a distributed call with the Kineto profiler will profile "
            "the caller, but not the worker.");
        break;
      default:
        break;
    }
  }
  // 发送消息
  return agent.send(dst, std::move(msg), rpcTimeoutSeconds);
  ;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
