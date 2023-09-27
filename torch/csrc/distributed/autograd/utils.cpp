#include <torch/csrc/autograd/functions/utils.h>
#include <aten/src/ATen/ThreadLocalState.h>
#include <c10/util/ThreadLocalDebugInfo.h>
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
using torch::distributed::rpc::RpcCommandBase;
using torch::distributed::rpc::WorkerInfo;

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
  auto grad_fn = std::make_shared<SendRpcBackward>();  // 构建了 SendRpcBackward
  grad_fn->set_next_edges( // 这里会设置其输出边
      torch::autograd::collect_next_edges(tensors_with_grad));

  // Add the appropriate input metadata for the grad_fn.
  for (const auto& tensor : tensors_with_grad) {
    grad_fn->add_input_metadata(tensor);  // 添加边 SendRpcBackward
  }

  // Record the send autograd function in our current context.
  // 插入到上下文
  autogradContext->addSendFunction(grad_fn, autogradMetadata.autogradMessageId);
}

ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const std::unordered_map<c10::Device, c10::Device>& deviceMap) {
  // Initialize autograd context if necessary.
  // 生成或者得到一个上下文，把发送方的 autogradContextId 传入，
  // 即利用 autogradContextId 作为key后续可以查找到这个上下文
  auto& autogradContainer = DistAutogradContainer::getInstance();
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

Message getMessageWithProfiling(
    torch::distributed::rpc::Message&& wrappedRpcMessage,
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
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(profilerConfig),
      globallyUniqueProfilingId);

  return std::move(wrappedProfilingMsg).toMessage();
}

//而 getMessageWithAutograd 会与上下文交互，其代码位于 torch/csrc/distributed/autograd/utils.cpp。
Message getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    torch::distributed::rpc::Message&& wrappedRpcMsg,
    MessageType msgType,
    bool forceGradRecording,
    const std::unordered_map<c10::Device, c10::Device>& deviceMap) {
// 获取到 DistAutogradContainer
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // If there is no valid context and no tensor requires grads, send original
  // rpc message. otherwise, attach grad info and grad functions and send
  // rpcWithAutograd message.
  auto tensorsRequireGrad =
      torch::autograd::compute_requires_grad(wrappedRpcMsg.tensors());
  if (!autogradContainer.hasValidContext() ||
      (!forceGradRecording && !tensorsRequireGrad)) {
    return std::move(wrappedRpcMsg);
  }

  // Retrieve the appropriate context to modify.
  auto autogradContext = autogradContainer.currentContext(); // 获取到上下文，每个worker都有自己的上下文

  // Wrap the original rpc with autograd information.
  // newAutogradMessageId 会生成一个messageID
  AutogradMetadata autogradMetadata( // 构建了 AutogradMetadata   //// 使用上下文id和消息id来构建元数据
      autogradContext->contextId(), autogradContainer.newAutogradMessageId());
  auto rpcWithAutograd = std::make_unique<RpcWithAutograd>(
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_,
      msgType,
      autogradMetadata,
      std::move(wrappedRpcMsg),
      deviceMap);

  if (tensorsRequireGrad) {
    // Record autograd information for 'send'.
    addSendRpcBackward(  // 这里把本地上下文，autograd 的元信息等一起打包
        autogradContext, autogradMetadata, rpcWithAutograd->tensors());
  }
  // Record the workerID
  autogradContext->addKnownWorkerId(dstId);

  return std::move(*rpcWithAutograd).toMessage();  // 最终构建了一个message
}

/*
3.2.2 sendMessageWithAutograd
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

/*
0x01 设计脉络
1.1 前文回顾
在前文之中当发送消息时候，
我们在 sendMessageWithAutograd 通过 getMessageWithAutograd 来获得了 FORWARD_AUTOGRAD_REQ 类型的消息。
*/
c10::intrusive_ptr<JitFuture> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    torch::distributed::rpc::Message&& wrappedRpcMsg,
    bool forceGradRecording,
    const float rpcTimeoutSeconds,
    bool forceDisableProfiling) {
  auto msg = getMessageWithAutograd( // 这里会与上下文交互，构建了 FORWARD_AUTOGRAD_REQ
      dst.id_,
      std::move(wrappedRpcMsg),
      MessageType::FORWARD_AUTOGRAD_REQ,
      forceGradRecording,
      agent.getDeviceMap(dst));

  c10::intrusive_ptr<JitFuture> fut;
  // If profiler is enabled, wrap this message with profiling metadata that will
  // tell the remote end to process this request with the profiler enabled.
  if (!forceDisableProfiling && torch::autograd::profiler::profilerEnabled()) {
    auto profilerConfig = torch::autograd::profiler::getProfilerConfig();
    auto msgWithProfiling = getMessageWithProfiling(
        std::move(msg),
        rpc::MessageType::RUN_WITH_PROFILING_REQ,  //构建消息
        // NOLINTNEXTLINE(performance-move-const-arg)
        std::move(profilerConfig));
    // 发送消息
    fut = agent.send(dst, std::move(msgWithProfiling), rpcTimeoutSeconds);
  } else {
    // 发送消息
    fut = agent.send(dst, std::move(msg), rpcTimeoutSeconds);
  }

  return fut;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
