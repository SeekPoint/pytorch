#include <ATen/core/functional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::Variable;
using torch::autograd::variable_list;

RecvRpcBackward::RecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    ContextPtr autogradContext,
    rpc::worker_id_t fromWorkerId,
    rpc::DeviceMap deviceMap)
    : autogradMetadata_(autogradMetadata),
      autogradContext_(std::move(autogradContext)),
      fromWorkerId_(fromWorkerId),
      deviceMap_(std::move(deviceMap)) {}
/*
3.1.2.2 PropagateGradientsReq
继续找谁发出来的 BACKWARD_AUTOGRAD_REQ，就是谁调用到了 toMessageImpl？
原来在 torch/csrc/distributed/autograd/functions/recvrpc_backward.cpp 这里构建了 PropagateGradientsReq，
会使用 toMessage 来构建一个消息。
即，RecvRpcBackward 的调用会发送 BACKWARD_AUTOGRAD_REQ。


所以我们知道，在 RecvRpcBackward 的执行时候，会发送 BACKWARD_AUTOGRAD_REQ，发送给下一个节点。具体哪里调用 RecvRpcBackward？我们会在下一篇 DistEngine 之中介绍。

此时具体如下，对应就是 worker 0 的 t3 给 worker 1 发送 BACKWARD_AUTOGRAD_REQ 消息。

                                                                +
                                                       worker 0 | worker 1
                                                                |
                                                                |
 RecvRpcBackward                         PropagateGradientsReq  |
       +                                          +             |
       |                                          |             |
       |                                          |             |
       |                                          |             |
       v                                          |             |
                                                  |             |
     apply()                                      |             |
       +                                          |             |
       |                                          v             |
       |                                                        |
       | +------------------------------>  toMessageImpl        |
       |                                          +             |
       |                                          |             |
       |   Message(BACKWARD_AUTOGRAD_REQ)         |             |
       | <----------------------------------------+             |
       |                                                        |
       |                                                        |
       v                                                        |
                                                                |
rpcAgent+>send(Message)  +-------------------------------------------->
       +                             BACKWARD_AUTOGRAD_REQ      |
       |                                                        |
       |                                                        |
       v                                                        |
                                                                +


*/
variable_list RecvRpcBackward::apply(variable_list&& grads) { // 调用Node
  std::vector<Variable> outputGrads;
  for (const auto i : c10::irange(grads.size())) {
    const auto& grad = grads[i];
    if (grad.defined()) {
      outputGrads.emplace_back(grad);
    } else {
      // Put in zeros for a tensor with no grad.
      outputGrads.emplace_back(input_metadata(i).zeros_like());
    }
  }

  auto sharedContext = autogradContext_.lock();
  TORCH_CHECK(
      sharedContext,
      c10::str(
          "Autograd context no longer valid! This usually ",
          "means the autograd context was cleaned up by a different thread due ",
          "to an error before RecvRcpBackward had a chance to run"));

  // Send the gradients over the wire and record the future in the autograd
  // context.
  PropagateGradientsReq gradCall(  // 这里构建了 PropagateGradientsReq
      autogradMetadata_,
      outputGrads,
      sharedContext->retrieveGraphTask()->keep_graph_);

  // Send the gradients over to the appropriate node.
  auto rpcAgent = rpc::RpcAgent::getCurrentRpcAgent();
  auto jitFuture = rpcAgent->send(  // 发送出去，就是给后向传播过程的下一个节点
      rpcAgent->getWorkerInfo(fromWorkerId_),
      std::move(gradCall).toMessage(),   // 这里调用了PropagateGradientsReq::toMessageImpl
      rpc::kUnsetRpcTimeout,
      deviceMap_);

  // Record the future in the context.
  sharedContext->addOutstandingRpc(jitFuture);

  // 'recv' function sends the gradients over the wire using RPC, it doesn't
  // need to return anything for any downstream autograd function.
  return variable_list();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
