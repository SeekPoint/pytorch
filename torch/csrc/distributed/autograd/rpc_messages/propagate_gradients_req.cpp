#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/irange.h>

namespace torch {
namespace distributed {
namespace autograd {

using rpc::Message;
using rpc::MessageType;
using torch::autograd::Variable;

PropagateGradientsReq::PropagateGradientsReq(
    const AutogradMetadata& autogradMetadata,
    std::vector<Variable> grads,
    bool retainGraph)
    : autogradMetadata_(autogradMetadata),
      grads_(std::move(grads)),
      retainGraph_(retainGraph) {}
/*
3.1.2 内部隐式发起
因为是隐式发起，所以代码比较隐蔽，我们这次采用从下至上的方式来剥丝抽茧。
我们知道，如果节点之间要求反向传播，会发送BACKWARD_AUTOGRAD_REQ，所以我们从 BACKWARD_AUTOGRAD_REQ 开始发起寻找。

3.1.2.1 BACKWARD_AUTOGRAD_REQ
在 torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.cpp 之中 PropagateGradientsReq::toMessageImpl 会调用到 BACKWARD_AUTOGRAD_REQ。

其 toMessageImpl 指明了本消息是 BACKWARD_AUTOGRAD_REQ。
*/
c10::intrusive_ptr<Message> PropagateGradientsReq::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  // Add all the grad tensors.
  ivalues.reserve(grads_.size() + 3);
  for (const auto& grad : grads_) {
    ivalues.emplace_back(grad);
  }

  // Now add autograd metadata.
  ivalues.emplace_back(autogradMetadata_.autogradContextId);
  ivalues.emplace_back(autogradMetadata_.autogradMessageId);

  // Add retain graph.
  ivalues.emplace_back(retainGraph_);

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensorTable),
      MessageType::BACKWARD_AUTOGRAD_REQ); // 这里会用到  // 这里指明了消息类型。
}

std::unique_ptr<PropagateGradientsReq> PropagateGradientsReq::fromMessage(
    const Message& message) {
  // Unpickle the message and retrieve tupleElements.
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  IValue tuple = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  const auto& tupleElements = tuple.toTupleRef().elements();

  // Build PropagateGradientsReq.
  TORCH_INTERNAL_ASSERT(tupleElements.size() >= 3);

  // Retrieve retainGraph.
  bool retainGraph = tupleElements.back().toBool();

  // Build AutogradMetadata.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t autogradContextId, autogradMessageId;
  autogradMessageId = tupleElements[tupleElements.size() - 2].toInt();
  autogradContextId = tupleElements[tupleElements.size() - 3].toInt();

  AutogradMetadata autogradMetadata(autogradContextId, autogradMessageId);

  // Retrieve the gradient tensors.
  std::vector<Variable> grads(tupleElements.size() - 3);
  for (const auto i : c10::irange(tupleElements.size() - 3)) {
    grads[i] = tupleElements[i].toTensor();
  }

  return std::make_unique<PropagateGradientsReq>(
      autogradMetadata, grads, retainGraph);
}

const AutogradMetadata& PropagateGradientsReq::getAutogradMetadata() {
  return autogradMetadata_;
}

const std::vector<torch::autograd::Variable>& PropagateGradientsReq::
    getGrads() {
  return grads_;
}

bool PropagateGradientsReq::retainGraph() {
  return retainGraph_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
