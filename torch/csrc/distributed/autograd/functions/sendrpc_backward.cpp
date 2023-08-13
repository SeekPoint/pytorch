#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>

namespace torch {
namespace distributed {
namespace autograd {

torch::autograd::variable_list SendRpcBackward::apply(
    torch::autograd::variable_list&& inputs) {
  TORCH_INTERNAL_ASSERT(
      inputs.empty(), "SendRpcBackward should receive no inputs");

  // Each grad variable should be valid!
  for (const auto& grad : grads_) {
    TORCH_INTERNAL_ASSERT(
        grad.defined(), "BUG!: SendRpcBackward didn't receive valid gradients");
  }

  // Simply forwards the gradients over.
  return std::move(grads_);
}
/*

之前看到，SendRpcBackward新增成员变量是 grads_，我们看看 grads_ 如何设置和使用？

SendRpcBackward 提供了 set, get 操作。

何时会使用？在 torch/csrc/distributed/rpc/request_callback_no_python.cpp 之中有 processBackwardAutogradReq。processBackwardAutogradReq 会：

    使用 sendFunction->setGrads(gradientsCall.getGrads()) 来设置远端传递来的梯度。
    调用 DistEngine::getInstance().executeSendFunctionAsync 来执行引擎开始本地后向计算。
对应了设计中如下文字，也就是被动进入引擎的起点：

    SendRpcBackward实际上是本地节点上autograd图的根。
    因此，它不会接收任何"输入"，而是RPC框架将梯度传递给该函数以启动局部autograd计算。

*/

void SendRpcBackward::setGrads(const torch::autograd::variable_list& grads) {
  grads_ = grads;
}

const torch::autograd::variable_list& SendRpcBackward::getGrads() const {
  return grads_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
