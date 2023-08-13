#pragma once

#include <torch/csrc/autograd/function.h>

namespace torch {
namespace distributed {
namespace autograd {
/*
1.2.1 剖析
SendRpcBackward 作为分布式autograd实现的一部分，每当我们将RPC从一个节点发送到另一个节点时，我们都会向autograd图添加一个"SendRpcBackward"autograd function。这是一个占位符函数，用于在向后传播时启动当前worker的autograd引擎。此autograd function的边是RPC方法的输入。

在向后传播过程中，此函数将在autograd引擎中排队等待执行，该引擎最终将运行autograd图的其余部分。

SendRpcBackward实际上是本地节点上autograd图的根。我们给出之前的示意图如下：
    SendRpcBackward不会接收任何 "输入"，而是RPC框架将梯度传递给该函数以启动局部autograd计算。
    SendRpcBackward的input边是RPC方法的输入，就是梯度。
*/
// As part of our distributed autograd implementation, whenever we send an RPC
// from one node to another, we add a 'SendRpcBackward' autograd function to the
// autograd graph. This is more or less a placeholder function that is used to
// kickoff the autograd engine on the current worker on the backward pass. The
// edges for this autograd function are the inputs to the RPC method.
//
// During the backward pass, this function is queued for execution in the
// autograd engine which eventually runs the rest of the autograd graph.
struct TORCH_API SendRpcBackward : public torch::autograd::Node {
 public:
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& inputs) override;

  // SendRpcBackward is actually the root of an autograd graph on the local
  // node. As a result, it doesn't receive any 'inputs', but rather the RPC
  // framework passes gradients over to this function to kickoff local autograd
  // computation.
  void setGrads(const torch::autograd::variable_list& grads);

  // Retrieve the grads for the function.
  const torch::autograd::variable_list& getGrads() const;

 private:
  torch::autograd::variable_list grads_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
