#pragma once

#include <torch/csrc/Export.h>
#include <cstdint>

namespace torch {
namespace distributed {
namespace autograd {
/*
1.2 总体思路
我们概括一下总体思路。

先看看问题：假如一套系统包括 a，b，c 三个节点，每个节点运行一个 worker，那么当运行一个传播操作，我们涉及到在这三个节点之间互相传播。因此我们需要一个机制，来在这三个节点之中唯一标示这个传播过程，在这个传播过程之中，也要在每一个节点之上把每一个send/recv都标示出来，这样才能让节点可以支持多个操作并行。

再看看解决方案：

    使用上下文来唯一标示一个传播过程。DistAutogradContext 存储在一个worker之上的每一个分布式autograd的相关信息，其在分布式 autograd 之中封装前向和后向传播，累积梯度，这避免了多个worker在彼此的梯度上互相影响。每个自动微分过程被赋予一个唯一的 autograd_context_id，在容器中，这个微分过程的上下文(DistAutogradContext) 依据这个autograd_context_id 来唯一确认。
    使用autogradMessageId 来表示一对 send/recv autograd 函数。每send-recv对被分配一个全局唯一的autograd_message_id 以唯一地标识该send-recv对。这对于在向后传播期间查找远程节点上的相应函数很有用。
    最后，每个worker需要有一个地方来保持上下文和messageid，所以有了DistAutogradContainer这个类。每个worker拥有唯一一个单例DistAutogradContainer，其负责：
        对于每一个自动微分过程存储其分布式上下文。
        一旦这个自动微分过程结束，就清除其数据。
这样，在前向传播期间，Pytorch 在上下文中存储每个 autograd 传播的send和recv函数。这确保我们在 autograd 图中保存对适当节点的引用以使其保持活动状态。除此之外，这也使得在向后传播期间很容易查找到对应的send和recv函数。

0x02 AutogradMetadata
2.1 定义
AutogradMetadata 这个类是用来在不同节点之间传递 autograd 的元信息，就是把上下文等信息封装了一下。即，发送方通知接收方自己的上下文信息，接收方会依据收到的这些上下文信息作相应处理。

我们提前剧透，接收方会使用 autogradContextId 和 autogradMessageId 分别作为 上下文 和 消息 的唯一标示。从注释之中可以知道。

    autogradContextId 是全局唯一整数，用来表示一个唯一的分布式 autograd 传播过程（包括前向传播和后向传播）。一个传播过程会包括在反向传播链条上的多对send/recv autograd 函数。
    autogradMessageId 是全局唯一整数，用来表示一对 send/recv autograd 函数。每send-recv对被分配一个全局唯一的autograd_message_id 以唯一地标识该send-recv对。这对于在向后传播期间查找远程节点上的相应函数很有用。


*/
// This structure represents autograd metadata that we need to pass across
// different nodes when we call an RPC which needs autograd computation.
struct TORCH_API AutogradMetadata {
  AutogradMetadata(int64_t autogradContextId, int64_t autogradMessageId);

  // autogradContextId_ is a globally unique integer that identifies a
  // particular distributed autograd pass.
  int64_t autogradContextId;
  // autogradMessageId_ is a globally unique integer that identifies a pair
  // of send/recv autograd functions.
  int64_t autogradMessageId;
};

} // namespace autograd
} // namespace distributed
} // namespace torch

//那么问题来了，autogradContextId 和 autogradMessageId 分别怎么做到全局（包括多个节点）唯一呢？
// autogradMessageId
