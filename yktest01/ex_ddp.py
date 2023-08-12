import torch
import torch.nn as nn

import torch.optim as optim #from torch import  optim

# 还没调试好！

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

model = ToyModel().to(rank)
# 构造DDP model
ddp_model = DDP(model, device_ids=[rank])

loss_fn = nn.MSELoss()
# 优化器要在构造DDP model之后，才能初始化。
optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = ddp_model(torch.randn(20, 10))
labels = torch.randn(20, 5).to(rank)
loss_fn(outputs, labels).backward()
optimizer.step()


'''
0x03 DDP 之中的优化器
下图来自快手八卦的论文，图中罗列了原生训练过程与DDP/Horovod的对比。

上面的 vanilla 就是原生训练过程，其中 U 部分对应的就是优化器过程。常规优化器主要功能就是根据梯度来更新模型当前参数 ： w.data -= w.grad * lr。
下面部分就是DDP/Horovod优化过程，可以看到，其后向计算和归并梯度是部分并行的。
img ===有图片！

3.1 流程
DDP 之中，依然使用的是普通优化器，但采用的是多进程方式，每个进程都完成训练的全部流程，只是在后向计算时候需要使用 all-reduce 来归并梯度。每个进程拥有自己独立的优化器，优化器也是常规优化器。

这里有两个特点：
    每个进程维护自己的优化器，并在每次迭代中执行一个完整的优化步骤。虽然这可能看起来是多余的，但由于梯度已经聚合（gather）并跨进程平均，因此梯度对于每个进程都是相同的，这意味着不需要参数广播步骤，减少了在节点之间传输张量所花费的时间。
    All-Reduce 操作是在后向传播之中完成的。
        在 DDP 初始化时候会生成一个Reducer，其内部会注册 autograd_hook。
        autograd_hook 在反向传播时候进行梯度同步。
DDP 选择了在 PyTorch 内核角度修改，在 DistributedDataParallel 模型的初始化和前向操作中做了处理。

具体逻辑如下：
    DDP 使用多进程并行加载数据，在 host 之上，每个worker进程都会把数据从硬盘加载到 page-locked memory。分布式 minibatch sampler 保证每个进程加载到的数据是彼此不重叠的。
    不需要广播数据，而是并行把 minibatch 数据从 page-locked memory 加载到每个GPU，每个GPU都拥有模型的一个副本，所以也不需要拷贝模型。
    在每个GPU之上运行前向传播，计算输出，每个GPU都执行同样的训练，不需要有主 GPU。
    在每个GPU之上计算损失，运行后向传播来计算梯度，在计算梯度同时对梯度执行all-reduce操作。
    更新模型参数。因为每个GPU都从完全相同的模型开始训练，并且梯度被all-reduced，因此每个GPU在反向传播结束时最终得到平均梯度的相同副本，所有GPU上的权重更新都相同，这样所有 worker 上的模型都一致，也就不需要模型同步了。
因为也是在模型的前向后向操作之中进行修改，所以优化器也不需要修改，每个worker分别在自己本地进程之中进行优化。

3.2 优化器状态
这里要留意的是，如何保证各个进程的优化器状态相同？
DDP 与优化器实际上没有关联，DDP不对此负责，所以需要用户协同操作来保证各进程之间的优化器状态相同。这就围绕着两个环节：
    优化器参数初始值相同。
        优化器初始值相同由 "用户在DDP模型创建后才初始化optimizer" 来确保。
    优化器参数每次更新值相同。
        每次更新的梯度都是all-reduce过的，所以各个优化器拿到的梯度delta数值是一样的。
        
        
图示如下：

+--------------------------------------------------------------------------------------+
| Process 1 on GPU 1                                                                   |
|                              +------------------------------+                        |
|                              | Backward                     |                        |
|                              |                              |                        |
| Forward +---->  Loss +-----> |  Compute  +---->  ALL+REDUCE | +---->  Optimizer.step |
|                              |                     ^        |                        |
|                              |                     |        |                        |
|                              +------------------------------+                        |
|                                                    |                                 |
|                                                    |                                 |
+--------------------------------------------------------------------------------------+
                                                     |
                                                     |
                                                     |
                                                     |
                                                     +
                                                 SYNC GRADS
                                                     +
                                                     |
                                                     |
                                                     |
+--------------------------------------------------------------------------------------+
| Process 2 on GPU 2                                 |                                 |
|                                                    |                                 |
|                              +------------------------------+                        |
|                              | Backward            |        |                        |
|                              |                     v        |                        |
| Forward +---->  Loss +-----> |  Compute  +---->  ALL+REDUCE | +---->  Optimizer.step |
|                              |                              |                        |
|                              |                              |                        |
|                              +------------------------------+                        |
|                                                                                      |
+--------------------------------------------------------------------------------------+
'''