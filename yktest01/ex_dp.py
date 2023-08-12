from math import pi
import torch
import torch.nn as nn

import torch.optim as optim #from torch import  optim

from torchviz import make_dot

# 还没调试好！

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

x = torch.tensor([pi/2,pi/3],requires_grad=True)
optimizer = torch.optim.SGD([x,],lr=0.2,momentum=0.5)

for step in range(11):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()

        for var_name in optimizer.state_dict():
            print(var_name, '\t', optimizer.state_dict()[var_name])
    f=-((x.sin()**3).sum())**3


model = torch.nn.DaraParallel(ToyModel);
optimizer = torch.optim.SGD(model.parameters(), 1.0e-5,
                            momentum=0.01,
                            weight_decay=0.01)

for batch_idx, (data, label) in pbar:
    if args.cuda:
        data, label = data.cuda(), label.cuda();  # 数据放到了默认GPU
    data_v = Variable(data)
    target_var = Variable(label)

    prediction = model(data_v, target_var, args)  # 多线程并行前向传播
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction, target_var)  # 在默认GPU之上计算loss

    optimizer.zero_grad()
    loss.backward()  # 多线程并行后向传播
    optimizer.step()  # 更新参数

'''
0x01 前文回顾
常规优化器主要功能就是使用梯度来进行优化，然后更新当前参数 ： w.data -= w.grad * lr，而且是严格有条理的进行。
数据并行之中的优化器就是另外一种情况，因为每个worker自己计算梯度，所以优化器主要技术难点是：
    每个worker有自己的优化器？还是只有一个worker才有优化器，由他统一做优化？
    如果只有一个优化器，如何把各个worker的梯度合并起来，每个worker都传给这唯一的优化器？
    如果每个worker有自己优化器，本地优化器优化到本地模型之中，如何确保每个worker之中的模型始终保持一致？
这随着具体框架方案不同而有具体分别。

0x02 DP 之中的优化器
2.1 流程
DP 之中，我们需要注意的是，PyTorch 使用了多线程并行，所以应用之中只有一个优化器，这个优化器也是普通类型的优化器，其流程如下：
    每个 GPU 在单独的线程上将针对各自的输入数据独立并行地进行 forward 计算，计算输出。
    在 master GPU 之上收集（gather）输出。
    在主GPU之上 计算损失。
    把损失在 GPUs 之间 scatter。
    在各个GPU之上运行后向传播，计算参数梯度。
    在 GPU 0 之上归并梯度。
    进行梯度下降，并用梯度更新主GPU上的模型参数。
    将更新后的模型参数复制到剩余的从属 GPU 中，进行后续迭代。
DP 修改了 forward 和 backward 方法，把每个线程的梯度归并在一起然后做优化，所以虽然是数据并行，但是优化器不需要做修改。


我们给出一个简化的图示如下，每个thread进行梯度计算，最后把梯度归并到GPU 0，在GPU 0之上进行优化：

             Forward                                                    Backward
      +-------------------+                                       +------------------+
  +-->+ Thread 0 on  GPU0 +--+                                +-->+ Thread 1 on GPU0 +-+
  |   +-------------------+  |          GPU 0                 |   +------------------+ |
  |   +-------------------+  | output +---------------+ loss  |   +------------------+ |
+---->+ Thread 1 on  GPU1 +---------> |  Compute Loss +---------->+ Thread 2 on GPU1 +---+
| |   +-------------------+  |        +---------------+       |   +------------------+ | |
| |   +-------------------+  |                                |   +------------------+ | |
| +-->+ Thread 2 on  GPU2 +--+                                +-->+ Thread 3 on GPU2 +-+ |
|     +-------------------+                                       +------------------+   |
|                                                                                        |
|                                                                                        |
|                                    GPU 0                                               |
|   Model                  +-------------------------+    gradient                       |
+--------------------------+     optimizer.step      |  <--------------------------------+
                           +-------------------------+
'''

