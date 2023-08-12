import torch
import torch.nn as nn

import torch.optim as optim #from torch import  optim

from torchviz import make_dot


import torch.nn.functional as F

'''
[源码解析] PyTorch 分布式 Autograd (1) ---- 设计
https://www.cnblogs.com/rossiXYZ/p/15621210.html


1.2 问题点
因为已经有了之前分析引擎等其他经历，所以我们结合之前得到的知识先整理出几个问题点，用来引导我们分析，我们按照 ：根据模型参数构建优化器 ---> 引擎计算梯度 ---> 优化器优化参数 ---> 优化器更新模型 这个顺序来分析。我们知道是autograd引擎计算了梯度，这样问题就来了：

根据模型参数构建优化器

采用 optimizer = optim.SGD(params=net.parameters(), lr = 1) 进行构造，这样看起来 params 被赋值到优化器的内部成员变量之上（我们假定是叫parameters）。
模型包括两个 Linear，这些层如何更新参数？
引擎计算梯度

如何保证 Linear 可以计算梯度？
对于模型来说，计算出来的梯度怎么和 Linear 参数对应起来？引擎计算出来的这些梯度累积在哪里？
优化器优化参数：

调用 step 进行优化，优化目标是优化器内部成员变量 self.parameters。
优化器更新模型：

如何把优化目标（self.parameters）的更新反应到模型参数（比如 Linear）的更新上？
下面图之中的数字和问号就对应了上面4个问题。

      +-------------------------------------------+                    +------------------+
      |ToyModel                                   |                    | Engine           |
      |                                           | forward / backward |                  |
      | Linear(10, 10)+--> ReLU +--> Linear(10, 5)| +----------------> | Compute gradient |
      |                                           |                    |        +         |
      +-------------------+-----------------------+                    |        |         |
                          |                                            |        |         |
                    1 ??? | parameters()                               +------------------+
                          |                                                     |
                          |                                                     | gradient
                          |   ^                                                 |
                          |   |                                                 v
                          |   | 4 ???                                        2 ???
                          |   |
      +------------------------------------------+
      |SGD                |   |                  |
      |                   |   |                  |
      |                   v   +                  |
      |                                          |
^ +---------------> self.parameters  +---------------->
|     |                                          |    |
|     |                                          |    |
|     +------------------------------------------+    |
|                                                     |
<---------------------------------------------------+ v
                     3 step()



'''
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

net = ToyModel()
print("Toymodel net is:\n", net)

print('net.state_dict:\n', net.state_dict())
for param_tensor in net.state_dict():
    print(param_tensor, '\t', net.state_dict()[param_tensor].size())

print('--------------------------------------------------------------')
optimizer = optim.SGD(params=net.parameters(), lr = 1)

print('Optimizer,s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])

optimizer.zero_grad()
input = torch.randn(10,10)
outputs = net(input)
print("outputs a is:\n", outputs)
outputs.backward(outputs)
print("outputs b is:\n", outputs)
optimizer.step()
print("outputs c is:\n", outputs)


NetVis = make_dot(outputs, params=dict(list(net.named_parameters()) + [('x', input)]))
NetVis.format = "bmp" # 文件格式
NetVis.directory = "data" # 文件生成的文件夹
NetVis.view() # 生成文件




'''
Parameter 构造函数中参数 requires_grad=True。这么设置就说明 Parameter 默认就是需要计算梯度的。
通过 parameters 方法来获取，其返回的是一个Iterator。
所以之前图可以拓展一下，现在 SGD 的 parameters 是一个指向 ToyModel._parameters 的 iterator，这说明优化器实际上是直接优化 ToyModel 的 _parameters。所以我们可以去掉原来图之中 4) 对应的问号。

      +-------------------------------------------+                    +------------------+
      |ToyModel                                   |                    | Engine           |
      |                                           | forward / backward |                  |
      | Linear(10, 10)+--> ReLU +--> Linear(10, 5)| +----------------> | Compute gradient |
      |                                           |                    |        +         |
      |         para_iterator = parameters()      |                    |        |         |
      |                   +          ^            |                    |        |         |
      |                   |          |            |                    +------------------+
      +-------------------------------------------+                             |
                          |          |                                          | gradient
                          |          |                                          |
                  1 ???   |          | 4 update                                 v
                          |          |                                       2 ???
                          |          |
      +----------------------------------------------------------------+
      |SGD                |          |                                 |
      |                   |          |                                 |
      |                   v          |                                 |
      |                              +                                 |
^ +--------> self.parameters = para_iterator(ToyModel._parameters) --------->
|     |                                                                |    |
|     |                                                                |    |
|     +----------------------------------------------------------------+    |
|                                                                           |
<-------------------------------------------------------------------------+ v
                     3 step()
                     



目前我们可以继续拓展。

Linear 里面的 weight，bias 都是 Parameter 类型。
Parameter 构造函数中参数 requires_grad=True。这么设置就说明 Parameter 默认是需要计算梯度的。
所以 Linear 的 weight，bias 就是需要引擎计算其梯度。
ToyModel 的 _parameters 成员变量通过 parameters 方法来获取，其返回的是一个Iterator。
这个 iterator 作为参数用来构建 SGD 优化器。
现在 SGD 优化器 的 parameters 是一个指向 ToyModel._parameters 的 iterator。这说明优化器实际上是直接优化 ToyModel 的 _parameters，对于例子就是全连接层的参数，图上对应两个Linear 发出的指向 parameters() 的箭头。
+--------------------------------------------------+                   +------------------+
| ToyModel                                         |                   | Engine           |
| +-------------------+             +------------+ |forward / backward |                  |
| | Linear(10, 10)    +--> ReLU +-->+Linear(10,5)| +-----------------> | Compute gradient |
| |                   |             |            | |                   |        +         |
| |  weight=Parameter |             |    weight  | |                   |        |         |
| |                   +----------+  |            | |                   |        |         |
| |  bias=Parameter   |          |  |    bias    | |                   +------------------+
| |                   |          |  |            | |                            |
| +-------------------+          |  +--+---------+ |                          2 | gradient
|                                |     |           |                            |
|                                |     |           |                            v
|                                v     v           |                           ???
|               para_iterator = parameters()       |
|                         +          ^             |
|                         |          |             |
|                         |          |             |
+--------------------------------------------------+
                          |          |
                   1 ???  |          | 4 update
                          |          |
                          |          |
      +----------------------------------------------------------------+
      |SGD                |          |                                 |
      |                   |          |                                 |
      |                   v          |                                 |
      |                              +                                 |
^ +--------> self.parameters = para_iterator(ToyModel._parameters) +-------->
|     |                                                                |    |
|     |                                                                |    |
|     +----------------------------------------------------------------+    |
|                                                                           |
<-------------------------------------------------------------------------+ v
                     3 step()
                     



我们用如下代码打印 param_groups出来看看。

net = nn.Linear(3, 3)
nn.init.constant_(net.weight, val=10)
nn.init.constant_(net.bias, val=5)
optimizer = optim.SGD(net.parameters(), lr=0.025)
print(optimizer.param_groups)
结果如下，第一个 3 x 3 是 net 的权重矩阵，1 x 3 是偏置矩阵。

[
  {'params': 
    [
      Parameter containing: # 权重矩阵
        tensor([[10., 10., 10.],
              [10., 10., 10.],
              [10., 10., 10.]], requires_grad=True), 
      Parameter containing: # 偏置矩阵
        tensor([5., 5., 5.], requires_grad=True)
    ], 
  'lr': 0.025, 
  'momentum': 0, 
  'dampening': 0, 
  'weight_decay': 0, 
  'nesterov': False
  }
]

我们更新一下，确定了 SGD 内部的成员变量名字是 param_groups，这是优化器的优化目标，其指向了 ToyModel._parameters 的 iterator。

 +-------------------------------------------------+                   +------------------+
 |ToyModel                                         |                   | Engine           |
 | +------------------+             +------------+ |forward / backward |                  |
 | |Linear(10, 10)    +--> ReLU +-->+Linear(10,5)| +-----------------> | Compute gradient |
 | |                  |             |            | |                   |        +         |
 | |  weight=Parameter|             |    weight  | |                   |        |         |
 | |                  +-----------+ |    bias    | |                   |        |         |
 | |  bias=Parameter  |           | +--+---------+ |                   +------------------+
 | |                  |           |    |           |                            |
 | +------------------+           |    |           |                          2 | gradient
 |                                v    v           |                            |
 |                         self._parameters        |                            v
 |                                  +              |                           ???
 |                                  |              |
 |                                  |              |
 |                                  v              |
 |              para_iterator = parameters()       |
 |                        +          ^             |
 |                        |          |             |
 |                        |          |             |
 +-------------------------------------------------+
                          |          |
                    1 ??? |          | 4 update
                          |          |
      +----------------------------------------------------------------+
      |SGD                |          |                                 |
      |                   |          |                                 |
      |                   v          |                                 |
      |                              +                                 |
^ +-------> self.param_groups = para_iterator(ToyModel._parameters) -------->
|     |                                                                |    |
|     |                                                                |    |
|     +----------------------------------------------------------------+    |
|                                                                           |
<-------------------------------------------------------------------------+ v
                     3 step()
          

?????怎么打印的        
我们打印 outputs 看看，可以看到其 next_functions 实际是有三个，说明前面的图例是我们简化的，我们需要再做进一步可视化。

outputs = {Tensor: 10} 
 T = {Tensor: 5} 
 data = {Tensor: 10} 
 device = {device} cpu
 dtype = {dtype} torch.float32
 grad = {NoneType} None
 grad_fn = {AddmmBackward} 
  metadata = {dict: 0} {}
  next_functions = {tuple: 3} 
   0 = {tuple: 2} (<AccumulateGrad object at 0x7f9c3e3bd588>, 0)
   1 = {tuple: 2} (<ReluBackward0 object at 0x7f9c3e5178d0>, 0)
   2 = {tuple: 2} (<TBackward object at 0x7f9c3e517908>, 0)
   __len__ = {int} 3
  requires_grad = {bool} True
 is_cuda = {bool} False
 is_leaf = {bool} False
 is_meta = {bool} False
 is_mkldnn = {bool} False
 is_mlc = {bool} False
 is_quantized = {bool} False
 is_sparse = {bool} False
 is_sparse_csr = {bool} False
 is_vulkan = {bool} False
 is_xpu = {bool} False
 layout = {layout} torch.strided
 name = {NoneType} None
 names = {tuple: 2} (None, None)
 ndim = {int} 2
 output_nr = {int} 0
 requires_grad = {bool} True   
 
 
 
 
 
 
 
 
 我们按照根据模型参数构建优化器 ---> 引擎计算梯度 ---> 优化器优化参数 ---> 优化器更新模型这个顺序来总结。

根据模型参数构建优化器

采用 optimizer = optim.SGD(params=net.parameters(), lr = 1) 进行构造，这样 params 被赋值到优化器的内部成员变量 param_groups 之上。
模型包括两个 Linear，这些层如何更新参数？
Linear 里面的 weight，bias 都是 Parameter 类型。
Parameter 构造函数中参数 requires_grad=True。这么设置就说明 Parameter 默认是需要计算梯度的。
所以 Linear 的 weight，bias 就是需要引擎计算其梯度。
weight，bias 被添加到 ToyModel 的 _parameters 成员变量 之中。
ToyModel 的 _parameters 成员变量通过 parameters 方法来获取，其返回的是一个Iterator。
用 这个 iterator 作为参数用来构建 SGD 优化器。
现在 SGD 优化器 的 parameters 是一个指向 ToyModel._parameters 的 iterator。这说明优化器实际上是直接优化 ToyModel 的 _parameters。
所以优化器就是直接优化更新 Linear 的 weight 和 bias。其实优化器就是一套代码而已，具体优化哪些东西，需要在构建时候指定，优化一个模型的参数也行，优化用户自己指定的其他变量也行。
引擎计算梯度

如何保证 Linear 可以计算梯度？
weight，bias 都是 Parameter 类型，默认是需要计算梯度的。
所以计算 weight，bias 梯度。
对于模型来说，计算出来的梯度怎么和 Linear 参数对应起来？引擎计算出来的这些梯度累积在哪里？
对应我们的示例，Linear 实例都是用户显式定义的，所以都是叶子节点。
叶子节点通过 AccumulateGrad 把梯度累积在模型参数张量 autograd_meta_.grad_ 之中。
优化器优化参数：

调用 step 进行优化，优化目标是优化器内部成员变量 self.parameters。
self.parameters 是一个指向 ToyModel._parameters 的 iterator。这说明优化器实际上是直接优化 ToyModel 的 _parameters。
优化器更新模型：

优化目标（self.parameters）的更新实际上就是直接作用到模型参数（比如 Linear 的 weight，bias）之上。
具体如图：

+---------------------------------------------------------------------+
| ToyModel                                                            |
|  +---------------------------------+                 +------------+ |                   +------------------+
|  | Linear(10, 10)                  +------> ReLU +-->+Linear(10,5)| |                   | Engine           |
|  |                                 |                 |            | |forward / backward |                  |
|  |  weight=Parameter               |                 |    weight  | +-----------------> | Compute gradient |
|  |                                 +---------------+ |    bias    | |                   |        +         |
|  |  +----------------------------+ |               | +--+---------+ |                   |        |         |
|  |  | bias=Parameter             | |               |    |           |                   |        |         |
|  |  |                            | |               |    |           |                   +------------------+
|  |  |                            | |               |    |           |  3 accumulate              |
|  |  |    autograd_meta_.grad_ <----------------------------------------------------+           2 | gradient
|  |  |                            | |               |    |           |              |             |
|  |  |    data                    | |               |    |           |              |             v
|  |  |                            | |               v    v           |              |
|  |  |                            | |        self._parameters        |              |    +------------------+
|  |  +----------------------------+ |                 +              |              |    | AccumulateGrad   |
|  +---------------------------------+                 |              |              |    |                  |
|                                                      |              |              |    |                  |
|                                                      v              |  5 update    -----------+ apply()    |
|                                  para_iterator = parameters()  <----------------+       |                  |
|                                            +                        |           |       |                  |
|                                            |                        |           |       +------------------+
|                                            |                        |           |
+---------------------------------------------------------------------+           |
                                           1 |                                    |
                                             |                                    |
              +---------------------------------------------------------------------------+
              | SGD                          |                                    |       |
              |                              |                                    |       |
              |                              v                                    +       |
              |                                                                 4 step()  |
      ^-------------> self.param_groups = para_iterator(ToyModel._parameters) +---------------->
      |       |                                                                           |    |
      |       |                                                                           |    |
      |       +---------------------------------------------------------------------------+    |
      |                                                                                        |
      <--------------------------------------------------------------------------------------+ v
                  
'''