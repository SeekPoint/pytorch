import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))


model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()

'''
这里最重要的是 labels = torch.randn(20, 5).to('cuda:1')，这保证了标签在 cuda:1'。

回忆一下之前forward的代码：self.net2(x.to('cuda:1'))。这两行代码确保标签与输出在同一设备 cuda:1' 上。

初始化之后如下：

+--------------------+                       +------------------------+
| cuda:0             |                       | cuda:1                 |
|                    |                       |                        |
|                    |                       |                        |
|                    |                       |                        |
|       net1(x)      |                       |        net2(x)         |
|                    |                       |                        |
|                    |                       |                        |
|                    |                       |                        |
+--------------------+                       +------------------------+

forward 操作和设定label之后如下，现在输出和label都在GPU 1 之上：

               +--------------------+                       +------------------------+
               | cuda:0             |                       | cuda:1                 |
               |                    |                       |                        |
               |                    |                       |                        |
               |                    |                       |                        |
x.to('cuda:0')-------> net1(x)  +-------> x.to('cuda:1') +-------->  net2(x)         |
               |                    |                       |                        |
               |                    |                       |   labels.to('cuda:1')  |
               |                    |                       |                        |
               +--------------------+                       +------------------------+
               '''