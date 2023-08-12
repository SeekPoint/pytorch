2.3 示例
我们再具体打印输出看看，模型包含了6个层，分为 3 个partitions，分区内的层数分别是：3个，2个，1个。

a = nn.Linear(1, 1)
b = nn.Linear(1, 1)
c = nn.Linear(1, 1)
d = nn.Linear(1, 1)
e = nn.Linear(1, 1)
f = nn.Linear(1, 1)

balance = [3,2,1] # 分成了3个partition，第一个partition包括3个层，第2个包括2个层，第3个包括1个层。
model = nn.Sequential(a, b, c, d, e, f)
print(model)
model = GPipe(model, balance, devices=['gpu', 'gpu','gpu'])
print(model)
结果如下，可以看到原模型被分成3个partition，每个 partition 都是一个Sequential。

Sequential(
  (0): Linear(in_features=1, out_features=1, bias=True)
  (1): Linear(in_features=1, out_features=1, bias=True)
  (2): Linear(in_features=1, out_features=1, bias=True)
  (3): Linear(in_features=1, out_features=1, bias=True)
  (4): Linear(in_features=1, out_features=1, bias=True)
  (5): Linear(in_features=1, out_features=1, bias=True)
)

GPipe(
  (partitions): ModuleList(
    (0): Sequential(
      (0): Linear(in_features=1, out_features=1, bias=True)
      (1): Linear(in_features=1, out_features=1, bias=True)
      (2): Linear(in_features=1, out_features=1, bias=True)
    )
    (1): Sequential(
      (3): Linear(in_features=1, out_features=1, bias=True)
      (4): Linear(in_features=1, out_features=1, bias=True)
    )
    (2): Sequential(
      (5): Linear(in_features=1, out_features=1, bias=True)
    )
  )
)
运行时变量如下：

model = {GPipe: 6}
 balance = {list: 3} [3, 2, 1]
 checkpoint = {str} 'except_last'
 chunks = {int} 1
 devices = {list: 3}
  0 = {device} gpu
  1 = {device} gpu
  2 = {device} gpu
 partitions = {ModuleList: 3}
   _modules =
   '0' = {Sequential: 3}
        Sequential(
        (0): Linear(in_features=1, out_features=1, bias=True)
        (1): Linear(in_features=1, out_features=1, bias=True)
        (2): Linear(in_features=1, out_features=1, bias=True))
   '1' = {Sequential: 2}
        Sequential(
        (3): Linear(in_features=1, out_features=1, bias=True)
        (4): Linear(in_features=1, out_features=1, bias=True))
   '2' = {Sequential: 1}
        Sequential(
        (5): Linear(in_features=1, out_features=1, bias=True))
'''
需要注意一点：GPipe 的 partitions 成员变量是 nn.ModuleList 类型。nn.ModuleList是一个容器，其储存不同 module，并自动将每个 module 的 parameters 添加到网络中。但是nn.ModuleList 并没有定义一个网络，而只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序，网络的执行顺序是根据 forward 函数来决定的。

随之而来问题就是：partition内部可以用Sequential来进行一系列的前向操作，但是如何配置partitions 之间的执行顺序？这个我们会在后续文章中分析。

2.4 总结
最后总结一下，流程是从上至下。

使用 balance_by_size 或者 balance_by_time 来先运行系统，得到 profile 结果。
然后使用 split_module 来对模型进行分割。
最后就得到了一个相对平衡的分区结果。
把这些分区分配到不同的设备之上。
具体如下图：

+-----------------------------------------------------------------------------------------+
|                                                                                         |
| Layer 1 +--->  Layer 2 +-----> Layer 3 +----->  Layer 4 +-----> Layer 5  +---> Layer 6  |
|                                                                                         |
+--------------------------+---------------------------+----------------------------------+
                           |                           |
           balance_by_size | 1                       1 |  balance_by_time
                           |                           |
                           v                           v
                [[1, 2, 3], [4, 5], [6]]         [[1, 2, 3, 4], [5, 6]]
                           +                           +
                           |                           |
                           +-----------+      +--------+
                                       |      |
                                       v      v
                                 2  split_module
                                          +
                                          |
                                          |
   3                                      v
 +------------------------------------------------------------------------------------+
 | +--------------------+         +---------------------+      +--------------------+ |
 | |Partition 1         |         |Partition 2          |      |Partition 3         | |
 | |                    |         |                     |      |                    | |
 | |      Layer 1       |    +---------> Layer 4        |      |                    | |
 | |         +          |    |    |         +           |  +------->   Layer 6      | |
 | |         |          |    |    |         |           |  |   |                    | |
 | |         v          |    |    |         |           |  |   |                    | |
 | |      Layer 2       |    |    |         |           |  |   |                    | |
 | |         +          |    |    |         v           |  |   |                    | |
 | |         |          |    |    |      Layer 5 +---------+   |                    | |
 | |         v          |    |    |                     |      |                    | |
 | |      Layer 3  +---------+    |                     |      |                    | |
 | |                    |         |                     |      |                    | |
 | +---------+----------+         +---------+-----------+      +-----------+--------+ |
 |           |                              |                              |          |
 +------------------------------------------------------------------------------------+
             |                              |                              |
           4 |                            4 |                            4 |
             v                              v                              v
   +---------+----------+         +---------+-----------+       +----------+---------+
   |                    |         |                     |       |                    |
   |    Device 1        |         |     Device 2        |       |     Device 3       |
   |                    |         |                     |       |                    |
   +--------------------+         +---------------------+       +--------------------+
至此，我们分析了自动平衡机制，下一篇我们看看如何切分数据和一些运行时机制。
'''