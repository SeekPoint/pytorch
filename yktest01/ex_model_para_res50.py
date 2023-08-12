'''
2.2 将模型并行应用到现有模块
还可以通过更改几行代码把一个现有的单 GPU 模块转换到在多个 GPU 上运行。下面的代码展示了如何分解 torchvision.models.resnet50()到两个 GPU之上。
基本想法是继承现有ResNet模块，并在构建过程中将层拆分为两个 GPU。然后，重载forward方法以便把两个子网络拼接起来，forward具体是通过相应地移动中间输出来完成。
'''

from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000

class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))


# 上述实现解决了模型太大而无法放入单个 GPU 的情况下的问题。但是，您可能已经注意到，即使您的模型适合这种情况，它也许会比在单个 GPU 上运行要慢。
# 这是因为，在任何时候，两个 GPU 中只有一个在工作，而另一个坐在那里什么也不做。在 layer2和layer3 之中需要把中间输出从cuda:0拷贝到 cuda:1，这将进一步引起性能恶化。
# 让我们运行一个实验，以更从一个可以量化地角度来了解执行时间。
# 在这个实验中，我们通过运行随机输入和标签来训练ModelParallelResNet50和现有 torchvision.models.resnet50()。
# 训练后，模型不会产生任何有用的预测，但我们可以对执行时间有一个合理的了解。

import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()
# 上述train(model)方法使用nn.MSELoss用作损失函数，使用optim.SGD作为优化器。它模仿 128 X 128图像的训练，这些图像被组织成 3 个批次，每批次包含 120 个图像。然后，我们使用timeit来运行 train(model) 10 次，并且用标准差来绘制执行时间。

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')

# 结果表明，模型并行需要的执行时间比但GPU实现需要的时间长 4.02/3.75-1=7%。所以我们可以得出结论，在 GPU 之间来回复制张量大约有 7% 的开销。



'''
2.3 问题与方案
2.3.1 目前状况
我们总结一下目前状况：

虽然有多块GPU，但是在整个执行过程中的每一个时刻，只有一个GPU在计算，其他GPU处于空闲状态。
另外还有中间计算结果在GPU之间的拷贝工作，这也使得性能恶化。
因此我们需要针对这两个问题进行针对性处理：

让所有 GPU 都动起来。
减少拷贝传输时间。
2.3.2 解决方案
两个问题解决方案如下：

让所有 GPU 都动起来的一种选择是加入流水线机制：将每个批次进一步划分，组成一个分割（split ）管道，这样当一个分割到达第二个子网络时，可以将接下来的分割送入第一个子网络。这样，两个连续的分割（split ）就可以在两个 GPU 上同时运行。

为什么可以做到这一点？这是因为 CUDA 的异步并行执行逻辑。

CUDA 的一些操作是异步的，比如：核发射，设备间数据拷贝，主机和设备内拷贝小存储块等等。
几乎所有具有计算能力1.1及更高计算能力的CUDA设备都支持并发复制和核执行，即数据拷贝和数值计算可以并行。
一些计算能力2.x的设备可并发执行多个内核。
在一些计算能力2.x的设备上，两个方向的拷贝可以并行（GPU到CPU，CPU到GPU）。
如何减少拷贝传输时间？这个可以使用一些硬件和软件的结合来增加带宽减少延迟，比如：

硬件层面包括：单机内部的PCIe、NVlink、NVSwitch；多机之间的RDMA网络（IB或RoCE）。
软件堆栈包括：GPUDirect的一系列技术：P2P（Peer-to-Peer），RDMA，Async，Storage等。
PyTorch使用了NCCL库（基于CUDA计算）。
'''


# 在接下来的实验中，我们进一步将每个"120 个图像批次" 分成 "20 个图像分割（split）"。由于 PyTorch 异步启动 CUDA 操作，因此实现不需要产生多个线程来实现并发。

class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')
# 请注意，设备到设备张量复制操作会在源设备和目标设备上的当前流上进行同步。如果创建多个流，则必须确保复制操作正确同步。在完成复制操作之前写入源张量或读取/写入目标张量可能会导致未定义的行为。上述实现仅在源设备和目标设备上使用默认流，因此没有必要强制执行额外的同步操作。

# 实验结果表明，把流水线输入加入到 ResNet50 的模型并行之后，训练过程加快了大约3.75/2.51-1=49%。虽然它离理想的 100% 加速还很远。
# 由于我们在流水线并行实现中引入了一个新参数split_sizes，因此尚不清楚此新参数如何影响整体训练时间。
# 直观地说，使用小的split_size会导致许多微小的 CUDA 核启动，而使用大split_size结果会导致在第一次和最后一次拆分期间产生相对较长的空闲时间。两者都不是最优的。split_size这个特定实验可能有一个最佳配置。让我们尝试通过使用几个不同的split_size值运行实验来找到它。

means = []
stds = []
split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

for split_size in split_sizes:
    setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    means.append(np.mean(pp_run_times))
    stds.append(np.std(pp_run_times))

fig, ax = plt.subplots()
ax.plot(split_sizes, means)
ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
ax.set_ylabel('ResNet50 Execution Time (Second)')
ax.set_xlabel('Pipeline Split Size')
ax.set_xticks(split_sizes)
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig("split_size_tradeoff.png")
plt.close(fig)
'''
结果表明，设置split_size为 12 实现了最快的训练速度，从而导致3.75/2.43-1=54%加速。我们仍有机会进一步加快训练进程。
例如，目前所有cuda:0上的操作都放在其默认流上。这意味着下一个拆分的计算不能与上一个拆分的复制操作重叠。
但是，由于 prev 和 next 拆分（split）是不同的张量，因此将一个张量的计算与另一个张量的拷贝重叠起来是没有问题的。
这种实现需要在两个GPU上使用多个流，并且不同的子网结构需要不同的流管理策略。
由于没有一个适用于所有模型并行用例的通用的多流解决方案，我们不会在本教程中讨论它。

这篇文章展示了几个性能测量。在您自己的机器上运行相同的代码时，您可能会看到不同的性能结果，因为结果取决于底层硬件和软件。
要为您的环境获得最佳性能，正确的方法是首先生成结果曲线，并根据曲线来确定最佳分割大小，然后将该分割大小应用到管道输入之上。
'''