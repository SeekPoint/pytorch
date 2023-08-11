import math
from typing import TypeVar, Optional, Iterator

import torch
from . import Sampler, Dataset
import torch.distributed as dist

__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)

# ref https://www.cnblogs.com/rossiXYZ/p/15142807.html
'''
0x02 PyTorch分布式加载
2.1 DDP
pytorch为数据分布式训练提供了多种选择。随着应用从简单到复杂，从原型到产品，常见的开发轨迹可以是：

    如果数据和模型能放入单个GPU，使用单设备训练，此时不用担心训练速度；
    如果服务器上有多个GPU，并且你在代码修改量最小的情况下加速训练，使用单个机器多GPU DataParallel；
    如果你想进一步加速训练并且愿意写一点代码来启动，使用单个机器多个GPU DistributedDataParallel；
    如果应用程序跨机器边界扩展，使用多机器DistributedDataParallel和启动脚本；
    如果预期有错误（比如OOM）或者资源在训练过程中可以动态连接和分离，使用torchelastic来启动分布式训练。
与本文最相关的部分就是DDP，Distributed Data-Parallel Training（DDP）是一个广泛采用的单程序多数据训练方法。使用DDP，模型会被复制到每个进程，然后每个模型副本会被输入数据样本的不同子集。DDP负责梯度通信以保持模型副本的同步，并将其与梯度计算重叠以加快训练速度。

2.2 分布式加载
我们首先要看看分布式加载的总体结构。
给出示例代码，可以看到主要使用了 DataSet, DistributedSampler，DataLoader 这三个实体。
sampler = DistributedSampler(dataset) if is_distributed else None
loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
for epoch in range(start_epoch, n_epochs):
	if is_distributed:
        sampler.set_epoch(epoch)
        train(loader)
这三个概念的逻辑关系如下：

    Dataset : 从名字可以知道，是数据集的意思。负责对原始训练数据的封装，将其封装成 Python 可识别的数据结构，Dataset的派生类必须提供接口一边获取单个数据。
    Sampler : 从名字可知，是采样器，负责采样方式或者说是采样策略，实现某种提取/采样策略从Dataset之中拿到数据索引，供DataLoade使用。可以认为，Sampler 是指挥者，负责决定战斗在哪里开展。
    DataLoader : 负责依据索引来从数据集中加载数据。支持 Map-style 和 Iterable-style 两种Dataset，支持单进程/多进程加载。Loader 就是具体作战的斗士，负责按照 Sampler的命令进行战斗。

具体如下图，简要说就是：
    DataSet 把数据集数目发给DistributedSampler。
    Sampler 按照某种规则发送数据indices给Loader。
    Loader 依据indices加载数据。
    Loader 把数据发给模型，进行训练。
+------------------------+                     +-----------+
|DistributedSampler      |                     |DataLoader |
|                        |     2 indices       |           |
|    Some strategy       +-------------------> |           |
|                        |                     |           |
|-------------+----------|                     |           |
              ^                                |           |  4 data  +-------+
              |                                |       -------------->+ train |
            1 | length                         |           |          +-------+
              |                                |           |
+-------------+----------+                     |           |
|DataSet                 |                     |           |
|        +---------+     |      3 Load         |           |
|        |  Data   +-------------------------> |           |
|        +---------+     |                     |           |
|                        |                     |           |
+------------------------+                     +-----------+
因为数据集不是分布式训练重点，所以本文接下来主要分析 Sampler。

Sampler 的重点就是：如何让每个worker在数据集中只加载自己所属的部分，并且worker之间实现对数据集的正交分配。

内部变量之间逻辑如下：
    从数据集获取长度length；
    从配置得到 num_replicas（有几个rank），本身rank；
    依据 数据集长度 和 num_replicas得到 num_samples 和 total_size；
    最终给出 indices = indices[rank: total_size: num_replicas]；
    返回 indices 给DataLoader
+-----------------------------------------------------------+
| DistributedSampler                                        |
|                                                           |
|    2                 2                                    |
|  rank   +---+  num_replicas                               |
|    +    |            +                                    |
|    |    |            | 3                                  |
|    |    |            |                                    |
|    |    |            v                                    |
|    |    |  num_samples = ceil(len(dataset)/ num_replicas) |
|    |    |            +                                    |
|    |    |            |                                    |
|    |    |            | 3                                  |
|    |    |            v                                    |
|    |    |      total_size = num_samples * num_replicas    |
|    |    |            +                                    |
|    |4   |4           |4                                   |
|    |    |            |                                    |
|    v    v            v                                    |
|  +-+----+------------+--------------------------------+   |          +-------------+
|  |                                                    |   | indices  |             |
|  | indices = indices[rank: total_size: num_replicas]  +------------->+  DataLoader |
|  |              ^                                     |   |    5     |             |
|  |              |                                     |   |          +-------------+
|  |              |                                     |   |
|  +----------------------------------------------------+   |
+-----------------------------------------------------------+
                  |
                1 | length
           +------+--------+
           |   DataSet     |
           +---------------+




我们扩展目前逻辑如下：
    从数据集获取长度length；
    从配置得到 num_replicas（有几个rank），本身rank，epoch；
    用 epoch 来设置random seed；
    利用random seed 对数据集 indices 进行打乱，后续就会一直使用 这个打乱的indices；
    依据 数据集长度 和 num_replicas得到 num_samples 和 total_size；
    结合之上各种数据条件，最终给出 indices = indices[rank: total_size: num_replicas]；
    返回 indices 给DataLoader
+-----------------------------------------------------------------+
| DistributedSampler                                              |
|                                                                 |
|                                                                 |
|    2       3                                                    |
|   epoch +------>  manual_seed(seed + epoch) +---------> indices |
|                                                              +  |
|                                                              |  |
|                                                              |  |
|    2                 2                                       |  |
|  rank   +---+  num_replicas                                4 |  |
|    +    |            +                                       |  |
|    |    |            | 5                                     |  |
|    |    |            |                                       |  |
|    |    |            v                                       |  |
|    |    |  num_samples = ceil(len(dataset)/ num_replicas)    |  |
|    |    |            +                                       |  |
|    |    |            |                                       |  |
|    |    |            | 5                                     |  |
|    |    |            v                                       |  |
|    |    |      total_size = num_samples * num_replicas       |  |
|    |    |            +                                       |  |
|    |6   |6           |6                                      |  |
|    |    |            |                                       |  |
|    v    v            v                                       |  |
|  +-+----+------------+--------------------------------+      |  |
|  |                                                    |      |  |
|  | indices = indices[rank: total_size: num_replicas]  | <----+  |
|  |              ^                          +          |         |
|  |              |                          |          |         |
|  |              |                          |          |         |
|  +----------------------------------------------------+         |
+-----------------------------------------------------------------+
                  |                          |
                1 | length                7  v indices
                  |
          +-------+--------+             +-------------+
          |                |             |             |
          |    DataSet     |             |  DataLoader |
          |                |             |             |
          +----------------+             +-------------+
          
          
'''
class DistributedSampler(Sampler[T_co]):
    '''
    对于数据并行和分布式训练，DistributedSampler 负责其数据采样的任务。
    DistributedSampler 是 Sampler 的派生类。当 DistributedDataParallel 使用DistributedSampler 时，
    每个并行的进程都会得到一个DistributedSampler 实例，
    这个DistributedSampler 实例会给DataLoader发送指示，从而 DataLoader 加载具体数据。
    DistributedSampler 加载策略负责只提供加载数据集中的一个子集，这些DistributedSampler 提供的子集之间不重叠，不交叉。
    '''
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        '''
        __init__初始化代码主要是设置了本worker节点的各种信息，比如数据集dataset，rank（全局GPU序号），num_replicas 副本数目。并且计算出来所有样本数目total_size。
        几个参数如下：
            dataset ： 就是采样的数据集。
            num_replicas ：参与分布式训练的进程数目，如果没有设置，则从group之中得到world_size作为进程数目。
            rank : 当前进程的序号，如果没有设置，则从group之中得到。
            shuffle ：采样是否需要打乱indices。
            seed ：如果需要打乱，则设定一个random seed。
            drop_last ：如果不能均匀分割数据，是否需要把无法分配的尾部数据丢掉。
            epoch ：每次epoch都会shuffle数据集，如何保持shuffle之后数据集一致性？就是通过epoch完成。
                '''
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    #当迭代容器中元素时的行为，实际上是返回是一个迭代器（通常是迭代器本身），每一次迭代得到的结果会被用来作为下一次迭代的初始值。
    def __iter__(self) -> Iterator[T_co]:
        '''
        __iter__ 代码的一个技术细节是：

        indices = indices[self.rank:self.total_size:self.num_replicas]

        当一个list之中有双引号，比如 list[start:end:step]，其意义是：

        start: 起始位置
        end: 结束位置
        step: 步长
        我们用一个例子来看看，比如：

        a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        print(a[0:15:3])
        print(a[1:15:3])
        print(a[2:15:3])
        得到：

        [1, 4, 7, 10, 13]
        [2, 5, 8, 11, 14]
        [3, 6, 9, 12, 15]
        因为 indices[self.rank:self.total_size:self.num_replicas] 之中，num_replicas 实际就是rank的总数，所以这里每个worker就会严格返回自己rank对应的那部分数据序号


        总结一下DistributedSampler的分配方法是：
            每段连续的 num_replicas 个数据被拆成一个一个，
            分给 num_replicas 个进程，而且是通过每个worker 的 rank 来获取数据，
            这样就达到了不重叠不交叉的目的，
            但也要注意的是：这样每个进程拿到的数据是不连续的。
        '''
        if self.shuffle:  # 如果需要shuffle，则会基于epoch和seed进行处理
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch) # 这里设置随机种子 设置 random 种子的具体使用是在迭代函数之中：
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:  # 否则直接返回数据集长度序列
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # 是否需要补齐数据
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # 依据自己的rank，依次返回自己的数据序号
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    #当被 len() 函数调用时的行为，一般返回迭代器中元素的个数。
    def __len__(self) -> int:
        return self.num_samples

    '''
    从下面代码可以看出来，如果需要分布式训练，就对 sampler 设置 epoch。
    sampler = DistributedSampler(dataset) if is_distributed else None
    loader = DataLoader(dataset, shuffle=(sampler is None), ...,
                                sampler=sampler)
        for epoch in range(start_epoch, n_epochs):
            if is_distributed:
                sampler.set_epoch(epoch) # 这设置epoch
            train(loader)
    具体对应 DistributedSampler 的实现。
    设置 epoch 很简单，就是配置下。
    '''
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


