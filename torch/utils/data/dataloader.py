r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""


'''
0x01 数据加载
1.1 加速途径
当分布式训练时候，为了加速训练，有三个层面的工作需要处理。

数据加载层面
多机通讯层面
代码层面
在数据层面，可以使用多进程并行加载来加速数据预处理过程，也有利用GPU特点来加速，比如Nvidia DALI 通过将数据预处理放到 GPU 处理来解决 CPU 瓶颈问题。

在多机通讯层面，有各种集合通信库可以利用，比如NCCL，OpenMPI, Gloo 等。

在代码层面，可以使用框架提供的分布式API，或者利用 Horovod 来改造单机版代码，使其支持分布式任务。

接下来我们就看看数据层面如何加速。

1.2 并行处理
AI框架的数据处理主要如下并行处理：

数据加载/处理使用CPU。
训练使用GPU。
在理想状态下，应该是每轮迭代训练之前，CPU就完成加载，准备好训练数据，这样训练就可以持续无缝迭代。

然而，GPU算力每年会提升一倍，CPU的提升速度远远落后于GPU，所以CPU会是拖后腿的那个角色。这里不仅仅是CPU算力不足的问题，也包括村存储中读取数据速度不足的问题。

因此，机器学习对于数据加载和前期预处理的要求越来越高，必须在GPU计算时间内，完成下一迭代数据的准备工作，不能让GPU因为等待训练数据而空闲。

1.3 流水线
对于机器学习训练，加载数据可以分为三个步骤：

将数据从磁盘或者分布式存储加载到主机（CPU）。
将数据从主机可分页内存传输到主机固定内存。
将数据从主机固定内存转移到主机GPU。
因此，流行的深度学习框架会依据加载步骤的特点和异构硬件的特点来进行流水线处理，从而提高数据处理过程的吞吐量。

流水线一般包括多个算子，每个算子内部由数据队列组成一个缓冲区，上游算子完成处理之后会传给给下游算子进行处理。这样每个算子任务会彼此独立，算子内部可以使用细粒度的多线程/多进程来并行加速，每个算子可以独立控制处理速度和内存以适配不同网络对于处理速度的需求。

如果算子内部数据队列不为空，模型就会一直源源不断获得数据，就不会因为等待训练数据而产生瓶颈。

下面是串行处理逻辑：

+------+            +-----------+           +---------------------------+
|      |            |           |           |                           |
| Data +----------> | Load Data +---------> | Transfer to Pinned Memory |
|      |            |           |           |                           |
+------+            +-----------+           +---------------------------+
下面是并行流水线逻辑：

                    +------------+
+--------+          |            |
|        |          | Process 1  |
| Data 1 +--------> |            +------+
|        |          | Load Data  |      |
+--------+          |            |      |
                    +------------+      |
                                        |
                                        |
                                        |
                    +------------+      |        +-----------------------------------+
+--------+          |            |      |        |                                   |
|        |          | Process 2  |      +------> | Pin-memory process                |
| Data 2 +--------> |            |               |                                   |
|        |          | Load Data  +-------------> |                                   |
+--------+          |            |               |        Transfer to Pinned Memory  |
                    +------------+       +-----> |                                   |
                                         |       |                                   |
                                         |       +-----------------------------------+
                                         |
+--------+          +------------+       |
|        |          |            |       |
| Data 3 +--------> | Process 3  +-------+
|        |          |            |
+--------+          | Load Data  |
                    |            |
                    +------------+




1.4 GPU
本文到现在是解决CPU侧的数据传输问题，即：从磁盘加载数据，从可分页到固定内存。

但是，从固定内存到GPU的数据传输（tensor.cuda()）也可以使用CUDA流进行流水线处理。

另外，深度学习应用程序需要复杂的多阶段数据处理管道，包括加载、解码、裁剪、调整大小和许多其他增强功能。这些目前在 CPU 上执行的数据处理管道已经成为瓶颈，限制了训练和推理的性能和可扩展性。

Nvidia DALI 通过将数据预处理放到 GPU 处理来解决 CPU 瓶颈问题，用户可以依据自己模型的特点，构建基于 GPU 的 pipeline，或者基于CPU的pipeline。

此处有img！！




我们小结一下多进程逻辑。

总体逻辑如下：

主进程把需要获取的数据 index 放入index_queue。
子进程从 index_queue 之中读取 index，进行数据读取，然后把读取数据的index放入worker_result_queue。
主进程的 pin_memory_thread 会从 worker_result_queue 读取数据index，依据这个index进行读取数据，进行处理，把结果放入 data_queue。
具体流程如下图:

在 _MultiProcessingDataLoaderIter 的初始化函数 __init__ 之中会进行初始化：
配置，生成各种成员变量，配置各种queue。
启动各个子进程。
启动主进程中的pin_memory的线程。
调用 _reset 函数，这是进一步完善业务初始化，也用来重置环境。上面已经启动了worker子进程，但是没有分配任务，所以reset函数会进行任务分配，预取。
接下来是一个预取操作（在看下图中一定要留意）。
_try_put_index 函数就是使用sampler获取下一批次的数据index。这里 _prefetch_factor 缺省值是 2，主要逻辑如下。
使用 _next_index 从sampler获取下一批次的index。
通过 _worker_queue_idx_cycle 找出下一个可用的工作worker，然后把index分给它。
并且调整主进程的信息。
拿到index之后，回到主线程。这里会进行数据提取。就是通过index_queue, data_queue与主进程交互。
从 index_queue 获取新的数据index；
如果没有设置本worker结束，就使用 fetcher获取数据。
然后把数据放入data_queue，并且通知主进程，这里需要注意，data_queue是传入的参数，如果设置了pin memory，则传入的是 worker_result_queue，否则传入 data_queue。
当用户迭代时，调用了Loader基类的 __next__ 函数 ，其调用 _next_data 从 DataLoader 之中获取数据。
使用 _get_data 如何从 self._data_queue 中取数据。
使用_process_data 设置下一次迭代的 index，即使用 _try_put_index，_next_index 来进行下一轮设置。
具体如下图：

user        _MultiProcessingDataLoaderIter   Sampler        Queue(index_queue)    Queue(data_queue)    _worker_loop     Fetcher
 +                       +                      +                  +                     +                  +              +
 |                       |                      |                  |                     |                  |              |
 |                       |                      |                  |                     |                  |              |
 |                       v                      |                  |                     |                  |              |
 |                   __init__                   |                  |                     |                  |              |
 |               1    _reset                    |                  |                     |                  |              |
 |                       +                      |                  |                     |                  |              |
 |                       |                      |                  |                     |                  |              |
 |                       |                      |                  |                     |                  |              |
 |                       v                      |                  |                     |                  |              |
 |            2   _try_put_index     next       |                  |                     |                  |              |
 |                  _next_index  +------------> |                  |                     |                  |              |
 |                       +                      |                  |                     |                  |              |
 |                       |  <-----------------+ |                  |                     |                  |              |
 |                       |           index      |                  |                     |                  |              |
 |                       |                      |                  |                     |                  |              |
 |                       | +------------------------------------>  |                     |                  |              |
 |                       |           put        |                  |                     |       get        |              |
 |                       |                      |                  +--------------------------------------> |              |
 |                       |                      |                  |                     |                  |    index     |
 |                       |                      |                  |                     |                  +------------> |
 |         next          |                      |                  |                     |                  | <----------+ |
 +---------------------> |                      |                  |                     | <----------------+    data      |
 |                       |                      |                  |                     |      data        |              |
 |                       +                      |                  |                     |                  |              |
 |                   _next_data                 |                  |                     |                  |              |
 |              3   _get_data          get      |                  |                     |                  |              |
 |                  _try_get_data  +-------------------------------------------------->  |                  |              |
 |                       +                      |                  |                     |                  |              |
 |                       |  <----------------------------------------------------------+ |                  |              |
 |                       |             data     |                  |                     |                  |              |
 |                       +                      |                  |                     |                  |              |
 |                   _process_data              |                  |                     |                  |              |
 |                  _try_put_index     next     |                  |                     |                  |              |
 |                  _next_index +-------------> |                  |                     |                  |              |
 |                       + <--------------------+                  |                     |                  |              |
 |                       |           index      |                  |                     |                  |              |
 |                       +---------------------------------------> |                     |       get        |              |
 | <-------------------+ |             put      |                  +------------------------------------->  |     index    |
 |        data           |                      |                  |                     |                  | +----------> |
 |                       |                      |                  |                     |                  +<-----------+ |
 v                       v                      v                  v                     v                  v     data     v



至此，我们把之前的pipeline图进一步细化，具体如下：

                                                  +------------+
                              +--------+          |            |
                              |        |          | Process 1  |
                      +-----> | Data 1 +--------> |            +------+
                      |       |        |          | Load Data  |      |
                      |       +--------+          |            |      |
                      |                           +------------+      |
                      |                                               |
                      |                                               |
                      |                                               |
+----------------+    |                           +------------+      |                                          +-------------------------+
|Main process    |    |       +--------+          |            |      |                                          |  pin_memory_thread      |
|                |    |       |        |          | Process 2  |      +------>  +------------------------+       |                         |          +------------+
|  index_queue   +----------> | Data 2 +--------> |            |                |                        |       |                         |          |            |
|                |    |       |        |          | Load Data  +------------->  |  _worker_result_queue  +-----> |  Write to pinned memory +--------> | data_queue |
|                |    |       +--------+          |            |                |                        |       |                         |          |            |
+----------------+    |                           +------------+       +----->  |                        |       |                         |          +------------+
                      |                                                |        +------------------------+       |                         |
                      |                                                |                                         +-------------------------+
                      |                                                |
                      |       +--------+          +------------+       |
                      |       |        |          |            |       |
                      +-----> | Data 3 +--------> | Process 3  +-------+
                              |        |          |            |
                              +--------+          | Load Data  |
                                                  |            |
                                                  +------------+


至此，PyTorch 分布式的数据加载部分分析完毕，下一篇我们回归到 Paracel 如何处理数据加载。

'''
import functools
import itertools
import logging
import os
import queue
import threading
import warnings

from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union

import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.utils.data.graph_settings

from torch._utils import ExceptionWrapper

from . import (
    IterDataPipe,
    MapDataPipe,
    IterableDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    Dataset,)

from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from . import _utils

__all__ = [
    "DataLoader",
    "get_worker_info",
    "default_collate",
    "default_convert",
]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]


# These functions used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate
default_convert = _utils.collate.default_convert

get_worker_info = _utils.worker.get_worker_info

logger = logging.getLogger(__name__)


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self):
        super().__init__(None)

    def __iter__(self):
        while True:
            yield None


def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    info = torch.utils.data.get_worker_info()
    assert info is not None
    total_workers = info.num_workers
    datapipe = info.dataset
    assert isinstance(datapipe, (IterDataPipe, MapDataPipe))
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, world_size, rank_id, sharding_group=SHARDING_PRIORITIES.DISTRIBUTED)
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, total_workers, worker_id, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


def _share_dist_seed(generator, pg):
    _shared_seed = torch.empty((), dtype=torch.int64).random_(generator=generator)
    if isinstance(pg, dist.ProcessGroup):
        dist.broadcast(_shared_seed, src=0, group=pg)
    return _shared_seed.item()

'''
DataLoader的作用是：结合Dataset和Sampler之后，在数据集上提供了一个迭代器。
可以这么理解：
    DataSet 是原始数据，Sampler 提供了如何切分数据的策略（或者说是提供了切分数据的维度），
    DataLoader就是依据策略来具体打工干活的，其中单进程加载就是一个人干活，多进程加载就是多拉几个人一起干活。
'''
class DataLoader(Generic[T_co]):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            `base_seed` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise if value of num_workers>0 default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): the data loader will copy Tensors
            into device pinned memory before returning them if pin_memory is set to true.


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.
    """
    '''
    dataset (Dataset) ：所加载的数据集。
    batch_size (int, optional) ：每个批次加载多少个样本。
    shuffle (bool, optional) ：如果为 True，则每个epoch 都会再打乱数据。
    sampler (Sampler or Iterable, optional) ：定义了如何从样本采样的策略。可以是任何实现了 __len__的迭代器。
    batch_sampler (Sampler or Iterable, optional) ：与sampler类似，但是每次返回一个批次的数据索引。
    num_workers (int, optional) ：数据加载的子进程数目。如果是 0，表示从主进程加载数据。
    collate_fn (callable, optional)：从一个小批次（ mini-batch）张量中合并出一个样本列表。当从 map-style 数据集做批量加载时候使用。
    pin_memory (bool, optional) : 如果为true，则在返回张量之前把张量拷贝到CUDA固定内存之中。
    drop_last (bool, optional) ：当数据集不能被均匀分割时，如果为true，丢掉最后一个不完整的批次。如果为False，那么最后一个批次的数据较小。
    timeout (numeric, optional): 如果是整数，则是worker收集批次数据的超时值。
    worker_init_fn (callable, optional)：如果非空，则会在seeding和数据加载之前被每个子进程调用，以Iworker id ([0, num_workers - 1])作为输入参数。
    generator (torch.Generator, optional)：如果非空，则被RandomSampler 用来产生随机索引，也被多进程用来产生 base_seed 。
    prefetch_factor (int, optional, keyword-only arg)：每个 worker 提前加载 的 sample 数量。
    persistent_workers (bool, optional)：如果为 True, 则在消费一次之后，data loader也 不会关掉worker进程。这允许workerDataset实例维持活动状态。
    '''
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.')
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError('prefetch_factor option should be non-negative')

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   _DataPipeSerializationWrapper container makes it easier to serialize without redefining pickler
        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if isinstance(dataset, IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
            # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
            elif shuffle not in {False, None}:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map



        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

    '''
    为了加速，PyTorch提供了多进程下载，
    只要把将参数 num_workers 设置为正整数，系统就会相应生成多进程处理，
    在这种模式下，每个worker都是一个独立进程。

    由上节我们可以知道，_SingleProcessDataLoaderIter 是单进程加载数据的核心，loader通过它来与sampler，dataset交互。
    在多进程中，这个核心对应的就是 _MultiProcessingDataLoaderIter。
    '''
    #具体会依据是否是多进程来区别生成。
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))
                    # error: Argument 1 to "get_context" has incompatible type "Union[str, bytes]"; expected "str"  [arg-type]
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)  # type: ignore[arg-type]

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super().__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    '''
    单进程模式下，Data Loader会在计算进程内加载数据，所以加载过程中可能会阻塞计算。

    for 语句会调用enumerate 会返回一个迭代器，以此来遍历数据集。
    在eumerate之中，dataloader 的 __next__(self) 方法会被调用，逐一获取下一个对象，从而遍历数据集。
            cuda0 = torch.device('cuda:0')  # CUDA GPU 0
            for i, x in enumerate(train_loader):
                x = x.to(cuda0)
                
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++            
    for data, label in train_loader:
        ......
    for 循环会调用 dataloader 的 __iter__(self) 方法，以此获得迭代器来遍历 dataset。
    在 __iter__(self) 方法中，dataloader 调用了 self._get_iterator() 方法，根据 num_workers 获得迭代器，并指示是进行单进程还是多进程处理。

    '''
    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused

        if self.persistent_workers and self.num_workers > 0:  # 如果是多进程或者设置了持久化
            #当多进程加载时候，在DataLoader声明周期之中，迭代器只被建立一次，这样worker可以重用迭代器。
            if self._iterator is None:  # 如果没有，才会新生成
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else: # 单进程
            return self._get_iterator() # 每次都直接生成新的  在单进程加载时候，应该每次生成，以避免重置状态。

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    '''
    从这里看出，dataloader 提供了 sampler（可以是batch_sampler 或者是其他 sampler 子类），
    然后 _SingleProcessDataLoaderIter 迭代 sampler 获得索引。

下面我们来看看 fetcher，fetcher 需要 index 来获取元素，
并同时支持 Map-style dataset（对应 _MapDatasetFetcher）和 Iterable-style dataset（对应 _IterableDatasetFetcher），
使其在 Dataloader 内能使用相同的接口 fetch，代码更加简洁。

    '''
    #这里关键函数之一就是_index_sampler，用来让迭代器调用sampler，我们接下来就会讲到
    @property
    def _index_sampler(self): #关键函数之一就是_index_sampler，用来让迭代器调用sampler
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def check_worker_number_rationality(self):
        # This function check whether the dataloader's worker number is rational based on
        # current system's resource. Current rule is that if the number of workers this
        # Dataloader will create is bigger than the number of logical cpus that is allowed to
        # use, than we will pop up a warning to let user pay attention.
        #
        # eg. If current system has 2 physical CPUs with 16 cores each. And each core support 2
        #     threads, then the total logical cpus here is 2 * 16 * 2 = 64. Let's say current
        #     DataLoader process can use half of them which is 32, then the rational max number of
        #     worker that initiated from this process is 32.
        #     Now, let's say the created DataLoader has num_works = 40, which is bigger than 32.
        #     So the warning message is triggered to notify the user to lower the worker number if
        #     necessary.
        #
        #
        # [Note] Please note that this function repects `cpuset` only when os.sched_getaffinity is
        #        available (available in most of Linux system, but not OSX and Windows).
        #        When os.sched_getaffinity is not available, os.cpu_count() is called instead, but
        #        it doesn't repect cpuset.
        #        We don't take threading into account since each worker process is single threaded
        #        at this time.
        #
        #        We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
        #        other than `torch.set_num_threads` to 1 in the worker process, if the passing
        #        in functions use 3rd party modules that rely on those threading flags to determine
        #        how many thread to create (eg. numpy, etc), then it is caller's responsibility to
        #        set those flags correctly.
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):

            suggested_max_worker_msg = ((
                "Our suggested max number of worker in current system is {}{}, which is smaller "
                "than what this DataLoader is going to create.").format(
                    num_worker_suggest,
                    ("" if cpuset_checked else " (`cpuset` is not taken into account)"))
            ) if num_worker_suggest is not None else (
                "DataLoader is not able to compute a suggested max number of worker in current system.")

            warn_msg = (
                "This DataLoader will create {} worker processes in total. {} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary.").format(
                    num_worker_created,
                    suggested_max_worker_msg)
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satify mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(_create_warning_msg(
                max_num_worker_suggest,
                self.num_workers,
                cpuset_checked))

'''
_BaseDataLoaderIter 是迭代器基类，我们挑选关键函数看看。
这里关键成员变量就是：
    _index_sampler：这里设置了loader 的 sampler，所以迭代器可以据此获取采样策略。
    _sampler_iter：得到 sampler 的迭代器。
    
_BaseDataLoaderIter 是所有 DataLoaderIter 的父类。
dataloader获得了迭代器之后，for 循环需要调用 __next__() 来获得下一个对象，从而实现遍历。
通过 __next__() 方法调用 _next_data() 获取数据。
'''
class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        # 初始化赋值一些 DataLoader 参数，
        # 以及用户输入合法性进行校验
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler # 得到采样策略
        self._num_workers = loader.num_workers
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank
        # for other backends, pin_memory_device need to set. if not set
        # default behaviour is CUDA device. if pin_memory_device is selected
        # and pin_memory is not set, the default behaviour false.
        if (len(loader.pin_memory_device) == 0):
            self._pin_memory = loader.pin_memory and torch.cuda.is_available()
            self._pin_memory_device = None
        else:
            if not loader.pin_memory:
                warn_msg = ("pin memory device is set and pin_memory flag is not used then device pinned memory won't be used"
                            "please set pin_memory to true, if you need to use the device pin memory")
                warnings.warn(warn_msg)

            self._pin_memory = loader.pin_memory
            self._pin_memory_device = loader.pin_memory_device
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler) # 得到sampler的迭代器
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)

    '''
    _try_put_index 函数就是使用sampler获取下一批次的数据index。这里 _prefetch_factor 缺省值是 2，主要逻辑如下。
        从sampler获取下一批次的index。
        通过 _worker_queue_idx_cycle 找出下一个可用的工作worker，然后把index分给它。
        并且调整主进程的信息。
    '''
    def _next_index(self):
        # 定义在基类 _BaseDataLoaderIter 之中，就是获取下一批index
        # sampler_iter 来自于 index_sampler
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError
    '''
    现在数据已经加载完毕，我们接下来看用户如何从DataLoader之中获取数据。
    这里有一个很关键的地方：如何保持在不同实验之中数据读取顺序的一致性。
    为了让多次实验之间可以比对，就需要尽量保证在这些实验中，每次读取数据的顺序都是一致的，这样才不会因为数据原因造成结果的误差。
    打破顺序一致性的最大可能就是乱序数据。而造成乱序问题的原因就是：多进程读取，可能某个进程快，某个进程慢。
    比如，用户这次需要读取6-19，16-26，37-46。但是某一个worker慢，6-19不能即时返回，另一个worker 的 16-26 先返回了，于是就会造成乱序。
    如何处理乱序数据？PyTorch的具体做法就是：DataLoader严格按照Sampler的顺序返回数据。
    如果某一个数据是乱序的，则会把它暂存起来，转而去获取下一个数据，见下面代码中 "store out-of-order samples" 注释处。
    等到应该返回时候（这个数据顺序到了）才返回。
    但是其风险就是数据返回会比当前请求慢，比如应该获取 6，但是Data queue里面没有这个数据，只有 16，27，于是用户只能等待 6 加载完成。
    解决慢的方法是：预取（prefetch）。就是在reset方法最后，提前提取若干index，让DataLoader提前去取，
    这虽然不能保证任意两次训练的数据返回顺序完全一致，但是可以最大限度保证。
    具体代码如下，首先，回忆基类的 __next__ 函数 ，可以看到其调用了 _next_data 获取数据。
    '''
    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()  # 获取数据  # 重点代码行，通过此获取数据
            self._num_yielded += 1
            if self._dataset_kind == _DatasetKind.Iterable and \
                    self._IterableDataset_len_called is not None and \
                    self._num_yielded > self._IterableDataset_len_called:
                # 忽略错误提示处理
                warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                            "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                  self._num_yielded)
                if self._num_workers > 0:
                    warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                 "IterableDataset replica at each worker. Please see "
                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                warnings.warn(warn_msg)
            return data

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)

'''
_SingleProcessDataLoaderIter 继承了 _BaseDataLoaderIter，可以看到，其增加了 _dataset_fetcher，在构造时候传入了 _collate_fn 等各种参数。
回忆下，__next__会调用 self._next_data() 获取数据，而在这里，_next_data 就会：
    使用 self._next_index()，其又会使用 _sampler_iter（采样器的迭代器）来获取indices 。
    使用 self._dataset_fetcher.fetch(index)来依据indices获取数据。
    

从 _SingleProcessDataLoaderIter 的初始化参数可以看到，
其在父类 _BaseDataLoaderIter 的基础上定义了 _dataset_fetcher，
并传入 _dataset，_auto_collation，_collate_fn 等参数，
用于定义获取数据的方式。其具体实现会在稍后解释。
在 _next_data() 被调用后，其需要 _next_index() 获取 index，并通过获得的 index 传入 _dataset_fetcher 中获取对应样本。
'''
class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            torch.utils.data.graph_settings.apply_sharding(
                self._dataset, self._world_size, self._rank, sharding_group=SHARDING_PRIORITIES.DISTRIBUTED)

        # 获取样本方法
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        # 获取样本
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data

'''
_MultiProcessingDataLoaderIter 中的注释十分详尽，值得大家深读，而且给出了逻辑流程图如下，其基本流程是围绕着三个queue进行的:
    主进程把需要获取的数据 index 放入index_queue，这是指定子进程需要获取哪些数据的队列。同时也给子进程传入结果队列，关于结果队列，有两个分支：
        如果设置了pin memory，则传入的是 worker_result_queue。
        否则传入 data_queue。
    子进程从 index_queue 之中读取 index，进行数据读取，然后把读取数据的index放入worker_result_queue，这是向主进程返回结果的队列。
    主进程进行处理，这里有两个分支：
        如果设置了pin memory，则主进程的 pin_memory_thread 会从 worker_result_queue 读取数据index，依据这个index进行读取数据，进行处理，把结果放入 data_queue，这是处理结果的队列。
        如果不需要pin memory，则结果已经存在 data_queue 之中，不做新操作。
    可以看到，每个进程的输入是一个队列index_queue ，输出也是一个队列worker_result_queue。主进程和子进程通过这2~3个 queue 联系了起来，从而达到解耦合和加速的作用。

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.

具体如下图所示，如果不需要 pin memory，则为：

                                               +-----------+
               indices  -------------+ indices | Worker    | Data
             +--------->+index queue +-------->+ Process   +------+
             |          |            |         |           |      |
             |          -------------+         +-----------+      |
             |                                                    |   +------------+
             |                                                    |   |            |
+---------+  |                                                    +--->            |
| Main    |  | indices  -------------+ indices +-----------+          |            |
| Process +------------>+index queue +-------->+ Worker    | Data     | Data Queue |
|         |  |          |            |         | Process   +---------->            |
+---------+  |          -------------+         |           |          |            |
             |                                 +-----------+      +--->            |
             |                                                    |   +------------+
             |                                                    |
             | indices  -------------+ indices +-----------+      |
             +--------->+index queue +-------->+ Worker    | Data |
                        |            |         | Process   +------+
                        -------------+         |           |
                                               +-----------+

当有pin memory时候，则是先进入 result queue，然后 pin_memory_thread 处理之后会转入到 data queue：

                                               +-----------+
               indices  -------------+ indices | Worker    | Data
             +--------->+index queue +-------->+ Process   +------+
             |          |            |         |           |      |
             |          -------------+         +-----------+      |
             |                                                    |   --------------+
             |                                                    |   |             |
+---------+  |                                                    +--->             |
| Main    |  | indices  -------------+ indices +-----------+          |             |
| Process +------------>+index queue +-------->+ Worker    | Data     | result_queue|
|         |  |          |            |         | Process   +---------->             |
+---------+  |          -------------+         |           |          |             |
             |                                 +-----------+      +--->             |
             |                                                    |   ---------+----+
             |                                                    |            |
             | indices  -------------+ indices +-----------+      |            |
             +--------->+index queue +-------->+ Worker    | Data |  +---------+--------+
                        |            |         | Process   +------+  | pin_memory_thread|
                        -------------+         |           |         |         |        |
                                               +-----------+         |         |        |
                                                                     |         |        |
                                                                     +------------------+
                                                                               |
                                                                               |
                                                                               |
                                                                               v
                                                                         +-----+------+
                                                                         | Data Queue |
                                                                         |            |
                                                                         +------------+

2.4.2 初始化
初始化函数如下，主要是：
    配置，生成各种成员变量，配置各种queue。
    启动各个子进程。
    启动主进程中的pin_memory的线程。
主要成员变量为：
    _index_queues: 这是一个queue 列表，列表的每一个元素是一个 queue，就是每个子进程的队列需要处理的数据index，每个子进程对应一个 queue。
    _worker_result_queue: 子进程处理完的 (idx, data)。
    data_queue: 经过主进程 pin_memory 线程处理之后的数据队列，如果不需要pin，则直接会使用 _worker_result_queue。
    _worker_queue_idx_cycle 用以找出下一个工作的worker。



每个 worker 一次产生一个 batch 的数据，返回 batch 数据前放入下一个批次要处理的数据下标，对应构造函数子进程初始化如下：
'''
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    '''
    _MultiProcessingDataLoaderIter有如下 flag 参数来协调各个 worker （包括各种queue）之间的工作：

    _send_idx: 发送索引，用来记录这次要放 index_queue 中 batch 的 idx

    _rcvd_idx: 接受索引，记录要从 data_queue 中取出的 batch 的 idx

    _task_info: 存储将要产生的 data 信息的 dict，key为 task idx（由 0 开始的整型索引），
                value 为 (worker_id,) 或 (worker_id, data)，
                分别对应数据 未取 和 已取 的情况

    _tasks_outstanding: 整型，代表已经准备好的 task/batch 的数量（可能有些正在准备中）
    '''
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may alreay be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children.
    #
    #      Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed (which, at least in
    #      CPython, is done via an `atexit` handler defined in
    #      `multiprocessing/util.py`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
    #      registered when an object requiring this mechanism is first
    #      created, e.g., `mp.Queue`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
    #      )
    #
    #      So in `__del__`, we check if `_utils.python_exit_status` is set or
    #      `None` (freed), and perform no-op if so.
    #
    #      However, simply letting library clean-up codes run can also be bad,
    #      because such codes (i.e., `multiprocessing.util._exit_function()`)
    #      include join putting threads for `mp.Queue`, which can be blocking.
    #      Hence, the main process putting threads are called with
    #      `cancel_join_thread` at creation.  See later section
    #      [ 3b. A process won't hang when putting into a queue; ]
    #      for more details.
    #
    #      Here are two example cases where library clean-up codes can run
    #      before `__del__` is called:
    #
    #        1. If we hold onto a reference to the iterator, it more often
    #           than not tries to do `multiprocessing` library cleaning before
    #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
    #           and thus prevents our cleaning-up code to run first.
    #
    #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
    #           When a process ends, it shuts the all its daemonic children
    #           down with a SIGTERM (instead of joining them without a timeout).
    #           Simiarly for threads, but by a different mechanism. This fact,
    #           together with a few implementation details of multiprocessing, forces
    #           us to make workers daemonic. All of our problems arise when a
    #           DataLoader is used in a subprocess, and are caused by multiprocessing
    #           code which looks more or less like this:
    #
    #               try:
    #                   your_function_using_a_dataloader()
    #               finally:
    #                   multiprocessing.util._exit_function()
    #
    #           The joining/termination mentioned above happens inside
    #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #           throws, the stack trace stored in the exception will prevent the
    #           frame which uses `DataLoaderIter` to be freed. If the frame has any
    #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #           its  `__del__`, which starts the shutdown procedure, will not be
    #           called. That, in turn, means that workers aren't notified. Attempting
    #           to join in `_exit_function` will then result in a hang.
    #
    #           For context, `_exit_function` is also registered as an `atexit` call.
    #           So it is unclear to me (@ssnl) why this is needed in a finally block.
    #           The code dates back to 2008 and there is no comment on the original
    #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #           the finally block and the `atexit` registration) that explains this.
    #
    #
    #      Finally, another choice is to just shutdown workers with logic in 1
    #      above whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           *exits*.
    #
    #           In case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever.  The usual
    #           solution for this in Python is calling  `q.cancel_join_thread`,
    #           which prevents automatically joining it when finalizing
    #           (exiting).
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           Hence,
    #             + For worker processes, we only do so (for their output
    #               queues, i.e., `worker_result_queue`) before exiting.
    #             + For `pin_memory_thread`, its output queue `data_queue` is a
    #               `queue.Queue` that does blocking `put` if the queue is full.
    #               So there is no above problem, but as a result, in
    #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
    #               that breaks not only upon success, but also when the main
    #               process stops reading, i.e., is shutting down.
    #             + For loader process, we `cancel_join_thread()` for all
    #               `_index_queues` because the whole purpose of workers and
    #               `pin_memory_thread` is to serve the loader process.  If
    #               loader process is already exiting, we don't really care if
    #               the queues are corrupted.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separete events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data:
    #                If is fetching from an `IterableDataset` and the iterator
    #                    is exhausted, send an `_IterableDatasetStopIteration`
    #                    object to signal iteration end. The main process, upon
    #                    receiving such an object, will send `None` to this
    #                    worker and not use the corresponding `index_queue`
    #                    anymore.
    #       If timed out,
    #          No matter `workers_done_event` is set (still need to see `None`)
    #          or not, must continue to next iteration.
    #   (outside loop)
    #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
    #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
    #                                          main process won't read from it;
    #                                          other workers will also call
    #                                          `cancel_join_thread`.)
    #
    # [ pin_memory_thread ]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While `pin_memory_thread_done_event` is not set:
    #     Get from `index_queue`.
    #       If timed out, continue to get in the next iteration.
    #       Otherwise, process data.
    #       While `pin_memory_thread_done_event` is not set:
    #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
    #         If timed out, continue to put in the next iteration.
    #         Otherwise, break, i.e., continuing to the out loop.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [ main process ]
    #   In the DataLoader Iter's `__del__`
    #     b. Exit `pin_memory_thread`
    #          i.   Set `pin_memory_thread_done_event`.
    #          ii   Put `None` in `worker_result_queue`.
    #          iii. Join the `pin_memory_thread`.
    #          iv.  `worker_result_queue.cancel_join_thread()`.
    #
    #     c. Exit the workers.
    #          i.   Set `workers_done_event`.
    #          ii.  Put `None` in each worker's `index_queue`.
    #          iii. Join the workers.
    #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
    #
    #        NOTE: (c) is better placed after (b) because it may leave corrupted
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timeing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader):
        super().__init__(loader)

        self._prefetch_factor = loader.prefetch_factor

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank)

        # No certainty which module multiprocessing_context is
        # # 把该worker取出的数放入该队列，用于进程间通信
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated] # 子进程输出，读取完数据的index
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = [] # 子进程输入，需读取数据的index
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]  # 索引队列，每个子进程一个队列放要处理的下标
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            # _worker_loop 的作用是：从index_queue中取索引，然后通过collate_fn处理数据，
            # 然后再将处理好的 batch 数据放到 data_queue 中。（发送到队列中的idx是self.send_idx）
            w = multiprocessing_context.Process(
                ## 每个worker子进程循环执行的函数，主要将数据以(idx, data)的方式传入_worker_result_queue中
                target=_utils.worker._worker_loop,  # worker进程主函数，把各种queue和函数传进去
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers, self._shared_seed))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue) # 把这个worker对应的index_queue放到主进程这里存起来，以后就可以交互了
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            # 用于存取出的数据进行 pin_memory 操作后的结果
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]  # pin 处理之后的数据结果
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      current_device,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue # 如果不需要pin，则直接使用_worker_result_queue

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True) # 继续完善业务 # __init__ 函数最后会调用 _reset 函数，这是进一步完善业务初始化，也用来重置环境

    '''
    __init__ 函数最后会调用 _reset 函数，这是进一步完善业务初始化，也用来重置环境。

    上小节函数中，已经启动了worker子进程，但是没有分配任务，所以_reset函数会进行任务分配，预取。
    
    _MultiProcessingDataLoaderIter有如下 flag 参数来协调各个 worker （包括各种queue）之间的工作：
    
    _send_idx: 发送索引，用来记录这次要放 index_queue 中 batch 的 idx
    
    _rcvd_idx: 接受索引，记录要从 data_queue 中取出的 batch 的 idx
    
    _task_info: 存储将要产生的 data 信息的 dict，key为 task idx（由 0 开始的整型索引），value 为 (worker_id,) 或 (worker_id, data)，分别对应数据 未取 和 已取 的情况
    
    _tasks_outstanding: 整型，代表已经准备好的 task/batch 的数量（可能有些正在准备中）
    
    _send_idx: 发送索引，记录下一次要放 index_queue 中 task batch 的 idx。
    
    _rcvd_idx: 接受索引，记录下一次要从 data_queue 中取出的 task batch 的 idx。_send_idx 和 _rcvd_idx 主要用来进行流量控制和确保接受索引有意义。
    
    _task_info: 存储将要产生的 data 信息的 dict，key为 task batch idx（由 0 开始的整型索引），value 为 (worker_id,) 或 (worker_id, data)，分别对应数据 未取 和 已取 的情况。_task_info的作用是依据 task batch idx 获取对应的 worker id 和暂存乱序数据。
    
    _tasks_outstanding: 整型，正在准备的 task/batch 的数量，实际上就是进行一些确认工作，没有太实际的意义。
    
    对于加载数据，每个 worker 一次产生一个 batch 的数据，返回 batch 数据前，会放入下一个批次要处理的数据下标，所以 reset 函数会把 _send_idx，_rcvd_idx 都恢复成0，这样下次迭代就可以重新处理。
    
    在 reset 方法最后，有一个预取数据操作。我们会在后面结合乱序处理进行讲解。
    '''
    def _reset(self, loader, first_iter=False):
        '''
        对于加载数据，每个 worker 一次产生一个 batch 的数据，返回 batch 数据前，
        会放入下一个批次要处理的数据下标，
        所以 reset 函数会把 _send_idx，_rcvd_idx 都恢复成0，
        这样下次迭代就可以重新处理。

        在 reset 方法最后，有一个预取数据操作。我们会在后面结合乱序处理进行讲解。
        '''
        super()._reset(loader, first_iter)
        # 发送索引，用来记录这次要放 index_queue 中 batch 的 idx
        self._send_idx = 0  # idx of the next task to be sent to workers

        #接受索引，记录要从 data_queue 中取出的 batch 的 idx
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__

        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}

        ## _tasks_outstanding 指示当前已经准备好的 task/batch 的数量（可能有些正在准备中）
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.

        # 每个worker的状态
        self._workers_status = [True for i in range(self._num_workers)]
        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration(self._shared_seed))
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1

        # 预取若干index，目的是为了配合后续的乱序处理。
        # prime the prefetch loop
        #DataLoader 通过指定 prefetch_factor （默认为 2）来进行数据的预取。  prefetch 功能仅适用于多进程加载中
        ## 初始化的时候，就将 2*num_workers 个 (batch_idx, sampler_indices) 放到 index_queue 中
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()  # 进行预取

    #_try_get_data 就是从 _data_queue 读取。主进程和worker进程通过queue上的put, get进行通讯交互。
    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        #_try_get_data 就是从 _data_queue 读取。主进程和worker进程通过queue上的put, get进行通讯交互。

        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

# NOTE [ DataLoader on Linux and open files limit ]
#
# On Linux when DataLoader is used with multiprocessing we pass the data between
# the root process and the workers through SHM files. We remove those files from
# the filesystem as soon as they are created and keep them alive by
# passing around their file descriptors through AF_UNIX sockets. (See
# docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
# the wiki (https://github.com/pytorch/pytorch/wiki).)
#
# This sometimes leads us to exceeding the open files limit. When that happens,
# and the offending file descriptor is coming over a socket, the `socket` Python
# package silently strips the file descriptor from the message, setting only the
# `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
# it _indicates that some control data were discarded due to lack of space in
# the buffer for ancillary data_). This might reflect the C implementation of
# AF_UNIX sockets.
#
# This behaviour can be reproduced with the script and instructions at the
# bottom of this note.
#
# When that happens, the standard Python `multiprocessing` (and not
# `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
#
# Sometimes, instead of the FD being stripped, you may get an `OSError:
# Too many open files`, both in the script below and in DataLoader. However,
# this is rare and seems to be nondeterministic.
#
#
#   #!/usr/bin/env python3
#   import sys
#   import socket
#   import os
#   import array
#   import shutil
#   import socket
#
#
#   if len(sys.argv) != 4:
#       print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
#       sys.exit(1)
#
#   if __name__ == '__main__':
#       dirname = sys.argv[1]
#       sock_path = dirname + "/sock"
#       iterations = int(sys.argv[2])
#       def dummy_path(i):
#           return dirname + "/" + str(i) + ".dummy"
#
#
#       if sys.argv[3] == 'send':
#           while not os.path.exists(sock_path):
#               pass
#           client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#           client.connect(sock_path)
#           for i in range(iterations):
#               fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
#               ancdata = array.array('i', [fd])
#               msg = bytes([i % 256])
#               print("Sending fd ", fd, " (iteration #", i, ")")
#               client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])
#
#
#       else:
#           assert sys.argv[3] == 'recv'
#
#           if os.path.exists(dirname):
#               raise Exception("Directory exists")
#
#           os.mkdir(dirname)
#
#           print("Opening socket...")
#           server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
#           server.bind(sock_path)
#
#           print("Listening...")
#           for i in range(iterations):
#               a = array.array('i')
#               msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
#               assert(len(ancdata) == 1)
#               cmsg_level, cmsg_type, cmsg_data = ancdata[0]
#               a.frombytes(cmsg_data)
#               print("Received fd ", a[0], " (iteration #", i, ")")
#
#           shutil.rmtree(dirname)
#
# Steps to reproduce:
#
# 1. Run two shells and set lower file descriptor limit in the receiving one:
# (shell1) ulimit -n 1020
# (shell2) ulimit -n 1022
#
# 2. Run the script above with the `recv` option in the first shell
# (shell1) ./test_socket.py sock_tmp 1017 recv
#
# 3. Run the script with the `send` option in the second shell:
# (shell2) ./test_socket.py sock_tmp 1017 send

    def _get_data(self):
        '''
        其次，我们看看 _get_data 如何从 self._data_queue 中取数据。
        具体是使用 _try_get_data 来提取。

            如果有超时配置，就按照超时读取。
            如果设置了pin memory，则从pin 线程处理之后的数据读取。
            否则循环读取worker处理的数据，直至获取到数据为止。
        '''
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        if self._timeout > 0: # 如果有超时配置，就按照超时读取
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        elif self._pin_memory:  # 从pin 线程处理之后的数据读取
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data() # 读取worker处理的数据
                if success:
                    return data

    '''
    所以，我们要看 _MultiProcessingDataLoaderIter 的_next_data。

        因为之前有预取了index，worker进程已经开始获取数据，所以主进程此时可以得到数据，如果没有数据，就继续while True等待。
        如果获取成功，则使用 _process_data 设定下一次的indx，准备下一次迭代。
        通过 _task_info 来记录乱序数据，如果暂时无法处理，就在这里保存。
    '''
    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.

            # 找到待取idx
            while self._rcvd_idx < self._send_idx: # 如果 待取batch idx < 已取batch idx  # 确保待处理的任务(待取的batch)下标 > 处理完毕要返回的任务(已经取完的batch)下标
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break   # 有数据或者正在工作，就跳出内部这个while
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)  # 设定下一次的indx，进行下一次迭代

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()  # 从 self._data_queue 中取数据  # 调用 self._try_get_data() 从 self._data_queue 中取数
            self._tasks_outstanding -= 1  # 正在准备的batch个数需要减1  # 表明预备好的batch个数需要减1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx: # 乱序数据
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]  # 正常数据
                return self._process_data(data)  # 设定下一次的indx，进行下一次迭代 # 返回数据

    def _try_put_index(self):
        '''
        _try_put_index 函数就是使用sampler获取下一批次的数据index。
        这里 _prefetch_factor 缺省值是 2，主要逻辑如下。
            从sampler获取下一批次的index。
            通过 _worker_queue_idx_cycle 找出下一个可用的工作worker，然后把index分给它。
            并且调整主进程的信息。
        '''

        # self._prefetch_factor 默认为 2
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index() # 获取下一批index
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]: # 如果已经工作，就继续找
                break
        else:
            # not found (i.e., didn't break)
            return

        # 以下是主进程进行相关记录
        # 给下一个工作worker放入 (任务index, 数据index), 就是给queue放入数据，
        # 所以worker loop之中就立刻会从queue中得到index，从而开始获取数据。
        self._index_queues[worker_queue_idx].put((self._send_idx, index)) # 放入 任务下标 和 数据下标

        # 记录 将要产生的 data 信息
        self._task_info[self._send_idx] = (worker_queue_idx,)

        # 正在处理的batch个数+1
        self._tasks_outstanding += 1  # _tasks_outstanding + 1，表明预备好的batch个数+1

        # send_idx 记录从sample_iter中发送索引到index_queue的次数  # send_idx 发送索引, 记录从sample_iter中发送索引到index_queue的次数
        self._send_idx += 1 # 递增下一批发送的task index

    # 设置下一次迭代是使用_process_data。
    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index() # 设定下一次的indx，进行下一次迭代   # 同上，主要放入队列索引 以及 更新flag
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data # 返回数据   #这样，多进程模式的 dataloader 就能通过多个 worker 的协作来共同完成数据的加载

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

    # staticmethod is used to remove reference to `_MultiProcessingDataLoaderIter`
    @staticmethod
    def _clean_up_worker(w):
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        self._shutdown_workers()
