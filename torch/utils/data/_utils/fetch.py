r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""


class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()

'''
对于 Iterable-style，因为 __init__ 方法内设置了 dataset 初始的迭代器，所以在fetch 方法内获取元素的时候，
如果是常规 sampler，index 其实已经不起作用，直接从dataset迭代器获取。如果是batch sampler，则index有效果。

此时总逻辑如下：

     +--------------------------+            +-------------------------------+
     | DataLoader               |            | _SingleProcessDataLoaderIter  |
     |                          |            |                               |
     |                          |            |               __next__        |
+---------------+ Sampler       |            |                               |
|    |                          |            |              _next_data +-----------+
|    |            Dataset       |            |                               |     |
|    |                          |            |              _next_index      |     |
|    |           __iter__       |            |                               |     |
|    |                          |            |             _index_sampler    |     |
|    |       _get_iterator  +--------------> |                    +          |     |
|    |                          |            |                    |          |     |
|    +--------------------------+            +-------------------------------+     |
|                                                                 |                |
|                                                                 |                |
|                                                                 |                |
|                                                                 |                |
|                                                                 |                |
|                           +----------------------------+        |                |
|                           |Sampler                     |        |                |
+------------------------>  |                            | <------+                |
                            |                            |                         |
                            |                            |                         |
                            |                            |                         |
                            +----------------------------+                         |
                                                                                   |
                                                                                   |
                            +----------------------------+                         |
                            |_BaseDatasetFetcher         |                         |
                            |                            |                         |
                            |                            |                         |
                            |          dataset           |                         |
                            |                            |  <----------------------+
                            |          collate_fn        |
                            |                            |
                            +----------------------------+

动态流程如下：

  User              DataLoader    _SingleProcessDataLoaderIter _DatasetKind   Sampler

    +                   +                    +                        +           +
    |                   |                    |                        |           |
    |         1         |                    |                        |           |
 enumerate-------->  __iter__                |                        |           |
    |                   +                    |                        |           |
    |                   |                    |                        |           |
    |                   |                    |                        |           |
    |                   |          2         v            3           v           |
    |              _get_iterator--------> __init__  +----------> create_fetcher   |
    |         4         |                    +                        +           |
    | <-----------------+                    |                        |           |
    |      iterator     |                    |                        |           |
    |                   |          5         |                        |           |
for loop +------------------------------> __next__                    |           |
    |                   |                    |                        |           |
    |                   |                    |                        |           |
    |                   |                    |                        |           |
    |                   |                _next_data                   |           |
    |                   |                    |                        |           |
    |                   |                    |                        |           |
    |                   |                    |           6  next      |           |
    |                   |                _next_index  +-------------------------> |
    |                   |                    |                        |           |
    |                   |                    |  <---------------------------------+
    |                   |                    |           7  index     |           |
    |                   |                    |                        |           |
    |                   |                    |                        |           |
    |                   |                    |        8 fetch(index)  |           |
    |                   |                    | +--------------------> |           |
    |                   |                    |                        |           |
    |                   |                    |  <---------------------+           |
    |                   |                    |         9  data        |           |
    |  <-------------------------------------+                        |           |
    |   10  data        |                    |                        |           |
    |                   |                    |                        |           |
    v                   v                    v                        v           v













最后，我们通过索引传入 fetcher，fetch 得到想要的样本。因此，整个过程调用关系总结如下：

loader.iter --> self._get_iterator() --> class _SingleProcessDataLoaderIter --> class _BaseDataLoaderIter --> __next__() --> self._next_data() --> self._next_index() -->next(self._sampler_iter) 即 next(iter(self._index_sampler)) --> 获得 index --> self._dataset_fetcher.fetch(index) --> 获得 data

而对于多进程而言，借用 PyTorch 内源码的注释，其运行流程解释如下：

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
复制
首先 dataloader 基于 multiprocessing 产生多进程，每个子进程的输入输出通过两个主要的队列（multiprocessing.Queue() 类）产生，分别为：

· index_queue：每个子进程的队列中需要处理的任务的下标

· _worker_result_queue：返回时处理完任务的下标

· data_queue：表明经过 pin_memory 处理后的数据队列

并且有以下这些比较重要的 flag 参数来协调各个 worker 之间的工作：

· _send_idx: 发送索引，用来记录这次要放 index_queue 中 batch 的 idx

· _rcvd_idx: 接受索引，记录要从 data_queue 中取出的 batch 的 idx

· _task_info: 存储将要产生的 data 信息的 dict，key为 task idx（由 0 开始的整形索引），value 为 (worker_id,) 或 (worker_id, data)，分别对应数据未取和已取的情况

· _tasks_outstanding: 整形，代表已经准备好的 task/batch 的数量（可能有些正在准备中）


'''
class _IterableDatasetFetcher(_BaseDatasetFetcher): #· 对于 Iterable-style: __init__ 方法内设置了 dataset 初始的迭代器，fetch 方法内获取元素，此时 index 其实已经没有多大作用了。
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            # 即auto_collation为True，表示使用batch_sampler。
            # 则使用possibly_batched_index，获取1个batch大小的样本
            # 对于batch_sampler（即auto_collation==True）
            # 直接使用往后遍历并提取len(possibly_batched_index)个样本（即1个batch的样本）
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            # sampler则直接往后遍历，提取1个样本
            # 对于sampler，直接往后遍历并提取1个样本
            data = next(self.dataset_iter)
        return self.collate_fn(data)

'''
我们接下来看看如何获取样本。就是通过索引传入 fetcher，从而获取想要的样本。

fetcher生成如下，这是在_SingleProcessDataLoaderIter初始化时候生成的：

class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)

对于Map-style，就使用 _MapDatasetFetcher 处理，就是使用 possibly_batched_index 从数据集之中提取数据，possibly_batched_index 是key。

如果有batch sampler，就使用 batch sampler。

如果需要从一个小批次（ mini-batch）张量中合并出一个样本列表。就使用 collate_fn后处理。
'''
class _MapDatasetFetcher(_BaseDatasetFetcher):  #· 对于 Map-style：直接输入索引 index，作为 map 的 key，获得对应的样本（即 value）。
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            # 如果配置了batch_sampler，_auto_collation就为True，
            # 那么就优先使用batch_sampler，此时fetcher中传入的就是一个batch的索引
            # 有batch_sampler，_auto_collation就为True，
            # 就优先使用batch_sampler，对应在fetcher中传入的就是一个batch的索引
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
