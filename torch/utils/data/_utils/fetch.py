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


'''
class _IterableDatasetFetcher(_BaseDatasetFetcher):
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
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            # 如果配置了batch_sampler，_auto_collation就为True，
            # 那么就优先使用batch_sampler，此时fetcher中传入的就是一个batch的索引
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
