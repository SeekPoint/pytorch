import bisect
import warnings
import math
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

# No 'default_generator' in torch/__init__.pyi
from torch import default_generator, randperm
from torch._utils import _accumulate

from ... import Generator, Tensor

__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "random_split",
]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

'''
1. 迭代器介绍

OK，在正式解析 PyTorch 中的 torch.utils.data 模块之前，我们需要理解一下 Python 中的迭代器（Iterator），因为在源码的 Dataset, Sampler 和 DataLoader 这三个类中都会用到包括 __len__(self)，__getitem__(self) 和 __iter__(self) 的抽象类的魔法方法。

· __len__(self)：定义当被 len() 函数调用时的行为，一般返回迭代器中元素的个数。

· __getitem__(self)：定义获取容器中指定元素时的行为，相当于 self[key] ，即允许类对象拥有索引操作。

· __iter__(self)：定义当迭代容器中的元素时的行为。

除此之外，我们也需要清楚两个概念：

· 迭代（Iteration）:当我们用一个循环（比如 for 循环）来遍历容器（比如列表，元组）中的元素时，这种遍历的过程可称为迭代。

· 可迭代对象（Iterable）：一般指含有 __iter__() 方法或 __getitem__() 方法的对象。我们通常接触的数据结构，如序列（列表、元组和字符串）还有字典等，都支持迭代操作,也可称为可迭代对象。

那什么是迭代器（Iterator）呢？简而言之，迭代器就是一种可以被遍历的容器类对象，但它又比较特别，它需要遵循迭代器协议，那什么又是迭代器协议呢？迭代器协议（iterator protocol）是指要实现对象的 __iter()__ 和 __next__() 方法。一个容器或者类如果是迭代器，那么就必须实现 __iter__() 方法以及重点实现 __next__() 方法，前者会返回一个迭代器（通常是迭代器对象本身），而后者决定了迭代的规则。现在，为更好地理解迭代器的内部运行机制，我们可以看一个斐波那契数列的迭代器实现例子：

一般而言，迭代器满足以下几种特性：

· 迭代器是⼀个对象，但比较特别，需要满足迭代器协议，他还可以被 for 语句循环迭代直到终⽌。

· 迭代器可以被 next() 函数调⽤，并返回⼀个值，亦可以被 iter() 函数调⽤，但返回的是一个迭代器（可以是自身）。

· 迭代器连续被 next() 函数调⽤时，依次返回⼀系列的值，但如果到了迭代的末尾，则抛出 StopIteration 异常，另外他可以没有末尾，但只要被 next() 函数调⽤，就⼀定会返回⼀个值。

· Python3 中， next() 内置函数调⽤的是对象的 __next__() ⽅法，iter() 内置函数调⽤的是对象的 __iter__() ⽅法。

那么，了解了什么是迭代器后，我们马上开始解析 torch.utils.data 模块，对于 torch.utils.data 而言，重点是其 Dataset，Sampler，DataLoader 三个模块，辅以 collate，fetch，pin_memory 等组件对特定功能予以支持。

Tips：涉及的源码皆以 PyTorch 1.7 为准
'''

'''
Dataset 主要负责对 raw data source 封装，将其封装成 Python 可识别的数据结构，其必须提供提取数据个体的接口。Dataset 中共有 Map-style datasets 和 Iterable-style datasets 两种：

1.1 Map-style dataset
torch.utils.data.Dataset 它是一种通过实现  __len__() 和 __getitem__() 方法来获取数据的 Dataset，它表示从（可能是非整数）索引/关键字到数据样本的映射。
因而，在我们访问 Map-style 的数据集时，使用 dataset[idx] 即可访问 idx 对应的数据。通常，我们使用 Map-style 类型的 dataset 居多，可以看到其数据接口定义如下：


1.3 其他 Dataset

除了 Map-style dataset 和 Iterable-style dataset 以外，PyTorch 也在此基础上提供了其他类型的 Dataset 子类：

· torch.utils.data.ConcatDataset：用于连接多个 ConcatDataset 数据集。

· torch.utils.data.ChainDataset：用于连接多个 IterableDataset 数据集，在 IterableDataset 的 __add__() 方法中被调用。

· torch.utils.data.Subset：用于获取指定一个索引序列对应的子数据集。

'''
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py

'''
1.2 Iterable-style dataset

torch.utils.data.IterableDataset 它是一种实现 __iter__() 来获取数据的 Dataset，
Iterable-style 的数据集特别适用于以下情况：随机读取代价很大甚至不可能，且 batch size 取决于获取到的数据。其接口定义如下：

特别地，当 DataLoader 的 num_workers > 0 时， 每个 worker 都将具有数据对象的不同样本。因此需要独立地对每个副本进行配置，以防止每个 worker 产生的数据不重复。
同时，数据加载顺序完全由用户定义的可迭代样式控制。这允许更容易地实现块读取和动态批次大小（例如，通过每次产生一个批次的样本）。

'''
class IterableDataset(Dataset[T_co]):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

    Example 1: splitting workload across all workers in :meth:`__iter__`::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
        >>> # xdoctest: +SKIP("Fails on MacOS12")
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [tensor([3]), tensor([4]), tensor([5]), tensor([6])]

        >>> # xdoctest: +REQUIRES(POSIX)
        >>> # Mult-process loading with two worker processes
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> # xdoctest: +IGNORE_WANT("non deterministic")
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

        >>> # With even more workers
        >>> # xdoctest: +IGNORE_WANT("non deterministic")
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
        [tensor([3]), tensor([5]), tensor([4]), tensor([6])]

    Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]
        >>>
        >>> # Directly doing multi-process loading yields duplicate data
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 3, 4, 4, 5, 5, 6, 6]

        >>> # Define a `worker_init_fn` that configures each dataset copy differently
        >>> def worker_init_fn(worker_id):
        ...     worker_info = torch.utils.data.get_worker_info()
        ...     dataset = worker_info.dataset  # the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # configure the dataset to only process the split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        ...     worker_id = worker_info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...

        >>> # Mult-process loading with the custom `worker_init_fn`
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
        [3, 4, 5, 6]
    """
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])

    # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

#· torch.utils.data.TensorDataset：用于获取封装成 tensor 的数据集，每一个样本都可通过索引张量来获得。
class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total

#· torch.utils.data.Subset：用于获取指定一个索引序列对应的子数据集。
class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset: Dataset[T], lengths: Sequence[Union[int, float]],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
