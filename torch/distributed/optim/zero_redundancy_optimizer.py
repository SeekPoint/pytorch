# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import io
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import Optimizer
import logging

__all__ = ["ZeroRedundancyOptimizer"]


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(value: Any, non_blocking: bool, device: torch.device) -> Any:
    r"""
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.

    .. note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = [_recursive_copy_to_device(val, non_blocking=non_blocking, device=device) for val in value]
        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(val, non_blocking=non_blocking, device=device) for key, val in value.items()
        }

    return value


def _is_trainable(param: torch.Tensor) -> bool:
    return param.requires_grad


def _broadcast_object(
    obj: Any,
    src_rank: int,
    group: object = dist.group.WORLD,
    dist_device: torch.device = torch.device("cpu"),
) -> Any:
    r"""
    Either broadcast from master to the fleet (default),
    or use the src setting as the original rank.
    """

    if dist.get_rank() == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(dist_device)
        data_send_tensor = torch.ByteTensor(data).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0]).to(dist_device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=dist_device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=dist_device)
    return obj


def _get_global_rank(group: Any, rank: int) -> int:
    return rank if group is dist.group.WORLD else dist.distributed_c10d._get_global_rank(group, rank)


class ZeroRedundancyOptimizer(Optimizer):
    r"""
    This class wraps an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>`
    and shards its states across ranks in the group as described by
    ZeRO_. The optimizer instance in each rank is only responsible for
    updating ``1 / world_size`` parameters and hence only needs to keep
    ``1 / world_size`` optimizer states. After parameters are updated locally,
    each rank will broadcast its parameters to all other peers to keep all
    model replicas in the same state. ``ZeroRedundancyOptimizer`` can be used
    in conjunction with :class:`torch.nn.parallel.DistributedDataparallel` to
    reduce per-rank peak memory consumption.

    ``ZeroRedundancyOptimizer`` use a greedy algorithm to pack a number of
    parameters at each rank. Each parameter belongs to a single rank and is not
    divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_views (bool): when enabled, parameters will
            be packed into larger buckets to speed up communication and
            ``param.data`` fields will point to bucket views at different
            offsets. When disabled, each individual parameter will be
            communicated separately, but ``params.data`` will stay intact.
        **default: all trailing arguments will be forwarded to the given optimizer.

    Example::

        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP

        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning: ZeroRedundancyOptimizer is experimental and subject to change.

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    """
    '''
    0x04 初始化
    我们首先从 __init__ 看看如何构建，其主要做了三步：
    
        初始化基类。
        初始化各种成员变量。
        使用 _update_trainable 内部同步&构建buffer，其内部会调用 _optim_constructor 来构建内部优化器。
        
        因为 Python 语言的特点，没有专门的地方来初始化成员变量，而是在程序运行之中遇到了某个变量就即时初始化。
        所以，我们不会按照程序实际初始化的顺序来分析，而是按照成员变量逻辑上初始化的顺序来分析。
    '''
    def __init__(
        self,
        params,
        optimizer_class: Type[Optimizer], # 就是被包装的原生优化器类型
        group: Optional[Any] = None,
        parameters_as_bucket_view: bool = False,
        **default: Any,
    ):
        # Hold all the model params in the root .param_groups
        # NOTE: the default constructor uses `add_param_group` which is partially overloaded here
        # we introduce the `initialized` flag for be able to dissociate the behaviour of
        # `add_param_group` in between super() and ZeroRedundancyOptimizer
        self.initialized = False
        super().__init__(params, default) # 初始化基类

        # Partition information. lazy evaluation, computed if requested
        self._per_device_params_cache: "OrderedDict[torch.device, List[List[Parameter]]]" = (
            OrderedDict()
        )  # device, rank, params

        # Build the wrapped optimizer, responsible for a shard of the params
        self._param_rank_cache: Dict[torch.Tensor, int] = {} # 初始化各种成员变量
        self._param_to_index_cache: Dict[int, int] = {}
        self._partition_parameters_cache: List[List[Dict]] = []
        self._index_to_param_cache: Dict[int, torch.Tensor] = {}
        self._all_params = params
        self._reference_is_trainable_mask = list(map(_is_trainable, self._all_params))

        # Build the wrapped optimizer, responsible for a shard of the params
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        # global是用来在进程之间同步
        self.global_rank = _get_global_rank(self.group, self.rank)
        self.parameters_as_bucket_view = parameters_as_bucket_view

        self._optim_defaults = default
        self._optim_constructor = optimizer_class # 如何生成原生优化器

        #  Optional consolidated optimizer state
        self._all_states: List[Dict[str, Any]] = []

        # Current default device is set by the parameters allocated to this rank
        self._device = list(self._per_device_params.keys())[0]
        self.buckets: Dict[torch.device, List[torch.Tensor]] = {}

        self._update_trainable() # 内部同步&构建buffer，调用 _optim_constructor 来构建内部优化器
        self.initialized = True

    def _clear_cache(self) -> None:
        self._partition_parameters_cache.clear()
        self._per_device_params_cache.clear()
        self._param_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()

    def add_param_group(self, param_group: dict) -> None:
        r"""
        Add a param group to the :class:`Optimizer` s ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized
                along with group specific optimization options.

        .. warning: This method handles updating the shards on all partitions,
            but needs to be called on all ranks. Calling this on a subset of the
            ranks will cause the training to hang, because communication
            primitives are called depending on the managed parameters, and
            expect all the ranks to participate on the sane set of parameters.
        """

        super().add_param_group(param_group)
        if self.initialized:
            # Force a re-partitioning
            self._clear_cache()

            param_groups = self.partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            if self.parameters_as_bucket_view:
                self._setup_flat_buffers()

    def consolidate_state_dict(self, to: int = 0) -> None:
        r"""
        Update the consolidated state_dict list, one per rank.

        Arguments:
            to (int): the rank that receives the global states. (default: 0)

        .. warning: This needs to be called on all replicas
        """

        # Sync lr and other attributes in case its been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        empty_messenger = torch.tensor([0], dtype=torch.uint8, device=self._device)

        # Pull the sharded state from all the other replicas
        # Store all the states in order, rank by rank

        # NOTE: In practice, `broadcast` is used, which is wasteful (gather would have been appropriate)
        # compatibility issues with some backends make the use of broadcast mandatory for now.
        # a possible follow up would be to move all sharded state management to RPC RRef

        self._all_states = []
        for rank in range(self.world_size):
            global_rank = _get_global_rank(self.group, rank)

            # This rank collects the whole state
            if self.rank == to:
                if rank == self.rank:
                    self._all_states.append(
                        _recursive_copy_to_device(
                            self.local_state_dict(),
                            non_blocking=True,
                            device=torch.device("cpu"),
                        )
                    )
                else:
                    # Fetch the optim state from the other replicas
                    replica_state = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.group,
                        dist_device=self._device,
                    )

                    self._all_states.append(
                        _recursive_copy_to_device(replica_state, non_blocking=True, device=torch.device("cpu"))
                    )
            else:
                # Acknowledge broadcasts, and send this rank's shard when needed
                # Default to CPU space to gain some memory headroom
                if rank == self.rank:
                    # Send the state to the reference replica
                    _ = _broadcast_object(
                        self.local_state_dict(),
                        src_rank=self.global_rank,
                        group=self.group,
                        dist_device=self._device,
                    )

                elif rank != to:
                    # Discard this tensor/rank, broadcast was being use for compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.group,
                        dist_device=self._device,
                    )

    '''
    以下分析的这些函数或者说成员变量都是在__init__方法之中被间接调用或者初始化。
    
    4.1 将参数分区
    partition_parameters 方法会将参数进行分区，其返回 _partition_parameters_cache。
    
    被包装（wrapped）的optimizer根据参数大小（而不是使用顺序）以排序贪婪（sorted-greedy）算法来对优化器状态进行分片，
    在每个rank中打包一些参数，这样每个参数都属于一个rank，不在ranks之间划分。
    分区是任意的，可能与参数注册或使用顺序不匹配。这是为了确保每个rank具有几乎相同大小的优化器内存。
    '''
    def partition_parameters(self) -> List[List[Dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Returns:
            a list of ``param_groups`` (which is a list of dict) where each
            element of the list contains the param_groups for a rank. Element 0
            corresponds to rank 0, etc. We need all the ranks for the broadcast
            inside ``step()``.
        """
        if len(self._partition_parameters_cache) == 0:
            self._partition_parameters_cache = [list() for _ in range(self.world_size)]

            # 生成一个数组，用来记录每个rank的大小，一共有world size个rank
            sizes = [0] * self.world_size

            for param_group in self.param_groups:  # 遍历参数组
                param_lists: List[List] = [list() for _ in range(self.world_size)]

                for param in param_group["params"]:
                    # Add this param to rank with smallest size.
                    rank = sizes.index(min(sizes))  # 找到最小的那个rank
                    param_lists[rank].append(param)  # 把参数放到最小rank之中
                    sizes[rank] += param.numel()  # 增加rank的大小

                for rank, params in enumerate(param_lists): # 遍历list
                    param_group_rank = copy.copy(param_group)
                    param_group_rank["params"] = params
                    self._partition_parameters_cache[rank].append(param_group_rank)

        return self._partition_parameters_cache
    '''
    这里就分区好了，最终返回一个param_groups 的列表（这是一个dict列表），
    列表的每个元素都包含一个rank的param_groups，比如元素0对应于rank 0，每个rank的group的参数有差不多大小。
    在step()中，我们需要所有rank的信息来进行广播。下图给出了rank 0和 rank 5 对应的param_groups。

_partition_parameters_cache

          +
          |
          |
          v                +---------------+
  +-------+---------+      | param_group   |
  |       0         +----> |               |      <-------+  100 M   +------------->
  +-----------------+      +---------------+
  |       1         |      |               |     +--------+---------+------+--------+
  +-----------------+      |   "params" +------> |param 1 | param 2 | ...  | param 6|
  |       2         |      |               |     |        |         |      |        |
  +-----------------+      +---------------+     +--------+---------+------+--------+
  |                 |
  |                 |
  |     ......      |
  |                 |      +---------------+
  +-----------------+      | param_group   |      <-------+  105 M  +----------------->
  |       5         +----> |               |
  +-----------------+      +---------------+     +--------+---------+-------+---------+
                           |               |     |        |         |       |         |
                           |  "params"  +------> | param 7| param 8 | ...   | param 11|
                           |               |     |        |         |       |         |
                           +---------------+     +--------+---------+-------+---------+
    '''



    '''
    4.3 _per_device_params
    现在，参数已经分配给各个rank，接下来就要具体分配到设备之上，每个设备上可能包含多个rank的参数组。
    _per_device_params 方法就是把优化器的param_groups在各个设备之间进行分配，其返回_per_device_params_cache。
    
    请注意，_per_device_params 这里包括全部的模型参数，虽然已经按照设备进行了分类。
    即，在每个ZeRO优化器之中都是相同的。这样ZeRO优化器之间可以广播同步这些参数。
    '''
    @property
    def _per_device_params(self) -> Dict[torch.device, List[List[Parameter]]]:
        r"""
        Sorted list of all the params, first per device then per rank.

        Within a list params are sorted per number of elements to allow for an easy bucketing.
        """
        if len(self._per_device_params_cache) == 0:
            # Go through all params, log them per device
            # The ordering is important here, needs to be the same on all ranks
            # So that ulterior broadcast calls are matching
            for param_group in self.param_groups: # 遍历参数
                for param in param_group["params"]:
                    device = param.device # 找到其设备
                    if self._per_device_params_cache.get(device) is None:
                        self._per_device_params_cache[device] = [[] for _ in range(self.world_size)]
                    # 每个设备内部还需要按照rank来分开
                    self._per_device_params_cache[device][self._param_to_rank[param]] += [param]

            # Sort param_lists by size
            for k in self._per_device_params_cache.keys():
                for r in self._per_device_params_cache[k]:
                    r.sort(key=lambda x: x.numel())

        return self._per_device_params_cache
    '''
    比如，下面 CPU，GPU 1（忽略），GPU 2 都有自己的参数列表，每个列表之内都是按照参数大小排列。

_per_device_params_cache

      +
      |                                      +--------+--------+-------+--------+
      |                                      |        |        |       |        |
      |                     +---------+      | param1 | param3 |param5 | param6 |
      v                     |         |      |        |        |       |        |
 +----+--------------+      | rank 0  +----> |  1k    |  2k    |  3k   |   7k   |
 |                   |      |         |      +--------+--------+-------+--------+
 |     "CPU"         +----> +---------+
 |                   |      |         |
 +-------------------+      | rank 1  |      +--------+--------+-------+--------+
 |                   |      |         +----> |        |        |       |        |
 |     "GPU 1"       |      +---------+      | param9 | param2 | param4| param8 |
 |                   |                       |        |        |       |        |
 +-------------------+                       |  0.5k  |  1k    |  4k   |   8k   |
 |                   |                       +--------+--------+-------+--------+
 |     "GPU 2"       |      +---------+
 |                   +----> |         |      +---------+------------+-----------+
 +-------------------+      |         |      |         |            |           |
                            | rank 5  +----> | param 11|  param 13  | param 15  |
                            |         |      |         |            |           |
                            +---------+      +---------+------------+-----------+
                            |         |
                            | rank 6  |      +---------+------------+-----------+
                            |         +----> |         |            |           |
                            |         |      | param 19|  param 12  | param 14  |
                            +---------+      |         |            |           |
                                             +---------+------------+-----------+

    '''

    '''
    4.2 将参数分给rank
    现在，参数已经分成大小相近的group，接下来需要把这些group分到各个rank之上。
    
    _param_to_rank 方法生成一个表，里面记录每一个参数对应的rank，就是哪个参数在哪个rank之中。
    '''
    @property
    def _param_to_rank(self) -> Dict[torch.Tensor, int]:
        r"""Look up table to match a given param with a data parallel rank"""
        if len(self._param_rank_cache) == 0:
            for rank, param_groups in enumerate(self.partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_rank_cache[param] = rank
        return self._param_rank_cache
    '''
    依据上图例子，我们知道param 1，param 2，param 6 在rank 0之中，param 8，param 11 在 rank 5 之中.....，具体如下：
    
    _param_rank_cache
    
          +
          |
          |
          |
          v
     +----+--------------+------------+
     |                   |            |
     |   param 1         |     0      |
     +--------------------------------+
     |                   |            |
     |   param 2         |     0      |
     +--------------------------------+
     |                   |            |
     |   param 6         |     0      |
     +--------------------------------+
     |                   |            |
     |   param 8         |     5      |
     +--------------------------------+
     |                   |            |
     |   param 11        |     5      |
     +--------------------------------+
     |                   |            |
     |   param n         |     n      |
     |                   |            |
     +-------------------+------------+
    '''

    @property
    def _param_to_index(self) -> Dict[int, int]:
        r"""
        Hash table in between parameter indices in the global optimizer scheme,
        and the actual params.
        """
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {
                id(p): i for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))
            }

        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> Dict[int, torch.Tensor]:
        r"""
        Hash table in between parameter indices in the global optimizer scheme,
        and the actual params.
        """
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = {i: p for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))}

        return self._index_to_param_cache

    '''
    0x05 更新参数
    我们接下来看看优化器如何更新参数，其逻辑如下：
    
        如果计算图有变化，则需要重新处理。
        调用 _sync_param_groups 将本地优化器参数同步给 ZeRO优化器，防止其被 scheduler 已经修改。
        调用 self.optim.step，让本地优化器在本地参数之上进行更新。
        调用 dist.broadcast 在ranks 之间同步参数。
        再次调用 _sync_param_groups 将本地优化器参数同步给 ZeRO优化器，因为其已经被更新了。
    '''
    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs: Any) -> Optional[float]:
        r"""
        Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        Returns:
            optional loss, depends on the underlying optimizer

        .. note: Any extra parameter is passed to the base optimizer as-is
        """

        # Check whether the model trainability graph changed
        # 如果计算图有变化，则需要重新处理
        trainable_mask = list(map(_is_trainable, self._all_params))
        if trainable_mask != self._reference_is_trainable_mask:
            logging.warning(
                "ZeroRedundancyOptimizer detected that the trainable params changed, updating the partitioning"
            )
            self._update_trainable()
            self._reference_is_trainable_mask = trainable_mask

        # Sync oss param_groups attributes in case they've been updated by a scheduler.
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        '''
        5.1 更新
        首先是本地更新模型参数。
        
            # 更新本地参数
            if closure is not None:
                loss = self.optim.step(closure=closure, **kwargs)  # type: ignore[call-arg]
            else:
                loss = self.optim.step(**kwargs)
                
        假设模型一共有8个参数，分成上下两个节点，每个节点有一个优化器。
        这里为了更好说明，在上下两个优化器中，把参数和rank序号大的放在上面。
        
        再次强调：模型参数，梯度都没有变化，只是本地 ZeroRedundancyOptimizer 指向了部分需要优化的参数，
        所以 ZeroRedundancyOptimizer 的优化器状态也相应减少了。
        
        所以，上下两个优化器之中，模型（需要优化的参数）大小都一样，但是：
        
            ZeroRedundancyOptimizer 0 之中，优化的是 rank 0，参数 0 ~ 3 是本地优化的，
            对应两个节点来说，这部分参数是全局最新的。
            
            ZeroRedundancyOptimizer 1 之中，优化的是 rank 1，参数 4 ~ 7 是本地优化的，
            对应两个节点来说，这部分参数是全局最新的。
            
        +--------------------------------------------------------------------------------+
        |                                                     ZeroRedundancyOptimizer 0  |
        |                                                                                |
        |   _per_device_params_cache                                                     |
        |       +                                                                        |
        |       |                                                                        |
        |       v          +--------+           +--------+--------+-------+--------+     |
        |   +---+-----+    | rank 1 |           |        |        |       |        |     |
        |   |         |    |        +---------> | param4 | param5 | param6| param7 |     |
        |   | "GPU"1" +--> +--------+           |        |        |       |        |     |
        |   |         |    |        |           +--------+--------+-------+--------+     |
        |   +---------+    | rank 0 |                                                    |
        |                  |        |           +--------+--------+-------+--------+     |
        |                  |        +---------> |        |        |       |        |     |
        |                  +--------+           | param0 | param1 |param2 | param3 | NEW |
        |                               +---->  |        |        |       |        |     |
        |   +----------------+          |       +--------+--------+-------+--------+     |
        |   |Local Optimizer |          |                                                |
        |   |                +----------+                                                |
        |   |                |                                                           |
        |   +----------------+                                                           |
        |                                                                                |  Node 0
        +--------------------------------------------------------------------------------+
        
        
        
        +--------------------------------------------------------------------------------+
        |                                                                                |  Node 1
        |                                                                                |
        |   _per_device_params_cache                                                     |
        |       +                                                                        |
        |       |                               +--------+--------+-------+--------+     |
        |       v          +--------+     +---> |        |        |       |        |     |
        |   +---+-----+    | rank 1 |     |     | param4 | param5 | param6| param7 | NEW |
        |   |         |    |        +---------> |        |        |       |        |     |
        |   | "GPU"1" +--> +--------+     |     +--------+--------+-------+--------+     |
        |   |         |    |        |     |                                              |
        |   +---------+    | rank 0 |     |     +--------+--------+-------+--------+     |
        |                  |        +---------> |        |        |       |        |     |
        |                  |        |     |     | param0 | param1 |param2 | param3 |     |
        |                  +--------+     |     |        |        |       |        |     |
        |                                 |     +--------+--------+-------+--------+     |
        |   +----------------+            |                                              |
        |   |Local Optimizer |            |                                              |
        |   |                +------------+                                              |
        |   |                |                                                           |
        |   +----------------+                                 ZeroRedundancyOptimizer 1 |
        |                                                                                |
        +--------------------------------------------------------------------------------+
        
        
  
        '''
        # Run the optimizer step on this shard only:
        # 更新本地参数
        if closure is not None:
            loss = self.optim.step(closure=closure, **kwargs)  # type: ignore[call-arg]
        else:
            loss = self.optim.step(**kwargs)

        '''
         5.2 广播
        首先需要注意，_per_device_params 这里包括全部的模型参数，虽然已经按照设备进行了分类。
        
        现在状态是，本rank的优化器参数（本分区）已经更新了，就是模型的部分得到了更新。
        为了维持模型的最新，需要彼此进行广播。
            11.png
        
        在本地更新参数后，每个rank将向所有其他对等方广播其参数，以保持所有模型副本处于相同状态。
    
        +--------------------------------------------------------------------------------+
        |                                                     ZeroRedundancyOptimizer 0  |
        |                                                                                |
        |   _per_device_params_cache                                                     |
        |       +                                                                        |
        |       |                                                                        |
        |       v          +--------+           +--------+--------+-------+--------+     |
        |   +---+-----+    | rank 1 |           |        |        |       |        |     |
        |   |         |    |        +---------> | param4 | param5 | param6| param7 |     |
        |   | "GPU"1" +--> +--------+           |        |        |       |        |     |
        |   |         |    |        |           +--------+--------+-------+--------+     |
        |   +---------+    | rank 0 |                                                    |
        |                  |        |           +--------+--------+-------+--------+     |
        |                  |        +---------> |        |        |       |        |     |
        |                  +--------+           | param0 | param1 |param2 | param3 | NEW |
        |                               +---->  |        |        |       |        |     |
        |   +----------------+          |       +---+----+---+----+-+-----+--+-----+     |
        |   |Local Optimizer |          |           |        |      |        |           |
        |   |                +----------+           |        |      |        |           |
        |   |                |                      |  ^     |  ^   |  ^     |   ^       |
        |   +----------------+                      |  |     |  |   |  |     |   |       |
        |                                           |  |     |  |   |  |     |   |       | Node 0
        +--------------------------------------------------------------------------------+
                                                    |  |     |  |   |  |     |   |
                                                    |  |     |  |   |  |     |   |
                                                    |  |     |  |   |  |     |   |
        +--------------------------------------------------------------------------------+
        |                                           |  |     |  |   |  |     |   |       | Node 1
        |                                           v  |     v  |   v  |     v   |       |
        |   _per_device_params_cache                   |        |      |         |       |
        |       +                                      |        |      |         |       |
        |       |                               +------+-+------+-+----+--+------+-+     |
        |       v          +--------+     +---> |        |        |       |        |     |
        |   +---+-----+    | rank 1 |     |     | param4 | param5 | param6| param7 | NEW |
        |   |         |    |        +---------> |        |        |       |        |     |
        |   | "GPU"1" +--> +--------+     |     +--------+--------+-------+--------+     |
        |   |         |    |        |     |                                              |
        |   +---------+    | rank 0 |     |     +--------+--------+-------+--------+     |
        |                  |        +---------> |        |        |       |        |     |
        |                  |        |     |     | param0 | param1 |param2 | param3 |     |
        |                  +--------+     |     |        |        |       |        |     |
        |                                 |     +--------+--------+-------+--------+     |
        |   +----------------+            |                                              |
        |   |Local Optimizer |            |                                              |
        |   |                +------------+                                              |
        |   |                |                                                           |
        |   +----------------+                                 ZeroRedundancyOptimizer 1 |
        |                                                                                |
        +--------------------------------------------------------------------------------+     
        '''
        # Sync all the updated shards in between the ranks
        handles = []
        if self.parameters_as_bucket_view:
            for device in self.buckets.keys():
                for src_rank, bucket in enumerate(self.buckets[device]):
                    global_src_rank = _get_global_rank(self.group, src_rank)
                    handles.append(dist.broadcast(tensor=bucket, src=global_src_rank, group=self.group, async_op=True))
        else:
            for device, per_rank_params in self._per_device_params.items(): # 遍历设备+其参数
                for dst_rank, params in enumerate(per_rank_params): # 遍历rank
                    global_dst_rank = _get_global_rank(self.group, dst_rank)
                    for param in params: # 对于每一个参数，都进行broadcast
                        handles.append(
                            dist.broadcast(tensor=param.data, src=global_dst_rank, group=self.group, async_op=True)
                        )

        _ = list(map(lambda x: x.wait(), handles))

        '''
        5.3 同步本地参数
        最后，需要再次调用 _sync_param_groups 将本地优化器参数同步给 ZeRO优化器，因为其已经被更新了。
        '''
        # Sync hypothethical new results from the wrapped optimizer to the exposed param_groups
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Restore the global parameter groups as well as the shard.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`
        """

        for key, value in state_dict["state"].items():
            param = self._index_to_param[key]

            # Populate the sharded optimizer state on the fly
            if self._param_to_rank[param] != self.rank:
                state_dict["state"][key] = None
            else:
                self.optim.state[param] = _recursive_copy_to_device(value, non_blocking=True, device=param.device)

        super().load_state_dict(state_dict)

        # Sync with the optimizer param groups
        ZeroRedundancyOptimizer._sync_param_groups(state_dict["param_groups"], self.param_groups)
        ZeroRedundancyOptimizer._sync_param_groups(self.param_groups, self.optim.param_groups)

    def local_state_dict(self) -> Dict:
        r"""
        Gets this rank's ``state_dict``.

        Returns:
            The state of the optimizer as a :class:`dict`.
            It contains two entries:

            * state - a dict holding current optimization state. Its content
                differs between optimizer classes.
            * param_groups - a dict containing all parameter groups
        """
        return self.optim.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Returns:
            the last known global optimizer state, which consist of a list of
            the shards.

        .. warning:
            If the state has not been consolidated, this returns a shard's worth,
            not the global state.

        .. warning:
            Returning the global state is limited to the replica which was
            responsible for the consolidation. The state may also not be up to
            date, depending on when :meth:`consolidate_state_dict` was last called.
        """

        if len(self._all_states) == 0:
            raise RuntimeError(
                "Optimizer state has not been consolidated on this rank. \
                Please call `consolidate_state_dict()` on all ranks beforehand if you meant to save the global state"
            )

        # Unify the shard states and the state that pytorch would expect, given the model.
        # Indexation needs several redirections, since each shard only knows a limited scope of the model
        # - get the pytorch compliant parameter indexing
        state_dict = super().state_dict()

        # - go through the per-shard states, which are all indexed locally
        for rank, s in enumerate(self._all_states):
            # -- match the local indexing and the global partition, update the corresponding saved state globally
            for local_pg, global_pg in zip(s["param_groups"], self.partition_parameters()[rank]):
                local_index_to_param_id = {
                    i_param: id(global_pg["params"][i]) for i, i_param in enumerate(local_pg["params"])
                }

                for local_param_index in local_pg["params"]:
                    # Update the state, if any
                    if local_param_index in s["state"].keys():
                        global_id = self._param_to_index[local_index_to_param_id[local_param_index]]
                        state_dict["state"][global_id] = s["state"][local_param_index]

        # Make sure that the parameters are sorted in the state, as expected
        state_dict["state"] = dict(sorted(state_dict["state"].items()))
        return state_dict

    @staticmethod
    def rank_local_state_dict(rank: int, state_dict: dict) -> dict:
        r"""
        Returns the local_state_dict for a given rank.

        Arguments:
            rank (int): rank to get ``local_state_dict`` for
            state_dict (dict): global ``state_dict``
        """
        param_groups = state_dict["param_groups"][state_dict["partition"][rank][0] : state_dict["partition"][rank][1]]
        return {"state": state_dict["state"][rank], "param_groups": param_groups}

    '''
    4.4.1 同步参数组
    _sync_param_groups 用来把内部优化器的参数组同步到本Zero优化器的参数组。
    '''
    @staticmethod
    def _sync_param_groups(source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]) -> None:
        r"""Sync learning rate and other optimizer attributes (needed to support schedulers)."""

        for source_group, destination_group in zip(source, destination):
            # Sync everything but the parameters
            for k in filter(lambda x: x != "params", source_group.keys()):
                destination_group[k] = source_group[k]

    '''
    4.4.2 建立single buffer
    如果设置了parameters_as_bucket_view，则调用_setup_flat_buffers 建立若干buffer。
    同样设备上同样rank的张量被视为一个buffer。就是处理 _per_device_params。
    '''
    def _setup_flat_buffers(self) -> None:
        r"""
        Make all params which are on the same device and tied to the same rank
        views of a single buffer. This is used at construction time, and anytime
        parameter trainability is changed (frozen or unfrozen) and
        ``_update_trainable`` is called.
        """

        for device, per_rank_params in self._per_device_params.items():
            # Only wipe the existing buckets if there are none
            # (could be that this is called twice, when trainability changes)
            if device not in self.buckets.keys():
                self.buckets[device] = []

            # Make parameters a view of the bucket
            for dst_rank, params in enumerate(per_rank_params):
                if len(params) > 0:

                    # Clone the non-trainable params, if in a bucket it will get destroyed
                    for param in filter(lambda x: not x.requires_grad, params):
                        param.data = param.data.detach().clone()

                    # Merge all the trainable params in a single bucket
                    trainable_params = list(filter(_is_trainable, params))
                    buffer_size = sum(map(lambda x: x.numel(), trainable_params))
                    bucket = torch.empty(buffer_size, dtype=params[0].dtype, device=device)
                    offset = 0

                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next

                    # Either replace the existing bucket, or create it
                    if len(self.buckets[device]) == dst_rank:
                        self.buckets[device].append(bucket)
                    else:
                        self.buckets[device][dst_rank] = bucket
                else:
                    self.buckets[device].append(torch.zeros(1, device=device))
    '''
    具体可以看看如下图例，同样设备上同样rank的张量被视为一个buffer。

    buckets
         +
         |
         |               +---------------------------------------+
         v               | Tensor                                |
    +----+-------+       | +-----------------------------------+ |
    |            |       | |                                   | |
    |  "CPU"     +-----> | | Param 1, param 2,  Param 3......  | |
    |            |       | +-----------------------------------+ |
    +------------+       +---------------------------------------+
    |            |
    |  "GPU 1"   +-----> +---------------------------------------+
    |            |       | Tensor                                |
    +------------+       | +-----------------------------------+ |
    |            |       | |                                   | |
    |            |       | | Param 6, Param 7,  Param 8......  | |
    |            |       | +-----------------------------------+ |
    |            |       +---------------------------------------+
    |            |
    +------------+
    '''

    '''
    4.4 _update_trainable
    因为某些参数会变化，所以需要在本地优化器和ZeroRedundancyOptimizer 之间彼此同步。
    
        首先得到 self._default_device 为 "CPU" 或者 "GPU #"。
        
        然后调用 _optim_constructor 来构建内部优化器。
        注意，这里就是告诉本地优化器，你就负责优化这些参数即可，不用管其他的shard。
        partition_parameters 方法前面提到，其会将参数进行分区，其返回 _partition_parameters_cache。
                
            # 只是选取自己rank对应的参数进行优化
            self.optim = self._optim_constructor(self.partition_parameters()[self.rank], **self._optim_defaults)
            
            # 运行时变量如下：
            #_optim_constructor = {type} <class 'torch.optim.adam.Adam'>
            #_optim_defaults = {dict: 1} {'lr': 0.01}
        接着，调用 _sync_param_groups 同步参数。
        
        最后，建立 flat buffer。
    
    具体代码如下：
    '''
    def _update_trainable(self) -> None:
        r"""
        Updates the partitioning and communication patterns if the trainability
        (``requires_grad``) of some parameters changed.
        """

        # Create the optim which will work on the param shard
        if not hasattr(self, "optim"):
            self._clear_cache()

            # 获得缺省设备
            self._default_device = list(self._per_device_params.keys())[0]

            # 构建本地优化器，只是选取本rank对应的参数
            self.optim = self._optim_constructor(self.partition_parameters()[self.rank], **self._optim_defaults)

            # 调用 _sync_param_groups 同步参数，self.optim 是被包装的优化器
            self._sync_param_groups(self.optim.param_groups, self.param_groups)

        if self.parameters_as_bucket_view:
            self._setup_flat_buffers() # 建立 flat buffer

    '''

我们用 rank 5 为例，其本地优化器就只是指向 _partition_parameters_cache[5] 对应的那部分待优化参数，
本地优化器只优化这些参数即可。

这样就实现了优化器参数分区。
_partition_parameters_cache[5] 这样的参数可以在后续被放置到 GPU 之上，这样每个GPU就只包括 优化器的部分分区。

需要注意的是：模型参数，梯度都没有变化，只是本地 ZeroRedundancyOptimizer 指向了部分需要优化的参数，
所以 ZeroRedundancyOptimizer 的优化器状态也相应减少了。

就下图来说，原先优化器需要优化全部的参数，可能有 100 M + 105 M + ....，现在ZeroRedundancyOptimizer只需要优化 105 M。

 _partition_parameters_cache

        +
        |
        |
        v                +---------------+
+-------+---------+      | param_group   |
|       0         +----> |               |      <-------+  100 M   +------------->
+-----------------+      +---------------+
|       1         |      |               |     +--------+---------+------+--------+
+-----------------+      |   "params" +------> |param 1 | param 2 | ...  | param 6|
|       2         |      |               |     |        |         |      |        |
+-----------------+      +---------------+     +--------+---------+------+--------+
|                 |
|                 |
|     ......      |
|                 |      +---------------+
+-----------------+      | param_group   |      <-------+  105 M  +----------------->
|       5         +----> |               |
+-----------------+      +---------------+     +--------+---------+-------+---------+
                         |               |     |        |         |       |         |
                    +--> |  "params"  +------> | param 7| param 8 | ...   | param 11|
                    |    |               |     |        |         |       |         |
                    |    +---------------+     +--------+---------+-------+---------+
                    |
                    |
                    |
+-----------------------+
| Local Optimizer   |   |
|                   |   |
|                   |   |
|                   +   |
|                       |
|                       |
|                       |
|                       |
+-----------------------+
我们还需要再细化一下，看看 _sync_param_groups 和 _setup_flat_buffers 这两个函数。
    '''
