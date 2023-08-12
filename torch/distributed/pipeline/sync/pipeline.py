# -*- coding: utf-8 -*-
# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""The pipeline parallelism of Pipe."""
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast, Sequence

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function

from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .worker import Task, create_workers

__all__: List[str] = ["Pipeline"]


Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional["Task"]]
    OutQueue = Queue[Tuple[bool, Union[Tuple["Task", Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue

'''
我们结合示例代码把传入的参数赋值一下，重新把方法解释如下，这样大家就可以更好的理解。

def depend(batches[i-1]: Batch, batches[i]: Batch) -> None:
    batches[i-1][0], phony = fork(batches[i-1][0])
    batches[i][0] = join(batches[i][0], phony)
具体逻辑如下，通过 phony 完成了一个桥接，即在正向传播之中，batches[i] 依赖 batches[i-1] 的执行结果。

      +----------------+          +--------------+
      |                |          |              |
      |  batches[i-1]  |          |  batches[i]  |
      |                |          |              |
      +----------+-----+          +-----+--------+
                 |                      |
                 |                      |
                 |                      |
                 v                      v
+--------------------------------------------------------+
| depend         |                      |                |
|                |                      |                |
|                |                      |                |
|                v                      |                |
|        +-----------------------+      |                |
|        | fork  |               |      |                |
|        |       |    get_phony  |      |                |
|        |       |        +      |      |                |
|        |       |        |      |      |                |
|        |       |        |      |      |                |
|        +-----------------------+      |                |
|                |        |             |                |
|                |        |             |                |
|                |        |             |                |
|                v        v             |                |
|    +-----------+--+  +--+-----+       |                |
|    |              |  |        |       |                |
|    | batches[i-1] |  | phony  |       |                |
|    |              |  |        |       |                |
|    +--------------+  +--+-----+       |                |
|                         |             |                |
|                         |             |                |
|                         v             v                |
|                      +--+------------------+           |
|                      |Join            |    |           |
|                      |                |    |           |
|                      |                |    |           |
|                      |                v    |           |
|                      +---------------------+           |
|                                       |                |
|                                       |                |
|                                       |                |
|                                       v                |
|                                 +-----+------+         |
|                                 |            |         |
|                                 | batches[i] |         |
|                                 |            |         |
|                                 +------------+         |
|                                                        |
+--------------------------------------------------------+
我们把多个 batches 联合起来看看，这样就能看出来一个依赖链条。

                  +----------------------------------------------------------+
                  | depend                                                   |
                  |                                                          |
                  | +------------+                                           |
 +-------------   | |fork        |     +-----------+                         |
 |            |   | |            |     |           |                         |
 |batches[i]  +----------------------> | batches[i]|                         |
 |            |   | |            |     |           |                         |
 +-------------   | |            |     +-----------+                         |
                  | |            |             +-------+                     |
                  | |            +-----------> | Join  |                     |
                  | |            |             |       |                     |
                  | +------------+             |       |                     |
 +-------------   |                            |       |    +--------------+ |
 |            |   |                            |       |    |              | |
 |batches[i+1]+-------------------------------------------->+ batches[i+1] | |
 |            |   |                            |       |    |              | |
 +---------+---   |                            |       |    +--------------+ |
           |      |                            +-------+                     |
           |      |                                                          |
           |      +----------------------------------------------------------+
           |      +----------------------------------------------------------+
           |      | depend                                                   |
           |      |                                                          |
           |      | +-------------+                                          |
           |      | |fork         |     +------------+                       |
           |      | |             |     |            |                       |
           +--------------------------> |batches[i+1]|                       |
                  | |             |     |            |                       |
                  | |             |     +------------+                       |
                  | |             |           +-------+                      |
                  | |             +---------> |Join   |                      |
                  | +-------------+           |       |                      |
+------------+    |                           |       |     +-------------+  |
|            |    |                           |       |     |             |  |
|batches[i+2]+--------------------------------------------> | batches[i+2]|  |
|            |    |                           |       |     |             |  |
+----------+-+    |                           |       |     +-------------+  |
           |      |                           +-------+                      |
           |      |                                                          |
           |      +----------------------------------------------------------+
           |
           |      +-----------------------------------------------------------+
           |      | depend                                                    |
           |      |                                                           |
           +----------------------------->    ......                          |
                  |                                                           |
                  |                                                           |
                  +-----------------------------------------------------------+
这样，上图就是前向计算图，于是在后向传播之中，batches[i] 就 必须在 batches[i-1] 之前完成了。

我们再结合论文的图来看看。

本来示例代码中是：

depend(batches[i-1], batches[i])
为了和论文中的图对应，我们修改为：

depend(batches[i], batches[i+1])
depend 代码也变化为：

def depend(batches[i]: Batch, batches[i+1]: Batch) -> None:
    batches[i][0], phony = fork(batches[i][0])
    batches[i+1][0] = join(batches[i+1][0], phony)
对应下图，就是在后向传播计算图之中 batches[i+1] 通过一个join, 一个fork，排在了 batches[i] 前面，就是下面大箭头所示，具体细化一下：

从这个图上，PyTorch 的 autograd 引擎不知道 Bi+1,j
 必须在 Bi,j
 之前运行，因此会打乱后向传播的时间流。因此，虚拟依赖（前面图的虚线箭头）必须在前向传播中被显式绘制出来。

图上的实线箭头依据后向传播图的方向来绘制，这些联系是在前向传播中被构建的。就是说，对于 Batchi
 来说，其反向传播顺序是固定的。就是上面一行内顺序是固定的，下面一行内顺序也是固定的。

但是，上下两行之间的顺序是不可知的，需要用虚线来保证，就是用 Join & Fork 来保证。



有个img!!!!!

'''

def _depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from_idx = fork_from.find_tensor_idx()
    join_to_idx = join_to.find_tensor_idx()

    fork_from[fork_from_idx], phony = fork(fork_from[fork_from_idx])
    join_to[join_to_idx] = join(join_to[join_to_idx], phony)

# 以下函数对算子进行了封装。
def _copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in batch])

# 以下函数对算子进行了封装。
def _wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in batch])

'''
我们再来看看代码。首先是生成时钟周期，这里：
    min(1+k, n) 就是在 k 时钟时候，可以启动的最大device数目（partition）。
    max(1+k-m, 0) 就是在 k 时钟时候，可以启动的最小微batch（micro-batch）。
所以最终返回的序列就是k 时钟时候，可以启动的（index of micro-batch，index of partition）序列。
'''
def _clock_cycles(m: int, n: int) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    # 我们解析一下，这里 k 就是时钟数，从1开始，最多时钟序号就是 m+n-1。
    # min(1+k, n) 就是在 k 时钟时候，可以启动的最大device数目
    # max(1+k-m, 0) 就是在 k 时钟时候，可以启动的最小batch
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


'''
设定 m = 4, n =3，solve(4,3) 的输出是：

[(0, 0)]
[(1, 0), (0, 1)]
[(2, 0), (1, 1), (0, 2)]
[(3, 0), (2, 1), (1, 2)]
[(3, 1), (2, 2)]
[(3, 2)]
因为论文有一个示例图，而这个图和注释&代码不完全一致，为了更好的说明，我们就按照图上来，因为图片是从 F1,1
开始，所以我们把注释修正以下:

# 0 (0,0)                   ----> clock 1 运行图上的 (1,1)
# 1 (1,0) (0,1)             ----> clock 2 运行图上的 (2,1) (1,2)
# 2 (2,0) (1,1) (0,2)       ----> clock 3 运行图上的 (3,1) (2,2) (1,3)
# 3       (2,1) (1,2)       ----> clock 4 运行图上的 (3,2) (2,3)
# 4             (2,2)       ----> clock 5 运行图上的 (3,3)
我们把 solve代码修改下，为了打印正确的index，这样大家就可以更好的把代码和图片对应起来了。

m=4 # m: number of micro-batches
n=3 # n: number of partitions
for k in range(m + n - 1):
    print( [(k - j + 1 , j +1 ) for j in range(max(1 + k - m, 0), min(1 + k, n))] )

打印是：
[(1, 1)]  # 第 1 轮训练计划 & 数据
[(2, 1), (1, 2)] # 第 2 轮训练计划 & 数据
[(3, 1), (2, 2), (1, 3)] # 第 3 轮训练计划 & 数据
[(4, 1), (3, 2), (2, 3)] # 第 4 轮训练计划 & 数据
[(4, 2), (3, 3)] # 第 5 轮训练计划 & 数据
[(4, 3)] # 第 6 训练计划 & 数据
我们把流水线的图再祭出来看看。

在这里插入图片描述

我们把上面的输出按照流水线的图绘制一下作为比对。

可以看到，前 4 个时钟周期内，分别有 4 个 micro-batch 进入了 cuda:0，分别是(1,1) (2,1) (3,1) (4,1) 。然后按照 clock_cycles 算法给出的顺序，每次迭代（时钟周期）内执行不同的schedule，经过了 6 个时钟周期之后，完成了第一轮 forward 操作。这就形成了流水线。

流水线优势在于，如果 number of micro-batches 配置的合适，那么可以在每个时钟周期内，最大程度的让所有设备都运行起来。与之对比，原生流水线每一时间只能让一个设备互活跃。

           +          +          +          +          +          +          +
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
 cuda:0    |  (1,1)   |   (2,1)  |  (3,1)   |   (4,1)  |          |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
 cuda:1    |          |   (1,2)  |  (2,2)   |   (3,2)  |  (4,2)   |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
 cuda:2    |          |          |  (1,3)   |   (2,3)  |  (3,3)   |  (4,3)   |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
           |          |          |          |          |          |          |
           | clock 1  |  clock 2 |  clock 3 |  clock 4 |  clock 5 |  clock 6 |
           +          +          +          +          +          +          +

+------------------------------------------------------------------------------>  Time

具体数据batch的走向是：

         +             +            +             +            +            +             +
         |             |            |             |            |            |             |
 cuda:0  |    (1,1)    |   (2,1)    |   (3,1)     |   (4,1)    |            |             |
         |      +      |     +      |       +     |       +    |            |             |
         |      |      |     |      |       |     |       |    |            |             |
         |      |      |     |      |       |     |       +----------+      |             |
         |      |      |     |      |       +-----------+      |     |      |             |
         |      |      |     +------------+       |     |      |     |      |             |
         |      |      |            |     |       |     |      |     |      |             |
         |      +------------+      |     |       |     |      |     |      |             |
         |             |     |      |     |       |     |      |     |      |             |
         |             |     |      |     v       |     v      |     v      |             |
         |             |     v      |             |            |            |             |
 cuda:1  |             |   (1,2)    |   (2,2)     |   (3,2)    |  (4,2)     |             |
         |             |     +      |     +       |      +     |      +     |             |
         |             |     |      |     |       |      |     |      |     |             |
         |             |     |      |     |       |      |     |      +-------------+     |
         |             |     |      |     |       |      +----------+       |       |     |
         |             |     |      |     +------------+       |    |       |       |     |
         |             |     +-----------+        |    |       |    |       |       |     |
         |             |            |    |        |    v       |    v       |       v     |
         |             |            |    v        |            |            |             |
 cuda:2  |             |            |   (1,3)     |   (2,3)    |  (3,3)     |     (4,3)   |
         |             |            |             |            |            |             |
         |             |            |             |            |            |             |
         |             |            |             |            |            |             |
         |   clock 1   |  clock 2   |   clock 3   |  clock 4   |  clock 5   |     clock 6 |
         +             +            +             +            +            +             +

+----------------------------------------------------------------------------------->  Time


'''

class Pipeline:
    """The pipeline parallelism for Pipe."""

    def __init__(
        self,
        partitions: List[nn.Sequential],
        devices: List[torch.device],
        copy_streams: List[List[AbstractStream]],
        skip_layout: SkipLayout,
        checkpoint_stop: int,
    ) -> None:
        self.partitions = partitions
        self.devices = devices
        self.copy_streams = copy_streams
        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop
        (self.in_queues, self.out_queues) = create_workers(devices)

    #在 Pipeline 类之中，我们可以看到，就是按照时钟周期来启动计算，这样在前向传播之中，就按照这个序列，像水波纹一样扩散。
    def run(self, batches: List[Batch]) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        partitions = self.partitions
        devices = self.devices
        skip_layout = self.skip_layout

        m = len(batches)
        n = len(partitions)

        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]

        for schedule in _clock_cycles(m, n): # 这里使用，给出了执行序列计划，后续按照这个来执行
            self.fence(batches, schedule, skip_trackers) # 构建后向传播依赖关系
            self.compute(batches, schedule, skip_trackers) # 进行计算

    #在 Pipeline 之中我们可以看到具体的使用方法，fence 方法（省略部分代码）利用 depend 来构建后向传播的依赖关系，确保 batches[i-1] 在 batches[i] 之后完成。
    #建立了图例之中的行，列 两种依赖关系。
    def fence(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0 and j != 0:
                _depend(batches[i - 1], batches[i]) # 在这里建立了后向传播依赖关系

            next_stream = copy_streams[j][i] # 拿到dst设备的拷贝流

            for prev_j, ns, name in skip_layout.copy_policy(j): # 残差连接相关设置
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            # 建立跨设备依赖关系，指定了 device[j-1] 的输出是 device[i] 的输入
            if j != 0:
                prev_stream = copy_streams[j - 1][i]  # 拿到src设备的拷贝流
                _copy(batches[i], prev_stream, next_stream) # 建立跨设备依赖关系

    '''
    batches[i] 这里是会变化的，比如 batches[0] 在经过 partitions[j] 的计算之后，会变成 batches[0][j]。
    对于 compute 方法，关键就是在最底部的代码 batches[i] = batch。
    就是把 第 j 个device 对 第 i 个 batch 的计算结果 赋值到 batches[i]，赋值之后，batches[i]就是 batches[i][j]，
    这样，在下次计算时候，构建的就是 F[i, j+1], 下一次 fence 之中的 depend 操作，就是针对 batches[i, j+1]。
    因此，在前向计算图上，通过这个赋值操作， batches[i, j+1] 就依赖 batches[i, j]，所以反向计算时候，batches[i, j + 1] 就必须在 batches[i, j] 之前完成。
    
    具体wait操作则是在 compute 之中调用
    '''
    def compute(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Runs tasks with synchronization to copy streams."""
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        # Disable checkpointing if in eval mode.
        if not self.partitions[0].training:
            checkpoint_stop = 0

        n = len(partitions)
        streams = [current_stream(d) for d in devices]
        exc_info: Optional[ExcInfo] = None

        # With checkpointing, the autograd graph looks like this diagram:
        # ┌─────┸──────┐
        # │    Copy    │
        # └─────┰──────┘   (fence)
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #       ┃          (compute)
        # ┌─────┸──────┐
        # │    Wait    │ [1] Synchronize the current stream with the copy stream.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │ Checkpoint │ [2] Compute a partition within checkpointing.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │    Wait    │ [3] Synchronize the copy stream with the current stream.
        # └─────┰──────┘
        #       ┠ ─ ─ ─ ┐
        #       ┃ ┌─────┴─────┐
        #       ┃ │ Recompute │ [4] Schedule the recomputation at backpropagation.
        #       ┃ └─────┬─────┘
        #       ┠ ─ ─ ─ ┘
        #       ┃
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # ┌─────┸──────┐   (fence)
        # │    Copy    │
        # └─────┰──────┘
        for i, j in schedule:  # 针对 schedule 之中的每一对 i,j
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                _wait(batch, copy_streams[j][i], streams[j]) # 这里保证了同步完成

            # Determine whether checkpointing or not.
            checkpoint = i < checkpoint_stop
            if checkpoint:

                def function(
                    *inputs,
                    partition: nn.Module = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return partition(*inputs)

                chk = Checkpointing(function, batch)  # type: ignore[arg-type]
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:

                def compute(
                    batch: Batch = batch,
                    partition: nn.Module = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> Batch:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return batch.call(partition) # 前向计算，计算以 partition为单位计算，partition内部的层是顺序计算，由 Sequential保证。

                task = Task(streams[j], compute=compute, finalize=None)
                del compute

            # Compute tasks in parallel. ([2] in the diagram)
            self.in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = self.out_queues[j].get()  # 获取 worker 的前向计算结果，就是 第 j 个device 对 第 i 个 batch 的计算结果

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n - 1:
                _wait(batch, streams[j], copy_streams[j][i])

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(devices[j]):
                task.finalize(batch)

            # 第 j 个device 对 第 i 个 batch 的计算 就是 F[i,j]
            # 这里是关键，就是把 第 j 个device 对 第 i 个 batch 的计算结果 赋值到 batches[i]，batches[i]就是 batches[i][j]，在下次计算时候，构建的就是 F[i,j+1], 下一次 fence 之中的 depend 操作，就是针对 batches[i,j+1]
            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

'''
关于这个赋值操作，其对应的grad_fn 是 PermuteBackward，比如：

a = torch.tensor([2., 3.], requires_grad=True)
c = a
c.backward(gradient=external_grad)
print(c)
具体是：

c = {Tensor: 2} tensor([2., 3.], requires_grad=True)
  T = {Tensor: 2} tensor([2., 3.], grad_fn=<PermuteBackward>)
现在，我们把下图进行升级。

                 +-------------------------------------------------------------------+
                 | depend                                                            |
                 |                                                                   |
                 | +---------------+                                                 |
                 | |fork           |                                                 |
+-------------   | |               |     +-----------+                               |
|            |   | |               |     |           |                               |
|batches[i]  +-------------------------> | batches[i]|                               |
|            |   | |               |     |           |                               |
+-------------   | |               |     +-----------+                               |
                 | |               |                                                 |
                 | |               |                                                 |
                 | |               |     +--------+    +-------+                     |
                 | |  get_phony +------> |        +--->+ Join  |                     |
                 | |               |     | phony  |    |       |                     |
                 | +---------------+     |        |    |       |                     |
                 |                       +--------+    |       |                     |
                 |                                     |       |                     |
+-------------   |                                     |       |    +--------------+ |
|            |   |                                     |       |    |              | |
|batches[i+1]+----------------------------------------------------->+ batches[i+1] | |
|            |   |                                     |       |    |              | |
+-------------   |                                     |       |    +--------------+ |
                 |                                     +-------+                     |
                 |                                                                   |
                 +-------------------------------------------------------------------+
我们进行横向拓展，得到如下，即一个batch 被分成两个小批次: batches[i]，batches[i+1] ，它们在两个设备 partitions[j]，partitions[j + 1] 之上流水线，这样行和列都有反向传播的依赖。

                                 F[i,j]                                                                            F[i,j+1]

                    +------------------------------------------------+                            +-----------------------------------------------+
                    | partitions[j]                                  |                            |  partitions[j+1]                              |
                    |                                                |                            |                                               |
                    | +--------------------+   +------------------+  |                            | +-------------------+   +------------------+  |
                    | |fence               |   | compute          |  |                            | | fence             |   | compute          |  |
                    | |                    |   |                  |  |                            | |                   |   |                  |  |
+--------------+    | |  +--------------+  |   |  +------------+  |  |     +-----------------+    | |   +-------------+ |   |  +------------+  |  |       +-----------------+
|              |    | |  | depend       |  |   |  |forward     |  |  |     |                 |    | |   | depend      | |   |  |forward     |  |  |       |                 |
|  batches[i]  +---------------------------------------------------------> | batches[i][j]   +----------------------------------------------------------> | batches[i][j+1] |
|              |    | |  |              |  |   |  |            |  |  |     |                 |    | |   |             | |   |  |            |  |  |       |                 |
+--------------+    | |  |              |  |   |  |            |  |  |     +-----------------+    | |   |             | |   |  |            |  |  |       +-----------------+
                    | |  |              |  |   |  +------------+  |  |                            | |   |             | |   |  +------------+  |  |
                    | |  |              |  |   |                  |  |                            | |   |             | |   |                  |  |
+--------------+    | |  |              |  |   +------------------+  |     +-----------------+    | |   |             | |   +------------------+  |       +-------------------+
|              |    | |  |              |  |                         |     |                 |    | |   |             | |                         |       |                   |
|  batches[i+1]+---------------------------------------------------------> | batches[i+1][j] +----------------------------------------------------------> | batches[i+1][j+1] |
|              |    | |  |              |  |                         |     |                 |    | |   |             | |                         |       |                   |
+--------------+    | |  +--------------+  |                         |     +-----------------+    | |   +-------------+ |                         |       +-------------------+
                    | |                    |                         |                            | |                   |                         |
                    | +--------------------+                         |                            | +-------------------+                         |
                    +------------------------------------------------+                            +-----------------------------------------------+


'''