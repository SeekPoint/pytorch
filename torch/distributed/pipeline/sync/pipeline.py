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


def _depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from_idx = fork_from.find_tensor_idx()
    join_to_idx = join_to.find_tensor_idx()

    fork_from[fork_from_idx], phony = fork(fork_from[fork_from_idx])
    join_to[join_to_idx] = join(join_to[join_to_idx], phony)


def _copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in batch])


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
                _depend(batches[i - 1], batches[i])

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j - 1][i]
                _copy(batches[i], prev_stream, next_stream)

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
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                _wait(batch, copy_streams[j][i], streams[j])

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
                        return batch.call(partition)

                task = Task(streams[j], compute=compute, finalize=None)
                del compute

            # Compute tasks in parallel. ([2] in the diagram)
            self.in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = self.out_queues[j].get()

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

            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
