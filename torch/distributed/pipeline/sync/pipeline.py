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

__all__: List[str] = []


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
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)

# 6.3.4 封装
# 以下函数对算子进行了封装。
def _copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def _wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


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
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


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

'''
2.5 使用
2.5.1 何时生成worker
使用例子位于 torchgpipe/pipeline.py，在 Pipeline 类之中的 run 函数中会生成workers。
我们可以看到，对于 Pipeline 来说，有意义的就是 (in_queues, out_queues)。
'''
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

        # 这里是按照算法有次序的运行多个fence, compute  , yknote代码有不同
        for schedule in _clock_cycles(m, n):  # 这里使用，给出了执行序列计划，后续按照这个来执行
            self.fence(batches, schedule, skip_trackers)
            # 把队列传递进去
            self.compute(batches, schedule, skip_trackers)

# 6.3.5 建立依赖关系
# fence 简化代码如下，其建立了图例之中的行，列 两种依赖关系。
# 2.3 使用
# 在 Pipeline 之中我们可以看到具体的使用方法，fence 方法（省略部分代码）利用 depend 来构建后向传播的依赖关系，确保 batches[i-1] 在 batches[i] 之后完成。
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

            # 拿到dst设备的拷贝流
            next_stream = copy_streams[j][i]

            # 残差连接相关设置
            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            # 建立跨设备依赖关系，指定了 device[j-1] 的输出是 device[i] 的输入
            if j != 0:
                prev_stream = copy_streams[j - 1][i]  # 拿到src设备的拷贝流
                _copy(batches[i], prev_stream, next_stream)   #建立跨设备依赖关系
'''
2.5.2 剖析
Torchgpipe 使用了 Python 的 Queue 数据结构。
Queue 类实现了一个基本的先进先出（FIFO）容器。
    A multi-producer, multi-consumer queue.
其主要方法是：
    Queue.get([block, [timeout]]) 读队列，从队列尾部移除元素，timeout为等待时间，如果队列满，则阻塞。
    Queue.put(item, [block, [timeout]]) 写队列，将元素添加到序列尾端，timeout为等待时间，如果队列空，则阻塞。
我个人更习惯于把 (in_queues, out_queues) 理解为类似 Linux 的 管道（Pipe）。
Linux 管道是一种最基本的IPC机制，作用于有血缘关系的进程之间，完成数据传递，具体特性如下：
    管道是由核函数管理的一个FIFO文件，其实是一个缓冲区，相当于我们放入内存中的一个管道，两个进程分别处于管道两端，通过这个管道来传递信息。
    管道的一端连接一个进程的输出。这个进程会向管道中放入信息。当管道被放满信息的时候，尝试放入信息的进程会等待，直到另一端的进程取出信息。
    管道的另一端连接另一个进程的输入，这个进程取出被放入管道的信息。当管道中没有信息的话，从管道中读取的进程会等待，直到另一端的进程放入信息。
具体回到 TorchPipe，我们提前看看论文的内容：
    对于这种细粒度的顺序控制，torchgpipe把checkpointing 使用两个单独的autograd函数Checkpoint和Recompute来实现。
    在任务 F′i,j 的执行时间之内，生成一对具有共享内存的Checkpoint和Recompute。
    该共享内存在向后传播中被使用，用于将通过执行Recompute生成的本地计算图传输到Checkpoint来进行反向传播。

于是，这里就有很多并行处理的需求，
于是我们可以看到 Pipeline 类的 compute 方法（省略部分代码）中有向 in_queues 之中放入 Task，
从 out_queues 之中去除 Task 的执行结果。
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
        for i, j in schedule:   # 并行执行  # 针对 schedule 之中的每一对 i,j
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                _wait(batch, copy_streams[j][i], streams[j])

            # Determine whether checkpointing or not.
            checkpoint = i < checkpoint_stop
            if checkpoint:

                def function(
                    input: TensorOrTensors,
                    partition: nn.Sequential = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return partition(input)

                chk = Checkpointing(function, batch)
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:

                def compute(
                    batch: Batch = batch,
                    partition: nn.Sequential = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> Batch:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return batch.call(partition)  # 前向计算，计算以 partition为单位计算，partition内部的层是顺序计算，由 Sequential保证
                # 生成一个Task
                task = Task(streams[j], compute=compute, finalize=None)
                del compute

            # Compute tasks in parallel. ([2] in the diagram)
            # 给第j个partition放入一个新的task。因为 i, j 已经在clock算法中设定了，所以前向传播就是按照这个来走的。
            self.in_queues[j].put(task)  # 让 worker计算

        for i, j in schedule:
            # 取出第j个partition的运行结果
            # 获取 worker 的前向计算结果，就是 第 j 个device 对 第 i 个 batch 的计算结果
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
                _wait(batch, streams[j], copy_streams[j][i])  # 这里保证了同步完成

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(devices[j]):
                task.finalize(batch)

            # 第 j 个device 对 第 i 个 batch 的计算 就是 F[i,j]
            ## 这里是关键，就是把 第 j 个device 对 第 i 个 batch 的计算结果 赋值到 batches[i]，batches[i]就是 batches[i][j]，
            # 在下次计算时候，构建的就是 F[i,j+1], 下一次 fence 之中的 depend 操作，就是针对 batches[i,j+1]
            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
