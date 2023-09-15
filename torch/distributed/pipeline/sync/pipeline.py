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

# 2.4 拷贝API
# 拷贝流的 API 如下，其实就是调用了 Copy 这个类的forward方法。
# 6.3.4 封装
# 以下函数对算子进行了封装。
def _copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])

# 2.5 等待API
# wait 则是调用了 Wait 类的forward方法。
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
'''
2.6.4 预先拷贝
fence 方法做了预先拷贝操作，其中会做如下操作：
    设定依赖关系，这个我们在前文中分析过。
    得到下一个设备的拷贝流。
    得到上一个设备的拷贝流。
    拷贝前面流到后续流。
    
我们按照之前文章的例子来看看，下面是一个schedule 生成序列。

m=4 # m: number of micro-batches
n=3 # n: number of partitions
for k in range(m + n - 1):
    print( [(k - j + 1 , j +1 ) for j in range(max(1 + k - m, 0), min(1 + k, n))] )

打印是：
    [(1, 1)]                  # 第 1 轮训练计划 & 数据
    [(2, 1), (1, 2)]          # 第 2 轮训练计划 & 数据
    [(3, 1), (2, 2), (1, 3)]  # 第 3 轮训练计划 & 数据
    [(4, 1), (3, 2), (2, 3)]  # 第 4 轮训练计划 & 数据
    [(4, 2), (3, 3)]          # 第 5 轮训练计划 & 数据
    [(4, 3)]                  # 第 6 轮训练计划 & 数据
前 6 个周期对应了如下时间流，第一个时钟周期 (1,1) 进入系统，第二个周期 (2,1) 进入系统 .....

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

我们以如下计划看看，重点是第 3 个时钟周期完成的任务。

第 2 个时钟周期完成了如下操作。
    [(2, 1), (1, 2)]         # 第 2 轮训练计划 & 数据
第 3 个时钟周期的计划如下：
    [(3, 1), (2, 2), (1, 3)] # 第 3 轮训练计划 & 数据
就是对 schedule 的每个 i, j，都分别拷贝 copy_streams[j-1][i] 到 copy_streams[j][i]。
注意 我们之前的提到的，_copy_streams[i][j] 之中，i 表示 device 的序列，j 表示 batch 序列，和schedule 的 i,j 恰好相反。
所以对于我们例子，在第 3 个时钟周期内的拷贝操作是 （这里 i 和 j 在循环和后续数组提取时候是相反，
这个恰好和schedule对应，于是负负得正，最终 i, j 可以对应上）：
    对于 (3, 1)，这个是新数据进入了 device 1，不需要拷贝。
    对于 (2, 2)，拷贝是 (2,1) 到 (2,2)。
    对于 (1, 3)，拷贝是 (1,2) 到 (1,3)。
具体如下图所示，这几个拷贝可以并行操作，因为拷贝流不是运行计算的缺省流，所以也可以和计算并行。

         +             +            +             +            +            +             +
         |             |            |             |            |            |             |
 cuda:0  |    (1,1)    |   (2,1)    |   (3,1)     |   (4,1)    |            |             |
         |             |     +      |             |            |            |             |
         |             |     |      |             |            |            |             |
         |             |     |      |             |            |            |             |
         |             |     |      |             |            |            |             |
         |             |     +------------+       |            |            |             |
         |             |            |     |       |            |            |             |
         |             |            |     |       |            |            |             |
         |             |            |     |       |            |            |             |
         |             |            |     v       |            |            |             |
         |             |            |             |            |            |             |
 cuda:1  |             |   (1,2)    |   (2,2)     |   (3,2)    |  (4,2)     |             |
         |             |     +      |             |            |            |             |
         |             |     |      |             |            |            |             |
         |             |     |      |             |            |            |             |
         |             |     +-----------+        |            |            |             |
         |             |            |    |        |            |            |             |
         |             |            |    |        |            |            |             |
         |             |            |    |        |            |            |             |
         |             |            |    v        |            |            |             |
 cuda:2  |             |            |   (1,3)     |   (2,3)    |  (3,3)     |     (4,3)   |
         |             |            |             |            |            |             |
         |             |            |             |            |            |             |
         |             |            |             |            |            |             |
         |   clock 1   |  clock 2   |   clock 3   |  clock 4   |  clock 5   |     clock 6 |
         +             +            +             +            +            +             +

+----------------------------------------------------------------------------------->  Time    
'''
# 1.3.1 构建依赖
# 在 Pipeline 之中，fence 方法（省略部分代码）利用 depend 来构建后向传播的依赖关系
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
                _depend(batches[i - 1], batches[i]) # 在这里建立了后向传播依赖关系  # 设定依赖关系

            # 拿到dst设备的拷贝流
            next_stream = copy_streams[j][i]  # 得到下一个设备的拷贝流，注意，这里和for的i,j相反

            # 残差连接相关设置
            for prev_j, ns, name in skip_layout.copy_policy(j):  # 因为篇幅原因，我们不分析这部分
                prev_stream = copy_streams[prev_j][i]  # 拷贝前面流到后续流
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            # 建立跨设备依赖关系，指定了 device[j-1] 的输出是 device[i] 的输入
            if j != 0:
                prev_stream = copy_streams[j - 1][i]  # 拿到src设备的拷贝流  # 得到上一个设备的拷贝流
                # 从之前的micro-batches进行拷贝
                _copy(batches[i], prev_stream, next_stream)   #建立跨设备依赖关系   # 拷贝前面流到后续流

'''
2.6.5 计算
compute 完成了如下步骤：

    使用 wait(batch, copy_streams[j][i], streams[j]) "拷贝流"同步到"计算流"，确保拷贝操作完成。
    
    其次进行计算。
    
    使用 wait(batch, streams[j], copy_streams[j][i]) 把计算结果从"计算流"同步到"拷贝流"，确保计算操作完成。
    
具体如下：

针对论文就是： 01-36.png
'''

'''
1.3.2 Queue
Worker 和主线程之间使用了 Python 的 Queue 数据结构进行交互。Queue 类实现了一个基本的先进先出（FIFO）容器，使用 put() 将元素添加到序列尾端，get() 从队列尾部移除元素。
    A multi-producer, multi-consumer queue.
两个关键函数是：
    get([block, [timeout]]) 读队列，timeout为等待时间，如果队列满，则阻塞。
    put(item, [block, [timeout]]) 写队列，timeout为等待时间，如果队列空，则阻塞。
1.3.3 计算
具体训练是通过 compute 函数完成。
'''

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
            if j != 0:  # 等待拷贝结束
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
            self.in_queues[j].put(task)  # 让 worker计算  # 并行执行操作

        # 这里进行了同步操作
        for i, j in schedule:
            # 取出第j个partition的运行结果
            # 获取 worker 的前向计算结果，就是 第 j 个device 对 第 i 个 batch 的计算结果
            ok, payload = self.out_queues[j].get()  # 等待运行结果

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n - 1:   # 拷贝输出
                _wait(batch, streams[j], copy_streams[j][i])  # 这里保证了同步完成  # 这里有同步

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

        '''
我们总结梳理一下大致业务逻辑（并行逻辑）：

    系统调用 spawn_workers 来生成若干 workers。
    
        spawn_workers 为每个 device 生成了一个 Thread，这个 Thread 的执行函数是 worker。
        spawn_workers 内部也会针对每一个device生成一个 in_queue, out_queue。
        所以可保证每个device之上是串行来执行业务操作。
        
        这些 queues 被添加到 (in_queues, out_queues) 之中。然后把 (in_queues, out_queues) 返回给 Pipeline 主线程。
        之后就是使用 (in_queues, out_queues) 作为各个task 之间传递信息的上下文。
        
    Pipeline 主线程得到 (in_queues, out_queues) 之后，使用clock_cycles 算法生成一系列迭代，每个迭代是一个schedule。
    
    对于每个迭代（schedule），先用fence来进行拷贝stream & 设定依赖，然后使用 compute 来进行训练。这就顺序启动了多个 compute。
    
    在每个 compute 之中，遍历这个 schedule，对于其中 (i, j) 运行一个Task，即找到其device对应的in_queue，把Task插进去。
    
    Worker Thread 阻塞在 in_queue 之上，如果发现有内容，就读取 Task，运行。
    虽然多个 compute 是顺序执行，但是因为compute 只是一个插入queue操作，可以立即返回。
    而多个 worker Thread 阻塞在 queue 之上，这之后是可以并行训练的。
    
    Worker Thread 把运行结果插入到 out_queue之中。
    
    compute 方法会取出 out_queue 之中的运行结果，进行后续处理。
具体如下图。

         +-------------------------------------------------------------------+       +-----------------------------------------+
         | Pipeline                                                          |  1    | spawn_workers                           |
         |                                     spawn_workers(devices)  +-----------> |                                         |
         |                                                                   |       | +-------------------------------------+ |
         |               for schedule in clock_cycles(m, n)                  |       | | workers                             | |
         |                     +                                             |       | |                                     | |
         |                     | 2                                           |       | |                                     | |
         |                     |                                             |       | |  device 1 : in_queue 1, out_queue 1 | |
         |                     +-----------+---------------+                 |       | |                                     | |
         |                     |           |               |                 |       | |  device 2 : in_queue 2, out_queue 2 | |
         |                     v           v               v                 |       | |                                     | |
         |  +------------------+------+        +-----------+--------------+  |       | |  device 3 : in_queue 3, out_queue 3 | |
         |  | compute                 |        | compute                  |  |       | |                                     | |
         |  |                         |  3     |                          |  |       | |                                     | |
         |  |  in_queues[j].put(task) |        |   in_queues[j].put(task) |  |       | +-------------------------------------+ |
         |  |                         | ...... |                          |  |       |                                         |
         |  |  out_queues[j].get()    |        |   out_queues[j].get()    |  |       +-----------------------------------------+
         |  |                         |        |                          |  |
         |  +----------+---+----------+        +----------------+----+----+  |
         |             |   ^                                    ^    |       |
         |             |   |                                    |    |       |
         +-------------------------------------------------------------------+
                     7 |   | 4                                7 |    | 4
                       |   |                                    |    |
                       v   |                                    |    v
                 +-----+---+------------------------------------+----+-----+
                 |                in_queues        out_queues              |
+------------>   |                                                         |  <--------------------+
|                +-----+---------------------------------------------+-----+                       |
| 6                    |                                             |                           6 |
|                    5 |                                             | 5                           |
|                      |                                             |                             |
|                      |                                             |                             |
|    +-------------------------------------+          +-------------------------------------+      |
|    | Thread 1        |        device 1   |          | Thread 2     |             device 3 |      |
|    |                 |                   |          |              |                      |      |
|    | +---------------------------------+ |          | +---------------------------------+ |      |
|    | | Worker        |                 | |          | | Worker     |                    | |      |
|    | |               v                 | |          | |            v                    | |      |
|    | |  task = in_queue.get()          | |          | |   task = in_queue.get()         | |      |
|    | |                                 | |  ......  | |                                 | |      |
|    | |  batch = task.compute()         | |          | |   batch = task.compute()        | |      |
|    | |                                 | |          | |                                 | |      |
+--------+out_queue.put((task, batch)))  | |          | |   out_queue.put((task, batch))+--------->+
     | |                                 | |          | |                                 | |
     | +---------------------------------+ |          | +---------------------------------+ |
     +-------------------------------------+          +-------------------------------------+

手机如下：  01-33.png

'''
