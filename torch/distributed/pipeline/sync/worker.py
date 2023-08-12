# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Multithreading in pipeline parallelism."""
from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, cast

import torch

from .microbatch import Batch
from .stream import AbstractStream, use_device, use_stream

__all__: List[str] = ["Task", "worker", "create_workers", "spawn_workers"]


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
Task 表示如何在一个分区上计算微批次数据（micro-batch）。它由两部分组成：
    compute应在工作线程中并发执行。
    finalize应在工作线程完成后执行。
可以理解为一个业务处理逻辑。如果有安卓经验的同学，可以理解为类似于 业务Message。其实 Android message也叫task，其封装了本任务携带的信息和处理该任务的handler。

这里的 Task 也是类似的，在构建Task 时候，就传入了 compute 方法和finalize方法，举例如下：

    task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
或者如下：
    def compute(batch: Batch = batch,
                partition: nn.Sequential = partition,
                skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                ) -> Batch:
        with use_skip_tracker(skip_tracker):
            return batch.call(partition)
    
    task = Task(streams[j], compute=compute, finalize=None)
具体Task定义如下，Task是绑定在 Stream 之上，即可以运行在任何device之上，这就用到了上一节的内容。
'''
class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(
        self, stream: AbstractStream, *, compute: Callable[[], Batch], finalize: Optional[Callable[[Batch], None]],
    ) -> None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self) -> Batch:
        ## 绑定在stream之上
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            return self._compute() # 调用传入的业务代码

    def finalize(self, batch: Batch) -> None:
        if self._finalize is None:
            return
        ## 绑定在stream之上
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            self._finalize(batch) # 调用传入的业务代码

'''
worker是用来运行task的，每个 device 有一个 worker 来负责执行这个 device 上的 task。如果有安卓经验的同学，可以理解为是 Looper。
需要注意，worker只是一个函数，如果运行，还需要一个线程作为寄托。这就是后续 spawn_workers 的工作。
'''
def worker(in_queue: InQueue, out_queue: OutQueue, device: torch.device) -> None:
    """The main loop of a worker thread."""
    with use_device(device):
        while True:
            task = in_queue.get() # 从输入队列中获取task

            if task is None:
                break

            try:
                batch = task.compute()  # 计算task
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue

            out_queue.put((True, (task, batch)))   # 把task和计算结果放到输出队列

    done = (False, None)
    out_queue.put(done)

def create_workers(devices: List[torch.device],) -> Tuple[List[InQueue], List[OutQueue]]:
    """Spawns worker threads. A worker thread is bound to a device."""
    in_queues: List[InQueue] = []
    out_queues: List[OutQueue] = []

    # Spawn workers.
    workers: Dict[torch.device, Tuple[InQueue, OutQueue]] = {}

    def normalize_device(device: torch.device) -> torch.device:
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", index=torch.cuda.current_device())

        if device.type == "cpu" and device.index is not None:
            return torch.device("cpu")

        return device

    for device in devices:
        device = normalize_device(device)  # 得到使用的设备

        try:
            in_queue, out_queue = workers[device]  # 临时放置queue
        except KeyError: # 如果 device 还没有生成对应的queues，则生成
            in_queue = Queue()  # 生成新的queue
            out_queue = Queue()

            # 取出queue
            workers[device] = (in_queue, out_queue)

            t = Thread(target=worker,  # Thread的执行程序是 worker 函数
                       args=(in_queue, out_queue, device),
                       daemon=True,)
            t.start()

        in_queues.append(in_queue) # 插入queue
        out_queues.append(out_queue) # 插入queue

    return (in_queues, out_queues)
'''
这里使用了 @contextmanager 注解，这是实现了上下文管理协议的对象，主要用于保存和恢复各种全局状态，关闭文件等，并为try...except...finally提供了一个方便使用的封装。

spawn_workers 为每个 device 生成了一个 Thread，这个 Thread 的执行函数是 worker。

spawn_workers 不止生成了若干 workers，也生成了一对消息队列 (in_queues, out_queues) ，这个 (in_queues, out_queues) 在Pipeline 生命周期之内全程都存在，具体来说是：

spawn_workers 内部会针对每一个device生成一个 in_queue, out_queue。所以可保证每个device之上是串行来执行业务操作。
in_queue, out_queue = workers[device]
这些 queues 被添加到 (in_queues, out_queues) 之中。
in_queues.append(in_queue)
out_queues.append(out_queue)
之后就是使用 (in_queues, out_queues) 作为各个task 之间传递信息的上下文。

in_queues 里面的顺序就是 device 的顺序，也就是partition的顺序。out_queues 亦然。

具体代码如下：
'''
@contextmanager
def spawn_workers(devices: List[torch.device],) -> Generator[Tuple[List[InQueue], List[OutQueue]], None, None]:
    try:
        (in_queues, out_queues) = create_workers(devices)
        yield (in_queues, out_queues)  # 返回给调用者
    finally:
        pass


'''
我们总结梳理一下大致业务逻辑（后文还会细化）：

    系统调用 spawn_workers 来生成若干 workers。
    
    spawn_workers 为每个 device 生成了一个 Thread，这个 Thread 的执行函数是 worker。
    spawn_workers 内部也会针对每一个device生成一个 in_queue, out_queue。所以可保证每个device之上是串行来执行业务操作。
    
    这些 queues 被添加到 (in_queues, out_queues) 之中。然后把 (in_queues, out_queues) 返回给 Pipeline 主线程。
    之后就是使用 (in_queues, out_queues) 作为各个task 之间传递信息的上下文。
    
    Pipeline 主线程得到 (in_queues, out_queues) 之后，如果要通过 compute 方法运行一个Task，就找到其device对应的in_queue，把Task插进去。
    Worker Thread 阻塞在 in_queue 之上，如果发现有内容，就读取 Task，运行Task。
    Worker Thread 把运行结果插入到 out_queue之中。
    Pipeline 的 compute 方法会取出 out_queue 之中的运行结果，进行后续处理。
如下图所示：

                           +-------------------------------------------------------------------------+
                           |                                1                                        |
                           |     +--------------------------------------------------------------+    |
                           |     |               3   (in_queues, out_queues)                    |    |
                           |     v                                                              |    v
+--------------------------------+---------+                                             +------+----+-----------------------------------------------------------------------+
| Pipeline                 |               |                                             | spawn_workers                                                                     |
|                          |               |                                             |                                                                                   |
|                          |               |                                             | +-------------------------------------+                                           |
|                          |               |                                             | | workers                             |                                           |
|                          |               |                                             | |                                     |     t = Thread(                           |
|                          +               |                                             | |                                     |       target=worker,                      |
|                 spawn_workers(devices)   |                                             | |  device 1 : in_queue 1, out_queue 1 |       args=(in_queue, out_queue, device), |
|                                          |                                             | |                                     |       daemon=True,                        |
|                                          |                                             | |  device 2 : in_queue 2, out_queue 2 |     )                                     |
| +--------------------------------------+ |                                             | |                                     |     t.start()                             |
| | compute                              | |                                             | |  device 3 : in_queue 3, out_queue 3 |          +                                |
| |                                      | |                                             | |                                     |          |                                |
| |                                      | |    4                                        | |                                     |          |                                |
| |  in_queues[j].put(task)  +-----------------------+                                   | +-------------------------------------+          |                                |
| |                                      | |         |                                   +-----------------------------------------------------------------------------------+
| |                                      | |         |                                                                                      | 2
| |  ok, payload = out_queues[j].get()<--------+     |         +---------------------+                                                      |
| |                                      | |   |     |         | in_queues           |                                                      v
| +--------------------------------------+ |   |     |         |                     |
|                                          |   |     +------------> in_queue 1 +--------+          +---------------------------------------------------------------------+
+------------------------------------------+   |               |    in_queue 2       |  |          | Thread                                                              |
                                               |               |    in_queue 3       |  |          |                                                                     |
                                               |               |                     |  | 5        |    +------------------------------------------------------------+   |
                                               | 7             +---------------------+  |          |    | Worker                                                     |   |
                                               |               +---------------------+  |          |    |                                                            |   |
                                               |               | out_queues          |  |          |    |        device 1      task = in_queue.get()                 |   |
                                               |               |                     |  |  task    |    |                                                            |   |
                                               +------------------+ out_queue 1 <--+ |  +----------------------> in_queue 1    batch = task.compute()                |   |
                                      (True, (task,,batch))    |    out_queue 2    | |             |    |                                                            |   |
                                                               |    out_queue 3    +---------------------------+ out_queue 1   out_queue.put((True, (task, batch)))  |   |
                                                               |                     |      6      |    |                                                            |   |
                                                               +---------------------+             |    +------------------------------------------------------------+   |
                                                                                                   +---------------------------------------------------------------------+



'''