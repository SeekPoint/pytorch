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

__all__: List[str] = []


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional["Task"]]
    OutQueue = Queue[Tuple[bool, Union[Tuple["Task", Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


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
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):  # 绑定在stream之上
            return self._compute()  # 调用传入的业务代码

    def finalize(self, batch: Batch) -> None:
        if self._finalize is None:
            return
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):  # 绑定在stream之上
            self._finalize(batch)   # 调用传入的业务代码

# 2.3 Worker
# worker是用来运行task的，每个 device 有一个 worker 来负责执行这个 device 上的 task。
# 如果有安卓经验的同学，可以理解为是 Looper。
# 需要注意，worker只是一个函数，如果运行，还需要一个线程作为寄托。这就是后续 spawn_workers 的工作。
def worker(in_queue: InQueue, out_queue: OutQueue, device: torch.device) -> None:
    """The main loop of a worker thread."""
    with use_device(device):
        while True:
            task = in_queue.get()  # 从输入队列中获取task

            if task is None:
                break

            try:
                batch = task.compute()  # 计算task
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue

            out_queue.put((True, (task, batch)))  # 把task和计算结果放到输出队列

    done = (False, None)
    out_queue.put(done)

'''
2.4 生成 worker
这里使用了 @contextmanager 注解，这是实现了上下文管理协议的对象，主要用于保存和恢复各种全局状态，
关闭文件等，并为try...except...finally提供了一个方便使用的封装。
    spawn_workers 为每个 device 生成了一个 Thread，这个 Thread 的执行函数是 worker。
    spawn_workers 不止生成了若干 workers，也生成了一对消息队列 (in_queues, out_queues) ，这个 (in_queues, out_queues) 在Pipeline 生命周期之内全程都存在，具体来说是：
    spawn_workers 内部会针对每一个device生成一个 in_queue, out_queue。所以可保证每个device之上是串行来执行业务操作。
        in_queue, out_queue = workers[device]
    这些 queues 被添加到 (in_queues, out_queues) 之中。
        in_queues.append(in_queue)
        out_queues.append(out_queue)
    之后就是使用 (in_queues, out_queues) 作为各个task 之间传递信息的上下文。 
    in_queues 里面的顺序就是 device 的顺序，也就是partition的顺序。out_queues 亦然。

具体代码如下：  yknote---名字改了--spawn_workers
'''
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
        device = normalize_device(device)   # 得到使用的设备

        try:
            in_queue, out_queue = workers[device]  # 临时放置queue
        except KeyError:   # 如果 device 还没有生成对应的queues，则生成
            in_queue = Queue()  # 生成新的queue
            out_queue = Queue()
            # 取出queue
            workers[device] = (in_queue, out_queue)# 赋值给workers

            t = Thread(target=worker,   # Thread的执行程序是 worker 函数
                       args=(in_queue, out_queue, device),
                       daemon=True,)
            t.start()  # 启动工作线程

        in_queues.append(in_queue)  # 插入queue
        out_queues.append(out_queue)  # 插入queue

    return (in_queues, out_queues)   # 返回给调用者  yknote代码有出入

@contextmanager
def spawn_workers(devices: List[torch.device],) -> Generator[Tuple[List[InQueue], List[OutQueue]], None, None]:
    try:
        (in_queues, out_queues) = create_workers(devices)
        yield (in_queues, out_queues)
    finally:
        pass
