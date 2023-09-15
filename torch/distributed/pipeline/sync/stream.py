# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for eliminating boilerplate code to handle abstract streams with
CPU device.
"""
from contextlib import contextmanager
from typing import Generator, List, Union, cast

import torch

__all__: List[str] = []


class CPUStreamType:
    pass


# The placeholder on place of streams for the CPU device instead of CUDA.
CPUStream = CPUStreamType()

# It represents both CUDA streams and the CPU stream.
AbstractStream = Union[torch.cuda.Stream, CPUStreamType]

'''
2.3.2 生成/获取
关于生成和获取的函数为：

    new_stream 会生成一个新的stream。
    
    current_stream 返回当前流。
    
    default_stream 返回了缺省流。
'''
def new_stream(device: torch.device) -> AbstractStream:
    """Creates a new stream for either CPU or CUDA device."""
    if device.type != "cuda":
        return CPUStream
    return torch.cuda.Stream(device)


def current_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != "cuda":
        return CPUStream
    return torch.cuda.current_stream(device)


def default_stream(device: torch.device) -> AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != "cuda":
        return CPUStream
    return torch.cuda.default_stream(device)


@contextmanager
def use_device(device: torch.device) -> Generator[None, None, None]:
    """:func:`torch.cuda.device` for either CPU or CUDA device."""
    if device.type != "cuda":
        yield
        return

    with torch.cuda.device(device):
        yield


@contextmanager
def use_stream(stream: AbstractStream) -> Generator[None, None, None]:
    """:func:`torch.cuda.stream` for either CPU or CUDA stream."""
    if not is_cuda(stream):
        yield
        return

    with torch.cuda.stream(as_cuda(stream)):
        yield


def get_device(stream: AbstractStream) -> torch.device:
    """Gets the device from CPU or CUDA stream."""
    if is_cuda(stream):
        return as_cuda(stream).device
    return torch.device("cpu")

'''
2.3.4 等待
以下方法封装了CUDA wait_stream 。
    如果两个流都是CUDA流，则就是一个流等待另外一个流完成。
    否则采用 synchronize() 来保证 CPU 等待 CUDA 完成。
因为这里流操作是异步的，所以当函数返回时候无法确定操作是否已经完成，所以将CPU和主机进行同步，或者CUDA流之间进行同步，以确保GPU完成流操作。

这里wait_stream和synchronize最终都会完成等待操作，比如synchronize最终调用到了 cudaDeviceSynchronize，
该方法将停止CPU端线程的执行，直到GPU端完成此前CUDA上的任务（包括kernel函数、数据拷贝等）。

既然已经把 Stream 操作进行了基础封装，torchgpipe 接下来就使用这些封装函数实现了拷贝操作和等待操作，我们接下来看看。
'''
def wait_stream(source: AbstractStream, target: AbstractStream) -> None:
    """:meth:`torch.cuda.Stream.wait_stream` for either CPU or CUDA stream. It
    makes the source stream wait until the target stream completes work queued.
    """
    if is_cuda(target):
        if is_cuda(source):
            # A CUDA stream waits another CUDA stream.
            as_cuda(source).wait_stream(as_cuda(target))
        else:
            # CPU waits a CUDA stream.
            as_cuda(target).synchronize()

    # If the target is CPU, synchronization is not required.

# 2.3.3 记录
# 以下方法用来封装了CUDA record_stream。
def record_stream(tensor: torch.Tensor, stream: AbstractStream) -> None:
    """:meth:`torch.Tensor.record_stream` for either CPU or CUDA stream."""
    if is_cuda(stream):
        # NOTE(sublee): record_stream() on a shifted view tensor throws
        # RuntimeError in PyTorch 1.1.0, and does nothing in 1.2.0. To safely
        # protect the tensor against unexpected reallocation, here we use a
        # temporal tensor associated with the same storage without shifting as
        # a workaround.
        #
        # Issue: https://github.com/pytorch/pytorch/issues/27366
        #
        tensor = tensor.new_empty([0]).set_(tensor.storage())

        # Typechecking: torch.cuda.Stream is incompatible with torch._C.Stream
        tensor.record_stream(as_cuda(stream))  # type: ignore[arg-type]


def is_cuda(stream: AbstractStream) -> bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def as_cuda(stream: AbstractStream) -> torch.cuda.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.cuda.Stream, stream)
