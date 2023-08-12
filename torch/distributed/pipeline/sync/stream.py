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

__all__: List[str] = ["CPUStreamType", "new_stream", "current_stream", "default_stream",
                      "use_device", "use_stream", "get_device", "wait_stream", "record_stream",
                      "is_cuda", "as_cuda"]

'''
2.1 Stream
Stream 类是用来封装 CUDA stream 和 CPU stream。代码位于：torchgpipe/stream.py。

CUDA流表示一个GPU操作队列，即某个设备绑定的，按照顺序执的核（kernel）序列。
我们可以把一个流看作是GPU之上的一个任务。
用户向流的队列上添加一系列操作，GPU会按照添加到流中的先后顺序而依次执行这一系列操作。
在同一个流之中，所有操作是串行序列化，因此这些操作永远不会并行。
因此，要想并行，两个操作必须位于不同的 stream 中。不同流中的核函数可以交错，甚至可能重叠
'''
class CPUStreamType:
    pass


# The placeholder on place of streams for the CPU device instead of CUDA.
CPUStream = CPUStreamType()

# It represents both CUDA streams and the CPU stream.
AbstractStream = Union[torch.cuda.Stream, CPUStreamType]


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

#torch.cuda.stream(stream) 的作用是选择给定流的上下文管理器。
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
        tensor = tensor.new_empty([0]).set_(tensor._typed_storage())

        # Typechecking: torch.cuda.Stream is incompatible with torch._C.Stream
        tensor.record_stream(as_cuda(stream))  # type: ignore[arg-type]


def is_cuda(stream: AbstractStream) -> bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def as_cuda(stream: AbstractStream) -> torch.cuda.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.cuda.Stream, stream)
