# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Autograd functions for stream-aware CUDA copy. It is used to overlap copy
and computation on the same GPU.
"""
from collections import deque
from typing import Deque, List, Optional, Tuple, Sequence

import torch
from torch import Tensor

from .stream import AbstractStream, current_stream, get_device, record_stream, use_stream, wait_stream

__all__: List[str] = []


Tensors = Sequence[Tensor]


# Common interface between :class:`Copy` and :class:`Wait`.
class Context:
    prev_stream: AbstractStream
    next_stream: AbstractStream

#Copy 拓展了torch.autograd.Function，主要就是应用record_stream来协助完成拷贝业务。
# 6.3.2 Copy 算子
class Copy(torch.autograd.Function):
    """Copies tensors on specific streams."""

    @staticmethod
    # type: ignore[override]
    def forward(ctx: Context, prev_stream: AbstractStream, next_stream: AbstractStream, *input: Tensor,) -> Tensors:
        # 这里会把拷贝操作的source，dst 都保存在上下文之中，反向操作时候会取出来
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        output = []
        output_stream = current_stream(get_device(next_stream))   # 得到下一个流

        with use_stream(prev_stream), use_stream(next_stream):
            for x in input:
                y = x.to(get_device(next_stream), non_blocking=True) # 进行拷贝操作  # 把 input 拷贝到 next_stream
                output.append(y)

                # 'prev_stream' is not where 'x' has been allocated.
                record_stream(x, prev_stream)  # 记录流，确保拷贝完成之前不会使用 x
                # 'y' has been allocated on 'next_stream'.
                # It might be used on the current stream captured as 'output_stream'.
                record_stream(y, output_stream)  # 记录流，确保拷贝完成之前不会使用 y

        return tuple(output)  # 返回输出

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor,) -> Tuple[Optional[Tensor], ...]:
        # 取出来上下文保存的拷贝操作的src，dst。
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        grad_input: Deque[Tensor] = deque(maxlen=len(grad_output))
        input_stream = current_stream(get_device(prev_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in reversed(grad_output):
                y = x.to(get_device(prev_stream), non_blocking=True) # 进行拷贝操作
                grad_input.appendleft(y)

                # 'next_stream' is not where 'x' has been allocated.
                record_stream(x, next_stream)
                # 'y' has been allocated on 'prev_stream'.
                # It might be used on the current stream captured as 'input_stream'.
                record_stream(y, input_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + tuple(grad_input)

# 6.3.3 Wait 算子
# Wait 算子代码如下，主要就是起到同步作用，等待拷贝操作的完成。
class Wait(torch.autograd.Function):
    """Synchronizes a stream to another stream.

    Place it just before you want to start an operation on the next stream,
    provided that all operations on the previous stream are done.

    """

    @staticmethod
    # type: ignore[override]
    def forward(ctx: Context, prev_stream: AbstractStream, next_stream: AbstractStream, *input: Tensor,) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        wait_stream(next_stream, prev_stream)

        return tuple(x.detach() for x in input)

    @staticmethod
    def backward(ctx: Context, *grad_input: Tensor,) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        wait_stream(prev_stream, next_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + grad_input
