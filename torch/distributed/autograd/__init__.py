
import sys
import torch


def is_available():
    return hasattr(torch._C, "_dist_autograd_init")


if is_available() and not torch._C._dist_autograd_init():
    raise RuntimeError("Failed to initialize torch.distributed.autograd")

if is_available():
    from torch._C._distributed_autograd import (
        get_gradients,
        backward,
        _init,
        _new_context,
        _release_context,
        _get_max_id,
        _is_valid_context,
        _retrieve_context,
        _current_context,
        _get_debug_info,
        DistAutogradContext,
    )


class context:
    '''
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = torch.rand((3, 3), requires_grad=True)
        >>>     t2 = torch.rand((3, 3), requires_grad=True)
        >>>     loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()

        我们找一找如何发起反向传播，按照从下往上的顺序进行。这里也有两种：

            一种是主动发起，比如上图之中 worker 0 的 loss 之上主动调用backward 方法。
            一种是内部隐式发起，比如上图的 worker 0 之中的 t3 如何通过 recv 告诉 worker 1，你应该启动反向传播了。

        >>>     dist_autograd.backward(context_id, [loss])
    '''
    #当生成时，__enter__ 会调用 _new_context() 在C++生成一个context。
    def __enter__(self):
        self.autograd_context = _new_context() # 这里生成一个上下文
        return self.autograd_context._context_id()

    def __exit__(self, type, value, traceback):
        _release_context(self.autograd_context._context_id())
