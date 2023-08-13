DDP Communication Hooks
=======================

DDP communication hook is a generic interface to control how to communicate
gradients across workers by overriding the vanilla allreduce in
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.>`_.
A few built-in communication hooks are provided,
and users can easily apply any of these hooks to optimize communication.
Besides, the hook interface can also support user-defined communication
strategies for more advanced use cases.

How to Use a Communication Hook?
--------------------------------

To use a communication hook, the user just needs to let the DDP model register
the hook before the training loop as below.

:func:`torch.nn.parallel.DistributedDataParallel.register_comm_hook`

What Does a Communication Hook Operate On?
------------------------------------------

A communication hook provides a flexible way to allreduce gradients.
Therefore, it mainly operates on the gradients on each replica before allreduce,
which are bucketized to increase the overlap between communication and computation.
Particularly, :class:`torch.distributed.GradBucket` represents a bucket of gradient tensors to be allreduced.

.. autoclass:: torch.distributed.GradBucket

.. autofunction:: torch.distributed.GradBucket.index
.. autofunction:: torch.distributed.GradBucket.buffer
.. autofunction:: torch.distributed.GradBucket.gradients
.. autofunction:: torch.distributed.GradBucket.is_last
.. autofunction:: torch.distributed.GradBucket.set_buffer
.. autofunction:: torch.distributed.GradBucket.parameters

Default Communication Hooks
---------------------------

Default communication hooks are simple **stateless** hooks, so the input state
in ``register_comm_hook`` is either a process group or ``None``.
The input ``bucket`` is a :class:`torch.distributed.GradBucket` object.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
.. autofunction:: allreduce_hook
.. autofunction:: fp16_compress_hook
.. autofunction:: bf16_compress_hook

Additionally, a communication hook wrapper is provided to support :meth:`~fp16_compress_hook` or :meth:`~bf16_compress_hook` as a wrapper,
which can be combined with other communication hooks.

.. autofunction:: fp16_compress_wrapper
.. autofunction:: bf16_compress_wrapper

PowerSGD Communication Hook
---------------------------

PowerSGD (`Vogels et al., NeurIPS 2019 <https://arxiv.org/abs/1905.13727>`_)
is a gradient compression algorithm, which can provide very high compression
rates and accelerate bandwidth-bound distributed training.
This algorithm needs to maintain both some hyperparameters and the internal
state. Therefore, PowerSGD communication hook is a **stateful** hook,
and the user needs to provide a state object defined as below.

PowerSGD State
^^^^^^^^^^^^^^^^

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
.. autoclass:: PowerSGDState

PowerSGD Hooks
^^^^^^^^^^^^^^^^

.. warning ::
    PowerSGD typically requires extra memory of the same size as the model's
    gradients to enable error feedback, which can compensate for biased
    compressed communication and improve accuracy.

.. warning ::
    PowerSGD hooks may conflict with `Apex automatic mixed precision package <https://github.com/NVIDIA/apex>`_.
    Please use PyTorch `native automatic mixed precision package <https://pytorch.org/docs/stable/amp.html>`_
    instead.

.. autofunction:: powerSGD_hook
.. autofunction:: batched_powerSGD_hook

Debugging Communication Hooks
-----------------------------

As the name implies, debugging communication hooks are **only** used for debugging and performance optimization purpose.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks

.. warning ::
    Debugging communication hooks do not necessarily output the correct results.

.. autofunction:: noop_hook

Checkpointing of Communication Hooks
------------------------------------

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook

A stateful communication hook can be saved as a part of model checkpointing to enable trainer restarts.
To make a hook serializable, ``__setstate__`` and ``__getstate__`` should be defined.

.. warning ::
    ``__getstate__`` should exclude non-serializable attributes from a returned dictionary.

.. warning ::
    ``__setstate__`` should properly initialize non-serializable attributes, excluded from a provided ``state``.

:class:`PowerSGDState` has ``__setstate__`` and ``__getstate__`` implemented and can be used as a reference.

.. class:: PowerSGDState
    :noindex:

    .. automethod:: PowerSGDState.__getstate__
    .. automethod:: PowerSGDState.__setstate__

Here is a simple, end-to-end example of saving and reloading PowerSGD state and hook.
关于分布式通信，PyTorch 提供的几个概念是：进程组，后端，初始化，Store。

进程组 ：DDP是真正的分布式训练，可以使用多台机器来组成一次并行运算的任务。为了能够让 DDP 的各个worker之间通信，PyTorch 设置了进程组这个概念。
后端 ：后端这个概念是一个逻辑上的概念。本质上后端是一种IPC通信机制。对于用户来说，就是采用那种方式来进行集合通信，从代码上看，就是走什么流程（一系列流程），以及后端使用 ProcessGroupMPI 还是 ProcessGroupGloo .....。
初始化 : 虽然有了后端和进程组的概念，但是如何让 worker 在建立进程组之前发现彼此？ 这就需要一种初始化方法来告诉大家传递一个信息：如何联系到其它机器上的进程？
Store : 可以认为是分布式键值存储，这个存储在组中的进程之间共享信息以及初始化分布式包 （通过显式创建存储来作为init_method的替代）。
1.2 初始化进程组
在调用任何 DDP 其他方法之前，需要使用torch.distributed.init_process_group()进行初始化进程组。


0x01 回顾
1.1 基础概念
关于分布式通信，PyTorch 提供的几个概念是：进程组，后端，初始化，Store。

进程组 ：DDP是真正的分布式训练，可以使用多台机器来组成一次并行运算的任务。为了能够让 DDP 的各个worker之间通信，PyTorch 设置了进程组这个概念。
后端 ：后端这个概念是一个逻辑上的概念。本质上后端是一种IPC通信机制。对于用户来说，就是采用那种方式来进行集合通信，从代码上看，就是走什么流程（一系列流程），以及后端使用 ProcessGroupMPI 还是 ProcessGroupGloo .....。
初始化 : 虽然有了后端和进程组的概念，但是如何让 worker 在建立进程组之前发现彼此？ 这就需要一种初始化方法来告诉大家传递一个信息：如何联系到其它机器上的进程？
Store : 可以认为是分布式键值存储，这个存储在组中的进程之间共享信息以及初始化分布式包 （通过显式创建存储来作为init_method的替代）。
1.2 初始化进程组
在调用任何 DDP 其他方法之前，需要使用torch.distributed.init_process_group()进行初始化进程组。

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
该方法会初始化默认分布式进程组和分布式包。此方法会阻塞，直到所有进程都加入，函数定义如下：

init_process_group ( backend ,
                       init_method = None ,
                       timeout = default_pg_timeout ,
                       world_size =- 1 ,
                       rank =- 1 ,
                       store = None ,
                       group_name = '' ,
                       pg_options = None )
初始化进程组有两种主要方法：

明确指定 store，rank 和 world_size。
指定 init_method（一个 URL 字符串），它指示在哪里/如何发现对等点。
如果两者都没有指定，init_method则假定为“env://”。

因此大家可以看到，store 和 init_method 是互斥的。

参数具体如下：

后端 – 要使用的后端。有效值包括mpi，gloo，和nccl。该字段应作为小写字符串（例如"gloo"）给出，也可以通过Backend属性（例如Backend.GLOO）访问 。如果在nccl后端每台机器上使用多个进程，则每个进程必须对其使用的每个 GPU 具有独占访问权限，因为在进程之间共享 GPU 可能会导致死锁。
init_method – 指定如何初始化进程组的 URL。如果未指定init_method或store指定，则默认为“env://” 。与 store互斥。
world_size – 参与作业的进程数。如果store指定，则 world_size 为必需。
rank – 当前进程的等级（它应该是一个介于 0 和world_size-1之间的数字）。如果store指定，则 rank 为必需。
store – 所有 worker 都可以访问的键/值存储，用于交换连接/地址信息。与init_method 互斥。
timeout – 针对进程组执行的操作超时。默认值等于 30 分钟。这适用于gloo后端。对于nccl，这仅在环境变量NCCL_BLOCKING_WAIT 或NCCL_ASYNC_ERROR_HANDLING设置为 1 时 适用。
group_name – 组名。
pg_options ( Process Group Options , optional ) – 进程组选项，指定在构建特定进程组期间需要传入哪些附加选项。
0x02 概念与设计
2.1 功能
默认情况下，集合通信在默认组（也称为world）上运行，并要求所有进程都进入分布式函数调用。但是，一些工作可以从更细粒度的通信中受益。这就是分布式组发挥作用的地方。new_group() 函数可用于创建一个新分布式组，这个新组是所有进程的任意子集。new_group() 返回一个不透明的组句柄，此句柄可以作为group参数提供给所有集合函数（集合函数是分布式函数，用于在某些编程模式中交换信息）。

2.2 本质
抛开概念，从代码看其本质。进程组就是给每一个训练的 process 建立一个通信thread。主线程（计算线程）在前台进行训练，这个通信 thread 在后台做通信。我们以 ProcessGroupMPI 为例，是在通信线程之中另外添加了一个 queue，做buffer 和 异步处理。这样，进程组中所有进程都可以组成一个集体在后台进行集合通信操作。

比如下面，左侧worker之中就有两个线程，计算线程负责计算梯度，然后要求通信线程与其它worker进行交换梯度。

+---------------------------------------------------------------+        +--------------+
| Worker Process                                                |        | Other Worker |
|                                                               |        |              |
|  +----------------------+       +-----------------------+     |        | +----------+ |
|  | Computation thread   |       | Communication thread  |     |        | |   Comm   | |
|  |                      |       |                       |     |        | |  thread  | |
|  |                      |       |                       |     |        | |          | |
|  |     Main Thread      |       |    workerThread_      |     |        | |          | |
|  |                      |       |                       |     |        | |          | |
|  |                      |       |                       |     |        | |          | |
|  | Gradient computation |       |                       |     |        | |          | |
|  |          +           |       |                       |     |        | |          | |
|  |          |           |       |                       |   + |    +   | |          | |
|  |          |           |       |                       |  /| |    |\  | |          | |
|  |          v           | /|_|\ |                       | / +-+----+ \ | |          | |
|  |    Does All+Reduce   |/ grad\|   Does communication  |/  Gradient  \| |          | |
|  |                      |\  _  /|                       |\            /| |          | |
|  |                      | \| |/ |                       | \ +-+----+ / | |          | |
|  |                      |       |                       |  \| |    |/  | |          | |
|  |                      |       |                       |   + |    +   | |          | |
|  |                      |       |                       |     |        | |          | |
|  |                      |       |                       |     |        | |          | |
|  +----------------------+       +-----------------------+     |        | +----------+ |
|                                                               |        |              |
+---------------------------------------------------------------+        +--------------+


::

    import os
    import sys
    import tempfile
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim

    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(24,24)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(24,12)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    def run_demo(demo_fn, world_size):
        mp.spawn(
            demo_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True)

    def demo_serialization(rank, world_size):
        setup(rank, world_size)

        CHECKPOINT = tempfile.gettempdir() + "/checkpoint.pt"

        model = SimpleModel().to(rank)
        ddp_model = DistributedDataParallel(model, device_ids=[rank])

        powersgd_hook = powerSGD.powerSGD_hook
        powersgd_state = powerSGD.PowerSGDState(process_group=None)

        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

        state = {
            'state_dict': ddp_model.state_dict(),
            'comm_hook': hook,
            'comm_hook_state': hook_state}

        if rank == 0:
            torch.save(state, CHECKPOINT)

        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(CHECKPOINT, map_location=map_location)

        ddp_model.load_state_dict(checkpoint['state_dict'])
        powersgd_hook = checkpoint['comm_hook']
        powersgd_state = checkpoint['comm_hook_state']

        ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

        if rank == 0:
            os.remove(CHECKPOINT)

        cleanup()

    if __name__ == "__main__":
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(demo_serialization, world_size)

Acknowledgements
----------------

Many thanks to PowerSGD paper author **Thijs Vogels** for the code review on
PowerSGD communication hook, as well as the
`comparison experiments <https://observablehq.com/@tvogels/powersgd-benchmark>`_,
which show that the performance of PowerSGD communication hook is on par with
the implementation in the original `paper <https://arxiv.org/abs/1905.13727>`_.
