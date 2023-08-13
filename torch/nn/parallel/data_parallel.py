import operator
import torch
import warnings
from itertools import chain
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)

__all__ = ['DataParallel', 'data_parallel']
'''
虽然输入数据是均等划分并且并行分配，但是output loss每次都会在第一块GPU聚合相加计算，所以第一块GPU的内存负载和使用率会大于其他显卡。

_check_balance 函数会检查负载是否平衡， 如果内存或者处理器 max/min > 0.75 会有警告。
'''
def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids]
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.

    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> # xdoctest: +SKIP
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        '''
        __init__ 三个输入参数定义如下：

            module ： 模型，
            device_ids ：训练的device，
            output_device ：保存输出结果的device。默认是在device_ids[0]，即第一块卡。
        '''
        super().__init__()
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")

        # 得到可用的GPU
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        # 没有输入的情况下，使用所有可见的GPU
        if device_ids is None:
            device_ids = _get_all_device_indices()

        # 把GPU列表上第一个作为输出，也会作为master
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        # 检查负载均衡
        _check_balance(self.device_ids)

        # 单卡就直接使用
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    '''
    0x04 前向传播
        DataParallel并行计算只存在在前向传播过程之中。
        
        4.1 总述
        之前示例之中已经用 cuda() 函数来把模型放到 GPU[0] 之上，GPU[0] 这里已经有了模型的parameters 和 buffers。
        
        model=model.cuda() 
        所以forward函数之中，就不用作这一步，而是从分发模型和数据开始，需要注意的是：每次前向传播的时候都会分发模型。具体分为几个步骤。
        
            验证：遍历module的parameters和buffers，看看是否都在GPU[0]之上，如果不在，报错。
            分发（(Scatter）输入数据：将输入数据根据其第一个维度（一般是 batch 大小）划分多份，传送到多个 GPU；
            复制（Replicate）模型：将模型分别拷贝到多个 GPU；
            并行应用（parallel_apply）：在多个模型之上并行进行前向传播。因为 GPU device_ids[0] 和 base parallelized module 共享存储，所以在device[0] 上的 in-place 更新也会被保留下来，其他的GPU则不会。
            收集（Gather）：收集从多个 GPU 上传送回来的数据；
            
    目前，我们已经使用 Scatter 函数将数据从 device[0] 分配并复制到不同的卡，
    用 Replicate 函数将模型从 device[0] 复制到不同的卡，这样各个卡都有了同样的模型和不同的数据，然后分别调用 forward 计算损失和梯度。
    也就是 parallel_apply 部分。

    现在要做的就是把分布式计算的梯度合并到 device[0]，就是 self.output_device。
    '''
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            # 如果机器上没有GPU，则直接用CPU运行
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            # 遍历module的parameters和buffers，看看是否都在GPU[0]之上，如果不在，报错
            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))
            # 现在GPU[0]上有了模型，开始训练

            # 首先分发输入  分发数据
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)

            # 如果只有单卡，直接使用
            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **kwargs[0])

            # 分发模型
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

            # 并行训练
            outputs = self.parallel_apply(replicas, inputs, kwargs)

            # 把前向传播的结果收集到master
            # 收集到 devices[0]
            return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids): #replicate 只是转发，我们还需要接着看。
        return replicate(module, device_ids, not torch.is_grad_enabled())

    #scatter 实际就是 scatter_kwargs 的封装，所以我们直接看 scatter_kwargs。
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device(device_type, device_ids[0])

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    # for module without any inputs, empty list and dict will be created
    # so the module can be executed on one device which is the first one in device_ids
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
