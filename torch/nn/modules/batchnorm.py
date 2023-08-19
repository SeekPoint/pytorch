from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer

from .. import functional as F
from .. import init
from ._functions import SyncBatchNorm as sync_batch_norm
from .lazy import LazyModuleMixin
from .module import Module

__all__ = ['BatchNorm1d', 'LazyBatchNorm1d', 'BatchNorm2d', 'LazyBatchNorm2d', 'BatchNorm3d',
           'LazyBatchNorm3d', 'SyncBatchNorm']

'''

1. BatchNorm 原理

有图！
BatchNorm 最早在全连接网络中被提出，对每个神经元的输入做归一化。扩展到 CNN 中，就是对每个卷积核的输入做归一化，或者说在 channel 之外的所有维度做归一化。 BN 带来的好处有很多，这里简单列举几个：

防止过拟合：单个样本的输出依赖于整个 mini-batch，防止对某个样本过拟合；
加快收敛：梯度下降过程中，每一层的 
 和 
 都会不断变化，导致输出结果的分布在不断变化，后层网络就要不停地去适应这种分布变化。用 BN 后，可以使每一层输入的分布近似不变。
防止梯度弥散：forward 过程中，逐渐往非线性函数的取值区间的上下限两端靠近，（以 Sigmoid 为例），此时后面层的梯度变得非常小，不利于训练。
BN 的数学表达为： 

这里引入了缩放因子 
 和平移因子 
 ，作者在文章里解释了它们的作用：

Normalize 到 
 , 
 会导致新的分布丧失从前层传递过来的特征与知识
以 Sigmoid 为例，加入 
 , 
 可以防止大部分值落在近似线性的中间部分，导致无法利用非线性的部分
 

2. BatchNorm 的 PyTorch 实现
PyTorch 中与 BN 相关的几个类放在 torch.nn.modules.batchnorm 中，包含以下几个类：

_NormBase：nn.Module 的子类，定义了 BN 中的一系列属性与初始化、读数据的方法；
_BatchNorm：_NormBase 的子类，定义了 forward 方法；
BatchNorm1d & BatchNorm2d & BatchNorm3d：_BatchNorm的子类，定义了不同的_check_input_dim方法。
2.1 _NormBase 类
2.1.1 初始化
_NormBase类定义了 BN 相关的一些属性，如下表所示：

attribute	meaning
num_features	输入的 channel 数
track_running_stats	默认为 True，是否统计 running_mean，running_var
running_mean	训练时统计输入的 mean，之后用于 inference
running_var	训练时统计输入的 var，之后用于 inference
momentum	默认 0.1，更新 running_mean，running_var 时的动量
num_batches_tracked	PyTorch 0.4 后新加入，当 momentum 设置为 None 时，使用 num_batches_tracked 计算每一轮更新的动量
affine	默认为 True，训练 weight 和 bias；否则不更新它们的值
weight	公式中的 \gamma，初始化为全 1 tensor
bias	公式中的 \beta，初始化为全 0 tensor

'''
class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    # 读checkpoint时会用version来区分是 PyTorch 0.4.1 之前还是之后的版本

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # 如果打开 affine，就使用缩放因子和平移因子
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # 训练时是否需要统计 mean 和 variance
        if self.track_running_stats:
            # buffer 不会在self.parameters()中出现
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        # 具体在 BN1d, BN2d, BN3d 中实现，验证输入合法性
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        '''
        1.4.3 _load_from_state_dict 妙用

        Example: 避免 BC-breaking
        在模型迭代的过程中，module 很容易出现 BC-breaking ，
        PyTorch 通过 _version 和 _load_from_state_dict 来处理的这类问题（这也是 PyTorch 推荐的方式）。 
        下面的代码是 _NormBase 类避免 BC-breaking 的方式。在 PyTorch 的开发过程中，N
        ormalization layers 在某个新版本中 引入了 num_batches_tracked 这个 key，给 BN 记录训练过程中经历的 batch 数，
        为了兼容旧版本训练的模型，PyTorch 修改了 _version，并修改了 _load_from_state_dict。
        
        
        这里再举一个 MMCV 中的例子，DCN 经历了一次重构，属性的名字经过了重命名。

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            version = local_metadata.get('version', None)
            if version is None or version < 2:
                # the key is different in early versions
                # In version < 2, DeformConvPack loads previous benchmark models.
                if (prefix + 'conv_offset.weight' not in state_dict
                        and prefix[:-1] + '_offset.weight' in state_dict):
                    state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                        prefix[:-1] + '_offset.weight')
                if (prefix + 'conv_offset.bias' not in state_dict
                        and prefix[:-1] + '_offset.bias' in state_dict):
                    state_dict[prefix +
                               'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                    '_offset.bias')
            if version is not None and version > 1:
                print_log(
                    f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                    'version 2.',
                    logger='root')
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys, unexpected_keys,
                                          error_msgs)
        Example: 模型无痛迁移
        如果在 MMDetection 中训练了一个 detector，MMDetection3D 中的多模态检测器想要加载这个预训练的检测器，很多权重名字对不上，又不想写一个脚本手动来转，可以使用 _load_from_state_dict 来进行。通过这种方式，MMDetection3D 可以加载并使用 MMDetection 训练的任意一个检测器。
        
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            # override the _load_from_state_dict function
            # convert the backbone weights pre-trained in Mask R-CNN
            # use list(state_dict.keys()) to avoid
            # RuntimeError: OrderedDict mutated during iteration
            for key_name in list(state_dict.keys()):
                key_changed = True
                if key_name.startswith('backbone.'):
                    new_key_name = f'img_backbone{key_name[8:]}'
                elif key_name.startswith('neck.'):
                    new_key_name = f'img_neck{key_name[4:]}'
                elif key_name.startswith('rpn_head.'):
                    new_key_name = f'img_rpn_head{key_name[8:]}'
                elif key_name.startswith('roi_head.'):
                    new_key_name = f'img_roi_head{key_name[8:]}'
                else:
                    key_changed = False
                if key_changed:
                    logger = get_root_logger()
                    print_log(
                        f'{key_name} renamed to be {new_key_name}', logger=logger)
                    state_dict[new_key_name] = state_dict.pop(key_name)
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys, unexpected_keys,
                                          error_msgs)
                                                  
        '''
        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                # 旧版本的checkpoint没有这个key，设置为0
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
    '''
    3.1 forward
    复习一下方差的计算方式： 
    
    单卡上的 BN 会计算该卡对应输入的均值、方差，然后做 Normalize；SyncBN 则需要得到全局的统计量，也就是“所有卡上的输入”对应的均值、方差。一个简单的想法是分两个步骤：
    
    每张卡单独计算其均值，然后做一次同步，得到全局均值
    用全局均值去算每张卡对应的方差，然后做一次同步，得到全局方差
    但两次同步会消耗更多时间，事实上一次同步就可以实现 
     和 
     的计算：
    
    
    只需要在同步时算好 
     和 
     即可。这里用一张图来描述这一过程。
    
    
    实现时，batchnorm.SyncBatchNorm 根据自身的超参设置、train/eval 等设置参数，并调用_functions.SyncBatchNorm，接口是def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size): 首先算一下单卡上的均值和方差：
    
    # 这里直接算invstd，也就是 1/(sqrt(var+eps))
    mean, invstd = torch.batch_norm_stats(input, eps)
    然后同步各卡的数据，得到mean_all和invstd_all，再算出全局的统计量，更新running_mean，running_var:
    
    # 计算全局的mean和invstd
    mean, invstd = torch.batch_norm_gather_stats_with_counts(
    '''
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # 如果在train状态且self.track_running_stats被设置为True，就需要更新统计量
        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                # 如果momentum被设置为None，就用num_batches_tracked来加权
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class _LazyNormBase(LazyModuleMixin, _NormBase):

    weight: UninitializedParameter  # type: ignore[assignment]
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            # affine and track_running_stats are hardcoded to False to
            # avoid creating tensors that will soon be overwritten.
            0,
            eps,
            momentum,
            False,
            False,
            **factory_kwargs,
        )
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = UninitializedParameter(**factory_kwargs)
            self.bias = UninitializedParameter(**factory_kwargs)
        if self.track_running_stats:
            self.running_mean = UninitializedBuffer(**factory_kwargs)
            self.running_var = UninitializedBuffer(**factory_kwargs)
            self.num_batches_tracked = torch.tensor(
                0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            self.num_features = input.shape[1]
            if self.affine:
                assert isinstance(self.weight, UninitializedParameter)
                assert isinstance(self.bias, UninitializedParameter)
                self.weight.materialize((self.num_features,))
                self.bias.materialize((self.num_features,))
            if self.track_running_stats:
                self.running_mean.materialize((self.num_features,))  # type:ignore[union-attr]
                self.running_var.materialize((self.num_features,))  # type:ignore[union-attr]
            self.reset_parameters()


class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0. The
    standard-deviation is calculated via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`, where :math:`N` is the batch size,
          :math:`C` is the number of features or channels, and :math:`L` is the sequence length
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm1d` module with lazy initialization of
    the ``num_features`` argument of the :class:`BatchNorm1d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    cls_to_become = BatchNorm1d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    '''
    2.2 BatchNormNd 类
    包括BatchNorm1d，BatchNorm2d，BatchNorm3d。区别只是检查了输入的合法性，这里简单贴一下BatchNorm2d的实现：
    '''
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm2d` module with lazy initialization of
    the ``num_features`` argument of the :class:`BatchNorm2d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    cls_to_become = BatchNorm2d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm3d` module with lazy initialization of
    the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    cls_to_become = BatchNorm3d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))

'''
3. SyncBatchNorm 的 PyTorch 实现
BN 的性能和 batch size 有很大的关系。batch size 越大，BN 的统计量也会越准。然而像检测这样的任务，占用显存较高，一张显卡往往只能拿较少的图片（比如 2 张）来训练，这就导致 BN 的表现变差。一个解决方式是 SyncBN：所有卡共享同一个 BN，得到全局的统计量。

PyTorch 的 SyncBN 分别在 torch/nn/modules/batchnorm.py 和 torch/nn/modules/_functions.py 做了实现。前者主要负责检查输入合法性，以及根据momentum等设置进行传参，调用后者。后者负责计算单卡统计量以及进程间通信。
'''
class SyncBatchNorm(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.

    Currently :class:`SyncBatchNorm` only supports
    :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    .. note::
        Synchronization of batchnorm statistics occurs only while training, i.e.
        synchronization is disabled when ``model.eval()`` is set or if
        ``self.training`` is otherwise ``False``.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))
        >>> r1, r2 = ranks[:4], ranks[4:]
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
        >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Optional[Any] = None,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        #   # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
        # under supported condition (single GPU per process)
        self.process_group = process_group

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.dim())
            )

    def _check_non_zero_input_channels(self, input):
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        # 接下来这部分与普通BN差别不大
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        ## 如果在train模式下，或者关闭track_running_stats，就需要同步全局的均值和方差
        need_sync = (bn_training and self.training and
                     torch.distributed.is_available() and torch.distributed.is_initialized())
        if need_sync:
            # currently only GPU input is supported
            if not input.is_cuda:
                raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        # 如果不需要同步，SyncBN的行为就与普通BN一致
        if not need_sync:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            return sync_batch_norm.apply(
                input,
                self.weight,
                self.bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,
                world_size,
            )

    # 把普通BN转为SyncBN, 主要做一些参数拷贝
    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`torch.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> # xdoctest: +SKIP("distributed")
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output


'''
2.1.2 模拟 BN forward
PyTorch 中 BN 的 Python 部分代码主要实现初始化、传参和底层方法调用。这里用 Python 模拟 BN 的底层计算。

import torch
import torch.nn as nn
import torch.nn.modules.batchnorm

# 创建随机输入
def create_inputs():
    return torch.randn(8, 3, 20, 20)

# 以 BatchNorm2d 为例
# mean_val, var_val 不为None时，不对输入进行统计，而直接用传进来的均值、方差
def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        # 这里需要注意，torch.var 默认算无偏估计，因此需要手动设置unbiased=False
        var_val = x.var([0, 2, 3], unbiased=False)

    x = x - mean_val[None, ..., None, None]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x
验证 dummy BN 输出的正确性：

bn_layer = nn.BatchNorm2d(num_features=3)
inputs = create_inputs()
# 用 pytorch 的实现 forward 
bn_outputs = bn_layer(inputs)
# 用 dummy bn 来 forward
_, _, expected_outputs = dummy_bn_forward(
    inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps)
assert torch.allclose(expected_outputs, bn_outputs)


2.1.3 running_mean、running_var 的更新
BatchNorm 默认打开 track_running_stats，因此每次 forward 时都会依据当前 minibatch 的统计量来更新 running_mean 和 running_var。

momentum 默认值为 0.1，控制历史统计量与当前 minibatch 在更新 running_mean、running_var 时的相对影响。



其中 
 、
 分别表示 
 的均值、方差；需要注意这里统计方差时用了无偏估计，与论文保持一致。手动对这一过程进行模拟，如下所示：

running_mean = torch.zeros(3)
running_var = torch.ones_like(running_mean)
momentum = 0.1 # 这也是BN初始化时momentum默认值
bn_layer = nn.BatchNorm2d(num_features=3, momentum=momentum)

# 模拟 forward 10 次
for t in range(10):
    inputs = create_inputs()
    bn_outputs = bn_layer(inputs)
    inputs_mean, inputs_var, _ = dummy_bn_forward(
        inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps
    )
    n = inputs.numel() / inputs.size(1)
    # 更新 running_var 和 running_mean
    running_var = running_var * (1 - momentum) + momentum * inputs_var * n / (n - 1)
    running_mean = running_mean * (1 - momentum) + momentum * inputs_mean

assert torch.allclose(running_var, bn_layer.running_var)
assert torch.allclose(running_mean, bn_layer.running_mean)
print(f'bn_layer running_mean is {bn_layer.running_mean}')
print(f'dummy bn running_mean is {running_mean}')
print(f'bn_layer running_var is {bn_layer.running_var}')
print(f'dummy bn running_var is {running_var}')
输出结果：

bn_layer running_mean is tensor([ 0.0101, -0.0013, 0.0101])
dummy bn running_mean is tensor([ 0.0101, -0.0013, 0.0101])
bn_layer running_var is tensor([0.9857, 0.9883, 1.0205])
dummy bn running_var is tensor([0.9857, 0.9883, 1.0205])
running_mean 的初始值为 0，forward 后发生变化。同时模拟 BN 的running_mean,running_var 也与 PyTorch 实现的结果一致。

以上讨论的是使用momentum的情况。在 PyTorch 0.4.1 后，加入了num_batches_tracked属性，统计 BN 一共 forward 了多少个 minibatch。当momentum被设置为None时，就由num_batches_tracked来控制历史统计量与当前 minibatch 的影响占比：




接下来手动模拟这一过程：

running_mean = torch.zeros(3)
running_var = torch.ones_like(running_mean)
num_batches_tracked = 0
# momentum 设置成 None，用 num_batches_tracked 来更新统计量
bn_layer = nn.BatchNorm2d(num_features=3, momentum=None)

# 同样是模拟 forward 10次
for t in range(10):
    inputs = create_inputs()
    bn_outputs = bn_layer(inputs)
    inputs_mean, inputs_var, _ = dummy_bn_forward(
        inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps
    )
    num_batches_tracked += 1
    # exponential_average_factor
    eaf = 1.0 / num_batches_tracked
    n = inputs.numel() / inputs.size(1)
    # 更新 running_var 和 running_mean
    running_var = running_var * (1 - eaf) + eaf * inputs_var * n / (n - 1)
    running_mean = running_mean * (1 - eaf) + eaf * inputs_mean

assert torch.allclose(running_var, bn_layer.running_var)
assert torch.allclose(running_mean, bn_layer.running_mean)

bn_layer.train(mode=False)
inference_inputs = create_inputs()
bn_outputs = bn_layer(inference_inputs)
_, _, dummy_outputs = dummy_bn_forward(
    inference_inputs, bn_layer.weight,
    bn_layer.bias, bn_layer.eps,
    running_mean, running_var)
assert torch.allclose(dummy_outputs, bn_outputs)
print(f'bn_layer running_mean is {bn_layer.running_mean}')
print(f'dummy bn running_mean is {running_mean}')
print(f'bn_layer running_var is {bn_layer.running_var}')
print(f'dummy bn running_var is {running_var}')
输出：

bn_layer running_mean is tensor([-0.0040, 0.0074, -0.0162])
dummy bn running_mean is tensor([-0.0040, 0.0074, -0.0162])
bn_layer running_var is tensor([1.0097, 1.0086, 0.9815])
dummy bn running_var is tensor([1.0097, 1.0086, 0.9815])
手动模拟的结果与 PyTorch 相同。


2.1.4 
 , 
 的更新
BatchNorm 的 weight，bias 分别对应公式里的 
 , 
 , 更新方式是梯度下降法。

import torchvision
from torchvision.transforms import Normalize, ToTensor, Compose
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

# 用 mnist 作为 toy dataset
mnist = torchvision.datasets.MNIST(root='mnist', download=True, transform=ToTensor())
dataloader = DataLoader(dataset=mnist, batch_size=8)

# 初始化一个带 BN 的简单模型
toy_model = nn.Sequential(nn.Linear(28 ** 2, 128), nn.BatchNorm1d(128),
                          nn.ReLU(), nn.Linear(128, 10), nn.Sigmoid())
optimizer = torch.optim.SGD(toy_model.parameters(), lr=0.1)

bn_1d_layer = toy_model[1]
print(f'Initial weight is {bn_layer.weight[:4].tolist()}...')
print(f'Initial bias is {bn_layer.bias[:4].tolist()}...\n')
# 模拟更新2次参数
for (i, data) in enumerate(dataloader):
    output = toy_model(data[0].view(data[0].shape[0], -1))
    (F.cross_entropy(output, data[1])).backward()
    # 输出部分参数的梯度，验证weight和bias确实是通过gradient descent更新的
    print(f'Gradient of weight is {bn_1d_layer.weight.grad[:4].tolist()}...')
    print(f'Gradient of bias is {bn_1d_layer.bias.grad[:4].tolist()}...')
    optimizer.step()
    optimizer.zero_grad()
    if i == 1:
        break
print(f'\nNow weight is {bn_1d_layer.weight[:4].tolist()}...')
print(f'Now bias is {bn_1d_layer.bias[:4].tolist()}...')

inputs = torch.randn(4, 128)
bn_outputs = bn_1d_layer(inputs)
new_bn = nn.BatchNorm1d(128)
bn_outputs_no_weight_bias = new_bn(inputs)

assert not torch.allclose(bn_outputs, bn_outputs_no_weight_bias)
输出：

Initial weight is [0.9999354481697083, 1.0033478736877441, 1.0019147396087646, 0.9986106157302856]...
Initial bias is [-0.0012734815245494246, 0.001349383033812046, 0.0013358002761378884, -0.0007148777367547154]...

Gradient of weight is [-0.0004475426103454083, -0.0021388232707977295, -0.0032624618615955114, -0.0009599098702892661]...
Gradient of bias is [0.00011698803427862003, -0.001291472464799881, -0.0023048489820212126, -0.0009493136312812567]...
Gradient of weight is [-0.00035325769567862153, -0.0014295700239017606, -0.002102235099300742, 0.000851186050567776]...
Gradient of bias is [-0.00026844028616324067, -0.00025666248984634876, -0.0017800561618059874, 0.00024933076929301023]...

Now weight is [1.0000154972076416, 1.0037046670913696, 1.0024511814117432, 0.9986214637756348]...
Now bias is [-0.0012583363568410277, 0.0015041964361444116, 0.0017442908138036728, -0.0006448794738389552]...
2.1.5 eval 模式
上面验证的都是 train 模式下 BN 的表现，eval 模式有几个重要的参数。

track_running_stats默认为True，train 模式下统计running_mean和running_var，eval 模式下用统计数据作为 
 和 
 。设置为False时，eval模式直接计算输入的均值和方差。
running_mean、running_var：train 模式下的统计量。
也就是说，BN.training 并不是决定 BN 行为的唯一参数。满足BN.training or not BN.track_running_stats就会直接计算输入数据的均值方差，否则用统计量代替。

# 切换到eval模式
bn_layer.train(mode=False)
inference_inputs = create_inputs()
# 输出前后的 running_mean 和 running_var，验证eval模式下不再更新统计量
print(f'bn_layer running_mean is {bn_layer.running_mean}')
print(f'bn_layer running_var is {bn_layer.running_var}')
bn_outputs = bn_layer(inference_inputs)
print(f'Now bn_layer running_mean is {bn_layer.running_mean}')
print(f'Now bn_layer running_var is {bn_layer.running_var}')
# 用之前统计的running_mean和running_var替代输入的running_mean和running_var
_, _, dummy_outputs = dummy_bn_forward(
    inference_inputs, bn_layer.weight,
    bn_layer.bias, bn_layer.eps,
    running_mean, running_var)
assert torch.allclose(dummy_outputs, bn_outputs)

# 关闭track_running_stats后，即使在eval模式下，也会去计算输入的mean和var
bn_layer.track_running_stats = False
bn_outputs_notrack = bn_layer(inference_inputs)
_, _, dummy_outputs_notrack = dummy_bn_forward(
    inference_inputs, bn_layer.weight,
    bn_layer.bias, bn_layer.eps)

assert torch.allclose(dummy_outputs_notrack, bn_outputs_notrack)
assert not torch.allclose(bn_outputs, bn_outputs_notrack)
输出结果如下：

bn_layer running_mean is tensor([-0.0143,  0.0089, -0.0062])
bn_layer running_var is tensor([0.9611, 1.0380, 1.0181])
Now bn_layer running_mean is tensor([-0.0143,  0.0089, -0.0062])
Now bn_layer running_var is tensor([0.9611, 1.0380, 1.0181])

'''