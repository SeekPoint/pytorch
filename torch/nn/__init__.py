from .modules import *  # noqa: F403
from .parameter import (
    Parameter as Parameter,
    UninitializedParameter as UninitializedParameter,
    UninitializedBuffer as UninitializedBuffer,
)
from .parallel import DataParallel as DataParallel
from . import init
from . import functional
from . import utils


def factory_kwargs(kwargs):
    r"""
    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like torch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = torch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(torch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # guarantee no input kwargs is untouched
    r = dict(kwargs.get("factory_kwargs", {}))
    for k in simple_keys:
        if k in kwargs:
            if k in r:
                raise TypeError(f"{k} specified twice, in **kwargs and in factory_kwargs")
            r[k] = kwargs[k]

    return r


'''

PyTorch 源码解读之 nn.Module：核心网络模块接口详解
https://zhuanlan.zhihu.com/p/340453841

目录
0 设计

1 nn.Module 实现

1.1 常用接口

1.1.1 __init__ 函数

1.1.2 状态的转换

1.1.3 参数的转换或转移

1.1.4 Apply 函数

1.2 属性的增删改查

1.2.1 属性设置

1.2.2 属性删除

1.2.3 常见的属性访问

1.3 Forward & Backward

1.3.1 Hooks

1.3.2 运行逻辑

1.4 模块存取

1.4.1 Hooks

1.4.2 功能实现

1.4.3 _load_from_state_dict 妙用



本次解读主要介绍 PyTorch 中的神经网络模块，即 torch.nn，其中主要介绍 nn.Module，其他模块的细节可以通过 PyTorch 的 API 文档进行查阅，一些较重要的模块如 DataParallel 和 BN/SyncBN 等，都有独立的文章进行介绍。

0 设计
nn.Module 其实是 PyTorch 体系下所有神经网络模块的基类，此处顺带梳理了一下 torch.nn 中的各个组件，他们的关系概览如下图所示。

图！！！

展开各模块后，模块之间的继承关系与层次结构如下图所示：

图！！！

从各模块的继承关系来看，模块的组织和实现有几个常见的特点，供 PyTorch 代码库的开发者参考借鉴：

一般有一个基类来定义接口，通过继承来处理不同维度的 input，如：
Conv1d，Conv2d，Conv3d，ConvTransposeNd 继承自 _ConvNd
MaxPool1d，MaxPool2d，MaxPool3d 继承自 _MaxPoolNd 等
每一个类都有一个对应的 nn.functional 函数，类定义了所需要的 arguments 和模块的 parameters，在 forward 函数中将 arguments 和 parameters 传给 nn.functional 的对应函数来实现 forward 功能。比如：
所有的非线性激活函数，都是在 forward 中直接调用对应的 nn.functional 函数
Normalization 层都是调用的如 F.layer_norm， F.group_norm 等函数
继承 nn.Module 的模块主要重载 init、 forward、 和 extra_repr 函数，含有 parameters 的模块还会实现 reset_parameters 函数来初始化参数

'''