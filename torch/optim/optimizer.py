from collections import OrderedDict, defaultdict, abc as container_abcs
import torch
from copy import deepcopy
from itertools import chain
import warnings
import functools
import math

from typing import Callable, Dict, List, Tuple

import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch._utils import is_compiling

__all__ = ['Optimizer', 'register_optimizer_step_pre_hook', 'register_optimizer_step_post_hook']
_global_optimizer_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_optimizer_post_hooks: Dict[int, Callable] = OrderedDict()
_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]

class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad

def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x.item()

def _stack_if_compiling(x):
    if not torch.jit.is_scripting() and is_compiling():
        return torch.stack(x)
    else:
        return x

def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)

# For any optimizer with a faster implementation, we attempt to default to the
# fastest + stablest whenever possible. For foreach, the requirements are to have
# native params all on CUDA. For fused, there's currently the additional requirement
# that the tensors' dtypes must be floating point. Neither alternative supports
# torch.jit.script nor differentiable, so we fall back to the single tensor
# implementation in those cases.
def _default_to_fused_or_foreach(params: List[torch.Tensor],
                                 differentiable: bool,
                                 use_fused: bool = False) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False
    fused = use_fused and all(
        p is None or (type(p) in _foreach_supported_types and p.is_cuda and torch.is_floating_point(p)) for p in params
    )
    foreach = not fused and all(
        p is None or (type(p) in _foreach_supported_types and p.is_cuda) for p in params
    )
    return fused, foreach


# Common doc strings among optimizers
_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. (default: None)"""

_fused_doc = r"""fused (bool, optional): whether the fused implementation (CUDA only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None)

    .. note:: The foreach and fused implementations are typically faster than the for-loop,
              single-tensor implementation. Thus, if the user has not specified BOTH flags
              (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
              implementation when the tensors are all on CUDA. For example, if the user specifies
              True for fused but nothing for foreach, we will run the fused implementation. If
              the user specifies False for foreach but nothing for fused (or False for fused but
              nothing for foreach), we will run the for-loop implementation. If the user specifies
              True for both foreach and fused, we will prioritize fused over foreach, as it is
              typically faster. We attempt to use the fastest, so the hierarchy goes fused ->
              foreach -> for-loop. HOWEVER, since the fused implementation is relatively new,
              we want to give it sufficient bake-in time, so we default to foreach and NOT
              fused when the user has not specified either flag."""

_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""

_maximize_doc = r"""maximize (bool, optional): maximize the params based on the
            objective, instead of minimizing (default: False)"""


def register_optimizer_step_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers. The hook should have the following
    signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemoveableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a post hook common to all optimizers. The hook should have the following
    signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemoveableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle


class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    '''
    Optimizer 是所有优化器的基类，它有如下主要公共方法:
        add_param_group : 添加可学习参数组。
        step : 进行一次参数更新操作。
        zero_grad : 在反向传播计算梯度之前对上一次迭代时的梯度清零。
        state_dict : 返回用 dict 结构表示的参数和状态。
        load_state_dict : 加载 dict 结构表示的参数和状态。
    2.1 初始化
    在 Optimizer 初始化函数之中，会做如下操作：
        初始化参数包括：可学习参数（params）和超参数（defaults）。
        在 self.defaults 之中保存 lr, momentun 等全局参数（超参数）。
        在 self.state 保存优化器当前状态。
        在 self.param_groups 之中保存所有待优化的变量。
        
    '''
    def __init__(self, params, defaults):
        torch._C._log_api_usage_once("python.optimizer")

        # 字典类型，子类传入，用于表示全部参数组的默认超参
        self.defaults = defaults  # 保存 lr, momentun 等全局参数
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

        self._patch_step_function()

        if isinstance(params, torch.Tensor): # params必须是字典或者tensors
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)  # 保存优化器当前状态
        self.param_groups = [] # 所有待优化的参数，其每一项是一个字典，对应一组待优化参数和其他相关参数

        param_groups = list(params) # 需要被优化的变量，是__init__ 传入的参数
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            # 将参数转换为字典
            param_groups = [{'params': param_groups}] # param_groups 是一个列表，其中一项是字典形式，优化变量被保存在其中。

        for param_group in param_groups:
            self.add_param_group(param_group) # 把param_groups所有项都加到self.param_groups之中

        # Allows _cuda_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True


    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_optimizer_step_pre_hooks' not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if '_optimizer_step_post_hooks' not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        self.defaults.setdefault('differentiable', False)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    # Currently needed by Adam and AdamW
    def _cuda_graph_capture_health_check(self):
        if torch.has_cuda and torch.cuda.is_available():
            capturing = torch.cuda.is_current_stream_capturing()

            if capturing and not all(group['capturable'] for group in self.param_groups):
                raise RuntimeError("Attempting CUDA graph capture of step() for an instance of " +
                                   self.__class__.__name__ +
                                   " but param_groups' capturable is False.")

            if (
                (not getattr(self, "_warned_capturable_if_run_uncaptured", False))
                and all(group['capturable'] for group in self.param_groups)
                and (not capturing)
            ):
                warnings.warn(
                    "This instance was constructed with capturable=True or some of all the param_groups came with capturable=True, "
                    "but step() is running without CUDA graph capture. If you never intend to graph-capture this "
                    "instance, capturable=True can impair performance, and you should set capturable=False."
                )
                self._warned_capturable_if_run_uncaptured = True

    def _optimizer_step_code(self):
        """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """
        pass

    @staticmethod
    def profile_hook_step(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self, *_ = args
            profile_name = "Optimizer.step#{}.step".format(self.__class__.__name__)
            with torch.autograd.profiler.record_function(profile_name):
                # call optimizer step pre hooks
                for pre_hook in chain(_global_optimizer_pre_hooks.values(), self._optimizer_step_pre_hooks.values()):
                    result = pre_hook(self, args, kwargs)
                    if result is not None:
                        if isinstance(result, tuple) and len(result) == 2:
                            args, kwargs = result
                        else:
                            raise RuntimeError(f"{func} must return None or a tuple of (new_args, new_kwargs),"
                                               f"but got {result}.")

                out = func(*args, **kwargs)
                self._optimizer_step_code()

                # call optimizer step post hooks
                for post_hook in chain(self._optimizer_step_post_hooks.values(), _global_optimizer_post_hooks.values()):
                    post_hook(self, args, kwargs)

                return out

        return wrapper

    def _patch_step_function(self):
        self._zero_grad_profile_name = "Optimizer.zero_grad#{}.zero_grad".format(self.__class__.__name__)
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = self.profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True

    def register_step_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""Register an optimizer step pre hook which will be called before
        optimizer step. It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_pre_hooks)
        self._optimizer_step_pre_hooks[handle.id] = hook
        return handle

    def register_step_post_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""Register an optimizer step post hook which will be called after optimizer step.
        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_post_hooks)
        self._optimizer_step_post_hooks[handle.id] = hook
        return handle

    '''
    PyTorch 的 state_dict 是 Python 的字典对象。

    对于模型，state_dict 会把每一层和其训练过程中需要学习的参数（比如权重和偏置）建立起来映射关系，只有参数可以训练的layer才会保存在模型的 state_dict 之中，如卷积层，线性层等。
    
    对于优化器，state_dict 是其状态信息，其包括了两组信息：
    
        state ：一个包括了优化器当前状态（也就是更新变量的过程之中计算得到的最新缓存变量）的字典。
            字典的 key 是缓存的index。
            字典的 value 也是一个字典，key 是缓存变量名，value 是相应的张量。
        param_groups : 一个包括了所有 param groups 的字典。
        
    1.2.5 state_dict() 和 load_state_dict
    这两个方法实现序列化和反序列化功能。
    
        state_dict(): 将优化器管理的参数和其状态信息以 dict 形式返回
        load_state_dict(state_dict): 加载之前返回的 dict，更新参数和其状态
        两个方法可用来实现模型训练中断后继续训练功能
    '''
    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            # 'params'采用不同规则
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            # 保存了参数的id，而并非参数的值
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        # 对self.param_groups进行遍历，进行pack
        param_groups = [pack_group(g) for g in self.param_groups]

        # 将state中的所有Tensor替换为相应的 use order indices
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,  # 状态
            'param_groups': param_groups,  # 待优化的参数
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if (key != "step"):
                    if param.is_floating_point():
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    '''
    1.2.4 zero_grad
    在反向传播计算梯度之前对上一次迭代时记录的梯度清零，参数set_to_none 设置为 True 时会直接将参数梯度设置为 None，
    从而减小内存使用, 但通常情况下不建议设置这个参数，因为梯度设置为 None 和 0 在 PyTorch 中处理逻辑会不一样。
    '''
    def zero_grad(self, set_to_none: bool = True):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get('foreach', False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if (not foreach or p.grad.is_sparse):
                                p.grad.zero_()
                            else:
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            if foreach:
                for _, per_dtype_grads in per_device_and_dtype_grads.items():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    #基类 Optimizer 定义了 step 方法接口，如下所示
    #子类如 SGD 需要实现 step 方法
    '''
    step 方法可传入闭包函数 closure，主要目的是为了实现如Conjugate Gradient和LBFGS等优化算法，
    这些算法需要对模型进行多次评估
    Python 中闭包概念：在一个内部函数中，对外部作用域的变量进行引用(并且一般外部函数的返回值为内部函数)，
    那么内部函数就被认为是闭包
    '''
    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError
    '''
    add_param_group 添加不同分组的可学习参数。代码如下（省略了大部分检验代码）。
    其中，param_groups目的是为了可以用 key-value 方式来访问待优化变量，这在fine tuning时候特别有用。
    '''
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params'] # 得到待优化的变量
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params] # 构建一个列表，其中就是待优化的变量
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        # 省略校验，比如必须是tensor类型，而且是叶子节点

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not self.defaults.get('differentiable', None) and not (param.is_leaf or param.retains_grad):
                raise ValueError("can't optimize a non-leaf Tensor")

        # 利用默认参数给所有组设置统一的超参
        for name, default in self.defaults.items():  # 缺省参数也加入到 param_group 之中
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)  # 所有组都设置同样的缺省参数（超参数）

        # 用set来去重
        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        # 更新自身的参数组中
        #利用 add_param_group 函数功能，可以对模型不同的可学习参数组设定不同的超参数，初始化优化器可传入元素是 dict 的 list，每个 dict 中的 key 是 params 或者其他超参数的名字如 lr，
        self.param_groups.append(param_group) # 加入到param_groups
