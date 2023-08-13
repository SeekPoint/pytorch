from . import comm
from torch._utils import _get_device_index

from collections import OrderedDict


def _is_script_module(module):
    import torch.jit
    return isinstance(module, torch.jit.ScriptModule)


def _is_script_method(module):
    import torch.jit
    return isinstance(module, torch._C.ScriptMethod)


def _init_script_module():
    import torch.jit
    return torch.jit.ScriptModule()


def _is_jit_enabled():
    import torch.jit
    return torch.jit._state._enabled

#_replicatable_module 用来检查模型是否可以安全拷贝。
# Check if we can safely replicate the module.
# there are two types of module:
# 1. python modules
# 2. ScriptModule
#
# currently a module cannot be replicated properly if the descendants of
# any ScriptModule contains python module (type 1 above)
def _replicatable_module(module, memo=None):

    # module.modules() contains module itself as the first element
    def descendant_modules(module):
        gen = module.modules()
        next(gen)
        return gen

    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()

    # memoize visited modules
    memo.add(module)
    if _is_script_module(module):
        memo.update(descendant_modules(module))
        return all(_is_script_module(descendant) for
                   descendant in descendant_modules(module))

    for child in module.children():
        # since any unreplicatable module will cause the check to return
        # False early, visited modules here can be safely ignored.
        if child in memo:
            continue
        if not _replicatable_module(child, memo):
            return False

    return True

#拷贝参数都用到了_broadcast_coalesced_reshape。
def _broadcast_coalesced_reshape(tensors, devices, detach=False):
    from ._functions import Broadcast
    if detach:
        # 如果是detach，就直接调用
        return comm.broadcast_coalesced(tensors, devices)
    else:
        # Use the autograd function to broadcast if not detach
        if len(tensors) > 0:
            # 否则先用Broadcast过度一下，最后还是调用broadcast_coalesced
            tensor_copies = Broadcast.apply(devices, *tensors)
            return [tensor_copies[i:i + len(tensors)]
                    for i in range(0, len(tensor_copies), len(tensors))]
        else:
            return []

'''
replicate 具体逻辑是：

    使用 _replicatable_module 看看是否可以安全的复制模型。
    
    看看有多少个GPU，需要复制多少份。
    
    复制操作。
    
        复制 parameters。
            使用 _broadcast_coalesced_reshape 来把parameters拷贝到各个GPU。
        复制buffers。
            首先统计一下buffers。
            记录需要求导的 buffer 的 index。
            记录不需要求导的 buffer 的 index。
            对于两种buffers分别使用_broadcast_coalesced_reshape拷贝到各个GPU。
        复制模型。
            modules()返回一个包含当前模型所有模块的迭代器。转变成list，可以认为把模型打平了。
            遍历modules，往每个module_copies里面添加模型的每一层。
            最终，module_copies[j] 里面包含了模型的每一层，即module_copies[j][i] 就是模型的第 i 层。
    配置操作。
        
        就是配置模型网络，把GPU中数据的 reference 配置到 modules 数组的每一个module 之中，这样这些 module 就是完备模型了。
        因为之前是把嵌套的模型网络打散了分别拷贝到GPU：buffers和parameters也分别拷贝到了GPU。现在需要把它们重新配置到浅拷贝的模型之中，这样就把模型逻辑补齐了。
        遍历模型每个子模块，只配置需要的部分参数。
            处理 其子_modules_。
            处理 其_parameters。
            处理 其 _buffers。
    后续并行操作时候，每一个 worker 会得到 modules 数组的每一个module，就在这个 module 之上进行训练。

'''
def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    if not devices:
        return []

    # 看看有多少个GPU，需要复制多少份
    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)  # 复制这些份

    # 1）复制操作

    # 复制参数 parameters
    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}

    # 拷贝到各个GPU,我们随后会讲解_broadcast_coalesced_reshape
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    # 复制buffers
    # 首先统计一下buffers
    buffers = list(network.buffers())
    buffers_rg = [] # 需要求导的
    buffers_not_rg = [] # 不需要求导的
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    # 记录需要求导的 buffer 的 index
    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}

    # 记录不需要求导的 buffer 的 index
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    # 对于两种buffers分别拷贝到各个GPU
    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

    # 准备拷贝模型网络
    modules = list(network.modules())  # modules()返回一个包含当前模型所有模块的迭代器。转变成list，可以认为把模型打平了
    module_copies = [[] for device in devices] # 为各个GPU准备好空list
    module_indices = {}

    # 得到模型的浅拷贝列表
    for i, module in enumerate(modules): # 遍历模型 list
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()  # 获取浅拷贝
            # This is a temporary fix for DDP. DDP needs to access the
            # replicated model parameters. It used to do so through
            # `mode.parameters()`. The fix added in #33907 for DP stops the
            # `parameters()` API from exposing the replicated parameters.
            # Hence, we add a `_former_parameters` dict here to support DDP.
            replica._former_parameters = OrderedDict()

            module_copies[j].append(replica)  # 往每个module_copies里面添加模型的每一层
    # 最终，module_copies[j] 里面包含了模型的每一层，即module_copies[j][i] 就是模型的第 i 层

    # 2）配置操作

    # 这一步的目的是：把GPU中数据的reference赋值到浅拷贝之中，变成完备模型。
    # 因为之前是把嵌套的模型网络打散了分别拷贝到GPU，buffers和parameters也分别拷贝到了GPU，现在把他们构建到浅拷贝的模型之中，把模型逻辑补齐。
    for i, module in enumerate(modules): # 遍历模型每个子模块，只赋值需要的部分参数

        # 处理其子_modules
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]  # module_copies[j]是第j个模型拷贝
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]  # module_copies[j]是第j个模型拷贝
                    setattr(replica, key, module_copies[j][module_idx]) # 设置第j个模型的对应部分，下同

        # 处理_parameters
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param

        # 处理 _buffers
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [module_copies[j][0] for j in range(num_replicas)]
