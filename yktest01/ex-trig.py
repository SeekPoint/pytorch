
# 先看一个简单的例子（摘自官方文档：https: // pytorch.org / docs / master / dynamo / get - started.html，感觉官方文档的示例和后文的展开有冲突，因此做了一些修改），如果能正常运行，就代表你的预发布版本的


#
# https://cloud.tencent.com/developer/article/2203325
# PyTorch 2.0 之 Dynamo: 窥探加速背后的真相

import torch
from torch._dynamo import optimize


def fn(x):
    a = torch.sin(x).cuda()
    b = torch.sin(a).cuda()
    return b


new_fn = optimize("inductor")(fn)
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor, input_tensor)

# 在上例中，我们用torch.compile
# 依次运行了两个pointwise算子（逐元素计算），但是会发现编译后的函数并没有加速。PyTorch2.0
# 将会从以下两个角度对其进行优化。
