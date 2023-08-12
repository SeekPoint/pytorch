'''
0x04 PyTorch 分布式优化器
首先说明一下，为了清晰的分析，我们后续忽略所有 script 相关部分。

4.1 示例
DistributedOptimizer 的使用方法如下：

获取要优化的远程参数列表 (RRef)。 这些也可以是包装在本地RRef中的本地参数。
将 Optimizer 类作为本地优化器来运行所有的RRef owner。
分布式优化器在每个 worker 节点上创建其本地优化器的实例，并持有这些本地优化器的 RRef。
当调用 torch.distributed.optim.DistributedOptimizer.step() 时，分布式优化器使用 RPC 在适当的远程 worker 上远程执行所有本地优化器。torch.distributed.optim.DistributedOptimizer.step 必须获得一个分布式autograd context_id作为输入，本地优化器将把梯度保存在相关的context之中。
如果多个并发的分布式优化器同时更新工作器上的相同参数，则这些更新将通过锁序列化。
看起来有点抽象，我们需要一步一步分析。

4.2 简单的端到端示例
综上所述，以下是使用分布式 autograd 和分布式优化器的简单端到端示例。 如果将代码放入名为“ dist_autograd_simple.py”的文件中，则可以使用命令MASTER_ADDR="localhost" MASTER_PORT=29500 python dist_autograd_simple.py运行该代码：
'''
import multiprocessing as mp
import torch
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer

def random_tensor():
    return torch.rand((3, 3), requires_grad=True)

def _run_process(rank, dst_rank, world_size):
    name = "worker{}".format(rank)
    dst_name = "worker{}".format(dst_rank)

    # Initialize RPC.
    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size
    )

    # Use a distributed autograd context.
    with dist_autograd.context() as context_id: # 本地优化器将把梯度保存在相关的context之中
        # Forward pass (create references on remote nodes).
        rref1 = rpc.remote(dst_name, random_tensor) # 在远端创建一个 random_tensor
        rref2 = rpc.remote(dst_name, random_tensor) # 在远端创建一个 random_tensor
        loss = rref1.to_here() + rref2.to_here() # 获取要优化的远程参数列表 (`RRef`)

        # Backward pass (run distributed autograd).
        dist_autograd.backward([loss.sum()])

        # Build DistributedOptimizer.
        dist_optim = DistributedOptimizer( # 分布式优化器在每个 worker 节点上创建其本地Optimizer的实例，并将持有这些本地优化器的 RRef。
        optim.SGD,
        [rref1, rref2],
        lr=0.05,
        )

        # Run the distributed optimizer step.
        dist_optim.step()

def run_process(rank, dst_rank, world_size):
    _run_process(rank, dst_rank, world_size)
    rpc.shutdown()

processes = []

# Run world_size workers.
world_size = 2
for i in range(world_size):
    p = mp.Process(target=run_process, args=(i, (i + 1) % 2, world_size))
    p.start()
    processes.append(p)

for p in processes:
    p.join()