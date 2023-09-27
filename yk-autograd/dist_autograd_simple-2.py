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