import torch
import torch.multiprocessing as mp
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
    with dist_autograd.context() as context_id:
        # Forward pass (create references on remote nodes).
        rref1 = rpc.remote(dst_name, random_tensor)
        rref2 = rpc.remote(dst_name, random_tensor)
        loss = rref1.to_here() + rref2.to_here()

        # Backward pass (run distributed autograd).
        dist_autograd.backward(context_id, [loss.sum()])

        # Build DistributedOptimizer.
        dist_optim = DistributedOptimizer(
        optim.SGD,
        [rref1, rref2],
        lr=0.05,
        )

        # Run the distributed optimizer step.
        dist_optim.step(context_id)

def run_process(rank, world_size):
    dst_rank = (rank + 1) % world_size
    _run_process(rank, dst_rank, world_size)
    rpc.shutdown()

if __name__ == '__main__':
  # Run world_size workers
  world_size = 2
  mp.spawn(run_process, args=(world_size,), nprocs=world_size)