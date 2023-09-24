import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

BACKEND = "nccl"
WORLD_SIZE = 2
NUM_INPUTS = 5

def worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

    num_inputs = 0
    with Join([model]):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()

    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")

def main():
    mp.spawn(worker, nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()

    '''
    这将产生以下输出（其中print来自 0 级和 1 级的 ranks，可以任意排序）：

Rank 0 has exhausted all 5 of its inputs!
Rank 1 has exhausted all 6 of its inputs!
    '''