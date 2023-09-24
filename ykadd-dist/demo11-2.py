from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.optim import Adam

def worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    optim = ZeRO(model.parameters(), Adam, lr=0.01)
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

    num_inputs = 0
    # Pass both `model` and `optim` into `Join()`
    with Join([model, optim]):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()
            optim.step()

    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")
# 这将产生与以前相同的输出。显着的变化是需要另外将ZeroRedundancyOptimizer实例传入 Join()。