import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc

def my_add(t1, t2):
  return torch.add(t1, t2)

# On worker 0:

# Setup the autograd context. Computations that take
# part in the distributed backward pass must be within
# the distributed autograd context manager.
with dist_autograd.context() as context_id:
  t1 = torch.rand((3, 3), requires_grad=True)
  t2 = torch.rand((3, 3), requires_grad=True)

  # Perform some computation remotely.
  t3 = rpc.rpc_sync("worker1", my_add, args=(t1, t2))

  # Perform some computation locally based on remote result.
  t4 = torch.rand((3, 3), requires_grad=True)
  t5 = torch.mul(t3, t4)

  # Compute some loss.
  loss = t5.sum()

  # Run the backward pass.
  dist_autograd.backward(context_id, [loss])

  # Retrieve the gradients from the context.
  dist_autograd.get_gradients(context_id)