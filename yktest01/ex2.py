from math import pi
import torch.optim

x = torch.tensor([pi/2,pi/3],requires_grad=True)
optimizer = torch.optim.SGD([x,],lr=0.2,momentum=0.5)

for step in range(11):
    if step:
        optimizer.zero_grad()
        f.backward()
        optimizer.step()

        for var_name in optimizer.state_dict():
            print(var_name, '\t', optimizer.state_dict()[var_name])
    f=-((x.sin()**3).sum())**3