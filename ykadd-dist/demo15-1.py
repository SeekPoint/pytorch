# --------- MNIST Network to train, from pytorch/examples -----
class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"Putting first 2 convs on {str(device)}")
        # Put conv layers on the first cuda device
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        # Put rest of the network on the 2nd cuda device, if there is one
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")

        print(f"Putting rest of layers on {str(device)}")
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.
def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


# --------- Launcher --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

