def run_trainer():
    batch = 5
    ntoken = 10
    ninp = 2
    nhid = 3
    nindices = 3
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer( # 创建分布式优化器
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    for epoch in range(10):
        for data, target in get_next_batch():
            # create distributed autograd context
            with dist_autograd.context() as context_id: # 创建分布式上下文
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss]) # 执行分布式后向传播
                # run distributed optimizer
                opt.step(context_id) # 分布式优化器进行更新
                # not necessary to zero grads since they are
                # accumulated into the distributed autograd context
                # which is reset every iteration.
        print("Training epoch {}".format(epoch))

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        # 启动了trainer
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        # trainer 业务逻辑
        _run_trainer()
    else:
        # 启动了参数服务器
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)