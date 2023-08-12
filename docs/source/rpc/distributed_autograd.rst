:orphan:

.. _distributed-autograd-design:

Distributed Autograd Design
===========================

This note will present the detailed design for distributed autograd and walk
through the internals of the same. Make sure you're familiar with
:ref:`autograd-mechanics` and the :ref:`distributed-rpc-framework` before
proceeding.

Background
^^^^^^^^^^

Let's say you have two nodes and a very simple model partitioned across two
nodes. This can be implemented using :mod:`torch.distributed.rpc` as follows:

.. code::

  import torch
  import torch.distributed.rpc as rpc

  def my_add(t1, t2):
    return torch.add(t1, t2)

  # On worker 0:
  t1 = torch.rand((3, 3), requires_grad=True)
  t2 = torch.rand((3, 3), requires_grad=True)

  # Perform some computation remotely.
  t3 = rpc.rpc_sync("worker1", my_add, args=(t1, t2))

  # Perform some computation locally based on remote result.
  t4 = torch.rand((3, 3), requires_grad=True)
  t5 = torch.mul(t3, t4)

  # Compute some loss.
  loss = t5.sum()

The main motivation behind distributed autograd is to enable running a backward
pass on such distributed models with the ``loss`` that we've computed and
record appropriate gradients for all tensors that require gradients.

.. attaching_send_recv_functions:

Autograd recording during the forward pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch 在前向传播期间构建 autograd 图，该图用于执行后向传播。有关更多详细信息，请参阅 autograd 如何编码历史记录。

对于分布式 autograd，我们需要在前向传播期间跟踪所有 RPC，以确保正确执行后向传播。为此，当执行 RPC 时候，我们把 send和recv functions 附加到autograd图之上。

该send函数附加到 RPC 的发起源节点之上，其输出边指向 RPC 输入张量的 autograd 函数。在向后传播期间，send函数的输入是从目标接收的，是对应recv函数的输出。
该recv函数附加到 RPC 的接受目标节点之上，其输入从某些运算符得到，这些运算符使用输入张量在RPC接受目标上执行。在后向传播期间，recv函数的输出梯度将被发送到源节点之上，并且作为send方法的输入。
每send-recv对被分配一个全局唯一的autograd_message_id 以唯一地标识该send-recv对。这对于在向后传播期间查找远程节点上的相应函数很有用。
对于RRef，每当我们调用torch.distributed.rpc.RRef.to_here() 时，我们都为涉及的张量添加了一个适当的send-recv对。
例如，这就是我们上面示例的 autograd 图的样子（为简单起见，t5.sum() 被排除在外）

PyTorch builds the autograd graph during the forward pass and this graph is
used to execute the backward pass. For more details see
:ref:`how-autograd-encodes-history`.

For distributed autograd, we need to keep track of all RPCs during the forward
pass to ensure the backward pass is executed appropriately. For this purpose,
we attach ``send`` and ``recv`` functions to the autograd graph when we perform
an RPC.

- The ``send`` function is attached to the source of the RPC and its output
  edges point to the autograd function for the input tensors of the RPC.
  The input for this function during the backward pass is received from the
  destination as the output of the appropriate ``recv`` function.
- The ``recv`` function is attached to the destination of the RPC and its
  inputs are retrieved from operators executed on the destination using the
  input tensors. The output gradients of this function are sent to the source
  node to the appropriate ``send`` function during the backward pass.
- Each ``send-recv`` pair is assigned a globally unique ``autograd_message_id``
  to uniquely identify the pair. This is useful to look up the corresponding
  function on a remote node during the backward pass.
- For :ref:`rref`, whenever we call :meth:`torch.distributed.rpc.RRef.to_here`
  we attach an appropriate ``send-recv`` pair for the tensors involved.

As an example, this is what the autograd graph for our example above would look
like (t5.sum() excluded for simplicity):

.. image:: ../_static/img/distributed_autograd/send_recv_functions.png

.. autograd_context:

Distributed Autograd Context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each forward and backward pass that uses distributed autograd is assigned a
unique :class:`torch.distributed.autograd.context` and this context has a
globally unique ``autograd_context_id``. This context is created on each node
as needed.


每个使用分布式 autograd 的前向和后向传播都被分配了一个唯一的torch.distributed.autograd.context，并且这个上下文具有一个全局唯一的autograd_context_id 。如果有需要，在每个节点上都会创建上下文。

上下文的作用如下：

运行分布式反向传播的多个节点可能会在同一个张量上累积梯度并且存储在张量的.grad之上。在我们运行优化器之前，张量的.grad可能累积了来自各种分布式反向传播的梯度。这类似于把torch.autograd.backward()在本地进行多次调用。为了提供一种把每个反向传播梯度分离开的方法，在每个反向传播过程里，梯度将被累积在torch.distributed.autograd.context 之中。
在前向传播期间，我们在上下文中存储每个 autograd 传播的send和recv函数。这确保我们在 autograd 图中保存对适当节点的引用以使其保持活动状态。除此之外，这也使得在向后传播期间很容易查找到对应的send和recv函数。
一般来说，我们也使用这个上下文来存储每个分布式 autograd 传播的一些元数据。
从用户的角度来看，autograd 上下文设置如下：


This context serves the following purpose:

1. Multiple nodes running distributed backward passes might accumulate
   gradients on the same tensor and as a result the ``.grad`` field of the
   tensor would have gradients from a variety of distributed backward passes
   before we have the opportunity to run the optimizer. This is similar to
   calling :meth:`torch.autograd.backward` multiple times locally. In order to
   provide a way of separating out the gradients for each backward pass, the
   gradients are accumulated in the :class:`torch.distributed.autograd.context`
   for each backward pass.
2. During the forward pass we store the ``send`` and ``recv`` functions for
   each autograd pass in this context. This ensures we hold references to the
   appropriate nodes in the autograd graph to keep it alive. In addition to
   this, it is easy to look up the appropriate ``send`` and ``recv`` functions
   during the backward pass.
3. In general we also use this context to store some metadata for each
   distributed autograd pass.

|

From the user's perspective the autograd context is setup as follows:

.. code::

  import torch.distributed.autograd as dist_autograd
  with dist_autograd.context() as context_id:
    loss = model.forward()
    dist_autograd.backward(context_id, loss)


需要注意的是，模型的前向传播必须在分布式autograd上下文管理器中调用，因为需要一个有效的上下文来确保：所有的send和recv方法被存储起来，并且在所有参与节点之上执行后向传播。
It is important to note that your model's forward pass must be invoked within
the distributed autograd context manager, as a valid context is needed in
order to ensure that all ``send`` and ``recv`` functions are stored properly
to run the backward pass across all participating nodes.

Distributed Backward Pass
^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we outline the challenge of computing dependencies accurately
during a distributed backward pass and describe a couple of algorithms (with
tradeoffs) on how we can execute a distributed backward pass.

Computing dependencies
----------------------

Consider the following piece of code being run on a single machine

.. code::

  import torch
  a = torch.rand((3, 3), requires_grad=True)
  b = torch.rand((3, 3), requires_grad=True)
  c = torch.rand((3, 3), requires_grad=True)
  d = a + b
  e = b * c
  d.sum.().backward()

This is what the autograd graph for the code above would look like:

.. image:: ../_static/img/distributed_autograd/local_dependencies.png
  :scale: 80%

作为反向传播的一部分，autograd 引擎执行的第一步是计算 autograd 图中每个节点的依赖项数量。这有助于 autograd 引擎知道图中的节点何时准备好了可以执行。括号内为数字add(1)和mul(0)表示依赖关系的数量。如您所见，这意味着在向后传播期间，add 节点需要 1 个输入，mul节点不需要任何输入（换句话说，不需要执行）。本地 autograd 引擎通过从根节点（在本例中是d）遍历图来计算这些依赖关系。

实际上，Autograd 图中的某些节点可能不会在向后传播中执行。这一事实对分布式 autograd 提出了挑战。考虑这段使用 RPC 的代码。

The first step the autograd engine performs as part of the backward pass is
computing the number of dependencies for each node in the autograd graph. This
helps the autograd engine know when a node in the graph is ready for execution.
The numbers in brackets for ``add(1)`` and ``mul(0)`` denote the number of
dependencies. As you can see, this means during the backward pass the ``add``
node needs 1 input and the ``mul`` node doesn't need any inputs (in other
words doesn't need to be executed). The local autograd engine computes these
dependencies by traversing the graph from the root nodes (``d`` in this case).

The fact that certain nodes in the autograd graph might not be executed in the
backward pass poses a challenge for distributed autograd. Consider this piece
of code which uses RPC.

.. code::

  import torch
  import torch.distributed.rpc as rpc

  a = torch.rand((3, 3), requires_grad=True)
  b = torch.rand((3, 3), requires_grad=True)
  c = torch.rand((3, 3), requires_grad=True)

  d = rpc.rpc_sync("worker1", torch.add, args=(a, b))
  e = rpc.rpc_sync("worker1", torch.mul, args=(b, c))
  loss = d.sum()

The associated autograd graph for the code above would be:

.. image:: ../_static/img/distributed_autograd/distributed_dependencies.png

计算此分布式 autograd 图的依赖项更具挑战性，并且需要一些开销（在计算或网络通信方面）。
对于性能敏感的应用，我们可以通过假设每个send和recv函数都是反向传播的有效成分来避免大量开销（大多数应用不会执行未使用的 RPC）。这简化了分布式 autograd 算法并且效率更高，但代价是应用程序需要了解这些限制。这种算法称为FAST模式算法，下面详细介绍。
在一般情况下， 作为向后传播的一部分，可能不需要每个send和recv函数都是有效的。为了解决这个问题，我们提出了一种SMART 模式算法，此算法将在后面的部分中描述。请注意，目前仅实现了FAST模式算法。

Computing dependencies of this distributed autograd graph is much more
challenging and requires some overhead (either in terms of computation or
network communication).

For performance sensitive applications we can avoid a
lot of overhead by assuming every ``send`` and ``recv`` function are valid as
part of the backward pass (most applications don't perform RPCs that aren't
used). This simplifies the distributed autograd algorithm and is much more
efficient, but at the cost that the application needs to be aware of the
limitations. This algorithm is called the `FAST mode algorithm`_ and is
described in detail below.

In the general case it might not be necessary that every ``send`` and ``recv``
function is valid as part of the backward pass. To address this, we have
proposed a `SMART mode algorithm`_ which is described in a later section.
Please note that currently, only the `FAST` mode algorithm is implemented.

.. _fast-mode-algorithm:

FAST mode algorithm
-------------------

The key assumption of this algorithm is that each ``send`` function has a
dependency of 1 when we run a backward pass. In other words, we assume we'll
receive a gradient over RPC from another node.

The algorithm is as follows:

我们从具有反向传播根的worker开始（所有根都必须是本地的）。
查找当前Distributed Autograd Context 的所有send函数 。
从提供的根和我们检索到的所有send函数开始，我们在本地计算依赖项 。
计算依赖项后，使用提供的根来启动本地 autograd 引擎。
当 autograd 引擎执行该recv函数时，该recv 函数通过 RPC 将输入梯度发送到适当的worker。每个recv函数都知道目标 worker id，因为它被记录为前向传播的一部分。通过autograd_context_id和 autograd_message_id 该recv函数被发送到远程主机。
当远程主机收到这个请求时，我们使用 autograd_context_id和autograd_message_id来查找适当的send函数。
如果这是worker第一次收到对给定 autograd_context_id的请求，它将按照上面的第 1-3 点所述在本地计算依赖项。
然后将在第6点接受到的send方法插入队列，以便在该worker的本地 autograd 引擎上执行。
最后，我们不是在 Tensor的.grad之上累积梯度，而是在每个Distributed Autograd Context之上分别累积梯度 。梯度存储在Dict[Tensor, Tensor]之中 ，Dict[Tensor, Tensor]基本上是从 Tensor 到其关联梯度的映射，并且可以使用 get_gradients() API检索该映射 。

1. We start from the worker which has the roots for the backward pass
   (all roots must be local).
2. Lookup all the ``send`` functions for the current
   `Distributed Autograd Context`_.
3. Compute dependencies locally starting from the provided roots and all the
   ``send`` functions we retrieved.
4. After computing dependencies, kick off the local autograd engine with the
   provided roots.
5. When the autograd engine executes the ``recv`` function, the ``recv``
   function sends the input gradients via RPC to the appropriate worker.
   Each ``recv`` function knows the destination worker id since it is recorded
   as part of the forward pass. The ``recv`` function also sends over the
   ``autograd_context_id`` and ``autograd_message_id`` to the remote host.
6. When this request is received on the remote host, we use the
   ``autograd_context_id`` and ``autograd_message_id`` to look up the
   appropriate ``send`` function.
7. If this is the first time a worker has received a request for the given
   ``autograd_context_id``, it will compute dependencies locally as described
   in points 1-3 above.
8. The ``send`` function retrieved in 6. is then enqueued for execution on the
   local autograd engine for that worker.
9. Finally, instead of accumulating the gradients on the ``.grad`` field of the
   Tensor, we accumulate the gradients separately per
   `Distributed Autograd Context`_. The gradients are stored in a
   ``Dict[Tensor, Tensor]``, which is basically a map from Tensor to its
   associated gradient and this map can be retrieved using the
   :meth:`~torch.distributed.autograd.get_gradients` API.

|

As an example the complete code with distributed autograd would be as follows:

.. code::

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

The distributed autograd graph with dependencies would be as follows (t5.sum() excluded for simplicity):

.. image:: ../_static/img/distributed_autograd/distributed_dependencies_computed.png

应用于上述示例的FAST 模式算法如下：

在Worker 0上，我们从根loss和send1开始计算依赖关系。 结果，send1对Worker 0的依赖数为 1，mul对Worker 0的依赖数为 1。
现在，我们在Worker 0上启动本地 autograd 引擎。 我们首先执行mul函数，将其输出作为t4的梯度，累积存储在 autograd 上下文中。 然后，我们执行recv2，它将这些梯度发送到Worker 1。
由于这是Worker 1第一次知道有关此反向传播的信息，因此它将进行依赖关系计算，并且相应地标记send2，add和recv1的依赖性。
接下来，在Worker 1的本地 autograd 引擎上将send2插入队列，该引擎将依次执行add和recv1。
当执行recv1时，它将梯度发送到Worker 0。
由于Worker 0已经计算了此向后传播的依赖性，因此它仅仅在本地将send1插入队列并且执行。
最后，t1，t2和t4的梯度会累积在分布式 Autograd 上下文中。

The `FAST mode algorithm`_ applied to the above example would be as follows:

1. On ``Worker 0`` we start from the roots ``loss`` and ``send1`` to compute
   dependencies. As a result ``send1`` is marked with a dependency of 1 and ``mul``
   on ``Worker 0`` is marked with a dependency of 1.
2. Now, we kickoff the local autograd engine on ``Worker 0``. We first execute
   the ``mul`` function, accumulate its output in the autograd context as the
   gradient for ``t4``. Then, we execute ``recv2`` which sends the gradients to
   ``Worker 1``.
3. Since this is the first time ``Worker 1`` has heard about this backward pass,
   it starts dependency computation and marks the dependencies for ``send2``,
   ``add`` and ``recv1`` appropriately.
4. Next, we enqueue ``send2`` on the local autograd engine of ``Worker 1``, which
   in turn executes ``add`` and ``recv1``.
5. When ``recv1`` is executed it sends the gradients over to ``Worker 0``.
6. Since ``Worker 0`` has already computed dependencies for this backward pass,
   it just enqueues and executes ``send1`` locally.
7. Finally, gradients for ``t1``, ``t2`` and ``t4`` are accumulated in the
   `Distributed Autograd Context`_.

SMART mode algorithm
--------------------
Full details of this algorithm are still in the works, but for the general idea
you can refer to **Distributed Autograd Algorithm Smart mode** section in the
`RFC`_.

Distributed Optimizer
^^^^^^^^^^^^^^^^^^^^^
该DistributedOptimizer操作如下：

获取要优化的远程参数（RRef）列表。这些参数也可以是包含在本地 RRef的本地参数。
将一个Optimizer类作为本地优化器，该优化器将在所有不同的RRef拥有者之上运行。
分布式优化器在每个工作节点上创建一个本地Optimizer实例，并且对于每一个Optimizer保存一个RRef。
当调用torch.distributed.optim.DistributedOptimizer.step()时，分布式优化器使用 RPC 在适当的远程工作者上远程执行所有本地优化器。必须为 torch.distributed.optim.DistributedOptimizer.step() 提供一个分布式autogradcontext_id。 本地优化器使用context_id 在相应上下文中存储梯度。
如果多个并发分布式优化器正在更新一个 worker 上的同一批参数，这些更新将通过锁来进行序列操作。

The :class:`~torch.distributed.optim.DistributedOptimizer` operates as follows:

1. Takes a list of remote parameters (:class:`~torch.distributed.rpc.RRef`) to
   optimize. These could also be local parameters wrapped within a local
   ``RRef``.
2. Takes a :class:`~torch.optim.Optimizer` class as the local
   optimizer to run on all distinct ``RRef`` owners.
3. The distributed optimizer creates an instance of the local ``Optimizer`` on
   each of the worker nodes and holds an ``RRef`` to them.
4. When :meth:`torch.distributed.optim.DistributedOptimizer.step` is invoked,
   the distributed optimizer uses RPC to remotely execute all the local
   optimizers on the appropriate remote workers. A distributed autograd
   ``context_id`` must be provided as input to
   :meth:`torch.distributed.optim.DistributedOptimizer.step`. This is used
   by local optimizers to apply gradients stored in the corresponding
   context.
5. If multiple concurrent distributed optimizers are updating the same
   parameters on a worker, these updates are serialized via a lock.

Simple end to end example
^^^^^^^^^^^^^^^^^^^^^^^^^

Putting it all together, the following is a simple end to end example using
distributed autograd and the distributed optimizer. If the code is placed into a
file called "dist_autograd_simple.py", it can be run with the command
:code:`MASTER_ADDR="localhost" MASTER_PORT=29500 python dist_autograd_simple.py`:

.. code::

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

.. _RFC: https://github.com/pytorch/pytorch/issues/23110
