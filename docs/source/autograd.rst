.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autosummary::
    :toctree: generated
    :nosignatures:

    backward
    grad

.. _forward-mode-ad:

Forward-mode Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    This API is in beta. Even though the function signatures are very unlikely to change, improved
    operator coverage is planned before we consider this stable.

Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
for detailed steps on how to use this API.

.. autosummary::
    :toctree: generated
    :nosignatures:

    forward_ad.dual_level
    forward_ad.make_dual
    forward_ad.unpack_dual

.. _functional-api:

Functional higher level API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
    This API is in beta. Even though the function signatures are very unlikely to change, major
    improvements to performances are planned before we consider this stable.

This section contains the higher level API for the autograd that builds on the basic API above
and allows you to compute jacobians, hessians, etc.

This API works with user-provided functions that take only Tensors as input and return
only Tensors.
If your function takes other arguments that are not Tensors or Tensors that don't have requires_grad set,
you can use a lambda to capture them.
For example, for a function ``f`` that takes three inputs, a Tensor for which we want the jacobian, another
tensor that should be considered constant and a boolean flag as ``f(input, constant, flag=flag)``
you can use it as ``functional.jacobian(lambda x: f(x, constant, flag=flag), input)``.

.. autosummary::
    :toctree: generated
    :nosignatures:

    functional.jacobian
    functional.hessian
    functional.vjp
    functional.jvp
    functional.vhp
    functional.hvp

.. _locally-disable-grad:

Locally disabling gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`locally-disable-grad-doc` for more information on the differences
between no-grad and inference mode as well as other related mechanisms that
may be confused with the two. Also see :ref:`torch-rst-local-disable-grad`
for a list of functions that can be used to locally disable gradients.

.. _default-grad-layouts:

Default gradient layouts
^^^^^^^^^^^^^^^^^^^^^^^^

When a non-sparse ``param`` receives a non-sparse gradient during
:func:`torch.autograd.backward` or :func:`torch.Tensor.backward`
``param.grad`` is accumulated as follows.

If ``param.grad`` is initially ``None``:

1. If ``param``'s memory is non-overlapping and dense, ``.grad`` is
   created with strides matching ``param`` (thus matching ``param``'s
   layout).
2. Otherwise, ``.grad`` is created with rowmajor-contiguous strides.

If ``param`` already has a non-sparse ``.grad`` attribute:

3. If ``create_graph=False``, ``backward()`` accumulates into ``.grad``
   in-place, which preserves its strides.
4. If ``create_graph=True``, ``backward()`` replaces ``.grad`` with a
   new tensor ``.grad + new grad``, which attempts (but does not guarantee)
   matching the preexisting ``.grad``'s strides.

The default behavior (letting ``.grad``\ s be ``None`` before the first
``backward()``, such that their layout is created according to 1 or 2,
and retained over time according to 3 or 4) is recommended for best performance.
Calls to ``model.zero_grad()`` or ``optimizer.zero_grad()`` will not affect ``.grad``
layouts.

In fact, resetting all ``.grad``\ s to ``None`` before each
accumulation phase, e.g.::

    for iterations...
        ...
        for param in model.parameters():
            param.grad = None
        loss.backward()

such that they're recreated according to 1 or 2 every time,
is a valid alternative to ``model.zero_grad()`` or ``optimizer.zero_grad()``
that may improve performance for some networks.

Manual gradient layouts
-----------------------

If you need manual control over ``.grad``'s strides,
assign ``param.grad =`` a zeroed tensor with desired strides
before the first ``backward()``, and never reset it to ``None``.
3 guarantees your layout is preserved as long as ``create_graph=False``.
4 indicates your layout is *likely* preserved even if ``create_graph=True``.

In-place operations on Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

In-place correctness checks
---------------------------

All :class:`Tensor` s keep track of in-place operations applied to them, and
if the implementation detects that a tensor was saved for backward in one of
the functions, but it was modified in-place afterwards, an error will be raised
once backward pass is started. This ensures that if you're using in-place
functions and not seeing any errors, you can be sure that the computed
gradients are correct.

0x01 总体逻辑
为了行文完整，我们从前文结尾处摘取了总体逻辑关系如下。
如果从计算图角度来看前向计算的过程，就是在构建图和执行图。"构建图"描述的是节点运算之间的关系。"执行图"则是在会话中执行这个运算关系，就是张量在计算图之中进行前向传播的过程。
前向计算依赖一些基础类，在具体分析前向传播之前，我们先要看看这些基础类之间的逻辑关系。从DAG角度来分析 PyTorch 这个系统，其具体逻辑如下。
    图表示计算任务。PyTorch把计算都当作是一种有向无环图，或者说是计算图，但这是一种虚拟的图，代码中没有真实的数据结构。
    计算图由节点（Node）和边（Edge）组成。
    节点（Node）代表了运算操作。
        一个节点通过边来获得 0 个或多个 Tensor，节点执行计算之后会产生 0 个或多个 Tensor。
        节点的成员变量 next_functions 是一个 tuple 列表，此列表就代表本节点要输出到哪些其他 Function。列表个数就是这个 grad_fn 的 Edge 数目，列表之中每一个 tuple 对应一条 Edge 信息，内容就是 (Edge.function, Edge.input_nr)。
    边（Edge）就是运算操作之间的流向关系。
        Edge.function ：表示此 Edge 需要输出到哪一个其他 Function。
        Edge.input_nr ：指定本 Edge 是 Function 的第几个输入。
    使用张量（ Tensor） 表示数据，就是在节点间流动的数据，如果没有数据，计算图就没有任何意义。
具体可以参见下图：
+---------------------+              +----------------------+
| SubBackward0        |              | PowBackward0         |
|                     |      Edge    |                      |  Edge
|   next_functions  +-----+--------> |     next_functions +----------> ...
|                     |   |          |                      |
+---------------------+   |          +----------------------+
                          |
                          |
                          |          +----------------------+
                          |  Edge    | MulBackward0         |
                          +--------> |                      |  Edge
                                     |     next_functions +----------> ...
                                     |                      |
                                     +----------------------+


前面提到，计算图构成了前向/反向传播的结构基础，而Tensor张量 是 PyTorch 中构建计算图的基础之一。
Tensor是PyTorch实现多维数组计算和自动微分的关键数据结构。
    Tensor类似于numpy的ndarray，可以对Tensor进行各种数学运算；
    当设置.requires_grad = True ，在Tensor之上进行的各种操作就会被记录下来，用于后续梯度计算。

我们看看其中的部分成员变量：
data：该张量的数据。
dtype ：该张量的数据类型。
device： 存放该张量的设备类型，比如 CPU 或者是 GPU。
grad：保存数据data对应的梯度，和数据data的形状一样。
    PyTorch会自动追踪和记录对与张量的所有操作，当前向计算完成后调用.backward()方法会自动计算梯度并且将计算结果保存到grad属性中。
    requires_grad = False时，grad为None。
    梯度值不会自动清空，每次在backward计算时都需要将前一时刻的梯度归零，否则梯度值会一直累加。
grad_fn：指向一个Function对象。
    这个Function对象用来在反向传播时候计算输入的梯度。
    若本张量是非叶节点，则 Function 是向叶节点方向操作的反向传播函数，比如例子里 O 节点对应的函数就是MulBackward，即乘法操作的反向函数；
    若本张量是叶节点且requires_grad为True，则 grad_fn 是None。
    grad_fn 有一个属性 next_functions，这是一个二维 tuple，形式为( (函数1， 整数1)，(函数2，整数2), ..., (函数n，整数n) )。 后续我们会详细解释。
is_leaf：记录该张量是否是叶子节点 。
    用户显式初始化的张量是叶子节点。
    所有requires_grad=False的张量按照惯例也是叶子节点。
    is_leaf 属性只有在需要求导的时候才有意义。对于任意一个张量来说，我们可以用 tensor.is_leaf 来判断它是否是叶子张量（leaf tensor）。在反向传播过程中，只有 is_leaf=True 的时候，需要求导张量的导数结果才会被保留下来。
    对于叶子节点来说，其 grad_fn 属性都为空；而对于非叶子结点来说，因为它们是通过一些操作生成的，所以其 grad_fn 不为空。
requires_grad : 设置为True则表示该Tensor需要求导，用于判断该tensor是否需要被跟踪并计算梯度。
    requires_grad属性默认为False，也就是Tensor变量默认是不需要求导的。
    如果一个节点的requires_grad是True，那么所有依赖它的节点的requires_grad也会是True。换言之，如果一个节点依赖的所有节点都不需要求导，那么它的requires_grad也会是False。因此在反向传播过程中，该节点所在的子图会被排除在计算过程之外。
Python的定义其实只是C++世界定义的一个映射，我们接下来就看看在C++如何定义。

早期版本之中，有Tensor和Variable两种数据结构来存储数据，Tensor只负责多维数组的运算。自动微分的职责是Variable完成的。
Variable包含了与autograd有关的属性，可以是计算图中的叶子节点，也可以是计算时候产生的中间变量。
在0.4.0版本之后，Tensor和Variable 的功能进行了合并，自动微分的使用就更加简单了。现在，Variable 其实就是Tensor，只是为了向后兼容，才保留这个名字。

Variable (deprecated)
^^^^^^^^^^^^^^^^^^^^^

.. warning::
    The Variable API has been deprecated: Variables are no longer necessary to
    use autograd with tensors. Autograd automatically supports Tensors with
    ``requires_grad`` set to ``True``. Below please find a quick guide on what
    has changed:

    - ``Variable(tensor)`` and ``Variable(tensor, requires_grad)`` still work as expected,
      but they return Tensors instead of Variables.
    - ``var.data`` is the same thing as ``tensor.data``.
    - Methods such as ``var.backward(), var.detach(), var.register_hook()`` now work on tensors
      with the same method names.

    In addition, one can now create tensors with ``requires_grad=True`` using factory
    methods such as :func:`torch.randn`, :func:`torch.zeros`, :func:`torch.ones`, and others
    like the following:

    ``autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)``

Tensor autograd functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :nosignatures:

   torch.Tensor.grad
   torch.Tensor.requires_grad
   torch.Tensor.is_leaf
   torch.Tensor.backward
   torch.Tensor.detach
   torch.Tensor.detach_
   torch.Tensor.register_hook
   torch.Tensor.retain_grad

:hidden:`Function`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Function

.. autosummary::
    :toctree: generated
    :nosignatures:

    Function.forward
    Function.backward
    Function.jvp
    Function.vmap

Context method mixins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When creating a new :class:`Function`, the following methods are available to `ctx`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    function.FunctionCtx.mark_dirty
    function.FunctionCtx.mark_non_differentiable
    function.FunctionCtx.save_for_backward
    function.FunctionCtx.set_materialize_grads

.. _grad-check:

Numerical gradient checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    gradcheck
    gradgradcheck

Profiler
^^^^^^^^

Autograd includes a profiler that lets you inspect the cost of different
operators inside your model - both on the CPU and GPU. There are three modes
implemented at the moment - CPU-only using :class:`~torch.autograd.profiler.profile`.
nvprof based (registers both CPU and GPU activity) using
:class:`~torch.autograd.profiler.emit_nvtx`.
and vtune profiler based using
:class:`~torch.autograd.profiler.emit_itt`.

.. autoclass:: torch.autograd.profiler.profile

.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.profile.export_chrome_trace
    profiler.profile.key_averages
    profiler.profile.self_cpu_time_total
    profiler.profile.total_average

.. autoclass:: torch.autograd.profiler.emit_nvtx
.. autoclass:: torch.autograd.profiler.emit_itt


.. autosummary::
    :toctree: generated
    :nosignatures:

    profiler.load_nvprof

Anomaly detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: detect_anomaly

.. autoclass:: set_detect_anomaly


Autograd graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Autograd exposes methods that allow one to inspect the graph and interpose behavior during
the backward pass.

The ``grad_fn`` attribute of a :class:`torch.Tensor` holds a  :class:`torch.autograd.graph.Node`
if the tensor is the output of a operation that was recorded by autograd (i.e., grad_mode is
enabled and at least one of the inputs required gradients), or ``None`` otherwise.

.. autosummary::
    :toctree: generated
    :nosignatures:

    graph.Node.name
    graph.Node.metadata
    graph.Node.next_functions
    graph.Node.register_hook
    graph.Node.register_prehook

Some operations need intermediary results to be saved during the forward pass
in order to execute the backward pass.
These intermediary results are saved as attributes on the ``grad_fn`` and can be accessed.
For example::

    >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
    >>> b = a.exp()
    >>> print(isinstance(b.grad_fn, torch.autograd.graph.Node))
    True
    >>> print(dir(b.grad_fn))
    ['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_raw_saved_result', '_register_hook_dict', '_saved_result', 'metadata', 'name', 'next_functions', 'register_hook', 'register_prehook', 'requires_grad']
    >>> print(torch.allclose(b.grad_fn._saved_result, b))
    True

You can also define how these saved tensors should be packed / unpacked using hooks.
A common application is to trade compute for memory by saving those intermediary results
to disk or to CPU instead of leaving them on the GPU. This is especially useful if you
notice your model fits on GPU during evaluation, but not training.
Also see :ref:`saved-tensors-hooks-doc`.

.. autoclass:: torch.autograd.graph.saved_tensors_hooks

.. autoclass:: torch.autograd.graph.save_on_cpu

.. autoclass:: torch.autograd.graph.disable_saved_tensors_hooks
