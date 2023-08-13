#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/distributed/autograd/autograd.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace autograd {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

//定义了 backward，get_gradients等等。
PyObject* dist_autograd_init(PyObject* _unused, PyObject* noargs) {
  auto autograd_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.autograd"));
  if (!autograd_module) {
    throw python_error();
  }

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = torch_C_m.def_submodule(
      "_distributed_autograd", "distributed autograd bindings");

  auto module = py::handle(m).cast<py::module>();

  auto distAutogradContext =
      shared_ptr_class_<DistAutogradContext>(module, "DistAutogradContext")
          .def(
              "_context_id",
              &DistAutogradContext::contextId,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_recv_functions",
              [](const DistAutogradContext& ctx) {
                std::map<int64_t, py::object> funcs;
                auto recvFunctions = ctx.recvFunctions();

                // Acquire GIL only when necessary to avoid deadlocks.
                pybind11::gil_scoped_acquire ag;
                for (const auto& map_entry : recvFunctions) {
                  funcs.emplace(
                      map_entry.first,
                      py::reinterpret_steal<py::object>(
                          torch::autograd::functionToPyObject(
                              map_entry.second)));
                }
                return funcs;
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_send_functions",
              [](const ContextPtr& ctx) {
                std::map<int64_t, py::object> funcs;
                auto sendFunctions = ctx->sendFunctions();

                // Acquire GIL only when necessary to avoid deadlocks.
                pybind11::gil_scoped_acquire ag;
                for (const auto& map_entry : sendFunctions) {
                  funcs.emplace(
                      map_entry.first,
                      py::reinterpret_steal<py::object>(
                          torch::autograd::functionToPyObject(
                              map_entry.second)));
                }
                return funcs;
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_known_worker_ids",
              &DistAutogradContext::getKnownWorkerIds,
              py::call_guard<py::gil_scoped_release>());
  //具体通过如下映射，我们可以看到 C++ 世界之中对应的方法，调用到了 DistAutogradContainer::getInstance().newContext()。
  module.def(
      "_new_context",
      []() -> const ContextPtr {
        return DistAutogradContainer::getInstance().newContext();
      },
      py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_release_context",
      [](int64_t context_id) {
        return DistAutogradContainer::getInstance().releaseContext(context_id);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_get_max_id",
      []() { return DistAutogradContainer::getInstance().getMaxId(); },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_is_valid_context",
      [](int64_t worker_id) {
        DistAutogradContainer::getInstance().isValidContext(worker_id);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_retrieve_context",
      [](int64_t context_id) -> const ContextPtr {
        return DistAutogradContainer::getInstance().retrieveContext(context_id);
      },
      py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_current_context",
      []() -> const ContextPtr {
        return DistAutogradContainer::getInstance().currentContext();
      },
      py::return_value_policy::reference,
      py::call_guard<py::gil_scoped_release>());
  //我们再推导，看看如何设置 worker id，找到了如下，看来需要看看 python 世界的 _init 方法。
  //来到 python 世界  def init_rpc(
  //，可以看到，使用了 rank 来作为参数，而 rank 是每个 worker 唯一的，这样就保证了 worker ID 唯一，从而 消息 id 唯一。
  /*
  我们把这些逻辑关系总结下来：

worker_id = rank;

container.worker_id_ = worker_id;

container.next_autograd_message_id_ = static_cast<int64_t>(worker_id) << kAutoIncrementBits
然后 next_autograd_message_id_ 内部递增。

int64_t DistAutogradContainer::newAutogradMessageId() {
  // Check for overflow into workerId_ section.
  TORCH_INTERNAL_ASSERT(next_autograd_message_id_ < max_id_);
  return next_autograd_message_id_++;
}
所以，AutogradMessageId 是全局唯一的。我们用图例来看看：

+----------------------------------------------------------------------------------------+
| worker                                                                                 |
|                       +-------------------------------------+                          |
|                       | DistAutogradContainer               |                          |
|                       |                                     |                          |
|                       |                                     |                          |
|              init()   |                                     |                          |
|      rank +--------------+----> worker_id_                  |                          |
|                1      |  |                                  |   newAutogradMessageId() |
|                       |  +----> next_autograd_message_id_+------------------+          |
|                       |                                     |          2    |          |
|                       +-------------------------------------+               |          |
|                                                                             |          |
|                                                                             |          |
|                                                                             |          |
|                                                                             |          |
|                     +---------------------------------------------------------------+  |
|                     | getMessageWithAutograd                                |       |  |
|                     |                                                       |       |  |
|                     |                                                       v       |  |
|                     |                                                               |  |
|                     |   AutogradMetadata autogradMetadata(contextId(), MessageId()) |  |
|                     |                           4                           3       |  |
|                     |                                                               |  |
|                     +---------------------------------------------------------------+  |
|                                                                                        |
+----------------------------------------------------------------------------------------+
为了看看 autogradContextId 为什么可以保证唯一，我们需要先分析 DistAutogradContainer 和 DistAutogradContext。
  */
  module.def(
      "_init",
      [](int64_t worker_id) { DistAutogradContainer::init(worker_id); },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_get_debug_info",
      []() { return DistEngine::getInstance().getDebugInfo(); },
      py::call_guard<py::gil_scoped_release>());

  py::options options;
  options.disable_function_signatures();

  module.def(
      "backward",
      backward,
      R"(
backward(context_id: int, roots: List[Tensor], retain_graph = False) -> None

Kicks off the distributed backward pass using the provided roots. This
currently implements the :ref:`fast-mode-algorithm` which
assumes all RPC messages sent in the same distributed autograd context
across workers would be part of the autograd graph during the backward pass.

We use the provided roots to discover the autograd graph and compute
appropriate dependencies. This method blocks until the entire
autograd computation is done.

We accumulate the gradients in the appropriate
:class:`torch.distributed.autograd.context` on each of the nodes. The autograd
context to be used is looked up given the ``context_id`` that is passed in when
:meth:`torch.distributed.autograd.backward` is called. If there is no valid
autograd context corresponding to the given ID, we throw an error. You can
retrieve the accumulated gradients using the
:meth:`~torch.distributed.autograd.get_gradients` API.

Arguments:
    context_id (int): The autograd context id for which we should retrieve the gradients.
    roots (list): Tensors which represent the roots of the autograd
                  computation. All the tensors should be scalars.
    retain_graph(bool, optional): If False, the graph used to compute the grad
                  will be freed. Note that in nearly all cases setting this
                  option to True is not needed and often can be worked around
                  in a much more efficient way. Usually, you need to set this
                  to True to run backward multiple times.

Example::
    >>> import torch.distributed.autograd as dist_autograd
    >>> with dist_autograd.context() as context_id:
    >>>     pred = model.forward()
    >>>     loss = loss_func(pred, loss)
    >>>     dist_autograd.backward(context_id, loss)
)",
      py::arg("contextId"),
      py::arg("roots"),
      py::arg("retain_graph") = false,
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "get_gradients",
      [](int64_t contextId) -> py::dict {
        const auto& autogradContext =
            DistAutogradContainer::getInstance().retrieveContext(contextId);
        auto ival = IValue(autogradContext->getGradients());

        // Acquire GIL only for pyobject conversion.
        pybind11::gil_scoped_acquire ag;
        return torch::jit::toPyObject(ival);
      },
      R"(
get_gradients(context_id: int) -> Dict[Tensor, Tensor]

Retrieves a map from Tensor to the appropriate gradient for that Tensor
accumulated in the provided context corresponding to the given ``context_id``
as part of the distributed autograd backward pass.

Arguments:
    context_id(int): The autograd context id for which we should retrieve the
                     gradients.

Returns:
    A map where the key is the Tensor and the value is the associated gradient
    for that Tensor.

Example::
    >>> import torch.distributed.autograd as dist_autograd
    >>> with dist_autograd.context() as context_id:
    >>>     t1 = torch.rand((3, 3), requires_grad=True)
    >>>     t2 = torch.rand((3, 3), requires_grad=True)
    >>>     loss = t1 + t2
    >>>     dist_autograd.backward(context_id, [loss.sum()])
    >>>     grads = dist_autograd.get_gradients(context_id)
    >>>     print(grads[t1])
    >>>     print(grads[t2])
)",
      py::arg("context_id"),
      py::call_guard<py::gil_scoped_release>());

  Py_RETURN_TRUE;
}
} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_dist_autograd_init", dist_autograd_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
