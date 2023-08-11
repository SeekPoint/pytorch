#pragma once

#include <torch/csrc/python_headers.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>

bool THPEngine_initModule(PyObject* module);

namespace torch {
namespace autograd {
namespace python {
//Python 的 tensors 被转换为 C++ 的 root。
//Python 的 grad_tensors 被转换为 C++ 的 grads。
//Python 的 inputs 被转换为 C++ 的 output_edges。
//最终把这三个变量传递给引擎：PythonEngine.execute(roots, grads, keep_graph, create_graph, accumulate_grad, output_edges)。
struct PythonEngine : public Engine {
  static Engine& get_python_engine();
  ~PythonEngine() override;
  void thread_init(
      int device,
      const std::shared_ptr<ReadyQueue>& ready_queue,
      bool should_increment) override;
  void thread_on_exception(
      std::shared_ptr<GraphTask> graph_task,
      const std::shared_ptr<Node>& fn,
      std::exception& e) override;
  variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      bool accumulate_grad,
      const edge_list& outputs = {}) override;

  c10::intrusive_ptr<at::ivalue::Future> execute_with_graph_task(
      const std::shared_ptr<GraphTask>& graph_task,
      std::shared_ptr<Node> graph_root,
      InputBuffer&& input_buffer) override;

  std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() override;
  std::unique_ptr<SavedVariableHooks> get_default_saved_variable_hooks()
      override;

 private:
  PythonEngine();
};

} // namespace python
} // namespace autograd
} // namespace torch
