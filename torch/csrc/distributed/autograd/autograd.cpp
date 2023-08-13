#include <ATen/record_function.h>
#include <torch/csrc/distributed/autograd/autograd.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr auto kDistAutogradBackwardProfilingKey =
    "torch::distributed::autograd::backward";

//引擎入口在 backward 函数中有调用，从 DistEngine::getInstance().execute 进入到引擎，由前文可知，这里是主动调用引擎。
void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.autograd.backward");
  RECORD_FUNCTION(
      kDistAutogradBackwardProfilingKey, std::vector<c10::IValue>());
  try {
  //可以看到，最终会调用到 DistEngine::getInstance().execute(context_id, roots, retain_graph) 完成反向传播。这就进入了引擎。
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
  } catch (std::exception& e) {
    // FIXME: crashes if exception type is not RuntimeError
    TORCH_CHECK(false, e.what());
  }
}

} // namespace autograd
} // namespace distributed
} // namespace torch
