#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/core/Tensor.h>

#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace autograd {

using function_constructor = std::function<std::shared_ptr<Node>(edge_list&&)>;

/**
 * Wraps the tensor outputs in variables and creates the grad_fn and sets the
 * grad_fn if necessary.
 */
TORCH_API variable_list wrap_outputs(
    const variable_list& inputs,
    tensor_list&& outputs,
    const function_constructor& ctr);

///  Checks that inputs contains exactly `args` items and that the first
///  `required_args`
/// items are not nullptr. If not specified, `required_args` defaults to `args`.
TORCH_API void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args = -1,
    bool allow_undefined = false);

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  void operator()(const c10::optional<at::Tensor>& tensor) {
    if (tensor.has_value()) {
      (*this)(*tensor);
    }
  }
  bool short_circuit() {
    return out;
  }
};

template <typename... Args>
inline bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}
/*
set_history 会把前向传播结果加入到history之中，具体就是遍历结果中的张量，然后把每一个张量加入到history。其中关键一点是调用了前面提到的 set_gradient_edge，把 grad_fn（就是 SubBackward0）配置给了result.autograd_meta_ 的 grad_fn_。

回忆一下 Tensor 的成员变量 grad_fn 定义。

grad_fn：指向一个Function对象。
    这个Function对象用来在反向传播时候计算输入的梯度。
    若本张量是非叶节点，则 Function 是向叶节点方向操作的反向传播函数，比如例子里 O 节点对应的函数就是MulBackward，即乘法操作的反向函数；
经过对比，就可以知道，前向操作的输入 result 在反向传播计算梯度时候，就会使用 grad_fn_ 来计算梯度，就是我们这里的 SubBackward0。这样就设置了反向传播如何针对输入来计算梯度。

*/
inline void set_history(
    at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  AT_ASSERT(grad_fn);
  if (variable.defined()) {
    // If the codegen triggers this, you most likely want to add your newly
    // added function to the DONT_REQUIRE_DERIVATIVE list in
    // tools/autograd/gen_variable_type.py
    TORCH_INTERNAL_ASSERT(isDifferentiableType(variable.scalar_type()));

    //// grad_fn 的 input_metadata 之中添加了输出实例，输出实例在反向传播时候就是输入
    auto output_nr = grad_fn->add_input_metadata(variable);

    // 输出实例 result 中设置上了grad_fn，这里配置了边，边就是 {grad_fn, output_nr}。
    // output_nr_被赋值成了"当前Variable信息在input_metadata_中的index"。
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    //// 设置成未定义
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

inline void set_history(
    std::vector<Variable>&& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn); // 调用到上面的函数
  }
}

inline void set_history(
    std::vector<Variable>& variables,
    const std::shared_ptr<Node>& grad_fn) {
  for (auto& variable : variables) {
    set_history(variable, grad_fn);
  }
}

inline bool isFwGradDefined(const c10::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() && t->_fw_grad(/*level */ 0).defined();
}

inline bool isFwGradDefinedTensorList(const at::ITensorListRef& variables) {
  bool ret = false;
  for (auto& variable : variables) {
    ret |= isFwGradDefined(variable);
  }
  return ret;
}

inline bool isFwGradDefinedTensorList(
    const c10::List<c10::optional<at::Tensor>> li) {
  bool ret = false;
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    ret |= (t.has_value() && isFwGradDefined(t.value()));
  }
  return ret;
}

} // namespace autograd
} // namespace torch
