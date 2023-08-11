#include <torch/csrc/autograd/functions/accumulate_grad.h>

#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace torch {
namespace autograd {

// AccumulateGrad sets sequence_nr to the max value so it's always called
// ASAP during backwards.
//这会new一个AccumulateGrad对象，使用UINT64_MAX 来初始化Function的sequence_nr_成员。
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Node(/*sequence_nr=*/UINT64_MAX), variable(std::move(variable_)) {
  add_input_metadata(variable);
}

/*
当调用 apply 时候， 有两个注意点：
传入的更新函数就是 { grad = std::move(grad_update); } 更新梯度。
mutable_grad 得到的是张量的梯度成员变量。

AccumulateGrad                                 Tensor           AutogradMeta
     +                                           +                   +
     |                                           |                   |
     |                                           |                   |
     |                                           |                   |
     v                                           |                   |
   apply(update_grad)                            |                   |
     +                                           |                   |
     |                                           |                   |
     |                                           |                   |
     |                                           |                   |
     v                                           |                   |
accumulateGrad                                   |                   |
     +                                           |                   |
     |                                           |                   |
     | result = variable_grad + new_grad         |                   |
     |                                           |                   |
     v                result                     v                   v
 update_grad +---------------------------->  mutable_grad +--->    grad_

或者如下，对于一个叶子张量，反向计算时候会调用AccumulateGrad进行累积梯度，然后更新到叶子张量的 grad_ 之中：

+----------------------------------------------+          +-------------------------+
|Tensor                                        |          |TensorImpl               |
|                                              |          |                         |
|                                              |  bridge  |                         |
|   <TensorImpl, UndefinedTensorImpl> impl_ +-----------> |    autograd_meta_ +---------+
|                                              |          |                         |   |
|                                              |          |                         |   |
+----------------------------------------------+          +-------------------------+   |
                                                                                        |
                                                                                        |
                                                                                        |
+-------------------------+                                                             |
| AutogradMeta            | <-----------------------------------------------------------+
|                         |
|                         |
|                         |            +------------------------------------------------+
|                         |            | AccumulateGrad                                 |
|      grad_fn_ +--------------------> |                                                |
|                         |            |                                                |
|                         |            |      apply(grads) {                            |
|                         |            |                                                |
|      grad_accumulator_  |            |         accumulateGrad(new_grad) {             |
|                         |            |                                                |
|                         |            |           result = variable_grad + new_grad    |
|                         |   update   |                                                |
|      grad_    <--------------------------------+ update_grad(result)                  |
|                         |            |                                                |
|                         |            |         }                                      |
|                         |            |      }                                         |
|                         |            |                                                |
|                         |            |                                                |
+-------------------------+            +------------------------------------------------+

现在我们知道了，梯度就是累积在叶子节点的 grad_ 之上
*/
auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable.grad_fn())
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  if (!variable.requires_grad())
    return {};

  // std::move(grads[0]) to avoid bumping up refcount
  at::Tensor new_grad = std::move(grads[0]);

  // Acquire lock to here protect thread safety on variable, this ensures
  // AccumulateGrad does not race to shared variable from different threads
  // when updating the gradients. We don't ensure thread safety on hooks
  // and rely on user to provide thread safe hooks
  // see Note [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);

  at::Tensor& grad = variable.mutable_grad(); // 得到变量的mutable_grad

  // If the function has post hooks (for example, a DDP allreduce hook),
  // call_function in Engine.cpp will temporarily bump the expected refcount
  // by one, hence the addition of !post_hooks().empty() for 'num_expected_refs'
  // in addition to the one reference that we're holding.
  // 'num_expected_refs' is used to determine whether or not we should clone
  // the grad or can steal the grad.
  accumulateGrad(
      variable,
      grad,
      new_grad,
      1 + !post_hooks().empty() /* num_expected_refs */,
      [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });

  return variable_list();
}
} // namespace autograd
} // namespace torch
