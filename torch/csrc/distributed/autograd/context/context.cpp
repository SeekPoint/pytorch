#include <torch/csrc/distributed/autograd/context/context.h>

#include <functional>

#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;

DistAutogradContext::DistAutogradContext(int64_t contextId)
    : contextId_(contextId),
      impl_(c10::impl::VirtualGuardImpl{
          at::hasCUDA() ? c10::DeviceType::CUDA : c10::DeviceType::CPU}) {}

int64_t DistAutogradContext::contextId() const {
  return contextId_;
}

std::unordered_set<rpc::worker_id_t> DistAutogradContext::getKnownWorkerIds()
    const {
  std::lock_guard<std::mutex> guard(lock_);
  return knownWorkerIds_;
};

void DistAutogradContext::addKnownWorkerId(const rpc::worker_id_t workerId) {
  std::lock_guard<std::mutex> guard(lock_);
  knownWorkerIds_.insert(workerId);
}

/*
addSendFunction 就是往 sendAutogradFunctions_ 之中添加SendRpcBackward，后续可以按照 message id 来得到这个 SendRpcBackward。

前面是从上下文构建的角度看，本次从上下文内容来看。

此时发送端逻辑如下：

+--------------------------------------------------------------+    +-------------------+
| worker                                                       |    |SendRpcBackward    |
| +---------------------------------------------------------+  |    |                   |
| | DistAutogradContext                                     |  |    |   input_metadata_ |
| |                                                 +-------------> |                   |
| |  contextId_ = context_id_1                      |       |  |    |   next_edges_     |
| |                                                 +       |  |    |                   |
| |  sendAutogradFunctions_ = [msg_id_1, SendRpcBackward_1] |  |    +-------------------+
| |                                                         |  |
| |                                                         |  |
| |  recvAutogradFunctions_                                 |  |
| |                                                         |  |
| +---------------------------------------------------------+  |
|                                                              |
+--------------------------------------------------------------+

                                                                                  sender
+---------------------------------------------------------------------------------------+



*/
void DistAutogradContext::addSendFunction(
    const std::shared_ptr<SendRpcBackward>& func,
    int64_t autograd_message_id) {
  TORCH_INTERNAL_ASSERT(func != nullptr);

  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      sendAutogradFunctions_.find(autograd_message_id) ==
      sendAutogradFunctions_.end());
  sendAutogradFunctions_.emplace(autograd_message_id, func);
}

//addRecvFunction 的添加操作如下，就是看看 recvAutogradFunctions_之中是否已经存在这个 message id 对应的算子，如果没有就添加 。
void DistAutogradContext::addRecvFunction(
    std::shared_ptr<RecvRpcBackward>& func,
    int64_t autograd_message_id) {
  TORCH_INTERNAL_ASSERT(func != nullptr);

  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      recvAutogradFunctions_.find(autograd_message_id) ==
      recvAutogradFunctions_.end());
  recvAutogradFunctions_.emplace(autograd_message_id, func);
}

std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>>
DistAutogradContext::sendFunctions() const {
  std::lock_guard<std::mutex> guard(lock_);
  return sendAutogradFunctions_;
}

/*
至此，逻辑拓展如下，在发送端和接收端都有一个 DistAutogradContext，其 id 都是 context_id_1。

在 每个 DistAutogradContext 之内，均以 msg_id_1 作为key，一个是 SendRpcBackward，一个建立了 RecvRpcBackward。

这就对应了设计之中提到的：

每个自动微分过程被赋予一个唯一的 autograd_context_id，在容器中，这个微分过程的上下文(DistAutogradContext) 依据这个autograd_context_id 来唯一确认。autograd_context_id 是一个 64 bit 的全局唯一id，前 16 bis 是 worker_id，后 48 位是在每个worker内部自动递增id。所以可见，一个Container 之中，是有多个Context的。

此容器还负责维护全局唯一的消息id，用来关联发送/接收自动微分函数对。格式类似于autograd_context_id，是一个64位整数，前16位是工作者id，后48位是worker内部自动递增的。

+----------------------------------------------------------------+
| worker                                                         |    +-------------------+
|                                                                |    |SendRpcBackward    |
|   +---------------------------------------------------------+  |    |                   |
|   | DistAutogradContext                                     |  |    |   input_metadata_ |
|   |                                                 +-------------> |                   |
|   |  contextId_ = context_id_1                      |       |  |    |   next_edges_     |
|   |                                                 +       |  |    |                   |
|   |  sendAutogradFunctions_ = [msg_id_1, SendRpcBackward_1] |  |    +-------------------+
|   |                                                         |  |
|   |  recvAutogradFunctions_                                 |  |
|   |                                                         |  |
|   +---------------------------------------------------------+  |
|                                                                |
|                             +                                  |
|                             |                                  |
+----------------------------------------------------------------+
                              |
                              |
                              |                                                     Sender
+-----------------------------------------------------------------------------------------+
                              |                                                     Receiver
                              |
                              v
+-----------------------------+----------------------------------+
| worker                                                         |
|                                                                |    +-------------------+
|   +---------------------------------------------------------+  |    |RecvRpcBackward    |
|   | DistAutogradContext                                     |  |    |                   |
|   |                                                         |  |    |                   |
|   |   contextId_ = context_id_1                 +-----------------> |   input_metadata_ |
|   |                                             |           |  |    |                   |
|   |   sendAutogradFunctions_                    |           |  |    |   next_edges_     |
|   |                                             +           |  |    |                   |
|   |   recvAutogradFunctions_ = [msg_id_1, RecvRpcBackward_1]|  |    +-------------------+
|   |                                                         |  |
|   +---------------------------------------------------------+  |
|                                                                |
+----------------------------------------------------------------+
我们加入 Container，再拓展一下目前逻辑如下：

每个worker 包括一个DistAutogradContainer。
每个 DistAutogradContainer 包括若干个 DistAutogradContext，依据 context id 提取 DistAutogradContext。
每个 DistAutogradContext 包括 sendAutogradFunctions_ 和 recvAutogradFunctions_，利用 msg id 来获取 SendRpcBackward 或者 RecvRpcBackward。
这样这个反向传播链条就构建了出来。

+------------------------------------------------------------------------------------------------------------------------------------+
| worker                                                                                                                             |
|                                                                                                                                    |
| +---------------------------------------+     +---------------------------------------------------------+    +-------------------+ |
| | DistAutogradContainer                 |     | DistAutogradContext                                     |    |SendRpcBackward    | |
| |                                       |     |                                                 +----------> |                   | |
| |   worker_id_                          |     |  contextId_ = ctx_id_1                          |       |    |   input_metadata_ | |
| |                                       |     |                                                 +       |    |                   | |
| |   next_autograd_message_id_     +---------> |  sendAutogradFunctions_ = [msg_id_1, SendRpcBackward_1] |    |   next_edges_     | |
| |                                 |     |     |                                                         |    |                   | |
| |   next_context_id_              |     |     |  recvAutogradFunctions_                                 |    +-------------------+ |
| |                                 +     |     |                                                         |                          |
| |   autograd_contexts_[ctx_id_1 : ctx]  |     +---------------------------------------------------------+                          |
| |                                       |                                                                                          |
| +----------------------------+----------+                                                                                          |
|                              |                                                                                                     |
+------------------------------------------------------------------------------------------------------------------------------------+
                               |
                               |
+-------------------------------------------------------------------------------------------------------------------------------------+
                               |
                               v
+------------------------------+-----------------------------------------------------------------------------------------------------+
| worker                                                                                                                             |
|                                                                                                                                    |
| +---------------------------------------+     +---------------------------------------------------------+    +-------------------+ |
| | DistAutogradContainer                 |     | DistAutogradContext                                     |    |RecvRpcBackward    | |
| |                                       |     |                                                 +----------> |                   | |
| |   worker_id_                          |     |  contextId_ = ctx_id_1                          |       |    |   input_metadata_ | |
| |                                       |     |                                                 |       |    |                   | |
| |   next_autograd_message_id_     +---------> |  sendAutogradFunctions_                         |       |    |   next_edges_     | |
| |                                 |     |     |                                                 +       |    |                   | |
| |   next_context_id_              |     |     |  recvAutogradFunctions_ = [msg_id_1, RecvRpcBackward_1] |    +-------------------+ |
| |                                 +     |     |                                                         |                          |
| |   autograd_contexts_[ctx_id_1 : ctx]  |     +---------------------------------------------------------+                          |
| |                                       |                                                                                          |
| +---------------------------------------+                                                                                          |
|                                                                                                                                    |
+------------------------------------------------------------------------------------------------------------------------------------+

至此，我们初步分析了上下文相关的类，下文我们把目前已经分析的内容结合起来，系统看看业务逻辑。
*/

std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>>
DistAutogradContext::recvFunctions() const {
  std::lock_guard<std::mutex> guard(lock_);
  return recvAutogradFunctions_;
}

void DistAutogradContext::accumulateGrad(
    const torch::autograd::Variable& variable,  // variable就是目标变量
    const torch::Tensor& grad,                  // grad就是梯度，需要累积到variable之上
    size_t num_expected_refs) {
  TORCH_INTERNAL_ASSERT(grad.defined());
  TORCH_INTERNAL_ASSERT(variable.requires_grad());

  std::lock_guard<std::mutex> guard(lock_);
  auto it = accumulatedGrads_.find(variable);
  at::Tensor old_grad;
  if (it != accumulatedGrads_.end()) {
    // Accumulate multiple grads on the same variable.
    old_grad = it->value();
  }

  // Gradients are computed using the forward streams. Local autograd
  // engine uses AccumulateGrad function to retrieve and apply forward
  // stream during the backward computation. In distributed autograd,
  // we directly call AccumulateGrad::accumulateGrad, and skip the
  // CUDA stream restoration from autograd function. Hence, we manually
  // call it here to get the streams correct.
  auto forward_stream =
      torch::autograd::impl::grad_accumulator(variable)->stream(
          grad.device().type());
  c10::OptionalStreamGuard stream_guard(forward_stream);

  // No higher order gradients supported in distributed autograd.
  AutoGradMode grad_mode(false);

  // TODO: Need to bump 'num_expected_refs' here when we support post_hooks for
  // distributed autograd as part of
  // https://github.com/pytorch/pytorch/issues/33482
  AccumulateGrad::accumulateGrad(
      variable,
      old_grad,
      grad,
      num_expected_refs,
      [this, &variable](at::Tensor&& grad_update) {
        auto device = grad_update.device();
        accumulatedGrads_.insert(variable, std::move(grad_update));
        recordGradEvent(device);
      });
}

std::shared_ptr<torch::autograd::GraphTask> DistAutogradContext::
    retrieveGraphTask() {
  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(graphTask_);
  return graphTask_;
}

void DistAutogradContext::setGraphTask(
    std::shared_ptr<torch::autograd::GraphTask> graphTask) {
  std::lock_guard<std::mutex> guard(lock_);
  TORCH_INTERNAL_ASSERT(
      !graphTask_,
      "Cannot set GraphTask multiple times for the same autograd context");
  graphTask_ = std::move(graphTask);
}

void DistAutogradContext::resetGraphTask() {
  std::lock_guard<std::mutex> guard(lock_);
  graphTask_ = nullptr;
}

void DistAutogradContext::addOutstandingRpc(
    const c10::intrusive_ptr<rpc::JitFuture>& jitFuture) {
  jitFuture->addCallback([this](rpc::JitFuture& future) {
    if (future.hasError()) {
      // If we have an error, let the local autograd engine know about it.
      std::unique_lock<std::mutex> lock(lock_);
      if (graphTask_) {
        graphTask_->set_exception_without_signal(nullptr);
        lock.unlock();
        if (!graphTask_->future_completed_.exchange(true)) {
          graphTask_->future_result_->setErrorIfNeeded(future.exception_ptr());
        }
      } else {
        LOG(WARNING) << "Ignoring error since GraphTask is no longer valid: "
                     << future.tryRetrieveErrorMessage();
      }
    }
  });
  std::lock_guard<std::mutex> guard(lock_);
  outStandingRpcs_.push_back(jitFuture);
}

void DistAutogradContext::clearOutstandingRpcs() {
  std::unique_lock<std::mutex> lock(lock_);
  outStandingRpcs_.clear();
}

void DistAutogradContext::recordGradEvent(c10::Device device) {
  if (device.is_cuda()) {
    auto iter = gradReadyEvents_.find(device);
    if (iter == gradReadyEvents_.end()) {
      c10::Event event(device.type());
      event.record(impl_.getStream(event.device()));
      gradReadyEvents_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(device),
          std::forward_as_tuple(std::move(event)));
    } else {
      iter->second.record(impl_.getStream(device));
    }
  }
}

//最后，分布式引擎会调用 clearAndWaitForOutstandingRpcsAsync 来等待处理完成。
c10::intrusive_ptr<c10::ivalue::Future> DistAutogradContext::
    clearAndWaitForOutstandingRpcsAsync() {
  std::unique_lock<std::mutex> lock(lock_);
  auto outStandingRpcs = std::move(outStandingRpcs_);
  lock.unlock();

  struct State {
    explicit State(int32_t count)
        : future(
              c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get())),
          remaining(count) {}
    c10::intrusive_ptr<c10::ivalue::Future> future;
    std::atomic<int32_t> remaining;
    std::atomic<bool> alreadySentError{false};
  };
  auto state = std::make_shared<State>(outStandingRpcs.size());
  if (outStandingRpcs.empty()) {
    state->future->markCompleted(c10::IValue());
  } else {
    for (auto& rpc : outStandingRpcs) {
      rpc->addCallback([state](rpc::JitFuture& future) {
        if (future.hasError()) {
          // If there's an error, we want to setError() on the future,
          // unless another error has already been sent - use a CAS to
          // guard.
          //
          // Don't decrement num remaining here! (We don't need to, since
          // memory handling is separate). If we simply don't decrement on
          // errors, reaching 0 means that there were no errors - and hence,
          // we can just markCompleted() without any other checking there.
          bool expectedAlreadySent = false;
          if (state->alreadySentError.compare_exchange_strong(
                  expectedAlreadySent, true)) {
            state->future->setError(future.exception_ptr());
          }
          return;
        }

        if (--state->remaining == 0) {
          state->future->markCompleted(c10::IValue());
        }
      });
    }
  }
  return state->future;
}

std::shared_ptr<SendRpcBackward> DistAutogradContext::retrieveSendFunction(
    int64_t autograd_message_id) {
  std::lock_guard<std::mutex> guard(lock_);
  auto it = sendAutogradFunctions_.find(autograd_message_id);
  TORCH_CHECK(
      it != sendAutogradFunctions_.end(),
      "Could not find send function for autograd message id: ",
      autograd_message_id);
  return it->second;
}

const c10::Dict<torch::Tensor, torch::Tensor> DistAutogradContext::
    getGradients() const {
  std::lock_guard<std::mutex> guard(lock_);
  // block current streams before accessing gradients to make sure that
  // gradient computations are finished before use.
  for (auto& entry : gradReadyEvents_) {
    auto& event = entry.second;
    event.block(impl_.getStream(event.device()));
  }
  return accumulatedGrads_; // 分布式梯度累积在这里
}

void DistAutogradContext::runGradCallbackForVariable(
    const torch::autograd::Variable& variable,
    GradCallback&& cb) {
  torch::Tensor grad;
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto it = accumulatedGrads_.find(variable);
    TORCH_INTERNAL_ASSERT(
        it != accumulatedGrads_.end(),
        "The grad for the variable should exist in dist_autograd context.");
    grad = it->value();
  }
  if (cb(grad)) {
    std::lock_guard<std::mutex> guard(lock_);
    auto device = grad.device();
    // Needs to update the grad in the map.
    accumulatedGrads_.insert_or_assign(variable, std::move(grad));
    recordGradEvent(device);
  }
}

namespace {
thread_local ContextPtr tl_context_ptr;
} // namespace

ThreadLocalDistAutogradContext::ThreadLocalDistAutogradContext(
    ContextPtr&& new_context)
    : prev_context_ptr_(std::move(tl_context_ptr)) {
  tl_context_ptr = std::move(new_context);
}

ThreadLocalDistAutogradContext::~ThreadLocalDistAutogradContext() {
  tl_context_ptr = std::move(prev_context_ptr_);
}

// static
ContextPtr ThreadLocalDistAutogradContext::getContextPtr() {
  return tl_context_ptr;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
