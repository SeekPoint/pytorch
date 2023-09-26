#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

/*
4.3 RequestCallbackImpl
这时候，读者会有疑问，之前 TensorPipeAgent 明明设置了 RequestCallbackImpl 作为回调函数，
怎么只调用了其 deserializePythonRpcCommand呢，deserialXXX 看起来是序列化相关的，按说应该调用一些业务处理函数，
比如processXXXX 之类的。我们接下来就看看 RequestCallbackImpl。

RequestCallbackImpl 定义在 torch/csrc/distributed/rpc/request_callback_impl.h。
*/
class TORCH_API RequestCallbackImpl : public RequestCallbackNoPython {
 public:
  std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const override;

  void processPythonCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const c10::intrusive_ptr<JitFuture>& responseFuture) const override;

  void processScriptCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const c10::intrusive_ptr<JitFuture>& responseFuture) const override;

  TypePtr getScriptRemoteCallType(
      ScriptRemoteCall& scriptRemoteCall) const override;

  void processScriptRemoteCall(
      ScriptRemoteCall& scriptRemoteCall,
      const std::function<void(void)>& postProcessing,
      std::vector<at::IValue>& stack,
      const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const override;

  void processPythonRemoteCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const c10::intrusive_ptr<JitFuture>& responseFuture,
      std::shared_ptr<LazyStreamContext> ctx) const override;

  void processPythonRRefFetchCall(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const c10::intrusive_ptr<JitFuture>& responseFuture,
      std::shared_ptr<LazyStreamContext> ctx) const override;

  void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const override;

  void processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const int64_t messageId,
      const c10::intrusive_ptr<JitFuture>& responseFuture,
      std::shared_ptr<LazyStreamContext> ctx) const override;

  bool cudaAvailable() const override;

  void processRRefBackward(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const c10::intrusive_ptr<JitFuture>& responseFuture) const override;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
