#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <mpi.h>

namespace c10d {

constexpr const char* MPI_BACKEND_NAME = "mpi";

//4.2.3 运行
//4.2.3.1 执行封装
//这里有两个封装，WorkEntry 封装计算执行，WorkMPI封装计算执行结果（因为计算是异步的）。具体如下：
//WorkEntry 是执行方法的封装，或者说每次需要执行的集合通信操作，都要封装在这里。
// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()),
        run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;

  // Copy of user provided outputs.
  const std::vector<at::Tensor> dst;

  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;
};

/*
4.2 C++ 世界
4.2.1 ProcessGroupMPI 定义
ProcessGroupMPI 定义位于 torch/lib/c10d/ProcessGroupMPI.cpp。这里相当于做了一个工作队列，以及异步操作。几个注意点如下：

    ProcessGroupMPI 类上的所有函数都应在组中的进程之间以相同的顺序调用。
    这是我们能够保证跨进程匹配相同调用的唯一方法。

    ProcessGroupMPI 类提供的所有MPI函数都在工作线程上异步调度。
    因此，ProcessGroupMPI 依赖于MPI实现，该实现用于提供 MPI_THREAD_SERIALIZED 的最小线程支持值。
    也就是说，进程可以是多线程的，多个线程可以进行MPI调用，
    但一次只能进行一个：MPI调用不是从两个不同的线程同时进行的（所有MPI调用都是序列化的）。
    但是，如果使用 MPI_THREAD_SERIALIZED，ProcessGroupMPI将只支持单个进程组。
    换句话说，全局创建的进程组不能超过1个。

    如果希望使用多个ProcessGroupMPI，它要求MPI实现的线程支持值为MPI\u thread\u multiple，
    也就是说，多个线程可以调用MPI，没有任何限制。

    还要注意，ProcessGroupMPI只支持单个张量操作。换句话说，输入张量向量的大小应始终为1。

    如果使用的MPI是CUDA-aware MPI，则可以支持CUDA tensor，并且ProcessGroupMPI将自动检测此支持。
*/
// ProcessGroupMPI implements MPI bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupMPI requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupMPI will only support a singe process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupMPI, it requres your MPI
// implemenation to have a thread support value of MPI_THREAD_MULTIPLE, that is,
// multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupMPI only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupMPI will automatically detect this support.
/*
具体如下图：

                                                                        +
                                                             Worker 1   |   Worker 2
                                                                        |
                                                                        |
                                                                        |
+-----------------+           +--------------------------------------+  |   +------------------------------------+            +---------------+
| Main Thread     |           |  ProcessGroupMPI                     |  |   | ProcessGroupMPI                    |            | Main Thread   |
|                 |           |                                      |  |   |                                    |            |               |
|                 |           |                                      |  |   |                                    |            |               |
|                 |           |                                      |  |   |                                    |            |               |
|                 |           |  +--------------------------------+  |  |   |  +------------------------------+  |            |               |
|                 |           |  |  runLoop        workerThread_  |  |  |   |  | runloop    workerThread_     |  |            |               |
|                 |           |  |                                |  |  |   |  |                              |  |            |               |
|                 |           |  |                                |  |  |   |  |                              |  |            |               |
|  +---------+    |           |  |   +-------------------------+  |  |  |   |  |  +-----------------------+   |  |            |               |
|  |         |    | allreduce |  |   | queue_                  |  |  |  |   |  |  | queue_                |   |  | allreduce  |   +---------+ |
|  | Reducer | +-------------------> |                         |  |  |  |   |  |  |                       | <-------------------+ |         | |
|  |         |    |           |  |   |                         |  |  |  |   |  |  |                       |   |  |            |   | Reducer | |
|  +---------+    |           |  |   |  +-------------------+  |  |  |  |   |  |  |  +-----------------+  |   |  |            |   |         | |
|                 |           |  |   |  |WorkEntry          |  |  |  |  |   |  |  |  | WorkEntry       |  |   |  |            |   +---------+ |
|                 |           |  |   |  |                   |  |  |  |  |   |  |  |  |                 |  |   |  |            |               |
|                 |           |  |   |  |   MPI_Allreduce <-----------------------------> MPI_Allreduce|  |   |  |            |               |
|                 |           |  |   |  |                   |  |  |  |  |   |  |  |  |                 |  |   |  |            |               |
|                 |           |  |   |  +-------------------+  |  |  |  |   |  |  |  +-----------------+  |   |  |            |               |
|                 |           |  |   |                         |  |  |  |   |  |  |                       |   |  |            |               |
|                 |           |  |   |                         |  |  |  |   |  |  |                       |   |  |            |               |
|                 |           |  |   +-------------------------+  |  |  |   |  |  +-----------------------+   |  |            |               |
|                 |           |  |                                |  |  |   |  |                              |  |            |               |
|                 |           |  +--------------------------------+  |  |   |  +------------------------------+  |            |               |
|                 |           |                                      |  |   |                                    |            |               |
|                 |           |                                      |  |   |                                    |            |               |
|                 |           |                                      |  |   |                                    |            |               |
+-----------------+           +--------------------------------------+  |   +------------------------------------+            +---------------+
                                                                        |
                                                                        |
                                                                        +

手机如下：  03-26.png


*/
class ProcessGroupMPI : public ProcessGroup {
 public:
// WorkMPI 是执行结果的封装。
//在往工作queue插入时候，实际插入的是二元组(WorkEntry, WorkMPI)，我们后续会讲解如何使用。
  class WorkMPI : public ProcessGroup::Work {
   public:
    explicit WorkMPI(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>>& inputTensors =
            c10::nullopt)
        : ProcessGroup::Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {}

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    friend class ProcessGroupMPI;

   private:
    void finishWorkMPI();
    void finishWorkMPIError(std::exception_ptr eptr);

    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  class AsyncWork : public ProcessGroup::Work {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>>& inputTensors =
            c10::nullopt);

    virtual ~AsyncWork();

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    void abort() override;

    std::vector<at::Tensor> result() override;

   protected:
    void populateException();

   private:
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_;
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupMPI(int rank, int size, MPI_Comm pgComm);

  virtual ~ProcessGroupMPI();

  // Abort the MPI program, needs to be called when exception is detected
  void abort();

  const std::string getBackendName() const override {
    return std::string(MPI_BACKEND_NAME);
  }

  c10::intrusive_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // Creating a new ProcessGroupMPI, will initiialize MPI if not initialized
  static c10::intrusive_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});

 protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkMPI>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  c10::intrusive_ptr<ProcessGroup::Work> enqueue(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const c10::optional<std::vector<at::Tensor>>& inputTensors = c10::nullopt);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();
  static std::once_flag onceFlagInitMPI;

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
};

} // namespace c10d
