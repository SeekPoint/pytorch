#pragma once

#include <cstring>
#include <mutex>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/util/Flags.h>

// TODO: rename to c10
C10_DECLARE_bool(caffe2_report_cpu_memory_usage);

namespace c10 {

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
C10_API void NoDelete(void*);

// A simple struct that is used to report C10's memory allocation,
// deallocation status and out-of-memory events to the profiler
//CPUAllocator.h定义了一个名为ProfiledCPUMemoryReporter的类，用于跟踪和记录CPU内存的分配和释放情况，并打印日志。
//类中声明了三个成员函数，使用互斥锁确保在多线程环境下的安全访问。
class C10_API ProfiledCPUMemoryReporter {
 public:
  ProfiledCPUMemoryReporter() = default;
  // 记录内存分配情况
  void New(void* ptr, size_t nbytes);
  // 记录内存不足情况
  void OutOfMemory(size_t nbytes);
  // 记录内存释放情况
  void Delete(void* ptr);

 private:
 // 互斥锁 用于保护对内部数据结构的并发访问
  std::mutex mutex_;
  // 用于存储每个已分配内存块的指针和其对应的字节数
  std::unordered_map<void*, size_t> size_table_;
  // 记录当前已分配的总字节数
  size_t allocated_ = 0;
  // 计数器 用于控制输出日志的频率
  size_t log_cnt_ = 0;
};

C10_API ProfiledCPUMemoryReporter& profiledCPUMemoryReporter();

// Get the CPU Allocator.
C10_API at::Allocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetCPUAllocator(at::Allocator* alloc, uint8_t priority = 0);

// Get the Default CPU Allocator
C10_API at::Allocator* GetDefaultCPUAllocator();

// Get the Default Mobile CPU Allocator
C10_API at::Allocator* GetDefaultMobileCPUAllocator();

// The CPUCachingAllocator is experimental and might disappear in the future.
// The only place that uses it is in StaticRuntime.
// Set the CPU Caching Allocator
C10_API void SetCPUCachingAllocator(Allocator* alloc, uint8_t priority = 0);
// Get the CPU Caching Allocator
C10_API Allocator* GetCPUCachingAllocator();

} // namespace c10
