#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>

#include <c10/util/intrusive_ptr.h>

namespace c10 {
/*
Tensor底层数据结构
在PyTorch中，Storage和StorageImpl是Tensor的内部类，它们的主要作用是管理数据和管理存储。
Storage类封装一个intrusive_ptr<StorageImpl>类型的成员变量，面向上层给Tensor提供访问和修改底层信息的方法。
而StorageImpl类面向底层，负责内存分配和管理。Tensor是通过类内继承的方式，即声明一个Storage类型的成员变量来使用相应的方法。

通过将Storage和StorageImpl解耦，使得PyTorch具有更高的灵活性和性能，同时隐藏了底层数据管理的复杂性。
但是也有人吐槽说Storage需要设计包含设备参数的全新API，现有的设计比较容易出bug。


*/
// A storage represents the underlying backing data buffer for a
// tensor.  This concept was inherited from the original Torch7
// codebase; we'd kind of like to get rid of the concept
// (see https://github.com/pytorch/pytorch/issues/14797) but
// it's hard work and no one has gotten around to doing it.
//
// NB: storage is supposed to uniquely own a data pointer; e.g.,
// two non-null data pointers alias if and only if they are from
// the same storage.  Technically you can violate this invariant
// (e.g., you can create a non-owning StorageImpl with at::from_blob)
// but a lot of things won't work correctly, including:
//
// - An ordinary deleter on such a storage is wrong, because normal deleters
//   assume unique ownership, but if you have two storages at the same data,
//   that implies there is some sort of shared ownership. So your deleter would
//   have to actually be internally doing some sort of refcount thing
// - Deepcopy in Python side relies on storage equality and not data pointer
//   equality; so if there are two separate storages pointing to the same data,
//   the data will actually get duplicated in that case (one data ptr before,
//   two data ptrs after)
// - Version counts won't work correctly, because we do all VC tracking at the
//   level of storages (unless you explicitly disconnect the VC with detach);
//   mutation because data pointers are the same are totally untracked
/*
StorageImpl继承自intrusive_ptr_target，目的是借助父类实现的计数功能，
然后结合智能指针c10::intrusive_ptr（其负责内存管理，但不负责计数）的帮助，就可以实现“侵入式”的引用计数指针。

Storage类和StorageImpl之间使用了bridge设计模式，主要是为了保证ABI的兼容。

StorageImpl主要封装了指向内存的指针、分配器等。
*/
struct C10_API StorageImpl : public c10::intrusive_ptr_target {  // 继承自 intrusive_ptr
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(std::move(size_bytes)),
        size_bytes_is_symbolic_(size_bytes_.is_symbolic()),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator) {
    if (resizable) { // 如内存大小可调整，则必须指定分配器
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

    // 分配一块大小为 size_bytes 的内存，根据该内存创建 StorageImpl
  // .is_symbolic() 返回 bool 值，判断 size_bytes 是否越界
// 从现有内存创建 StorageImpl
  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            size_bytes.is_symbolic()
                ? allocator->allocate(0)
                : allocator->allocate(size_bytes.as_int_unchecked()),
            allocator,
            resizable) {}
// 默认移动赋值
  StorageImpl& operator=(StorageImpl&& other) = default;
// 禁用拷贝赋值
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
// 默认移动构造
  StorageImpl(StorageImpl&& other) = default;
// 禁用拷贝构造
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() override = default;

  void reset() {
    data_ptr_.clear();
    size_bytes_ = 0;
    size_bytes_is_symbolic_ = false;
  }

  template <typename T>
  inline T* data() const {
    return unsafe_data<T>();
  }
// 返回指向内存的指针并转换为 T*
  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  // 重写 intursive_ptr 的方法
  void release_resources() override {
    data_ptr_.clear();
  }

  size_t nbytes() const {
    TORCH_CHECK(!size_bytes_is_symbolic_);
    return size_bytes_.as_int_unchecked();
  }

  SymInt sym_nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
    size_bytes_is_symbolic_ = false;
  }

  void set_nbytes(c10::SymInt size_bytes) {
    size_bytes_ = std::move(size_bytes);
  }

  bool resizable() const {
    return resizable_;
  };

  at::DataPtr& data_ptr() {
    return data_ptr_;
  };

  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  };

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    at::DataPtr old_data_ptr(std::move(data_ptr_));
    data_ptr_ = std::move(data_ptr);
    return old_data_ptr;
  };

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    return data_ptr_.get();
  }

  void* data() const {
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  };

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

  Device device() const {
    return data_ptr_.device();
  }

  void set_resizable(bool resizable) {
    if (resizable) {
      // We need an allocator to be resizable
      AT_ASSERT(allocator_);
    }
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes) {
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    size_bytes_is_symbolic_ = false;
    allocator_ = nullptr;
    resizable_ = false;
  }

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }

 private:
 // 指针 指向内存
  DataPtr data_ptr_;
    // 表示内存大小（字节数）
  // 这里的 SymInt 类封装一个 int64_t 类型的成员变量
  SymInt size_bytes_;
  // size_bytes_ 越界标志
  bool size_bytes_is_symbolic_;

  // 标记内存能否被重分配
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  // 分配器指针
  Allocator* allocator_;
};
} // namespace c10
