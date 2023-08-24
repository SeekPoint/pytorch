#pragma once
#include <memory>

#include <c10/macros/Macros.h>

namespace c10 {
// 定义 DeleterFnPtr 为一个函数指针，该函数接受一个 void* 类型的参数并返回 void
// 删除器 作用是释放 void* 类型指针所指向的内存块。
using DeleterFnPtr = void (*)(void*);

namespace detail {

// Does not delete anything
// 默认删除器
C10_API void deleteNothing(void*);

// A detail::UniqueVoidPtr is an owning smart pointer like unique_ptr, but
// with three major differences:
//
//    1) It is specialized to void
//
//    2) It is specialized for a function pointer deleter
//       void(void* ctx); i.e., the deleter doesn't take a
//       reference to the data, just to a context pointer
//       (erased as void*).  In fact, internally, this pointer
//       is implemented as having an owning reference to
//       context, and a non-owning reference to data; this is why
//       you release_context(), not release() (the conventional
//       API for release() wouldn't give you enough information
//       to properly dispose of the object later.)
//
//    3) The deleter is guaranteed to be called when the unique
//       pointer is destructed and the context is non-null; this is different
//       from std::unique_ptr where the deleter is not called if the
//       data pointer is null.
//
// Some of the methods have slightly different types than std::unique_ptr
// to reflect this.
//
//这个类位于最底层，用来直接维护tensor所需的内存。
*/
class UniqueVoidPtr {
 private:
  // Lifetime tied to ctx_
  void* data_;
  // 用于管理与 data_ 相关的上下文，其中包含一个删除器指针。
  std::unique_ptr<void, DeleterFnPtr> ctx_;

 public:
  UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, &deleteNothing) {}
  explicit UniqueVoidPtr(void* data)
      : data_(data), ctx_(nullptr, &deleteNothing) {}
  UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)
      : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing) {}
  void* operator->() const {
    return data_;
  }
  void clear() {
    ctx_ = nullptr;
    data_ = nullptr;
  }
  void* get() const {
    return data_;
  }
  void* get_context() const {
    return ctx_.get();
  }
  void* release_context() {
    return ctx_.release();
  }
  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return std::move(ctx_);
  }
  C10_NODISCARD bool compare_exchange_deleter(
      DeleterFnPtr expected_deleter,
      DeleterFnPtr new_deleter) {
    if (get_deleter() != expected_deleter)
      return false;
    ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
    return true;
  }

  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    if (get_deleter() != expected_deleter)
      return nullptr;
    return static_cast<T*>(get_context());
  }
  operator bool() const {
    return data_ || ctx_;
  }
  DeleterFnPtr get_deleter() const {
    return ctx_.get_deleter();
  }
};
/*
在pytorch/c10/util/UniqueVoidPtr.h文件中定义了UniqueVoidPtr类，相当于C++中的unique_ptr。

unique_ptr是C++标准库中的一个智能指针，用于自动管理动态分配的对象的生命周期。
它在单个编译单元内通常可以很好地工作，但在跨编译单元边界时，由于不同编译单元的内存管理可能不一致，可能会导致内存泄漏或者释放错误的内存。
因此，为了避免这些问题，PyTorch选择将unique_ptr封装为UniqueVoidPtr来进行统一的内存管理。

UniqueVoidPtr提供了一个统一的接口来管理内存资源，特别是在涉及到跨编译单元的情况下，可以更好地保证内存的正确释放。
通过使用UniqueVoidPtr，PyTorch可以自行管理内存资源，并在需要释放资源时进行正确的处理，从而避免内存泄漏等问题。

此外，UniqueVoidPtr还提供了更多的灵活性，因为它可以持有任意类型的内存块，而不仅仅局限于特定的数据类型。
这在处理不同类型的资源时非常有用。详见注释：
*/
// Note [How UniqueVoidPtr is implemented]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UniqueVoidPtr solves a common problem for allocators of tensor data, which
// is that the data pointer (e.g., float*) which you are interested in, is not
// the same as the context pointer (e.g., DLManagedTensor) which you need
// to actually deallocate the data.  Under a conventional deleter design, you
// have to store extra context in the deleter itself so that you can actually
// delete the right thing.  Implementing this with standard C++ is somewhat
// error-prone: if you use a std::unique_ptr to manage tensors, the deleter will
// not be called if the data pointer is nullptr, which can cause a leak if the
// context pointer is non-null (and the deleter is responsible for freeing both
// the data pointer and the context pointer).
//
// So, in our reimplementation of unique_ptr, which just store the context
// directly in the unique pointer, and attach the deleter to the context
// pointer itself.  In simple cases, the context pointer is just the pointer
// itself.
/*
与intrusive_ptr 不同的是，UniqueVoidPtr并没有采用模版编程，而是将数据指针声明为void* data_，即data_可以指向任意数据类型连续内存的起始地址。
UniqueVoidPtr.h 中还定义了负责释放void*类型指针的删除器。
*/

inline bool operator==(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return !sp;
}
inline bool operator==(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return !sp;
}
inline bool operator!=(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return sp;
}
inline bool operator!=(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return sp;
}

} // namespace detail
} // namespace c10
