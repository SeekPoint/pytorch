#include <c10/core/CopyBytes.h>
#include <c10/util/Logging.h>

namespace c10 {
/*
CopyBytesFunction表的初始化
Tensor的数据需要在设备与设备之间拷贝，拷贝的方法因设备而异，在PyTorch的C10系统中定义了一个global的g_copy_bytes三维数组，用来维护设备与copy function的mapping关系，如下所示：

static CopyBytesFunction g_copy_bytes[2][COMPILE_TIME_MAX_DEVICE_TYPES]
                                     [COMPILE_TIME_MAX_DEVICE_TYPES];
这个三维数组的第一维表示是同步（0）还是异步（1），第二维表示copy的源设备，第三维表示copy的目的设备，g_copy_bytes三维数组通过如下7个宏进行初始化：

REGISTER_COPY_BYTES_FUNCTION(DeviceType::CUDA,DeviceType::CUDA,caffe2::CUDAContext::CopyBytesSync,caffe2::CUDAContext::CopyBytesAsync);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CUDA,DeviceType::CPU,caffe2::CUDAContext::CopyBytesSync,caffe2::CUDAContext::CopyBytesAsync);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPU,DeviceType::CUDA,caffe2::CUDAContext::CopyBytesSync,caffe2::CUDAContext::CopyBytesAsync);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPU,DeviceType::CPU,caffe2::CopyBytesWrapper);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::IDEEP,DeviceType::CPU,CopyBytesWrapper);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::CPU,DeviceType::IDEEP,CopyBytesWrapper);
REGISTER_COPY_BYTES_FUNCTION(DeviceType::IDEEP,DeviceType::IDEEP,CopyBytesWrapper);
这就实现了数据可以从cuda到cpu、从cpu到cuda、从cuda到cuda、从cpu到cpu的拷贝。宏展开后如下所示：

#define REGISTER_COPY_BYTES_FUNCTION(from, to, ...)           \
  namespace {                                                 \
  static _CopyBytesFunctionRegisterer C10_ANONYMOUS_VARIABLE( \
      g_copy_function)(from, to, __VA_ARGS__);                \
  }

_CopyBytesFunctionRegisterer::_CopyBytesFunctionRegisterer(
    DeviceType fromType,
    DeviceType toType,
    CopyBytesFunction func_sync,
    CopyBytesFunction func_async) {
  auto from = static_cast<int>(fromType);
  auto to = static_cast<int>(toType);
  if (!func_async) {
    // default to the sync function
    func_async = func_sync;
  }
  g_copy_bytes[0][from][to] = func_sync;
  g_copy_bytes[1][from][to] = func_async;
}
如此通过global constructor便初始化完成了g_copy_bytes数组。
*/
// First dimension of the array is `bool async`: 0 is sync,
// 1 is async (non-blocking)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static CopyBytesFunction g_copy_bytes[2][COMPILE_TIME_MAX_DEVICE_TYPES]
                                     [COMPILE_TIME_MAX_DEVICE_TYPES];

_CopyBytesFunctionRegisterer::_CopyBytesFunctionRegisterer(
    DeviceType fromType,
    DeviceType toType,
    CopyBytesFunction func_sync,
    CopyBytesFunction func_async) {
  auto from = static_cast<int>(fromType);
  auto to = static_cast<int>(toType);
  if (!func_async) {
    // default to the sync function
    func_async = func_sync;
  }
  CHECK(
      g_copy_bytes[0][from][to] == nullptr &&
      g_copy_bytes[1][from][to] == nullptr)
      << "Duplicate registration for device type pair "
      << c10::DeviceTypeName(fromType) << ", " << c10::DeviceTypeName(toType);
  g_copy_bytes[0][from][to] = func_sync;
  g_copy_bytes[1][from][to] = func_async;
}

void CopyBytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async) {
  auto ptr = g_copy_bytes[async ? 1 : 0][static_cast<int>(src_device.type())]
                         [static_cast<int>(dst_device.type())];
  CAFFE_ENFORCE(
      ptr,
      "No function found for copying from ",
      c10::DeviceTypeName(src_device.type()),
      " to ",
      c10::DeviceTypeName(dst_device.type()));
  ptr(nbytes, src, src_device, dst, dst_device);
}

} // namespace c10
