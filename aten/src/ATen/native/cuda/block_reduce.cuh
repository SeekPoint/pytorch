#pragma once

#include <thrust/tuple.h>

#include <ATen/native/SharedReduceOps.h>
#include <ATen/cuda/DeviceUtils.cuh>

namespace at {
namespace native {
namespace cuda_utils {

constexpr int kCUDABlockReduceNumThreads = 512;
// Algorithmic limitation: BlockReduce does two WarpReduce calls, each
// of which reduces C10_WARP_SIZE elements. So, at most
// C10_WARP_SIZE**2 elements can be reduced at a time.
// NOTE: This is >= the max block size on current hardware anyway (1024).
constexpr int kCUDABlockReduceMaxThreads = C10_WARP_SIZE * C10_WARP_SIZE;

// Sums `val` accross all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

struct Block1D {
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x / C10_WARP_SIZE;
    }
};

struct Block2D {
    static __forceinline__ __device__ int Tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x * blockDim.y / C10_WARP_SIZE;
    }
};

// Sums `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
'''
Pytorch CUDA源码解析 - BlockReduceSum
https://zhuanlan.zhihu.com/p/584936904

想要对Pytorch中的CUDA算子做一些学习和解读，加深一下CUDA软硬件知识。首选了Pytorch中比较常用的Reduce求和函数。本文需要读者有基础的CUDA知识，比如warp的概念等
lane 中文意思为车道，在CUDA表示一个 warp 中的 thread 个数，
在1D Block中在一个 lane 中的索引lane_index为 [0, warpSize - 1]。
 block 中会有多个lane，lane_id = threadIdx.x ，最多有 1024 / warpSize = 32个lane。

 BlockReduceSum 函数体分析：
(1) 通过 B::Tid() 获取 threadIdx.x，lid 表示当前 thread 在lane中的索引，wid表示上文中提到的lane_ID;
(2) 我们知道每个thread对应寄存器中都有一个val值，WarpReduceSum 函数便是对warp中的所有thread所持有的val进行求和，进一步分析 WarpReduceSum 做了什么。


为了简便来看，将宏WARP_SHFL_DOWN展开，并认为宏USE_ROCM未定义

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff， val, offset, warpSize);//warpSize为内置数据，值为32
  }
  return val;
}


这里使用了warp原语__shfl_down_sync来对一个warp内的val进行规约求和。
__shfl_down_sync的官方介绍warp-shuffle-functions 、 using-cuda-warp-level-primitives


shfl_down_sync()执行过程
函数原型：T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
mask表示一个warp中thread的激活表;
var表示规约求和的变量；
delta表示当前线程与另一个线程求和时跨越的线程偏移;
width表示求和的宽度（个数）
根据循环体for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1)，第一次循环时delta是16，即Lane0（Lane0表示一个warp中的lane_index）与Lane16求和，Lane1与Lane17求和，***，依此类推，一个warp内的32个thread的val规约成16个val。第二次循环时delta是8，即Lane0与Lane4求和，Lane1与Lane5，与上图的第一次行为相同。依次类推，最终不同warp中的val规约求和到了Lane0所持有的val中。

至此执行完 val = WarpReduceSum(val); 后，所有warp的和都规约到了每个warp中的lane0的线程中了，即lid == 0的线程，wid则代表了不同的lane（或不同的warp）。（这里留下一个待验证的东西，如果mask和width随着循环不断改变，性能是否会有细微提升呢？）

(3) 下面要将各个warp规约求和的值进行一次规约，需要通过shared-memory将数据保存到同一个warp中的不同线程中，在数据保存前需要__syncthreads(); 同步一下，为了防止当BlockReduceSum在一行中被调用引起的冲突（待理解后补充原因）

(4) 默认申请32大小的shared-memory（32其实也是一个Block最大的lane个数），当Block内线程束较少时，无法刷新shared-memory上全部的32个值，需要对未使用到的内存进行初始化；

(5) 梅开二度，再次使用 WarpReduceSum 对这32个thread的值进行求和，最终一个Block内的值便全部规约求和到了threadIdx.x == 0的线程所持有的val值了。这也就是说对于调用BlockReduceSum函数的代码来说，在使用规约求和后的值时需要通过threadIdx.x == 0的线程获取。


看下Pytorch算子如何实际使用 BlockReduceSum函数，下面是group_norm_backward中的部分代码(Compute1dBackwardFusedParamsCUDAKernel)：

  if (blockDim.x <= C10_WARP_SIZE) {
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE];
    __shared__ T_ACC db_shared[C10_WARP_SIZE];
    sum1 = cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D);
    const T_ACC x = (sum2 * static_cast<T_ACC>(mean[ng]) - sum1) *
        static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
        static_cast<T_ACC>(rstd[ng]) * s;
    c2[ng] = x;
    c3[ng] = -x * static_cast<T_ACC>(mean[ng]) -
        sum2 * static_cast<T_ACC>(rstd[ng]) * s;
  }
无须理会这个函数做了什么事情（因为工作原因刚好对group_norm_backward的CUDA实现进行过阅读，如果有小伙伴感兴趣的话可以留言），
可以看到在使用sum1和sum2时是通过 if (threadIdx.x == 0) 来完成的。

至此，便完成了Pytorch中BlockReduceSum函数的解析，第一次写文章，如有不对之处还请指出。


'''
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

} // namespace cuda_utils
} // namespace native
} // namespace at
