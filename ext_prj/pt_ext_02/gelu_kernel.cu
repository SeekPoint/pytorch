作者：旅行大喵
链接：https://www.zhihu.com/question/354886894/answer/888033350
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define BLOCKSIZE 1024

__device__ float sigmoid(float x) {
    return 1.0/(1+expf(-x));
}

__global__ void gelu_fwd_cuda(float* input, float* ret,
                              int64_t size) {
    int64_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        ret[idx] = input[idx]*sigmoid(1.702*input[idx]);
    }
}

__global__ void gelu_bwd_cuda(float* grad_out, float* input,
                              float* ret, int64_t size) {
    int64_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        float tmp = sigmoid(1.702*input[idx]);
        ret[idx] = grad_out[idx]*(tmp + 1.702*input[idx]*tmp*(1-tmp));
    }
}

__host__ void gelu_fwd_interface(float* input, float* ret, int64_t size) {
    int64_t nblock = (size + BLOCKSIZE - 1)/BLOCKSIZE;
    // nblock = 1;
    // printf("%d, %d", nblock, BLOCKSIZE);
    // gpuErrchk(cudaPeekAtLastError());
    gelu_fwd_cuda<<<nblock, BLOCKSIZE>>>(input, ret, size);
    gpuErrchk(cudaPeekAtLastError());
}

__host__ void gelu_bwd_interface(float* grad_out, float* input, float* ret,
                        int64_t size) {
    int64_t nblock = (size + BLOCKSIZE - 1)/BLOCKSIZE;
    // nblock = 1;
    // printf("%d, %d", nblock, BLOCKSIZE);
    // gpuErrchk(cudaPeekAtLastError());
    gelu_bwd_cuda<<<nblock, BLOCKSIZE>>>(grad_out, input,
                                         ret, size);
    gpuErrchk(cudaPeekAtLastError());
}