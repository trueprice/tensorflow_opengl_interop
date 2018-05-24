#include "CudaNNUpsampleOp.h"

__global__ void CudaNNUpsampleOp_NNUpsample_kernel(const unsigned int width,
                                                   const unsigned int height,
                                                   const float* input,
                                                   float* output) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x > width) {
    return;
  }

  extern __shared__ float values[];

  const unsigned int c = blockIdx.z;
  const unsigned int y = blockIdx.y;

  values[threadIdx.x] = input[(c * height + y) * width + x];

  __syncthreads();

  unsigned int offset =
      2 * ((2 * (c * height + y)) * width + blockDim.x * blockIdx.x);
  output[offset + threadIdx.x] = values[threadIdx.x >> 1];
  output[offset + 2 * width + threadIdx.x] = values[threadIdx.x >> 1];

  const unsigned int n = min(blockDim.x, width - blockDim.x * blockIdx.x);
  offset += n;
  output[offset + threadIdx.x] = values[(n >> 1) + (threadIdx.x >> 1)];
  output[offset + 2 * width + threadIdx.x] =
      values[(n >> 1) + (threadIdx.x >> 1)];
}

void CudaNNUpsampleOp::NNUpsample(const size_t channels, const size_t width,
                                  const size_t height, const float* input,
                                  float* output) {
  const dim3 block_dim(256, 1, 1);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, height,
                      channels);
  // TODO (True): stream
  const size_t shmem_size = block_dim.x * sizeof(float);
  CudaNNUpsampleOp_NNUpsample_kernel<<<grid_dim, block_dim, shmem_size>>>
      (width, height, input, output);
}
