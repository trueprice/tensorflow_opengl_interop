#include "CudaBilinearUpsampleOp.h"

__global__ void CudaBilinearUpsampleOp_BilinearUpsample_kernel(
    const size_t channels, const size_t width, const size_t height,
    cudaTextureObject_t input, float* output) {
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;

  for (; z < channels; z += blockDim.z * gridDim.z) {
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    for (; y < height; y += blockDim.y * gridDim.y) {
      size_t x = blockIdx.x * blockDim.x + threadIdx.x;

      for (; x < width; x += blockDim.x * gridDim.x) {
        const size_t offset = (z * height + y) * width + x;
        output[offset] = tex2DLayered<float>(input, x * 0.5f, y * 0.5f, z);
      }
    }
  }
}

void CudaBilinearUpsampleOp::BilinearUpsample(
    const size_t channels, const size_t width, const size_t height,
    cudaTextureObject_t input, float* output) {
  const dim3 block_dim(32, 1, 1);
  const dim3 grid_dim(32, 32, 32);
  // TODO (True): stream
  CudaBilinearUpsampleOp_BilinearUpsample_kernel<<<grid_dim, block_dim>>>
      (channels, width, height, input, output);
}
