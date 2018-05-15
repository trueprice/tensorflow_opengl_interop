#include "CopyToFramebufferOp.h"

__device__ uchar rgb_float_to_uchar(float x) {
  return __saturatef(x) * 255.f;
}

__global__ void CopyToFramebufferOp_CopyToTexture_kernel(
    const size_t width, const size_t height, const float* rgb,
    cudaSurfaceObject_t rgba) {
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  for (; y < height; y += blockDim.y * gridDim.y) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    for (; x < width; x += blockDim.x * gridDim.x) {
      const size_t offset = (y * width + x) * 3;
      const uchar4 value = make_uchar4(
          rgb_float_to_uchar(rgb[offset]),
          rgb_float_to_uchar(rgb[offset + 1]),
          rgb_float_to_uchar(rgb[offset + 2]),
          255);

      surf2Dwrite(value, rgba, sizeof(uchar4) * x, y, cudaBoundaryModeZero);
    }
  }
}

void CopyToFramebufferOp::CopyToTexture(const size_t width, const size_t height,
                                        const float* rgb,
                                        cudaSurfaceObject_t rgba) {
  const dim3 block_dim(32, 1);
  const dim3 grid_dim(32, 32);
  // TODO (True): stream
  CopyToFramebufferOp_CopyToTexture_kernel
      <<<grid_dim, block_dim>>>(width, height, rgb, rgba);
}