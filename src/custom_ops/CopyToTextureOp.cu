#include "CopyToTextureOp.h"

__device__ unsigned char rgb_float_to_uchar(float x) {
  return __saturatef(x) * 255.f;
}

__global__ void CopyToTextureOp_CopyToTexture_kernel(const size_t width,
                                                     const size_t height,
                                                     const float* rgb,
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

void CopyToTextureOp::CopyToTexture(const size_t width, const size_t height,
                                    const float* in_tensor,
                                    cudaSurfaceObject_t out_texture) {
  const dim3 block_dim(256, 1, 1);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, height, 1);
  // TODO (True): stream
  CopyToTextureOp_CopyToTexture_kernel<<<grid_dim, block_dim>>>(
      width, height, in_tensor, out_texture);
}
