#include "CopyToTextureOp.h"

__global__ void CopyToTextureOp_CopyToTexture_kernel(const size_t width,
                                                     const size_t height,
                                                     const float* rgb,
                                                     cudaSurfaceObject_t rgba) {
  const size_t c = blockIdx.z;

  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  for (; y < height; y += blockDim.y * gridDim.y) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    for (; x < width; x += blockDim.x * gridDim.x) {
      const size_t offset = (c * height + y) * width + x;
      const float value = (c < 3) ? rgb[offset] : 1.f;

      surf2Dwrite(value, rgba, sizeof(float4) * x + sizeof(float) * c, y,
                  cudaBoundaryModeZero);
    }
  }
}

void CopyToTextureOp::CopyToTexture(const size_t width, const size_t height,
                                    const float* in_tensor,
                                    cudaSurfaceObject_t out_texture) {
  const dim3 block_dim(256, 1, 1);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                      (height + block_dim.y - 1) / block_dim.y, 4);
  // TODO (True): stream
  CopyToTextureOp_CopyToTexture_kernel<<<grid_dim, block_dim>>>(
      width, height, in_tensor, out_texture);
}
