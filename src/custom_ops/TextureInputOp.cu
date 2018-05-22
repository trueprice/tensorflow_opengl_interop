#include "TextureInputOp.h"

__global__ void TextureInputOp_CopyToTexture_kernel(
    const size_t width, const size_t height, const cudaTextureObject_t rgba,
    float* rgb) {
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  for (; y < height; y += blockDim.y * gridDim.y) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    for (; x < width; x += blockDim.x * gridDim.x) {
      const size_t offset = (y * width + x) * 3;
      const float4 value = tex2D<float4>(rgba, x, y);  // in [0, 1]

      rgb[offset] = value.x;
      rgb[offset + 1] = value.y;
      rgb[offset + 2] = value.z;
    }
  }
}

void TextureInputOp::CopyToTensor(const size_t width, const size_t height,
                                  cudaTextureObject_t in_texture,
                                  float* out_tensor) {
  const dim3 block_dim(256, 1, 1);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, height, 1);
  // TODO (True): stream
  TextureInputOp_CopyToTexture_kernel
      <<<grid_dim, block_dim>>>(width, height, in_texture, out_tensor);
}
