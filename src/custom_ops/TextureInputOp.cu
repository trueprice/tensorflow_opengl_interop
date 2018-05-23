#include "TextureInputOp.h"

#include <array>

__constant__ cudaTextureObject_t c_RGBAInputs[TextureInputOp::NUM_INPUTS];

__global__ void TextureInputOp_CopyToTexture_kernel(const size_t width,
                                                    const size_t height,
                                                    float* rgb) {
  cudaTextureObject_t t = c_RGBAInputs[blockIdx.z];

  for (size_t c = 0; c < 3; ++c) {
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    for (; y < height; y += blockDim.y * gridDim.y) {
      size_t x = blockIdx.x * blockDim.x + threadIdx.x;

      for (; x < width; x += blockDim.x * gridDim.x) {
        const size_t offset = ((blockIdx.z * 3 + c) * height + y) * width + x;
        const float4 value = tex2D<float4>(t, x, y);  // in [0, 1]

        rgb[offset] = *(reinterpret_cast<const float*>(&value) + c);
      }
    }
  }
}

void TextureInputOp::CopyToTensor(
    const size_t width, const size_t height,
    const std::array<cudaTextureObject_t, TextureInputOp::NUM_INPUTS>&
        in_textures,
    float* out_tensor) {
  const dim3 block_dim(256, 1, 1);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                      (height + block_dim.y - 1) / block_dim.y,
                      TextureInputOp::NUM_INPUTS);

  cudaMemcpyToSymbol(c_RGBAInputs, in_textures.data(),
                     NUM_INPUTS * sizeof(cudaTextureObject_t));

  // TODO (True): stream
  TextureInputOp_CopyToTexture_kernel
      <<<grid_dim, block_dim>>>(width, height, out_tensor);
}
