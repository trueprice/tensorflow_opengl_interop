#include "TextureInputOp.h"

__global__ void TextureInputOp_CopyToTexture_kernel(
    const size_t width, const size_t height,
    const cudaTextureObject_t rgba_0,
    const cudaTextureObject_t rgba_1,
    const cudaTextureObject_t rgba_2,
    const cudaTextureObject_t rgba_3,
    const cudaTextureObject_t rgba_4,
    float* rgb) {
  const cudaTextureObject_t* textures[5] = {
    &rgba_0, &rgba_1, &rgba_2, &rgba_3, &rgba_4
  };

  for (int tex_id = 0; tex_id < 5; ++tex_id)
  for (int channel_id = 0; channel_id < 3; ++channel_id)
  {
    size_t offset_texture = tex_id * 3;
    size_t offset_channel = (offset_texture + channel_id) * width * height;

    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; y < height; y += blockDim.y * gridDim.y) {
      size_t offset_row = offset_channel + y * width;

      size_t x = blockIdx.x * blockDim.x + threadIdx.x;
      for (; x < width; x += blockDim.x * gridDim.x) {
        const float4 value = tex2D<float4>((*textures[tex_id]), x, y);  // in [0, 1]
        const size_t offset = offset_row + x;

        rgb[offset] = *(((float*)&value) + channel_id);
      }
    }
  }
}

void TextureInputOp::CopyToTensor(const size_t width, const size_t height,
				  cudaTextureObject_t in_texture_a,
				  cudaTextureObject_t in_texture_b,
				  cudaTextureObject_t in_texture_c,
				  cudaTextureObject_t in_texture_d,
				  cudaTextureObject_t in_texture_e,
                                  float* out_tensor) {
  const dim3 block_dim(32, 32, 1);
  const dim3 grid_dim(32, 32);
  // TODO (True): stream
  TextureInputOp_CopyToTexture_kernel
      <<<grid_dim, block_dim>>>(width, height,
				in_texture_a, in_texture_b, in_texture_c, in_texture_d, in_texture_e,
				out_tensor);
}
