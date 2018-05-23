#include "TextureInputOp.h"

#include <cuda_gl_interop.h>

#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
#define CUDA_CHECK_ERROR CUDA_CHECK(cudaPeekAtLastError())

//------------------------------------------------------------------------------

TextureInputOp::TextureInputOp(tensorflow::OpKernelConstruction* context)
    : tensorflow::OpKernel(context) {
  tensorflow::int64 value;
  context->GetAttr("GLFWwindow_ptr", &value);
  window_ = reinterpret_cast<GLFWwindow*>(value);

  context->GetAttr("texture_id_0",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_a_));
  context->GetAttr("texture_id_1",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_b_));
  context->GetAttr("texture_id_2",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_c_));
  context->GetAttr("texture_id_3",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_d_));
  context->GetAttr("texture_id_4",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_e_));

  context->GetAttr("shape", &shape_);

  glfwMakeContextCurrent(window_);
  cudaGraphicsGLRegisterImage(&cudaTexture_a_, texture_id_a_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsReadOnly);
  cudaGraphicsGLRegisterImage(&cudaTexture_b_, texture_id_b_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsReadOnly);
  cudaGraphicsGLRegisterImage(&cudaTexture_c_, texture_id_c_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsReadOnly);
  cudaGraphicsGLRegisterImage(&cudaTexture_d_, texture_id_d_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsReadOnly);
  cudaGraphicsGLRegisterImage(&cudaTexture_e_, texture_id_e_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsReadOnly);

  glfwMakeContextCurrent(0);
}

TextureInputOp::~TextureInputOp() {
  glfwMakeContextCurrent(window_);
  cudaGraphicsUnregisterResource(cudaTexture_a_);
  cudaGraphicsUnregisterResource(cudaTexture_b_);
  cudaGraphicsUnregisterResource(cudaTexture_c_);
  cudaGraphicsUnregisterResource(cudaTexture_d_);
  cudaGraphicsUnregisterResource(cudaTexture_e_);
  glfwMakeContextCurrent(0);
  CUDA_CHECK_ERROR
}

void TextureInputOp::Compute(tensorflow::OpKernelContext* context) {;
  //    const auto stream =
  //        static_cast<stream_executor::cuda::CUDAStream*>(
  //            context->op_device_context()->stream()->implementation())
  //            ->cuda_stream();
  //    LOG(INFO) << "::" << stream;

  cudaDeviceSynchronize();
  glfwMakeContextCurrent(window_);

  auto makeTextureObject = [&](cudaTextureObject_t& in_texture,
			       cudaGraphicsResource_t& cudaTexture_)
  {
    cudaGraphicsMapResources(1, &cudaTexture_);  //, stream);

    cudaArray_t texture_array;
    cudaGraphicsSubResourceGetMappedArray(&texture_array, cudaTexture_, 0, 0);

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = texture_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;  // converts to [0., 1.]

    cudaCreateTextureObject(&in_texture, &res_desc, &texDesc, nullptr);
  };

  cudaTextureObject_t in_texture_a, in_texture_b, in_texture_c, in_texture_d, in_texture_e;
  makeTextureObject(in_texture_a, cudaTexture_a_);
  makeTextureObject(in_texture_b, cudaTexture_b_);
  makeTextureObject(in_texture_c, cudaTexture_c_);
  makeTextureObject(in_texture_d, cudaTexture_d_);
  makeTextureObject(in_texture_e, cudaTexture_e_);

  tensorflow::Tensor* output_tensor = nullptr;
  context->allocate_output(0, shape_, &output_tensor);

  CopyToTensor(shape_.dim_size(3), shape_.dim_size(2),
	       in_texture_a, in_texture_b, in_texture_c, in_texture_d, in_texture_e,
               output_tensor->flat<float>().data());

  auto destroyTextureObject = [&](cudaTextureObject_t& in_texture,
				  cudaGraphicsResource_t& cudaTexture_)
  {
    cudaDestroyTextureObject(in_texture);
    cudaGraphicsUnmapResources(1, &cudaTexture_);  //, stream);
  };

  destroyTextureObject(in_texture_a, cudaTexture_a_);
  destroyTextureObject(in_texture_b, cudaTexture_b_);
  destroyTextureObject(in_texture_c, cudaTexture_c_);
  destroyTextureObject(in_texture_d, cudaTexture_d_);
  destroyTextureObject(in_texture_e, cudaTexture_e_);

  // No need to un-set the gl context here. CopyTextureOp will take care of it.
}
