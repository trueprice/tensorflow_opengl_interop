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

  context->GetAttr("texture_id",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_));

  context->GetAttr("shape", &shape_);

  glfwMakeContextCurrent(window_);
  cudaGraphicsGLRegisterImage(&cudaTexture_, texture_id_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsReadOnly);
  glfwMakeContextCurrent(0);
}

TextureInputOp::~TextureInputOp() {
  glfwMakeContextCurrent(window_);
  cudaGraphicsUnregisterResource(cudaTexture_);
  glfwMakeContextCurrent(0);
  CUDA_CHECK_ERROR
}

void TextureInputOp::Compute(tensorflow::OpKernelContext* context) {
  //    const auto stream =
  //        static_cast<stream_executor::cuda::CUDAStream*>(
  //            context->op_device_context()->stream()->implementation())
  //            ->cuda_stream();
  //    LOG(INFO) << "::" << stream << std::endl;

cudaDeviceSynchronize();

  if (glfwGetCurrentContext() != window_) {
    glfwMakeContextCurrent(window_);
  }

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

  cudaTextureObject_t in_texture;
  cudaCreateTextureObject(&in_texture, &res_desc, &texDesc, nullptr);

  tensorflow::Tensor* output_tensor = nullptr;
  context->allocate_output(0, shape_, &output_tensor);

  CopyToTensor(shape_.dim_size(1), shape_.dim_size(0), in_texture,
               output_tensor->flat<float>().data());

  cudaDestroyTextureObject(in_texture);

  cudaGraphicsUnmapResources(1, &cudaTexture_);  //, stream);
}
