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

  context->GetAttr(
      "texture_ids",
      reinterpret_cast<std::vector<tensorflow::int32>*>(&textureIds_));

  numInputs_ = textureIds_.size();

  context->GetAttr("shape", &shape_);

  glfwMakeContextCurrent(window_);
  cudaTextures_.resize(numInputs_);
  for (size_t i = 0; i < numInputs_; ++i) {
    cudaGraphicsGLRegisterImage(&cudaTextures_[i], textureIds_[i],
                                GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
  }
  glfwMakeContextCurrent(0);
}

TextureInputOp::~TextureInputOp() {
  glfwMakeContextCurrent(window_);
  for (size_t i = 0; i < numInputs_; ++i) {
    cudaGraphicsUnregisterResource(cudaTextures_[i]);
  }
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

  cudaGraphicsMapResources(numInputs_, cudaTextures_.data());  //, stream);

  std::vector<cudaTextureObject_t> in_textures(numInputs_);

  for (size_t i = 0; i < numInputs_; ++i) {
    cudaArray_t texture_array;
    cudaGraphicsSubResourceGetMappedArray(&texture_array, cudaTextures_[i], 0,
                                          0);

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
    //texDesc.readMode = cudaReadModeNormalizedFloat;  // converts to [0., 1.]
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&in_textures[i], &res_desc, &texDesc, nullptr);
  }

  tensorflow::Tensor* output_tensor = nullptr;
  context->allocate_output(0, shape_, &output_tensor);

  CopyToTensor(shape_.dim_size(3), shape_.dim_size(2), in_textures,
               output_tensor->flat<float>().data());


  for (size_t i = 0; i < numInputs_; ++i) {
    cudaDestroyTextureObject(in_textures[i]);
  }

  cudaGraphicsUnmapResources(numInputs_, cudaTextures_.data());  //, stream);
}
