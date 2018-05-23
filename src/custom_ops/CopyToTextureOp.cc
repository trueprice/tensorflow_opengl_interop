#include "CopyToTextureOp.h"

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

CopyToTextureOp::CopyToTextureOp(tensorflow::OpKernelConstruction* context)
    : tensorflow::OpKernel(context) {
  tensorflow::int64 value;
  context->GetAttr("GLFWwindow_ptr", &value);
  window_ = reinterpret_cast<GLFWwindow*>(value);

  context->GetAttr("texture_id",
                   reinterpret_cast<tensorflow::int32*>(&texture_id_));

  glfwMakeContextCurrent(window_);
  cudaGraphicsGLRegisterImage(&cudaTexture_, texture_id_, GL_TEXTURE_2D,
                              cudaGraphicsMapFlagsWriteDiscard);
  glfwMakeContextCurrent(0);
}

CopyToTextureOp::~CopyToTextureOp() {
  glfwMakeContextCurrent(window_);
  cudaGraphicsUnregisterResource(cudaTexture_);
  glfwMakeContextCurrent(0);
  CUDA_CHECK_ERROR
}

void CopyToTextureOp::Compute(tensorflow::OpKernelContext* context) {
  const tensorflow::Tensor& input_tensor = context->input(0);
  const size_t height = input_tensor.dim_size(2);
  const size_t width = input_tensor.dim_size(3);

  //    const auto stream =
  //        static_cast<stream_executor::cuda::CUDAStream*>(
  //            context->op_device_context()->stream()->implementation())
  //            ->cuda_stream();
  //    LOG(INFO) << "::" << stream;

  cudaDeviceSynchronize();

  cudaGraphicsMapResources(1, &cudaTexture_);  //, stream);

  cudaArray_t texture_array;
  cudaGraphicsSubResourceGetMappedArray(&texture_array, cudaTexture_, 0, 0);

  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = texture_array;

  cudaSurfaceObject_t out_surface;
  cudaCreateSurfaceObject(&out_surface, &res_desc);

  CopyToTexture(width, height, input_tensor.flat<float>().data(), out_surface);

  cudaDestroySurfaceObject(out_surface);

  cudaGraphicsUnmapResources(1, &cudaTexture_);  //, stream);

  glfwMakeContextCurrent(0);
}
