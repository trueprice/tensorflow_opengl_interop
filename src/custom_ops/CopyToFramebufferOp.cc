#include "CopyToFramebufferOp.h"
#include <cstdint>

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

CopyToFramebufferOp::CopyToFramebufferOp(
    tensorflow::OpKernelConstruction* context)
    : tensorflow::OpKernel(context) {
  tensorflow::int64 value;
  context->GetAttr("framebuffer_ptr", &value);
  framebuffer_ = reinterpret_cast<fribr::Framebuffer*>(value);

  context->GetAttr("GLFWwindow_ptr", &value);
  window_ = reinterpret_cast<GLFWwindow*>(value);

  //    framebuffer_->bind();
  glfwMakeContextCurrent(window_);
  cudaGraphicsGLRegisterImage(&cudaFramebuffer_,
                              framebuffer_->get_textures()[0]->get_id(),
                              GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
  glfwMakeContextCurrent(0);
  //    framebuffer_->unbind();
}

CopyToFramebufferOp::~CopyToFramebufferOp() {
  glfwMakeContextCurrent(window_);
  cudaGraphicsUnregisterResource(cudaFramebuffer_);
  glfwMakeContextCurrent(0);
  CUDA_CHECK_ERROR
}

void CopyToFramebufferOp::Compute(tensorflow::OpKernelContext* context) {
  const tensorflow::Tensor& input_tensor = context->input(0);
  const size_t height = input_tensor.dim_size(1);
  const size_t width = input_tensor.dim_size(2);

  //    const auto stream =
  //        static_cast<stream_executor::cuda::CUDAStream*>(
  //            context->op_device_context()->stream()->implementation())
  //            ->cuda_stream();
  //    LOG(INFO) << "::" << stream;

  glfwMakeContextCurrent(window_);

  // TODO (True): check whether the bind/unbind is necessary
  //framebuffer_->bind();
  cudaGraphicsMapResources(1, &cudaFramebuffer_);  //, stream);

  cudaArray_t fbo_data;
  cudaGraphicsSubResourceGetMappedArray(&fbo_data, cudaFramebuffer_, 0, 0);

  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = fbo_data;

  cudaSurfaceObject_t fbo_surface;
  cudaCreateSurfaceObject(&fbo_surface, &res_desc);

  CopyToTexture(width, height, input_tensor.flat<float>().data(), fbo_surface);

  cudaDestroySurfaceObject(fbo_surface);

  cudaGraphicsUnmapResources(1, &cudaFramebuffer_);  //, stream);
  //framebuffer_->unbind();

  glfwMakeContextCurrent(0);
}
