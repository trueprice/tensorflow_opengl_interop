#ifndef COPY_TO_FRAMEBUFFER_OP_H_
#define COPY_TO_FRAMEBUFFER_OP_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "gl_wrappers.h"

class CopyToFramebufferOp : public tensorflow::OpKernel {
 public:
  explicit CopyToFramebufferOp(tensorflow::OpKernelConstruction* context);

  ~CopyToFramebufferOp();

  void Compute(tensorflow::OpKernelContext* context) override;

  static void CopyToTexture(const size_t width, const size_t height,
                            const float* rgb, cudaSurfaceObject_t rgba);

 private:
  fribr::Framebuffer* framebuffer_;
  GLFWwindow* window_;
  struct cudaGraphicsResource* cudaFramebuffer_;
};

#endif // COPY_TO_FRAMEBUFFER_OP_H_
