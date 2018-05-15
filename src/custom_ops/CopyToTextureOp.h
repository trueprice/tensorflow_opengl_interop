#ifndef COPY_TO_TEXTURE_OP_H_
#define COPY_TO_TEXTURE_OP_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class CopyToTextureOp : public tensorflow::OpKernel {
 public:
  explicit CopyToTextureOp(tensorflow::OpKernelConstruction* context);

  ~CopyToTextureOp();

  void Compute(tensorflow::OpKernelContext* context) override;

  static void CopyToTexture(const size_t width, const size_t height,
                            const float* in_tensor,
                            cudaSurfaceObject_t out_texture);

 private:
  GLuint texture_id_;
  GLFWwindow* window_;
  cudaGraphicsResource_t cudaTexture_;
};

#endif // COPY_TO_TEXTURE_OP_H_
