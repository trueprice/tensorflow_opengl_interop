#ifndef TEXTURE_INPUTS_OP_H_
#define TEXTURE_INPUTS_OP_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class TextureInputOp : public tensorflow::OpKernel {
 public:
  explicit TextureInputOp(tensorflow::OpKernelConstruction* context);

  ~TextureInputOp();

  void Compute(tensorflow::OpKernelContext* context) override;

  static void CopyToTensor(const size_t width, const size_t height,
                           cudaTextureObject_t in_texture, float* out_tensor);

 private:
  GLuint texture_id_;
  GLFWwindow* window_;
  tensorflow::TensorShape shape_;
  cudaGraphicsResource_t cudaTexture_;
};

#endif  // TEXTURE_INPUTS_OP_H_
