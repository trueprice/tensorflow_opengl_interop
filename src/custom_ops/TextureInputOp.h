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
                           cudaTextureObject_t in_texture_a,
                           cudaTextureObject_t in_texture_b,
			   cudaTextureObject_t in_texture_c,
			   cudaTextureObject_t in_texture_d,
			   cudaTextureObject_t in_texture_e,
			   float* out_tensor);

 private:
  GLuint texture_id_a_;
  GLuint texture_id_b_;
  GLuint texture_id_c_;
  GLuint texture_id_d_;
  GLuint texture_id_e_;
  GLFWwindow* window_;
  tensorflow::TensorShape shape_;
  cudaGraphicsResource_t cudaTexture_a_;
  cudaGraphicsResource_t cudaTexture_b_;
  cudaGraphicsResource_t cudaTexture_c_;
  cudaGraphicsResource_t cudaTexture_d_;
  cudaGraphicsResource_t cudaTexture_e_;
};

#endif  // TEXTURE_INPUTS_OP_H_
