#ifndef TEXTURE_INPUTS_OP_H_
#define TEXTURE_INPUTS_OP_H_

#include <array>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class TextureInputOp : public tensorflow::OpKernel {
 public:
  static const size_t NUM_INPUTS = 5;

  explicit TextureInputOp(tensorflow::OpKernelConstruction* context);

  ~TextureInputOp();

  void Compute(tensorflow::OpKernelContext* context) override;

  static void CopyToTensor(
      const size_t width, const size_t height,
      const std::array<cudaTextureObject_t, NUM_INPUTS>& in_textures,
      float* out_tensor);

 private:
  std::vector<GLuint> texture_ids_;
  std::array<cudaGraphicsResource_t, NUM_INPUTS> cudaTextures_;
  GLFWwindow* window_;
  tensorflow::TensorShape shape_;
};

#endif  // TEXTURE_INPUTS_OP_H_
