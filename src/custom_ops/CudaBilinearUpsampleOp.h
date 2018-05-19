#ifndef CUDA_BILINEAR_UPSAMPLE_OP_H_
#define CUDA_BILINEAR_UPSAMPLE_OP_H_

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class CudaBilinearUpsampleOp : public tensorflow::OpKernel {
 public:
  explicit CudaBilinearUpsampleOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override;

  static void BilinearUpsample(const size_t channels, const size_t width,
                               const size_t height, cudaTextureObject_t input,
                               float* output);
};

#endif  // CUDA_BILINEAR_UPSAMPLE_OP_H_
