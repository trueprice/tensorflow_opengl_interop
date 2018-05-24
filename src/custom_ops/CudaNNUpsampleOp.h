#ifndef CUDA_NN_UPSAMPLE_OP_H_
#define CUDA_NN_UPSAMPLE_OP_H_

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

class CudaNNUpsampleOp : public tensorflow::OpKernel {
 public:
  explicit CudaNNUpsampleOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override;

  static void NNUpsample(const size_t channels, const size_t width,
                         const size_t height, const float* input,
                         float* output);
};

#endif  // CUDA_NN_UPSAMPLE_OP_H_
