#include "CudaNNUpsampleOp.h"

#include <chrono>

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

void CudaNNUpsampleOp::Compute(tensorflow::OpKernelContext* context) {
  const tensorflow::Tensor& input_tensor = context->input(0);
  const tensorflow::int32 channels = input_tensor.dim_size(1);
  const tensorflow::int32 height = input_tensor.dim_size(2);
  const tensorflow::int32 width = input_tensor.dim_size(3);

  //    const auto stream =
  //        static_cast<stream_executor::cuda::CUDAStream*>(
  //            context->op_device_context()->stream()->implementation())
  //            ->cuda_stream();
  //    LOG(INFO) << "::" << stream;
  
  // Create output tensor and perform the operation.
  tensorflow::TensorShape output_shape({1, channels, 2 * height, 2 * width});
  tensorflow::Tensor* output_tensor = nullptr;
  context->allocate_output(0, output_shape, &output_tensor);

  NNUpsample(channels, width, height, input_tensor.flat<float>().data(),
             output_tensor->flat<float>().data());
}
