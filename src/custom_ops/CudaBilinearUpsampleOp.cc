#include "CudaBilinearUpsampleOp.h"

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

void CudaBilinearUpsampleOp::Compute(tensorflow::OpKernelContext* context) {
  const tensorflow::Tensor& input_tensor = context->input(0);
  const tensorflow::int32 channels = input_tensor.dim_size(1);
  const tensorflow::int32 height = input_tensor.dim_size(2);
  const tensorflow::int32 width = input_tensor.dim_size(3);

  //    const auto stream =
  //        static_cast<stream_executor::cuda::CUDAStream*>(
  //            context->op_device_context()->stream()->implementation())
  //            ->cuda_stream();
  //    LOG(INFO) << "::" << stream;
  
  // Copy to a cudaArray object.
  cudaArray_t cuda_array;
  auto channel_desc = cudaCreateChannelDesc<float>();
  auto extent = make_cudaExtent(channels, width, height);
  cudaMalloc3DArray(&cuda_array, &channel_desc, extent, cudaArrayLayered);

  struct cudaMemcpy3DParms memcpy3D_params = {0};
  memcpy3D_params.srcPtr =
      make_cudaPitchedPtr(const_cast<float*>(input_tensor.flat<float>().data()),
                          width * sizeof(float), width, height);
  memcpy3D_params.dstArray = cuda_array;
  memcpy3D_params.kind = cudaMemcpyDeviceToDevice;
  memcpy3D_params.extent = extent;
  cudaMemcpy3DAsync(&memcpy3D_params);

  // Interface the array with texture memory.
  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = cuda_array;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;  // enables bilinear sampling
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t in_texture;
  cudaCreateTextureObject(&in_texture, &res_desc, &texDesc, nullptr);

  // Create output tensor and perform the operation.
  tensorflow::TensorShape output_shape({1, channels, 2 * height, 2 * width});
  tensorflow::Tensor* output_tensor = nullptr;
  context->allocate_output(0, output_shape, &output_tensor);

  BilinearUpsample(channels, width, height, in_texture,
                   output_tensor->flat<float>().data());

  cudaDestroyTextureObject(in_texture);
  cudaFreeArray(cuda_array);
}
