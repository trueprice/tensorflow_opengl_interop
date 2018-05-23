#include "tensorflow/core/framework/op.h"

#include "custom_ops/CopyToTextureOp.h"
#include "custom_ops/CudaBilinearUpsampleOp.h"
#include "custom_ops/TextureInputOp.h"

REGISTER_OP("CopyToTexture")
    .Attr("GLFWwindow_ptr: int")
    .Attr("texture_id: int")
    .Input("in_tensor: float");

REGISTER_KERNEL_BUILDER(Name("CopyToTexture").Device(tensorflow::DEVICE_GPU),
                        CopyToTextureOp);

REGISTER_OP("CudaBilinearUpsample")
    .Input("in_tensor: float")
    .Output("out_tensor: float");

REGISTER_KERNEL_BUILDER(Name("CudaBilinearUpsample")
                            .Device(tensorflow::DEVICE_GPU),
                        CudaBilinearUpsampleOp);

REGISTER_OP("TextureInput")
    .Attr("GLFWwindow_ptr: int")
    .Attr("texture_ids: list(int)")
    .Attr("shape: shape")
    .Output("out_tensor: float");

REGISTER_KERNEL_BUILDER(Name("TextureInput").Device(tensorflow::DEVICE_GPU),
                        TextureInputOp);
