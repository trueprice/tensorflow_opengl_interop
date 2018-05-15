#include "tensorflow/core/framework/op.h"

#include "custom_ops/CopyToTextureOp.h"
#include "custom_ops/TextureInputsOp.h"

REGISTER_OP("CopyToTexture")
    .Attr("GLFWwindow_ptr: int")
    .Attr("texture_id: int")
    .Input("in_tensor: float");

REGISTER_KERNEL_BUILDER(Name("CopyToTexture").Device(tensorflow::DEVICE_GPU),
                        CopyToTextureOp);

/*
REGISTER_OP("TextureInputs")
    .Attr("GLFWwindow_ptr: int")
    .Attr("texture_ids: list(int)")
    .Attr("output_shapes: list(shape)")
    .Output("out_tensors: list(float)");

REGISTER_KERNEL_BUILDER(Name("TextureInputs").Device(tensorflow::DEVICE_GPU),
                        TextureInputs);
*/
