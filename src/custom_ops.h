#include "tensorflow/core/framework/op.h"

#include "custom_ops/CopyToFramebufferOp.h"

REGISTER_OP("CopyToFramebuffer")
    .Attr("framebuffer_ptr: int")
    .Attr("GLFWwindow_ptr: int")
    .Input("in_tensor: float");

REGISTER_KERNEL_BUILDER(Name("CopyToFramebuffer")
                            .Device(tensorflow::DEVICE_GPU),
                        CopyToFramebufferOp);
