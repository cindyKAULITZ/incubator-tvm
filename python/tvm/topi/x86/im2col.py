import logging

import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..nn.im2col import *
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.util import get_pad_tuple
from ..nn import pad, mirror_pad
from ..util import get_const_tuple, traverse_inline
from . import conv2d_avx_1x1, conv2d_avx_common

logger = logging.getLogger("topi")

def im2col_transform(data, strides, padding, dilation, channel,kernel_size, transform_tag, out_dtype):
    # default
    # packed_out = conv2d_NCHWc(data, kernel, strides, padding, dilation,
    #                           layout, layout, out_dtype)
    # return unpack_NCHWc_to_nchw(packed_out, out_dtype)
    
    # im2col
    packed_out = im2col_transform_compute(data, strides, padding, dilation, channel,kernel_size, transform_tag, out_dtype)
    return packed_out



@autotvm.register_topi_compute("im2col_transform_compute.x86")
def im2col_transform_compute(cfg, Input, strides, padding, dilation, channel, kernel_size, transform_tag, out_dtype=None):

    # if len(Input.shape) == 5:
    #     N, ic_chunk, ih, iw, ic_bn = get_const_tuple(Input.shape)
    #     oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
    #     in_channel = ic_chunk * ic_bn
    #     num_filter = oc_chunk * oc_bn
    # else:
    N, in_channel, ih, iw = get_const_tuple(Input.shape)
    num_filter=1
    ic=1 
    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    # padding
    HPAD = pt + pb
    WPAD = pl + pr
    
    dilation_h, dilation_w = dilation if isinstance(dilation, (tuple, list)) \
        else (dilation, dilation)

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1
    
    out_height = (ih + HPAD - dilated_kernel_h) // sh + 1
    out_width = (iw + WPAD - dilated_kernel_w) // sw + 1

    DO_PAD = (HPAD != 0 and WPAD != 0)
    if DO_PAD:
        data_pad = pad(Input, (0, 0, HPAD//2, WPAD//2), name="data_pad")
    else:
        data_pad = Input
    
    ALIGN = 16
    def upround(x, align):
        return (x + align - 1) // align * align
    reduce_len = upround(in_channel * kernel_height * kernel_width, ALIGN)


    if (transform_tag == "weight"):
         # A [CO, CI * KH * KW]
        A = te.compute((upround(num_filter, ALIGN), reduce_len), lambda i, j:
                        Input[i][j // kernel_width // kernel_height][j // kernel_width % kernel_height][j % kernel_width], name='A')
        return A
    elif (transform_tag == "data"):
        # B [CI * KH * KW, N * OH * OW]
        B = te.compute((reduce_len, upround(N * out_height * out_width, ALIGN)), lambda i, j:\
                    te.if_then_else(te.all(i < in_channel * kernel_height * kernel_width, j < N * out_height * out_width),
                    data_pad[j // (out_height*out_width)][i // (kernel_height*kernel_width)][j // out_width % out_height*sh + i // kernel_width % kernel_height]
                    [j % out_width*sw + i % kernel_width],
                    tvm.tir.const(0, data_pad.dtype)), name='B')
        return B

# def schedule_im2col_transform(outs):
#     """Create schedule for tensors"""
#     return schedule_im2col_transform_(outs)


# @autotvm.register_topi_schedule("im2col_transform.x86")
# def schedule_im2col_transform_(cfg, outs):
#     """Create schedule for tensors"""
#     target = tvm.target.Target.current(allow_none=False)
#     outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
#     if target.kind.name not in ("llvm", "c"):
#         raise RuntimeError("schedule not registered for '%s'" % target)
#     s = te.create_schedule([x.op for x in outs])
    
#     return s
