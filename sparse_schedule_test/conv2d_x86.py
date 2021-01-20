"""
The list of available gluon models are here
https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html
So far, we confirmed that the model alexnet, resnet18_v1, resnet50_v1 can be loaded to nnvm
Other models do not work with NNVM at the moment.
See this issue https://github.com/dmlc/nnvm/issues/203.
"""

import mxnet as mx
# import nnvm
import onnx
import tvm
import numpy as np
import struct
# import tvm
# # import tvm.relay as relay
# # import tvm.topi
# from tvm.topi.util import get_const_int, simplify, const_matrix, get_const_tuple
# from tvm.topi.nn.util import get_pad_tuple
# from tvm.topi.nn.pad import pad
# from tvm.topi.nn.conv2d import conv2d_hwcn, conv2d_nhwc

import os
# import tvm
import time
import itertools
import numpy as np
# import tensorflow as tf
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import data_dep_optimization as ddo

# from mxnet.gluon.model_zoo.vision import get_model
# from mxnet.gluon.utils import download
from PIL import Image
# from matplotlib import pyplot as plt

############################################
# Get Model and open some label file


# def int_sigmoid(op):
#     return topi.cast(((256*256*4*256) / (256 + tvm.call_extern("int16","riscv_int_exp",-op.args[0])))>>8,"int16")

# def compute_conv2d(attrs, inputs, _):

#     # @tvm.target.generic_func
#     def _conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
#         """Convolution operator in NCHW layout.

#         Parameters
#         ----------
#         Input : tvm.Tensor
#             4-D with shape [batch, in_channel, in_height, in_width]

#         Filter : tvm.Tensor
#             4-D with shape [num_filter, in_channel, filter_height, filter_width]

#         stride : int or a list/tuple of two ints
#             Stride size, or [stride_height, stride_width]

#         padding : int or str
#             Padding size, or ['VALID', 'SAME']

#         dilation: int or a list/tuple of two ints
#             dilation size, or [dilation_height, dilation_width]

#         Returns
#         -------
#         Output : tvm.Tensor
#             4-D with shape [batch, out_channel, out_height, out_width]
#         """
#         if out_dtype is None:
#             out_dtype = Input.dtype
#         assert isinstance(stride, int) or len(stride) == 2
#         assert isinstance(dilation, int) or len(dilation) == 2
#         if isinstance(stride, int):
#             stride_h = stride_w = stride
#         else:
#             stride_h, stride_w = stride

#         if isinstance(dilation, int):
#             dilation_h = dilation_w = dilation
#         else:
#             dilation_h, dilation_w = dilation

#         batch, in_channel, in_height, in_width = Input.shape
#         num_filter, channel, kernel_h, kernel_w = Filter.shape
#         # compute the output shape
#         dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
#         dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
#         pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
#             padding, (dilated_kernel_h, dilated_kernel_w))
#         out_channel = num_filter
#         out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
#         out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
#         # compute graph
#         pad_before = [0, 0, pad_top, pad_left]
#         pad_after = [0, 0, pad_down, pad_right]
#         temp = pad(Input, pad_before, pad_after, name="pad_temp")
#         print("in_channel = ", in_channel)
#         rc = tvm.reduce_axis((0, in_channel), name='rc')
#         ry = tvm.reduce_axis((0, kernel_h), name='ry')
#         rx = tvm.reduce_axis((0, kernel_w), name='rx')
#         print("in _conv2d_nchw")

#         # inp = tvm.placeholder((temp, Filter), name="inp")
#         # conv2d_nchw = tvm.placeholder((batch, out_channel, out_height, out_width), name = 'conv2d_nchw')
#         # s_state = tvm.placeholder((num_filter,channel, kernel_h, kernel_w ))
#         # s_init = tvm.compute(
#         #     (batch, out_channel, out_height, out_width),
#         #     lambda nn, ff, yy, xx: 
#         #         temp[nn, rc, yy * stride_h + ry * dilation_h,
#         #             xx * stride_w + rx * dilation_w].astype(out_dtype) *
#         #         Filter[ff, rc, ry, rx].astype(out_dtype), tag="conv2d_nchw")
#         # print("s_init.shape")
#         # print(s_init.shape)
#         # s_update = tvm.compute(
#         #     (num_filter,channel, kernel_h, kernel_w ),
#         #     lambda nn, ff,yy,xx:
#         #         s_state[ff-1,rc,ry,rx] + s_state[ff,rc,ry,rx],
#         #          tag="conv2d_nchw")
#         # res = tvm.scan(s_init, s_update, s_state, tag="conv2d_nchw")
#         # return res
#         # mysum = tvm.comm_reducer(lambda x, y: x+y,
#         #     lambda t: tvm.const(0, dtype=t), name="mysum")

#         # # conv - multiply
#         # a = tvm.compute(
#         #     (batch, in_channel, in_height, in_width),
#         #     lambda nn, ff, yy, xx:
#         #         temp[nn, rc, yy * stride_h + ry * dilation_h,
#         #             xx * stride_w + rx * dilation_w].astype(out_dtype) *
#         #         Filter[ff, rc, ry, rx].astype(out_dtype), tag="conv2d_nchw")
        
#         # # conv - sum 
#         # b = tvm.compute(
#         #     (batch, out_channel, out_height, out_width),
#         #     lambda nn, ff, yy, xx: a[nn,ff,yy,xx]*2, tag="conv2d_nchw")
#         # return b
        
#         def _myconv(nn, ff, yy, xx):
#             tmp_list = []
#             for rc in range(0, in_channel):
#                 for ry in range(0, in_height):
#                     for rx in range(0, in_width):
#                         tmp_list = temp[nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(out_dtype) * Filter[ff, rc, ry, rx].astype(out_dtype)
#             a = sorted(tmp_list)
#             temp[nn,ff,yy * stride_h ,xx * stride_w]
#         def _mysum(nn,ff,yy,xx):
#             a = 0
#             b = 0
#             for ryy in range(0,3):
#                 for rxx in range(0,3):
#                     # tmp = tvm.if_then_else(a > b, 1, 0)
#                     # tmp1 = tvm.if_then_else(a > b, a,b)
#                     # tmp2 = tmp1
#                     # tmp1 += temp[nn,0, yy+ryy, xx+rxx]*Filter[ff,0,ryy,rxx]
#                     # a = tvm.if_then_else(tmp2 == a, tmp1, a)
#                     # b = tvm.if_then_else(tmp2 == b, tmp1, b)
#                     a += temp[nn,0, yy+ryy, xx+rxx]*Filter[ff,0,ryy,rxx]
#             return a
#         ans = tvm.compute(
#             (batch, out_channel, out_height, out_width),
#             lambda nn, ff, yy, xx: tvm.sum(
#             temp[nn, rc, yy * stride_h + ry * dilation_h,
#                  xx * stride_w + rx * dilation_w].astype(out_dtype) *
#             Filter[ff, rc, ry, rx].astype(out_dtype),
#             axis=[rc, ry, rx]), tag="conv2d_nchw")


#         # ##### Use custom function
#         # ans = tvm.compute((batch, out_channel, out_height, out_width),
#         #     lambda nn, ff, yy, xx: 
#         #     _mysum(nn,ff,yy,xx), tag="conv2d_nchw")

#         # s = tvm.create_schedule(ans.op)
#         # op_list = [i for i in ans.op.input_tensors]
#         # op_list.append(ans)
#         # print(op_list)
#         # print(tvm.lower(s,op_list,simple_mode=True))
    
#         # print("DDDDD")

#         return ans
        # return tvm.compute(
        # (batch, out_channel, out_height, out_width),
        # lambda nn, ff, yy, xx:
        #     temp[nn, rc, yy * stride_h + ry * dilation_h,
        #          xx * stride_w + rx * dilation_w].astype(out_dtype) *
        #     Filter[ff, rc, ry, rx].astype(out_dtype))

    # @tvm.target.generic_func
    # def _conv2d(input, filter, strides, padding, dilation, layout='NCHW', out_dtype=None):
    #     """Conv2D operator.

    #     Parameters
    #     ----------
    #     input : tvm.Tensor
    #         4-D with shape [batch, in_channel, in_height, in_width]

    #     filter : tvm.Tensor
    #         4-D with shape [num_filter, in_channel, filter_height, filter_width]

    #     strides : int or a list/tuple of two ints
    #         stride size, or [stride_height, stride_width]

    #     padding : int or a list/tuple of two ints
    #         padding size, or [pad_height, pad_width]

    #     dilation: int or a list/tuple of two ints
    #         dilation size, or [dilation_height, dilation_width]

    #     layout : str
    #         layout of data

    #     Returns
    #     -------
    #     output : tvm.Tensor
    #         4-D with shape [batch, out_channel, out_height, out_width]
    #     """
    #     # search platform specific declaration first
    #     # default declaration
    #     if layout == 'NCHW':
    #         print("call _conv2d_nchw")
    #         return _conv2d_nchw(input, filter, strides, padding, dilation, out_dtype)
    #     if layout == 'HWCN':
    #         print("call conv2d_hwcn")
    #         return conv2d_hwcn(input, filter, strides, padding, dilation, out_dtype)
    #     if layout == 'NHWC':
    #         print("call conv2d_nhwc")
    #         return conv2d_nhwc(input, filter, strides, padding, dilation, out_dtype)
    #     raise ValueError("not support this layout {} yet".format(layout))



    # """Compute definition of conv2d"""
    # padding = attrs.get_int_tuple("padding")
    # strides = attrs.get_int_tuple("strides")
    # dilation = attrs.get_int_tuple("dilation")
    # groups = attrs.get_int("groups")
    # channels = attrs.get_int("channels")
    # layout = attrs["layout"]

    # assert layout == "NCHW" or layout == "NHWC"
    # (dilation_h, dilation_w) = dilation
    # if dilation_h < 1 or dilation_w < 1:
    #     print("HERE")
    #     raise ValueError("dilation should be positive value")
    # elif dilation == (1, 1):
    #     kernel = inputs[1]
    # elif layout == "NCHW":
    #     print("OR HERE")
    #     kernel = topi.nn.dilate(inputs[1], [1, 1, dilation_h, dilation_w])
    # else: #layout == NHWC
    #     print("OR OR HERE")
    #     kernel = topi.nn.dilate(inputs[1], [1, dilation_h, dilation_w, 1])

    # if groups == 1:
    #     print("2 HERE")
        
    #     print(inputs[0].shape)
    #     print(type(inputs[0]))
    #     out = _conv2d(inputs[0], kernel, strides, padding, dilation)
    #     # bias = np.array([100]).astype('float32')
    #     # expand_axis = 1 if layout == "NCHW" else 0
    #     # bias = topi.expand_dims(bias, axis=expand_axis, num_newaxis=2)
    #     # out = topi.broadcast.add(out, bias)
    # elif groups == get_const_int(inputs[0].shape[1]) and groups == channels:
    #     print("2 OR HERE")
    #     out = topi.nn.depthwise_conv2d_nchw(inputs[0], kernel, strides, padding)
    # else:
    #     print("2 OR OR HERE")
    #     raise ValueError("not support arbitrary group number for now")
    # if attrs.get_bool("use_bias"):
    #     bias = inputs[2]
    #     expand_axis = 1 if layout == "NCHW" else 0
    #     bias = topi.expand_dims(bias, axis=expand_axis, num_newaxis=2)
    #     out = topi.broadcast.add(out, bias)
    # print("return out")
    # return out



############################################
# Schedule Registry
# @reg.register_convert_op_layout("nn.conv2d", level = 15)
# def register_convert_op_layout

from tvm.relay import data_dep_optimization as ddo

def run_sparse(mod, params, shape_dict, target, ctx, bs_r, sparsity):
    mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    mod, params = ddo.sparse_conv2d.convert(mod, params, (bs_r, 1), sparsity_threshold=0.2)
    for key in params.keys():
        arr = params[key].asnumpy()
        print("key %s's arr : " ,key)
        print(" shape : " , arr.shape)
        print(" value : " , arr)
    print("Block Sparse Model with {blocksize}x1 blocks:".format(blocksize=bs_r))
    return mod, params 
    # run_relay_graph(mod, params, shape_dict, target, ctx)

######################################################################
# now compile the graph
# import nnvm.compiler
from tvm.contrib import graph_runtime
# from tvm.contrib.debugger import debug_runtime as graph_runtime
import sys

# Choose Target
target = 'llvm -system-lib'
# target = 'llvm -target=riscv64-unkown-elf -system-lib'


if target == 'llvm -system-lib':


    # Parameters setting
    np_input = np.array([[[
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1] 
                    ]]]).astype('float32')
    shape_dict = {'data': np_input.shape}
    ctx = tvm.cpu(0)
    dtype = 'float32'

     # Load model
    
    # mx_sym, args, auxs = mx.model.load_checkpoint("simple_conv", 0) 
    mx_sym, args, auxs = mx.model.load_checkpoint("sparse_conv", 0) 
    sym, params = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype, args,auxs)
    print(sym)
    # sym , params = run_sparse(sym, params, shape_dict, target, ctx, 1, 0.002)
    print(sym)

    # print("key = ", params.keys())
    for key in params.keys():
        print("key = ", key)
        print("params = ", params[key])
        arr = params[key].asnumpy()
    
    path = "./" 
    modelname = "test"
    input_name = "data"
    
    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(sym, target=target, params=params)
    
   
    # Run Model
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(np_input.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)

    print("tvm_output")
    print(tvm_output)


