# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Conv2D schedule on x86"""

import logging

import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.pad import pad
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple, traverse_inline
from . import conv2d_avx_1x1, conv2d_avx_common

logger = logging.getLogger('topi')

def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False,
                        layout='NCHW'):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)
    if is_depthwise:
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, out_dtype)
        from .depthwise_conv2d import _fallback_schedule
        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
        is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
        if is_kernel_1x1:
            conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        else:
            conv2d_avx_common._fallback_schedule(cfg, wkl)

@conv2d_infer_layout.register("cpu")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, _, dtype = workload
    batch_size, in_channel, in_height, in_width = data[1]
    out_channel, _, k_height, k_width = kernel[1]
    idxdiv = tvm.tir.indexdiv

    pt, pl, pb, pr = get_pad_tuple(padding, (k_height, k_width))
    out_height = idxdiv(in_height + pt + pb - k_height, strides[0]) + 1
    out_width = idxdiv(in_width + pl + pr - k_width, strides[1]) + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, idxdiv(in_channel, tile_ic), in_height, in_width, tile_ic)
    in_layout = "NCHW%dc" % tile_ic
    out_shape = (batch_size, idxdiv(out_channel, tile_oc), out_height, out_width, tile_oc)
    out_layout = "NCHW%dc" % tile_oc
    return ((in_shape, in_layout),), ((out_shape, out_layout),)

def schedule_conv2d_nhwc(outs):
    """Create schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    def _callback(op):
        if 'conv2d_nhwc' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, h_pad, w_pad, c_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, h, w, c = C.op.axis
            s[C].vectorize(c)

            O = output_op.output(0)
            if len(O.op.axis) == 4: # schedule bias + bn + relu
                n, h, w, c = O.op.axis
                fused = s[O].fuse(n, h, w)
                s[O].parallel(fused)
                channels = int(O.shape[-1])
                if channels % 64 == 0:
                    c, ci = s[O].split(c, 64)
                    s[O].vectorize(ci)
                if C != O:
                    s[C].compute_at(s[O], c)

    traverse_inline(s, output_op, _callback)
    return s

def conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype):
    layout = "NCHW"
    # packed_out = conv2d_NCHWc(data, kernel, strides, padding, dilation,
    #                           layout, layout, out_dtype)
    # return unpack_NCHWc_to_nchw(packed_out, out_dtype)
    
    # im2col
    packed_out = conv2d_NCHWc_im2col(data, kernel, strides, padding, dilation,
                              layout, layout, out_dtype)
    return packed_out

def schedule_conv2d_nchw(outs):
    """Create schedule for tensors"""
    return schedule_conv2d_NCHWc(outs)

def _pack_data(cfg, data, kernel):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    data = te.compute((n, ic_chunk, ih, iw, ic_bn),
                      lambda bs, c, h, w, vc: data[bs, c*ic_bn + vc, h, w],
                      name="data_vec")

    kernel = te.compute(
        (oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn),
        lambda occ, icc, k_h, k_w, icb, ocb:
        kernel[occ * oc_bn + ocb, icc * ic_bn + icb, k_h, k_w],
        name="kernel_vec")

    return data, kernel

@autotvm.register_topi_compute("conv2d_NCHWc_im2col.x86")
def conv2d_NCHWc_im2col(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype='float32'):
    """Compute conv2d with NCHWc layout."""
    print('in_conv2d_im2col')
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        N, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        N, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, ic, kernel_height, kernel_width = get_const_tuple(kernel.shape)

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
        data_pad = pad(data, (0, 0, HPAD//2, WPAD//2), name="data_pad")
        # data_pad = infer_pad(data, (0, 0, HPAD//2, WPAD//2), name="data_pad")
    else:
        data_pad = data
    
    ALIGN = 16
    def upround(x, align):
        return (x + align - 1) // align * align
    reduce_len = upround(in_channel * kernel_height * kernel_width, ALIGN)

    # A [CO, CI * KH * KW] [M, K*K*C]
    A = te.compute((upround(num_filter, ALIGN), reduce_len), lambda i, j:
                    kernel[i][j // kernel_width // kernel_height][j // kernel_width % kernel_height][j % kernel_width], name='A')

    # B [CI * KH * KW, N * OH * OW] [K*K*C, N*H*W]
    B = te.compute((reduce_len, upround(N * out_height * out_width, ALIGN)), lambda i, j:\
                te.if_then_else(te.all(i < in_channel * kernel_height * kernel_width, j < N * out_height * out_width),
                       data_pad[j // (out_height*out_width)][i // (kernel_height*kernel_width)][j // out_width % out_height*sh + i // kernel_width % kernel_height]
                       [j % out_width*sw + i % kernel_width],
                       tvm.tir.const(0, data_pad.dtype)), name='B')

    gemm_n, gemm_l, gemm_m = A.shape[0], reduce_len, B.shape[1]

    # C [CO, N * OH * OW]
    k = te.reduce_axis((0, gemm_l), name='k')
    C = te.compute((gemm_n, gemm_m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

    oshape = (N, num_filter, out_height, out_width)

    return te.compute(oshape, lambda n, co, h, w:\
                 C[co][n * out_width * out_width + h * out_width + w] + tvm.tir.const(0, C.dtype) * C[gemm_n-1, gemm_m-1],
                         name='conv2d_NCHWc', tag='im2col_conv_output')

@autotvm.register_topi_compute("conv2d_NCHWc.x86")
def conv2d_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = \
            get_const_tuple(kernel.shape)
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64,
                     policy="verbose")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(cfg, te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            te.placeholder((num_filter, in_channel, kernel_height, kernel_width),
                                           dtype=kernel.dtype),
                            strides, padding, out_dtype)

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, in_channel // cfg["tile_ic"].size[-1],
                      ih, iw, cfg["tile_ic"].size[-1])
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (num_filter // cfg["tile_oc"].size[-1],
                      in_channel // cfg["tile_ic"].size[-1],
                      kernel_height, kernel_width,
                      cfg["tile_ic"].size[-1],
                      cfg["tile_oc"].size[-1])
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel")
        else:
            data, kernel = _pack_data(cfg, data, kernel)

    return nn.conv2d_NCHWc(data,
                           kernel,
                           strides,
                           padding,
                           dilation,
                           layout,
                           out_layout,
                           out_dtype)

@autotvm.register_topi_schedule("conv2d_NCHWc.x86")
def schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            _, _, kh, kw, _, _, = get_const_tuple(kernel_vec.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


# FIXME - https://github.com/apache/incubator-tvm/issues/4122
# _declaration_conv_nhwc_pack expects kernel layout to be HWOI. However, the tests use HWIO
# layout. Commenting until we have clarity about the nhwc_pack implementation from the author.
# elif layout == 'NHWC' and kh == 1 and kw == 1 and kernel.dtype == "int8":
#     if cfg.is_fallback:
#         _get_default_config(cfg, data, kernel, strides, padding, out_dtype, False, layout)
#     # specialize for INT8 1X1 conv on X86
#     return conv2d_avx_1x1._declaration_conv_nhwc_pack(cfg, data, kernel, strides,
#                                                       padding, dilation, out_dtype)
