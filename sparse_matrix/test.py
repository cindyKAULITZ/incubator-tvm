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
    path = "./" 
    modelname = "test"
    input_name = "data"

     # Load model
    
    # mx_sym, args, auxs = mx.model.load_checkpoint("simple_conv", 0) 
    mx_sym, args, auxs = mx.model.load_checkpoint("../sparse_schedule_test/sparse_conv2_dense", 0) 
    sym, params = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype, args,auxs)
    print(sym)
    from graph_viewer import MixedFxper
    from graph_rewriter import InsertTransHelper
    # new_func = MixedFxper('float32', input_name).visit(sym['main'])
    # sym['main'] = new_func
    # print('new mod', sym['main'])

    mhelper = InsertTransHelper(sym)
    
    mhelper.transform_conv2d()

    # sym , params = run_sparse(sym, params, shape_dict, target, ctx, 1, 0.002)
    # desired_layouts = {'nn.conv2d': ['IMCO', 'IMCO']}
    # desired_layouts = {'nn.conv2d': ['NCHW', '11HW']}
    # seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
    #                             relay.transform.ConvertLayout(desired_layouts)])
    

    # print("key = ", params.keys())
    # for key in params.keys():
    #     print("key = ", key)
    #     print("params = ", params[key])
    #     arr = params[key].asnumpy()
    
    
    with tvm.transform.PassContext(opt_level=3):
        # sym = seq(sym)
        print(sym)
    # with relay.build_config(opt_level=0):
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


