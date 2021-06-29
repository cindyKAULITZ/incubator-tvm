import mxnet as mx
import onnx
import tvm
import numpy as np
import struct
import os
import time
import itertools
import numpy as np
from tvm import relay
from tvm.relay import data_dep_optimization as ddo
from PIL import Image

from tvm.contrib import graph_runtime
# from tvm.contrib.debugger import debug_runtime as graph_runtime

from graph_viewer import MixedFxper
from graph_rewriter import InsertTransHelper

np_input = np.array([[[
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1] 
                    ]]]).astype('float32')
shape_dict = {'data': np_input.shape}
ctx = tvm.cpu(0)
dtype = 'float32'
target = 'llvm'

def run_relay_graph(mod, params):
    print(mod)
    
    with tvm.transform.PassContext(opt_level=0): 
        graph, lib, params = relay.build(mod, target=target, params=params)

    m = graph_runtime.create(graph, lib, ctx)
    m.set_input('data', tvm.nd.array(np_input.astype('float32')))
    m.set_input(**params)
    m.run()

    # get outputs
    # tmp = m.debug_get_output("fused_nn_dense_0")
    # print(tmp)
    tvm_output = m.get_output(0)
    print("tvm_output")
    print(tvm_output)

    ftimer = m.module.time_evaluator("run", ctx, repeat=10, number=100)
    prof_res = np.array(ftimer().results) * 1000000

    return prof_res


def run_dense(mod, params):
    print("Dense Model:")
    return run_relay_graph(mod, params)


def run_sparse(mod, params, bs_r, sparsity, mode):
    print("Sparse Model:")
    mod, params = ddo.bsr_dense.convert(mod["main"], params, (bs_r, 1), sparsity_threshold=sparsity,mode=mode)
    return run_relay_graph(mod, params)

def load_model(model_type, sparsity):

    # Load model
    if(model_type == "dense"):
        mx_sym, args, auxs = mx.model.load_checkpoint("../sparse_schedule_test/dense_test", 0)
    elif(model_type == "sparse"):
        # mx_sym, args, auxs = mx.model.load_checkpoint("../sparse_schedule_test/dense(sparse"+sparsity+")_big_test", 0) 
        mx_sym, args, auxs = mx.model.load_checkpoint("../sparse_schedule_test/dense(sparse"+sparsity+")_test", 0) 
    
    sym, params = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype, args,auxs)

    return sym, params


def print_runtime(run_type, runtime, remove_outlier):
    if (remove_outlier) :
        max_deviations = 0.5
        # print(runtime)
        # print("\nremove outlier with max_deviations = ", max_deviations)
        mean = np.mean(runtime)
        standard_deviation = np.std(runtime)
        distance_from_mean = abs(runtime - mean)
        not_outlier = distance_from_mean < max_deviations * standard_deviation
        runtime = np.array(runtime)[not_outlier]

    print(
        "%-40s %-10s (%s)"
        % (run_type + " Runtime:", "%.2f us" % np.mean(runtime), "%.2f us" % np.std(runtime))
    )
    return np.mean(runtime)

def debug_check_weight(params, axis):
    s = []
    def nnz_count(x):
        sparsity = 1.0 - (np.count_nonzero(x) / x.size)
        s.append(sparsity)
        return sparsity
    def draw(name, arr):
        import matplotlib.pyplot as plt 
        # x-coordinates of left sides of bars  
        left = list(range(0, arr.size()))
        
        # heights of bars 
        height = arr

        # plotting a bar chart 
        plt.bar(left, height,  
                width = 0.8, color = ['green']) 
        # plot title 
        plt.title(name) 
        # function to show the plot 
        plt.show() 
        
    for key in params.keys():
        arr = params[key].asnumpy()
        print("key %s's arr : " ,key)
        np.apply_along_axis( nnz_count, axis=axis, arr=arr )

        draw(key, arr)
        

def run_single(model_type, sparsity):

    r_t = []

    print("Model type : ", model_type, " | Density : ", sparsity)
    sym, params = load_model(model_type,sparsity)
    r_t.append(run_dense(sym, params))
    # print("Axis = 0")
    # debug_check_weight(params,0)
    # print("Axis = 1")
    # debug_check_weight(params,1)

    sym, params = load_model(model_type,sparsity)
    r_t.append(run_sparse(sym, params, 1, 0,"bsr"))
    
    d_t = print_runtime("Normal dense op", r_t[0], True)
    s_t = print_runtime("Compress sparse_dense op", r_t[1], True)
    print("(speedup : %.2f" % (d_t/s_t) , ")\n")

def run_all(model_type):

    r_t = []
    r_t_d = []

    for i in range(1,8):
        sym, params = load_model(model_type,"00"+str(i))
        r_t_d.append(run_dense(sym, params))

        sym, params = load_model(model_type,"00"+str(i))
        r_t.append(run_sparse(sym, params, 1, 0,mode = "bsr"))

    # print(r_t)
    print("Model type : ", model_type)
    for i in range(1,8):
        print("Weight density : 0.0"+str(i))
        d_t = print_runtime("Normal dense op", r_t_d[i-1], True)
        s_t = print_runtime("Compress sparse_dense op", r_t[i-1], True)
        print("(speedup : %.2f" % (d_t/s_t) , ")\n")

# run_all("dense")
# run_all("sparse")

run_single("dense", "002")
run_single("sparse", "002")