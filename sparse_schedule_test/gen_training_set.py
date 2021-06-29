import os
import numpy as np
import logging
import sys
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import tvm
import tvm.relay as relay
from tvm.relay import data_dep_optimization as ddo
import struct
# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime
import scipy.sparse as sparse
import scipy.stats as stats
import pandas as pd
import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.random.seed(42)
np.set_printoptions(threshold=sys.maxsize)


def run_sparse(mod, params, shape_dict, target, ctx, bs_r, sparsity, mode):
    # mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    mod, params = ddo.bsr_dense.convert(mod["main"], params, (bs_r, 2), sparsity_threshold=sparsity, mode=mode)
    return mod, params


def gen_model(weight, h, w):
    np_dense_weight = weight.astype('float32')
    nd_weight=nd.array(np_dense_weight) 
    W = mx.initializer.Constant(nd_weight)
    dense_layer = nn.Dense(h ,activation= "relu", use_bias=False, weight_initializer=W, bias_initializer=None)
    
    np_input = sparse.random(64, w, density=1.0)
    np_input = np_input.toarray().astype('float32')
    mxnet_input = mx.nd.array(np_input)
    flat = mxnet_input.flatten()

    net = nn.HybridSequential()
    net.add(dense_layer)
    net.initialize()
    net.hybridize()
    net.forward(mx.nd.array(mxnet_input))
    return np_input, net

def run_single_dense(np_input, net, mode):
    shape_dict = {'data': np_input.shape}
    #target = 'llvm -target=riscv64-unkown-elf -system-lib'
    target='llvm -system-lib'
    ctx = tvm.cpu(0)
    dtype = 'float32'
    mod, params = relay.frontend.from_mxnet(net, shape_dict,)
    threshold = 0.0
    if mode != "nor":
        mod , params = run_sparse(mod, params, shape_dict, target, ctx, 16, threshold, mode)
    
    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(mod, target, params=params)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(np_input.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    # tvm_output = m.get_output(0)
    ftimer = m.module.time_evaluator("run", ctx, repeat=5, number=5)
    prof_res = np.array(ftimer().results) * 1000
    print(
        "%-20s %-19s (%s)"
        % ("Runtime:", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )
    return np.mean(prof_res)

def process_dict(model_name, p_dict):
    name = []
    h = []
    w = []
    sp = []
    avg = []
    dev = []
    label = []
    
    absp = []
    sbsp = []
    sblk = []
    sratio = []
    def cal_sp(p):
        row_sp = []
        b_sp = []
        for r in p:
            sparsity = 1.0 - (np.count_nonzero(r) / r.size)
            row_sp.append(sparsity)
        
        tmp = []
        for r in range(p.shape[0]):
            # print(p.size)
            # print(p.shape[0])
            if r+1 < p.shape[0]:
                tmp.append(p[r][0])
                tmp.append(p[r+1][0])
            for i in range(1, p.shape[1]):
                if r+1 < p.shape[0]:
                    tmp.append(p[r][i])
                    tmp.append(p[r+1][i])
                if i%16 == 0 and len(tmp)!=0:
                    tmp_sp = tmp.count(0.0) / len(tmp)
                    b_sp.append(tmp_sp)
                    tmp.clear()
                    if r+1 < p.shape[0]:
                        tmp.append(p[r][i])
                        tmp.append(p[r+1][i])        
            tmp.clear()
            r += 1
        # print("print(b_sp): ",b_sp)
        # print("AVG = ", np.mean(np.array(b_sp)))
        save_blk = 0
        save_ratio = 0
        for i in b_sp:
            if i == 1.0:
                save_blk+=1
        save_ratio = 1-(((p.size-np.count_nonzero(p))-(save_blk*32))/p.size)

        return np.mean(np.array(row_sp)), np.std(np.array(row_sp)), np.mean(np.array(b_sp)) ,np.std(np.array(b_sp)), save_blk ,save_ratio
    for key, val in p_dict.items():
        print("run : ", key)
        name.append(key)
        w_np = val
        h.append(w_np.shape[0])
        w.append(w_np.shape[1])
        sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
        sp.append(np.count_nonzero(w_np))
        avg_row, std_row, avg_bsp, std_bsp, s_blk, s_ratio = cal_sp(w_np)
        avg.append(avg_row)
        dev.append(std_row)
        absp.append(avg_bsp)
        sbsp.append(std_bsp)
        sblk.append(s_blk)
        sratio.append(s_ratio)
        np_input, net = gen_model(w_np, w_np.shape[0], w_np.shape[1])
        csr_time = run_single_dense(np_input, net, "csr")
        bsr_time = run_single_dense(np_input, net, "bsr")
        # if bsr_time > csr_time:
        #     #csr = 0
        #     label.append(0)
        # else:
        #     #bsr = 1
        #     label.append(1)
        nor_time = run_single_dense(np_input, net, "nor")
        if bsr_time >= csr_time and nor_time >= csr_time:
            #csr = 0
            label.append(0)
        elif csr_time > bsr_time and nor_time >= bsr_time:
            #bsr = 1
            label.append(1)
        elif csr_time > nor_time and bsr_time > nor_time:
            #nor = 2
            label.append(2)
    df = pd.DataFrame(data=param_dict.keys(),columns=['name'])
    df['height'] = h
    df['width'] = w
    df['nnz'] = sp
    df['avg_row'] = avg
    df['std_row'] = dev
    df['avg_bsp'] = absp
    df['std_bsp'] = sbsp
    df['s_blk'] = sblk
    # df['s_blk'] = sblk
    df['s_ratio'] = sratio
    df['label'] = label
    print(df) 
    df.to_csv(path_or_buf="./raw_dict/sblk_3label_random_gen_"+model_name+".csv", mode='a', header=False, index=False)

model_name = "new_sort_random"
param_dict = dict()
np.random.seed(42)
import sys
a=int(sys.argv[1])
print("a = ",a)
for i in range(a ,a+15):
    
    height = random.randint(800, 6000)
    weight = random.randint(800, 6000)
    for j in range(0, 5):
        density = random.uniform(0, 0.7)
        A = sparse.random(height, weight, density=density)
        A = A.toarray()
        param_dict[i*10+j] = A
process_dict(model_name, param_dict)