# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
import logging
import sys


# %%
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


# %%
import tvm
import tvm.relay as relay
# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime
import scipy.sparse as sparse
import scipy.stats as stats
import numpy as np


# %%
np.random.seed(42)

density = 0.10
export_name = "dense(sparse010)_big_test"

in_shape = 16
s1 = 8
s2 = 8192
s3 = 2048
s4 = 256
s5 = 10

A = sparse.random(s1, in_shape, density=density)
# Convert the sparse matrix to a full matrix
A = A.toarray()
np_dense_weight = A.astype('float32')



np.set_printoptions(threshold=sys.maxsize)




# %%
nd_weight=nd.array(np_dense_weight) 

# print(nd_weight)


# %%
W = mx.initializer.Constant(nd_weight)

# print(W)



# %%

dense_layer = nn.Dense(s1 ,activation= "relu", use_bias=False, weight_initializer=W, bias_initializer=None)

# %%
np_input = sparse.random( 64, in_shape, density=1.0)
np_input = np_input.toarray().astype('float32')
# np_input = np.array([
#                     [ 0.1, 0.2, 0.5, 0.1], 
#                     [ 0.1, 0.2, 0.5, 0.1], 
#                     [ 0.1, 0.2, 0.5, 0.1], 
#                     [ 0.1, 0.2, 0.5, 0.1] 
#                     ]).astype('float32')


# %%
mxnet_input = mx.nd.array(np_input)
# print("mxnet_input")
# print(mxnet_input)
flat = mxnet_input.flatten()
# print("flat input")
# print(flat)

# %%
net = nn.HybridSequential()

net.add(dense_layer)

net.initialize()


# %%
print("net")
print(net)

net.initialize()
net.hybridize()
net.forward(mx.nd.array(mxnet_input))

# net.export(export_name, epoch=0)
print("dense_test")


# %%
mxnet_output = net(mxnet_input)
# print("mxnet_output")
# print(mxnet_output)



# %%
#tvm setting
shape_dict = {'data': np_input.shape}
#target = 'llvm -target=riscv64-unkown-elf -system-lib'
target='llvm -system-lib'
ctx = tvm.cpu(0)
dtype = 'float32'


# %%
#tvm parse mxnet model
mod, params = relay.frontend.from_mxnet(net, shape_dict,)

# %%
print(mod)

# %%
# print("params")
# print(params)


# %%
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target, params=params)

# %%
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('data', tvm.nd.array(np_input.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)

# %%
print("tvm_output")
print(tvm_output)

ftimer = m.module.time_evaluator("run", ctx, repeat=5, number=5)
prof_res = np.array(ftimer().results) * 1000
print(
    "%-20s %-19s (%s)"
    % ("Runtime:", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
)

# %%
# #save graph. lib. params. in directory
# path_ll = "./model.ll"
# with open(path_ll, "w") as f:
#     f.write(lib.get_source())

# path_graph = "./model.json"
# #path_graph = path_dir+"/model.graph"
# with open(path_graph, "w") as fo:
#     fo.write(graph)

# path_params = "./model.params"
# with open(path_params, "wb") as fo:
#     fo.write(relay.save_param_dict(params))


# print("save module finish")
