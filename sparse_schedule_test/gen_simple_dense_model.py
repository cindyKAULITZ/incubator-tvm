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
s1 = 32768
s2 = 8192
s3 = 2048
s4 = 256
s5 = 10

A = sparse.random(s1, in_shape, density=density)
# Convert the sparse matrix to a full matrix
A = A.toarray()
np_dense_weight = A.astype('float32')

B = sparse.random(s2, s1, density=density)
# Convert the sparse matrix to a full matrix
B = B.toarray()
np_dense_weight2 = B.astype('float32')

C = sparse.random(s3, s2, density=density)
# Convert the sparse matrix to a full matrix
C = C.toarray()
np_dense_weight3 = C.astype('float32')

D = sparse.random(s4, s3, density=density)
# Convert the sparse matrix to a full matrix
D = D.toarray()
np_dense_weight4 = D.astype('float32')

E = sparse.random(s5, s4, density=density)
# Convert the sparse matrix to a full matrix
E = E.toarray()
np_dense_weight5 = E.astype('float32')

np.set_printoptions(threshold=sys.maxsize)

# print("sparsity :", 1.0-density)
# print("np_dense_weight")
# print(np_dense_weight)
# print("np_dense_weight2")
# print(np_dense_weight2)
# print("np_dense_weight3")
# print(np_dense_weight3)
# print("np_dense_weight4")
# print(np_dense_weight4)
# print("np_dense_weight5")
# print(np_dense_weight5)


# %%
nd_weight=nd.array(np_dense_weight) 
nd_weight2=nd.array(np_dense_weight2)
nd_weight3=nd.array(np_dense_weight3)
nd_weight4=nd.array(np_dense_weight4)
nd_weight5=nd.array(np_dense_weight5)
# print(nd_weight)


# %%
W = mx.initializer.Constant(nd_weight)
W2 = mx.initializer.Constant(nd_weight2)
W3 = mx.initializer.Constant(nd_weight3)
W4 = mx.initializer.Constant(nd_weight4)
W5 = mx.initializer.Constant(nd_weight5)
# print(W)



# %%

dense_layer = nn.Dense(s1 ,activation= "relu", use_bias=False, weight_initializer=W, bias_initializer=None)
dense_layer2 = nn.Dense(s2,activation= "relu", use_bias=False, weight_initializer=W2, bias_initializer=None)
dense_layer3 = nn.Dense(s3,activation= "relu", use_bias=False, weight_initializer=W3, bias_initializer=None)
dense_layer4 = nn.Dense(s4,activation= "relu", use_bias=False, weight_initializer=W4, bias_initializer=None)
dense_layer5 = nn.Dense(s5,activation= "relu", use_bias=False, weight_initializer=W5, bias_initializer=None)


# %%
np_input = np.array([[[
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1] 
                    ]]]).astype('float32')


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
print("mxnet_output")
print(mxnet_output)



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
