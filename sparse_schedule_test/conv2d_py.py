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
from tvm.contrib import graph_runtime
# from tvm.relay.transform import FastMath


# %%
np_weight = np.array( [[[[ 0.1, 0.0, 0.0], 
                       [ 0.1, 0.0, 0.0],
                       [ 0.1, 0.0, 0.0]]]]).astype('float32')


# %%
nd_weight=nd.array(np_weight)
print(nd_weight)


# %%
W = mx.initializer.Constant(nd_weight)
print(W)


# %%
np_bias = np.array([0.01]).astype('float32')


# %%
nd_bias=nd.array(np_bias)
print(nd_bias)


# %%
B = mx.initializer.Constant(nd_bias)
print(B)


# %%
conv2d_layer = nn.Conv2D(channels=1, kernel_size=(3, 3), 
                         strides=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1,
                         layout='NCHW', activation= None, use_bias=False, weight_initializer=W,bias_initializer=B, in_channels=0)

# dense_layer = nn.Dense(units=1 activation= None, use_bias=False, weight_initializer=W, bias_initializer=B)


# %%

# print("conv2d_layer.bias.data")
# print(conv2d_layer.bias.data)


# %%
# mxnet exp custom layer
"""
class op(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(op, self).__init__(**kwargs)

    def forward(self, x):
        return x.softmax()
"""


# %%
np_input = np.array([[[
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1], 
                    [ 0.1, 0.2, 0.5, 0.1] 
                    ]]]).astype('float32')


# %%
mxnet_input = nd.array(np_input)
print("mxnet_input")
print(mxnet_input)
flat = mxnet_input.flatten()
print("flat input")
print(flat)

# %%
net = nn.HybridSequential()
net.add(conv2d_layer)
# net.hybridize()
net.collect_params()
net.initialize()


# %%
net.summary(mxnet_input)


# %%
print("net")
print(net)



net(nd.array(mxnet_input))
print(nd.array(mxnet_input))

net.hybridize()
net.forward(nd.array(mxnet_input))
net(nd.array(mxnet_input))
# print(net.params())
net.export("sparse_conv", epoch=0)
# net.save_parameters("simple_conv")
# mx.model.save_checkpoint("simple_conv",0,)
# net.save("simple_conv",1)
# def save_checkpoint(epoch, module, callback):
#     arg_params, aux_params = module.get_params()
#     module.set_params(arg_params, aux_params)
#     callback(epoch, module.symbol, arg_params, aux_params)

# save_checkpoint(0, net, callback)

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
print("params")
print(params)


# %%
#logging.getLogger("compile_engine").setLevel(logging.INFO)
#logging.getLogger("compile_engine").addHandler(logging.StreamHandler(sys.stdout))
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
#save graph. lib. params. in directory
path_ll = "./model.ll"
with open(path_ll, "w") as f:
    f.write(lib.get_source())

path_graph = "./model.json"
#path_graph = path_dir+"/model.graph"
with open(path_graph, "w") as fo:
    fo.write(graph)

path_params = "./model.params"
with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))


print("save module finish")

# %%
#print(graph)


# %%
#print(lib.get_source())


# %%


