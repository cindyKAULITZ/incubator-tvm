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
"""
Compile PyTorch Models
======================
**Author**: `Alex Wong <https://github.com/alexwong/>`_

This article is an introductory tutorial to deploy PyTorch models with Relay.

For us to begin with, PyTorch should be installed.
TorchVision is also required since we will be using it as our model zoo.

A quick solution is to install via pip

.. code-block:: bash

    pip install torch==1.4.0
    pip install torchvision==0.5.0

or please refer to official site
https://pytorch.org/get-started/locally/

PyTorch versions should be backwards compatible but should be used
with the proper TorchVision version.

Currently, TVM supports PyTorch 1.4 and 1.3. Other versions may
be unstable.
"""

import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import onnx

from tvm.relay import data_dep_optimization as ddo
import struct
from PIL import Image

import pandas as pd
import pickle
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    # print(image.shape)
    image /= np.array([58.395, 57.12, 57.375])
    image = np.array(image)
    image = image.transpose((2, 0, 1))
    print(image.shape)
    image = image[np.newaxis, :]
    # print(type(image))
    
    return image

def extract_input_shape(model):
    """ Extract initial input's shape from loaded model

    Parameters
    --
    model : relay.Module

    Returns
    --
    shape : list

    """
    shape_info = model.graph.input[0].type.tensor_type.shape.dim

    shape = []
    for i in shape_info:
        str_info = str(i).strip()
        str_dim = str_info.split(' ')[1]
        shape.append(int(str_dim))
    return shape

def run_sparse(mod, params, shape_dict, target, ctx, bs_r, sparsity, mode):
    # mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    mod, params = ddo.bsr_dense.convert(mod["main"], params, (bs_r, 2), sparsity_threshold=sparsity, mode=mode)
    print(mod)

    return mod, params

def process_dict(model_name, p_dict):
    name = []
    h = []
    w = []
    sp = []
    avg = []
    dev = []
    label = []
    b_sp = []
    absp = []
    sbsp = []
    sblk = []
    sratio = []
    def cal_sp(p):
        row_sp = []
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
                # assume block size = (16, 2)
                if i%16 == 0 and len(tmp)!=0:
                    tmp_sp = tmp.count(0.0) / len(tmp)
                    b_sp.append(tmp_sp)
                    tmp.clear()
                    if r+1 < p.shape[0]:
                        tmp.append(p[r][i])
                        tmp.append(p[r+1][i])        
            tmp.clear()
            r += 1
        save_blk = 0
        save_ratio = 0
        for i in b_sp:
            if i == 1.0:
                save_blk+=1
        save_ratio = 1-(((p.size-np.count_nonzero(p))-(save_blk*32))/p.size)

        return np.mean(np.array(row_sp)), np.std(np.array(row_sp)), np.mean(np.array(b_sp)) ,np.std(np.array(b_sp)), save_blk ,save_ratio
    for key, val in p_dict.items():
        print(key)
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

    df = pd.DataFrame(data=h,columns=['height'])
    df['height'] = h
    df['width'] = w
    df['nnz'] = sp
    df['avg_row'] = avg
    df['std_row'] = dev
    df['avg_bsp'] = absp
    df['std_bsp'] = sbsp
    df['s_blk'] = sblk
    df['s_blk'] = sblk
    df['s_ratio'] = sratio
    # df['label'] = label
    print(df) 
    return df

target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
dtype = 'float32'

# get model
# model_name = "mobilenet_v1"
# model_path = "/home/hhliao/.cache/sparsezoo/c0673523-c5fe-40e4-bfdc-f999876371ea/model.onnx"
model_name = "resnet_v1_18"
model_path = "/home/hhliao/.cache/sparsezoo/ce190f33-2433-439a-8040-940c6d17993d/model.onnx"
# model_name = "resnet_v1_34"
# model_path = "/home/hhliao/.cache/sparsezoo/3757d663-867c-4440-8f34-8ea56565e011/model.onnx"
# model_name = "resnet_v1_50"
# model_path = "/home/hhliao/.cache/sparsezoo/acdf345e-2b22-47fd-bfbf-1fa84db71394/model.onnx"
# model_name = "vgg11"
# model_path = "/home/hhliao/.cache/sparsezoo/4c9264b6-43bb-4f8d-8de9-ac6f1f4c443e/model.onnx"
# model_name = "vgg16"
# model_path = "/home/hhliao/.cache/sparsezoo/f77b1101-cf95-4188-95da-4cabe03ede1e/model.onnx"
# model_name = "vgg19"
# model_path = "/home/hhliao/.cache/sparsezoo/01fb525c-938f-43b7-a40f-1622ced17af9/model.onnx"
# model_name = "InceptionV3"
# model_path = "/home/hhliao/.cache/sparsezoo/cd1503dd-a2e9-49df-9aa8-e532f7156976/model.onnx"
model = onnx.load(model_path)

# from onnxsim import simplify
# model, check = simplify(model)
# assert check, "Simplified ONNX model could not be validated"

# get shape
input_name = "input"
# input_name = model.graph.input[0].name

shape = extract_input_shape(model)
dummy_input = np.zeros(shape, dtype = dtype)
shape_dict = {input_name : dummy_input.shape}

# build relay
mod, params = relay.frontend.from_onnx(model, shape_dict, dtype=dtype)



######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
from graph_viewer import MixedFxper
from graph_rewriter import InsertTransHelper

threshold = 0.0

mhelper = InsertTransHelper(mod, params)
param_dict = mhelper.get_transform_list(threshold)

print("process")
df = process_dict(model_name, param_dict)
print("load")
clf = pickle.load(open('../sparse_schedule_test/XGB.sav', 'rb'))
print("pred")
pred = clf.predict(df)
print("print pred")
print(pred)


# import sys
# run_mode=int(sys.argv[1])
run_mode = -1
print("run_mode = ",run_mode)

# 0: csr(sparse_dense), 1: bsr(sparse_dense), 2: no compress(dense)
for i in range(0, len(pred)):
    if run_mode == -1:
        pass
    elif run_mode == 0:
        pred[i] = 0
    elif run_mode == 1:
        pred[i] = 1
    elif run_mode == 2:
        pred[i] = 2

mode = "bsr"
mod , params, param_dict = mhelper.transform_conv2d(threshold, pred)
print("mod")
print(mod)
print("end MOD")
mod , params = run_sparse(mod, params, shape_dict, target, ctx, 16, threshold, pred)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.

with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target, params=params)


#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

image_label = open("/home/hhliao/Datasets/ILSVRC_2012_img_val/label_data/value.txt", "rb")


######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime

start_run = 1
test_runs = 1
success = 0
success_list = []
predict_list = []
correct_list = []
from torchvision import transforms

for i in range(start_run-1):
    image_label.readline()

print("run test")
for i in range(start_run, start_run+test_runs):

    # img_url = "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true"
    # img_path = download_testdata(img_url, "cat.png", module="data")
    if(i < 10):
        img_path = "/home/hhliao/Datasets/ILSVRC_2012_img_val/ILSVRC2012_img_val/ILSVRC2012_val_0000000"+ str(i) +".JPEG"
    else:
        img_path = "/home/hhliao/Datasets/ILSVRC_2012_img_val/ILSVRC2012_img_val/ILSVRC2012_val_000000"+ str(i) +".JPEG"
    # lack of pic 34...
    if i == 34:
        continue
    img = Image.open(img_path).resize((224, 224))

    

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)

    # img = transform_image(img)
    # img = np.expand_dims(img, 0)

    dtype = "float32"
    print("create runtime")
    m = graph_runtime.create(graph, lib, ctx)

    # Set inputs
    print("set input")
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    m.set_input(**params)
    # Execute
    print("execute")
    m.run()
    # Get outputs
    print("get output")
    tvm_output = m.get_output(0)


    print("top - 1")
    # Get top-1 result for TVM
    top1_tvm = np.argmax(tvm_output.asnumpy()[0])
    tvm_class_key = class_id_to_key[top1_tvm]
    correct_value = int(image_label.readline())

    predict_list.append(top1_tvm)
    correct_list.append(correct_value)

    print("Predict top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
    print("Correct top-1 id: {}, class name: {}".format(correct_value, key_to_classname[class_id_to_key[correct_value]]))
    if top1_tvm == correct_value:
        print("Correct",i, "\n")
        success+=1
        success_list.append(i)
    else:
        print("Fail",i, "\n")

print("Accuracy:", success/test_runs)
print("predict_list: ", predict_list)
print("correct_list: ", correct_list)

# Run time evaluator

# ftimer = m.module.time_evaluator("run", ctx, repeat=5, number=5)
# prof_res = np.array(ftimer().results) * 1000
# print(
#     "%-20s %-19s (%s)"
#     % ("Runtime:", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
# )

