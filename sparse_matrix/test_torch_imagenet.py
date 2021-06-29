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
import torch
import torchvision
import torch.nn as nn

from tvm.relay import data_dep_optimization as ddo
import struct
import pandas as pd
import pickle
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from distiller.utils import normalize_module_name
from distiller.models import create_model
# from distiller.models.cifar10.resnet_cifar import resnet20_cifar
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from typing import Union, List, Dict, Any, cast

# RESNET

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# MOBILENET
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

#VGGs
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def load_model(path, name, dataset_name):
    load_mod = create_model(False, dataset_name, name, parallel = False, device_ids = -1)
    checkpoint = torch.load(path, map_location = torch.device("cpu"))
    checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
    load_mod.load_state_dict(checkpoint['state_dict'], False)
    load_mod.to("cpu")
    load_mod.eval()

    return load_mod
def run_sparse(mod, params, shape_dict, target, ctx, bs_r, sparsity,mode):
    # mod, params = ddo.simplify_fc_transpose.convert(mod["main"], params)
    mod, params = ddo.bsr_dense.convert(mod["main"], params, (bs_r, 1), sparsity_threshold=sparsity, mode=mode)
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
    # df.to_csv(path_or_buf="./raw_dict/bsp_up75_random_gen_"+model_name+".csv", mode='a', header=False, index=False)


######################################################################
# Load a pretrained PyTorch model
# -------------------------------
dataset = "Imagenet"

# mobilenet_v1
# model_path = "/home/hhliao/.cache/sparsezoo/c0673523-c5fe-40e4-bfdc-f999876371ea/pytorch/model.pth"
# torch_model = MobileNet().cpu()

#resnet_v1_18
model_path = "/home/hhliao/.cache/sparsezoo/ce190f33-2433-439a-8040-940c6d17993d/pytorch/model.pth"
torch_model = ResNet(BasicBlock, [2, 2, 2, 2]).cpu()

#resnet_v1_34
# model_path = "/home/hhliao/.cache/sparsezoo/3757d663-867c-4440-8f34-8ea56565e011/pytorch/model.pth"
# torch_model = ResNet(BasicBlock, [3, 4, 6, 3]).cpu()

#resnet_v1_50
# model_path = "/home/hhliao/.cache/sparsezoo/acdf345e-2b22-47fd-bfbf-1fa84db71394/pytorch/model.pth"
# torch_model = ResNet(Bottleneck, [3, 4, 6, 3]).cpu()

#vgg11
# model_path = "/home/hhliao/.cache/sparsezoo/4c9264b6-43bb-4f8d-8de9-ac6f1f4c443e/pytorch/model.pth"
# torch_model = _vgg('vgg11', 'A', False, True, True)

#vgg16
# model_path = "/home/hhliao/.cache/sparsezoo/f77b1101-cf95-4188-95da-4cabe03ede1e/pytorch/model.pth"
# torch_model = _vgg('vgg16', 'D', False, True, True)

#vgg19
# model_path = "/home/hhliao/.cache/sparsezoo/01fb525c-938f-43b7-a40f-1622ced17af9/pytorch/model.pth"
# torch_model = _vgg('vgg19', 'E', False, True, True)

#InceptionV3
# model_path = "/home/hhliao/.cache/sparsezoo/cd1503dd-a2e9-49df-9aa8-e532f7156976/pytorch/model.pth"



state_dict = torch.load(model_path, map_location = torch.device("cpu"))
torch_model.load_state_dict(state_dict,False)


# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(torch_model, input_data).eval()

######################################################################
# Load a test image
# -----------------
# Classic cat example!
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

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

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)


######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
from graph_viewer import MixedFxper
from graph_rewriter import InsertTransHelper

print(mod)
threshold = 0.0

# print(mod)
mhelper = InsertTransHelper(mod, params)
param_dict = mhelper.get_transform_list(threshold)


print("process")
df = process_dict("tmp_weights", param_dict)
print("load")
clf = pickle.load(open('../sparse_schedule_test/XGB.sav', 'rb'))
print("pred")
pred = clf.predict(df)
print("print pred")
print(pred)


import sys
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
mod , params = run_sparse(mod, params, shape_list, target, ctx, 16, threshold, pred)



######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.

with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target, params=params)


######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime


dtype = "float32"
m = graph_runtime.create(graph, lib, ctx)

# Set inputs
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
m.set_input(**params)
# Execute
m.run()
# Get outputs
tvm_output = m.get_output(0)

ftimer = m.module.time_evaluator("run", ctx, repeat=5, number=5)
prof_res = np.array(ftimer().results) * 1000
print(
    "%-20s %-19s (%s)"
    % ("Runtime:", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
)

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

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.asnumpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = torch_model(torch_img)

    # Get top-1 result for PyTorch
    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))