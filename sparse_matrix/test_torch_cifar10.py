################################################################
# model created via distiller.models
# load checkpoint created by distiller -> compress_classifier
# build model via from_pytorch()
################################################################
import pickle
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import torchvision.transforms as transforms
import torch
# import torchvision

import tvm
import tvm.relay as relay

from distiller.utils import normalize_module_name
from distiller.models import create_model
# from distiller.models.cifar10.resnet_cifar import resnet20_cifar

from PIL import Image
from tvm.relay import data_dep_optimization as ddo
import struct


def load_model(path, name, dataset_name):
    load_mod = create_model(False, dataset_name, name, parallel = False, device_ids = -1)
    checkpoint = torch.load(path, map_location = torch.device("cpu"))
    checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
    load_mod.load_state_dict(checkpoint['state_dict'], False)
    load_mod.to("cpu")
    load_mod.eval()

    return load_mod

def transform_img_to_tensor(img_list):
    trans_imgs = []

    for img_x in img_list:
        img_x = img_x.reshape(32, 32, 3) 

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img_x = test_transform(img_x)
        # img_x = img_x[np.newaxis, :]
        img_x = np.expand_dims(img_x, 0)
        trans_imgs.append(img_x)
    
    return trans_imgs

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')

    return dict

def test_cifar_10():
    data_dict = unpickle("/home/hhliao/distiller/examples/data.cifar10/cifar-10-batches-py/test_batch")
    input_img =data_dict[b'data']
    image_label = data_dict[b'labels']

# schedule
def schedule_injective_default(_, outs, target):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    return s
# tvm.relay.op.register_schedule("nn.bias_add", schedule_injective_default, level=15)

# Save model to target directory
def save(target_dir, model_name):
    with open(target_dir + '/' + model_name + '.ll', 'w') as _f:
        _f.write(lib.get_source())
    with open(target_dir + '/' + model_name + '.graph', 'w') as _f:
        _f.write(graph)
    with open(target_dir + '/' + model_name + '.params', 'wb') as _f:
        _f.write(relay.save_param_dict(params))
    print("save finish")


def run_sparse(mod, params, shape_dict, target, ctx, bs_r, sparsity, mode):
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
        save_blk = 0
        save_ratio = 0
        for i in b_sp:
            if i == 1.0:
                save_blk+=1
        save_ratio = 1-(((p.size-np.count_nonzero(p))-(save_blk*32))/p.size)

        return np.mean(np.array(row_sp)), np.std(np.array(row_sp)), np.mean(np.array(b_sp)) ,np.std(np.array(b_sp)), save_blk ,save_ratio
    for key, val in p_dict.items():
        # print(key)
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
    # df['s_blk'] = sblk
    df['s_blk'] = sblk
    df['s_ratio'] = sratio
    print(df) 
    return df
    # df.to_csv(path_or_buf="./raw_dict/bsp_up75_random_gen_"+model_name+".csv", mode='a', header=False, index=False)


# set target
ctx = tvm.cpu(0)
dtype = 'float32'
target_host = 'llvm'
target = 'llvm'
# target = 'llvm -target=riscv64-unknown-elf -system-lib'

###########################
# load model from pytorch
###########################
dataset = 'cifar10'

#resnet20
# model_path = '/home/hhliao/distiller/examples/classifier_compression/logs/resnet20_sparsity_6574/checkpoint.pth.tar'           
# model_path = '/home/hhliao/distiller/examples/classifier_compression/logs/2021.03.16-002926/best.pth.tar' #91.87           
#resnet56
# model_path = '/home/hhliao/distiller/examples/classifier_compression/logs/resnet56_sparsity_8994/best.pth.tar'
#vgg11
model_path = "/home/hhliao/distiller/examples/classifier_compression/logs/2021.06.24-180402/best.pth.tar" # half dense/half 80%
# model_path = "/home/hhliao/distiller/examples/classifier_compression/logs/vgg11_sparsity_8319/best.pth.tar"
# model_path = "/home/hhliao/distiller/examples/classifier_compression/logs/vgg11_train_clean/best.pth.tar"
# vgg16    
# model_path = '/home/hhliao/distiller/examples/classifier_compression/logs/2021.03.16-033224/best.pth.tar'  #93.46
# model_path = "/home/hhliao/distiller/examples/classifier_compression/logs/vgg16_train_clean/best.pth.tar"        


# model_name = 'resnet56_cifar'
# model_name = 'resnet20_cifar'
model_name = 'vgg11_cifar'
# model_name = 'vgg16_cifar'

input_shape = [1, 3, 32, 32]
input_data = torch.randn(input_shape)
input_name = 'input0'  # only one input, set it to this name

shape_list = [(input_name, input_shape)]

model = load_model(model_path, model_name, dataset)
scripted_model = torch.jit.trace(model, input_data).eval()


mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
input_name = mod['main'].params[0].name_hint
print(input_name)

##########################
# Replace to Sparse_Dense
##########################

from graph_viewer import MixedFxper
from graph_rewriter import InsertTransHelper

threshold = 0.000
mhelper = InsertTransHelper(mod, params)
param_dict = mhelper.get_transform_list(threshold)


df = process_dict(model_name, param_dict)
clf = pickle.load(open('../sparse_schedule_test/XGB.sav', 'rb'))
pred = clf.predict(df)
print(pred)

mode = "bsr"
mod , params, param_dict = mhelper.transform_conv2d(threshold, pred)
print("mod")
print(mod)
print("end MOD")
mod , params = run_sparse(mod, params, shape_list, target, ctx, 16, threshold, pred)


# build model
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target,target_host=target_host, params=params)


# export IR
# path_ll = "./a.ll"
# with open(path_ll, "w") as f:
#     f.write(lib.get_source())


##########################
# Datasets Setting
##########################

start_run = 0
test_runs = 1
success = 0
success_list = []
predict_list = []
correct_list = []

# datasets setting
data_dict = unpickle("/home/hhliao/distiller/examples/data.cifar10/cifar-10-batches-py/test_batch")
input_img = data_dict['data']
image_label = data_dict['labels']

input_img = np.vstack(input_img).reshape(-1, 3, 32, 32)
input_img = input_img.transpose((0, 2, 3, 1))  # convert to HWC
input_img = transform_img_to_tensor(input_img)


##########################################
# Create Runtime
# check lib's funtion name.
# try if save graph,lib,params separatly.
##########################################

# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime

m = graph_runtime.create(graph, lib, ctx)


# run test
for i in range(start_run, start_run + test_runs):

    print("i = {}, label = {}".format(i, image_label[i])) 
    input_pic = input_img[i]
    m.set_input(input_name, tvm.nd.array(input_pic.astype(dtype)))
    m.set_input(**params)
    
    # execute
    m.run()

    #################################################################################################
    # Get outputs
    # Check answer with predict output
    # tvm_np_output = intrip.evaluate()(tvm.nd.array(input_pic.astype(dtype)), **params).asnumpy()
    #################################################################################################

    tvm_output = m.get_output(0)
    tvm_np_output=tvm_output.asnumpy()[0]
    top1 = np.argmax(tvm_np_output)
    value = int(image_label[i])

    print("predict : ", top1)
    print("answer : ", value)
    
    predict_list.append(top1)
    correct_list.append(value)
    if(value == top1):
        print("Correct.")
        success_list.append(i+1)
        success += 1
    else:
        print("Fail.")




# print out results and information
print("Test model : {}\nTest runs : {}".format(model_name, test_runs))
print("Accuracy rate: ", success/test_runs)
print("Sucess_list: ", success_list)
print("predict_list: ", predict_list)
print("correct_list: ", correct_list)

# print out cost time
ftimer = m.module.time_evaluator("run", ctx, repeat=5, number=5)
prof_res = np.array(ftimer().results) * 1000
print(
    "%-20s %-19s (%s)"
    % ("Runtime:", "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
)



