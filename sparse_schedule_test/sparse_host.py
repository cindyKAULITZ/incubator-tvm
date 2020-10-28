################################################################
# model created via distiller.models
# load checkpoint created by distiller -> compress_classifier
# build model via from_pytorch()
################################################################
import pickle
import numpy as np

import torchvision.transforms as transforms
import torch
# import torchvision
import sys
sys.path.insert(1, '/home/hhliao/distiller/distiller/')

from utils import normalize_module_name
from models import create_model

import tvm
import tvm.relay as relay
# from distiller.models.cifar10.resnet_cifar import resnet20_cifar


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
    data_dict = unpickle("./cifar-10-batches-py/test_batch")
    input_img =data_dict[b'data']
    image_label = data_dict[b'labels']

# Save model to target directory
def save(target_dir, model_name):
    with open(target_dir + '/' + model_name + '.ll', 'w') as _f:
        _f.write(lib.get_source())
    with open(target_dir + '/' + model_name + '.graph', 'w') as _f:
        _f.write(graph)
    with open(target_dir + '/' + model_name + '.params', 'wb') as _f:
        _f.write(relay.save_param_dict(params))
    print("save finish")


# set target
src = 'onnx'
target_host = 'llvm'
target = 'llvm'
# target = 'llvm -target=riscv64-unknown-elf -system-lib'


###########################
# load model from pytorch
###########################

model_path = 'checkpoint.pth.tar'
dataset = 'cifar10'
model_name = 'resnet20_cifar'

input_shape = [1, 3, 32, 32]
input_data = torch.randn(input_shape)
input_name = 'input0'  # only one input, set it to this name

shape_list = [(input_name, input_shape)]

model = load_model(model_path, model_name, dataset)
scripted_model = torch.jit.trace(model, input_data).eval()

mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
input_name = mod['main'].params[0].name_hint
print(input_name)

#######################
# load model from onnx
#######################

# tar = '/home/hhliao/prune_CNN/model.onnx' 
#tar = './model.onnx' 
#model = onnx.load(tar)

# tmp_in = [0]*32*32*3
# x = np.array(tmp_in)
# x.resize((1, 3, 32, 32))
# shape_dict = {input_name: x.shape}
# mod, params = relay.frontend.from_onnx(model, shape_dict)



# schedule
def schedule_injective_default(_, outs, target):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    return s



# tvm.relay.op.register_schedule("nn.bias_add", schedule_injective_default, level=15)
# relay.op.op.register_strategy("nn.conv2d", schedule_conv2d, level=15)

# We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
desired_layouts = {'nn.conv2d': ['NCHW', 'default']}

# Convert the layout to NCHW
# RemoveUnunsedFunctions is used to clean up the graph.
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                relay.transform.ConvertLayout(desired_layouts)])

with tvm.transform.PassContext(opt_level=0):
    mod = seq(mod)
# build routine
with relay.build_config(opt_level = 0):
    graph, lib, params = relay.build(mod, target,target_host=target_host, params=params)
    # lib = relay.build(mod, target, target_host = target_host, params = params)
#with relay.build_config(opt_level = 0):
#    graph, lib, params = relay.build(mod, target, params=params)

# print('graph', graph)
# print('lib', lib.type_key)
# for key in params.keys():
#     print(key)




from tvm.contrib import graph_runtime
# from tvm.contrib.debugger import debug_runtime as graph_runtime

test_runs = 1
start_run = 0
success = 0
success_list = []

ctx = tvm.cpu(0)
dtype = 'float32'

data_dict = unpickle("/home/hhliao/distiller/examples/data.cifar10/cifar-10-batches-py/test_batch")
input_img = data_dict['data']
image_label = data_dict['labels']

input_img = np.vstack(input_img).reshape(-1, 3, 32, 32)
input_img = input_img.transpose((0, 2, 3, 1))  # convert to HWC
input_img = transform_img_to_tensor(input_img)


##########################################
# check lib's funtion name.
# try if save graph,lib,params separatly.
m = graph_runtime.create(graph, lib, ctx)

# m = graph_runtime.GraphModule(lib['default'](ctx))

# run test
for i in range(start_run, start_run + test_runs):

    print("i = {}, label = {}".format(i, image_label[i])) 
    input_pic = input_img[i]
    
    m.set_input(input_name, tvm.nd.array(input_pic.astype(dtype)))
    m.set_input(**params)
    
    # execute
    m.run()


    ############################################
    # Get outputs
    # Check answer with predict output
    # tvm_np_output = intrip.evaluate()(tvm.nd.array(input_pic.astype(dtype)), **params).asnumpy()

    tvm_output = m.get_output(0)
    tvm_np_output=tvm_output.asnumpy()[0]
    top1 = np.argmax(tvm_np_output)
    value = int(image_label[i])

    print("predict : ", top1)
    print("answer : ", value)
    
    if(value == top1):
        print("Correct.")
        success_list.append(i+1)
        success += 1
    else:
        print("Fail.")
        # pass

# print("Sucess_list: ", success_list)
print("Test model : {}\nTest runs : {}".format(model_name, test_runs))
print("Accuracy rate: ", success/test_runs)




