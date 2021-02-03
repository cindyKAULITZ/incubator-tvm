import numpy as np
import json
import sys
import tvm
import onnx
import tvm.relay as relay
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import struct

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

def create_mod(source, model, dtype='float32'):
    """ Create relay.Moudle from mxnet or onnx pre-trained model

    Parameters
    --
    source : string
        onnx / mxnet
    
    model : string
        onnx : path
        mxnet : model name

    dtype : string
        as target dtype for input graph and params
    
    
    Returns
    --
    mod : relay.Module
    
    params : dict or str to NDArray
             input parameters of graph
    
    """
    if source == 'onnx':
        # get model
        # model_path = download_testdata(model_url, "super_resolution.onnx", module="onnx")
        model = onnx.load(model)

        # get shape
        input_name = model.graph.input[0].name
        shape = extract_input_shape(model)
        dt = float if dtype != 'int8' else np.int8
        dummy_input = np.zeros(shape, dtype = dt)
        shape_dict = {input_name : dummy_input.shape}

        # build relay
        mod, params = relay.frontend.from_onnx(model, shape_dict, dtype=dtype)

    elif source == 'mxnet':
        # get model
        model = get_model(model, pretrained = True)

        # get shape
        input_name = 'data'
        shape = [1, 3, 224, 224]
        dummy_input = np.zeros(shape, dtype = float)
        shape_dict = {input_name : dummy_input.shape}

        # build relay
        mod, params = relay.frontend.from_mxnet(model, shape_dict, dtype=dtype)

    return mod, params

# set target
# onnx ------------------------------------
src = 'onnx'
tar = '/home/chchang/workspaces/NewRVV/Sample_DLR/model/LeNet_Mnist-img2col/LeNet_Mnist.nnef.onnx' 

# target = 'llvm'
target = 'llvm -system-lib'

mod, params = create_mod(src, tar, 'float32')
print('mod : ', mod)

from graph_viewer import MixedFxper
from graph_rewriter import InsertTransHelper
mhelper = InsertTransHelper(mod)
mhelper.transform_conv2d()

# build
with tvm.transform.PassContext(opt_level=0):
    # sym = seq(sym)
    print(mod)
# with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target=target, params=params)
print("build finish")


def transform_image(image):
    # print(image.shape)
    # image = np.array(image) - np.array([123., 117., 104.])
    # image /= np.array([58.395, 57.12, 57.375])
    image = np.array(image)
    # image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    # print(type(image))
    
    return image

def get_cifar10_datas_labels(path, batch_size, shape):
    target_imgs = []
    target_labels = []

    # laod datas
    with open(path + '/cifar10_test_data_10000x224x224_float32.bin','rb') as image_test:

        # start to load
        for i in range(batch_size):
            # load img
            target_img = np.zeros([150528],dtype=float)
            for i in range (150528): 
                buffer = image_test.read(4)
                c = struct.unpack("f", buffer)[0]
                target_img[i] = c
            # reshape
            target_img = target_img.reshape(1, 224, 224, 3) if shape == [1, 224, 224, 3] else target_img.reshape(1, 3, 224, 224) 
            target_imgs.append(target_img)

    # laod labels
    with open(path + '/test_batch.bin','rb') as image_test_label:

        # start to load
        for i in range(batch_size):
            buffer = image_test_label.read(1)
            image_test_label.read(3072)
            target_label = struct.unpack("b", buffer)[0]
            target_labels.append(target_label)
            # print("target_label: ", target_labels)
              
    return target_imgs, target_labels


# # build routine
# with relay.build_config(opt_level = 0):
#     graph, lib, params = relay.build(mod, target, params=params)
# print('graph', graph)

from tvm.contrib import graph_runtime
    # from tvm.contrib.debugger import debug_runtime as graph_runtime

test_runs = 10
success = 0
success_list = []

img_input=[0]*784
target=[0]*10

image_test = open("/home/hhliao/Datasets/mnist/t10k-images-idx3-ubyte", "rb")
image_test_label = open("/home/hhliao/Datasets/mnist/t10k-labels-idx1-ubyte", "rb")
image_test.seek(16,1)
image_test_label.seek(8,1)


x = np.array(img_input)
x.resize((1,784))

input_name = "input_Placeholder"
shape_dict = {input_name: x.shape}

for i in range(0, test_runs):
    image_data = image_test.read(784)
    image_data_label = image_test_label.read(1)

    for n in range(0,784):
        #   print(image_data[num])
        # if(image_data[n] < 128):
        #     img_input[n] = 0
        # else:
        #     img_input[n] = 1
        img_input[n] = image_data[n]/255
        #print(num,struct.unpack('<B', image_data[num]),img_input[num])
        #print(img_input)
    result = image_data_label[0]




    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)

    input_name = "input_Placeholder"
    m.set_input(input_name, tvm.nd.array(np.array(img_input).astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()

    ############################################
    # Get outputs
    # Check answer with predict output
    tvm_output = m.get_output(0)
    tvm_np_output=tvm_output.asnumpy()[0]
    top1 = np.argmax(tvm_np_output)

    value = result
    # value = int(image_label.readline())
    print("=== test data %d ===" % i)
    # print('\npredict top-1:', top1, synset[top1])
    print("predict : ", top1)
    # print('label: ', value, synset[value])
    print("answer  : ", value)
    if(value == top1):
        print("Correct.")
        success_list.append(i+1)
        success += 1
    else:
        print("Fail.")

print("sucess_list", success_list)
print("accuracy rate: ", success/test_runs)