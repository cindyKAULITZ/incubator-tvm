import os

for i in range(-1, 3):
    print("run ",i)
    # os.system("python3 test_onnx_imagenet.py "+ str(i))
    os.system("python3 test_torch_imagenet.py "+ str(i))
