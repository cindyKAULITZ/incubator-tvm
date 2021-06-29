from sparsezoo import Zoo
from sparsezoo.models.classification import mobilenet_v1

models = Zoo.search_models( domain="cv", sub_domain="classification", 
                            architecture="resnet_v1", sub_architecture="18",
                            framework="pytorch",dataset="imagenet", 
                            optim_name="pruned")
print(models)

models[0]. download()

print(models[0].onnx_file.downloaded_path())
