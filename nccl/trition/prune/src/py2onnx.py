import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from torchvision.models import resnet50
import os
import onnx
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
batch_size = 128
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
# 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
n_class = 10
#model = resnet50(pretrained=True)
model_path='/pytorch/prune/resnet50-prune.pth'
#weights_path ='/pytorch/prune/resnet50-prune.pth'
#model.load_state_dict(torch.load(model_path,weights_only=False))
#model.eval()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(pretrained=False)
torch.load(model_path,weights_only=False)
#model.load_state_dict(torch.load(weights_path, map_location=device))
#model = model.to(device)

iterator = iter(test_loader)
images, batch_targets = next(iterator)
print(f"data shape : {images.shape}")  # ouput: torch.Size([10, 3])
print(f"data label: {batch_targets[:3]}")
def torch2onnx(model):
    img=None
    half = False # True if FP16,False if FP32
    #half = True  # True if FP16,False if FP32
    model.eval()
    if half:
        model = model.to(device).half()
        img = images.to(device).half()
    else:
        model = model.to(device)
        img = images.to(device)
    print(img.shape)
    onnx_path = './models'
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)
    f = os.path.join(onnx_path, 'res50_model.onnx')
    torch.onnx.export(model, img, f,
    verbose=False, 
    opset_version=12,
    training=torch.onnx.TrainingMode.EVAL,
    do_constant_folding=False,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},# shape(1,3,640,640)
    'output': {0: 'batch', 1: 'anchors'}# shape(1,25200,85)
    } 
    )
    model_onnx = onnx.load(f)
    # onnx.checker.check_model(model_onnx)
    print(model_onnx.graph.output)
def save_onnx_model(model):
    dummy_inputs = torch.randn(1, 3, 32, 32)
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    onnx_model_path = f"resnet_trained_for_cifar10.onnx"
    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_model_path,
        export_params=True,
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
#save_onnx_model(model)
torch2onnx(model)
