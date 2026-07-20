import torch
import torch_tensorrt
import torchvision.models as models
#import timm
import time
import numpy as np
import torch.backends.cudnn as cudnn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

#model = timm.create_model('resnet18',pretrained=True)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

model = model.eval().to("cuda")
detections_batch = model(torch.randn(128, 3, 224, 224).to("cuda"))
    
model = model.eval().to("cuda")

traced_model = torch.jit.trace(model, torch.randn((1,3,224,224)).to("cuda"))
torch.jit.save(traced_model, "traced_model.jit.pt")
print('***saved torchscript model***\n')

trt_model = torch_tensorrt.compile(model, 
    inputs= [torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions= { torch_tensorrt.dtype.half, torch.float, torch.half} # Run with FP16
)

torch.jit.save(trt_model, "models/resnet18_trt/1/model.plan")
print('***saved tensorrt model***\n')

print('----------------------Starting model validation----------------------\n')

model = torch.jit.load("models/resnet18_trt/1/model.plan")
print(model(torch.randn(1,3,224,224).float().to('cuda')), '\n')

print('----------------------Model loads and works----------------------\n')
print('Exiting...')
