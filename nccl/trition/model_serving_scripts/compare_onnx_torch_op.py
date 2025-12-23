import torch
import torchvision
import numpy as np
import onnxruntime as ort
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 32
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Get a batch of images
images, labels = next(iter(testloader))

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()

with torch.no_grad():
    pytorch_output_resnet18 = resnet18(images[:1]).numpy()

ort_session_resnet18 = ort.InferenceSession("./onnx_files/resnet18_fp32.onnx")
ort_inputs_resnet18 = {ort_session_resnet18.get_inputs()[0].name: images[:1].numpy()}
onnx_output_resnet18 = ort_session_resnet18.run(None, ort_inputs_resnet18)[0]

resnet18_onnx_match = np.allclose(pytorch_output_resnet18, onnx_output_resnet18, rtol=1e-5, atol=1e-5)

print(f"Inference results match: {resnet18_onnx_match}")
print(pytorch_output_resnet18)
print(onnx_output_resnet18)

