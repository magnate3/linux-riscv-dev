
import torch
import torchvision
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
resnet50 = models.resnet50(pretrained=True)

resnet18.eval()
resnet50.eval()


torch.onnx.export(resnet18, images[:1], "resnet18_fp32.onnx", opset_version=11)
torch.onnx.export(resnet50, images[:1], "resnet50_fp32.onnx", opset_version=11)

print("ResNet models converted to ONNX (FP32)!")