import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
from torchvision.models import resnet50
from torchvision import models
import torch_pruning as tp
from tinyimagenet import TinyImageNet
import torchvision.transforms as transforms
from pathlib import Path
imageNet_total_cls=1000

def create_dataloader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
        )
    return dataloader

def create_imagenet_dataloader():
    splits = ["val","test","train"]
    split=splits[0]
    tinyimagenet_data_dir = "./tinyimagenet/torchvision/tinyimagenet/"
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(TinyImageNet.mean,TinyImageNet.std)]
        )

    dataset= TinyImageNet(Path(tinyimagenet_data_dir),split=split,imagenet_idx=True,transform=transform)
    #dataset= TinyImageNet(Path(tinyimagenet_data_dir),split=split,imagenet_idx=False,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32)
    print(f"TinyImageNet, split {split}, loading all samples with dataloader:")
    return dataloader
def predict(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    accuracy = 100 * correct / total
    loss = running_loss / len(dataloader)

    return loss, accuracy
def test3(device):
    num_classes = imageNet_total_cls
    model = resnet50(weights =  models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    #model = resnet50(pretrained=True,weights = models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    #model_path ="models/best.pth"
    model_path ="models/resnet50-0676ba61.pth"
    model.load_state_dict(torch.load(model_path))
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loader = create_imagenet_dataloader()
    test_loss, test_accuracy = predict(model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Accuracy: {test_accuracy:.2f}%")
def test4_pruned_model(device):
    num_classes = imageNet_total_cls
    model_path ="models/best.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model=checkpoint['model']
    #print(model_state_dict)
    #print(model_state_dict)
    #print(f"num class {model.num_classes}")
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loader = create_imagenet_dataloader()
    test_loss, test_accuracy = predict(model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Accuracy: {test_accuracy:.2f}%")
    #torch.save(model.state_dict(), 'models/prune.pt') 
    dummies = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, 
        dummies,
        'models/model.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
        
def test2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = imageNet_total_cls
    model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
    #print(model.fc)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_loader = create_imagenet_dataloader()
    test_loss, test_accuracy = predict(model, val_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Accuracy: {test_accuracy:.2f}%")
    #print(model)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test3(device)
    test4_pruned_model(device)
    test2()
