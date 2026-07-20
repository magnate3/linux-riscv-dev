import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# --- 1. 配置参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
PRUNING_RATIO = 0.3  # 物理剪掉 30% 的通道
# 确保此路径指向解压后的 imagenetv2-matched-frequency-format-val 文件夹
V2_PATH = "./imagenetv2/imagenetv2-matched-frequency-format-val" 

# --- 2. 加载预训练模型 ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)

# --- 3. 准备 ImageNet-v2 数据加载器 ---
transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#v2_dataset = datasets.ImageFolder(V2_PATH, transform=transform)
## 3. 核心校准逻辑
## 检查 dataset.classes 是否与官方 meta 一致
## 如果你的文件夹名是 "0", "1", "2"...
## 它们被读入后 dataset.classes 会变成 ['0', '1', '10', '100'...] (字符串排序)
## 我们需要强制让文件夹 "0" 对应标签 0，文件夹 "1" 对应标签 1
#v2_dataset.class_to_idx = {str(i): i for i in range(1000)}
#v2_dataset.classes = [str(i) for i in range(1000)]
#v2_loader = torch.utils.data.DataLoader(v2_dataset, batch_size=BATCH_SIZE, shuffle=False)
class NumericImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # 强制按数字大小排序：0, 1, 2... 999
        classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))], 
                         key=lambda x: int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# 使用自定义的加载器
v2_dataset = NumericImageFolder(root=V2_PATH, transform=transform)
v2_loader = torch.utils.data.DataLoader(v2_dataset, batch_size=BATCH_SIZE, shuffle=False)
model.eval()
print(f"model参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
with torch.no_grad():
    inputs, labels = next(iter(v2_loader))
    outputs = model(inputs.to(device))
    _, predicted = torch.max(outputs, 1)

    print(f"预测值: {predicted[:5].tolist()}")
    print(f"真实标签: {labels[:5].tolist()}")

    acc = (predicted == labels.to(device)).sum().item() / labels.size(0)
    print(f"单 Batch 准确率: {acc*100:.2f}%")
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

# --- 4. 结构化剪枝 (最新 API) ---
print(f"剪枝前准确率: {evaluate(model, v2_loader):.2f}%")
def prune(model, loader):
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    # TaylorImportance 结合数据梯度评估通道重要性
    importance = tp.importance.TaylorImportance()
    
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # 不剪最后分类层
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (7, 7): # 排除第一层大卷积
            ignored_layers.append(m)
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        ch_sparsity=PRUNING_RATIO,
        ignored_layers=ignored_layers,
    )
    
    # 梯度收集：Taylor 准则需要反向传播
    model.zero_grad()
    for i, (inputs, targets) in enumerate(loader):
        if i >= 10: break # 使用约 640 张图校准重要性
        outputs = model(inputs.to(device))
        loss = torch.nn.functional.cross_entropy(outputs, targets.to(device))
        loss.backward()
    
    # 执行物理剪枝
    pruner.step()
    print(f"剪枝后参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"剪枝后准确率: {evaluate(model, loader):.2f}%")
    
    # --- 5. 导出 ONNX (用于 ncnn 或 TensorRT) ---
    model.eval()
    onnx_path = "ncnn/resnet50_pruned_v2.onnx"
    torch.onnx.export(model, example_inputs, onnx_path, opset_version=12, 
                      input_names=['input'], output_names=['output'])
    print(f"模型已导出至: {onnx_path}")
prune(model,v2_loader)
