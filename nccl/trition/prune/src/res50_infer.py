import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet50
from tqdm import tqdm
#import registry
# ====================== 配置 ======================
DATA_ROOT = "./tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/"
MODEL_PATH = "models/best.pth"
ORIG_MODEL_PATH = "models/resnet50-0676ba61.pth"
BATCH_SIZE = 128
imageNet_total_cls=200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================

# 标签字典：数字 -> 类别名
def get_label_map(root):
    with open(os.path.join(root, "wnids.txt")) as f:
        wnids = [line.strip() for line in f]
    wnid2name = {}
    with open(os.path.join(root, "words.txt")) as f:
        for line in f:
            wnid, name = line.strip().split("\t", 1)
            wnid2name[wnid] = name
    return {i: wnid2name[wnid] for i, wnid in enumerate(wnids)}

label_map = get_label_map(DATA_ROOT)

# 预处理
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 数据集
val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
# 获取每张图片路径（用于显示文件名）
img_paths = [s[0] for s in val_dataset.samples]
#class_to_idx = val_dataset.class_to_idx
idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
def evalution(model):
    # 开始测试
    correct = 0
    total = 0
    
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Testing")):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # 本批次在整个数据集中的起始索引
            start = batch_idx * BATCH_SIZE
            for i in range(len(images)):
                idx = start + i
                img_path = img_paths[idx]
                img_name = os.path.basename(img_path)
    
                gt = labels[i].item()
                pred = preds[i].item()
    
                gt_name = label_map[gt]
                pred_name = label_map[pred]

                gt_label = idx_to_class[gt]
                pred_label = idx_to_class[pred]
    
                ok = gt == pred
                correct += ok
                total += 1
                if ok :
                    print(f"{img_name}\t{gt_name}\t{gt_label}\t{pred_name}\t{pred_label}\t{pred}\t{ok}")
    acc = 100.0 * correct / total
    print("-" * 80)
    print(f"Total: {total}  Correct: {correct}  Acc: {acc:.2f}%")
def save_prune():
    num_classes = imageNet_total_cls
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    #model = models.resnet50(pretrained=False,num_classes=imageNet_total_cls)
    #model=checkpoint['model']
    #print(model["model_state_dict"])
    #model = resnet50(pretrained=False, num_classes=imageNet_total_cls).state_dict()
    #print(checkpoint['model'])
    #model.load_state_dict(checkpoint['model'])
    #
    #Linear(in_features=1413, out_features=1000, bias=True)
    #print(model.fc)
    model = checkpoint['model']
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(DEVICE)
    print(model.fc)
    #print(checkpoint.optimizer)
    #print(checkpoint.lr_scheduler)
    model.eval()
    dummies = torch.randn(1, 3, 224, 224).to(DEVICE)
    torch.onnx.export(
        model, 
        dummies,
        'models/model.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
def prune_evalution():
    num_classes = imageNet_total_cls
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    model=checkpoint['model']
    #
    #Linear(in_features=1413, out_features=1000, bias=True)
    #print(model.fc)
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(DEVICE)
    model.eval()
    evalution(model)
    dummies = torch.randn(1, 3, 224, 224).to(DEVICE)
    torch.onnx.export(
        model, 
        dummies,
        'models/model.onnx',
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11
    )
def origin_evalution():
    num_classes = 1000
    #num_classes = imageNet_total_cls
    #model = models.resnet50(pretrained=False)
    model = models.resnet50(weights =  models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(ORIG_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    evalution(model)
#prune_evalution()
#origin_evalution()
save_prune()
