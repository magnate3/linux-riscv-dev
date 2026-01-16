# export_model.py
import torch
import torchvision.models as models

# 加载预训练的ResNet-18模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
model.cuda() # 将模型移动到GPU

# 创建一个示例输入张量
# 形状必须与模型预期的一致：(批量大小, 通道数, 高度, 宽度)
dummy_input = torch.randn(1, 3, 224, 224, device="cuda")

# 使用TorchScript跟踪模型
traced_model = torch.jit.trace(model, dummy_input)

# 保存跟踪后的模型
traced_model.save("models/resnet18_pytorch/1/model.pt")

print("models/resnet18_pytorch/1/model.pt")
