import torch
import time
import torchvision
import numpy as np
import onnxruntime as ort
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gc

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 32
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Get a batch of images
# images, labels = next(iter(testloader))

def clear_memory():
    gc.collect()  # Python garbage collection
    if "cuda" in ort.get_available_providers():
        torch.cuda.empty_cache()

# To simulate a cold start
clear_memory()
ort_session_resnet18 = ort.InferenceSession("./onnx_files/resnet18_fp32.onnx", providers=["CUDAExecutionProvider"])

times = []
total_batches = 50 

for i, (images, _) in enumerate(testloader):
    if i == total_batches:
        break

    ort_inputs = {ort_session_resnet18.get_inputs()[0].name: images[:1].numpy()}
    
    start_time = time.perf_counter()
    ort_session_resnet18.run(None, ort_inputs)
    end_time = time.perf_counter()
    
    times.append(end_time - start_time)

avg_latency = np.mean(times) * 1000  # Convert to ms
throughput = batch_size / np.mean(times)  # Images per second

print(f"Avg Latency per Batch: {avg_latency:.2f} ms")
print(f"Throughput: {throughput:.2f} images/sec")

