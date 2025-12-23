import time
import numpy as np
import tritonclient.http as httpclient
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

def prepare_data():
    """Prepare CIFAR10 dataset with the same preprocessing as your original code"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return testloader

def run_inference():
    """Run inference using Triton client"""
    # Create client
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # First, load the model explicitly
    client.load_model("resnet18")
    
    # Check if model loaded successfully
    if not client.is_model_ready("resnet18"):
        print("Model failed to load!")
        return
    
    print("Model loaded successfully")
    
    # Prepare data
    testloader = prepare_data()
    
    # Run inference
    times = []
    total_batches = 10
    batch_size = 32
    
    for i, (images, _) in enumerate(testloader):
        if i == total_batches:
            break
        
        # Convert to numpy and ensure correct shape
        input_data = images.numpy()
        
        # Create the input object for the model
        inputs = []
        inputs.append(httpclient.InferInput("input.1", input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Create the output objects to hold the results
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("output"))
        
        # Measure inference time
        start_time = time.perf_counter()
        results = client.infer("resnet18", inputs, outputs=outputs)
        end_time = time.perf_counter()
        
        # Calculate latency and throughput
        latency = (end_time - start_time) * 1000  # Convert to ms
        times.append(latency)
        
        # Get the result
        output = results.as_numpy("output")
        
        print(f"Batch {i+1} - Latency: {latency:.2f} ms")
    
    avg_latency = np.mean(times)
    throughput = batch_size / (np.mean(times) / 1000)  # Images per second
    
    print(f"\nAvg Latency per Batch: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    
    # Unload the model when done
    client.unload_model("resnet18")

if __name__ == "__main__":
    run_inference()
