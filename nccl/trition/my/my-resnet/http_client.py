import torch
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
#import torch.nn as nn
#import torch.optim as optim
#import cv2
import time
import numpy as np
np.set_printoptions(suppress=True)
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import os
from utils.readData import read_dataset
classes = ('plane', 'car', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#import onnx
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
IMAGE_RESOLUTION = (224, 224)
def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)  # Remove batch dimension if it was added

    return padded_images
def preprocess_image(image,image_resolution):
    # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
    # Handle both [B, C, H, W] and [B, H, W, C] formats
    is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1
    train = True
    wrist = True
    if is_channels_first:
        # Convert [B, C, H, W] to [B, H, W, C] for processing
        image = image.permute(0, 2, 3, 1)

    if image.shape[1:3] != image_resolution:
        # logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
        image = resize_with_pad_torch(image, *image_resolution)

    if train:
        # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
        image = image / 2.0 + 0.5

        # Apply PyTorch-based augmentations
        if not wrist:
            # Geometric augmentations for non-wrist cameras
            height, width = image.shape[1:3]

            # Random crop and resize
            crop_height = int(height * 0.95)
            crop_width = int(width * 0.95)

            # Random crop
            max_h = height - crop_height
            max_w = width - crop_width
            if max_h > 0 and max_w > 0:
                # Use tensor operations instead of .item() for torch.compile compatibility
                start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                # print(f"image: {image.shape}, start h: {start_h}, start w: {start_w}")
                image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

            # Resize back to original size
            image = torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Random rotation (small angles)
            # Use tensor operations instead of .item() for torch.compile compatibility
            angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
            if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                # Convert to radians
                angle_rad = angle * torch.pi / 180.0

                # Create rotation matrix
                cos_a = torch.cos(angle_rad)
                sin_a = torch.sin(angle_rad)

                # Apply rotation using grid_sample
                grid_x = torch.linspace(-1, 1, width, device=image.device)
                grid_y = torch.linspace(-1, 1, height, device=image.device)

                # Create meshgrid
                grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                # Expand to batch dimension
                grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                # Apply rotation transformation
                grid_x_rot = grid_x * cos_a - grid_y * sin_a
                grid_y_rot = grid_x * sin_a + grid_y * cos_a

                # Stack and reshape for grid_sample
                grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                image = torch.nn.functional.grid_sample(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

        # Color augmentations for all cameras
        # Random brightness
        # Use tensor operations instead of .item() for torch.compile compatibility
        brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
        image = image * brightness_factor

        # Random contrast
        # Use tensor operations instead of .item() for torch.compile compatibility
        contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        # Random saturation (convert to HSV, modify S, convert back)
        # For simplicity, we'll just apply a random scaling to the color channels
        # Use tensor operations instead of .item() for torch.compile compatibility
        saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
        gray = image.mean(dim=-1, keepdim=True)
        image = gray + (image - gray) * saturation_factor

        # Clamp values to [0, 1]
        image = torch.clamp(image, 0, 1)

        # Back to [-1, 1]
        image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    return image
def reshape_transform():
    def fn(data):
        batch_size = data.size(0)
        return data.reshape(batch_size, -1)
    return fn
def visualize_predictions(imgs, classes,targets,predicts,num_images= 8):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    images_shown = 0
    for i in range(0,imgs.size(0)):
        img = imgs[i].cpu()
        true_label = classes[targets[i]]
        pred_label = classes[predicts[i]]
        color = 'green' if targets[i] == predicts[i] else 'red'
        ax = axes[images_shown//4, images_shown%4]
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        images_shown += 1
        if images_shown >= num_images:
            break
    ax.axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig("test.png")
#ransfos = transforms.Compose([torchvision.transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.4749), (0.2382)), reshape_transform()])
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
batch_size = 128
#train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='../dataset')
transfos = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.4749), (0.2382))])
test = datasets.CIFAR10("../../dataset", train = False, download = True, transform = transfos)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)
iterator = iter(test_loader)
images, batch_targets = next(iterator)
#triton_client = grpcclient.InferenceServerClient(url='127.0.0.1:8001', verbose=False)  # 固定不用改
#with grpcclient.InferenceServerClient(url="localhost:8001") as triton_client:
with httpclient.InferenceServerClient(url="localhost:8000") as triton_client:
    outputs = [httpclient.InferRequestedOutput('output'),]
    
    print(images.shape)
    #images= preprocess_image(images,IMAGE_RESOLUTION)
    #images= resize_with_pad_torch(images,256,256)
    #print(images.shape)
    img = images.cpu()
    #img = images.cpu().numpy()
    
    print(img.shape)
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    print(img.shape)
    #img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    #img = torch.clamp(img, 0, 1)

    img = img.numpy()
    inputs = [httpclient.InferInput('images', img.shape, 'FP32')]
    inputs[0].set_data_from_numpy(img)
    start_time = time.time()
    result = triton_client.infer(
            'model_test',
            inputs,
            request_id=str('random'),
            model_version='',
            outputs=outputs)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print(outputs)
    triton_out = outputs = result.as_numpy('output').round(3)
    predicted = outputs.argmax(-1)
    print("max of predict {} ".format(predicted))
    print([classes[predicted[i]] for i in range(len(predicted))] , '\n', outputs)
    visualize_predictions(images,classes,batch_targets,predicted)
