import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

# 1. Read the image using OpenCV (BGR format, NumPy array, HWC dimensions)
# Replace 'path/to/your/image.jpg' with your actual image path
cv_image = cv2.imread('car.jpg') 

if cv_image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

# 2. Convert the color space from BGR to RGB
# This is crucial because PyTorch models typically expect RGB input
rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

# 3. Define the transform to convert the image to a PyTorch tensor
# ToTensor() scales pixel values from [0, 255] to [0.0, 1.0] and changes
# the layout from HWC to CHW.
transform = transforms.Compose([
    transforms.ToTensor()
])

# 4. Apply the transform to get the PyTorch tensor
tensor_image = transform(rgb_image)

print(f"Original image shape (HWC): {rgb_image.shape}")
print(f"Tensor shape (CHW): {tensor_image.shape}")
print(f"Tensor dtype: {tensor_image.dtype}")
print(f"Pixel value range: min={tensor_image.min()}, max={tensor_image.max()}")
