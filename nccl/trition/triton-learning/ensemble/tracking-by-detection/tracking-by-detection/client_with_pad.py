import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import tritonclient.grpc as grpcclient
from collections import defaultdict
import time
import argparse
import io
from PIL import Image
np.bool = np.bool_


        


#WIDTH = 1024
#HIGHT = int(4915200/WIDTH)
#
#INPUT_SHAPE = [1, 3,HIGHT , WIDTH]

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
def preprocess_image(raw,shape):
    """
    Placeholder for image preprocessing.
    In real application, this would handle actual image data.
    """
    #return np.random.rand(*shape).astype("uint8")
    #return  np.resize(raw, shape)
    if raw is None:
        return np.random.rand(*shape).astype(np.float32)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    #image = np.array(image) / 255.0
    #image = image.resize((1024,1024))
    image = image.resize((shape[2], shape[3]))
    image_array = np.array(image)

    #if len(image_array.shape) == 2:
    #    image_array = np.stack([image_array] * 3, axis=-1)

    #image_array = image_array.transpose(2, 0, 1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype("uint8")
def pic_to_tensor_u8(path):
     # 1. Read the image using OpenCV (BGR format, NumPy array, HWC dimensions)
     # Replace 'path/to/your/image.jpg' with your actual image path
     cv_image = cv2.imread(path) 
     
     if cv_image is None:
         raise FileNotFoundError("Image not found. Check the file path.")
     
     # 2. Convert the color space from BGR to RGB
     # This is crucial because PyTorch models typically expect RGB input
     rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
     tensor_uint8 = torch.from_numpy(rgb_image)
     # Optional: If you need the channel dimension first (C, H, W) for transforms/models
     # OpenCV/NumPy uses (H, W, C)
     #tensor_uint8 = tensor_uint8.permute(2, 0, 1)
     print(f"Tensor shape: {tensor_uint8.shape}")
     print(f"Tensor dtype: {tensor_uint8.dtype}")
     # The output will confirm dtype=torch.uint8
     return tensor_uint8
def  pic_to_tensor_f32(path):
     # 1. Read the image using OpenCV (BGR format, NumPy array, HWC dimensions)
     # Replace 'path/to/your/image.jpg' with your actual image path
     cv_image = cv2.imread(path) 
     
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
     return tensor_image
def pic_to_tensor():
    # Read BGR image
    cv_image_bgr = cv2.imread('path/to/your/image.jpg')
    
    # Convert to RGB (HWC format)
    rgb_image = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert NumPy array to Torch tensor
    # The tensor will have dtype=torch.uint8 initially
    tensor_uint8 = torch.from_numpy(rgb_image)
    
    # Convert to float and scale to [0.0, 1.0], then rearrange dimensions from HWC to CHW
    tensor_float = tensor_uint8.float() / 255.0
    tensor_final = tensor_float.permute(2, 0, 1) 
    # or use .permute(2, 0, 1).contiguous() if further operations require contiguous memory
    
    print(f"Final tensor shape (CHW): {tensor_final.shape}")
def main():
    client = grpcclient.InferenceServerClient(url='localhost:8001')

    #image_data = pic_to_tensor_u8('track.jpg')
    #image_data = pic_to_tensor_u8('car3.jpg')
    image_data = pic_to_tensor_f32('car3.jpg')
    #image_data = pic_to_tensor_u8('car.jpg')
    #image_data = read_image('car.jpg')
    #image_data = np.fromfile("mug.jpg", dtype="uint8")
    #image_data = np.fromfile("track6.png", dtype="uint8")
    #image_data = np.fromfile("track5.png", dtype="uint8")
    #image_data = np.fromfile("track4.png", dtype="uint8")
    #image_data = np.fromfile("track3.png", dtype="uint8")
    #image_data = np.fromfile("track2.png", dtype="uint8")
    #im_cv = cv2.imread("frame.png")
    #im_cv = cv2.imread("track.png")
    #img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    #image_data = np.asarray(img_rgb)
    print(image_data.shape)
    #WIDTH = 4096
    #WIDTH = 2048
    WIDTH = 1024
    #WIDTH =  916
    #HIGHT = int(4915200/WIDTH)
    #HIGHT = int(image_data.size/WIDTH)

    #INPUT_SHAPE = [1, 3,640, 640]
    #INPUT_SHAPE = [1, 3,HIGHT , WIDTH]
    #image_data= preprocess_image(image_data,INPUT_SHAPE)
    print(image_data.shape)
    image_data = resize_with_pad_torch(image_data,1024,1024)
    image_data= image_data.to(torch.uint8)
    print(image_data.shape)
    image_data = image_data.permute(0,3, 2, 1)
    print(image_data.shape)
    image_data = image_data.numpy()
    #image_data = np.expand_dims(image_data, axis=0)
    input_tensor = grpcclient.InferInput(
        'input_image', image_data.shape, 'UINT8')
    input_tensor.set_data_from_numpy(image_data)

    start_time = time.time()

    results = client.infer(
        model_name='tracking_by_detection',
        inputs=[input_tensor],
        sequence_id=id(123456),
        sequence_start=0,
        sequence_end=1)
    fps = int(1.0 / (time.time() - start_time))

    detections = results.as_numpy('detections')


if __name__ == '__main__':
    main()
