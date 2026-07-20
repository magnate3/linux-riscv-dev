import numpy as np
from PIL import Image
from torchvision import transforms

# preprocessing function
def rn50_preprocess(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(img).numpy()

transformed_img = rn50_preprocess()
print(transformed_img.shape, transformed_img.dtype)
flattened_img = transformed_img.flatten()
print(flattened_img.shape, flattened_img.dtype)
np.savetxt("img1.txt", flattened_img, delimiter=" ", fmt="%.6f", newline=" ")
print(flattened_img[:5])
