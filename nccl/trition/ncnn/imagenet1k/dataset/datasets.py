import os
import re
from typing import Tuple, Callable, List
#import torch
#from torch.utils.data import Dataset
#from torchvision import transforms as T
#from torchvision.io import decode_image
#from PIL import Image
from imagenet_labels import imagenet1K_codes_to_labels, imagenet1K_labels_to_names, imagenet1K_labels_to_codes, imagebet1K_val_groundtruth_labels


# > Attention : utilisation d'un répertoire de données en dehors de l'environnement du notebook !

# Source : https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

DATASET_0 = {
    # Source: https://en.wikipedia.org/wiki/ImageNet#ImageNet-1K
    "name": "imagenet-1K",
    "path": "",
    "size": 1_281_167,
    # Source: https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
    "means": imagenet_mean,
    "stds": imagenet_std
}

DATASET_1 = {  # 50K images for evaluation
    "name": "imagenet_val_images (50K images)",
    "path": "/Users/me/Documents/Work/Dev/_data/imagenet_val_images",
    "mounted_path": "/Volumes/me/Documents/Work/Dev/_data/imagenet_val_images",
    "size": 50000,
    "means": imagenet_mean,
    "stds": imagenet_std
}

DATASET_2 = { # 1K images - 1K classes catalog label code <-> label name
    "name": "imagenet-sample-images (1K images)",
    "path": "/Users/me/Documents/Work/Dev/_data/imagenet-sample-images-master",
    "mounted_path": "/Volumes/me/Documents/Work/Dev/_data/imagenet-sample-images-master",
    "size": 1000,
    "means": imagenet_mean,
    "stds": imagenet_std
}

DATASET_3 = { # 1K images - 1K classes catalog label code <-> label name
    "name": "ILSVRC2012_img_val",
    "path": "/Users/me/Documents/Work/Dev/_data/ILSVRC2012_img_val",
    "mounted_path": "/Volumes/me/Documents/Work/Dev/_data/ILSVRC2012_img_val",
    "size": 50000,
    "means": imagenet_mean,
    "stds": imagenet_std
}
def get_label_code_from_filename(filename: str, dataset_name: str) -> str:
    """
    Get the label code from the filename if presents

    Args:
        filename (str): filename
        datapath (str): path to the dataset
    Returns:
        str: label code
    """
    if dataset_name == DATASET_0["name"]:
        return None
    elif dataset_name == DATASET_1["name"]:
        m = re.match(r"^[^_]+_[^_]+_[^_]+_([^.]+)\.JPEG$", filename)
    elif dataset_name == DATASET_2["name"]:
        m = re.match(r"^([^_]+)_([^.]+)\.JPEG$", filename)
    else:
        return None

    return m.group(1)
def get_label_data_from_filename(filename: str, dataset_name: str) -> Tuple[str, int, str]:
    """
    Return name from filename, or (name, label) regarding a name to label dictionnary

    Args:
        filename (str): filename
        datapath (str): path to the dataset

    Returns:
        str|Tuple[str, int, str]: (code, label, name)
    """
    if dataset_name == DATASET_3["name"] or dataset_name == DATASET_1["name"]:
        file_number = get_file_number_from_filename(filename, dataset_name)
        label_idx = imagebet1K_val_groundtruth_labels.get(file_number, None)
        label_code = imagenet1K_labels_to_codes.get(label_idx, None)
        label_name = imagenet1K_labels_to_names.get(label_idx, None)
    else:
        label_code = get_label_code_from_filename(filename, dataset_name)
        label_idx = imagenet1K_codes_to_labels.get(label_code, None)
        label_name = imagenet1K_labels_to_names.get(label_idx, None)

    return label_code, label_idx, label_name
if __name__ == "__main__":
    # TESTS

    # -- Test of get_label_code_from_filename --
    tests = [
        {
            "dataset_name": DATASET_1["name"],
            "filename": "ILSVRC2012_val_00000026_n04380533.JPEG",
            "expected": "n04380533"
        },
        {
            "dataset_name": DATASET_1["name"],
            "filename": "ILSVRC2012_val_00000152_n03710193.JPEG",
            "expected": "n03710193"
        },
        {
            "dataset_name": DATASET_2["name"],
            "filename": "n02089078_black-and-tan_coonhound.JPEG",
            "expected": "n02089078"
        },
        {
            "dataset_name": DATASET_2["name"],
            "filename": "n02395406_hog.JPEG",
            "expected": "n02395406"
        },
    ]

    for test in tests:
        dataset_name = test["dataset_name"]
        filename = test["filename"]
        expected = test["expected"]
        result = get_label_code_from_filename(filename, dataset_name)
        assert result == expected, f"Expected {expected}, but got {result} for {filename} in {dataset_name}"
        print(f"Expected {expected}, but got {result} for {filename} in {dataset_name}")
    print("-- Tests of get_label_code_from_filename() passed")
    filename = "n02395406_hog.JPEG"
    code, label, name = get_label_data_from_filename(filename, DATASET_2["name"])
    print(f" code, label, name ={code, label, name}")
