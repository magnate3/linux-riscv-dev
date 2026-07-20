
import csv

from pathlib import Path
import shutil
from typing import List, Tuple
from urllib.error import URLError

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg,download_and_extract_archive
import logging

import imagenet1k
#from tinyimagenet import imagenet1k

mirrors = [
    "http://cs231n.stanford.edu/",
]

resources = [
    ("tiny-imagenet-200.zip", "90528d7ca1a48142e341f4ef8d21d0de"),
]

def check_exists(root:Path,resources:List[Tuple[str,str]]) -> bool:
        return all(check_integrity(root/file,md5) for file, md5 in resources)

def download_resources(root:Path,mirrors:List[str],resources:List[Tuple[str,str]]):
        
        
        root.mkdir(exist_ok=True,parents=True)
        # download files
        for filename, md5 in resources:
            for mirror in mirrors:
                url = f"{mirror}{filename}"
                try:
                    logging.info(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=root, filename=filename, md5=md5)
                except URLError as error:
                    logging.warn(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    logging.info("")
                break
            else:
                raise RuntimeError(f"Error downloading {filename} from all mirrors.")
        check_exists(root,resources)
import csv

def preprocess_val(root:Path):
    root = root /"val"
    annotations_filepath = root/"val_annotations.txt"
    images_root = root/"images"

    if not images_root.exists() and annotations_filepath.exists() :
        #if the `images`` folder was deleted but the annotations file exists
        # then consider the preprocessing as finished
        return
    logging.info(f"Preprocessing validation set in {images_root}")
    delimiter = "\t"
    with open(annotations_filepath) as f:
        reader = csv.reader(f,delimiter=delimiter)
        files = list(reader)
        class_metadata = {}
        for row in files:
            
            name,klass = row[0],row[1]
            old_filepath = images_root/name
            class_path = root / klass /"images"
            new_filepath = class_path/name
            # move to folder with same names as class to mimic training data
            missing = False
            if old_filepath.exists():
                if not new_filepath.exists():
                    class_path.mkdir(exist_ok=True,parents=True)
                    shutil.move(old_filepath,new_filepath)
                else:
                    raise ValueError(f"File {name} exists is both input and output folders")
            else:
                if not new_filepath.exists():
                    missing = True
                    logging.warning(f"Warning: file {name} ignored: it exists in {annotations_filepath} but is missing from both {class_path} and {images_root} folders.")
                #else: if the filepath exists, do nothing, to recover from a previous
                # interrupted preprocessing step

            if not missing:
                # Append to dict to generate new txt with annotations
                if not klass in class_metadata:
                    class_metadata[klass]=[]
                # remove class id from row
                row = [row[0]]+row[2:]
                class_metadata[klass].append(row)
        
        for klass,files in class_metadata.items():
            class_annotation_path = root/klass/ f"{klass}_boxes.txt"
            with open(class_annotation_path,"w") as f:
                writer = csv.writer(f,delimiter=delimiter)
                for row in files:
                    writer.writerow(row)
        logging.warning(f"About to delete {images_root}")
        images_root.rmdir()



    

class TinyImageNet(ImageFolder):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def target_transform_imagenet_idx(self,tinyimagenet_idx):
        print(tinyimagenet_idx)
        klass = self.idx_to_class[tinyimagenet_idx]
        imagenet_idx = imagenet1k.class_to_idx[klass]
        return imagenet_idx

    def __init__(self, root: Path, split: str = "train",transform=None, target_transform=None,imagenet_idx=False) -> None:
        if isinstance(root,str):
            root = Path(root)
        self.class_to_imagenet_idx = imagenet1k.class_to_idx 
        assert split in ["train","val","test"]
        root = root.expanduser()
        images_root = root/"tiny-imagenet-200/"
        if not images_root.exists():
            download_resources(root,mirrors,resources)        
        preprocess_val(images_root)
        if target_transform is None:
            target_transform = lambda x: x
        if imagenet_idx:
            new_target_transform = lambda x: target_transform(self.target_transform_imagenet_idx(x))
        else:
            new_target_transform=target_transform
        super().__init__(images_root/split,transform=transform,target_transform=new_target_transform)
        self.idx_to_words,self.idx_to_class = self.load_words_classes(images_root)
    
    def load_words_classes(self,root:Path):
        tiny_classes = open(root/"wnids.txt", 'r').read().splitlines()
        with open(root/"words.txt") as f:
            words = {}
            classes = {}
            reader = csv.reader(f,delimiter="\t")
            id = 0
            for (klass,keywords) in reader:
                if klass in tiny_classes:
                    words[id] = keywords.split(",")
                    classes[id] = klass
                    id=id+1
        return words,classes


if __name__ == '__main__':
    from torchvision import transforms

    logging.basicConfig(level=logging.INFO)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(TinyImageNet.mean,TinyImageNet.std)]
        )

    split ="val"
    dataset = TinyImageNet(Path("./torchvision/tinyimagenet/"),split=split,transform=transform)
    n = len(dataset)
    print(f"TinyImageNet, split {split}, has  {n} samples.")
    print("Showing some samples")
    for i in range(0,n,n//5):
        image,klass = dataset[i]
        print(f"Sample of class {klass:3d}, image shape {image.shape}, words {dataset.idx_to_words[klass]}")


