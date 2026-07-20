from tinyimagenet import TinyImageNet
from pathlib import Path
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
splits = ["val","test","train"]
for split in splits:

    dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"),split=split,imagenet_idx=False)
    n = len(dataset)   
    print(f"TinyImageNet, split {split}, has  {n} samples. Loading all samples:")
    # n_samples = 1000
    # print(f"Showing info of {n_samples} samples...")

    for i in tqdm(range(n)):
        image,klass = dataset[i]
        id = dataset.idx_to_class[klass]
        imagenet_klass = dataset.class_to_imagenet_idx[id]
