from tinyimagenet import TinyImageNet
from pathlib import Path
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
splits = ["val","test","train"]
for split in splits:

    dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"),split=split,imagenet_idx=True)
    print(dataset.idx_to_class)
    print(dataset.class_to_idx)
    n = len(dataset)
    print(f"TinyImageNet, split {split}, has  {n} samples. Loading all samples with imagenet_idx:")

    for i in tqdm(range(0,n)):
        image,klass = dataset[i]