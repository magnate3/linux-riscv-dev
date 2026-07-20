# torchvision-tinyimagenet
Dataset class for PyTorch and the TinyImageNet dataset.

# Installation

``` pip install tinyimagenet ```

# How to use
````
from tinyimagenet import TinyImageNet
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

split ="val"
dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"),split=split)
n = len(dataset)
print(f"TinyImageNet, split {split}, has  {n} samples.")
n_samples = 5
print(f"Showing info of {n_samples} samples...")
for i in range(0,n,n//n_samples):
    image,klass = dataset[i]
    print(f"Sample of class {klass:3d}, image {image}, words {dataset.idx_to_words[klass]}")
````

You can also check the [quickstart notebook](https://colab.research.google.com/drive/1FCDsDJg86mCjyeAWOxDW9iF49goWCx4j?usp=sharing)
