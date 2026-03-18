


```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/pytorch:24.05-py3 bash
PYTHONPATH=$PYTHONPATH:/pytorch/prune/tinyimagenet
```

# dataset

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

Stanford prepared the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset for their [CS231n](http://cs231n.stanford.edu/) course. The dataset spans 200 image classes with 500 training examples per class. The dataset also has 50 validation and 50 test examples per class.

The images are down-sampled to 64x64 pixels vs. 256x256 for the original ImageNet. The full ImageNet dataset also has 1000 classes.    

```
u/tiny-imagenet-200.zip
Downloading https://cs231n.stanford.edu/tiny-imagenet-200.zip to torchvision/tinyimagenet/tiny-imagenet-200.zip
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 248100043/248100043 [08:15<00:00, 500671.17it/s]
Extracting torchvision/tinyimagenet/tiny-imagenet-200.zip to torchvision/tinyimagenet
INFO:root:
INFO:root:Preprocessing validation set in torchvision/tinyimagenet/tiny-imagenet-200/val/images
WARNING:root:About to delete torchvision/tinyimagenet/tiny-imagenet-200/val/images
TinyImageNet, split val, has  10000 samples.
Showing some samples
Sample of class   0, image shape torch.Size([3, 64, 64]), words ['goldfish', ' Carassius auratus']
Sample of class  40, image shape torch.Size([3, 64, 64]), words ['walking stick', ' walkingstick', ' stick insect']
Sample of class  80, image shape torch.Size([3, 64, 64]), words ['bucket', ' pail']
Sample of class 120, image shape torch.Size([3, 64, 64]), words ['miniskirt', ' mini']
Sample of class 160, image shape torch.Size([3, 64, 64]), words ['teapot']
```

+ 数据集   

`train/n01443537/images/`   `n01443537_boxes.txt `

```
ls tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/
test  train  val  wnids.txt  words.txt
```


```
ls   /pytorch/prune/tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/train/n01443537/
images  n01443537_boxes.txt
```

```
ls   /pytorch/prune/tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/train/n01443537/images/n01443537_1.JPEG 
/pytorch/prune/tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/train/n01443537/images/n01443537_1.JPEG
```


#  ImageFolder
```Text
在 PyTorch  ImageFolder  里，label index（标签编号） 是这样确定的：
 
1. 规则（非常重要）
 
1. 自动扫描  root  下的所有子文件夹
​
2. 把文件夹名称当作类别名
​
3. 对这些文件夹名进行 字母排序（sorted）
​
4. 按排序后的顺序，从 0 开始依次分配 index：0,1,2,...,199
 
也就是说：
 
index = sorted(类别文件夹列表) 中的位置
 
```
 
+  如何查看 index ↔ 类别对应关系   


```
from torchvision import datasets

dataset = datasets.ImageFolder("tiny-imagenet-200/val")

# 类别文件夹名（已排序）
print(dataset.classes)

# 字典：文件夹名 → index
print(dataset.class_to_idx)

# index → 文件夹名（常用）
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
print(idx_to_class)
```

##  TinyImageNet

原始TinyImageNet
```
val/
  images/
    val_01.JPEG
    val_02.JPEG
    ...
  val_annotations.txt
```
原始TinyImageNet需要用脚本整理

```
import os
import shutil

def organize_tiny_val(root="tiny-imagenet-200"):
    val_dir = os.path.join(root, "val")
    img_dir = os.path.join(val_dir, "images")
    anno_file = os.path.join(val_dir, "val_annotations.txt")

    with open(anno_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        img_name = parts[0]
        wnid = parts[1]

        cls_dir = os.path.join(val_dir, wnid)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(img_dir, img_name)
        dst = os.path.join(cls_dir, img_name)
        shutil.move(src, dst)

    os.rmdir(img_dir)
    print("val 已按类别整理完成")

organize_tiny_val()
```