from torchvision import datasets
import json
DATA_ROOT_DIR = "torchvision/tinyimagenet/tiny-imagenet-200/"
VAL_DATA_DIR = DATA_ROOT_DIR + "val"
dataset = datasets.ImageFolder(VAL_DATA_DIR)

def get_tiny_label_dict(root=DATA_ROOT_DIR):
    # 读取200个类别ID
    with open(f"{root}/wnids.txt") as f:
        wnids = [line.strip() for line in f]

    # 读取类别名称
    wnid_to_name = {}
    with open(f"{root}/words.txt") as f:
        for line in f:
            wnid, name = line.strip().split("\t", 1)
            wnid_to_name[wnid] = name

    # 数字标签 → 类别名
    idx_to_name = {i: wnid_to_name[wn] for i, wn in enumerate(wnids)}
    return idx_to_name

def get_tiny_label(root=DATA_ROOT_DIR,index=0):
    # 读取200个类别ID
    with open(f"{root}/wnids.txt") as f:
        wnids = [line.strip() for line in f]
    return wnids[index]
# 类别文件夹名（已排序）
#print(dataset.classes)

# 字典：文件夹名 → index
#print(dataset.class_to_idx)

# index → 文件夹名（常用）
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
#print(idx_to_class)
label_dict = get_tiny_label_dict()
print(get_tiny_label(DATA_ROOT_DIR,84))
print(idx_to_class[84])
filename = 'label_to_tiny_class.json'
with open(filename, 'w') as f:
     json.dump(idx_to_class, f, indent=4)
     #json.dump(label_dict, f, indent=4)
