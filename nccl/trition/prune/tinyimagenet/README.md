


```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/pytorch:24.05-py3 bash
PYTHONPATH=$PYTHONPATH:/pytorch/prune/tinyimagenet
```


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

```
ls tinyimagenet/torchvision/tinyimagenet/tiny-imagenet-200/
test  train  val  wnids.txt  words.txt
```