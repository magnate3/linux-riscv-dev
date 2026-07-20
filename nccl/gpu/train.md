
[Part 1. Single GPU Example (this article) — Training ResNet34 on CIFAR10](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8)   
[Part2. Data Parallel — Training code & issue between DP and NVLink](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8)   
[Part3. Distributed Data Parallel — Training code & Analysis](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8)   
[Part4. Torchrun — Fault tolerance](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8)   

# 数据集

[dataset of the url  with the ssl](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)     
[dataset of the url without the ssl](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)    

```
root@ubuntu:/pytorch# ls cifar-10-batches-py/
batches.meta  data_batch_1  data_batch_2  data_batch_3  data_batch_4  data_batch_5  readme.html  test_batch
root@ubuntu:/pytorch# mv cifar-10-batches-py/* ResNet18_Cifar10_95.46/dataset/
root@ubuntu:/pytorch# 
```

将cifar-10-python.tar.gz 拷贝到 ResNet18_Cifar10_95.46/dataset/

```
cp cifar-10-python.tar.gz  ResNet18_Cifar10_95.46/dataset/
```

#  ResNet18_Cifar10

```
/pytorch/ResNet18_Cifar10_95.46# python3 train.py 
Using downloaded and verified file: dataset/cifar-10-python.tar.gz
Extracting dataset/cifar-10-python.tar.gz to dataset
Files already downloaded and verified
Files already downloaded and verified
```


+ mkdir checkpoint解决如下bug
```
/pytorch/ResNet18_Cifar10_95.46# python3 train.py 
Using downloaded and verified file: dataset/cifar-10-python.tar.gz
Extracting dataset/cifar-10-python.tar.gz to dataset
Files already downloaded and verified
Files already downloaded and verified
  0%|                                                                                                                                                                    | 0/250 [00:00<?, ?it/s]Accuracy: 32.96083860759494 %
Epoch: 1        Training Loss: 2.026326         Validation Loss: 1.780202
Validation loss decreased (inf --> 1.780202).  Saving model ...
  0%|                                                                                                                                                                    | 0/250 [00:19<?, ?it/s]
Traceback (most recent call last):
  File "train.py", line 109, in <module>
    torch.save(model.state_dict(), 'checkpoint/resnet18_cifar10.pt')
  File "/usr/local/lib/python3.8/dist-packages/torch/serialization.py", line 422, in save
    with _open_zipfile_writer(f) as opened_zipfile:
  File "/usr/local/lib/python3.8/dist-packages/torch/serialization.py", line 309, in _open_zipfile_writer
    return container(name_or_buffer)
  File "/usr/local/lib/python3.8/dist-packages/torch/serialization.py", line 287, in __init__
    super(_open_zipfile_writer_file, self).__init__(torch._C.PyTorchFileWriter(str(name)))
RuntimeError: Parent directory checkpoint does not exist.
```


#    lenet

```
python3 main.py --net-type 'lenet' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10'  --dataroot ../ResNet18_Cifar10_95.46/dataset/
```

```
CIFAR-10:

python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --nfilters 256 --batch-size 20 --learning-rate 1e-4 --first_filter_size 0 --filter_size 0 --nmasks 1 --level 0.1 --optim-method Adam --nepochs 450

MNIST:

python main.py --net-type 'lenet' --dataset-test 'MNIST' --dataset-train 'MNIST'


transfer CIFAR-10 on CIFAR-100:
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR100' --dataset-train 'CIFAR100' --transfer True --resume '/home/eli/Workspace/pnn-results/CIFAR-10/model_epoch_268_acc_86.40.pth' --nfilters 256 --batch-size 20 --learning-rate 1e-4 --first_filter_size 0 --filter_size 0 --nmasks 1 --level 0.1 --optim-method Adam --nepochs 25


transfer EMNIST on MNIST:
python main.py --net-type 'lenet' --dataset-test 'MNIST' --dataset-train 'MNIST' --transfer True --resume '/home/eli/Workspace/pnn-results/EMINST/model_epoch_30_acc_85.39.pth' --nfilters 256 --batch-size 20 --learning-rate 1e-4 --first_filter_size 0 --filter_size 0 --nmasks 1 --level 0.1 --optim-method Adam --nepochs 25

transfer CIFAR-100 on CIFAR-10:
python main.py --net-type 'perturb_resnet18' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10' --transfer True --resume '/home/eli/Workspace/pnn-results/CIFAR-100/model_epoch_58_acc_55.00.pth' --nfilters 256 --batch-size 20 --learning-rate 1e-4 --first_filter_size 0 --filter_size 0 --nmasks 1 --level 0.1 --optim-method Adam --nepochs 25
```


```
Best Accuracy: 56.49  (epoch 125)
ls results/2025-12-09_07-12-35/Save/model_epoch_125_acc_56.49.pth
```


```
 python3 test1.py --net-type 'lenet' --dataset-test 'CIFAR10' --dataset-train 'CIFAR10'  --dataroot ../ResNet18_Cifar10_95.46/dataset/   --resume results/2025-12-09_07-12-35/Save/model_epoch_125_acc_56.49.pth


****** Creating lenet model ******




Loading model from saved checkpoint at results/2025-12-09_07-12-35/Save/model_epoch_125_acc_56.49.pth




****** Preparing CIFAR10 dataset *******


Files already downloaded and verified
Files already downloaded and verified
```