# 数据集

[the url to one with the ssl](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 
[the url to one with out the ssl](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 

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