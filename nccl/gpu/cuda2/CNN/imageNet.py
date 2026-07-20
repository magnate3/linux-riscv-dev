import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from imagenet_seq.data import ImagenetLoader
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
imagenet_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

# train_dataset = datasets.ImageFolder(traindir, imagenet_transforms)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

train_loader = ImagenetLoader(args.data, 'train', imagenet_transforms,
        batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
