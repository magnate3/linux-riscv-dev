import torch
import random
from dataloader import Dataloader
import utils
import os
from datetime import datetime
import argparse
import math
import numpy as np
from torch import nn
import models
import torch.optim as optim

import matplotlib.pyplot as plt
NORMALIZATION_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZATION_STD = [0.2470, 0.2435, 0.2616]
classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
result_path = "results/"
result_path = os.path.join(result_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='Your project title goes here')

# ======================== Data Setings ============================================
parser.add_argument('--dataset-test', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataset-train', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataroot', type=str, default='./data', metavar='', help='path to the data')
parser.add_argument('--save', type=str, default=result_path +'Save', metavar='', help='save the trained models here')
parser.add_argument('--logs', type=str, default=result_path +'Logs', metavar='', help='save the training log files here')
parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')

# ======================== Network Model Setings ===================================

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--use_act', dest='use_act', action='store_true')
feature_parser.add_argument('--no-use_act', dest='use_act', action='store_false')
parser.set_defaults(use_act=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--unique_masks', dest='unique_masks', action='store_true')
feature_parser.add_argument('--no-unique_masks', dest='unique_masks', action='store_false')
parser.set_defaults(unique_masks=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug', dest='debug', action='store_true')
feature_parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_masks', dest='train_masks', action='store_true')
feature_parser.add_argument('--no-train_masks', dest='train_masks', action='store_false')
parser.set_defaults(train_masks=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--mix_maps', dest='mix_maps', action='store_true')
feature_parser.add_argument('--no-mix_maps', dest='mix_maps', action='store_false')
parser.set_defaults(mix_maps=False)

parser.add_argument('--filter_size', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--first_filter_size', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--nfilters', type=int, default=64, metavar='', help='number of filters in each layer')
parser.add_argument('--nmasks', type=int, default=1, metavar='', help='number of noise masks per input channel (fan out)')
parser.add_argument('--level', type=float, default=0.5, metavar='', help='noise level for uniform noise')
parser.add_argument('--scale_noise', type=float, default=1.0, metavar='', help='noise level for uniform noise')
parser.add_argument('--noise_type', type=str, default='uniform', metavar='', help='type of noise')
parser.add_argument('--dropout', type=float, default=0.5, metavar='', help='dropout parameter')
parser.add_argument('--net-type', type=str, default='resnet18', metavar='', help='type of network')
parser.add_argument('--act', type=str, default='relu', metavar='', help='activation function (for both perturb and conv layers)')
parser.add_argument('--pool_type', type=str, default='max', metavar='', help='pooling function (max or avg)')

# ======================== Training Settings =======================================
parser.add_argument('--batch-size', type=int, default=64, metavar='', help='batch size for training')
parser.add_argument('--nepochs', type=int, default=150, metavar='', help='number of epochs to train')
parser.add_argument('--nthreads', type=int, default=4, metavar='', help='number of threads for data loading')
parser.add_argument('--manual-seed', type=int, default=1, metavar='', help='manual seed for randomness')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--optim-method', type=str, default='SGD', metavar='', help='the optimization routine ')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='', help='learning rate')
parser.add_argument('--learning-rate-decay', type=float, default=None, metavar='', help='learning rate decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='', help='weight decay')
parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='', help='Beta 1 parameter for Adam')
parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='', help='Beta 2 parameter for Adam')

args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.saveargs(args)

class Model:
    def __init__(self, args):
        self.cuda = torch.cuda.is_available()
        self.lr = args.learning_rate
        self.dataset_train_name = args.dataset_train
        self.nfilters = args.nfilters
        self.batch_size = args.batch_size
        self.level = args.level
        self.net_type = args.net_type
        self.nmasks = args.nmasks
        self.unique_masks = args.unique_masks
        self.filter_size = args.filter_size
        self.first_filter_size = args.first_filter_size
        self.scale_noise = args.scale_noise
        self.noise_type = args.noise_type
        self.act = args.act
        self.use_act = args.use_act
        self.dropout = args.dropout
        self.train_masks = args.train_masks
        self.debug = args.debug
        self.pool_type = args.pool_type
        self.mix_maps = args.mix_maps

        if self.dataset_train_name.startswith("CIFAR"):
            self.input_size = 32
            self.nclasses = 10
            if self.filter_size < 7:
                self.avgpool = 4
            elif self.filter_size == 7:
                self.avgpool = 1

        elif self.dataset_train_name.startswith("MNIST"):
            self.nclasses = 10
            self.input_size = 28
            if self.filter_size < 7:
                self.avgpool = 14  #TODO
            elif self.filter_size == 7:
                self.avgpool = 7

        self.model = getattr(models, self.net_type)(
            nfilters=self.nfilters,
            avgpool=self.avgpool,
            nclasses=self.nclasses,
            nmasks=self.nmasks,
            unique_masks=self.unique_masks,
            level=self.level,
            filter_size=self.filter_size,
            first_filter_size=self.first_filter_size,
            act=self.act,
            scale_noise=self.scale_noise,
            noise_type=self.noise_type,
            use_act=self.use_act,
            dropout=self.dropout,
            train_masks=self.train_masks,
            pool_type=self.pool_type,
            debug=self.debug,
            input_size=self.input_size,
            mix_maps=self.mix_maps
        )

        self.loss_fn = nn.CrossEntropyLoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if args.optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=self.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)  #increase weight decay for no-noise large models
        elif args.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim_method == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=self.lr,  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            """
            # use this to set different learning rates for training noise masks and regular parameters:
            self.optimizer = optim.SGD([{'params': [param for name, param in self.model.named_parameters() if 'noise' not in name]},
                                        {'params': [param for name, param in self.model.named_parameters() if 'noise' in name], 'lr': self.lr * 10},
                                        ], lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True) #"""
        else:
            raise(Exception("Unknown Optimization Method"))


    def learning_rate(self, epoch):
        if self.dataset_train_name == 'CIFAR10':
            new_lr = self.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 90)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'CIFAR100':
            new_lr = self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'MNIST':
            new_lr = self.lr * ((0.2 ** int(epoch >= 30)) * (0.2 ** int(epoch >= 60))* (0.2 ** int(epoch >= 90)))
        elif self.dataset_train_name == 'FRGC':
            new_lr = self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'ImageNet':
            decay = math.floor((epoch - 1) / 30)
            new_lr = self.lr * math.pow(0.1, decay)
            #print('\nReducing learning rate to {}\n'.format(new_lr))
        return new_lr


    def train(self, epoch, dataloader):
        self.model.train()

        lr = self.learning_rate(epoch+1)

        for param_group in self.optimizer.param_groups:
            #print(param_group)  #TODO figure out how to set diff learning rate to noise params if train_masks
            param_group['lr'] = lr

        losses = []
        accuracies = []
        for i, (input, label) in enumerate(dataloader):
            if self.cuda:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            loss = self.loss_fn(output, label)
            if self.debug:
                print('\nBatch:', i)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.data.max(1)[1]

            acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size

            losses.append(loss.item())
            accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

    def test(self, dataloader):
        self.model.eval()
        losses = []
        accuracies = []
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloader):
                if self.cuda:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                loss = self.loss_fn(output, label)

                pred = output.data.max(1)[1]
                acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size
                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

print('\n\n****** Creating {} model ******\n\n'.format(args.net_type))
if args.resume is None or False ==  os.path.exists(args.resume):
    print('\n\nLoading model from saved checkpoint at {} is not exist \n\n'.format(args.resume))
    exit(0)
print('\n\nLoading model from saved checkpoint at {}\n\n'.format(args.resume))
setup = Model(args)
print('\n\n****** Preparing {} dataset *******\n\n'.format(args.dataset_train))
dataloader = Dataloader(args, setup.input_size)
train_loader, test_loader = dataloader.create()
setup.model = torch.load(args.resume)
model = setup.model
batch_size = args.batch_size
#model.load_state_dict(torch.load(args.resume))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

total_sample = 0
right_sample = 0
model.eval()  # 验证模型
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(device)
    # convert output probabilities to predicted class(将输出概率转换为预测类)
    _, pred = torch.max(output, 1)    
    # compare predictions to true label(将预测与真实标签进行比较)
    correct_tensor = pred.eq(target.data.view_as(pred))
    # correct = np.squeeze(correct_tensor.to(device).numpy())
    total_sample += batch_size
    for i in correct_tensor:
        if i:
            right_sample += 1
print("Accuracy:",100*right_sample/total_sample,"%")
def show_train_data(train_loader):
    # Get a batch from the training data loader
    images, labels = next(iter(train_loader))
    
    # Reshape the images back to 3x32x32 for plotting and transpose to 32x32x3 for matplotlib
    images = images.permute(0, 2, 3, 1)
    
    # Display a few images
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i in range(5):
        # Denormalize the image for display
        img = images[i].numpy() * np.array(NORMALIZATION_STD) + np.array(NORMALIZATION_MEAN)
        img = np.clip(img, 0, 1)  # Clip values to be within [0, 1] after denormalization
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {classes[labels[i]]}")
        axes[i].axis('off')
    #plt.show()
    plt.savefig("train.png")
# Visualize predictions
def visualize_predictions(model, test_loader, classes, device, num_images=8):
    model.eval()
    images_shown = 0

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break

                # Denormalize image for display
                img = inputs[i].cpu()
                img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)

                ax = axes[images_shown//4, images_shown%4]
                ax.imshow(img.permute(1, 2, 0))

                true_label = classes[targets[i]]
                pred_label = classes[predicted[i]]
                color = 'green' if targets[i] == predicted[i] else 'red'

                ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
                ax.axis('off')

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    #plt.show()
    plt.savefig("test.png")
show_train_data(train_loader)
#visualize_predictions(model, test_loader, num_images=10)
visualize_predictions(model, test_loader, classes, device)



