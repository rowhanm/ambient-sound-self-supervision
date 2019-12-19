#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../deepcluster/")

import torch
import torchvision
import torchvision.transforms as transforms


import argparse
import os
import time

import models
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler


from util import AverageMeter, learning_rate_decay, load_model, Logger


# In[20]:


model_path = sys.argv[2]
pretext_cluster_size = 60
conv = int(sys.argv[1]) # 1,2,3,4,5
tencrops = True
exp = "./exp"
epochs = 90
batch_size = 128
lr = 0.01
momentum = 0.9
wd = -4


# In[3]:


best_prec1 = 0
checkpoint = torch.load(model_path)
alexnet = models.__dict__["alexnet"](sobel=False, out=pretext_cluster_size)
def rename_key(key):
    if not 'module' in key:
        return key
    return ''.join(key.split('.module'))

checkpoint = {rename_key(key): val
                for key, val
                in checkpoint.items()}
alexnet.load_state_dict(checkpoint)
alexnet.cuda()
cudnn.benchmark = True


# In[4]:


# freeze the features layers
for param in alexnet.features.parameters():
    param.requires_grad = False


# In[5]:


criterion = nn.CrossEntropyLoss().cuda()


# In[6]:


dataset_path = os.path.join("/scratch/work/public/imagenet", 'train')


# In[7]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# In[8]:


transformations_val = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
        ]


# In[9]:


transformations_train = [transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.RandomCrop(224),
#                              transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize]


# In[10]:


dataset = datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose(transformations_train)
    )


# In[11]:


validation_split = .2
shuffle_dataset = True
random_seed= 42


# In[12]:


dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# In[21]:


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, num_workers=8, pin_memory=True)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)


# In[14]:


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, conv, num_labels):
        super(RegLog, self).__init__()
        self.conv = conv
        if conv==1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv==2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv==3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


# In[16]:


# logistic regression
reglog = RegLog(conv, len(dataset.classes)).cuda()
optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, reglog.parameters()),
    lr,
    momentum=momentum,
    weight_decay=10**wd
)


# In[17]:


def forward(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, reglog, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # freeze also batch norm layers
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        #adjust learning rate
        learning_rate_decay(optimizer, len(train_loader) * epoch + i, lr)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        # compute output

        output = forward(input_var, model, reglog.conv)
        output = reglog(output)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, reglog, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    for i, (input_tensor, target) in enumerate(val_loader):
        if tencrops:
            bs, ncrops, c, h, w = input_tensor.size()
            input_tensor = input_tensor.view(-1, c, h, w)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = reglog(forward(input_var, model, reglog.conv))

        if tencrops:
            output_central = output.view(bs, ncrops, -1)[: , ncrops / 2 - 1, :]
            output = softmax(output)
            output = torch.squeeze(output.view(bs, ncrops, -1).mean(1))
        else:
            output_central = output

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input_tensor.size(0))
        top5.update(prec5[0], input_tensor.size(0))
        loss = criterion(output_central, target_var)
        losses.update(loss.item(), input_tensor.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


# In[18]:


# create logs
exp_log = os.path.join(exp, 'log')
if not os.path.isdir(exp_log):
    os.makedirs(exp_log)

loss_log = Logger(os.path.join(exp_log, 'loss_log'))
prec1_log = Logger(os.path.join(exp_log, 'prec1'))
prec5_log = Logger(os.path.join(exp_log, 'prec5'))


# In[22]:


for epoch in range(epochs):
    end = time.time()
    train(train_loader, alexnet, reglog, criterion, optimizer, epoch)
    prec1, prec5, loss = validate(val_loader, alexnet, reglog, criterion)
    loss_log.log(loss)
    prec1_log.log(prec1)
    prec5_log.log(prec5)
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        filename = 'imagenet_model_best.pth.tar'
    else:
        filename = 'imagenet_checkpoint.pth.tar'
    torch.save({
        'epoch': epoch + 1,
        'arch': 'alexnet',
        'state_dict': model.state_dict(),
        'prec5': prec5,
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, os.path.join(exp, filename))


# In[ ]:




