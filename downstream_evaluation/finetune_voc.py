import sys
import argparse
import os
import math
import time
import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
device = torch.device('cuda')

import utils.models

from sklearn import metrics
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ft_util import AverageMeter, load_model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

fc6_8 = True
train_batchnorm = False

if sys.argv[1] == "all":
    fc6_8 = False
    train_batchnorm = True

pretext_cluster_size = int(sys.argv[2])

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

def train(loader, model, optimizer, criterion, fc6_8, losses, it=0, total_iterations=None, stepsize=None, verbose=True):
    # to log
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    current_iteration = it

    # use dropout for the MLP
    model.train()
    # in the batch norms always use global statistics
    model.features.eval()

    for (input, target) in loader:
        # measure data loading time
        data_time.update(time.time() - end)
        
        # adjust learning rate
        if current_iteration != 0 and current_iteration % stepsize == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                print('iter {0} learning rate is {1}'.format(current_iteration, param_group['lr']))

        # move input to gpu
        input = input.cuda(non_blocking=True)

        # forward pass with or without grad computation
        output = model(input)

        target = target.float().cuda()
        mask = (target == 255)
        loss = torch.sum(criterion(output, target).masked_fill_(mask, 0)) / target.size(0)

        # backward 
        optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # and weights update
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if verbose is True and current_iteration % 25 == 0:
            print('Iteration[{0}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   current_iteration, batch_time=batch_time,
                   data_time=data_time, loss=losses))
        current_iteration = current_iteration + 1
        if total_iterations is not None and current_iteration == total_iterations:
            break
    return current_iteration


def evaluate(loader, model, eval_random_crops):
    model.eval()
    gts = []
    scr = []
    for crop in range(9 * eval_random_crops + 1):
        for i, (input, target) in enumerate(loader):
            # move input to gpu and optionally reshape it
            if len(input.size()) == 5:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w)
            input = input.cuda(non_blocking=True)

            # forward pass without grad computation
            with torch.no_grad():
                output = model(input)
            if crop < 1 :
                    scr.append(torch.sum(output, 0, keepdim=True).cpu().numpy())
                    gts.append(target)
            else:
                    scr[i] += output.cpu().numpy()
    gts = np.concatenate(gts, axis=0).T
    scr = np.concatenate(scr, axis=0).T
    aps = []
    for i in range(20):
        # Subtract eps from score to make AP work for tied scores
        ap = metrics.average_precision_score(gts[i][gts[i]<=1], scr[i][gts[i]<=1]-1e-5*gts[i][gts[i]<=1])
        aps.append( ap )
    print(np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]))
    
    
class VOC2007_dataset(torch.utils.data.Dataset):
    def __init__(self, voc_dir, split='trainval', transform=None):
        # Find the image sets
        image_set_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        image_sets = glob.glob(os.path.join(image_set_dir, '*_' + split + '.txt'))
        assert len(image_sets) == 20
        # Read the labels
        self.n_labels = len(image_sets)
        images = defaultdict(lambda:-np.ones(self.n_labels, dtype=np.uint8)) 
        for k, s in enumerate(sorted(image_sets)):
            for l in open(s, 'r'):
                name, lbl = l.strip().split()
                lbl = int(lbl)
                # Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
                if lbl < 0:
                    lbl = 0
                elif lbl == 0:
                    lbl = 255
                images[os.path.join(voc_dir, 'JPEGImages', name + '.jpg')][k] = lbl
        self.images = [(k, images[k]) for k in images.keys()]
        np.random.shuffle(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i][0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.images[i][1]
    
checkpoint = torch.load(sys.argv[3])
alexnet = models.__dict__["alexnet"](sobel=False, out=pretext_cluster_size)
def rename_key(key):
    if not 'module' in key:
        return key
    return ''.join(key.split('.module'))

checkpoint = {rename_key(key): val
                for key, val
                in checkpoint.items()}
alexnet.load_state_dict(checkpoint)
alexnet.top_layer = nn.Linear(alexnet.top_layer.weight.size(1), 20)
alexnet.cuda()
cudnn.benchmark = True


for y, m in enumerate(alexnet.classifier.modules()):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0.1)
    alexnet.top_layer.bias.data.fill_(0.1)
    
    
if fc6_8:
   # freeze some layers 
    for param in alexnet.features.parameters():
        param.requires_grad = False
    # unfreeze batchnorm scaling
    if train_batchnorm:
        for layer in alexnet.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                for param in layer.parameters():
                    param.requires_grad = True

optimizer = torch.optim.SGD(
    filter(lambda x: x.requires_grad, alexnet.parameters()),
    lr=0.003,
    momentum=0.9,
    weight_decay=1e-6,
)


criterion = nn.BCEWithLogitsLoss(reduction='none')

dataset = VOC2007_dataset(sys.argv[4], split="trainval", transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.1, 0.5), ratio=(1, 1)),
            transforms.ToTensor(),
            normalize,
         ]))

loader = torch.utils.data.DataLoader(dataset,
         batch_size=16, shuffle=False,
         num_workers=8, pin_memory=True)

print('Start training')
it = 0
losses = AverageMeter()
while it < 200000:
    it = train(
        loader,
        alexnet,
        optimizer,
        criterion,
        fc6_8,
        losses,
        it=it,
        total_iterations=200000,
        stepsize=20000,
    )

print('Evaluation')
transform_eval = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.1, 0.5), ratio=(1, 1)), 
    transforms.ToTensor(),
    normalize,
]

print('Train set')
train_dataset = VOC2007_dataset(sys.argv[4], split="trainval", transform=transforms.Compose(transform_eval))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=24, 
    pin_memory=True,
)
evaluate(train_loader, alexnet, eval_random_crops=True)


print('Test set')
test_dataset = VOC2007_dataset(sys.argv[4], split='test', transform=transforms.Compose(transform_eval))
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=24, 
    pin_memory=True,
)
evaluate(test_loader, alexnet, eval_random_crops=True)
    
    
