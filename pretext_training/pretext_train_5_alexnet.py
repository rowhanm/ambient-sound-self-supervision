import sys

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import h5py
import math

import models
from util import AverageMeter, Logger, UnifLabelSampler

import torch.nn as nn
import IPython.display as ipd
from PIL import Image
device = torch.device('cuda')

import matplotlib.pyplot as plt

import time
from pathlib import Path
import cv2
import glob

print("[LOADING MODELS..]")

alexnet = models.__dict__["alexnet"](sobel=False,out=5)
alexnet.features = torch.nn.DataParallel(alexnet.features)
alexnet.cuda()
cudnn.benchmark = True


print("[LOADING DATAFRAMES..]")
train_dataframe = pd.read_csv("train_dataset_5.csv", header=None)
count=0
train_badlist=[]
for i in range(len(train_dataframe)):
    img = os.path.join("./dataset/frames",train_dataframe.iloc[i,0])
    if not os.path.exists(img):
        count+=1
        train_badlist.append(train_dataframe.iloc[i,0])
        
        
val_dataframe = pd.read_csv("val_dataset_5.csv", header=None)
count=0
val_badlist=[]
for i in range(len(val_dataframe)):
    img = os.path.join("./dataset/frames",val_dataframe.iloc[i,0])
    if not os.path.exists(img):
        count+=1
        val_badlist.append(val_dataframe.iloc[i,0])


for index,row in train_dataframe.iterrows():
    if row[0] in train_badlist:
        train_dataframe.drop(index, inplace=True)
for index,row in val_dataframe.iterrows():
    if row[0] in val_badlist:
        val_dataframe.drop(index, inplace=True)


print("[LOADING DATA..]")

class SoundsDataset(Dataset):
    def __init__(self, dataframe, h5_file, transform=None):
        super(SoundsDataset, self).__init__()
        self.h5 = h5_file
        self.transform = transform
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with h5py.File(self.h5, 'r') as hf:    
            dset = hf["img_"+str(idx)]
            label = hf["lab_"+str(idx)].value[0]    
            image = Image.fromarray(np.array(dset[:,:,:])) 
            if self.transform:
                image = self.transform(image)
            return image,label
        
def get_data():
    t0,t1,t2,t3,t4,t5,t6,t7,t8,t9 = get_transforms()
    train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t0),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t1),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t2),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t3),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t4),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t5),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t6),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t7),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t8),
        SoundsDataset(train_dataframe, './dataset/train_dataset_5.hdf5', transform=t9)
    ]),
    batch_size=256, shuffle=True, num_workers = 8)
    
    val_loader = torch.utils.data.DataLoader(
        SoundsDataset(val_dataframe, './dataset/val_dataset_5.hdf5', transform=t0),
        batch_size=256, shuffle=False, num_workers = 8)
    
    return train_loader, val_loader    
    
    
# Get all the possible transforms for a given size.
def get_transforms():
    # Keep the same
    t0 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Scale brightness between the range (1.5,3.5)
    t1 = transforms.Compose([
        transforms.ColorJitter(brightness=2.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])
    
    # Scale saturation between (1,2)
    t2 = transforms.Compose([
        transforms.ColorJitter(saturation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Scale contrast between (1,1.5)
    t3 = transforms.Compose([
        transforms.ColorJitter(contrast=1.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Scale hue
    t4 = transforms.Compose([
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Random horizontal flips
    t5 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Random shearing
    t6 = transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=3),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Random Translation
    t7 = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.2,0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Random perspective change
    t8 = transforms.Compose([
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    # Random rotation
    t9 = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4054, 0.3780, 0.3547), std = (0.2221, 0.2151, 0.2112))])

    return t0,t1,t2,t3,t4,t5,t6,t7,t8,t9

train_dl, val_dl = get_data()

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)
            
            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(alexnet.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=200)

def train(model, iterator, optimizer, criterion, sched=None):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch_idx,(x, y) in enumerate(iterator):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        fx = model(x)
        
        loss = criterion(fx, y)
        
        acc = calculate_accuracy(fx, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        
        lr_sched_test = scheduler.get_lr()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


print("[TRAINING..]")

#state_dict = torch.load("ambient-alexnet-model_5.pt")
#alexnet.load_state_dict(state_dict)

#print("[LOADED EXISTING MODEL]")

EPOCHS = 200

best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(alexnet, train_dl, optimizer, criterion, scheduler)
    valid_loss, valid_acc = evaluate(alexnet, val_dl, criterion)
    print("[EPOCH " + str(epoch)+"]")
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(alexnet.state_dict(), 'ambient-alexnet-model_5.pt')
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
