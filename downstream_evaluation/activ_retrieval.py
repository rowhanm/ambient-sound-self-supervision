import os
from scipy.ndimage.filters import gaussian_filter
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from shutil import copyfile

import itertools
import cv2

import utils.models


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pretext_cluster_size = 60

data = sys.argv[1]
conv = int(sys.argv[4])
model_path = sys.argv[3]
arch = sys.argv[2]
count = 9

model = models.__dict__[arch](sobel=False, out=pretext_cluster_size)
checkpoint = torch.load(model_path)
def rename_key(key):
    if not 'module' in key:
        return key
    return ''.join(key.split('.module'))

checkpoint = {rename_key(key): val
                for key, val
                in checkpoint.items()}
model.load_state_dict(checkpoint)


model.to(device)
for params in model.parameters():
    params.requires_grad = False
model.eval();

repo = os.path.join(sys.argv[5], 'conv' + str(conv))
if not os.path.isdir(repo):
    os.makedirs(repo)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

tra = [transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize]


dataset = datasets.ImageFolder(data, transform=transforms.Compose(tra))

subs = list(range(0, len(dataset)//10)) # Run on 1/10th of the dataset

subset = torch.utils.data.Subset(dataset, subs)

dataloader = torch.utils.data.DataLoader(subset, batch_size=256, num_workers=4)

def forward(model, my_layer, x):
    layer = 1
    res = {}
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.ReLU):
                if layer == my_layer:
                    for channel in range(int(x.size()[1])):
                        key = 'layer' + str(layer) + '-channel' + str(channel)
                        res[key] = torch.squeeze(x.mean(3).mean(2))[:, channel]
                    return res
                layer = layer + 1
    return res

layers_activations = {}

for i, (input_tensor, _) in enumerate(dataloader):
    input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
    activations = forward(model, conv, input_var)

    if i == 0:
        layers_activations = {filt: np.zeros(len(subset)) for filt in activations}
    if i < len(dataloader) - 1:
        e_idx = (i + 1) * 256
    else:
        e_idx = len(subset)
    s_idx = i * 256
    for filt in activations:
        layers_activations[filt][s_idx: e_idx] = activations[filt].cpu().data.numpy()

    if i % 100 == 0:
        print('{0}/{1}'.format(i, len(dataloader)))

for filt in layers_activations:
    repofilter = os.path.join(repo, filt)
    if not os.path.isdir(repofilter):
        os.mkdir(repofilter)
    top = np.argsort(layers_activations[filt])[::-1]
    if count > 0:
        top = top[:count]
    imgs = []
    name = os.path.join(repofilter, "combined.jpg")
    for pos, img in enumerate(top):
        src, _ = dataset.imgs[img]
        copyfile(src, os.path.join(repofilter, "{}_{}".format(pos, src.split('/')[-1])))
        imgs.append(os.path.join(repofilter, "{}_{}".format(pos, src.split('/')[-1])))
    imgs = [cv2.imread(im) for im in imgs]
    imgs = [cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC) for img in imgs]
    #Define the shape of the image to be replicated (all images should have the same shape)
    img_h, img_w, img_c = imgs[0].shape

    #Define the margins in x and y directions
    m_x = margin
    m_y = margin

    #Size of the full size image
    mat_x = img_w * w + m_x * (w - 1)
    mat_y = img_h * h + m_y * (h - 1)

    #Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
    imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
    imgmatrix.fill(255)

    #Prepare an iterable with the right dimensions
    positions = itertools.product(range(h), range(w))

    for (y_i, x_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    cv2.imwrite(name, resized, compression_params)
