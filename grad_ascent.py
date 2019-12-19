#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from scipy.ndimage.filters import gaussian_filter
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# In[2]:


sys.path.append("../deepcluster/")
import models


# In[28]:


device = torch.device('cuda')
pretext_cluster_size = 60
conv = 1
model_path = "./ambient-alexnet-model_60.pt"
arch = "alexnet"
learning_rate = 3
weight_decay = 0.00001
sigma = 0.3
step = 5
niter = 50000
CONV = {'alexnet': [96, 256, 384, 384, 256]}


# In[29]:


checkpoint = torch.load(model_path)
model = models.__dict__[arch](sobel=False, out=pretext_cluster_size)
def rename_key(key):
    if not 'module' in key:
        return key
    return ''.join(key.split('.module'))

checkpoint = {rename_key(key): val
                for key, val
                in checkpoint.items()}
model.load_state_dict(checkpoint)


# In[30]:


model.cuda()
for params in model.parameters():
    params.requires_grad = False
model.eval();


# In[31]:


def deprocess_image(x):
    x = x[0, :, :, :]
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[32]:


def gradient_ascent(f):
    print(f)
    sys.stdout.flush()
    fname_out = '{0}/layer{1}-channel{2}.jpeg'.format(repo, conv, f)

    img_noise = np.random.normal(size=(224,224,3)) * 20 + 128
    img_noise = img_noise.astype('float32')
    inp = transforms.ToTensor()(img_noise)
    inp = torch.unsqueeze(inp, 0)

    for it in range(niter):
        x = torch.autograd.Variable(inp.cuda(), requires_grad=True)
        out = forward(model, conv-1, f, x)
        criterion = nn.CrossEntropyLoss()
        filt_var = torch.autograd.Variable(torch.ones(1).long()*f).cuda()
        output = out.mean(3).mean(2)
        loss = - criterion(output, filt_var) - weight_decay*torch.norm(x)**2

        # compute gradient
        loss.backward()

        # normalize gradient
        grads = x.grad.data.cpu()
        grads = grads.div(torch.norm(grads)+1e-8)

        # apply gradient
        inp = inp.add(learning_rate*grads)

        # gaussian blur
        if it%step == 0:
            inp = gaussian_filter(torch.squeeze(inp).numpy().transpose((2, 1, 0)),
                                   sigma=(sigma, sigma, 0))
            inp = torch.unsqueeze(torch.from_numpy(inp).float().transpose(2, 0), 0)

        # save image at the last iteration
        if it == niter - 1:
            a = deprocess_image(inp.numpy())
            Image.fromarray(a).save(fname_out)


# In[33]:


def forward(model, layer, channel, x):
    count = 0
    for y, m in enumerate(model.features.modules()):
        if not isinstance(m, nn.Sequential):
            x = m(x)
            if isinstance(m, nn.Conv2d):
                if count == layer:
                    res = x
            if isinstance(m, nn.ReLU):
                if count == layer:
                    # check if channel is not activated
                    if x[:, channel, :, :].mean().data.cpu().numpy() == 0:
                        return res
                    return x
                count = count + 1


# In[34]:


repo = os.path.join("./visuals", 'conv' + str(conv))
if not os.path.isdir(repo):
    os.makedirs(repo)


# In[35]:


for i in range(CONV[arch][conv-1]):
    gradient_ascent(i)


# In[ ]:


import IPython.display as ipd
ipd.Image("visuals/conv1/layer1-channel0.jpeg")


# In[ ]:




