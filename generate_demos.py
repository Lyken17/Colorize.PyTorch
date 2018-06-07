import os, os.path as osp
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms

from model import TransformerNet
from transforms.color_space import Linearize, SRGB2XYZ, XYZ2CIE, CIE2XYZ, XYZ2SRGB, DeLinearize

model = TransformerNet(in_channels=2, out_channels=1)
model = nn.DataParallel(model)

state = torch.load("epoch_30.model")
model.load_state_dict(state)

SRGB2LMS = transforms.Compose([
    Linearize(),
    SRGB2XYZ(),
    XYZ2CIE(),
])

LMS2SRGB = transforms.Compose([
    CIE2XYZ(),
    XYZ2SRGB(),
    DeLinearize(),
])


def cut_range(img):
    img[img > 1] = 1
    img[img < 0] = 0
    return img


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    SRGB2LMS
])

visualize = transforms.Compose([
    LMS2SRGB,
    cut_range,
    transforms.ToPILImage()
])

raw_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

raw_dataset = datasets.ImageFolder("/mnt/hdd/download/mscoco", raw_transform)
train_dataset = datasets.ImageFolder("/mnt/hdd/download/mscoco", transform)

mse_loss = nn.MSELoss()


loss_hist = {}

for index, data in enumerate(train_dataset):
    img, label = data
    if label != 0:  # only test
        continue
    print(index)

    origin = transforms.ToPILImage()(raw_dataset[index][0])
    origin.save("demo/%d_origin.png" % index)

    imgs = torch.unsqueeze(img, 0)
    LS = torch.cat([imgs[:, :1, :, :].clone(), imgs[:, -1:, :, :].clone()], dim=1)
    gt = imgs[:, 1:2, :, :].clone().requires_grad_(False).cuda()
    M = model(LS)
    loss = mse_loss(M, gt)

    loss_hist["img-%d" % index] = loss.cpu().item()

    recover_LMS = torch.cat([
        imgs[:, :1, :, :].clone(),
        # torch.ones_like(M.cpu()),
        M.cpu(),
        imgs[:, -1:, :, :].clone()
    ], dim=1)
    recover = visualize(torch.squeeze(recover_LMS, 0))
    recover.save("demo/%d_recover.png" % index)
    
    if index > 200:
        break
    
with open ("hist.json", "w+") as fp:
    json.dump(loss_hist, fp, indent=4)
