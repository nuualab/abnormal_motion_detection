#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import os
import json
import torch
import cv2
import pandas as pd
import numpy as np
import glob2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import albumentations as A
import albumentations.pytorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import random
from pytorchcv.model_provider import get_model as ptcv_get_model

inputdir = './input/'
outputdir = './output/'
weightdir = "./FallDown_efficientnetb4b_github/fallweight.pth"

testfile = sorted(glob2.glob(inputdir+'/*'))

net = ptcv_get_model('efficientnet_b4b', pretrained=True)

class EfficientNet_model(nn.Module):
    def __init__(self, net):
        super(EfficientNet_model, self).__init__()        
        self.backbone = net.features
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(1792, 1)
        
    def forward(self, input):
        x = self.backbone(input)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        output = self.out(x)
        
        return output

Net = EfficientNet_model(net).cuda()
Net.load_state_dict(torch.load(weightdir))
Net.requires_grad_(False)
Net.eval()

val_transform = albumentations.Compose(
    [
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensor()
    ]
)