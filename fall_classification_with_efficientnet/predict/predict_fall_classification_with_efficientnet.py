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
