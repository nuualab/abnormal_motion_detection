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

class ResNest_model(nn.Module):
    def __init__(self, net):
        super(ResNest_model, self).__init__()  
        #del net.fc
        self.backbone = net
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 1)
        
    def forward(self, input):
        x = self.backbone(input)
        #x = torch.flatten(x, 1)
        x = self.drop(x)
        output = self.out(x)
        
        return output

class FallDataset(Dataset):

    def __init__(self, img_ids, transform):
        self.img_ids = img_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image = self.img_ids[idx]
        img = cv2.imread(image, cv2.IMREAD_COLOR)[..., ::-1]
        
        if self.transform:
            augmented = self.transform(image=img) 
            image = augmented['image']
        
        return image
    
def falldown(testfile, Net, threshold):
    testset = FallDataset(testfile, val_transform)
    test_loader = DataLoader(testset, batch_size=1, num_workers = 2, shuffle=False)
    scores= []
    label = []
    for j, d in enumerate(test_loader):
                with torch.no_grad():
                    score = F.sigmoid(Net(d.cuda()))
                    scores += score.tolist()
                    
    S = np.array(scores)
    Scores = np.concatenate(S)

    for j in range(len(Scores)):
        if Scores[j]>(threshold):
            label.append(1)
        else:
            label.append(0)
    
    return label  

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str,
                help="Where is input dir")
    parser.add_argument('--device', type=str,
                help="Which device")

    args = parser.parse_args()
    
    inputdir = args.inputdir
    device = args.device
    threshold = 0.8

    outputdir = './output/'
    weightdir = "./FallDown_resnest100_github/fallweight.pth"
  

    testfile = sorted(glob2.glob(inputdir+'/*'))

    torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
    net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest100', pretrained=False)
    Net = ResNest_model(net).to(device)
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

    output = falldown(testfile, Net, threshold)
    answer = pd.DataFrame(testfile)  
    answer['label'] = output
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    answer.to_csv(outputdir+'output.txt', index=False,  header=None)




