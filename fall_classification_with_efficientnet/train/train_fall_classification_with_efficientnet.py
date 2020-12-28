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
from torch.optim import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import random
from tqdm.autonotebook import tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--falldir', type=str,
            help="Where is input dir")
parser.add_argument('--Nofalldir', type=str,
            help="Where is input dir")
parser.add_argument('--device', type=str,
            help="Which device")
parser.add_argument('--lr', type=float,
            help="Learning Rate")
parser.add_argument('--epochs', type=int,
            help="Num Epochs")
args = parser.parse_args()

NF = sorted(glob2.glob(args.Nofalldir))
F = sorted(glob2.glob(args.falldir))
device = args.device
lr = args.lr
num_epochs = args.epochs
outputdir = './output/'

train_F = []
valid_F = []
for i in range(len(F)):
    if i % 20 == 0:
        valid_F.append(F[i])
    else:
        train_F.append(F[i])

train_NF = []
valid_NF = []
for i in range(len(NF)):
    if i % 20 == 0:
        valid_NF.append(NF[i])
    else:
        train_NF.append(NF[i])
            
train_image = train_F + train_NF
valid_image = valid_F + valid_NF
train_label = [1] * len(train_F) + [0] * len(train_NF)
valid_label = [1] * len(valid_F) + [0] * len(valid_NF)

def compute_iou(box_a, box_b):
    max_x = min(box_a[3], box_b[3])
    max_y = min(box_a[4], box_b[4])
    min_x = max(box_a[1], box_b[1])
    min_y = max(box_a[2], box_b[2])
    
    intersection = max(max_x-min_x, 0) * max(max_y-min_y, 0)

    area_a = (box_a[3] - box_a[1]) * (box_a[4] - box_a[2])
    area_b = (box_b[4] - box_b[2]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union

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

albumentations_transform = A.Compose([
    A.Resize(256, 256), 
    A.RandomCrop(224, 224),
    A.Flip(p=0.5), # Same with transforms.RandomHorizontalFlip()
    A.OneOf([A.MotionBlur(p=0.3),
                         A.Blur(p=0.3),
                          A.GaussNoise(p=0.3) ]),
    A.Rotate(p=0.2),
    
    A.RandomBrightnessContrast(p=0.2),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    albumentations.pytorch.transforms.ToTensor()
    
])

val_transform = albumentations.Compose(
    [
        albumentations.Resize(224, 224),
        albumentations.CenterCrop(height=128, width=128),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensor()
    ]
)

transform = A.Compose([
        A.Resize(256, 256), 
        A.RandomCrop(224, 224),   
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=0.2),
        A.RandomBrightnessContrast(p=0.2, brightness_limit = 0.8, contrast_limit = 0.8), 
    
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.Blur(p=0.2),
        ],p=0.2),

        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),                   
        ], p=0.2),

        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2), 

        A.OneOf([
            A.RandomFog(p=0.05),
            A.RandomShadow(p=0.05),
            A.RandomSnow(p=0.05)
        ], p=0.1),
    
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensor()
    ])

def UpDown_Resize(image, upsize, downsize, p=0.1):
    num = random.random()
    if num<p:
        image = cv2.resize(image, (upsize,upsize))
        image = cv2.resize(image, (downsize,downsize))
        
    return image

def black_BBox_inPerson(image, num=5, p=0.1, size=20):
    if random.random()<p:
        h,w,_ = image.shape
        x1 = 0
        y1 = 0
        x2 = w-20
        y2 = h-20  
        s = int(min(size, min((x2 - x1) / 8, (y2 - y1) / 8)))
        if x2 <= 0 or y2 <= 0:
            num = 0
        for i in range(num):
            xR = random.randint(x1,x2)
            yR = random.randint(y1,y2)
            image[yR:yR + s, xR:xR + s] = (0,0,0)        
        
    return image

def cvt_img_day_to_night(img):
    if random.choice((0, 0, 0, 1)) == 0:
        return img

    # (90, 70, 55): 석양
    # (70, 85, 90): 새벽
    # (random.randint(50, 150), ) * 3: 그냥 어두워지는 것
    sub_val = random.choice([(90, 70, 55), (random.randint(50, 150), ) * 3, (70, 85, 90)])
    night_img = cv2.subtract(img, np.full(img.shape, sub_val, dtype=np.uint8))

    resize_val = 1 / (1.2 + random.random() * 0.6)
    night_img = cv2.resize(night_img, dsize=(0, 0), fx=resize_val, fy=resize_val, interpolation=cv2.INTER_LINEAR)
    night_img = cv2.resize(night_img, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    return night_img

class Dataset(Dataset):

    def __init__(self, img_ids, labels, transform):
        self.img_ids = img_ids
        self.labels  = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image = self.img_ids[idx]
        img = cv2.imread(image, cv2.IMREAD_COLOR)[..., ::-1]
        #size = 224
        img, scale = aspectaware_resize_padding(img, size, size)
        img = UpDown_Resize(img, upsize=1024, downsize=128, p=0.1)
        img = black_BBox_inPerson(img, num=3, p=0.2)
        img = cvt_img_day_to_night(img)       
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=img) 
            image = augmented['image']
        
        return image, torch.as_tensor([label], dtype=torch.float32)

def aspectaware_resize_padding(img, width, height, means =(0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225),                               interpolation=None):
    #normalization
    image = (img / 255 - means) / std 
    
    old_h, old_w, c = image.shape    
    if old_h < old_w:
        new_w = width
        new_h = int(width / old_w * old_h)
        scale = width/old_w
    else: 
        new_h = height
        new_w = int(height/old_h * old_w)
        scale = height/old_h
   
    canvas = np.zeros((height, width, c), np.float32)
    canvas[...] = means

    image = cv2.resize(image, (new_w, new_h))

    canvas[:new_h, :new_w] = image
    
    return canvas, scale 

trainset = Dataset(train_image, train_label, transform)
train_loader = DataLoader(trainset, batch_size=32, num_workers=2, shuffle=True)
validset = Dataset(valid_image, valid_label, val_transform)
valid_loader = DataLoader(validset, batch_size=32, num_workers=2, shuffle=False)

if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        
def save_checkpoint(model, name):
    torch.save(model.state_dict(), os.path.join(outputdir, name))

param_optimizer = list(Net.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

optimizer = torch.optim.AdamW(Net.parameters(), 1e-5)   
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


Loss = nn.BCEWithLogitsLoss().cuda()

acc = []
for epoch in range(num_epochs):   
    Net.train()
    epoch_loss = []
    iters = len(train_loader)
    progress_bar = tqdm(train_loader)
    step = 1
    total_loss = 0.0
    for i, batch in enumerate(progress_bar):       
        images, targets = batch
        imgs = images.cuda().float()
        batch_size = images.shape[0]
        label0 = targets.cuda()
        #print(imgs.size())
        #print(label.size())
        optimizer.zero_grad()
   #     #cls_loss, reg_loss = model(imgs, annot)
        output = Net(imgs)
   #     #print(output.size())
        cls_loss = Loss(output, label0)
        cls_loss = cls_loss.mean()

        loss = cls_loss
        if loss == 0 or not torch.isfinite(loss):
                continue
        #loss = Variable(loss, requires_grad = True)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step() #epoch + i / iters
        epoch_loss.append(float(loss))

        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}.  Total loss: {:.5f}.'.format(
                                step, epoch+1, num_epochs, loss.item()))
        step += 1
        total_loss = total_loss + loss
        if (step == len(train_loader)):
            save_checkpoint(Net, f'_efficientnetb4b_{epoch+1}.pth')
            print('checkpoint...')
        if step % 300 == 0:
            print(f'step : {step}')
            print(total_loss / step)
    print('Epoch {}, lr {}'.format(
            epoch+1, optimizer.param_groups[0]['lr']))
            
    Net.eval()    
    total_loss = 0.0
    answer = 0
    for i, batch in enumerate(valid_loader):  
        with torch.no_grad():
            images, targets = batch
            imgs = images.cuda().float()
            batch_size = images.shape[0]
            label0 = targets.cuda()
            #print(imgs.size())
            #print(label.size())
            optimizer.zero_grad()
            #cls_loss, reg_loss = model(imgs, annot)
            output = Net(imgs)
            #print(output.size())
            cls_loss = Loss(output, label0)
            cls_loss = cls_loss.mean()
            total_loss = total_loss + cls_loss
            acc_output = torch.sigmoid(output)
            for j in range(len(acc_output)):
                if acc_output[j] >= 0.5 and label0[j] == 1:
                    answer += 1
                elif acc_output[j] < 0.5 and label0[j] == 0:
                    answer += 1
                else:
                    answer += 0
                    
    print(f'epoch: {epoch + 1}, valid_loss: {total_loss / len(valid_loader)}')
    print(f'epoch: {epoch + 1}, accuracy: {answer / len(validset)}')
    acc.append(answer / len(validset))
    


