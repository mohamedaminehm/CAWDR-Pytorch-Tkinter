
from tqdm import tqdm
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import scipy as sp
import matplotlib.pyplot as plt
from skimage import io, transform, morphology, measure 
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, shannon_entropy
from sklearn.model_selection import train_test_split
from scipy import ndimage as ndi
from collections import Counter

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import mahotas

import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import torchextractor as tx
from PIL import Image 
import math
import pickle



import gc

#gc.collect()

MODEL_SEG = 'UNET_RESNET34ImgNet'#'IUNET'#'UNET_RESNET34ImgNet' # UNET |  | UNET_RESNET34ImgNet 
# Fetch U-Net with a pre-trained RESNET34 encoder on imagenet
if MODEL_SEG == 'UNET_RESNET34ImgNet':
    #!pip install git+https://github.com/qubvel/segmentation_models.pytorch > /dev/null 2>&1 # Install segmentations_models.pytorch, with no bash output.
    import segmentation_models_pytorch as smp



class CustomDataset(Dataset):
    def __init__(self, im_data, transform=None):
        self.im_data = im_data
        self.transform = transform
        self.img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # use mean and std from ImageNet 

    def __len__(self):
        return len(self.im_data)
       
               
    def __getitem__(self, index):
        img = self.im_data[index]
        
        if self.transform is not None: 
            img = self.transform(img)
            
        return self.img_transform(img)


def crop_data(img,target_shape= (512,512)):
        imgs = []

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        #img = (img/255 - mean) / std

        x = int(img.shape[1] / 8)
        img_1 = img[:, :x]
        img_2 = img[:, x:x*2]
        img_3 = img[:, x*2:x*3]
        img_4 = img[:, x*3:x*4]
        img_5 = img[:, x*4:x*5]
        img_6 = img[:, x*5:x*6]
        img_7 = img[:, x*6:x*7]
        img_8 = img[:, x*7:]



        img_1 = cv2.resize(img_1,target_shape)
        img_2 = cv2.resize(img_2,target_shape)
        img_3 = cv2.resize(img_3,target_shape)
        img_4 = cv2.resize(img_4,target_shape)
        img_5 = cv2.resize(img_5,target_shape)
        img_6 = cv2.resize(img_6,target_shape)
        img_7 = cv2.resize(img_7,target_shape)
        img_8 = cv2.resize(img_8,target_shape)


        imgs.append(img_1)
        imgs.append(img_2)
        imgs.append(img_3)
        imgs.append(img_4)
        imgs.append(img_5)
        imgs.append(img_6)
        imgs.append(img_7)
        imgs.append(img_8)


        return imgs

    
class SegmentationClass():
    def __init__(self,original_image):
        self.original_image = original_image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seg_model = smp.Unet("resnet50")#, encoder_weights="imagenet", activation=None)

        FREEZE_RESNET = False   # if UNET_RESNET34ImgNet
        if FREEZE_RESNET == False:
            for name, p in self.seg_model.named_parameters():
                if "encoder" in name:
                    p.requires_grad = False
        model_path = Path('seg_model/model_6.pt')
        state = torch.load(str(model_path),map_location=torch.device('cpu'))
        epoch = state['epoch']
        step = state['step']
        self.seg_model.load_state_dict(state['model'])
        self.seg_model.eval()
        print(f'segmentation model trained over {epoch} epoch !')


        #self.get_model = getattr(torchvision.models, model_name)
    

    def segmentation(self):
        print("segmentation start ..")
        image = self.original_image
        image = cv2.resize(image,(4000,800))
        imgs = crop_data(image)
        inp = CustomDataset(imgs, transform=None)
        ldr = DataLoader(inp, batch_size=2, shuffle=False, num_workers=0)
        
        rst = []
        for i_batch, inputs in tqdm(enumerate(ldr), total=len(ldr)):
            images = inputs.float().to(self.device)
            out = self.seg_model.forward(images)
            out = ((out > 0).float()) > 0.5
            out = out.data.cpu()
            print(out.shape)
            for i in range(len(out)):
                rst.append(out[i][0])

        #images = next(iter(ldr))
        #images = images.float().to(self.device)
        #out = self.seg_model.forward(images)
        #out = ((out > 0).float()) * 255
        #images = images.data.cpu()
        #out = out.data.cpu()
        resultat_ = np.hstack(rst).astype(np.uint8)
        print("segmentation was finished")
        del self.seg_model
        #gc.collect()
        #resultat_ = resultat_.astype(int)
        print(resultat_.shape)
        mask_resultat = morphology.remove_small_objects(resultat_ > 0 , 100,connectivity=4)
        mask_resultat = morphology.remove_small_holes(mask_resultat, 100,connectivity=1)

        #footprint = morphology.disk(1)
        #wh = morphology.white_tophat(mask_resultat * 1 , footprint)

        return ((mask_resultat )*255).astype(np.uint8)  # -wh
    
    def segmentation2(self):
        print("segmentation start ..")
        image = self.original_image
        image = cv2.resize(image,(512,512))
        #inp = CustomDataset([image], transform=None)
        #ldr = DataLoader(inp, batch_size=1, shuffle=False, num_workers=0)
        #images = next(iter(ldr))
        images = transforms.ToTensor()(image)
        images =transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images).unsqueeze_(0)

        images = images.float().to(self.device)
        out = self.seg_model.forward(images)
        out = ((out > 0).float()) > 0.5
        #images = images.data.cpu()
        out = out.data.cpu()
        #resultat_ = np.hstack([out[0][0],out[1][0],out[2][0],out[3][0],out[4][0],out[5][0]])
        resultat_ = np.hstack([out[0][0]])
        resultat_ = cv2.resize(resultat_.astype(np.uint8), (4000,800))
        print("segmentation was finished")
        del self.seg_model
        gc.collect()

        #resultat_ = resultat_.astype(np.int8) + 128
        mask_resultat = morphology.remove_small_objects(resultat_> 0, 100, connectivity=4)
        mask_resultat = morphology.remove_small_holes(mask_resultat, 100, connectivity=1)
        
        #footprint = morphology.disk(1)
        #wh = morphology.white_tophat(mask_resultat * 1 , footprint)

        

        #mask_resultat = clear_border(mask_resultat * 1)
        return (mask_resultat * 255).astype(np.uint8)