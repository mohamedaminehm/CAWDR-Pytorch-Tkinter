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





import socket
import pickle 
import cv2
import struct

HOST = "127.0.0.1"
PORT = 5050

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]



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


###### Segmentation with U-Net ######
class Segmentation():
    def __init__(self):
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

    def segmentation(self,image):
            print("segmentation start ..")
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
            #del self.seg_model

            return resultat_


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))
    s.listen()
    print("server is listening in prot 5050")

    seg = Segmentation()
    print("segmentation model initialized")
    while True:
        conn, addr = s.accept()
        with conn:
            print("connected to" , addr)
            #while True:

            data = b""
            payload_size = struct.calcsize(">L")
            print("payload_size: {}".format(payload_size))
            while len(data) < payload_size:
                #print(f"recv: {len(data)}")
                data += conn.recv(4096)
                if not data:
                    continue
                    
            print(f'done recv !! {len(data)}')
            

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('>L', packed_msg_size)[0]
            while len(data) < msg_size:
                data += conn.recv(4096)
            print(f'recv: {len(data)}')
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            plt.imsave('original_rev_server.png',frame)

            frame = seg.segmentation(frame)

            plt.imsave('mask_seg.png',frame)
            print(frame.shape)

            #### PROCESSING #####
            #image_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #cv2.imshow('server side',frame)
            #cv2.waitKey(1) & 0xFF 


            result, image = cv2.imencode('.png',frame, encode_param)
            #result, image = cv2.imencode('.jpg',frame, encode_param)
            data = pickle.dumps(image, 0)
            size = len(data)

            
            try:
                conn.sendall(struct.pack(">L",size) + data)
            except ConnectionResetError as e :
                print(e)
                break
            
                