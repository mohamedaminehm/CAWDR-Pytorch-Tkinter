
from tqdm import tqdm
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import scipy as sp
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, concatenate_images
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

types = ['','SL','CL','PO','LP','LF','WH']
cls = [(0,0,0), (0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255)] 

model_name = 'densenet121'
layer_name = 'features'

svm_filename = "svm_model/sp_svm/model.pkl"
pca_filename = "svm_model/sp_svm/pca.pkl"
scaler_filename = "svm_model/sp_svm/scaler.pkl"

svm_filename_d = "svm_model/densnet_svm/densnet_pca_svm_model.pkl"
pca_filename_d = "svm_model/densnet_svm/pca.pkl"
scaler_filename_d = "svm_model/densnet_svm/scaler.pkl"

svm_filename_f = "svm_model/sp_dnet_svm/dnet_sp_svm_model.pkl"
pca_filename_f = "svm_model/sp_dnet_svm/pca.pkl"
scaler_filename_f = "svm_model/sp_dnet_svm/scaler.pkl"


# 2 STAGE CLASSIFIER
svm_filename_1s = "svm_model/pipeline_2stage/1stage_zer_hu_geometry_gan/model.pkl"
pca_filename_1s = "svm_model/pipeline_2stage/1stage_zer_hu_geometry_gan/pca.pkl"
scaler_filename_1s = "svm_model/pipeline_2stage/1stage_zer_hu_geometry_gan/scaler.pkl"
pipeline_1s = "svm_model/pipeline_2stage/1stage_zer_hu_geometry_gan/pipeline1.pkl"


svm_filename_2s_c = "svm_model/pipeline_2stage/2stage_zer_hu_cirvular_gabor_geo/model.pkl"
pca_filename_2s_c = "svm_model/pipeline_2stage/2stage_zer_hu_cirvular_gabor_geo/pca.pkl"
scaler_filename_2s_c = "svm_model/pipeline_2stage/2stage_zer_hu_cirvular_gabor_geo/scaler.pkl"
pipeline_2s_c = "svm_model/pipeline_2stage/2stage_zer_hu_cirvular_gabor_geo/pipeline2c.pkl"


svm_filename_2s_l = "svm_model/pipeline_2stage/2stage_linear_gabor_geo_gan/model.pkl"
pca_filename_2s_l = "svm_model/pipeline_2stage/2stage_linear_gabor_geo_gan/pca.pkl"
scaler_filename_2s_l = "svm_model/pipeline_2stage/2stage_linear_gabor_geo_gan/scaler.pkl"
pipeline_2s_l = "svm_model/pipeline_2stage/2stage_linear_gabor_geo_gan/pipeline2L.pkl"




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

class Dataset_feats(torch.utils.data.Dataset):
    
    def __init__(self, data, size):
        self.data = data
        self.size = size
        self.img_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        image = self.data[idx]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_to_square(image, self.size)
        image = pad(image, self.size, self.size)
        
        #tensor = image_to_tensor(image, normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
            
        return self.img_transform(image / 255)

'''model = smp.Unet("resnet50")#, encoder_weights="imagenet", activation=None)
FREEZE_RESNET = True   # if UNET_RESNET34ImgNet
if FREEZE_RESNET == False:
    for name, p in model.named_parameters():
        if "encoder" in name:
            p.requires_grad = False
model_path = Path('seg_model/model_7.pt')
state = torch.load(str(model_path),map_location=torch.device('cpu'))
epoch = state['epoch']
step = state['step']
model.load_state_dict(state['model'])'''



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


class ClassificationClass():
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


        self.get_model = getattr(torchvision.models, model_name)
        #feats_model_path = "feats_extraction_model/densenet121-a639ec97.pth"
        #self.feats_model = torch.load(str(feats_model_path),map_location=torch.device(self.device))


        
        
        

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


    def get_dl_feats(self,data,size):
        feats_model =self.get_model(pretrained=True).double().to(self.device)
        feats_model.eval()
        #self.feats_model = self.feats_model.double().to(self.device)
        #self.feats_model.eval()
        model = tx.Extractor(feats_model, [layer_name])

        features = []
        '''def hook(module, input, output):
            N,C,H,W = output.shape
            output = output.reshape(N,C,-1)
            features.append(output.mean(dim=2).cpu().detach().numpy())
        handle = model._modules.g*
        et(layer_name).register_forward_hook(hook)'''

        dataset = Dataset_feats(data, size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for i_batch, inputs in enumerate(loader):
            _,feats = model(inputs.to(self.device))
            features.append(feats[layer_name].mean(dim=2).mean(dim=2).cpu().detach().numpy())

        features = np.concatenate(features)
        features = pd.DataFrame(features)
        #handle.remove()
        del model
        #gc.collect()

        return features 

    
    def classification(self,mask_resultat):

        print(np.unique(mask_resultat))
        print("instance segmentation and cropping start ..")
        mask_resultat = cv2.resize(mask_resultat, (self.original_image.shape[1],self.original_image.shape[0]))

        labelss = measure.label(mask_resultat,connectivity=2)
        #orig_image1 = cv2.resize(self.original_image, (mask_resultat.shape[1],mask_resultat.shape[0]))
        orig_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2GRAY)
        props = measure.regionprops(labelss , orig_image)
        print(props)
        size = 224
        bboxes = []
        temp = []
        for prop in props:
            if prop.area > 50:
                bboxes.append(prop.bbox)
                temp.append(prop)
        props = temp
        del temp
        print(len(bboxes))
        data = get_cropped_def(self.original_image,bboxes)
        #print(len(data))
        print("instance segmentation and cropping was finished ..")
        print("features extraction start ..")

        #get_model = getattr(torchvision.models, model_name)
        
        #feats = self.get_dl_feats(data,size)
        #print("resnet features ..")
        s_p_feats,_ = get_sp_feats(props,data)
        print("spatial parametres ..")

        #df_ = pd.concat([feats, s_p_feats],axis=1)
        print("svm classification start ..")

        predictions = def_predict(s_p_feats,scaler_filename,pca_filename,svm_filename)
        predictions = [int(e) for e in predictions]
        print("preparing result :) ..")
        print(predictions)
        k=0
        imm = self.original_image.copy()
        for bbox in bboxes:
            if (props[k].area > 50):
                lb = f'{types[predictions[k]]}:{props[k].area}'
                labelSize=cv2.getTextSize(lb,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
                _x1 = bbox[1]
                _y1 = bbox[0]#+int(labelSize[0][1]/2)
                _x2 =_x1+labelSize[0][0]
                _y2 = bbox[3]-int(labelSize[0][1])
                #cv2.rectangle(imm,(_x1,_y1),(_x2,_y2),cls[predictions[k]],cv2.FILLED)
                cv2.putText(imm,lb,(bbox[1],bbox[0]),cv2.FONT_HERSHEY_COMPLEX,0.5,cls[predictions[k]],1)
                cv2.rectangle(imm, (bbox[1], bbox[0]), (bbox[3], bbox[2]), cls[predictions[k]], 2)
                #cv2.putText(imm, lb, (bbox[1], bbox[3]+5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX ,1,cls[predictions[k]],1,cv2.LINE_AA)
            k+=1
        
        return imm
    def classification_gabor_2stage(self,mask_resultat):

        print(np.unique(mask_resultat))
        print("instance segmentation and cropping start ..")
        mask_resultat = cv2.resize(mask_resultat, (self.original_image.shape[1],self.original_image.shape[0]))

        labelss = measure.label(mask_resultat,connectivity=2)
        #orig_image1 = cv2.resize(self.original_image, (mask_resultat.shape[1],mask_resultat.shape[0]))
        orig_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2GRAY)
        props = measure.regionprops(labelss , orig_image)
        size = 224
        bboxes = []
        temp = []
        for prop in props:
            if prop.area > 30:
                bboxes.append(prop.bbox)
                temp.append(prop)
        props = temp
        del temp
        print(len(bboxes))
        data = get_cropped_def(self.original_image,bboxes)
        data_masks = get_cropped_def(mask_resultat,bboxes)

        print("instance segmentation and cropping was finished ..")
        print("features extraction start ..")

        anomaly = False

        predictions = []
        if len(data) > 0 :
            s_p_feats, s_p_feats_new = get_sp_feats(props,data)
            hu_z_feats = cal_hu_z_feats(data_masks)
            gabor_feats = cal_gabor_feats(data)
        
            print("spatial parametres ..")

            df_first_stage = pd.DataFrame() 
            df_first_stage = df_first_stage.append(pd.concat([hu_z_feats,gabor_feats,s_p_feats_new],axis=1))


            df_first_stage.to_csv('file1.csv')
            #df_ = pd.concat([feats, s_p_feats],axis=1)
            print("svm classification start ..")

            predictions,prediction_proba = def_predict_pipeline(df_first_stage.values,pipeline_1s)

            predictions = [int(e) for e in predictions]

            df_first_stage['bbindex'] = pd.Series([i for i in range(len(bboxes))]).values
            df_first_stage['fsp'] = pd.Series(predictions).values

            df_anb = df_first_stage[df_first_stage['fsp'] == 0].iloc[:,:-1]
            df_circular = df_first_stage[df_first_stage['fsp'] == 1].iloc[:,:-1]
            df_linear = df_first_stage[df_first_stage['fsp'] == 2].iloc[:,32:-1]




            if not df_circular.empty and df_linear.empty :
                predictions_cir,prediction_proba_cir = def_predict_pipeline(df_circular.iloc[:,:-1].values,pipeline_2s_c)
                predictions_cir = [int(e) for e in predictions_cir]
                bboxes_cir_ind = df_circular.loc[:,['bbindex']].values
                j=0
                for i in range(len(predictions)):
                    if i == int(bboxes_cir_ind[j]):
                        predictions[i] = predictions_cir[j] + 1
                        if j < (bboxes_cir_ind.shape[0] - 1):
                            j+=1
            elif df_circular.empty and not df_linear.empty:
                predictions_lin,prediction_proba_lin = def_predict_pipeline(df_linear.iloc[:,:-1].values,pipeline_2s_l)
                predictions_lin = [int(e) for e in predictions_lin]
                bboxes_lin_ind = df_linear.loc[:,['bbindex']].values

                j=0
                for i in range(len(predictions)):
                    if i == int(bboxes_lin_ind[j]):
                        predictions[i] = predictions_lin[j] + 4
                        if j < (bboxes_lin_ind.shape[0] - 1):
                            j+=1
            

            else:
                predictions_cir,prediction_proba_cir = def_predict_pipeline(df_circular.iloc[:,:-1].values,pipeline_2s_c)
                predictions_cir = [int(e) for e in predictions_cir]
                bboxes_cir_ind = df_circular.loc[:,['bbindex']].values

                print("######lin shape ####")
                print(df_linear.iloc[:,:-1].shape)

                predictions_lin,prediction_proba_lin = def_predict_pipeline(df_linear.iloc[:,:-1].values,pipeline_2s_l)
                predictions_lin = [int(e) for e in predictions_lin]
                bboxes_lin_ind = df_linear.loc[:,['bbindex']].values

                j=0
                k=0
                for i in range(len(predictions)) :
                    if i == int(bboxes_cir_ind[j]):
                        predictions[i] = predictions_cir[j] + 1
                        if j < (bboxes_cir_ind.shape[0] - 1):
                            j+=1
                    elif i ==  int(bboxes_lin_ind[k]):
                        predictions[i] = predictions_lin[k] + 4
                        if k < (bboxes_lin_ind.shape[0] -1) :
                            k+=1
                    else:
                        continue
        else : 
            anomaly = True


        
        types1 = ['AMB', 'SL','P','WH','CR','LP','LF']
        cls1 = [(0,0,0),(0,0,255),(255,0,0),(255,255,0),(0,255,0),(0,255,255),(255,0,255)] 


        
        print("preparing result :) ..")
        print(predictions)

        histo = hist_img(predictions)


        
        k=0
        imm = self.original_image.copy()
        if  anomaly:
            cv2.putText(imm,'Pas de defaut -> piece valide',(80,80),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0))
        
        for bbox in bboxes:
            if (props[k].area > 10):
                lb = f'{types1[predictions[k]]}'#':{props[k].area}'
                labelSize=cv2.getTextSize(lb,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
                _x1 = bbox[1]
                _y1 = bbox[0]#+int(labelSize[0][1]/2)
                _x2 =_x1+labelSize[0][0]
                _y2 = bbox[3]-int(labelSize[0][1])
                #cv2.rectangle(imm,(_x1,_y1),(_x2,_y2),cls[predictions[k]],cv2.FILLED)
                cv2.putText(imm,lb,(bbox[1],bbox[0]),cv2.FONT_HERSHEY_COMPLEX,0.5,cls1[predictions[k]],1)
                cv2.rectangle(imm, (bbox[1], bbox[0]), (bbox[3], bbox[2]), cls1[predictions[k]], 2)
                #cv2.putText(imm, lb, (bbox[1], bbox[3]+5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX ,1,cls[predictions[k]],1,cv2.LINE_AA)
            k+=1
        
        return cv2.cvtColor(imm,cv2.COLOR_BGR2RGB) , histo
    
    def classification_d(self,mask_resultat):


        print(mask_resultat.shape)
        print("instance segmentation and cropping start ..")
        mask_resultat = cv2.resize(mask_resultat, (self.original_image.shape[1],self.original_image.shape[0]))

        labelss = measure.label(mask_resultat ,connectivity=2)
        #orig_image1 = cv2.resize(self.original_image, (mask_resultat.shape[1],mask_resultat.shape[0]))
        orig_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2GRAY)
        props = measure.regionprops(labelss , orig_image)
        size = 224
        bboxes = []
        temp = []
        for prop in props:
            if prop.area > 50:
                bboxes.append(prop.bbox)
                temp.append(prop)
        props = temp
        del temp
        print(len(bboxes))
        data = get_cropped_def(self.original_image,bboxes)
        print("instance segmentation and cropping was finished ..")
        print("features extraction start ..")

        #get_model = getattr(torchvision.models, model_name)
        
        feats = self.get_dl_feats(data,size)
        #print("resnet features ..")
        #s_p_feats = get_sp_feats(props)
        #print("spatial parametres ..")

        #df_ = pd.concat([feats, s_p_feats],axis=1)
        print("svm classification start ..")

        predictions = def_predict(feats,scaler_filename_d,pca_filename_d,svm_filename_d)
        predictions = [int(e) for e in predictions]
        print("preparing result :) ..")

        k=0
        imm = self.original_image.copy()
        for bbox in bboxes:
            if (props[k].area > 200):
                lb = f'{types[predictions[k]]}:{props[k].area}'
                labelSize=cv2.getTextSize(lb,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
                _x1 = bbox[1]
                _y1 = bbox[0]#+int(labelSize[0][1]/2)
                _x2 =_x1+labelSize[0][0]
                _y2 = bbox[3]-int(labelSize[0][1])
                #cv2.rectangle(imm,(_x1,_y1),(_x2,_y2),cls[predictions[k]],cv2.FILLED)
                cv2.putText(imm,lb,(bbox[1],bbox[0]),cv2.FONT_HERSHEY_COMPLEX,1,cls[predictions[k]],1, cv2.LINE_AA)
                cv2.rectangle(imm, (bbox[1], bbox[0]), (bbox[3], bbox[2]), cls[predictions[k]], 2)
                #cv2.putText(imm, lb, (bbox[1], bbox[3]+5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX ,1,cls[predictions[k]],1,cv2.LINE_AA)
            k+=1
        
        return imm
    def classification_f(self,mask_resultat):


        print(mask_resultat.shape)
        print("instance segmentation and cropping start ..")
        mask_resultat = cv2.resize(mask_resultat, (self.original_image.shape[1],self.original_image.shape[0]))

        labelss = measure.label(mask_resultat,connectivity=2 )
        #orig_image1 = cv2.resize(self.original_image, (mask_resultat.shape[1],mask_resultat.shape[0]))
        orig_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2GRAY)
        props = measure.regionprops(labelss , orig_image)
        size = 224
        bboxes = []
        temp = []
        for prop in props:
            if prop.area > 200:
                bboxes.append(prop.bbox)
                temp.append(prop)
        props = temp
        del temp
        
        print(len(bboxes))
        data = get_cropped_def(self.original_image,bboxes)
        print("instance segmentation and cropping was finished ..")
        print("features extraction start ..")

        #get_model = getattr(torchvision.models, model_name)
        
        feats = self.get_dl_feats(data,size)
        print("deep features was extracted")
        #print("resnet features ..")
        s_p_feats = get_sp_feats(props)
        print("geometric features was extracted")

        #print("spatial parametres ..")

        df_ = pd.concat([feats, s_p_feats],axis=1)
        print("svm classification start ..")

        predictions = def_predict(df_,scaler_filename_f,pca_filename_f,svm_filename_f)
        predictions = [int(e) for e in predictions]
        print("preparing result :) ..")

        k=0
        imm = self.original_image.copy()
        for bbox in bboxes:
            if (props[k].area > 200):
                lb = f'{types[predictions[k]]}:{props[k].area}'
                labelSize=cv2.getTextSize(lb,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
                _x1 = bbox[1]
                _y1 = bbox[0]#+int(labelSize[0][1]/2)
                _x2 =_x1+labelSize[0][0]
                _y2 = bbox[3]-int(labelSize[0][1])
                #cv2.rectangle(imm,(_x1,_y1),(_x2,_y2),cls[predictions[k]],cv2.FILLED)
                cv2.putText(imm,lb,(bbox[1],bbox[0]),cv2.FONT_HERSHEY_COMPLEX,1,cls[predictions[k]],1,cv2.LINE_AA)
                cv2.rectangle(imm, (bbox[1], bbox[0]), (bbox[3], bbox[2]), cls[predictions[k]], 2)
                #cv2.putText(imm, lb, (bbox[1], bbox[3]+5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX ,1,cls[predictions[k]],1,cv2.LINE_AA)
            k+=1
        
        return imm







def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """
    
    attributes_list = []
    
    for i, test_attribute in enumerate(dir(im_props[0])):
        
        #Attribute should not start with _ and cannot return an array
        #does not yet return tuples
        if test_attribute[:1] != '_' and not\
                isinstance(getattr(im_props[0], test_attribute), np.ndarray):                
            attributes_list += [test_attribute]
            
    return attributes_list


def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """

    attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]
        
        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)





kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


def gabor_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = shannon_entropy(filtered)
    return feats

def cal_gabor_feats(data):
    cl = [f'c{i}' for i in range(32)]
    gabor_data = pd.DataFrame(columns=cl)

    for i in range(len(data)):
        gray = cv2.cvtColor(data[i],cv2.COLOR_BGR2GRAY)
        feats = gabor_feats(gray,kernels)
        val = feats.reshape(1,-1).tolist()[0]
        gabor_data = gabor_data.append(pd.Series(val, index=cl), ignore_index=True) 
    
    return gabor_data




# prepare filter bank kernels




def cal_hu_z_feats(imgs_mask):
    h_colum = ['h1','h2','h3','h4','h5','h6','h7']
    for i in range(1,26):
        h_colum.append(f'z{i}')
    hu_z_feats = pd.DataFrame(columns=h_colum)
    for i in range(len(imgs_mask)):
        moments = cv2.moments(imgs_mask[i])
        huMoments = cv2.HuMoments(moments)
        feats = huMoments.reshape(-1).tolist()

        z = mahotas.features.zernike_moments(imgs_mask[i], min(imgs_mask[i].shape)).tolist()
        [feats.append(e) for e in z ]

        hu_z_feats = hu_z_feats.append(pd.Series(feats, index=h_colum), ignore_index=True)
    return hu_z_feats 




def get_sp_feats(props,defects_):
    data = regionprops_to_df(props).values

    s_colum = ['area','centroid','convex_area','eccentricity','equivalent_diameter','euler_number','extent','filled_area','inertia_tensor_eigvals1','inertia_tensor_eigvals2','local_centroidx','local_centroidy','major_axis_length','mean_intensity','minor_axis_length','orientation','perimeter','solidity']
    s_p_feats = pd.DataFrame(columns=s_colum)

    for i in range(len(data)):
        feats = data[i]
        feats = [e for j, e in enumerate(feats) if j not in [1,2,11,14,16,20,22]]
        feats[1] = feats[1][0]
        feats.insert(9,feats[8][1])
        feats[8] = feats[8][0]
        feats.insert(11,feats[10][1])
        feats[10] = feats[10][0]
        s_p_feats = s_p_feats.append(pd.Series(feats, index=s_colum), ignore_index=True) 

    
    s_colum_new = ['Compactness','Elongation','Rectangularity','Anisometry','Lengthening_index','Dev_ind_inscrib_circle','Rowndness','Eccentricity','Solidity']  #,'Eccentricity','Solidity'
    s_p_feats_new = pd.DataFrame(columns=s_colum_new)

    for index, row in s_p_feats.iterrows():
        new_row = []
        Compactness = 4 * np.pi * row['area'] / (row['perimeter'] **2 + 1e-7 ) 
        Elongation = defects_[index].shape[0] /  defects_[index].shape[1] # aspect ratio
        Rectangularity = row['area'] / ( defects_[index].shape[0] * defects_[index].shape[1] + 1e-7) # extent
        Eccentricity = row['eccentricity']
        Solidity = row['solidity']
        Anisometry = row['major_axis_length'] / (row ['minor_axis_length'] + 1e-7)
        Lengthening_index = np.pi * row['equivalent_diameter'] **2 / (4 * row['area']+ 1e-7)
        Dev_ind_inscrib_circle = (1 - (np.pi * row['major_axis_length'] ** 2 )) /(4 * row['area']+ 1e-7)
        Rowndness = 4 * np.pi * row['area'] / (row['perimeter'] ** 2+ 1e-7)

        new_row = [Compactness,Elongation,Rectangularity,Anisometry,Lengthening_index,Dev_ind_inscrib_circle,Rowndness,Eccentricity,Solidity]
        s_p_feats_new = s_p_feats_new.append(pd.Series(new_row, index=s_colum_new), ignore_index=True) 

    return s_p_feats , s_p_feats_new



def get_cropped_def(image,bboxes):
    data =[]
    for bx in bboxes:
        if (bx[0] < 10) or (bx[2] +10 > image.shape[1]):
            data.append(image[bx[0]:bx[2],bx[1]-10:bx[3]+10])
        elif (bx[1] < 10) or (bx[3]+10 > image.shape[0]):
            data.append(image[bx[0]-10:bx[2]+10,bx[1]:bx[3]])
        else:
            data.append(image[bx[0]-10:bx[2]+10,bx[1]-10:bx[3]+ 10])
    return data



def def_predict(feats,scaler_file,pca_file,svm_file):
    with open(svm_file, 'rb') as file:
        svm = pickle.load(file)
    with open(scaler_file, 'rb') as file:
        scaler = pickle.load(file)
    with open(pca_file, 'rb') as file:
        pca = pickle.load(file)
    feats = scaler.transform(feats)
    feats = pca.transform(feats)
    prediction = svm.predict(feats)
    del svm
    del scaler
    del pca
    return prediction


def def_predict_pipeline(feats,pipeline_file):
    with open(pipeline_file, 'rb') as file:
        pipe = pickle.load(file)


    p = np.array(pipe.decision_function(feats))
    prob = np.exp(p)/np.sum(np.exp(p),axis=1, keepdims=True) 
    classes = pipe.predict(feats)

    
    
    del pipe
    return classes,prob





def resize_to_square(image, size):
    h, w, d = image.shape
    ratio = size / max(h, w)
    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)
    return resized_image



def pad(image, min_height, min_width):
    h,w,d = image.shape

    if h < min_height:
        h_pad_top = int((min_height - h) / 2.0)
        h_pad_bottom = min_height - h - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if w < min_width:
        w_pad_left = int((min_width - w) / 2.0)
        w_pad_right = min_width - w - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))


def hist_img(pred):
    type_list = ['Ambiguous','Slag','Porosity','Worm hole','Crack','L.penetration','L.fusion'] 
    count = [0,0,0,0,0,0,0]
    for k,v in Counter(pred).items():
        count[k] = v

    fig = plt.figure(figsize=(20,10))
    plt.bar(type_list, count)
    plt.ylabel('Distrubution des defauts')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape((fig.canvas.get_width_height()[::-1] + (3,)))

    return img
