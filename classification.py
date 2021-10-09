
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure 
from skimage.measure import shannon_entropy
from scipy import ndimage as ndi
from collections import Counter

from skimage.filters import gabor_kernel
import mahotas

import pickle

types = ['','SL','CL','PO','LP','LF','WH']
cls = [(0,0,0), (0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255)] 



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










class ClassificationClass():
    def __init__(self,original_image):
        self.original_image = original_image



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
        
        return imm , histo
    
    







def scalar_attributes_list(im_props):
    
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
