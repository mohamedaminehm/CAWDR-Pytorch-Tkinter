from tkinter import Toplevel, Button, RIGHT
import numpy as np
import cv2
from classification import ClassificationClass
from segmentation import SegmentationClass



import os
import sys
import random
import warnings
import cv2
import gc




class SegmFrame(Toplevel):

    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.master.after_seg = self.master.processed_image
        self.original_image = self.master.processed_image
        self.segmented_image = self.master.segmented_image
        self.classified_image = self.master.classified_image
        self.histogramme_pred = self.master.histogramme_pred
        self.filtered_image = None

        self.class_class = ClassificationClass(self.original_image)
        self.class_seg = SegmentationClass(self.original_image)


        self.binary_button = Button(master=self, text="Segmentation")
        self.fast_binary_button = Button(master=self, text="fast binary")

        self.multi_class_button = Button(master=self, text="multi_class")

        self.classification_button = Button(master=self, text="clf with spatial feats")
        self.classification_d_button = Button(master=self, text="clf with densnet feats")
        self.classification_s_d_button = Button(master=self, text="Classification")


        self.cancel_button = Button(master=self, text="Cancel")
        self.apply_button = Button(master=self, text="Apply")

        self.binary_button.bind("<ButtonRelease>", self.binary_button_released)
        

        
        self.classification_s_d_button.bind("<ButtonRelease>", self.classification_s_d_button_released)

        self.fast_binary_button.bind("<ButtonRelease>", self.fast_binary_button_released)
        self.multi_class_button.bind("<ButtonRelease>", self.multi_class_button_released)
        self.classification_button.bind("<ButtonRelease>", self.classification_button_released)
        self.classification_d_button.bind("<ButtonRelease>", self.classification_d_button_released)



        self.apply_button.bind("<ButtonRelease>", self.apply_button_released)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)

        self.binary_button.pack()
        #self.fast_binary_button.pack()

        #self.multi_class_button.pack()
        #self.classification_button.pack()
        #self.classification_d_button.pack()
        self.classification_s_d_button.pack()



        self.cancel_button.pack(side=RIGHT)
        self.apply_button.pack()


    def fast_binary_button_released(self, event):
        self.fast_binary()
        self.show_image()
    def binary_button_released(self, event):
        self.binary()
        self.show_image()

    def multi_class_button_released(self, event):
        self.multi()
        self.show_image()

    
    
    def classification_button_released(self,event):
        self.classify()
        self.show_image()
    
    def classification_d_button_released(self,event):
        self.classify(flag="D")
        self.show_image()

    def classification_s_d_button_released(self,event):
        self.classify(flag="F")
        self.show_image()

    def apply_button_released(self, event):
        self.master.processed_image = self.filtered_image
        self.master.segmented_image = self.segmented_image
        self.master.classified_image = self.classified_image 
        self.master.histogramme_pred =  self.histogramme_pred
        self.show_image()
        self.close()

    def cancel_button_released(self, event):
        self.master.image_viewer.show_image()
        self.close()

    def show_image(self):
        self.master.image_viewer.show_image(img=self.filtered_image)


    def binary(self):
        if self.segmented_image is None:
            resultat_  = self.class_seg.segmentation()
            self.segmented_image = resultat_
            #cv2.imwrite("mask_morph.png", self.segmented_image )
        self.filtered_image = self.segmented_image

    
    def fast_binary(self):
        if self.segmented_image is None:
            resultat_  = self.class_seg.segmentation2()
            self.segmented_image = resultat_
        self.filtered_image = self.segmented_image
        

    
    
    def multi(self):
        self.filtered_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)

    
    def classify(self, flag=None):
        if self.classified_image is None:
            if self.segmented_image is None:
                resultat_  = self.class_seg.segmentation()
                self.segmented_image = resultat_
            if flag=="D":
                self.classified_image = self.class_class.classification_d(self.segmented_image)
            elif flag=="F":
                self.classified_image, self.histogramme_pred = self.class_class.classification_gabor_2stage(self.segmented_image)
            else:
                self.classified_image = self.class_class.classification(self.segmented_image)
        self.filtered_image = self.classified_image
        

        



    
    

    def close(self):
        self.destroy()