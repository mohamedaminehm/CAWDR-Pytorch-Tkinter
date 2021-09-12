from tkinter import Frame, Button, LEFT
from tkinter import filedialog
from tkinter.constants import RIGHT
from filterFrame import FilterFrame
from adjustFrame import AdjustFrame
from segmentationFrame import SegmFrame

import cv2
from PIL import Image
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


class EditBar(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        self.new_button = Button(self, text="New")
        self.save_button = Button(self, text="Save")
        self.save_as_button = Button(self, text="Save As")
        self.draw_button = Button(self, text="Draw")
        self.crop_button = Button(self, text="Crop")
        self.filter_button = Button(self, text="Filter")
        self.adjust_button = Button(self, text="Adjust")
        self.clear_button = Button(self, text="Clear")

        self.zoom_button = Button(self, text="Zoom")
        self.segmentation_button = Button(self, text="Intelligent detection")

        self.new_button.bind("<ButtonRelease>", self.new_button_released)
        self.save_as_button.bind("<ButtonRelease>", self.save_as_button_released)
        self.crop_button.bind("<ButtonRelease>", self.crop_button_released)
        self.adjust_button.bind("<ButtonRelease>", self.adjust_button_released)
        self.zoom_button.bind("<ButtonRelease>", self.zoom_button_released)
        self.segmentation_button.bind("<ButtonRelease>", self.segmentation_button_released)
        self.clear_button.bind("<ButtonRelease>", self.clear_button_released)


        self.save_button.bind("<ButtonRelease>", self.save_button_released)
        self.draw_button.bind("<ButtonRelease>", self.draw_button_released)
        self.filter_button.bind("<ButtonRelease>", self.filter_button_released)



        self.new_button.pack(side=LEFT)
        self.save_button.pack(side=LEFT)
        self.save_as_button.pack(side=LEFT)
        #self.draw_button.pack(side=LEFT)
        self.crop_button.pack(side=LEFT)
        #self.filter_button.pack(side=LEFT)
        self.adjust_button.pack(side=LEFT)
        

        self.zoom_button.pack(side=RIGHT)
        self.segmentation_button.pack(side=RIGHT)
        self.clear_button.pack(side=RIGHT)


    def new_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.new_button:
            if self.master.is_draw_state:
                self.master.image_viewer.deactivate_draw()
            if self.master.is_crop_state:
                self.master.image_viewer.deactivate_crop()

            filename = filedialog.askopenfilename()
            print(filename)

            if filename.split('.')[-1] == 'dcm':
                img_data = read_xray(filename)
                image = enhance_contrast(image_matrix=img_data,bins=256)
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                print(np.max(image))

            else:
                image = cv2.imread(filename)

            if image is not None:
                self.master.filename = filename
                self.master.original_image = image.copy()
                self.master.processed_image = image.copy()
                self.master.segmented_image = None
                self.master.classified_image = None
                self.master.image_viewer.show_image()
                self.master.is_image_selected = True

    def save_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()

                save_image = self.master.processed_image
                image_filename = self.master.filename
                cv2.imwrite(image_filename, save_image)
    
    def save_as_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_as_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                original_file_type = self.master.filename.split('.')[-1]
                filename = filedialog.asksaveasfilename()
                filename_orig = filename + ".png"#+ original_file_type 
                
                filename_seg = filename + "_seg.png" #+ original_file_type
                filename_clas = filename + "_class.png"# + original_file_type
                filename_hist = filename + "_hist.png"# + original_file_type
                filename_slice = filename + "_slice.png"# + original_file_type





                save_image = self.master.original_image
                cv2.imwrite(filename_orig, save_image)
                if self.master.after_seg is not None:
                    cv2.imwrite(filename_slice,self.master.after_seg)
                if self.master.segmented_image is not None:
                    cv2.imwrite(filename_seg,cv2.resize(self.master.segmented_image,(self.master.after_seg.shape[1],self.master.after_seg.shape[0])))
                if self.master.classified_image is not None:
                    cv2.imwrite(filename_clas,self.master.classified_image)
                if self.master.histogramme_pred is not None:
                    cv2.imwrite(filename_hist,self.master.histogramme_pred)

                self.master.filename = filename

    def draw_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.draw_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                else:
                    self.master.image_viewer.activate_draw()



    def crop_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.crop_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.n.deactivate_crop()
                else:
                    self.master.image_viewer.activate_crop()
        

    def filter_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.filter_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                self.master.filter_frame = FilterFrame(master=self.master)
                self.master.filter_frame.grab_set()


    def adjust_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.adjust_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                self.master.adjust_frame = AdjustFrame(master=self.master)
                self.master.adjust_frame.grab_set()


    def clear_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.clear_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.is_zoom_state:
                    self.master.image_viewer.deactivate_zoom()


                self.master.processed_image = self.master.original_image.copy()
                self.master.image_viewer.show_image()

    def zoom_button_released(self,event):
        if self.winfo_containing(event.x_root, event.y_root) == self.zoom_button:
            if self.master.is_image_selected:
                if self.master.is_draw_state:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.n.deactivate_crop()
                else:
                    self.master.image_viewer.activate_zoom()

    def segmentation_button_released(self,event):
        if self.winfo_containing(event.x_root, event.y_root) == self.segmentation_button:
            if self.master.is_image_selected: 
                if self.master.is_image_selected:
                    self.master.image_viewer.deactivate_draw()
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                if self.master.segmented_image is not None:
                    self.master.after_seg = None
                    self.master.segmented_image = None
                    self.master.classified_image = None
                
                self.master.segmentationFrame = SegmFrame(master=self.master)
                self.master.segmentationFrame.grab_set()






def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im



def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return np.uint8(image_eq)




                        


        
        



