from tkinter import Frame, Canvas, CENTER, ROUND, S, SW,NE
from typing import Text
from PIL import Image, ImageTk
import cv2


class ImageViewer(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master, bg="gray", width=1200, height=400)

        self.shown_image = None
        self.x = 0
        self.y = 0
        self.crop_start_x = 0
        self.crop_start_y = 0
        self.crop_end_x = 0
        self.crop_end_y = 0
        self.draw_ids = list()
        self.rectangle_id = 0
        self.ratio = 0

        self.zoom_start_x = 0
        self.zoom_start_y = 0
        self.zoom_end_x = 0
        self.zoom_end_y = 0
        self.rectangle_id_zoom = 0
        self.delta = 0.9
        self.imscale = 1.0
        self.imageid = None
        self.image = None
        self.text = None

        self.canvas = Canvas(self, bg="gray", width=1000, height=500)
        self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)




    def show_image(self, img=None):
        self.clear_canvas()



        if img is None:
            image = self.master.processed_image.copy()
        else:
            image = img

        if len(image.shape) == 3 :
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image.shape
        else:
            height, width = image.shape
        ratio = height / width

        new_width = width
        new_height = height

        if height > self.winfo_height() or width > self.winfo_width():
            if ratio < 1:
                new_width = self.winfo_width()
                new_height = int(new_width * ratio)
            else:
                new_height = self.winfo_height()
                new_width = int(new_height * (width / height))
        

        self.shown_image = cv2.resize(image, (new_width, new_height))
        self.shown_image = ImageTk.PhotoImage(Image.fromarray(self.shown_image))

        self.ratio = height / new_height

        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width / 2, new_height / 2, anchor=CENTER, image=self.shown_image)
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    


    ##ZOOM
    def show_perm_image(self, img = None):
        ''' Show image on the Canvas '''
        if self.imageid:
            self.canvas.delete('all')
            self.imageid = None
            self.canvas.imagetk = None  # delete previous image from the canvas
        self.image = self.master.processed_image.copy()
        width, height,_ = self.image.shape
        
        new_size =  int(self.imscale * height) , int(self.imscale * width)
        imagetk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.image,(new_size))))
        # Use self.text object to set proper coordinates
        self.imageid = self.canvas.create_image(int(self.imscale * height)/2,int(self.imscale * width) /2 , anchor=CENTER, image=imagetk)
        self.canvas.lower(self.imageid)  # set it into background
        self.canvas.imagetk = imagetk


    def activate_draw(self):
        self.canvas.bind("<ButtonPress>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.master.is_draw_state = True

    def activate_crop(self):
        self.canvas.bind("<ButtonPress>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.crop)
        self.canvas.bind("<ButtonRelease>", self.end_crop)

        self.master.is_crop_state = True

    ## ZOOM
    def activate_zoom(self):
        self.canvas.bind("<ButtonPress>", self.start_zoom)
        self.canvas.bind("<B1-Motion>", self.zoom)
        #self.canvas.bind("<ButtonRelease>", self.end_zoom)
        self.canvas.bind("<MouseWheel>",self.zoomer)

        self.master.is_zoom_state = True


        

    def deactivate_draw(self):
        self.canvas.unbind("<ButtonPress>")
        self.canvas.unbind("<B1-Motion>")

        self.master.is_draw_state = False

    def deactivate_crop(self):
        self.canvas.unbind("<ButtonPress>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease>")
        self.master.is_crop_state = False

    ##ZOOM
    def deactivate_zoom(self):
        self.canvas.unbind("<ButtonPress>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease>")
        self.master.is_zoom_state = False


    def start_draw(self, event):
        self.x = event.x
        self.y = event.y
    
    def draw(self, event):
        self.draw_ids.append(self.canvas.create_line(self.x, self.y, event.x, event.y, width=2,
                                                    fill="red", capstyle=ROUND, smooth=True))

        cv2.line(self.master.processed_image, (int(self.x * self.ratio), int(self.y * self.ratio)),
                (int(event.x * self.ratio), int(event.y * self.ratio)),
                (0, 0, 255), thickness=int(self.ratio * 2),
                lineType=8)

        self.x = event.x
        self.y = event.y

    def start_crop(self, event):
        self.crop_start_x = event.x
        self.crop_start_y = event.y

    def crop(self, event):
        if self.rectangle_id:
            self.canvas.delete(self.rectangle_id)

        self.crop_end_x = event.x
        self.crop_end_y = event.y

        self.rectangle_id = self.canvas.create_rectangle(self.crop_start_x, self.crop_start_y,
                                                        self.crop_end_x, self.crop_end_y, width=1)


    def end_crop(self, event):
        if self.crop_start_x <= self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x > self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x <= self.crop_end_x and self.crop_start_y > self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)
        else:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)

        x = slice(start_x, end_x, 1)
        y = slice(start_y, end_y, 1)

        self.master.processed_image = self.master.processed_image[y, x]

        self.show_image()
    ##ZOOM
    def start_zoom(self, event):
        self.image = self.master.processed_image.copy()
        #self.zoom_start_x = event.x
        #self.zoom_start_y = event.y
        self.canvas.scan_mark(event.x, event.y)

    def zoom(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        #if self.rectangle_id_zoom:
        #    self.canvas.delete(self.rectangle_id_zoom)

        #self.zoom_end_x = event.x
        #self.zoom_end_y = event.y

        #self.rectangle_id_zoom = self.canvas.create_rectangle(self.zoom_start_x, self.zoom_start_y,self.zoom_end_x, self.zoom_end_y, width=1)


    def end_zoom(self, event):
        if self.zoom_start_x <= self.zoom_end_x and self.zoom_start_y <= self.zoom_end_y:
            start_x = int(self.zoom_start_x * self.ratio)
            start_y = int(self.zoom_start_y * self.ratio)
            end_x = int(self.zoom_end_x * self.ratio)
            end_y = int(self.zoom_end_y * self.ratio)
        elif self.zoom_start_x > self.zoom_end_x and self.zoom_start_y <= self.zoom_end_y:
            start_x = int(self.zoom_end_x * self.ratio)
            start_y = int(self.zoom_start_y * self.ratio)
            end_x = int(self.zoom_start_x * self.ratio)
            end_y = int(self.zoom_end_y * self.ratio)
        elif self.zoom_start_x <= self.zoom_end_x and self.zoom_start_y > self.zoom_end_y:
            start_x = int(self.zoom_start_x * self.ratio)
            start_y = int(self.zoom_end_y * self.ratio)
            end_x = int(self.zoom_end_x * self.ratio)
            end_y = int(self.zoom_start_y * self.ratio)
        else:
            start_x = int(self.zoom_end_x * self.ratio)
            start_y = int(self.zoom_end_y * self.ratio)
            end_x = int(self.zoom_start_x * self.ratio)
            end_y = int(self.zoom_start_y * self.ratio)

        x = slice(start_x, end_x, 1)
        y = slice(start_y, end_y, 1)

        self.master.processed_image = self.master.processed_image[y, x]

        self.show_image()
    def zoomer(self,event):
        self.text = self.canvas.create_text(0, 0, anchor='nw', text='Scroll to zoom')
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:
            scale        *= self.delta
            self.imscale *= self.delta
        if event.num == 4 or event.delta == 120:
            scale        /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, scale, scale)
        self.show_perm_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))



    def clear_canvas(self):
        self.canvas.delete("all")


    def clear_draw(self):
        self.canvas.delete(self.draw_ids)
















