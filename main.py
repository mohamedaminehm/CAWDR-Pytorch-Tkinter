import tkinter as tk
from tkinter import ttk
from editBar import EditBar
from imageViewer import ImageViewer


class Main(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.filename = ""
        self.original_image = None
        self.after_seg = None
        self.processed_image = None
        self.segmented_image = None
        self.classified_image = None
        self.histogramme_pred = None
        self.is_image_selected = False
        self.is_draw_state = False
        self.is_crop_state = False

        self.is_zoom_state = False


        self.filter_frame = None
        self.adjust_frame = None
        
        self.segmentationFrame = None

        self.title("CAWDR")

        self.editbar = EditBar(master=self)
        separator1 = ttk.Separator(master=self, orient=tk.HORIZONTAL)
        self.image_viewer = ImageViewer(master=self)

        self.editbar.pack(pady=10)
        separator1.pack(fill=tk.X, padx=20, pady=5)
        self.image_viewer.pack(fill=tk.BOTH, padx=20, pady=10, expand=1)



root = Main()
root.mainloop()
root.destroy()