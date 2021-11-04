#### FOR DEPLOYMENT ####





from pathlib import Path
import numpy as np
import cv2
from skimage import morphology 





#### Socket communication #####
import socket
import pickle
import cv2
import struct

HOST = "127.0.0.1"
PORT = 5050

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

def sendrecv_image_to_server(image):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST,PORT))
        
        result, image = cv2.imencode('.jpg', image, encode_param)
        data = pickle.dumps(image, 0)
        size = len(data)

        s.sendall(struct.pack(">L",size)+data)

        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))
        while len(data) < payload_size:
            #print(f"recv: {len(data)}")
            data += s.recv(4096)
            #if not data:
            #    break
        print('done recv !!')

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack('>L', packed_msg_size)[0]
        while len(data) < msg_size:
            data += s.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)

        print(frame)
        print(frame.shape)
        

        return frame

    
class SegmentationClass():
    def __init__(self,original_image):
        self.original_image = original_image


        #self.get_model = getattr(torchvision.models, model_name)
    

    def server_segmentation(self):
        image = self.original_image
        image = cv2.resize(image,(4000,800))
        resultat_ = sendrecv_image_to_server(image)
        mask_resultat = morphology.remove_small_objects(resultat_ > 0 , 100,connectivity=4)
        mask_resultat = morphology.remove_small_holes(mask_resultat, 100,connectivity=1)
        return ((mask_resultat )*255).astype(np.uint8)

    