import cv2
import os
import numpy as np
from PIL import Image
from model_images import getFaces, highlightFaces, readImage, cv2_to_pil

path = os.path.join(os.getcwd(), 'yalefaces')
new_path = os.path.join(os.getcwd(), 'Yale')


for count, filename in enumerate(os.listdir(path)): 
    src = os.path.join(path, filename)
    dst = os.path.join(new_path, filename + '.jpg')
    os.rename(src, dst)
