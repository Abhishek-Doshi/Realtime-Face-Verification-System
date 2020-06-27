import cv2
import os
import numpy as np
from PIL import Image
from model_images import getFaces, highlightFaces, readImage, cv2_to_pil

path = os.path.join(os.getcwd(), 'IMDB_Data')
new_path = os.path.join(os.getcwd(), 'Data')

file_no = 0

for file in os.listdir(path) :

    print('Processing file: ', file_no, '\n')
    img_dir = os.path.join(path, file)
    list_images = os.listdir(img_dir)
    new_file = os.path.join(new_path, 'file_' + str(file_no))
    os.mkdir(new_file)

    img_no = 0

    for image in list_images:
        img = readImage(os.path.join(img_dir, image), cv_2= True)
        images = getFaces(img)
        if len(images) != 0:
            img = images[0]
            name = str(img_no) + '.jpg'
            save_path = os.path.join(new_file, name)
            img.save(save_path)
            img_no += 1
    
    file_no += 1