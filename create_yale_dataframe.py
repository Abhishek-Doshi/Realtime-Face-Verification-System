import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from model_images import readImage, cv2_to_pil
from model import findCosineDifference, findCosineSimilarity, findEuclideanDistance, findL1Norm, getFeatureVector, preprocess_image

path = os.path.join(os.getcwd(), 'Yale')
img_list = os.listdir(path)
feature_col_idx = [idx for idx in range(2622)] 
arr = np.empty((0,2622), float)

def generate_dataframe(data):
    df = pd.DataFrame(data, columns = feature_col_idx)
    return df

def appendImgToNPA(image, arr, name):
    print('Processing image ', name)
    vec = getFeatureVector(image)
    arr = np.append(arr, [vec], axis = 0)
    return arr

'''
for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    image = readImage(img_path)
    img = np.asarray(image)
    image_np = np.empty(img.shape, float)
    image_np = np.append(image_np, img, axis = 0)
    image_np = np.append(image_np, img, axis = 0)
    image = Image.fromarray(image_np)
    arr = appendImgToNPA(image, arr, filename)
'''
for filename in img_list:
    path = os.path.join(os.getcwd(), 'Yale', filename)
    img_path = os.path.join(path)
    image = readImage(img_path)
    arr = appendImgToNPA(image, arr, path)

df = generate_dataframe(arr)
df.to_csv(r'Yale_DataFrame.csv', index = False)