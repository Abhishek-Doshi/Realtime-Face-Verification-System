import os
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from model_images import readImage, getFaces
from model import getFeatureVector

def extract_valid_dataset(source = None, destination = None):

    if (source == None): source = os.path.join(os.getcwd(), 'lfw-deepfunneled')
    if (destination == None): destination = os.path.join(os.getcwd(), 'lfw_data')

    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        if len(os.listdir(folder_path)) > 2:
            dest = shutil.move(folder_path, destination)

def extract_faces(source = None, destination = None):
    
    if (source == None): source = os.path.join(os.getcwd(), 'lfw_data')
    if (destination == None): destination = os.path.join(os.getcwd(), 'lfw_data_faces')

    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        os.mkdir(os.path.join(destination, folder))
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            print(img_path)
            faces = getFaces(readImage(img_path, cv_2 = True))
            if len(faces) > 0:
                face = faces[0]
                face.save(os.path.join(destination, folder, img))
        
def generate_dataframe(data, feature_col_idx):
    print('Creating Dataframe Matrix...')
    df = pd.DataFrame(data, columns = feature_col_idx)
    print('Dataframe Created.')
    return df

def appendImgToNPA(image, arr):
    vec = getFeatureVector(image)
    arr = np.append(arr, [vec], axis = 0)
    return arr

def generate_np_matrix(source = None):
    print('Creating Numpy Matrix...')
    if (source == None): source = os.path.join(os.getcwd(), 'lfw')
    img_arr = []

    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        
        for img in os.listdir(folder_path):
            img_path =  os.path.join(folder_path, img)
            image = readImage(img_path)
            ima = np.array(image)
            img_arr.append(ima)
            
    np_arr = np.array(img_arr)
    print('Numpy Matrix Created.')
    return np_arr

def get_feature_matrix(np_arr):
    print('Creating Feature Matrix...')
    feature_matrix = getFeatureVector(np_arr, matrix = True)
    print('Feature Matrix Created.')
    return feature_matrix
'''
def get_average_vectors(df, n_index):
    feature_col_idx = [col for col in range(2623)]
    avg_df = pd.DataFrame(columns = feature_col_idx)
    for idx in range(n_index):
        tmp = df[df.index == idx]
        tmp = tmp.mean(axis = 0)
        avg_df.append(tmp)
    return avg_df

def append_index(df):
    index = []
    source = os.path.join(os.getcwd(), 'lfw')
    for idx, folder in enumerate(os.listdir(source)):
        folder_path = os.path.join(source, folder)
        size = len(os.listdir(folder_path))
        for i in range(size):
            index.append(idx) 
    index = pd.Series(index)
    df = pd.concat([df, index], ignore_index=True)
    return df, idx + 1

df = pd.read_csv(r'LFW_DataFrame.csv')
df, n_index = append_index(df)
df = get_average_vectors(df, n_index)
df.to_csv('lfw_avg.csv', index = False)
'''

def get_average_vectors(df, avg_index):    
    avg_np = np.empty((0,2622))
    for i in range(len(avg_index) - 1):
        tmp = df.loc[avg_index[i]: avg_index[i+1], :]
        tmp = np.asarray(tmp.mean(axis = 0))
        avg_np = np.append(avg_np, tmp)
    avg_np = np.reshape(avg_np, (860, 2622))
    avg_df = pd.DataFrame(avg_np)
    return avg_df

def avg_index():
    index = [0]
    source = os.path.join(os.getcwd(), 'lfw')
    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        size = len(os.listdir(folder_path))
        index.append(index[-1] + size)
    return index
