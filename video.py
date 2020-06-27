import cv2
from model_images import getFaces, getFaceCoordinates
from PIL import Image, ImageGrab
import numpy as np
from model import getFeatureVector, verifyVecs, verifyVecMat, verifyVecMatBeta
import pickle
import os
import pandas as pd
import time

path = os.path.join(os.getcwd(), 'Clusters_LFW')
loaded_model = pickle.load(open('Clusters_LFW\model.sav', 'rb'))
id_list = os.listdir(os.path.join(os.getcwd(), 'lfw'))
df_list = []

def preload_dataframe():
    time_0 = time.time()
    for idx in range(len(os.listdir(path)) - 1):
        df_list.append(pd.read_csv(os.path.join(path,'Cluster_'+str(idx),'cluster_'+str(idx)+'_data.csv')))
    time_1 = time.time()
    print('Task: Preload Dataframes     Time Required: ', time_1-time_0)

def query_dataframe():
    pass

def highlightFaces(image, scaleFactor = 1.3, minNeighbors = 5, pil =True, text = False):
    faces = getFaceCoordinates(image, scaleFactor, minNeighbors)
    for i,(x,y,w,h) in enumerate(faces):
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
        if text != False:
            cv2.putText(image, text[i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
    if pil: image = cv2_to_pil(image)
    return image

def search_preloaded_clusters_Beta(clusters, vec_matrix):
    
    print(clusters)
    time_0 = time.time()
    match_index = []
    idx_col = [df_list[cluster]['Unnamed: 0'] for cluster in clusters]
    tmp_df_list = [np.asarray(df_list[cluster].drop(columns = 'Unnamed: 0')) for cluster in clusters]
    time_1 = time.time()
    print('Task: Load Dataframe         Time Required: ', time_1-time_0)

    max_ids = verifyVecMatBeta(np.asarray(vec_matrix), np.asarray(tmp_df_list))
    for i,match_idx in enumerate(max_ids):
        if match_idx != None: match_index.append(idx_col[i][match_idx])
        else:  match_index.append(None)
    
    time_2 = time.time()     
    print('Task: Query Dataframe        Time Required: ', time_2-time_1)
    return match_index
    

def search_preloaded_clusters(clusters, vec_matrix):
    match_index = []
    for idx in range(len(clusters)):
        
        time_0 = time.time()
        df = df_list[clusters[idx]]
        idx_col = df['Unnamed: 0']
        time_1 = time.time()
        print('Task: Load Dataframe         Time Required: ', time_1-time_0)

        cl_np = np.asarray(df.drop(columns = 'Unnamed: 0'))
        match_idx = verifyVecMat(np.asarray(vec_matrix[idx]), cl_np)
        if match_idx != None: match_index.append(idx_col[match_idx])
        else:  match_index.append(None)
        time_2 = time.time()
        print('Task: Query Dataframe        Time Required: ', time_2-time_1)
    return match_index

def search_clusters(clusters, vec_matrix):
    match_index = []
    for idx in range(len(clusters)):
        
        time_0 = time.time()
        cl_path = os.path.join(path,'Cluster_'+str(clusters[idx]),'cluster_'+str(clusters[idx])+'_data.csv')
        df = pd.read_csv(cl_path)
        idx_col = df['Unnamed: 0']
        time_1 = time.time()
        print('Task: Load Dataframe         Time Required: ', time_1-time_0)

        cl_np = np.asarray(df.drop(columns = 'Unnamed: 0'))
        match_idx = verifyVecMat(np.asarray(vec_matrix[idx]), cl_np)
        if match_idx != None: match_index.append(idx_col[match_idx])
        else:  match_index.append(None)
        time_2 = time.time()
        print('Task: Query Dataframe        Time Required: ', time_2-time_1)
    return match_index

preload_dataframe()

while True:

    scr_pil = np.array(ImageGrab.grab(bbox = (0, 100, 400, 300)))
    scr_cv = cv2.cvtColor(scr_pil, cv2.COLOR_BGR2RGB)
    time0 = time.time()
    faces = getFaces(scr_cv)
    time1 = time.time()
    print('Task: Extract Faces          Time Required: ', time1-time0)

    faces = np.asarray([np.asarray(face) for face in faces])
    time2 = time.time()
    print('Task: Faces as Numpy         Time Required: ', time2-time1)

    print(len(faces), ' faces detected.')

    text = []

    if len(faces) > 0:
        time3 = time.time()
        vec_matrix = getFeatureVector(faces, matrix = True)
        time4 = time.time()
        print('Task: Get Feature Vector     Time Required: ', time4-time3)

        clusters = loaded_model.predict(vec_matrix)
        time5 = time.time()
        print('Task: Predict Cluster        Time Required: ', time5-time4)

        match_index = search_preloaded_clusters(clusters, vec_matrix)
        time6 = time.time()
        print('Task: Search Clusters        Time Required: ', time6-time5)
        for idx in match_index:
            if idx != None: text.append(id_list[idx]) 
            else: text.append('Unknown')
        time7 = time.time()
        print('Task: Process Frame          Time Required: ', time7-time6)

    time8 = time.time() 
    print('Total                        Time Required: ', time8-time0, '\n')
    image = highlightFaces(scr_cv, pil = False, text = text)
    cv2.imshow('Video', image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break