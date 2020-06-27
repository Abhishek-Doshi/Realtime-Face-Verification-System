from model import verifyFace, verifyFaceVector, verifyVecs, getFeatureVector
from model_images import getFaces, readImage
import numpy as np
import cv2
from PIL import Image
import time
import pickle
import os
import pandas as pd

loaded_model = pickle.load(open('model.sav', 'rb'))

while True:

    tic = time.time()
    
    IMG = readImage('download.jpg', cv_2 = True)
    faces = getFaces(IMG)
    
    if len(faces) != 0:

        face = faces[0]
        vec = getFeatureVector(face)
        cluster = loaded_model.predict([vec])
    
        path = os.path.join(os.getcwd(), 'Clusters')
        cl_path = os.path.join(path, 'Cluster_' + str(cluster[0]), 'cluster_' + str(cluster[0]) + '_data.csv')
        df = pd.read_csv(cl_path)
        vectors = df.to_numpy()

        for r_idx, og_vec in enumerate(vectors):
            if verifyFaceVector(og_vec, vec, print_score = True):
                
                break
        
        toc = time.time()
        time_req = toc - tic
        print("Time Required: ", time_req)
    
    else: print('No face found. \n')