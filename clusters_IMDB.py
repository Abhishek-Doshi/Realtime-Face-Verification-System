import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pickle
import time

feature_col_idx = [idx for idx in range(2622)] 
df = pd.read_csv(r'C:\Users\Home\Desktop\Abhishek\Face Verification\IMDB_file_0_DataFrame.csv')
path = os.path.join(os.getcwd(), 'Clusters')
n_clusters = 10

def scale_data(df):
    scaler = MinMaxScaler()
    scaler.fit(df[df.columns])
    df[df.columns] = scaler.transform(df[df.columns])
    return df

def cluster_data(df, n_clusters = 10):
    model = KMeans(n_clusters = n_clusters)
    model.fit(df[df.columns])
    df['cluster'] = model.predict(df[df.columns])
    return model, df

def segregate_data_by_clusters(n_clusters = 10):
    for i in range(n_clusters):
        print('Saving Cluster: ', i)
        tmp = pd.DataFrame(columns = feature_col_idx)
        tmp = df[df.cluster==i].copy()
        os.mkdir(os.path.join(path, 'Cluster_' + str(i)))
        save_path = os.path.join(path, 'Cluster_' + str(i), 'cluster_' + str(i) +'_data.csv')
        tmp.to_csv(save_path, index = True)

def save_model(model):
    print('Saving Model...\n')
    filename = 'Clusters\model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print('Model Saved Successfully.')

tic = time.time()
df = scale_data(df)
model, df = cluster_data(df, n_clusters = 10)
segregate_data_by_clusters(n_clusters = 10)
toc = time.time()
save_model(model)

time_req = toc - tic
print('Time Required: ', time_req)