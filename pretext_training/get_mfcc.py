


import librosa
from librosa.feature import mfcc
import os
import numpy as np
import scipy
import os.path
import pandas as pd
from sklearn.cluster import KMeans
import h5py
import cv2

from utils import data_cleaner

audio_dir = sys.argv[1]

audio_names = []

def get_mfcc(audio_dir):
    result_array = np.empty((0,20,155))
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            try:
                audio_names.append(filename)     
                y, sr = librosa.core.load(audio_dir+"/"+filename, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                mfcc = mfcc.flatten()
                mfcc = np.pad(mfcc, (0, (3100-mfcc.shape[0])%3100), 'constant').reshape((20,155))
                result_array = np.append(result_array, [mfcc], axis=0)
            except Exception as e:
                print(e)
                pass
    return result_array    

result = get_mfcc(audio_dir)
results = np.reshape(result,(result.shape[0],-1))

path_to_feats = sys.argv[2]
number_of_clusters = int(sys.argv[3])


# Run KMeans

kmeans = KMeans(n_clusters=number_of_clusters).fit(results)

# Generate CSV file with labels
myfile = open(sys.argv[2] + '/mfcc_k'+str(number_of_clusters)+'.csv', 'w')
for i in range(len(audio_names)):
    myfile.write(audio_names[i][:-4]+".jpg,"+str(kmeans.labels_[i])+"\n")
myfile.close()

#Shuffle CSV file
os.system('shuf '+path_to_feats+'/mfcc_k'+str(number_of_clusters)+'.csv > '+path_to_feats+'/mfcc_k'+str(number_of_clusters)+'_shuf.csv')

# Split file into train and validation CSV

number_of_valid_samples = 5208 # Change as per your need 

os.system('awk \'NR < '+str(number_of_valid_samples)+' { print >> "'+path_to_feats+'/val_mfcc_k'+str(number_of_clusters)+'.csv"; next } {print >> "'+path_to_feats+'/train_mfcc_k'+str(number_of_clusters)+'.csv" }\' '+path_to_feats+'/mfcc_k'+str(number_of_clusters)+'_shuf.csv')

# 3. Create Trainable Dataset

train_df, val_df = data_cleaner.clean_dataset(path_to_feats+"/train_mfcc_k"+str(number_of_clusters)+".csv", path_to_feats+"/val_mfcc_k"+str(number_of_clusters)+".csv")

data_cleaner.create_h5(train_df, val_df, path_to_feats+"/train_mfcc_k"+str(number_of_clusters)+".hdf5", path_to_feats+"/val_mfcc_k"+str(number_of_clusters)+".hdf5")
