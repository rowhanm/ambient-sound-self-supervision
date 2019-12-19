#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# In[ ]:


# #### 1. Get MFCC

# In[2]:

'''
audio_dir = './dataset/audio'

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


# #### 2. Get Kmeans Clusters

# In[7]:


kmeans_5 = KMeans(n_clusters=5).fit(results)
myfile_5 = open('./mfcc/mfcc_k5.csv', 'w')
for i in range(len(audio_names)):
    myfile_5.write(audio_names[i][:-4]+".jpg,"+str(kmeans_5.labels_[i])+"\n")
myfile_5.close()


# In[8]:


os.system('shuf ./mfcc/mfcc_k5.csv > ./mfcc/mfcc_k5_shuf.csv')
os.system('awk \'NR < 5208 { print >> "./mfcc/val_mfcc_k5.csv"; next } {print >> "./mfcc/train_mfcc_k5.csv" }\' ./mfcc/mfcc_k5_shuf.csv')


# In[ ]:


kmeans_10 = KMeans(n_clusters=10).fit(results)
myfile_10 = open('./mfcc/mfcc_k10.csv', 'w')
for i in range(len(audio_names)):
    myfile_10.write(audio_names[i][:-4]+".jpg,"+str(kmeans_10.labels_[i])+"\n")
myfile_10.close()
os.system('shuf ./mfcc/mfcc_k10.csv > ./mfcc/mfcc_k10_shuf.csv')
os.system('awk \'NR < 5208 { print >> "./mfcc/val_mfcc_k10.csv"; next } {print >> "./mfcc/train_mfcc_k10.csv" }\' ./mfcc/mfcc_k10_shuf.csv')


# In[ ]:


kmeans_15 = KMeans(n_clusters=15).fit(results)
myfile_15 = open('./mfcc/mfcc_k15.csv', 'w')
for i in range(len(audio_names)):
    myfile_15.write(audio_names[i][:-4]+".jpg,"+str(kmeans_15.labels_[i])+"\n")
myfile_15.close()
os.system('shuf ./mfcc/mfcc_k15.csv > ./mfcc/mfcc_k15_shuf.csv')
os.system('awk \'NR < 5208 { print >> "./mfcc/val_mfcc_k15.csv"; next } {print >> "./mfcc/train_mfcc_k15.csv" }\' ./mfcc/mfcc_k15_shuf.csv')


# In[ ]:


kmeans_30 = KMeans(n_clusters=30).fit(results)
myfile_30 = open('./mfcc/mfcc_k30.csv', 'w')
for i in range(len(audios_names)):
    myfile_30.write(audios_names[i][:-4]+".jpg,"+str(kmeans_30.labels_[i])+"\n")
myfile_30.close()
os.system('shuf ./mfcc/mfcc_k30.csv > ./mfcc/mfcc_k30_shuf.csv')
os.system('awk \'NR < 5208 { print >> "./mfcc/val_mfcc_k30.csv"; next } {print >> "./mfcc/train_mfcc_k30.csv" }\' ./mfcc/mfcc_k30_shuf.csv')


# In[ ]:


kmeans_60 = KMeans(n_clusters=60).fit(results)
myfile_60 = open('./mfcc/mfcc_k60.csv', 'w')
for i in range(len(audios_names)):
    myfile_60.write(audios_names[i][:-4]+".jpg,"+str(kmeans_60.labels_[i])+"\n")
myfile_60.close()
os.system('shuf ./mfcc/mfcc_k60.csv > ./mfcc/mfcc_k60_shuf.csv')
os.system('awk \'NR < 5208 { print >> "./mfcc/val_mfcc_k60.csv"; next } {print >> "./mfcc/train_mfcc_k60.csv" }\' ./mfcc/mfcc_k60_shuf.csv')


# In[ ]:





# #### 3. Create Dataset

# In[ ]:

'''



# In[ ]:


def clean_dataset(train_csv, val_csv):
    print("Creating df")
    train_dataframe = pd.read_csv(train_csv, header=None)
    count=0
    train_badlist=[]
    for i in range(len(train_dataframe)):
        img = os.path.join("./dataset/frames",train_dataframe.iloc[i,0])
        if not os.path.exists(img):
            count+=1
            train_badlist.append(train_dataframe.iloc[i,0])


    val_dataframe = pd.read_csv(val_csv, header=None)
    count=0
    val_badlist=[]
    for i in range(len(val_dataframe)):
        img = os.path.join("./dataset/frames",val_dataframe.iloc[i,0])
        if not os.path.exists(img):
            count+=1
            val_badlist.append(val_dataframe.iloc[i,0])
            
    for index,row in train_dataframe.iterrows():
        if row[0] in train_badlist:
            train_dataframe.drop(index, inplace=True)
    for index,row in val_dataframe.iterrows():
        if row[0] in val_badlist:
            val_dataframe.drop(index, inplace=True)
    print(len(train_dataframe))
    print(len(val_dataframe)) 
    return train_dataframe, val_dataframe


# In[ ]:


def create_h5(train_dataframe, val_dataframe, train_h5_path, val_h5_path):
    root = './dataset/frames/'
    print("Creating h5: ", train_h5_path)
    train_f = h5py.File(train_h5_path, mode='w')
    print(len(train_dataframe))
    for index,row in train_dataframe.iterrows():
        print(row[0])
        if index%1000 == 0:
            train_f.flush()
        img = cv2.imread(root+row[0])
        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_set = train_f.create_dataset(
                name='img_'+str(index),
                data=img,
                shape=(227,227,3),
                maxshape=(227,227,3),
                compression="gzip",
                compression_opts=4)
        label_set = train_f.create_dataset(
                name='lab_'+str(index),
                data=row[1],
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=3)
        if index == len(train_dataframe):
            break
    train_f.close()
    val_f = h5py.File(val_h5_path, mode='w')
    for index,row in val_dataframe.iterrows():
        if index%1000 == 0:
            val_f.flush()
        img = cv2.imread(root+row[0])
        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vimg_set = val_f.create_dataset(
                name='img_'+str(index),
                data=img,
                shape=(227,227,3),
                maxshape=(227,227,3),
                compression="gzip",
                compression_opts=4)
        vlabel_set = val_f.create_dataset(
                name='lab_'+str(index),
                data=row[1],
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=3)
        if index == len(val_dataframe):
            break
    val_f.close()
    # Clean up required since writing h5 has some issues
    # So loop through index numbers and write them again if it wasnt written
    for i in range(len(train_dataframe)):
        with h5py.File(train_h5_path, 'r+') as hf:    
            try:
                dset = hf["img_"+str(i)]
                label = hf["lab_"+str(i)].value[0]
            except Exception as e:
                img = cv2.imread(root+train_dataframe.iloc[i,0])
                img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_set = hf.create_dataset(
                        name='img_'+str(i),
                        data=img,
                        shape=(227,227,3),
                        maxshape=(227,227,3),
                        compression="gzip",
                        compression_opts=4)
                label_set = hf.create_dataset(
                        name='lab_'+str(i),
                        data=train_dataframe.iloc[i,1],
                        shape=(1,),
                        maxshape=(None,),
                        compression="gzip",
                        compression_opts=3)
    for i in range(len(val_dataframe)):
        with h5py.File(val_h5_path, 'r+') as hf:    
            try:
                dset = hf["img_"+str(i)]
                label = hf["lab_"+str(i)].value[0]
            except Exception as e:
                img = cv2.imread(root+val_dataframe.iloc[i,0])
                img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_set = hf.create_dataset(
                        name='img_'+str(i),
                        data=img,
                        shape=(227,227,3),
                        maxshape=(227,227,3),
                        compression="gzip",
                        compression_opts=4)
                label_set = hf.create_dataset(
                        name='lab_'+str(i),
                        data=train_dataframe.iloc[i,1],
                        shape=(1,),
                        maxshape=(None,),
                        compression="gzip",
                        compression_opts=3)


# In[ ]:


train_df_k5, val_df_k5 = clean_dataset("./mfcc/train_mfcc_k5.csv", "./mfcc/val_mfcc_k5.csv")
create_h5(train_df_k5, val_df_k5, "./mfcc/train_mfcc_k5.hdf5", "./mfcc/val_mfcc_k5.hdf5")


# In[ ]:

train_df_k10, val_df_k10 = clean_dataset("./mfcc/train_mfcc_k10.csv", "./mfcc/val_mfcc_k10.csv")
create_h5(train_df_k10, val_df_k10, "./mfcc/train_mfcc_k10.hdf5", "./mfcc/val_mfcc_k10.hdf5")


# In[ ]:


train_df_k15, val_df_k15 = clean_dataset("./mfcc/train_mfcc_k15.csv", "./mfcc/val_mfcc_k15.csv")
create_h5(train_df_k15, val_df_k15, "./mfcc/train_mfcc_k15.hdf5", "./mfcc/val_mfcc_k15.hdf5")


# In[ ]:
'''
train_df_k30, val_df_k30 = clean_dataset("./mfcc/train_mfcc_k30.csv", "./mfcc/val_mfcc_k30.csv")
create_h5(train_df_k30, val_df_k30, "./mfcc/train_mfcc_k30.hdf5", "./mfcc/val_mfcc_k30.hdf5")


# In[ ]:


train_df_k60, val_df_k60 = clean_dataset("./mfcc/train_mfcc_k60.csv", "./mfcc/val_mfcc_k60.csv")
create_h5(train_df_k60, val_df_k60, "./mfcc/train_mfcc_k60.hdf5", "./mfcc/val_mfcc_k60.hdf5")
'''
