import os
import numpy as np
import os.path
import pandas as pd
import h5py
import cv2


def clean_dataset(train_csv, val_csv):
    print("Creating df")
    train_dataframe = pd.read_csv(train_csv, header=None)
    count=0
    train_badlist=[]
    for i in range(len(train_dataframe)):
        img = os.path.join(sys.argv[4] ,train_dataframe.iloc[i,0])
        if not os.path.exists(img):
            count+=1
            train_badlist.append(train_dataframe.iloc[i,0])


    val_dataframe = pd.read_csv(val_csv, header=None)
    count=0
    val_badlist=[]
    for i in range(len(val_dataframe)):
        img = os.path.join(sys.argv[4],val_dataframe.iloc[i,0])
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



def create_h5(train_dataframe, val_dataframe, train_h5_path, val_h5_path):
    root = sys.argv[4]
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
