# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def Readdata(dataset):
    bj={'WT':'WT-SpCas9','eSp':'raw_eSpCas9','HF1':'SpCas9-HF1','xCas':'raw_xCas',
        'SniperCas':'raw_SniperCas','HypaCas9':'raw_HypaCas9','SpCas9':'raw_SpCas9',
        'CRISPRON':'CRISPRON','HT_Cas9':'HT_cas9','chari2015Train293T':'chari2015Train293T',
        'doench2016_hg19':'doench2016_hg19','doench2016plx_hg19':'doench2016plx_hg19',
        'hart2016-Hct1162lib1Avg':'hart2016-Hct1162lib1Avg',
        'hart2016-HelaLib1Avg':'hart2016-HelaLib1Avg','hart2016-HelaLib2Avg':'hart2016-HelaLib2Avg',
        'hart2016-Rpe1Avg':'hart2016-Rpe1Avg','xu2015TrainHl60':'xu2015TrainHl60',
        'chari2015TrainK562':'chari2015TrainK562','doench2014-HS':'doench2014-HS',
        'doench2014-Mm':'doench2014-Mm','morenoMateos2015':'morenoMateos2015',
        'xu2015TrainKbm7':'xu2015TrainKbm7'}
    data=[]
    path='./Datasets/'+bj[dataset]+'.csv'
    f_csv=csv.reader(open(path))
    for row in f_csv:
        data.append(row[1:])
    return data[1:]

def Readbenchmark():
    path='./Datasets/benchmark.csv'
    f_csv=csv.reader(open(path))
    data=[]
    for row in f_csv:
        if (len(row[0])==23):
            data.append(row[0])
    return data
def process(dataset):
    bj={'A':0,'G':1,'C':2,'T':3}
    data=[]
    target=[]
    for i in dataset:
        if (len(str(i[0]))==23 and len(i[2])!=0):
            target.append(i[2])
            data.append(i[0])
        #k=i[0]
        #tmp=np.zeros((21,4))
        #for i in range(21):
        #    tmp[i][bj[k[i]]]=1
        #print(k)
        #print(tmp)
        #data.append(tmp)
    return data,target

def process_no_normalize(dataset):
    bj={'A':0,'G':1,'C':2,'T':3}
    data=[]
    target=[]
    for i in dataset:
        if (len(str(i[0]))==23 and len(i[2])!=0):
            target.append(i[1])
            data.append(i[0])
        #k=i[0]
        #tmp=np.zeros((21,4))
        #for i in range(21):
        #    tmp[i][bj[k[i]]]=1
        #print(k)
        #print(tmp)
        #data.append(tmp)
    return data,target


def calculate():
    datasets=['WT','eSp','HF1','xCas','SniperCas','SniperCas','HypaCas9','SpCas9','CRISPRON','HT_Cas9']
    
    
    x_data=np.zeros(shape=(1,72))
    bj={2:'A',3:"T",4:'C',5:"G"}
    zp={}
    for i in datasets:
        dt="data/"+i+".npy"
        x=np.load(dt)
        x=np.array(x)
        for j in x:
            c=""
            for tmp in j[1:24]:
                c=c+bj[tmp]
            if (c in zp.keys()):
                zp[c]=zp[c]+1
            else:
                zp[c]=1
    extra=Readbenchmark()
    for i in extra:
        if i in zp.keys():
            zp[i]=zp[i]+1
        else:
            zp[i]=1
    
    res=[]
    for i in zp.keys():
        tmp=Dimer_split_seqs(i)
        res.append(tmp)
    res=np.array(res)
    return res

from sklearn.model_selection import train_test_split

def split_and_save_dataset(dataset_name, test_size=0.15):  
    # 读取数据  
    data = Readdata(dataset_name)  
    # 处理数据  
    sequences, targets = process_no_normalize(data)  
      
    # 将数据和目标转换为numpy数组  
    X = np.array(sequences)
    y = np.array(targets)  
      
    # 分割数据集  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=40)  
      
    # 保存训练集和测试集  
    np.save(f'./data/{dataset_name}_train_X.npy', X_train)  
    np.save(f'./data/{dataset_name}_train_y.npy', y_train)  
    np.save(f'./data/{dataset_name}_test_X.npy', X_test)  
    np.save(f'./data/{dataset_name}_test_y.npy', y_test) 

def split_and_save_dataset_no_normalize(dataset_name, test_size=0.15):  
    # 读取数据  
    data = Readdata(dataset_name)  
    # 处理数据  
    sequences, targets = process(data)  
      
    # 将数据和目标转换为numpy数组  
    X = np.array(sequences)
    y = np.array(targets)  
      
    # 分割数据集  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)  
      
    # 保存训练集和测试集  
    np.save(f'./data/{dataset_name}_train_X.npy', X_train)  
    np.save(f'./data/{dataset_name}_train_y.npy', y_train)  
    np.save(f'./data/{dataset_name}_test_X.npy', X_test)  
    np.save(f'./data/{dataset_name}_test_y.npy', y_test) 
if __name__ == "__main__":
    

    #datasets=['WT','eSp','HF1','xCas','SniperCas','SniperCas','HypaCas9','SpCas9','CRISPRON','HT_Cas9']
    datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']
    for i in datasets:
        split_and_save_dataset(i)
        k1,k2,k3,k4=load_data_final(i)
        print(k1.shape[0]+k3.shape[0],i)

    from utils import load_data
    k1,k2,k3,k4=load_data_final("WT")
    #split_and_save_dataset_no_normalize("WT")

