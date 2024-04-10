# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
import math
from utils import *
import time
def conduct(x):
    a=0
    b=0
    c=0
    for i in x:     
        if (i<0.25):
            a=a+1
        elif (i>0.75):
            c=c+1
        else:
            b=b+1
    return [a,b,c]

def mixup(input_x, input_y, label_size, alpha):
    # get mixup lambda
    batch_size = tf.shape(input_x)[0]
    #input_y = tf.one_hot(input_y, depth=label_size)
    
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    index = tf.compat.v1.random_shuffle(tf.range(batch_size))
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input
    xmix = tf.cast(input_x,tf.float64) * mix + tf.cast(random_x,tf.float64) * (1 - mix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    return xmix, ymix


#nts=np.load("xCas.npy")
#y = np.array(y, dtype='float64')

def compute_optimal_transport(M, r, c, lam=1, epsilon=0.01):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)

    # Avoiding poor math condition
    P /= P.sum()
    u = np.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    t=0
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        # Shape (n, )
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
        t=t+1
        if (t>200):
            break
    #print(t)
    return P, np.sum(P * M)

#import time

def compare(a,b):
    dis=np.linalg.norm(a-b,ord=2)
    return dis

def min_max_normalize(arr):
    min_val = min(arr)
    max_val = max(arr)
    normalized_arr= (arr-min_val)/(max_val-min_val)
    #normalized_arr = [(x - min_val) / (max_val - min_val) for x in arr]
    return normalized_arr


def Add_noise(x):
    lens=tf.shape(x)[1]
    noise_add=np.zeros(shape=(lens,))
    k=0.1
    for i in range(lens):
        noise_add[i]=random.uniform(1-k,1+k)
    x=x*noise_add
    return x

def int_array(x):
    x2=np.array(x)
    shape0=x.shape[0]
    shape1=x.shape[1]
    for i in range(shape0):
        for j in range(shape1):
            if (x2[i][j]<0.5):
                x2[i][j]=0
            else:
                x2[i][j]=1
    return x2

def augmix(input_x,input_y,before,after,alpha):
    batch_size = tf.shape(input_x)[0]
    lens=tf.shape(input_x)[1]
    
    random_x_middle=before.predict(input_x)
    random_x_middle=Add_noise(random_x_middle)
    random_x=after.predict(random_x_middle)
    random_x=np.reshape(random_x,newshape=(-1,23,4))
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)
    
    t_middle=before.predict(input_x)
    t_middle=mix * t_middle + (1-mix) * (before.predict(random_x))
    xmix=after.predict(t_middle)
    ymix=input_y

    return xmix,ymix


import matplotlib.pyplot as plt


def analysis(input_x,input_y,before,after):
    batch_size = tf.shape(input_x)[0]
    lens=tf.shape(input_x)[1]
    
    alpha=0.1
    random_x_middle=before.predict(input_x)
    random_x_middle=Add_noise(random_x_middle)
    random_x=after.predict(random_x_middle)
    random_x=np.reshape(random_x,newshape=(-1,23,4))
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)
    
    t_middle=before.predict(input_x)
    t_middle=mix * t_middle + (1-mix) * (before.predict(random_x))
    xmix=after.predict(t_middle)
    input_x2=np.reshape(input_x,newshape=(-1,92))
    c=input_x2-xmix
    c_final=c.sum(axis=1)
    res=0
    for i in c_final:
        if (i!=0):
            res=res+1
    print(res)
    plt.plot(c_final)

def label_correction_pre(params,train_data,train_label,val_data,val_label,dataset):
    path1="model/"+dataset+".h5"
    pathtmp1="model/"+dataset+"tmp1.h5"
    pathtmp2="model/"+dataset+"tmp2.h5"
    m1=tr_try(params)
    m2=tr_try(params)
    batch_size=btsz
    epochs=50
    learning_rate=params['train_base_learning_rate']
    
    train_data=convert_one_hot(train_data)
    val_data=convert_one_hot(val_data)

    
    m1.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    m2.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    result1=Result()
    result2=Result()
    
    batch_end_callback1 = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_test_weight(m1, val_data,val_label,result1,pathtmp1
                                                                )))
    batch_end_callback2 = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_test_weight(m2, val_data,val_label,result2,pathtmp2
                                                                )))

    m1.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=[batch_end_callback1])

    m2.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=[batch_end_callback2])
    
def label_correction(params, train_data, train_label,dataset,r=0.2
                     ,threshold=0.1,f=0.0):
    path1="model/"+dataset+".h5"
    pathtmp1="model/"+dataset+"tmp1.h5"
    pathtmp2="model/"+dataset+"tmp2.h5"
    m1=tr_try(params)
    m2=tr_try(params)
    batch_size=btsz
    epochs=50
    learning_rate=params['train_base_learning_rate']

    m1.load_weights(pathtmp1)
    m2.load_weights(pathtmp2)
    
    train_data=convert_one_hot(train_data)
    
    res1=m1.predict(train_data)
    res2=m2.predict(train_data)
    res1=np.reshape(res1,newshape=(-1,))
    res2=np.reshape(res2,newshape=(-1,))
    #return res1
    batch=train_data.shape[0]
    loss1=np.zeros(shape=(batch,))
    loss2=np.zeros(shape=(batch,))
    
    for i in range(0,batch):
        loss1[i]=(res1[i]-train_label[i])**2
        loss2[i]=(res2[i]-train_label[i])**2
    
    index1=loss1.argsort()
    index2=loss2.argsort()
    
    lens=int(batch*(1-r))
    needre={}
    for i in range(lens,batch):
        if (needre.get(index1[i])):
            needre[index1[i]]=needre[index1[i]]+1
        else:
            needre[index1[i]]=1
        if (needre.get(index2[i])):
            needre[index2[i]]=needre[index2[i]]+1
        else:
            needre[index2[i]]=1
    needrefinal=[]
    for key in needre:
        if (needre[key]>1):
            needrefinal.append(key)
    needrefinal=np.array(needrefinal)
    
    for i in needrefinal:
        if (abs(res1[i]-res2[i])<threshold):
            train_label[i]=f*train_label[i]+(1.0-f)*((res1[i]+res2[i])/2.0)
    
    return train_label
