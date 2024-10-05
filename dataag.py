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
def Alignmix(xa,xb,lanmda):
    # x= c x h x w
    #start=time.time()
    c=1
    #return 1
    xa=np.reshape(xa,newshape=(c,-1))
    xb=np.reshape(xb,newshape=(c,-1))
    
    # x = c x k
    
    r=[]
    for i in xa[0]:
        r.append(i)
    

    c=[]
    for i in xb[0]:
        c.append(i)
    
    
    r=np.array(r)
    c=np.array(c)

    # r= (512,) c=(512,)
    xs=512
    cost=np.zeros(shape=(xs,xs))
    for i in range(xs):
        for j in range(i,xs):
            cost[i][j]=abs(r[i]-c[j])
            cost[j][i]=cost[i][j]
    
    #middle=time.time()
    P,rara=compute_optimal_transport(cost, r, c, lanmda)
    P=P*(xs*1.0)
    #print(P.shape,xb.shape)
    Aalign=np.matmul(xb,P.transpose())
    
    mixup=lanmda*xa+(1-lanmda)*Aalign

    return mixup

def Alignmix_ten(xa,xb,mixrate,model,aftermodel,input_y,random_y):
    # x= c x h x w
    #start=time.time()
    
    c=1
    #return 1
    xa=np.reshape(xa,newshape=(c,-1))
    xb=np.reshape(xb,newshape=(c,-1))
    
    # x = c x k
    
    r=[]
    for i in xa[0]:
        r.append(i)
    

    c=[]
    for i in xb[0]:
        c.append(i)
    
    
    r=np.array(r)
    c=np.array(c)

    # r= (512,) c=(512,)
    xs=512
    cost=np.zeros(shape=(xs,xs))
    for i in range(xs):
        for j in range(i,xs):
            cost[i][j]=abs(r[i]-c[j])
            cost[j][i]=cost[i][j]
    
    #middle=time.time()
    P,rara=compute_optimal_transport(cost, r, c)
    P=P*(xs*1.0)
    #print(P.shape,xb.shape)
    Aalign=np.matmul(xb,P.transpose())
    
    
    #third=time.time()
    mixtenx=[]
    mixteny=[]
    targety=[]
    for i in range(10):
        lanmda=mixrate[i]
        tmpx=lanmda*xa+(1-lanmda)*Aalign
        tmpy=lanmda*input_y+(1-lanmda)*random_y
        mixtenx.append(tmpx)
        mixteny.append(tmpy)
    mixtenx=np.array(mixtenx)
    mixtenx=np.reshape(mixtenx,newshape=(-1,512))
    mixteny=np.array(mixteny)

    revise_x=aftermodel.predict(mixtenx)
    revise_x=np.reshape(revise_x,newshape=(-1,23,4,1))
    #revise_x=revise(revise_x)
    mixtenx=np.array(revise_x)

    tmpy2=model.predict(revise_x)
    targety=np.array(tmpy2)
    tr2=np.array(targety)
    #final1=time.time()
    for i in range(10):
        targety[i]=abs(mixteny[i]-targety[i])
        #targety[i]=abs(targety[i]-0.3)
    index=np.argmax(targety)
    
    targety=np.reshape(targety,newshape=(10,))
    needindex=targety.argsort()
    k=0
    mixupdropout=0.2
    tf=True
    while (k<9 and tf):
        if (random.random()<mixupdropout):
            tf=False
        k=k+1
    index=needindex[k]
    # 手动校准
    mixteny[index]=(mixteny[index]+tr2[index])/2.0
    #final2=time.time()
    #print(middle-start,third-middle,final1-third,final2-final1)
    return mixtenx[index],mixteny[index]

def Almix(input_x, input_y, label_size, alpha):
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
    xmix = []
    for i in range(batch_size):
        if (i%1000==0):
            print("Now i=",i)
        xmix.append(Alignmix(input_x[i], random_x[i], mix))
    xmix=np.array(xmix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    xmix=np.reshape(xmix,newshape=(batch_size,512))
    return xmix, ymix

import numpy as np
from sklearn.neighbors import KernelDensity


def get_mixup_sample_rate(data_list,y_list, kernel="gaussian",bandwidth=1.0):
    data_list=data_list.reshape(data_list.shape[0],-1)
    mix_idx = []
    N = len(data_list)
    
    index=[]
    ######## use kde rate or uniform rate #######
    for i in range(N):
        data_i = data_list[i]
        data_i = data_i.reshape(-1,data_i.shape[0])
        kd = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data_i)
        each_rate = np.exp(kd.score_samples(data_list))
        each_rate /= np.sum(each_rate)  
        tmp=wt_sample(each_rate)
        index.append(tmp)
        if (i%1000==0):
            print("Now i=",i)
    index=np.array(index)
    #self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    return index



def weight_sampling(w_list):
    ran = np.random.uniform(0,1)
    sum=0
    for i in range(len(w_list)):
        sum+=w_list[i]
        if(ran<sum):
            return i

def wt_sample(w_list):
    tmp=weight_sampling(w_list)
    return np.array(tmp)

def AL_mixup(input_x, input_y, alpha):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    batch_size = tf.shape(input_x)[0]
    
    index=get_mixup_sample_rate(input_y, input_y)
    #index = tf.compat.v1.random_shuffle(tf.range(batch_size))

    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input

    xmix = []
    for i in range(batch_size):
        if (i%1000==0):
            print("Now i=",i)
        xmix.append(Alignmix(input_x[i], random_x[i], mix))
    xmix=np.array(xmix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    xmix=np.reshape(xmix,newshape=(batch_size,512))
    return xmix, ymix

def C_mixup(input_x, input_y, alpha):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    
    index=get_mixup_sample_rate(input_y, input_y)
    #index=wt_sample(mdx)
    
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input
    xmix = tf.cast(input_x,tf.float64) * mix + tf.cast(random_x,tf.float64) * (1 - mix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    return xmix, ymix

def C_AL_mixup_REM(input_x, input_y,model,aftermodel):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    batch_size = tf.shape(input_x)[0]
    
    #index=get_mixup_sample_rate(input_y, input_y)
    index = tf.compat.v1.random_shuffle(tf.range(batch_size))
    #index = tf.compat.v1.random_shuffle(tf.range(batch_size))
    
    mixrate=[]
    for i in range(1,11):
        alpha=i*1.0/10.0
        mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
        mix = tf.maximum(mix, 1 - mix)
        mixrate.append(mix)
    mixrate=np.array(mixrate)

    # get random shuffle sample
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input

    xmix = []
    ymix = []
    for i in range(batch_size):
        if (i%1000==0):
            print("Now i=",i)
        finalx,finaly=Alignmix_ten(input_x[i], random_x[i], mixrate,model,aftermodel,input_y[i],random_y[i])
        xmix.append(finalx)
        ymix.append(finaly)
    xmix=np.array(xmix)
    ymix=np.array(ymix)
    ymix=np.reshape(ymix,newshape=(ymix.shape[0],))
    return xmix, ymix

def compare(a,b):
    dis=np.linalg.norm(a-b,ord=2)
    return dis

def min_max_normalize(arr):
    min_val = min(arr)
    max_val = max(arr)
    normalized_arr= (arr-min_val)/(max_val-min_val)
    #normalized_arr = [(x - min_val) / (max_val - min_val) for x in arr]
    return normalized_arr

def C_mixup_noretry(input_x, input_y, alpha,index):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    
    #index=get_mixup_sample_rate(input_y, input_y)
    #index=wt_sample(mdx)
    
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input
    xmix = tf.cast(input_x,tf.float64) * mix + tf.cast(random_x,tf.float64) * (1 - mix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    return xmix, ymix

def final_mix(input_x, input_y,model,aftermodel):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    batch_size = tf.shape(input_x)[0]
    
    # get mixed input
    
    xmix= []
    ymix= []
    sq=np.zeros(shape=(batch_size,))
    sd=np.zeros(shape=(batch_size,))
    st=np.zeros(shape=(batch_size,))
    index=get_mixup_sample_rate(input_y, input_y)
    for i in range(11):
        rate=i*0.1
        fx,fy=C_mixup_noretry(input_x,input_y,rate,index)
        tr=aftermodel.predict(fx)
        tr2=np.reshape(tr,newshape=(-1,23,4,1))
        targety=model.predict(tr2)
        for k in range(batch_size):
            sq[k]=abs(fy[k]-targety[k])*abs(fy[k]-targety[k])
            sd[k]=-1*np.linalg.norm(input_x[k]-fx[k],ord=2)
        
        sq=min_max_normalize(sq)
        sd=min_max_normalize(sd)
        sq=np.array(sq)
        sd=np.array(sd)
        st=2*sq+sd
        #st=np.array(st)
        indexe=st.argsort()
        j=batch_size/10
        j=int(j)
        indexf=indexe[:j]
        fy=np.reshape(fy,newshape=(-1,1))
        gx=tr[indexf]
        gy=fy[indexf]
        xmix.append(gx)
        ymix.append(gy)
    
    xmix=np.concatenate(xmix)
    ymix=np.concatenate(ymix)
    ymix=np.reshape(ymix,newshape=(-1,))
    return xmix, ymix

def Alignnomix(xa,xb):
    # x= c x h x w
    #start=time.time()
    c=1
    #return 1
    xa=np.reshape(xa,newshape=(c,-1))
    xb=np.reshape(xb,newshape=(c,-1))
    
    # x = c x k
    
    r=[]
    for i in xa[0]:
        r.append(i)
    

    c=[]
    for i in xb[0]:
        c.append(i)
    
    
    r=np.array(r)
    c=np.array(c)

    # r= (512,) c=(512,)
    xs=512
    cost=np.zeros(shape=(xs,xs))
    for i in range(xs):
        for j in range(i,xs):
            cost[i][j]=abs(r[i]-c[j])
            cost[j][i]=cost[i][j]
    
    #middle=time.time()
    P,rara=compute_optimal_transport(cost, r, c)
    P=P*(xs*1.0)
    #print(P.shape,xb.shape)
    Aalign=np.matmul(xb,P.transpose())
    
    return Aalign

def final_mix_withal(input_x, input_y,model,aftermodel):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    batch_size = tf.shape(input_x)[0]
    
    # get mixed input
    
    xmix= []
    ymix= []
    sq=np.zeros(shape=(batch_size,))
    sd=np.zeros(shape=(batch_size,))
    st=np.zeros(shape=(batch_size,))
    #index=get_mixup_sample_rate(input_y, input_y)
    index=get_mixup_sample_rate(input_x,input_x)
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    tmpx=[]
    for i in range(batch_size):
        tmp1=Alignnomix(input_x[i],random_x[i])
        tmpx.append(tmp1)
    tmpx=np.concatenate(tmpx)
    for i in range(11):
        rate=i*0.1
        
        mix = tf.compat.v1.distributions.Beta(rate,rate).sample(1)
        mix = tf.maximum(mix, 1 - mix)

        mix=tf.cast(mix,tf.double)
        
        fx = tf.cast(input_x,tf.float64) * mix + tf.cast(tmpx,tf.float64) * (1 - mix)
        fy = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
        tr=aftermodel.predict(fx)
        tr2=np.reshape(tr,newshape=(-1,23,4,1))
        targety=model.predict(tr2)
        for k in range(batch_size):
            sq[k]=abs(fy[k]-targety[k])*abs(fy[k]-targety[k])
            sd[k]=-1*np.linalg.norm(input_x[k]-fx[k],ord=2)
        
        sq=min_max_normalize(sq)
        sd=min_max_normalize(sd)
        sq=np.array(sq)
        sd=np.array(sd)
        st=sq+sd
        #st=np.array(st)
        indexe=st.argsort()
        j=batch_size/10
        j=int(j)
        indexf=indexe[:j]
        fy=np.reshape(fy,newshape=(-1,1))
        gx=tr[indexf]
        gy=fy[indexf]
        xmix.append(gx)
        ymix.append(gy)
    
    xmix=np.concatenate(xmix)
    ymix=np.concatenate(ymix)
    ymix=np.reshape(ymix,newshape=(-1,))
    return xmix, ymix

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


