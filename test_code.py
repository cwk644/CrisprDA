import Transformer as tr
from Transformer import Transformer
from Transformer import *
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Reshape, Lambda, Permute, Flatten, Dropout
from tensorflow.keras.layers import Embedding, Concatenate, Add
from tensorflow.keras.layers import Conv1D, AveragePooling1D,Conv2D
from tensorflow.keras.layers import Cropping1D,Embedding
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import Callback, LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import *
import concurrent.futures
import tensorflow as tf
import os
import numpy as np
import pandas as pd

from dataag import *
from utils import *
from read import *
from sklearn.model_selection import train_test_split
import math
from ParamsDetail import ModelParams_WT
#from main import transformer_decoder
#from main import Decoder
from decoder import transformer_decoder
from decoder import Decoder

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# params global
epochglobal=50
btsz=50

def get_position_embedding(x):
    len1=x.shape[0]
    len2=92
    res=np.zeros(shape=(len1,len2))
    for i in range(len1):
        for j in range(len2):
            res[i][j]=j+1
    res=np.array(res)
    return res

def leaky_relu(x, alpha=0.2):
    return tf.keras.layers.LeakyReLU(alpha=alpha)(x)

def ROPE(x):
    length=x.shape[1]
    dim=x.shape[2]
    cos=np.zeros(shape=(length,dim))
    sin=np.zeros(shape=(length,dim))
    xt=np.zeros(shape=(length,dim))
    for i in range(length):
        for j in range(dim):
            theta=math.pow(10000,(-2.0*(j//2))/(dim*1.0))
            cos[i][j]=math.cos(i*theta)
            sin[i][j]=math.sin(i*theta)
            xt[i][j]=math.pow(-1,j+1)
    
    res=x*cos+xt*x*sin
    return res



def tr_try(params):
    dropout_rate = params['dropout_rate']
    #dropout_rate=0.1
    # transformer module
    onehot_input=Input(shape=(23,4,1,),name='onehot_input')

    next_input=onehot_input
    next_output=Conv2D(256,(1,1),padding='same',activation=leaky_relu,name='next_output')(next_input)
    
    
    #1 transformer
    transformer_input=Conv2D(512,(1,4),activation=leaky_relu,padding='valid')(next_output)
    T=tf.reshape(transformer_input,shape=(-1,23,512))
    T=ROPE(T)
    
    sample_transformer = tr.MultiHeadAttention(512, 16)
    x, attention_weights = sample_transformer(T,T,T,None)
    
    x_l=tf.reshape(x,shape=(-1,23,1,512))
    x_l=tf.keras.layers.Conv2DTranspose(256, (1,4), activation=leaky_relu)(x_l)
    
    
    # 1 cnn
    final_conv=next_output
    cn_conv3=Conv2D(128,(1,1),padding='same',activation=leaky_relu)(final_conv)
    cn_conv4=Conv2D(128,(3,4),padding='same',activation=leaky_relu)(cn_conv3)
    cn_conv5=Conv2D(256,(1,1),padding='same',activation=leaky_relu)(cn_conv4)
    
    # 2 cnn
    
    cn_next1=Add()([cn_conv5,x_l])
    cn_conv13=Conv2D(128,(1,1),padding='same',activation=leaky_relu)(cn_next1)
    cn_conv14=Conv2D(128,(3,4),padding='same',activation=leaky_relu)(cn_conv13)
    cn_conv15=Conv2D(256,(1,1),padding='same',activation=leaky_relu)(cn_conv14)

    
    # 2 trans
    trans1=Conv2D(512,(1,4),activation=leaky_relu,padding='valid')(cn_conv14)
    trans1=tf.reshape(trans1,shape=(-1,23,512))
    trans1=Add()([x,trans1])
    trans1=ROPE(trans1)
    sample_transformer1 = tr.MultiHeadAttention(512, 16)
    trans1_output, attention_weights1 = sample_transformer1(trans1,trans1,trans1,None)
    

    
    trans2_output=tf.reshape(trans1_output,shape=(-1,23,512))
    trans2_output_s=tf.reshape(trans1_output,shape=(-1,23,1,512))

    trans2_output_s=tf.keras.layers.Conv2DTranspose(256, (1,4),activation=leaky_relu)(trans2_output_s)
    

    # 3 cnn
    cn_next3=Add()([cn_conv15,trans2_output_s])
    cn_conv31=Conv2D(128,(1,1),padding='same',activation=leaky_relu)(cn_next3)
    cn_conv32=Conv2D(128,(3,4),padding='same',activation=leaky_relu)(cn_conv31)
    cn_conv33=Conv2D(256,(1,1),padding='same',activation=leaky_relu)(cn_conv32)
    cn_3=cn_conv33
    
    
    left=Flatten()(cn_3)
    right=Flatten()(trans2_output)
    outputleft=Dense(500,activation=leaky_relu)(left)
    outputright=Dense(500,activation=leaky_relu)(right)
    finaloutput=Concatenate()([outputleft,outputright])
    
    channel_output2 = Flatten(name='final_features')(finaloutput)
    
    
    final=channel_output2
    

    dense1 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation=leaky_relu,
                   name="dense2")(final)
    drop1 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(64, activation=leaky_relu, name="dense3")(drop1)
    drop2 = Dropout(dropout_rate)(dense2)

    output = Dense(1, activation="linear", name="output")(drop2)
    
    model = Model(inputs=[onehot_input], outputs=[output])
    return model


def label_correction_pre(params,train_data,train_label,val_data,val_label,dataset):
    path1="model/"+dataset+".h5"
    pathtmp1="model/"+dataset+"tmp1.h5"
    pathtmp2="model/"+dataset+"tmp2.h5"
    m1=tr_try(params)
    m2=tr_try(params)
    batch_size=btsz
    epochs=40
    learning_rate=params['train_base_learning_rate']
    
    #train_data,val_data,train_label,val_label=train_test_split(train_data,train_label,test_size=0.1)


    m1.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    m2.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    result1=Result()
    result2=Result()
    
    batch_end_callback1 = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_test_weight(m1, train_data,train_label,result1,pathtmp1
                                                                )))
    batch_end_callback2 = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_test_weight(m2, train_data,train_label,result2,pathtmp2
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
    learning_rate=params['train_base_learning_rate']

    m1.load_weights(pathtmp1)
    m2.load_weights(pathtmp2)

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


def train(params, train_data, train_label,test_data, test_label,dataset,isMixup=False,isCNLC=False):
    
    #m = transformer_ont(params)
    #m = tr_try(params)
    m=tr_try(params)
    path1="CrisprDA/initial"+dataset+".h5"
    path2="CrisprDA/automix"+dataset+".h5"
    path3="CrisprDA/CNLC"+dataset+".h5"
    path4="CrisprDA/automix+cnlc"+dataset+".h5"

    #np.random.seed(1337)
    result=Result()
    #at_test_weight is used for Ubuntu.
    #batch_size=params['train_batch_size']
    #epochs=params['train_epochs_num']
    batch_size=50
    epochs=epochglobal
    learning_rate=params['train_base_learning_rate']
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    

    if ((isMixup==False) and (isCNLC==False)):
        m.load_weights(path1)
    elif (isMixup==True and isCNLC==False):
        m.load_weights(path2)
    elif (isMixup==False and isCNLC==True):
        m.load_weights(path3)
    else:
        m.load_weights(path4)
    
    test_pred=m.predict(test_data)
    resultspearman,resultpearson=get_spearman(test_pred,test_y)
    

    return resultspearman,resultpearson

if __name__ == "__main__":
    from ParamsDetail2 import ParamsDetail

    np.random.seed(1337)


    ModelParam=['ModelParams_WT','ModelParams_ESP','ModelParams_HF','ModelParams_xCas',
                 'ModelParams_SniperCas','ModelParams_SpCas9','ModelParams_HypaCas9']
    

    #datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]

   
    datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']

    for dataset in datasets:
        res=[222222]
        needtrain=True
        train_x,train_y,test_x,test_y= load_data_final(dataset)
        restmp="model/"+dataset+".npy"
        c=np.array(res)
        np.save(restmp,c)
        params=ParamsDetail[dataset]
        if (needtrain):
            ob,ob2=train(params,train_x,train_y,test_x,test_y,dataset,False,0)
            print("Spearmanscore:",ob, "Pearsonscore: ",ob2)
            res.append(ob)
            res.append(ob2)
            res.append(11111)
            c=np.array(res)
            np.save(restmp,c)
    




