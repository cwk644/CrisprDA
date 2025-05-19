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



import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, Model, optimizers, Input, utils
from tensorflow.keras.layers import Conv1D, Dropout, AveragePooling1D, Flatten, Dense, concatenate, SpatialDropout1D
from scipy import stats
from random import randint
import sys

length=23 #23nt sgRNA
def Deep_xCas9(filter_size, filter_num, node_1=80, node_2=60, length=23):    
    input_seq = tf.keras.Input(shape=(1, length, 4), name='input_onehot')  # Assuming a fixed length  
  
    # 确定卷积核的数量  
    if isinstance(filter_num, int):  
        filters_to_use = filter_num * 3  
    elif isinstance(filter_num, (list, tuple)):  
        filters_to_use = filter_num[0] * 3  
    else:  
        raise ValueError("filter_num should be an integer or a sequence of integers.")  
  
    # 确定卷积核的大小  
    if isinstance(filter_size, int):  
        kernel_size_to_use = (1, filter_size)  
    elif isinstance(filter_size, (list, tuple)):  
        kernel_size_to_use = (1, filter_size[0])  
    else:  
        raise ValueError("filter_size should be an integer or a sequence of integers.")  
  
    # Convolutional layer  
    conv_layer = Conv2D(filters=filters_to_use, kernel_size=kernel_size_to_use, padding='valid', activation=None)(input_seq)  
    conv_layer = ReLU()(conv_layer)  
    conv_layer = Dropout(0.3)(conv_layer)  
  
    # Flatten layer  
    flatten_layer = Flatten()(conv_layer)  
  
    # Fully Connected Layer 1  
    fc1 = Dense(node_1, activation='relu')(flatten_layer)  # 直接使用字符串 'relu' 作为激活函数名  
    fc1 = Dropout(0.3)(fc1)  
  
    # Fully Connected Layer Feat（这部分在您提供的代码中被注释掉了，这里也保持注释）  
    # feat_transform = Dense(node_2, activation=None)(input_feat)  
    # combined = concatenate([fc1, feat_transform])  # 如果feat_transform被使用的话  
    # bio-feature was unused.
    # 如果没有feat_transform，则直接使用fc1  
    combined = fc1  
    #combined = ReLU()(combined)  
    combined = Dropout(0.3)(combined)  
  
    # Output Layer  
    output_layer = Dense(1, activation=None, name='output')(combined)  
  
    # Create the model  
    model = tf.keras.Model(inputs=[input_seq], outputs=output_layer)  
  
    return model


def train(train_x,train_y,test_x,test_y,dataset):
    

    model=Deep_xCas9(filter_size=3,filter_num=60)
    #model.summary()
    path1="model/"+dataset+".h5"
    path2="model/"+dataset+"after.h5"
    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    path4="model/"+dataset+str(dataset)+"final.h5"
    path5="model/"+"decoderwithscore.h5"
    path6="model/"+"decoderfinalpart2.h5"
    TEST_N="./model/"+str(dataset)
#np.random.seed(1337)
    
    train_x=np.reshape(train_x,newshape=(-1,1,23,4))
    test_x=np.reshape(test_x,newshape=(-1,1,23,4))
#at_test_weight is used for Ubuntu.
#batch_size=params['train_batch_size']
#epochs=params['train_epochs_num']
    BATCH_SIZE=500 
    EPOCHS=500
    #epochs=2 #for test
    #learning_rate=0.005
    learning_rate=0.0001 #special for WT,eSp,HF1. or Spearmanscore will be NaN
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = callbacks.ModelCheckpoint(str(TEST_N) + '.model.best', verbose=1, save_best_only=True)
    
    train_data,val_data,train_label,val_label=train_test_split(train_x,train_y,test_size=0.1)
    history = model.fit(train_data,train_label, validation_data=(val_data,val_label), batch_size=BATCH_SIZE, \
        epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc])
    
    model=tf.saved_model.load(str(TEST_N) + '.model.best')
    test_x=test_x.astype('float32')
    test_pred=model.signatures['serving_default'](input_onehot=test_x)
    test_y=np.reshape(test_y,newshape=(-1,))
    test_pred=np.array(test_pred['output'])
    test_pred=np.reshape(test_pred,newshape=(-1,))
    resultspearman,resultpearson=get_spearman(test_pred,test_y)
    return resultspearman,resultpearson
if __name__ == "__main__":
    from ParamsDetail2 import ParamsDetail

    np.random.seed(1337)
    # model = transformer_ont_biofeat(params)

    #  print("Loading weights for the models")
    #  model.load_weights("models/BestModel_WT_withbio.h5")

    ModelParam=['ModelParams_WT','ModelParams_ESP','ModelParams_HF','ModelParams_xCas',
                 'ModelParams_SniperCas','ModelParams_SpCas9','ModelParams_HypaCas9']
    
    #wawa=test(params,train_x,train_y,test_x,test_y,dataset)

    #train(params,train_x,train_y,test_x,test_y,dataset)


    #use one autoencoder-decoder for all datasets
    #train_decoder(decoderparams)
    datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    #datasets=['eSp']
    #datasets=['WT']
    
    
   
    #datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']
    datasets=['WT','eSp','HF1']
    for dataset in datasets:
        res=[222222]
        needtrain=True
        train_x,train_y,test_x,test_y= load_data_final(dataset)
        restmp="model/"+dataset+".npy"
        c=np.array(res)
        np.save(restmp,c)
        if (needtrain):
            ob,ob2=train(train_x,train_y,test_x,test_y,dataset)
            print("Spearmanscore:",ob, "Pearsonscore: ",ob2)
            res.append(ob)
            res.append(ob2)
            res.append(11111)
            c=np.array(res)
            np.save(restmp,c)
        

    



