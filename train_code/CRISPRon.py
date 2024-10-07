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

# check of inputs and setting a few base parameters

LEARN=0.0001 #as CRISPRon supplymentary table.7 shown
BATCH_SIZE=500  # as CRISPRon in his part of "The CRISPRon deep learning model"
EPOCHS=500 #the minimum of CRISPRon shown
OPT = 'adam'
if OPT =='adam':
    optimizer = optimizers.Adam(LEARN)
elif OPT == 'rmsprop':
    optimizer = optimizers.RMSprop(LEARN)
else:
    raise Exception

#length of input seq
eLENGTH=23 
# For the unity of test models. All models sgRNA inputs were 20nt+3PAM
#depth of onehot encoding
eDEPTH=4
def CRISPRon():
    input_c = Input(shape=(eLENGTH, eDEPTH,), name="input_onehot")
    
    for_dense = list()
    conv1_out = Conv1D(100, 3, activation='relu', input_shape=(eLENGTH,4,), name="conv_3")(input_c)
    conv1_dropout_out = Dropout(0.3, name="drop_3")(conv1_out)
    conv1_pool_out = AveragePooling1D(2, padding='SAME', name="pool_3")(conv1_dropout_out)
    conv1_flatten_out = Flatten(name="flatten_3")(conv1_pool_out)
    for_dense.append(conv1_flatten_out)

    #second convolution layer
    conv2_out = Conv1D(70, 5, activation='relu', input_shape=(eLENGTH,4,), name="conv_5")(input_c)
    conv2_dropout_out = Dropout(0.3, name="drop_5")(conv2_out)
    conv2_pool_out = AveragePooling1D(2, padding='SAME', name="pool_5")(conv2_dropout_out)
    conv2_flatten_out = Flatten(name="flatten_5")(conv2_pool_out)
    for_dense.append(conv2_flatten_out)

    #third convolution layer
    conv3_out = Conv1D(40, 7, activation='relu', input_shape=(eLENGTH,4,), name="conv_7")(input_c)
    conv3_dropout_out = Dropout(0.3, name="drop_7")(conv3_out)
    conv3_pool_out = AveragePooling1D(2, padding='SAME', name="pool_7")(conv3_dropout_out)
    conv3_flatten_out = Flatten(name="flatten_7")(conv3_pool_out)
    for_dense.append(conv3_flatten_out)

    #concatenation of conv layers and deltaGb layer
    if len(for_dense) == 1:
        concat_out = for_dense[0]
    else:
        concat_out = concatenate(for_dense)

    for_dense1 = list()
    
    #first dense (fully connected) layer
    dense0_out = Dense(80, activation='relu', name="dense_0")(concat_out)
    dense0_dropout_out = Dropout(0.3, name="drop_d0")(dense0_out)
    for_dense1.append(dense0_dropout_out)

    '''
    
    #Gb input used raw
    if TYPE.find('G') > -1:
        for_dense1.append(input_g)
    in our tests ,Gb was not used.
    '''

    if len(for_dense1) == 1:
        concat1_out = for_dense1[0]
    else:
        concat1_out = concatenate(for_dense1)


    #first dense (fully connected) layer
    dense1_out = Dense(80, activation='relu', name="dense_1")(concat1_out)
    dense1_dropout_out = Dropout(0.3, name="drop_d1")(dense1_out)
    
    #second dense (fully connected) layer
    dense2_out = Dense(60, activation='relu', name="dense_2")(dense1_dropout_out)
    dense2_dropout_out = Dropout(0.3, name="drop_d2")(dense2_out)
    
    #output layer
    output = Dense(1, name="output")(dense2_dropout_out)

    #model construction
    model= Model(inputs=input_c, outputs=[output])
    
    return model


def train(train_x,train_y,test_x,test_y,dataset,isMixup=False,alpha=0.0):

    TEST_N="./model/"+str(dataset)
    
    model=CRISPRon()
    model.summary()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #utils.plot_model(model, to_file=str(TEST_N) + '.model.png', show_shapes=True, dpi=600)


    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    train_data,val_data,train_label,val_label=train_test_split(train_x,train_y,test_size=0.1)
    if (isMixup):
        path3="model/"+"decoder.h5"
        path3_5="model/"+"decoderpart2.h5"
        m2=transformer_decoder()
        before=Model(inputs=m2.input,outputs=m2.get_layer("middle").output)
        m2.load_weights(path3)
        train_data_middle=before.predict(train_data)
        after=Decoder()
        after.load_weights(path3_5)

        #analysis(train_data,train_label, before, after)

        tx,ty=augmix(train_data,train_label,before,after,alpha)
        splx=np.reshape(tx,newshape=(-1,23,4))
        #ty=label_correction(params, revise(splx),ty, dataset,f=0.9,r=0.8)
        train_data=np.concatenate((splx,train_data))
        train_label=np.concatenate((ty,train_label))

        TEST_N=TEST_N+str(alpha)
    mc = callbacks.ModelCheckpoint(str(TEST_N) + '.model.best', verbose=1, save_best_only=True)
    train_data,train_label=DeepCRISPR(train_data,train_label)
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


def label_correction_pre(train_data,train_label,val_data,val_label,dataset):
    
    TEST_A="./model/"+str(dataset)+"tmp1"
    TEST_B="./model/"+str(dataset)+"tmp2"
    m1=CRISPRon()
    m2=CRISPRon()
    
    m1.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    m2.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    
    
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    mc1 = callbacks.ModelCheckpoint(str(TEST_A) + '.model.best', verbose=1, save_best_only=True)
    mc2 = callbacks.ModelCheckpoint(str(TEST_B) + '.model.best', verbose=1, save_best_only=True)
    
    history = m1.fit(train_data,train_label, validation_data=(train_data,train_label), batch_size=BATCH_SIZE, \
        epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc1])
    
    history2 = m2.fit(train_data,train_label, validation_data=(train_data,train_label), batch_size=BATCH_SIZE, \
        epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc2])
    
def label_correction(train_data, train_label,dataset,r=0.2
                     ,threshold=0.1,f=0.0):
    TEST_A="./model/"+str(dataset)+"tmp1"
    TEST_B="./model/"+str(dataset)+"tmp2"
    
    m1=tf.saved_model.load(str(TEST_A) + '.model.best')
    m2=tf.saved_model.load(str(TEST_B) + '.model.best')
    
    train_data=train_data.astype('float32')
    test_pred=m1.signatures['serving_default'](input_onehot=train_data)
    test_pred=np.array(test_pred['output'])

    res1=test_pred
    
    test_pred2=m2.signatures['serving_default'](input_onehot=train_data)
    test_pred2=np.array(test_pred2['output'])
    
    res2=test_pred2
    
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


def train_with_val(train_data, train_label,val_data,val_label,test_data, test_label,dataset,isMixup=False,alpha=0.0):
    

    
    model=CRISPRon()
    TEST_N="./model/"+str(dataset)
    TEST_A="./model/"+str(dataset)+"tmp1"
    TEST_B="./model/"+str(dataset)+"tmp2"
    

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #utils.plot_model(model, to_file=str(TEST_N) + '.model.png', show_shapes=True, dpi=600)

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
    if (isMixup):
        path3="model/"+"decoder.h5"
        path3_5="model/"+"decoderpart2.h5"
        m2=transformer_decoder()
        before=Model(inputs=m2.input,outputs=m2.get_layer("middle").output)
        m2.load_weights(path3)
        train_data_middle=before.predict(train_data)
        after=Decoder()
        after.load_weights(path3_5)

        #analysis(train_data,train_label, before, after)

        tx,ty=augmix(train_data,train_label,before,after,alpha)
        splx=np.reshape(tx,newshape=(-1,23,4))
        #ty=label_correction(params, revise(splx),ty, dataset,f=0.9,r=0.8)
        train_data=np.concatenate((splx,train_data))
        train_label=np.concatenate((ty,train_label))

    TEST_N=TEST_N+str(alpha)
    mc = callbacks.ModelCheckpoint(str(TEST_N) + '.model.best', verbose=1, save_best_only=True)
    #train_data,train_label=DeepCRISPR(train_data,train_label)
    history = model.fit(train_data,train_label, validation_data=(val_data,val_label), batch_size=BATCH_SIZE, \
        epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc])
    
    model=tf.saved_model.load(str(TEST_N) + '.model.best')
    test_x=test_data.astype('float32')
    test_pred=model.signatures['serving_default'](input_onehot=test_x)
    test_y=np.reshape(test_label,newshape=(-1,))
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
    datasets=['WT']
    datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    #datasets=['eSp']
    #datasets=['WT']
    
    
   
    datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']
    datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9",'chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']
    #datasets=['WT']
    
    for dataset in datasets:
        res=[222222]
        needtrain=False
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
        
        



