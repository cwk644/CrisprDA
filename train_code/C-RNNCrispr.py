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
def C_RNNCrispr():
    seq_input = Input(shape=(23, 4),name='onehot_input')
    seq_conv1 = Convolution1D(256, 5, kernel_initializer='random_uniform', name='seq_conv1')(seq_input)
    seq_act1 = Activation('relu')(seq_conv1)
    seq_pool1 = MaxPooling1D(2)(seq_act1)
    seq_drop1 = Dropout(0.2)(seq_pool1)
    gru1 = Bidirectional(GRU(256, kernel_initializer='he_normal', dropout=0.3, recurrent_dropout=0.2), name='gru1')(seq_drop1)
    seq_dense1 = Dense(256, name='seq_dense1')(gru1)
    seq_act2 = Activation('relu')(seq_dense1)
    seq_drop2 = Dropout(0.3)(seq_act2)
    seq_dense2 = Dense(128, name='seq_dense2')(seq_drop2)
    seq_act3 = Activation('relu')(seq_dense2)
    seq_drop3 = Dropout(0.2)(seq_act3)
    seq_dense3 = Dense(64, name='seq_dense3')(seq_drop3)
    seq_act4 = Activation('relu')(seq_dense3)
    seq_drop4 = Dropout(0.2)(seq_act4)
    seq_dense4 = Dense(40, name='seq_dense4')(seq_drop4)
    seq_act5 = Activation('relu')(seq_dense4)
    seq_drop5 = Dropout(0.2)(seq_act5)
    
    '''
    epi_input = Input(shape=(23, 4))
    epi_conv1 = Convolution1D(256, 5, name='epi_conv1')(epi_input)
    epi_act1 = Activation('relu')(epi_conv1)
    epi_pool1 = MaxPooling1D(2)(epi_act1)
    epi_drop1 = Dropout(0.3)(epi_pool1)
    epi_dense1 = Dense(256, name='epi_dense1')(epi_drop1)
    epi_act2 = Activation('relu')(epi_dense1)
    epi_drop2 = Dropout(0.2)(epi_act2)
    epi_dense2 = Dense(128, name='epi_dense2')(epi_drop2)
    epi_act3 = Activation('relu')(epi_dense2)
    epi_drop3 = Dropout(0.3)(epi_act3)
    epi_dense3 = Dense(64, name='epi_dense3')(epi_drop3)
    epi_act4 = Activation('relu')(epi_dense3)
    epi_drop4 = Dropout(0.3)(epi_act4)
    epi_act5 = Dense(40, name='epi_dense4')(epi_drop4)
    epi_out = Activation('relu')(epi_act5)
    '''
    #seq_epi_m = Multiply()([seq_drop5, epi_out])
    seq_epi_m=seq_drop5
    seq_epi_drop = Dropout(0.2)(seq_epi_m)
    seq_epi_flat = Flatten()(seq_epi_drop)
    seq_epi_output = Dense(1, activation='linear',name='output')(seq_epi_flat)

    #epi part was not used.
    model = Model(inputs=[seq_input], outputs=[seq_epi_output])
    
    return model
def train(train_x,train_y,test_x,test_y,dataset):
    

    model=C_RNNCrispr()
    #model.summary()
    path1="model/"+dataset+".h5"
    path2="model/"+dataset+"after.h5"
    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    path4="model/"+dataset+str(dataset)+"final.h5"
    path5="model/"+"decoderwithscore.h5"
    path6="model/"+"decoderfinalpart2.h5"
    TEST_N="./model/"+str(dataset)

    BATCH_SIZE=256
    EPOCHS=200
    #epochs=2 #for test
    learning_rate=0.005 #unknown in the original paper, thus use 0.005.
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))
    
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = callbacks.ModelCheckpoint(str(TEST_N) + '.model.best', verbose=1, save_best_only=True)
    
    train_data,val_data,train_label,val_label=train_test_split(train_x,train_y,test_size=0.1)
    history = model.fit(train_data,train_label, validation_data=(val_data,val_label), batch_size=BATCH_SIZE, \
        epochs=EPOCHS, use_multiprocessing=True, workers=16, verbose=2, callbacks=[es, mc])
    
    model=tf.saved_model.load(str(TEST_N) + '.model.best')
    test_x=test_x.astype('float32')
    test_pred=model.signatures['serving_default'](onehot_input=test_x)
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
        

    



