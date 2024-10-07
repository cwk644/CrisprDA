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

import tensorflow as tf
import os
import numpy as np
import pandas as pd

from dataag import *
from utils import *
from read import *
from sklearn.model_selection import train_test_split
import sklearn
import Transformer as tr

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

def transformer_decoder():
    
    input = Input(shape=(23,4,),name="inputinput")
    
    
    Embedd1=Dense(64,activation="relu")(input)

    sample_transformer2 = Transformer(4, 128, 50, 64, 8, 111, rate=0.1)
    x2, attention_weights2 = sample_transformer2(Embedd1, Embedd1, training=False, encoding_padding_mask=None,
                                              decoder_mask=None, encoder_decoder_padding_mask=None)
    
    
    
    f=Flatten()(x2)
    
    
    d1=Dense(512,name='middle')(f)
    
    d2=Dense(2944,name="m3")(d1)
    
    f2=tf.reshape(d2,(-1,23,128))
    
    sample_transformer = Transformer(4, 128, 50, 128, 8, 111, rate=0.1)
    x, attention_weights = sample_transformer(f2,f2, training=False, encoding_padding_mask=None,
                                              decoder_mask=None, encoder_decoder_padding_mask=None)



    dense1 = Dense(64,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense2")(x)

    dense2 = Dense(32, activation="relu", name="dense3")(dense1)

    dense3=Dense(4,activation='softmax',name='dense4')(dense2)

    final=Flatten()(dense3)
    
    model = Model(inputs=[input], outputs=[final])
    return model


def Decoder():
    
    input= Input(shape=(512,))
    
    d2=Dense(2944,name="m3")(input)
    
    f2=tf.reshape(d2,(-1,23,128))
    
    sample_transformer = Transformer(4, 128, 50, 128, 8, 111, rate=0.1)
    x, attention_weights = sample_transformer(f2,f2, training=False, encoding_padding_mask=None,
                                              decoder_mask=None, encoder_decoder_padding_mask=None)
    

    
    dense1 = Dense(64,
                       kernel_regularizer=regularizers.l2(1e-4),
                       bias_regularizer=regularizers.l2(1e-4),
                       activation="relu",
                       name="dense2")(x)
    
    dense2 = Dense(32, activation="relu", name="dense3")(dense1)
    
    dense3=Dense(4,activation='softmax',name='dense4')(dense2)
    
    final=Flatten()(dense3)
    
    model = Model(inputs=[input], outputs=[final])
    return model


def Readbenchmark():
    path='./Datasets/benchmark.csv'
    f_csv=csv.reader(open(path))
    data=[]
    label=[]
    for row in f_csv:
        if (len(row[0])==23):
            data.append(row[0])
            label.append(row[1])
    return data,label



def train_decoder():

    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    datasets=['WT','eSp','HF1','xCas','SniperCas','SniperCas','HypaCas9','SpCas9']
    
    '''
    train_data=calculate()
    np.save("./calculate.npy",train_data)
    '''
    train_data=np.load("./calculate.npy")

    
    train_label=convert_one_hot(train_data)

    test_label=convert_one_hot(train_data)

    # 46x4=184
    new_train_label=np.reshape(train_label,newshape=(-1,92))
    new_test_label=np.reshape(test_label,newshape=(-1,92))
    
    train_data=train_label
    test_data=test_label
    
    result=Result()
    result.Best=999

    #decoder=decoder2()
    decoder=transformer_decoder()
    #decoder.summary()
    
    decoder.summary()
    batch_size=100
    epochs=50
    learning_rate=0.0001
    
    decoder.compile(loss='mse'  , optimizer=Adam(learning_rate=learning_rate))
    
    decoder.summary()
    #test mode
    
    #test model
    #return acc_rate(test_data,decoder)

    #decoder.load_weights(path3)

    #print("test_data's acc is:" ,acc_rate(test_data,decoder))


    
    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_testmse(decoder,train_data, 
                                                                   new_train_label,result,path3
                                                              )))

    decoder.fit(train_data, new_train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.1,
          callbacks=[batch_end_callback])
    return "OK"
    
def train_decoder_part2():

    
    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    
    decoder=transformer_decoder()
    decoder.load_weights(path3)
    batch_size=100
    epochs=50
    learning_rate=0.0001
    
    train_data=np.load("./calculate.npy")
    
    train_label=convert_one_hot(train_data)
    
    train_data=train_label
    
    train_label=np.reshape(train_label,newshape=(-1,92))
    
    m=Model(inputs=decoder.input,outputs=decoder.get_layer("middle").output)

    train_middle=m.predict(train_data)
    
    decoderpart2=Decoder()
    
    decoderpart2.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    before=Model(inputs=decoder.input,outputs=decoder.get_layer("middle").output)
    
    final_train_data=before.predict(train_data)
    
    new_train_label=np.reshape(train_label,newshape=(-1,92))
    
    new_test_label=new_train_label
    
    result=Result()
    result.Best=999
    
    batch_end_callback2 = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_testmse(decoderpart2,
                                                                   final_train_data, 
                                                                   new_test_label,result,
                                                                   path3_5
                                                                )))
        
    decoderpart2.fit(final_train_data, train_label,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1,
              callbacks=[batch_end_callback2])
    return "OK"
if __name__ == "__main__":
    a1=train_decoder()
    a2=train_decoder_part2()