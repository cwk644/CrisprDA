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
#from main import transformer_decoder
#from main import Decoder
from decoder import transformer_decoder
from decoder import Decoder

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# params global
epochglobal=50
btsz=50
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


def test(params, train_data, train_label, val_data,val_label,test_data, test_label,dataset):
    m=tr_try(params)
    path1="after/"+dataset+".h5"

    #np.random.seed(1337)

    #at_test_weight is used for Ubuntu.
    #batch_size=params['train_batch_size']
    #epochs=params['train_epochs_num']
    batch_size=50
    epochs=epochglobal
    learning_rate=params['train_base_learning_rate']
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    m.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    #m.load_weights(path1)
    #m.summary()
    
    train_data=convert_one_hot(train_data)
    #train_data=label_smooth(train_data)
    val_data=convert_one_hot(val_data)
    #val_data=label_smooth(val_data)
    test_data=convert_one_hot(test_data)
    #test_data=label_smooth(test_data)

 

    m.load_weights(path1)
    test_pred=m.predict(test_data)
    result=get_spearman(test_pred,test_label)
    

    return result
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
    #datasets=['WT']
    
    for dataset in datasets:
        train_x, test_x, val_x,val_y,train_y, test_y= load_data(dataset)
        params=ParamsDetail[dataset]
        result=test(params,train_x,train_y,val_x,val_y,test_x,test_y,dataset)
        print(dataset,":",result)
        

    



