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
from t_SNE import *
from sklearn.model_selection import train_test_split
import math

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# params global
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

def CrisprDA(params):
    dropout_rate = params['dropout_rate']
    dropout_rate = 0.4
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
    m1=CrisprDA(params)
    m2=CrisprDA(params)
    batch_size=50
    epochs=40
    learning_rate=params['train_base_learning_rate']
    
    #train_data,val_data,train_label,val_label=train_test_split(train_data,train_label,test_size=0.1)

    train_data,val_data,train_label,val_label=train_test_split(train_data,train_label,test_size=0.1)
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
    m1=CrisprDA(params)
    m2=CrisprDA(params)
    batch_size=50
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

class SharedPool:
    def __init__(self, max_len=30000):
        self.X, self.y = [], []
        self.max_len = max_len
    def add(self, x, y):
        self.X.extend(x); self.y.extend(y); 
        if len(self.X) > self.max_len:
            keep = np.random.choice(len(self.X), self.max_len, replace=False)
            self.X = [self.X[i] for i in keep]
            self.y = [self.y[i] for i in keep]

    def sample(self, k):
        if len(self.X) == 0: return np.array([]), np.array([]), np.array([])
        idx = np.random.choice(len(self.X), min(k, len(self.X)), replace=False)
        return np.array(self.X)[idx], np.array(self.y)[idx]
    
def train(params,train_data, train_label,val_data,val_label,test_data, test_label,dataset,alpha=0.0,isMixup=True,isEnsemble=True,
         isCNLC = False,isCNLC_t = 0):
    
    #m = transformer_ont(params)
    #m = tr_try(params)
    m=CrisprDA(params)
    path1="model/"+dataset+".h5"
    
    result=Result()
    batch_size=params['train_batch_size']
    epochs=params['train_epochs_num']
    #epochs = 0
    learning_rate=params['train_base_learning_rate']
    
    m.summary()
    print(train_data.shape)
    #train_data,val_data,train_label,val_label=train_test_split(train_data,train_label,test_size=0.1)
    
    enc = encoder_model()
    dec = decoder_model()
    alpha = 0.8
    selection_ratio_A = 0.45
    selection_ratio_B = 0.45
    selection_ratio_C = 0.1
    best_spearman = 0
    save_path = path1
    alpha_max = 0.9
    alpha_min = 0.1
    alpha_step = 0.1
    patience = 10

    pathA="model/"+dataset+"A.h5"
    pathB="model/"+dataset+"B.h5"
    pathC="model/"+dataset+"C.h5"
    
    modelA = CrisprDA(params)  
    modelB = CrisprDA(params)  
    modelC = CrisprDA(params) 
    
    modelA.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    modelB.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    modelC.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    resultA=Result()
    resultB=Result()
    resultC=Result()
    
    batch_end_callbackA = LambdaCallback(on_epoch_end=
                                    lambda batch, logs:
                                    print(get_score_at_test_weight(modelA, val_data, val_label,resultA,pathA
                                                       )))
    batch_end_callbackB = LambdaCallback(on_epoch_end=
                                lambda batch, logs:
                                print(get_score_at_test_weight(modelB, val_data, val_label,resultB,pathB
                                                   )))
    batch_end_callbackC = LambdaCallback(on_epoch_end=
                            lambda batch, logs:
                            print(get_score_at_test_weight(modelC, val_data, val_label,resultC,pathC
                                               )))
    
    hardpool = SharedPool()
    train_x_enh=[train_data,train_data,train_data]
    train_y_enh=[train_label,train_label,train_label]
    if (isEnsemble):
        modelA.fit(train_data,train_label,batch_size=batch_size,epochs=3,verbose=2,callbacks=batch_end_callbackA)
        modelB.fit(train_data,train_label,batch_size=batch_size,epochs=3,verbose=2,callbacks=batch_end_callbackB)
        modelC.fit(train_data,train_label,batch_size=batch_size,epochs=3,verbose=2,callbacks=batch_end_callbackC)
        train_x_middle, train_y_middle = filter_middle(train_data,train_label) 
        # not used in small dataset.
        #train_x_middle, train_y_middle = train_data,train_label
        
        for epoch in range(epochs):
            
            if (isMixup):
                x_aug_A, y_aug_A = Automix_three_methods(
                    train_x_middle, train_y_middle,
                    enc, dec,
                    alpha_A=alpha, alpha_B=alpha,alpha_C=alpha,
                    selection_ratio_A=selection_ratio_A,
                    selection_ratio_B=selection_ratio_B,
                    selection_ratio_C=selection_ratio_C
                )
                x_aug_B, y_aug_B = Automix_three_methods(
                    train_x_middle, train_y_middle,
                    enc, dec,
                    alpha_A=alpha, alpha_B=alpha,alpha_C=alpha,
                    selection_ratio_A=selection_ratio_B,
                    selection_ratio_B=selection_ratio_A,
                    selection_ratio_C=selection_ratio_C
                )

                x_aug_C, y_aug_C = Automix_three_methods(
                    train_x_middle, train_y_middle,
                    enc, dec,
                    alpha_A=alpha, alpha_B=alpha,alpha_C=alpha,
                    selection_ratio_A=selection_ratio_C,
                    selection_ratio_B=selection_ratio_B,
                    selection_ratio_C=selection_ratio_A
                )
                y_aug_A=label_correction(params,x_aug_A,y_aug_A,dataset,f=0.2,r=1.0)
                y_aug_B=label_correction(params,x_aug_A,y_aug_A,dataset,f=0.2,r=1.0)
                y_aug_C=label_correction(params,x_aug_C,y_aug_C,dataset,f=0.2,r=1.0)
                # small datasets not need C-mixup
                train_x_enh[0] = np.concatenate([train_data, x_aug_A], axis=0)
                train_y_enh[0] = np.concatenate([train_label, y_aug_A], axis=0)
                train_x_enh[1] = np.concatenate([train_data, x_aug_B], axis=0)
                train_y_enh[1] = np.concatenate([train_label, y_aug_B], axis=0)
                train_x_enh[2] = np.concatenate([train_data, x_aug_C], axis=0)
                train_y_enh[2] = np.concatenate([train_label, y_aug_C], axis=0)
            else:
                train_x_enh[0] = train_data
                train_y_enh[0] = train_label
                train_x_enh[1] = train_data
                train_y_enh[1] = train_label
                train_x_enh[2] = train_data
                train_y_enh[2] = train_label
            #visualize_tsne(m,train_data,train_label,train_x_enh,train_y_enh,epoch=epoch)
            #visualize_tsne(m,test_data,test_label,train_x_enh,train_y_enh,epoch=epoch)
            # --- 2) 
            
            
            modelA.fit(train_x_enh[0], train_y_enh[0],
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                     callbacks=batch_end_callbackA)

            modelB.fit(train_x_enh[1], train_y_enh[1],
              batch_size=batch_size,
              epochs=1,
              verbose=2,
             callbacks=batch_end_callbackB)

            modelC.fit(train_x_enh[2], train_y_enh[2],
              batch_size=batch_size,
              epochs=1,
              verbose=2,
             callbacks=batch_end_callbackC)
            
            # --- 3) 验证 ---

            y_pred_val = modelA.predict(val_data)
            rho_val = get_spearman(y_pred_val, val_label)
            print(f"Epoch {epoch+1}, alpha={alpha:.2f} -> Spearman val = {rho_val:.4f}")
            # early stopping 检查
            if rho_val > best_spearman:
                best_spearman = rho_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Spearman no improvement for {patience} epochs, stop at epoch {epoch+1}. best={best_spearman:.4f}")
                    break

            last_rho = rho_val
        modelA.load_weights(pathA)
        modelB.load_weights(pathB)
        modelC.load_weights(pathC)
        y_pred_test_A = modelA.predict(test_x)
        y_pred_test_B = modelB.predict(test_x)
        y_pred_test_C = modelC.predict(test_x)
        y_pred_test = (y_pred_test_A+y_pred_test_B+y_pred_test_C)/3.0
        # 最终对test集评估
        #y_pred_test = m.predict(test_x)
        test_spearman = get_spearman(y_pred_test, test_y)
        test_pearson = get_pearson(y_pred_test, test_y)
        print(f"Final test set Spearman={test_spearman:.4f}, Pearson={test_pearson:.4f}")
        return test_spearman, test_pearson
    else:
        if (isMixup):
            #visualize_tsne(m,train_data,train_label,train_data,train_label,epoch=0)
            m.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
            batch_end_callback = LambdaCallback(on_epoch_end=
                                    lambda batch, logs:
                                    print(get_score_at_test_weight(m, val_data, val_label,result,path1
                                                    )))

            #train_x_middle, train_y_middle = filter_middle(train_data,train_label) 
            # not used in small dataset.
            train_x_middle, train_y_middle = train_data,train_label
            x_aug_A, y_aug_A = Automix_three_methods(
                    train_x_middle, train_y_middle,
                    enc, dec,
                    alpha_A=alpha, alpha_B=alpha,alpha_C=alpha,
                    selection_ratio_A=0.0,
                    selection_ratio_B=1,
                    selection_ratio_C=0.0
                )
            train_x_enh_min = np.concatenate([train_data, x_aug_A], axis=0)
            train_y_enh_min = np.concatenate([train_label, y_aug_A], axis=0)
            m.load_weights(path1)
            visualize_tsne(m,train_data[:1000],train_label[:1000],train_x_enh_min[:1000],train_y_enh_min[:1000],epoch=0)
            m.fit(train_x_enh_min, train_y_enh_min,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  callbacks=[batch_end_callback])    
            
            m.load_weights(path1)
            y_pred_test=m.predict(test_data)
            test_spearman = get_spearman(y_pred_test, test_y)
            test_pearson = get_pearson(y_pred_test, test_y)
            print(f"Final test set Spearman={test_spearman:.4f}, Pearson={test_pearson:.4f}")
            return test_spearman, test_pearson
        else:
            m.load_weights(path1)
            visualize_tsne(m,train_data,train_label,train_data,train_label,epoch=0)
            #train_x_middle, train_y_middle = filter_middle(train_data,train_label) 
            # not used in small dataset.
            '''
            path1 = "model/"+dataset+str(isCNLC_t)+".h5"
            if (isCNLC):
                path1 = "model/"+dataset+"_CNLC_"+str(isCNLC_t)+".h5"
            '''
            '''
            train_x_middle, train_y_middle = train_data,train_label
            m.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
            batch_end_callback = LambdaCallback(on_epoch_end=
                                    lambda batch, logs:
                                    print(get_score_at_test_weight(m, val_data, val_label,result,path1
                                                    )))

            
            m.fit(train_data, train_label,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  callbacks=[batch_end_callback])    
            
            m.load_weights(path1)
            y_pred_test=m.predict(test_data)
            test_spearman = get_spearman(y_pred_test, test_y)
            test_pearson = get_pearson(y_pred_test, test_y)
            print(f"Final test set Spearman={test_spearman:.4f}, Pearson={test_pearson:.4f}")
            return test_spearman, test_pearson
            '''

    
if __name__ == "__main__":
    from ParamsDetail2 import ParamsDetail
    np.random.seed(1337)
    datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    #datasets=['SpCas9']
    #datasets=['WT','eSp','HF1','xCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    #datasets=['WT']
    datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']
    #datasets=['chari2015Train293T','hart2016-Hct1162lib1Avg','xu2015TrainHl60']
    #datasets=['chari2015Train293T']

    datasets=['WT']
    #datasets=['eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    for dataset in datasets:
        params=ParamsDetail[dataset]
        res=[222222]
        train_x,train_y,test_x,test_y= load_data_final(dataset)
        restmp="model/"+dataset+".npy"

        train_data,val_data,train_label,val_label=train_test_split(train_x,train_y,test_size=0.1)
        print(np.max(train_label),dataset)
        #label_correction_pre(params, train_x,train_y,val_data,val_label,dataset)
        #plot_spearman_distribution_line(train_label,dataset)
        #train_label = label_correction(params,train_data,train_label,dataset,f=0.6)

        for i in range(1,2):
            rate=i*1.0/10.0
            ob,ob2=train(params,train_data,train_label,val_data,val_label,test_x,test_y,dataset,rate,isMixup=False,isEnsemble=False)
            print("Spearmanscore:",ob, "Pearsonscore: ",ob2)
            res.append(ob)
            res.append(11111)
            c=np.array(res)
            np.save(restmp,c)
        




