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



os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


globalepoch=1
def transformer_ont(params):
    dropout_rate = 0.2
    
    input = Input(shape=(72,))
    input_nuc = input[:, :24]
    input_dimer = input[:, 24:48]
    input_pos = input[:, 48:72]
    
    embedded_nuc = Embedding(30, params['nuc_embedding_outputdim'], input_length=24)(input_nuc)
    embedded_dimer = Embedding(30, params['nuc_embedding_outputdim'], input_length=24)(input_dimer)

    conv1_nuc = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], padding='same', activation="relu", name="conv1_nuc")(embedded_nuc)
    conv1_dimer = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], padding='same', activation="relu", name="conv1_dimer")(embedded_dimer)

    pool1_nuc = AveragePooling1D(1, padding='same')(conv1_nuc)
    drop1_nuc = Dropout(dropout_rate)(pool1_nuc)
    pool1_dimer = AveragePooling1D(1, padding='same')(conv1_dimer)
    drop1_dimer = Dropout(dropout_rate)(pool1_dimer)

    pool_seq = Add()([pool1_nuc, pool1_dimer])
    drop_seq = Add()([drop1_nuc, drop1_dimer])

    emd_pos = Embedding(30, params['conv1d_filters_num'], input_length=24)(input_pos)
    
    pool1 = Add()([pool_seq, emd_pos])
    drop1 = Add()([drop_seq, emd_pos])
    

    #pool2 = Add()([pool1_dimer, emd_pos])
    #drop2 = Add()([drop1_dimer, emd_pos])
    
    
    conv2 = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], activation="relu", name="conv2")(pool1)
    conv3 = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], activation="relu", name="conv3")(drop1)
    

    sample_transformer = Transformer(params['transformer_num_layers'], params['transformer_final_fn'], 50, params['conv1d_filters_num'], 8, params['transformer_ffn_1stlayer'], rate=0.1)
    x, attention_weights = sample_transformer(conv3, conv2, training=False, encoding_padding_mask=None,
                                              decoder_mask=None, encoder_decoder_padding_mask=None)
    
    
    '''
    my_concat = Lambda(lambda x: Concatenate(axis=1)([x[0], x[1]]))
    weight_1 = Lambda(lambda x: x * 0.2)
    weight_2 = Lambda(lambda x: x * 0.8)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(x)
    flat = my_concat([weight_1(flat1), weight_2(flat2)])
    '''
    flat=Flatten(name='Flatten')(x)
    dense1 = Dense(64,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense2")(flat)
    drop1 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(32, activation="relu", name="dense3")(drop1)
    drop2 = Dropout(dropout_rate)(dense2)

    output = Dense(1, activation="linear", name="output")(drop2)

    model = Model(inputs=[input], outputs=[output])
    return model

def transformer_decoder(params):
    dropout_rate = 0.2
    input = Input(shape=(23,4,),name="inputinput")
    
    
    Embedd1=Dense(64,activation="relu")(input)

    sample_transformer2 = Transformer(4, 128, 50, 64, 8, params['transformer_ffn_1stlayer'], rate=0.1)
    x2, attention_weights2 = sample_transformer2(Embedd1, Embedd1, training=False, encoding_padding_mask=None,
                                              decoder_mask=None, encoder_decoder_padding_mask=None)
    
    
    
    f=Flatten()(x2)
    
    
    d1=Dense(512,name='middle')(f)
    
    d2=Dense(2944,name="m3")(d1)
    
    f2=tf.reshape(d2,(-1,23,128))
    
    sample_transformer = Transformer(3, params['conv1d_filters_num'], 50, 128, 8, params['transformer_ffn_1stlayer'], rate=0.1)
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


def Decoder(params):
    dropout_rate = 0.2
    
    input= Input(shape=(512,))
    


    sample_transformer2 = Transformer(4, 128, 50, 64, 8, params['transformer_ffn_1stlayer'], rate=0.1)
                              
    
    d2=Dense(2944,name="m3")(input)
    
    f2=tf.reshape(d2,(-1,23,128))
    
    sample_transformer = Transformer(3, params['conv1d_filters_num'], 50, 128, 8, params['transformer_ffn_1stlayer'], rate=0.1)
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


def train(params, train_data, train_label, val_data,val_label,test_data, test_label,dataset):
    m = transformer_ont(params)
    
    path1="model/"+dataset+".h5"
    path2="model/"+dataset+"after.h5"
    #np.random.seed(1337)
    result=Result()
    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_test_weight(m, val_data, val_label,result,path1
                                                                )))
    #at_test_weight is used for Ubuntu.
    batch_size=50
    epochs=globalepoch
    learning_rate=params['train_base_learning_rate']
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    m.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    #m.load_weights(path1)
    m.summary()
    
    

    m.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=[batch_end_callback])    

    m.load_weights(path1)
    test_pred=m.predict(test_data)
    result=get_spearman(test_pred,test_label)
    

    return result
  
def train_decoder(params):

    path3="model/"+"decoder.h5"
    path4="model/"+"decoderpart2.h5"
    
    datasets=['WT','eSp','HF1','xCas','SniperCas','SniperCas','HypaCas9','SpCas9']
    
    
    train_data=calculate()

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
    decoder=transformer_decoder(params)
    #decoder.summary()
    
    decoder.summary()
    batch_size=100
    epochs=50
    learning_rate=0.0001
    
    decoder.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    
    decoder.summary()
    #test mode
    
    #test model
    #return acc_rate(test_data,decoder)

    #decoder.load_weights(path3)

    #print("test_data's acc is:" ,acc_rate(test_data,decoder))



    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_testmse(decoder,test_data, new_test_label,result,path3
                                                              )))

    decoder.fit(train_data, new_train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.1,
          callbacks=[batch_end_callback])
    
    decoder.load_weights(path3)
    
    decoderpart2=Decoder(params)
    
    decoderpart2.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    
    before=Model(inputs=decoder.input,outputs=decoder.get_layer("middle").output)
    
    final_train_data=before.predict(train_data)
    
    batch_end_callback2 = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_testmse(decoderpart2,final_train_data, new_test_label,result,path4
                                                                )))
        
    decoderpart2.fit(final_train_data, new_train_label,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1,
              callbacks=[batch_end_callback2])

def total_train(params, train_data, train_label, val_data,val_label,test_data, test_label,dataset,params2,mixup,ismixup=True,isRem=True):
    
    
    path1="model/"+dataset+".h5"
    path2="model/"+dataset+"after.h5"
    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    path4="model/"+dataset+str(mixup)+"final.h5"
    #np.random.seed(1337)
    
    #m=transformer_decoder(params)

    m=transformer_decoder(params2)

    
    m.load_weights(path3)
    #m.summary()
    #return 1
    batch_size=50
    epochs=globalepoch
    learning_rate=params['train_base_learning_rate']
    #learning_rate=0.00001
    
    spec_t=convert_one_hot(train_data)
    #spec_t=one_hot_target(train_data)
    #spec_t=np.reshape(spec_t,newshape=(-1,92))

    new_model=transformer_ont(params)
    new_model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    new_model.summary()
    #new_model.load_weights(path3,by_name=True)
    # train 
    

    if (ismixup):
        before=Model(inputs=m.input,outputs=m.get_layer("middle").output)
        train_data_middle=before.predict(spec_t)
        
        #tx,ty=Almix(train_data_middle,train_label,1,mixup)
        after=Decoder(params2)
        after.load_weights(path3_5)
        if (isRem):
            smodel=transformer_ont(params)
            smodel.load_weights(path1)
            tx,ty=C_AL_mixup_REM(train_data_middle,train_label,smodel,after)
            splx=tx
        else:
            tx,ty=AL_mixup(train_data_middle,train_label,mixup)    
            final_train_data=after.predict(tx)
            
            splx=revise(final_train_data)
        
        '''
    
        
        #return splx
        '''
    #print(splx)
    #return splx
    if (ismixup):
        total_x=np.concatenate((splx,train_data))
        total_y=np.concatenate((ty,train_label))
    else:
        total_x=train_data
        total_y=train_label

    #if no mixup
    '''
    total_x=train_data
    total_y=train_label
    test_data=convert_one_hot(test_data)
    '''

    
    #new_model.load_weights(path1)
    result=Result()
    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch, logs:
                                        print(get_score_at_test_weight(new_model, val_data, val_label,result,path4
                                                                )))
    #new_model.load_weights(path4)
    
    new_model.fit(total_x, total_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1,
              callbacks=[batch_end_callback])


    new_model.load_weights(path4)
    test_pred=new_model.predict(test_data)
    result=get_spearman(test_pred,test_label)
    return result
    
    '''
    m = transformer_ont(params)
    
    path1="model/"+dataset+".h5"
    path2="model/"+dataset+"after.h5"
    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    path4="model/"+dataset+"final.h5"
    path5="model/"+dataset+"finetune.h5"
    #np.random.seed(1337)
    
    
    c=transformer_decoder(params)
    c.load_weights(path3)
    spec_train=convert_one_hot(train_data)
    spec_test=convert_one_hot(test_data)
    before=Model(inputs=c.input,outputs=c.get_layer("middle").output)
    

    after=Decoder(params)
    after.load_weights(path3_5)

    #after=Model(inputs=c.get_layer("middle").output,outputs=c.output)
    
    #return revise(after.predict(before.predict(spec_train)))
    print("train_data's acc is:" ,acc_rate_2(spec_train,before,after))
    print("test_data's acc is:" ,acc_rate_2(spec_test,before,after))

    
    m.load_weights(path1)
    
    
    print(get_score_at_testwt(m, test_data, test_label))
    
    m.load_weights(path4)
    
    print(get_score_at_testwt(m, test_data, test_label))
    '''

def train_pre(params, train_data, train_label, test_data, test_label,dataset):
    #path1="model/"+dataset+".h5"
    from sklearn.model_selection import KFold
    t_data=np.concatenate((train_data,test_data))
    t_label=np.concatenate((train_label,test_label))
    nums=1
    full_shape=t_data.shape[0]
    KF=KFold(n_splits=5)
    count=1
    batch_size=50
    epochs=globalepoch
    learning_rate=params['train_base_learning_rate']
    res=np.zeros(shape=(7,))
    models=[]
    for train_index,test_index in KF.split(t_data,t_label):
        path1="model/"+dataset+str(count)+".h5"
        new_model=transformer_ont(params)
        new_model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        result=Result()
        tmp_train_x=t_data[train_index]
        tmp_train_y=t_label[train_index]
        while (tmp_train_x.shape[0]<nums):
            tmp_data,tmp_label=C_mixup(tmp_train_x,tmp_train_y,0.2)
            tmp_train_x=np.concatenate((tmp_data,tmp_train_x))
            tmp_train_y=np.concatenate((tmp_label,tmp_train_y))
        tmp_test_x=t_data[test_index]
        tmp_test_y=t_label[test_index]
        batch_end_callback = LambdaCallback(on_epoch_end=
                                            lambda batch, logs:
                                            print(get_score_at_test_weight(new_model, tmp_test_x, tmp_test_y,result,path1
                                                                    )))

        new_model.fit(tmp_train_x, tmp_train_y,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      callbacks=[batch_end_callback])
        res[count]=result.Best
        count=count+1
        models.append(new_model.predict(train_data))
    # print(models)
    ct=(models[0]+models[1]+models[2]+models[3]+models[4])/5.0
    train_label=np.array(ct)
        
    path1="model"+dataset+"6"+".h5"
    new_model=transformer_ont(params)
    new_model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    result=Result()
    batch_end_callback = LambdaCallback(on_epoch_end=
                                            lambda batch, logs:
                                            print(get_score_at_test_weight(new_model, test_data, test_label,result,path1
                                                                    )))

    new_model.fit(train_data, train_label,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      callbacks=[batch_end_callback])
        
    res[6]=result.Best
    restmp="model/"+dataset+"waraorao.npy"
    print(res)
    np.save(restmp,res)
    
    train_label=np.reshape(train_label,newshape=(-1,))    
    return train_data,train_label

if __name__ == "__main__":
    from ParamsDetail import ModelParams_HF
    from ParamsDetail import ModelParams_SpCas9
    from ParamsDetail import ModelParams_xCas
    from ParamsDetail import ModelParams_WT
    from ParamsDetail import ParamsDetail
    np.random.seed(1337)
    decoderparams=ModelParams_WT
    # model = transformer_ont_biofeat(params)

    #  print("Loading weights for the models")
    #  model.load_weights("models/BestModel_WT_withbio.h5")

    ModelParam=['ModelParams_WT','ModelParams_ESP','ModelParams_HF','ModelParams_xCas',
                 'ModelParams_SniperCas','ModelParams_SpCas9','ModelParams_HypaCas9']
    
    #wawa=test(params,train_x,train_y,test_x,test_y,dataset)

    #train(params,train_x,train_y,test_x,test_y,dataset)


    #use one autoencoder-decoder for all datasets
    #train_decoder(decoderparams)
    #datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    
    '''
    datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg',
              'hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg',
              'xu2015TrainHl60']
    '''
    
    '''
    for dataset in datasets:
        dt="data/"+dataset+".npy"
        data = Readdata(dataset)
        x, y = process(data)
        
        nts=[]
        for i in x:
            tmp=Dimer_split_seqs(i)
            nts.append(tmp)
        nts = np.array(nts)
        
        np.save(dt,nts)
        
        
        
        nts=np.load(dt)
        nts=np.array(nts,dtype=('float64'))
        y = np.array(y, dtype='float64')
        
        train_x,train_y,val_x,val_y,test_x,test_y=get_split_dataset(nts,y)
        save_split(train_x, train_y, val_x, val_y, test_x, test_y, dataset)
    '''
    datasets=['WT']
    for dataset in datasets:
        res=[222222]
        needtrain=True
        train_x, test_x, val_x,val_y,train_y, test_y= load_data(dataset)
        restmp="model/"+dataset+".npy"
        c=np.array(res)
        np.save(restmp,c)
        params=ParamsDetail[dataset]
        if (needtrain):
            ob=train(params,train_x[:10],train_y[:10],val_x,val_y,test_x,test_y,dataset)
            res.append(ob)
            c=np.array(res)
            np.save(restmp,c)
        for i in range(1,11):
            print("Hello,world!")
            mixup=i*1.0/10.0
            #train_x,train_y=train_pre(params,train_x,train_y,val_x,val_y,dataset)
            res.append(111)
            #train_x,train_y=train_pre2(params,train_x,train_y,val_x,val_y,test_x,test_y,dataset)
            ob=total_train(params,train_x[:10],train_y[:10],val_x,val_y,test_x,test_y,dataset,decoderparams,mixup,ismixup=True,isRem=False)
            res.append(ob)
            c=np.array(res)
            np.save(restmp,c)
            print("complete!!!!!!")
    



