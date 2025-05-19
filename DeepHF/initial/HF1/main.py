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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class GetBest(Callback):
    def __init__(self,filepath=None, monitor='val_loss', save_best=False,verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.save_best = save_best
        self.filepath = filepath
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                    #self.model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve.' %
                              (epoch + 1, self.monitor)) 
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f.' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)
        #self.model.save(self.filepath, overwrite=True)
        
fc_activation_dict = {'1':'relu','2':'tanh', '3':'sigmoid', '4':'hard_sigmoid', '0':'elu'}
initializer_dict = {'1':'lecun_uniform','2':'normal', '3':'he_normal', '0':'he_uniform'}
optimizer_dict = {'1':SGD,'2':RMSprop, '3':Adagrad, '4':Adadelta,'5':Adam,'6':Adamax,'0':Nadam}

def DeepHF_pre(x):
    lens=x.shape[0]
    k=np.zeros(shape=(lens,22))
    for i in range(lens):
        k[i][0]=1
        for j in range(21):
            k[i][j+1]=np.argmax(x[i][j])+2
    return k

def lstm_model(model_type='esp', batch_size=70, epochs=40, initializer='0',em_dim=40,em_drop=0.3,
                rnn_units=200, rnn_drop=0.6, rnn_rec_drop=0.2, fc_num_hidden_layers=6,
                fc_num_units=50, fc_drop=0.1,fc_activation='0',optimizer='6',learning_rate=0.001,
                validation_split=0.1,shuffle=False):
    dataset='HF1' #manaully.
    
    train_x, train_y, test_x, test_y= load_data_final(dataset)

    train_x=DeepHF_pre(train_x)
    test_x=DeepHF_pre(test_x)
    fc_activation = fc_activation_dict[str(fc_activation)]
    initializer = initializer_dict[str(initializer)]
    optimizer = optimizer_dict[str(optimizer)]
    sequence_input = Input(name = 'seq_input', shape = (22,))

    embedding_layer = Embedding(7,em_dim,input_length=22)
    embedded = embedding_layer(sequence_input)
    embedded = SpatialDropout1D(em_drop)(embedded)
    x = embedded

    #RNN
    lstm = LSTM(rnn_units, dropout=rnn_drop, 
                kernel_regularizer='l2',recurrent_regularizer='l2',
                recurrent_dropout=rnn_rec_drop, return_sequences=True)
    x = Bidirectional(lstm)(x)
    x = Flatten()(x)

    #biological featues
    #biological_input = Input(name = 'bio_input', shape = (11,))
    #x = keras.layers.concatenate([x, biological_input])


    for l in range(fc_num_hidden_layers):
        x = Dense(fc_num_units, activation=fc_activation)(x)
        x = Dropout(fc_drop)(x)
    #finish model
    mix_output = Dense(1, activation='linear',name='mix_output')(x)

    #model = Model(inputs=[sequence_input, biological_input], outputs=[mix_output])
    model = Model(inputs=[sequence_input], outputs=[mix_output])
    model.compile(loss='mse', optimizer=optimizer(lr=0.001))
    
    np.random.seed(1337)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    get_best_model = GetBest('models/' + model_type + '_rnn.hd5',monitor='val_loss', verbose=1, mode='min')
    
    

    

    

    model.fit([train_x], 
                 train_y,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.1,
                 shuffle=False,
                 callbacks=[get_best_model, early_stopping])    
    

    test_pred=model.predict(test_x)
    result=get_spearman(test_pred,test_y)
    
    mse=mean_squared_error(test_y, test_pred)
    
    print("Spearmanscore:" ,result)
    model.save_weights("./model/"+str(dataset)+str(batch_size)+str(epochs)+str(initializer)+str(em_dim)+str(em_drop)+str(rnn_units)
                       +str(rnn_drop)+str(rnn_rec_drop)+str(fc_num_hidden_layers)+str(fc_num_units)+str(fc_drop)+str(fc_activation)+str(optimizer)
                       +".h5")
    
    return result,mse

import logging
import GPy
import GPyOpt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from dotmap import DotMap

def cat_decode(x):
    opt= DotMap()
    opt.em_drop = float(x[:, 0])
    opt.rnn_drop = float(x[:, 1])
    opt.rnn_rec_drop = float(x[:, 2])
    opt.fc_drop = float(x[:, 3])
    
    opt.batch_size = int(x[:, 4])
    opt.epochs = int(x[:, 5]) 
    opt.em_dim =int(x[:, 6])
    opt.rnn_units = int(x[:, 7])
    opt.fc_num_hidden_layers = int(x[:, 8]) 
    opt.fc_num_units = int(x[:, 9])
    
    opt.fc_activation = int(x[:,10])
    opt.optimizer = int(x[:,11])
    return opt
    


#choose model to be trained
enzyme = 'HF1'

# create logger with 'Model_application'
logger = logging.getLogger('Model')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(enzyme + '_RNN_biofeat.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
def f(x):
    opt = cat_decode(x)
    param = {'model_type':enzyme,#mannualy set model_type as enzyme
            'em_drop':opt.em_drop,
           'rnn_drop':opt.rnn_drop,
           'rnn_rec_drop':opt.rnn_rec_drop,
           'fc_drop':opt.fc_drop,

           'batch_size':opt.batch_size,
           'epochs':opt.epochs, 
           'em_dim':opt.em_dim,
           'rnn_units':opt.rnn_units, 
           'fc_num_hidden_layers':opt.fc_num_hidden_layers, 
           'fc_num_units':opt.fc_num_units,

           'fc_activation':opt.fc_activation,
           'optimizer':opt.optimizer}
    
    
    res,evaluation = lstm_model(**param)
    logger.info('----------')
    logger.info('params:{}'.format(param))
    logger.info('spearman:{}'.format(res))
    return evaluation

if __name__ == "__main__":

    datasets=['WT','eSp','HF1','xCas','SniperCas','HypaCas9','SpCas9',"xCas","CRISPRON","HT_Cas9"]
    datasets=['HF1']
    '''
    datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg',
              'hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg',
              'xu2015TrainHl60']
    '''
    
    
    #for only RNN
    # WT
    '''
    em_drop=0.3
    rnn_drop=0.6
    rnn_rec_drop=0.2
    fc_drop=0.1
    batch_size=70
    epochs=40
    em_dim=40
    rnn_units=200
    fc_num_hidden_layers=5
    fc_num_units=50
    fc_activation='0'
    optimizer='6'
    '''
        
    #esp
    '''
     em_drop=0.2
     rnn_drop=0.2
     rnn_rec_drop=0.4
     fc_drop=0.4
     batch_size=70
     epochs=50
     em_dim=44
     rnn_units=70
     fc_num_hidden_layers=2
     fc_num_units=300
     fc_activation='3'
     optimizer='6'
    '''
    #hf1
    '''
     em_drop=0.2
     rnn_drop=0.5
     rnn_rec_drop=0.1
     fc_drop=0.4
     batch_size=80
     epochs=50
     em_dim=44
     rnn_units=80
     fc_num_hidden_layers=3
     fc_num_units=300
     fc_activation='3'
     optimizer='6'
    '''
    
    bounds = [
          #Discrete  
          {'name':'em_drop','type':'discrete','domain':(0.1,0.2,0.3)},
          {'name':'rnn_drop','type':'discrete','domain':(0.1,0.2,0.4,0.4,0.5,0.6,0.7)},
          {'name':'rnn_rec_drop','type':'discrete','domain':(0.1,0.2,0.4,0.4,0.5,0.6,0.7)},
          {'name':'fc_drop','type':'discrete','domain':(0.1,0.2,0.4,0.4,0.5,0.6,0.7)},
          #Discrete
          {'name':'batch_size', 'type': 'discrete','domain': (20,30, 40, 50,60, 70, 80,90, 100)},
          {'name':'epochs', 'type': 'discrete','domain': (20, 25, 30, 35, 40, 45, 50, 55)},
          {'name':'em_dim', 'type': 'discrete','domain': (16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68)},
          {'name':'rnn_units', 'type': 'discrete','domain': (20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240)},
          {'name':'fc_num_hidden_layers', 'type': 'discrete','domain': (1,2,3,4,5,6)},
          {'name':'fc_num_units', 'type': 'discrete','domain': (50, 60, 70,80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400)},
          #Categorical
          {'name':'fc_activation', 'type': 'categorical','domain':tuple(map(int,tuple(fc_activation_dict.keys())))},
          {'name':'optimizer', 'type': 'categorical','domain':tuple(map(int,tuple(optimizer_dict.keys())))}
         ]
    
    '''
    for dataset in datasets:
        res=[222222]
        needtrain=False
        restmp="model/"+dataset+".npy"
        c=np.array(res)
        np.save(restmp,c)
        print("Hello,world!")
        ob=lstm_model(dataset)
        res.append(ob)
        c=np.array(res)
        np.save(restmp,c)
    '''
    opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=30)
    opt_model.run_optimization(max_iter = 300)
    logger.info("optimized loss: {0}".format(opt_model.fx_opt))
    for i,v in enumerate(bounds):
        name = v['name']
        logger.info('parameter {}:{}'.format(name,opt_model.x_opt[i]))
            




