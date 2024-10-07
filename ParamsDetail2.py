ModelParams_WT = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.00002,
    'dropout_rate':0.4
    ,'f':0.9

}

ModelParams_ESP = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.00002,
    'dropout_rate':0.2
    ,'f':0.9
}

ModelParams_HF = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.00002,
    'dropout_rate':0.2
    ,'f':0.9

}

ModelParams_xCas = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.00002,
    'dropout_rate':0.2
    ,'f':0.9
}

ModelParams_SniperCas = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.00002,
    'dropout_rate':0.2
    ,'f':0.9
}

ModelParams_SpCas9 = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.00008,
    'dropout_rate':0.2
    ,'f':0.9
}

ModelParams_HypaCas9 = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.0001,
    'dropout_rate':0.2
    ,'f':0.9
}

ModelParams_CRISPRON = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.0001,
    'dropout_rate':0.2
    ,'f':0.9
}
ModelParams_HTCas9 = {
    'train_batch_size':50,
    'train_epochs_num':50,
    'train_base_learning_rate':0.0001,
    'dropout_rate':0.2
    ,'f':0.9
}

ModelParams_small= {
    'train_batch_size':20,
    'train_epochs_num':100,
    'train_base_learning_rate':0.0001,
    'dropout_rate':0.2
    ,'f':0.9
}

Params = {
    'ModelParams':ModelParams_SpCas9
    }

ParamsDetail={"WT":ModelParams_WT,
              "eSp":ModelParams_ESP,
              "HF1":ModelParams_HF,
              "xCas":ModelParams_xCas,
              "SniperCas":ModelParams_SniperCas,
              "SpCas9":ModelParams_SpCas9,
              "HypaCas9":ModelParams_HypaCas9,
              "CRISPRON":ModelParams_CRISPRON,
              "HT_Cas9":ModelParams_HTCas9,
              "doench2016_hg19":ModelParams_small,
              "doench2016plx_hg19":ModelParams_small,
              "hart2016-Hct1162lib1Avg":ModelParams_small,
              "hart2016-HelaLib1Avg":ModelParams_small,
              "hart2016-HelaLib2Avg":ModelParams_small,
              "hart2016-Rpe1Avg":ModelParams_small,
              "xu2015TrainHl60":ModelParams_small,
              "chari2015Train293T":ModelParams_small,
              'chari2015TrainK562':ModelParams_small,
              'doench2014-HS':ModelParams_small,
              'doench2014-Mm':ModelParams_small,
              'morenoMateos2015':ModelParams_small,
              'xu2015TrainKbm7':ModelParams_small
              }