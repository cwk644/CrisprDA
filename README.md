# CrisprDA
Attention mechanism with CNN architecture model for sgRNA activity prediction

## OS dependencies
We use [tensorflow](https://www.tensorflow.org/) as the backend for training and testing.

[ViennaRNA](http://rna.tbi.univie.ac.at/) should be downloaded and installed in advance in order to capture the important biological features of sgRNA.

The required packages are:
+ python==3.6.9
+ tensorflow-gpu==2.5.0

## Tested demo with testsets
`python test_code.py`

## Files and directories description
+ [CrisprDA](https://github.com/cwk644/CrisprDA/tree/master/CrisprDA) the weights for the CrisprDA model trained by different datasets
+ [CRISPRON](https://github.com/cwk644/CrisprDA/tree/master/CRISPRON) the weights for the CRISPRon model trained by different datasets
+ [DeepCas9](https://github.com/cwk644/CrisprDA/tree/master/DeepCas9) the weights for the DeepCas9 model trained by different datasets
+ [DeepSpCas9](https://github.com/cwk644/CrisprDA/tree/master/DeepSpCas9) the weights for the DeepSpCas9 model trained by different datasets
+ [DeepHF](https://github.com/cwk644/CrisprDA/tree/master/DeepHF) the weights for the DeepHF model trained by different datasets
+ [C-RNNCrispr](https://github.com/cwk644/CrisprDA/tree/master/C-RNNCrispr) the weights for the C-RNNCrispr model trained by different datasets
+ CrisprDA, CRISPRon, DeepCas9, DeepSpCas9 and C-RNNCrispr include corresponding model trained with Automix, CNLC and the combination of Automix and CNLC besides the pure model
+ DeepHF includes Automix model and the pure model. Specially, Bayes tuning log is located in DeepHF/WT/, DeepHF/HF1/, DeepHF/eSp/
+ [new_test](https://github.com/cwk644/CrisprDA/tree/master/new_test) part of Automix_ensemble and t-test for CNLC

+ [data](https://github.com/cwk644/CrisprDA/tree/master/data) 17 sgRNA datasets used in this study,saved in numpy for training
+ [Datasets](https://github.com/cwk644/CrisprDA/tree/master/Datasets) sgRNA datasets saved in the form of csv
+ [model](https://github.com/cwk644/CrisprDA/tree/master/model) the weight for autoencoder in Automix
+ [train_code](https://github.com/cwk644/CrisprDA/tree/master/train_code) initial code for competing methods, which doesn't include the step for Automix and CNLC
+ [ParamDetail.py](https://github.com/cwk644/CrisprDA/tree/master/ParamDetail.py) parameters for CrisprDA
+ [ParamDetail2.py](https://github.com/cwk644/CrisprDA/tree/master/ParamDetail2.py) parameters for CrisprDA
+ [Transformer.py](https://github.com/cwk644/CrisprDA/tree/master/Transformer.py) the module of Transformer, which is the location of the function of attention mechanism
+ [dataag.py](https://github.com/cwk644/CrisprDA/tree/master/dataag.py) data augmentation method included in this study with some abandonded methods
+ [main.py](https://github.com/cwk644/CrisprDA/tree/master/main.py) CrisprDA main procedure, which includes each train step
+ [read.py](https://github.com/cwk644/CrisprDA/tree/master/read.py) pre_process for datasets, which transform csv into numpy
+ [test_code.py](https://github.com/cwk644/CrisprDA/tree/master/test_code.py) test_code for CrisprDA
+ [utils.py](https://github.com/cwk644/CrisprDA/tree/master/utils.py) code for evaluation metircs setting


## how to add new datasets
+ use the function split_and_save_dataset in [read.py](https://github.com/cwk644/CrisprDA/tree/master/read.py) to split the data and save numpy. Before the usage of function, datasets should be transformed into pre-set form of csv file according to the function

## how to use Automix and CNLC?
The result of experiment is multifarious. If you only want to use Automix,we collect the code in dataag_use.py by such followed function
```
augmented_dataset = augmix_reivse(input_x,input_y,encoder,decoder,alpha) (Automix)
augmented_dataset = Automix_three_methods(input_x, input_y, encoder, decoder,
                          # alpha for each method
                          alpha_A=0.4, alpha_B=0.4, alpha_C=0.4,
                          # ratio for each method
                          selection_ratio_A=0.3, 
                          selection_ratio_B=0.3,
                          selection_ratio_C=0.4,
                          seed_start=10, 
                          delta_c=0.1)  # Automix_ensemble

