# CrisprDA
An hyrbid architecture of CNN and transformer for sgRNA cleavage efficiency prediction, with data augmentation methods called Automix and CNLC.

## OS dependencies
+ Tensorflow-gpu==2.3.2
+ Python==3.9.0
+ keras==2.3.0
+ hyperopt==0.2.5
+ pandas=1.4.4
+ numpy=1.21.5

## Test demo with testsets
The model is included in anothertry.py and data augmentation methods is writed in dataag.py
**Note: network parameters of different variants are set in [ParamsDetail2.py](https://github.com/cwk644/CrisprDA/ParamsDetail2.py), and corresponding parameters need to be changed in [ParamsDetail2.py](https://github.com/cwk644/CrisprDA/ParamsDetail2.py)) after the test dataset is replaced. The default network parameter is corresponding to dataset WT-SpCas9.**

1. run pure net (with out Automix and CNLC)
```
python initial.py
```
2. test CrisprDA with combination of Automix and CNLC
```
python augmented.py
```
In addition, the code of Automix and CNLC are in dataag.py(https://github.com/cwk644/CrisprDA/dataag.py)

## Files and directories description
+ [ParamsDetail2.py](https://github.com/cwk644/CrisprDA/ParamsDetail2.py) the weights for the CrisprDA model trained by different datasets
+ [Transformer.py](https://github.com/cwk644/CrisprDA/Transformer.py) code for attention mechanisam and Transformer
+ [initial.py](https://github.com/cwk644/CrisprDA/initial.py) test of  CrisprDA with out data augmentation 
+ [augmented.py](https://github.com/cwk644/CrisprDA/augmented.py) test of CrisprDA with data augmentation of Automix and CNLC
+ [decoder.py](https://github.com/cwk644/CrisprDA/decoder.py) code for Auto encoder used in Automix
+ [dataag.py](https://github.com/cwk644/CrisprDA/dataag.py) code for Automix and CNLC
+ [utils.py](https://github.com/cwk644/CrisprDA/decoder.py) code for data preprocessing and evaluation metrics setting
+ [initial](https://github.com/cwk644/CrisprDA/initial) CrisprDA model for different datasets without data augmentation methods
+ [after](https://github.com/cwk644/CrisprDA/after) CrisprDA model for different datasets without combination of Automix and CNLC
+ [train](https://github.com/cwk644/CrisprDA/train) train datasets for CrisprDA
+ [val](https://github.com/cwk644/CrisprDA/val) validation datasets for CrisprDA
+ [test](https://github.com/cwk644/CrisprDA/test) test datasets for CrisprDA
