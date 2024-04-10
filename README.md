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

1. run and test
```
python anothertry.py
```
