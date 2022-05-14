<p align="center"><img src="https://user-images.githubusercontent.com/62410732/165705895-77c97081-7df2-402d-8199-29d1c33027d2.png" alt="logo" width="400px" /></p>
<h3 align="center">
<p> A deep learning method to predict drug-likeness based on the graph convolutional attention network (D-GCAN) directly from molecular structures.<br></h3>
<h4 align="center">
---

A deep learning method was developed to predict **d**rug-likeness based on the **g**raph **c**onvolutional **a**ttention **n**etwork (D-GCAN) directly from molecular structures. The model combined the advantages of graph convolution and attention mechanism. D-GCAN is a promising tool to predict drug-likeness for selecting potential candidates and accelerating the process of drug discovery by excluding unpromising candidates and avoiding unnecessary biological and clinical testing. 

![图片](https://user-images.githubusercontent.com/62410732/143736741-05e00f97-b01c-4130-8faa-562b51c0a4b4.png)


## Motivation

The drug-likeness has been widely used as a criterion to distinguish drug-like molecules from non-drugs. Developing reliable computational methods to predict drug-likeness of candidate compounds is crucial to triage unpromising molecules and accelerate the drug discovery process.

## Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)

## Discussion

The [Discussion](https://github.com/JinyuSun-csu/D-GCAN/tree/main/Discussion) folder contains the scripts for evaluating the classification performance.  We compared sevaral common methods widely used in drug-likeness prediction, such as [GNN](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/GNN.py),[RF](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/GNN.py), [CNN](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/RF.py),[SVC](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/SVC.py),and [GPC](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/GPC.py).

## Usage

If you want to retrain the model, please put the molecule's SMILES files in to data directory and run [D-GCAN](https://github.com/Jinyu-Sun1/D-GCAN/blob/main/main/D_GCAN.py). The test set can be replaced by changing the path. It is as simple as

```
import train

test = train.train('../dataset/data_test.txt')
```

If you want to make the prediction of druglikeness of unknown molecule, we provide the trained model to rapidly generation predictions

```
import predict

predict = predict.predict('../dataset/test.txt',property=False)
```

The D-GCAN-screened GDB-13 database [(S-GDB13)](https://doi.org/10.5281/zenodo.5700830) is a more drug-like database and can be used to find new drug candidates.

