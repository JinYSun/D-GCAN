# D-GAT
The D-GAT can predict the druglikeness with the GAT-learned representations. It takes molecular graph as the input, and the predicted results as the output.


Motivation

The drug-likeness is an essential criterion to distinguish drug-like molecules from non-drugs. It can be used to assist the selection of lead compounds with higher quality in early stages of drug discovery and improve the success rate of drug development. Therefore, scientists are in urgent need of advanced machine learning methods to predict the drug-likeness from molecular structure with high accuracy and fast speed.
Depends

Anaconda for python 3.8

conda install pytorch

conda install -c conda-forge rdkit

1. Discussion

The Discussion folder contains the scripts for evaluating the classification performance.  We compared RF,GPC, CNN,GNN,SVC.

Usage

If you want to make the prediction of druglikeness of unknown molecule, please put the molecule's SMILES files in to data directory and run preprocess.py and D-gat.py
