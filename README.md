# D-GAT
The D-GAT can predict the druglikeness with the GAT-learned representations. It takes molecular graph as the input, and the predicted results as the output.


# Motivation

The drug-likeness is an essential criterion to distinguish drug-like molecules from non-drugs. It can be used to assist the selection of lead compounds with higher quality in early stages of drug discovery and improve the success rate of drug development. Therefore, scientists are in urgent need of advanced machine learning methods to predict the drug-likeness from molecular structure with high accuracy and fast speed.

# Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)

# Discussion

The [Discussion](https://github.com/JinyuSun-csu/D-GAT/tree/main/Discussion) folder contains the scripts for evaluating the classification performance.  We compared sevaral common methods widely used in drug-likeness prediction, such as [GNN](https://github.com/JinyuSun-csu/D-GAT/blob/main/Discussion/GNN.py),[RF](https://github.com/JinyuSun-csu/D-GAT/blob/main/Discussion/GNN.py), [CNN](https://github.com/JinyuSun-csu/D-GAT/blob/main/Discussion/RF.py),[SVC](https://github.com/JinyuSun-csu/D-GAT/blob/main/Discussion/SVC.py),and [GPC](https://github.com/JinyuSun-csu/D-GAT/blob/main/Discussion/GPC.py).

# Usage

If you want to make the prediction of druglikeness of unknown molecule, please put the molecule's SMILES files in to data directory and run [D-GAT](https://github.com/JinyuSun-csu/D-GAT/blob/main/model/D-GAT.py).
