# D-GCAN
The D-GCAN can predict the druglikeness with the graph neural network-learned representations. It takes molecular graph as the input, and the predicted results as the output.The D-GCAN is a promising tool to predict drug-likeness for selecting lead drug candidates, improving the accuracy of drug development and accelerating the drug discovery process.
![图片](https://user-images.githubusercontent.com/62410732/143736741-05e00f97-b01c-4130-8faa-562b51c0a4b4.png)


# Motivation

The drug-likeness is an essential criterion to distinguish drug-like molecules from non-drugs. It can be used to assist the selection of lead compounds with higher quality in early stages of drug discovery and improve the success rate of drug development. Therefore, scientists are in urgent need of advanced machine learning methods to predict the drug-likeness from molecular structure with high accuracy and fast speed.

# Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)

# Discussion

The [Discussion](https://github.com/JinyuSun-csu/D-GCAN/tree/main/Discussion) folder contains the scripts for evaluating the classification performance.  We compared sevaral common methods widely used in drug-likeness prediction, such as [GNN](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/GNN.py),[RF](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/GNN.py), [CNN](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/RF.py),[SVC](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/SVC.py),and [GPC](https://github.com/JinyuSun-csu/D-GCAN/blob/main/Discussion/GPC.py).

# Usage

If you want to make the prediction of druglikeness of unknown molecule, please put the molecule's SMILES files in to data directory and run [D-GCAN](https://github.com/JinyuSun-csu/D-GCAN/blob/main/model/D-GCAN.py).
