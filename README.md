<p align="center"><img src="https://user-images.githubusercontent.com/62410732/165705895-77c97081-7df2-402d-8199-29d1c33027d2.png" alt="logo" width="400px" /></p>
<h3 align="center">
<p> Prediction of Drug-likeness using Graph Convolutional Attention Network.<br></h3>
<h4 align="center">
-----------------------------------------------------------------------------------------------------------------



A deep learning method was developed to predict **d**rug-likeness based on the **g**raph **c**onvolutional **a**ttention **n**etwork (D-GCAN) directly from molecular structures. The model combined the advantages of graph convolution and attention mechanism. D-GCAN is a promising tool to predict drug-likeness for selecting potential candidates and accelerating the process of drug discovery by excluding unpromising candidates and avoiding unnecessary biological and clinical testing. 



![图片](https://user-images.githubusercontent.com/62410732/143736741-05e00f97-b01c-4130-8faa-562b51c0a4b4.png)




## Motivation

The drug-likeness has been widely used as a criterion to distinguish drug-like molecules from non-drugs. Developing reliable computational methods to predict drug-likeness of candidate compounds is crucial to triage unpromising molecules and accelerate the drug discovery process.




## Depends

[Anaconda for python 3.8](https://www.python.org/)

[conda install pytorch](https://pytorch.org/)

[conda install -c conda-forge rdkit](https://rdkit.org/)




## Discussion

The [Discussion](https://github.com/JinYSun/D-GCAN/tree/main/Discussion) folder contains the scripts for evaluating the classification performance.  We compared sevaral common methods widely used in drug-likeness prediction, such as [GNN](https://github.com/JinYSun/D-GCAN/tree/main/Discussion/GNN.py),[RF](https://github.com/JinYSun/D-GCAN/tree/main/Discussion/GNN.py), [CNN](https://github.com/JinYSun/D-GCAN/tree/main/Discussion/RF.py),[SVC](https://github.com/JinYSun/D-GCAN/tree/main/Discussion/SVC.py),and [GPC](https://github.com/JinYSun/D-GCAN/tree/main/Discussion/GPC.py).




## Usage

If you want to retrain the model, please put the molecule's SMILES files in to data directory and run [D-GCAN](https://github.com/JinYSun/D-GCAN/tree/main/DGCAN/DGCAN.py). The test set can be replaced by changing the path. It is as simple as

```
import train
test = train.train('../dataset/bRo5.txt',  
    radius = 1,         
    dim = 52,         
    layer_hidden = 4,  
    layer_output = 10, 
    dropout = 0.45,   
    batch_train = 8,   
    batch_test = 8,   
    lr =3e-4,          
    lr_decay = 0.85,   
    decay_interval = 25,
    iteration = 140,    
    N = 5000,           
    dataset_train='../dataset/data_train.txt') 
```

If you want to make the prediction of druglikeness of unknown molecule, we provide the trained model to rapidly generation predictions

```
import predict
test = predict.predict('../dataset/bRo5.txt',
    radius = 1,
    property = True,   #True if drug-likeness is known 
    dim = 52 ,
    layer_hidden = 4,
    layer_output = 10,
    dropout = 0.45,
    batch_train = 8,
    batch_test = 8,
    lr = 3e-4,
    lr_decay = 0.85,
    decay_interval = 25 ,
    iteration = 140,
    N = 5000)

```

or you can run [run.py](https://github.com/JinYSun/D-GCAN/blob/main/DGCAN/run.py) and modify the hyperparameters of the neural network to optimize the model .

The D-GCAN-screened GDB-13 database [(S-GDB13)](https://doi.org/10.5281/zenodo.7054367) is a more drug-like database and can be used to find new drug candidates.

#### -Notice-

As described in paper, the prediction of drug-likeness was deeply influenced by the dataset, especially the negative set. If necessary, retrain the model on your dataset.



# Contact

Jinyu Sun E-mail: jinyusun@csu.edu.cn



# Cite


	@article{10.1093/bioinformatics/btac676,
	author = {Sun, Jinyu and Wen, Ming and Wang, Huabei and Ruan, Yuezhe and Yang, Qiong and Kang, Xiao and Zhang, Hailiang and Zhang, Zhimin and Lu, Hongmei},
	title = "{Prediction of Drug-likeness using Graph Convolutional Attention Network}",
	journal = {Bioinformatics},
	year = {2022},
	month = {10}
	}
