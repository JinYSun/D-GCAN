# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:19:10 2022

@author: BM109X32G-10GPU-02
"""
import torch
import pandas as pd
import train
import predict
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import rdBase, Chem
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
from rdkit.Chem import Lipinski

tes = train.train('../dataset/bRo5.txt', #test set   
    radius = 1,        #hops of radius subgraph: 1, 2 
    dim = 52,          #dimension of graph convolution layers
    layer_hidden = 4,  #Number of graph convolution layers
    layer_output = 10, #Number of dense layers
    dropout = 0.45,    #drop out rate :0-1
    batch_train = 8,   # batch of training set
    batch_test = 8,    #batch of test set
    lr =3e-4,          #learning rate: 1e-5,1e-4,3e-4, 5e-4, 1e-3, 3e-3,5e-3
    lr_decay = 0.85,   #Learning rate decay:0.5, 0.75, 0.85, 0.9
    decay_interval = 25,#Number of iterations for learning rate decay:10,25,30,50
    iteration = 140,    #Number of iterations 
    N = 5000,           #length of embedding: 2000,3000,5000,7000 
    dataset_train='../dataset/data_train.txt') #training set

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
