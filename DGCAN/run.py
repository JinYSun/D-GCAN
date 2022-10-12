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

tes = train.train('../dataset/data_test.txt',   
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


test = predict.predict('../dataset/beyondRo5.txt',
    radius = 1,
    property = True,  
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
