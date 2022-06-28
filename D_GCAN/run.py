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

tes = train.train('../dataset/world_wide.txt')
# test = predict.predict('../dataset/world_wide.txt',property=True)
