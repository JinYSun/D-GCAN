

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:09:31 2022

@author:Jinyu-Sun
"""

import timeit
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import preprocess as pp
import pandas as pd
import matplotlib.pyplot as plt
from D_GCAN import MolecularGraphNeuralNetwork,Trainer,Tester

def predict (test_name,property,   radius = 1,
    dim = 52 ,
    layer_hidden = 4,
    layer_output = 10,
    dropout = 0.45,
    batch_train = 8,
    batch_test = 8,
    lr =3e-4,
    lr_decay = 0.85,
    decay_interval = 25 ,
    iteration = 140,
    N = 5000):
 
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration, dropout) = map(int, [radius, dim, layer_hidden, layer_output,
                                     batch_train, batch_test,
                                     decay_interval, iteration, dropout])
                                     
    lr, lr_decay = map(float, [lr, lr_decay])
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')

    lr, lr_decay = map(float, [lr, lr_decay])

    path = ''
    dataname = ''

    torch.manual_seed(0)
    model = MolecularGraphNeuralNetwork(
        N, dim, layer_hidden, layer_output, dropout).to(device)
    models=torch.load('model/model.h5')
    
#    models.eval()
    model.load_state_dict(models)
    model.eval()
    tester = Tester(model,batch_test)
    dataset_dev=pp.create_testdataset(test_name, path, dataname,property)
    np.random.seed(0)
    np.random.shuffle(dataset_dev)

   
    prediction_dev, loss_dev, dev_res =  tester.test_classifier(dataset_dev)


    if property == True:
        
        res_dev  = dev_res.T
        cnd_matrix=confusion_matrix(res_dev[:,0], res_dev[:,1])
        cnd_matrix
    
        tn2 = cnd_matrix[0,0]
        tp2 = cnd_matrix[1,1]
        fn2 = cnd_matrix[1,0]
        fp2 = cnd_matrix[0,1]
    
    
        bacc_dev = ((tp2/(tp2+fn2))+(tn2/(tn2+fp2)))/2#balance accurance
        pre_dev= tp2/(tp2+fp2)#precision/q+
        rec_dev = tp2/(tp2+fn2)#recall/se
        sp_dev=tn2/(tn2+fp2)
        q__dev=tn2/(tn2+fn2)
        f1_dev = 2*pre_dev*rec_dev/(pre_dev+rec_dev)#f1score
        mcc_dev = ((tp2*tn2) - (fp2*fn2))/math.sqrt((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2))#Matthews correlation coefficient
        acc_dev=(tp2+tn2)/(tp2+fp2+fn2+tn2)#accurancy
        #fpr_dev, tpr_dev, thresholds_dev =roc_curve(res_dev[:,0],res_dev[:,1])
        print('bacc_dev:',bacc_dev)
        print('pre_dev:',pre_dev)
        print('rec_dev:',rec_dev)
        print('f1_dev:',f1_dev)
        print('mcc_dev:',mcc_dev)
        print('sp_dev:',sp_dev)
        print('q__dev:',q__dev)
        print('acc_dev:',acc_dev)

    elif property == False:
        res_dev =  dev_res.T

    return res_dev
