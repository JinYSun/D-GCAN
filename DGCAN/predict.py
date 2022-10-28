

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
from DGCAN import MolecularGraphNeuralNetwork,Trainer,Tester
def metrics(cnd_matrix):
    '''Evaluation Metrics'''
    
    tn = cnd_matrix[0, 0]
    tp = cnd_matrix[1, 1]
    fn = cnd_matrix[1, 0]
    fp = cnd_matrix[0, 1]

    bacc = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2  # balance accurance
    pre = tp / (tp + fp)  # precision/q+
    rec = tp / (tp + fn)  # recall/se
    sp = tn / (tn + fp)
    q_ = tn / (tn + fn)
    f1 = 2 * pre * rec / (pre + rec)  # f1score
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))  # Matthews correlation coefficient
    acc = (tp + tn) / (tp + fp + fn + tn)  # accurancy
    
    print('bacc:', bacc)
    print('pre:', pre)
    print('rec:', rec)
    print('f1:', f1)
    print('mcc:', mcc)
    print('sp:', sp)
    print('q_:', q_)
    print('acc:', acc)
    
    
def predict (test_name, property, radius, dim, layer_hidden, layer_output, dropout, batch_train,
    batch_test, lr, lr_decay, decay_interval, iteration, N):
    '''
    
    Parameters
    ----------
    data_test = '../dataset/data_test.txt', #test set   
    radius = 1,        #hops of radius subgraph: 1, 2 
    dim = 64,          #dimension of graph convolution layers
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
    dataset_train = '../dataset/data_train.txt') #training set

    Returns
    -------
    res_dev 
    Predicting results

    '''
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
    model.load_state_dict(torch.load(r'model/model.pth'))
    model.eval()
    tester = Tester(model,batch_test)
    dataset_dev=pp.create_testdataset(test_name, path, dataname,property)
    np.random.seed(0)
    #np.random.shuffle(dataset_dev)  
    prediction_dev, loss_dev, dev_res =  tester.test_classifier(dataset_dev)
    if property == True:    
        res_dev  = dev_res.T
        cnd_matrix=confusion_matrix(res_dev[:,0], res_dev[:,1])
        cnd_matrix 
        metrics(cnd_matrix)
    elif property == False:
        res_dev =  dev_res.T[:,1]

    return res_dev
