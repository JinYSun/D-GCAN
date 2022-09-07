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
from sklearn.metrics import roc_auc_score, roc_curve,auc
from sklearn.metrics import confusion_matrix
import preprocess as pp
import pandas as pd
import matplotlib.pyplot as plt
from DGCAN import MolecularGraphNeuralNetwork,Trainer,Tester

def metrics(cnf_matrix):
    '''Evaluation Metrics'''
    tn = cnf_matrix[0, 0]
    tp = cnf_matrix[1, 1]
    fn = cnf_matrix[1, 0]
    fp = cnf_matrix[0, 1]

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
    
def train (test_name, radius, dim, layer_hidden, layer_output, dropout, batch_train,
    batch_test, lr, lr_decay, decay_interval, iteration, N , dataset_train):
    
    dataset_test = test_name
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

    print('-' * 100)
    print('Just a moment......')
    print('-' * 100)
    path = ''
    dataname = ''
    
    dataset_train=   pp.create_dataset(dataset_train,path,dataname)
    #dataset_train,dataset_test = pp.split_dataset(dataset_train,0.9)
    #dataset_test=   pp.create_dataset(dataset_dev,path,dataname)    
    dataset_test= pp.create_dataset(dataset_test,path,dataname)
    np.random.seed(0)
    np.random.shuffle(dataset_train)
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))
    print('-' * 100)

    print('Creating a model.')
    torch.manual_seed(0)
    model = MolecularGraphNeuralNetwork(
        N, dim, layer_hidden, layer_output, dropout).to(device)
    trainer = Trainer(model,lr,batch_train)
    tester = Tester(model,batch_test)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    file_result = path + '../DGCAN/results/AUC' + '.txt'
    #    file_result = '../output/result--' + setting + '.txt'
    result = 'Epoch\tTime(sec)\tLoss_train\tLoss_test\tAUC_train\tAUC_test'
    file_test_result = path + 'test_prediction' + '.txt'
    file_predictions = path + 'train_prediction' + '.txt'
    file_model = '../DGCAN/model/model' + '.h5'
    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(0)

    start = timeit.default_timer()

    for epoch in range(iteration):
        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        # [‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]
        prediction_train, loss_train, train_res = trainer.train(dataset_train)
        prediction_test, loss_test, test_res = tester.test_classifier(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-' * 100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train, loss_test, prediction_train, prediction_test]))
        tester.save_result(result, file_result)
        tester.save_model(model, file_model)
        print(result)
    model.eval()
    prediction_test, loss_test, test_res = tester.test_classifier(dataset_test)
    res_test = test_res.T

    cnf_matrix = confusion_matrix(res_test[:, 0], res_test[:, 1])
    fpr, tpr, thresholds = roc_curve(res_test[:, 0], res_test[:, 1])
    AUC = auc(fpr, tpr)
    print('auc:',AUC)
    metrics(cnf_matrix)
    return res_test