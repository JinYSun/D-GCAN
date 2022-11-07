

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
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device('cuda')
   
else:
    device = torch.device('cpu')
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout 
        self.concat = concat
        self.in_features = in_features   #dim of input feature
        self.out_features = out_features #dim of output feature
        self.alpha = alpha               # negative_slope leakyrelu
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))       
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):   
        """
        input: input_feature [N, in_features] in_features indicates the number of elements of the input feature vector of the node
        adj: adjacency matrix of the graph dimension [N, N] non-zero is one, data structure basics
        """
        h = torch.mm(input, self.W)    # [N, out_features]
        N = h.size()[0]     #Number of nodes of the graph
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)     # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec =-9e10 *torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)    
        # indicates that if the adjacency matrix element is greater than 0, then the two nodes are connected and the attention factor at that position is retained.
        # Otherwise it is necessary to mask and set to a very small value, the reason is that this minimum value will be disregarded during softmax.
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)  
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        """
        n_heads indicates that there are several GAL layers, which are finally stitched together, similar to self-attention
        to extract features from different subspaces.
        """
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid,56, dropout=dropout, alpha=alpha, concat=False)
        self.nheads=nheads

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        #x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        
        z=torch.zeros_like(self.attentions[1](x, adj))
        for att in self.attentions:
            z=torch.add(z, att(x, adj))
        x = z/self.nheads 
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.softmax(x, dim=1)

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, dropout):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.layer_hidden=layer_hidden
        self.layer_output=layer_output
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])

        self.W_output = nn.ModuleList([nn.Linear(56,56) for _ in range(layer_output)])
        self.W_property = nn.Linear(56, 2)
       
        self.dropout = dropout
        self.alpha = 0.25
        self.nheads = 2
        self.attentions = GAT(dim, dim, dropout, alpha=self.alpha, nheads=self.nheads).to(device)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))

        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def gnn(self, inputs):
        """Cat or pad each input data for batch processing."""
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adj = self.pad(adjacencies, 0)
        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)

        for l in range(self.layer_hidden):
            hs = self.update(adj, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)
        """Attention layer"""
        molecular_vectors = self.attentions(fingerprint_vectors, adj)
        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(molecular_vectors, molecular_sizes)
        return Smiles, molecular_vectors

    def mlp(self, vectors):
        """Regressor based on multilayer perceptron."""
        for l in range(self.layer_output):
        
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = torch.sigmoid(self.W_property(vectors))
        return outputs

    def forward_classifier(self, data_batch):

        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])


        with torch.no_grad():
            Smiles, molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors)
          
        predicted_scores = predicted_scores.to('cpu').data.numpy()
        predicted_scores = [s[1] for s in predicted_scores]
        correct_labels = correct_labels.to('cpu').data.numpy()

        return Smiles,predicted_scores, correct_labels


class Tester(object):
    def __init__(self, model,batch_test):
        self.model = model
        self.batch_test=batch_test
    def test_classifier(self, dataset):
        N = len(dataset)
        SMILES, P, C = '', [], []
        for i in range(0, N, self.batch_test):
            data_batch = list(zip(*dataset[i:i + self.batch_test]))
            Smiles, predicted_scores, correct_labels = self.model.forward_classifier( data_batch)
            SMILES += ' '.join(Smiles) + ' '
           
            P.append(predicted_scores)
            C.append(correct_labels)
        SMILES = SMILES.strip().split()
        tru = np.concatenate(C)
        pre = np.concatenate(P)
        pred = [1 if i >0.15 else 0 for i in pre]
        #AUC = roc_auc_score(tru, pre)
        cnf_matrix=confusion_matrix(tru,pred)
        tn = cnf_matrix[0, 0]
        tp = cnf_matrix[1, 1]
        fn = cnf_matrix[1, 0]
        fp = cnf_matrix[0, 1]
        acc = (tp + tn) / (tp + fp + fn + tn)
        #  Tru=map(str,np.concatenate(C))
        #  Pre=map(str,np.concatenate(P))
        #  predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, Tru, Pre)])
        predictions = np.stack((tru, pred, pre))
        return acc,  predictions

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')

def dump_dictionary(dictionary, filename):
    with open('../DGCAN/model'+filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
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
    dataset_train = ('../dataset/data_train.txt') #training set

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
    prediction_dev,  dev_res =  tester.test_classifier(dataset_dev)
    if property == True:    
        res_dev  = dev_res.T
        cnd_matrix=confusion_matrix(res_dev[:,0], res_dev[:,1])
        cnd_matrix 
        metrics(cnd_matrix)
    elif property == False:
        res_dev =  dev_res.T[:,1]

    return res_dev
