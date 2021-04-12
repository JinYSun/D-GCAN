import timeit

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
import matplotlib
import matplotlib.cm as cm
import pickle
from pathlib import Path

from numbers import Number
from IPython.display import SVG
from IPython.display import Image,display
from rdkit import Chem
from rdkit.Chem import rdDepictor,Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
import joblib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.in_features = in_features  
        self.out_features = out_features  
        self.alpha = alpha  
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))) 
        nn.init.xavier_uniform_(self.a.data, gain=1.414) 
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        h = torch.mm(input, self.W)
       
        N = h.size()[0]
  
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
       
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
       
        zero_vec = -9e15 * torch.ones_like(e)
        
        attention = torch.where(adj > 0, e, zero_vec)
       
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
        self.dropout = dropout
     
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
      
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, 54, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.softmax(x, dim=1)

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, dropout):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])

        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 2)

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

        Smiles, fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
      

        adj = self.pad(adjacencies, 0)
       # print(adj.size())
        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
         
        for l in range(layer_hidden):
            hs = self.update(adj, fingerprint_vectors, l)
            
          
        m= torch.mean(hs,dim=1)
        min = torch.min(m)
        max= torch.max(m)
        vectors = (m-min)/(max-min)
       # print(vectors)
        # molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        """Molecular vector by sum or mean of the fingerprint vectors."""
        
        molecular_vectors = self.attentions(fingerprint_vectors, adj)#cat adj
           
        molecular_vectors = self.sum(molecular_vectors, molecular_sizes)#BATCH*X
        
        return Smiles, molecular_vectors,vectors

    def mlp(self, vectors):
        """Classifier  based on multilayer perceptron给予多层感知器的分类器."""
        for l in range(layer_output):
            
            vectors = torch.relu(self.W_output[l](vectors))
            
        outputs = torch.sigmoid(self.W_property(vectors))
        return outputs

    def forward_classifier(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])

        if train:
            Smiles, molecular_vectors,vectors = self.gnn(inputs)

            predicted_scores = self.mlp(molecular_vectors)

            loss = F.cross_entropy(predicted_scores, correct_labels)
            predicted_scores = predicted_scores.to('cpu').data.numpy()
            predicted_scores = [s[1] for s in predicted_scores]

            correct_labels = correct_labels.to('cpu').data.numpy()

            return loss, predicted_scores, correct_labels
        else:
            with torch.no_grad():
                Smiles, molecular_vectors,vectors = self.gnn(inputs)
               
                predicted_scores = self.mlp(molecular_vectors)
                loss = F.cross_entropy(predicted_scores, correct_labels)
            predicted_scores = predicted_scores.to('cpu').data.numpy()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.to('cpu').data.numpy()

            return Smiles, loss, predicted_scores, correct_labels,vectors

        
class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        P, C = [], []
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i + batch_train]))
            loss, predicted_scores, correct_labels = self.model.forward_classifier(data_batch, train=True)

            P.append(predicted_scores)
            C.append(correct_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        tru = np.concatenate(C)
        pre = np.concatenate(P)
        AUC = roc_auc_score(tru, pre)
        pred = [1 if i > 0.4 else 0 for i in pre]
        predictions = np.stack((tru, pred, pre))
        return AUC, loss_total, predictions


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_classifier(self, dataset):
        N = len(dataset)
        loss_total = 0
        SMILES, P, C = '', [], []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i + batch_test]))
            (Smiles, loss, predicted_scores, correct_labels,vectors) = self.model.forward_classifier(
                data_batch, train=False)
           
            loss_total += loss.item()
            P.append(predicted_scores)
            C.append(correct_labels)
            
            zero = torch.zeros_like(vectors)
            watom=torch.where(vectors>0.1,vectors,zero)
            try:
                mol = (Chem.MolFromSmiles(Smiles[0]))
                for atom in mol.GetAtoms():
                    atomid = atom.GetIdx()
                radius={}
                atom_colors = {i: [(0.5,1-(watom[i].data.item()),1)] for i in range(0,atomid+1)}
                name=Smiles[0]+'    SCORE=' +str(predicted_scores[0])
                d2d = rdMolDraw2D.MolDraw2DCairo(900,500)
                d2d.DrawMoleculeWithHighlights(mol,name,atom_colors,atom_colors,radius,{})
                d2d.FinishDrawing()
                graph=d2d.WriteDrawingText(Smiles[0]+'.png')
                #display(graph)
                Image(Smiles[0]+'.png')
                   
                mol2 =Chem.MolFromSmiles(Smiles[1])
                for atom in mol2.GetAtoms():
                    atomid2 = atom.GetIdx()
                radius={}
                atom_colors2= {i: [(0.5,1-(watom[i+atomid+1].data.item()),1)] for i in range(0,atomid2+1)}
                name=Smiles[1]+'    SCORE=' +str(predicted_scores[1])
                d2d = rdMolDraw2D.MolDraw2DCairo(900,500)
                d2d.DrawMoleculeWithHighlights(mol2,name,atom_colors2,atom_colors2,radius,{})
                d2d.FinishDrawing()
                graph=d2d.WriteDrawingText(Smiles[1]+'.png')
             #   display(graph)
                Image(Smiles[1]+'.png')
            except:
                mol = (Chem.MolFromSmiles(Smiles[0]))
                for atom in mol.GetAtoms():
                    atomid = atom.GetIdx()
                radius={}
                atom_colors = {i: [(0.5,1-(watom[i].data.item()),1)] for i in range(0,atomid)}
                name=Smiles[0]+'    SCORE=' +str(predicted_scores[0])
                d2d = rdMolDraw2D.MolDraw2DCairo(900,500)
                d2d.DrawMoleculeWithHighlights(mol,name,atom_colors,atom_colors,radius,{})
                d2d.FinishDrawing()
                graph=d2d.WriteDrawingText(Smiles[0]+'.png')
                #display(graph)
                Image(Smiles[0]+'.png')
                   
                mol2 =Chem.MolFromSmiles(Smiles[1])
                for atom in mol2.GetAtoms():
                    atomid2 = atom.GetIdx()
                radius={}
                atom_colors2= {i: [(0.5,1-(watom[i+atomid+1].data.item()),1)] for i in range(0,atomid2)}
                name=Smiles[1]+'    SCORE=' +str(predicted_scores[1])
                d2d = rdMolDraw2D.MolDraw2DCairo(900,500)
                d2d.DrawMoleculeWithHighlights(mol2,name,atom_colors2,atom_colors2,radius,{})
                d2d.FinishDrawing()
                graph=d2d.WriteDrawingText(Smiles[1]+'.png')
             #   display(graph)
                Image(Smiles[1]+'.png')
        SMILES = SMILES.strip().split()
        tru = np.concatenate(C)
        
         
        #plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        pre = np.concatenate(P)
        AUC = roc_auc_score(tru, pre)
        pred = [1 if i > 0.4 else 0 for i in pre]
        Tru=map(str,np.concatenate(C))
        Pre=map(str,np.concatenate(P))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, Tru, Pre)])
        predictions = np.stack((tru, pred, pre))
        return Smiles, AUC, loss_total, predictions,vectors

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(111)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def edit_dataset(drug, non_drug, task):
    np.random.seed(111)  # fix the seed for shuffle.

    if task == 'balance':
        # np.random.shuffle(non_drug)
        non_drug = non_drug[0:len(drug)]

    else:
        np.random.shuffle(non_drug)
    np.random.shuffle(drug)
    dataset_train_drug, dataset_test_drug = split_dataset(drug, 0.9)
    #dataset_train_drug,dataset_dev_drug =  split_dataset(dataset_train_drug, 0.9)
    dataset_train_no, dataset_test_no = split_dataset(non_drug, 0.9)
   # dataset_train_no,dataset_dev_no =  split_dataset(dataset_train_no, 0.9)
    dataset_train = dataset_train_drug + dataset_train_no
    dataset_test = dataset_test_drug+ dataset_test_no
    #  dataset_dev = dataset_dev_drug+dataset_dev_no
    return dataset_train, dataset_test


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":

    radius = 1
    dim = 54
    layer_hidden = 7
    layer_output = 10
    dropout = 0.45
    batch_train = 10
    batch_test = 10
    lr = 3e-4
    lr_decay = 0.85
    decay_interval = 10
    iteration=120
    N = 5000
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration, dropout) = map(int, [radius, dim, layer_hidden, layer_output,
                                     batch_train, batch_test,
                                     decay_interval, iteration, dropout])
    lr, lr_decay = map(float, [lr, lr_decay])


    device = torch.device('cuda')
       
    print('-' * 100)

    #    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    print('-' * 100)
    path = 'E:/code/drug/'
    dataname = 'drugnn/'

    dataset_drug=pp.create_dataset('drug.txt', path, dataname)
    dataset_nondrug= pp.create_dataset('non-drug.txt', path, dataname)

    dataset_train, dataset_test = edit_dataset(dataset_drug, dataset_nondrug, 'balance')
    # dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    # print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-' * 100)

    print('Creating a model.')
    torch.manual_seed(111)
    model = MolecularGraphNeuralNetwork(
        N, dim, layer_hidden, layer_output, dropout).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    file_result = path + 'AUC' + '.txt'
    #    file_result = '../output/result--' + setting + '.txt'
    result = 'Epoch\tTime(sec)\tLoss_train\tLoss_test\tAUC_train\tAUC_test'
    file_test_result = path + 'test_prediction' + '.txt'
    file_predictions = path + 'train_prediction' + '.txt'
    
    file_model = path + 'model' + '.h5'
    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(111)

    start = timeit.default_timer()
   
    for epoch in range(iteration):
        
        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        # [‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]
        prediction_train, loss_train, train_res = trainer.train(dataset_train)
        
        
        # prediction_dev,dev_res = tester.test_classifier(dataset_dev)
        Smiles, prediction_test, loss_test, test_res,vectors  = tester.test_classifier(dataset_test)

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

        print(result)

    loss = pd.read_table(file_result)
    plt.plot(loss['Loss_train'], color='r', label='Loss of train set')
    plt.plot(loss['Loss_test'], color='y', label='Loss of train set')
    plt.plot(loss['AUC_train'], color='y', label='AUC of train set')
    plt.plot(loss['AUC_test'], color='b', label='AUC of test set')
    # plt.plot(loss['AUC_test'], color='y',label='AUC of test set')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(path + 'loss.tif', dpi=300)
    plt.show()
    colors = ['#00CED1', '#DC143C']

    target_names = np.array(['druglike', 'not-drug'])
    lw = 2
    res_test = test_res.T

    for color, i, target_name in zip(colors, [1, 0], target_names):
        plt.scatter((res_test[res_test[:, 0] == i, 0]), (res_test[res_test[:, 0] == i, 2]), color=color, alpha=.8,
                    lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('the results of gnn classification')
    res_train = train_res.T
    cn_matrix = confusion_matrix(res_train[:, 0], res_train[:, 1])
    cn_matrix

    tn1 = cn_matrix[0, 0]
    tp1 = cn_matrix[1, 1]
    fn1 = cn_matrix[1, 0]
    fp1 = cn_matrix[0, 1]

    bacc_train = ((tp1 / (tp1 + fn1)) + (tn1 / (tn1 + fp1))) / 2  # balance accurance
    pre_train = tp1 / (tp1 + fp1)  # precision/q+
    rec_train = tp1 / (tp1 + fn1)  # recall/se
    sp_train = tn1 / (tn1 + fp1)
    q__train = tn1 / (tn1 + fn1)
    f1_train = 2 * pre_train * rec_train / (pre_train + rec_train)  # f1score
    mcc_train = ((tp1 * tn1) - (fp1 * fn1)) / math.sqrt(
        (tp1 + fp1) * (tp1 + fn1) * (tn1 + fp1) * (tn1 + fn1))  # Matthews correlation coefficient
    acc_train = (tp1 + tn1) / (tp1 + fp1 + fn1 + tn1)  # accurancy
    fpr_train, tpr_train, thresholds_train = roc_curve(res_train[:, 0], res_train[:, 1])
    print('bacc_train:', bacc_train)
    print('pre_train:', pre_train)
    print('rec_train:', rec_train)
    print('f1_train:', f1_train)
    print('mcc_train:', mcc_train)
    print('sp_train:', sp_train)
    print('q__train:', q__train)
    print('acc_train:', acc_train)

    '''    
    res_dev  = dev_res.T
    cn_matrix=confusion_matrix(res_dev[:,0], res_dev[:,1])
    cn_matrix

    tn2 = cn_matrix[0,0]
    tp2 = cn_matrix[1,1]
    fn2 = cn_matrix[1,0]
    fp2 = cn_matrix[0,1]


    bacc_dev = ((tp2/(tp2+fn2))+(tn2/(tn2+fp2)))/2#balance accurance
    pre_dev= tp2/(tp2+fp2)#precision/q+
    rec_dev = tp2/(tp2+fn2)#recall/se
    sp_dev=tn2/(tn2+fp2)
    q__dev=tn2/(tn2+fn2)
    f1_dev = 2*pre_dev*rec_dev/(pre_dev+rec_dev)#f1score
    mcc_dev = ((tp2*tn2) - (fp2*fn2))/math.sqrt((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2))#Matthews correlation coefficient
    acc_dev=(tp2+tn2)/(tp2+fp2+fn2+tn2)#accurancy
    fpr_dev, tpr_dev, thresholds_dev =roc_curve(res_dev[:,0],res_dev[:,1])
    print('bacc_dev:',bacc_dev)
    print('pre_dev:',pre_dev)
    print('rec_dev:',rec_dev)
    print('f1_dev:',f1_dev)
    print('mcc_dev:',mcc_dev)
    print('sp_dev:',sp_dev)
    print('q__dev:',q__dev)
    print('acc_dev:',acc_dev)

    '''

    cnf_matrix = confusion_matrix(res_test[:, 0], res_test[:, 1])
    cnf_matrix

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
    fpr, tpr, thresholds = roc_curve(res_test[:, 0], res_test[:, 1])
    print('bacc:', bacc)
    print('pre:', pre)
    print('rec:', rec)
    print('f1:', f1)
    print('mcc:', mcc)
    print('sp:', sp)
    print('q_:', q_)
    print('acc:', acc)
    print('auc:', prediction_test)

