# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:04:01 2021

@author: BM109X32G-10GPU-02
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:46:29 2020

@author: de''
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:40:57 2020

@author: de''
"""

from sklearn.datasets import make_blobs
import json
import numpy as np
import math
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def split_smiles(smiles, kekuleSmiles=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    except:
        pass
    splitted_smiles = []
    for j, k in enumerate(smiles):
        if len(smiles) == 1:
            return [smiles]
        if j == 0:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            else:
                splitted_smiles.append(k)
        elif j != 0 and j < len(smiles) - 1:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            elif k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)

        elif j == len(smiles) - 1:
            if k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
    return splitted_smiles

def get_maxlen(all_smiles, kekuleSmiles=True):
    maxlen = 0
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        maxlen = max(maxlen, len(spt))
    return maxlen
def get_dict(all_smiles, save_path, kekuleSmiles=True):
    words = [' ']
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        for w in spt:
            if w in words:
                continue
            else:
                words.append(w)
    with open(save_path, 'w') as js:
        json.dump(words, js)
    return words

def one_hot_coding(smi, words, kekuleSmiles=True, max_len=1000):
    coord_j = []
    coord_k = []
    spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
    if spt is None:
        return None
    for j,w in enumerate(spt):
        if j >= max_len:
            break
        try:
            k = words.index(w)
        except:
            continue
        coord_j.append(j)
        coord_k.append(k)
    data = np.repeat(1, len(coord_j))
    output = sparse.csr_matrix((data, (coord_j, coord_k)), shape=(max_len, len(words)))
    return output

if __name__ == "__main__":
    data_train= pd.read_csv('E:/code/drug/drugnn/data_train.csv')
    data_test=pd.read_csv('E:/code/drug/drugnn/worddrug.csv')
    inchis = list(data_train['SMILES'])
    rts = list(data_train['type'])
    
    smiles, targets = [], []
    for i, inc in enumerate(tqdm(inchis)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)
            targets.append(rts[i])
            
    words = get_dict(smiles, save_path='E:\code\FingerID Reference\drug-likeness/dict.json')
    
    features = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=600)
        if xi is not None:
            features.append(xi.todense())
    features = np.asarray(features)
    targets = np.asarray(targets)
    X_train=features
    Y_train=targets
      

   # physical_devices = tf.config.experimental.list_physical_devices('CPU') 
   # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  #  tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    
  
    inchis = list(data_test['SMILES'])
    rts = list(data_test['type'])
    
    smiles, targets = [], []
    for i, inc in enumerate(tqdm(inchis)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)
            targets.append(rts[i])
            
   # words = get_dict(smiles, save_path='D:/工作文件/work.Data/CNN/dict.json')
    
    features = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=600)
        if xi is not None:
            features.append(xi.todense())
    features = np.asarray(features)
    targets = np.asarray(targets)
    X_test=features
    Y_test=targets
   
  #  kernel = 1.0 * RBF(0.8)
    #model = RandomForestClassifier(n_estimators=10,max_features='auto', max_depth=None,min_samples_split=2, bootstrap=True)
    model = GPC(  random_state=111)
   
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=0.05, verbose=0, mode='min')
    #mcp_save = ModelCheckpoint('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/single_model.h5', save_best_only=True, monitor='accuracy', mode='auto')
  #  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    from tensorflow.keras import backend as K
    X_train = K.cast_to_floatx(X_train).reshape((np.size(X_train,0),np.size(X_train,1)*np.size(X_train,2)))

    Y_train = K.cast_to_floatx(Y_train)
    
#    X_train,Y_train = make_blobs(n_samples=300, n_features=n_features, centers=6)
    model.fit(X_train, Y_train)


 #   model = load_model('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/single_model.h5')
    Y_predict = model.predict(K.cast_to_floatx(X_test).reshape((np.size(X_test,0),np.size(X_test,1)*np.size(X_test,2))))
     #Y_predict = model.predict(X_test)#训练数据
    x = list(Y_test)
    y = list(Y_predict)
    from pandas.core.frame import DataFrame   
    x=DataFrame(x)
    y=DataFrame(y)
  #  X= pd.concat([x,y], axis=1)
    #X.to_csv('C:/Users/sunjinyu/Desktop/FingerID Reference/drug-likeness/CNN/molecularGNN_smiles-master/0825/single-CNN-seed444.csv')
    Y_predict = [1 if i >0.4 else 0 for i in Y_predict]

    cnf_matrix=confusion_matrix(Y_test, Y_predict)
    cnf_matrix
    
    tn = cnf_matrix[0,0]
    tp = cnf_matrix[1,1]
    fn = cnf_matrix[1,0]
    fp = cnf_matrix[0,1]
    
    bacc = ((tp/(tp+fn))+(tn/(tn+fp)))/2#balance accurance
    pre = tp/(tp+fp)#precision/q+
    rec = tp/(tp+fn)#recall/se
    sp=tn/(tn+fp)
    q_=tn/(tn+fn)
    f1 = 2*pre*rec/(pre+rec)#f1score
    mcc = ((tp*tn) - (fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))#Matthews correlation coefficient
    acc=(tp+tn)/(tp+fp+fn+tn)#accurancy
    fpr, tpr, thresholds =roc_curve(Y_test, Y_predict)
    AUC = auc(fpr, tpr)
    print('bacc:',bacc)
    print('pre:',pre)
    print('rec:',rec)
    print('f1:',f1)
    print('mcc:',mcc)
    print('sp:',sp)
    print('q_:',q_)
    print('acc:',acc)
    print('auc:',AUC)
