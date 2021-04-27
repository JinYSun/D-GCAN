# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 02:33:28 2021

@author: wei
"""

import os
import sys
sys.path.insert(0, os.getcwd()) # add current working directory to pythonpath
import json
from scipy import sparse
import keras.backend as K
import numpy as np
import pandas as pd
import preprocess as pp
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
#from keras.backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Use only the 1st GPU

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#原版 sess = tf.Session(config=config)
sess =tf.compat.v1.Session(config=config)  
from keras import callbacks
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D, Conv1D, MaxPooling1D, GRU, Bidirectional
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix

from tqdm import tqdm
from rdkit import Chem

import warnings
import argparse
import gc


def generate_tokens(smiles, len_percentile=100):
    """
    Generate character tokens from smiles
    :param smiles: Pandas series, containing smiles
    :param len_percentile: percentile of smiles length to set as max length
    :return tokens
    :return num_words
    :return max_phrase_len
    """ 
    
    # Get max length of smiles
    smiles_len = smiles.apply(lambda p: len(p))
    max_phrase_len = int(np.percentile(smiles_len, len_percentile))
    print('True max length is ' + str(np.max(smiles_len)) + ', ' + str(max_phrase_len) + ' is set the length cutoff.')
        
    # Get unique words
    unique_words = np.unique(np.concatenate(smiles.apply(lambda p: np.array(list(p))).values, axis=0))
    num_words = len(unique_words)
    print('Vocab size is ' + str(num_words))
    
    tokenizer = Tokenizer(
        num_words = num_words,
        filters = '$',
        char_level = True,
        oov_token = '_'
    )
    
    tokenizer.fit_on_texts(smiles)
    sequences = tokenizer.texts_to_sequences(smiles)
    tokens = pad_sequences(sequences, maxlen = max_phrase_len, padding='post', truncating='post')
    
    return tokens, num_words, max_phrase_len
    
    
def create_model(model_type, num_words, input_length, output_dim=1, dropout_rate=0.0):
    """Build different sequence model
    :param model_type: str, can be 'cnn', 'lstm'
    :param num_words: int
    :param input_length: int
    :param output_dim: int
    :return model: Keras model
    """ 

def create_model(model_type, num_words, input_length, output_dim=1, dropout_rate=0.0):
    """Build different sequence model
    :param model_type: str, can be 'cnn', 'lstm'
    :param num_words: int
    :param input_length: int
    :param output_dim: int
    :return model: Keras model
    """ 
    
    model = Sequential()
    if model_type == 'lstm': # LSTM - LSTM
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    
    elif model_type == 'cnn': # 1D CNN
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Conv1D(192, 10, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(192, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    else:
        raise ValueError(model_type + ' is not supported.')
 
    model.summary()    
    return model

def read_data(data_path, col_smiles='smiles', col_target='type'):
    """Split original data into train data and test data.
    :param data_path: str, path to the a CSV data file
    :param col_smiles: str, name of smiles column
    :param col_target: str, name of target column
    :param test_ratio: float, proportion of the original data for testset, must be from 0 to 1
    :param seed: int, randomization seed for reproducibility
    :return (X, y)
    """
    
    # read data
    df = pd.read_csv(data_path, sep=',')
    df_no_na = df[[col_smiles, col_target]].dropna()

    X = df_no_na[col_smiles]
    y = df_no_na[col_target].values
    
    return X, y

def get_prediction_score(y_label, y_predict):
    """Evaluate predictions using different evaluation metrics.
    :param y_label: list, contains true label
    :param y_predict: list, contains predicted label
    :return scores: dict, evaluation metrics on the prediction
    """
    scores = {}
    scores['accuracy'] = accuracy_score(y_label, y_predict)
    scores['f1-score'] = f1_score(y_label, y_predict, labels=None, average='macro', sample_weight=None)
    scores['Cohen kappa'] = cohen_kappa_score(y_label, y_predict)
    scores['Confusion Matrix'] = confusion_matrix(y_label, y_predict)
    
    return scores

def build_sequence_model(trainset, testset, model_type, num_words, input_length, output_dim=1, dropout_rate=0.0,
                     batch_size=32, nb_epochs=100, lr=0.001,
                     save_path=None):
    """Train and evaluate CNN model
    :param trainset: (X_train, Y_train)
    :param testset: (X_test, Y_test)
    :param model_type: str, can be 'cnn', 'lstm'
    :param num_words: int
    :param input_length: int
    :param output_dim: int
    :param batch_size: int, batch size for model training
    :param nb_epochs: int, number of training epoches
    :param lr: float, learning rate
    :param save_path: path to save model
    :return model: fitted Keras model
    :return scores: dict, scores on test set for the fitted Keras model
    """


    
    # Create model
    model = create_model(model_type = model_type, num_words = num_words, input_length = input_length, output_dim = output_dim, dropout_rate = dropout_rate)
    
    # Callback list
    #callback_list = []
    # monitor val_loss and terminate training if no improvement
    #early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, \
                #patience=20, verbose=2, mode='auto', restore_best_weights=True)
    #callback_list.append(early_stop)
    
    #if save_path is not None:
        # save best model based on val_acc during training
        #checkpoint = callbacks.ModelCheckpoint(os.path.join(save_path, model_type + '.h5'), monitor='val_acc', \
                    #verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        #callback_list.append(checkpoint)
        
    # Get train and test set
    (X_train, Y_train) = trainset
    (X_test, Y_test) = testset
    
    # Compute class weights
    weight_list = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    weight_dict = {}
    for i in range(len(np.unique(Y_train))):
        weight_dict[np.unique(Y_train)[i]] = weight_list[i]
    
    # Train only classification head
    optimizer = Adam(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epochs, \
                        class_weight=weight_dict, verbose=2)


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

def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
   # np.random.seed(111)  # fix the seed for shuffle.
    #np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]
def edit_dataset(drug,non_drug,task):
    np.random.seed(111)  # fix the seed for shuffle.
    
    if task =='balance':
        #np.random.shuffle(non_drug)
        non_drug=non_drug[0:len(drug)]
       
    else:
        np.random.shuffle(non_drug)
    np.random.shuffle(drug)
    dataset_train_drug, dataset_test_drug = split_dataset(drug, 0.9)
   # dataset_train_drug,dataset_dev_drug =  split_dataset(dataset_train_drug, 0.9)
    dataset_train_no, dataset_test_no = split_dataset(non_drug, 0.9)
   # dataset_train_no,dataset_dev_no =  split_dataset(dataset_train_no, 0.9)
    dataset_train =  dataset_train_drug+dataset_train_no
    dataset_test= dataset_test_drug+dataset_test_no
  #  dataset_dev = dataset_dev_drug+dataset_dev_no
    return dataset_train, dataset_test  

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
#把python对象转换成json对象生成一个fp的文件流，和文件相关。

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

if __name__ == '__main__':
    
    path='E:\code\drug\drugnn/'
    dataname=''
    
    dataset_drug = pp.create_dataset('drug.txt',path,dataname)
    dataset_nondrug = pp.create_dataset('non-drug.txt',path,dataname)
    
    dataset_train, dataset_test = edit_dataset(dataset_drug, dataset_nondrug,'balance')    
   
    data_path = os.path.join('E:\code\FingerID Reference\drug-likeness', 'Anti-Cardiovascular Disease.csv')
    model_list = ['cnn', 'lstm']
    batch_size = 16
    nb_epochs = 100
    lr = 0.001
    save_path = 'E:\code\drug\drugnn/'
    
     # parse parameters
    parser = argparse.ArgumentParser(description='Build CNN models')
    parser.add_argument('--data_path', help='A path to csv data file')
    parser.add_argument('--batch_size', type=int, help='Batch size for model training')
    parser.add_argument('--nb_epochs', type=int, help='Number of training epoches')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--save_path', help='A path to save fitted models')
    
    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path
    #if args.edit_dataset:
        #edit_dataset = args.edit_dataset
    if args.batch_size:
        batch_size = args.batch_size
    if args.nb_epochs:
        nb_epochs = args.nb_epochs
    if args.lr:
        lr = args.lr
    if args.save_path:
        save_path = args.save_path
      
    # Make save_path
    if save_path is not None:
        os.makedirs(os.path.join(save_path, 'sequence_models'), exist_ok=True)
        
    # Read data
    #smiles, y = read_data(data_path, col_smiles='smiles', col_target='HIV_active')
    smiles, y = read_data(edit_dataset, col_smiles='smiles', col_target='type')
    tokens, num_words, max_phrase_len = generate_tokens(smiles, len_percentile=100)
    
    # Get train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(tokens, y, test_size = 0.2, shuffle=True, stratify=y,
                                                      random_state = 0)

                                                        # Build en evaluate graph models
    model_scores = []
    for model_type in model_list:
        model, scores = build_sequence_model((X_train, Y_train), (X_test, Y_test), model_type, num_words, 
                                             output_dim=1, dropout_rate=0.0,
                                             batch_size=batch_size, nb_epochs=nb_epochs, lr=lr,
                             save_path=os.path.join(save_path, 'sequence_models', model_type + '.h5'))
        model_scores.append(scores)
            
        # force release memory
        K.clear_session()
        del model
        gc.collect()
        
    # Summarize model performance
    model_df = pd.DataFrame({'model': model_list,
                             'accuracy': [score['accuracy'] for score in model_scores],
                            'f1-score': [score['f1-score'] for score in model_scores],
                            'Cohen kappa': [score['Cohen kappa'] for score in\
                                                        model_scores],
                            'Confusion Matrix': [score['Confusion Matrix'] for score in\
                                                             model_scores]                            
                             })
    model_df = model_df[['model', 'accuracy', 'f1-score', 'Cohen kappa',
                         'Confusion Matrix']]
    model_df.to_csv(os.path.join('C:/Users/wei/Desktop/', 'summary_sequence_model.csv'), index=False)
    model_df.sort_values(by=['accuracy', 'f1-score', 'Cohen kappa'],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))
