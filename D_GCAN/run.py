# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:19:10 2022

@author: BM109X32G-10GPU-02
"""
import torch
import pandas as pd
import train
import predict
test = train.train('../dataset/world_wide.txt')
f =pd.read_table('../dataset/zinc15.txt')
#predict = predict.predict('../dataset/world_wide.txt',property=True)
torch.cuda.empty_cache()
for i in range (307):
    file = f.iloc[1000*i:1000*i+1000,0]
    file=pd.DataFrame(file)
    file.to_csv('../dataset/zinc15/'+str(i)+'.txt',index=None)
    
for i in range (307):
    predict = predict.predict('../dataset/zinc15/'+str(i)+'.txt',property=False)
    res=pd.DataFrame(predict)
    res.to_csv('../dataset/zinc15/'+str(i)+'.csv',index=None)
 