# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:19:10 2022

@author: BM109X32G-10GPU-02
"""
import pandas as pd
import D_GCAN
#test = D_GCAN.train('../dataset/data_test.txt')
f =pd.read_table('../dataset/zinc15.txt')

for i in range (307):
    file = f.iloc[1000*i:1000*i+1000,0]
    file=pd.DataFrame(file)
    file.to_csv('../dataset/zinc15/'+str(i)+'.txt',index=None)
    
for i in range (307):
    predict = D_GCAN.predict('../dataset/zinc15/'+str(i)+'.txt',property=False)
    res=pd.DataFrame(predict)
    res.to_csv('../dataset/zinc15/'+str(i)+'.csv',index=None)
 