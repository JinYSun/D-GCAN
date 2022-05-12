# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:19:10 2022

@author: BM109X32G-10GPU-02
"""
import pandas as pd
import D_GCAN
#test = D_GCAN.train('../dataset/data_test.txt')
with open ('../dataset/zinc15.txt') as f:
    file = f.read()
for i in range (1000):
    f =list(file)[1000*i]
predict = D_GCAN.predict('../dataset/zinc15.txt',property=False)
 