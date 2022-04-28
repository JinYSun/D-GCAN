# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:19:10 2022

@author: BM109X32G-10GPU-02
"""

import D_GCAN
test = D_GCAN.train('../dataset/data_test.txt')
predict = D_GCAN.predict('../dataset/test.txt',property=False)
 