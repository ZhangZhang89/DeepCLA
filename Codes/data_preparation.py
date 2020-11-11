# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:58:00 2020

@author: DIY
"""

import numpy as np

def encode_data_preparation(ebgw):
    x = ebgw/255
    inputsize = [33,20]
    x = x.reshape(-1,inputsize[0],inputsize[1],1)
    return x