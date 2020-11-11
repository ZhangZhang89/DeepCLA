# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:58:42 2020

@author: DIY
"""

def data(filepath):
    f = open(filepath)
    lines = f.read().splitlines()
    return lines