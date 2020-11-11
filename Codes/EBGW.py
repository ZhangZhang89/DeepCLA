# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:59:06 2020

@author: DIY
"""

import numpy as np
import math
import pandas as pd

def read_file(filepath):
    try:
        f = open(filepath)
    except IOError:
        print('Failed to open'+ filepath +',please check if file is exist or not!')
        exit()
    else:
        f = open(filepath)
        lines = f.read().splitlines()
        sequence = []
        idlist = []
        for i in range(len(lines)):
            if lines[i][0] == '>':
                idlist.append(lines[i])
                sequence.append(lines[i+1])
        return idlist,sequence

def encode(sequence): 
    try:
        ebgw = []
        for s in range(0,len(sequence)):
            lines = sequence[s]
            for t in lines:
                t !='ACDEFGHIKLMNPQRSTVWY'
    except IOError:
        print('Warning: The sequence contains 20 unusual amino acids!')
        exit()
    else:
        ebgw = []
        for s in range(0,len(sequence)):
            lines = sequence[s]
            l=int(len(lines))
                
            C1='AFGILMPVW'
            C2='CNQSTY'
            C3='HKR'
            C4='DE'
                   
            data = []
            code = []
            data1 = []
            data2 = []
            data3 = []
            
            A = []    
            for j in range(l):
                pos = [ii for ii,v in enumerate(C1) if v == lines[j]]
                pos1 = [ii for ii,v in enumerate(C2) if v == lines[j]]
                pos2 = [ii for ii,v in enumerate(C3) if v == lines[j]]
                pos3 = [ii for ii,v in enumerate(C4) if v == lines[j]]
                if len(pos) == 1 or len(pos1) == 1:
                    A.append(1)
                elif len(pos2) == 1 or len(pos3) == 1 :
                    A.append(0)
                else:
                    A.append(0)
                pos = []
                pos1 = []
                pos2 = []
                pos3 = []    
            data1.append(A)
                                    
            B = []
            for j in range(l):
                pos = [ii for ii,v in enumerate(C1) if v == lines[j]]
                pos1 = [ii for ii,v in enumerate(C3) if v == lines[j]]
                pos2 = [ii for ii,v in enumerate(C2) if v == lines[j]]
                pos3 = [ii for ii,v in enumerate(C4) if v == lines[j]]
                if len(pos) == 1 or len(pos1) == 1:
                    B.append(1)
                elif len(pos2) == 1 or len(pos3) == 1 :
                    B.append(0)
                else:
                    B.append(0)
                pos = []
                pos1 = []
                pos2 = []
                pos3 = []
            data2.append(B)
            
            C = []
            for j in range(l):
                pos = [ii for ii,v in enumerate(C1) if v == lines[j]]
                pos1 = [ii for ii,v in enumerate(C4) if v == lines[j]]
                pos2 = [ii for ii,v in enumerate(C2) if v == lines[j]]
                pos3 = [ii for ii,v in enumerate(C3) if v == lines[j]]
                if len(pos) == 1 or len(pos1) == 1:
                    C.append(1)
                elif len(pos2) == 1 or len(pos3) == 1:
                    C.append(0)
                else:
                    C.append(0) 
                pos = []
                pos1 = []
                pos2 = []
                pos3 = []
            data3.append(C)
            
            data = np.hstack((data1,data2,data3))
            m,n = np.shape(data)
            k1 = 220
            x1 = []
            x2 = []
            x3 = []
            for i in range(m):
                x11 = []
                x22 = []
                x33 = []
                a = 0
                b = 0
                c = 0
                for j in range(int(k1)):
                    a = sum(data1[i][0:int(math.floor(l*(j+1)/k1))])/math.floor(l*(j+1)/k1)
                    b = sum(data2[i][0:int(math.floor(l*(j+1)/k1))])/math.floor(l*(j+1)/k1)
                    c = sum(data3[i][0:int(math.floor(l*(j+1)/k1))])/math.floor(l*(j+1)/k1)
                    x11.append(a)
                    x22.append(b)
                    x33.append(c)
                x1.append(x11)
                x2.append(x22)
                x3.append(x33)
            code = np.hstack((x1,x2,x3))
            code = np.array(code)
            ebgw.extend(code)
        ebgw = np.array(ebgw)
        return ebgw


        
        
