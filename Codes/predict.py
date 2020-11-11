# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:59:34 2020

@author: DIY
"""

import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from EBGW import read_file
from EBGW import encode
from data_preparation import encode_data_preparation
from data_read import data
import csv


#main function
def main():
    parser = argparse.ArgumentParser(description = 'DeepCLA: A hybrid deep learning approach for the identification of clathrin')
    parser.add_argument('-input', dest = 'inputfile', type = str, help = 'Protein sequences to be predicted in txt format.', required = True)
    parser.add_argument('-threshold', dest = 'threshold_value', type = float, help = 'Please input a value between 0 and 1', required = True)
    parser.add_argument('-output', dest = 'outputfile', type = str, help = 'Saving the prediction result in csv format.', required = False)
    args = parser.parse_args()
    inputfile = args.inputfile;
    threshold = args.threshold_value;
    outputfile = args.outputfile;
    
    print("Protein sequences are Encoding...")
    
    Original = read_file(inputfile)
    Protein = encode(Original[1])
    x_test = encode_data_preparation(Protein)
    y_test = []
    for i in range(int(len(x_test))):
        y_test.append([1,0]) 
    y_test = np.array(y_test)
    print("Loading model...")
       
    ckpt_dir="./best_model"
    saver = tf.train.Saver
    sess = tf.Session() 
  
    saver = tf.train.import_meta_graph(ckpt_dir+"/best_model.ckpt-390.meta")      
    
    gragh = tf.get_default_graph()  # Gets the current graph 
    #tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # Gets the names of all the variables in the current diagram
    keep_prob = gragh.get_tensor_by_name('Placeholder:0')
    xs = gragh.get_tensor_by_name('Placeholder_1:0')
    ys = gragh.get_tensor_by_name('Placeholder_2:0')

    predict = gragh.get_tensor_by_name('fw_side/fw_side/add_1:0')  #Gets the network output value
    saver.restore(sess, ckpt_dir+"/best_model.ckpt-390")
    prediction = sess.run(predict,feed_dict = {xs:x_test,ys:y_test,keep_prob:0.8})
    result_c = None

    id_name = []
    sequence = []
    probability = []
    print("Protein are predicted as follows:")
    
    lines = data(inputfile)
    
    for i in range(int(len(lines)/2)):        
        if prediction[i][1]>threshold:
            id_name.append(lines[2*i])
            sequence.append(lines[2*i+1])
            var = '%.11f' % prediction[i][1]
            probability.append(var)
            print(lines[2*i])
            print(lines[2*i+1])
            print("probability value:"+str(var))
        else:
            print(lines[2*i])
            print(lines[2*i+1])
            print("Nonclathrin")
            
    result_c = np.column_stack((id_name,sequence,probability))
    result = pd.DataFrame(result_c)
    result.to_csv(outputfile,index = False,header = None, sep = '\t',quoting = csv.QUOTE_NONNUMERIC)
    print ("Successfully predicted for clathrin protein !\n")
    
if __name__ == "__main__" :
    main()
    