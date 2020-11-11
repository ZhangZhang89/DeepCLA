# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:57:06 2020

@author: DIY
"""

import tensorflow as tf
import numpy as np
import pandas as pd


def Model(xs,w1,w2,w3,weights,bias,keep_prob): 
    w1 = tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.000005)(tf.Variable(tf.random_normal([5,5,1,128]),dtype = tf.float32)))
    w2 = tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.000005)(tf.Variable(tf.random_normal([5,5,128,256]),dtype = tf.float32)))
    w3 = tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.000005)(tf.Variable(tf.random_normal([256*9*5,784]),dtype = tf.float32)))

    keep_prob = tf.placeholder(tf.float32)
    lstm_size = 625
    max_time = 28
    num_units = 28
    conv1 = tf.nn.conv2d(xs, w1, strides=[1, 1, 1, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1,axis = -1,momentum = 0.99,training = True)
    conv1_out = tf.nn.relu(conv1_bn)
    pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1_out = tf.nn.dropout(pool1, keep_prob) 
    
    conv2 = tf.nn.conv2d(pool1_out, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2,axis = -1,momentum = 0.99,training = True)
    conv2_out = tf.nn.relu(conv2_bn)
    pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = tf.reshape(pool2, [-1, w3.get_shape().as_list()[0]])
    pool2_out = tf.nn.dropout(pool2,keep_prob)
    
    h_pool = tf.matmul(pool2_out, w3)
    h_pool = tf.nn.relu(h_pool)
    h_pool_out = tf.nn.dropout(h_pool, keep_prob)
    inputs = tf.reshape(h_pool_out,[-1,max_time,num_units])
    lstm_fw1 = tf.nn.rnn_cell.LSTMCell(num_units = lstm_size)
    lstm_fw2 = tf.nn.rnn_cell.LSTMCell(num_units = lstm_size)
    lstm_forward = tf.nn.rnn_cell.MultiRNNCell(cells = [lstm_fw1,lstm_fw2])
    
    lstm_bw1 = tf.nn.rnn_cell.LSTMCell(num_units = lstm_size)
    lstm_bw2 = tf.nn.rnn_cell.LSTMCell(num_units = lstm_size)
    lstm_backward = tf.nn.rnn_cell.MultiRNNCell(cells = [lstm_bw1,lstm_bw2])
    
    outputs,final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_forward,cell_bw = lstm_backward,inputs = inputs,dtype = tf.float32)
    state_forward = final_state[0][-1][-1]
    state_backward = final_state[0][-1][-1]
    output = tf.matmul(state_forward+state_backward,weights)+bias
    return output

lstm_size = 625
n_classes = 2  
weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev = 0.1))
bias = tf.Variable(tf.constant(0.1,shape = [n_classes]))
xs = tf.placeholder(tf.float32,[None,33,20,1])
ys = tf.placeholder(tf.float32,[None,2])
  
with tf.name_scope("fw_side"),tf.variable_scope("fw_side", reuse=tf.AUTO_REUSE):
    prediction = tf.layers.dense(Model,n_classes,name = 'softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = ys))
    tf.add_to_collection('losses',cross_entropy)
    total_loss = tf.add_n(tf.get_collection('losses'))
    train_step_l2_norm = tf.train.AdamOptimizer(1e-3).minimize(total_loss)
    correct_prediction = tf.equal(tf.argmax(ys,1),tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


