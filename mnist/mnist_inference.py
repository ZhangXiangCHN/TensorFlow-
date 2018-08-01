# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:12:52 2018

@author: Zhang Xiang
"""
import tensorflow as tf

INPUT_NODE = 784
OUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    # 前向传播过程, 包含一个隐含层
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer=regularizer)
        biases = tf.get_variable('biases',  [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUT_NODE], regularizer=regularizer)
        biases = tf.get_variable('biases',  [OUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    
    return layer2