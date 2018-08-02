# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:12:52 2018

@author: Zhang Xiang
"""
import tensorflow as tf

INPUT_NODE = 784
OUT_NODE = 10
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1

# 第一个卷积层
CONV1_SIZE = 5
CONV1_DEEP = 32

# 第二个卷积层
CONV2_SIZE = 5
CONV2_DEEP = 64

# 全连接层节点个数
FC_SIZE = 512
DROPOUT_RATE = 0.5

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer, train=None):
    # 前向传播过程, 包含一个隐含层
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = get_weight_variable([CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNEL, CONV1_DEEP], regularizer=regularizer)
        conv1_biases = tf.get_variable('bias',  [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = get_weight_variable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], regularizer=regularizer)
        conv2_biases = tf.get_variable('bias',  [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = get_weight_variable([nodes, FC_SIZE], regularizer=regularizer)
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    
    with tf.name_scope('layer6_dropout1'):
        if train:
            fc1 = tf.nn.dropout(fc1, DROPOUT_RATE)
    
    with tf.variable_scope('layer7_fc2'):
        fc2_weights = get_weight_variable([FC_SIZE, OUT_NODE], regularizer=regularizer)
        fc2_biases = tf.get_variable('bias', [OUT_NODE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    
    return logit