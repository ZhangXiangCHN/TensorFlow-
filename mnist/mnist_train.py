# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:02:05 2018

@author: Zhang Xiang
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './PATH/'
MODEL_NAME = 'model.pkl'


def train(mnist):
    # 训练过程
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    
    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    # 学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, 
            global_step, 
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY
            )
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_op, variable_average_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x:xs, y_:ys})
            
            if i%1000 == 0:
                print('After %d training steps, the loss on training batch is %g.'%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    
def main(argv = None):
    mnist = input_data.read_data_sets('./mnist.data', one_hot=True)
    train(mnist)
    
if __name__ == '__main__':
    tf.app.run()