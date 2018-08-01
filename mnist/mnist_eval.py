# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:30:42 2018

@author: Zhang Xiang
"""
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVER_INTERVAL_SECS = 10

def evaluate(mnist):
    # 评估函数
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUT_NODE], name='y-input')
        
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        
        y = mnist_inference.inference(x, None)  # 验证的时候，正则化不需要
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        
        while True:    #每隔10秒，运行一次，获取最新的model来评估
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                        mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
                    print('After %s training steps, validation accuracy = %g'%(global_step, accuracy_score))
                else:
                    print('No check file found')
                    return
            time.sleep(EVER_INTERVAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets('./mnist.data', one_hot=True)
    evaluate(mnist)
    
if __name__ == "__main__":
    tf.app.run()