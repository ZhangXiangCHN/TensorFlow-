import tensorflow as tf
import glob
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = 'flower_processed_data.npy'
TRAIN_FILE = './log/model.ckpt'   # 
# 谷歌训练好的模型文件
CKPT_FILE = './inception_v3.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5
IMG_HEIGHT = 299
IMG_WIDTH = 299
CHANNELS = 3

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'

if not os.path.exists(TRAIN_FILE):
    os.makedirs(TRAIN_FILE)

def get_tuned_variables():
    # 保留不要更新的weights等
    exceptions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    for var in slim.get_model_variables():
        excluded = False
        for exception in exceptions:
            if var.op.name.startswith(exception):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return variables_to_restore

def get_trainable_variables():
    # 获取需要训练的weights等
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []

    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    return variables_to_train

def main():
    if not os.path.exists(TRAIN_FILE):
        os.makedirs(TRAIN_FILE)

    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    training_labels = processed_data[1]
    n_training_examples = len(training_labels)
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print('%d training numbers, %d validation numbers, %d testing numbers'
                    %(n_training_examples, len(validation_labels), len(testing_labels)))

    # 定义V3的输入
    images = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, CHANNELS], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    #定义V3模型
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES, is_training=True)

    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()   # minimize 中用到 trainable_variables
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss(), var_list=trainable_variables)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # code change here
    # exclude = CHECKPOINT_EXCLUDE_SCOPES.split(',')
    # tuned_var = slim.get_variables_to_restore(exclude=exclude)
    tuned_var = get_tuned_variables()
    load_fn = slim.assign_from_checkpoint_fn(
                        CKPT_FILE, tuned_var, ignore_missing_vars = True)  # ERROR: tensor shape not match

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)   # 需要先初始化

        print('Loading tuned_variables from %s' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            print('Steps:', i)   # code test
            sess.run(train_step, feed_dict = {
                                        images:training_images[start:end],
                                        labels:training_labels[start:end]})

            # 输出日志
            if i % 30 == 0 or i+1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                                         images:validation_images, labels:validation_labels})
                print('Step:%d, validation accuracy = %.lf%%'%(i, validation_accuracy*100))

            strat = end
            if start == n_training_examples:
                start = 0
            end = start + BATCH
            if end > n_training_examples:
                end = n_training_examples

        test_accuracy = sess.run(evaluation_step, feed_dict={
                                    images:testing_images, labels:testing_labels})
        print('the testing accuracy:%.lf%%' % (test_accuracy*100))

if __name__ == "__main__":
    main()
