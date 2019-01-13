# 输入数据处理框架
import tensorflow as tf

# 假设已经处理，并存在tfrecords数据 ./tf_records
files = tf.train.match_filenames_once('./tf_records')   # 正则化匹配数据文件
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 解析stfrecord数据
reader = tf.TFRecordReader()
-, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
            serialized_example,
            features={
                'image':tf.FixedLenFeature([], tf.string),
                'label':tf.FixedLenFeature([], tf.int64),
                'height':tf.FixedLenFeature([], tf.int64),
                'width':tf.FixedLenFeature([], tf.int64),
                'channels':tf.FixedLenFeature([], tf.int64),
                })
image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

decode_image = tf.decode_raw(image, tf.uint8)
decode_image.set_shape([height, width, channels])

image_size = 299
distorted_image = preprocess_for_train(decode_image, image_size, image_size, None)

# get the batch
min_after_dequeue = 200
batch_size = 64
capacity = min_after_dequeue + 3*batch_size    # 可以随便设置
image_batch, label_batch = tf.train.shuffle_batch([distorted_image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue)

# train op
learning_rate = 0.01
logit = inference(image_batch)   # 网络 model inference
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)

    # 停止所有线程
    coord.request_stop()
    coord.join(threads=threads)
