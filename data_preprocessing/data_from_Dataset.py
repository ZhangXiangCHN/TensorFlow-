# 输入数据处理框架   数据集方法实现
import tensorflow as tf

train_files = tf.train.match_filenames_once('./train_file-*')
test_files = tf.train.match_filenames_once('./test_file-*')

def parser(record):
    features = tf.parse_single_example(
                record,
                features={
                    'image':tf.FixedLenFeature([], tf.string),
                    'label':tf.FixedLenFeature([], tf.int64),
                    'height':tf.FixedLenFeature([], tf.int64),
                    'width':tf.FixLenFeature([], tf.int64),
                    'channels':tf.FixedLenFeature([], tf.int64),
                    })

    decode_image = tf.decode_raw(features['iamge'], tf.uint8)
    decode_image.set_shape([features['height'], features['width'], features['channels']])

    label = features['label']
    return decode_image, label

image_size = 299
batch_size = 64
shuffle_buffer = 10000

# 构建 dataset
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

dataset = dataset.map(
                lambda image, label : (preprocess_for_train(image,
                                    image_size, image_size, None), label))

dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

# tf.train.match_filenames_once() 的作用机制和tfplaceholder()相似，所以使用make_initializable_iterator()迭代器
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = dataset.get_next()

learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# 处理测试数据集
test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parser).map(
                                lambda image, label: (tf.image.resize_images(image,
                                size=[image_size, image_size]), label))

test_dataset = test_dataset.batch(batch_size=batch_size)

# 定义测试集迭代器
test_iterator = test_dataset.make_initializable_iterator()
test_image_batch, test_label_batch = test_dataset.get_next()

test_logit = inference(test_image_batch)
predictions = tf.argmax(test_logit, axis=-1, output_type=tf.ingt32)

# 会话 Session
with tf.Session() as sess:
    init = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run(init)

    # 初始化数据迭代器
    sess.run(iterator.initializer)

    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break

    sess.run(test_iterator.initializer)
    test_results = []
    test_labels = []
    while True:
        try:
            pred, label = sess.run([predictions, test_label_batch])
            test_result.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

    correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
    accuracy = sum(correct)/len(correct)

    print('Test accuracy is:', accuracy)
