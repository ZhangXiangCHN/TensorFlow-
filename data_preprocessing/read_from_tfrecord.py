# 解析TFRecord 文件
import tensorflow as tf

reader = tf.TFRecordReader()

filename = './tf_records'
filename_queue = tf.train.string_input_producer([filename])

_, serialized_examples = reader.read(filename_queue)

# tf.FixedLenFeature() 方法解析的结果是一个Tensor
features = tf.parse_single_example(
                serialized_examples,
                features={
                    'image_raw':tf.FixedLenFeature([], tf.string),
                    'label':tf.FixedLenFeature([], tf.int64),
                    'pixels':tf.FixedLenFeature([], tf.int64)
                    })
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    sess.run([image, label, pixels])
    print('image:', image, 'label', label, 'pixels', pixels)   # Tensor
