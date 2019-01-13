import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(image):
    # 对图像进行预处理，随机调整亮度，对比度等
    image = tf.image.random_brightness(image, max_delta=32./255)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return tf.clip_by_value(image, 0.0, 1.0)  # 限制大小

def preprocess_for_train(image, height, width, bbox):
    # 对原始数据进行处理， 使输出结果可以作为网络的输入层
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin , bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),
                                                bounding_boxes=bbox, min_object_covered=0.5)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 以上对于数据框进行处理，裁剪等操作

    # 调整数据为网络所需的大小
    distorted_image = tf.image.resize_images(
                    distorted_image, [height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image)

    return distorted_image

image_raw_data = tf.gfile.FastGFile('./picture/cat.jpg', 'rb').read()   # 直接读取图片
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    #input boxes must be 3-dimensional [batch, num_boxes, coords]
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.9]]])

    for i in range(6):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()
