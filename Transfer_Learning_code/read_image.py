import tensorflow as tf
import glob
import os
import numpy as np
from tensorflow.python.platform import gfile

INPUT_DATA = './flower_photos'
OUTPUT_FILE = 'flower_processed_data.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
IMG_HEIGHT = 299
IMG_WIDTH = 299

def creat_image_lists(sess, testing_percentage, validation_perentage):
    # 将数据集划分为训练集、验证集和测试集
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]   # 对路径INPUT_DATA进行遍历，os.walk()返回3元组
    is_root_dirs = True

    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_images = []
    testing_labels = []

    current_label = 0
    all_count = 0
    for sub_dir in sub_dirs:
        if is_root_dirs:
            is_root_dirs = False
            continue

        # 确定图片地址
        extensions = ['jpeg', 'jpg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*."+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        image_count = 0
        # 读取图片
        for file_name in file_list:
            image_count += 1
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
            image_value = sess.run(image)

            # 随机分类
            chance = np.random.randint(100)
            if chance < validation_perentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage+validation_perentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)

        current_label += 1
        all_count += image_count
        print("the total number of this class is:%d"%image_count)

    print("the total number is:%d"%all_count)
    print('train image:%d, validation image:%d, test image:%d'%(len(training_images),
                                        len(validation_images), len(testing_images)))

    # 打乱训练数据
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    # 通过数组的形式保存image数据和label标签
    return np.asarray([training_images, training_labels,
                        validation_images, validation_labels,
                        testing_images, testing_labels])

def main():
    with tf.Session() as sess:
        processed_data = creat_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        np.save(OUTPUT_FILE, processed_data)

if __name__ == "__main__":
    main()
