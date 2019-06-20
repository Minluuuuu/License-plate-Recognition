import os
import test
import cv2
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from pip._vendor.distlib._backport import shutil

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 32
PROVINCES = ("京", "闽", "粤", "苏", "沪", "浙","川","鄂","赣","甘","贵","桂","黑","冀","津","吉","辽","鲁","蒙","宁","青","琼"
             ,"陕","晋","皖","湘","新","豫","渝","粤","云","藏")
nProvinceIndex = 0
time_begin = time.time()

def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)


def province_test():
    province_graph = tf.Graph()
    with province_graph.as_default():
        with tf.Session(graph=province_graph) as sess_p:

            # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver_p = tf.train.import_meta_graph(
                "./train-saver/province/model.ckpt.meta")
            model_file = tf.train.latest_checkpoint("./train-saver/province")
            saver_p.restore(sess_p, model_file)

            # 第一个卷积层
            W_conv1 = sess_p.graph.get_tensor_by_name("W_conv1:0")
            b_conv1 = sess_p.graph.get_tensor_by_name("b_conv1:0")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 第二个卷积层
            W_conv2 = sess_p.graph.get_tensor_by_name("W_conv2:0")
            b_conv2 = sess_p.graph.get_tensor_by_name("b_conv2:0")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 全连接层
            W_fc1 = sess_p.graph.get_tensor_by_name("W_fc1:0")
            b_fc1 = sess_p.graph.get_tensor_by_name("b_fc1:0")
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)

            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层
            W_fc2 = sess_p.graph.get_tensor_by_name("W_fc2:0")
            b_fc2 = sess_p.graph.get_tensor_by_name("b_fc2:0")

            # 定义优化器和训练op
            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            for n in range(1, 2):
                path = "test_images/%s.bmp" % (n)
                img = Image.open(path)
                width = img.size[0]
                height = img.size[1]
                img_data = [[0] * SIZE for i in range(1)]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 190:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0

                result = sess_p.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})
                max1 = 0
                max2 = 0
                max3 = 0
                max1_index = 0
                max2_index = 0
                max3_index = 0
                for j in range(7):
                    if result[0][j] > max1:
                        max1 = result[0][j]
                        max1_index = j
                        continue
                    if (result[0][j] > max2) and (result[0][j] <= max1):
                        max2 = result[0][j]
                        max2_index = j
                        continue
                    if (result[0][j] > max3) and (result[0][j] <= max2):
                        max3 = result[0][j]
                        max3_index = j
                        continue

                nProvinceIndex = max1_index
        #         print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
        #             PROVINCES[max1_index], max1 * 100, PROVINCES[max2_index], max2 * 100, PROVINCES[max3_index],
        #             max3 * 100))
        # print("省份简称是: %s" % PROVINCES[nProvinceIndex])
        return PROVINCES[nProvinceIndex]

LETTERS_DIGITS_SECOND = (
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
    "Z")


def province_letter_test():
    license_num = ""
    letter_graph = tf.Graph()
    with letter_graph.as_default():
        with tf.Session(graph=letter_graph) as sess:

            # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver = tf.train.import_meta_graph(
                "./train-saver/letters/model.ckpt.meta")
            model_file = tf.train.latest_checkpoint("./train-saver/letters")
            saver.restore(sess, model_file)

            # 第一个卷积层
            W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
            b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 第二个卷积层
            W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
            b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 全连接层
            W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
            b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)

            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层
            W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
            b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

            # 定义优化器和训练op
            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            for n in range(2, 3):
                path = "test_images/%s.bmp" % (n)
                img = Image.open(path)
                width = img.size[0]
                height = img.size[1]

                img_data = [[0] * SIZE for i in range(1)]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 80:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0

                result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})

                max1 = 0
                max2 = 0
                max3 = 0
                max1_index = 0
                max2_index = 0
                max3_index = 0
                for j in range(24):
                    if result[0][j] > max1:
                        max1 = result[0][j]
                        max1_index = j
                        continue
                    if (result[0][j] > max2) and (result[0][j] <= max1):
                        max2 = result[0][j]
                        max2_index = j
                        continue
                    if (result[0][j] > max3) and (result[0][j] <= max2):
                        max3 = result[0][j]
                        max3_index = j
                        continue

                if n == 3:
                    license_num += "-"
                license_num = license_num + LETTERS_DIGITS_SECOND[max1_index]
                # print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
                #     #LETTERS_DIGITS_SECOND[max1_index], max1 * 100, LETTERS_DIGITS_SECOND[max2_index], max2 * 100,
                #     LETTERS_DIGITS_SECOND[max3_index],
                #     max3 * 100))

        #print("城市代号是: 【%s】" % license_num)
        return license_num


'''
车牌号码  后五位识别  
'''

LETTERS_DIGITS = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")


def last_5_num_test():
    license_num = ""
    last_5_num_graph = tf.Graph()
    with last_5_num_graph.as_default():
        with tf.Session(graph=last_5_num_graph) as sess:
            # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
            x = tf.placeholder(tf.float32, shape=[None, SIZE])
            x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
            saver = tf.train.import_meta_graph(
                "./train-saver/digits/model.ckpt.meta")
            model_file = tf.train.latest_checkpoint("./train-saver/digits")
            #print("main3")
            saver.restore(sess, model_file)

            # 第一个卷积层
            W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
            b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 第二个卷积层
            W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
            b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

            # 全连接层
            W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
            b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
            h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

            # dropout
            keep_prob = tf.placeholder(tf.float32)

            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # readout层
            W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
            b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

            # 定义优化器和训练op
            conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            for n in range(3, 8):
                path = "test_images/%s.bmp" % (n)
                img = Image.open(path)
                width = img.size[0]
                height = img.size[1]

                img_data = [[0] * SIZE for i in range(1)]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 190:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0

                result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})

                max1 = 0
                max2 = 0
                max3 = 0
                max1_index = 0
                max2_index = 0
                max3_index = 0
                for j in range(34):
                    if result[0][j] > max1:
                        max1 = result[0][j]
                        max1_index = j
                        continue
                    if (result[0][j] > max2) and (result[0][j] <= max1):
                        max2 = result[0][j]
                        max2_index = j
                        continue
                    if (result[0][j] > max3) and (result[0][j] <= max2):
                        max3 = result[0][j]
                        max3_index = j
                        continue

                license_num = license_num + LETTERS_DIGITS[max1_index]
        #         print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
        #             LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100,
        #             LETTERS_DIGITS[max3_index],
        #             max3 * 100))
        #
        # print("车牌编号是: 【%s】" % license_num)
        return license_num

def pre():
    first = province_test()
    second = province_letter_test()
    last = last_5_num_test()
    test = first + second + last
    return test
    #print(test)
# if __name__ == '__main__':
#     first = province_test()
#     second = province_letter_test()
#     last = last_5_num_test()
#     test = first + second + last
#
#     #print(first, second, last)
#     #data = open("./data.txt",'w')
#     #print(test,file=data)
#     #data.close()
#     print(test)
