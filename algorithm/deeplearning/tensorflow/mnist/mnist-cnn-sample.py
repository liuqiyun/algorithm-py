#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/06/19

训练一个神经网络解决 MNIST 图库的分类问题

基于：《TensorFlow 实战Google深度学习框架》

@author: liuqiyun
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data;

import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Avoid error on mac:  Closed OMP: Error #15: Initializing 
# libiomp5.dylib, but found libiomp5.dylib already initialized. 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# MNIST数据集相关的常数
# 一张图片的像素 = 28 * 28 = 784
INPUT_NODE = 784        # 输入层节点数，总像素数
OUTPUT_NODE = 10        # 输出层节点数，总类别数

# 配置神经网络参数
LAYER1_NODE = 500       # 隐藏层节点数，使用单隐藏层作为样例

BATCH_SIZE = 100        # 一个训练batch中的训练数据个数。数字越小越接近随机梯度下降，
                        # 数字越大越接近梯度下降


mnist = input_data.read_data_sets("/Users/liuqiyun/Documents/tech/ML/data/mnist",one_hot=True)


# 定义神经网络的参数
# =============================================================================
# # 会产生 2*3 的矩阵。矩阵中元素的初始值为均值为0，标准差为1、且满足正态分布的随机数
# # 这个矩阵是tf自定义的矩阵
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# b1 = tf.Variable(tf.random_normal([1, 3], stddev=1, seed=1))
# 
# # 会产生 3*1 的矩阵。矩阵中元素的初始值为均值为0，标准差为1、且满足正态分布的随机数
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# b2 = tf.Variable(tf.random_normal([1],stddev=1,seed=1))
# =============================================================================

# 生成隐藏层参数
w1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )
b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
# 生成输出层参数
w2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1)
    )
b2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))




# Placeholder 是方便指定输入值的机制。在执行的时候，可以才指定输入值的集合
# 这样，程序会对输入值集合里的每个输入值进行单独计算，并得出单独的结果，最终将所有结果合成一个集合输出
x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

# 定义神经网络向前传播的过程
# matmul 是矩阵乘法
# =============================================================================
# a = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
# y = tf.nn.sigmoid(tf.matmul(a, w2) + b2)
# =============================================================================

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.matmul(layer1, w2) + b2

# 定义损失函数和反向传播的算法
# =============================================================================
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# =============================================================================
#基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
#使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)


# 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件
# 和评判训练的效果
validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
# 准备测试数据。在真实应用中，这部分数据在训练时是不可见的，这个数据只是作为
# 模型优劣的最后评价标准
test_feed= {x: mnist.test.images,
                         y_: mnist.test.labels}


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建一个会话来运行 Tensorflow 程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    # 初始化变量
    sess.run(init_op)
# =============================================================================
#     # 打印出在训练之前神经网络参数的值
#     print(sess.run(w1)) # 获取 w1
#     print(sess.run(w2)) # 获取 w2
#     print(sess.run(b1)) # 获取 b1
#     print(sess.run(b2)) # 获取 b2
# =============================================================================
    
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 产生这一轮使用的一个batch的训练数据，并运行训练过程
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        # 通过选取的样本训练神经网络并更新参数(x为输入参数，y_为输出参数)
        # 更新参数部分使用了placeholder机制
        sess.run(train_step, feed_dict={x: xs, y_: ys})
        if i % 1000 == 0:
            # 每隔一段时间（此处为每1000次训练后），计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: xs, y_: ys})
            print("After %d training step(s), cross entropy on all data is %g" 
                  %(i, total_cross_entropy))
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d training step(s), validation accuracy"
                      " is %g" % (i, validate_acc))
    
    # 训练结束之后，在测试数据上检测神经网络模型的最终正确率
    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s), test accuracy "
              " is %g" % (STEPS, test_acc))
    
# =============================================================================
#     print(sess.run(w1)) # 获取 w1
#     print(sess.run(w2)) # 获取 w2
#     print(sess.run(b1)) # 获取 b1
#     print(sess.run(b2)) # 获取 b2
# =============================================================================
    
# =============================================================================
#     # 利用训练好的模型，来进行预测
#     # 预测输入X的类别
#     X1 = [[0.7, 0.2], [1.1, 2.0], [0.1, 0.01]]
#     Y1 = [1, 0, 1]
#     pred_Y1 = sess.run(y,feed_dict={x:X1})
#     index = 1
#     for pred,real in zip(pred_Y1,Y1):
#         print(pred,real)
# =============================================================================



