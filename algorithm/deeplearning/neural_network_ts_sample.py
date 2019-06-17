#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/06/12

训练一个神经网络解决二分类问题。使用Tensorflow实现

基于：《TensorFlow 实战Google深度学习框架》

@author: liuqiyun
"""

import tensorflow as tf

from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
# 会产生 2*3 的矩阵。矩阵中元素的初始值为均值为0，标准差为1、且满足正态分布的随机数
# 这个矩阵是tf自定义的矩阵
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
b1 = tf.Variable(tf.random_normal([1, 3], stddev=1, seed=1))
# 会产生 3*1 的矩阵。矩阵中元素的初始值为均值为0，标准差为1、且满足正态分布的随机数
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
b2 = tf.Variable(tf.random_normal([1],stddev=1,seed=1))


# Placeholder 是方便指定输入值的机制。在执行的时候，可以才指定输入值的集合
# 这样，程序会对输入值集合里的每个输入值进行单独计算，并得出单独的结果，最终将所有结果合成一个集合输出
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络向前传播的过程
# matmul 是矩阵乘法
#a = tf.matmul(x, w1)
# 对第二层网络也加上激活函数
a = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
# y = tf.matmul(a, w2)
#y = tf.nn.sigmoid(tf.matmul(a, w2))
y = tf.nn.sigmoid(tf.matmul(a, w2) + b2)

# 定义损失函数和反向传播的算法
# =============================================================================
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# =============================================================================
#基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
#使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)



# 通过随机数生成一个模拟数据机
rdm = RandomState(1)
dataset_size = 12800
# X 为一个 dataset_size*2 的二维数组
X = rdm.rand(dataset_size, 2)
# 定义规则来给出样本的标签。在这里所有 x1+x2<1 的样例都被认为是正样本（比如零件合格）
# 对于 int(x1+x2 < 1)， 如果x1+x2<1，那么得到的整型值为1（正样本），否则为0（负样本）
# Y 会是一个数组，数组的元素数量等于 X的dataset_size
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行 Tensorflow 程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    # 初始化变量
    sess.run(init_op)
    # 打印出在训练之前神经网络参数的值
    print(sess.run(w1)) # 获取 w1
    print(sess.run(w2)) # 获取 w2
    print(sess.run(b1)) # 获取 b1
    print(sess.run(b2)) # 获取 b2
    
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取 batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        
        # 通过选取的样本训练神经网络并更新参数(x为输入参数，y_为输出参数)
        # 更新参数部分使用了placeholder机制
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间（此处为每1000次训练后），计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" 
                  %(i, total_cross_entropy))
    
    print(sess.run(w1)) # 获取 w1
    print(sess.run(w2)) # 获取 w2
    print(sess.run(b1)) # 获取 b1
    print(sess.run(b2)) # 获取 b2
    
    # 利用训练好的模型，来进行预测
    # 预测输入X的类别
    X1 = [[0.7, 0.2], [1.1, 2.0], [0.1, 0.01]]
    Y1 = [1, 0, 1]
    pred_Y1 = sess.run(y,feed_dict={x:X1})
    index = 1
    for pred,real in zip(pred_Y1,Y1):
        print(pred,real)
    
# =============================================================================‘
# 以下想通过直接调用训练好参数来进行预测的方式并不能得到预想的结果，因为w1等变量属于tf的变量，
# 需要通过tf的接口使用
#     # 测试训练结果
#     x1 = [[0.7, 0.2]]
#     a1 = tf.matmul(x1, w1)
#     #y1 = tf.matmul(a1, w2)  
#     y1 = tf.nn.sigmoid(tf.matmul(a1, w2) + b) 
#    # sess.run(y1)
#     print("y1 = %d, and the correct result should y1 = 1" %(sess.run(y1)))
#     x2 = [[1.1, 2.0]]
#     a2 = tf.matmul(x2, w1)
#     #y2 = tf.matmul(a2, w2)  
#     y2 = tf.nn.sigmoid(tf.matmul(a2, w2) + b)  
# #    sess.run(y2)
#     print("y2 = %d, and the correct result should y2 = 0" %(sess.run(y2)))
# =============================================================================
    




