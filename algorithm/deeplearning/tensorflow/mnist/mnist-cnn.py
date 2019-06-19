#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:11:44 2019

@author: liuqiyun
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Avoid error on mac:  Closed OMP: Error #15: Initializing 
# libiomp5.dylib, but found libiomp5.dylib already initialized. 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

mnist = input_data.read_data_sets("/Users/liuqiyun/Documents/tech/ML/data/mnist",one_hot=True)
#mnist = input_data.read_data_sets("/root/py/MNIST_data",one_hot=True)


input = tf.placeholder(tf.float32,[None,784])
input_image = tf.reshape(input,[-1,28,28,1])

y = tf.placeholder(tf.float32,[None,10])

# input 代表输入，filter 代表卷积核
def conv2d(input,filter):
    return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')
# 池化层
def max_pool(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 初始化卷积核或者是权重数组的值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 初始化bias的值
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

#[filter_height, filter_width, in_channels, out_channels]
#定义了卷积核
filter = [3,3,1,32]

filter_conv1 = weight_variable(filter)
b_conv1 = bias_variable([32])
# 创建卷积层，进行卷积操作，并通过Relu激活，然后池化
h_conv1 = tf.nn.relu(conv2d(input_image,filter_conv1)+b_conv1)
h_pool1 = max_pool(h_conv1)

h_flat = tf.reshape(h_pool1,[-1,14*14*32])

W_fc1 = weight_variable([14*14*32,768])
b_fc1 = bias_variable([768])
h_fc1 = tf.matmul(h_flat,W_fc1) + b_fc1

W_fc2 = weight_variable([768,10])
b_fc2 = bias_variable([10])

y_hat = tf.matmul(h_fc1,W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat ))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

#    for i in range(10000):
    for i in range(1):

        batch_x,batch_y = mnist.train.next_batch(50)

        if i % 100 == 0:
            # 执行到这一句的时候，python kernel died
            train_accuracy = accuracy.eval(feed_dict={input:batch_x,y:batch_y})
            print("step %d,train accuracy %g " %(i,train_accuracy))

        train_step.run(feed_dict={input:batch_x,y:batch_y})

        # sess.run(train_step,feed_dict={x:batch_x,y:batch_y})

    print("test accuracy %g " % accuracy.eval(feed_dict={input:mnist.test.images,y:mnist.test.labels}))
