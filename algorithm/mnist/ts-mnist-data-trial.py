#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:13:30 2019

@author: liuqiyun
"""

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# MNIST_data 代表当前程序文件所在的目录中，用于存放MNIST数据的文件夹，如果没有则新建，然后下载．
mnist = input_data.read_data_sets("/Users/liuqiyun/Documents/tech/ML/data/mnist",one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

#获取第二张图片
image = mnist.train.images[1,:]
#将图像数据还原成28*28的分辨率
image = image.reshape(28,28)
#打印对应的标签
print(mnist.train.labels[1])

plt.figure()
plt.imshow(image)
plt.show()
