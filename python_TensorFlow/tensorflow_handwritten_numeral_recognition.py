#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project   : AI_Expositor_BAT
# File      : tensorflow_handwritten_numeral_recognition.py
# Author    : GXL
# Date      : 2019/7/26

# 手写体数字识别
# 使用softmax回归解决这个问题
import tensorflow as tf
import numpy as np


# 步骤：
# 1、为输入X和y定义placeholder，定义权重W和b
# 2、定义模型结构
# 3、定义损失函数
# 4、定义优化算法
# 5、训练模型
# 6、评估模型

# step 1，None会根据值自动计算出结果
from tensorflow.python.keras.api.keras.datasets import mnist

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))

# step 2，matmul表示矩阵相乘
y = tf.nn.softmax(tf.matmul(x, W) + b)

# step 3，损失函数使用交叉熵 H(p, q),p为期望值，q为输出值。reduce_mean表示所有样本求和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# step 4，使用0.5的算子最小化损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# step 5
sess = tf.InteractiveSession()
# 获取随机变量的初始化函数
tf.global_variables_initializer.run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

# step 6
cross_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.lables}))