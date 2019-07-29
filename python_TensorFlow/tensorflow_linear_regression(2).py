#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project   : AI_Expositor_BAT
# File      : tensorflow_linear_regression(2).py
# Author    : GXL
# Date      : 2019/7/26

# 线性回归-使用高层次API
import tensorflow as tf
import numpy as np

# --------------------标准结构，可以直接使用--------------------
# 定义特征列表，此处只有一个数值特征
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
# 通过estimator拿到线性回归模型
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
# ----------------------------------------------------------

x_train = np.array([1, 2, 3, 4])
y_train = np.array([0, -1, -2, -3])

x_eval = np.array([2, 5, 8, 1])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# ----------------------定义输入数据--------------------------
# 输入，shuffle表示是否随机打乱
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

# 训练集
input_fn_train = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

# 评估集合
input_fn_eval = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
# ----------------------------------------------------------

# 与上面对应,训练模型
estimator.train(input_fn=input_fn, steps=1000)

# （评估模型）查看在训练集上面的效果
train_metrics = estimator.evaluate(input_fn=input_fn_train)
# （评估模型）查看测试集上面的效果
eval_metrics = estimator.evaluate(input_fn=input_fn_eval)
print(train_metrics)
print(eval_metrics)
