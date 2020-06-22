---
layout: post
title:  "[G] TensorFlow"
date:   2020-06-06 13:20:00 +0800
categories: TensorFlow
tags: tensorflow [G]
---

## slim
> slim 是TensorFlow的简洁版。

### 示例
```ruby
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 原生卷积层定义
inputs = ...
with tf.variable_scope('conv1') as scope:
    weights = tf.get_variabel(scope.name + 'w_1',
                              [3, 3, 3, 16],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    biases = tf.get_variabel(scope.name + 'b_1',
                             [16],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

# slim 版本
inputs = ...
net = slim.conv2d(inputs, 16, [3,3], scope='conv1')
# inputs 就是网络输入，16是输出神经元的个数，[3,3]是该层卷积核的大小
```

