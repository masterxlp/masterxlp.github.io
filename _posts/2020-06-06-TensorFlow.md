---
layout: post
title:  "[G] TensorFlow"
date:   2020-06-06 13:20:00 +0800
categories: Code Tools
tags: TensorFlow G
author: Xlp
---
* content
{:toc}

> 该部分记录了本人对Tensorflow的api以及某些算法的Tensorflow实现




## 算法实现
## lstm
### 原理
LSTM可处理序列式的数据，其每个cell是由三种门限机制组成的，详见图1.

<div align="center"><img src="../../../../image/LSTM结构.png" width="60%" height="60%"></div>
<div align="center">图1. LSTM cell结构图</div>

<div align="center"><img src="../../../../image/LSTM中的符号说明.png" width="60%" height="60%"></div>
<div align="center">图2. LSTM cell结构图中的符号说明</div>

可以看出，遗忘门实际上做的是一个sigmoid的计算，其输入为上一个cell的隐藏状态 $h_{t-1}$ 以及当前时刻的输入 $X_t$；
记忆门做了两个事，一个是sigmoid计算，一个tanh计算，其中sigmoid的计算的输入与tanh的计算的输入一样，为 上一个cell的隐藏状态 $h_{t-1}$ 以及当前时刻的输入 $X_t$；
输出门也是做了一个sigmoid的计算，其输入也为上一个cell的隐藏状态 $h_{t-1}$ 以及当前时刻的输入 $X_t$；
然后，新状态的值为遗忘门的输出与上一个状态的值的点积加上记忆门的两个输出结果的点积；
最后，隐藏状态的值为输出门的输出与新状态的tanh计算后的输出的点积。

### Tensorflow实现

```ruby
# 生成各个门的参数
def _generate_params_for_lstm_cell(x_size, h_size, bias_size):
    x_w = tf.get_variable('x_weights', x_size)
    h_w = tf.get_variable('h_weights', h_size)
    b = tf.get_variable('biases', bias_size, initializer=tf.constant_initializer(0.0))
    return x_w, h_w, b

scale = 1.0 / math.sqrt(num_embedding_size + num_lstm_nodes[-1]) / 3.0
lstm_init = tf.random_uniform_initializer(-scale, scale)

with tf.variable_scope('lstm_nn', initializer=lstm_init):
    # 生成参数
    with tf.variable_scope('inputs'):
        ix, ih, ib = _generate_params_for_lstm_cell(
            x_size = [num_embedding_size, num_lstm_nodes[0]],
            h_size = [num_lstm_nodes[0], num_lstm_nodes[0]],
            bias_size = [1, num_lstm_nodes[0]],
        )

    with tf.variable_scope('outputs'):
        ox, oh, ob = _generate_params_for_lstm_cell(
            x_size = [num_embedding_size, num_lstm_nodes[0]],
            h_size = [num_lstm_nodes[0], num_lstm_nodes[0]],
            bias_size = [1, num_lstm_nodes[0]],
        )

    with tf.variable_scope('forget'):
        fx, fh, fb = _generate_params_for_lstm_cell(
            x_size = [num_embedding_size, num_lstm_nodes[0]],
            h_size = [num_lstm_nodes[0], num_lstm_nodes[0]],
            bias_size = [1, num_lstm_nodes[0]],
        )

    with tf.variable_scope('memory'):
        cx, ch, cb = _generate_params_for_lstm_cell(
            x_size = [num_embedding_size, num_lstm_nodes[0]],
            h_size = [num_lstm_nodes[0], num_lstm_nodes[0]],
            bias_size = [1, num_lstm_nodes[0]],
        )

    # 定义状态和隐藏状态
    # state 表示上一个cell的细胞状态
    state = tf.Variable(tf.zeros([batch_size, num_lstm_nodes[0]]), trainable=False)
    # h 表示上一个cell的隐藏状态
    h = tf.Variable(tf.zeros([batch_size, num_lstm_nodes[0]]), trainable=False)

    # 根据每一时刻的输入执行cell的计算
    for i in range(num_timesteps):
        # embed_input 表示第t时刻的输入
        embed_input = embed_input[:, i, :]     # 第一个是取batch_size（全取），第二个是取第i个词，第三个维度表示embedding的大小
        embed_input = tf.reshape(embed_input, [batch_size, num_embedding_size])

        # 计算遗忘门的输出
        forget_gate = tf.sigmoid(tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)    # sigmoid(WX + Wh + b)
        # 计算输入门的输出
        input_gate = tf.sigmoid(tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
        # 计算输出门的输出
        output_gate = tf.sigmoid(tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)
        # 中间状态的计算
        mid_state = tf.tanh(tf.matmul(embed_input, cx) + tf.matmul(h, ch) + cb)
        # 计算新的细胞状态
        state = mid_state * input_gate + state * forget_gate         # mid_state * input_gate: 输入门与中间态的点积为记忆门的输出
        # 计算新的隐藏状态
        h = output_gate * tf.tanh(state)

    last = h   # 最后一个cell的输出
```




## api
### get_variabel

```
tf.get_variable(
    name,                       # 变量的名称
    shape = None,               # 变量的形状
    dtype = None,               # 变量的数据类型，默认为 DT_FLOAT
    initializer = None,         # 变量的初始化方式，默认为None，表示使用变量域内已经定义好的初始化方法，如果变量域内没有定义，那么使用 glorot_uniform_initializer 进行初始化
    regularizer = None,         # 一个标准化函数，默认为None，表示使用变量域内已经定义好的标准化函数，如果不存在，那么不进行标准化
    trainable = True,           # 是否加入计算图进行训练，默认为True
    collections = None,         # 将变量加入到图的collections列表中，默认添加到GraphKeys.GLOBAL_VARIABLES中
    caching_device = None,      # 设备字符串或者函数，表明读取变量缓存的位置，默认为变量的设备
    partitioner = None,         # 接收一个完全定义了TensorShape和dtype的变量，并返回一个为每个轴分区的列表
    validate_shape = True,      # 如果为False，那么允许变量被初始化为一个未知shape的值，否则，必须用一个已知shape的值来初始化变量
    use_resource = None,        # 如果为False，创建一个regular的变量，否则，创建一个具有良好语义定义的experimental ResourceVariable，默认为False
    custom_getter = None,
    constraint = None,
)

Returns: 
    返回一个新创建的或已经存在的Tensor变量
```

### initializer 合集
> 该小结是对tensorflow中的各种初始化方法的总结

```
# 常量初始化函数
tf.constant_initializer()

# 满足正态分布的初始化
tf.random_normal_initializer()

# 满足截取的正太分布的初始化
tf.truncated_normal_initializer()

# 满足均匀分布的初始化
tf.random_uniform_initializer()

# 满足均匀分布，但不影响输出数量级的随机值初始化
tf.uniform_unit_scaling_initializer()

# 零初始化
tf.zeros_initializer()

# 一初始化
tf.ones_initializer()
```


## 其他
### slim
> slim 是TensorFlow的简洁版。

#### 示例
```ruby
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 原生卷积层定义
inputs = ...
with tf.variable_scope('conv1') as scope:
    weights = tf.get_variable(scope.name + 'w_1',
                              [3, 3, 3, 16],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    biases = tf.get_variable(scope.name + 'b_1',
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



