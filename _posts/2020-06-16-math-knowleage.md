---
layout: post
title:  "[G] 重要的数学知识"
date:   2020-06-16 16:54:00 +0800
categories: math
---

### 01. KL散度
主要内容来源于[维基百科](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5)

> KL散度（Kullback-Leibler divergence, KLD），在信息系统中称为 **相对熵** （relative entropy），在连续时间序列中称为 **随机性** （randomness），
> 在统计模型推断中称为 **信息增益** （information gain），也称为 **信息散度**（information divergence）。

KL散度是两个概率分布P和Q差别的非对称性的度量。
KL散度是用来度量使用基于Q的分布来编码服从P的分布的样本所需的额外的平均比特数。
KL散度在不同的场景下代表的具体函数不尽相同，但其本质都大同小异。
典型情况下，P表示数据的真实分布，Q表示数据的理论分布、估计的模型分布、或P的近似分布。

#### 定义
对于离散随机变量，其概率分布P和Q的KL散度定义为

$$
\begin{align}
D_{KL}(P||Q) &= -\sum_i P(i) \mathop{ln} \frac{Q(i)}{p(i)} \\
&= \sum_i P(i) \mathop{ln} \frac{P(i)}{Q(i)} \tag{1.1}
\end{align}
$$

即：按概率P求得的P和Q的对数商的平均值。
KL散度仅当概率P和Q各自总和均为1，且对于任何 $i$ 满足 $Q(i) > 0, p(i) > 0$ 时，才有定义。
式中出现 $0 \mathop{ln} 0$ 的情况，其值按 $0$ 处理。

对于连续随机变量，其概率分布P和Q可按积分形式定义

$$
\begin{align}
D_{KL}
$$

### 02. 矩阵分解