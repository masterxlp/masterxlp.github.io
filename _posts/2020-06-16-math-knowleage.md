---
layout: post
title:  "[G] 重要的数学知识"
date:   2020-06-16 16:54:00 +0800
categories: math
tags: essential G
author: Xlp
---
* content
{:toc}

## 01. KL散度
主要内容来源于[维基百科](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5)

> KL散度（Kullback-Leibler divergence, KLD），在信息系统中称为 **相对熵** （relative entropy），在连续时间序列中称为 **随机性** （randomness），
> 在统计模型推断中称为 **信息增益** （information gain），也称为 **信息散度**（information divergence）。





KL散度是两个概率分布$P$和$Q$差别的 **非对称性** 的度量。
KL散度是用来度量使用基于$Q$的分布来编码服从P的分布的样本所需的额外的平均比特数。
KL散度在不同的场景下代表的具体函数不尽相同，但其本质都大同小异。
典型情况下，$P$表示数据的真实分布，$Q$表示数据的理论分布、估计的模型分布、或P的近似分布。

### 定义
对于离散随机变量，其概率分布$P$和$Q$的KL散度定义为

$$
\begin{align}
D_{KL}(P||Q) &= -\sum_i P(i) \mathop{ln} \frac{Q(i)}{p(i)} \\
&= \sum_i P(i) \mathop{ln} \frac{P(i)}{Q(i)} \tag{1.1}
\end{align}
$$

即：按概率$P$求得的$P$和$Q$的对数商的平均值。
KL散度仅当概率$P$和$Q$各自总和均为1，且对于任何 $i$ 满足 $Q(i) > 0, p(i) > 0$ 时，才有定义。
式中出现 $0 \mathop{ln} 0$ 的情况，其值按 $0$ 处理。

对于连续随机变量，其概率分布P和Q可按积分形式定义

$$
\begin{align}
D_{KL}(P||Q) = \int_{-\infty}^{+\infty} p(x) \mathop{ln} \frac{p(x)}{q(x)} dx \tag{1.2}
\end{align}
$$

其中$p$和$q$分布表示分布$P$和$Q$的密度。

更一般地，若$P$和$Q$为集合$X$的概率测度，且 $P$ 关于 $Q$ 绝对连续，则从 $P$ 到 $Q$ 的KL散度定义为

$$
\begin{align}
D_{KL}(P||Q) = \int_{X} \mathop{ln} \frac{dP}{dQ}dP \tag{1.3}
\end{align}
$$

其中，假定右侧的表达形式存在，则 $\frac{dP}{dQ}$ 为 $Q$ 关于 $P$ 的 R-N 导数。

相应地，若$P$关于$Q$绝对连续，则

$$
\begin{align}
D_{KL}(P||Q) &= \int_{X} \mathop{ln} \frac{dP}{dQ}dP \\
&= \int_{X} \frac{dP}{dQ} \mathop{ln} \frac{dP}{dQ} dQ \tag{1.4}
\end{align}
$$

### 特性

- 相对熵的值为非负数：$D_{KL}(P \parallel Q) >= 0$
- 由吉布斯不等式可知，当且仅当 $P = Q$ 时，$D_{KL}(P \parallel Q) = 0$
- 尽管从直觉上KL散度是一个度量或距离函数，但是事实上，它并不是一个真正的度量或距离
- KL散度不具有对称性：从分布$P$到$Q$的距离通常并不等于从$Q$到$P$的距离，即 $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$

### KL散度和其他量的关系

- 自信息与KL散度
  - $I(X)$ 表示 $X$ 的信息量，其本身就是个随机变数

$$
\begin{align}
\mathop{I}(m) &= D_{KL}(\delta_{im}||{p_i})$ \\
&= - ln(P(X))
\end{align}
$$

- 互信息和KL散度

$$
\begin{align}
\mathop{I}(X;Y) &= D_{KL}(P(X,Y) || P(X)P(Y)) \\
&= E_{X}[D_{KL}(P(Y|X)||P(X))] \\
&= E_{Y}[D_{KL}(P(X|Y)||P(X))]
\end{align}
$$

- 信息熵和KL散度

$$
\begin{align}
H(X) &= E_{x}[I(x)] \\
&= log N - D_{KL}(P(X) \parallel P_{U}(X)) \\
&= E [- ln (P(X))] \\
&= \sum_i P(x_i)I(x_i) = - \sum_i P(x_i) ln P(x_i)
\end{align}
$$

- 条件熵和KL散度

$$
\begin{align}
H(X \mid Y) &= log N - D_{KL}(P(X, Y) \parallel P_{U}(X)P(Y)) \\
&= log N - D_{KL}(P(X, Y) \parallel P(X)P(Y)) - D_{KL}(P(X) \parallel P_{U}(X)) \\
&= H(X) - I(X; Y) \\
&= log N - E_{Y}[D_{KL}(P(X \mid Y) \parallel P_{U}(X))]
\end{align}
$$

- 交叉熵和KL散度

$$
\begin{align}
H(p, q) = E_{p}[-log q] = H(p) + D_{KL}(p \mid q)
\end{align}
$$

## 02. 矩阵分解















































