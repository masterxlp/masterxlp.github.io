---
layout: post
title:  "[D] Machine Learning Algorithms"
date:   2020-07-23 15:34:00 +0800
categories: ML
tags: 机器学习算法 D
author: Xlp
---
* content
{:toc}


## 简介

> 本部分包含对各种经典的机器算法的总结，包括但不限于SVM、Decision Tree、Naive Bayes、Logistic Regression、Boosting、Bagging等算法。




## SVM
> 支持向量机（Support Vector Machine, SVM）是一种二分类模型，其基本模型定义为 **「特征空间上间隔最大的线性分类器」**，其学习策略为 **「最大化间隔」**，且最终可转化为一个凸二次规划问题的求解。

我们将从四部分来了解支持向量机的原理：线性可分时的SVM、线性不可分时的SVM、目标函数、SMO算法。
具体来讲，线性可分时的SVM最要包括线性分类、函数间隔、几何间隔；线性不可分时的SVM主要包括核函数；目标函数主要包括带约束的目标函数、对偶问题、KKT条件。

### 线性分类
对于一组线性可分的数据，必定能找到一个线性分类器，将这组数据分开。
如果用 $x$ 表示数据，$y \in {-1, 1}$ 表示类别，那么这个线性分类器的学习目标就是要在 $n$ 维的数据空间中找到一个超平面 $w^{T} x + b = 0$ 可以将不同类别的数据分离开来，如图1所示。

<div align="center"><img src="../../../../image/svm/线性分类标准图.jpeg"></div>
<div align="center">图1. 超平面分类示意图</div>

但是，正如图2所示的那样，在解空间中存在着很多个分类超平面都可以将这组数据分离开来，那么究竟哪个分类超平面最好呢？

<div align="center"><img src="../../../../image/svm/线性分类多线图.jpeg"></div>
<div align="center">图2. 多个分类超平面的分类示意图</div>

从容错的角度来讲，我们用于训练的数据仅仅只是一小部分，还有大多数的数据我们无法获得，不能参与训练，因此，这就要求我们的模型能够有一定的容错空间，能够对未知的数据依然分类正确。
从图3来讲，对于图2中的超平面（2）来讲，当加入类别属于-1的新数据（橙色的小球）和类别属于+1的数据（紫色小球）时，它依然可以正确分类；而对于超平面（4），它却无法正确分类。

<div align="center"><img src="../../../../image/svm/超平面的容错性示意图.jpeg"></div>
<div align="center">图3. 超平面的容错性示意图</div>

因此，对于未知数据具有较强的容错能力的分类超平面是最理想的分类超平面。
那么，反应到数学中，如图4所示我们可以通过间隔来表示它：间隔越大，容错性就越强。

<div align="center"><img src="../../../../image/svm/gap.jpeg"></div>
<div align="center">图4. 间隔示意图</div>

我们定义：分类函数 $f(x) = w^T x + b$
- 当 $f(x) = 0$ 时，$x$ 位于分类超平面上；
- 当 $f(x) > 0$ 时，$x$ 位于分类超平面的上方；
- 当 $f(x) < 0$ 时，$x$ 位于分类超平面的下方；

我们定义：$|w^T x + b| = r$ 表示图4中的两条橙色的线，位于这两条线上的点被称为 **「支持向量」**
- 分类超平面仅与支持向量有关，与其他数据无关；
- 两侧的间隔相等；

### 间隔
> 间隔分为函数间隔和几何间隔，其大小为数据到超平面的距离。

#### 函数间隔
在超平面 $w^T x + b = 0$ 确定的情况下，$|w^T x + b|$ 可以表示点 $x$ 到超平面的远近，且当 $(w^T x + b)y > 0$ 时，表明分类正确，反之不正确。

「函数间隔」定义为：  

$$
\begin{align}
\hat{\gamma} = y \cdot (w^T x + b) = y \cdot f(x) \tag{1}
\end{align}
$$ 

超平面 $(w,b)$ 关于数据集 $T$ 中的所有样本点 $(x_i, y_i)$ 的函数间隔最小值，被定为超平面 $(w,b)$ 关于训练集 $T$ 的函数间隔：

$$
\begin{align}
\hat{\gamma} = min \hat{\gamma}_i (i = 1, \cdots, n) \tag{2}
\end{align}
$$

但是，函数间隔存在一个问题：如果成比例的改变 $w$ 和 $b$，则函数间隔的值 $f(x)$ 也成比例的改变。
这是显而易见的，$y$ 是类别固定不变，$f(x) = w^T x + b \rightlefft f'(x) = (2w)^T x + 2b : f'(x) = 2f(x)$。

#### 几何间隔
事实上，我们可以对法向量 $w$ 加以约束，从而得到真正的点到超平面的距离「几何间隔」。

如图5所示，$w$ 使垂直于超平面的一个法向量，$\gamma$ 表示点 $x$ 到超平面的距离，$x'$ 表示点 $x$ 在超平面上的投影。

根据平面几何知识，有：

$$
\begin{align}
x = x' + \gamma \frac{w}{\Vert w \Vert} \tag{3}
\end{align}
$$

其中，$\frac{w}{\Vert w \Vert}$ 为单位向量。

又 $x'$ 在超平面上，有：

$$
\begin{align}
w^T x' + b &= 0 \\
w^T x' &= -b \tag{4}
\end{align}
$$

由式（3）-（4）有：

$$
\begin{align}
w^T x &= w^T x' + \gamma \frac{w^T w}{\Vert w \Vert} \\
\gamma &= \frac{w^T x + b}{\Vert w \Vert} \tag{5} \\
\gamma y &= \frac{y(w^T x + b)}{\Vert w \Vert} \\
\tilde{\gamma} &= \frac{yf(x)}{\Vert w \Vert} \\
\tilde{\gamma} &= \frac{\hat{\gamma}}{\Vert w \Vert} \tag{6}
\end{align}
$$

可以看出，几何间隔就是函数间隔除以 $\Vert w \Vert$。
事实上，函数间隔 $\hat{\gamma} = |f(x)|$，只是人为定义的一个间隔度量，几何间隔 $\frac{|f(x)|}{\Vert w \Vert}$ 才是直观上的点到超平面的距离。

更重要的是，几何间隔不存在函数间隔的缺陷：当 $|f(x)|$ 成比例变化时，$\Vert w \Vert$ 也在成比例变化，两者相互抵消，导致几何间隔不发生改变。

### 目标函数
我们的目标就是在众多的分类超平面中找到那个最优的分类超平面。
而之前我们也提到：最优的分类超平面其实就是容错率最高的（可信度最大的），即间隔最大的分类超平面。
因此，最大间隔分类器的目标函数可定义为：

$$
\begin{align}
&max \tilde(\gamma) \\
s.t.\ y_i(w^T x_i + b) &= \hat{\gamma_i} >= \hat{\gamma} (i = 1, \cdots, n) \tag{7}
\end{align}
$$

令函数间隔 $\hat{\gamma} = 1$时，有目标函数：

$$
\begin{align}
&max \frac{1}{\Vert w \Vert} \\
s.t.\ y_i(w^T x_i + b) &>= 1 (i = 1, \cdots, n) \tag{8}
\end{align}
$$







## Decision Tree


## Naive Bayes


## Logistic Regression


## Boosting


## Bagging