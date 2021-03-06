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
> 具体概括为以下三种情况：
>> 当训练样本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机；  
>> 当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支持向量机；  
>> 当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。  

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
> 为什么要采用间隔最大化？  
>> 当训练数据线性可分时，存在无穷个分离超平面可以将两类数据正确分开。
>> 感知机利用误分类最小策略，求得分离超平面，不过此时的解有无穷多个。
>> 线性可分支持向量机利用间隔最大化求得最优分离超平面，这时，解是唯一的。
>> 另一方面，此时的分隔超平面所产生的分类结果是最鲁棒的，对未知实例的泛化能力最强。

#### 函数间隔
> 一个点距离分离超平面的远近可以表示分类预测的确信程度，在超平面 $w^T x + b = 0$ 确定的情况下，$\vert w^T x + b \vert$ 能够表示点 $x$ 距离超平面的远近，而 $w^T x + b$ 的符号与类标记 $y$ 的符号是否一致能够表示分类是否正确
> 所以，可用 $y(w^T x + b)$ 来表示分类的正确性和确信度。 

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
这是显而易见的，$y$ 是类别固定不变，$f(x) = w^T x + b \rightarrow f'(x) = (2w)^T x + 2b : f'(x) = 2f(x)$。

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
&w^T x' + b = 0 \\
&w^T x' = -b \tag{4}
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

更重要的是，几何间隔不存在函数间隔的缺陷：当 $\vert f(x) \vert$ 成比例变化时，$\Vert w \Vert$ 也在成比例变化，两者相互抵消，导致几何间隔不发生改变。

### 目标函数
> SVM 有两个目标：第一个是使间隔最大化，第二个是使样本正确分类；
>> 间隔最大化表现为：$max_{w, b} \frac{1}{\Vert w \Vert}$;  
>> 使样本分类正确表现为约束：$y_i(w^T x_i &+ b) \geq 1 (i = 1, \cdots, n)$;   

> 间隔最大化的直观解释：对训练数据集找到几何间隔最大的超平面意味着以充分大的确信度对训练数据进行分类。  
> 也就是说，不仅将正负实例点分开，而且对最难分的实例点（离超平面近的点）也有足够大的确信度将它们分开，这样超平面应对未知的新实例有很好的分类预测能力。

我们的目标就是在众多的分类超平面中找到那个最优的分类超平面。
而之前我们也提到：最优的分类超平面其实就是容错率最高的（可信度最大的），即间隔最大的分类超平面。
因此，最大间隔分类器的目标函数可定义为：

$$
\begin{align}
&max\ \tilde{\gamma} \\
s.t.\ y_i(w^T x_i + b) &= \hat{\gamma_i} \geq \hat{\gamma} (i = 1, \cdots, n) \tag{7}
\end{align}
$$

事实上，函数间隔 $\hat{\gamma}$ 的取值并不影响最优化问题的解。
因此，令函数间隔 $\hat{\gamma} = 1$时，有目标函数：

$$
\begin{align}
&max_{w,b}\ \frac{1}{\Vert w \Vert} \\
s.t.\ y_i(w^T x_i &+ b) \geq 1 (i = 1, \cdots, n) \tag{8}
\end{align}
$$

如图4所示，中间的绿色的线就是最优超平面，其到两个橙色线的距离相等，这个距离便是几何间隔 $\tilde{\gamma}$，橙线上的线则是支持向量，且满足 $y(w^T x + b) = 1$，而对于所有不是支持向量的点，有 $y(w^T x + b) > 1$。

式（8）等价于：

$$
\begin{align}
&min_{w,b}\ \frac{1}{2}\Vert w \Vert^2 \\
s.t.\ y_i(w^T x_i &+ b) \geq 1 (i = 1, \cdots, n) \tag{9}
\end{align}
$$

即，最大化 $\frac{1}{\Vert w \Vert}$ 就等价于 最小化 $\frac{1}{2}\Vert w \Vert^2$。

### 对偶问题
式（9）可以通过拉格朗日对偶性来进行求解。

那么什么是拉格朗日对偶性呢？

简单来讲，通过给每一个约束条件加上一个拉格朗日乘子 $\alpha$，定义拉格朗日函数（通过拉格朗日乘子将约束条件融合到目标函数中去）：

$$
\begin{align}
L(w, b, \alpha) = \frac{1}{2} \Vert w \Vert^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1] \tag{10}
\end{align}
$$

然后令

$$
\begin{align}
\theta(w) = max_{\alpha_i \geq 0}\ L(w, b, \alpha) \tag{11}
\end{align}
$$

显然对于式（11）有：当约束条件不满足时，有 $y_i(w^T x_i + b) < 1$，显然有 $\theta(w) = \infty$（只要令 $\alpha = \infty$ 即可）；
而当约束条件满足时，最优值为 $\theta(w) = \frac{1}{2}\Vert w \Vert^2$。
简单来讲就是说，当满足约束时约束，第二项为正值，

因此，为了使式（11）等价于式（9），我们需要最小化式（11）：

$$
\begin{align}
min_{w,b}\ \theta(w) = min_{w,b} max_{\alpha_i \geq 0}\ L(w, b, \alpha) = p^{\ast} \tag{12}
\end{align}
$$

这里 $p^{\ast}$ 是式（12）的最优解，即为原始问题的最优解，且和最初问题的解是等价的。

由于先求最小值再求最大值不好求解，因此交换最小最大为最大最小，有：
$$
\begin{align}
max_{\alpha_i \geq 0}min_{w,b}\ L(w, b, \alpha) = d^{\ast} \tag{13}
\end{align}
$$

交换之后的新问题是交换之前原始问题的对偶问题，这个新问题的最优解为 $d^{\ast}$ ，且满足 $d^{\ast} \leq p^{\ast}$，在 **「KKT条件」** 下，等号成立。

之所以从极小极大（minmax）的原始问题转化为极大极小（maxmin）的对偶问题是因为：
- $d^{\ast}$ 是 $p^{\ast}$ 的近似解；
- 对偶问题更容易求解；

### KKT条件
#### 一般对偶问题的KKT条件
对于含有不等式约束的优化问题，将其转化为对偶问题：

$$
\begin{align}
&max_{a,b}\ min_{x} L(a, b, x) \\
s.&t.\ a_i \geq 0 ;\ i = 1, \cdots, n \tag{14}
\end{align}
$$

其中，$L(a, b, x)$ 为由所有不等式约束、等式约束和目标函数全部写成的一个式子：$L(a, b, x) = f(x) + a \cdot g(x) + b \cdot h(x)$，
KKT条件是说原始问题最优值 $x^{\ast}$ 和对偶问题最优值 $a^{\ast}, b^{\ast}$ 必须满足一下条件：

$$
\begin{align}
&1.\ \nabla_x L(a^{\ast}, b^{\ast}, x^{\ast}) = 0,\ \nabla_a L(a^{\ast}, b^{\ast}, x^{\ast}) = 0,\ \nabla_b L(a^{\ast}, b^{\ast}, x^{\ast}) = 0; \tag{15} \\
&2.\ a^{\ast} \cdot g_i(x^{\ast}) = 0; \tag{16} \\
&3.\ g_i(x^{\ast}) \le 0; \tag{17} \\
&4.\ a_{i}^{\ast} \geq 0,\ h_j(x) = 0. \tag{18}
\end{align}
$$

当原始问题的解和对偶问题的解满足KKT条件，并且 $f(x), g_i(x)$ 是凸函数时，原始问题的解和对偶问题的解相等。

#### SVM目标函数的对偶函数的KKT条件
由式（10）、（15）-（18）可知，KKT条件为：

$$
\begin{align}
\nabla_w L(w^{\ast}, b^{\ast}, \alpha^{\ast}) = w^{\ast} - \sum_{i=1}^{n} \alpha_i x_i y_i = 0& \tag{19} \\
\nabla_b L(w^{\ast}, b^{\ast}, \alpha^{\ast}) = - \sum_{i=1}^{n} \alpha_i y_i = 0& \tag{20} \\
\alpha_{i}^{\ast} g_i(w^{\ast}) = 0,\ i = 1, \cdots, n& \tag{21} \\
g_i(w^{\ast}) \le 0,\ i = 1, \cdots, n& \tag{22} \\
\alpha_{i}^{\ast} \geq 0,\ i = 1, \cdots, n \tag{23}
\end{align}
$$

其中，$g_i(w^{\ast}) = y_i(w^{\ast T} x_i + b) - 1$。

### 求解对偶问题的三个步骤
**「第一步」**   
首先固定 $\alpha$，让 $L$ 对 $w, b$ 最小化；   
由式（19）-（20）带入式（10）可得：

$$
\begin{align}
L(w, b, \alpha) &= \frac{1}{2} \Vert w \Vert^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1] \\
&= \frac{1}{2} \sum_{i, j = 1}^{n} \alpha_i \alpha_j y_i y_j x_{i}^{T} x_j - \sum_{i=1}^{n} \alpha_i [y_i (\sum_{j=1}^{n} \alpha_j y_j x_{i}^{T}  x_j  + b) - 1] \\
&= \frac{1}{2} \sum_{i, j = 1}^{n} \alpha_i \alpha_j y_i y_j x_{i}^{T} x_j - \sum_{i, j = 1}^{n} \alpha_i \alpha_j y_i y_j x_{i}^{T} x_j - \sum_{i=1}^{n}\alpha_i y_i b + \sum_{i=1}^{n}\alpha_i \\
&= \sum_{i=1}^{n}\alpha_i - \frac{1}{2} \sum_{i, j = 1}^{n} \alpha_i \alpha_j y_i y_j x_{i}^{T} x_j \tag{24}
\end{align}
$$

**「第二步」**  
对 $\alpha$ 最大化  

$$
\begin{align}
max_{\alpha}\ \sum_{i=1}^{n} &\alpha_i - \frac{1}{2} \sum_{i,j = 1}{n} \alpha_i \alpha_j y_i y_j x_{i}^{T} x_j \\
s.t.\ &\alpha_i \geq 0,\ i = 1, \cdots, n \\
&\sum_{i=1}^{n} \alpha_i y_i = 0 \tag{25}
\end{align}
$$

式（24）只包含了变量 $\alpha_i$，因此只要得到式（24）的最优值 $\alpha^{\ast}$，就可以得到原问题的最优解 $w^{\ast} = \sum_{i=1}^{n} \alpha_i y_i x_i$，
然后通过

$$
\begin{align}
b^{\ast} = - \frac{max_{i:y_i = -1} w^{\ast T} x_i + min_{i:y_i = 1} w^{\ast T} x_i}{2} \tag{26}
\end{align}
$$

**「第三步」**  
利用SMO算法求解对偶因子 $\alpha$


### SMO算法


### 核函数
在实际应用中，大部分任务都是线性不可分的，那么该怎么做呢？

对于线性不可分的数据，若是将特征空间通过高维映射，那么映射后的特征在高位特征空间是线性可分的。
假设映射函数为 $\phi$，那么根据经典SVM有：

$$
\begin{align}
&1.\ f(\phi(x)) = w^T \phi(x) + b \\
&2.\ min_{w,b}\ \frac{1}{2} \Vert w \Vert^2\ \ s.t.\ y_i (w^T \phi(x) + b) \geq 1 \\
&3.\ L(w, b, \alpha) = \frac{1}{2} \Vert w \Vert^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T \phi(x) + b) - 1] \\
&4.\ \theta(\phi(x)) = max_{\alpha_i \geq 0} L(w, b, \alpha) \\
&5.\ min_{w, b}\ \theta(\phi(x)) = min_{w,b}\ max_{\alpha}\ L(w, b, \alpha) = p^{\ast} \\
&6.\ max_{\alpha}\ min_{w,b}\ L(w, b, \alpha) = q^{\ast} \le p{\ast} \\
&7.\ KKT : w^{\ast} = \sum_i \alpha_i y_i \phi(x)_i,\ \sum_{i=1}^{n} \alpha_i y_i = 0,\ \alpha_i[y_i(w^T \phi(x_i) + b) - 1] = 0,\ \alpha_i \geq 0,\ y_i(w^T \phi(x_i) + b) - 1 \geq 0 \\
&8.\ max_{\alpha} = \sum_{i=1}^{n} \alpha_i - \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \phi(x_i) \phi(x_j)\ \ s.t.\ \sum_{i=1}^{n} \alpha_i y_i = 0,\ \alpha_i \geq 0,\ i = 1, \cdots, n
\end{align}
$$

但是由于映射后的特征维度很高，如果直接做点积运算，会发生维度爆炸。
幸运的是，核函数的值与高位特征的点积结果相同，这样我们首先不用真正的做高维映射，其次我们不用真正的计算出高位特征向量，而是使用原始特征进行简单的核运算就可以了。
用数学表达就是：$K(x, y) = \phi(x) \cdot \phi(y)$.

常用的核函数有：
- 线性核：$K(x, y) = x^T y + c$
- 多项式核：$K(x, y) = (a x^T y + c)^d$
- 高斯核（或称为径向基核，RBF）：$K(x, y) = exp(- \frac{\Vert x - y \Vert^2}{2 \sigma^2})$
- 幂指数核：$K(x, y) = exp(- \frac{\Vert x - y \Vert}{2 \sigma^2})$
- 拉普拉斯核：$K(x, y) = exp(- \frac{\Vert x - y \Vert}{\sigma})$

如何选择核函数：
- 当特征维数 `d` 超过样本数 `m` 时 (文本分类问题通常是这种情况), 使用线性核;
- 当特征维数 `d` 比较小，样本数 `m` 中等时, 使用RBF核;
- 当特征维数 d 比较小. 样本数 m 特别大时, 支持向量机性能通常不如深度神经网络;


### 松弛变量
不管是在原特征空间还是高维映射特征空间，我们都假设样本线性可分。
虽然理论上我们总能找到一个高维映射是数据线性可分，但是在实际任务中，寻找一个合适的核函数很困难。
此外，由于数据有噪音存在，一味追求数据的线性可分可能会使模型陷入过拟合。
因此，我们 **放宽对样本的要求，允许少量样本分类错误。**

基于此，我们对之前的目标函数进行修改：我们在之前的目标函数上加入一个误差，这就相当于，我们允许原先的目标出错。
引入松弛变量 $\xi_i \geq 0$，目标函数变为

$$
\begin{align}
min&_{w, b, \xi} \frac{1}{2} \Vert w \Vert^2 + C \sum_{i=1}^{n} \xi_i \\
s.t.\ &y_i(w^T x_i + b) \geq 1 - \xi_i \\
&\xi_i \geq 0,\ i = 1, \cdots, n \tag{27}
\end{align}
$$

其中，$\xi_i = l_{hinge}(z) = max(0, 1 - y_i(w^T x_i + b))$

### SVM的优缺点
优点：
- 由于SVM是一个凸优化问题，所以求得的解一定是全局最优而不是局部最优。
- 不仅适用于线性线性问题还适用于非线性问题(用核技巧)。
- 拥有高维样本空间的数据也能用SVM，这是因为数据集的复杂度只取决于支持向量而不是数据集的维度，这在某种意义上避免了“维数灾难”。
- 理论基础比较完善(例如神经网络就更像一个黑盒子)。

缺点：
- 二次规划问题求解将涉及m阶矩阵的计算(m为样本的个数), 因此SVM不适用于超大数据集(SMO算法可以缓解这个问题)。
- 只适用于二分类问题。(SVM的推广SVR也适用于回归问题；可以通过多个SVM的组合来解决多分类问题)

### SVM为什么对缺失数据敏感？
这里说的缺失数据是指缺失某些特征数据，向量数据不完整。
SVM 没有处理缺失值的策略。
而 SVM 希望样本在特征空间中线性可分，所以特征空间的好坏对SVM的性能很重要。
缺失特征数据将影响训练结果的好坏。

## Decision Tree
> 决策树是一种基本的分类和回归方法。
> 在分类问题中，表示基于特征对实例进行分类的过程，它可以认为是 「if-then」 规则的集合，也可以认为是定义在特征空间与类空间上的条件概率分布，
> 其主要优点是模型具有可读性，分类速度快。
> 决策树学习通常包含3个步骤：特征选择、决策树生成、决策树修剪。

**定义「决策树」**  
分类决策树模型是一种描述对实例进行分类的树形结构。
决策树由结点和有向边组成，结点有两种类型：内部结点和叶结点。
内部结点表示一个特征或属性，叶结点表示一个类。

- 决策树的路径或其对应的 if-then 规则集合具有一个重要性质：互斥且完备；
- 决策树学习本质上就是从训练集中归纳出一组分类规则；
- 决策树学习的目标就是生成一颗与训练数据矛盾较小的决策树，同时具有很好的泛化能力；
- 决策树学习的损失函数通常是正则化的极大似然函数；
- 决策树学习的策略是以损失函数为目标函数的最小化；
- 由于从所有可能的决策树中选取最优决策树是NP完全问题，因此，在实际中，通常采用启发式的方法，近似求解这一最优化问题，这样得到的决策树是次最优的；
- 决策树学习的算法通常是一个递归地选择最优特征，并根据该特征对训练数据进行分割，使得对各个子数据集有一个最好的分类的过程；
- 决策树的生成对应于模型的局部选择，决策树的剪枝对应于模型的全局选择；
- 决策树的生成只考虑局部最优，相对地，决策树的剪枝则是考虑全局最优；

### 特征选择
> 特征选择在于选取对训练数据具有分类能力的特征，这样可以提高决策树的学习的效率。

#### 信息增益
直观上，如果一个特征具有更好的分类能力，或者说，按照这一特征将数据集分割子集，使得各个子集在当前条件下有最好分类，那么这个特征就应该被选择以划分特征空间。

信息增益（information gain）就能够很好的表示这一直观的准则。

**「熵」**   
熵（entropy）是表示随机变量不确定性的度量。

设 $X$ 是一个取有限个值的离散随机变量，其概率分布为

$$
\begin{align}
P(X = x_i) = p_i,\ i = 1, 2, \cdots, n
\end{align}
$$

则随机变量 $X$ 的熵定义为

$$
\begin{align}
H(X) = -\ \sum_{i=1}^{n}\ p_i\ log\ p_i \tag{1}
\end{align}
$$

其中，若 $p_i = 0$，则定义 $0\ log\ 0 = 0$，通常，式（1）中的对数以 $2$ 或者 $e$ 为底，这时熵的单位分别被称为 比特 或 纳特。

由定义可知，熵只依赖于 $X$ 的分类，而与 $X$ 的取值无关，所以 $X$ 的熵也可以记为 $H(p)$ 

$$
\begin{align}
H(p) = -\ \sum_{i=1}^{n}\ p_i\ log\ p_i \tag{2}
\end{align}
$$

- 熵越大，随机变量的不确定性就越大；
  - 当 $p = 0$ 或 $p = 1$ 时 $H(p) = 0$，此时随机变量完全没有不确定性；
  - 当 $p = 0.5$ 时，$H(p) = 1$，熵的取值最大，随机变量的不确定性也最大；

**「条件熵」**    
条件熵（conditional entropy）表示在已知随机变量 $X$ 的条件下，随机变量 $Y$ 的不确定性。
- **条件熵越小，说明条件 $X$ 对 $Y$ 在分类时的决定性或重要性就越高；**

条件熵 $H(Y \mid X)$ 定义为：

$$
\begin{align}
H(Y \mid X) = \sum_{i=1}^{n}\ p_i\ H(Y \mid X = x_i) \tag{3}
\end{align}
$$

其中，$p_i = P(X = x_i),\ i=1, 2, \cdots, n$.

- 当熵和条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的熵和条件熵分别被称为经验熵和经验条件熵；

**「信息增益」**   
信息增益（information gain）表示得知特征 $X$ 的信息而使得类 $Y$ 的信息的不确定性减少的程度。

定义信息增益：特征 $A$ 对训练集 $D$ 的信息增益 $g(D, A)$，定义为集合 $D$ 的经验熵 $H(D)$ 与特征 $A$ 给定的条件下，$D$ 的经验条件熵 $H(D \mid A)$ 之差，即

$$
\begin{align}
g(D, A) = H(D) - H(D \mid A) \tag{4}
\end{align}
$$

- 一般地，熵 $H(Y)$ 与条件熵 $H(Y \mid X)$ 之差称为互信息；
- 决策树中的信息增益等价于训练数据集中类和特征的互信息；
- 信息增益的含义：由特征 $A$ 而使得对数据集 $D$ 的分类的不确定性减少的程度；
- 显然，信息增益越大，特征的分类能力就越强；

**「信息增益的算法」**   
```
Information Gain Algorithm

Input: 训练数据 D 和 特征 A

Start
01. 计算数据集 D 的经验熵 H(D)
```

$$
\begin{align}
H(D) = - \sum_{k=1}^{K} \frac{\vert C_k \vert}{\vert D \vert} log_2 \frac{\vert C_k \vert}{\vert D \vert} \tag{5}
\end{align}
$$

```
02. 计算特征 A 对数据集 D 的条件经验熵 H(D | A)
```

$$
\begin{align}
H(D \mid A) &= \sum_{i = 1}^{n} \frac{\vert D_i \vert}{\vert D \vert} H(D_i) \\
&= \sum_{i=1}^{n} \frac{\vert D_i \vert}{\vert D \vert} \sum_{k=1}^{K} \frac{\vert D_{ik} \vert}{\vert D_i \vert} log_2 \frac{\vert D_{ik} \vert}{\vert D_i \vert} \tag{6}
\end{align}
$$

```
03. 计算信息增益 g(D, A)
```

$$
\begin{align}
g(D, A) = H(D) - H(D \mid A)
\end{align}
$$

```
Output: 特征 A 对训练集 D 的信息增益 g(D, A).
```


其中，$\vert D \vert$ 表示样本总量，$\vert C_k \vert$ 表示属于类 $C_k$ 的样本的个数，$\vert D_i \vert$ 表示子集 $D_i$ 的样本个数，$D_{ik}$ 表示子集 $D_i$ 中属于类 $C_k$ 的样本的集合，$D_{ik}$ 表示$D_{ik}$ 的样本个数。

#### 信息增益比
> 以信息增益作为划分训练集的特征，存在偏向于选择特征取值较多的特征的问题。

信息增益比（information gain ratio）可以对这一问题进行矫正。

定义信息增益比：特征 $A$ 对训练集 $D$ 的信息增益 $g_R(D, A)$，定义为其信息增益 $g(D, A)$ 与训练数据集 $D$ 关于特征 $A$ 的值的熵 之比，即

$$
\begin{align}
g_R(D,A) = \frac{g(D, A)}{H_A(D)} \tag{7}
\end{align}
$$

其中，$H_A(D) = - \sum_{i=1}^{n} \frac{\vert D_i \vert}{\vert D \vert} log_2 \frac{\vert D_i \vert}{\vert D \vert}$，$n$ 是特征 $A$ 取值的个数。

### 决策树的生成
#### ID3算法
> ID3 算法的核心是在决策树各个结点上应用信息增益准则来选择特征。
> ID3 算法相当于用极大似然法进行概率模型的选择。

```
ID3 算法

Input: 训练数据集 D，特征集 A，阈值 epsilon

Start
01. 若 D 中所有实例属于同一类 C_k，则 T 为单结点树，并将 C_k 作为该结点的类标记，返回 T；
02. 若 A 为空集，则 T 为单结点树，并将 D 中的实例数最多的类 C_k 作为该结点的类标记，返回 T；
03. 否则，计算 A 中各特征对 D 的信息增益，选择信息增益最大的特征 A_g；
04. 如果 A_g 的信息增益小于阈值 epsilon，则置 T 为单结点树，并将 D 中实例数最多的类 C_k 作为该结点的类标记，返回 T；
05. 否则，对 A_g 的每一个可能值 a_i，依 A_g = a_i 将 D 分割为若干个非空子集 D_i，将 D_i 中实例数最多的类作为标记，构建子结点，由结点和子结点构成树 T，返回 T；
06. 对第 i 个子结点，以 D_i 为训练集，以 A - {A_g} 为特征集，递归地调用第（01）-（05）步，得到子树 T_i，返回 T_i；

Output：决策树 T.
```

- ID3 算法只有树的生成，所以该算法生成的树容易过拟合

#### C4.5算法
> C4.5算法与ID3算法相似，C4.5 在树的生成过程中，使用信息增益比来选择特征.

```
C4.5 算法

Input: 训练数据集 D，特征集 A，阈值 epsilon

Start
01. 若 D 中所有实例属于同一类 C_k，则 T 为单结点树，并将 C_k 作为该结点的类标记，返回 T；
02. 若 A 为空集，则 T 为单结点树，并将 D 中的实例数最多的类 C_k 作为该结点的类标记，返回 T；
03. 否则，计算 A 中各特征对 D 的信息增益比，选择信息增益比最大的特征 A_g；
04. 如果 A_g 的信息增益小于阈值 epsilon，则置 T 为单结点树，并将 D 中实例数最多的类 C_k 作为该结点的类标记，返回 T；
05. 否则，对 A_g 的每一个可能值 a_i，依 A_g = a_i 将 D 分割为若干个非空子集 D_i，将 D_i 中实例数最多的类作为标记，构建子结点，由结点和子结点构成树 T，返回 T；
06. 对第 i 个子结点，以 D_i 为训练集，以 A - {A_g} 为特征集，递归地调用第（01）-（05）步，得到子树 T_i，返回 T_i；

Output：决策树 T.
```

### 决策树的剪枝
> 在决策树学习中，将已生成的树进行简化的过程称为剪枝。
> 具体地，剪枝从已生成的树上裁掉一些叶子或子树，并将其根结点或父结点作为新的叶结点，从而简化分类树模型。

决策树的剪枝往往通过极小化决策树整体的损失函数来实现。  
设树 T 的叶结点个数为 $\vert T \vert$，t 是树 T 的叶结点，该叶结点有 N 个样本点，其中 k 类的样本点有 $N_{tk}$ 个，$k = 1, 2, \cdots K$，$H_t(T)$ 为叶结点 t 上的经验熵，$\alpha \geq 0$ 为参数，则决策树学习的损失函数
可以定义为

$$
\begin{align}
C_{\alpha}(T) = \sum_{t=1}^{\vert T \vert} N_tH_t(T) + \alpha \vert T \vert \tag{8}
\end{align}
$$

其中，经验熵为 

$$
\begin{align}
H_t(T) = - \sum_{k} \frac{N_{tk}}{N_t} log \frac{N_{tk}}{N_t} \tag{9}
\end{align}
$$

在损失函数中，记式（8）右边的第一项为

$$
\begin{align}
C(T) = \sum_{t=1}^{\vert T \vert} N_tH_t(T) = -\sum_{t=1}^{\vert T \vert} \sum_{k=1}{K} N_{tk} log \frac{N_{tk}}{N_t} \tag{10}
\end{align}
$$

于是，式（8）可重写为

$$
\begin{align}
C_{\alpha} (T) = C(T) + \alpha \vert T \vert \tag{11}
\end{align}
$$

- $C(T)$ 表示模型对训练数据的预测误差，即模型与训练数据的拟合程度，$\vert T \vert$ 表示模型复杂度，参数 $\alpha \geq 0$ 控制两者之间的影响；
  - 当 $\alpha = 0$ 时，意味着只考虑模型与训练集的拟合程度，不考虑模型的复杂度；
  - 当 $\alpha$ 较大时，促使选择较简单的模型（树）；
  - 当 $\alpha$ 较小时，促使选择较复杂的模型（树）；
- 剪枝，就是当 $\alpha$ 确定时，选择损失函数最小的模型（树），即损失函数最小的子树；
- 决策树的生成只考虑提高信息增益（比），以对训练数据进行更好的拟合；
- 而剪枝则是通过优化损失函数的同时，还考虑减小模型的复杂度；

**「剪枝算法」**   
```
树的剪枝算法

Input：生成算法产生的整个树 T，参数 alpha

Start
01. 计算每个结点的经验熵；
02. 递归的从树的叶结点向上回缩：
    假设一组叶结点回缩到其父结点之前和之后的整体树分别为 T_B, T_A，
    其对应的损失函数值分别是 C_{\alpha}(T_B), C_{\alpha}(T_A)，
    如果 C_{\alpha}(T_A) <= C_{\alpha}(T_B)，则进行剪枝，即将父结点变为新的叶结点.
03. 返回（02），直至不能继续为止，得到损失函数最小的子树 T_{\alpha}
```

### CART 算法
**「基尼指数」**  
集合 D 的基尼指数（CART）

$$
\begin{align}
Gini(D) &= 1 - \sum_{k=1}^{K} (\frac{\vert C_k \vert}{\vert D \vert})^2 \\
&= 1 - \sum_{k=1}^{K} p_{k}^{2} \tag{12}
\end{align}
$$

特征 A 条件下集合 D 的基尼指数

$$
\begin{align}
Gini(D, A) &= \frac{\vert D_1 \vert}{\vert D \vert}Gini(D_1) + \frac{\vert D_2 \vert}{\vert D \vert}Gini(D_2) \tag{13}
\end{align}
$$

**「CART生成算法」**  
```
CART 生成算法

Input: 训练数据集 D，停止计算的条件

Start
01. 根据训练数据集，从根结点开始，递归地对每个结点进行以下操作，构建二叉决策树：
02.     设结点的训练数据集为 D，计算现有特征对该数据集的基尼指数，此时对每一个特征 A，
        对其可能取的每个值 a，根据样本点 A = a 的测试为“是”或“否”将 D 分割成 D_1 和 D_2 两部分，
        利用式（13）计算 A = a 时的基尼指数；
03.     在所有可能的特征 A 以及它们所有可能的切分点 a 中，选择基尼指数最小的特征及其对应的切分点作为最优特征和最优切分点，
        依最优特征与最优切分点，从现结点生成两个子结点，将训练数据集依特征分配到两个子结点中去；
04.     对两个子结点递归地调用（01）和（02）步，直至满足停止条件；
05.     生成 CART 树；

Output：CART决策树
```

- 算法停止计算的条件是结点中的样本数小于预定阈值，或样本集的基尼指数小于预定阈值（样本基本属于同一类），或者没有更多特征；

```
CART 剪枝算法

Input：CART 算法生成的决策树 T_0

Start
01. 设 k = 0，T = T_0
02. 设 alpha = + infty
03. 自下而上地对各内部结点 t 计算 C(T_t)，|T_t| 以及 g(t) = (C(t) - C(T_t)) / (|T_t| - 1)
    alpha = min(alpha, g(t))，这里 T_t 以 t 为根结点的子树，C(T_t) 是对训练数据的预测误差，
    |T_t| 是 T_t 的叶结点数；
04. 对 g(t) = alpha 的内部结点 t 进行剪枝，并对叶结点 t 以多数表决法决定其类，得到树 T；
05. 设 k = k + 1, alpha_k = alpha, T_k = T；
06. 如果 T_k 不是由根结点及两个叶结点构成的树，则返回步骤（03）；否则令 T_k = T_n；
07. 采用交叉验证法在子树序列 T_0，T_1，...，T_n 中选取最优子树 T_alpha；

Output：最优决策树 T_alpha.
```



## Naive Bayes
> 朴素贝叶斯法是基于 **「贝叶斯定理」** 和 **「特征条件独立假设」** 的分类方法。
> 其基本思想为：对于给定的数据集，首先基于 **特征条件独立假设** 学习 输入/输出的联合概率分布；然后基于此模型，对于给定的输入 $x$ ，利用贝叶斯定理求出后验概率最大的输出 $y$。

对于给定的训练数据集

$$
\begin{align}
T = {(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)}
\end{align}
$$

根据数据集计算其先验概率分布

$$
\begin{align}
P(Y = c_k),\ k = 1, 2, \cdots, K \tag{1}
\end{align}
$$

以及条件概率分布

$$
\begin{align}
P(X = x \mid Y = c_k) = P(X^{(1)} = x^{(1)}, \cdots, X^{(n)} = x^{(n)}),\ k = 1, 2, \cdots, K \tag{2}
\end{align}
$$

条件概率分布 $P(X = x \mid Y = c_k)$ 有指数级的参数量，其估计是不可行的。
事实上，假设输入 $x$ 的每一个维度有 $S_j$ 个取值，而类别 $Y$ 有 $K$ 个取值，那么参数个数就达到了 $K \prod_{j=1}^{n} S_j$。

**「特征条件独立假设」**   
> 朴素贝叶斯法对条件概率分布做了条件独立性的假设，即在类确定的条件下，特征两两之间相互独立。

于是，式（2）基于此假设有：

$$
\begin{align}
P(X = x \mid Y = c_k) &= P(X^{(1)} = x^{(1)}, \cdots, X^{(n)} = x^{(n)}) \\
&= \prod_{j=1}^{n} P(X^{(j)} = x^{(j)} \mid Y = c_k) \tag{3}
\end{align}
$$

**朴素贝叶斯法实际上学习到生成数据的机制，所以属于「生成模型」**。
条件独立假设是说用于分类的特征在类确定的条件下都是条件独立的。

**「贝叶斯定理」**   
> 贝叶斯定理是关于随机变量 $A$ 和 $B$ 的条件概率的一则定理  
> 贝叶斯定理：$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$   
>> 其中，$P(B)$ 不为 $0$.

朴素贝叶斯法分类时，对给定的输入 $x$，通过学习到的模型（先验概率和条件概率）计算后验概率分布 $P(Y = c_k \mid X = x)$，将后验概率最大的类作为 $x$ 的类输出。 
由贝叶斯定理得后验概率的计算：

$$
\begin{align}
P(Y = c_k \mid X = x) &= \frac{P(X = x \mid Y = c_k) P(Y = c_k)}{\sum_{k} P(X = x \mid Y = c_k) P(Y = c_k)} \\
&= \frac{P(Y = c_k) \prod_{j} P(X^{(j)} = x^{(j)} \mid Y = c_k)}{\sum_{k} P(Y = c_k) \prod_{j} P(X^{(j)} = x^{(j)} \mid Y = c_k)} \tag{4}
\end{align}
$$

**「朴素贝叶的分类器」**可以表示为

$$
\begin{align}
y = f(x) = argmax_{c_k}\ \frac{P(Y = c_k) \prod_{j} P(X^{(j)} = x^{(j)} \mid Y = c_k)}{\sum_{k} P(Y = c_k) \prod_{j} P(X^{(j)} = x^{(j)} \mid Y = c_k)} \tag{5}
\end{align}
$$

注意：式（5）的分母对于所有的 $c_k$ 都是相同的，所以有

$$
\begin{align}
y = f(x) = argmax_{c_k}\ P(Y = c_k) \prod_{j} P(X^{(j)} = x^{(j)} \mid Y = c_k) \tag{6}
\end{align}
$$

> 后验概率最大化的含义：朴素贝叶斯将实例分到后验概率最大的类中，这等价于期望风险最小化。

### 朴素贝叶斯的参数估计
**「极大似然估计」**   
> 在朴素贝叶斯法中，学习意味着估计 $P(Y = c_k)$ 和 $P(X^{(j)} = x^{(j)} \mid Y = c_k)$。

那么，如何估计先验概率和条件概率呢？

可以通过应用极大似然估计法估计概率，具体地：  
**先验概率** $P(Y = c_k)$ 的极大似然估计为

$$
\begin{align}
P(Y = c_k) = \frac{\sum_{i=1}^{N} I(y_i = c_k)}{N},\ k = 1, 2, \cdots, K \tag{7}
\end{align}
$$

其中，$I(y_i = c_k)$ 为指示函数：当括号中的条件成立时，值为 $1$，否则为 $0$.

**条件概率** $P(X^{(j)} = a_{jl} \mid Y = c_k)$ 的极大似然估计为

$$
\begin{align}
P(&X^{(j)} = a_{jl} \mid Y = c_k) = \frac{\sum_{i=1}^{N} I(x_{i}^{(j)} = a_{jl}, y_i = c_k)}{\sum_{i=1}^{N} I(y_i = c_k)}\\
&j = 1, 2, \cdots, n;\ l = 1, 2, \cdots, S_j;\ k = 1, 2, \cdots, K \tag{8}
\end{align}
$$

其中， $x_{i}^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特征；$a_{jl}$ 是第 $j$ 个特征可能取的第 $l$ 个值；

**「贝叶斯估计」**    
> 用极大似然估计可能会出现所要估计的概率值为 $0$ 的情况，这时会影响到后验概率的计算结果，使分类产生偏差。

为了解决这一问题，我们采用贝叶斯估计来代替极大似然估计，具体地，**条件概率的贝叶斯估计** 是：

$$
\begin{align}
P(&X^{(j)} = a_{jl} \mid Y = c_k) = \frac{\sum_{i=1}^{N} I(x_{i}^{(j)} = a_{jl}, y_i = c_k) + \lambda}{\sum_{i=1}^{N} I(y_i = c_k) + S_j \lambda}\\
&j = 1, 2, \cdots, n;\ l = 1, 2, \cdots, S_j;\ k = 1, 2, \cdots, K \tag{9}
\end{align}
$$

其中， $\lambda \geq 0$，这等价于在随机变量各个取值的频数上赋予一个正数 $\lambda > 0$，当 $\lambda = 0$ 时，就是极大似然估计。
常取 $\lambda = 1$，这时称为拉普拉斯平滑。

**先验概率的贝叶斯估计** 是：

$$
\begin{align}
P(Y = c_k) = \frac{\sum_{i=1}^{N} I(y_i = c_k) + \lambda}{N + K \lambda},\ k = 1, 2, \cdots, K \tag{10}
\end{align}
$$

### 朴素贝叶斯分类算法

```
Naive Bayes Algorithm

Input: 训练数据集 T = {(x1, y1), (x2, y2), ..., (xN, yN)}

Start
01. 根据极大似然估计（或贝叶斯估计）计算先验概率及条件概率
    具体参见式（7）-（8）（或（9）-（10））
02. 对于给定的实例 x = (x^(1), x^(2), ..., x^(n))^T，计算后验概率
    具体参见式（4）
03. 确定实例 x 的类
    具体参见式（6）

Output: 实例 x 的分类
```


## Logistic Regression


## Boosting  
> 参考自 [CSDN](https://blog.csdn.net/weixin_42933718/java/article/details/88421574)   
> 参考自[知乎](https://zhuanlan.zhihu.com/p/57689719)

Boosting是一种模型的组合方式，我们熟悉的AdaBoost就是一种Boosting的组合方式。
和随机森林并行训练不同的决策树最后组合所有树的bagging方式不同，Boosting是一种递进的组合方式，每一个新的分类器都在前一个分类器的预测结果上改进，所以说boosting是减少bias而bagging是减少variance的模型组合方式。

### GDBT
模型表示为

$$
\begin{align}
F(x) = \sum_{m=1}^{M} \gamma_m h_m(x) \tag{1}
\end{align}
$$

GDBT是一个加性模型，是通过不断迭代拟合样本真实值与当前分类器的残差 $y - \hat{y}_{h_{m-1}}$ 来逼近真实值的。
因此，第 m 个分类器的预测结果为：

$$
\begin{align}
F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) \tag{2}
\end{align}
$$

而 $h_m(x)$ 的优化目标就是最小化当前预测结果 $F_{m-1}(x_i) + h(x_i)$ 和 $y_i$ 之间的差距：

$$
\begin{align}
h_m = argmin_h \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + h(x_i))
\end{align}
$$

### AdaBoost
> 参考自[知乎](https://zhuanlan.zhihu.com/p/57689719)

### XGBoost

## Bagging