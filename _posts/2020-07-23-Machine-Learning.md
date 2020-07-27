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


## Bagging