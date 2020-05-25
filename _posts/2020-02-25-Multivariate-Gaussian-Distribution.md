---
layout: post
title:  "Multivariate Gaussian Distribution"
date:   2020-05-25 12:17:00 +0800
categories: Guassian Math
---

转载自[知乎](https://zhuanlan.zhihu.com/p/58987388 "多元高斯分布详解")

## Multivariate Gaussian Distribution
> 由中心极限定理我们知道，大量独立同分布的随机变量的均值在做适当标准化后会依分布收敛于高斯分布，这使得高斯分布具有普适性的建模能力.
> 数学上，当使用高斯分布对贝叶斯推断的似然和先验进行建模时，得到的后验同样为高斯分布，即其具有共轭先验性质.

### 多元标准高斯分布
#### 一元高斯分布
若随机变量 $X \sim \mathcal{N}(\mu, \sigma^2)$ ，则有如下的概率密度函数
$$
\begin{align}
p(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} \tag{1}
\end{align}
$$

$$
\begin{align}
1 = \int_{-\infty}^{\infty}{p(x)dx} \tag{2}
\end{align}
$$

而如果我们对随机变量 $X$ 进行标准化，使用 $Z = \frac{x - \mu}{\sigma}$ 对 (1) 进行换元，继而有
$$
\begin{align}
\because\ x(z) = z \cdot \sigma + \mu
\end{align}
$$

$$
\begin{align}
\therefore\ p(x(z)) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}z^2}
\end{align}
$$

$$
\begin{align}
\therefore\ 1\ = \int_{-\infty}^{\infty}{p(x(z))dx}
\end{align}
$$

$$
\begin{align}
\qquad \qquad  = \int_{-\infty}^{\infty}{\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}z^2} dx}
\end{align}
$$

$$
\begin{align}
\qquad \qquad  = \int_{-\infty}^{\infty}{\frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}z^2} dz} \tag{3}
\end{align}
$$

此时我们说随机变量 $Z \sim \mathcal{N}(0,1)$ 服从一元标准高斯分布，其均值 $\mu = 0$，方差 $\sigma^2 = 1$，其概率密度函数为

$$
\begin{align}
p(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}z^2} \tag{4}
\end{align}
$$

**随机变量** $X$ **标准化的过程，实际上是消除量纲影响和分布差异的过程。
通过将随机变量的值减去其均值再除以标准差，使得随机变量与其均值的差距可以用若干个标准差来衡量，从而实现了不同随机变量与其对应均值的差距
可以以一种相对的距离来比较。**

#### 多元标准高斯分布
那么一元标准高斯分布与多元标准高斯分布有什么关系呢？事实上，多元标准高斯分布的概率密度函数正式从式(4)中导出的。
假设我们有随机向量 $\overrightarrow{Z} = [Z_1, \cdots, Z_n]^T$，其中 $Z_i \sim \mathcal{N}(0,1)\ (i=1, \cdots, n)$
且 $Z_i, Z_j\ (i,j=1, \cdots, n,\ i \neq j)$ 彼此独立，即随机向量中的每个随机变量 $Z_i$ 都服从标准高斯分布且两两彼此独立，
则由(4)与独立变量概率密度函数之间的关系，我们可以得到随机向量 $\overrightarrow{Z} = [Z_1, \cdots, Z_n]^T$ 的联合概率密度函数为

$$
\begin{align}
p(z_1, \cdots, z_n) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}(z_i)^2}
\end{align}
$$

$$
\begin{align}
\quad = \frac{1}{(2\pi)^{\frac{n}{2}}} e^{-\frac{1}{2}(Z^{T}Z)}
\end{align}
$$

$$
\begin{align}
1 = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} p(z_1, \cdots, z_n) dz_1 \cdots dz_n \tag{5}
\end{align}
$$

我们称随机向量 $\overrightarrow{Z} \sim \mathcal{N}(\overrightarrow{0}, \boldsymbol{I})$，即随机向量服从均值为零向量，协方差矩阵为单位矩阵的高斯分布。
在这里，















