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
1 = \int_{-\infty}^{\infty}{p(x)dx} \tag{2}
\end{align}
$$

而如果我们对随机变量 $X$ 进行标准化，使用 $Z = \frac{x - \mu}{\sigma}$ 对 (1) 进行换元，继而有
$$
\begin{align}
\because\ x(z) = z \dot \sigma + \mu
\end{align}
$$

$$
\begin{align}
\therefore\ p(x(z)) = \frac{}
\end{align}
$$