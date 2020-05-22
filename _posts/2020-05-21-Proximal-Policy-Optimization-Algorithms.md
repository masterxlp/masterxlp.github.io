---
layout: post
title:  "Proximal Policy Optimization Algorithms [paper]"
date:   2020-05-21 14:48:00 +0800
categories: RL PPO paper
---

## Proximal Policy Optimization Algorithms
### Abstract


我们提出了一种用于强化学习的策略梯度类的方法，该方法表现为通过与环境交互来sample datas和借助 *stochastic gradient ascent* 来优化一个“surrogate”目标函数，两者之间交替执行。 
与每采样一次数据就进行一次梯度更新的standard策略梯度方法不同，我们提出了一种新颖的目标函数，可以实现多个epochs的小批量更新。
这个新方法我们把它称为**近端策略优化**（proximal policy optimization, ppo），该方法拥有置信域策略优化（TRPO）的部分优势，但是该方法有具有比TRPO更易实现、更一般化、以及更优的样本复杂性的优点。
我们在一组benchmark的任务中测试了PPO算法，包括仿真机器人运动（simulated robotic locomotion）和Atari游戏，实验结果表明PPO比其他在线策略梯度表现的都要好，总体上在样本的复杂性、简易性以及有效性之间达到了一个有效的平衡。


### Introduction

近年来，一些基于神经网络函数逼近的强化学习方法被提出，其中最具影响力的是 *deep Q-learning*、*vanilla policy gradient* 以及 *trust region policy gradient*。
然而这些方法在scalable（大型模型和并行实现）、数据效率以及健壮性（在不进行超参数调优的情况下）方面还有提升的空间。
Q-learning（基于函数逼近）在很多简单的问题上表现不佳，且可解释性较差；
vanilla policy gradient方法在数据效率和健壮性上表现不足；
而TRPO方法则相对来说比较复杂，且与包含noise（比如dropout）或者参数共享（策略与值函数或者辅助任务之间）的架构不兼容。  

本文试图通过引入一种算法来改善目前的状况，该算法在仅使用一阶优化的情况下拥有高的数据效率和TRPO的可靠表现。
我们提出了一个基于`clipped probability ratios`的新式目标，它表示的是策略性能的“pessimistic”的估计（即下界）。
为了优化策略，我们交替的执行基于策略进行采样和在采样的数据上进行几个epochs的优化。

我们实验比较了多种不同形式的 **surrogate objective** 的表现，发现objective为 **clipped probability ratios** 形式时表现最佳。
我们也将PPO算法和先前的几种文献中的算法进行了比较。
在连续控制任务中，PPO算法比这些算法表现的都要好；在Atari游戏中，PPO算法明显比A2C算法表现的要好（就样本复杂度而言），与ACER表现相似，但是PPO更加简单。

### Background: Policy Optimization
#### Policy Gradient Methods

Policy gradient 方法的工作机制为：计算一个policy gradient的估计值，然后插入到梯度上升算法中。
最常用的梯度估计的形式为：
$$
\begin{align}
\hat{g} = \hat{\mathbb{E}}_t [\nabla_{\theta} log \pi_{\theta}(a_t|s_t) \hat{A}_t] \tag{1}
\end{align}
$$
其中，$\pi_\theta$ 是一个随机策略，$\hat{A}_t$ 是t时刻优势函数的估计值。
这里的期望 $\hat{\mathbb{E}}_t[...]$ 表示的是在一个采样和优化交替进行的算法中，有限样本集的经验平均值。
通过构造一个梯度为策略梯度估计的目标函数以自动微分软件来实现梯度的求解；估计值 $\hat{g}$ 通过微分下面的目标函数得到：
$$
\begin{align}
L^{PG}(\theta) = \hat{\mathbb{E}}_t[log \pi_\theta(a_t|s_t)\hat{A}_t] \tag{2}
\end{align}
$$
然而不能使用相同的trajectory来对 $L^{PG}$ 损失执行多个时间步的优化，因为这常常会导致过大的策略更新（虽然该设置在6.1节中没有展示出来，但是它的结果与“no clipping or penalty”的设置是相似的）。

### Trust Region Methods

TRPO是在约束策略更新大小的情况下最大化目标函数（“surrogate”目标）。
$$
\begin{align}
\mathop{maximize}\limits_{\theta} \quad \hat{\mathbb{E}}_t [\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t] \tag{3}
\end{align}
$$

$$
\begin{align}
subject\ to\ \quad \hat{\mathbb{E}}_t[KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t)]] \leq \delta \tag{4}
\end{align}
$$
其中，$\theta_old$ 表示的是更新之前的策略参数向量。在对目标做线性近似以及对约束做二次约束之后，该优化问题可以通过 *共轭梯度算法* 被有效地近似解决。

该理论证明TRPO实际上建议使用一个惩罚而不是约束，也就是，在一些系数 $\beta$ 优化下面这个无约束的优化问题：
$$
\begin{align}
\mathop{maximize}\limits_{\theta} \quad \hat{\mathbb{E}}_t [\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t - \beta KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t)]] \tag{5}
\end{align}
$$
这遵循这样一个事实：一个确定的surrogate的目标（在状态上计算最大KL散度而不是求均值KL）在策略 $\pi$ 的性能上表现为一个下界（即pessimistic bound）。
然而，TRPO使用的是一个“hard”约束而不是惩罚，这是因为很难选择一个在不同问题上都表现很好的单个 $\beta$ 值，甚至是在这样的单一问题上：在学习上造成的特征改变，都无法找到一个有效的固定值。
因此，为了实现我们的目标，我们设计了一种模型TRPO单调提升的一阶算法，实验表明选择一个固定的惩罚系数 $\beta$ 以及使用SGD算法优化带惩罚的目标（方程5）是困难的。

### Clipped Surrogate Objective

特别地，让 $r_t(\theta)$ 表示 **probability ratio** $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，则有 $r(\theta_{old}) = 1$。TRPO最大化 *surrogate objective* 表示为：
$$
\begin{align}
L^{CPI}(\theta) = \hat{\mathbb{E}}_t [\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t] = \hat{\mathbb{E}}_t [r_t(\theta) \hat{A}_t] \tag{6}
\end{align}
$$
上标 *CPI* 指的是 **conservation policy iteration**（保守策略迭代）。
在最大化 $L^{CPI}$ 目标函数时，如果不加以约束将会导致过大的策略更新；因此，我们现在考虑如何修改这个目标函数，以惩罚策略的改变，使得 $r_t(\theta)$ 远离1。

我们提出的目标函数为：
$$
\begin{align}
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [min(r_t(\theta) \hat{A}_t, clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t)] \tag{7}
\end{align}
$$
其中，*epsilon* 是一个超参数（$\epsilon = 0.2$）。
目标函数 $L^{CLIP}(\theta)$（方程7）的解释为：*min* 内部的第一项其实就是 $L^{CPI}$；第二项（$clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t$）通过截断 *probability ratio* 修改这个 *surrogate* 目标，这意味着移除区间 $[1 - \epsilon, 1 + \epsilon]$ 之外的 $r_t$ 的值对目标函数的影响。
最后，我们对 “clipped” 和 “unclipped”的目标取最小值，所以这最后的目标是 “unclipped” 目标的一个下界（即 pessimistic bound）。
因此，在这样的理论下，我们仅仅当 $r_t$ 可以提升目标时才会忽略 $r_t$ 的改变；当 $r_t$ 使得目标函数变差时，加入 $r_t$ 的改变。
（**译者注：即只有当新旧策略的比值能够提升目标函数时，才会不进行clip的操作，否则进行clip的操作以使得目标函数不会变差，达到单调提升的效果**）。
值得注意的是，当 $\theta = \theta_{old}$ 时（即 $r = 1$），$L^{CLIP}(\theta) = L^{CPI}(\theta)$。
然而，它们随着 $\theta$ 逐渐远离 $\theta_{old}$ 而变得不同。
图1画出了在时间t时 $L^{CLIP}$ 的变化；注意到当优势值为正或负时， *probability ratio* 在点 $1 - \epsilon$ 或点 $1 + \epsilon$ 处被相应的截断。
![Figure 1](../image/ppo截断图.png "clipped objective value")

图2提供了另一种关于 *surrogate objective* $L^{CLIP}$ 的直观理解。
它展示了当我们在一个连续控制问题中沿着由PPO算法获得的策略更新方向插值（interpolate）时，几种目标函数是如何变化的。
我们可以看到对过大的策略更新施加惩罚的目标函数 $L^{CLIP}$ 是 $L^{CPI}$ 的一个 *lower bound*。
![Figure 2](../image/ppo-几种surrogate目标的比较.png "several surrogate objective compare")

### Adaptive KL Penalty Coefficient








































