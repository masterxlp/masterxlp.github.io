---
layout: post
title:  "[A] GVF Translation"
date:   2020-05-26 17:56:00 +0800
categories: RL MultiTask
tags: Horde GVF A
author: Xlp
---
* content
{:toc}

## 简介
> Title: General Value Function  
> 2011 - International Foundation for Autonomous Agents and Multiagent Systems  
> Author: Richard S. Sutton, Joseph Modayil, etc.    
> Link: [GVF](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/horde1.pdf)





## ABSTRACT
对于机器人和其他人工智能系统来说，在复杂多变的环境中保持精确的 "world knowledge" 是一个一直以来就存在的问题。
我们的被称为 `Horde` 的结构就是用来处理这个问题的，它是由大量的相互独立的子强化学习智能体（或者称为 `demons`）组成的。
每一个 “demon” 负责回答一个预测(predictive)或以目标为导向(goal-oriented)的“world”的问题，从而以一种分解的、模块化的方式对系统的整体知识做出贡献。
这里的问题是以值函数(value function)的形式体现出来的，但是每个 "demon" 有它们自己的策略、奖励函数、终止函数以及终止奖励函数，这些函数与那些基本问题是无关的。
所有的 “demon” 同时并行的进行学习，以便从系统作为一个整体所采取的任何“actions”中提取最大化的训练信息。
基于梯度的时序差分学习方法在这个离策略设置下被用于去学习一个高效的、可靠的函数逼近器。
Horde在固定的时间和内存中运行，因此，它适合在像机器人等实时应用程序中进行在线学习。
我们在一个多传感器的移动机器人上展示了使用“Horde”成功从离策略经验中学习以目标为导向的行为和远期预测的结果。
Horde是迈向实时架构的重要一步，可以有效地从无监督的 sensorimotor interaction 中学习一般化的知识。

## THE PROBLEM OF EXPRESSIVE AND LEARNABLE KNOWLEDGE
如何学习、表示以及使用一般意义上的world knowledge，仍是人工智能(AI)中的一个关键的开放性的问题。
有一些基于一阶谓词逻辑和贝叶斯网络的高级表示语言具有非常强大的表达能力，但在这些语言中，学习知识是很困难的而且使用计算成本也很昂贵。
还有一些低级语言，像微分方程和状态转化矩阵，可以在无监督的情况下从数据中学习出来，但是这些语言的表达能力要差的多。
而且即使是稍微有点超前的知识，像“If I keep moving, I will bump into something within a few seconds”，也不能用微分方程直接表达出来，且从微分方程中计算的代价也会很昂贵。
我们对于其他可以从无监督的sensorimotor数据中学习的具有强表达能力的形式的知识还有很大的探索空间。

本文从价值函数的概念出发，结合强化学习的思想和算法，提出了一种新的知识表示方法。
在我们的方法中，知识被表示成大量并行学习的值函数的近似，每个值函数都有它们自己的策略、伪奖励函数、伪终止函数以及伪终止奖励函数。
使用这种多个近似值函数形式的学习系统以前被探索为带有"options"的TD(temporal-difference)网络。
我们的架构被称为 `Horde` ，它在处理状态和函数近似（不预测状态表示）方面更加直接，并且在"off-policy"学习方面使用了更加有效的算法，这一点与时间差分网络不同。
本文还扩展了先前的工作，在物理机器人上演示了实时学习。

先前在关于在以"sensorimotor data"为基础并能从这些数据中学习的情况下表示一般意义上的知识的工作至少可以追溯到Cunningham (1972) and Becker (1973)。
Drescher(1991)考虑了一个模拟的机器人婴儿学习布尔事件的条件概率表。Ring(1997)探索了序列的层次表示的连续学习。Cohen等人(1997)从模拟的经验中探索了象征性音调的形成。
Kaelbling et al.(2001)和Pasula et al.(2007)探索了随机域中关系规则表示的学习。
所有这些系统都涉及到重要知识的学习，但距离从"sensorimotor data"中学习还很遥远。
以前从"sensorimotor data"中学习的researchers包括：
Pierce和Kuipers(1997)，他们学习了空间模型和控制律，Oates等人(2000)，他们学习了机器人的集群轨迹，Yu和Ballard(2004)，他们学习了单词的含义，Natale(2005)，他们学习了目标导向的物理动作。
所有这些作品都学到了重要的知识，但都是专门针对某一特定类型的知识;他们使用的知识表示不像多个近似值函数的知识表示那样普遍。

## VALUE FUNCTIONS AS SEMANTICS
近似值函数作为一个知识表示语言的一个独特的、吸引人的特点是，它们在"sensorimotor"交互中有明确的语义，有清晰的真理概念。
只要近似值函数的值与它所近似的数学定义上的值函数的值相匹配，那么我们就认为表示为近似值函数的一些知识是正确的，或者更精确的说，是准确的。
一个值函数提问一个问题 -- 未来的累积回报是多少？ -- 近似值函数提供关于这个问题的答案。
近似值函数就是知识，它与定义了知识的准确含义（对未来的实际奖励）的值函数相匹配。
目前工作的思想是，对于基础语义的值函数方法可以被扩展到奖励以外的所有"world knowledge"的理论。
在本节，我们为奖励和传统的价值函数正式定义了这种思想，并且在下节我们会扩展它们到知识和一般的值函数。

在标准的强化学习框架中，AI智能体与它所在的world的交互被分解为一系列离散的时间步，$t = 1, 2, \cdots$，每个时间步或许与毫秒相对应(a fraction of a second)。
在每一个时间步中的world的状态被表示为，$S_t \in \mathcal{S}$，被智能体所观测到，或许还会被用来选择一个动作，$A_t \in \mathcal{A}$，以作为回应。
一个时间步后，智能体接收一个实值奖励，$R_{t+1} \in \mathbb{R}$，以及下一个状态，$S_{t+1} \in \mathcal{S}$，如此循环。
在不损失一般性的前提下，我们考虑用一个确定性的 **奖励函数** $r$ ，$\mathcal{S} \rightarrow \mathbb{R}$，来生成奖励，有：$R_t = r(S_t)$。

传统强化学习关注的是学习一个随机行为选择 **策略** $\pi$：$\mathcal{S} \times \mathcal{A} \rightarrow [0,1]$，它给出了在每一个状态下选择每一个动作的概率，$\pi(s,a) = \mathbb{P}(A_t=a|S_t=s)$。
非正式地，一个好的策略是：随着时间的推移，智能体接收到大量累积奖励。
例如，在游戏中，奖励可能对应每回合输赢的点数；在比赛中，每个时间步的奖励或许是 $-1$。
在情景性的问题中，智能体与world的交互是由多个有限轨迹（情景）组成，这些轨迹可以在better or worse的方式中结束。
例如，在游戏中，或许会生成一个移动序列，然后以赢、输或者平局结束，每种结果都会对应不同的数值，这些数值可能是 $+1, -1 或者 0$。
一场比赛可能会圆满结束，也可能会被罚下场，这是两种完全不同的结果，即使使用的时间是相同的。
另一个例子是最优控制：在最优控制中，每一步的成本（例如，与能量消耗有关的成本）加上最终的成本（例如，与最终状态到目标状态的距离有关的成本）是常见的。
一般来说，一个问题可能既有确定的奖励函数又有终止奖励函数，$z:\ \mathcal{S} \rightarrow \mathbb{R}$，其中 $z(s)$ 是如果到达状态$s$发生终止时，接收到的终止奖励。

现在，我们把终止的过程正式化。
在很多强化学习问题中，特别是非情景问题中，通常会给延迟奖励(delayed rewards)一个小的权重，一般地，对于每一步的延迟通过系数 $\gamma \in [0,1)$ 来折扣它。
关于折扣的一种理解是，把它看作一个终止的固定概率，$1 - \gamma$，以及一个永远为0的终止奖励。
更一般的，我们可以把它们看作是一个任意的奖励函数，$\gamma:\ \mathcal{S} \rightarrow [0,1]$，$1 - \gamma(s)$ 表示到达状态 $s$ 时终止的概率，这时，将会记录相对应的终止奖励 $z(s)$。
总回报(`return`)是一个随机变量，被表示为从时间 $t$ 开始的轨迹 $G_t$，它是直到 $T$ 时刻终止发生时这一段时间内的每个时间步所接受的奖励的和，再加上再 $S_T$ 状态下接收到的最终的终止奖励:

$$
\begin{align}
G_t = \sum_{k =t+1}^{T} r(S_k) + z(S_T) \tag{1}
\end{align}
$$

传统的动作值函数 $Q^{\pi}:\ \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ 被定为轨迹的回报的期望，轨迹是从给定的状态和动作开始，依据策略 $\pi$ 选择动作直至根据 $\gamma$ 到达终止（从而确定了终止的时间$T$）:

$$
\begin{align}
Q^{\pi}(s,a) = \mathbb{E}[G_t|S_t = s, A_t = a, A_{t+1:T-1} \sim \pi, T \sim \gamma]
\end{align}
$$

这种“期望”在给定一个特定的world的状态转换结构（例如，马尔可夫决策过程）时得到了很好的定义。
如果一个AI智能体拥有一个近似值函数，$\hat{Q}:\ \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$，那么它可以根据 $\hat{Q}$ 与 $Q^\pi$ 的接近程度来评估该近似值函数的准确性，
例如，在一些状态-动作对的分布上计算它的平方误差的期望，$(Q^{\pi}(s,a) - \hat{Q}(s,a))^2$。
在实践中，准确度量这种误差是不可能的，但是值函数 $Q^\pi$ 仍然为知识 $\hat{Q}$ 提供了一个有用的理论语义和基础真理。
这个值函数($Q^\pi(s,a)$)是对“在策略 $\pi$ 下每一个状态-动作对的回报是多少？”问题的准确数值回答，而近似值函数($\hat(Q)$)提供了一个近似值回答。
在这个精确的意义上，这个值函数为AT智能体的近似值函数所表示的知识提供了一个语义。

最后，我们注意到，评估策略的值函数通常只是为了改进策略。
给定一个策略 $\pi$ 以及它的值函数 $Q^\pi$，我们可以构造一个新的确定性的贪婪策略 $\pi' = greedy(Q^\pi)$，因此有，$\pi'(s, \mathop{argmax}\limits_{a} Q^\pi(s,a)) = 1$，且新策略在某种意义上被确保会有所提升，
即 $\forall\ s \in \mathcal{S}, a \in \mathcal{A},\ Q^{\pi'}(s,a) \geq Q^\pi(s,a)$，当且仅当两个策略都是最优时取等号。
通过逐步的估计和提升，使得期望回报最优的策略可以被找到。
这样，值函数理论可以为以目标为导向的知识（控制）和预测知识提供一个语义。

## FROM VALUES TO KNOWLEDGE (GENERAL VALUE FUNCTIONS)
在明确了传统的价值函数如何为即将到来的奖励的知识提供grounded（落地的）语义之后，在本节，我们将展示 `通用价值函数` 如何为更一般的世界知识提供grounded语义。

首先注意，尽管动作值函数 $Q^\pi$ 只有策略这一个传统意义上的上标，但是它同样依赖于奖励 $r$ 和终止奖励函数 $z$。
这些函数同样可以被认为是和策略 $\pi$ 一样以同样的形式被输入到值函数中。
也就是说，我们或许可以定义一个更一般的值函数，$Q^{\pi,r,z}$，它将使用式(1)中定义的任意伪奖励函数 $r$ 和伪终止奖励函数 $z$代表的回报。
例如，假设我们正在玩一个游戏，这个游戏的基本终止奖励为：赢了 $z = +1$，输了 $z = -1$（每一个时间步的奖励为 $r = 0$）。
除此之外，我们或许还会提出另一个独立的问题：游戏还会走多少步？
这可以表示为一个具有伪奖励函数 $r = 1$、伪终止奖励函数 $z = 0$ 的通用值函数。

从值函数到通用值函数的第二步是转换终止函数 $\gamma$ 为一个伪形式。
这稍微更实质一些，因为与奖励和终止奖励不同，它们在任何形式中都不涉及到状态演变，但是终止通常是指正常状态转换流中的中断，以及充值到初始状态或者初始状态分布。
对于伪终止，我们简单地忽略了这个常规终止的附加含义。
这个真正的、基本的问题仍然有真正的终点，也可能根本没有终点。
然而，我们可以认为 `pseudo terminations` 在任何时候都可能发生。
例如，在比赛中，我们可以考虑一个在一半赛程终止的伪终止函数。
在一般意义上，这是关于值函数问题的一个很好的定义。
或者说，如果我们是赛手的配偶，那么我们或许不关心比赛什么时候结束，而是关心赛手什么时候回家吃晚饭，这可能是我们的伪终止。
对于同一个world（相同的动作和状态转移），有很多预测问题可以以通用值函数的形式定义。

正式地，我们定义 **通用值函数，GVF** 为一个函数 $q\ : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$，它有四个 auxiliary functional 的输入 $\pi, \gamma, r, z$，
这与之前的定义有相同的范围，但是现在它们表示任意的奖励、终止奖励以及终止函数，与基本问题的奖励、终止奖励以及终止函数没有什么必要的关系：

$$
\begin{align}
q(s, a; \pi, \gamma, r, z) = \mathbb{E}[G_t|S_t = s, A_t = a, A_{t+1:T-1} \sim \pi, T \sim \gamma]
\end{align}
$$

其中，$G_t$ 仍然是由式(1)定义的那样，但是现在是对给定函数的。
这四个函数，$\pi, \gamma, r, z$，被统称为GVF的 *question functions*；它们定义了GVF的问题或者语义。
注意到传统的值函数为GVF的一个特例。
因此，我们可以任务所有的值函数都是GVF的一种情况。
为了简单期间，在本文的其余部分，我们有时使用“value function”表示通用情况，当需要消除歧义时使用“conventional value function”。
我们也会去掉问题函数的前缀"pseudo-"，当不存在歧义时。
在我们稍后提出的机器人实验中，不存在特别的基本问题，所以不存在混淆。

## THE HORDE ARCHITECTURE
**Horde** 结构是有许多被称为 **demons** 的“子智能体”组成的完整智能体构成。
每一个demon是一个独立的强化学习智能体，负责学习基本智能体羽环境交互中的一小部分知识。
每一个demon学习一个关于GVF $q$ 的近似 $\hat{q}$，对应于四个问题函数 $\pi, \gamma, r, z$ 关于这个demon的设置。

现在我们来描述Horde机制，它可以用有限数量的 **weights** 来近似GVF，并学习这些 weights。
本文采用标准的线性方法来进行函数逼近。
我们假设，每一个时间步的状态和动作，$S_t, A_t$，都被翻译(可能不完全通过sensory解读)为一个固定大小的 **feature vector** $\phi_t = \phi(S_t,A_t) \in \mathbb{R}^n$，
其中 $n \ll |\mathcal{S}|$。
对于所有的状态-动作对，我们表示所有的特征集为 $\Phi$。
在我们的实验中，特征向量时通过tile（平铺）编码来构造的，因此为二进制表示，$\phi_t \in {0,1}^n$，具有一个常数1的特征。
我们也关注了 $|\mathcal{S}|$ 很大的情况，可能是无限的，但是 $|\mathcal{A}|$ 是有限的，相比来说很小，这在强化学习问题中很常见。
这都是一些实用的特殊情况，但是对于我们的方法来说，没有一种是必要的。
我们近似的GVF，表示为 $\hat{q} \ :\ \mathcal{S} \times \mathcal{A} \times \mathbb{R}^n \rightarrow \mathbb{R}$，在特征向量中是线性的：

$$
\begin{align}
\hat{q}(s,a,\theta) = \theta^T\phi(s,a)
\end{align}
$$

其中，$\theta \in \mathbb{R}^n$ 是可以被学习的 **weights** 向量，$v^T w = \sum_i v_i w_i$ 表示两个向量 $v 和 w$ 的内积。

为了学习weights，我们使用最近开发的梯度下降时序差分算法。
这个算法的独特之处在于，它可以从 *off-policy* 的经验中稳定有效地学习函数逼近。
off-policy的经验指的是由被称为 **behavior policy** 的策略产生的经验，它与被称为 **target policy** 的用于学习的策略不同。
似乎天生就面临着要从无监督的交互中有效地学习知识，因为想要并行的学习很多策略（每一个GVF的不同目标策略$\pi$），但是，当然，每次只能按照一个策略执行。

对于一个典型的GVF，行为策略所采取的动作只有在偶然情况下才会匹配目标策略，并且很少会连续执行多个步骤。
为了有效地学习，我们需要能够从这些相关经验的片段中进行学习，这就需要off-policy的学习。
另一种选择（on-policy的学习）要求仅仅从那些完全匹配GVF目标策略的经验片段中学习，这是一种不常见的情况。
如果可以从不完整的经验片段中进行off-policy的学习，那么它可以进行大量的并行学习，并且比on-policy的学习要快得多。

直到最近几年，才出现了能够可靠地使用函数逼近和适用于实时学习和预测的off-policy学习算法。
特别地，在这篇文章中我们使用 $GQ(\lambda)$ 算法。
对于每一个GVF，该算法维护除了 $\theta$ 和资格迹向量 $e \in \mathbb{R}^n$之外的第二组weights $w \in \mathbb{R}^n$。
这三个向量被初始化为0。
然后，每一步 $GQ(\lambda)$ 计算两个中间量 $\bar{\phi}_t \in \mathbb{R}^n$ 和 $\delta_t \in \mathbb{R}$：

$$
\begin{align}
\bar{\phi}_t &= \sum_a \pi(S_{t+1}, a)\phi(S_{t+1}, a)\\
\delta_t &= r(S_{t+1}) + (1 - \gamma(S_{t+1}))z(S_{t+1}) + \gamma(S_{t+1})\theta^T\bar{\phi}_t - \theta^T\phi(S_t,A_t)
\end{align}
$$

然后更新这三个向量：

$$
\begin{align}
\theta_{t+1} &= \theta_t + \alpha_\theta(\delta_t e_t - \gamma(S_{t+1})(1 - \lambda(S_{t+1}))(w_t^T e_t)\bar{\phi}_t)\\
w_{t+1} &= w_t + \alpha_w (\delta_t e_t - (w_t^T \phi(S_t,A_t))\phi(S_t,A_t))\\
e_t &= \phi(S_t,A_t) + \gamma(S_t)\lambda(S_t)\frac{\pi(S_t,A_t)}{b(S_t,A_t)}e_{t-1}
\end{align}
$$

其中，$b\ :\ \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$ 是行为策略，$\lambda\ :\ S \rightarrow [0,1]$ 是一个资格迹函数，它决定了在 $TD(\lambda)$ 算法中的资格迹的衰变率。
注意，这个算法的每一步计算随特征的个数 $n$ 线性扩展。
然而，如果特征是二值的，那么只要稍加注意，每一步的复杂性就可以被控制在特征"1"的数量的倍数上。

通过 $GQ(\lambda)$ 算法渐进近似可以被发现时依赖于特征向量 $\Phi$、行为策略 $b$、以及资格迹函数 $\lambda$。
这三个函数统称为应答函数。
在这篇文章的实验中，我们通常使用固定的$\lambda$，并且所有的demons共享$\Phi$ 和 $b$。
最后，我们注意，Maei and Sutton 定义了一个终止函数 $\beta$，它和我们的 $\gamma$ 具有相反的意义；也就是说，$\beta(s) = 1 - \gamma(s)$。
这只是存粹的符号差异，并不会以任何方式影响算法。

我们可以认为有两种demons。
一种是带有给定目标策略 $\pi$ 的demon，被称为 `prediction demon`，而另一种demon的目标策略是与我们的近似GVF对应的贪婪策略
（即 $\pi = greedy(\hat{q})$ 或 $\pi(s, \mathop{argmax}\limits_{a} \hat{q}(s,a,\theta)) = 1$)，被称为 `contorl demon`。
控制demons可以学习和表示如何到达目标，而在预测demons中知识是更好的描述事实的思想。
demons不完全独立的一种方式是，预测demons可以参考控制demons的目标策略。
例如，在这种方式中，人们可以问这样的问题“If I follow this wall as long as I can, will my light sensor then have a high reading?”。
Demons也可以使用其他在这个问题上的答案（就像在时序差分网络中那样）。
这允许一个demon学习一个像“接近一个障碍”这样的概念，意味着在几秒钟内读取一个高碰撞传感器的随机动作的概率，然后另一个demon基于这种概念来学习一些东西，
就像通过在它的终止奖励函数(即 $z(s) = \mathop{max}\limits_{a} \hat{q}(s,a,\theta_{first\ demon})$)中使用第一个demon的近似GVF，“如果我沿着这面墙走到头，那么我将会靠近一个障碍吗？”。

## RESULTS WITH HORDE ON THE CRITIERBOT



















































