---
layout: post
title:  "[A] Unicorn Translation"
date:   2020-05-29 14:28:00 +0800
categories: RL MultiTask
tags: UVFA variant A
author: Xlp
---
* content
{:toc}

## 简介 
> Title: Unicorn: Continual learning with a universal, off-policy agent  
> 2018 - Computer Science > Machine Learning - arXiv  
> Author: DeepMind   
> Link: [Unicorn](https://arxiv.org/abs/1802.08294)





## Abstract
一些real-word领域最好的描述是作为一个单任务来描述，但是对于其他领域来说，这种观点是有局限性的。
相反，随着智能体的competence的提高，一些任务的复杂性也不断增加。
在持续学习中（也称为终身学习），没有明确的任务边界或curricula。
随着学习中的智能体变的越来越强大，持续学习仍然是阻碍快速进步的前沿领域之一。
为了测试持续学习的能力，我们考虑一个具有挑战性的3D领域，该领域具有隐式的任务序列和稀疏的奖励。
我们提出了一种新颖的智能体结构，“独角兽(Unicorn)”，它展示了强大的持续学习能力，并在上面提出的领域中超过了几种基线智能体的表现。
智能体通过联合表示以及使用一个并行的off-policy的设定高效的学习多个策略来实现这一点。

## Introduction
持续学习，是一种利用之前获得的知识或技巧的方式从有关连续任务的经验中学习的方法，它一直是人工智能领域的一个长期挑战。
该方法的一个主要优点就是，它为一个完全自主的智能体提高了增量地构构建其能力以及在不需要人为的提供数据集、任务边界或reward shaping的情况下解决在丰富(rich)、复杂的环境中表示它的挑战的潜力。
相反，随着智能体能力的提升，它会考虑增加任务的复杂性。
一个理想的持续学习智能体应该能够：(A) 解决多种任务；(B) 当任务相关时表现出协同效应； (C) 处理任务之间的深度依赖结构（例如，一把锁只有当它的钥匙被拿起时才能解锁）。

在监督学习设置下存在着大量连续学习技术。
然而，正如 Parisi et al. 提到的那样，需要更多的研究来解决在不确定的环境中使用自主的智能体进行持续学习的问题 -- 这非常适合强化学习。
以前使用关于强化学习的持续学习工作，特别是关于使用深度依赖结构解决任务的工作，通常focused on将学习分为两个阶段：首先，要求分别获得各自的skills，然后再从新组合，以解决更具有挑战性的任务。
Finn et al. 将此定义为 *meta learning*，并对智能体进行任务分配，这些任务被明确设计来适应该分布中一个新的任务，这几乎不需要额外的学习。
在线性强化学习设定中，当它们遇到时，通过策略梯度学习一个latent basis去解决新任务。
Brunskill and Li 得出了在终身学习中进行option discovery的样本复杂性的界限。

在这项工作中，我们的目标是使用 `single-stage end-to-end learning` 来解决具有深度依赖结构的任务。
此外，我们旨在所有任务上训练智能体而不考虑它们的复杂性。
然后，experience可以在任务之间共享，从而使得智能体能够有效地并行开发每个任务的能力。
为了实现这一点，我们需要将 **task generalization**（任务泛化）和 **off-policy learning** 结合起来。
有许多强化学习技术通过将任务直接合并到值函数的定义中来执行任务泛化。
*Universal Value Function Approximators (UVFAs)* 是这些工作的最新研究成果，它通过共享参数高效地将 a Horde of demons 组合到一个值函数中。
我们将UVFAs（它原来表现为两步学习）与off-policy goal learning相结合，更新到端到端的、最先进的并行智能体结构中，以实现持续学习。
这些components(组件)的新式组合产生了一个持续学习的智能体，我们称为 `Unicorn`，它能够大规模地学习具有深度依赖结构（Figure 1a, top）的non-trivial的任务。
独角兽智能体通过在任务之间共享experience、重用表示以及技能解决这些领域的相关问题，并且表现超过了baselines methods（Figure 1a, bottom）。
我们也证明了独角兽可以轻而易举地：(A) 解决多种任务（没有依赖关系）以及 (B) 当任务相关时表现出协同效应。
![Figure 1](../../../../image/Unicorn-structure.png "Unicorn structure")

## Background
**Reinforcement Learning**(RL) 是一种计算框架，用于在不确定的序贯决策问题中做决策。
强化学习问题被描述为一个Markov decision process (MDP)，定义为五元组 $<\mathcal{S}, \mathcal{A}, r, \mathcal{P}, \gamma>$，其中 $\mathcal{S}$ 表示状态集，$\mathcal{A}$ 表示动作集，
$r\ :\ \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ 表示奖励函数，$\mathcal{P}\ :\ \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$ 表示转移概率分布，
$\gamma \in [0,1)$ 表示折扣系数。
策略 $\pi$ 映射状态 $s \in \mathcal{S}$ 到动作的概率分布。
我们定义给定时间步 $t$ 时的 *return* 为折扣奖励的和：$R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r{t+k+1}$，其中 $r_t = r(s_t, a_t)$。
动作值函数 $Q^\pi(s,a) = \mathbb{E}^\pi [R_t|s_t = s, a_t = a]$ 估计一个智能体的 *return* 的期望值，该智能体在遵循策略 $\pi$ 之后，在一些状态 $s \in \mathcal{S}$ 下选择动作 $a \in \mathcal{A}$。
最优动作值函数 $$Q^{*}(s,a)$$ 估计的是基于最优策略 $$\pi^{*}$$ 时的 *return* 的期望。

**Q-learning** 可以通过一个迭代引导过程来估计这个最优值函数 $Q^{*}(s,a)$，这其中 $Q(s_t,a_t)$ 朝着导向目标 $Z_t$ 更新，
$Z_t$ 是用下一个状态的估计Q值来构造的：$Z_t = r_{t+1} + \gamma \mathop{max}\limits_{a} Q(s_{t+1}, a)$。
$\delta_t = Z_t - Q(s_t, a_t)$ 表示的是时间差分误差(TD误差)。

**Multi-step Q-learning** variants 在单个导引目标中使用多条transitions。
一个常用的选择是 **n-step return**，它定义为：$G_t^{(n)} = \sum_{k=1}^{n} \gamma^{k-1} r_{t+k} + \gamma^n \mathop{max}\limits_{a} Q(s_{t+n, a})$。
在计算 n-step returns 时，目标策略和行为策略之间的n步动作选择可能不一致。
为了使用off-policy的方法，可以使用各种技术来纠正这种不匹配。
我们通过在选择non-greedy的action时截断returns来处理off-policy修正，正如 Watkins 建议的那样。

**Universal Value Function Approximators**(UVFA)扩展值函数为以单个目标 $g \in \mathcal{G}$ 为条件的值函数，这种函数近似（如深度神经网络）共享一个内在的、独立于目标的状态表示 $f(s)$。
因此，UVFA $Q(s,a;g)$ 能够紧凑的表示多种策略；例如，以任意单个目标 $g$ 为条件产生相对应的贪婪策略。
先前，UVFA是通过两步实现的，包括一个矩阵分解步来学习embedding以及一个单独的多变量回归过程。
相比之下，unicorn在一个离策略目标学习和纠正的联合并行训练中端到端的学习 $Q(s,a;g)$。

**Tasks vs. goals** 在本文中我们对任务(task $\tau$)和目标信号(goal signal $g$) 赋予了不同的含义。
*goal signal* 调整智能体的 *hehavior*（例如，作为UVFA的输入）。
相比之下，任务定义为一个伪奖励(pseudo-reward) $r_\tau$（例如，$r_{key} = 1$ if a key was collected and 0 otherwise）。
在学习过程中，每条transition上的包含所有伪奖励的向量对应智能体来说都是可见的，即使它正在pursue一个specific goal。
每组实验定 $K$ 个离散的任务 ${\tau_1, \tau_2, \cdots, \tau_K}$。
在迁移实验中，任务被分为$K'$个训练任务和 $K - K'$ 个hold-out任务。

## Unicorn
这一节将介绍Unicorn智能体结构，为了促进持续学习它具有以下性能：
*(A):* 智能体应该具有同时学习多个任务的能力，以及在不断遇到的新任务的领域学习的能力。
我们使用联合并行训练设置，让不同的actor完成不同的任务以达到这个性能。
*(B):* 随着智能体累积越来越多的知识，我们希望它通过重用(reuse)一些知识来解决相关的任务，以实现一般化(泛化)。
这是通过使用一个单独的UVFA来获得关于所有任务的知识来实现的，通过分离相关目标和不相关目标的表示来促进泛化。
*(C):* 在任务具有深度依赖结构的领域中，智能体应当以然是有效的。
这是最具有挑战性的，但是可以通过在所有任务的experience中off-policy learning的方式来实现。
例如，一个actor是以door为目标的，有时候它打开了一个门，但随后错误的打开了一个chest（柜子）。
对于以door为目标的actor来说，这是一个无关紧要的event（因为没有奖励），但是当学习关于chest的任务时，这个相同的event是highly interesting的，因为它是一个罕见的non-zero奖励的transition。

### Value function architecture
Unicorn agent的一个关键部分就是UVFA，它是一个是用来学习逼近 $Q(s,a;g)$ 的逼近器，例如神经网络。
这个逼近器的强大之处在于它能以信号目标 $g$ 为条件。
这就使得UVFA可以同时学习多个任务，而任务本身的难度可能也不尽相同（例如，具有深度依赖关系的任务）。
UVFA的示意图如图2所示：当前的可视输入帧由卷积神经网络(CNN)来处理，然后是一个循环的长短期记忆(LSTM)层。
在引用[11]中，先前的action和reward是observe的一部分。
LSTM的输出与一个“inventory stack”连接在一起，形成与目标无关的状态表示 $f(s)$。
然后该向量与一个目标 $g$ 相连接，通过一个以ReLU为非线性激活函数的两层MLP处理输出Q-value的向量（对于每一个可能的action $a \in \mathcal{A}$）。
所有这些部分的可训练参数的组合被表示为 $\theta$。
关于这个网络和超参数的进一步的细节可以在附录A中找到。
![Figure 1](../image/UVFA结构图.png "Unicorn structure")

### Behaviour policy
在每一个episode开始时，以均匀分布采样得到目标信号 $g_i$，并在整个episode中保持不变。
在当前目标信号 $g_i$ 上以UVFA为条件执行 $\epsilon - greedy$ 策略：以 $\epsilon$ 的概率从 $\mathcal{A}$ 中均匀采样动作 $a_t$，否则 $a_t = \mathop{argmax}\limits_{a} Q(s_t, a; g_i)$。

### Off-policy multi-task learning
Unicorn的另一个关键的部分是学习多任务off-policy的能力。
因此，即使它对于特定的任务使用on-policy，它仍然可以并行的从共享的experience中学习其他任务。
具体地，当从一个transitions的序列中学习的时候，在训练集中对于所有的目标信号 $g_i$ 进行 Q-values的估计，
对于每一个相对应的任务 $\tau_i$ 计算n-step returns $G_{t,i}^{(n)} = \sum_{k=1}^{n} \gamma^{k-1} r_{\tau_i}(s_{t+k},a_{t+k}) + \gamma^n \mathop{max}\limits_{a} Q(s_{t+n}, a; g_j)$。
当一个策略以目标信号 $g_i$ （关于这条trajectory的 on-policy goal）为条件产生一条trajectory，但是被用来学习另一个目标信号 $g_j$ （关于这条trajectory的 off-policy goal）的策略时，往往会出现动作不匹配的情况，
因此，off-policy 的多步 bootstrapped 的目标变的越来越不准确。
因此，根据引用[50]，当选择的动作与以目标 $g_j$ 为条件的策略选择的动作不匹配时（即 $a_t \neq \mathop{argmax}\limits_{a} Q(s_t, a; g_i)$），我们通过bootstrapping来阶段n-step return。
通过对任务和unrolled的长度为 $H$ 的trajectory的TD误差的和应用梯度下降方法来更新网络，产生平方损失(式1)，这里的误差没有传播到目标 $G_{t,i}^{(n)}$。

$$
\begin{align}
\mathcal{L} = \frac{1}{2} \sum_{i=1}^{K'} \sum_{t=0}^{H} (G_{t,i}^{(n)} - Q(s_t, a_t; g_i))^2 \tag{1}
\end{align}
$$


### Parallel agent implememtation
为了有效地训练这样一个系统，我们采用一个由多个actor组成的并行的智能体，每一个actor单独运行在一个机器上（CPU），生成一个与环境交互的序列，以及一个单独的learner（GPU机器），
它从一个queue中取出一些experience，处理成一个mini-batch，然后更新value网络。
这与最近提出的Importance Weighted Actor-Learner Architecture agent 类似。

对一些目标信号 $g_i$ 每个actor连续地执行这最新的策略。
然后它们产生相应的experience（这些经验存储在一个global queue中），以长度为 $H$ 的trajectory的形式送给learner。
在产生新的trajectory之前，actor会向learner请求最新的UVFA参数 $\theta$。
注意，所有M个actor是并行运行的，在任意给定的时间，它们都follow不同的goal。

Learner从global queue中取一个batch大小的trajectories，输入到神经网络中，根据式1计算loss，更新参数 $\theta$ ，并且为actor提高最新的参数 $\theta$。
Batching发生在所有的 $k'$ 个训练任务以及 $B \times H$ （$B$ trajectories of leangth $H$）时间步上。
与DQN不同，我们没有使用目标网络，没有使用经验回放：大量不同的experience足够保证稳定性。
















































