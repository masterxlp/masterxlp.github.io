---
layout: post
title:  "[A] A Multi-Objective Deep Reinforcement Learning Framework"
date:   2020-06-04 11:24:00 +0800
categories: Multi-Objective DRL
---

## [A] A Multi-Objective Deep Reinforcement Learning Framework
> 2018 - arXiv  
> Author: Thanh Thi Nguyen  
> Link: [原文链接](https://arxiv.org/abs/1803.02965)

### Abstract
这篇文章提出了一个新的基于DQN (Deep Q-networks) 的多目标深度强化学习框架。
我们提出建议使用线性和非线性方法来开发包含single-policy和multi-policy在内的多目标深度强化学习（Multi-Objective Deep Reinforcement Learning, MODRL）框架。
在包含两个目标的深海宝藏环境和三个目标的mountain car问题的两个benchmark问题上的实验结果证明，
该框架可以有效地收敛于最优的帕累托（Pareto）解。
该框架是通用的，它可以在不同的复杂环境中实现不同的深度学习算法。
因此，它克服了目前文献中标准多目标强化学习（Multi-Objective Reinforcement Learning, MORL）方法存在的许多问题。
该框架创建了一个平台作为测试环境，以开发用于解决与当前MORL相关的各种问题的方法。
该框架实现的细节可以参考：[链接](http://www.deakin.edu.au/~thanhthi/drl.htm)

### Introduction
到目前为止，大多数多目标强化学习（MORL）研究都是在相对简单的gridworld任务上进行的，因此为了应用于更复杂的问题领域，将现有算法扩展到更复杂的函数逼近是很重要的。
现有的算法，像tabular Q-learning，都占用了很大的内存，在环境状态空间很大时效率低下且不切实际。
深底强化学习（DRL）方法是克服这个问题的可能的解决方案，因为它仅需要存储这个神经网络或者经验回放的内存。

目前，只有少量关于深度多目标强化学习的研究。
因此，还没有出现标准的benchmarks。
Mossalam et al. 扩展DQN来处理single-policy的线性多目标强化学习。
然后，他们通过在外循环中嵌入扩展的DQN算法解决了寻找凸覆盖集（convex Converage set, CCS - 对任何可能的权重向量都可用的最优策略的完整集合）的multi-policy任务，它被用来确定用于训练的权重向量，从而建立CCS。
他们以两种不同的方式使用两个gridworld小任务作为MODRL的测试问题。
第一种方法是他们直接向DNN（Deep Neural Network）提供了底层离散的或连续的状态信息 -- 这些信息是低维的，所以DQN的容量对于这些任务来说基本是多余的。
第二种方法是MODRL方法的一个更好的evaluation，因为他们使用了一种环境的可视化来生成图片，用于作为DNN的输入。
他们表明，当外部循环改变权重时，通过保留部分（而不是全部）的DNN可以achieve efficiencies。
总的来说，该方法解决了multi-policy linear MORL的问题，但是它是通过串联式的学习这些策略来实现的，而不是并行式的。

Tajmajer也扩展了DQN，但是他使用了一种基于subsumption架构的非线性动作选择方法。
