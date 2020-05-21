---
layout: post
title:  "Proximal Policy Optimization Algorithms [paper]"
date:   2020-05-21 14:48:00 +0800
categories: RL PPO paper
---

## Proximal Policy Optimization Algorithms
### Abstract


我们提出了一种用于强化学习的策略梯度类的方法，该方法表现为通过与环境交互来sample datas和借助`stochastic gradient ascent`来优化一个“surrogate”目标函数，两者之间交替执行。 
与每采样一次数据就进行一次梯度更新的standard策略梯度方法不同，我们提出了一种新颖的目标函数，可以实现多个epochs的小批量更新。
这个新方法我们把它称为`近端策略优化`（proximal policy optimization, ppo），该方法拥有置信域策略优化（TRPO）的部分优势，但是该方法有具有比TRPO更易实现、更一般化、以及更优的样本复杂性的优点。
我们在一组benchmark的任务中测试了PPO算法，包括仿真机器人运动（simulated robotic locomotion）和Atari游戏，实验结果表明PPO比其他在线策略梯度表现的都要好，总体上在样本的复杂性、简易性以及有效性之间达到了一个有效的平衡。


### Introduction

近年来，一些基于神经网络函数逼近的强化学习方法被提出，其中最具影响力的是`deep Q-learning`、`vanilla policy gradient`以及`trust region policy gradient`。
然而这些方法在scalable（大型模型和并行实现）、数据效率以及健壮性（在不进行超参数调优的情况下）方面还有提升的空间。
Q-learning（基于函数逼近）在很多简单的问题上表现不佳，且可解释性较差；
vanilla policy gradient方法在数据效率和健壮性上表现不足；
而TRPO方法则相对来说比较复杂，且与包含noise（比如dropout）或者参数共享（策略与值函数或者辅助任务之间）的架构不兼容。  

本文试图通过引入一种算法来改善目前的状况，该算法在仅使用一阶优化的情况下拥有高的数据效率和TRPO的可靠表现。
我们提出了一个基于`clipped probability ratios`的新式目标，它表示的是策略性能的“pessimistic”的估计（即下界）。
为了优化策略，我们交替的执行基于策略进行采样和在采样的数据上进行几个epochs的优化。