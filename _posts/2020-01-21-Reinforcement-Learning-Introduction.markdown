---
layout: post
title: "RL之过程描述"
categories: RL
tags: RL Introduction Process
author: XuLipeng
---

* content
{:toc}


## 前言

这是一篇关于各经典论文对基本的 `RL` 过程的描述的综述(为了方便记忆理解，同一概念统一符号表示)。


## 综述

标准的强化学习（`Reinfocement Learning`） 是由智能体（`agent`）和环境（`enviroment`）组成的。智能体在离散的时间步下与环境进行交互；智能体与环境的一步交互过程可以被描述为：智能体首先观察环境的状态（`state`），然后智能体做出相应的动作（`action`）并作用于环境，最后智能体会接受到环境关于智能体所做动作的反馈，它包括环境的新的状态以及标量的奖励（`reward`）。  
`Note`：1. 智能体是根据策略（`Policy`）来决定智能体在观察到状态`\pi`


## 经典论文参考列表

`DQN`原文链接：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)   
`DDPG`原文链接：[Continous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)  
`A3C`原文链接：[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
`HER`原文链接：[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)  
`UVFA`原文链接：[Universal Value Function Approximator](http://proceedings.mlr.press/v37/schaul15.pdf)  