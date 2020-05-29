---
layout: post
title:  "[D] Multi-Task Learning"
date:   2020-05-28 10:16:00 +0800
categories: RL Multi-Task
---

## [D] Multi-Task Learning
> 多任务学习(MTL)是同时学习多个相关任务，旨在通过使用多个任务共享的相关信息获得更好的性能，其原理是通过利用任务相关性结构在所有任务的联合假设空间中引入归纳偏差。
> 它还可以防止单个任务中的过度适应，从而具有更好的泛化能。

`定义2.1` **多任务学习(MTL)** 是关于同时学习多个任务 $T = {1, 2, \cdots, N}$ 的过程。
每个任务 $t \in T$ 有自己的训练数据 $D_t$，其目标是使所有任务的性能最大化。

### 多任务学习中的任务相关性
在MTL中假设各个任务之间是密切相关的，在任务相关性方面存在的不同假设导致了不同的建模方法。

- Evgeniou和 Pontil［2004］假设各个任务的所有数据都来自同一个空间，而且所有任务模型都接近全局模型。在这种假设下，他们使用正则化的任务耦合参数来对任务之间的关系进行建模；
- Baxter［2000］、Ben-David 和Schuller［2003］假设所有任务都基于相同的表示模型，也就是使用一组共同的已学习特征；
- Daumé Ⅲ,2009；Lee et al.,2007；Yu et al.,2005 假设参数具有相同的先验假设；
- Argyriou et al.,2008 假设任务只在原始空间的较低秩中进行共享（而不是共享整个空间），即任务参数在低维子空间中由不同任务共享；然而，低秩空间对任务不进行区分，当涉及到一些不相关任务时，这种方法的性能会下降；
- Jacob et al.,2009；Xue et al.,2007 假设存在不相交的任务组，并用聚类对任务进行分组，同一簇中的任务被认为是相似的；
- Kumar 等人［2012］假设每个任务的参数向量是有限数量的潜在基础任务或潜在组件的一个线性组合，他们没有使用不相交任务组的假设，而是认为不同组中的任务可以在一个或多个基础上相互重叠，基于这个想法，他们提出了一个名为GO-MTL的模型；
- Maurer等人［2013］提出了在多任务学习中使用稀疏编码和字典学习
- Ruvolo和Eaton［2013b］提出了高效终身学习算法(Efficient Lifelong Learning Algorithm, ELLA) 来扩展GO-MTL，该算法可以显著提高效率，并使其成为在线方法，从而满足LL定义；

### UVFA和GVF
`General Value Functions` 的思想是 Richard S. Sutton 等人在 2011 年提出的，其目的是为了在一个经验流中学习多个值函数。
事实上，该文章是将传统意义上的值函数扩展到了通用的形式下：传统的值函数的计算依赖的是确定的奖励函数、确定的终止奖励函数、策略以及终止的概率；
而GVF则是将奖励函数、终止奖励函数、终止概率变为pseudo的形式，它的含义是不再局限于特定形式，可以表示不同goal、task或demon特有的奖励函数、终止奖励函数以及终止概率。
如在游戏中，奖励可能指的是赢了得+1的奖励($z=+1$)、输了得-1的奖励($z=-1$)，每一步的奖励值 $r=0$，而我们关注的是在游戏中一共走了多少步，这时伪奖励函数 $r=1$、伪终止奖励函数 $z=0$。
注意到，传统的值函数其实是通用值函数的一个特例。

`Horde结构` 是应用 GVFs 的结构，它训练多个GVF以学习环境中的多个知识以回答不同的问题，每一个GVF都有一个独立的策略、奖励函数、终止函数以及终止奖励函数。

`Universal Value Function Approximators` 文章提供了两种关于强化学习应用UVFA的方法：一种是基于Horde学习出的GVFs中训练UVFA；另一种是从UVFA自身中引导训练；

#### UVFA based on Horde
**--------------------------------------------------------------------------------------------------------------------**  
**Algorithm 1** UVFA learning from Horde targets  
**--------------------------------------------------------------------------------------------------------------------**  
01. **Input:** rank $n$, training goals $\mathcal{G}_T$, budgets $b_1, b_2, b_3$  
02. Initialise transition history $\mathcal{H}$  
03. **for** $t = 1$ **to** $b_1$ **do**  
04. &emsp; $\mathcal{H} \leftarrow \mathcal{H} \cup (s_t, a_t, \gamma_{ext}, s_{t+1})$  
05. **end for**
06. **for** $i = 1$ **to** $b_2$ **do**  
07. &emsp; Pick a random transition $t$ from $\mathcal{H}$  
08. &emsp; Pick a random goal $g$ from $\mathcal{G}_T$  
09. &emsp; Update $Q_g$ given a transition $t$   
10. **end for**
11. Initialise data matrix $M$  
12. **for** $(s_t, a_t, \gamma_{ext}, s_{t+1})$ **in** $\mathcal{H}$ **do**  
13. &emsp; **for** $g$ **in** $\mathcal{G}_T$ **do**  
14. &emsp; &emsp; $M_{t,g} \leftarrow Q_g(s_t, a_t)$  
15. &emsp; **end for**  
16. **end for**  
17. Compute rank-$n$ factorisation $M \approx \hat{\phi}^T \hat{\psi}$  
18. Initialise embedding networks $\phi$ and $\psi$  
19. **for** $i = 1$ **to** $b_3$ **do**  
20. &emsp; Pick a random transition $t$ from $\mathcal{H}$  
21. &emsp; Do regression update of $\phi(s_t,a_t)$ toward $\hat{\phi}_t$  
22. &emsp; Pick a random goal $g$ from $\mathcal{G}_T$  
23. &emsp; Do regression update of $\psi(g) toward \hat{\psi}_g$  
24. **end for**  
25. **return** $Q(s,a,g) = h(\phi(s,a), \psi(g))$   

**--------------------------------------------------------------------------------------------------------------------**  

其核心理念就是通过部分状态和目标的value function训练出更为一般通用的value function，可以泛化到unseen的状态和目标中去。

### UVFA based on bootstrapping

$$
\begin{align}
Q(s_t, a_t, g) \leftarrow \alpha(r_g + \gamma_g \mathop{max}\limits_{a'}Q(s_{t+1}, a', g)) + (1 - \alpha)Q(s_t, a_t, g)
\end{align}
$$

其核心理念就是通过end-to-end的方式来直接学习UVFA，这种方法的缺点是不稳定，泛化得到的策略的质量可能会差一些。





























