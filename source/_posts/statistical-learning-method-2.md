---
title: 《统计学习方法》二. 感知机
tags:
  - 统计学习方法
  - machine learning
  - book
categories:
  - 统计学习方法
date: 2018-11-09 10:51:47
mathjax: true
---

感知机(preceptron)
=============
> 属于判别模型, 输入为实例的特征向量, 输出为实例的类别
> 是一种线性分类模型

## 感知机模型
+ $f(x) = sign(\omega\cdot x + b), 其中sign(x) = \begin{cases} +1, &{x \ge 0} \\ -1, &{x \lt 0} \end{cases} $
+ 线性分类器: $f(x) = \lbrace f|f(x) = \omega\cdot x + b \rbrace$

## 感知机学习策略
+ 数据集的线性可分性: 存在某个超平面$ S: \omega\cdot x + b = 0 $能够将数据集的正实例点和负实例点完全正确的划分到超平面的两侧
> 即对所有实例$i$有: $y_i = \begin{cases} +1, &{\omega\cdot x + b \ge 0} \\ -1, &{\omega\cdot x + b\lt 0} \end{cases} $
+ 感知机就是要找出这样一个超平面，即确定$\omega$和$b$, 定义(经验)损失函数并将损失函数极小化
> 损失函数: $\displaystyle L(\omega, b) = -\sum_{x_i \in M} y_i(\omega\cdot x_i + b)$
> 其中$M$为所有误分类点的集合

<!-- more -->

## 感知机学习算法
+ 即求解$\displaystyle \min_{\omega, b} L(\omega, b)$的最优化问题
> 任意选取一个超平面$\omega_0, b_0$, 然后用梯度下降法不断地极小化目标函数
> 选取$y_i(\omega\cdot x_i + b) \le 0$
> $$ \omega \gets \omega + \eta y_i x_i$$ $$ b \gets b + \eta y_i $$
> $\eta(0\le\eta\lt 0)$是步长, 又称为学习率
+ 算法的收敛性证明
> 即证明: 设数据集是线性可分的, 经过有限次迭代可以得到一个将训练数据集完全正确划分的分离超平面及感知机模型
> 最终得到误分类次数$k \le (\frac{R}{\gamma})^2$, 其中$\displaystyle R = \max_{1\le i \le N}\parallel \hat{x}_i \parallel$
> 当训练集线性不可分时, 算法不收敛, 迭代结果会发生震荡
+ 对偶形式
> 感知机模型$\displaystyle f(x) = sign(\sum_{j=1}^N \alpha_j y_j x_j\cdot x + b)$
> 其中$\alpha_i = n_i\eta$, 且迭代过程为: $\begin{cases} \alpha_i & \gets \alpha_i + \eta \\ b & \gets b + \eta y_i \end{cases}$

