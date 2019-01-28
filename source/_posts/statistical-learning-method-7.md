---
title: 《统计学习方法》七. 支持向量机
tags:
  - 统计学习方法
  - machine learning
  - book
categories:
  - 统计学习方法
date: 2018-11-12 09:50:36
mathjax: true
---


支持向量机
=======================
> 支持向量机(**support vector machines SVM**)是一种二分类模型


线性可分支持向量机与硬间隔最大化
-----------------------
+ 线性可分支持向量机
> 给定线性可分的训练数据集, 通过间隔最大化或等价地求解相应的凸二次规划问题学习得到的分离超平面为 $$ w^\ast \cdot x + b^\ast = 0 $$ 以及相应的分类决策函数 $$ f(x) = sign(w^\ast \cdot x + b^\ast) $$ 称为线性可分支持向量机
+ 函数间隔: 
> 对于给定的训练数据集$T$和超平面$(w,b)$, 定义超平面$(w,b)$关于样本点$(x_i, y_i)$的函数间隔为$$\hat{\gamma_i} = y_i(w\dot x_i + b)$$
> 定义超平面$(w,b)$关于训练数据集$T$函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i, y_i)$的函数间隔之最小值, 即$$\displaystyle \hat{\gamma} =  \min_{i=1,2,\cdots,N} \hat{\gamma_i}$$
+ 几何间隔: $$\gamma_i = y_i\left(\frac{w}{\parallel w \parallel} \cdot x_i + \frac{b}{\parallel w \parallel} \right)$$ $$\displaystyle \gamma = \min_{i=1,2,\cdots,N} \gamma_i$$
+ 间隔最大化
