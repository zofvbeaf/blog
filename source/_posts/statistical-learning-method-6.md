---
title: 《统计学习方法》六. 逻辑斯蒂回归与最大熵模型
tags:
  - 统计学习方法
  - machine learning
  - book
categories:
  - 统计学习方法
date: 2018-11-11 16:54:25
mathjax: true
---


逻辑斯蒂回归与最大熵模型
=============================
> 逻辑斯蒂回归(**logistic regression**)是统计学习中的经典**分类**方法
> 最大熵是概率学习的一个准则, 将其推广到分类问题得到最大熵模型(**maximum entropy model**)
> 两者都属于对数线性模型



逻辑斯蒂回归
-----------------------------
+ 设$X$是连续随机变量, $X$服从逻辑斯蒂分布是指$X$具有下列分布函数和密度函数: 
$$ F(x) = P(X \le x) = \frac{1}{1+e^{-(x-\mu)/\gamma}} $$
$$ f(x) = F'(x) = \frac{e^{-(x-\mu)/\gamma}}{\gamma(1+e^{-(x-\mu)/\gamma})^2} $$
+ 该曲线是以点$(\mu, \frac{1}{2})$为中心堆成的$S$型曲线

