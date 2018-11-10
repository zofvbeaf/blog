---
title: 《统计学习方法》四. 朴素贝叶斯法
tags:
  - 统计学习方法
  - machine learning
  - book
categories:
  - 统计学习方法
date: 2018-11-10 15:23:25
mathjax: true
---


朴素贝叶斯法 
=================
+ 朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法 
+ 朴素贝叶斯法与贝叶斯估计时不同的概念

## 朴素贝叶斯法的学习与分类
+ 朴素贝叶斯分类器可以表示为: 
$$\displaystyle y = \arg\max_{c_k} P(Y=c_k) \prod_j P(X^{(j)}=x^{(j)}|Y=c_k) $$   
+ 即后验概率最大化
+ 后验概率最大化的含义: 根据期望风险最小化准则可以得到后验概率最大化准则 

## 朴素贝叶斯法的参数估计
+ 极大似然法
  + 在朴素贝叶斯法中, 学习意味着估计$P(Y=c_k)$和$P(X^{(j)}=x^{(j)}|Y=c_k)$
  + 先验概率$P(Y=c_k)$的极大似然估计为: $\displaystyle P(Y=c_k) = \frac{\displaystyle\sum_{i=1}^N I(y_i = c_k)}{N}, \;\; k=1,2,\cdots,K$
  + 设第$j$个特征$x^{(j)}$可能取值的集合为$\lbrace a_j1, a_j2, \cdots, a_{jS_j} \rbrace$, 条件概率$P(x^{(j)}=a_{jl} |y=c_k)$的极大似然估计是:
  $$P(X^{(j)}=a_{jl} |Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl} , y_i=c_k)}{\displaystyle\sum_{i=1}^N I(y_i = c_k)}$$
  $$j=1,2,\cdots,n; \;\; l=1,2,\cdots,S_j; \;\; k=1,2,\cdots,K$$
 
+ 朴素贝叶斯算法
+ 贝叶斯估计:
> 极大似然估计可能出现所要估计的概率为$0$的情况, 这会影响到后验概率的计算结果, 使分类产生偏差, 解决这一问题的方法是采用贝叶斯估计
  $$P_\lambda (X^{(j)}=a_{jl} |Y=c_k) = \frac{\displaystyle \sum_{i=1}^N I(x_i^{(j)}=a_{jl} , y_i=c_k) + \lambda}{\displaystyle\sum_{i=1}^N I(y_i = c_k) + S_j \lambda } \;\;\;\;$$ 
  $$P_\lambda (Y=c_k) = \frac{\displaystyle\sum_{i=1}^N I(y_i = c_k) + \lambda}{N + K\lambda}$$
  其中$\lambda \ge 0$, $\lambda = 0$是极大似然估计, $\lambda = 1$是拉普拉斯平滑(**Laplace smoothing**)








