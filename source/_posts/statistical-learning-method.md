---
title: 《统计学习方法》
tags:
  - 统计学习方法
  - machine learning
  - book
categories:
  - 统计学习方法
date: 2018-11-09 10:51:47
mathjax: true
---

前言
===========
+ 只摘录关键知识点，不做详细笔记，仅为方便日后复习有个方向
+ 学习: 如果一个系统能够通过执行某个过程改进它的性能，这就是学习。 --- Herbert A. Simon


1.统计学习
===========
+ 统计学习的特点，对象，目的，方法，研究
+ 本章主要将监督学习

## 监督学习
+ 从给定有限的训练数据出发, 假定数据是独立同分布的, 而且假设模型属于某个假设空间, 应用某一评价准则, 从假设空间中选取一个最优的模型, 使它对已给训练数据及未知测试数据再给定评价标准意义下有最准确的预测.
+ 基本概念：**input space, output space, feature space**
+ 其它名词：**instance, feature vector, 联合概率分布**
+ 假设空间: $ \mathcal{F} = \\{ f\;|  \mathit{Y} = f(X) \\} $
+ 最终变成求 $\min\limits_{f\in\mathcal{F}}R_{emp}(f)$ 或 $min_{f\in\mathcal{F}}R_{srm}(f)$ 的问题


<!-- more -->

## 统计学习三要素
+ 方法 = 模型 + 策略 + 算法

### 策略
+ 损失函数: 0-1, quadratic, absolute, logarithmic
+ 风险函数(期望损失): $ R_{exp}(f) = E_{p}[L(Y, f(X))] = \int_{x\times y}L(y, f(x))P(x,y)dxdy $
+ 经验风险(经验损失)(empirical loss): $\displaystyle R_{emp}(f) = \frac{1}{N}\sum_{i=1}^{N}L(y_i, f(x_i)) $
+ 根据大数定理, 可用$R_{emp}(f)$估计$R_{exp}(f)$, 但由于现实中样本有限, 甚至很少，所以需要矫正$R_{emp}(f)$
+ 经验风险最小化(ERM)和结构风险最小化(SRM)
  + ERM: 用最优化方法求解$\min\limits_{f\in\mathcal{F}}R_{emp}(f)$
    + 样本容量很小时容易过拟合(over-fitting), 但样本容量大时，学习效果很好
    + 当模型是条件概率分布，损失函数是对数损失函数时，经验风险最小化就等价于极大似然估计(MLE)([证明](http://datahonor.com/2017/03/03/最大似然估计与经验风险最小化/)). 
  + SRM: 等价于正则化(regularizer), 即求 $\min\limits_{f\in\mathcal{F}}R_{srm}(f)$
    + 结构风险: $\displaystyle R_{srm}(f) = R_{emp}(f) + \lambda J(f)$ 
      + 其中 $\lambda J(f)$ 位正则化项或罚项(penalty term)
      + $J(f)$是模型空间复杂度, 为定义在$\mathcal{F}$上的泛函. $f$越复杂, $J(f)$越大.
      + $\lambda \ge 0$是系数, 用以权衡经验风险和模型复杂度
      + $R_{srm}(f)$小要求$R_{emp}(f)$和$J(f)$同时小, $R_{srm}(f)$小的模型往往对训练数据以及未知的测试数据都有较好的预测
      + 当模型是条件概率分布, 损失函数是对数损失函数, 模型复杂度由模型的先验概率分布表示时，结构风险最小化就等价于最大后验概率估计(MAP)(证明)

## 模型评估与选择
+ 训练误差(**tranning error**): 模型关于训练数据集的平均损失
+ 测试误差(**test error**): 模型关于测试数据集的平均损失, 反映了模型的预测能力(泛化能力 **generalization ability**)
+ 过拟合: 所选模型参数过多, 对训练数据预测的很好, 对测试数据预测的很差
+ 模型选择时要选择复杂度适当的模型, 防止过拟合.

## 正则化与交叉验证
+ 此为常用的两种模型选择方法

### 正则化
+ $\min\limits_{f\in\mathcal{F}}R_{srm}(f)$
+ 正则化项: 一般是模型复杂度的单调递增函数，模型越复杂，正则化值越大
> 如参数向量$w$的$L_1$范数$\parallel w_1 \parallel$或$L_2$范数$\frac{1}{2}\parallel w_1 \parallel^2$
+ 模型越复杂, 先验概率越大

### 交叉验证
+ 样本充足时, 可随机切成训练集(用于训练模型), 验证集(用于模型选择, 选择预测误差最小的模型)和测试集(模型评估)
+ 交叉验证: 重复使用数据, 反复切, 反复训练, 测试及模型选择
+ 简单交叉验证: 随机切成训练集和测试集, 选测试误差最小的模型
+ $S$折交叉验证(S-fold cross validation): 切成$S$份, 每次选$S-1$份训练, $1$份测试, 重复$S$次
+ 留一交叉验证: $S=N$, 数据集为$N$, 数据集较少时用

## 泛化能力
+ 即模型$\hat{f}$的预测能力, 用$R_{exp}(\hat{f})$来表示
+ 泛化误差上界: $R(f) \le \hat{R}(f) + \varepsilon(d, N, \delta)$
> $R(f)$为泛化误差
> $\le$右边为泛化误差上界
> $\hat{R}(f)$为训练误差
> $\varepsilon(d, N, \delta) = \sqrt{\frac{1}{2N}(\log d + \log \frac{1}{\delta})}$
+ 训练误差小的模型, 泛化误差也会小

## 生成模型与判别模型
+ 监督学习的方法可以分为: 生成方法(**generative approach**)和判别方法(**discriminative approach**)
+ 生成方法: 先学习$P(X,Y)$再求出$P(Y|X) = \frac{P(X,Y)}{P(X)}$
> 如:朴素贝叶斯法和隐马尔可夫模型
+ 判别方法: 直接学习$f(X)$或$P(Y|X)$
> 如:$k$近邻法, 感知机, 决策树,  逻辑斯蒂回归模型, 最大熵模型, 支持向量机, 提升方法和条件随机场等
> 存在隐变量时, 判别方法不能用

## 分类问题
+ $P(Y|X)$作为分类器 
+ 分类准确率(**accuracy**): 对于给定的测试数据集, 分类正确的样本数与总样本数之比
+ 精确率(**precision**):$P = \frac{TP}{TP+FP}$ 
> True, False, Positive, Negative
+ 召回率(**recall**): $R = \frac{TP}{TP+FN}$
+ $P$和$R$的调和均值$F_1$: $\frac{1}{F_1} = \frac{1}{P} + \frac{1}{R}$, 即$F_1 = \frac{2TP}{2TP+FP+FN}$
+ 许多统计学习方法可以用于分类
> 如: $k$近邻法, 感知机, 朴素贝叶斯法, 决策树, 决策列表, 逻辑斯蒂回归模型, 支持向量机, 提升方法, 贝叶斯网络, 神经网络, Winnow等

## 标注(tagging)问题
+ 可以认为书分类问题的推广，也是更复杂的结构预测(**structure prediction**)问题的简单形式
+ 输入一个观测序列$x_{N+1} = (x_{N+1}^{(1)}, x_{N+1}^{(2)}, x_{N+1}^{(3)}, ... ,x_{N+1}^{(n)})^T$, 找到使条件概率$$P((y_{N+1}^{(1)}, y_{N+1}^{(2)}, y_{N+1}^{(3)}, ... ,y_{N+1}^{(n)})|(x_{N+1}^{(1)}, x_{N+1}^{(2)}, x_{N+1}^{(3)}, ... ,x_{N+1}^{(n)})$$最大的标记序列$y_{N+1} = (y_{N+1}^{(1)}, y_{N+1}^{(2)}, y_{N+1}^{(3)}, ... ,y_{N+1}^{(n)})^T$
> 常用的标注方法:隐马尔科夫模型`和条件随机场

## 回归问题
+ 等价于函数拟合: 选择一条函数曲线使其很好地拟合已知数据且很好地预测未知数据
+ 分类
> 按输入变量的个数: 一元回归和多元回归
> 输入与输出变量的关系模型: 线性回归和非线性回归
> 损失函数是平方损失函数时: 可用最小二乘法(**least squares**)求解



2.感知机(preceptron)
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


3.$k$-NN ($k$-nearest neighbor)
==============
> 一种基本的分类与回归的方法


## 算法
+ 实例 $x$ 所属的类$y$, 有: 
$$\displaystyle y = \arg\max_{c_j}\sum_{x_i \in N_k(x)} I(y_i = c_j), \;\;\;\; i = 1,2,\cdots ,N; \;\; j=1,2,\cdots ,K$$
+ $k=1$时为最近邻法

## $k$近邻模型
+ 模型三要素: 距离度量, $k$值的选择和分类决策规则的确定
+ 距离度量
  + 一般用欧式距离, 也可以是其它距离, 如$L_p$距离(Minkowski距离): $\displaystyle L_p(x_i, x_j) = \left(\sum_{l = 1}^n |x_i^{(l)} - x_j^{(l)}|^p\right)^{\frac{1}{p}}$
  > 欧式距离: $p=2$
  > 曼哈顿距离: $p=1$
  > 各个坐标距离的最大值: $p=\infty$
+ $k$的选择
  + 较小: 近似误差小, 估计误差大
  + 较大: 近似误差大, 估计误差小
  + 应用中通常选取较小的$k$值, 采用交叉验证法选取最优的$k$值
+ 分类决策规则
  + 经验风险最小化: 即$\displaystyle \sum_{x_i \in N_k(x)} I(y_i = c_j)$最大化

## $k$近邻法的实现: $kd$树
+ $kd$树是一种对$k$维空间进行存储以便对其进行快速检索的树形数据结构.
+ $kd$树的每个节点对应于一个$k$维超矩形区域.
+ 此处的$k$与$k$近邻法的$k$不同.
+ $kd$树搜索的平均时间复杂度为$O(\log N)$, 更适用于训练实例数远大于空间维数时的$k$近邻搜索.
+ 当空间维数接近训练实例数时, 它的效率会迅速下降, 几乎接近线性扫描.


4.朴素贝叶斯法 
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


5.决策树
================
+ 决策树是一种基本的分类与回归方法
+ 本质上是从训练数据集归纳出一组分类规则
+ 决策树学习通常包括三个步骤: 特征选择, 决策树的生成和决策树的修剪
+ 内部节点表示一个特征或属性, 叶节点表示一个类

决策树模型与学习
--------------------
+ 损失函数通常是正则化的极大似然函数
+ 需要自下而上进行剪枝, 去掉过于细分的叶节点, 使其回退到父节点, 甚至更高的节点, 避免过拟合, 使其有更好的泛化能力
+ 决策树的生成只考虑局部最优, 决策树的生成则考虑全局最优

特征选择
--------------------
+ 通常的特征选择的准则是信息增益或信息增益比
+ 熵(**entropy**): $\displaystyle H(p) = \sum_{i=1}^n p_i \log p_i$
+ 条件熵(**conditional entropy**): $\displaystyle H(Y|X) = \sum_{i=1}^n p_i H(Y|X=x_i)$
+ 当熵和条件熵中的概率由数据估计(特别是极大似然估计得到)时, 所对应的熵与条件熵分别称为经验熵与经验条件熵
+ 信息增益表示得知特征$X$的信息而使得类$Y$的信息的不确定性减少的程度
+ 特征$A$对训练数据集$D$的信息增益$g(D,A) = H(D) - H(D|A)$, 也成为互信息(**mutual information**)
+ 信息增益比: 
  $$g_R (D,A) = \frac{g(D,A)}{H_A(D)}$$ 
  其中，$H_A (D)$表示训练数据集$D$关于特征$A$的值的熵
  $$\displaystyle H_A (D) = -\sum_{i=1}^n \frac{|D_i|}{D} \log_2 \frac{|D_i|}{D}$$

决策树的生成
--------------------

### ID3算法
> 输入: 训练数据集D, 特征集A, 阈值\varepsilon
> 每次选择$ g(D,A) $最大的特征点递归构建, 直到所有特征的$g(D,A)$均很小($\lt\varepsilon$)或没有特征可以选择为止

### C4.5算法
+ 用信息增益比来选择特征


决策树的剪枝
--------------------
+ 损失函数: $C_\alpha (T) = C(T) + \alpha |T|$, 其中 
$$\displaystyle C(T) = \sum_{t=1}^{|T|}N_t H_t (T) \;\;\;\;\;  H_t (T) = -\sum_k \frac{N_{tk}}{N_t} \log \frac{N_{tk}}{N_t}$$
$N_{tk}$表示树$T$的某一叶节点$t$的第$k$类样本点的数量
+ 若一组叶节点回缩前后的树分别为$T_B$和$T_A$, 当$C_\alpha (T_A) \le C_\alpha (T_B)$时进行剪枝, 将父节点变为新的叶节点
+ 利用损失函数最小原则进行剪枝就是用正则化的极大似然估计进行模型选择 


CART算法
-------------------
> 分类与回归树(Classification and Regression Trees)
> 递归构建二叉决策树再剪枝
> 具体见《统计学习方法》

### CART生成
+ 回归树的生成: 用平方误差最小化准则, 最小二乘回归树生成算法
+ 分类树的生成: 用基尼系数选择最优特征, 同时决定该特征的最优二值切分点

### CART剪枝
+ 首先从生成算法产生的决策树$T_0$底端开始不断剪枝, 直到$T_0$的根节点, 形成一个子树序列$\lbrace T_0, T_1, \cdots, T_n \rbrace$; 
然后通过交叉验证法再独立的验证数据集上对子树序列进行测试, 从中选择最优子树


6.逻辑斯蒂回归与最大熵模型
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



7.支持向量机
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









