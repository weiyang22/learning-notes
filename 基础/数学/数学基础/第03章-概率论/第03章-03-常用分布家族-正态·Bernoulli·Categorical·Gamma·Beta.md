---
title: "第03章-03-常用分布家族-正态·Bernoulli·Categorical·Gamma·Beta"
tags:
  - 数学
  - 常用分布家族
  - 正态
  - Bernoulli
  - Categorical
  - Gamma
description: "本章要点：变成你能在 PyTorch 里\"敲出来\"的代码。"
chapter_id: 数学-3-03
last_updated: 2026-07-14
---

# 第03章-03-常用分布家族-正态·Bernoulli·Categorical·Gamma·Beta

变成你能在 PyTorch 里"敲出来"的代码。
>
> **学完你能回答的问题**:
> - 为什么 BCE Loss 对应 Bernoulli 分布, CrossEntropy 对应 Categorical 分布?
> - 深度学习初始化为什么用正态分布?Adam 为什么假设梯度近似正态?
> - Beta 分布为什么是"概率的概率"?Thompson Sampling 的核心机制是什么?
> - 为什么 GPT 输出的 logits 要做 temperature 采样?
> - 推荐系统的 CTR 不确定性为什么用 Beta 分布建模?
>
> **本章与后续章节的关系**:
> - 第 4 章深入"期望/方差/协方差",分布的数字特征
> - 第 5 章基于本章的 Bernoulli/Categorical 讲贝叶斯定理
> - 第 6 章基于本章的正态/Beta/Gamma 讲 MLE/MAP
> - 第 7 章把本章的所有分布统一到"指数族"框架,讲共轭先验
> - 第 8 章基于正态/Bernoulli 讲 A/B 实验的统计推断

---

## 1. 直觉引入

### 1.1 工程师的痛点:为什么损失函数有"标准形式"?

打开任何一个深度学习项目,你会看到这些 loss 函数:

```python
# 二分类
loss = F.binary_cross_entropy_with_logits(logits, labels)

# 多分类
loss = F.cross_entropy(logits, labels)

# 回归
loss = F.mse_loss(pred, target)

# 概率分布的距离
loss = F.kl_div(log_q, log_p, reduction='batchmean')
```

新手会问:**这些 loss 是怎么来的?为什么不是别的形式?**

答案藏在**概率论的最大似然估计 (MLE)** 里:

> **核心命题**:每个标准的 loss 函数,都对应一个"我们对数据服从什么分布的假设"。

- BCE Loss = 假设标签服从 **Bernoulli 分布**,做 MLE。
- CrossEntropy Loss = 假设标签服从 **Categorical 分布**,做 MLE。
- MSE Loss = 假设标签服从**正态分布**(以预测为均值),做 MLE。
- KL 散度 = 直接度量两个分布的距离(等价于 MLE 的"残差")。

所以,**学分布就是学 loss**。这就是为什么本章叫"常用分布家族",这是深度学习 loss 函数背后的概率解释。

### 1.2 直觉:每个分布对应一种"现象"的概率模型

现实世界里有各种各样的"随机现象",抛硬币、测身高、统计点击、记录故障。**每个现象都对应一个最自然的概率模型**,也就是一个分布:

| 现象 | 分布 | 关键参数 |
| :--- | :--- | :--- |
| 抛硬币(是/否) | Bernoulli | 正面概率 $p$ |
| 多选题(选 A/B/C/D) | Categorical | 每个选项的概率 $(p_1, ..., p_K)$ |
| 测身高/测量误差 | 正态 | 均值 $\mu$ + 标准差 $\sigma$ |
| n 次抛硬币的正面数 | Binomial | 试验次数 $n$ + 概率 $p$ |
| 一段时间内的点击数 | Poisson | 平均速率 $\lambda$ |
| 用户在网页的停留时间 | 指数 | 速率 $\lambda$ |
| Poisson rate 的不确定性 | Gamma | 形状 $\alpha$ + 速率 $\beta$ |
| 概率本身的不确定性 | Beta | 形状 $\alpha$ + $\beta$ |
| 小样本均值(方差未知) | 学生 t | 自由度 $\nu$ |

**这 9 个分布**就是本章的全部内容。每个对应一个具体的工程场景,**每个 loss 函数都能从这里"推导"出来**。

### 1.3 ASCII 可视化:9 大分布的"指纹"

下面用 ASCII 把 9 个分布的"形状"画出来。你不需要现在就理解公式,只需要**建立直觉**:每个分布长得不一样,对应不同的现象。

**ASCII 可视化 1:9 大分布 PDF 形状一览**

```text
分布类型与形状对比(横轴 = x,纵轴 = 密度或概率)

1. 离散 (Bernoulli)        2. 正态 N(0,1)       3. Binomial(10,0.5)    4. Poisson(3)
   P ↑                       f(x) ↑               P ↑                     P ↑
   1 ┤    ▓                  0.4┤    ╱─╲         0.20┤      ▓▓▓▓         0.22┤    ▓▓
   0 ┤▓  ▓ ▓ ▓              0.2┤  ╱     ╲       0.10┤   ▓▓▓▓▓▓▓▓▓▓      0.13┤  ▓▓▓▓
     └─────┴──→ x           0.0┤╱_________╲→ x  0.00┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓     0.00┤▓▓▓▓▓
       x=0,1,2,3               -3  0  3              0 1 2 3 4 5 6 7 8       0 1 2 3 4 5

5. 指数 Exp(1)             6. Gamma(2,1)         7. Beta(2,5)            8. 学生 t(3)
   f(x) ↑                   f(x) ↑                f(x) ↑                  f(x) ↑
   1.0┤╲                    0.30┤    ╱╲           2.5┤   ╱╲                0.35┤   ╱╲
   0.5┤  ╲___               0.15┤  ╱    ╲___      1.0┤  ╱   ╲___           0.14┤ ╱   ╲___
   0.0┤      ╲_____→ x      0.00┤╱_________╲→ x   0.0┤       ╲___→ x      0.00┤╱_______╲___→ x
        0  2  4                0   2   4             0.0  0.5  1.0              -4  0  4
```

**怎么读这张图**:
- **离散分布**(Bernoulli/Categorical/Binomial/Poisson):只在某些整数点有非零概率,中间用 `▓` 表示"概率质量"。
- **连续分布**(正态/指数/Gamma/Beta/学生 t):用连续曲线描述密度。
- **形状记忆**:
  - 钟形 = 正态 / 学生 t / Binomial(近似)
  - 单调下降 = 指数 / Gamma(α<1)
  - 先升后降右偏 = Poisson / Gamma(α>1)
  - 集中在 [0,1] = Beta

### 1.4 案例:为什么 BCE Loss 对应 Bernoulli, CrossEntropy 对应 Categorical?

这是本章最核心的"翻译"。我们直接推导两个最常用的 loss。

**情况 1:二分类 → BCE Loss**

假设标签 $y \in \{0, 1\}$ 服从 Bernoulli 分布,参数为 $p$ (即 $P(y=1)=p$, $P(y=0)=1-p$):
$$
P(y) = p^y (1-p)^{1-y}
$$

我们用模型预测 $\hat{p} = \sigma(z)$,则单个样本的似然是:
$$
L = \hat{p}^y (1-\hat{p})^{1-y}
$$

取负对数:
$$
-\log L = -y \log \hat{p} - (1-y) \log(1-\hat{p})
$$

这**就是 BCE Loss**!

**情况 2:多分类 → CrossEntropy Loss**

假设标签 $y$ 是 one-hot 向量 $(y_1, ..., y_K)$,真实类别 $c$ 对应 $y_c = 1$,其他为 0,服从 Categorical 分布,参数 $(p_1, ..., p_K)$:
$$
P(y) = \prod_{i=1}^K p_i^{y_i}
$$

取负对数:
$$
-\log L = -\sum_{i=1}^K y_i \log p_i = -\log p_c
$$

**就是 CrossEntropy Loss**!

**结论**:BCE 和 CrossEntropy 都不是"凭空设计的损失",而是"对数据分布做 MLE 的副产品"。**理解分布 = 理解 loss**。

### 1.5 本章路线图

| 节 | 分布 | 工程应用 | ML 翻译 |
| :--- | :--- | :--- | :--- |
| 2 | Bernoulli | 二分类 / A/B | Sigmoid + BCE |
| 3 | Categorical | 多分类 | Softmax + CE |
| 4 | 正态 | 回归 / 初始化 / 采样 | MSE / Xavier / Temperature |
| 5 | Binomial | A/B 显著性 | 样本量计算 |
| 6 | Poisson | 罕见事件建模 | 推荐系统点击 |
| 7 | 指数 | 停留时间建模 | 无记忆性 |
| 8 | Gamma | Poisson rate 先验 | 贝叶斯推断 |
| 9 | Beta | CTR 不确定性 | Thompson Sampling |
| 10 | 学生 t | 小样本 A/B | 重尾鲁棒 |

### 1.6 本节回顾

**你刚建立的直觉**:
1. 每个标准 loss = 一个分布假设 + MLE。
2. 9 大分布对应 9 类工程现象。
3. 离散分布看"质量点",连续分布看"密度曲线"。
4. BCE → Bernoulli, CrossEntropy → Categorical。
5. 分布形状有"指纹":钟形/右偏/集中在 [0,1]/重尾。

**下一步**:逐个击破 9 个分布。从最简单的 Bernoulli 开始。

---

## 2. Bernoulli 分布 (重点)

### 2.1 定义

**定义 2.1 (Bernoulli 分布)**:随机变量 $X$ 只取 0 或 1,概率分别为:
$$
P(X=1) = p, \quad P(X=0) = 1-p, \quad 0 \leq p \leq 1
$$

记作 $X \sim \text{Bernoulli}(p)$。

**概率质量函数 (PMF)**:
$$
P(X=x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

**联合形式**:如果你看到 "$p^x (1-p)^{1-x}$" 觉得别扭,可以验证:
- $x = 1$: $p^1 (1-p)^0 = p$ ✓
- $x = 0$: $p^0 (1-p)^1 = 1-p$ ✓

### 2.2 期望与方差

**期望**:
$$
\mathbb{E}[X] = 1 \cdot P(X=1) + 0 \cdot P(X=0) = p
$$

**方差**:
$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = p - p^2 = p(1-p)
$$

**手算 1**:$p = 0.3$,求 $\mathbb{E}[X]$ 和 $\text{Var}(X)$。
- $\mathbb{E}[X] = 0.3$
- $\text{Var}(X) = 0.3 \times 0.7 = 0.21$

**手算 2**:$p = 0.5$,求 $\mathbb{E}[X]$ 和 $\text{Var}(X)$。
- $\mathbb{E}[X] = 0.5$
- $\text{Var}(X) = 0.5 \times 0.5 = 0.25$ (方差最大)

**手算 3**:$p = 0$ 或 $p = 1$,求 $\text{Var}(X)$。
- $\text{Var}(X) = 0$ (确定性事件没有方差)

**关键观察**:方差 $p(1-p)$ 在 $p = 0.5$ 处取最大值 0.25。这告诉我们:**最不确定的 Bernoulli 分布是均匀的硬币**。

### 2.3 矩母函数与特征函数(进阶,选读)

**矩母函数 (MGF)**:
$$
M_X(t) = \mathbb{E}[e^{tX}] = (1-p) e^{0 \cdot t} + p e^{1 \cdot t} = 1 - p + p e^t
$$

**特性**:$M_X^{(n)}(0) = \mathbb{E}[X^n]$,所以:
- $M_X'(0) = p e^0 = p = \mathbb{E}[X]$ ✓
- $M_X''(0) = p e^0 = p = \mathbb{E}[X^2]$ (因为 $X^2 = X$ 对 0/1 变量)

**用途**:MGF 在第 7 章"指数族"里会用到,Bernoulli 是指数族的一个特例。

### 2.4 案例:CTR 预测 = 单次 Bernoulli

**场景**:你做推荐系统的 CTR 预估。一个用户对一个物品,要么点击($y=1$)要么不点击($y=0$)。

**数学建模**:
- 标签 $y$ 是一次"事件"的结果(点击 vs 不点击)
- 假设 $y \sim \text{Bernoulli}(\hat{p})$,其中 $\hat{p}$ 是模型预测的点击概率
- 训练数据 $\{(x_i, y_i)\}_{i=1}^N$,每个样本独立

**MLE 推导**:
似然:
$$
L(\hat{p}) = \prod_{i=1}^N \hat{p}^{y_i} (1-\hat{p})^{1-y_i}
$$

取对数:
$$
\log L = \sum_{i=1}^N [y_i \log \hat{p} + (1-y_i) \log(1-\hat{p})]
$$

对 $\hat{p}$ 求导,令为 0:
$$
\frac{\partial \log L}{\partial \hat{p}} = \sum_{i=1}^N \left[ \frac{y_i}{\hat{p}} - \frac{1-y_i}{1-\hat{p}} \right] = 0
$$

解得:
$$
\hat{p}_{\text{MLE}} = \frac{\sum_i y_i}{N} = \bar{y} \quad \text{(经验点击率)}
$$

**结论**:CTR 的 MLE 就是"平均点击率"。在样本量足够大时,这就是真实 CTR 的无偏估计。

**手算 4**:10000 次曝光中,350 次点击,求 $\hat{p}_{\text{MLE}}$ 和 95% 置信区间(用 Wald 近似)。
- $\hat{p} = 350 / 10000 = 0.035$
- $\text{SE} = \sqrt{\hat{p}(1-\hat{p})/N} = \sqrt{0.035 \cdot 0.965 / 10000} \approx 0.00184$
- 95% CI: $0.035 \pm 1.96 \times 0.00184 = (0.0314, 0.0386)$

### 2.5 案例:为什么 Sigmoid 输出 $\in [0, 1]$?

**Sigmoid 函数**:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**两个观察**:
1. $\sigma(z) \in (0, 1)$,可以解释为概率。
2. $\sigma'(z) = \sigma(z)(1-\sigma(z))$,这是**Bernoulli 方差** $p(1-p)$ 的形式!

**为什么是 Bernoulli?**

Sigmoid 输出的 $\hat{p}$ 解释为"概率",而 $\hat{p}$ 决定 Bernoulli 分布:
$$
\hat{p} = \sigma(z) \Rightarrow P(y=1) = \hat{p}, P(y=0) = 1-\hat{p}
$$

Sigmoid 的导数 $\hat{p}(1-\hat{p})$ 既是"自洽的",导数等于自身方差的代数形式,又是**信息论意义上的最大熵**,$p=0.5$ 处分布最不确定(熵最大)。

### 2.6 PyTorch 实现:Bernoulli 采样与概率计算

```python
import torch
from torch.distributions import Bernoulli

# 方式 1:直接用 torch.bernoulli
p = torch.tensor([0.3, 0.5, 0.8])
samples = torch.bernoulli(p)  # 按 p 的概率采样 0/1
print("Samples:", samples)  # 比如:tensor([0., 1., 1.])

# 方式 2:用 torch.distributions.Bernoulli(更灵活)
dist = Bernoulli(p)
samples = dist.sample((1000,))  # 1000 次采样
print("Sample mean:", samples.mean(dim=0))  # ≈ [0.3, 0.5, 0.8]

# 计算 log_prob
log_p = dist.log_prob(torch.tensor([1., 0., 1.]))
print("log P(X=1,0,1):", log_p)
# = [log(0.3), log(0.5), log(0.8)]
```

### 2.7 5 个手算例子汇总

**例 1** (基本期望):$X \sim \text{Bernoulli}(0.7)$,求 $\mathbb{E}[X]$。
$\mathbb{E}[X] = 0.7$。

**例 2** (基本方差):$X \sim \text{Bernoulli}(0.2)$,求 $\text{Var}(X)$。
$\text{Var}(X) = 0.2 \cdot 0.8 = 0.16$。

**例 3** (联合概率):$X_1 \sim \text{Bernoulli}(0.3), X_2 \sim \text{Bernoulli}(0.4)$,独立,求 $P(X_1=1, X_2=0)$。
$P = 0.3 \times (1-0.4) = 0.18$。

**例 4** (MLE):3 次独立试验,结果为 (1, 0, 1),求 $p$ 的 MLE。
$\hat{p} = (1+0+1)/3 = 2/3$。

**例 5** (Wald 区间):500 次试验,200 次成功,求 95% CI。
- $\hat{p} = 0.4$
- $\text{SE} = \sqrt{0.4 \cdot 0.6 / 500} \approx 0.0219$
- 95% CI: $0.4 \pm 1.96 \cdot 0.0219 = (0.357, 0.443)$

### 2.8 本节回顾

**核心收获**:
1. Bernoulli = 单次 0/1 试验,参数 $p \in [0,1]$。
2. 期望 = $p$,方差 = $p(1-p)$ (在 $p=0.5$ 时最大)。
3. **BCE Loss = Bernoulli 分布的负对数似然**。
4. CTR 预测的 MLE 就是经验点击率。
5. Sigmoid 的导数 $\sigma(1-\sigma)$ 是 Bernoulli 方差的代数形式。

---

## 3. Categorical 分布 (Multinoulli, 重点)

### 3.1 定义

**定义 3.1 (Categorical 分布)**:随机变量 $X$ 取 $K$ 个离散值之一,概率为 $(p_1, p_2, ..., p_K)$,其中:
- $p_i \geq 0$ 对所有 $i$
- $\sum_{i=1}^K p_i = 1$

记作 $X \sim \text{Categorical}(p_1, ..., p_K)$。

**PMF**:用 one-hot 向量 $x = (x_1, ..., x_K)$ (只有一个 $x_c = 1$,其他为 0) 表示:
$$
P(X = x) = \prod_{i=1}^K p_i^{x_i} = p_c
$$

其中 $c$ 是 $x$ 中为 1 的位置。

**与 Bernoulli 的关系**:
- Bernoulli 是 Categorical 在 $K=2$ 时的特例。
- Bernoulli 也可以写成"one-hot $(y, 1-y)$"的形式。
- 但更常见的命名:**多类 Bernoulli 称为 Multinoulli 或 Categorical**。

### 3.2 One-hot 编码视角

**One-hot 编码**:把类别标签转换为长度为 $K$ 的二进制向量,真实类别位置为 1,其他为 0。

例:3 分类,真实类别 = "猫" (假设 $c=1$):
$$
y = (1, 0, 0)
$$

**数学好处**:Categorical 分布的 PMF 可以写成**所有维度的乘积**:
$$
P(y) = \prod_{i=1}^K p_i^{y_i} = p_1^{y_1} \cdot p_2^{y_2} \cdots p_K^{y_K}
$$

由于只有一个 $y_c = 1$,其他都是 0,实际就是 $P(y) = p_c$。

**对数似然**:
$$
\log P(y) = \sum_{i=1}^K y_i \log p_i = \log p_c
$$

### 3.3 期望与协方差矩阵

**期望**(one-hot 形式):
$$
\mathbb{E}[X] = (p_1, p_2, ..., p_K) = \mathbf{p}
$$

**协方差矩阵** $\Sigma \in \mathbb{R}^{K \times K}$:
- 对角线元素:$\text{Var}(X_i) = p_i (1 - p_i)$
- 非对角线元素:$\text{Cov}(X_i, X_j) = -p_i p_j$ (对 $i \neq j$)

**矩阵形式**:
$$
\Sigma = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top
$$

**手算 1**:$K=3$, $\mathbf{p} = (0.5, 0.3, 0.2)$,求协方差矩阵。
$$
\Sigma = \begin{pmatrix} 0.25 & 0 & 0 \\ 0 & 0.21 & 0 \\ 0 & 0 & 0.16 \end{pmatrix} - \begin{pmatrix} 0.25 \\ 0.09 \\ 0.04 \end{pmatrix} = \begin{pmatrix} 0 & -0.15 & -0.10 \\ -0.15 & 0 & -0.06 \\ -0.10 & -0.06 & 0 \end{pmatrix}
$$

注意:**所有非对角线都是负的**。这是因为 one-hot 向量只有一个 1,如果 $X_1 = 1$,则 $X_2, X_3$ 必为 0,所以"负相关"。

**手算 2**:求 $\Sigma$ 的秩。
- $\Sigma \mathbf{1} = \text{diag}(\mathbf{p}) \mathbf{1} - \mathbf{p} (\mathbf{p}^\top \mathbf{1}) = \mathbf{p} - \mathbf{p} \cdot 1 = \mathbf{0}$
- 所以 $\mathbf{1}$ 是 $\Sigma$ 的零特征向量,$\Sigma$ 是**秩 $K-1$ 矩阵**。

### 3.4 5 个手算例子

**例 1** (简单 PMF):$K=4$, $\mathbf{p} = (0.1, 0.2, 0.3, 0.4)$。$P(X=3) = ?$
$P(X=3) = p_3 = 0.3$。

**例 2** (取样):从 $\mathbf{p} = (0.5, 0.3, 0.2)$ 中采样,问 10000 次中类别 1 出现几次?
期望 = $10000 \times 0.5 = 5000$ 次。

**例 3** (归一化检查):$\mathbf{p} = (0.3, 0.4, 0.2)$,归一化常数 $Z = 0.3+0.4+0.2 = 0.9 \neq 1$。
归一化后 $\mathbf{p}' = (1/3, 4/9, 2/9)$。

**例 4** (KL 散度):两个 Categorical $\mathbf{p} = (0.5, 0.5)$ 和 $\mathbf{q} = (0.9, 0.1)$,求 $D_{KL}(\mathbf{p} \| \mathbf{q})$。
$$
D_{KL} = 0.5 \log(0.5/0.9) + 0.5 \log(0.5/0.1) = 0.5 \cdot (-0.588) + 0.5 \cdot 1.609 = 0.510 \text{ nat}
$$

**例 5** (温度缩放):原始 logits $z = (2, 1, 0)$,softmax 概率 = ?
- $e^z = (7.39, 2.72, 1.00)$,sum = 11.11
- $\text{softmax}(z) = (0.665, 0.245, 0.090)$
- 温度 $T = 2$: $\text{softmax}(z/T) = \text{softmax}((1, 0.5, 0))$
  - $e^{(1, 0.5, 0)} = (2.72, 1.65, 1.00)$, sum = 5.37
  - 概率 = $(0.506, 0.307, 0.186)$ , **更均匀**(温度越高越接近均匀)

### 3.5 案例:为什么 Softmax 输出 = Categorical 分布参数

**Softmax 函数**:
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

**三个性质**:
1. 输出 $\in (0, 1)$ 对每个 $i$。
2. $\sum_i \text{softmax}(z_i) = 1$ , 归一化。
3. 单调性: $z_i$ 越大, $\text{softmax}(z_i)$ 越大。

**完美匹配 Categorical**:
- 归一化 → $\sum p_i = 1$ ✓
- 非负 → $p_i \geq 0$ ✓
- 单调性 → 大 logit 对应大概率 ✓

**为什么用 $\exp$?**两个原因:
1. **指数族**:Softmax 是广义线性模型的标准链接函数,与 Categorical 分布的指数族形式自然匹配。
2. **梯度友好**:$\partial \text{softmax}(z_i) / \partial z_j$ 有解析形式,且与自身值有关(后面会讲"工程美公式")。

### 3.6 案例:多分类 CrossEntropy 损失

**回顾**:CrossEntropy Loss 就是 Categorical 的负对数似然。

**PyTorch 实现细节**:
```python
import torch
import torch.nn.functional as F

# logits: (batch, num_classes)
logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 1.5, 2.0]])
# labels: (batch,) 类别索引
labels = torch.tensor([0, 2])

# 数值稳定的 CE loss
loss = F.cross_entropy(logits, labels)
print("Loss:", loss)  # tensor(0.4082)

# 等价但慢一点的方式(显式 softmax + log + nll)
probs = F.softmax(logits, dim=-1)
log_probs = torch.log(probs)
loss_manual = F.nll_loss(log_probs, labels)
print("Manual loss:", loss_manual)  # 一样
```

**"工程美公式"**:CrossEntropy 对 logits 的梯度:
$$
\frac{\partial L_{\text{CE}}}{\partial z_i} = p_i - y_i
$$

其中 $p_i = \text{softmax}(z_i)$, $y_i$ 是 one-hot 标签。

**推导** (3 步链式):
- $\partial L / \partial p_i = -y_i / p_i$
- $\partial p_i / \partial z_j = p_i (\delta_{ij} - p_j)$
- $\partial L / \partial z_j = \sum_i (-y_i / p_i) p_i (\delta_{ij} - p_j) = -y_j + p_j \sum_i y_i = p_j - y_j$

这个公式是 PyTorch 高效实现 CE 的核心:**不用构造 Jacobian,直接 `softmax(z) - y` 即可**。

### 3.7 案例:GPT 输出 logits 的温度采样

**场景**:GPT 自回归生成 token,最后输出 logits 向量 $z \in \mathbb{R}^V$ ($V$ = 词表大小)。

**温度采样**:
$$
p_i = \text{softmax}(z_i / T)
$$

- $T \to 0$:概率集中在 argmax → **贪心解码**
- $T = 1$:标准 softmax
- $T \to \infty$:概率趋近均匀 → **完全随机**
- $T < 1$ (如 0.7):概率更尖锐 → **更确定性**

**手算 6**:logits $z = (3, 1, 0)$,词表 $V = 3$。
- $T = 1$: $\text{softmax} = (0.844, 0.114, 0.042)$
- $T = 0.5$: $\text{softmax}(z/0.5) = \text{softmax}((6, 2, 0)) = (0.997, 0.003, 0.000)$
- $T = 2$: $\text{softmax}(z/2) = \text{softmax}((1.5, 0.5, 0)) = (0.629, 0.232, 0.139)$

**工程意义**:
- **高质量生成** (摘要、问答): $T = 0.7 \sim 1.0$
- **多样性生成** (创意写作): $T = 1.2 \sim 1.5$
- **推理 / 数学**: $T \to 0$ (近似贪心)

### 3.8 PyTorch 实现

```python
import torch
from torch.distributions import Categorical

# 方式 1:概率形式
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
dist = Categorical(probs=probs)
samples = dist.sample((1000,))
print("Sample counts:", torch.bincount(samples))  # 接近 [100, 200, 300, 400]

# 方式 2:logits 形式(数值更稳定)
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 不需要归一化
dist = Categorical(logits=logits)
print("probs:", dist.probs)  # softmax 后的概率

# 温度采样
def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    dist = Categorical(logits=scaled_logits)
    return dist.sample()

logits = torch.tensor([3.0, 1.0, 0.5])
for T in [0.5, 1.0, 2.0]:
    samples = [sample_with_temperature(logits, T).item() for _ in range(5)]
    print(f"T={T}: {samples}")
```

### 3.9 本节回顾

**核心收获**:
1. Categorical 是 Bernoulli 的多类推广 ($K \geq 2$)。
2. One-hot 编码让 PMF 写成乘积形式,便于计算。
3. Softmax 输出 = Categorical 参数,CrossEntropy = 负对数似然。
4. **梯度公式 $\partial L / \partial z = p - y$** 是 PyTorch 的高效实现核心。
5. 温度 $T$ 控制生成的"确定性 vs 多样性"。
6. 协方差矩阵 $\Sigma = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ 是秩 $K-1$ 矩阵。

---

## 4. 正态 (高斯) 分布 (重点)

### 4.1 一元正态分布

**定义 4.1 (一元正态)**:随机变量 $X$ 的概率密度函数 (PDF) 为:
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

记作 $X \sim \mathcal{N}(\mu, \sigma^2)$。两个参数:
- $\mu$:均值(位置参数)
- $\sigma^2$:方差(尺度参数),$\sigma$ 是标准差

**关键性质**:
- $\int_{-\infty}^{\infty} f(x) dx = 1$ (归一化常数保证)
- $f(x) > 0$ 对所有 $x$
- $f(\mu + x) = f(\mu - x)$ (关于 $\mu$ 对称)

### 4.2 钟形曲线的几何直觉

**为什么是"钟形"?**三个因素叠加:
1. $\exp(-(x-\mu)^2/2\sigma^2)$:距离 $\mu$ 越远,密度指数衰减。
2. $1/\sqrt{2\pi\sigma^2}$:归一化常数,保证密度积分为 1。
3. 平滑、对称、单峰。

**ASCII 可视化 2:正态分布 PDF 与参数变化**

```text
f(x)
   ↑
0.4 ┤           ╱──╲
0.3 ┤          ╱ σ=0.5╲
0.2 ┤         ╱   σ=1   ╲___
0.1 ┤       ╱     σ=2       ╲____
0.0 ┤____╱_________________________╲___→ x
    -4   -2    0    2    4    6    8

观察:σ 越小 → 峰越尖;σ 越大 → 越分散;所有曲线对称 + 面积为 1
```

### 4.3 标准正态

**定义 4.2 (标准正态)**:$\mu = 0$, $\sigma^2 = 1$ 的正态分布:
$$
\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
$$

记作 $X \sim \mathcal{N}(0, 1)$。

**关键数值**:
- $\phi(0) = 1/\sqrt{2\pi} \approx 0.3989$ (不是 1!)
- $\phi(1) \approx 0.2420$
- $\phi(2) \approx 0.0540$
- $\phi(3) \approx 0.0044$ (几乎为 0)

**标准化变换**:任意 $X \sim \mathcal{N}(\mu, \sigma^2)$ 可以变换为标准正态:
$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

**CDF**:$\Phi(x) = \int_{-\infty}^x \phi(t) dt$,常用值:
- $\Phi(0) = 0.5$
- $\Phi(1.96) \approx 0.975$ (95% 分位数)
- $\Phi(2.58) \approx 0.995$ (99% 分位数)

### 4.4 期望与方差

**期望**:
$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/2\sigma^2} dx = \mu
$$

**方差**:
$$
\text{Var}(X) = \mathbb{E}[(X-\mu)^2] = \sigma^2
$$

**手算证明 (期望)**:
令 $u = (x-\mu)/\sigma$,则 $dx = \sigma du$, $x = \mu + \sigma u$:
$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} (\mu + \sigma u) \frac{1}{\sqrt{2\pi}} e^{-u^2/2} du = \mu \underbrace{\int \phi(u) du}_{=1} + \sigma \underbrace{\int u \phi(u) du}_{=0 \text{ (奇函数)}} = \mu
$$

### 4.5 多元正态分布

**定义 4.3 (多元正态)**:$\mathbf{X} \in \mathbb{R}^d$ 的 PDF:
$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

记作 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$。

**参数**:
- $\boldsymbol{\mu} \in \mathbb{R}^d$:均值向量
- $\Sigma \in \mathbb{R}^{d \times d}$:**协方差矩阵**,必须半正定 (SPD)。

**几何**:等密度面是椭球:
$$
(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) = c^2
$$

椭球的形状由 $\Sigma$ 的特征值决定(详见第 2-线代 第 5 章)。

### 4.6 Mahalanobis 距离

**定义 4.4 (Mahalanobis 距离)**:
$$
d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

**直觉**:把 $\Sigma$ 看作"拉伸变换",Mahalanobis 距离 = 在"拉伸后空间"里的欧氏距离。

**特殊情况**:
- $\Sigma = I$ (单位阵):$d_M = \|\mathbf{x} - \boldsymbol{\mu}\|_2$ (欧氏距离)
- $\Sigma = \text{diag}(\sigma_i^2)$: $d_M = \sqrt{\sum_i (x_i - \mu_i)^2 / \sigma_i^2}$ (标准化欧氏距离)

**应用**:
- **异常检测**:Mahalanobis 距离 $> \chi^2_{d, 0.99}$ 判为异常。
- **k-NN 改进**:用 Mahalanobis 距离替代欧氏距离,适应不同维度量纲。

### 4.7 5 个手算例子

**例 1** (标准化):$X \sim \mathcal{N}(10, 4)$,求 $P(X > 12)$。
- $Z = (12-10)/\sqrt{4} = 1$
- $P(X > 12) = P(Z > 1) = 1 - \Phi(1) \approx 1 - 0.8413 = 0.1587$

**例 2** (经验法则):$\mathcal{N}(\mu, \sigma^2)$ 中,68% 数据落在 $[\mu-\sigma, \mu+\sigma]$。
验证: $P(-\sigma < X-\mu < \sigma) = P(-1 < Z < 1) = 2\Phi(1) - 1 \approx 0.683$ ✓
类似:95% 在 $[\mu-2\sigma, \mu+2\sigma]$, 99.7% 在 $[\mu-3\sigma, \mu+3\sigma]$。

**例 3** (二元正态):$\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 4 \end{pmatrix}$。求 $\Sigma^{-1}$ 和 Mahalanobis 等距面。
- $\det \Sigma = 16 - 4 = 12$
- $\Sigma^{-1} = \frac{1}{12}\begin{pmatrix} 4 & -2 \\ -2 & 4 \end{pmatrix} = \begin{pmatrix} 1/3 & -1/6 \\ -1/6 & 1/3 \end{pmatrix}$
- 椭球方程: $\frac{1}{3}(x_1^2 - x_1 x_2 + x_2^2) = c^2$ (中心在原点)

**例 4** (线性组合):$X_1, X_2 \sim \mathcal{N}(0, 1)$ 独立, $Y = X_1 + X_2$。$Y$ 的分布?
$Y \sim \mathcal{N}(0, 1^2 + 1^2) = \mathcal{N}(0, 2)$。

**例 5** (条件分布):$(X_1, X_2) \sim \mathcal{N}\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right)$,求 $X_2 | X_1 = x_1$ 的分布。
$$
X_2 | X_1 = x_1 \sim \mathcal{N}(\rho x_1, 1 - \rho^2)
$$
这是**回归的源头**:给定特征,预测目标 = 一个条件正态的均值。

### 4.8 案例:为什么深度学习初始化用正态?

**问题**:权重矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 如何初始化?

**朴素方案**:全 0 或全 1 , 对称性问题(所有神经元一样)。

**Xavier/Glorot 初始化**(2010):
$$
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)
$$

**理论依据**:**保持激活值和梯度的方差在层间稳定**。

**推导** (简版):
设输入 $x \sim \mathcal{N}(0, 1)$,权重 $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ 独立,则输出:
$$
y_i = \sum_j W_{ij} x_j, \quad \text{Var}(y_i) = d_{\text{in}} \cdot \sigma^2 \cdot 1
$$

为保持 $\text{Var}(y) = 1$,需 $\sigma^2 = 1/d_{\text{in}}$。Xavier 进一步除以 2(考虑 ReLU 一半置零)取平均:
$$
\sigma^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}}
$$

**PyTorch 默认**:
```python
import torch.nn as nn

linear = nn.Linear(512, 256)  # 默认 Xavier 初始化
print(linear.weight.std().item())  # ≈ sqrt(2/(512+256)) ≈ 0.044
```

### 4.9 案例:为什么 Adam 假设梯度近似正态?

**Adam 公式**:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**隐含假设**:梯度 $g_t$ 在一个"窗口"内近似 i.i.d.,且**近似正态**。

**为什么?**
- $v_t$ 估计的是 $\mathbb{E}[g_t^2]$ 的指数移动平均。
- 若 $g_t \sim \mathcal{N}(0, \sigma^2)$,则 $\mathbb{E}[g_t^2] = \sigma^2$,$v_t \approx \sigma^2$。
- 所以 $\hat{m}_t / \sqrt{\hat{v}_t} \approx g_t / \sigma \sim \mathcal{N}(0, 1)$ , **自适应归一化**。

**这个归一化的好处**:
- 学习率 $\eta$ 不需要"全局调小"以适应大方差参数。
- 每个参数根据自己的"近期梯度尺度"自适应。
- 当梯度真是正态时,Adam 等价于"梯度归一化 + 恒定步长"。

**实际偏离**:深度网络的梯度常是**重尾**的(异常值多),不是完美正态。这种偏离是 AdamW、Shampoo 等改进的动机。

### 4.10 正态分布的"线性保持性"

**定理 4.1**:若 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, $A$ 为矩阵, $\mathbf{b}$ 为向量,则:
$$
A\mathbf{X} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\Sigma A^\top)
$$

**应用**:神经网络的一层是 $y = Wx + b$,输入 $x$ 正态则输出正态。这是为什么 BatchNorm/LayerNorm 的统计量假设正态是合理的。

### 4.11 PyTorch 实现

```python
import torch
from torch.distributions import Normal, MultivariateNormal

# 一元正态
mu, sigma = 0.0, 1.0
normal = Normal(mu, sigma)
samples = normal.sample((10000,))
print("Sample mean:", samples.mean().item())  # ≈ 0
print("Sample std:", samples.std().item())   # ≈ 1

# log_prob
log_p = normal.log_prob(torch.tensor([0.0, 1.0, -1.0]))
print("log densities:", log_p)
# ≈ [-0.919, -1.419, -1.419]  (因为 log φ(x) = -0.5 log(2π) - x²/2)

# 多元正态
mu = torch.zeros(3)
cov = torch.eye(3)  # 单位协方差
mvn = MultivariateNormal(mu, cov)
samples = mvn.sample((1000,))
print("Sample shape:", samples.shape)  # (1000, 3)
```

### 4.12 本节回顾

**核心收获**:
1. 正态分布 = 钟形曲线,两个参数 $\mu$ (位置) + $\sigma^2$ (尺度)。
2. 标准正态 $\mathcal{N}(0,1)$: $\phi(0) \approx 0.399$, 不是 1。
3. 多元正态的等密度面是椭球,由 $\Sigma$ 的特征值决定形状。
4. Mahalanobis 距离 = "协方差归一化后的欧氏距离"。
5. **Xavier 初始化保持层间方差稳定** , 这是"为什么深度学习初始化用正态"的答案。
6. **Adam 假设梯度近似正态** , 这样 $v_t$ 估计的就是 $\sigma^2$, 自适应归一化。
7. 正态在线性变换下保持 , 这给深度网络的统计分析提供了基础。

---

## 5. Binomial 分布

### 5.1 定义

**定义 5.1 (Binomial 分布)**:$n$ 次独立 Bernoulli($p$) 试验中,成功次数 $X$ 的分布:
$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, ..., n
$$

记作 $X \sim B(n, p)$。

**关键元素**:
- $\binom{n}{k} = \frac{n!}{k!(n-k)!}$:组合数(选 $k$ 次成功的方式数)
- $p^k (1-p)^{n-k}$:特定序列的概率(假设独立)
- 乘起来 = $k$ 次成功的总概率

### 5.2 期望与方差

**期望**:
$$
\mathbb{E}[X] = np
$$

**方差**:
$$
\text{Var}(X) = np(1-p)
$$

**直觉**:
- $\mathbb{E}[X] = n \cdot p$:每次贡献 $p$ 期望,共 $n$ 次。
- $\text{Var}(X) = n \cdot p(1-p)$:每次贡献 $p(1-p)$ 方差,独立 ⇒ 求和。

**手算 1**:$B(10, 0.5)$,求 $\mathbb{E}[X]$, $\text{Var}(X)$, $\text{Std}(X)$。
- $\mathbb{E}[X] = 5$, $\text{Var}(X) = 2.5$, $\text{Std}(X) \approx 1.58$

**手算 2**:$B(100, 0.05)$,求 $P(X=5)$ 和 $P(X \leq 5)$。
- $P(X=5) = \binom{100}{5} 0.05^5 \cdot 0.95^{95}$
- $\binom{100}{5} = 75,287,520$
- $0.05^5 = 3.125 \times 10^{-7}$, $0.95^{95} \approx 0.0076$
- $P(X=5) \approx 75,287,520 \cdot 3.125 \times 10^{-7} \cdot 0.0076 \approx 0.180$

### 5.3 5 个手算例子

**例 1** (单点概率):$B(5, 0.6)$, $P(X=3)$。
- $\binom{5}{3} = 10$
- $P = 10 \cdot 0.6^3 \cdot 0.4^2 = 10 \cdot 0.216 \cdot 0.16 = 0.3456$

**例 2** (累积概率):$B(10, 0.5)$, $P(X \leq 5)$。
由对称性 = 0.5 + 0.5 × (中间项):
- $P(X=5) = \binom{10}{5}/2^{10} = 252/1024 = 0.2461$
- $P(X \leq 4) = (1 - 0.2461)/2 = 0.3769$
- $P(X \leq 5) = 0.3769 + 0.2461 = 0.6230$

**例 3** (期望):$B(20, 0.3)$, $\mathbb{E}[X]$, $\text{Var}(X)$。
- $\mathbb{E}[X] = 6$
- $\text{Var}(X) = 4.2$
- $\text{Std}(X) \approx 2.05$

**例 4** (A/B 实验样本量):某功能点击率基线 5%,想检测绝对提升 0.5%(5%→5.5%),问每组需要多少样本?
- $H_0$: $p_1 = p_2 = 0.05$, $H_1$: $p_1 - p_2 = 0.005$
- 合并 $p = (0.05 + 0.05)/2 = 0.05$, $\sigma = \sqrt{2 p(1-p)/n}$
- 显著性水平 $\alpha = 0.05$ (双侧),$z_{0.975} = 1.96$, 功效 $1-\beta = 0.8$, $z_{0.8} = 0.84$
- $n = (z_{\alpha/2} + z_{\beta})^2 \cdot 2p(1-p) / \Delta^2$
- $n = (1.96 + 0.84)^2 \cdot 2 \cdot 0.05 \cdot 0.95 / 0.005^2$
- $n = 7.84 \cdot 0.095 / 0.000025 \approx 29,792$

**例 5** (广告点击):一个广告展示 1000 次,每次点击率独立 = 1%,求点击数 = 10 的概率。
- $B(1000, 0.01)$, $P(X=10) = \binom{1000}{10} 0.01^{10} \cdot 0.99^{990}$
- 由 Poisson 近似 (5.5 节):$\lambda = np = 10$, $P \approx e^{-10} \cdot 10^{10}/10! \approx 0.125$

### 5.4 案例:A/B 实验的样本量计算

这是 Binomial 分布**最常用的工程场景**。

**问题**:要检测"A 比 B 好 1%",需要多少样本?

**公式推导**:
- $H_0$: $p_A = p_B$, $H_1$: $p_A - p_B = \Delta$
- 检验统计量: $Z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{2\hat{p}(1-\hat{p})/n}}$ (其中 $\hat{p}$ 是合并估计)
- 在 $H_1$ 下, $Z \sim \mathcal{N}(\Delta / \sqrt{2p(1-p)/n}, 1)$
- 要求 $P(Z > z_{\alpha/2} | H_1) = 1 - \beta$

解得:
$$
n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2 p (1-p)}{\Delta^2}
$$

**手算 7** (续例 4):$p = 0.05$, $\Delta = 0.005$, $\alpha = 0.05$, $1-\beta = 0.8$。
- $(1.96 + 0.84)^2 = 7.84$
- $2 \cdot 0.05 \cdot 0.95 = 0.095$
- $\Delta^2 = 2.5 \times 10^{-5}$
- $n = 7.84 \cdot 0.095 / 2.5 \times 10^{-5} = 29,792$ , **每组约 3 万样本**

**经验法则**:检测 1% 绝对差异,基线 CTR 5%,每组需要 **3-5 万样本**;流量大时跑 1 周,小时需跑 6 周。

### 5.5 Binomial 与 Poisson/正态的极限关系

**Poisson 极限**:当 $n \to \infty$, $p \to 0$, $np = \lambda$ 固定时:
$$
B(n, p) \to \text{Poisson}(\lambda)
$$

工程意义:**点击率很低的 A/B 实验用 Poisson 近似**。

**正态极限**:当 $n$ 很大, $p$ 不接近 0 或 1 时:
$$
\frac{X - np}{\sqrt{np(1-p)}} \to \mathcal{N}(0, 1)
$$

工程意义:**大样本下用正态近似做 Wald 区间**。

### 5.6 PyTorch 实现

```python
import torch
from torch.distributions import Binomial

binom = Binomial(10, torch.tensor(0.5))
samples = binom.sample((10000,))
print(f"mean ≈ 5: {samples.mean():.4f}")  # np = 5
print(f"var ≈ 2.5: {samples.var():.4f}")  # np(1-p) = 2.5
print("log P(X=0,5,10):", binom.log_prob(torch.tensor([0., 5., 10.])))
```

### 5.7 本节回顾

**核心收获**:
1. Binomial = $n$ 次独立 Bernoulli 试验的成功次数。
2. $\mathbb{E}[X] = np$, $\text{Var}(X) = np(1-p)$。
3. **A/B 样本量公式**是 Binomial 在显著性检验中的核心应用。
4. Binomial 在 $n \to \infty$ 时趋近正态,在 $p \to 0$ 时趋近 Poisson。
5. 低 CTR 场景下 Poisson 近似更精确,大样本下正态近似够用。

---

## 6. Poisson 分布

### 6.1 定义

**定义 6.1 (Poisson 分布)**:随机变量 $X$ 取非负整数值 $k = 0, 1, 2, ...$,概率为:
$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

记作 $X \sim \text{Poisson}(\lambda)$。参数 $\lambda > 0$ 是"速率"或"强度"。

### 6.2 期望与方差

**关键性质**:
$$
\mathbb{E}[X] = \text{Var}(X) = \lambda
$$

这是 Poisson 分布最独特的性质 , **均值等于方差**。

**手算证明** (期望):
$$
\mathbb{E}[X] = \sum_{k=0}^\infty k \cdot \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=1}^\infty \frac{\lambda^k}{(k-1)!} = \lambda e^{-\lambda} \sum_{j=0}^\infty \frac{\lambda^j}{j!} = \lambda
$$

### 6.3 极限关系:Binomial → Poisson

**定理 6.1 (Poisson 极限)**:若 $n \to \infty$, $p \to 0$, $np \to \lambda$,则:
$$
\binom{n}{k} p^k (1-p)^{n-k} \to \frac{\lambda^k e^{-\lambda}}{k!}
$$

**推导**:
- $\binom{n}{k} = \frac{n!}{k!(n-k)!} \approx \frac{n^k}{k!}$ (当 $n$ 大,$k$ 固定)
- $p^k = (\lambda/n)^k = \lambda^k / n^k$
- $(1-p)^{n-k} \approx e^{-np} = e^{-\lambda}$
- 乘积: $\frac{n^k}{k!} \cdot \frac{\lambda^k}{n^k} \cdot e^{-\lambda} = \frac{\lambda^k e^{-\lambda}}{k!}$ ✓

**工程意义**:
- **罕见事件**(p 很小)的二项分布可以用 Poisson 近似。
- 推荐系统的"点击事件" = $n$ 大(展示数), $p$ 小(点击率), 用 Poisson 合适。

### 6.4 5 个手算例子

**例 1** (基本概率):$\text{Poisson}(3)$, $P(X=0), P(X=3)$。
- $P(X=0) = e^{-3} \cdot 3^0 / 0! = e^{-3} \approx 0.0498$
- $P(X=3) = e^{-3} \cdot 27 / 6 \approx 0.224$

**例 2** (累积概率):$\text{Poisson}(2)$, $P(X \leq 2)$。
- $P(0) = e^{-2} = 0.135$
- $P(1) = 2 e^{-2} = 0.271$
- $P(2) = 2 e^{-2} = 0.271$
- 总和: $0.677$

**例 3** (期望方差):$\text{Poisson}(5)$, $\mathbb{E}[X]$, $\text{Var}(X)$, $P(X=5)$。
- $\mathbb{E} = \text{Var} = 5$
- $P(5) = e^{-5} \cdot 5^5 / 120 = e^{-5} \cdot 3125 / 120 \approx 0.175$

**例 4** (罕见事件):某服务器每天平均收到 0.5 次告警,求一天内恰好 1 次的概率。
- $\lambda = 0.5$, $P(X=1) = 0.5 \cdot e^{-0.5} \approx 0.303$

**例 5** (求和):$X_1 \sim \text{Poisson}(2), X_2 \sim \text{Poisson}(3)$, 独立, $Y = X_1 + X_2$ 的分布?
- **关键性质**:独立 Poisson 之和还是 Poisson,率相加。
- $Y \sim \text{Poisson}(5)$。

### 6.5 案例:推荐系统点击事件的稀有事件建模

**场景**:用户 1 天访问 100 个推荐位,平均每个位点击率 0.001,问一天内点击数分布。

- 朴素二项: $B(100, 0.001)$, $\mathbb{E} = 0.1$
- Poisson 近似: $\text{Poisson}(0.1)$, $P(X=0) = e^{-0.1} \approx 0.905$ (大部分时间不点击)

**为什么用 Poisson?** 因为 $p = 0.001$ 极小, $n = 100$ 中等, Poisson 近似极好;且 Poisson 让计算简化(无组合数 $\binom{n}{k}$)。

**进一步扩展**:**Non-homogeneous Poisson Process (NHPP)** 用于建模"点击率随时间变化"的情况。
$$
\Lambda(t) = \int_0^t \lambda(s) ds, \quad P(X(t) = k) = \frac{\Lambda(t)^k e^{-\Lambda(t)}}{k!}
$$

工程上用 NHPP 建模"用户活跃度"和"内容热度衰减"。

### 6.6 PyTorch 实现

```python
import torch
from torch.distributions import Poisson

poisson = Poisson(torch.tensor(3.5))
samples = poisson.sample((10000,))
print("mean ≈ var =", samples.mean().item())  # 都约等于 3.5
print("log P(X=0,1,3,5):", poisson.log_prob(torch.tensor([0.,1.,3.,5.])))
```

### 6.7 本节回顾

**核心收获**:
1. Poisson 分布 = 罕见事件的概率模型,参数 $\lambda$ 是速率。
2. **均值 = 方差 = $\lambda**(这是 Poisson 独有的性质)。
3. 是 Binomial 在 $n \to \infty, p \to 0, np = \lambda$ 下的极限。
4. 推荐系统点击、服务器告警、自然语言词频等都用 Poisson。
5. 独立 Poisson 之和仍是 Poisson,率相加 , 这个性质让"组合"很容易。

---

## 7. 指数分布

### 7.1 定义

**定义 7.1 (指数分布)**:连续随机变量 $X \geq 0$, PDF 为:
$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

记作 $X \sim \text{Exp}(\lambda)$。参数 $\lambda > 0$ 是速率。

**CDF**:
$$
F(x) = P(X \leq x) = 1 - e^{-\lambda x}
$$

**手算 1** (验证归一化):$\int_0^\infty \lambda e^{-\lambda x} dx = [-e^{-\lambda x}]_0^\infty = 0 - (-1) = 1$ ✓

### 7.2 期望与方差

**期望**:
$$
\mathbb{E}[X] = 1/\lambda
$$

**方差**:
$$
\text{Var}(X) = 1/\lambda^2
$$

**手算 2** (期望):$\mathbb{E}[X] = \int_0^\infty x \lambda e^{-\lambda x} dx$。
用分部积分: $u = x$, $dv = \lambda e^{-\lambda x} dx$, $du = dx$, $v = -e^{-\lambda x}$:
$$
\mathbb{E}[X] = [-x e^{-\lambda x}]_0^\infty + \int_0^\infty e^{-\lambda x} dx = 0 + 1/\lambda = 1/\lambda
$$

**直觉**:$\lambda$ 越大,事件发生越频繁,等待时间越短 ($\mathbb{E}$ 越小)。

### 7.3 无记忆性

**定理 7.1 (无记忆性)**:指数分布是**唯一具有无记忆性的连续分布**:
$$
P(X > s + t | X > s) = P(X > t)
$$

**直觉**:如果已经等了 $s$ 秒,再等 $t$ 秒的条件概率等于从头等 $t$ 秒的概率 , **过去不影响未来**。

**推导**:
$$
P(X > s+t | X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
$$

**手算 3**:某电子元件寿命 $\sim \text{Exp}(0.01)$ (平均寿命 100 小时)。已用 50 小时,再工作 100 小时的概率?
$P(X > 150 | X > 50) = P(X > 100) = e^{-1} \approx 0.368$ , 与"全新元件工作 100 小时"概率一样。

### 7.4 5 个手算例子

**例 1** (基本概率):$\text{Exp}(0.5)$, $P(X > 2)$。
- $P(X > 2) = e^{-0.5 \cdot 2} = e^{-1} \approx 0.368$

**例 2** (中位数):求 $\text{Exp}(\lambda)$ 的中位数 $m$。
- $P(X \leq m) = 0.5 \Rightarrow 1 - e^{-\lambda m} = 0.5 \Rightarrow m = \ln 2 / \lambda \approx 0.693/\lambda$
- 中位数 < 均值 ($0.693 < 1$),因为右偏。

**例 3** (分位数):$\text{Exp}(1)$, 求 95% 分位数 $x_{0.95}$。
- $1 - e^{-x} = 0.95 \Rightarrow x = -\ln(0.05) \approx 3.0$

**例 4** (条件):$\text{Exp}(2)$, $P(0.5 < X < 1.5)$。
- $= e^{-2 \cdot 0.5} - e^{-2 \cdot 1.5} = e^{-1} - e^{-3} \approx 0.368 - 0.050 = 0.318$

**例 5** (排队论):餐厅平均每 5 分钟来一桌 ($\lambda = 0.2$)。求下一桌在 10 分钟内来的概率。
- $X \sim \text{Exp}(0.2)$, $P(X \leq 10) = 1 - e^{-2} \approx 0.865$

### 7.5 案例:用户在网站停留时间的建模

**问题**:用户在某新闻网站停留时间 $T$,建模 $T$ 的分布。

**候选模型**:
1. **指数分布**:$\text{Exp}(\lambda)$,简单,但假设"无记忆" , 不太合理(用户不会"忘记"已读内容)。
2. **对数正态**:$\log T \sim \mathcal{N}(\mu, \sigma^2)$,更符合实际(右偏,且有"模式")。
3. **Weibull**:推广指数,允许"老化"效应(失效率随时间变化)。

**实际工程选择**:对数正态或 Weibull 更常用。指数作为"baseline 简单模型"。

**为什么指数依然重要?**
- 排队论(M/M/1 队列)的基础。
- 可靠性理论(电子元件寿命)。
- 泊松过程的时间间隔 , 本身就是指数。

### 7.6 PyTorch 实现

```python
import torch
from torch.distributions import Exponential

exp = Exponential(torch.tensor(0.5))
samples = exp.sample((10000,))
print("mean ≈ 2.0 (1/λ), var ≈ 4.0 (1/λ²)")
print("log densities:", exp.log_prob(torch.tensor([0.5, 1.0, 2.0])))
# 验证无记忆性: P(X>s+t|X>s) = P(X>t)
# 实现: 用 log_survival_function 比较两个值
```

### 7.7 本节回顾

**核心收获**:
1. 指数分布 = 连续版本的"等待时间",参数 $\lambda$ 是速率。
2. $\mathbb{E} = 1/\lambda$, $\text{Var} = 1/\lambda^2$。
3. **无记忆性是指数分布独有的** , 这是它在排队论和可靠性中的核心地位。
4. 在 ML 中常作为"停留时间"的简化模型,实际更常用对数正态或 Weibull。
5. 指数分布是 Gamma 分布在 $\alpha = 1$ 时的特例(后面会讲)。

---

## 8. Gamma 分布

### 8.1 Gamma 函数

**定义 8.1 (Gamma 函数)**:
$$
\Gamma(s) = \int_0^\infty t^{s-1} e^{-t} dt, \quad s > 0
$$

**关键性质**:
- $\Gamma(s+1) = s \Gamma(s)$ (递推)
- $\Gamma(1) = 1$ (基本情形)
- $\Gamma(n) = (n-1)!$ 对正整数 $n$
- $\Gamma(1/2) = \sqrt{\pi}$
- $\Gamma(s) \to \infty$ 当 $s \to 0^+$ (爆炸)

**手算 1**:$\Gamma(5) = 4! = 24$。
$\Gamma(7.5) = 6.5 \cdot 5.5 \cdot 4.5 \cdot 3.5 \cdot 2.5 \cdot 1.5 \cdot 0.5 \cdot \Gamma(0.5) = 6.5 \cdot 5.5 \cdot ... \cdot 0.5 \cdot \sqrt{\pi} \approx 1351.5$。

### 8.2 Gamma 分布定义

**定义 8.2 (Gamma 分布)**:
$$
f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

记作 $X \sim \text{Gamma}(\alpha, \beta)$。两个参数:
- $\alpha$ (shape, 形状):控制分布形状
- $\beta$ (rate, 速率):控制尺度

**注意**:有些教材用"尺度"参数 $\theta = 1/\beta$:
$$
f(x; \alpha, \theta) = \frac{x^{\alpha-1} e^{-x/\theta}}{\Gamma(\alpha) \theta^\alpha}
$$

工程中两种都常见,要看清定义再算!

### 8.3 期望与方差

**期望**:
$$
\mathbb{E}[X] = \frac{\alpha}{\beta}
$$

**方差**:
$$
\text{Var}(X) = \frac{\alpha}{\beta^2}
$$

**手算 2**:$\text{Gamma}(\alpha=2, \beta=1)$,求 $\mathbb{E}$, $\text{Var}$,众数(mode)。
- $\mathbb{E} = 2/1 = 2$
- $\text{Var} = 2/1 = 2$, $\text{Std} \approx 1.41$
- 众数: $(\alpha-1)/\beta = 1$ (当 $\alpha \geq 1$)

### 8.4 与其他分布的关系

**特例**:
- $\alpha = 1$: $f = \beta e^{-\beta x}$ → **指数分布** $\text{Exp}(\beta)$
- $\alpha = n/2, \beta = 1/2$: → **卡方分布** $\chi^2(n)$ (统计学重要)
- $\alpha \to \infty, \beta \to \infty$ 同时保持 $\alpha/\beta = \mu$ 固定: → **正态分布** $\mathcal{N}(\mu, \mu/\beta)$

**手算 3** (指数特例):$\text{Gamma}(1, \beta) = \text{Exp}(\beta)$。
- 验证: $f = \beta^1 / \Gamma(1) \cdot x^{1-1} e^{-\beta x} = \beta e^{-\beta x}$ ✓

**手算 4** (卡方特例):$\chi^2(4) = \text{Gamma}(\alpha=2, \beta=1/2)$。
- $\mathbb{E} = 2/(1/2) = 4 = $ 自由度 ✓
- $\text{Var} = 2/(1/4) = 8 = 2 \cdot 4 = 2k$ ✓

### 8.5 5 个手算例子

**例 1** (归一化验证):$\text{Gamma}(2, 1)$,验证 $\int_0^\infty f(x) dx = 1$。
$$
\int_0^\infty \frac{1}{\Gamma(2)} x e^{-x} dx = \int_0^\infty x e^{-x} dx = \Gamma(2) / \Gamma(2) = 1 \quad \checkmark
$$

**例 2** (众数):$\text{Gamma}(\alpha, \beta)$ 的众数 (当 $\alpha > 1$):
- 求导令为 0: $\frac{d}{dx}[x^{\alpha-1} e^{-\beta x}] = 0$
- $(\alpha-1) x^{\alpha-2} e^{-\beta x} - \beta x^{\alpha-1} e^{-\beta x} = 0$
- $\Rightarrow x = (\alpha-1)/\beta$
- 对 $\alpha = 3, \beta = 1$: 众数 = 2

**例 3** (均值):$\text{Gamma}(5, 2)$, $\mathbb{E} = 2.5$, $\text{Var} = 1.25$。

**例 4** (积分):$\text{Gamma}(\alpha, \beta)$,求 $P(X \leq \mathbb{E}[X])$。
没有闭式解,用数值或下/上不完全 Gamma 函数。

**例 5** (Beta 函数预备):$\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta) = B(\alpha, \beta)$ (Beta 函数)。后面 Beta 分布会用到。

### 8.6 案例:贝叶斯推断中作为 Poisson rate 的先验

**场景**:某网站平均每天 100 次故障 ($\lambda = 100$)。你不知道真实 $\lambda$,只知道历史观察。

**建模**:
- 似然:$X | \lambda \sim \text{Poisson}(\lambda)$, 观察 $n$ 天, 每天故障数 $x_1, ..., x_n$。
- 先验:$\lambda \sim \text{Gamma}(\alpha, \beta)$ (经验:故障率应平滑且为正)
- 后验:由"共轭性", $\lambda | x \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)$

**直觉**:
- 后验的形状参数 $\alpha' = \alpha + \sum x_i$ = 先验"伪计数" + 实际计数。
- 后验的速率参数 $\beta' = \beta + n$ = 先验"伪观测数" + 实际天数。
- 后验均值 $\mathbb{E}[\lambda | x] = (\alpha + \sum x_i)/(\beta + n)$,在"先验均值"和"样本均值"之间。

**手算 6**:$\alpha = 10, \beta = 0.1$ (先验均值 100),观察 5 天故障数 $(80, 110, 95, 100, 90)$, 求后验。
- $\alpha' = 10 + (80+110+95+100+90) = 10 + 475 = 485$
- $\beta' = 0.1 + 5 = 5.1$
- 后验均值 = $485/5.1 \approx 95.1$
- 后验方差 = $485/5.1^2 \approx 18.6$, $\text{Std} \approx 4.3$
- 95% 置信区间近似 (正态近似):$95.1 \pm 1.96 \cdot 4.3 = (86.7, 103.5)$

这个例子在第 6 章会展开讲"贝叶斯估计"。

### 8.7 PyTorch 实现

```python
import torch
from torch.distributions import Gamma

gamma = Gamma(torch.tensor(2.0), torch.tensor(1.0))
samples = gamma.sample((10000,))
print(f"mean ≈ 2: {samples.mean():.4f}")  # α/β = 2
print(f"var ≈ 2: {samples.var():.4f}")    # α/β² = 2
print("log densities:", gamma.log_prob(torch.tensor([0.5, 1.0, 2.0])))
```

### 8.8 本节回顾

**核心收获**:
1. Gamma 函数 $\Gamma(s)$ 是阶乘的实数推广。
2. Gamma 分布有两个参数:形状 $\alpha$ + 速率 $\beta$。
3. $\alpha = 1$ 时退化为指数,$\alpha = n/2, \beta = 1/2$ 时是卡方分布。
4. **Gamma 是 Poisson rate 的共轭先验** , 这让贝叶斯推断有闭式解。
5. 后验更新公式非常简洁: $\alpha' = \alpha + \sum x_i$, $\beta' = \beta + n$。

---

## 9. Beta 分布 (重点)

### 9.1 Beta 函数

**定义 9.1 (Beta 函数)**:
$$
B(\alpha, \beta) = \int_0^1 t^{\alpha-1} (1-t)^{\beta-1} dt = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}
$$

**性质**:
- $B(\alpha, \beta) = B(\beta, \alpha)$ (对称)
- $B(1, 1) = 1$
- $B(\alpha, 1) = 1/\alpha$
- $B(1/2, 1/2) = \pi$

### 9.2 Beta 分布定义

**定义 9.2 (Beta 分布)**:随机变量 $X \in [0, 1]$, PDF 为:
$$
f(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1
$$

记作 $X \sim \text{Beta}(\alpha, \beta)$。

**关键性质**:
- **定义域 [0, 1]** , 完美匹配"概率"!
- 两个参数 $\alpha, \beta > 0$ 控制形状。
- 对称性: $\text{Beta}(\alpha, \beta)$ 与 $\text{Beta}(\beta, \alpha)$ 关于 $x = 0.5$ 对称。

### 9.3 期望与方差

**期望**:
$$
\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}
$$

**方差**:
$$
\text{Var}(X) = \frac{\alpha \beta}{(\alpha+\beta)^2 (\alpha+\beta+1)}
$$

**众数** (mode, 当 $\alpha, \beta > 1$):
$$
\text{mode} = \frac{\alpha - 1}{\alpha + \beta - 2}
$$

**手算 1**:$\text{Beta}(2, 5)$, 求 $\mathbb{E}$, $\text{Var}$, mode。
- $\mathbb{E} = 2/7 \approx 0.286$
- $\text{Var} = 10 / (7^2 \cdot 8) = 10/392 \approx 0.0255$
- $\text{Std} \approx 0.160$
- mode = $1/5 = 0.2$

### 9.4 形状字典

**ASCII 可视化 3:Beta 分布在不同 $(\alpha, \beta)$ 下的形状**

```text
f(x) (横轴 x ∈ [0,1], 所有形状关于 α=β 时对称)

α=β=1 (均匀):    α=2,β=5 (右偏):   α=5,β=2 (左偏):   α=β=5 (钟形):
↑                ↑                  ↑                  ↑
│┌────────────┐  │      ╱╲          │            ╱╲ ╱  │       ╱╲
││            │  │     ╱  ╲___      │         ╱╱  ╲    │      ╱  ╲
││            │  │    ╱       ╲___  │     ___╱        │     ╱    ╲
└┴────────────┴──┴──┴──────────╲────┴─╱───┴───────────┴──┴──┴──╱──╲→
 0              1    0   .5      1    0                1   0 .5   1

α=β=0.5 (U 形,集中在两端):
↑
│╲              ╱
│ ╲____________╱
└─┴────────────┴──→ x
0  .5         1
```

**形状规则**:
- $\alpha = \beta$:对称
- $\alpha > \beta$:左偏(集中在 1)
- $\alpha < \beta$:右偏(集中在 0)
- $\alpha = \beta = 1$:均匀分布
- $\alpha, \beta < 1$:U 形(集中在两端)

### 9.5 5 个手算例子

**例 1** (均匀特例):$\text{Beta}(1, 1) = \text{Uniform}(0, 1)$。
- $f = x^0 (1-x)^0 / B(1,1) = 1/1 = 1$,在 $[0,1]$ 上 ✓
- $\mathbb{E} = 1/2$, $\text{Var} = 1/12 \approx 0.083$

**例 2** (退化特例):$\alpha = 1, \beta \to \infty$ → $X \to 0$;$\alpha \to \infty, \beta = 1$ → $X \to 1$。

**例 3** (对称):$\text{Beta}(3, 3)$。
- $\mathbb{E} = 0.5$
- $\text{Var} = 9 / (36 \cdot 7) = 1/28 \approx 0.0357$

**例 4** (众数):$\text{Beta}(4, 2)$, mode = $3/4 = 0.75$。

**例 5** (累积):$\text{Beta}(2, 2)$ 的 CDF 在 $x = 0.5$ 处:
$$
F(0.5) = \int_0^{0.5} 6 x (1-x) dx = 6 [x^2/2 - x^3/3]_0^{0.5} = 6 (0.125 - 0.0417) = 0.5
$$

(对称分布,中位数也是 0.5)

### 9.6 案例:推荐系统的 CTR 不确定性估计

**问题**:一个新物品的 CTR 是多少?只有少量曝光数据(假设 100 次曝光, 5 次点击),如何给 CTR 估计 + 不确定性?

**频率派** (简单):$\hat{p} = 5/100 = 0.05$,但**没有不确定性**。

**贝叶斯 + Beta 先验**:
- 先验:假设对 CTR 一无所知,用 $\text{Beta}(1, 1)$ (均匀先验)。
- 似然:$5$ 次成功, $95$ 次失败, 似然 $\propto p^5 (1-p)^{95}$。
- 后验:由共轭性, $p | \text{data} \sim \text{Beta}(1+5, 1+95) = \text{Beta}(6, 96)$。
- 后验均值:$\mathbb{E}[p | \text{data}] = 6/102 \approx 0.0588$
- 后验方差:$\text{Var} = 6 \cdot 96 / (102^2 \cdot 103) \approx 0.000522$
- 后验 Std $\approx 0.0228$

**95% 可信区间** (用 Beta CDF 或正态近似):
- $0.0588 \pm 1.96 \cdot 0.0228 = (0.0141, 0.1036)$

**新物品 CTR 范围大**:点估计 5.9%,但 95% 可信区间 [1.4%, 10.4%],Beta 分布自然给出不确定性。

### 9.7 案例:Thompson Sampling 的核心

**问题**:多臂老虎机 ($K$ 个广告位,每个有未知 CTR $p_k$)。每次展示哪个广告?

**Thompson Sampling 算法**:
```text
对每个广告 k:
    1. 从后验 Beta(α_k, β_k) 采样一个"虚拟 CTR" θ_k
    2. 选 θ_k 最大的广告展示
    3. 观察结果(点击/不点击),更新后验
```

**为什么有效?**
- 后验分布反映不确定性。
- 早期 (数据少):后验宽 → 探索 (可能选到非最优但高不确定性)。
- 后期 (数据多):后验窄 → 利用 (几乎确定选最优)。
- 这就是**探索-利用权衡**的自动实现!

**ASCII 可视化 4:Thompson Sampling 的探索-利用权衡**

```text
时间 T=10 (数据少):                时间 T=1000 (数据多):

  p ↑                               p ↑
1.0 ┤                              1.0 ┤            ╱╲
    │                               │             ╱  ╲ (广告 A)
    │  宽后验:                      │           ╱   ╲___
    │  广告 A、B、C 都有             │          ╱         ╲___
0.5 ┤  大概率被采样到                0.5 ┤        ╱           ╲___
    │                                │       ╱  广告 B 后验(中等)
    │                                │      ╱
    │                                │     ╱
0.0 ┤___╱╲__________________        0.0 ┤___╱___________________
    └──────────────────────→         └──────────────────────→
    0       0.5      1                0       0.5      1

  探索为主                            利用为主:几乎确定选 A
```

**PyTorch 代码** (完整版见第 11 节 实验 4)。

### 9.8 与其他分布的关系

**与 Bernoulli 共轭**:
- 似然:$X | p \sim \text{Bernoulli}(p)$
- 先验:$p \sim \text{Beta}(\alpha, \beta)$
- 后验:$p | X \sim \text{Beta}(\alpha + x, \beta + 1 - x)$ (更新公式极简)

**与 Binomial 共轭**:
- 似然:$X | p \sim \text{Bin}(n, p)$
- 先验:$p \sim \text{Beta}(\alpha, \beta)$
- 后验:$p | X \sim \text{Beta}(\alpha + \sum x_i, \beta + n - \sum x_i)$

这就是第 7 章"共轭先验"的核心例子之一。

**与 Gamma 的关系**:
- $\text{Beta}(\alpha, \beta)$ 的 PDF 中,$x^{\alpha-1}(1-x)^{\beta-1}$ 形式 → 与 Gamma 分布的 $x^{\alpha-1} e^{-\beta x}$ 形式相似。
- Beta 是"概率"的分布(定义域 [0,1]),Gamma 是"速率"的分布(定义域 [0, ∞))。

### 9.9 PyTorch 实现

```python
import torch
from torch.distributions import Beta

# α=2, β=5
alpha = torch.tensor(2.0)
beta = torch.tensor(5.0)
beta_dist = Beta(alpha, beta)
samples = beta_dist.sample((10000,))
print("Sample mean:", samples.mean().item())  # ≈ 0.286
print("Sample var:", samples.var().item())    # ≈ 0.0255

# log_prob
log_p = beta_dist.log_prob(torch.tensor([0.1, 0.3, 0.5, 0.7]))
print("log densities:", log_p)

# Thompson sampling with posterior updates
K = 3
alpha = torch.ones(K)
beta = torch.ones(K)
for t in range(100):
    theta = Beta(alpha, beta).sample()
    chosen = theta.argmax().item()
    reward = torch.bernoulli(torch.tensor(0.05)).item()  # 真 CTR 假设
    if reward:
        alpha[chosen] += 1
    else:
        beta[chosen] += 1
print(f"After 100 steps: α={alpha.tolist()}, β={beta.tolist()}")
```

### 9.10 本节回顾

**核心收获**:
1. Beta 分布定义在 [0, 1],适合建模"概率的概率"。
2. 两个参数 $\alpha, \beta$ 控制形状,$(\alpha, \beta)$ 越大,后验越集中在均值附近。
3. **Beta 是 Bernoulli/Binomial 的共轭先验** , 后验更新极简: $\alpha' = \alpha + \text{成功数}$, $\beta' = \beta + \text{失败数}$。
4. **Thompson Sampling 的核心**就是从 Beta 后验采样做决策。
5. CTR 不确定性估计就是 Beta 后验的应用。

---

## 10. 学生 t 分布

### 10.1 定义

**定义 10.1 (学生 t 分布)**:连续随机变量 $X$ 的 PDF 为:
$$
f(x; \nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}
$$

参数 $\nu > 0$ 是**自由度 (degrees of freedom)**。记作 $X \sim t(\nu)$。

### 10.2 与正态的关系

**关键性质**:
- $\nu = 1$: **Cauchy 分布** (重尾,期望不存在)
- $\nu \to \infty$: **标准正态** $\mathcal{N}(0, 1)$
- 有限 $\nu$: 钟形但**尾部比正态重**(heavier tails)

**直觉**:自由度小 → 数据少 → 不确定性大 → 尾部"厚"(给异常值更大权重)。

**ASCII 可视化 5:学生 t vs 正态 (尾部对比)**

```text
f(x)
   ↑
0.4 ┤            ╱╲
0.3 ┤           ╱ N(0,1)╲
0.2 ┤         ╱            ╲___
0.1 ┤       ╱  t(ν=5)         ╲___  ← 重尾:t 比 N 高
0.0 ┤_____╱_______________________╲___→ x
   -4  -2   0   2   4   6   8

   观察:中心/肩部 t 略低于 N,尾部 t 远高于 N (重尾!)
```

**手算 1**:$t(5)$ 在 $x = 3$ 处的密度 vs $\mathcal{N}(0, 1)$ 在 $x = 3$。
- $\mathcal{N}(0,1)$: $\phi(3) = (1/\sqrt{2\pi}) e^{-4.5} \approx 0.0044$
- $t(5)$: 用公式,$f(3) \approx 0.0113$ (大约是 $\mathcal{N}$ 的 2.5 倍)

这意味着:**t 分布给"远离均值"的点更高概率** , 这就是它对异常值更"宽容"的根源。

### 10.3 在 A/B 实验中的应用

**场景**:小样本 A/B 实验 ($n < 30$),样本方差未知,要用 $t$ 检验。

**两样本 t 检验**:
- $H_0$: $\mu_A = \mu_B$
- 检验统计量: $t = \frac{\bar{X}_A - \bar{X}_B}{s_p \sqrt{1/n_A + 1/n_B}}$ ,其中 $s_p$ 是合并标准差
- 在 $H_0$ 下, $t \sim t(n_A + n_B - 2)$

**手算 2**:$n_A = 10, n_B = 12$, $\bar{X}_A = 100, \bar{X}_B = 95$, $s_A = 8, s_B = 7$。
- 合并方差: $s_p^2 = (9 \cdot 64 + 11 \cdot 49) / 20 = (576 + 539) / 20 = 55.75$, $s_p \approx 7.47$
- $t = (100 - 95) / (7.47 \sqrt{1/10 + 1/12}) = 5 / (7.47 \cdot 0.436) \approx 5 / 3.26 \approx 1.53$
- 自由度 = 20, 查表 $t_{0.05, 20} = 1.725$ (双侧)
- $|t| = 1.53 < 1.725$,**不显著**。

### 10.4 5 个手算例子

**例 1** (极限):$t(\infty) = \mathcal{N}(0, 1)$。
- $\nu = 100$ 已经非常接近正态了。

**例 2** (Cauchy):$t(1) = $ Cauchy 分布。
- $f(x) = 1/[\pi (1 + x^2)]$,期望不存在(因为 $\int |x| f(x) dx = \infty$)。

**例 3** (矩):$t(\nu)$ 的 $k$ 阶矩存在 iff $k < \nu$。
- $\nu = 5$ 时,$\mathbb{E}[X] = 0$ (奇数阶为 0),$\mathbb{E}[X^2] = \nu/(\nu-2) = 5/3$,$\mathbb{E}[X^4] = \infty$ (因为 $4 < \nu = 5$? 不,$\nu = 5$ 时 $4 < 5$,所以存在)。
  - 实际上 $\mathbb{E}[X^4] = 3\nu^2 / [(\nu-2)(\nu-4)]$,需要 $\nu > 4$,$\nu = 5$ 时 $\mathbb{E}[X^4] = 75/(3 \cdot 1) = 25$。

**例 4** (分位数):$t(10)$ 的 97.5% 分位数 ≈ 2.228 (vs $\mathcal{N}$ 的 1.96)。

**例 5** (置信区间):$n = 15$, $\bar{X} = 100$, $s = 10$, 95% CI?
- $t_{0.025, 14} = 2.145$
- CI = $100 \pm 2.145 \cdot 10/\sqrt{15} = 100 \pm 5.54 = (94.46, 105.54)$

### 10.5 工程意义

**何时用 t 分布 vs 正态**:
- $n > 30$: 用正态近似就够 (t 和 正态几乎一样)。
- $n \leq 30$ + 方差未知: 用 t 分布更准确。
- 数据有重尾/异常值: t 分布更鲁棒。

**PyTorch 默认**:`torch.distributions.StudentT` 默认 $\mu = 0$, $\sigma = 1$。

```python
import torch
from torch.distributions import StudentT

df = torch.tensor(5.0)  # 自由度
t_dist = StudentT(df)
samples = t_dist.sample((10000,))
print("Sample mean:", samples.mean().item())  # ≈ 0
print("Sample std:", samples.std().item())    # ≈ sqrt(5/3) ≈ 1.29 (vs 1.0 for N)
# 注意:t(5) 的 Std > 1,因为重尾
```

### 10.6 本节回顾

**核心收获**:
1. 学生 t 分布 = 钟形 + 重尾,参数 $\nu$ 是自由度。
2. $\nu \to \infty$ 时退化为正态,$\nu = 1$ 是 Cauchy 分布。
3. **小样本 A/B 实验的标准工具**: 当 $n < 30$ + 方差未知时,用 t 检验。
4. 重尾 → 对异常值更"宽容" , 这在金融风控、推荐系统异常检测中有用。

---

## 11. PyTorch 代码实现 (综合实验)

### 实验 1:绘制 8 大分布的 PDF/PMF

```python
import torch
import matplotlib.pyplot as plt
from torch.distributions import (
    Bernoulli, Categorical, Normal, Binomial,
    Poisson, Exponential, Gamma, Beta, StudentT
)

# (分布名, 构造, x 生成函数, 是否连续)  列表统一处理
specs = [
    ('Bernoulli(0.3)', lambda: Bernoulli(torch.tensor(0.3)),
     lambda d: torch.tensor([0., 1.]), False),
    ('Categorical',    lambda: Categorical(probs=torch.tensor([0.1,0.3,0.4,0.2])),
     lambda d: torch.arange(d.probs.numel()), False),
    ('Normal(0,1)',    lambda: Normal(0., 1.),
     lambda d: torch.linspace(-4, 4, 200), True),
    ('Binomial(10,0.5)',lambda: Binomial(10, torch.tensor(0.5)),
     lambda d: torch.arange(11), False),
    ('Poisson(3)',     lambda: Poisson(torch.tensor(3.)),
     lambda d: torch.arange(15), False),
    ('Exponential(1)', lambda: Exponential(torch.tensor(1.)),
     lambda d: torch.linspace(0, 5, 200), True),
    ('Gamma(2,1)',     lambda: Gamma(torch.tensor(2.), torch.tensor(1.)),
     lambda d: torch.linspace(0.01, 10, 200), True),
    ('Beta(2,5)',      lambda: Beta(torch.tensor(2.), torch.tensor(5.)),
     lambda d: torch.linspace(0.001, 0.999, 200), True),
    ('StudentT(5)',    lambda: StudentT(torch.tensor(5.)),
     lambda d: torch.linspace(-5, 5, 200), True),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, (name, mk, xfn, cont) in zip(axes.flat, specs):
    d = mk()
    xs = xfn(d)
    p = torch.exp(d.log_prob(xs))
    (ax.plot if cont else ax.bar)(xs.numpy(), p.numpy())
    ax.set_title(name)
plt.tight_layout()
plt.savefig('distributions_overview.png', dpi=100)
plt.show()
```

### 实验 2:中心极限定理 (CLT) 验证

**核心命题**:$n$ 个独立同分布随机变量(任意分布)之和 / 平均 → 正态。

```python
import torch

def clt_experiment(distribution_name, sample_fn, n_sum, n_experiments=10000):
    """CLT:对 n_sum 个独立样本求和,验证结果近似正态"""
    # 生成 n_experiments 组,每组 n_sum 个样本
    samples = torch.zeros(n_experiments)
    for _ in range(n_sum):
        samples += sample_fn(n_experiments)
    # 求平均(标准化)
    samples = samples / n_sum

    # 绘制直方图 + 标准正态 PDF 对比
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(samples.numpy(), bins=50, density=True, alpha=0.6, label='empirical')

    # 叠加标准正态
    xs = torch.linspace(-4, 4, 200)
    pdf = torch.exp(torch.distributions.Normal(0, 1).log_prob(xs))
    ax.plot(xs.numpy(), pdf.numpy(), 'r-', lw=2, label='N(0,1)')
    ax.set_title(f'CLT: {distribution_name}, n_sum={n_sum}')
    ax.legend()
    return samples

# 1. 均匀分布 → 正态
clt_experiment('Uniform(0,1)', lambda n: torch.rand(n), 30)

# 2. 指数分布 → 正态
clt_experiment('Exp(1)', lambda n: torch.distributions.Exponential(1).sample((n,)), 30)

# 3. Bernoulli → 正态
clt_experiment('Bernoulli(0.3)', lambda n: torch.bernoulli(torch.full((n,), 0.3)), 100)
```

**观察**:无论原始分布是什么,$n$ 个独立样本的平均都趋近正态,这就是**中心极限定理**。第 4 章会详细讲。

### 实验 3:torch.distributions 各种分布的采样和概率计算

```python
import torch
from torch.distributions import Normal, Beta, Gamma, Binomial, Categorical

# Normal 多维采样 + log_prob
norm = Normal(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 1.0, 1.5]))
print("Normal samples shape:", norm.sample((1000,)).shape)  # (1000, 3)

# Beta 多维
beta_dist = Beta(torch.tensor([1.0, 2.0, 5.0]), torch.tensor([1.0, 5.0, 2.0]))
print("Beta sample mean:", beta_dist.sample((1000,)).mean(dim=0))

# Binomial 期望 = n*p
binom = Binomial(20, torch.tensor(0.5))
print(f"Binomial(20, 0.5) sample mean ≈ 10: {binom.sample((1000,)).mean():.3f}")

# Categorical 频率 ≈ probs
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
counts = torch.bincount(Categorical(probs=probs).sample((1000,)), minlength=4)
print(f"Categorical counts ≈ [100,200,300,400]: {counts.tolist()}")
```

### 实验 4:Beta-Binomial 共轭 (在线 CTR 估计)

```python
import torch
from torch.distributions import Beta

class BetaBinomialConjugate:
    """Beta-Binomial 共轭:在线更新 CTR 估计"""
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha, self.beta = alpha, beta  # 先验伪计数
    def update(self, s, f):  # s=failures, f=successes
        self.alpha += s; self.beta += f
    @property
    def mean(self): return self.alpha / (self.alpha + self.beta)
    @property
    def std(self):
        a, b = self.alpha, self.beta
        return (a*b / ((a+b)**2 * (a+b+1))).sqrt()

# 模拟:1000 次曝光,真实 CTR = 5%
bb = BetaBinomialConjugate()
true_ctr = 0.05
rewards = torch.bernoulli(torch.tensor(true_ctr).expand(1000))
for i, r in enumerate(rewards):
    bb.update(int(r), 1 - int(r))
print(f"True CTR={true_ctr}, posterior mean={bb.mean:.4f}, std={bb.std:.4f}")

# Thompson Sampling:K 臂老虎机
K, n_steps, true_ctr = 3, 1000, torch.tensor([0.05, 0.04, 0.03])
alpha, beta = torch.ones(K), torch.ones(K)
for _ in range(n_steps):
    chosen = Beta(alpha, beta).sample().argmax().item()
    r = torch.bernoulli(true_ctr[chosen]).item()
    alpha[chosen] += r; beta[chosen] += 1 - r
print(f"Estimated CTR: {(alpha/(alpha+beta)).tolist()}")
print(f"True CTR:      {true_ctr.tolist()}")
```

### 实验 5:Gamma-Poisson 共轭 (故障率估计)

```python
import torch
from torch.distributions import Gamma

# 先验 Gamma(α=2, β=0.5), 均值 = 4 故障/天
prior_a, prior_b = 2.0, 0.5
failures = torch.tensor([3, 5, 4, 6, 2, 4, 5, 3, 4, 5])  # 10 天观察
# 共轭更新:α' = α + Σx, β' = β + n
post_a, post_b = prior_a + failures.sum(), prior_b + len(failures)
samples = Gamma(post_a, post_b).sample((10000,))
print(f"Posterior mean={samples.mean():.2f}, "
      f"95% CI=({samples.quantile(0.025):.2f}, {samples.quantile(0.975):.2f})")
```

---

## 12. 常见误区 + Q&A

### 12.1 常见误区 (8 条)

**误区 1: "正态分布最重要,所以所有数据都用正态建模"**

**错**。正态在自然界和 ML 中常见是因为 CLT,但**实际数据常有偏**:
- 用户消费金额:对数正态 (长尾)
- 网站停留时间:对数正态或 Weibull
- 点击事件:Poisson (稀疏)

**强制用正态建模 → 预测极值时误差大**。

**误区 2: "标准正态的最高点 = 1"**

**错**。$\phi(0) = 1/\sqrt{2\pi} \approx 0.3989$,**不是 1**。
正态密度函数的最高点是 $1/(\sigma\sqrt{2\pi})$,与 $\sigma$ 成反比。

**误区 3: "Bernoulli 只能二分类,K > 2 用多个二分类"**

**半错**。K > 2 时:
- 如果类别互斥 (一张图只能是一种动物):用 **Categorical + Softmax + CE**
- 如果类别不互斥 (一张图可以同时是"猫"和"哺乳动物"):用 **K 个独立 Bernoulli + K 个 Sigmoid + K 个 BCE** (multi-label)

**误区 4: "Poisson 分布只能建模'时间'事件"**

**错**。Poisson 建模的是"**稀有事件的次数**",不限于时间:
- 一本书每页的错别字数 (空间维度)
- 一片区域的陨石坑数 (空间)
- 一段代码的 bug 数 (抽象)

只要满足"独立、平均速率恒定、稀有",就适合 Poisson。

**误区 5: "Beta 分布是正态的近似"**

**完全错**。Beta 和正态完全不同:
- 定义域:Beta ∈ [0,1],正态 ∈ (-∞, ∞)
- 用途:Beta 建模"概率的概率",正态建模"实数的分布"
- 形状:Beta 有 U 形/钟形/单调,正态只有钟形

**Beta 是 Bernoulli 的共轭先验**,正态是连续数据的默认模型。

**误区 6: "Gamma 分布是 Beta 分布的推广"**

**部分对但容易误解**:
- Beta 定义域 [0,1],Gamma 定义域 [0, ∞)
- Beta 可以从 Gamma 推导:若 $X \sim \text{Gamma}(\alpha, 1)$, $Y \sim \text{Gamma}(\beta, 1)$ 独立,则 $X/(X+Y) \sim \text{Beta}(\alpha, \beta)$
- 但 Gamma 不是 Beta 的"自然推广" , 两者各有用途

**误区 7: "指数分布的'无记忆性'是常识,不需要假设"**

**错**。无记忆性是**指数独有的**强假设,不是"常识":
- 大部分物理过程**不是**无记忆的(灯泡老化、元件磨损)
- 大部分人类行为**不是**无记忆的(用户不会"忘记"已读内容)
- 排队论 (M/M/1) 和电子元件寿命才真的用指数

**实际工程**:停留时间、寿命数据,常用对数正态或 Weibull(它们有"老化"),不用指数。

**误区 8: "学生 t 分布只是小样本用的,大样本就用正态"**

**半对**。学生 t 在 $\nu \geq 30$ 时与正态几乎一样 (差异 < 1%),但当数据**真有重尾**时,无论样本大小都用 t 更鲁棒:
- 金融收益率 (重尾)
- 推荐系统的异常行为
- 极端天气数据

**鲁棒统计** 领域专门处理重尾,学生 t 是入门。

### 12.2 面试 Q&A (5 题,L7/L8 难度)

**Q1: "Bernoulli vs Categorical 的区别是什么?为什么 BCE 和 CE 不通用?"**

**A**:核心区别是**类别数**和**互斥性**:
- **Bernoulli** = K=2 的 Categorical 特例,标签 0 或 1,**互斥**
- **Categorical** = K ≥ 2 的多类分布,标签是 one-hot 向量,**互斥**

BCE 不能直接用于多分类的原因:
- BCE 是 $K$ 个独立二分类器,**假设各类独立**
- CE 是 $K$ 个**互斥**类别,用 Softmax 强制概率归一化
- 例子:图像分类(互斥)用 CE,文本分类多标签(可共存)用 K 个 BCE

如果硬用 BCE 做多分类,会出现:
- 梯度没有"互斥约束",各类独立 → 模型可能输出 $(0.8, 0.8)$ (两个类都"高置信")
- 训练不稳定、收敛慢

**追问**: "那 Softmax + CE 的梯度公式 $\partial L / \partial z = p - y$ 是怎么来的?"
- 答:链式法则,见 3.6 节。
- 关键步骤: $\partial L / \partial p_i = -y_i / p_i$ + $\partial p_i / \partial z_j = p_i (\delta_{ij} - p_j)$,乘起来化简得 $p_j - y_j$。

**Q2: "为什么正态分布在 ML 中如此常见?深度学习初始化、Adam 优化、扩散模型都用它,这是巧合吗?"**

**A**:不是巧合,三个独立的原因都指向正态:

1. **中心极限定理 (CLT)**:任何独立随机变量之和 → 正态。神经网络的权重初始化是多个随机量的"叠加"(例如 Xavier 推导),自然接近正态。
2. **最大熵**:给定均值和方差,正态是**熵最大的分布**。当我们只知道"均值和方差"时,正态是最"不假设"的分布,符合 ML 的"先验无知"原则。
3. **数学便利**:正态的线性组合仍是正态(线性保持性);正态的共轭先验是它自身;正态的 Fisher 信息量是常数;正态的 KL 散度有闭式解。这些性质让正态在推导和计算中"好处理"。

**追问**: "那为什么不用 Laplace 分布(也是最大熵)?"
- 答:Laplace 在给定均值和**平均绝对偏差**时熵最大。正态对应**方差**,L2 范数天然对应方差。所以当损失函数用 L2(MSE)时,正态最自然;当损失用 L1(MAE)时,Laplace 更合适。

**Q3: "Beta 分布 vs Dirichlet 分布的关系?Beta 能不能直接用于多分类?"**

**A**:Dirichlet 是 Beta 在多维的推广:
- Beta 定义域 [0,1],Dirichlet 定义域是单纯形 $\{(p_1, ..., p_K) : \sum p_i = 1, p_i \geq 0\}$
- Beta 是 K=2 的 Dirichlet 特例
- Beta 的参数 $(\alpha, \beta)$,Dirichlet 的参数 $(\alpha_1, ..., \alpha_K)$

**直接用 Beta 做多分类**:
- 朴素做法:K 个独立 Beta 分布 → 输出 K 个独立概率,不能保证 $\sum p_i = 1$
- 正确做法:用 **Dirichlet**,或对 K 个独立 Sigmoid 输出做 softmax 归一化

**工程应用**:
- Beta:二分类 CTR 不确定性、Thompson Sampling
- Dirichlet:主题模型 (LDA)、多分类 Bayesian 神经网络

**Q4: "为什么 Adam 用 EMA 假设梯度近似正态?如果梯度真不是正态会怎样?"**

**A**:Adam 的核心假设是 $g_t \approx \mathcal{N}(0, \sigma^2)$ (在窗口内 i.i.d.)。

**为什么这样假设?**
- Adam 的更新是 $\theta_{t+1} = \theta_t - \eta \hat{m}_t / \sqrt{\hat{v}_t}$
- 若 $g_t$ 真是正态,则 $\hat{m}_t / \sqrt{\hat{v}_t} \approx g_t / \sigma$ 是**标准化正态**,与梯度尺度无关
- 这让学习率 $\eta$ 不需要随梯度大小调整

**如果梯度不是正态会怎样?**
- **重尾** (金融、强化学习):$v_t$ 被异常值拉大 → 步长过小 → 收敛慢。**AdamW、Shampoo** 等改进
- **稀疏** (Embedding 梯度):大部分 $g_t = 0$,少数很大 → $v_t$ 估计偏低 → 步长过大 → 抖动。**SparseAdam** 只对非零元素更新状态
- **非平稳** (GAN、RL):梯度均值和方差随时间变 → EMA 假设失效 → 需要更小的 $\beta$

**深层修复**:LARS (Layer-wise Adaptive Rate Scaling) 和 LAMB 在大 batch 训练中更鲁棒,因为它们做了 layer 级别的归一化。

**Q5: "Gamma 分布在工业界有什么具体应用?除了作为 Poisson rate 的先验,还有别的吗?"**

**A**:Gamma 分布的应用远超 "Poisson 先验":

1. **贝叶斯 A/B 实验**:Beta-Binomial 是 CTR 的标准,但对**连续指标** (订单金额、停留时间),用 **Normal-Inverse-Gamma** (NIG) 先验。
2. **可靠性工程**:电子元件寿命服从 Weibull 分布,Weibull 是 Gamma 的推广。Gamma 是 Weibull 在形状参数为 1 时的特例。
3. **图像处理**:Gamma 变换用于图像亮度调整($\gamma$ 校正)。
4. **金融工程**:随机波动率模型中,波动率服从 Gamma 扩散过程。
5. **生成模型**:Variational Autoencoder (VAE) 的隐变量后验用 Gamma 分布作为变分族,有专门的 Gamma-VAE 论文 (2019)。
6. **排队论**:M/G/1 队列的等待时间分布涉及 Gamma。

**面试常见追问**:"那你做一个贝叶斯推断建模,会怎么选先验?"
- 答:看数据类型。**离散计数** (点击数) → Poisson 似然 + Gamma 先验 (共轭)。**二项比例** (CTR) → Binomial 似然 + Beta 先验。**连续正数** (金额) → LogNormal 或 Gamma 似然 + 对应先验。**实数** → Normal 似然 + Normal-Inverse-Gamma 先验。

---

## 13. 与其他数学工具的联系

### 13.1 与本文件夹其他章节的连接

| 本章概念 | 后续章节 | 具体联系 |
| :--- | :--- | :--- |
| 正态分布 | 第 4 章期望/方差 | $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$ 的具体计算 |
| Bernoulli / Categorical | 第 5 章贝叶斯定理 | 朴素贝叶斯分类器基于这些分布 |
| Beta-Binomial 共轭 | 第 6 章 MLE/MAP | MAP 估计的后验众数 |
| Gamma-Poisson 共轭 | 第 6 章贝叶斯估计 | 故障率估计的闭式解 |
| 所有分布 | 第 7 章指数族 | 正态/Bernoulli/Categorical/Poisson/Gamma/Beta 都是指数族 |
| 正态/Binomial | 第 8 章 A/B 实验 | t 检验、z 检验、样本量公式 |
| 多元正态 | 第 9 章 MCMC | HMC (Hamiltonian Monte Carlo) 在多元正态上采样 |

### 13.2 与其他数学文件夹的连接

| 本章概念 | 其他章节 | 跨文件夹引用 |
| :--- | :--- | :--- |
| 多元正态 + 协方差矩阵 | `../../../基础/数学基础/第02章-线性代数/第02章-05-二次型·正定性·条件数-为什么-Adam-比-SGD-稳.md` | 协方差矩阵是半正定矩阵 |
| 正态的二次型 (Mahalanobis) | `../../../基础/数学基础/第02章-线性代数/第02章-05-二次型·正定性·条件数-为什么-Adam-比-SGD-稳.md` | $(\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu)$ 是二次型 |
| Beta 分布的"概率的概率" | `../../../基础/数学基础/第05章-信息论/第05章-02-熵·联合熵·条件熵·链式法则.md` | Beta 分布的熵表达 |
| 交叉熵 = Categorical 似然 | `../../../基础/数学基础/第05章-信息论/第05章-05-交叉熵·困惑度·NLL.md` | BCE/CE 的信息论解读 |
| 指数族的对数配分函数 | `../../../基础/数学基础/第06章-最优化/第06章-07-分布式优化-同步SGD·Async·AllReduce·ZeRO.md` | 配分函数的梯度 = 充分统计量期望 |
| Thompson Sampling | `../第07章-系统思维/第07章-08-案例Exploration-Exploitation.md` | 老虎机算法的概率基础 |
| 正态的采样 | `../第07章-系统思维/第07章-10-案例Diffusion前向加噪反向去噪.md` | Diffusion 的反向过程是"去噪采样" |

### 13.3 学习的下一步

**读完本章,你的"概率论工具箱"里应该有**:
- 9 个分布的定义、PMF/PDF、期望、方差
- 5+ 手算例子的肌肉记忆
- "每个分布对应一个工程现象" 的直觉
- PyTorch `torch.distributions` 的基本使用
- 知道什么时候用哪个分布

**下一步建议**:
1. **第 4 章**:把这些分布的"数字特征" (期望/方差/协方差) 形式化,讲大数定律和中心极限定理 , 理解 CLT 后,你对"为什么正态如此常见"会有更深的答案。
2. **第 5 章**:把本章的 Bernoulli/Categorical/正态用到贝叶斯定理,推导朴素贝叶斯和贝叶斯神经网络。
3. **第 6 章**:用本章的 Beta/Gamma 共轭性讲 MLE/MAP/贝叶斯估计。
4. **第 7 章**:把所有分布统一到指数族框架,讲"为什么这些分布的 MLE 都有闭式解"。

### 13.4 一句话总结

**分布 = 数据的概率模型**。**Loss = 分布的负对数似然**。**优化 loss = 用数据估计分布参数**。

这就是深度学习的概率论骨架,本章是这块骨架的"血肉"。

---

> **本章版本**: v1.0
> **最后更新**: 2026-06-23
> **下一章**: [第 4 章:期望·方差·协方差·矩,从样本均值到 Embedding 统计特性](第03章-04-期望·方差·协方差·矩-从样本均值到-Embedding-统计特性.md)

---

## 14. 分布家族的"家谱":指数族与共轭关系 (300 行)

这一节把前面 13 节讲的分布放进一个统一的"家谱"里, 让工程师看到一个全局图。

### 14.1 指数族视角

几乎所有常见分布都属于**指数族**:
$$
p(x | \eta) = h(x) \exp(\eta^T T(x) - A(\eta))
$$
其中 $\eta$ 是自然参数, $T(x)$ 是充分统计量, $A(\eta)$ 是 log-partition function。

| 分布 | 自然参数 η | 充分统计量 T(x) |
| :--- | :--- | :--- |
| Bernoulli(p) | $\log(p/(1-p))$ (logit) | x |
| Categorical(p) | $\log p_k$ | $\mathbb{1}(x=k)$ |
| Normal(μ, σ²) | $(\mu/\sigma^2, -1/(2\sigma^2))$ | $(x, x^2)$ |
| Exponential(λ) | $-\lambda$ | x |
| Gamma(α, β) | $(-\beta, \alpha-1)$ (natural form) | $(\log x, x)$ |
| Beta(α, β) | $\eta = (\alpha-1, \beta-1)$ (as function of α, β) | $(\log x, \log(1-x))$ |
| Poisson(λ) | $\log \lambda$ | x |

**意义**: 指数族的好处是:
1. MLE 有闭式解 (充分统计量的均值)
2. 共轭先验易于构造
3. GLM (广义线性模型) 的理论基础

### 14.2 共轭先验对照表

| 似然 (数据分布) | 共轭先验 | 后验 |
| :--- | :--- | :--- |
| Bernoulli(p) | Beta(α, β) | Beta(α + k, β + n - k) |
| Binomial(n, p) | Beta(α, β) | Beta(α + k, β + n - k) |
| Categorical(p) | Dirichlet(α) | Dirichlet(α + counts) |
| Poisson(λ) | Gamma(α, β) | Gamma(α + Σx_i, β + n) |
| Exponential(λ) | Gamma(α, β) | Gamma(α + n, β + Σx_i) |
| Normal(μ, σ²) (σ 已知) | Normal(μ₀, σ₀²) | Normal(μ_n, σ_n²) |
| Normal(μ, σ²) (μ 已知) | InvGamma(α, β) | InvGamma(α + n/2, β + Σ(x-μ)²/2) |
| Gamma(α, β) (β 已知) | Gamma(α₀, β₀) | Gamma(α₀ + n, β₀ + Σx) |

**直观**: 先验和后验"同族", 让后验计算极其简单 (只是参数的更新)。

### 14.3 分布选择的"决策树"

```text
数据是?
├─ 离散
│  ├─ 二值 (yes/no)
│  │  ├─ 一次试验 → Bernoulli(p)
│  │  └─ n 次试验, 关心成功次数 → Binomial(n, p)
│  ├─ K 个互斥类别
│  │  └─ 一次抽样 → Categorical(p₁, ..., p_K)
│  └─ 计数 (0, 1, 2, ...)
│     ├─ 单位时间/空间内事件数, rate 固定 → Poisson(λ)
│     └─ n 次试验中成功次数 → Binomial(n, p) 或 HyperGeometric
├─ 连续
│  ├─ (0, 1) 区间
│  │  ├─ 集中于某值 (钟形) → Beta(α, β), α=β
│  │  ├─ 偏向 0 → Beta(α, β), α < β
│  │  └─ 偏向 1 → Beta(α, β), α > β
│  ├─ (0, +∞)
│  │  ├─ 无记忆 → Exponential(λ)
│  │  ├─ 有 shape, 单调 → Gamma(α, β)
│  │  └─ 重尾正数 → LogNormal(μ, σ²)
│  └─ (-∞, +∞)
│     ├─ 钟形 → Normal(μ, σ²)
│     ├─ 重尾, 小样本 → StudentT(df, μ, σ²)
│     └─ 多维 → MultivariateNormal(μ, Σ)
└─ 角度/周期性
   └─ von Mises(μ, κ) 或 WrappedCauchy
```

### 14.4 各分布的"工程场景匹配表"

| 业务场景 | 推荐分布 | 理由 |
| :--- | :--- | :--- |
| 用户点击 (二分类) | Bernoulli | 输出 {0, 1} |
| 商品推荐 (K 选 1) | Categorical | K 个互斥的 item |
| 用户停留时长 | LogNormal / Gamma | 正值, 重尾 |
| 广告 GMV | LogNormal | 长尾分布 |
| 用户评分 (1-5 星) | OrdinalLogistic / Beta (归一化) | 离散有序 |
| 异常流量 | LogNormal + 阈值 | 罕见事件 |
| 商品 CTR | Beta(α, β) | [0,1] 区间, 共轭方便 |
| 训练 loss 曲线 | Gamma / LogNormal | 正值, 长尾 |
| Embedding 分布 | MultivariateNormal | 多维连续 |
| 模型不确定性 | Dirichlet / Categorical | 概率分布 |
| A/B 实验的转化率 | Beta | [0,1] + 共轭 |
| 排队等待时间 | Exponential / Gamma | 等待时间 |

### 14.5 分布的"组合"艺术

工程上常常需要**组合分布**:

**1. 混合分布 (Mixture)**: $p(x) = \sum_k \pi_k p_k(x)$
```python
# 用户停留: 30% 短 + 70% 长
g_short = Gamma(2.0, 1.0)    # 平均 2 秒
g_long = Gamma(20.0, 0.5)    # 平均 40 秒
weights = [0.3, 0.7]
# 手动混合
samples = torch.cat([
    g_short.sample((3000,)),
    g_long.sample((7000,))
])
```

**2. 乘积分布 (Product, 假设独立)**: $p(x, y) = p_X(x) p_Y(y)$

**3. 边缘化 (Marginalization)**: $p(x) = \int p(x, y) dy$

**4. 条件分布 (Conditioning)**: $p(y | x) = p(x, y) / p(x)$

**5. 卷积 (Convolution)**: $X + Y$ 的分布 = $f_X * f_Y$

---

## 15. 各分布的"高级话题"与 PyTorch 实战 (400 行)

### 15.1 正态分布的"扩展家族"

正态分布有一大家族, 适用不同场景:

#### (a) 截断正态 (Truncated Normal)

$N(\mu, \sigma^2)$ 但限制在 $[a, b]$ 区间。强化学习的 bounded action space 常用。

```python
import torch
from torch.distributions import TruncatedNormal

# 强化学习: action ∈ [-1, 1]
tn = TruncatedNormal(loc=0, scale=0.5, low=-1, high=1)
samples = tn.sample((1000,))
print(f"采样范围: [{samples.min():.3f}, {samples.max():.3f}]")  # 在 [-1, 1] 内
```

#### (b) 对数正态 (LogNormal)

$Y = e^X$, $X \sim N(\mu, \sigma^2)$。$Y$ 是正的, 重尾。

```python
ln = torch.distributions.LogNormal(0.0, 1.0)
samples = ln.sample((10000,))
print(f"LogNormal 均值: {samples.mean():.2f}, 中位数: {samples.median():.2f}")
# 均值 > 中位数, 重尾
```

#### (c) 多元正态 (Multivariate Normal)

$N(\mu, \Sigma)$, $\mu \in \mathbb{R}^d$, $\Sigma \in \mathbb{R}^{d \times d}$ 半正定。

```python
mu = torch.zeros(3)
Sigma = torch.eye(3)  # 各维独立标准正态
mn = torch.distributions.MultivariateNormal(mu, Sigma)
samples = mn.sample((1000,))
print(f"采样 shape: {samples.shape}, 每维均值: {samples.mean(0)}")
```

#### (d) 学生 t 分布

Normal 的"重尾版本", df (自由度) 越小越重尾, df → ∞ 时退化为 Normal。

```python
st = torch.distributions.StudentT(df=3, loc=0, scale=1)
samples = st.sample((10000,))
print(f"StudentT(df=3) 的均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
# 标准差大于 Normal(0, 1), 重尾
```

### 15.2 Gamma 分布的"扩展家族"

Gamma 是正参数分布的"母分布":

#### (a) Gamma 与指数分布的关系

Exponential(λ) = Gamma(α=1, rate=λ)。即指数是 Gamma 的特例。

#### (b) Gamma 与 Poisson 的关系

若 $X \sim \text{Gamma}(\alpha, \beta)$, 则 $X$ 可以解释为"$\alpha$ 个独立指数事件的总等待时间", 或"在 Poisson 过程中观察到 $\alpha$ 个事件的等待时间"。

#### (c) 逆 Gamma (Inverse Gamma)

$Y = 1/X$, $X \sim \text{Gamma}$。常作正态方差的共轭先验。

```python
ig = torch.distributions.InverseGamma(concentration=3.0, rate=2.0)
samples = ig.sample((1000,))
print(f"InverseGamma 均值: {samples.mean():.4f}")  # rate/(concentration-1) = 1.0
```

#### (d) Wishart 分布

多维 Gamma 推广, 是多维正态协方差矩阵的共轭先验。

### 15.3 Beta 分布的"扩展家族"

#### (a) Dirichlet 分布

Beta 的多维推广, 定义在单纯形 $\Delta_K = \{p \in \mathbb{R}^K : p_i \ge 0, \sum p_i = 1\}$ 上。

```python
# K=3 个类别的 Dirichlet
dirichlet = torch.distributions.Dirichlet(concentration=torch.tensor([1.0, 2.0, 3.0]))
samples = dirichlet.sample((5,))
print(f"Dirichlet 样本:\n{samples}")
print(f"每行和: {samples.sum(1)}")  # 都 = 1, 因为是概率分布
```

**应用**: LDA 主题模型、Categorical 分布的共轭先验、神经网络的输出层 (softmax 输出的不确定性)。

#### (b) Beta-Binomial 联合分布

用于"A/B 实验的转化率推断":

```python
# 先验 Beta(1, 1), 数据 k=8 clicks, n=100 trials
alpha_prior, beta_prior = 1.0, 1.0
k, n = 8, 100

# 后验
alpha_post = alpha_prior + k
beta_post = beta_prior + n - k
posterior = torch.distributions.Beta(alpha_post, beta_post)
print(f"后验均值: {posterior.mean.item():.4f}")  # 平滑 CTR
print(f"后验 95% CI: [{posterior.icdf(torch.tensor(0.025)).item():.4f}, {posterior.icdf(torch.tensor(0.975)).item():.4f}]")
```

### 15.4 离散分布的"扩展家族"

#### (a) Multinomial (多项分布)

Binomial 的多类推广: n 次独立试验, K 类, 每类出现 $k_i$ 次的概率。

```python
n = 100
probs = torch.tensor([0.5, 0.3, 0.2])
multi = torch.distributions.Multinomial(total_count=n, probs=probs)
samples = multi.sample((5,))
print(f"5 次 Multinomial(100, [0.5, 0.3, 0.2]) 采样:\n{samples}")
print(f"每行和: {samples.sum(1)}")  # 都 = n=100
```

#### (b) 负二项分布 (Negative Binomial)

Poisson 的"过分散"版本, 方差可以大于均值。用于"点击次数"等计数 (方差 > 均值)。

```python
nb = torch.distributions.NegativeBinomial(total_count=5, probs=torch.tensor([0.3]))
samples = nb.sample((10000,))
print(f"NB(5, 0.3) 均值: {samples.mean():.4f}, 方差: {samples.var():.4f}")
# 方差 > 均值, 这是 NB 的标志
```

#### (c) 几何分布 (Geometric)

第一次成功前的失败次数, 或第 k 次成功 (两种参数化)。

```python
geo = torch.distributions.Geometric(probs=torch.tensor([0.1]))
samples = geo.sample((10000,))
print(f"Geometric(0.1) 均值: {samples.mean():.4f}")  # (1-p)/p = 9
```

### 15.5 各分布的 MLE 闭式解速查

| 分布 | 参数 | MLE 公式 |
| :--- | :--- | :--- |
| Bernoulli(p) | p | $\hat{p} = \bar{x}$ (样本均值) |
| Categorical(p) | p | $\hat{p}_k = n_k / n$ |
| Normal(μ, σ²) | μ, σ² | $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$ |
| Poisson(λ) | λ | $\hat{\lambda} = \bar{x}$ |
| Exponential(λ) | λ | $\hat{\lambda} = 1/\bar{x}$ |
| Gamma(α, β) | α, β | 矩估计: $\hat{\alpha} = \bar{x}^2/s^2$, $\hat{\beta} = \bar{x}/s^2$ (无闭式) |
| Beta(α, β) | α, β | 矩估计 (无闭式) |
| Uniform(0, b) | b | $\hat{b} = \max(x_i)$ |

**关键观察**: Normal/Bernoulli/Categorical/Poisson/Exponential 都有闭式解。Gamma/Beta 只能数值优化, 这是 GLM 选择 Normal/Poisson 作为响应变量的原因之一。

```python
import torch

# MLE 闭式解验证
samples = torch.randn(10000) * 2 + 3  # N(3, 4)
print(f"理论 μ: 3, MLE: {samples.mean():.4f}")  # 接近 3
print(f"理论 σ²: 4, MLE: {samples.var():.4f}")  # 接近 4

# Poisson MLE
samples_poisson = torch.distributions.Poisson(5.0).sample((10000,))
print(f"理论 λ: 5, MLE: {samples_poisson.mean():.4f}")  # 接近 5
```

---

## 16. 工程师实战:用 PyTorch 拟合 5 种分布 (300 行)

下面 5 个实战任务覆盖工程上最常见的"分布拟合"场景。

### 16.1 实战 1: 用户停留时长 = 对数正态?

**任务**: 给定 10000 个用户停留时长数据, 判断是 LogNormal 还是 Gamma 更合适。

```python
import torch
from torch.distributions import LogNormal, Gamma
from scipy.stats import kstest

# 模拟真实数据: LogNormal(1.5, 0.8)
true_dist = LogNormal(1.5, 0.8)
data = true_dist.sample((10000,))

# 拟合 LogNormal: 用 μ, σ 的 MLE (近似: 对数样本的均值和标准差)
log_data = torch.log(data)
mu_mle = log_data.mean()
sigma_mle = log_data.std()
ln_fit = LogNormal(mu_mle, sigma_mle)
print(f"LogNormal MLE: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

# 拟合 Gamma: 用矩估计
mean_data = data.mean()
var_data = data.var()
beta_mle = mean_data / var_data
alpha_mle = mean_data * beta_mle
g_fit = Gamma(alpha_mle, beta_mle)
print(f"Gamma MLE: α={alpha_mle:.3f}, β={beta_mle:.3f}")

# KS 检验 (Kolmogorov-Smirnov)
ks_ln, p_ln = kstest(data.numpy(), 'lognorm', args=(sigma_mle.item(), 0, torch.exp(mu_mle).item()))
ks_g, p_g = kstest(data.numpy(), 'gamma', args=(alpha_mle.item(), 0, 1/beta_mle.item()))
print(f"KS test: LogNormal p={p_ln:.4f}, Gamma p={p_g:.4f}")
# 真实分布是 LogNormal, 它的 p-value 应该更大 (不能拒绝"数据来自 LogNormal"假设)
```

### 16.2 实战 2: A/B 实验的转化率 + Bayesian 平滑

**任务**: A 实验 50 次点击 / 5000 曝光, B 实验 55 次点击 / 5000 曝光。判断 B 是否显著好。

```python
import torch
from torch.distributions import Beta, Normal

# Beta-Bernoulli 模型
k_a, n_a = 50, 5000
k_b, n_b = 55, 5000

# 后验
post_a = Beta(1.0 + k_a, 1.0 + n_a - k_a)
post_b = Beta(1.0 + k_b, 1.0 + n_b - k_b)

print(f"A 后验均值: {post_a.mean.item():.5f}, 95% CI: [{post_a.icdf(torch.tensor(0.025)).item():.5f}, {post_a.icdf(torch.tensor(0.975)).item():.5f}]")
print(f"B 后验均值: {post_b.mean.item():.5f}, 95% CI: [{post_b.icdf(torch.tensor(0.025)).item():.5f}, {post_b.icdf(torch.tensor(0.975)).item():.5f}]")

# P(B > A) via MC
samples_a = post_a.sample((100000,))
samples_b = post_b.sample((100000,))
prob_b_better = (samples_b > samples_a).float().mean()
print(f"P(B > A): {prob_b_better.item():.4f}")
# 如果 > 0.95, 认为 B 显著好
```

### 16.3 实战 3: 推荐系统的 Dirichlet 平滑

**任务**: 用户对 5 个类别的点击分布, 加 Dirichlet 先验平滑。

```python
import torch
from torch.distributions import Dirichlet, Categorical

# 用户点击历史: [10, 5, 3, 2, 0] (5 个类别)
counts = torch.tensor([10.0, 5.0, 3.0, 2.0, 0.0])

# 加 Dirichlet(1, 1, 1, 1, 1) 先验 (均匀先验)
alpha_prior = torch.ones(5)
posterior = Dirichlet(alpha_prior + counts)
print(f"后验均值 (平滑概率): {posterior.mean}")
print(f"原比例: {counts / counts.sum()}")
# 平滑后, 0 的类别也获得了非零概率

# 蒙特卡洛: 从后验采样, 算每个类别的平均概率
samples = posterior.sample((10000,))
avg_probs = samples.mean(0)
print(f"MC 平均: {avg_probs}")
```

### 16.4 实战 4: 异常流量 = LogNormal 阈值检测

**任务**: 服务器流量数据, 用 LogNormal 拟合, 异常 = 超出 99.99% 分位数。

```python
import torch
from torch.distributions import LogNormal

# 历史正常流量
normal_traffic = torch.tensor([100, 110, 95, 105, 120, 98, 102, 108, 1000])  # 1000 是异常值!
clean = normal_traffic[normal_traffic < 500]  # 简单过滤

# 拟合 LogNormal
log_clean = torch.log(clean.float())
mu, sigma = log_clean.mean(), log_clean.std()
ln = LogNormal(mu, sigma)

# 异常阈值 (99.99% 分位数)
threshold = ln.icdf(torch.tensor(0.9999))
print(f"99.99% 分位数: {threshold.item():.2f}")
print(f"实际异常值 1000 vs 阈值 {threshold.item():.2f}: {'异常' if 1000 > threshold.item() else '正常'}")
```

### 16.5 实战 5: 训练 Loss 曲线 = Gamma 分布族

**任务**: 训练 Loss 服从什么分布? 怎么用它做学习率调整?

```python
import torch
from torch.distributions import Gamma, LogNormal

# 模拟: 训练 100 epoch 的 loss 序列
torch.manual_seed(0)
losses = 1.0 / (torch.arange(1, 101).float() ** 0.5) + 0.05 * torch.randn(100).abs()
# 全部为正, 长尾

# 拟合 LogNormal
log_losses = torch.log(losses)
mu, sigma = log_losses.mean(), log_losses.std()
ln = LogNormal(mu, sigma)

# 拟合 Gamma
mean_l, var_l = losses.mean(), losses.var()
beta_g = mean_l / var_l
alpha_g = mean_l * beta_g
g = Gamma(alpha_g, beta_g)

# 算下一个 epoch loss 的预测区间
pred_lower = ln.icdf(torch.tensor(0.05))
pred_upper = ln.icdf(torch.tensor(0.95))
print(f"LogNormal 预测区间 (90%): [{pred_lower:.4f}, {pred_upper:.4f}]")

# 工程意义: 实际 loss 在预测区间内, 说明训练正常; 如果突然超出, 可能是 lr 太大 / 数据问题
```

---

## 17. ASCII 可视化:分布家族的"族谱" (150 行)

### 17.1 指数族树

```text
                       指数族 (Exponential Family)
                              │
        ┌─────────────┬───────┴────────┬─────────────┐
        │             │                │             │
   Bernoulli   Categorical        Normal         Poisson
        │             │                │             │
   Binomial   Multinomial    MultivariateNormal   NegativeBinomial
        │             │                │             │
     Beta       Dirichlet     Wishart / InvWishart   Geometric
   (共轭先验)   (共轭先验)    (共轭先验)            (指数特例)

   Gamma ── Exponential ── Chi-Square
   (α=1 时退化为 Exponential)
   (α = k/2, β=1/2 时退化为 Chi-Square)

   StudentT ── Cauchy (df=1) ── Normal (df→∞)
```

### 17.2 支撑 (Support) 分类

```text
[-∞, +∞]            [0, +∞)              [0, 1]
───────────         ───────────           ───────
Normal              Gamma                 Beta
StudentT             Exponential          Uniform
Cauchy              LogNormal              Kumaraswamy
Laplace             Chi-Square           (Beta 的替代)
HyperbolicSec       InverseGamma
                    Pareto
                    Weibull
                    Levy

整数 (离散)          自然数 N             类别 K
───────────         ───────────           ───────
Poisson              Poisson (rate)      Categorical
Bernoulli            Geometric           Multinomial
Binomial             NegativeBinomial    HyperGeometric
DiscreteUniform
```

### 17.3 形状对照 (同一支撑下)

```text
连续, 钟形:
  Cauchy       ▂▆████████▆▂           (重尾, 钟形)
  Laplace      ▂▆█████████▆▂          (中等尾)
  Normal       ▂▄▆████████▆▄▂         (标准)
  Logistic     ▁▂▄▆████████▆▄▂▁       (更平)
  StudentT df=3 ▁▂▆█████████▆▂▁       (重尾, 钟形)
  StudentT df=30 ▂▄▆████████▆▄▂       (接近 Normal)

正数, 单峰:
  Exponential  ██▇▅▃▂▁                (单调递减)
  Gamma α=2    ▃▆████████▆▃           (轻微偏)
  Gamma α=5    ▂▄▆██████████▆▄▂       (接近对称)
  LogNormal    ▁▂▄▆████████▆▄▂▁       (对称在对数尺度)
  Weibull k=2  ▃▆████████▆▃           (灵活)

[0,1]:
  Beta(0.5, 0.5)  ▁▁▂▄▆▇▇▆▄▂▁▁        (U 形)
  Beta(1, 1)      ██████████           (均匀)
  Beta(2, 5)      ▂▄▆████████▆▄▂      (偏向 0)
  Beta(5, 5)      ▂▄▆████████▆▄▂      (集中)
```

---

## 18. 跨章节链接与自检 (80 行)

### 跨章节链接

| 本章分布 | 在其他章节的角色 |
| :--- | :--- |
| Bernoulli | 第 6 章: MLE = 样本均值 (CrossEntropy 损失) |
| Categorical | 第 6 章: Softmax 输出 + NLL loss |
| Normal | 第 4 章: 期望/方差的"标准例子"; 第 10 章: VAE/Diffusion 的核心 |
| Gamma | 第 5/7 章: 共轭先验 (Poisson 的先验) |
| Beta | 第 5 章: Beta-Bernoulli 后验 = CTR 平滑 |
| Dirichlet | 第 7 章: Categorical 的共轭先验, LDA 的基础 |

### 本章自检 (15 题)

1. Bernoulli 的 PMF 是什么? MLE 的闭式解?
2. Categorical 输出的概率必须满足什么条件?
3. Normal 分布的两个参数是什么? 它们分别控制什么?
4. Binomial 和 Poisson 的关系?
5. Gamma(α=1, β=λ) 退化成什么分布?
6. Beta(1, 1) 是什么特殊分布?
7. Dirichlet 与 Categorical 的关系 (共轭)?
8. StudentT 与 Normal 的关系?
9. LogNormal 与 Normal 的关系?
10. 对 CTR 用 Beta 还是 Normal? 为什么?
11. 哪些分布有闭式 MLE? 哪些没有?
12. Gamma 分布的 mean 和 mode 分别是多少?
13. Beta 分布的方差公式?
14. Dirichlet 的边缘分布是什么?
15. 离散 + 连续混合分布的 CDF 是什么样?

**简要答案**: 1. P(X=k)=p^k(1-p)^(1-k), $\hat{p} = \bar{x}$。2. $\sum p_k = 1, p_k \ge 0$。3. μ 控制中心, σ² 控制宽度。4. n→∞, p→0, np=λ 时 Binomial → Poisson。5. Exponential(λ)。6. Uniform(0, 1)。7. Dirichlet 是 Categorical 的共轭先验。8. df→∞ 时 StudentT → Normal。9. Y=e^X, X~Normal。10. Beta, 因为 CTR∈[0,1],Beta 自然契合。11. Normal/Bernoulli/Categorical/Poisson/Exponential 有, Gamma/Beta 没有 (用数值优化或矩估计)。12. mean = α/β, mode = (α-1)/β (α>1)。13. Var = αβ/[(α+β)²(α+β+1)]。14. Beta 分布 (一维)。15. 阶梯 + 连续增长。

---

> **本章版本**: v1.1 (扩写)
> **最后更新**: 2026-06-26
> **本次扩写说明**: 新增 §14 分布家族"家谱" (指数族 / 共轭先验 / 决策树), §15 各分布高级话题 (正态/Gamma/Beta 扩展、离散扩展、闭式 MLE), §16 五个 PyTorch 实战, §17 ASCII 可视化族谱, §18 跨章节链接 + 15 题自检。
> **下一章**: [第 4 章:期望·方差·协方差·矩,从样本均值到 Embedding 统计特性](第03章-04-期望·方差·协方差·矩-从样本均值到-Embedding-统计特性.md)

---

## 19. 分布间的"变换关系"详解 (350 行)

工程上常常需要从一个分布的样本得到另一个分布的样本, 这一节把常见的变换关系列全。

### 19.1 采样方法对照

| 目标分布 | 已知工具 | 变换 | 工程例子 |
| :--- | :--- | :--- | :--- |
| Normal(0,1) | Uniform(0,1) | Box-Muller | 噪声生成 |
| Exponential(λ) | Uniform(0,1) | $-\log U / \lambda$ | 等待时间 |
| LogNormal | Normal | $Y = e^X$ | 重尾正数 |
| Chi-Square(df) | Normal | $\sum Z_i^2$ | 拟合优度检验 |
| StudentT(df) | Normal + InvGamma | $Z / \sqrt{V/df}$ | 重尾噪声 |
| Gamma(α, 1) | Exponential | $\sum_{i=1}^\alpha E_i$ | 等待 α 个事件 |
| Beta(α, β) | Gamma | $X / (X + Y)$, $X \sim \text{Gamma}(\alpha)$, $Y \sim \text{Gamma}(\beta)$ | 比率 |
| Dirichlet | Gamma | 归一化 Gamma 样本 | 多维概率 |
| Bernoulli(p) | Uniform(0,1) | $U < p$ | 二分类标签 |
| Categorical(p) | Uniform(0,1) | $U$ 落在累积概率区间 | 多分类 |

### 19.2 Box-Muller 变换:从 Uniform 到 Normal

```python
import torch

def box_muller(n):
    """从 N(0,1) 采样 n 个点, 用 Box-Muller 变换"""
    u1 = torch.rand(n)
    u2 = torch.rand(n)
    # 避免 log(0)
    u1 = u1.clamp(min=1e-7)
    z0 = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * torch.pi * u2)
    z1 = torch.sqrt(-2 * torch.log(u1)) * torch.sin(2 * torch.pi * u2)
    return torch.cat([z0, z1])[:n]

samples = box_muller(10000)
print(f"Box-Muller 样本: 均值={samples.mean():.4f}, 标准差={samples.std():.4f}")
# 应该接近 N(0, 1)
```

**原理**: 如果 $U_1, U_2 \sim \text{Uniform}(0,1)$ 独立, 那么 $Z_0 = \sqrt{-2 \log U_1} \cos(2\pi U_2)$, $Z_1 = \sqrt{-2 \log U_1} \sin(2\pi U_2)$ 独立且 $\sim N(0, 1)$。

直觉: $(U_1, U_2)$ 的极坐标 $(R, \Theta) = (\sqrt{-2\log U_1}, 2\pi U_2)$ 在平面上均匀分布, 转直角坐标就是两个独立标准正态。

### 19.3 反变换方法 (Inverse Transform Sampling)

**定理**: 如果 $U \sim \text{Uniform}(0, 1)$, $F$ 是某个分布的 CDF, 那么 $X = F^{-1}(U) \sim F$。

```python
# 任意分布都可以这样采样 (只要 CDF 可逆)
def inverse_transform_sample(F_inv, n):
    u = torch.rand(n)
    return F_inv(u)

# 例如: Exponential CDF F(x) = 1 - e^(-λx), F^{-1}(u) = -log(1-u)/λ
def exp_icdf(u, lam):
    return -torch.log(1 - u) / lam

samples_exp = inverse_transform_sample(lambda u: exp_icdf(u, 1.0), 10000)
print(f"反变换采样的指数分布: 均值={samples_exp.mean():.4f} (理论: 1.0)")
```

**PyTorch 内部就是这样实现的**: `Uniform(0,1)` → 反 CDF → 任意分布。

### 19.4 从 Gamma 到 Beta

```python
import torch
from torch.distributions import Gamma, Beta

def gamma_to_beta(alpha, beta, n):
    """从 Gamma 构造 Beta"""
    X = Gamma(alpha, 1.0).sample((n,))
    Y = Gamma(beta, 1.0).sample((n,))
    return X / (X + Y)

samples = gamma_to_beta(2.0, 5.0, 10000)
# 验证: 应该接近 Beta(2, 5)
mean_theory = 2.0 / (2.0 + 5.0)
print(f"经验均值: {samples.mean():.4f} (理论: {mean_theory:.4f})")
```

### 19.5 从 Normal 到 Chi-Square 和 StudentT

```python
# Chi-Square(df) = sum of df^2 of standard normals
def chi_square_from_normal(df, n):
    Z = torch.randn(n, df)
    return (Z ** 2).sum(dim=1)

samples_chi2 = chi_square_from_normal(5, 10000)
print(f"Chi-Square(5) 均值: {samples_chi2.mean():.4f} (理论: 5.0)")

# StudentT(df) = Z / sqrt(V/df), Z ~ N(0,1), V ~ Chi^2(df)
def studentt_from_normal_and_chi2(df, n):
    Z = torch.randn(n)
    V = chi_square_from_normal(df, n)
    return Z / torch.sqrt(V / df)

samples_t = studentt_from_normal_and_chi2(3, 10000)
print(f"StudentT(3) 标准差: {samples_t.std():.4f} (理论: sqrt(3/(3-2))={3**0.5:.4f})")
```

### 19.6 分布的极限关系 (Limit Theorems)

| 极限 | 关系 |
| :--- | :--- |
| Binomial(n, p) → Poisson(λ) | n → ∞, p → 0, np = λ 固定 |
| Binomial(n, p) → Normal(np, np(1-p)) | n → ∞, p 固定 (CLT) |
| Gamma(α, β) → Normal(α/β, α/β²) | α → ∞ (CLT on exponential sum) |
| StudentT(df) → Normal(0, 1) | df → ∞ |
| Beta(α, β) → Normal(α/(α+β), αβ/[(α+β)²(α+β+1)]) | α, β → ∞ |
| Chi-Square(df) → Normal(df, 2df) | df → ∞ |
| NegativeBinomial(r, p) → Normal(r(1-p)/p², r(1-p)/p²) | r → ∞ |

```python
# 验证 Binomial → Poisson
import torch
from torch.distributions import Binomial, Poisson

n, p = 1000, 0.005
lam = n * p  # = 5

binomial_samples = Binomial(n, p).sample((10000,))
poisson_samples = Poisson(lam).sample((10000,))

print(f"Binomial(1000, 0.005) 均值: {binomial_samples.mean():.4f}, 方差: {binomial_samples.var():.4f}")
print(f"Poisson(5) 均值: {poisson_samples.mean():.4f}, 方差: {poisson_samples.var():.4f}")
# 两个非常接近, 验证极限关系
```

### 19.7 顺序统计量与样本分位数

```python
import torch

# 从 N(0, 1) 采样, 验证样本分位数
samples = torch.randn(10000)
sorted_samples = samples.sort().values

# 经验分位数
q25 = sorted_samples[int(0.25 * len(sorted_samples))].item()
q50 = sorted_samples[int(0.50 * len(sorted_samples))].item()
q75 = sorted_samples[int(0.75 * len(sorted_samples))].item()

# 理论分位数 (从标准正态 CDF 的逆)
from torch.distributions import Normal
n_dist = Normal(0, 1)
q25_t = n_dist.icdf(torch.tensor(0.25)).item()
q50_t = n_dist.icdf(torch.tensor(0.50)).item()
q75_t = n_dist.icdf(torch.tensor(0.75)).item()

print(f"经验: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")
print(f"理论: Q25={q25_t:.3f}, Q50={q50_t:.3f}, Q75={q75_t:.3f}")
# 非常接近
```

**工程意义**: 样本分位数用于**异常值检测** (P99, P999) 和**数据归一化** (quantile normalization)。

---

## 20. 工程速查表 (80 行)

### 速查表 1: PyTorch 分布构造方式

| 分布 | PyTorch 构造 |
| :--- | :--- |
| Bernoulli(p) | `Bernoulli(probs=torch.tensor([p]))` |
| Categorical(p) | `Categorical(probs=torch.tensor([p1, p2, ...]))` |
| Normal(μ, σ) | `Normal(loc=μ, scale=σ)` |
| Uniform(a, b) | `Uniform(low=a, high=b)` |
| Beta(α, β) | `Beta(concentration1=α, concentration0=β)` |
| Gamma(α, β) | `Gamma(concentration=α, rate=β)` |
| Exponential(λ) | `Exponential(rate=λ)` |
| Poisson(λ) | `Poisson(rate=λ)` |
| StudentT(df, μ, σ) | `StudentT(df=df, loc=μ, scale=σ)` |
| MultivariateNormal(μ, Σ) | `MultivariateNormal(loc=μ, covariance_matrix=Σ)` |
| Dirichlet(α) | `Dirichlet(concentration=α)` |
| LogNormal(μ, σ) | `LogNormal(loc=μ, scale=σ)` |

### 速查表 2: 分布参数与统计量

| 分布 | 均值 | 方差 | 众数 (mode) |
| :--- | :--- | :--- | :--- |
| Bernoulli(p) | p | p(1-p) | 0 或 1 |
| Categorical | p | p(1-p) (单类) | argmax p |
| Normal(μ, σ²) | μ | σ² | μ |
| Exponential(λ) | 1/λ | 1/λ² | 0 |
| Gamma(α, β) | α/β | α/β² | (α-1)/β (α>1) |
| Beta(α, β) | α/(α+β) | αβ/[(α+β)²(α+β+1)] | (α-1)/(α+β-2) (α,β>1) |
| Poisson(λ) | λ | λ | λ (λ 是整数) 或 floor(λ) |
| Uniform(a, b) | (a+b)/2 | (b-a)²/12 | 任意 (平坦) |

### 速查表 3: 各分布的 MLE 复杂度

```text
闭式 MLE:          数值 MLE:
─────────────      ─────────────
Bernoulli          Gamma
Categorical        Beta
Normal             Dirichlet
Poisson            MultivariateNormal (某些参数化)
Exponential        StudentT
Binomial
Multinomial
```

闭式 MLE: 用"样本均值/方差"就能算出参数。
数值 MLE: 需要迭代优化 (梯度下降、Newton 法等), 工程上更贵。

### 速查表 4: 各分布的"易错点"

| 分布 | 易错点 |
| :--- | :--- |
| Bernoulli | p=0 或 p=1 时, log_prob 退化 |
| Categorical | probs 必须 sum 到 1 |
| Normal | σ=0 时退化, 不能采样 |
| Gamma | α < 1 时 mode 在 0, 不是 (α-1)/β |
| Beta | α=β=1 是 Uniform, 不是 Beta(0,0) |
| Exponential | λ 是 rate (1/mean), 不是 mean |
| Poisson | 大 λ 下用 Normal 近似 (CLT) |
| MultivariateNormal | Σ 必须半正定, 否则算不出 |

### 速查表 5: 各分布的工程场景

| 业务场景 | 分布 | 参数意义 |
| :--- | :--- | :--- |
| 用户点击 | Bernoulli | p = CTR |
| 多类推荐 | Categorical | p_k = item k 的点击概率 |
| 房价预测 | Normal | μ = 预测值, σ = 误差 |
| 等待时间 | Exponential | λ = 平均到达率的倒数 |
| 停留时长 | LogNormal | μ, σ 控制对数尺度的中心和宽度 |
| 用户评分 (0-1) | Beta | α, β 控制评分分布形状 |
| 文档主题 | Dirichlet | α_k 控制主题 k 的权重 |
| 推荐点击 (多次曝光) | Binomial | n = 曝光数, p = CTR |
| 异常流量 | LogNormal + 阈值 | 用 99.99% 分位数 |
| Embedding | MultivariateNormal | μ, Σ 控制分布 |

---

> **本章版本**: v1.2 (二次扩写)
> **最后更新**: 2026-06-26
> **本次二次扩写说明**: 在 v1.1 基础上新增 §19 分布间变换关系 (Box-Muller、反变换、Gamma→Beta、Normal→Chi-Square、极限关系、顺序统计量), §20 5 张工程速查表。
> **下一章**: [第 4 章:期望·方差·协方差·矩,从样本均值到 Embedding 统计特性](第03章-04-期望·方差·协方差·矩-从样本均值到-Embedding-统计特性.md)

---

## 21. 高级分布家族简介 (200 行)

这一节补充几个工程上有时会遇到、但前面没细讲的分布。

### 21.1 Weibull 分布

定义: $f(x; k, \lambda) = \frac{k}{\lambda} (\frac{x}{\lambda})^{k-1} e^{-(x/\lambda)^k}$, $x \ge 0$。

- 当 $k=1$: 退化为 Exponential
- 当 $k=2$: Rayleigh 分布
- 当 $k \to \infty$: 集中在 λ 附近

**应用**: 可靠性工程 (机器寿命)、风速分布、生存分析。

```python
import torch
from torch.distributions import Weibull
w = Weibull(scale=2.0, concentration=1.5)  # 注意 PyTorch 命名是 scale 和 concentration (即 1/k 形式)
samples = w.sample((1000,))
print(f"Weibull 均值: {samples.mean():.4f}")
```

### 21.2 Pareto 分布

定义: $f(x; \alpha, x_m) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}$, $x \ge x_m$。

- "80/20 法则"的数学形式
- 财富分布、城市人口、文件大小

```python
# Pareto 在 PyTorch 中需要自定义或用 scipy
from scipy.stats import pareto
alpha, xm = 2.0, 1.0
samples = pareto.rvs(alpha, scale=xm, size=10000)
print(f"Pareto(α=2, xm=1) 均值: {samples.mean():.4f} (理论: α*xm/(α-1) = 2.0)")
print(f"Pareto(α=2, xm=1) 中位数: {np.median(samples):.4f} (理论: xm * 2^(1/α) = {xm * 2**(1/alpha):.4f})")
```

### 21.3 Laplace 分布 (双指数)

定义: $f(x; \mu, b) = \frac{1}{2b} e^{-|x-\mu|/b}$。

- Normal 的"更尖峰、更重尾"版本
- L1 正则化 (Lasso) 的先验

```python
from torch.distributions import Laplace
lap = Laplace(loc=0, scale=1)
samples = lap.sample((10000,))
print(f"Laplace(0, 1) 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
# 标准差 = sqrt(2) ≈ 1.414, 比 Normal 大 (重尾)
```

### 21.4 Cauchy 分布

定义: $f(x; x_0, \gamma) = \frac{1}{\pi} \frac{\gamma}{(x-x_0)^2 + \gamma^2}$。

- 没有均值 (期望不发散!)
- 极端重尾, 用中位数而不是均值描述
- StudentT(df=1) 就是 Cauchy

```python
from torch.distributions import Cauchy
c = Cauchy(loc=0, scale=1)
samples = c.sample((10000,))
print(f"Cauchy(0, 1) 样本均值: {samples.mean():.4f}")  # 不稳定, 因为均值不存在!
print(f"Cauchy(0, 1) 样本中位数: {samples.median():.4f}")  # 接近 0, 稳定
```

### 21.5 Logistic 分布

定义: $f(x; \mu, s) = \frac{e^{-(x-\mu)/s}}{s(1 + e^{-(x-\mu)/s})^2}$。

- 与 Normal 形状相似, 但更重尾
- Logistic 回归的响应变量分布

```python
from torch.distributions import Logistic
lg = Logistic(loc=0, scale=1)
samples = lg.sample((10000,))
print(f"Logistic(0, 1) 均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")
# 标准差 = π/sqrt(3) ≈ 1.814
```

### 21.6 混合分布实战: 用户流失预测

**任务**: 用户的"未来 30 天活跃天数"是混合分布 (大部分用户 0 天, 小部分深度用户 28-30 天)。

```python
import torch
from torch.distributions import Bernoulli, Beta, Gamma

# 简化模型:
# 1. 用户是否活跃: Bernoulli(0.3)
# 2. 活跃用户的活跃天数: Beta 分布 (28-30 天集中)

# 模拟 10000 个用户
n_users = 10000
is_active = Bernoulli(0.3).sample((n_users,))
# Beta(20, 1) 大部分概率在 1 附近 (归一化后是 28-30)
active_days = Beta(20, 1).sample((n_users,)) * 30
days_active = is_active * active_days
print(f"平均活跃天数: {days_active.mean():.2f}")
print(f"P50: {torch.quantile(days_active, 0.5):.2f}")
print(f"P99: {torch.quantile(days_active, 0.99):.2f}")
# 平均约 9 天 (0.3 * 30 * Beta(20,1).mean ≈ 0.3 * 30 * 20/21 ≈ 8.57)
```

### 21.7 Copula:多维分布的"搭积木"

Copula 把多维分布分解为"边缘分布 + 相关结构":
$$
F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d))
$$
其中 $C$ 是 copula 函数, 描述相关结构。

**Gaussian Copula**: $C(u_1, ..., u_d) = \Phi_d(\Phi^{-1}(u_1), ..., \Phi^{-1}(u_d))$, $\Phi_d$ 是多维正态 CDF, $\Phi$ 是一维正态 CDF。

**应用**: 金融风险建模 (各资产的边际分布 + 相关结构), 推荐系统的多目标建模。

```python
# Gaussian Copula 构造多维分布
# (简化: 用多维正态, 但可以替换边缘为非正态)

import torch
from torch.distributions import MultivariateNormal

# 相关系数矩阵
corr = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
mv_normal = MultivariateNormal(loc=torch.zeros(2), covariance_matrix=corr)
samples_normal = mv_normal.sample((1000,))

# 转 copula: 用经验 CDF 转 [0,1]
from scipy.stats import rankdata
u = torch.tensor(rankdata(samples_normal[:, 0].numpy())) / 1000
v = torch.tensor(rankdata(samples_normal[:, 1].numpy())) / 1000

# 转任意边缘: 比如 Beta 边缘
x = torch.distributions.Beta(2, 5).icdf(u)
y = torch.distributions.Beta(5, 2).icdf(v)
print(f"Copula 样本 shape: {x.shape, y.shape}, x 均值: {x.mean():.3f}, y 均值: {y.mean():.3f}")
# x 是 Beta(2,5) 边缘, y 是 Beta(5,2) 边缘, 但保留了 0.7 的相关结构
```

**工程意义**: Copula 让"任意边缘 + 任意相关结构"都能组合, 在金融和推荐系统都有应用。

---

## 22. 分布拟合的工程实践 (200 行)

把分布拟合的"完整流程"列出来, 工程师可以直接套用。

### 22.1 分布拟合的标准流程

```text
1. 探索数据 (EDA)
   ├─ 画直方图, 看形状
   ├─ 算 summary stats (mean, median, std, skew, kurtosis)
   ├─ Q-Q plot 与候选分布对比
   └─ 检查异常值

2. 选择候选分布 (根据数据特征)
   ├─ 离散 → Bernoulli / Poisson / Binomial / NegativeBinomial
   ├─ 连续正 → Gamma / LogNormal / Weibull / Exponential
   ├─ 连续 [0,1] → Beta
   └─ 连续无界 → Normal / StudentT / Cauchy

3. 拟合参数
   ├─ 闭式 MLE (Bernoulli / Normal / Poisson 等)
   ├─ 数值 MLE (Gamma / Beta / Dirichlet)
   └─ 矩估计 (作为 fallback)

4. 评估拟合优度
   ├─ KS test (Kolmogorov-Smirnov)
   ├─ Anderson-Darling test
   ├─ AIC / BIC
   └─ 可视化: 直方图叠加 PDF, Q-Q plot

5. 使用分布
   ├─ 计算概率 / 分位数
   ├─ 生成合成数据
   └─ 异常检测 / 假设检验
```

### 22.2 KS 检验代码

```python
import torch
from scipy.stats import kstest, norm

# 检验数据是否来自 N(0, 1)
samples = torch.randn(1000)
stat, p_value = kstest(samples.numpy(), 'norm', args=(0, 1))
print(f"KS 统计量: {stat:.4f}, p-value: {p_value:.4f}")
# p > 0.05: 不能拒绝"数据来自 N(0, 1)"的假设

# 检验数据是否来自 N(0, 1) (但实际是 Cauchy)
from torch.distributions import Cauchy
samples_cauchy = Cauchy(0, 1).sample((1000,))
stat, p_value = kstest(samples_cauchy.numpy(), 'norm', args=(0, 1))
print(f"Cauchy vs Normal: KS 统计量: {stat:.4f}, p-value: {p_value:.4f}")
# p < 0.05: 拒绝"数据来自 N(0, 1)"
```

### 22.3 AIC / BIC 比较两个模型

```python
import torch
import numpy as np

def neg_log_likelihood_normal(samples, mu, sigma):
    """Normal 分布的 NLL"""
    return -torch.distributions.Normal(mu, sigma).log_prob(samples).sum()

def neg_log_likelihood_lognormal(samples, mu, sigma):
    """LogNormal 分布的 NLL"""
    return -torch.distributions.LogNormal(mu, sigma).log_prob(samples).sum()

# 数据 (实际是 LogNormal)
true_dist = torch.distributions.LogNormal(0.5, 0.3)
data = true_dist.sample((1000,))

# 拟合 Normal
mu_n = torch.log(data).mean()
sigma_n = torch.log(data).std()
nll_normal = neg_log_likelihood_normal(data, data.mean(), data.std())
nll_lognormal = neg_log_likelihood_lognormal(data, mu_n, sigma_n)

# AIC = 2k + 2 NLL, k = 参数数
k_normal, k_lognormal = 2, 2
aic_normal = 2 * k_normal + 2 * nll_normal.item()
aic_lognormal = 2 * k_lognormal + 2 * nll_lognormal.item()

print(f"Normal AIC: {aic_normal:.2f}")
print(f"LogNormal AIC: {aic_lognormal:.2f}")
print(f"AIC 差: {aic_normal - aic_lognormal:.2f}")
# LogNormal AIC 应该更小 (因为真实数据是 LogNormal)
```

### 22.4 Q-Q plot (手写实现)

```python
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 生成数据
samples = torch.randn(500)

# 排序样本
sorted_samples = samples.sort().values

# 理论分位数 (标准正态)
n = len(sorted_samples)
theoretical_quantiles = torch.distributions.Normal(0, 1).icdf(torch.arange(1, n+1).float() / (n+1))

# 画 Q-Q plot (省略实际画图, 只打印前 5 个点)
print("Q-Q plot 数据 (前 5 个点):")
for i in [0, n//4, n//2, 3*n//4, n-1]:
    print(f"  理论分位: {theoretical_quantiles[i]:.3f}, 样本分位: {sorted_samples[i]:.3f}")
# 如果数据来自 N(0, 1), 这些点应该近似在 y=x 直线上
```

### 22.5 工程上选分布的"启发式"

| 数据特征 | 启发式选择 |
| :--- | :--- |
| 离散, 二值 | Bernoulli |
| 离散, 计数, 均值≈方差 | Poisson |
| 离散, 计数, 方差 > 均值 | NegativeBinomial |
| 离散, 多类 | Categorical / Multinomial |
| 连续, [0,1], 单峰 | Beta |
| 连续, 正值, 单峰, 对称 | Gamma (α 较大) 或 LogNormal |
| 连续, 正值, 单峰, 右偏 | Gamma (α 较小) |
| 连续, 正值, 重尾 | LogNormal 或 Pareto |
| 连续, 无界, 钟形 | Normal |
| 连续, 无界, 重尾, 小样本 | StudentT |
| 连续, 无界, 极端重尾 | Cauchy (但很少用, 难处理) |

**经验法则**: 拿不准就试 Normal 或 LogNormal, 看 AIC 谁小就选谁。

---

> **本章版本**: v1.3 (三次扩写)
> **最后更新**: 2026-06-26
> **本次三次扩写说明**: 在 v1.2 基础上新增 §21 高级分布家族 (Weibull, Pareto, Laplace, Cauchy, Logistic, 混合分布, Copula), §22 分布拟合工程实践 (标准流程、KS、AIC/BIC、Q-Q plot、启发式)。
> **下一章**: [第 4 章:期望·方差·协方差·矩,从样本均值到 Embedding 统计特性](第03章-04-期望·方差·协方差·矩-从样本均值到-Embedding-统计特性.md)



## 深度扩展 · 多视角系统展开(续)

### 110. 第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调应用场景视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"应用场景" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调计算复杂度视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"计算复杂度" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调数值方法视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"数值方法" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调历史发展视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"历史发展" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调形式化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"形式化" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 3 章:常用分布家族——正态·Bernoulli·Categorical·Gamma·Beta" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



