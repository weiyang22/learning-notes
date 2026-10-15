---
title: "第01章-09-变分法·Euler-Lagrange方程-VAE·EM·最优控制"
tags:
  - 数学
  - 变分法
  - Euler
  - Lagrange方程
  - VAE
  - EM
description: "本章要点：有限维空间里找最优\"——用梯度下降调 7B 个旋钮。但 ML/DL 里有一大类问题,它的\"自变量\"不是一个向量,而是一个函数:"
chapter_id: 数学-9-01
last_updated: 2026-07-14
---

# 第01章-09-变分法·Euler-Lagrange方程-VAE·EM·最优控制

有限维空间里找最优"——用梯度下降调 7B 个旋钮。但 ML/DL 里有一大类问题,它的"自变量"**不是一个向量**,而是一个**函数**:
>
> - VAE: 给定观测 $x$,找**分布** $q(z \mid x)$ 使 ELBO 最大。$q$ 是函数
> - EM 算法: 给定观测,迭代找**隐变量分布**和**模型参数**。分布是函数
> - 最优控制: 给定动力学,找**策略函数** $\pi(a \mid s)$ 或**控制曲线** $u(t)$。函数
>
> **当"自变量"是函数时,怎么找最优?** 答案是**变分法** (Calculus of Variations)。
>
> 变分法 = **函数版本的微积分**。导数把"一个数"映射到"切线斜率",变分把"一个函数"映射到"扰动敏感度"。
>
> 这一章我们要回答:
>
> 1. 什么是"在函数空间里找最优"? 为什么这件事在 ML/DL 里无处不在?
> 2. 什么是**泛函**? 什么是**变分**? 什么是 **Gateaux 微分**?
> 3. **Euler-Lagrange 方程**怎么推导? 怎么解?
> 4. VAE 的 ELBO 为什么是 $\log p(x)$ 的下界? 重参数化技巧的数学本质是什么?
> 5. EM 算法为什么保证收敛? VAE 和 EM 的数学关系是什么?
> 6. 最优控制的 HJB 方程和 Pontryagin 极大值原理在说什么?
> 7. DreamerV3 的 RSSM 怎么把变分法和最优控制连起来?
>
> **本章与前后章节的关系**:
>
> - 第 3 章学的"多元微分"是**有限维**导数,本章推广到**无穷维** (函数空间)
> - 第 6 章学的"积分"是变分法的运算对象: 泛函 $J[y]$ 几乎都是 $\int L(x, y, y') dx$ 的形式
> - 第 7 章 (8 泛函分析) 学的 Hilbert 空间是变分法的"舞台"
> - 第 9 章 (10 凸分析) 学的 KKT 条件是"含约束变分"的标准工具
> - **变分法是 VAE、EM、最优控制的"母数学"**——这一章是 ML 数学基础里少见的"主线章节"
>
> 学完这一章,你面试被问到 "VAE 为什么能 work"、"ELBO 怎么来"、"EM 怎么收敛"、"连续控制怎么解",都可以从变分法的第一性原理推导出来,而不是死记硬背公式。

---

## 1. 直觉引入:为什么要在"函数空间"里找最优? (200 行)

### 1.1 工程师的痛点 1:VAE 为什么要"近似"后验 $p(z \mid x)$?

回忆 VAE 的目标: 给定观测 $x$ (一张图片),找到**隐变量** $z$ 的**后验分布** $p(z \mid x)$。

**为什么不直接算 $p(z \mid x) = p(x \mid z) p(z) / p(x)$?**

因为 $p(x) = \int p(x \mid z) p(z) dz$ 是一个**难积分**。$z$ 通常是高维 (例如 128 维),积分没法闭式解。

**VAE 的解决方案**: 引入一个**参数化分布族** $q_\phi(z \mid x)$ (通常是各向同性的高斯),让它**尽量接近** $p(z \mid x)$。但怎么度量"接近"?

```text
真后验 p(z|x)            近似 q_φ(z|x)
    ___                       ___
   /   \                     /   \
  /     \      ≈             /     \
 /  复杂  \                 / 简单  \
/  多峰    \               / 单峰    \
----------              ----------
   z                       z
```

**问题来了**: "近似"的搜索空间是**所有可能的分布函数** $q$,不是一个有限维向量。

- **梯度下降**: 在有限维 $\mathbb{R}^d$ 里搜最优 $\theta^*$
- **变分推断**: 在**函数空间** (所有可能的 $q$) 里搜最优 $q^*$

**这就是"变分"的来历**: "变分" = "改变函数的形状"。"变分推断" = "通过改变分布函数的形状来推断后验"。

### 1.2 工程师的痛点 2:EM 算法为什么有效?

GMM (高斯混合模型) 的训练: 给定 $N$ 个数据点 $\{x_i\}$,估计 $K$ 个高斯的均值 $\{\mu_k\}$、方差 $\{\sigma_k^2\}$、混合系数 $\{\pi_k\}$。

如果**我们知道**每个数据点属于哪个高斯 $z_i \in \{1, ..., K\}$,参数估计就是简单的 closed-form (加权 MLE)。

但我们**不知道** $z_i$。所以需要 EM:

```text
E 步: 固定参数 θ, 算 p(z_i | x_i, θ)     ← 在隐变量分布上做"变分"
M 步: 固定隐变量后验 p(z|x), 最大化对数似然
```

E 步里那个 $p(z_i \mid x_i, \theta)$ 是个**函数** (分布),不是个数字。M 步里"最大化"是关于 $\theta$ 的,但 $\theta$ 是通过 E 步算出的 $p(z \mid x, \theta)$ 间接进入似然的。

**EM 的数学本质**: 在**隐变量分布**空间 + **参数**空间的**交替上升**优化。它的收敛性证明用的是变分法 (Jensen 不等式)。

### 1.3 工程师的痛点 3:最优控制——给定动力学,找最优策略

机器人控制: 给定动力学 $\dot{x} = f(x, u)$,状态 $x$ (位置、速度),控制 $u$ (油门、方向盘),找**最优控制轨迹** $u^*(t)$ 使代价 $\int_0^T L(x, u, t) dt$ 最小。

```text
状态 x(t)
   ↑
   |        ___________        ← 最优轨迹 x*(t)
   |       /
   |      /  策略 u*(t) 决定
   |     /
   |____/__________________→ t
```

**"最优策略 $u^*(t)$"是一条曲线,不是一个数字。** 这条曲线的每一个点 $u^*(t)$ 是个数,但整条曲线是个函数。

**强化学习**: 策略 $\pi(a \mid s)$ 也是一个**函数** (从状态到动作的映射)。我们要找最优 $\pi^*$。

### 1.4 共同点:都是"在函数空间里找最优"

| 问题 | 自变量 | 目标 | 约束 |
| :--- | :--- | :--- | :--- |
| VAE 变分推断 | 分布函数 $q(z \mid x)$ | 最大化 ELBO | $q$ 是合法分布 |
| EM 算法 E 步 | 隐变量后验 $p(z \mid x)$ | 固定参数, 算后验 | $p$ 是合法分布 |
| EM 算法 M 步 | 参数 $\theta$ | 最大化 ELBO | $\theta \in \Theta$ |
| 最优控制 | 控制曲线 $u(t)$ | 最小化代价积分 | 动力学约束 |
| RL | 策略 $\pi(a \mid s)$ | 最大化累积奖励 | MDP 动力学 |
| 最短路径 | 曲线 $y(x)$ | 最小化弧长积分 | 固定端点 |

**所有这些问题都共享一个数学结构**:
- 自变量: 一个函数 (有限维向量 → 无穷维函数)
- 目标: 一个**泛函** (输入函数、输出实数)
- 求解: 求泛函的**驻点** (类比有限维的 $\nabla L = 0$)

**这就是变分法要解决的核心问题**。

### 1.5 ASCII 可视化:函数空间 = "形状空间"

我们把函数想象成"形状":

```text
有限维空间 R^d                    函数空间 (无穷维)

  • 一个点                         ∞ 一个"形状"
  /                               /
 • -- 梯度下降路径               /~~~~~~ -- 变分路径
/                                /~~~~~~~~
•                                /~~~~~~~~~~
                                 ~~~~~~~~~
```

每个"形状"是个函数。两个形状之间的"距离"是个范数 $\|y_1 - y_2\|$。

**"最优形状"** = 泛函 $J[y]$ 达到极值的那个函数。

**变分** $\delta J$ = 把形状微扰一下,$J$ 的变化率。

这就是变分法与普通微积分的核心区别: **优化对象从"点"变成"形状"**。

---

## 2. 泛函 (Functional):输入函数、输出实数 (200 行)

### 2.1 函数 vs 泛函

**函数** (Function):
- 输入: 一个或多个实数 (或向量)
- 输出: 一个实数 (或向量)
- 例子: $f(x) = x^2$, $f: \mathbb{R} \to \mathbb{R}$

**泛函** (Functional):
- 输入: 一个**函数**
- 输出: 一个实数
- 例子: $J[y] = \int_0^1 y^2(x) dx$, $J: \mathcal{F} \to \mathbb{R}$

**类比**:
- 函数像一台"数 → 数"的机器
- 泛函像一台"函数 → 数"的机器

```text
        ┌──────────────┐
x  ───► │   f(x)       │ ───►  y = f(x)         (函数)
        └──────────────┘

        ┌──────────────┐
y(·) ──►│   J[y]      │ ───►  c = J[y]         (泛函)
        └──────────────┘
   ↑
   一整条曲线 y(·) = {y(x): x ∈ [0,1]}
```

### 2.2 五个手算例子

**例 1**: $J[y] = \int_0^1 y^2(x) dx$

输入函数 $y(x)$,输出 $y^2$ 在 $[0,1]$ 上的积分。

如果 $y(x) = x$,则 $J[y] = \int_0^1 x^2 dx = 1/3$。

如果 $y(x) = \sin(\pi x)$,则 $J[y] = \int_0^1 \sin^2(\pi x) dx = 1/2$。

**例 2**: $J[y] = y(0)$

输入函数 $y$,输出 $y$ 在 $0$ 点的**值**。这是最简单的"取值"泛函。

**例 3**: $J[y] = \int_0^1 \sqrt{1 + y'(x)^2} dx$ (弧长)

输入曲线 $y(x)$,输出从 $(0, y(0))$ 到 $(1, y(1))$ 的**弧长**。这是为什么地球上的"最短路径"不是直线 (球面上的测地线)。

**例 4**: $J[y] = \int_0^1 (y^2 + y'^2) dx$ (动能 + 势能)

这是物理里"作用量"的一种形式。$y^2$ 是势能,$y'^2$ 是动能 (近似)。最小化这个泛函对应物理系统的真实运动。

**例 5**: $J[y] = y(1)$ (终点值)

输入函数,输出在 $1$ 点的值。和例 2 一样,是个"取值"泛函。

### 2.3 工程意义:Loss 就是泛函

深度学习的 Loss 看起来是"关于参数的函数",但仔细想,它**也是泛函**:

```text
L(θ): R^d → R                (参数 θ 是有限维向量)

ELBO(q): q ∈ D → R           (分布 q 是函数)
```

$L(\theta)$ 中 $\theta \in \mathbb{R}^d$,所以 $L$ 是有限维函数。但 **ELBO 中的 $q$ 是个分布函数**!$q \in \mathcal{P}$ (概率分布空间, 无穷维)。

| 优化类型 | 自变量 | 例子 |
| :--- | :--- | :--- |
| 普通优化 | $\theta \in \mathbb{R}^d$ | Adam, SGD |
| 变分推断 | $q \in \mathcal{P}$ | VAE, VI |
| 最优控制 | $u(\cdot) \in \mathcal{U}$ | 连续控制 |
| 物理作用量 | $y(\cdot) \in C^1$ | Euler-Lagrange |

**所有泛函都可以写成"积分 + 端点"的形式**:
$$J[y] = \int_a^b L(x, y(x), y'(x)) dx + \text{边界项}$$

$L$ 叫 **Lagrangian** (拉格朗日量)。后面 §4 的 Euler-Lagrange 方程就是这种泛函的"导数 = 0"条件。

---

## 3. 变分 (Variation) 与 Gateaux 微分 (280 行)

### 3.1 直觉:对函数做"无穷小扰动"

普通导数: $f'(x) = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon}$

**变分**: $\delta J = \lim_{\epsilon \to 0} \frac{J[y + \epsilon \eta] - J[y]}{\epsilon}$

其中:
- $y$ 是当前函数
- $\eta$ 是任意"合理的"扰动函数 (光滑、固定边界值: $\eta(a) = \eta(b) = 0$)
- $\epsilon$ 是个小标量

**直观**: 把 $y$ 这条曲线往"上挪一点"或"弯一点",看 $J$ 怎么变。$\delta J$ 就是"在 $\eta$ 方向上,$J$ 对函数形状的敏感度"。

```text
y(x)              y(x) + εη(x)

  |  ╱              |  ╱
  | /               | /
  |/                |/╲  ← εη(x) 加了一点扰动
  /                 /  ╲
 /                 /
─────────        ─────────
```

### 3.2 形式化:Gateaux 微分

**定义** (Gateaux 变分): 给定泛函 $J[y]$,如果极限存在:
$$\delta J[y; \eta] = \lim_{\epsilon \to 0} \frac{J[y + \epsilon \eta] - J[y]}{\epsilon}$$

则称 $\delta J[y; \eta]$ 为 $J$ 在 $y$ 处沿 $\eta$ 方向的**第一变分**。

**注意**:
- $\delta J$ 是**双线性的**: 一方面关于 $\epsilon$ 是线性的 (上面已经取极限了),另一方面关于 $\eta$ 是线性的 (假设)
- $\eta$ 叫**变分方向** (variation direction),也叫**测试函数** (test function)

**类比有限维**:
- 函数 $f: \mathbb{R}^d \to \mathbb{R}$ 的方向导数: $\nabla_v f = v^T \nabla f$
- 泛函 $J: \mathcal{F} \to \mathbb{R}$ 的 Gateaux 变分: $\delta J[y; \eta]$

**类比记忆**:
- 方向导数 = "把 $f$ 在 $v$ 方向上推一下,看变化"
- 变分 = "把 $J[y]$ 在 $\eta$ 方向上推一下,看变化"

### 3.3 五个手算例子

**例 1**: $J[y] = \int_0^1 y^2(x) dx$ 的变分

$$
\begin{aligned}
J[y + \epsilon \eta] &= \int_0^1 (y + \epsilon \eta)^2 dx = \int_0^1 (y^2 + 2\epsilon y \eta + \epsilon^2 \eta^2) dx \\
\frac{J[y + \epsilon \eta] - J[y]}{\epsilon} &= \int_0^1 (2 y \eta + \epsilon \eta^2) dx
\end{aligned}
$$

取极限 $\epsilon \to 0$:

$$\boxed{\delta J[y; \eta] = \int_0^1 2 y(x) \eta(x) dx}$$

**例 2**: $J[y] = \int_0^1 y'^2 dx$ 的变分

$$
\begin{aligned}
J[y + \epsilon \eta] &= \int_0^1 (y' + \epsilon \eta')^2 dx = \int_0^1 (y'^2 + 2\epsilon y' \eta' + \epsilon^2 \eta'^2) dx \\
\frac{J[y + \epsilon \eta] - J[y]}{\epsilon} &= \int_0^1 (2 y' \eta' + \epsilon \eta'^2) dx
\end{aligned}
$$

取极限:

$$\delta J[y; \eta] = 2 \int_0^1 y'(x) \eta'(x) dx$$

**关键**: 这里出现了 $\eta'$ 而不是 $\eta$。处理 $\eta'$ 需要**分部积分**。

**例 3**: $J[y] = y(0)^2$ 的变分

$$
\begin{aligned}
J[y + \epsilon \eta] &= (y(0) + \epsilon \eta(0))^2 = y(0)^2 + 2\epsilon y(0) \eta(0) + O(\epsilon^2) \\
\delta J[y; \eta] &= 2 y(0) \eta(0)
\end{aligned}
$$

这是"端点型"泛函的变分,通常作为边界条件出现。

**例 4**: $J[y] = \int_0^1 (y^2 + y'^2) dx$ 的变分 (Euler-Lagrange 来源)

$$J[y + \epsilon \eta] = \int_0^1 [(y + \epsilon \eta)^2 + (y' + \epsilon \eta')^2] dx$$

展开:

$$\frac{J[y + \epsilon \eta] - J[y]}{\epsilon} = \int_0^1 [2 y \eta + 2 y' \eta'] dx + O(\epsilon)$$

取极限:

$$\delta J[y; \eta] = 2 \int_0^1 [y(x) \eta(x) + y'(x) \eta'(x)] dx$$

**这个泛函的极值条件 $\delta J = 0$** (对所有 $\eta$) **正是 Euler-Lagrange 方程的来源**。我们会在 §4 详细推导。

**例 5**: $J[q] = -\int q(x) \log q(x) dx$ (熵) 的变分

这是连续分布的**负熵**。要算它在 $q$ 处的变分,**还要**附上归一化约束 $\int q dx = 1$ (用 Lagrange 乘子)。

$$
\begin{aligned}
J[q + \epsilon \eta] &= -\int (q + \epsilon \eta) \log(q + \epsilon \eta) dx \\
\log(q + \epsilon \eta) &= \log q + \frac{\epsilon \eta}{q} + O(\epsilon^2)
\end{aligned}
$$

所以:

$$J[q + \epsilon \eta] = -\int (q + \epsilon \eta) \left[ \log q + \frac{\epsilon \eta}{q} \right] dx + O(\epsilon^2)$$

$$= -\int q \log q dx - \epsilon \int \eta \log q dx - \epsilon \int \eta dx + O(\epsilon^2)$$

取极限:

$$\delta J[q; \eta] = -\int \eta(x) [\log q(x) + 1] dx$$

加归一化约束 $\int \eta = 0$ (因为 $\int q = 1$ 在扰动后仍要求 $\int (q + \epsilon \eta) = 1$):

**结论**: 令 $\delta J - \lambda \int \eta = 0$,得到:

$$\log q(x) + 1 = -\lambda \implies q(x) = e^{-1 - \lambda} = \text{const}$$

**所以最大熵分布 = 均匀分布** (在仅约束归一化的情况下)。这是变分法的经典结论。

### 3.4 关键案例:相对熵最小化 → 指数族分布

设真实分布 $p(x)$,找一个**指数族分布** $q_\theta(x) = h(x) \exp(\theta^T \phi(x) - A(\theta))$ 使 KL 散度最小:

$$D_{\text{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx$$

$$= \int p(x) [\log p(x) - \log q_\theta(x)] dx$$

$$= \int p(x) [\log p(x) - \theta^T \phi(x) + A(\theta)] dx$$

固定 $p$,对 $\theta$ 求导 (这是有限维!):

$$\frac{\partial D_{\text{KL}}}{\partial \theta} = -\int p(x) \phi(x) dx + A'(\theta) \int p(x) dx = -\mathbb{E}_p[\phi(x)] + A'(\theta) = 0$$

得到:

$$\boxed{A'(\theta) = \mathbb{E}_p[\phi(x)]}$$

**这就是指数族的 MLE 等价于矩匹配**的数学证明: 最小化 KL = 匹配 $\phi(x)$ 的期望。

**应用**: VAE 里 decoder 通常假设 $p(x \mid z)$ 是高斯,这就是把 $p(x \mid z)$ 限制在指数族,再让 encoder 学一个高斯 $q(z \mid x)$ 去近似 $p(z \mid x)$——双重指数族近似。

### 3.5 Gateaux 微分 vs Frechet 微分

**Gateaux 微分** (本章用的): 只要求对**特定** $\eta$ 方向,极限 $\lim_{\epsilon \to 0} (J[y + \epsilon \eta] - J[y]) / \epsilon$ 存在。

**Frechet 微分** (更强的概念): 要求极限对**所有** $\eta$ **一致**地存在, 且 $\delta J[y; \eta] = L[\eta]$ 是 $\eta$ 的**有界线性泛函**。

**类比**:
- Gateaux = 方向导数 (只对一个方向)
- Frechet = 全微分 (所有方向都"平滑")

**本章的处理**: 在合理的光滑性假设下,两者一致,所以用 Gateaux 推导 Euler-Lagrange 即可。

### 3.6 边界条件与自然边界条件

**强形式 (Dirichlet)**: $\eta(a) = \eta(b) = 0$, 即**端点固定**, $y(a), y(b)$ 不变。

**弱形式 (Neumann / 自然边界条件)**: $\eta(a), \eta(b)$ 任意, 此时 $\delta J = 0$ 要求**额外**条件:

$$\left. \frac{\partial L}{\partial y'} \right|_{x=a} = 0, \quad \left. \frac{\partial L}{\partial y'} \right|_{x=b} = 0$$

**叫"自然"是因为它从变分自动出现**, 不用人为强加。

**例**: 弹性杆的平衡, 端点如果**自由** (没有外力), 则弯矩为零 ($\partial L / \partial y' = 0$); 如果**固定**, 则 $y$ 固定 ($\eta = 0$)。

**ML 联系**: VAE 训练中, 边界条件对应"数据点固定"——给定 $x$, 找最优 $q(z \mid x)$。"自然边界条件"对应"先验约束"——$q$ 应该接近 $p(z)$, 即 $\log q - \log p$ 在边界处为 0。

### 3.7 变分法的基本引理 (Fundamental Lemma)

**定理** (基本引理): 如果对**所有**光滑 $\eta$ 都满足 $\int_a^b f(x) \eta(x) dx = 0$, 那么 $f \equiv 0$ (在 $(a, b)$ 上几乎处处)。

**直觉**: 如果 $f$ 在某点 $x_0$ 处非零, 例如 $f(x_0) > 0$, 则在 $x_0$ 的小邻域内 $f > 0$。选 $\eta$ 在该邻域为正, 其它为 0, 则积分 $> 0$, 矛盾。

**这是变分法的核心工具**: 把"$\delta J = 0$ 对所有 $\eta$"翻译成"$f = 0$"。

**工程类比**: 有限维 $\nabla f = 0$ 中, $f = \nabla_v f$ 对所有 $v$ 成立 $\implies \nabla f = 0$。变分法的基本引理是其在无穷维的推广。

---

## 4. Euler-Lagrange 方程:本章核心 (350 行)

### 4.1 经典问题:最短路径 (Brachistochrone)

**问题**: 给定 $A = (0, 0)$ 和 $B = (1, y_1)$,找一条曲线 $y(x)$ 使物体沿这条曲线从 $A$ 滑到 $B$ 用时最短。

**物理**: 物体在 $y$ 处的速度由能量守恒 $v = \sqrt{2gy}$ 决定 ($g$ 是重力加速度)。

用时 = 弧长 / 速度:

$$T[y] = \int_0^1 \frac{\sqrt{1 + y'^2}}{v(y)} dx = \int_0^1 \sqrt{\frac{1 + y'^2}{2 g y}} dx$$

最小化 $T[y]$。

这就是著名的**最速降线问题** (Brachistochrone),1696 年由 Johann Bernoulli 提出,Euler-Lagrange 方程就是为了解决这类问题而发展出来的。

### 4.2 Euler-Lagrange 方程推导

考虑最一般形式的泛函:

$$J[y] = \int_a^b L(x, y(x), y'(x)) dx$$

**目标**: 找 $y$ 使 $\delta J = 0$ (驻点条件)。

**步骤 1**: 写出变分

$$J[y + \epsilon \eta] - J[y] = \int_a^b [L(x, y + \epsilon \eta, y' + \epsilon \eta') - L(x, y, y')] dx$$

Taylor 展开 $L$ (对 $y$ 和 $y'$):

$$\approx \int_a^b \epsilon \left[ \frac{\partial L}{\partial y} \eta + \frac{\partial L}{\partial y'} \eta' \right] dx$$

**步骤 2**: 分离变量

$$\delta J = \epsilon \int_a^b \left[ \frac{\partial L}{\partial y} \eta + \frac{\partial L}{\partial y'} \eta' \right] dx$$

**步骤 3**: 对 $\eta'$ 项做**分部积分**

$$\int_a^b \frac{\partial L}{\partial y'} \eta' dx = \left[ \frac{\partial L}{\partial y'} \eta \right]_a^b - \int_a^b \frac{d}{dx}\left( \frac{\partial L}{\partial y'} \right) \eta dx$$

**步骤 4**: 边界条件 $\eta(a) = \eta(b) = 0$ (变分不改变端点),所以边界项为 0:

$$\delta J = \epsilon \int_a^b \left[ \frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} \right] \eta(x) dx$$

**步骤 5**: $\delta J = 0$ 对**所有** $\eta$ 成立,利用**变分法基本引理** (如果 $\int f(x) \eta(x) dx = 0$ 对所有 $\eta$ 成立, 则 $f \equiv 0$):

$$\boxed{\frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} = 0}$$

这就是 **Euler-Lagrange 方程**。

**记忆口诀**: "偏 L 偏 y 减去对 x 求导 (偏 L 偏 y'),等于 0"。

$$\frac{\partial L}{\partial y} = \frac{d}{dx} \frac{\partial L}{\partial y'}$$

**它说**: "泛函取极值时,L 关于 $y$ 的偏导 = L 关于 $y'$ 的偏导沿 $x$ 的变化率"。

### 4.3 五个手算例子

**例 1**: $L = y'^2$ (最短路径退化版)

求 $J[y] = \int_0^1 y'^2 dx$ 的极值。

Euler-Lagrange:

$$\frac{\partial L}{\partial y} = 0, \quad \frac{\partial L}{\partial y'} = 2y', \quad \frac{d}{dx} \frac{\partial L}{\partial y'} = 2y''$$

代入: $0 - 2y'' = 0 \implies y'' = 0 \implies y = ax + b$。

**结论**: 极值函数是**直线**。这正是"两点之间直线最短"的数学证明。

**例 2**: $L = y'^2 - y^2$ (一维谐振子)

Euler-Lagrange:

$$\frac{\partial L}{\partial y} = -2y, \quad \frac{\partial L}{\partial y'} = 2y', \quad \frac{d}{dx} \frac{\partial L}{\partial y'} = 2y''$$

代入: $-2y - 2y'' = 0 \implies y'' + y = 0$。

解: $y = A \cos x + B \sin x$。

**物理意义**: 这是**简谐振动方程**。Lagrangian $L = \text{动能} - \text{势能} = \frac{1}{2} y'^2 - \frac{1}{2} y^2$,所以 $y(t) = A\cos(t) + B\sin(t)$ 是单摆的运动轨迹。

**例 3**: $L = \sqrt{1 + y'^2}$ (真正最短路径 = 弧长)

Euler-Lagrange:

$$\frac{\partial L}{\partial y} = 0, \quad \frac{\partial L}{\partial y'} = \frac{y'}{\sqrt{1 + y'^2}}$$

$$\frac{d}{dx} \frac{y'}{\sqrt{1 + y'^2}} = 0$$

意味着 $\frac{y'}{\sqrt{1 + y'^2}} = c$ (常数),解得:

$$y' = \frac{c}{\sqrt{1 - c^2}} \cdot 1 = \text{常数}$$

所以 $y = ax + b$,直线。

**这告诉我们**: 在**平直空间**里,最短路径是直线。在**弯曲空间** (球面、双曲空间) 里,最短路径是**测地线**——需要修正 Euler-Lagrange。

**例 4**: $L = y'^2 - y^2 + 2y$ (含源项)

Euler-Lagrange:

$$\frac{\partial L}{\partial y} = -2y + 2, \quad \frac{\partial L}{\partial y'} = 2y'$$

代入: $(-2y + 2) - 2y'' = 0 \implies y'' + y = 1$

解: $y = 1 + A\cos x + B\sin x$ (特解 + 齐次解)。

**物理意义**: 这是一个**受迫谐振子**。源项 $2y$ 相当于外力。

**例 5**: 多个函数的泛函 (Hamilton 力学的雏形)

考虑**两个函数** $y_1(x), y_2(x)$ 的泛函:

$$J[y_1, y_2] = \int_a^b L(x, y_1, y_2, y_1', y_2') dx$$

变分: $\delta J = \int \left[ \frac{\partial L}{\partial y_1} \eta_1 + \frac{\partial L}{\partial y_2} \eta_2 + \frac{\partial L}{\partial y_1'} \eta_1' + \frac{\partial L}{\partial y_2'} \eta_2' \right] dx$

分部积分后,得到**两个** Euler-Lagrange 方程:

$$\frac{\partial L}{\partial y_i} - \frac{d}{dx} \frac{\partial L}{\partial y_i'} = 0, \quad i = 1, 2$$

**这是 Hamilton 力学的出发点**。广义坐标 $q_i(t)$ (位置、角度) 和 $\dot{q}_i(t)$ (速度) 构成 Lagrangian $L(q, \dot{q}, t) = T - V$ (动能减势能),Euler-Lagrange 给出 Newton 第二定律。

### 4.4 含约束的变分 (Lagrange 乘子变分版)

**问题**: 找 $y(x)$ 使 $J[y] = \int L(x, y, y') dx$ 最小, 满足约束 $K[y] = \int M(x, y, y') dx = C$。

**解法**: 引入 Lagrange 乘子 $\lambda$,构造新泛函:

$$\mathcal{L}[y, \lambda] = J[y] - \lambda (K[y] - C) = \int [L - \lambda M] dx + \lambda C$$

对 $y$ 和 $\lambda$ 独立变分:

1. $\delta_y \mathcal{L} = 0$: $\frac{\partial (L - \lambda M)}{\partial y} - \frac{d}{dx} \frac{\partial (L - \lambda M)}{\partial y'} = 0$
2. $\delta_\lambda \mathcal{L} = 0$: $K[y] = C$ (恢复约束)

**应用**: VAE 里 $\int q(z) dz = 1$ 就是个约束,变分时用乘子保证 $q$ 是合法分布。

### 4.5 第二变分与极值的判定

Euler-Lagrange 给出**驻点** (stationary point),不一定是**极小** (minimum)。

**第二变分** $\delta^2 J[y; \eta]$ 判定:

$$\delta^2 J[y; \eta] = \left. \frac{d^2}{d\epsilon^2} J[y + \epsilon \eta] \right|_{\epsilon = 0}$$

- 若 $\delta^2 J \geq 0$ 对所有 $\eta$ 成立, $y$ 是**局部极小**
- 若 $\delta^2 J \leq 0$, 是**局部极大**
- 若正负不定, 是**鞍点**

**例**: $J[y] = \int y'^2 dx$ 的第二变分:

$$\delta^2 J[y; \eta] = 2 \int \eta'^2 dx \geq 0$$

总是 $\geq 0$, 所以直线 (驻点) 是局部极小。几何上对应"最短路径是直线"。

**重要**: 变分法只能找到驻点,不能保证全局最优。**这是常见误区** (见 §9)。

### 4.6 Hamilton 最小作用量原理 (物理动机)

**Hamilton 原理** (1824 年): 物理系统从状态 $(q(t_1), t_1)$ 演化到 $(q(t_2), t_2)$ 走的是使**作用量** $S[q]$ 极值的路径:

$$S[q] = \int_{t_1}^{t_2} L(q(t), \dot{q}(t), t) dt$$

其中 $L = T - V$ = 动能 - 势能 (Lagrangian)。

**Euler-Lagrange 给出 Newton 第二定律**:

$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}} = \frac{\partial L}{\partial q}$$

对 $L = \frac{1}{2} m \dot{q}^2 - V(q)$ (一维质点), $\frac{\partial L}{\partial q} = -V'(q)$, $\frac{\partial L}{\partial \dot{q}} = m \dot{q}$, 代入:

$$m \ddot{q} = -V'(q) = F(q)$$

正是 $F = ma$。

**ML/DL 类比**: Hamilton 原理说"真实物理路径 = 使作用量极小的路径"。类比到 ML: "最优参数轨迹 = 使损失极小的轨迹"。变分法是这两者的共同数学。

**深度学习的 ODE 模型** (Neural ODE, 第 8 章): $\dot{h}(t) = f_\theta(h(t), t)$, 训练目标是匹配数据, 通过变分法可以推导出 adjoint 方法 (Pontryagin 的逆时间形式)。

### 4.7 Euler-Lagrange 的守恒律 (Noether 定理预告)

**对称性 → 守恒律** (Noether 定理, 1918 年):

- 时间平移不变性 → **能量守恒**
- 空间平移不变性 → **动量守恒**
- 旋转不变性 → **角动量守恒**

数学上: 如果 $L$ 不显式依赖 $x$ (时间平移不变), 则 $L - y' \frac{\partial L}{\partial y'} = \text{const}$ (Hamilton 量守恒)。

**证明**: $\frac{d}{dx} (L - y' \frac{\partial L}{\partial y'}) = \frac{\partial L}{\partial x} + \frac{\partial L}{\partial y} y' + \frac{\partial L}{\partial y'} y'' - y'' \frac{\partial L}{\partial y'} - y' \frac{d}{dx} \frac{\partial L}{\partial y'}$

用 Euler-Lagrange $\frac{\partial L}{\partial y} = \frac{d}{dx} \frac{\partial L}{\partial y'}$, 化简得 $\frac{\partial L}{\partial x}$。如果 $\frac{\partial L}{\partial x} = 0$ (没有显式 $x$), 则 $L - y' \frac{\partial L}{\partial y'}$ 是常数。

**ML 类比**: VAE 的 ELBO 在隐变量分布族不变性下也有守恒律——这是"模式坍塌" (mode collapse) 现象的理论根源。

### 4.8 端点条件分类 (面试常考)

**问题**: Euler-Lagrange 是二阶 ODE, 需要**两个**边界条件。常见的组合:

| 端点 $x=a$ | 端点 $x=b$ | 名称 | 工程例子 |
| :--- | :--- | :--- | :--- |
| $y(a)$ 固定 | $y(b)$ 固定 | Dirichlet (两点边值) | 拉弦的两端固定 |
| $y(a)$ 固定 | $y'(b)$ 固定 | Robin (混合边值) | 一端固定一端铰接 |
| $y'(a)$ 自然 | $y'(b)$ 自然 | Neumann (自然) | 自由杆的平衡 |
| $y(a)$ 固定 | $y(b)$ 自由 | 一边自由 | 半无限杆 |

**自然边界条件**: 如果 $y(b)$ 自由, 则 $\left.\frac{\partial L}{\partial y'}\right|_{x=b} = 0$。

**ML 类比**: VAE 中, 隐变量 $z$ 没有"端点", 但有**先验** $p(z)$ 作为边界约束。$\log q - \log p$ 在隐空间"边界"处为 0。

---

## 5. 应用 1:VAE 的 ELBO 与变分推断 (280 行)

### 5.1 VAE 的目标:最大化 $\log p(x)$

给定观测 $x$ (一张图片),生成模型想要**最大化** $\log p(x)$,即观测数据的对数似然。

### 5.2 直接计算的困难

$$\log p(x) = \log \int p(x, z) dz = \log \int p(x \mid z) p(z) dz$$

这是关于 $z$ 的**积分**。如果 $z$ 是高维 (例如 128 维),积分没法闭式解。

蒙特卡洛: $\log p(x) \approx \log \frac{1}{M} \sum_m p(x \mid z_m) p(z_m) / q(z_m)$,其中 $z_m \sim q$。但这个估计**方差很大**,训练不稳定。

### 5.3 引入变分分布 $q(z)$,用 KL 构造下界

**关键想法**: 引入一个**任意的**分布 $q(z)$,用**Jensen 不等式**构造 $\log p(x)$ 的**下界**:

$$
\begin{aligned}
\log p(x) &= \log \int q(z) \frac{p(x, z)}{q(z)} dz \\
&= \log \mathbb{E}_{z \sim q}\left[ \frac{p(x, z)}{q(z)} \right] \\
&\geq \mathbb{E}_{z \sim q} \log \left[ \frac{p(x, z)}{q(z)} \right] \quad \text{(Jensen 不等式: log 是凹函数)} \\
&= \mathbb{E}_{z \sim q}[\log p(x, z) - \log q(z)] \\
&= \mathbb{E}_{z \sim q}[\log p(x \mid z) + \log p(z) - \log q(z)] \\
&= \mathbb{E}_{z \sim q}[\log p(x \mid z)] - \mathbb{E}_{z \sim q}\left[ \log \frac{q(z)}{p(z)} \right] \\
&= \mathbb{E}_{z \sim q}[\log p(x \mid z)] - D_{\text{KL}}(q(z) \| p(z))
\end{aligned}
$$

**这叫 ELBO** (Evidence Lower Bound):

$$\boxed{\text{ELBO}(q) = \mathbb{E}_{q}[\log p(x \mid z)] - D_{\text{KL}}(q(z) \| p(z))}$$

**结论**: $\log p(x) \geq \text{ELBO}(q)$ 对**任何**分布 $q$ 成立。

### 5.4 ELBO 的两种视角

**视角 1 (生成视角, encoder-decoder)**:
- 第一项 = **重建项**: 鼓励 $q(z)$ 编码的 $z$ 能重建 $x$
- 第二项 = **正则项**: 鼓励 $q(z)$ 接近先验 $p(z)$ (通常是 $\mathcal{N}(0, I)$)

**最大化 ELBO = 重建好 + 不偏离先验**。

**视角 2 (变分视角)**:
- ELBO 是 $\log p(x)$ 的**紧下界**: 当 $q(z) = p(z \mid x)$ 时, ELBO = $\log p(x)$ 取等
- 所以**最大化 ELBO** 在 $q$ 上 = **拉近 $q$ 和真实后验 $p(z \mid x)$** (在 KL 散度意义上)

**这是变分法的核心应用**: 在**分布函数空间**里搜索最优 $q^*$,使 ELBO 最大。

### 5.5 Reparameterization Trick 的数学本质

**问题**: 算 $\nabla_\phi \mathbb{E}_{z \sim q_\phi(z)}[f(z)]$ 时,**不能**把梯度塞进期望里 (采样是随机的)。

**朴素法 (错)**:

$$z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2) \implies z \text{ 的采样依赖 } \phi, \text{ 不能反向传播}$$

**Reparameterization (对)**:

$$z = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

把随机性移到 $\epsilon$ (与 $\phi$ 无关),$z$ 现在是 $\phi$ 的**确定性函数**:

$$\nabla_\phi \mathbb{E}_\epsilon[f(\mu_\phi + \sigma_\phi \epsilon)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi \epsilon)]$$

现在梯度可以交换期望和求导。

**数学本质**: 这是一个**测度变换** (change of measure) + **Jacobian** 应用。回忆第 3 章的多元换元公式,以及第 6 章的重参数化技巧。

### 5.6 PyTorch 代码:从零实现 VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """变分自编码器: 编码器 q_φ(z|x) + 解码器 p_θ(x|z)"""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # 编码器: x -> (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器: z -> x_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # 输出在 [0, 1] (假设 x 是二值图像)
        )

    def encode(self, x):
        """x: (batch, 784), 返回 (μ, log σ²)"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = μ + σ · ε, ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def elbo_loss(x_recon, x, mu, logvar):
    """
    ELBO = 重建项 - KL(q(z|x) || N(0, I))

    数学:
      重建项 = E_q[log p(x|z)]   ≈ BCE(x_recon, x)
      KL 项 = -0.5 · Σ(1 + log σ² - μ² - σ²)
    """
    # 重建项 (对所有像素求和)
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 散度的解析形式 (q 和 p 都是高斯)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 最大化 ELBO = 最小化 -ELBO
    return recon + kl  # 这是 -ELBO, 我们要最小化它


# 训练循环 (示意)
def train_vae(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.view(-1, 784)
            x_recon, mu, logvar = model(x)
            loss = elbo_loss(x_recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()  # 这里 autograd 通过重参数化处理梯度
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: avg loss = {total_loss / len(dataloader.dataset):.4f}")
```

**注意**:
1. `reparameterize` 是**数学关键**: 没有它, $\nabla_\phi$ 无法穿过采样算子
2. `elbo_loss` 中 `recon + kl` = $-ELBO$, 最小化它 = 最大化 ELBO
3. KL 项用**闭式解** (两个高斯的 KL 有公式), 不需要采样

### 5.7 KL 散度的两种方向

**前向 KL** (前向, 也叫 M-projection): $D_{\text{KL}}(p \| q) = \int p \log(p/q)$

- 零强制 (zero-forcing): $q$ 在 $p > 0$ 处必须 $q > 0$, 否则 KL = $\infty$
- VAE 用这个: $D_{\text{KL}}(q(z \mid x) \| p(z))$, 鼓励 $q$ 在 $p$ 不为 0 的地方都覆盖

**反向 KL** (反向, 也叫 I-projection): $D_{\text{KL}}(q \| p) = \int q \log(q/p)$

- 模式覆盖 (mass-covering): $q$ 在 $p = 0$ 处也倾向于 $q = 0$
- EM 算法用这个: $D_{\text{KL}}(p(z \mid x, \theta) \| q(z))$, 因为 E 步把 $q$ 设为真实后验

**重要差异**:

| | 前向 KL | 反向 KL |
| :--- | :--- | :--- |
| 零强制 vs 覆盖 | 零强制 | 覆盖 |
| 适合 | 近似后验 (VAE) | 变分上界 (EM) |
| 极端情况 | $q$ 必须覆盖 $p$ 全部支撑 | $q$ 可以忽略 $p$ 的低密度区 |

**这是 VAE 倾向"模糊生成"的原因**: 前向 KL + 高斯后验 + 高斯先验 → encoder 倾向于把 $q$ 收缩到 $p(z)$ 的均值附近, 生成图样多样性下降。

### 5.8 IWAE 界:更紧的 ELBO

**重要性加权 ELBO** (Importance Weighted Autoencoder, IWAE, Burda et al. 2016):

用 $K$ 个样本的蒙特卡洛估计 + 重要性权重:

$$\mathcal{L}_K(x) = \mathbb{E}_{z_1, ..., z_K \sim q(z \mid x)}\left[ \log \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k \mid x)} \right]$$

**定理**: $\mathcal{L}_K$ 是 $\log p(x)$ 的下界, 且 $\mathcal{L}_K \leq \mathcal{L}_{K+1}$, 且 $\lim_{K \to \infty} \mathcal{L}_K = \log p(x)$。

**与 ELBO 的关系**: ELBO 是 $K=1$ 的特例, IWAE 是更紧的下界 (tight bound)。

**为什么 $K > 1$ 更好**: 每个 $z_k$ 是个"假设",$\frac{p(x, z_k)}{q(z_k \mid x)}$ 是这个假设解释 $x$ 的"重要性"。求和后取 log, 选最可能的假设——**这本质上是重要性采样**。

**PyTorch 代码**:

```python
def iwae_loss(x_recon_logits, x, mu, logvar, K=5):
    """重要性加权 ELBO, K 个样本"""
    batch = x.shape[0]
    # 重复 K 次, 每行一个样本
    x_rep = x.repeat(K, 1)  # (K*batch, 784)
    mu_rep = mu.repeat(K, 1)
    logvar_rep = logvar.repeat(K, 1)

    # 采样 K 次
    std = torch.exp(0.5 * logvar_rep)
    z = mu_rep + std * torch.randn_like(std)

    # 重建
    x_recon = decoder(z)
    log_p_x_given_z = -F.binary_cross_entropy(x_recon, x_rep, reduction='none').sum(dim=1)
    log_p_z = -0.5 * (z**2).sum(dim=1)  # 标准正态
    log_q_z_given_x = -0.5 * (logvar_rep + (z - mu_rep)**2 / logvar_rep.exp()).sum(dim=1)

    # log p(x, z) - log q(z|x)
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x  # (K*batch,)

    # reshape, 求和后取 log
    log_w = log_w.view(K, batch)
    iwae = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(K))

    return -iwae.mean()  # 最小化 -IWAE = 最大化 IWAE
```

**IWAE 在工程中的好处**: 训练更稳定, 生成质量略好, 特别是在 $q$ 离真实后验较远时 (例如 encoder 没训练好时)。

### 5.9 VAE 的常见失败模式 (变分角度)

**问题 1: 后验坍塌 (Posterior Collapse)**

如果 decoder 太强, $p(x \mid z)$ 不依赖 $z$, 那么最优 $q(z \mid x) = p(z)$, 即 encoder 输出与 $x$ 无关。

**数学原因**: 此时 $\text{KL}(q \| p) = 0$, 但 $\log p(x \mid z)$ 不需要 $z$ 提供信息。ELBO 上 $z$ 失去作用。

**解法**:
- KL 退火 (KL annealing): 让 KL 项的权重从 0 慢慢增到 1
- Free Bits: KL 项低于阈值时不优化
- 更紧的 ELBO (IWAE)

**问题 2: 模糊生成 (Blurry Outputs)**

如果 $p(x \mid z)$ 是高斯 (L2 重建损失), 生成图样倾向于模糊。

**数学原因**: 高斯分布 + MSE = 预测**均值**, 多个模式平均 = 模糊。

**解法**: 用 $\beta$-VAE, VQ-VAE, 或对抗训练 (GAN 思路)。

**问题 3: 训练不稳定**

如果 encoder 输出 $\sigma$ 很小 (例如 $\sigma \to 0$), KL 项 $\supset \log \sigma^2 \to -\infty$, 梯度爆炸。

**解法**: 限制 $\log \sigma^2$ 的范围, 例如用 softplus 重参数化 $\sigma = \text{softplus}(\cdot)$。

### 5.10 VAE 推导的另一种视角:信息论

**信息论角度**: ELBO = 重建项 - KL = 最大化互信息 $I(x; z)$ 的下界。

具体地:

$$\text{ELBO} = \underbrace{\mathbb{E}_q[\log p(x \mid z)]}_{\text{重建}} - \underbrace{D_{\text{KL}}(q(z \mid x) \| p(z))}_{\text{正则}}$$

**信息瓶颈** (Information Bottleneck, IB) 视角: 找 $z$ 使 $I(x; z)$ 大 (保留信息) 同时 $I(z; y)$ 大 (预测标签)。

VAE 的 ELBO 与 IB 的目标在某种意义下是**对偶**——这个连接在第 5 章 (信息论) 的 IB 章节会详细展开。

---

## 6. 应用 2:EM 算法与变分 (220 行)

### 6.1 EM 算法:含隐变量模型的 MLE

**问题**: 数据 $x$ 由含隐变量 $z$ 的模型生成, 找参数 $\theta$ 最大化 $\log p(x \mid \theta)$。

$$\log p(x \mid \theta) = \log \int p(x, z \mid \theta) dz$$

**这个积分通常没法算** (隐变量空间可能很大)。

### 6.2 E 步:固定 $\theta$, 算隐变量后验

$$q^*(z) = p(z \mid x, \theta^{(t)}) = \frac{p(x \mid z, \theta^{(t)}) p(z \mid \theta^{(t)})}{p(x \mid \theta^{(t)})}$$

这步计算 $q^*(z)$, 在变分法看来是"在分布空间找一个**特定的** $q$", 即真实后验。

### 6.3 M 步:固定 $q$, 最大化参数 $\theta$

$$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{z \sim q^*}[\log p(x, z \mid \theta)]$$

这步用 $q^*$ 算出来的期望去更新 $\theta$。

### 6.4 案例:GMM 的 EM,手算 2 个高斯、5 个数据点

**设定**:
- 2 个高斯: $\mu_1, \sigma_1^2, \pi_1$ 和 $\mu_2, \sigma_2^2, \pi_2$
- 5 个数据点: $x = \{1.0, 1.5, 2.0, 8.0, 9.0\}$ (明显两簇)
- 初始: $\mu_1 = 2, \sigma_1 = 1, \pi_1 = 0.5$, $\mu_2 = 8, \sigma_2 = 1, \pi_2 = 0.5$

**E 步** (固定参数, 算每个点属于每个高斯的"责任" $\gamma_{ik} = p(z_i = k \mid x_i)$):

对 $x_1 = 1.0$:
$$\gamma_{11} = \frac{\pi_1 \mathcal{N}(1.0 \mid 2, 1)}{\pi_1 \mathcal{N}(1.0 \mid 2, 1) + \pi_2 \mathcal{N}(1.0 \mid 8, 1)}$$
$$= \frac{0.5 \cdot \frac{1}{\sqrt{2\pi}} e^{-0.5 \cdot 1^2}}{0.5 \cdot \frac{1}{\sqrt{2\pi}} e^{-0.5} + 0.5 \cdot \frac{1}{\sqrt{2\pi}} e^{-0.5 \cdot 49}}$$
$$\approx \frac{0.5 \cdot 0.2420}{0.5 \cdot 0.2420 + 0.5 \cdot e^{-24.5}} \approx 1.0$$

类似的, $x_2, x_3$ 都约属于 cluster 1; $x_4, x_5$ 约属于 cluster 2。

**M 步** (固定 $\gamma$, 更新 $\mu_k, \sigma_k^2, \pi_k$):

$$N_k = \sum_i \gamma_{ik}, \quad \mu_k = \frac{\sum_i \gamma_{ik} x_i}{N_k}, \quad \sigma_k^2 = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)^2}{N_k}$$

更新后: $\mu_1 \approx 1.5, \mu_2 \approx 8.5$ (向真实聚类中心靠拢)。

### 6.5 VAE 与 EM 的对比

| 维度 | EM | VAE |
| :--- | :--- | :--- |
| 隐变量 | 离散 (cluster id) 或低维 | 高维连续 |
| E 步 | 计算真实后验 $p(z \mid x, \theta)$ | **近似** 后验 $q_\phi(z \mid x)$ |
| M 步 | 闭式更新 | 梯度上升 |
| 是否变分 | 后验精确时不是 | **必须变分** (后验难算) |
| 收敛保证 | 一定收敛 (但到局部最优) | ELBO 单调不降, 局部最优 |

**数学本质相同**: 都在隐变量分布 $q(z)$ 上做**交替优化**。区别在于 EM 用**精确** E 步 (假设后验可解), VAE 用**变分** E 步 (后验不可解, 用参数化 $q_\phi$ 近似)。

### 6.6 为什么 EM 算法一定收敛 (但可能到局部最优)

**定理**: EM 算法的对数似然 $\log p(x \mid \theta^{(t)})$ 单调**非降**:
$$\log p(x \mid \theta^{(t+1)}) \geq \log p(x \mid \theta^{(t)})$$

**证明概要**:
1. 用 Jensen 不等式把 $\log p(x \mid \theta)$ 写成 ELBO + KL(真实后验 || $q$)
2. E 步令 $q = p(z \mid x, \theta^{(t)})$, KL 项 = 0, ELBO = $\log p(x \mid \theta^{(t)})$
3. M 步最大化 ELBO, 所以 $\log p(x \mid \theta^{(t+1)}) \geq \text{ELBO}^{(t)} = \log p(x \mid \theta^{(t)})$

但因为 ELBO 关于 $\theta$ 是**非凸**的, M 步可能只找到局部极大。所以最终收敛到**局部最优**, 不保证全局最优。

### 6.7 ELBO 单调非降的完整证明

**定理** (Dempster 1977): 任意 E 步 + M 步后, $\log p(x \mid \theta^{(t+1)}) \geq \log p(x \mid \theta^{(t)})$。

**证明**:

**Step 1**: 对任意分布 $q(z)$, Jensen 不等式给出:

$$\log p(x \mid \theta) = \log \int q(z) \frac{p(x, z \mid \theta)}{q(z)} dz \geq \int q(z) \log \frac{p(x, z \mid \theta)}{q(z)} dz = \text{ELBO}(q, \theta)$$

**Step 2**: E 步取 $q^{(t)}(z) = p(z \mid x, \theta^{(t)})$:

$$\text{ELBO}(q^{(t)}, \theta^{(t)}) = \int p(z \mid x, \theta^{(t)}) \log \frac{p(x, z \mid \theta^{(t)})}{p(z \mid x, \theta^{(t)})} dz$$

$$= \int p(z \mid x, \theta^{(t)}) \log p(x \mid \theta^{(t)}) dz = \log p(x \mid \theta^{(t)})$$

**Step 3**: M 步找 $\theta^{(t+1)} = \arg\max_\theta \text{ELBO}(q^{(t)}, \theta)$, 所以:

$$\text{ELBO}(q^{(t)}, \theta^{(t+1)}) \geq \text{ELBO}(q^{(t)}, \theta^{(t)}) = \log p(x \mid \theta^{(t)})$$

**Step 4**: 由 $\log p(x \mid \theta^{(t+1)}) \geq \text{ELBO}(q^{(t)}, \theta^{(t+1)})$:

$$\boxed{\log p(x \mid \theta^{(t+1)}) \geq \log p(x \mid \theta^{(t)})}$$

证毕。$\square$

**注意**: 这个证明**不要求** M 步找到全局最大, 任何**上升** $\theta$ 都满足单调性。但实际中, 局部 M 步可能让收敛变慢。

### 6.8 GMM 完整 EM 例子

```python
import numpy as np

def gmm_em_full(x, K=2, steps=20, seed=42):
    """GMM 完整 EM, 打印每步结果"""
    np.random.seed(seed)
    N = len(x)

    # 初始化 (故意选不好, 让 EM 来修正)
    mu = np.array([float(x.min()), float(x.max())])  # 极端值
    sigma = np.ones(K) * (x.std() * 2)  # 大方差
    pi = np.ones(K) / K

    print(f"{'Step':>4} | {'mu':>20} | {'sigma':>20} | {'log_lik':>10}")
    print("-" * 70)

    for step in range(steps):
        # E 步: 算责任
        log_gamma = np.zeros((N, K))
        for k in range(K):
            log_gamma[:, k] = np.log(pi[k] + 1e-10) - 0.5 * np.log(2 * np.pi) - np.log(sigma[k]) - 0.5 * ((x - mu[k]) / sigma[k])**2
        log_gamma -= log_gamma.max(axis=1, keepdims=True)  # 数值稳定
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # 计算对数似然
        log_lik = 0
        for i in range(N):
            ll = 0
            for k in range(K):
                ll += pi[k] * np.exp(-0.5 * ((x[i] - mu[k]) / sigma[k])**2) / (sigma[k] * np.sqrt(2 * np.pi))
            log_lik += np.log(ll + 1e-10)

        print(f"{step:>4} | {str(mu.round(3)):>20} | {str(sigma.round(3)):>20} | {log_lik:>10.4f}")

        # M 步: 更新参数
        N_k = gamma.sum(axis=0) + 1e-10
        mu = (gamma * x[:, None]).sum(axis=0) / N_k
        sigma = np.sqrt((gamma * (x[:, None] - mu)**2).sum(axis=0) / N_k) + 1e-6
        pi = N_k / N

    return mu, sigma, pi, gamma


# 测试: 两簇明显的数据
x = np.array([1.0, 1.5, 2.0, 8.0, 9.0])
gmm_em_full(x, K=2, steps=10)
```

**预期输出**: $\mu$ 从 $[1.0, 9.0]$ 收敛到 $\approx [1.5, 8.5]$, $\sigma$ 从大变小, 对数似然单调上升。

### 6.9 EM 的变种: GEM、VAEM、EM 变分推断

**广义 EM (Generalized EM, GEM)**: M 步不要求找到 ELBO 的精确最大值, 只需**增加** ELBO 即可:

$$\theta^{(t+1)}: \text{ELBO}(\theta^{(t+1)}) \geq \text{ELBO}(\theta^{(t)})$$

**优点**: M 步成本低 (用 SGD 一步代替完整优化), 适合大规模问题。

**变分 EM (Variational EM)**: E 步用**变分分布** $q(z)$ 近似真实后验 (因为后验难算), 然后在变分下做 EM。这正是 **VAE 训练**的思路——E 步用 $q_\phi(z \mid x)$ 近似, M 步更新 decoder $\theta$。

**VAE = 变分 EM 的随机版本**:
- E 步: $q_\phi(z \mid x)$ 近似后验 (变分)
- M 步: 用 reparameterization 算 ELBO 关于 $\theta$ 的梯度 (随机)
- 这是**随机变分 EM (Stochastic Variational EM)**

### 6.10 EM 与 VAE 的统一视角

EM 和 VAE **都在做同一件事**: 在**隐变量分布** $q(z)$ 上做优化, 然后更新**模型参数** $\theta$。

| 步骤 | EM (经典) | VAE (变分) |
| :--- | :--- | :--- |
| E 步 | $q = p(z \mid x, \theta)$ 精确 | $q_\phi(z \mid x)$ 变分近似 |
| M 步 | 闭式更新 $\theta$ | 梯度上升更新 $\theta$ |
| 训练 | 交替, 收敛保证 | 联合梯度下降 (reparameterization) |
| 适用 | 后验可解 | 后验不可解 |

**统一目标**: 都在优化 **ELBO** (变分下界)。EM 是 ELBO 取等的特例 (变分精确), VAE 是 ELBO 取近似的特例 (变分近似)。

**重要洞察**: 这告诉我们 VAE 不是"特殊发明", 而是 **EM 算法框架在深度学习时代的自然推广**。从 EM 到 VAE 的演进, 是把"闭式 E 步"换成"神经网络 + 重参数化"。

---

## 7. 应用 3:最优控制与强化学习 (200 行)

### 7.1 最优控制问题

**设定**: 系统动力学 $\dot{x}(t) = f(x(t), u(t), t)$, 代价函数:

$$J[u] = \int_0^T L(x(t), u(t), t) dt + \Phi(x(T))$$

**目标**: 找控制轨迹 $u^*(t)$ 使 $J$ 最小。

**本质**: 这是个**带约束的变分问题**——约束是动力学 ODE。

### 7.2 Hamilton-Jacobi-Bellman (HJB) 方程

**值函数** $V(x, t) = $ 从状态 $x$ 时刻 $t$ 出发到终止的最优代价:

$$V(x, t) = \min_u \left[ \int_t^T L(x, u, s) ds + \Phi(x(T)) \right]$$

**HJB 方程** (最优控制的偏微分方程):

$$-\frac{\partial V}{\partial t} = \min_u \left[ L(x, u, t) + \nabla_x V \cdot f(x, u, t) \right]$$

$$V(x, T) = \Phi(x)$$

**直觉**: "当前价值" = "未来最优代价"。右边取 $\min$ 是因为我们要选最优 $u$。

**与 Euler-Lagrange 的关系**: HJB 是**最优控制**的 Euler-Lagrange 等价形式。它通过引入值函数 $V$ 把 ODE 约束的变分问题**转化为 PDE**。

### 7.3 Pontryagin 极大值原理

Pontryagin 极大值原理是另一个角度: 引入**协态** (co-state) $\lambda(t)$, 构造 **Hamiltonian**:

$$H(x, u, \lambda, t) = L(x, u, t) + \lambda^T f(x, u, t)$$

**必要条件**:

$$\dot{x}^* = \frac{\partial H}{\partial \lambda} = f(x^*, u^*, t), \quad \dot{\lambda}^* = -\frac{\partial H}{\partial x}$$

$$u^*(t) = \arg\min_u H(x^*, u, \lambda^*, t)$$

这是**两点边值问题** (Two-Point Boundary Value Problem): $\dot{x}$ 给出初值条件 $x(0) = x_0$, $\dot{\lambda}$ 给出终值条件 $\lambda(T) = \nabla \Phi(x(T))$。

### 7.4 连续控制 (MuJoCo) vs 离散控制 (Atari)

**连续控制** (HalfCheetah, Walker):
- 状态 $s \in \mathbb{R}^d$ (关节角度、角速度)
- 动作 $a \in \mathbb{R}^m$ (扭矩)
- 策略 $\pi: \mathbb{R}^d \to \mathbb{R}^m$ (回归)
- 数学模型: **ODE + 连续控制**
- 算法: SAC, TD3, PPO (连续版), DreamerV3 (RSSM)

**离散控制** (Atari):
- 状态 $s$: 图像 (像素)
- 动作 $a$: 离散 (按键)
- 策略 $\pi: s \to \Delta(\mathcal{A})$ (Categorical 分布)
- 数学模型: **MDP + 离散动作**
- 算法: DQN, PPO, A2C

**DreamerV3 的 RSSM** (Recurrent State-Space Model, 第 7 章 8 案例) = 用 VAE 学**隐状态** + RNN 学**动力学** + 连续控制。

### 7.5 RL:策略 $\pi(a \mid s)$ 是"控制"

RL 的策略 $\pi(a \mid s)$ 是一个**函数** (从状态到动作分布)。

**策略梯度定理**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau) R(\tau)]$$

这是**函数空间**上的梯度 (策略 $\pi$ 是函数)。

**变分视角**:
- TRPO/PPO 优化 $\pi$ 使 $J(\pi)$ 最大 = 变分
- SAC (Soft Actor-Critic) 在策略上加**熵正则** = **最大熵 RL** (用变分法, 约束下最大化期望奖励 + 熵)

### 7.6 DreamerV3 = 变分法 + 最优控制

DreamerV3 (Hafner 2023) 的核心:

1. **RSSM (Recurrent State-Space Model)** = **VAE 风格的隐变量模型**, 学一个隐状态 $z_t$ 表示历史
2. **学习动力学** $p(z_{t+1} \mid z_t, a_t)$ = 学 ODE 的离散版
3. **在隐空间做 Actor-Critic** = **最优控制** (HJB 的近似解)

整个 pipeline = **变分推断 (学隐变量) + 最优控制 (在隐空间做规划)**。

### 7.7 Pontryagin 极大值原理的推导

考虑**带约束变分**: 找 $u(t)$ 最小化 $J[u] = \int_0^T L(x, u, t) dt$, 约束 $\dot{x} = f(x, u, t)$, $x(0) = x_0$, $x(T)$ 自由。

**变分法的工具**: 用 Lagrange 乘子把约束塞进目标。

**Step 1**: 引入协态 $\lambda(t)$, 构造 Hamiltonian:

$$H(x, u, \lambda, t) = L(x, u, t) + \lambda^T f(x, u, t)$$

**Step 2**: 构造增广泛函:

$$\tilde{J}[x, u, \lambda] = \int_0^T \left[ L(x, u, t) + \lambda^T (\dot{x} - f(x, u, t)) \right] dt$$

**Step 3**: 对 $x$ 变分, 取 $\delta x$ 在端点 $\delta x(0) = 0$ (因为 $x(0)$ 固定), $\delta x(T)$ 任意 (因为 $x(T)$ 自由):

$$\delta \tilde{J} = \int_0^T \left[ \frac{\partial H}{\partial x} \delta x + \lambda^T \delta \dot{x} \right] dt$$

分部积分 $\int \lambda^T \delta \dot{x} dt = [\lambda^T \delta x]_0^T - \int \dot{\lambda}^T \delta x \, dt$:

$$\delta \tilde{J} = \int_0^T \left[ \frac{\partial H}{\partial x} + \dot{\lambda} \right]^T \delta x \, dt + \lambda(T)^T \delta x(T)$$

要求 $\delta \tilde{J} = 0$ 对所有 $\delta x, \delta x(T)$:

- $\dot{\lambda} = -\frac{\partial H}{\partial x}$
- 终值: $\lambda(T) = 0$ (因为 $x(T)$ 自由)

**Step 4**: 对 $u$ 变分 (无约束变分), 内部最优点条件:

$$\frac{\partial H}{\partial u} = 0$$

即 $u^*(t) = \arg\min_u H(x^*(t), u, \lambda^*(t), t)$。

**Pontryagin 极大值原理总结**:

$$\boxed{\begin{cases}
\dot{x}^* = \frac{\partial H}{\partial \lambda} = f(x^*, u^*, t) & \text{动力学} \\
\dot{\lambda}^* = -\frac{\partial H}{\partial x} & \text{协态方程} \\
u^*(t) = \arg\min_u H(x^*, u, \lambda^*, t) & \text{最优控制} \\
\lambda(T) = \nabla_x \Phi(x(T)) & \text{终值条件}
\end{cases}}$$

**这是个两点边值问题**: $\dot{x}$ 给初值 $x(0) = x_0$, $\dot{\lambda}$ 给终值 $\lambda(T) = \nabla \Phi(x(T))$。需要**反时间**积分协态方程, 或者用**打靶法** (shooting method) 数值求解。

### 7.8 连续控制的反向传播 (Backpropagation Through Time)

连续控制的"反向传播"通过 ODE 求解器自动完成:

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

# 动力学: dx/dt = f(x, u, t)
class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # state (2) + control (1)
            nn.Tanh(),
            nn.Linear(64, 2),  # dx/dt
        )

    def forward(self, t, x):
        # 注意: u 在外部给定, 这里仅用作参数
        # 实际中 u = policy(x), 整套可微
        u = torch.zeros(x.shape[0], 1, device=x.device)
        inp = torch.cat([x, u], dim=-1)
        return self.net(inp)


# Neural ODE 风格的状态演化
def continuous_policy_gradient(steps=100):
    dynamics = Dynamics()
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-3)

    for step in range(steps):
        # 初始状态: 随机
        x0 = torch.randn(32, 2)
        t = torch.linspace(0, 1, 10)

        # 前向: 解 ODE
        x_traj = odeint(dynamics, x0, t, method='dopri5')
        # x_traj shape: (10, 32, 2)  时间 × 批次 × 状态

        # 代价: 到达 (1, 0) + 控制能量
        target = torch.tensor([1.0, 0.0])
        cost = ((x_traj[-1] - target)**2).sum() + 0.01 * (x_traj**2).sum()

        # 反向: 整个轨迹可微
        optimizer.zero_grad()
        cost.backward()  # 自动微分通过 ODE solver
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: cost = {cost.item():.4f}")
```

**关键**: `torchdiffeq` 用 adjoint method, 内存复杂度是 $O(1)$ 而不是 $O(T)$。这正是 Pontryagin 极大值原理在工程中的实现——**反时间积分协态方程**。

### 7.9 Model Predictive Control (MPC):实际工程方案

**问题**: HJB 方程和 Pontryagin 极大值原理给出**理论**最优控制, 但实际系统**难精确积分**。

**MPC**: 在每个时刻 $t$, 用**模型**预测未来 $H$ 步的轨迹, 求最优控制序列, 执行第一步, 然后滚动。

```python
import torch
import numpy as np


class CartPoleMPC:
    """简化版 CartPole MPC: 在每步规划未来"""

    def __init__(self, horizon=20, n_samples=100):
        self.H = horizon  # 预测步数
        self.N = n_samples  # 候选控制序列数

    def dynamics(self, x, u):
        """x: (4,) [位置, 速度, 角度, 角速度], u: (1,) 力"""
        # 简化的 CartPole 动力学
        g, m, M, l = 9.8, 0.1, 1.0, 0.5
        pos, vel, theta, omega = x
        force = u.squeeze()

        theta_acc = (g * torch.sin(theta) - torch.cos(theta) * (force + m * l * omega**2 * torch.sin(theta)) / (M + m)) / (l * (4/3 - m * torch.cos(theta)**2 / (M + m)))
        x_acc = (force + m * l * (omega**2 * torch.sin(theta) - theta_acc * torch.cos(theta))) / (M + m)
        return torch.stack([vel, x_acc, omega, theta_acc])

    def cost(self, x_traj, u_traj):
        """状态误差 + 控制能量"""
        target = torch.tensor([0.0, 0.0, 0.0, 0.0])
        state_cost = ((x_traj - target)**2).sum(dim=1)
        control_cost = 0.01 * (u_traj**2).sum(dim=1)
        return state_cost + control_cost

    def plan(self, x0):
        """规划: 随机采样 N 个控制序列, 选最优的第一动作"""
        # 随机采样控制序列
        u_seqs = torch.randn(self.N, self.H, 1) * 2.0  # 控制范围 [-3, 3]

        # 模拟每条序列
        x_traj = x0.unsqueeze(0).expand(self.N, -1).clone()
        x_trajs = [x_traj.clone()]
        for t in range(self.H):
            x_traj = torch.stack([self.dynamics(x_traj[i], u_seqs[i, t]) for i in range(self.N)])
            x_trajs.append(x_traj)
        x_trajs = torch.stack(x_trajs, dim=1)  # (N, H+1, 4)

        # 算代价
        costs = self.cost(x_trajs[:, :-1], u_seqs).sum(dim=1)  # (N,)

        # 选最优
        best_idx = costs.argmin()
        return u_seqs[best_idx, 0]
```

**MPC 的思想**: 不算完整 Pontryagin 解 (两点边值太难), 而是用**采样** + **局部规划**近似。这正是 CEM (Cross-Entropy Method), MPPI (Model Predictive Path Integral), PETS 等算法的核心。

### 7.10 强化学习:策略 $\pi$ 作为变分对象

**策略梯度** (REINFORCE):

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau) \cdot R(\tau)]$$

其中 $R(\tau)$ 是轨迹 $\tau$ 的累积奖励。

**变分视角**: 这是个**函数空间**的梯度。策略 $\pi_\theta(a \mid s)$ 是个函数族 (参数化为 $\theta$), 我们要最大化 $J(\pi)$ = 关于策略的泛函。

**TRPO/PPO**: 在函数空间做**信赖域优化**。KL 散度 $\text{KL}(\pi_\theta \| \pi_{\theta_{\text{old}}}) \leq \delta$ 限制策略更新幅度——KL 散度是函数空间的"距离"。

**SAC (Soft Actor-Critic)**: 最大熵 RL

$$J(\pi) = \mathbb{E}_\pi\left[\sum_t r_t + \alpha H(\pi(\cdot \mid s_t))\right]$$

其中 $H(\pi(\cdot \mid s_t))$ 是策略熵。**这是变分 RL**: 在奖励最大化和策略熵之间做权衡。

---

## 8. PyTorch 代码实现 (200 行)

### 实验 1:PyTorch 实现 VAE (含 ELBO 推导)

(见 §5.6 的代码)

### 实验 2:手算 GMM 的 EM 一步

```python
import numpy as np

def gmm_em_one_step(x, mu, sigma, pi):
    """
    GMM 的 EM 一步:
    E 步: 算责任 (responsibility) γ_ik = p(z_i=k | x_i, θ)
    M 步: 更新参数
    """
    N = len(x)
    K = len(mu)
    gamma = np.zeros((N, K))

    # E 步
    for i in range(N):
        for k in range(K):
            gamma[i, k] = pi[k] * np.exp(-0.5 * ((x[i] - mu[k]) / sigma[k])**2) / sigma[k]
        gamma[i] /= gamma[i].sum()  # 归一化

    # M 步
    N_k = gamma.sum(axis=0)  # 每个高斯的"有效样本数"
    mu_new = (gamma * x[:, None]).sum(axis=0) / N_k
    sigma_new = np.sqrt((gamma * (x[:, None] - mu_new)**2).sum(axis=0) / N_k)
    pi_new = N_k / N

    return mu_new, sigma_new, pi_new, gamma


# 测试
x = np.array([1.0, 1.5, 2.0, 8.0, 9.0])
mu = np.array([2.0, 8.0])
sigma = np.array([1.0, 1.0])
pi = np.array([0.5, 0.5])

for step in range(5):
    mu, sigma, pi, gamma = gmm_em_one_step(x, mu, sigma, pi)
    print(f"Step {step+1}: mu={mu.round(2)}, sigma={sigma.round(2)}, pi={pi.round(2)}")
    print(f"  gamma=\n{gamma.round(3)}")
```

**预期输出**:
- $\mu$ 会从 $(2, 8)$ 收敛到 $(1.5, 8.5)$
- $\sigma$ 会从 $(1, 1)$ 收敛到更小的值
- $\gamma$ 会变成接近单位矩阵 (每个点明确归属一个簇)

### 实验 3:简单最优控制 (双积分系统)

```python
import torch
import torch.nn as nn

# 双积分系统: x'' = u (加速度 = 控制)
# 状态: [位置, 速度]
# 控制: 加速度 u

def dynamics(x, u, dt=0.1):
    """x: (batch, 2), u: (batch, 1)"""
    pos = x[..., 0:1] + x[..., 1:2] * dt + 0.5 * u * dt**2
    vel = x[..., 1:2] + u * dt
    return torch.cat([pos, vel], dim=-1)


# 用神经网络学一个策略 π(s) -> u
class Policy(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),  # 把控制限制在 [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# 训练: 让小车从 x=0 移动到目标 x=10
def train_policy(steps=1000, target=10.0):
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for step in range(steps):
        # 随机初始状态
        x0 = torch.randn(64, 2) * 0.5
        x0[:, 0] -= target / 2  # 起点在目标左边

        # 模拟 T=20 步
        x = x0
        total_cost = 0
        for t in range(20):
            u = policy(x)
            cost = (x[..., 0] - target)**2 + 0.01 * u.squeeze()**2  # 位置误差 + 控制代价
            total_cost = total_cost + cost.sum()
            x = dynamics(x, u)

        optimizer.zero_grad()
        total_cost.backward()  # 通过动力学反向传播
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: cost = {total_cost.item():.2f}")


train_policy()
```

**注意**: 这里 `x = dynamics(x, u)` 是**可微分**的 (PyTorch 张量), 所以 cost 关于 policy 参数的梯度可以通过**整个轨迹**反传——这是**连续控制**和**离散控制**在代码层面的关键区别。

### 实验 4:用 `torch.autograd.functional` 验证变分

```python
import torch
import numpy as np
from torch.autograd.functional import jvp

# 定义泛函 J[y] = ∫ y^2 dx, 离散化为求和
def J(y):
    """y: (n,)"""
    return (y**2).sum() * 0.01  # 假设 dx = 0.01


# 定义一个具体的 y
y = torch.linspace(0, 1, 100, requires_grad=True)

# 定义扰动 η
eta = torch.sin(torch.linspace(0, np.pi, 100))

# 用 jvp (Jacobian-Vector Product) 算变分 δJ[y; η]
# 数学上: δJ[y; η] = lim_{ε→0} (J(y+εη) - J(y)) / ε = η^T · ∇J(y)
_, dv = jvp(J, (y,), (eta,))
print(f"δJ[y; η] = {dv.item():.6f}")

# 验证: 解析上 δJ[y; η] = 2 ∫ y · η dx
expected = 2 * (y * eta).sum() * 0.01
print(f"解析值: {expected.item():.6f}")
print(f"误差: {abs(dv.item() - expected.item()):.6e}")
```

**预期**: 数值解和解析解差 $10^{-6}$ 量级, 验证变分公式正确。

### 实验 5:用 Euler-Lagrange 解一维谐振子

```python
import torch

# 一维谐振子: L = (1/2)(y'^2 - y^2), Euler-Lagrange: y'' + y = 0
# 解: y = A cos(x) + B sin(x)

def euler_lagrange_oscillator(x_eval, A=1.0, B=0.5):
    """直接给 Euler-Lagrange 的解"""
    return A * torch.cos(x_eval) + B * torch.sin(x_eval)


# 用神经网络近似 y(x), 让 loss = |LHS - RHS|^2
def train_oscillator(steps=2000):
    y_net = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(y_net.parameters(), lr=1e-3)

    for step in range(steps):
        x = torch.linspace(0, 2 * 3.14159, 100).reshape(-1, 1).requires_grad_(True)
        y = y_net(x)

        # dy/dx 用 autograd
        dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]

        # Euler-Lagrange 残差: y'' + y = 0
        residual = d2y_dx2 + y
        loss = (residual**2).mean()

        # 边界条件: y(0) = 1, y(pi/2) = 0.5
        bc_loss = (y_net(torch.tensor(`0.0`) - 1.0)**2 +
                   y_net(torch.tensor(`1.5708`) - 0.5)**2)

        total = loss + 100 * bc_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}: residual = {loss.item():.6f}, bc = {bc_loss.item():.6f}")

    # 验证: 与解析解对比
    x_test = torch.linspace(0, 2 * 3.14159, 100).reshape(-1, 1)
    y_pred = y_net(x_test).detach()
    y_true = euler_lagrange_oscillator(x_test, A=1.0, B=0.5)
    print(f"预测 vs 解析最大误差: {(y_pred - y_true).abs().max().item():.6f}")


train_oscillator()
```

**预期**: 神经网络学的 $y(x)$ 接近解析解 $A\cos x + B\sin x$, 验证 Euler-Lagrange 方程的物理意义。

### 实验 6:用变分法解最速降线 (Brachistochrone)

```python
import torch

def train_brachistochrone(steps=3000):
    """
    最速降线: L = sqrt((1 + y'^2) / y), y(0) = 0, y(1) = 1
    解析解: 旋轮线 x = a(t - sin t), y = a(1 - cos t)
    """
    y_net = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )

    optimizer = torch.optim.Adam(y_net.parameters(), lr=1e-3)

    for step in range(steps):
        x = torch.linspace(0.001, 1.0, 100).reshape(-1, 1).requires_grad_(True)
        y = y_net(x)

        # 自动微分算 y'
        dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

        # Lagrangian: L = sqrt((1 + y'^2) / y)
        L = torch.sqrt((1 + dy_dx**2) / (y + 1e-3))

        # Euler-Lagrange: ∂L/∂y - d/dx(∂L/∂y') = 0
        # 这里直接近似: 用 Euler-Lagrange 等价形式
        # 让 (y - x) 最小作为代理 (旋轮线接近 x^{2/3} 形状)

        # 用 d/dx(L - y' ∂L/∂y') = 0 (因为 ∂L/∂x = 0)
        dL_dyp = dy_dx / torch.sqrt((1 + dy_dx**2) * (y + 1e-3))  # 偏 L / 偏 y'
        beltrami = L - dy_dx * dL_dyp  # 应为常数

        # Beltrami 恒等式残差: d/dx (L - y' ∂L/∂y') = 0
        d_beltrami = torch.autograd.grad(beltrami.sum(), x, create_graph=True)[0]
        residual = d_beltrami**2

        # 边界
        bc = (y_net(torch.tensor(`0.0`)) - 0)**2 + (y_net(torch.tensor(`1.0`)) - 1)**2

        loss = residual.mean() + 100 * bc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}: residual = {residual.mean().item():.6f}, bc = {bc.item():.6f}")

    return y_net


# 训练并对比旋轮线
brach = train_brachistochrone()
# 旋轮线解析: x = (t - sin t) / (2π), y = (1 - cos t) / 2
t = torch.linspace(0.001, 2 * 3.14159, 100)
x_cyc = (t - torch.sin(t)) / (2 * 3.14159)
y_cyc = (1 - torch.cos(t)) / 2
# 缩放使 x 从 0 到 1, y 从 0 到 1
print(f"旋轮线终点: y({x_cyc[-1]:.3f}) = {y_cyc[-1]:.3f}")
```

**预期**: 神经网络解接近旋轮线 (Cycloid), 验证变分法能解经典物理问题。

### 实验 7:完整 VAE 训练循环 (MNIST)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST 数据加载
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

# 模型 (从 §5.6)
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 训练
def train_mnist_vae(epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)

            # 重建 + KL
            recon = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {total_loss / len(mnist.dataset):.4f}")


# 运行 (需要几分钟)
# train_mnist_vae(epochs=5)
```

### 实验 8:用 PyTorch 验证 Euler-Lagrange 一维谐振子

完整代码已在 §8 实验 5 中给出。核心思想是: 用神经网络 $y_\theta(x)$ 近似解, 损失函数 = $|\text{Euler-Lagrange 残差}|^2$ + 边界条件违反。

**这是 PINN (Physics-Informed Neural Networks)** 的核心思想: 把物理方程 (Euler-Lagrange, Navier-Stokes, Maxwell) 作为损失的一部分。

---

## 9. 常见误区 (100 行, 7 条)

### ❌ 误区 1:"变分 = 变分法 = calculus of variations"

"变分" 和 "变分法" 是同一个东西的两个中文名, 都指 calculus of variations。不是"变分 = variation", "变分法 = method of variation"。**变分** 是名词,指函数的扰动; **变分法** 是方法论。

### ❌ 误区 2:"Euler-Lagrange 给出全局最优"

Euler-Lagrange 给出的是**驻点** (stationary point),不一定是极小点。可能是:
- 局部极小
- 局部极大
- 鞍点

判定需要**第二变分** $\delta^2 J$ 的符号。经典反例: $J[y] = \int_0^1 y'^2 dx$, 驻点是 $y = ax + b$ (直线), 但加上约束 $\int y dx = 1$ 后, 不一定是极小。

### ❌ 误区 3:"VAE 的 ELBO 就是 $\log p(x)$"

**ELBO 是 $\log p(x)$ 的下界** ($\leq$), 不是等于。等号成立当且仅当 $q(z) = p(z \mid x)$, 即近似分布等于真实后验。但真实后验通常不可解,所以 ELBO **严格小于** $\log p(x)$。

### ❌ 误区 4:"EM 算法一定到全局最优"

EM 保证**单调非降**, 但只保证收敛到**局部最优**。反例: GMM 在初始化不好时 (例如两个高斯初始化成相同位置), EM 会**卡在鞍点**。实践中用 K-means 初始化或多次随机重启。

### ❌ 误区 5:"变分推断 = VAE"

变分推断 (Variational Inference, VI) 是个**通用框架**: 用参数化分布 $q_\theta$ 近似难解的后验 $p(z \mid x)$。VAE 是 VI 的一种特殊形式 (用神经网络做 encoder)。其他 VI 方法: 指数族 VI, 平均场 VI, 随机梯度 VI (SVI), 黑盒 VI (BBVI)。

### ❌ 误区 6:"重参数化只是为了减少方差"

重参数化 (Reparameterization) 的**真正目的**是**让梯度穿过采样算子**。如果只用 score function estimator ($\nabla \log q$), 可以算梯度但**方差极大**。重参数化把方差降低 1-2 个数量级, 才能让 VAE 实际训练起来。

### ❌ 误区 7:"$\delta J = 0$ 对所有 $\eta$ 成立 = $y$ 是极值"

**这是必要条件, 不是充分条件**。$\delta J = 0$ 给出驻点。要确认是极小, 还要看 $\delta^2 J \geq 0$。这和有限维 $\nabla f = 0$ 但 Hessian 不定时是鞍点, 道理完全一样。

---

## 10. 面试 Q&A (80 行, 5 题 L7/L8 难度)

### Q1:VAE 为什么要变分推断? 直觉是什么?

**A**: VAE 想学 $p(x)$ (生成模型) 和 $p(z \mid x)$ (推断/编码)。直接算 $p(z \mid x) = p(x \mid z) p(z) / p(x)$ 需要算 $p(x) = \int p(x \mid z) p(z) dz$, 这个积分**指数级难**(高维)。变分推断的直觉是: **与其硬算后验, 不如引入一个简单的参数化分布 $q(z \mid x)$, 让它"贴着"真实后验**——这是"用简单的形状拟合复杂的形状"。具体数学上, 用 KL 散度构造下界 ELBO, 最大化 ELBO = 拉近 $q$ 和 $p(z \mid x)$。这是 Jensen 不等式的应用: log 是凹函数, $\log \mathbb{E}[\cdot] \geq \mathbb{E}[\log \cdot]$。

### Q2:ELBO 为什么是 $\log p(x)$ 的下界?

**A**: 核心是 Jensen 不等式 (log 是凹函数):

$$\log p(x) = \log \mathbb{E}_{q}\left[\frac{p(x, z)}{q(z)}\right] \geq \mathbb{E}_q \log \left[\frac{p(x, z)}{q(z)}\right] = \text{ELBO}$$

等号成立当且仅当 $p(x, z)/q(z)$ 是常数, 即 $q(z) \propto p(x, z) = p(z \mid x) p(x)$, 等价于 $q(z) = p(z \mid x)$。**关键点**: 最大化 ELBO 同时最大化 $\log p(x)$ (因为 ELBO 是下界), 但不需要直接算 $p(x)$。

### Q3:EM 算法为什么保证收敛?

**A**: 因为每步都**单调不降**对数似然。证明用 Jensen 不等式:

1. E 步令 $q = p(z \mid x, \theta^{(t)})$, 此时 ELBO = $\log p(x \mid \theta^{(t)})$
2. M 步更新 $\theta^{(t+1)}$ 最大化 ELBO, 所以新 ELBO $\geq$ 旧 ELBO
3. 因为 $\log p(x \mid \theta) \geq$ ELBO, 所以 $\log p(x \mid \theta^{(t+1)}) \geq \log p(x \mid \theta^{(t)})$

单调有界序列收敛, 所以 EM 收敛。**但只保证局部最优**, 因为 M 步的非凸优化可能卡在局部极大。

### Q4:Reparameterization Trick 的数学本质?

**A**: **测度变换 + 链式法则**。朴素法中 $z \sim q_\phi(z)$, 采样算子不可微。重参数化把"从 $q_\phi$ 采样"重写为:

$$z = g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$\epsilon$ 与 $\phi$ 无关, 所以 $z$ 关于 $\phi$ 是**确定性函数**。梯度可以走链式法则:

$$\nabla_\phi \mathbb{E}_\epsilon[f(g_\phi(\epsilon))] = \mathbb{E}_\epsilon[\nabla_\phi f(g_\phi(\epsilon)) \cdot \nabla_\phi g_\phi(\epsilon)]$$

数学本质: **重参数化让变分下界关于参数的梯度可计算**, 是 PyTorch 实现 VAE 的关键技术。**不重参数化方差极大, 训练不动**。

### Q5:连续控制 vs 离散控制在数学上的差异?

**A**:
- **连续控制**: 动作 $a \in \mathbb{R}^m$ 是向量, 策略 $\pi(s) = a$ 直接回归。动力学 $\dot{x} = f(x, u)$ 是 ODE, 整个轨迹**可微**, 梯度可通过动力学反传。HJB 方程 + Pontryagin 极大值原理是连续控制的解析工具。
- **离散控制**: 动作 $a$ 是离散, 策略 $\pi(a \mid s)$ 是 Categorical 分布。梯度要用 **score function / REINFORCE** 估计 (因为采样不可微)。Bellman 方程是离散控制的解析工具。
- **联系**: 离散可以看成"连续控制 + argmax"; 连续可以离散化。DreamerV3 在**连续隐空间**做控制, 把离散动作转成连续向量。

---

## 11. 与其他数学工具的联系 (60 行)

**变分法是连接有限维优化和函数空间优化的桥梁**:

- **第 3 章 多元微分** → **本章 变分法** (把偏导数推广到 Gateaux 微分, 把梯度推广到函数空间的方向导数)
- **第 6 章 积分** → **本章 泛函** (泛函的核心形式是 $\int L dx$, 变分需要分部积分)
- **第 8 章 泛函分析** → **本章 变分法** (Hilbert 空间是变分法的"舞台", 变分是 Hilbert 空间上的微分)
- **第 10 章 凸分析** → **本章 含约束变分** (Lagrange 乘子在函数空间的推广)
- **3-概率论** → **本章 变分推断** (ELBO 是概率论 + 变分法的结合)
- **5-信息论** → **本章 KL 散度** (KL 在变分推断中是核心度量)
- **6-最优化** → **本章 函数优化** (把 SGD/Adam 推广到函数空间)
- **7-系统思维** → **本章 最优控制** (HJB 方程是 RL 的理论根基)

**变分法的 ML/DL 应用清单**:

| ML/DL 现象 | 变分法视角 |
| :--- | :--- |
| VAE 训练 | 最大化 ELBO = 在分布函数空间搜最优 $q$ |
| EM 算法 | 隐变量后验的交替优化 |
| 强化学习 (SAC) | 最大熵 RL = 变分约束下的奖励最大化 |
| DreamerV3 / 世界模型 | 隐空间变分推断 + 连续控制 |
| Diffusion Model | 分数函数 = 对数密度梯度, 用变分下界训练 |
| Normalizing Flow | 可逆变换保持密度, 变分下界的目标 |
| Flow Matching | 连续 Normalizing Flow, ODE 形式 |
| GFlowNet | 配分函数估计, 变分下界训练 |
| PINN | 物理方程作为损失 (Euler-Lagrange 残差) |
| Optimal Transport | Wasserstein 距离 = 变分对偶 |

**核心命题**: **变分法是 ML/DL 中"在无穷维空间找最优"问题的统一框架**。这一章是少数几个"不讲变分就看不懂 ML 论文"的章节。

### 变分法的"数学品味"

**品味 1**: 变分法的核心是把"找最优"从有限维推广到无穷维, 但思想没变——还是"梯度 = 0"或"驻点"。

**品味 2**: 分部积分是变分法的核心技巧, 它把"$\eta'$ 出现"变成"$\eta$ 出现", 从而利用基本引理得方程。

**品味 3**: 物理的 Lagrangian = 动能 - 势能, 工程的 Lagrangian = 代价 + 约束, 数学的 Lagrangian = 泛函积分的"被积函数"。三者在变分法下统一。

**品味 4**: ELBO 是"对数似然的下界", 这个 bound 既可以是**精确** (EM), 也可以是**近似** (VAE), 区别在于变分分布 $q$ 是否能精确匹配真实后验。

**品味 5**: 最优控制的 HJB 方程和 Pontryagin 极大值原理是"函数空间 Lagrange 乘子"——把约束塞进 Lagrangian, 然后变分。

**品味 6**: 变分法是**积分 + 微分**的真正交汇点——你既要算积分 (泛函的值), 又要"求导" (变分), 还要解方程 (Euler-Lagrange)。这一章是整个高数部分的"毕业考"。

---

## 附录 A:Euler-Lagrange 方程速查表

| Lagrangian $L$ | Euler-Lagrange | 解 |
| :--- | :--- | :--- |
| $y'^2$ | $y'' = 0$ | $y = ax + b$ |
| $y'^2 - y^2$ | $y'' + y = 0$ | $A\cos x + B\sin x$ |
| $y'^2 + y^2$ | $y'' - y = 0$ | $A\cosh x + B\sinh x$ |
| $\sqrt{1 + y'^2}$ | $\frac{y'}{\sqrt{1+y'^2}} = c$ | $y = ax + b$ |
| $y'^2 - y^2 + 2y$ | $y'' + y = 1$ | $1 + A\cos x + B\sin x$ |
| $y'^2 \cdot y$ (Fermat) | $y \cdot y'' = -y'^2$ | 旋轮线 (Cycloid) |

## 附录 B:变分法核心公式速查

| 概念 | 公式 |
| :--- | :--- |
| 泛函 | $J[y] = \int L(x, y, y') dx$ |
| 第一变分 | $\delta J[y; \eta] = \lim_{\epsilon \to 0} \frac{J[y + \epsilon \eta] - J[y]}{\epsilon}$ |
| Euler-Lagrange | $\frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} = 0$ |
| 含约束 EL | $\frac{\partial (L - \lambda M)}{\partial y} - \frac{d}{dx} \frac{\partial (L - \lambda M)}{\partial y'} = 0$ |
| 第二变分 | $\delta^2 J[y; \eta] = \frac{d^2}{d\epsilon^2} J[y + \epsilon \eta]\big\|_{\epsilon = 0}$ |
| ELBO | $\log p(x) \geq \mathbb{E}_q[\log p(x \mid z)] - D_{\text{KL}}(q \| p(z))$ |
| HJB | $-\frac{\partial V}{\partial t} = \min_u [L + \nabla_x V \cdot f]$ |
| Pontryagin | $\dot{x}^* = \frac{\partial H}{\partial \lambda}, \dot{\lambda}^* = -\frac{\partial H}{\partial x}$ |

## 附录 C:本章与面试常考点的对照

| 面试问题 | 本章对应小节 |
| :--- | :--- |
| "ELBO 怎么来的?" | §5.3 |
| "VAE 为什么需要重参数化?" | §5.5 |
| "EM 怎么保证收敛?" | §6.6 |
| "VAE 和 EM 的关系?" | §6.5 |
| "HJB 方程在说什么?" | §7.2 |
| "连续控制和离散控制的区别?" | §7.4 |
| "为什么最短路径是直线?" | §4.3 例 3 |
| "变分法的核心是什么?" | §3.2 + §4.2 |

## 12. 习题与解答 (15 题,从基础到 L9)

> **使用说明**: 这 15 题按难度递进,覆盖了变分法、ELBO、EM、最优控制的全部核心定理。建议:
> - 1-5: 基础 (L4-L5),用于巩固公式
> - 6-10: 中等 (L6-L7), 用于面试准备
> - 11-15: 进阶 (L8-L9), 用于研究面试或工程深度问题

### 习题 1 (L4): 最简单的泛函变分

**问题**: 算 $J[y] = \int_0^1 (y^2 + y') dx$ 的 Euler-Lagrange 方程。

**解答**:

Lagrangian: $L = y^2 + y'$。

计算偏导:
- $\frac{\partial L}{\partial y} = 2y$
- $\frac{\partial L}{\partial y'} = 1$
- $\frac{d}{dx} \frac{\partial L}{\partial y'} = 0$

代入 Euler-Lagrange 方程 $\frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} = 0$:

$$2y - 0 = 0 \implies y = 0$$

**结论**: 驻点是 $y \equiv 0$, 此时 $J[0] = 0$。验证: 对任意 $y$, $J[y] = \int y^2 dx + \int y' dx = \int y^2 dx + y(1) - y(0) \geq y(1) - y(0)$。如果 $y(0) = 0, y(1) = 0$ (Dirichlet), 那 $J \geq 0$, $y \equiv 0$ 是极小。

**洞察**: 这个泛函没有"动能项" ($y'^2$),只有势能 $y^2$,所以极值点就是势能极小的点 ($y = 0$)。

### 习题 2 (L4): 受迫振子的 Euler-Lagrange

**问题**: Lagrangian $L = \frac{1}{2}(y'^2 - y^2) + y$, 求 Euler-Lagrange 方程并解。

**解答**:

偏导:
- $\frac{\partial L}{\partial y} = -y + 1$
- $\frac{\partial L}{\partial y'} = y'$
- $\frac{d}{dx} \frac{\partial L}{\partial y'} = y''$

代入: $-y + 1 - y'' = 0$, 即 $y'' + y = 1$。

**求通解**: 齐次解 $y_h = A\cos x + B\sin x$, 特解 $y_p = 1$ (因为 $y_p'' = 0$,代入得 $-y_p + 1 = 0 \implies y_p = 1$)。

**通解**: $y = 1 + A\cos x + B\sin x$。

**物理意义**: 这是**受迫简谐振子**，单位恒力 + 弹性回复力。平衡点从 $0$ 移到 $1$ (恒力把"自然位置"推到 1)。

### 习题 3 (L5): VAE 的 ELBO 推导

**问题**: 严格推导 VAE 的 ELBO, 写出每一步用了什么数学性质。

**解答**:

从 $\log p(x)$ 出发:

**Step 1**: 引入任意分布 $q(z)$:

$$\log p(x) = \log \int p(x, z) dz = \log \int q(z) \cdot \frac{p(x, z)}{q(z)} dz$$

**Step 2**: 写成期望形式 (测度变换):

$$= \log \mathbb{E}_{z \sim q}\left[\frac{p(x, z)}{q(z)}\right]$$

**Step 3**: 用 Jensen 不等式 (log 是凹函数, 见第 6 章):

$$\geq \mathbb{E}_{z \sim q} \log \left[\frac{p(x, z)}{q(z)}\right]$$

**Step 4**: 展开 $p(x, z) = p(x \mid z) p(z)$:

$$= \mathbb{E}_q[\log p(x \mid z) + \log p(z) - \log q(z)]$$

**Step 5**: 拆成两项:

$$= \underbrace{\mathbb{E}_q[\log p(x \mid z)]}_{\text{重建项}} - \underbrace{D_{\text{KL}}(q(z) \| p(z))}_{\text{正则项}}$$

**结论**: $\log p(x) \geq \text{ELBO}(q)$。

**等号条件**: Jensen 不等式取等要求随机变量是常数, 即 $\frac{p(x, z)}{q(z)} = c$ 对所有 $z$ 成立。所以 $q(z) \propto p(x, z)$, 即 $q(z) = p(z \mid x)$。

### 习题 4 (L5): 指数族 KL 最小化的解

**问题**: 设 $p(x)$ 是给定分布, $q_\theta(x) = h(x) \exp(\theta^\top \phi(x) - A(\theta))$ 是指数族。最小化 $D_{\text{KL}}(p \| q_\theta)$ 关于 $\theta$,证明 $\nabla_\theta A(\theta) = \mathbb{E}_{p}[\phi(x)]$。

**解答**:

$D_{\text{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx = \int p(x) [\log p(x) - \log q_\theta(x)] dx$

展开 $\log q_\theta(x) = \log h(x) + \theta^\top \phi(x) - A(\theta)$:

$$= \int p(x) \log p(x) dx - \int p(x) \log h(x) dx - \theta^\top \int p(x) \phi(x) dx + A(\theta) \int p(x) dx$$

注意 $\int p(x) dx = 1$:

$$= \text{const}(p) - \theta^\top \mathbb{E}_p[\phi(x)] + A(\theta)$$

对 $\theta$ 求导:

$$\nabla_\theta D_{\text{KL}} = -\mathbb{E}_p[\phi(x)] + \nabla_\theta A(\theta) = 0$$

$$\implies \nabla_\theta A(\theta) = \mathbb{E}_p[\phi(x)] \quad \blacksquare$$

**ML 联系**: 高斯分布 $\mathcal{N}(\mu, \sigma^2)$ 的指数族表示: $\phi(x) = (x, x^2)$, $\theta = (\mu/\sigma^2, -1/(2\sigma^2))$, $A(\theta) = \mu^2/(2\sigma^2) + \log \sigma$。最小化 KL = 让 $\nabla_\theta A(\theta) = (\mu, \mu^2 + \sigma^2)$,即匹配均值和二阶矩。

### 习题 5 (L6): 二次型泛函的 Euler-Lagrange

**问题**: $J[y] = \int_0^1 \left( \frac{1}{2} p(x) y'^2 + \frac{1}{2} q(x) y^2 - r(x) y \right) dx$, 其中 $p, q, r$ 是给定函数。求 Euler-Lagrange 方程。

**解答**:

偏导:
- $\frac{\partial L}{\partial y} = q(x) y - r(x)$
- $\frac{\partial L}{\partial y'} = p(x) y'$
- $\frac{d}{dx} \frac{\partial L}{\partial y'} = p'(x) y' + p(x) y''$

Euler-Lagrange: $q y - r - p' y' - p y'' = 0$, 即:

$$-\frac{d}{dx}\left(p(x) \frac{dy}{dx}\right) + q(x) y = r(x)$$

**这是 Sturm-Liouville 方程**, 出现在:
- 量子力学 (Schrödinger 方程的定态形式)
- 弦振动 (波动方程分离变量后)
- 热传导 (分离变量后)

**边界条件**:
- Dirichlet: $y(0) = y(1) = 0$
- Neumann: $\left. p y' \right|_{0,1} = 0$
- Robin: $y + \alpha y' = 0$

### 习题 6 (L6): KL 散度的对称性反驳

**问题**: 证明 $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$, 给出反例。

**解答**:

设 $p = \text{Bernoulli}(0.5)$, $q = \text{Bernoulli}(0.1)$。

$D_{\text{KL}}(p \| q) = 0.5 \log \frac{0.5}{0.1} + 0.5 \log \frac{0.5}{0.9} \approx 0.5 \times 1.609 + 0.5 \times (-0.588) = 0.510$

$D_{\text{KL}}(q \| p) = 0.1 \log \frac{0.1}{0.5} + 0.9 \log \frac{0.9}{0.5} \approx 0.1 \times (-1.609) + 0.9 \times 0.588 = -0.161 + 0.529 = 0.368$

差 $0.510 - 0.368 = 0.142 \neq 0$。

**结论**: KL 不对称。

**直觉**: $D_{\text{KL}}(p \| q)$ "零强制" $q$ 在 $p$ 不为 0 的地方都覆盖; $D_{\text{KL}}(q \| p)$ "质量覆盖" $q$ 在 $p$ 为 0 的地方也收缩到 0。两种语义不同。

### 习题 7 (L6): GMM 的 E 步手算

**问题**: GMM 有 2 个高斯, $\mu_1 = 0, \sigma_1 = 1, \pi_1 = 0.5$; $\mu_2 = 3, \sigma_2 = 1, \pi_2 = 0.5$。数据点 $x = 1.5$。算 E 步的责任 $\gamma_1, \gamma_2$。

**解答**:

$\gamma_k = p(z = k \mid x, \theta) = \frac{\pi_k \mathcal{N}(x \mid \mu_k, \sigma_k^2)}{\sum_{k'} \pi_{k'} \mathcal{N}(x \mid \mu_{k'}, \sigma_{k'}^2)}$

计算似然:

$\pi_1 \mathcal{N}(1.5 \mid 0, 1) = 0.5 \cdot \frac{1}{\sqrt{2\pi}} e^{-1.5^2 / 2} = 0.5 \cdot 0.3989 \cdot 0.3247 \approx 0.0648$

$\pi_2 \mathcal{N}(1.5 \mid 3, 1) = 0.5 \cdot \frac{1}{\sqrt{2\pi}} e^{-(1.5-3)^2 / 2} = 0.5 \cdot 0.3989 \cdot e^{-1.125} = 0.5 \cdot 0.3989 \cdot 0.3247 \approx 0.0648$

(巧合,因为 $|1.5 - 0| = |3 - 1.5| = 1.5$)

所以 $\gamma_1 = \gamma_2 = 0.5$。

**验证**: 这是 $x = 1.5$ 正好在两个高斯的"对称点",所以责任各半。如果 $x = 1$ (更靠近 $\mu_1 = 0$), $\gamma_1 > 0.5$。

### 习题 8 (L7): 变分法的"反问题"，猜 Lagrangian

**问题**: 已知 Euler-Lagrange 方程 $y'' - y = 0$, 找一个 Lagrangian $L$。

**解答**:

Euler-Lagrange 是 $\frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} = 0$, 即 $\frac{\partial L}{\partial y} = \frac{d}{dx} \frac{\partial L}{\partial y'}$。

要得到 $y'' - y = 0$, 试 $L = \frac{1}{2}(y'^2 - y^2)$:

- $\frac{\partial L}{\partial y} = -y$
- $\frac{\partial L}{\partial y'} = y'$, $\frac{d}{dx} = y''$

代入: $-y - y'' = 0$, 即 $y'' + y = 0$。

不对! 符号反了。试 $L = \frac{1}{2}(y'^2 + y^2)$:

- $\frac{\partial L}{\partial y} = y$
- $\frac{d}{dx} y' = y''$

代入: $y - y'' = 0$, 即 $y'' = y$。对! ✓

**一般方法**: Euler-Lagrange 是个二阶 ODE, 假设 $L = \frac{1}{2} A y'^2 + B y y' + C y^2 + ...$ 然后对比系数。

**重要**: Lagrangian **不唯一**。可以加 $\frac{d}{dx} F(x, y)$, 变分后该项消失 (因为 $\delta \int \frac{dF}{dx} dx = \delta [F]_a^b = 0$ 当 $\delta y(a) = \delta y(b) = 0$)。

### 习题 9 (L7): HJB 方程的推导

**问题**: 写出 HJB 方程 $-\frac{\partial V}{\partial t} = \min_u [L + \nabla_x V \cdot f]$, 并解释每项的含义。

**解答**:

**值函数定义**: $V(x, t) = \min_u \int_t^T L(x, u, s) ds + \Phi(x(T))$

**Step 1**: 写成递归形式。把积分区间 $[t, T]$ 拆成 $[t, t + \Delta t]$ 和 $[t + \Delta t, T]$:

$$V(x, t) = \min_u \int_t^{t+\Delta t} L(x, u, s) ds + V(x + \Delta x, t + \Delta t)$$

**Step 2**: Taylor 展开 $V(x + \Delta x, t + \Delta t) \approx V + \nabla_x V \cdot \Delta x + \frac{\partial V}{\partial t} \Delta t + O(\Delta t^2)$。

代入 $\Delta x \approx f(x, u, t) \Delta t$:

$$V = \min_u [L \Delta t + V + \nabla_x V \cdot f \Delta t + V_t \Delta t]$$

**Step 3**: 消去 $V$ (两边都有), 除以 $\Delta t$, $\Delta t \to 0$:

$$0 = \min_u [L + \nabla_x V \cdot f + V_t]$$

即 $V_t = -\min_u [L + \nabla_x V \cdot f]$, 或:

$$-\frac{\partial V}{\partial t} = \min_u \left[ L(x, u, t) + \nabla_x V \cdot f(x, u, t) \right] \quad \blacksquare$$

**各项含义**:
- $L$: 瞬时代价
- $\nabla_x V \cdot f$: 状态变化对未来价值的影响 (Co-state term)
- $\min_u$: 选择最优控制
- $-V_t$: 价值随时间的减少率 (从 $t$ 到 $t+dt$ 损失的价值)

### 习题 10 (L7): VAE 后验坍塌的定量分析

**问题**: 证明: 如果 decoder $p(x \mid z) = \mathcal{N}(x \mid g_\theta(z), \sigma^2 I)$ 与 $z$ 无关 (即 $g_\theta(z) = c$ 是个常数), 那么最优 $q(z \mid x) = p(z)$, ELBO 中 $z$ 不携带关于 $x$ 的信息。

**解答**:

ELBO $= \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - D_{\text{KL}}(q(z \mid x) \| p(z))$

如果 $g_\theta(z) = c$ 与 $z$ 无关, 那么 $\log p(x \mid z) = \log \mathcal{N}(x \mid c, \sigma^2 I)$ 与 $z$ 无关。

ELBO $= \log p(x \mid z = \cdot) - D_{\text{KL}}(q(z \mid x) \| p(z))$

第一项与 $q$ 无关。第二项 $\geq 0$, 最小化第二项要求 $q(z \mid x) = p(z)$。

代入 $q^* = p(z)$:

ELBO$^* = \log \mathcal{N}(x \mid c, \sigma^2 I) - 0 = \log \mathcal{N}(x \mid c, \sigma^2 I)$

此时 $z$ 完全与 $x$ 独立, $q(z \mid x) = p(z)$ 不依赖 $x$。**encoder 失效**。

**这就是后验坍塌 (posterior collapse)**。

**解法**: 强制让 decoder 必须依赖 $z$, 例如:
- 加弱 decoder (限制容量)
- KL 退火 (KL 权重从 0 → 1)
- Free bits (KL 小于阈值时不优化)
- 信息瓶颈 (强制 $I(x; z) > 0$)

### 习题 11 (L8): 双重积分的变分

**问题**: Lagrangian $L = L(x, y, z, y', z')$, 推导 Euler-Lagrange 方程组。

**解答**:

**变分**: $J[y, z] = \int_a^b L(x, y, z, y', z') dx$

扰动 $y \to y + \epsilon \eta$, $z \to z + \epsilon \zeta$, $\eta, \zeta$ 任意 (在端点为 0):

$$\delta J = \epsilon \int_a^b \left[ \frac{\partial L}{\partial y} \eta + \frac{\partial L}{\partial y'} \eta' + \frac{\partial L}{\partial z} \zeta + \frac{\partial L}{\partial z'} \zeta' \right] dx$$

分部积分 $\eta'$ 和 $\zeta'$ 项 (边界项因 $\eta(a) = \eta(b) = 0$ 消失):

$$\delta J = \epsilon \int_a^b \left[ \left( \frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} \right) \eta + \left( \frac{\partial L}{\partial z} - \frac{d}{dx} \frac{\partial L}{\partial z'} \right) \zeta \right] dx$$

由基本引理 ($\eta, \zeta$ 独立):

$$\frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} = 0, \quad \frac{\partial L}{\partial z} - \frac{d}{dx} \frac{\partial L}{\partial z'} = 0$$

**这是 Hamilton 力学的出发点**: $y$ 和 $z$ 是两个广义坐标 $q_1, q_2$, Lagrangian $L(q_1, q_2, \dot{q}_1, \dot{q}_2, t)$, Euler-Lagrange 给出两个耦合的 ODE。

### 习题 12 (L8): 最速降线 (Brachistochrone) 的 Euler-Lagrange 推导

**问题**: Lagrangian $L = \sqrt{(1 + y'^2)/(2gy)}$, 起点 $y(0) = 0$, 终点 $y(x_1) = y_1$。求 Euler-Lagrange 方程。

**解答**:

Lagrangian: $L = \frac{1}{\sqrt{2g}} \cdot \frac{\sqrt{1 + y'^2}}{\sqrt{y}}$。常数 $\frac{1}{\sqrt{2g}}$ 可以忽略, 简化为 $L = \frac{\sqrt{1 + y'^2}}{\sqrt{y}}$。

注意 $L$ 不显式依赖 $x$, 用 **Beltrami 恒等式** (后面 §4.9 会详细讲):

$$L - y' \frac{\partial L}{\partial y'} = C \text{ (常数)}$$

计算:

$$\frac{\partial L}{\partial y'} = \frac{1}{\sqrt{y}} \cdot \frac{y'}{\sqrt{1 + y'^2}}$$

所以 $y' \frac{\partial L}{\partial y'} = \frac{y'^2}{\sqrt{y(1+y'^2)}}$。

$L - y' \frac{\partial L}{\partial y'} = \frac{\sqrt{1+y'^2}}{\sqrt{y}} - \frac{y'^2}{\sqrt{y(1+y'^2)}} = \frac{(1+y'^2) - y'^2}{\sqrt{y(1+y'^2)}} = \frac{1}{\sqrt{y(1+y'^2)}} = C$

所以:

$$y(1 + y'^2) = \frac{1}{C^2} = 2a \text{ (定义常数 2a)}$$

参数化解: 设 $y = a(1 - \cos\theta)$, $y' = \frac{dy}{dx} = \frac{a \sin\theta \, d\theta/dx}{1} = \frac{d\theta/dx \cdot a \sin\theta}{1}$, 代入:

$a(1 - \cos\theta) (1 + a^2 \sin^2 \theta (d\theta/dx)^2) = 2a$

化简: $(1 - \cos\theta) (1 + a^2 \sin^2\theta (d\theta/dx)^2) = 2$

使用 $1 + \cos\theta = 2\cos^2(\theta/2)$, $1 - \cos\theta = 2\sin^2(\theta/2)$:

$2\sin^2(\theta/2) (1 + a^2 \sin^2\theta (d\theta/dx)^2) = 2$

$\sin^2(\theta/2) (1 + a^2 \sin^2\theta (d\theta/dx)^2) = 1$

复杂化简后得到 **旋轮线** (Cycloid) 参数方程:

$$x = a(\theta - \sin\theta), \quad y = a(1 - \cos\theta)$$

**结论**: 最速降线**不是直线**, 而是**旋轮线** (圆周上一点的轨迹)。1696 年 Johann Bernoulli 提出, 是变分法的"开山问题"。

### 习题 13 (L8): 推导 IWAE 界比 ELBO 紧

**问题**: 证明 $\mathcal{L}_K(x) = \mathbb{E}_{q(z_{1:K} \mid x)}[\log \frac{1}{K} \sum_k \frac{p(x, z_k)}{q(z_k \mid x)}]$ 满足:
1. $\mathcal{L}_K \leq \log p(x)$ (下界)
2. $\mathcal{L}_K \leq \mathcal{L}_{K+1}$ (随 K 单调上升)
3. $\lim_{K \to \infty} \mathcal{L}_K = \log p(x)$ (极限)

**解答**:

**(1) 下界**:

对任意 $K$ 和 $z_{1:K}$:

$\log p(x) = \log \mathbb{E}_{q(z \mid x)}\left[\frac{p(x, z)}{q(z \mid x)}\right]$

$= \log \mathbb{E}_{z_{1:K} \sim q}\left[\frac{1}{K} \sum_k \frac{p(x, z_k)}{q(z_k \mid x)}\right]$

$\geq \mathbb{E}_{z_{1:K} \sim q} \log \left[\frac{1}{K} \sum_k \frac{p(x, z_k)}{q(z_k \mid x)}\right] = \mathcal{L}_K(x)$ (Jensen)

**(2) 单调性**:

$\mathcal{L}_{K+1} = \mathbb{E}_{q(z_{1:K+1} \mid x)} \log \left[\frac{1}{K+1} \sum_{k=1}^{K+1} w_k\right]$, 其中 $w_k = \frac{p(x, z_k)}{q(z_k \mid x)}$。

注意: $\frac{1}{K+1} \sum_{k=1}^{K+1} w_k = \frac{K}{K+1} \cdot \frac{1}{K} \sum_{k=1}^K w_k + \frac{1}{K+1} w_{K+1}$。

设 $\bar{w}_K = \frac{1}{K} \sum_{k=1}^K w_k$, 那 $\frac{1}{K+1} \sum_{k=1}^{K+1} w_k = \frac{K}{K+1} \bar{w}_K + \frac{1}{K+1} w_{K+1}$。

取 $\log$, 用 $\log$ 的凹性:

$\log\left[\frac{K}{K+1} \bar{w}_K + \frac{1}{K+1} w_{K+1}\right] \geq \frac{K}{K+1} \log \bar{w}_K + \frac{1}{K+1} \log w_{K+1}$

对 $w_{K+1}$ 取期望 (与 $w_{1:K}$ 独立):

$\mathbb{E}_{w_{K+1}}\left[\frac{K}{K+1} \log \bar{w}_K + \frac{1}{K+1} \log w_{K+1}\right] = \frac{K}{K+1} \log \bar{w}_K + \frac{1}{K+1} \mathbb{E}[\log w_{K+1}]$

$\geq \frac{K}{K+1} \log \bar{w}_K + \frac{1}{K+1} \log \mathbb{E}[w_{K+1}]$ (Jensen)

$= \frac{K}{K+1} \log \bar{w}_K + \frac{1}{K+1} \log 1 = \frac{K}{K+1} \log \bar{w}_K$

其中用了 $\mathbb{E}_{q(z \mid x)}[w] = \mathbb{E}[\frac{p(x,z)}{q(z \mid x)}] = p(x) = 1$ (实际中 $w$ 是密度比)。

取 $z_{1:K}$ 的期望:

$\mathcal{L}_{K+1} \geq \mathbb{E}_{z_{1:K}}\left[\frac{K}{K+1} \log \bar{w}_K\right] = \frac{K}{K+1} \mathcal{L}_K$

但 $\frac{K}{K+1} < 1$, 这给的是反方向! 修正: 实际上 Burda et al. 2016 的原始证明用**重要性权重视角**:

$\frac{1}{K+1} \sum_{k=1}^{K+1} w_k = \frac{1}{K} \sum_{k=1}^{K+1} \frac{K}{K+1} w_k$

把 $\frac{1}{K} \sum_{k=1}^{K+1} \tilde{w}_k$ 其中 $\tilde{w}_k = \frac{K}{K+1} w_k$ (重整化的重要性权重), 用类似推导得 $\mathcal{L}_{K+1} \geq \mathcal{L}_K$。

**(3) 极限**: 大数定律 + Jensen 反向 + 单调有界 = 极限 $\log p(x)$。

**完整证明**: 见 Burda, Grosse, Salakhutdinov (2016) "Importance Weighted Autoencoders"。

### 习题 14 (L9): Pontryagin 极大值原理完整推导 (含终值约束)

**问题**: 系统 $\dot{x} = f(x, u)$, 代价 $J = \int_0^T L(x, u, t) dt + \Phi(x(T))$, 约束 $x(0) = x_0$, $x(T) = x_T$ (固定终值)。推导最优性的必要条件。

**解答**:

**Step 1**: 引入协态 $\lambda(t)$, 构造 Hamiltonian:

$$H(x, u, \lambda, t) = L(x, u, t) + \lambda^T f(x, u, t)$$

**Step 2**: 增广泛函:

$$\tilde{J}[x, u, \lambda] = \int_0^T [L + \lambda^T(\dot{x} - f)] dt + \Phi(x(T))$$

**Step 3**: 变分。$\delta x(0) = 0$ (固定初值), $\delta x(T) = 0$ (固定终值, 与 §7.7 不同!):

$$\delta \tilde{J} = \int_0^T \left[ \frac{\partial H}{\partial x} \delta x + \lambda^T \delta \dot{x} + \frac{\partial H}{\partial u} \delta u \right] dt$$

分部积分 $\int \lambda^T \delta \dot{x} dt = [\lambda^T \delta x]_0^T - \int \dot{\lambda}^T \delta x \, dt$。因 $\delta x(0) = \delta x(T) = 0$, 边界项消失:

$$\delta \tilde{J} = \int_0^T \left[ \left( \frac{\partial H}{\partial x} + \dot{\lambda} \right)^T \delta x + \frac{\partial H}{\partial u} \delta u \right] dt$$

**Step 4**: 对 $\delta x, \delta u$ 独立变分:

- $\dot{\lambda} = -\frac{\partial H}{\partial x}$
- $\frac{\partial H}{\partial u} = 0$

**Step 5**: $\lambda(0)$ 自由 (因 $x(0)$ 固定但边界项消掉了), $\lambda(T)$ 自由 (因 $x(T)$ 固定)。

**两点边值**:

$$\boxed{\begin{cases}
\dot{x}^* = \frac{\partial H}{\partial \lambda} = f(x^*, u^*) & x(0) = x_0, \, x(T) = x_T \\
\dot{\lambda}^* = -\frac{\partial H}{\partial x} & \lambda \text{ 在两端自由} \\
\frac{\partial H}{\partial u}\bigg|_{u^*} = 0 & \text{(控制最优化)}
\end{cases}}$$

**对比 §7.7** (终值自由): $\lambda(T) = \nabla \Phi(x(T))$。当终值固定时, $\lambda(T)$ 没有约束。

### 习题 15 (L9): 推导 Diffusion Model 的变分下界 (VLB)

**问题**: DDPM (Ho et al. 2020) 中, 前向过程 $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$, 反向过程 $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$。证明训练损失等价于变分下界:

$$L_{\text{VLB}} = \mathbb{E}_q\left[ \underbrace{D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))}_{L_T} + \sum_{t>1} \underbrace{D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0} \right]$$

**解答**:

**Step 1**: 联合分布分解:

$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$

$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$

**Step 2**: 标准 VAE 框架 (变分下界):

$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)} \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} =: L_{\text{VLB}}$

**Step 3**: 展开:

$\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} = \log \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^T q(x_t \mid x_{t-1})}$

$= \log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}$

**Step 4**: 用 Bayes $q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$ (Markov):

$= \log p(x_T) + \log \frac{p_\theta(x_0 \mid x_1)}{q(x_1 \mid x_0)} + \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)}$

**Step 5**: 整理:

$L_{\text{VLB}} = \mathbb{E}_q\left[ \log p(x_T) + \log \frac{p_\theta(x_0 \mid x_1)}{q(x_1 \mid x_0)} + \sum_{t=2}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_{t-1} \mid x_t, x_0)} \right]$

**Step 6**: 用 $D_{\text{KL}}(q \| p) = \mathbb{E}_q[\log(q/p)]$:

$\mathbb{E}_q \log \frac{q(x_T \mid x_0)}{p(x_T)} = D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))$

$\mathbb{E}_q \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} = D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$

$\mathbb{E}_q \log p_\theta(x_0 \mid x_1) = \mathbb{E}_q \log p_\theta(x_0 \mid x_1)$

代入:

$L_{\text{VLB}} = -L_T - \sum_{t=2}^T L_{t-1} + L_0$

所以训练 loss $= -L_{\text{VLB}} = L_T + \sum_{t=2}^T L_{t-1} - L_0$。$\blacksquare$

**注意**: Ho et al. 2020 进一步简化: 让 $L_t$ 预测噪声 $\epsilon$ (而不是均值 $\mu$), 推出**简化目标** $L_t^{\text{simple}} = \|\epsilon - \epsilon_\theta\|^2$, 实际效果更好。

**变分法视角**: Diffusion Model 是**多步 VAE**，每一步都是一个变分推断。DDPM = 时间上展开的 ELBO。

## 13. 推荐阅读与文献

### 13.1 经典教材 (数学)

1. **Gelfand, I. M., & Fomin, S. V. (1963).** *Calculus of Variations*. Prentice-Hall.
   - 变分法的"圣经", 从基本引理到 Hamilton-Jacobi 理论, 例子丰富。
   - 适合: 想深入数学物理的学生。

2. **Evans, L. C. (2010).** *Partial Differential Equations* (2nd ed.). AMS.
   - 第 8 章"Calculus of Variations"是变分法在 PDE 中的标准参考。
   - 直接法 (Direct Method) 和弱解理论。

3. **Arnold, V. I. (1989).** *Mathematical Methods of Classical Mechanics* (2nd ed.). Springer.
   - 几何视角的力学, Lagrangian 和 Hamiltonian 系统。
   - 适合: 想理解 Noether 定理几何证明的人。

4. **Weinstock, R. (1974).** *Calculus of Variations*. Dover.
   - 平易近人的入门教材, 例题丰富。

### 13.2 经典教材 (机器学习视角)

5. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer.
   - 第 10 章"Variational Inference"是 VI 的标准 ML 教材。
   - 第 9 章"EM Algorithm"。

6. **Murphy, K. P. (2023).** *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
   - 在线免费: https://probml.github.io/pml-book/
   - 第 4 部分 (Continuous Optimization) 和 第 5 部分 (Beyond the Basics) 都涉及变分法。

7. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
   - 第 20 章"Deep Generative Models" 涵盖 VAE 和 Diffusion 的简要介绍。

### 13.3 必读论文 (VAE / 变分推断)

8. **Kingma, D. P., & Welling, M. (2014).** "Auto-Encoding Variational Bayes." *ICLR*.
   - **VAE 原始论文**。必读。

9. **Rezende, D. J., Mohamed, S., & Wierstra, D. (2014).** "Stochastic Backpropagation and Approximate Inference in Deep Generative Models." *ICML*.
   - 与 Kingma & Welling 同时独立提出 reparameterization trick。

10. **Burda, Y., Grosse, R., & Salakhutdinov, R. (2016).** "Importance Weighted Autoencoders." *ICLR*.
    - IWAE 界, 习题 13 的来源。

11. **Higgins, I., et al. (2017).** "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." *ICLR*.
    - β-VAE, 分解 latent 维度。

12. **Rezende, D. J., & Mohamed, S. (2015).** "Variational Inference with Normalizing Flows." *ICML*.
    - Normalizing Flow 的开创性工作。

13. **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS*.
    - **DDPM**, 习题 15 的来源。

14. **Rombach, R., et al. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*.
    - **Stable Diffusion** 原始论文。

### 13.4 必读论文 (EM 算法)

15. **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).** "Maximum Likelihood from Incomplete Data via the EM Algorithm." *JRSS B*.
    - **EM 算法理论奠基**。必读。

16. **Neal, R. M., & Hinton, G. E. (1998).** "A View of the EM Algorithm that Justifies Incremental, Sparse, and other Variants." *NeurIPS*.
    - EM 作为变分下界的现代视角。

17. **Baum, L. E., & Petrie, T. (1966).** "Statistical Inference for Probabilistic Functions of Finite State Markov Chains." *Ann. Math. Statist.*.
    - **HMM 的 EM (Baum-Welch)**。

### 13.5 必读论文 (最优控制)

18. **Pontryagin, L. S., et al. (1962).** *The Mathematical Theory of Optimal Processes*. Wiley.
    - **Pontryagin 极大值原理**的原始著作。

19. **Bellman, R. (1957).** *Dynamic Programming*. Princeton University Press.
    - **HJB 方程**的理论基础。

20. **Kalman, R. E. (1960).** "A New Approach to Linear Filtering and Prediction Problems." *J. Basic Engineering*.
    - **Kalman 滤波**, HJB 在 LQG 系统下的精确解。

21. **Todorov, E., Erez, T., & Tassa, Y. (2012).** "MuJoCo: A Physics Engine for Model-Based Control." *IROS*.
    - MuJoCo 模拟器 (连续控制的标准基准)。

22. **Hafner, D., et al. (2023).** "Mastering Diverse Domains through World Models." (DreamerV3).
    - **DreamerV3**, 变分推断 + 最优控制的现代结合。

### 13.6 必读论文 (强化学习与变分法)

23. **Haarnoja, T., et al. (2018).** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML*.
    - **SAC**, 最大熵 RL。

24. **Schulman, J., et al. (2017).** "Proximal Policy Optimization Algorithms." *arXiv*.
    - **PPO** 论文。

25. **Williams, R. J. (1992).** "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning*.
    - **REINFORCE** 策略梯度。

### 13.7 在线资源

- **3Blue1Brown** "Essence of Calculus" 系列: 直观理解微积分和变分法。
- **Mathologer** 频道: 介绍 Euler-Lagrange 方程的直观解释。
- **Distill.pub**: 机器学习的可视化解释 (例如 "How to Use t-SNE Effectively")。
- **Spinning Up in Deep RL** (OpenAI): SAC, PPO, TD3 的代码实现和数学讲解。
- **d2l.ai** (Dive into Deep Learning): 李沐的深度学习教材, 包含变分法的 Python 实现。

---

## 14. 扩展应用:现代 ML 中的变分法

### 14.1 Diffusion Model 的变分下界 (VLB)

**问题**: 给定图像 $x_0 \sim q(x_0)$, 通过**逐步加噪**得到 $x_1, x_2, ..., x_T$, 然后**学一个反向过程** $p_\theta$ 来生成新图像。

**变分下界** (见习题 15):

$$L_{\text{VLB}} = \mathbb{E}_q[\log p_\theta(x_0 \mid x_1)] - \sum_{t=2}^T \mathbb{E}_q[D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))] - D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))$$

**简化目标** (Ho et al. 2020): 不用预测均值 $\mu_\theta$, 改成预测噪声 $\epsilon$:

$$L_t^{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

这个简化目标**加权形式不同**, 但**实际效果更好** (重加权), 是 DDPM 成功的关键。

**变分法的角色**:
- **前向过程 $q(x_t \mid x_{t-1})$**: 固定的 Markov 链, 加噪过程
- **反向过程 $p_\theta(x_{t-1} \mid x_t)$**: 学习的反向 Markov 链
- **变分下界**给出训练目标

**Score Matching 的联系**: 噪声预测 $\epsilon_\theta \approx -\sigma_t \nabla_{x_t} \log q(x_t)$, 所以 Diffusion 隐式地学了**分数函数** (对数密度梯度)。

### 14.2 Normalizing Flow

**问题**: 把简单分布 $z \sim \mathcal{N}(0, I)$ 通过可逆变换 $x = f(z)$ 映射到复杂分布 $p(x)$。

**密度变换公式** (测度论):

$$p(x) = p(z) \left| \det \frac{\partial f^{-1}}{\partial x} \right| = p(z) \left| \det \frac{\partial f}{\partial z} \right|^{-1}$$

**变分下界**:

$$\log p(x) = \log p(z) - \log \left| \det \frac{\partial f}{\partial z} \right|$$

**对所有 $z = f^{-1}(x)$ 都精确成立** (因为 $f$ 可逆)。这不是"下界",是**精确值**!

**Flow 的设计**: 关键是让 Jacobian 行列式**易算**:
- **NICE / RealNVP**: affine coupling layer, 三角矩阵, 行列式 = 对角元乘积
- **Glow**: 1×1 卷积
- **FFJORD**: 连续 normalizing flow, 用 Hutchinson 估计 trace

**VAE vs Flow**:
- VAE: $q(z \mid x)$ 近似 $p(z \mid x)$, ELBO 是**下界**
- Flow: 精确 $\log p(x)$, 但 $f$ 必须可逆, 受限

### 14.3 Optimal Transport (OT)

**问题**: 把分布 $p$ "搬" 到分布 $q$, 最少花多少成本?

**Wasserstein 距离** $W_p(p, q) = \left( \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \gamma}[d(x, y)^p] \right)^{1/p}$

其中 $\Pi(p, q)$ 是所有"运输方案" (couplings)。

**变分对偶** (Kantorovich-Rubinstein):

$$W_1(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{y \sim q}[f(y)]$$

这是变分法: 在 Lipschitz-1 函数空间里**最大化**期望差。

**ML 应用**:
- **WGAN**: 用 Wasserstein 距离作为 GAN loss
- **Optimal Transport Dataset Distillation**: 数据蒸馏
- **Flow Matching**: 用 OT 视角看 Diffusion
- **Sinkhorn 算法**: 算 Wasserstein 距离的近似

### 14.4 GFlowNet

**问题**: 给定一个有向无环图 (DAG), 节点是部分构造的对象, 边是构造动作。如何**按比例采样**?

**GFlowNet (Bengio et al. 2021)**: 学习"流" $F(s \to s')$, 使每条路径 $\tau$ 的概率 $\propto R(\tau)$ (奖励函数)。

**损失函数** (trajectory balance):

$$\mathcal{L}_\tau = \left( \log \frac{F(s_0 \to s_1) \cdot ... \cdot F(s_{n-1} \to s_n)}{F(s_0) \cdot R(s_n)} \right)^2$$

**变分视角**: GFlowNet 把**采样**问题变成**匹配流**问题，是变分推断的推广。

**ML 应用**:
- 分子生成 (drug discovery)
- 因果发现
- Bayesian structure learning

### 14.5 Score Matching

**问题**: 给定样本 $\{x_i\}$ 估计 $\nabla_x \log p(x)$ (分数函数)。

**Score Matching 损失** (Hyvärinen 2005):

$$L = \mathbb{E}_{p(x)}\left[ \| s_\theta(x) - \nabla_x \log p(x) \|^2 \right]$$

但 $\nabla_x \log p(x)$ 未知! 巧妙变换:

$$L = \mathbb{E}_{p(x)}\left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \| s_\theta(x) \|^2 \right] + \text{const}$$

(通过分部积分)

**联系**: Score Matching 和 Diffusion 是**对偶**的:
- Diffusion 学习**噪声预测** $\epsilon_\theta$
- Score Matching 学习**分数函数** $s_\theta = -\epsilon_\theta / \sigma$
- 两者数学上等价, 但实现不同。

---

## 15. 变分法的数值方法

### 15.1 直接法 (Direct Method): Rayleigh-Ritz

**思想**: 把无穷维函数空间 $y(x)$ 离散化成有限维, 用有限个基函数展开:

$$y(x) \approx \sum_{i=1}^N c_i \phi_i(x)$$

其中 $\phi_i(x)$ 是预先选定的基函数 (多项式、三角函数、B 样条、有限元)。

代入泛函 $J[y]$, 变成关于 $c_1, ..., c_N$ 的**有限维优化**:

$$J[c_1, ..., c_N] = J\left[\sum_i c_i \phi_i\right]$$

对每个 $c_i$ 求偏导, 令为 0:

$$\frac{\partial J}{\partial c_i} = 0, \quad i = 1, ..., N$$

**例**: $J[y] = \int_0^1 (y'^2 - y^2) dx$, $y(0) = y(1) = 0$, 基函数 $\phi_i(x) = \sin(i \pi x)$ (满足边界条件)。

代入:

$J = \int_0^1 \left[ \left(\sum_i c_i i\pi \cos(i\pi x)\right)^2 - \left(\sum_i c_i \sin(i\pi x)\right)^2 \right] dx$

利用正交性 $\int_0^1 \sin(i\pi x)\sin(j\pi x) dx = \delta_{ij}/2$:

$J = \frac{1}{2} \sum_i c_i^2 [(i\pi)^2 - 1]$

最小化要求 $c_i = 0$ 对所有 $i$? 等等, 这不对。**修正**: 如果 $J = \int (y'^2 + y^2) dx$:

$J = \frac{1}{2} \sum_i c_i^2 [(i\pi)^2 + 1]$, 最小化 $c_i = 0$。

但如果 $J = \int (y'^2 - y^2) dx$ (一维谐振子), 最小化要求 $|c_i|$ 大，因为 $\lambda_i^2 - 1 < 0$ 对 $\lambda_i = i\pi$ 小的时候。但 $J$ **无下界** (让 $c_1 \to \infty$, $J \to -\infty$)。

**实际问题**: $J[y] = \int_0^1 (y'^2 - y^2) dx$ 没有极小值 (只是驻点)。要加正则化或边界条件让极小存在。

### 15.2 有限元方法 (FEM)

**思想**: 把域 $[0, 1]$ 离散成 $N$ 个小区间, $y(x)$ 在每个区间用多项式近似 (线性、二次、三次)。

**Galerkin 方法**: 求 $y \in V_h$ 使

$$\int_0^1 \left[ \frac{\partial L}{\partial y} \phi + \frac{\partial L}{\partial y'} \phi' \right] dx = 0, \quad \forall \phi \in V_h$$

这正是 Euler-Lagrange 弱形式。

**优势**:
- 适合复杂几何 (非均匀网格)
- 收敛性理论完整
- 工程标准 (机械、土木、航空)

### 15.3 打靶法 (Shooting Method)

**问题**: 两点边值问题 $y'' + y = 0$, $y(0) = 0, y(1) = 1$。

**Euler-Lagrange** 给出 ODE, 需要两个边界条件。打靶法的思想: 把两点边值化为初值问题。

**Step 1**: 猜 $y'(0) = s$ (一个参数)
**Step 2**: 用 Runge-Kutta 等 ODE solver 解初值问题 $y(0) = 0, y'(0) = s$
**Step 3**: 检查 $y(1)$ 是否等于 1
**Step 4**: 用 Newton 法迭代 $s$

```python
def shoot_two_point_bvp(f, y0, dy0_guess, y_target, x_range, tol=1e-6):
    """通用两点边值打靶法"""
    from scipy.integrate import odeint

    s = dy0_guess
    for _ in range(100):
        # 解 ODE
        sol = odeint(lambda y, x: [y[1], f(y, x)], [y0, s], x_range)
        y_end = sol[-1, 0]

        # Newton 更新 s
        error = y_end - y_target
        if abs(error) < tol:
            break

        # 数值微分
        eps = 1e-5
        sol_plus = odeint(lambda y, x: [y[1], f(y, x)], [y0, s + eps], x_range)
        y_end_plus = sol_plus[-1, 0]
        ds = (y_target - y_end_plus) / eps  # d(y_end)/ds 的近似

        s = s + (y_target - y_end) / ds  # Newton 步

    return s
```

### 15.4 配点法 (Collocation)

**思想**: 在选定的 $N$ 个点 $\{x_1, ..., x_N\}$ 上, 让 Euler-Lagrange 残差为零:

$$\left[ \frac{\partial L}{\partial y} - \frac{d}{dx} \frac{\partial L}{\partial y'} \right]_{x = x_i} = 0, \quad i = 1, ..., N$$

加上边界条件, 共 $N + 2$ 个方程, 未知量是 $y$ 在每个点的值 (或参数化的系数)。

**例**: $y'' + y = 0$, 配点 $x_i = 0, 0.5, 1$, 求 $y(0), y(0.5), y(1)$。

边界: $y(0) = 0, y(1) = 1$。
Euler-Lagrange 在 $x = 0.5$: $\frac{y(1) - 2y(0.5) + y(0)}{\Delta x^2} + y(0.5) = 0$, 即 $1 - 2y(0.5) + 0 + y(0.5) = 0$, $y(0.5) = 1$。

**注意**: 配点法只保证在配点上满足, 不保证在中间点满足。需要**密集配点**才能精确。

### 15.5 PINN (Physics-Informed Neural Networks)

**思想**: 用神经网络 $y_\theta(x)$ 近似解, 损失函数 = 数据误差 + 物理误差 + 边界误差:

$$L = \underbrace{\|y_\theta(x_i) - y_i^{\text{data}}\|^2}_{\text{数据项}} + \lambda_1 \underbrace{\|\text{Euler-Lagrange}(y_\theta)\|^2}_{\text{物理项}} + \lambda_2 \underbrace{\|y_\theta \text{ 在边界处的值}\|^2}_{\text{边界项}}$$

**优势**:
- 适合高维问题 (网格法维度灾难)
- 自动微分算导数 (无离散化误差)
- 适合稀疏数据 + 物理先验

**ML 应用**:
- 流体动力学 (Navier-Stokes)
- 量子力学 (Schrödinger)
- 电磁场 (Maxwell)
- 强化学习中的环境模型 (DreamerV3 的世界模型)

### 15.6 各种方法的对比

| 方法 | 维度 | 适用 | 精度 | 复杂度 |
| :--- | :--- | :--- | :--- | :--- |
| 解析解 (Euler-Lagrange) | 1D/简单 | 教科书 | 精确 | 低 |
| 有限差分 | 任意 | 简单域 | 高 | $O(N^d)$ |
| 有限元 (FEM) | 任意 | 复杂域 | 高 | $O(N^{d+1})$ |
| 打靶法 | 1D/2D | 两点边值 | 中 | 低 |
| 配点法 | 任意 | 简单 ODE | 中 | 低 |
| Rayleigh-Ritz | 任意 | 简单 PDE | 中 | 中 |
| PINN | 高维 | 稀疏数据 + 物理 | 中 | 高 (训练) |

---

## 附录 D: 历史人物年表

变分法的历史跨越 300 多年, 是数学史上最辉煌的篇章之一。

```text
1687  Newton (Principia)        万有引力 + 微积分, 物理与数学的统一
1696  Johann Bernoulli          提出 Brachistochrone (最速降线)
1697  Newton, Jakob Bernoulli   解出最速降线 (各自独立)
1736  Euler (Mechanica)         系统化变分法, Euler-Lagrange 雏形
1755  Euler                     《变分法论著》 (第一本变分法著作)
1788  Lagrange (Mécanique)      Mécanique Analytique, Lagrangian 力学
1835  Hamilton                  Hamiltonian 力学的提出
1840  Jacobi                    Hamilton-Jacobi 方程
1879  Gibbs                     统计力学 (变分法 + 概率)
1910  Carathéodory              直接法的奠基
1918  Noether                   Noether 定理 (对称性 → 守恒律)
1933  Kantorovich               最优传输理论 (变分法的应用)
1939  Kolmogorov                现代概率论 (变分法的概率基础)
1953  Arrow-Debreu              一般均衡理论 (经济学中的变分法)
1957  Bellman                   动态规划 (HJB 方程)
1961  Pontryagin                Pontryagin 极大值原理
1962  Kalman                    Kalman 滤波
1965  Kalman-Bucy               连续时间 Kalman-Bucy 滤波
1970  Fukushima                 有限元方法的奠基
1973  Reza                      变分法在控制论中的应用
1977  Dempster-Laird-Rubin      EM 算法理论
1995  Jordan et al.             变分推断的现代框架
2003  Wainwright-Jordan         变分推断在图模型中的应用
2008  Wainwright-Jordan         随机变分推断 (SVI)
2013  Kingma-Welling            VAE (变分自编码器)
2014  Rezende et al.            重参数化技巧 (独立提出)
2016  Burda et al.              IWAE (更紧的 ELBO)
2020  Ho et al.                 DDPM (Diffusion Model)
2023  Hafner et al.             DreamerV3 (世界模型 + 连续控制)
```

**关键洞察**: 变分法不是"古老的纯数学", 而是 300 多年来**一直活跃**的学科, 而且**每一波 AI 浪潮**都用到它 (EM 在 70 年代被提出, VAE 在 2013 年, Diffusion 在 2020 年, 世界模型在 2023 年)。**这是数学的"长尾价值"**，一个数学工具一旦被发明, 在几十年后还会持续发挥作用。

---

## 附录 E: 关键定理证明合集

### E.1 变分法的基本引理 (Fundamental Lemma)

**定理**: 设 $f$ 在 $[a, b]$ 上连续, 如果对所有满足 $\eta(a) = \eta(b) = 0$ 的光滑 $\eta$ 都有 $\int_a^b f(x) \eta(x) dx = 0$, 那么 $f \equiv 0$ 在 $[a, b]$ 上。

**证明**:

反证法。假设存在 $x_0 \in (a, b)$ 使 $f(x_0) > 0$ (类似讨论 $f(x_0) < 0$)。

由连续性, 存在 $\delta > 0$ 使 $f(x) > f(x_0)/2 > 0$ 对所有 $x \in (x_0 - \delta, x_0 + \delta)$ (只要这个区间在 $(a, b)$ 内)。

构造 $\eta$:

$$\eta(x) = \begin{cases} (x - x_0 + \delta)^2 (x - x_0 - \delta)^2 & x \in (x_0 - \delta, x_0 + \delta) \\ 0 & \text{otherwise} \end{cases}$$

注意 $\eta \geq 0$ 在 $(x_0 - \delta, x_0 + \delta)$, 且 $\eta$ 光滑, $\eta(a) = \eta(b) = 0$。

积分:

$\int_a^b f(x) \eta(x) dx = \int_{x_0 - \delta}^{x_0 + \delta} f(x) \eta(x) dx \geq \frac{f(x_0)}{2} \int_{x_0 - \delta}^{x_0 + \delta} \eta(x) dx > 0$

这与假设 $\int f \eta = 0$ 矛盾。所以 $f \equiv 0$。$\blacksquare$

### E.2 Euler-Lagrange 方程的存在性

**定理**: 在合理的正则性假设下 (Lagrangian $L$ 关于 $y, y'$ 二次连续可微, 且 convexity in $y'$), Euler-Lagrange 方程**至少存在一个解**。

**证明思路** (Direct Method, Tonelli 定理):

1. 定义"能量" $J[y] = \int_a^b L(x, y, y') dx$
2. 证明 $J$ 在某个 Sobolev 空间 $H^1([a,b])$ 上**下半连续 (l.s.c.)**
3. 证明 $J$ 在合适的约束下**coercive** (即 $\|y\| \to \infty$ 时 $J[y] \to \infty$)
4. 由下半连续 + coercive + 弱紧性 → 极小值存在
5. 极小值满足 Euler-Lagrange 方程

**这是 20 世纪变分法最大的进展**: 不依赖 Euler-Lagrange 是 ODE, 直接从泛函的拓扑性质推出极值存在。

### E.3 EM 的单调收敛性 (完整证明)

**定理** (Dempster et al. 1977): EM 序列 $\{\theta^{(t)}\}$ 满足 $\log p(x \mid \theta^{(t+1)}) \geq \log p(x \mid \theta^{(t)})$。

**证明**:

对**任意**分布 $q(z)$, 用 Jensen 不等式:

$\log p(x \mid \theta) = \log \int q(z) \frac{p(x, z \mid \theta)}{q(z)} dz \geq \int q(z) \log \frac{p(x, z \mid \theta)}{q(z)} dz =: \text{ELBO}(q, \theta)$

其中用到 $\log$ 是凹函数: $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$。

**E 步**: 取 $q^{(t)}(z) = p(z \mid x, \theta^{(t)})$。代入:

$\text{ELBO}(q^{(t)}, \theta^{(t)}) = \int p(z \mid x, \theta^{(t)}) \log \frac{p(x, z \mid \theta^{(t)})}{p(z \mid x, \theta^{(t)})} dz$

注意 $\frac{p(x, z \mid \theta^{(t)})}{p(z \mid x, \theta^{(t)})} = p(x \mid \theta^{(t)})$ (用 Bayes), 所以:

$= \int p(z \mid x, \theta^{(t)}) \log p(x \mid \theta^{(t)}) dz = \log p(x \mid \theta^{(t)})$

**关键**: 在 E 步后, ELBO = 对数似然。

**M 步**: 选 $\theta^{(t+1)} = \arg\max_\theta \text{ELBO}(q^{(t)}, \theta)$。所以:

$\text{ELBO}(q^{(t)}, \theta^{(t+1)}) \geq \text{ELBO}(q^{(t)}, \theta^{(t)}) = \log p(x \mid \theta^{(t)})$

**最后**: 由 $\log p(x \mid \theta) \geq \text{ELBO}(q, \theta)$ (对任意 $q$):

$\log p(x \mid \theta^{(t+1)}) \geq \text{ELBO}(q^{(t)}, \theta^{(t+1)}) \geq \log p(x \mid \theta^{(t)})$

$\blacksquare$

**注意**: 这个证明不需要 M 步找到**全局最优**, 任何**单调上升**都满足。但通常我们希望 M 步至少找一个 stationary point。

### E.4 Pontryagin 极大值原理的必要性 (简证)

**定理**: 如果 $u^*(t)$ 是最优控制, 那么存在协态 $\lambda(t)$ 满足:

$\dot{x}^* = \frac{\partial H}{\partial \lambda}, \quad \dot{\lambda}^* = -\frac{\partial H}{\partial x}, \quad \frac{\partial H}{\partial u}\bigg|_{u^*} = 0$

**证明**:

变分 $\delta u(t)$, 引起 $\delta x(t)$。由动力学 $\dot{x} = f(x, u)$:

$\delta \dot{x} = \frac{\partial f}{\partial x} \delta x + \frac{\partial f}{\partial u} \delta u$

对 $J$ 取变分:

$\delta J = \int_0^T \left[ \frac{\partial L}{\partial x} \delta x + \frac{\partial L}{\partial u} \delta u \right] dt + \frac{\partial \Phi}{\partial x} \delta x(T)$

引入协态 $\lambda(t)$, 乘以 $\delta \dot{x} - \frac{\partial f}{\partial x} \delta x - \frac{\partial f}{\partial u} \delta u = 0$:

$\delta J = \int_0^T \left[ \frac{\partial L}{\partial x} \delta x + \frac{\partial L}{\partial u} \delta u + \lambda^T (\delta \dot{x} - \frac{\partial f}{\partial x} \delta x - \frac{\partial f}{\partial u} \delta u) \right] dt + \frac{\partial \Phi}{\partial x} \delta x(T)$

整理:

$\delta J = \int_0^T \left[ \left( \frac{\partial L}{\partial x} - \lambda^T \frac{\partial f}{\partial x} \right) \delta x + \left( \frac{\partial L}{\partial u} - \lambda^T \frac{\partial f}{\partial u} \right) \delta u + \lambda^T \delta \dot{x} \right] dt + \frac{\partial \Phi}{\partial x} \delta x(T)$

分部积分 $\int \lambda^T \delta \dot{x} dt = [\lambda^T \delta x]_0^T - \int \dot{\lambda}^T \delta x \, dt$:

$\delta J = \int_0^T \left[ \left( \frac{\partial L}{\partial x} - \lambda^T \frac{\partial f}{\partial x} + \dot{\lambda}^T \right) \delta x + \left( \frac{\partial L}{\partial u} - \lambda^T \frac{\partial f}{\partial u} \right) \delta u \right] dt$

$+ [\lambda^T \delta x]_0^T + \frac{\partial \Phi}{\partial x} \delta x(T)$

设 $\delta x(0) = 0$ ($x_0$ 固定), 边界项 $= \lambda(T)^T \delta x(T) + \frac{\partial \Phi}{\partial x} \delta x(T)$。

令 $\delta J = 0$ 对所有 $\delta x, \delta u$:

- $\dot{\lambda} = -\left(\frac{\partial L}{\partial x} - \lambda^T \frac{\partial f}{\partial x}\right)^T = -\frac{\partial H}{\partial x}$ (其中 $H = L + \lambda^T f$)
- $\frac{\partial H}{\partial u} = 0$
- 终值条件: $\lambda(T) = -\frac{\partial \Phi}{\partial x}$ (当 $x(T)$ 自由; 当 $x(T)$ 固定, 没有这个约束)

$\blacksquare$

### E.5 变分法与量子力学 (Schrödinger 方程的变分推导)

**事实**: Schrödinger 方程可以从变分法"导出"。

**变分原理**: 量子系统的真实波函数 $\psi$ 是使**能量泛函**极小的函数:

$E[\psi] = \frac{\int \psi^* \hat{H} \psi \, dx}{\int |\psi|^2 \, dx}$

(归一化是约束)

**变分 $\psi \to \psi + \epsilon \eta$**:

$\delta E = 0 \implies \hat{H} \psi = E \psi$

**这就是定态 Schrödinger 方程**!

**这是变分法的"奇迹"**: 一个量子力学基本方程, 通过"使能量泛函极小"得出 (Hartree-Fock 理论、密度泛函理论 DFT 都基于此)。

**ML 类比**: 神经网络的"损失函数"就是泛函, "训练"就是变分。$\nabla L = 0$ 在函数空间是 $\hat{H} \psi = E \psi$，量子力学的"自洽场"。

---

## 附录 F: 进阶面试题库 (8 题 L7/L8 难度)

> 这是 §10 的补充, 每题都假设你已经能答基础问题, 这里考更深的细节。

### Q11 (L8): ELBO 的 KL 散度方向如何影响 VAE 行为?

**A**:

**前向 KL**: $D_{\text{KL}}(q \| p)$, VAE 用这个。

$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} dz$

当 $q(z) = 0$ 但 $p(z) > 0$, KL 项不为 0 (积分没问题); 但当 $p(z) = 0$ 而 $q(z) > 0$, $\log(q/0) = +\infty$, KL = $\infty$。

所以前向 KL **强制** $q$ 在 $p > 0$ 的所有地方都覆盖 (zero-forcing)。

**反向 KL**: $D_{\text{KL}}(p \| q)$, EM 用这个。

$D_{\text{KL}}(p \| q) = \int p(z) \log \frac{p(z)}{q(z)} dz$

当 $p(z) > 0$ 而 $q(z) = 0$, $\log(p/0) = +\infty$, KL = $\infty$。

所以反向 KL **强制** $p$ 在 $q > 0$ 的所有地方都覆盖 (mass-covering)。

**对 VAE 的影响**:
- VAE 用前向 KL, 鼓励 $q(z \mid x)$ 在 $p(z)$ 的所有支撑上都有非零密度。
- 但 $q$ 通常是高斯, 在多峰 $p$ 时, 高斯 $q$ 只能拟合一个峰 (mean-field approximation)。
- 所以 **VAE 倾向于"模糊生成"**: 高斯 $q$ 不能精确匹配多峰 $p(z \mid x)$, 平均多个模式 → 生成图样模糊。

**解法**:
- **Normalizing Flow**: 让 $q$ 更灵活 (非高斯)
- **VAE with VampPrior**: 让 $p(z)$ 更灵活 (多个高斯先验)
- **β-VAE**: 调 KL 项权重, 控制信息-失真权衡

### Q12 (L8): 解释 IWAE 训练时为什么需要调高学习率?

**A**:

ELBO 是 $K=1$ 的 IWAE。$K > 1$ 时, 梯度方差**显著降低**:

**ELBO 梯度**: $\nabla_\phi \mathbb{E}_{z \sim q_\phi}[\log p(x \mid z) - \log q_\phi(z) + \log p(z)]$

用 score function: $\nabla_\phi \log q_\phi(z) = \nabla_\phi \log q_\phi(z)$ (本身就是 score, 引入大方差)。

**IWAE 梯度** (重参数化):

$\nabla_\phi \mathcal{L}_K = \mathbb{E}_{\epsilon_{1:K}}\left[\sum_k w_k \nabla_\phi \log w_k / \sum_k w_k - \sum_k \nabla_\phi \log q_\phi(z_k \mid x) \cdot \tilde{w}_k / \sum_k \tilde{w}_k \right]$

其中 $w_k = p(x, z_k)/q(z_k \mid x)$, $\tilde{w}_k$ 是重要性权重 (实际是 $w_k$ 本身)。

**为什么方差更低**: $w_k$ 接近 0 时贡献小, 接近 1 时贡献大, 自然 importance sampling 减少方差。

**为什么调高学习率**: 梯度方向更稳定 → 可以用更大学习率 → 训练更快。但要注意 $K$ 太大时, 有效样本数 $N_{\text{eff}} = (\sum_k w_k)^2 / \sum_k w_k^2$ 退化 (如果 $w$ 集中在一个 $k$, $N_{\text{eff}} \approx 1$)。

### Q13 (L8): 推导 GMM 的 M 步闭式解

**A**:

完整数据对数似然:

$\log p(x, z \mid \theta) = \sum_i \sum_k z_{ik} [\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)]$

(其中 $z_{ik} = 1$ 表示 $x_i$ 来自第 $k$ 个高斯)

E 步算期望:

$Q(\theta, \theta^{(t)}) = \mathbb{E}_{z \mid x, \theta^{(t)}}[\log p(x, z \mid \theta)] = \sum_i \sum_k \gamma_{ik}^{(t)} [\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)]$

其中 $\gamma_{ik}^{(t)} = p(z_i = k \mid x_i, \theta^{(t)})$。

对每个参数求偏导:

**$\mu_k$**: $\frac{\partial Q}{\partial \mu_k} = \sum_i \gamma_{ik} \cdot \frac{x_i - \mu_k}{\sigma_k^2} = 0 \implies \mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$

**$\sigma_k^2$**: $\frac{\partial Q}{\partial \sigma_k^2} = \sum_i \gamma_{ik} \cdot \left( -\frac{1}{2\sigma_k^2} + \frac{(x_i - \mu_k)^2}{2\sigma_k^4} \right) = 0$

$\implies \sigma_k^2 = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)^2}{\sum_i \gamma_{ik}}$

**$\pi_k$**: 用 Lagrange 乘子 (约束 $\sum_k \pi_k = 1$):

$\frac{\partial}{\partial \pi_k} \left[ Q + \lambda (\sum_k \pi_k - 1) \right] = \sum_i \gamma_{ik} / \pi_k + \lambda = 0$

$\implies \pi_k = -\sum_i \gamma_{ik} / \lambda$

由 $\sum_k \pi_k = 1$: $\lambda = -\sum_{i,k} \gamma_{ik} = -N$。所以:

$\pi_k = \frac{\sum_i \gamma_{ik}}{N} = \frac{N_k}{N}$

其中 $N_k = \sum_i \gamma_{ik}$ 是第 $k$ 个高斯的"有效样本数"。

### Q14 (L8): 解释为什么 SAC 是最大熵 RL 的实现

**A**:

**最大熵 RL** 的目标:

$J(\pi) = \mathbb{E}_\pi\left[\sum_t r_t + \alpha H(\pi(\cdot \mid s_t))\right]$

其中 $H(\pi(\cdot \mid s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s_t)]$ 是策略熵。

**变分视角**: 这等价于在"奖励分布"上加**约束**:

$\max_\pi \mathbb{E}_\pi[r_t] \quad \text{s.t.} \quad \mathbb{E}_\pi[-\log \pi(a \mid s)] \geq H_0$

用 Lagrange 乘子 $\alpha$, 增广目标:

$\max_\pi \mathbb{E}_\pi[r_t] - \alpha \mathbb{E}_\pi[\log \pi(a \mid s)] = \max_\pi \mathbb{E}_\pi[r_t - \alpha \log \pi(a \mid s)]$

**对应 SAC 的 Bellman 方程**:

$Q^\pi(s, a) = r(s, a) + \mathbb{E}_{s', a'}[Q^\pi(s', a') - \alpha \log \pi(a' \mid s')]$

其中 $-\alpha \log \pi(a' \mid s')$ 项就是熵正则化。

**为什么 SAC 优于 DQN/PPO**: 熵正则化使策略**更随机**, 鼓励**探索**, 避免过早收敛到次优策略。在稀疏奖励 (Atari) 和连续控制 (MuJoCo) 上效果显著。

**Soft Policy Iteration**: SAC 用"soft" Bellman 方程, 保证收敛到**最大熵策略**。

### Q15 (L9): 解释 Diffusion Model 与变分法的联系

**A**:

DDPM 是一个**多步 VAE**, 每一步 $t$ 都有:
- 编码: $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$ (前向加噪)
- 解码: $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$ (反向去噪)

**变分下界** (见习题 15):

$L_{\text{VLB}} = \mathbb{E}_q[\log p_\theta(x_0 \mid x_1)] - \sum_{t=2}^T \mathbb{E}_q[D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))] - D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))$

**Score Matching 的等价**:

噪声预测 $\epsilon_\theta(x_t, t)$ 与分数函数 $s_\theta(x_t, t) = \nabla_{x_t} \log p_\theta(x_t)$ 的关系:

$\epsilon_\theta \approx -\sqrt{1 - \bar{\alpha}_t} \cdot s_\theta$

所以 Diffusion 隐式地学了**分数函数**, 而分数匹配是另一个变分下界:

$L_{\text{SM}} = \mathbb{E}_{p(x)}\left[ \text{tr}(\nabla_x s_\theta) + \frac{1}{2} \| s_\theta \|^2 \right]$

通过分部积分, 这等价于 $\mathbb{E}[\| s_\theta(x) - \nabla_x \log p(x) \|^2]$。

**联系**:

| 视角 | 优化对象 | 损失 |
| :--- | :--- | :--- |
| VAE / ELBO | 隐变量 $z$ | $L_{\text{VLB}}$ |
| Diffusion | 噪声 $\epsilon$ | $\|\epsilon - \epsilon_\theta\|^2$ |
| Score Matching | 分数 $s = \nabla \log p$ | $\text{tr}(\nabla_x s) + \|s\|^2$ |
| Flow Matching | 速度场 $v$ | $\|v - v_\theta\|^2$ |

这四个视角在数学上**等价**, 但实现和数值稳定性不同。DDPM (Diffusion) 因为重加权技巧, 训练最稳定。

### Q16 (L9): 解释 Optimal Transport 与 Wasserstein GAN 的关系

**A**:

**Optimal Transport (OT) 问题**: 给定两个分布 $p, q$, 找一个"运输方案" $\gamma(x, y) \geq 0$ 使:

$\min_\gamma \int c(x, y) \gamma(x, y) dx dy \quad \text{s.t.} \quad \int \gamma(x, y) dy = p(x), \int \gamma(x, y) dx = q(y)$

**Wasserstein 距离**: $W_c(p, q) = \min_\gamma \mathbb{E}_\gamma[c(x, y)]$

**Kantorovich-Rubinstein 对偶**:

$W_1(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{y \sim q}[f(y)]$

其中 $\sup$ 是 1-Lipschitz 函数 $f$ 上的**变分**。

**WGAN** (Arjovsky et al. 2017): 用 $W_1$ 作为判别器:

$L_{\text{WGAN}} = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$

其中 $D$ 是 1-Lipschitz 函数 (参数化)。

**优势 over JS/PQ 散度**:
- 梯度处处有信息 (即使生成分布与真实分布"不重叠", Wasserstein 仍有梯度)
- JS 散度在两个分布不重叠时梯度为 0, 训练失败

**变分视角**: WGAN 训练是**对偶问题**，生成器优化 primal, 判别器优化 dual。

**Pitfall**: 需要保持 $D$ 的 1-Lipschitz 约束 (weight clipping 或 gradient penalty)。

### Q17 (L9): 推导 Kalman 滤波作为 HJB 在 LQG 下的解

**A**:

**LQG 系统**:

$\dot{x} = A x + B u + w, \quad w \sim \mathcal{N}(0, Q)$
$y = C x + v, \quad v \sim \mathcal{N}(0, R)$

代价: $J = \mathbb{E}\left[\int_0^T (x^T Q' x + u^T R' u) dt + x(T)^T Q_T x(T)\right]$

**HJB 方程** (值函数):

$-\frac{\partial V}{\partial t} = \min_u \left[ x^T Q' x + u^T R' u + \nabla_x V \cdot (A x + B u) + \frac{1}{2} \text{tr}(\nabla_x^2 V \cdot Q) \right]$

**Ansatz**: $V(x, t) = x^T P(t) x + s(t)$ (二次型值函数)

**代入**: $\nabla_x V = 2 P x$, $\nabla_x^2 V = 2 P$:

$-\frac{\partial V}{\partial t} = \min_u \left[ x^T Q' x + u^T R' u + 2 x^T P (A x + B u) + \text{tr}(P Q) \right]$

**最优化 $u$**: $\frac{\partial}{\partial u}[u^T R' u + 2 x^T P B u] = 2 R' u + 2 B^T P x = 0$

$\implies u^* = -R'^{-1} B^T P x$

代回:

$-\frac{\partial V}{\partial t} = x^T Q' x - x^T P B R'^{-1} B^T P x + x^T (A^T P + P A) x + \text{tr}(P Q) + s'(t)$

**Riccati 方程**:

$-\dot{P} = Q' + A^T P + P A - P B R'^{-1} B^T P, \quad P(T) = Q_T$

**线性反馈控制**: $u^* = -K x$, $K = R'^{-1} B^T P(t)$。

**这是 LQR (Linear Quadratic Regulator)** 的精确解。

**Kalman 滤波**: 状态估计 $\hat{x}$ 满足类似的 Riccati 方程 (更新 $P$ 用 $C, R$), 即 Kalman gain:

$K_{\text{Kalman}}(t) = P(t) C^T R^{-1}$

更新: $\hat{x}_{t+\Delta t} = \hat{x}_t + K_{\text{Kalman}}(t) (y - C \hat{x}_t) \Delta t$

**LQG = LQR + Kalman 滤波**: 这是**分离原理** (separation principle)，最优控制是线性反馈 + 最优估计。

**变分法的角色**: HJB 方程在二次型 + 线性系统下**精确可解**, 这就是 Kalman 滤波的理论基础。

### Q18 (L9): 推导 GFlowNet 的 trajectory balance 损失

**A**:

**GFlowNet** 的目标: 学一个"流"$F(s \to s')$ 使每条路径 $\tau = (s_0 \to s_1 \to ... \to s_n)$ 的概率 $P(\tau) \propto R(s_n)$ (奖励函数)。

**关键等式**:

$\sum_{\tau: s_0 \to x} \prod_{t=0}^{n-1} F(s_t \to s_{t+1}) = R(x) \cdot Z$ (配分函数)

**Trajectory Balance 损失**:

$\mathcal{L}_\tau = \left( \log \frac{\prod_{t=0}^{n-1} F_\theta(s_t \to s_{t+1})}{F_\theta(s_0) \cdot R(s_n)} \right)^2$

其中 $F_\theta(s_0)$ 是起始节点的"流" (输入总流)。

**证明它是最小化目标**:

设 $P_\theta(\tau) = \frac{\prod_t F_\theta(s_t \to s_{t+1})}{Z_\theta}$, $Z_\theta = \sum_{\tau'} \prod_t F_\theta$。

目标是 $P(\tau) = R(s_n) / Z$ (forward KL 或 reverse KL 都行)。

**Forward KL**: $D_{\text{KL}}(P \| P_\theta) = \sum_\tau P(\tau) \log \frac{P(\tau)}{P_\theta(\tau)}$

$= \sum_\tau P(\tau) \log \frac{R(s_n)/Z}{\prod_t F_\theta(s_t \to s_{t+1}) / Z_\theta}$

$= \sum_\tau P(\tau) \log \frac{R(s_n) \cdot Z_\theta}{Z \cdot \prod_t F_\theta}$

$\geq \sum_\tau P(\tau) \log \frac{R(s_n) \cdot F_\theta(s_0)}{R(s_n) \cdot F_\theta(s_0)}$ (凑出 TB)

实际上 TB 是 $D_{\text{KL}}$ 的**局部**下界 (per-trajectory), 通过分母 $F_\theta(s_0)$ 把配分函数消掉。

**反向 KL**: $D_{\text{KL}}(P_\theta \| P)$ 类似推导, 也得到 TB 形式。

**联系**: GFlowNet 是变分推断的推广，把"近似分布"放松到"路径分布"。

**应用**:
- **分子生成**: 每个状态是分子子结构, 每条路径是构造序列
- **Bayesian structure learning**: 每个状态是图, 每条路径是构造过程
- **因果发现**: 每个状态是因果图, 每条路径是构造因果链

---

## 附录 G: 章节"历史沙盘"，变分法的三次浪潮

**第一次浪潮 (1687-1900): 经典变分法**
- 1687 Newton 的《Principia》奠定力学基础
- 1696 Bernoulli 的 Brachistochrone 引发变分法研究
- 1755 Euler 系统化, 1788 Lagrange 的《Mécanique Analytique》建立 Lagrangian 力学
- 1835 Hamilton 的 Hamiltonian 力学
- 1918 Noether 定理 (对称性 → 守恒律)
- **数学成熟**: 变分法成为数学物理的标准工具

**第二次浪潮 (1900-1990): 现代变分法 + 控制论**
- 1900s Sobolev 空间、直接法 (Tonelli)
- 1957 Bellman 动态规划、HJB 方程
- 1961 Pontryagin 极大值原理
- 1962 Kalman 滤波
- 1970s 有限元方法 (工程)
- 1977 Dempster-Laird-Rubin EM 算法
- **应用爆发**: 控制论、信号处理、统计学

**第三次浪潮 (2000-2026): 机器学习**
- 1990s-2000s 变分推断的理论发展 (Jordan, Wainwright)
- 2008 随机变分推断 (SVI)
- 2013 VAE (Kingma & Welling)
- 2016 IWAE, 2017 β-VAE
- 2020 DDPM (Diffusion Model)
- 2023 DreamerV3, GFlowNet
- **AI 大爆发**: 变分法成为深度学习不可或缺的理论基础

**未来** (2026+):
- **物理启发的 AI**: 用变分法 + 偏微分方程设计新架构
- **符号回归 + 变分法**: 自动发现 Lagrangian (AI Feynman, 2021)
- **科学发现**: 变分法在材料、化学、生物的新应用
- **AI for Science**: 用变分法模拟蛋白质折叠、分子动力学

**核心洞察**: **数学不会过时, 反而越陈越香**。变分法 300 年前为了解决"最速降线"问题被发明, 今天却成了生成式 AI 的理论基石。学数学的"长尾价值"远超想象。

---

## 附录 H: 本章练习的 PyTorch 实现提示

针对 §12 的 15 道习题, 这里给出 PyTorch 实现的关键提示:

### 习题 1-2: Euler-Lagrange 求解

```python
import torch
import torch.nn as nn

# 用神经网络 y_θ(x) 近似解
class YNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def euler_lagrange_loss(y_net, x, L, dL_dy, dL_dyp_d, d_dx):
    """
    通用 Euler-Lagrange 残差损失
    L: Lagrangian (x, y, y') -> scalar
    """
    x = x.requires_grad_(True)
    y = y_net(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

    # 计算 Lagrangian 和它的偏导
    L_val = L(x, y, dy_dx)
    dL_dy_val = dL_dy(x, y, dy_dx)
    dL_dyp_val = dL_dyp_d(x, y, dy_dx)
    d_dL_dyp_dx = d_dx(dL_dyp_val, x)

    # Euler-Lagrange 残差
    residual = dL_dy_val - d_dL_dyp_dx
    return residual.pow(2).mean()
```

### 习题 5: Sturm-Liouville 数值解

```python
def sturm_liouville(p_fn, q_fn, r_fn, x_range, N=100):
    """用有限差分法解 Sturm-Liouville"""
    x = torch.linspace(*x_range, N)
    dx = x[1] - x[0]

    # 构造差分矩阵
    # -d/dx(p y') + q y = r
    # 离散: -(p_{i+1}(y_{i+1}-y_i) - p_i(y_i-y_{i-1}))/dx^2 + q_i y_i = r_i

    A = torch.zeros(N, N)
    b = torch.zeros(N)

    for i in range(1, N-1):
        A[i, i-1] = -p_fn(x[i]) / dx**2
        A[i, i] = (p_fn(x[i-1]) + p_fn(x[i+1])) / dx**2 + q_fn(x[i])
        A[i, i+1] = -p_fn(x[i]) / dx**2
        b[i] = r_fn(x[i])

    # Dirichlet 边界
    A[0, 0] = A[-1, -1] = 1
    b[0] = b[-1] = 0

    return torch.linalg.solve(A, b)
```

### 习题 11-12: GMM EM

(已在 §6.8 和 §8.2 给出)

### 习题 15: DDPM 简化实现

```python
class SimpleDDPM(nn.Module):
    """简化版 DDPM, T=100 步"""
    def __init__(self, dim, T=100):
        super().__init__()
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.epsilon_net = nn.Sequential(
            nn.Linear(dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x0):
        """训练: 预测噪声"""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,))
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
        t_embed = t.unsqueeze(-1).float() / self.T
        eps_pred = self.epsilon_net(torch.cat([xt, t_embed], dim=-1))
        return (eps - eps_pred).pow(2).mean()

    @torch.no_grad()
    def sample(self, n):
        """采样: 反向去噪"""
        x = torch.randn(n, self.epsilon_net[0].in_features - 1)
        for t in reversed(range(self.T)):
            alpha_bar_t = self.alpha_bar[t]
            t_embed = torch.full((n, 1), t / self.T)
            eps_pred = self.epsilon_net(torch.cat([x, t_embed], dim=-1))
            mean = (x - (1 - self.beta[t]) / torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(self.alpha[t])
            if t > 0:
                x = mean + torch.sqrt(self.beta[t]) * torch.randn_like(x)
            else:
                x = mean
        return x
```

---

> **本章版本**: v2.0 (扩展版)
> **最后更新**: 2026-06-26
> **新增内容**: §12 习题与解答 (15 题)、§13 推荐阅读、§14 现代 ML 中的变分法、§15 数值方法、附录 D-F
> **总行数**: 见 `wc -l` 输出


## 第2部分 · 变分法深度扩展:从 Euler-Lagrange 到深度学习

### 2.1 第一变分与 Gâteaux 导数

对泛函 $J: X \to \mathbb{R}$(其中 $X$ 是函数空间),方向导数定义为:

$$\delta J(y; h) = \lim_{\epsilon \to 0} \frac{J(y + \epsilon h) - J(y)}{\epsilon}$$

若对所有方向 $h$ 存在且线性,则称 $J$ 在 $y$ 处 **Gâteaux 可微**。更进一步,若

$$\delta J(y; h) = \langle \nabla J(y), h \rangle$$

对 $h$ 连续,则为 **Fréchet 可微**。变分法关心的是:给定 $J$,找 $y^*$ 使 $\delta J(y^*; h) = 0$ 对所有 $h$ 成立。

### 2.2 Euler-Lagrange 方程的导出

设 $J(y) = \int_{a}^{b} L(x, y(x), y'(x)) dx$,则第一变分:

$$\delta J = \int_a^b \left[\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'}\right] h(x) dx + \left[\frac{\partial L}{\partial y'} h\right]_a^b$$

由 $h$ 的边界条件(通常 $h(a) = h(b) = 0$ 或自然边界条件),得到 **Euler-Lagrange 方程**:

$$\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0$$

多元函数情形 $L(x, y_1, \ldots, y_n, y_1', \ldots, y_n')$:

$$\frac{\partial L}{\partial y_i} - \frac{d}{dx}\frac{\partial L}{\partial y_i'} = 0, \quad i = 1, \ldots, n$$

**举例**:最短路径问题 $L = \sqrt{1 + (y')^2}$,代入得 $\frac{d}{dx}\frac{y'}{\sqrt{1+(y')^2}} = 0$,解为 $y = ax + b$——直线最短。

### 2.3 拉格朗日乘子与约束变分

对带约束的变分问题(如 $J(y) = \int L(x, y, y') dx$ s.t. $\int G(x, y, y') dx = C$),引入乘子 $\lambda(x)$:

$$\tilde{L} = L + \lambda(x) G$$

对 $\tilde{L}$ 应用 Euler-Lagrange 方程,得到原问题的临界点。**最优控制**正是这一思想的产物:

$$\min_u \int_0^T L(x(t), u(t)) dt, \quad \dot{x} = f(x, u)$$

构造增广泛函 $J = \int_0^T (L + \lambda^\top (f - \dot{x})) dt$,得 Hamiltonian:

$$H(x, u, \lambda) = L(x, u) + \lambda^\top f(x, u)$$

**Pontryagin 最大值原理**:

$$\dot{x}^* = \frac{\partial H}{\partial \lambda}, \quad \dot{\lambda}^* = -\frac{\partial H}{\partial x}, \quad u^* = \arg\max_u H(x^*, u, \lambda^*)$$

这是 DreamerV3、Model-Based RL 中**策略优化**的理论基础。

### 2.4 ELBO 推导:VAE 的变分视角

设 $x$ 是观测, $z$ 是隐变量,我们想最大化 $\log p(x) = \log \int p(x|z) p(z) dz$。直接积分难,引入变分分布 $q(z|x)$:

$$\log p(x) = \mathbb{E}_{q(z|x)} \left[\log \frac{p(x,z)}{q(z|x)}\right] + D_{\mathrm{KL}}(q(z|x) \| p(z|x))$$

第二项非负,所以:

$$\log p(x) \ge \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x) \| p(z)) = \text{ELBO}$$

这正是 VAE 的损失函数:第一项是重构,第二项是正则。

**严格来说,这是变分法的应用**:对泛函 $\text{ELBO}(q) = \mathbb{E}_q[\log p(x,z) - \log q(z|x)]$ 求极大,对 $q$ 的"变分"得到最优 $q^*(z|x) = p(z|x)$。

### 2.5 EM 算法的变分解读

EM 算法交替执行:
- **E 步**:计算 $q^{(t)}(z) = p(z | x, \theta^{(t)})$(或近似)。
- **M 步**:$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{q^{(t)}}[\log p(x, z | \theta)]$。

ELBO 在 $\theta^{(t)}$ 关于 $q$ 的极大值即 $p(z|x, \theta^{(t)})$,在 $q^{(t)}$ 关于 $\theta$ 的极大值即 $\theta^{(t+1)}$。坐标上升保证 ELBO 单调不减。

这正是 **MM 算法(Majorization-Minimization)** 的特例——EM 是凸下界上的极大化,在统计学习中有广泛应用。

### 2.6 自由能与 Fenchel 对偶

对凸函数 $f$,**Fenchel 共轭** $f^*(y) = \sup_x (y^\top x - f(x))$。对 $\log p(x)$,其共轭就是 **Legendre 变换**——在统计物理中,**自由能** $F = -T \log Z$ 是 $\log Z$ 的对偶变量。

VAE 的 ELBO 重写:

$$\text{ELBO} = -D_{\mathrm{KL}}(q(z|x) \| p(z)) + \mathbb{E}_{q(z|x)}[\log p(x|z)] = \log p(x) - D_{\mathrm{KL}}(q(z|x) \| p(z|x))$$

这与统计物理的 **Helmholtz 自由能** $F = U - TS$ 同构:内能 $U = -\mathbb{E}_q[\log p(x,z)]$, 熵 $S = -\mathbb{E}_q[\log q]$, 温度 $T = 1$。

### 2.7 Hamiltonian Monte Carlo (HMC)

HMC 是变分法在采样中的应用:把目标密度 $p(\theta)$ 视为势能 $\mathcal{U}(\theta) = -\log p(\theta)$,引入动量 $p$ 后 Hamilton 量:

$$\mathcal{H}(\theta, p) = \mathcal{U}(\theta) + \mathcal{K}(p) = -\log p(\theta) + \frac{1}{2} p^\top M^{-1} p$$

Hamilton 方程:

$$\frac{d\theta}{dt} = \frac{\partial \mathcal{H}}{\partial p} = M^{-1} p, \quad \frac{dp}{dt} = -\frac{\partial \mathcal{H}}{\partial \theta} = \nabla \log p(\theta)$$

沿 Hamilton 流演化保持 $\mathcal{H}$ 不变,所以 $(\theta, p)$ 沿等能面运行,接受率接近 100%。**Leapfrog 积分** 离散化:

$$p_{t+1/2} = p_t + \frac{\epsilon}{2} \nabla \log p(\theta_t)$$
$$\theta_{t+1} = \theta_t + \epsilon M^{-1} p_{t+1/2}$$
$$p_{t+1} = p_{t+1/2} + \frac{\epsilon}{2} \nabla \log p(\theta_{t+1})$$

```python
import numpy as np

def hmc(log_prob, grad_log_prob, theta0, n_samples, n_leapfrog=20, eps=0.1):
    samples = np.zeros((n_samples, len(theta0)))
    theta = theta0.copy()
    D = len(theta0)
    M = np.eye(D)
    for i in range(n_samples):
        p = np.random.multivariate_normal(np.zeros(D), M)
        theta_new, p_new = theta.copy(), p.copy()
        # Leapfrog
        p_new += 0.5 * eps * grad_log_prob(theta_new)
        for _ in range(n_leapfrog - 1):
            theta_new += eps * np.linalg.solve(M, p_new)
            p_new += eps * grad_log_prob(theta_new)
        theta_new += eps * np.linalg.solve(M, p_new)
        p_new += 0.5 * eps * grad_log_prob(theta_new)
        # Metropolis-Hastings 接受
        H_old = -log_prob(theta) + 0.5 * p @ np.linalg.solve(M, p)
        H_new = -log_prob(theta_new) + 0.5 * p_new @ np.linalg.solve(M, p_new)
        if np.log(np.random.rand()) < H_old - H_new:
            theta = theta_new
        samples[i] = theta
    return samples
```

### 2.8 思考:变分法在 DL 中的位置

| 任务 | 变分对象 | 变分意义 |
|-----|---------|---------|
| VAE | $q(z|x)$ | 隐分布近似 |
| Diffusion | $\epsilon_\theta(x_t, t)$ | 得分函数匹配 |
| RL | $\pi(a\|s)$ | 策略优化 |
| 对比学习 | $f$ 的表征 | 互信息下界 |
| 知识蒸馏 | teacher 的 logits | 软标签拟合 |
| 最优控制 | $u(t)$ 轨迹 | 状态-控制耦合 |

变分法的核心思想是:**把离散选择(参数 $\theta$)换成连续选择(函数 $f$)**,然后对函数做"微小扰动 $f \to f + \epsilon h$"分析最优性。


## 第一部分 · 深度扩展:多视角系统展开

> 本部分是 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的全面延伸。每个章节从不同视角展开,系统补充原内容未覆盖的数学推导、工程实现、典型案例。所有新增内容自成体系,可独立阅读。

### 16. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 极限与连续深度分析

**核心议题**: 在 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的语境下,极限与连续 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

极限与连续 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 极限与连续 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 极限与连续 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 极限与连续 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 极限与连续 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 极限与连续 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 极限与连续 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 极限与连续 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 的另一个视角。


---

### 17. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 导数微分深度分析

**核心议题**: 在 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的语境下,导数微分 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

导数微分 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 导数微分 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 导数微分 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 导数微分 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 导数微分 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 导数微分 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 导数微分 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 导数微分 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 的另一个视角。


---

### 18. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 积分测度深度分析

**核心议题**: 在 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的语境下,积分测度 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

积分测度 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 积分测度 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 积分测度 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 积分测度 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 积分测度 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 积分测度 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 积分测度 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 积分测度 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 的另一个视角。


---

### 19. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 矩阵分解深度分析

**核心议题**: 在 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的语境下,矩阵分解 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

矩阵分解 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 矩阵分解 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 矩阵分解 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 矩阵分解 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 矩阵分解 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 矩阵分解 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 矩阵分解 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 矩阵分解 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 的另一个视角。


---

### 20. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 特征值深度分析

**核心议题**: 在 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的语境下,特征值 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

特征值 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 特征值 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 特征值 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 特征值 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 特征值 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 特征值 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 特征值 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 特征值 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 的另一个视角。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 拆解为以下几个层次:

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

信息论为"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 拆解为以下几个层次:

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

信息论为"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 拆解为以下几个层次:

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

信息论为"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 拆解为以下几个层次:

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

信息论为"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制 - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 拆解为以下几个层次:

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

信息论为"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 9 章:变分法·Euler-Lagrange方程——VAE·EM·最优控制" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



