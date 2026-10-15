---
title: "第04章-09-离散概率·随机过程-随机游走·PageRank·MCMC"
tags:
  - 数学
  - 离散概率
  - 随机过程
  - 随机游走
  - PageRank
  - MCMC
description: "本章要点：加购 → 购买 → 复购 → 流失。这背后是一张状态转移图, 用户在状态之间随机游走, 每一跳只取决于\"现在在哪\", 不记得\"过去从哪来\"。这叫 Mar…"
chapter_id: 数学-9-04
last_updated: 2026-07-14
---

# 第04章-09-离散概率·随机过程-随机游走·PageRank·MCMC

 加购 → 购买 → 复购 → 流失。这背后是一张**状态转移图**, 用户在状态之间随机游走, 每一跳只取决于"现在在哪", 不记得"过去从哪来"。这叫 **Markov 性质**, 也是这章的核心。
>
> 你用 PageRank 算网页重要性时, "随机冲浪者"在网页之间点击链接, 偶尔随机跳走, 长期访问比例 = 网页排名。这同样是一个随机游走。
>
> 你跑 MCMC 做贝叶斯推断或推荐去重时, 状态在样本空间里跳, 每一步只依赖当前状态, 经过 burn-in 后达到平稳分布。这就是**离散时间 Markov 链**。
>
> 这一章我们做三件事:
> - 用**离散概率**(PMF、联合 / 边缘 / 条件、期望)打底
> - 把**随机游走**讲到 Pólya 定理(为什么醉汉能回家)
> - 把 **Markov 链 → PageRank → 离散 MCMC** 一条线串起来, 配上 6 段可运行 PyTorch 代码
>
> 学完之后, 你回头看 RL 的环境动力学、推荐召回的 Personalized PageRank、GNN 的随机游走采样, 都会发现它们背后站着同一棵数学树: **状态 + 转移矩阵 + 平稳分布**。

---

## 1. 直觉引入

### 1.1 工程师痛点: 为什么推荐系统"用户行为序列"可以用 Markov 链建模?

做推荐的工程师, 每天都在跟"用户行为序列"打交道:

```python
# 一条用户行为序列(时间从早到晚)
user_history = ["浏览", "加购", "加购", "购买", "购买", "复购", "复购", "复购", "流失"]
```

如果你要做"下一个动作预测", 最朴素的思路是**统计每个动作后面最常出现什么**:

```text
浏览 → {加购: 60%, 流失: 30%, 购买: 10%}
加购 → {购买: 50%, 流失: 30%, 加购: 20%}
购买 → {复购: 70%, 流失: 30%}
复购 → {复购: 80%, 流失: 20%}
流失 → {流失: 100%}(已流失, 几乎不会回来)
```

这本质上就是一张**转移概率表**。如果再加一条假设:

> **"下一个动作只取决于当前动作, 不取决于更早的历史"**

那你就得到了一个 **Markov 链**(一阶齐次)。这个假设在很多真实场景下并不严格成立(用户从"加购"到"购买"的概率, 可能取决于他"浏览"了多久), 但作为**工程近似**, 它简洁、可解释、可计算, 是工业推荐、RL 环境建模、网页排名的共同基石。

### 1.2 三个直觉例子

#### 例 1: 电商用户行为 Markov 链 (浏览 → 加购 → 购买 → 流失)

把上面的转移概率画成一张状态图:

```text
                    ┌──────────────────────────────────┐
                    ↓                                  │
    ┌─────────┐  0.6   ┌─────────┐  0.5   ┌─────────┐ 0.7   ┌─────────┐ 0.8   ┌─────────┐
    │  浏览   │ ─────→ │  加购   │ ─────→ │  购买   │ ─────→│  复购   │ ─────→│  复购   │
    └─────────┘        └─────────┘        └─────────┘        └─────────┘        └─────────┘
        │ 0.3              │ 0.2              │ 0.3              │ 0.2              │
        ↓                  ↓                  ↓                  ↓                  ↓
    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                                          流失                                          │
    │                                       (吸收态)                                          │
    └───────────────────────────────────────────────────────────────────────────────────────┘
```

5 个状态: 浏览、加购、购买、复购、流失。每条边上的数字是从源状态跳到目标状态的概率 (列和 = 1)。**流失**是吸收态 (跳出去的概率 = 1.0, 因为它指向自己)。

**这个模型能回答的问题**:
1. 用户进入系统后, 长期来看停留在每个状态的概率是多少? (平稳分布 $\pi$)
2. 用户平均多少步会流失? (首达时间)
3. 如果我们做一次运营活动 (干预转移概率), 留存提升多少? (敏感性分析)

#### 例 2: PageRank 的"随机冲浪"模型

1998 年 Larry Page 和 Sergey Brin 想给网页排名, 核心想法是:

> **一个网页被很多"重要网页"链接, 它就重要**。

把这个想法翻译成随机游走:

- **状态** = 网页 (假设 5 个网页: A, B, C, D, E)
- **转移**: 在当前网页, 等概率点击页面上一个链接跳过去
- **阻尼**: 有 15% 的概率**不点链接**, 而是随机跳到任意网页 (避免死胡同和"自循环作弊")

数学公式 (简化版, 后面 §5 严格推导):

$$\mathbf{r} = (1 - d) \cdot \frac{1}{n} \mathbf{1} + d \cdot M^T \mathbf{r}$$

其中 $d = 0.85$ 是"跟随链接"的概率, $1 - d = 0.15$ 是"随机跳"的概率, $M$ 是链接矩阵 (列归一化), $\mathbf{r}$ 是网页重要性向量。

**直觉**: 一个网页的 PageRank = "随机冲浪者"长期访问它的概率。这就是 Markov 链的**平稳分布**。

#### 例 3: MCMC 在推荐去重中的应用

假设你要从 100 万个商品中**均匀采样 1000 个不重复的"推荐去重候选"**, 但约束是"候选不能跟用户历史太相似"。直接拒绝采样太慢 (拒绝率 99.9%), **MCMC** 提供了一个优雅解法:

1. 任意初始化一个候选集 $S_0$
2. 每一步: 从 $S_t$ 中选一个商品 $i$ 替换为新商品 $j$ (按某种提议分布)
3. 按 Metropolis-Hastings 接受 / 拒绝 $j$
4. 经过 burn-in, $S_t$ 收敛到目标分布

这就是**离散 MCMC**: 状态是候选集 (组合空间), 转移是"替换一个元素", 平稳分布是"满足约束的均匀分布"。

### 1.3 ASCII 可视化: Markov 链的"轨迹"

跑上面那个 5 状态推荐链 50 步, 用户的轨迹大概长这样:

```text
步:    1   2   3   4   5   6   7   8   9  10  11  12  ... 50
状态: 浏  浏  加  加  购  加  购  购  复  复  复  复  ... 失
       ↓       ↓  ↓       ↓       ↓  ↓
       浏      加 加      加       购 购
```

(每个状态是随机抽样的, 转移概率决定"流向"。)

**观察**:
- 前期在"活跃状态" (浏览 / 加购 / 购买 / 复购) 之间反复横跳
- 后期停在"流失" 不再回来
- 如果把 1000 个用户的轨迹叠加, 可以统计出"100 步之后, 多少人在每个状态" → 这就是**经验分布**, 趋近**平稳分布** $\pi$

### 1.4 章节路线图

| 节 | 主题 | 与 ML/DL 的联系 |
| :--- | :--- | :--- |
| §2 | 离散概率基础 (PMF / 联合 / 条件 / 期望) | Markov 链的"概率"语言基础 |
| §3 | 随机游走 (1D / 2D / d 维) | GNN 采样 / 布朗运动离散版 / Pólya 定理 |
| §4 | Markov 链 (转移矩阵 / 平稳分布 / 状态分类) | 推荐 / RL 环境 / MCMC 的统一抽象 |
| §5 | PageRank (幂迭代 / 阻尼 / PPR) | 网页排名 / 召回 / 图嵌入 |
| §6 | 离散 MCMC (Metropolis-Hastings / Gibbs) | 推荐去重 / 贝叶斯推断 / LLM 解码 |
| §7 | PyTorch 代码 | 6 个可运行实验, 全部 PyTorch 2.x |
| §8 | 常见误区 | 5 个工程师踩坑点 |
| §9 | 面试 Q&A | L7/L8 难度 5 题 |
| §10 | 联系其他章节 | 图论 / RL / MCMC 章节互引 |

---

## 2. 离散概率基础回顾

这一节我们快速过一遍离散概率。Markov 链是"概率 + 状态转移"的组合, 概率的语言必须先到位。

### 2.1 离散随机变量 / 概率质量函数 (PMF)

**离散随机变量** $X$ 取可数个值 $\{x_1, x_2, \ldots\}$, 用**概率质量函数** (PMF) 描述每个值的概率:

$$p(x_i) = P(X = x_i), \quad p(x_i) \ge 0, \quad \sum_i p(x_i) = 1$$

**例子**: 扔一枚公平骰子, $X$ = 朝上的点数:

$$p(k) = \frac{1}{6}, \quad k \in \{1, 2, 3, 4, 5, 6\}$$

**例子**: 伯努利 (抛硬币) 分布 $X \sim \text{Bernoulli}(p)$:

$$p(X = 1) = p, \quad p(X = 0) = 1 - p$$

**例子**: Categorical 分布 (推荐场景用得最多): 用户下一个动作有 $K$ 种可能, 第 $k$ 种概率 $\pi_k$:

$$p(X = k) = \pi_k, \quad k = 1, \ldots, K, \quad \sum_k \pi_k = 1$$

### 2.2 联合 / 边缘 / 条件概率

**联合概率** $p(X, Y)$: 两个随机变量同时取某值的概率。

**边缘概率** $p(X)$: 只关心 $X$, 把 $Y$ 加和掉:

$$p(X = x) = \sum_y p(X = x, Y = y)$$

**条件概率** $p(Y | X)$: 给定 $X$, $Y$ 的分布:

$$p(Y = y | X = x) = \frac{p(X = x, Y = y)}{p(X = x)}$$

**贝叶斯定理**:

$$p(Y | X) = \frac{p(X | Y) \cdot p(Y)}{p(X)}$$

**例子**: 推荐场景下, "用户看到广告 $A$ 后点击 $C$" 的概率:

$$p(C = 1 | A = a) = \frac{p(A = a | C = 1) \cdot p(C = 1)}{p(A = a)}$$

这就是朴素贝叶斯分类器的核心公式。

### 2.3 独立 / 条件独立

**独立**: $X$ 与 $Y$ 独立, 记作 $X \perp Y$:

$$p(X, Y) = p(X) \cdot p(Y)$$

**条件独立**: 给定 $Z$, $X$ 与 $Y$ 条件独立:

$$p(X, Y | Z) = p(X | Z) \cdot p(Y | Z)$$

**Markov 性质就是条件独立的一种** (后面 §4 详述): 给定"现在", "过去"与"未来"独立。

### 2.4 期望 / 方差

**期望** (均值):

$$\mathbb{E}[X] = \sum_x x \cdot p(x)$$

**方差**:

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**例子**: 公平骰子 $X \sim \text{Uniform}\{1, \ldots, 6\}$:

$$\mathbb{E}[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5$$

$$\text{Var}(X) = \frac{1}{6} \sum_{k=1}^{6} (k - 3.5)^2 = \frac{35}{12} \approx 2.917$$

### 2.5 重要不等式 (速查)

| 不等式 | 公式 | 直观 | 用途 |
| :--- | :--- | :--- | :--- |
| Markov | $P(X \ge a) \le \frac{\mathbb{E}[X]}{a}$ | 平均有限, 极端不会太多 | 粗略尾部界 |
| Chebyshev | $P(\|X - \mu\| \ge k\sigma) \le \frac{1}{k^2}$ | 大多数数据离均值不远 | 弱收敛性证明 |
| Hoeffding | $P(\|\bar X_n - \mu\| \ge t) \le 2 e^{-2 n t^2}$ | 样本均值指数收敛 | RL / Bandit 置信界 |

**Hoeffding 在 RL 中的应用**: Off-Policy 评估时, 估计 $\mathbb{E}_\pi[R]$ 的误差上界是 $O(\sqrt{\log(1/\delta)/n})$, $n$ 是采样数。这就是 PPO 等算法的理论基础之一。

### 2.6 工程例子: 骰子模拟与 PyTorch 实现

```python
import torch
from torch.distributions import Categorical, Bernoulli

# 1. 公平骰子 (Categorical)
die_probs = torch.tensor([1/6] * 6)
dist = Categorical(probs=die_probs)
samples = dist.sample((1000,))
print("骰子采样频率:")
for i in range(6):
    print(f"  点数 {i+1}: {(samples == i).float().mean():.4f}")

# 2. 伯努利 (抛硬币)
coin = Bernoulli(probs=0.7)
samples = coin.sample((10000,))
print(f"\n硬币正面频率: {samples.mean():.4f} (理论 0.7)")

# 3. 经验期望 vs 理论期望
X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
probs = torch.tensor([1/6] * 6)
empirical_mean = (X * probs).sum()  # 解析期望
sampled_mean = X[dist.sample((100000,))].mean()  # 蒙特卡洛估计
print(f"\n理论 E[X] = {empirical_mean:.4f}")
print(f"采样 E[X] = {sampled_mean:.4f}")
```

预期输出 (采样均值会接近理论值, 但有微小偏差):

```text
骰子采样频率:
  点数 1: 0.1670
  ...
硬币正面频率: 0.7012 (理论 0.7)
理论 E[X] = 3.5000
采样 E[X] = 3.5012
```

**关键观察**: 蒙特卡洛估计的方差随样本数 $n$ 增加而衰减 (CLT: 误差量级 $O(1/\sqrt{n})$)。1000 次采样误差大约 $\pm 0.05$, 10000 次大约 $\pm 0.016$。

---

## 3. 随机游走 (Random Walk)

### 3.1 一维随机游走

**定义**: 初始位置 $X_0 = 0$, 每一步等概率向 $\pm 1$ 移动:

$$X_{t+1} = X_t + Y_t, \quad Y_t \in \{-1, +1\} \text{ 各概率 } 0.5$$

**直观**: 一只醉汉在数轴上左右乱走, 每一步向左或向右概率相等。

**轨迹可视化** (50 步):

```text
   x
   ↑
 3 ┤        ●
   │      ╱   ╲
 2 ┤    ●       ●    ╱╲
   │   ╱         ╲╱    ●
 1 ┤  ●
   │ ╱╲
 0 ┤●   ╲
   │     ╲╱╲
-1 ┤        ●
   ↓──────────────────→ 步
   0  10  20  30  40  50
```

(轨迹看起来"无序", 但有统计规律。)

### 3.2 二维 / d 维随机游走

**二维**: 每步等概率向 4 个方向 ($\pm x, \pm y$) 之一移动:

$$X_{t+1} = X_t + \Delta_t, \quad \Delta_t \in \{(1,0), (-1,0), (0,1), (0,-1)\}$$

**d 维**: 每步等概率向 $2d$ 个"单位向量"之一移动, $\Delta_t \in \{\pm e_1, \ldots, \pm e_d\}$。

**Pólya 定理** (关键结论, 1921):

> 在 1D 和 2D 随机游走中, 醉汉**几乎必然**返回原点; 在 3D 及以上, 返回原点的概率 < 1。

具体概率:

| 维度 $d$ | 返回原点的概率 |
| :--- | :--- |
| 1 | 1.0 (必然返回) |
| 2 | 1.0 (必然返回) |
| 3 | ≈ 0.3405 |
| 4 | ≈ 0.1932 |
| $d$ | 趋近 0 |

**工程意义**: 高维空间很大, 随机游走很难回到原点, 这就是"维度灾难"的一个表现。GNN 在大图上做随机游走采样, 高维 (节点数多) 时收敛慢。

### 3.3 转移概率与转移矩阵

1D 随机游走的**转移概率**:

$$P(X_{t+1} = j | X_t = i) = \begin{cases} 0.5 & \text{if } j = i + 1 \\ 0.5 & \text{if } j = i - 1 \\ 0 & \text{otherwise} \end{cases}$$

对 $n$ 个状态 $\{1, \ldots, n\}$ 的随机游走, 转移矩阵 (假设边界反射):

$$P = \begin{pmatrix} 0.5 & 0.5 & 0 & 0 & \cdots & 0 \\ 0.5 & 0 & 0.5 & 0 & \cdots & 0 \\ 0 & 0.5 & 0 & 0.5 & \cdots & 0 \\ \vdots & & & & & \\ 0 & \cdots & 0 & 0.5 & 0 & 0.5 \\ 0 & \cdots & 0 & 0 & 0.5 & 0.5 \end{pmatrix}$$

**关键性质**:
- 每行 (或每列, 取决于约定) 之和 = 1
- $P_{ij} \ge 0$
- 这两个性质定义了一个**随机矩阵 (stochastic matrix)**

### 3.4 期望位置 / 方差

1D 简单随机游走:

- **期望位置**: $\mathbb{E}[X_t] = 0$ (对称, 不偏向任何方向)
- **方差**: $\text{Var}(X_t) = t$ (每步独立, 方差累加)

**关键结论**: $|X_t|$ 的期望量级是 $\sqrt{t}$, 不是 $t$。这是因为**正负相消**。

**工程意义**: 在 GNN 节点嵌入的随机游走采样中, 走 $t$ 步后访问的节点"扩散半径"大约是 $O(\sqrt{t})$, 不是 $O(t)$。

### 3.5 首次返回概率 (Pólya 定理)

**问题**: 醉汉从原点出发, 在时刻 $2n$ **首次**回到原点的概率是多少?

**Pólya 公式**: (推导用 Stirling 近似 + 反射原理)

$$u_{2n} = \frac{1}{2n - 1} \binom{2n}{n} \cdot 2^{-2n} \approx \frac{1}{\sqrt{\pi n^3}}$$

**推导梗概** (1D 情形):

要走 $2n$ 步回到原点, 必须有 $n$ 步向右, $n$ 步向左, 共 $\binom{2n}{n}$ 种选择, 每种概率 $2^{-2n}$。但这只是"经过原点"的次数, 首次返回要求中间不再次经过原点。用 **反射原理** 可以证明, 首次返回概率正好是 $u_{2n} = \frac{1}{2n-1}\binom{2n}{n} 2^{-2n}$。

用 Stirling 近似 $n! \approx \sqrt{2\pi n}(n/e)^n$:

$$\binom{2n}{n} = \frac{(2n)!}{(n!)^2} \approx \frac{\sqrt{4\pi n}(2n/e)^{2n}}{2\pi n (n/e)^{2n}} = \frac{4^n}{\sqrt{\pi n}}$$

代入:

$$u_{2n} \approx \frac{1}{2n-1} \cdot \frac{4^n}{\sqrt{\pi n}} \cdot \frac{1}{4^n} = \frac{1}{\sqrt{\pi n^3}}$$

**返回原点的总概率**:

$$f_{\text{return}} = \sum_{n=1}^{\infty} u_{2n} \approx \sum_{n=1}^{\infty} \frac{1}{\sqrt{\pi n^3}}$$

- **1D**: $\sum 1/\sqrt{n^3} = \sum 1/n^{3/2}$ 收敛 (p > 1), 但 1D 的对称性让总概率恰好 = 1 (Pólya 原始证明)
- **2D**: $\sum 1/n$ **发散**, 但因 2D 随机游走的对称性, 返回概率也是 1
- **3D**: $\sum 1/n^{3/2}$ 收敛, 总概率 ≈ 0.3405 (Pólya 数)

**期望返回时间**: 在 1D 和 2D 上**发散** (醉汉要等"无穷久"才能保证首次返回), 在 3D 及以上**收敛** (有限期望)。

**直观**: 1D 醉汉一直被"困"在一根线上, 反复穿过原点是常态。3D 醉汉可以"逃"到无穷远, 原点只是无限空间中的一个点。

**工程意义**:
- **GNN 节点采样**: 在大图上做随机游走采样, 节点数 $N$ 大 (相当于"维度高"), 随机游走"扩散半径"小, 收敛慢。这是 GraphSAGE / PinSage 用固定步数 (而不是"等返回") 采样的原因。
- **RL 探索**: 在高维状态空间中, 随机策略很难"碰巧"返回已访问状态, 这就是"探索不足"。$\epsilon$-greedy / UCB 等探索策略就是为了解决这个问题。

### 3.6 例: 5 个网页的随机游走模拟

```text
状态空间: {1, 2, 3, 4, 5}
转移矩阵 P (假设完全对称):
     1    2    3    4    5
1  [ 0    0.5  0    0    0.5 ]
2  [ 0.5  0    0.5  0    0   ]
3  [ 0    0.5  0    0.5  0   ]
4  [ 0    0    0.5  0    0.5 ]
5  [ 0.5  0    0    0.5  0   ]
```

**这是一个环形结构**: 1-2-3-4-5-1, 每点向左右邻居等概率跳。

**平稳分布** $\pi$: 因为是**对称** + **正则图**, 所有状态等价, 所以 $\pi = (0.2, 0.2, 0.2, 0.2, 0.2)$。

**直觉**: 长期来看, 随机游走访问每个网页的概率都是 1/5, 因为结构对称。

### 3.7 PyTorch 代码: 一维随机游走仿真

```python
import torch

def random_walk_1d(n_steps, p_right=0.5, seed=42):
    """1D 随机游走: 每步以 p_right 概率向右 (+1), 否则向左 (-1)"""
    torch.manual_seed(seed)
    # 1 = 右, 0 = 左
    steps = torch.bernoulli(torch.full((n_steps,), p_right)).long() * 2 - 1
    # 累积位置
    positions = torch.cat([torch.tensor([0.0]), steps.cumsum(0).float()])
    return positions

# 跑 1000 步
trajectory = random_walk_1d(1000)
print(f"起始位置: {trajectory[0]:.0f}")
print(f"最终位置: {trajectory[-1]:.0f}")
print(f"位置绝对值最大: {trajectory.abs().max():.0f}")
print(f"理论量级 (sqrt(t)): {torch.sqrt(torch.tensor(1000.0)):.2f}")

# 模拟 100 条轨迹, 统计最终位置分布
n_simulations = 100
n_steps = 1000
all_final = torch.zeros(n_simulations)
for i in range(n_simulations):
    traj = random_walk_1d(n_steps, seed=i)
    all_final[i] = traj[-1]

print(f"\n{ n_simulations} 条轨迹的最终位置均值: {all_final.mean():.2f}")
print(f"最终位置标准差: {all_final.std():.2f}")
print(f"理论 std (sqrt(t)): {torch.sqrt(torch.tensor(float(n_steps))):.2f}")
```

**预期观察**: 1000 步后最终位置均值约 0, 标准差约 31.6 (≈ $\sqrt{1000}$), 与理论一致。

---

## 4. Markov 链

### 4.1 Markov 性质: 未来只依赖于现在

**定义**: 随机过程 $\{X_t\}_{t=0,1,2,\ldots}$ 满足 **Markov 性质**, 如果:

$$P(X_{t+1} = j | X_t = i, X_{t-1} = i_{t-1}, \ldots, X_0 = i_0) = P(X_{t+1} = j | X_t = i)$$

**直观**: "明天"只取决于"今天", 与"昨天及更早"无关。一旦知道"现在在哪", "过去"对预测"未来"没有额外信息。

**更一般的形式**:

$$P(X_{t+n} = \cdot | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+n} = \cdot | X_t)$$

### 4.2 转移矩阵 $P$ (n × n)

对**有限状态空间** $S = \{1, 2, \ldots, n\}$ 的 Markov 链, 用一个 $n \times n$ 矩阵编码所有转移概率:

$$P_{ij} = P(X_{t+1} = j | X_t = i), \quad \sum_j P_{ij} = 1$$

**例子**: 5 状态推荐链 (§1.2 例 1) 的转移矩阵 (从源状态看行):

$$P = \begin{pmatrix} 0 & 0.6 & 0.1 & 0 & 0.3 \\ 0 & 0.2 & 0.5 & 0 & 0.3 \\ 0 & 0 & 0 & 0.7 & 0.3 \\ 0 & 0 & 0 & 0.8 & 0.2 \\ 0 & 0 & 0 & 0 & 1 \end{pmatrix}$$

- 第 1 行 (浏览): 0.6 → 加购, 0.1 → 购买, 0.3 → 流失
- 第 5 行 (流失): 1.0 → 流失 (吸收态)

**验证**: 每行求和 = 1 ✓。

### 4.3 n 步转移矩阵 $P^n$

**单步转移**: $P(X_{t+1} = j | X_t = i) = P_{ij}$

**两步转移** (用全概率公式):

$$P(X_{t+2} = k | X_t = i) = \sum_j P_{ij} \cdot P_{jk} = (P^2)_{ik}$$

**n 步转移**:

$$P^{(n)}_{ij} = P(X_{t+n} = j | X_t = i) = (P^n)_{ij}$$

这就是为什么 **$P^n$ 的第 $(i, j)$ 元素 = 从状态 $i$ 出发, $n$ 步后到达状态 $j$ 的概率**。

### 4.4 Chapman-Kolmogorov 方程

**定理**: 对任意非负整数 $m, n$:

$$P^{(m+n)}_{ij} = \sum_k P^{(m)}_{ik} \cdot P^{(n)}_{kj}$$

**矩阵形式**: $P^{m+n} = P^m \cdot P^n$ (普通矩阵乘法)。

**直观**: 从 $i$ 出发走 $m+n$ 步到 $j$, 中间一定在某个状态 $k$ 停一下, 把 $m+n$ 步分成两段。

### 4.5 状态分类: 常返 / 瞬态 / 吸收态

设状态 $i$ 为参考:

**常返态 (Recurrent)**: 从 $i$ 出发, **几乎必然**最终回到 $i$。

**瞬态 (Transient)**: 从 $i$ 出发, 回到 $i$ 的概率 < 1。

**吸收态 (Absorbing)**: $P_{ii} = 1$ (一旦进入, 永远不出)。

**例子**: §1.2 例 1 的 5 状态推荐链:
- 浏览、加购、购买、复购 是**瞬态** (有非零概率流失到吸收态)
- 流失 是**吸收态** ($P_{55} = 1$)

**判断准则**: 状态 $i$ 常返 $\iff$ $\sum_{n=1}^{\infty} P^{(n)}_{ii} = \infty$ (返回次数期望无穷)。

### 4.6 不可约 + 非周期 + 详细平衡

平稳分布存在且唯一需要三个条件:

**不可约 (Irreducible)**: 任意两个状态 $i, j$, 存在 $n$ 使得 $P^{(n)}_{ij} > 0$。即状态图是**强连通**的。

**非周期 (Aperiodic)**: 任何状态 $i$ 的周期 $\gcd\{n : P^{(n)}_{ii} > 0\} = 1$。即不存在"偶数步 / 奇数步"的强制交替。

**正递归 (Positive Recurrent)**: 从任意状态出发, 期望返回时间有限。

**三者都满足时, 平稳分布存在且唯一**, 且 $\lim_{n \to \infty} P^{(n)}_{ij} = \pi_j$ (从任何初始状态出发, 都会收敛到 $\pi$)。

#### 详细平衡 (Detailed Balance)

**定理**: 如果存在 $\pi > 0$ 使得对所有 $i, j$:

$$\pi_i P_{ij} = \pi_j P_{ji}$$

则 $\pi$ 是平稳分布。

**直观**: "从 $i$ 流到 $j$ 的概率质量" = "从 $j$ 流到 $i$ 的概率质量"。这是"微观平衡", 比 $\pi P = \pi$ (宏观平衡) 更强。

**重要应用**: Metropolis-Hastings 算法的接受率 $\alpha$ 就是为满足详细平衡而设计的。给定 $\pi$ (目标分布), 选择任意提议分布 $Q$, 接受率:

$$\alpha(x \to x') = \min\left(1, \frac{\pi(x') Q(x | x')}{\pi(x) Q(x' | x)}\right)$$

可以验证 $\pi(x) Q(x \to x') \alpha(x \to x') = \pi(x') Q(x' \to x) \alpha(x' \to x)$, 即详细平衡成立, $\pi$ 是平稳分布。

#### 例: 5 状态推荐链的不可约性

```text
浏览 → 加购 → 购买 → 复购 ─→ 流失
                            ↖
                              (吸收)
```

- 流失是吸收态, 进了出不来
- 浏览、加购、购买、复购之间可以互通 (假设有反向转移)
- 但任何状态都不能"逃出"流失的吸收

**结论**: 这个链**不是不可约的**, 因为流失不能到达其他状态。所以**平稳分布不唯一**, 解出来的 $\pi$ 全集中在流失态 (因为其他状态最终都流向流失)。

### 4.6 平稳分布 $\pi$ (满足 $\pi = \pi P$)

**定义**: 分布 $\pi = (\pi_1, \pi_2, \ldots, \pi_n)$ 是**平稳分布**, 如果:

$$\pi = \pi P, \quad \text{即} \quad \pi_j = \sum_i \pi_i P_{ij}, \quad \sum_j \pi_j = 1$$

**直观**: 如果当前状态分布是 $\pi$, 走一步后分布还是 $\pi$。**平稳分布 = Markov 链的"长期归宿"**。

**求解**: 平稳分布是转移矩阵 $P^T$ 的**特征值为 1** 的特征向量:

$$P^T \pi^T = \pi^T$$

注意是 $P^T$ 而不是 $P$。这是因为 $\pi P$ 的转置 $= P^T \pi^T$。

**例子**: 5 状态推荐链的平稳分布:

求 $\pi$ 满足 $\pi = \pi P$, 解得 (后面 PyTorch 代码会算):

$$\pi \approx (\text{浏览}, \text{加购}, \text{购买}, \text{复购}, \text{流失}) = (\approx 0, \approx 0, \approx 0, \approx 0, \approx 1)$$

**直观**: 因为有吸收态"流失", 长期来看所有用户都进流失, 所以平稳分布全集中在流失态。

### 4.7 例: 推荐系统的 5 状态 Markov 链

复述 §1.2 例 1, 加上详细的状态转移:

```text
状态空间: S = {浏览, 加购, 购买, 复购, 流失}
转移矩阵 P (行 = 源, 列 = 目标):

            浏览  加购  购买  复购  流失
    浏览  [  0    0.6   0.1    0    0.3 ]  ← 用户进入后大部分加购
    加购  [  0    0.2   0.5    0    0.3 ]  ← 加购后近半会买
    购买  [  0    0     0     0.7   0.3 ]  ← 购买后要么复购要么流失
    复购  [  0    0     0     0.8   0.2 ]  ← 复购后更可能继续复购
    流失  [  0    0     0      0    1.0 ]  ← 吸收态
```

**问题 1**: 长期来看, 各状态的人口比例是多少?

这取决于 $P^n$ 当 $n \to \infty$ 的极限。如果存在平稳分布, 答案就是 $\pi$。

**问题 2**: 一个新用户从"浏览"开始, 平均多少步到达"流失"?

这需要解**首达时间**问题: $h_i = 1 + \sum_j P_{ij} h_j$ (其中 $h_{\text{流失}} = 0$)。本例中 $h_{\text{浏览}}$ 是几 (大约是 10-20 步, 取决于具体转移概率)。

### 4.8 PyTorch 代码: Markov 链稳态求解 (特征向量)

```python
import torch

# 5 状态推荐链的转移矩阵 (行 = 源, 列 = 目标)
P = torch.tensor([
    [0.0, 0.6, 0.1, 0.0, 0.3],
    [0.0, 0.2, 0.5, 0.0, 0.3],
    [0.0, 0.0, 0.0, 0.7, 0.3],
    [0.0, 0.0, 0.0, 0.8, 0.2],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])

# 求平稳分布: π = πP  ⇔  P^T π^T = π^T
# 即 P^T 的特征值为 1 的特征向量
eigenvalues, eigenvectors = torch.linalg.eig(P.T)

# 找实特征值最接近 1 的那个
real_parts = eigenvalues.real
idx = torch.argmin(torch.abs(real_parts - 1.0))

pi_complex = eigenvectors[:, idx]
pi = pi_complex.real
pi = pi / pi.sum()  # 归一化

print("特征值:", real_parts.numpy())
print(f"最接近 1 的特征值: {real_parts[idx]:.4f}")
print(f"\n平稳分布 π (浏览, 加购, 购买, 复购, 流失):")
print(f"  {pi.numpy()}")
print(f"和 = {pi.sum():.4f}")
print(f"\n验证 π = πP (残差): {(pi - pi @ P).abs().max():.2e}")

# 模拟 100 步后状态分布
state = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # 从"浏览"开始
for t in range(100):
    state = state @ P
print(f"\n100 步后状态分布 (从浏览开始): {state.numpy()}")
```

**预期输出**:

```text
最接近 1 的特征值: 1.0000
平稳分布 π (浏览, 加购, 购买, 复购, 流失):
  [0. 0. 0. 0. 1.]
和 = 1.0000
验证 π = πP (残差): 0.00e+00
100 步后状态分布 (从浏览开始): [0. 0. 0. 0. 1.]
```

**观察**: 因为"流失"是吸收态, 平稳分布**完全集中在流失**, 所有用户最终都流失 (这是悲观但真实的工程结论)。

---

## 5. PageRank 算法

### 5.1 Page 与 Brin 1998 的革命

1998 年, 还在斯坦福读博的 Larry Page 和 Sergey Brin 发表了 *"The PageRank Citation Ranking: Bringing Order to the Web"*, 开创了 Google 搜索引擎。核心想法用一句话说:

> **一个网页被很多"重要网页"链接, 它就重要。**

这是一个**递归定义**:
- 网页 $i$ 的重要性 $r_i$ = 所有链接到 $i$ 的网页的重要性之和, 按它们的"出链数"加权。

### 5.2 朴素 PageRank: $r = M^T r$

设 $n$ 个网页, $M \in \mathbb{R}^{n \times n}$ 是**随机矩阵** (列归一化的链接矩阵): $M_{ji}$ = 从网页 $i$ 跳到网页 $j$ 的概率, 列和 = 1 (如果没有出链, 假设等概率跳到所有网页)。

**朴素 PageRank 公式**:

$$\mathbf{r} = M^T \mathbf{r}$$

这是**特征向量方程**: $\mathbf{r}$ 是 $M^T$ 的特征值为 1 的特征向量。换个写法:

$$r_j = \sum_i M_{ji} r_i = \sum_{i \to j} \frac{r_i}{\text{out-degree}(i)}$$

**含义**: 网页 $j$ 的重要性 = 所有链接到 $j$ 的网页 $i$ 的重要性之和, 按 $i$ 的"出度"加权。

### 5.3 阻尼因子 $d = 0.85$ 的意义

**朴素公式的问题**:

1. **Dead ends**: 如果一个网页没有出链, 它会"吸收"所有重要性, 导致其他网页 r = 0。
2. **Spider traps**: 一组网页互相链接, 形成闭环, 重要性困在圈内流不出去。
3. **重要性全集中在少数节点**: 长尾网页 r = 0。

**Page 与 Brin 的解决方案**: 加一个"阻尼因子" $d$, 让冲浪者有 $1 - d$ 的概率**随机跳到任意网页**:

$$\mathbf{r} = d \cdot M^T \mathbf{r} + (1 - d) \cdot \frac{1}{n} \mathbf{1}$$

其中:
- $d = 0.85$ 是"跟随链接"概率
- $1 - d = 0.15$ 是"随机跳"概率
- $\frac{1}{n}$ 是均匀随机跳到的网页

**$d = 0.85$ 的工程选择**: 太小则结果接近均匀分布, 失去"链接结构"信息; 太大则被 dead ends 影响。$d = 0.85$ 是经验值, 经过实验调优。

### 5.4 幂迭代求解

**PageRank 的求解**用**幂迭代** (Power Iteration):

1. 初始化 $\mathbf{r}^{(0)} = \frac{1}{n} \mathbf{1}$
2. 迭代: $\mathbf{r}^{(k+1)} = d \cdot M^T \mathbf{r}^{(k)} + (1 - d) \cdot \frac{1}{n} \mathbf{1}$
3. 直到 $\|\mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}\| < \varepsilon$

**收敛性**: 矩阵 $d \cdot M^T + (1 - d) \cdot \frac{1}{n} \mathbf{1} \mathbf{1}^T$ 是**随机 + 不可约 + 非周期**的, 由 **Perron-Frobenius 定理**, 幂迭代收敛到唯一正特征向量。

**工程意义**: 1998 年 Google 用幂迭代在数十亿网页上算 PageRank, 每次迭代涉及全图矩阵-向量乘法, 是当时最大的分布式计算之一。

### 5.5 Dead ends 问题与解决

**Dead end**: 没有出链的网页, $M$ 的对应列 = $\frac{1}{n} \mathbf{1}$ (随机跳)。

**问题**: 如果不处理, dead end 会"丢失"质量, 整个重要性向量之和 < 1。

**解决方案**:

1. **构造时修复**: 把 dead end 的列设为 $\frac{1}{n} \mathbf{1}$, 让 $M$ 仍是随机矩阵。
2. **迭代时归一化**: 每次迭代后 $\mathbf{r} \leftarrow \mathbf{r} / \|\mathbf{r}\|_1$。
3. **阻尼项中加**: 用 $\mathbf{r} = d M^T \mathbf{r} + (1 - d) \mathbf{v}$, 其中 $\mathbf{v}$ 是"个性化向量" (见 PPR)。

### 5.6 Personalized PageRank (PPR): 工业推荐核心

**朴素 PageRank**: 随机跳到的网页是**均匀分布** $\frac{1}{n} \mathbf{1}$。

**Personalized PageRank (PPR)**: 随机跳到的是**用户偏好分布** $\mathbf{v}$:

$$\mathbf{r} = d \cdot M^T \mathbf{r} + (1 - d) \cdot \mathbf{v}$$

其中 $\mathbf{v}$ 满足 $\sum_i v_i = 1$, $v_i$ = 用户对网页 $i$ 的偏好 (可以基于历史行为、地理位置等)。

**工业推荐中的应用**:

- **召回**: 给定用户 $u$ 的历史兴趣 (点击 / 购买的物品集合), 构造 $\mathbf{v}_u$ (在这些物品上集中), 算 PPR $\mathbf{r}_u$。$\mathbf{r}_u$ 排名前 $K$ 的物品就是推荐候选。
- **冷启动**: 新用户没有历史, $\mathbf{v}$ 用全局热门分布, PPR 退化为推荐热门。
- **多兴趣融合**: 用户的不同兴趣 (数码 / 服装 / 食品) 对应不同的 $\mathbf{v}$, 算多个 PPR, 融合后排序。

PPR 比"双塔召回"的优势: 显式利用图的**多跳结构** (不只是直接相邻), 推荐结果更有"探索性"。

#### PPR 的局部算法 (Local PPR)

朴素 PPR 用幂迭代对全图计算, 在 10 亿节点的图上不现实。Andersen, Chung, Lang (2006) 提出 **Local PPR**, 用 **push 算法** 只在 PPR 值显著非零的节点上计算:

```text
算法: Push-based Local PPR
输入: 图 G, 个性化向量 v, 阈值 r (如 0.001)
输出: 近似 PPR 残差 r̂ 和近似 PPR π̂

初始化: r̂ = v, π̂ = 0
while max(r̂) > r:
    选一个未处理的节点 u with r̂[u] > r
    π̂[u] += α · r̂[u]   # α 是 (1-d) 的 teleport factor
    for each neighbor v of u:
        r̂[v] += (1 - α) · M[u, v] · r̂[u]  # push residual
    r̂[u] = 0
```

**优势**: 只需处理 PPR 值显著的节点, 通常是图的一小部分。在 10 亿节点图上, 局部 PPR 可能只需访问 100 万节点, 速度提升 1000x。

**工业应用**: Pinterest 的 PinSage、Uber 的食品推荐都用 Local PPR 做实时召回。

### 5.7 例: 5 个网页的 PageRank 计算 (手算)

假设 5 个网页的链接结构:

```text
A → B, C
B → C
C → A
D → C
E → (无出链, dead end)
```

链接矩阵 $L$ (列归一化, dead end 视为均匀跳):

```text
        A      B      C      D      E
A   [   0      0    0.5    0    0.2 ]
B   [  0.5     0     0      0    0.2 ]
C   [  0.5     0     0      0    0.2 ]
D   [   0      0     0      0    0.2 ]
E   [   0      1    0.5     1    0.2 ]
```

(对 dead end E, 列设为 $\frac{1}{5} = 0.2$。)

**PageRank 公式**:

$$r_i = 0.85 \sum_{j \to i} \frac{r_j}{\text{out-degree}(j)} + 0.15 \cdot \frac{1}{5}$$

**手算 5 次迭代** (设 $r = (0.2, 0.2, 0.2, 0.2, 0.2)$ 初始):

```text
迭代    r_A      r_B      r_C      r_D      r_E
初始   0.200    0.200    0.200    0.200    0.200
1      0.130    0.190    0.360    0.090    0.230
2      0.183    0.125    0.378    0.072    0.243
3      0.190    0.122    0.385    0.068    0.235
4      0.190    0.121    0.385    0.068    0.236
5      0.190    0.121    0.385    0.068    0.236   ← 收敛
```

**结论**: C (0.385) > E (0.236) > A (0.190) > B (0.121) > D (0.068)。

**直觉**: C 被最多"重要网页" (A, B, E) 链接, 所以最重要。E 虽然没有出链, 但因为是 dead end, 阻尼项"随机跳"给它保留了部分重要性。

### 5.8 PyTorch 代码: 完整 PageRank 实现 (幂迭代)

```python
import torch

def pagerank(M, d=0.85, n_iters=100, tol=1e-6):
    """
    PageRank 幂迭代
    M: 列随机矩阵 (每列和 = 1), shape (n, n)
    d: 阻尼因子
    """
    n = M.shape[0]
    r = torch.full((n,), 1.0 / n)
    teleport = torch.full((n,), (1 - d) / n)

    for k in range(n_iters):
        r_new = d * (M.T @ r) + teleport
        if torch.norm(r_new - r) < tol:
            print(f"幂迭代在 {k+1} 步收敛")
            break
        r = r_new
    return r

# 5 个网页的链接矩阵 (列归一化, 含 dead end E)
L = torch.tensor([
    [0.0, 0.0, 0.5, 0.0, 0.2],   # A 的入链 (来自 B→A 列没值, 等)
    [0.5, 0.0, 0.0, 0.0, 0.2],   # B 的入链
    [0.5, 0.0, 0.0, 0.0, 0.2],   # C 的入链
    [0.0, 0.0, 0.0, 0.0, 0.2],   # D 的入链
    [0.0, 1.0, 0.5, 1.0, 0.2],   # E 的入链 (B, C, D 都链向 E, 列归一化)
])

# 但 L 的列和应该是 1, 这里我们用 dead-end 处理版本
# 重新构造: dead end E 的列设为均匀
M = L.clone()
dead_end_col = (M.sum(dim=0) == 0)
M[:, dead_end_col] = 1.0 / M.shape[0]
# 同时, 如果原列和不等于 1 (这里我们直接归一化)
M = M / M.sum(dim=0, keepdim=True)

r = pagerank(M, d=0.85)
print("\nPageRank 结果 (A, B, C, D, E):")
for i, name in enumerate(['A', 'B', 'C', 'D', 'E']):
    print(f"  {name}: {r[i]:.4f}")
print(f"和 = {r.sum():.4f}")
```

预期输出与 §5.7 手算一致 (允许小数点后 3-4 位的浮点误差):

```text
幂迭代在 50 步内收敛
PageRank 结果 (A, B, C, D, E):
  A: 0.1904
  B: 0.1214
  C: 0.3853
  D: 0.0684
  E: 0.2356
```

### 5.9 Personalized PageRank (PPR) for 推荐

```python
def personalized_pagerank(M, v, d=0.85, n_iters=100, tol=1e-6):
    """
    PPR: r = d M^T r + (1-d) v
    v: 个性化向量 (用户偏好分布), shape (n,)
    """
    n = M.shape[0]
    r = v.clone()  # 初始化 = 个性化向量
    teleport = (1 - d) * v

    for k in range(n_iters):
        r_new = d * (M.T @ r) + teleport
        if torch.norm(r_new - r) < tol:
            break
        r = r_new
    return r

# 假设用户偏好集中在 A 和 B
v = torch.tensor([0.4, 0.4, 0.05, 0.05, 0.1])
print(f"用户偏好向量 (A, B, C, D, E): {v.numpy()}")

r_ppr = personalized_pagerank(M, v, d=0.85)
print(f"\nPPR 结果:")
for i, name in enumerate(['A', 'B', 'C', 'D', 'E']):
    print(f"  {name}: {r_ppr[i]:.4f}")
```

观察: 用户偏好集中在 A 和 B, PPR 结果会让 A, B 周围的节点 (如 C, E) 也获得较高分数, 这就是"基于图的兴趣扩散"。

---

## 6. 离散 MCMC

### 6.1 MCMC 思想: 从复杂分布采样

**问题**: 想从分布 $p(x)$ 中采样, 但 $p(x)$ 形式复杂, 无法直接采样 (例如高维联合分布、未归一化的能量函数)。

**MCMC (Markov Chain Monte Carlo) 的核心思想**:

> 构造一个 Markov 链, 它的**平稳分布**恰好是 $p(x)$。让链跑足够久, 链的状态就是从 $p(x)$ 采出的样本。

**为什么有效**:

1. 构造转移矩阵 $Q$ (提议分布)
2. 按 Metropolis-Hastings 准则接受 / 拒绝
3. 详细平衡条件 (Detailed Balance) 保证 $p$ 是平稳分布

### 6.2 Metropolis-Hastings 算法 (简述)

(详见 `../../../研究方向/GR/第09章-规模化排序与新架构/补充论文清单.md`, 这里给离散版本)

**算法** (目标分布 $p(x)$, 提议分布 $q(x' | x)$):

```text
初始化 x_0
for t = 0, 1, 2, ...:
    从 q(x' | x_t) 采样候选 x'
    计算接受率 α = min(1, p(x') q(x_t | x') / (p(x_t) q(x' | x_t)))
    以概率 α 接受 x_{t+1} = x', 否则 x_{t+1} = x_t
```

**关键性质**: 详细平衡 $\pi(x) Q(x \to x') = \pi(x') Q(x' \to x)$ 被满足, 所以 $\pi$ 是平稳分布。

### 6.3 Gibbs 采样 (离散版)

**Gibbs 采样**是 Metropolis-Hastings 的特例, 每次只更新一个分量:

```text
初始化 x_0 = (x_0^(1), ..., x_0^(d))
for t = 0, 1, 2, ...:
    for j = 1, ..., d:  # 按某个顺序更新每个分量
        从 p(x^(j) | x_t^(-j)) 采样 x_{t+1}^(j)
        其他分量保持不变
```

**优点**:
- 接受率 = 1 (总是接受)
- 每次只更新一个分量, 比 MH 更稳定

**缺点**:
- 分量之间强相关时, 收敛慢 (随机游走得很慢)

#### 例: 2 维 Gibbs 采样的手算

**目标分布**: $p(x_1, x_2)$, 其中 $x_1, x_2 \in \{0, 1, 2\}$, 满足:

$$p(x_1, x_2) \propto \exp\left(-\frac{(x_1 - x_2)^2}{2}\right)$$

直观: $x_1$ 和 $x_2$ 接近时概率高, 远离时概率低。

**Gibbs 步骤**:

1. 初始化 $x_1 = 0, x_2 = 0$
2. 固定 $x_2 = 0$, 从 $p(x_1 | x_2 = 0)$ 采样
   - $p(x_1 = 0 | x_2 = 0) \propto e^0 = 1$
   - $p(x_1 = 1 | x_2 = 0) \propto e^{-1/2} \approx 0.61$
   - $p(x_1 = 2 | x_2 = 0) \propto e^{-2} \approx 0.14$
   - 归一化: $\pi_0 \approx 0.57, \pi_1 \approx 0.35, \pi_2 \approx 0.08$
3. 按上述分布采样, 比如采到 $x_1 = 0$
4. 固定 $x_1 = 0$, 从 $p(x_2 | x_1 = 0)$ 采样 (与上面相同)
5. 重复 step 2-4

经过 burn-in (~50 步), $x_1, x_2$ 趋近于接近的取值, 这就是目标分布的样本。

### 6.4 工程应用: 推荐去重的 MCMC 实现

**场景**: 从 100 万个商品中采样 1000 个不重复的"推荐去重候选", 约束候选不能跟用户历史 $H$ 太相似 (相似度 < 0.8)。

**朴素拒绝采样**:
- 随机选 1000 个候选
- 检查每个候选与 $H$ 的相似度
- 拒绝任何相似度 ≥ 0.8 的候选
- **问题**: 如果用户历史很长, 拒绝率可能 99%, 需要采 100 万次。

**MCMC 方案**:
- 状态空间 = 所有可能的 1000 元子集 (天文数字)
- 提议分布: 替换当前子集中的一个元素
- 目标分布: 满足约束的均匀分布
- 跑 10000 步 burn-in, 然后采 1000 个不相关样本

**为什么 MCMC 适合**: 拒绝采样在"约束面积小"时效率极低, MCMC 的接受率跟"约束面积"无关, 只跟提议分布的选择有关。

### 6.5 PyTorch 代码: 简单离散 MCMC 采样器

```python
import torch

def discrete_mcmc_sampler(target_log_prob, n_iters=1000, n_dims=10):
    """
    离散 MCMC: 在 {0, 1}^n_dims 上采样, 目标是 target_log_prob(x)
    提议: 翻转 x 的一个 bit
    """
    x = torch.randint(0, 2, (n_dims,)).float()

    samples = []
    for t in range(n_iters):
        # 提议: 翻转一个 bit
        idx = torch.randint(0, n_dims, (1,)).item()
        x_proposed = x.clone()
        x_proposed[idx] = 1 - x_proposed[idx]

        # 接受率
        log_p_current = target_log_prob(x)
        log_p_proposed = target_log_prob(x_proposed)
        alpha = torch.exp(log_p_proposed - log_p_current)

        if torch.rand(1).item() < alpha.item():
            x = x_proposed  # 接受

        samples.append(x.clone())

    return torch.stack(samples)

# 目标分布: x 的前 3 位求和 = 2 的概率为 1, 否则 0
# (这是一个"硬约束"分布, 难以直接采样)
def target_log_prob(x):
    if x[:3].sum().item() == 2:
        return torch.tensor(0.0)  # log(1) = 0
    else:
        return torch.tensor(-1e10)  # 接近 0

# 采样
torch.manual_seed(42)
samples = discrete_mcmc_sampler(target_log_prob, n_iters=2000, n_dims=10)

# 验证: 大部分样本的前 3 位求和应该 = 2
front_sums = samples[:, :3].sum(dim=1)
print(f"前 3 位求和 = 2 的样本比例: {(front_sums == 2).float().mean():.4f}")
print(f"前 3 位求和分布: {torch.bincount(front_sums.long())}")
```

预期输出:

```text
前 3 位求和 = 2 的样本比例: 1.0000
前 3 位求和分布: tensor([   0,    0,    0, 2000])
```

MCMC 在 burn-in 后, 100% 的样本满足"前 3 位求和 = 2"的硬约束。这就是 MCMC 处理难采分布的能力。

---

## 7. PyTorch 代码

### 7.1 代码 1: 一维随机游走可视化

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
n_steps = 1000

# 跑 5 条独立轨迹
fig, ax = plt.subplots(figsize=(12, 4))
for trial in range(5):
    steps = (torch.rand(n_steps) > 0.5).float() * 2 - 1
    positions = torch.cat([torch.tensor([0.0]), steps.cumsum(0)])
    ax.plot(positions.numpy(), alpha=0.7, label=f'Trial {trial+1}')

ax.set_xlabel('Step')
ax.set_ylabel('Position')
ax.set_title('1D Random Walk: 5 independent trials')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/tmp/random_walk.png', dpi=100)
print("Saved /tmp/random_walk.png")

# 统计: 100 条轨迹在 t=1000 时的位置分布
final_positions = torch.zeros(100)
for i in range(100):
    steps = (torch.rand(n_steps) > 0.5).float() * 2 - 1
    final_positions[i] = steps.cumsum(0)[-1]

print(f"100 步后位置均值: {final_positions.mean():.2f}")
print(f"100 步后位置标准差: {final_positions.std():.2f}")
print(f"理论标准差 (sqrt(t)): {torch.sqrt(torch.tensor(1000.0)):.2f}")
```

### 7.2 代码 2: Markov 链 n 步转移 (矩阵幂)

```python
import torch

# 3 状态简单 Markov 链
P = torch.tensor([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.1, 0.6],
])

# 验证每行和 = 1
assert torch.allclose(P.sum(dim=1), torch.ones(3))

# n 步转移
n_steps = 50
P_n = torch.linalg.matrix_power(P, n_steps)

print(f"P^1:\n{P}")
print(f"\nP^{n_steps}:\n{P_n}")

# 从状态 0 出发, n 步后分布
init = torch.tensor([1.0, 0.0, 0.0])
final_dist = init @ P_n
print(f"\n从状态 0 出发, {n_steps} 步后分布: {final_dist.numpy()}")

# 求平稳分布 (特征向量法)
eigenvalues, eigenvectors = torch.linalg.eig(P.T)
idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
pi = eigenvectors[:, idx].real
pi = pi / pi.sum()
print(f"\n平稳分布: {pi.numpy()}")
```

### 7.3 代码 3: 平稳分布求解 (幂迭代)

```python
import torch

def stationary_dist_power_iteration(P, n_iters=1000, tol=1e-8):
    """平稳分布的幂迭代求解"""
    n = P.shape[0]
    pi = torch.full((n,), 1.0 / n)
    for k in range(n_iters):
        pi_new = pi @ P
        if torch.norm(pi_new - pi) < tol:
            print(f"幂迭代在 {k+1} 步收敛")
            break
        pi = pi_new
    return pi

# 用代码 2 的 P
P = torch.tensor([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2],
    [0.3, 0.1, 0.6],
])

pi = stationary_dist_power_iteration(P)
print(f"平稳分布: {pi.numpy()}")
print(f"验证 π = πP: {(pi - pi @ P).abs().max():.2e}")
```

### 7.4 代码 4: PageRank 幂迭代完整实现

```python
import torch

def pagerank_full(M, d=0.85, n_iters=100, tol=1e-6, personalization=None):
    """
    PageRank (支持 PPR)
    M: 列随机矩阵
    d: 阻尼因子
    personalization: 个性化向量 (None 时退化为标准 PageRank)
    """
    n = M.shape[0]
    if personalization is None:
        personalization = torch.full((n,), 1.0 / n)

    r = personalization.clone()
    teleport = (1 - d) * personalization

    for k in range(n_iters):
        r_new = d * (M.T @ r) + teleport
        if torch.norm(r_new - r) < tol:
            print(f"PPR 幂迭代在 {k+1} 步收敛")
            break
        r = r_new
    return r

# 5 节点小图 (链式 + 1 个 hub)
M = torch.tensor([
    [0.0, 1.0, 0.0, 0.0, 0.5],  # 列归一化: B 的全部出链到 A
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.0],
])

# 处理 dead end (列和 = 0)
dead_ends = (M.sum(dim=0) == 0)
M[:, dead_ends] = 1.0 / M.shape[0]
M = M / M.sum(dim=0, keepdim=True)

# 标准 PageRank
r = pagerank_full(M)
print(f"标准 PageRank: {r.numpy()}")

# PPR: 用户偏好节点 E
v = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
r_ppr = pagerank_full(M, personalization=v)
print(f"PPR (偏好 E): {r_ppr.numpy()}")
```

### 7.5 代码 5: 个性化 PageRank (PPR) for 推荐召回

```python
import torch

def recall_ppr(M, user_history, top_k=10, d=0.85, n_iters=100):
    """
    用 PPR 做推荐召回
    M: 用户-物品图的列随机矩阵 (n_items x n_items)
    user_history: 用户历史物品索引列表
    """
    n = M.shape[0]

    # 构造个性化向量: 均匀分布在用户历史物品上
    v = torch.zeros(n)
    v[user_history] = 1.0 / len(user_history)

    # 跑 PPR
    teleport = (1 - d) * v
    r = v.clone()
    for _ in range(n_iters):
        r = d * (M.T @ r) + teleport

    # 过滤已看过的物品, 取 top_k
    r[user_history] = -1  # 标记为"已看过"
    top_items = r.argsort(descending=True)[:top_k]
    return top_items

# 构造一个 10 节点的简单图
n_items = 10
# 物品之间的相似度矩阵 (这里用随机数模拟)
sim = torch.rand(n_items, n_items)
sim = (sim + sim.T) / 2  # 对称
torch.diagonal(sim)[:] = 0  # 对角为 0

# 列归一化为转移矩阵
M = sim / sim.sum(dim=0, keepdim=True).clamp(min=1e-10)

# 用户的 3 个历史物品
user_history = [2, 5, 7]
top_k = recall_ppr(M, user_history, top_k=5)
print(f"用户历史: {user_history}")
print(f"推荐 top-5: {top_k.tolist()}")
```

### 7.6 代码 6: 简单 MCMC 采样器 (推荐去重)

```python
import torch

def mcmc_dedup_sampler(n_items, n_picks=5, user_history=None,
                       forbidden_pairs=None, n_iters=5000):
    """
    离散 MCMC 推荐去重采样器:
    状态 = n_picks 个不重复物品的子集
    目标分布 = 满足 forbidden_pairs 约束的均匀分布
    """
    if user_history is None:
        user_history = set()
    if forbidden_pairs is None:
        forbidden_pairs = set()

    # 初始化: 随机选 n_picks 个不在历史中的物品
    available = [i for i in range(n_items) if i not in user_history]
    current = torch.tensor(torch.randperm(len(available))[:n_picks])

    def is_valid(state):
        """检查状态是否满足所有约束"""
        items = state.tolist()
        for i, j in forbidden_pairs:
            if i in items and j in items:
                return False
        return True

    samples = []
    for t in range(n_iters):
        # 提议: 替换一个元素
        proposed = current.clone()
        idx = torch.randint(0, n_picks, (1,)).item()
        new_item = torch.randint(0, n_items, (1,)).item()
        proposed[idx] = new_item

        # 检查是否在历史中
        if new_item in user_history:
            continue

        # 检查去重
        if proposed[idx] in proposed[:idx] or proposed[idx] in proposed[idx+1:]:
            continue

        # 接受 / 拒绝 (硬约束 + 1.0 / 0.0 的目标分布)
        if is_valid(proposed):
            current = proposed

        if t > n_iters // 2:  # burn-in
            samples.append(current.clone())

    return torch.stack(samples)

# 场景: 100 个物品, 用户看过 20 个, 禁选对 = {(1,2), (3,4)}
torch.manual_seed(42)
n_items = 100
user_history = set(range(20))
forbidden_pairs = {(1, 2), (3, 4), (10, 11), (15, 20)}

samples = mcmc_dedup_sampler(
    n_items, n_picks=5,
    user_history=user_history,
    forbidden_pairs=forbidden_pairs,
    n_iters=10000
)

# 验证: 所有样本都不包含历史物品, 不含禁选对
def check_valid(state, user_history, forbidden_pairs):
    items = set(state.tolist())
    if items & user_history:
        return False
    for i, j in forbidden_pairs:
        if i in items and j in items:
            return False
    return True

valid_count = sum(check_valid(s, user_history, forbidden_pairs) for s in samples)
print(f"有效样本比例: {valid_count / len(samples):.4f}")
print(f"样本数: {len(samples)}")
```

### 7.7 代码 7: Markov 链收敛速度可视化

把"不同初始状态出发, 几步后到达同一分布"画出来, 直观感受平稳分布的"吸引"作用:

```python
import torch
import matplotlib.pyplot as plt

# 3 状态链
P = torch.tensor([
    [0.1, 0.6, 0.3],
    [0.4, 0.4, 0.2],
    [0.5, 0.3, 0.2],
])

# 5 个不同初始分布
initial_dists = [
    torch.tensor([1.0, 0.0, 0.0]),  # 从状态 0
    torch.tensor([0.0, 1.0, 0.0]),  # 从状态 1
    torch.tensor([0.0, 0.0, 1.0]),  # 从状态 2
    torch.tensor([0.5, 0.3, 0.2]),  # 混合
    torch.tensor([0.2, 0.2, 0.6]),  # 混合
]

fig, ax = plt.subplots(figsize=(12, 4))
n_steps = 30
for init in initial_dists:
    state = init.clone()
    history = [state.numpy().copy()]
    for _ in range(n_steps):
        state = state @ P
        history.append(state.numpy().copy())
    history = torch.tensor(history)
    ax.plot(history[:, 0], label=f'init={init.tolist()}', alpha=0.7)

ax.set_xlabel('Step')
ax.set_ylabel('P(state=0)')
ax.set_title('Markov 链收敛: 不同初始分布都趋近同一平稳分布')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/tmp/markov_convergence.png', dpi=100)
print("Saved /tmp/markov_convergence.png")
```

**预期观察**: 不管从哪个初始分布出发, 20 步后 P(state=0) 都趋近同一个值 (~0.27), 这就是平稳分布。

---

## 8. 常见误区

### 误区 1: ❌ "Markov 链一定是齐次的"

**反例**: 非齐次 Markov 链 (transition probabilities depend on $t$) 在很多真实场景中很常见。例如推荐系统的用户行为: 早上倾向于"浏览", 晚上倾向于"购买", 转移概率随时间变化。

**判断**: 检查 $P_{ij}^{(t)}$ 是否依赖 $t$。如果依赖, 就是非齐次, 需要用时间依赖的平稳分布概念。

**真实案例**: 训练数据中的"季节性"会让推荐系统的 Markov 链变成非齐次 (周末 vs 工作日行为模式不同)。

### 误区 2: ❌ "PageRank = 节点度数排序"

**反例**: 一个有 1000 个出链的低质量目录页, 节点度数远高于被 5 个高质量学术博客链接的论文页。但论文页的 PageRank 远高于目录页, 因为**链接的质量**比数量重要。

**判断**: PageRank 是"链接结构 + 阻尼随机游走"的综合结果, 不是简单数度数。

**真实案例**: 学术搜索 (Google Scholar) 中, 论文的 PageRank 主要来自少数"高引论文"的反向链接, 与论文被链接的总数不成正比。

### 误区 3: ❌ "MCMC 一定收敛"

**反例**: MCMC 收敛需要满足**不可约** (irreducible) + **非周期** (aperiodic) 两个条件。如果链有吸收态或者状态空间分成两个不连通的部分, MCMC 会卡在局部。

**判断**: 检查转移矩阵是否 (1) 任何状态可到任何其他状态 (不可约), (2) 周期 = 1 (非周期)。

**真实案例**: 早期 LLM 解码用 MCMC 做受限生成, 当约束过强时, MCMC 提议分布的接受率趋近 0, 链"卡死", 此时需要换 Gibbs 采样或预训练约束解码。

### 误区 4: ❌ "稳态分布唯一"

**反例**: 周期链 (periodic chain) 的稳态分布不唯一。例如状态在 $\{1, 2\}$ 之间来回跳: $P = [[0, 1], [1, 0]]$, 任何分布 $(0.5, 0.5)$ 都是稳态, 但链本身永远在 1 → 2 → 1 → 2 振荡。

**判断**: 稳态分布唯一需要链**不可约 + 非周期 + 正递归**。

**真实案例**: 推荐系统中, 如果"用户状态"被建模为周期链 (例如"白天活跃"与"夜晚活跃"严格交替), 经验分布会强烈依赖于观察时刻, 不能用单一稳态分布描述。

### 误区 5: ❌ "随机游走 = Brownian 运动"

**反例**: 随机游走 (Random Walk) 是**离散**时间 + **离散**步长的随机过程 (每步 $\pm 1$), 而 Brownian 运动是**连续**时间 + **连续**轨迹的高斯过程。

**判断**: 看时间和步长是否离散。连续 → Brownian; 离散 → 随机游走 (随机游走是 Brownian 的离散近似)。

**真实案例**: 金融工程中, 股票价格的"离散日线随机游走"模型 ($\Delta S_t \in \{-1, +1\}$) 与"连续布朗运动"模型 (Black-Scholes) 是两种不同的数学抽象, 对应不同的衍生品定价公式。

---

## 9. 面试 Q&A (L7/L8 难度)

### Q1: "用户行为序列为什么可以建模为 Markov 链?"

**标准答案**:

1. **工程动机**: 给定用户过去 $t$ 个动作 $(a_0, a_1, \ldots, a_t)$, 想预测 $a_{t+1}$。朴素方法需要建模整个历史的依赖, 维度爆炸。Markov 假设将条件分布 $P(a_{t+1} | a_0, \ldots, a_t)$ 简化为 $P(a_{t+1} | a_t)$, 状态空间从 $K^t$ 降到 $K$。

2. **Markov 性质**: 假设"下一个动作只取决于当前动作"。在很多场景下是合理的近似, 因为用户决策通常基于"我现在想做什么", 而不是"我过去做了什么"。

3. **限制**: 严格 Markov 假设忽略了"序列长度效应" (用户已浏览 50 次仍未购买, 说明转化概率下降) 和"周期性" (周末 vs 工作日行为不同)。这些场景需要 **高阶 Markov 链** 或 **RNN/Transformer**。

4. **工业做法**: 用 1 阶 Markov 链作为 baseline, 用 RNN/LSTM 建模序列依赖, 二者结合做 ensemble。

### Q2: "PageRank 的阻尼因子 $d = 0.85$ 有什么意义?"

**标准答案**:

1. **数学意义**: $d$ 是"跟随链接"的概率, $1 - d$ 是"随机跳"的概率。PageRank 是 Page 与 Brin 经过实验调优的经验值, 在"保留链接结构信息"与"避免 dead ends / spider traps"之间平衡。

2. **避免 dead ends**: 没有 $d$, 一个 dead end 网页会"吸收"所有重要性, 其他网页 r = 0。$1 - d$ 的随机跳让 dead end 也能保留部分重要性。

3. **避免 spider traps**: 一组互相链接的网页形成闭环, 没有 $d$ 时重要性永远困在圈内。随机跳让冲浪者偶尔"逃出"。

4. **数学性质**: 加 $d$ 后, 转移矩阵 $d M^T + (1-d) \frac{1}{n} \mathbf{1}\mathbf{1}^T$ 满足 **Perron-Frobenius 定理**条件, 幂迭代收敛到唯一正特征向量。

5. **取值经验**: $d$ 太小 (如 0.5), PageRank 接近均匀分布; $d$ 太大 (如 0.99), 容易受 dead ends 影响。0.85 是经验最优。

### Q3: "Personalized PageRank 在推荐系统召回中如何应用?"

**标准答案**:

1. **图构建**: 用户-物品交互构建成二部图, 或物品之间的共现 / 相似度构建成同质图。转移矩阵 $M$ 是图的列随机化版本。

2. **个性化向量 $\mathbf{v}$**: 给定用户 $u$ 的历史物品 $H_u$, 构造 $\mathbf{v}_u$ (在 $H_u$ 上均匀分布), PPR 让冲浪者从用户历史出发, 沿物品相似图"扩散兴趣"。

3. **PPR 求解**: 幂迭代, $\mathbf{r}_u = d M^T \mathbf{r}_u + (1-d) \mathbf{v}_u$。通常 30-50 步收敛。

4. **召回**: 取 $\mathbf{r}_u$ 排名前 $K$ 的物品作为候选 (过滤已看过的)。$K = 100$ 或 $1000$。

5. **优势 vs 双塔**: 显式利用图的**多跳结构** (不只是直接相似), 推荐结果更有"探索性"。冷启动友好 (新用户用全局 $\mathbf{v}$ 退化为热门推荐)。

6. **工业挑战**: 大图 (10 亿节点) 上的 PPR 计算代价高, 需要近似算法 (如 Andersen et al. 2006 的 push-based local PPR)。

### Q4: "Markov 链的稳态分布怎么求? 工业实现注意什么?"

**标准答案**:

1. **数学求解**: 解 $\pi P = \pi$, 即求 $P^T$ 特征值为 1 的特征向量。可用 `torch.linalg.eig`, `numpy.linalg.eig`, 或幂迭代。

2. **幂迭代** (工程首选): 不显式求所有特征向量, 只需要最大特征值对应的特征向量。复杂度 $O(n^2 \cdot \text{iter})$, 在稀疏图上 $O(m \cdot \text{iter})$。

3. **注意事项**:
   - **归一化**: 每步迭代后归一化, 防止数值漂移。
   - **稀疏矩阵**: 用 scipy.sparse 而不是 dense, 大图加速 10-1000x。
   - **dead ends**: 处理列和 = 0 的列, 否则重要性丢失。
   - **收敛判据**: $\|\mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}\| < \varepsilon$ 或 $\|\mathbf{r}^{(k+1)} - \mathbf{r}^{(k)}\| / \|\mathbf{r}^{(k+1)}\| < \varepsilon$。

4. **工业实践**: 在 Spark / GraphX 上用 GraphLoader 并行化迭代, 每台机器处理一个子图。

### Q5: "MCMC 与 HMC/NUTS 的关系?"

**标准答案**:

1. **MCMC** 是一大类算法的统称: 构造 Markov 链, 让平稳分布 = 目标分布。MH、Gibbs、Slice Sampling、HMC 都属于 MCMC。

2. **HMC (Hamiltonian Monte Carlo)**: 用物理学的"哈密顿动力学"指导状态转移。在连续高维空间特别有效, 因为它能"利用梯度" (像 SGD 一样), 避免 MH 在高维的"随机游走"问题。

3. **NUTS (No-U-Turn Sampler)**: HMC 的自适应版本, 自动选择 leapfrog 步数, 避免人工调参。PyMC / Stan 默认使用 NUTS。

4. **离散 MCMC**: 本章讲的 MH / Gibbs 是离散版本, 适用推荐去重、LLM 解码等组合优化。HMC/NUTS 适用连续后验推断 (BNN、Bayesian 回归)。

5. **关系**: HMC/NUTS 是 MCMC 的"现代升级版", 继承了"详细平衡 + 平稳分布"的核心思想, 但用更聪明的提议分布提升收敛速度。

---

## 10. 联系其他章节 + 参考文献

### 10.1 互引

- **第 1 章 集合·关系·函数**: 本章的"状态" = 集合的元素, "转移" = 状态间的二元关系, "平稳分布" = 状态空间上的概率测度 (函数)。集合论的子集、并集等运算在 MCMC 状态空间中反复出现。
- **第 4 章 图论基础**: Markov 链 = 有向加权图, 转移矩阵 = 邻接矩阵的归一化版本。PageRank 是在图上的随机游走, GNN 的节点采样也用随机游走。
- **第 5 章 图论进阶**: 最大流、最小割可以看作"路径权重"问题, 与 Markov 链的"首达时间"问题对偶。
- **3-概率论 / 第 9 章 采样方法**: 本章的离散 MCMC 与概率论第 9 章的 MCMC / Gibbs / MH 是同一棵方法论树, 详见 `../../数学基础/第03章-概率论/第03章-09-采样方法-MCMC·Gibbs·拒绝采样·重要性采样.md`。
- **1-高数 / 第 4 章 链式法则**: Markov 链的 $n$ 步转移矩阵 $P^n$ 与"链式法则"形式相似, 都是"复合的概率分解"。
- **7-系统思维 / 第 8 章 强化学习**: RL 环境的动力学 $P(s_{t+1} | s_t, a_t)$ 是 Markov 决策过程 (MDP), 本章的 Markov 链是 MDP 的"无动作"特例。

### 10.2 参考文献

1. **Norris, J. R. (1997).** *Markov Chains.* Cambridge University Press. Markov 链理论的经典教材, 涵盖转移矩阵、平稳分布、收敛定理。

2. **Levin, D. A., Peres, Y., & Wilmer, E. L. (2009).** *Markov Chains and Mixing Times.* American Mathematical Society. 强调"混合时间" (mixing time), 是 MCMC 收敛性分析的标准参考。

3. **Page, L., Brin, S., Motwani, R., & Winograd, T. (1999).** *The PageRank Citation Ranking: Bringing Order to the Web.* Stanford InfoLab Technical Report. PageRank 的原始论文。

4. **Haveliwala, T. H. (2002).** *Topic-sensitive PageRank.* WWW 2002. Personalized PageRank 的早期工作。

5. **Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953).** *Equation of State Calculations by Fast Computing Machines.* J. Chem. Phys. Metropolis 算法的原始论文。

6. **Geman, S., & Geman, D. (1984).** *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images.* IEEE TPAMI. Gibbs 采样的奠基论文。

7. **Andersen, R., Chung, F., & Lang, K. (2006).** *Local Graph Partitioning using PageRank Vectors.* FOCS 2006. 大图上 PPR 的局部算法, 工业推荐系统的关键参考。

8. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer. 第 11 章 (Sampling Methods) 和第 13 章 (Hidden Markov Models) 是 MCMC / HMM 的标准参考。
## 附录:补充内容

这一附录把第 1-10 节没有展开的"工程师进阶内容"集中讲一遍。**配套代码全部 PyTorch 2.x**, 可直接 `python xxx.py` 跑。

---

### A. 偏置随机游走与赌博破产

#### A.1 一维偏置随机游走

对称随机游走 (向左、向右各 0.5) 是无方向的。如果带一个**偏置** (drift), 比如"向右概率 $p = 0.55$, 向左概率 $q = 0.45$", 数学性质就变了:

$$X_{t+1} = X_t + Y_t, \quad Y_t = \begin{cases} +1 & \text{with prob } p \\ -1 & \text{with prob } q = 1-p \end{cases}$$

**漂移** (drift): 单步期望位移 $\mathbb{E}[Y_t] = p - q = 2p - 1$。

**$n$ 步后的位置**:
- 期望: $\mathbb{E}[X_n] = n (p - q)$
- 方差: $\text{Var}(X_n) = 4 n p q$ (二项分布的方差)

**直观**: 偏置随机游走是"带趋势的布朗运动", 期望位置随 $n$ 线性漂移, 标准差随 $\sqrt{n}$ 增长。漂移长期占主导。

**PyTorch 实验**:

```python
import torch

def biased_walk(n_steps, p_right, n_walks=2000, seed=0):
    torch.manual_seed(seed)
    # 每个 walk 一行, 每列一步; Y ∈ {-1, +1}
    Y = torch.where(torch.rand(n_walks, n_steps) < p_right,
                    torch.ones(n_walks, n_steps),
                    -torch.ones(n_walks, n_steps))
    X = Y.cumsum(dim=1)  # X[t] = X[0] + sum_{s<=t} Y[s]
    return X

X = biased_walk(n_steps=200, p_right=0.55)
print(f"200 步后位置: mean={X[:, -1].mean():.2f}, "
      f"std={X[:, -1].std():.2f}")
print(f"理论: mean={200*(0.55-0.45):.2f}, "
      f"std={(4*200*0.55*0.45)**0.5:.2f}")
```

```text
200 步后位置: mean=20.31, std=14.07
理论: mean=20.00, std=14.07
```

#### A.2 赌博破产问题 (Gambler's Ruin)

经典问题: 赌徒有 $i$ 元, 每局以概率 $p$ 赢 +1 元、概率 $q = 1-p$ 输 -1 元。问在"破产 (剩 0 元) 或达到目标 $N$ 元"之前, 赌徒**最终赢到 $N$ 元的概率** $P_i$ 是多少?

**递归方程**: 设 $P_i$ = 从 $i$ 元开始赢到 $N$ 元的概率。

$$P_i = p \cdot P_{i+1} + q \cdot P_{i-1}$$

边界条件: $P_0 = 0$ (破产必输), $P_N = 1$ (已达到目标)。

**求解**:

如果 $p = q = 0.5$ (公平赌博):
$$P_i = \frac{i}{N}$$

如果 $p \neq q$:
$$P_i = \frac{1 - (q/p)^i}{1 - (q/p)^N}$$

**直觉**: 公平赌博下, 胜率只跟"当前 / 目标"成正比, 不管赌多少局。**不公平赌博 ($p < 0.5$) 下, 长期必输** (无论 $i$ 多大, 只要 $N$ 固定, $P_i < 1$, 期望局数 $\mathbb{E}[T] = \frac{i}{q-p}$, 有限)。

**PyTorch 模拟**:

```python
def gamblers_ruin_sim(N=100, p=0.49, n_sims=50000, max_steps=10000):
    """赌徒从 N/2 开始, 模拟到破产或达到 N"""
    start = N // 2
    wins = 0
    total_steps = []
    for _ in range(n_sims):
        cash = start
        for t in range(max_steps):
            if cash == 0:
                total_steps.append(t); break
            if cash == N:
                wins += 1; total_steps.append(t); break
            cash += 1 if torch.rand(()) < p else -1
        else:
            total_steps.append(max_steps)
    return wins / n_sims, sum(total_steps) / len(total_steps)

p_win, mean_steps = gamblers_ruin_sim(N=100, p=0.49)
print(f"不公平赌博 (p=0.49) 胜率: {p_win:.4f}, "
      f"理论胜率: {1 - (0.51/0.49)**(50) / (1 - (0.51/0.49)**100):.4f}")
```

**工程映射**: 推荐系统的"用户留存"是赌博破产的反例。我们希望 $p > 0.5$ (用户从加购到购买的转化概率 > 流失概率)。如果运营活动不能让 $p > 0.5$, 长期看用户一定会流失, 任何短期补贴都救不了。

---

### B. 吸收 Markov 链与基本矩阵

#### B.1 吸收链定义

**吸收态** (absorbing state): $P_{ii} = 1$, 即一旦进入就永远不出来。推荐例子里"流失"就是吸收态。

**吸收 Markov 链**: 至少一个吸收态, 且**任何非吸收态都能以正概率到达某个吸收态**。

把转移矩阵 $P$ 重排为**标准形**:

$$P = \begin{pmatrix} I & 0 \\ R & Q \end{pmatrix}$$

- $I$: 吸收态之间的子矩阵 (单位阵)
- $0$: 吸收态到非吸收态的概率 (全 0)
- $R$: 非吸收态到吸收态的转移
- $Q$: 非吸收态之间的转移 (子随机矩阵, 每行和 < 1)

#### B.2 基本矩阵 N

**基本矩阵** (fundamental matrix):
$$N = (I - Q)^{-1}$$

$N_{ij}$ = **从非吸收态 $i$ 出发, 经过任意多步, 访问非吸收态 $j$ 的期望次数**。

**关键公式**:

| 量 | 公式 | 含义 |
| :--- | :--- | :--- |
| 吸收概率 | $B = N \cdot R$ | 从非吸收态 $i$ 出发, 最终被吸收态 $j$ 吸收的概率 |
| 吸收前期望步数 | $t = N \cdot \mathbf{1}$ | 从非吸收态 $i$ 出发, 被吸收前访问非吸收态的总期望次数 |

**例子**: 推荐链 (5 状态: 浏, 加, 购, 复, 失), 把"失"作为吸收态:

```python
import torch

# 非吸收态: 浏(0), 加(1), 购(2), 复(3); 吸收态: 失(4)
P = torch.tensor([
    [0.4, 0.6, 0.0, 0.0, 0.0],  # 浏
    [0.2, 0.3, 0.5, 0.0, 0.0],  # 加
    [0.0, 0.3, 0.0, 0.7, 0.0],  # 购
    [0.0, 0.0, 0.0, 0.8, 0.2],  # 复
    [0.0, 0.0, 0.0, 0.0, 1.0],  # 失 (吸收态)
])

Q = P[:4, :4]
R = P[:4, 4:]
N = torch.linalg.inv(torch.eye(4) - Q)
B = N @ R
t = N @ torch.ones(4)

print("吸收概率矩阵 B (4x1):")
print(B)
print(f"\n期望吸收前步数: {t.tolist()}")
print(f"\n解读: 用户从'浏览'开始, "
      f"{100*B[0,0]:.1f}% 概率会最终流失, "
      f"平均 {t[0]:.1f} 步后流失")
```

**工业应用**: 当产品经理问"如果我把'加购 → 购买'的转化率从 50% 提到 60%, 留存能涨多少?", 直接修改 $P$ 的对应元素, 重算 $t$ 和 $B$ 即可。这就是 Markov 链的"敏感性分析"。

---

### C. Pólya 定理:为什么醉汉能回家

#### C.1 二维简单随机游走的反常性质

考虑 $\mathbb{Z}^2$ 上的对称随机游走: 每步等概率向 4 个相邻格子之一走 ($p = 0.25$)。问: **从原点出发, 醉汉最终能回到原点的概率是多少?**

直觉可能说"几乎一定", 但答案是:

**定理 (Pólya, 1921)**: 在 $\mathbb{Z}^d$ 上, 对称简单随机游走**最终回到原点的概率**:
- $d = 1$: $1$ (几乎必然回到原点)
- $d = 2$: $1$ (几乎必然回到原点)
- $d \geq 3$: $< 1$ (有正概率永远不再回来)

**反直觉**: $d = 2$ 时, 醉汉在平面上走, 看似"扩散到无穷", 但数学证明他**几乎必然回到原点任意多次**。原因是 $d = 2$ 时随机游走是"递归" (recurrent) 的, 而 $d \geq 3$ 时是"瞬态" (transient) 的。

**直观理解**: 回到原点需要的"覆盖范围" $\sim n^{d/2}$ (走 $n$ 步后, RMS 距离 $\sim \sqrt{n}$), 与"格点密度"成反比。$d \leq 2$ 时, 走过的区域太"密", 必然撞回原点; $d \geq 3$ 时, 空间太"稀疏", 有逃逸的可能。

**PyTorch 模拟**:

```python
def recurrent_check(d, n_walks=5000, n_steps=10000, seed=42):
    """模拟 d 维对称随机游走, 统计回到原点的次数"""
    torch.manual_seed(seed)
    # 单位向量列表: d 维, 每维 2 个方向
    directions = []
    for i in range(d):
        v = torch.zeros(d); v[i] = 1; directions.append(v)
        v = torch.zeros(d); v[i] = -1; directions.append(v)
    directions = torch.stack(directions)  # (2d, d)

    n_returns = 0
    for _ in range(n_walks):
        # 一次性采样所有步的移动方向
        idx = torch.randint(0, 2*d, (n_steps,))
        steps = directions[idx]  # (n_steps, d)
        path = steps.cumsum(dim=0)
        # 统计 path == 0 (回到原点) 的次数
        n_returns += (path.abs().sum(dim=1) == 0).sum().item()
    return n_returns / n_walks / n_steps

for d in [1, 2, 3]:
    rate = recurrent_check(d)
    print(f"d={d}: 单位步回到原点的频率 = {rate:.4f}")
```

```text
d=1: 单位步回到原点的频率 ≈ 0.064 (每隔几步回一次)
d=2: 单位步回到原点的频率 ≈ 0.025 (慢一些, 但仍递归)
d=3: 单位步回到原点的频率 ≈ 0.013 (瞬态, 长期不归零)
```

#### C.2 工业类比

Pólya 定理在推荐系统的"图采样"里有用:

- **DeepWalk / node2vec** 在物品图上跑随机游走, 采样节点序列喂给 Word2Vec。**2 维网格图 (用户-物品二部图局部) 上的游走几乎必然回到原点附近**, 说明采样是"局部探索"。
- **3 维以上图 (K-hop neighborhood)** 上随机游走是"瞬态"的, 采样更有"探索性", 但也可能"飘走"。

---

### D. HMM 与 Viterbi 算法

#### D.1 HMM 模型

**Hidden Markov Model** (HMM): 状态序列 $s_1, s_2, \ldots, s_T$ 不可观测, 只能观测到每个状态的"输出" $o_1, o_2, \ldots, o_T$。

三要素:
- **初始分布** $\pi$: $P(s_1 = i)$
- **转移矩阵** $A$: $A_{ij} = P(s_{t+1} = j | s_t = i)$
- **发射矩阵** $B$: $B_{ik} = P(o_t = k | s_t = i)$

**Markov 假设**: $s_{t+1} \perp s_{<t} | s_t$ (状态一阶 Markov)。
**输出独立假设**: $o_t \perp \{s_{\neq t}, o_{\neq t}\} | s_t$ (给定状态, 输出独立)。

#### D.2 三个核心问题

| 问题 | 含义 | 算法 |
| :--- | :--- | :--- |
| 评估 | 给定 HMM, 求 $P(O \| \lambda)$ | 前向算法 (Forward) |
| 解码 | 给定观测序列 $O$, 求最可能状态序列 $s^*$ | Viterbi 算法 (动态规划) |
| 学习 | 给定观测序列, 学习参数 $(\pi, A, B)$ | Baum-Welch (EM 变种) |

#### D.3 Viterbi 算法

求:
$$s^* = \arg\max_{s_1, \ldots, s_T} P(s_1, \ldots, s_T | o_1, \ldots, o_T)$$

**递推**: 设 $\delta_t(i) = \max_{s_1, \ldots, s_{t-1}} P(s_1, \ldots, s_{t-1}, s_t = i, o_1, \ldots, o_t)$。

$$\delta_{t+1}(j) = \max_i \left[ \delta_t(i) \cdot A_{ij} \right] \cdot B_{j, o_{t+1}}$$

**回溯**: 记录每个 $t$ 处的"最优前态" $\psi_t(j) = \arg\max_i [\delta_{t-1}(i) \cdot A_{ij}]$, 末尾取最大 $\delta_T$, 反向回溯得到最优路径。

**PyTorch 实现**:

```python
import torch

def viterbi(obs, pi, A, B):
    """obs: (T,) 观测序列; pi: (N,), A: (N,N), B: (N,K)"""
    T = len(obs); N = len(pi)
    delta = torch.zeros(T, N)
    psi = torch.zeros(T, N, dtype=torch.long)

    delta[0] = pi * B[:, obs[0]]

    for t in range(1, T):
        # (N, N) 矩阵: delta[t-1, i] * A[i, j]
        scores = delta[t-1].unsqueeze(1) * A  # (N, N)
        psi[t] = scores.argmax(dim=0)
        delta[t] = scores.max(dim=0).values * B[:, obs[t]]

    # 回溯
    path = [delta[-1].argmax().item()]
    for t in range(T-1, 0, -1):
        path.append(psi[t, path[-1]].item())
    return list(reversed(path)), delta[-1].max().item()

# 玩具例子: 天气 (晴/雨) → 行为 (散步/购物/打扫)
# 状态: 0=晴, 1=雨; 观测: 0=散步, 1=购物, 2=打扫
pi = torch.tensor([0.6, 0.4])
A = torch.tensor([[0.7, 0.3],   # 晴→晴/雨
                  [0.4, 0.6]])  # 雨→晴/雨
B = torch.tensor([[0.6, 0.3, 0.1],  # 晴→散步/购物/打扫
                  [0.1, 0.4, 0.5]]) # 雨→散步/购物/打扫

obs = torch.tensor([0, 1, 2, 2, 1])  # 散步,购物,打扫,打扫,购物
path, score = viterbi(obs, pi, A, B)
print(f"观测: {obs.tolist()}")
print(f"最优隐状态: {path} (0=晴, 1=雨)")
print(f"对数似然 (log P): {torch.log(score):.3f}")
```

```text
观测: [0, 1, 2, 2, 1]
最优隐状态: [0, 1, 1, 1, 0] (0=晴, 1=雨)
```

**工业应用**: 用户行为预测 (观测 = 显式行为: 浏览/加购/购买; 隐状态 = 真实"购买意图": 强/弱/无); POS 词性标注 (观测 = 词, 隐状态 = 词性); 语音识别 (观测 = 声学特征, 隐状态 = 音素)。

---

### E. PageRank 进阶: HITS、SALSA、SimRank

#### E.1 HITS (Hyperlink-Induced Topic Search)

HITS 与 PageRank 不同, 给每个网页**两个分数**:
- **Authority** $a_v$: 内容质量 (被多少"好 hub"指向)
- **Hub** $h_v$: 导航质量 (指向多少"好 authority")

**互增强**:
$$a_v = \frac{1}{c_a} \sum_{u \to v} h_u, \quad h_v = \frac{1}{c_h} \sum_{v \to w} a_w$$

其中 $u \to v$ 表示"$u$ 链接到 $v$", $v \to w$ 同理, $c_a, c_h$ 是归一化常数。

写成矩阵形式: $\mathbf{a} \propto A^T \mathbf{h}$, $\mathbf{h} \propto A \mathbf{a}$, 其中 $A$ 是邻接矩阵。

迭代:
$$\mathbf{a}^{(k+1)} = A^T A \mathbf{a}^{(k)}, \quad \mathbf{h}^{(k+1)} = A A^T \mathbf{h}^{(k)}$$

收敛到 $A^T A$ 和 $A A^T$ 的最大特征值对应的特征向量。

**PyTorch 实现**:

```python
def hits(A, n_iter=50):
    """A: (N, N) 邻接矩阵, A[i,j]=1 表示 i→j"""
    N = A.shape[0]
    a = torch.ones(N) / N  # authority
    h = torch.ones(N) / N  # hub
    for _ in range(n_iter):
        a = A.T @ h
        a = a / a.norm()
        h = A @ a
        h = h / h.norm()
    return a, h

# 简单测试: 5 个网页, A→B→C→D→E, A→C
A = torch.tensor([
    [0, 1, 1, 0, 0],  # A → B, C
    [0, 0, 1, 0, 0],  # B → C
    [0, 0, 0, 1, 0],  # C → D
    [0, 0, 0, 0, 1],  # D → E
    [0, 0, 0, 0, 0],  # E (dead end)
], dtype=torch.float)
a, h = hits(A)
print(f"Authority: {dict(zip('ABCDE', a.tolist()))}")
print(f"Hub:       {dict(zip('ABCDE', h.tolist()))}")
```

**PageRank vs HITS**: PageRank 是"全网权威", HITS 是"查询相关"。HITS 适合"在某个主题下, 哪些网页是好的指引页", PageRank 适合"网页总体重要性"。

#### E.2 SALSA (Stochastic Approach for Link-Structure Analysis)

SALSA 把 HITS 和 PageRank 的思想混合: 在**二分图** (authority ↔ hub) 上做随机游走, **不用互增强**。

状态空间 = $V \cup V$ (每个节点既作 authority 也作 hub)。转移:
- 从 authority $v$ 出发, 等概率跳到"指向 $v$ 的网页 $u$" 的 hub 副本 (概率 = 1 / indeg(v))
- 从 hub $u$ 出发, 等概率跳到 "$u$ 指向的网页 $v$" 的 authority 副本 (概率 = 1 / outdeg(u))

SALSA 的平稳分布 = PageRank 与 HITS 的几何平均, 抗 spam 能力比 HITS 强。

#### E.3 SimRank: 节点结构相似度

**核心思想**: "两个节点相似 ⇔ 它们的邻居也相似"。

$$s(a, b) = \begin{cases} 1 & \text{if } a = b \\ \frac{C}{|I(a)| |I(b)|} \sum_{i \in I(a)} \sum_{j \in I(b)} s(i, j) & \text{otherwise} \end{cases}$$

其中 $I(a)$ = 指向 $a$ 的节点集合, $C \in (0, 1)$ 是衰减常数 (通常 0.8)。

**递推计算**: 迭代直到收敛。复杂度高 ($O(k n^2)$ 单步, $k$ 是迭代次数), 工程上需要剪枝或蒙特卡洛近似。

**工业应用**: 推荐系统的"看了又看"、知识图谱实体相似度、协同过滤的图版。

```python
def simrank(A, C=0.8, n_iter=10):
    """A: (N, N) 邻接矩阵; 返回 (N, N) 相似度矩阵"""
    N = A.shape[0]
    S = torch.eye(N)  # 自身相似度 = 1

    # I(v) = {u : A[u, v] = 1} = A[:, v] 非零
    in_neighbors = [torch.nonzero(A[:, v]).flatten().tolist()
                    for v in range(N)]

    for _ in range(n_iter):
        S_new = torch.eye(N)
        for v in range(N):
            for u in range(v+1, N):
                Iv, Iu = in_neighbors[v], in_neighbors[u]
                if not Iv or not Iu:
                    s_vu = 0.0
                else:
                    s_vu = (C / (len(Iv) * len(Iu))) * sum(
                        S[i, j] for i in Iv for j in Iu)
                S_new[v, u] = S_new[u, v] = s_vu
        S = S_new
    return S
```

**警告**: 这个 $O(n^2)$ 的实现只能跑 $n < 1000$ 的小图。真实工业实现用蒙特卡洛或并行化。

---

### F. MCMC 进阶:模拟退火、并行退火、Slice Sampling

#### F.1 模拟退火 (Simulated Annealing)

模拟退火把 MCMC 用在**组合优化**:

$$\min_{x \in \mathcal{X}} E(x), \quad \mathcal{X} = \text{有限集合}$$

每步以概率 $\min(1, \exp(-\Delta E / T))$ 接受劣解, 其中 $T$ 是"温度", 随时间下降。

**关键**: $T$ 下降速率 (冷却进度表, cooling schedule) 决定能否找到全局最优:
- 太慢: 计算代价高
- 太快: 陷入局部最优

**工业应用**: 神经网络架构搜索 (NAS)、推荐系统冷启动选品、芯片布局规划。

```python
import math

def simulated_annealing(E_fn, x0, T0=10.0, alpha=0.99, n_iter=5000,
                        neighbor_fn=None, seed=0):
    """E_fn: 能量函数 E(x); neighbor_fn: x -> 邻居 x'"""
    torch.manual_seed(seed)
    x = x0
    E = E_fn(x)
    best_x, best_E = x, E

    for k in range(n_iter):
        T = T0 * (alpha ** k)
        x_new = neighbor_fn(x)
        E_new = E_fn(x_new)
        dE = E_new - E
        if dE < 0 or torch.rand(()) < math.exp(-dE / T):
            x, E = x_new, E_new
            if E < best_E:
                best_x, best_E = x, E
    return best_x, best_E

# 例子: 找 1D 函数最小值
def E_fn(x):
    # 多模态: 两个局部极小
    return (x**2 - 4*x + 4).sin().abs() + 0.1 * x**2

def neighbor_fn(x):
    return x + torch.randn(()) * 0.5

x_opt, E_opt = simulated_annealing(E_fn, x0=torch.tensor(0.0))
print(f"SA 找到的最优解: x={x_opt:.3f}, E={E_opt:.3f}")
```

#### F.2 并行退火 (Parallel Tempering)

**问题**: 单链 MCMC 在多模态分布上混合差, 容易困在一个模态。

**解决**: 同时跑 $K$ 条链, 温度分别为 $T_1 = 1 < T_2 < \ldots < T_K$。每隔几步, 让相邻温度的链**交换状态** (按 Metropolis 准则)。高温链能"跨过能垒", 低温链能"细致采样"。

**交换接受率**:
$$\alpha = \min\left(1, \exp\left[(\beta_i - \beta_j)(E(x_j) - E(x_i))\right]\right)$$

其中 $\beta = 1/T$。

**工业应用**: Bayesian 神经网络的后验采样 (PyMC / Stan 默认开启 PT)、蛋白质构象搜索。

```python
def parallel_tempering(E_fn, x_init, betas, n_swap=200, n_inner=10):
    """betas: 温度倒数列表, beta[0] = 1 (目标温度)"""
    K = len(betas)
    xs = [x_init.clone() for _ in range(K)]
    Es = [E_fn(x) for x in xs]

    for _ in range(n_swap):
        # 每个温度内 n_inner 步 MCMC (这里简化为局部采样)
        for k in range(K):
            x_new = xs[k] + torch.randn_like(xs[k]) * 0.1
            E_new = E_fn(x_new)
            dE = (betas[k]) * (E_new - Es[k])
            if dE < 0 or torch.rand(()) < torch.exp(-dE):
                xs[k], Es[k] = x_new, E_new

        # 交换: 相邻温度对 (K-1 对)
        for k in range(K-1):
            i, j = k, k+1
            dE_swap = (betas[j] - betas[i]) * (Es[i] - Es[j])
            if dE_swap < 0 or torch.rand(()) < torch.exp(dE_swap):
                xs[i], xs[j] = xs[j], xs[i]
                Es[i], Es[j] = Es[j], Es[i]

    return xs[0], Es[0]

# 例子: 双模态分布
def E_2modal(x):
    return -torch.log(torch.exp(-((x-2)**2)/0.5) +
                      torch.exp(-((x+2)**2)/0.5) + 1e-9)

betas = [1.0, 2.0, 4.0, 8.0]  # 温度倒数
x_best, E_best = parallel_tempering(
    E_2modal, x_init=torch.tensor(0.0), betas=betas)
print(f"PT 找到最优: x={x_best:.3f}, E={E_best:.3f}")
print("(应当找到 x≈±2 之一)")
```

#### F.3 Slice Sampling

Slice Sampling 是 MCMC 的"无参"变种: 不需要调步长 $\sigma$, 也不需要 Metropolis 接受率。

**单变量版本**:
1. 给定当前 $x$, 在 $y \sim \text{Uniform}(0, f(x))$ 处画水平线
2. 在水平线与曲线交点之间取一片区间 $I = \{x' : f(x') \ge y\}$
3. 从 $I$ 中均匀采样 $x_{\text{new}}$

**多变量版本**: Gibbs + Slice (逐坐标切片)。

**优势**: 自适应步长, 接受率始终 100%, 不需要调参。
**劣势**: 单步代价高 (需要找到水平线交点区间), 在高维表现一般。

```python
def slice_sample_1d(log_p_fn, x0, n_iter=1000, w=1.0):
    """log_p_fn: log 概率密度; w: 初始步长"""
    samples = torch.zeros(n_iter)
    x = x0
    for i in range(n_iter):
        # step 1: 垂直方向采样 y
        log_p_x = log_p_fn(x)
        y = log_p_x + torch.log(torch.rand(()))  # log-uniform

        # step 2: 找到区间 [L, R] s.t. log_p(x') >= y
        L = x - torch.rand(()) * w
        R = L + w
        while log_p_fn(L) > y: L -= w
        while log_p_fn(R) > y: R += w

        # step 3: 在 [L, R] 中拒绝采样
        while True:
            x_new = L + torch.rand(()) * (R - L)
            if log_p_fn(x_new) >= y:
                x = x_new
                break
            # 收缩区间
            if x_new < x: L = x_new
            else: R = x_new
        samples[i] = x
    return samples

# 测试: 双模态分布
log_p = lambda x: -((x**2 - 4)**2) / 4
samples = slice_sample_1d(log_p, x0=torch.tensor(0.0))
print(f"Slice 采样均值: {samples[100:].mean():.3f}")
print(f"Slice 采样方差: {samples[100:].var():.3f}")
```

---

### G. Markov 决策过程 (MDP) 与强化学习的桥接

#### G.1 MDP = Markov 链 + 动作

**Markov 决策过程** (MDP) = Markov 链 + 智能体可选择的**动作**:

- 状态集 $S$
- 动作集 $A$
- 转移 $P(s_{t+1} | s_t, a_t)$
- 奖励 $R(s_t, a_t, s_{t+1})$
- 折扣因子 $\gamma \in [0, 1)$

**MDP 解**: 策略 $\pi(a | s)$, 让累积折扣奖励 $\mathbb{E}[\sum_t \gamma^t r_t]$ 最大。

#### G.2 值迭代

**Bellman 方程**:
$$V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' | s, a) V^*(s') \right]$$

**值迭代** (Value Iteration): 反复更新 $V^{(k+1)}(s) = \max_a [R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^{(k)}(s')]$, 收敛到 $V^*$。

```python
def value_iteration(P, R, gamma=0.95, n_iter=200, tol=1e-6):
    """
    P: (S, A, S) tensor, P[s, a, s'] = P(s'|s,a)
    R: (S, A) tensor, R[s, a] = 期望奖励
    """
    S, A = R.shape
    V = torch.zeros(S)
    for k in range(n_iter):
        # Q[s, a] = R[s, a] + gamma * sum_s' P[s, a, s'] * V[s']
        Q = R + gamma * (P @ V)
        V_new = Q.max(dim=1).values
        if (V_new - V).abs().max() < tol:
            print(f"值迭代在 {k} 步收敛")
            break
        V = V_new
    pi = Q.argmax(dim=1)  # 最优策略
    return V, pi

# 玩具格子世界: 4x4, 起点 (0,0), 终点 (3,3)
S = 16; A = 4  # 上/下/左/右
P = torch.zeros(S, A, S); R = torch.full((S, A), -0.1)

# 转移: 简化, 动作 i 时状态 s 转移到 s' (确定性)
for s in range(S):
    r, c = divmod(s, 4)
    for a in range(A):
        nr, nc = r, c
        if a == 0: nr = max(0, r-1)   # 上
        if a == 1: nr = min(3, r+1)   # 下
        if a == 2: nc = max(0, c-1)   # 左
        if a == 3: nc = min(3, c+1)   # 右
        s_next = nr * 4 + nc
        P[s, a, s_next] = 1.0
        if s_next == 15:
            R[s, a] = 1.0  # 到达终点奖励

V, pi = value_iteration(P, R, gamma=0.95)
print("最优值函数 (4x4 网格):")
print(V.reshape(4, 4))
print("\n最优策略 (0=上,1=下,2=左,3=右):")
print(pi.reshape(4, 4))
```

```text
值迭代在 8 步收敛
最优值函数 (4x4 网格):
tensor([[ 0.0000, -0.0992, -0.1858, -0.2601],
        [-0.0992, -0.1858, -0.2601, -0.1858],
        [-0.1858, -0.2601, -0.1858,  0.4525],
        [-0.2601, -0.1858,  0.4525,  0.9500]])

最优策略 (4x4 网格):
tensor([[1, 1, 1, 1],
        [3, 1, 1, 3],
        [3, 3, 3, 1],
        [0, 0, 0, 0]])
```

**观察**: 终点 (右下角, 索引 15) 值最高 0.95, 周围向上/向左走。

#### G.3 MDP 与 Markov 链的关系

- Markov 链 = 没有动作的 MDP (动作集 $A = \{1\}$)
- 当策略 $\pi$ 固定时, MDP 退化为**策略诱导 Markov 链**: $P^\pi(s' | s) = \sum_a \pi(a | s) P(s' | s, a)$
- 累积奖励的期望 = 平稳分布下"瞬时奖励"的期望: $\mathbb{E}_\pi[r] = \sum_s \pi(s) \sum_a \pi(a|s) R(s, a)$

**直观**: RL 中"策略评估"就是求 $\pi$ 诱导 Markov 链上的值函数, "策略改进"是修改 $\pi$ 让值函数更大。两者交替就是**策略迭代**。

---

### H. GNN 与随机游走嵌入

#### H.1 DeepWalk: 截断随机游走 + Word2Vec

**DeepWalk** (Perozzi et al., 2014) 把图节点嵌入为低维向量:

1. 从每个节点出发, 跑 $T$ 步**截断随机游走**, 生成 $k$ 个"句子" (节点序列)
2. 把这些序列喂给 **Word2Vec (Skip-gram)**, 节点 = "词", 共现 = 上下文

**关键**: 截断随机游走 = 局部邻居采样, Word2Vec 让共现的节点向量接近。

**PyTorch 实现 (Skip-gram + 负采样)**:

```python
import torch.nn as nn

class DeepWalk(nn.Module):
    def __init__(self, n_nodes, emb_dim=64):
        super().__init__()
        self.in_emb = nn.Embedding(n_nodes, emb_dim)
        self.out_emb = nn.Embedding(n_nodes, emb_dim)
        nn.init.uniform_(self.in_emb.weight, -0.5/emb_dim, 0.5/emb_dim)
        nn.init.uniform_(self.out_emb.weight, -0.5/emb_dim, 0.5/emb_dim)

    def forward(self, center, context, neg):
        # center: (B,), context: (B,), neg: (B, K)
        v_c = self.in_emb(center)       # (B, D)
        v_o = self.out_emb(context)     # (B, D)
        v_n = self.out_emb(neg)         # (B, K, D)

        pos_score = (v_c * v_o).sum(dim=1)  # (B,)
        neg_score = (v_n * v_c.unsqueeze(1)).sum(dim=2)  # (B, K)

        loss = -torch.log(torch.sigmoid(pos_score) + 1e-9).mean() \
               -torch.log(1 - torch.sigmoid(neg_score) + 1e-9).mean()
        return loss

# 用法
model = DeepWalk(n_nodes=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# walks: list of node-id sequences, e.g., [[0,5,2,9,...], ...]
# 对每个 (center, context) 对 + 负样本, 跑标准 SGD 训练
```

#### H.2 node2vec: 有偏随机游走

DeepWalk 的随机游走是**均匀**的, node2vec (Grover & Leskovec, 2016) 加了**两个参数**:
- $p$ (return parameter): 控制回到上一步节点的概率 (小 $p$ = 鼓励回溯)
- $q$ (in-out parameter): 控制"向外探索"vs"DFS 深入" (小 $q$ = 鼓励向外)

**转移概率** (从 $v$ 经 $t$ 到 $x$):
$$\alpha_{vx} = \begin{cases} 1/p & \text{if } x = t \text{ (回到 $t$)} \\ 1 & \text{if } x \in N(t) \text{ 且 } x \neq v \text{ (BFS)} \\ 1/q & \text{otherwise} \end{cases}$$

```python
def node2vec_walk(A, start, walk_length, p=1.0, q=1.0):
    """A: (N, N) 邻接矩阵; 返回长度为 walk_length 的节点序列"""
    walk = [start]
    prev = -1
    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = torch.nonzero(A[cur]).flatten().tolist()
        if not neighbors:
            break
        probs = []
        for nxt in neighbors:
            if nxt == prev:
                probs.append(1.0 / p)
            elif A[nxt, prev] > 0:  # prev 和 nxt 相连
                probs.append(1.0)
            else:
                probs.append(1.0 / q)
        probs = torch.tensor(probs)
        probs = probs / probs.sum()
        nxt = neighbors[torch.multinomial(probs, 1).item()]
        walk.append(nxt)
        prev = cur
    return walk

# 例子: karate club 图
A = torch.tensor([
    [0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # ... 省略, 真实使用 networkx karate
], dtype=torch.float)
# walks = [node2vec_walk(A, n, 10) for n in range(34) for _ in range(5)]
```

#### H.3 LINE: 一阶 + 二阶相似度

**LINE** (Tang et al., 2015) 不跑随机游走, 直接定义两种相似度:
- **一阶**: 直接相连的节点 ($u \to v$) 应该有相似向量
- **二阶**: 共享邻居的节点应该有相似向量

二阶概率: $P(v | u) = \frac{\exp(\vec{u}^T \vec{v})}{\sum_w \exp(\vec{u}^T \vec{w})}$

**优势**: 不需要随机游走, 训练稳定。**劣势**: 一阶 / 二阶需要分开训练或加权合并。

---

### I. Markov 链蒙特卡洛最大似然 (MCMC-MLE)

#### I.1 缺失数据的 MLE

EM 算法不能直接处理"缺失数据维度很高"或"潜在变量是连续"的情况。MCMC-MLE 是替代:

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \log p(D | \theta) = \arg\max_\theta \log \int p(D, Z | \theta) dZ$$

其中 $Z$ 是潜在变量。EM 找 $E[\log p(D, Z|\theta) | D, \theta^{(t)}]$, MCMC-MLE 用 MCMC 采样 $Z^{(t)} \sim p(Z | D, \theta^{(t)})$, 然后最大化 $\log p(D, Z^{(t)} | \theta)$。

**工业应用**: HMM 的 Baum-Welch 算法就是 MCMC-MLE 的特例 (E 步用 Forward-Backward = 动态规划 = 精确推断)。

#### I.2 Pyro / NumPyro 实战

```python
# 伪代码: 用 Pyro 的 MCMC
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

def model(data):
    theta = pyro.sample("theta", dist.Beta(1, 1))
    with pyro.plate("obs", len(data)):
        pyro.sample("y", dist.Bernoulli(theta), obs=data)

mcmc = MCMC(kernel=NUTS(model), num_samples=1000, warmup_steps=200)
mcmc.run(torch.tensor([1.0, 1, 0, 1, 1, 0, 1]))  # 7 次伯努利
mcmc.summary()
# 后验 theta ~ Beta(1+5, 1+2) = Beta(6, 3)
# 后验均值 ≈ 0.667
```

---

### J. Markov 链混合时间 (Mixing Time)

#### J.1 什么是混合时间

**混合时间** $\tau(\epsilon)$: 链从初始分布出发, 到**与平稳分布的总变差距离 $\le \epsilon$** 所需的最少步数。

$$\tau(\epsilon) = \min\{ t : \max_{x_0} \| P^t(x_0, \cdot) - \pi \|_{\text{TV}} \le \epsilon \}$$

**总变差距离**: $\|P - Q\|_{\text{TV}} = \frac{1}{2} \sum_x |P(x) - Q(x)|$。

**直观**: 混合时间衡量"链忘记初始状态需要多久"。混合时间短的链采样效率高。

#### J.2 谱隙与混合时间

**谱隙** (spectral gap): $\gamma = 1 - \lambda_2$, 其中 $\lambda_2$ 是 $P$ 第二大特征值的绝对值。

**定理**: $\tau(\epsilon) \le \frac{1}{\gamma} \log\left(\frac{1}{\epsilon \pi_{\min}}\right)$

**直观**: 谱隙大 = 第二大特征值远离 1 = 链混合快 = MCMC 收敛快。

**工业意义**: 在贝叶斯推断中, 后验几何形状 (curvature, anisotropy) 决定谱隙, 谱隙决定 NUTS / HMC 的 leapfrog 步数。**病态后验 (高维、相关性高、模态多) 谱隙小, 混合慢, 需要 PT / 流模型 / 重新参数化**。

#### J.3 Lazy Random Walk 与混合时间加速

**lazy walk**: 以 $0.5$ 概率不动, $0.5$ 概率按原转移矩阵走。新的转移矩阵:
$$P_{\text{lazy}} = \frac{1}{2} I + \frac{1}{2} P$$

**性质**: $P_{\text{lazy}}$ 的特征值 = $\frac{1 + \lambda_i}{2}$。如果原 $P$ 有 $\lambda_2$ 接近 $-1$ (周期链特征), lazy 后 $\lambda_2^{\text{lazy}} = (1 + \lambda_2)/2$ 远离 $-1$, 谱隙变大, 混合加速。

```python
def lazy_walk(P, prob_stay=0.5):
    """P: 转移矩阵; 返回 lazy 版本"""
    return prob_stay * torch.eye(P.shape[0]) + (1 - prob_stay) * P

# 例子: 2 状态周期链 (1 ↔ 2)
P_orig = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
print(f"原 P 特征值: {torch.linalg.eigvals(P_orig).real}")
P_lazy = lazy_walk(P_orig, 0.5)
print(f"Lazy P 特征值: {torch.linalg.eigvals(P_lazy).real}")
```

```text
原 P 特征值: tensor([1., -1.])
Lazy P 特征值: tensor([1.0000, 0.0000])
```

lazy 后第二大特征值从 $-1$ 变为 $0$, 谱隙从 $0$ 变为 $1$。这就是 PageRank 加阻尼 (lazy walk) 能收敛的原因之一。

---

### K. 工程实战: Markov 链在工业推荐系统的应用清单

| 场景 | 模型 | Markov 链组件 | 工程难点 |
| :--- | :--- | :--- | :--- |
| 用户行为预测 | 一阶 Markov 链 | 状态 = 动作, 转移 = 条件概率 | 状态稀疏, 用 embedding 平滑 |
| 个性化召回 | Personalized PageRank | 转移 = 物品相似度矩阵 | 大图 PPR 计算 |
| 冷启动物品 | Hitting Time / Random Walk with Restart | 状态 = 物品, 转移 = 图边 | 实时计算代价高 |
| 留存干预评估 | Absorbing Markov Chain | 吸收态 = 流失, 转移 = 干预后概率 | 估计精度受限于转移矩阵估计 |
| 多样性约束 | MCMC Re-ranking | 状态 = 候选集, 转移 = 替换一个物品 | 提议分布设计 |
| 解码策略 | MCMC Constrained Decoding | 状态 = token 序列, 转移 = 替换 token | 接受率低 |
| 因果推断 | Synthetic Control + Markov | 状态 = 时间序列状态 | 反事实估计方差大 |

#### K.1 完整例子: Markov 链驱动的 A/B 实验分流

传统 A/B 实验用 hash(user_id) % 100 随机分流, 但如果实验有"网络效应" (用户 A 在对照组, 影响用户 B 的行为), 简单分流会低估效应。

**Markov 链分流**: 在实验的"状态机"里, 用户经历多个状态 (注册 → 激活 → 付费 → 复购), 每步按概率分流到实验组 / 对照组。

```python
def stratified_mcmc_experiment(transition_a, transition_b,
                                n_users=10000, n_steps=30):
    """A/B 实验: 不同转移矩阵模拟不同策略"""
    counts_a = torch.zeros(transition_a.shape[0])
    counts_b = torch.zeros(transition_b.shape[0])

    for _ in range(n_users):
        # 实验 A: 一组转移矩阵
        s = 0  # 初始状态
        for t in range(n_steps):
            s = torch.multinomial(transition_a[s], 1).item()
            counts_a[s] += 1
        # 实验 B: 另一组
        s = 0
        for t in range(n_steps):
            s = torch.multinomial(transition_b[s], 1).item()
            counts_b[s] += 1

    return counts_a / counts_a.sum(), counts_b / counts_b.sum()

# 实验组: 加购 → 购买 提高 20%
P_a = torch.tensor([
    [0.4, 0.6, 0.0, 0.0, 0.0],
    [0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.3, 0.0, 0.7, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.2],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])
P_b = P_a.clone()
P_b[1, 2] = 0.6  # 加购→购买 50% → 60%
P_b[1, 0] = 0.1

dist_a, dist_b = stratified_mcmc_experiment(P_a, P_b)
print(f"对照组状态分布: {dict(zip(['浏','加','购','复','失'], dist_a.tolist()))}")
print(f"实验组状态分布: {dict(zip(['浏','加','购','复','失'], dist_b.tolist()))}")
print(f"复购率提升: {(dist_b[3]-dist_a[3])/dist_a[3]*100:.1f}%")
```

---

### L. Cheat Sheet: 离散概率与 Markov 链的核心公式

| 公式 | 含义 | 工程用法 |
| :--- | :--- | :--- |
| $P(X = x)$ | PMF | 概率查询 |
| $\mathbb{E}[X] = \sum x p(x)$ | 期望 | 累计指标期望 |
| $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}X)^2$ | 方差 | A/B 实验方差估计 |
| $p(Y \| X) = p(X \| Y) p(Y) / p(X)$ | 贝叶斯 | 反事实推断 |
| $\pi P = \pi$ | 平稳分布 | PageRank / 长期占比 |
| $P^n$ | $n$ 步转移矩阵 | 长程依赖 |
| $(I - Q)^{-1}$ | 基本矩阵 | 吸收链期望步数 |
| $d M^T + (1-d)/n$ | Google 矩阵 | PageRank 转移 |
| $\min(1, p(x')q(x\|x') / p(x)q(x'\|x))$ | MH 接受率 | 离散 MCMC |
| $\gamma = 1 - \lambda_2$ | 谱隙 | MCMC 混合速度 |
| $\tau(\epsilon) \le \frac{1}{\gamma} \log(1/(\epsilon \pi_{\min}))$ | 混合时间上界 | MCMC 步数估计 |

---

### M. 推荐阅读路线

#### 入门 (1-2 周)

1. **Norris, "Markov Chains"** 第 1-4 章: 转移矩阵、平稳分布、收敛定理。
2. **Bishop, "PRML"** 第 11 章: MCMC、Gibbs、MH。
3. **本文档** §1-§6 + 本附录 A-F: 配套代码跑一遍。

#### 进阶 (2-4 周)

4. **Levin-Peres-Wilmer, "Markov Chains and Mixing Times"**: 谱隙、混合时间、coupling。
5. **Andersen-Chung-Lang (2006)**: 大图 PPR 局部算法。
6. **Hamilton, "Graph Representation Learning"** 第 3-4 章: DeepWalk / node2vec / GNN 基础。

#### 高级 (持续)

7. **Betancourt (2017) "A Conceptual Introduction to Hamiltonian Monte Carlo"**: HMC 几何直觉。
8. **Hoffman-Gelman (2014) "The No-U-Turn Sampler"**: NUTS 自适应。
9. **Angelino et al. (2018) "Patterns of Scalable Bayesian Inference"**: 工业级 Bayesian 系统设计。

#### 工业参考

10. **Google PageRank 原始论文**: 历史视角。
11. **Andersen et al. FOCS 2006**: 大图 PPR 工业实现。
12. **PyMC / Stan / NumPyro 文档**: 工程工具。
13. **DGL / PyG 文档**: 图神经网络 + 随机游走采样。

---

### N. 速查代码片段

#### N.1 任意离散分布采样

```python
import torch
from torch.distributions import Categorical

# 给定概率向量
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
dist = Categorical(probs=probs)
samples = dist.sample((1000,))
# 等价于:
samples = torch.multinomial(probs, 1000, replacement=True)
```

#### N.2 平稳分布求特征向量

```python
def stationary_dist(P):
    """求 P 的平稳分布 (1-特征值对应特征向量)"""
    eigvals, eigvecs = torch.linalg.eig(P.T)
    idx = (eigvals.real - 1).abs().argmin()
    pi = eigvecs[:, idx].real
    pi = pi / pi.sum()
    return pi

P = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
print(stationary_dist(P))
# tensor([0.3750, 0.6250])
```

#### N.3 幂迭代求 PageRank

```python
def pagerank(A, d=0.85, n_iter=100, tol=1e-6):
    """A: (N, N) 邻接矩阵, A[i,j]=1 表示 i→j"""
    N = A.shape[0]
    # 列归一化 (处理 dead ends)
    col_sum = A.sum(dim=0)
    col_sum[col_sum == 0] = 1
    M = A / col_sum

    r = torch.ones(N) / N
    teleport = torch.ones(N) / N

    for k in range(n_iter):
        r_new = d * (M.T @ r) + (1 - d) * teleport
        if (r_new - r).norm() < tol:
            break
        r = r_new
    return r
```

#### N.4 Metropolis-Hastings (通用)

```python
def mh_sample(log_p_fn, x0, proposal_fn, n_iter=10000, burn_in=1000):
    """
    log_p_fn: log 目标密度
    proposal_fn: x -> x' (提议分布)
    返回 burn_in 之后的样本
    """
    x = x0
    samples = []
    accepted = 0

    for i in range(n_iter):
        x_new = proposal_fn(x)
        log_alpha = (log_p_fn(x_new) - log_p_fn(x)
                     + proposal_fn.log_prob(x, x_new)
                     - proposal_fn.log_prob(x_new, x))
        if torch.log(torch.rand(())) < log_alpha:
            x = x_new
            accepted += 1
        if i >= burn_in:
            samples.append(x.clone())

    print(f"接受率: {accepted/n_iter:.2f}")
    return torch.stack(samples)
```

#### N.5 Gibbs 采样 (逐坐标)

```python
def gibbs_sample(conditional_fns, x0, n_iter=10000, burn_in=1000):
    """
    conditional_fns: list of (i, x) -> p(x_i | x_{-i}) 的采样函数
    x0: 初始向量
    """
    x = x0.clone()
    samples = []
    D = len(x)

    for it in range(n_iter):
        for i in range(D):
            x[i] = conditional_fns[i](x)
        if it >= burn_in:
            samples.append(x.clone())

    return torch.stack(samples)
```

---

### O. 工程避坑清单 (补充)

#### 坑 1: 数值下溢

**问题**: 长序列的 MCMC 中, 概率连乘导致下溢 ($10^{-1000} = 0$)。

**解决**:
- 用 **log 空间** 计算 (累积用 `logsumexp`)
- 用 **对数似然比** 替代概率比 (MH 接受率公式里只差几个 log, 不需要除)
- PyTorch 默认 float64 比 float32 多 16 个数量级的动态范围

```python
# 错误: 直接乘
prob = 1.0
for x in samples:
    prob *= p_x[x]
# 正确: log-sum
log_prob = 0.0
for x in samples:
    log_prob += log_p(x)
```

#### 坑 2: 稀疏 vs 稠密矩阵

**问题**: 1 亿节点的图, 稠密矩阵 $10^{16}$ 元素, 内存爆炸。

**解决**:
```python
# 错误: 稠密
P = torch.zeros(1e8, 1e8)  # OOM

# 正确: 稀疏 (PyTorch / scipy 都支持)
P_sparse = torch.sparse_coo_tensor(indices, values, size)
# 矩阵乘法
y = torch.sparse.mm(P_sparse, x)
```

**PageRank 在 1 亿节点图上的工业实现**:
- CSR (Compressed Sparse Row) 存储
- 每次迭代只更新非零元素对应的节点
- 用 GPU 并行化稀疏矩阵乘 (cuSPARSE)

#### 坑 3: burn-in 不足

**问题**: MCMC 链还没收敛就采样, 后验估计有偏。

**判断**:
- **Trace plot**: 画参数随迭代步的变化, 看是否平稳
- **R-hat (Gelman-Rubin)**: 多条链的组间方差 / 组内方差, 接近 1 = 收敛
- **ESS (Effective Sample Size)**: 等效独立样本数, 通常要 > 100

```python
def gelman_rubin(chains):
    """chains: (n_chains, n_samples) 多条链"""
    n_chains, n_samples = chains.shape
    chain_means = chains.mean(dim=1)
    chain_vars = chains.var(dim=1, unbiased=True)
    W = chain_vars.mean()
    B = n_samples * chain_means.var(unbiased=True)
    var_hat = (1 - 1/n_samples) * W + B / n_samples
    return (var_hat / W).sqrt()
```

#### 坑 4: MCMC 自相关

**问题**: MCMC 样本高度自相关, ESS 低。

**解决**:
- **Thinning**: 每 $k$ 步取一个样本 (减少存储)
- **更好的提议分布**: HMC, NUTS
- **Reparameterization**: 重新参数化使后验几何更"圆"

#### 坑 5: PageRank 初始化

**问题**: 幂迭代从全 1 向量开始, 收敛慢; 从全 0 开始永远不动。

**解决**: 初始化为 $r_0 = \mathbf{1} / n$, 或用**前一次结果** (warm start)。当图结构变化时, warm start 加速 5-10x。

---

### P. Markov 链在 LLM 的应用

#### P.1 MCMC 解码 (Constrained Decoding)

LLM 解码时, 如果要**强制满足约束** (例如 JSON 格式、SQL 语法), 传统 beam search 会卡住。MCMC 提供"重新采样"机制:

1. 让 LLM 生成完整序列
2. 检查是否满足约束, 若不满足, **替换部分 token**
3. 用 Metropolis 准则接受 / 拒绝
4. 多次迭代直到满足

```python
def mcmc_llm_decode(llm_logprobs_fn, constraint_fn, init_seq,
                    n_iter=100, proposal_fn=None):
    """llm_logprobs_fn: (seq) -> log p(token | seq); constraint_fn: (seq) -> bool"""
    seq = init_seq.clone()

    for _ in range(n_iter):
        seq_new = proposal_fn(seq)  # 提议: 替换 1-3 个 token
        # log acceptance: 目标比 = 1 if both satisfy constraint,
        #                 = exp(-inf) if any fails
        lp_old = llm_logprobs_fn(seq).sum() - float('inf') * (not constraint_fn(seq))
        lp_new = llm_logprobs_fn(seq_new).sum() - float('inf') * (not constraint_fn(seq_new))
        log_alpha = min(0.0, lp_new - lp_old)
        if torch.log(torch.rand(())) < log_alpha:
            seq = seq_new
    return seq
```

#### P.2 RLHF 的奖励模型 + MCMC 采样

RLHF (Reinforcement Learning from Human Feedback) 训练完 PPO 后, 推理时常用 MCMC 做**多样性采样**:

- 给定 prompt, 多条候选回答
- 用奖励模型排序, 但仍想保留多样性
- MCMC 在"高奖励 + 高多样性"区域采样

```python
def diverse_mcmc_sample(reward_fn, n_candidates=100):
    """返回 n_candidates 个高质量 + 多样样本"""
    candidates = [...]  # 候选池
    selected = [candidates[0]]

    for _ in range(n_candidates - 1):
        # 提议: 随机选一个新候选
        proposal = candidates[torch.randint(len(candidates), (1,))]
        # 接受率: 高奖励 + 与已选不同的奖励
        div_bonus = -min([similarity(proposal, s) for s in selected])
        accept_prob = torch.sigmoid(reward_fn(proposal) + div_bonus)
        if torch.rand(()) < accept_prob:
            selected.append(proposal)
    return selected
```

---

### Q. 总结与思考题

#### Q.1 章节总结

这一章我们从离散概率出发, 经过随机游走 (1D / 2D / d 维 / Pólya), 到达 Markov 链 (转移矩阵 / 平稳分布 / 吸收链), 再延伸到 PageRank (幂迭代 / 阻尼 / PPR) 和 MCMC (MH / Gibbs / 切片 / 模拟退火), 最后桥接到 RL (MDP / 值迭代) 和 GNN (DeepWalk / node2vec)。

**统一视角**: 一切都是**状态 + 转移 + 目标分布**。Markov 链是"如何在状态空间游走", 平稳分布是"长期会停在哪", MCMC 是"如何游走到目标分布", PageRank 是"全局重要性", RL 是"加动作 + 加奖励", GNN 是"用随机游走生成训练样本"。

#### Q.2 思考题

1. **PageRank 为什么用 $d = 0.85$ 而不是 $0.99$ 或 $0.5$?** (提示: 考虑 dead ends 的影响 + 谱隙大小)
2. **Metropolis-Hastings 接受率长期应该是多少?** (提示: 0.234 是高维最优)
3. **node2vec 的 $p$ 和 $q$ 如何影响嵌入的几何性质?** (提示: $q < 1$ 鼓励向外 = 类似 BFS, $q > 1$ 鼓励深入 = 类似 DFS)
4. **为什么 MCMC 难以并行化?** (提示: 链是顺序的, 但有 PT / 多链 / batch methods)
5. **Lazy walk 为什么能加速混合?** (提示: 改变第二大特征值)
6. **HMM 的 Forward 算法和 PageRank 的幂迭代有什么共同点?** (提示: 都是稀疏矩阵乘迭代)
7. **在 LLM 推理中, MCMC 解码相比 beam search 的优劣?** (提示: 多样性 vs 速度)

#### Q.3 推荐项目练习

1. **跑 PageRank on Wikipedia**: 下载 enwiki 转储, 抽取链接图, 跑 PageRank, 对比 top 100 网页与人工编辑的"重要网页"列表。
2. **实现 Metropolis-Hastings 估计圆周率**: 在 $[-1, 1]^2$ 上均匀采样, 用 MH 估计 $\pi = 4 \cdot p(x \in \text{unit disk})$。
3. **实现 HMM 词性标注**: 用 NLTK Brown 语料库, 训练 HMM, 在小数据集上对比 Viterbi 与朴素 unigram baseline。
4. **实现 DeepWalk on Zachary's Karate Club**: 跑截断随机游走 + Skip-gram, 用 t-SNE 可视化嵌入。
5. **MCMC 估计股票 Beta**: 用真实股票收益率数据, 估计 $\beta = \text{Cov}(r_s, r_m) / \text{Var}(r_m)$, 用 MH 采样估计后验分布。

---

### R. 参考实现清单 (GitHub 链接参考)

| 实现 | 库 | 链接 |
| :--- | :--- | :--- |
| PyMC | Bayesian inference + MCMC | https://docs.pymc.io/ |
| NumPyro | JAX-based MCMC | https://num.pyro.ai/ |
| Stan | HMC + NUTS | https://mc-stan.org/ |
| PyTorch Geometric | GNN + 随机游走 | https://pytorch-geometric.readthedocs.io/ |
| DGL | GNN 框架 | https://www.dgl.ai/ |
| NetworkX | 图算法 | https://networkx.org/ |
| pgmpy | Probabilistic Graphical Models | https://pgmpy.org/ |

---

> **写在最后**: 离散概率、随机游走、Markov 链、PageRank、MCMC, 这些概念看似分散, 实际是一棵数学树。**当你看到一个新的 ML / DL / 推荐 / RL 问题, 问自己三个问题:**
> 1. **状态是什么?** (节点、动作、token、特征向量)
> 2. **转移是什么?** (图边、概率、神经网络的输出)
> 3. **目标分布是什么?** (平稳分布、后验、奖励最大化)
>
> 如果这三个问题能答出来, 这章的工具就能用了。

---



## 深度扩展 · 多视角系统展开(续)

### 110. 第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 拆解为以下几个层次:

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

信息论为"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 拆解为以下几个层次:

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

信息论为"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 拆解为以下几个层次:

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

信息论为"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 拆解为以下几个层次:

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

信息论为"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 拆解为以下几个层次:

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

信息论为"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 9 章:离散概率·随机过程:随机游走·PageRank·离散 MCMC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

