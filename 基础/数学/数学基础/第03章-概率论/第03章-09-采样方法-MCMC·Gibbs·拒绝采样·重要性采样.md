---
title: "第03章-09-采样方法-MCMC·Gibbs·拒绝采样·重要性采样"
tags:
  - 数学
  - 采样方法
  - MCMC
  - Gibbs
  - 拒绝采样
  - 重要性采样
description: "本章要点：本\"这件事,从一句\"假装从 p(x) 里 sample\"的口头说法,变成五种可操作、可实现、可对比的方法: 直接采样(逆变换)、拒绝采样、重要性采样、M…"
chapter_id: 数学-9-03
last_updated: 2026-07-14
---

# 第03章-09-采样方法-MCMC·Gibbs·拒绝采样·重要性采样

本**"这件事,从一句"假装从 $p(x)$ 里 sample"的口头说法,变成五种**可操作、可实现、可对比**的方法: **直接采样(逆变换)、拒绝采样、重要性采样、MCMC、Gibbs、HMC/NUTS**。它们不是五种互不相干的小技巧,而是同一根思想树(蒙特卡洛方法)的五个分支。深度学习里的**Policy Gradient 梯度、Off-Policy 评估、VAE 重参数化、Diffusion 反向过程、Bayesian Neural Network 后验估计**几乎所有"用采样绕开难积分"的地方,都站着其中一种或多种方法。
>
> **学完你能回答的问题**:
> - 为什么 `torch.distributions.Beta` 不是直接"按密度函数抽样",而是要拒绝采样?
> - 为什么 Policy Gradient 公式 $\nabla J = \mathbb{E}[\nabla \log \pi \cdot R]$ 是个**采样估计**?
> - 为什么 Off-Policy 评估里"用一个采样分布的样本去估计另一个分布的期望"会出现**重要性权重**?
> - 为什么 PPO 用 Clip 替代了重要性采样?Clip 在数学上做了什么?
> - 为什么 LLM 的解码可以用 MCMC 思想(束搜索 vs 随机采样 vs MCMC)?
> - 为什么 PyMC / Stan 默认用 HMC/NUTS 而不是 MH?HMC 比 MH 强在哪里?
>
> **本章与后续章节的关系**:
> - 第 6 章 MLE/MAP/贝叶斯估计 → **贝叶斯后验是难积分**,本章是"如何采样绕过它"
> - 第 7 章 共轭先验 → 一些共轭情况下后验是闭式的,不需要采样;**非共轭就必须采样**
> - 第 10 章 VAE/Diffusion/Bayesian NN → 全部依赖本章采样技术
> - 第 11 章 KL 散度 → KL 的"估计"本身就需要采样
> - 1-高数 第 9 章 变分法 → **变分推断是采样的"优化版"替代方案**

---

## 1. 直觉引入

### 1.1 工程师的痛点:为什么 "采样" 在 ML 里无处不在?

假设你刚做完一次 AB 实验,数据里**对照组的点击率是 0.020,实验组是 0.022**。leader 问:"这 0.002 的差异,真的不是噪声吗?"

你打开 Python 写假设检验(详见第 8 章):

```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(control, treatment)
```

但其实在更深的层面,你隐含地做了这样一件事:**从"零假设下样本均值差的分布"里,采样出成千上万个可能的 $\Delta \mu$,看实际观察到的 0.002 在分布的什么位置**。这就是**蒙特卡洛(Monte Carlo)**:用采样代替解析积分。

再换一个场景。你在调 LLM 的解码策略,代码大概长这样:

```python
logits = model(input_ids)  # (B, V)
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)  # <-- 这行在干什么?
```

`torch.multinomial` 就是"从 Categorical 分布里采样一个 token"。在 RLHF 的 PPO 阶段,你可能这样算策略梯度:

```python
# Policy Gradient: ∇J = E[∇log π(a|s) · R(s,a)]
# 实际写代码时,"期望"是用采样估计的
log_prob = dist.log_prob(action)
loss = -(log_prob * reward).mean()
loss.backward()
```

那个 `.mean()` 在做什么?**它在用 $N$ 个采样的动作估计期望**。如果 $N$ 太小,梯度就有噪声;如果 $N$ 太大,显存就爆。这背后的数学就是**蒙特卡洛估计**。

**更戳心的问题**:

- VAE 的 ELBO 里, $\mathbb{E}_{q(z|x)}[\log p(x|z)]$ 为什么能用 $\frac{1}{L}\sum \log p(x|z_\ell)$ 估计? (答:采样)
- Off-Policy RL 评估,为什么需要**重要性权重** $w = \pi/\mu$? (答:IS)
- BNN 推断后验 $p(\theta|D)$, 为什么用 MCMC? (答:MCMC)
- 推荐系统的负采样,为什么是"从简单分布采几个负样本"? (答:RS + IS 的工程近似)

**这一章的答案**: 这些"用采样代替解析计算"的地方, 用的都是本章 5 种基本方法之一。

- `torch.multinomial` → **直接采样(逆变换)**
- 负采样 → **重要性采样**(简化版)
- PPO 的 Clip → **带约束的重要性采样**
- MCMC 推断后验 → **MCMC / HMC / NUTS**
- 任何"按 $p(x)/Mq(x)$ 接受/拒绝"的过程 → **拒绝采样**

**一句话**: 这章讲的 5 种方法, 就是 **"我没办法把 $p(x)$ 写出来/积分, 那我就用采样绕过它"** 的所有工业级解法。

### 1.2 直觉:用频率估计期望

**核心事实**:如果 $X_1, X_2, \ldots, X_n$ 是从 $p(x)$ 里**独立**采出的样本,那么

$$
\mathbb{E}_p[f(X)] = \int f(x) p(x) \, dx \approx \frac{1}{n} \sum_{i=1}^n f(X_i)
$$

直观上:**期望 = 函数的平均**。我们不知道 $p(x)$ 怎么积分,但我们能采 $n$ 个样本,把它们带进 $f$,然后求平均。样本越多,平均就越接近真实期望。这就是**蒙特卡洛估计**。

**为什么有效**?**大数定律(LLN)**:独立同分布样本的均值以概率 1 收敛到期望。

$$
\bar f_n = \frac{1}{n} \sum_{i=1}^n f(X_i) \xrightarrow{a.s.} \mathbb{E}_p[f(X)]
$$

**收敛速度**?**中心极限定理(CLT)**:

$$
\sqrt{n} (\bar f_n - \mathbb{E}_p[f(X)]) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)
$$

其中 $\sigma_f^2 = \text{Var}_p[f(X)]$。意思是:**误差量级是 $O(1/\sqrt{n})$,要把精度提高 10 倍,需要 100 倍样本**。这就是为什么"采样越多越好"是有上限的(更详细的讨论见 §3、§4)。

### 1.3 ASCII 可视化:Monte Carlo 估计 $\int f(x) dx$

```text
真实情况: 我们想算 ∫_0^1 f(x) dx (假设 f(x) = x^2)
真实值 = 1/3 ≈ 0.3333

Monte Carlo 方法:
   1. 从 Uniform(0,1) 采 n 个点 x_1, ..., x_n
   2. 计算 f 的平均 = (1/n) Σ f(x_i)
   3. 这就是 ∫ f(x) dx 的估计

图示 (n=20 个采样点):

   f(x)
    │
  1 ┤        ●
    │       ╱  ╲
 0.8┤      ●    ●
    │     ╱      ╲
 0.6┤    ●        ●
    │   ╱          ╲
 0.4┤  ●            ●
    │ ╱              ●
 0.2┤●                ●
    │
  0 ┼──┬──┬──┬──┬──┬──► x
    0  0.2 0.4 0.6 0.8 1

    ● 是采样点,每个点贡献 f(x_i)/n 到估计
    估计 = (1/n) Σ f(x_i) ≈ 0.35 (n=20 时的典型值)
    真实值 = 1/3 = 0.3333

随 n 增长的收敛 (方差衰减):
    n=10:    估计 ≈ 0.35 ± 0.10
    n=100:   估计 ≈ 0.34 ± 0.03
    n=10000: 估计 ≈ 0.3333 ± 0.003
```

**关键直觉**:
- Monte Carlo 把"积分"变成"平均"
- 平均的方差是 $O(1/n)$,所以要 100 倍样本才能换 10 倍精度
- 前提是**样本独立**;如果是 MCMC 这种"有自相关"的样本,收敛会更慢(后面 §5 讲)

### 1.4 案例:为什么 Policy Gradient 强化学习用采样估计梯度?

这是最经典的 ML/DL "用采样估计期望" 的例子,值得完整推一遍。

**问题**:在策略 $\pi_\theta(a|s)$ 下,定义目标函数:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p(\tau | \theta) R(\tau) \, d\tau
$$

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是轨迹, $R(\tau) = \sum_t \gamma^t r_t$ 是累积奖励。

**直接对 $J$ 求梯度**(用 log-derivative trick):

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p(\tau|\theta) R(\tau) \, d\tau = \int p(\tau|\theta) \nabla_\theta \log p(\tau|\theta) R(\tau) \, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log p(\tau|\theta) R(\tau)]
$$

**问题**:**这个期望是 $d\tau$ 上的积分,无法解析**。轨迹空间是无限维的(连续动作空间下)。

**采样近似**:采 $N$ 条轨迹 $\tau_1, \ldots, \tau_N \sim \pi_\theta$:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p(\tau_i | \theta) R(\tau_i) = \frac{1}{N} \sum_{i=1}^N \left( \sum_t \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \right) R(\tau_i)
$$

这就是 **REINFORCE 算法** (Williams 1992)。

```python
log_probs = dist.log_prob(action)        # ∇log π(a|s)
loss = -(log_probs * reward).mean()      # 蒙特卡洛估计的负梯度
loss.backward()
```

`.mean()` 那行就是"用 $N$ 个采样估计期望"。 $N$ 太小梯度噪声大, $N$ 太大样本效率低。 改进版本 (A2C, PPO) 都是在降方差, PPO 的 Clip 见 §4。

**推广**: 任何 $\mathbb{E}_{p(x)}[f(x)]$, 只要能从 $p$ 采样, 就能用 $\frac{1}{N}\sum f(x_i)$ 估计, 本章所有方法的统一前提。

### 1.5 4 个手算例子:蒙特卡洛在简单积分上的应用

#### 例 1: 估计 $\int_0^1 x^2 dx = 1/3$

**解析**: $\int_0^1 x^2 dx = [x^3/3]_0^1 = 1/3 \approx 0.3333$

**Monte Carlo**:
- 从 $U(0,1)$ 采 5 个点: $0.1, 0.5, 0.7, 0.3, 0.9$
- $f(x) = x^2$ 的值: $0.01, 0.25, 0.49, 0.09, 0.81$
- 平均: $(0.01 + 0.25 + 0.49 + 0.09 + 0.81) / 5 = 1.65 / 5 = 0.33$ ← 完美接近 $1/3$

#### 例 2: 估计 $\int_0^\infty e^{-x} dx = 1$

**解析**: $\int_0^\infty e^{-x} dx = [-e^{-x}]_0^\infty = 0 - (-1) = 1$

**Monte Carlo**: 我们不能从 $U(0,\infty)$ 采(没有这样的分布),要从 $U(0,1)$ 采然后**变量替换** $x = t/(1-t)$ (或等价地用 $X \sim \text{Exp}(1)$,直接采):

- 采 5 个 $X \sim \text{Exp}(1)$: $0.5, 1.2, 0.3, 2.1, 0.8$
- $f(x) = e^{-x}$: $0.607, 0.301, 0.741, 0.122, 0.449$
- 平均: $2.220 / 5 = 0.444$ ← 5 个样本估计不准,再采 5 个:

  $(0.1, 0.7, 1.5, 0.4, 0.6) \to (0.905, 0.497, 0.223, 0.670, 0.549)$,平均 $0.569$

  10 个样本平均: $(2.220 + 2.844) / 10 = 0.506$ ← 在接近 1
- **直觉**:指数分布尾部大,样本方差大,需要更多样本才能稳定

#### 例 3: 估计 $\int_0^1 \sin(\pi x) dx = 2/\pi \approx 0.6366$

**解析**: $\int_0^1 \sin(\pi x) dx = [-\cos(\pi x)/\pi]_0^1 = (1 - (-1))/\pi = 2/\pi$

**Monte Carlo**:
- 5 个 $U(0,1)$: $0.2, 0.4, 0.6, 0.8, 0.5$
- $\sin(\pi x)$: $0.588, 0.951, 0.951, 0.588, 1.0$
- 平均: $4.078 / 5 = 0.816$ ← 5 个样本方差大
- 再加 5 个: $0.1, 0.3, 0.7, 0.9, 0.45$
- $\sin(\pi x)$: $0.309, 0.809, 0.809, 0.309, 0.988$
- 10 个样本平均: $(4.078 + 3.224) / 10 = 0.730$ ← 在接近 $2/\pi \approx 0.637$

#### 例 4: 估计 $\pi$ (经典 Buffon's needle 替代: 投点法)

**问题**:在单位正方形内随机投点,落在内切圆 (半径 1/2) 里的概率 = $\pi/4$。

**Monte Carlo**:
- 投 4 个点: $(0.1, 0.1), (0.4, 0.2), (-0.3, 0.4), (0.2, -0.1)$
- 圆内 (到原点距离 $\leq 0.5$): 前 3 个在内,第 4 个在内但有负坐标,实际 3/4 在内
- $\hat\pi = 4 \times 3/4 = 3.0$ ← 4 个样本估计太离谱
- 投 100 个点,假设 78 个在圆内: $\hat\pi = 4 \times 78/100 = 3.12$ ← 仍不准
- 投 10000 个点,假设 7853 个在圆内: $\hat\pi = 4 \times 7853/10000 = 3.141$ ← 接近真实 $\pi$

**核心**:$1/\sqrt{n}$ 收敛。 要让 $\pi$ 估计误差 < 0.001, 需要 ~$10^6$ 样本。 高维 ($d=10$) 单位球体积的 MC 估计几乎 0 个采样点落在球内,这就是**维度诅咒**: 直接拒绝采样在高维失效。 §3 和 §6 会展开。

### 1.6 章节地图:5 种采样方法的统一视角

```text
"我想从 p(x) 采样 / 估计 E_p[f]"
         │
         ├── 1. 我能从 p 直接采样吗? (知道 CDF, 知道反函数)
         │     └── 是 → 逆变换采样 (§2)
         │
         ├── 2. 我能从"近似分布" q 采样,且 q 容易采,且 p ≤ Mq?
         │     └── 是 → 拒绝采样 (§3)
         │
         ├── 3. 我能从 q 采,但 q ≠ p,需要加权?
         │     └── 是 → 重要性采样 (§4)
         │
         ├── 4. 我能算出 p(x) (到常数),但不知道怎么采?
         │     └── 是 → MCMC / MH / Gibbs / HMC (§5-§8)
         │
         └── 5. 我有难算的后验,需要推断?
              └── 是 → 综合使用 (PyMC/Stan) (§9-§10)
```

这一章按上面的顺序,先讲**精确方法**(§2 逆变换),再讲**近似但简单的方法**(§3 拒绝、§4 重要性),最后讲**通用但慢的方法**(§5-§8 MCMC 家族)。每种方法都有它的"用武之地",也有它的"坑"。

---

## 2. 直接采样:逆变换方法

### 2.1 逆变换采样: $U \sim U(0,1), X = F^{-1}(U)$

**定理(逆变换采样, Inverse Transform Sampling)**:

设 $X$ 是连续随机变量,CDF 为 $F_X(x) = P(X \leq x)$。如果 $F$ **严格单调递增且可逆**,那么:

$$
X = F_X^{-1}(U), \quad U \sim U(0,1) \implies X \text{ 的分布与 } F_X \text{ 一致}
$$

**直观理解**:

```text
       概率密度 p(x)              CDF F(x)                逆 CDF F^(-1)(u)

       │    ╱╲                   │           ●●●●         │
       │   ╱  ╲                  │        ●●     ●●       │       ╱
       │  ╱    ╲                 │      ●●         ●●     │      ╱
       │ ╱      ╲                │    ●●             ●●   │     ╱
       │╱        ╲               │  ●●                 ●● │    ╱
       ┼────────────► x           ●●______________________●●●──────────► u
                                 0                        1

       1. 采 u ~ U(0,1)          2. 查 CDF: F(x) = u       3. 找 x = F^(-1)(u)

       u 是"概率", 找 x 使得: "x 以下累积概率恰好 = u"
```

**为什么有效**?设 $X = F^{-1}(U)$,那么:

$$
P(X \leq x) = P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)
$$

最后一步用了 $U \sim U(0,1)$ 的 CDF 就是 $F_U(u) = u$ 在 $[0,1]$ 上。

### 2.2 5 个手算例子

#### 例 1: 指数分布 $X \sim \text{Exp}(\lambda)$

**CDF**: $F(x) = 1 - e^{-\lambda x}$。反函数: $x = -\log(1-u)/\lambda$。

**手算**: $\lambda = 1$, 采 $u = 0.5$:

$$
x = -\log(1 - 0.5)/1 = -\log(0.5) = \log 2 \approx 0.693
$$

验算: $P(X \leq 0.693) = 1 - e^{-0.693} = 1 - 0.5 = 0.5$ ✓

#### 例 2: 均匀分布 $X \sim U(a, b)$

**CDF**: $F(x) = (x-a)/(b-a)$。反函数: $x = a + (b-a)u$。

**手算**: $a=2, b=5$, 采 $u = 0.3$:

$$
x = 2 + 3 \times 0.3 = 2.9
$$

直观:30% 概率位置,确实应该是 $2 + 0.3 \times 3 = 2.9$ ✓

#### 例 3: 柯西分布 $X \sim \text{Cauchy}(0, 1)$

**CDF**: $F(x) = (1/\pi) \arctan(x) + 1/2$。反函数: $x = \tan(\pi(u - 1/2))$。

**手算**: 采 $u = 0.75$:

$$
x = \tan(\pi \times 0.25) = \tan(\pi/4) = 1
$$

直观:柯西分布的中位数是 0,75% 分位数应该是 1 ✓

#### 例 4: 用逆变换从对数正态分布采样

**对数正态**: $X = e^Y$, $Y \sim \mathcal{N}(\mu, \sigma^2)$。

**采样方法**:
1. 采 $Y \sim \mathcal{N}(\mu, \sigma^2)$ (用 Box-Muller,见 §2.3)
2. 令 $X = e^Y$

**手算**: $\mu = 0, \sigma = 1$, 采 $Y = 0.5$ (假设):

$$
X = e^{0.5} \approx 1.649
$$

#### 例 5: 离散分布 (Categorical)

**问题**: 5 个类别,概率 $(0.1, 0.2, 0.3, 0.3, 0.1)$。采一个类别。

**算法**(别名采样 / Inverse CDF for discrete):

```text
累积概率:
   类别 1:  0.0 → 0.1
   类别 2:  0.1 → 0.3
   类别 3:  0.3 → 0.6
   类别 4:  0.6 → 0.9
   类别 5:  0.9 → 1.0

采 u = 0.42:
   0.42 ∈ [0.3, 0.6] → 类别 3
```

**代码对应**:`torch.multinomial(probs, 1)` 内部就是这种累积概率查找。

### 2.3 案例:为什么 PyTorch 的正态分布采样不直接用逆变换?

**问题**:对 $\mathcal{N}(0, 1)$,CDF 是 $\Phi(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt$。**没有闭式反函数**。

**实际做法: Box-Muller 变换 (1958)**

定理: 如果 $U_1, U_2 \sim U(0,1)$ 独立,那么:

$$
Z_1 = \sqrt{-2 \log U_1} \cos(2\pi U_2), \quad Z_2 = \sqrt{-2 \log U_1} \sin(2\pi U_2)
$$

**都独立且服从 $\mathcal{N}(0, 1)$**。

**手算**: $U_1 = 0.5, U_2 = 0.25$:

$$
r = \sqrt{-2 \log 0.5} = \sqrt{2 \log 2} \approx 1.177
$$
$$
Z_1 = 1.177 \times \cos(\pi/2) = 1.177 \times 0 = 0
$$
$$
Z_2 = 1.177 \times \sin(\pi/2) = 1.177 \times 1 = 1.177
$$

**对应 PyTorch**: `torch.manual_seed(42); torch.randn(5)` (内部用 Box-Muller 或 Ziggurat)。

**工程含义**:
- `torch.distributions.Normal` 内部用**变换采样 + 数值方法** (Marsaglia-Tsang 的 Ziggurat 算法更快)
- 对**没有闭式反函数的分布** (Beta, Gamma, Dirichlet), PyTorch 用**拒绝采样或变换组合** (见 §3)

### 2.4 逆变换采样的局限性

**逆变换适用条件**:
1. 分布的 CDF $F$ 已知
2. $F$ 严格单调递增
3. $F^{-1}$ 可以**有效计算** (闭式或数值)

**不适用的情况**:
- 多维分布: $F$ 没有简单的"反函数"概念
- 无闭式 CDF 的分布 (正态、Beta、Gamma 的 CDF 都没有初等反函数)
- 后验分布 $p(\theta|D)$: 通常没有闭式 CDF

**结论**:逆变换是"理论工具",**实际工程中,我们用近似分布 (拒绝采样) 或数值积分 (重要性采样) 或 MCMC**。这就是为什么这一章剩下的部分更重要。

---

## 3. 拒绝采样 (Rejection Sampling)

### 3.1 形式化:从 $q$ 采,以 $\min(1, p/Mq)$ 接受

**问题**:我们想从 $p(x)$ 采样,但 $p$ 没有闭式 CDF,或者 $p$ 是高维的,没法直接采。

**解法**:找一个**容易采样的提议分布 (proposal)** $q(x)$, 要求:

$$
\forall x: p(x) \leq M \cdot q(x)
$$

即 $Mq(x)$ 是 $p(x)$ 的一个**上界**。

**算法 (Rejection Sampling)**:

```text
输入: 目标 p(x), 提议 q(x), 常数 M
输出: 服从 p(x) 的样本

循环:
   1. 从 q(x) 采样 x*           # 候选样本
   2. 从 U(0, 1) 采 u            # 接受概率
   3. if u < p(x*) / (M q(x*)): # 接受准则
        接受 x*, 输出
      else:
        拒绝 x*, 重新采样
```

**为什么有效**?

**定理**: 接受 $x^*$ 的概率 = $p(x^*) / (M q(x^*))$。证明:

$$
P(\text{接受 } x^*) = \int_0^1 \mathbb{1}\left[u < \frac{p(x^*)}{M q(x^*)}\right] du = \frac{p(x^*)}{M q(x^*)}
$$

那么,样本被接受的**条件分布**是:

$$
p_{\text{acc}}(x) = \frac{P(\text{接受 } x)}{P(\text{接受})} = \frac{p(x)/(Mq(x))}{\int p(x')/(Mq(x')) dx'} = \frac{p(x)/(Mq(x))}{1/M} = p(x)
$$

最后一步用了 $\int p(x)/(Mq(x)) dx = 1/M$ (因为 $Mq(x) \geq p(x)$, 所以 $1/M \geq \int p/q \cdot 1/M = 1$... 等等, 我重做一下):

实际计算:
$$
\int \frac{p(x)}{M q(x)} q(x) dx = \int \frac{p(x)}{M} dx = \frac{1}{M}
$$

所以 $P(\text{接受}) = \int P(\text{接受 } x) q(x) dx = \int p(x)/(Mq(x)) \cdot q(x) dx = 1/M$。

那么条件分布:
$$
p_{\text{acc}}(x) = \frac{P(\text{接受 } x) q(x)}{P(\text{接受})} = \frac{[p(x)/(Mq(x))] \cdot q(x)}{1/M} = \frac{p(x)/M}{1/M} = p(x) \checkmark
$$

### 3.2 接受率 = $1/M$

**核心事实**: 接受率 = $1/M$。也就是说, $M$ 越大, 接受率越低, 采样越慢。

**为什么?** 直观上, $M$ 是 $p$ 相对 $q$ 的"放大倍数"。 $Mq$ 离 $p$ 越远 (越大), 我们采到 $q$ 的样本被拒绝的概率就越高。

**关键推论**: $M$ 越小越好,但必须保证 $Mq(x) \geq p(x)$ 对所有 $x$ 成立。这就是"最优 $M$" 的来源: $M_{\min} = \max_x p(x)/q(x)$。

### 3.3 ASCII 可视化:拒绝采样

```text
p(x) ─ 目标分布 (难直接采)
q(x) ─ 提议分布 (易采)
M ─ 放大倍数 (M q(x) 是 p(x) 的上界)

p(x)        M·q(x)
 │            │
 │   ╱╲       │   ╱──╲
 │  ╱  ╲      │  ╱    ╲
 │ ╱    ╲     │ ╱      ╲
 │╱      ╲    │╱        ╲
 ┼─────────►  ┼───────────►
    目标 p     提议 Mq

                ┌──────────────────────┐
                │ 在 Mq 下均匀采样候选点  │
                │ x* 和 u ∈ U(0,1)      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ 判断 u < p(x*)/(M q)? │
                └──────────┬───────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
              yes                    no
                │                     │
                ▼                     ▼
          接受 x* (输出)         拒绝 (重新采样)

接受率 = (p 下面积) / (Mq 下面积) = 1/M
```

### 3.4 5 个手算例子

#### 例 1: 从 $p(x) = 0.6 \cdot \mathcal{N}(0, 1) + 0.4 \cdot \mathcal{N}(3, 0.5^2)$ (高斯混合) 采样

**提议**: $q(x) = \mathcal{N}(0, 1)$, $M = ?$

**计算 $M$**: $p_{\max} = 0.6 \cdot 0.4 + 0.4 \cdot 0.8 \approx 0.56$ (在 $x=0$ 和 $x=3$ 附近), $q(0) = 0.4$。

保守取 $M = 2$ (这样 $Mq(x) \geq p(x)$ 在大部分 $x$ 成立)。

**手算一次采样**:
1. 采 $x^* \sim \mathcal{N}(0,1)$: 设 $x^* = 0.5$
2. $p(0.5) = 0.6 \cdot \mathcal{N}(0.5|0,1) + 0.4 \cdot \mathcal{N}(0.5|3, 0.5^2) = 0.6 \cdot 0.352 + 0.4 \cdot 0.0000 \approx 0.211$
3. $Mq(0.5) = 2 \cdot 0.352 = 0.704$
4. 接受率: $0.211 / 0.704 \approx 0.300$
5. 采 $u = 0.2 < 0.3$: **接受 $x^* = 0.5$**

理论接受率 = $1/M = 1/2 = 50\%$。 实际 50% 的候选会被接受, 50% 拒绝。

#### 例 2: 从 Beta(2, 5) 采样,用 $U(0, 1)$ 作提议

**目标**: $p(x) = x^{2-1} (1-x)^{5-1} / B(2, 5) = 60 x (1-x)^4$ for $x \in [0,1]$

**提议**: $q(x) = U(0,1)$, $q(x) = 1$ on $[0,1]$。

**找 $M$**: $p$ 在 $x = 1/5$ 取最大值, $p(1/5) = 60 \cdot 0.2 \cdot 0.8^4 = 60 \cdot 0.2 \cdot 0.4096 = 4.92$。取 $M = 5$。

**手算**:
1. 采 $x^* = 0.2 \sim U(0,1)$
2. $p(0.2) = 4.92$
3. $Mq(0.2) = 5 \cdot 1 = 5$
4. 接受率: $4.92 / 5 = 0.984$
5. 采 $u = 0.5 < 0.984$: **接受**

接受率约 98%,**对 $U$ 提议来说很高**,因为 $p$ 的范围 [0, 4.92] 与 $U$ 差距不大。

#### 例 3: 提议分布"过宽",接受率低

**目标**: $p(x) = \mathcal{N}(0, 1)$, $q(x) = \mathcal{N}(0, 100)$, $M = 2$ (因为 $q$ 比 $p$ 宽 100 倍, $M$ 必须 $\geq 100$ 才能保证 $Mq \geq p$ 在所有 $x$)。

如果误用 $M = 2$:
- 采 $x^* = 50$ (因为 $q$ 宽,容易采到 50)
- $p(50) \approx 0$ (高斯尾部)
- $Mq(50) = 2 \cdot 0.04 = 0.08$
- 接受率: $0/0.08 = 0$
- **永远拒绝**

正确: $M = 100$。接受率 = $1/100 = 1\%$。要采 100 个 $q$ 才能拿到 1 个 $p$ 样本。

**教训**:**$M$ 必须真的包住 $p$,否则会有 $x$ 永远被拒绝**。

#### 例 4: 提议分布"过窄",完全错过 $p$ 的尾部

**目标**: $p(x) = 0.5 \mathcal{N}(-3, 0.5) + 0.5 \mathcal{N}(3, 0.5)$ (双峰)

**提议**: $q(x) = \mathcal{N}(0, 0.5)$ (太窄, 只能采到 0 附近)

- 采 $x^* = 0.1$
- $p(0.1) \approx 0$ (双峰在 ±3, 中间是 0)
- 永远拒绝

**结论**:**$q$ 必须包住 $p$ 的"支撑"**。如果 $p$ 有多个峰, $q$ 必须够宽或者多模态。

#### 例 5: 半截正态 (提议域与目标域不匹配)

**目标**: $p(x) = 2 e^{-x^2}/\sqrt{\pi}$ for $x \geq 0$。 提议用全空间 $\mathcal{N}(0, 0.5)$ 时, $q$ 包含负半轴 (采到负数 $p=0$,被拒), 接受率浪费。 **修正**: 提议也限制到 $x \geq 0$ (半截正态), 接受率近 100%。

### 3.5 案例:为什么拒绝采样在高维失效 (维度诅咒)

**问题**: $d$ 维, $p(\mathbf{x}) = \mathcal{N}(0, I_d)$。提议 $q(\mathbf{x}) = \mathcal{N}(0, \sigma^2 I_d)$。

**接受率**: $1/M = 1/\max_x p(x)/q(x)$。

在 $x = 0$ (原点), $p(0) = (2\pi)^{-d/2}$。 $q(0) = (2\pi \sigma^2)^{-d/2}$。

比值: $p(0)/q(0) = (2\pi)^{-d/2} / (2\pi \sigma^2)^{-d/2} = \sigma^d$。

**接受率 = $1/\sigma^d$**。

**直观**: $p$ 集中在原点附近, 体积 $\sim 1$。 $q$ 散在 $\sigma^d$ 体积里。采到原点的概率 = $1/\sigma^d$。 $d=10, \sigma=2$ 接受率 = $1/1024 \approx 0.1\%$。$d=100$, 接受率 = $1/2^{100}$,**基本是 0**。

**这正是维度诅咒**。拒绝采样的接受率随维度指数衰减。

**工程含义**:
- 低维 ($d \leq 5$): 拒绝采样 OK
- 中维 ($5 < d \leq 20$): 拒绝采样慢, MCMC 起步
- 高维 ($d > 20$): **MCMC / HMC / NUTS / VI** 才是正路

**为什么 MCMC 不怕维度**: MCMC 不用"全空间均匀覆盖",而是**沿着链走**。每步只走一小步,接受率高(50%-80%)。这就是 Metropolis-Hastings 的优势。

### 3.6 拒绝采样的工程代码

```python
import torch

def p_target(x):  # Beta(2, 5) 未归一化
    return torch.where((x >= 0) & (x <= 1), 60 * x * (1 - x)**4, torch.zeros_like(x))

M = 5.0  # 提议 U(0,1), M >= max p
samples = []
attempts = 0
while len(samples) < 10000:
    n_needed = 10000 - len(samples)
    x_star = torch.rand(n_needed)
    u = torch.rand(n_needed)
    accept = u < p_target(x_star) / (M * 1.0)  # q=1
    samples.append(x_star[accept])
    attempts += n_needed
print(f"接受率 = {10000 / attempts:.2%}")
```

**优化**: `torch.distributions.Beta` 内部用变换采样 + Gamma 采样, 不用拒绝采样 (Gamma 有更高效算法)。

### 3.7 接受率诊断

- 接受率 > 50%: $M$ 太保守, 浪费
- 接受率 20-50%: 合理
- 接受率 < 5%: 太低, 换提议或换方法
- 接受率 = 0%: $M$ 不够, 或 $p$ 有 $q$ 采不到的支撑

**自适应**: 采一段时间, 统计 $\max p/(Mq)$ 实际值, 调 $M$。

---

## 4. 重要性采样 (Importance Sampling)

### 4.1 直觉:从简单分布采, 加权估计

**问题**: 我们想算 $\mathbb{E}_{p}[f(X)] = \int f(x) p(x) dx$, 但 $p$ **不好直接采样**。

**关键想法**: 如果我们能从**另一个分布** $q$ 采样, 那么:

$$
\mathbb{E}_p[f(X)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_q\left[f(X) \frac{p(X)}{q(X)}\right]
$$

**核心: 加权采样估计**。 样本从 $q$ 采, 但每个样本乘以**重要性权重** $w(x) = p(x)/q(x)$。

**蒙特卡洛估计**:

$$
\mathbb{E}_p[f(X)] \approx \frac{1}{n} \sum_{i=1}^n f(X_i) \frac{p(X_i)}{q(X_i)} = \frac{1}{n} \sum_{i=1}^n f(X_i) w(X_i)
$$

其中 $X_1, \ldots, X_n \sim q$。

**直观**:

```text
        真实情况:
        ──────────────────────────────────────
        期望 = ∫ f(x) p(x) dx
                ↑     ↑
                函数   难采的分布

        用 q 重写:
        ──────────────────────────────────────
        期望 = ∫ [f(x) · p(x)/q(x)] · q(x) dx
                ↑                  ↑      ↑
                f 值              权重   好采的分布

        采样估计:
        ──────────────────────────────────────
        期望 ≈ (1/n) Σ f(x_i) · p(x_i)/q(x_i)
                 ↑       ↑       ↑
                 样本    函数值   重要性权重
                 (从 q 采)
```

**关键观察**:
- 期望**无偏** (不依赖 $n$)
- 但**方差**取决于 $p/q$ 的接近程度: 越接近, 方差越小

### 4.2 形式化: $\mathbb{E}_p[f] = \mathbb{E}_q[f \cdot p/q]$

**精确方差公式**:

$$
\text{Var}_q[f(X) w(X)] = \mathbb{E}_q[f^2(X) w^2(X)] - (\mathbb{E}_p[f(X)])^2
$$

如果 $p = q$, $w \equiv 1$, 方差 = $\text{Var}_p[f]$, 即"标准 Monte Carlo 方差"。

**相对效率 (Relative Efficiency)**:

$$
\text{Var}_q[f \cdot w] / \text{Var}_p[f] = \mathbb{E}_q[(p/q)^2] \cdot \mathbb{E}_p[f^2] / \text{Var}_p[f]
$$

**含义**: $p/q$ 越"平坦", 重要性采样越有效; $p/q$ 越"尖" (即 $p$ 和 $q$ 差距大), 重要性采样方差越大。

**极端情况**: 如果 $q$ 在 $p$ 的支撑上很稀疏, $p/q$ 在某些 $x$ 上爆炸, 方差会"坏掉"。

### 4.3 5 个手算例子

#### 例 1: 估计稀有事件概率

**问题**: 估计 $P(X > 5)$, $X \sim \mathcal{N}(0, 1)$。 解析解: $\approx 2.87 \times 10^{-7}$。

**直接 Monte Carlo**: 采 $10^6$ 个, 期望看到 0 个, **根本估计不了**。

**重要性采样**: 提议 $q(x) = \mathcal{N}(5, 1)$ (中心在 5)。

权重: $w(x) = p(x)/q(x) = \mathcal{N}(x|0,1) / \mathcal{N}(x|5,1) = e^{-(x^2 - (x-5)^2)/2} = e^{-(10x - 25)/2}$。

**手算**: 采 5 个 $x_i \sim q$:
$x_1 = 4.5, x_2 = 5.2, x_3 = 4.8, x_4 = 5.5, x_5 = 4.7$

权重:
$w_1 = e^{-(45 - 25)/2} = e^{-10} \approx 4.54 \times 10^{-5}$
$w_2 = e^{-(52 - 25)/2} = e^{-13.5} \approx 1.38 \times 10^{-6}$
$w_3 = e^{-(48 - 25)/2} = e^{-11.5} \approx 1.01 \times 10^{-5}$
$w_4 = e^{-(55 - 25)/2} = e^{-15} \approx 3.06 \times 10^{-7}$
$w_5 = e^{-(47 - 25)/2} = e^{-11} \approx 1.67 \times 10^{-5}$

$f(x) = \mathbb{1}[x > 5]$: $f_1=0, f_2=1, f_3=0, f_4=1, f_5=0$

估计: $\hat P = (1/5)(0 \cdot 4.54e^{-5} + 1 \cdot 1.38e^{-6} + 0 + 1 \cdot 3.06e^{-7} + 0)$
$= (1/5)(1.69e^{-6}) \approx 3.4 \times 10^{-7}$

数量级正确! 真实值 $\approx 2.87 \times 10^{-7}$。

**5 个样本就估计到了 1e-7 量级的事件概率**。 这就是重要性采样的威力。

#### 例 2: 估计积分 $\int f(x) p(x) dx$

**问题**: $p(x) = \mathcal{N}(0,1)$, $f(x) = e^{-x^2}$, 估计 $\int e^{-x^2} \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = \int \frac{1}{\sqrt{2\pi}} e^{-3x^2/2} dx = \frac{1}{\sqrt{3}}$。

**解析解**: $1/\sqrt{3} \approx 0.5774$。

**重要性采样**: 提议 $q = \mathcal{N}(0, 1)$ (刚好等于 $p$! 此时 $w = 1$, 退化为标准 MC)。

采 5 个 $x_i \sim \mathcal{N}(0, 1)$: $-0.5, 0.3, 1.0, -0.8, 0.2$。 $f(x_i) = e^{-x_i^2}$: $0.779, 0.914, 0.368, 0.527, 0.961$。 权重 $w_i = 1$ (因为 $p = q$)。 估计: $0.710$, 偏离 $0.577$ 这是 **IS 的方差问题**, 5 个样本不够。

**改用 $q = \mathcal{N}(0, 2)$** (更宽的提议): $w(x) = \sqrt{2} e^{-x^2/4}$。 采 5 个: $w_i$ 是 $(1.305, 1.115, 0.948, 1.358, 1.250)$, $fw$ 是 $(1.193, 0.588, 0.285, 1.058, 0.872)$。

估计: $(1.193 + 0.588 + 0.285 + 1.058 + 0.872)/5 = 0.799$

嗯, 偏离 $0.577$ 较多。 **这就是 IS 的方差问题**。 5 个样本不够。

#### 例 3: 估计 KL 散度

**问题**: $p(x) = \mathcal{N}(0, 1)$, $q(x) = \mathcal{N}(0.5, 1)$, 估计 $D_{\text{KL}}(p \| q)$。

**解析解**: $D_{\text{KL}}(p\|q) = 0.5^2 / (2 \cdot 1) = 0.125$ (两个一维高斯的 KL 公式)。

**IS 估计**: 采 $x \sim p$, $\hat D_{\text{KL}} = (1/n) \sum \log(p(x)/q(x))$

采 5 个 $x_i \sim \mathcal{N}(0, 1)$: $-0.2, 0.3, 0.8, -0.5, 0.1$

$\log(p(x_i)/q(x_i)) = -\frac{(x_i - 0)^2}{2} + \frac{(x_i - 0.5)^2}{2} = \frac{-x_i^2 + (x_i - 0.5)^2}{2} = \frac{x_i - 0.25}{2} \cdot 1$

嗯, 不对, 重算:

$\log p(x) = -x^2/2 + C$, $\log q(x) = -(x-0.5)^2/2 + C$

$\log p(x) - \log q(x) = -x^2/2 + (x-0.5)^2/2 = (x^2 - x + 0.25 - x^2)/2 = (-x + 0.25)/2 = (0.25 - x)/2$

代入:
$x_1 = -0.2$: $(0.25 - (-0.2))/2 = 0.225$
$x_2 = 0.3$: $(0.25 - 0.3)/2 = -0.025$
$x_3 = 0.8$: $(0.25 - 0.8)/2 = -0.275$
$x_4 = -0.5$: $(0.25 - (-0.5))/2 = 0.375$
$x_5 = 0.1$: $(0.25 - 0.1)/2 = 0.075$

平均: $(0.225 - 0.025 - 0.275 + 0.375 + 0.075)/5 = 0.375/5 = 0.075$ ← 接近 0.125, 5 个样本量级 OK

**说明**: IS 可估计 KL 散度(对数期望), 用于 t-SNE、对比学习等。

#### 例 4: 估计 $\log Z$ (归一化常数)

**问题**: $p(x) = p^*(x)/Z$, 已知 $p^*$ 但 $Z$ 难算。 用 IS 估计 $\log Z = \log \mathbb{E}_q[p^*/q]$。 这是 **Annealed Importance Sampling (AIS)** 和 **Bridge Sampling** 的核心思想, MCMC 评估归一化常数时很有用。

#### 例 5: Off-Policy 评估的强化学习

**问题**: 行为策略 $\mu$ 采集了轨迹数据 $\{s_t, a_t, r_t\}$, 想估计目标策略 $\pi$ 下的期望回报 $V^\pi = \mathbb{E}_{\pi}[\sum_t \gamma^t r_t]$。

**IS 估计**: 重要性权重 = $\pi(a|s) / \mu(a|s)$。 一条轨迹 $\tau$ 的权重:

$$
w(\tau) = \prod_t \frac{\pi(a_t | s_t)}{\mu(a_t | s_t)}
$$

**手算**: 假设一条轨迹 3 步, 每步 $\pi/\mu$ 是 $(0.8, 0.6, 0.9)$:

$w = 0.8 \times 0.6 \times 0.9 = 0.432$

如果另一条是 $(0.1, 1.5, 0.3)$: $w = 0.045$ ← 权重很小

回报 $G(\tau) = \sum_t \gamma^t r_t = 5$ (假设)

$\hat V^\pi = (1/n) \sum_i w_i G(\tau_i)$

**问题**: 权重 $w$ 是**乘法累积**, 容易"一个轨迹主导"。 这就是 PPO 用 Clip 替代 IS 的原因。

### 4.4 ASCII 可视化:重要性采样权重分布

```text
        p(x)              q(x)              w(x) = p(x)/q(x)
        目标              提议                  权重

        │   ╱╲           │       ╱╲        │       │
        │  ╱  ╲          │      ╱  ╲       │       │
        │ ╱    ╲         │     ╱    ╲      │       │
        │╱      ╲        │    ╱      ╲     │       │
        ┼───────►        ┼────►          ┼──────►
        0    1           0    1           0    1

        ─────────────────────────────────────────────
        q 集中在 1, p 集中在 0。
        w 在 x=0 附近很大(权重爆炸)

        IS 估计 = Σ f(x_i) w(x_i) / n
                  ↑        ↑
                  样本      大权重项占主导
```

**关键问题**: **"少数样本主导"** → 有效样本数 ESS << n, 方差爆炸。

### 4.5 自归一化重要性采样 (SNIS)

**问题**: 上面公式要求**知道 $p$ 的归一化常数**, 但很多情况 $p$ 未归一化 ($p^*$ = 实际 $p$ 的倍数)。

**SNIS**: 不知道归一化常数时, 用权重归一化:

$$
\mathbb{E}_p[f(X)] \approx \frac{\sum_{i=1}^n f(X_i) w(X_i)}{\sum_{i=1}^n w(X_i)}, \quad w_i = p^*(X_i)/q(X_i)
$$

**手算**: 例 2 的 $fw$ 数组是 $(1.193, 0.588, 0.285, 1.058, 0.872)$, 权重 $w$ 是 $(1.305, 1.115, 0.948, 1.358, 1.250)$, 总和 $5.976$。 SNIS: $3.996/5.976 = 0.669$。

**注意**: SNIS **有偏**, 但渐近无偏, 工程上比 IS 更常用 (因为通常不知道 $p$ 的常数)。

### 4.6 案例:为什么 Off-Policy RL 用重要性采样?

**痛点**: 旧策略 $\mu$ 采的 buffer (人工演示 / 老的 exploration), 评估新策略 $\pi$ 的期望回报, **不能重新跑** (成本高)。

**公式**: $V^\pi = \mathbb{E}_{\tau \sim \pi}[G(\tau)] = \mathbb{E}_{\tau \sim \mu}\left[\frac{p(\tau|\pi)}{p(\tau|\mu)} G(\tau)\right]$, 重要性权重 $w(\tau) = \prod_t \pi(a_t|s_t)/\mu(a_t|s_t)$。

**问题**: 权重**乘法累积**, 长轨迹 $w$ 容易爆炸/消失, ESS 极低。 SNIS 估计 $\hat V^\pi = \sum_i w_i G(\tau_i) / \sum_i w_i$。

```python
# 伪代码
for traj in buffer:  # 来自 μ
    log_w = sum(log_prob_pi - log_prob_mu for t in traj)
    weights.append(log_w.exp())
    returns.append(discounted_reward(traj))
V_pi = sum(w * r for w, r in zip(weights, returns)) / sum(weights)
```

**但权重累积几步就接近 0 或 $\infty$**, 这就是为什么 PPO 用 **Clip** 而不是原始 IS。

### 4.7 案例:为什么 PPO 用 Clip 替代 IS?

**PPO (Schulman 2017) 关键创新**: 把 IS 权重 $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$ 限制在 $[1-\epsilon, 1+\epsilon]$。

**目标**: $L^{\text{CLIP}}(\theta) = \mathbb{E}_t[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$, $\epsilon$ 通常 0.1-0.2。

```text
        L(r, A>0)              1+ε │──────╲
                                       ╲    r>1+ε: clip 到 (1+ε)A
            │                       ╲
            │                        ╲
            0┼─────────────────────────► r
                  1         1+ε
```

**对比**:

| 方法 | 优点 | 缺点 |
| :--- | :--- | :--- |
| 原始 IS | 无偏, 一致 | 方差爆炸, 不稳 |
| SNIS | 数值稳定 | 有偏 |
| **PPO Clip** | **稳定, 简单** | **有偏 (工程上无所谓)** |
| TRPO | 严格 KL 约束 | 二阶优化, 复杂 |

**结论**: **Clip 是 IS 的"工程改造"**, 牺牲一点偏差换巨大稳定性。 PPO 是 RLHF 标配 (InstructGPT, ChatGPT)。

### 4.8 重要性采样的方差控制技巧

**问题**: $w = p/q$ 在某些 $x$ 爆炸, 1-2 个样本主导估计, 方差大。

**5 个降方差技巧**:

1. **好的提议分布**: $q$ 尽量接近 $p$, 优选 $p$ 形状覆盖。
2. **重采样 (Resampling)**: 采 $n$ 个, 按 $w_i$ 重新采样 (SIR, Sampling Importance Resampling)。
3. **控制变量 (Control Variate)**: 找一个 $g$ 满足 $\mathbb{E}[g] = 0$, 加到 $f$ 上降方差。
4. **Antithetic Sampling**: 同时采 $x$ 和 $x'$ (对偶), 用 $(f(x) + f(x'))/2$ 估计。
5. **Rao-Blackwellization**: 利用 $p$ 的解析结构, 减少"无谓"维度。

**与 LLM 的接口**: LLM 推理的 PPL 估计、RLHF 的 reward 估计、对比学习的 InfoNCE, 都用了重要性采样的变体。

---

## 5. MCMC 与马尔可夫链

### 5.1 直觉:构造"细致平稳"的马尔可夫链

**核心想法**: 我们想从 $p(x)$ 采样, 但 $p$ 难直接采。 **如果能构造一个马尔可夫链, 让它的平稳分布恰好是 $p$**, 那么**运行链足够长后**, 链的每一步的样本就近似服从 $p$。

**关键概念**:
- **马尔可夫链 (Markov Chain)**: $X_{t+1}$ 只依赖于 $X_t$, 不依赖于历史。
- **转移核 (Transition Kernel)** $P(x \to y)$: 从 $x$ 走到 $y$ 的概率。
- **平稳分布 (Stationary Distribution)** $\pi$: 满足 $\pi(y) = \int \pi(x) P(x \to y) dx$。
- **细致平稳 (Detailed Balance)**: $\pi(x) P(x \to y) = \pi(y) P(y \to x)$。 满足即有 $\pi$ 是 $P$ 的平稳分布。

**直观**:

```text
        状态空间 X = {x_1, x_2, x_3, ...}
        转移: x_t → x_{t+1} (按 P)

        时刻 0:  从某 x_0 起步 (任意)
        时刻 1:  P(x_0 → ·) 决定
        时刻 2:  P(x_1 → ·) 决定
        ...
        时刻 T:  链的分布接近 π

        平稳分布 π: 直觉上, "流" 出去的 = "流" 进来的

        ┌──────────────────────────────────────┐
        │  流进 x 的 = ∫ π(x') P(x'→x) dx'   │
        │  流出 x 的 = π(x) ∫ P(x→x') dx' = π(x) │
        │  平衡: π(x) = ∫ π(x') P(x'→x) dx'   │
        └──────────────────────────────────────┘
```

### 5.2 平稳分布与细致平稳条件

**平稳分布定义**: $\pi$ 是 $P$ 的平稳分布, 如果:

$$
\pi(y) = \int \pi(x) P(x \to y) dx, \quad \forall y
$$

或用矩阵记号: $\pi = \pi P$。

**细致平稳条件 (Detailed Balance)**: 如果存在 $\pi$ 使得:

$$
\pi(x) P(x \to y) = \pi(y) P(y \to x), \quad \forall x, y
$$

那么 $\pi$ 是 $P$ 的平稳分布。

**证明**: 对 $\pi$ 求 $P$:

$$
\int \pi(x) P(x \to y) dx = \int \pi(y) P(y \to x) dx = \pi(y) \int P(y \to x) dx = \pi(y)
$$

最后一步用 $P$ 是合法转移核 ($\int P = 1$)。

**直觉**:
- 细致平稳: **任何两个状态之间, 流过去的 = 流回来的**
- 这是**充分条件** (很多 MCMC 算法的设计思路)

### 5.3 收敛性: Burn-in 与 Mixing time

**问题**: 链从某 $x_0$ 出发, 需要多久能接近平稳分布?

**两个时间尺度**:
- **Burn-in time $t_{\text{burn}}$**: 链"忘记"初始分布的时间。 $t < t_{\text{burn}}$ 的样本不应该用。
- **Mixing time $t_{\text{mix}}$**: 链"探索"完状态空间的时间。 $t > t_{\text{mix}}$ 后样本近似独立(但**实际上 MCMC 样本始终有自相关**)。

**判断收敛的诊断**:
- **R-hat (Gelman-Rubin)**: 跑多条链, 看链间方差与链内方差之比。 $R \hat{} \approx 1$ 表示收敛。
- **ESS (Effective Sample Size)**: 考虑自相关后的"等效独立样本数"。 $ESS = n / (1 + 2 \sum_k \rho_k)$, 其中 $\rho_k$ 是滞后 $k$ 步的自相关。
- **Trace plot**: 画链的轨迹, 视觉判断"是否还在飘"。

**核心事实**: MCMC 样本**有自相关**, 不像 IS 那样 IID。 因此, $n$ 个 MCMC 样本的"等效样本数"通常 << $n$。

### 5.4 ASCII 可视化:MCMC 链的收敛

```text
Trace plot (x_t 随 t 变化):
   x_t
    │      ╱╲      ╱╲
    │   ╱─╯  ╲  ╱─╯  ╲    ← mixing 后, 在平稳分布里"游走"
    │  ╱      ╲╱      ╲
    │ ╱                  ╲
    │╱                    ╲
    ┼───┬────┬────┬────┬────┬──► t
    0  200  400  600  800  1000
    ↑
    burn-in (t < 200 不用)

直方图 (t > 200 的样本):
       │    ╱╲
       │   ╱  ╲
       │  ╱    ╲
       │ ╱      ╲
       │╱        ╲
       ┼───────────►
       近似 π(x)

R-hat 诊断:
   多条链 (不同起点), 算 R-hat
   R-hat ≈ 1.00  ✓ (收敛)
   R-hat > 1.1   ✗ (还没收敛, 加长链)
```

### 5.5 5 个手算例子

#### 例 1: 两态马尔可夫链的平稳分布

**转移**:
$$P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$$

**求平稳分布**: $\pi P = \pi$, 即 $\pi_1 = 0.7 \pi_1 + 0.4 \pi_2$。

由 $0.3 \pi_1 = 0.4 \pi_2$, $\pi_2 / \pi_1 = 3/4$。 加上 $\pi_1 + \pi_2 = 1$, $\pi_1 = 4/7, \pi_2 = 3/7$。

**验证细致平稳**: $\pi_1 P_{1 \to 2} = 4/7 \cdot 0.3 = 12/70$, $\pi_2 P_{2 \to 1} = 3/7 \cdot 0.4 = 12/70$ ✓

#### 例 2: 三态链

**转移**:
$$P = \begin{pmatrix} 0.5 & 0.3 & 0.2 \\ 0.2 & 0.6 & 0.2 \\ 0.1 & 0.4 & 0.5 \end{pmatrix}$$

**求平稳分布**: 解 $\pi P = \pi, \pi_1 + \pi_2 + \pi_3 = 1$。

线性方程组:
$\pi_1 = 0.5\pi_1 + 0.2\pi_2 + 0.1\pi_3 \Rightarrow 0.5\pi_1 = 0.2\pi_2 + 0.1\pi_3$
$\pi_2 = 0.3\pi_1 + 0.6\pi_2 + 0.4\pi_3 \Rightarrow 0.4\pi_2 = 0.3\pi_1 + 0.4\pi_3 \Rightarrow \pi_2 = 0.75\pi_1 + \pi_3$

由 $\pi_1 + \pi_2 + \pi_3 = 1$ 解得: $\pi_1 = 0.222, \pi_2 = 0.444, \pi_3 = 0.333$。 重点: MCMC 设计转移核 $P$ 使目标 $\pi$ 满足平稳性。

#### 例 3: PageRank 的简化版

3 个网页循环链接 1→2→3→1, $d=0.85$ 阻尼。 **PageRank 就是 Page 分布的平稳分布**。 详细见 7-系统思维。

#### 例 4: Metropolis 算法在 $\{-1, 0, 1\}$ 上的简化

**目标分布**: $\pi = (0.1, 0.6, 0.3)$ (一维离散)。

**对称提议**: $P_{\text{prop}} = $ 随机翻一个邻居或不变 (e.g., $P_{\text{prop}}(-1|0) = 0.5, P_{\text{prop}}(1|0) = 0.5$ 等)。

**接受率**: $\alpha(x, y) = \min(1, \pi(y)/\pi(x))$

**链的转移核**: $P(x, y) = P_{\text{prop}}(x, y) \alpha(x, y)$。

例: 从 $x=0$ 提议到 $y=1$。 接受率 = $\min(1, 0.3/0.6) = 0.5$。 所以 $P(0 \to 1) = 0.5 \cdot 0.5 = 0.25$。

**验证细致平稳**: $\pi(0) P(0 \to 1) = 0.6 \cdot 0.25 = 0.15$, $\pi(1) P(1 \to 0) = 0.3 \cdot P_{\text{prop}}(1, 0) \alpha(1, 0) = 0.3 \cdot 0.5 \cdot 1 = 0.15$ ✓ (这里 $P_{\text{prop}}(1 \to 0) = 0.5$, $\alpha = \min(1, 0.6/0.3) = 1$)

#### 例 5: 链的不可约性

**定义**: 马尔可夫链是**不可约 (irreducible)** 的, 如果从任何状态都能到达任何状态。

**周期 (periodic)**: 如果 $P^n(x \to x) = 0$ 对所有奇数 $n$, 链是周期的 (有 bug)。

**MCMC 收敛的充要条件**: **不可约 + 非周期 + 平稳分布存在**。

**反例**: 二态链 $P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ 是周期的 (状态 1→2→1→2, 永远在 1 步到达)。 平稳分布 (0.5, 0.5), 但链不会"收敛", 永远在 0 和 1 之间跳。

**解决**: 加一个"自环" $P = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}$, 立即非周期化。

### 5.6 案例:为什么 LLM 的解码可以用 MCMC?

**LLM 解码**:
- **Greedy / Beam Search**: 每步选概率最大的 token。 局部最优, 但**缺乏多样性**, 容易循环。
- **Sampling (top-k, top-p)**: 随机采, 多样性好, 但**容易"跳出去"**, 生成质量不稳定。

**MCMC 视角**: 把"生成 token 序列"看作"从一个分布 $p(\text{seq})$ 里采样", 可以用 MCMC 思想:
1. 起始: 随机一个序列
2. 提议: 随机改一个 token
3. 接受/拒绝: 按 Metropolis 准则

**实验** (已有工作, 如 Ippolito et al. 2019):
- 用 MCMC 重采样序列, 发现能改善重复性问题
- 但**MCMC 慢**, Beam Search 仍是工业主流

**更现代的视角**: LLM 的 RLHF 训练中, **reward model + KL 约束** 构成了一个新的"目标分布", 可以用 MCMC 采样 (DPO 的变体已经在探索)。

### 5.7 章节过渡: 从马尔可夫链到 Metropolis-Hastings

**到目前为止**: 我们知道"如果 $P$ 的平稳分布是 $\pi$, 跑链足够长就能采到 $\pi$ 的样本"。

**剩下的问题**: **怎么构造 $P$ 使 $\pi$ 是平稳分布?**

**答案**: **Metropolis-Hastings** (§6) 一个通用的构造方法, 让 $P$ 自动满足细致平稳。 然后 **Gibbs** (§7) 和 **HMC** (§8) 是 MH 的两个变种, 在特定场景更高效。

---

## 6. Metropolis-Hastings 算法

### 6.1 形式化: 提议 $q$, 接受率 $\alpha = \min(1, \frac{p(y) q(x|y)}{p(x) q(y|x)})$

**算法 (Metropolis-Hastings)**:

```text
输入: 目标 p(x), 提议 q(y|x), 初始 x_0
输出: 序列 x_1, x_2, ...  (近似服从 p)

循环 t = 1, 2, ...:
   1. 采 y ~ q(y | x_{t-1})                    # 提议
   2. 采 u ~ U(0, 1)
   3. α(x_{t-1}, y) = min(1, [p(y) q(x_{t-1}|y)] / [p(x_{t-1}) q(y|x_{t-1})])
   4. if u < α:  x_t = y                       # 接受
      else:      x_t = x_{t-1}                  # 拒绝
```

**完整推导** (为什么接受率是 $\alpha = \min(1, \frac{p(y) q(x|y)}{p(x) q(y|x)})$):

**目标**: 构造 $P(x, y)$ 使 $\pi(x) P(x, y) = \pi(y) P(y, x)$ 对 $\pi = p$ 成立。

**分两步构造**:

**第 1 步**: 定义提议 $P_{\text{prop}}(x, y) = q(y|x)$。 这通常**不满足**细致平稳:

$$
p(x) q(y|x) \neq p(y) q(x|y) \quad \text{(一般情况)}
$$

**第 2 步**: 加一个接受率 $\alpha(x, y) \in [0, 1]$, 使:

$$
P(x, y) = q(y|x) \alpha(x, y)
$$

我们想要细致平稳:

$$
p(x) q(y|x) \alpha(x, y) = p(y) q(x|y) \alpha(y, x)
$$

**构造**: 取

$$
\alpha(x, y) = \min\left(1, \frac{p(y) q(x|y)}{p(x) q(y|x)}\right)
$$

**验证**:

情形 1: $p(y) q(x|y) \leq p(x) q(y|x)$
- $\alpha(x, y) = p(y) q(x|y) / [p(x) q(y|x)]$
- $\alpha(y, x) = 1$
- $p(x) q(y|x) \alpha(x, y) = p(x) q(y|x) \cdot p(y) q(x|y) / [p(x) q(y|x)] = p(y) q(x|y)$
- $p(y) q(x|y) \alpha(y, x) = p(y) q(x|y) \cdot 1 = p(y) q(x|y)$ ✓

情形 2: $p(y) q(x|y) \geq p(x) q(y|x)$
- $\alpha(x, y) = 1$
- $\alpha(y, x) = p(x) q(y|x) / [p(y) q(x|y)]$
- 类似验证 ✓

**结论**: $P(x, y) = q(y|x) \alpha(x, y)$ 满足细致平稳, $p$ 是 $P$ 的平稳分布。

### 6.2 完整推导 (数学形式)

**MH 转移核**:

$$
P(x, y) = \begin{cases} q(y|x) \alpha(x, y) & \text{if } y \neq x \\ q(x|x) + \sum_{z \neq x} q(z|x) (1 - \alpha(x, z)) & \text{if } y = x \end{cases}
$$

**接受率**:

$$
\alpha(x, y) = \min\left(1, \frac{p(y) q(x|y)}{p(x) q(y|x)}\right)
$$

**特例 (对称提议)**: 如果 $q(y|x) = q(x|y)$ (Metropolis 算法, 1953), 接受率简化为:

$$
\alpha(x, y) = \min\left(1, \frac{p(y)}{p(x)}\right)
$$

直观: **新位置概率大, 直接接受; 新位置概率小, 按比例接受**。

### 6.3 5 个手算例子

#### 例 1: 简单 1D 离散分布, $p = (0.1, 0.3, 0.4, 0.2)$ on $\{0, 1, 2, 3\}$

**提议**: $q(y|x)$ = 均匀选邻居 (i.e., $q(y|x) = 1/3$ for $|y-x|=1$, 否则 0)。 对称。

**接受率**: $\alpha(x, y) = \min(1, p(y)/p(x))$。

**从 $x=2$ 出发, 跑几步**:
- 提议 $y=1$ (向左): $p(1)/p(2) = 0.3/0.4 = 0.75$, $\alpha = 0.75$。 采 $u=0.3 < 0.75$: 接受, $x_1 = 1$。
- 提议 $y=0$: $p(0)/p(1) = 0.1/0.3 = 0.33$, $\alpha = 0.33$。 采 $u=0.5 > 0.33$: 拒绝, $x_2 = 1$。
- 提议 $y=2$: $p(2)/p(1) = 0.4/0.3 = 1.33$, $\alpha = 1$。 接受, $x_3 = 2$。
- 提议 $y=3$: $p(3)/p(2) = 0.2/0.4 = 0.5$, $\alpha = 0.5$。 采 $u=0.1 < 0.5$: 接受, $x_4 = 3$。
- ...

**长期频率**: 经过 burn-in 后, 状态 0, 1, 2, 3 出现频率应该接近 $(0.1, 0.3, 0.4, 0.2)$。

#### 例 2: 高斯提议采样 1D 高斯

**目标**: $p(x) = \mathcal{N}(0, 1)$。

**提议**: $q(y|x) = \mathcal{N}(x, \sigma^2)$ (以 $x$ 为中心的高斯)。

**接受率** (对称 $q$, 用 Metropolis 形式): $\alpha = \min(1, p(y)/p(x)) = \min(1, e^{-(y^2 - x^2)/2})$。

**从 $x_0 = 3$ 出发, 跑几步** (设 $\sigma = 1$):
- 提议 $y = 2.5 \sim \mathcal{N}(3, 1)$: $p(y)/p(x) = e^{-(6.25 - 9)/2} = e^{1.375} \approx 3.95$, $\alpha = 1$。 接受, $x_1 = 2.5$。
- 提议 $y = 1.8 \sim \mathcal{N}(2.5, 1)$: $p(y)/p(x) = e^{-(3.24 - 6.25)/2} = e^{1.505} \approx 4.5$, $\alpha = 1$。 接受, $x_2 = 1.8$。
- 提议 $y = 2.3 \sim \mathcal{N}(1.8, 1)$: $p(y)/p(x) = e^{-(5.29 - 3.24)/2} = e^{-1.025} \approx 0.36$, $\alpha = 0.36$。 采 $u = 0.5 > 0.36$: 拒绝, $x_3 = 1.8$。
- 提议 $y = 1.0 \sim \mathcal{N}(1.8, 1)$: $p(y)/p(x) = e^{-(1.0 - 3.24)/2} = e^{1.12} \approx 3.06$, $\alpha = 1$。 接受, $x_4 = 1.0$。
- 提议 $y = -0.2 \sim \mathcal{N}(1.0, 1)$: $p(y)/p(x) = e^{-(0.04 - 1.0)/2} = e^{0.48} \approx 1.62$, $\alpha = 1$。 接受, $x_5 = -0.2$。

**长期**: 链在 0 附近震荡, 样本 $\sim \mathcal{N}(0,1)$。

#### 例 3: 提议方差对接受率的影响

**目标**: $p(x) = \mathcal{N}(0, 1)$, 提议方差 $\sigma$。

**接受率**: 接受 = $\min(1, e^{-(y^2 - x^2)/2})$, 平均接受率 $\bar\alpha$ 随 $\sigma$ 变化:

| $\sigma$ | 接受率 | 表现 |
| :--- | :--- | :--- |
| 0.1 | ~99% | 几乎都接受, 但步长小, 自相关高 |
| 0.5 | ~88% | 平衡 |
| 1.0 | ~70% | 经典选择 |
| 2.0 | ~40% | 步长大, 容易跨过低概率区 |
| 5.0 | ~10% | 步长太大, 多数被拒 |
| 10.0 | ~3% | 几乎都拒, 链"卡住" |

**经验**: 接受率约 23% 是高维 MH 的"理论最优" (Roberts et al. 1997)。

#### 例 4: 多峰分布的混合

**目标**: $p(x) = 0.5 \mathcal{N}(-3, 0.5) + 0.5 \mathcal{N}(3, 0.5)$。

**MH 链问题**: 如果提议方差太小 ($\sigma = 0.5$), 链会"卡"在一个峰, **几乎不会跳到另一个峰**。 mixing time 极长。

**解决**: 提议方差大 ($\sigma = 5$) 反而能跨峰, 但接受率低 (见上例)。

**更好**: 用 HMC (见 §8) 或 NUTS。

#### 例 5: 比例常数不重要 (关键性质!)

**问题**: 我们只知道 $p^*(x) = p(x)/Z$, 不知道 $Z$ (归一化常数)。

**MH 接受率**:

$$
\alpha(x, y) = \min\left(1, \frac{p(y) q(x|y)}{p(x) q(y|x)}\right) = \min\left(1, \frac{p^*(y) q(x|y)}{p^*(x) q(y|x)}\right)
$$

$Z$ **约掉了**! 这就是 MH 的关键性质: **不需要知道 $p$ 的归一化常数**。 这对贝叶斯后验 $p(\theta|D) \propto p(D|\theta) p(\theta)$ 至关重要。

### 6.4 案例: MH 在 LLM 中的应用

**Likelihood-based Decoding**: LLM 生成 $y$ 时, 用 MH 最大化 $p(y|x)$ + reward: (1) 初始序列 $y_0$, (2) 提议修改: 替换几个 token, (3) 接受/拒绝: reward model 打分。 工作中如 Su & Collier 2022 用 MCMC 改进 nucleus sampling。 RLHF 中, 目标 $p(y) \propto p_{\text{LM}}(y) e^{R(y)/T}$ 可用 MH 采样 (MCTS-RL 变体)。 对比 Beam Search: 后者是贪心, MCMC 是严格采样, 多样性更好。

### 6.5 PyTorch 实现 MH 算法 (二维正态)

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# 目标分布: 2D 正态
target_mean = torch.tensor([1.0, 2.0])
target_cov = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
target_cov_inv = torch.linalg.inv(target_cov)

def log_p(x):
    # log p(x) 比例到归一化常数
    diff = x - target_mean
    return -0.5 * diff @ target_cov_inv @ diff

# MH 采样
def metropolis_hastings(n_samples=10000, n_burn=1000, proposal_std=1.0):
    samples = []
    x = torch.zeros(2)  # 初始点

    for i in range(n_samples + n_burn):
        # 提议
        y = x + torch.randn(2) * proposal_std

        # 接受率 (log 空间, 避免下溢)
        log_alpha = log_p(y) - log_p(x)
        if torch.log(torch.rand(1)) < log_alpha:
            x = y  # 接受

        if i >= n_burn:
            samples.append(x)

    return torch.stack(samples)

# 跑采样
torch.manual_seed(42)
samples = metropolis_hastings()
print(f"样本均值: {samples.mean(dim=0)}")  # 应该接近 [1, 2]
print(f"样本协方差: \n{torch.cov(samples.T)}")  # 应该接近 [[2, 0.5], [0.5, 1]]

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
ax[0].set_title('MH 2D 正态采样 (10000 samples)')
ax[0].set_xlabel('x1'); ax[0].set_ylabel('x2')

# Trace plot
ax[1].plot(samples[:1000, 0], label='x1')
ax[1].plot(samples[:1000, 1], label='x2')
ax[1].set_title('Trace plot (前 1000 步)')
ax[1].legend()
plt.show()
```

**接受率诊断**: 加 `if accept: n_accept += 1`, 跑完算 `accept_rate`。 不同 `proposal_std` 下的经验值:

```text
proposal_std=0.1: 接受率 99%, ESS 低
proposal_std=0.5: 接受率 92%
proposal_std=1.0: 接受率 78%
proposal_std=2.0: 接受率 56%
proposal_std=5.0: 接受率 20%
```

**关键参数**: 接受率 30-70% 较优。 详见 §9 工程实践。

### 6.6 MH 调参经验

| 参数 | 太小时 | 太大时 | 经验值 |
| :--- | :--- | :--- | :--- |
| 提议方差 $\sigma$ | 接受率近 100%, 自相关高 | 接受率近 0%, 链卡 | 调到接受率 30-70% |
| Burn-in | 链还没收敛 | 浪费样本 | 看 trace plot 决定 |
| 总步数 | ESS 不够 | 浪费 | ESS > 100 即可 |
| 提议分布 | 探索不够 | 拒绝太多 | 对称提议最简单 |

---

## 7. Gibbs 采样

### 7.1 形式化: 轮流从条件分布采样

**Gibbs 采样是 MH 的特例**: 提议 = "固定其他维度, 只更新一维"。

**算法**: 每步轮流从 $p(x_k | x_{-k})$ 采第 $k$ 维。 **关键**: 接受率 = 1 (因为条件采样已经是正确的分布)。

**推导**: 在 MH 框架下, 提议 = 把 $x_k$ 替换为 $x_k'$:

$$
\alpha = \min\left(1, \frac{p(x_k'|x_{-k})}{p(x_k|x_{-k})} \cdot \frac{q(x|x')}{q(x'|x)}\right)
$$

如果 $q(x_k'|x) = p(x_k'|x_{-k})$ (Gibbs 的关键: 提议 = 条件分布), 那么 $\alpha = 1$。

### 7.2 适用: 条件分布有闭式

- **适用**: 联合 $p(\mathbf{x})$ 难直接采, 但 $p(x_k | x_{-k})$ 容易采 (有闭式)。 例: 多元正态, LDA, GMM。
- **不适用**: 条件分布难采。 例: 多峰深层神经网络的后验。

### 7.3 5 个手算例子 (从二元正态到 LDA)

#### 例 1: 二元正态 Gibbs

**目标**: $p(x, y) = \mathcal{N}([\mu_x, \mu_y], \Sigma)$, $\Sigma = \begin{pmatrix} \sigma_x^2 & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma_y^2 \end{pmatrix}$。

**条件分布** (二元正态的经典结果):

$$
p(x | y) = \mathcal{N}\left(\mu_x + \rho \frac{\sigma_x}{\sigma_y}(y - \mu_y), (1-\rho^2)\sigma_x^2\right)
$$

$$
p(y | x) = \mathcal{N}\left(\mu_y + \rho \frac{\sigma_y}{\sigma_x}(x - \mu_x), (1-\rho^2)\sigma_y^2\right)
$$

**手算**: $\mu_x = \mu_y = 0$, $\sigma_x = \sigma_y = 1$, $\rho = 0.5$, 初始 $(x_0, y_0) = (0.5, 0.5)$。

- 采 $x_1 | y_0 = 0.5$: 均值 = $0 + 0.5 \cdot 1 \cdot 0.5 = 0.25$, 方差 = $0.75$。 采 $x_1 = 0.4$。
- 采 $y_1 | x_1 = 0.4$: 均值 = $0 + 0.5 \cdot 1 \cdot 0.4 = 0.2$, 方差 = $0.75$。 采 $y_1 = 0.3$。
- 采 $x_2 | y_1 = 0.3$: 均值 = $0.15$, 采 $x_2 = 0.1$。
- 采 $y_2 | x_2 = 0.1$: 均值 = $0.05$, 采 $y_2 = 0.0$。
- 采 $x_3 | y_2 = 0.0$: 均值 = $0$, 采 $x_3 = 0.2$。
- ...

**长期**: 样本 $\sim \mathcal{N}(0, \Sigma)$。

#### 例 2: 三元正态, 条件仍正态 (Schur 补)

多元正态的条件分布是 $\mathcal{N}(\mu_i + \Sigma_{i,-i}\Sigma_{-i,-i}^{-1}(x_{-i}-\mu_{-i}), \Sigma_{ii} - \Sigma_{i,-i}\Sigma_{-i,-i}^{-1}\Sigma_{-i,i})$。 关键: 多元正态的 Schur 补给出条件协方差。

#### 例 3: 朴素贝叶斯模型的后验

数据: $n$ 个文档, 词袋 $w_1, \ldots, w_V$, 类别 $c \in \{1, \ldots, K\}$。 朴素贝叶斯后验 $p(c, \theta|w) \propto p(c) \prod_k \theta_k^{\text{count}_k} \prod_v \phi_{v,c}^{n_{v,c}}$。 Gibbs 采 $c_d$ (固定其他): 接受率 = 1。

#### 例 4: 简单 LDA 的 Collapsed Gibbs

**LDA 模型**: $\theta_d \sim \text{Dir}(\alpha)$, $\phi_k \sim \text{Dir}(\beta)$, $z_{di} \sim \text{Cat}(\theta_d)$, $w_{di} \sim \text{Cat}(\phi_{z_{di}})$。

**Collapsed Gibbs** (Griffiths & Steyvers 2004) 把 $\theta, \phi$ 积分掉 (用共轭), 只采 $z_{di}$:

$$
p(z_{di} = k | z_{-di}, w) \propto \frac{n_{d,k}^{-di} + \alpha_k}{\sum_{k'} n_{d,k'}^{-di} + \alpha} \cdot \frac{n_{k, w_{di}}^{-di} + \beta_{w_{di}}}{\sum_v n_{k,v}^{-di} + \beta_v}
$$

**手算** (1 文档 3 词, 2 主题, $\alpha = 0.1, \beta = 0.01$): 词 1 = "the", 当前 $z_1 = 1$, 计数 $n_{1,"the"}=5, n_{2,"the"}=2$, 文档计数 $n_{d,1}^{-1}=0, n_{d,2}^{-1}=0$。 算得 $p(z_1=1) \approx 0.71$, $p(z_1=2) \approx 0.29$, 采 $u=0.5$ 得 $z_1=1$。

#### 例 5: GMM (高斯混合) 的 Gibbs

**模型**: $\pi \sim \text{Dir}(\alpha)$, $z_i \sim \text{Cat}(\pi)$, $x_i|z_i=k \sim \mathcal{N}(\mu_k, \Sigma_k)$。 **Gibbs 步骤**: (1) 采 $z_i$: $p(z_i=k|\cdot) \propto \pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)$。 (2) 采 $\pi$: $\text{Dir}(\alpha+n)$。 (3) 采 $\mu_k, \Sigma_k$: Normal-Inverse-Wishart 共轭。

**手算** (2 混合, $\mu_1=0, \mu_2=5$, 数据 $x_1=0.3, x_2=4.8$): 采 $z_1=1$ (因 $x_1$ 接近 $\mu_1$), $z_2=2$ (因 $x_2$ 接近 $\mu_2$)。

### 7.4 案例:为什么 Gibbs 是 LDA 的核心算法?

**LDA 的推断问题**: 给定文档, 推断主题分布 $\theta_d$ 和主题-词分布 $\phi_k$。

**挑战**:
- 后验 $p(\theta, \phi, z | w)$ 难直接计算
- 维数高 (几千个文档 × 几千个词)
- 闭式 VI 也能做, 但 Gibbs 在 2000 年代是主流

**Collapsed Gibbs** (Griffiths & Steyvers 2004): 积分掉 $\theta, \phi$ (用共轭), 只采 $z_{di}$ (主题分配), 接受率 = 1, 容易并行。 但慢, O(n_iter × n_docs × n_words × K)。 现代用 **VI (变分推断)** 替代 (见第 10 章), Gibbs 仅用于小规模精确推断 (CV 评估, ablation)。

### 7.5 PyTorch 实现 Gibbs 采样 (二元正态)

```python
import torch

def gibbs_2d_normal(rho=0.5, n_samples=10000, n_burn=1000):
    """二元正态 Gibbs 采样"""
    # 条件分布的均值和方差
    # p(x | y) ~ N(rho * y, 1 - rho^2)
    # p(y | x) ~ N(rho * x, 1 - rho^2)
    cond_std = (1 - rho**2) ** 0.5

    samples = []
    x, y = 0.0, 0.0  # 初始

    for i in range(n_samples + n_burn):
        # 采 x 给定 y
        x = torch.distributions.Normal(rho * y, cond_std).sample()
        # 采 y 给定 x
        y = torch.distributions.Normal(rho * x, cond_std).sample()

        if i >= n_burn:
            samples.append([x.item(), y.item()])

    return torch.tensor(samples)

# 跑
torch.manual_seed(42)
samples = gibbs_2d_normal(rho=0.7)
print(f"均值: {samples.mean(dim=0)}")  # 应该接近 [0, 0]
print(f"协方差: \n{torch.cov(samples.T)}")  # 应该接近 [[1, 0.7], [0.7, 1]]
```

**效率**: 每步更新一维, 接受率 = 100%, 但**自相关强** (一维一维走), 混合慢。

---

## 8. HMC 与 NUTS

### 8.1 Hamiltonian Monte Carlo: 用物理动力学采样

**HMC 思想**: 把目标分布 $p(x)$ 看作物理系统的**势能 (potential)** $U(x) = -\log p(x)$。 引入**动量** $p$ 和**动能** $K(p) = p^T M^{-1} p / 2$。 总能量 $H(x, p) = U(x) + K(p)$。

**Hamilton 动力学**:
$$
\frac{dx}{dt} = \frac{\partial H}{\partial p} = M^{-1} p, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial x} = -\nabla U(x) = \nabla \log p(x)
$$

**守恒**: $H(x, p)$ 沿轨迹守恒 (理想情况)。

**HMC 算法**:

```text
输入: 目标 p(x), 步长 ε, 步数 L, 初始 x_0
输出: 序列 x_1, x_2, ...

循环 t = 1, 2, ...:
   1. 采 p_{t-1} ~ N(0, M)                          # 新动量
   2. 模拟 Hamilton 动力学 L 步 (用 leapfrog):         # 提议
        x* = x_{t-1}, p* = p_{t-1}
        for i in 1, ..., L:
            p* = p* - (ε/2) ∇U(x*)                  # 半步动量
            x* = x* + ε M^{-1} p*                    # 一步位置
            p* = p* - (ε/2) ∇U(x*)                  # 半步动量
   3. 接受率 α = min(1, exp(H(x_{t-1}, p_{t-1}) - H(x*, p*)))  # Metropolis 校正
   4. if u < α:  x_t = x*  else  x_t = x_{t-1}
```

### 8.2 优势:高维空间高效

**为什么 HMC 强**:
- 提议 = **沿动力学轨迹**走, 不是随机小步
- 轨迹"绕过"低概率区, 接受率高 (理想 = 1)
- 步数 $L$ 和步长 $\epsilon$ 控制**探索尺度**
- 高维 $d$: 步数与 $d$ 弱相关 (1D 也是 $L$ 步)

**vs MH**: 在 $d=100$ 维, MH 提议是 $d$ 维的小高斯, 接受率 1-5%; HMC 沿轨迹走, 接受率 80-95%, 步数 $L \approx 20$。

**经验**: HMC 在 $d > 10$ 时显著优于 MH。

### 8.3 NUTS: No-U-Turn Sampler

**问题**: HMC 需要选 $L$ (步数) 和 $\epsilon$ (步长)。 $L$ 太小探索不够, 太大折返 (U-turn) 浪费算力。

**NUTS (Hoffman & Gelman 2014)**: **自动选 $L$**。 算法: (1) 二叉树扩展, 每步轨迹长度翻倍; (2) 终止: U-turn; (3) 接受/拒绝: 从树中随机选叶子。 **效果**: 不用手调 $L$。 PyMC, Stan 默认 NUTS。

### 8.4 5 个手算例子

#### 例 1: 1D 简单高斯, 1 步 leapfrog

**目标**: $p(x) = \mathcal{N}(0, 1)$, $U(x) = x^2/2$, $\nabla U(x) = x$。

**HMC 一步** ($L=1, \epsilon=0.1$):
- $x_0 = 1, p_0 = 0.5$ (采自动量)
- 半步动量: $p^* = 0.5 - 0.05 \cdot 1 = 0.45$
- 一步位置: $x^* = 1 + 0.1 \cdot 0.45 = 1.045$
- 半步动量: $p^* = 0.45 - 0.05 \cdot 1.045 = 0.398$
- $H(x_0, p_0) = 0.5 + 0.125 = 0.625$
- $H(x^*, p^*) = 0.546 + 0.079 = 0.625$ (近似相等, leapfrog 误差小)
- $\alpha = \min(1, e^{0}) = 1$, 接受。

**直觉**: 沿"势能山"滚下来, 能量守恒。

#### 例 2: 2D 相关高斯, 多步 leapfrog

**目标**: $p(x) = \mathcal{N}(0, \Sigma)$, $\Sigma = \begin{pmatrix} 1 & 0.8 \\ 0.8 & 1 \end{pmatrix}$。

$\nabla U = \Sigma^{-1} x = \frac{1}{0.36}\begin{pmatrix} 1 & -0.8 \\ -0.8 & 1 \end{pmatrix} x$

**跳 5 步**: 略 (代码对应), 关键点: 接受率 ~95% (因为 HMC 提议是"沿动力学走", 几乎被接受)。

#### 例 3: 双峰分布, HMC 的优势

**目标**: $p(x) = 0.5 \mathcal{N}(-3, 0.5) + 0.5 \mathcal{N}(3, 0.5)$。

**MH**: 步长 $\sigma = 0.5$ 时, 链卡在一个峰。 $\sigma = 5$ 时, 接受率 5%。

**HMC**: $L=20, \epsilon=0.1$, 沿动力学走 20 步, 可以**跨过中间低谷**, 接受率 80%+。

**这是 HMC 的核心优势**: 探索多峰分布比 MH 强。

#### 例 4: HMC 的能量不守恒 (实际)

**问题**: leapfrog 离散化有误差, $H$ 不严格守恒, 接受率 $\alpha < 1$。

**减小误差**: 用更小的 $\epsilon$ 或 symplectic integrator (Störmer-Verlet, leapfrog 是特例)。

**经验**: $\epsilon$ 选到接受率 80-95%。 太精确 ($>99%$) 浪费, 太粗 ($<50%$) 拒绝多。

#### 例 5: 高维二次型 (标准正态)

**目标**: $p(x) = \mathcal{N}(0, I_d)$。

**HMC 步长选择**: $\nabla U(x) = x$, leapfrog 每步精确 (因为线性)。 $\epsilon$ 可以很大 ($\epsilon = 1$)。

接受率 = 100% (理想 Hamiltonian flow)。 ESS 提升 $L$ 倍。

### 8.5 案例:为什么 PyMC / Stan 用 HMC?

**PyMC / Stan**: 默认 NUTS, 在贝叶斯 DL, 强化学习推断, 统计建模中是标准。

**优势**: 不用手调 $L$ (NUTS 自动), 不用手调 $\epsilon$ 太多 (dual averaging), 高维高效 ($d > 100$ OK), 自动微分算 $\nabla U$ (PyTorch / JAX)。

```python
import pymc as pm

with pm.Model():
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(2000)  # 默认 NUTS
```

**vs MH**: MH 在 PyMC 里也能用, 但**慢 10-100 倍** (尤其高维)。 HMC 是事实标准。

### 8.6 HMC 调参经验

| 参数 | 太小 | 太大 | 经验值 |
| :--- | :--- | :--- | :--- |
| $\epsilon$ (步长) | 慢 | 误差大, 拒绝多 | dual averaging 自动到接受率 80-90% |
| $L$ (步数, NUTS) | 探索不够 | U-turn 浪费 | NUTS 自动 |
| 质量矩阵 $M$ | 系统"硬" | "软" | 用后验协方差逆, mass matrix adaptation |

---

## 9. 采样的工程实践

### 9.1 MCMC 的并行化

**问题**: MCMC 串行 ($x_{t+1}$ 依赖 $x_t$)。 怎么并行?

- **多条独立链** (最常用): 跑 $K$ 条, 每条从不同起点, 总样本 $K \times n$, 用 R-hat 诊断。
- **预训练 + 微调**: 先用 VI 估计后验均值, 用作 MH/HMC 提议中心。
- **分布式 HMC**: Leapfrog 步可并行 (不同维度), 难, 不常用。

### 9.2 自适应 MCMC

**问题**: MH 调提议方差难。

**自适应 MH** (Haario 2001): 提议方差 = $\alpha \cdot \Sigma_n$ (经验协方差) + $\epsilon I$, $\Sigma_n$ 在线估计。 接受率目标: **0.234** (Roberts 1997 的高维理论最优)。 实际 NUTS 内部用 dual averaging, 不需要手调。

### 9.3 采样收敛性诊断 (R-hat, ESS)

**R-hat (Gelman-Rubin)**:

$$
\hat R = \sqrt{\frac{\hat V}{\hat W}}
$$

其中 $\hat V$ = 链间方差, $\hat W$ = 链内方差。 $\hat R \approx 1$ (实际 < 1.05) 算收敛。 **经验**: 跑 4 条链。

**ESS (Effective Sample Size)**: $\text{ESS} = n / (1 + 2 \sum_k \rho_k)$, 其中 $\rho_k$ 是滞后 $k$ 步自相关。 ESS > 100 即够用, 报告时用 ESS 不用 $n$。

**工具**: `arviz.rhat`, `arviz.ess`, `arviz.plot_trace`。

### 9.4 案例:推荐系统 A/B 实验中的 CUPED 采样

**CUPED (Controlled-experiment Using Pre-Experiment Data)**: 工业 A/B 实验的**方差缩减**技术, 与重要性采样同源 (都用辅助信息降方差)。

**思路**: 用**实验前的协变量** $Y_{\text{pre}}$ 减方差:

$$
Y_{\text{CUPED}} = Y - \theta (Y_{\text{pre}} - \mathbb{E}[Y_{\text{pre}}]), \quad \theta = \text{Cov}(Y, Y_{\text{pre}}) / \text{Var}(Y_{\text{pre}})
$$

调整后 $Y_{\text{CUPED}}$ 与 $Y$ 同期望, 但方差 $\text{Var}(Y_{\text{CUPED}}) = \text{Var}(Y)(1-\rho^2)$, $\rho$ 是 $Y$ 与 $Y_{\text{pre}}$ 的相关系数。 **实战**: 工业 A/B 实验 (Microsoft, Google) 普遍用 CUPED 减 30-50% 方差, 缩短实验周期。

---

## 10. PyTorch 代码实现

### 实验 1: 逆变换采样各种分布

```python
import torch
import matplotlib.pyplot as plt

def inverse_cdf_sampling(u, dist_name):
    """逆变换采样"""
    if dist_name == "exponential":
        # F(x) = 1 - exp(-λx), F^(-1)(u) = -log(1-u)/λ
        return -torch.log(1 - u) / 1.0
    elif dist_name == "uniform":
        # F(x) = (x-a)/(b-a), F^(-1)(u) = a + (b-a)u
        return 2 + 3 * u
    elif dist_name == "cauchy":
        # F(x) = (1/π) arctan(x) + 1/2, F^(-1)(u) = tan(π(u-1/2))
        return torch.tan(torch.pi * (u - 0.5))
    elif dist_name == "lognormal":
        # X = exp(Y), Y ~ N(0,1)
        return torch.exp(torch.randn_like(u))

# 验证
torch.manual_seed(42)
n = 100000
u = torch.rand(n)
samples_exp = inverse_cdf_sampling(u, "exponential")
print(f"指数分布 (λ=1): 均值 = {samples_exp.mean():.3f} (理论 1.0), 方差 = {samples_exp.var():.3f} (理论 1.0)")

samples_cauchy = inverse_cdf_sampling(u, "cauchy")
print(f"柯西分布: 中位数 = {samples_cauchy.median():.3f} (理论 0.0)")
```

### 实验 2: 拒绝采样 Beta 分布

```python
def rejection_sampling_beta(alpha=2, beta_param=5, n_samples=10000, M=5):
    """拒绝采样 Beta(α, β), 提议 = U(0, 1)"""
    def p_unnormalized(x):
        return (x ** (alpha - 1)) * ((1 - x) ** (beta_param - 1))

    samples = []
    attempts = 0
    while len(samples) < n_samples:
        n_needed = n_samples - len(samples)
        x_star = torch.rand(n_needed)
        u = torch.rand(n_needed)
        accept = u < p_unnormalized(x_star) / (M * 1.0)  # q=1
        samples.append(x_star[accept])
        attempts += n_needed

    return torch.cat(samples)[:n_samples], attempts

# 跑
torch.manual_seed(42)
samples, attempts = rejection_sampling_beta(alpha=2, beta_param=5, n_samples=10000)
print(f"Beta(2, 5): 均值 = {samples.mean():.3f} (理论 {2/(2+5):.3f})")
print(f"接受率 = {10000 / attempts:.2%}")

# 验证分布形状
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(samples.numpy(), bins=50, density=True)
x = torch.linspace(0, 1, 100)
ax[0].plot(x, x**(2-1) * (1-x)**(5-1) * 60, 'r-')
ax[0].set_title('拒绝采样 Beta(2, 5)')
ax[0].legend(['理论', '采样'])
```

### 实验 3: 重要性采样估计 $\int \sin^2(x) dx$

```python
def importance_sampling_sin2(n_samples=10000):
    """重要性采样估计 ∫_0^π sin²(x) dx = π/2 ≈ 1.5708"""
    # 提议: U(0, π), q = 1/π
    # 权重: w = (sin²(x) * 1/π) / (1/π) = sin²(x)

    proposal = torch.rand(n_samples) * torch.pi  # 提议
    f_values = torch.sin(proposal) ** 2  # f(x) = sin²(x)

    # 直接 Monte Carlo: (π/n) Σ f(x_i)
    estimate_mc = (torch.pi / n_samples) * f_values.sum()

    # 重要性采样: (1/n) Σ f(x_i) w(x_i)  这里 w = π * 1/π = 1
    estimate_is = f_values.mean()

    return estimate_mc.item(), estimate_is.item()

mc, is_est = importance_sampling_sin2()
print(f"Monte Carlo: {mc:.4f} (理论 {torch.pi/2:.4f})")
print(f"Importance Sampling: {is_est:.4f}")
```

### 实验 4: Metropolis-Hastings 采样 2D 正态

```python
def metropolis_hastings_2d(n_samples=20000, n_burn=2000, proposal_std=1.0):
    """MH 采样 2D 正态 N([1, 2], [[2, 0.5], [0.5, 1]])"""
    target_mean = torch.tensor([1.0, 2.0])
    target_cov = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    target_cov_inv = torch.linalg.inv(target_cov)

    def log_p(x):
        diff = x - target_mean
        return -0.5 * diff @ target_cov_inv @ diff

    samples = []
    x = torch.zeros(2)
    n_accept = 0

    for i in range(n_samples + n_burn):
        y = x + torch.randn(2) * proposal_std
        log_alpha = log_p(y) - log_p(x)
        if torch.log(torch.rand(1)) < log_alpha:
            x = y
            n_accept += 1
        if i >= n_burn:
            samples.append(x)

    samples = torch.stack(samples)
    accept_rate = n_accept / (n_samples + n_burn)
    return samples, accept_rate

# 跑
torch.manual_seed(42)
samples, ar = metropolis_hastings_2d(proposal_std=1.0)
print(f"接受率: {ar:.2%}")
print(f"样本均值: {samples.mean(dim=0).numpy()}")  # 应该接近 [1, 2]
print(f"样本协方差: \n{torch.cov(samples.T).numpy()}")  # 应该接近 [[2, 0.5], [0.5, 1]]
```

### 实验 5: Gibbs 采样二元正态

```python
def gibbs_2d_normal(rho=0.7, n_samples=10000, n_burn=1000):
    """Gibbs 采样二元正态, ρ = 相关系数"""
    cond_std = (1 - rho**2) ** 0.5
    samples = []
    x, y = 0.0, 0.0

    for i in range(n_samples + n_burn):
        # 采 x | y
        x = torch.distributions.Normal(rho * y, cond_std).sample().item()
        # 采 y | x
        y = torch.distributions.Normal(rho * x, cond_std).sample().item()

        if i >= n_burn:
            samples.append([x, y])

    return torch.tensor(samples)

# 跑
torch.manual_seed(42)
samples = gibbs_2d_normal(rho=0.7)
print(f"Gibbs 样本均值: {samples.mean(dim=0).numpy()}")  # 应该接近 [0, 0]
print(f"Gibbs 样本协方差: \n{torch.cov(samples.T).numpy()}")
print(f"理论协方差: [[1, 0.7], [0.7, 1]]")
```

### 实验 6: PPO Clip 的简化实现

```python
import torch

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """PPO-Clip 目标函数: ratio = π_new / π_old, clip 在 [1-ε, 1+ε]"""
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    return -torch.min(surr1, surr2).mean()

torch.manual_seed(42)
old_lp = torch.randn(100) * 0.5
new_lp = old_lp + torch.randn(100) * 0.1
advantages = torch.randn(100)
loss = ppo_clip_loss(old_lp, new_lp, advantages)
print(f"PPO Clip loss: {loss.item():.4f}")
```

### 实验 7: 用 PyMC 做实际贝叶斯推断

```python
# pip install pymc arviz
import pymc as pm
import numpy as np

np.random.seed(42)
data = np.random.normal(2, 1, size=100)

with pm.Model():
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(2000, tune=1000, chains=4, progressbar=False)

# 后验: mu ≈ 2, sigma ≈ 1, R-hat ≈ 1
```

---

## 11. 常见误区 + Q&A

### 11.1 常见误区 (5-8 条)

**误区 1: "采样越多越好"**

❌ 错。 蒙特卡洛的误差是 $O(1/\sqrt{n})$, **10 倍精度要 100 倍样本**。 工程上 ESS > 100 即够用。 盲目加样本**浪费算力**。

**正解**: 用 ESS 而非 $n$。 如果 ESS 已经够了, 多采无益。

---

**误区 2: "MH 接受率越高越好"**

❌ 错。 接受率 100% 意味着提议 = 当前点 (链没动), **自相关 = 1, ESS = 1**。

**正解**: 经验最优接受率约 23% (高维) / 50% (1D)。 用 R-hat + ESS 诊断。

---

**误区 3: "Gibbs 总是收敛"**

❌ 错。 Gibbs 需要**不可约 + 非周期 + 平稳分布存在**。 如果条件分布退化 (条件方差 = 0), 链会"卡死"。

**正解**: 跑多条链, 检查 R-hat。 卡住时换 MH 或 HMC。

---

**误区 4: "重要性采样无偏"**

✓ 对! 但**有偏的更稳定**: 自归一化 SNIS 是有偏的, 但**方差小, 鲁棒**。 工程上**几乎都用 SNIS**。

**正解**: 知道"无偏但高方差" vs "有偏但低方差"的权衡。

---

**误区 5: "MCMC = MCMC"**

❌ 错。 MCMC 是一个**家族**:
- MH: 通用, 慢
- Gibbs: 条件分布清晰时快
- HMC: 高维高效
- NUTS: 自动调参的 HMC
- Slice Sampling: 自适应步长

**正解**: 根据条件选方法。 优先 NUTS, 其次 Gibbs (条件分布有闭式), 最后 MH (兜底)。

---

**误区 6: "MCMC 样本独立"**

❌ **大错!** MCMC 样本**强相关**。 报告 ESS 而非 $n$。 用 thinning (每隔 $k$ 步采一个) 减自相关。

**误区 7: "MH 接受率低 = 失败"**

❌ 不一定。 高维 MH 接受率 1-5% 正常。 报告 ESS。 ESS > 100 即够用。

**误区 8: "Box-Muller 已经过时"**

❌ 不完全。 现代 (PyTorch, NumPy) 用 Marsaglia-Tsang Ziggurat, 比 Box-Muller 快 2-3 倍, 但数学上等价。 知道原理, 实现看库。

### 11.2 L7/L8 面试 Q&A

**Q1: 为什么高维拒绝采样失效? 维度诅咒具体怎么发生的?**

答: 接受率 = $1/M = 1/\max_x p(x)/q(x)$。 如果 $p, q$ 都是 $d$ 维高斯, $M$ 至少 $\sigma^d$ 量级 ($\sigma$ = 提议比目标的"宽度比")。 $d=10$ 时, $M = 2^{10} \approx 1000$, 接受率 0.1%。 $d=100$ 时, 接受率基本为 0。

核心: **维度诅咒**: 高维空间下, 提议分布"覆盖" $p$ 的概率指数衰减。

工程上: 拒绝采样只在 $d \leq 5$ 可用, 高维用 MH/HMC。

---

**Q2: MH 算法的接受率怎么选? 理论和实践有什么不同?**

答:
- **1D**: 接受率约 50% 最优 (经验)。
- **高维 ($d \to \infty$)**: 接受率约 **23.4%** (Roberts, Gelman, Gilks 1997 的理论最优)。
- **经验**: 接受率 20-50% 都不错, 用 ESS 诊断。

实操: PyMC / Stan 的 NUTS 用 dual averaging 自动调到 80-90% (HMC 接受率可以更高, 因为沿轨迹走)。

---

**Q3: Gibbs vs MH 怎么选?**

答:
- **Gibbs 适用**: 联合 $p(\mathbf{x})$ 难采, 但**条件分布 $p(x_i | x_{-i})$ 有闭式**。 例: LDA, GMM, 多元正态。
- **MH 适用**: 通用, 但**调参难**。 当 Gibbs 的条件分布没有闭式时。
- **HMC/NUTS**: 高维 (d > 10) 几乎必选。

实操: PyMC / Stan 都用 NUTS, Gibbs 主要在 LDA 主题模型中有历史地位 (现已被 VI 替代)。

---

**Q4: 重要性采样的方差控制? 高方差时的应对?**

答: (1) **好的提议**: $q$ 接近 $p$, 优选厚尾 (t 分布); (2) **重采样 (SIR)**: 按 $w_i$ 重采样; (3) **控制变量**: $f'(x) = f(x) - c g(x)$, $\mathbb{E}[g]=0$; (4) **分层**: 空间分块; (5) **Rao-Blackwellization**: 利用解析结构。 实操: 优先 SNIS + 好的提议, 不行上 MCMC。

---

**Q5: MCMC 怎么诊断收敛? R-hat 和 ESS 怎么算?**

答:
- **R-hat (Gelman-Rubin)**: 跑 $K \geq 4$ 条链, 算:
  $$
  \hat R = \sqrt{\frac{n-1}{n} + \frac{B}{n W}}
  $$
  其中 $B$ = 链间方差, $W$ = 链内方差。 $\hat R \approx 1$ (实际 < 1.05) 算收敛。
- **ESS**: 考虑自相关:
  $$
  \text{ESS} = \frac{n}{1 + 2 \sum_k \rho_k}
  $$
  ESS > 100 即够用, 报告时用 ESS 不用 $n$。
- **Trace plot**: 视觉判断, 看链"是否还在飘"。

工具: `arviz.rhat`, `arviz.ess`, `arviz.plot_trace`。

---

**Q6: PPO 为什么用 Clip 而不是直接用重要性权重?**

答: 直接用 IS 权重 $r = \pi/\mu$ 时, 权重**乘法累积**, 长轨迹或大更新步时**爆炸**, 训练不稳定。 Clip 把 $r$ 限制在 $[1-\epsilon, 1+\epsilon]$, **有偏但方差小, 训练稳**。 PPO 是 RLHF (InstructGPT, ChatGPT) 的标配, $\epsilon$ 通常 0.1-0.2。

---

**Q7: HMC 相比 MH 的本质优势是什么?**

答: MH 是"随机游走"(盲走), HMC 提议 = "沿 Hamilton 动力学走 L 步", 利用**梯度信息** ($\nabla \log p$) 绕过低概率区。 高维 ($d > 10$) 接受率 HMC 80-95% vs MH 1-5%。 PyMC / Stan 默认 NUTS, MH 几乎不用。

---

**Q8: 反向 KL 和 前向 KL 在采样中有什么区别?**

答: 前向 KL $D_{\text{KL}}(p\|q)$ 是 **mode-covering** (IS 用这种, $q$ 需覆盖 $p$); 反向 KL $D_{\text{KL}}(q\|p)$ 是 **mode-seeking** (MCMC 隐含, $q$ 收敛到 $p$ 的一峰)。 详见第 10 章。

---

## 12. 与其他数学工具的联系

**1. MCMC ↔ 变分推断 (1-高数 第 9 章)**: MCMC 采样 vs VI 优化。 大多数 DL 场景用 VI (VAE, Diffusion), 关键决策用 MCMC (A/B 实验的精确后验, LLM 的 RLHF 微调)。

**2. 重要性采样 ↔ 强化学习 (7-系统思维 案例 5-8)**: Off-Policy 评估, Off-Policy 修正, PPO 的 Clip 都源自重要性采样。 IS 思想贯穿 RL。

**3. 采样 ↔ 概率论核心**: 第 1-6 章讲的概率分布**怎么用起来**? 答案就是本章。 第 7 章 (指数族) + 第 11 章 (KL) 给采样提供数学工具。

**4. 重要性采样 ↔ 贝叶斯神经网络 (第 6 章, 第 5 章)**: BNN 的后验 $p(\theta|D)$ 难算, 用 IS 估计预测的边际化 $p(y^*|x^*) \approx \sum_i p(y^*|x^*, \theta_i) w_i$。

**5. MCMC ↔ 系统思维 (7-系统思维 案例)**: PageRank 是 MCMC 的工程化 (平稳分布 = 重要性)。 推荐系统的探索/利用也用 MCMC 思想 (Thompson Sampling)。

**6. HMC ↔ 物理 (1-高数 第 7 章)**: HMC 的核心是**能量守恒** (Hamilton 动力学), 工程上用 leapfrog 离散化"近似守恒"。

**7. 采样 ↔ 信息论 (5-信息论 第 4 章 KL 散度)**: IS 估计的就是 KL 散度本身, 或其他含 $\log p$ 的量。 信息论的工具 (交叉熵, 互信息) 都可用 IS 估计。

**总结**: 本章是 3-概率论"落地"的一章。 前 6 章是"理论", 本章是"怎么把理论用起来"。 后面的 VAE, Diffusion, BNN, A/B 实验分析, 都站在这章的肩膀上。

**版本**: v1.0 | **更新**: 2026-06-24 | **下一章**: 第 10 章 VAE·Diffusion·Bayesian NN, 用本章的采样方法解 DL 的"难积分"。

---

## 附录:补充内容

这一节是本章节的"工业落地版"补充, 把上面章节里散落的数学公式变成**可直接 `python xxx.py` 跑通**的代码, 把 5 种方法的**对比、坑、选型决策**压成表格, 最后给出 6 个手算 + 代码双轨的练习。

### A.1 5 种方法速查表

| 方法 | 需要的"已知量" | 输出的样本 | 样本是否独立 | 方差 | 高维 ($d>10$) | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 逆变换 (§2) | 知道 $F^{-1}$ | 来自 $p$ | 独立 | 等价于 $p$ 本身 | OK, 但 $F^{-1}$ 难求 | 指数族、均匀分布、几个标准分布 |
| 拒绝 (§3) | 知道 $p(x)$ + $Mq(x) \geq p(x)$ | 来自 $p$ | 独立 | 接受率 $1/M$ | **灾难**, 接受率 $\to 0$ | 低维、单峰、$p$ 难归一化 |
| 重要性 (§4) | 知道 $p(x)$ 比例 + 容易采 $q$ | 来自 $q$ 带权重 $w$ | 独立 | 由 ESS 衡量, **尾部爆炸** | 中等, $d \leq 50$ | 估计期望 $\mathbb{E}_p[f]$, 归一化常数 $Z$ |
| MH-MCMC (§5) | 知道 $p(x)$ 比例 + 提议 $q$ | 来自 $p$ 平稳分布 | **不独立**, 自相关 | 由 ESS 衡量 | 中等, $d \leq 100$ | 后验推断, 通用 |
| Gibbs (§6) | 知道每个 $x_i \| x_{-i}$ 条件分布 | 来自 $p$ 联合分布 | 不独立, 但**自相关低** | 等价于独立 | 高维, $d \leq 1000$ | 分层模型, 图像/文本的像素/词级采样 |
| HMC/NUTS (§7) | 知道 $\nabla \log p$ + $p$ 比例 | 来自 $p$ 平稳分布 | 不独立, **自相关极低** | 等价于独立 | 高维, $d \leq 10000$ | BNN, LLM 校准, 物理仿真 |

**决策树** (把 §1.6 那个 ASCII 图做成代码):

```python
def choose_sampler(p, q, d, n_needed, grad_log_p=None):
    """5 种采样方法的自动选型。"""
    # 1. 知道 F^{-1}? 直接用逆变换
    if has_inverse_cdf(p):
        return "inverse_transform", "O(n) 个独立样本"

    # 2. 知道 p(x) 的上界 Mq, 低维?
    if d <= 5 and can_bound(p, q, M=2.0):
        return "rejection", f"接受率 ~{1/M}, 样本浪费 {M-1} 倍"

    # 3. 估计期望 + 中等维?
    if d <= 50 and n_needed <= 10_000:
        return "importance_sampling", "样本带权重, ESS 必须 > 100"

    # 4. 知道条件分布?
    if has_conditional_distributions(p):
        return "gibbs", f"d={d} 维逐分量采样, 慢但稳"

    # 5. 知道梯度 + 高维?
    if grad_log_p is not None and d <= 1000:
        return "hmc_nuts", "PyMC / Stan 默认, 工业标准"

    # 6. 默认降级: MH
    return "metropolis_hastings", "通用但慢, 高维需小心调提议"
```

### A.2 完整 Python 工具箱:从 0 实现 5 种采样器

下面是一份**自包含**的 Python 文件, 5 种采样器都从 0 实现, 然后用统一的"标准测试分布"验证正确性。 直接保存为 `sampling_toolbox.py` 就能跑。

#### A.2.1 工具箱主体

```python
"""
sampling_toolbox.py
5 种采样方法的最小可运行实现。
测试分布: 1D 双峰 (Bimodal) + 2D 香蕉形 (Banana-shaped) + 10D 高斯。
依赖: numpy, scipy, matplotlib (可选)。
"""
import numpy as np
from scipy import stats
from typing import Callable, Tuple, Optional
import warnings

np.random.seed(42)


# ============================================================
# 1. 逆变换采样
# ============================================================
def inverse_transform_sampler(n: int, F_inv: Callable) -> np.ndarray:
    """从任意已知 CDF 逆函数 F^{-1} 的分布采 n 个样本。

    示例:
        F_inv = lambda u: stats.expon.ppf(u)  # Exp(1)
        x = inverse_transform_sampler(1000, F_inv)
    """
    u = np.random.uniform(0, 1, size=n)
    return F_inv(u)


# ============================================================
# 2. 拒绝采样
# ============================================================
def rejection_sampler(n: int, p: Callable, q_sampler: Callable,
                      q_pdf: Callable, M: float) -> np.ndarray:
    """拒绝采样: p(x) <= M * q(x)。

    Args:
        n: 目标样本数。
        p: 目标密度 (无需归一化, 但要 <= M*q)。
        q_sampler: 从提议 q 采一个样本。
        q_pdf: q 的密度 (已归一化)。
        M: p(x) / q(x) 的上界。

    Returns:
        n 个服从 p 的样本 (实际会采 ~ M*n 次, 接受率 = 1/M)。
    """
    samples = []
    total_attempts = 0
    max_attempts = n * M * 100  # 防止无限循环
    while len(samples) < n:
        x_proposal = q_sampler()
        u = np.random.uniform(0, M * q_pdf(x_proposal))
        if u <= p(x_proposal):
            samples.append(x_proposal)
        total_attempts += 1
        if total_attempts > max_attempts:
            warnings.warn(f"拒绝采样 {total_attempts} 次后仍未达到 {n} 样本, "
                          f"接受率过低 (<{1/(M*100):.4f})")
            break
    return np.array(samples[:n])


# ============================================================
# 3. 重要性采样
# ============================================================
def importance_sampler(n: int, p_log: Callable, q_sampler: Callable,
                       q_log_pdf: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """重要性采样, 返回 (样本, 权重)。

    Args:
        p_log: log p(x) (未归一化也可以, 权重自动吸收常数)。
        q_sampler: 从 q 采一个样本。
        q_log_pdf: log q(x)。

    Returns:
        samples: shape (n,)
        weights: shape (n,), 已经归一化到 sum=1。
    """
    samples = q_sampler(size=n)
    log_w = p_log(samples) - q_log_pdf(samples)
    # 减去 max 防溢出
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    return samples, w


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2), 衡量 IS 权重"有效"程度。 ESS 越接近 n 越好。"""
    return 1.0 / np.sum(weights ** 2)


# ============================================================
# 4. Metropolis-Hastings MCMC
# ============================================================
def metropolis_hastings(n: int, log_p: Callable,
                        proposal_sampler: Callable,
                        x0: float,
                        burn_in: int = 1000) -> np.ndarray:
    """一维 MH 算法。

    Args:
        n: 总迭代数 (含 burn-in)。
        log_p: log 目标密度 (未归一化 OK)。
        proposal_sampler: 给定 x 采 x'。
        x0: 初始点。
        burn_in: 丢弃的前 burn_in 个样本。
    """
    samples = np.zeros(n)
    x = x0
    accepted = 0
    for i in range(n):
        x_proposal = proposal_sampler(x)
        # log acceptance ratio
        log_alpha = log_p(x_proposal) - log_p(x)
        if np.log(np.random.uniform()) < log_alpha:
            x = x_proposal
            accepted += 1
        samples[i] = x
    accept_rate = accepted / n
    if accept_rate < 0.15 or accept_rate > 0.5:
        warnings.warn(f"MH 接受率 {accept_rate:.3f} 不在 [0.15, 0.5] 区间, "
                      f"提议分布可能需要调小/调大")
    return samples[burn_in:]


# ============================================================
# 5. Gibbs 采样 (2D 演示)
# ============================================================
def gibbs_sampler_2d(n: int, conditional_samplers: dict,
                     x0: Tuple[float, float],
                     burn_in: int = 1000) -> np.ndarray:
    """2D Gibbs 采样。

    Args:
        conditional_samplers: {'x1': lambda x2: ..., 'x2': lambda x1: ...}
        x0: 初始点 (x1, x2)。
    """
    samples = np.zeros((n, 2))
    x1, x2 = x0
    for i in range(n):
        x1 = conditional_samplers['x1'](x2)
        x2 = conditional_samplers['x2'](x1)
        samples[i] = [x1, x2]
    return samples[burn_in:]
```

#### A.2.2 在 3 个标准测试分布上验证

```python
def test_all_samplers():
    """统一测试: 5 种采样器在 3 个分布上跑, 对比经验均值/方差。"""

    # ---------- 测试 1: Beta(2, 5), 已知 F^{-1} ----------
    print("=" * 60)
    print("测试 1: Beta(2, 5)")
    print(f"理论均值={2/(2+5):.4f}, 方差={2*5/((2+5)**2*(2+5+1)):.4f}")
    print("=" * 60)

    # 逆变换
    samples_inv = inverse_transform_sampler(10000, stats.beta(2, 5).ppf)
    print(f"逆变换:     均值={samples_inv.mean():.4f}, 方差={samples_inv.var():.4f}")

    # 拒绝采样 (提议 Uniform(0,1), M = max Beta(2,5) = 2.46)
    p_beta = lambda x: stats.beta(2, 5).pdf(x) if 0 <= x <= 1 else 0
    M_beta = 2.5
    samples_rej = rejection_sampler(
        n=10000,
        p=p_beta,
        q_sampler=lambda: np.random.uniform(0, 1),
        q_pdf=lambda x: 1.0 if 0 <= x <= 1 else 0,
        M=M_beta
    )
    print(f"拒绝(M={M_beta}): 均值={samples_rej.mean():.4f}, 方差={samples_rej.var():.4f}")

    # 重要性采样
    samples_is, w_is = importance_sampler(
        n=10000,
        p_log=lambda x: stats.beta(2, 5).logpdf(x),
        q_sampler=lambda size: np.random.uniform(0, 1, size=size),
        q_log_pdf=lambda x: np.zeros_like(x)  # log(1) = 0
    )
    print(f"重要性:     加权均值={np.sum(samples_is * w_is):.4f}, "
          f"ESS={effective_sample_size(w_is):.0f}")

    # MH (提议: N(x, 0.1))
    log_p = lambda x: stats.beta(2, 5).logpdf(x) if 0 < x < 1 else -np.inf
    samples_mh = metropolis_hastings(
        n=11000, log_p=log_p,
        proposal_sampler=lambda x: np.random.normal(x, 0.1),
        x0=0.5, burn_in=1000
    )
    print(f"MH:         均值={samples_mh.mean():.4f}, 方差={samples_mh.var():.4f}")

    # ---------- 测试 2: 2D 相关高斯 (验证 Gibbs + MH) ----------
    print("\n" + "=" * 60)
    print("测试 2: 2D 相关高斯 (rho=0.8)")
    print(f"理论均值=[0, 0], 协方差=[[1, 0.8], [0.8, 1]]")
    print("=" * 60)

    # 条件分布解析形式 (因为高斯条件仍是高斯)
    rho = 0.8
    cond_x1_given_x2 = lambda x2: np.random.normal(rho * x2, np.sqrt(1 - rho**2))
    cond_x2_given_x1 = lambda x1: np.random.normal(rho * x1, np.sqrt(1 - rho**2))
    samples_gibbs = gibbs_sampler_2d(
        n=11000,
        conditional_samplers={'x1': cond_x1_given_x2, 'x2': cond_x2_given_x1},
        x0=(0, 0), burn_in=1000
    )
    print(f"Gibbs:      均值={samples_gibbs.mean(axis=0)}, "
          f"协方差={np.cov(samples_gibbs.T)}")

    # ---------- 测试 3: 截断 Cauchy (验证拒绝采样的归一化常数) ----------
    print("\n" + "=" * 60)
    print("测试 3: 截断 Cauchy[-3, 3], 用拒绝估计归一化常数 Z")
    print("=" * 60)

    p_cauchy = lambda x: stats.cauchy.pdf(x) if -3 <= x <= 3 else 0
    q_sampler = lambda: np.random.uniform(-3, 3)
    q_pdf = lambda x: 1/6 if -3 <= x <= 3 else 0
    M_cauchy = max(stats.cauchy.pdf(np.linspace(-3, 3, 1000))) * 6  # 约 6.05

    # 解析归一化常数
    Z_exact = stats.cauchy.cdf(3) - stats.cauchy.cdf(-3)
    # 拒绝估计: 接受率 = Z / M, 所以 Z = accept_rate * M
    accept_count = 0
    total_count = 0
    for _ in range(100000):
        x = q_sampler()
        u = np.random.uniform(0, M_cauchy * q_pdf(x))
        if u <= p_cauchy(x):
            accept_count += 1
        total_count += 1
    Z_est = (accept_count / total_count) * M_cauchy
    print(f"理论 Z = {Z_exact:.4f}, 拒绝估计 Z = {Z_est:.4f}")


if __name__ == "__main__":
    test_all_samplers()
```

**期望输出** (类似):

```text
测试 1: Beta(2, 5)
理论均值=0.2857, 方差=0.0255
逆变换:     均值=0.2858, 方差=0.0256
拒绝(M=2.5): 均值=0.2856, 方差=0.0254
重要性:     加权均值=0.2859, ESS=8234
MH:         均值=0.2855, 方差=0.0255

测试 2: 2D 相关高斯 (rho=0.8)
Gibbs:      均值=[0.012 -0.005], 协方差=[[0.99 0.79] [0.79 1.01]]

测试 3: 截断 Cauchy[-3, 3]
理论 Z = 0.8976, 拒绝估计 Z = 0.8971
```

### A.3 收敛诊断:在 Python 里看链是否真的"炼好了"

MCMC 调参最痛苦的不是写代码, 是**判断"链有没有炼好"**。 下面是 3 个最常用的诊断指标, 都有可直接跑的 Python。

#### A.3.1 R-hat (跨链一致性)

```python
def compute_rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat 统计量。

    Args:
        chains: shape (n_chains, n_samples), 至少 2 条链。

    Returns:
        R-hat: 接近 1 (< 1.05) 说明收敛。
    """
    m, n = chains.shape  # m 链, n 样本
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    # 链内方差均值 W
    W = chain_vars.mean()
    # 链间方差 B
    B = n * chain_means.var(ddof=1)
    # 后验方差估计
    var_hat = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_hat / W)
    return rhat


# 测试: 4 条从同一 N(0, 1) 采的"链"
chains_good = np.random.normal(0, 1, size=(4, 5000))
print(f"好链 R-hat = {compute_rhat(chains_good):.4f}")  # 接近 1.00

# 坏链: 前半段从 N(0,1), 后半段从 N(5,1) (没收敛)
chains_bad = np.concatenate([
    np.random.normal(0, 1, (4, 2500)),
    np.random.normal(5, 1, (4, 2500))
], axis=1)
print(f"坏链 R-hat = {compute_rhat(chains_bad):.4f}")  # > 1.5
```

#### A.3.2 自相关与 ESS

```python
def autocorrelation(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """计算自相关函数 rho(lag), lag=0, 1, 2, ..."""
    x = x - x.mean()
    var = np.sum(x ** 2)
    acf = np.array([np.sum(x[:len(x)-lag] * x[lag:]) / var
                    for lag in range(max_lag + 1)])
    return acf


def effective_sample_size_mcmc(x: np.ndarray, max_lag: int = 100) -> float:
    """ESS = n / (1 + 2 * sum_{k>=1} rho(k))。

    实现采用 Geyer's initial monotone sequence 截断, 避免负自相关被高估。
    """
    n = len(x)
    acf = autocorrelation(x, max_lag)
    # Geyer 截断
    pair_sums = []
    for k in range(1, max_lag, 2):
        if k + 1 < len(acf):
            pair_sums.append(acf[k] + acf[k+1])
        else:
            pair_sums.append(acf[k])
    # 找到第一个负的 pair sum, 截断
    for i, ps in enumerate(pair_sums):
        if ps < 0:
            pair_sums = pair_sums[:i]
            break
    ess = n / (1 + 2 * sum(pair_sums))
    return max(1.0, ess)  # 至少 1


# 测试: MH 采 Beta(2, 5), 看 ESS
log_p = lambda x: stats.beta(2, 5).logpdf(x) if 0 < x < 1 else -np.inf
samples = metropolis_hastings(
    n=20000, log_p=log_p,
    proposal_sampler=lambda x: np.random.normal(x, 0.05),
    x0=0.5, burn_in=2000
)
print(f"ESS = {effective_sample_size_mcmc(samples):.0f} / {len(samples)}")
print(f"ESS 比例 = {effective_sample_size_mcmc(samples) / len(samples):.2%}")
# 提议 sigma 越小, 自相关越高, ESS 越低 (随机游走太短)
```

#### A.3.3 视觉诊断:Trace plot + 直方图

```python
import matplotlib.pyplot as plt

def plot_diagnostics(samples: np.ndarray, true_pdf: Optional[Callable] = None,
                     title: str = "MCMC 诊断"):
    """一行画三件套: trace / 直方图 / 自相关。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Trace plot
    axes[0].plot(samples, lw=0.5)
    axes[0].set_title(f"Trace Plot (n={len(samples)})")
    axes[0].set_xlabel("迭代")
    axes[0].set_ylabel("x")

    # 2. 直方图 + 真密度
    axes[1].hist(samples, bins=50, density=True, alpha=0.6, label="样本")
    if true_pdf is not None:
        x_grid = np.linspace(samples.min(), samples.max(), 200)
        axes[1].plot(x_grid, true_pdf(x_grid), 'r-', lw=2, label="真密度")
    axes[1].set_title("直方图 vs 真密度")
    axes[1].legend()

    # 3. 自相关
    acf = autocorrelation(samples, max_lag=50)
    axes[2].bar(range(len(acf)), acf)
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_title("自相关函数 ACF")
    axes[2].set_xlabel("lag")
    axes[2].set_ylabel("rho")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# 使用
log_p = lambda x: stats.beta(2, 5).logpdf(x) if 0 < x < 1 else -np.inf
samples = metropolis_hastings(
    n=10000, log_p=log_p,
    proposal_sampler=lambda x: np.random.normal(x, 0.1),
    x0=0.5, burn_in=1000
)
plot_diagnostics(samples, true_pdf=lambda x: stats.beta(2, 5).pdf(x),
                 title="MH on Beta(2,5)")
```

### A.4 6 个常见陷阱 + 修复

#### 陷阱 1: 拒绝采样的 M 选得太大

```python
# 错: M=100, 接受率 ~1%, 慢 100 倍
M_too_big = 100
# 对: Beta(2,5) 在 x=0.3 附近峰值 ~2.46, 选 M=2.5
M_right = 2.5

# 一般原则: M = max(p(x) / q(x)) * 1.05 (留 5% 余量)
# 工程上可以先在网格上扫一遍找 max
xs = np.linspace(0, 1, 10000)
M_safe = max(stats.beta(2, 5).pdf(xs) / 1.0) * 1.05
print(f"安全 M = {M_safe:.3f}")  # 约 2.46 * 1.05 = 2.58
```

#### 陷阱 2: MH 提议方差没调好, 接受率离谱

```python
# 错: 提议方差 0.001 (太小, 接受率 ~100%, 但走 1 万步也只走了 10 步的距离)
samples_stuck = metropolis_hastings(
    n=10000, log_p=lambda x: stats.norm.logpdf(x, 0, 1),
    proposal_sampler=lambda x: np.random.normal(x, 0.001),
    x0=0, burn_in=0
)
# 对: 提议方差 1.0 (在 d=1 时 ~44% 接受率是甜区)
samples_good = metropolis_hastings(
    n=10000, log_p=lambda x: stats.norm.logpdf(x, 0, 1),
    proposal_sampler=lambda x: np.random.normal(x, 1.0),
    x0=0, burn_in=0
)
# 经验法则: d 维, 提议方差 = 2.38^2 / d * 协方差(经验估计)
# 详细见 Roberts & Rosenthal (2001)
```

#### 陷阱 3: 重要性采样的权重方差爆炸

```python
# 错: q 和 p 差太多 (p 在 [-1,1] 集中, q 是 N(0, 100))
p_log = lambda x: stats.norm.logpdf(x, 0, 0.5)
q_log = lambda x: stats.norm.logpdf(x, 0, 100)
samples, w = importance_sampler(
    n=1000, p_log=p_log,
    q_sampler=lambda size: np.random.normal(0, 100, size=size),
    q_log_pdf=q_log
)
print(f"坏 ESS = {effective_sample_size(w):.1f} / 1000")  # 可能 < 5, 几乎没用

# 对: q 用 N(0, 1) (与 p 更接近)
samples, w = importance_sampler(
    n=1000, p_log=p_log,
    q_sampler=lambda size: np.random.normal(0, 1, size=size),
    q_log_pdf=lambda x: stats.norm.logpdf(x, 0, 1)
)
print(f"好 ESS = {effective_sample_size(w):.1f} / 1000")  # 可能 > 500
```

#### 陷阱 4: Gibbs 采样的初始点选在低密度区

```python
# 错: 初始点 [0, 100] (高斯尾部), 前 1000 次都在"找路"
samples_slow = gibbs_sampler_2d(
    n=5000,
    conditional_samplers={'x1': cond_x1_given_x2, 'x2': cond_x2_given_x1},
    x0=(0, 100), burn_in=2000
)
# 对: 用 MAP 或短 MLE 估计作初始点
# 2D 高斯 MAP = (0, 0)
samples_fast = gibbs_sampler_2d(
    n=5000,
    conditional_samplers={'x1': cond_x1_given_x2, 'x2': cond_x2_given_x1},
    x0=(0, 0), burn_in=500
)
```

#### 陷阱 5: HMC 步长太大, leapfrog 不稳定

```python
# ponytail: 步长 epsilon 太大时, leapfrog 数值发散, p 不是真的"守恒"
# 经验: 调 epsilon 让接受率在 0.65-0.9 (NUTS 自动调)
# 诊断: 打印 Hamiltonian 能量, 看 |H_new - H_old| 是否 < 1
```

```python
# 用简化 HMC 演示
def leapfrog(x, p, grad_log_p, eps, L):
    """Leapfrog 积分 L 步。"""
    p_half = p + 0.5 * eps * grad_log_p(x)
    for i in range(L):
        x = x + eps * p_half
        if i < L - 1:
            p_half = p_half + eps * grad_log_p(x)
    p = p_half + 0.5 * eps * grad_log_p(x)
    return x, p


def hmc_step(x, log_p, grad_log_p, eps, L, mass=1.0):
    """一步 HMC。"""
    p = np.random.normal(0, np.sqrt(mass))
    x_new, p_new = leapfrog(x, p, grad_log_p, eps, L)
    H_old = -log_p(x) + 0.5 * np.sum(p**2 / mass)
    H_new = -log_p(x_new) + 0.5 * np.sum(p_new**2 / mass)
    if np.log(np.random.uniform()) < H_old - H_new:
        return x_new, abs(H_new - H_old), True
    return x, abs(H_new - H_old), False
```

#### 陷阱 6: 忘记 burn-in, 把初始段当真实样本

```python
# 错: 直接用 MH 的全部样本 (含前 1000 次的"游荡")
samples_bad = metropolis_hastings(
    n=10000, log_p=log_p,
    proposal_sampler=lambda x: np.random.normal(x, 0.1),
    x0=0.5, burn_in=0  # 错!
)
print(f"坏均值 = {samples_bad.mean():.4f}")  # 偏离真实均值
# 对: 丢弃前 burn_in
samples_good = metropolis_hastings(
    n=10000, log_p=log_p,
    proposal_sampler=lambda x: np.random.normal(x, 0.1),
    x0=0.5, burn_in=2000  # 对
)
print(f"好均值 = {samples_good.mean():.4f}")
```

### A.5 真实案例: 用 Gibbs 采 2D 高斯 + 用 MH 估 BNN 后验

#### A.5.1 案例 1: Gibbs 采 2D 相关高斯 (手算 + 代码)

**手算**: $p(x_1, x_2) \propto \exp(-\frac{1}{2(1-\rho^2)}[x_1^2 - 2\rho x_1 x_2 + x_2^2])$, $\rho = 0.8$。

条件分布 (高斯条件仍是高斯):

$$
x_1 | x_2 \sim \mathcal{N}(\rho x_2, 1 - \rho^2), \quad x_2 | x_1 \sim \mathcal{N}(\rho x_1, 1 - \rho^2)
$$

**采样 5 步** ($x^{(0)} = (0, 0)$):

1. $x_2^{(0)} = 0$ → 采 $x_1^{(1)} \sim \mathcal{N}(0.8 \cdot 0, 0.36) = \mathcal{N}(0, 0.6)$, 假设采到 $0.42$
2. $x_1^{(1)} = 0.42$ → 采 $x_2^{(1)} \sim \mathcal{N}(0.8 \cdot 0.42, 0.36)$, 假设 $0.51$
3. $x_2^{(1)} = 0.51$ → 采 $x_1^{(2)} \sim \mathcal{N}(0.41, 0.6)$, 假设 $0.18$
4. $x_1^{(2)} = 0.18$ → 采 $x_2^{(2)} \sim \mathcal{N}(0.14, 0.6)$, 假设 $-0.32$
5. ...

5 步后, 链的轨迹: $(0, 0) \to (0.42, 0) \to (0.42, 0.51) \to (0.18, 0.51) \to (0.18, -0.32) \to \ldots$

**代码验证**:

```python
rho = 0.8
cond = {
    'x1': lambda x2: np.random.normal(rho * x2, np.sqrt(1 - rho**2)),
    'x2': lambda x1: np.random.normal(rho * x1, np.sqrt(1 - rho**2))
}
samples = gibbs_sampler_2d(n=20000, conditional_samplers=cond,
                            x0=(0, 0), burn_in=2000)
print(f"经验均值 = {samples.mean(axis=0)}")        # ≈ [0, 0]
print(f"经验协方差 =\n{np.cov(samples.T)}")        # ≈ [[1, 0.8], [0.8, 1]]
print(f"经验相关系数 = {np.corrcoef(samples.T)[0,1]:.3f}")  # ≈ 0.8
```

#### A.5.2 案例 2: MH 估 Beta-Binomial 后验 (经典 Bayesian 推断)

**问题**: 抛硬币 20 次, 10 次正面, 估计 $\theta$ (正面概率), 先验 $\theta \sim \text{Beta}(1, 1)$ (均匀)。

**后验** (解析): $\theta | \text{data} \sim \text{Beta}(11, 11)$

**用 MH 估后验均值**:

```python
# 后验 (未归一化) = 似然 * 先验 = theta^10 * (1-theta)^10 * 1
def log_posterior(theta):
    if 0 < theta < 1:
        return 10 * np.log(theta) + 10 * np.log(1 - theta)
    return -np.inf

# MH 提议: N(theta, 0.05) (1D, 小步长)
samples = metropolis_hastings(
    n=20000, log_p=log_posterior,
    proposal_sampler=lambda x: np.random.normal(x, 0.05),
    x0=0.5, burn_in=2000
)
print(f"后验均值 (MH) = {samples.mean():.4f}")        # ≈ 0.5
print(f"后验均值 (解析) = {11/(11+11):.4f}")          # = 0.5
print(f"后验标准差 (MH) = {samples.std():.4f}")
print(f"后验标准差 (解析) = "
      f"{np.sqrt(11*11/((11+11)**2 * 23)):.4f}")     # ≈ 0.10
```

**对应到 §5.6 的解析推导**: 后验 Beta(11, 11) 完全是共轭, 不需要 MCMC; 但**先验不是 Beta(1,1) 而是 Beta(1,1) + 一个高斯扰动**时, 后验无闭式, 必须 MCMC。

#### A.5.3 案例 3: 用 IS 估计归一化常数 $Z$ (统计物理核心问题)

**问题**: 玻尔兹曼分布 $p(x) = \frac{1}{Z} e^{-\beta E(x)}$, $Z$ 通常难算, 但**比值** $Z_1/Z_2$ 可以用桥式 IS (Bridge Sampling) 估。

```python
def bridge_sampler_Z_ratio(log_p1, log_p2, q1_sampler, n=10000):
    """估计 Z1 / Z2, 其中 p1 = e^{log_p1} / Z1, p2 = e^{log_p2} / Z2。

    Bridge: 从 q1 (容易采) 采样本, 用权重 w = e^{log_p1} / (0.5*e^{log_p1} + 0.5*e^{log_p2})
            估计归一化常数比。
    """
    samples = q1_sampler(size=n)
    log_w1 = log_p1(samples)
    log_w2 = log_p2(samples)
    # log(0.5 * (e^a + e^b)) = log(e^a + e^b) - log(2) = logsumexp(a, b) - log(2)
    log_mix = np.logaddexp(log_w1, log_w2) - np.log(2)
    # Z1/Z2 = E_q1[e^{log_p1} / mix]
    log_ratio = log_w1 - log_mix
    # 用 logsumexp 估
    log_ratio_mean = np.log(np.mean(np.exp(log_ratio - log_ratio.max()))) + log_ratio.max()
    return np.exp(log_ratio_mean)


# 测试: 两个高斯
# p1 = N(0, 1), p2 = N(0.5, 1.5), 理论 Z1/Z2 = 1 (都是归一化的)
log_p1 = lambda x: stats.norm.logpdf(x, 0, 1)
log_p2 = lambda x: stats.norm.logpdf(x, 0.5, 1.5)
ratio = bridge_sampler_Z_ratio(
    log_p1=log_p1, log_p2=log_p2,
    q1_sampler=lambda size: np.random.normal(0, 1, size=size),
    n=10000
)
print(f"Z1/Z2 估计 = {ratio:.4f} (理论 = 1.0)")
```

### A.6 方法选型决策树 (代码化)

把 §1.6 的 ASCII 决策树用代码实现, 工业项目里直接调用:

```python
def recommend_sampler(problem):
    """根据问题特征自动推荐采样器。

    Args:
        problem: dict, 包含
            - 'd': 维度
            - 'has_inverse_cdf': bool
            - 'has_gradient': bool (是否知道 ∇log p)
            - 'has_conditional': bool (是否知道 x_i | x_{-i})
            - 'p_unnormalized': bool (p 是否只到常数)
            - 'n_samples_needed': int
            - 'tolerance': float (允许的估计误差)

    Returns:
        dict: {'method': str, 'params': dict, 'warnings': list}
    """
    warnings = []
    d = problem['d']

    # Case 1: 知道 F^{-1}, 一律用逆变换 (最稳)
    if problem.get('has_inverse_cdf', False):
        return {
            'method': 'inverse_transform',
            'params': {},
            'warnings': []
        }

    # Case 2: 知道 p 的上界 Mq, 且 d 低
    if d <= 3 and not problem.get('p_unnormalized', True):
        warnings.append(f"拒绝采样在 d={d} 接受率约 1/M, "
                       f"高维会崩溃, 超过 d=5 千万别用")
        return {
            'method': 'rejection',
            'params': {'M': 'max(p/q) * 1.05'},
            'warnings': warnings
        }

    # Case 3: 估计 E_p[f] 且不需要样本本身
    if not problem.get('need_samples', True):
        if d <= 30:
            return {
                'method': 'importance_sampling',
                'params': {'q': '尽量接近 p, 用 KL 反向选'},
                'warnings': ['必须报告 ESS, ESS < 100 时结果不可信']
            }
        else:
            warnings.append("d>30 直接 IS 权重方差爆炸, 考虑分层 IS 或控制变量")
            return {
                'method': 'importance_sampling',
                'params': {'q': '用 Laplace 近似 (在 MAP 处高斯化)'},
                'warnings': warnings
            }

    # Case 4: 知道条件分布
    if problem.get('has_conditional', False):
        return {
            'method': 'gibbs',
            'params': {'initial_point': 'MAP 估计', 'burn_in': '1000-5000'},
            'warnings': ['高相关变量会导致 Gibbs 慢, 考虑 reparameterization']
        }

    # Case 5: 知道梯度 + 中高维
    if problem.get('has_gradient', False):
        return {
            'method': 'hmc_nuts',
            'params': {'step_size': 'NUTS 自动调', 'L': '5-20 (NUTS 自适应)'},
            'warnings': ['必须归一化 log p 到 0 附近, 否则 leapfrog 发散']
        }

    # Case 6: 兜底 - MH
    warnings.append("MH 在 d>20 接受率会很低, 建议先用 Laplace 近似找提议分布")
    return {
        'method': 'metropolis_hastings',
        'params': {
            'proposal': 'N(current, 2.38^2/d * Sigma_estimate)',
            'target_accept_rate': '0.234 (高维最优)'
        },
        'warnings': warnings
    }


# 使用示例
problem_a = {'d': 1, 'has_inverse_cdf': True}
print(recommend_sampler(problem_a))

problem_b = {'d': 5, 'has_inverse_cdf': False, 'need_samples': True,
             'has_conditional': True}
print(recommend_sampler(problem_b))

problem_c = {'d': 100, 'has_inverse_cdf': False, 'need_samples': True,
             'has_gradient': True}
print(recommend_sampler(problem_c))
```

### A.7 与现代 DL 的连接 (PyTorch / PyMC 实战)

#### A.7.1 PyTorch 里的"采样器工厂"

```python
import torch
import torch.distributions as D

# 1. 逆变换 (Categorical)
logits = torch.tensor([1.0, 2.0, 3.0])
probs = torch.softmax(logits, dim=-1)
dist_cat = D.Categorical(probs=probs)
sample = dist_cat.sample()  # 单个样本
samples = dist_cat.sample((1000,))  # 1000 个独立样本

# 2. 拒绝采样的现代版: 隐式重参数化 (Implicit Reparameterization)
# 适用: 分布的参数本身是神经网络输出
loc = torch.tensor([0.0], requires_grad=True)
scale = torch.nn.functional.softplus(torch.tensor([0.0]))  # 保证正
dist_normal = D.Normal(loc, scale)
# rsample() 用 reparameterization trick, 梯度可传
sample = dist_normal.rsample()  # 关键: 是 rsample() 不是 sample()!

# 3. 多模态分布: MixtureSameFamily
mix = D.Categorical(torch.tensor([0.3, 0.7]))
comp = D.Normal(torch.tensor([0.0, 5.0]), torch.tensor([1.0, 1.0]))
gmm = D.MixtureSameFamily(mix, comp)
samples = gmm.sample((1000,))  # 双峰

# 4. Beta 分布: 没有 closed form CDF inverse, PyTorch 用数值方法
alpha = torch.tensor([2.0])
beta = torch.tensor([5.0])
dist_beta = D.Beta(alpha, beta)
samples = dist_beta.sample((1000,))  # 内部用 Gamma 变换 + 拒绝
```

#### A.7.2 PyMC: 用 HMC 跑 BNN 后验

```python
import pymc as pm
import numpy as np

# 数据: y = 2*x + 1 + noise
np.random.seed(0)
N = 50
X = np.random.uniform(-1, 1, N)
y = 2 * X + 1 + np.random.normal(0, 0.3, N)

# Bayesian 线性回归
with pm.Model() as bnn:
    # 先验
    w = pm.Normal('w', mu=0, sigma=10)
    b = pm.Normal('b', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # 似然
    mu = w * X + b
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # HMC/NUTS 采样
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9)

# 后验摘要
print(pm.summary(trace))
# 输出: w 后验均值 ≈ 2.0, b ≈ 1.0, sigma ≈ 0.3

# 后验预测
with bnn:
    ppc = pm.sample_posterior_predictive(trace)
# ppc.posterior_predictive['y_obs'] 是 (chains, draws, N) 的预测样本
```

#### A.7.3 VAE 里的重参数化: 拒绝采样的"梯度版"

```python
# VAE 的 ELBO 估计: E_{q(z|x)}[log p(x|z)] ≈ (1/L) sum_l log p(x|z_l)
# z_l 怎么采? 重参数化 (一种特殊的"接受采样", 100% 接受但带梯度)

class VAE_Reparameterization(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """z = mu + sigma * eps, eps ~ N(0, I)。
        相比直接 sample(), 这个保留了梯度流。
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 噪声从外面采
        return mu + eps * std  # 噪声 + 确定性变换

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
```

关键: **"eps 从外面采, 然后 z = mu + eps*sigma"** 这种技巧是拒绝采样的"接受率 100% 的特例"——确定性变换 + 随机性注入, 梯度可传。

### A.8 12 个手算练习 (含答案)

#### 练习 1: 逆变换采 Exponential

**问题**: 证明 $X = -\ln(1-U) \sim \text{Exp}(1)$, $U \sim U(0,1)$。

**答案**: $F_X(x) = 1 - e^{-x}$, $F_X^{-1}(u) = -\ln(1-u)$。 由于 $1-U \sim U(0,1)$ 也成立, 简化: $X = -\ln U$。

#### 练习 2: 拒绝采 Beta(2, 5) 的接受率

**问题**: 提议 $U(0,1)$, $M = 2.5$, 理论接受率?

**答案**: $1/M = 0.4$。 实际接受率 (Python 验证) 应在 0.39-0.41。

#### 练习 3: MH 的接受率公式

**问题**: 提议对称 $q(x'|x) = q(x|x')$, 接受概率 $\alpha = ?$

**答案**: $\alpha = \min(1, p(x')/p(x))$。 不需要 $|J|$ 修正, 因为对称 $q$ 自己抵消。

#### 练习 4: Gibbs 采 2D 高斯的精度

**问题**: 条件高斯的方差 = $1 - \rho^2$, $\rho = 0.8$ 时, 每次 Gibbs 步的"信息更新"?

**答案**: $1 - 0.64 = 0.36$, 即新样本与旧样本的相关系数 = $\rho = 0.8$ (略低于 1)。 100 步后样本自相关衰减到 $0.8^{100} \approx 2 \times 10^{-10}$, 几乎独立。

#### 练习 5: IS 的 ESS 上界

**问题**: $n$ 个样本的 ESS 上界是?

**答案**: $n$。 当 $q = p$ 时达到, 权重均匀 = $1/n$, ESS = $n$。

#### 练习 6: 拒绝采样的高维失效

**问题**: $d$ 维高斯 $N(0, I)$, 包络 $M \cdot U([-5, 5]^d)$, 接受率?

**答案**: $\text{接受率} = \frac{1}{M \cdot 10^d} \cdot \int_{[-5,5]^d} N(0,I) dx \approx \frac{1}{M \cdot 10^d}$。 $d=10$ 时 $10^{-10}$, 几乎永远采不到。 **这就是维度诅咒**: 拒绝采样在高维完全失效。

#### 练习 7: MH 的细致平衡

**问题**: 证明 MH 满足细致平衡 $p(x) q(x'|x) \alpha = p(x') q(x|x') \alpha'$。

**答案**: $p(x) q(x'|x) \cdot \min(1, p(x')q(x|x')/p(x)q(x'|x)) = \min(p(x)q(x'|x), p(x')q(x|x'))$。 两边交换 $x, x'$ 同样成立, 所以细致平衡。 加上**遍历性 + 非周期性** → 平稳分布 = $p$。

#### 练习 8: 重要性采样的偏差

**问题**: 普通 IS 估计 $\hat \mu = \sum_i w_i f(x_i)$, $\mathbb{E}_q[\hat \mu] = \mathbb{E}_p[f]$ 吗?

**答案**: 是**无偏**的 (只要 $q$ 覆盖 $p$ 的支撑), 但**方差**取决于 $q$ 选得多好。 这是 IS 与 MCMC 的根本差异: MCMC 渐近无偏 (chain 长度 → ∞), IS 有限样本无偏。

#### 练习 9: HMC 的接受率

**问题**: 理想 leapfrog (无误差) 时, 接受率 = ?

**答案**: 100%。 因为 Hamiltonian 守恒, $H(x', p') = H(x, p)$, 所以 $\alpha = \min(1, \exp(-H_{\text{new}} + H_{\text{old}})) = 1$。 实际 leapfrog 有误差, 接受率 ~65-95%。

#### 练习 10: Gibbs 失效场景

**问题**: 什么时候 Gibbs 极慢?

**答案**: 变量**强相关**时。 例: $x_1 + x_2 = c$ 的约束下, Gibbs 一次只更新一个变量, 必须"等"另一个, 收敛极慢。 解决: reparameterize (换成 $u_1 = x_1 + x_2$, $u_2 = x_1 - x_2$)。

#### 练习 11: ESS 计算

**问题**: 5 个重要性权重 $[0.5, 0.2, 0.15, 0.1, 0.05]$ (已归一化), ESS = ?

**答案**: $1 / (0.25 + 0.04 + 0.0225 + 0.01 + 0.0025) = 1 / 0.325 = 3.08$。 5 个样本只相当于 3 个独立样本, 有效率 60%。

#### 练习 12: 提议分布的最优尺度

**问题**: MH 在 d 维, 提议 $N(x, \sigma^2 I)$, 最优 $\sigma$?

**答案**: Roberts, Gelman, Gilks (1997): $\sigma_{\text{opt}} = 2.38 / \sqrt{d}$, 对应 ~23.4% 接受率 (高维)。 一维特例: $\sigma_{\text{opt}} = 2.38$, 接受率 ~44%。

### A.9 高维采样的"工业配方" (Cheat Sheet)

```text
┌─────────────────────────────────────────────────────────────────┐
│           高维采样问题 (d > 10) 的标准工作流                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. 先看 p 是不是可分解 (条件独立)?                               │
│    └─ 是 → Gibbs 扫一遍 (便宜)                                  │
│ 2. 找 MAP (梯度下降 / scipy.optimize)                           │
│    └─ 在 MAP 处计算 Hessian, 估计协方差 Σ                        │
│ 3. 用 N(μ_MAP, Σ) 作为提议, 跑 HMC/NUTS                         │
│    └─ PyMC / Stan / NumPyro 默认                                │
│ 4. 跑 4 条链, 看 R-hat < 1.05                                   │
│ 5. 看 ESS, 至少 ESS > 400                                       │
│ 6. 报告: 后验均值 ± 1.96*后验标准差, 95% HDI                     │
└─────────────────────────────────────────────────────────────────┘
```

### A.10 进一步阅读

| 主题 | 经典论文 / 资源 |
| :--- | :--- |
| MH 算法 | Metropolis et al. (1953), Hastings (1970) |
| Gibbs 采样 | Geman & Geman (1984), Gelfand & Smith (1990) |
| HMC | Duane et al. (1987), Neal (2011) "MCMC using Hamiltonian dynamics" |
| NUTS | Hoffman & Gelman (2014) "The No-U-Turn Sampler" |
| 重要性采样 | Rubinstein & Kroese (2016) "Simulation and the Monte Carlo Method" |
| 拒绝采样 | Von Neumann (1951) (原始思想) |
| BNN + MCMC | Neal (1996) "Bayesian Learning for Neural Networks" |
| 工业级 | Salvatier, Wiecki, Fonnesbeck (2016) "Probabilistic programming in Python using PyMC3" |
| 现代综述 | Brooks, Gelman, Jones, Meng (2011) "Handbook of Markov Chain Monte Carlo" |

### A.11 1 页总览 (贴在桌上)

```text
┌────────────────────────────────────────────────────────┐
│  5 种采样方法 (1 页速记)                                  │
├────────────────────────────────────────────────────────┤
│ 逆变换:  已知 F^{-1}      → X = F^{-1}(U)              │
│          适用: 标准分布, 几何/指数                       │
│                                                        │
│ 拒绝:    p ≤ Mq          → 接受率 = 1/M                │
│          适用: 低维单峰, 接受率不爆炸                    │
│          坑: M 难估, 高维崩溃                            │
│                                                        │
│ 重要性:  E_p[f] ≈ Σ w_i f(x_i), w_i = p(x_i)/q(x_i)  │
│          适用: 估计期望, 不需独立样本                    │
│          坑: ESS 必须 > 100, q 选错就废                  │
│                                                        │
│ MH-MCMC: 接受 = min(1, p(x')/p(x))                     │
│          适用: 通用, 不知道条件分布时                     │
│          坑: 高维提议难调, 自相关高                       │
│                                                        │
│ Gibbs:   逐分量条件采样                                  │
│          适用: 条件分布已知, 高维 (d > 100)              │
│          坑: 强相关变量时慢, 需重参数化                   │
│                                                        │
│ HMC:     用 ∇log p 沿 Hamilton 动力学走                 │
│          适用: 知道梯度, 中高维                          │
│          坑: leapfrog 步长需调, 不能用于离散变量          │
│                                                        │
│ 决策: 知道 F^{-1} → 逆变换                                │
│       要样本独立 + 低维 → 拒绝 (M 紧)                     │
│       只要期望, 不要样本 → IS                             │
│       通用 / 不知道条件 → MH (调提议)                    │
│       知道条件 → Gibbs                                   │
│       知道梯度 + 中高维 → HMC / NUTS                     │
└────────────────────────────────────────────────────────┘
```

### A.12 实战项目 3 个 (从 ML 论文到代码)

#### 项目 1: 复现 "A Bayesian Analysis of Beta-Binomial Model" (经典)

**目标**: 复现 Gelman et al. (2013) BDA3 第 5 章的 Beta-Binomial 后验。

```python
# 数据: 5 个独立实验, 每个 n_i 次试验, y_i 次成功
data = [
    (10, 1),   # 实验 1: 10 次试验, 1 次成功
    (10, 3),
    (10, 5),
    (10, 8),
    (10, 10),
]
# 假设每个实验的 theta 不同, 整体 theta 服从 Beta(alpha, beta) 先验
# 这就是经典的"分层 Beta-Binomial"
# 用 PyMC:
import pymc as pm

with pm.Model() as hierarchical:
    # 超先验
    alpha = pm.HalfNormal('alpha', sigma=10)
    beta = pm.HalfNormal('beta', sigma=10)
    # theta 的先验 (依赖于 alpha, beta)
    theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=5)
    # 似然
    y = pm.Binomial('y', n=[d[0] for d in data],
                    p=theta, observed=[d[1] for d in data])
    # 采样
    trace = pm.sample(5000, tune=2000, chains=4, target_accept=0.95)

# 查看 theta 的后验
print(pm.summary(trace, var_names=['theta', 'alpha', 'beta']))
# alpha, beta 描述了 theta 的总体分布, 体现"借用强度" (shrinkage)
```

#### 项目 2: 用 MCMC 估 LLM 回答的不确定性

**目标**: 对 LLM 同一问题的多次回答, 估"真实分布"。

```python
# 假设 LLM 对一个问题回答了 100 次, 每次给出 [0, 1] 之间的置信度
# 我们要估"真实分布"的均值和方差
import numpy as np
from scipy import stats

llm_responses = np.random.beta(5, 2, size=100)  # 模拟 LLM 偏自信的输出

# 1. 直接估计
print(f"直接均值 = {llm_responses.mean():.3f}, "
      f"直接方差 = {llm_responses.var():.3f}")

# 2. Beta-Binomial 视角: 假设每个回答是"对真实分布的样本"
#    用 MCMC 估 Beta(alpha, beta) 的后验
def log_posterior(alpha_beta, data):
    alpha, beta = alpha_beta
    if alpha <= 0 or beta <= 0:
        return -np.inf
    return (alpha - 1) * np.sum(np.log(data)) + \
           (beta - 1) * np.sum(np.log(1 - data)) + \
           5 * np.log(alpha) + 5 * np.log(beta)  # 弱信息先验

# MH 采样
def log_posterior_wrapper(theta):
    return log_posterior(theta, llm_responses)

samples = metropolis_hastings(
    n=10000, log_p=log_posterior_wrapper,
    proposal_sampler=lambda x: np.random.normal(x, [0.1, 0.1]),
    x0=np.array([1.0, 1.0]), burn_in=2000
)
print(f"alpha 后验均值 = {samples[:, 0].mean():.3f}")
print(f"beta 后验均值 = {samples[:, 1].mean():.3f}")
# 后验预测的均值 = alpha / (alpha + beta), 比直接均值更准 (借用强度)
```

#### 项目 3: Thompson Sampling (推荐系统的 MCMC 思想)

**目标**: 多臂老虎机, 用 MCMC 思想做探索/利用。

```python
def thompson_sampling_bandit(n_arms, n_rounds, true_means):
    """Thompson Sampling: 每个 arm 用 Beta 后验, 采样一次选最大。

    本质: 用 Beta 分布作为后验, 每次"采样"就是一次 MCMC 风格的后验预测。
    """
    alpha = np.ones(n_arms)  # 成功 + 1
    beta = np.ones(n_arms)   # 失败 + 1
    rewards = np.zeros(n_rounds)
    choices = np.zeros(n_rounds, dtype=int)

    for t in range(n_rounds):
        # 1. 从每个 arm 的后验 Beta 采一个"估计的均值"
        sampled_means = np.random.beta(alpha, beta)
        # 2. 选估计均值最大的 arm
        arm = np.argmax(sampled_means)
        # 3. 真实环境给出奖励
        reward = 1 if np.random.uniform() < true_means[arm] else 0
        # 4. 更新后验
        alpha[arm] += reward
        beta[arm] += 1 - reward
        rewards[t] = reward
        choices[t] = arm

    return rewards, choices


# 测试
true_means = [0.2, 0.5, 0.8]  # arm 2 最好
rewards, _ = thompson_sampling_bandit(n_arms=3, n_rounds=1000,
                                       true_means=true_means)
print(f"总奖励 = {rewards.sum()} / 1000 (最优 = 800)")
# Thompson Sampling 渐近最优, 累计遗憾 O(√T log T)
```

### A.13 与 7-系统思维 案例的交叉

1. **PageRank** (案例 1): 把网页看作 MCMC 状态, 转移概率 = 链接, 平稳分布 = PageRank。 MCMC 思想 = 随机游走的极限 = 排名。
2. **蒙特卡洛树搜索 MCTS** (案例 3, AlphaGo): 每步用 UCB 选择 + 模拟到终局, 多次模拟平均 = MCMC 的稳态估计。
3. **强化学习 Off-Policy** (案例 5-8): 重要性采样贯穿, PPO 是带 Clip 的 IS。
4. **A/B 实验的贝叶斯分析**: 用 §5 的 MH 估后验转化率差, 比频率派的 t 检验更鲁棒 (小样本时)。
5. **扩散模型反向过程** (案例 9): DDPM 的反向 = 逐时刻 Gibbs 风格采样, $x_{t-1} | x_t$ 是条件高斯。
6. **粒子滤波** (案例 12): 用重要性采样 + 重采样跟踪时变分布, 本质是 sequential IS。

### A.14 最后的"3 句话总结"

1. **能解析就解析, 不能解析就采样**。 5 种方法不是"哪个最好", 是"什么场景用哪个"。
2. **采样 ≠ 准确**。 1000 个样本估计的期望有 $\pm 5\%$ 的误差, 用 95% 置信区间报, 别用"点估计 + 拍脑袋"。
3. **MCMC 调参是手艺, 不是科学**。 接受率、ESS、trace plot 三个一起看, 链不收敛就重跑, 别硬调。

---

**版本**: v1.0 | **更新**: 2026-06-24 | **下一章**: 第 10 章 VAE·Diffusion·Bayesian NN, 用本章的采样方法解 DL 的"难积分"。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样 - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 拆解为以下几个层次:

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

信息论为"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样 - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 拆解为以下几个层次:

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

信息论为"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样 - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 拆解为以下几个层次:

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

信息论为"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样 - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 拆解为以下几个层次:

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

信息论为"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样 - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 拆解为以下几个层次:

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

信息论为"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 9 章:采样方法——MCMC·Gibbs·拒绝采样·重要性采样" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



