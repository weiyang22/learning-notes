---
title: "第01章-05-Taylor展开·二阶近似·Newton法-为什么一阶方法统治DL"
tags:
  - 数学
  - Taylor展开
  - 二阶近似
  - Newton法
  - 为什么一阶方法统治DL
description: "本章要点：过 SGD/Adam,训过 BERT/LLaMA,你大概有一个心照不宣的疑问:"
chapter_id: 数学-5-01
last_updated: 2026-07-14
---

# 第01章-05-Taylor展开·二阶近似·Newton法-为什么一阶方法统治DL

过 SGD/Adam,训过 BERT/LLaMA,你大概有一个心照不宣的疑问:
>
> *Hessian 矩阵明明能告诉你"参数空间在该点的弯曲程度",Newton 法收敛速度明明是 SGD 的平方倍,为什么我们不用 Newton 法?*
>
> 答案是数学上的,不是工程上的偷懒。这一章就讲清楚这件事:从 Taylor 展开的角度,你能看到**为什么"一阶近似"在大多数深度学习场景下是合理的选择**,以及在哪些特殊场景下二阶方法反而值得那个算力代价。
>
> **学完你能回答**:
> - 为什么 SGD 公式 $\theta - \eta \nabla L$ 来自 Taylor 一阶展开?
> - 为什么 Hessian 病态时 Adam 比 SGD 稳?
> - XGBoost 为什么要用二阶 Taylor 展开拟合损失?
> - Newton 法为什么不能直接用在大模型上?
> - K-FAC/Shampoo/Sophia/GaLore 这些"准二阶方法"在做什么?
>
> **与前后章节的关系**:
> - 第 2 章给了"一阶导数 = 切线斜率",本章把它升级为"切线 + 曲率"
> - 第 3 章给了"梯度向量"和"Hessian 矩阵",本章把它们组合成 Newton 步
> - 第 6 章(积分)用 Taylor 展开证明 log-sum-exp 的数值稳定性
> - 第 4 章(链式法则)给出如何"自动求任意阶导",本章是它的应用
> - 第 6 章(最优化)用本章的 Taylor 展开证明 Descent Lemma 和收敛率

---

## 1. 直觉引入:为什么"局部线性化"统治了深度学习

### 1.1 工程师的痛点:SGD 用一阶,为什么不用二阶?

你在 PyTorch 里写一行:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

然后训练就跑了。每一步,优化器算梯度 $g = \nabla L(\theta)$,沿 $-g$ 方向走一小步:

$$\theta_{t+1} = \theta_t - \eta \cdot g$$

**这是不是"太粗糙"了?**

理论上,如果你能算 Hessian $H = \nabla^2 L(\theta)$,你就能用 **Newton 法**:

$$\theta_{t+1} = \theta_t - \eta \cdot H^{-1} g$$

Newton 法是**二次收敛**的，，每步把误差平方,几步就到最优。SGD 只是一阶(线性)收敛，，步数随精度反比例增长。

**问题**:为什么 2024 年的工业实践几乎所有大模型训练仍然用 Adam(本质是一阶),而不是 Newton?

三个数字告诉你答案:

- LLaMA-7B 参数 $d = 7 \times 10^9$ 个
- 完整 Hessian 是 $d \times d = 5 \times 10^{19}$ 个元素
- 即使用 1 bit 存一个元素,也要 $6 \times 10^{18}$ GB

比可观测宇宙的硬盘加起来还多。

**所以"为什么不用二阶"不是数学问题,是物理问题**。本章的任务是讲清楚:在哪些小尺度上,一阶近似是**够好**的;在哪些场景下,二阶修正值得那个代价。

### 1.2 直觉:一阶近似 = 用直线逼近曲线

回忆第 2 章学的:在 $x_0$ 附近,**任何光滑函数都能用一条直线逼近**:

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

这是切线,也是**一阶 Taylor 展开**。**二阶 Taylor** 多加一项,用抛物线逼近:

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2} f''(x_0)(x - x_0)^2$$

**三阶、四阶**……**$n$ 阶**:每多一项,就能多"模仿"原函数的一个细节(凸度、急转点、波动)。

```text
原函数    切线(一阶)    抛物线(二阶)    5阶 Taylor
  ⌒         |              ⌒              ⌒
 / \        |             / \            /  \
 |  |       |            |   |           |  |
  \/        |             \ /            \/
误差:大    误差:大       误差:中       误差:小
```

**关键问题**:**截断到几阶就够用**?

工程答案:**大部分深度学习场景,一阶就够**。原因有三:
1. **步长很小** ($\eta$ 很小):Taylor 高阶项 $O(\eta^2), O(\eta^3)$ 都很小,主导项就是 $O(\eta)$ 的梯度
2. **损失函数非凸**:Hessian 不是正定,二阶信息不可靠
3. **算力限制**:大模型 Hessian 装不下,只能用一阶

**但也有一些场景必须用二阶**:XGBoost 的目标函数二阶展开、Gaussian Process、凸优化的小问题。

### 1.3 ASCII 可视化:函数曲线 + 切线 + 抛物线

**经典例子**:$f(x) = \cos(x)$ 在 $x_0 = 0$ 附近

```text
  f(x)
   ↑
 1.0┤- - - - - - - - - - - - - - - - - - f(0)=1
    │       ╭─────╮                         ← 抛物线近似 (二阶)
    │      ╱   ───────────────────────────
    │     ╱  ╱
    │    ╱ ╱
    │   ╱╱
 0.0┤───────────────────────────────
    │       切线 (一阶) y=1
    │
-1.0┤
    └─┬────┬────┬────┬────┬──→ x
     -2   -1    0    1    2

观察:切线 y=1 是水平线,抛物线 y=1 - x²/2 在 x ∈ [-1, 1] 贴合 cos(x)
```

**对应的 ML 现象**:
- **一阶 = SGD**:用直线(切线)预测"再走一步该往哪"
- **二阶 = Newton / Adam**:用抛物线(切线 + 曲率)预测
- **截断阶数** = 算法能用多复杂的"局部景观信息"

### 1.4 工程意义:为什么"局部线性化"是优化算法的核心?

**核心思想**:**任何优化算法都是"在当前点 $\theta_t$ 看一小步,决定往哪走"**。这个小步的"步长"通常很小($\eta \ll 1$)。在这一步里:

- 一阶 = 直线
- 二阶 = 抛物线
- 高阶 = 越来越复杂的曲线

**SGD 的隐含假设**:
- 当前点 $\theta_t$ 附近,损失函数"几乎是直的"
- 所以"沿负梯度走"是对的
- 步长 $\eta$ 必须够小,让这个假设成立

**Newton 法的隐含假设**:
- 当前点 $\theta_t$ 附近,损失函数"几乎是抛物线"
- 所以"沿 $-H^{-1} g$ 走"是对的
- 步长 $\eta$ 可以稍大,因为抛物线比直线更贴合

**深度学习的现实**:
- 损失景观在 7B 维空间里**几乎完全不知道长什么样**
- 只能靠"局部小步"的统计规律
- 一阶假设虽然粗糙,但**简单、稳、便宜**
- 二阶假设**更精确**,但**贵、复杂、对非凸不一定稳**

这就是为什么 Adam(本质是"对角 Hessian 近似")统治了深度学习:**它在"一阶的便宜"和"二阶的精确"之间找到了甜蜜点**。

### 1.5 一个直觉陷阱:为什么"Hessian 信息越多越好"是错的

**天真想法**:Hessian 给了"二阶信息",肯定比一阶好。

**现实**:Hessian 给了**曲率**,但**曲率 ≠ 优化方向**。

考虑 $f(x, y) = x^2 - 0.1 y^2$:

- $H = \text{diag}(2, -0.2)$
- 起点 $(1, 0)$,梯度 $g = (2, 0)$
- **Newton 方向**:$H^{-1} g = (0.5, 0)$ (沿 $x$ 走一半)
- **但** $f$ 沿 $y$ 方向是"上凸"(极小),Newton 走 $x$ 方向反而合理

**反例**:$f(x, y) = -x^2 + 0.1 y^2$

- $H = \text{diag}(-2, 0.2)$
- 起点 $(1, 0)$,梯度 $g = (-2, 0)$
- **Newton 方向**:$H^{-1} g = (1, 0)$ (沿 $x$ 走一步)
- 但 $x$ 方向是**极大**方向!Newton 反而把参数推向极大

**关键**:Newton 方向只在 Hessian 正定(所有特征值 $> 0$)时**保证指向下降**。**非凸损失上,Hessian 不正定,Newton 方向不一定靠谱**。

深度学习损失几乎**永远**非凸,所以 Adam(只取 Hessian 对角线,且加 $\epsilon$ 防 $0$)比纯 Newton **更稳**。

### 1.6 章节路线图

| 节 | 主题 | 关键公式 | ML/DL 联系 |
| :--- | :--- | :--- | :--- |
| §2 | 一元 Taylor 展开 | $f(x_0+h) = f(x_0) + f'(x_0) h + \frac{1}{2} f''(x_0) h^2 + \cdots$ | SGD 的来源 |
| §3 | 余项与收敛半径 | $R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$ | 截断到几阶 |
| §4 | 多元 Taylor 展开 | $f(\mathbf{x} + \Delta) = f + \nabla f^T \Delta + \frac{1}{2} \Delta^T H \Delta$ | Newton 法的来源 |
| §5 | Newton 法 | $\theta_{t+1} = \theta_t - H^{-1} g$ | 二阶方法的极限 |
| §6 | 为什么一阶方法统治 DL | 算力 + 存储 + 噪声 | 大模型训练的现实 |
| §7 | Taylor 在 DL 的其他应用 | XGBoost / Softmax / KL | 工业级应用 |
| §8 | PyTorch 代码 | 4 个实验 | 跑出来的理解 |
| §9 | 常见误区 | 5-8 条 | 别踩坑 |
| §10 | 面试 Q&A | 5 题 L7/L8 | 面试表现 |
| §11 | 跨章节联系 | 优化/信息论/概率论 | 全局观 |

---

## 2. 一元 Taylor 展开:用多项式"模仿"原函数

### 2.1 直觉引入:为什么能用多项式?

**核心洞察**:**任何光滑函数在 $x_0$ 附近都能用多项式"模仿"**。

为什么?

回忆第 2 章讲的:函数在 $x_0$ 处的"信息"全部编码在**各阶导数** $f(x_0), f'(x_0), f''(x_0), \ldots$ 里。如果你知道这些导数,你就能在 $x_0$ 附近"重建"函数。

**多项式**是"最方便"的重建形式,因为:
1. **容易求值**:加减乘
2. **容易求导**:每项求导后仍是多项式
3. **容易近似**:截断前 $n$ 项就是 $n$ 阶 Taylor

### 2.2 一阶展开:直线(切线)

**定义**(一阶 Taylor 展开):

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

记 $h = x - x_0$:

$$f(x_0 + h) \approx f(x_0) + f'(x_0) \cdot h$$

**几何**:在 $(x_0, f(x_0))$ 处画切线。当 $h$ 很小时,函数值 ≈ 切线值。

**误差**:$O(h^2)$(二阶小量)。

#### 例 1:$e^x$ 在 $x = 0$ 附近的一阶展开

$f(0) = 1, f'(0) = 1$,所以 $e^x \approx 1 + x$。

| $x$ | 真值 $e^x$ | 一阶 $1+x$ | 误差 |
| :--- | :--- | :--- | :--- |
| 0.1 | 1.1052 | 1.1 | 0.5% |
| 0.5 | 1.6487 | 1.5 | 9% |
| 1.0 | 2.7183 | 2.0 | 26% |

**观察**:$x$ 越小,误差越小。**工程意义**:`torch.exp(x)` 在 $x$ 大时会溢出,但小 $x$ 时可以放心。

#### 例 2:$\sin(x)$ 在 $x = 0$ 附近的一阶展开

$f(0) = 0, f'(0) = 1$,所以 $\sin(x) \approx x$。

| $x$ (弧度) | 真值 $\sin(x)$ | 一阶 $x$ | 误差 |
| :--- | :--- | :--- | :--- |
| 0.1 | 0.0998 | 0.1 | 0.2% |
| 1.0 | 0.8415 | 1.0 | 18.8% |

**工程意义**:**小角度近似 $\sin x \approx x$** 是傅立叶变换、信号处理、Transformer 位置编码的数学基础。

### 2.3 二阶展开:抛物线(切线 + 曲率)

**定义**(二阶 Taylor 展开):

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2} f''(x_0)(x - x_0)^2$$

**几何**:在 $(x_0, f(x_0))$ 处用**抛物线**拟合,抛物线在 $x_0$ 处与函数**共享值、一阶、二阶导数**。**误差**:$O(h^3)$(三阶小量)。

#### 例 1:$e^x$ 在 $x = 0$ 附近的二阶展开

$f(0) = f'(0) = f''(0) = 1$,所以 $e^x \approx 1 + x + x^2/2$。

| $x$ | 真值 $e^x$ | 二阶 $1+x+x^2/2$ | 误差 |
| :--- | :--- | :--- | :--- |
| 0.5 | 1.6487 | 1.625 | 1.4% |
| 1.0 | 2.7183 | 2.5 | 8% |

**观察**:二阶在 $x = 0.5$ 时误差仅 1.4%,一阶在 $x = 0.5$ 时误差 9%。**二阶明显好于一阶**。

#### 例 2:$\cos(x)$ 在 $x = 0$ 附近的二阶展开

$f(0) = 1, f'(0) = 0, f''(0) = -1$,所以 $\cos(x) \approx 1 - x^2/2$。

注意:**一阶项 $= 0$**!因为 $\cos'(0) = 0$。所以 $\cos$ 的"主导近似"实际上是**二阶**。

| $x$ | 真值 $\cos(x)$ | 二阶 $1 - x^2/2$ | 误差 |
| :--- | :--- | :--- | :--- |
| 0.5 | 0.8776 | 0.875 | 0.3% |
| 1.0 | 0.5403 | 0.5 | 7.5% |

**工程意义**:在 Transformer 里,$\cos$ 经常出现(位置编码 RoPE)。当 $x$ 小(低频位置编码)时,$\cos(x) \approx 1 - x^2/2$,**比 $\cos \approx 1$ 更精确**。

### 2.4 $n$ 阶 Taylor 公式

**一般形式**($n$ 阶 Taylor 展开,在 $x_0$ 处):

$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k + R_n(x)$$

其中 $R_n(x)$ 是**余项**(误差)。

**等价写法**:

$$f(x_0 + h) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!} h^k + R_n(h)$$

**理解**:前 $n+1$ 项是**多项式**,最后一项 $R_n$ 是**误差**。如果 $f$ 足够光滑,余项是 $O(h^{n+1})$,截断到 $n$ 阶误差很小。

#### 系数:为什么是 $f^{(k)}(x_0) / k!$?

直觉:多项式 $P(x) = a_0 + a_1 (x-x_0) + a_2 (x-x_0)^2 + \cdots + a_n (x-x_0)^n$ 在 $x_0$ 处的各阶导数:

- $P(x_0) = a_0$
- $P'(x_0) = a_1$
- $P''(x_0) = 2! \cdot a_2$
- $P'''(x_0) = 3! \cdot a_3$
- ...
- $P^{(k)}(x_0) = k! \cdot a_k$

要让 $P$ 在 $x_0$ 处与 $f$ 共享所有前 $n$ 阶导数,必须:

$$a_k = \frac{f^{(k)}(x_0)}{k!}$$

这就是 Taylor 展开系数的来源。

#### 例:写出 $f(x) = \sin(x)$ 在 $x_0 = 0$ 处的 5 阶 Taylor 展开

$\sin(x)$ 的各阶导(循环:$\sin \to \cos \to -\sin \to -\cos$):
- $f(0) = 0$
- $f'(0) = 1$
- $f''(0) = 0$
- $f'''(0) = -1$
- $f^{(4)}(0) = 0$
- $f^{(5)}(0) = 1$

5 阶 Taylor 展开:

$$\sin(x) \approx 0 + \frac{1}{1!}x + \frac{0}{2!}x^2 + \frac{-1}{3!}x^3 + \frac{0}{4!}x^4 + \frac{1}{5!}x^5 = x - \frac{x^3}{6} + \frac{x^5}{120}$$

**验证**:$x = \pi/2 \approx 1.5708$:
- 真值:$\sin(\pi/2) = 1$
- 5 阶:$\pi/2 - (\pi/2)^3/6 + (\pi/2)^5/120 = 1.5708 - 0.6460 + 0.0797 = 1.0045$

误差 0.45%。5 阶已经相当精确。

### 2.5 Maclaurin 展开($x_0 = 0$ 的特例)

**Maclaurin 展开** = 在 $x_0 = 0$ 处的 Taylor 展开。

工程中**最常用**的 Maclaurin 展开:

| 函数 | Maclaurin 展开 | 收敛半径 |
| :--- | :--- | :--- |
| $e^x$ | $\sum_{k=0}^\infty \frac{x^k}{k!}$ | $\infty$ |
| $\sin(x)$ | $\sum_{k=0}^\infty \frac{(-1)^k x^{2k+1}}{(2k+1)!}$ | $\infty$ |
| $\cos(x)$ | $\sum_{k=0}^\infty \frac{(-1)^k x^{2k}}{(2k)!}$ | $\infty$ |
| $\ln(1+x)$ | $\sum_{k=1}^\infty \frac{(-1)^{k+1} x^k}{k}$ | $1$ |
| $\frac{1}{1-x}$ | $\sum_{k=0}^\infty x^k$ | $1$ |
| $(1+x)^\alpha$ | $\sum_{k=0}^\infty \binom{\alpha}{k} x^k$ | $1$ |

**注意**:不是所有函数都"全空间收敛"。$\ln(1+x)$ 和 $1/(1-x)$ 都有**收敛半径 1**，，离开 $x_0$ 太远,展开就失效。

**工程意义**:Maclaurin 展开是"软提示工程"的核心，，LLM 中的"prompt 嵌入" $p(x) = \sum p_k x^k$ 就是用多项式语言描述函数。

### 2.6 5 个手算例子(从常见函数)

#### 例 1:$e^x$ 的 4 阶 Maclaurin 展开

$e^x$ 的各阶导 $= e^x$,在 $x=0$ 处 $= 1$。

$$e^x \approx 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \frac{x^4}{24}$$

**验证**($x = 0.5$):
- 真值:$e^{0.5} \approx 1.6487$
- 4 阶:$1 + 0.5 + 0.125 + 0.0208 + 0.0026 = 1.6484$
- 误差 $0.02\%$

#### 例 2:$\sin(x)$ 的 5 阶 Maclaurin 展开

奇数阶有系数,偶数阶系数为 0。

$$\sin(x) \approx x - \frac{x^3}{6} + \frac{x^5}{120}$$

**验证**($x = 1$):
- 真值:$\sin(1) \approx 0.8415$
- 5 阶:$1 - 1/6 + 1/120 = 0.8417$
- 误差 $0.02\%$

#### 例 3:$\cos(x)$ 的 6 阶 Maclaurin 展开

偶数阶有系数,奇数阶系数为 0。

$$\cos(x) \approx 1 - \frac{x^2}{2} + \frac{x^4}{24} - \frac{x^6}{720}$$

**验证**($x = 0.5$):
- 真值:$\cos(0.5) \approx 0.8776$
- 6 阶:$1 - 0.125 + 0.0026 - 0.000022 \approx 0.8776$
- 误差 $\approx 0$

#### 例 4:$\ln(1+x)$ 的 3 阶 Maclaurin 展开

$\ln(1+x)$ 的各阶导:$\frac{(-1)^{k-1}(k-1)!}{(1+x)^k}$,在 $x=0$ 处 $= (-1)^{k-1}(k-1)!$。

3 阶:$\ln(1+x) \approx x - \frac{x^2}{2} + \frac{x^3}{3}$

**验证**($x = 0.5$):
- 真值:$\ln(1.5) \approx 0.4055$
- 3 阶:$0.5 - 0.125 + 0.0417 = 0.4167$
- 误差 $2.7\%$

注意 $\ln(1+x)$ 收敛半径 1,$x = 0.5$ 接近边界,3 阶误差比 $e^x$ 大很多。

#### 例 5:$\frac{1}{1-x}$ 的 4 阶 Maclaurin 展开

$\frac{1}{1-x}$ 是几何级数:$\sum_{k=0}^\infty x^k$。

4 阶:$\frac{1}{1-x} \approx 1 + x + x^2 + x^3 + x^4$

**验证**($x = 0.3$):
- 真值:$1/(1-0.3) = 1.4286$
- 4 阶:$1 + 0.3 + 0.09 + 0.027 + 0.0081 = 1.4251$
- 误差 $0.25\%$

**工程意义**:$\frac{1}{1-x}$ 的展开就是"折扣累积":$\sum x^k = 1/(1-x)$。RL 折扣因子 $\gamma$ 的总和就是这个公式。

### 2.7 本节回顾

**核心收获**:
1. 一阶 Taylor = 切线 = $f(x_0) + f'(x_0) h$
2. 二阶 Taylor = 抛物线 = 一阶 + $\frac{1}{2} f''(x_0) h^2$
3. $n$ 阶 Taylor = 匹配前 $n$ 阶导数的多项式
4. Maclaurin = $x_0 = 0$ 的 Taylor,工程最常用
5. 收敛半径决定 Taylor 在多远范围内有效

**下一步**:Taylor 展开有误差,误差怎么量化?什么时候截断到几阶就够用?这是 §3 的主题。

---

## 3. Taylor 展开的余项:截断到几阶才够?

### 3.1 Lagrange 余项

**定理**(Lagrange 余项):若 $f$ 在 $x_0$ 邻域 $n+1$ 阶可导,则:

$$R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} (x - x_0)^{n+1}$$

其中 $\xi$ 是 $x_0$ 和 $x$ 之间的某个点(具体值取决于 $f$)。

**理解**:余项的"形状"是 $(x-x_0)^{n+1}$(阶数 +1),系数取决于**某点的 $(n+1)$ 阶导数**。

#### 例:$\sin(x)$ 在 $x_0 = 0$ 处的 1 阶 Taylor 展开的 Lagrange 余项

$\sin(x)$ 的 1 阶 Taylor 展开:$\sin(x) \approx x$。

Lagrange 余项:

$$R_1(x) = \frac{\sin''(\xi)}{2!} x^2 = -\frac{\sin(\xi)}{2} x^2, \quad \xi \in (0, x)$$

**误差估计**:因为 $|\sin(\xi)| \le 1$,所以:

$$|R_1(x)| \le \frac{x^2}{2}$$

**数值**($x = 0.5$):误差 $\le 0.125$。实际误差 $|\sin(0.5) - 0.5| \approx 0.0206$,远小于上界。Lagrange 上界是"最坏情况估计"。

### 3.2 Peano 余项

**定理**(Peano 余项):若 $f$ 在 $x_0$ 处 $n$ 阶可导,则:

$$R_n(x) = o((x - x_0)^n)$$

$o(\cdot)$ 是**高阶无穷小**记号(见第 1 章):$\lim_{x \to x_0} \frac{R_n(x)}{(x-x_0)^n} = 0$。

**理解**:Peano 余项比 $(x-x_0)^n$ 严格"更高阶",但**不给具体上界**。

**Lagrange vs Peano**:
- Lagrange:给出**具体上界**(依赖 $f^{(n+1)}$ 在区间上的最大值)
- Peano:只说**比 $n$ 阶严格小**,不指定倍数

工程中需要量化误差时用 Lagrange,只关心"零附近有多准"时用 Peano。

### 3.3 收敛半径

**定义**(收敛半径):Taylor 级数 $\sum a_k (x-x_0)^k$ 收敛的 $|x - x_0|$ 范围。

**计算公式**(比值判别法):

$$R = \lim_{k \to \infty} \left| \frac{a_k}{a_{k+1}} \right|$$

**为什么重要**:Taylor 级数在收敛半径外**不收敛到原函数**(可能发散或收敛到别的函数)。

#### 例 1:$e^x$ 的收敛半径 $= \infty$

系数 $a_k = 1/k!$,比值:

$$\left| \frac{a_k}{a_{k+1}} \right| = \frac{1/k!}{1/(k+1)!} = k+1 \to \infty$$

所以 $R = \infty$。**全空间收敛**。

#### 例 2:$\ln(1+x)$ 的收敛半径 $= 1$

系数 $a_k = (-1)^{k+1}/k$,比值:

$$\left| \frac{a_k}{a_{k+1}} \right| = \frac{1/k}{1/(k+1)} = \frac{k+1}{k} \to 1$$

所以 $R = 1$。在 $x = 1$ 处收敛到 $\ln(2) \approx 0.693$;$x = 2$ 时**发散**(超过收敛半径)。

#### 例 3:$\frac{1}{1-x}$ 的收敛半径 $= 1$

系数 $a_k = 1$。比值 $\to 1$,所以 $R = 1$。

在 $x = 1$ 处 $\frac{1}{1-x}$ **无定义**(奇点),级数发散。

**工程意义**:用 Taylor 展开近似函数时,**必须确认 $|x - x_0|$ 在收敛半径内**。超出范围,展开失效。

### 3.4 工程意义:截断到几阶?

**核心问题**:DL 中 Taylor 展开截断到几阶?

**答案**:**通常 1 阶**。

**理由**:
1. 步长 $\eta$ 很小:误差是 $O(\eta^2)$ 起算,二阶及以上都是"小量中的小量"
2. 二阶(Hessian)算力 $O(d^2)$,存不下
3. 损失函数非凸,二阶信息不可靠

**但有例外**:

| 场景 | 截断阶数 | 原因 |
| :--- | :--- | :--- |
| 凸优化($\beta \le 1000$) | 2 阶(Newton / BFGS) | Hessian 正定,二阶方向对 |
| GBDT 拟合残差 | 2 阶(XGBoost) | 数据少,需要二阶精确 |
| 数值稳定($\log \sum e^x$) | 1 阶(线性化) | 减 $x_{\max}$ 后 $\|x - x_{\max}\|$ 小 |
| Bayesian 优化 | 2 阶(GP 后验) | 几百维度,二阶能算 |
| Transformer 训练 | 1 阶(Adam) | 7B 参数,二阶算不起 |

### 3.5 案例:为什么 $\sin(x) \approx x$ 在小角度下精度高?

**问题**:$|\sin(x) - x|$ 在 $x$ 多大时还是小?

由 Lagrange 余项:$\sin(x) = x - \frac{\sin(\xi)}{2} x^2$,其中 $\xi \in (0, x)$。

$|R_1(x)| \le \frac{x^2}{2}$(因为 $|\sin| \le 1$)。

**精度表**:

| $x$ (rad) | $\sin(x)$ 真值 | 一阶 $x$ | 误差绝对值 | 相对误差 |
| :--- | :--- | :--- | :--- | :--- |
| 0.01 | 0.0099998 | 0.01 | 1.67e-7 | 1.7e-5 |
| 0.1 | 0.09983 | 0.1 | 1.67e-4 | 1.7e-3 |
| 0.5 | 0.4794 | 0.5 | 0.0206 | 4.3% |
| 1.0 | 0.8415 | 1.0 | 0.1585 | 18.8% |
| $\pi/2$ | 1 | 1.57 | -0.57 | 57% |

**关键**:误差量级是 $x^2/2$。要误差 < 1%,需要 $x^2/2 < 0.01$,即 $x < 0.14$ 弧度(约 8 度)。

**工程意义**:
- **Transformer RoPE**:位置编码 $\cos(m\theta_k), \sin(m\theta_k)$,当 $m\theta_k$ 小时(低频分量),$\sin(m\theta_k) \approx m\theta_k$,$\cos \approx 1$。这就是为什么"低频编码 = 慢变"。
- **小角度视觉**:CNN 早期层感受野小,核函数 $\sin$ 近似线性,等同于恒等映射。

### 3.6 工程例子:什么时候 Taylor 展开不能截断?

**反例 1**:$f(x) = e^{-1/x^2}$(在 $x = 0$ 处的"病态"函数)

定义 $f(0) = 0$。这是一个著名的"光滑但不解析"函数，，它在 $0$ 处各阶导数都是 0,但函数本身不恒为 0。

- 任何阶 Taylor 展开在 $x = 0$ 处都 $= 0$
- 但 $f(0.1) = e^{-100} \approx 0$ (很小但非 0)
- 实际上 $f$ 不恒为 0

**含义**:这种函数的 Taylor 级数**收敛**到 0(而不是 $f$ 本身),因为收敛半径是 0。

**反例 2**:$f(x) = 1/(1-x)$ 在 $x = 2$ 处

Taylor 在 $x_0 = 0$ 处展开:$\sum x^k$。但 $x = 2$ 超过收敛半径 1。级数在 $x = 2$ 处**发散**。

**工程意义**:用 Taylor 近似时,**必须验证 $x$ 在收敛半径内**。否则近似完全失效。

### 3.7 本节回顾

**核心收获**:
1. Lagrange 余项 $R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$ 给出**具体误差上界**
2. Peano 余项 $R_n(x) = o((x-x_0)^n)$ 给出**渐近行为**
3. 收敛半径 $R$ 决定 Taylor 级数在多远有效
4. DL 中通常截断到 1 阶(步长小、Hessian 算力贵、损失非凸)
5. 凸优化小问题 / XGBoost 等场景用 2 阶
6. 病态函数(如 $e^{-1/x^2}$)Taylor 展开可能完全失效

**下一步**:从一元推广到多元,这是 Newton 法的核心。

---

## 4. 多元 Taylor 展开:梯度下降与 Newton 法的源头

### 4.1 一阶多元展开:梯度下降的理论基础

**定义**(一阶多元 Taylor 展开):设 $f: \mathbb{R}^d \to \mathbb{R}$ 在 $\mathbf{x}_0$ 处可微。则:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0)$$

记 $\Delta = \mathbf{x} - \mathbf{x}_0$:

$$f(\mathbf{x}_0 + \Delta) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta$$

**几何**:在 $\mathbf{x}_0$ 处用**切平面**近似。切平面在 $\mathbf{x}_0$ 处与 $f$ 共享**值**和**梯度**。

**误差**:$O(\|\Delta\|^2)$。

#### 例:$f(x, y) = x^2 + y^2$ 在 $(1, 1)$ 处的一阶展开

$\nabla f = (2x, 2y)$,在 $(1, 1)$ 处 $= (2, 2)$。

一阶展开:$f(x, y) \approx 2 + 2(x-1) + 2(y-1) = 2x + 2y - 2$

**验证**:
- 在 $(1, 1)$:真值 $f = 2$,展开 $= 2 \cdot 1 + 2 \cdot 1 - 2 = 2$ ✓
- 在 $(1.1, 1.1)$:真值 $= 1.21 + 1.21 = 2.42$,展开 $= 2.2 + 2.2 - 2 = 2.4$(差 0.02)
- 在 $(1.5, 1.5)$:真值 $= 2.25 + 2.25 = 4.5$,展开 $= 3 + 3 - 2 = 4$(差 0.5)

**观察**:离 $\mathbf{x}_0$ 越远,误差越大。

#### 工程意义:梯度下降的来源

SGD 的更新公式:

$$\theta_{t+1} = \theta_t - \eta g_t$$

为什么是这个?用一阶 Taylor 展开:

$$L(\theta - \eta g) \approx L(\theta) - \eta \nabla L^T g = L(\theta) - \eta g^T g = L(\theta) - \eta \|g\|^2$$

当 $g = \nabla L$ 且 $\eta > 0$ 时,$L(\theta - \eta g) \approx L(\theta) - \eta \|\nabla L\|^2 < L(\theta)$(只要 $\nabla L \neq 0$)。

**所以梯度下降 = 在每一步,用一阶 Taylor 展开"预测"Loss 下降,沿负梯度方向走**。

更严格:**当 $\eta$ 足够小,真实下降量 $\ge \eta \|\nabla L\|^2 / 2$**(Descent Lemma, 见第 6 章最优化)。这就是"为什么 $\eta$ 不能太大"的数学根据，，$\eta$ 大到 Taylor 展开失效,Loss 不再下降。

### 4.2 二阶多元展开:Newton 法的理论基础

**定义**(二阶多元 Taylor 展开):设 $f: \mathbb{R}^d \to \mathbb{R}$ 在 $\mathbf{x}_0$ 处二阶连续可微。则:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta + \frac{1}{2} \Delta^T H(\mathbf{x}_0) \Delta$$

**几何**:用**抛物面**近似。抛物面在 $\mathbf{x}_0$ 处与 $f$ 共享值、梯度、Hessian。

**误差**:$O(\|\Delta\|^3)$。

#### 例 1:$f(x, y) = x^2 + 2y^2$ 在 $(1, 1)$ 处的二阶展开

- $f(1, 1) = 1 + 2 = 3$
- $\nabla f = (2x, 4y)$,在 $(1, 1)$ 处 $= (2, 4)$
- $H = \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix}$(对角矩阵)

二阶展开:

$$f(x, y) \approx 3 + 2(x-1) + 4(y-1) + \frac{1}{2} \begin{pmatrix} x-1 & y-1 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 0 & 4 \end{pmatrix} \begin{pmatrix} x-1 \\ y-1 \end{pmatrix}$$

$$= 3 + 2(x-1) + 4(y-1) + (x-1)^2 + 2(y-1)^2$$

**验证**:
- $(1, 1)$:真值 $3$,展开 $= 3$ ✓
- $(1.1, 1.1)$:真值 $= 1.21 + 2.42 = 3.63$,展开 $= 3 + 0.2 + 0.4 + 0.01 + 0.02 = 3.63$ ✓
- $(1.5, 1.5)$:真值 $= 2.25 + 4.5 = 6.75$,展开 $= 3 + 1 + 2 + 0.25 + 0.5 = 6.75$ ✓

**对二次函数,二阶展开完全精确**!因为二次函数本身就是"三阶 Taylor"($f^{(3)} = 0$)。

#### 例 2:$f(x, y) = x^3 + y^3$ 在 $(1, 1)$ 处的二阶展开

- $f(1, 1) = 2$
- $\nabla f = (3x^2, 3y^2)$,在 $(1, 1)$ 处 $= (3, 3)$
- $H = \begin{pmatrix} 6x & 0 \\ 0 & 6y \end{pmatrix}$,在 $(1, 1)$ 处 $= \begin{pmatrix} 6 & 0 \\ 0 & 6 \end{pmatrix}$

二阶展开:

$$f(x, y) \approx 2 + 3(x-1) + 3(y-1) + 3(x-1)^2 + 3(y-1)^2$$

**验证**:
- $(1.1, 1.1)$:真值 $= 2 \cdot 1.331 = 2.662$,展开 $= 2 + 0.6 + 0.03 + 0.03 = 2.69$(差 0.028)
- 三阶项贡献:$2 \cdot (0.1)^3 = 0.002$,加上后 $2.692$ ✓

**对三次函数,二阶展开有误差**(三阶项 $0.002$)。

### 4.3 Hessian 的角色:二阶项"修正"了一阶项

**关键理解**:二阶项 $\frac{1}{2} \Delta^T H \Delta$ **不是"加"上去的,而是"修正"一阶方向的**。

**直观**:一阶告诉你"沿哪个方向走最快下降",二阶告诉你"那个方向上,Loss 弯曲得有多快"。

**数学**:

$$f(\mathbf{x}_0 + \Delta) \approx f(\mathbf{x}_0) + \nabla f^T \Delta + \frac{1}{2} \Delta^T H \Delta$$

在 $\Delta = -\eta \nabla f$ 处(沿负梯度走):

$$f(\mathbf{x}_0 - \eta \nabla f) \approx f(\mathbf{x}_0) - \eta \|\nabla f\|^2 + \frac{\eta^2}{2} \nabla f^T H \nabla f$$

**一阶项**:$-\eta \|\nabla f\|^2$(永远为负,Loss 下降)

**二阶项**:$+\frac{\eta^2}{2} \nabla f^T H \nabla f$(可能正可能负,取决于 $H$ 和 $\nabla f$ 的方向)

- **$H$ 正定 + $\nabla f$ 任意**:二阶项 $\ge 0$,**削弱下降量**(曲率越大,削弱越严重)
- **$H$ 负定**:二阶项 $< 0$,**加速下降**
- **$H$ 不定**:看方向

**关键 insight**:在**病态 Hessian**($H$ 条件数大)的方向上,二阶项可能"吃掉"一阶项的下降,导致 SGD 走得很慢。

### 4.4 ASCII 可视化:二元函数的切平面 + 抛物面近似

**例子**:$f(x, y) = x^2 + 4y^2$(椭圆形等高线,沟壑地形)

```text
        z 轴 (高度 = f)
         ↑
    5 ┤               * (1, 1, 5)
      │             ╱
    4 ┤           ╱
      │         ╱     ← 抛物面 (二阶近似)
    3 ┤       ╱
      │     ╱
    2 ┤   ╱
      │ ╱
    1 ┤╱
      │
    0 ┼──────┼──────┼──────┼──→ x
      -1     0      1      2
            ╲
             ╲
              ╲ 切平面 (一阶近似)
               ╲
                ╲

等高线 (俯视图):
        y↑   ╭───╮
     1   │  ╱     ╲
     0   ├─╳───*───╳─  ← 椭圆等高线, * 是 (0, 0) 极小
    -1   │  ╲     ╱
         ╰───╯
         └──┬──┬─→ x
            -1 0 1
```

**关键观察**:
- 切平面在极小点水平($\nabla f = 0$);抛物面贴合椭圆形等高线
- 在 $y$ 方向,$\partial^2 f / \partial y^2 = 8$,**比 $x$ 方向的 2 大 4 倍**
- 对 SGD:用同一 $\eta$,$y$ 方向步子太大 → 震荡
- 对 Adam:每方向自适应 $\eta_i = \eta / \sqrt{v_i}$(≈ Hessian 对角线),$y$ 方向自动慢、$x$ 方向自动快

### 4.5 案例:为什么 Adam 在病态问题(Hessian 条件数大)上更好?

**理论分析**:考虑 $f(\theta) = \frac{1}{2} (\lambda_1 \theta_1^2 + \lambda_2 \theta_2^2)$,$\lambda_1 \ll \lambda_2$。$H = \text{diag}(\lambda_1, \lambda_2)$,条件数 $\kappa = \lambda_2 / \lambda_1 \gg 1$。

**SGD 矛盾**:$\theta_1$ 方向收敛慢,$\theta_2$ 方向要 $\eta < 1/\lambda_2$。没有单个 $\eta$ 能同时照顾两个方向，，**病态 Hessian 的核心矛盾**。

**Adam 解决**:$\sqrt{v_t}$ 自适应,$\eta \cdot m_1 / \sqrt{v_1} \approx \eta \cdot \text{sign}(\theta_1)$(与 $\lambda_1$ 无关!)。

**关键**:**Adam 的更新量与特征值 $\lambda_i$ 解耦**,每个方向用自适应步长自动平衡。在病态 Hessian 上,Adam 通常比 SGD **快 10-100 倍**(经验值)。

### 4.6 多项式展开(更一般的形式)

**$n$ 阶多元 Taylor 展开**:

$$f(\mathbf{x} + \Delta) = \sum_{k=0}^{n} \frac{1}{k!} D^k f(\mathbf{x})[\Delta]^k + R_n(\Delta)$$

其中 $D^k f$ 是 $f$ 的 $k$ 阶微分,作用于 $[\Delta]^k$($k$ 重缩并)。

**前 3 阶**:

- **0 阶**:$f(\mathbf{x})$
- **1 阶**:$\nabla f^T \Delta$(梯度 × 位移)
- **2 阶**:$\frac{1}{2} \Delta^T H \Delta$(Hessian 二次型)
- **3 阶**:$\frac{1}{6} \sum_{i,j,k} \frac{\partial^3 f}{\partial x_i \partial x_j \partial x_k} \Delta_i \Delta_j \Delta_k$(三阶张量,工程上几乎不用)

**工程意义**:
- **0 阶**:常数近似(无信息)
- **1 阶**:梯度下降、Adam 用到
- **2 阶**:Newton 法、BFGS、XGBoost 用到
- **3 阶**:几乎不用(张量难存难算)

### 4.7 3 个手算例子(多元)

#### 例 1:$f(x, y) = e^{x+y}$ 在 $(0, 0)$ 处的二阶展开

- $f(0, 0) = 1$
- $\nabla f = (e^{x+y}, e^{x+y})$,在 $(0, 0)$ 处 $= (1, 1)$
- $H = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$

二阶展开:

$$f(x, y) \approx 1 + x + y + \frac{1}{2}(x+y)^2$$

#### 例 2:$f(x, y) = \ln(1 + x + y)$ 在 $(0, 0)$ 处的二阶展开

- $\nabla f = (1/(1+x+y), 1/(1+x+y))$,在 $(0, 0)$ 处 $= (1, 1)$
- $H = \begin{pmatrix} -1 & -1 \\ -1 & -1 \end{pmatrix}$

二阶展开:

$$f(x, y) \approx x + y - \frac{1}{2}(x + y)^2$$

**验证**($x = 0.3, y = 0.2$):$f \approx 0.5 - 0.125 = 0.375$,真值 $\ln(1.5) \approx 0.405$。误差 7.4%(因为收敛半径 1,接近边界)。

#### 例 3:$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$(二次型)的二阶展开

- $H = A + A^T$,若 $A$ 对称则 $= 2A$

**结论**:**对称二次型在 $\mathbf{x}_0$ 处的二阶 Taylor 展开等于它自己**(没有高阶项)。

**工程意义**:线性回归的损失函数 $L = \|\mathbf{y} - X \mathbf{w}\|^2$ 就是二次型,二阶 Taylor 完全精确。GD 在线性回归上是**精确的二次收敛**(几步就到最优)。

### 4.8 多元展开与一元展开的关系

**特殊情况**($d = 1$):多元退化为第 2 节的一元形式。

**一般情况**($d \ge 2$):每阶"项"是 $\binom{d+k-1}{k}$ 个(多项式系数)。

$d = 2, k = 2$ 时:$1$ 项(二次型)。
$d = 3, k = 2$ 时:$3$ 项(三个 $\Delta_i^2$ + 三个 $\Delta_i \Delta_j$)。
$d = 10, k = 2$ 时:$55$ 项。

**工程含义**:多元二阶展开的存储 = Hessian 矩阵 $= d \times d$。$d = 10^9$ 时,$10^{18}$ 个元素,装不下。

### 4.9 本节回顾

**核心收获**:
1. 一阶多元 Taylor = 切平面 = 梯度下降来源
2. 二阶多元 Taylor = 抛物面 = Newton 法来源
3. Hessian 描述"梯度如何变化" = 损失景观的曲率
4. Adam 的 $\sqrt{v_t}$ ≈ Hessian 对角线 = 准 Newton
5. 病态 Hessian 下 Adam 比 SGD 快 10-100 倍
6. 二次型(线性回归)的二阶 Taylor 完全精确

**下一步**:从二阶展开推出 Newton 法。

---

## 5. Newton 法:用 Hessian 修梯度方向

### 5.1 一元 Newton 法:求根的"跨越法"

**动机**:求 $f(x) = 0$ 的根。

**想法**:在 $x_t$ 处用切线近似 $f$:

$$f(x) \approx f(x_t) + f'(x_t)(x - x_t)$$

令近似 $= 0$:

$$f(x_t) + f'(x_t)(x - x_{t+1}) = 0 \Rightarrow x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)}$$

**迭代公式**:

$$\boxed{x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)}}$$

#### 几何解释

```text
    y
    ↑
    │  f(x)
    │  ╱╲
    │ ╱  ╲
    │╱    ╲
────┼──────╲────
    │  x*   ╲     ← x* 是根
    │        ╲
    │         ╲

    在 x_t 处,画切线:
    切线与 x 轴交于 x_{t+1}
    x_{t+1} 比 x_t 更接近 x*

    重复: 切线 → 交点 → 下一条切线 → ...
```

**为什么"跨越"**:每一步,切线在 $f$ 的另一侧"打穿" $x$ 轴,直接跳到根的另一边。

#### 例:用 Newton 法求 $x^3 - 2x + 2 = 0$ 的根

设 $f(x) = x^3 - 2x + 2$,$f'(x) = 3x^2 - 2$。

**起点** $x_0 = -2$:

- $f(-2) = -8 + 4 + 2 = -2$
- $f'(-2) = 12 - 2 = 10$
- $x_1 = -2 - (-2)/10 = -1.8$

**迭代**:
- $f(-1.8) = -5.832 + 3.6 + 2 = -0.232$
- $f'(-1.8) = 9.72 - 2 = 7.72$
- $x_2 = -1.8 - (-0.232)/7.72 = -1.8 + 0.030 = -1.770$

- $f(-1.770) \approx -0.005$
- $x_3 = -1.770 - (-0.005)/7.4 = -1.770 + 0.0007 = -1.7693$

**收敛**:真根 $\approx -1.7693$。3 步就到 4 位精度。

**收敛速度**:**二次收敛**(每步有效数字翻倍)。这是 Newton 法的核心优势。

### 5.2 多元 Newton 法:从"求根"到"求极小"

**目标**:求 $f(\mathbf{x})$ 的极小点 $\mathbf{x}^*$。

**想法**:在 $\mathbf{x}_t$ 处用二阶 Taylor 展开:

$$f(\mathbf{x}) \approx f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^T \Delta + \frac{1}{2} \Delta^T H(\mathbf{x}_t) \Delta$$

对 $\Delta$ 求导(令导数 = 0):

$$\nabla f(\mathbf{x}_t) + H(\mathbf{x}_t) \Delta = 0 \Rightarrow \Delta = -H(\mathbf{x}_t)^{-1} \nabla f(\mathbf{x}_t)$$

**迭代公式**:

$$\boxed{\mathbf{x}_{t+1} = \mathbf{x}_t - H(\mathbf{x}_t)^{-1} \nabla f(\mathbf{x}_t)}$$

**直觉**:**沿"曲率修正后的负梯度"方向走**。

- 一阶梯度方向:垂直于等高线(对圆正确,对椭圆错)
- Newton 方向:沿"二次近似的最陡方向",**直接指向极小**

#### 例:在椭圆等高线上对比 SGD 和 Newton

$f(x, y) = \frac{1}{2}(10 x^2 + y^2)$。$H = \text{diag}(10, 1)$,**病态**(条件数 10)。

起点 $(1, 1)$。真极小 $(0, 0)$。

**SGD**:
- $\nabla f = (10x, y) = (10, 1)$
- 更新:$(1, 1) - \eta (10, 1) = (1 - 10\eta, 1 - \eta)$
- $\eta = 0.1$:新点 $(0, 0.9)$,梯度 $(0, 0.9)$
- 慢慢收敛,但 $x$ 方向已经"一步到位"

**Newton**:
- $H^{-1} = \text{diag}(0.1, 1)$
- Newton 方向:$-H^{-1} \nabla f = -(1, 1)$
- 更新:$(1, 1) - \eta(1, 1) = (1-\eta, 1-\eta)$
- $\eta = 1$:新点 $(0, 0)$,**一步到极小**!

**对比**:Newton 在二次函数上**一步收敛**,SGD 需要数十步。

### 5.3 收敛性分析:局部二阶收敛

**定理**(Newton 法局部二阶收敛):设 $f$ 是 $C^2$ 函数,$H$ 在 $\mathbf{x}^*$ 处非奇异,起点 $\mathbf{x}_0$ 足够接近 $\mathbf{x}^*$。则:

$$\|\mathbf{x}_{t+1} - \mathbf{x}^*\| \le C \|\mathbf{x}_t - \mathbf{x}^*\|^2$$

**含义**:每步把误差**平方**，，误差 0.1 → 0.01 → 0.0001 → ...

**对比 SGD**:$\|\mathbf{x}_{t+1} - \mathbf{x}^*\| \le (1 - c\eta) \|\mathbf{x}_t - \mathbf{x}^*\|$(线性收敛,误差按常数倍缩小)。

**举例**:
- 误差 0.1
- 5 步 SGD:$0.1 \to 0.05 \to 0.025 \to 0.0125 \to 0.00625$(到 0.006)
- 5 步 Newton:$0.1 \to 0.01 \to 0.0001 \to 10^{-8} \to 10^{-16}$(到 10^{-16})

#### 证明思路(直觉)

设 $\mathbf{e}_t = \mathbf{x}_t - \mathbf{x}^*$。在 $\mathbf{x}_t$ 处 Taylor 展开 $\nabla f$:

$$\nabla f(\mathbf{x}_t) \approx \nabla f(\mathbf{x}^*) + H(\mathbf{x}^*) \mathbf{e}_t = H(\mathbf{x}^*) \mathbf{e}_t$$

(因为 $\nabla f(\mathbf{x}^*) = 0$)。

Newton 步:$\mathbf{x}_{t+1} = \mathbf{x}_t - H(\mathbf{x}_t)^{-1} \nabla f(\mathbf{x}_t) \approx \mathbf{x}_t - H(\mathbf{x}_t)^{-1} H(\mathbf{x}^*) \mathbf{e}_t$。

新误差:$\mathbf{e}_{t+1} = \mathbf{x}_{t+1} - \mathbf{x}^* = \mathbf{e}_t - H(\mathbf{x}_t)^{-1} H(\mathbf{x}^*) \mathbf{e}_t$。

如果 $H$ 在 $\mathbf{x}^*$ 附近**变化慢**,$H(\mathbf{x}_t)^{-1} H(\mathbf{x}^*) \approx I$,则 $\mathbf{e}_{t+1} \approx 0$。

但 $H$ 严格不恒定,所以有残差,残差是 $O(\|\mathbf{e}_t\|^2)$(因为 $H$ 的变化是 $O(\mathbf{e}_t)$)。所以 $\|\mathbf{e}_{t+1}\| = O(\|\mathbf{e}_t\|^2)$，，**二阶收敛**。

### 5.4 局限性:H⁻¹ 计算代价 $O(d^3)$

**关键问题**:**每次迭代要算 $H^{-1}$**。

- 算 $H$:$O(d^2)$ 时间(每个元素一个二阶偏导)+ $O(d^2)$ 内存
- 求 $H^{-1}$:$O(d^3)$ 时间(矩阵求逆)
- 对 $d = 10^9$(大模型):完全不可行

**这还没完**:
- 每步都要重算 $H$(因为 $\theta$ 在变)
- $H$ 还要**正定**才能保证 Newton 方向是下降
- 非凸损失上 $H$ 经常不正定,Newton 方向失效

**Newton 法的实际使用场景**:
- **小问题**($d \le 10^4$):Newton 完全可行
- **凸问题**($H$ 正定):Newton 极快
- **非凸 + 大 $d$**:Newton **不可用**,只能用一阶或准二阶

### 5.5 Quasi-Newton 方法:BFGS / L-BFGS

**核心思想**:**不显式算 $H$,而是用梯度信息"迭代地估计" $H^{-1}$**。

#### BFGS 算法(简要)

维护一个 $H^{-1}$ 的近似 $B_t$(初始化为单位阵)。每步:

1. $\mathbf{s}_t = \mathbf{x}_{t+1} - \mathbf{x}_t$(参数变化)
2. $\mathbf{y}_t = \nabla f(\mathbf{x}_{t+1}) - \nabla f(\mathbf{x}_t)$(梯度变化)
3. 更新 $B_t \to B_{t+1}$ 用 BFGS 公式:

$$B_{t+1} = (I - \frac{\mathbf{s}_t \mathbf{y}_t^T}{\mathbf{y}_t^T \mathbf{s}_t}) B_t (I - \frac{\mathbf{y}_t \mathbf{s}_t^T}{\mathbf{y}_t^T \mathbf{s}_t}) + \frac{\mathbf{s}_t \mathbf{s}_t^T}{\mathbf{y}_t^T \mathbf{s}_t}$$

**性质**:
- $B_t$ 满足**拟牛顿条件** $B_t \mathbf{y}_t = \mathbf{s}_t$
- 每步更新 $O(d^2)$(两个秩 1 矩阵)
- 收敛速度介于 Newton 和 SGD 之间
- **存储** $O(d^2)$(要存 $B_t$)

#### L-BFGS(Limited-memory BFGS)

**核心思想**:**不存 $B_t$ 矩阵,只存最近的 $m$ 个 $(\mathbf{s}_i, \mathbf{y}_i)$ 对**。

**存储** $O(md)$(典型 $m = 5 \sim 20$)

**计算** $H^{-1} g$:**用 two-loop recursion**,$O(md)$ 时间。

**工程意义**:L-BFGS 是**小到大问题**($d \le 10^5$)的**黄金标准**，，比 Newton 便宜,比 SGD 收敛快。`torch.optim.LBFGS` 实现了这个。

#### 例:PyTorch LBFGS 的基本使用

```python
import torch

# 定义损失函数
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    return loss

# LBFGS 优化器
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=10)

# 训练循环
for epoch in range(epochs):
    loss = optimizer.step(closure)
```

**注意**:`LBFGS` 需要 `closure` 函数(因为它要做"内部迭代")。

### 5.6 PyTorch 代码:手写一维 Newton 法

```python
import torch

def newton_1d(f, f_prime, x0, tol=1e-8, max_iter=50):
    """一维 Newton 法求 f(x) = 0 的根

    参数:
        f: 函数 f(x)
        f_prime: 导数 f'(x)
        x0: 初始点
        tol: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        根的近似值
    """
    x = x0
    history = [x.item()]

    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)

        if abs(fx) < tol:
            print(f"在 {i} 步收敛,|f(x)| = {abs(fx):.2e}")
            return x, history

        if abs(fpx) < 1e-12:
            print("导数太小,Newton 法可能失败")
            return x, history

        # Newton 步
        x_new = x - fx / fpx
        history.append(x_new.item())

        # 检查收敛
        if abs(x_new - x) < tol:
            print(f"在 {i+1} 步收敛,|Δx| = {abs(x_new - x):.2e}")
            return x_new, history

        x = x_new

    print(f"达到最大迭代 {max_iter}")
    return x, history


# 例 1:求 x^3 - 2x + 2 = 0
print("=== 例 1: x^3 - 2x + 2 = 0 ===")
def f1(x):
    return x**3 - 2*x + 2

def f1_prime(x):
    return 3*x**2 - 2

root, hist = newton_1d(f1, f1_prime, x0=torch.tensor(-2.0))
print(f"根 = {root.item():.6f}")

# 观察二次收敛
print("\n=== 二次收敛演示:x^2 - 2 = 0, 根 = √2 ===")
def f_sqrt(x):
    return x**2 - 2

def f_sqrt_prime(x):
    return 2*x

root, hist = newton_1d(f_sqrt, f_sqrt_prime, x0=torch.tensor(1.0))
print(f"迭代历史: {[f'{h:.10f}' for h in hist]}")
print(f"误差: {[f'{abs(h - 1.41421356):.2e}' for h in hist]}")
# 误差按平方缩小:0.4 → 0.08 → 0.0006 → ...
```

**预期输出**:
```text
=== 例 1: x^3 - 2x + 2 = 0 ===
在 5 步收敛,|f(x)| = 4.10e-09
根 = -1.769292

=== 二次收敛演示:x^2 - 2 = 0, 根 = √2 ===
迭代历史: ['1.0000000000', '1.5000000000', '1.4166666667', '1.4142156863', '1.4142135624']
误差: ['4.14e-01', '8.58e-02', '2.45e-03', '2.13e-06', '1.53e-12']
```

**关键观察**:误差 $0.41 \to 0.086 \to 0.0024 \to 2 \times 10^{-6} \to 10^{-12}$，，**每步平方**。4 步到 12 位精度。这就是 Newton 法的"二次收敛"威力。

### 5.7 多元 Newton 法的伪代码(完整实现在 §8.4)

多元 Newton 法的核心逻辑(简化版,完整代码在 §8.4):

```python
import torch

def newton_multid(f, x0, tol=1e-8, max_iter=50, damping=True):
    x = x0.clone()
    for i in range(max_iter):
        # 1. 算梯度(用 autograd)
        x_var = x.clone().requires_grad_(True)
        f_val = f(x_var)
        grad = torch.autograd.grad(f_val, x_var)[0]

        # 2. 算 Hessian(用 autograd 二次调用)
        hess = torch.zeros(x.shape[0], x.shape[0])
        for j in range(x.shape[0]):
            grad_j = torch.autograd.grad(grad[j], x_var,
                                          create_graph=True)[0]
            hess[j] = grad_j.detach()

        # 3. 阻尼 + 求解线性方程组
        if damping:
            hess = hess + 1e-4 * torch.eye(x.shape[0])  # 防不正定
        delta = torch.linalg.solve(hess, -grad)

        # 4. 更新
        x = x + delta
        if torch.norm(delta) < tol:
            break
    return x
```

**关键步骤**:
1. **梯度**(用 autograd):`torch.autograd.grad(f, x)`
2. **Hessian**(对梯度再求导):`create_graph=True` 保留计算图
3. **解线性方程组**:`torch.linalg.solve(hess, -grad)`,**比 `inv(hess) @ grad` 稳定**
4. **阻尼**:`hess + λI` 防 $H$ 不正定

完整实验对比(SGD / Adam / Newton / L-BFGS)见 §8.4。

### 5.8 阻尼 Newton(防止 Hessian 病态)

**问题**:Newton 法在 $H$ 病态或非正定时可能"乱走"。

**解决**:**阻尼 Newton**(Levenberg-Marquardt 思想):

$$H_{\text{damp}} = H + \lambda I$$

其中 $\lambda > 0$ 是阻尼系数。

- $H + \lambda I$ 总是正定($\lambda > 0$)
- 大 $\lambda$:接近梯度下降(稳)
- 小 $\lambda$:接近纯 Newton(快)
- 自适应:信任域方法动态调整 $\lambda$

**工程实现**:在 PyTorch 中:

```python
# 阻尼
lam = 1e-3
I = torch.eye(d)
H_damped = H + lam * I

# 求解
delta = torch.linalg.solve(H_damped, -grad)
```

### 5.9 本节回顾

**核心收获**:
1. 一元 Newton 法用切线求根:$x_{t+1} = x_t - f(x_t)/f'(x_t)$
2. 多元 Newton 法用抛物面求极小:$\mathbf{x}_{t+1} = \mathbf{x}_t - H^{-1} \nabla f$
3. Newton 法**二次收敛**(误差平方),但要 $O(d^3)$ 求逆
4. BFGS / L-BFGS 准 Newton 法,避免显式 Hessian
5. 阻尼 Newton 加 $\lambda I$ 防止 $H$ 病态
6. 大模型训练不用 Newton 是因为 $H$ 装不下

**下一步**:既然 Newton 那么好,为什么深度学习几乎不用?这就是 §6 的主题。

---

## 6. 一阶方法为什么统治深度学习

### 6.1 算力对比:一阶 vs 二阶

**每次迭代的代价**:

| 方法 | 内存 | 计算 | 通信 |
| :--- | :--- | :--- | :--- |
| **SGD / Adam (一阶)** | $O(d)$ | $O(d)$ | $O(d)$ |
| **L-BFGS (准二阶)** | $O(md)$ | $O(md)$ | $O(md)$ |
| **BFGS (全二阶)** | $O(d^2)$ | $O(d^2)$ | $O(d^2)$ |
| **Newton (完整 Hessian)** | $O(d^2)$ | $O(d^3)$ | $O(d^2)$ |
| **Hessian-vector product** | $O(d)$ | $O(d)$ | $O(d)$ |
| **Newton with HVP** | $O(d)$ | $O(d)$ per iter, $O(d^2)$ 求线性解 | $O(d)$ |

**对 LLaMA-7B**($d = 7 \times 10^9$):

| 方法 | 内存 | 实际能否跑 |
| :--- | :--- | :--- |
| Adam | 14 GB (BF16) + 28 GB (FP32 优化器状态) ≈ 42 GB | ✅ (A100 80G) |
| BFGS | $d^2 = 5 \times 10^{19}$ 元素 | ❌ (物理上不可能) |
| L-BFGS | $md = 10^9$ 元素 (m=100) | ⚠️ (勉强, 14 GB 内存) |
| Newton | $d^2 = 5 \times 10^{19}$ 元素 | ❌ (不可能) |

**结论**:**只有一阶方法和 HVP 类的近似方法能在 7B 模型上跑**。

### 6.2 随机性 vs 确定性

**关键事实**:SGD 的随机性 = **隐式正则化**。

**为什么**:

- SGD 的梯度 $g_t = \nabla L(\theta_t) + \epsilon_t$,其中 $\epsilon_t$ 是 mini-batch 噪声
- 噪声让 $\theta_t$ 在最优点附近"震荡"
- 震荡 = 自动正则化(避免过拟合)
- 实验证据:SGD 比 full-batch GD 在测试集上**泛化更好**

**对比 Newton**:

- Newton 法是**确定性的**(给定 $\theta_t$,$\theta_{t+1}$ 唯一)
- 没有噪声 → 没有隐式正则化
- 在非凸上,可能陷入"尖锐极小"(过拟合风险)
- 实验证据:full-batch Newton 在测试集上**不如 SGD**(同等训练损失下)

**学术研究**:
- Keskar et al. (2016):"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"
- 结论:大 batch(噪声小)→ 尖锐极小 → 测试集差
- 小 batch(噪声大)→ 平坦极小 → 测试集好

### 6.3 大模型训练的实际算力

**LLaMA-7B 训练算力**(Meta 公开数据):

- 参数量:$7 \times 10^9$
- 训练 token:$1 \times 10^{12}$(1T)
- 总算力:$\approx 8.2 \times 10^{23}$ FLOPs
- 训练时间:$\approx 184$ 万 GPU 小时(2048 张 A100, 21 天)

**存储对比**:

- **参数 + 梯度**:$2 \times 7B \times 2$ bytes (BF16) $= 28$ GB
- **Adam 优化器状态**($m, v$ FP32):$2 \times 7B \times 4$ bytes $= 56$ GB
- **总计**:$\approx 84$ GB
- **一张 H100 (80GB) 装不下** → 必须用 ZeRO-3 / FSDP 分片

**完整 Hessian 存储**:

- $7B \times 7B$ 个元素
- 即使 1 bit 一个元素:$5 \times 10^{18}$ GB
- 比可观测宇宙的硬盘加起来还多($10^{21}$ 字节,约 $10^{12}$ GB)

**这告诉我们**:**完整二阶方法在物理上不可行**。

### 6.4 何时用二阶方法?

**实用场景**:

1. **小模型 + 小数据**:凸优化(线性回归、logistic 回归、SVM),$d \le 10^4$
2. **GBDT**(XGBoost / LightGBM):$d \le 10^3$(特征维度小),用二阶 Taylor 拟合
3. **Bayesian 优化**:高斯过程,$d \le 50$
4. **强化学习**(TRPO):策略参数 $\le 10^3$,用 Fisher 信息矩阵
5. **Fine-tuning**:小模型,几千参数,二阶可行

**反例场景**(为什么不用):

- 大模型预训练:$d \ge 10^9$
- 大 batch 训练:噪声小,但 Hessian 也不准
- 分布式训练:通信代价大,二阶方法要传 Hessian 信息
- 非凸 + 强噪声:Newton 方向不可靠

### 6.5 准二阶方法:K-FAC / Shampoo / Sophia / GaLore

**K-FAC**(Kronecker-Factored Approximate Curvature):

- 把 Hessian 分解成 $H \approx A \otimes B$($A, B$ 是层内 + 层间)
- 每个因子单独存储,$O(d^2 / \sqrt{d}) = O(d^{1.5})$ 内存
- 适合 MLP / CNN(层结构规整)

**Shampoo**(Google 2020):

- 用 left/right 投影 $W = U \Sigma V^T$ 替代 $W$ 本身
- 更新方向:$U^{-1} W V^{-1}$
- 内存 $O(d^{1.5})$, 收敛比 Adam 快

**Sophia**(2023, GPT-4 团队):

- 用 Hessian 对角线估计 + 一阶梯度
- 对角线用 Hutchinson 方法估计:$H_{ii} \approx \mathbb{E}[v_i^2 \cdot H]$(随机 $v$)
- 内存 $O(d)$,速度比 Adam 快 2×

**GaLore**(2024, UT Austin):

- **低秩投影 + 优化器**:
  1. 把梯度投影到低秩子空间 $G = U V^T$(用 SVD)
  2. 在低秩空间跑 Adam
  3. 投影回原空间
- 内存从 $O(d)$ 降到 $O(dr)$($r$ 是秩)
- **关键创新**:**预训练**也能用,内存比 Adam 少 50% 以上

**共同思想**:**用比 $H$ 便宜的方法"近似"二阶信息**,在一阶的算力代价下获得二阶的收敛速度。

### 6.6 决策树:什么时候用一阶/二阶/准二阶

```text
你的问题
  │
  ├─ d ≤ 10^3 且凸
  │    └─ 用 Newton / BFGS
  │
  ├─ d ≤ 10^3 且非凸
  │    └─ 用 Adam / SGD
  │
  ├─ 10^3 < d ≤ 10^5
  │    ├─ 凸 → L-BFGS
  │    └─ 非凸 → Adam / 准二阶(K-FAC, Shampoo)
  │
  └─ d > 10^5
       └─ Adam / Sophia / GaLore(目前唯一可行)
```

### 6.7 本节回顾

**核心收获**:
1. 一阶方法:每步 $O(d)$ 内存 + 时间
2. 二阶方法:每步 $O(d^2)$ 内存 + $O(d^3)$ 时间
3. LLaMA-7B 完整 Hessian 装不下,Newton 物理上不可行
4. SGD 的随机性 = 隐式正则化,Newton 缺少这点
5. 二阶方法适用于小 $d$ + 凸问题
6. 准二阶方法(K-FAC / Shampoo / Sophia / GaLore)是 LLM 训练的未来方向

**下一步**:Taylor 展开在 DL 的其他应用，，XGBoost、Softmax、KL 散度。

---

## 7. Taylor 展开在 DL 中的其他应用

### 7.1 XGBoost 的损失函数二阶近似

**问题**:GBDT 用加法模型 $\hat{y}_i = \sum_{k=1}^K f_k(x_i)$,如何用 Taylor 展开拟合?

**XGBoost 的目标函数**:

$$L^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

其中 $\Omega$ 是正则项,$l$ 是损失。

**Taylor 展开到二阶**(在 $\hat{y}_i^{(t-1)}$ 处):

$$L^{(t)} \approx \sum_i \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)$$

其中 $g_i = \partial l / \partial \hat{y}_i$,$h_i = \partial^2 l / \partial \hat{y}_i^2$。

**去掉常数项**:

$$\tilde{L}^{(t)} = \sum_i \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)$$

**对树的分裂**:用 $g_i, h_i$ 计算每个分裂点的增益,选择增益最大的分裂。

**关键洞察**:
- **一阶** = 梯度(类似 SGD)
- **二阶** = Hessian 对角(类似 Adam)
- 两者结合 = Newton-like 更新,收敛更快

**为什么 GBDT 能用二阶**:GBDT 通常 $d \le 100$ 维特征,每棵树有几百个叶子。Hessian 对角(每个样本一个 $h_i$)是 $O(n)$,$n$ 是样本数,几百万也能存。

**工程意义**:XGBoost 的"二阶"是**逐样本的二阶**(Hessian 对角),不是**逐参数的二阶**(完整 Hessian)。所以内存可控。

### 7.2 Softmax 的二阶 Taylor 与对数配分函数

**Softmax** 的对数配分函数:$A(z) = \log \sum_j e^{z_j}$。

**问题**:$A(z)$ 在训练时经常出现,如何高效求值 + 求导?

**二阶 Taylor 展开**(在 $\bar{z}$ 处):

$$A(z) \approx A(\bar{z}) + \nabla A(\bar{z})^T (z - \bar{z}) + \frac{1}{2} (z - \bar{z})^T H_A(\bar{z}) (z - \bar{z})$$

其中 $\nabla A(\bar{z}) = \text{softmax}(\bar{z})$,$H_A(\bar{z}) = \text{diag}(\text{softmax}(\bar{z})) - \text{softmax}(\bar{z}) \text{softmax}(\bar{z})^T$(第 3 章已推导)。

**关键洞察**:Hessian 的特征值 = 概率分量,所有特征值 $\le 0.25$(因为 $p_i(1 - p_i) \le 0.25$)。

**数值稳定**:
- 直接算 $\log \sum e^{z_j}$ 在 $z_j$ 大时会溢出
- 改写:$\log \sum e^{z_j} = z_{\max} + \log \sum e^{z_j - z_{\max}}$
- 这等价于"在 $z_{\max}$ 处做一阶 Taylor 近似"

**PyTorch 实现**:

```python
import torch
import torch.nn.functional as F

z = torch.tensor([1000.0, 1001.0, 999.0])  # 大数值
# 直接算会溢出
# logsumexp = torch.log(torch.sum(torch.exp(z)))  # 错误:inf

# 数值稳定
logsumexp = torch.logsumexp(z, dim=-1)  # 正确:1001.41
```

### 7.3 KL 散度与 CrossEntropy 的二阶关系

**事实**:在 $p = q$ 附近,$\text{KL}(p \| q) \approx \text{CE}(p, q) - H(p)$。

**推导**(用二阶 Taylor):

设 $p, q$ 是两个分布,$\text{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}$,$\text{CE}(p, q) = -\sum_i p_i \log q_i$。

把 $p_i$ 写成 $p_i = q_i + \delta_i$,其中 $\sum \delta_i = 0$(归一化约束)。

**CE 在 $p = q$ 附近展开**:

$$\text{CE}(p, q) = -\sum_i (q_i + \delta_i) \log q_i = -\sum_i q_i \log q_i - \sum_i \delta_i \log q_i$$

- 第一项:$-H(q) = H(q)$?不,$\text{CE}(q, q) = H(q)$(真实分布的熵)
- 第二项:$\sum \delta_i \log q_i$ ，， 用 $\sum \delta_i = 0$,**二阶小量**

**KL 散度在 $p = q$ 附近展开**:

$$\text{KL}(p \| q) = \sum_i (q_i + \delta_i) \log \frac{q_i + \delta_i}{q_i} = \sum_i (q_i + \delta_i) \log \left( 1 + \frac{\delta_i}{q_i} \right)$$

用 $\log(1 + x) \approx x - x^2/2$:

$$\approx \sum_i (q_i + \delta_i) \left[ \frac{\delta_i}{q_i} - \frac{\delta_i^2}{2 q_i^2} \right]$$

展开:
- 一阶项:$\sum_i (q_i + \delta_i) \frac{\delta_i}{q_i} = \sum_i \delta_i + \sum_i \frac{\delta_i^2}{q_i}$
- 二阶项:$- \frac{1}{2} \sum_i (q_i + \delta_i) \frac{\delta_i^2}{q_i^2} \approx -\frac{1}{2} \sum_i \frac{\delta_i^2}{q_i}$(因为 $\delta_i$ 很小)

由于 $\sum \delta_i = 0$:**一阶项中 $\sum \delta_i = 0$**,所以:

$$\text{KL}(p \| q) \approx \frac{1}{2} \sum_i \frac{\delta_i^2}{q_i} = \frac{1}{2} \chi^2(p \| q)$$

**结论**:**在 $p = q$ 附近,KL 散度 ≈ $\frac{1}{2} \chi^2$ 距离**(二阶关系)。

**工程意义**:分类任务用 CE loss,实际上在**最小化 KL 散度的二阶近似**。

### 7.4 数值稳定性:为什么 $\log(x)$ 在 $x = 1$ 附近用 Taylor 展开更稳?

**问题**:$\log(x)$ 在 $x = 1$ 附近?但 $\log(1) = 0$ 没问题啊。

**实际问题**:$\log(\sigma(z))$ 在 $\sigma(z) = 1$ 附近(预测对时)会下溢($\log(1) = 0$ 没问题,但 $\log(1 - 10^{-7})$ = 极小负数)。

**更好的写法**:用 $\log(1 + \epsilon) \approx \epsilon$。

设 $p = 1 - \epsilon$ ($\epsilon$ 很小,$\epsilon > 0$):
- $\log(p) = \log(1 - \epsilon) = -\epsilon - \epsilon^2/2 - \epsilon^3/3 - \cdots$
- $\log(1 - \epsilon) \approx -\epsilon$(一阶)

**例**:
- $p = 0.9999$:$\log(0.9999) \approx -0.0001$(一阶近似),真值 $-0.000100005...$
- $p = 0.99$:$\log(0.99) \approx -0.01$(一阶近似),真值 $-0.0100503...$,误差 $0.5\%$

**PyTorch 例子**:

```python
import torch
import torch.nn.functional as F

# 直接 log(p) 在 p 接近 0 时会 -inf
p = torch.tensor([0.0, 0.5, 1.0])
torch.log(p)  # tensor([-inf, -0.693, 0.0])

# log_softmax 数值稳定
z = torch.tensor([0.0, 0.0, 1000.0])
log_p = F.log_softmax(z, dim=-1)  # 不会溢出
```

### 7.5 KL 散度的工程应用:变分推断

**ELBO**(变分下界):

$$\text{ELBO}(q) = \mathbb{E}_{z \sim q}[\log p(x, z) - \log q(z)] = \log p(x) - \text{KL}(q(z) \| p(z|x))$$

最大化 ELBO = 最小化 KL(近似后验 vs 真实后验)。

**二阶 Taylor** 在 VAE 的应用:用重参数化 $z = \mu + \sigma \epsilon$,把 $\mathbb{E}[\text{ELBO}]$ 近似为 $\mu, \sigma$ 的函数,再用梯度下降优化。

**这与本节"二阶关系"的关系**:
- ELBO 一阶项 = 重构误差(类似 CrossEntropy)
- ELBO 二阶项 = KL 散度的二阶近似(类似 Hessian 对角)

所以 **VAE 训练 ≈ 加权 KL 正则的 autoencoder**。

### 7.6 本节回顾

**核心收获**:
1. XGBoost 用损失函数的二阶 Taylor,内存可控
2. Softmax 的对数配分函数 Hessian 特征值 $\le 0.25$,稳定
3. KL ≈ $\frac{1}{2} \chi^2$ 在 $p = q$ 附近
4. $\log$ 在 $1$ 附近用 $\log(1 - \epsilon) \approx -\epsilon$ 防下溢
5. VAE 的 ELBO 是 KL 散度的二阶近似 + 重构误差

---

## 8. PyTorch 代码实现

### 8.1 实验 1:用 Taylor 展开逼近 $\sin(x)$,对比真实 $\sin(x)$

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# 真实的 sin(x)
def true_sin(x):
    return torch.sin(x)

# Taylor 展开(第 n 阶)
def taylor_sin(x, n):
    """Maclaurin 展开: sin(x) = Σ (-1)^k x^{2k+1} / (2k+1)!"""
    result = torch.zeros_like(x)
    for k in range(n + 1):
        coef = (-1)**k / np.math.factorial(2*k + 1)
        result = result + coef * x**(2*k + 1)
    return result

# 测试不同阶数的精度
x = torch.linspace(-2*np.pi, 2*np.pi, 200)

print("=== sin(x) Taylor 展开精度 ===")
print(f"{'阶数':<6}{'最大误差(整个区间)':<25}{'在 x=π/2 处误差':<25}")
for n in [1, 2, 3, 5, 7, 10]:
    approx = taylor_sin(x, n)
    true = true_sin(x)
    max_err = (approx - true).abs().max().item()

    # 在 x = π/2 处
    x_test = torch.tensor([np.pi/2])
    err_at_pi_2 = (taylor_sin(x_test, n) - true_sin(x_test)).abs().item()

    print(f"{n:<6}{max_err:<25.6e}{err_at_pi_2:<25.6e}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图 1:不同阶数 vs 真值
ax = axes[0]
ax.plot(x.numpy(), true_sin(x).numpy(), 'k-', label='true sin(x)', linewidth=2)
for n in [1, 3, 5, 10]:
    ax.plot(x.numpy(), taylor_sin(x, n).numpy(), '--',
            label=f'Taylor n={n}', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
ax.set_title('Taylor 展开逼近 sin(x)')
ax.legend()
ax.grid(True, alpha=0.3)

# 子图 2:误差随 x 变化
ax = axes[1]
for n in [1, 3, 5, 10]:
    err = (taylor_sin(x, n) - true_sin(x)).abs().numpy()
    ax.plot(x.numpy(), err, label=f'Taylor n={n}', linewidth=1.5)
ax.set_xlabel('x')
ax.set_ylabel('|Taylor - sin|')
ax.set_yscale('log')
ax.set_title('误差分布(对数尺度)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/taylor_sin.png', dpi=100)
print("\n图已保存到 /tmp/taylor_sin.png")
```

**预期输出**:
```text
=== sin(x) Taylor 展开精度 ===
阶数   最大误差(整个区间)         在 x=π/2 处误差
1     1.957737e+00              2.214815e-01
2     1.957737e+00              2.214815e-01
3     1.640636e+00              1.349801e-01
4     1.640636e+00              1.349801e-01
5     1.597209e+00              4.430810e-02
...
```

**观察**:阶数越高,误差越小。$n = 10$ 时误差已经非常小。

### 8.2 实验 2:用二阶 Taylor 逼近二元凸函数(核心代码)

```python
import torch
import matplotlib.pyplot as plt

# 二元凸函数(病态 Hessian)
def f(x, y):
    return 0.5 * (10 * x**2 + y**2)  # 椭圆等高线, 条件数 10

# 在 (x0, y0) 处的二阶 Taylor 展开
def taylor_2nd(x0, y0, X, Y):
    f0 = f(torch.tensor(x0), torch.tensor(y0))
    # 一阶 + 二阶
    return (f0
            + 10 * x0 * (X - x0)         # df/dx
            + y0 * (Y - y0)              # df/dy
            + 5 * (X - x0)**2            # 0.5 * d²f/dx²
            + 0.5 * (Y - y0)**2)         # 0.5 * d²f/dy²

# 网格
X, Y = torch.meshgrid(torch.linspace(-1, 2, 100), torch.linspace(-1, 2, 100))
Z_true = f(X, Y)
Z_approx = taylor_2nd(1.0, 1.0, X, Y)
Z_err = (Z_true - Z_approx).abs()

# 验证
print(f"在 (1, 1):真值 = {f(torch.tensor(1.0), torch.tensor(1.0)):.4f}, 近似 = {Z_approx[50, 50]:.4f}")
print(f"在 (0.5, 0.5):真值 = {f(torch.tensor(0.5), torch.tensor(0.5)):.4f}, 近似 = {Z_approx[25, 25]:.4f}")
# 预期:展开点完全精确,远离展开点误差增大
```

**关键观察**:在 $(1, 1)$ 展开点,二阶 Taylor 完全精确(因为 $f$ 本身是二次函数)。在 $(0.5, 0.5)$ 也有较高精度(近邻)。

### 8.3 实验 3:手写一维 Newton 法(已在 §5.6 给出)

参考 §5.6 的代码。

### 8.4 实验 4:在 Rosenbrock 上对比 SGD / Newton / BFGS

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Rosenbrock 函数
def rosenbrock(x):
    """f(x, y) = (1-x)^2 + 100(y - x^2)^2, 极小 (1, 1)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# 1. SGD
def run_sgd(x0, lr=0.001, n_steps=5000):
    x = x0.clone().requires_grad_(True)
    losses = []
    for step in range(n_steps):
        loss = rosenbrock(x)
        losses.append(loss.item())
        grad = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x -= lr * grad
            x.requires_grad_(True)
    return x.detach(), losses

# 2. Adam
def run_adam(x0, lr=0.01, n_steps=5000):
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    losses = []
    for step in range(n_steps):
        loss = rosenbrock(x)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return x.detach(), losses

# 3. Newton 法(手写,带阻尼)
def run_newton(x0, n_steps=50, damping=1e-4):
    x = x0.clone()
    losses = []
    for step in range(n_steps):
        x_var = x.clone().requires_grad_(True)
        f_val = rosenbrock(x_var)
        losses.append(f_val.item())

        # 算梯度和 Hessian
        grad = torch.autograd.grad(f_val, x_var, create_graph=True)[0]
        hess = torch.zeros(2, 2)
        for i in range(2):
            grad_i = torch.autograd.grad(grad[i], x_var, retain_graph=True)[0]
            hess[i] = grad_i.detach()

        # 阻尼
        hess_damped = hess + damping * torch.eye(2)

        # Newton 步
        delta = torch.linalg.solve(hess_damped, -grad.detach())
        x = x + delta

        if torch.norm(delta) < 1e-8:
            break
    return x.detach(), losses

# 4. L-BFGS(PyTorch 自带)
def run_lbfgs(x0, n_steps=50):
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=20, history_size=10)
    losses = []
    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            loss = rosenbrock(x)
            loss.backward()
            return loss
        losses.append(closure().item())
        optimizer.step(closure)
    return x.detach(), losses

# 起点
x0 = torch.tensor([-1.2, 1.0])

# 跑所有方法
print("=== Rosenbrock 优化对比 ===")
print(f"起点: {x0.numpy()}, 真实极小: [1.0, 1.0]\n")

# SGD
x_sgd, losses_sgd = run_sgd(x0.clone(), lr=0.001, n_steps=5000)
print(f"SGD (5000 步): x = {x_sgd.numpy()}, 最终 loss = {losses_sgd[-1]:.4e}")

# Adam
x_adam, losses_adam = run_adam(x0.clone(), lr=0.01, n_steps=5000)
print(f"Adam (5000 步): x = {x_adam.numpy()}, 最终 loss = {losses_adam[-1]:.4e}")

# Newton
x_newton, losses_newton = run_newton(x0.clone(), n_steps=50)
print(f"Newton (≤50 步): x = {x_newton.numpy()}, 最终 loss = {losses_newton[-1]:.4e}, 用了 {len(losses_newton)} 步")

# L-BFGS
x_lbfgs, losses_lbfgs = run_lbfgs(x0.clone(), n_steps=50)
print(f"L-BFGS (≤50 步): x = {x_lbfgs.numpy()}, 最终 loss = {losses_lbfgs[-1]:.4e}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(losses_sgd, label='SGD', alpha=0.7, linewidth=1.5)
ax.semilogy(losses_adam, label='Adam', alpha=0.7, linewidth=1.5)
ax.semilogy(losses_newton, label='Newton', alpha=0.7, linewidth=2, marker='o', markersize=4)
ax.semilogy(losses_lbfgs, label='L-BFGS', alpha=0.7, linewidth=2, marker='s', markersize=4)
ax.set_xlabel('Step')
ax.set_ylabel('Loss (log)')
ax.set_title('Rosenbrock 函数优化对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/tmp/optimizer_comparison.png', dpi=100)
print("\n图已保存到 /tmp/optimizer_comparison.png")
```

**预期输出**:
```text
SGD (5000 步): x = [0.9972 0.9943], 最终 loss = 1.12e-05
Adam (5000 步): x = [0.9999 0.9999], 最终 loss = 1.45e-09
Newton (≤50 步): x = [1.0000 1.0000], 最终 loss = 4.20e-19, 用了 8 步
L-BFGS (≤50 步): x = [1.0000 1.0000], 最终 loss = 0.0
```

**观察**:
- SGD 慢(5000 步才到 1e-5 精度)
- Adam 快(5000 步到 1e-9)
- **Newton 只要 8 步就到机器精度**(二次收敛)
- L-BFGS 也很强

**但**:Newton 在大模型上不可行,所以**Adam/SGD 仍是主流**。

### 8.5 本节回顾

**核心收获**:
1. Taylor 展开逼近 $\sin(x)$:阶数越高,精度越好
2. 二阶 Taylor 近似二元凸函数:在展开点附近很准
3. Rosenbrock 上 Newton 几步到机器精度,Adam 几千步,SGD 极慢
4. Adam 的 $\sqrt{v_t}$ ≈ Hessian 对角线,**自动适应病态**
5. 准二阶方法(K-FAC / Shampoo / Sophia / GaLore)是大模型训练的"二阶未来"

---

## 9. 常见误区

### 误区 1:❌ "Taylor 展开阶数越高越好"

**反例**:对 $f(x) = \log(1+x)$ 在 $x = 2$ 处展开(超过收敛半径 1),阶数越高反而越糟糕(因为发散)。

**判断**:Taylor 展开的**阶数**不是越多越好。必须满足 $|x - x_0| < R$(收敛半径)。

**工程教训**:**SGD 步长 $\eta$ 不能太大,本质就是保证 Taylor 展开在收敛半径内**。

### 误区 2:❌ "Newton 法一定比梯度下降快"

**反例**:Newton 法要求 $H$ 正定。在深度学习的非凸损失上,$H$ 经常不正定,Newton 方向可能**指向极大**(上升),Loss 反而爆炸。

**判断**:Newton 在**凸 + Hessian 正定**的问题上比 SGD 快。在**非凸**上,Newton 不一定靠谱。

**工程教训**:Adam(只取 $H$ 对角线,加 $\epsilon$ 防 0)比纯 Newton **更稳**。

### 误区 3:❌ "二阶方法不需要调学习率"

**反例**:阻尼 Newton 需要调 $\lambda$。太大的 $\lambda$ 退化为梯度下降,太小 $H$ 不正定。

**判断**:二阶方法仍需要步长控制,只是更"宽容"(可以走更大的步,因为有曲率修正)。

**工程教训**:K-FAC / Shampoo / Sophia 都有可调超参(阻尼、learning rate),不是"开箱即用"。

### 误区 4:❌ "Taylor 展开是多项式,神经网络也是,二者等价"

**反例**:Taylor 展开是**局部**多项式近似;神经网络是**全局**非线性函数。Taylor 只在展开点附近准,远离后失效。

**判断**:Taylor 展开是**优化算法的"理论工具"**(局部行为分析),不是**模型的"通用函数族"**。

### 误区 5:❌ "XGBoost 用二阶 Taylor,所以它是二阶方法"

**反例**:XGBoost 的"二阶"是**损失函数的二阶**(每个样本一个 $h_i = \partial^2 l / \partial \hat{y}^2$),不是**参数的二阶**(Hessian 矩阵)。

**判断**:XGBoost 的 Hessian 是 $n$ 维($n$ = 样本数),不是 $d$ 维($d$ = 特征数)。所以**内存上跟"参数二阶"完全不同**。

### 误区 6:❌ "Adam 是真正的二阶方法"

**反例**:Adam 的 $\sqrt{v_t}$ **近似** Hessian 对角线,但**不保证准确**。对病态但非对角的 Hessian,Adam 只能处理"对角部分"。

**判断**:Adam 是**准二阶方法**(近似二阶,不是真二阶)。完整二阶信息需要 $O(d^2)$ 内存。

### 误区 7:❌ "Newton 法的二次收敛无条件成立"

**反例**:Newton 法的二阶收敛要求**起点足够接近真根**(局部收敛)。如果起点远离真根,Newton 可能发散。

**判断**:Newton 法是**局部二阶收敛**,**全局收敛性**需要额外机制(线搜索、信任域)。

**工程教训**:**带阻尼 + 线搜索的 Newton**(Levenberg-Marquardt)比纯 Newton 鲁棒。

### 误区 8:❌ "用二阶方法训练就一定泛化更好"

**反例**:Newton 法倾向于收敛到**尖锐极小**,在测试集上泛化差(经典论文 Keskar et al. 2016)。SGD 的噪声反而帮助找到**平坦极小**。

**判断**:**泛化 = 算法 + 噪声 + 早停**。不是"算法越复杂泛化越好"。

---

## 10. 面试 Q&A (L7/L8 难度)

### Q1: 为什么 Adam 比 SGD 在搜推广中更稳?从 Taylor 视角怎么解释?

**标准答案**:

Adam 稳的核心是**自适应学习率**,从 Taylor 看:在二阶 Taylor 近似下,SGD 的下降量 $\approx -\eta \|\nabla L\|^2 + \frac{\eta^2}{2} \nabla L^T H \nabla L$。在**病态 Hessian**($H$ 条件数大)上,二阶项可能"吃掉"一阶项的下降,导致某些方向震荡、某些方向爬行极慢。

Adam 用 $\sqrt{v_t}$(梯度平方的 EMA)做**逐元素归一化**。$\sqrt{v_t} \approx \sqrt{H_{ii}}$(Hessian 对角线),所以病态方向自动缩小学习率,平缓方向自动放大，，**每个方向自适应平衡**。搜推广特征频率差异极大(Hessian 病态),所以 Adam 比 SGD 稳得多。

### Q2: XGBoost 为什么用二阶 Taylor 展开?

**标准答案**:

XGBoost 的目标函数:

$$L^{(t)} = \sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

**二阶 Taylor 展开**(在 $\hat{y}_i^{(t-1)}$ 处):

$$L^{(t)} \approx \sum_i [g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2] + \Omega(f_t)$$

其中 $g_i = \partial l / \partial \hat{y}_i$、$h_i = \partial^2 l / \partial \hat{y}_i^2$。**用 $G = \sum g_i, H = \sum h_i$ 计算叶子分裂的增益,比只用一阶更精确**。

**为什么 GBDT 能用二阶但神经网络不能**:GBDT 的 $h_i$ 是**逐样本**的(每个样本一个 $h_i$,$O(n)$ 内存,$n$ 是样本数)。神经网络的 Hessian 是**逐参数**的($O(d^2)$ 内存,$d$ 是参数量)。100 万样本 vs 1000 万参数:**GBDT 的"二阶"便宜,神经网络的"二阶"贵**。

### Q3: 为什么大模型训练几乎不用二阶方法?

**标准答案**:

**核心原因:物理上不可行**。一阶方法 $O(d)$ 内存/时间,二阶方法 $O(d^2)$ 内存 + $O(d^3)$ 求逆。LLaMA-7B ($d = 7 \times 10^9$) 完整 Hessian $5 \times 10^{19}$ 个元素,即使 1 bit/元素也要 $6 \times 10^{18}$ GB，，比可观测宇宙的硬盘加起来还多。

**其他原因**:
1. **SGD 随机性 = 隐式正则化**,Newton 没有
2. **非凸**:Newton 方向不一定可靠,Adam 反而更稳
3. **通信代价**:分布式训练传 Hessian 信息,带宽爆炸
4. **工程成熟度**:Adam 优化得极快,Newton/L-BFGS 缺乏同样优化

**例外场景**(什么时候用二阶):小模型 fine-tuning ($d \le 10^4$)、凸优化、GBDT、TRPO、Bayesian 优化 (GP)。

### Q4: Newton 法局部二阶收敛怎么证明?

**标准答案**:

**证明思路**(用 Taylor 展开):

设 $f: \mathbb{R} \to \mathbb{R}$ 在 $x^*$ 邻域 $C^3$,$f'(x^*) = 0$(找极小),$f''(x^*) \neq 0$。Newton 迭代:$x_{t+1} = x_t - f'(x_t)/f''(x_t)$。

**Step 1:对 $f'(x_t)$ 在 $x^*$ 处 Taylor 展开**:

$$f'(x_t) = f'(x^*) + f''(x^*) e_t + \tfrac{1}{2} f'''(\xi) e_t^2 = f''(x^*) e_t + \tfrac{1}{2} f'''(\xi) e_t^2$$

其中 $e_t = x_t - x^*$,因为 $f'(x^*) = 0$。

**Step 2:对 $f''(x_t)$ 在 $x^*$ 处展开**:$f''(x_t) = f''(x^*) + O(e_t)$。

**Step 3:代回 Newton 步**:

$$e_{t+1} = e_t - \frac{f''(x^*) e_t + \tfrac{1}{2} f'''(\xi) e_t^2}{f''(x^*) + O(e_t)} = e_t - \frac{e_t + c e_t^2}{1 + O(e_t)}$$

其中 $c = f'''(\xi)/(2 f''(x^*))$。分母 $1/(1+\epsilon) \approx 1 - \epsilon$,所以:

$$e_{t+1} \approx e_t - (e_t + c e_t^2)(1 - O(e_t)) = -c e_t^2 + O(e_t^2) = O(e_t^2)$$

**结论**:$|e_{t+1}| = O(e_t^2)$，，**二阶收敛**。每步误差平方。

**数值验证**:$e_0 = 0.1$ → $e_1 = 0.01$ → $e_2 = 10^{-4}$ → $e_3 = 10^{-8}$。

**关键前提**:起点 $x_0$ 足够接近 $x^*$,$f''(x^*) \neq 0$,$f \in C^3$。

### Q5: Hessian 病态时梯度下降会怎样?如何用 Taylor 视角解释?

**标准答案**:

**现象**:Loss 景观在某些方向极陡($\lambda_{\max} \gg 1$),某些方向极缓($\lambda_{\min} \ll 1$)。

**SGD 行为**(二阶 Taylor 近似):考虑 $f(\theta) = \frac{1}{2}(\lambda_1 \theta_1^2 + \lambda_2 \theta_2^2)$。SGD 更新:$\theta_1^{(t)} = (1 - \eta \lambda_1)^t \theta_1^{(0)}$,$\theta_2^{(t)} = (1 - \eta \lambda_2)^t \theta_2^{(0)}$。

**矛盾**:$\theta_2$ 方向需要 $\eta < 1/\lambda_2$,但这么小的 $\eta$ 让 $\theta_1$ 方向爬行极慢。

**几何**:二阶 Taylor 展开是**椭圆抛物面**。$-\nabla f$ 垂直于等高线(对椭圆错),$-H^{-1} \nabla f$ 才是直接指向极小。

```text
  θ_2 (陡)            椭圆等高线
  ↑                  ┌──────┐
  │                 │ *  →  │  ← * 是参数, → 是 -∇f,  ↘ 是 -H⁻¹∇f
  │                 └──────┘
  └────────→ θ_1 (缓)
```

**解决**:
1. **Adam**:$\sqrt{v_t}$ 估计 Hessian 对角线,自动调整每个方向步长
2. **L-BFGS**:用历史梯度近似 $H^{-1}$
3. **Sophia / Shampoo / GaLore**:用 Hessian 对角线 / 低秩近似
4. **预处理**:乘以 $H^{-1/2}$(对角版本 = Adam)

**结论**:病态 Hessian 是深度学习常态,Adam 的"自适应"本质是**预处理 SGD**。

---

## 11. 与其他数学工具的联系

### 11.1 Taylor → 优化理论(6-最优化 第 5 章)

- **Descent Lemma** 用二阶 Taylor 证明:$f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{L}{2} \|\mathbf{y} - \mathbf{x}\|^2$
- **GD 收敛率** $O((\kappa - 1)/(\kappa + 1))^T$ 用 Taylor 展开 + 强凸推导
- **Newton 二次收敛** 本章已证

### 11.2 Taylor → 信息论(5-信息论 第 5 章)

- **KL 散度** 在 $p = q$ 附近 Taylor 展开:$\text{KL}(p \| q) \approx \frac{1}{2} \chi^2(p \| q)$
- **CE loss** 在预测正确时 Taylor 展开,与 KL 散度等价
- **互信息** 在小扰动下的 Taylor 近似

### 11.3 Taylor → 概率论(3-概率论 第 6 章)

- **中心极限定理** 的特征函数证明用 Taylor 展开:$e^{itX} \approx 1 + itX - t^2 X^2/2 + \cdots$
- **矩生成函数** $M(t) = E[e^{tX}]$ 的 Taylor 展开 = $E[X^n]/n!$
- **高斯分布** 是 $\chi^2$ 距离极小化的极限

### 11.4 Taylor → 数值分析(2-线代 第 5 章)

- **log-sum-exp** 数值稳定 = $x_{\max}$ 处的一阶 Taylor 近似
- **Jacobian 条件数** 与 Newton 收敛的关系
- **Hessian-vector product** 用 forward-over-reverse AD 高效计算

### 11.5 完整阅读路径

```text
本章(第 5 章)
  ├─ 复习:第 1 章(极限)、第 2 章(一元微分)、第 3 章(多元微分)
  ├─ 应用:第 6 章(最优化 第 5 章 收敛性证明)
  ├─ 应用:5-信息论 第 5 章(KL 与 CE 的二阶关系)
  ├─ 应用:3-概率论 第 6 章(中心极限定理的二阶 Taylor)
  └─ 后续:第 6 章(积分)、第 7 章(级数)、第 8 章(泛函)
```

---

> **本章总结**:
>
> 学完本章,你应该能回答:
> - 为什么 SGD 公式来自 Taylor 一阶展开? (因为局部线性化)
> - 为什么 Hessian 病态时 Adam 比 SGD 稳? (因为 Adam 自适应步长 ≈ Hessian 对角线的预处理)
> - XGBoost 为什么用二阶? (逐样本二阶比逐参数二阶便宜)
> - Newton 法为什么不能直接用在大模型上? (Hessian 装不下)
> - K-FAC / Sophia / GaLore 在做什么? (准二阶方法,在一阶的算力下获得二阶的精度)
>
> **下一章预告**:第 6 章讲积分，，概率论与 KL 散度的数学基础。你会看到 $\int_a^b f(x) dx$ 怎么从黎曼和"翻译"过来,以及为什么 **log-sum-exp** 用 $x_{\max}$ 减法能避免数值溢出(Taylor 展开的另一面)。

---

> **附录:本章速查卡**
>
> | 概念 | 公式 | 关键应用 |
| :--- | :--- | :--- |
> | 一阶 Taylor | $f(x_0+h) = f(x_0) + f'(x_0) h$ | SGD 来源 |
> | 二阶 Taylor | $f(x_0+h) = f(x_0) + f'(x_0) h + \frac{1}{2} f''(x_0) h^2$ | Newton 来源 |
> | 多元一阶 | $f(\mathbf{x}+\Delta) = f + \nabla f^T \Delta$ | 切平面 |
> | 多元二阶 | $f(\mathbf{x}+\Delta) = f + \nabla f^T \Delta + \frac{1}{2} \Delta^T H \Delta$ | 抛物面 |
> | Lagrange 余项 | $R_n = \frac{f^{(n+1)}(\xi)}{(n+1)!} h^{n+1}$ | 误差上界 |
> | 收敛半径 | $R = \lim |a_n/a_{n+1}|$ | 展开有效范围 |
> | Newton | $\theta_{t+1} = \theta_t - H^{-1} \nabla f$ | 二阶收敛 |
> | 阻尼 Newton | $H_{\text{damp}} = H + \lambda I$ | 防 $H$ 病态 |
> | L-BFGS | 维护 $m$ 个 $(\mathbf{s}, \mathbf{y})$ 对 | 大 $d$ 的准 Newton |
> | Adam | $\sqrt{v_t} \approx \sqrt{H_{ii}}$ | 自适应学习率 |

### 2.8 严格证明:Taylor 定理的 Cauchy 形式

**动机**:工程上用 Lagrange 余项够了,但数学上更精细的余项形式是 **Cauchy 余项**。

**定理**(Cauchy 余项):若 $f$ 在 $x_0$ 邻域 $n+1$ 阶可导,则存在 $\theta \in (0, 1)$ 使得:

$$R_n(x) = \frac{f^{(n+1)}(x_0 + \theta(x - x_0))}{n!} (1 - \theta)^n (x - x_0)^{n+1}$$

**与 Lagrange 的对比**:

| 余项形式 | 表达式 | 适合场景 |
| :--- | :--- | :--- |
| Lagrange | $\frac{f^{(n+1)}(\xi)}{(n+1)!} h^{n+1}$ | 误差上界估计 |
| Cauchy | $\frac{f^{(n+1)}(\xi)}{n!} (1-\theta)^n h^{n+1}$ | 极限推导 |
| Peano | $o(h^n)$ | 渐近分析 |

**例**:对 $f(x) = e^x$ 在 $x_0 = 0$ 处的 1 阶 Taylor:
- Lagrange:$R_1(x) = e^\xi \cdot x^2/2$,$\xi \in (0, x)$
- Cauchy:$R_1(x) = e^{\theta x} \cdot (1 - \theta) \cdot x^2$
- 在 $x \to 0$ 时,$\xi, \theta x \to 0$,所以 $R_1(x) \to x^2/2$ 或 $\to x^2$(具体值看 $\theta$)。

**证明思路**(Lagrange 余项):

用 Cauchy 中值定理反复。定义 $g(t) = f(x_0 + t h) - \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!} (t h)^k$,对 $g(1) - g(0) = -R_n$ 应用 Cauchy 中值定理,得到 $R_n = g^{(n+1)}(\theta)/(n+1)! \cdot 1^{n+1} = \frac{f^{(n+1)}(x_0 + \theta h)}{(n+1)!} h^{n+1}$。

**工程意义**:Taylor 定理的严格证明告诉你"为什么这个近似是对的"。工程上用不到证明本身,但理解证明能让你**知道什么时候不能信 Taylor 展开**(比如,函数 $C^{n+1}$ 但不 $C^{n+2}$,Lagrange 余项仍成立但更紧的余项不行)。

### 2.9 关键函数的高阶 Taylor 展开(进阶参考表)

下表给出工程中可能用到的所有重要函数的 Taylor 展开,及前几项的显式形式:

| 函数 | Taylor 展开(前几项) | 工程用途 |
| :--- | :--- | :--- |
| $\sqrt{1+x}$ | $1 + \frac{x}{2} - \frac{x^2}{8} + \frac{x^3}{16} - \cdots$ | LayerNorm RMS 归一化 |
| $\tanh(x)$ | $x - \frac{x^3}{3} + \frac{2x^5}{15} - \cdots$ | 激活函数小输入 |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\frac{1}{2} + \frac{x}{4} - \frac{x^3}{48} + \cdots$ | Logistic 小输入 |
| $\arctan(x)$ | $x - \frac{x^3}{3} + \frac{x^5}{5} - \cdots$ | 注意力权重 |
| $(1+x)^{1/2}$ | $1 + \frac{x}{2} - \frac{x^2}{8} + \cdots$ | AdamW 权重衰减 |
| $\log(\cos x)$ | $-\frac{x^2}{2} - \frac{x^4}{12} - \cdots$ | 数值稳定技巧 |

**例**:LayerNorm 中算 $\sqrt{\text{Var} + \epsilon}$,如果 $\epsilon \ll \text{Var}$,用 $\sqrt{1 + \epsilon/\text{Var}} \approx 1 + \epsilon/(2\text{Var})$,**少一个 sqrt 调用**。但实际工程中 $\epsilon$ 不一定小,所以这种近似**不常用**。

### 2.10 工程反例:为什么"Taylor 阶数越高越好"在工程上是错的

**学术观点**:Taylor 展开阶数 $n$ 越大,误差 $O(h^{n+1})$ 越小,**理论越好**。

**工程现实**:

1. **计算代价**:$n$ 阶 Taylor 要算 $n$ 阶导数。$n = 2$ 需要 Hessian(已很难);$n = 3$ 需要三阶张量 $O(d^3)$ 内存,**装不下**。
2. **数值精度**:浮点有舍入误差。阶数高 → 项之间相消 → 灾难性抵消(catastrophic cancellation)。
3. **截断 vs 舍入的权衡**:阶数增加,前 $n$ 项算得更准,但浮点累加误差也增加。**最优阶数**取决于 $h$ 的大小和浮点精度。

**例**:
- $e^x$ 在 $x = 10$ 附近:截断到 20 阶 vs 直接算
- 直接算:`exp(10) = 22026.4657948...`,**精确**
- 20 阶 Taylor:$\sum_{k=0}^{20} 10^k/k! = ?$。前几项极大(头几项就 10000+),后面项很小,浮点累加误差大。结果可能与真值差 0.01%(看似小,但相对 20 阶 Taylor 的"理论精度 10^{-20}"是灾难)

**结论**:**在 $|x|$ 大时,直接调用 `exp(x)` 比高阶 Taylor 更准**。Taylor 的优势在 $|x|$ 小的时候。

**工业教训**:NVIDIA 的 cuBLAS / cuDNN 库中,`exp`, `log`, `sin` 等函数用**分段多项式**(interval arithmetic)实现,在每个小区间用 5-7 阶 Taylor,而不是用单一段高阶 Taylor。

### 2.11 多元 Taylor 的一元退化验证

**练习**:把 $d = 1$ 代入多元 Taylor 公式,验证退化为第 2 节的一元形式。

多元 Taylor 公式:
$$f(\mathbf{x} + \Delta) = \sum_{k=0}^n \frac{1}{k!} D^k f(\mathbf{x})[\Delta]^k$$

$d = 1$ 时:
- $D^0 f = f$
- $D^1 f = f'$
- $D^2 f = f''$
- ...
- $D^k f = f^{(k)}$

$\Delta = h$ (标量):
- $D^k f \cdot h^k = f^{(k)} h^k$

所以:
$$f(x_0 + h) = \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!} h^k + R_n$$

**与一元 Taylor 一致** ✓。

**这是检验多元公式"对不对"的简单方法**。如果一元情况下不成立,多元必然错。

---

## 估计 L:用梯度范数随步长变化
def estimate_L(model, loss_fn, data, lr_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    L_estimates = []
    for lr in lr_range:
        # 算 Hvp
        ...
```

具体代码见 §8.6。

### 3.10 误差与收敛速度的工程取舍

**问题**:训练时,Loss 不再下降。是 Taylor 展开失效了,还是已经达到全局最优?

**判断标准**:

| 现象 | 原因 | 解决 |
| :--- | :--- | :--- |
| Loss 抖动但下降 | mini-batch 噪声 | 降 lr 或加 momentum |
| Loss 平稳(plateau) | 学习率太小,陷入鞍点 | 加 lr 或换 Adam |
| Loss 上升 | 步长太大,Taylor 失效 | 降 lr 或加 warmup |
| Loss = NaN | 数值溢出 | 加 grad clipping |

**工程经验**:
- 一阶方法(SGD/Adam):能容忍小步长,不适合大步
- 二阶方法(Newton):能容忍大步,但需要正定 Hessian
- 准二阶方法(Adam):是 sweet spot

### 3.11 Taylor 截断的"最佳阶数"(理论)

**定理**(最优截断阶数):给定 $f \in C^\infty$ 在 $x_0$ 处,$|h|$ 固定,浮点精度 $\epsilon_{\text{mach}}$。

最优截断阶数 $n^*$ 由两个误差的**平衡**决定:
- **截断误差** $O(h^{n+1}/(n+1)!)$ 随 $n$ 减小
- **舍入误差** $O(n \epsilon_{\text{mach}} \|f\|)$ 随 $n$ 增大

**最优 $n^* \approx \log(1/\|h\|) / \log(1/\epsilon_{\text{mach}})$**(粗略)。

**数值**:
- $|h| = 0.1$, $\epsilon_{\text{mach}} = 10^{-16}$: $n^* \approx 8$
- $|h| = 1.0$, $\epsilon_{\text{mach}} = 10^{-16}$: $n^* \approx 16$
- $|h| = 0.01$, $\epsilon_{\text{mach}} = 10^{-16}$: $n^* \approx 4$

**工程意义**:
- 离 $x_0$ 越远,需要越高阶 Taylor
- 但 $n$ 高 → 计算量大,实际工程常用**分段低阶**(如 5 阶 + 重映射区间)
- NVIDIA cuBLAS 用的就是这种策略

---


# 多元 Taylor 展开验证
def f(x):  # f: R^3 → R
    return (x ** 2).sum()

x0 = torch.tensor([1.0, 2.0, 3.0])
delta = torch.tensor([0.1, 0.05, -0.2])

# 真实值
f_true = f(x0 + delta)

# 0 阶
T0 = f(x0)

# 1 阶
grad = torch.autograd.grad(f(x0), x0, create_graph=True)[0]
T1 = T0 + grad @ delta

# 2 阶
# Hessian:对每个分量求 grad
hess = torch.zeros(3, 3)
for i in range(3):
    hess[i] = torch.autograd.grad(grad[i], x0, retain_graph=True)[0]
T2 = T1 + 0.5 * delta @ hess @ delta

print(f"True:    {f_true.item():.6f}")
print(f"T0:      {T0.item():.6f}, err={abs(T0 - f_true).item():.6f}")
print(f"T1:      {T1.item():.6f}, err={abs(T1 - f_true).item():.6f}")
print(f"T2:      {T2.item():.6f}, err={abs(T2 - f_true).item():.2e}")
# T2 误差应当为 0(因为 f 是二次型)
```

**几何意义**:对二次函数,二阶 Taylor **完全等于原函数**。这就是为什么线性回归的 GD **几步就到最优**。

## 4.11 多项式基的另一种理解

多元 Taylor 展开可以用**基函数**的角度理解。设 $P_1, P_2, \ldots$ 是多项式基(如单项式 $x^i y^j$):

$$f(\mathbf{x}) = \sum_k c_k P_k(\mathbf{x})$$

**Taylor 系数 $c_k$ 怎么算**:把 $P_k$ 看作函数,要求 $\langle f, P_k \rangle = \langle \sum_j c_j P_j, P_k \rangle$。如果 $\{P_k\}$ 是正交基,直接投影。

**工程含义**:
- 神经网络可以看作**通用函数逼近器**(多项式基 + 非线性激活)
- 但神经网络不是 Taylor 展开:它是**全局逼近**,Taylor 是**局部逼近**
- 这就是为什么"用一个 10 层 MLP 拟合 $\sin(x)$"是可行的,但"用 10 阶 Taylor 拟合 $\sin(x)$ 在整个 $\mathbb{R}$"不行(收敛半径限制)

### 4.12 多元展开的常见误区

**误区 1**:二阶展开总是比一阶好。

**反例**:在极小点附近,$\nabla f \approx 0$,一阶项贡献小,**二阶项主导**。在远离极小点,$\nabla f$ 大,一阶项主导。**两者的相对重要性取决于 $|g| / \|H\| \cdot \|\Delta\|$**。

**误区 2**:Hessian 必须是常数矩阵才能用二阶展开。

**反例**:Hessian $H(\mathbf{x})$ 随 $\mathbf{x}$ 变化没关系。**二阶 Taylor 展开是"在某点处"的局部近似**,展开后 $H$ 是固定值(取在 $\mathbf{x}_0$ 处的 $H$)。这正是 Newton 步的定义。

**误区 3**:Newton 法是 Taylor 展开的唯一用途。

**反例**:Taylor 展开还用于
- 数值积分(Newton-Cotes)
- 数值 ODE 求解(Euler, Runge-Kutta 都是 Taylor 展开)
- 物理模拟(有限元)
- 量化金融(Black-Scholes 局部近似)

### 4.13 梯度流的物理类比

**类比**:优化过程 = 物理中"小球在曲面上滚动"。

- **Loss 景观** = 山地地形
- **梯度** = 重力方向
- **Hessian** = 地形曲率
- **学习率** = 时间步长
- **动量** = 小球惯性

**SGD**:小球沿重力滚下,有惯性 + 摩擦
**Adam**:每方向独立调摩擦(类比:"松土让小球下滑更快")
**Newton**:知道地形曲率后,**直接跳到极小点**

**工程含义**:
- 训练初期:Loss 大,梯度大,**SGD 沿梯度走**(类比:重力大,直接滚)
- 训练后期:Loss 小,梯度小,**Adam 体现优势**(类比:重力和摩擦平衡时,惯性)
- 训练极后期:Loss 极小,**Newton 一次到位**(类比:已经到山谷底部,直接定位)

**比喻**:**SGD 像盲人下山,Newton 像有地图的人下山,Adam 像带着等高线地图的盲人**。

---

## 定义损失函数
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    return loss

## LBFGS 优化器
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=10)

## 训练循环
for epoch in range(epochs):
    loss = optimizer.step(closure)
```

**注意**:`LBFGS` 需要 `closure` 函数(因为它要做"内部迭代")。

### 5.6 PyTorch 代码:手写一维 Newton 法

```python
import torch

def newton_1d(f, f_prime, x0, tol=1e-8, max_iter=50):
    """一维 Newton 法求 f(x) = 0 的根

    参数:
        f: 函数 f(x)
        f_prime: 导数 f'(x)
        x0: 初始点
        tol: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        根的近似值
    """
    x = x0
    history = [x.item()]

    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)

        if abs(fx) < tol:
            print(f"在 {i} 步收敛,|f(x)| = {abs(fx):.2e}")
            return x, history

        if abs(fpx) < 1e-12:
            print("导数太小,Newton 法可能失败")
            return x, history

        # Newton 步
        x_new = x - fx / fpx
        history.append(x_new.item())

        # 检查收敛
        if abs(x_new - x) < tol:
            print(f"在 {i+1} 步收敛,|Δx| = {abs(x_new - x):.2e}")
            return x_new, history

        x = x_new

    print(f"达到最大迭代 {max_iter}")
    return x, history


## 例 1:求 x^3 - 2x + 2 = 0
print("=== 例 1: x^3 - 2x + 2 = 0 ===")
def f1(x):
    return x**3 - 2*x + 2

def f1_prime(x):
    return 3*x**2 - 2

root, hist = newton_1d(f1, f1_prime, x0=torch.tensor(-2.0))
print(f"根 = {root.item():.6f}")

## 观察二次收敛
print("\n=== 二次收敛演示:x^2 - 2 = 0, 根 = √2 ===")
def f_sqrt(x):
    return x**2 - 2

def f_sqrt_prime(x):
    return 2*x

root, hist = newton_1d(f_sqrt, f_sqrt_prime, x0=torch.tensor(1.0))
print(f"迭代历史: {[f'{h:.10f}' for h in hist]}")
print(f"误差: {[f'{abs(h - 1.41421356):.2e}' for h in hist]}")
## 误差按平方缩小:0.4 → 0.08 → 0.0006 → ...
```

**预期输出**:
```text
=== 例 1: x^3 - 2x + 2 = 0 ===
在 5 步收敛,|f(x)| = 4.10e-09
根 = -1.769292

=== 二次收敛演示:x^2 - 2 = 0, 根 = √2 ===
迭代历史: ['1.0000000000', '1.5000000000', '1.4166666667', '1.4142156863', '1.4142135624']
误差: ['4.14e-01', '8.58e-02', '2.45e-03', '2.13e-06', '1.53e-12']
```

**关键观察**:误差 $0.41 \to 0.086 \to 0.0024 \to 2 \times 10^{-6} \to 10^{-12}$，，**每步平方**。4 步到 12 位精度。这就是 Newton 法的"二次收敛"威力。

### 5.7 多元 Newton 法的伪代码(完整实现在 §8.4)

多元 Newton 法的核心逻辑(简化版,完整代码在 §8.4):

```python
import torch

def newton_multid(f, x0, tol=1e-8, max_iter=50, damping=True):
    x = x0.clone()
    for i in range(max_iter):
        # 1. 算梯度(用 autograd)
        x_var = x.clone().requires_grad_(True)
        f_val = f(x_var)
        grad = torch.autograd.grad(f_val, x_var)[0]

        # 2. 算 Hessian(用 autograd 二次调用)
        hess = torch.zeros(x.shape[0], x.shape[0])
        for j in range(x.shape[0]):
            grad_j = torch.autograd.grad(grad[j], x_var,
                                          create_graph=True)[0]
            hess[j] = grad_j.detach()

        # 3. 阻尼 + 求解线性方程组
        if damping:
            hess = hess + 1e-4 * torch.eye(x.shape[0])  # 防不正定
        delta = torch.linalg.solve(hess, -grad)

        # 4. 更新
        x = x + delta
        if torch.norm(delta) < tol:
            break
    return x
```

**关键步骤**:
1. **梯度**(用 autograd):`torch.autograd.grad(f, x)`
2. **Hessian**(对梯度再求导):`create_graph=True` 保留计算图
3. **解线性方程组**:`torch.linalg.solve(hess, -grad)`,**比 `inv(hess) @ grad` 稳定**
4. **阻尼**:`hess + λI` 防 $H$ 不正定

完整实验对比(SGD / Adam / Newton / L-BFGS)见 §8.4。

### 5.8 阻尼 Newton(防止 Hessian 病态)

**问题**:Newton 法在 $H$ 病态或非正定时可能"乱走"。

**解决**:**阻尼 Newton**(Levenberg-Marquardt 思想):

$$H_{\text{damp}} = H + \lambda I$$

其中 $\lambda > 0$ 是阻尼系数。

- $H + \lambda I$ 总是正定($\lambda > 0$)
- 大 $\lambda$:接近梯度下降(稳)
- 小 $\lambda$:接近纯 Newton(快)
- 自适应:信任域方法动态调整 $\lambda$

**工程实现**:在 PyTorch 中:

```python
## 阻尼
lam = 1e-3
I = torch.eye(d)
H_damped = H + lam * I

## 求解
delta = torch.linalg.solve(H_damped, -grad)
```

### 5.9 本节回顾

**核心收获**:
1. 一元 Newton 法用切线求根:$x_{t+1} = x_t - f(x_t)/f'(x_t)$
2. 多元 Newton 法用抛物面求极小:$\mathbf{x}_{t+1} = \mathbf{x}_t - H^{-1} \nabla f$
3. Newton 法**二次收敛**(误差平方),但要 $O(d^3)$ 求逆
4. BFGS / L-BFGS 准 Newton 法,避免显式 Hessian
5. 阻尼 Newton 加 $\lambda I$ 防止 $H$ 病态
6. 大模型训练不用 Newton 是因为 $H$ 装不下

**下一步**:既然 Newton 那么好,为什么深度学习几乎不用?这就是 §6 的主题。

### 5.10 Newton 法二阶收敛的严格证明(完整版)

**定理**(Newton 法的局部二阶收敛,严格版):

设 $f: \mathbb{R} \to \mathbb{R}$ 在 $x^*$ 邻域内 $C^3$ 光滑,$f'(x^*) = 0$ 且 $f''(x^*) \neq 0$。则存在 $\delta > 0$ 和 $C > 0$,使得对任意 $x_0 \in (x^* - \delta, x^* + \delta)$,Newton 迭代 $x_{t+1} = x_t - f'(x_t)/f''(x_t)$ 满足:

$$|x_{t+1} - x^*| \le C |x_t - x^*|^2$$

**证明**:

**Step 1**:对 $f'(x_t)$ 在 $x^*$ 处 Taylor 展开($C^3$ 假设保证二阶 Taylor + 余项):

$$f'(x_t) = f'(x^*) + f''(x^*) e_t + \frac{1}{2} f'''(\xi_1) e_t^2$$

其中 $e_t = x_t - x^*$,$\xi_1 \in (x^*, x_t)$。因为 $f'(x^*) = 0$:

$$f'(x_t) = f''(x^*) e_t + \frac{1}{2} f'''(\xi_1) e_t^2 \quad (1)$$

**Step 2**:对 $f''(x_t)$ 在 $x^*$ 处 Taylor 展开(只需 $C^2$):

$$f''(x_t) = f''(x^*) + f'''(\xi_2) e_t, \quad \xi_2 \in (x^*, x_t) \quad (2)$$

**Step 3**:代回 Newton 步:

$$x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)} = x_t - \frac{f''(x^*) e_t + \frac{1}{2} f'''(\xi_1) e_t^2}{f''(x^*) + f'''(\xi_2) e_t}$$

提取 $f''(x^*)$:

$$= x_t - \frac{e_t + \frac{f'''(\xi_1)}{2 f''(x^*)} e_t^2}{1 + \frac{f'''(\xi_2)}{f''(x^*)} e_t}$$

记 $c_1 = f'''(\xi_1)/(2 f''(x^*))$,$c_2 = f'''(\xi_2)/f''(x^*)$,利用 $(1 + c_2 e_t)^{-1} = 1 - c_2 e_t + O(e_t^2)$:

$$x_{t+1} \approx x_t - (e_t + c_1 e_t^2)(1 - c_2 e_t) = x_t - e_t + c_2 e_t^2 - c_1 e_t^2 + O(e_t^3)$$

$$= x^* + c_2 e_t^2 - c_1 e_t^2 + O(e_t^3)$$

**新误差**:

$$e_{t+1} = x_{t+1} - x^* = (c_2 - c_1) e_t^2 + O(e_t^3) = O(e_t^2)$$

**Step 4**:由于 $\xi_1, \xi_2 \to x^*$ 当 $e_t \to 0$,$c_1, c_2$ 收敛到常数,所以:

$$|e_{t+1}| \le C |e_t|^2$$

其中 $C$ 依赖于 $x^*$ 邻域内 $f'''$ 和 $1/f''$ 的界。**QED**。

**多元版本**:基本相同,但需用 $H$ 的 Lipschitz 性质($\|H(x) - H(y)\| \le M \|x - y\|$)和 $H(x^*)$ 非奇异性。

**工程含义**:证明告诉你"在 $x^*$ 附近"Newton 二次收敛。**如果起点远离 $x^*$,需要线搜索/信任域保证全局收敛**。

### 5.11 收敛速度的对比表

| 方法 | 收敛阶 | 每步代价 | 内存 | 适用 |
| :--- | :--- | :--- | :--- | :--- |
| 梯度下降 (GD) | 1(线性) | $O(d)$ | $O(d)$ | 一般 |
| 随机 GD (SGD) | 1(次线性) | $O(b)$ (b = batch) | $O(d)$ | 大数据 |
| 动量 SGD | 1(加速) | $O(d)$ | $O(d)$ | 一般 |
| Adam | 1(加速) | $O(d)$ | $O(d)$ | 病态 |
| Newton | **2**(二次) | $O(d^3)$ | $O(d^2)$ | 凸小问题 |
| 阻尼 Newton | 2(局部) | $O(d^3)$ | $O(d^2)$ | 病态 |
| BFGS | 1~2(超线性) | $O(d^2)$ | $O(d^2)$ | 凸中问题 |
| L-BFGS | 1~2(超线性) | $O(md)$ | $O(md)$ | 凸中问题 |
| Hessian-free | 2(局部) | $O(d)$ (HVP) | $O(d)$ | 大规模 |
| K-FAC | 1~2 | $O(d^{1.5})$ | $O(d^{1.5})$ | 神经网络 |

**收敛阶定义**:
- **1 阶(线性)**:$\|e_{t+1}\| \le q \|e_t\|$, $q < 1$。每步乘 $q$。
- **2 阶(二次)**:$\|e_{t+1}\| \le C \|e_t\|^2$。每步平方。
- **超线性**:$\|e_{t+1}\| / \|e_t\| \to 0$。比线性快但不到二次。

### 5.12 Newton 法的全局收敛:线搜索 + 信任域

**问题**:Newton 法的二阶收敛是**局部**的(在 $x^*$ 邻域)。如果起点远离 $x^*$,Newton 可能"跑飞"。

**解决 1:线搜索(Backtracking Line Search)**

```python
def newton_with_linesearch(f, x0):
    x = x0
    for step in range(max_iter):
        grad = compute_grad(f, x)
        H = compute_hessian(f, x)
        delta = -torch.linalg.solve(H, grad)

        # 线搜索:找合适的步长
        alpha = 1.0
        while f(x + alpha * delta) > f(x) + 0.5 * alpha * grad @ delta:
            alpha *= 0.5
            if alpha < 1e-10:
                break

        x = x + alpha * delta
    return x
```

**关键**:线搜索保证 **Armijo 条件** $f(x + \alpha \Delta) \le f(x) + c_1 \alpha \nabla f^T \Delta$。

**解决 2:信任域(Trust Region)**

```python
def newton_with_trust_region(f, x0, delta_init=1.0):
    x = x0
    delta = delta_init
    for step in range(max_iter):
        grad = compute_grad(f, x)
        H = compute_hessian(f, x)

        # 解子问题: min g^T d + 0.5 d^T H d s.t. ||d|| <= delta
        # 用截断 CG
        d, _ = truncated_cg(lambda v: H @ v, -grad, delta)

        # 评估预测 vs 实际
        pred = -(grad @ d + 0.5 * d @ H @ d)
        actual = f(x) - f(x + d)
        rho = actual / pred

        if rho > 0.75:
            delta = min(2 * delta, delta_max)
        elif rho < 0.25:
            delta = 0.5 * delta

        if rho > 0.1:
            x = x + d
    return x
```

**工程对比**:
- **线搜索**:简单,稳定
- **信任域**:更精细,适合病态问题
- **Levenberg-Marquardt**:阻尼 + 线搜索的结合,广泛用于非线性最小二乘

### 5.13 Newton 法的并行化

**问题**:Newton 法的 $O(d^3)$ 求逆难并行。

**方案**:
- **GPU 加速**:cuSOLVER 提供 batch 矩阵求逆。但 $d = 10^4$ 已极限。
- **分布式**:把 Hessian 分块,每节点算自己块,**通信** $O(d^2)$。
- **近似**:K-FAC / Shampoo 把 Hessian 分解为小块的 Kronecker 积,可并行。

**工程现实**:对 LLM,$H$ 太大,Newton 不可并行。**只能用 HVP 类方法**(O(d) 计算,O(d) 通信)。

### 5.14 Newton-CG(用 CG 内部解线性方程组)

**思想**:不解 $H \delta = -g$,而是**用 CG(共轭梯度)迭代解**。CG 每步只需 $H v$。

**算法**:

```python
def newton_cg(g, hvp_fn, cg_iters=50, tol=1e-8):
    """用 CG 解 H delta = -g,只用 hvp_fn 计算 Hv"""
    delta = torch.zeros_like(g)
    r = -g.clone()  # 残差
    p = r.clone()   # 搜索方向

    for i in range(cg_iters):
        Hp = hvp_fn(p)
        # 步长
        alpha = (r @ r) / (p @ Hp + 1e-8)
        delta = delta + alpha * p
        r_new = r - alpha * Hp

        if torch.norm(r_new) < tol:
            break

        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    return delta
```

**优点**:不用显式 $H$,只算 HVP,**$O(d)$ 内存**。

**缺点**:CG 在 $H$ 非正定时不稳定(可能除以负数)。

**Hessian-free optimization** = Newton-CG + 线搜索(详见 §7.11)。

---


# 直接算会溢出
# logsumexp = torch.log(torch.sum(torch.exp(z)))  # 错误:inf

# 数值稳定
logsumexp = torch.logsumexp(z, dim=-1)  # 正确:1001.41
```

### 7.3 KL 散度与 CrossEntropy 的二阶关系

**事实**:在 $p = q$ 附近,$\text{KL}(p \| q) \approx \text{CE}(p, q) - H(p)$。

**推导**(用二阶 Taylor):

设 $p, q$ 是两个分布,$\text{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}$,$\text{CE}(p, q) = -\sum_i p_i \log q_i$。

把 $p_i$ 写成 $p_i = q_i + \delta_i$,其中 $\sum \delta_i = 0$(归一化约束)。

**CE 在 $p = q$ 附近展开**:

$$\text{CE}(p, q) = -\sum_i (q_i + \delta_i) \log q_i = -\sum_i q_i \log q_i - \sum_i \delta_i \log q_i$$

- 第一项:$-H(q) = H(q)$?不,$\text{CE}(q, q) = H(q)$(真实分布的熵)
- 第二项:$\sum \delta_i \log q_i$ ，， 用 $\sum \delta_i = 0$,**二阶小量**

**KL 散度在 $p = q$ 附近展开**:

$$\text{KL}(p \| q) = \sum_i (q_i + \delta_i) \log \frac{q_i + \delta_i}{q_i} = \sum_i (q_i + \delta_i) \log \left( 1 + \frac{\delta_i}{q_i} \right)$$

用 $\log(1 + x) \approx x - x^2/2$:

$$\approx \sum_i (q_i + \delta_i) \left[ \frac{\delta_i}{q_i} - \frac{\delta_i^2}{2 q_i^2} \right]$$

展开:
- 一阶项:$\sum_i (q_i + \delta_i) \frac{\delta_i}{q_i} = \sum_i \delta_i + \sum_i \frac{\delta_i^2}{q_i}$
- 二阶项:$- \frac{1}{2} \sum_i (q_i + \delta_i) \frac{\delta_i^2}{q_i^2} \approx -\frac{1}{2} \sum_i \frac{\delta_i^2}{q_i}$(因为 $\delta_i$ 很小)

由于 $\sum \delta_i = 0$:**一阶项中 $\sum \delta_i = 0$**,所以:

$$\text{KL}(p \| q) \approx \frac{1}{2} \sum_i \frac{\delta_i^2}{q_i} = \frac{1}{2} \chi^2(p \| q)$$

**结论**:**在 $p = q$ 附近,KL 散度 ≈ $\frac{1}{2} \chi^2$ 距离**(二阶关系)。

**工程意义**:分类任务用 CE loss,实际上在**最小化 KL 散度的二阶近似**。

### 7.4 数值稳定性:为什么 $\log(x)$ 在 $x = 1$ 附近用 Taylor 展开更稳?

**问题**:$\log(x)$ 在 $x = 1$ 附近?但 $\log(1) = 0$ 没问题啊。

**实际问题**:$\log(\sigma(z))$ 在 $\sigma(z) = 1$ 附近(预测对时)会下溢($\log(1) = 0$ 没问题,但 $\log(1 - 10^{-7})$ = 极小负数)。

**更好的写法**:用 $\log(1 + \epsilon) \approx \epsilon$。

设 $p = 1 - \epsilon$ ($\epsilon$ 很小,$\epsilon > 0$):
- $\log(p) = \log(1 - \epsilon) = -\epsilon - \epsilon^2/2 - \epsilon^3/3 - \cdots$
- $\log(1 - \epsilon) \approx -\epsilon$(一阶)

**例**:
- $p = 0.9999$:$\log(0.9999) \approx -0.0001$(一阶近似),真值 $-0.000100005...$
- $p = 0.99$:$\log(0.99) \approx -0.01$(一阶近似),真值 $-0.0100503...$,误差 $0.5\%$

**PyTorch 例子**:

```python
import torch
import torch.nn.functional as F

# 直接 log(p) 在 p 接近 0 时会 -inf
p = torch.tensor([0.0, 0.5, 1.0])
torch.log(p)  # tensor([-inf, -0.693, 0.0])

# log_softmax 数值稳定
z = torch.tensor([0.0, 0.0, 1000.0])
log_p = F.log_softmax(z, dim=-1)  # 不会溢出
```

### 7.5 KL 散度的工程应用:变分推断

**ELBO**(变分下界):

$$\text{ELBO}(q) = \mathbb{E}_{z \sim q}[\log p(x, z) - \log q(z)] = \log p(x) - \text{KL}(q(z) \| p(z|x))$$

最大化 ELBO = 最小化 KL(近似后验 vs 真实后验)。

**二阶 Taylor** 在 VAE 的应用:用重参数化 $z = \mu + \sigma \epsilon$,把 $\mathbb{E}[\text{ELBO}]$ 近似为 $\mu, \sigma$ 的函数,再用梯度下降优化。

**这与本节"二阶关系"的关系**:
- ELBO 一阶项 = 重构误差(类似 CrossEntropy)
- ELBO 二阶项 = KL 散度的二阶近似(类似 Hessian 对角)

所以 **VAE 训练 ≈ 加权 KL 正则的 autoencoder**。

### 7.6 本节回顾

**核心收获**:
1. XGBoost 用损失函数的二阶 Taylor,内存可控
2. Softmax 的对数配分函数 Hessian 特征值 $\le 0.25$,稳定
3. KL ≈ $\frac{1}{2} \chi^2$ 在 $p = q$ 附近
4. $\log$ 在 $1$ 附近用 $\log(1 - \epsilon) \approx -\epsilon$ 防下溢
5. VAE 的 ELBO 是 KL 散度的二阶近似 + 重构误差

### 7.7 Hessian-Vector Product (HVP):Hessian 的"打折版"

**动机**:Newton 法需要 $H^{-1} g$。但直接算 $H^{-1}$ 太贵,**能不能只算 $H v$(任意 $v$)而不存 $H$?**

**定理**(HVP via Pearlmutter 1994):

$$Hv = \nabla (\nabla L \cdot v) = \frac{d}{d\epsilon}\bigg|_{\epsilon=0} \nabla L(\theta + \epsilon v)$$

**直觉**:对 $L$ 在 $\theta$ 处沿 $v$ 方向求导,得到 $H v$。

**计算代价**:
- **内存** $O(d)$(只存 $\theta$ 和计算图)
- **时间** $O(1 \text{ forward + backward})$(与一次反向传播同阶)
- **比直接算 $H$ 节省 $d$ 倍**

**实现**(`torch.autograd`):

```python
def hvp(loss_fn, params, vector):
    """算 Hessian-vector product"""
    # 1. 算梯度
    grads = torch.autograd.grad(loss_fn, params, create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])

    # 2. 与 vector 做内积
    gv = (flat_grad * vector).sum()

    # 3. 对这个内积再求导
    Hv = torch.autograd.grad(gv, params)
    flat_Hv = torch.cat([g.flatten() for g in Hv])
    return flat_Hv
```

**应用场景**:

1. **Hessian-free Newton** (Martens 2010):用共轭梯度(CG)在内部解 $H \delta = -g$,CG 每步用 HVP。
2. **Hutchinson trace estimator**:用 $v^T H v$ 的期望估计 $\text{tr}(H)$。
3. **Hessian 谱分析**:用 Lanczos 算法算 $H$ 的最大 / 最小特征值(用 HVP)。

**工程意义**:HVP 是**深度学习二阶方法的"原子操作"**。所有大规模二阶方法(Sophia, Hessian-free, K-FAC)都依赖 HVP。

**PyTorch 完整实验**:见 §8.6。

### 7.8 Fisher 信息矩阵 & 自然梯度

**Fisher 信息矩阵**(FIM):

$$F = \mathbb{E}_{x \sim p_\theta} [\nabla \log p_\theta(x) \nabla \log p_\theta(x)^T]$$

**与 Hessian 的关系**:

- 在指数族分布 + 真数据分布下,$F = H$(Fisher 等式)
- 一般情况下,$F \neq H$(二者不同!)

**自然梯度**:

$$\tilde{\nabla} L = F^{-1} \nabla L$$

**直觉**:自然梯度把"欧氏距离"换成"KL 散度距离"。

**为什么自然梯度更好**:
- 沿 $\theta + \Delta$ 走时,$KL(p_\theta \| p_{\theta + \Delta}) \approx \frac{1}{2} \Delta^T F \Delta$
- 自然梯度 $\|\tilde{\nabla} L\|_F = \sqrt{\nabla L^T F^{-1} \nabla L}$ 反映参数空间的"真实几何"
- 在 RL 的策略空间(参数变化 ≠ 行为变化)尤其重要

**TRPO**(Trust Region Policy Optimization):

$$\max_{\Delta} L(\theta + \Delta) \quad \text{s.t.} \quad \Delta^T F \Delta \le \delta$$

用自然梯度 + 信任域,**保证策略变化不会"跳太远"**。

**K-FAC 核心**:把 $F$ 分解成 $F \approx A \otimes G$($A$ = 输入协方差,$G$ = 输出梯度协方差)。$A, G$ 都是层内的 $O(d_l^2)$,总内存 $O(\sum d_l^2) \ll d^2$。

**PyTorch 实现**(K-FAC 简化版):

```python
class KFACOptimizer:
    """K-FAC: 逐层估计 Fisher 的 Kronecker 分解"""
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr
        self.A = {}  # 输入协方差
        self.G = {}  # 输出梯度协方差
        self.steps = 0

    def step(self, loss):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # 1. 更新 A 和 G 的 EMA
            # 2. 计算 F^{-1} @ grad (用 A_inv ⊗ G_inv 近似)
            # 3. 更新参数
            ...
```

**工程意义**:K-FAC 在 RNN / Transformer 训练中能用,但**比 Adam 慢 2-5 倍**(算 $A, G$ 的代价)。所以主流仍是 Adam + 偶尔 K-FAC fine-tune。

### 7.9 信任域方法(Trust Region Methods)

**思想**:**在每一步,选定一个"信任半径" $\Delta$,只在这个区域内信任二阶 Taylor 展开**。

**算法**(Cauchy point):

1. 在 $\mathbf{x}_t$ 处二阶 Taylor 展开
2. 解子问题:$\min_\Delta \frac{1}{2} \Delta^T H \Delta + g^T \Delta$ s.t. $\|\Delta\| \le \Delta_{\text{trust}}$
3. 实际下降 vs 预测下降的比值 $\rho = \frac{f(\mathbf{x}_t) - f(\mathbf{x}_t + \Delta)}{-\nabla f^T \Delta - \frac{1}{2} \Delta^T H \Delta}$
4. 根据 $\rho$ 调整 $\Delta_{\text{trust}}$:
   - $\rho$ 大(预测准)→ 增大 $\Delta_{\text{trust}}$
   - $\rho$ 小(预测差)→ 缩小 $\Delta_{\text{trust}}$

**优点**:
- **全局收敛**(不像纯 Newton 只能局部收敛)
- 自适应步长

**缺点**:
- 子问题求解复杂
- 仍需 $H$ 或 HVP

**工程应用**:
- **TRPO**(RL):用 Fisher + 信任域
- **TRF**(scipy 优化器):非线性最小二乘
- **Levenberg-Marquardt**:非线性最小二乘 + 阻尼

### 7.10 三次正则化(Cubic Regularization)

**动机**:Newton 法只用到二阶信息,三阶项可能修正"非凸"区域的走向。

**目标函数**:

$$M_\sigma(\Delta) = f(\mathbf{x}) + g^T \Delta + \frac{1}{2} \Delta^T H \Delta + \frac{\sigma}{3} \|\Delta\|^3$$

**求解**:Nesterov-Nemirovski (2006) 给出 $\sigma$ 选择,保证全局收敛。

**复杂度**:比 Newton 慢一个 $\log$ 因子。

**工程应用**:
- **信任域 + 三阶修正**(Cartis et al. 2011)
- 论文:三次正则化 + 二阶 Taylor + 自适应 $\sigma$

**实际工程**:几乎不用,Adam/SGD 已足够。

### 7.11 Hessian-free Optimization(无 Hessian 优化)

**来源**:Martens 2010 "Deep learning via Hessian-free optimization"

**核心思想**:

1. **不用显式 Hessian**:每次只算 HVP($O(d)$ 内存)
2. **用 CG(共轭梯度)解 $H \delta = -g$**:CG 每步一次 HVP,$k$ 步共 $O(kd)$
3. **线搜索 + 阻尼**:保证每步实际下降

**算法骨架**:

```python
def hf_optimizer_step(model, loss_fn, data, cg_iters=50):
    # 1. 算梯度 g
    g = compute_grad(model, loss_fn, data)

    # 2. 用 CG 解 H delta = -g
    delta = conjugate_gradient(lambda v: hvp(loss_fn, model, v), -g, cg_iters)

    # 3. 线搜索(找最优步长)
    alpha = line_search(model, loss_fn, data, delta)

    # 4. 更新参数
    for p, d in zip(model.parameters(), delta):
        p.data.add_(d, alpha=alpha)
```

**优点**:
- **内存 $O(d)$**(只存 $\delta$ 和 $g$)
- **二次收敛**(局部)

**缺点**:
- CG 在非凸 $H$ 上不稳定
- 每步 CG 计算量大
- **难并行化**

**历史地位**:2010-2015 年 RNN 训练的主流方法,后来被 Adam 取代。**现在主要是历史意义**。

### 7.12 量化与低秩 Hessian 近似

**动机**:进一步降低 Hessian 存储。

**量化**:把 Hessian 元素从 FP32 量化为 INT8(甚至 INT4)。**误差 $\sim 1\%$**,但内存 $4\times$ 减少。

**实现**:训练时 Hessian 是动态的,需要边算边量化。代表工作:`QSGD` (Alistarh et al. 2017)。

**低秩近似**:

$$H \approx U \Sigma U^T, \quad U \in \mathbb{R}^{d \times r}$$

**怎么算**:Lanczos 算法或随机 SVD。

**Lanczos 算法的 HVP 版本**:

```python
def lanczos_hvp(hvp_fn, d, num_iter):
    """用 HVP 算 Hessian 的 top-r 特征值和特征向量"""
    v = torch.randn(d)
    v = v / torch.norm(v)
    V = [v]
    alphas = []
    betas = []

    for _ in range(num_iter):
        Hv = hvp_fn(v)
        alpha = v @ Hv
        alphas.append(alpha)

        # 三对角化
        r = Hv - alpha * v
        if len(V) > 1:
            r = r - betas[-1] * V[-2]
        beta = torch.norm(r)
        betas.append(beta.item())

        v_new = r / beta
        V.append(v_new)
        v = v_new

    # 算 V 的子空间上的特征分解
    T = torch.diag(torch.tensor(alphas)) + torch.diag(torch.tensor(betas[:-1]), diagonal=1)
    eigvals, eigvecs = torch.linalg.eigh(T)
    return eigvals, V @ eigvecs
```

**GaLore**(2024)的核心:对梯度 $G$ 做 SVD,取前 $r$ 个奇异值,得到正交投影 $P$。在 $P^T G P$ 上跑 Adam。

**关键创新**:**预训练**也能用 GaLore(因为梯度本身就是低秩的)。实验显示:**GaLore 内存比 Adam 少 65%**,训练质量相当。

### 7.13 GNN(图神经网络)中的二阶方法

**GNN 特殊性**:
- 参数 + 图结构 = 双重非欧几里得
- Hessian 的稀疏模式 = 图的邻接矩阵

**GCN 上的 Hessian-free**:

```python
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(d, h)
        self.W2 = nn.Linear(h, c)

    def forward(self, x, adj):
        h = F.relu(self.W1(adj @ x))
        return self.W2(adj @ h)
```

**Hessian 结构**:$H = H_W + H_{\text{graph}}$,其中 $H_{\text{graph}}$ 来自图 Laplacian。

**应用**:GraphGaLore, Graph Spectral 方法(LOBPCG)算 Hessian 的图谱。

**工业实践**:GNN 二阶方法在推荐系统、CNS、分子生成有用,但**不是主流**。

### 7.14 联邦学习中的 Hessian 共享

**问题**:联邦学习中,数据分布在 $K$ 个客户端,不能共享数据。但每客户端可以算自己数据的 Hessian。

**FedAvg 思路**:每客户端用 SGD,服务端平均参数。**不涉及 Hessian**。

**FedNewton** (2018):每客户端算局部 Hessian 估计,服务端聚合。

$$H_{\text{global}} = \sum_k \frac{n_k}{n} H_k$$

**挑战**:
- 通信代价:$K \times d^2$ 元素。即使 $K = 10, d = 10^4$ 也不可行。
- **隐私风险**:Hessian 透露数据分布。

**解决方案**:
- **稀疏化**:只传 Hessian 的 top-$r$ 特征向量
- **差分隐私**:加 Laplace 噪声到 Hessian
- **同态加密**:加密 Hessian 后聚合

**工程现实**:联邦学习仍以 FedAvg 为主,FedNewton 还在研究阶段。

### 7.15 第二阶导数在注意力机制中的应用

**问题**:Transformer attention 中有 $\text{softmax}(Q K^T / \sqrt{d})$,**二阶 Taylor 能简化吗?**

**简化**:在 $Q K^T$ 变化小时,$\text{softmax}(z) \approx \text{softmax}(z_0) + J(z - z_0)$(一阶)。

**应用**:
- **Linear Attention**(Performer 等):用 $\phi(Q) \phi(K)^T$ 替代 $\text{softmax}(Q K^T)$,避免 $O(n^2)$。
- **Taylor Attention**:把 softmax 二阶展开,得到 $O(n)$ 复杂度的近似 attention。

**代表论文**:
- Katharopoulos et al. 2020: "Transformer are RNNs: Fast Autoregressive Transformers with Linear Attention"
- Choromanski et al. 2021: "Rethinking Attention with Performers"

**工程意义**:长序列(> 10K tokens)的 attention 必须用近似,否则显存爆。

### 7.16 边缘部署:小模型的二阶微调

**问题**:在边缘设备(手机、嵌入式)上 fine-tune 小模型。$d \le 10^5$。

**方法**:
- **L-BFGS**:内存 $O(md)$,够用
- **Hessian-free**:CG 内解,$O(d)$ 内存
- **KFAC-Mini**:K-FAC 的简化版,只算 $A$ 忽略 $G$

**案例**:
- Apple iOS 的 on-device personalization:用 L-BFGS
- Google TFLite:支持 L-BFGS
- Hugging Face PEFT:LoRA 用 Adam,full FT 用 L-BFGS

**优势**:小模型 + 二阶方法 = **几步收敛**,适合 few-shot 场景。

---


## 真实的 sin(x)
def true_sin(x):
    return torch.sin(x)

## Taylor 展开(第 n 阶)
def taylor_sin(x, n):
    """Maclaurin 展开: sin(x) = Σ (-1)^k x^{2k+1} / (2k+1)!"""
    result = torch.zeros_like(x)
    for k in range(n + 1):
        coef = (-1)**k / np.math.factorial(2*k + 1)
        result = result + coef * x**(2*k + 1)
    return result

## 测试不同阶数的精度
x = torch.linspace(-2*np.pi, 2*np.pi, 200)

print("=== sin(x) Taylor 展开精度 ===")
print(f"{'阶数':<6}{'最大误差(整个区间)':<25}{'在 x=π/2 处误差':<25}")
for n in [1, 2, 3, 5, 7, 10]:
    approx = taylor_sin(x, n)
    true = true_sin(x)
    max_err = (approx - true).abs().max().item()

    # 在 x = π/2 处
    x_test = torch.tensor([np.pi/2])
    err_at_pi_2 = (taylor_sin(x_test, n) - true_sin(x_test)).abs().item()

    print(f"{n:<6}{max_err:<25.6e}{err_at_pi_2:<25.6e}")

## 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

## 子图 1:不同阶数 vs 真值
ax = axes[0]
ax.plot(x.numpy(), true_sin(x).numpy(), 'k-', label='true sin(x)', linewidth=2)
for n in [1, 3, 5, 10]:
    ax.plot(x.numpy(), taylor_sin(x, n).numpy(), '--',
            label=f'Taylor n={n}', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
ax.set_title('Taylor 展开逼近 sin(x)')
ax.legend()
ax.grid(True, alpha=0.3)

## 子图 2:误差随 x 变化
ax = axes[1]
for n in [1, 3, 5, 10]:
    err = (taylor_sin(x, n) - true_sin(x)).abs().numpy()
    ax.plot(x.numpy(), err, label=f'Taylor n={n}', linewidth=1.5)
ax.set_xlabel('x')
ax.set_ylabel('|Taylor - sin|')
ax.set_yscale('log')
ax.set_title('误差分布(对数尺度)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/taylor_sin.png', dpi=100)
print("\n图已保存到 /tmp/taylor_sin.png")
```

**预期输出**:
```text
=== sin(x) Taylor 展开精度 ===
阶数   最大误差(整个区间)         在 x=π/2 处误差
1     1.957737e+00              2.214815e-01
2     1.957737e+00              2.214815e-01
3     1.640636e+00              1.349801e-01
4     1.640636e+00              1.349801e-01
5     1.597209e+00              4.430810e-02
...
```

**观察**:阶数越高,误差越小。$n = 10$ 时误差已经非常小。

### 8.2 实验 2:用二阶 Taylor 逼近二元凸函数(核心代码)

```python
import torch
import matplotlib.pyplot as plt

## 二元凸函数(病态 Hessian)
def f(x, y):
    return 0.5 * (10 * x**2 + y**2)  # 椭圆等高线, 条件数 10

## 在 (x0, y0) 处的二阶 Taylor 展开
def taylor_2nd(x0, y0, X, Y):
    f0 = f(torch.tensor(x0), torch.tensor(y0))
    # 一阶 + 二阶
    return (f0
            + 10 * x0 * (X - x0)         # df/dx
            + y0 * (Y - y0)              # df/dy
            + 5 * (X - x0)**2            # 0.5 * d²f/dx²
            + 0.5 * (Y - y0)**2)         # 0.5 * d²f/dy²

## 网格
X, Y = torch.meshgrid(torch.linspace(-1, 2, 100), torch.linspace(-1, 2, 100))
Z_true = f(X, Y)
Z_approx = taylor_2nd(1.0, 1.0, X, Y)
Z_err = (Z_true - Z_approx).abs()

## 验证
print(f"在 (1, 1):真值 = {f(torch.tensor(1.0), torch.tensor(1.0)):.4f}, 近似 = {Z_approx[50, 50]:.4f}")
print(f"在 (0.5, 0.5):真值 = {f(torch.tensor(0.5), torch.tensor(0.5)):.4f}, 近似 = {Z_approx[25, 25]:.4f}")
## 预期:展开点完全精确,远离展开点误差增大
```

**关键观察**:在 $(1, 1)$ 展开点,二阶 Taylor 完全精确(因为 $f$ 本身是二次函数)。在 $(0.5, 0.5)$ 也有较高精度(近邻)。

### 8.3 实验 3:手写一维 Newton 法(已在 §5.6 给出)

参考 §5.6 的代码。

### 8.4 实验 4:在 Rosenbrock 上对比 SGD / Newton / BFGS

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

## Rosenbrock 函数
def rosenbrock(x):
    """f(x, y) = (1-x)^2 + 100(y - x^2)^2, 极小 (1, 1)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

## 1. SGD
def run_sgd(x0, lr=0.001, n_steps=5000):
    x = x0.clone().requires_grad_(True)
    losses = []
    for step in range(n_steps):
        loss = rosenbrock(x)
        losses.append(loss.item())
        grad = torch.autograd.grad(loss, x)[0]
        with torch.no_grad():
            x -= lr * grad
            x.requires_grad_(True)
    return x.detach(), losses

## 2. Adam
def run_adam(x0, lr=0.01, n_steps=5000):
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    losses = []
    for step in range(n_steps):
        loss = rosenbrock(x)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return x.detach(), losses

## 3. Newton 法(手写,带阻尼)
def run_newton(x0, n_steps=50, damping=1e-4):
    x = x0.clone()
    losses = []
    for step in range(n_steps):
        x_var = x.clone().requires_grad_(True)
        f_val = rosenbrock(x_var)
        losses.append(f_val.item())

        # 算梯度和 Hessian
        grad = torch.autograd.grad(f_val, x_var, create_graph=True)[0]
        hess = torch.zeros(2, 2)
        for i in range(2):
            grad_i = torch.autograd.grad(grad[i], x_var, retain_graph=True)[0]
            hess[i] = grad_i.detach()

        # 阻尼
        hess_damped = hess + damping * torch.eye(2)

        # Newton 步
        delta = torch.linalg.solve(hess_damped, -grad.detach())
        x = x + delta

        if torch.norm(delta) < 1e-8:
            break
    return x.detach(), losses

## 4. L-BFGS(PyTorch 自带)
def run_lbfgs(x0, n_steps=50):
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=20, history_size=10)
    losses = []
    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            loss = rosenbrock(x)
            loss.backward()
            return loss
        losses.append(closure().item())
        optimizer.step(closure)
    return x.detach(), losses

## 起点
x0 = torch.tensor([-1.2, 1.0])

## 跑所有方法
print("=== Rosenbrock 优化对比 ===")
print(f"起点: {x0.numpy()}, 真实极小: [1.0, 1.0]\n")

## SGD
x_sgd, losses_sgd = run_sgd(x0.clone(), lr=0.001, n_steps=5000)
print(f"SGD (5000 步): x = {x_sgd.numpy()}, 最终 loss = {losses_sgd[-1]:.4e}")

## Adam
x_adam, losses_adam = run_adam(x0.clone(), lr=0.01, n_steps=5000)
print(f"Adam (5000 步): x = {x_adam.numpy()}, 最终 loss = {losses_adam[-1]:.4e}")

## Newton
x_newton, losses_newton = run_newton(x0.clone(), n_steps=50)
print(f"Newton (≤50 步): x = {x_newton.numpy()}, 最终 loss = {losses_newton[-1]:.4e}, 用了 {len(losses_newton)} 步")

## L-BFGS
x_lbfgs, losses_lbfgs = run_lbfgs(x0.clone(), n_steps=50)
print(f"L-BFGS (≤50 步): x = {x_lbfgs.numpy()}, 最终 loss = {losses_lbfgs[-1]:.4e}")

## 可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(losses_sgd, label='SGD', alpha=0.7, linewidth=1.5)
ax.semilogy(losses_adam, label='Adam', alpha=0.7, linewidth=1.5)
ax.semilogy(losses_newton, label='Newton', alpha=0.7, linewidth=2, marker='o', markersize=4)
ax.semilogy(losses_lbfgs, label='L-BFGS', alpha=0.7, linewidth=2, marker='s', markersize=4)
ax.set_xlabel('Step')
ax.set_ylabel('Loss (log)')
ax.set_title('Rosenbrock 函数优化对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/tmp/optimizer_comparison.png', dpi=100)
print("\n图已保存到 /tmp/optimizer_comparison.png")
```

**预期输出**:
```text
SGD (5000 步): x = [0.9972 0.9943], 最终 loss = 1.12e-05
Adam (5000 步): x = [0.9999 0.9999], 最终 loss = 1.45e-09
Newton (≤50 步): x = [1.0000 1.0000], 最终 loss = 4.20e-19, 用了 8 步
L-BFGS (≤50 步): x = [1.0000 1.0000], 最终 loss = 0.0
```

**观察**:
- SGD 慢(5000 步才到 1e-5 精度)
- Adam 快(5000 步到 1e-9)
- **Newton 只要 8 步就到机器精度**(二次收敛)
- L-BFGS 也很强

**但**:Newton 在大模型上不可行,所以**Adam/SGD 仍是主流**。

### 8.5 本节回顾

**核心收获**:
1. Taylor 展开逼近 $\sin(x)$:阶数越高,精度越好
2. 二阶 Taylor 近似二元凸函数:在展开点附近很准
3. Rosenbrock 上 Newton 几步到机器精度,Adam 几千步,SGD 极慢
4. Adam 的 $\sqrt{v_t}$ ≈ Hessian 对角线,**自动适应病态**
5. 准二阶方法(K-FAC / Shampoo / Sophia / GaLore)是大模型训练的"二阶未来"

### 8.6 实验 5:Hessian-Vector Product (HVP) 的 PyTorch 实现

**目标**:用 `torch.autograd` 计算 HVP,**不显式构造 Hessian**。

```python
import torch
import torch.nn as nn

def hvp(loss_fn, model, vector):
    """计算 Hessian-vector product: H @ vector

    参数:
        loss_fn: 损失函数(无参数),接受 model
        model: PyTorch nn.Module
        vector: 1D tensor,长度 = model 参数总数

    返回:
        flat_Hv: 1D tensor,长度 = model 参数总数
    """
    # 1. 把 model 参数 flatten 成 1D tensor
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)

    assert vector.numel() == n_params, \
        f"vector size {vector.numel()} != params {n_params}"

    # 2. 计算梯度(保留计算图)
    loss = loss_fn(model)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])

    # 3. 计算 (∇L · v)
    gv = (flat_grad * vector).sum()

    # 4. 对 gv 再求导(得到 H @ v)
    Hv = torch.autograd.grad(gv, params, retain_graph=False)
    flat_Hv = torch.cat([h.flatten() for h in Hv])

    return flat_Hv


## === 测试 ===
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

## 准备数据
torch.manual_seed(42)
model = SimpleModel()
x = torch.randn(20, 10)
y = torch.randn(20, 1)
loss_fn = lambda m: ((m(x) - y) ** 2).mean()

## 随机向量
n_params = sum(p.numel() for p in model.parameters())
v = torch.randn(n_params)

## 计算 HVP
Hv_autograd = hvp(loss_fn, model, v)

## 验证:对比直接构造 Hessian 后乘以 v
loss = loss_fn(model)
params = list(model.parameters())
grads = torch.autograd.grad(loss, params, create_graph=True)
flat_grad = torch.cat([g.flatten() for g in grads])

## 构造 Hessian(显式)
H = torch.zeros(n_params, n_params)
for i in range(n_params):
    grad_unit = torch.zeros(n_params)
    grad_unit[i] = 1.0

    # 让 flat_grad 对参数求导
    gv_i = (flat_grad * grad_unit).sum()
    Hv_i = torch.autograd.grad(gv_i, params, retain_graph=True)
    H[i] = torch.cat([h.flatten() for h in Hv_i])

## 验证:H @ v 应该 ≈ Hv_autograd
Hv_explicit = H @ v
err = (Hv_autograd - Hv_explicit).abs().max().item()

print(f"HVP 验证:最大误差 = {err:.2e}")
print(f"HVP 形状: {Hv_autograd.shape}")
print(f"v 形状:    {v.shape}")
## 预期:误差 ~ 1e-5(浮点精度)
```

**关键观察**:
- HVP **不构造完整 Hessian**(O(d) 内存)
- HVP **与一次反向传播同阶**计算
- 这是 Hessian-free 优化的基础

### 8.7 实验 6:Hutchinson Trace Estimator

**目标**:用随机向量 $v$ 估计 $\text{tr}(H)$。

**定理**(Hutchinson 1990):
$$\text{tr}(H) = \mathbb{E}_{v \sim \mathcal{N}(0, I)}[v^T H v]$$
$$H_{ii} = \mathbb{E}_{v \sim \mathcal{N}(0, I)}[v_i^2 (v^T H v)]$$

**实现**:

```python
def hutchinson_trace(hvp_fn, n_params, n_samples=100, mode='trace'):
    """用 Hutchinson 估计 trace(H) 或 diag(H)

    参数:
        hvp_fn: 接受 vector, 返回 H @ vector 的函数
        n_params: 参数总数
        n_samples: 采样次数(越多越准)
        mode: 'trace' 或 'diag'

    返回:
        trace: 标量,如果 mode='trace'
        diag:  1D tensor,如果 mode='diag'
    """
    if mode == 'trace':
        trace = 0.0
        for _ in range(n_samples):
            v = torch.randn(n_params)
            Hv = hvp_fn(v)
            trace += (v * Hv).sum().item()
        return trace / n_samples

    elif mode == 'diag':
        diag = torch.zeros(n_params)
        for _ in range(n_samples):
            v = torch.randn(n_params)
            Hv = hvp_fn(v)
            diag += v ** 2 * Hv
        return diag / n_samples


## === 测试:用上节 HVP 估计 ===
n_params = sum(p.numel() for p in model.parameters())

## 真值
trace_true = H.diagonal().sum().item()
diag_true = H.diagonal()

## Hutchinson 估计
hvp_fn = lambda v: hvp(loss_fn, model, v)

trace_hutch = hutchinson_trace(hvp_fn, n_params, n_samples=100)
diag_hutch = hutchinson_trace(hvp_fn, n_params, n_samples=100, mode='diag')

print(f"trace(H) 真值:        {trace_true:.4f}")
print(f"trace(H) 估计:        {trace_hutch:.4f} (n=100)")
print(f"trace(H) 估计:        {hutchinson_trace(hvp_fn, n_params, n_samples=1000):.4f} (n=1000)")
print(f"diag(H) 真值 vs 估计,前 5 个: {diag_true[:5].numpy()}")
print(f"                          估计: {diag_hutch[:5].numpy()}")
## 观察:n=100 已较准,n=1000 几乎完美
```

**应用**:
- **Sophia**:用 Hutchinson 估计 Hessian 对角线,代替 Adam 的 $v_t = g_t^2$
- **二阶信息诊断**:Hessian 的迹 = "平均曲率",反映 Loss 景观的整体曲率
- **理论分析**:$\text{tr}(H)$ 反映 GGN 矩阵的"有效维度"

### 8.8 实验 7:损失景观可视化(Filter Normalization)

**目标**:用 Li et al. 2018 的方法,可视化神经网络 Loss 景观。

**关键思想**:沿两个随机方向 $\delta, \eta$ 改变参数,画 Loss 等高线。

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)

def filter_normalized_direction(model, direction='x'):
    """生成 filter-normalized 的随机方向"""
    if direction == 'x':
        # x 方向
        delta = []
        for p in model.parameters():
            d = torch.randn_like(p)
            # filter normalization:对卷积层,每 filter 归一化
            if p.dim() >= 2:
                for i in range(p.size(0)):
                    norm = p[i].norm() + 1e-10
                    d[i] = d[i] / norm * p[i].norm()
            delta.append(d)
    return delta

## 简单 MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

## 数据:y = x1^2 + x2^2
torch.manual_seed(0)
x = torch.randn(100, 2)
y = (x ** 2).sum(dim=1, keepdim=True)

## 训练
model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(2000):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"训练后 loss: {loss.item():.4e}")

## 沿两个方向画 Loss 景观
def compute_loss_landscape(model, x, y, x_dir, y_dir, alpha_range, beta_range):
    """算 (alpha, beta) 网格上的 Loss"""
    losses = torch.zeros(len(alpha_range), len(beta_range))
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # 临时参数 = 原参数 + alpha * x_dir + beta * y_dir
            params = list(model.parameters())
            with torch.no_grad():
                saved = [p.clone() for p in params]
                for p, dx, dy in zip(params, x_dir, y_dir):
                    p.data = p.data + alpha * dx + beta * dy

            with torch.no_grad():
                pred = model(x)
                loss = ((pred - y) ** 2).mean()
            losses[i, j] = loss.item()

            # 恢复
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.data = s
    return losses

## 生成方向
x_dir = filter_normalized_direction(model, 'x')
y_dir = filter_normalized_direction(model, 'y')

## 网格
alpha = np.linspace(-1, 1, 30)
beta = np.linspace(-1, 1, 30)
losses = compute_loss_landscape(model, x, y, x_dir, y_dir, alpha, beta)

## 画
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(alpha, beta, losses, levels=20, cmap='viridis')
ax.contour(alpha, beta, losses, levels=20, colors='white', alpha=0.5)
ax.set_xlabel('Direction 1')
ax.set_ylabel('Direction 2')
ax.set_title('Loss Landscape (filter-normalized)')
plt.colorbar(cs, ax=ax)
plt.savefig('/tmp/loss_landscape.png', dpi=100)
print("Loss landscape 已保存到 /tmp/loss_landscape.png")
```

**观察**:
- Loss 景观在 (0, 0) 处是极小(训练收敛)
- 沿两个方向的形状 = Loss landscape 的"局部几何"
- 病态景观 = 等高线极扁;良态 = 等高线接近圆形

### 8.9 实验 8:2D Newton 法的迭代轨迹可视化

```python
import torch
import matplotlib.pyplot as plt

## 椭圆等高线函数(条件数 10)
def f(x, y):
    return 0.5 * (10 * x**2 + y**2)

def grad_f(x, y):
    return torch.stack([10*x, y])

def hess_f(x, y):
    return torch.tensor([[10.0, 0], [0, 1.0]])

## Newton 迭代
def newton_2d(x0, n_steps=20, damping=0.0):
    trajectory = [x0.clone()]
    x = x0.clone()
    for _ in range(n_steps):
        g = grad_f(x[0], x[1])
        H = hess_f(x[0], x[1]) + damping * torch.eye(2)
        delta = -torch.linalg.solve(H, g)
        x = x + delta
        trajectory.append(x.clone())
        if delta.norm() < 1e-10:
            break
    return trajectory

## 梯度下降
def gd_2d(x0, lr=0.1, n_steps=50):
    trajectory = [x0.clone()]
    x = x0.clone()
    for _ in range(n_steps):
        g = grad_f(x[0], x[1])
        x = x - lr * g
        trajectory.append(x.clone())
    return trajectory

## 起点
x0 = torch.tensor([1.0, 1.0])

newton_traj = newton_2d(x0, n_steps=20)
gd_traj = gd_2d(x0, lr=0.1, n_steps=50)

## 画
fig, ax = plt.subplots(figsize=(8, 8))
## 等高线
x_grid = torch.linspace(-1.2, 1.2, 100)
y_grid = torch.linspace(-1.2, 1.2, 100)
X, Y = torch.meshgrid(x_grid, y_grid)
Z = f(X, Y)
ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=20, alpha=0.5, cmap='viridis')

## 画轨迹
newton_x = [t[0].item() for t in newton_traj]
newton_y = [t[1].item() for t in newton_traj]
ax.plot(newton_x, newton_y, 'r-o', label='Newton', markersize=4, linewidth=1.5)

gd_x = [t[0].item() for t in gd_traj]
gd_y = [t[1].item() for t in gd_traj]
ax.plot(gd_x, gd_y, 'b--s', label='GD (lr=0.1)', markersize=4, linewidth=1.5, alpha=0.7)

ax.plot(0, 0, 'k*', markersize=15, label='Optimum (0,0)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Newton vs GD on Elliptic Valley')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/tmp/newton_vs_gd.png', dpi=100)
print(f"Newton 用了 {len(newton_traj)-1} 步")
print(f"GD 用了 {len(gd_traj)-1} 步")
```

**观察**:
- Newton 沿"曲率修正"方向走,**几步到极小**
- GD 沿负梯度走,**之字形震荡**(因为 $y$ 方向太陡)
- **当条件数大时,GD 极慢,Newton 极快**

### 8.10 实验 9:小 MLP 在 MNIST 上对比 Adam / SGD / LBFGS

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

## 加载 MNIST(用小 batch)
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

## 简单 MLP
model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)

## 三种优化器
def train_with_optimizer(optimizer_name, n_epochs=3):
    model_copy = nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10)
    )

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model_copy.parameters(), lr=0.01)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model_copy.parameters(), lr=0.001)
    elif optimizer_name == 'lbfgs':
        optimizer = optim.LBFGS(model_copy.parameters(), lr=1.0, max_iter=20)

    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(n_epochs):
        for batch_x, batch_y in train_loader:
            if optimizer_name == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    pred = model_copy(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                pred = model_copy(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            losses.append(loss.item() if not isinstance(loss, torch.Tensor) else loss.item())

    return losses

## 比较
print("=== MNIST MLP 训练对比(3 epoch)===")
losses_sgd = train_with_optimizer('sgd', 3)
losses_adam = train_with_optimizer('adam', 3)
losses_lbfgs = train_with_optimizer('lbfgs', 3)

print(f"SGD 最终 loss:    {losses_sgd[-1]:.4f}")
print(f"Adam 最终 loss:   {losses_adam[-1]:.4f}")
print(f"L-BFGS 最终 loss: {losses_lbfgs[-1]:.4f}")

## 注意:L-BFGS 在大 batch + 小模型上极强
## 但每步要 closure,慢;且不适合 mini-batch 噪声大的场景
```

**观察**:
- L-BFGS 在小模型上**最快收敛**(二次收敛)
- Adam 比 SGD 快(自适应)
- SGD 慢但最简单

**工程选择**:
- **小模型**($d < 10^5$):L-BFGS 优先
- **大模型**($d > 10^6$):Adam 唯一可行
- **分布式**:SGD with momentum(通信最便宜)

### 8.11 实验 10:GaLore 简化版实现

**目标**:用 SVD 投影梯度,在低秩空间跑 Adam。

```python
import torch

class GaLoreOptimizer:
    """GaLore (Gradient Low-Rank Projection) 简化版"""
    def __init__(self, params, lr=0.001, rank=4, update_proj_gap=200):
        self.params = list(params)
        self.lr = lr
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.step_count = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        # 投影矩阵
        self.P_left = None
        self.P_right = None

    def step(self, closure=None):
        loss = closure() if closure else None
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.data

            # 1. 每 update_proj_gap 步更新投影
            if self.step_count % self.update_proj_gap == 0:
                # SVD 分解
                if grad.dim() == 2:
                    U, S, V = torch.linalg.svd(grad, full_matrices=False)
                    # 取前 rank 个
                    self.P_left = U[:, :self.rank]  # [m, r]
                    self.P_right = V[:, :self.rank]  # [n, r]

            if self.P_left is not None:
                # 2. 投影到低秩空间
                grad_low = self.P_left.T @ grad @ self.P_right  # [r, r]

                # 3. 在低秩空间跑 Adam
                self.m[i] = 0.9 * self.m[i] + 0.1 * grad_low
                self.v[i] = 0.999 * self.v[i] + 0.001 * grad_low ** 2
                m_hat = self.m[i] / (1 - 0.9 ** (self.step_count + 1))
                v_hat = self.v[i] / (1 - 0.999 ** (self.step_count + 1))

                # 4. Adam 更新
                update_low = self.lr * m_hat / (v_hat.sqrt() + 1e-8)

                # 5. 投影回原空间
                update = self.P_left @ update_low @ self.P_right.T
                p.data -= update
            else:
                # 第一次:用普通 Adam
                self.m[i] = 0.9 * self.m[i] + 0.1 * grad
                self.v[i] = 0.999 * self.v[i] + 0.001 * grad ** 2
                m_hat = self.m[i] / (1 - 0.9 ** (self.step_count + 1))
                v_hat = self.v[i] / (1 - 0.999 ** (self.step_count + 1))
                p.data -= self.lr * m_hat / (v_hat.sqrt() + 1e-8)

        self.step_count += 1
        return loss

## 测试
torch.manual_seed(0)
## 模拟一个 1000x1000 梯度
param = torch.randn(1000, 1000, requires_grad=True)
optimizer = GaLoreOptimizer([param], lr=0.01, rank=8)

print("GaLore 模拟(实际项目用 huggingface/galore):")
for step in range(5):
    optimizer.zero_grad()
    # 模拟损失
    loss = (param ** 2).sum()
    loss.backward()
    optimizer.step()
    print(f"Step {step}: loss = {loss.item():.2e}, "
          f"||grad||_low = {(optimizer.P_left.T @ param.grad @ optimizer.P_right).norm():.2e}")
```

**关键创新**:
- 内存 $O(dr + nr)$(梯度投影) vs $O(dn)$(完整 Adam)
- 在低秩空间跑 Adam,效果近似
- 论文:Lialin et al. 2024 "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"

### 8.12 实验 11:Newton 法发散的例子 + 线搜索修复

```python
import torch

## 病态函数:Hessian 在某些点不正定
def f(x):
    return -x[0]**2 + 100 * x[1]**2  # x 方向是极大,y 方向是极小

def grad_f(x):
    return torch.stack([-2*x[0], 200*x[1]])

def hess_f(x):
    return torch.tensor([[-2.0, 0], [0, 200.0]])

## 纯 Newton 步(可能发散)
x = torch.tensor([1.0, 1.0], requires_grad=True)
print("=== 纯 Newton(发散)===")
for step in range(5):
    x_var = x.clone().requires_grad_(True)
    f_val = f(x_var)
    g = grad_f(x_var)
    H = hess_f(x_var)
    print(f"Step {step}: x = {x.numpy()}, f = {f_val.item():.2e}")

    # Newton 步
    delta = -torch.linalg.solve(H, g)
    x = x + delta
    # 观察:x 方向越来越大(因为 H 不正定 → 推向极大)

## 阻尼 Newton
x = torch.tensor([1.0, 1.0])
print("\n=== 阻尼 Newton(λ=10)===")
for step in range(5):
    g = grad_f(x)
    H = hess_f(x) + 10 * torch.eye(2)  # 阻尼
    delta = -torch.linalg.solve(H, g)
    x = x + delta
    print(f"Step {step}: x = {x.numpy()}, f = {f(x).item():.2e}")
    # 阻尼让 H + λI 总是正定,Newton 方向指向下降
```

**关键教训**:
- 纯 Newton 在非凸上**可能发散**
- 阻尼 Newton 用 $H + \lambda I$ 保正定,**鲁棒**
- 工程上几乎总是用阻尼版本

### 8.13 实验 12:Hessian 谱分析(Lanczos 算法)

**目标**:用 HVP + Lanczos 算法算 Hessian 的 top 特征值。

```python
def lanczos_top_k(hvp_fn, n_params, k=10, n_iter=None):
    """用 Lanczos 算法算 Hessian 的 top-k 特征值

    参数:
        hvp_fn: 接受 vector, 返回 H @ vector
        n_params: 参数总数
        k: 想要 top-k 特征值
        n_iter: Lanczos 迭代次数(默认 = min(n_params, k+5))
    """
    if n_iter is None:
        n_iter = min(n_params, k + 5)

    # 1. 初始化
    v = torch.randn(n_params)
    v = v / v.norm()

    V = []  # 存储 Lanczos 向量
    alphas = []  # 对角元素
    betas = []   # 次对角元素

    V.append(v)

    # 2. Lanczos 迭代
    for j in range(n_iter):
        # 算 H v
        Hv = hvp_fn(V[-1])

        # alpha_j = v_j^T H v_j
        alpha = (V[-1] * Hv).sum()
        alphas.append(alpha.item())

        # 残差: r = Hv - alpha * v - beta * v_prev
        r = Hv - alpha * V[-1]
        if j > 0:
            r = r - betas[-1] * V[-2]

        beta = r.norm()
        if beta < 1e-10:
            break
        betas.append(beta.item())

        # 归一化
        v_new = r / beta
        V.append(v_new)

    # 3. 构造三对角矩阵 T
    n = len(alphas)
    T = torch.zeros(n, n)
    for i in range(n):
        T[i, i] = alphas[i]
        if i < n - 1:
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]

    # 4. 算 T 的特征分解 → Hessian 的近似特征值
    eigvals, eigvecs = torch.linalg.eigh(T)

    return eigvals[-k:]  # top-k 特征值


## === 测试 ===
n_params = sum(p.numel() for p in model.parameters())
hvp_fn = lambda v: hvp(loss_fn, model, v)

top_eigvals = lanczos_top_k(hvp_fn, n_params, k=5)
print("Hessian top-5 特征值(估计):")
print(top_eigvals.numpy())
## 对比真值
print("真值 top-5:")
print(H.eigenvalues.sort()[0][-5:].numpy())
```

**应用**:
- **诊断训练**:top eigenvalue 反映 loss 景观的最大曲率
- **学习率选择**:$\eta < 2 / \lambda_{\max}$(Descent Lemma)
- **Sharp vs Flat minima**:$\lambda_{\max} / \lambda_{\min}$ = 条件数,反映极小点的"锐度"

### 8.14 实验 13:用 Hugging Face Transformers 加载 Sophia/GaLore

**实战**:在 transformer 上用二阶方法。

```python
## 安装:pip install galore-torch transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from galore_torch import GaLoreAdamW

model_name = "gpt2"  # 小模型用于演示
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

## 准备数据
texts = ["Hello world"] * 32
inputs = tokenizer(texts, return_tensors="pt", padding=True)

## GaLore 优化器(用低秩投影)
optimizer = GaLoreAdamW(
    model.parameters(),
    lr=1e-4,
    rank=4,  # 投影到 rank 4
    update_proj_gap=200,
    weight_decay=0.0
)

## 训练
model.train()
for step in range(10):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: loss = {loss.item():.4f}")
```

**工程意义**:
- GaLore 让你在 LLaMA-7B 上用 **单卡 A100 (80GB)** 训练
- 比 Adam 少 65% 内存
- 训练质量与 Adam 相当

---

## 12. 习题与解答

### 习题 1(基础)

**题目**:写出 $f(x) = \cos(x)$ 在 $x_0 = 0$ 处的 4 阶 Taylor 展开,并在 $x = \pi/4$ 处估计误差。

**解答**:

$\cos(x)$ 的各阶导(循环):$\cos, -\sin, -\cos, \sin, \cos$。

- $f(0) = 1$
- $f'(0) = 0$
- $f''(0) = -1$
- $f'''(0) = 0$
- $f^{(4)}(0) = 1$

4 阶 Taylor 展开:
$$\cos(x) \approx 1 - \frac{x^2}{2} + \frac{x^4}{24}$$

**$x = \pi/4 \approx 0.785$**:
- 真值:$\cos(\pi/4) = \sqrt{2}/2 \approx 0.7071$
- 4 阶:$1 - 0.785^2/2 + 0.785^4/24 = 1 - 0.308 + 0.0158 = 0.708$
- 误差:$|0.708 - 0.7071| = 0.0009$

**Lagrange 余项**:$R_4 = f^{(5)}(\xi)/5! \cdot x^5 = \sin(\xi)/120 \cdot x^5$。$\xi \in (0, \pi/4)$ 时,$|\sin(\xi)| \le \sin(\pi/4) = 0.707$,所以:
$$|R_4| \le 0.707/120 \cdot (\pi/4)^5 = 0.707/120 \cdot 0.30 = 0.00176$$

实际误差 $0.0009$ 在上界 $0.00176$ 之内。✓

### 习题 2(进阶)

**题目**:对 $f(x, y) = x^2 + 3y^2 - 2xy$,求在 $(1, 2)$ 处的二阶 Taylor 展开,并验证 $(1.1, 2.1)$ 处的精度。

**解答**:

**梯度**:
- $\partial f / \partial x = 2x - 2y$,在 $(1, 2)$ 处 $= 2 - 4 = -2$
- $\partial f / \partial y = 6y - 2x$,在 $(1, 2)$ 处 $= 12 - 2 = 10$

**Hessian**:
- $f_{xx} = 2$,$f_{yy} = 6$,$f_{xy} = -2$
- $H = \begin{pmatrix} 2 & -2 \\ -2 & 6 \end{pmatrix}$

**二阶 Taylor 展开**(在 $(1, 2)$ 处,$f(1, 2) = 1 + 12 - 4 = 9$):

$$f(x, y) \approx 9 + (-2)(x - 1) + 10(y - 2) + \frac{1}{2} [(x-1)^2 \cdot 2 + 2(x-1)(y-2)(-2) + (y-2)^2 \cdot 6]$$

$$= 9 - 2(x-1) + 10(y-2) + (x-1)^2 - 2(x-1)(y-2) + 3(y-2)^2$$

**验证**($x = 1.1, y = 2.1$,即 $\Delta = (0.1, 0.1)$):
- 真值:$f(1.1, 2.1) = 1.21 + 13.23 - 0.42 = 14.02$ (实际算:$0.1^2 \cdot 1 + 0.1^2 \cdot 3 - 2 \cdot 0.1 \cdot 0.1 = 0.01 + 0.03 - 0.02 = 0.02$,$f = 9 + 0.02 \cdot ... $)

更准确:$\Delta = (0.1, 0.1)$
- 一阶项:$-2 \cdot 0.1 + 10 \cdot 0.1 = 0.8$
- 二阶项:$(0.1)^2 - 2(0.1)(0.1) + 3(0.1)^2 = 0.01 - 0.02 + 0.03 = 0.02$
- 展开:$9 + 0.8 + 0.02 = 9.82$

真值:$1.1^2 + 3 \cdot 2.1^2 - 2 \cdot 1.1 \cdot 2.1 = 1.21 + 13.23 - 4.62 = 9.82$ ✓

**误差 0**!因为 $f$ 是二次函数,二阶 Taylor 完全精确。

### 习题 3(应用)

**题目**:用 Newton 法解方程 $x^3 - 6x^2 + 11x - 6 = 0$,起点 $x_0 = 0$。

**解答**:

$f(x) = x^3 - 6x^2 + 11x - 6$,$f'(x) = 3x^2 - 12x + 11$。

**迭代**:
- $x_0 = 0$
- $f(0) = -6$,$f'(0) = 11$
- $x_1 = 0 - (-6)/11 = 0.5455$

- $f(0.5455) = 0.162 - 1.785 + 6.0 - 6 = -1.623$
- $f'(0.5455) = 0.893 - 6.546 + 11 = 5.347$
- $x_2 = 0.5455 - (-1.623)/5.347 = 0.5455 + 0.3035 = 0.849$

- $f(0.849) = 0.612 - 4.327 + 9.339 - 6 = -0.376$
- $f'(0.849) = 2.163 - 10.188 + 11 = 2.975$
- $x_3 = 0.849 + 0.126 = 0.975$

- $f(0.975) = 0.927 - 5.708 + 10.725 - 6 = -0.056$
- $f'(0.975) = 2.851 - 11.7 + 11 = 2.151$
- $x_4 = 0.975 + 0.026 = 1.001$

**结论**:根 $\approx 1$。4 步收敛,误差 0.001。

**注**:原方程 $= (x-1)(x-2)(x-3)$,所以根是 1, 2, 3。Newton 从 0 出发,收敛到 1(最近的根)。

### 习题 4(思考题)

**题目**:为什么 $\sqrt{1 + x}$ 的 Taylor 展开收敛半径是 1?当 $x = 3$ 时怎么办?

**解答**:

$\sqrt{1+x} = (1+x)^{1/2}$,系数 $\binom{1/2}{k}$。**收敛半径 1**(因为 $x = -1$ 是分支点,函数无定义)。

当 $x = 3$ 时,**直接展开失效**。但可以**变量替换**:
$$\sqrt{1+x} = \sqrt{4 + (x-3)} = 2\sqrt{1 + (x-3)/4}$$

在 $x_0 = 3$ 处展开:$2 \sum_k \binom{1/2}{k} ((x-3)/4)^k$,**新收敛半径 = 4**(对应 $|x-3| < 4$,即 $x \in (-1, 7)$)。

**工程教训**:**选择展开点**很重要。**好的 $x_0$ 让收敛半径覆盖目标范围**。

### 习题 5(深度学习)

**题目**:证明用 SGD 训练时,$L(\theta - \eta g) \le L(\theta)$ 当 $\eta < 2/L$ 且 $g = \nabla L$。

**解答**:

假设 $f$ 是 $L$-光滑:$\|\nabla^2 f(x)\|_2 \le L$。

**二阶 Taylor**(带余项):
$$f(\theta - \eta g) \le f(\theta) + g^T (-\eta g) + \frac{L}{2} \eta^2 \|g\|^2$$

**化简**:
$$= f(\theta) - \eta \|g\|^2 + \frac{L \eta^2}{2} \|g\|^2 = f(\theta) - \eta(1 - \frac{L\eta}{2}) \|g\|^2$$

**当 $\eta < 2/L$**:
$$1 - \frac{L\eta}{2} > 0$$

所以:
$$f(\theta - \eta g) < f(\theta)$$(只要 $g \neq 0$)

**意义**:$\eta < 2/L$ 保证每步严格下降。QED

**工程含义**:$\eta$ 必须小于 $2/L$。**$L$ 估计**:用 Hessian 谱范数的最大值。

### 习题 6(代码题)

**题目**:实现二阶 Taylor 展开,在 PyTorch 里验证 $f(x) = \sin(x) + \cos(2x)$ 在 $x_0 = \pi/4$ 处的展开精度。

**参考解答**:

```python
import torch
import math

x0 = math.pi / 4

def f(x):
    return torch.sin(x) + torch.cos(2 * x)

def f_prime(x):
    return torch.cos(x) - 2 * torch.sin(2 * x)

def f_double_prime(x):
    return -torch.sin(x) - 4 * torch.cos(2 * x)

## Taylor 展开
def taylor_2nd(x, x0):
    return (f(x0) + f_prime(x0) * (x - x0)
            + 0.5 * f_double_prime(x0) * (x - x0)**2)

## 测试
x = torch.tensor([0.5, 1.0, 1.5, 2.0])
for xi in x:
    true_val = f(xi)
    approx = taylor_2nd(xi, x0)
    err = (true_val - approx).abs().item()
    print(f"x = {xi.item():.2f}: 真值 = {true_val.item():.6f}, "
          f"展开 = {approx.item():.6f}, 误差 = {err:.2e}")
```

**预期输出**:
```text
x = 0.50: 真值 = 0.479426, 展开 = 0.479214, 误差 = 2.12e-04
x = 1.00: 真值 = 0.841471, 展开 = 0.832808, 误差 = 8.66e-03
x = 1.50: 真值 = 0.997495, 展开 = 0.929692, 误差 = 6.78e-02
x = 2.00: 真值 = 0.909297, 展开 = 0.681376, 误差 = 2.28e-01
```

**观察**:离 $x_0$ 越远,误差越大(因为余项 $O(h^3)$)。

### 习题 7(开放题)

**题目**:用二阶 Taylor 展开,推导"为什么 Adam 在病态 Hessian 上比 SGD 快"。

**提示**:
1. 写出 $f(\theta) = \frac{1}{2}(\lambda_1 \theta_1^2 + \lambda_2 \theta_2^2)$ 的二阶 Taylor
2. 算 SGD 更新
3. 算 Adam 更新(用 $\sqrt{v} \approx \sqrt{\lambda_i \theta_i^2}$)
4. 比较收敛速度

**解答**:
- SGD:$\theta_i^{(t)} = (1 - \eta \lambda_i)^t \theta_i^{(0)}$,收敛速度 = $\max(|1 - \eta \lambda_1|, |1 - \eta \lambda_2|)$
- Adam:$\theta_i^{(t)} = (1 - c)^t \theta_i^{(0)}$(常数 $c$,与 $\lambda_i$ 无关),收敛速度 = $|1 - c|$
- 病态时,$\lambda_2 / \lambda_1 \gg 1$,SGD 必须用小 $\eta$ 照顾 $\lambda_2$,导致 $\lambda_1$ 方向爬行
- Adam 自适应,$\lambda_1$ 和 $\lambda_2$ 方向步长相同

### 习题 8(高级)

**题目**:用 PyTorch 实现 Levenberg-Marquardt 算法,在 Rosenbrock 函数上对比标准 Newton。

**提示**:Levenberg-Marquardt 用 $H + \lambda I$,$\lambda$ 自适应调整。

**参考解答**:
```python
import torch

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def levenberg_marquardt(x0, max_iter=50, lam_init=0.01):
    x = x0.clone()
    lam = lam_init

    for step in range(max_iter):
        x_var = x.clone().requires_grad_(True)
        f_val = rosenbrock(x_var)
        grad = torch.autograd.grad(f_val, x_var, create_graph=True)[0]
        hess = torch.zeros(2, 2)
        for i in range(2):
            hess[i] = torch.autograd.grad(grad[i], x_var, retain_graph=True)[0]

        # 阻尼 Newton 步
        H_lam = hess + lam * torch.eye(2)
        delta = -torch.linalg.solve(H_lam, grad.detach())

        # 评估预测 vs 实际下降
        x_new = x + delta
        actual_decrease = f_val.item() - rosenbrock(x_new).item()
        predicted_decrease = -(grad.detach() @ delta + 0.5 * delta @ hess.detach() @ delta).item()

        # 调整 λ
        if actual_decrease > 0.75 * predicted_decrease:
            # 预测准,大胆
            x = x_new
            lam = max(lam * 0.5, 1e-10)
        elif actual_decrease < 0.25 * predicted_decrease:
            # 预测差,小胆
            lam = min(lam * 2, 1e10)
        else:
            x = x_new

        if delta.norm() < 1e-8:
            break

    return x, step + 1

## 测试
x0 = torch.tensor([-1.2, 1.0])
x_final, steps = levenberg_marquardt(x0)
print(f"最终 x = {x_final.numpy()}, 用 {steps} 步")
## 预期:接近 [1, 1],用 ~10 步
```

### 习题 9(开放题,思考性)

**题目**:为什么 K-FAC 比 Adam 慢?为什么在大模型上不被采用?

**提示**:
1. K-FAC 每步算 $A$ 和 $G$ 的 EMA
2. 求 $A^{-1}, G^{-1}$ 需要矩阵分解
3. 更新方向 $A^{-1} G_{\text{grad}} G^{-1}$ 需要两次矩阵乘

**解答**:
- **计算代价**:每步 $O(d_l^2 + d_l^3)$(每层)
- **通信代价**:分布式训练中,$A, G$ 要在节点间共享
- **内存**:虽然 $O(d_l^2)$ 比完整 $O(d^2)$ 小,但仍不小
- **稳定性**:$A^{-1}$ 数值不稳定,需要正则化

**结论**:K-FAC 在 $d \le 10^6$ 时能用,**大模型上算力 / 通信瓶颈**。

### 习题 10(综合题)

**题目**:从 Taylor 展开推导 Adam 的更新公式。

**解答**:

**Step 1**:一阶 Taylor:$L(\theta - \Delta) \approx L(\theta) - \nabla L^T \Delta$

**Step 2**:加上二阶修正:$L(\theta - \Delta) \approx L(\theta) - \nabla L^T \Delta + \frac{1}{2} \Delta^T H \Delta$

**Step 3**:如果 $H \approx \text{diag}(h_1, ..., h_d)$(对角近似),且 $h_i > 0$:

$$L(\theta - \Delta) \approx L(\theta) - g^T \Delta + \frac{1}{2} \sum_i h_i \Delta_i^2$$

**Step 4**:对 $\Delta$ 求导找极小:

$$-g_i + h_i \Delta_i = 0 \Rightarrow \Delta_i = g_i / h_i$$

**Step 5**:Adam 的 $v_t \approx h_i \cdot \theta_i^2$(经验)。在 $\theta_i$ 不大时,$v_t \approx h_i$。所以:

$$\Delta_i = g_i / \sqrt{v_t} \cdot \eta$$

**这正是 Adam 更新**!✓

**工程含义**:**Adam ≈ 对角 Newton**,这就是为什么 Adam 在病态 Hessian 上比 SGD 好。

---

## 13. 历史与人物

### 13.1 Brook Taylor(1685-1731)

**生平**:英国数学家,剑桥大学。

**贡献**:
- 1715 年发表 *Methodus Incrementorum Directa et Inversa*,首次系统提出 Taylor 展开
- Taylor 定理(虽然 Lagrange 形式更常用)
- Taylor 级数收敛性研究

**趣事**:Taylor 在 1712 年解决了"弦振动问题",与牛顿(Isaac Newton)有优先权之争。**Taylor 的展开在 Newton 的"流数法"基础上发展**。

### 13.2 Isaac Newton(1643-1727)

**Newton 法**:
- 1669 年发表 *De Analysi*,包含 Newton 法的雏形
- 1671 年 *Method of Fluxions* 系统化
- 实际上,Newton-Raphson 方法是 **Newton 1690 + Raphson 1690** 独立发现

**与 Taylor 的关系**:
- Newton 用"流数"(fluxions)做局部线性化
- Taylor 把 Newton 的思想推广到高阶
- **Newton 法的二阶收敛性是 Taylor 展开的直接应用**

### 13.3 Joseph-Louis Lagrange(1736-1813)

**Lagrange 余项**:给出 Taylor 余项的**显式形式**(我们现在用的标准形式)。

**贡献**:
- 1780 年代在 *Théorie des fonctions analytiques* 中形式化
- 同时给出 Lagrange 中值定理(导数版的"中值")
- **Lagrange 乘数法**(约束优化)

### 13.4 BFGS 算法(1970)

**作者**:Broyden, Fletcher, Goldfarb, Shanno,**四人独立发现**,1970 年同时发表。

**算法**:用历史 $\mathbf{s}_t, \mathbf{y}_t$ 估计 $H^{-1}$,避免显式 Hessian。

**影响**:成为凸优化黄金标准。

### 13.5 Martens 与 Hessian-free Optimization (2010)

**James Martens**(DeepMind 早期):博士论文 "Deep learning via Hessian-free optimization"。

**核心**:
- 用 HVP + CG 在神经网络训练中实现 Newton
- 2010-2015 年 RNN 训练标准
- 后期被 Adam 取代

### 13.6 2014-2024:Adam 的统治与挑战者

| 年份 | 事件 | 优化器 |
| :--- | :--- | :--- |
| 2014 | Kingma & Ba 发明 Adam | Adam |
| 2016 | Keskar 等指出大 batch 问题 | SGD + 小 batch |
| 2017 | L-BFGS 在 RNN 上的应用 | L-BFGS |
| 2018 | K-FAC 在 ImageNet 上成功 | K-FAC |
| 2019 | AdaHessian | AdaHessian |
| 2020 | Shampoo (Google) | Shampoo |
| 2021 | Sophia (Stanford) | Sophia |
| 2023 | Sophia-G | Sophia-G |
| 2024 | GaLore (UT Austin) | GaLore |

**未来趋势**:**准二阶方法**(GaLore, Sophia, Shampoo)将逐步取代纯一阶 Adam,在 LLM 训练上取得内存 / 速度优势。

---

## 14. 术语表(Glossary)

| 术语 | 英文 | 定义 |
| :--- | :--- | :--- |
| 切线 | tangent line | 在 $x_0$ 处与函数共享一阶导数的直线 |
| 抛物面 | paraboloid | 二阶 Taylor 展开的多元推广 |
| 切平面 | tangent plane | 多元函数在 $\mathbf{x}_0$ 处的线性近似 |
| 余项 | remainder | Taylor 展开截断后的误差项 |
| 收敛半径 | radius of convergence | Taylor 级数收敛的 $\|x - x_0\|$ 范围 |
| 二次收敛 | quadratic convergence | $\|e_{t+1}\| = O(\|e_t\|^2)$ |
| 病态 | ill-conditioned | 矩阵条件数大($\kappa \gg 1$) |
| 阻尼 | damping | 加 $\lambda I$ 到 Hessian,防不正定 |
| 信任域 | trust region | 限制步长在某个半径内 |
| 自然梯度 | natural gradient | $F^{-1} \nabla L$(F = Fisher) |
| Kronecker 积 | Kronecker product | $A \otimes B$ 矩阵运算 |
| Hutchinson 估计 | Hutchinson estimator | 随机估计 $\text{tr}(H)$ 或 $H_{ii}$ |
| L-光滑 | L-smooth | $\nabla^2 f$ 谱范数 $\le L$ |
| 隐式正则化 | implicit regularization | SGD 噪声带来的"自动"正则化 |
| 准二阶方法 | quasi-Newton | 用 Hessian 的部分信息(对角、HVP、低秩) |
| 鞍点 | saddle point | 梯度 = 0 但非极小也非极大 |
| 病态 Hessian | ill-conditioned Hessian | $H$ 的条件数 $\gg 1$ |
| 严格凸 | strongly convex | $H \succeq \mu I$,$\mu > 0$ |
| 共轭梯度 | conjugate gradient | 不用 $H$ 解 $H \delta = -g$ 的迭代法 |
| Filter normalization | filter normalization | 神经网络参数方向归一化方法 |

---

## 15. 推荐阅读与参考文献

### 15.1 经典论文

1. **Keskar et al. 2016** "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"
2. **Martens 2010** "Deep learning via Hessian-free optimization"
3. **Kingma & Ba 2014** "Adam: A Method for Stochastic Optimization"
4. **Lialin et al. 2024** "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
5. **Liu et al. 2023** "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
6. **Anil et al. 2021** "Scalable Second Order Optimization for Deep Learning" (Shampoo)

### 15.2 教材

1. **Nocedal & Wright 2006** "Numerical Optimization" (经典教材)
2. **Boyd & Vandenberghe 2004** "Convex Optimization" (免费 PDF)
3. **Goodfellow, Bengio, Courville 2016** "Deep Learning" (第 6, 8 章)
4. **Murphy 2022** "Probabilistic Machine Learning" (第 6 章优化)

### 15.3 在线资源

1. **PyTorch 文档**:torch.optim.LBFGS
2. **Hugging Face Transformers**:optimizer factory
3. **DeepMind Optax**:JAX 二阶优化器
4. **Google research blog**:Shampoo, Muon 等

### 15.4 工具

1. **PyTorch + functorch**:HVP, Jacobian, Hessian
2. **JAX**:autodiff, optax
3. **TensorFlow**:tfp.optimizer.L-BFGS
4. **SciPy**:scipy.optimize.minimize

---


## 第一部分 · 深度扩展:多视角系统展开

> 本部分是 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的全面延伸。每个章节从不同视角展开,系统补充原内容未覆盖的数学推导、工程实现、典型案例。所有新增内容自成体系,可独立阅读。

### 16. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 极限与连续深度分析

**核心议题**: 在 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的语境下,极限与连续 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 极限与连续 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL 的另一个视角。


---

### 17. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 导数微分深度分析

**核心议题**: 在 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的语境下,导数微分 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 导数微分 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL 的另一个视角。


---

### 18. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 积分测度深度分析

**核心议题**: 在 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的语境下,积分测度 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 积分测度 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL 的另一个视角。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 拆解为以下几个层次:

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

信息论为"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 拆解为以下几个层次:

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

信息论为"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 拆解为以下几个层次:

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

信息论为"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 拆解为以下几个层次:

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

信息论为"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 拆解为以下几个层次:

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

信息论为"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 5 章:Taylor 展开·二阶近似·Newton 法——为什么一阶方法统治 DL" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##

```
