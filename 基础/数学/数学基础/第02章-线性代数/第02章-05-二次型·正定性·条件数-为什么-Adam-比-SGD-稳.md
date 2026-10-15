---
title: "第02章-05-二次型·正定性·条件数-为什么-Adam-比-SGD-稳"
tags:
  - 数学
  - 二次型
  - 正定性
  - 条件数
  - 为什么
  - Adam
description: "本章要点：们合起来,回答一个贯穿深度学习训练的核心问题:"
chapter_id: 数学-5-02
last_updated: 2026-07-14
---

# 第02章-05-二次型·正定性·条件数-为什么-Adam-比-SGD-稳

们合起来,回答一个**贯穿深度学习训练的核心问题**:
>
> > **"为什么 Adam 在搜推广 Embedding 上比 SGD 稳得多?为什么 Newton 法在病态问题上崩?为什么 BN/LN 能加速收敛?"**
>
> 答案藏在三个看似独立、实则一脉相承的线代概念里:**二次型、正定性、条件数**。
>
> - **二次型** $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 是"任何损失函数在局部的二阶近似"(Taylor 展开的二次项)
> - **正定性** 决定了二次型的"形状":椭圆(凸)还是双曲(鞍)
> - **条件数** 衡量"形状有多扁":圆(好)还是细长的沟壑(差)
>
> 这三件套是**深度学习训练动力学**的"X 光片"。Loss landscape 长什么样、GD/SGD 收敛多快、Adam 为何能自适应,本质上都是**Hessian 矩阵的谱性质**问题。
>
> **学完你能回答**:
> - 二次型 $\mathbf{x}^T A \mathbf{x}$ 是什么,为什么 MSE loss 就是二次型
> - 正定矩阵的 4 个等价判据,什么场景下用哪个
> - 条件数 $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ 怎么算、怎么影响训练
> - 为什么搜推广 Embedding 的 Hessian 特别病态,Adam 为何能救
> - 强凸性 $\mu I \preceq H \preceq L I$ 怎么给出 GD 的线性收敛率
> - 面试 L7/L8 难度的 5 个高频问题
>
> **与前后章节的关系**:
> - 第 3 章 Cholesky 分解 ↔ 本章 SPD 矩阵的判定
> - 第 4 章特征值 ↔ 本章正定性的等价判据、谱条件数
> - **1-高数 第 3 章 Hessian** ↔ 本章"Hessian 决定 loss 形状"
> - **1-高数 第 5 章 Taylor** ↔ 本章"二次型 = Taylor 展开的二次项"
> - **1-高数 第 10 章凸分析** ↔ 本章"正定 = 强凸 = 收敛率保证"
> - 6-最优化第 5 章 ↔ 本章条件数在优化中的角色
> - 7-系统思维 ↔ 搜推广案例对"病态"的工程应对
>
> 准备好了吗?我们从工程师每天都在用的两个 optimizer 开始。

---

## 1. 直觉引入:为什么 Adam 比 SGD 稳?

### 1.1 工程师的痛点:搜推广 Embedding 上 SGD 死活不收敛

**场景 1:线性回归,MSE 训练,什么都好用**。

你训一个 `nn.Linear(input_dim=10, output_dim=1)`,用 `SGD(lr=0.01)`,loss 平稳从 0.5 → 0.1 → 0.01。30 行代码,10 分钟搞定。

**场景 2:搜推广双塔召回,Embedding 几十亿 ID**。

你的 Embedding 表有 10 亿个 item,每个 ID 一个 64 维向量。热门 ID(周杰伦、林俊杰)每天出现 100 万次,冷门 ID(某 1995 年发表的小众论文)整周出现 0 次。**用 SGD 训练,loss 在前 1000 步震荡,热门 ID 的 embedding 飞了,冷门 ID 几乎不更新,val AUC 还掉了。**

换 Adam。同样的 lr,前 200 步就稳定,val AUC 一路涨。**为什么?**

**场景 3:你看了论文,知道"Adam 是自适应学习率"。但面试官追问"自适应"具体做了什么,为什么搜推广场景特别需要,你开始支吾。**

这章就是来打通这一层的。答案藏在**Hessian 矩阵的形状**里。

### 1.2 直觉:Hessian 矩阵的"形状"决定了优化的难度

回忆 1-高数第 3 章:**任何二阶连续可微的标量函数 $f(\mathbf{x})$,在某点 $\mathbf{x}_0$ 附近可以 Taylor 展开到二阶**:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^T H(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)$$

其中 $H(\mathbf{x}_0)$ 是 **Hessian 矩阵**($\mathbf{x}_0$ 处的二阶偏导矩阵)。在极小值附近,$\nabla f = 0$,所以 loss 局部形态**完全由 Hessian 决定**:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^T H (\mathbf{x} - \mathbf{x}_0)$$

这个**二次项**就是本章的核心对象,也就是**二次型**。它长什么样,**优化就有多难**。

### 1.3 ASCII 可视化:凸 landscape vs 病态 landscape

**好条件(well-conditioned)的损失**:Hessian 各方向曲率接近,$f(x,y) = x^2 + y^2$:

```text
        z
        ↑
        │  ╱──╲
        │ │    │     ← 圆形等高线(俯视)
        │ │    │        曲率均匀
        │  ╲──╱
        └───────→ x
              ╲
               → y
```

俯视(看等高线):

```text
        y
        ↑
        │   ┌──────┐
        │  │        │    ← 圆
        │  │   •    │       梯度方向均匀
        │   └──────┘
        └──────────→ x
           (• = 最优解)
```

GD 走的方向几乎就是"指向最优"的直线,收敛快。

**病态(ill-conditioned)的损失**:一个方向曲率大、另一个方向曲率小,$f(x,y) = x^2 + 100 y^2$:

```text
        z
        ↑
        │  ╱╲          ← 像一条狭长的"山谷"
        │ │  ╲╲
        │ │    ╲╲
        │ │      ╲╲
        └─┴────────→ x
            ↑ y 方向(陡)
```

俯视:

```text
        y
        ↑
        │  ┌────────────┐
        │  │              │   ← 极扁的椭圆
        │  │ •            │      x 方向:平(梯度小)
        │  └──────────────┘      y 方向:陡(梯度大)
        └──────────────────→ x
            (• = 最优解)
```

GD 沿"陡"的 y 方向来回震荡(因为梯度大、步长大),沿"平"的 x 方向爬行(梯度小、步长小)。**震荡 + 慢爬 = 收敛极慢**。这就是"病态"。

**Adam 的本质**:**给每个参数单独算一个学习率**。在 y 方向(梯度大)用小 lr,在 x 方向(梯度小)用大 lr。**等价于把椭圆"压圆"**,然后再走 GD。**几何上,Adam 是 preconditioner**(后面 §6.3 详述)。

### 1.4 二次型 = 优化的"指纹"

**核心命题**:

> 任何函数在某点附近的二阶行为,完全由 Hessian 矩阵的二次型 $\mathbf{x}^T H \mathbf{x}$ 决定。
>
> 因此:**学习二次型 = 学习优化的局部几何**。

这一章三件套,本质上都在回答同一个问题:

| 工具 | 问什么 | ML/DL 对应 |
| :--- | :--- | :--- |
| **二次型** $\mathbf{x}^T A \mathbf{x}$ | "这个矩阵把球拉成什么形状?" | 任何 loss 在局部的二阶近似 |
| **正定** $A \succ 0$ | "这个形状是椭圆(凸)吗?" | Hessian 正定 = loss 在该点凸 |
| **条件数** $\kappa(A)$ | "椭圆有多扁?" | 椭圆越扁,GD 越慢,Adam 越有用 |

### 1.5 章节路线图

```text
1. 直觉引入                              ← 你在这里
2. 二次型 (Quadratic Form)                ← §200 行(基础)
3. 正定矩阵 (Positive Definite)            ← §300 行(重点)
4. 负定与不定                              ← §80 行
5. 矩阵的条件数 (Condition Number)         ← §250 行(重点)
6. 条件数在 DL 中的应用                     ← §200 行(重点)
7. 强凸性 (Strong Convexity)              ← §150 行
8. 病态 Hessian 与 Adam 的胜利             ← §120 行
9. PyTorch 代码实现                        ← §150 行
10. 常见误区
11. 面试 Q&A
12. 与其他数学工具的联系
```

---

## 2. 二次型 (Quadratic Form)

### 2.1 定义:$\mathbf{x}^T A \mathbf{x}$

**形式定义**:

> 给定 $n \times n$ 矩阵 $A$ 和向量 $\mathbf{x} \in \mathbb{R}^n$,**二次型** 是标量函数:
>
> $$q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$$

注意是**双线性**形式:$\mathbf{x}$ 的每个分量,既出现在"行"也出现在"列"。

**展开**:
$$\mathbf{x}^T A \mathbf{x} = (x_1, \ldots, x_n) \begin{pmatrix} a_{11} & \cdots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{n1} & \cdots & a_{nn} \end{pmatrix} \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix} = \sum_{i,j} a_{ij} x_i x_j$$

### 2.2 对称化:任何二次型可写成对称矩阵形式

**关键事实**:**二次型的值只取决于 $A$ 的对称部分**。

**证明**:把 $A$ 拆成对称和反对称部分:$A = A_s + A_a$,其中 $A_s = (A + A^T)/2$,$A_a = (A - A^T)/2$。

注意 $A_a$ 是反对称:$A_a^T = -A_a$。所以:
$$\mathbf{x}^T A_a \mathbf{x} = (\mathbf{x}^T A_a \mathbf{x})^T = \mathbf{x}^T A_a^T \mathbf{x} = -\mathbf{x}^T A_a \mathbf{x}$$

唯一满足 $c = -c$ 的实数是 $c = 0$。所以 $\mathbf{x}^T A_a \mathbf{x} = 0$,因此:
$$\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T A_s \mathbf{x} = \mathbf{x}^T \frac{A + A^T}{2} \mathbf{x}$$

**工程意义**:讨论正定性、特征值时,我们**总是可以默认 $A$ 是对称的**(否则用 $A_s$ 替代)。这给我们一个关键工具:**谱定理**(第 4 章)适用于二次型。

### 2.3 几何意义:在 $\mathbf{x}$ 点的"二次项"

回忆 Taylor 展开到二阶(1-高数第 5 章):
$$f(\mathbf{x}_0 + \Delta \mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T H(\mathbf{x}_0) \Delta \mathbf{x}$$

最后一项就是**二次型**。物理上:
- 一次项 $\nabla f^T \Delta \mathbf{x}$:线性变化
- 二次项 $\Delta \mathbf{x}^T H \Delta \mathbf{x}$:曲率
- 一次项为零的点($\nabla f = 0$)是临界点,**二次项决定了临界点的性质**(极小/极大/鞍点)

### 2.4 5 个手算例子

#### 例 1:最简单的二次型(单位矩阵)

$A = I$,$\mathbf{x} = (x_1, x_2)^T$。

$$q(\mathbf{x}) = \mathbf{x}^T I \mathbf{x} = x_1^2 + x_2^2$$

**几何**:圆形等高线(俯视看是圆)。正定,完全对称,**好条件**。

#### 例 2:对角矩阵

$A = \text{diag}(3, 1)$。

$$q(\mathbf{x}) = 3 x_1^2 + x_2^2$$

**几何**:椭圆等高线(在 $x_1$ 方向更陡)。正定(特征值都 > 0),**条件数 $\kappa = 3/1 = 3$**(略病态)。

**手算验证**:$\mathbf{x}^T A \mathbf{x} = (x_1, x_2) \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = (3x_1, x_2) \cdot (x_1, x_2)^T = 3x_1^2 + x_2^2$ ✓

#### 例 3:对称矩阵

$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$。

$$q(\mathbf{x}) = 2 x_1^2 + 2 x_1 x_2 + 2 x_2^2 = 2(x_1^2 + x_1 x_2 + x_2^2)$$

**几何**:椭圆(主轴不在坐标轴方向,需要旋转才能对角化)。

**特征值**:第 4 章已算,$\lambda_1 = 3, \lambda_2 = 1$。条件数 $\kappa = 3$。

#### 例 4:不定矩阵

$A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$。

$$q(\mathbf{x}) = x_1^2 - x_2^2$$

**几何**:**双曲等高线**($x_1^2 - x_2^2 = c$)。**非凸**。在原点处 $q = 0$,但沿任意方向出发,$q$ 可正可负(鞍点)。

#### 例 5:高维情形(搜推广场景的近似)

设 Embedding 维度 $n = 4$,$\mathbf{x} = (x_1, x_2, x_3, x_4)^T$ 代表 4 个参数的偏移,$A$ 是 4×4 Hessian 矩阵。

$$q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} a_{ij} x_i x_j$$

如果 $a_{11} = 100$($x_1$ 对应热门 ID 1 的 embedding 梯度大),$a_{44} = 0.01$($x_4$ 对应冷门 ID 4 几乎无梯度),其余 $a_{ii} = 1$,非对角元素为 0:

$$q(\mathbf{x}) = 100 x_1^2 + x_2^2 + x_3^2 + 0.01 x_4^2$$

**条件数** $\kappa = 100 / 0.01 = 10000$。**巨病态**。GD 走 $x_4$ 方向慢得跟蜗牛一样。这就是搜推广 Embedding 的真实困境。

### 2.5 案例:为什么 MSE Loss 是二次型?

**线性回归**模型 $\hat{y} = X \theta$,真实值 $y$,MSE 损失:
$$L(\theta) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N} \|y - X \theta\|_2^2$$

展开:
$$L(\theta) = \frac{1}{N} (y - X \theta)^T (y - X \theta) = \frac{1}{N} (y^T y - 2 y^T X \theta + \theta^T X^T X \theta)$$

整理:
$$L(\theta) = \underbrace{\frac{1}{N} \theta^T X^T X \theta}_{\text{二次型}} \underbrace{- \frac{2}{N} y^T X \theta}_{\text{一次项}} + \underbrace{\frac{1}{N} y^T y}_{\text{常数}}$$

**关键观察**:
- 二次型矩阵 = $\frac{1}{N} X^T X$(**Gram 矩阵**,永远是半正定!见 §3.7)
- 如果 $X$ 列满秩,$X^T X$ 严格正定
- Hessian = $\frac{2}{N} X^T X$ = 这个二次型矩阵的 2 倍

**工程意义**:MSE 损失是**精确的二次函数**,没有更高阶项。这就是为什么线性回归有闭式解 $\hat{\theta} = (X^T X)^{-1} X^T y$,为什么 GD 在 MSE 上是**严格的线性收敛**。

**对比**:**深度网络的 loss 不是二次型**,它有 ReLU 的非线性、Softmax 的指数,有高阶项。但**在极小值附近**(训练后期),Taylor 展开的高阶项可忽略,**局部的 loss ≈ 二次型**。这就是为什么 SGD/Adam 在训练后期表现稳定。

### 2.6 二次型在 DL 中的其他应用

- **核函数** $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$ 是两个向量的内积,内核矩阵 $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ 是 Gram 矩阵 → **半正定**。
- **协方差矩阵** $\Sigma = \mathbb{E}[(\mathbf{x} - \mu)(\mathbf{x} - \mu)^T]$,对任意 $\mathbf{v}$,$\mathbf{v}^T \Sigma \mathbf{v} = \mathbb{E}[(\mathbf{v}^T (\mathbf{x} - \mu))^2] \geq 0$ → **半正定**。
- **Fisher 信息矩阵** $F = \mathbb{E}[\nabla \log p \cdot \nabla \log p^T]$ → **半正定**(信息度量是"非负"的)。
- **L2 正则化** $\frac{\lambda}{2} \|\theta\|_2^2$ 给 loss 加上 $\frac{\lambda}{2} \theta^T I \theta$,Hessian 增加 $\lambda I$ → **让 loss 强凸**(§7)。

---

## 3. 正定矩阵 (Positive Definite)

### 3.1 定义:所有方向上 $\mathbf{x}^T A \mathbf{x} > 0$

**核心定义**(对称矩阵版本):

> 实对称矩阵 $A \in \mathbb{R}^{n \times n}$ 称为**正定**(Positive Definite, PD),记作 $A \succ 0$,如果:
>
> $$\forall \mathbf{x} \in \mathbb{R}^n, \mathbf{x} \neq \mathbf{0}, \quad \mathbf{x}^T A \mathbf{x} > 0$$

**直观含义**:**矩阵 $A$ 决定的二次型 $q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 是一个"开口向上的碗"**,在原点处 $q = 0$,沿任何方向离开原点,$q$ 都严格增加。

```text
        z = q(x)
        ↑
        │      ╱╲         ← 正定:像碗,任何方向上 z > 0
        │    ╱    ╲
        │  ╱        ╲
        │╱            ╲
        └───────────────→ x
              (原点)
```

**对照**:

| 矩阵类型 | 定义 | 等高线 | 几何形状 |
| :--- | :--- | :--- | :--- |
| **正定** $A \succ 0$ | $\forall \mathbf{x} \neq 0, \mathbf{x}^T A \mathbf{x} > 0$ | 椭圆(包围原点) | 椭球面(碗) |
| **半正定** $A \succeq 0$ | $\forall \mathbf{x}, \mathbf{x}^T A \mathbf{x} \geq 0$ | 椭圆 + 直线 | 平底碗 |
| **不定** (Indefinite) | $\exists \mathbf{x}_+, \mathbf{x}_- $ 使 $q(\mathbf{x}_+) > 0, q(\mathbf{x}_-) < 0$ | 双曲线 | 马鞍 |

**为什么要求对称?** §2.2 已证:非对称矩阵的反对称部分对二次型贡献为 0,所以正定性**只与对称部分有关**。但非对称矩阵本身的"正定"概念($\forall \mathbf{x} \neq 0, \mathbf{x}^T A \mathbf{x} > 0$)已经隐含 $A$ 是对称的(否则 $\mathbf{x}^T A \mathbf{x}$ 可能既依赖 $A$ 又依赖 $A^T$)。**约定**:本章"正定"默认指**实对称正定** (Symmetric Positive Definite, SPD)。

### 3.2 半正定矩阵 (Positive Semidefinite, PSD)

**定义**:
$$A \succeq 0 \iff \forall \mathbf{x}, \mathbf{x}^T A \mathbf{x} \geq 0$$

**关键差异**:半正定允许 $q(\mathbf{x}) = 0$ 对某些非零 $\mathbf{x}$ 成立。这些 $\mathbf{x}$ 构成 $A$ 的**零空间 (null space)** 或**核 (kernel)**。

**几何**:平底碗。沿某些方向 $q$ 不变(平的方向就是零空间中的方向)。

```text
        z = q(x)
        ↑
        │   ┌─────┐
        │   │     │       ← 半正定:有些方向 z = 0(平底)
        │   │     │
        │   └─────┘
        └───────────→ x
            ↑ 平的方向(核)
```

**例子 1**:$A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$,$q(\mathbf{x}) = x_1^2$。对 $\mathbf{x} = (0, 1)^T$,$q = 0$。半正定但**不正定**。

**例子 2**(后面 §3.7):任何 Gram 矩阵 $K_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle$ 都是 PSD。当某些 $\mathbf{x}_i$ 线性相关时,$K$ 半正定但奇异(非满秩)。

### 3.3 四个等价判据 (重点)

正定性的判定有 4 个相互等价的判据,工程中根据**矩阵大小、数据结构、计算资源**选择合适的:

#### 判据 1:特征值判据 (适合中等规模矩阵)

$$A \succ 0 \iff \text{所有特征值 } \lambda_i > 0$$

**推导**:$A$ 对称,谱定理 $A = Q \Lambda Q^T$。对任意 $\mathbf{x} \neq 0$,令 $\mathbf{y} = Q^T \mathbf{x}$,则 $\|\mathbf{y}\|_2 = \|\mathbf{x}\|_2 > 0$ (因 $Q$ 正交)。所以:
$$\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T Q \Lambda Q^T \mathbf{x} = \mathbf{y}^T \Lambda \mathbf{y} = \sum_i \lambda_i y_i^2$$

所有 $\lambda_i > 0$ 时,只要 $\mathbf{x} \neq 0$ (即 $\mathbf{y} \neq 0$),$\sum \lambda_i y_i^2 > 0$。反之若有 $\lambda_k \leq 0$,取 $\mathbf{y} = \mathbf{e}_k$,则 $\mathbf{x} = Q \mathbf{e}_k$ 是 $A$ 的特征向量,$\mathbf{x}^T A \mathbf{x} = \lambda_k \leq 0$。

**工程意义**:这是最直接的判据,PyTorch 一行 `torch.linalg.eigvalsh(A).min() > 0`。但计算特征值复杂度 $O(n^3)$,**大矩阵不划算**。

#### 判据 2:顺序主子式判据 (Sylvester 判据, 适合小矩阵)

定义 $k$ 阶顺序主子式 (Leading Principal Minor):
$$\Delta_k = \det(A_{1:k, 1:k})$$
即取 $A$ 的前 $k$ 行、前 $k$ 列组成的子矩阵的行列式。

$$A \succ 0 \iff \Delta_1, \Delta_2, \ldots, \Delta_n \text{ 都} > 0$$

**例子** $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$:
- $\Delta_1 = 2 > 0$ ✓
- $\Delta_2 = \det(A) = 4 - 1 = 3 > 0$ ✓
- $A$ 正定 ✓

**优点**:只需计算行列式(虽然也是 $O(n^3)$,但对小矩阵更直观)。
**缺点**:对大矩阵不实用,且**需要对称矩阵才能用**(非对称矩阵的 LPM 不判定正定性)。

#### 判据 3:分解判据 (最重要、最常用)

$$A \succ 0 \iff \exists B \in \mathbb{R}^{n \times n} \text{ 列满秩, 使 } A = B^T B$$

**推导**:
- ($\Leftarrow$) 若 $A = B^T B$ 且 $B$ 列满秩,则对 $\mathbf{x} \neq 0$,$B \mathbf{x} \neq 0$ (因 $B$ 列满秩)。所以 $\mathbf{x}^T A \mathbf{x} = \|B \mathbf{x}\|_2^2 > 0$。
- ($\Rightarrow$) 若 $A \succ 0$,由谱定理 $A = Q \Lambda Q^T$,$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$,$\lambda_i > 0$。取 $B = \Lambda^{1/2} Q^T$,则 $B$ 列满秩(因为 $\Lambda^{1/2}$ 对角元都 > 0,$Q$ 正交),且 $B^T B = Q \Lambda^{1/2} \Lambda^{1/2} Q^T = Q \Lambda Q^T = A$。

**工程意义**:
- 这个判据给出了**构造正定矩阵的方法**:任意 $B^T B$ 都是 PSD,$B$ 列满秩时是 PD。
- $B$ 不唯一,但有一个特别重要的特例:**Cholesky 分解** ($B$ 下三角)。

#### 判据 4:Cholesky 分解存在性 (数值上最常用)

$$A \succ 0 \iff A \text{ 有 Cholesky 分解 } A = L L^T, \ L \text{ 下三角, 对角元} > 0$$

**为什么是"最常用"**:
- Cholesky 分解**只需要** $O(n^3/3)$ FLOPs,比特征值分解快约 3 倍
- 它**自带判定**:如果分解过程中碰到 `sqrt(非正数)`,就说明 $A$ 不是 PD
- 数值上极其稳定(LDL 变种甚至不用开方)

**算法骨架**(Cholesky-Banachiewicz):
```text
for i = 1 to n:
    L[i,i] = sqrt(A[i,i] - sum_{k<i} L[i,k]^2)
    for j = i+1 to n:
        L[j,i] = (A[j,i] - sum_{k<i} L[j,k] * L[i,k]) / L[i,i]
```

**PyTorch 调用**:`torch.linalg.cholesky(A)`。如果 $A$ 不是 PD,抛 `LinAlgError`。

### 3.4 5 个手算例子

#### 例 1:单位矩阵 $I$

$I$ 是 PD:$\mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|_2^2 > 0$ (当 $\mathbf{x} \neq 0$)。特征值都 = 1,Cholesky $L = I$,条件数 = 1。**完全对称**的"圆碗"。

#### 例 2:对角矩阵 $A = \text{diag}(3, 1)$

**判据 1**(特征值):$3 > 0, 1 > 0$ → PD ✓

**判据 2**(LPM):$\Delta_1 = 3, \Delta_2 = 3$ 都 > 0 → PD ✓

**判据 3**(分解):取 $B = \text{diag}(\sqrt{3}, 1)$,则 $B^T B = \text{diag}(3, 1) = A$ ✓

**判据 4**(Cholesky):$L = \text{diag}(\sqrt{3}, 1)$,$L L^T = A$ ✓

**几何**:椭圆等高线,$x_1$ 方向曲率 = 3,$x_2$ 方向曲率 = 1,**条件数** = 3(略病态)。

#### 例 3:对称非对角矩阵 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$

**判据 1**:特征值 $\lambda_1 = 3, \lambda_2 = 1$(第 4 章算过) → PD ✓

**判据 2**:$\Delta_1 = 2, \Delta_2 = 4 - 1 = 3$ 都 > 0 → PD ✓

**判据 4**(Cholesky 手算):
- $L_{11} = \sqrt{A_{11}} = \sqrt{2}$
- $L_{21} = A_{21} / L_{11} = 1 / \sqrt{2} = \sqrt{2}/2$
- $L_{22} = \sqrt{A_{22} - L_{21}^2} = \sqrt{2 - 1/2} = \sqrt{3/2}$

$$L = \begin{pmatrix} \sqrt{2} & 0 \\ \sqrt{2}/2 & \sqrt{3/2} \end{pmatrix}, \quad L L^T = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} \checkmark$$

**手算验证**:$\mathbf{x}^T A \mathbf{x} = 2 x_1^2 + 2 x_1 x_2 + 2 x_2^2 = 2[(x_1 + x_2/2)^2 + 3 x_2^2/4] > 0$ ✓

#### 例 4:不定矩阵 $A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

**判据 1**:特征值 $1, -1$,**有一个 < 0** → 不正定(实际上不定)。

**判据 2**:$\Delta_1 = 1 > 0$,但 $\Delta_2 = -1 < 0$ → 不正定 ✗

**几何**:双曲等高线 $x_1^2 - x_2^2 = c$。**马鞍面**。

#### 例 5:3×3 PD 矩阵(实际工程常用规模)

$$A = \begin{pmatrix} 4 & 2 & 1 \\ 2 & 5 & 3 \\ 1 & 3 & 6 \end{pmatrix}$$

**判据 2**(LPM 验证):
- $\Delta_1 = 4 > 0$ ✓
- $\Delta_2 = 4 \cdot 5 - 2 \cdot 2 = 16 > 0$ ✓
- $\Delta_3 = \det(A) = 4(30 - 9) - 2(12 - 3) + 1(6 - 5) = 84 - 18 + 1 = 67 > 0$ ✓

**结论**:PD ✓

**判据 4**(Cholesky 手算):
- $L_{11} = \sqrt{4} = 2$
- $L_{21} = 2/2 = 1, L_{31} = 1/2$
- $L_{22} = \sqrt{5 - 1^2} = 2$
- $L_{32} = (3 - 1 \cdot 1/2) / 2 = 2.5 / 2 = 5/4$
- $L_{33} = \sqrt{6 - (1/2)^2 - (5/4)^2} = \sqrt{6 - 0.25 - 1.5625} = \sqrt{4.1875} \approx 2.046$

$$L = \begin{pmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ 0.5 & 1.25 & 2.046 \end{pmatrix}, \quad L L^T = A \checkmark$$

### 3.5 案例:Hessian 半正定 → 凸性证明

**核心命题**(1-高数 第 10 章已述,这里给线代证明):

> $f(\mathbf{x})$ 二阶连续可微,则 $f$ **凸** $\iff$ Hessian $H(\mathbf{x}) \succeq 0$ (对所有 $\mathbf{x}$)。

**证明思路**:
- ($\Leftarrow$) 对任意两点 $\mathbf{x}, \mathbf{y}$,Taylor 展开到二阶:
$$f(\mathbf{y}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{1}{2} (\mathbf{y} - \mathbf{x})^T H(\boldsymbol{\xi}) (\mathbf{y} - \mathbf{x}), \quad \boldsymbol{\xi} \in [\mathbf{x}, \mathbf{y}]$$

  若 $H(\boldsymbol{\xi}) \succeq 0$,最后一项 $\geq 0$,所以 $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x})$,这是凸函数的一阶刻画(1-高数第 10 章)。
- ($\Rightarrow$) 反方向用反证法:若 $H(\mathbf{x}_0)$ 不 PSD,则存在 $\mathbf{v}$ 使 $\mathbf{v}^T H(\mathbf{x}_0) \mathbf{v} < 0$,沿 $\mathbf{v}$ 方向二阶 Taylor 展开得 $f(\mathbf{x}_0 + t \mathbf{v}) < f(\mathbf{x}_0)$(对充分小 $t > 0$),违反凸性。

**工程意义**:
- **判断 loss 是否凸**:看 Hessian 的所有特征值是否非负(对所有 $\mathbf{x}$)。深度网络的 loss 一般**非凸**(中间层 ReLU/Softmax 引入非线性),所以训练是**非凸优化**。
- **线性回归的 MSE loss**:Hessian = $\frac{2}{N} X^T X$,**永远 PSD**(实际上 $X$ 列满秩时 PD)。所以 MSE 是**严格的凸函数**,GD 一定收敛到全局最优。
- **交叉熵 + Softmax + Linear**:Hessian 也 PSD(凸组合),但加上深层网络就不凸了。

### 3.6 案例:协方差矩阵总是 PSD

**命题**:对任何随机向量 $\mathbf{X} \in \mathbb{R}^n$,协方差矩阵 $\Sigma = \mathbb{E}[(\mathbf{X} - \mu)(\mathbf{X} - \mu)^T]$ **总是半正定**。

**证明**(几乎 1 行):
$$\forall \mathbf{v}, \quad \mathbf{v}^T \Sigma \mathbf{v} = \mathbb{E}[(\mathbf{v}^T (\mathbf{X} - \mu))^2] \geq 0$$

**核心洞察**:$\mathbf{v}^T (\mathbf{X} - \mu)$ 是随机变量,期望它的平方**必非负**。

**工程意义**:
- GP (高斯过程) 中,核矩阵 $K$ 必须是 PSD(否则 GP 概率模型无意义)。**实际工程上**:用 `RBF + jitter` (加 $10^{-6} I$) 保证数值 PSD。
- PCA 的协方差矩阵是 PSD,做 SVD 后奇异值 = 标准差。
- 推荐的隐式反馈矩阵(用户-物品评分) **不是**协方差矩阵,不能直接假设 PSD,需要预处理(如 `+1` 让最小值变成 0,避免负相关)。

### 3.7 案例:Gram 矩阵 $K_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle$ 总是 PSD

**定义**:给定 $m$ 个向量 $\mathbf{x}_1, \ldots, \mathbf{x}_m \in \mathbb{R}^n$,Gram 矩阵是 $K \in \mathbb{R}^{m \times m}$,$K_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle$。

**命题**:$K \succeq 0$ (半正定)。

**证明**:
$$K = X^T X, \quad X = [\mathbf{x}_1, \ldots, \mathbf{x}_m] \in \mathbb{R}^{n \times m}$$
对任意 $\mathbf{v} \in \mathbb{R}^m$:
$$\mathbf{v}^T K \mathbf{v} = \mathbf{v}^T X^T X \mathbf{v} = \|X \mathbf{v}\|_2^2 \geq 0$$

注意:用判据 3,这里 $K = X^T X$ 是 Gram 分解,**总是半正定**,**当 $X$ 列满秩(即 $\mathbf{x}_1, \ldots, \mathbf{x}_m$ 线性无关)时 $K$ 严格正定**。

**工程意义**:
- **核方法 (Kernel Methods)** 中,Kernel 矩阵 $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ 必须 PSD,否则 SVM/GP 无意义。Mercer 定理保证合法的核函数(线性、RBF、多项式)给出 PSD。
- **RBF Kernel** $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ 总是 PSD(因为是无限维内积)。
- **Embedding 检索中**的相似度矩阵 $K_{ij} = \cos(\mathbf{x}_i, \mathbf{x}_j)$ 是 PSD 矩阵(因为 $\cos = \langle \mathbf{x}_i, \mathbf{x}_j \rangle / (\|\mathbf{x}_i\| \|\mathbf{x}_j\|)$,只要向量归一化)。
- **Hessian of MSE** = $\frac{2}{N} X^T X$,所以**总是 PSD**(✓ 与 §2.5 一致)。

### 3.8 ASCII 可视化:4 个等价判据如何"看"同一个矩阵

```text
                ┌──────────────────────────────────┐
                │         同一个 A (SPD)           │
                └──────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    特征值判据          LPM 判据           Cholesky 判据
    λ_i > 0           Δ_k > 0            A = L L^T
    (谱定理)        (Sylvester)        (算法 O(n³/3))
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                    分解判据 (B 列满秩)
                       A = B^T B
                              │
                              ▼
                         A 是 SPD
```

---

## 4. 负定与不定

### 4.1 负定 (Negative Definite) 与负半定 (Negative Semidefinite)

**定义**:
- **负定** $A \prec 0$:$\forall \mathbf{x} \neq 0, \mathbf{x}^T A \mathbf{x} < 0$
- **负半定** $A \preceq 0$:$\forall \mathbf{x}, \mathbf{x}^T A \mathbf{x} \leq 0$

**关键事实**:$A$ 负定 $\iff$ $-A$ 正定。**判据**:$-A$ 满足 §3.3 任一判据。

**几何**:**倒扣的碗**,在原点处 $q = 0$,沿任何方向离开, $q$ 严格减小。

```text
        z = q(x)
        ↑
        │╲            ╱
        │  ╲        ╱
        │    ╲    ╱      ← 负定:倒碗
        │      ╲╱
        └───────────────→ x
              (原点)
```

**应用**:
- **Newton 法的下降方向**:如果 $H \prec 0$ (局部极大),Newton 方向 $\mathbf{d} = -H^{-1} \nabla f$ 实际指向"上山",需要额外处理(如 damped Newton)。
- **博弈论的 payoff 矩阵**:零和博弈的 payoff 矩阵有时是负定的。

### 4.2 不定 (Indefinite):鞍点的本质

**定义**:$\exists \mathbf{x}_+, \mathbf{x}_-$ 使 $\mathbf{x}_+^T A \mathbf{x}_+ > 0$ 且 $\mathbf{x}_-^T A \mathbf{x}_- < 0$。

**等价判据**:$A$ 不定 $\iff$ $A$ 有正特征值也有负特征值(等价于 $A$ 不是 PSD 也不是 NSD)。

**几何**:**马鞍面**。沿一个方向上升,沿另一个方向下降。

```text
        z = q(x)
        ↑         ╱╲
        │        ╱  ╲        ← 不定:马鞍
        │   ────╱────╲────
        │      ╱      ╲
        │     ╱        ╲
        └────┼──────────┼───→ x
             ↑          ↑
          下山方向    下山方向
             (还有上山方向:垂直纸面)
```

### 4.3 鞍点 (Saddle Point) 与 DL 的关系

**定义**:点 $\mathbf{x}^*$ 是 $f$ 的鞍点,如果 $f(\mathbf{x}^*) = 0$ (在 Hessian 二次型意义下),但 $\mathbf{x}^*$ 既不是局部极大也不是局部极小。

**等价刻画**:$\nabla f(\mathbf{x}^*) = 0$ 且 $H(\mathbf{x}^*)$ **不定**。

**鞍点的"困难"**:
- 梯度为 0,**GD 卡住**(不前进也不后退)
- 但**沿某个方向下降有限,沿另一个方向下降更多**

**深度学习的损失 landscape 中**:
- 鞍点**远比局部极小多**(Dauphin et al. 2014)
- 局部极小的 loss 值往往接近全局最优(Choromanska 2015)
- 所以 GD 在深度学习中**不会卡在鞍点**(因为随机梯度噪声会"踢出"鞍点)

**与正定的关系**:
| 临界点类型 | Hessian 性质 | GD 行为 |
| :--- | :--- | :--- |
| 局部极小 | PD | 收敛(凸区域) |
| 局部极大 | ND | GD 远离 |
| 鞍点 | 不定 | 卡住(但 SGD 噪声会推开) |

### 4.4 一个手算例子

$A = \begin{pmatrix} 2 & 0 \\ 0 & -3 \end{pmatrix}$。

- 特征值 $2, -3$:一正一负 → **不定**
- $\Delta_1 = 2 > 0$,但 $\Delta_2 = -6 < 0$:LPM 不全正 → 不定

手算验证:
- $\mathbf{x} = (1, 0)^T$,$q = 2 > 0$
- $\mathbf{x} = (0, 1)^T$,$q = -3 < 0$
- $\mathbf{x} = (1, 1)^T$,$q = 2 - 3 = -1 < 0$

这是典型鞍点:沿 $x$ 方向上升(凸),沿 $y$ 方向下降(凹)。

### 4.5 案例:Newton 法在鞍点处的失败

**Newton 法的更新**:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - H(\mathbf{x}_t)^{-1} \nabla f(\mathbf{x}_t)$$

**当 $H$ 不定时**(鞍点),$H^{-1}$ **仍然存在**(假设 $\det H \neq 0$)。但 Newton 方向 $\mathbf{d} = -H^{-1} \nabla f$ **行为奇怪**:沿 $H$ 的正特征方向,它**指向极大**(而非极小),**反而把优化器推向"上山"**。

**修复**:
- **Levenberg-Marquardt**:用 $H + \lambda I$ 替代 $H$,$\lambda > 0$ 强制 PD。
- **Trust Region**:在 $\mathbf{x}_t$ 周围约束步长 $\|\mathbf{d}\| \leq \Delta$,避免"飞出去"。
- **现代 DL**:几乎不用纯 Newton 法,用近似二阶方法(K-FAC, Shampoo, Sophia),自动处理不定性。

---

## 5. 矩阵的条件数 (Condition Number)

### 5.1 定义:矩阵放大误差的能力

**核心定义**:

> 矩阵 $A \in \mathbb{R}^{n \times n}$ 可逆,其**条件数**(Condition Number)为
>
> $$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$

其中 $\|\cdot\|$ 是任意**诱导范数**(subordinate matrix norm)。不同范数给出不同的条件数,但**谱条件数**最常用。

**物理直觉**:
- $\|A\|$ = $A$ 把单位向量**最多拉多长**(放大因子上限)
- $\|A^{-1}\|$ = $A^{-1}$ 把单位向量最多拉多长 = $A$ 把单位向量**至少压多扁**(压缩因子上限的倒数)
- **乘积** = $A$ 把某些方向**放大 $L$ 倍**的同时把另一些方向**压缩 $L$ 倍** → "形状歪得多厉害"

**误差传播公式**(为什么重要):
$$\frac{\|\delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \cdot \frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|}$$

其中 $A \mathbf{x} = \mathbf{b}$ 的**输入** $\mathbf{b}$ 有微小扰动 $\delta \mathbf{b}$,输出 $\mathbf{x}$ 的扰动 $\delta \mathbf{x}$ 最多被放大 $\kappa(A)$ 倍。**条件数 = 误差放大系数**。

### 5.2 谱条件数 (最常用)

**定义**:用谱范数 (spectral norm, 即 2-范数),$A$ 对称时:
$$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \frac{|\lambda_{\max}(A)|}{|\lambda_{\min}(A)|}$$

(对一般矩阵,用奇异值;对称矩阵奇异值 = 特征值的绝对值)

**重要性质**:
- $\kappa(A) \geq 1$(因为 $\|A\| \cdot \|A^{-1}\| \geq \|A A^{-1}\| = \|I\| = 1$)
- $\kappa(A) = 1$ $\iff$ $A$ 是**正交矩阵的标量倍**(对对称矩阵就是 $\pm I$,即"完美球"形状)
- $\kappa(A)$ **越大**,$A$ 越"病态"

### 5.3 几何意义:最大/最小方向比值

对 SPD 矩阵 $A = Q \Lambda Q^T$ ($\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$,$\lambda_i > 0$):
- 在特征向量 $\mathbf{q}_i$ 方向,二次型 $q(t \mathbf{q}_i) = \lambda_i t^2$
- "曲率最大方向"对应 $\lambda_{\max}$
- "曲率最小方向"对应 $\lambda_{\min}$
- $\kappa = \lambda_{\max} / \lambda_{\min}$ = **最陡方向 / 最平方向**

**可视化**(2D 椭圆):

```text
         y
         ↑
         │      ╱╲
         │     ╱  ╲
         │    ╱    ╲     ← 椭圆等高线 (x^T A x = c)
         │   ╱      ╲
         │  ╱   ╱    ╲
         │ │   ╱      │
         │ │  ╱  •    │   ← 极小点(原点)
         │ │ ╱        │
         │  ╲─────────╱
         └──────────────→ x
            ↑          ↑
         最陡方向   最平方向
         (λ_max)    (λ_min)
```

**条件数与椭圆形状**:
- $\kappa = 1$:椭圆 = 圆(完美好条件)
- $\kappa = 10$:椭圆略扁
- $\kappa = 100$:椭圆明显扁
- $\kappa = 10^4$:椭圆细长如针(典型搜推广 Embedding 的 Hessian)
- $\kappa = \infty$:$A$ 奇异,某些方向完全压扁(零曲率)

### 5.4 病态 (Ill-Conditioned) vs 良态 (Well-Conditioned)

**经验阈值**(工程界共识):

| 条件数 $\kappa$ | 状态 | 数值稳定性 | GD 行为 |
| :--- | :--- | :--- | :--- |
| $1 - 10$ | 优秀 | 极稳定 | 收敛快 |
| $10 - 100$ | 良好 | 稳定 | 正常 |
| $100 - 10^3$ | 略病态 | 略有误差 | 需调学习率 |
| $10^3 - 10^6$ | 病态 | 数值解不可信 | GD 慢 |
| $> 10^6$ | 严重病态 | 直接报错 | GD 不收敛 |

**注意**:阈值因问题而异。**稀疏线性方程组** $\kappa > 10^4$ 就崩溃;**稠密 Hessian** 在 DL 中 $\kappa > 10^6$ 才算"病得离谱"。

### 5.5 5 个手算例子

#### 例 1:单位矩阵 $A = I$

$\kappa(I) = 1$(谱范数 = 1,逆矩阵 = $I$,范数 = 1)。

**几何**:圆形等高线。**完美好条件**。

#### 例 2:对角矩阵 $A = \text{diag}(3, 1)$

$\sigma_{\max} = 3, \sigma_{\min} = 1$,$\kappa = 3/1 = 3$。

**几何**:略扁椭圆,$\kappa$ 略 > 1。**良好条件**。

#### 例 3:对称矩阵 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$

第 4 章算过 $\lambda_1 = 3, \lambda_2 = 1$,所以 $\kappa = 3$。

#### 例 4:病态矩阵 $A = \text{diag}(100, 1)$

$\kappa = 100$。

**几何**:明显扁椭圆。沿 $x_1$ 方向曲率 = 100(陡),沿 $x_2$ 方向曲率 = 1(平)。**GD 在 $x_2$ 方向爬行极慢**。

#### 例 5:搜推广 Embedding 的近似 $A = \text{diag}(100, 1, 1, 0.01)$

$\lambda_{\max} = 100, \lambda_{\min} = 0.01$,$\kappa = 10000$。

**含义**:
- 热门 ID ($x_1$ 方向)梯度大、Hessian 元素大 → 步长必须小,否则震荡
- 冷门 ID ($x_4$ 方向)梯度小、Hessian 元素接近 0 → 步长可以大,但梯度噪声掩盖信号
- **单学习率无法同时优化**:GD 失败

#### 谱条件数的计算验证(PyTorch 风格)

```python
import torch
A = torch.tensor([[3., 0.], [0., 1.]])
eigvals = torch.linalg.eigvalsh(A)  # [1., 3.]
kappa = eigvals.max() / eigvals.min()  # 3.0
```

### 5.6 工程意义:条件数决定两件事

#### (1) 数值解的稳定性

**定理**(误差传播):
$$\frac{\|\delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \cdot \frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|}$$

**例子**:$\kappa(A) = 10^4$,输入 $\mathbf{b}$ 有 $10^{-6}$ 的相对误差,输出 $\mathbf{x}$ 的相对误差可能达 $10^{-2}$(扩大 10000 倍)。

**实际工程影响**:
- **线性方程组求解** $A \mathbf{x} = \mathbf{b}$:用 `torch.linalg.solve` 而不是 `inv(A) @ b`,前者对条件数更鲁棒。
- **最小二乘**:用 QR 或 SVD 分解,**永远不要**用 $A^T A$ 求逆(条件数平方化!)
- **GP / Kernel 方法**:加 `+ 1e-6 * I` (jitter) 改善条件数。

#### (2) 梯度下降的收敛速度

**定理**(强凸 + L-光滑的 GD 收敛率):
$$\|x_{t+1} - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^t \|x_0 - x^*\|^2$$

其中 $\mu = \lambda_{\min}(H), L = \lambda_{\max}(H)$,所以 $L/\mu = \kappa(H)$。

**直观**:
- 收敛率 $= 1 - 1/\kappa$
- $\kappa = 1$:每步误差减少 $\to 0$(几步收敛)
- $\kappa = 10$:每步误差 $\times 0.9$(需要 ~20 步降到 1%)
- $\kappa = 10^4$:每步误差 $\times 0.9999$(需要 ~$10^4$ 步!)

**这是 Adam 存在的根本原因**。SGD 在 $\kappa = 10^4$ 的 Hessian 上,每步只走 $0.01\%$ 的距离,需要上亿步才能收敛。**Adam 用二阶动量 preconditioning,把条件数拉回接近 1**。

### 5.7 案例:为什么 Newton 法在病态 Hessian 上失败?

**Newton 法的"一步到位"假设**:用 $H^{-1}$ 抵消 Hessian 的"病态",**理论上**每步都直接指向极小。

**实际上**:
1. **计算 $H^{-1}$ 本身**就要面对 $\kappa(H)$ 大的问题(求逆放大了输入误差)
2. **Hessian 在深度学习中几乎不正定**(ReLU/Softmax 引入非线性),Newton 方向可能**指向极大**
3. **步长控制**:Newton 法的步长 = 1(完全相信 Hessian),但 $H$ 不准时步长太大,**直接飞出去**

**例子**:$H = \text{diag}(10^4, 1)$,$\kappa = 10^4$。

**Newton 方向**:$\mathbf{d} = -H^{-1} \nabla f = (-g_1/10^4, -g_2)^T$。$x_1$ 方向步长太小, $x_2$ 方向步长太大。**和 GD 的问题一样**,只是 GD 把它们"颠倒了"。

**修复**:
- **Damped Newton**:`$H + \lambda I$`,$\lambda$ 取 $\kappa$ 的上界(常用 $\sqrt{\text{trace}(H)/n}$)
- **L-BFGS**:用历史梯度近似 $H^{-1}$,自动控制步长
- **Trust Region**:在 $\mathbf{x}_t$ 邻域内约束步长

### 5.8 案例:为什么 Adam 用梯度自适应缩放克服病态?

**Adam 的更新**:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

其中 $\hat{v}_t$ 是**梯度平方的移动平均**(二阶动量)。

**核心观察**:$\sqrt{\hat{v}_t}$ **逐元素地**等价于**对角 Hessian 的近似**。

**几何解释**:
- 在梯度大的方向(对应 $H$ 元素大),$\sqrt{v_t}$ 大,有效学习率 $\eta / \sqrt{v_t}$ **变小** → 抑制震荡
- 在梯度小的方向(对应 $H$ 元素小),$\sqrt{v_t}$ 小,有效学习率变大 → 加速爬行
- **结果**:把"病态椭圆"压成"圆",条件数变 1

**严格数学**:
- 真 Hessian $H$ 的对角线 $\text{diag}(H) = [h_{11}, h_{22}, \ldots]$
- 经验上,$h_{ii} \approx \mathbb{E}[(\partial L / \partial \theta_i)^2]$(Fisher 信息矩阵对角线)
- 而 $\mathbb{E}[g_i^2]$ 就是 $v_t$ 的近似

**所以 Adam ≈ 对角 preconditioning**(详见 §6.3)。

---

## 6. 条件数在 DL 中的应用

### 6.1 Hessian 条件数与 SGD 收敛

**深度学习的 loss landscape**:
- 训练初期:loss 下降快,梯度大但粗糙
- 训练中期:loss 进入"平台期",梯度小
- 训练末期:loss 在最优点附近震荡

**Hessian 条件数的演化**:
- 训练初期:Hessian 条件数极大(对应 loss 急剧下降的不均匀性)
- 训练中期:条件数逐渐减小
- 训练末期:在最优点附近,条件数往往**仍然很大**(因为不同参数对最终 loss 的敏感度差异巨大)

**SGD 在高 $\kappa$ 下的震荡模式**:

```text
        loss
         ↑
         │  ──────╮     ← 沿"陡"方向震荡
         │ │      │
         │ │      │  ← 沿"平"方向慢爬
         │ │      │
         └─┴──────┴────→ epoch
           ↑  ↑
          震荡 慢爬
```

SGD 沿最陡方向(主特征方向)反复**过冲**,沿最平方向**缓慢前进**。

### 6.2 Pre-conditioned SGD = 缩小条件数

**Preconditioning 的核心思想**:**找一个矩阵 $P$,把原问题变换成"好条件"的问题**。

**原始 GD**:$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$

**Pre-conditioned GD**:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta P^{-1} \nabla f(\mathbf{x}_t)$$

**理想选择**:$P = H$(Hessian 矩阵),这就是 Newton 法。**问题**:$H$ 太大,无法存储/求逆。

**实际选择**:
- **对角 Preconditioner**:$P = \text{diag}(H)$ 或 $P = \text{diag}(\sqrt{v_t})$(Adam 的做法)
- **AdaGrad**:累积平方梯度作为 preconditioner
- **L-BFGS**:用历史梯度构造低秩 preconditioner
- **K-FAC**:用 Fisher 信息矩阵的 Kronecker 分解构造块对角 preconditioner

**效果**:
- 原始 SGD 收敛率:$(1 - 1/\kappa)^t$
- Pre-conditioned SGD 收敛率:$(1 - 1/\kappa_{\text{new}})^t$,其中 $\kappa_{\text{new}} \approx 1$ → **指数级加速**

### 6.3 Adam 二阶动量 ≈ Hessian 对角线 → Preconditioning

**Adam 完整更新**(PyTorch 风格):
```python
m = beta1 * m + (1 - beta1) * g          # 一阶动量(均值)
v = beta2 * v + (1 - beta2) * g**2       # 二阶动量(未中心化方差)
m_hat = m / (1 - beta1**t)               # bias correction
v_hat = v / (1 - beta2**t)
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

**核心命题**:$\sqrt{v_t}$ **逐元素**近似 Hessian 的**对角线**的平方根。

**直觉**:
- 在稳态,$v_t \approx \mathbb{E}[g^2] = \text{Var}(g) + (\mathbb{E}[g])^2$
- Hessian 对角线 $H_{ii} \approx \mathbb{E}[g_i^2]$(二次型在稳态的曲率)
- 所以 $\sqrt{v_t} \approx \sqrt{H_{ii}}$

**Pre-conditioning 解释**:
$$\theta \leftarrow \theta - \frac{\eta \cdot m}{\sqrt{v} + \epsilon} \approx \theta - \eta \cdot H^{-1/2}_{\text{diag}} \cdot \nabla f$$

这是**对角 Hessian 倒数** preconditioning 的近似。

**对比**:
| Optimizer | Preconditioner 类型 | 缩放后的有效条件数 |
| :--- | :--- | :--- |
| SGD | $I$ | $\kappa(H)$ |
| AdaGrad | $\text{diag}(\sum g^2)$ | $\approx 1$ (前几步) |
| RMSProp | $\text{diag}(\text{EMA}(g^2))$ | $\approx 1$ |
| Adam | $\text{diag}(\text{EMA}(g^2))$ + momentum | $\approx 1$ + 一阶动量加速 |
| L-BFGS | 低秩近似 $H^{-1}$ | $\approx 1$ |

### 6.4 案例:搜推广 Embedding 的 Hessian 为什么病态?

**场景**:
- Embedding 表: $10^9$ 个 ID,每个 64 维
- 热门 ID(周杰伦):每天出现 $10^6$ 次
- 冷门 ID(某 1995 年论文):整周出现 0 次

**Embedding 矩阵 $E \in \mathbb{R}^{10^9 \times 64}$,损失 $L$ 对 $E$ 的梯度**:

```text
热 ID 梯度 (热门 item):
  ∂L/∂e_{hot}  ≈  大(每次出现都有 loss)
  → 累积梯度 ≈ 大值
  → Hessian 对角线 ≈ 大值 (≈ 10000)

冷 ID 梯度 (冷门 item):
  ∂L/∂e_{cold}  ≈  小或 0 (几乎不更新)
  → 累积梯度 ≈ 小值或 0
  → Hessian 对角线 ≈ 小值 (≈ 0.01 或更小)
```

**Hessian 条件数** ≈ 10000 / 0.01 = **$10^6$**。**严重病态**。

**SGD 的失败**:
- 用同一 lr 更新所有 ID
- 热 ID:梯度大,lr 大 → 飞了
- 冷 ID:梯度小,lr 小 → 几乎不动
- **结果**:热门 ID 的 embedding 震荡,冷门 ID 不更新

**Adam 的胜利**:
- 热 ID:$v_t$ 大 → 有效 lr 小 → 抑制震荡
- 冷 ID:$v_t$ 小 → 有效 lr 大 → 加速更新(即使梯度小也走得动)
- **结果**:所有 ID 都被合理更新,val AUC 显著涨

**这就是为什么搜推广默认用 Adam,而不是 SGD**。

### 6.5 案例:BN/LN 如何改善条件数?

**Batch Normalization (BN)**:
- 对每个 batch,标准化激活:$x \leftarrow (x - \mu_{\text{batch}}) / \sigma_{\text{batch}}$
- 引入 2 个可学习参数 $\gamma, \beta$

**为什么能改善条件数**:
- 标准化后,激活的均值 = 0,方差 = 1(每个 channel)
- **前向**:下游层的输入分布稳定,不需要不断适应
- **反向**:梯度流经 BN 时也被标准化,**Jacobian 的条件数变小**

**数学**:
- 假设某层输入 $x$ 的方差巨大,导致输出 $y = W x$ 的方差也巨大
- BN 后 $y \leftarrow \gamma (y - \mu)/\sigma + \beta$,方差被强制 = $\gamma^2$
- 下游层的 Hessian 不再受 $W$ 的范数影响,**条件数显著降低**

**Layer Normalization (LN)** 与 BN 类似,但在**单个样本的特征维度**上标准化(不依赖 batch):
- LN 在 Transformer 中标配,因为 NLP 任务的 batch 维度小/变化大
- LN 也改善条件数,但机制略不同(主要稳定层内分布)

**实证**:He et al. 2015 论文中,ResNet 加 BN 后能用**大学习率**(如 0.1),不加 BN 只能用 $10^{-3}$。背后原因就是 BN 让 Hessian 条件数减小,GD 更稳。

### 6.6 案例:为什么 Transformer 用 Warmup?

**Warmup**:训练初期 lr 从 0 线性/指数增长到目标值,训练一段时间后再 decay。

**没有 Warmup 的失败**:
- Transformer 刚初始化时,所有权重接近 0
- 但 Embedding 表的初始化方差不同(常用 `std=0.02`)
- **初期 Hessian 条件数可能极大**($10^6$ 量级)
- 大 lr 直接炸梯度

**Warmup 的作用**:
- 初期 lr 小(0 → 目标值),让模型"适应"当前的 Hessian 结构
- 几 k 步后 lr 稳定在较大值,此时 Hessian 已经变得"友好"
- 训练中后段再 decay,精细收敛

**实证**:Vaswani et al. 2017 (Attention Is All You Need) 论文标配 Warmup。没有 Warmup,Transformer 在小数据集上几乎不收敛。

**深层数学**:GoogLeNet/Inception 同期工作发现,深度网络训练初期 Hessian 的最大特征值会**先涨后跌**(Loshchilov & Hutter 2017)。Warmup 让 lr 跟上这个曲线,避免"早期爆炸"。

### 6.7 案例:L-BFGS 在小 batch DL 中几乎不用

**L-BFGS 的优势**:用最近 $m$ 步梯度构造 $H^{-1}$ 的低秩近似,比 Newton 便宜得多。**在小规模凸优化中是黄金标准**。

**为什么 DL 几乎不用**:
1. **非凸**:L-BFGS 依赖强凸假设,DL loss 一般非凸
2. **大规模**:每次更新要存 $m$ 个梯度向量 ($m=10 \sim 100$),对亿级参数不现实
3. **Mini-batch**:梯度是噪声的,L-BFGS 的曲率估计不准
4. **Saddle points**:L-BFGS 卡在鞍点附近(Newton 方向指向极大)

**DL 的替代**:
- 大规模:Adam (对角 preconditioning)
- 中规模:K-FAC (Kronecker 分解 preconditioning)
- 小规模:L-BFGS 仍可用(如 small MLP, RNN)

---

## 7. 强凸性 (Strong Convexity)

### 7.1 定义:二次型的下界

**核心定义**:
> 函数 $f: \mathbb{R}^n \to \mathbb{R}$ 称为 $\mu$-**强凸**,如果存在常数 $\mu > 0$ 使:
>
> $$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{\mu}{2} \|\mathbf{y} - \mathbf{x}\|_2^2$$

**几何含义**:**任何切线都被抛物线 $f(\mathbf{x}) + \nabla f^T (\mathbf{y} - \mathbf{x}) + \frac{\mu}{2} \|\mathbf{y} - \mathbf{x}\|^2$ 包在下方**。**最陡的曲率至少是 $\mu$**。

**强凸 vs 凸**:
- 凸:$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f^T (\mathbf{y} - \mathbf{x})$(切线下界)
- $\mu$-强凸:再加一项 $\frac{\mu}{2} \|\mathbf{y} - \mathbf{x}\|^2$(二次下界)

```text
        f
        ↑
        │      ╱╲
        │     ╱  ╲         ← 强凸:比"凸函数最小曲率 μ"还陡
        │    ╱    ╲
        │   ╱      ╲
        │  ╱        ╲
        │ ╱──────────╲ ← 切线
        └───────────────→ x
```

### 7.2 等价刻画:Hessian 半正定的代数判据

**定理**(1-高数第 10 章):
$$f \text{ 是 } \mu\text{-强凸} \iff H(\mathbf{x}) \succeq \mu I, \quad \forall \mathbf{x}$$

即:**Hessian 的最小特征值 $\geq \mu$**(对所有 $\mathbf{x}$)。

**推导**:Taylor 展开到二阶:
$$f(\mathbf{y}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x}) + \frac{1}{2} (\mathbf{y} - \mathbf{x})^T H(\boldsymbol{\xi}) (\mathbf{y} - \mathbf{x})$$

若 $H(\boldsymbol{\xi}) \succeq \mu I$,最后一项 $\geq \frac{\mu}{2} \|\mathbf{y} - \mathbf{x}\|^2$。

### 7.3 等价刻画:Hessian 也有上界 → L-光滑

**对称地**,如果 Hessian 有上界:
$$H(\mathbf{x}) \preceq L I, \quad \forall \mathbf{x}$$

则 $f$ 是 $L$-**光滑**(L-smooth),即 Hessian 的最大特征值 $\leq L$。

**合并**:
$$\mu I \preceq H(\mathbf{x}) \preceq L I, \quad \forall \mathbf{x}$$

则 $f$ 既是 $\mu$-强凸又是 $L$-光滑。**条件数**:
$$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}} \leq \frac{L}{\mu}$$

### 7.4 收敛率:GD 在 $\mu$-强凸 + $L$-光滑上是线性收敛

**定理**:GD 在 $\mu$-强凸 + $L$-光滑函数上,使用学习率 $\eta = 1/L$:
$$\|\mathbf{x}_{t+1} - \mathbf{x}^*\|_2^2 \leq \left(1 - \frac{\mu}{L}\right)^t \|\mathbf{x}_0 - \mathbf{x}^*\|_2^2 = \left(1 - \frac{1}{\kappa}\right)^t \|\mathbf{x}_0 - \mathbf{x}^*\|_2^2$$

**收敛率** = $O(\kappa \log(1/\epsilon))$,即达到 $\epsilon$ 精度需要 $O(\kappa \log(1/\epsilon))$ 步。

**对照**:
- 非强凸 GD: $O(1/\epsilon)$ 步(次线性)
- $\mu$-强凸 GD: $O(\kappa \log(1/\epsilon))$ 步(**线性**,指数级快)

### 7.5 案例:L2 正则化让 loss 强凸

**L2 正则化**:
$$\tilde{L}(\theta) = L(\theta) + \frac{\lambda}{2} \|\theta\|_2^2$$

**对 Hessian 的影响**:
$$H_{\tilde{L}}(\theta) = H_L(\theta) + \lambda I$$

**关键**:即使 $H_L$ 奇异或不定,**加上 $\lambda I$ 后**,$H_{\tilde{L}}$ 的最小特征值 $\geq \lambda > 0$。

**所以**:$\tilde{L}$ 总是 $\lambda$-强凸。

**例子**:
- MSE loss: $H = \frac{2}{N} X^T X$ PSD,但如果 $X$ 不满秩,$H$ 奇异 → 不严格强凸
- 加 L2 后:$H = \frac{2}{N} X^T X + \lambda I$ **严格 PD** → 严格 $\lambda$-强凸

**工程意义**:
- L2 正则化 = **数学上的"万金油"**:既防过拟合(参数小),又保证凸性(优化稳)
- 但 L2 太大会**梯度消失**(主要是 $\lambda \theta$ 项的贡献)
- 现代实践:Adam + Weight Decay (解耦),而非 Adam + L2

### 7.6 强凸性的"边界"

**严格强凸 vs 半强凸**:
- 严格强凸($\mu > 0$):GD **线性收敛**
- 半强凸($\mu = 0$):只是凸函数,GD $O(1/\epsilon)$ 收敛
- 非凸:GD **没有收敛保证**,但实践中**经常 work**(1-高数第 10 章)

**深度学习不是强凸的**:
- ReLU/Softmax/Attention 引入非线性,Hessian 处处变化
- 但**L2 正则化让 loss "局部强凸"**(在某个局部邻域内,$H \succeq \lambda I$)
- 所以深度学习训练**总体非凸**,**局部凸**

---

## 8. 病态 Hessian 与 Adam 的胜利

### 8.1 Hessian 病态的 4 个原因

**(1) 参数频率差异**:Embeddings 出现频率相差 $10^3 \sim 10^6$ 倍,对应梯度幅度差异巨大

**(2) 激活稀疏性**:ReLU 网络大量激活为 0,只有少数"通路"上有梯度,造成 Hessian 在多数方向上"平"

**(3) 归一化不足**:深层网络激活分布不断漂移,某些层输出方差极大,导致下一层 Hessian 病态

**(4) Loss 函数本身**:CrossEntropy + Softmax + 极端标签(全 0/1)使 loss 在某些方向上"几乎线性",Hessian 接近 0

### 8.2 SGD 在病态问题的震荡模式

**Rosenbrock 函数**(经典病态测试):
$$f(x, y) = (1-x)^2 + 100 (y - x^2)^2$$

- 最优点 $(1, 1)$,但**谷底狭长**,$\kappa(H) \approx 2500$

```text
        y
        ↑
        │      ╱╲
        │     ╱  ╲
        │    ╱    ╲    ← Rosenbrock 的"香蕉谷"
        │   ╱      ╲
        │  ╱        ╲
        │ ╱          ╲
        │╱            ╲
        └─────────────→ x
            ↑ (1,1) 在谷底
```

**SGD 行为**:
- 初期:在 $y$ 方向(陡)大幅震荡
- 中期:沿 $x$ 方向(平)爬行极慢
- **结果**:几百步都到不了最优点

### 8.3 Adam 的 3 个修复

**修复 1**:逐元素学习率缩放($\eta / \sqrt{v_t}$)
- 抑制陡方向的震荡
- 加速平方向的爬行
- 把椭圆"压圆"

**修复 2**:一阶动量($m_t$)累积过去梯度
- 减少震荡的方差(类似 EMA)
- 在平方向提供"惯量"

**修复 3**:Bias correction
- 训练初期 $m_t, v_t$ 接近 0(EMA 还没积累)
- 除以 $(1 - \beta^t)$ 修正,让初期 lr 不偏小

### 8.4 案例:SGD vs Adam 在 Rosenbrock 对比

**理论预期**:
- SGD 在 Rosenbrock 上**几百步**才能收敛(震荡 + 慢爬)
- Adam 几十步就能到最优点

**PyTorch 代码**(详细见 §9 实验 4):

```python
import torch
f = lambda xy: (1 - xy[0])**2 + 100 * (xy[1] - xy[0]**2)**2

# SGD
x_sgd = torch.tensor([-1.5, 2.5], requires_grad=True)
opt = torch.optim.SGD([x_sgd], lr=1e-4)
# ... 1000 步后,SGD 还在震荡

# Adam
x_adam = torch.tensor([-1.5, 2.5], requires_grad=True)
opt = torch.optim.Adam([x_adam], lr=0.1)
# ... 100 步就接近 (1,1)
```

**结论**:**Adam 在病态损失上比 SGD 快 10-100 倍**。

### 8.5 进一步的工程方法

Adam 不是万能的。在病态问题上还有:

| 方法 | 机制 | 工程实践 |
| :--- | :--- | :--- |
| **AdamW** | 解耦的 weight decay | Transformer 标配 |
| **LAMB** | Layer-wise adaptive lr | 大 batch 训练 |
| **Lion** | 单值动量 + sign 更新 | 比 Adam 省显存 |
| **Sophia** | Hessian 对角线估计 | GPT-2 训练 |
| **Shampoo** | 完整矩阵 preconditioning | Google 大模型 |

---

## 9. PyTorch 代码实现

### 实验 1:计算矩阵条件数

```python
import torch

def condition_number(A, ord=None):
    """计算矩阵的条件数。"""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    # 谱条件数 (默认)
    if ord in (None, 2):
        # 用 SVD
        sv = torch.linalg.svdvals(A)
        return sv.max() / sv.min()

    # Frobenius 范数条件数(不常用,仅演示)
    elif ord == 'fro':
        return torch.linalg.norm(A, 'fro') * torch.linalg.norm(torch.linalg.inv(A), 'fro')

    # 1-范数条件数
    elif ord == 1:
        col_sum = A.abs().sum(dim=0).max()
        inv_col_sum = torch.linalg.inv(A).abs().sum(dim=0).max()
        return col_sum * inv_col_sum

    else:
        raise NotImplementedError(f"ord={ord} not supported")


# 测试
A = torch.tensor([[3., 1.], [1., 3.]])
print("kappa(I):", condition_number(torch.eye(5)))     # 1.0
print("kappa(A):", condition_number(A))                # ~3.0

# 病态矩阵
B = torch.diag(torch.tensor([100., 1., 0.01]))
print("kappa(B):", condition_number(B))                # 10000
```

### 实验 2:病态 vs 良态 GD 收敛

```python
import torch
import matplotlib.pyplot as plt

# 定义两个二次型 loss
def loss_well(x):  # kappa = 1
    return ((x - torch.tensor([1.0, 1.0]))**2).sum()

def loss_ill(x):   # kappa = 10000
    diff = x - torch.tensor([1.0, 1.0])
    return 100 * diff[0]**2 + 0.01 * diff[1]**2

# GD 在两个 loss 上的收敛曲线
def gd_run(loss_fn, x0, lr, n_steps):
    x = x0.clone().requires_grad_(True)
    losses = []
    for _ in range(n_steps):
        opt = torch.optim.SGD([x], lr=lr)
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

x0 = torch.tensor([0.0, 0.0])
loss_well_hist = gd_run(loss_well, x0, lr=0.1, n_steps=200)
loss_ill_hist  = gd_run(loss_ill,  x0, lr=0.001, n_steps=5000)

# 良态 loss:200 步到 1e-6
# 病态 loss:即使调小 lr,5000 步也只能到 ~0.5(还在震荡)
```

### 实验 3:可视化 $f = x^2 + 100 y^2$ 的等高线 + GD 轨迹

```python
import numpy as np
import matplotlib.pyplot as plt

# 网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + 100 * Y**2

# GD 轨迹
def gd_path(lr, n_steps):
    px, py = -1.5, 2.5
    path = [(px, py)]
    for _ in range(n_steps):
        gx, gy = 2*px, 200*py
        px -= lr * gx
        py -= lr * gy
        path.append((px, py))
    return np.array(path)

path_sgd = gd_path(lr=0.001, n_steps=50)
path_adam_like = gd_path(lr=0.001, n_steps=50)  # 简化版,真正的 Adam 见 §9.4

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cs = axes[0].contour(X, Y, Z, levels=20)
axes[0].plot(path_sgd[:, 0], path_sgd[:, 1], 'r-o', markersize=4)
axes[0].set_title('SGD on $f=x^2+100y^2$')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

cs = axes[1].contour(X, Y, Z, levels=20)
axes[1].plot(path_adam_like[:, 0], path_adam_like[:, 1], 'b-o', markersize=4)
axes[1].set_title('Adam-like (lr_adapted) on $f=x^2+100y^2$')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
plt.show()
```

**预期**:SGD 轨迹在 $y$ 方向(陡)大幅震荡,在 $x$ 方向(平)慢爬;Adam-like 在两个方向都"匀速"前进。

### 实验 4:SGD vs Adam 在病态 loss 上

```python
import torch

# Rosenbrock 函数
def rosenbrock(xy):
    return (1 - xy[0])**2 + 100 * (xy[1] - xy[0]**2)**2

# SGD
x_sgd = torch.tensor([-1.5, 2.5], requires_grad=True)
opt_sgd = torch.optim.SGD([x_sgd], lr=1e-4)
print("SGD step | loss | x, y")
for step in range(0, 2000, 200):
    for _ in range(200):
        opt_sgd.zero_grad()
        loss = rosenbrock(x_sgd)
        loss.backward()
        opt_sgd.step()
    print(f"  {step+200:4d} | {loss.item():.4f} | {x_sgd.tolist()}")

# Adam
x_adam = torch.tensor([-1.5, 2.5], requires_grad=True)
opt_adam = torch.optim.Adam([x_adam], lr=0.05)
print("\nAdam step | loss | x, y")
for step in range(0, 500, 50):
    for _ in range(50):
        opt_adam.zero_grad()
        loss = rosenbrock(x_adam)
        loss.backward()
        opt_adam.step()
    print(f"  {step+50:4d} | {loss.item():.4f} | {x_adam.tolist()}")

# 预期:SGD 2000 步 loss 还有 0.5+
#      Adam 500 步 loss < 0.001
```

### 实验 5:验证 Adam 二阶动量 ≈ Hessian 对角线

```python
import torch

# 简单 2D 二次型 f(x) = 0.5 * x^T H x
H = torch.tensor([[4.0, 0.0], [0.0, 0.01]])  # 条件数 400

def f(x):
    return 0.5 * x @ H @ x

# 手动估计 Hessian 对角线:数值梯度
x = torch.tensor([0.5, 0.5])
eps = 1e-4
h_diag = torch.zeros(2)
for i in range(2):
    e = torch.zeros(2); e[i] = 1
    g_plus = torch.autograd.grad(f(x + eps * e), x)[0][i]
    g_minus = torch.autograd.grad(f(x - eps * e), x)[0][i]
    h_diag[i] = (g_plus - g_minus) / (2 * eps)
print("Hessian 对角线:", H.diag().tolist())  # [4.0, 0.01]
print("数值估计:    ", h_diag.tolist())

# Adam 二阶动量模拟
x = torch.tensor([0.5, 0.5], requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)
v_estimate = torch.zeros(2)
for step in range(500):
    opt.zero_grad()
    loss = f(x)
    loss.backward()
    # 取 grad
    g = x.grad.clone()
    # EMA of g^2 (beta2 = 0.999)
    v_estimate = 0.999 * v_estimate + 0.001 * g**2
    opt.step()

print("v_t (Adam 累积):", v_estimate.tolist())
print("Hessian diag:    ", H.diag().tolist())
# 预期:v_t 接近 Hessian diag(数量级一致)
```

---

## 10. 常见误区

### 误区 1: "正定 = 特征值都 > 0"

**正确**:对**实对称矩阵**,正定 $\iff$ 特征值都 > 0。**对一般矩阵**,特征值可能为复数,正定性需要 $A = A^T$ + 特征值 > 0。**反例**:$A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$,特征值都 = 1,但 $A$ 不对称,不是正定($\mathbf{x}^T A \mathbf{x} = x_1^2 + 2 x_1 x_2 + x_2^2 = (x_1 + x_2)^2 \geq 0$,**实际是半正定**,但 $\det = 1 \neq 0$ 让人误以为正定)。

### 误区 2: "正定矩阵一定可逆"

**正确**:**正定 $\Rightarrow$ 可逆**(因为所有特征值都 > 0,$\det = \prod \lambda_i > 0$,所以 $\det \neq 0$,可逆)。但反之不成立:可逆 $\not\Rightarrow$ 正定(可能不定)。**口诀**:"PD 是可逆的子集"。

### 误区 3: "半正定 + 半正定 = 半正定"

**正确**:**PSD + PSD = PSD**(逐项加,每个方向都 $\geq 0$,和仍然 $\geq 0$)。但**反之** PSD = PSD 不一定 PSD(虽然 PSD - PSD 一般不是 PSD,**可以是负定的**)。

### 误区 4: "正定 - 正定 = 正定"

**正确**:**反例**:$I$ 正定, $2I$ 正定, 但 $I - 2I = -I$ **负定**。正定 - 正定**通常不定或负定**。

### 误区 5: "条件数大 → 模型差"

**正确**:条件数描述**优化难度**,不描述**模型质量**。线性回归即使 $\kappa(X^T X) = 10^6$,闭式解 $\hat{\theta} = (X^T X)^{-1} X^T y$ 仍然是无偏最优。**条件数大 = 训练慢 / 数值解不稳,不是模型差**。

### 误区 6: "Hessian 半正定 = loss 凸"

**正确**:Hessian 半正定是 loss 凸的**充分条件**,但**不是必要条件**。即使 Hessian 在某些点不定,函数仍可能是凸(如分段线性凸函数)。但**对二阶可微函数**,凸 $\iff$ Hessian PSD(对所有点)。

### 误区 7: "深度学习 loss 凸"

**正确**:**绝大多数深度学习 loss 非凸**(中间层非线性引入)。但 L2 正则化让 loss 局部强凸,**整体非凸但局部凸**。GD 在非凸 loss 上仍能 work,原因见 1-高数第 10 章。

### 误区 8: "Adam 永远比 SGD 好"

**正确**:**不一定**。在**好条件**的凸问题上(如线性回归),SGD **收敛率可以更高**(Adam 的 preconditioning 噪声反而干扰)。Adam 在**病态 + 稀疏梯度**上优势明显。**经验**:搜推广默认 Adam,CV/NLP 默认 SGD+Momentum 或 AdamW。

---

## 11. 面试 Q&A

### Q1: 为什么 Adam 在搜推广 Embedding 上比 SGD 稳?从条件数角度。

**A**:搜推广 Embedding 表有 $10^9$ 个 ID,热门 ID 出现频率 $10^6$,冷门 ID 出现频率 $\sim 0$。**Hessian 对角线** $H_{ii}$ 正比于 ID 的累积梯度 $\sum g_i^2$,热 ID 的 $H_{ii}$ 比冷 ID 大 $10^6$ 倍。**Hessian 条件数** $\kappa(H) \geq 10^6$,**严重病态**。

SGD 用同一学习率,热 ID 处 lr 太大震荡、冷 ID 处 lr 太小不更新。**Adam 用二阶动量 $\sqrt{v_t}$ 做逐元素学习率缩放**:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t$$

热 ID 处 $v_t$ 大,有效 lr 小,抑制震荡;冷 ID 处 $v_t$ 小,有效 lr 大,加速更新。**几何上**:Adam ≈ 对角 Hessian 倒数 preconditioning,把条件数从 $10^6$ 拉回 $\sim 1$,**收敛率从 $O(\kappa) = O(10^6)$ 降到 $O(1)$**,等价于上万倍加速。

### Q2: Hessian 半正定 = loss 凸,但深度学习是非凸的,这矛盾吗?

**A**:**不矛盾**。Hessian 半正定是**该点局部凸**的判据。但**深度学习 Hessian 处处变化**:
- 不同参数空间的局部,Hessian 谱不同
- 训练过程中,Hessian 也在变化(参数更新)
- 局部小邻域内 Hessian 可能 PSD,但**整个 loss landscape** 充满鞍点(Choromanska 2015:鞍点数量 $\gg$ 局部极小)

深度学习 loss 是**整体非凸、局部近似凸**。GD 在局部凸区域表现好,但全局不保证。**为什么 GD 仍能 work?** 1-高数第 10 章详述:Choromanska 2015 的结果(局部极小值接近全局最优)、Kawaguchi 2016 的理论、Jacot 2018 的 NTK 框架。**简答**:在过参数化网络中,局部极小和全局极小的 loss 值差距不大,GD 找任何一个都行。

### Q3: Pre-conditioning 的本质是什么?

**A**:**Pre-conditioning = 找一个变换 $P$,把"病态"问题变成"好条件"问题**。

数学上,原始 GD 是 $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f$,Pre-conditioned GD 是 $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta P^{-1} \nabla f$。

**理想选择** $P = H$(Hessian) → Newton 法,一步收敛。但 $H$ 太大、求逆太贵。

**工程选择**:
- 对角 $P = \text{diag}(H)$:K-FAC、Adam、RMSProp(后者用 $\text{diag}(g^2)$ 近似)
- 低秩 $P$:L-BFGS(用历史梯度构造)
- Block $P$:Shampoo(把矩阵 preconditioning 分块)

**效果**:变换后的有效条件数 $\kappa_{\text{new}} = \kappa(P^{-1/2} H P^{-1/2}) \approx 1$,收敛率从 $O(\kappa)$ 降到 $O(1)$。

### Q4: 为什么 BN/LN 能改善训练?

**A**:BN/LN **逐层标准化激活**($x \leftarrow (x - \mu) / \sigma$ 后乘 $\gamma$ 加 $\beta$)。**对 Hessian 的影响**:

1. **下游层输入分布稳定**:每层输入的均值 = 0,方差 = $\gamma^2$,**不依赖上层权重**
2. **Jacobian 条件数小**:激活标准化后,反向传播的 Jacobian 谱被约束在小范围
3. **允许大学习率**:ResNet 加 BN 后能从 $10^{-3}$ 提到 $10^{-1}$(He 2015),因为条件数变小,大 lr 不炸

**L2 正则化 + BN/LN**:常见组合。L2 让 Hessian 加 $\lambda I$,BN/LN 让 Hessian 对角线稳定。**两者协同**:优化更容易 + 泛化更好。

### Q5: 为什么 Newton 法在病态问题上失败?

**A**:**Newton 法用 $H^{-1}$ 抵消 Hessian 的病态**,理论上一步收敛。但实际有三个问题:

1. **$H^{-1}$ 的计算本身受病态影响**:求逆是 $O(n^3)$,且 $\kappa(H^{-1}) = \kappa(H)$,**求逆过程放大了输入误差**
2. **$H$ 不定**(深度网络常见):Newton 方向 $\mathbf{d} = -H^{-1} \nabla f$ **沿正特征方向指向极大**,把优化器推向"上山"
3. **步长 = 1**(完全相信 Hessian):但 $H$ 估计不准时步长太大,**直接飞出可行域**

**修复**:
- **Damped Newton**:$H + \lambda I$,$\lambda$ 取 $\kappa$ 的上界
- **Trust Region**:约束 $\|\mathbf{d}\| \leq \Delta$
- **现代 DL**:用 K-FAC、Shampoo、Sophia 等近似二阶方法,**自动处理不定性 + 控制步长**

---

## 12. 与其他数学工具的联系

| 联系 | 内容 |
| :--- | :--- |
| **二次型 ↔ 凸优化** (1-高数 第 10 章) | 凸函数的二阶刻画就是 $\mathbf{x}^T H \mathbf{x} \geq 0$;二次型是凸优化的"原型" |
| **正定 ↔ Cholesky** (第 3 章) | SPD 矩阵的充要条件 = Cholesky 分解存在;Cholesky 是 SPD 矩阵的"身份证" |
| **条件数 ↔ 最优化** (6-最优化 第 5 章) | 条件数决定 GD 的收敛率 $O(\kappa \log(1/\epsilon))$,强凸优化的核心 |
| **强凸 ↔ 收敛率** (1-高数 第 5 章 Taylor) | 强凸 = Hessian 有下界 $\mu I$,Taylor 展开给出"二次下界",直接推 GD 线性收敛 |
| **病态 ↔ 系统思维** (7-系统思维) | 病态是"放大误差的系统特性",搜推广、推荐、RL 都大量出现,需要系统级应对(preconditioning、归一化、warmup) |
| **二次型 ↔ SVD** (第 3 章) | $A^T A$ 的特征值 = $A$ 奇异值的平方;谱条件数 = $\sigma_{\max}/\sigma_{\min}$ |
| **Hessian ↔ Newton** (1-高数 第 5 章) | Newton 方向 = $-H^{-1} \nabla f$,用 Hessian 二阶信息加速 |
| **PSD ↔ Mercer 定理** (1-高数 第 8 章 RKHS) | Mercer 定理:合法的核函数 $\iff$ 核矩阵 PSD,对所有样本;这是核方法的数学基础 |

---

## 附录 A:本章速查卡

### 核心定义

| 概念 | 定义 | 记号 |
| :--- | :--- | :--- |
| 二次型 | $\mathbf{x}^T A \mathbf{x}$ | $q(\mathbf{x})$ |
| 正定 | $\forall \mathbf{x} \neq 0, \mathbf{x}^T A \mathbf{x} > 0$ | $A \succ 0$ |
| 半正定 | $\forall \mathbf{x}, \mathbf{x}^T A \mathbf{x} \geq 0$ | $A \succeq 0$ |
| 负定 | $\forall \mathbf{x} \neq 0, \mathbf{x}^T A \mathbf{x} < 0$ | $A \prec 0$ |
| 不定 | 既有正又有负 | - |
| 谱条件数 | $\sigma_{\max} / \sigma_{\min}$ | $\kappa_2(A)$ |
| 强凸 | $\mathbf{x}^T H \mathbf{x} \geq \mu \|\mathbf{x}\|^2$ | $\mu$-强凸 |
| $L$-光滑 | $\mathbf{x}^T H \mathbf{x} \leq L \|\mathbf{x}\|^2$ | $L$-光滑 |

### 4 个正定性判据

| 判据 | 条件 | 适用场景 |
| :--- | :--- | :--- |
| 特征值 | $\lambda_i > 0$ | 中等矩阵 ($n \leq 1000$) |
| LPM | $\Delta_k > 0$ for $k = 1..n$ | 小矩阵 ($n \leq 5$) |
| 分解 | $\exists B$ 列满秩, $A = B^T B$ | 理论证明 |
| Cholesky | $A = L L^T$, $L$ 对角元 > 0 | 数值验证 / 大矩阵 |

### 重要定理

| 定理 | 内容 |
| :--- | :--- |
| 谱定理(对称) | $A = Q \Lambda Q^T$, $\Lambda$ 对角,$Q$ 正交 |
| Sylvester 判据 | $A \succ 0 \iff$ 所有 LPM > 0 |
| Cholesky 存在 | $A \succ 0 \iff A = L L^T$ |
| GD 收敛率($\mu$-强凸 + $L$-光滑) | $(1 - \mu/L)^t$ |
| 误差传播 | $\|\delta \mathbf{x}\|/\|\mathbf{x}\| \leq \kappa \cdot \|\delta \mathbf{b}\|/\|\mathbf{b}\|$ |

### PyTorch 速查

```python
# 1. 特征值判据
torch.linalg.eigvalsh(A).min() > 0

# 2. Cholesky
L = torch.linalg.cholesky(A)  # 不定则抛 LinAlgError

# 3. 条件数
sv = torch.linalg.svdvals(A)
kappa = sv.max() / sv.min()

# 4. 求解 Ax = b (避免显式求逆)
x = torch.linalg.solve(A, b)

# 5. 最小二乘 (用 SVD, 不要用 (A.T @ A).inverse())
x = torch.linalg.lstsq(A, b).solution
```

### 常见矩阵的条件数估计

| 矩阵类型 | 条件数 | 状态 |
| :--- | :--- | :--- |
| 单位矩阵 | 1 | 完美 |
| 对角 (a, b), a/b ≈ 1 | $\approx 1$ | 好 |
| Hilbert 矩阵 $H_{ij} = 1/(i+j-1)$ | $O((1+\sqrt{2})^{4n}/\sqrt{n})$ | 灾难性 |
| Vandermonde 矩阵 | $O(\exp(c n))$ | 灾难性 |
| 随机高斯矩阵 | $\approx 10$ | 好 |
| 病态搜推广 Embedding Hessian | $10^4 \sim 10^7$ | 病态 |

---

## 附录 B:与文件夹其他章节的引用图谱

```text
                    第 5 章:二次型·正定性·条件数
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   1-高数 第 3 章          1-高数 第 5 章          1-高数 第 10 章
   Hessian                Taylor 展开              凸分析/强凸
   (二阶导数)            (二次型 = Taylor         (Hessian PSD = 凸)
                          二阶项)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                      6-最优化 第 5 章
                      凸优化·收敛率
                      (条件数决定收敛速度)
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
           Adam 优化器      Newton 法       Trust Region
           (对角 precond)   (H^{-1})        (约束步长)
                │
                ▼
           7-系统思维 案例 4
           搜推广 Embedding 训练
           (病态 Hessian 的工程应对)
```

---

## 附录 C:参考文献

1. Horn, R. A., & Johnson, C. R. (2012). *Matrix Analysis*. Cambridge University Press. (二次型/正定/条件数的"圣经")
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. (强凸/收敛率的经典)
3. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer. (Newton/Trust Region 的工程视角)
4. Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization". ICLR. (Adam 原始论文)
5. Miyato, T., et al. (2018). "Spectral Normalization for Generative Adversarial Networks". ICLR. (谱归一化 GAN)
6. Dauphin, Y. N., et al. (2014). "Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization". NeurIPS. (深度学习鞍点)
7. Choromanska, A., et al. (2015). "The Loss Surfaces of Multilayer Networks". AISTATS. (局部极小接近全局)
8. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification". ICCV. (BN/初始化)
9. Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS. (Transformer + Warmup)
10. Loshchilov, I., & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts". ICLR. (Warmup + Cosine Annealing)
11. Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning". SIAM Review. (优化方法综述)
12. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6/7/8. (二次型/优化基础)

---

> **本章总结**:
>
> 二次型 $\mathbf{x}^T A \mathbf{x}$ 是"任何函数在局部的二阶指纹",正定性决定它是凸(碗)还是不定(马鞍),条件数 $\kappa$ 衡量凸的"扁度"。
>
> 在深度学习中,**Hessian 矩阵的谱性质**决定了:
> - GD 收敛多快(条件数小 → 快)
> - Adam 为什么稳(preconditioning 把 $\kappa$ 拉回 ~1)
> - 搜推广 Embedding 为什么病态(频率差异 $10^6$ 倍)
> - BN/LN 为什么有效(标准化 → 改善条件数)
> - Transformer 为什么需要 Warmup(初期 $\kappa$ 大,需小 lr)
>
> **面试 L7/L8 标准**:能从"二次型 → 正定 → 条件数 → Adam preconditioning"这条线,串联起 Hessian、Newton 法、强凸、收敛率这些工具,讲清楚工程现象的数学根因。
>
> 下一章 (第 6 章) 我们进入**矩阵微积分**,讲 Jacobian、Hessian 的计算和反向传播的数学骨架。

---

## 13. 工程师常见误区与踩坑地图

### 坑 1: "Adam 一定比 SGD 好?"

**症状**: 默认用 Adam 训练所有模型,从来不用 SGD。

**根因**: Adam 的二阶矩缩放适合**病态问题**,但有偏置、不收敛到最优(Reddi 2018);SGD+momentum 泛化通常更好。

**解药**:
- 视觉任务 (CNN):SGD+momentum 仍是最优。
- NLP / 序列模型:Adam / AdamW 是标配。
- 推荐 / 搜推:Adam + 稀疏梯度处理。
- GAN:Adam(注意 β1=0 而非 0.9)。

### 坑 2: "L2 正则 = weight decay?"

**症状**: 加 `weight_decay=1e-4` 当成 L2 正则。

**根因**: 在 Adam 下,L2 正则和 weight decay **不等价**(因为 Adam 自带梯度缩放)。

**解药**:
- 用 Adam 时,**用 AdamW**(解耦的 weight decay,Loshchilov 2019)。
- 用 SGD 时,L2 正则 ≈ weight decay。

### 坑 3: "Loss 出现 NaN,一定是 LR 太大?"

**症状**: Loss NaN,第一反应减小 LR,但其实可能是其他原因。

**根因排查**:
1. 数值溢出(log(0), exp 太大, sqrt 负数)。
2. 梯度爆炸(没用 clip)。
3. fp16 精度不够(用 fp32 + GradScaler)。
4. 数据有 NaN/Inf(检查输入)。
5. 矩阵条件数 $\kappa$ 太大(病态问题)。

**解药**:
- 加 NaN 检测:`if torch.isnan(loss).any(): break`
- 用 gradient clipping:`clip_grad_norm_(max_norm=1.0)`
- 混合精度:`torch.cuda.amp.autocast`
- 数据预处理检查

### 坑 4: "BatchNorm 一定要放在 ReLU 之前/之后?"

**症状**: BN 位置不对导致训练失败。

**根因**: BN 归一化激活前的输入,会改变 ReLU 后输出的分布。

**解药**:
- **Conv → BN → ReLU** (原始论文顺序,经典 CNN)
- **Conv → ReLU → BN** (某些 ResNet 变体)
- **Linear → ReLU → BN** (NLP 不常用,通常用 LayerNorm)
- 经验:**先 BN 后激活**更稳定。

### 坑 5: "Loss 下降到 0 但准确率不变?"

**症状**: 训练 loss 接近 0,但验证准确率随机水平。

**根因**: 模型过拟合 / 标签泄露 / 评估 bug。

**解药**:
- 监控 **val loss** 而不是 train loss。
- 检查数据 pipeline(标签是否错位)。
- 加正则 / Dropout / 数据增强。
- 用 early stopping。

### 坑 6: "AdamW 的 lr 和 SGD 的 lr 不通用"

**症状**: SGD 用 lr=0.1,Adam 也用 lr=0.1,效果差。

**根因**: Adam 自适应学习率,需要的 lr 比 SGD 小 10-100 倍。

**解药**:
- SGD/CNN: lr ≈ 0.1
- Adam/Transformer: lr ≈ 1e-4 ~ 1e-3
- AdamW/LoRA: lr ≈ 1e-4 ~ 5e-4

### 坑 7: "模型 loss 不下降但梯度范数很小"

**症状**: 梯度范数 $\|g\| \approx 1e-8$,loss 不动。

**根因**: 进入了**平坦区域**(Hessian $\lambda_{\min} \approx 0$ = 鞍点 / 平坦极小值)。

**解药**:
- 加噪声(SGD 的隐式正则化)。
- 用 Adam(自适应 lr 可跳出)。
- 用 momentum。
- 用 learning rate warmup + cosine decay。

### 坑 8: "用 Adam 但忘了 bias correction"

**症状**: 训练初期 loss 异常。

**根因**: Adam 在前几步 $m_t, v_t$ 接近 0,导致更新步长异常小。

**解药**:
- PyTorch 的 Adam 自动做 bias correction。
- 自己实现时别忘: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$。

### 坑 9: "fp16 训练时 loss 抖动"

**症状**: 用 mixed precision 训练,loss 抖动大。

**根因**: fp16 精度不够,Hessian 病态时更新步长被量化误差放大。

**解药**:
- 关键参数(Embedding, Loss, Norm)用 fp32。
- 用 GradScaler。
- 或直接用 bf16(更大动态范围)。

### 坑 10: "训练 loss 突然变 NaN,怀疑模型崩塌"

**症状**: 训练 N step 后 loss = NaN。

**根因**: 可能是 Embedding 出现全 0 行(冷启动) + 数值极端。

**解药**:
- 检查 Embedding 初始化(避免全 0)。
- 加 `torch.nn.utils.clip_grad_norm_`。
- 用 `torch.nn.utils.spectral_norm` 约束 Lipschitz。
- 学习率 warmup(避免初期大 lr 破坏 Embedding)。

---

## 14. 完整案例: 在 2D 凸函数上演示 GD vs Adam vs Newton's method

### 14.1 病态二次函数: $\kappa = 100$

```python
import numpy as np
import matplotlib.pyplot as plt

# 病态二次函数: f(x) = 0.5 * (100 x_1^2 + x_2^2)
H = np.diag([100.0, 1.0])
def f(x):
    return 0.5 * x @ H @ x

def grad_f(x):
    return H @ x

# GD 优化
def gd(x0, lr=0.05, n_steps=100):
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        trajectory.append(x.copy())
    return np.array(trajectory)

# Adam 优化
def adam(x0, lr=0.5, n_steps=100):
    x = x0.copy()
    m, v = np.zeros_like(x), np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    trajectory = [x.copy()]
    for t in range(1, n_steps + 1):
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(x.copy())
    return np.array(trajectory)

# Newton's method
def newton(x0, n_steps=20):
    x = x0.copy()
    H_inv = np.linalg.inv(H)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x = x - H_inv @ grad_f(x)
        trajectory.append(x.copy())
    return np.array(trajectory)

# 跑
x0 = np.array([10.0, 1.0])
traj_gd = gd(x0)
traj_adam = adam(x0)
traj_newton = newton(x0)

print(f"GD 最终 loss: {f(traj_gd[-1]):.6f}, 步数: 100")
print(f"Adam 最终 loss: {f(traj_adam[-1]):.6f}, 步数: 100")
print(f"Newton 最终 loss: {f(traj_newton[-1]):.6f}, 步数: 20")
```

### 14.2 可视化

```python
# 画等高线 + 轨迹
x1 = np.linspace(-12, 12, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * (100 * X1**2 + X2**2)

plt.figure(figsize=(12, 4))
for i, (traj, name) in enumerate([(traj_gd, "GD"), (traj_adam, "Adam"), (traj_newton, "Newton")]):
    plt.subplot(1, 3, i+1)
    plt.contour(X1, X2, Z, levels=20)
    plt.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=3)
    plt.title(f"{name}\nfinal f = {f(traj[-1]):.4f}")
    plt.xlabel("x1")
    plt.ylabel("x2")
plt.tight_layout()
plt.savefig("optim_compare.png", dpi=100)
```

### 14.3 直觉对比

```text
f(x) = 50x_1² + 0.5x_2² (κ=100)

         x2
         ↑
      1 ─┤ ╲
         │   ╲
         │     ╲
      0 ─┤───────●────  x1
         │   ╱  ↓ 10
         │ ╱
         │╱

GD: 在 x1 方向震荡(因为 Hessian 大), 在 x2 方向慢
Adam: 自适应缩放, 各方向差不多
Newton: 一步到位(用 Hessian^-1 缩放)
```

---

## 15. 几何直觉强化: 二次型的"几何词典"

### 15.1 正定二次型 $A \succ 0$

$\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$。

```text
A = I (单位阵)      A = diag(3, 1)      A = 旋转 + 缩放

    y                  y                    y
    ↑                  ↑                    ↑
    │ ● 0.5            │ ●0.5              ╱●0.5
    │  ╲               │  ╲               ╱╱
    │ ●1.0            │ ●1.0              ╱●1.0
    │                  │                 ╱
    └──── x            └──── x           └──── x

球面               椭圆              倾斜椭圆
```

### 15.2 不定二次型(鞍点)

$A$ 有正有负特征值。

```text
A = diag(1, -1)

    y
    ↑
    │ ● 马鞍面: 沿 y 方向凹下,
    │           沿 x 方向凸上
    │
    └──── x
```

### 15.3 半正定 $A \succeq 0$

某些方向二次型 = 0。

```text
A = diag(1, 0)

    y
    ↑
    │ 沿 y 方向二次型恒 = 0 (平)
    │ 沿 x 方向 > 0
    │
    └──── x
```

### 15.4 条件数 $\kappa$ 决定"等高线形状"

```text
κ = 1: 圆形 → 各方向步长一致 → GD 容易
κ = 10: 椭圆 → GD 需要小心 lr
κ = 100: 极扁椭圆 → GD 严重震荡
κ → ∞: 病态 → GD 几乎不收敛
```

---

## 16. 动手算一算(带完整答案)

### 16.1 基础题

**题 1**: 判断 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ 是否正定。

<details>
<summary>答案</summary>

**方法 1**: 顺序主子式。
- $a_{11} = 2 > 0$ ✓
- $\det(A) = 4 - 1 = 3 > 0$ ✓

**正定** ✓

**方法 2**: 特征值。
特征值 $\lambda = 3, 1$,都 > 0。

**正定** ✓
</details>

**题 2**: 判断 $A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$ 是否正定。

<details>
<summary>答案</summary>

**顺序主子式**:
- $a_{11} = 1 > 0$ ✓
- $\det(A) = 1 - 4 = -3 < 0$ ✗

**不正定**(特征值 $\lambda = 3, -1$,有负的)。
</details>

**题 3**: 计算 $A = \begin{pmatrix} 4 & 2 \\ 2 & 2 \end{pmatrix}$ 的 Cholesky 分解。

<details>
<summary>答案</summary>

$L_{11} = \sqrt{4} = 2$
$L_{21} = 2/2 = 1$
$L_{22} = \sqrt{2 - 1^2} = 1$

$L = \begin{pmatrix} 2 & 0 \\ 1 & 1 \end{pmatrix}$,验证: $LL^\top = \begin{pmatrix} 4 & 2 \\ 2 & 2 \end{pmatrix}$ ✓
</details>

### 16.2 中级题

**题 4**: 计算 $A = \begin{pmatrix} 100 & 0 \\ 0 & 1 \end{pmatrix}$ 的条件数。

<details>
<summary>答案</summary>

$\sigma_{\max} = 100, \sigma_{\min} = 1$。
$\kappa_2(A) = 100/1 = 100$。

病态。
</details>

**题 5**: 证明 Hessian PSD 当且仅当函数凸。

<details>
<summary>答案</summary>

**充分性**: Hessian PSD → $\mathbf{x}^\top H \mathbf{x} \geq 0$ → Taylor 展开的二阶项 $\geq 0$ → $f$ 凸。

**必要性**: $f$ 凸 → 二阶 Taylor 展开 $\Delta^2 f = \mathbf{x}^\top H \mathbf{x} \geq 0$ → $H$ PSD。
</details>

**题 6**: 用 Python 验证 Adam preconditioning 把条件数拉回 ~1。

<details>
<summary>答案</summary>

```python
import torch

# 模拟病态 Hessian
H = torch.diag(torch.tensor([100.0, 1.0]))

# SGD
x = torch.tensor([10.0, 1.0], requires_grad=True)
opt_sgd = torch.optim.SGD([x], lr=0.01)
for _ in range(100):
    loss = 0.5 * x @ H @ x
    opt_sgd.zero_grad()
    loss.backward()
    opt_sgd.step()
print(f"SGD 最终 loss: {loss.item():.4f}")

# Adam(自适应)
x = torch.tensor([10.0, 1.0], requires_grad=True)
opt_adam = torch.optim.Adam([x], lr=0.5)
for _ in range(100):
    loss = 0.5 * x @ H @ x
    opt_adam.zero_grad()
    loss.backward()
    opt_adam.step()
print(f"Adam 最终 loss: {loss.item():.4f}")
```
</details>

### 16.3 高级题(ML/DL)

**题 7**: 为什么 Adam 在 RNN/LSTM 训练中比 SGD 稳?

<details>
<summary>答案</summary>

RNN/LSTM 的 loss landscape 极度病态(梯度通过 BPTT 长链传播,梯度范数变化 10^6 倍)。

Adam 通过自适应缩放($g / \sqrt{v + \epsilon}$),让不同时间步的梯度更新步长一致,等效于 preconditioning。

SGD 没有自适应机制,在病态方向上会震荡甚至发散。
</details>

**题 8**: 实现一个简化版的 Adam 优化器。

<details>
<summary>答案</summary>

```python
import torch

class SimpleAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']

                # 更新 m, v
                state['m'] = group['beta1'] * state['m'] + (1 - group['beta1']) * grad
                state['v'] = group['beta2'] * state['v'] + (1 - group['beta2']) * grad**2

                # bias correction
                m_hat = state['m'] / (1 - group['beta1']**t)
                v_hat = state['v'] / (1 - group['beta2']**t)

                # 更新参数
                p.data -= group['lr'] * m_hat / (torch.sqrt(v_hat) + group['eps'])
```
</details>

**题 9**: 为什么 Transformer 训练需要 warmup?

<details>
<summary>答案</summary>

训练初期,参数的 Hessian 条件数极大(模型刚从随机初始化,各层输出尺度差异大)。如果用大学习率,模型会"破坏"已经学到的 Embedding。

**Warmup 让 lr 从 0 慢慢增加**,让 Hessian 条件数有时间"自然化"(LayerNorm + 残差 + 数据驱动)。

warmup 步数 ≈ O(d / κ_max)。
</details>

**题 10**: 推荐系统中,Embedding 条件数极大,如何处理?

<details>
<summary>答案</summary>

**问题**: 高频物品 Embedding 范数 100+,低频物品 < 0.01 → κ = 10000+。

**修法**:
1. **频次倒数加权采样**:梯度 = 1/freq,平衡各物品梯度量级。
2. **正则化**: $\|e_i\|^2 \geq \delta$,防低频物品范数崩塌。
3. **共享 Embedding**:低频物品共享高频物品 Embedding(Two-tower + side info)。
4. **Hash Embedding**:多个物品共享一个 Embedding,降低存储 + 平衡梯度。

**示例**:
```python
# 频次倒数加权
freq = torch.tensor([10000, 100, 1])
weight = 1.0 / torch.sqrt(freq)
loss = (weight * per_item_loss).sum()
```
</details>

---

## 17. 跨章关联与延伸阅读

### 17.1 与第 2 章的接口

- 二次型 $\mathbf{x}^\top A \mathbf{x}$ 是"矩阵作为双线性映射"。
- 矩阵条件数 = 矩阵"放大误差的能力"。

### 17.2 与第 3 章的接口

- **Cholesky** $A = LL^\top$ 只对 SPD 矩阵存在。
- **SVD 条件数** $\sigma_{\max}/\sigma_{\min}$ = 谱条件数。
- **Eckart-Young**:截断 SVD 的 Frob 误差 = $\sqrt{\sum_{i>k} \sigma_i^2}$。

### 17.3 与第 4 章的接口

- **谱定理**:对称矩阵 $A = Q\Lambda Q^\top$。
- **正定性 ↔ 所有特征值 > 0**。
- **谱条件数** = $\lambda_{\max}/\lambda_{\min}$。

### 17.4 与第 6 章(矩阵微积分)的接口

- **Hessian** = 二次型的矩阵。
- **Newton 方向** $-H^{-1} \nabla f$ = 二次型最小化方向。
- **强凸性** = $\mathbf{x}^\top H \mathbf{x} \geq \mu \|\mathbf{x}\|^2$。

### 17.5 与第 8 章(高维几何)的接口

- 随机矩阵的条件数 ≈ $O(d)$(高维自然病态)。
- 这就是为什么高维问题需要 **preconditioning**。

### 17.6 与第 9 章(稀疏与低秩)的接口

- LoRA 通过**降维**降低条件数。
- Embedding 压缩同样减少参数空间,改善条件数。

---

## 18. 论文笔记: 用二次型/正定性读懂 5 篇优化论文

### 18.1 Adam (Kingma & Ba 2015)

**核心**: 自适应学习率 = $g / \sqrt{v}$。

**数学**:
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ (一阶矩)
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ (二阶矩)
- $\theta_{t+1} = \theta_t - \alpha \cdot m_t / (\sqrt{v_t} + \epsilon)$

**二次型视角**: $\text{diag}(1/\sqrt{v_t}) \approx H^{-1/2}$,做 preconditioning。

### 18.2 AdaGrad (Duchi 2011)

**核心**: 累积所有历史梯度平方。

$\theta_{t+1} = \theta_t - \alpha \cdot g_t / \sqrt{\sum_{s \le t} g_s^2}$

**问题**: 学习率单调递减,后期学不动。

### 18.3 RMSProp (Hinton 2012)

**核心**: 指数加权累积梯度平方。

$v_t = \beta v_{t-1} + (1-\beta) g_t^2$

**修法**: 加权,不让历史主导。

### 18.4 AdamW (Loshchilov & Hutter 2019)

**核心**: 权重衰减与梯度更新解耦。

$\theta_{t+1} = \theta_t - \alpha \cdot (m_t / \sqrt{v_t} + \lambda \theta_t)$

**对比 Adam + L2**: Adam 的 L2 正则会被自适应缩放"打折",AdamW 不会。

### 18.5 SAM (Foret et al. 2020)

**核心**: Sharpness-Aware Minimization。

$\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$

**数学**: 等价于最小化 $\lambda_{\max}(H)$(Hessian 最大特征值)。

---

## 19. 关键公式速查

| 概念 | 公式 | PyTorch 一行 |
| :--- | :--- | :--- |
| 二次型 | $\mathbf{x}^\top A \mathbf{x}$ | `x @ A @ x` |
| 正定性 | $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x}$ | `(torch.linalg.eigvalsh(A) > 0).all()` |
| 谱条件数 | $\kappa = \sigma_{\max}/\sigma_{\min}$ | `torch.linalg.cond(A)` |
| Cholesky | $A = LL^\top$ | `torch.linalg.cholesky(A)` |
| Hessian | $H = \nabla^2 f$ | autograd double backward |
| 强凸 | $\mathbf{x}^\top H \mathbf{x} \geq \mu \|\mathbf{x}\|^2$ | - |
| $L$-光滑 | $\mathbf{x}^\top H \mathbf{x} \leq L \|\mathbf{x}\|^2$ | - |
| 预条件 | $M^{-1} g$ | $M$ 的对角线 1/sqrt(v) |
| 阻尼 Newton | $(H + \lambda I)^{-1} g$ | - |

---

## 20. 章节总结图

```text
              二次型 q(x) = x^T A x
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     正定 PSD         不定          负定
        │              │              │
        ▼              ▼              ▼
   凸函数 (bowl)   鞍点 (saddle)   凹函数
        │              │              │
        ▼              ▼              ▼
   ┌─────────────────────────────────────┐
   │ DL 应用:                            │
   │ - Hessian PSD → 凸优化             │
   │ - 条件数 κ → 收敛速度              │
   │ - 强凸 → GD 线性收敛               │
   │ - L-光滑 → GD 步长上界             │
   │ - Adam preconditioning             │
   │ - Spectral Norm 约束              │
   │ - LoRA / Embedding 压缩            │
   └─────────────────────────────────────┘
```

**核心**: 二次型 + 正定性 + 条件数 = 优化理论的核心。Hessian 决定 loss landscape 形状,条件数决定收敛速度,Adam 是 preconditioning 的简化版。

---

> **本章版本**: v1.1
> **最后更新**: 2026-06-26
> **本章扩充内容**: 13. 工程师误区(10 坑) / 14. GD vs Adam vs Newton 案例 / 15. 几何 ASCII / 16. 动手算一算(10 题) / 17. 跨章关联 / 18. 5 篇论文解读 / 19. 公式速查 / 20. 总结图

---

## 21. 强凸性详细推导(给"想真正搞懂收敛率"的读者)

### 21.1 强凸性定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是 $\mu$-强凸,如果对所有 $\mathbf{x}, \mathbf{y}$:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

**直观**: 函数图被一个**曲率 $\mu$ 的抛物线**从下方"托住"。

### 21.2 强凸性 ↔ Hessian 条件

$f$ 是 $\mu$-强凸 $\Leftrightarrow$ $\nabla^2 f(\mathbf{x}) \succeq \mu I$ for all $\mathbf{x}$。

即 Hessian 的最小特征值 $\geq \mu$。

### 21.3 $L$-光滑性定义

函数 $f$ 是 $L$-光滑,如果 $\nabla^2 f(\mathbf{x}) \preceq L I$(Hessian 最大特征值 $\leq L$)。

**直观**: 函数不会"过陡",梯度变化率有上界 $L$。

### 21.4 GD 在 $L$-光滑 $\mu$-强凸函数上的收敛率

$$\|\mathbf{x}_k - \mathbf{x}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|^2$$

**收敛率**: $O\left(\left(1 - \frac{1}{\kappa}\right)^k\right)$, $\kappa = L/\mu$ 是条件数。

**条件数小** → 收敛快;**条件数大** → 收敛慢。

### 21.5 例子: 二次函数 $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top A \mathbf{x} - \mathbf{b}^\top \mathbf{x}$

- $\nabla f = A\mathbf{x} - \mathbf{b}$
- $\nabla^2 f = A$
- $L = \lambda_{\max}(A)$, $\mu = \lambda_{\min}(A)$(假设 $A \succ 0$)
- $\kappa = \lambda_{\max}/\lambda_{\min}$(谱条件数)
- 收敛率 $= (1 - 1/\kappa)^k$

**特殊情况**: $\kappa = 1$(球面等高线) → 一次收敛。
$\kappa = 100$(极扁椭圆) → 需要 ≈ 100k 步收敛。

---

## 22. Adam 的 preconditioning 机制详解

### 22.1 Adam 的等效形式

Adam 更新: $\theta_{t+1} = \theta_t - \alpha m_t / \sqrt{v_t}$。

把 $\sqrt{v_t}$ 看作**对角 preconditioner** $D_t = \text{diag}(\sqrt{v_t} + \epsilon)$:

更新 ≈ $-\alpha D_t^{-1} m_t$

**对比牛顿法**: $-H^{-1} \nabla f$(理想 preconditioner)。

**Adam ≈ 近似 Newton**($D_t^{-1}$ ≈ $\text{diag}(H)^{-1/2}$)

### 22.2 为什么 $v_t$ ≈ Hessian 的对角线?

对 $i$:

$v_{t,i} \approx \mathbb{E}[g_{t,i}^2]$

如果 $g_i$ 是均值为 0 的随机变量,且函数二阶可微:

$\text{Var}(g_i) = \mathbb{E}[(\partial_i L)^2] \approx (\partial_i^2 L)^2 \text{Var}(\mathbf{x})$ ≈ Hessian 对角元素

所以 $\sqrt{v_i} \approx |\partial_i^2 L|$,$1/\sqrt{v_i} \approx 1/|\partial_i^2 L|$ ≈ Hessian 的对角逆。

### 22.3 Adam vs L-BFGS

| 优化器 | preconditioner 类型 | 复杂度 | 适用 |
| :--- | :--- | :--- | :--- |
| SGD | $I$(无) | $O(d)$ | CNN / 大 batch |
| Adam | $\text{diag}(\sqrt{v})$(对角) | $O(d)$ | Transformer / 稀疏 |
| L-BFGS | 有限记忆逆 Hessian | $O(d \cdot \text{mem})$ | 小模型 / 凸 |
| Newton | $H^{-1}$(精确) | $O(d^3)$ | 小模型,严格凸 |

### 22.4 Adam 的失败案例

虽然 Adam 一般稳,但有些场景 SGD 更好:

1. **CNN 视觉**: SGD+momentum 收敛到更平坦的极小值,泛化更好。
2. **凸优化**: Adam 的偏置会导致不收敛到最优(Reddi 2018 修正)。
3. **强化学习**: Adam 的非平稳性可能让策略震荡。

**经验**:
- 视觉: SGD+momentum
- 序列 / Transformer: AdamW
- 推荐 / 搜推: Adam(处理稀疏梯度)
- RL: Adam(注意 β1=0)

---

## 23. 条件数的"工业启示录"

### 23.1 病态的 6 个根源

| 根源 | 描述 | 修法 |
| :--- | :--- | :--- |
| 特征尺度差异 | 数据 x1 ∈ [0, 1], x2 ∈ [0, 1e6] | 标准化 / 归一化 |
| 参数尺度差异 | LR 大, Embedding 小 | 共享 lr / 分组 lr |
| 网络深度 | 深层网络梯度消失/爆炸 | 残差 / Norm / 初始化 |
| 长序列 | RNN/LSTM BPTT 链长 | 截断 BPTT / 残差 |
| 数据稀疏 | 推荐 Embedding 频次悬殊 | 频次倒数加权 |
| 高维 | 高维空间自然病态 | 降维 / 正则 |

### 23.2 修法"工具箱"

1. **数据预处理**: 标准化 (z-score) → 改善 $\kappa_X$。
2. **参数初始化**: Xavier / Kaiming → 控制激活方差。
3. **BatchNorm / LayerNorm**: 强制中间特征同尺度。
4. **残差连接**: 让梯度直接流回,避免消失。
5. **自适应优化**: Adam / RMSProp → 自动 preconditioning。
6. **学习率调度**: warmup + decay → 稳定训练。
7. **正则化**: $\|W\|^2$ → 限制参数范数,改善条件数。
8. **梯度裁剪**: 防止爆炸,稳定数值。

### 23.3 实战判断: 我的模型病态吗?

**症状 → 诊断**:

| 症状 | 病态指标 |
| :--- | :--- |
| 训练 loss 震荡 | $\kappa(H)$ 大 |
| 不同方向更新步长差异大 | Hessian 各方向 λ 差异大 |
| 大 lr 崩溃,小 lr 不动 | 病态方向过拟合,正常方向欠拟合 |
| Adam 比 SGD 显著更好 | Hessian 对角异质(Adam preconditioning 有用) |
| 训练到一半 loss NaN | 数值发散(λ_max 太大) |

**诊断工具**:
- 监控梯度范数变化(梯度爆炸信号)。
- 用 `torch.autograd.functional.hessian` 算小模型的 Hessian。
- 用 Hessian-vector product (HVPs) 估计 λ_max(用 Lanczos / Power iteration)。

---

## 24. 给"读论文遇到 condition number / curvature"的指南

### 24.1 看到 "condition number" 时,问自己 4 个问题

1. **是什么矩阵的条件数?**
   - Hessian? → loss landscape 的扁度
   - 数据协方差? → 特征尺度差异
   - 线性方程组系数? → 数值稳定性

2. **用什么范数定义?**
   - 谱条件数 $\kappa_2 = \sigma_{\max}/\sigma_{\min}$(最常用)
   - Frobenius 条件数(少见)
   - 行/列范数条件数(简化估计)

3. **是"理论上"还是"数值上"病态?**
   - 理论病态: 数学上必须条件数大(如高维)
   - 数值病态: 由浮点精度引起(可加正则修)

4. **怎么修?**
   - 数据标准化
   - Adam / 自适应优化
   - 正则化
   - 降维 / 稀疏化

### 24.2 看到 "Hessian" 时,识别论文的目的

| 论文目的 | Hessian 用法 |
| :--- | :--- |
| 解释 loss landscape | 算 Hessian 特征值谱 |
| 加速收敛 | 用 Hessian 信息(Newton, BFGS, Adam) |
| 泛化分析 | Hessian 的 trace / λ_max 与过拟合相关 |
| 鲁棒性 | SAM 最小化 λ_max |
| 二阶优化 | 显式算 Hessian |

---

## 25. 二次型与"机器学习 5 大核心问题"

### 25.1 监督学习

**目标**: $\min_\theta L(\theta) = \frac{1}{n} \sum_i \ell(f_\theta(\mathbf{x}_i), y_i)$

**二次型近似**: Taylor 展开 $L(\theta + \delta) \approx L(\theta) + \nabla L^\top \delta + \frac{1}{2} \delta^\top H \delta$

**优化**:
- $H$ 病态 → Adam preconditioning
- $H$ 接近单位阵 → SGD 几乎 Newton
- $H$ 严重病态 → 减 lr 或加正则

### 25.2 无监督学习(PCA)

**目标**: 找主成分方向 = 协方差矩阵最大特征向量。

**矩阵形式**: $\Sigma = Q\Lambda Q^\top$。

### 25.3 推荐系统(MF)

**目标**: $\min_{P, Q} \|R - P^\top Q\|^2 + \lambda(\|P\|^2 + \|Q\|^2)$

**Hessian 视角**: 关于 $P$ 的 Hessian ≈ $Q Q^\top$(半正定),条件数 ≈ 用户 Embedding 异质程度。

### 25.4 强化学习

**策略梯度**: $\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) Q(s, a)]$

**Hessian 视角**: 策略的"锐度"决定 RL 训练的稳定性。**TRPO / PPO** 限制策略更新步长 ≈ 限制 KL 散度 ≈ 限制 Hessian 的作用范围。

### 25.5 自监督学习(对比学习)

**InfoNCE loss**: $-\log \frac{e^{s(z, z^+)/\tau}}{\sum_z e^{s(z, z^-)/\tau}}$

**梯度景观**: 对比学习的梯度景观**几何不平滑**,$H$ 条件数大,**特别需要 Adam + warmup**。

---

## 26. 关键概念的"反直觉"事实清单

### 26.1 "凸函数也能过拟合"

**事实**: 强凸函数 + 有限样本 → 过拟合(虽然有唯一最优解)。

**原因**: 强凸 + 大模型 → 训练误差小,但泛化误差大。

**修法**: 正则化(等价于把 $H$ 调整为 $H + \lambda I$,改善条件数)。

### 26.2 "Loss landscape 不是唯一的"

**事实**: 神经网络 loss landscape 有**海量**局部极小点和鞍点,不是单一 bowl。

**直观**: 高维 + 非凸 → 必然存在大量"等价"极小值。

**意义**: 不必追求"找到全局最优",任何一个平坦极小点泛化都不错。

### 26.3 "Adam 的偏置是问题"

**事实**: Adam 的更新 $m / \sqrt{v}$ 在统计上**有偏**,不收敛到凸函数的全局最优(Reddi 2018)。

**修法**: AMSGrad(用 max 而不是 EMA),或用 AdamW + 适当 lr。

### 26.4 "BN 让 Loss landscape 更平滑"

**事实**: BN 后,Hessian 的 $\lambda_{\max}$ 通常减小 10-100 倍 → $\kappa$ 改善 → 收敛加快。

**直观**: BN 把每层输入归一化,消除"输入分布漂移"导致的病态。

### 26.5 "小 batch 反而泛化好"

**事实**: 小 batch SGD 比大 batch SGD 泛化更好(S Keskar 2016)。

**数学**: 小 batch 噪声相当于"平坦度正则化",让模型找到更宽的极小点。

---

## 27. 关键论文的"二次型视角"

### 27.1 On the importance of initialization (Mishkin 2016)

**核心**: 矩阵初始化影响训练动力学。

**二次型视角**: 初始化 $W$ 决定前向传播的 Jacobian $J = \prod_l W_l$,$J$ 的条件数决定信号传播。

### 27.2 Identity Mappings in Deep Residual Networks (He 2016)

**核心**: 残差连接让网络更容易训练。

**二次型视角**: 残差 = $f + g$ 而不是 $f \circ g$,Jacobian 接近 $I + \frac{\partial g}{\partial x}$,条件数 ≈ 1(因为 $I$ 占主导)。

### 27.3 Layer Normalization (Ba 2016)

**核心**: LayerNorm 改善 Transformer 训练。

**二次型视角**: LayerNorm 把每层输入归一化,让 Hessian 各方向 λ 同量级(改善条件数)。

### 27.4 Sharpness-Aware Minimization (Foret 2020)

**核心**: 找平坦极小点 = 改善泛化。

**二次型视角**: 平坦极小点 = $\lambda_{\max}(H)$ 小 = 损失对参数扰动不敏感。

---

## 28. 章节自检 25 题

### 基础题

1. 写出二次型 $q(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ 的 3 种理解方式。
2. 正定矩阵 $\Leftrightarrow$ 所有特征值 > 0,正确吗?
3. 半正定与正定的区别是什么?
4. 计算 $A = \begin{pmatrix} 4 & 2 \\ 2 & 2 \end{pmatrix}$ 的 Cholesky。
5. 条件数 $\kappa_2(A)$ 的定义?

### 中级题

6. 解释 Hessian PSD 与凸函数的关系。
7. 强凸函数的 GD 收敛率推导?
8. 为什么 $\kappa$ 大会让 GD 震荡?
9. Adam 是如何做 preconditioning 的?
10. 病态矩阵的 3 个常见来源?

### 高级题

11. 推导 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x}$ 的 Newton 方向。
12. 解释为什么 Transformer 训练需要 warmup。
13. 推荐系统中,如何处理 Embedding 的高条件数?
14. L-BFGS 与 Adam 在 preconditioning 上的区别?
15. SAM 是如何最小化 $\lambda_{\max}(H)$ 的?

### 编程题

16. 用 NumPy 实现一个 Adam 优化器。
17. 验证 $\mathbf{x}^\top A \mathbf{x} > 0$ 当且仅当 $\lambda_{\min}(A) > 0$。
18. 用 PyTorch 计算一个 MLP 的 Hessian 矩阵(对小型模型)。
19. 比较 Adam 和 SGD 在病态二次函数上的收敛速度。
20. 实现一个 toy Cholesky 分解算法。

### 论文题

21. 读 Adam 论文,识别 preconditioning 形式。
22. 读 SAM 论文,识别 $\lambda_{\max}(H)$ 最小化的数学。
23. 读 LayerNorm 论文,识别为什么对 Transformer 重要。
24. 读 ResNet 论文,识别残差如何改善条件数。
25. 解释 L-BFGS 用"limited memory"近似 Hessian 的动机。

<details>
<summary>对应小节</summary>

1-5: §1-§3
6-10: §4-§5
11-15: §21-§23
16-20: §22, §25
21-25: §18, §27
</details>

---

## 29. 一句话核心 takeaway

**二次型 $\mathbf{x}^\top A \mathbf{x}$ 是函数的"二阶指纹",正定性决定凸性,条件数 $\kappa$ 决定优化难度。深度学习中 Hessian 的谱性质解释了 Adam 为什么稳、Transformer 为什么需要 warmup、推荐 Embedding 为什么病态、BN/LN 为什么有效——一切优化问题最终都回到"二次型 + 正定性 + 条件数"。**

---

## 30. 数值稳定性与工程小贴士

### 30.1 二次型计算的稳定性

```python
# 不稳定: x @ A @ x 可能数值溢出
x = torch.randn(1000) * 100
A = torch.eye(1000) * 1e6
val = x @ A @ x  # 很大

# 更稳定: 用 torch.linalg.matmul 链式,分步算
Ax = A @ x
val = (x * Ax).sum()

# 或用 eigh 分解 + 计算
eigvals, Q = torch.linalg.eigh(A)
# x^T A x = (Q^T x)^T Λ (Q^T x) = Σ λ_i (Q^T x)_i^2
y = Q.T @ x
val = (y**2 * eigvals).sum()
```

### 30.2 求解 SPD 线性方程组

**首选**: Cholesky (2 倍 LU 快,数值更稳)

```python
A = torch.randn(100, 100)
A = A @ A.T + torch.eye(100)  # SPD
b = torch.randn(100)

# Cholesky
L = torch.linalg.cholesky(A)
x = torch.linalg.solve_triangular(L.T, torch.linalg.solve_triangular(L, b, upper=False), upper=True)

# 或直接用 solve + cholesky_factor
x = torch.cholesky_solve(b, L)
```

### 30.3 Hessian 计算

```python
import torch.autograd.functional as F

def loss_fn(params):
    # 参数平铺/还原
    ...

# PyTorch 计算 Hessian(只对小模型可行)
params = torch.randn(10, requires_grad=True)
H = F.hessian(loss_fn, params)

# 大模型: 用 Hessian-vector product (HVP)
def hvp(loss_fn, params, v):
    g = torch.autograd.grad(loss_fn(params), params, create_graph=True)[0]
    gv = (g * v).sum()
    Hv = torch.autograd.grad(gv, params)[0]
    return Hv

# 估计 λ_max: 用幂迭代
v = torch.randn_like(params)
for _ in range(50):
    Hv = hvp(loss_fn, params, v)
    v = Hv / Hv.norm()
lambda_max = (v * hvp(loss_fn, params, v)).sum()
```

### 30.4 推荐 Embedding 的正定性处理

```python
# Item embedding table
item_emb = nn.Embedding(1000000, 64)
nn.init.normal_(item_emb.weight, std=0.01)

# 加 L2 正则(防止范数过大)
item_emb.weight = nn.Parameter(
    torch.clamp(item_emb.weight, min=-10, max=10)  # 数值裁剪
)

# 频次倒数加权 loss(平衡各物品梯度)
freq = torch.tensor(item_counts, dtype=torch.float)
inv_freq = 1.0 / torch.sqrt(freq + 1)
loss = (per_item_loss * inv_freq).mean()
```

### 30.5 数值梯度检查

```python
# 验证 autograd 梯度 vs 数值梯度
def numerical_grad(f, x, eps=1e-6):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_flat = x.view(-1)
        x_flat[i] += eps
        f_plus = f(x).item()
        x_flat[i] -= 2 * eps
        f_minus = f(x).item()
        x_flat[i] += eps
        grad.view(-1)[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def analytical_grad(f, x):
    x = x.clone().requires_grad_(True)
    loss = f(x)
    loss.backward()
    return x.grad

# 比较
x = torch.randn(10)
diff = (numerical_grad(f, x) - analytical_grad(f, x)).norm()
print(f"梯度误差: {diff:.6f}")  # 应该 < 1e-5
```

---

## 31. 章节核心 takeaway 的"三种表达"

### 31.1 一句话版

二次型、正定性、条件数 = 优化的"三件套",深度学习的每一种优化现象都能用它解释。

### 31.2 一段话版

深度学习的优化,本质上是在一个由 Hessian 矩阵决定的二次型 landscape 上找最小点。Hessian 的正定性决定 landscape 是凸(好优化)还是不定(难优化);Hessian 的条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 决定优化的快慢,$\kappa$ 大则 GD 慢、Adam 帮上忙。实践中,我们通过 BN/LN、残差、warmup、Embedding 正则等手段,把 $\kappa$ 拉回可优化范围。

### 31.3 一张图版

```text
Loss landscape 形状 = Hessian 谱
            │
            ├── 全正 (PSD) → 凸 → GD 收敛
            ├── 有正有负 → 不定 → 鞍点问题
            └── 强凸 (μI ≤ H ≤ LI) → κ = L/μ 决定收敛
                                          │
                                          ├── κ 小 → SGD 够用
                                          └── κ 大 → Adam 必备
                                                       │
                                                       └── 更病态 → L-BFGS / Newton
```

---

## 32. 后续章节衔接

| 已学 | 下章衔接 |
| :--- | :--- |
| 二次型 + 正定性 | → Hessian = 二次型的矩阵 (第 6 章) |
| 条件数 | → 数值稳定性 (第 6 章矩阵求导) |
| 强凸性 + GD | → Newton / L-BFGS (第 6 章 + 1-高数 第 10 章) |
| Adam preconditioning | → 矩阵求导的链式法则 (第 6 章) |
| Hessian 谱 | → 谱方法 (第 4 章 + 第 8 章) |

---

## 33. 附录: 7 道补充练习(精选 ML/DL 应用)

### 练习 A: 用 NumPy 实现牛顿法,验证二次函数一次收敛

<details>
<summary>答案</summary>

```python
import numpy as np

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

# 牛顿法
def newton(A, b, x0, n_steps=5):
    x = x0.copy()
    for _ in range(n_steps):
        grad = A @ x - b
        x = x - np.linalg.solve(A, grad)
    return x

x0 = np.array([10.0, 10.0])
x_final = newton(A, b, x0)
print("Newton 解:", x_final)
print("精确解:", np.linalg.solve(A, b))
# 应该一致(数值精度内)
```
</details>

### 练习 B: 验证 Adam 在病态二次函数上的优势

<details>
<summary>答案</summary>

```python
import torch

# 病态: κ = 100
H = torch.diag(torch.tensor([100.0, 1.0]))

# SGD
x = torch.tensor([10.0, 1.0], requires_grad=True)
opt = torch.optim.SGD([x], lr=0.01)
for _ in range(50):
    loss = 0.5 * x @ H @ x
    opt.zero_grad(); loss.backward(); opt.step()
print(f"SGD loss: {loss.item():.4f}")

# Adam
x = torch.tensor([10.0, 1.0], requires_grad=True)
opt = torch.optim.Adam([x], lr=0.5)
for _ in range(50):
    loss = 0.5 * x @ H @ x
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Adam loss: {loss.item():.6f}")  # 更小
```
</details>

### 练习 C: 实现 L2 正则 vs Weight Decay(对比)

<details>
<summary>答案</summary>

```python
import torch

# L2 正则
loss_with_l2 = base_loss + l2_lambda * (W**2).sum()

# Adam 下,loss_with_l2 的梯度 = base_grad + 2 l2_lambda W
# 但 Adam 自带 m/sqrt(v) 缩放 → L2 梯度被"打折"
# 效果: 模型权重范数 < l2_lambda 应有的

# Weight decay (AdamW)
# 直接在更新中减 λW,不被 m/sqrt(v) 缩放
opt = torch.optim.AdamW([W], lr=lr, weight_decay=lambda_)
opt.step()  # 等价于: W -= lr * (m/sqrt(v) + lambda_ * W)
```
</details>

### 练习 D: 在 toy 推荐系统上对比 MF 的 L2 与正交正则

<details>
<summary>答案</summary>

```python
# L2 正则
loss = ((R - P @ Q.T)**2).mean() + lambda_l2 * ((P**2).sum() + (Q**2).sum())

# 正交正则(防止 P 的列共线)
loss = ((R - P @ Q.T)**2).mean() + lambda_orth * ((P @ P.T - torch.eye(P.shape[0]))**2).sum()
```
</details>

### 练习 E: 用条件数诊断矩阵病态程度

<details>
<summary>答案</summary>

```python
import torch

# 检查矩阵病态
A = torch.randn(100, 100)
print(f"Frob 条件数: {torch.linalg.cond(A, p='fro'):.4f}")
print(f"谱条件数: {torch.linalg.cond(A, p=2):.4f}")

# 用 SVD 直接看
U, S, V = torch.linalg.svd(A, full_matrices=False)
print(f"奇异值范围: {S.min().item():.6e} 到 {S.max().item():.6e}")
print(f"条件数 = S.max()/S.min() = {(S.max()/S.min()).item():.4f}")
```
</details>

### 练习 F: 实现 LayerNorm 并观察改善条件数

<details>
<summary>答案</summary>

```python
import torch

# 模拟病态激活
x = torch.randn(100, 1024) * torch.tensor([100.0, 0.01])  # 特征尺度差异

# 不做 LN: Hessian 条件数大
h_unnormalized = x.cov()  # 协方差
cond_unnormalized = torch.linalg.cond(h_unnormalized)

# 做 LN
x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
h_normalized = x_norm.cov()
cond_normalized = torch.linalg.cond(h_normalized)

print(f"无 LN 协方差条件数: {cond_unnormalized:.4f}")
print(f"有 LN 协方差条件数: {cond_normalized:.4f}")  # 接近 1
```
</details>

### 练习 G: 用 Newton 法优化带二次正则的 LR

<details>
<summary>答案</summary>

```python
import torch

# 模拟 LR: f(w) = 0.5 ||Xw - y||^2 + 0.5 λ ||w||^2
X = torch.randn(100, 10)
y = torch.randn(100)
w = torch.zeros(10, requires_grad=True)
lambda_reg = 0.1

# Hessian: X^T X + λ I
H = X.T @ X + lambda_reg * torch.eye(10)
H_inv = torch.linalg.inv(H)

# Newton 一步
loss = 0.5 * ((X @ w - y)**2).sum() + 0.5 * lambda_reg * (w**2).sum()
loss.backward()
with torch.no_grad():
    w.copy_(w - H_inv @ w.grad)

print("Newton 解:", w.numpy())
print("精确解:", torch.linalg.solve(H, X.T @ y).numpy())
```
</details>

---

> **本章版本**: v1.2
> **最后更新**: 2026-06-26
> **本章扩充内容**: 21-29 节: 强凸性详细推导 / Adam preconditioning 详解 / 工业病态诊断 / 论文指南 / 5 大 ML 问题应用 / 反直觉事实 / 自检 25 题 / 总结



## 深度扩展 · 多视角系统展开(续)

### 110. 第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳 - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 拆解为以下几个层次:

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

信息论为"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳 - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 拆解为以下几个层次:

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

信息论为"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳 - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 拆解为以下几个层次:

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

信息论为"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳 - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 拆解为以下几个层次:

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

信息论为"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳 - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 拆解为以下几个层次:

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

信息论为"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 5 章:二次型·正定性·条件数——为什么 Adam 比 SGD 稳" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

