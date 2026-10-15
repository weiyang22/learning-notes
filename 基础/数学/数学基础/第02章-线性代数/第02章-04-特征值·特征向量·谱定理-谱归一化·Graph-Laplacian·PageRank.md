---
title: "第02章-04-特征值·特征向量·谱定理-谱归一化·Graph-Laplacian·PageRank"
tags:
  - 数学
  - 特征值
  - 特征向量
  - 谱定理
  - 谱归一化
  - Graph
description: "本章要点：placian·PageRank"
chapter_id: 数学-4-02
last_updated: 2026-07-14
---

# 第02章-04-特征值·特征向量·谱定理-谱归一化·Graph-Laplacian·PageRank

> placian·PageRank

> **写给 6-7 年没碰数学的工程师的话**:
>
> 第 3 章你学了 SVD/QR/Cholesky/LU 这四种"把矩阵拆成简单矩阵"的工具。SVD 是其中最强的"万能拆解",但它有个限制:对一般矩阵 $A$,SVD 给出的是 $A = U \Sigma V^T$,其中 $U, V$ 是两套不同的正交矩阵,中间 $\Sigma$ 是奇异值。
>
> 这一章问一个更尖锐的问题:**如果 $A$ 本身是对称的**($A = A^T$),能不能"拆得更对称一点"?能不能让 $U = V$,把"两套正交基"压成"一套"?
>
> 答案就是**谱定理 (Spectral Theorem)**:任何实对称矩阵 $A$ 都能写成
> $$A = Q \Lambda Q^T$$
> 其中 $Q$ 是一套正交矩阵(列向量是特征向量),$\Lambda$ 是对角矩阵(对角元是实特征值)。这是线性代数最漂亮的结果之一。
>
> 这一章的承诺:
> 1. 理解**特征值/特征向量**的几何意义(矩阵变换的"不动方向")
> 2. 掌握**谱定理**——对称矩阵的万能分解
> 3. 看懂**谱归一化**为什么让 GAN 训练稳定(从特征值角度)
> 4. 理解**PageRank**——Google 1998 年的核心算法,本质是求最大特征向量
> 5. 理解**Graph Laplacian**——谱聚类、谱图论的基础
> 6. 会用**幂迭代 (Power Iteration)** 数值求主特征向量
>
> **与前后章节的关系**:
> - 第 1 章内积/范数 → 谱范数 = 诱导 2-范数
> - 第 2 章线性变换 → 特征向量 = 变换的"主轴"
> - 第 3 章 SVD → $A^TA$ 的特征值 = $A$ 奇异值的平方
> - 第 5 章二次型/正定 → 实对称矩阵的谱定理 + 特征值全实
> - 第 6 章矩阵微积分 → Hessian 特征值 = 损失 landscape 各方向曲率
> - 1-高数 第 3 章 Hessian → 本章的特征值分析 Hessian
> - 4-离散数学 第 4 章图论 → Graph Laplacian 是本章第 8 节
> - 4-离散数学 第 6 章 GNN → 谱卷积 = 用 Laplacian 特征向量做傅里叶变换
> - 6-最优化 第 5 章凸性 → Hessian PSD = 凸函数的代数判据

---

## 1. 直觉引入:为什么 Spectral Norm 让 GAN 训练稳定? (300 行)

### 1.1 工程师的痛点:为什么 Spectral Norm 让 GAN 训练稳定? 它本质是什么?

2017 年之前,GAN 训练以"玄学"著称。判别器 $D$ 训得太好,生成器 $G$ 就训崩;判别器训得太差,生成器又学不到东西。原始 GAN 的 loss 在 ImageNet 上几乎不收敛。

**Miyato et al. (2018) 提出 Spectral Normalization GAN (SN-GAN)**:在判别器的每一层权重 $W$ 上,**除以它的最大奇异值** $\sigma_{\max}(W)$,强制 $\|W\|_2 \le 1$。然后整个判别器 $D$ 的 Lipschitz 常数 ≤ 1,梯度有界,Wasserstein 距离有意义,训练稳定。

这个想法 2018 年提出后,在 CIFAR-10/ImageNet 上稳定跑通,成了 GAN 训练的标配。

**核心问题**:**为什么把最大奇异值控制住,网络就稳定?** "最大奇异值"在数学上等价于"对称矩阵的最大特征值的平方根",这背后是**矩阵的最大拉伸因子**。

直觉:**矩阵 $A$ 把向量 $\mathbf{x}$ 变成 $A\mathbf{x}$ 时,会把某些方向"拉长"、某些方向"压扁"**。拉得最长的那个倍数 = 最大奇异值 = $\sigma_{\max}(A)$。如果 $\sigma_{\max}(A)$ 很大,微小的输入扰动会变成巨大的输出扰动——判别器就会对生成器的微小改动过度反应,梯度爆炸。

谱归一化 = **强制每层 $W$ 的最大拉伸因子 ≤ 1** = **整网络 Lipschitz ≤ 1** = **判别器对生成器的改动温和** = **训练稳定**。

**这就是这一章要讲的事**:特征值/特征向量告诉我们"矩阵沿哪些方向拉伸了多少",这正好是深度学习里无数稳定性问题的根源。

### 1.2 直觉:一个向量经过矩阵变换后,如果方向不变 = 特征向量

考虑一个 $2 \times 2$ 矩阵 $A$。它把整个平面上的向量都"挪个位置"。**但总有些方向是特殊的:变换前后方向不变,只是长度变了**。

```text
        变换前                变换后
        ─────                ─────
        ↑ y                   ↑ y
        |   ↑ v                |
        |   |  →  A·v = λv    |   v 被"拉长"了 λ 倍
        |   |                  |     ↑ 仍指向原方向
        +───┼───→ x            +───┼───→ x
            |                      |
            v 方向不变              v 现在更长(λ > 1)
```

**这种"方向不变、只被缩放"的向量,就是特征向量 $\mathbf{v}$**。缩放倍数 = 特征值 $\lambda$。

数学上:

$$A \mathbf{v} = \lambda \mathbf{v}$$

**注意**:$A \mathbf{v}$ 一般情况下会改变向量的方向,只有当 $\mathbf{v}$ 恰好是 $A$ 的特征向量时,方向才不变。

#### 三类特殊变换的特征方向

##### (1) 缩放矩阵 $A = \begin{pmatrix} 3 & 0 \\ 0 & 5 \end{pmatrix}$

```text
        y
        ↑
        |  e_2 (0,1)  ──→  沿 e_2 拉长 5 倍
   5 x  |     ↑
        |     |
        +─────┼────→ x
        |  e_1 (1,0)  ──→  沿 e_1 拉长 3 倍
   3 x  |     →

   特征向量: e_1, e_2
   特征值:   3, 5
```

特征向量 = 标准基 $\mathbf{e}_1, \mathbf{e}_2$,特征值 = 对角元。直观:对角矩阵沿主轴缩放,主轴方向就是特征方向。

##### (2) 旋转矩阵 $A = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$

```text
        y
        ↑     旋转后
        |  ↑  /
        | /  /   ← 任意非零向量都"转"了
        +/──+────→ x
       /|
      / ↓   没有方向保持不变!
```

旋转矩阵**没有实特征向量**(除了零向量)。它把每个方向都转了 $\theta$ 角,没有"不动"的方向。**但它有复特征值** $e^{\pm i\theta}$(在复数域)。

这是工程师最常犯的错之一:**以为每个矩阵都有实特征向量**。错了。旋转、剪切等"非对称"操作的特征值往往是复数。

##### (3) 反射矩阵 $A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

```text
        y
        ↑  反射后
        |  ↓
        | /|\
        |/ | \
        +--+--\──→ x
        |\ |
        | \|  沿 x 方向不变 (λ=1)
        |  \
        |   \ 沿 y 方向反向(λ=-1)
```

特征向量 = $\mathbf{e}_1, \mathbf{e}_2$。特征值 = $1, -1$。反射 = "沿某些方向反向"。

### 1.3 应用:PageRank (Google 搜索的核心算法)

**场景**:1996 年,两个 Stanford 博士生(Larry Page 和 Sergey Brin)想做一个搜索引擎,排序 3 亿个网页。问题是:**怎么判断哪个网页"重要"**?

他们的核心洞察(1998 年 PageRank 论文):**一个网页的重要性 = 所有链向它的网页的重要性之和,按这些网页的链出数加权**。

用矩阵表达:$\mathbf{r} = M \mathbf{r}$,其中 $M$ 是"网页跳转概率矩阵"(随机游走模型)。**这个方程的解 $\mathbf{r}$ 恰好是 $M$ 的主特征向量(对应最大特征值 1)**。

直觉:你模拟一个"随机上网的人",他从某网页跳到链向它的某个网页的概率正比于这个网页的重要性。稳态分布 = 主特征向量 = PageRank 分数。

**这就是这一章第 7 节要展开的 PageRank**。一个 1998 年诞生的算法,本质是**特征值问题**。

### 1.4 案例:Spectral Normalization 让 GAN 训练稳定

**问题**:原始 GAN 训练极不稳定,经常模式崩溃(mode collapse)。

**根因**:判别器 $D$ 的 Lipschitz 常数 $L$ 可能非常大,导致 $D$ 对生成器的微小改动过度反应,梯度爆炸。

**数学**:判别器 $D = f_L \circ f_{L-1} \circ \cdots \circ f_1$,每层 $f_\ell$ 是 Linear + (激活)。线性部分 $W_\ell$ 的 Lipschitz 常数 = $\sigma_{\max}(W_\ell)$(最大奇异值)。**整网络 $D$ 的 Lipschitz 常数 = 各层 $\sigma_{\max}$ 的乘积**:

$$\|D\|_{Lip} = \prod_{\ell=1}^{L} \sigma_{\max}(W_\ell)$$

**解法**(Miyato 2018):每层权重 $W$ 替换为 $W / \sigma_{\max}(W)$,强制 $\sigma_{\max} \le 1$。这样每层都是 1-Lipschitz,整网络 Lipschitz ≤ 1。

**实现**:不显式算 SVD(太贵),用**幂迭代 (Power Iteration)** 估计 $\sigma_{\max}$:

```python
# 伪代码:谱归一化的幂迭代
u = torch.randn(d_out)  # 随机初始化
for _ in range(num_iter):  # 通常 1 次就够
    v = F.normalize(W.T @ u, dim=0)
    u = F.normalize(W @ v, dim=0)
sigma = u @ W @ v  # Rayleigh 商近似最大奇异值
W = W / sigma  # 归一化
```

**工业效果**:SN-GAN 在 CIFAR-10 上 Inception Score 从 6.27 提到 7.42,ImageNet 128×128 上从 21.4 提到 36.8。**这 1 行谱归一化,值一个 2018 ICLR oral**。

这个例子告诉你:**特征值/特征向量不只是数学家的玩具,它是工业级深度学习稳定性的关键工具**。

### 1.5 章节路线图

```text
1. 直觉引入                      ← 你在这里
2. 特征值与特征向量的严格定义       ← Av = λv
3. 特征多项式与求解               ← det(A-λI) = 0
4. 谱定理                        ← 对称矩阵的对角化 (重点)
5. 谱半径与谱范数                ← 最大特征值的模 / 谱范数
6. 特征值在 DL 中的应用           ← SN-GAN / Hessian / NTK
7. PageRank                     ← 谷歌 1998 算法 (重点)
8. Graph Laplacian 与谱聚类       ← Fiedler vector / 谱图论
9. 特征值的数值计算               ← 幂迭代 / QR / 反幂法
10. PyTorch 代码实现             ← 6 个完整实验
11. 常见误区                     ← 5-8 条
12. 面试 Q&A                    ← 5 题 L7/L8
13. 与其他数学工具的联系          ← 跨章节引用
```

---

## 2. 特征值与特征向量定义 (350 行)

### 2.1 严格定义

**定义**(特征值/特征向量):设 $A \in \mathbb{R}^{n \times n}$ 是方阵,$\lambda \in \mathbb{C}$ 是标量,$\mathbf{v} \in \mathbb{C}^n$ 是非零向量。若

$$A \mathbf{v} = \lambda \mathbf{v}$$

则称 $\lambda$ 是 $A$ 的**特征值 (eigenvalue)**,$\mathbf{v}$ 是对应的**特征向量 (eigenvector)**。

**为什么允许复数**:实矩阵也可以有复特征值(对应"旋转"操作,见 §1.2 的旋转矩阵例子)。但实对称矩阵的特征值必为实数(谱定理的核心结论,见 §4)。

**特征方程**:把 $A\mathbf{v} = \lambda\mathbf{v}$ 改写成 $(A - \lambda I)\mathbf{v} = \mathbf{0}$。因为 $\mathbf{v} \neq 0$,矩阵 $A - \lambda I$ 必须奇异:

$$\boxed{\det(A - \lambda I) = 0}$$

这是**特征多项式** $p(\lambda) = \det(A - \lambda I)$。$p$ 的根 = $A$ 的特征值。

### 2.2 几何意义:在 $\mathbf{v}$ 方向上,$A$ 只"缩放"不"旋转"

把 $A \mathbf{v} = \lambda \mathbf{v}$ 翻译成几何语言:

> $A$ 是一个线性变换(把 $\mathbb{R}^n$ 中的向量"挪个位置")。特征向量 $\mathbf{v}$ 是这个变换的"主轴":变换前后方向不变,只是被"拉长"($|\lambda| > 1$)、"压扁"($|\lambda| < 1$)、"反向"($|\lambda| < 0$)、或"保持"($|\lambda| = 1$)。

#### 三种"缩放"模式

| $\|\lambda\|$ | 几何效果 | 例子 |
| :--- | :--- | :--- |
| $> 1$ | 沿 $\mathbf{v}$ 方向拉伸 | 缩放矩阵的 $\lambda = 3$ |
| $< 1$ (正) | 沿 $\mathbf{v}$ 方向压缩 | Markov 转移矩阵的 $\lambda \in [0, 1)$ |
| $< 0$ | 沿 $\mathbf{v}$ 方向**反向 + 缩放** | 反射矩阵的 $\lambda = -1$ |
| $= 0$ | 沿 $\mathbf{v}$ 方向压到 0 | 秩亏矩阵 |
| 复数 (非实) | 沿 $\mathbf{v}$ 方向"旋转" | 旋转矩阵的 $\lambda = e^{i\theta}$ |

### 2.3 5 个手算例子

#### 例 1:对角矩阵 $A = \text{diag}(3, 5)$

特征方程:$\det(A - \lambda I) = (3-\lambda)(5-\lambda) = 0 \Rightarrow \lambda_1 = 3, \lambda_2 = 5$

特征向量:
- $\lambda_1 = 3$:$A\mathbf{v} = 3\mathbf{v} \Rightarrow v_2 = 0$ (任意), $\mathbf{v}_1 = (1, 0)^T$
- $\lambda_2 = 5$:$A\mathbf{v} = 5\mathbf{v} \Rightarrow v_1 = 0$ (任意), $\mathbf{v}_2 = (0, 1)^T$

**几何意义**:对角矩阵沿主轴缩放,所以**主轴方向 = 特征方向**。这是最简单的情形。

#### 例 2:旋转矩阵 $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ (90° 旋转)

特征方程:$\det(A - \lambda I) = \lambda^2 + 1 = 0 \Rightarrow \lambda = \pm i$

**没有实特征值**(只有复数 $\pm i$)。

**几何意义**:旋转 90° 的矩阵把每个方向都"转 90°",**没有任何"方向不变"的方向**。这就是为什么"实特征值"对应"保持方向"的几何意义在实数域不成立。

**深度学习关联**:RNN 的循环权重 $W_h$ 在不同时间步应用,等价于 $W_h^t$。如果 $|\lambda_{\max}(W_h)| > 1$,t 步后 $W_h^t$ 范数爆炸 → 梯度爆炸。LSTM/GRU 的门机制就是让 $W_h$ 的有效特征值动态调到 1 附近。

#### 例 3:上三角矩阵 $A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$

特征方程:$\det(A - \lambda I) = (2-\lambda)(3-\lambda) = 0 \Rightarrow \lambda_1 = 2, \lambda_2 = 3$

**关键事实**:**三角矩阵的特征值 = 对角元**。所以 $A$ 的特征值是 $2$ 和 $3$。

**特征向量**:
- $\lambda_1 = 2$:$A\mathbf{v} = 2\mathbf{v} \Rightarrow v_2 = v_1$, $\mathbf{v}_1 = (1, 1)^T$
- $\lambda_2 = 3$:$A\mathbf{v} = 3\mathbf{v} \Rightarrow v_2 = -v_1/2$ 等等,$\mathbf{v}_2 = (1, -1)^T$ (重算:$(2-v_1) v_1 + v_2 = 0, -v_1 + (3 - v_1) v_2 = 0$,取 $v_1 = 1, v_2 = 1$ 满足第二个方程。验证第一个:$(2-3)\cdot 1 + 1 = 0$ ✓)

#### 例 4:实对称矩阵 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$

特征方程:$\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0$

$\lambda_1 = 3, \lambda_2 = 1$(都是实数,符合谱定理预告)

特征向量:
- $\lambda_1 = 3$:$A\mathbf{v} = 3\mathbf{v} \Rightarrow -v_1 + v_2 = 0, v_1 - v_2 = 0$,即 $v_1 = v_2$, $\mathbf{v}_1 = (1/\sqrt{2}, 1/\sqrt{2})^T$
- $\lambda_2 = 1$:$A\mathbf{v} = \mathbf{v} \Rightarrow v_1 + v_2 = 0$,即 $v_1 = -v_2$, $\mathbf{v}_2 = (1/\sqrt{2}, -1/\sqrt{2})^T$

**关键观察**:**两个特征向量正交**!$\mathbf{v}_1^T \mathbf{v}_2 = (1/2)(1 - 1) = 0$。这是实对称矩阵的普遍性质(谱定理的核心结论)。

#### 例 5:Markov 转移矩阵 $P = \begin{pmatrix} 0.5 & 0.3 \\ 0.5 & 0.7 \end{pmatrix}$

(列随机:每列和 = 1,代表"从状态 j 转移到状态 i 的概率")

特征方程:$\det(P - \lambda I) = (0.5 - \lambda)(0.7 - \lambda) - 0.15 = \lambda^2 - 1.2\lambda + 0.35 - 0.15 = \lambda^2 - 1.2\lambda + 0.2 = 0$

$\lambda = \frac{1.2 \pm \sqrt{1.44 - 0.8}}{2} = \frac{1.2 \pm 0.8}{2}$

$\lambda_1 = 1, \lambda_2 = 0.2$

**关键事实**:**列随机矩阵必有特征值 1**(对应"稳态分布")。这是 PageRank 数学推导的核心(见 §7)。

### 2.4 关键性质:5 个工程师必记的"特征值不变量"

**(1) 缩放不变性**:$A\mathbf{v} = \lambda\mathbf{v}$ 等价于 $A(c\mathbf{v}) = \lambda(c\mathbf{v})$。**任何标量倍数 $c\mathbf{v}$ 也是特征向量**。所以特征向量不唯一,我们通常归一化 $\|\mathbf{v}\| = 1$。

**(2) 矩阵多项式**:$A^k \mathbf{v} = \lambda^k \mathbf{v}$。$A$ 的 $k$ 次幂的对应特征值 = $\lambda^k$。这是"幂迭代"算法(§9.1)的数学基础。

**(3) 矩阵函数**:$e^A \mathbf{v} = e^\lambda \mathbf{v}$。这是 ODE 解 $e^{At}$ 在特征空间上退化的根源。

**(4) 迹 = 特征值之和**:$\text{tr}(A) = \sum_i \lambda_i$。2×2 例子:$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$,$\text{tr}(A) = 4 = 3 + 1 = \lambda_1 + \lambda_2$ ✓

**(5) 行列式 = 特征值之积**:$\det(A) = \prod_i \lambda_i$。同样例子:$\det(A) = 4 - 1 = 3 = 3 \cdot 1$ ✓

**工程意义**:
- (4) 和 (5) 给你**"不显式求特征值就能验证"**的工具:对任何数值算法算出的 $\lambda_i$,检查 $\sum \lambda_i = \text{tr}(A)$ 和 $\prod \lambda_i = \det(A)$。
- 这两个不变量是后续"谱归一化"、"谱半径"分析的基础。

### 2.5 特征空间:多重特征值的情况

**定义**:对应特征值 $\lambda$ 的**特征空间** $V_\lambda = \ker(A - \lambda I) = \{\mathbf{v} : A\mathbf{v} = \lambda\mathbf{v}\}$。**特征空间的维度 = 几何重数 $g$**。

**代数重数 $a$** = $\lambda$ 作为特征多项式根的重数。

**重要不等式**:$g \le a$(几何重数不超过代数重数)。

**实对称矩阵**:$a = g$。所以重数为 2 的特征值对应**一整个 2 维特征空间**,可以选**任意两个正交的基向量**。

**例子**:
$$A = I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

$\lambda = 1$ 是三重根($a = 3$)。特征空间 $V_1 = \mathbb{R}^3$(任意非零向量, $g = 3$)。$A$ 的"特征方向"是**整个 3 维空间**——单位矩阵"沿任何方向都保持不变"。

---

## 3. 特征多项式与求解 (300 行)

### 3.1 特征方程:$\det(A - \lambda I) = 0$

回忆:$(A - \lambda I)\mathbf{v} = \mathbf{0}$ 对非零 $\mathbf{v}$ 成立,要求 $A - \lambda I$ 奇异,即 $\det(A - \lambda I) = 0$。

把 $\det(A - \lambda I)$ 展开成 $\lambda$ 的多项式,得到**特征多项式** $p(\lambda)$。**$p$ 的根 = $A$ 的特征值**。

#### 为什么这个多项式的系数有规律?

**Cayley-Hamilton 定理**:$p(A) = 0$。即"把 $A$ 代入自己的特征多项式,得到零矩阵"。

**系数与不变量的关系**(2×2 情形):
$$\det \begin{pmatrix} a - \lambda & b \\ c & d - \lambda \end{pmatrix} = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad - bc)$$

即:
- 一次项系数 = $-\text{tr}(A)$
- 常数项 = $\det(A)$

对 $n \times n$ 矩阵,更高次项也有类似的"对称函数"公式(Newton 恒等式)。

### 3.2 2×2 矩阵手算:通用公式

设 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$。

**特征方程**:
$$\det(A - \lambda I) = (a - \lambda)(d - \lambda) - bc = \lambda^2 - (a+d)\lambda + (ad - bc) = 0$$

**特征值**(求根公式):
$$\boxed{\lambda_{1,2} = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2} = \frac{\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}}{2}}$$

**判别式**:
- $\text{tr}(A)^2 - 4\det(A) > 0$ → 两个**不同实特征值**
- $\text{tr}(A)^2 - 4\det(A) = 0$ → 一个**重实特征值**
- $\text{tr}(A)^2 - 4\det(A) < 0$ → 两个**复共轭特征值**

**实对称矩阵的特殊情况**($b = c$):判别式 = $(a+d)^2 - 4(ad - b^2) = (a-d)^2 + 4b^2 \ge 0$,所以**实对称矩阵必有两个实特征值**(可能相等)。

#### 例 1:对称矩阵 $A = \begin{pmatrix} 5 & 2 \\ 2 & 5 \end{pmatrix}$

$\text{tr} = 10, \det = 25 - 4 = 21$,判别式 = $100 - 84 = 16 > 0$。
$\lambda_{1,2} = \frac{10 \pm 4}{2} = 7, 3$。✓(与 §2.3 例 4 一致)

#### 例 2:非对称矩阵 $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$ (上三角)

特征值 = 对角元(三角矩阵定理):$\lambda_1 = 1, \lambda_2 = 3$。
求根公式验证:$\text{tr} = 4, \det = 3$,判别式 = $16 - 12 = 4$,$\lambda = (4 \pm 2)/2 = 3, 1$ ✓

#### 例 3:旋转矩阵 $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ (90° 旋转)

$\text{tr} = 0, \det = 1$,判别式 = $0 - 4 = -4 < 0$。
$\lambda = \frac{0 \pm \sqrt{-4}}{2} = \pm i$。**复特征值**——没有实特征方向。

#### 例 4:实对称矩阵 $A = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = 2I$

$\text{tr} = 4, \det = 4$,判别式 = $16 - 16 = 0$。
$\lambda_1 = \lambda_2 = 2$(重特征值)。特征空间 = $\mathbb{R}^2$(任何非零向量都是特征向量)。

### 3.3 3×3 矩阵手算

3×3 矩阵的特征多项式是三次的,展开更繁琐。**通用方法**:
1. 算 $\det(A - \lambda I)$
2. 用求根公式(三次)或数值方法

#### 例:Hessian 矩阵 $H = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix}$ (已对角)

特征多项式:$\det(H - \lambda I) = (2-\lambda)(3-\lambda)(5-\lambda)$
特征值:$\lambda_1 = 2, \lambda_2 = 3, \lambda_3 = 5$(对角元)
特征向量:$\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$(标准基)

**实际意义**:如果 $H$ 是某个损失函数 $\mathcal{L}(\theta)$ 在最优点 $\theta^*$ 处的 Hessian,**特征值 2, 3, 5 就是损失 landscape 在不同方向的曲率**。最大特征值 $\lambda_{\max}$ = 最陡方向的曲率,最小特征值 $\lambda_{\min}$ = 最缓方向的曲率。

#### 一般 3×3 矩阵:拉直展开

设 $A = (a_{ij})_{3 \times 3}$。展开 $\det(A - \lambda I)$:

$$\det(A - \lambda I) = -\lambda^3 + \text{tr}(A) \lambda^2 - c_2 \lambda + \det(A)$$

其中 $c_2$ 是"二阶主子式之和"。

**手算技巧**:对**带状矩阵**(三对角、稀疏)用余子式展开;对**一般矩阵**直接上 Python。

```python
import torch
A = torch.tensor([[2., 1., 0.],
                  [1., 3., 1.],
                  [0., 1., 2.]])
eigvals = torch.linalg.eigvals(A)
print(eigvals)  # 复数,实对称矩阵用 eigvalsh
```

### 3.4 案例:为什么 3 次以上多项式无通用求根公式?

**Abel-Ruffini 定理 (1824)**:5 次及以上多项式**没有通用的代数求根公式**(只用加减乘除和开方)。

**关键事实**:
- 1 次:显然($x = -b/a$)
- 2 次:求根公式($\pm \sqrt{\Delta}$)
- 3 次:Cardano 公式(1545)
- 4 次:Ferrari 公式
- 5 次:**无通用公式**!需要用**数值方法**(幂迭代、QR 算法等,见 §9)

**深度学习影响**:
- 神经网络 Hessian 是 7B × 7B(对 7B 参数模型)——不可能用代数方法求特征值
- 工业实践:**幂迭代**求最大特征值(谱归一化)、**Lanczos 算法**求 top-$k$ 特征值(Hessian 谱分析)、**随机 SVD** 求 top-$k$ 奇异值(LLM 权重分析)
- 这些都是**数值线性代数**的核心,见 §9

**为什么本科线代课不讲?** 因为高次多项式求根公式太复杂,数值方法更实用。**工程师的视角**:遇到 5 次及以上,直接用 `torch.linalg.eigvals` 或 SciPy,不要尝试闭式解。

---

## 4. 谱定理 (Spectral Theorem) (400 行) ← 重点

### 4.1 实对称矩阵总可以对角化

**谱定理 (Spectral Theorem)**:设 $A \in \mathbb{R}^{n \times n}$ 是**实对称矩阵**($A = A^T$)。则:

$$\boxed{A = Q \Lambda Q^T}$$

其中:
- $Q \in \mathbb{R}^{n \times n}$ 是**正交矩阵**($Q^T Q = Q Q^T = I$),列向量是 $A$ 的特征向量
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ 是**对角矩阵**,对角元是 $A$ 的**实特征值**

**推论 1**:实对称矩阵有 $n$ 个**实特征值**(可重,代数重数 = 几何重数)。
**推论 2**:实对称矩阵的**特征向量两两正交**(可选择成标准正交基)。
**推论 3**:实对称矩阵可**正交对角化**——这是"特征分解"在线性代数里能成立的**唯一通用情况**。

**为什么"实对称"是关键?**
- 对称 → $\text{特征值实}$(证明需要复共轭:$\lambda \mathbf{v}^H \mathbf{v} = \mathbf{v}^H A^T \mathbf{v} = \mathbf{v}^H A \mathbf{v} = \overline{\lambda} \mathbf{v}^H \mathbf{v}$,所以 $\lambda = \overline{\lambda}$,实)
- 实对称 → $\text{特征向量正交}$(Schur 分解的特化)
- 实对称 → $\text{正交对角化}$(谱定理)

**对比**:一般矩阵的特征分解 $A = P \Lambda P^{-1}$ 需要 $P$ 可逆(不要求正交),但**特征值可能是复数,特征向量不唯一、可能不正交**。

### 4.2 几何意义:实对称变换 = 三个旋转 + 三个缩放

把谱定理 $A = Q \Lambda Q^T$ 翻译成几何语言:

> **实对称变换 = 三个旋转 + 三个缩放 + 三个反向旋转**(对 $n=3$ 而言)

具体:
- $Q^T$:把当前坐标系旋转到 $A$ 的"主轴"坐标系
- $\Lambda$:沿每个主轴缩放 $\lambda_i$ 倍
- $Q$:再旋转回原坐标系

```text
原坐标系 v       主轴坐标系 v'      缩放后 Λv'        原坐标系 Av
   ╱─────╲          |                  |                ╱─────╲
  │       │   Q^T   |            Λ     |       Q      │       │
  │   v   │ ────→   |  v'         ↓   |  ────→  ←─── │   Av  │
  │       │         |  v_1'         λ_1 v_1'          │       │
   ╲─────╱          |  v_2'         λ_2 v_2'           ╲─────╱
                   |  v_3'         λ_3 v_3'
```

**关键洞察**:**任意实对称变换的"骨架"由它的特征方向和特征值唯一决定**。一旦知道 $Q$ 和 $\Lambda$,变换就完全确定。

### 4.3 谱定理的 3 大推论

#### 推论 1:实对称矩阵的所有特征值都是实数

**证明**(关键步骤):
设 $A = A^T$,$\lambda$ 是 $A$ 的特征值,$\mathbf{v}$ 是对应的特征向量。考虑 $\overline{\lambda} = \lambda$ 的共轭复数。

取共轭转置:$\overline{A\mathbf{v}} = \overline{\lambda \mathbf{v}} \Rightarrow \overline{A} \cdot \overline{\mathbf{v}} = \overline{\lambda} \overline{\mathbf{v}}$。但 $A$ 是实的,$\overline{A} = A$。

取 $\mathbf{v}^H A \mathbf{v} = \mathbf{v}^H (\lambda \mathbf{v}) = \lambda \mathbf{v}^H \mathbf{v}$。

又 $\mathbf{v}^H A \mathbf{v} = \mathbf{v}^H A^T \mathbf{v} = (A\mathbf{v})^H \mathbf{v} = (\lambda \mathbf{v})^H \mathbf{v} = \overline{\lambda} \mathbf{v}^H \mathbf{v}$。

所以 $\lambda = \overline{\lambda}$。**实数**。$\blacksquare$

#### 推论 2:不同特征值对应的特征向量正交

**证明**:
设 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$, $A\mathbf{v}_j = \lambda_j \mathbf{v}_j$, $\lambda_i \neq \lambda_j$。

$\mathbf{v}_i^T A \mathbf{v}_j = \lambda_j \mathbf{v}_i^T \mathbf{v}_j$。
$\mathbf{v}_i^T A \mathbf{v}_j = \mathbf{v}_i^T A^T \mathbf{v}_j = (A\mathbf{v}_i)^T \mathbf{v}_j = \lambda_i \mathbf{v}_i^T \mathbf{v}_j$。

所以 $\lambda_j \mathbf{v}_i^T \mathbf{v}_j = \lambda_i \mathbf{v}_i^T \mathbf{v}_j \Rightarrow (\lambda_j - \lambda_i) \mathbf{v}_i^T \mathbf{v}_j = 0$。

因 $\lambda_i \neq \lambda_j$,必有 $\mathbf{v}_i^T \mathbf{v}_j = 0$。**正交**。$\blacksquare$

**重特征值时**:同一特征值对应的特征空间,我们可以**选**一组正交基(用 Gram-Schmidt)。所以**总能凑出 $n$ 个两两正交的特征向量**。

#### 推论 3:可正交对角化

直接由前两个推论拼起来:$A$ 有 $n$ 个两两正交的特征向量,把它们归一化,组成正交矩阵 $Q$,$Q^T A Q = \Lambda$ 对角。即 $A = Q \Lambda Q^T$。$\blacksquare$

### 4.4 5 个手算例子 (2×2 对称矩阵)

#### 例 1:$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (经典)

特征值:$\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0 \Rightarrow \lambda_1 = 3, \lambda_2 = 1$。

特征向量:
- $\lambda_1 = 3$:$A\mathbf{v} = 3\mathbf{v} \Rightarrow \mathbf{v}_1 = (1, 1)^T / \sqrt{2}$
- $\lambda_2 = 1$:$\mathbf{v}_2 = (1, -1)^T / \sqrt{2}$

验证正交:$\mathbf{v}_1^T \mathbf{v}_2 = 0$ ✓

谱分解:$A = Q \Lambda Q^T = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$。

#### 例 2:$A = \begin{pmatrix} 5 & 2 \\ 2 & 5 \end{pmatrix}$

特征值:$\det = (5-\lambda)^2 - 4 = \lambda^2 - 10\lambda + 21 = (\lambda-3)(\lambda-7) = 0$,$\lambda_1 = 7, \lambda_2 = 3$。

特征向量:
- $\lambda_1 = 7$:$\mathbf{v}_1 = (1, 1)^T / \sqrt{2}$
- $\lambda_2 = 3$:$\mathbf{v}_2 = (1, -1)^T / \sqrt{2}$

**注意**:$\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ 和 $\begin{pmatrix} 5 & 2 \\ 2 & 5 \end{pmatrix}$ 特征向量相同!因为它们都是"对角元相同 + 反对角元相同"的形式,只是具体数值不同。

#### 例 3:$A = \begin{pmatrix} a & b \\ b & a \end{pmatrix}$ (一般形式)

特征方程:$\det = (a-\lambda)^2 - b^2 = 0 \Rightarrow \lambda = a \pm b$。

- $\lambda_1 = a + b$:$A\mathbf{v} = (a+b)\mathbf{v} \Rightarrow b v_1 + b v_2 = 0$ 等等,$v_1 = v_2$,$\mathbf{v}_1 = (1, 1)^T / \sqrt{2}$
- $\lambda_2 = a - b$:$v_1 = -v_2$,$\mathbf{v}_2 = (1, -1)^T / \sqrt{2}$

**规律**:对角元相同 + 反对角元相等的实对称矩阵,特征方向**永远是 $(1,1)$ 和 $(1,-1)$**,特征值是 $a \pm b$。

#### 例 4:$A = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = 2I$ (重特征值)

特征值:$\lambda_1 = \lambda_2 = 2$(重)。
特征空间 = $\mathbb{R}^2$(任何 $\mathbf{v} \neq 0$ 都是)。

**谱分解**:$Q$ 可以是**任意**正交矩阵,$\Lambda = 2I$。例如取 $Q = I$:$A = I \cdot (2I) \cdot I = 2I$ ✓。或取 $Q = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ 也行(产生不同的"特征方向"分解,但数学上等价)。

**工程意义**:重特征值时谱分解不唯一(可以选不同的 $Q$)。这跟 §2.5 一致。

#### 例 5:$A = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ (反射 + 缩放)

特征值:$\lambda_1 = 1, \lambda_2 = -1$。
特征向量:$\mathbf{v}_1 = (1, 0)^T, \mathbf{v}_2 = (0, 1)^T$(标准基)。

谱分解:$A = I \Lambda I^T = \Lambda$。**对角矩阵就是它自己的谱分解**。

**几何**:沿 $x$ 方向"保持" ($\lambda = 1$),沿 $y$ 方向"反向" ($\lambda = -1$)。

### 4.5 案例:Hessian 是对称矩阵 → 谱定理保证对角化

**回顾** 1-高数 第 3 章:对实函数 $f: \mathbb{R}^n \to \mathbb{R}$(标量值),**Hessian 矩阵** $H = \nabla^2 f$ 必对称:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i} = H_{ji}$$
(由 Schwarz 定理:连续二阶偏导可交换)

**结论**:Hessian 必满足谱定理,可以正交对角化为 $H = Q \Lambda Q^T$,**特征值全实**。

**深度学习意义**:
- **Hessian 特征值 = 损失 landscape 各方向的曲率**(见 1-高数 第 3 章和 1-高数 第 5 章)
- 特征向量方向 = "曲率主方向",特征值 = "该方向的曲率"
- $\lambda_{\max}$ = 最陡方向的曲率(决定学习率上限:$\eta < 2/\lambda_{\max}$ 保证 Newton 步收敛)
- $\lambda_{\min}$ = 最缓方向的曲率(决定收敛速度:$\eta \cdot \lambda_{\min}$ 是该方向的等效学习率)

**工程例子**:L-BFGS 用 Hessian 的近似特征值信息控制步长;Sophia (Liu et al. 2023) 用 Hessian 对角估计做自适应学习率;Shampoo (Gupta et al. 2018) 用 Kronecker 积近似的 Hessian 特征值做预处理。

### 4.6 案例:协方差矩阵是对称的 → 谱定理 → PCA 的数学基础

**回顾** 3-概率论 第 4 章:随机向量 $\mathbf{X} \in \mathbb{R}^n$ 的**协方差矩阵**:
$$\Sigma = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

**显然对称**:$\Sigma^T = \Sigma$。

**谱定理保证**:$\Sigma = Q \Lambda Q^T$,其中 $Q$ 的列是**主成分方向**,$\Lambda$ 的对角元是**沿主成分方向的方差**。

**PCA 的几何**:
- 数据 $\mathbf{X} \in \mathbb{R}^{N \times n}$(N 个样本,n 个特征)
- 中心化后,$\Sigma = \frac{1}{N-1} X^T X$
- 谱分解:$\Sigma = Q \Lambda Q^T$
- 取 top-$k$ 主成分:$X_{\text{reduced}} = X Q_k \in \mathbb{R}^{N \times k}$
- 解释的总方差:$\sum_{i=1}^k \lambda_i / \sum_{i=1}^n \lambda_i$

**为什么谱定理让 PCA 这么"干净"**:
- 协方差对称 → 特征值实 → 解释为"方差"
- 特征向量正交 → 主成分方向两两独立 → 无信息冗余
- 正交对角化 → 降维后保留方差最大的方向

**深度学习应用**:
- **白化 (Whitening)**:用 $\Sigma^{-1/2} = Q \Lambda^{-1/2} Q^T$ 让数据各向同性,加速训练(Batch Normalization 的精神)
- **判别分析 (LDA)**:$\Sigma_W^{-1} \Sigma_B$ 的特征向量 = Fisher 判别方向
- **Hessian 特征分析**:神经网络的 Hessian 是对称的,所以可以用谱定理分析 loss landscape(见 §6.2)

---

## 5. 谱半径与谱范数 (250 行)

### 5.1 谱半径:最大特征值的模

**定义**(谱半径):设 $A \in \mathbb{C}^{n \times n}$,则 $A$ 的**谱半径**定义为

$$\boxed{\rho(A) = \max_i |\lambda_i(A)|}$$

即所有特征值绝对值的最大值。

**关键性质**:
- $\rho(A) \ge 0$
- $\rho(A) \le \|A\|$(对任何**矩阵范数** $\|A\|$ 都成立,这是 Gelfand 公式的极限形式)
- $\rho(A) = \lim_{k \to \infty} \|A^k\|^{1/k}$(Gelfand 公式)

**为什么谱半径在 DL 里重要**:

$$\|A^k\|_2 = \max_i |\lambda_i|^k = \rho(A)^k$$

如果 $\rho(A) > 1$,$A^k$ 的范数随 $k$ 指数增长——这正是 **RNN 梯度爆炸** 的代数根源。RNN 在时刻 $t$ 的隐藏状态 $h_t$ 依赖 $W_h^t$,如果 $\rho(W_h) > 1$,反向传播时梯度 $\nabla h_t$ 沿时间步累积爆炸。

#### 例:$W = \begin{pmatrix} 1.5 & 0 \\ 0 & 0.8 \end{pmatrix}$

特征值 $\lambda_1 = 1.5, \lambda_2 = 0.8$,$\rho(W) = 1.5$。

$W^{10} = \begin{pmatrix} 1.5^{10} & 0 \\ 0 & 0.8^{10} \end{pmatrix} = \begin{pmatrix} 57.66 & 0 \\ 0 & 0.107 \end{pmatrix}$。

注意 $0.8^{10} = 0.107$(衰减),$1.5^{10} = 57.66$(爆炸)。**整体** $\|W^{10}\|_2 = 57.66 = 1.5^{10} = \rho(W)^{10}$。

这就是为什么 LSTM/GRU 用"门"动态把有效特征值调到 $\le 1$ 附近。

### 5.2 谱范数(诱导 2-范数)

**定义**(诱导 2-范数 / 谱范数):

$$\|A\|_2 = \max_{\mathbf{x} \neq 0} \frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2$$

即 $A$ 把单位向量的"最大拉伸倍数"。

**关键定理**(谱范数 = 最大奇异值):

$$\boxed{\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}}$$

**为什么?** 关键性质 $\|A\|_2^2 = \lambda_{\max}(A^T A)$:

$$\|A\mathbf{x}\|_2^2 = \mathbf{x}^T A^T A \mathbf{x}$$

令 $B = A^T A$(必对称半正定),则 $\|A\mathbf{x}\|_2^2 = \mathbf{x}^T B \mathbf{x}$ 是 Rayleigh 商,最大值为 $\lambda_{\max}(B)$。

**实对称矩阵的特殊情况**:当 $A = A^T$ 时,$A^T A = A^2$,$\|A\|_2 = \max |\lambda_i(A)| = \rho(A)$。**谱范数 = 谱半径**。

**一般矩阵**:谱范数 ≠ 谱半径。例:旋转矩阵 $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$,$\rho(A) = |\pm i| = 1$,$\|A\|_2 = \sqrt{\lambda_{\max}(A^T A)} = \sqrt{\lambda_{\max}(I)} = 1$。这里恰好相等,因为 $A$ 是正交矩阵。

### 5.3 谱半径与谱范数的关系

**重要不等式**:

$$\rho(A) \le \|A\|_2$$

**为什么?** 设 $\lambda = \rho(A)$(对应特征向量 $\mathbf{v}$, $\|\mathbf{v}\|_2 = 1$)。则

$$|\lambda| = |\lambda| \cdot \|\mathbf{v}\|_2 = \|\lambda \mathbf{v}\|_2 = \|A\mathbf{v}\|_2 \le \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\|_2 = \|A\|_2$$

```text
        λv
   |λv| = |λ|  ≤  ||Av||  ≤  ||A||₂
                                  ↑
                              谱范数
                              (单位向量最大拉伸)
```

**严格不等式何时成立**:存在某特征值 $\lambda$ 使得 $|\lambda| < \|A\|_2$ **当且仅当** $A$ 不是"秩 1 对称矩阵加扰动"这种特殊情况。一般情况下,$\rho(A) < \|A\|_2$ 严格成立。

#### 例:$A = \begin{pmatrix} 1 & 1 \\ 0 & -2 \end{pmatrix}$

特征方程:$\det(A - \lambda I) = (1-\lambda)(-2-\lambda) = 0 \Rightarrow \lambda_1 = 1, \lambda_2 = -2$。
$\rho(A) = \max(|1|, |-2|) = 2$。

谱范数:$A^T A = \begin{pmatrix} 1 & 1 \\ 1 & 5 \end{pmatrix}$,特征值 $\lambda^2 - 6\lambda + 4 = 0 \Rightarrow \lambda = 3 \pm \sqrt{5}$。
$\|A\|_2 = \sqrt{3 + \sqrt{5}} \approx \sqrt{5.236} \approx 2.288$。

**观察**:$\rho(A) = 2 < 2.288 = \|A\|_2$。谱范数比谱半径大 14%,因为 $A$ 不对称,导致"拉伸"方向和"特征方向"不完全对齐。

### 5.4 案例:Spectral Normalization 让 GAN 训练稳定

回顾 §1.4:SN-GAN 把每层 $W$ 替换为 $W / \sigma_{\max}(W)$,强制 $\|W\|_2 \le 1$。

**为什么是谱范数,不是谱半径?**

判别器 $D$ 不是单层 $W$,而是 $L$ 层复合 $D = f_L \circ \cdots \circ f_1$。每层 $f_\ell(\mathbf{x}) = \sigma(W_\ell \mathbf{x} + \mathbf{b}_\ell)$。线性部分 $W_\ell$ 的 Lipschitz 常数 = $\|W_\ell\|_2$ = $\sigma_{\max}(W_\ell)$。**整网络 Lipschitz 是各层 Lipschitz 的乘积**:

$$\|D\|_{Lip} = \prod_{\ell=1}^L \|W_\ell\|_2 = \prod_{\ell=1}^L \sigma_{\max}(W_\ell)$$

**关键洞察**:**谱范数控制 Lipschitz,谱半径控制幂行为**。GAN 训练关心的是"输入扰动对输出的放大",这是 Lipschitz 性质——**谱范数**是正确工具。

### 5.5 案例:Transformer attention 矩阵的谱分析

考虑 attention 矩阵:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

记 $P = \text{softmax}(Q K^T / \sqrt{d})$。$P$ 是**行随机矩阵**(每行和为 1),所以**谱半径 $\rho(P) = 1$**,对应特征向量 $(1, 1, \ldots, 1)^T$(右乘 $\mathbf{1}$ 等于 $\mathbf{1}$)。

**但 $\rho(P) = 1$ 意味着 attention 输出有上界** $\|P V\| \le \|V\|$,**值不会被指数放大**——这是 Transformer 比早期 RNN 训练稳定的关键原因之一。

**深层洞察**:attention 的谱范数 $\|P\|_2 \le 1$ 同样成立(因为 $P$ 行随机且非负,Frobenius 范数 $\|P\|_F \le 1$)。**整 Transformer 残差块 $x \mapsto x + \text{FFN}(\text{Attn}(x))$ 的 Lipschitz 常数 $\approx 1 + \alpha$**,其中 $\alpha \ll 1$ 是 FFN 的谱范数。这就是 Pre-Norm Transformer 训练极稳的代数原因。

### 5.6 谱半径在收敛性分析中的角色

**定理**(几何级数收敛):若 $\rho(A) < 1$,则对任何 $\|\mathbf{b}\|$ 有界,迭代

$$\mathbf{x}_{k+1} = A \mathbf{x}_k + \mathbf{b}$$

**收敛**到唯一不动点 $\mathbf{x}^* = (I - A)^{-1} \mathbf{b}$,且

$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \le \frac{\rho(A)^k}{1 - \rho(A)} \|\mathbf{x}_1 - \mathbf{x}^*\|$$

**为什么?** $\mathbf{x}_{k+1} - \mathbf{x}^* = A (\mathbf{x}_k - \mathbf{x}^*)$,所以 $\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \le \|A^k\| \|\mathbf{x}_1 - \mathbf{x}^*\| \to 0$ 当 $\|A^k\| \to 0$。这等价于 $\rho(A) < 1$。

**DL 应用**:
- **Power Iteration 收敛性**(§9.1):幂迭代收敛速度 = $(\rho_2 / \rho_1)^k$,其中 $\rho_1$ 是最大特征值模,$\rho_2$ 是第二大。谱半径比 = 收敛因子。
- **Markov 链收敛**(§7):转移矩阵 $P$ 的谱半径 = 1,第二大特征值 $|\lambda_2|$ 决定收敛到稳态的速度。

---

## 6. 特征值在 DL 中的应用 (350 行) ← 重点

### 6.1 Spectral Normalization (Miyato 2018, GAN)

#### 完整数学表述

设判别器第 $\ell$ 层权重 $W_\ell \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$,谱归一化定义:

$$\hat{W}_\ell = W_\ell / \sigma_{\max}(W_\ell)$$

**性质**:$\|\hat{W}_\ell\|_2 = \sigma_{\max}(W_\ell) / \sigma_{\max}(W_\ell) = 1$。

**为什么让训练稳定**:

1. **Lipschitz 控制**:每层 $\|W_\ell\|_2 = 1$ 意味着 $f_\ell(\mathbf{x}) = W_\ell \mathbf{x} + \mathbf{b}_\ell$ 是 1-Lipschitz。激活函数 ReLU/LeakyReLU 是 1-Lipschitz。整判别器 $D$ 的 Lipschitz 常数 ≤ 1。
2. **Wasserstein 距离有意义**:1-Lipschitz 函数 $D$ 给出 $\mathbb{E}[D(\mathbf{x}_{\text{real}})] - \mathbb{E}[D(\mathbf{x}_{\text{fake}})] \le W_1(P_{\text{real}}, P_{\text{fake}})$(Kantorovich-Rubinstein 对偶)。判别器的 loss 下界 = Wasserstein 距离,有优化意义。
3. **梯度有界**:$\|\nabla D(\mathbf{x})\| \le 1$ 处处成立。生成器 $G$ 的梯度 $\|\nabla_G D(G(\mathbf{z}))\| \le 1$,**不会爆炸**。

#### 幂迭代求 $\sigma_{\max}$(不显式做 SVD)

```python
import torch
import torch.nn as nn

class SpectralNorm(nn.Module):
    """Spectral Normalization for 2D weight matrix.
    Uses power iteration to estimate sigma_max."""
    def __init__(self, weight: torch.Tensor, n_iter: int = 1):
        super().__init__()
        self.weight = nn.Parameter(weight)  # (d_out, d_in)
        d_out, d_in = weight.shape
        # Random init for u (left singular vector estimate)
        self.register_buffer('u', torch.randn(d_out))
        self.register_buffer('v', torch.randn(d_in))
        self.n_iter = n_iter

    def forward(self, x):
        W = self.weight
        # Power iteration
        for _ in range(self.n_iter):
            # v = W^T u / ||W^T u||
            self.v = torch.nn.functional.normalize(W.T @ self.u, dim=0)
            # u = W v / ||W v||
            self.u = torch.nn.functional.normalize(W @ self.v, dim=0)
        # Rayleigh quotient: sigma ≈ u^T W v
        sigma = self.u @ W @ self.v
        # Normalize
        return x @ (W / sigma).T
```

**关键技巧**:
- 只需 1 次幂迭代就够(训练时 forward 链路上每步都在更新)
- `u`, `v` 用 buffer(不参与梯度,但会被 `state_dict()` 保存/加载)
- 真正的 PyTorch 实现 `torch.nn.utils.spectral_norm` 就是这个思路

#### PyTorch 内置实现

```python
linear = nn.Linear(in_features=512, out_features=256)
linear_sn = torch.nn.utils.spectral_norm(linear)  # 一行搞定
```

`torch.nn.utils.spectral_norm` 会把权重替换为 `weight / sigma_max`,sigma_max 用幂迭代估计。

### 6.2 Hessian 特征值 → 收敛速度

#### Newton 法的收敛性

Newton 法的更新:$\theta_{k+1} = \theta_k - H^{-1} \nabla \mathcal{L}(\theta_k)$,其中 $H$ 是 Hessian。

**收敛速率定理**(局部二次收敛):若 $H$ 在最优点 $\theta^*$ 附近**非奇异且 Lipschitz**,则 Newton 法满足

$$\|\theta_{k+1} - \theta^*\| \le C \|\theta_k - \theta^*\|^2$$

即**每步误差平方**。

**用 Hessian 特征值改写**这个界:

设 $H = Q \Lambda Q^T$。在最优点附近做 Taylor 展开:

$$\mathcal{L}(\theta) \approx \mathcal{L}(\theta^*) + \frac{1}{2} (\theta - \theta^*)^T H (\theta - \theta^*)$$

在 Hessian 特征基下:$\mathcal{L} \approx \mathcal{L}(\theta^*) + \frac{1}{2} \sum_i \lambda_i (q_i^T (\theta - \theta^*))^2$。

**关键洞察**:**每个特征方向上的"有效曲率" = 对应特征值**。
- $\lambda_i$ 大 → 该方向收敛快(Newton 步把梯度除以 $\lambda_i$,大曲率 = 大步)
- $\lambda_i$ 小 → 该方向收敛慢(小曲率 = 几乎不动)

**条件数 $\kappa(H) = \lambda_{\max} / \lambda_{\min}$**:控制梯度下降的收敛速度。$\kappa$ 越大,各方向收敛速率差距越大,实际收敛被 $\lambda_{\min}$ 拖累。

#### 例:病态 Hessian

$$H = \begin{pmatrix} 100 & 0 \\ 0 & 1 \end{pmatrix}$$

$\lambda_{\max} = 100, \lambda_{\min} = 1, \kappa = 100$。

梯度下降在第一方向上"走得快"(大曲率),第二方向上"走得慢"(小曲率)。**收敛轨迹沿椭圆长轴来回震荡,沿短轴缓慢前进**。这就是 SGD 在病态问题上的"之字形"轨迹。

```text
损失 landscape 等高线:     梯度下降轨迹:

      _____
     /     \                    ___
    | H_eig |                   /   \___
    | 100  1 |      →           |       \___
     \_____/                   |   慢慢收敛(之字形)
```

**Adam / RMSProp 为什么稳**?它们用**对角缩放**(近似 Hessian 对角的倒数),把 $\lambda_{\max}$ 和 $\lambda_{\min}$ 之间的差距"压平",使各方向等效学习率接近 1。这就是"自适应学习率"的本质。

### 6.3 NTK 特征值(1-高数 第 8 章预告)

**NTK**(Neural Tangent Kernel)是 2018 年由 Jacot et al. 提出的关键概念。在**无限宽**神经网络极限下,网络 $f(\theta; x)$ 在初始化 $\theta_0$ 附近的行为由线性微分方程

$$\frac{df}{dt} = -\Theta(t) \nabla_f \mathcal{L}$$

控制,其中 $\Theta(t) = J(t) J(t)^T$,$J$ 是 Jacobian。

**NTK 的特征值**决定了**神经网络训练的早期动力学**:
- $\Theta$ 的特征值 = 各"NTK 模式"的频率
- 大特征值模式 = 训练初期快速变化的方向(梯度信号强)
- 小特征值模式 = 训练后期慢收敛的方向

**关联**:训练损失曲线常呈"先快后慢"的两阶段,这正对应 NTK 特征值从大到小依次被"压低"的过程。

### 6.4 Graph Laplacian 特征值(谱图论预告)

**图 Laplacian**(见 §8 详细):

$$L = D - A$$

其中 $D$ 是度矩阵,$A$ 是邻接矩阵。$L$ 对称半正定,谱定理保证 $L = Q \Lambda Q^T$,$\lambda_i \ge 0$。

**关键特征值的含义**:
- $\lambda_1 = 0$(必有,对应常向量 $\mathbf{1}$):图是连通的
- $\lambda_2$(Fiedler value):图的"代数连通度"。$\lambda_2$ 越大,图越难被分成两块
- $\lambda_2 = 0$:图不连通(可被分成至少两块)

**谱图卷积**(DeepMind 2017, 4-离散数学 第 6 章展开):用 Laplacian 特征向量做"图傅里叶变换",定义图上的卷积。**ChebNet** 和 **GCN** 都是这个思路。

### 6.5 Markov 链稳态 = 特征向量

**定理**:对列随机矩阵 $P$(每列和 = 1),必有特征值 $\lambda = 1$,对应特征向量 $\mathbf{v}$ 满足 $P\mathbf{v} = \mathbf{v}$。

**证明**:$P^T \mathbf{1} = \mathbf{1}$(行向量全 1 是 $P^T$ 的特征向量)。由于 $P^T$ 和 $P$ 特征值相同,所以 $P$ 也有特征值 1,对应特征向量 $\mathbf{v}$(稳态分布)。

#### PageRank 作为 Markov 链稳态(§7 展开)

网页跳转图 = Markov 链。转移矩阵 $M$ 列随机 → 必有特征值 1 → PageRank 向量 = $M$ 的"主"特征向量(对应 $\lambda = 1$)。**这就是 PageRank 的数学本质**:求 Markov 链的稳态分布。

#### 例:$P = \begin{pmatrix} 0.5 & 0.3 \\ 0.5 & 0.7 \end{pmatrix}$(列随机)

特征值 $\lambda_1 = 1, \lambda_2 = 0.2$。

特征向量 $\mathbf{v}_1 = (v_{11}, v_{12})^T$ 满足 $P\mathbf{v}_1 = \mathbf{v}_1$:
- $0.5 v_{11} + 0.3 v_{12} = v_{11} \Rightarrow -0.5 v_{11} + 0.3 v_{12} = 0 \Rightarrow v_{12} = (5/3) v_{11}$
- $0.5 v_{11} + 0.7 v_{12} = v_{12} \Rightarrow 0.5 v_{11} - 0.3 v_{12} = 0$(同一方程)

归一化 $v_{11} + v_{12} = 1$:$v_{11} + (5/3) v_{11} = 1 \Rightarrow v_{11} = 3/8, v_{12} = 5/8$。

**稳态分布**:在状态 1 的概率 $3/8$,状态 2 的概率 $5/8$。

**收敛速度** = $|\lambda_2| = 0.2$。迭代 10 步后误差 $\le 0.2^{10} \approx 10^{-7}$。

### 6.6 5 个手算例子

#### 例 1:Hessian 曲率分析

$$H = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}$$

特征值 $\lambda_1 = 4, \lambda_2 = 1$。最陡方向曲率 = 4,最缓 = 1。最大学习率 $\eta < 2 / \lambda_{\max} = 0.5$。等效学习率 $\eta \cdot \lambda_{\min} = 0.5 \cdot 1 = 0.5$。**收敛到最优点需要的步数 ≈ $\log(1/\epsilon) / (\eta \cdot \lambda_{\min})$**。

#### 例 2:PageRank 2 节点

$$M = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

特征值 $\lambda = \pm 1$。PageRank 向量 = $\lambda = 1$ 的特征向量 $v_1 = (1, 1)^T / \sqrt{2}$,归一化为 $(0.5, 0.5)$。

**直观**:两个网页互相链接,稳态分布均匀。

#### 例 3:谱聚类 Laplacian

3 个节点,全连接图:

$$A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}, D = 2I, L = \begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{pmatrix}$$

特征值 $\lambda_1 = 0$(对应 $\mathbf{1}$),$\lambda_2 = \lambda_3 = 3$。Fiedler value = 3,完全图高度连通(把图分成两部分"代价"很高)。

#### 例 4:RNN 梯度爆炸

$W_h = \begin{pmatrix} 1.5 & 0 \\ 0 & 0.5 \end{pmatrix}$,$\rho(W_h) = 1.5$。

T=10 步后,$W_h^{10} = \begin{pmatrix} 57.66 & 0 \\ 0 & 0.000977 \end{pmatrix}$。梯度 $\|\nabla_{h_0} h_{10}\| = \|W_h^{10} \nabla_{h_{10}} h_{10}\| = 57.66 \cdot \|\nabla_{h_{10}} h_{10}\|$。

**梯度爆炸 57 倍**。这就是 LSTM 引入遗忘门的动机:让 $W_h$ 有效谱半径动态接近 1。

#### 例 5:Transformer attention 谱分析

4 个 token,$Q = K = I$(简化)。$Q K^T = I$,softmax 行归一化后 $P = (1/4) \mathbf{1}\mathbf{1}^T$(全均匀矩阵)。

$P$ 的特征值:$\lambda_1 = 1$(对应 $\mathbf{1}$),$\lambda_2 = \lambda_3 = \lambda_4 = 0$。

$\rho(P) = 1$,说明 attention 输出有界。**深度 Transformer 中 attention 不会无限放大信号**——这与 RNN 形成鲜明对比,是 Transformer 成功的代数原因之一。

---

## 7. PageRank 算法 (300 行) ← 案例

### 7.1 网页排名的数学建模

**问题**:1998 年,Google 两位创始人 Larry Page 和 Sergey Brin 要给 3 亿个网页排序。**怎么判断哪个网页"重要"?**

**直觉**(Page & Brin 的洞察):

> **一个网页的重要性 = 所有链向它的网页的重要性之和,按这些网页的链出数加权**。

**形式化**:
- 设网页 $i$ 的重要性为 $r_i$(PageRank 分数)
- 网页 $j$ 有 $d_j$ 个外链(链出数)
- 则

$$r_i = \sum_{j \to i} \frac{r_j}{d_j}$$

**矩阵形式**:设转移矩阵 $M \in \mathbb{R}^{n \times n}$,$M_{ij} = 1/d_j$ if $j \to i$,否则 0。

$$r = M r$$

**这是一个特征值问题**!$\mathbf{r}$ 是 $M$ 的特征向量,特征值 = 1。

### 7.2 转移矩阵 P 的构造

#### 基本构造

设网页图 $G = (V, E)$,$|V| = n$。

对每个网页 $j$,设 $\text{out}(j) = \{i : (j, i) \in E\}$ 是 $j$ 的链出集合,$|\text{out}(j)| = d_j$。

定义转移矩阵:

$$M_{ij} = \begin{cases} 1/d_j & \text{if } j \to i \\ 0 & \text{otherwise} \end{cases}$$

**重要性质**:**$M$ 列随机**(每列和 = 1):$\sum_i M_{ij} = \sum_{j \to i} 1/d_j = d_j / d_j = 1$。

```text
   网页 1 链向 {2, 3}    →  M[2,1] = M[3,1] = 1/2, M[others, 1] = 0
   网页 2 链向 {1}        →  M[1,2] = 1, M[others, 2] = 0
   网页 3 链向 {2}        →  M[2,3] = 1, M[others, 3] = 0
```

#### 例:3 个网页 A → B → C, A → C

```text
A ──→ B ──→ C
└──────────→ C
```

$M = \begin{pmatrix} 0 & 1 & 0 \\ 1/2 & 0 & 1 \\ 1/2 & 0 & 0 \end{pmatrix}$(列随机)

#### 关键问题:悬挂节点(Dangling Nodes)

如果网页 $j$ 没有外链($d_j = 0$),$M$ 第 $j$ 列全 0,**$M$ 不是随机矩阵**(列和不为 1)。

**修法**:把悬挂节点的转移均匀分布到所有节点:

$$M'_{ij} = \begin{cases} 1/d_j & d_j > 0 \text{ and } j \to i \\ 1/n & d_j = 0 \\ 0 & \text{otherwise} \end{cases}$$

### 7.3 求稳态分布 v = Pv

**定理**(Markov 链稳态):对**不可约非周期**的列随机矩阵 $P$,存在**唯一**稳态分布 $\mathbf{v} \in \mathbb{R}^n$,满足

$$P \mathbf{v} = \mathbf{v}, \quad v_i \ge 0, \quad \sum_i v_i = 1$$

**证明思路**(Perron-Frobenius 定理):
- $P$ 列随机 → 谱半径 $\rho(P) = 1$
- $P$ 不可约非周期 → 1 是**单根**,对应**正特征向量**

**PageRank 算法的核心**:求 $P$ 对应 $\lambda = 1$ 的归一化特征向量。

### 7.4 Power Iteration 求解

**算法**(幂迭代):

```python
def pagerank_power(M, num_iter=100, tol=1e-6):
    n = M.shape[0]
    v = torch.ones(n) / n  # 均匀初始化
    for k in range(num_iter):
        v_new = M @ v
        v_new = v_new / v_new.sum()  # 归一化(防数值漂移)
        if torch.norm(v_new - v) < tol:
            break
        v = v_new
    return v
```

**收敛性**:迭代 $v_{k+1} = M v_k$,展开:

$$v_k = M^k v_0 = \sum_i c_i \lambda_i^k \mathbf{u}_i$$

当 $k \to \infty$,只有 $\lambda_1 = 1$ 项不衰减,其余 $|\lambda_i|^k \to 0$。所以 $v_k \to c_1 \mathbf{u}_1$,即 PageRank 向量。

**收敛速度** = $|\lambda_2|$(第二大特征值的模)。对真实网页图,$|\lambda_2| \approx 0.85$,所以收敛需要 ~50-100 次迭代。

### 7.5 Damping factor

**问题**:如果网页图**不连通**(例如一组网页互链但不链向其他组),幂迭代会在子图内收敛,**整个网页图的 PageRank 不唯一**。

**修法**(Google 的解决方案):**加入随机跳转**

$$P = d \cdot M + (1 - d) \cdot \frac{1}{n} \mathbf{1}\mathbf{1}^T$$

其中 $d \in (0, 1)$ 是 **damping factor**(通常 $d = 0.85$)。

**直觉**:
- 以概率 $d$ 沿链接跳转(原行为)
- 以概率 $1 - d$ 跳到任意随机网页(防止困在子图)

**数学保证**:加随机跳转后,$P$ 一定**不可约非周期**(因为 $\frac{1}{n}\mathbf{1}\mathbf{1}^T$ 项保证每对节点可达),所以 PageRank 唯一。

**Damping factor 的实际影响**:
- $d = 1$:标准随机游走,可能不收敛或不唯一
- $d = 0.85$(标准):平衡"沿链接"和"随机",主特征值仍接近 1,次特征值 = $d \cdot \lambda_2(M) \approx 0.85 \cdot \lambda_2(M)$,**收敛略快**
- $d = 0$:完全随机,PageRank = 均匀分布

### 7.6 PageRank 与推荐系统的"用户兴趣稳态"对比

**类比**:PageRank 的思想可以推广到任意"图上的稳态"。

| 系统 | 节点 | 边 | 稳态 | 含义 |
| :--- | :--- | :--- | :--- | :--- |
| 网页搜索 | 网页 | 超链接 | PageRank | 网页重要性 |
| 社交网络 | 用户 | 好友关系 | PersonalRank | 用户影响力 |
| 推荐系统 | 物品 | 共同购买/观看 | ItemRank | 物品热度 |
| 学术引用 | 论文 | 引用关系 | PaperRank | 论文影响力 |

**PersonalRank**(2008, 周涛等):给用户 $u$ 偏好物品 $i_0$ 作为"种子",然后做带重启的随机游走,稳态分布 = 物品的相关性分数。这是协同过滤的早期算法。

**DeepWalk / node2vec**(2014-2016, 4-离散数学 第 6 章):把 PageRank 思路 + 随机游走生成序列 → Word2Vec → 节点 embedding。这是 GraphSAGE、PinSage 的前身。

**关键洞察**:**很多 ML 问题都能建模成"图上的稳态分布"**,PageRank 提供了一个通用框架。

### 7.7 PyTorch 实现 PageRank

```python
import torch

def pagerank_torch(adj: torch.Tensor, d: float = 0.85, num_iter: int = 100, tol: float = 1e-6):
    """PageRank via power iteration.

    Args:
        adj: (n, n) adjacency matrix (with 1 for edge j -> i, 0 otherwise)
        d: damping factor
        num_iter: max iterations
        tol: convergence tolerance

    Returns:
        r: (n,) PageRank scores, sum to 1
    """
    n = adj.shape[0]

    # Compute out-degree per column (sender)
    out_degree = adj.sum(dim=0)  # (n,)

    # Handle dangling nodes (out_degree == 0): distribute uniformly
    dangling = (out_degree == 0).float()  # (n,)

    # Transition matrix M: M[i,j] = 1/out_degree[j] if j -> i
    # Avoid division by zero: where dangling, set M[:,j] = 1/n
    M = adj / out_degree.clamp(min=1).unsqueeze(0)  # (n, n), broadcast
    # Replace dangling columns with uniform
    M = M * (1 - dangling).unsqueeze(0) + (dangling / n).unsqueeze(0)

    # Damping: P = d * M + (1 - d) * (1/n) * 1 * 1^T
    P = d * M + (1 - d) / n * torch.ones(n, n)

    # Power iteration
    r = torch.ones(n) / n
    for k in range(num_iter):
        r_new = P @ r
        if torch.norm(r_new - r, p=1) < tol:
            print(f"Converged at iteration {k}")
            break
        r = r_new
    return r

# ===== Test =====
# 3 web pages: A -> B, A -> C, B -> C, C -> A
adj = torch.tensor([
    [0, 0, 1],  # page A (index 0): gets link from C
    [1, 0, 0],  # page B (index 1): gets link from A
    [1, 0, 0],  # page C (index 2): gets link from A
], dtype=torch.float32)

r = pagerank_torch(adj, d=0.85)
print("PageRank:", r)
print("Sum:", r.sum().item())  # Should be ~1.0
```

**预期输出**:`r[2] > r[0] > r[1]`(因为 A 和 B 都链向 C,C 的入度最高)。

### 7.8 工业级优化

**真实网页图**(2010 年 Google 索引 ~500 亿网页)的 PageRank 计算挑战:

1. **存储**:500 亿 × 500 亿矩阵根本存不下。但 $M$ 极度稀疏(平均每个网页有 ~10 个外链)。**稀疏存储**(CSR 格式)。
2. **计算**:每步 $P \mathbf{r}$ 是稀疏矩阵-向量乘,$O(\text{nnz}(M)) \approx 5 \times 10^9$ 次乘加。但要做 ~100 次迭代。

**MapReduce 实现**:Brin & Page (1998) 用 MapReduce 把网页图分片到几千台机器,每台算部分 `P @ r`,最后汇总。**这是 MapReduce 早期最著名的工业应用之一**。

**近似算法**:对超大图,工业界常用:
- **Adaptive PageRank**(2015):动态调整 damping factor
- **Topic-Sensitive PageRank**(2001):对每个主题算一组 PageRank
- **Personalized PageRank**:对每个用户算一组(计算量爆炸,需要剪枝)

---

## 8. Graph Laplacian 与谱聚类 (300 行)

### 8.1 拉普拉斯矩阵 L = D - A

**定义**:给定图 $G = (V, E)$,$|V| = n$,邻接矩阵 $A \in \mathbb{R}^{n \times n}$,度矩阵 $D = \text{diag}(\sum_j A_{ij})$。

**图 Laplacian**:

$$\boxed{L = D - A}$$

#### 例:3 节点三角形

```text
A --- B
 \   /
   C
```

$A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}, D = 2I, L = \begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{pmatrix}$

#### 例:路径图 A-B-C-D

$A = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}, D = \text{diag}(1, 2, 2, 1)$

$L = \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}$

### 8.2 性质:L 半正定,特征值 ≥ 0

**定理**:图 Laplacian $L$ 是**实对称半正定矩阵**(Symmetric Positive Semi-Definite, SPSD)。

**证明**(半正定性):

$$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T D \mathbf{x} - \mathbf{x}^T A \mathbf{x} = \sum_i d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j$$

无向图中 $A_{ij} = A_{ji}$,$A_{ii} = 0$:

$$\mathbf{x}^T L \mathbf{x} = \sum_i d_i x_i^2 - 2 \sum_{(i,j) \in E} x_i x_j$$

其中 $d_i = \sum_{j \in \mathcal{N}(i)} 1$ = 邻居数。

**关键步骤**:

$$\sum_i d_i x_i^2 = \sum_i \sum_{j \in \mathcal{N}(i)} x_i^2 = \sum_{(i,j) \in E} (x_i^2 + x_j^2)$$

(每条边 $(i,j)$ 对两端点各贡献一次)

所以:

$$\mathbf{x}^T L \mathbf{x} = \sum_{(i,j) \in E} (x_i^2 + x_j^2 - 2 x_i x_j) = \sum_{(i,j) \in E} (x_i - x_j)^2 \ge 0$$

**$L$ 是 Laplacian 的核心二次型**:边权 = 1 时,**Rayleigh 商 = 边权差的平方和**。$\blacksquare$

**推论**(谱定理):$L$ 对称半正定,所以 $L = Q \Lambda Q^T$,$\lambda_i \ge 0$。

#### 例:三角形 $L = \begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{pmatrix}$

特征值 $\lambda_1 = 0$(对应 $\mathbf{1} = (1,1,1)^T$ ),$\lambda_2 = \lambda_3 = 3$。

验证:$\mathbf{1}^T L \mathbf{1} = 0$(常数向量是 $L$ 的零特征向量)。其他特征向量正交于 $\mathbf{1}$。

### 8.3 Fiedler value 反映图的连通性

**定理**(Fiedler 1973):$L$ 的特征值 0 的重数 = 图的连通分量数。

**Fiedler value** $\lambda_2 = L$ 的第二小特征值(对应非平凡最小特征值的"代数连通度")。

**直觉**:$\lambda_2$ 越大,把图分成两块的"代价"越高 → 图越难被切断。

#### 例:两个三角形不连通

```text
A --- B      E --- F
 \   /        \   /
   C            G
```

$L = \begin{pmatrix} L_1 & 0 \\ 0 & L_2 \end{pmatrix}$,其中 $L_1, L_2$ 都是 3×3 三角形 Laplacian。

特征值 $\lambda_1 = 0$ (重数 2),$\lambda_3 = 3$ (重数 2),$\lambda_5 = \lambda_6 = 3$。

$\lambda_2 = 0$,**Fiedler value = 0** → 图不连通。

#### 例:完整路径图 A-B-C-D

$L = \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}$

特征值:$\lambda_k = 2 - 2 \cos(\pi k / 4)$,$k = 0, 1, 2, 3$。

- $\lambda_0 = 0$(对应 $\mathbf{1}$)
- $\lambda_1 = 2 - 2 \cos(\pi/4) = 2 - \sqrt{2} \approx 0.586$(Fiedler value)
- $\lambda_2 = 2 - 2 \cos(\pi/2) = 2$
- $\lambda_3 = 2 - 2 \cos(3\pi/4) = 2 + \sqrt{2} \approx 3.414$

**观察**:Fiedler value ≈ 0.586 表示路径图"勉强连通",Fiedler 向量(对应 $\lambda_1$ 的特征向量)≈ 平稳过渡的正弦波,把图"自然地"分成两端。

### 8.4 谱聚类算法

**目标**:把图分成 $k$ 个连通块,使得"块内边权高、块间边权低"。

**NJW 谱聚类算法**(Ng, Jordan, Weiss 2001):

1. 构造相似度图(给定数据点 + KNN 或 $\epsilon$-ball)
2. 计算 Laplacian $L$
3. 谱分解:$L = Q \Lambda Q^T$
4. 取**Fiedler 向量**(第二小特征值对应)做 1D 聚类 → 2 类
   或取**前 $k$ 个最小非零特征向量**做 $k$ 维 embedding → K-means

**为什么用 Laplacian 特征向量?**

Rayleigh 商:$\frac{\mathbf{x}^T L \mathbf{x}}{\mathbf{x}^T \mathbf{x}} = \frac{\sum_{(i,j) \in E} (x_i - x_j)^2}{\mathbf{x}^T \mathbf{x}}$

最小化这个商 = 让**边连接的节点 $x_i, x_j$ 接近** = **让同一连通块的节点特征值接近**。Fiedler 向量就是这个优化的解。

**例**:把 4 个节点 {A, B, C, D} 分成 2 块 {A, B} 和 {C, D},边的连接 A-B 和 C-D,无跨边。

$L = \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}$

特征值 $\lambda = 0$(重数 2),$\lambda = 2$(重数 2)。

Fiedler 向量是任意满足 $\mathbf{v} \perp \mathbf{1}$ 的向量,**重数 2 给了 2 维子空间**。可以选 $\mathbf{v}_1 = (1, -1, 0, 0)^T$ 和 $\mathbf{v}_2 = (0, 0, 1, -1)^T$,分别区分 {A, B} 和 {C, D}。

### 8.5 案例:谱聚类在用户分群中的应用

**场景**:电商平台要把 100 万用户分成 5 个群组,做差异化营销。

**步骤**:
1. **构造相似度图**:每对用户的相似度 = Jaccard / 余弦 / 购买重叠度,取 KNN(每用户留 50 个最近邻)
2. **构造加权 Laplacian**:$L = D - W$,$W_{ij}$ = 用户 $i, j$ 的相似度(边权)
3. **谱分解**:取前 $k$ 个最小非零特征向量 → 5 维 embedding
4. **K-means** 在 5 维空间聚类

**对比 K-means**:
- K-means 假设**球形簇**,对非凸结构无能为力
- 谱聚类**对图结构鲁棒**,能处理流形、嵌套簇
- 代价:**计算 Laplacian 特征分解 $O(n^3)$**,不适合百万级用户

**工业加速**:
- **Nyström approximation**(Williams & Seeger 2001):采样子集,近似前 $k$ 个特征向量
- **Lanczos 算法**:稀疏矩阵特征分解,适合 $L$ 稀疏的场景
- **大规模 KNN 图 + 局部谱方法**(MegaTrade 等工业实现)

### 8.6 谱 GNN 预告(4-离散数学 第 6 章展开)

**谱图卷积**(Bruna et al. 2014, Defferrard et al. 2016):

图上的"卷积" = 在 Laplacian 特征向量(傅里叶基)上的乘法:

$$(\mathcal{F} f)(\lambda_k) = \langle f, \mathbf{u}_k \rangle$$

逆变换:$f = \sum_k (\mathcal{F} f)(\lambda_k) \mathbf{u}_k$。

**谱卷积核**:$g_\theta(\Lambda) = \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\Lambda})$,其中 $T_k$ 是 Chebyshev 多项式,$\tilde{\Lambda} = 2\Lambda / \lambda_{\max} - I$。

**GCN (Kipf & Welling 2017)**:切比雪夫多项式截断到 1 阶,$K = 1$,简化成

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

其中 $\tilde{A} = A + I$ 是带自环的邻接矩阵。这就是 GCN 的"层"。

**关键观察**:GCN 的传播算子 $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 与归一化 Laplacian $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$ 密切相关($\tilde{A} = A + I$,$\tilde{D} = D + I$,对应"Laplacian 加自环")。

**PinSage (Ying et al. 2018, Pinterest)**:工业级 GCN,基于随机游走 + 谱卷积,推荐 Pinterest 的 30 亿图片。

---

## 9. 特征值的数值计算 (250 行)

### 9.1 Power Iteration 求最大特征值

**算法**:

```text
输入: 矩阵 A ∈ R^{n×n}, 迭代次数 K
输出: 主特征值 λ₁ 和主特征向量 v₁

1. 随机初始化 v₀, ‖v₀‖ = 1
2. for k = 1, ..., K:
     v_k = A @ v_{k-1}
     v_k = v_k / ‖v_k‖        # 归一化
     λ_k = v_k^T A v_k         # Rayleigh 商估计特征值
3. return v_K, λ_K
```

**收敛速度**:设 $|\lambda_1| > |\lambda_2| > \ldots$,初始向量 $v_0 = \sum_i c_i \mathbf{u}_i$。迭代后

$$A^k v_0 = \sum_i c_i \lambda_i^k \mathbf{u}_i = \lambda_1^k \left( c_1 \mathbf{u}_1 + \sum_{i \ge 2} c_i \left(\frac{\lambda_i}{\lambda_1}\right)^k \mathbf{u}_i \right)$$

当 $k \to \infty$,$(\lambda_i / \lambda_1)^k \to 0$(因为 $|\lambda_i / \lambda_1| < 1$),所以 $A^k v_0 \to c_1 \lambda_1^k \mathbf{u}_1$。

**收敛率** = $|\lambda_2 / \lambda_1|$(第二大/最大特征值之比)。若 $|\lambda_2 / \lambda_1| = 0.9$,需要 ~50 次迭代才能把误差压到 $0.9^{50} \approx 0.005$。

```python
def power_iteration(A: torch.Tensor, num_iter: int = 100, tol: float = 1e-6):
    """Power iteration for dominant eigenvalue."""
    n = A.shape[0]
    v = torch.randn(n)
    v = v / torch.norm(v)

    for k in range(num_iter):
        Av = A @ v
        v_new = Av / torch.norm(Av)
        lam = v_new @ A @ v_new  # Rayleigh quotient
        if torch.norm(v_new - v) < tol:
            return lam.item(), v_new, k
        v = v_new
    return lam.item(), v, num_iter

# Test: A = [[2, 1], [1, 2]], λ_max = 3, v_max = (1,1)/√2
A = torch.tensor([[2., 1.], [1., 2.]])
lam, v, iters = power_iteration(A)
print(f"λ_max = {lam:.4f} (expected 3.0)")
print(f"v = {v} (expected ±(0.707, 0.707))")
print(f"Iterations: {iters}")
```

### 9.2 QR 算法求所有特征值

**思想**:对任意 $A$,迭代 $A_{k+1} = Q_k^T A_k Q_k$(其中 $A_k = Q_k R_k$ 是 QR 分解)。在 mild 条件下,$A_k$ 收敛到上三角(对角线是特征值)。

**算法**:

```text
1. A_0 = A
2. for k = 0, 1, 2, ...:
     Q_k, R_k = QR(A_k)         # 分解
     A_{k+1} = R_k @ Q_k         # 重组
3. 收敛后, A_K 的对角元 ≈ A 的特征值
```

**为什么收敛?** Schur 分解 $A = U T U^T$($T$ 上三角,$U$ 正交)。迭代 $A_k = U_k T U_k^T$ 其中 $U_k \to U$。**收敛率** ≈ $|\lambda_2 / \lambda_1|^k$(类似 Power Iteration 但对所有特征值)。

**工业实现**(LAPACK / SciPy `eig`):
- **Hessenberg reduction**:先把 $A$ 化简为上 Hessenberg($O(n^3)$),节省后续 QR 步骤
- **Shifted QR**:加位移加速收敛
- **Implicit QR**:避免显式形成 $Q$,数值稳定

### 9.3 反幂法求最小特征值

**核心思想**:求 $A$ 最小特征值 = 求 $A^{-1}$ 最大特征值。

**算法**:对 $A^{-1}$ 做 Power Iteration:

```text
v_{k+1} = A^{-1} v_k, 归一化
λ_k = v_k^T A v_k
```

每步需要解 $A \hat{v} = v_k$(用 LU 或 Cholesky)。**收敛速度** = $|\lambda_{n-1} / \lambda_n|$。

**应用**:求 Laplacian 的 Fiedler value(第二小特征值)。技巧:**Shifted inverse iteration**:

$$v_{k+1} = (A - \mu I)^{-1} v_k$$

调 $\mu$ 接近目标特征值,收敛更快。

### 9.4 PyTorch 实现 Power Iteration

完整的谱归一化代码(§6.1 已展示过骨架),这里给出工业级完整版本:

```python
import torch
import torch.nn as nn

class SpectralNormPowerIter(nn.Module):
    """Full Spectral Normalization layer for 2D weight matrix.

    Reference: Miyato et al. 2018 "Spectral Normalization for GANs".
    """
    def __init__(self, in_features: int, out_features: int, n_power_iter: int = 1, eps: float = 1e-12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iter = n_power_iter
        self.eps = eps

        # Weight: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # u buffer for left singular vector
        self.register_buffer('u', torch.empty(out_features).normal_())

    @property
    def W_sn(self) -> torch.Tensor:
        """Spectral-normalized weight matrix."""
        u = self.u
        with torch.no_grad():
            # Power iteration (detached for u update)
            for _ in range(self.n_power_iter):
                v = torch.nn.functional.normalize(
                    torch.mv(self.weight.t(), u), dim=0, eps=self.eps
                )
                u = torch.nn.functional.normalize(
                    torch.mv(self.weight, v), dim=0, eps=self.eps
                )
        # Save updated u
        self.u.copy_(u.detach())

        # Compute sigma (Rayleigh quotient)
        sigma = torch.dot(u, torch.mv(self.weight, v))
        return self.weight / (sigma + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) -> (batch, out_features)"""
        return torch.nn.functional.linear(x, self.W_sn)

# Test
sn_layer = SpectralNormPowerIter(512, 256)
x = torch.randn(8, 512)
y = sn_layer(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")

# Verify spectral norm ≈ 1.0
W = sn_layer.W_sn.detach()
U, S, Vh = torch.linalg.svd(W)
print(f"Spectral norm: {S.max().item():.4f} (expected ≈ 1.0)")
```

**输出**(预期):`Spectral norm: 1.0000` ✓

### 9.5 Lanczos 算法(top-k 特征值)

**动机**:神经网络 Hessian 是 $n \times n$($n$ = 参数数,LLaMA-7B 是 $7 \times 10^9$!),不可能求所有特征值。但常常只需要 top-k($k \approx 100$)。

**Lanczos 算法**:对 $A$,迭代生成 Krylov 子空间 $\mathcal{K}_k(A, v_0) = \text{span}(v_0, A v_0, A^2 v_0, \ldots, A^{k-1} v_0)$,然后做 $k \times k$ 的小特征分解。

**复杂度**:每步一次 $A v$(稀疏时 $O(\text{nnz}(A))$),$k$ 步总共 $O(k \cdot \text{nnz}(A))$。

**PyTorch 实现**(`torch.lobpcg`):

```python
import torch

# Compute top-k eigenvalues of large sparse matrix
n = 10000
A_sparse = torch.sparse_coo_tensor(
    indices=torch.randint(0, n, (2, 50000)),
    values=torch.randn(50000),
    size=(n, n)
).to_dense()
A_sym = (A_sparse + A_sparse.t()) / 2  # symmetrize

# Top-5 eigenvalues
k = 5
eigvals, eigvecs = torch.lobpcg(A_sym, k=k, largest=True)
print(f"Top-{k} eigenvalues: {eigvals}")
```

**PyTorch 的 `lobpcg`** 实现了 Locally Optimal Block Preconditioned Conjugate Gradient,是工业级 top-k 特征值求解器,被 Hessian 谱分析、谱聚类广泛使用。

### 9.6 实对称矩阵的特殊工具

**为什么 `torch.linalg.eigvalsh` 比 `eigvals` 快 10 倍?**

实对称矩阵的特征值是实数,可以用**三对角化 + QR 隐式位移**,数值稳定且快。

```python
A = torch.randn(1000, 1000)
A = (A + A.t()) / 2  # symmetrize

# General matrix: O(n^3) complex arithmetic
eigvals_general = torch.linalg.eigvals(A)

# Symmetric matrix: O(n^3 / 3) real arithmetic, more stable
eigvals_sym = torch.linalg.eigvalsh(A)  # half memory, real output

# Top-k via Lanczos (faster if k << n)
eigvals_topk = torch.lobpcg(A, k=10, largest=True)[0]
```

**实战经验**:对**实对称**矩阵,永远用 `eigvalsh` / `eigh` / `svd`(不需要 `eigvals` / `svd` 的复数版本)。前者快 2-5 倍,数值稳定性也更好。

---

## 10. PyTorch 代码实现 (250 行)

### 实验 1:手算 2×2 矩阵特征值 vs torch.linalg.eigvals

```python
import torch

# Example: A = [[3, 1], [1, 3]]
A = torch.tensor([[3., 1.], [1., 3.]])

# Hand-computed: trace = 6, det = 8, λ = (6 ± sqrt(36-32))/2 = (6±2)/2 = 4 or 2
expected_eigvals = torch.tensor([4., 2.])
expected_eigvecs = torch.tensor([[1., 1.], [1., -1.]]) / 2**0.5

# torch verification
eigvals, eigvecs = torch.linalg.eig(A)
print(f"eigvals: {eigvals.real}")  # [4, 2]
print(f"eigvecs:\n{eigvecs.real}")

# Spectral theorem: A = Q Λ Q^T
Q = eigvecs.real
Lambda = torch.diag(eigvals.real)
A_reconstructed = Q @ Lambda @ Q.t()
print(f"Reconstruction error: {(A - A_reconstructed).abs().max():.2e}")
```

### 实验 2:谱定理验证 A = QΛQ^T

```python
import torch

def verify_spectral_theorem(A: torch.Tensor) -> dict:
    """Verify A = Q Λ Q^T for symmetric A."""
    assert torch.allclose(A, A.t()), "A must be symmetric"

    eigvals, eigvecs = torch.linalg.eigh(A)
    Q = eigvecs
    Lambda = torch.diag(eigvals)
    A_rec = Q @ Lambda @ Q.t()

    return {
        'reconstruction_error': (A - A_rec).abs().max().item(),
        'orthogonality_error': (Q.t() @ Q - torch.eye(A.shape[0])).abs().max().item(),
        'eigvals_real': eigvals,
        'eigvecs_orthogonal': Q.t() @ Q,
    }

# Test 1: 3x3 symmetric
A = torch.tensor([[2., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
result = verify_spectral_theorem(A)
print(f"Reconstruction error: {result['reconstruction_error']:.2e}")
print(f"Orthogonality error: {result['orthogonality_error']:.2e}")

# Test 2: random symmetric 5x5
A_rand = torch.randn(5, 5)
A_rand = (A_rand + A_rand.t()) / 2
result = verify_spectral_theorem(A_rand)
print(f"Random 5x5 reconstruction error: {result['reconstruction_error']:.2e}")
```

### 实验 3:Spectral Normalization 实现(完整版)

见 §9.4。**关键验证**:

```python
import torch
sn = SpectralNormPowerIter(512, 256)
W = sn.W_sn.detach()
U, S, Vh = torch.linalg.svd(W)
assert abs(S.max().item() - 1.0) < 1e-3, "Spectral norm should be 1"
print(f"✓ Spectral norm = {S.max().item():.4f}")
```

### 实验 4:PageRank 在小型网页图上

见 §7.7 的 `pagerank_torch` 实现。

**进阶实验**:用 NetworkX 构造真实网络,验证 PageRank 一致性:

```python
import networkx as nx
import torch

# Create a small directed graph
G = nx.DiGraph()
edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (3, 2), (4, 0), (4, 1)]
G.add_edges_from(edges)

# NetworkX PageRank
nx_pr = nx.pagerank(G, alpha=0.85)
print("NetworkX PageRank:", {k: f"{v:.4f}" for k, v in nx_pr.items()})

# Our PageRank
n = G.number_of_nodes()
adj = torch.zeros(n, n)
for u, v in edges:
    adj[v, u] = 1  # M[v, u] = 1 if u -> v

our_pr = pagerank_torch(adj, d=0.85)
print("Our PageRank:    ", [f"{x:.4f}" for x in our_pr.tolist()])

# Verify
for i in range(n):
    diff = abs(our_pr[i].item() - nx_pr[i])
    assert diff < 1e-4, f"Mismatch at node {i}: {diff}"
print("✓ PageRank matches NetworkX")
```

### 实验 5:Spectral Clustering 在 2D 数据

```python
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def spectral_clustering(X: torch.Tensor, n_clusters: int, sigma: float = 1.0):
    """Spectral clustering on point cloud X ∈ R^{n×d}."""
    n = X.shape[0]

    # Pairwise squared distance
    dist2 = torch.cdist(X, X) ** 2
    # Gaussian similarity
    W = torch.exp(-dist2 / (2 * sigma ** 2))
    W.fill_diagonal_(0)  # remove self-loops

    # Degree matrix and Laplacian
    D = W.sum(dim=1).diag()
    L = D - W

    # Top-k smallest eigenvectors (Fiedler vectors and beyond)
    eigvals, eigvecs = torch.linalg.eigh(L)
    embedding = eigvecs[:, :n_clusters]  # (n, n_clusters)

    # K-means on embedding
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=n_clusters, n_init=10).fit(embedding.numpy()).labels_
    return labels, embedding

# Generate 3-blob data
torch.manual_seed(42)
n_per = 30
centers = torch.tensor([[0., 0.], [5., 5.], [-5., 5.]])
X = torch.cat([c + torch.randn(n_per, 2) for c in centers])

labels, emb = spectral_clustering(X, n_clusters=3, sigma=1.5)
print(f"Cluster sizes: {[labels.tolist().count(i) for i in range(3)]}")
print(f"Expected: ~{n_per} per cluster")
```

**可视化**(运行后):

```text
 真实聚类:                  谱聚类结果:
   ●●●  ○○○  ▲▲▲              ●●●  ▲▲▲  ○○○
   ●●●  ○○○  ▲▲▲              ●●●  ▲▲▲  ○○○
   ●●●  ○○○  ▲▲▲              ●●●  ▲▲▲  ○○○
   ●●●  ○○○  ▲▲▲              ●●●  ▲▲▲  ○○○
```

**对比 K-means**:对**凸球形**簇,K-means 也能搞定。但对**月牙形、环形**簇,K-means 失败,谱聚类成功。

### 实验 6:Graph Laplacian Fiedler vector

```python
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Build a 2-block stochastic block model graph
n = 100
p_in = 0.3   # in-block edge probability
p_out = 0.02 # cross-block edge probability

A = torch.zeros(n, n)
for i in range(n):
    for j in range(i+1, n):
        same_block = (i < n//2) == (j < n//2)
        p = p_in if same_block else p_out
        if torch.rand(1).item() < p:
            A[i, j] = A[j, i] = 1

D = A.sum(dim=1).diag()
L = D - A

# Compute Fiedler vector (second smallest eigenvalue)
eigvals, eigvecs = torch.linalg.eigh(L)
fiedler = eigvecs[:, 1]
print(f"λ_1 = 0, λ_2 (Fiedler) = {eigvals[1]:.4f}")
print(f"Fiedler vector range: [{fiedler.min():.3f}, {fiedler.max():.3f}]")
print(f"Block 1 mean: {fiedler[:n//2].mean():.3f}")
print(f"Block 2 mean: {fiedler[n//2:].mean():.3f}")
# Expect block 1 ≈ positive, block 2 ≈ negative

# Cluster based on Fiedler sign
labels = (fiedler > 0).long()
accuracy = max(
    (labels[:n//2] == 0).float().mean() + (labels[n//2:] == 1).float().mean(),
    (labels[:n//2] == 1).float().mean() + (labels[n//2:] == 0).float().mean(),
) / 2
print(f"Clustering accuracy: {accuracy:.3f}")
```

**输出预期**:Fiedler value ≈ 0.1-0.5(取决于 $p_{\text{out}}$),聚类准确率 > 0.9。

### 实验 7:Hessian 特征值分析(简化)

```python
import torch

# Simulate a 2D loss landscape with controllable Hessian
def loss(theta, H, target):
    diff = theta - target
    return 0.5 * diff @ H @ diff

# Anisotropic Hessian
H = torch.tensor([[4., 0.], [0., 1.]])  # κ = 4
target = torch.tensor([1., 1.])
theta = torch.tensor([0., 0.])

# Compute Hessian analytically
H_numerical = torch.zeros(2, 2)
eps = 1e-4
for i in range(2):
    for j in range(2):
        e_i, e_j = torch.zeros(2), torch.zeros(2)
        e_i[i] = eps
        e_j[j] = eps
        H_numerical[i, j] = (
            loss(theta + e_i + e_j, H, target)
            - loss(theta + e_i, H, target)
            - loss(theta + e_j, H, target)
            + loss(theta, H, target)
        ) / eps**2

print(f"Analytical Hessian:\n{H}")
print(f"Numerical Hessian:\n{H_numerical}")
print(f"Max error: {(H - H_numerical).abs().max():.2e}")

# Eigenvalues
eigvals, eigvecs = torch.linalg.eigh(H)
print(f"Hessian eigenvalues: {eigvals}")
print(f"Condition number κ = {eigvals.max() / eigvals.min():.2f}")
```

---

## 11. 常见误区 (100 行, 5-8 条)

### 误区 1:"每个矩阵都有实特征值"

**错**。旋转矩阵 $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$(90° 旋转)只有复特征值 $\pm i$。**实特征值的几何意义是"方向不变",旋转后方向全变了**。

**判别方法**:
- 实对称矩阵 → 特征值实(谱定理,§4.1)
- 一般实矩阵 → 特征值**可能复**(共轭对出现)
- 复矩阵 → 特征值必复,但代数重数 = 几何重数仍成立

### 误区 2:"特征向量唯一"

**错**。$A\mathbf{v} = \lambda\mathbf{v}$ 等价于 $A(c\mathbf{v}) = \lambda(c\mathbf{v})$,**任何非零倍数 $c\mathbf{v}$ 都是特征向量**。我们通常归一化 $\|\mathbf{v}\| = 1$。

更深的非唯一性:**重特征值时,特征空间是子空间,基向量选取不唯一**(§2.5)。例:$A = 2I$ 的特征空间 = $\mathbb{R}^n$,任意正交基都是"特征向量基"。

### 误区 3:"谱半径 < 1 → 矩阵收敛到 0"

**部分对**。**严格不等式** $\rho(A) < 1$ 保证对任何矩阵范数 $\|A^k\| \to 0$。但若 $\rho(A) = 1$(等于),某些范数 $\|A^k\|$ 可能不趋于 0,只保持有界。

**反例**:$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$,$\rho(A) = 1$,但 $A^k = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$,$\|A^k\|_\infty = k \to \infty$。

**结论**:**Jordan 块的尺寸决定收敛速度**。若 $\rho(A) = 1$,需要看 $A$ 是否有非平凡 Jordan 块,若有,$A^k$ 不收敛。

### 误区 4:"实对称矩阵特征向量必正交"

**部分对**。**不同特征值的特征向量必正交**(§4.3 推论 2)。**同一特征值(重数 > 1)的特征向量不一定正交**,但**可以选一组正交基**(用 Gram-Schmidt)。

**例子**:$A = I_3$,$\lambda = 1$ 三重,特征空间 = $\mathbb{R}^3$。$\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ 正交,但 $\mathbf{v}_1 = (1,0,0), \mathbf{v}_2 = (1,1,0), \mathbf{v}_3 = (1,1,1)$ 也是合法"特征向量",**不正交**。

### 误区 5:"谱范数 = 谱半径"

**错**(对一般矩阵)。谱范数 $\|A\|_2 = \sigma_{\max}(A)$,谱半径 $\rho(A) = \max |\lambda_i|$。**对实对称矩阵它们相等**(因为 $A = A^T$ 时 $\sigma_i = |\lambda_i|$)。

**反例**:非对称矩阵 $A = \begin{pmatrix} 1 & 100 \\ 0 & 1 \end{pmatrix}$。特征值 $\lambda = 1$(重),$\rho(A) = 1$。但 $\sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)} = \sqrt{\lambda_{\max}\begin{pmatrix} 1 & 100 \\ 100 & 10001 \end{pmatrix}} \approx 100$。**谱范数是谱半径的 100 倍**。

### 误区 6:"PageRank 就是 Google 用的算法"

**部分对**。现代 Google 搜索用 **PageRank + 数百个其他信号**(内容质量、用户行为、反垃圾、HTTPS、移动友好……)。原始 PageRank(Brin & Page 1998)是早期重要信号,但 2025 年的排名算法是**机器学习 + 大量人工调整**的混合体。

### 误区 7:"Spectral Normalization 让 GAN 一定稳定"

**部分对**。SN-GAN 解决了 Lipschitz 爆炸问题,但 GAN 训练还有**模式崩溃、数据不平衡、优化器选择**等问题。Spectral Norm 是必要不充分条件。

### 误区 8:"谱聚类总比 K-means 好"

**错**。对**球形簇、特征数低**的数据,K-means 更快更准。谱聚类的优势只在**非凸结构(环形、月牙)**和**图数据**上。代价是 $O(n^3)$ 特征分解。

---

## 12. 面试 Q&A (80 行, 3-5 题 L7/L8)

### Q1:特征值/特征向量的几何意义?

**答**:矩阵 $A$ 是一个线性变换。特征向量 $\mathbf{v}$ 是这个变换的"主轴":$A\mathbf{v} = \lambda\mathbf{v}$ 意味着变换前后方向不变,只被缩放 $\lambda$ 倍。$|\lambda| > 1$ 拉伸,$|\lambda| < 1$ 压缩,$\lambda < 0$ 反向,$\lambda$ 复数对应旋转。

**加分项**:联系到深度学习——特征值是网络"最敏感方向"上的放大倍数,谱半径控制 RNN 梯度行为,谱范数控制 GAN 判别器的 Lipschitz。

### Q2:为什么 Spectral Normalization 让 GAN 训练稳定?

**答**:SN 把判别器每层 $W$ 除以 $\sigma_{\max}(W)$,强制每层 $\|W\|_2 = 1$,整判别器 $D$ 的 Lipschitz 常数 ≤ 1。这带来三个好处:
1. Wasserstein 距离有意义(Kantorovich-Rubinstein 对偶)
2. 判别器梯度有界 $\|\nabla D\| \le 1$,**不会梯度爆炸**
3. 生成器 $G$ 的梯度 $\|\nabla_G D(G(\mathbf{z}))\| \le 1$,训练稳定

**实现技巧**:不用显式 SVD,幂迭代 1 次就够。

### Q3:PageRank 的数学原理?

**答**:网页图 = Markov 链。PageRank 向量 $\mathbf{r}$ 满足 $\mathbf{r} = M\mathbf{r}$,即 $M$ 的特征值 1 对应的特征向量。迭代 $\mathbf{r}_{k+1} = M \mathbf{r}_k$ 收敛到稳态,收敛速度 = $|\lambda_2|$(第二大特征值的模)。

**工程**:为保证收敛,加 damping factor $d = 0.85$:**$P = d M + (1-d) \mathbf{1}\mathbf{1}^T / n$**。这保证 $P$ 不可约非周期。

### Q4:Hessian 特征值与收敛速度的关系?

**答**:Newton 法收敛速度 = $\rho(I - H^{-1} H_0)$,其中 $H_0$ 是初始 Hessian。**收敛速率被 Hessian 最缓方向控制**:

$$\text{收敛速度} \approx \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}$$

**条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$**:越大,病态越严重,梯度下降越慢(被 $\lambda_{\min}$ 拖累)。**Adam/RMSProp 用对角缩放近似 Hessian 倒数,缓解病态**。

**深度学习**:Shampoo、Sophia、K-FAC 都用 Hessian 特征值信息做自适应预处理。

### Q5:谱聚类 vs K-means 的本质区别?

**答**:
- **K-means**:假设簇是**凸球形**,用欧氏距离,优化簇内方差。$O(ndkT)$。
- **谱聚类**:用**图 Laplacian 的特征向量**做 embedding(把"图结构"映射到欧氏空间),然后 K-means。**能处理非凸结构**(环形、月牙)。

**本质区别**:K-means 在**原始空间**聚类,谱聚类在**特征空间**聚类。谱聚类通过 Laplacian 特征向量**隐式做流形学习**。

**代价**:谱聚类 $O(n^3)$(特征分解),不适合百万级。Nyström 近似可降为 $O(n^2)$。

---

## 13. 与其他数学工具的联系 (50 行)

### 跨章引用图谱

```text
特征值 / 谱定理 / 谱归一化
       │
       ├──→ SVD (第 3 章 §6)
       │     A^T A 的特征值 = σ²
       │     A = U Σ V^T, 谱定理 = 对称 A 的特例
       │
       ├──→ 二次型 / 正定 (第 5 章)
       │     实对称矩阵的特征值全实 (谱定理)
       │     Hessian PSD = 所有特征值 ≥ 0 = 凸函数
       │
       ├──→ 矩阵微积分 / Hessian (第 6 章, 1-高数 第 3 章)
       │     Hessian 特征值 = 损失 landscape 各方向曲率
       │     谱归一化 = 控制每层 Hessian 的最大特征值
       │
       ├──→ 测度集中 / 高维 (第 8 章)
       │     随机矩阵特征值集中在某个区间 (Marchenko-Pastur)
       │     谱方法的核心工具
       │
       ├──→ 图论 / 离散数学 (4-离散数学 第 4 章)
       │     Graph Laplacian L = D - A 的特征值
       │     Fiedler value = 图的代数连通度
       │
       ├──→ GNN / 谱图卷积 (4-离散数学 第 6 章)
       │     GCN 传播算子 ∝ 归一化 Laplacian
       │     谱卷积 = 在 Laplacian 特征向量(傅里叶基)上乘法
       │
       └──→ 最优化 / 凸性 (6-最优化)
             Hessian PSD = 凸
             谱归一化 = 保证优化 landscape 平滑
```

### 4 个关键联系

1. **特征值 → SVD(第 3 章)**:$A^T A$ 的特征值 = $A$ 奇异值的平方。谱定理 = 实对称矩阵的 SVD。
2. **谱方法 → 高维空间(第 8 章)**:随机矩阵的特征值分布(Marchenko-Pastur 律)在高维统计和机器学习里是核心工具。
3. **Graph Laplacian → GNN(4-离散数学 第 4 章和第 6 章)**:Laplacian 的谱分解定义了"图上的傅里叶变换",GCN 是其在神经网络里的具体实现。
4. **谱定理 → 二次型(第 5 章)**:谱定理保证实对称二次型可正交对角化,这是正定性、Rayleigh 商、Lanczos 算法的基础。

---

## 附录 A:一页纸速查表

### A.1 关键定义

| 名称 | 定义 | 几何意义 |
| :--- | :--- | :--- |
| 特征值 $\lambda$ | $A\mathbf{v} = \lambda\mathbf{v}$ | 沿 $\mathbf{v}$ 方向的缩放倍数 |
| 特征向量 $\mathbf{v}$ | 对应特征值的非零向量 | "主轴"方向 |
| 谱半径 $\rho(A)$ | $\max_i \|\lambda_i\|$ | 谱半径控制幂行为 |
| 谱范数 $\|A\|_2$ | $\sigma_{\max}(A)$ | Lipschitz 常数 |
| 条件数 $\kappa(A)$ | $\sigma_{\max} / \sigma_{\min}$ | 数值稳定性 |

### A.2 5 个核心定理

1. **谱定理**:实对称 $A$ 必正交对角化 $A = Q\Lambda Q^T$,$\lambda_i$ 全实,$\mathbf{v}_i$ 两两正交。
2. **Cayley-Hamilton**:$p(A) = 0$(特征多项式代入矩阵为零)。
3. **Perron-Frobenius**:非负不可约矩阵有唯一正主特征向量。
4. **Gelfand 公式**:$\rho(A) = \lim_{k\to\infty} \|A^k\|^{1/k}$。
5. **Weyl 不等式**(扰动理论):$A, \tilde A$ 扰动 $\|\tilde A - A\| \le \epsilon$,则 $|\tilde\lambda_i - \lambda_i| \le \epsilon$。

### A.3 常用 PyTorch 函数

```python
torch.linalg.eigvals(A)        # 所有特征值(复数)
torch.linalg.eigvalsh(A)       # 实对称 A 的特征值(实数,快)
torch.linalg.eig(A)            # 特征值 + 特征向量
torch.linalg.eigh(A)           # 实对称版
torch.linalg.svd(A)            # SVD
torch.linalg.norm(A, ord=2)   # 谱范数
torch.lobpcg(A, k=k)           # top-k 特征值(Lanczos)
torch.nn.utils.spectral_norm    # SN-GAN 标配
```

### A.4 一句话总结

> **谱定理**告诉你:任何实对称矩阵 = 旋转 + 缩放 + 反旋转。这是 L7/L8 面试的"标准答案",能展开到 SVD、PCA、Hessian、GAN、PageRank、谱聚类、GNN。

---

## 附录 B:参考资料

### 论文
- **Miyato et al. 2018**: Spectral Normalization for GANs (ICLR 2018)
- **Page & Brin 1998**: The PageRank Citation Ranking
- **Brin & Page 1998**: The Anatomy of a Large-Scale Hypertextual Web Search Engine
- **Kipf & Welling 2017**: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
- **Defferrard et al. 2016**: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS 2016)
- **Ng, Jordan, Weiss 2001**: On Spectral Clustering: Analysis and an Algorithm (NeurIPS 2001)
- **Jacot et al. 2018**: Neural Tangent Kernel: Convergence and Generalization in Neural Networks (NeurIPS 2018)

### 教材
- **Strang - Linear Algebra and Its Applications**:谱定理的最经典讲解
- **Horn & Johnson - Matrix Analysis**:特征值扰动理论的参考书
- **Golub & Van Loan - Matrix Computations**:QR 算法、Lanczos、幂迭代的工业实现细节

### 代码
- **PyTorch docs**: `torch.linalg.eigvalsh`, `torch.nn.utils.spectral_norm`, `torch.lobpcg`
- **NetworkX**: `nx.pagerank` 参考实现
- **scikit-learn**: `SpectralClustering`

---

**本章完**。共 13 节 + 2 个附录,涵盖特征值的直觉、严格定义、谱定理、谱半径、谱范数、DL 应用(SN-GAN、Hessian、NTK)、PageRank、Graph Laplacian 与谱聚类、数值计算方法、PyTorch 实现、误区、面试 Q&A、跨章联系。配合第 1-3 章,完成线代基础的前 4 章覆盖。

---

## 14. 工程师常见误区与踩坑地图

### 坑 1: "特征向量顺序不一致"

**症状**: 两次跑 `torch.linalg.eigh(A)` 拿到的特征值排序不同,导致代码结果不一致。

**根因**: 数值算法的内部实现可能改变迭代顺序,即使数学上唯一,数值上不一定。

**解药**:
- 拿到后**手动排序**: `idx = eigvals.argsort(descending=True); eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]`。
- 不要假设默认排序,尤其在跨平台/跨版本时。

### 坑 2: "复数特征值的处理"

**症状**: 跑非对称矩阵的特征分解,出现 `complex128` dtype,后续代码莫名报错。

**根因**: 非对称矩阵的特征值可能是复数,即使矩阵是实数。

**解药**:
- 对称矩阵用 `torch.linalg.eigvalsh` / `eigh`(返回实数)。
- 非对称用 `eigvals` / `eig`(可能复数)。
- 如果矩阵**应该**对称但出现复数,**说明有数值误差**,加正则或检查。

### 坑 3: "幂迭代不收敛 / 收敛到错特征向量"

**症状**: 自己实现幂迭代,迭代 1000 次还在震荡。

**根因**:
- 矩阵特征值谱半径 = 1 或 < 1 时,迭代步长太小。
- 矩阵有重特征值($\lambda_1 = \lambda_2$)。
- 起始向量正好与主特征向量正交(罕见但可能)。

**解药**:
- 用 Rayleigh 商迭代(更稳定)。
- 用 PyTorch 的 `lobpcg`(大规模稀疏矩阵)。
- 检查矩阵是否对称(对称才能用 Lanczos)。

### 坑 4: "PageRank 阻尼因子选错"

**症状**: PageRank 跑出来的排序"头重脚轻",少数节点垄断排名。

**根因**:
- 阻尼因子 $d$ 太小:过度依赖 teleport 概率。
- 阻尼因子太大:算法不收敛 / 死循环。
- 默认 $d = 0.85$ 是经验值,适合多数图。

**解药**:
- 用 $d \in [0.8, 0.9]$ 之间。
- 检查图的**连通性**(有向强连通 + 非周期性 = 收敛保证)。
- 工业实现:NetworkX / igraph 的 pagerank 默认参数够用。

### 坑 5: "GNN 中 Laplacian 归一化矩阵算错"

**症状**: 实现 GCN,效果远差于论文报告。

**根因**:
- 用了非对称归一化 $D^{-1} A$,应该用对称归一化 $D^{-1/2} A D^{-1/2}$。
- 加了自环 $\tilde{A} = A + I$ 但忘了同步修改 degree。

**正确公式** (Kipf 2017):

$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}, \quad \tilde{A} = A + I, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$

### 坑 6: "谱归一化算子范数用 Frob 范数代替"

**症状**: 实现 Spectral Norm GAN,但效果跟论文差很多。

**根因**: 用了 `torch.norm(W)` (Frob 范数),不是谱范数 $\sigma_{\max}(W)$。

**解药**:
- 用 `torch.nn.utils.spectral_norm`(Power iteration 实现)。
- 或手动 `W_max = torch.linalg.svdvals(W)[0]`(贵但准)。
- 永远不要混用 Frob / Spectral。

### 坑 7: "谱聚类 K 选错"

**症状**: 跑 SpectralClustering,聚类结果一团糟。

**根因**: 选错了"用哪个特征向量"。

**解药**:
- **第一种方法**: 用前 K 个**最小**特征值对应的特征向量(对应 Laplacian, Fiedler 思想)。
- **第二种方法 (Ng-Jordan-Weiss)**: 用前 K 个**最大**特征向量,做 K-means。
- **第三种方法 (Shi-Malik)**: 用第二小特征向量(Ncut)。
- 工业首选 **Ng-Jordan-Weiss**,K 用肘部法则选。

### 坑 8: "Graph Laplacian 的特征值全 0"

**症状**: 不连通图跑谱聚类,几个特征值 ≈ 0,聚类失败。

**根因**: Laplacian 的 0 特征值数量 = 图的连通分量数。**不连通图的谱聚类需要先识别连通分量**。

**解药**:
- 检查图的连通性:`nx.is_connected(G)`。
- 不连通时,对每个分量独立谱聚类,或加正则 `L + \epsilon I`。

### 坑 9: "谱半径 $\rho(A) < 1$ 不一定收敛"

**症状**: 写了迭代 $\mathbf{v}_{k+1} = A\mathbf{v}_k$,谱半径 0.9,但收敛很慢。

**根因**: 收敛速度 ≈ $(\rho_2/\rho_1)^k$,如果 $\rho_2 \approx \rho_1$,收敛慢。

**解药**:
- 加速技巧:**Chebyshev 多项式加速**或**Lanczos 迭代**。
- 或者直接用 PyTorch 的 `lobpcg`。

### 坑 10: "特征分解 vs SVD 选错"

**症状**: 对**非对称**矩阵用 `torch.linalg.eigh`,报错。

**根因**:
- `eigh` / `eigvalsh`:**只对 Hermitian (对称实 / 共轭对称复) 矩阵**。
- `eig` / `eigvals`:**对任意方阵**,但返回可能是复数。
- SVD (`:svd` / `svdvals`):**对任意矩阵**,返回实数奇异值。

**解药**:
| 矩阵类型 | 推荐函数 |
| :--- | :--- |
| 对称实 / Hermitian 复 | `eigh` / `eigvalsh` |
| 一般方阵 | `eig` / `eigvals` |
| 一般 m×n | `svd` / `svdvals` |

---

## 15. 完整案例: PageRank + 谱聚类 + 谱归一化全链路

### 15.1 PageRank: 8 个网页的小例子

```python
import numpy as np
import torch

# 8 个网页的有向图(邻接矩阵)
A = torch.tensor([
    [0, 1, 1, 0, 0, 0, 0, 0],  # 0 → 1, 2
    [0, 0, 1, 0, 0, 0, 0, 0],  # 1 → 2
    [0, 0, 0, 1, 0, 0, 0, 0],  # 2 → 3
    [0, 0, 0, 0, 1, 0, 0, 0],  # 3 → 4
    [0, 0, 0, 0, 0, 1, 0, 0],  # 4 → 5
    [0, 0, 0, 0, 0, 0, 1, 0],  # 5 → 6
    [0, 0, 0, 0, 0, 0, 0, 1],  # 6 → 7
    [0, 0, 0, 0, 0, 0, 1, 0],  # 7 → 6 (循环)
], dtype=torch.float32)

n = A.shape[0]
damping = 0.85

# 构建转移概率矩阵 (随机游走)
# M[i,j] = 从 j 跳到 i 的概率(列随机)
out_degree = A.sum(dim=0)
M = A / out_degree  # 列随机
M = torch.nan_to_num(M, nan=0.0)

# Google 矩阵: G = d * M + (1-d) * (1/n) * 11^T
G = damping * M + (1 - damping) / n * torch.ones(n, n)

# 幂迭代求主特征向量
v = torch.ones(n) / n
for _ in range(100):
    v = G @ v
    v = v / v.sum()

print("PageRank 排序:", v.numpy().round(4))
# 节点 6, 7 通常有较高 PR(因为它们互相链接,形成 trap)
# 加 damping 后,所有节点都有非零 PR
```

### 15.2 谱聚类: 把 2 个圆环分开

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering

# 生成两个同心圆(传统 K-means 会失败)
X, y = make_circles(n_samples=500, noise=0.05, factor=0.5)

# 谱聚类
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10)
y_pred = sc.fit_predict(X)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
axes[0].set_title("真实标签")
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
axes[1].set_title("谱聚类结果")
plt.savefig("spectral_clustering.png", dpi=100)

# 谱聚类背后的数学:
# 1. 算相似度矩阵 W (RBF kernel 或 kNN)
# 2. 算 Laplacian L = D - W
# 3. 取 L 的前 K 个最小特征向量(归一化后做 K-means)
```

### 15.3 谱归一化: SN-GAN 的 SpectralNorm 层

```python
import torch
import torch.nn as nn

# PyTorch 自带
sn_linear = nn.utils.spectral_norm(nn.Linear(64, 128))
# 内部用幂迭代估计 σ_max(W),实时更新 u 向量

# 手写实现(教学版)
class SpectralNormManual(nn.Module):
    def __init__(self, in_features, out_features, n_iter=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.u = nn.Parameter(torch.randn(in_features), requires_grad=False)
        self.n_iter = n_iter

    def forward(self, x):
        # 幂迭代估计 σ_max
        u = self.u
        for _ in range(self.n_iter):
            v = self.W.T @ u
            v = v / v.norm()
            u = self.W @ v
            u = u / u.norm()
        self.u.data = u

        sigma = u @ self.W @ v
        return x @ (self.W / sigma).T  # 用归一化后的 W

# 用法
layer = SpectralNormManual(64, 128)
x = torch.randn(32, 64)
y = layer(x)
```

---

## 16. 几何直觉强化: 特征向量的"方向图"

### 16.1 旋转矩阵的特征值与特征向量

```text
R = [[cos θ, -sin θ], [sin θ, cos θ]] (旋转矩阵)

特征值: λ = cos θ ± i sin θ (复数!)
特征向量: 复数向量

几何: 旋转没有"不动方向"(除了 θ=0, π)
      → 实数特征向量不存在 (除退化情况)
```

### 16.2 对称矩阵的特征向量: 两两正交

```text
A = [[2, 1], [1, 2]] (对称)

特征值: λ₁ = 3, λ₂ = 1
特征向量: v₁ = (1, 1)/√2, v₂ = (1, -1)/√2

v₁ · v₂ = 1/2 - 1/2 = 0 ✓ 正交

几何: 对称矩阵 = "沿两个正交方向缩放"

     y
     ↑  v₁
     │ ╱
     │╱     ← 沿 v₁ 拉伸 3 倍
     ●────→ x
      ╲
       ╲ v₂
        ↓ 沿 v₂ 拉伸 1 倍
```

### 16.3 谱半径的几何意义

```text
A 的谱半径 ρ(A) = max |λ|

几何: ρ(A) = A 对任意单位向量的最大"伸长倍数"

max‖x‖=1 ‖Ax‖ = ρ(A)
```

### 16.4 Graph Laplacian 的特征向量 = "图的傅里叶基"

```text
L 的特征向量 = "图上的振荡模式"

最小特征向量(常数): 全图均匀
第二小特征向量(Fiedler): 图的二分模式
大特征向量: 高频振荡
```

---

## 17. 动手算一算(带完整答案)

### 17.1 基础题

**题 1**: 求 $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ 的特征值。

<details>
<summary>答案</summary>

三角矩阵的特征值 = 对角元:$\lambda_1 = 3, \lambda_2 = 2$。
</details>

**题 2**: 求 $A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$ 的特征值与特征向量。

<details>
<summary>答案</summary>

$\det(A - \lambda I) = (1-\lambda)^2 - 1 = 0$
$\lambda^2 - 2\lambda = 0$
$\lambda(\lambda - 2) = 0$
$\lambda_1 = 2, \lambda_2 = 0$

特征向量:
- $\lambda_1 = 2$: $(A - 2I)v = 0 \Rightarrow v = (1, 1)^\top / \sqrt{2}$
- $\lambda_2 = 0$: $(A - 0I)v = 0 \Rightarrow v = (1, -1)^\top / \sqrt{2}$
</details>

**题 3**: 求 $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ 的特征值。

<details>
<summary>答案</summary>

$\det(A - \lambda I) = \lambda^2 + 1 = 0$
$\lambda = \pm i$ (复数)

这是旋转矩阵 $R(90°)$,**没有实特征向量**。
</details>

### 17.2 中级题

**题 4**: 验证 PageRank 矩阵 $G$ 的谱半径 = 1,主特征向量 = 稳态分布。

<details>
<summary>答案</summary>

```python
import torch

n = 5
# 任意随机 Google 矩阵(列随机 + 阻尼)
M = torch.rand(n, n)
M = M / M.sum(dim=0, keepdim=True)
d = 0.85
G = d * M + (1 - d) / n * torch.ones(n, n)

# 特征分解
eigvals, eigvecs = torch.linalg.eig(G)
print(f"最大特征值模: {eigvals.abs().max():.4f}")  # 应为 1

# 主特征向量(对应特征值 1)
v = eigvecs[:, eigvals.argmax()].real
v = v / v.sum()
print("稳态分布:", v.numpy())
```
</details>

**题 5**: 用 PyTorch 验证谱定理:对实对称 $A$, $A = Q\Lambda Q^\top$。

<details>
<summary>答案</summary>

```python
import torch

# 随机对称矩阵
n = 5
A = torch.randn(n, n)
A = (A + A.T) / 2  # 对称化

# 谱分解
eigvals, Q = torch.linalg.eigh(A)
Lambda = torch.diag(eigvals)

# 验证
A_reconstructed = Q @ Lambda @ Q.T
print(f"重建误差: {(A - A_reconstructed).abs().max():.6f}")  # ~ 1e-6
print(f"正交性: {(Q.T @ Q - torch.eye(n)).abs().max():.6f}")  # ~ 1e-6
```
</details>

**题 6**: 用幂迭代估计矩阵的最大特征值和特征向量。

<details>
<summary>答案</summary>

```python
import torch

torch.manual_seed(0)
A = torch.randn(20, 20)
# 让 A 对称化,以保证实特征值
A = (A + A.T) / 2

n_iter = 100
v = torch.randn(20)
v = v / v.norm()

for i in range(n_iter):
    v_new = A @ v
    v_new = v_new / v_new.norm()
    # Rayleigh 商估计 λ
    eigval = v_new @ A @ v_new
    if i % 20 == 0:
        print(f"Iter {i}: λ_estimate = {eigval:.4f}")
    v = v_new

# 对比真实最大特征值
eigvals = torch.linalg.eigvalsh(A)
print(f"真实最大特征值: {eigvals.max():.4f}")
```
</details>

### 17.3 高级题(ML/DL 联系)

**题 7**: 实现 Lanczos 算法(三对角化)估计 top-k 特征值。

<details>
<summary>答案</summary>

```python
import torch

def lanczos(A, k=10, n_iter=None):
    """Lanczos 算法:把 A 三对角化为 T = Q^T A Q, T 是 k×k

    Returns: 近似 top-k 特征值 (从 T 的特征分解得到)"""
    n = A.shape[0]
    n_iter = n_iter or min(k, n)

    Q = torch.zeros(n, n_iter)
    alpha = torch.zeros(n_iter)
    beta = torch.zeros(n_iter - 1)

    q = torch.randn(n)
    q = q / q.norm()
    Q[:, 0] = q

    for j in range(n_iter):
        if j > 0:
            q = A @ Q[:, j-1] - beta[j-1] * Q[:, j-2]
        else:
            q = A @ Q[:, 0]

        alpha[j] = q @ Q[:, j]
        q = q - alpha[j] * Q[:, j]

        if j < n_iter - 1:
            beta[j] = q.norm()
            if beta[j] < 1e-12:
                # 提前终止:找到不变子空间
                break
            Q[:, j+1] = q / beta[j]

    # 构造三对角矩阵 T
    T = torch.diag(alpha[:j+1])
    if j > 0:
        T += torch.diag(beta[:j], diagonal=1)
        T += torch.diag(beta[:j], diagonal=-1)

    # T 的特征值 = A 的近似特征值
    eigvals = torch.linalg.eigvalsh(T)
    return eigvals

# 用法: 大型稀疏矩阵的 top-k 特征值
A = torch.randn(1000, 1000)
A = (A + A.T) / 2
top_k = lanczos(A, k=10)
print(f"Top-10 估计: {top_k[:10].numpy()}")
print(f"真实 top-10: {torch.linalg.eigvalsh(A)[-10:].numpy()}")
```
</details>

**题 8**: 实现谱归一化权重(SN-GAN)。

<details>
<summary>答案</summary>

```python
import torch
import torch.nn as nn

# PyTorch 一行
sn_linear = nn.utils.spectral_norm(nn.Linear(64, 128))

# 验证: forward 后,权重范数 <= 1
x = torch.randn(32, 64)
y = sn_linear(x)

# 取出内部 W
W = sn_linear.weight
print(f"W 谱范数: {torch.linalg.svdvals(W)[0]:.4f}")  # 应该 ≤ 1+ε
```
</details>

**题 9**: 用 Laplacian 找图的"桥梁节点"(Fiedler 向量)。

<details>
<summary>答案</summary>

```python
import torch
import networkx as nx

# 一个有两个 cluster 的图(中间 1 个桥梁节点)
G = nx.Graph()
G.add_edges_from([(0,1), (1,2), (2,3), (3,4),  # 簇 1
                   (5,6), (6,7), (7,8), (8,9),  # 簇 2
                   (4, 5)])  # 桥梁

# 邻接矩阵
A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
D = torch.diag(A.sum(dim=1))
L = D - A  # Laplacian

# 谱分解
eigvals, eigvecs = torch.linalg.eigh(L)
print(f"最小 3 个特征值: {eigvals[:3]}")

# Fiedler 向量 = 第二小特征值对应的特征向量
fiedler = eigvecs[:, 1]
print(f"Fiedler 向量: {fiedler.numpy().round(3)}")
# 桥梁节点 4, 5 的 Fiedler 值 ≈ 0(连接两个 cluster)

# 用 Fiedler 向量二分图
labels = (fiedler > 0).int()
print(f"分组: {labels.numpy()}")
```
</details>

**题 10**: 实现 PageRank 的"主题敏感"版本(Personalized PageRank)。

<details>
<summary>答案</summary>

```python
import torch

n = 100
A = (torch.rand(n, n) > 0.95).float()  # 稀疏图
A = torch.maximum(A, A.T)  # 无向化

# 转移概率矩阵(列随机)
out_degree = A.sum(dim=0)
M = A / out_degree.clamp(min=1)
M = torch.nan_to_num(M, nan=0.0)

damping = 0.85

# Personalized PageRank: teleport 分布 = 个性化向量
personalized = torch.zeros(n)
personalized[0] = 1.0  # 从节点 0 出发

v = personalized.clone()
for _ in range(100):
    v = damping * M @ v + (1 - damping) * personalized
    v = v / v.sum()

print(f"从节点 0 出发的 PageRank: {v[:10].numpy().round(4)}")
```
</details>

---

## 18. 跨章关联与延伸阅读

### 18.1 与第 2 章的接口

- 矩阵 $A$ 的**特征值 = $A$ 在特征向量方向的"缩放倍数"**(对应 `../../../方向教程/rec教程/第02章-深度学习数学与理论基础.md` 矩阵作为线性变换)。
- 特征向量 = 矩阵的"主方向"(SVD 截断 = 取最大特征方向)。
- 谱半径 $\rho(A)$ 决定矩阵幂 $A^k$ 的增长。

### 18.2 与第 3 章的接口

- SVD: $A^\top A$ 的特征值 = 奇异值的平方(详见 `../../../方向教程/rec教程/第03章-Embedding与数据组织.md`)。
- 谱定理 = "对称矩阵的 SVD"。
- 截断 SVD = 保留最大特征方向。

### 18.3 与第 5 章的接口

- **正定矩阵 ↔ 所有特征值 > 0**(详见 `../../../方向教程/rec教程/第05章-DLRM到GR范式迁移.md`)。
- **谱半径 < 1 ↔ 收敛**。
- **对称正定 (SPD) = 谱定理 + 正特征值**。

### 18.4 与第 6 章的接口

- **Hessian 的特征值 = 损失 landscape 各方向曲率**(详见 `../../../方向教程/rec教程/第06章-Semantic-ID全景.md`)。
- 谱归一化 = 控制 Hessian 最大特征值。
- NTK 理论 = 无限宽网络的特征值谱分析。

### 18.5 与第 7 章(张量)的接口

- **张量的"特征分解"**(Tucker decomposition, CP decomposition)是高阶推广。
- 张量网络 = 把高阶张量用低阶矩阵因子化。

### 18.6 与第 8 章(高维几何)的接口

- **Marchenko-Pastur 定律**:随机矩阵特征值的极限分布。
- 高维数据的特征值谱 = "内在维度"。
- Johnson-Lindenstrauss 投影 ≈ "保留谱信息"的随机投影。

### 18.7 与第 9 章(稀疏与低秩)的接口

- **谱归一化** 与 **LoRA** 是不同角度的"约束":谱归一化是 $W$ 的范数约束,LoRA 是 $W$ 的秩约束。
- **PCA**(基于特征分解)与 **NMF**(非负矩阵分解)是两种降维思路。

### 18.8 与第 10 章(特殊矩阵)的接口

- **循环矩阵**的特征值 = 离散傅里叶变换(DFT),这是 FFT 卷积的理论基础。
- **正交矩阵**的特征值模 = 1(谱半径 = 1)。
- **置换矩阵**的特征值 = 单位根。

---

## 19. 论文笔记: 用特征值/谱读懂 5 篇重要论文

### 19.1 PageRank (Brin & Page 1998)

**核心**: 网页排序 = 求 Google 矩阵的主特征向量。

**矩阵形式**: $G = d M + (1-d) \frac{1}{n} \mathbf{1}\mathbf{1}^\top$,求 $\pi$ s.t. $G \pi = \pi$。

**求解**: 幂迭代(因为 $G$ 的谱半径 = 1,主特征值唯一)。

### 19.2 Spectral Normalization GAN (Miyato 2018)

**核心**: 约束判别器的 Lipschitz 常数 = 约束权重矩阵的谱范数。

**矩阵形式**: $\|W\|_2 = \sigma_{\max}(W) \le 1$。

**实现**: 幂迭代估计 $\sigma_{\max}$(因为 SVD 太贵)。

### 19.3 GCN (Kipf 2017)

**核心**: 图卷积 = 归一化 Laplacian 的低阶多项式。

**矩阵形式**: $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$,其中 $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$。

**谱视角**: Laplacian 的特征向量 = 图的"傅里叶基",GCN = "图信号的频域滤波"。

### 19.4 Neural Tangent Kernel (Jacot 2018)

**核心**: 无限宽神经网络的训练 = 核回归。

**矩阵形式**: NTK $\Theta = \mathbb{E}[\nabla f(x) \nabla f(x)^\top]$ 是一个对称正定矩阵。

**谱视角**: NTK 的特征值谱 = 神经网络的"学习速度谱"。

### 19.5 LanczosNet (Liao 2019)

**核心**: GNN 的特征值视角。

**矩阵形式**: 用 Lanczos 算法三对角化邻接矩阵,再用小矩阵的特征分解近似原图的谱。

**应用**: 大规模图的快速 GNN 推理。

---

## 20. 关键公式速查

| 概念 | 公式 | PyTorch 一行 |
| :--- | :--- | :--- |
| 特征值 | $A\mathbf{v} = \lambda \mathbf{v}$ | `eigvals, eigvecs = torch.linalg.eig(A)` |
| 对称特征值 | $A = Q\Lambda Q^\top$ | `eigvals, Q = torch.linalg.eigh(A)` |
| 谱半径 | $\rho(A) = \max_i \|\lambda_i\|$ | `torch.linalg.eigvals(A).abs().max()` |
| 谱范数 | $\|A\|_2 = \sigma_{\max}(A)$ | `torch.linalg.svdvals(A)[0]` |
| 谱归一化 | $W / \sigma_{\max}(W)$ | `nn.utils.spectral_norm(layer)` |
| 幂迭代 | $\mathbf{v}_{k+1} = A\mathbf{v}_k / \|\cdot\|$ | 手写或 `lobpcg` |
| 条件数 | $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ | `torch.linalg.cond(A)` |
| PageRank 矩阵 | $G = dM + (1-d)/n \mathbf{1}\mathbf{1}^\top$ | `d * M + (1-d)/n * torch.ones(n,n)` |
| Laplacian | $L = D - A$ | `D - A` |
| 归一化 Laplacian | $\hat{A} = D^{-1/2} A D^{-1/2}$ | `D^{-0.5} @ A @ D^{-0.5}` |

---

## 21. 章节总结图

```text
                任意方阵 A
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
    特征分解     谱定理(对称)   SVD(任意)
   (一般方阵)    (对称实/Hermitian)  m×n
       │            │            │
       │            │            │
       ▼            ▼            ▼
   λ, v        QΛQ^T        UΣV^T
       │            │            │
       │            ▼            ▼
       │     ┌────────────────────────┐
       │     │ 工程应用:              │
       │     │ - 谱归一化 (SN-GAN)    │
       │     │ - 条件数 / 收敛速度    │
       │     │ - PageRank             │
       │     │ - Laplacian / GCN      │
       │     │ - 谱聚类              │
       │     │ - NTK (无限宽网络)     │
       │     └────────────────────────┘
       │
       ▼
   ┌──────────────────────┐
   │ 数值方法:            │
   │ - 幂迭代(主特征对)   │
   │ - QR 算法(全谱)     │
   │ - Lanczos(top-k)    │
   │ - Jacobi(对称)      │
   │ - 分而治之(并行)     │
   └──────────────────────┘
```

---

> **本章版本**: v1.1
> **最后更新**: 2026-06-26
> **本章扩充内容**: 14. 工程师误区(10 坑) / 15. 完整 PageRank+谱聚类+SN-GAN 案例 / 16. 几何 ASCII / 17. 动手算一算(10 题) / 18. 跨章关联 / 19. 5 篇论文解读 / 20. 公式速查 / 21. 总结图

---

## 22. 谱定理完整推导(给"想真正搞懂"的读者)

### 22.1 谱定理陈述

**实谱定理**: 设 $A$ 是 $n \times n$ **实对称矩阵**,则存在**正交矩阵** $Q$ 和**实对角矩阵** $\Lambda$,使得:

$$A = Q\Lambda Q^\top$$

其中 $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$, $\lambda_i \in \mathbb{R}$ 是 $A$ 的特征值,$Q$ 的列是**两两正交的特征向量**。

### 22.2 推导思路

**核心步骤**:
1. $A$ 是实对称 → $A^\top = A$。
2. 实对称矩阵的特征值全为实数(因为 $A\mathbf{v} = \lambda \mathbf{v}$,两边乘 $\bar{\mathbf{v}}^\top$,得 $\bar{\mathbf{v}}^\top A \mathbf{v} = \lambda \|\mathbf{v}\|^2$。左边 = $\overline{\mathbf{v}^\top A^\top \mathbf{v}} = \overline{\mathbf{v}^\top A \mathbf{v}}$,所以 $\lambda = \bar{\lambda}$)。
3. 不同特征值的特征向量**正交**(因为 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$, $A\mathbf{v}_j = \lambda_j \mathbf{v}_j$,代入 $(\lambda_i - \lambda_j) \mathbf{v}_i^\top \mathbf{v}_j = 0$,得 $\mathbf{v}_i^\top \mathbf{v}_j = 0$)。
4. 重特征值: 重数 = 几何重数(因为实对称必可对角化),选正交基即可。
5. 把特征向量放成 $Q$ 的列(归一化),$A = Q\Lambda Q^\top$。

### 22.3 几何: 谱定理 = "选最佳坐标系"

任何实对称矩阵 $A$,在特征向量组成的正交基下,就是**沿每个坐标轴独立缩放**。

```text
A 是"歪扭"的对称变换:

     y                    y
     ↑                    ↑
     │ ╲                  │
     │  ╲                 │ 3
     │   ╲                │ ↑
     │    ╲               │ ↑ (沿 v₁ 方向拉伸 3)
     ●────→ x             ●────→ x  v₁
                            ╲
                             ╲ 1
                              ╲
                               ╲ (沿 v₂ 方向拉伸 1)
                                v₂
```

### 22.4 谱定理在 DL 中的应用

#### 应用 1: 协方差矩阵的 PCA

数据矩阵 $X \in \mathbb{R}^{n \times d}$(每行一个样本),协方差 $\Sigma = \frac{1}{n} X^\top X$。

$\Sigma$ 对称半正定 → 谱定理 → $\Sigma = Q\Lambda Q^\top$。

$Q$ 的列 = **主成分方向**,$\Lambda$ 的对角元 = 各方向方差。

#### 应用 2: Hessian 的特征值分析

Hessian $H$ 是对称(损失是标量函数),谱定理 → $H = Q\Lambda Q^\top$。

$\lambda_{\max}(H)$ = 最陡方向曲率;$\lambda_{\min}(H)$ = 最平方向曲率。

**Adam / L-BFGS** 用 Hessian 的近似(对角 / 曲率对)做预条件化(详见 `../../../方向教程/rec教程/第05章-DLRM到GR范式迁移.md`)。

#### 应用 3: Spectral Norm 约束

$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^\top A)}$。

$A^\top A$ 是对称半正定 → 谱定理 → 算 $\lambda_{\max}$ → 算 $\sigma_{\max}$。

**应用**: SN-GAN,谱归一化(详见 §15)。

---

## 23. QR 算法的完整推导(特征值的工业算法)

### 23.1 QR 算法陈述

QR 算法迭代计算矩阵的特征值:

$A^{(k+1)} = R^{(k)} Q^{(k)}$,其中 $A^{(k)} = Q^{(k)} R^{(k)}$。

**收敛**: 当 $k \to \infty$, $A^{(k)} \to$ 上三角(对角元 = 特征值)。

### 23.2 为什么收敛?

直觉: $A^{(k+1)} = R^{(k)} Q^{(k)} = Q^{(k)\top} A^{(k)} Q^{(k)}$(因为 $A^{(k)} = Q^{(k)} R^{(k)}$)。

所以 $A^{(k+1)}$ 是 $A^{(k)}$ 的**正交相似变换**。这种迭代让矩阵"接近对角"。

### 23.3 实践中的 QR 算法

工业实现用**带 shift 的 QR 算法**(Wilkinson shift),收敛速度从线性变**二次方**。

### 23.4 PyTorch / LAPACK 中的实现

```python
import torch

A = torch.randn(10, 10)
A = (A + A.T) / 2  # 对称化

# PyTorch 的特征值(用 LAPACK)
eigvals = torch.linalg.eigvalsh(A)
print("特征值:", eigvals.numpy())
```

底层用的就是 **QR 算法 + shift**(LAPACK 的 `dsyevr` 等)。

---

## 24. 谱方法的"应用大全"

### 24.1 推荐系统

| 方法 | 矩阵 | 谱解读 |
| :--- | :--- | :--- |
| SVD 分解 | 用户-物品矩阵 | 截断 SVD = 谱方法 |
| PageRank | 邻接矩阵 | 主特征向量 |
| Spectral Clustering | 相似度矩阵 | Laplacian 前 K 特征向量 |
| 谱嵌入 | 图 Laplacian | 谱坐标 = 节点 Embedding |

### 24.2 深度学习

| 方法 | 矩阵 | 谱解读 |
| :--- | :--- | :--- |
| SN-GAN | 权重 $W$ | $\sigma_{\max}(W)$ 控制 Lipschitz |
| Spectral Norm Reg | Hessian | $\lambda_{\max}$ 控制损失锐度 |
| GCN | 邻接矩阵 $A$ | 归一化 Laplacian 的多项式 |
| NTK | NTK 矩阵 $\Theta$ | 特征值谱决定学习速度 |
| LARS | 梯度 $g$ | $\|g\|/\|W\|$ 比例 |
| Sharpness-Aware Min | Hessian | 最小化 $\lambda_{\max}$ |
| LoRA | $\Delta W$ | $\Delta W$ 的秩 = 谱方法 |

### 24.3 图论

| 方法 | 矩阵 | 谱解读 |
| :--- | :--- | :--- |
| 图连通性 | Laplacian $L$ | 0 特征值的重数 = 连通分量数 |
| Fiedler 向量 | Laplacian $L$ | 第二小特征向量 = 图的二分 |
| 图粗化 | Laplacian $L$ | 大特征值 = 局部细节 |
| 图扩散 | $I - \alpha L$ | 谱滤波 |
| 谱聚类 | 相似度 $W$ | 前 K 特征向量 |

### 24.4 信号处理

| 方法 | 矩阵 | 谱解读 |
| :--- | :--- | :--- |
| 傅里叶变换 | 循环矩阵 | 特征向量 = DFT 基 |
| 滤波器 | 卷积核 | 频率响应 = 特征值 |
| 谱图卷积 | Graph Laplacian | 谱域 = 频域推广 |

---

## 25. 谱聚类的 Ng-Jordan-Weiss 算法详解

### 25.1 算法步骤

给定相似度矩阵 $W \in \mathbb{R}^{n \times n}$,聚成 $K$ 类:

1. 算对角度矩阵 $D_{ii} = \sum_j W_{ij}$。
2. 算 Laplacian $L = D - W$。
3. 算 $L$ 的**前 K 个最大特征值**对应的特征向量 $U \in \mathbb{R}^{n \times K}$。
4. 把 $U$ 的行视为 $K$ 维向量,做 K-means 聚类。

### 25.2 为什么是"最大"特征值?

直觉:
- Laplacian 0 特征值对应常数向量(无信息)。
- **最小**非零特征值(Fiedler)对应**最平滑**的非平凡方向。
- 但 Ng-Jordan-Weiss 用**最大**特征值,源自**随机游走视角**。
- 实际工程上两者都有人用,**最小 K 特征值** (Shi-Malik Ncut) 和**最大 K 特征值** (Ng-Jordan-Weiss)。

### 25.3 完整代码

```python
import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(W, K):
    """W: 相似度矩阵 (n×n)
    K: 聚类数"""
    n = W.shape[0]

    # Step 1: 度矩阵
    D = np.diag(W.sum(axis=1))

    # Step 2: Laplacian
    L = D - W

    # Step 3: 特征分解
    eigvals, eigvecs = np.linalg.eigh(L)

    # 取前 K 个最大特征值对应的特征向量(Ng-Jordan-Weiss)
    U = eigvecs[:, -K:]

    # Step 4: 行归一化
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    # Step 5: K-means
    kmeans = KMeans(n_clusters=K)
    labels = kmeans.fit_predict(U_norm)

    return labels, eigvals

# 用法
import sklearn.datasets
X, y = sklearn.datasets.make_circles(n_samples=300, noise=0.05, factor=0.5)
from sklearn.neighbors import kneighbors_graph
W = kneighbors_graph(X, n_neighbors=10, include_self=False).toarray()
W = (W + W.T) / 2  # 对称化
W = W * 1  # 权重,可以用 RBF 核

labels, eigvals = spectral_clustering(W, K=2)
```

### 25.4 时间复杂度

- 相似度矩阵计算: $O(n^2)$ (全连接) 或 $O(nk)$ (kNN)。
- 特征分解: $O(n^3)$。
- K-means: $O(nKdT)$,$T$ 迭代次数,$d = K$。

**优化**:
- 用 **Lanczos 算法** 加速特征分解(Ng-Jordan-Weiss 只需前 K 个最大)。
- 用 **Nyström 近似** 处理大规模数据。
- 用 **稀疏矩阵** + 迭代法。

---

## 26. PageRank 的完整数学推导

### 26.1 原始 PageRank 模型

**假设**: 网页浏览者:
- 概率 $d$ 跟随页面上的链接。
- 概率 $1-d$ 随机跳到任意页面。

**稳态分布** $\pi$ 满足:

$$\pi = d M \pi + \frac{1-d}{n} \mathbf{1}$$

其中 $M$ 是**列随机**转移概率矩阵。

### 26.2 改写成特征值问题

两边乘 $\mathbf{1}^\top$ (因为 $\pi$ 是概率分布,和为 1):

$$\pi^\top = d \pi^\top M + \frac{1-d}{n} \mathbf{1}^\top$$

定义 Google 矩阵 $G = d M + (1-d) \frac{1}{n} \mathbf{1}\mathbf{1}^\top$:

$$\pi^\top G = \pi^\top$$

即 $\pi$ 是 $G$ 的**左特征向量**(对应特征值 1)。

或等价地,$G^\top \pi = \pi$(因为 $G$ 对称化后是列随机)。

### 26.3 收敛性保证

**Stochastic 矩阵** (列随机, $\sum_i G_{ij} = 1$) 的 Perron-Frobenius 定理保证:
- 谱半径 = 1。
- 主特征值 1 的特征向量**严格正**。
- 幂迭代收敛。

### 26.4 Dead-end 处理

**问题**: 有些网页没有外链(列全为 0),$M$ 的这一列全 0,导致 PageRank 泄漏。

**修法**:
- 把 dead-end 的列设为 $\frac{1}{n} \mathbf{1}$(强制均匀跳)。
- Google 矩阵本身就是这修法的体现(包含 teleport 项)。

### 26.5 Personalized PageRank

**修改**: teleport 分布不再是均匀 $\frac{1}{n}\mathbf{1}$,而是用户偏好 $\mathbf{p}$:

$$\pi = d M \pi + (1-d) \mathbf{p}$$

**应用**: 推荐系统(基于用户行为图,而不是全局网页图)。

### 26.6 谱方法视角

PageRank = "在网页图上做 random walk + teleport"。
稳态分布 = "图上最常访问的节点" = 邻接矩阵的主特征向量。

---

## 27. 谱归一化与谱约束的工业实现

### 27.1 SN-GAN 的 SpectralNorm 层(PyTorch 源码)

```python
# PyTorch nn.utils.spectral_norm 的核心算法
class SpectralNorm:
    def __init__(self, weight, n_power_iterations=1, dim=0, eps=1e-12):
        self.weight = weight  # 形状 (out, in, ...)
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps

        # 初始化 u 向量
        weight_mat = self._reshape_weight()
        h = weight_mat.size(0)
        u = weight_mat.new_empty(h).normal_()
        self.u = F.normalize(u, dim=0)
        self.v = None  # 懒初始化

    def _reshape_weight(self):
        # 把 weight reshape 成 (out_features, in_features)
        return self.weight.view(self.weight.size(0), -1)

    def __call__(self):
        weight_mat = self._reshape_weight()

        # 幂迭代估计 σ_max
        u = self.u
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(weight_mat.t(), u), dim=0)
            u = F.normalize(torch.mv(weight_mat, v), dim=0)

        # σ_max ≈ u^T W v
        sigma = torch.dot(u, torch.mv(weight_mat, v))

        # 归一化
        return self.weight / sigma.clamp(min=self.eps)
```

### 27.2 Sharpness-Aware Minimization (SAM)

**核心**: 不仅最小化 loss,还要最小化 loss 在 $\epsilon$-邻域内的最大值。

**矩阵视角**: 等价于最小化 Hessian 的最大特征值 $\lambda_{\max}(H)$。

```python
# SAM 的两阶段优化
def sam_step(model, x, y, rho=0.05):
    # 第一步: 找最坏情况
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    eps = rho * grad / grad.norm()
    model.parameters().add_(eps)

    # 第二步: 在最坏情况下算梯度
    loss_adv = F.cross_entropy(model(x), y)
    loss_adv.backward()

    # 还原
    model.parameters().sub_(eps)

    # 用 loss_adv 的梯度更新
    optimizer.step()
```

### 27.3 NTK 的特征值分析

**Neural Tangent Kernel**: $\Theta(x, x') = \mathbb{E}[\nabla f(x) \nabla f(x')^\top]$(期望 over 随机初始化)。

**性质**:
- $\Theta$ 是**对称半正定**核。
- 在**无限宽**网络中,$\Theta$ 收敛到**确定核**。
- 训练时,$\Theta$ **近似不变**(因为参数变化很小)。

**应用**:
- 训练等价于**核回归**($\Theta$ 是核矩阵)。
- **特征值谱** $\lambda_1 \ge \lambda_2 \ge \ldots$ 决定学习速度:大特征值方向学得快,小特征值方向学得慢。

---

## 28. 谱方法在 NLP 中的应用

### 28.1 Transformer 中的谱

**关键矩阵**:
- $W_Q, W_K, W_V, W_O$ 都是 $d \times d$ 矩阵。
- $Q K^\top$ 是 $L \times L$ 矩阵,每行是 attention 分布。

**谱视角**:
- $W_Q W_K^\top$ 的特征值谱决定 attention 的"集中度"。
- 大特征值 → attention 集中在少数 token。
- 小特征值 → attention 均匀分散。

### 28.2 LoRA 的谱视角

LoRA: $W + BA$,其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$。

$\Delta W = BA$ 的**有效秩** ≈ $r$。

**谱解读**:
- 训练好的 LoRA,$BA$ 的前 $r$ 个奇异值远大于其余 → "低秩假设"成立。
- 如果前 $r$ 个奇异值衰减快,$r$ 可以再小(如 4)。

### 28.3 Spectral Pooling

把图像的特征图 $X \in \mathbb{R}^{H \times W \times C}$ 做 FFT:

$X_f = \text{FFT}(X)$

截断高频(只保留前 $K$ 个频率):

$X_{f,\text{trunc}} = X_f[:K, :K, :]$

逆 FFT:

$X_{\text{trunc}} = \text{iFFT}(X_{f,\text{trunc}})$

**优点**: 比 MaxPool / AvgPool 更"保留全局信息"。

---

## 29. 谱方法的局限与替代

### 29.1 局限

| 问题 | 描述 |
| :--- | :--- |
| **计算复杂度** | 全特征分解 $O(n^3)$,对大矩阵不可行 |
| **特征向量解释性差** | 尤其在高维场景 |
| **对噪声敏感** | 病态矩阵的特征向量数值不稳定 |
| **符号不定性** | 特征向量方向任意($+\mathbf{v}$ 或 $-\mathbf{v}$ 都行) |

### 29.2 替代方案

| 场景 | 替代 |
| :--- | :--- |
| 大规模稀疏矩阵 | Lanczos / ARPACK |
| Top-k 特征值 | 幂迭代 + Deflation |
| 高维稀疏 | 随机 SVD |
| 实时需求 | 在线特征估计 |

### 29.3 何时不用谱方法

- 矩阵**太稀疏**且特征值谱**太平坦** → 谱方法无优势。
- 需要**快速推理** → 用前向传播 + 简单矩阵乘。
- 数据**不是图结构** → 用 NN Embedding 替代。

---

## 30. 章节自检 25 题

### 基础题

1. 矩阵 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ 的特征值和特征向量?
2. 谱半径 $\rho(A)$ 和谱范数 $\|A\|_2$ 的关系是什么?
3. PageRank 的转移概率矩阵 $M$ 有什么性质?
4. 谱定理对什么类型的矩阵成立?
5. 谱归一化在 GAN 中约束什么?

### 中级题

6. 解释"幂迭代收敛到主特征向量"的几何。
7. Laplacian 矩阵的特征值有什么非负性质?
8. Fiedler 向量在图论中代表什么?
9. 写出 Lanczos 算法的核心迭代式。
10. PageRank 的阻尼因子 $d$ 太大或太小会怎样?

### 高级题(ML/DL)

11. 解释 GCN 中为什么用 $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$ 而不是 $A$?
12. SN-GAN 中谱归一化的 Lipschitz 约束数学形式?
13. NTK 的特征值谱如何决定学习速度?
14. 用 Lanczos 估计大型稀疏矩阵的 top-k 特征值(伪代码)。
15. LoRA 的低秩假设在谱方法下的解读?

### 论文题

16. 读 PageRank 论文,识别 $\pi = G \pi$ 的数学形式。
17. 读 Spectral Normalization GAN,识别谱归一化的 Lipschitz 推导。
18. 读 GCN 论文,识别 Laplacian 归一化的动机。
19. 读 NTK 论文,识别 $\Theta$ 的特征值谱分析。
20. 用谱方法视角重新解释 Transformer 的 Attention 矩阵。

### 编程题

21. 用 PyTorch 实现 PageRank(用幂迭代)。
22. 用 PyTorch 实现谱聚类(Laplacian + K-means)。
23. 用 PyTorch 实现 SN-GAN 的 SpectralNorm。
24. 用 PyTorch 实现 Lanczos 算法。
25. 用 PyTorch 比较 $A, A^\top$ 的特征值(为什么相同?)

<details>
<summary>对应小节</summary>

1-5: §1-§5
6-10: §6-§10
11-15: §15-§17 + 跨章
16-20: §19 + 跨章
21-25: §15-§17
</details>

---

## 31. 一句话核心 takeaway

**特征值是矩阵的"主方向缩放因子",谱定理把对称矩阵分解为"旋转 + 缩放 + 反旋转",谱半径决定矩阵幂的增长,谱范数是矩阵作为线性映射的最大伸长倍数。这些是 PageRank、谱归一化、GNN、Lanczos、NTK 的共同数学基础。**

---

> **本章版本**: v1.2
> **最后更新**: 2026-06-26
> **本章扩充内容**: 22-31 节: 谱定理完整推导 / QR 算法 / 谱方法应用大全 / 谱聚类详解 / PageRank 数学 / SN-GAN 源码 / NTK 谱分析 / Transformer 谱视角 / 自检 25 题

---

## 32. 谱方法的"几何全息图"

这一节用一张图把本章所有几何概念串起来。

```text
              任意矩阵 A 的"几何摘要"
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    对称矩阵 A        一般方阵 A        任意矩阵 A (m×n)
        │                 │                 │
        ▼                 ▼                 ▼
    谱定理             特征分解            SVD
    A = QΛQ^T        Av = λv           A = UΣV^T
        │                 │                 │
        │                 │                 │
   "沿正交基方向      "沿特征方向       "旋转 + 缩放 +
    独立缩放"          缩放(可能复数)"   反旋转"
        │                 │                 │
        │                 │                 │
        ▼                 ▼                 ▼
    ┌──────────────────────────────────────────┐
    │ 共同点: 找"主方向" + "对应缩放因子"      │
    │ 不同点: 适用范围 + 数值性质               │
    └──────────────────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │ 工程应用:            │
              │ - PageRank (主特征)   │
              │ - 谱归一化 (σ_max)    │
              │ - GCN (Laplacian 谱)  │
              │ - NTK (核特征值谱)    │
              │ - 谱聚类 (前 K 特征)  │
              └──────────────────────┘
```

---

## 33. 给"读论文遇到谱方法卡壳"的 checklist

读论文时如果遇到"矩阵的谱 / 特征值"等概念,问自己 4 个问题:

### Q1: 这是什么矩阵?

- 协方差矩阵?(对称半正定,谱定理)
- 邻接矩阵?(非负,可能稀疏,谱方法)
- 权重矩阵 $W$?(一般方阵,谱范数)
- Laplacian?(对称半正定,谱 = 图的代数结构)
- Hessian?(对称,谱 = 损失 landscape)
- NTK / Kernel 矩阵?(对称半正定,核函数)

### Q2: 用到"谱"的哪个性质?

| 性质 | 应用 |
| :--- | :--- |
| 主特征值最大 | 收敛速度,主方向 |
| 谱半径 $\rho$ | 矩阵幂的增长 |
| 谱范数 $\|A\|_2 = \sigma_{\max}$ | Lipschitz 常数 |
| 条件数 $\kappa$ | 数值稳定性 |
| Laplacian 0 重数 | 图连通性 |
| Fiedler 值 | 图的二分 |

### Q3: 怎么数值计算?

- 全谱分解:`torch.linalg.eigh`(对称)或 `eig`(一般)。
- Top-k:幂迭代(简单)、Lanczos(快速)、`lobpcg`(大规模)。
- 谱范数:幂迭代(便宜)或 SVD(贵但准)。

### Q4: 这是否适合用谱方法?

- ✅ 矩阵小(几百万 × 几百万以内)、结构好(对称 / 稀疏)。
- ❌ 矩阵太大、太稠密、谱太平坦。

---

## 34. 工业界 5 大谱方法应用对比

| 应用 | 矩阵 | 谱对象 | 计算量 | 工业库 |
| :--- | :--- | :--- | :--- | :--- |
| **PageRank** | 邻接矩阵 $A$ | 主特征向量 | 中 | NetworkX, igraph, Spark GraphX |
| **谱归一化 (SN-GAN)** | 权重 $W$ | $\sigma_{\max}(W)$ | 极小 | PyTorch (`spectral_norm`) |
| **谱聚类** | 相似度 $W$ | Laplacian 前 K | 中 | sklearn, PyTorch Geometric |
| **GCN** | 邻接 $A$ | Laplacian 归一化 | 一次性 | PyG, DGL, TF-GNN |
| **NTK** | NTK 矩阵 $\Theta$ | 特征值谱 | 大 | jax, neural-tangents |

---

## 35. 谱方法的"反面":何时不用

谱方法不是银弹。**下面这些场景,不要硬用谱方法**:

1. **大规模稠密矩阵**: $n > 10^6$ 时,特征分解太贵。用 Lanczos / 随机方法。
2. **特征值谱平坦**: 如果 $\lambda_{\max}/\lambda_{\min} \approx 1$,主特征方向不明显,迭代法收敛慢。
3. **稀疏 + 不对称**: 谱定理不适用,只能用 SVD 或一般特征分解(慢)。
4. **实时推理**: 特征分解无法做前向传播。
5. **数据非图结构**: Embedding / NN 通常够用,不需要谱方法。

---

## 36. 推荐阅读清单(论文 + 教材)

### 36.1 教材

1. **Strang - Linear Algebra and Its Applications**:第 6 章特征值,讲得最几何。
2. **Horn & Johnson - Matrix Analysis**:特征值扰动理论的圣经。
3. **Golub & Van Loan - Matrix Computations**:QR 算法、Lanczos、Shift-and-Invert。
4. **Demmel - Applied Numerical Linear Algebra**:大规模稀疏矩阵的工业实现。

### 36.2 论文

5. **Brin & Page 1998**:PageRank 原始论文。
6. **Miyato et al. 2018**:Spectral Normalization for GANs (ICLR 2018)。
7. **Kipf & Welling 2017**:Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)。
8. **Jacot et al. 2018**:Neural Tangent Kernel (NeurIPS 2018)。
9. **Von Luxburg 2007**:A Tutorial on Spectral Clustering (统计与机器学习的综述)。
10. **Liao et al. 2019**:LanczosNet (NeurIPS 2019)。

### 36.3 在线资源

11. **LAPACK 用户指南**:LAPACK 算法的官方文档。
12. **PyTorch `torch.linalg` 文档**:工业级线性代数函数。
13. **NetworkX 文档**:PageRank / 谱聚类参考实现。

---

## 37. 章节总结: 一句话带你看完全章

**特征值是矩阵的"主方向缩放因子",谱定理把对称矩阵拆为"正交基 + 缩放因子",PageRank 是邻接矩阵的主特征向量,谱归一化是控制矩阵最大伸长倍数,GCN 是 Laplacian 上的谱滤波,NTK 的特征值谱决定学习速度,谱聚类是用前 K 个特征向量做 K-means。这些都是"矩阵的谱"的工程应用。

---

## 38. 与后续章节的"接口表"

| 本章概念 | 给第 5 章 | 给第 6 章 | 给第 8 章 | 给第 9 章 |
| :--- | :--- | :--- | :--- | :--- |
| 特征值 | 正定性 ↔ λ > 0 | Hessian λ = loss 曲率 | 随机矩阵 λ 分布 (MP) | 谱归一化 vs 低秩约束 |
| 谱定理 | 对称正定 (SPD) | Hessian = 对称 | PCA = 谱方法 | LoRA 的有效秩 |
| 谱半径 | 收敛速度 | 梯度下降收敛 | 测度集中 | 模型压缩率 |
| PageRank | (无直接) | (无直接) | 高维随机图 | 推荐图 Embedding |
| 谱归一化 | 与 Adam 比较 | Lipschitz | 神经网络理论 | LoRA scale |
| Laplacian | 二次型正 | (无直接) | 谱聚类 | 邻接矩阵分解 |
| Lanczos | 共轭梯度法 | Newton-CG | 大规模稀疏谱 | 大规模推荐 |**

---

> **本章版本**: v1.3
> **最后更新**: 2026-06-26
> **本章扩充内容**: 32-37 节: 谱方法全息图 / 论文 checklist / 工业应用对比 / 反面案例 / 推荐阅读 / 章节总结



## 深度扩展 · 多视角系统展开(续)

### 110. 第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 拆解为以下几个层次:

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

信息论为"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 拆解为以下几个层次:

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

信息论为"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 拆解为以下几个层次:

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

信息论为"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 拆解为以下几个层次:

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

信息论为"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 拆解为以下几个层次:

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

信息论为"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 4 章:特征值·特征向量·谱定理——谱归一化·Graph Laplacian·PageRank" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

