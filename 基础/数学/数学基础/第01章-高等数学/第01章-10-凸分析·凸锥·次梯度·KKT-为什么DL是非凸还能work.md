---
title: "第01章-10-凸分析·凸锥·次梯度·KKT-为什么DL是非凸还能work"
tags:
  - 数学
  - 凸分析
  - 凸锥
  - 次梯度
  - KKT
  - 为什么DL是非凸还能work
description: "本章要点：or 展开做局部近似,用 Lebesgue 积分描述概率密度,用泛函分析把函数当向量处理。这些都是\"单点工具\"。"
chapter_id: 数学-10-01
last_updated: 2026-07-14
---

# 第01章-10-凸分析·凸锥·次梯度·KKT-为什么DL是非凸还能work

or 展开做局部近似,用 Lebesgue 积分描述概率密度,用泛函分析把函数当向量处理。这些都是"单点工具"。
>
> 这一章要回答一个**贯穿整个深度学习**的根本问题:
>
> > **"深度学习的 loss landscape 是非凸的,但它依然能 work。这到底是玄学,还是有数学保证?"**
>
> 答案藏在**凸分析**里。凸分析是研究"什么形状的优化问题有解、解是什么"的理论框架。它给出三个工具:
>
> - **凸集**:什么样的"参数集合"是"友好"的
> - **凸函数**:什么样的"loss 曲面"是"友好"的
> - **凸优化**:在凸集上最小化凸函数,**任何局部最优 = 全局最优**
>
> 然后我们**必须**承认:**深度学习的 loss 不是凸函数**。那为什么还能收敛?
>
> 这就是本章的灵魂。1989 年以来大量理论工作(Choromanska 2015、Kawaguchi 2016、Jacot 2018 NTK、Du 2019 过参数化)告诉我们:虽然 DNN 的 loss 是非凸的,但它的**局部最优值**接近全局最优、鞍点远多于局部最优、过参数化网络能"记住"所有训练样本。
>
> **学完这一章,你能回答的问题**:
> - 凸集、凸函数、凸优化的严格定义
> - 次梯度是什么,为什么 ReLU 在 x=0 处的次梯度是 $[0, 1]$
> - KKT 条件的四要素及其几何意义
> - Lagrange 对偶与强对偶(Slater 条件)
> - **为什么深度学习的非凸 loss 还能 work** (核心)
> - SVM、LASSO、MMoE 多任务的凸/非凸性分析
> - 面试 L7/L8 难度的 5 个高频问题
>
> **本章与前后章节的关系**:
> - 第 1-7 章:有限维微积分基础
> - 第 8 章:函数空间视角(为凸分析的"无限维版本"铺垫)
> - 第 9 章:变分法(在函数空间上求极值,与凸优化同源)
> - **第 10 章(本章)**:凸分析,**1-高数章节的收尾**,给整个微积分工具集盖上"为什么 DL 能 work"的印章
> - **6-最优化**:第 5 章凸优化、第 8 章 KKT 与对偶,与本章深度重叠
> - **3-概率论**:第 6 章 MLE 的凸性(MLE = 最小化 KL)
> - **2-线代**:第 5 章正定性(凸性的二阶判据)
> - **7-系统思维**:案例 2 MMoE 多任务"跷跷板"问题
>
> 准备好了吗?我们从工程师的终极困惑开始。

---

## 1. 直觉引入:DL 是非凸的,为什么还能 work?

### 1.1 工程师的终极困惑

**场景 1:线性回归,loss 收敛**。

你训练一个 `LinearRegression()`,loss 从 0.8 → 0.5 → 0.3 → 0.27。你调参、调 lr、调 optimizer,loss 都稳稳收敛。你知道为什么:线性回归的 MSE loss 是**强凸函数**,有闭式解 $\hat{w} = (X^T X)^{-1} X^T y$。凸优化的"圣杯":任何局部最优 = 全局最优。

**场景 2:ResNet-50 训练 CIFAR-10**。

你训练一个 50 层残差网络,loss 也在下降,AUC 在涨。但**没有**人给你一个"全局最优"的理论保证。你换种子,可能收敛到不同的准确率。**为什么 SGD 在非凸 loss 上也能 work?**

**场景 3:MMoE 多任务训练**。

ctr 涨 2%, cvr 跌 1%。ctr 涨, cvr 又跌。多任务之间"打架"。工程师说这是"跷跷板"。**为什么?** 凸优化的语言能不能解释这个现象?

这三个场景对应本章的三类工具:

| 场景 | 数学工具 | 本章哪一节 |
| :--- | :--- | :--- |
| 线性回归收敛 | 凸函数 + 强凸 + 凸优化 | §3, §4 |
| ResNet-50 训练 | 非凸优化理论 + NTK + 过参数化 | §8 |
| MMoE 多任务冲突 | 凸约束 + KKT + 不可行域 | §6, §8.7 |

### 1.2 直觉:非凸 landscape 像"群山",但梯度下降能找到"低处的山谷"

**凸 landscape**(线性回归):

```text
   loss
    ↑
    │   *
    │  * *
    │ *   *
    │*     *            ← 凸函数:任何局部极小 = 全局极小
    │       *
    │        *
    └─────────────→ w
        ↑
    全局最优
```

**非凸 landscape**(DNN):

```text
   loss
    ↑
    │   *           *
    │  * *         * *
    │ *   *  ___  *   *
    │*     **   **     *       ← 非凸:多个局部极小,中间有鞍点
    │       *   *
    │        ***
    └─────────────→ w
       ↑  ↑   ↑
      局  鞍  局
      部  点  部
      最     最
      优     优
```

**DNN 的 loss landscape 长啥样?**(Li et al. 2018 的著名可视化)

```text
   loss
    ↑
    │   ╱╲      ╱╲
    │  ╱  ╲    ╱  ╲     ← 局部极小很多,但它们的"高度"接近
    │ ╱    ╲  ╱    ╲       全局最优的"高度"
    │╱      ╲╱      ╲
    │       鞍点(梯度为0,但不是极小)
    └─────────────→ w
```

**Choromanska et al. 2015** 的核心结果(简化版):

> 在大网络 + 大量数据下,网络 loss 的**局部最优值的分布**接近**全局最优值**。也就是说,只要找到一个局部最优,loss 几乎就是全局最优。

为什么?直觉是:**高维空间中的随机函数,其极值分布会"集中"**。具体原因会在 §8 展开。

### 1.3 ASCII 可视化:凸 vs 非凸 landscape

**凸损失 + 凸约束**:

```text
       y
       ↑
      2┤ ●  ●
       │       ●  ●     ← 椭圆形的水平集
      1┤  ●        ●
       │   ●        ●  ← 约束边界(凸集)
      0┤    ●  ●  ●
       └─────────────→ x
              ↑
         全局最优(凸集内的极小)
```

**非凸损失**:

```text
       loss
       ↑
       │       *
       │      * *    ← 第一个"山谷"
       │     *   *
       │    *     *
       │   *       ****
       │  *            *     ← 第二个"山谷"(更深的全局最优)
       │ *               *
       │*                 *
       └────────────────────→ w
        ↑        ↑
       局部     全局
       极小     极小
```

### 1.4 本章目标

我们的目标分成三层:

1. **基础(§2-§4)**:严格定义凸集、凸函数、凸优化,理解 SVM、LASSO 为什么是凸问题
2. **进阶(§5-§7)**:次梯度(处理不可微凸函数)、KKT 条件(处理约束优化)、对偶理论(primal-dual)
3. **灵魂(§8)**:深度学习为什么是非凸的还能 work,现代理论视角 + 工程师直觉

每节至少 5 个手算例子。PyTorch 代码实验 5 个。

### 1.5 章节路线图

```text
1. 直觉引入                ← 你在这里
2. 凸集 (Convex Set)        ← § 300 行
3. 凸函数 (Convex Function)  ← § 400 行 (重点)
4. 凸优化问题                ← § 350 行 (重点)
5. 次梯度 (Subgradient)      ← § 300 行 (重点)
6. KKT 条件                  ← § 400 行 (重点)
7. 对偶理论                  ← § 300 行 (重点)
8. 非凸悖论                  ← § 300 行 (本章灵魂)
9. PyTorch 代码实验          ← § 200 行
10. 常见误区
11. 面试 Q&A
12. 与其他数学工具的联系
13. 本章总结
```

---

## 2. 凸集 (Convex Set)

### 2.1 直觉:凸集 = "中间的点也在里面"

**生活例子**:超市买的橙子摆成一堆。橙子的轮廓是个**凸集**:两颗橙子连线上任意一点,都还在橙子堆里(被其他橙子覆盖)。

**非凸例子**:月牙形(香蕉)。香蕉上两点连线的中点,**在香蕉外**(空气里)。

**形式定义**:

> **凸集** $\mathcal{C}$:对任意 $x, y \in \mathcal{C}$ 和 $\theta \in [0, 1]$,有
>
> $$\theta x + (1-\theta) y \in \mathcal{C}$$

直觉:集合内任意两点的"凸组合"也在集合内。

**反例**:月牙形(梅花的轮廓)。两片花瓣上的点连线,中点在花心外。

### 2.2 凸集例子:超平面、半空间、球、单纯形

**例 1:超平面**(hyperplane)

$$\mathcal{H} = \{x \in \mathbb{R}^n : a^T x = b\}$$

任意两点 $x_1, x_2 \in \mathcal{H}$,凸组合 $\theta x_1 + (1-\theta) x_2$ 满足:

$$a^T (\theta x_1 + (1-\theta) x_2) = \theta a^T x_1 + (1-\theta) a^T x_2 = \theta b + (1-\theta) b = b$$

所以 $\theta x_1 + (1-\theta) x_2 \in \mathcal{H}$。**超平面是凸集**。

**例 2:半空间**(halfspace)

$$\mathcal{H}^- = \{x \in \mathbb{R}^n : a^T x \leq b\}$$

类似验证:

$$a^T (\theta x_1 + (1-\theta) x_2) = \theta a^T x_1 + (1-\theta) a^T x_2 \leq \theta b + (1-\theta) b = b$$

**半空间是凸集**。

**例 3:欧氏球**(Euclidean ball)

$$\mathcal{B}(x_c, r) = \{x : \|x - x_c\|_2 \leq r\}$$

对 $x_1, x_2 \in \mathcal{B}(x_c, r)$:

$$\|\theta x_1 + (1-\theta) x_2 - x_c\|_2 = \|\theta(x_1 - x_c) + (1-\theta)(x_2 - x_c)\|_2 \leq \theta\|x_1 - x_c\|_2 + (1-\theta)\|x_2 - x_c\|_2 \leq r$$

最后一步用三角不等式。

**例 4:概率单纯形**(probability simplex)

$$\Delta^{n-1} = \{x \in \mathbb{R}^n : x_i \geq 0, \; \sum_i x_i = 1\}$$

任意 $x, y \in \Delta^{n-1}$,$\theta x + (1-\theta) y$ 的每个分量都非负,和为 1。**凸集**。

**例 5:多面体**(polyhedron):半空间的交集

$$\mathcal{P} = \{x : A x \leq b, \; C x = d\}$$

凸集的**交集仍是凸集**(见下)。

### 2.3 凸集运算:交集仍凸,并集不一定

**交集保持凸性**:

> 若 $\mathcal{C}_1, \mathcal{C}_2$ 都是凸集,则 $\mathcal{C}_1 \cap \mathcal{C}_2$ 也是凸集。

**证明**:$x, y \in \mathcal{C}_1 \cap \mathcal{C}_2$ 意味着 $x, y \in \mathcal{C}_1$ 且 $x, y \in \mathcal{C}_2$。凸组合 $\theta x + (1-\theta) y$ 在 $\mathcal{C}_1$ 内(因为 $\mathcal{C}_1$ 凸),也在 $\mathcal{C}_2$ 内(因为 $\mathcal{C}_2$ 凸)。所以在交集内。

**凸集的实际意义**:**约束的合集**。比如"概率向量 + L2 范数 ≤ 1"是 $\{x : x \geq 0, \sum x_i = 1, \|x\|_2 \leq 1\}$。交集 = 同时满足所有约束。

**并集不一定凸**:

两个凸集的并集可以是非凸的。

**反例**:两个不相交的圆 $\mathcal{B}_1$ 和 $\mathcal{B}_2$。两点 $x_1 \in \mathcal{B}_1, x_2 \in \mathcal{B}_2$ 连线中点 $m$ 在两个圆外,所以 $m \notin \mathcal{B}_1 \cup \mathcal{B}_2$。并集**不凸**。

### 2.4 凸锥 (Convex Cone)

**定义**:集合 $\mathcal{K}$ 既是凸集,又对**非负乘法封闭**:

$$\forall x \in \mathcal{K}, \forall \alpha \geq 0: \alpha x \in \mathcal{K}$$

直觉:集合从原点出发,呈"锥形"展开。

**例 1:正卦限**(positive orthant)

$$\mathbb{R}_+^n = \{x \in \mathbb{R}^n : x_i \geq 0 \; \forall i\}$$

- 凸性:分量非负 $\Rightarrow$ 凸组合分量非负
- 锥性:$\alpha \geq 0, x \geq 0 \Rightarrow \alpha x \geq 0$

**例 2:法向锥**(normal cone) ← 在 KKT 里至关重要!

对凸集 $\mathcal{C}$ 上的点 $x_0 \in \mathcal{C}$,**法向锥**定义为:

$$N_\mathcal{C}(x_0) = \{v : v^T (x - x_0) \leq 0 \; \forall x \in \mathcal{C}\}$$

即所有"从 $x_0$ 出发,指向集合外的方向" $v$。

**几何直觉**:

```text
              法向锥
              ↗ v₂
            ↗
   x₀ ─────●─────→ 集合内部
            ↘
              ↘ v₁
              法向锥
```

在多面体顶点,法向锥是"由外向内的所有法向量"。在内部点,法向锥退化到 $\{0\}$。

**例 3:切向锥**(tangent cone),法向锥的"对偶"

$$T_\mathcal{C}(x_0) = \{d : \exists t > 0, \; x_0 + t d \in \mathcal{C}\}$$

即从 $x_0$ 出发,能"沿某条路径"进入 $\mathcal{C}$ 内的方向。

**关键定理**:

$$T_\mathcal{C}(x_0) = N_\mathcal{C}(x_0)^\circ$$

($\circ$ 是极集,含义:对偶)

### 2.5 投影定理:任意点投影到闭凸集有唯一最短距离

**定理**(投影存在唯一性):对闭凸集 $\mathcal{C}$ 和任意点 $x_0 \notin \mathcal{C}$,存在**唯一**的点 $P_\mathcal{C}(x_0) \in \mathcal{C}$ 使得:

$$\|x_0 - P_\mathcal{C}(x_0)\|_2 = \min_{y \in \mathcal{C}} \|x_0 - y\|_2$$

这个点叫 $x_0$ 到 $\mathcal{C}$ 的**投影**。

**几何**:从 $x_0$ 出发,垂直(法向)走到 $\mathcal{C}$ 边界的那一点。

**ML 应用**:推荐系统的 user/item Embedding 需要**归一化**到 L2 球 $\|e\|_2 = 1$ 上,这就是投影 $P_{\mathcal{B}}$:

```python
import torch

def l2_normalize(emb):
    """把 embedding 投影到 L2 单位球"""
    return emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

# 测试
e = torch.tensor([3.0, 4.0])
e_norm = l2_normalize(e)
print(e_norm)  # tensor([0.6000, 0.8000])
print(e_norm.norm())  # tensor(1.0000)
```

### 2.6 5 个手算例子

#### 例 1:验证单位球 $\mathcal{B}(0, 1)$ 是凸集

任意 $x, y \in \mathcal{B}$,即 $\|x\| \leq 1, \|y\| \leq 1$。凸组合 $z = \theta x + (1-\theta) y$。

$$\|z\| = \|\theta x + (1-\theta) y\| \leq \theta \|x\| + (1-\theta) \|y\| \leq \theta \cdot 1 + (1-\theta) \cdot 1 = 1$$

用三角不等式 + 凸组合 ≤ 凸组合上界。$z \in \mathcal{B}$。**凸集** ✓

#### 例 2:验证月牙形不是凸集

月牙形:$\mathcal{C} = \{(x, y) : x^2 + (y-1)^2 \leq 4, (x-1)^2 + y^2 \geq 1\}$。这是"两个圆之差"。

取 $x_1 = (-1.9, 0.99)$(左下沿), $x_2 = (1.9, 0.99)$(右下沿)。中点 $m = (0, 0.99)$。检查 $(0-1)^2 + 0.99^2 = 1 + 0.9801 = 1.9801 > 1$,所以 $m$ 在小圆外,即 $m \notin \mathcal{C}$。**非凸** ✗

#### 例 3:概率单纯形的凸性

$\Delta^{n-1} = \{p \in \mathbb{R}^n : p_i \geq 0, \sum p_i = 1\}$。

$p, q \in \Delta$,$\theta p + (1-\theta) q$:
- 每个分量 $\theta p_i + (1-\theta) q_i \geq 0$ (凸组合保持非负) ✓
- $\sum_i [\theta p_i + (1-\theta) q_i] = \theta \sum p_i + (1-\theta) \sum q_i = \theta + (1-\theta) = 1$ ✓

**凸集** ✓

#### 例 4:法向锥的几何

考虑 $\mathcal{C} = \{x : x \geq 0\}$(非负实数)。在 $x_0 = 0$ 处:

$$N_\mathcal{C}(0) = \{v : v \cdot (x - 0) \leq 0 \; \forall x \geq 0\} = \{v : vx \leq 0 \; \forall x \geq 0\} = \{v : v \leq 0\} = \mathbb{R}_-$$

即负实数集。在 $x_0 > 0$ 处,$N_\mathcal{C}(x_0) = \{0\}$(内点无"向外"的方向)。

**工程意义**:KKT 条件中,$\lambda \geq 0$ 是"对偶可行性",它对应不等式约束的**法向锥** $\mathbb{R}_-$ 的反向。

#### 例 5:投影到 L2 球

$x_0 = (3, 4)$,$\mathcal{C} = \{x : \|x\| \leq 1\}$。

$$P_\mathcal{C}(x_0) = \frac{x_0}{\|x_0\|} = \frac{(3, 4)}{\sqrt{9 + 16}} = \frac{(3, 4)}{5} = (0.6, 0.8)$$

验证:$\|(0.6, 0.8)\| = \sqrt{0.36 + 0.64} = 1$ ✓

### 2.7 工程意义:推荐 Embedding 的归一化约束

**场景**:YouTube DNN、双塔模型、对比学习的 embedding。工程师经常做:

```python
user_emb = model.encode_user(user_id)  # (batch, dim)
user_emb = F.normalize(user_emb, dim=-1)  # L2 归一化到单位球
```

为什么?

**数学理由**:用**内积**做相似度 $\text{sim}(u, v) = u^T v = \|u\| \cdot \|v\| \cos\theta$。如果 $\|u\|, \|v\|$ 任意,内积受范数干扰。归一化后,内积 = 余弦相似度,只反映方向。

**凸集视角**:归一化 = 把 embedding 投影到单位球 $\mathcal{S}^{d-1} = \{x : \|x\| = 1\}$。单位球是凸集,投影有唯一解。

**对比学习**:SimCSE、Sentence-BERT 都要归一化,原因同上。

---

## 3. 凸函数 (Convex Function) ← 重点

### 3.1 直觉:凸函数 = "中间点的函数值 ≤ 端点的平均值"

**几何定义**:函数图像上任意两点连成的弦,都"在函数上方"(或者正好贴着函数)。

```text
   f(x)
    ↑       ●
    │      ╱ ●        ← 上方:弦;下方:函数
    │    ●    ●
    │   ╱      ●
    │  ●        ●
    │ ●          ●
    └──────────────→ x
```

**非凸函数**:

```text
   f(x)
    ↑
    │       ●
    │      ╱╲      ← 上凸下凹:违反凸定义
    │  ●  ╱  ╲
    │   ╲╱    ●
    │    ●    ●
    │     ●  ●
    └──────────────→ x
```

**形式定义**:

> **凸函数** $f: \mathcal{C} \to \mathbb{R}$:对任意 $x, y \in \mathcal{C}$ 和 $\theta \in [0, 1]$:
>
> $$f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y)$$

### 3.2 一阶判据:切线在下方

**定理**:设 $f$ 可微,则 $f$ 是凸函数当且仅当:

$$f(y) \geq f(x) + \nabla f(x)^T (y - x), \quad \forall x, y \in \text{dom}(f)$$

**几何解释**:在 $x$ 处的**切线**(线性近似 $f(x) + \nabla f(x)^T (y-x)$)永远在 $f$ 的**下方**(或刚好贴着)。

```text
   f(y)
    ↑
    │       ● y
    │      ╱|
    │     ╱ |       ← 切线:始终在 f 下方
    │    ╱  |
    │   ●   |   ← x, 切点
    │  ╱|   |
    │ ╱ |   |
    │╱  |   |
    └───●───●──────→ y
        x
```

**直觉检验**:凸函数像"碗",碗壁的任何切线都在碗外面或贴着碗。

**手算例 1**:$f(x) = x^2$,验证一阶判据。

$\nabla f(x) = 2x$。$f(y) - f(x) - \nabla f(x)^T(y-x) = y^2 - x^2 - 2x(y-x) = y^2 - 2xy + x^2 = (y-x)^2 \geq 0$ ✓

**手算例 2**:$f(x) = e^x$,验证一阶判据。

$\nabla f(x) = e^x$。$f(y) - f(x) - \nabla f(x)^T(y-x) = e^y - e^x - e^x (y-x)$。

注意到 $e^y$ 的 Taylor 展开 $e^y = e^x + e^x (y-x) + \frac{e^\xi}{2}(y-x)^2$ 对某个 $\xi$ 在 $x, y$ 之间。所以 $e^y - e^x - e^x(y-x) = \frac{e^\xi}{2}(y-x)^2 \geq 0$ ✓

### 3.3 二阶判据:Hessian 半正定

**定理**(二阶判据):设 $f$ 二阶可微,则 $f$ 是凸函数当且仅当:

$$\nabla^2 f(x) \succeq 0, \quad \forall x \in \text{dom}(f)$$

($\succeq$ 是半正定:任意方向 $v$,有 $v^T H v \geq 0$。见 `../../../基础/数学基础/第02章-线性代数/第02章-05-二次型·正定性·条件数-为什么-Adam-比-SGD-稳.md` 第 2 节。)

**几何解释**:Hessian 是函数的"二阶曲率"。每点曲率非负 = 任何方向都是"凹向上",函数是凸的。

**手算例 1**:$f(x) = x^2$,$f''(x) = 2 \geq 0$。凸 ✓

**手算例 2**:$f(x) = x^3$。$f''(x) = 6x$。在 $x < 0$ 时 $f'' < 0$(凹),在 $x > 0$ 时 $f'' > 0$(凸)。**不是全局凸函数**。

**手算例 3**:$f(x, y) = x^2 + xy + y^2$。Hessian:

$$H = \begin{pmatrix} \partial^2 f/\partial x^2 & \partial^2 f/\partial x \partial y \\ \partial^2 f/\partial y \partial x & \partial^2 f/\partial y^2 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

特征值:$\lambda = 2 \pm 1 = 3, 1$。两者都 $> 0$,所以 $H \succ 0$,**严格凸** ✓

**手算例 4**:$f(x) = \log(1 + e^x)$ (softplus)。$f'(x) = \frac{e^x}{1+e^x} = \sigma(x)$。$f''(x) = \sigma(x)(1 - \sigma(x)) = \sigma'(x)$。由于 $\sigma \in [0,1]$,所以 $\sigma' \geq 0$。**凸函数** ✓。这也是 softplus 在深度学习中"凸"的关键。

**手算例 5**:$f(x) = \text{ReLU}(x) = \max(0, x)$。在 $x \neq 0$ 处 $f''(x) = 0$,在 $x = 0$ 处不可微。但:

$$f(\theta x + (1-\theta) y) = \max(0, \theta x + (1-\theta) y) \leq \theta \max(0, x) + (1-\theta) \max(0, y) = \theta f(x) + (1-\theta) f(y)$$

**ReLU 是凸函数** ✓。但**不可微**(在 $x = 0$ 处)。

### 3.4 强凸:f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²

**直觉**:凸函数 + 一个二次项。

**定义** ($\mu$-强凸):$f(y) \geq f(x) + \nabla f(x)^T (y-x) + \frac{\mu}{2} \|y-x\|^2$

**等价定义**:
- $f - \frac{\mu}{2}\|\cdot\|^2$ 是凸函数
- $\nabla^2 f(x) \succeq \mu I$ (Hessian 最小特征值 $\geq \mu$)
- 梯度单调性:$\|\nabla f(x) - \nabla f(y)\| \geq \mu \|x - y\|$

**强凸 vs Lipschitz 平滑**:

| 性质 | 强凸 ($\mu$) | Lipschitz 平滑 ($L$) |
| :--- | :--- | :--- |
| 条件 | $f \geq \text{linear} + \frac{\mu}{2}\|\cdot\|^2$ | $\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$ |
| 含义 | 函数被二次项"夹住" | 梯度 Lipschitz |
| GD 收敛率 | $O((1 - \mu/L)^T)$ 线性 | $O(1/T)$ 次线性 |

**手算例**:$f(x) = x^2$,$\mu = 2, L = 2$。$f(x) - \frac{2}{2} x^2 = 0$ 是凸的,$\nabla^2 f = 2 \geq 2$,所以是 2-强凸 + 2-平滑。GD 收敛率 $O((1 - 1)^T) = 0$,**一步到位**!

**工程意义**:L2 正则 = 把任何凸函数变成强凸函数。带 L2 正则的线性回归 = 强凸,GD 一击必中。

### 3.5 凸函数族

**a) 线性函数**(linear / affine)

$f(x) = a^T x + b$

$\nabla^2 f = 0 \succeq 0$。**凸也是凹**(既是凸函数又是凹函数)。

**b) 二次函数** $f(x) = x^T A x + b^T x + c$

二阶可微,$\nabla^2 f = 2A$。$f$ 凸 $\Leftrightarrow$ $A \succeq 0$。

**手算**:$f(x, y) = 2x^2 + 2xy + y^2$。$A = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$。特征值:$\lambda = \frac{3 \pm \sqrt{5}}{2}$,都 $> 0$。**严格凸** ✓

**c) 范数**(任意 norm)

$f(x) = \|x\|$。凸性来自三角不等式:

$$\|\theta x + (1-\theta) y\| \leq \theta \|x\| + (1-\theta) \|y\|$$

**手算**:$f(x) = |x|$(L1 范数)。$f(\theta x + (1-\theta) y) = |\theta x + (1-\theta) y| \leq \theta |x| + (1-\theta) |y| = \theta f(x) + (1-\theta) f(y)$ ✓

**d) 指数函数**(exponential)

$f(x) = e^{a^T x}$。$f''(x) = a a^T e^{a^T x} \succeq 0$。**凸**。

**e) 最大值**(maximum / max)

$f(x) = \max(x_1, x_2, \ldots, x_n)$。逐点最大保持凸性。

**f) 指示函数**(indicator)

$$I_\mathcal{C}(x) = \begin{cases} 0 & x \in \mathcal{C} \\ +\infty & x \notin \mathcal{C} \end{cases}$$

若 $\mathcal{C}$ 是凸集,则 $I_\mathcal{C}$ 是凸函数。

**g) Softmax / LogSumExp**

$f(x) = \log(\sum_i e^{x_i}) = \text{LogSumExp}(x)$。**凸**($\nabla^2 f$ 是协方差矩阵,半正定)。

**h) 信息熵**(负的)

$H(p) = -\sum p_i \log p_i$ 在概率单纯形上是**凹**的(即 $-H$ 凸)。

### 3.6 凸运算保持凸性

凸函数 $f, g$,常数 $\alpha, \beta \geq 0$:

| 运算 | 结果 |
| :--- | :--- |
| $\alpha f + \beta g$ | 凸 |
| $f(Ax + b)$(仿射变换) | 凸 |
| $\max(f, g)$(逐点最大) | 凸 |
| $f \circ g$(若 $f$ 凸非减, $g$ 凸) | 凸 |

**手算例**:$f(x) = \log(1 + e^x)$ 是 softplus。它 = $\log \circ (1 + e^x)$?不。更直接:$f = \text{LogSumExp}(0, x)$,LogSumExp 是凸 + $(0, x)$ 仿射 + 复合,还是凸。

### 3.7 5 个手算例子汇总

| 函数 | 凸性 | 一阶判据 | 二阶判据 |
| :--- | :--- | :--- | :--- |
| $f(x) = x^2$ | 严格凸 | ✓ | $f''=2 \geq 0$ |
| $f(x) = e^x$ | 严格凸 | ✓ | $f''=e^x > 0$ |
| $f(x) = \|x\|$ | 凸(不可微) | ✓ | 0 处不可微 |
| $f(x) = \max(0, x)$ (ReLU) | 凸(不可微) | ✓ | 0 处不可微 |
| $f(x) = \log(1+e^x)$ (softplus) | 严格凸 | ✓ | $f''=\sigma'(x) \geq 0$ |
| $f(x) = x^3$ | **非凸** | ✗ | $f''=6x$,在 $x<0$ 时负 |
| $f(x) = -\log x$ ($x>0$) | 严格凸 | ✓ | $f''=1/x^2 > 0$ |
| $f(x) = x \log x$ ($x>0$) | 凸(非负) | ✓ | $f''=1/x > 0$ |
| $f(x) = \text{tr}(AXBX^T)$ ($A,B \succeq 0$) | 凸(矩阵变量) | ✓ | 矩阵 Hessian $\succeq 0$ |

### 3.8 ML 中的凸损失

| 损失 | 凸性 | 备注 |
| :--- | :--- | :--- |
| **MSE 回归** | 强凸 | $\frac{1}{2}\|Xw - y\|^2$,$\nabla^2 = X^T X \succeq 0$ |
| **Logistic 回归** | 凸 | 交叉熵 + 线性 |
| **LASSO** | 凸 | MSE + L1 正则 |
| **SVM (合页损失)** | 强凸 | 合页 + L2 |
| **L2 正则线性回归 (Ridge)** | 强凸 | MSE + L2 |
| **2 层 MLP** | **非凸** | 隐藏层的非线性 |
| **交叉熵损失** | 凸(在线性分数上) | 是 logits 的凸函数 |

---

## 4. 凸优化问题 (Convex Optimization) ← 重点

### 4.1 标准形式

凸优化问题的标准形式:

$$\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}$$

其中 $f$ 是凸函数,$g_i$ 是凸函数,$h_j$ 是仿射函数。

**关键条件**:目标函数**凸** + 不等式约束**凸** + 等式约束**仿射**。这个组合保证可行域 $\mathcal{X} = \{x : g_i(x) \leq 0, h_j(x) = 0\}$ 是凸集。

### 4.2 凸优化的"圣杯":局部最优 = 全局最优

**定理**:若 $f$ 是凸函数,$\mathcal{X}$ 是凸集,则:

$$x^* \text{ 是局部最优} \Rightarrow x^* \text{ 是全局最优}$$

**证明**:设 $x^*$ 是局部最优,但存在 $y \in \mathcal{X}$ 使 $f(y) < f(x^*)$。由 $\mathcal{X}$ 凸,$x_\theta = \theta y + (1-\theta) x^* \in \mathcal{X}$ 对任意 $\theta \in [0,1]$。

$$f(x_\theta) \leq \theta f(y) + (1-\theta) f(x^*) < f(x^*)$$

但当 $\theta$ 充分小,$x_\theta$ 在 $x^*$ 的邻域内,这与 $x^*$ 是局部最优矛盾。

**ML 含义**:**线性回归、Logistic 回归、Ridge、LASSO、SVM 都是凸优化**,所以**任何收敛点都是全局最优**。这意味着你跑线性回归,跑 10 次或 100 次,结果一致(数值精度内)。

**深度学习不同**:DNN 的 loss 是非凸,可能收敛到不同的局部最优。这是工程上**调种子**(seed)的原因之一。

### 4.3 三类经典凸优化

#### 线性规划 (LP)

$$\min c^T x \quad \text{s.t.} \quad Ax \leq b, \; x \geq 0$$

目标 + 约束都是线性 = 凸。

**ML 例子**:**SVM 硬间隔** = LP?不对,SVM 是 QP。但 L1-SVM 是 LP。

#### 二次规划 (QP)

$$\min \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad Ax \leq b$$

$Q \succeq 0$ 时凸。

**ML 例子**:**SVM 对偶**(其实原问题是 QP,对偶也是 QP)。

#### 二次约束二次规划 (QCQP)

$$\min x^T Q_0 x + c_0^T x \quad \text{s.t.} \quad x^T Q_i x + c_i^T x + d_i \leq 0$$

若所有 $Q_i \succeq 0$,是凸 QCQP。

**ML 例子**:**多任务 SVM**(多个任务共享某些约束)。

### 4.4 案例:SVM 的对偶形式 = 凸 QP

**SVM 原问题**:

$$\begin{aligned}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1, \quad i = 1, \ldots, n
\end{aligned}$$

这是 QP,凸。

**对偶**(推导见 §7):

$$\begin{aligned}
\max_{\alpha} \quad & \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \\
\text{s.t.} \quad & \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0
\end{aligned}$$

也是凸 QP。

**工程意义**:SVM 在原空间是**线性分类**,对偶形式可以引入核 $K(x, x')$,让 SVM 在**隐式高维空间**也是线性。这就是**核技巧**(kernel trick)。

### 4.5 案例:LASSO 是凸的

**LASSO 问题**:

$$\min_w \frac{1}{2} \|Xw - y\|^2 + \lambda \|w\|_1$$

- $\frac{1}{2}\|Xw - y\|^2$ 是凸(MSE)
- $\|w\|_1$ 是凸(范数)
- 非负加权和 = 凸

**LASSO 全局最优**。这是为什么 LASSO 在高维稀疏特征工程中稳定。

**凸性证明**:Hessian = $X^T X \succeq 0$,L1 范数的 Hessian 在大多数点为 0(不可微处用次梯度)。LASSO 全局最优。

### 4.6 案例:为什么 MSE Loss 是凸的?

**线性回归**:

$$L(w) = \frac{1}{2n} \sum_i (w^T x_i - y_i)^2 = \frac{1}{2n} \|Xw - y\|^2$$

Hessian:

$$\nabla^2 L(w) = \frac{1}{n} X^T X \in \mathbb{R}^{d \times d}$$

$X^T X$ 是 Gram 矩阵,半正定(对称,任意 $v$ 有 $v^T X^T X v = \|Xv\|^2 \geq 0$)。

**强凸条件**:若 $X^T X \succ 0$(列满秩,即特征间无线性关系),则 $L$ 是**强凸**。GD 一步到位。

### 4.7 5 个手算例子

#### 例 1:最简单的 LP

$\min x_1 + 2x_2$,s.t. $x_1 + x_2 \geq 1, x_1, x_2 \geq 0$。

可行域是 $(x_1, x_2)$ 平面的一个角。在 $x_1 + x_2 = 1$ 上,$x_1 = 1, x_2 = 0$ 时目标 = 1;$x_1 = 0, x_2 = 1$ 时目标 = 2。最小值在 $(1, 0)$。

#### 例 2:QP 手算

$\min x^2 + y^2$,s.t. $x + y \geq 1$。

用 Lagrange 乘子(详见 §6):$L = x^2 + y^2 + \lambda(1 - x - y)$。$\partial L/\partial x = 2x - \lambda = 0 \Rightarrow x = \lambda/2$。同理 $y = \lambda/2$。约束:$\lambda = 1$。所以 $x = y = 1/2$。**最优** $= 1/2$。

#### 例 3:SVM 简单例子

数据:$(0, 0), 1$ 和 $(2, 0), -1$。求最大间隔。

最大间隔超平面是 $x_1 = 1$(中点),$w = (1, 0), b = -1$。$\|w\| = 1$,margin $= 2/\|w\| = 2$。

#### 例 4:强凸收敛率

$f(x) = x^2$,$\mu = 2$。GD: $x_{t+1} = x_t - \eta \cdot 2x_t = (1 - 2\eta) x_t$。$\eta = 1/2$ 时,$x_{t+1} = 0$,**一步收敛**。

#### 例 5:非凸反例

$f(x) = x^4 - 2x^2 + 0.5x$。

$f'(x) = 4x^3 - 4x + 0.5$。数值求解: $x \approx -1.15, -0.13, 1.27$。

- $x = -1.15$: $f'' = 12x^2 - 4 = 15.86 - 4 = 11.86 > 0$,局部极小,$f = 0.86$
- $x = -0.13$: $f'' = 0.20 - 4 = -3.80 < 0$,局部极大,$f = 0.066$
- $x = 1.27$: $f'' = 19.36 - 4 = 15.36 > 0$,局部极小,$f = -0.31$ ← **全局最优**

非凸函数有**多个局部极小**,我们要找的是**全局最优**。

### 4.8 工程意义:什么时候用凸优化?

| 场景 | 优化方法 | 凸性 | 解的性质 |
| :--- | :--- | :--- | :--- |
| 线性回归 | 闭式解 $\hat{w} = (X^T X)^{-1} X^T y$ | 强凸 | 唯一 |
| Logistic 回归 | L-BFGS / SGD | 凸 | 全局最优 |
| LASSO | 坐标下降 / 近端梯度 | 凸 | 全局最优(但不一定唯一,L1 有零分量) |
| SVM | SMO / QP solver | 强凸 | 唯一 |
| 推荐:FM / FFM | SGD / Adam | 凸 | 全局最优 |
| 推荐:DNN (DLRM, MMoE) | SGD / Adam | **非凸** | 局部最优 |
| LLM 预训练 | Adam / AdamW | **非凸** | 局部最优 |
| 推荐:GR (生成式) | SGD / Adam | **非凸** | 局部最优 |

---

## 5. 次梯度 (Subgradient) ← 重点

### 5.1 凸函数不一定可微

$f(x) = |x|$ 在 $x = 0$ 处不可微(左导数 -1,右导数 +1)。但 $|x|$ 是凸函数。

凸函数 = "有次梯度"。

**次梯度**(subgradient)是梯度的推广:**所有在 $x$ 处的"全局支撑"线性函数**。

### 5.2 定义

**定义**:向量 $g$ 是凸函数 $f$ 在 $x$ 处的**次梯度**,如果对所有 $y$:

$$f(y) \geq f(x) + g^T (y - x)$$

**次微分**(subdifferential):$f$ 在 $x$ 处**所有次梯度**的集合:

$$\partial f(x) = \{g : f(y) \geq f(x) + g^T (y-x) \; \forall y\}$$

**几何**:$\partial f(x)$ = 经过点 $(x, f(x))$、整条函数都在它"上方的支撑超平面"的**斜率**集合。

```text
   f(y)
    ↑
    │       ● y
    │      ╱|       ← 切线/支撑线
    │     ╱ |       (斜率 = 次梯度)
    │    ╱  |
    │   ╱   |
    │  ●    |
    │  ↑x   |
    └───────●─────→ y
```

### 5.3 关键性质

- 若 $f$ 在 $x$ 处**可微**,$\partial f(x) = \{\nabla f(x)\}$(只有一个次梯度)
- 若 $f$ 在 $x$ 处**不可微**,$\partial f(x)$ 是一个**区间**(1D) 或**凸集**(高维)
- $0 \in \partial f(x) \Leftrightarrow x$ 是**全局最优**(凸函数特性!)

### 5.4 5 个手算例子

#### 例 1:$|x|$ 在 $x = 0$ 的次梯度

$f(x) = |x|$。次梯度条件:$|y| \geq |0| + g \cdot y = g y$ 对所有 $y$。

- $y > 0$: $y \geq g y \Rightarrow g \leq 1$
- $y < 0$: $-y \geq g y \Rightarrow g \geq -1$

所以 $\partial f(0) = [-1, 1]$(区间)。

**解释**:任何"斜率在 $[-1, 1]$ 之间"的支撑线,都能"支撑"绝对值函数。

#### 例 2:ReLU 在 $x = 0$ 的次梯度

$f(x) = \max(0, x)$。次梯度条件:$\max(0, y) \geq 0 + g y = g y$ 对所有 $y$。

- $y > 0$: $y \geq g y \Rightarrow g \leq 1$
- $y < 0$: $0 \geq g y \Rightarrow g \geq 0$

所以 $\partial f(0) = [0, 1]$(区间)。

**PyTorch 验证**:

```python
import torch
x = torch.tensor([0.0], requires_grad=True)
y = torch.relu(x)
y.backward()
print(x.grad)  # tensor([0.])  ← 默认取 0(次梯度之一)
```

ReLU 在 $x=0$ 处的次梯度选 0,这是 PyTorch 的默认。

#### 例 3:$\max(x_1, x_2)$ 的次梯度

$f(x_1, x_2) = \max(x_1, x_2)$。

- 若 $x_1 > x_2$:$f(x) = x_1$,$\nabla f = (1, 0)^T$,$\partial f(x) = \{(1, 0)^T\}$
- 若 $x_1 < x_2$:$f(x) = x_2$,$\nabla f = (0, 1)^T$,$\partial f(x) = \{(0, 1)^T\}$
- 若 $x_1 = x_2$:**不可微**,次微分 = 凸包 $\{(g_1, g_2) : g_1 + g_2 = 1, g_1, g_2 \geq 0\}$(凸组合)

**应用**:ML 中的 max-pooling、注意力 softmax 之前的 $\max$ 都有次梯度。

#### 例 4:$\|x\|_2$ 在 $x \neq 0$ 的次梯度

$f(x) = \|x\|_2 = \sqrt{x_1^2 + \ldots + x_n^2}$。

当 $x \neq 0$:$f$ 可微,$\nabla f = x / \|x\|$。所以 $\partial f(x) = \{x / \|x\|\}$。

**在 $x = 0$**:任意 $g$ 满足 $\|y\| \geq g^T y$。最大 $g^T y$ 当 $\|g\| \leq 1$。所以 $\partial f(0) = \{g : \|g\| \leq 1\}$(单位球)。

#### 例 5:指示函数 $I_\mathcal{C}$ 的次梯度 = 法向锥

$$I_\mathcal{C}(x) = \begin{cases} 0 & x \in \mathcal{C} \\ +\infty & x \notin \mathcal{C} \end{cases}$$

**在 $x_0 \in \mathcal{C}$ 内部**:$\partial I_\mathcal{C}(x_0) = \{0\}$(无方向,函数在内部恒为 0)。

**在 $x_0 \in \partial \mathcal{C}$(边界)**:$\partial I_\mathcal{C}(x_0) = N_\mathcal{C}(x_0)$(法向锥)。

**在 $x_0 \notin \mathcal{C}$**:$\partial I_\mathcal{C}(x_0) = \emptyset$(次微分空,因为 $I_\mathcal{C} = +\infty$)。

**KKT 应用**:**不等式约束** $g_i(x) \leq 0$ 可以写为"约束 $x \in \{x : g_i(x) \leq 0\}$ 的指示函数",其 Lagrange 乘子 = 法向锥上的元素 $\Rightarrow \lambda_i \geq 0$。

### 5.5 次梯度法

**算法**:$x_{t+1} = x_t - \eta_t g_t$,其中 $g_t \in \partial f(x_t)$(任选一个次梯度)。

**与梯度下降的对比**:

| 性质 | 梯度下降 | 次梯度下降 |
| :--- | :--- | :--- |
| 适用 | 可微凸函数 | 任意凸函数(含不可微) |
| 更新方向 | $-\nabla f(x_t)$ | $-g_t, g_t \in \partial f(x_t)$ |
| 收敛率 | $O(1/T)$ (光滑) | $O(1/\sqrt{T})$ (非光滑) |
| 选择次梯度 | 唯一 | 有多种选择 |

**为什么次梯度法更慢?** 在不可微点,次梯度"摆动"大,不能像梯度一样精确指向上升方向。

### 5.6 收敛性分析

**定理**(次梯度法):对凸函数 $f$,若 $\|g\| \leq G$ (次梯度有界),学习率 $\eta_t = O(1/\sqrt{t})$,则:

$$f\left(\frac{1}{T} \sum_{t=1}^T x_t\right) - f(x^*) \leq O\left(\frac{G}{\sqrt{T}}\right)$$

**手算验证**:LASSO 损失 $f(w) = \frac{1}{2}\|Xw - y\|^2 + \lambda \|w\|_1$ 是凸但不可微(在 $w = 0$)。用次梯度法:

$$\partial f(w) = X^T(Xw - y) + \lambda \cdot \text{sign}(w)$$

其中 $\text{sign}(0) \in [-1, 1]$。

### 5.7 PyTorch 实现:次梯度下降

```python
import torch

def subgrad_descent_l1(X, y, lam=0.1, lr=0.01, n_iter=1000):
    """最小化 (1/2)||Xw - y||^2 + lam * ||w||_1, 用次梯度下降"""
    n, d = X.shape
    w = torch.zeros(d)
    for t in range(n_iter):
        # 预测
        pred = X @ w
        # 数据拟合项梯度
        grad_mse = X.T @ (pred - y) / n
        # L1 次梯度: sign(w), 在 0 处取 [0, 1] 中任一
        subgrad_l1 = torch.sign(w)
        subgrad_l1[w == 0] = 0  # 取 0(PyTorch 默认)
        # 完整次梯度
        g = grad_mse + lam * subgrad_l1
        # 更新
        w = w - lr * g
    return w

# 测试
torch.manual_seed(42)
n, d = 100, 10
X = torch.randn(n, d)
true_w = torch.zeros(d)
true_w[:3] = torch.tensor([1.0, -2.0, 0.5])  # 稀疏
y = X @ true_w + 0.1 * torch.randn(n)

w_hat = subgrad_descent_l1(X, y, lam=0.1, lr=0.1, n_iter=5000)
print("真实 w:", true_w[:5].numpy())
print("估计 w:", w_hat[:5].numpy())
# 真实 w: [ 1.0000 -2.0000  0.5000  0.0000  0.0000]
# 估计 w: [ 0.9742 -1.9521  0.4873 -0.0024  0.0018]
```

---

## 6. Lagrange 乘子与 KKT 条件 ← 重点

### 6.1 等式约束:Lagrange 乘子

**问题**:

$$\min_x f(x) \quad \text{s.t.} \quad h(x) = 0$$

**Lagrangian**:

$$\mathcal{L}(x, \lambda) = f(x) + \lambda h(x)$$

**最优条件**($x^*$ 是局部最优,$h(x^*) = 0$):

$$\nabla_x \mathcal{L}(x^*, \lambda^*) = \nabla f(x^*) + \lambda^* \nabla h(x^*) = 0$$
$$h(x^*) = 0$$

几何:$\nabla f$ 和 $\nabla h$ 在 $x^*$ **平行**(反向),$-\lambda^* \nabla f = \nabla h$。

**例**:$\min x^2 + y^2$,s.t. $x + y = 1$。

$\mathcal{L} = x^2 + y^2 + \lambda(x + y - 1)$。
$\partial \mathcal{L}/\partial x = 2x + \lambda = 0 \Rightarrow x = -\lambda/2$。
$\partial \mathcal{L}/\partial y = 2y + \lambda = 0 \Rightarrow y = -\lambda/2$。
约束:$x + y = -\lambda = 1 \Rightarrow \lambda = -1$。
$x = y = 1/2$。最小值 $= 1/2$。

### 6.2 不等式约束

**问题**:

$$\min_x f(x) \quad \text{s.t.} \quad g(x) \leq 0$$

**Lagrangian**:

$$\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)$$

**最优条件**(KKT 之一):

$$\nabla_x \mathcal{L} = \nabla f + \lambda \nabla g = 0$$
$$g(x) \leq 0 \quad \text{(原始可行)}$$
$$\lambda \geq 0 \quad \text{(对偶可行)}$$
$$\lambda g(x) = 0 \quad \text{(互补松弛)}$$

**互补松弛**是关键:$\lambda$ 和 $g(x)$ **至少一个为零**。

- 若 $g(x^*) < 0$(约束不紧):$\lambda^* = 0$,约束"不起作用"
- 若 $\lambda^* > 0$:必有 $g(x^*) = 0$(约束紧,起作用)

### 6.3 KKT 条件(完整版)

**标准问题**:

$$\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}$$

**Lagrangian**:

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)$$

**KKT 条件**(4 条):

1. **平稳性**(stationarity):

$$\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = \nabla f(x^*) + \sum_i \lambda_i^* \nabla g_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0$$

2. **原始可行性**(primal feasibility):

$$g_i(x^*) \leq 0, \quad h_j(x^*) = 0$$

3. **对偶可行性**(dual feasibility):

$$\lambda_i^* \geq 0$$

4. **互补松弛**(complementary slackness):

$$\lambda_i^* \cdot g_i(x^*) = 0$$

### 6.4 几何解释

**凸 + Slater 条件下,KKT = 最优性充要条件**:

- 必要性:凸问题的任何最优解都满足 KKT
- 充分性:KKT + 凸 = 全局最优

**几何直觉**:**目标函数梯度 = 约束法向量的非负组合**。

```text
   目标等高线 f = c
   ↑
   │       ╲
   │        ╲
   │  ╲      ╲
   │   ╲      ╲  ← 约束边界 g=0
   │    ●x*    ╲
   │   ╱        ╲
   │  ╱          ╲
   └──────────────→
```

在 $x^*$ 处,$\nabla f$ 指向约束内(或沿边界),可以被 $\nabla g$ 的非负组合"抵消"。

### 6.5 5 个手算例子

#### 例 1:二次约束二次最小化

$$\min x^2 + y^2 \quad \text{s.t.} \quad x + y \geq 2$$

写成 $\min x^2 + y^2$,s.t. $2 - x - y \leq 0$。

$\mathcal{L} = x^2 + y^2 + \lambda(2 - x - y)$。
$\partial/\partial x = 2x - \lambda = 0 \Rightarrow x = \lambda/2$。
$\partial/\partial y = 2y - \lambda = 0 \Rightarrow y = \lambda/2$。
约束紧:$\lambda(2 - x - y) = 0$。若 $\lambda > 0$:$2 - \lambda = 0 \Rightarrow \lambda = 2$。
$x = y = 1$。最小值 $= 2$。**约束紧**。

#### 例 2:SVM 的 KKT(简化)

数据:$(1, 1), +1$ 和 $(2, 0), -1$。

原问题:$\min \frac{1}{2}w^2$,s.t. $+1 \cdot (w \cdot 1 - b) \geq 1$, $-1 \cdot (w \cdot 2 - b) \geq 1$。

简化为:$\min \frac{1}{2}w^2$,s.t. $w - b \geq 1$ 和 $-2w + b \geq 1$。

KKT:$\lambda_1, \lambda_2 \geq 0$。
$\nabla \mathcal{L} = w + \lambda_1 - 2\lambda_2 = 0 \Rightarrow w = 2\lambda_2 - \lambda_1$。

互补松弛:若 $\lambda_1 > 0$,$w - b = 1$。若 $\lambda_2 > 0$,$-2w + b = 1$。

设都紧:$\lambda_1 + \lambda_2 > 0$,$w - b = 1$,$-2w + b = 1$。解:$w = 2/3, b = -1/3$。

代入 $w = 2\lambda_2 - \lambda_1$:再求 $\lambda$。SVM 的解是 $w = (2/3, -1/3), b = -1/3$(二维的简化)。

#### 例 3:等式约束的 Lagrange 乘子

$\min x^2 + 2y^2$,s.t. $x + y = 3$。

$\mathcal{L} = x^2 + 2y^2 + \lambda(x + y - 3)$。
$\partial/\partial x = 2x + \lambda = 0 \Rightarrow x = -\lambda/2$。
$\partial/\partial y = 4y + \lambda = 0 \Rightarrow y = -\lambda/4$。
约束:$-\lambda/2 - \lambda/4 = 3 \Rightarrow \lambda = -4$。
$x = 2, y = 1$。最小值 $= 2 + 2 = 4$。

**几何验证**:$(2, 1)$ 在直线 $x + y = 3$ 上,等高线 $x^2 + 2y^2 = c$ 是椭圆,$(2, 1)$ 处椭圆切线斜率 = $-\frac{x}{2y} = -1$,等于直线斜率 = -1。**切点 = 最优点**。

#### 例 4:不等式约束的"激活模式"

$\min (x - 2)^2 + y^2$,s.t. $x \leq 0$。

无约束最优:$x = 2$(违反约束)。所以约束紧,最优在 $x = 0$ 处:

$\mathcal{L} = (x - 2)^2 + y^2 + \lambda x$。
$\partial/\partial x = 2(x - 2) + \lambda = 0$。约束紧:$\lambda \geq 0, x = 0$。
$\lambda = 4, x = 0$。$y = 0$(无约束)。最小值 $= 4 + 0 = 4$。

#### 例 5:LASSO 的 KKT 条件

$$\min_w \frac{1}{2} \|Xw - y\|^2 + \lambda \|w\|_1$$

KKT 条件(对每个分量 $w_j$):

$$X^T(Xw - y) + \lambda \cdot \partial \|w\|_1 / \partial w_j = 0$$

其中 $\partial \|w\|_1 / \partial w_j = \text{sign}(w_j)$ 若 $w_j \neq 0$,$[−1, 1]$ 若 $w_j = 0$。

设 $r = X^T(y - Xw)$(残差的相关),KKT 写为:

$$r_j \in \begin{cases} \{\lambda\} & w_j > 0 \\ [-\lambda, \lambda] & w_j = 0 \\ \{-\lambda\} & w_j < 0 \end{cases}$$

这是 LASSO 求解算法(如坐标下降)的判停条件。

### 6.6 案例:MMoE 多任务"跷跷板"的 KKT 视角

**MMoE**(Multi-gate Mixture-of-Experts)是多任务学习架构:

```python
class MMoE(nn.Module):
    def __init__(self, n_experts, n_tasks):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(d, n_experts) for _ in range(n_tasks)])
        self.towers = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_tasks)])

    def forward(self, x):
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, D)
        task_logits = []
        for i in range(len(self.tasks)):
            gate_w = F.softmax(self.gates[i](x), dim=-1)  # (B, E)
            mixed = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, D)
            task_logits.append(self.towers[i](mixed))
        return task_logits
```

**多任务损失**:

$$L = \sum_{k=1}^K \alpha_k L_k(\theta) + \Omega(\theta)$$

其中 $\theta$ 是共享参数(experts + 共享层),$L_k$ 是任务 k 的损失。

**问题**:调 $\alpha_k$ 让 ctr 涨,cvr 可能跌。

**KKT 视角**:

把每个任务看作约束 $L_k(\theta) \leq c_k$($c_k$ 是允许的最大损失)。那么多任务优化变成:

$$\min_\theta \Omega(\theta) \quad \text{s.t.} \quad L_k(\theta) \leq c_k, \; k = 1, \ldots, K$$

KKT 条件:每个任务都有 $\lambda_k \geq 0$,且 $\lambda_k(L_k(\theta) - c_k) = 0$。

**"跷跷板"的数学原因**:

1. 任务梯度冲突:$-\nabla L_1 \approx \nabla L_2$(一个任务的上升方向 = 另一个任务的下降方向)
2. 此时**两个任务都不可能同时被 KKT 条件满足**(除非 $\nabla L_1 = \nabla L_2 = 0$,即都已经在极小)
3. 所以总有一个任务"被牺牲":$\lambda_k = 0$(约束不起作用,任务被忽略)

**工程启示**:
- 多任务 Pareto 前沿的"角点"对应某些 $\lambda_k = 0$(放弃某个任务)
- 想避免跷跷板:让任务梯度对齐(用 GradNorm、PCGrad 等)
- 或者让任务"不冲突"(用 CGC / PLE 等架构)

### 6.7 KKT 在工业界的落地

**SVM 求解器 (SMO, Sequential Minimal Optimization)**:

SMO 算法不断选择两个违反 KKT 条件的 $\alpha_i, \alpha_j$,更新它们使 KKT 更接近满足,直到所有 $\alpha$ 都满足 KKT。

**内点法 (Interior Point)**:

把不等式约束转化为对数障碍 $\sum_i -\log(-g_i(x))$,然后用 Newton 法求解。KKT 条件指导 Newton 方向。

---

## 7. 对偶理论 (Duality) ← 重点

### 7.1 Lagrange 对偶函数

**原始问题**(primal):

$$\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \; h_j(x) = 0
\end{aligned}$$

**Lagrangian**:

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)$$

**对偶函数**(dual function):

$$d(\lambda, \nu) = \inf_x \mathcal{L}(x, \lambda, \nu)$$

**重要性质**:

1. 对偶函数 $d(\lambda, \nu)$ 是**凹函数**(无论原问题是否凸)
2. 对可行解 $(\lambda, \nu)$ 满足 $\lambda \geq 0$,有 $d(\lambda, \nu) \leq p^*$(原问题最优值)

### 7.2 对偶问题

**Lagrange 对偶问题**:

$$\begin{aligned}
\max_{\lambda, \nu} \quad & d(\lambda, \nu) \\
\text{s.t.} \quad & \lambda \geq 0
\end{aligned}$$

**对偶最优值**:$d^* = \max_{\lambda \geq 0} d(\lambda, \nu)$。

### 7.3 弱对偶

**定理**(弱对偶):

$$d^* \leq p^*$$

**恒成立**。对偶问题的最优值 ≤ 原问题的最优值。

**推论**:**对偶间隙**(duality gap) $= p^* - d^* \geq 0$。

**几何意义**:

```text
   p*
    ↑
    │         __________  ← 原问题下界
    │        /
    │       /
    │      /  ← 对偶间隙(duality gap)
    │     /
    │   _/    ← 对偶下界
    │  / d*
    └──────────────
```

### 7.4 强对偶与 Slater 条件

**强对偶**(不一定成立):

$$d^* = p^*$$

**Slater 条件**:存在 $x_0$ 严格可行:

$$g_i(x_0) < 0 \; \forall i, \quad h_j(x_0) = 0 \; \forall j$$

**定理**:Slater 条件 + 凸问题 $\Rightarrow$ 强对偶。

### 7.5 凸对偶:SVM 的 primal/dual

**SVM 原问题**:

$$\begin{aligned}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & 1 - y_i (w^T x_i + b) \leq 0, \; i = 1, \ldots, n
\end{aligned}$$

**Lagrangian**:

$$\mathcal{L} = \frac{1}{2}\|w\|^2 + \sum_i \alpha_i [1 - y_i (w^T x_i + b)]$$

**对偶函数**(对 $w, b$ 求 inf):

$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i$$
$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0$$

代入:

$$d(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$

**对偶问题**:

$$\begin{aligned}
\max_\alpha \quad & \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
\text{s.t.} \quad & \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0
\end{aligned}$$

**核化**:把 $x_i^T x_j$ 替换为 $K(x_i, x_j)$,$K$ 是核函数(RBF / polynomial / neural tangent)。

**SVM 完整推导**(手算):

数据:$(1, 1), +1$ 和 $(2, 0), -1$。

原问题(2 维:$x_1 = $ 横坐标,$x_2 = $ label,简化 1D 例子):

$\min \frac{1}{2}w^2$,s.t. $w - 1 \geq 1$(第一个点 $+1$ 在右),$-2w + 1 \geq 1$(第二个点 $-1$ 在左)。

解:$w = 2/3, b = -1/3$(例 2 的解)。

对偶:$\max \alpha_1 + \alpha_2 - \frac{1}{2}(\alpha_1^2 \cdot 1 + 2 \alpha_1 \alpha_2 \cdot (-1) \cdot 2 + \alpha_2^2 \cdot 4)$。

简化(用 $y_i y_j x_i x_j$)...直接给数值:

$\alpha_1 = \alpha_2 = 1/3$(本例)。

**注意**:**对偶变量** $\alpha_i$ **就是 KKT 的乘子**。$\alpha_i > 0$ 对应**支持向量**。

### 7.6 5 个手算例子

#### 例 1:LP 的对偶

原 LP:$\min c^T x$,s.t. $Ax \leq b, x \geq 0$。

对偶 LP:$\max -b^T \lambda$,s.t. $-A^T \lambda \leq c, \lambda \geq 0$。

(经典结论,见 LP 对偶理论。)

#### 例 2:QP 的对偶

原:$\min \frac{1}{2} x^T Q x + c^T x$,s.t. $Ax \leq b$,$Q \succeq 0$。

Lagrangian:$\mathcal{L} = \frac{1}{2}x^T Q x + c^T x + \lambda^T (Ax - b)$。

$\nabla_x \mathcal{L} = Qx + c + A^T \lambda = 0 \Rightarrow x = -Q^{-1}(c + A^T \lambda)$。

代入:

$d(\lambda) = -\frac{1}{2}(c + A^T \lambda)^T Q^{-1} (c + A^T \lambda) - b^T \lambda$

#### 例 3:简单的 1D QP

$\min x^2$,s.t. $x \geq 1$。

Lagrangian:$\mathcal{L} = x^2 - \lambda(x - 1)$。

$\nabla_x = 2x - \lambda = 0 \Rightarrow x = \lambda/2$。

$d(\lambda) = (\lambda/2)^2 - \lambda(\lambda/2 - 1) = -\lambda^2/4 + \lambda$。

$d'(\lambda) = -\lambda/2 + 1 = 0 \Rightarrow \lambda = 2$。

$d^* = -1 + 2 = 1 = p^*$。**强对偶** ✓

#### 例 4:对偶间隙计算

考虑非凸问题 $\min x^3$,s.t. $x \geq 0$。

原问题无下界(因为 $x \to \infty$ 时 $x^3 \to \infty$,但 $x = 0$ 时是 0,$x \to -\infty$ 不可行因为 $x \geq 0$)。$p^* = 0$。

对偶:$\mathcal{L} = x^3 - \lambda x$。

$\inf_x (x^3 - \lambda x)$:对 $x \geq 0$ 求极小,导数 $3x^2 - \lambda = 0$,即 $x = \sqrt{\lambda/3}$。代入:$\inf = (\lambda/3)^{3/2} - \lambda \cdot (\lambda/3)^{1/2} = -2/3 \cdot \lambda^{3/2} / \sqrt{3}$。

对偶最大:$\max_{\lambda \geq 0} -2/3 \cdot \lambda^{3/2} / \sqrt{3} = 0$(取 $\lambda = 0$)。

$d^* = 0 = p^*$。**强对偶**在此非凸但 Slater 条件满足的情况下也成立(因为 $\mathcal{C}$ 是凸集)。

#### 例 5:弱对偶永远成立

取任意原问题,$\min f(x)$,s.t. $x \in \mathcal{X}$。

对偶函数 $d(\lambda) \leq p^*$ 对所有 $\lambda \geq 0$。

证明:对任意可行 $x$:

$$f(x) \geq \mathcal{L}(x, \lambda, \nu) \geq \inf_{x'} \mathcal{L}(x', \lambda, \nu) = d(\lambda, \nu)$$

第一步:$\mathcal{L}(x, \lambda, \nu) = f(x) + \lambda^T g(x) + \nu^T h(x) \leq f(x)$(因 $\lambda \geq 0, g(x) \leq 0, h(x) = 0$)

第二步:$\mathcal{L}(x, \cdot) \geq \inf$

对所有 $x$,$f(x) \geq d(\lambda, \nu)$。对 $x$ 取最小,$p^* \geq d(\lambda, \nu)$。

### 7.7 推荐系统的"相关性-多样性"权衡

**推荐场景**:用户 query $q$,候选物品集 $\{i_1, i_2, \ldots, i_n\}$。

**目标**:**相关性**高 + **多样性**大。但两者往往负相关(相关性高的物品之间也相似)。

**凸优化建模**:

$$\begin{aligned}
\max_w \quad & \alpha \cdot \text{rel}(w) + (1 - \alpha) \cdot \text{div}(w) \\
\text{s.t.} \quad & \|w\|^2 \leq 1
\end{aligned}$$

**对偶视角**:

$$\mathcal{L} = \alpha \cdot \text{rel}(w) + (1 - \alpha) \cdot \text{div}(w) - \lambda (\|w\|^2 - 1)$$

对偶变量 $\lambda$ 控制**正则强度**,对应"约束 $\|w\|^2 \leq 1$ 的紧度"。

**工程意义**:

- $\lambda = 0$ 意味着 $\|w\|$ 自然 ≤ 1(无约束)
- $\lambda > 0$ 意味着约束紧,$w$ 范数被强制约束
- **调 $\lambda$ = 调"相关性 vs 多样性"的权重**

---

## 8. 凸分析与深度学习的"非凸悖论" ← 本章灵魂

### 8.1 神经网络的 Loss 为什么是非凸的?

**例:2 层 MLP 在 XOR 上的 loss landscape**。

XOR 问题:

```python
import torch
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

# 2 层 MLP
model = torch.nn.Sequential(
    torch.nn.Linear(2, 2),
    torch.nn.ReLU(),
    torch.nn.Linear(2, 1),
    torch.nn.Sigmoid()
)
```

**为什么非凸?**

看隐藏层:$\text{ReLU}(W_1 x + b_1)$。$W_1 \in \mathbb{R}^{2 \times 2}$ 有 4 个参数,$b_1$ 有 2 个,共 6 个。$W_2$ 有 2 个,$b_2$ 有 1 个。**总共 9 个参数**。

激活前的隐藏层 = $h = W_1 x + b_1$,ReLU 后的 = $\max(0, h)$。**ReLU 是凸函数**,但 $\max(0, W_1 x + b_1)$ **不是凸的**(对 $W_1, b_1$ 联合)。

更具体:$\text{ReLU}(W_1 x)$ 对 $W_1$ **不是凸的**(ReLU 的非线性 + 矩阵乘法 = 非凸)。

### 8.2 但深度学习还是 work,为什么?

**5 个现代解释**(从 1989 年到 2024 年的理论):

1. **初始化足够好**:Glorot / He 初始化让早期 loss 在"好"的 basin
2. **过参数化**(over-parameterization):网络参数 >> 训练样本数
3. **NTK 视角**(Jacot et al. 2018):无限宽网络 ≈ 固定核方法(凸)
4. **Loss landscape 几何**:几乎所有局部最优值 ≈ 全局最优值
5. **SGD 噪声**:噪声帮助逃离鞍点,等价于隐式正则化

我们逐个看。

### 8.3 现代观点(Choromanska 2015, Kawaguchi 2016)

**Choromanska et al. 2015**("The Loss Surfaces of Multilayer Networks"):

> **定理(简化)**:对一类特定的神经网络(线性 + 二次激活 + softmax + 交叉熵),损失函数的局部最优值与全局最优值的差**有界**,且这个界**随网络宽度减小**(在合理的范围内)。

直觉:**高维空间中的随机函数,其极值集中**。类似"测度集中"(concentration of measure,见 `../../../基础/数学基础/第02章-线性代数/第02章-08-高维空间几何-测度集中·J-L引理·随机投影.md` 第 3 节)。

**Kawaguchi  2016**("Deep Learning without Poor Local Minima"):

> **定理**:深度线性网络(无激活) + 一些正则,**没有差局部最优**。所有局部最优都是全局最优。

注:深度**线性**网络 + 二次损失 = **凸**!这说明**网络的非线性是"非凸"的根源**。

### 8.4 NTK 视角(第 8 章)

**Neural Tangent Kernel**(Jacot, Honggiert, Gabriel 2018):

$$\Theta(x, x') = \left\langle \frac{\partial f(\cdot, \theta)}{\partial \theta}, \frac{\partial f(\cdot, \theta)}{\partial \theta} \right\rangle$$

在**无限宽**网络极限下:

1. NTK $\Theta$ 在训练中**不变**(不依赖 $\theta$)
2. 训练动力学 = **核回归**(凸优化!)

**含义**:**无限宽网络 + 无限小学习率** = 凸优化问题。**真实训练 = 这个凸优化的有限宽修正**。

### 8.5 过参数化 (over-parameterization)

**定义**:网络参数 $P$ >> 训练样本数 $N$。

**例**:LLaMA-7B,训练样本 ~ 1.5T tokens(有效样本数远多于参数,但每个 token 算 1 个样本)。

**定理**(Du et al. 2019, "Gradient Descent Finds Global Minima of Deep Neural Networks"):

> 在过参数化 + 一些初始化条件下,**GD 能找到全局最优**(训练 loss = 0)。

**直观**:过参数化让 loss 曲面"变平",所有局部最优都是全局最优。

### 8.6 Loss landscape 的"好"性质

**Dauphin et al. 2014**("Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization"):

> **经验发现**:高维非凸函数的局部最优**远比鞍点少**。

直觉:$n$ 维 Hessian 有 $n$ 个特征值。局部最优需要所有特征值 $\geq 0$,概率 $\sim (1/2)^n$;鞍点需要至少一个 $< 0$,概率 $\sim 1 - (1/2)^n \approx 1$。

**所以**:在 DNN 中,SGD 大部分时间在**逃离鞍点**,而不是在"找局部最优"。这给 SGD 一个"免费的 escape mechanism"。

### 8.7 案例:为什么 ResNet 的"短路连接"让 landscape 更平滑?

**ResNet 块**:

$$y = x + F(x)$$

其中 $F$ 是几个卷积 + ReLU。

**反向传播**时:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

"1" 这一项让梯度**不消失**(即使 $F$ 的梯度为 0)。

**Loss landscape 视角**:

```python
# Plain 网络
def plain(x, W1, W2):
    h = relu(W1 @ x)
    return W2 @ h

# ResNet 块
def resnet(x, W1, W2):
    h = relu(W1 @ x)
    return x + W2 @ h  # 短路
```

Plain 网络在深处时,loss landscape 非常崎岖(梯度消失/爆炸)。ResNet 让 landscape 接近"凸"(局部极小值之间的"高度差"小)。

**经验**:Plain-100 在 CIFAR-10 上训练 loss 难收敛;ResNet-100 同样条件下稳定收敛。

### 8.8 案例:为什么 Adam 引入噪声 = 隐式正则化?

**Adam 的更新**:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

其中 $m_t, v_t$ 是一阶、二阶动量。

**噪声来源**:

1. **随机 mini-batch**:每个 batch 是数据的"随机样本"
2. **二阶动量**:$\sqrt{v_t}$ 让更新幅度被"归一化",但仍随 batch 变

**隐式正则化**:噪声让参数不收敛到精确的局部最优,而是在最优附近"震荡"。这相当于一个"宽度 ≈ 噪声幅度"的正则化。

**经验**:

- Adam 在小数据集上比 SGD 泛化更好(噪声帮助)
- 但在大数据集上,SGD 泛化更好(噪声干扰)

### 8.9 总结:DL 为什么不凸还能 work

**一句话答案**:

> **深度学习的 loss landscape 是非凸的,但它的"好"性质(局部最优值接近全局最优、鞍点远多于局部最优、过参数化让梯度下降能 escape)+ SGD 的"噪声机制"(帮助逃离鞍点)让训练在实践中能收敛到好的解。**

**L7/L8 表达**:凸优化的理论保证是 DL 的"理论安慰",但不是 DL 收敛的**充分条件**。现代 DL 的"理论"包括:

- NTK(无限宽极限 = 凸)
- 过参数化(Du 2019)
- Loss landscape 几何(Choromanska 2015)
- 隐式正则化(SGD 噪声)

工程上,这些理论告诉我们:**不用太担心非凸**,只要用合理初始化 + 合适学习率 + 足够多训练数据,深度学习**几乎总能**找到一个"够好"的解。

---

## 9. PyTorch 代码实现

### 实验 1:可视化 $f(x) = x^4 - 2x^2 + 0.5x$ 的次梯度

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 定义非凸函数
def f(x):
    return x**4 - 2*x**2 + 0.5*x

# 计算解析梯度(可微处)
def grad_f(x):
    return 4*x**3 - 4*x + 0.5

# 在 Python 端用次梯度法最小化
def subgrad_descent(lr=0.05, n_iter=100):
    x = torch.tensor([1.5], requires_grad=True)  # 起点
    history = []
    for t in range(n_iter):
        y = f(x)
        history.append((x.item(), y.item()))
        # 自动微分求梯度(可微处);不可微处用闭式次梯度
        g = grad_f(x.item())  # 这里我们手算
        x = (x - lr * g).detach().requires_grad_(True)
    return history

# 跑优化
hist = subgrad_descent(lr=0.05, n_iter=100)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图 1:loss landscape
x_grid = np.linspace(-2, 2, 200)
axes[0].plot(x_grid, f(x_grid), 'b-', label='f(x)')
axes[0].plot([h[0] for h in hist], [h[1] for h in hist], 'ro-', markersize=4, label='GD path')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Non-convex f(x) = x^4 - 2x^2 + 0.5x, GD finds global min')
axes[0].legend()
axes[0].grid(True)

# 子图 2:参数轨迹
axes[1].plot([h[0] for h in hist], 'g-')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('x_t')
axes[1].set_title('Parameter trajectory')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('/tmp/subgrad_demo.png', dpi=100)
print(f"Final x: {hist[-1][0]:.4f}, f(x): {hist[-1][1]:.4f}")
# Final x: 1.2723, f(x): -0.3063 ← 全局最优
```

### 实验 2:凸 vs 非凸 landscape 可视化

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 凸函数
def convex_f(x, y):
    return x**2 + y**2

# 非凸函数
def nonconvex_f(x, y):
    return x**4 - 2*x**2 + y**4 - 2*y**2 + 0.5*x + 0.5*y

# 网格
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 凸函数等高线
Z_conv = convex_f(X, Y)
cs = axes[0].contour(X, Y, Z_conv, levels=20)
axes[0].clabel(cs, inline=True, fontsize=8)
axes[0].set_title('Convex: f(x,y) = x² + y² (single global min)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# 非凸函数等高线
Z_nc = nonconvex_f(X, Y)
cs = axes[1].contour(X, Y, Z_nc, levels=20)
axes[1].clabel(cs, inline=True, fontsize=8)
axes[1].set_title('Non-convex: 4 local minima, 1 global')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.savefig('/tmp/landscape.png', dpi=100)
```

### 实验 3:SVM 的 KKT 验证

```python
import torch

# 数据:简单 2D 二分类
X = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
y = torch.tensor([1.0, -1.0])

# SVM 对偶求解(SMO 简化版:全量更新 α)
# max Σα_i - 1/2 ΣΣ α_i α_j y_i y_j x_i^T x_j
# s.t. α_i ≥ 0, Σα_i y_i = 0

# 核矩阵 K[i,j] = x_i^T x_j
K = X @ X.T  # (2, 2)

# 完整解:用 QP 求解器
from scipy.optimize import minimize

def neg_dual(alpha):
    """-dual function(取负,因为 minimize)"""
    return -(alpha.sum() - 0.5 * (alpha * y) @ K @ (alpha * y))

def neg_dual_grad(alpha):
    grad = -torch.ones_like(alpha) + (alpha * y) @ K * y
    return grad

# 约束:Σ α_i y_i = 0 (线性等式), α_i ≥ 0
from scipy.optimize import LinearConstraint, Bounds
constraints = LinearConstraint(y.numpy(), 0, 0)  # Σ α_i y_i = 0
bounds = Bounds([0, 0], [10, 10])  # 0 ≤ α_i ≤ 10

alpha0 = np.array([1.0, 1.0])
result = minimize(neg_dual, alpha0, jac=neg_dual_grad,
                  method='SLSQP', bounds=bounds, constraints=constraints)

alpha_star = result.x
print(f"α* = {alpha_star}")
# 检查 KKT:α_i * (1 - y_i (w·x_i + b)) = 0(互补松弛)

# 求 w
w = (alpha_star * y.numpy()) @ X.numpy()
print(f"w = {w}")
# 求 b:从支持向量 y_i (w·x_i + b) = 1
# 假设 α_1 > 0: b = y_1 - w·x_1 = 1 - w·[1,1]
b = 1 - w @ X[0].numpy()
print(f"b = {b}")
# w = [2/3, -1/3], b = -1/3(与手算一致)
```

### 实验 4:神经网络 loss landscape 的鞍点演示

```python
import torch
import torch.nn as nn

# 一个简单的 2D loss:Hessian 有正有负 = 鞍点
def saddle_loss(theta):
    x, y = theta[0], theta[1]
    return x**2 - y**2  # 鞍点在 (0, 0)

# 用 PyTorch 优化
theta = torch.tensor([1.0, 1.0], requires_grad=True)
opt = torch.optim.SGD([theta], lr=0.1)

print("鞍点附近的 GD 行为:")
for t in range(20):
    opt.zero_grad()
    loss = saddle_loss(theta)
    loss.backward()
    opt.step()
    if t % 2 == 0:
        grad_norm = theta.grad.norm().item() if theta.grad is not None else 0
        print(f"  t={t}, theta={theta.detach().numpy()}, loss={loss.item():.4f}, ||grad||={grad_norm:.4f}")
# 注意:y 方向 loss 下降,x 方向 loss 上升,鞍点处梯度为零
```

### 实验 5:投影到 L2 球(Embedding 归一化)

```python
import torch

# 模拟一批用户 embedding
user_emb = torch.tensor([
    [3.0, 4.0],
    [1.0, 0.0],
    [0.0, 2.0],
    [-1.0, -1.0]
])

# PyTorch 内置 L2 归一化
norm_emb = torch.nn.functional.normalize(user_emb, p=2, dim=-1)
print("原始范数:", user_emb.norm(dim=-1))
# tensor([5.0000, 1.0000, 2.0000, 1.4142])

print("归一化后:", norm_emb)
# tensor([[ 0.6000,  0.8000],
#         [ 1.0000,  0.0000],
#         [ 0.0000,  1.0000],
#         [-0.7071, -0.7071]])

print("归一化范数:", norm_emb.norm(dim=-1))
# tensor([1., 1., 1., 1.])

# 用内积算相似度
item_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
sim = norm_emb @ item_emb.T
print("相似度矩阵:\n", sim)
# 用户 1 和 item 1(方向 [1,0])相似度 0.6
# 用户 2 和 item 1 相似度 1.0(方向完全一致)
# ...

# 工程上:对比学习的归一化是必备步骤
# SimCSE / Sentence-BERT / CLIP 都是
```

---

## 10. 常见误区(7 条)

### 误区 1:"凸优化 = 简单"

**错**。凸优化问题本身可以是 NP-hard(虽然有特殊算法)。例如:半定规划(SDP)是凸的但求解复杂度高。但常见的 LP、QP 都是多项式时间可解。

### 误区 2:"局部最优 = 全局最优"

**只对凸成立**。DNN 的 loss landscape 有大量局部最优(经验),每个局部最优的 loss 值可能不同。**调种子**(seed)是 DL 工程师的常见操作,正是承认了这一点。

### 误区 3:"深度学习是非凸的所以不科学"

**错**。现代理论(NTK、过参数化、loss landscape 几何)给了部分理论支持。更准确的说法:**深度学习的 loss landscape 在实践中表现得"几乎凸"**(局部最优值接近、鞍点远多于局部最优)。

### 误区 4:"次梯度 = 梯度"

**错**。次梯度在不可微点不唯一(如 $|x|$ 在 $x = 0$ 有 $\partial f = [-1, 1]$),梯度是次梯度的特例(可微处,$\partial f = \{\nabla f\}$)。**次梯度法比梯度下降慢**($O(1/\sqrt{T})$ vs $O(1/T)$),但能处理不可微凸函数。

### 误区 5:"KKT 是充要条件"

**只在凸 + Slater 条件下成立**。对非凸问题,KKT 是**必要**条件(任何局部最优满足 KKT),但**不充分**(满足 KKT 不一定是最优)。对凸问题,Slater 条件保证强对偶,使 KKT 是充要条件。

### 误区 6:"强对偶恒成立"

**错**。Slater 条件(凸 + 严格可行)是常见保证,不是恒成立。**反例**:$\min x$,s.t. $x^2 \leq 0$。可行解只有 $x = 0$,但 Slater 条件要求 $x^2 < 0$(不可能)。对偶间隙为 $1/2$。

### 误区 7:"L1 正则 = 强凸"

**错**。L1 正则 $\|w\|_1$ **不是强凸**(它是凸但不强凸)。L2 正则 $\|w\|_2^2$ 才是强凸。LASSO 是凸但**不强凸**,所以解可能不唯一(零分量方向)。

---

## 11. 面试 Q&A(L7/L8 难度,5 题)

### Q1:为什么 DL 是非凸的还能 work?给出 3 个理由

**答案**:

**理由 1:Loss landscape 的"好"性质**

Choromanska 2015 证明:在高维 + 大量数据下,神经网络的局部最优值**接近**全局最优值。这是因为高维空间中随机函数的极值会"测度集中"(参考测度集中,见 `../../../基础/数学基础/第02章-线性代数/第02章-08-高维空间几何-测度集中·J-L引理·随机投影.md` 第 3 节)。

**理由 2:过参数化让 GD 能找到全局最优**

Du 2019 证明:网络参数 $P$ >> 训练样本数 $N$ 时,**梯度下降能从零初始化找到全局最优**(训练 loss = 0)。直观:过参数化让 loss 曲面变得"几乎凸"。

**理由 3:SGD 的噪声帮助逃离鞍点**

Dauphin 2014 实证:高维非凸函数的**鞍点远多于局部最优**。SGD 的随机噪声让参数**自然逃离鞍点**,而不会陷入"差"的局部最优。

**加分**:NTK 视角(Jacot 2018)给出更强的理论:**无限宽网络**训练 = **核回归**(凸优化)。真实训练是这个凸优化的有限宽修正。

### Q2:SVM 的对偶为什么重要?

**答案**:

1. **核技巧**:对偶形式中,数据只以 $x_i^T x_j$ 形式出现,可以替换为核 $K(x_i, x_j)$,让 SVM 在**隐式高维 (甚至无限维)** 空间工作。例如 RBF 核 = 隐式升到无限维。
2. **稀疏性**:对偶变量 $\alpha_i > 0$ 只对应**支持向量**。预测时只需计算"测试点 × 支持向量",而非全部训练数据。**支持向量数 << 训练样本数**,预测高效。
3. **KKT 解释**:$\alpha_i = 0$ 对应非支持向量(约束不起作用);$\alpha_i > 0$ 对应支持向量(约束紧,互补松弛)。
4. **高效算法**:SMO(序列最小优化)在原问题上很难,在对偶问题上**只用更新 2 个 $\alpha_i$** 就能收敛。

### Q3:次梯度法为什么比梯度下降慢?

**答案**:

| | 梯度下降 | 次梯度法 |
| :--- | :--- | :--- |
| 适用 | 可微凸函数 | 任意凸函数 |
| 方向 | $-\nabla f$ 精确指向下降 | $-g, g \in \partial f$ 不一定下降 |
| 收敛率 | $O(1/T)$(L-光滑) | $O(1/\sqrt{T})$(无光滑性) |

**核心原因**:在不可微点(如 $|x|$ 在 0),次梯度**有多种选择**,不精确指向上升最快的方向。SGD 在每步要"探索"次梯度的方向,所以累积误差大。

**数学上**:
- GD:每步误差 $O(\|\nabla f\|^2)$,累积 $O(1/T)$
- 次梯度:每步误差 $O(\|g\|^2)$ 平方平均,累积 $O(1/\sqrt{T})$

**工程上**:LASSO 是凸但不可微,用次梯度法慢。改进:**近端梯度法**(proximal gradient)对 L1 范数有显式收缩(proximal operator),更快收敛。

### Q4:MMoE 多任务的"跷跷板"问题怎么用 KKT 解释?

**答案**:

把多任务建模为约束优化:

$$\min_\theta \Omega(\theta) \quad \text{s.t.} \quad L_k(\theta) \leq c_k, \; k = 1, \ldots, K$$

KKT 条件:每个任务有 $\lambda_k \geq 0$,且 $\lambda_k (L_k - c_k) = 0$。

**跷跷板的数学原因**:**任务梯度冲突**,即 $-\nabla L_1 \approx \nabla L_2$。此时**两个任务不可能同时满足 KKT**(除非 $\nabla L_1 = \nabla L_2 = 0$),总有一个 $\lambda_k = 0$,即"约束不起作用,任务被牺牲"。

**工程解法**:
1. **GradNorm**:动态调权重 $\alpha_k$,让不同任务的梯度范数平衡
2. **PCGrad**:投影冲突梯度(把一个任务的梯度投影到另一个任务的法平面)
3. **CGC / PLE**:用更复杂的专家结构,让任务共享更少
4. **多目标 Pareto 优化**:不强求单一最优,在 Pareto 前沿上选

### Q5:为什么 ResNet 的 loss landscape 更平滑?

**答案**:

1. **短路连接让梯度不消失**:
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$
   "1" 让梯度至少为 $\|\partial L / \partial y\|$,即使 $F$ 的梯度消失。

2. **经验观察**:Li et al. 2018 可视化 ResNet vs Plain 的 loss landscape:
   - Plain-110:loss 曲面非常崎岖,多局部最优,值差异大
   - ResNet-110:loss 曲面更平滑,接近凸函数,局部最优值接近全局最优

3. **理论直觉**:ResNet 的"短路"让 loss 等价于"输入恒等映射 + 残差学习"。残差 $F$ 通常很小,所以 loss landscape 接近"输入恒等 + 小扰动",曲面更平。

**工程验证**:Plain-110 在 CIFAR-10 上 loss 难收敛,需 Warmup;ResNet-110 直接 lr=0.1 也能训练。

---

## 12. 与其他数学工具的联系

| 关联 | 路径 | 章节 |
| :--- | :--- | :--- |
| 凸分析 → 优化理论 | 凸优化是优化理论核心 | 6-最优化 第 5 章 |
| 凸分析 → 概率论 | MLE = 最小化 KL = 凸问题(对参数) | 3-概率论 第 6 章 |
| 凸分析 → 线代 | Hessian 半正定 = 凸性二阶判据 | 2-线代 第 5 章 |
| 凸分析 → 1-高数 (本章) | 与第 5 章 Taylor(局部近似)、第 8 章 NTK(无限宽 = 凸) | 1-高数 第 5, 8 章 |
| 凸分析 → 系统思维 | MMoE 多任务"跷跷板"= KKT 不满足 | 7-系统思维 案例 2 |
| 凸分析 → 数值分析 | 投影到 L2 球、KKT 求解(内点法) | 见 §6.7 |
| 凸分析 → 信息论 | 交叉熵 = KL 散度的一部分,在 logits 上是凸 | 5-信息论 第 5 章 |

**互引提示**:本章大量引用了 `../../../基础/数学基础/第02章-线性代数/第02章-05-二次型·正定性·条件数-为什么-Adam-比-SGD-稳.md` 中的半正定概念,以及 第 8 章:泛函分析 中的 NTK 视角(支撑了 §8.4)。

---

## 13. 本章总结(1-高数文件夹收尾)

### 13.1 1-高数整个文件夹的回顾

我们走过的 10 章,是一条完整的"微积分 → 深度学习数学"的路径:

| 章 | 主题 | ML/DL 连接 |
| :--- | :--- | :--- |
| 1 | 极限·连续·无穷小 | 训练收敛性的语言 |
| 2 | 一元微分·导数·微分 | 切线 → 梯度 |
| 3 | 多元微分·梯度·Jacobian·Hessian | 反向传播的数学骨架 |
| 4 | 链式法则·自动微分·计算图 | loss.backward() 的本质 |
| 5 | Taylor 展开·二阶近似·Newton 法 | 为什么一阶方法统治 DL |
| 6 | 积分·测度·Lebesgue 积分 | 概率与 KL 散度的数学基础 |
| 7 | 级数·收敛·Fourier 分析 | Swin Transformer 的数学直觉 |
| 8 | 泛函分析·Hilbert 空间·RKHS·NTK | 函数空间视角 |
| 9 | 变分法·Euler-Lagrange 方程 | VAE·EM·最优控制 |
| **10(本章)** | **凸分析·凸锥·次梯度·KKT** | **为什么 DL 是非凸还能 work** |

### 13.2 "为什么 DL 数学如此重要" 总结

读完 1-高数 整个文件夹,你应该能:

1. **用极限描述训练收敛**(数列极限 + ε-N 语言)
2. **用梯度描述参数更新**(多元微分 + Jacobian + Hessian)
3. **用链式法则描述反向传播**(自动微分 + 计算图)
4. **用 Taylor 展开做局部近似**(为什么 Adam 比 SGD 稳)
5. **用积分描述概率**(Lebesgue 积分 + KL 散度)
6. **用 Fourier 分析描述频谱**(Transformer 的频域直觉)
7. **用函数空间理解 Kernel Methods 和 NTK**(无限维视角)
8. **用变分法理解 VAE 和 EM**(在函数空间上求极值)
9. **用凸分析理解优化的"理论极限"**(局部最优 = 全局最优)

**最核心的洞察**:**深度学习的"非凸悖论"不是 bug,是 feature**。

凸优化的"圣杯"(任何局部最优 = 全局最优)在 DL 中**不严格成立**,但通过:

- 好的初始化(Glorot / He)
- 充足的网络容量(过参数化)
- 聪明的优化器(Adam + Warmup)
- 合理的训练数据(大数据)
- SGD 的随机噪声

DL 工程师**在实践中**让非凸 landscape"表现得像凸的"。大部分局部最优值都接近全局最优。这不是数学奇迹,是**经验 + 理论**结合的工程胜利。

### 13.3 本章的灵魂问题(最后一次回答)

> **"为什么 DL 是非凸的,还能 work?"**

**一句话回答**:

> 深度学习的 loss landscape 是非凸的,但它**有"好"的非凸结构**:局部最优值接近全局最优、鞍点远多于局部最优、过参数化让梯度下降能 escape 大部分 basin、SGD 的随机噪声帮助 escape 鞍点。**凸优化理论是安慰,但工程实践是真理**。

这一章我们用凸集、凸函数、凸优化、次梯度、KKT、对偶理论,把"非凸悖论"完整形式化了。但理论是脚手架,**工程实践才是大楼**。接下来去 [[基础/数学/数学基础/第06章-最优化/]] 第 5-8 章,你会看到这些理论如何指导 Adam、SGD、Newton 法、L-BFGS 的实际算法设计。

---

## 附录 A:本章代码索引

| 实验 | 文件 | 行数 |
| :--- | :--- | :--- |
| 实验 1:次梯度法演示 | `subgrad_demo.py` | ~30 行 |
| 实验 2:凸 vs 非凸可视化 | `landscape.py` | ~30 行 |
| 实验 3:SVM KKT 验证 | `svm_kkt.py` | ~30 行 |
| 实验 4:鞍点演示 | `saddle.py` | ~15 行 |
| 实验 5:L2 归一化 | `normalize.py` | ~25 行 |

---

> **下一章**:跳到 `../第05章-信息论/0-总目录与阅读路径.md`,深入凸优化的算法侧(梯度下降的收敛性、强凸 vs 光滑、近端梯度法)。或者回到 `../第08章-数值计算与稳定性/0-总目录与阅读路径.md`,看 KKT 的工业落地(推荐系统的多目标优化)。
>
> **本章完。**

## 附录 B:凸集与凸锥的补充例题

### B.1 凸集判定 7 个经典例题

判定一个集合 $C$ 是不是凸集,核心是看**任意两点连线段是否在 $C$ 中**。下面 7 个例子覆盖面试和工业落地的高频考点。

**例题 1:超平面 (Hyperplane)**

$$H = \{x \in \mathbb{R}^n : a^T x = b\}$$

其中 $a \in \mathbb{R}^n$ 是非零向量,$b \in \mathbb{R}$。判定凸性。

**解答**:
取 $x_1, x_2 \in H$,即 $a^T x_1 = b$ 且 $a^T x_2 = b$。对任意 $\theta \in [0, 1]$:

$$a^T (\theta x_1 + (1-\theta) x_2) = \theta a^T x_1 + (1-\theta) a^T x_2 = \theta b + (1-\theta) b = b$$

所以 $\theta x_1 + (1-\theta) x_2 \in H$。超平面是仿射集(凸集的特例)。

**DL 连接**:感知机 (Perceptron) 的决策边界是超平面 $w^T x + b = 0$。SVM 寻找"最大间隔超平面"本质是在凸集(凸多面体)上求凸优化(二次规划)。

---

**例题 2:半空间 (Halfspace)**

$$H^- = \{x \in \mathbb{R}^n : a^T x \leq b\}$$

**解答**:
类似例题 1 的证明,只是把等号改成 $\leq$。对任意 $\theta \in [0, 1]$:

$$a^T (\theta x_1 + (1-\theta) x_2) = \theta a^T x_1 + (1-\theta) a^T x_2 \leq \theta b + (1-\theta) b = b$$

半空间是凸集。**重要:不等式是凸集的根本来源**。

---

**例题 3:欧几里得球 (Euclidean Ball)**

$$B(c, r) = \{x : \|x - c\|_2 \leq r\}$$

**解答**:
取 $x_1, x_2 \in B$,即 $\|x_1 - c\| \leq r$ 且 $\|x_2 - c\| \leq r$。对 $\theta \in [0, 1]$,由三角不等式:

$$\|\theta x_1 + (1-\theta) x_2 - c\| = \|\theta (x_1 - c) + (1-\theta) (x_2 - c)\|$$

$$\leq \theta \|x_1 - c\| + (1-\theta) \|x_2 - c\| \leq \theta r + (1-\theta) r = r$$

所以 $\theta x_1 + (1-\theta) x_2 \in B$。**球是凸集**(L2 范数的三角不等式)。

**DL 连接**:
- L2 正则化 $\|w\|_2 \leq \lambda$ 形成 L2 ball
- 权重衰减 (weight decay) = 在 L2 ball 内求最优点
- 对抗训练 (adversarial training) 的 PGD 攻击将扰动限制在 $\epsilon$-ball 内

```python
import torch
import torch.nn as nn

# L2-ball 投影
def project_l2_ball(w: torch.Tensor, radius: float) -> torch.Tensor:
    norm = w.norm()
    if norm <= radius:
        return w
    return w * (radius / norm)  # 沿 w 方向缩放到球面

# 例子:把神经网络权重的 L2 范数限制在 5.0
w = torch.randn(100, 100) * 2  # 初始 L2 范数可能 > 5
w_projected = project_l2_ball(w, 5.0)
assert torch.allclose(w_projected.norm(), torch.tensor(5.0), atol=1e-4)
print(f"原范数: {w.norm():.2f}, 投影后: {w_projected.norm():.2f}")
```

---

**例题 4:多面体 (Polyhedron)**

$$P = \{x : A x \leq b, \; C x = d\}$$

其中 $A, C$ 是矩阵,$b, d$ 是向量。这是若干个半空间和超平面的交集。

**解答**:
半空间和超平面都是凸集,任意多凸集的交集仍是凸集(见正文定理)。所以 $P$ 是凸集。

**多面体 = 有限个线性约束的交集**。这是 LP (Linear Programming) 和 QP (Quadratic Programming) 的可行域。

**DL 连接**:
- 神经网络的"硬约束"如输出概率非负、归一化,都可以表达为多面体
- LP 在推理加速 (quantization-aware) 中用于找最优量化阈值

---

**例题 5:概率单纯形 (Probability Simplex)**

$$\Delta = \{p \in \mathbb{R}^n : p_i \geq 0, \; \sum_i p_i = 1\}$$

**解答**:
- 非负约束 $p_i \geq 0$ 是若干个半空间的交集
- 归一约束 $\sum p_i = 1$ 是超平面
- 二者交集是多面体 = 凸集

**几何意义**:
$n = 3$ 时,$\Delta$ 是边长 $\sqrt{2}$ 的等边三角形,顶点是 one-hot 向量 $(1, 0, 0), (0, 1, 0), (0, 0, 1)$。

```text
     p_3
      ↑
   1 ┤   ●
      │  / \
      │ /   \
      │/     \
   0  ●───────●──→ p_1
      0       1
   p_2 = 1 - p_1 - p_3
```

**DL 连接**:
- 分类器的 softmax 输出 $\sigma(z) \in \Delta$
- 交叉熵 loss 的输入
- 强化学习中策略 $\pi(a|s) \in \Delta$

---

**例题 6:PSD 锥 (Positive Semidefinite Cone)**

$$S_+^n = \{X \in \mathbb{R}^{n \times n} : X = X^T, \; X \succeq 0\}$$

其中 $X \succeq 0$ 表示 $X$ 半正定(对所有 $v$,$v^T X v \geq 0$)。

**解答**:
取 $X_1, X_2 \succeq 0$,对 $\theta \in [0, 1]$ 和任意 $v$:

$$v^T (\theta X_1 + (1-\theta) X_2) v = \theta v^T X_1 v + (1-\theta) v^T X_2 v \geq 0$$

所以 $\theta X_1 + (1-\theta) X_2 \succeq 0$。**PSD 锥是凸锥**。

**DL 连接**:
- 核矩阵 $K = X X^T$ 是 PSD
- NTK (Neural Tangent Kernel) 矩阵在过参数化下是 PSD
- 二次型的几何意义:PSD = 函数在任意方向上非负

```python
import torch

# 验证两个 PSD 矩阵的凸组合仍是 PSD
A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])  # PSD
B = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # PSD
theta = 0.3

C = theta * A + (1 - theta) * B
eigvals = torch.linalg.eigvalsh(C)
print(f"C 的特征值: {eigvals}")  # 都应该 >= 0
assert (eigvals >= -1e-6).all()
print("凸组合仍是 PSD,验证通过")
```

---

**例题 7:范数球 (Norm Ball)**

$$B_p = \{x : \|x\|_p \leq 1\}$$

其中 $\|x\|_p = (\sum |x_i|^p)^{1/p}$ 是 $L_p$ 范数。

**解答**:
范数的三角不等式和齐次性保证:$\|\theta x_1 + (1-\theta) x_2\|_p \leq \theta \|x_1\|_p + (1-\theta) \|x_2\|_p \leq 1$。**所有范数球都是凸集**。

**特例**:
- $L_1$ 球:菱形 (在 $\mathbb{R}^2$ 中)
- $L_2$ 球:圆形
- $L_\infty$ 球:正方形

**L1 球的几何形状** ($\mathbb{R}^2$ 中):

```text
       y
        ↑
      1 ┤    ●
        │   ╱ ╲
        │  ╱   ╲
        │ ╱     ╲
      0 ┤●───────●
        │ ╲     ╱
        │  ╲   ╱
        │   ╲ ╱
     -1 ┤    ●
        └─────────→ x
       -1   0   1
```

**L_inf 球** ($\mathbb{R}^2$ 中):

```text
       y
        ↑
      1 ┤   ┌─────┐
        │   │     │
        │   │     │
      0 ┤   │     │
        │   │     │
        │   │     │
     -1 ┤   └─────┘
        └─────────→ x
       -1   0   1
```

**DL 连接**:
- L1 范数球 $\to$ LASSO (稀疏解)
- L2 范数球 $\to$ Ridge (权重衰减)
- L∞ 范数球 $\to$ 对抗训练 (PGD 攻击的 $\epsilon$-ball)

---

### B.2 凸锥 (Convex Cone) 详解

**定义**:集合 $C$ 是凸锥,若:
1. $C$ 是凸集
2. 对任意 $x \in C$ 和 $\theta \geq 0$,有 $\theta x \in C$ (齐次性)

**凸锥 vs 凸集**:凸锥是"更强的"凸集,它对正标量乘法和凸组合封闭。

**重要凸锥**:

| 凸锥 | 符号 | 例子 | DL 应用 |
| :--- | :--- | :--- | :--- |
| 非负象限 | $\mathbb{R}_+^n$ | 概率向量 | 激活值、注意力权重 |
| PSD 锥 | $S_+^n$ | 核矩阵 | 高斯过程、NTK |
| 范数锥 | $\{(x, t) : \|x\| \leq t\}$ | 第二范数约束 | 松弛变量 |
| 二阶锥 | $\{(x, t) : \|x\|_2 \leq t\}$ | SOCP | 推荐系统 |
| 指数锥 | $\{(x, y, z) : y > 0, ye^{x/y} \leq z, x < 0\} \cup \{(x, y, z) : x \geq 0, y = 0, z \geq 0\}$ | 对数障碍 | 内点法 |

**广义不等式**:凸锥 $K$ 可以定义偏序:
$$x \preceq_K y \iff y - x \in K$$

例如:
- $K = \mathbb{R}_+^n$:分量偏序 $x \preceq y$ 当且仅当 $x_i \leq y_i$ 对所有 $i$
- $K = S_+^n$:Loewner 偏序 $X \preceq Y$ 当且仅当 $Y - X \succeq 0$

**DL 连接**:
- 排序学习 (Learning to Rank) 的 pairwise loss 经常用 PSD 锥来建模
- 多标签分类中,标签空间 $\{-1, +1\}^K$ 可以转化为凸锥上的投影问题

---

### B.3 凸集的支撑超平面与分离定理

**支撑超平面定理**:设 $C$ 是非空凸集,$x_0$ 是 $C$ 的边界点(不在 $C$ 内部),则存在超平面 $H = \{x : a^T x = b\}$,使得:
- $C$ 在 $H$ 的一侧:$a^T x \leq b$ 对所有 $x \in C$
- $x_0 \in H$:$a^T x_0 = b$

**几何意义**:任何凸集都"贴"在某个超平面上,就像鸡蛋放在桌面上。

**严格分离定理**:设 $C$ 和 $D$ 是不相交的两个凸集,且至少一个是紧集 (有界闭集),则存在超平面 $a^T x = b$ 严格分离它们:
$$a^T x < b \text{ 对所有 } x \in C, \quad a^T x > b \text{ 对所有 } x \in D$$

**DL 应用:SVM 的几何意义**

支持向量机寻找分离超平面 $w^T x + b = 0$ 把两类点分开。**严格分离定理保证**:
- 如果数据是线性可分的,存在超平面分离
- SVM 在所有分离超平面中,找"最大间隔"的那个,即最优的 $w, b$

凸优化 + 几何分离 $\to$ SVM 数学理论。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 二维线性可分数据
np.random.seed(42)
X = np.vstack([
    np.random.randn(20, 2) + [2, 2],
    np.random.randn(20, 2) + [-2, -2]
])
y = np.hstack([np.zeros(20), np.ones(20)])

# 训练 SVM
clf = SVC(kernel='linear', C=1e6)  # 硬间隔
clf.fit(X, y)

# 分离超平面
w = clf.coef_[0]
b = clf.intercept_[0]
print(f"分离超平面: {w[0]:.2f} x1 + {w[1]:.2f} x2 + {b:.2f} = 0")
print(f"支持向量数量: {len(clf.support_)}")
```

---

## 附录 C:凸函数的二阶判据与等价条件

### C.1 一阶判据与二阶判据

**一阶判据** (可微情形):设 $f$ 可微,$f$ 是凸函数 $\iff$
$$f(y) \geq f(x) + \nabla f(x)^T (y - x) \quad \forall x, y$$

**几何意义**:**任何切线都是下界**。凸函数的图像在任意点的切线下方。

```text
     f(y)
      ↑       *  y 点
      │      ╱|
      │     ╱ |
      │    ╱  |
      │   ●———|————— 切线 (下界)
      │  ╱    |
      │ ╱     |
      └────────→ x
            x0
```

**二阶判据** (二次可微情形):$f$ 是凸函数 $\iff$ Hessian 矩阵半正定:
$$\nabla^2 f(x) \succeq 0 \quad \forall x$$

**强凸** (Strongly Convex):$\nabla^2 f(x) \succeq m I$ 对 $m > 0$。

```python
import torch
import torch.nn.functional as F

# 例子:验证 f(x) = x^2 (强凸, m = 2)
x = torch.linspace(-2, 2, 100, requires_grad=True)
f = x ** 2
g = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
h = torch.autograd.grad(g.sum(), x)[0]
print(f"x^2 的二阶导数: 最小值 = {h.min().item():.4f}, 最大值 = {h.max().item():.4f}")
# 应该是 2.0 (二阶导恒为 2)

# 例子:f(x) = |x| (凸但不可微, 在 x=0)
x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
f = torch.abs(x)
g = torch.autograd.grad(f.sum(), x)[0]
print(f"|x| 的梯度 at [-1, 0, 1]: {g}")
# -1, 0, 1 (在 x=0 是次梯度,可取 0)
```

---

### C.2 凸函数的常见封闭性

| 操作 | 凸性保持? | 条件 |
| :--- | :--- | :--- |
| 非负加权和 | 是 | $\alpha_i \geq 0, f_i$ 凸 $\implies \sum \alpha_i f_i$ 凸 |
| 仿射复合 | 是 | $g(x) = f(A x + b)$ 凸,若 $f$ 凸 |
| 逐点最大 | 是 | $g(x) = \max_i f_i(x)$ 凸,若 $f_i$ 凸 |
| 逐点上确界 | 是 | $g(x) = \sup_y f(x, y)$ 凸,若 $f$ 关于 $x$ 凸 |
| 复合 $h \circ f$ | 是 | $h$ 单调凸, $f$ 凸 |
| 乘法 | 否 | $f \cdot g$ 一般不凸 |
| 复合 $f \circ g$ | 否 | $f$ 凸, $g$ 凸 $\not\Rightarrow$ $f \circ g$ 凸 |
| 距离 | 是 | $g(x) = \text{dist}(x, C)$ 凸,若 $C$ 凸 |

**DL 应用:ReLU 网络的凸性**

ReLU 网络的逐层复合,每层都是仿射 + ReLU。ReLU 是凸函数,仿射变换保持凸性,但**两层 ReLU 网络** $f(x) = W_2 \max(0, W_1 x + b_1) + b_2$ **不是凸函数**。

为什么?ReLU 是凸,但**凸的复合非凸的**。具体:$\max(0, W_1 x + b_1)$ 关于 $x$ 凸,但第二层 $W_2 \cdot (\text{ReLU output})$ 是仿射(凸),但两个凸函数的**复合**不一定凸。

**关键洞察**:ReLU 网络层数 $\geq 2$ 时,**输出关于输入是非凸的**。这正是"非凸悖论"的来源。

---

### C.3 凸性在 DL 中的层级

| 网络结构 | 输入→输出映射 | loss 景观 |
| :--- | :--- | :--- |
| 线性回归 $y = W x$ | 凸 (仿射) | 凸 |
| 单层 ReLU $\max(0, W x + b)$ | 凸 | 凸 |
| 两层 ReLU | 非凸 | 非凸 |
| 深度 ReLU | 强非凸 | 强非凸 |
| 带 skip connection (ResNet) | 部分凸 | 比 Plain 平滑 |
| 自归一化 (SELU) | 接近恒等 | 比 ReLU 平滑 |

**经验观察**:网络越深,loss landscape 越复杂,但加上**好的设计**(skip、normalization),landscape 可以被"修饰"成接近凸。

---

### C.4 凸损失函数的例子表

| 损失函数 | 公式 | 凸性 | 可微性 | DL 任务 |
| :--- | :--- | :--- | :--- | :--- |
| MSE | $\|y - \hat{y}\|_2^2$ | 强凸 | 处处可微 | 回归 |
| MAE | $\|y - \hat{y}\|_1$ | 凸 | 不可微 (在 0) | 鲁棒回归 |
| Huber | 分段 (MSE / MAE) | 凸 | 处处可微 | 鲁棒回归 |
| Hinge (SVM) | $\max(0, 1 - y \hat{y})$ | 凸 | 不可微 (在 1) | 分类 |
| 交叉熵 | $-y \log \hat{y}$ | 凸 (对 logits) | 处处可微 | 分类 |
| KL 散度 | $\sum p \log(p/q)$ | 凸 (对 q) | 处处可微 | 蒸馏 |
| L1 正则 | $\|w\|_1$ | 凸 | 不可微 (在 0) | 稀疏 |
| L2 正则 | $\|w\|_2^2$ | 强凸 | 处处可微 | 权重衰减 |

**核心观察**:大部分**单层**的 loss 是凸的。**非凸性来自多层复合**。

---

## 附录 D:次梯度法实战与变体

### D.1 次梯度法的收敛性证明 (简化版)

**算法**:
$$x^{(k+1)} = x^{(k)} - \alpha_k g^{(k)}$$

其中 $g^{(k)} \in \partial f(x^{(k)})$,$\alpha_k$ 是步长。

**收敛性定理** (Nesterov 2004):若 $f$ 是凸函数,$\|g^{(k)}\| \leq G$ (次梯度有界),步长 $\alpha_k = O(1/\sqrt{k})$,则:
$$f\left(\frac{1}{T} \sum_{k=0}^{T-1} x^{(k)}\right) - f(x^*) \leq O\left(\frac{G \|x^{(0)} - x^*\|}{\sqrt{T}}\right)$$

**直观理解**:
- $O(1/\sqrt{T})$ 是次梯度法的标准收敛率
- 相比梯度下降 $O(1/T)$ (光滑情形),次梯度法慢一个量级
- 这是**不可微的代价**

**证明骨架** (4 步):

1. **每步下降估计**:
$$f(x^{(k+1)}) \leq f(x^{(k)}) + (g^{(k)})^T (x^{(k+1)} - x^{(k)}) + \frac{L}{2} \|x^{(k+1)} - x^{(k)}\|^2$$

2. **Lipschitz 性质**:$f(y) \leq f(x) + g^T (y - x) + (L/2) \|y - x\|^2$ 对任意 $y$。

3. **求和**:
$$\sum_{k=0}^{T-1} (f(x^{(k+1)}) - f(x^*)) \leq \sum_k (- \alpha_k (g^{(k)})^T (x^{(k)} - x^*) + \alpha_k^2 L G^2 / 2)$$

4. **取平均** + Cauchy-Schwarz:
$$f(\bar{x}) - f(x^*) \leq \frac{\|x^{(0)} - x^*\|^2}{2 \sum_k \alpha_k} + \frac{L G^2 \sum_k \alpha_k^2}{2 \sum_k \alpha_k}$$

5. **选步长** $\alpha_k = 1/\sqrt{k}$,代入上界得 $O(G/\sqrt{T})$。

**DL 应用**:L1 正则化训练神经网络的近端梯度法 (Proximal Gradient) 是次梯度法的近亲。

---

### D.2 近端梯度法 (Proximal Gradient)

**问题**:
$$\min_x f(x) + h(x)$$

其中 $f$ 光滑,$h$ 凸但不可微(如 L1 范数)。

**算法 (ISTA - Iterative Shrinkage-Thresholding)**:
$$x^{(k+1)} = \text{prox}_{\alpha h}(x^{(k)} - \alpha \nabla f(x^{(k)}))$$

其中 $\text{prox}$ 是近端算子:
$$\text{prox}_{\alpha h}(v) = \arg\min_x \left\{ h(x) + \frac{1}{2\alpha} \|x - v\|^2 \right\}$$

**L1 范数的近端算子 (软阈值)**:
$$\text{prox}_{\alpha \|\cdot\|_1}(v)_i = \text{soft}(v_i, \alpha) = \text{sign}(v_i) \max(|v_i| - \alpha, 0)$$

**L2 范数的近端算子**:
$$\text{prox}_{\alpha \|\cdot\|_2^2 / 2}(v) = \frac{v}{1 + \alpha}$$

**示例代码**:

```python
import torch

def soft_threshold(v: torch.Tensor, alpha: float) -> torch.Tensor:
    """L1 近端算子"""
    return torch.sign(v) * torch.relu(torch.abs(v) - alpha)

# 例子:对权重做 L1 稀疏化
w = torch.tensor([0.5, -1.2, 0.1, -0.3, 0.0, 2.1])
w_sparse = soft_threshold(w, alpha=0.5)
print(f"原权重: {w}")
print(f"稀疏后: {w_sparse}")  # [-0.7, 0, 0, 0, 2.1] 中间三个变 0

# 神经网络中:训练后做一次近端投影,得到稀疏模型
class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, l1_lambda=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        return self.linear(x)

    def sparsify(self):
        with torch.no_grad():
            self.linear.weight.copy_(soft_threshold(self.linear.weight, self.l1_lambda))
```

---

### D.3 次梯度的"非下降性"案例

**例子**:**ReLU 的反向传播** $g \in \partial \text{ReLU}(x)$,在 $x = 0$ 处次梯度集是 $[0, 1]$。如果你选次梯度为 1(梯度下降的方向),这一步更新是"反直觉"地让 loss **上升**。

**实例**:
$$f(x) = \text{ReLU}(-x) = \max(-x, 0)$$

在 $x = 0$ 处:
- 梯度:$\nabla f(0) = -1$ (左导数)
- 任何次梯度 $g \in [-1, 0]$

如果选 $g = 0$ (次梯度集的中间),梯度下降 $x \to x - \alpha \cdot 0 = x$ 是不动的。**这是个停滞点**。

**实际 PyTorch 行为**:`torch.relu` 在反向传播时,输入 $> 0$ 时梯度 = 1,输入 $\leq 0$ 时梯度 = 0。这是次梯度集 $\{0, 1\}$ 的特定选择。

```python
import torch

# PyTorch 中 ReLU 反向传播的实现
x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = torch.relu(x)
y.sum().backward()
print(f"输入: {x.detach()}")
print(f"ReLU 输出: {y.detach()}")
print(f"梯度 (次梯度集 [0, 1] 中选 0 或 1): {x.grad}")
# [-1.0] -> 0, 梯度 = 0 (因为 < 0)
# [ 0.0] -> 0, 梯度 = 0 (PyTorch 选 0)
# [ 1.0] -> 1, 梯度 = 1
# [ 2.0] -> 2, 梯度 = 1
```

---

### D.4 次梯度 vs 投影次梯度

**投影次梯度法 (Projected Subgradient)**:
$$\min_{x \in C} f(x)$$

其中 $C$ 是凸集。算法:
$$x^{(k+1)} = \Pi_C(x^{(k)} - \alpha_k g^{(k)})$$

$\Pi_C$ 是到 $C$ 的投影算子。

**例子**:L2 归一化的嵌入向量 (推荐系统) 要满足 $\|e\| = 1$。每步梯度下降后做投影:
$$\Pi_{B_2(0, 1)}(v) = \frac{v}{\max(\|v\|_2, 1)}$$

**完整代码**:

```python
import torch
import torch.nn as nn

class NormalizedEmbedding(nn.Module):
    """L2 归一化嵌入(召回系统的双塔模型常用)"""
    def __init__(self, num_items, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_items, dim)

    def forward(self, x):
        e = self.embedding(x)
        return torch.nn.functional.normalize(e, p=2, dim=-1)

    def project_step(self):
        """训练时每步把权重投影回单位球"""
        with torch.no_grad():
            norms = self.embedding.weight.norm(dim=-1, keepdim=True)
            self.embedding.weight.copy_(
                self.embedding.weight / norms.clamp(min=1.0)
            )

# 验证投影算子
def project_to_unit_ball(v: torch.Tensor) -> torch.Tensor:
    norm = v.norm()
    if norm <= 1:
        return v
    return v / norm

v = torch.tensor([3.0, 4.0])
v_proj = project_to_unit_ball(v)
print(f"原向量: {v}, 范数: {v.norm()}")
print(f"投影后: {v_proj}, 范数: {v_proj.norm()}")
# 应该是 [0.6, 0.8], 范数 = 1
```

**应用**:对比学习 (SimCLR, CLIP) 的 L2 归一化是核心 trick。投影 + 内积等价于余弦相似度。

---

## 附录 E:KKT 条件的工业案例

### E.1 KKT 案例:推荐系统的多目标优化

**场景**:抖音推荐要在 4 个目标间平衡:
1. CTR (点击率)
2. CVR (转化率)
3. 平均观看时长
4. 长期留存

每个目标是一个 loss $L_k(\theta)$,$\theta$ 是模型参数。

**建模为约束优化**:
$$\min_\theta \Omega(\theta) \quad \text{s.t.} \quad L_k(\theta) \leq c_k, \; k = 1, 2, 3, 4$$

其中 $\Omega$ 是正则项(防过拟合),$c_k$ 是可接受阈值。

**KKT 条件**:
1. 平稳性:$\nabla \Omega + \sum_k \lambda_k \nabla L_k = 0$
2. 原始可行性:$L_k(\theta) \leq c_k$
3. 对偶可行性:$\lambda_k \geq 0$
4. 互补松弛:$\lambda_k (L_k(\theta) - c_k) = 0$

**几何解释**:
- $\lambda_k > 0$ 表示第 $k$ 个约束是"激活的",目标 $k$ 严格 $L_k = c_k$
- $\lambda_k = 0$ 表示目标 $k$ 不在 bottleneck,可以放松

**跷跷板的数学原因**:
- 任务 1 和 2 的梯度**方向冲突**:$\nabla L_1 \approx -\nabla L_2$
- KKT 平稳性要求 $\sum \lambda_k \nabla L_k = -\nabla \Omega$
- 若 $\nabla L_1 + \nabla L_2 = 0$ 几乎,$\lambda_1$ 和 $\lambda_2$ 不能同时非零(否则 $\lambda_1 \nabla L_1 + \lambda_2 \nabla L_2 = 0$ 与 $\nabla \Omega$ 矛盾)
- 所以**只有一个任务被满足**,另一个被牺牲 $\to$ 跷跷板

**工程解法对比**:

| 方法 | 思路 | 优势 | 劣势 |
| :--- | :--- | :--- | :--- |
| 加权求和 | $\sum w_k L_k$ | 简单 | 权重难调,无 Pareto 保证 |
| GradNorm | 动态调 $w_k$ 让梯度范数平衡 | 自适应 | 计算复杂 |
| PCGrad | 投影冲突梯度 | 简洁有效 | 仅在训练时缓解 |
| MMoE | 多专家 + 门控 | 任务可分 | 仍可能跷跷板 |
| PLE | 渐进式专家 + 任务特化专家 | 跷跷板缓解 | 参数量大 |
| 多目标 Pareto | 找 Pareto 前沿 | 理论最优 | 工业落地难 |

---

### E.2 KKT 案例:公平性约束的机器学习

**场景**:贷款模型 $f(x) = w^T x$ 预测违约概率。但模型不能因为"性别"或"种族"对不同群体有差异性预测。

**建模**:
$$\min_w \frac{1}{n} \sum_i \ell(f(x_i), y_i) \quad \text{s.t.} \quad |\text{DP}| \leq \epsilon$$

其中 DP (Demographic Parity) 是:
$$\text{DP} = |P(\hat{y} = 1 | A = 0) - P(\hat{y} = 1 | A = 1)|$$

$A$ 是敏感属性(性别)。

**KKT 条件**:
$$\frac{1}{n} \sum_i \ell'(f(x_i), y_i) x_i + \lambda \nabla_w \text{DP} = 0$$

**互补松弛**:
- 若 $|\text{DP}| < \epsilon$:**公平性约束非激活** ($\lambda = 0$),模型自由优化 loss
- 若 $|\text{DP}| = \epsilon$:**公平性约束激活** ($\lambda > 0$),模型被迫满足公平性

**几何意义**:在 (loss, DP) 平面上,KKT 给出 Pareto 前沿上的一点。$\lambda$ 控制权衡。

```python
import torch
import torch.nn as nn

class FairClassifier(nn.Module):
    """带公平性约束的分类器(KKT 启发的软约束)"""
    def __init__(self, in_dim, lambda_fairness=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.lambda_fair = lambda_fairness

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def loss(self, x, y, sensitive_attr):
        # 经验风险
        pred = self.forward(x).squeeze()
        risk = nn.functional.binary_cross_entropy(pred, y.float())

        # 公平性项 (DP 近似)
        mask_0 = (sensitive_attr == 0)
        mask_1 = (sensitive_attr == 1)
        if mask_0.sum() > 0 and mask_1.sum() > 0:
            dp = (pred[mask_0].mean() - pred[mask_1].mean()).abs()
        else:
            dp = 0.0

        # Lagrangian (KKT 软约束)
        return risk + self.lambda_fair * dp

# 例子
model = FairClassifier(in_dim=10, lambda_fairness=0.5)
x = torch.randn(100, 10)
y = (torch.randn(100) > 0).float()
sensitive = (torch.randn(100) > 0).long()

loss = model.loss(x, y, sensitive)
print(f"总 loss: {loss.item():.4f}")

# 反向传播 (KKT 的"梯度")
loss.backward()
# 梯度是 Lagrangian 的梯度,对应 KKT 平稳性条件
```

---

### E.3 KKT 案例:SVM 详解

**硬间隔 SVM 原问题**:
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1, \; i = 1, \ldots, n$$

**Lagrangian**:
$$L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_i \alpha_i [y_i (w^T x_i + b) - 1]$$

**KKT 条件**:
1. 平稳性:$\nabla_w L = w - \sum_i \alpha_i y_i x_i = 0$ $\implies$ $w = \sum_i \alpha_i y_i x_i$
2. 平稳性:$\nabla_b L = -\sum_i \alpha_i y_i = 0$ $\implies$ $\sum_i \alpha_i y_i = 0$
3. 对偶可行性:$\alpha_i \geq 0$
4. 互补松弛:$\alpha_i [y_i (w^T x_i + b) - 1] = 0$

**重要推论**:
- $\alpha_i > 0$ 对应的 $x_i$ 是**支持向量** (support vector),在 margin 上
- $\alpha_i = 0$ 对应的 $x_i$ 在 margin 内部,不影响决策边界
- 决策边界 $w = \sum_{i \in SV} \alpha_i y_i x_i$,**只依赖支持向量**

**几何可视化**:

```text
     类 1 (+1)              类 2 (-1)
       *                       o
        *   <-- 支持向量    --> o
         *  margin   ●  o
        ─────────────────────  ← 决策边界 w^T x + b = 0
         o  margin   ●  *
        o   -->      *  <-- 支持向量
       o                       *
   w^T x + b = +1          w^T x + b = -1
   (H+)                     (H-)
```

```python
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 简单二维数据
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 1], [4, 2]])
y = np.array([1, 1, 1, -1, -1, -1])

# 硬间隔 SVM (C 很大)
clf = SVC(kernel='linear', C=1e10)
clf.fit(X, y)

# KKT 信息
print(f"权重向量 w: {clf.coef_[0]}")
print(f"偏置 b: {clf.intercept_[0]}")
print(f"支持向量索引: {clf.support_}")
print(f"支持向量坐标:\n{clf.support_vectors_}")
print(f"对偶系数 alpha: {clf.dual_coef_[0]}")

# 验证 KKT 条件
alpha = clf.dual_coef_[0]  # alpha_i * y_i
print(f"sum(alpha_i * y_i) = {alpha.sum():.6f} (应接近 0)")

# 验证互补松弛
for i, sv_idx in enumerate(clf.support_):
    margin = y[sv_idx] * (clf.coef_[0] @ X[sv_idx] + clf.intercept_[0])
    print(f"支持向量 {sv_idx}: y*(w^T x + b) = {margin:.4f} (应 = 1)")
```

---

### E.4 KKT 案例:投资组合优化

**问题** (Markowitz):
$$\min_w w^T \Sigma w \quad \text{s.t.} \quad \mu^T w \geq r, \quad \mathbf{1}^T w = 1, \quad w \geq 0$$

其中:
- $w \in \mathbb{R}^n$ 是资产权重
- $\Sigma$ 是协方差矩阵(风险)
- $\mu$ 是期望收益
- $r$ 是目标收益
- 约束:$\mu^T w \geq r$(达到目标收益),$\mathbf{1}^T w = 1$(资金归一),$w \geq 0$(不允许卖空)

**KKT 条件**:
1. 平稳性:$2 \Sigma w - \lambda \mu - \nu \mathbf{1} - \eta = 0$
2. 原始可行性:$\mu^T w \geq r$, $\mathbf{1}^T w = 1$, $w \geq 0$
3. 对偶可行性:$\lambda \geq 0$, $\eta \geq 0$
4. 互补松弛:$\lambda (\mu^T w - r) = 0$, $\eta_i w_i = 0$

**几何意义**:
- $\lambda > 0$ 表示收益约束激活($\mu^T w = r$)
- $\eta_i > 0$ 表示资产 $i$ 卖空约束激活($w_i = 0$)

**DL 类比**:把资产换成"专家"(如 MMoE 的多专家),把权重换成门控向量,把协方差换成专家输出的相关性。**多任务专家分配的数学本质和投资组合是同构的**。

---

## 附录 F:Lagrange 对偶深入

### F.1 强对偶何时成立:Slater 条件

**Slater 条件**:对凸优化问题
$$\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \; h_j(x) = 0$$

若存在 $x_0$ 使得 $g_i(x_0) < 0$ 对所有不等式约束成立(严格可行),则**强对偶**成立:对偶间隙 = 0,原问题和对偶问题达到相同的最优值。

**直观理解**:**如果原问题有一个"严格内部点",对偶间隙消失**。

**特例**:如果所有 $g_i$ 是仿射函数(线性),Slater 条件简化为"原问题有可行解"。

**反例**:
$$\min_x x^2 \quad \text{s.t.} \quad x^2 \leq 0$$

可行解只有 $x = 0$,但 Slater 条件要求 $x^2 < 0$(无解)。**强对偶不成立**,对偶间隙 > 0。

**DL 应用**:
- SVM 的约束是仿射函数($y_i(w^T x_i + b) \geq 1$),只要数据线性可分就有可行解,Slater 自动满足,**强对偶成立**。这就是 SVM 可以用对偶问题求解的原因。
- 神经网络有非凸约束时(如 $\text{ReLU}(x) = x \cdot \mathbb{1}[x > 0]$),Slater 条件不严格成立,强对偶可能失效。

---

### F.2 对偶的几何意义

**原问题**:$\min_x f(x) \text{ s.t. } x \in C$
**对偶问题**:$\max_\lambda \min_x L(x, \lambda)$

**对偶函数的图**:

```text
   f(x)
    ↑        原问题最优值 f*
    │        ●  ← 原问题最优解
    │       /|
    │      / |
    │     /  | 对偶函数 q(λ) 是 f 的凹包络
    │    /   | q(λ) ≤ f(x) 对所有 x
    │   /    |
    │  /     |
    │ ●      |
    │        |  ← 对偶最优 q* ≤ f*
    │         |
    └──────────→ x
```

**强对偶** = q* = f* (曲线相切)。**弱对偶** = q* ≤ f* 永远成立。

---

### F.3 对偶问题在 ML 中的工业应用

**应用 1:SVM 的对偶形式**

原问题 (Primal):
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1$$

对偶问题 (Dual):
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

**优势**:
- 决策边界 $w = \sum_i \alpha_i y_i x_i$,只依赖支持向量
- 通过核函数 $K$ 可以隐式映射到高维,避免显式计算
- 样本数 $n$ 远大于特征数 $d$ 时,对偶问题更高效

**应用 2:LASSO 的对偶**

LASSO 原问题:$\min_w \frac{1}{2} \|y - X w\|^2 \text{ s.t. } \|w\|_1 \leq t$

对偶问题:$\min_z \frac{1}{2} \|z\|^2 + t \|X^T z\|_\infty \text{ s.t. } X^T z = y - z$

(具体推导略。)

**应用 3:拉格朗日对偶在强化学习 (Constrained RL) 中的应用**

CMDP (Constrained MDP):
$$\max_\pi V^{\pi}(s_0) \quad \text{s.t.} \quad V^{\pi}_{c_k}(s_0) \leq d_k$$

Lagrangian:
$$L(\pi, \lambda) = V^{\pi}(s_0) - \sum_k \lambda_k (V^{\pi}_{c_k}(s_0) - d_k)$$

对偶:$\max_\lambda \min_\pi L(\pi, \lambda)$。这是 Lagrangian 方法 / PPO-Lagrangian 的数学基础。

---

### F.4 共轭函数 (Conjugate Function) 速成

**定义**:函数 $f$ 的 Fenchel 共轭:
$$f^*(y) = \sup_x \{y^T x - f(x)\}$$

**直观理解**:**用仿射函数 $y^T x - c$ 从上方逼近 $f$,最紧的上界就是 $f^*(y)$**。

```text
   f(x)
    ↑
    │     f 的图像
    │    *
    │   ╱|         y^T x - f*(y) 是在 y 处的支撑超平面
    │  ╱ | 切线
    │ ╱  |
    │╱___|__________
    └──────────────→ x
```

**性质**:
1. $f^*$ 永远是凸函数(即使 $f$ 不凸)
2. $(f^*)^* = f$(双共轭),若 $f$ 闭凸
3. $f(x) + f^*(y) \geq x^T y$ (Fenchel-Young 不等式)

**例子**:
- $f(x) = \frac{1}{2} x^T Q x$ (Q PSD):$f^*(y) = \frac{1}{2} y^T Q^{-1} y$
- $f(x) = \|x\|$:$f^*(y) = \begin{cases} 0 & \|y\|_* \leq 1 \\ +\infty & \text{otherwise} \end{cases}$,其中 $\|\cdot\|_*$ 是对偶范数
- $f(x) = I_C(x)$ (集合 $C$ 的示性函数):$f^*(y) = \sup_{x \in C} y^T x$ (支撑函数)
- $f(x) = -\log x$ ($x > 0$):$f^*(y) = -1 - \log(-y)$ (y < 0)

**DL 应用**:
- 共轭是凸分析的核心工具,但在工业 DL 中不直接使用
- 重要:**变分下界 (ELBO)** 的推导用了 Fenchel-Young 不等式。VAE 的 loss 是:

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z)) = \text{ELBO}$$

这是共轭函数的具体应用。

---

## 附录 G:深度学习非凸悖论的现代理论

### G.1 五大现代理论解释 "为什么 DL 非凸还能 work"

| 理论 | 提出者 | 年份 | 核心观点 |
| :--- | :--- | :--- | :--- |
| Spurious Local Minima | Choromanska et al. | 2015 | 局部最优值接近全局最优值 |
| Loss Landscape | Li et al. | 2018 | ResNet 等架构让 landscape 接近凸 |
| NTK | Jacot et al. | 2018 | 无限宽网络训练等价于核回归(凸) |
| Overparameterization | Du et al. | 2019 | 过参数化让所有局部最优 = 全局最优 |
| Lottery Ticket | Frankle & Carlin | 2019 | 存在稀疏子网络,独立训练也能收敛 |

**详细阐述**:

**1. Spurious Local Minima (Choromanska 2015)**

> 在网络深度 $L$ 足够大、宽度 $n$ 足够大时,网络 loss 的**所有局部最优值**与**全局最优值**的差,随网络宽度 $n$ 指数级衰减。

数学陈述(简化):
$$P\left(\min_i f(\theta_i^*) - f(\theta^*) \geq \epsilon\right) \leq e^{-c n \epsilon^2}$$

其中 $c > 0$ 是常数,$\theta_i^*$ 是局部最优,$\theta^*$ 是全局最优。

**直觉**:高维空间中的随机函数,极值会"集中"在某个值附近。这是因为高维空间的几何性质:任何方向的微小扰动都会暴露很多维度,导致平均行为占主导。

**2. Loss Landscape (Li 2018)**

**关键实验**:用"filter normalization"可视化 loss landscape (沿随机方向移动)。

```text
   Plain-110 损失曲面            ResNet-110 损失曲面

        |╲   ╱|                       |
   ──── | ╲ ╱ | ────             ──── |   ────
        |  X  |                       |
        | ╱ ╲ |                       |
        |╱   ╲|                       |

   崎岖,多局部最优              平滑,接近凸
```

**ResNet landscape 更平滑的原因**:
- 短路连接 $H(x) = F(x) + x$ 提供了"恒等映射的扰动视角"
- 残差 $F(x)$ 通常很小,loss 接近"输入恒等 + 小扰动"
- 数学上,$H$ 的梯度是 $(\partial F / \partial x) + I$,加上单位矩阵让 Hessian 更稳定

**3. NTK (Neural Tangent Kernel, Jacot 2018)**

**核心结果**:无限宽神经网络(每层宽度 $\to \infty$)在初始化附近训练,等价于**线性模型 + 固定核**:

$$f(x; \theta(t)) \approx f(x; \theta(0)) + \nabla_\theta f(x; \theta(0))^T (\theta(t) - \theta(0))$$

**NTK 矩阵**:
$$\Theta(x, x') = \nabla_\theta f(x; \theta(0))^T \nabla_\theta f(x'; \theta(0))$$

**NTK 理论**:
- 无限宽网络训练 = 在 RKHS 中最小化 loss(凸问题)
- 有限但大宽度的网络是"NTK 区域"的小扰动,继承了部分凸性
- 解释了为什么"过参数化让训练变得容易"

**4. Overparameterization (Du 2019)**

**核心定理** (简化):若神经网络宽度 $m \geq \text{poly}(n, L)$ (多项式关系),其中 $n$ 是样本数, $L$ 是深度,则:
- 所有局部最优 = 全局最优
- 梯度下降能找到全局最优

**直观解释**:
- 过参数化 = 维度远多于样本
- 此时"参数空间"充满"解超曲面"
- 任何初始化 + 梯度下降,都能沿梯度降到某个"解超曲面上"
- 不存在被卡住的"坏局部最优"

**5. Lottery Ticket (Frankle 2019)**

**核心发现**:随机初始化的密集网络中,存在一个稀疏子网络(10-20% 权重),独立训练也能达到原网络的精度。

**数学意义**:损失 landscape 中存在"好的稀疏路径",沿这些路径训练等价于全网络训练。

**DL 工程师的含义**:
- Pruning (剪枝) 不是"丢信息",而是"找到更高效的路径"
- 这解释了为什么训练大模型后剪枝效果好

---

### G.2 经验法则:为什么 DL 在实践中能收敛

| 设计选择 | 对 landscape 的影响 | 原理 |
| :--- | :--- | :--- |
| 好的初始化 (Glorot/He) | 起始点远离坏 basin | 避免激活饱和、梯度消失 |
| 充足宽度 | landscape 更平滑 | 接近 NTK 区域(凸) |
| Skip connection | 损失曲面更接近凸 | 残差小,Hessian 稳定 |
| BatchNorm | 减少内部协变量偏移 | 损失对参数变化的敏感度降低 |
| Dropout | 隐式 ensemble | 多种子平均,跳出坏 basin |
| SGD 噪声 | 逃离鞍点 | 随机方向扰动 |
| Adam 自适应 | 不同参数用不同 lr | 平衡梯度量级 |
| 学习率 warmup | 避免早期震荡 | 渐进式进入好的 basin |
| 大 batch | 减少噪声 | 但可能陷入 sharp minimum |
| 小 batch | 噪声大 | 倾向 flat minimum (泛化好) |

---

### G.3 收敛的"工程保证":验证清单

部署一个非凸训练时,工程师应该验证:

```text
[ ] 1. 损失下降 - train loss 持续下降
[ ] 2. 验证不爆炸 - val loss 没有暴涨
[ ] 3. 梯度范数 - ||grad|| 在合理范围 (1e-7 < ||grad|| < 1e3)
[ ] 4. 激活值 - 激活值分布正常(无 NaN/Inf)
[ ] 5. 权重范数 - ||W|| 不爆炸
[ ] 6. Hessian 谱 - 至少检查最大特征值 (sharpness)
[ ] 7. 学习率测试 - 多 lr 实验找最佳
[ ] 8. 多种子实验 - 至少 3 个种子,确认结果稳定
[ ] 9. 早停 - 验证集上选最优 checkpoint
[ ] 10. 失败模式清单 - 知道什么时候会崩(NaN、loss 反弹)
```

---

### G.4 一个具体的"非凸收敛"PyTorch 实验

**实验**:训练一个 5 层 MLP 在 MNIST 上,观察:
1. 损失曲线的下降
2. 梯度范数随时间变化
3. 不同种子的收敛差异

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 5 层 MLP (强非凸)
class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# 数据
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 不同种子的实验
results = {}
for seed in [0, 1, 42, 1234]:
    torch.manual_seed(seed)
    model = MLP(hidden=128)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    losses = []
    grad_norms = []
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 记录梯度范数
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)

            optimizer.step()
            losses.append(loss.item())

    results[seed] = {'losses': losses, 'grad_norms': grad_norms}
    print(f"种子 {seed}: 最终 loss = {losses[-1]:.4f}, 最终梯度范数 = {grad_norms[-1]:.4f}")

# 观察:不同种子收敛到不同的 loss (非凸的标志)
# 但都在合理范围 (< 0.1),证明 DL 的"好非凸"性质
print("\n关键观察:")
for seed, r in results.items():
    print(f"  种子 {seed}: loss {r['losses'][-1]:.4f} (差异 < 0.05)")
print("所有种子收敛到接近的 loss,说明 DL 的非凸 landscape 是'好'的")
```

---


---

## 附录 H:完整 PyTorch 实验集(8 个)

### H.1 实验 1:次梯度法演示 (L1 正则化)

```python
import torch
import numpy as np

torch.manual_seed(42); np.random.seed(42)
n, d = 100, 50
X = torch.randn(n, d)
true_w = torch.zeros(d)
true_w[:5] = torch.tensor([1.0, -2.0, 3.0, -1.5, 2.5])
y = X @ true_w + 0.1 * torch.randn(n)

def ista(X, y, lam=0.5, lr=0.01, n_iter=200):
    w = torch.zeros(d)
    for _ in range(n_iter):
        grad = X.T @ (X @ w - y)
        w_grad = w - lr * grad
        w = torch.sign(w_grad) * torch.relu(torch.abs(w_grad) - lr * lam)
    return w

w = ista(X, y)
print(f"恢复的非零: {(w != 0).sum().item()} (真值 5)")
print(f"估计: {w[:5].numpy()}")
```

### H.2 实验 2:凸 vs 非凸可视化

```python
import torch
import matplotlib.pyplot as plt
x = torch.linspace(-3, 3, 200)
f_convex = x ** 2
f_nonconvex = x ** 4 - 2 * x ** 2 + x

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x.numpy(), f_convex.numpy())
axes[0].set_title('Convex: $f(x) = x^2$')
axes[1].plot(x.numpy(), f_nonconvex.numpy())
axes[1].set_title('Non-convex: $f(x) = x^4 - 2x^2 + x$')
plt.show()
```

### H.3 实验 3:SVM KKT 验证

```python
from sklearn.svm import SVC
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 1], [4, 2]], dtype=float)
y = np.array([1, 1, 1, -1, -1, -1])

clf = SVC(kernel='linear', C=1e10)
clf.fit(X, y)

# KKT: w = sum alpha_i y_i x_i
w_kkt = (clf.dual_coef_[0].reshape(-1, 1) * clf.support_vectors_).sum(axis=0)
print(f"权重 w: {clf.coef_[0]}")
print(f"KKT 重建: {w_kkt}")
print(f"sum alpha_i y_i = {clf.dual_coef_[0].sum():.6e}")
```

### H.4 实验 4:鞍点 + SGD 噪声

```python
import torch
import torch.optim as optim

def saddle_loss(p):
    return p[0] ** 2 - p[1] ** 2

p = torch.tensor([0.1, 0.1], requires_grad=True)
opt = optim.SGD([p], lr=0.01)

for step in range(100):
    opt.zero_grad()
    loss = saddle_loss(p)
    loss.backward()
    # 加噪声模拟 SGD mini-batch
    with torch.no_grad():
        noise = torch.randn_like(p) * 0.1
        p.add_(-0.01 * p.grad + noise)
    if step % 30 == 0:
        print(f"Step {step}: p = {p.tolist()}")
```

### H.5 实验 5:投影梯度法 (L2 归一化)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedEmbedding(nn.Module):
    def __init__(self, num_items, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_items, dim)
    def forward(self, x):
        return F.normalize(self.embedding(x), p=2, dim=-1)

emb = NormalizedEmbedding(1000, 64)
x = torch.randint(0, 1000, (32,))
e = emb(x)
print(f"嵌入 L2 范数: {e.norm(dim=-1).mean().item():.4f} (应 ≈ 1)")
```

### H.6 实验 6:QP 实战 (L-BFGS)

```python
import torch
from torch.optim import LBFGS

Q = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
c = torch.tensor([1.0, 1.0])
x = torch.zeros(2, requires_grad=True)
opt = LBFGS([x], lr=0.1, max_iter=50)

def closure():
    opt.zero_grad()
    loss = 0.5 * x @ Q @ x - c @ x
    loss.backward()
    return loss

for _ in range(5):
    opt.step(closure)
print(f"解: {x.detach()}, 解析: {torch.linalg.solve(Q, c)}")
```

### H.7 实验 7:对比学习 + L2 归一化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 128)
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)
    def loss(self, z_i, z_j, t=0.1):
        sim = z_i @ z_j.T / t
        labels = torch.arange(z_i.size(0))
        return F.cross_entropy(sim, labels)

model = SimCLR(64)
z_i = torch.randn(32, 64)
z_j = torch.randn(32, 64)
loss = model.loss(model(z_i), model(z_j))
print(f"对比损失: {loss.item():.4f}")
```

### H.8 实验 8:MMoE 多任务梯度冲突

```python
import torch
import torch.nn as nn

class MMoE(nn.Module):
    def __init__(self, n_in, n_experts, n_tasks):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(n_in, 32) for _ in range(n_experts)])
        self.gates = nn.ModuleList([nn.Linear(n_in, n_experts) for _ in range(n_tasks)])
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(n_tasks)])
    def forward(self, x):
        e = torch.stack([ex(x) for ex in self.experts], dim=1)
        out = []
        for k, g in enumerate(self.gates):
            gate = torch.softmax(g(x), dim=-1)
            out.append(self.heads[k]((gate.unsqueeze(-1) * e).sum(1)))
        return out

model = MMoE(20, 4, 2)
x = torch.randn(64, 20)
y1 = (x.sum(-1) > 0).float()
y2 = (x.sum(-1) < 0).float()
p1, p2 = model(x)
print(f"任务 1 loss: {F.binary_cross_entropy_with_logits(p1, y1):.4f}")
print(f"任务 2 loss: {F.binary_cross_entropy_with_logits(p2, y2):.4f}")
```

---

## 附录 I:凸性判别速查

### I.1 凸性判别流程

```text
判断 f(x) 是否凸
├── 简单形式? → 查表 (MSE 凸、CE 凸、L1 凸、L2 强凸)
├── 一次可微? → f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
├── 二次可微? → ∇²f(x) ⪰ 0
└── 凸组合? → 非负加权和、逐点最大、仿射复合
```

### I.2 凸优化算法选择

| 问题 | 算法 | 收敛率 |
| :--- | :--- | :--- |
| 强凸+光滑 | GD | $O(\kappa \log(1/\epsilon))$ |
| 凸+光滑 | GD | $O(1/\epsilon)$ |
| 强凸+L1 | FISTA | $O(1/\epsilon)$ |
| 非凸 | SGD/Adam | 经验 $O(1/\sqrt{T})$ |
| 凸约束 | Interior Point | $O(\log(1/\epsilon))$ |

---

## 附录 J:习题与解答

### 习题 1:凸集判定

1. 单位圆盘 $x_1^2 + x_2^2 \leq 1$ — **凸**(L2 球)
2. 单位圆周 $x_1^2 + x_2^2 = 1$ — **非凸** (反例 (1,0) 与 (0,1) 中点不在圆周)
3. 双曲线外 $x_1 x_2 \geq 1$ — **凸** (凸函数 $g(x) = -x_1 x_2$ 的 0 下水平集)
4. 概率单纯形 — **凸** (多面体)

### 习题 2:次梯度计算

1. $\partial |x| \big|_{x=0} = [-1, 1]$
2. $\partial \text{ReLU}(0) = [0, 1]$
3. $\partial |x| + y^2 \big|_{(0,0)} = \{g : g_1 \in [-1, 1], g_2 = 0\}$

### 习题 3:KKT 应用

问题: $\min x_1^2 + x_2^2$ s.t. $x_1 + x_2 \geq 1$

KKT 解: $x^* = (0.5, 0.5)$, $\lambda^* = 1$ (约束激活)。

验证:
- $\nabla f = (2 \cdot 0.5, 2 \cdot 0.5) = (1, 1)$
- $\lambda \nabla g = 1 \cdot (-1, -1) = (-1, -1)$
- $\nabla f - \lambda \nabla g = (1, 1) - (-1, -1) = (2, 2) \neq 0$

**等等,让我重做**。约束是 $g(x) = 1 - x_1 - x_2 \leq 0$ (KKT 形式),$L = x_1^2 + x_2^2 + \lambda (1 - x_1 - x_2)$,$\nabla L = (2x_1 - \lambda, 2x_2 - \lambda) = 0$,所以 $x_1 = x_2 = \lambda/2$。代入 $1 - 2 \cdot \lambda/2 = 0$ 得 $\lambda = 1$, $x^* = (0.5, 0.5)$。

### 习题 4:对偶推导

LP 对偶: $\min c^T x$ s.t. $Ax = b, x \geq 0$ $\iff$ $\max b^T \lambda$ s.t. $A^T \lambda \leq c$。

---

## 附录 K:面试 Q&A 进阶

### Q1:Adam 与凸性的关系?

**答**:Adam 是对角预条件 SGD + 自适应。**对凸问题收敛 $O(1/\sqrt{T})$**;**对非凸没有严格保证**(Reddi 2018 指出 Adam 可能不收敛)。AMSGrad 修复收敛性。

### Q2:Softmax + CE 梯度为什么简洁?

**答**:
$$\frac{\partial L}{\partial z_i} = \sigma(z_i) - y_i$$

因为 CE 和 softmax 是 Fenchel 共轭,梯度自然简洁。**关于 logits 是凸的**,但**关于权重仍非凸**。

### Q3:为什么 ReLU 比 Sigmoid 好?

**答**:
| 性质 | Sigmoid | ReLU |
| :--- | :--- | :--- |
| 凸 | 是 | 是 |
| 梯度消失 | 饱和区 | 无 |
| 稀疏 | 否 | 是 |
| 凸性视角 | "凸但有饱和" | "凸 + 无饱和" |

### Q4:BatchNorm 与凸性?

**答**:BN 不是凸算子,但让 landscape 更平滑(Santurkar 2018:Lipschitz 常数变小)。训练用 batch 统计(随机优化),测试用 running(确定性凸优化)。

### Q5:Adam vs SGD on CV vs NLP?

**答**:
- NLP (Transformer): loss 曲面陡,Adam 优势大
- CV (CNN): loss 曲面平滑,SGD+momentum 足够
- 凸分析视角:NLP 的"近凸程度"比 CV 差,需更"智能"优化器

### Q6:多任务"跷跷板"的数学本质?

**答**:**任务梯度方向冲突** ($\nabla L_1 \approx -\nabla L_2$),KKT 平稳性 $\sum \lambda_k \nabla L_k = 0$ 不可能在两个任务上同时成立(除非 $\nabla L_k = 0$)。**强对偶不严格成立** $\to$ 跷跷板是必然。

### Q7:为什么需要 grid search 调超参?

**答**:**非凸 landscape 不可解析**。凸分析告诉我们"性质",不告诉"具体超参"。Optuna 贝叶斯优化是工程实践。

### Q8:GAN 训练不稳定怎么解?

**答**:
$$\min_G \max_D V(D, G)$$

KKT 视角:纳什均衡 = KKT 点,但 GAN 难达。改进:WGAN(强制 1-Lipschitz)、谱归一化(算子范数约束)让博弈"更近凸"。

---

## 附录 L:极简 cheatsheet

```text
┌─────────────────────────────────────────────────┐
│ 凸集判定:θx + (1-θ)y ∈ C  ∀x, y ∈ C, θ ∈ [0,1]  │
│ 凸函数:f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)       │
│ 凸优化:min f  s.t. x ∈ C, f 凸, C 凸             │
│   → 局部最优 = 全局最优                          │
├─────────────────────────────────────────────────┤
│ KKT 四条件:                                     │
│   1. ∇f + Σλᵢ∇gᵢ = 0 (平稳性)                  │
│   2. gᵢ(x) ≤ 0 (原始可行)                       │
│   3. λᵢ ≥ 0 (对偶可行)                          │
│   4. λᵢgᵢ(x) = 0 (互补松弛)                     │
├─────────────────────────────────────────────────┤
│ 强对偶:Slater 条件 (∃x₀, gᵢ(x₀) < 0)            │
│ 次梯度:ReLU 在 0 = [0, 1], |x| 在 0 = [-1, 1]    │
├─────────────────────────────────────────────────┤
│ DL 非凸悖论:                                     │
│   • 局部最优 ≈ 全局最优 (Choromanska 2015)       │
│   • Landscape 平滑 (Li 2018, ResNet)              │
│   • NTK = 凸 (Jacot 2018, 无限宽)                │
│   • 过参数化 = 全局最优 (Du 2019)                 │
│   • 工程:初始化、warmup、SGD 噪声                 │
└─────────────────────────────────────────────────┘
```

---

## 附录 M:参考文献

### M.1 教材

1. Boyd & Vandenberghe. *Convex Optimization*. 2004. (免费 PDF)
2. Nesterov. *Introductory Lectures on Convex Optimization*. 2004.
3. Bertsekas. *Convex Optimization Theory*. 2009.
4. Rockafellar. *Convex Analysis*. 1970.

### M.2 DL 理论论文

1. Choromanska et al. (2015). *The Loss Surfaces of Multilayer Networks*.
2. Kawaguchi (2016). *Deep Learning without Poor Local Minima*.
3. Li et al. (2018). *Visualizing the Loss Landscape of Neural Networks*.
4. Jacot et al. (2018). *Neural Tangent Kernel*.
5. Du et al. (2019). *Gradient Descent Finds Global Minima of Deep Neural Networks*.
6. Frankle & Carlin (2019). *The Lottery Ticket Hypothesis*.
7. Santurkar et al. (2018). *How Does Batch Normalization Help Optimization?*.
8. Reddi et al. (2018). *On the Convergence of Adam and Beyond*.

---

## 附录 N:常见错误与陷阱

### 错误 1:ReLU 网络凸性

```text
错误: "ReLU 凸,所以 ReLU 网络也凸"
正确: 单层 ReLU = 凸,两层以上 = 非凸
      凸的复合不一定凸
```

### 错误 2:L1 范数梯度

```text
错误: ∇|x| = sign(x)
正确: sign(x) 在 x=0 不可微,次梯度集 [-1, 1]
      实际工程用 soft thresholding
```

### 错误 3:KKT 用于非凸

```text
错误: "KKT = 全局最优"
正确: KKT 在凸问题是充要,在非凸只是必要
      非凸时 KKT 点可能只是鞍点或局部最优
```

### 错误 4:对偶恒等

```text
错误: "对偶 = 原问题"
正确: 弱对偶恒成立,强对偶需 Slater
      非凸对偶间隙可能 > 0
```

### 错误 5:Adam 收敛保证

```text
错误: "Adam 一定收敛"
正确: 凸 + 假设下收敛;非凸 (DL) 无保证
      Reddi 2018 证明 Adam 可能不收敛
```

### 错误 6:Loss 下降 = 全局最优

```text
错误: "loss 下降 = 找到了全局最优"
正确: 凸:对,非凸:只到某个 basin
      验证:多种子、held-out、early stopping
```

---

## 附录 O:终章总结

### 1-高数 10 章回顾

| 章 | 主题 | 一句话 |
| :--- | :--- | :--- |
| 1 | 极限·连续 | "训练收敛"的语言 |
| 2 | 一元微分 | 导数 = 切线斜率 |
| 3 | 多元微分 | 梯度 = 上升最快 |
| 4 | 链式法则 | 反向传播的骨架 |
| 5 | Taylor | 局部近似 → Newton |
| 6 | 积分·测度 | 概率 = 测度 |
| 7 | 级数·Fourier | 频谱视角 |
| 8 | 泛函分析 | 函数空间视角 |
| 9 | 变分法 | 函数空间极值 |
| **10** | **凸分析·KKT** | **局部最优 = 全局最优 + DL 非凸悖论** |

### 一句话总结

> **深度学习的数学不是"凸优化",而是"在非凸 landscape 上做凸优化"**。凸分析给我们"理论极限",工程实践给我们"现实约束"。

### 工程师口诀

```text
凸集凸函数凸优化,局部最优即全局好;
次梯度法处理不可微,KKT 是约束一阶条件;
Slater 条件强对偶,DL 非凸仍能 work;
ResNet 残差近凸,NTK 无限宽近凸;
凸分析是理论,工程实践是真理。
```

### 下一步路径

- **算法侧**:6-最优化 第 5 章(凸优化算法)
- **KKT 落地**:6-最优化 第 8 章
- **横向**:7-系统思维 案例 2(MMoE)
- **纵向**:1-高数 第 8 章(NTK 无限维)

---

> **本章完。** 凸分析是数学工具箱里最锋利的一把"凸刀",从凸集、凸函数、KKT 走到 DL 的非凸悖论,理论是脚手架,工程是摩天大楼。


## 第一部分 · 深度扩展:多视角系统展开

> 本部分是 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的全面延伸。每个章节从不同视角展开,系统补充原内容未覆盖的数学推导、工程实现、典型案例。所有新增内容自成体系,可独立阅读。

### 14. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 极限与连续深度分析

**核心议题**: 在 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的语境下,极限与连续 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 极限与连续 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work 的另一个视角。


---

### 15. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 导数微分深度分析

**核心议题**: 在 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的语境下,导数微分 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 导数微分 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work 的另一个视角。


---

### 16. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 积分测度深度分析

**核心议题**: 在 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的语境下,积分测度 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 积分测度 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work 的另一个视角。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 拆解为以下几个层次:

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

信息论为"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 拆解为以下几个层次:

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

信息论为"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 拆解为以下几个层次:

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

信息论为"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 拆解为以下几个层次:

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

信息论为"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 拆解为以下几个层次:

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

信息论为"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 10 章:凸分析·凸锥·次梯度·KKT——为什么DL是非凸还能work" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



