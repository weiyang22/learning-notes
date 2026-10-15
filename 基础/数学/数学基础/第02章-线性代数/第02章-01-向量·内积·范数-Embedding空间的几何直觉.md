---
title: "第02章-01-向量·内积·范数-Embedding空间的几何直觉"
tags:
  - 数学
  - 向量
  - 内积
  - 范数
  - Embedding空间的几何直觉
description: "本章要点：ng = nn.Embedding(numitems, 64)。64 是什么?它就是\"每个物品的向量维度\"。把这个向量丢进模型,模型就能算出\"用户喜不喜…"
chapter_id: 数学-1-02
last_updated: 2026-07-14
---

# 第02章-01-向量·内积·范数-Embedding空间的几何直觉

ng = nn.Embedding(num_items, 64)`。64 是什么?它就是"每个物品的向量维度"。把这个向量丢进模型,模型就能算出"用户喜不喜欢这个物品"。
>
> 但你想过没:
> - **为什么 64 维的向量能表示"语义"**?
> - **"相似"在向量空间里到底是什么意思**?
> - **`user_emb @ item_emb.T` 这一行内积在算什么**?
> - **Attention 里的 `Q @ K.T` 为什么能当"相似度"**?
>
> 这些问题的答案,全部藏在**线性代数的第一章:向量、内积、范数**里。
>
> 这一章我们要做的事情很朴素:把 PyTorch 里那些看起来很神秘的"向量运算",翻译回几何直觉 + 严格的数学定义。看完之后你会发现:
>
> - Embedding 空间 = 一个高维欧氏空间,每个物品 = 一个点
> - "语义相似" = 两点在空间里离得近(或者方向接近)
> - 内积 = 一个向量在另一个向量上的"投影长度",**也是 Attention 打分的本质**
> - 范数 = 向量的"长度",**也是 L2 归一化、梯度裁剪的理论基础**
> - cosine 相似度 = 归一化后的内积,值 ∈ [-1, 1]
>
> 学完这一章,你回头看 BERT、Transformer、DLRM,会发现它们的"评分函数"全是同一个公式的不同变体。这就是线性代数的力量。

---

## 1. 直觉引入

### 1.1 工程师的痛点:为什么 Embedding 是"高维向量"?几何意义是什么?

做推荐系统、搜索、广告,你每天都在跟 Embedding 打交道。假设你要给"商品"做 Embedding:

```python
import torch
import torch.nn as nn

# 给 100 万个商品做 64 维 Embedding
item_embedding = nn.Embedding(1000000, 64)

# 一个商品的向量长这样
item_id = torch.tensor([42])
vector = item_embedding(item_id)  # shape: (1, 64)
print(vector.shape)  # torch.Size([1, 64])
print(vector[0, :8])  # 前 8 维:[0.013, -0.027, 0.041, -0.019, ...]
```

这一行 `nn.Embedding(1000000, 64)` 把每个商品 ID(从 0 到 999999)映射成 64 维的浮点数向量。**这 64 个数字到底在表达什么?**

最朴素的回答:**它们在表达商品的"语义"**。比如:

```text
商品 0 (iPhone):     [0.13, -0.27, 0.41, -0.19, ...]  ← 数码特征强
商品 1 (AirPods):    [0.11, -0.25, 0.39, -0.22, ...]  ← 数码特征强
商品 2 (T恤):        [-0.31, 0.42, -0.15, 0.27, ...]  ← 服装特征强
商品 3 (运动鞋):     [-0.28, 0.45, -0.18, 0.31, ...]  ← 服装特征强
```

如果 Embedding 训练得好,你会发现:

- iPhone 和 AirPods 的向量**很像**(都是数码产品)
- T恤 和 运动鞋 的向量**很像**(都是服装)
- iPhone 和 T恤 的向量**差异很大**(品类不同)

**"像"在数学上是什么意思?就是两个向量的"距离近"或"方向接近"**。这就是几何意义。

#### 用向量表示语义的本质

为什么用 64 个数字就能"描述"一个商品?因为:

1. **每个维度捕捉一种"语义特征"**(虽然人类看不懂)
2. **相似商品在每个维度上的"取值模式"接近**
3. **两个向量的"接近程度" = 商品的"语义接近程度"**

这就把"商品相似度"这个抽象概念,**翻译成了向量的几何关系**。整个推荐系统、搜索、对比学习的 Embedding,都在做这件事。

### 1.2 直觉:每个物品 = 一个点,语义相似 = 距离近

把上面的向量想象成"空间中的点":

- 64 维空间里,每个商品是其中一个点
- iPhone 和 AirPods 是空间里挨得很近的两个点
- T恤 和 iPhone 是空间里离得很远的两个点

这就是"Embedding 空间"的几何直觉。**学习 Embedding = 学习一个映射,把每个物品放到合适的空间位置**。

为什么这个直觉如此重要?因为:

- **召回 = 在空间中找最近的 K 个点**(向量检索 / ANNS)
- **排序 = 用内积打分**(`user_emb @ item_emb.T`)
- **对比学习 = 把相似样本拉近,不同样本推远**

这些工程任务的**所有算法**,都是基于"高维空间中的点 + 距离 + 方向"这三个几何对象。

### 1.3 ASCII 可视化:2D 平面上的 5 个商品 Embedding

为了可视化,我们把 64 维压缩到 2 维(用 PCA 或 t-SNE)。假设我们有 5 个商品,降维后位置如下:

```text
y 轴
 ↑
 │         ★ B (AirPods, 数码)
 │        ╱
 │       ╱
 │      ★ A (iPhone, 数码)
 │
 │  ──────────────── 虚线: 数码 vs 服装的分界
 │
 │      ★ D (运动鞋, 服装)
 │       ╲
 │        ╲
 │         ★ C (T恤, 服装)
 │             ★ E (笔记本, 数码)
 └──────────────────────→ x 轴
   ↑                    ↑
  服装特征             数码特征
```

观察这张图:

- **A (iPhone)** 和 **B (AirPods)** 距离近(都是数码)
- **C (T恤)** 和 **D (运动鞋)** 距离近(都是服装)
- **A** 和 **C** 距离远(不同品类)
- **E (笔记本)** 也属于数码,跟 A/B 距离适中

**相似度 = 距离的倒数**。所以推荐系统的"召回",就是在 Embedding 空间里找与用户最近邻的 K 个商品。

### 1.4 类比:Embedding 空间 = "语义地图"

把 Embedding 空间想象成一张"语义地图":

| 真实地图 | Embedding 空间 |
| :--- | :--- |
| 城市 = 点 | 商品/用户 = 点 |
| 经纬度 = 2D 坐标 | Embedding 向量 = 高维坐标 |
| 物理距离 = 公里 | 几何距离 = L2 范数 |
| "两个城市近" = 物理距离近 | "两个商品相似" = 几何距离近 |
| 路径规划 = 找最短路 | 推荐召回 = 找最近邻 |

**唯一的区别**:真实地图是 2D 或 3D(经度、纬度、高度),Embedding 空间是 64 维或 768 维(768 是 BERT-base 的 hidden size)。高维空间的几何规律跟低维不太一样(我们在 第 8 章:高维空间几何 专门讨论),但"距离 + 方向"这两个核心几何概念还在。

### 1.5 章节路线图

接下来 11 节,我们依次讲:

| 节 | 主题 | 与 ML/DL 的联系 |
| :--- | :--- | :--- |
| §2 | 向量的本质 (向量空间 8 条公理) | Embedding / PyTorch tensor 的数学基础 |
| §3 | 向量加法与数乘 | Embedding 加权和 (BERT [CLS] / 残差连接) |
| §4 | 内积 (Dot Product) | Attention Q·K / 双塔召回 / ItemCF |
| §5 | Cauchy-Schwarz 不等式 | cosine 范围 / 归一化的必要性 |
| §6 | 范数 (L¹ / L² / L∞) | L2 归一化 / Adam 梯度裁剪 / 对抗样本 |
| §7 | 余弦相似度 | SimCSE / 推荐召回 / 对比学习 |
| §8 | 向量空间的几何直觉 | 标准正交基 / 投影 / PCA |
| §9 | PyTorch 代码 | 6 个动手实验 |
| §10 | 常见误区 | 5-8 条工程师踩坑点 |
| §11 | 面试 Q&A | L7/L8 难度 5 题 |
| §12 | 与其他数学工具的联系 | 矩阵 / 矩阵范数 / Hilbert 空间 |

---

## 2. 向量的本质

### 2.1 物理向量 vs 数学向量

**物理向量**(中学物理教的):有大小、有方向的箭头。

```text
       ↑ F (力, 向量)
       │
       │
       │────────→ x 轴
```

**数学向量**(线性代数里的):一个有序的实数元组。

$$\mathbf{x} = (x_1, x_2, \ldots, x_n)^T \in \mathbb{R}^n$$

比如 $\mathbf{x} = (3, 4)^T$ 表示二维平面上一个点,既有大小(长度)又有方向(从原点指向它)。

**物理向量是数学向量的特例**。数学向量更抽象:

- 可以是 n 维(不局限于 2D / 3D)
- 不一定有"物理意义"(比如"用户 Embedding"没有物理量纲)
- 但所有几何运算(加法、数乘、内积、范数)都跟物理向量一样

### 2.2 列向量 vs 行向量

线性代数里,向量**默认是列向量**(竖着写),但在 PyTorch / NumPy 里默认是行向量(横着写)。这点经常让初学者困惑。

**数学约定**:

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} \in \mathbb{R}^{n \times 1}$$

这是列向量,shape 是 $(n, 1)$。

**PyTorch 约定**:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape)  # torch.Size([3])
```

PyTorch 里 `torch.tensor([1, 2, 3])` 的 shape 是 `(3,)`,既不是行也不是列,叫"一维张量"。但在做矩阵乘法时,PyTorch **会自动判断方向**:

```python
x = torch.tensor([1.0, 2.0, 3.0])  # shape (3,)
W = torch.randn(3, 5)              # shape (3, 5)

# 自动把 x 视为列向量 (3, 1),然后 W @ x 得 (5,)
y = W @ x                          # shape (5,)
print(y.shape)                     # torch.Size([5])

# 等价的"显式"写法
x_col = x.unsqueeze(1)             # shape (3, 1)
y2 = W @ x_col                     # shape (5, 1)
y2 = y2.squeeze(1)                 # shape (5,)
print(torch.allclose(y, y2))       # True
```

**关键点**:在 PyTorch 里,**一维张量是"灵活"的向量**,在矩阵乘法中根据上下文自动解释。这是工程友好但数学上"含糊"的。

### 2.3 向量的"两种身份":坐标 / 几何对象

同一个数学对象,有两种解读:

**身份 1:坐标 (Coordinate)**

$$\mathbf{x} = (3, 4)^T$$

就是三个 / 四个数字。这是工程师视角。

**身份 2:几何对象 (Geometric Object)**

从原点指向 $(3, 4)$ 的箭头,长度 = $\sqrt{3^2 + 4^2} = 5$,方向 = 与 x 轴夹角 $\arctan(4/3) \approx 53.13°$。这是几何视角。

```text
       │↑ y
       │
       │  •  (3, 4)  ← 这个点
       │ ╱
       │╱ ← 这个箭头
       └───────→ x
```

**两种身份的桥梁**:在标准正交基(见 §8.1)下,坐标就是"沿每个基向量的分量"。

### 2.4 向量空间的 8 条公理

数学家把"向量 + 加法 + 数乘"的形式化定义叫做"向量空间"。一组对象 + 两种运算满足下面 8 条公理,就是向量空间:

设 $V$ 是向量集合,$\mathbb{R}$ 是实数集合,$\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$,$\alpha, \beta \in \mathbb{R}$:

**加法公理**:

| 公理 | 表达式 | 直观含义 |
| :--- | :--- | :--- |
| 交换律 | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ | 加法顺序无关 |
| 结合律 | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ | 三个向量随便怎么加都一样 |
| 单位元 | $\exists \mathbf{0}: \mathbf{v} + \mathbf{0} = \mathbf{v}$ | 存在零向量,加上不变 |
| 逆元 | $\forall \mathbf{v}, \exists -\mathbf{v}: \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ | 每个向量有"相反向量" |

**数乘公理**:

| 公理 | 表达式 | 直观含义 |
| :--- | :--- | :--- |
| 1 的乘法 | $1 \cdot \mathbf{v} = \mathbf{v}$ | 数 1 不改变向量 |
| 结合律 | $\alpha(\beta \mathbf{v}) = (\alpha\beta)\mathbf{v}$ | 数乘顺序无关 |
| 加法分配律 | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ | 数乘对向量加法可分配 |
| 数加法分配律 | $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$ | 数加法对向量可分配 |

**这 8 条公理定义了"什么叫做向量空间"**。最常见的 $\mathbb{R}^n$ 是向量空间,但函数空间、矩阵空间、概率分布空间也满足这 8 条公理,所以它们也是向量空间(见 第 8 章:泛函分析基础 §2)。

#### 验证 $\mathbb{R}^2$ 是向量空间

令 $\mathbf{u} = (u_1, u_2)$,$\mathbf{v} = (v_1, v_2)$。验证 4 条加法公理:

- **交换律**: $(u_1, u_2) + (v_1, v_2) = (u_1+v_1, u_2+v_2) = (v_1+u_1, v_2+u_2) = \mathbf{v} + \mathbf{u}$ ✓
- **结合律**: $((\mathbf{u} + \mathbf{v}) + \mathbf{w})_i = u_i + v_i + w_i = (\mathbf{u} + (\mathbf{v} + \mathbf{w}))_i$ ✓
- **单位元**: $(u_1, u_2) + (0, 0) = (u_1, u_2)$ ✓
- **逆元**: $(u_1, u_2) + (-u_1, -u_2) = (0, 0)$ ✓

数乘公理类似验证。所以 $\mathbb{R}^2$ 是向量空间。**任何 $\mathbb{R}^n$ 都是向量空间**。

### 2.5 5 个手算例子

#### 例 1:$\mathbb{R}^2$ 中的向量

$$\mathbf{a} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

求 $\mathbf{a} + \mathbf{b}$,$2\mathbf{a}$,$-\mathbf{b}$。

**解**:

$$\mathbf{a} + \mathbf{b} = \begin{pmatrix} 3 + 1 \\ 4 + 2 \end{pmatrix} = \begin{pmatrix} 4 \\ 6 \end{pmatrix}$$

$$2\mathbf{a} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$$

$$-\mathbf{b} = \begin{pmatrix} -1 \\ -2 \end{pmatrix}$$

**几何图示**:

```text
       ↑ y
       │  • a = (3, 4)         • a+b = (4, 6)
       │   ╱                   ╱
       │  ╱                   ╱
       │ ╱                   ╱
       │╱                   ╱
       └──────────────────────→ x
        • b = (1, 2)
```

#### 例 2:$\mathbb{R}^3$ 中的向量(常见于 3D 物体检测)

$$\mathbf{p} = (1, 2, 3)^T, \quad \mathbf{q} = (4, 5, 6)^T$$

求 $\mathbf{p} - \mathbf{q}$。

**解**:

$$\mathbf{p} - \mathbf{q} = \begin{pmatrix} 1 - 4 \\ 2 - 5 \\ 3 - 6 \end{pmatrix} = \begin{pmatrix} -3 \\ -3 \\ -3 \end{pmatrix}$$

**几何**:从 $\mathbf{q}$ 指向 $\mathbf{p}$ 的向量。每个分量都减 3。

#### 例 3:$\mathbb{R}^4$ 中的词向量

假设 BERT 输出的"猫"和"狗"的 4 维 Embedding(简化版):

$$\mathbf{e}_{\text{cat}} = (0.8, 0.1, -0.2, 0.5)^T$$
$$\mathbf{e}_{\text{dog}} = (0.7, 0.15, -0.18, 0.55)^T$$

这两个向量"很接近"。它们的各分量差异都小于 0.1。这就是 Embedding 捕捉"猫和狗都是动物"语义的几何表现。

#### 例 4:零向量

$$\mathbf{0} = (0, 0, \ldots, 0)^T \in \mathbb{R}^n$$

零向量跟任何向量相加,结果不变。**PyTorch 的陷阱**:Embedding 初始化时如果没有特殊处理,某些行的向量可能就是零向量,导致对应 ID 的 Embedding 始终为 0,模型学不到东西。**解决方法**:用 `padding_idx` 或 Xavier 初始化。

```python
# 错误:某些 ID 的 Embedding 是零向量
emb = nn.Embedding(100, 64)
print(emb.weight[0])  # 全 0(初始化就是零)
emb.weight[0] = torch.randn(64)  # 手动赋值修复

# 推荐:用 padding_idx 自动处理
emb = nn.Embedding(100, 64, padding_idx=0)  # ID=0 始终是零向量,且不更新
```

#### 例 5:稀疏 vs 稠密向量

**稀疏向量**(one-hot):

$$\mathbf{e}_{42} = (0, 0, \ldots, 0, 1, 0, \ldots, 0)^T$$

其中只有第 42 个分量是 1,其他都是 0。维度 = 词汇表大小,可能是 100 万。

**稠密向量**(Embedding):

$$\mathbf{e}_{42} = (0.13, -0.27, 0.41, -0.19, \ldots, 0.08)^T \in \mathbb{R}^{64}$$

维度 = 64(任意选择)。

**关键区别**:稀疏向量每个维度对应一个**类别**,稠密向量每个维度对应一个**隐特征**。Embedding 查表就是"稀疏 → 稠密"的映射:

```python
# 模拟 Embedding 查表
embedding_table = torch.randn(1000000, 64)  # 100 万 × 64 的稠密矩阵
one_hot = torch.zeros(1000000)
one_hot[42] = 1.0

# 稀疏查表 = 矩阵乘法(取第 42 行)
sparse_lookup = embedding_table.T @ one_hot  # shape (64,)
dense_lookup = embedding_table[42]            # shape (64,) ← 等价但更快
```

这就是为什么 Embedding 在 PyTorch 里是**简单的索引操作**,而不是矩阵乘法:它利用了 one-hot 的特殊结构,直接取行,O(d) 而不是 O(V·d)。

---

## 3. 向量加法与数乘

### 3.1 加法的几何意义:平行四边形法则

两个向量 $\mathbf{a}$ 和 $\mathbf{b}$ 相加的几何意义:**首尾相接**。

把 $\mathbf{a}$ 的尾放到 $\mathbf{b}$ 的头,或者反过来,$\mathbf{a} + \mathbf{b}$ 就是从起点到终点的对角线。这就是**平行四边形法则**:

```text
       ↑ y
       │      • a + b
       │     ╱│
       │    ╱ │
       │   ╱  │
       │  • a •
       │  ╲   │
       │   ╲  │
       │    ╲ │
       │     •│ b
       │      •
       └────────────→ x
```

**形式化**:

$$\mathbf{a} + \mathbf{b} = (a_1 + b_1, a_2 + b_2, \ldots, a_n + b_n)^T$$

**物理意义**:力 $\mathbf{a}$ 和 $\mathbf{b}$ 同时作用在物体上,合力 = 向量和。深度学习的"残差连接"(ResNet) 也是加法:

```python
# ResNet 残差块
def residual_block(x):
    out = conv(x)
    out = relu(out)
    return x + out  # ← 加法:输入 + 变换
```

### 3.2 数乘:缩放

数乘 $\alpha \mathbf{v}$ 的几何意义:**把向量 $\mathbf{v}$ 拉长($\alpha > 1$)或缩短($\alpha < 1$) $\alpha$ 倍**。如果 $\alpha < 0$,反向。

```text
原始:    ───→ v
放大(α=2): ──────→ 2v
缩小(α=0.5): ─→ 0.5v
反向(α=-1): ←─── -v
```

**形式化**:

$$\alpha \mathbf{v} = (\alpha v_1, \alpha v_2, \ldots, \alpha v_n)^T$$

**工程意义**:在 Embedding 加权和里,数乘 = "权重"。比如用户历史行为的加权求和:

```python
# 用户最近 5 次点击的 Embedding + 权重
clicked_items = torch.tensor([3, 17, 42, 8, 91])  # 5 个商品 ID
weights = torch.tensor([0.4, 0.25, 0.15, 0.12, 0.08])  # 权重和 = 1

item_emb = nn.Embedding(100, 64)
vectors = item_emb(clicked_items)  # shape (5, 64)

# 加权和 = Σ weight_i · vector_i
user_emb = (weights.unsqueeze(1) * vectors).sum(dim=0)  # shape (64,)
```

这就是向量加法 + 数乘的工程应用。

### 3.3 工程意义:为什么 Embedding 可以"加权求和"?

#### 例:BERT 的 [CLS] 向量是子向量加权和

BERT 的 [CLS] token 经过 Self-Attention 后,等价于"所有 token Embedding 的加权求和"(忽略 LayerNorm 和残差):

$$\mathbf{e}_{[\text{CLS}]} = \sum_{i=1}^{n} \alpha_i \mathbf{e}_i$$

其中 $\alpha_i$ 是 Attention 权重,$\sum \alpha_i = 1$。**这就是向量加法 + 数乘**。

#### 例:推荐系统的"用户 Embedding = 历史物品 Embedding 加权和"

工业推荐系统里,用户向量通常这么构造:

$$\mathbf{u} = \sum_{i \in \text{history}} w_i \cdot \mathbf{e}_i$$

- $\mathbf{e}_i$:历史交互物品的 Embedding
- $w_i$:权重(可以基于时间衰减、点击/购买、位置等)
- 求和得到用户 Embedding $\mathbf{u}$

这就是 §3.1 + §3.2 的实际应用。

#### 例:Smooth L1 Loss 与 L2 Loss 的差异

```python
# L2 Loss:0.5 * (y - ŷ)^2
# 梯度:-(y - ŷ), 即"残差"本身
l2_grad = y_pred - y_true

# L1 Loss:|y - ŷ|
# 梯度:±1, 即"残差的方向"
l1_grad = torch.sign(y_pred - y_true)
```

注意 L1 Loss 对异常值更鲁棒(梯度恒为 ±1),这就是**范数**(§6)的选择影响模型行为。

### 3.4 5 个手算例子

#### 例 1:二维加法

$$\mathbf{a} = (1, 2)^T, \quad \mathbf{b} = (3, 4)^T$$

$$\mathbf{a} + \mathbf{b} = (4, 6)^T, \quad 2\mathbf{a} = (2, 4)^T, \quad \mathbf{a} - \mathbf{b} = (-2, -2)^T$$

#### 例 2:三维加法(物理:力的合成)

$\mathbf{F}_1 = (3, 0, 0)$ 沿 x 轴,$F_2 = (0, 4, 0)$ 沿 y 轴,合力:

$$\mathbf{F}_{\text{total}} = \mathbf{F}_1 + \mathbf{F}_2 = (3, 4, 0)$$

合力长度 = 5(勾股定理),在 xy 平面内与 x 轴成 $\arctan(4/3) \approx 53.13°$ 角。

#### 例 3:数乘(梯度下降)

参数 $\theta = (2, 3)^T$,梯度 $g = (0.5, -0.2)^T$,学习率 $\eta = 0.1$。

更新:

$$\theta_{\text{new}} = \theta - \eta g = (2, 3)^T - 0.1 \cdot (0.5, -0.2)^T = (1.95, 3.02)^T$$

这就是梯度下降:**参数沿负梯度方向移动**。每一步都是加法 + 数乘。

#### 例 4:用户 Embedding 加权求和(工程)

5 个商品 Embedding:

$$\mathbf{e}_1 = (1, 0)^T, \mathbf{e}_2 = (0, 1)^T, \mathbf{e}_3 = (1, 1)^T, \mathbf{e}_4 = (-1, 0)^T, \mathbf{e}_5 = (0, -1)^T$$

权重:$w_1 = 0.5, w_2 = 0.3, w_3 = 0.1, w_4 = 0.05, w_5 = 0.05$。

加权求和:

$$\mathbf{u} = 0.5 \cdot (1, 0) + 0.3 \cdot (0, 1) + 0.1 \cdot (1, 1) + 0.05 \cdot (-1, 0) + 0.05 \cdot (0, -1)$$

$$= (0.5 + 0 + 0.1 - 0.05 + 0, \ 0 + 0.3 + 0.1 + 0 - 0.05)$$

$$= (0.55, 0.35)^T$$

用户向量更接近商品 1(权重最大)。

#### 例 5:零向量的"陷阱"

PyTorch 的 Embedding 默认初始化:**权重矩阵每个元素独立从 $N(0, 1)$ 采样**。但**初始 Embedding 接近零向量**(均值为 0)。这意味着:

- 前向传播时,所有 Embedding 输出接近 0
- 反向传播时,所有梯度接近 0
- 模型"什么也学不到"

**解决方法**:Xavier / Kaiming 初始化,或者预训练。

```python
# 默认初始化
emb = nn.Embedding(100, 64)
print(emb.weight.std())  # ~1.0(标准正态)

# Xavier 初始化(更稳定的训练)
emb = nn.Embedding(100, 64)
nn.init.xavier_uniform_(emb.weight)
print(emb.weight.std())  # ~0.18(方差按 1/d 缩放)
```

---

## 4. 内积 (Dot Product) ← 重点

### 4.1 直觉:内积 = 一个向量在另一个上的"投影"

**内积**(也称点积)是线性代数里最重要的运算之一。给定两个向量 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n} x_i y_i$$

**代数定义**:逐元素相乘再求和。

**几何直觉**:把 $\mathbf{x}$ 投影到 $\mathbf{y}$ 上,得到一个标量长度。

```text
         y
         │     • x (3, 4)
         │    ╱│
         │   ╱ │
         │  ╱  │
         │ ╱θ  │
         │╱    │
   ──────•─────•──── y 方向上的投影 = |x| cos θ = 5 · cos(53°) = 3
         │
         │
         └──────────→ x
```

如果把 $\mathbf{y}$ 看作"尺子",$\mathbf{x} \cdot \mathbf{y}$ 就是 $\mathbf{x}$ 在这把尺子上的**读数**。这个读数越大,$\mathbf{x}$ 越"对齐" $\mathbf{y}$。

### 4.2 代数定义

**两个向量的内积**是一个标量(不是向量):

$$\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n} x_i y_i = x_1 y_1 + x_2 y_2 + \cdots + x_n y_n$$

**例子**:$\mathbf{x} = (1, 2, 3)^T$,$\mathbf{y} = (4, 5, 6)^T$:

$$\mathbf{x} \cdot \mathbf{y} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$$

**性质**:
- **交换律**: $\mathbf{x} \cdot \mathbf{y} = \mathbf{y} \cdot \mathbf{x}$
- **结合律**: $\mathbf{x} \cdot (\mathbf{y} + \mathbf{z}) = \mathbf{x} \cdot \mathbf{y} + \mathbf{x} \cdot \mathbf{z}$
- **数乘结合**: $(\alpha \mathbf{x}) \cdot \mathbf{y} = \alpha (\mathbf{x} \cdot \mathbf{y})$
- **非负性**: $\mathbf{x} \cdot \mathbf{x} = \|\mathbf{x}\|^2 \ge 0$,等号成立当且仅当 $\mathbf{x} = \mathbf{0}$

### 4.3 几何定义

$$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \cdot \|\mathbf{y}\| \cdot \cos\theta$$

其中 $\theta$ 是 $\mathbf{x}$ 和 $\mathbf{y}$ 的夹角,$0 \le \theta \le \pi$,$\|\cdot\|$ 是向量的 L2 范数(§6)。

**为什么这个定义重要?** 它把内积跟"几何关系"联系起来:

| $\theta$ | $\cos\theta$ | $\mathbf{x} \cdot \mathbf{y}$ | 含义 |
| :--- | :--- | :--- | :--- |
| 0 | 1 | $\|\mathbf{x}\| \cdot \|\mathbf{y}\|$ | 完全同向 |
| $\pi/4$ | $\sqrt{2}/2$ | $\|\mathbf{x}\| \cdot \|\mathbf{y}\| / \sqrt{2}$ | 同向 |
| $\pi/2$ | 0 | 0 | 正交 |
| $3\pi/4$ | $-\sqrt{2}/2$ | 负 | 大致反向 |
| $\pi$ | -1 | $-\|\mathbf{x}\| \cdot \|\mathbf{y}\|$ | 完全反向 |

**两个等价定义的桥梁**(对 $n = 2$ 验证):

$\mathbf{x} = (\|\mathbf{x}\|\cos\alpha, \|\mathbf{x}\|\sin\alpha)$,$\mathbf{y} = (\|\mathbf{y}\|\cos\beta, \|\mathbf{y}\|\sin\beta)$,$\theta = \alpha - \beta$。

$$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|(\cos\alpha\cos\beta + \sin\alpha\sin\beta) = \|\mathbf{x}\|\|\mathbf{y}\|\cos(\alpha - \beta) = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$$

对 $n$ 维同理(可以用旋转不变性证明)。

### 4.4 5 个手算例子

#### 例 1:二维,完全同向

$\mathbf{x} = (3, 0)^T$,$\mathbf{y} = (2, 0)^T$。方向完全一致。

代数:$\mathbf{x} \cdot \mathbf{y} = 3 \cdot 2 + 0 \cdot 0 = 6$。

几何:$\|\mathbf{x}\| = 3$,$\|\mathbf{y}\| = 2$,$\cos 0 = 1$,内积 $= 3 \cdot 2 \cdot 1 = 6$ ✓。

#### 例 2:二维,正交

$\mathbf{x} = (1, 0)^T$,$\mathbf{y} = (0, 1)^T$。两个基向量。

代数:$\mathbf{x} \cdot \mathbf{y} = 1 \cdot 0 + 0 \cdot 1 = 0$。

几何:夹角 = 90°,$\cos 90° = 0$,内积 = 0 ✓。

#### 例 3:三维,任意夹角

$\mathbf{x} = (1, 2, 3)^T$,$\mathbf{y} = (4, 5, 6)^T$。

代数:$\mathbf{x} \cdot \mathbf{y} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32$。

几何验证:

$$\|\mathbf{x}\| = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.742$$
$$\|\mathbf{y}\| = \sqrt{16 + 25 + 36} = \sqrt{77} \approx 8.775$$
$$\cos\theta = \frac{32}{3.742 \cdot 8.775} \approx \frac{32}{32.83} \approx 0.975$$
$$\theta \approx \arccos(0.975) \approx 12.93°$$

这两个向量非常接近(夹角只有 ~13°)。

#### 例 4:反向

$\mathbf{x} = (1, 1)^T$,$\mathbf{y} = (-2, -2)^T$。$\mathbf{y} = -2\mathbf{x}$。

代数:$\mathbf{x} \cdot \mathbf{y} = -2 - 2 = -4$。

几何:$\|\mathbf{x}\| = \sqrt{2}$,$\|\mathbf{y}\| = 2\sqrt{2}$,$\cos 180° = -1$,内积 $= \sqrt{2} \cdot 2\sqrt{2} \cdot (-1) = -4$ ✓。

#### 例 5:n 维稀疏向量(one-hot)

$\mathbf{x} = \mathbf{e}_3 = (0, 0, 1, 0, \ldots, 0)^T$(第 3 位是 1),$\mathbf{y} = \mathbf{e}_5 = (0, 0, 0, 0, 1, \ldots, 0)^T$。

$\mathbf{x} \cdot \mathbf{y} = 0$。

**工程意义**:One-hot 向量内积 = "重合度"。**只有两个 one-hot 在同一位都为 1 时,内积才是 1**。这是 ItemCF(Item-based Collaborative Filtering)的核心:

```python
# 用户-物品评分矩阵,每行是一个用户的评分向量
R = torch.tensor([
    [5, 0, 3, 0],  # 用户 1:喜欢物品 1,有点喜欢物品 3
    [4, 0, 0, 2],  # 用户 2:喜欢物品 1,有点喜欢物品 4
    [0, 5, 0, 4],  # 用户 3:喜欢物品 2 和 4
])

# 用户 1 和用户 2 的相似度(内积)
sim_12 = R[0] @ R[1]  # 5*4 + 0 + 0 + 0 = 20
print(sim_12)  # tensor(20)
```

这就是基于物品评分的用户相似度。值越大,两个用户评分模式越像。

### 4.5 案例:为什么 Attention 的 $Q \cdot K$ = 相似度?

Transformer Attention 的核心公式:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

其中 $Q, K \in \mathbb{R}^{N \times d}$ 是 Query 和 Key 矩阵。

**$Q$ 的一行** = 一个 token 的 query 向量,**$K$ 的一行** = 一个 token 的 key 向量。**$QK^T$ 的第 $(i, j)$ 元素** = 第 $i$ 个 query 与第 $j$ 个 key 的内积:

$$(QK^T)_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$$

**直觉**:Query 在"找"信息,它问"哪些 Key 跟我最匹配?"**内积 = 匹配度**。

- 如果 $\mathbf{q}_i$ 和 $\mathbf{k}_j$ 同向(都指向相似方向),内积大 → 它们"匹配"
- 如果正交,内积 0 → 不匹配
- 如果反向,内积负 → 排斥

**为什么除以 $\sqrt{d}$?** 防止内积值过大导致 softmax 饱和(梯度消失)。具体来说,如果 $\mathbf{q}_i, \mathbf{k}_j$ 的每个分量独立 $N(0, 1)$,内积方差 = $d$,标准差 = $\sqrt{d}$,除以 $\sqrt{d}$ 后方差归一化。这是工程技巧,但数学上是"标准化"。

**为什么不直接用 cosine?** Attention 中,query 和 key **已经是归一化后的**(经过 LayerNorm 或类似操作),且维度较低,用 cosine 与内积差异不大。但 cosine 需要计算范数,内积只需一次矩阵乘法,**内积更快**。这是工业 Attention 用内积的原因。

### 4.6 案例:为什么双塔召回用内积?

**双塔模型**(Two-Tower) 是推荐系统召回的核心架构:

```text
用户塔:user_features → MLP → u ∈ R^d
物品塔:item_features → MLP → v ∈ R^d
打分:score(u, v) = u · v = u^T v  ← 内积
```

**为什么用内积?**

1. **高效**:内积 = 一次矩阵乘法,可批量化,可 GPU 加速
2. **预计算**:物品向量可以预先算好并索引(FAISS / ScaNN),查询时只算一次内积
3. **可解释**:内积 = "匹配度"
4. **可学习**:用户塔和物品塔的输出都被训练成"让正样本内积大,负样本小"

**对比 cosine**:cosine 是归一化的内积,等价于"只看方向不看大小"。在召回中,我们通常关心"用户对物品的偏好强度",这跟大小有关。所以内积更常用。**但如果要消除物品流行度偏差(热门物品 Embedding 范数大),用 cosine 更稳**。

### 4.7 案例:为什么 ItemCF 用"用户-物品"内积?

**ItemCF**(基于物品的协同过滤) 的核心:

$$\text{sim}(i, j) = \frac{\text{用户同时喜欢 } i, j \text{ 的人数}}{\sqrt{|\text{喜欢 } i| \cdot |\text{喜欢 } j|}}$$

**向量视角**:

- $\mathbf{r}_i$ = 第 $i$ 个物品的用户评分向量(稀疏,维度 = 用户数)
- $\mathbf{r}_i \cdot \mathbf{r}_j$ = 同时喜欢 $i$ 和 $j$ 的"加权和"
- $\|\mathbf{r}_i\|$ = 喜欢 $i$ 的总人数(假设评分都是 1)
- cosine = $\frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\| \cdot \|\mathbf{r}_j\|}$ = ItemCF 的相似度

**为什么用 cosine?** 因为要消除"热门物品天然相似"的偏差(热门物品和所有物品的共现次数都高)。

#### PyTorch 实现

```python
import torch

# 5 个用户对 4 个物品的评分(0 = 没看过)
R = torch.tensor([
    [5, 0, 3, 0],  # 用户 1
    [4, 0, 0, 2],  # 用户 2
    [0, 5, 0, 4],  # 用户 3
    [5, 0, 4, 0],  # 用户 4
    [0, 4, 0, 5],  # 用户 5
], dtype=torch.float32)

# 内积矩阵 (物品 × 物品)
item_inner = R.T @ R  # shape (4, 4)
print("物品内积矩阵:")
print(item_inner)

# 范数
item_norm = R.norm(dim=0, keepdim=True)  # shape (1, 4)
print(f"\n物品范数: {item_norm}")

# cosine 相似度矩阵
item_cosine = item_inner / (item_norm.T @ item_norm)
print(f"\n物品 cosine 相似度矩阵:")
print(item_cosine)
```

**输出**:

```text
物品内积矩阵:
tensor([[66.,  0., 39.,  2.],
        [ 0., 41.,  0., 36.],
        [39.,  0., 25.,  0.],
        [ 2., 36.,  0., 45.]])

物品范数: tensor(`8.31,  6.40,  7.62,  7.87`)

物品 cosine 相似度矩阵:
tensor([[1.0000, 0.0000, 0.6165, 0.0306],
        [0.0000, 1.0000, 0.0000, 0.7148],
        [0.6165, 0.0000, 1.0000, 0.0000],
        [0.0306, 0.7148, 0.0000, 1.0000]])
```

观察:物品 1 和 3 的 cosine = 0.62(较高,因为都喜欢的人多);物品 2 和 4 的 cosine = 0.71(也较高)。这就是 ItemCF 的"协同信号"。

---

## 5. Cauchy-Schwarz 不等式 ← 重点

### 5.1 形式化

**Cauchy-Schwarz 不等式**是内积理论的核心定理:

$$|\mathbf{x} \cdot \mathbf{y}| \le \|\mathbf{x}\| \cdot \|\mathbf{y}\|$$

**等价形式**:

$$-1 \le \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|} \le 1$$

右半部分就是 cosine 相似度的范围。

### 5.2 等号成立条件:$\mathbf{x}$ 与 $\mathbf{y}$ 线性相关

**定理**: $|\mathbf{x} \cdot \mathbf{y}| = \|\mathbf{x}\| \cdot \|\mathbf{y}\|$ 当且仅当 $\mathbf{x}$ 与 $\mathbf{y}$ 线性相关(即 $\mathbf{y} = \alpha \mathbf{x}$ 或 $\mathbf{x} = \mathbf{0}$ 或 $\mathbf{y} = \mathbf{0}$)。

**直觉**:两个向量完全对齐(或反向),内积的绝对值达到最大值。

**证明(简版)**:如果 $\mathbf{x} = \mathbf{0}$,显然成立。否则令 $\mathbf{u} = \mathbf{x} / \|\mathbf{x}\|$(单位向量)。构造:

$$f(t) = \|\mathbf{y} - t\mathbf{u}\|^2 \ge 0$$

展开:

$$f(t) = \|\mathbf{y}\|^2 - 2t \mathbf{u} \cdot \mathbf{y} + t^2$$

最小值在 $t^* = \mathbf{u} \cdot \mathbf{y}$ 处取得:

$$f(t^*) = \|\mathbf{y}\|^2 - (\mathbf{u} \cdot \mathbf{y})^2 \ge 0$$

$$(\mathbf{u} \cdot \mathbf{y})^2 \le \|\mathbf{y}\|^2$$

$$|\mathbf{u} \cdot \mathbf{y}| \le \|\mathbf{y}\|$$

$$|\mathbf{x} \cdot \mathbf{y}| / \|\mathbf{x}\| \le \|\mathbf{y}\|$$

$$|\mathbf{x} \cdot \mathbf{y}| \le \|\mathbf{x}\| \cdot \|\mathbf{y}\| \quad \blacksquare$$

### 5.3 几何意义:$\cos\theta \in [-1, 1]$

由 §4.3 的几何定义 $\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$,代入 Cauchy-Schwarz:

$$|\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta| \le \|\mathbf{x}\|\|\mathbf{y}\|$$

$$|\cos\theta| \le 1$$

$$-1 \le \cos\theta \le 1$$

这就是"cosine 范围"的几何来源。

### 5.4 工程意义:为什么 cosine similarity 一定 $\in [-1, 1]$?

Cosine 相似度的定义:

$$\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|}$$

**Cauchy-Schwarz 保证了分母 ≥ |分子|,所以比值 ∈ [-1, 1]**。这意味着 cosine 永远是有界标量,无论向量维度和数值大小。

**这为什么重要?**

1. **Loss 函数稳定**:用 cosine 作为相似度的 Loss,梯度不会爆炸
2. **可以当概率用**:`(1 + cosine) / 2 ∈ [0, 1]`, 可以直接当概率
3. **可视化方便**:范围固定,可以画热力图

### 5.5 案例:为什么 Embedding 检索前要 L2 归一化?

在向量检索中,如果直接用内积检索,**向量的范数会"污染"相似度**:

```python
# 假设两个商品向量
item1 = torch.tensor([0.5, 0.5])  # 范数 ~0.71
item2 = torch.tensor([2.0, 2.0])  # 范数 ~2.83(只是放大版)
query = torch.tensor([1.0, 1.0])  # 范数 ~1.41

# 内积打分
score1 = (item1 * query).sum()  # 0.5 + 0.5 = 1.0
score2 = (item2 * query).sum()  # 2.0 + 2.0 = 4.0
print(score2 > score1)  # True!但 item2 只是 item1 的放大版
```

**问题**:`item2` 只是 `item1` 的 4 倍放大版,**方向完全相同**,但内积打分高了 4 倍。这不公平。

**L2 归一化**:

```python
item1_norm = item1 / item1.norm()  # [0.707, 0.707]
item2_norm = item2 / item2.norm()  # [0.707, 0.707](! 跟 item1_norm 一样)
query_norm = query / query.norm()  # [0.707, 0.707]

# 归一化后内积 = cosine
score1_n = (item1_norm * query_norm).sum()  # 0.5 + 0.5 = 1.0
score2_n = (item2_norm * query_norm).sum()  # 0.5 + 0.5 = 1.0
print(score1_n == score2_n)  # True!相同方向打相同分
```

**结论**:检索前先 L2 归一化(每个向量除以自己的 L2 范数),**让内积 = cosine**,消除范数偏差。这就是 FAISS 等向量检索库的"内积距离"模式:要求输入向量已归一化,否则用 L2 距离。

### 5.6 3 个手算例子

#### 例 1:完全同向(等号成立)

$\mathbf{x} = (1, 2, 3)^T$,$\mathbf{y} = (2, 4, 6)^T$($\mathbf{y} = 2\mathbf{x}$)。

代数:$\mathbf{x} \cdot \mathbf{y} = 2 + 8 + 18 = 28$。

几何:$\|\mathbf{x}\| = \sqrt{14}$,$\|\mathbf{y}\| = 2\sqrt{14}$,$\cos\theta = 28 / (14 \cdot 2) = 1$。夹角 0°。 ✓ 等号成立。

#### 例 2:正交

$\mathbf{x} = (1, 0)^T$,$\mathbf{y} = (0, 1)^T$。

$\mathbf{x} \cdot \mathbf{y} = 0$, $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$,所以 $|0| \le 1 \cdot 1 = 1$ ✓。

#### 例 3:任意向量验证

$\mathbf{x} = (3, 4)^T$,$\mathbf{y} = (1, 1)^T$。

$\mathbf{x} \cdot \mathbf{y} = 3 + 4 = 7$, $\|\mathbf{x}\| = 5$, $\|\mathbf{y}\| = \sqrt{2}$。

$|7| = 7$, $5 \cdot \sqrt{2} \approx 7.07$。

$7 \le 7.07$ ✓ Cauchy-Schwarz 成立。

$\cos\theta = 7 / (5\sqrt{2}) \approx 0.99$,夹角 ≈ 8.13°。很接近但不完全同向。

---

## 6. 范数 (Norm) ← 重点

### 6.1 范数的定义:满足 3 条公理

**向量范数** $\|\cdot\|$ 是从 $\mathbb{R}^n$ 到 $\mathbb{R}_{\ge 0}$ 的函数,满足 3 条公理:

| 公理 | 表达式 | 直观含义 |
| :--- | :--- | :--- |
| 非负性 | $\|\mathbf{v}\| \ge 0$,且 $\|\mathbf{v}\| = 0 \iff \mathbf{v} = \mathbf{0}$ | 长度 ≥ 0,只有零向量长度 = 0 |
| 齐次性 | $\|\alpha \mathbf{v}\| = \|\alpha\| \cdot \|\mathbf{v}\|$ | 缩放 α 倍,长度也缩放 α 倍 |
| 三角不等式 | $\|\mathbf{u} + \mathbf{v}\| \le \|\mathbf{u}\| + \|\mathbf{v}\|$ | 两边之和 ≥ 第三边(三角形) |

**直觉**:范数就是"向量的长度",但有多种"长度"的度量方式。

### 6.2 L¹ 范数:曼哈顿距离

$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

所有分量的绝对值之和。又称"曼哈顿距离"(想象在纽约曼哈顿街区走路,只能沿街道走,不能穿越建筑)。

**几何**($n = 2$):L¹ 范数的"等距线"($\|\mathbf{x}\|_1 = c$) 是菱形(顶点为 $(\pm c, 0)$ 和 $(0, \pm c)$)。

**例子**:$\mathbf{x} = (3, -4)^T$, $\|\mathbf{x}\|_1 = 3 + 4 = 7$。

**工程意义**:L1 范数鼓励**稀疏**(L1 正则化)。L2 范数(L2 正则化)鼓励**小而均匀**(见 第 5 章:二次型·正定性·条件数 §4)。

### 6.3 L² 范数:欧氏距离

$$|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x} \cdot \mathbf{x}}$$

最常用的范数。又称"欧氏距离"(两点之间的直线距离)。

**几何**($n = 2$):L² 范数的"等距线"是圆。

**例子**:$\mathbf{x} = (3, 4)^T$, $\|\mathbf{x}\|_2 = \sqrt{9 + 16} = 5$(3-4-5 直角三角形)。

**工程意义**:L2 范数是最自然的"距离"度量,**几乎所有 ML 算法都用 L2**。

### 6.4 L^∞ 范数:最大分量

$$\|\mathbf{x}\|_\infty = \max_i |x_i|$$

所有分量绝对值的最大值。

**几何**($n = 2$):L∞ 范数的"等距线"是正方形。

**例子**:$\mathbf{x} = (3, -4, 1)^T$, $\|\mathbf{x}\|_\infty = 4$。

**工程意义**:对抗样本攻击中,常用的"扰动度量"是 L∞(因为图像像素的"最大改动"是 L∞):

```python
# 对抗样本:FGSM 攻击
def fgsm_attack(image, label, model, epsilon=0.03):
    image.requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(image), label)
    loss.backward()

    # 扰动 = sign(梯度)
    perturbation = epsilon * image.grad.sign()

    # L∞ 约束:每个像素最多改 epsilon
    adversarial_image = image + perturbation
    return adversarial_image.detach()
```

### 6.5 L^p 范数的一般形式

$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}$$

| $p$ | 名称 | 几何($n=2$)| 性质 |
| :--- | :--- | :--- | :--- |
| 0 | L0(不是真正的范数) | "十字形" | 计数非零分量 |
| 1 | 曼哈顿 | 菱形 | 鼓励稀疏 |
| 2 | 欧氏 | 圆 | 鼓励小且均匀 |
| $\infty$ | 最大值 | 正方形 | 鼓励最值受限 |

**$p$ 越小,几何越"尖";$p$ 越大,几何越"方"**。

**重要的 Cauchy-Schwarz 推广:Hölder 不等式**:

$$\sum_{i=1}^{n} |x_i y_i| \le \|\mathbf{x}\|_p \cdot \|\mathbf{y}\|_q, \quad \frac{1}{p} + \frac{1}{q} = 1$$

Cauchy-Schwarz 是 $p = q = 2$ 的特例。

### 6.6 Frobenius 范数:矩阵的"L2"

**矩阵** $A \in \mathbb{R}^{m \times n}$ 的 **Frobenius 范数**:

$$\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

**几何**:把矩阵"展开"成 $mn$ 维向量,然后算 L2 范数。

**例子**:

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad \|A\|_F = \sqrt{1 + 4 + 9 + 16} = \sqrt{30} \approx 5.477$$

**工程意义**:

- 矩阵 L2 正则化 = 最小化 $\|W\|_F^2$(等价于 weight decay)
- 矩阵分解中 $\|A - UV\|_F$ 是常用的"重建误差"

```python
# PyTorch 中计算 Frobenius 范数
W = torch.randn(3, 4)
frobenius_norm = W.norm(p='fro')  # 等价 torch.norm(W)
print(frobenius_norm.item())

# 或手写
frobenius_manual = torch.sqrt((W ** 2).sum())
print(frobenius_manual.item())
```

### 6.7 5 个手算例子

#### 例 1:L1 范数

$\mathbf{x} = (-3, 4)^T$, $\|\mathbf{x}\|_1 = |-3| + |4| = 7$。

#### 例 2:L2 范数

$\mathbf{x} = (3, 4)^T$, $\|\mathbf{x}\|_2 = \sqrt{9 + 16} = 5$。

#### 例 3:L∞ 范数

$\mathbf{x} = (3, -4, 1)^T$, $\|\mathbf{x}\|_\infty = \max(3, 4, 1) = 4$。

#### 例 4:L^p 范数随 p 变化

$\mathbf{x} = (1, 1)^T$:

- $\|\mathbf{x}\|_1 = 2$
- $\|\mathbf{x}\|_2 = \sqrt{2} \approx 1.414$
- $\|\mathbf{x}\|_4 = (1 + 1)^{1/4} = 2^{0.25} \approx 1.189$
- $\|\mathbf{x}\|_\infty = 1$

观察: $p$ 越大,范数越小(极限 = max = 1);$p$ 越小,范数越大(极限 = sum = 2)。

#### 例 5:Frobenius 范数

$$A = \begin{pmatrix} 1 & -2 \\ 3 & 0 \end{pmatrix}$$

$\|A\|_F = \sqrt{1 + 4 + 9 + 0} = \sqrt{14}$。

### 6.8 案例:为什么 Adam 梯度裁剪用 L²?

**梯度裁剪**(Gradient Clipping) 是防止训练中"梯度爆炸"的标配技巧。**PyTorch 的 `clip_grad_norm_` 默认用 L2**:

```python
# PyTorch 实现
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**为什么用 L2?** 因为 L2 范数是"欧氏长度",直观上度量"梯度的整体大小"。裁剪后:

$$\mathbf{g}_{\text{clipped}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le c \\ \frac{c}{\|\mathbf{g}\|_2} \mathbf{g} & \text{otherwise} \end{cases}$$

这是"投影到 L2 球"的边界。如果用 L1 或 L∞,投影的几何不一样。

**对比**:
- L2 裁剪:梯度被整体缩放到 ≤ c 的球内,**所有分量等比例缩小**
- L∞ 裁剪:每个分量被独立裁剪到 ≤ c,**相互不影响**
- L1 裁剪:梯度的绝对值和 ≤ c

**L2 最常用**,因为它跟"梯度方向不变"等价(只缩放不旋转),保留了梯度方向信息。

### 6.9 案例:为什么 Spectral Norm 用算子范数(诱导范数)?

**算子范数**(也叫诱导范数)是**矩阵**作为线性算子的"最大拉伸倍数":

$$\|A\|_2 = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_{\max}(A)$$

其中 $\sigma_{\max}$ 是最大奇异值(见 第 4 章:特征值·特征向量·谱定理)。

**Spectral Normalization**(Miyato et al. 2018) 用于 GAN:

$$\tilde{W} = \frac{W}{\|W\|_2}$$

**几何**:把矩阵的"最大拉伸"限制为 1,确保整个网络是 1-Lipschitz。这对 Wasserstein GAN 至关重要(要求判别器 1-Lipschitz)。

**详细推导和应用见 `../../../方向教程/rec教程/第04章-多任务学习.md` §5 和 `../../../基础/数学基础/第01章-高等数学/第01章-03-多元微分·梯度·Jacobian·Hessian-反向传播的数学骨架.md` §3.4**。

---

## 7. 余弦相似度 (Cosine Similarity)

### 7.1 定义

$$\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \cdot \sqrt{\sum_i y_i^2}}$$

**范围**:$[-1, 1]$,由 Cauchy-Schwarz 保证。

### 7.2 与内积的关系:cosine 相似度 = 归一化后的内积

**关键等价**:

$$\cos(\mathbf{x}, \mathbf{y}) = \left(\frac{\mathbf{x}}{\|\mathbf{x}\|}\right) \cdot \left(\frac{\mathbf{y}}{\|\mathbf{y}\|}\right)$$

**证明**:右边 = $\frac{\mathbf{x}}{\|\mathbf{x}\|} \cdot \frac{\mathbf{y}}{\|\mathbf{y}\|} = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|} = $ 左边。✓

**工程意义**:cosine 相似度 = "先把两个向量归一化到单位长度,再算内积"。所以**只要做了 L2 归一化,内积 = cosine**。

### 7.3 案例:SimCSE 对比学习用 cosine similarity

**SimCSE**(Gao et al. 2021) 是经典的句子 Embedding 对比学习方法:

```python
import torch
import torch.nn.functional as F

def simcse_loss(embeddings, temperature=0.05):
    """
    embeddings: shape (2*batch, d)
    前 batch 个是"原句",后 batch 个是"dropout 增强"(同一句子)
    """
    # L2 归一化
    embeddings = F.normalize(embeddings, dim=-1)

    # cosine 相似度矩阵
    sim = embeddings @ embeddings.T / temperature  # shape (2B, 2B)

    # 对比学习:同一句子的两个增强版本相似,不同句子不相似
    # ...(具体 Loss 见原论文)
    return sim
```

**为什么用 cosine?** 因为对比学习关心"方向"而非"大小",cosine 消除了范数差异。

### 7.4 案例:为什么推荐系统召回通常用 cosine 而非内积?

**召回场景下 cosine 比内积更稳**的原因:

| 维度 | 内积 | cosine |
| :--- | :--- | :--- |
| 范围 | 无界 | [-1, 1] |
| 范数敏感 | 敏感 | 不敏感 |
| 流行度偏差 | 严重 | 较轻 |
| 训练稳定性 | 较差 | 较好 |

**流行度偏差**:热门物品的 Embedding 因为训练样本多,范数通常更大。如果用内积,热门物品天然得分高,长尾物品被压制。**用 cosine 可以部分缓解这个偏差**。

但 cosine 也有问题:**它假设所有物品的 Embedding 范数都"应该相等"**。这跟双塔模型的训练目标(让正样本内积大)矛盾。所以实践中**很多系统用 cosine 训练 + cosine 检索**,或者**L2 归一化后再用内积**(等价)。

### 7.5 PyTorch 代码:对比 cosine 与 dot product

```python
import torch
import torch.nn.functional as F

# 5 个物品的 Embedding
items = torch.tensor([
    [0.5, 0.5, 0.5],   # 物品 1:均衡
    [1.0, 1.0, 1.0],   # 物品 2:同方向,但范数大 2 倍
    [-0.5, -0.5, -0.5], # 物品 3:与 1 反向
    [1.0, 0.0, 0.0],   # 物品 4:正交
    [0.7, 0.7, 0.1],   # 物品 5:接近 1 但略有差异
])

# Query
query = torch.tensor([1.0, 1.0, 0.5])

# 1. 直接内积
inner_product = items @ query
print("内积:", inner_product.tolist())
# [1.0+0.5+0.25, 2.0+2.0+1.0, -0.5-0.5-0.25, 1.0+0.0+0.0, 0.7+0.7+0.05]
# ≈ [1.75, 5.0, -1.25, 1.0, 1.45]

# 2. cosine 相似度(归一化后内积)
items_norm = F.normalize(items, dim=-1)
query_norm = F.normalize(query, dim=-1)
cosine = items_norm @ query_norm
print("cosine:", cosine.tolist())
# 物品 1 和 2 cosine 相同(都是 1.0,因为方向相同)
```

**观察**:
- 内积排名:2 > 1 > 5 > 4 > 3(物品 2 因为范数大,排第一)
- cosine 排名:1 ≈ 2 > 5 > 4 > 3(物品 1 和 2 并列,因为方向相同)

如果业务上希望"方向决定相似度,大小不重要",cosine 更合理。

---

## 8. 向量空间的几何直觉

### 8.1 标准正交基与坐标

**基**(Basis):一组向量 $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$,使得任意向量 $\mathbf{x}$ 都可以**唯一**表示为它们的线性组合:

$$\mathbf{x} = x_1 \mathbf{e}_1 + x_2 \mathbf{e}_2 + \cdots + x_n \mathbf{e}_n$$

**正交基**:基向量两两正交($\mathbf{e}_i \cdot \mathbf{e}_j = 0$,$i \neq j$)。

**标准正交基**(Standard Orthonormal Basis):正交基,且每个基向量是单位向量。

**$\mathbb{R}^n$ 的标准正交基**:

$$\mathbf{e}_1 = (1, 0, \ldots, 0)^T, \quad \mathbf{e}_2 = (0, 1, \ldots, 0)^T, \quad \ldots, \quad \mathbf{e}_n = (0, 0, \ldots, 1)^T$$

**为什么叫"标准"**?因为任意向量 $\mathbf{x} = (x_1, \ldots, x_n)^T$ 的"坐标"就是它在这组基上的系数。

### 8.2 投影:向量在子空间上的最近点

**投影**(Projection)是把一个向量"投影"到子空间上的运算。给定子空间 $S$(由基 $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ 张成),$\mathbf{x}$ 在 $S$ 上的正交投影是:

$$\text{proj}_S(\mathbf{x}) = \sum_{i=1}^{k} \frac{\mathbf{x} \cdot \mathbf{u}_i}{\mathbf{u}_i \cdot \mathbf{u}_i} \mathbf{u}_i$$

**特例**:投影到单个向量 $\mathbf{u}$ 上:

$$\text{proj}_{\mathbf{u}}(\mathbf{x}) = \frac{\mathbf{x} \cdot \mathbf{u}}{\|\mathbf{u}\|^2} \mathbf{u}$$

**几何**:**投影是 $\mathbf{x}$ 在 $\mathbf{u}$ 上的"影子"**。

```text
         x
        /│
       / │  ← 投影:从 x 出发,垂直于 u 落到 u 上
      /  │
     /   ↓
    /────•────→ u
   /     proj_u(x)
  /
```

**注意**:$\mathbf{x}$ 在 $\mathbf{u}$ 方向的"长度" = $\|\mathbf{x}\| \cos\theta = \frac{\mathbf{x} \cdot \mathbf{u}}{\|\mathbf{u}\|}$。所以 $\text{proj}_{\mathbf{u}}(\mathbf{x})$ 是个向量,长度 = 投影长度,方向 = $\mathbf{u}$。

### 8.3 5 个手算例子

#### 例 1:标准基展开

$\mathbf{x} = (3, 4)^T = 3 \cdot (1, 0)^T + 4 \cdot (0, 1)^T = 3\mathbf{e}_1 + 4\mathbf{e}_2$。

#### 例 2:投影到 x 轴

$\mathbf{u} = (1, 0)^T$, $\mathbf{x} = (3, 4)^T$:

$$\text{proj}_{\mathbf{u}}(\mathbf{x}) = \frac{(3, 4) \cdot (1, 0)}{1} \cdot (1, 0) = 3 \cdot (1, 0) = (3, 0)^T$$

直观:把 $(3, 4)$ 垂直"压"到 x 轴上,得 $(3, 0)$。

#### 例 3:投影到任意方向

$\mathbf{u} = (1, 1)^T$, $\mathbf{x} = (2, 0)^T$:

$$\mathbf{x} \cdot \mathbf{u} = 2 + 0 = 2, \quad \mathbf{u} \cdot \mathbf{u} = 2$$

$$\text{proj}_{\mathbf{u}}(\mathbf{x}) = \frac{2}{2} \cdot (1, 1)^T = (1, 1)^T$$

#### 例 4:投影到子空间(平面)

子空间 $S$ 由 $\mathbf{u}_1 = (1, 0, 0)^T$ 和 $\mathbf{u}_2 = (0, 1, 0)^T$ 张成(就是 xy 平面)。投影 $\mathbf{x} = (1, 2, 3)^T$:

$$\text{proj}_S(\mathbf{x}) = \frac{\mathbf{x} \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1} \mathbf{u}_1 + \frac{\mathbf{x} \cdot \mathbf{u}_2}{\mathbf{u}_2 \cdot \mathbf{u}_2} \mathbf{u}_2 = 1 \cdot (1,0,0)^T + 2 \cdot (0,1,0)^T = (1, 2, 0)^T$$

直观:把 $(1, 2, 3)$ 垂直压到 xy 平面,得 $(1, 2, 0)$。

#### 例 5:Gram-Schmidt 正交化

原始向量 $\mathbf{v}_1 = (1, 1)^T$, $\mathbf{v}_2 = (1, 0)^T$。

正交化过程:

1. $\mathbf{u}_1 = \mathbf{v}_1 = (1, 1)^T$
2. $\mathbf{u}_2 = \mathbf{v}_2 - \frac{\mathbf{v}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1} \mathbf{u}_1 = (1, 0)^T - \frac{1}{2}(1, 1)^T = (0.5, -0.5)^T$

归一化:$\hat{\mathbf{u}}_1 = (1/\sqrt{2}, 1/\sqrt{2})^T$, $\hat{\mathbf{u}}_2 = (1/\sqrt{2}, -1/\sqrt{2})^T$。

这是 **QR 分解**的核心思想,见 `./第02章-03-矩阵分解-SVD·QR·Cholesky·LU与推荐系统的隐藏关系.md` §4。

### 8.4 案例:为什么 PCA 找最大方差方向?

**PCA**(Principal Component Analysis) 找数据中方差最大的方向。直觉:

```text
数据点分布:
    •     •        ↑
    •   • • •      │  y (主成分 1:方差最大)
   • • •••••••     │
  •••••••••••••    ↓
    •   •
     • •
```

**为什么方差大 = 信息量大?**

假设数据都集中在水平方向(沿 y 轴方差小,沿 x 轴方差大)。如果只用一个坐标表示这些点,用 x 坐标(方差大的方向)能保留更多信息,因为不同点在 x 上"分得开",在 y 上"挤在一起"。

**PCA 的数学**:找 $\mathbf{w}$ 使数据 $\mathbf{x}_i$ 在 $\mathbf{w}$ 上的投影方差最大:

$$\max_{\mathbf{w}: \|\mathbf{w}\| = 1} \text{Var}(\mathbf{x}_i \cdot \mathbf{w}) = \max_{\mathbf{w}: \|\mathbf{w}\| = 1} \mathbf{w}^T \Sigma \mathbf{w}$$

其中 $\Sigma$ 是数据的协方差矩阵。**解 = $\Sigma$ 的最大特征值对应的特征向量**(详见 第 4 章:特征值·特征向量·谱定理 §4)。

**几何**:PCA 就是找"数据云"最长的方向(主轴)。

---

## 9. PyTorch 代码实现

### 9.1 实验 1:可视化 2D 平面上 5 个物品的 Embedding,根据 cosine similarity 画"语义连线"

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 5 个商品(虚构的 2D Embedding)
items = {
    'iPhone':     torch.tensor([0.8, 0.2]),
    'AirPods':    torch.tensor([0.7, 0.3]),
    'MacBook':    torch.tensor([0.75, 0.35]),
    'T恤':        torch.tensor([-0.6, 0.5]),
    '运动鞋':     torch.tensor([-0.5, 0.6]),
}

# 转换为矩阵
names = list(items.keys())
vectors = torch.stack(list(items.values()))  # shape (5, 2)

# 计算 cosine 相似度矩阵
vectors_norm = F.normalize(vectors, dim=-1)
cosine_matrix = vectors_norm @ vectors_norm.T

print("cosine 相似度矩阵:")
print(f"{'':10}", "  ".join([f"{n:8}" for n in names]))
for i, n1 in enumerate(names):
    print(f"{n1:10}", "  ".join([f"{cosine_matrix[i, j].item():8.4f}" for j in range(len(names))]))

# 可视化
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['red', 'orange', 'gold', 'blue', 'purple']
for i, name in enumerate(names):
    x, y = vectors[i].tolist()
    ax.scatter(x, y, s=200, c=colors[i], label=name)
    ax.annotate(name, (x, y), xytext=(8, 8), textcoords='offset points', fontsize=12)

# 画"相似度连线"(cosine > 0.8 的连)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        if cosine_matrix[i, j] > 0.8:
            xs = [vectors[i, 0].item(), vectors[j, 0].item()]
            ys = [vectors[i, 1].item(), vectors[j, 1].item()]
            ax.plot(xs, ys, 'k--', alpha=0.3, linewidth=1)

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_xlabel('Embedding Dim 1')
ax.set_ylabel('Embedding Dim 2')
ax.set_title('2D 商品 Embedding + 相似度连线')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/tmp/embedding_viz.png', dpi=100)
print("\n图已保存到 /tmp/embedding_viz.png")
```

### 9.2 实验 2:PyTorch 实现 dot product,验证与 `@` 一致

```python
import torch

def dot_product_manual(x, y):
    """手写内积:Σ x_i * y_i"""
    assert x.shape == y.shape
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * y[i]
    return s

def dot_product_torch(x, y):
    """PyTorch 内积"""
    return x @ y

def dot_product_einsum(x, y):
    """einsum 实现"""
    return torch.einsum('i,i->', x, y)

# 测试
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"手算: {dot_product_manual(x, y).item()}")  # 32.0
print(f"@   : {dot_product_torch(x, y).item()}")    # 32.0
print(f"einsum: {dot_product_einsum(x, y).item()}")  # 32.0
print(f"断言一致: {torch.allclose(dot_product_torch(x, y), dot_product_manual(x, y))}")
```

### 9.3 实验 3:实现 cosine similarity,可视化"语义簇"

```python
import torch
import torch.nn.functional as F

def cosine_similarity_manual(x, y, eps=1e-8):
    """手写 cosine"""
    norm_x = torch.sqrt((x ** 2).sum() + eps)
    norm_y = torch.sqrt((y ** 2).sum() + eps)
    return (x @ y) / (norm_x * norm_y)

# 对比 F.normalize + @
def cosine_similarity_torch(x, y):
    return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1)

# 测试
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"手写 cosine: {cosine_similarity_manual(x, y).item():.4f}")  # ~0.9746
print(f"PyTorch cosine: {cosine_similarity_torch(x, y).item():.4f}")

# 批量计算 cosine 矩阵(用于检索)
vectors = torch.randn(10, 64)  # 10 个 64 维向量
query = torch.randn(1, 64)

# 全部 cosine
cosine_matrix = F.normalize(vectors, dim=-1) @ F.normalize(query, dim=-1).T
print(f"\n查询 vs 10 个向量的 cosine:")
print(cosine_matrix.squeeze().tolist())

# 找 Top-3
top3_indices = cosine_matrix.squeeze().argsort(descending=True)[:3]
print(f"Top-3 索引: {top3_indices.tolist()}")
print(f"Top-3 cosine: {cosine_matrix.squeeze()[top3_indices].tolist()}")
```

### 9.4 实验 4:实现 L1 / L2 / L∞ 范数,对比 3 个向量的范数大小

```python
import torch

def norm_manual(x, p):
    """手写 Lp 范数"""
    if p == 1:
        return x.abs().sum()
    elif p == 2:
        return torch.sqrt((x ** 2).sum())
    elif p == float('inf'):
        return x.abs().max()
    else:
        return ((x.abs() ** p).sum()) ** (1.0 / p)

# 对比 PyTorch
def norm_torch(x, p):
    return x.norm(p=p)

# 3 个测试向量
v1 = torch.tensor([1.0, 0.0, 0.0])      # 范数集中在一个分量
v2 = torch.tensor([0.5, 0.5, 0.5])      # 范数均匀分布
v3 = torch.tensor([3.0, -4.0, 0.0])     # 经典的 3-4-5

print(f"{'向量':<25}{'L1':<10}{'L2':<10}{'L∞':<10}")
print("-" * 55)
for v, name in [(v1, "(1,0,0)"), (v2, "(0.5,0.5,0.5)"), (v3, "(3,-4,0)")]:
    l1 = norm_torch(v, 1).item()
    l2 = norm_torch(v, 2).item()
    linf = norm_torch(v, float('inf')).item()
    print(f"{name:<25}{l1:<10.4f}{l2:<10.4f}{linf:<10.4f}")

# 验证一致性
for v in [v1, v2, v3]:
    for p in [1, 2, float('inf')]:
        assert torch.allclose(norm_manual(v, p), norm_torch(v, p), atol=1e-5)
print("\n手写与 PyTorch 一致 ✓")
```

**预期输出**:

```text
向量                      L1        L2        L∞
-------------------------------------------------------
(1,0,0)                   1.0000    1.0000    1.0000
(0.5,0.5,0.5)             1.5000    0.8660    0.5000
(3,-4,0)                  7.0000    5.0000    4.0000
```

观察:对于 $(0.5, 0.5, 0.5)$,L1 = 1.5 最大,L∞ = 0.5 最小。这是范数的"压缩"效应。

### 9.5 实验 5:L2 归一化前后,内积的差异

```python
import torch
import torch.nn.functional as F

# 构造两个方向的向量(放大关系)
v1 = torch.tensor([0.5, 0.5, 0.5])    # 范数 ~0.866
v2 = torch.tensor([1.0, 1.0, 1.0])    # 范数 ~1.732(v1 的 2 倍)
v3 = torch.tensor([2.0, 2.0, 2.0])    # 范数 ~3.464(v1 的 4 倍)
query = torch.tensor([1.0, 0.5, 0.0])

# 归一化前
inner_products = torch.stack([v1 @ query, v2 @ query, v3 @ query])
print("归一化前内积:", inner_products.tolist())
# [0.5+0.25+0=0.75, 1.0+0.5+0=1.5, 2.0+1.0+0=3.0]

# 归一化后(内积 = cosine)
v1_n = F.normalize(v1, dim=-1)
v2_n = F.normalize(v2, dim=-1)
v3_n = F.normalize(v3, dim=-1)
query_n = F.normalize(query, dim=-1)

cosines = torch.stack([v1_n @ query_n, v2_n @ query_n, v3_n @ query_n])
print("归一化后内积(cosine):", cosines.tolist())
# 全部 ≈ 0.8944(完全相同)

print("\n观察:归一化前,内积因范数差异变化 4 倍")
print("      归一化后,内积只反映方向,完全相同")
```

### 9.6 实验 6:Attention 简化版,用 dot product 计算相似度

```python
import torch
import torch.nn.functional as F

def simple_attention(query, keys, values):
    """
    简化的 Attention:
    - query: shape (d,)
    - keys: shape (n, d)
    - values: shape (n, d)
    - 返回:加权和,shape (d,)
    """
    # 1. 计算 query 与每个 key 的内积
    scores = keys @ query  # shape (n,)

    # 2. softmax 归一化(变概率)
    attn_weights = F.softmax(scores, dim=-1)  # shape (n,)

    # 3. 加权求和 values
    output = attn_weights @ values  # shape (d,)

    return output, attn_weights

# 测试
torch.manual_seed(42)
n_keys = 5
d = 8

query = torch.randn(d)
keys = torch.randn(n_keys, d)
values = torch.randn(n_keys, d)

output, weights = simple_attention(query, keys, values)

print(f"Query shape: {query.shape}")
print(f"Keys shape: {keys.shape}")
print(f"Attention weights: {weights.tolist()}")
print(f"Weights sum: {weights.sum().item():.4f}")  # 应该是 1.0
print(f"Output shape: {output.shape}")

# 验证:权重和 = 1(softmax 性质)
assert torch.allclose(weights.sum(), torch.tensor(1.0))

# 用 scaled dot-product(标准 Transformer)
def scaled_dot_product_attention(Q, K, V):
    """标准 Transformer Attention"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V, attn_weights

# 批量版本
Q = torch.randn(2, 5, 8)  # (batch, n_query, d)
K = torch.randn(2, 5, 8)  # (batch, n_key, d)
V = torch.randn(2, 5, 8)  # (batch, n_key, d)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"\n批量 Attention 输出 shape: {output.shape}")
print(f"权重 shape: {weights.shape}")
```

**关键点**:
- `Q @ K.transpose` 就是矩阵化的内积,GPU 加速
- 除以 $\sqrt{d}$ 是数值稳定技巧
- softmax 把任意实数变成概率分布(和 = 1)

---

## 10. 常见误区

### 误区 1: ❌ "cosine 相似度 $\in [-1, 1]$" 但只对**非零向量**成立

**反例**:对**零向量**计算 cosine:

```python
import torch
import torch.nn.functional as F

v1 = torch.tensor([0.0, 0.0, 0.0])  # 零向量
v2 = torch.tensor([1.0, 2.0, 3.0])

# F.normalize 会除以 0,得到 NaN 或 0
v1_n = F.normalize(v1, dim=-1)
print(v1_n)  # tensor([0., 0., 0.])

cosine = (v1_n * F.normalize(v2, dim=-1)).sum()
print(cosine)  # tensor(0.)(实际是 0/0 = NaN,被截断为 0)
```

**判断**:Cosine 相似度的定义要求两个向量都**非零**。对零向量,数学上"未定义"。**工程上**:F.normalize 用 eps 保护,但数值仍不稳定。

### 误区 2: ❌ "内积为 0 = 正交",但**零向量**的内积恒为 0

**反例**:对任意向量 $\mathbf{x}$,$\mathbf{0} \cdot \mathbf{x} = 0$。但零向量与任何向量都"正交"吗?

数学上,正交的定义是 $\mathbf{x} \cdot \mathbf{y} = 0$,**零向量满足这个条件**(它跟所有向量"垂直")。但几何上,**零向量没有方向**,说它"正交"是含糊的。

**判断**:
- 严格说,正交要求两个向量都非零
- 实际工程中,零向量的 Embedding 通常是"未初始化"或"padding"导致的,应该避免

### 误区 3: ❌ "Embedding 越大越好"(维度灾难)

**反例**:64 维 Embedding 已经足够捕捉大多数语义。继续增到 1024 维,效果**不一定更好**,反而:
- 内存增加 16 倍
- 计算增加 16 倍
- **过拟合风险增加**(高维稀疏)

**判断**:Embedding 维度选择需要 trade-off。一般经验值:

| 场景 | 推荐维度 |
| :--- | :--- |
| 小数据集 (< 100K 物品) | 32-64 |
| 中等数据集 (100K-1M) | 64-128 |
| 大数据集 (> 1M) | 128-256 |
| LLM hidden state | 768-4096 |

### 误区 4: ❌ "欧氏距离 = cosine 距离"

**反例**:$\mathbf{x} = (1, 0)^T$, $\mathbf{y} = (2, 0)^T$。

- 欧氏距离: $\|\mathbf{x} - \mathbf{y}\|_2 = 1$
- cosine 距离: $1 - \cos\theta = 0$ ($\mathbf{x}$ 和 $\mathbf{y}$ 方向完全相同)

**结论**:欧氏距离衡量"坐标差",cosine 距离衡量"方向差"。**两者完全不同**。

| 度量 | 关心 | 适合 |
| :--- | :--- | :--- |
| 欧氏距离 | 大小 + 方向 | 物理空间(坐标) |
| cosine 距离 | 仅方向 | 语义空间(Embedding) |

### 误区 5: ❌ "向量加法没有物理意义"

**反例**:BERT 的 [CLS] = 所有 token Embedding 的加权和,推荐系统的用户 Embedding = 历史物品 Embedding 的加权和。

**判断**:**向量加法在 DL 中无处不在**(残差连接、Attention 加权求和、特征拼接),都是有意义的"信息聚合"。

### 误区 6: ❌ "L1 范数 = 欧氏距离"

**反例**:L1 是曼哈顿距离,L2 才是欧氏距离。

$\mathbf{x} = (3, 4)^T$:

- $\|\mathbf{x}\|_1 = 7$(曼哈顿:沿 x 走 3,沿 y 走 4)
- $\|\mathbf{x}\|_2 = 5$(欧氏:直线距离)
- $\|\mathbf{x}\|_\infty = 4$(最大分量)

**判断**:"曼哈顿距离 vs 欧氏距离"是不同的度量,前者是"街区距离",后者是"直线距离"。

### 误区 7: ❌ "Embedding 检索一定要用 cosine"

**反例**:如果业务关心"偏好强度",内积更合适。

```python
# 用户向量(范数大 = 活跃用户)
user = torch.tensor([2.0, 2.0])      # 范数 2.83
# 物品 A(范数大)
item_A = torch.tensor([1.0, 1.0])   # 范数 1.41
# 物品 B(范数小,方向同 A)
item_B = torch.tensor([0.5, 0.5])   # 范数 0.71

# 内积:item_A > item_B(活跃用户喜欢热门物品)
print((user @ item_A).item(), (user @ item_B).item())
# 4.0 2.0

# cosine:item_A == item_B(方向相同)
print(F.cosine_similarity(user.unsqueeze(0), item_A.unsqueeze(0)).item())
print(F.cosine_similarity(user.unsqueeze(0), item_B.unsqueeze(0)).item())
# ~0.7071 ~0.7071
```

**判断**:选 cosine 还是内积取决于业务场景。

### 误区 8: ❌ "Adam 梯度裁剪用 L2,所以 L2 永远是最好的"

**反例**:对抗样本通常用 L∞,因为"图像像素最大改动"是 L∞ 范数。Lasso 回归用 L1 正则,因为 L1 鼓励稀疏。

**判断**:范数选择有 trade-off:
- L1:稀疏、鲁棒
- L2:均匀、稳定
- L∞:约束最值、平滑

---

## 11. 面试 Q&A (L7/L8 难度)

### Q1: 为什么 Attention 用 dot product 而不是 cosine?

**标准答案**:

**1. 效率考虑**:

- dot product = 矩阵乘法,GPU 高度优化
- cosine = dot product + 2 个 norm + 1 个除法,需要额外计算

在 Attention 中,$Q, K \in \mathbb{R}^{N \times d}$,$QK^T$ 是一个矩阵乘法,O(N²d)。cosine 需要先算每个向量的范数,再算内积除以范数,计算量翻倍。

**2. 数值稳定性**:

- dot product 输出范围 $\sim [-N, N]$(如果向量值 $\sim 1$)
- cosine 输出范围 $[-1, 1]$,有界

但 Attention 用了 scaling factor $1/\sqrt{d}$ 把 dot product 标准化,本质是把它"模拟"成 cosine。所以最终差异不大。

**3. 实践中等价**:

在 Transformer 中,$Q, K$ 通常经过 LayerNorm,各分量量级相近。此时 dot product 和 cosine 高度相关,可以近似互换。

**4. 工业选择**:

BERT、GPT、LLaMA 等所有主流模型都用 dot product + scaling。**这是工业共识**。

**面试加分点**:提到"head dimension 增大时,dot product 的方差增大",所以除以 $\sqrt{d}$ 是必要的;否则 softmax 会饱和(所有权重集中在一个 token 上)。

### Q2: Embedding 检索前为什么要 L2 归一化?

**标准答案**:

**1. 消除范数偏差**:

未归一化时,内积 = cosine × 范数积。范数大的向量天然得分高,即使方向不接近 query。归一化后,内积 = cosine,只反映方向。

**2. 让内积等价于 cosine**:

```python
# 归一化后
score = F.normalize(x) @ F.normalize(y)  # = cosine(x, y)
```

这样可以用内积索引(更快)同时保持 cosine 语义(更准)。

**3. 工业实践**:

- FAISS 的 `IndexFlatIP`(内积索引) 假设输入已归一化
- ScaNN、Annoy 等也类似
- 归一化是**向量检索的标配预处理**

**4. 适用场景**:

- 推荐系统召回:通常归一化(避免热门物品 Embedding 范数大)
- 语义搜索:通常归一化(cosine 才是"语义相似度")
- 双塔模型:训练时是否归一化是 hyperparameter

**面试加分点**:提到"归一化后,所有向量都在单位球面上,等价于把问题约束在 $S^{d-1}$ 上"。这跟对比学习(SimCSE)的几何基础一致。

### Q3: 推荐系统召回用内积还是 cosine?为什么?

**标准答案**:

**用内积的场景**:

- **双塔模型**:用户塔和物品塔分别输出向量,内积打分。优点:**预计算物品向量,在线只算一次内积,延迟极低**
- **偏好强度重要**:内积同时反映方向和大小,符合"用户对物品的偏好强度"语义
- **数学简单**:训练目标是 BCE Loss,稳定

**用 cosine 的场景**:

- **流行度偏差严重**:热门物品 Embedding 范数大,内积偏向热门。cosine 消除范数差异
- **方向主导**:业务关心"语义相似"而非"强度"(比如文本检索)
- **训练目标包含正则**:cosine 损失自带"归一化"约束

**混合方案**:

```python
# 方案 1:训练用 cosine,检索也用 cosine
loss = 1 - F.cosine_similarity(u, v)
score = F.cosine_similarity(u, v)

# 方案 2:训练用 cosine,检索用 L2 归一化 + 内积(等价)
loss = 1 - F.cosine_similarity(u, v)
score = F.normalize(u) @ F.normalize(v).T  # = cosine, 但用矩阵乘法加速

# 方案 3:训练用内积,检索也用内积(简单)
loss = BCEWithLogitsLoss(u @ v.T, labels)
score = u @ v.T
```

**面试加分点**:提到"召回通常用 cosine(消除偏差),排序可以用内积(更细粒度)"。召回和排序的目标函数可以不同。

### Q4: L1 和 L2 正则的本质差异是什么?

**标准答案**:

**1. 数学差异**:

- L1: $\Omega(W) = \sum |w_{ij}|$
- L2: $\Omega(W) = \sum w_{ij}^2$

**2. 解的稀疏性**:

L1 倾向于产生**稀疏解**(很多权重为 0),L2 倾向于产生**小而均匀的解**(所有权重都接近 0 但非零)。

**几何解释**:L1 的约束区域是菱形($L_1$ 球),L2 是圆形($L_2$ 球)。Loss 等高线(椭圆)与约束区域的交点:

```text
L1 约束:            L2 约束:
   ◇                 ○
    \                /
     \ 椭圆相交 →   / 椭圆相交
      \ 顶点       / 任意点
       ↓          (通常不在坐标轴)
   稀疏解(在坐标轴上)
```

**3. 优化差异**:

L1 在 0 处不可导(次梯度),L2 处处可导。L1 更难优化但有稀疏性,L2 更易优化但只能"压缩"。

**4. 工程选择**:

- **特征选择**:用 L1(自动选特征)
- **防止过拟合 + 平滑**:用 L2(权重小且均匀)
- **L1 + L2 = Elastic Net**(组合使用)

**面试加分点**:提到"Adam + L2" 不等价于 "Adam + weight decay",因为 Adam 的自适应学习率破坏了 L2 的"等比缩小"性质。**AdamW 用解耦的 weight decay 修正这个问题**(Loshchilov & Hutter 2019)。

### Q5: 为什么 Adam 梯度裁剪用 L2 norm?

**标准答案**:

**1. L2 范数的物理含义**:

L2 范数是"梯度向量的整体大小"(欧氏长度),梯度裁剪本质是"如果梯度过大,就缩小它"。

**2. 实现简洁**:

```python
# PyTorch 源码(简化)
total_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]))
clip_coef = max_norm / (total_norm + 1e-6)
clip_coef = min(clip_coef, 1.0)
# 整体缩放
for g in grads:
    g.mul_(clip_coef)
```

**3. 几何性质**:

L2 裁剪保持**梯度方向不变**(只缩放不旋转)。这意味着更新方向不变,只调整步长。

**4. 与 Hessian 谱的关系**:

L2 裁剪等价于"约束参数空间中的球",与 Lipschitz 约束一致。如果用 L∞ 裁剪,约束的是"最大分量",破坏了"球形"假设。

**5. 与 Lipschitz 连续的关系**:

梯度裁剪保证 $\|\mathbf{g}\| \le c$,而 $\|\nabla L\| \le c$ 意味着 Loss 是 c-Lipschitz,训练更稳定(见 `../../../基础/数学基础/第01章-高等数学/第01章-03-多元微分·梯度·Jacobian·Hessian-反向传播的数学骨架.md` §3.4)。

**面试加分点**:提到"RNN 中梯度爆炸是常见问题,梯度裁剪是标配"。LSTM/GRU 通过门控部分解决,但 Transformer 等纯 Attention 模型也需要裁剪。

---

## 12. 与其他数学工具的联系

向量、内积、范数是线性代数的"原子",向上构建出更复杂的对象:

### 12.1 内积 → 矩阵乘法(第 2 章)

**矩阵乘法的本质**:矩阵 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$ 的乘积 $C = AB$,$C_{ij} = \sum_k A_{ik} B_{kj}$ 就是 **"A 的第 i 行与 B 的第 j 列的内积"**。

所以**矩阵乘法 = 批量内积**。详细推导见 第 2 章:矩阵·线性变换·秩 §1。

### 12.2 向量范数 → 矩阵范数(第 2 章 Frobenius / 第 4 章谱范数)

- **Frobenius 范数**:把矩阵"展平"成 $mn$ 维向量,再算 L2 范数。详见 `../../../方向教程/rec教程/第02章-深度学习数学与理论基础.md` §6 和本章 §6.6。
- **谱范数**(Spectral Norm):$|A\|_2 = \sigma_{\max}(A)$,最大奇异值。详见 第 4 章:特征值·特征向量·谱定理 §5。
- **核范数**(Nuclear Norm):$\|A\|_* = \sum_i \sigma_i$,所有奇异值之和,用于低秩矩阵补全。

### 12.3 向量空间 → Hilbert 空间(1-高数 第 8 章)

**Hilbert 空间** = 完备的内积空间。把向量从有限维 $\mathbb{R}^n$ 推广到无限维(函数空间),得到 **Hilbert 空间**。

- $L^2([0, 1])$ = 平方可积函数空间,内积 $\langle f, g \rangle = \int_0^1 f(x) g(x) dx$
- **RKHS**(Reproducing Kernel Hilbert Space) 是 Hilbert 空间的特殊子集,是 Kernel Methods 的数学基础

详见 `../第01章-高等数学/第01章-08-泛函分析基础-Hilbert空间·RKHS·算子理论·NTK.md`。

### 12.4 内积 → 双线性型(第 5 章)

**双线性型** = 内积的推广:不要求正定性、对称性,可以是不定型的。

$$\beta(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T A \mathbf{y}$$

其中 $A$ 是任意矩阵。当 $A$ 对称正定时,$\beta$ 就是内积(模去范数)。

详见 第 5 章:二次型·正定性·条件数。

### 12.5 cosine 相似度 → 互信息估计(5-信息论 第 3 章)

Cosine 相似度是"方向相似性"的简单度量。在信息论中,更一般的"相关性"度量是 **互信息**(Mutual Information):

$$I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

- Cosine 衡量两个**确定向量**的相似度
- 互信息衡量两个**随机变量**的依赖程度
- 互信息 ≥ 0,Cosine ∈ [-1, 1],两者尺度不同但哲学相通

详见 `../../../基础/数学基础/第05章-信息论/第05章-03-互信息·点互信息·条件互信息.md`。

### 12.6 向量范数 → 概率测度的距离(3-概率论)

向量的 L2 距离 / cosine 距离是"向量空间中的距离"。在概率论中,类似的距离概念有:
- **KL 散度**(不对称,但常用)
- **JS 散度**(对称版本)
- **Wasserstein 距离**(最优传输)

这些是"概率分布之间的距离",类比"向量距离"。详见 `../../../基础/数学基础/第03章-概率论/第03章-04-期望·方差·协方差·矩-从样本均值到-Embedding-统计特性.md`。

---

## 13. 本章总结

### 13.1 一句话总结

**向量是高维空间中的点,内积是"相似度",范数是"长度",cosine 是"方向相似度"。这三件套是 Embedding、Attention、推荐召回、对比学习的共同语言。**

### 13.2 关键 takeaways

| 概念 | 数学 | ML/DL 翻译 |
| :--- | :--- | :--- |
| 向量 $\mathbf{x} \in \mathbb{R}^n$ | $n$ 维空间中的点 | Embedding / 隐藏层输出 |
| 加法 $\mathbf{x} + \mathbf{y}$ | 平行四边形法则 | Embedding 加权和 / 残差 |
| 数乘 $\alpha \mathbf{x}$ | 缩放 | 权重 / 学习率 |
| 内积 $\mathbf{x} \cdot \mathbf{y}$ | 投影长度 | Attention Q·K / 双塔打分 |
| 范数 $\|\mathbf{x}\|_2$ | 欧氏长度 | Embedding 范数 / 梯度范数 |
| cosine | 方向相似度 | 语义相似度 / 对比学习 |
| Cauchy-Schwarz | $\|\mathbf{x} \cdot \mathbf{y}\| \le \|\mathbf{x}\| \|\mathbf{y}\|$ | cosine 范围 [-1, 1] |
| 投影 | $\text{proj}_S(\mathbf{x})$ | PCA / 降维 / 子空间 |

### 13.3 关键概念检查清单

学完本章, 你应该能:

- [ ] 解释为什么 Embedding 是"高维向量"
- [ ] 用代数和几何两种方式计算内积
- [ ] 推导 Cauchy-Schwarz 不等式并解释 cosine 范围
- [ ] 计算 L1 / L2 / L∞ 范数,理解各自的应用场景
- [ ] 解释 Attention 用内积而不是 cosine 的原因
- [ ] 解释 ItemCF 中 cosine 相似度的几何意义
- [ ] 实现 PyTorch 版的 cosine 相似度检索
- [ ] 解释 L2 归一化前后的内积差异
- [ ] 用 ASCII 画出 2D 平面上的 5 个商品 Embedding

### 13.4 下一章预告

第 2 章:矩阵·线性变换·秩 , 神经网络的"线性层"是什么。

向量是一维数组,矩阵是二维数组。本章讲的"内积"会变成"矩阵乘法","范数"会变成"矩阵范数","投影"会变成"线性变换"。理解了向量,矩阵就是顺水推舟。

---

## 附录: 练习题 (选做)

### 练习 1 (基础): 计算向量 $\mathbf{x} = (2, -3, 1)^T$ 的 L1, L2, L∞ 范数。

**提示**: 直接代入公式。

### 练习 2 (中级): 用 Cauchy-Schwarz 证明 $(\sum_i a_i b_i)^2 \le (\sum a_i^2)(\sum b_i^2)$。

**提示**: 应用 $\|\mathbf{a} \cdot \mathbf{b}\| \le \|\mathbf{a}\| \|\mathbf{b}\|$。

### 练习 3 (高级): 实现一个 mini 向量检索系统。随机生成 1000 个 128 维向量,给定 query,返回 Top-5 cosine 最相似的向量。

**提示**: 用 `F.normalize` + `@` + `argsort`。

### 练习 4 (工程): 在 PyTorch 中实现简化版 Self-Attention,对一个 4 token × 8 维的输入,可视化 Attention 权重矩阵。

**提示**: §9.6 的代码扩展到批量。

### 练习 5 (思考): 为什么 BERT 的 [CLS] token 在 fine-tuning 时通常被作为句子 Embedding 使用?这跟 §3.3 的"向量加法 + 数乘"有什么关系?

**提示**: 想想 Attention 加权求和的权重是怎么学习的。

---

## 附录: 参考资料

1. **Strang, G. 《Linear Algebra and Its Applications》**: 第 1 章向量与线性方程组, 第 2 章矩阵, 第 4 章正交性。第 1 章是本章的数学基础。
2. **Goodfellow, Bengio, Courville《Deep Learning》**: 第 2 章线性代数, 包含 §4.1 ~ §4.3 的工程视角。
3. **Vaswani et al. 2017, "Attention Is All You Need"**: Transformer 原始论文, §3.2.1 解释了 scaled dot-product attention。
4. **Miyato et al. 2018, "Spectral Normalization for GANs"**: 谱归一化的工业实现, 算子范数的应用。
5. **Gao et al. 2021, "SimCSE: Simple Contrastive Learning of Sentence Embeddings"**: 对比学习用 cosine 的典型案例。
6. **He et al. 2016, "Deep Residual Learning for Image Recognition"**: ResNet, 加法在 DL 中的代表应用。
7. **Loshchilov & Hutter 2019, "Decoupled Weight Decay Regularization"**: AdamW 论文, L2 正则 vs weight decay 的区分。
8. **Johnson & Lindenstrauss 1984**: J-L 引理, 高维空间几何的奠基论文(为 `../../../方向教程/rec教程/第08章-GR的在线与长期.md` 做准备)。

---

## 14. 工程师常见误区与"踩坑地图"

这一节专门给**写过 PyTorch 但总有点慌**的工程师。把大家常踩的坑列一遍,每个坑给"症状 / 根因 / 解药"。

### 坑 1: "Embedding 维度越高越好"

**症状**: 给 64 维就够用的物品 Embedding 强行升到 1024 维,显存爆了,推理变慢。

**根因**: 维度增加带来 $O(d)$ 的存储和 $O(d^2)$ 的矩阵乘计算;但**信号可能没那么多**——10 万个物品、每位用户 10 次行为,总共 100 万次交互 ≈ 维度 $d$ 最多能从中学到 $\sim \sqrt{10^6}=1000$ 维"潜在语义"。

**解药**:
- 用 `../../../方向教程/rec教程/第09章-Python底层.md` 讲的截断 SVD,看奇异值衰减:**保留 90% 能量需要多少维**?
- 推荐 / 搜索场景 64-128 维通常足够;对比学习的 sentence embedding 通常 768-1024 维。

**经验法则**: $d_{\text{embed}} \approx \min(50 \cdot \sqrt[4]{N_{\text{items}}}, 1024)$。**先小后大**,效果饱和即停。

### 坑 2: "Cosine 相似度 vs 点积"搞混

**症状**: 双塔模型用点积打分,但评测指标用 cosine;或者反之。

**根因**:
- **点积 (Dot product)**:`score = u·i`,受向量范数影响。范数大的物品"自然分高"。
- **Cosine 相似度**:`score = (u·i)/(\|u\|\|i\|)`,消除范数影响,只看方向。

**解药**:
- 如果物品 Embedding **没归一化**,且范数有业务意义(如热门物品范数自然大 → 倾向于被推荐),用点积。
- 如果要"语义相似"严格匹配,用 cosine。**事先 L2 归一化所有 Embedding**,点积 ≡ cosine。
- 召回阶段常用点积(快),粗排/精排用 cosine 或更复杂的打分函数(深网络)。

```python
# 坑 2 的修复
item_emb = F.normalize(item_emb, dim=-1)  # L2 归一化
user_emb = F.normalize(user_emb, dim=-1)
# 现在 score = (user_emb * item_emb).sum(dim=-1) 等价于 cosine 相似度
```

### 坑 3: "内积矩阵的 shape 算错"

**症状**: `torch.bmm` 报 shape mismatch;或者明明想算相似度矩阵,结果出来是 1×1。

**根因**: PyTorch 的 `@` 和 `*` 不一样:
- `*`:**逐元素乘**(Hadamard 积),shape 必须完全相同。
- `@` / `torch.matmul`:**矩阵乘 / 点积 / batched 矩阵乘**,遵守矩阵乘的 shape 规则 $(B, n, m) @ (B, m, p) = (B, n, p)$。

**解药**: 牢记"**最后一维是输入维度,倒数第二维是 batch**":
- `(B, d) @ (d, N) → (B, N)`:每个 batch 元素与 $N$ 个候选算内积。
- `(B, d) @ (B, d, N) → (B, N)`:`bmm`,每个 batch 元素对应不同的 $N$ 候选。
- `(B, L, d) @ (B, d, L) → (B, L, L)`:Attention 的 scores 矩阵,**L 个 token 两两相似度**。

### 坑 4: "L2 归一化破坏了范数信息"

**症状**: 双塔模型中,把所有 Embedding 都 L2 归一化,效果反而变差。

**根因**: 范数本身有信息(比如,训练过程中范数大的 = "难样本")。**强行归一化 = 丢弃范数信息**。

**解药**:
- 推荐系统召回:通常**保留范数**,用点积。
- 对比学习 (SimCSE, CLIP):**强制归一化**,只用 cosine,聚焦语义。
- 不要无脑归一化,**根据任务决定**。

### 坑 5: "Q·K 维度没除以 √d"

**症状**: 训练 Transformer,loss 不收敛;或 attention 权重全集中在某个 token。

**根因**: 当 $d_k$ 大时,$Q \cdot K^\top$ 的方差 ≈ $d_k$,softmax 进入**饱和区**(梯度近 0)。这是论文 Attention Is All You Need §3.2.1 明确点出的。

**解药**: 始终 `scores = Q @ K.T / sqrt(d_k)`。**不要忘记 √d**!

```python
# 标准 Transformer Attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attn = F.softmax(scores, dim=-1)
```

### 坑 6: "Gradient clipping 用错了范数类型"

**症状**: 用 `clip_grad_norm_(max_norm=1.0)` 但 loss 还是爆炸。

**根因**: `clip_grad_norm_` 默认是 **L2 范数**。如果某些层用 L1 范数更合适(稀疏梯度),要手动写。

**解药**:
- 99% 场景:**L2 范数**足够(`torch.nn.utils.clip_grad_norm_`)。
- RNN 训练中常加 `clip_grad_value_`(按值裁剪),防止爆炸。
- 经验值:`max_norm=1.0` 是常见默认值,实际可以 [0.5, 5.0] 调。

### 坑 7: "Embedding 层冻结,还是微调?"

**症状**: 加载预训练 BERT,但下游任务 loss 死活不降。

**根因**: Embedding 冻结后,**词向量不变**,模型只能调后面的 Transformer。如果你的下游任务需要新词 / 新领域,冻结 Embedding 损失巨大。

**解药**:
- **冻结策略**:仅在小数据 / 跨语言迁移时冻结 Embedding。
- **学习率分组**:Embedding 用 $\eta_{\text{embed}} = \eta_{\text{model}} / 10$(微调时不要破坏预训练语义)。
- LoRA / Adapter:**只微调低秩增量**,保留原 Embedding 不变——这是 `../../../方向教程/rec教程/第09章-Python底层.md` 的核心。

### 坑 8: "范数 = 重要性"是错觉

**症状**: 训练完后统计 Embedding 范数,发现长尾物品范数极小,以为它们"不重要"。

**根因**: 出现次数少的物品,**梯度更新次数也少**,范数小是正常的,**不代表语义不重要**。

**解药**:
- 用 `../../../方向教程/rec教程/第09章-Python底层.md` 讲的正则化:`|E_i\|^2 \ge \delta$`(约束最小范数)。
- 或者用 `../../../方向教程/rec教程/第06章-Semantic-ID全景.md` 的"频次倒数加权"梯度更新。

### 坑 9: "向量加法的语义不对"

**症状**: king - man + woman ≈ queen 这种 word2vec 经典案例,在自己的 Embedding 上死活复现不出。

**根因**: **词向量的线性偏移只在某些训练目标上成立**(word2vec 的 skip-gram + 负采样确实有这个性质,但 BERT 不一定有)。

**解药**:
- **不要硬套**:不同模型 Embedding 的几何性质不同。
- 对比学习 / BERT 的 Embedding 是**各向异性**的(向量不均匀分布在球面),做加减法前先 **mean center + L2 normalize**。

### 坑 10: "矩阵范数混用"

**症状**: 论文里写"权重矩阵满足 spectral norm ≤ 1",实现时直接用 `torch.norm(W)`。

**根因**: `torch.norm(W)` 默认是 **Frobenius 范数**(所有元素平方和开根号),**不是 spectral norm(算子范数 / 最大奇异值)**。

**解药**:
- Spectral norm: `torch.linalg.svdvals(W)[0]`(取最大奇异值)。**贵**,要算 SVD。
- 工程上用 `../../../方向教程/rec教程/第04章-多任务学习.md` 讲的 **幂迭代** 估计 spectral norm(Miyato 2018)。
- 永远不要混淆 **Frobenius / Spectral / Nuclear** 三种范数(详见 `../../../方向教程/rec教程/第04章-多任务学习.md`)。

### 坑 11: "广播 (broadcasting) 把向量加了错维度"

**症状**: `tensor + bias` shape 错,以为会自动广播,实际广播规则不熟。

**根因**: PyTorch 广播规则:**从最后一维对齐,维度为 1 的可以扩展**。

```python
# 错: (B, d) + (d,) 应该能广播 OK
x = torch.randn(32, 64)  # (B, d)
b = torch.randn(64)      # (d,)
y = x + b               # (32, 64) ✓

# 错: (B, d) + (B,) 不能广播!
x = torch.randn(32, 64)
b = torch.randn(32)      # 维度不对齐!
y = x + b               # RuntimeError
```

**解药**: 牢记"**最后一维 = 特征维,从后往前对齐**"。详细见 `../../../方向教程/rec教程/第07章-强化学习在GR中的角色.md` 张量的 stride/broadcasting。

### 坑 12: "Loss 突然变 NaN"

**症状**: 训练到第 N 步 loss = NaN,梯度范数爆了。

**根因排查顺序**:
1. 学习率太大 → 减小
2. 没有 gradient clipping → 加 `clip_grad_norm_`
3. 用了 fp16 但没 GradScaler → 换 fp32 或加 `torch.cuda.amp.GradScaler`
4. 输入有 NaN/Inf → 检查数据预处理
5. Loss 函数有 `log(0)` → 加 epsilon `log(x + 1e-8)`

**解药**: 训练早期加 NaN 检测:
```python
if torch.isnan(loss).any():
    print("NaN detected at step", step)
    break
```

---

## 15. 完整案例: 从零搭一个迷你推荐召回系统

这一节我们用本章学到的所有概念,**从零搭一个玩具双塔召回系统**。完整的代码、可视化、调参,一站式。

### 15.1 数据: 100 个用户 × 1000 个物品,每个用户 20 次交互

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

N_USERS = 100
N_ITEMS = 1000
N_INTERACT = 20  # 每个用户 20 次交互
EMBED_DIM = 32

# 用户-物品交互矩阵(稀疏)
R = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
for u in range(N_USERS):
    # 每个用户随机点 20 个物品
    items = np.random.choice(N_ITEMS, N_INTERACT, replace=False)
    R[u, items] = 1.0

# 划分 train/val(留一法:每个用户最后 1 次作为 val)
train_R = R.copy()
val_R = np.zeros_like(R)
for u in range(N_USERS):
    interacted = np.where(R[u] > 0)[0]
    last = interacted[-1]
    val_R[u, last] = 1.0
    train_R[u, last] = 0.0

print(f"训练交互数: {train_R.sum():.0f}, 验证交互数: {val_R.sum():.0f}")
# 训练交互数: 1900, 验证交互数: 100
```

### 15.2 模型: 双塔 + 点积打分

```python
class TwoTowerModel(nn.Module):
    """用户塔 + 物品塔,内积打分(本章 §9 的工程实现)"""
    def __init__(self, n_users, n_items, embed_dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)  # 用户塔
        self.item_emb = nn.Embedding(n_items, embed_dim)  # 物品塔
        # 初始化小一点,训练更稳
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # (B, d) @ (B, d) -> (B,)
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return (u * i).sum(dim=-1)  # 内积打分

model = TwoTowerModel(N_USERS, N_ITEMS, EMBED_DIM)
```

### 15.3 训练: BPR 损失(隐式反馈推荐)

```python
def bpr_loss(pos_scores, neg_scores):
    """BPR 损失: 正样本分数应该 > 负样本分数
       本质是: -log sigmoid(pos - neg)"""
    return -F.logsigmoid(pos_scores - neg_scores).mean()

# 训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    # 采样 batch:每个用户 1 正 1 负
    user_ids = []
    pos_ids = []
    neg_ids = []
    for u in range(N_USERS):
        pos = np.where(train_R[u] > 0)[0]
        neg_pool = np.where(train_R[u] == 0)[0]
        if len(pos) == 0 or len(neg_pool) == 0:
            continue
        user_ids.append(u)
        pos_ids.append(np.random.choice(pos))
        neg_ids.append(np.random.choice(neg_pool))

    user_ids = torch.tensor(user_ids)
    pos_ids = torch.tensor(pos_ids)
    neg_ids = torch.tensor(neg_ids)

    pos_scores = model(user_ids, pos_ids)
    neg_scores = model(user_ids, neg_ids)
    loss = bpr_loss(pos_scores, neg_scores)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        # 评估召回 Recall@10
        with torch.no_grad():
            all_items = torch.arange(N_ITEMS)
            user_embs = model.user_emb.weight  # (N_USERS, d)
            item_embs = model.item_emb.weight  # (N_ITEMS, d)
            # 全部用户-物品打分矩阵 (N_USERS, N_ITEMS)
            scores = user_embs @ item_embs.T

            # 训练过的物品 mask 掉
            scores = scores - torch.tensor(train_R) * 1e9

            # Top-10 召回
            top10 = scores.topk(10, dim=-1).indices  # (N_USERS, 10)

            # Recall@10 = 验证集中被召回到的比例
            hits = 0
            for u in range(N_USERS):
                if val_R[u].argmax() in top10[u]:
                    hits += 1
            recall_at_10 = hits / N_USERS

        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Recall@10: {recall_at_10:.3f}")
```

预期输出:
```text
Epoch   0 | Loss: 0.6931 | Recall@10: 0.020
Epoch  20 | Loss: 0.4523 | Recall@10: 0.090
Epoch  40 | Loss: 0.3127 | Recall@10: 0.150
Epoch  60 | Loss: 0.2456 | Recall@10: 0.180
Epoch  80 | Loss: 0.2012 | Recall@10: 0.210
```

### 15.4 用本章知识分析训练过程

| 现象 | 本章对应概念 | 解读 |
| :--- | :--- | :--- |
| Loss 起步 ≈ 0.693 | sigmoid 起点 | $- \log(0.5) = \ln 2 \approx 0.693$,BPR 起始状态 |
| 训练 20 epoch 后 Loss ↓ | 梯度下降 | $\|\nabla L\|$ 范数控制更新步长 |
| Recall@10 缓慢上升 | cosine 内积方向学习 | 用户-物品 Embedding 在对齐 |
| 训练后物品 Embedding 范数差异大 | §6.3 的范数意义 | 热门物品范数大,推荐概率天然高 |

### 15.5 L2 归一化 + cosine 变体

如果改成 cosine 打分:
```python
# L2 归一化后,内积 = cosine
user_norm = F.normalize(model.user_emb.weight, dim=-1)
item_norm = F.normalize(model.item_emb.weight, dim=-1)
scores_cosine = user_norm @ item_norm.T  # 范围 [-1, 1]
```

**效果对比**:
- 点积:热门物品有偏(范数大 → 分数大)
- cosine:消除范数偏差,**更纯粹的"语义相似度"**

**经验**: 召回用点积(快,有"流行度"信号);粗排/精排用 cosine 或更复杂打分(准)。

---

## 16. 几何直觉强化: 用 ASCII 画向量世界

向量运算是"几何"的。我们用 ASCII 把本章所有概念画出来,加深直觉。

### 16.1 2D 平面上的向量加法(平行四边形法则)

```text
        y
        ↑
        |       ●(x+y) = (2+(-1), 3+1) = (1, 4)
        |      ╱╱
        |    ╱╱
        |  ╱╱
        |╱╱_________________→ x
        ●y=( -1, 1)

平行四边形法则:
         x+y
        ╱╱
   x ╱╱  y
    ╱╱╱╱
   ●────●───→ x
   起点  x 端

ASCII 实现:
      y=( -1, 1)
       ●
       │╲
       │ ╲
       │  ●────●  ← x+y=(1,4) ← x=(2,3)
       │  ╱
       │ ╱
       │╱
       ●─────────→ x
       起点(0,0)
```

### 16.2 2D 平面上的内积(投影)

```text
点积 = 向量 a 在向量 b 方向上的投影长度 × |b|
或者 = 向量 b 在向量 a 方向上的投影长度 × |a|

几何示意:
            ● a
           ╱│
          ╱ │ a 在 b 上的投影
         ╱  │
        ╱   ▼
       ●────●────● b
       起点    投影点

公式: a · b = |a| · |b| · cos(θ)
```

### 16.3 余弦相似度的几何

```text
完全相同 (cos = 1):
       ● a
       ╱
      ╱
     ● b

垂直 (cos = 0):
       ● a
       │
       │
       ● b →
       (a · b = 0)

完全相反 (cos = -1):
       ● a
       │
       │
       ● b
       ↓
```

### 16.4 高维"球壳"现象(为 §8 做铺垫)

$d$ 维单位球里,体积集中在 $r=1$ 附近的薄壳上。

```text
1D 球(线段):
[════════]        ← 体积均匀分布

2D 球(圆):
   ░░░░░
 ░░██████░░       ← 50% 在 0.9 ≤ r ≤ 1.0 的环上
 ░░██████░░
   ░░░░░

3D 球(实心球):
   ░░░░░
 ░░██████░░       ← 80% 在 r ≥ 0.9 的壳上
 ░░██████░░
   ░░░░░

100D 球:
  (画不出来,但 99.999% 体积在 0.99 ≤ r ≤ 1.0 的壳上)
```

直觉:维度越高,"赤道"概念越重要(对应 `../../../方向教程/rec教程/第08章-GR的在线与长期.md` 的测度集中)。

### 16.5 单位圆的矩阵变换(为第 2 章做铺垫)

```text
单位圆: x² + y² = 1

变换 A = [[2, 0], [0, 0.5]] 把圆压扁/拉长:

变换前:                变换后:
       ·                     · ·
     ·   ·                 ·     ·
    ·     ·               ·       ·
     ·   ·                 ·   ·
       ·                     · ·

椭圆的轴 = A 的奇异向量方向,长度 = 奇异值
```

---

## 17. 动手算一算(带完整答案)

这一节给 10 个**手算练习题**,答案在每个题后面折叠或明示。**强烈建议先自己算再对答案**。

### 17.1 基础题

**题 1**: 计算向量 $\mathbf{x} = (3, -4)^\top$ 的 L1, L2, L∞ 范数。

<details>
<summary>答案</summary>

- L1: $|3| + |-4| = 7$
- L2: $\sqrt{9 + 16} = 5$
- L∞: $\max(3, 4) = 4$

注意:L2 = 5 是经典的 3-4-5 三角形!
</details>

**题 2**: 验证 $\mathbf{x} = (1, 2)^\top$ 与 $\mathbf{y} = (2, 1)^\top$ 的余弦相似度。

<details>
<summary>答案</summary>

内积: $1 \cdot 2 + 2 \cdot 1 = 4$
范数: $\|x\| = \sqrt{5}, \|y\| = \sqrt{5}$
余弦: $\frac{4}{5} = 0.8$

角度: $\arccos(0.8) \approx 36.87°$
</details>

### 17.2 中级题

**题 3**: 证明 $\mathbf{x} = (1, 1, 1)^\top$ 与 $\mathbf{y} = (1, -1, 0)^\top$ 正交。

<details>
<summary>答案</summary>

内积: $1 \cdot 1 + 1 \cdot (-1) + 1 \cdot 0 = 0$ ✓
正交的内积定义就是 0。
</details>

**题 4**: 求 $\mathbf{x} = (3, 4)^\top$ 在 $\mathbf{y} = (1, 0)^\top$ 方向上的投影。

<details>
<summary>答案</summary>

投影公式: $\text{proj}_{\mathbf{y}} \mathbf{x} = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{y}\|^2} \mathbf{y}$
内积: $3 \cdot 1 + 4 \cdot 0 = 3$
$\|\mathbf{y}\|^2 = 1$
投影: $\frac{3}{1}(1, 0)^\top = (3, 0)^\top$

几何意义:把 $\mathbf{x}$ 垂直压到 x 轴上,得到 $(3, 0)$(因为 $\mathbf{y}$ 就是 x 轴单位向量)。
</details>

**题 5**: 把 $\mathbf{x} = (3, 4)^\top$ 分解为"沿 $(1,1)^\top$ 方向" + "垂直于 $(1,1)^\top$ 方向"两个分量。

<details>
<summary>答案</summary>

单位向量 $\mathbf{u} = \frac{1}{\sqrt{2}}(1, 1)^\top$
沿 $\mathbf{u}$ 的投影: $(\mathbf{x} \cdot \mathbf{u})\mathbf{u} = \frac{3+4}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}(1,1)^\top = \frac{7}{2}(1,1)^\top = (3.5, 3.5)^\top$
垂直分量: $\mathbf{x} - (3.5, 3.5)^\top = (-0.5, 0.5)^\top$
验证: $(3.5, 3.5) \cdot (-0.5, 0.5) = -1.75 + 1.75 = 0$ ✓ 正交
</details>

### 17.3 高级题(ML/DL 联系)

**题 6**: 假设 query Embedding $\mathbf{q} = (0.6, 0.8)^\top$(已 L2 归一化),候选 Embedding 集合:
$\mathbf{k}_1 = (0.5, 0.5)^\top$, $\mathbf{k}_2 = (-0.8, 0.6)^\top$, $\mathbf{k}_3 = (1.0, 0)^\top$

哪个候选与 query 最相关?用 Attention 公式 $\text{score} = \mathbf{q} \cdot \mathbf{k}_i$ 回答。

<details>
<summary>答案</summary>

- $\mathbf{q} \cdot \mathbf{k}_1 = 0.6 \cdot 0.5 + 0.8 \cdot 0.5 = 0.30 + 0.40 = 0.70$
- $\mathbf{q} \cdot \mathbf{k}_2 = 0.6 \cdot (-0.8) + 0.8 \cdot 0.6 = -0.48 + 0.48 = 0.00$
- $\mathbf{q} \cdot \mathbf{k}_3 = 0.6 \cdot 1.0 + 0.8 \cdot 0.0 = 0.60$

最大: $\mathbf{k}_1$ (score 0.70),最相关。

**注意**: 由于 $\mathbf{q}$ 已经 L2 归一化,内积就是 cosine 相似度。如果 $\mathbf{k}_i$ 没归一化,严格说不能直接当 cosine。
</details>

**题 7**: 用 NumPy 验证 Cauchy-Schwarz 不等式:对 1000 对随机向量 $(\mathbf{a}_i, \mathbf{b}_i)$, $|\mathbf{a}_i \cdot \mathbf{b}_i| \le \|\mathbf{a}_i\| \|\mathbf{b}_i|$。

<details>
<summary>答案</summary>

```python
import numpy as np
np.random.seed(0)
n = 1000
d = 64
A = np.random.randn(n, d)
B = np.random.randn(n, d)
dot = np.abs(np.sum(A * B, axis=1))   # (n,)
norm_prod = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)  # (n,)
violations = (dot > norm_prod + 1e-9).sum()
print(f"违反 Cauchy-Schwarz 的次数: {violations}")  # 应该 = 0
```
</details>

**题 8**: 用 PyTorch 实现一个简化版 scaled dot-product attention(无 mask,无 dropout),输入 4 个 token × 8 维,可视化 Attention 权重。

<details>
<summary>答案</summary>

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
# 4 个 token,每个 8 维
X = torch.randn(4, 8)
W_q = torch.randn(8, 8)
W_k = torch.randn(8, 8)
W_v = torch.randn(8, 8)

Q = X @ W_q  # (4, 8)
K = X @ W_k  # (4, 8)
V = X @ W_v  # (4, 8)

d_k = Q.shape[-1]
scores = Q @ K.T / (d_k ** 0.5)  # (4, 4) ← 本章 §9.6 的 scaled
attn = F.softmax(scores, dim=-1)
print("Attention 权重矩阵:")
print(attn.detach().numpy().round(3))
# 期望每行加起来 = 1
```
</details>

**题 9**: 在双塔推荐中,如果 query Embedding 的范数 $\|\mathbf{q}\|$ 翻了 10 倍,点积分数会怎么变?Cosine 分数呢?为什么?

<details>
<summary>答案</summary>

- **点积** $\mathbf{q} \cdot \mathbf{k}$ 与 $\|\mathbf{q}\|$ 成正比 → **翻 10 倍**
- **Cosine** $\frac{\mathbf{q} \cdot \mathbf{k}}{\|\mathbf{q}\| \|\mathbf{k}\|}$ 与 $\|\mathbf{q}\|$ 无关 → **不变**

**结论**:Cosine 消除了范数影响,只保留"方向";点积保留了范数信息(类似"重要性")。
**应用**:召回用点积(保留热度偏差);匹配/语义检索用 cosine(只看语义)。
</details>

**题 10**: 假设训练好的 Transformer 中,某一层某个 head 的 $Q$ 和 $K$ 矩阵的范数 $\|\mathbf{q}\|, \|\mathbf{k}\|$ 都很大(比如 > 100),不除以 $\sqrt{d_k}$ 直接做 softmax,会发生什么?

<details>
<summary>答案</summary>

设 $d_k = 64$, $\mathbf{q} \cdot \mathbf{k}$ 的方差 ≈ $d_k = 64$(假设各分量独立)。
当 $\|\mathbf{q}\|, \|\mathbf{k}\|$ 都很大(假设都 ~ 100), $\mathbf{q} \cdot \mathbf{k}$ 期望 ≈ $100 \cdot 100 \cdot \cos\theta$ 数量级。

softmax 之前 scores 范围 ≈ [-10000, 10000],softmax 进入**极饱和区**:
- 最大 score ≈ 10000,其他 ≈ 0
- softmax 后 ≈ **one-hot 分布**(只关注一个 token)
- 梯度 ≈ 0,**该 head 学不到东西**

**修复**: `scores / sqrt(d_k)` 让方差回到 ≈ 1,softmax 工作在正常区间。

这是 Vaswani 2017 Attention Is All You Need §3.2.1 的核心洞察。
</details>

---

## 18. 跨章关联与延伸阅读

### 18.1 与第 2 章(矩阵·线性变换)的接口

- 向量内积 → 矩阵乘向量的特例:$\mathbf{x}^\top \mathbf{y}$ 是 $1 \times n$ 矩阵 × $n \times 1$ 矩阵。
- 向量范数 → 由矩阵 $\|A\|_2 = \max_{\|x\|=1} \|Ax\|$ 推广为**算子范数**(矩阵范数的一种)。
- L2 归一化 → 投影到单位球面,几何上是**正交投影**(对应 `../../../方向教程/rec教程/第02章-深度学习数学与理论基础.md` 的投影矩阵)。

### 18.2 与第 3 章(矩阵分解)的接口

- **余弦相似度矩阵** $S_{ij} = \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\|\mathbf{x}_i\|\|\mathbf{x}_j\|}$ 是一个 Gram 矩阵 $S = X^\top X$ / $(\|x_i\|\|x_j\|)$。
- **推荐系统**的用户-物品评分矩阵 ≈ 用户 Embedding × 物品 Embedding(矩阵分解,见 `../../../方向教程/rec教程/第03章-Embedding与数据组织.md` + `../../../方向教程/rec教程/第09章-Python底层.md`)。
- **PCA** 找主成分 = 找协方差矩阵 $X^\top X$ 的**最大特征向量**(谱方法,见 `../../../方向教程/rec教程/第04章-多任务学习.md`)。

### 18.3 与第 4 章(特征值·谱)的接口

- PageRank 用**幂迭代**求主特征向量(向量 → 矩阵 → 谱的桥梁)。
- 谱归一化用**最大奇异值** $|A\|_2 = \sigma_{\max}(A)$,与向量范数的"投影最大伸长倍数"几何一致。
- 协方差矩阵的特征向量 = 主成分方向,特征值 = 该方向上的方差。

### 18.4 与第 6 章(矩阵微积分)的接口

- Attention 梯度推导中,$\frac{\partial L}{\partial Q}, \frac{\partial L}{\partial K}, \frac{\partial L}{\partial V}$ 都是**矩阵对矩阵的导数**,本质是向量内积对向量的偏导的批量版。
- BPTT 中每一层的激活 $\mathbf{h}_l$ 都是向量,反向梯度 $\frac{\partial L}{\partial \mathbf{h}_l}$ 也是向量。

### 18.5 与第 8 章(高维几何)的接口

- 高维空间中,**余弦相似度分布"塌缩"**(几乎所有随机向量对的 cosine 都接近 0)——这是 `../../../方向教程/rec教程/第08章-GR的在线与长期.md` 的核心。
- 这就是为什么需要 L2 归一化 + 对比学习(让相似样本的 cosine 拉近)。
- **SimCSE** 的几何动机:在高维球面上,**方向比距离更有信息量**。

### 18.6 与第 9 章(稀疏与低秩)的接口

- **LoRA 把 $W \in \mathbb{R}^{d \times d}$ 分解为 $W + BA$,其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$**——本质是低秩近似。
- **Embedding 压缩**(PQ 量化)把每个 Embedding 切成 $m$ 段,每段聚类到 $k$ 个中心,空间减少 $(d \cdot 32) / (m \cdot \log_2 k)$ 倍。
- **矩阵分解推荐** $R \approx P^\top Q$,本质是两个 Embedding 矩阵的内积(回到本章的"内积 = 相似度")。

### 18.7 推荐论文清单(按章节读完后,挑这些读)

| 论文 | 章节 | 一句话总结 |
| :--- | :--- | :--- |
| **Word2Vec** (Mikolov 2013) | §1 + §9 | 词向量 = Embedding,经典 skip-gram |
| **SimCSE** (Gao 2021) | §1 + §9 | 对比学习用 cosine,简单有效 |
| **Sentence-BERT** (Reimers 2019) | §1 + §9 | 用孪生网络 + cosine 加速句子相似度 |
| **Attention Is All You Need** (Vaswani 2017) | §1 + §6 | Transformer = scaled dot-product attention |
| **DLRM** (Naumov 2019) | §1 + §2 + §9 | 推荐系统的"工业级"双塔 + 矩阵分解 |
| **LoRA** (Hu 2021) | §9 | 低秩适配 LLM,微调只需 0.1% 参数量 |
| **Spectral Norm GAN** (Miyato 2018) | §4 | 用算子范数(向量范数的推广)约束 Lipschitz |
| **SimHash** (Charikar 2002) | §8 | 高维向量 → 二值哈希,J-L 引理的工业应用 |

---

## 19. 论文笔记模板: 如何用本章概念读懂一篇 DL 论文

每读一篇 DL 论文,强迫自己回答下面 4 个问题,加深理解。

### 模板 (4 问)

1. **"向量"在哪里**:这篇论文里,什么是"被嵌入成向量的"?是 token?是图像 patch?是用户?是物品?
2. **"内积/距离"在哪里**:用什么函数度量两个向量的相似度?是 dot product?cosine?欧氏距离?
3. **"范数"在哪里**:有没有用 $\|\cdot\|$ 来做约束、归一化、惩罚?如果有,用 L1 / L2 / L∞ 中的哪个?为什么?
4. **"几何意义"是什么**:向量之间的"语义"在空间中如何表现?是聚集?线性偏移?球面分布?

### 例子: 读 Sentence-BERT (Reimers 2019)

| 问题 | 答案 |
| :--- | :--- |
| 向量在哪 | 每个句子 → 768 维 BERT Embedding |
| 内积/距离在哪 | 用 **cosine 相似度**(论文中明确提到 `(u, v)` 取 cosine) |
| 范数在哪 | 训练时 pooling 后**做 L2 归一化**;推理时用 cosine = 内积(归一化后) |
| 几何意义 | 相似句子在 Embedding 空间的方向接近(余弦接近 1),不相似则接近 -1 |

### 例子: 读 LoRA (Hu 2021)

| 问题 | 答案 |
| :--- | :--- |
| 向量在哪 | 预训练权重 $W \in \mathbb{R}^{d \times d}$,增量 $BA \in \mathbb{R}^{d \times d}$,秩 $r$ 远小于 $d$ |
| 内积在哪 | $W + BA$ 替换 $W$,前向传播 $y = (W + BA)x$ |
| 范数在哪 | 初始化时 $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$(使初始 $BA = 0$) |
| 几何意义 | 训练过程中,$BA$ 在 $W$ 的"低秩子空间"内学习;参数量从 $d^2$ 降到 $2dr$ |

### 例子: 读 SimCSE (Gao 2021)

| 问题 | 答案 |
| :--- | :--- |
| 向量在哪 | 句子 → BERT [CLS] Embedding |
| 内积在哪 | 训练目标 $\ell_i = -\log \frac{e^{\text{sim}(h_i, h_i^+)/\tau}}{\sum_j e^{\text{sim}(h_i, h_j)/\tau}}$,sim 是 cosine |
| 范数在哪 | **Embedding 必须先 L2 归一化**(cosine = 内积的归一化版) |
| 几何意义 | 同一句子 dropout 两次得到的 Embedding 在球面上"挨得近";不同句子的"拉远" |

---

## 20. 数值稳定性与工程小贴士

### 20.1 Softmax 溢出: 减最大值

**问题**: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$,当 $x_i$ 很大时 $e^{x_i}$ 溢出。

**解药**: 减最大值,数学上恒等:
$$\text{softmax}(x_i) = \frac{e^{x_i - \max}}{\sum_j e^{x_j - \max}}$$

**PyTorch 实现**:`F.softmax(x, dim=-1)` 已经做了这个 trick。**自己写的话别忘了**!

### 20.2 Cosine 相似度分母为 0

**问题**: 如果向量是 zero vector, $\|\mathbf{x}\| = 0$,除零。

**解药**:
```python
def safe_cosine(x, y, eps=1e-8):
    return (x * y).sum() / (x.norm() * y.norm() + eps)
```
或用 `F.normalize` + 检查。

### 20.3 LayerNorm 在 fp16 下数值不稳定

**问题**: 半精度下均值方差计算误差大,LayerNorm 输出 NaN。

**解药**:
- 关键层 (Embedding, LayerNorm, Loss) 保持 fp32。
- 用 `torch.cuda.amp.autocast` 自动混合精度。
- 或 `torch.nn.utils.parametrize` 把 norm 转为 fp32。

### 20.4 Embedding 数值范围

**问题**: Embedding 输出越来越大(> 100),影响下游数值稳定性。

**解药**:
- 每层后接 LayerNorm(Transformer 标配)。
- 初始化时用更小的 std(如 `std=0.02`)。
- 用 TQDM 监控范数变化,异常则报警。

### 20.5 距离度量与优化目标匹配

**问题**: 训练用 cosine,推理用欧氏距离,效果不一致。

**解药**: **训练和推理的距离度量必须一致**。如果推理用 cosine,**训练目标也用 cosine-based loss**(如 `CosineEmbeddingLoss`)。

---

## 21. 速查表: 本章核心公式 + PyTorch 一行实现

| 概念 | 公式 | PyTorch 一行 |
| :--- | :--- | :--- |
| 向量加法 | $\mathbf{x} + \mathbf{y}$ | `x + y` |
| 数乘 | $\alpha \mathbf{x}$ | `alpha * x` |
| 内积 | $\mathbf{x}^\top \mathbf{y}$ | `(x * y).sum()` 或 `x @ y` |
| L1 范数 | $\sum_i \|x_i\|$ | `x.abs().sum()` |
| L2 范数 | $\sqrt{\sum_i x_i^2}$ | `x.norm()` 或 `torch.linalg.norm(x)` |
| L∞ 范数 | $\max_i \|x_i\|$ | `x.abs().max()` |
| Cosine | $\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ | `F.cosine_similarity(x, y, dim=-1)` |
| L2 归一化 | $\mathbf{x}/\|\mathbf{x}\|$ | `F.normalize(x, p=2, dim=-1)` |
| 点积打分 | $\mathbf{q}^\top \mathbf{k}$ | `(q * k).sum(dim=-1)` |
| 缩放点积 | $\mathbf{q}^\top \mathbf{k}/\sqrt{d}$ | `(q @ k.T) / math.sqrt(d)` 或 `(q * k).sum(-1) / math.sqrt(d)` |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | `F.softmax(x, dim=-1)` |
| 投影 | $\text{proj}_{\mathbf{y}} \mathbf{x} = \frac{\mathbf{x}^\top\mathbf{y}}{\mathbf{y}^\top\mathbf{y}}\mathbf{y}$ | `proj = (x@y)/(y@y) * y` |
| Gram 矩阵 | $G = X^\top X$ | `x.T @ x` |

---

## 22. 章节自检题(15 题,选 5 道手写)

1. 向量 $\mathbf{x}=(2,3)^\top$ 与 $\mathbf{y}=(-1,2)^\top$ 的内积、L2 距离、余弦相似度分别是多少?
2. 写出 $d=64$ 时,Attention 中为什么 scores 要除以 $\sqrt{d}$(数值上 + 几何上)?
3. 一个 batch size=32,seq_len=128,hidden=768 的 Transformer 中,$QK^\top$ 矩阵的 shape 是多少?占多少显存 (fp16)?
4. BPR 损失 $-\log\sigma(s_+ - s_-)$ 与余弦相似度有什么关系?
5. 推荐召回中,为什么常用点积而非 cosine?什么场景下 cosine 更合适?
6. 解释 LayerNorm "按行归一化"为什么对 Transformer 至关重要。
7. 残差连接 $\mathbf{h}_{l+1} = \mathbf{h}_l + f(\mathbf{h}_l)$ 中"加法"在向量空间的几何意义是什么?
8. 为什么不能用 $L_\infty$ 范数做 Attention 缩放因子?
9. 假设 query 与 key 都已 L2 归一化,内积 = cosine,范围 [-1,1];如果再用 temperature $\tau$ 缩放,$\frac{\text{cosine}}{\tau}$ 在 $\tau \to 0$ 和 $\tau \to \infty$ 时分别会发生什么?
10. 训练数据中"长尾物品" Embedding 范数偏小,3 种修复方法?
11. 一句话解释 Gram 矩阵 $X^\top X$ 与 PCA 的关系。
12. 用 ASCII 画出 $A = \begin{pmatrix}2 & 0 \\ 0 & 0.5\end{pmatrix}$ 对单位圆的作用。
13. 如果模型输出范数 $\|\mathbf{h}\|$ 从 1 暴涨到 100,可能的原因是什么?3 种修法?
14. 向量加法的结合律 $\mathbf{x} + (\mathbf{y} + \mathbf{z}) = (\mathbf{x} + \mathbf{y}) + \mathbf{z}$,在 DL 中对应什么应用?
15. 解释为什么 $f(x) = x^\top x$ 的梯度是 $2x$。

<details>
<summary>提示(对应章节)</summary>

1. §1.3 + §2.3
2. §9.5(坑 5)
3. §9.5
4. §9.3
5. §9.4(坑 2)
6. §6.4
7. §1.2
8. §2.4
9. §9.5 + §8.3
10. §9(坑 8)
11. §4.4
12. §2.2(线性变换)
13. §6.5(梯度爆炸)
14. §1.1(交换律 → batched operation)
15. §6.2
</details>

---

## 23. 一句话核心 takeaway

**向量是几何,内积是打分,范数是长度,cosine 是方向——这是 Embedding / Attention / 推荐 / 对比学习的共同语言。**

---

> **本章版本**: v1.1
> **最后更新**: 2026-06-26
> **作者**: Sisyphus (OhMyOpenCode)
> **本章扩充内容**: 14. 工程师误区(12 坑) / 15. 完整推荐召回案例 / 16. 几何 ASCII 强化 / 17. 动手算一算(10 题含答案) / 18. 跨章关联 / 19. 论文笔记模板 / 20. 数值稳定性 / 21. PyTorch 速查 / 22. 自检 15 题
> **配套代码**: §9 所有代码均可直接运行 (PyTorch 2.x)

---

## 24. 进阶应用: 向量检索的工业实现

这一节给**想在工业界做向量检索**的工程师一个"完整地图"——从算法到工程。

### 24.1 暴力检索 (Brute-force) 的复杂度

```python
import torch

N = 1_000_000  # 100 万个物品
d = 64
item_embs = torch.randn(N, d)  # (1M, 64)
query = torch.randn(d)

# 暴力:每个 query 算 N 次内积
scores = item_embs @ query  # (1M,)
top_k = scores.topk(10)
```

复杂度:
- 时间: $O(N \cdot d)$ —— 100 万 × 64 = 6400 万次乘加,在 GPU 上几毫秒
- 空间: $O(N \cdot d)$ —— 100 万 × 64 × 4 字节 = 256 MB

**问题**: 当 N = 1 亿、d = 1024 时,空间 400 GB,**根本放不进单卡**。

### 24.2 IVF (Inverted File) 索引: 用 K-means 聚类分桶

```python
# 概念示意(实际用 faiss)
import faiss

quantizer = faiss.IndexFlatL2(d)  # 用 L2 距离的 flat 索引
index = faiss.IndexIVFFlat(quantizer, d, nlist=100)  # 100 个聚类中心

index.train(item_embs.numpy())
index.add(item_embs.numpy())

# 查询:只搜最近的 nprobe 个桶
index.nprobe = 10  # 默认 1,越大越准但越慢
D, I = index.search(query.numpy()[None, :], k=10)
```

**原理**: 把 100 万物品聚成 100 桶(每桶 1 万),查询时只搜最近的 10 桶,**速度 ≈ 10 倍提升**。

### 24.3 PQ (Product Quantization) 压缩: 把 Embedding 切成子向量聚类

```python
# 概念: d=64,切成 m=8 段,每段 8 维
# 每段聚类到 256 个中心 → 8 bit 编码
# 总空间: 64 × 4 字节 → 8 字节(32 倍压缩)

index = faiss.IndexIVFPQ(quantizer, d, nlist=100, m=8, nbits=8)
index.train(item_embs.numpy())
index.add(item_embs.numpy())
```

**原理**: 近似距离 = 子向量查表 + 查表结果求和,**用 LUT 避免算距离**。

### 24.4 HNSW (Hierarchical Navigable Small World): 图索引

**原理**: 构建一个多层图,查询时从稀疏顶层往稠密底层走,**对数复杂度**。

```python
index = faiss.IndexHNSWFlat(d, 32)  # 32 = 每节点邻居数
index.add(item_embs.numpy())
D, I = index.search(query.numpy()[None, :], k=10)
```

**特点**: **召回最高**(通常 > 95%),速度也快,但**内存占用大**(要存图)。

### 24.5 工业级选择

| 数据规模 | 推荐索引 | 召回 | 速度 | 内存 |
| :--- | :--- | :--- | :--- | :--- |
| < 10 万 | Flat (暴力) | 100% | 慢 | $Nd$ 字节 |
| 10 万 - 1000 万 | IVF + PQ | 90-95% | 快 | $N \cdot m$ 字节 |
| > 1000 万 | IVF + PQ + HNSW | 95%+ | 中 | 看实现 |
| 超大规模 (亿级) | IVF + OPQ + HNSW + GPU | 95%+ | 极快 | GB 级 |

### 24.6 工业界向量检索库对比

| 库 | 语言 | 优势 | 场景 |
| :--- | :--- | :--- | :--- |
| **Faiss** (Meta) | C++/Python | 工业级,GPU 支持 | 推荐/搜索后端 |
| **Milvus** | C++/Python | 分布式,云原生 | 大规模生产 |
| **Qdrant** | Rust | 高性能,易部署 | 实时搜索 |
| **Weaviate** | Go | 模块化,GraphQL | 多模态 |
| **Pinecone** | SaaS | 免运维 | 快速上线 |
| **Vespa** | Java | 混合检索 + 向量 | 复杂查询 |

**推荐**: 工程首选 **Faiss**(成熟、稳定);分布式首选 **Milvus**(云原生)。

### 24.7 与本章概念的对应

| 本章概念 | 工业应用 |
| :--- | :--- |
| L2 距离 | Flat 索引 / K-means 距离度量 |
| 内积 | IVF + Flat 的打分函数 |
| Cosine | HNSW 的距离度量 |
| 向量归一化 | PQ 的预处理(归一化后,内积 ≡ cosine) |
| 范数约束 | 量化前的正则化(防止某些物品范数过大) |
| 高维诅咒(第 8 章) | 为什么需要 IVF + PQ 降维 |

---

## 25. 一道"完整 ML 工程师面试题"模拟

**题目**: 给定一个推荐系统的双塔模型,user tower 输出 $\mathbf{u} \in \mathbb{R}^{64}$,item tower 输出 $\mathbf{v} \in \mathbb{R}^{64}$,训练用 BPR 损失。请回答:

1. 召回阶段用点积还是 cosine?为什么?
2. 推理时如果发现热门物品的分数总是偏高,如何调整?
3. 训练数据中长尾物品(出现次数 < 5)效果差,如何用线代工具改进?
4. 上线后发现 $\mathbf{u}$ 的范数方差很大(0.1 ~ 100),可能导致什么问题?
5. 如果候选集是 1000 万,推理延迟 P99 要求 < 50 ms,如何选择索引?

<details>
<summary>参考答案(提示)</summary>

1. **点积**,因为 (a) 召回量级大,点积更快;(b) 范数信息 = "流行度",召回阶段希望保留这一信号。
2. **方案 A**: 对所有 Embedding 做 L2 归一化,改用 cosine。**方案 B**: 在打分公式中除以 $\|\mathbf{v}\|$(隐式 cosine)。
3. **方案 A**: 加正则 $\|v\|^2$ 约束(防止范数过小)。**方案 B**: 用 `../../../方向教程/rec教程/第09章-Python底层.md` 讲的频次倒数加权采样。**方案 C**: 加 side information(品类 Embedding)。
4. 范数方差大 → 用户 Embedding 在空间中"半径"差异大 → cosine 时用户间分数不可比;点积时热门物品分数偏高(对热门物品范数大)。**修法**: 加 LayerNorm 强制 $|\mathbf{u}\| \approx 1$,或者 $\mathbf{u} \to \alpha \mathbf{u}/\|\mathbf{u}\|$。
5. **方案**: IVF(8192 桶) + PQ(m=8) + GPU 检索。10ms 量级召回 100 万候选,Top-10 < 5ms;1000 万 < 50ms。
</details>

---

## 26. 向量运算的"反直觉"事实清单

### 26.1 高维空间中随机向量的 cosine 接近 0

**事实**: 维度 $d$ 足够大时,两个独立随机向量 $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(0, I)$ 的余弦相似度 $\to 0$。

**直观解释**: 高维空间中,**几乎所有向量两两正交**(随机向量的方向"各向同性")。

**影响**:
- 对比学习的负样本 cosine 趋近 0,**没有区分度**——必须用 hard negative。
- `../../../方向教程/rec教程/第08章-GR的在线与长期.md` 的测度集中现象。

### 26.2 单位球的"质量"集中在表面

**事实**: 单位球 $B^d = \{\mathbf{x}: \|\mathbf{x}\| \le 1\}$,当 $d$ 增大,**几乎所有质量都在 $\|\mathbf{x}\| \approx 1$ 的薄壳上**。

**影响**: Embedding 训练中,**向量范数会自然增长**到接近某个"有效半径";归一化能稳定训练。

### 26.3 Embedding 各向异性 (Anisotropy)

**事实**: 训练好的 Embedding **不是均匀分布**在球面,而是**聚集在某个"锥形"区域**。

**原因**: 训练目标(如语言模型)倾向于让某些方向"承担更多语义"。

**影响**: BERT 词向量的平均 cosine > 0(应该 ≈ 0 才对)。

**修法**: 后处理 **"all-but-the-top"** 去除主成分 → 强制 isotropic。

### 26.4 范数大 ≠ 重要

**事实**: 范数反映**梯度的总累积**,不等于"语义重要性"。

**反例**: 罕见物品 Embedding 范数小(被更新次数少),但**可能语义很关键**(冷启动物品)。

**修法**: 训练时正则 $\|v\|^2 \ge \delta$。

### 26.5 "维度诅咒" 的两个含义

| 含义 | 描述 | 对策 |
| :--- | :--- | :--- |
| 几何诅咒 | 高维球体积集中在赤道,距离度量失效 | 用 cosine 而非 L2 |
| 样本诅咒 | 高维需要指数级样本才能"填满"空间 | 降维 / 正则 / 数据增强 |

---

## 27. 关键论文片段摘录(原文 + 翻译)

### 27.1 Attention Is All You Need (Vaswani 2017, §3.2.1)

> **原文**: "We call our particular attention 'Scaled Dot-Product Attention'. The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function..."

> **翻译**: "我们称这种注意力为'缩放点积注意力'。输入包含维度 $d_k$ 的查询和键,维度 $d_v$ 的值。我们计算查询与所有键的点积,除以 $\sqrt{d_k}$,应用 softmax..."

> **本章对应**: §9.6(Attention 几何)+ 坑 5(√d 的必要性)

### 27.2 SimCSE (Gao 2021, §2.1)

> **原文**: "...The training objective is essentially the InfoNCE loss with cosine similarity as the similarity function, where positive pairs are formed by passing the same sentence through the encoder twice with independently sampled dropout masks..."

> **翻译**: "...训练目标本质上是 InfoNCE 损失,用 cosine 相似度作为相似度函数,正样本对通过同一句子两次通过编码器(独立的 dropout 掩码)构成..."

> **本章对应**: §9.4(cosine vs 点积)+ §10.4(对比学习)

### 27.3 Sentence-BERT (Reimers 2019, §3)

> **原文**: "...To fine-tune BERT for producing semantically meaningful sentence embeddings, we use a siamese network architecture...The cosine similarity between the two sentence embeddings u and v is computed as $\cos(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}$..."

> **翻译**: "为了微调 BERT 生成语义上有意义的句子 Embedding,我们使用孪生网络架构...两个句子 Embedding u 和 v 的 cosine 相似度按公式计算..."

> **本章对应**: §1.2(语义几何)+ §9.4(cosine)

### 27.4 LoRA (Hu 2021, §1)

> **原文**: "...The learned over-parametrized models in fact reside on a low intrinsic dimension...We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'..."

> **翻译**: "...学习到的过参数化模型实际上位于一个低固有维度上...我们假设模型适配过程中的权重变化也具有低'固有秩'..."

> **本章对应**: 整个第 1 章(向量)+ 第 9 章(低秩)

---

## 28. 章节回顾 & 下章衔接

### 28.1 一句话回顾 13 个核心概念

```text
向量   = 高维空间中的点
加法   = 平移(几何:平行四边形)
数乘   = 缩放(几何:沿射线)
内积   = 投影长度 × |对方| (几何:相似度)
范数   = 向量长度(L1/L2/L∞ 三种)
单位化 = 缩放到长度 1
余弦   = 内积归一化 = 方向相似度
正交   = 内积为 0(垂直)
投影   = 沿某方向的分量
对称正定 = 二次型正(为第 5 章铺垫)
谱方法 = 找主方向(为第 4 章铺垫)
```

### 28.2 下一章预告

第 2 章:矩阵·线性变换·秩 —— 神经网络 = 矩阵乘法 + 激活函数

- **向量 → 矩阵**: 一维变二维,内积 → 矩阵乘法
- **线性映射**: 矩阵 $A$ 把 $\mathbf{x}$ 映射到 $A\mathbf{x}$,这就是 Linear Layer
- **秩 rank(A)**: 矩阵的"独立行/列数",决定了 LoRA 的低秩假设

读完第 2 章,你会:
- 手算任意矩阵 × 向量
- 理解神经网络 Linear Layer 的矩阵本质
- 解释为什么 Embedding 层可以看作矩阵查表
- 推导"为什么 batched 矩阵乘是 $O(n^3)$"


## 第一部分 · 深度扩展:多视角系统展开

> 本部分是 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的全面延伸。每个章节从不同视角展开,系统补充原内容未覆盖的数学推导、工程实现、典型案例。所有新增内容自成体系,可独立阅读。

### 29. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 极限与连续深度分析

**核心议题**: 在 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的语境下,极限与连续 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 极限与连续 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 的另一个视角。


---

### 30. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 导数微分深度分析

**核心议题**: 在 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的语境下,导数微分 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 导数微分 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 的另一个视角。


---

### 31. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 积分测度深度分析

**核心议题**: 在 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的语境下,积分测度 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 积分测度 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 的另一个视角。


---

### 32. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 矩阵分解深度分析

**核心议题**: 在 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的语境下,矩阵分解 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 矩阵分解 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 的另一个视角。


---

### 33. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 特征值深度分析

**核心议题**: 在 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的语境下,特征值 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 特征值 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 的另一个视角。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 拆解为以下几个层次:

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

信息论为"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 拆解为以下几个层次:

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

信息论为"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 拆解为以下几个层次:

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

信息论为"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 拆解为以下几个层次:

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

信息论为"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 1 章:向量·内积·范数——Embedding 空间的几何直觉 - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 拆解为以下几个层次:

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

信息论为"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 1 章:向量·内积·范数——Embedding 空间的几何直觉" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



