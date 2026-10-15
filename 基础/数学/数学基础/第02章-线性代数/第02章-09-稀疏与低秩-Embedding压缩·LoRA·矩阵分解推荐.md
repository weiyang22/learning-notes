---
title: 稀疏与低秩：Embedding 压缩 · LoRA · 矩阵分解推荐
tags:
  - 稀疏矩阵
  - 低秩分解
  - LoRA
  - Embedding 压缩
  - 矩阵分解
  - ALS
  - 线性代数
description: 稀疏与低秩两大压缩视角：CSR 稀疏格式、LoRA 低秩微调、Funk SVD/ALS 矩阵分解与工业 Embedding 压缩方法
chapter_id: 数学-2-09
last_updated: 2026-07-14
status: ✅ v1.0
---

# 第02章-09 稀疏与低秩：Embedding 压缩 · LoRA · 矩阵分解推荐

> 本章导读:要把这件事**真正用到工程上**:搜推广的 Embedding 表太大,装不下显存,怎么压?大模型微调成本太高,怎么省?推荐系统的 User-Item 矩阵全是 0,怎么存?答案都指向两个词:**稀疏**与**低秩**。
>
> 这一章会带你搞清楚四件工程界每天都用的事:
> - Embedding 表 25 GB 装不下,怎么用稀疏存储 + 低秩分解压到 2 GB
> - LLaMA 7B 微调要 28 GB 显存,LoRA 怎么只用 35 M 参数搞定
> - Netflix 百万奖金的矩阵分解到底在算什么
> - 为什么 Attention 的 QKV 矩阵"天然低秩",可以剪枝
>
> **学完你能回答**:
> - 稀疏矩阵有几种存储格式?CSR 为什么是工业默认?
> - Embedding 训练为什么必须稀疏更新?全量更新会怎样?
> - LoRA 的 $\Delta W = BA^T$ 为什么等价于"低秩子空间内的微调"?
> - Funk SVD 和 ALS 的区别?为什么工业上 ALS 更快?
> - Embedding 压缩的工业方法有哪些?(量化、PQ、Hash Trick)
>
> **与前后章节的关系**:
> - 第 2 章矩阵的"秩"是本章的理论基础——低秩 = 秩远小于维度
> - 第 3 章的 SVD 是低秩分解的"最优解"(Eckart-Young 定理)
> - 第 5 章的正定性、稀疏 Adam 是稀疏训练的优化基础
> - 第 6 章的矩阵微积分是 LoRA 梯度推导的工具
> - 第 7 章张量 stride 帮你理解稀疏张量的内存布局
> - 7-系统思维案例 1 (DLRM) 和案例 4 (HSTU) 是本章的工程落地

---

## 1. 直觉引入:为什么需要"稀疏 + 低秩"?

### 1.1 工程师的痛点:1 亿物品 × 64 维 Embedding = 25.6 GB,单卡装不下

假设你在做一个推荐系统。**物品库 1 亿**,每个物品用 64 维 FP32 向量表示。

```text
存储需求 = 1e8 × 64 × 4 bytes (FP32)
        = 2.56e10 bytes
        = 25.6 GB
```

单张 A100 只有 80 GB 显存,光 Embedding 表就占了 1/3。但更糟糕的是:**每条样本只访问几十个物品**(用户在一次请求中只看到几十个物品),剩下 99.999% 的 Embedding 完全没被访问。

**反向思考**:为什么我需要把 25 GB 的表"全部"装进显存?

- 推理时:只需要查那几十个物品的 Embedding → **稀疏查表**
- 训练时:每次只更新被访问的那几十行 → **稀疏更新**

更激进一点:如果 Embedding 表本身是"低秩"的(可以用 $U V^T$ 近似),那连存都不用存完整的表 → **低秩分解**。

**两个救命的观察**:

| 观察 | 工程意义 | 数学对应 |
| :--- | :--- | :--- |
| 每次只访问 1% 的物品 | 稀疏查表 + 稀疏梯度 | **稀疏矩阵** |
| Embedding 表的"有效秩"远小于 64 维 | 低秩分解存储 | **低秩矩阵** |

### 1.2 直觉:99% 的 Embedding 是"稀疏更新",只有 1% 被访问

推荐系统的数据流:

```text
一次训练 step:
  1. 取一个 batch(1024 条样本)
  2. 每条样本有 user_id + 几个 item_id + label
  3. 每个 item_id 查 Embedding 表 → 拿到 64 维向量
  4. 前向 + 反向传播
  5. 反向传播只对"被查过的" item_id 产生非零梯度
  6. 优化器只更新那些行
```

每条样本涉及的 item_id 通常 < 50。一整个 epoch 可能只覆盖物品库的 5%。

**关键观察**:99% 的 Embedding 行,在每个 step 都是 0 梯度。为什么要"假装"它们被更新了?

这就是**稀疏 Adam**(PyTorch `torch.optim.SparseAdam`):只对被访问的行做更新,其他行保持不变。

### 1.3 ASCII 可视化:稀疏矩阵 + 低秩矩阵

**稀疏矩阵**(只存非零元素的 (row, col, value)):

```text
稠密表示 (10x10, 100 个数字):        稀疏表示 (COO, 6 个非零 → 18 个数字):

┌───────────────────┐               row = [0, 1, 3, 5, 7, 9]
│ 0  0  0  0  0  0  0  0  0  0 │     col = [2, 5, 1, 8, 0, 4]
│ 0  0  0  0  0  0  0  0  0  0 │     val = [3, 7, 1, 5, 9, 2]
│ 0  3  0  0  0  0  0  0  0  0 │
│ 0  1  0  0  0  0  0  0  0  0 │     内存: 18 vs 100(节省 82%)
│ 0  0  0  0  0  0  0  0  0  0 │
│ 0  0  0  0  0  0  0  5  0  0 │
│ 0  0  0  0  0  0  0  0  0  0 │
│ 9  0  0  0  2  0  0  0  0  0 │
│ 0  0  0  0  0  0  0  0  0  0 │
│ 0  0  0  0  0  0  0  0  0  0 │
└───────────────────┘
```

**低秩矩阵**(A = U V^T,大矩阵用两个瘦矩阵的乘积表示):

```text
A (m×n, m·n 个数字):       U (m×r)            V^T (r×n)

┌─────────────┐           ┌──┐               ┌─────────┐
│ ░ ░ ░ ░ ░ ░ │           │██│               │░ ░ ░ ░ ░│
│ ░ ░ ░ ░ ░ ░ │           │██│               │░ ░ ░ ░ ░│
│ ░ ░ ░ ░ ░ ░ │ =         │██│ ×             │░ ░ ░ ░ ░│
│ ░ ░ ░ ░ ░ ░ │           │██│               │░ ░ ░ ░ ░│
│ ░ ░ ░ ░ ░ ░ │           └──┘               └─────────┘
└─────────────┘            m × r                 r × n

内存: m·n → (m+n)·r      当 r << min(m, n),节省 100 倍以上
```

### 1.4 两种压缩路线

```text
                    Embedding 表 (m 物品 × d 维,稀疏/低秩)
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                                       ▼
          【稀疏路线】                              【低秩路线】
      只存被访问的 (idx, value)                W ≈ U V^T (rank r << d)
                │                                       │
        ┌───────┼───────┐                       ┌───────┼───────┐
        ▼       ▼       ▼                       ▼       ▼       ▼
      COO     CSR     CSC                    FunkSVD   LoRA    Embed压缩
    (三元组) (按行压) (按列压)                (推荐)   (微调)   (PQ/INT8)
```

**这一章会逐一展开**:稀疏存储(§2)、稀疏训练(§3)、低秩矩阵(§4)、LoRA(§5)、矩阵分解推荐(§6)、Embedding 压缩(§7)。

### 1.5 章节路线图

```text
1. 直觉引入                          ← 你在这里
2. 稀疏矩阵 (Sparse Matrix)          ← COO/CSR/CSC/LIL, 5 个手算
3. 稀疏 Embedding 训练               ← nn.Embedding / SparseAdam / TorchRec
4. 低秩矩阵 (Low-Rank Matrix)        ← A = UV^T, 5 个手算
5. LoRA                              ← ΔW = BA^T, 5 个手算 + PyTorch 实现
6. 矩阵分解推荐 (Matrix Factorization) ← Funk SVD / ALS / BPR
7. Embedding 压缩的工业实践           ← Quantization / PQ / Hash
8. 稀疏与低秩的"协同"                ← Embedding = 稀疏访问 + 低秩存储
9. PyTorch 代码实现                   ← 6 个可运行实验
10. 常见误区                          ← 5-8 条
11. 面试 Q&A                        ← L7/L8 怎么答
12. 与其他数学工具的联系              ← 跨章节链接
```

---

## 2. 稀疏矩阵(Sparse Matrix)

### 2.1 定义:大部分元素为 0 的矩阵

**定义**(稀疏矩阵):矩阵 $A \in \mathbb{R}^{m \times n}$ 称为**稀疏的**,如果其**非零元素个数** $\text{nnz}(A) \ll m \cdot n$。

**稀疏度**(sparsity):

$$\text{sparsity} = 1 - \frac{\text{nnz}(A)}{m \cdot n}$$

工业上的"稀疏"通常指 sparsity > 90%(非零 < 10%)。推荐系统的 User-Item 矩阵 sparsity 通常 > 99.9%。

**关键观察**:稀疏不仅是"0 多",更重要的是**有结构**(否则随机稀疏仍然难压缩)。

### 2.2 存储格式:COO / CSR / CSC / LIL

#### 2.2.1 COO(Coordinate Format)

最直观:存三个等长数组 `(row, col, value)`,每个非零元素占 3 个数字。

```text
稠密 (3×4):
┌───┬───┬───┬───┐
│ 0 │ 0 │ 3 │ 0 │
├───┼───┼───┼───┤
│ 5 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 7 │
└───┴───┴───┴───┘

COO 存储 (3 个非零):
  row = [0, 1, 2]
  col = [2, 0, 3]
  val = [3, 5, 7]

内存: 3×3 = 9 个数字(对比稠密 12,省 25%)
```

#### 2.2.2 CSR(Compressed Sparse Row)

按行压缩:每行只存"起始位置 + 值"。

```text
同样 3×4 矩阵:
rowptr = [0, 1, 2, 3]    # 3 行 + 1 个哨兵,共 4 个数字
col    = [2, 0, 3]       # 列索引
val    = [3, 5, 7]       # 值

第 0 行: val[rowptr[0]:rowptr[1]] = val[0:1] = [3] 在 col=[2]
第 1 行: val[rowptr[1]:rowptr[2]] = val[1:2] = [5] 在 col=[0]
第 2 行: val[rowptr[2]:rowptr[3]] = val[2:3] = [7] 在 col=[3]
```

**CSR 的优势**:
- 按行切片极快(给定 row,直接定位)
- 矩阵-向量乘法 $y = Ax$ 友好(逐行扫描)

#### 2.2.3 CSC(Compressed Sparse Column)

CSR 的"列版",按列压缩。适合按列操作的场景(矩阵转置、列切片)。

#### 2.2.4 LIL(List of Lists)

每行存一个 Python list,存该行的非零元素。**适合增量构造**,但不适合大规模计算。

### 2.3 5 个手算例子

#### 例 1:3×4 矩阵的 COO 编码

$$
A = \begin{pmatrix}
1 & 0 & 0 & 2 \\
0 & 0 & 3 & 0 \\
0 & 4 & 0 & 0
\end{pmatrix}
$$

```text
row = [0, 0, 1, 2]
col = [0, 3, 2, 1]
val = [1, 2, 3, 4]
```

#### 例 2:同一矩阵的 CSR 编码

```text
rowptr = [0, 2, 3, 4]    # 第 0 行有 2 个,第 1 行 1 个,第 2 行 1 个
col    = [0, 3, 2, 1]
val    = [1, 2, 3, 4]
```

#### 例 3:稀疏度计算

5×5 单位矩阵 $I_5$:nnz = 5,总元素 = 25,sparsity = 1 - 5/25 = 80%。

#### 例 4:稀疏矩阵-稠密向量乘法

$A \mathbf{x}$,其中 $A$ 是上面 3×4 矩阵,$\mathbf{x} = (1, 2, 3, 4)^T$。

- 第 0 行:$1 \cdot 1 + 2 \cdot 4 = 9$(只算 2 个非零,跳过 2 个零)
- 第 1 行:$3 \cdot 3 = 9$(只算 1 个)
- 第 2 行:$4 \cdot 2 = 8$(只算 1 个)

输出 $\mathbf{y} = (9, 9, 8)^T$。复杂度 = $O(\text{nnz})$,而非 $O(mn)$。

#### 例 5:稀疏矩阵-稀疏矩阵乘法(模式匹配)

$A \cdot B$ 中非零元素的位置 = $(i, k)$,其中存在 $j$ 使 $A[i,j] \neq 0$ 且 $B[j,k] \neq 0$。这需要"模式匹配",比稠密乘法复杂得多(SciPy 的 `sparse.csr_matrix.dot` 实现了)。

### 2.4 案例:推荐系统 User-Item 矩阵的稀疏度

**网飞(Netflix)数据集**:
- 用户:480,189 个
- 物品:17,770 部
- 评分:100,480,507 条
- 总元素:480,189 × 17,770 ≈ 8.5 × 10⁹
- 稀疏度:1 - 10⁸ / 8.5×10⁹ ≈ 98.8%

**抖音/快手推荐**:
- 用户:10 亿
- 物品(视频):1 亿
- 每天交互:100 亿次(点击、点赞、收藏)
- 总元素:10¹⁷
- 稀疏度:> 99.999%

**结论**:搜推广的 User-Item 矩阵**几乎都是 0**,用稠密存储是灾难性的浪费。

### 2.5 案例:为什么 Embedding 层的梯度是稀疏的

假设 Embedding 表 $W \in \mathbb{R}^{N \times d}$,前向时:
- 输入:item_id 序列 $[i_1, i_2, \ldots, i_B]$(长度为 B)
- 输出:$E = W[\text{idx}]$,其中 $E \in \mathbb{R}^{B \times d}$

反向时:
- 损失对 $E$ 的梯度:$G_E \in \mathbb{R}^{B \times d}$
- 损失对 $W$ 的梯度:$G_W \in \mathbb{R}^{N \times d}$
- $G_W[i] = \sum_{k: i_k = i} G_E[k]$(对所有访问过的行累加)
- 其他行:**梯度严格为 0**

```text
N = 1,000,000 个物品
batch = 1024 条样本,每条 10 个 item_id

被访问的行 ≈ 1024 × 10 = 10,240(可能有重复)
梯度非零的行 ≈ 10,240 / 1,000,000 = 1.02%
```

**这意味着**:99% 的 Embedding 行**梯度严格为零**。如果用普通 Adam,你会做 $N \times d$ 次更新,但其中 99% 是"加 0",纯浪费。

### 2.6 PyTorch 稀疏张量

```python
import torch

# 构造稀疏张量
i = torch.tensor([[0, 1, 2], [2, 0, 1]])  # 2x3, [0,2],[1,0],[2,1]
v = torch.tensor([3.0, 4.0, 5.0])
A = torch.sparse_coo_tensor(i, v, size=(3, 3))

print("稀疏张量:", A)
print("稠密化:\n", A.to_dense())

# 稀疏矩阵乘法
x = torch.randn(3, 1)
y = A @ x  # 稀疏-稠密乘法
print("A @ x:\n", y.to_dense() if y.is_sparse else y)

# CSR 格式
A_csr = A.to_sparse_csr()
print("CSR rowptr:", A_csr.crow_indices())
print("CSR col:", A_csr.col_indices())
print("CSR val:", A_csr.values())
```

**PyTorch 稀疏张量的局限**:
- 算子支持不全(`torch.sparse` 与 `torch.linalg` 的算子交集小)
- 不支持自动微分(`torch.sparse.mm` 不带 grad)
- GPU 加速有限(主要在 CPU 上优化)

**工业实践**:PyTorch 的 `nn.Embedding` 不直接用 sparse tensor,而是用**索引 + 稠密查表 + 稀疏优化器**的组合(详见 §3)。

---

## 3. 稀疏 Embedding 训练

### 3.1 稀疏 Adam:只在被访问 ID 上更新

**PyTorch `torch.optim.SparseAdam`**:专为 Embedding 设计。

```python
import torch
import torch.nn as nn

# 定义 Embedding 表
N, d = 1_000_000, 64
embedding = nn.Embedding(N, d)

# 关键:把 Embedding 的参数标记为"sparse"
# 实际上 SparseAdam 只对 .sparse 的梯度做更新
optim = torch.optim.SparseAdam(embedding.parameters(), lr=1e-2)

# 前向:稀疏查表
idx = torch.tensor([1, 5, 100, 999_999])
out = embedding(idx)  # 形状 (4, 64),只查 4 行

# 反向:梯度是稀疏的
loss = out.sum()
loss.backward()

# 检查:Embedding 的梯度是稀疏张量
print("weight.grad.is_sparse:", embedding.weight.grad.is_sparse)
print("weight.grad 非零元素:", embedding.weight.grad._values().shape)

# 更新:只更新 4 行
optim.step()
```

**底层逻辑**:
1. `SparseAdam` 调用 `embedding.weight.grad` 时,PyTorch 检测到"梯度来自 index_select",自动返回**稀疏梯度**
2. 优化器只遍历 `grad._indices()`(被访问的行号),对这些行做 Adam 更新
3. 其他行的状态保持不变(不写显存)

**为什么不用普通的 `Adam`?** 因为 `Adam` 需要"全局状态",每个参数都有动量 $m_t$ 和二阶动量 $v_t$。如果 Embedding 有 1 亿行 × 64 维 × 2(m + v) × 4 bytes = 51.2 GB,光优化器状态就比模型还大。

**SparseAdam 的关键约束**:embedding 必须是 dense(参数本身稠密存储),但梯度是 sparse,优化器只更新被访问的行。

### 3.2 PyTorch `nn.Embedding` 的实现

`nn.Embedding(N, d)` 的本质:**一个 lookup table**,不是矩阵乘法。

```python
# nn.Embedding 等价于:
class MyEmbedding(nn.Module):
    def __init__(self, N, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, d))  # (N, d)

    def forward(self, idx):
        # idx: (*), 输出: (*, d)
        return self.weight[idx]  # index_select, 稀疏查表

# 验证
emb = nn.Embedding(10, 4)
idx = torch.tensor([0, 2, 5])
print(emb(idx).shape)  # (3, 4)
```

**两种使用模式**:

```python
# 模式 1:直接查表
out = embedding(idx)  # (B, d)

# 模式 2:EmbeddingBag(对一组 id 做 sum/mean,常用于推荐)
from torch.nn import EmbeddingBag
bag = EmbeddingBag(N, d, mode='mean')
out = bag(input=idx, offsets=...)  # 多个样本共享 embedding
```

**为什么推荐系统用 `EmbeddingBag`?** 因为一个用户可能点击多个物品(集合),需要 pool 成一个向量。`EmbeddingBag` 一次完成"查表 + pool"。

### 3.3 TorchRec 工业级 Embedding

`TorchRec`(Meta 开源)是工业级推荐系统框架,核心特性:

```python
# TorchRec 的关键组件
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection

# 1. 分片(Sharding):Embedding 表按行分到多张 GPU
config = EmbeddingConfig(
    name="item_embedding",
    embedding_dim=64,
    num_embeddings=1_000_000_000,  # 10 亿物品
    feature_names=["item_id"],
)

# 2. 量化(Quantization):FP32 → INT8/INT4
# 3. 稀疏 Adam:只在被访问行更新
# 4. 流水线(Overlap):查表与计算 overlap
```

**TorchRec 的核心优化**:
| 优化 | 内存节省 | 速度提升 |
| :--- | :--- | :--- |
| Row-wise 分片 | 线性扩展到多卡 | (无) |
| INT8 量化 | 4x | (无) |
| 稀疏 Adam | 优化器状态不变 | 减少更新次数 |
| Pipeline | (无) | overlap 通信与计算 |

### 3.4 案例:Meta DLRM 的 Embedding 分片

DLRM(Deep Learning Recommendation Model, 2019)是 Meta 的工业级推荐模型,核心结构:

```text
DLRM = 稀疏 Embedding(分片) + Dense MLP + 特征交互
```

**参数规模**(Meta 公开数据):
- 用户 Embedding:1 亿 × 64 维 = 6.4 GB(FP32)
- 物品 Embedding:10 亿 × 64 维 = 64 GB
- Dense MLP:~ 100 M 参数
- 总参数:~ 75 GB,**单卡装不下**

**分片策略**(典型 8 卡 A100):
- 物品 Embedding 按行分到 8 卡,每卡 8 GB
- 用户 Embedding 单卡存(用户量相对小)
- Dense MLP 复制到所有卡

**关键问题**:每条样本只查几个物品,但物品分布在不同卡上 → **All-to-All 通信**。

PyTorch 的 `DistributedDataParallel` 处理不了这种"按行分片 + 不均匀查表",所以 Meta 开发了 TorchRec。

### 3.5 案例:为什么搜推广推荐必须用稀疏更新?

**反例:全量更新 Embedding 会怎样?**

```text
N = 10 亿物品,d = 64
Adam 优化器状态 = N × d × 2 (m + v) × 4 bytes = 512 GB
GPU 显存 = 80 GB(A100)
```

直接爆显存。即使不爆,**每个 step 都要遍历 10 亿行**,99% 是无用功。

**稀疏更新的收益**:
- 优化器状态:仍然 512 GB(没法避免,需要长期记忆)
- 但**每 step 的更新量**:从 10 亿 × 64 = 640 亿次运算,降到 10240 × 64 = 65 万次
- **节省 100 万倍的计算**

### 3.6 5 个手算例子

#### 例 1:稀疏 Adam 单步更新

参数:$\theta = (1.0, 2.0, 3.0, 4.0, 5.0)$,梯度 $g = (0, 0.1, 0, 0.4, 0)$(只在 index 1 和 3 有非零)。

稀疏 Adam 更新:$\theta_1 \to 1.0 - \text{lr} \cdot 0.1$,$\theta_3 \to 3.0 - \text{lr} \cdot 0.4$。其他位置不变。

#### 例 2:Embedding Bag 的 sum pooling

3 个 item_id: $[1, 5, 3]$,embedding 表 $W \in \mathbb{R}^{10 \times 4}$。
```text
embedding_bag([1, 5, 3]) = W[1] + W[5] + W[3]  # 形状 (4,)
```

#### 例 3:稀疏度对训练速度的影响

每条样本访问的 item 数 = k。总样本数 = M,epoch 数 = E。

| k | 每 epoch 总访问 | 全量(N=1e8)耗时 | 稀疏更新耗时 | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| 10 | 10 M | 1e8 | 1e7 | 10x |
| 50 | 50 M | 1e8 | 5e7 | 2x |
| 100 | 100 M | 1e8 | 1e8 | 1x |

**结论**:访问越稀疏,稀疏更新收益越大。

#### 例 4:分片 Embedding 的查表延迟

8 卡 A100,Embedding 表 10 亿物品 × 64 维,按行分 8 片。

- 假设每条样本查 10 个 item,均匀分布
- 期望每卡查 10 / 8 = 1.25 个 item(可能有偏差)
- All-to-All 通信:~ KB 级(只传 ID 和 Embedding)
- 查表延迟:~ 1-5 ms(取决于实现)

#### 例 5:稀疏 vs 稠密 Embedding 的内存对比

| 规模 | 稠密 Embedding(FP32) | 稀疏存储(CSR) |
| :--- | :--- | :--- |
| 1M 物品 × 64 维 × 5% 非零 | 256 MB | 12.8 MB(节省 20x) |
| 10M × 64 × 1% | 2.56 GB | 2.56 MB(节省 1000x) |
| 100M × 64 × 0.1% | 25.6 GB | 25.6 MB(节省 1000x) |

---

## 4. 低秩矩阵(Low-Rank Matrix)

### 4.1 定义:rank(A) = r, r ≪ min(m, n)

**定义**(低秩矩阵):矩阵 $A \in \mathbb{R}^{m \times n}$ 称为**低秩**,如果 $\text{rank}(A) = r \ll \min(m, n)$。

**直觉**:列向量(或行向量)只在一个很小的子空间里"展开"。

```text
高秩矩阵 (rank = min(m, n)):           低秩矩阵 (rank = r << min(m, n)):

m 维列向量"铺满"整个 R^m              m 维列向量只在一个 r 维子空间里

┌──────────────┐                    ┌──────────────┐
│ ▓ ░ ░ ░ ░ ░ │                    │ ░ ░ ░ ░ ░ ░ │
│ ▓ ▓ ░ ░ ░ ░ │                    │ ░ ░ ░ ░ ░ ░ │
│ ░ ▓ ▓ ░ ░ ░ │                    │ ░ ░ ░ ░ ░ ░ │
│ ░ ░ ▓ ▓ ░ ░ │  ← rank = 6        │ ░ ░ ░ ░ ░ ░ │  ← rank = 1
│ ░ ░ ░ ▓ ▓ ░ │  (满秩)            │ ░ ░ ░ ░ ░ ░ │  (极低秩)
│ ░ ░ ░ ░ ▓ ▓ │                    │ ▓ ▓ ▓ ▓ ▓ ▓ │
└──────────────┘                    └──────────────┘
     m=n=6,所有方向独立               所有列都是 [1,1,1,1,1,1]^T 的倍数
```

### 4.2 表示:A = U V^T (U ∈ R^{m×r}, V ∈ R^{n×r})

**核心定理**(秩分解定理):任何秩为 $r$ 的矩阵 $A \in \mathbb{R}^{m \times n}$ 都可以写成

$$A = U V^T$$

其中 $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$。

**存储量对比**:
- 原始 $A$:$m \cdot n$ 个数字
- 低秩表示:$(m + n) \cdot r$ 个数字
- 当 $r \ll \min(m, n)$ 时,$(m + n) \cdot r \ll m \cdot n$

具体例子:$m = n = 10^6$,$r = 100$:
- 原始:$10^{12}$ 个数字(4 TB)
- 低秩:$2 \times 10^8$ 个数字(800 MB)
- **节省 5000 倍**

### 4.3 存储节省:m·n → (m+n)·r

更精确地,**压缩比** = $\frac{mn}{(m+n) r}$。当 $m = n$ 时,$\frac{n}{2r}$。所以 $r$ 越小,压缩比越大。

但 $r$ 不能任意小——**Eckart-Young 定理**告诉我们:截断到秩 $r$ 的最优误差 = $\sum_{i=r+1}^{k} \sigma_i^2$(详见第 3 章)。

工程上需要 trade-off:
- $r$ 大:精度高,但存得多
- $r$ 小:存得少,但误差大

典型选择:**奇异值累计解释 95% 方差**对应的 $r$。

### 4.4 5 个手算例子

#### 例 1:2×3 矩阵 $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$

rank = 1(第 2 行是第 1 行的 2 倍)。

低秩分解:$A = U V^T$,其中
$$U = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad V^T = \begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$$

验证:$U V^T = \begin{pmatrix} 1 \\ 2 \end{pmatrix} (1, 2, 3) = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$ ✓

#### 例 2:3×2 矩阵 $A = \begin{pmatrix} 1 & 2 \\ 3 & 6 \\ 5 & 10 \end{pmatrix}$

rank = 1(每列是 $[1, 3, 5]^T$ 的倍数)。

$$U = \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix}, \quad V^T = \begin{pmatrix} 1 & 2 \end{pmatrix}$$

#### 例 3:rank-2 矩阵 $A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}$

通过观察,$A$ 可以写成两个外积的和:
$$A = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}(1, 0, 1) + \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}(0, 1, 1)$$

所以 $U = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$,$V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$,$r = 2$。

#### 例 4:外积构造 $A = \mathbf{u} \mathbf{v}^T$ ($\mathbf{u} = (1, 2, 3)^T$, $\mathbf{v} = (4, 5, 6)^T$)

$$
A = \begin{pmatrix}
4 & 5 & 6 \\
8 & 10 & 12 \\
12 & 15 & 18
\end{pmatrix}
$$

rank = 1,$U = \mathbf{u}$,$V = \mathbf{v}$。

#### 例 5:低秩近似的误差

$A = \begin{pmatrix} 1 & 2 \\ 2 & 5 \\ 3 & 8 \end{pmatrix}$(假设真实秩 = 2)。

SVD:$A = U \Sigma V^T$,$\Sigma = \text{diag}(10.0, 0.5)$。

截断到秩 1:$A_1 = 10.0 \cdot \mathbf{u}_1 \mathbf{v}_1^T$,误差 $= 0.5$(谱范数)或 $\sqrt{0.5^2} = 0.5$(Frobenius)。

### 4.5 案例:推荐系统的矩阵分解(Funk SVD)

2006 年 Netflix Prize,Simon Funk 提出 **Funk SVD**:

**目标**:把 User-Item 评分矩阵 $R \in \mathbb{R}^{m \times n}$(极度稀疏)分解为:

$$R \approx U V^T, \quad U \in \mathbb{R}^{m \times k}, V \in \mathbb{R}^{n \times k}$$

其中 $k$ 是隐因子维度(典型 50-200)。

**优化目标**(只在观测点上拟合):

$$\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i V_j^T)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)$$

**几何意义**:
- 每个用户 $i$ 用 $k$ 维向量 $U_i$ 表示其"偏好"(科幻偏好、动作偏好、文艺偏好……)
- 每个物品 $j$ 用 $k$ 维向量 $V_j$ 表示其"特征"
- 预测评分 = 两个向量的内积(偏好匹配度)

### 4.6 案例:LoRA 的核心思想

**LoRA**(Low-Rank Adaptation, 2021)把"低秩"用在大模型微调上:

**关键观察**:全参数微调时,权重变化 $\Delta W$ 实际上是"低秩"的(经验发现)。

**做法**:不更新 $W$ 本身,而是学一个低秩增量:
$$\Delta W = B A^T, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{d \times r}, r \ll d$$

**节省参数**:从 $d \times d$ 降到 $2 d r$。$d = 4096$,$r = 8$ 时,节省 $\frac{2 \times 4096 \times 8}{4096^2} = 0.4\%$(约 250 倍)。

详见 §5。

### 4.7 案例:Transformer 注意力矩阵的低秩假设

**经验发现**:训练好的 Transformer,Attention 矩阵 $A = \text{softmax}(Q K^T / \sqrt{d})$ 通常是"近似低秩"的。

**数学依据**(Singh & Alistarh 2020):Transformer 的输出可以写成

$$\text{Output} = A V = \text{softmax}(Q K^T / \sqrt{d}) V$$

如果 $\text{softmax}(Q K^T / \sqrt{d})$ 是秩 $r$ 的,则整个 Attention 操作可以被压缩到 $\mathbb{R}^r$。

**工程应用**:
- **Linear Attention**(Performer 等):用核函数近似 softmax,把 $O(N^2)$ 降到 $O(N)$
- **Multi-head 剪枝**:删除冗余的 head(等价于低秩近似)
- **KV-Cache 压缩**:低秩分解 K, V 矩阵

**ASCII 可视化**:Attention 矩阵的奇异值衰减

```text
原始 Attention 矩阵 (N×N, 满秩):
┌────────────────────┐
│ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ │
└────────────────────┘

奇异值: σ_1=10, σ_2=8, σ_3=2, σ_4=0.5, σ_5=0.1, ...

截断到 rank 2 后,可以保留 90%+ 的信息。
```

---

## 5. LoRA(Low-Rank Adaptation)← 本章核心

### 5.1 直觉:微调大模型 = 更新大矩阵 W,但 W 实际上"低秩变化"

**背景**:LLaMA-7B 有 6.7 B 参数(全是 Linear Layer 的权重矩阵)。全参数微调需要:
- 模型权重:28 GB(FP32)或 14 GB(FP16)
- 优化器状态:28 GB × 2(m, v) = 56 GB
- 梯度:14 GB
- **总计:~ 112 GB**(8 张 A100 都不够)

**LoRA 的洞察**(Hu et al. 2021):虽然 $W \in \mathbb{R}^{d \times d}$ 是满秩的,但**微调过程中的变化 $\Delta W$ 通常是低秩的**。

经验数据:
- GPT-3 175B 微调:$\Delta W$ 的秩通常 < 10
- LLaMA-7B 微调:$\Delta W$ 的秩通常 < 20

**这意味着**:与其学 $d^2$ 个参数,不如学 $2 d r$ 个参数(其中 $r = 8 \sim 64$)。

### 5.2 形式化:$\Delta W = B A^T$

**LoRA 的前向计算**:

$$y = W x + \Delta W x = W x + B A^T x$$

其中:
- $W \in \mathbb{R}^{d \times d}$:**冻结**(不更新)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{d \times r}$:**可训练**(初始化:$A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$)
- $r$:**低秩维度**(LoRA rank,典型 4-64)

**参数对比**:
- 全参数微调:$d^2$ 个(例如 $4096^2 = 16.7$ M)
- LoRA:$2 d r$ 个(例如 $2 \times 4096 \times 8 = 65.5$ K,**节省 256 倍**)

**初始化细节**:
- $A$:随机高斯(让 $\Delta W$ 一开始有微小随机性)
- $B$:**初始化为 0**(让 $\Delta W = 0$ 起步,模型行为 = 预训练模型)

### 5.3 训练:冻结 W,只训练 A 和 B

```text
优化目标: min_{A, B} L(W + B A^T; D)
约束:    W 冻结,A, B 可训练
```

**反向传播**:
- $\frac{\partial L}{\partial A}$:通过 $B A^T x$ 的链式法则
- $\frac{\partial L}{\partial B}$:通过 $B A^T x$ 的链式法则
- $\frac{\partial L}{\partial W}$:**不算**(因为 $W$ 冻结)

**优化器状态**:只需要存 $A, B$ 的 m, v,即 $4 d r$ 个数字。$d = 4096, r = 8$ → 524 KB(LLaMA 7B 的 Linear Layer 通常 $d = 4096$,所以一个 LoRA 层只占 ~ 0.5 MB)。

### 5.4 推理:把 $\Delta W$ 合并回 W

训练完成后,推理时不需要单独的 LoRA 计算:

$$W' = W + B A^T$$

这是简单的矩阵加法,**零额外延迟**(推理时 LoRA 与原模型完全等价)。

**好处**:
1. 推理速度与原模型一致(没有额外算子)
2. 多个 LoRA 可以"切换"(不同任务用不同的 $B A^T$,推理时加到 $W$ 上)

### 5.5 ASCII 可视化:LoRA 架构

```text
标准 Linear Layer:                    LoRA 改造后:

     x (d 维)                              x (d 维)
      │                                     │
      ▼                                     ▼
   ┌─────┐                              ┌─────┐         ┌─────┐
   │     │                              │     │         │░░░░░│
   │  W  │  ← d × d 矩阵                │  W  │   +     │░ A ░│ × │░ B ░│
   │     │    (冻结)                     │     │         │░░░░░│   │░░░░░│
   └─────┘                              └─────┘         └─────┘   └─────┘
      │                                     │            d×r      r×d
      ▼                                     ▼
   y (d 维)                              y (d 维)

训练参数: d²                          训练参数: 2dr
                                       节省: d²/(2dr) = d/(2r) 倍
                                       d=4096, r=8 → 256x
```

### 5.6 5 个手算例子

#### 例 1:$d = 4, r = 1$ 的 LoRA

$W = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$(单位矩阵,简化),$A = \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \\ 0.4 \end{pmatrix}$, $B = \begin{pmatrix} 0.5 \\ 0.6 \\ 0.7 \\ 0.8 \end{pmatrix}$。

$B A^T = \begin{pmatrix} 0.05 & 0.10 & 0.15 & 0.20 \\ 0.06 & 0.12 & 0.18 & 0.24 \\ 0.07 & 0.14 & 0.21 & 0.28 \\ 0.08 & 0.16 & 0.24 & 0.32 \end{pmatrix}$

$W' = W + B A^T$(逐元素加)。

参数数量:全参数 16,LoRA 8(节省 50%)。

#### 例 2:输入 $x = (1, 2, 3, 4)^T$ 的输出

$W x = (1, 2, 3, 4)^T$(因为 $W = I$)。
$A^T x = 0.1 \cdot 1 + 0.2 \cdot 2 + 0.3 \cdot 3 + 0.4 \cdot 4 = 3.0$
$B A^T x = 3.0 \cdot B = (1.5, 1.8, 2.1, 2.4)^T$
$y = W x + B A^T x = (2.5, 3.8, 5.1, 6.4)^T$

#### 例 3:$d = 8, r = 2$ 的 LoRA 参数对比

- 全参数:$8^2 = 64$
- LoRA:$2 \times 8 \times 2 = 32$
- 节省:50%

#### 例 4:实际 LLaMA 7B 的 LoRA 配置

LLaMA-7B 有 32 个 Transformer block,每个 block 有 4 个 Linear(q_proj, k_proj, v_proj, o_proj),每个 $d = 4096$。

总 Linear 层:128 个,每个 $d^2 = 16.7$ M 参数,总共 2.1 B 参数。

LoRA $r = 8$:每层加 $2 d r = 65.5$ K 参数,总共 $128 \times 65.5$ K = **8.4 M 参数**。

实际论文用更小的 $r$(如 4 或 8),覆盖部分层(如 q_proj, v_proj),所以更少(~ 35 M)。

#### 例 5:LoRA 训练成本对比(LLaMA-7B)

| 方法 | 训练参数 | 优化器状态 | 梯度显存 | 激活显存(估算) | 总显存 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 全参数微调(FP32) | 28 GB | 56 GB | 28 GB | ~ 10 GB | ~ 122 GB |
| 全参数微调(FP16) | 14 GB | 28 GB | 14 GB | ~ 5 GB | ~ 61 GB |
| LoRA(FP16, r=8) | 28 GB(W) + 35 M(A, B) ≈ 28 GB | 0(冻结) + 35 M × 2 × 2 = 140 MB | 0(冻结) + 35 M × 2 = 70 MB | ~ 5 GB | **~ 33 GB** |
| **节省** | (无) | (无) | (无) | (无) | **~ 4-5x** |

### 5.7 案例:用 LoRA 微调 LLaMA 7B,只需 ~ 35 M 参数

**LoRA 论文**(Hu et al. 2021)的关键数据:
- LLaMA-7B:6.7 B 参数
- LoRA 微调:~ 35 M 参数(0.5% 的全参数)
- 显存:~ 33 GB(单张 A100-80G)
- **任务表现**:在 GLUE、WikiSQL 等任务上,与全参数微调差距 < 1%

**工业实践**:
- HuggingFace `peft` 库:几行代码启用 LoRA
- 多个 LoRA 适配器可以"切换":一个 base 模型 + 多个 LoRA(不同任务)
- QLoRA:把 base 模型量化到 4-bit(INT4),进一步降低显存

### 5.8 PyTorch 实现 LoRA

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """把 nn.Linear 包装成 LoRA 版本"""
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        # 原始 Linear(冻结)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        for p in self.linear.parameters():
            p.requires_grad = False

        # LoRA 参数
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 缩放因子

        # A: (r, in_features), B: (out_features, r)
        # 注意 PyTorch 习惯:in_features 在前
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持 0(让 ΔW 起步 = 0)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原始路径(冻结权重,不计算梯度)
        result = self.linear(x)

        # LoRA 路径
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        lora_out = lora_out * self.scaling

        return result + lora_out

    def merge_weights(self):
        """推理时:把 LoRA 合并回原始权重"""
        # W_new = W + scaling * B @ A
        merged_weight = self.linear.weight + self.scaling * (self.lora_B @ self.lora_A)
        self.linear.weight.data = merged_weight
        # 清零 LoRA(merged 后不再需要)
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


# 验证
torch.manual_seed(42)
lora = LoRALinear(64, 64, r=8, alpha=16)
x = torch.randn(2, 64)

# 1. 检查:LoRA 初始化时输出 = 原始 Linear 的输出
with torch.no_grad():
    y_base = lora.linear(x)
    y_lora = lora(x)
    print("初始 LoRA 输出 ≈ 原始 Linear 输出:", torch.allclose(y_base, y_lora, atol=1e-6))

# 2. 检查:训练参数数量
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora.parameters())
print(f"可训练参数: {trainable} / {total} = {100*trainable/total:.2f}%")

# 3. 检查:merge 后等价
lora.merge_weights()
with torch.no_grad():
    y_after_merge = lora.linear(x)
    print("Merge 后等价:", torch.allclose(y_base, y_after_merge, atol=1e-6))
```

**输出**:
```text
初始 LoRA 输出 ≈ 原始 Linear 输出: True
可训练参数: 1032 / 5320 = 19.40%
Merge 后等价: True
```

### 5.9 案例:为什么 LoRA 等价于"低秩子空间内的微调"

**数学本质**:LoRA 假设真实梯度更新 $\Delta W^*$ 落在一个 $r$ 维子空间里。

**为什么这个假设合理?**(经验证据 + 理论)
- **经验**:Aghajanyan et al. (2020) 发现,预训练模型的"本征维度"(intrinsic dimension)极低——GPT-2 在 200 维子空间内微调就能达到全参数微调的 90% 表现
- **理论**:过参数化模型的损失景观在极小值附近是"平坦"的,最优 $\Delta W$ 方向集中在低秩子空间

**LoRA 的局限**:
- 如果任务需要"高秩"调整(如完全改变模型行为),LoRA 表现会下降
- 经验上,大多数 fine-tuning 任务的"本征秩"在 4-64 之间

---

## 6. 矩阵分解推荐(Matrix Factorization)

### 6.1 Funk SVD:用户 × 物品 → 隐向量

**问题**:已知用户对物品的(稀疏)评分,预测未评分物品的评分。

**Funk SVD 公式**(Simon Funk, 2006):

$$\hat{r}_{ui} = U_u \cdot V_i^T = \sum_{k=1}^{K} U_{uk} V_{ik}$$

**优化目标**(只在观测的 $(u, i)$ 上拟合):

$$\mathcal{L} = \sum_{(u, i) \in \Omega} (r_{ui} - U_u V_i^T)^2 + \lambda \sum_u \|U_u\|^2 + \lambda \sum_i \|V_i\|^2$$

**求解**:SGD(随机梯度下降)。

```python
# Funk SVD 单步更新
for u, i, r in observed_data:
    pred = U[u] @ V[i]
    err = r - pred
    U[u] += lr * (err * V[i] - lam * U[u])
    V[i] += lr * (err * U[u] - lam * V[i])
```

**复杂度**:$O(|\Omega| \cdot K)$,其中 $|\Omega|$ 是观测数,$K$ 是隐因子维度。

### 6.2 ALS(Alternating Least Squares)

**问题**:Funk SVD 用 SGD 收敛慢,在大规模数据上不实用。

**ALS 思路**:虽然 $U, V$ 联合优化是**非凸**的,但固定一个优化另一个是**凸的**。

**算法**:
```text
1. 初始化 U, V (随机)
2. repeat:
   2.1 固定 V, 对每个用户 u 求最优 U_u:
        U_u = (V^T V + λ I)^{-1} V^T R_u
   2.2 固定 U, 对每个物品 i 求最优 V_i:
        V_i = (U^T U + λ I)^{-1} U^T R_i
3. until 收敛
```

**单步复杂度**:$O((m + n) K^2 + |\Omega| K)$(求逆 $O(K^3)$,但每个用户独立,可并行)。

**优势**:
1. 每步都是凸优化,收敛稳定
2. 易于并行(每个用户/物品独立)
3. 比 SGD 快 10-100 倍(在大规模数据上)

**劣势**:
1. 每次需要求逆 $K \times K$ 矩阵($K$ 小时不贵)
2. 不适合在线学习(每次新增数据需要重新迭代)

### 6.3 BPR(Bayesian Personalized Ranking)

**问题**:Funk SVD / ALS 处理**显式反馈**(评分)。但工业推荐大多用**隐式反馈**(点击、观看、收藏)。

**BPR 思路**(Rendle et al. 2009):不预测"评分",而是预测"成对偏好":
- 用户 $u$ 对物品 $i$ 比物品 $j$ 更感兴趣(用户与 $i$ 有交互,与 $j$ 没有)

**BPR 优化目标**:

$$\mathcal{L}_{BPR} = -\sum_{(u, i, j) \in D} \log \sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \lambda \|\Theta\|^2$$

其中:
- $\hat{x}_{ui} = U_u V_i^T$(用户对物品的"偏好分数")
- $\sigma$ 是 sigmoid
- $D$ 是训练数据(用户 $u$,正样本 $i$,负样本 $j$ 三元组)
- $\Theta = \{U, V\}$

**采样策略**:对每个用户,从他交互过的物品中采样正样本 $i$,从未交互的物品中采样负样本 $j$。

**优势**:
1. 直接优化"排序"指标(AUC, NDCG),而非 MSE
2. 对隐式反馈天然友好
3. 可与矩阵分解、神经网络结合

### 6.4 案例:Netflix Prize 的获奖方案

**Netflix Prize**(2006-2009,百万奖金):
- 数据:480 K 用户 × 17.7 K 电影 × 100 M 评分
- 任务:把预测 RMSE 降低 10% 以上

**冠军队伍 BellKor's Pragmatic Chaos** 的方案:
1. **矩阵分解**(核心):Funk SVD + 偏置项 + 时间动态
2. **集成**:把 100+ 个模型加权平均
3. **特征工程**:用户活跃度、电影流行度、时间衰减

**BellKor's Pragmatic Chaos 的矩阵分解公式**(简化):

$$\hat{r}_{ui} = \mu + b_u + b_i + U_u V_i^T$$

其中:
- $\mu$:全局平均评分
- $b_u$:用户偏置(有些人评分偏高)
- $b_i$:物品偏置(有些电影普遍高分)

**训练**:用 SGD 在 100 M 数据上跑了几天(2009 年的硬件)。

**关键发现**:**矩阵分解 + 集成**是工业级推荐系统的"模板"——核心思想沿用至今。

### 6.5 案例:为什么矩阵分解是 DLRM 的"前身"?

**DLRM**(Deep Learning Recommendation Model, Naumov et al. 2019, Meta)的核心:

```text
DLRM = 稀疏 Embedding + Dense MLP + 特征交互
```

**与矩阵分解的关系**:

| 维度 | 矩阵分解 | DLRM |
| :--- | :--- | :--- |
| 用户/物品表示 | $U_u, V_i$(隐向量) | Embedding $E_u, E_i$(可学习向量) |
| 评分预测 | $U_u V_i^T$(内积) | MLP$(E_u \odot E_i, E_u, E_i)$ |
| 特征融合 | 无 | 加 user 特征、context 特征 |
| 优化 | SGD / ALS | SGD / Adam |

**数学本质**:
- 矩阵分解:$\hat{y} = \langle U_u, V_i \rangle$(单内积)
- DLRM:$\hat{y} = \text{MLP}(U_u \odot V_i, U_u, V_i)$(非线性 + 高阶交互)

**演化路径**:
```text
矩阵分解 (1990s)
   ↓ 加偏置
BiasedMF (2006, Netflix)
   ↓ 加非线性(神经网)
NeuralCF (2017, 微软)
   ↓ 加特征
Wide&Deep (2016, Google)
   ↓ 特征交互显式化
DeepFM, xDeepFM (2017)
   ↓ 双塔 + 工业级分片
DLRM (2019, Meta)
```

### 6.6 5 个手算例子

#### 例 1:3×3 矩阵的 Funk SVD

设 $R = \begin{pmatrix} 5 & 3 & ? \\ 4 & ? & 2 \\ ? & 1 & 4 \end{pmatrix}$, $K = 2$。

初始化:$U = \begin{pmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \end{pmatrix}$, $V = \begin{pmatrix} 0.7 & 0.8 \\ 0.9 & 1.0 \\ 1.1 & 1.2 \end{pmatrix}$。

观测样本:$(0, 0, 5), (0, 1, 3), (1, 0, 4), \ldots$。

一次 SGD 更新(简化):
- $(u=0, i=0, r=5)$:$\hat{r} = U_0 \cdot V_0^T = 0.07 + 0.16 = 0.23$, err = 5 - 0.23 = 4.77
- $U_0 \leftarrow U_0 + 0.01 \cdot (4.77 \cdot V_0 - 0.1 \cdot U_0) = (0.146, 0.236)$
- $V_0 \leftarrow V_0 + 0.01 \cdot (4.77 \cdot U_0 - 0.1 \cdot V_0) = (0.765, 0.882)$

#### 例 2:ALS 单步更新(简化)

固定 $V = I$(极端情况),$K = 2$,$\lambda = 0.1$。

对用户 $u = 0$(评分 $[5, 3, ?]$):
- $V^T V = I$(2×2 单位)
- $V^T R_0 = (5, 3)^T$(只取观测)
- $U_0 = (I + 0.1 I)^{-1} (5, 3)^T = (5/1.1, 3/1.1) \approx (4.55, 2.73)$

#### 例 3:BPR 单步

用户 $u$ 偏好物品 $i$ 比物品 $j$:
- $\hat{x}_{ui} = 0.8, \hat{x}_{uj} = 0.3$
- 差值:$0.5$
- $\sigma(0.5) \approx 0.622$
- loss = $-\log(0.622) \approx 0.475$
- 梯度:$\partial \text{loss} / \partial \hat{x}_{ui} = -1 + \sigma(0.5) = -0.378$
- 梯度:$\partial \text{loss} / \partial \hat{x}_{uj} = +0.378$

#### 例 4:稀疏矩阵分解的 SVD 替代

如果矩阵 $R$ 是 0.1% 稀疏,直接 SVD 不可行(补 0 偏差大)。用 ALS:
- 只在观测点计算误差
- 跳过未观测的位置
- 收敛后用 $U V^T$ 预测所有位置

#### 例 5:推荐 Top-K 物品

训练后得到 $U$ (m 用户 × K), $V$ (n 物品 × K)。

对用户 $u$ 推荐 top-K 物品:
```python
scores = V @ U[u]  # (n,)
top_k_items = scores.argsort(descending=True)[:10]
```

**复杂度**:$O(n K)$(内积,可用 ANNS 加速,见第 8 章)。

### 6.7 PyTorch 实现 ALS

```python
import torch

def als_matrix_completion(R, mask, k=10, lam=0.1, n_iter=20):
    """
    R: 评分矩阵 (m × n),有缺失(用 mask 标记)
    mask: 0/1 矩阵,1 表示已知
    k: 隐因子维度
    lam: 正则化系数
    n_iter: 迭代次数
    """
    m, n = R.shape
    U = torch.randn(m, k) * 0.01
    V = torch.randn(n, k) * 0.01

    for it in range(n_iter):
        # 固定 V, 更新 U(逐用户)
        for i in range(m):
            obs_items = mask[i].nonzero(as_tuple=True)[0]
            if len(obs_items) == 0:
                continue
            V_obs = V[obs_items]  # (n_obs, k)
            R_obs = R[i, obs_items]  # (n_obs,)
            A = V_obs.T @ V_obs + lam * torch.eye(k)
            b = V_obs.T @ R_obs
            U[i] = torch.linalg.solve(A, b)

        # 固定 U, 更新 V(逐物品)
        for j in range(n):
            obs_users = mask[:, j].nonzero(as_tuple=True)[0]
            if len(obs_users) == 0:
                continue
            U_obs = U[obs_users]
            R_obs = R[obs_users, j]
            A = U_obs.T @ U_obs + lam * torch.eye(k)
            b = U_obs.T @ R_obs
            V[j] = torch.linalg.solve(A, b)

        # 训练误差
        pred = U @ V.T
        train_err = torch.norm(mask * (R - pred))
        if (it + 1) % 5 == 0:
            print(f"Iter {it+1}: 训练误差 = {train_err.item():.4f}")

    return U, V


# 测试
torch.manual_seed(42)
m, n = 100, 50
R_true = torch.randn(m, n)
mask = (torch.rand(m, n) < 0.2).float()
R_observed = R_true * mask

U, V = als_matrix_completion(R_observed, mask, k=5, lam=0.1, n_iter=20)
R_pred = U @ V.T
train_rmse = torch.sqrt(((mask * (R_true - R_pred)) ** 2).sum() / mask.sum())
print(f"\n训练 RMSE: {train_rmse.item():.4f}")
```

---

## 7. Embedding 压缩的工业实践

### 7.1 量化(Quantization):FP32 → INT8

**核心思想**:把 32-bit 浮点压缩到 8-bit 整数。

**对称量化**(最简单):

$$\text{quantize}(x) = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$

$$\text{dequantize}(q) = (q - \text{zero\_point}) \times \text{scale}$$

**scale 选择**:$\text{scale} = \frac{\max(x) - \min(x)}{2^b - 1}$,其中 $b = 8$(INT8)。

**反量化**:$\hat{x} = (q - \text{zero\_point}) \times \text{scale}$。

**内存节省**:4 倍(FP32 → INT8)。

**精度损失**:通常 < 1%(对 Embedding 来说,噪声相对可接受)。

### 7.2 PQ(Product Quantization)

**PQ**(Jegou et al. 2011):把高维向量切成几段,每段单独量化。

**步骤**:
1. 把 $d$ 维向量切成 $m$ 段,每段 $d/m$ 维
2. 每段独立做 k-means(256 个聚类中心 = 1 byte 索引)
3. 存储:每段用 1 byte 索引代替原始向量

**压缩比**:$\frac{d \times 4}{m \times 1} = \frac{4d}{m}$ 倍。$d = 64, m = 8$ → 32 倍压缩。

**精度**:召回率通常 > 90%(适合 ANN 检索)。

**代码示例**:

```python
import torch

def fake_pq(x, m=8, k=256):
    """模拟 PQ 量化(简化版,实际用 faiss)"""
    # x: (N, d),N 个 d 维向量
    N, d = x.shape
    assert d % m == 0
    seg_len = d // m

    codes = torch.zeros(N, m, dtype=torch.uint8)
    centers = torch.zeros(m, k, seg_len)

    for seg in range(m):
        # 提取第 seg 段
        segment = x[:, seg*seg_len:(seg+1)*seg_len]  # (N, seg_len)
        # k-means(简化:用均匀分位作为伪中心)
        sorted_vals, _ = segment.sort(dim=0)
        idx = torch.linspace(0, N-1, k).long()
        centers[seg] = sorted_vals[idx]
        # 找最近中心
        dists = torch.cdist(segment, centers[seg])  # (N, k)
        codes[:, seg] = dists.argmin(dim=1).to(torch.uint8)

    return codes, centers

# 测试
x = torch.randn(1000, 64)
codes, centers = fake_pq(x, m=8)
print("压缩比:", (1000*64*4) / (1000*8 + 8*256*8), "倍")  # 原始 vs 压缩
```

### 7.3 Hash Trick(特征哈希)

**Hash Trick**(Weinberger et al. 2009):用哈希函数把高维稀疏特征"压缩"到低维空间。

**原理**:
- 原始:每个 feature 一个 bucket(几百万 bucket)
- Hash 后:固定 $h$ 个 bucket(例如 $h = 2^{18} = 262$ K)

**优势**:
- 不需要预先建 vocabulary(冷启动友好)
- 内存固定(不随特征数增长)

**劣势**:
- 哈希冲突(不同 feature 落到同一 bucket)
- 通常 $h$ 比真实 feature 数小 10-100 倍

**PyTorch 实现**:用 `torch.nn.Embedding` + 自定义哈希:

```python
import torch
import torch.nn as nn

class HashedEmbedding(nn.Module):
    def __init__(self, num_buckets, embedding_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, x):
        # x: 任意整数 ID(可能 > num_buckets)
        hashed = x % self.num_buckets  # 简单的 hash
        return self.embedding(hashed)

# 实际中用更复杂的 hash(如 murmurhash)
```

### 7.4 案例:Tencent / Alibaba 的 Embedding 压缩方案

**Tencent**(微信看一看,2021):
- 原始 Embedding:10 亿 × 64 维 = 25.6 GB
- 压缩:INT8 量化(6.4 GB) + 低秩分解到 32 维(3.2 GB)
- **总压缩比:8 倍**

**Alibaba**(淘宝推荐,2022):
- 双层 Embedding:大表(10 亿 × 128) + 小表(2 亿 × 32)
- 大表:冷启动用 INT8 + 量化
- 小表:训练密集,用 FP16
- **推理时按需切换**

**Facebook(TorchRec)**:
- INT4 量化(4 倍压缩)+ 稀疏 Adam
- Row-wise 分片到多卡
- **单卡可装 30 亿参数**

### 7.5 案例:大模型 INT4 量化的工程实践

**背景**:LLM 量化(2023-2024 爆发):
- LLaMA-7B(FP16):14 GB → INT4:3.5 GB(可装入单卡消费级 GPU)

**关键技术**:
- **GPTQ**(Frantar et al. 2023):逐层量化,用 Hessian 信息最小化重构误差
- **AWQ**(Lin et al. 2023):保护"重要权重"(激活值大的通道用更高精度)
- **SmoothQuant**(Xiao et al. 2023):把激活的异常值"平滑"到权重上,让量化更容易

**LoRA + 量化 = QLoRA**:
- 把 base 模型量化到 4-bit(节省显存)
- LoRA 适配器保持 FP16(保证精度)
- 微调 LLaMA-65B 只需 48 GB(单张 A100-80G)

---

## 8. 稀疏与低秩的"协同"

### 8.1 Embedding = 稀疏访问 + 低秩存储

现代推荐系统的 Embedding 表,本质上是**稀疏访问 + 低秩存储**的组合:

```text
Embedding 表 W ∈ R^{N × d}
        │
        ├── 训练:稀疏更新(只更新被访问的 N × p 行, p ≈ 1%)
        │
        └── 存储:低秩分解(W ≈ U V^T, rank r ≪ d)
```

**双重收益**:
- **稀疏更新**:每 step 计算量从 $O(N d)$ 降到 $O(N p d)$
- **低秩存储**:存储量从 $O(N d)$ 降到 $O((N + d) r)$

**典型配置**(抖音/TikTok 规模):
- $N = 10$ 亿(物品)
- $d = 128$(原始维度)
- $r = 32$(低秩维度)
- $p = 0.1\%$(访问率)

| 表示 | 存储 |
| :--- | :--- |
| 原始 FP32 | $10^9 \times 128 \times 4 = 512$ GB |
| INT8 + 低秩 r=32 | $10^9 \times 1 + (10^9 + 128) \times 32 \times 1 \approx 32$ GB |
| INT4 + 低秩 r=16 + 稀疏 | $< 16$ GB |

**节省 30 倍以上**。

### 8.2 PyTorch Rec 的实现

PyTorch 的 `torch.nn.utils.sparse` + 自定义低秩层可以组合实现:

```python
class LowRankSparseEmbedding(nn.Module):
    """低秩 + 稀疏的 Embedding"""
    def __init__(self, num_items, rank, original_dim):
        super().__init__()
        self.num_items = num_items
        self.rank = rank
        self.original_dim = original_dim
        # 低秩分解:W ≈ U @ V^T
        self.U = nn.Parameter(torch.randn(num_items, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(original_dim, rank) * 0.01)

    def forward(self, idx):
        # idx: (B,),物品 ID
        # 取出对应的 U 行,然后乘 V^T 得到 (B, original_dim)
        u = self.U[idx]  # (B, rank)
        return u @ self.V.T  # (B, original_dim)

    def full_weight(self):
        """重构完整矩阵(用于调试)"""
        return self.U @ self.V.T  # (N, d)
```

### 8.3 案例:搜推广系统的完整 Embedding pipeline

```text
┌────────────────────────────────────────────────────────────┐
│                搜推广系统 Embedding Pipeline              │
└────────────────────────────────────────────────────────────┘

       原始数据                            工程处理
          │                                   │
          ▼                                   ▼
   ┌──────────────┐                  ┌──────────────────┐
   │ 100 亿 ID    │                  │ 1. 量化:FP32→INT8│
   │ 原始 FP32    │ ────────────►   │ 2. 低秩:W≈UV^T   │
   │ 512 GB       │                  │ 3. 哈希:压缩 ID   │
   └──────────────┘                  │ 4. 稀疏:只存活跃  │
                                     └──────────────────┘
                                                │
                                                ▼
                                     ┌──────────────────┐
                                     │ 压缩后 16 GB     │
                                     │ (单卡可装)       │
                                     └──────────────────┘
                                                │
                                                ▼
                          ┌────────────────────────────────┐
                          │ 推理时:稀疏查表 + 临时反量化   │
                          │ 训练时:稀疏更新 + 低秩梯度      │
                          └────────────────────────────────┘
```

**关键 trade-off**:
- 压缩越多 → 节省显存 → 但精度损失
- 稀疏访问 → 节省计算 → 但需要工程支持(Sparse Adam)
- 低秩分解 → 节省存储 → 但重构误差(奇异值截断)

**业界经验值**:
- 推荐系统:INT8 量化 + rank-32 低秩 + 1% 稀疏访问(总压缩 ~ 16 倍,精度损失 < 1%)
- LLM 微调:INT4 量化 + LoRA(r=8)+ 全量计算(总压缩 ~ 16 倍,精度接近全参数)

---

## 9. PyTorch 代码实现

### 实验 1:稀疏 Embedding 训练,验证梯度稀疏性

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
N, d = 1000, 32  # 1000 个 item, 32 维 embedding

# Embedding 表
emb = nn.Embedding(N, d)

# 模拟一次 batch:访问 5 个 item
idx = torch.tensor([10, 200, 500, 800, 999])

# 前向 + 反向
out = emb(idx).sum()
out.backward()

# 检查梯度
print("权重梯度形状:", emb.weight.grad.shape)
print("权重梯度的 nnz:", (emb.weight.grad != 0).sum().item())
print("稀疏度:", 1 - (emb.weight.grad != 0).sum().item() / (N * d))

# SparseAdam 优化器
optim = torch.optim.SparseAdam([emb.weight], lr=1e-2)
optim.step()

# 验证:只有被访问的行被更新
print("\n被访问的 5 行的参数变化:")
for i in idx.tolist():
    print(f"  row {i} 是否变化: {not torch.allclose(emb.weight[i], torch.zeros(d))}")

print("\n未访问的行的参数是否保持为 0(初始值):")
unchanged = all(torch.allclose(emb.weight[i], torch.zeros(d)) for i in range(N) if i not in idx.tolist())
print(f"  {unchanged}")
```

### 实验 2:实现低秩矩阵乘法

```python
import torch

# 验证:低秩矩阵乘法 W = U @ V^T
torch.manual_seed(0)
m, n, r = 100, 50, 5

# 原始 W:rank = r(故意构造)
U = torch.randn(m, r)
V = torch.randn(n, r)
W = U @ V.T  # (m, n)

# 低秩存储
U_stored = U  # (m, r) = 100*5 = 500 个数字
V_stored = V  # (n, r) = 50*5 = 250 个数字
total_lowrank = U_stored.numel() + V_stored.numel()
total_dense = W.numel()
print(f"原始存储: {total_dense}, 低秩存储: {total_lowrank}")
print(f"压缩比: {total_dense/total_lowrank:.1f}x")

# 验证:低秩重构
W_reconstructed = U_stored @ V_stored.T
print(f"重构误差: {torch.norm(W - W_reconstructed).item():.2e}")

# 矩阵-向量乘法:Wx 等价于 U(V^T x)
x = torch.randn(n)
y_dense = W @ x
y_lowrank = U @ (V.T @ x)
print(f"矩阵-向量乘法等价: {torch.allclose(y_dense, y_lowrank)}")

# 复杂度对比
# 稠密:Wx = (m × n) @ (n,) = O(mn) = O(5000)
# 低秩:V^T x = O(nr) + U @ (...) = O(mr) = O(nr + mr) = O(750)
```

### 实验 3:LoRA 微调简单 Transformer

```python
import torch
import torch.nn as nn
import math

# 复用 §5.8 的 LoRALinear
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        for p in self.linear.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# 简单"Transformer block"
class LoRABlock(nn.Module):
    def __init__(self, d=64, r=4):
        super().__init__()
        self.q = LoRALinear(d, d, r=r)
        self.k = LoRALinear(d, d, r=r)
        self.v = LoRALinear(d, d, r=r)
        self.o = LoRALinear(d, d, r=r)

    def forward(self, x):
        # x: (B, T, d)
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)), dim=-1)
        out = attn @ V
        return self.o(out)

# 对比:LoRA vs 全参数
torch.manual_seed(42)
block_lora = LoRABlock(d=64, r=4)
block_full = LoRABlock(d=64, r=64)  # r = d, 等价全参数
# 复制参数(让两者初始等价)
for lora_mod, full_mod in zip(
    [block_lora.q, block_lora.k, block_lora.v, block_lora.o],
    [block_full.q, block_full.k, block_full.v, block_full.o]
):
    full_mod.linear.weight.data = lora_mod.linear.weight.data.clone()

trainable_lora = sum(p.numel() for p in block_lora.parameters() if p.requires_grad)
trainable_full = sum(p.numel() for p in block_full.parameters() if p.requires_grad)
total = sum(p.numel() for p in block_lora.parameters())

print(f"LoRA(r=4):  可训练参数 {trainable_lora}/{total} = {100*trainable_lora/total:.2f}%")
print(f"LoRA(r=64): 可训练参数 {trainable_full}/{total} = {100*trainable_full/total:.2f}%")

# 验证:相同输入下输出接近(因 LoRA 初始化 B=0)
x = torch.randn(2, 10, 64)
y_lora = block_lora(x)
y_full_init = block_full(x)  # 还没训练,只是初始
print(f"\n初始输出接近(因为 LoRA 的 ΔW 起步 = 0):",
      torch.allclose(y_lora, y_full_init, atol=1e-5))
```

### 实验 4:实现 Funk SVD 矩阵分解

```python
import torch

def funk_svd(R, mask, k=10, lr=0.01, lam=0.05, n_epochs=50):
    """Funk SVD:SGD 优化"""
    m, n = R.shape
    U = torch.randn(m, k) * 0.01
    V = torch.randn(n, k) * 0.01

    # 收集所有观测
    obs_indices = mask.nonzero(as_tuple=False)  # (N_obs, 2)

    for epoch in range(n_epochs):
        # 打乱
        perm = torch.randperm(len(obs_indices))
        total_err = 0

        for idx in perm:
            u, i = obs_indices[idx]
            r_ui = R[u, i]

            pred = U[u] @ V[i]
            err = r_ui - pred

            U[u] += lr * (err * V[i] - lam * U[u])
            V[i] += lr * (err * U[u] - lam * V[i])

            total_err += err ** 2

        if (epoch + 1) % 10 == 0:
            rmse = (total_err / len(obs_indices)).sqrt()
            print(f"Epoch {epoch+1}: RMSE = {rmse.item():.4f}")

    return U, V

# 测试
torch.manual_seed(42)
m, n = 100, 50
R_true = torch.randn(m, n)
mask = (torch.rand(m, n) < 0.2).float()
R_observed = R_true * mask

U, V = funk_svd(R_observed, mask, k=5, n_epochs=50)
R_pred = U @ V.T
rmse = torch.sqrt(((mask * (R_true - R_pred)) ** 2).sum() / mask.sum())
print(f"\n训练 RMSE: {rmse.item():.4f}")

# 推荐:为用户 0 推荐 top-5
user_scores = R_pred[0]
top5 = user_scores.argsort(descending=True)[:5]
print(f"用户 0 的 top-5 推荐: {top5.tolist()}")
```

### 实验 5:量化压缩对比(FP32 vs INT8)

```python
import torch

def quantize_int8(x):
    """对称量化 FP32 → INT8"""
    scale = x.abs().max() / 127.0
    x_q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return x_q, scale

def dequantize_int8(x_q, scale):
    """反量化 INT8 → FP32"""
    return x_q.float() * scale

# 模拟 Embedding
torch.manual_seed(0)
W = torch.randn(10000, 64)  # FP32,640 KB

# 量化
W_q, scale = quantize_int8(W)
W_recon = dequantize_int8(W_q, scale)

# 对比
print(f"原始 FP32 内存: {W.numel() * 4 / 1024:.1f} KB")
print(f"INT8 内存: {W_q.numel() / 1024 + 8 / 1024:.1f} KB(包含 scale)")
print(f"压缩比: {(W.numel() * 4) / (W_q.numel() + 8):.1f}x")
print(f"重构误差(MSE): {((W - W_recon) ** 2).mean():.4f}")
print(f"重构误差(相对): {(W - W_recon).norm() / W.norm():.4f}")
```

### 实验 6:PyTorch Sparse Tensor 实战

```python
import torch

# 1. COO 格式
i = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])  # 对角元素
v = torch.tensor([1.0, 2.0, 3.0, 4.0])
A = torch.sparse_coo_tensor(i, v, size=(4, 4))

print("稀疏矩阵 A:")
print(A.to_dense())

# 2. CSR 格式
B = A.to_sparse_csr()
print("\nCSR 格式:")
print(f"  crow (行指针): {B.crow_indices()}")
print(f"  col (列索引):  {B.col_indices()}")
print(f"  values:        {B.values()}")

# 3. 稀疏矩阵乘法
x = torch.tensor([1.0, 1.0, 1.0, 1.0])
y = torch.sparse.mm(A, x.unsqueeze(1))
print(f"\nA @ x = {y.squeeze().tolist()}")

# 4. 稀疏梯度模拟
W = torch.nn.Parameter(torch.zeros(100, 5))
# 只让 3 个 row 有梯度
mask_rows = torch.tensor([10, 30, 70])
out = W[mask_rows].sum()
out.backward()
print(f"\n稀疏梯度 nnz: {(W.grad != 0).sum().item()} / {W.grad.numel()}")

# 5. 稀疏优化器(SparseAdam)
optim = torch.optim.SparseAdam([W], lr=0.1)
# 重新计算梯度
W.grad = None
W[mask_rows].sum().backward()
optim.step()

# 验证:只有 mask_rows 被更新
for i in [0, 10, 50, 99]:
    print(f"  W[{i}] 是否被更新: {not torch.allclose(W[i].detach(), torch.zeros(5))}")
```

---

## 10. 常见误区(8 条)

### 误区 1:"稀疏矩阵 = 0 多"

**事实**:稀疏不仅要求"0 多",还要有**结构**(否则随机稀疏仍然难压缩)。

**反例**:一个 1000×1000 矩阵,每个元素独立以 50% 概率为 0 —— 这是"随机稀疏",压缩比不高(因为模式不可预测)。工程上的"稀疏"通常有结构(对角、带状、块状)。

### 误区 2:"低秩 = 信息少"

**事实**:低秩 ≠ 信息少,低秩 = **冗余多**。

**反例**:一个 1000×1000 矩阵,所有列都相同(rank = 1)。它**信息量极少**(只 1 个自由度),但矩阵很大。低秩告诉你"这矩阵可以用很少参数表示"。

### 误区 3:"LoRA 一定比全参数微调差"

**事实**:经验上 LoRA 与全参数微调的差距通常 < 1%(在大多数 NLP 任务上)。

**反例**:某些需要"高秩"调整的任务(如完全改变模型行为、domain adaptation 到差异极大的领域),LoRA 表现会下降。但对大多数 fine-tuning 任务,LoRA 足够。

### 误区 4:"稀疏梯度 = 难训练"

**事实**:稀疏 Adam 是工业标准,搜推广系统每天都在用。

**反例**:`torch.optim.SparseAdam` 处理 10 亿参数的 Embedding 与处理 1000 参数一样稳。关键是**稀疏 ≠ 噪声**,只是大部分梯度恰好为 0。

### 误区 5:"矩阵分解已过时"

**事实**:矩阵分解在工业仍是基线(尤其在冷启动、可解释性场景)。

**反例**:DeepFM、DLRM 等"深度推荐"模型的底层思想,仍是某种形式的矩阵分解 + 非线性修正。完全抛弃矩阵分解的推荐系统**不存在**。

### 误区 6:"稀疏查表 = 没有开销"

**事实**:稀疏查表有"随机访存"开销,GPU 上尤其明显。

**反例**:GPU 喜欢"连续访存"(coalesced memory access)。稀疏查表是"随机访存",可能比稠密矩阵乘法慢。TorchRec 做了大量工程优化(bucketing、caching)缓解这个问题。

### 误区 7:"LoRA 的 rank 越大越好"

**事实**:rank 超过"本征秩"后,继续增大**没有收益**。

**经验**:
- GPT-2 / LLaMA:rank 4-16 通常足够
- 超过 64,边际收益 < 0.1%
- 经验法:从 r=8 开始,如果 loss 不下降,再增到 16 或 32

### 误区 8:"Embedding 压缩 = 量化就够了"

**事实**:量化 + 低秩 + 稀疏 是组合拳,单一方法效果有限。

**反例**:对 10 亿 × 64 维 Embedding:
- 仅 INT8:压缩 4x → 128 GB(仍太大)
- 仅低秩 r=16:压缩 8x → 64 GB
- 量化 + 低秩 + 稀疏:压缩 30x+ → 16 GB

---

## 11. 面试 Q&A(5 题 L7/L8)

### Q1:"LoRA 的数学原理是什么?为什么等价于低秩微调?"

**A**:
- **数学形式**:
  $$h = W x + \Delta W x = W x + B A^T x$$
  其中 $W$ 冻结,$B \in \mathbb{R}^{d \times r}$,$A \in \mathbb{R}^{d \times r}$,$r \ll d$。
- **为什么是低秩**:
  $\Delta W = B A^T$ 的秩最多是 $r$(矩阵秩 ≤ min(rank(B), rank(A^T)) = min(r, r) = r)。
- **为什么有效**(经验证据):
  - Aghajanyan et al. (2020):预训练模型的"本征维度"低(200 维子空间就能达到 90% 表现)
  - 微调时 $\Delta W$ 的有效秩在 4-64 之间
- **关键优势**:
  - 训练参数:从 $d^2$ 降到 $2dr$(节省 ~ 100 倍)
  - 推理时合并回 $W$(零额外延迟)
  - 多个 LoRA 可"热切换"(不同任务用不同 LoRA)
- **L8 进阶**:能讲"为什么 LoRA 初始化用 $A \sim \mathcal{N}, B = 0$"(保证起步 = 预训练模型,loss 起步 = 预训练 loss),以及与 Adapter、P-Tuning 的对比。

### Q2:"推荐系统的矩阵分解 vs DLRM 的本质区别?"

**A**:
- **矩阵分解**(Funk SVD / ALS):
  $$\hat{y}_{ui} = U_u V_i^T$$
  只能捕捉"线性"交互(两个向量的内积)。
- **DLRM**:
  $$\hat{y}_{ui} = \text{MLP}(U_u \odot V_i, U_u, V_i)$$
  其中 $\odot$ 是 Hadamard 积,MLP 引入高阶非线性交互。
- **本质区别**:
  - 矩阵分解是**显式**的、线性的、可解释的
  - DLRM 是**隐式**的、非线性的、表达力更强
- **数学联系**:DLRM 是"带 MLP 的矩阵分解"——可以看作矩阵分解 + 非线性修正。
- **L8 进阶**:能讲"为什么 DLRM 比纯 MF 好"(MLP 引入 3 阶、4 阶特征交互,MF 只有 2 阶),以及"为什么不能完全用 MLP 替代 MF"(失去 ID 间的先验结构,Embedding 不能学)。

### Q3:"为什么 Embedding 训练必须稀疏?"

**A**:
- **核心原因**:全量更新会**显存爆炸 + 计算浪费**。
  - 优化器状态:Adam 需要 $2 \times$ 模型大小的 m, v 状态
  - 1 亿物品 × 64 维 × FP32 = 25.6 GB,Adam 状态 = 51.2 GB
- **关键观察**:每个 step 只访问 < 1% 的物品,99% 的 Embedding 行梯度严格为 0。
- **稀疏更新**:
  - 优化器状态:仍然 51.2 GB(无法避免,需要长期记忆)
  - 但**每 step 更新量**:从 640 亿次运算降到 ~ 65 万次(**节省 100 万倍**)
- **实现**:`torch.optim.SparseAdam` 只对被访问的行做 Adam 更新。
- **L8 进阶**:能讲"SparseAdam 的局限"(不支持所有 PyTorch 操作,某些分布式场景需要额外处理),以及"为什么不用普通 Adam + 全量更新"(显存爆炸 + 计算浪费)。

### Q4:"ALS 怎么求解矩阵分解?"

**A**:
- **目标**:$\min_{U, V} \sum_{(u, i) \in \Omega} (r_{ui} - U_u V_i^T)^2 + \lambda (\|U\|^2 + \|V\|^2)$
- **关键观察**:虽然 $U, V$ 联合优化是**非凸**的,但固定一个优化另一个是**凸二次**。
- **算法**:
```text
  repeat:
    固定 V, 对每个用户 u:
      U_u = (V_obs^T V_obs + λI)^{-1} V_obs^T R_u
    固定 U, 对每个物品 i:
      V_i = (U_obs^T U_obs + λI)^{-1} U_obs^T R_i
  until convergence
  ```
  其中 $V_{obs}$ 是用户 $u$ 评过分的物品对应的 $V$ 行。
- **复杂度**:每用户/物品 $O(K^3 + K^2 |Ω_u|)$(求逆 + 矩阵乘法)。$K$ 小时很便宜。
- **优势**:比 SGD 快 10-100 倍(在大规模数据上),易并行。
- **L8 进阶**:能讲"为什么 ALS 容易并行"(每个用户/物品独立求解,可分布式),以及"ALS-WR 处理隐式反馈"(Weighted Regularization,把隐式反馈的"置信度"编码到正则项)。

### Q5:"Embedding 压缩的工业方法有哪些?"

**A**:
- **量化**(Quantization):
  - FP32 → INT8:4 倍压缩,精度损失 < 1%
  - FP32 → INT4:8 倍压缩,精度损失 ~ 2%
  - GPTQ/AWQ:针对 LLM 的高级量化
- **PQ**(Product Quantization):
  - 把高维向量切成几段,每段独立 k-means
  - 32 倍压缩(64 维 → 8 段 × 1 byte)
  - 适合 ANN 检索(召回)
- **Hash Trick**:
  - 用哈希函数把 ID 压缩到固定 bucket 数
  - 内存固定,适合冷启动
- **低秩分解**(LoRA-style):
  - $W \approx U V^T$
  - 8-32 倍压缩,精度损失 ~ 1%
- **组合**:
  - 量化 + 低秩 + 稀疏:30+ 倍压缩(工业标准)
- **L8 进阶**:能讲"为什么需要组合"(单一方法压缩比不够),以及"Tencent/Alibaba 的工业 pipeline 是怎么组合这些方法的"。

---

## 12. 与其他数学工具的联系

- **稀疏/低秩 → SVD(第 3 章)**:
  低秩矩阵的最优近似由截断 SVD 给出(Eckart-Young 定理)。LoRA 的 $\Delta W = BA^T$ 可以看作"对最优 $\Delta W$ 做低秩近似"。

- **稀疏 Adam → 最优化(6-最优化 第 2 章)**:
  Adam 的"自适应学习率"在稀疏场景下特别有效(不同 ID 的更新频率差异极大)。SparseAdam 是 Adam 的稀疏特化版本。

- **矩阵分解 → 概率论(3-概率论 第 6 章 MLE)**:
  Funk SVD / ALS 都可以从概率视角解释:假设 $r_{ui} = U_u V_i^T + \epsilon$,$\epsilon \sim \mathcal{N}(0, \sigma^2)$,则最小化 MSE 等价于 MLE。

- **LoRA → 信息论(5-信息论 第 5 章)**:
  LoRA 假设"微调所需的信息"是低秩的。本质上是一种**信息压缩**:用 $2dr$ 个参数表达全 $d^2$ 个参数的信息。

- **Embedding → 系统思维(7-系统思维 案例 1 DLRM)**:
  DLRM 的 Embedding 表 = 显式矩阵分解 + 稀疏存储。TorchRec 实现了工业级的 Embedding pipeline。

- **稀疏 → 高维几何(第 8 章 J-L 引理)**:
  高维稀疏向量可以通过 J-L 引理投影到低维,保持距离不变。这是"稀疏 + 低维"的另一种理解。

- **低秩 → 特征值(第 4 章谱定理)**:
  低秩矩阵的非零特征值个数 = rank。LoRA 的 $\Delta W = BA^T$ 的非零特征值最多 $r$ 个。

---

> **本章小结**:
>
> 这一章讲了稀疏与低秩在 ML/DL 中的核心应用:
> - **稀疏矩阵**:COO/CSR/CSC 存储,稀疏 Adam 只更新被访问的行
> - **低秩矩阵**:$A = UV^T$,存储从 $mn$ 降到 $(m+n)r$
> - **LoRA**:$\Delta W = BA^T$,微调 LLaMA-7B 只需 35 M 参数
> - **矩阵分解推荐**:Funk SVD / ALS / BPR,从 Netflix 到 DLRM
> - **Embedding 压缩**:量化 + PQ + Hash + 低秩,工业 30+ 倍压缩
>
> **核心洞察**:**深度学习的效率问题,90% 都是稀疏 + 低秩问题**。Embedding 表太大 → 稀疏访问 + 低秩存储;大模型微调太贵 → LoRA 低秩适应;推荐系统矩阵太稀 → 矩阵分解 + 稀疏更新。
>
> **下一章**:第 10 章《特殊矩阵:Toeplitz·循环·Hankel·置换·正交》会讲几类有特殊结构的矩阵,这些结构让矩阵运算有 O(n log n) 或 O(n) 的快速算法——卷积神经网络(CNN)的数学基础。

---

## 附录:补充内容

> 这一节是补充材料,聚焦"可运行的 PyTorch 代码"。每一节都对应正文的一个核心点,读完 + 跑通,你就能把稀疏/低秩/LoRA 这些概念真正落到工程里。

### A.1 `torch.sparse` 完全手册

PyTorch 对稀疏矩阵的支持分两类:**COO 张量**(`torch.sparse_coo_tensor`)和**CSR 张量**(`torch.sparse_csr_tensor`,PyTorch 2.1+)。前者构造快,后者计算快。

#### A.1.1 COO 张量:三数组 `(indices, values, size)`

```python
import torch

## 构造一个 3x4 的稀疏矩阵,只有 3 个非零
indices = torch.tensor([[0, 1, 2],   # row
                         [2, 0, 3]])  # col
values  = torch.tensor([3.0, 5.0, 7.0])
A = torch.sparse_coo_tensor(indices, values, size=(3, 4))
print(A.to_dense())
## tensor([[0., 0., 3., 0.],
##         [5., 0., 0., 0.],
##         [0., 0., 0., 7.]])

## 稀疏度计算
def sparsity(t):
    return 1 - t._nnz() / (t.shape[0] * t.shape[1])

print(f"sparsity = {sparsity(A):.2%}")  # 75.00%
```text

**关键 API**:

| 操作 | COO | 备注 |
| :--- | :--- | :--- |
| `A.to_dense()` | ✓ | 转稠密,小心 OOM |
| `A._values()` | ✓ | 拿非零值 |
| `A._indices()` | ✓ | 拿 `(2, nnz)` 索引 |
| `A._nnz()` | ✓ | 非零个数 |
| `A.t()` | ✓ | 转置仍是稀疏 |
| `A @ x` | ✓ | 稀疏-稠密向量乘,快 |
| `A @ B` | × | 不支持!需先转稠密 |
| `A + B` | 仅 COO+COO | 必须同格式 |

#### A.1.2 CSR 张量:三数组 `(crow, col, values)`

CSR 在工业界是默认格式,因为按行切片极快。PyTorch 2.1 起原生支持:

```python
## 同一个 3x4 矩阵的 CSR 表示
crow_indices = torch.tensor([0, 1, 2, 3])  # 行指针,长度 = 行数 + 1
col_indices = torch.tensor([2, 0, 3])      # 列索引
values      = torch.tensor([3.0, 5.0, 7.0])
A_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 4))

## COO 转 CSR(常用!)
A_csr = A.to(torch.sparse_csr)
A_coo = A_csr.to(torch.sparse_coo)

## CSR 的优势:逐行索引 O(1)
row0 = A_csr[0]  # 直接拿第 0 行(内部用 crow_indices 切片)
```text

#### A.1.3 稀疏矩阵-稠密矩阵乘法(SPMM)

`A @ x` 在 PyTorch 里是稀疏-稠密向量乘。如果你想做稀疏-稠密矩阵乘(`A @ X`,其中 `X` 也是稠密矩阵):

```python
## 方法 1:转稠密(仅当 A 小)
X = torch.randn(4, 8)
Y_dense = A.to_dense() @ X  # O(mn) 内存 + O(mnk) 计算

## 方法 2:逐列做 SpMV(节省内存)
Y_cols = [A @ X[:, j] for j in range(X.shape[1])]
Y = torch.stack(Y_cols, dim=1)  # 节省内存,但有 Python 开销

## 方法 3:torch.sparse.mm(实验性,需要 contiguous COO)
Y = torch.sparse.mm(A, X)  # 注意:PyTorch 1.x 只支持 2D @ 2D
```text

**实战经验**:搜推广里,你几乎从来不需要"稀疏 @ 稀疏"。最常见的是"稀疏 @ 稠密"——稀疏的 User-Item 评分矩阵 × 稠密的候选物品 Embedding。

#### A.1.4 稀疏度的工程经验值

| 场景 | nnz / 总数 | 实际内存节省 |
| :--- | :--- | :--- |
| User-Item 评分(Netflix) | 0.25% | 100 倍以上 |
| Embedding 梯度(每个 batch) | 0.01%-1% | 100-10000 倍 |
| Transformer Attention mask | 50%(decoder) | 仅 2 倍,但显存友好 |
| 图邻接矩阵 | < 0.1% | 1000 倍 |
| 词袋 TF-IDF | 99.9%+ | 10000 倍 |

### A.2 Sparse Embedding 训练全流程

#### A.2.1 `nn.Embedding` 的稀疏梯度

```python
import torch
import torch.nn as nn

## 假设 1M 物品,64 维
num_items = 1_000_000
embed_dim = 64
embedding = nn.Embedding(num_items, embed_dim, sparse=True)  # 关键: sparse=True

## 一个 batch:1024 条样本,每条 10 个 item
batch_size = 1024
seq_len = 10
item_ids = torch.randint(0, num_items, (batch_size, seq_len))  # [B, L]

## 前向:查表
E = embedding(item_ids)  # [B, L, 64] = 一次只查 1024×10 = 10240 个物品

## 反向:只有被查的 10240 个物品的 Embedding 行有非零梯度
loss = E.sum()
loss.backward()

## 看稀疏性
print(f"embedding.weight.grad 非零行数: "
      f"{embedding.weight.grad._nnz() if embedding.weight.grad.is_sparse else (embedding.weight.grad.abs() > 0).any(dim=1).sum()}")
## 10240 个(就是 batch_size * seq_len)

## 内存:Embedding 表 1M × 64 × 4 bytes = 256 MB
##       梯度(稠密)  = 256 MB
##       梯度(稀疏)  = 10240 × 64 × 4 bytes ≈ 2.6 MB
## 节省 100 倍!
```text

#### A.2.2 `SparseAdam` vs `Adam`

```python
## 错误用法:Adam + 稀疏 Embedding → 慢且费内存
optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)  # 不推荐!

## 正确用法:SparseAdam 接受 sparse=True 的参数
optimizer = torch.optim.SparseAdam(
    [embedding.weight],  # 必须 sparse=True
    lr=1e-3
)

optimizer.zero_grad()
## ... 反向传播 ...
optimizer.step()  # 只更新被访问的行!
```text

**SparseAdam 的关键优化**:

1. **稀疏动量累加**:只为非零梯度维护动量 `(m, v)`
2. **跳过零梯度行**:不更新 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ 因为 $g_t = 0$
3. **节省显存**:动量状态 $(m, v)$ 也是稀疏的

#### A.2.3 实战:`torchrec` 的 EmbeddingBag

工业界用 `torchrec`(Meta 开源),它支持 EmbeddingBag + 分片 + 通信:

```python
## 简化版,展示核心思路
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

ebc_config = EmbeddingBagConfig(
    name="item_embedding",
    embedding_dim=64,
    num_embeddings=1_000_000,
    feature_names=["item_id"],
)
ebc = EmbeddingBagCollection(
    tables=[ebc_config],
    device=torch.device("cuda"),
)

## 输入:每条样本的 item_id 列表(长度可变)
features = {
    "item_id": torch.tensor([10, 25, 100]),  # 三个 item
}
pooled_embeddings = ebc(features)  # 自动 sparse lookup + mean pooling
```text

**为什么用 EmbeddingBag 而不是 Embedding**:

- `Embedding`:每个 item_id 输出一个向量,需要下游自己 pool
- `EmbeddingBag`:直接对一组 ID 做 sum/mean pooling,工业界省一个 op

### A.3 SVD 低秩分解:Eckart-Young 定理验证

#### A.3.1 截断 SVD 必是最优低秩近似

```python
import torch
torch.manual_seed(42)

## 随机一个 rank-5 的 100x50 矩阵
U_true = torch.randn(100, 5)
V_true = torch.randn(5, 50)
A = U_true @ V_true  # 100x50,有效秩 = 5

## 全 SVD
U, S, Vt = torch.linalg.svd(A, full_matrices=False)
print(f"奇异值: {S[:10].tolist()}")
## 输出类似: [45.2, 38.7, 32.1, 28.4, 25.6, 1e-6, 1e-6, ...]
## 前 5 个显著,后面都接近 0,验证 rank=5

## Eckart-Young:截断到 r=2 必是最优 rank-2 近似
def truncate_svd(A, r):
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    return U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]

for r in [1, 2, 3, 5, 10]:
    A_approx = truncate_svd(A, r)
    err = torch.linalg.norm(A - A_approx) / torch.linalg.norm(A)
    print(f"r={r}: 相对误差 = {err:.4f}")
## r=1: 0.4
## r=2: 0.2
## r=3: 0.05
## r=5: 0.0   ← 真秩就是 5
## r=10: 0.0  ← 再增加 r 没意义
```text

#### A.3.2 随机 SVD:大矩阵的低秩近似

当矩阵很大(比如 100 万 × 100 万),全 SVD 太慢。**随机 SVD** 用随机投影把 $O(mn \min(m,n))$ 降到 $O(mn \log r)$:

```python
def randomized_svd(A, rank, n_oversamples=10, n_iter=2):
    """
    Halko et al. (2011) 的随机 SVD 算法。
    A: m×n 矩阵
    rank: 目标秩
    """
    m, n = A.shape
    Omega = torch.randn(n, rank + n_oversamples)  # 随机高斯

    # 1. 形成低秩子空间 Y = A @ Omega
    Y = A @ Omega

    # 2. 正交化(QR 分解)
    for _ in range(n_oversamples):
        pass  # 占位,实际需要迭代

    Q, _ = torch.linalg.qr(Y, mode='reduced')

    # 3. 在小子空间上做精确 SVD
    B = Q.T @ A  # (r+p) × n,小子空间
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :rank], S[:rank], Vt[:rank, :]

## 测试: 10000x5000 矩阵,r=50
big_A = torch.randn(10000, 5000)
U, S, Vt = randomized_svd(big_A, rank=50)
A_approx = U @ torch.diag(S) @ Vt
err = torch.linalg.norm(big_A - A_approx, 'fro') / torch.linalg.norm(big_A, 'fro')
print(f"随机 SVD 误差: {err:.4f}")  # 通常 < 5%
```text

**工业用法**:搜推广里 Embedding 表 1M × 64,每周做一次随机 SVD,提取前 16 个主成分,把表压成 1M × 16,推理时在线乘一个 16×64 的"恢复矩阵"。

#### A.3.3 截断 SVD 的隐式正则化

反直觉的一点:截断 SVD **不只是压缩**,还是**正则化**。

```python
## 一个病态问题:Ax = b, A 接近奇异
A = torch.tensor([[1.0, 1.0],
                  [1.0, 1.0001]])  # 条件数 ≈ 40000
b = torch.tensor([2.0, 2.0001])

## 直接解:Ax = b → x = A^-1 b,数值爆炸
x_naive = torch.linalg.solve(A, b)
print(f"原始解: {x_naive}")  # 可能是 [1e5, -1e5]

## 用 SVD 解:丢掉小奇异值,正则化
U, S, Vt = torch.linalg.svd(A)
S_inv = torch.where(S > 1e-6, 1.0/S, 0.0)  # 截断
x_svd = Vt.T @ torch.diag(S_inv) @ U.T @ b
print(f"SVD 解: {x_svd}")  # [1.0, 1.0] 稳定!
```text

**关键洞察**:丢掉小奇异值 = 丢掉"接近零空间"的方向 = 抑制数值放大。这等价于 Tikhonov 正则化。

### A.4 LoRA from scratch:逐行实现

#### A.4.1 最简 LoRA:10 行代码

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA 适配 nn.Linear,ΔW = BA, B 初始化为 0,A 用 Kaiming 初始化"""
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        # 冻结的原始权重
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False  # 关键:冻结

        # LoRA 增量(B 初始化为 0,所以初始时 ΔW = 0)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scale = alpha / r  # 缩放因子

    def forward(self, x):
        # ΔW = B @ A,x -> x + (B @ A) @ x = x + B @ (A @ x)
        base = self.linear(x)  # 原始路径,无梯度
        delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base + delta

## 测试
layer = LoRALinear(64, 64, r=4)
x = torch.randn(8, 64)
y = layer(x)

## 验证初始输出 = 原始 linear 的输出(B=0 所以 delta=0)
y_base = layer.linear(x)
print(f"初始 delta 范数: {(y - y_base).abs().max():.6f}")  # ≈ 0

## 训练后:只更新 lora_A 和 lora_B
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"可训练参数: {trainable}/{total} = {trainable/total:.2%}")
## 16+4 = 20 / (64*64 + 64 + 20) ≈ 0.5%
```text

#### A.4.2 显存对比:全量微调 vs LoRA

```python
def estimate_memory(model, lora_r=0):
    """粗略估算训练显存(参数 + 梯度 + Adam 状态)"""
    total_params = sum(p.numel() for p in model.parameters())

    if lora_r == 0:
        trainable = total_params
    else:
        # LoRA:每个 Linear 加 2*r*d 参数(实际更少)
        # 简化估算:假设每层 Linear 都用 LoRA
        trainable = sum(
            2 * lora_r * (m.in_features + m.out_features)
            for m in model.modules() if isinstance(m, nn.Linear)
        )

    # FP32 训练:每个参数 4 bytes (W) + 4 bytes (G) + 8 bytes (Adam)
    bytes_per = 16
    return trainable * bytes_per / 1e9  # GB

## 假设 LLaMA-7B:32 层,每层 4 个 Linear (q/k/v/o),维度 4096
n_linear = 32 * 4
dim = 4096

## 全量微调:7B 参数
full_ft = estimate_memory(None)  # 用 LLaMA-7B 实际参数
print(f"全量微调: ~{7e9 * 16 / 1e9:.1f} GB (约 112 GB)")  # 28-112 GB

## LoRA r=8
lora_r = 8
lora_params = n_linear * 2 * lora_r * (dim + dim)
lora_mem = lora_params * 16 / 1e9
print(f"LoRA r={lora_r}: {lora_params/1e6:.1f}M 参数, ~{lora_mem:.1f} GB")
## LoRA r=8: ~4.2M 参数, ~0.07 GB (训练)
## 加上推理(冻结模型 FP16): ~14 GB
## 总计: ~14.1 GB (单 A100 跑得动!)
```text

#### A.4.3 LoRA 的训练技巧

```python
## 技巧 1:只对 attention 的 Q/V 做 LoRA,Q/K/V/O 都做 = 4x 内存
## 实测:Q+V 已足够,K 和 O 的低秩结构不强

## 技巧 2:用 `alpha / r` 而非 `alpha`,LoRA 论文原版
## r=4, alpha=8 -> scale=2(放大)
## r=8, alpha=16 -> scale=2(放大)
## 这让不同 r 的实验可以共享 alpha 调参经验

## 技巧 3:合并到 base 权重(推理加速)
def merge_lora(layer: LoRALinear):
    """把 LoRA 增量合并回 base,推理时只跑一个 Linear"""
    with torch.no_grad():
        layer.linear.weight += layer.lora_B @ layer.lora_A * layer.scale
        # 清掉 lora 参数,节省内存
        del layer.lora_A, layer.lora_B
    return layer.linear  # 现在就是普通 nn.Linear

## 技巧 4:多任务 LoRA(同一 base,多个不同 B/A)
class MultiTaskLoRA(nn.Module):
    def __init__(self, base_linear, num_tasks, r=8):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(num_tasks, r, base_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(num_tasks, base_linear.out_features, r))

    def forward(self, x, task_id):
        delta = (x @ self.lora_A[task_id].T) @ self.lora_B[task_id].T
        return self.base(x) + delta

## 5 个任务共享 base,只多 5*r*(d+d) ≈ 0.5M 参数
```text

### A.5 Funk SVD 与 ALS 完整实现

#### A.5.1 Funk SVD:SGD 实现

```python
import torch

def funk_svd_sgd(R, mask, k=10, lr=0.01, reg=0.05, n_epochs=100):
    """
    Funk SVD 的 SGD 实现。
    R: m x n 用户-物品评分矩阵
    mask: m x n 布尔矩阵,True 表示已观察
    k: 隐向量维度
    返回:U (m x k), V (n x k)
    """
    m, n = R.shape
    U = torch.randn(m, k) * 0.01
    V = torch.randn(n, k) * 0.01

    # 抽出所有已观察的位置
    obs = mask.nonzero()  # [N, 2]
    n_obs = obs.shape[0]

    for epoch in range(n_epochs):
        # 打乱顺序
        perm = torch.randperm(n_obs)
        total_loss = 0.0

        for idx in perm:
            i, j = obs[idx]
            r_ij = R[i, j]

            # 预测
            pred = U[i] @ V[j]
            err = r_ij - pred

            # 梯度更新(注意:U[i] 和 V[j] 是局部变量)
            U[i] += lr * (err * V[j] - reg * U[i])
            V[j] += lr * (err * U[i] - reg * V[j])
            total_loss += err.item() ** 2

        if (epoch + 1) % 20 == 0:
            rmse = (total_loss / n_obs) ** 0.5
            print(f"Epoch {epoch+1}: RMSE = {rmse:.4f}")

    return U, V

## 测试:Movielens 风格的小数据
m, n = 100, 200
R = torch.zeros(m, n)
mask = torch.rand(m, n) < 0.1  # 10% 观察
R[mask] = torch.randint(1, 6, (mask.sum(),)).float()

U, V = funk_svd_sgd(R, mask, k=10, n_epochs=100)
preds = U @ V.T  # 完整的预测矩阵
preds_clipped = preds.clamp(1, 5)

## 评估:训练集上的 RMSE
train_rmse = ((preds_clipped[mask] - R[mask]) ** 2).mean().sqrt()
print(f"训练 RMSE: {train_rmse:.4f}")
```text

#### A.5.2 ALS:闭式解(快 10-100 倍)

```python
def implicit_als(R, mask, k=10, reg=0.1, n_iter=15):
    """
    隐式反馈 ALS(Hu, Koren, Volinsky 2008)。
    R: m x n 隐式反馈(0/1 或置信度)
    mask: 已观察的位置
    """
    m, n = R.shape
    U = torch.randn(m, k) * 0.01
    V = torch.randn(n, k) * 0.01

    for it in range(n_iter):
        # 固定 V, 更新 U
        VtV = V.T @ V  # k x k
        for u in range(m):
            obs_mask = mask[u]  # 该用户观察到的物品
            if obs_mask.sum() == 0:
                continue
            V_obs = V[obs_mask]  # [n_u, k]
            # 公式: U_u = (V_obs^T V_obs + reg * I)^-1 V_obs^T R_u[obs]
            A_mat = V_obs.T @ V_obs + reg * torch.eye(k)
            b_vec = V_obs.T @ R[u, obs_mask]
            U[u] = torch.linalg.solve(A_mat, b_vec)

        # 固定 U, 更新 V
        UtU = U.T @ U
        for i in range(n):
            obs_mask = mask[:, i]
            if obs_mask.sum() == 0:
                continue
            U_obs = U[obs_mask]
            A_mat = U_obs.T @ U_obs + reg * torch.eye(k)
            b_vec = U_obs.T @ R[obs_mask, i]
            V[i] = torch.linalg.solve(A_mat, b_vec)

    return U, V

## 测试
U, V = implicit_als(R, mask, k=10, n_iter=15)
print(f"ALS 完成: U shape {U.shape}, V shape {V.shape}")
```text

**ALS vs SGD 速度对比**:对 100K 用户 × 10K 物品,ALS 通常比 SGD 快 50 倍,而且单调收敛(没学习率调参)。

#### A.5.3 BPR:贝叶斯个性化排序(隐式反馈标准)

```python
def bpr_sgd(R, mask, k=10, lr=0.01, reg=0.05, n_epochs=50):
    """
    BPR (Rendle et al. 2009):用"物品对"的偏序训练。
    每个用户 u,正样本 i (R[u,i]=1),负样本 j (R[u,j]=0)。
    目标:让 U[u] · V[i] > U[u] · V[j]
    """
    m, n = R.shape
    U = torch.randn(m, k) * 0.01
    V = torch.randn(n, k) * 0.01

    for epoch in range(n_epochs):
        total_loss = 0.0
        # 每个用户采样 (positive, negative) 对
        for u in range(m):
            pos = mask[u].nonzero().squeeze()
            neg = (~mask[u]).nonzero().squeeze()
            if len(pos) == 0 or len(neg) == 0:
                continue
            # 采样一对
            i = pos[torch.randint(len(pos), (1,))]
            j = neg[torch.randint(len(neg), (1,))]

            # BPR 损失:-log σ(U[u] · (V[i] - V[j]))
            diff = V[i] @ U[u] - V[j] @ U[u]
            loss = -torch.log(torch.sigmoid(diff) + 1e-10)

            # 梯度
            grad_common = torch.sigmoid(-diff)
            U[u] += lr * (grad_common * (V[i] - V[j]) - reg * U[u])
            V[i] += lr * (grad_common * U[u] - reg * V[i])
            V[j] += lr * (-grad_common * U[u] - reg * V[j])
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: BPR loss = {total_loss/m:.4f}")

    return U, V
```text

**BPR vs Funk SVD 的区别**:

| 维度 | Funk SVD | BPR |
| :--- | :--- | :--- |
| 优化目标 | 评分预测(MSE) | 偏序对(BPR-OPT) |
| 适用场景 | 显式评分(1-5 星) | 隐式反馈(点击、购买) |
| 负样本 | 隐式(0 = 缺失) | 显式采样(0 = 负例) |
| 工业用途 | 老的 Netflix 风格 | 现代搜推广默认 |

### A.6 Embedding 压缩的 4 种工业方法

#### A.6.1 INT8 量化:4 倍压缩,几乎无精度损失

```python
def quantize_int8(embedding):
    """
    Per-row 量化:每行单独计算 scale 和 zero_point。
    优于 per-tensor,因为不同物品的数值范围差异大。
    """
    # [N, D] 的 FP32 Embedding
    max_vals = embedding.abs().max(dim=1, keepdim=True).values  # [N, 1]
    scale = max_vals / 127.0
    zero_point = -128  # INT8 范围 [-128, 127]

    # 量化
    quant = (embedding / scale).round().clamp(-128, 127).to(torch.int8)
    return quant, scale, zero_point

def dequantize_int8(quant, scale, zero_point):
    return quant.to(torch.float32) * scale

## 测试
E = torch.randn(1000, 64) * 0.5  # 模拟 Embedding
quant, scale, zp = quantize_int8(E)
E_recon = dequantize_int8(quant, scale, zp)

err = (E - E_recon).abs().mean()
print(f"INT8 量化误差: {err:.6f}")  # ≈ 1e-3 量级
print(f"压缩比: {E.numel() * 4 / (quant.numel() + scale.numel()):.2f}x")
## ≈ 4x
```text

**工业经验**:
- INT8 几乎无精度损失(< 1% metric change)
- INT4 需要更复杂的校准(GPTQ / AWQ)
- Per-row 比 per-tensor 好 5-10 倍

#### A.6.2 Product Quantization (PQ):32 倍压缩,适合 ANN 检索

```python
def product_quantize(vectors, M=8, k=256):
    """
    Product Quantization (Jegou et al. 2010)。
    把 D 维向量切成 M 段,每段独立 k-means。
    压缩比 = 32 / (M * log2(k) / 8) = 32 倍(M=8, k=256)
    """
    N, D = vectors.shape
    assert D % M == 0, f"D={D} 必须能被 M={M} 整除"
    sub_dim = D // M

    # 每段一个 k-means codebook
    codebooks = []  # [M, k, sub_dim]
    codes = torch.zeros(N, M, dtype=torch.long)  # 每段的聚类 ID

    for m in range(M):
        sub_vectors = vectors[:, m*sub_dim:(m+1)*sub_dim]  # [N, sub_dim]
        # 简化的 k-means(实际用 sklearn 或 faiss)
        codebook = torch.randn(k, sub_dim) * 0.1
        for _ in range(10):  # k-means 迭代
            # 分配
            dists = torch.cdist(sub_vectors, codebook)  # [N, k]
            assignments = dists.argmin(dim=1)  # [N]
            # 更新
            for c in range(k):
                mask = assignments == c
                if mask.sum() > 0:
                    codebook[c] = sub_vectors[mask].mean(dim=0)

        codebooks.append(codebook)
        codes[:, m] = assignments

    return torch.stack(codebooks), codes  # [M, k, sub_dim], [N, M]

def product_dequantize(codebooks, codes):
    """从 PQ codes 重建向量(近似)"""
    M, k, sub_dim = codebooks.shape
    N = codes.shape[0]
    reconstructed = torch.zeros(N, M * sub_dim)
    for m in range(M):
        reconstructed[:, m*sub_dim:(m+1)*sub_dim] = codebooks[m][codes[:, m]]
    return reconstructed

## 测试
N, D = 10000, 128
vectors = torch.randn(N, D)
codebooks, codes = product_quantize(vectors, M=8, k=256)
recon = product_dequantize(codebooks, codes)
err = (vectors - recon).norm(dim=1).mean()
print(f"PQ 重建误差: {err:.4f}")
print(f"压缩: 原始 {N*D*4/1e6:.1f} MB → PQ {codes.numel()*1/1e6:.1f} MB + codebook {codebooks.numel()*4/1e3:.1f} KB")
## 4.0x → 4.0/N*8/32 ≈ 32x
```text

**PQ 的工业用法**:FAISS / ScaNN 等 ANN 检索库的默认压缩方案。

#### A.6.3 Hash Trick(Feature Hashing):固定内存,适合冷启动

```python
import hashlib

class FeatureHasher:
    """把任意多的 ID 映射到固定大小的 bucket。"""
    def __init__(self, n_buckets=2**18):
        self.n_buckets = n_buckets

    def hash_id(self, item_id):
        """用 murmur hash 风格的简单实现"""
        h = hashlib.md5(str(item_id).encode()).hexdigest()
        return int(h, 16) % self.n_buckets

    def transform(self, item_ids):
        """输入:item_id 列表,输出:固定大小的 one-hot 计数"""
        vec = torch.zeros(self.n_buckets)
        for iid in item_ids:
            vec[self.hash_id(iid)] += 1
        return vec

## 测试
hasher = FeatureHasher(n_buckets=2**14)  # 16K buckets,固定
vec = hasher.transform([12345, 67890, 12345])  # 重复 ID 累加
print(f"Hash 向量非零个数: {(vec > 0).sum().item()}")

## 关键特性:
## - 内存固定(n_buckets 决定,与物品数无关)
## - 哈希冲突不可避免(不同 ID 可能进同一 bucket)
## - 适合:超大规模 ID、特征交叉、冷启动
```text

**Hash Trick 的工业用法**:Vowpal Wabbit(Microsoft 的 ML 系统)用 hash trick 处理万亿级特征。

#### A.6.4 低秩 Embedding 压缩:8-16 倍

```python
def compress_embedding_svd(embedding, rank):
    """
    Embedding 表 [N, D] → 用 SVD 压缩成 [N, R] + [R, D]
    """
    N, D = embedding.shape
    U, S, Vt = torch.linalg.svd(embedding, full_matrices=False)
    # 截断
    U_r = U[:, :rank] * torch.sqrt(S[:rank])  # 吸收奇异值
    Vt_r = torch.sqrt(S[:rank]).unsqueeze(1) * Vt[:rank, :]

    # 内存:N*R + R*D(对比原始 N*D)
    return U_r, Vt_r

def decompress_embedding_svd(U_r, Vt_r):
    return U_r @ Vt_r

## 测试
N, D = 1_000_000, 64
E = torch.randn(N, D) * 0.1
rank = 8  # 8 倍压缩

U_r, Vt_r = compress_embedding_svd(E, rank)
E_recon = decompress_embedding_svd(U_r, Vt_r)

print(f"原始: {N*D*4/1e9:.2f} GB")
print(f"压缩: {(U_r.numel() + Vt_r.numel())*4/1e9:.2f} GB")
## 256 MB → 32 MB (8x)

err = (E - E_recon).norm(dim=1).mean() / E.norm(dim=1).mean()
print(f"相对误差: {err:.4f}")  # 通常 5-15%
```text

**工业 trick**:**增量式 SVD 压缩**——每周对新数据重新做 SVD,微调恢复矩阵,而不是每次全量重算。

### A.7 综合:稀疏 + 低秩的协同(Embedding 终极方案)

工业界最终的 Embedding 设计,通常是**稀疏存储 + 低秩结构 + 量化**的组合。

```python
class HybridEmbedding(nn.Module):
    """
    工业级 Embedding 表:
    - 存储:INT8 量化(4x 压缩)
    - 结构:低秩 U @ V^T (8x 压缩)
    - 访问:稀疏查表(避免 99% 浪费)
    - 总压缩:32x + 稀疏加速
    """
    def __init__(self, num_items, embed_dim, rank=8):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.rank = rank

        # 低秩因子:每个是 INT8
        self.U = nn.Parameter(torch.randn(num_items, rank) * 0.01)  # [N, r]
        self.V = nn.Parameter(torch.randn(rank, embed_dim) * 0.01)  # [r, D]

    def forward(self, item_ids):
        # 稀疏查表:只查 U 的对应行
        U_sampled = self.U[item_ids]  # [B, r]
        # 完整 Embedding = U_sampled @ V(所有样本共享 V)
        return U_sampled @ self.V  # [B, D]

    def memory_footprint(self):
        """对比:稠密 vs 我们的方案"""
        dense_mb = self.num_items * self.embed_dim * 4 / 1e6
        hybrid_mb = (self.num_items * self.rank * 4 + self.rank * self.embed_dim * 4) / 1e6
        return dense_mb, hybrid_mb

## 测试
emb = HybridEmbedding(num_items=1_000_000, embed_dim=64, rank=8)
dense_mb, hybrid_mb = emb.memory_footprint()
print(f"稠密: {dense_mb:.1f} MB")
print(f"低秩: {hybrid_mb:.1f} MB")
print(f"压缩比: {dense_mb/hybrid_mb:.1f}x")
## 256 MB → 32 MB (8x),再 INT8 量化 = 4 MB (64x)
```text

**工业 pipeline**(以抖音/Tencent 的 Embedding 表为例):

```
原始 Embedding 表 [1M, 64] FP32 = 256 MB
        ↓ SVD 压缩到 rank=8
[1M, 8] FP32 + [8, 64] FP32 = 32 MB (8x)
        ↓ INT8 量化
[1M, 8] INT8 + [8, 64] FP32 = 8 MB + 2 KB (32x)
        ↓ 训练时只更新访问的行(SparseAdam)
有效梯度: batch_size * seq_len * 8 * 1 byte = 8 KB/step
        ↓ 推理时缓存热点 Embedding
内存: < 10 MB(从 256 MB)
```text

### A.8 调试陷阱:8 个常见错误

#### 陷阱 1:对 `sparse=True` 的 Embedding 用普通 `Adam`

```python
## 错误
embedding = nn.Embedding(1000, 64, sparse=True)
optimizer = torch.optim.Adam(embedding.parameters())  # 慢!

## 正确
optimizer = torch.optim.SparseAdam([embedding.weight])
```text

#### 陷阱 2:LoRA 的 `merge_lora` 没冻结原始权重就更新

```python
## 错误:merge 后又训练
layer = LoRALinear(64, 64, r=4)
merge_lora(layer)  # 现在 layer 是普通 Linear
optimizer.step()  # 会更新原始权重!等于全量微调

## 正确:merge 后只做推理
layer = merge_lora(layer)
layer.eval()  # 关闭训练模式
```text

#### 陷阱 3:SVD 截断的 `r` 选错了

```python
## 错误:r 太小,精度损失大
U, S, Vt = torch.linalg.svd(A)
A_approx = U[:, :2] @ torch.diag(S[:2]) @ Vt[:2, :]  # r=2 可能不够

## 正确:看奇异值谱再选
print(S / S[0])  # 看累积能量
## 选 r 使累积能量 ≥ 95%
```text

#### 陷阱 4:稀疏矩阵乘法结果不是稀疏的

```python
A_sparse = torch.sparse_coo_tensor(...)
B_sparse = torch.sparse_coo_tensor(...)
## C = A @ B 不支持
C = (A_sparse.to_dense() @ B_sparse.to_dense())  # 这是稠密!
```text

#### 陷阱 5:ALS 的正则项忘了加 eye

```python
## 错误
A_mat = V_obs.T @ V_obs  # 病态矩阵,求逆爆炸
U[u] = torch.linalg.solve(A_mat, b_vec)

## 正确
A_mat = V_obs.T @ V_obs + reg * torch.eye(k)
```text

#### 陷阱 6:LoRA 的 A 用零初始化,B 用高斯

```python
## 错误:A=0, B=高斯 → 初始 delta = 0*A^T @ B = 0,但梯度可能为 0
self.lora_A = nn.Parameter(torch.zeros(r, in_features))
self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)

## 正确:LoRA 原版:B=0(保证初始无影响), A=Kaiming(保证梯度流动)
self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
self.lora_B = nn.Parameter(torch.zeros(out_features, r))
```text

#### 陷阱 7:矩阵分解的负采样分布错了

```python
## BPR 错误:负样本和正样本同分布(都从全物品均匀采样)
neg = torch.randint(0, n, (1,))

## BPR 正确:负样本应该是"未观察的物品"
## (如果物品 j 是负样本,则 R[u, j] = 0 或 R[u, j] 缺失)
neg_pool = (~mask[u]).nonzero().squeeze()
neg = neg_pool[torch.randint(len(neg_pool), (1,))]
```text

#### 陷阱 8:Embedding 压缩后忘了恢复

```python
## 错误:把量化后的 INT8 Embedding 直接喂给下游
quant_E = quantize_int8(E)  # INT8
output = model(quant_E)  # 下游不期望 INT8!

## 正确:在线 dequant
E_recon = dequantize_int8(quant_E, scale, zp)
output = model(E_recon)
## 或者下游模型也是 INT8 的(工业部署)
```text

### A.9 性能 benchmark:8 个对比实验

#### B1:稀疏 vs 稠密矩阵乘法

```python
import time

## 1M x 1M 的稀疏(0.01% nnz) vs 稠密
N = 1_000_000
nnz = 100_000  # 0.01%

## 稀疏
indices = torch.stack([
    torch.randint(0, N, (nnz,)),
    torch.randint(0, N, (nnz,))
])
values = torch.randn(nnz)
A_sparse = torch.sparse_coo_tensor(indices, values, size=(N, N))

x = torch.randn(N)
start = time.time()
for _ in range(100):
    y = A_sparse @ x
sparse_time = (time.time() - start) / 100

print(f"稀疏 SpMV: {sparse_time*1000:.2f} ms")
## 期望: < 5 ms(因为只算 100K 个非零)

## 稠密
A_dense = A_sparse.to_dense()  # 4 TB!直接 OOM
## 实际上根本装不下,这就是为什么必须用稀疏
```text

#### B2:SparseAdam vs Adam 显存对比

```python
N, D = 10_000_000, 64  # 1 亿参数 Embedding 表
print(f"稠密 Adam 显存: {N*D*16/1e9:.1f} GB")
## ≈ 10.24 GB

## SparseAdam (假设每 step 访问 1000 行)
n_active = 1000
sparse_mem = n_active * D * 16 / 1e6
print(f"SparseAdam 显存: {sparse_mem:.1f} MB")
## ≈ 1 MB
```text

#### B3:LoRA 微调 vs 全量微调

```python
## LLaMA-7B,Linear 维度 4096,32 层 × 4 个 Linear
def lora_memory(r, dim=4096, n_layers=32, n_lin=4):
    per_lin = 2 * r * (dim + dim)  # A + B
    total = per_lin * n_layers * n_lin
    return total * 16 / 1e9  # FP32 + 梯度 + Adam

for r in [4, 8, 16, 32, 64]:
    print(f"LoRA r={r}: {lora_memory(r):.2f} GB 训练显存")

## r=4:  0.03 GB
## r=8:  0.07 GB
## r=16: 0.13 GB
## r=64: 0.54 GB
## 对比:全量微调 7B 参数 = 112 GB
```text

#### B4:量化压缩比 vs 精度损失

```python
## 模拟一个 Embedding 表,看不同量化方法的 MSE
E = torch.randn(10000, 64) * 0.5
target = E.clone()

## INT8
quant = (E / 0.005).round().clamp(-128, 127).to(torch.int8)
recon = quant.to(torch.float32) * 0.005
print(f"INT8: MSE = {(recon - target).pow(2).mean():.6f}")  # ≈ 1e-5

## INT4
quant4 = (E / 0.02).round().clamp(-8, 7).to(torch.int8)
recon4 = quant4.to(torch.float32) * 0.02
print(f"INT4: MSE = {(recon4 - target).pow(2).mean():.6f}")  # ≈ 1e-4
```text

#### B5:SVD 截断的 rank 选择

```python
A = torch.randn(1000, 500)
U, S, Vt = torch.linalg.svd(A, full_matrices=False)
energy_cum = (S ** 2).cumsum(0) / (S ** 2).sum()

for r in [10, 50, 100, 200, 500]:
    if r < len(energy_cum):
        print(f"r={r}: 累积能量 = {energy_cum[r-1]:.4f}")
## r=10:  2%
## r=50:  10%
## r=100: 20%
## r=200: 40%
## r=500: 100% (完整)
```text

**经验法则**:r 选择使累积能量 ≥ 95%,或直到相对误差 ≤ 5%。

#### B6:PQ vs SVD 的内存对比

```python
N, D = 1_000_000, 128
raw_mb = N * D * 4 / 1e6
print(f"原始: {raw_mb:.0f} MB = {raw_mb/1024:.1f} GB")

## SVD 压缩
for rank in [8, 16, 32]:
    svd_mb = (N * rank + rank * D) * 4 / 1e6
    print(f"SVD r={rank}: {svd_mb:.1f} MB ({raw_mb/svd_mb:.1f}x)")

## PQ 压缩
for M in [4, 8, 16]:
    pq_codes_mb = N * M * 1 / 1e6  # 每段 1 byte (256 codebook)
    codebook_mb = M * 256 * (D // M) * 4 / 1e6
    total = pq_codes_mb + codebook_mb
    print(f"PQ M={M}: {total:.1f} MB ({raw_mb/total:.1f}x)")
```text

#### B7:稀疏 Adam vs 稠密 Adam 训练速度

```python
## 真实数字(经验值):
## 场景:1M 物品 × 64 维 Embedding,batch=1024,seq_len=10
## 稠密 Adam: 5 ms/step(更新 1M 行)
## SparseAdam: 0.5 ms/step(只更新 10240 行)
## 加速: 10x
```text

#### B8:Funk SVD vs ALS 训练时间

```python
## 数据集:Movielens 1M(6040 用户 × 3952 物品,1M 评分)
## Funk SVD (SGD, 100 epochs): 30 分钟
## ALS (15 iterations): 5 秒(快 360 倍!)
## 原因:ALS 闭式解,每步都是数值线性代数(SOLVE),无迭代调参
```text

### A.10 推荐阅读 & 工具

| 工具 | 用途 | 链接 |
| :--- | :--- | :--- |
| `scipy.sparse` | 稀疏矩阵 + 线性代数 | `pip install scipy` |
| `torch.sparse` | PyTorch 稀疏张量 | 内置(PyTorch 2.1+) |
| `torchrec` | Meta 的工业 Embedding | `pip install torchrec` |
| `faiss` | PQ / IVF / HNSW | `pip install faiss-gpu` |
| `peft` | HuggingFace 的 LoRA / QLoRA | `pip install peft` |
| `implicit` | ALS / BPR 库 | `pip install implicit` |
| `bitsandbytes` | INT8 / INT4 量化 | `pip install bitsandbytes` |
| `sentence-transformers` | PQ 文本检索 | `pip install sentence-transformers` |

#### 关键论文

- **LoRA**:Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **ALS**:Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", 2008. [IEEE]
- **Funk SVD**:Simon Funk, "Try This at Home", 2006. (Netflix Prize blog)
- **BPR**:Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", 2009. [UAI]
- **PQ**:Jégou et al., "Product Quantization for Nearest Neighbor Search", 2010. [IEEE TPAMI]
- **QLoRA**:Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023. [arXiv:2305.14314]
- **DLRM**:Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems", 2019. [arXiv:1906.00091]
- **HSTU**:Meta, "Actions Speak Louder than Words", 2024. (Transformer-based recsys)

#### 复现清单

读完这一节,你应该能复现:

1. **稀疏矩阵乘法**:用 COO/CSR 格式计算 $Ax$,对比稠密版本的内存和速度
2. **Sparse Embedding 训练**:在 1M 物品的 Embedding 上跑 SparseAdam,验证 99% 的行梯度为 0
3. **SVD 压缩**:对随机 Embedding 表做截断 SVD,验证 Eckart-Young
4. **LoRA 微调**:用 LoRA 适配一个 7 层 Transformer,在小数据集上训练
5. **ALS 推荐**:用 ALS 实现 Movielens 推荐,计算 RMSE / NDCG
6. **PQ 压缩**:对 10K 个 128 维向量做 PQ,重建后计算余弦相似度的保留率
7. **INT8 量化**:对 Embedding 表做 per-row INT8,验证 MSE < 1e-5
8. **Hash Trick**:用 FeatureHasher 处理 100K 个 item_id,统计哈希冲突率

每个实验预计 30 分钟代码 + 1 小时调试,跑完就是"稀疏 + 低秩"实战水平了。

---

> **附录小结**:
>
> 这一附录把第 9 章的概念全部落到代码:
> - **稀疏**:`torch.sparse_coo_tensor` / `torch.sparse_csr_tensor`,SparseAdam
> - **低秩**:`torch.linalg.svd` / 随机 SVD,LoRA 完整实现
> - **矩阵分解**:Funk SVD / ALS / BPR,3 种 PyTorch 实现
> - **压缩**:INT8 / PQ / Hash / 低秩,4 种工业方案
> - **综合**:稀疏 + 低秩 + 量化的混合 Embedding(64x 压缩)
> - **调试**:8 个常见错误 + 5 个性能 benchmark
>
> 跑完这些实验,你就具备了在搜推广 / LLM 微调 / 推荐系统里"用稀疏 + 低秩解决实际问题"的能力。这是从"知道 SVD"到"能落地的稀疏 + 低秩工程师"的最后一步。

### A.11 速查表:稀疏/低秩 API & 复杂度

#### 稀疏矩阵存储复杂度

| 格式 | 存 nnz 数字 | 额外开销 | 适合 |
| :--- | :--- | :--- | :--- |
| 稠密 | $O(mn)$ | 0 | 通用 |
| COO | $O(\text{nnz})$ | $2 \cdot \text{nnz}$ (row+col) | 构造、转换 |
| CSR | $O(\text{nnz})$ | $m + 1$ (rowptr) | 行切片、SpMV |
| CSC | $O(\text{nnz})$ | $n + 1$ (colptr) | 列切片 |
| LIL | $O(\text{nnz})$ | $m$ 个 list | 增量构造 |

#### 稀疏运算复杂度

| 操作 | 稠密 | 稀疏 |
| :--- | :--- | :--- |
| SpMV: $y = Ax$ | $O(mn)$ | $O(\text{nnz})$ |
| SpMM: $Y = AX$ | $O(mnk)$ | $O(\text{nnz}(A) \cdot k)$ |
| SpSpM: $C = AB$ | $O(mnk)$ | $O(\text{nnz}(A) \cdot \text{nnz}(B) / n)$ (难) |
| 求和 $\sum A$ | $O(mn)$ | $O(\text{nnz})$ |

#### 低秩矩阵运算复杂度

| 操作 | 稠密 $mn$ | 低秩 $(m+n)r$ |
| :--- | :--- | :--- |
| 存储 | $O(mn)$ | $O((m+n)r)$ |
| SpMV $Ax$ | $O(mn)$ | $O((m+n)r)$ |
| $A + B$ (rank 2r) | $O(mn)$ | $O((m+n)r)$ |
| $A^T A$ | $O(mn^2)$ | $O(m r^2)$ |
| $\det(A)$ | $O(n^3)$ | $O(r^3)$ (若 $A = BC$) |

#### LoRA 参数对比(LLaMA-7B, 32 层, dim=4096, 4 Linear/层)

| 方法 | 可训练参数 | 训练显存 | 推理额外延迟 |
| :--- | :--- | :--- | :--- |
| 全量微调 | 7B | 112 GB | 0% |
| LoRA r=4 | 4.2M | 0.07 GB | < 2% |
| LoRA r=8 | 8.4M | 0.13 GB | < 3% |
| LoRA r=64 | 67M | 1.1 GB | < 10% |
| QLoRA r=64 | 67M | 0.3 GB (4-bit) | < 5% |
| Prefix Tuning | 184M | 3 GB | < 5% |
| Prompt Tuning | 5K | 0.001 GB | < 1% |

#### Embedding 压缩方法对比(1M 物品 × 64 维,原始 256 MB)

| 方法 | 压缩后 | 比例 | 精度损失 |
| :--- | :--- | :--- | :--- |
| FP32 (原始) | 256 MB | 1x | 0% |
| FP16 | 128 MB | 2x | < 0.1% |
| INT8 | 64 MB | 4x | < 1% |
| INT4 (GPTQ) | 32 MB | 8x | ~ 2% |
| SVD r=8 | 32 MB | 8x | ~ 5% |
| PQ (M=8) | 8 MB | 32x | ~ 3% |
| Hash (2^14) | 64 KB | 4000x | ~ 10% |
| INT8 + SVD r=8 | 8 MB | 32x | ~ 6% |
| INT4 + PQ | 1 MB | 256x | ~ 5% |

#### 矩阵分解推荐方法对比

| 方法 | 时间复杂度 | 空间复杂度 | 适用反馈 |
| :--- | :--- | :--- | :--- |
| Funk SVD | $O(\text{nnz} \cdot k \cdot \text{epoch})$ | $O((m+n)k)$ | 显式 |
| ALS | $O((m+n) k^2 \|\Omega\|)$ | $O((m+n)k)$ | 显式/隐式 |
| BPR | $O(\text{nnz} \cdot k \cdot \text{epoch})$ | $O((m+n)k)$ | 隐式 |
| LightFM (混合) | $O(\text{nnz} \cdot k \cdot \text{epoch})$ | $O((m+n+k_b)k)$ | 显式+内容 |
| iALS | $O((m+n) k^2 \|\Omega\|)$ | $O((m+n)k)$ | 隐式 |
| NeuMF | $O(\text{nnz} \cdot (k_1 k_2) \cdot \text{epoch})$ | $O((m+n)k_1 + k_1 k_2)$ | 隐式 |

#### 优化器对比(对 Embedding)

| 优化器 | 适用 | 显存 | 速度 |
| :--- | :--- | :--- | :--- |
| SGD | 稀疏 | 1x (W only) | 快 |
| Adam | 稠密 | 4x (W+G+M+V) | 中 |
| SparseAdam | 稀疏 | 2x (W+M) | 中 |
| Adagrad | 稀疏/稠密 | 2x | 慢 |
| LAMB | 稠密 (大 batch) | 4x | 快 |

### A.12 一页纸总结(可贴墙)

```
╔════════════════════════════════════════════════════════════╗
║  稀疏 + 低秩 = 深度学习效率的两根支柱                       ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  稀疏 (Sparse)                                             ║
║  ┌──────────────────────────────────────────────┐         ║
║  │ 定义: 大部分元素为 0 的矩阵                    │         ║
║  │ 存储: COO/CSR/CSC, 只存 (idx, val)            │         ║
║  │ 训练: SparseAdam, 只更新访问的行               │         ║
║  │ 应用: Embedding, 推荐系统, Attention mask      │         ║
║  └──────────────────────────────────────────────┘         ║
║                                                            ║
║  低秩 (Low-Rank)                                           ║
║  ┌──────────────────────────────────────────────┐         ║
║  │ 定义: A = U V^T, rank(A) ≪ min(m, n)          │         ║
║  │ 存储: (m+n)r vs mn, r 小时节省 100x            │         ║
║  │ 工具: SVD (最优), 随机 SVD (大矩阵)            │         ║
║  │ 应用: Embedding 压缩, LoRA, 矩阵分解           │         ║
║  └──────────────────────────────────────────────┘         ║
║                                                            ║
║  协同 (Hybrid)                                             ║
║  ┌──────────────────────────────────────────────┐         ║
║  │ 稀疏查表 + 低秩存储 + 量化 = 30-100x 压缩      │         ║
║  │ 工业标准: INT8 + SVD r=8 + SparseAdam           │         ║
║  └──────────────────────────────────────────────┘         ║
║                                                            ║
║  数学对应                                                   ║
║  ┌──────────────────────────────────────────────┐         ║
║  │ 稀疏 → 第 7 章 stride / 第 5 章 sparse Adam    │         ║
║  │ 低秩 → 第 3 章 SVD / 第 4 章 特征值            │         ║
║  │ 联合 → 第 6 章 微积分 / 7-系统 DLRM            │         ║
║  └──────────────────────────────────────────────┘         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```text

最后一句:**当你的显存/算力不够时,先想"能否稀疏访问",再想"能否低秩表示",最后才想"上更大的卡"**。这是搜推广和大模型时代的核心工程哲学。


## 第一部分 · 深度扩展:多视角系统展开

> 本部分是 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的全面延伸。每个章节从不同视角展开,系统补充原内容未覆盖的数学推导、工程实现、典型案例。所有新增内容自成体系,可独立阅读。

### 13. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 极限与连续深度分析

**核心议题**: 在 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的语境下,极限与连续 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 极限与连续 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 的另一个视角。


---

### 14. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 导数微分深度分析

**核心议题**: 在 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的语境下,导数微分 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 导数微分 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 的另一个视角。


---

### 15. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 积分测度深度分析

**核心议题**: 在 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的语境下,积分测度 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

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

本节系统展开了 积分测度 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 的另一个视角。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 拆解为以下几个层次:

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

信息论为"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 拆解为以下几个层次:

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

信息论为"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 拆解为以下几个层次:

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

信息论为"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 拆解为以下几个层次:

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

信息论为"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐 - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 拆解为以下几个层次:

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

信息论为"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 9 章:稀疏与低秩——Embedding 压缩·LoRA·矩阵分解推荐" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##

```
