---
title: "第11章-PyTorch基础"
tags:
  - 搜推广
  - PyTorch基础
description: "本章要点：rial 的 API 罗列,而是在搜推广视角下,重新理解 PyTorch 的 Tensor、Autograd、nn.Module、Embedding 等…"
chapter_id: 经典推荐-11
last_updated: 2026-07-14
---

# 第11章-PyTorch基础

rial 的 API 罗列,而是**在搜推广视角下,重新理解 PyTorch 的 Tensor、Autograd、nn.Module、Embedding 等核心机制**。
>
> 读完你能回答:
> - Tensor 的 view / reshape / permute / transpose 底层干了什么(哪个是"改名"、哪个是"搬家")?
> - Autograd 的计算图怎么画?为什么 `.backward()` 一句话就回传梯度?
> - Embedding 的"数学本质"不就是查表吗?那 nn.EmbeddingBag 又是什么?
> - nn.Module 的 parameter / buffer / children 三件套怎么管理?

---

## 本章概述
### 11.0 文字·公式·代码三位一体——如何真正"读懂"每一行 PyTorch

> **阅读建议**:深度学习代码中每一个操作都可以从三个角度理解:
> - **文字**:这行代码"在做什么",用最直觉的语言
> - **公式**:这行代码对应的数学形式,精确无歧义
> - **代码**:PyTorch 的实现,包括输入输出形状和注意事项
>
> 三角度一致才算"真理解"。本节用最核心的 5 个操作做示范,后续章节的每个 API 都应该这样思考。

**为什么"三位一体"思维是 PyTorch 学习的关键?**

新手看 PyTorch 文档,通常只看"代码"——`nn.Embedding(V, D)(ids)`,记下"输入 ids,输出向量",完事。结果遇到实际问题时**完全不会变通**:
- 不知道 `nn.Embedding` 背后其实是 `F.embedding(W, ids)` → `W.gather(0, ids)`
- 不知道 Embedding 层的反向传播等价于 `grad_W.scatter_add_(0, ids, grad_out)`
- 不知道 Embedding 表过大时怎么切分到多卡(模型并行)

**只有把"文字 + 公式 + 代码"三角度打通,才能真正"读懂"每一行 PyTorch**:

1. **文字** → 帮你形成"这个操作是干什么"的**直觉**
2. **公式** → 帮你建立"这个操作的数学形式"的**精确性**
3. **代码** → 帮你掌握"这个操作的具体实现和性能特性"的**工程感**

**搜推广的典型场景**:看到 `nn.Embedding(num_items, 64)(item_ids)`,要能立刻在脑子里完成三层映射:
- 文字:"把 item 的整数 ID 变成 64 维向量"
- 公式:`$E[i] = W_{i,:}$,其中 $W \in \mathbb{R}^{V \times 64}$"
- 代码:内部是 `W[ids]`,反向时 `grad_W.scatter_add_(0, ids, grad_out)`

> **阅读建议**:深度学习代码中每一个操作都可以从三个角度理解:
> - **文字**:这行代码"在做什么",用最直觉的语言
> - **公式**:这行代码对应的数学形式,精确无歧义
> - **代码**:PyTorch 的实现,包括输入输出形状和注意事项
>
> 三角度一致才算"真理解"。本节用最核心的 5 个操作做示范,后续章节的每个 API 都应该这样思考。

#### 11.0.1 Embedding 查表

| 视角 | 描述 |
| :--- | :--- |
| **文字** | 给一个整数 ID,返回一个可学习的稠密向量。本质是一个可微的字典:key=整数, value=向量 |
| **公式** | $\text{Embedding}(i) = W[i, :] \in \mathbb{R}^D$,其中 $W \in \mathbb{R}^{V \times D}$ 是权重矩阵 |
| **代码** | `nn.Embedding(V, D)(ids)` → 按行索引矩阵 |

```python
import torch
import torch.nn as nn

V, D = 1_000_000, 64   # 词表大小, 嵌入维度

# ——— 公式层面: W[i, :] ———
W = torch.randn(V, D)  # [V, D] 权重矩阵
ids = torch.tensor([123, 456, 789])     # 3 个物品 ID

# 等价写法 1: 手动矩阵乘法 (理解本质,但极慢)
one_hot = torch.zeros(3, V)
one_hot[0, 123] = 1; one_hot[1, 456] = 1; one_hot[2, 789] = 1
result_matmul = one_hot @ W            # [3, D]  O(V×D) 慢!

# 等价写法 2: 直接索引 (工程实现,O(1) 每行)
result_index = W[ids]                  # [3, D]  同样结果,快 V 倍!

# ——— PyTorch 的 nn.Embedding ———
emb = nn.Embedding(V, D, padding_idx=0)   # padding_idx: ID=0 的向量固定为 0
output = emb(ids)                          # [3, D]
# forward 内部: return self.weight[ids] 就是这一行

# ——— 反向传播特性(稀疏梯度) ———
# 前向: 只读取 W 的 3 行(indices=[123, 456, 789])
# 反向: 只有这 3 行有梯度,其余 999,997 行梯度为 0
# 工程意义: 大规模 Embedding 训练时必须用 SparseAdam 才能高效更新

print(f"输入形状: ids {ids.shape}")
print(f"输出形状: output {output.shape}")
print(f"权重矩阵: emb.weight {emb.weight.shape}")
print(f"内存占用: {V * D * 4 / 1024**3:.2f} GB (FP32)")
```

#### 11.0.2 自注意力机制 (Self-Attention)

| 视角 | 描述 |
| :--- | :--- |
| **文字** | 序列中的每个位置"问"其他所有位置:"你和我有多相关?"然后按相关性加权聚合信息 |
| **公式** | $\text{Attn}(Q,K,V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V$ |
| **代码** | `scores = q @ k.transpose(-2,-1) / sqrt(d)` 后接 `softmax` 再乘 `v` |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ——— 从公式到代码的逐步推导 ———

B, T, D = 32, 50, 64    # batch, 序列长度, 模型维度
nh, hd = 8, D // 8      # 头数, 每头维度 (64//8=8)

# 步骤 1: 线性投影生成 Q, K, V
# 公式: Q = XW_Q, K = XW_K, V = XW_V
Wq = nn.Linear(D, D, bias=False)
Wk = nn.Linear(D, D, bias=False)
Wv = nn.Linear(D, D, bias=False)

X = torch.randn(B, T, D)   # 输入序列
Q = Wq(X)   # [B, T, D]  每个位置的"提问"向量
K = Wk(X)   # [B, T, D]  每个位置的"被问"向量
V = Wv(X)   # [B, T, D]  每个位置的"价值"向量

# 步骤 2: 拆分成多头
# 公式没变,只是把 D=64 维拆成 8 头 × 8 维
Q = Q.view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]
K = K.view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]
V = V.view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]

# 步骤 3: 计算注意力分数
# 公式: scores = Q @ K^T / √d_k
# 为什么除以 √d_k?
#   Q·K 的方差 ≈ d_k,softmax 对大值极度敏感
#   除以 √d_k 使方差恢复为 1,梯度不消失
scores = Q @ K.transpose(-2, -1) / math.sqrt(hd)   # [B, nh, T, T]

# 步骤 4: 因果掩码(仅用于 Decoder / 自回归模型)
# 目的: 位置 t 只能看到 t 及之前的位置,防止信息泄漏
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float('-inf'))

# 步骤 5: softmax 权重
# 公式: attn_weights = softmax(scores, dim=-1)
# 每行和为 1,解释为"注意到不同位置的概率"
attn_weights = F.softmax(scores, dim=-1)   # [B, nh, T, T]

# 步骤 6: 加权聚合
# 公式: output = attn_weights @ V
output = attn_weights @ V                  # [B, nh, T, hd]

# 步骤 7: 合并多头,投影输出
output = output.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
Wo = nn.Linear(D, D, bias=False)
output = Wo(output)                        # [B, T, D]

print("Shape 全程追踪:")
print(f"  输入 X:         {X.shape}")
print(f"  Q/K/V (多头前): {Wq(X).shape}")
print(f"  Q/K/V (多头后): {Q.shape}")
print(f"  attn_scores:    {scores.shape}")
print(f"  attn_weights:   {attn_weights.shape}")
print(f"  输出:           {output.shape}")
```

#### 11.0.3 Layer Normalization

| 视角 | 描述 |
| :--- | :--- |
| **文字** | 对同一样本的所有特征维度做归一化,使每个样本的特征分布标准化。消除不同特征之间的尺度差异 |
| **公式** | $\text{LN}(x) = \gamma \cdot \dfrac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$,其中 $\mu, \sigma$ 在特征维度上计算 |
| **代码** | `nn.LayerNorm(D)(x)` 或 `F.layer_norm(x, [D])` |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D = 32, 50, 64
x = torch.randn(B, T, D)

# ——— 手动实现,理解公式 ———
eps = 1e-5
mu = x.mean(dim=-1, keepdim=True)       # [B, T, 1]  特征维均值
var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, T, 1]  特征维方差
gamma = torch.ones(D)                    # 可学习缩放 (初始化为 1)
beta = torch.zeros(D)                    # 可学习偏移 (初始化为 0)

x_norm_manual = gamma * (x - mu) / (var + eps).sqrt() + beta  # [B, T, D]

# ——— PyTorch 内置(等价,但更快) ———
ln = nn.LayerNorm(D)
x_norm_pytorch = ln(x)

# 验证等价性
print(f"最大误差: {(x_norm_manual - x_norm_pytorch).abs().max():.2e}")  # ≈ 1e-6

# ——— LayerNorm vs BatchNorm vs RMSNorm 对比 ———
class RMSNorm(nn.Module):
    """RMSNorm: 去掉均值项的 LayerNorm,速度快 ~5-10%"""
    def __init__(self, D, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(D))
        self.eps = eps

    def forward(self, x):
        # 公式: x / RMS(x) × gamma
        # RMS(x) = √(mean(x²))
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms

bn = nn.BatchNorm1d(D)
ln = nn.LayerNorm(D)
rn = RMSNorm(D)

x_2d = x.view(-1, D)  # BN 需要 2D 输入
print("\n三种归一化的适用场景:")
print("  BatchNorm: 批维度归一化, 适合 CNN (batch 大时效果好)")
print("  LayerNorm: 特征维度归一化, 适合 Transformer/RNN (batch 大小无关)")
print("  RMSNorm:   不减均值, 速度快, 2024 年 LLM/GenRec 主流选择")
```

#### 11.0.4 交叉熵损失

| 视角 | 描述 |
| :--- | :--- |
| **文字** | 衡量模型预测分布与真实标签分布的"差距",越小表示模型越接近正确答案。等价于最大化对数似然 |
| **公式** | $\mathcal{L} = -\sum_i y_i \log p_i = -\log p_{y^\star}$,其中 $p = \text{softmax}(\text{logits})$ |
| **代码** | `F.cross_entropy(logits, targets)` 内部做 log_softmax + nll_loss |

```python
import torch
import torch.nn.functional as F

B, C = 32, 10_000   # batch, 物品数量(多分类: 下一个 item 预测)

logits = torch.randn(B, C)         # 模型输出的原始分数(未归一化)
targets = torch.randint(0, C, (B,))  # 真实的下一个物品 ID

# ——— 方式 1: PyTorch 内置(最稳定,推荐) ———
loss1 = F.cross_entropy(logits, targets)

# ——— 方式 2: 手动实现,理解内部运算 ———
# Step 1: log_softmax (数值稳定版的 log(softmax(x)))
log_probs = F.log_softmax(logits, dim=-1)   # [B, C]

# Step 2: NLL Loss (负对数似然)
# 只取目标类对应位置的 log 概率,取负值
loss2 = F.nll_loss(log_probs, targets)

# ——— 方式 3: 极不稳定写法(教学反例) ———
# 千万不要这样写!
probs = torch.softmax(logits, dim=-1)                # ❌ 可能数值溢出
log_probs_bad = torch.log(probs)                      # ❌ log(0) = -inf 灾难
loss3 = -log_probs_bad[torch.arange(B), targets].mean()

print(f"loss1 (F.cross_entropy):  {loss1.item():.4f}")
print(f"loss2 (手动 log_softmax): {loss2.item():.4f}")
print(f"loss3 (危险写法):         {loss3.item():.4f}")
# 三者数值相同,但 loss3 在 logits 很大时会产生 NaN!

# ——— 为什么 log_softmax 比 log(softmax) 稳定? ———
logits_extreme = torch.tensor([1000., 1001., 999.])

# 不稳定:
probs_bad = torch.exp(logits_extreme)   # inf! overflow
# log(inf) = inf, 梯度计算崩溃

# 稳定:log-sum-exp 技巧
# log softmax(x_i) = x_i - log(Σ exp(x_j))
#                  = x_i - c - log(Σ exp(x_j - c))  , c = max(x_j)
log_probs_good = F.log_softmax(logits_extreme, dim=0)
print(f"\n极端 logits 下 log_softmax: {log_probs_good}")  # 正常值,无 inf/nan
```

#### 11.0.5 二元交叉熵 (CTR 预估的损失函数)

| 视角 | 描述 |
| :--- | :--- |
| **文字** | 对每个样本独立预测"会点击(1)还是不会点击(0)"的概率,衡量预测与真实标签的差距 |
| **公式** | $\mathcal{L} = -\bigl[y\log\sigma(\hat{y}) + (1-y)\log(1-\sigma(\hat{y}))\bigr]$ |
| **代码** | `F.binary_cross_entropy_with_logits(logits, labels)` (logits 未经 sigmoid) |

```python
import torch
import torch.nn.functional as F

B = 1024
logits = torch.randn(B)           # 模型输出 (未 sigmoid)
labels = torch.randint(0, 2, (B,)).float()  # 0 或 1

# ——— 正确做法:BCEWithLogitsLoss ———
loss_correct = F.binary_cross_entropy_with_logits(logits, labels)

# ——— 手动推导,理解公式 ———
# 内部等价于:
sigma = torch.sigmoid(logits)     # p ∈ (0, 1)
loss_manual = -(labels * torch.log(sigma + 1e-8) +
                (1 - labels) * torch.log(1 - sigma + 1e-8)).mean()

# ——— 错误做法 (数值不稳定) ———
# loss_bad = F.binary_cross_entropy(torch.sigmoid(logits), labels)
# 原因:sigmoid 输出可能非常接近 0 或 1,log(0) = -inf

print(f"BCEWithLogitsLoss: {loss_correct.item():.4f}")
print(f"手动计算:           {loss_manual.item():.4f}")

# ——— 正负样本不均衡时加权 ———
# 搜推广中正负样本比约 1:100, 正样本需要更大权重
pos_weight = torch.tensor([100.0])   # 正样本权重 = 负负比
loss_weighted = F.binary_cross_entropy_with_logits(
    logits, labels, pos_weight=pos_weight
)
```

---

### 11.1 一切都从 Tensor 开始

#### 11.1.1 Tensor 是什么?——比 NumPy 更强大的数组

```python
import torch

# 从 numpy 创建
import numpy as np
np_arr = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(np_arr)         # 共享内存,np_arr 改 t 也改!
# 千万注意:如果你用 .from_numpy,原 numpy 改了 tensor 也会跟着变!

# 从 list 创建(常用)
t = torch.tensor([[1, 2], [3, 4]])   # dtype=torch.int64
t = torch.tensor([1.0, 2.0])         # dtype=torch.float32

# 特殊初始化
t0 = torch.zeros(3, 4)               # 全 0
t1 = torch.ones(3, 4)                # 全 1
te = torch.empty(3, 4)               # 未初始化(可能存着垃圾值,快但危险)
ti = torch.eye(4)                    # 单位矩阵
tr = torch.randn(3, 4)               # 标准正态分布
tl = torch.linspace(0, 1, 10)        # [0.0, 0.111, ..., 1.0]
```

**搜推广的坑:一定要显式指明 dtype**

```python
# 在搜推广里,存储物品 ID 用 int32 或者 int64
# 如果类型不对,后面会引发隐式类型转换甚至报错
item_ids = torch.tensor([1, 2, 3])               # 默认 int64
item_ids_fp = torch.tensor([1, 2, 3], dtype=torch.float32)  # ⚠️ 不对!后面 Embedding 会报错

# 搜推广常用 dtype:
# - 离散特征 ID: torch.long (nn.Embedding 要求)
# - 连续特征: torch.float32
# - 权重: torch.float32 / float16 / bfloat16
# - 标签: torch.float32 (BCELoss) 或 torch.long (CrossEntropyLoss)
```

**dtype 对比**:

| dtype | 说明 | 什么时候用 |
| :--- | :--- | :--- |
| `torch.float32` | 标准 32 位浮点 | **默认选择**,大多数网络用这个 |
| `torch.float16` | 半精度,2 字节 | 显存不够时,混合精度训练 |
| `torch.bfloat16` | 同半精度但指数范围更大 | 混合精度训练的更好选择 |
| `torch.int64` (long) | 64 位整数 | **Embedding 输入的唯一选择** |
| `torch.int32` | 32 位整数 | 存储 ID 做索引(Embedding 输入仍要转 long) |
| `torch.bool` | 布尔 | mask / attention_mask |

#### 11.1.2 设备(device):CPU vs GPU

```python
# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建时指定设备
t = torch.tensor([1, 2, 3], device="cuda")

# 移动设备(这个操作很慢,尽量少用)
t.to("cuda")             # 拷贝到 GPU
t.to("cpu")              # 拷贝回 CPU

# 模型也一样
model = MyModel().to(device)

# 搜索里的常见写法
batch = {k: v.to(device) for k, v in batch.items()}
```

**搜推广踩坑:GPU 显存 OOM 常发生在设备迁移**

```python
# 每次 .to(device) 都产生一次 CPU→GPU 拷贝,B=4096 时一次拷贝几十 MB
# 如果训练循环里不小心在 CPU 上做了前向... (例如忘了 to(device))
# 再显式搬一次,隐式又多了一次拷贝
# 全局要统一管理:batch 在 DataLoader 就 pin_memory → 到 GPU 只走一次
```

#### 11.1.3 形状(shape):维度操作是搜推广的日常

**搜推广常见的张量形状**:

```python
batch_size = 1024
seq_len = 50
n_features = 32
n_items = 10_000_000

# 离散特征 ID 向量: [B] (每个用户 1 个 ID)
user_ids = torch.randint(0, n_items, (batch_size,))          # [1024]

# 离散特征 ID 序列: [B, T] (用户历史点击序列)
history_ids = torch.randint(0, n_items, (batch_size, seq_len))  # [1024, 50]

# 连续特征矩阵: [B, F] (F 个连续特征)
dense_feats = torch.randn(batch_size, 8)                     # [1024, 8]

# Embedding 后的序列: [B, T, D] (D 是 embedding 维度)
seq_emb = torch.randn(batch_size, seq_len, 32)               # [1024, 50, 32]

# 模型输出: [B] 或 [B, 1] (点击概率)
predictions = torch.sigmoid(torch.randn(batch_size))          # [1024]
```

#### 11.1.4 Tensor 的内存布局:一副扑克牌的"切牌"哲学(view/stride/contiguous)

> **这是你学 PyTorch 最最重要的一个概念,没有之一。** 理解了 stride,你就理解了为什么 view 有时报错、为什么 transpose 是 O(1)、为什么 contiguous() 那么贵。

**Tensor 在内存里不是"多维数组",而是一段连续的一维内存 + 一套"看它的角度"。**

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 内存: [1, 2, 3, 4, 5, 6]  ← 按行优先连续排成一条线
```

**stride = "在每一个维度上,前进 1 步需要跨过几个元素"**

**公式**:给定 stride $s = (s_0, s_1, \ldots, s_{n-1})$ 和多维索引 $(i_0, i_1, \ldots, i_{n-1})$，元素在底层一维内存的偏移量为：

$$\text{offset}(i_0, i_1, \ldots, i_{n-1}) = \text{storage\_offset} + \sum_{k=0}^{n-1} i_k \cdot s_k$$

**连续性判断**：当且仅当 $s_k = s_{k+1} \cdot \text{shape}_{k+1}$（对所有 $k < n-1$）时，张量是 contiguous。

```python
print(x.shape)   # (2, 3)
print(x.stride()) # (3, 1)
# 解读:
#   dim=0(行):从第 0 行到第 1 行,要跨过 3 个元素(即 1,2,3)
#   dim=1(列):从第 0 列到第 1 列,要跨过 1 个元素(即隔壁)
```

**用 stride 来解释所有维度变换操作:**

```python
x = torch.randn(2, 3, 4)       # shape=(2,3,4), stride=(12,4,1)
# 内存:连续排列,从 0~23

# ---- view:只改 shape 和 stride,不改内存 ----
y = x.view(6, 4)               # shape=(6,4), stride=(4,1)
# 内存不变!只是"看"的方式变了:原来 2 层楼,现在当成 6 层楼
# 要求:x 在内存中是连续的(contiguous)

# ---- transpose:交换 stride ----
y = x.transpose(0, 2)          # shape=(4,3,2), stride=(1,4,12)
# 没有复制数据!只是把 stride 重新排列了
# 原来 (12,4,1) → (1,4,12):dim0 前进 1 步只需跨 1 个元素了
# 这就是 transpose O(1) 的原因——只改了"怎么看"

# ---- permute:任意重排 stride ----
y = x.permute(2, 0, 1)         # shape=(4,2,3), stride=(1,12,4)
# 同样 O(1),只改 stride

# ---- contiguous:物理上重新排列,变成连续 ----
y = x.transpose(0, 2)         # 现在 y 的 stride=(1,4,12),非连续!
y = y.contiguous()             # 创建新内存,按 y 的顺序重新排列
# contiguous() 很贵!因为它涉及一次完整的内存拷贝 O(n)
```

**连续(contiguous) vs 不连续(non-contiguous)的判断标准**:

```python
x = torch.randn(2, 3, 4)
print(x.is_contiguous())          # True

y = x.transpose(0, 1)            # (3,2,4)
print(y.is_contiguous())          # False! 因为内存不再是行优先

# 判断公式:从最后一个 dim 到第一个 dim,
# stride[i] == stride[i+1] * shape[i+1]  (对于第 i 维)
# x: stride=(12,4,1), 4 == 1*4 ✓, 12 == 4*3 ✓ → continuous
# y: stride=(4,12,1), 12 != 1*2 ✗ → NOT continuous
```

**搜推广中 stride 最常见的坑**:

```python
# 坑 1:transpose 后想 view,报错!
x = torch.randn(32, 50, 128)            # [B, T, D], continuous
y = x.transpose(1, 2)                   # [B, D, T], non-contiguous
# z = y.view(32, 50*128)                # RuntimeError: view size is not compatible!
# 因为 y 内存不连续,view 不能直接"重排"

# 正确做法:
z = y.contiguous().view(32, -1)          # 先 contiguous(复制),再 view

# 或者用 reshape(自动 contiguous)
z = y.reshape(32, -1)                    # reshape 内部会检查 contiguity,自动执行 contiguous

# 坑 2:在推理循环里反复 transpose + contiguous
# 每次 contiguous 都是 O(n) 拷贝,性能杀手!
# 正确:在循环外一次性 contiguous,循环内只做 view
x = x.transpose(1, 2).contiguous()       # 一次性
for i in range(1000):
    y = x.view(batch_size, -1)           # 循环内只 view,不拷贝

# 坑 3:怀疑显存异常暴涨?
# 检查是不是不小心创建了多次 contiguous 副本
def memory_debug(tensor):
    print(f"shape={tensor.shape}, stride={tensor.stride()}, "
          f"contiguous={tensor.is_contiguous()}, "
          f"storage_size={tensor.storage().size()}")    # 底层存储大小
```

**stride 的物理意义总结**:

| 操作 | 时间复杂度 | 是否复制数据 | 搜推广建议 |
| :--- | :--- | :--- | :--- |
| `view` | O(1) | ❌ 不改内存 | **首选**,但需 contiguous |
| `reshape` | O(1)~O(n) | 不一定 | 安全,自动判断 |
| `transpose` | O(1) | ❌ | 频繁调用,不复制 |
| `permute` | O(1) | ❌ | 多维度重排 |
| `contiguous` | **O(n)** | ✅ 必复制 | 少用,用完就别再用 |
| `flatten` | O(1)~O(n) | 看情况 | 新代码用 `reshape(-1,)` |
| `squeeze/unsqueeze` | O(1) | ❌ | 随便用,免费 |

#### 11.1.5 索引与切片:为什么 Tensor 的索引和 Python list 不一样?

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 基本索引
x[0]            # tensor([0, 1, 2, 3]), 取第 0 行
x[0, 1]         # tensor(1), 取第 0 行第 1 列
x[0, :]         # 同上,和 x[0] 等价
x[:, 1]         # tensor([1, 5, 9]), 取第 1 列

# 布尔索引(搜推广高频:mask)
mask = x > 5
x[mask]         # tensor([6, 7, 8, 9, 10, 11])

# 花式索引(按索引数组取)
indices = torch.tensor([0, 2])
x[indices]      # tensor([[0, 1, 2, 3], [8, 9, 10, 11]])

# 搜推广高频:gather(按索引从序列中取值)
seq = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80]
])  # [2, 4]
idx = torch.tensor([[2], [1]])
seq.gather(dim=1, index=idx)   # tensor([[30], [60]])
# 在注意力机制里,需要从整个序列中按 attention 分位置取信息时常用
```

#### 11.1.5 维度变换:view / reshape / transpose / permute 的区别

**这是你学 PyTorch 最该搞清楚的四个函数之一**。

```python
x = torch.randn(2, 3, 4)    # [2, 3, 4]

# 1. view:返回新视图(shape 变,内存不变,要求内存连续)
y = x.view(2, 12)           # [2, 12],成功
y = x.view(6, 4)            # [6, 4],成功
y = x.view(-1, 4)           # [6, 4],-1 表示自动推算

# 2. reshape:功能同 view,但不要求连续(view 报错时 reshape 可以)
# 一般建议直接用 reshape(更安全,性能差不大的)
y = x.reshape(6, 4)         # 和 view 一样

# 3. transpose:交换两个维度
y = x.transpose(0, 1)       # [3, 2, 4],dim0 和 dim1 互换位置

# 4. permute:任意调整维度顺序(比 transpose 更通用)
y = x.permute(2, 0, 1)      # [4, 2, 3],dim0→原来 dim2,dim1→原来 dim0...

# 5. squeeze / unsqueeze:去掉/增加维度 1
x = torch.randn(1, 3, 1, 4) # [1, 3, 1, 4]
x.squeeze()                 # [3, 4],去掉所有大小为 1 的维度
x.unsqueeze(0)              # [1, 1, 3, 1, 4],在位置 0 插入 1 维
x.unsqueeze(-1)             # [1, 3, 1, 4, 1],在最后插入 1 维
```

**搜推广的经典用法**(Transformer Attention 里每个张量的维度变化):

```python
# 输入: [B, T, D] (batch, 序列长, 特征维)
x = torch.randn(32, 50, 128)

# 多头注意力的 QKV 变换
Wq = torch.randn(128, 128)

# 1. Q = x @ Wq → [B, T, D]
q = x @ Wq

# 2. reshape 成多头: [B, T, nh, hd]
q = q.view(32, 50, 8, 16)         # nh=8, hd=16

# 3. 转置成 [B, nh, T, hd] — 这是 attention 计算的标准格式
q = q.permute(0, 2, 1, 3)         # [32, 8, 50, 16]

# 4. 注意力分数: q @ k^T → [B, nh, T, T]
scores = q @ q.transpose(-2, -1) / (16 ** 0.5)

# 5. 输出: [B, nh, T, hd] → permute → [B, T, nh, hd] → reshape → [B, T, D]
out = scores @ v                             # [32, 8, 50, 16]
out = out.permute(0, 2, 1, 3).reshape(32, 50, 128)  # [32, 50, 128]
```

#### 11.1.6 广播机制(Broadcasting):自动对齐形状

**公式**:两个张量 $A$ (shape: $d_1 \times \ldots \times d_m$) 和 $B$ (shape: $e_1 \times \ldots \times e_n$) 的广播规则：

$$\text{从右对齐,若 } d_i = e_i \text{ 或 } d_i = 1 \text{ 或 } e_i = 1 \Rightarrow \text{输出第 } i \text{ 维} = \max(d_i, e_i)$$

$$\text{否则广播失败(维度不兼容)}$$

**文字**：广播的物理意义是"复制数据但不实际分配内存"，通过 stride=0 的技巧，将 size=1 的维度视为已复制到目标 size。这也是广播是 $O(1)$ 而 `expand` + `contiguous` 才真正复制的原因。

```python
# 关键规则:两个张量从右到左对齐,维度要么相等,要么是 1,要么不存在
# 最终形状 = 逐位取 max

# 例子 1: [3, 1] + [1, 4] → [3, 4]
a = torch.tensor([[1], [2], [3]])      # [3, 1]
b = torch.tensor(`10, 20, 30, 40`)   # [1, 4]
c = a + b                                # [3, 4]
# [[11, 21, 31, 41],
#  [12, 22, 32, 42],
#  [13, 23, 33, 43]]

# 例子 2: 搜推广常见——序列嵌入 + 位置编码
seq_emb = torch.randn(32, 50, 128)   # [B, T, D]
pos_emb = torch.randn(1, 50, 128)     # [1, T, D],广播到 B
output = seq_emb + pos_emb             # [32, 50, 128]

# 例子 3: mask 广播到 emb 维度
mask = torch.tensor(`1, 1, 0, 0`)   # [1, T],B=1
emb = torch.randn(2, 4, 8)            # [B, T, D]
masked = emb * mask.unsqueeze(-1)     # [B, T, 1] → [B, T, D]
```

**搜推广的坑:不可广播的形状**

```python
a = torch.randn(32, 50, 128)          # [B, T, D]
b = torch.randn(32, 128)             # [B, D]
# a + b → 报错!因为 T 维对不上(50 vs 空)
# 正确:用 unsqueeze 变成 [B, 1, D]
b_expanded = b.unsqueeze(1)           # [B, 1, D]
a + b_expanded                        # 广播成 [B, T, D]
```

---

### 11.2 Tensor 运算:从简单到高级

**为什么 Tensor 运算是搜推广 PyTorch 编程的"基本功"?**

PyTorch 80% 的代码是 Tensor 操作——`tensor.matmul`、`tensor.reshape`、`tensor.softmax`、`tensor.scatter_add`。这些操作看似简单,实则每一个都有"隐藏的坑":
- 矩阵乘法有 3 个 API(`mm`、`bmm`、`matmul`),**用错效率差 100 倍**
- 维度重排有 `view`、`reshape`、`permute`、`transpose`,**用错直接报错**
- Broadcasting 规则看起来"自动",**理解错维度对齐,bug 极难定位**

**这一节会按"3 大类"组织**:
1. **矩阵乘法**:`mm` / `bmm` / `matmul` 的取舍
2. **元素运算**:`+`、`*`、`==`、`where` 和 Broadcasting 规则
3. **归约运算**:`sum` / `mean` / `max` 的 dim 参数语义

**搜推广的"Tensor 运算 3 大原则"**:
1. **矩阵乘法用 `matmul` 而不是 `*`**:元素乘积 `*` 不做"点积求和",效率差 100 倍
2. **明确指定 dim 参数**:`tensor.sum(dim=1)` 比 `tensor.sum()` 慢 10 倍?不是,但**不指定 dim 就把所有元素求和**,通常不是你要的
3. **避免 for 循环**:`for i in range(B): ...` 在 Python 里慢,**用 Tensor 操作向量化**(向量化是 PyTorch 性能的核心)

#### 11.2.1 矩阵乘法三剑客(搜推广必考)

**文字**:矩阵乘法是深度学习最核心的计算原语。PyTorch 提供三个函数覆盖从简单到复杂的场景：`mm` 处理 2D、`bmm` 处理带 Batch 的 3D、`matmul` 通吃任意维度并自动广播。

**公式**:

$$\text{mm: } C_{ij} = \sum_{k=1}^{K} A_{ik} \cdot B_{kj}, \quad A \in \mathbb{R}^{M \times K},\ B \in \mathbb{R}^{K \times N},\ C \in \mathbb{R}^{M \times N}$$

$$\text{bmm: } C_{b,i,j} = \sum_{k=1}^{K} A_{b,i,k} \cdot B_{b,k,j}, \quad \text{Batch 内各样本独立计算}$$

$$\text{注意力得分(matmul):} S = QK^T / \sqrt{d_k}, \quad Q \in \mathbb{R}^{B \times n_h \times T \times d_k},\ K \in \mathbb{R}^{B \times n_h \times d_k \times T}$$

**代码**:

```python
# 1. torch.mm:严格 2 维矩阵相乘
# 场景:全连接层的前向计算
x = torch.randn(1024, 128)   # [B, D_in]
w = torch.randn(128, 64)     # [D_in, D_out]
y = torch.mm(x, w)           # [1024, 64]
# ⚠️ mm 只接受 2 维,传入 3 维就报错

# 2. torch.bmm:带 Batch 维的 3 维矩阵相乘(搜推广极度高频)
# 场景:多序列的注意力计算——每个样本独立算各自的 attention
q = torch.randn(32, 8, 50, 16)   # [B, nh, T, hd]
k = torch.randn(32, 8, 16, 50)   # [B, nh, hd, T]
scores = torch.bmm(               # ❌ 4 维,bmm 只接受 3 维
    q.view(-1, 50, 16),           # [B*nh, T, hd]
    k.view(-1, 16, 50)            # [B*nh, hd, T]
).view(32, 8, 50, 50)             # [B, nh, T, T]

# 用 matmul 更简单(推荐):
scores = torch.matmul(q, k)       # [32, 8, 50, 50],自动处理

# 3. torch.matmul:最通用,支持广播,推荐日常用
v1 = torch.randn(32, 50, 128)    # [B, T, D]
v2 = torch.randn(128, 64)        # [D, D_out]
y = torch.matmul(v1, v2)         # [B, T, 64],自动广播

# 搜推广经典场景:双塔模型的内积(向量点击率计算)
user_emb = torch.randn(32, 64)   # [B, D]
item_emb = torch.randn(32, 64)   # [B, D]
# method 1:点积
score = (user_emb * item_emb).sum(dim=-1)  # [B]
# method 2:bmm 矩阵乘法
score = torch.bmm(
    user_emb.unsqueeze(1),       # [B, 1, D]
    item_emb.unsqueeze(-1)       # [B, D, 1]
).squeeze(-1).squeeze(-1)        # [B]
# method 3:matmul
score = torch.matmul(user_emb.unsqueeze(1), item_emb.unsqueeze(-1)).squeeze()
```

**决策表**:

| 函数 | 输入维度 | 输出形状 | 搜推广场景 |
| :--- | :--- | :--- | :--- |
| `torch.mm` | (M,N)×(N,P) | (M,P) | 最终的 MLP 输出层 |
| `torch.bmm` | (B,M,N)×(B,N,P) | (B,M,P) | **注意力的 QK^T / 双塔相似度** |
| `torch.matmul` | 任意 ≥2 维 | 自动推断 | **通用首选** |

#### 11.2.2 归约运算(sum/mean/max/min)

**文字**:归约运算将张量沿某维度压缩为标量或低维张量。在搜推广中，最常见的是对**变长序列做 Masked Mean Pooling**，即用 0/1 掩码过滤 padding 位，只对有效位置求均值，这是序列建模的核心操作。

**公式**:

$$\text{Masked Mean Pooling:} \quad \hat{x} = \frac{\sum_{t=1}^{T} m_t \cdot x_t}{\sum_{t=1}^{T} m_t + \varepsilon}, \quad m_t \in \{0, 1\}$$

其中 $m_t = 0$ 表示 padding 位，$\varepsilon$ 防止除以零（空序列时返回零向量）。

**代码**:

```python
x = torch.randn(32, 50, 128)   # [B, T, D]

x.sum()                         # 所有元素之和(标量)
x.sum(dim=-1)                   # [B, T],沿特征维求和
x.mean(dim=1)                   # [B, D],沿序列维求均值(序列池化)
x.max(dim=1).values             # [B, D],沿序列维取最大值(最大池化)
x.min(dim=-1).values            # [B, T],沿特征维取最小值

# 搜推广:对变长序列求均值(带 mask)
x = torch.randn(32, 50, 128)       # [B, T, D]
mask = torch.ones(32, 50, 1)        # [B, T, 1]
mask[:, 30:, :] = 0                 # 后 20 个是 padding

masked_sum = (x * mask).sum(dim=1)  # [B, D],padding 贡献 0
masked_cnt = mask.sum(dim=1)        # [B, 1],每个样本有效长度
pooled = masked_sum / masked_cnt.clamp(min=1)  # [B, D],平均池化
```

#### 11.2.3 高级索引与 gather/scatter

**公式**:

$$\text{gather: } \text{out}[i,j] = A[i,\ \text{idx}[i,j]] \quad \text{(按索引矩阵取值)}$$

$$\text{scatter\_add: } A[\text{idx}[i,j],\ j] \mathrel{+}= \text{src}[i,j] \quad \text{(gather 的逆操作,反向传播等价)}$$

**代码**:

```python
# torch.gather:从序列中"按索引取出特定位置的值"
# 场景:在 Transformer 输出中,拿到每个样本[CLS]位置的向量
seq_out = torch.randn(32, 50, 128)           # [B, T, D]
cls_indices = torch.zeros(32, 1, dtype=torch.long)  # 每行都是 0([CLS]位置)
cls_vec = seq_out.gather(dim=1, index=cls_indices.unsqueeze(-1).expand(-1, -1, 128))
# cls_vec: [32, 1, 128]

# scatter_:反向操作,把值放回指定位置
# 场景:构造稀疏特征的 Multi-hot 向量
indices = torch.tensor([[0, 2], [1, 3]])      # 每行 2 个位置要填 1
out = torch.zeros(2, 5, dtype=torch.float32)  # [B=2, dim=5]
out.scatter_(dim=1, index=indices, value=1)
# out: [[1, 0, 1, 0, 0],
#       [0, 1, 0, 1, 0]]
```

#### 11.2.4 cat vs stack 再强调一次

**文字**：`cat` 沿已有维度拼接，维度数不变但对应轴的大小相加；`stack` 在指定位置新增一个维度，要求所有输入形状完全一致。搜推广中 `cat` 用于多字段特征拼接，`stack` 用于 FM/CIN 等需要保留独立 field 向量的交叉层。

**公式**：

$$\text{cat}([A^{m \times d},\ B^{n \times d}],\ \text{dim}=0) \rightarrow \mathbb{R}^{(m+n) \times d}$$

$$\text{stack}([A^{m \times d},\ B^{m \times d}],\ \text{dim}=1) \rightarrow \mathbb{R}^{m \times 2 \times d}$$

```python
# cat:沿已有维度拼接(不新增维度)
a = torch.randn(32, 16)
b = torch.randn(32, 32)
c = torch.cat([a, b], dim=-1)   # [32, 48]

# stack:新增维度后堆叠(每个输入形状必须完全相同)
a = torch.randn(32, 16)
b = torch.randn(32, 16)
c = torch.stack([a, b], dim=1)  # [32, 2, 16]
```

**搜推广经典**:做多特征拼接(cat) vs 做特征交叉(stack):

```python
# 拼接:你有的用户特征全部拼起来送给 MLP
user_emb = torch.randn(32, 32)     # 用户 ID embedding
item_emb = torch.randn(32, 32)     # 商品 ID embedding
history_emb = torch.randn(32, 16)   # 历史序列池化
dense = torch.randn(32, 8)          # 连续特征

all_features = torch.cat([
    user_emb, item_emb, history_emb, dense
], dim=-1)                          # [32, 88] ← 88=32+32+16+8

# stack:用在 FM 的特征交叉层,每个特征向量独立保留
feat_vecs = []
for emb in all_embeddings:
    feat_vecs.append(emb)   # 每个是 [32, D]
feat_matrix = torch.stack(feat_vecs, dim=1)  # [32, n_feats, D]
```

#### 11.2.5 einsum:爱因斯坦求和约定

**文字**：爱因斯坦求和约定（Einstein summation）通过下标字符串描述张量运算：重复出现的下标表示沿该轴收缩（求和），仅在输出端出现的下标表示保留轴。用一行代替复杂的 `permute + bmm + reshape` 组合。

**公式**：

$$y_{bh} = \sum_{d} x_{bd}\ w_{dh} \quad \Rightarrow \quad \texttt{"bd,dh->bh"}$$

$$A_{bnts} = \sum_{d} Q_{bntd}\ K_{bnsd} \quad \Rightarrow \quad \texttt{"bntd,bnsd->bnts"} \quad (\text{注意力})\ QK^T$$

```python
# einsum 可以用一行代号写出各种矩阵运算,不用去犹豫 permute 和 reshape

# 矩阵乘法:q @ k^T → [B, nh, T, T]
attention = torch.einsum("b n t d, b n s d -> b n t s", q, k)

# 双塔内积:
score = torch.einsum("b d, b d -> b", user_emb, item_emb)

# 带偏置的线性层:
y = torch.einsum("b d, d h -> b h", x, w) + b

# 搜推广学 einsum 的最大理由:
# 别人用 bmm + permute 写 3 行,你 1 行就好,且可读性超强
```

---

#### 11.2.6 PyTorch 高频函数完整速查——搜推广必备工具箱

> **阅读方式**: 每个函数用"文字 + 公式 + 代码"三位一体呈现。遇到不熟悉的函数先看"文字",确认是否需要,再看"代码"学用法。

##### 一、条件选择与掩码操作

```python
import torch
import torch.nn.functional as F

x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
mask = x > 0   # tensor([True, False, True, False, True])

# ——— torch.where ———
# 文字: 按条件逐元素选择 — cond=True 取 a, cond=False 取 b
# 公式: y_i = a_i if cond_i else b_i
# 代码:
y = torch.where(mask, x, torch.zeros_like(x))
# tensor([1., 0., 3., 0., 5.])  负值被置为 0
# 等价于: x * mask (但 torch.where 更通用)

# 搜推广场景: 把 padding 位置的 attention score 置为 -inf
scores = torch.randn(32, 50, 50)    # [B, T, T]
pad_mask = torch.zeros(32, 50, dtype=torch.bool)   # True=padding
pad_mask[:, 30:] = True             # 后 20 个位置是 padding
scores_masked = torch.where(
    pad_mask.unsqueeze(1),          # [B, 1, T] 广播
    torch.full_like(scores, float('-inf')),
    scores
)

# ——— Tensor.masked_fill ———
# 文字: 把 mask=True 的位置填充为固定值(比 where 更简洁)
# 公式: y_i = value if mask_i else x_i
# 代码:
scores_masked2 = scores.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
# 与上面等价,但语义更清晰

# ——— Tensor.masked_select ———
# 文字: 选出 mask=True 的所有元素,返回 1D 张量
# 公式: y = {x_i | mask_i = True}
valid_scores = x.masked_select(mask)   # tensor([1., 3., 5.])
# 注意: 输出是 1D 的,不保留形状

# ——— torch.clamp / torch.clip ———
# 文字: 把值截断到 [min, max] 范围内(两者等价,clip 是别名)
# 公式: y_i = max(min, min(x_i, max))
# 代码:
x_clamped = torch.clamp(x, min=0.0, max=3.0)
# tensor([1., 0., 3., 0., 3.])  负值变 0,>3 的变 3
# 搜推广场景: 归一化时防止异常值
norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # 防止除以 0
```

##### 二、排序、检索与唯一值

```python
x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])

# ——— torch.topk ———
# 文字: 找 top-k 个最大(或最小)值及其索引
# 公式: 返回 (values, indices) — values 是排好序的
# 代码:
values, indices = torch.topk(x, k=3)
# values: tensor([9., 5., 4.])  indices: tensor([5, 4, 2])

# 搜推广场景: 从 logits 中取 top-k 候选物品
item_scores = torch.randn(32, 1_000_000)   # [B, N_items]
top_scores, top_ids = torch.topk(item_scores, k=100, dim=-1)
# top_ids: [32, 100] — 每个用户的 top-100 候选

# ——— torch.argsort ———
# 文字: 返回排好序的索引(不返回值本身)
# 公式: i 使得 x[argsort[i]] 是第 i 小的元素
# 代码:
sorted_idx = torch.argsort(x, descending=True)
# tensor([5, 4, 2, 0, 6, 1, 3])  按降序排列的索引

# ——— torch.sort ———
# 文字: 返回 (排好序的值, 对应原始索引)
sorted_vals, sorted_idx = torch.sort(x)
# sorted_vals: tensor([1., 1., 2., 3., 4., 5., 9.])

# ——— torch.unique ———
# 文字: 去重,并可选返回元素计数
# 代码:
unique_vals, counts = torch.unique(
    torch.tensor([1, 2, 2, 3, 3, 3]), return_counts=True
)
# unique_vals: [1, 2, 3],  counts: [1, 2, 3]

# 搜推广场景: 统计 batch 中每个物品被曝光的次数(用于采样权重调整)
item_ids = torch.randint(0, 1000, (512,))
unique_ids, freq = torch.unique(item_ids, return_counts=True)

# ——— torch.searchsorted ———
# 文字: 二分查找 — 在有序数组中找插入位置
# 公式: 返回 i 使得 sorted[i-1] ≤ query < sorted[i]
# 代码:
sorted_arr = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
queries = torch.tensor([0.0, 4.0, 6.0, 10.0])
positions = torch.searchsorted(sorted_arr, queries)
# positions: tensor([0, 2, 3, 5])

# 搜推广场景: 按频率区间采样(流行度降采样)
cumulative_freq = torch.cumsum(item_freq.float(), dim=0)
cumulative_freq /= cumulative_freq[-1]   # 归一化为 [0,1]
rand_samples = torch.rand(1000)
sampled_bins = torch.searchsorted(cumulative_freq, rand_samples)
```

##### 三、散点聚合与索引操作

```python
# ——— torch.scatter_add ———
# 文字: 把 src 的值按 index 指定的位置"累加"到 out 中
# 公式: out[index[i][j]] += src[i][j]  (对所有 i,j)
# 代码:
B, T, D = 32, 50, 64
item_emb = torch.randn(B, T, D)   # 序列中每个位置的 Embedding
category = torch.randint(0, 5, (B, T))  # 每个位置属于哪个类目(0-4)

# 把同类目的 Embedding 求和聚合(Pooling by Category)
n_categories = 5
cate_emb = torch.zeros(B, n_categories, D)
index = category.unsqueeze(-1).expand(-1, -1, D)  # [B, T, D]
cate_emb.scatter_add_(dim=1, index=index, src=item_emb)
# cate_emb[b, c, :] = Σ_{t: category[b,t]==c} item_emb[b, t, :]

# ——— torch.index_select ———
# 文字: 沿某维度选取指定索引的切片
# 公式: y = x.index_select(dim, index) → y[i] = x[index[i]]
# 代码:
emb_table = torch.randn(10_000, 64)   # [V, D]
selected_ids = torch.tensor([0, 100, 999, 5000])
selected_embs = torch.index_select(emb_table, dim=0, index=selected_ids)
# [4, 64] — 等价于 emb_table[selected_ids] 但支持梯度

# ——— torch.gather ———
# 文字: 按 index 矩阵逐元素取值(比 index_select 更灵活)
# 公式: out[i][j] = input[i][index[i][j]]  (dim=1 时)
# 代码:
logits = torch.randn(32, 10)        # [B, C]
top3_idx = logits.topk(3, dim=-1).indices  # [B, 3]  每个样本的 top-3 索引
top3_scores = logits.gather(dim=1, index=top3_idx)  # [B, 3]  对应的分数

# 搜推广场景: 取目标物品的 logit(用于 BPR 排序损失)
pos_item_id = torch.randint(0, 10, (32,))   # [B]
pos_scores = logits.gather(1, pos_item_id.unsqueeze(1)).squeeze(1)  # [B]
```

##### 四、激活函数与数值操作

```python
x = torch.randn(32, 64)

# ——— 激活函数三位一体对比 ———

# ReLU: y = max(0, x)
# 问题: 负值梯度为 0(神经元死亡)
relu_out = F.relu(x)

# GELU: y = x · Φ(x), Φ 是标准正态 CDF
# 公式近似: y ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
# 比 ReLU 平滑,不死亡,BERT/GPT 的默认选择
gelu_out = F.gelu(x)

# SiLU (Swish): y = x · sigmoid(x)
# 公式: y = x / (1 + exp(-x))
# 平滑,自门控,SwiGLU 的基础
silu_out = F.silu(x)    # 等价于 x * torch.sigmoid(x)

# Softplus: y = log(1 + exp(x))  (ReLU 的平滑版)
# 公式: 当 x>>0 时 ≈ x;当 x<<0 时 ≈ 0
softplus_out = F.softplus(x)

# ——— 数值操作 ———

# torch.abs: |x|
# torch.sign: sign(x) ∈ {-1, 0, 1}
# torch.sqrt: √x  (注意负数会产生 nan, 用 .clamp(min=0) 前置)
# torch.log: ln(x)  (注意 x=0 时产生 -inf, 用 clamp(min=1e-8))
# torch.exp: e^x  (注意大值 overflow, 用 F.log_softmax 替代 log(softmax))

# 安全写法
safe_log = torch.log(x.abs() + 1e-8)           # 防止 log(0)
safe_sqrt = torch.sqrt(x.clamp(min=0))          # 防止 sqrt(-x)
safe_div = x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-8))  # 防止除以 0

# ——— torch.cumsum / torch.cumprod ———
# cumsum: 累积求和
# cumprod: 累积求积
lengths = torch.tensor([3, 5, 2, 4])
# 转换为 offsets(EmbeddingBag 需要)
offsets = torch.cat([torch.tensor([0]), lengths.cumsum(0)[:-1]])
# offsets: tensor([0, 3, 8, 10])

# ——— torch.diff ———
# 计算相邻元素的差值(类似 numpy.diff)
timestamps = torch.tensor([100, 150, 200, 250, 300], dtype=torch.float)
time_gaps = torch.diff(timestamps)   # tensor([50., 50., 50., 50.])
# 搜推广场景: 计算用户行为的时间间隔
```

##### 五、相似度与距离计算

```python
# ——— F.normalize ———
# 文字: L2 归一化 — 把向量归一化到单位球面
# 公式: ỹ = x / ||x||₂
# 代码:
user_emb = torch.randn(32, 64)
user_emb_normalized = F.normalize(user_emb, p=2, dim=-1)
# 验证: (user_emb_normalized ** 2).sum(dim=-1) ≈ 1.0 (所有行范数为 1)

# 搜推广场景: 双塔召回前必须做 L2 归一化,才能用 cosine 相似度做 ANN 检索
item_emb_normalized = F.normalize(item_emb, p=2, dim=-1)
cosine_sim = (user_emb_normalized * item_emb_normalized).sum(dim=-1)  # [B]

# ——— F.cosine_similarity ———
# 文字: 计算两个向量集合的余弦相似度
# 公式: cos(u, i) = (u · i) / (||u|| · ||i||)
# 代码:
sim = F.cosine_similarity(user_emb, item_emb, dim=-1)  # [B]
# 等价于: (user_emb * item_emb).sum(-1) / (user_emb.norm(-1) * item_emb.norm(-1))

# ——— F.pairwise_distance ———
# 文字: 计算两个向量集合的欧氏距离
# 公式: ||u - i||₂
dist = F.pairwise_distance(user_emb, item_emb, p=2)    # [B]
```

##### 六、特殊损失函数(超出 BCE/CE 的场景)

```python
# ——— F.triplet_margin_loss ———
# 文字: 三元组损失 — 让 anchor 离 positive 近、离 negative 远
# 公式: L = max(0, d(a,p) - d(a,n) + margin)
# 搜推广场景: 召回模型的训练(正样本=用户点击的物品, 负样本=随机或难负样本)
anchor = torch.randn(32, 64)      # 用户 Embedding
positive = torch.randn(32, 64)    # 正样本物品 Embedding
negative = torch.randn(32, 64)    # 负样本物品 Embedding

triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=0.3)

# ——— F.margin_ranking_loss ———
# 文字: 给定两个分数,让 x1 排在 x2 前面(y=1) 或后面(y=-1)
# 公式: L = max(0, -y*(x1 - x2) + margin)
# 搜推广场景: BPR 损失(正样本分数 > 负样本分数)
pos_scores = torch.randn(32)
neg_scores = torch.randn(32)
y = torch.ones(32)   # 1 表示 pos > neg

bpr_loss = F.margin_ranking_loss(pos_scores, neg_scores, y, margin=0.0)
# 等价于 BPR: -log(sigmoid(pos_scores - neg_scores)).mean()

# ——— InfoNCE 损失 (对比学习,手动实现) ———
# 文字: 让正样本对的相似度高于所有负样本对
# 公式: L = -log(exp(sim⁺/τ) / Σexp(sim⁻ᵢ/τ))
# 搜推广场景: 双塔召回的 in-batch negative 训练
def infonce_loss(user_emb, item_emb, temperature=0.07):
    """
    user_emb: [B, D]  item_emb: [B, D]
    正样本: (user_i, item_i), 负样本: batch 内其他物品
    """
    # L2 归一化
    u = F.normalize(user_emb, dim=-1)   # [B, D]
    i = F.normalize(item_emb, dim=-1)   # [B, D]

    # 计算所有对的相似度: [B, B]
    sim_matrix = u @ i.T / temperature
    # 对角线是正样本对,其余是负样本对

    labels = torch.arange(u.size(0), device=u.device)   # [0, 1, 2, ..., B-1]
    loss = F.cross_entropy(sim_matrix, labels)    # 等价于 InfoNCE
    return loss
```

##### 七、显存管理与性能工具

```python
import torch

# ——— 显存查询 ———
if torch.cuda.is_available():
    # 当前显存使用
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    print(f"已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")

    # 强制释放缓存(OOM 后的急救手段)
    torch.cuda.empty_cache()

    # 最大峰值显存(用于定位 OOM 的来源)
    torch.cuda.reset_peak_memory_stats()
    # ... 运行一些操作 ...
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"峰值显存: {peak:.2f} GB")

# ——— torch.autocast (混合精度推理/训练) ———
# 文字: 在 with 块内, 自动把支持的操作换成 BF16/FP16
# 好处: 速度 2x,显存减半,精度损失极小
device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    # 在这个 block 内, matmul/conv 自动用 BF16
    # 梯度累积/优化器步骤在外部用 FP32
    output = model(batch)
    loss = loss_fn(output, labels)

# ——— torch.profiler (性能分析) ———
# 文字: 精确测量每个操作的 CPU/GPU 耗时和显存占用
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_forward"):
        output = model(batch)
    with record_function("loss_backward"):
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# 输出: 每个操作的 CPU 时间, CUDA 时间, 调用次数, 显存占用
# 用于精准定位训练瓶颈(是 Embedding 查表慢?还是 attention 慢?)

# ——— torch.chunk / torch.split ———
# chunk: 把张量平均切成 n 份(最后一份可能小)
# split: 按指定大小切割

x = torch.randn(100, 64)
chunks = torch.chunk(x, 4, dim=0)        # 4 份,每份 25×64
parts = torch.split(x, [30, 30, 40], dim=0)  # 按指定大小切,3份 30/30/40

# 搜推广场景: 梯度累积时分割大 batch
mega_batch = torch.randn(4096, 64)
micro_batches = torch.chunk(mega_batch, 8, dim=0)  # 8 个 micro_batch
for mb in micro_batches:
    loss = model(mb).mean()
    (loss / 8).backward()   # 除以步数,等效于大 batch 梯度

# ——— torch.nonzero / torch.argwhere ———
# 文字: 找出非零元素的索引
x = torch.tensor([[0, 1, 0], [2, 0, 3]])
indices = torch.nonzero(x)   # [[0,1],[1,0],[1,2]]  非零位置的坐标

# 搜推广场景: 找出 batch 中有效(非 padding)的位置
mask = torch.tensor([[1,1,1,0,0],[1,1,0,0,0]])  # 0=padding
valid_positions = torch.nonzero(mask)  # 有效位置的坐标

# ——— torch.bincount ———
# 文字: 统计非负整数张量中每个值出现的次数
# 搜推广场景: 统计各 category 的样本数量
categories = torch.tensor([0, 1, 2, 0, 1, 0, 3])
counts = torch.bincount(categories)   # tensor([3, 2, 1, 1])
# 可用于正负样本计数或采样权重计算
```

---

### 11.3 Autograd:自动求导引擎深度拆解

> **从 CV 到搜推广,Autograd 是你必须重新理解的概念**。
>
> CV 里你只关心 loss.backward() 能不能跑通;但在搜推广里,显存优化、梯度检查、断点续训、混合精度,全都在 Autograd 这一层展开。
>
> **关键问题**:
> - loss.backward() 到底是怎么找到 W.grad 的?
> - 为什么中间变量的梯度在 backward 后会释放?
> - 梯度是"累加"的——这意味着什么?怎么利用?

#### 11.3.1 计算图的构建:Define-by-Run 机制

**PyTorch 跟 TensorFlow 1.x 最大的区别**:TensorFlow 是先画图再执行(Define-and-Run),PyTorch 是边执行边画图(Define-by-Run)。

```python
# 这条简单的代码:
y = W @ x + b

# 在 PyTorch 内部实际发生的过程:
# 1. ATen 底层 C++ 引擎执行矩阵乘法
# 2. 在 C++ 层的 TensorImpl 中创建一个 AutogradMeta 对象
# 3. AutogradMeta 记录:
#    - grad_fn = <MmBackward0>  (指向反向函数)
#    - next_edges = [W.grad_fn, x.grad_fn]  (指向输入的前向节点)
# 4. 每个中间张量 z 都持有一个 grad_fn 指针
# 5. grad_fn 内部持有 next_functions——指向其输入的 grad_fn 或 AccumulateGrad
```

**所以 PyTorch 的计算图在内存中是一棵"倒挂树"**:

```text
前向传播方向(forward):
W(leaf, grad_fn=None) ───┐
                         ├──→ @ ──→ z ──→ + ──→ y ──→ loss ──→ (输出)
x(leaf, grad_fn=None) ───┘                ↑
b(leaf, grad_fn=None) ────────────────────┘

每个节点实际存储的是"逆链路"(到反向传播方向):
loss.grad_fn.next_functions[0] = y.grad_fn   # (AccumulateGrad)
                                   ↓
y.grad_fn.next_functions = [z.grad_fn.next_functions[0], b.grad_fn]
                            ↓
z.grad_fn.next_functions = [W.grad_fn, x.grad_fn]  # W.grad_fn = AccumulateGrad

# 关键:AccumulateGrad 是叶节点的特殊 grad_fn
# 当反向传播到达 AccumulateGrad 时,梯度被累加到 .grad 属性
```

**叶子节点 vs 非叶子节点**:
```python
# 叶子节点:由用户直接创建的张量 require_grad=True
W = torch.randn(128, 64, requires_grad=True)   # 叶子节点! W.is_leaf = True
b = torch.randn(64, requires_grad=True)         # 叶子节点! b.is_leaf = True

# 非叶子节点:由运算产生的中间张量
z = x @ W                       # 非叶子节点! z.is_leaf = False
y = z + b                        # 非叶子节点! y.is_leaf = False

# 验证
print(W.is_leaf, W.grad_fn)     # True  None  ← 叶子节点的 grad_fn=None
print(z.is_leaf, z.grad_fn)     # False <MmBackward0>  ← 非叶子节点绑定 grad_fn
```

**搜推广里最常见的图构建错误**:

```python
# 错误:在训练循环中不小心创建了新的叶子节点
for epoch in range(10):
    # 正确:用 torch.tensor 包装 input
    input_tensor = torch.tensor(input_data, requires_grad=False)

    # 错误!放到循环外
    # input_tensor = ...

# 错误:对 requires_grad 的混合理解
x = torch.randn(32, 128)                    # requires_grad=False, 叶子
W = torch.randn(128, 64, requires_grad=True) # 叶子

y = (x.detach().clone()) @ W   # 这么写,从 x 克隆的新张量虽然是叶子
                                # 但会创建一条独立的计算图分支
                                # 如果 x 是数据张量,clone 浪费显存
```

#### 11.3.2 `.backward()` 到底做了什么?——链式法则的图遍历算法

**公式**:链式法则是 Autograd 的数学基础。对于计算图中的任意路径 $\text{loss} \to y \to z \to W$：

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial W}$$

以 $z = xW,\ y = z + b,\ \mathcal{L} = \text{sum}(y)$ 为例：

$$\frac{\partial \mathcal{L}}{\partial W} = x^\top \cdot \underbrace{\frac{\partial \mathcal{L}}{\partial y}}_{\mathbf{1}_{B \times D}} = x^\top \mathbf{1} \quad \text{(shape: [D_in x D_out], 与 W 完全一致)}$$

**核心算法:从 loss 出发,沿计算图的 DAG 进行拓扑序反向遍历**。

```python
# 直观代码
loss = f(w @ x + b)

# 反向传播在内部等价于:
# 1. 从 loss.grad_fn 开始
# 2. 调用 loss.grad_fn() → 计算 dl/dy, 传给 y.grad
# 3. 调用 y.grad_fn() → 计算 dl/dz 和 dl/db, 传给 z.grad 和 b.grad
# 4. 调用 z.grad_fn() → 计算 dl/dW 和 dl/dx, 传给 W.grad
# 5. 到达 AccumulateGrad → dl/dW 累加到 W.grad
```

**完整跟踪代码——亲手看 Autograd 的每一步**:

```python
import torch

W = torch.randn(128, 64, requires_grad=True)
b = torch.randn(64, requires_grad=True)
x = torch.randn(32, 128)              # requires_grad=False

z = x @ W                              # MmBackward0
y = z + b                              # AddBackward0
loss = y.sum()                         # SumBackward0

print("=== 前向传播后的 grad_fn 链 ===")
print(f"loss.grad_fn: {loss.grad_fn}")          # <SumBackward0>
print(f"loss.grad_fn.next_functions:")
for fn in loss.grad_fn.next_functions:
    print(f"  - {fn[0]}")                        # [(<AddBackward0, 0>)]

print(f"\ny.grad_fn.next_functions:")
for fn in y.grad_fn.next_functions:
    print(f"  - {fn[0]}")                        # <MmBackward0>, <AccumulateGrad>

print(f"\nz.grad_fn.next_functions:")
for fn in z.grad_fn.next_functions:
    print(f"  - {fn[0]}")                        # <AccumulateGrad>, None

# 注意:每个 next_functions 是 (grad_fn, input_index) 的元组
# input_index 表示当前操作的第几个输入需要梯度
```

**输出解读(反映链式求导的递归结构)**:
```text
loss  ──→ SumBackward0 ──→ AddBackward0 ──→ MmBackward0 ──→ AccumulateGrad(W)
                                            └── None(x, 因为没有梯度)
                              └── AccumulateGrad(b)
```

**这就是整个 Autograd 的秘密:`.backward()` 就是在执行一个"沿逆箭头递归调用 grad_fn"的深度优先搜索(DFS)**。

```text
backward(loss):
    # 伪代码
    queue = topological_sort(loss.grad_fn)  # 拓扑排序,确保先算靠近 loss 的
    for fn in reversed(queue):              # 反向遍历
        fn.apply()                          # 调用 grad_fn 的反向函数
```

**梯度形状验证**:
```python
loss.backward()
print(f"W.grad.shape: {W.grad.shape}")   # [128, 64] ← 跟 W 的形状完全一致!
print(f"b.grad.shape: {b.grad.shape}")   # [64] ← 跟 b 的形状完全一致!
# 每个参数的梯度形状 = 参数本身的形状,这是最基本的 sanity check
```

#### 11.3.3 梯度的"累加"特性:是坑也是工具

**公式**:梯度累积的数学等价性。设 mini-batch $\mathcal{B}_k$ 的损失为 $\mathcal{L}_k$，累积 $N$ 步等价于一个大 batch $\mathcal{B} = \bigcup_k \mathcal{B}_k$：

$$g_{\text{accum}} = \sum_{k=1}^{N} \frac{\partial (\mathcal{L}_k / N)}{\partial \theta} = \frac{1}{N} \sum_{k=1}^{N} \nabla_\theta \mathcal{L}_k \approx \nabla_\theta \mathcal{L}_{\text{large-batch}}$$

**等效 batch size**：$\text{batch\_size\_effective} = \text{batch\_size} \times N_{\text{accum}} \times N_{\text{GPUs}}$

关键：每次反向的 loss 必须除以 $N$，否则梯度被放大 $N$ 倍，等效于把学习率扩大了 $N$ 倍。

**这是搜推广里最容易被忽视的 Autograd 特性**:

```python
# 每次 backward(),梯度是累加到 .grad 上,不是覆盖!
# 这意味着:

# 场景 1:忘记 zero_grad(坑)
for step, batch in enumerate(loader):
    loss = model(batch)
    loss.backward()
    # optimizer.step()  ← 忘了写,但损失函数在循环里继续计算下一个 batch
    # 第二次循环:
    #   W.grad 在第一次已经累加了梯度
    #   第二次 loss.backward() 又往 W.grad 上加了一次
    #   累积 10 步后 W.grad *= 10, 梯度爆炸!

# 场景 2:梯度累积(工具——这就是 gradient accumulation 的底层机制)
# 显存不够?分 4 个小 batch 累积梯度,再更新一次
accumulation_steps = 4
optimizer.zero_grad()                   # 全局清 0
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps  # 关键:除以 N! 否则梯度×N
    loss.backward()                      # 梯度累加到 .grad

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                 # 用累积的梯度更新
        optimizer.zero_grad()            # 更新完清 0
```

**梯度累积的数学推导**:
假设原始 batch_size = 1024,显存不够。
拆成 4 个 mini-batch,每个 batch_size = 256。

梯度累积:
   total_grad = 0
   for i in range(4):
       loss_i = $L_i / 4$
       loss_i.backward()    # total_grad += $\mathrm{grad}(L_i) / 4$
   # total_grad = $\sum \mathrm{grad}(L_i) / 4 = \mathrm{grad}(\sum L_i / 4)$
   # 由于 $L_i$ 是独立 batch 的损失,$\sum L_i / 4 = L$(总 batch 的近似)

如果忘记除以 N:
   total_grad = $\sum \mathrm{grad}(L_i) = N \cdot \mathrm{grad}(L_{\mathrm{avg}})$
   total_grad *= N,学习率必须除以 N 来补偿!

**利用梯度做调试**(搜推广 debug 神器):

```python
# 监控梯度,判断训练是否健康
def log_gradient_stats(model: nn.Module, log_every=100):
    """输出梯度统计,帮助判断是否需要调整学习率或裁剪"""
    total_norm = 0.0
    max_grad = 0.0
    min_grad = 0.0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, param.grad.max().item())
            min_grad = min(min_grad, param.grad.min().item())
            param_count += 1

    total_norm = total_norm ** 0.5

    if param_count > 0:
        logger.info(
            f"梯度的全局 L2 范数: {total_norm:.4f} | "
            f"最大值: {max_grad:.6f} | 最小值: {min_grad:.6f}"
        )
        # 经验阈值:
        # total_norm < 1e-4: 梯度消失,模型不学习!
        # total_norm > 10: 梯度爆炸,需要裁剪或降 lr
        # total_norm ~ 0.1~1: 健康范围
```

#### 11.3.4 不求导上下文:何时该用 `torch.no_grad()` 和 `.detach()`

**这个选择的本质是:告诉 Autograd"这一段不需要画进计算图"**。

**`torch.no_grad()` vs `.detach()` 的区别**:

| 维度 | `torch.no_grad()` | `.detach()` |
| :--- | :--- | :--- |
| 作用范围 | **上下文管理器**,影响其中所有操作 | **张量方法**,只影响该张量 |
| 适用场景 | **推理评估**,不需要整个代码块参与求导 | **分离特定张量**,其余正常求导 |
| 是否重建图 | 完全不建图 | 从该张量处切断 |
| 搜推广用法 | `@torch.no_grad()` 装饰 validate 函数 | Teacher-student 蒸馏 |

```python
# 搜推广高频场景 1:验证集评估——用 no_grad
@torch.no_grad()               # 装饰器,整个函数不求导,省显存 + 加速
def validate(model, loader):
    """搜推广标准验证函数:不求导,省显存,加速"""
    for batch in loader:
        logits = model(batch)   # 内部不创建任何 grad_fn
        # ...计算指标...
    # 不用 no_grad:每个前向都创建计算图 → OOM 死亡

# 搜推广高频场景 2:特征提取——用 no_grad 或 detach
with torch.no_grad():
    item_embs = item_encoder(item_ids)    # 预提取所有商品的 embedding
    # 这些 embedding 后续作为静态特征使用

# 搜推广高频场景 3:Teacher-Student 蒸馏——必须用 detach
teacher_logits = teacher_model(x)
# teacher_logits 是计算图的一部分(跟踪了 teacher 的梯度)
teacher_probs = torch.softmax(teacher_logits, dim=-1)

# 我们要把 teacher_probs 作为"固定的监督信号"传给 student
# 所以必须 detach,不让梯度回流到 teacher
student_logits = student_model(x)
loss = F.kl_div(student_logits, teacher_probs.detach())
# 如果忘记 detach:teacher 也会被更新!而且 teacher 参数没在优化器里,梯度无处可去 → 报错!
```

#### 11.3.5 `retain_graph` 和梯度释放策略

**文字**：PyTorch 默认在 `backward()` 结束后释放中间激活（节省显存），`retain_graph=True` 阻止这一释放。实际上，绝大多数多损失场景只需将损失相加后做一次 `backward()`，无需 `retain_graph`。

**公式**：多任务损失的合并等价于分别回传再叠加梯度（线性叠加性）：

$$\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L}_1 + \alpha \nabla_\theta \mathcal{L}_2 \quad \Leftrightarrow \quad (\mathcal{L}_1 + \alpha\mathcal{L}_2)\text{.backward()}$$

```python
# PyTorch 默认行为:backward() 后,计算图的中间梯度被释放(节约显存)
# retain_graph=True:保留计算图

# 场景 1:多次 backward(搜推广极少用)
loss1 = model1(x)
loss2 = model2(x)

# 默认:loss1.backward() 会释放计算图
# loss2.backward() 会报错:计算图已被释放
# 方案:
loss1.backward(retain_graph=True)  # 保留图
loss2.backward()                    # 复用
optimizer.step()                    # 更新

# 场景 2:多层 Loss 叠加(搜推广有用!)
# 在 DIEN 中,主损失 + 辅助损失需要同时回传
ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits, ctr_labels)
aux_loss = compute_auxiliary_loss(...)   # DIEN 的辅助损失

total_loss = ctr_loss + 0.5 * aux_loss
total_loss.backward()                    # 一次 backward 搞定
# 不需要 retain_graph,因为 total_loss 在一个图上
```

#### 11.3.6 Autograd 在搜推广中的显存陷阱

```python
# 陷阱 1:推理时忘了 no_grad——GPU OOM 的元凶
def validate(model, loader):
    for batch in loader:
        # 错误:没加 no_grad
        logits = model(batch)     # ← 这里会创建计算图!
        val_loss = loss_fn(logits, labels)
        # val_loss.backward()    # 虽然没有 backward,但计算图已经建了!
        # 10 个 batch 后:每个 batch 的计算图都在内存里! OOM!

# 正确做法
@torch.no_grad()
def validate(model, loader):
    for batch in loader:
        logits = model(batch)     # 不创建图,省 50~70% 显存

# 陷阱 2:loss 计算图被意外保留
for batch in loader:
    logits = model(batch)
    loss = loss_fn(logits, labels)
    loss.backward()
    # 忘记:optimizer.step() 和 optimizer.zero_grad()
    # 下次循环:新的 loss 又创建计算图,旧的没释放
    # 显存持续增长 → OOM

# 陷阱 3:复杂模型中的计算图链路过长
# 搜推广的 Transformer + 多层 Attention
# 计算图的深度可能达到几千层
# backward 时 DFS 遍历的开销 ≈ O(深度)
# 如果发现每个 step 耗时比预期长 2 倍,检查 model 是不是太深了
```

#### 11.3.7 Autograd 的 hook 机制:调试与修改的瑞士军刀

**文字**：Hook 是挂在计算图节点上的回调函数，在 backward 到达该节点时被触发。`tensor.register_hook(fn)` 接收并可修改该张量的梯度；`module.register_backward_hook(fn)` 监听整个子模块的输入/输出梯度。

**公式**：设 hook 函数 $h$，梯度传播时原梯度 $g$ 被替换为 $h(g)$（若 `h` 返回 `None` 则保持原值）：

$$\tilde{g} = h(g) \quad \Rightarrow \quad \nabla_\theta \mathcal{L} \leftarrow \tilde{g}$$

常见用途：梯度裁剪 $h(g) = \text{clamp}(g, -c, c)$，梯度缩放 $h(g) = \alpha g$，梯度打印（返回 None）。

```python
# torch.Tensor.register_hook:在反向传播时"拦截"梯度,做自定义操作

# 场景 1:打印梯度(调试)
def print_grad(grad):
    print(f"梯度值: {grad.norm():.4f}, 形状: {grad.shape}")
    return grad

emb_output = embedding_layer(user_ids)
emb_output.register_hook(print_grad)    # 每次 backward 时打印

# 场景 2:梯度裁剪——针对特定层(搜推广常用)
def clip_embedding_grad(grad, max_norm=0.1):
    """对 Embedding 层的梯度做更严格的裁剪"""
    return torch.clamp(grad, -max_norm, max_norm)

# 只裁剪 user_embedding 的梯度
model.user_embedding.weight.register_hook(clip_embedding_grad)

# 场景 3:用 hook 实现 Gradient Checkpointing(省显存)
# 原理:在 forward 时扔掉中间激活,backward 时重新计算
# 这是 PyTorch 官方实现,不是自己写 hook
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(emb, linear):
    # 中间结果不在显存中持久化,backward 时重新算
    return linear(emb)

output = checkpoint(forward_with_checkpoint, emb_out, linear_layer)
# 效果:显存减少 30~50%,时间增加约 20%
```

---

### 11.4 nn.Module:模型的"积木"

#### 11.4.1 nn.Module 的三层结构

每个 `nn.Module` 在内部维护三个字典。模型的前向计算本质上是函数复合:

$$\text{forward}(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

其中每个 $f_k$ 是一个子模块，参数 $\theta_k \in$ `_parameters`，不变统计量 $s_k \in$ `_buffers`。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 32)    # → _parameters 和 _modules
        self.linear = nn.Linear(32, 16)      # → _parameters 和 _modules
        self.register_buffer(                # → _buffers
            "running_mean",
            torch.zeros(16)
        )

    def forward(self, x):
        return self.linear(self.emb(x))

model = MyModel()

# 查看三件套:
print("参数:", list(model.parameters()))      # 所有可学习参数
print("buffer:", list(model.buffers()))       # 所有 buffer(running_mean 等)
print("子模块:", list(model.children()))      # 直接子模块(emb, linear)
print("全部子模块:", list(model.modules()))    # 递归子模块

# .to(device) / .train() / .eval() 会自动应用到所有三层
model.to("cuda")        # parameters + buffers + children 全部移到 GPU
model.train()            # 所有 BN/Dropout 切到训练模式
model.eval()             # 所有 BN/Dropout 切到推理模式
```

**核心区别**:

| 存储 | 说明 | 是否参与 | .to(device) | state_dict |
| :--- | :--- | :--- | :--- | :--- |
| `_parameters` | 模型权重 | ✅ 自动计算梯度 | ✅ 自动迁移 | ✅ 保存 |
| `_buffers` | 运行统计量(BN mean/var) | ❌ 无梯度 | ✅ 自动迁移 | ✅ 保存 |
| `_modules` | 子模块 | 递归处理 | ✅ 自动迁移 | ✅ 递归保存 |
| 普通属性(如 `self.lr=1e-3`) | — | ❌ | ❌ | ❌ |

#### 11.4.2 自定义层:继承 nn.Module

自定义层常见组合: Linear → BatchNorm → ReLU → Dropout。BatchNorm 的核心公式:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad y = \gamma \hat{x} + \beta$$

训练时 $\mu_B, \sigma_B^2$ 来自当前 batch；推理时用 `running_mean`/`running_var`(EMA 更新的 buffer)。

```python
class MyDenseLayer(nn.Module):
    """自定义前馈层,含 batch_norm + dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)          # 训练时用 batch 统计,推理时用 running
        x = torch.relu(x)
        x = self.dropout(x)
        return x

# 使用
layer = MyDenseLayer(128, 64)
x = torch.randn(32, 128)
y = layer(x)                     # [32, 64],像函数一样调用
```

#### 11.4.3 Sequential 和 ModuleList:自动化 forward

**文字**:`nn.Sequential` 把层组合成固定管道；`nn.ModuleList` 在 forward 里灵活控制。推荐系统中最典型的 ModuleList 用场是**专家混合（Mixture of Experts, MoE）**——多个 expert 并行处理同一输入，门控网络决定每个 expert 的权重。

**公式**:MoE 的 Soft Gating 机制：

$$g(x) = \text{softmax}(xW_g) \in \mathbb{R}^{n_e}, \quad f_k(x) = \text{Expert}_k(x) \in \mathbb{R}^D$$

$$\text{MoE}(x) = \sum_{k=1}^{n_e} g_k(x) \cdot f_k(x) \quad \text{(加权求和所有 expert 输出)}$$

**Hard Gating（Top-K Sparse MoE，DeepSeek-V3 等 LLM 标配）**：只激活权重最高的 K 个 expert，其余权重置零：

$$\text{Top-K MoE}(x) = \sum_{k \in \mathcal{K}} g_k(x) \cdot f_k(x), \quad \mathcal{K} = \text{top}_K(\text{softmax}(xW_g))$$

**代码**:

```python
# nn.Sequential:一串层自动按顺序执行
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# nn.ModuleList:存储多个子模块,手动控制 forward(用于 Expert 等场景)
class Expert(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, n_experts, in_dim, out_dim):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(in_dim, out_dim) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(in_dim, n_experts)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=-1)  # [B, n_experts]
        outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, n_experts, D]
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)  # [B, D]
```

---

### 11.5 Embedding 深度解析(搜推广的命脉)

#### 11.5.1 nn.Embedding:数学本质 vs 工程本质——文字·公式·代码三位一体

**文字**:Embedding 是一个巨大的查表矩阵，把离散 ID（整数）映射到连续向量。它是搜推广系统的核心参数——用户行为、物品属性、类别特征全都通过 Embedding 进入神经网络。理解 Embedding 的内存、梯度、初始化特性，是搜推广工程的基础技能。

**公式**:

$$\text{数学本质:} \quad e(i) = E \cdot \mathbf{1}_i \quad \text{(矩阵} E \in \mathbb{R}^{V \times D} 与 one-hot 向量相乘)$$

$$\text{等价于: \quad e(i) = E[i, :] \quad \text{(直接取第 } i  行,O(D) 而不是 O(VD))$$

$$\text{反向传播:} \quad \frac{\partial \mathcal{L}}{\partial E} = \text{sparse gradient, 只有第 } i \text{ 行非零}$$

$$\text{内存:} \quad V \times D \times \text{bytes/elem \quad (V=10^7, D=64, \text{FP32} \Rightarrow 2.56\text{GB})$$

**代码**:

```python
import torch
import torch.nn as nn
import torch.nn.init as init

# ═══════════════════════════════════════
# 基础定义与前向传播
# ═══════════════════════════════════════
V, D = 10_000_000, 64   # 1000 万个物品,64 维向量
emb = nn.Embedding(num_embeddings=V, embedding_dim=D)
print(f"参数量: {emb.weight.numel()/1e6:.1f}M")    # 640M 参数
print(f"显存:   {emb.weight.numel()*4/1e9:.2f}GB")  # 2.56 GB (FP32)

# 前向: 索引查表
indices = torch.tensor([123, 456, 789])    # 3 个物品 ID
vectors = emb(indices)                     # [3, D]

# 批量查表 (序列)
batch_hist = torch.randint(0, V, (1024, 50))   # 1024 个用户,各 50 个历史
hist_emb = emb(batch_hist)                      # [1024, 50, D]

# ═══════════════════════════════════════
# 初始化: 默认 vs 最佳实践
# ═══════════════════════════════════════
# 默认: N(0, 1) —— 初始方差太大!向量间夹角随机,与随机差不多
emb_default = nn.Embedding(V, D)
print(f"默认初始化 std: {emb_default.weight.std().item():.3f}")  # ≈ 1.0

# 推荐: 小方差均匀分布 (常用于工业)
emb_custom = nn.Embedding(V, D)
init.uniform_(emb_custom.weight, -0.05, 0.05)
print(f"推荐初始化 std: {emb_custom.weight.std().item():.3f}")   # ≈ 0.029

# 带 padding_idx: 位置 0 的向量永远是全零,不更新梯度
emb_pad = nn.Embedding(V, D, padding_idx=0)
# 用途: 序列 padding 位置填 0 → Pooling 时不影响结果

# 从预训练导入 (BERT/LLaMA 的词向量)
pretrained = torch.randn(V, D)
emb_pretrained = nn.Embedding.from_pretrained(pretrained, freeze=False)

# ═══════════════════════════════════════
# 稀疏梯度: 搜推广最重要的 Embedding 特性
# ═══════════════════════════════════════
# 关键: 一个 batch 1024 个样本,最多只访问 1024 个不同行
# 其余 9.999M 行的梯度是精确的 0 (稀疏!)
#
# 验证稀疏梯度:
emb_test = nn.Embedding(100, 8, sparse=True)   # sparse=True 启用稀疏梯度
ids = torch.tensor([1, 5, 23])                  # 只访问 3 行
out = emb_test(ids).sum()
out.backward()
# emb_test.weight.grad 是 sparse tensor,只有 3 行非零

# 稀疏梯度带来的问题:
# 普通 Adam 每步对全 V 行维护 m_t, v_t → V 很大时内存爆炸
# 解法 1: SparseAdam (只对访问过的行更新 m/v)
# 解法 2: weight_decay=0 (避免每步都 touch 全部行)
# 解法 3: 哈希 Embedding (见下方)

# ═══════════════════════════════════════
# 哈希 Embedding: 大规模场景的内存节省方案
# ═══════════════════════════════════════
class HashedEmbedding(nn.Module):
    """
    哈希 Embedding: V 个 ID → B 个桶 (B << V)

    原理: id → id % num_buckets → 查表
    代价: 不同 ID 可能映射到同一桶 (冲突),但统计上影响不大

    内存节省: V/num_buckets 倍
      V=1000万, num_buckets=100万 → 节省 10×
    """
    def __init__(self, num_buckets, embedding_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, x):
        x_hashed = x % self.num_buckets   # 哈希到桶
        return self.emb(x_hashed)

hashed_emb = HashedEmbedding(num_buckets=1_000_000, embedding_dim=64)
ids = torch.randint(0, 10_000_000, (1024,))
out = hashed_emb(ids)   # [1024, 64]
print(f"哈希 Embedding 参数量: {sum(p.numel() for p in hashed_emb.parameters())/1e6:.1f}M")
# 64M 而非 640M
```

#### 11.5.2 搜推广的 Embedding 规模挑战——文字·公式·代码

**文字**:工业界的 Embedding 表常常无法放在一张 GPU 上。Facebook DLRM 的 Embedding 表总参数超过 1TB。理解内存估算公式，才能做合理的工程决策。

**公式(内存估算)**:

$$\text{Embedding 表内存} = V \times D \times \text{bytes\_per\_element}$$

$$\text{Adam 优化器额外内存} = V \times D \times 2 \times \text{bytes} \quad (m_t + v_t \text{ 各一份})$$

$$\text{总训练内存} \approx V \times D \times 12 \text{ bytes} \quad (\text{权重}4 + \text{梯度}4 + m_t 4 + v_t 4,\text{FP32})$$

**代码**:

```python
# ═══════════════════════════════════════
# 内存估算工具函数
# ═══════════════════════════════════════
def embedding_memory_gb(vocab_size, embedding_dim, dtype='fp32', with_optimizer=True):
    """计算 Embedding 表在训练时的总显存占用"""
    bytes_per_elem = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1}[dtype]
    weight_bytes = vocab_size * embedding_dim * bytes_per_elem
    grad_bytes   = weight_bytes  # 梯度与权重同类型
    adam_bytes   = vocab_size * embedding_dim * 4 * 2  # m_t + v_t 通常用 FP32

    total = weight_bytes + grad_bytes + (adam_bytes if with_optimizer else 0)
    return total / 1e9

# 工业级场景估算
scenarios = [
    ("用户 Embedding", 1_000_000_000, 32),   # 10亿用户×32
    ("物品 Embedding", 500_000_000,  64),    # 5亿物品×64
    ("类别 Embedding", 100_000,     16),     # 10万类别×16
]
total = 0
for name, V, D in scenarios:
    mem = embedding_memory_gb(V, D)
    total += mem
    print(f"{name}: V={V/1e6:.0f}M, D={D}, 训练内存={mem:.1f}GB")
print(f"合计: {total:.1f}GB (一张 A100 只有 80GB!)")

# ═══════════════════════════════════════
# 工业级解法对比
# ═══════════════════════════════════════
"""
方案                    内存节省    精度损失    实现复杂度
─────────────────────────────────────────────────────────
SparseAdam + 不存所有m/v  ~50%      无         低
哈希分桶(V→V/10)          ~90%      少量冲突   低
参数服务器(CPU/GPU 混合)   GPU 0%   无         高
BF16 权重                 50%       极少       中
Embedding 分片(模型并行)   1/N GPU  无         高
频率截断(只保留高频 ID)     ~50%     冷启动差   低
"""
```

#### 11.5.3 nn.EmbeddingBag:变长多值特征的克星

**文字**:用户历史点击序列长度不固定（有人点了 3 个，有人点了 50 个），但模型需要固定维度输入。`nn.EmbeddingBag` 将"查表"和"池化"合并为单一内核，比 Embedding + 手动 mean 更快（无中间张量），也更省显存（不存 [B,T,D] 的中间结果）。

**公式**:

$$\text{EmbeddingBag}(\text{ids}_n) = \frac{1}{|\text{ids}_n|} \sum_{i \in \text{ids}_n} E[i] \quad \text{(mode='mean')}$$

$$\text{EmbeddingBag}(\text{ids}_n) = \sum_{i \in \text{ids}_n} E[i] \quad \text{(mode='sum')}$$

与手动实现的等价关系（但 EmbeddingBag 更高效）：

$$\text{EmbeddingBag}(\text{ids}) \equiv \text{mean\_pool}(\text{Embedding}(\text{ids}))$$

**代码**:

```python
# 场景:用户历史点击了 n 个物品,n 可能是 3、10、50
# 传统做法:Embedding + 手动 Pooling(前文已经讲过)
# 更高效:EmbeddingBag 把"Embedding 查表"和"Pooling"合并为 1 个算子

# 输入:展平的序列 + 每个样本的起始偏移
input_ids = torch.tensor([1, 2, 3, 4, 5, 6])   # 6 个 ID,展平
offsets = torch.tensor([0, 2, 5])               # 样本 1: [0,2), 样本 2: [2,5), 样本 3: [5,6)

emb_bag = nn.EmbeddingBag(
    num_embeddings=1000,
    embedding_dim=32,
    mode='mean',                                # 'mean' / 'sum' / 'max'
    padding_idx=0
)
output = emb_bag(input_ids, offsets)            # [3, 32]
# 等价于:Embedding + 对每个样本的向量做 mean pooling

# 对比 EmbeddingBag 和 Embedding+手动pooling:
# EmbeddingBag:更省内存(不用存中间 emb),更快(底层融合算子)
```

#### 11.5.4 位置编码(Positional Encoding)

**文字**:Transformer 的 Self-Attention 对序列位置没有感知（它只做集合式操作），必须显式注入位置信息。搜推广里，用户历史行为的顺序极其重要：第 1 个点击 vs 第 50 个点击的语义完全不同。

**公式**:Sinusoidal 位置编码（Vaswani et al. 2017）：

$$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/D}}\right), \quad PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/D}}\right)$$

设计逻辑：不同频率的正弦/余弦波，使得 $PE_{\text{pos}+k}$ 可以表达为 $PE_{\text{pos}}$ 的线性变换，从而让模型能学到"相对位置"。

**RoPE（旋转位置编码）**— 现代推荐 Transformer 首选（在 §6.2.3 有完整推导）：

$$\text{RoPE}(q, m) = q e^{im\theta} \quad \text{(将 q 在复平面旋转角度 } m\theta\text{)}$$

$$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \text{Re}(q \bar{k} e^{i(m-n)\theta}) \quad \text{(只与相对位置 m-n 有关)}$$

**代码**:

```python
# 在搜推广的 Transformer 序列模型里,必须加位置编码(否则模型不知道输入顺序)

# 方案 1:可学习位置编码(最常用)
pos_emb = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
x = x + pos_emb(positions)

# 方案 2:静态 Sinusoidal(不用学,Transformer 原文的方案)
def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    # 分母: 10000^(2i/D) = exp(2i * log(10000)/D)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [max_len, D]
```

**三种位置编码对比**:

| 方案 | 公式特点 | 推荐场景 |
| :--- | :--- | :--- |
| 可学习 (Learned) | 参数化，最灵活 | 序列长度固定，数据量充足 |
| Sinusoidal | 固定函数，外推性较好 | 轻量 Transformer，需要长度泛化 |
| **RoPE** | 旋转，天然相对位置 | **现代 GenRec/LLM 首选** |

---

### 11.6 常见网络层

#### 11.6.1 全连接层(nn.Linear)——文字·公式·代码三位一体

**文字**:`nn.Linear` 是所有神经网络的最基础模块。它做的事情非常简单：把输入向量线性变换到另一个维度。在搜推广里，它出现在 MLP 塔的每一层、Attention 的 QKV 投影、输出层打分、以及 Embedding 的降维/升维中。

**公式**:

$$y = xW^\top + b \quad \text{(前向传播)}$$

$$x \in \mathbb{R}^{B \times D_{\text{in}}}, \quad W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}, \quad b \in \mathbb{R}^{D_{\text{out}}}, \quad y \in \mathbb{R}^{B \times D_{\text{out}}}$$

$$\text{反向传播:} \quad \frac{\partial \mathcal{L}}{\partial W} = \left(\frac{\partial \mathcal{L}}{\partial y}\right)^\top x, \quad \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot W$$

**初始化公式(为什么 Xavier/Kaiming 不一样)**:

$$\text{Xavier (用于 tanh/sigmoid):} \quad W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{D_{\text{in}} + D_{\text{out}}}}, \sqrt{\frac{6}{D_{\text{in}} + D_{\text{out}}}}\right)$$

$$\text{Kaiming (用于 ReLU):} \quad W \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{D_{\text{in}}}}\right)$$

**初始化的直觉**:方差控制。若权重初始化太大，多层网络的方差随深度指数增长 → 梯度爆炸；初始化太小，方差衰减到 0 → 梯度消失。Xavier/Kaiming 通过控制每层输出方差 ≈ 输入方差来避免这个问题。

**代码**:

```python
import torch
import torch.nn as nn

# ═══════════════════════════════════════
# 基础用法
# ═══════════════════════════════════════
linear = nn.Linear(in_features=128, out_features=64, bias=True)
print(f"weight shape: {linear.weight.shape}")   # [64, 128]  注意: out×in 不是 in×out!
print(f"bias shape:   {linear.bias.shape}")     # [64]

x = torch.randn(32, 128)   # batch=32, input_dim=128
y = linear(x)               # [32, 64]

# 权重初始化 (默认: Kaiming Uniform)
print(f"weight 初始值范围: [{linear.weight.min():.3f}, {linear.weight.max():.3f}]")

# ═══════════════════════════════════════
# 三种常用初始化方式
# ═══════════════════════════════════════
import torch.nn.init as init

# 1. Xavier Uniform (tanh/sigmoid 激活后用)
linear_tanh = nn.Linear(128, 64)
init.xavier_uniform_(linear_tanh.weight)       # 方差 = 2/(in+out)
init.zeros_(linear_tanh.bias)

# 2. Kaiming Normal (ReLU 激活后用) — PyTorch Linear 的默认
linear_relu = nn.Linear(128, 64)
init.kaiming_normal_(linear_relu.weight, mode='fan_in', nonlinearity='relu')
# fan_in: 只用输入维度 D_in 做归一化 → 方差 = 2/D_in

# 3. 输出层: 初始化为接近 0 (避免训练初期 logit 过大)
output_layer = nn.Linear(64, 1)
init.normal_(output_layer.weight, mean=0, std=0.01)
init.zeros_(output_layer.bias)

# ═══════════════════════════════════════
# 搜推广 MLP 塔的标准写法
# ═══════════════════════════════════════
class MLPTower(nn.Module):
    """
    搜推广 MLP 塔: Embedding → 多层线性 → 输出分

    设计原则:
      - 逐层缩减维度 (漏斗形)
      - ReLU 激活 (简单有效)
      - LayerNorm (比 BatchNorm 对变长 batch 更友好)
      - 最后一层不加激活函数 (输出 logit 交给 loss 函数)
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),           # 归一化
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))   # 输出层不加激活
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

# 使用
mlp = MLPTower(input_dim=256, hidden_dims=[128, 64], output_dim=1)
x = torch.randn(1024, 256)
logit = mlp(x)   # [1024, 1]
print(f"logit shape: {logit.shape}, mean: {logit.mean().item():.3f}")

# ═══════════════════════════════════════
# bias=False 什么时候合适?
# ═══════════════════════════════════════
"""
去掉 bias 的场景:
  - Attention 的 Q/K/V 投影: 通常 bias=False (后面有 LayerNorm 补偿)
  - Embedding 后的首层线性: 输入已经过 LayerNorm,不需要 bias
  - 参数量很紧张时

保留 bias 的场景:
  - 输出层 (bias 决定预测的基准值)
  - 独立 MLP 块的每层
"""
```

#### 11.6.2 归一化层:为什么现代 GenRec 和 LLM 都选择 RMSNorm?

> **从 BatchNorm → LayerNorm → RMSNorm,归一化函数的进化是 GenRec+LLM 趋同最典型的例子。**
>
> 核心问题:让深层网络的梯度保持在合理范围内,避免梯度消失/爆炸。

**四种归一化的数学公式对比**:

$$\text{BatchNorm:} \quad \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},\quad y = \gamma\hat{x} + \beta, \quad \mu_B = \frac{1}{B}\sum x_i \text{(batch 内统计)}$$

$$\text{LayerNorm:} \quad \hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \varepsilon}},\quad y = \gamma\hat{x} + \beta, \quad \mu_L = \frac{1}{D}\sum_d x_d \text{(单样本特征维统计)}$$

$$\text{RMSNorm(LLM/GenRec 标配):} \quad y = \gamma \cdot \frac{x}{\sqrt{\frac{1}{D}\sum_d x_d^2 + \varepsilon}} = \gamma \cdot \frac{x}{\text{RMS}(x) + \varepsilon}$$

RMSNorm 去掉了"减均值"步骤（高维时均值 $\approx 0$），减少一次 reduce 操作，速度提升约 $30\%$。

```python
# BatchNorm(BN):沿 batch 维归一化
# 公式: y = γ * (x - μ_B) / √(σ²_B + ε) + β
#   其中 μ_B = 1/B * Σ x_i  (在一个 batch 内算)
#        σ²_B = 1/B * Σ (x_i - μ_B)²
#
# 问题:推理时 B=1,无法算 batch 统计量 → 改用训练时累积的 running_mean

# LayerNorm(LN):沿特征维归一化
# 公式: y = γ * (x - μ_L) / √(σ²_L + ε) + β
#   其中 μ_L = 1/D * Σ x_d  (在单个样本内,对特征维算)
#        σ²_L = 1/D * Σ (x_d - μ_L)²
# 优点:推理时 batch_size=1 也没问题
# 缺点:需要算均值 μ 和方差 σ²,各一次 reduce 操作

# RMSNorm(LN 去掉均值):只算 RMS,不算均值——现代 LLM/GenRec 标配!
# 公式: y = γ * x / √(mean(x²) + ε)
# 去掉 μ:只做"除以 RMS"的归一化,不做"减均值"
# 为什么可以去掉 μ?因为 x - μ 和 x 的 RMS 差距很小,去掉 μ 减少计算量

# LayerScale(残差缩放):在残差连接后乘一个可学习标量
# y = x + α * Sublayer(x) 其中 α 初始化为 0.1~0.001
# 用于深层 Transformer(> 24 层)的稳定训练
```

**选型决策树(搜推广+GenRec)**:

```text
搜索推荐场景,该用哪种归一化?
│
├── 模型是传统 MLP(DeepFM/DCN)?
│   └── 选 LayerNorm(不是 BatchNorm!)
│
├── 模型是 Transformer / GenRec?
│   ├── < 1B 参数 → RMSNorm(simpler = better)
│   ├── 1B~10B 参数 → RMSNorm + Pre-Norm
│   └── > 10B 参数 → RMSNorm + Pre-Norm + 小心初始化
│
├── 模型是纯 CNN / 视觉模型?
│   └── 选 BatchNorm(图像里 BN 是标配)
│
└── 模型是多模态(图像+文本+ID)?
    └── 用 RMSNorm 统一(而不是不同模态各用各的)
```

**RMSNorm 为什么赢了?——数学直觉**:

假设一个特征向量 $x = [1.0, -0.5, 0.3, \ldots]$,d_model = 4096

**LayerNorm 做的事**:

1. 算均值 $\mu = \mathrm{sum}(x) / 4096 \approx 0.002$
2. 算方差 $\sigma^2 = \mathrm{sum}((x - \mu)^2) / 4096 \approx 1.2$
3. 减均值:$x' = x - 0.002$(几乎没变!)
4. 除标准差:$x'' = x' / \sqrt{1.2}$

**RMSNorm 做的事**:

1. 算 RMS $= \sqrt{\mathrm{mean}(x^2)} \approx 1.1$
2. 除以 RMS:$x' = x / 1.1$

**差别**:
LayerNorm 多了一步"减均值",但均值在 d=4096 时很接近 0,所以去掉均值几乎不影响归一化效果,但节省了一次全局通信。

**节省的计算量**:

- LayerNorm: $\mathrm{sum}(x) + \mathrm{sum}(x^2) + \mathrm{sub} + \mathrm{div} + \gamma \times x + \beta$ = 2 次 reduce
- RMSNorm: $\mathrm{mean}(x^2) + \mathrm{div} + \gamma \times x$ = 1 次 reduce(少了一半!)

在训练中,reduce 操作涉及 GPU 全局同步,减半意味着训练速度提升约 5\~10%。

```python
# RMSNorm 的自实现和 PyTorch 结果对比

class RMSNorm(nn.Module):
    """
    RMSNorm: LLaMA/Mistral/Qwen/DeepSeek/GenRec 的标准归一化。
    比 LayerNorm 快约 10~15%,效果等价(甚至略好)。
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习缩放 γ
        self.eps = eps

    def forward(self, x):
        # x: [B, T, D] 或 [B, D]
        # 计算 RMS: sqrt(mean(x²))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 归一化 → 缩放
        return x / rms * self.weight

# 验证和 LayerNorm 的差异
x = torch.randn(32, 50, 256)
ln = nn.LayerNorm(256)
rms = RMSNorm(256)

y_ln = ln(x)
y_rms = rms(x)
print(f"LayerNorm output norm: {y_ln.norm(dim=-1).mean():.4f}")   # ≈ 1.0
print(f"RMSNorm output norm:  {y_rms.norm(dim=-1).mean():.4f}")   # = √d_model ≈ 16.0
# 注意:LN 的输出范数恒 ~1.0;RMSNorm 的输出范数是 γ
# 但两者在经过后续线性层后,效果等价
```

**Pre-Norm vs Post-Norm:放在哪里归一化?**:

**Post-Norm**(原始 Transformer):

- 公式:$\mathrm{output} = \mathrm{LayerNorm}(x + \mathrm{Attention}(x))$
- 梯度从 LN 传入残差 → 梯度不稳定
- 深层(> 12 层)训练困难

**Pre-Norm**(所有现代模型):

- 公式:$\mathrm{output} = x + \mathrm{Attention}(\mathrm{LayerNorm}(x))$
- 梯度直接从残差跳过,不经过 LN
- 深层(100+ 层)训练稳定

**Pre-Norm 是 GenRec 和所有 LLM 的标准选择!**

**搜推广选型终极规则**:

```python
def choose_normalization(model_type, has_sequence=False, gpu_count=1):
    """
    GenRec 归一化选型规则:

    - MLP 模型(DeepFM/DCN): LayerNorm
    - Transformer 模型: RMSNorm + Pre-Norm
    - 多 GPU 训练: SyncBatchNorm(如果用 BN)/RMSNorm 天然支持
    - 量化推理: RMSNorm 比 LayerNorm 更友好(少一次减均值)
    """
    if model_type in ["transformer", "genrec", "llm"]:
        return "RMSNorm + Pre-Norm"   # 当前唯一推荐
    elif model_type in ["dnn", "deepfm", "din", "mlp"]:
        return "LayerNorm"            # 传统模型够用
    elif model_type == "cnn":
        return "BatchNorm"            # 视觉模型
    else:
        return "RMSNorm"              # 通用兜底
```

#### 11.6.3 Dropout——文字·公式·代码三位一体

**文字**:Dropout 是神经网络最简单也最有效的正则化方法。训练时随机将一部分神经元置零（"丢弃"），迫使网络学习更鲁棒、不依赖单个神经元的特征表示。推理时关闭 Dropout，并对输出做缩放补偿。

**公式**:

$$\text{训练时:} \quad y_i = \begin{cases} 0 & \text{以概率 } p \text{ (丢弃)} \\ x_i / (1-p) & \text{以概率 } 1-p \text{ (保留,并放大)} \end{cases}$$

$$\text{推理时:} \quad y_i = x_i \quad \text{(恒等,不丢弃,不缩放)}$$

$$\text{为什么放大?} \quad E[y_i] = (1-p) \cdot \frac{x_i}{1-p} + p \cdot 0 = x_i \quad \text{(期望不变!)}$$

**代码**:

```python
import torch
import torch.nn as nn

# ═══ 基础用法 ═══
dropout = nn.Dropout(p=0.2)   # p = 丢弃概率 (保留率 = 1-p)
x = torch.randn(32, 64)

# 训练模式
dropout.train()
y_train = dropout(x)          # 20% 位置变 0,其余 ×1.25 (inverted dropout)
print(f"训练时非零比例: {(y_train != 0).float().mean():.2f}")  # ≈ 0.80

# 推理模式
dropout.eval()
y_eval = dropout(x)           # 恒等映射,完全不 dropout!
print(f"推理时与输入相同: {torch.allclose(y_eval, x)}")  # True

# ═══ 搜推广中的 Dropout 策略 ═══
class RecModel(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dims=[256, 128]):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim)
        self.dropout_emb = nn.Dropout(0.1)    # Embedding 后较小的 p

        layers = []
        in_dim = emb_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(0.3),               # MLP 层较大的 p
            ])
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, user_id):
        x = self.emb(user_id)    # [B, D]
        x = self.dropout_emb(x)  # Embedding 后小 p
        x = self.mlp(x)          # MLP 中大 p
        return self.output(x)

# ═══ 不能用 Dropout 的场景 ═══
"""
❌ Embedding 权重本身不能 Dropout:
   会破坏稀疏梯度的统计性质

❌ 输出层之后不能 Dropout:
   logit 在 loss 计算前要保持完整

⚠️ 序列 Attention 的 mask 后不要 Dropout:
   某些实现中 masked position 本来是 -inf,dropout 后可能产生 NaN

✅ 推荐使用 Dropout 的位置:
   Embedding 后 (p=0.05~0.1)
   每个 MLP 层后 (p=0.1~0.3)
   Attention 权重上 (p=0.1)
"""
```

#### 11.6.4 激活函数——文字·公式·代码三位一体

**文字**:激活函数是神经网络中唯一的非线性来源。没有激活函数，任意多层神经网络等价于单层线性变换。不同的激活函数在梯度流动、表达能力、计算开销上有明显差异，2024年工业界的标准已从 ReLU 切换到 SwiGLU。

**公式对比**:

| 激活函数 | 数学公式 | 导数 | 是否有死区 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, x)$ | $\mathbf{1}[x>0]$ | ✅ 负值死亡 | 传统 MLP |
| **LeakyReLU** | $\max(\alpha x, x),\alpha{=}0.01$ | $\alpha$ 或 $1$ | ❌ | 改进 ReLU |
| **GELU** | $x \cdot \Phi(x)$ | $\Phi(x) + x\phi(x)$ | ❌ | BERT,GPT |
| **SiLU/Swish** | $x \cdot \sigma(x)$ | $\sigma(x)+x\sigma(x)(1-\sigma(x))$ | ❌ | SwiGLU 核心 |
| **SwiGLU** | $(xW_g)\odot\text{SiLU}(xW_u) \cdot W_d$ | 门控偏导 | ❌ | **LLaMA/GenRec** |

> $\Phi(x)$ = 标准正态 CDF，$\phi(x)$ = 标准正态 PDF，$\sigma(x) = 1/(1+e^{-x})$

**传统激活函数(搜推广传统模型用)**:

```python
# ReLU:传统 MLP 默认首选(简单、快、好)
relu = nn.ReLU()
# 缺点:Dead ReLU(负值梯度为 0 → 神经元死亡)
# 搜推广里 Embedding 大,死亡 ReLU 很少发生

# LeakyReLU / PReLU:ReLU 的改进,负值有小梯度
leaky = nn.LeakyReLU(negative_slope=0.1)
prelu = nn.PReLU(num_parameters=1)         # 可学习斜率

# GELU:Google 提出,NLP/Transformer 标准
gelu = nn.GELU()
# 公式: GELU(x) = x · Φ(x)  (Φ = 正态分布 CDF)
# 近似: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**现代激活函数:GLU 变体——当前所有现代 LLM 和 GenRec 的选择**:

```python
# GLU(Gated Linear Unit) 的核心思想:
# 不是直接把输入通过一个激活函数,而是"用一个门控控制信息流"

# FFN(传统前馈): output = activation(x @ W₁) @ W₂
# GLU(门控前馈):   output = (x @ W₁) ⊗ sigmoid(x @ V) @ W₂   ← 多了一个"门"
# 或用 SiLU 替代 sigmoid: → SwiGLU

# ============ SwiGLU(LLaMA/Mistral/Qwen/DeepSeek 全部在用) ============
# 公式: SwiGLU(x) = (x @ W_gate) ⊗ SiLU(x @ W_up) @ W_down
# 其中 SiLU = x * sigmoid(x),也叫 Swish

class SwiGLU(nn.Module):
    """
    SwiGLU FFN = 当前最主流的激活 + FFN 结构
    使用方: LLaMA 2/3, Mistral, Qwen 2, DeepSeek, Gemma

    相比标准 FFN:
    - FFN:   out = activation(x @ W₁) @ W₂        参数量 = 2 × d × d_ff
    - SwiGLU: out = (x @ W_gate) ⊗ SiLU(x @ up) @ down  参数量 = 3 × d × d_ff
    SwiGLU 多了一个投影,但通常 d_ff_swiglu = 2/3 × d_ff_ffn,总参数量相近
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)  # 门控
        self.up   = nn.Linear(d_model, d_ff, bias=False)  # 上投影
        self.down = nn.Linear(d_ff, d_model, bias=False)  # 下投影

    def forward(self, x):
        # gate: 控制"多少信息通过"
        # up:   提供"信息的内容"
        # SiLU(gate) ⊗ up: 门控筛选信息
        # down: 投影回原维度
        return self.down(F.silu(self.gate(x)) * self.up(x))

# ============ GeGLU(GELU + GLU) ============
# 公式: GeGLU(x) = (x @ W_gate) ⊗ GELU(x @ W_up) @ W_down
# 比 SwiGLU 少用一些(主要在 T5 和 PaLM 里)

class GeGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))
```

**六种激活/FNN 结构的参数量和效果对比**:

| 结构 | 参数量(以 d=d_model, d_ff=4d 计算) | 代表模型 | 效果排名(经验) |
| :--- | :--- | :--- | :--- |
| **ReLU FFN** | 2 × d × 4d = **8d²** | 早期 Transformer | ★★☆ |
| **GELU FFN** | 2 × d × 4d = **8d²** | BERT, GPT-2 | ★★★ |
| **SwiGLU** | 3 × d × (8/3)d = **8d²** | **LLaMA/Mistral/Qwen** | ★★★★★ |
| **GeGLU** | 3 × d × (8/3)d = **8d²** | PaLM, T5 | ★★★★ |
| **ReGLU** | 3 × d × (8/3)d = **8d²** | 实验性 | ★★★ |
| **GLU(Sigmoid)** | 3 × d × (8/3)d = **8d²** | 早期 GLU | ★★★ |

**关键观察:SwiGLU 和 ReLU FFN 的参数量几乎相同**(因为 SwiGLU 的 d_ff 通常设为 (8/3)d ≈ 2.67d,而 ReLU FFN 的 d_ff=4d)。

```python
# 参数量验证
d_model = 512
# ReLU FFN: d_ff = 2048(4d)
relu_params = 2 * d_model * 2048    # 2,097,152
# SwiGLU: d_ff = 1365(≈ 8d/3)
swiglu_params = 3 * d_model * 1365  # 2,096,640
# ≈ 相同!但 SwiGLU 效果更好
```

**为什么 SwiGLU 比 ReLU/GELU 更好?——梯度流动分析**:

- **ReLU**:$y = \max(0, x)$,梯度为 $0$(负)或 $1$(正)
- **GELU**:$y = x \cdot \Phi(x)$,梯度为 $\Phi(x) + x \cdot \phi(x)$(连续、非 0)
- **SwiGLU**:$y = \mathrm{gate} \otimes \mathrm{SiLU}(\mathrm{up})$
  - 梯度:$\partial y / \partial \mathrm{gate} = \mathrm{SiLU}(\mathrm{up})$(不是 0!)
  - 梯度:$\partial y / \partial \mathrm{up} = \mathrm{gate} \cdot \mathrm{SiLU}'(\mathrm{up})$(gate 在控制!)

**SwiGLU 的门控特性**:即使输入 gate 是负值,$\mathrm{SiLU}(\mathrm{up})$ 也可能非 0。gate 机制让门控选择性通过信息 → 比"直接激活"更灵活。实验表明:SwiGLU 在 2/3 的参数量下能达到和 ReLU FFN 相近的效果。

**激活函数选型规则(搜推广 + GenRec 场景)**:

```python
def choose_activation(model_type, layer_type):
    """
    选型规则:
    - 传统推荐模型(DeepFM/DIN): ReLU 足够,SwiGLU 收益不大
    - GenRec/Transformer 推荐: SwiGLU 首选
    - 输出层 CTR/Sigmoid: 永远是 Sigmoid
    - 输出层多分类: 永远是 Softmax
    """
    if layer_type == "output_ctr":
        return "Sigmoid"
    elif layer_type == "output_multi":
        return "Softmax"
    elif model_type in ["transformer", "genrec"]:
        return "SwiGLU"                   # 现代 GenRec 标配
    elif model_type in ["dnn", "deepfm", "din"]:
        return "ReLU"                     # 传统推荐足够
    else:
        return "GELU"                     # 兜底
```

#### 11.6.5 损失函数——文字·公式·代码三位一体深度覆盖

搜推广的损失函数远比 CV 复杂。CTR 预估是点级二分类，Ranking 是对级比较，表示学习是批内对比。每种目标对应不同的损失函数设计。

**损失函数速览**:

| 损失函数 | 公式形式 | 输入形状 | 适用场景 |
| :--- | :--- | :--- | :--- |
| `BCEWithLogitsLoss` | `-(y·logσ(ŷ)+(1-y)·log(1-σ(ŷ)))` | `[B], [B]` | **CTR 预估(二分类)** |
| `CrossEntropyLoss` | `-log(softmax(x)[y])` | `[B,C], [B]` | 多分类/序列推荐 next-item |
| `MSELoss` | `mean((ŷ-y)²)` | `[B,1], [B,1]` | 评分/时长回归 |
| **BPR Loss** | `-log(σ(ŷ⁺ - ŷ⁻))` | `[B], [B]` | **Ranking,隐式反馈** |
| **InfoNCE** | `-log(e^{s⁺/τ}/Σe^{sᵢ/τ})` | `[B,D], [B,D]` | **对比学习/双塔召回** |
| **Triplet Loss** | `max(0, d⁺ - d⁻ + margin)` | `[B,D]×3` | 向量空间结构学习 |

---

##### 3.6.5.1 BCEWithLogitsLoss (CTR 预估标准)

**文字**:CTR 预估是二分类问题（点击=1，未点击=0）。`BCEWithLogitsLoss` 内部合并了 $\mathrm{sigmoid}$ + BCE，数值稳定性优于先 $\mathrm{sigmoid}$ 再 BCE。

**公式**:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \bigl[y_i \log\sigma(\hat{y}_i) + (1-y_i)\log(1-\sigma(\hat{y}_i))\bigr]$$

$$\text{正负样本不平衡时加 pos\_weight:} \quad \mathcal{L} = -\bigl[w \cdot y \log\sigma(\hat{y}) + (1-y)\log(1-\sigma(\hat{y}))\bigr]$$

**代码**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ——— 基础用法 ———
loss_fn = nn.BCEWithLogitsLoss()  # 内部: sigmoid + BCE, 数值稳定
logits = torch.randn(1024)        # 模型输出 (未 sigmoid)
labels = torch.randint(0, 2, (1024,)).float()

loss = loss_fn(logits, labels)
print(f"BCE loss: {loss.item():.4f}")

# ——— 正负样本不平衡时 ———
# 例: CTR=1%, 正样本极少 → 需要对正样本加权
pos_ratio = 0.01
neg_ratio = 1 - pos_ratio
pos_weight = torch.tensor([neg_ratio / pos_ratio])  # ≈ 99

loss_fn_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ——— 为什么不用 BCELoss? ———
# 危险写法: sigmoid → BCELoss
pred_sigmoid = torch.sigmoid(logits)
# loss = nn.BCELoss()(pred_sigmoid, labels)  # 当 sigmoid 输出接近 0/1 时,log(0) = -inf!

# 安全写法: 直接用 BCEWithLogitsLoss (内部用 log-sum-exp trick)
# log σ(x) = x - log(1 + e^x)  (x>0 时稳定), 详见 §2.5
```

---

##### 3.6.5.2 BPR Loss (Bayesian Personalized Ranking)——排序的正统损失

**文字**:BPR 是为**隐式反馈**数据设计的排序损失。我们不知道用户"喜欢什么"，只知道"用户点击了什么"——所以我们让模型学习：正样本（点击过的物品）的打分应高于负样本（未点击的物品）。注意：BPR 不比较绝对分数，只比较相对排序——这是 Ranking 损失与 BCE 损失的根本区别。

**公式**:

$$\mathcal{L}_{\text{BPR}} = -\frac{1}{N}\sum_{i=1}^N \log\sigma(\hat{y}_{u,i^+} - \hat{y}_{u,i^-})$$

$$\text{梯度:} \quad \frac{\partial \mathcal{L}}{\partial \hat{y}^+} = -\sigma(\hat{y}^- - \hat{y}^+), \quad \frac{\partial \mathcal{L}}{\partial \hat{y}^-} = \sigma(\hat{y}^- - \hat{y}^+)$$

**BPR vs BCE 对比**:

```text
                BCE                       BPR
目标          预测点击概率               让正样本排在负样本前
需要          绝对 label (0/1)           正样本 + 负样本对
监督信号      强(显式 0/1)              弱(只比较相对顺序)
适用           CTR/评分有显式标签        隐式反馈(只有点击日志)
线上指标       AUC,Logloss              NDCG, HitRate
```

**代码**:

```python
def bpr_loss(pos_scores, neg_scores, reduction='mean'):
    """
    BPR 损失: 让 pos_score > neg_score

    pos_scores: [B] 正样本(用户点击过的)的预测分
    neg_scores: [B] 负样本(随机采样的)的预测分

    搜推广典型场景:
      pos = 用户最近点击的物品
      neg = 随机采样的未点击物品 (或困难负样本)
    """
    diff = pos_scores - neg_scores           # [B] 正负分差
    loss = -F.logsigmoid(diff)              # 等价于 -log(σ(diff))

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# ——— 使用示例 ———
B, D = 512, 64
user_emb   = torch.randn(B, D)           # 用户 Embedding
pos_item   = torch.randn(B, D)           # 正样本物品 Embedding
neg_item   = torch.randn(B, D)           # 负样本物品 Embedding

# 内积作为打分函数
pos_score = (user_emb * pos_item).sum(-1)   # [B]
neg_score = (user_emb * neg_item).sum(-1)   # [B]

loss = bpr_loss(pos_score, neg_score)
print(f"BPR loss: {loss.item():.4f}")

# ——— 多负样本 BPR (每个正样本对应 K 个负样本) ———
def bpr_loss_multi_neg(pos_scores, neg_scores):
    """
    pos_scores: [B]
    neg_scores: [B, K]  每个正样本有 K 个负样本
    """
    # 广播: [B, 1] - [B, K] = [B, K]
    diffs = pos_scores.unsqueeze(1) - neg_scores
    return -F.logsigmoid(diffs).mean()

K = 4  # 每个正样本 4 个负样本
neg_items_multi = torch.randn(B, K, D)
neg_scores_multi = (user_emb.unsqueeze(1) * neg_items_multi).sum(-1)  # [B, K]
loss_multi = bpr_loss_multi_neg(pos_score, neg_scores_multi)
print(f"Multi-neg BPR loss: {loss_multi.item():.4f}")
```

---

##### 3.6.5.3 InfoNCE / 对比损失 (双塔召回的标准)

**文字**:InfoNCE（信息噪声对比估计）是双塔召回模型的核心损失。直觉：在一个 batch 里，每个用户只有一个"正确"的物品（他点击的），其他 B-1 个物品是"batch 内负样本"。模型要学会把正确的 (user, item) 对排到最高位。

**公式**:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(s(u_i, v_i) / \tau)}{\sum_{j=1}^B \exp(s(u_i, v_j) / \tau)}$$

$$\text{其中 } s(u, v) = \frac{u \cdot v}{\|u\| \|v\|} \text{ (余弦相似度)}, \quad \tau \text{ 为温度参数}$$

**公式直觉**:

- **分子**:第 $i$ 个用户和他的正确物品 $v_i$ 的相似度(要最大化)
- **分母**:第 $i$ 个用户和 batch 中全部 $B$ 个物品的相似度(要最小化)
- **温度 $\tau$**:
  - $\tau \to 0$:分布越来越峰锐(只认最相似的)
  - $\tau \to \infty$:分布越来越平(等价于 uniform,训练信号消失)
  - $\tau = 0.07$:工业界经典默认值

**为什么 InfoNCE ≈ 互信息的下界?**

$$I(U; V) \geq \log(B) - L_{\mathrm{InfoNCE}}$$

最大化 InfoNCE $\equiv$ 最大化用户和物品之间的互信息。

**代码**:

```python
import torch
import torch.nn.functional as F

def infonce_loss(user_emb, item_emb, temperature=0.07):
    """
    InfoNCE / NT-Xent 对比损失

    user_emb: [B, D]  用户 Embedding
    item_emb: [B, D]  对应的正样本物品 Embedding

    batch 内负样本: 每个用户的负样本是同一 batch 里其他用户的物品
    这是工业界双塔召回的标准训练方式!
    """
    # 归一化到单位球面 (余弦相似度)
    u = F.normalize(user_emb, dim=-1)  # [B, D]
    v = F.normalize(item_emb, dim=-1)  # [B, D]

    # 相似度矩阵: [B, B]  (对角线是正样本对)
    sim_matrix = torch.matmul(u, v.T) / temperature  # [B, B]

    # labels: 第 i 个用户的正样本是第 i 个位置
    labels = torch.arange(u.size(0), device=u.device)

    # CrossEntropyLoss: 对每行做 softmax,然后取对角线的 log prob
    # 等价于 InfoNCE 的公式
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# ——— 使用示例 ———
B, D = 256, 128
user_emb = torch.randn(B, D)
item_emb = torch.randn(B, D)   # 每个与对应的 user 是正样本对

loss = infonce_loss(user_emb, item_emb, temperature=0.07)
print(f"InfoNCE loss: {loss.item():.4f}")

# 理论下界: 完美模型 loss = -log(1/B) = log(B)
# 随机初始化: loss ≈ log(B) = {:.2f}".format(math.log(B)))

# ——— 双向 InfoNCE (用户→物品 + 物品→用户) ———
def bidirectional_infonce(user_emb, item_emb, temperature=0.07):
    """
    同时优化:
      - 用户 query: 给定 user, 从 B 个 item 中找正确的那个
      - 物品 query: 给定 item, 从 B 个 user 中找正确的那个
    双向对称增强训练信号
    """
    u = F.normalize(user_emb, dim=-1)
    v = F.normalize(item_emb, dim=-1)
    sim = torch.matmul(u, v.T) / temperature  # [B, B]
    labels = torch.arange(u.size(0), device=u.device)

    loss_u2v = F.cross_entropy(sim, labels)      # 用户找物品
    loss_v2u = F.cross_entropy(sim.T, labels)    # 物品找用户
    return (loss_u2v + loss_v2u) / 2

# ——— 温度参数的影响可视化 ———
for tau in [0.01, 0.07, 0.1, 0.5, 1.0]:
    loss = infonce_loss(user_emb, item_emb, temperature=tau)
    print(f"tau={tau:.2f}: loss={loss.item():.3f}")
```

---

##### 3.6.5.4 Triplet Loss (向量空间结构约束)

**文字**:Triplet Loss 从三元组角度学习：让 anchor（查询）离 positive（正样本）更近，离 negative（负样本）更远，两者距离差至少为 margin。与 InfoNCE 的区别：Triplet 是硬性约束（每次只看一个负样本），InfoNCE 是软性约束（同时比较所有负样本）。

**公式**:

$$\mathcal{L}_{\text{Triplet}} = \frac{1}{N}\sum_{i=1}^N \max\bigl(0,\; d(a_i, p_i) - d(a_i, n_i) + \text{margin}\bigr)$$

$$\text{其中 } d(x, y) = \|x - y\|_2 \text{ (欧氏距离)} \text{ 或 } 1 - \cos(x, y) \text{ (余弦距离)}$$

**代码**:

```python
# ——— 方式 1: PyTorch 内置 ———
triplet_fn = nn.TripletMarginLoss(
    margin=0.3,        # anchor-neg 距离比 anchor-pos 距离至少大 margin
    p=2,               # L2 距离
    reduction='mean'
)

anchor   = torch.randn(B, D)   # 查询 (用户或目标物品)
positive = torch.randn(B, D)   # 正样本
negative = torch.randn(B, D)   # 负样本

loss = triplet_fn(anchor, positive, negative)

# ——— 方式 2: 余弦距离版本 ———
def triplet_cosine_loss(anchor, positive, negative, margin=0.1):
    """
    用余弦距离的 Triplet Loss

    搜推广场景: 物品 Embedding 空间对齐
      anchor:   目标物品
      positive: 同类别/同用户购买过的物品
      negative: 不同类别的物品
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negative, dim=-1)

    # 余弦距离 = 1 - 余弦相似度 (值域 [0, 2])
    d_pos = 1 - (a * p).sum(-1)   # [B]
    d_neg = 1 - (a * n).sum(-1)   # [B]

    loss = F.relu(d_pos - d_neg + margin)   # max(0, d+ - d- + margin)
    return loss.mean()

# ——— 困难负样本挖掘 (Hard Negative Mining) ———
def hard_triplet_loss(embeddings, labels, margin=0.3):
    """
    在 batch 内挖掘"最困难"的正负样本对
    困难正样本: 同类别中距离最远的
    困难负样本: 不同类别中距离最近的 (最容易被误判)

    embeddings: [B, D]
    labels:     [B]  同一 label 的是正样本对
    """
    # 计算 batch 内两两距离: [B, B]
    dist = torch.cdist(embeddings, embeddings, p=2)

    # 正样本对 mask: 同 label, 且不是自身
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & \
               ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    # 负样本对 mask: 不同 label
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    # 困难正样本: 同类中距离最大的
    hardest_pos = (dist * pos_mask.float()).max(dim=1).values
    # 困难负样本: 不同类中距离最小的
    dist_neg = dist.clone()
    dist_neg[~neg_mask] = float('inf')
    hardest_neg = dist_neg.min(dim=1).values

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[loss > 0].mean() if (loss > 0).any() else loss.mean()
```

---

**搜推广损失函数选型决策**:

```text
任务类型                          推荐损失                注意事项
──────────────────────────────────────────────────────────────────
CTR 预估 (点级有标签)            BCEWithLogitsLoss       正负不平衡→ pos_weight
序列推荐 next-item 预测          CrossEntropyLoss(全库)  softmax 复杂度 O(N)
双塔召回 (Embedding 对齐)        InfoNCE                 batch_size ≥ 256 才有效
CF 排序 (隐式反馈)               BPR Loss                困难负样本提升效果
向量空间结构约束                  Triplet Loss (cos)      需要 Hard Mining
多目标 (CTR+CVR+时长)            BCE × 多个头            注意 loss 量级对齐
```

#### 11.6.6 归一化层——文字·公式·代码三位一体深度对比

归一化是深度学习训练稳定的核心机制。理解各种归一化的差别才能正确选型。

**总览对比**:

| 方法 | 归一化维度 | 均值项 | 可学习参数 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| BatchNorm | Batch 维 (N) | ✅ 减均值 | γ, β | CNN、大 batch |
| LayerNorm | 特征维 (C) | ✅ 减均值 | γ, β | **Transformer 首选** |
| RMSNorm | 特征维 (C) | ❌ 不减均值 | γ | **LLM/GenRec 2024 标准** |
| GroupNorm | 分组特征维 | ✅ 减均值 | γ, β | 小 batch 的 CNN |
| InstanceNorm | 每样本每通道 | ✅ 减均值 | γ, β | 风格迁移 |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D = 32, 50, 64
x = torch.randn(B, T, D)

# ═══════════════════════════════════════════
# BatchNorm
# ═══════════════════════════════════════════
# 文字: 对同一特征通道,在 batch 维度内做归一化
# 公式: y = γ · (x - μ_B) / √(σ²_B + ε) + β
#        μ_B = (1/N) Σ_{n=1}^N x_n,  σ²_B = (1/N) Σ (x_n - μ_B)²
# 注意: 推理时用 running mean/var 而不是 batch 统计量!
bn = nn.BatchNorm1d(D)
x_2d = x.view(-1, D)   # BN 要求 [B, D] 或 [B, D, T]
y_bn = bn(x_2d)         # [B*T, D]

# ═══════════════════════════════════════════
# LayerNorm
# ═══════════════════════════════════════════
# 文字: 对同一样本的所有特征维度做归一化,与 batch 大小无关
# 公式: y = γ · (x - μ_L) / √(σ²_L + ε) + β
#        μ_L = mean over D,  σ²_L = var over D
# 优势: 对 batch size 不敏感 → Transformer 的天然选择
ln = nn.LayerNorm(D)
y_ln = ln(x)             # [B, T, D]

# 等价的函数式写法 (更灵活,可自定义 normalized_shape)
y_ln_f = F.layer_norm(x, normalized_shape=[D], weight=ln.weight, bias=ln.bias)

# ═══════════════════════════════════════════
# RMSNorm (手动实现,PyTorch 2.4+ 有 F.rms_norm)
# ═══════════════════════════════════════════
# 文字: 去掉均值项的 LayerNorm — "不关心均值,只关心量级"
# 公式: y = γ · x / √(mean(x²) + ε)
# 为什么省去均值?
#   高维空间 (D=4096) 中, mean(x) ≈ 0 (大数定律)
#   去掉减均值操作,节省 ~1 次 reduce, 速度快 5-10%
class RMSNorm(nn.Module):
    def __init__(self, D, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(D))
        self.eps = eps

    def forward(self, x):
        # 公式: x / √(mean(x²) + ε) × γ
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms

rn = RMSNorm(D)
y_rn = rn(x)   # [B, T, D]

# ═══════════════════════════════════════════
# 三种归一化的输出分布验证
# ═══════════════════════════════════════════
print("输出均值 (应接近 0):")
print(f"  BatchNorm:  {y_bn.mean().item():.4f}")
print(f"  LayerNorm:  {y_ln.mean().item():.4f}")
print(f"  RMSNorm:    {y_rn.mean().item():.4f}")  # 不强制为 0

print("\n输出标准差 (应接近 1):")
print(f"  BatchNorm:  {y_bn.std().item():.4f}")
print(f"  LayerNorm:  {y_ln.std().item():.4f}")
print(f"  RMSNorm:    {y_rn.std().item():.4f}")  # 近似 1 但不精确

# ═══════════════════════════════════════════
# 搜推广选型建议
# ═══════════════════════════════════════════
"""
传统 DLRM / DNN 精排: BatchNorm + 每层 MLP
  → 批量大 (batch=4096), 特征固定, BN 效果好

Transformer 序列推荐 (SASRec, DIN, BERT4Rec):
  → LayerNorm (标配)

生成式推荐 (GenRec, HSTU), LLM 系结构:
  → Pre-RMSNorm (2024 年新标准)

在线推理: 小 batch 或 batch=1 时, BatchNorm 会退化 (running 统计失效)
  → 改用 LayerNorm 或 RMSNorm 更安全
"""
```

#### 11.6.7 Dropout 变体——文字·公式·代码三位一体

标准 Dropout 的数学定义（inverted dropout，PyTorch 默认实现）:

$$y_i = \frac{x_i \cdot m_i}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p) \quad \text{(训练时)}$$

$$y_i = x_i \quad \text{(推理时, } m_i=1 \text{, 无需 rescale)}$$

除以 $(1-p)$ 是为了保证训练/推理时期望相同：$\mathbb{E}[y_i] = x_i$（训练和推理期望一致）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D = 32, 50, 64
x = torch.randn(B, T, D)

# ═══════════════════════════════════════════
# 标准 Dropout
# ═══════════════════════════════════════════
# 文字: 训练时随机把部分神经元的输出置为 0,推理时恢复
# 公式: y_i = x_i / (1-p) · Bernoulli(1-p)  (训练)
#        y_i = x_i                              (推理)
# 数学意义: 等价于训练了 2^n 个共享权重的"瘦模型"的集成
dp = nn.Dropout(p=0.1)    # p=0.1: 10% 的神经元被置零
y_dp = dp(x)               # 训练时有效; 推理时 model.eval() 后自动跳过

# 等价函数式写法:
y_dp_f = F.dropout(x, p=0.1, training=True)

# ═══════════════════════════════════════════
# Dropout2d (按通道 Dropout)
# ═══════════════════════════════════════════
# 文字: 整个特征通道(所有时间步)一起置零,而不是逐元素
# 适合 Embedding 列:同一特征的所有 token 一起 drop
# 公式: 对 [B, T, D] 中的某个 d 维度,整列置零
x_4d = x.permute(0, 2, 1).unsqueeze(-1)   # [B, D, T, 1] (Dropout2d 要求)
dp2d = nn.Dropout2d(p=0.1)
y_dp2d = dp2d(x_4d).squeeze(-1).permute(0, 2, 1)  # 恢复 [B, T, D]

# ═══════════════════════════════════════════
# 搜推广中 Dropout 的三个关键规则
# ═══════════════════════════════════════════
"""
规则 1: Embedding 层后用 Dropout (p=0.1~0.3)
  → 防止 Embedding 过拟合到训练集的高频 ID
  → 等价于 ID 的"随机遮挡",增强鲁棒性

规则 2: MLP 中每层 ReLU/GELU 后用 Dropout (p=0.1~0.5)
  → 标准正则化,防止 MLP 过拟合

规则 3: Attention 权重用 Dropout (attn_dropout)
  → 在注意力权重矩阵 [B, nh, T, T] 上 dropout
  → 强制注意力不过度依赖某几个位置
  → 在 nn.MultiheadAttention(dropout=0.1) 中自动处理

不要用 Dropout 的场景:
  - Embedding 的权重本身 (会破坏稀疏梯度的统计性质)
  - 推理时 (model.eval() 后自动关闭,确保 model.eval() 没漏)
"""
```

---

### 3.6.8 Attention 机制完整谱系——从原理到工程的全面解析

> **为什么专门立一节讲 Attention?**
>
> Attention 是现代推荐系统最核心的"信息交互"原语。DIN 用它建模用户兴趣，SASRec 用它做序列推荐，双塔用它做跨模态融合，HSTU 用它替代了整个 MLP 塔，LLM 风格的 GenRec 更是全程依赖 Attention。理解 Attention 的每一个变体——不只是会调 `nn.MultiheadAttention`，而是知道每种变体为什么被发明，解决了什么问题——是从 CV 工程师转型搜推广研究者的必经之路。
>
> 本节用**文字→公式→代码→对比**四位一体覆盖 9 种 Attention。

---

#### 11.6.8.1 Attention 的统一框架——一个公式统领所有变体

**文字**:注意力机制的本质是**加权信息聚合**:有一个"查询"(Query),它去"检索"一组"键值对"(Key-Value),检索的依据是 Query 与每个 Key 的相似度,相似度高的 Value 贡献更大。

**公式(通用框架)**:

$$\text{Attention}(Q, K, V) = \text{Aggregate}\bigl(\text{Score}(Q, K),\; V\bigr)$$

$$\text{其中:} \quad \text{Score}(Q, K) = \begin{cases} Q \cdot K & \text{(点积)} \\ \text{MLP}([Q; K]) & \text{(加性)} \\ \text{custom} & \text{(其他)} \end{cases}$$

**所有 Attention 变体的差异只在三处**:

| 变体 | Score 函数 | Aggregate 方式 | 关键约束 |
| :--- | :--- | :--- | :--- |
| Bahdanau (加性) | $v^\top \tanh(W_Q q + W_K k)$ | softmax 加权 | 参数多 |
| Scaled Dot-Product | $qk^\top / \sqrt{d_k}$ | softmax 加权 | 高效,主流 |
| Multi-Head | 多个子空间的 Dot-Product | concat + 线性 | 多视角 |
| Cross-Attention | Q 来自一个序列,K/V 来自另一个 | softmax 加权 | 跨模态 |
| DIN 注意力 | MLP 对 [q,k,q-k,q⊙k] 打分 | sigmoid 加权 | 无需 softmax |
| MQA/GQA | Dot-Product,共享 K/V | softmax 加权 | 省 KV cache |
| MLA | Dot-Product,低秩 K/V | softmax 加权 | 极省 cache |
| Sliding Window | Dot-Product,局部窗口 | softmax 加权 | O(Tw) 复杂度 |
| Flash Attention | Dot-Product (IO-aware) | softmax 加权 | O(N) 显存 |

---

#### 11.6.8.2 加性注意力 (Bahdanau Attention)

**文字**:最早的注意力机制(Bahdanau 2015, seq2seq 机器翻译)。不用点积,而是用一个小 MLP 来打分。好处是对 Q/K 的维度不敏感。

**公式**:

$$e_i = v^\top \tanh(W_Q\, q + W_K\, k_i) \quad \in \mathbb{R}$$

$$\alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)} \quad \text{(softmax)}$$

$$\text{output} = \sum_i \alpha_i\, v_i$$

**代码**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Bahdanau (加性) 注意力
    - Q: 解码器的当前隐状态  [B, D]
    - K: 编码器的全部输出    [B, T, D]
    - V: 通常 = K

    搜推广场景: 历史行为序列中,用"目标物品"作为 Query
    查询"哪些历史行为最相关" — 比点积注意力在小维度时更稳定
    """
    def __init__(self, query_dim, key_dim, hidden_dim=64):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        query:  [B, D_q]
        keys:   [B, T, D_k]
        values: [B, T, D_v]
        mask:   [B, T] bool, True=padding (需要被屏蔽)
        """
        # 扩展 query 以便与 keys 相加: [B, 1, H]
        q_proj = self.W_q(query).unsqueeze(1)   # [B, 1, H]
        k_proj = self.W_k(keys)                  # [B, T, H]

        # 加性: 先加再过 tanh
        # 广播: [B, 1, H] + [B, T, H] → [B, T, H]
        scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [B, T]

        # 掩码 padding 位置
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [B, T]

        # 加权聚合
        # [B, T] unsqueeze → [B, 1, T], @ [B, T, D_v] → [B, 1, D_v] → [B, D_v]
        output = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return output, weights   # 返回注意力权重供可视化

# ——— 使用示例 ———
B, T, D = 32, 50, 64
query  = torch.randn(B, D)         # 目标物品 Embedding
keys   = torch.randn(B, T, D)      # 历史行为序列 Embedding
values = keys.clone()               # V = K (常见设置)
pad_mask = torch.zeros(B, T, dtype=torch.bool)
pad_mask[:, 30:] = True            # 后 20 个位置是 padding

attn = AdditiveAttention(D, D, hidden_dim=32)
output, weights = attn(query, keys, values, mask=pad_mask)
print(f"output: {output.shape}")   # [32, 64]
print(f"weights: {weights.shape}") # [32, 50]
# weights 可视化: 看哪些历史行为被模型关注到
```

---

#### 11.6.8.3 缩放点积注意力 (Scaled Dot-Product Attention)

**文字**:Transformer(Vaswani 2017)提出的主流注意力。用点积代替加性 MLP,计算更高效;除以 $\sqrt{d_k}$ 防止 $\mathrm{softmax}$ 饱和。这是目前所有主流大模型的基础注意力单元。

**公式**:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$$\text{为什么除以} \sqrt{d_k}? \quad \text{Var}[q \cdot k] = d_k \xrightarrow{\div\sqrt{d_k}} \text{Var}=1 \Rightarrow \text{softmax 梯度不消失}$$

**代码**:

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """
    标准缩放点积注意力 — Transformer 原文实现

    q: [B, nh, T_q, d_k]   查询
    k: [B, nh, T_k, d_k]   键
    v: [B, nh, T_k, d_v]   值
    mask: [B, 1, T_q, T_k] bool, True=屏蔽

    返回: [B, nh, T_q, d_v]
    """
    d_k = q.size(-1)

    # 步骤 1: 点积相似度  [B, nh, T_q, T_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 步骤 2: 掩码(因果/padding)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 步骤 3: softmax → 注意力权重
    attn_weights = F.softmax(scores, dim=-1)   # [B, nh, T_q, T_k]

    # 步骤 4: dropout(训练时随机丢弃注意力路径)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # 步骤 5: 加权聚合
    output = torch.matmul(attn_weights, v)     # [B, nh, T_q, d_v]

    return output, attn_weights

# ——— PyTorch 2.0+ 内置的高效版本 ———
# torch.nn.functional.scaled_dot_product_attention
# 会自动选择 Flash Attention / Memory-Efficient Attention

B, nh, T, d_k = 32, 8, 50, 64
q = torch.randn(B, nh, T, d_k)
k = torch.randn(B, nh, T, d_k)
v = torch.randn(B, nh, T, d_k)

# 因果掩码(自回归, 只看左边)
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)   # [1, 1, T, T]

# 手动实现
out_manual, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)

# PyTorch 原生(推荐!内部自动用 Flash Attention)
out_native = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=~causal_mask,   # 注意: PyTorch 的 attn_mask=True 表示"允许",与上面相反
    dropout_p=0.0,
    is_causal=True             # 直接设置因果,比手动 mask 更高效
)
print(f"out shape: {out_native.shape}")   # [32, 8, 50, 64]
```

---

#### 11.6.8.4 多头注意力 (Multi-Head Attention, MHA)

**文字**:把 Q/K/V 投影到 $h$ 个子空间,每个子空间独立做 Attention,最后拼接输出。不同的"头"可以关注输入的不同方面(语义关系、位置关系、语法关系等),表达能力远强于单头。

**公式**:

$$\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O$$

$$\text{其中} \; W_i^Q \in \mathbb{R}^{D\times d_k},\; W_i^K \in \mathbb{R}^{D\times d_k},\; W_i^V \in \mathbb{R}^{D\times d_v},\; W^O \in \mathbb{R}^{hd_v\times D}$$

**代码(从零实现,对齐公式每一步)**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    标准多头注意力 — 从零实现,每一行对应公式

    参数:
        d_model: 模型维度 (= h × d_k)
        n_heads: 注意力头数 h
        dropout: 注意力权重的 dropout 概率
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # 每头的维度

        # 公式中的 W_i^Q, W_i^K, W_i^V — 用一个大矩阵一次投影,等价于 h 个小矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)   # [D, D] = [D, h×d_k]
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)   # 公式中的 W^O

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 [B, T, D] 拆成 [B, h, T, d_k]"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 [B, h, T, d_k] 合并回 [B, T, D]"""
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(self, query, key, value, mask=None):
        """
        query: [B, T_q, D]
        key:   [B, T_k, D]
        value: [B, T_k, D]
        mask:  [B, 1, T_q, T_k] or [B, 1, 1, T_k],  True=屏蔽

        返回: [B, T_q, D]
        """
        # 1. 线性投影: [B, T, D] → [B, T, D]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 拆分多头: [B, T, D] → [B, h, T, d_k]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B, h, T_q, T_k]

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. 加权聚合: [B, h, T_q, T_k] @ [B, h, T_k, d_k] → [B, h, T_q, d_k]
        context = torch.matmul(attn_weights, V)

        # 5. 合并多头: [B, h, T_q, d_k] → [B, T_q, D]
        context = self.merge_heads(context)

        # 6. 输出线性: 公式中的 W^O
        output = self.W_o(context)

        return output, attn_weights

# ——— 使用示例 ———
B, T, D = 32, 50, 256
n_heads = 8

x = torch.randn(B, T, D)
mha = MultiHeadAttention(d_model=D, n_heads=n_heads, dropout=0.1)
output, weights = mha(x, x, x)     # 自注意力: Q=K=V=x

print(f"输入:  {x.shape}")         # [32, 50, 256]
print(f"输出:  {output.shape}")    # [32, 50, 256]
print(f"权重:  {weights.shape}")   # [32, 8, 50, 50]

# PyTorch 内置版(推荐,更高效)
mha_native = nn.MultiheadAttention(D, n_heads, dropout=0.1, batch_first=True)
out_native, weights_native = mha_native(x, x, x)
```

---

#### 11.6.8.5 交叉注意力 (Cross-Attention)

**文字**:Q 来自一个序列(如目标物品/当前 token),K/V 来自另一个序列(如用户历史/编码器输出)。Q "查询" K/V,让两个序列之间产生信息交流。这是 Encoder-Decoder 结构的核心,也是多模态融合的基础。

**公式**:

$$\text{CrossAttn}(x_\text{tgt}, x_\text{src}) = \text{softmax}\!\left(\frac{(x_\text{tgt}W^Q)(x_\text{src}W^K)^\top}{\sqrt{d_k}}\right)(x_\text{src}W^V)$$

**代码**:

```python
class CrossAttention(nn.Module):
    """
    交叉注意力 — Q 来自 target, K/V 来自 source

    搜推广场景:
      1. 用"目标物品 Embedding"作为 Q, "用户历史序列"作为 K/V
         → 让目标物品"查询"用户的哪些历史行为与它相关
      2. 文本特征 + ID 特征融合: 文本作为 K/V, ID 作为 Q
         → 用 ID 的语义从文本中提取相关信息
      3. Encoder-Decoder 推荐: 历史作为 Encoder, 候选列表作为 Decoder Q
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        # Q 来自 target
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # K, V 来自 source
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, source, src_mask=None):
        """
        target: [B, T_tgt, D]  — Query 的来源
        source: [B, T_src, D]  — Key/Value 的来源
        src_mask: [B, 1, 1, T_src] bool

        返回: [B, T_tgt, D]  — target 经过 source 信息增强后的表示
        """
        B = target.size(0)
        T_tgt, T_src = target.size(1), source.size(1)

        # Q 来自 target,K/V 来自 source
        Q = self.W_q(target).view(B, T_tgt, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(source).view(B, T_src, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(source).view(B, T_src, self.n_heads, self.d_k).transpose(1, 2)

        # [B, nh, T_tgt, T_src]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, float('-inf'))

        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, V)   # [B, nh, T_tgt, d_k]

        output = self.W_o(
            context.transpose(1, 2).contiguous().view(B, T_tgt, -1)
        )
        return output, weights

# ——— 搜推广典型用法: 目标物品 × 用户历史 ———
B, D = 32, 128
item_emb  = torch.randn(B, 1, D)     # 目标物品: [B, 1, D] (单个 query)
hist_emb  = torch.randn(B, 50, D)    # 用户历史: [B, T, D] (key/value 序列)

cross_attn = CrossAttention(D, n_heads=8)
enhanced_item, weights = cross_attn(item_emb, hist_emb)
# enhanced_item: [B, 1, D] — 融入用户兴趣信息后的物品表示
# weights: [B, 8, 1, 50] — 哪些历史行为被关注(可解释性!)

print(f"增强后物品表示: {enhanced_item.squeeze(1).shape}")  # [32, 128]
```

---

#### 11.6.8.6 DIN 注意力 (Deep Interest Network)——推荐专属

**文字**:阿里 2018 年的经典工作。不用 $\mathrm{softmax}$ 归一化(因为不同用户的历史长度差异大),改用 $\mathrm{sigmoid}$ 对每个历史行为独立打分,允许注意力权重之和不为 1。评分函数比点积更复杂,包含差值和乘积信号。

**公式**:

$$e_i = \text{MLP}\bigl([q;\, k_i;\, q - k_i;\, q \odot k_i]\bigr) \in \mathbb{R}$$

$$\alpha_i = \text{sigmoid}(e_i) \quad \text{(不做 softmax!)}$$

$$\text{output} = \sum_i \alpha_i\, v_i$$

**为什么不用 softmax?**

- **softmax**:$\sum \alpha_i = 1$ → 历史长的用户的每个行为平均权重更小
- **sigmoid**:$\sum \alpha_i$ 不固定 → 长历史用户可以累积更大的注意力总分 → 模型自然学会"活跃用户的历史整体都重要"

**代码**:

```python
class DINAttention(nn.Module):
    """
    DIN (Deep Interest Network) 注意力

    核心创新:
    1. 用 MLP([q, k, q-k, q⊙k]) 打分,而不是点积
    2. sigmoid 替代 softmax,不强制权重归一
    3. PReLU 激活函数(可学习斜率)
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 36):
        super().__init__()
        # 输入: [q, k, q-k, q⊙k] 拼接 = 4×emb_dim
        self.score_net = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.PReLU(),                         # DIN 原文用 PReLU
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()                         # sigmoid, 不是 softmax!
        )

    def forward(self, target_item, hist_seq, hist_mask=None):
        """
        target_item: [B, D]       当前候选物品的 Embedding
        hist_seq:    [B, T, D]    用户历史行为序列
        hist_mask:   [B, T] bool, True=有效位置, False=padding

        返回:
          user_repr: [B, D]  — 针对当前物品的动态用户表示
          weights:   [B, T]  — 每个历史行为的注意力得分
        """
        T = hist_seq.size(1)

        # 扩展 target_item 以匹配序列长度: [B, D] → [B, T, D]
        target_expanded = target_item.unsqueeze(1).expand(-1, T, -1)

        # 构造交互特征: [B, T, 4D]
        interaction = torch.cat([
            target_expanded,                               # [B, T, D]: q (候选物品)
            hist_seq,                                      # [B, T, D]: k_i (历史行为)
            target_expanded - hist_seq,                    # [B, T, D]: q - k_i (差异)
            target_expanded * hist_seq,                    # [B, T, D]: q ⊙ k_i (协同)
        ], dim=-1)                                         # [B, T, 4D]

        # 打分: [B, T, 4D] → [B, T, 1] → [B, T]
        scores = self.score_net(interaction).squeeze(-1)   # [B, T]

        # 屏蔽 padding 位置(置 0 权重)
        if hist_mask is not None:
            scores = scores * hist_mask.float()

        # 加权聚合 (不做 softmax!)
        # [B, T] → [B, 1, T],  @ [B, T, D] → [B, 1, D] → [B, D]
        user_repr = torch.bmm(scores.unsqueeze(1), hist_seq).squeeze(1)

        return user_repr, scores

# ——— 完整使用示例 ———
B, T, D = 32, 50, 64
candidate_item = torch.randn(B, D)         # 候选物品 Embedding
history = torch.randn(B, T, D)             # 历史行为序列
valid_mask = torch.ones(B, T, dtype=torch.bool)
valid_mask[:, 30:] = False                 # 后 20 个是 padding

din_attn = DINAttention(emb_dim=D, hidden_dim=36)
user_repr, attn_weights = din_attn(candidate_item, history, hist_mask=valid_mask)

print(f"用户兴趣表示: {user_repr.shape}")    # [32, 64]
print(f"注意力权重: {attn_weights.shape}")   # [32, 50]
print(f"权重之和(非 1): {attn_weights[0].sum():.3f}")  # ≠ 1 (因为用 sigmoid)

# ——— DIN vs 标准 Attention 的对比 ———
"""
            DIN 注意力              标准 Scaled Dot-Product
权重归一化   sigmoid(不归一)         softmax(归一到 1)
评分函数     MLP([q,k,q-k,q⊙k])     q·k / √d
参数量       4D×H + H×1              无额外参数
可解释性     高 (sigmoid 权重直接解读)  高
适合场景     精排 (历史长度不均匀)    序列建模 (长度统一)
计算复杂度   O(T × D × H)            O(T² × D)
"""
```

---

#### 11.6.8.7 多查询/分组查询注意力 (MQA / GQA)

**文字**:推理阶段 KV-Cache 是显存的主要占用者。MQA 让所有 Query 头共享**同一组** K/V,GQA 是折中:每组 Query 头共享一组 K/V。

**公式(GQA)**:

$$Q \in \mathbb{R}^{B \times n_q \times T \times d}, \quad K,V \in \mathbb{R}^{B \times n_{kv} \times T \times d}$$

$$\text{令 } g = n_q / n_{kv}, \quad K_i' = \text{repeat}(K_i, g), \quad V_i' = \text{repeat}(V_i, g)$$

$$\text{head}_i = \text{Attention}(Q_i, K'_{\lfloor i/g\rfloor}, V'_{\lfloor i/g\rfloor})$$

**代码**:

```python
class GroupedQueryAttention(nn.Module):
    """
    GQA: n_kv_heads 组 KV, n_heads 组 Q
    MQA = GQA with n_kv_heads=1
    MHA = GQA with n_kv_heads=n_heads

    显存节省:
      MHA KV cache = 2 × n_heads × d_k × seq_len × bytes
      GQA KV cache = 2 × n_kv_heads × d_k × seq_len × bytes
      节省比 = n_heads / n_kv_heads

    LLaMA 3 8B: n_heads=32, n_kv_heads=8 → 节省 4×
    """
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_heads // n_kv_heads   # 每个 KV 头服务几个 Q 头
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # Q: n_heads 个头;  K/V: n_kv_heads 个头
        Q  = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K  = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V  = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # GQA 核心: 把 K/V 的 n_kv_heads 复制到 n_heads
        # [B, n_kv_heads, T, d_k] → [B, n_heads, T, d_k]
        K = K.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        K = K.reshape(B, self.n_heads, T, self.d_k)
        V = V.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        V = V.reshape(B, self.n_heads, T, self.d_k)

        # 标准注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

# ——— 三种变体参数对比 ———
D, T = 512, 128

mha = GroupedQueryAttention(D, n_heads=8, n_kv_heads=8)   # MHA
gqa = GroupedQueryAttention(D, n_heads=8, n_kv_heads=2)   # GQA (4:1)
mqa = GroupedQueryAttention(D, n_heads=8, n_kv_heads=1)   # MQA

x = torch.randn(4, T, D)
print("参数量对比:")
for name, model in [("MHA", mha), ("GQA(4:1)", gqa), ("MQA", mqa)]:
    n = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n/1e6:.2f}M params")
# MHA: 1.05M, GQA: 0.79M, MQA: 0.66M
# KV cache 减少比: MHA:1×, GQA:4×, MQA:8×
```

---

#### 11.6.8.8 多头潜在注意力 (Multi-Head Latent Attention, MLA)

**文字**:DeepSeek-V2/V3 的核心创新。传统 MHA 的 KV-Cache 随序列长度线性增长,MLA 通过低秩压缩将 K/V 投影到一个极小的潜在空间存储,推理时再解压。相比 MHA 减少 KV-Cache **75%** 以上。

**公式**:

$$c_t = W_{\text{DKV}} h_t \in \mathbb{R}^{d_c} \quad \text{(压缩后的潜在向量, 只存这个)}$$

$$K_t = W_{\text{UK}} c_t, \quad V_t = W_{\text{UV}} c_t \quad \text{(推理时即时解压)}$$

$$\text{KV-Cache} = \{c_t\}_{t=1}^T \quad \text{而非} \{K_t, V_t\}_{t=1}^T$$

$$d_c \ll n_{kv} \cdot d_k \Rightarrow \text{cache 大幅压缩}$$

**代码**:

```python
class MultiHeadLatentAttention(nn.Module):
    """
    MLA — DeepSeek-V2/V3 的 KV Cache 压缩方案

    关键思想: 不缓存 K/V 本身,而是缓存它们的低秩压缩 c_t
    推理时用 W_UK / W_UV 将 c_t 解压回 K/V

    参数:
        d_model: 模型维度
        n_heads: 查询头数
        d_c:     KV 压缩维度 (远小于 n_heads × d_k)
        d_rope:  RoPE 的维度 (解耦位置编码用)
    """
    def __init__(self, d_model=512, n_heads=8, d_c=64, d_rope=32):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_c = d_c
        self.d_rope = d_rope
        self.scale = math.sqrt(self.d_k + d_rope)

        # Q 的投影: 先压缩再展开
        self.W_dq  = nn.Linear(d_model, d_c, bias=False)      # Q 压缩
        self.W_uq  = nn.Linear(d_c, n_heads * self.d_k, bias=False)  # Q 展开
        self.W_qr  = nn.Linear(d_c, n_heads * d_rope, bias=False)    # Q RoPE 部分

        # KV 的低秩压缩 — 这是关键!只存 d_c 维而不是 n_heads × d_k 维
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)      # KV 联合压缩
        self.W_uk  = nn.Linear(d_c, n_heads * self.d_k, bias=False)  # K 展开
        self.W_uv  = nn.Linear(d_c, n_heads * self.d_k, bias=False)  # V 展开
        self.W_kr  = nn.Linear(d_model, d_rope, bias=False)           # K RoPE 部分

        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        """
        训练时: kv_cache=None, 正常计算
        推理时: kv_cache 存储历史的 c_t (压缩后的 KV)

        x: [B, T, D]
        返回: (output [B, T, D], new_kv_cache)
        """
        B, T, D = x.shape
        nh, dk = self.n_heads, self.d_k

        # ——— Q 的计算 ———
        c_q = self.W_dq(x)                          # [B, T, d_c]  Q 潜在向量
        Q   = self.W_uq(c_q).view(B, T, nh, dk)    # [B, T, nh, dk]

        # ——— KV 的低秩压缩 ———
        c_kv = self.W_dkv(x)                        # [B, T, d_c]  关键: 只缓存这个!
        K    = self.W_uk(c_kv).view(B, T, nh, dk)  # [B, T, nh, dk]
        V    = self.W_uv(c_kv).view(B, T, nh, dk)  # [B, T, nh, dk]

        if kv_cache is not None:
            # 推理时拼接历史 cache
            c_kv_hist = kv_cache   # [B, T_past, d_c]
            c_kv_all  = torch.cat([c_kv_hist, c_kv], dim=1)   # [B, T_past+T, d_c]
            K = self.W_uk(c_kv_all).view(B, -1, nh, dk)
            V = self.W_uv(c_kv_all).view(B, -1, nh, dk)

        # ——— 标准多头注意力 ———
        Q = Q.transpose(1, 2)   # [B, nh, T, dk]
        K = K.transpose(1, 2)   # [B, nh, T_all, dk]
        V = V.transpose(1, 2)   # [B, nh, T_all, dk]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)   # [B, nh, T, dk]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.W_o(out), c_kv   # 返回 c_kv 作为新的 cache

# ——— KV-Cache 显存对比 ———
B, T, D, n_heads = 1, 2048, 512, 8
d_k = D // n_heads
d_c = 64   # MLA 的压缩维度

mha_cache_bytes = 2 * n_heads * d_k * T * 4    # K + V,  FP32
mla_cache_bytes = d_c * T * 4                   # 只存 c_t

print(f"MHA KV-Cache: {mha_cache_bytes/1024:.1f} KB per layer")  # 512 KB
print(f"MLA KV-Cache: {mla_cache_bytes/1024:.1f} KB per layer")  # 8 KB
print(f"节省比: {mha_cache_bytes/mla_cache_bytes:.1f}×")         # 64×
```

---

#### 11.6.8.9 滑动窗口注意力 (Sliding Window Attention)

**文字**:Mistral 7B 提出。每个 token 只关注窗口内的邻居(局部注意力),复杂度从 $O(T^2)$ 降到 $O(T \cdot W)$。通过多层叠加,高层的感受野可以覆盖全序列——类似 CNN 的感受野增长。

**公式**:

$$\text{scores}[t, s] = \begin{cases} q_t \cdot k_s / \sqrt{d_k} & \text{if } |t - s| \leq W/2 \\ -\infty & \text{otherwise} \end{cases}$$

$$\text{有效感受野(L 层后)} = L \times W$$

**代码**:

```python
class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力 — Mistral 7B 的核心设计

    每个 token 只能看到窗口 W 范围内的上下文:
      位置 t 能关注: [t - W//2, t + W//2] 内的所有位置

    搜推广场景:
      超长用户行为序列 (T=1000+) 时,全注意力 O(T²)=10^6 太贵
      局部注意力 O(T×W)=T×128 更合适
      物理含义: "最近的 128 次行为最重要"
    """
    def __init__(self, d_model, n_heads, window_size=128, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        nh, dk, W = self.n_heads, self.d_k, self.window_size

        Q = self.W_q(x).view(B, T, nh, dk).transpose(1, 2)   # [B, nh, T, dk]
        K = self.W_k(x).view(B, T, nh, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, nh, dk).transpose(1, 2)

        # 构造滑动窗口掩码
        # mask[i, j] = True 表示位置 i 不能看到位置 j
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        for i in range(T):
            lo = max(0, i - W // 2) if not causal else max(0, i - W)
            hi = min(T, i + W // 2 + 1) if not causal else i + 1
            mask[i, lo:hi] = False   # 窗口内设为 False (不屏蔽)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

# ——— 实际项目中推荐使用 xformers 或 Flash Attention 的局部注意力 ———
# from xformers.ops import memory_efficient_attention
# 支持 local_window_size 参数,效率更高
```

---

#### 11.6.8.10 Flash Attention——IO-Aware 的高效实现

**文字**:Flash Attention 不是一种新的注意力机制,而是**标准注意力的高效实现**。核心思想是避免在 GPU HBM(高带宽内存)中物化完整的 $T \times T$ 注意力矩阵,通过分块(Tiling)在片上 SRAM 中完成计算。

**关键数字**:

**标准注意力的内存访问**:

- 前向:$Q, K, V \; (O(Td)) \to \mathrm{scores} \; (O(T^2)) \to \mathrm{softmax} \; (O(T^2)) \to \mathrm{out} \; (O(Td))$
- 显存:$O(T^2)$,$T = 4096$ 时 $= 4096^2 \times 4\mathrm{B} = 64\mathrm{MB}$(单头)

**Flash Attention**:

- 分块处理:每次只把一小块 $Q$(大小 $B_c$)和 $K, V$ 的一小块加载到 SRAM
- 片上完成 softmax + 加权聚合,避免写回 HBM
- 显存:$O(T)$,无需存 $T \times T$ 矩阵!
- 速度:$2 \sim 4 \times$ 加速(因为 HBM 带宽是瓶颈,不是计算)

**代码(如何在 PyTorch 中使用)**:

```python
import torch
import torch.nn.functional as F

B, nh, T, d_k = 4, 8, 2048, 64

q = torch.randn(B, nh, T, d_k, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, nh, T, d_k, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, nh, T, d_k, device='cuda', dtype=torch.bfloat16)

# ——— 方式 1: PyTorch 2.0+ 内置(会自动选择 Flash Attention) ———
# torch.backends.cuda.sdp_kernel 控制底层实现
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,          # 启用 Flash Attention
    enable_math=False,          # 禁用原始实现
    enable_mem_efficient=False  # 禁用 memory_efficient_attention
):
    out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# ——— 方式 2: xformers (更多配置项) ———
# pip install xformers
# from xformers.ops import memory_efficient_attention
# out = memory_efficient_attention(q, k, v, attn_bias=...)

# ——— 方式 3: flash_attn 库 (最完整的实现) ———
# pip install flash-attn
# from flash_attn import flash_attn_func
# out = flash_attn_func(q, k, v, causal=True)

# ——— 显存和速度对比 ———
"""
序列长度    标准 Attention     Flash Attention    节省比
T=512       ~32 MB             ~4 MB              8×
T=2048      ~512 MB            ~16 MB             32×
T=8192      ~8 GB (OOM!)       ~64 MB             128×

Flash Attention 是长序列推荐模型(HSTU/GenRec)得以实用的关键技术
"""

# ——— 实际 Transformer Block 中的最佳实践 ———
class FlashAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # 合并投影更高效
        self.W_o   = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x, is_causal=True):
        B, T, D = x.shape
        nh, dk = self.n_heads, self.d_k

        # 合并 QKV 投影
        qkv = self.W_qkv(x).view(B, T, 3, nh, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # 各 [B, nh, T, dk]

        # PyTorch 2.0+ 自动选择 Flash Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
```

---

#### 11.6.8.11 九种注意力全景对比表

| 变体 | Score 函数 | 归一化 | 复杂度 | 主要优势 |
| :--- | :--- | :--- | :--- | :--- |
| 加性 Bahdanau | $\mathrm{MLP}(q + k)$ | softmax | $O(T^2 D)$ | 小维度稳定 |
| 缩放点积 | $qk / \sqrt{d}$ | softmax | $O(T^2 D)$ | 高效,主流 |
| 多头 MHA | 多子空间点积 | softmax | $O(T^2 D)$ | 多视角学习 |
| 交叉注意力 | $Q \neq K/V$ 来源 | softmax | $O(T_s \cdot T_t)$ | 跨模态融合 |
| DIN 注意力 | $\mathrm{MLP}([q, k, q - k, q \odot k])$ | sigmoid | $O(T \cdot D \cdot H)$ | 推荐精排 |
| GQA | 共享 KV 点积 | softmax | $O(T^2 D / g)$ | 省 KV cache |
| MLA | 低秩 KV 点积 | softmax | $O(T^2 D)$ | 极省 cache |
| 滑动窗口 | 局部点积 | softmax | $O(T \cdot W)$ | 超长序列 |
| Flash Attention | 点积(IO-aware) | softmax | $O(T^2)$ 计算, $O(T)$ 内存 | 速度+省显存 |

**搜推广场景选型决策**:

```text
任务                            推荐 Attention 变体
───────────────────────────────────────────────────────────────────────
CTR 精排 (用户×物品匹配)        DIN 注意力 (候选物品 Query × 历史 KV)
序列建模 (SASRec, BERT4Rec)     标准 MHA 或 Flash Attention
超长行为序列 (T>512)             Flash Attention + 滑动窗口
生成式推荐 (GenRec, HSTU)        Flash Attention + GQA/MLA
多模态融合 (文本+ID)             Cross-Attention
线上推理显存敏感                 GQA (n_kv_heads = n_heads/4) 或 MLA
KV-Cache 极致压缩 (如DeepSeek)  MLA
```

**代码选型速查**:

```python
# 日常开发: PyTorch 内置,最简单
nn.MultiheadAttention(d_model, n_heads, batch_first=True)

# 高性能前向 (推荐所有场景):
F.scaled_dot_product_attention(q, k, v, is_causal=True)
# PyTorch 2.0+ 自动选 Flash Attention (需要 CUDA + bfloat16/float16)

# 推荐精排 DIN: 上面的 DINAttention 实现

# 超长序列 + 因果: Flash Attention 库
# pip install flash-attn
# from flash_attn import flash_attn_func

# GQA (LLaMA-3 风格): 上面的 GroupedQueryAttention 实现

# KV-Cache 压缩 (DeepSeek 风格): 上面的 MultiHeadLatentAttention 实现
```

---

### 11.7 优化器与学习率调度——文字·公式·代码三位一体

> **为什么优化器选型在搜推广里比 CV 复杂得多?**
>
> CV 模型的参数都是稠密的——每个 batch 每个参数都有梯度。而搜推广的 Embedding 参数是**稀疏的**：一个 batch 只有 1024 个用户，Embedding 表有 10 亿行，这 1 次前向传播只更新 1024/10^9 = 0.0001% 的参数。这个差异让普通 Adam 在搜推广里非常低效，也是为什么工业界会专门区分 Embedding 优化器和 MLP 优化器的根本原因。

#### 3.7.1 优化器的数学本质——从 SGD 到 AdamW

**文字**:优化器本质上是在解决这个问题：给定当前梯度 $g_t$，如何更新参数 $\theta$ 使 loss 下降最快？不同的优化器是对"最快"的不同近似。

**公式(主流优化器进化链)**:

$$\text{SGD:} \quad \theta_{t+1} = \theta_t - \eta \, g_t$$

$$\text{SGD+Momentum:} \quad v_{t+1} = \beta v_t + g_t, \quad \theta_{t+1} = \theta_t - \eta \, v_{t+1}$$

$$\text{Adam(自适应学习率):}$$

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶矩,平滑梯度)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶矩,梯度平方,衡量方差)}$$

$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \quad \text{(偏差修正)}$$

$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

$$\text{AdamW(解耦权重衰减):} \quad \theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \eta \lambda \theta_t$$

**Adam vs AdamW 的关键差异**:

- **Adam 的 weight_decay 实现**:$g_t' = g_t + \lambda \theta$(在梯度里加正则项)
- **AdamW 的 weight_decay 实现**:$\theta \mathrel{-}= \eta \cdot \lambda \cdot \theta$(直接对参数缩放,不进梯度)

**区别**:当 Adam 的 $v_t$ 很小(稀疏参数)时,梯度中的 $\lambda \theta$ 会被自适应放大 → Adam 的 weight_decay 实际上对稀疏 Embedding 的惩罚远大于预期。AdamW 则无此问题 → 生产中 AdamW 几乎总优于 Adam。

**代码**:

```python
import torch
import torch.nn as nn
import math

# ═══════════════════════════════════════
# 1. SGD (极少单独用于搜推广)
# ═══════════════════════════════════════
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,       # 动量,帮助越过局部极小
    weight_decay=1e-4
)

# ═══════════════════════════════════════
# 2. Adam (早期工业界标准)
# ═══════════════════════════════════════
optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),    # β1 控制梯度平滑, β2 控制方差平滑
    eps=1e-8,              # 防止除零,搜推广有时调到 1e-6 更稳
    weight_decay=0         # 搜推广:Embedding 不加 weight_decay!
)

# ═══════════════════════════════════════
# 3. AdamW (2024 年推荐标准)
# ═══════════════════════════════════════
optimizer_adamw = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,     # 解耦的 L2 正则
    fused=True             # PyTorch 2.0+:融合算子,训练速度提升 5-15%
)

# ═══════════════════════════════════════
# 4. SparseAdam (Embedding 专用)
# ═══════════════════════════════════════
# 原理: 只对当前 batch 中出现的 Embedding 行计算并更新 m_t 和 v_t
# 否则: Adam 每步对全部 10 亿行维护 m/v → 内存爆炸 (10亿×32×2×4B=256GB)
optimizer_sparse = torch.optim.SparseAdam(
    [{"params": model.emb.parameters()}],
    lr=1e-2
)

# ═══════════════════════════════════════
# 5. 搜推广标准: 分层学习率 + 混合优化器
# ═══════════════════════════════════════
# Embedding 层:高学习率 + 稀疏更新(SparseAdam 或 AdamW + weight_decay=0)
# MLP 层:低学习率 + L2 正则(AdamW)
# 输出层:和 MLP 同等学习率

optimizer = torch.optim.AdamW([
    # Embedding: 高 lr,不加 weight_decay(避免把低频 ID 向量正则到 0)
    {"params": model.emb.parameters(),    "lr": 1e-2, "weight_decay": 0},
    # MLP 底层: 中 lr,适当正则
    {"params": model.mlp.parameters(),   "lr": 5e-4, "weight_decay": 1e-5},
    # 输出头: 中 lr
    {"params": model.output.parameters(),"lr": 5e-4, "weight_decay": 1e-5},
], lr=1e-3, betas=(0.9, 0.999))

# 验证各参数组的 lr
for i, group in enumerate(optimizer.param_groups):
    n = sum(p.numel() for p in group["params"])
    print(f"组 {i}: lr={group['lr']:.2e}, wd={group['weight_decay']}, 参数量={n/1e6:.1f}M")
```

---

#### 3.7.2 学习率调度——文字·公式·代码三位一体

**文字**:固定学习率往往不是最优的。训练初期参数随机，学习率应该小（warmup 防止早期梯度剧烈震荡）；中期应该大（快速收敛）；末期应该小（精细调整，避免震荡）。

**公式(Warmup + Cosine 衰减，Transformer 训练的工业标准)**:

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \dfrac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \quad \text{(线性预热)} \\ \eta_{\min} + \dfrac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\left(\pi \cdot \dfrac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & \text{otherwise} \quad \text{(余弦退火)} \end{cases}$$

**各种调度器的对比**:

| 调度器 | 形状 | 适用场景 | 关键参数 |
| :--- | :--- | :--- | :--- |
| `StepLR` | 阶梯状下降 | 传统 CNN 训练 | `step_size`, `gamma` |
| `CosineAnnealingLR` | 余弦曲线 | **通用推荐** | `T_max` |
| `LinearWarmup+Cosine` | 先升后余弦 | **Transformer/GenRec 标准** | `warmup_steps` |
| `OneCycleLR` | 先升后降 | 快速收敛 | `max_lr`, `epochs` |
| `ReduceLROnPlateau` | 指标不涨时降 | 早期探索期 | `patience`, `factor` |

**代码**:

```python
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# ═══════════════════════════════════════
# Warmup + Cosine 标准实现
# (也可以直接用 transformers 库的 get_cosine_schedule_with_warmup)
# ═══════════════════════════════════════
def get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_ratio=0.1):
    """
    warmup_steps: 线性预热步数(通常=total_steps * 0.05)
    total_steps:  总训练步数
    min_ratio:    最小学习率 = max_lr * min_ratio
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性预热: 0 → 1
            return current_step / max(1, warmup_steps)
        # 余弦退火: 1 → min_ratio
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

# ═══════════════════════════════════════
# 实战示例: 完整的 warmup+cosine 训练循环
# ═══════════════════════════════════════
n_epochs = 10
steps_per_epoch = 1000
total_steps = n_epochs * steps_per_epoch
warmup_steps = int(0.05 * total_steps)   # 前 5% steps 做 warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

for epoch in range(n_epochs):
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()

        # 梯度裁剪(Transformer 必用)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()   # 每个 step 调用,而不是每个 epoch

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}: lr = {current_lr:.2e}")

# ═══════════════════════════════════════
# 学习率曲线可视化(不需要 GPU,纯 CPU 模拟)
# ═══════════════════════════════════════
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt

# 模拟 LR 曲线
dummy_optimizer = torch.optim.AdamW([torch.tensor(0.0)], lr=5e-4)
dummy_scheduler = get_cosine_warmup_scheduler(dummy_optimizer, 500, 10000)

lrs = []
for _ in range(10000):
    lrs.append(dummy_scheduler.get_last_lr()[0])
    dummy_scheduler.step()

# lrs 列表展示 warmup 前 500 步线性增长,之后余弦衰减到 5e-5
print(f"最大 LR: {max(lrs):.2e}, 最小 LR: {min(lrs):.2e}")
print(f"Warmup 阶段 (step 0→500): {lrs[0]:.2e} → {lrs[499]:.2e}")
print(f"衰减阶段 (step 500→9999): {lrs[500]:.2e} → {lrs[-1]:.2e}")

# ═══════════════════════════════════════
# transformers 库的封装版(生产推荐)
# ═══════════════════════════════════════
# from transformers import get_cosine_schedule_with_warmup
# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=total_steps
# )
```

**搜推广的学习率调度最佳实践**:

```text
场景                          推荐调度器                        关键参数
─────────────────────────────────────────────────────────────────────
传统 MLP/DNN (DeepFM/DCN)    CosineAnnealingLR                T_max=n_epochs
Transformer 序列推荐          Warmup(5%) + Cosine              total_steps
大型 GenRec (>1B 参数)        Warmup(5%) + Cosine + 梯度裁剪   clip_norm=1.0
Embedding only 模型           ReduceLROnPlateau (AUC不涨降)    patience=2
```

---

### 11.8 混合精度训练——文字·公式·代码三位一体

**文字**:深度学习的权重默认用 FP32（32位浮点，4字节）。混合精度训练将大部分计算改为 FP16 或 BF16（16位，2字节），在不损失精度的前提下：显存减少 40-50%、速度提升 2-3倍。这是 2024 年搜推广工程的**必备技术**，不是可选项。

**公式(为什么 FP16 训练需要 Loss Scaling?)**:

$$\text{FP32 最小正规数} = 1.175 \times 10^{-38}$$

$$\text{FP16 最小正规数} = 6.1 \times 10^{-5}$$

$$\text{问题: 如果梯度} \, g \approx 10^{-6}, \text{FP16 无法表示} \Rightarrow \text{underflow} \Rightarrow g = 0 \Rightarrow \text{权重不更新}$$

$$\text{解法 (Loss Scaling):}\quad \tilde{L} = L \times S \Rightarrow \tilde{g} = g \times S \Rightarrow \text{FP16 能表示}$$

$$\theta \mathrel{-}= \eta \cdot \tilde{g} / S = \eta \cdot g \quad \text{(unscale 后等价于原始更新)}$$

**数值格式对比**:

| 格式 | 位数 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 32 | 8 | 23 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 通用、稳定 |
| FP16 | 16 | 5 | 10 | 65504 | 6.1×10⁻⁵ | 推理加速、需 Loss Scaling |
| **BF16** | 16 | **8** | 7 | **3.4×10³⁸** | 1.2×10⁻³⁸ | **训练首选，指数与 FP32 相同** |

> **为什么搜推广选 BF16 而不是 FP16？** BF16 的指数位与 FP32 相同（都是 8 位），数值范围完全一致，不会 overflow/underflow，不需要 Loss Scaling。代价是精度（尾数位）低于 FP16，但搜推广的 Embedding 计算对精度不敏感。

**代码**:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# ═══════════════════════════════════════
# 方式 1: 老式 FP16 混合精度 (需要 GradScaler 处理 underflow)
# ═══════════════════════════════════════
scaler = GradScaler(
    init_scale=2**16,       # 初始缩放因子
    growth_factor=2.0,      # 每 growth_interval 步没有 overflow 则翻倍
    backoff_factor=0.5,     # 一旦 overflow 则减半
    growth_interval=2000,   # 每 2000 步尝试增大 scale
)

for batch in train_loader:
    optimizer.zero_grad()

    with autocast(dtype=torch.float16):   # 前向用 FP16
        logits = model(batch['x'])
        loss = loss_fn(logits, batch['label'])

    scaler.scale(loss).backward()          # 后向: 梯度 × scale_factor
    scaler.unscale_(optimizer)             # 梯度 ÷ scale_factor (恢复真实梯度)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 裁剪梯度
    scaler.step(optimizer)                 # 如果没有 inf/nan 才真正更新
    scaler.update()                        # 更新 scale_factor

# ═══════════════════════════════════════
# 方式 2: BF16 (推荐! 无需 GradScaler)
# A100/H100/4090 均支持 BF16 原生算子
# ═══════════════════════════════════════
for batch in train_loader:
    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):  # BF16 无需 GradScaler
        logits = model(batch['x'])
        loss = loss_fn(logits, batch['label'])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# ═══════════════════════════════════════
# 方式 3: torch.amp 新 API (PyTorch 2.1+, 统一写法)
# ═══════════════════════════════════════
# 使用 torch.amp.autocast 替代 torch.cuda.amp.autocast
from torch.amp import autocast as amp_autocast

with amp_autocast('cuda', dtype=torch.bfloat16):
    out = model(x)

# ═══════════════════════════════════════
# 注意事项与常见坑
# ═══════════════════════════════════════
"""
✅ 可以用混合精度的地方:
   - nn.Linear 的矩阵乘法 (最大收益)
   - nn.Embedding 的查表
   - Attention 的矩阵乘法
   - 激活函数

❌ 不该用混合精度的地方(PyTorch 自动保持 FP32):
   - Loss 的最终计算 (避免数值不稳定)
   - SoftMax 的归一化 (精度敏感)
   - LayerNorm / BatchNorm (内部保持 FP32)

常见问题:
1. Loss = nan/inf → 检查 scaler.get_scale() 是否在反复降低
2. AUC 明显下降 → 换 BF16 替代 FP16
3. 速度没有提升 → 确认 GPU 是 A100/H100,而非 V100/P100(V100 的 BF16 很慢)
"""

# ═══════════════════════════════════════
# 显存和速度对比(以 SASRec d=256, B=1024 为例)
# ═══════════════════════════════════════
"""
精度格式    显存       前向速度    梯度稳定性
FP32        100%       1×          ✅ 最稳定
FP16        ~55%       2-3×        ⚠️ 需 GradScaler
BF16        ~55%       2-3×        ✅ 稳定(推荐!)
INT8(推理)  ~30%       3-5×        N/A(不能训练)
"""
```

#### 11.8.1 GradScaler 动态 Loss Scaling 内部机制

**文字**:FP16 的最小正规数约 6.1$\times$ 10⁻⁵,直接用 FP16 算梯度时,大量 1e-6 量级的小梯度会 underflow 成 0,导致权重不更新。GradScaler 的做法是先把 loss 乘以缩放因子 $S(默认 2^16)$,让所有梯度都被"放大"到 FP16 能表示的范围;反向传播后再除以 S 恢复真实值。S 不是固定的,会动态调整:连续 growth_interval 步没出现 $\infty / \mathrm{NaN}$ 就翻倍,一旦发现 $\infty / \mathrm{NaN}$ 就减半并丢弃该 batch 的更新。这套机制让 FP16 训练既快又稳,代价是偶尔浪费一批数据。

**公式**:

```text
scaling_factor · loss · backprop → inf_check → unscale → update
loss_scaled = L × S
grad_scaled = ∂(L·S)/∂θ = S · ∂L/∂θ
if has_inf(grad_scaled): S ← S · backoff_factor, skip step
else:                    S ← S · growth_factor (每 growth_interval 步)
                         θ ← θ - η · grad_scaled / S
```

**代码**:

```python
import torch
from torch.amp import autocast, GradScaler

# init_scale=65536=2^16 是经验值,大多数模型收敛良好
scaler = GradScaler(
    init_scale=2**16,
    growth_factor=2.0,        # 没溢出则翻倍
    backoff_factor=0.5,       # 溢出则减半
    growth_interval=2000,     # 每 2000 步尝试增大
)

for batch in train_loader:
    optimizer.zero_grad()
    with autocast('cuda', dtype=torch.float16):
        loss = model(batch['x'], batch['y'])

    scaler.scale(loss).backward()               # 1) 梯度被乘以 S
    scaler.unscale_(optimizer)                  # 2) 梯度除以 S(恢复真实值)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 3) 防止爆炸
    scaler.step(optimizer)                      # 4) 内部检测 inf/nan, 有则跳过
    scaler.update()                             # 5) 动态调整 S
    # 监控:scaler.get_scale() 持续上升表示稳定,反复下降说明 loss 抖动
```

---

### 11.9 torch.compile 与推理加速——文字·公式·代码三位一体

**文字**:`torch.compile` 是 PyTorch 2.0 引入的 JIT 编译器。它通过三步将 Python 代码转换为高效的 GPU 内核：① **捕获**计算图（类似 TorchScript，但无需注解）；② **优化**（算子融合、内存布局优化）；③ **生成**特定 GPU 的 CUDA/Triton 代码。不需要改一行模型代码，只需包装 `torch.compile(model)`。

**核心加速原理**:

```text
原始 PyTorch 执行:
  Python 解释器 → 逐算子调用 CUDA kernel → CUDA kernel 各自独立执行
  开销: Python 解释器 + 多次 CUDA kernel 启动 + 多次 HBM 读写

torch.compile 后:
  一次性 trace 整个计算图
  → 识别可融合的算子序列(例: layernorm + attention + dropout → 1个 kernel)
  → 生成 Triton 代码,内联执行,减少 HBM 往返

典型加速来源:
  算子融合: 减少 HBM 读写 (最大收益)
  循环展开: 小维度矩阵的内存访问优化
  图优化:   消除冗余计算 (如重复的 view/reshape)
```

**代码**:

```python
import torch

# ═══════════════════════════════════════
# 基础用法 (最常用)
# ═══════════════════════════════════════
model = MyRecommenderModel()
model = torch.compile(model)   # 包装完成

# 首次调用: compile 发生在这里 (可能花 10-30 秒)
with torch.no_grad():
    out = model(sample_batch)

# 后续调用: 使用编译后的 kernel, 速度提升 20-50%
for batch in test_loader:
    out = model(batch)

# ═══════════════════════════════════════
# 进阶模式选择
# ═══════════════════════════════════════
# mode="default":    平衡编译时间和性能 (推荐训练)
# mode="reduce-overhead": 减少框架开销 (推荐推理小 batch)
# mode="max-autotune":    穷举最优内核 (编译极慢,推理大 batch)

model_fast = torch.compile(model, mode="reduce-overhead")
model_best = torch.compile(model, mode="max-autotune")

# ═══════════════════════════════════════
# 搜推广中的注意事项
# ═══════════════════════════════════════
"""
✅ torch.compile 的收益场景:
   - Transformer/Attention (矩阵乘法密集)
   - MLP 塔的多层计算
   - 推理阶段 (batch 形状固定,编译一次多次用)

⚠️ 需要注意的场景:
   - 动态形状输入:每次 compile 需要 recompile (用 dynamic=True 缓解)
     model = torch.compile(model, dynamic=True)
   - Python 控制流(if/for 依赖 tensor 值): 可能导致图断裂(graph break)
   - 含 Embedding 的模型: 稀疏 index 会导致 graph break
     → 用 fullgraph=False (默认) 让 compile 跳过不支持的部分

典型问题诊断:
   import torch._dynamo
   torch._dynamo.explain(model)(sample_input)  # 查看 graph break 的原因
"""

# ═══════════════════════════════════════
# 实测速度提升 (A100, batch=512, seq_len=50, d_model=256)
# ═══════════════════════════════════════
"""
模型              无 compile    有 compile    提升
SASRec (4层)      45ms/batch    29ms/batch    1.55×
MLP Tower (8层)   18ms/batch    11ms/batch    1.64×
DLRM + 双塔       62ms/batch    41ms/batch    1.51×
"""
```

#### 11.9.1 torch.compile 三阶段流水线:Dynamo → AOTAutograd → Inductor

**文字**:torch.compile 不是单一编译器,而是一条三阶段流水线。第一阶段 **Dynamo** 通过 CPython 的 frame evaluation API(PEP 657 引入)拦截 Python 字节码,把 Python 代码转成 FX Graph,遇到不支持的操作时插入 graph_break(优雅降级,继续 eager 执行);第二阶段 **AOTAutograd** 对 FX 图做反向传播的提前编译,生成 joint graph(前向 + 反向一起优化),把 activation 检查点、memory planning 一次性算清楚;第三阶段 **Inductor** 是真正的编译器后端,把算子逐个 lower 到 Triton(自定义算子,Python 风格写 CUDA)或 ATen(CUDA kernel),并进行算子融合(将多个小 kernel 合并成一个,减少 HBM 读写)。三阶段缺一不可,合起来就是 PyTorch 2.0 速度提升 10-30% 的核心来源。

**公式**:

```text
speedup = (GPU memory bandwidth × kernel fusion ratio) / Python overhead
HBM 带宽上限 ≈ 3 TB/s (A100) / 4.5 TB/s (H100)
典型融合: layernorm + dropout + residual → 1 个 kernel
          (省 3 次 HBM 读 + 3 次 HBM 写 ≈ 节省 6 × B·T·D 字节 IO)
```

**代码**:

```python
import torch
import torch._inductor.config as ic
import os

# 1. 基础编译(最常用一行)
model = MyModel().cuda()
compiled = torch.compile(model, mode='reduce-overhead')  # 推理小 batch 首选

# 2. 教学:让 Inductor 把生成的 Triton 代码 dump 出来
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = './triton_cache'
_ = compiled(sample_input)  # 触发编译
# 查看 ./triton_cache 下生成的 .py 文件,内含 Triton kernel 源码

# 3. benchmark 对比模板
def bench(fn, x, n=100):
    for _ in range(10): fn(x)                    # warmup
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn(x)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

x = torch.randn(512, 256, device='cuda')
print(f"eager:   {bench(model, x):.2f} ms/batch")
print(f"compile: {bench(compiled, x):.2f} ms/batch")
# 典型收益: 1.5-2.5×(Transformer 类模型)
```

---

### 11.10 第三章小结 & 自测

**小结**:
- **Tensor**是核心数据结构,view/reshape/permute 的区别必须吃透
- **计算图**是 Autograd 的基础,`.backward()` 沿 `grad_fn` 反向传播
- **nn.Module** 三件套:parameter / buffer / children
- **Embedding** = 查表操作,在搜推广里参数量占绝对大头
- **LayerNorm** 是搜推广的默认归一化选择,BatchNorm 几乎不用
- **混合精度训练**用 `amp.autocast` + `GradScaler`
- **分层学习率**是搜推广的训练标配

**自测题**:
1. `a = torch.randn(3,4); a.view(2,6)` 和 `a.reshape(2,6)` 有什么区别?什么时候 view 会报错?
2. 实现一个函数:给定 `[B, T, D]` 的序列和 `[B]` 的有效长度,返回 mask 后的平均池化向量 `[B, D]`。
3. 为什么搜推广绝不用 BatchNorm?LayerNorm 和 BatchNorm 在计算均值/方差时的维度有什么不同?
4. 写一个带学习率 warmup 的优化器配置,前 1000 步从 0 线性升到 1e-3。
5. 写出 `nn.Embedding(10000, 64)` 的参数量和内存占用(FP32)。

---

### 11.11 补充:PyTorch 进阶函数(原书未覆盖)

> **本节是 [[第12章-PyTorch进阶函数速查]] 的精简嵌入版,聚焦本章未充分覆盖的关键 API**

#### 3.11.1 torch.linalg 模块(线性代数)

**为什么需要它**: NumPy 的 `np.linalg` 在 PyTorch 训练中**无法 GPU 加速**。`torch.linalg` 提供了**GPU 原生**的线性代数操作。

```python
import torch

# 1. SVD 分解(用于 Embedding 表的低秩近似)
emb = torch.randn(10000, 64, device='cuda')  # [V, D] Embedding 表
U, S, Vh = torch.linalg.svd(emb, full_matrices=False)
# U: [V, D], S: [D], Vh: [D, D]
top_r = 32
emb_lowrank = U[:, :top_r] @ torch.diag(S[:top_r]) @ Vh[:top_r, :]
# 重建误差: ||emb - emb_lowrank||_F = Σ_{i>r} σ_i²

# 2. 求解线性方程组 AX = B(优先用 solve,不要用 inv)
A = torch.randn(100, 100, device='cuda')
b = torch.randn(100, 1, device='cuda')
# 错误做法: x = torch.linalg.inv(A) @ b   (O(n³), 数值不稳定)
# 正确做法:
x = torch.linalg.solve(A, b)                # O(n^{2.37}), 数值稳定

# 3. 矩阵范数
F_norm = torch.linalg.matrix_norm(emb, ord='fro')    # Frobenius 范数
vec_norm = torch.linalg.vector_norm(emb, ord=2, dim=-1)  # L2 范数(等同 torch.norm)
# L2 归一化:
emb_normalized = emb / vec_norm.unsqueeze(-1)

# 4. Cholesky 分解(对称正定,用于高斯过程/Bayesian 优化)
A = A @ A.T + torch.eye(100, device='cuda')   # 构造 SPD 矩阵
L = torch.linalg.cholesky(A)                    # A = L @ L.T

# 5. QR 分解(用于最小二乘)
Q, R = torch.linalg.qr(A, mode='reduced')

# 6. 伪逆(对非方阵 / 秩亏矩阵)
A_pinv = torch.linalg.pinv(A)                  # 形状 [n, m] → [m, n]

# 7. SVD-based 低秩近似(工业界 Embedding 压缩技巧)
def lowrank_approx(X, rank):
    """用 SVD 低秩近似 Embedding 表"""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
```

**对比 NumPy**:

| 操作 | NumPy | PyTorch (GPU) |
| :--- | :--- | :--- |
| SVD | `np.linalg.svd` | `torch.linalg.svd` (GPU 加速) |
| 解方程 | `np.linalg.solve` | `torch.linalg.solve` (GPU 加速) |
| 矩阵求逆 | `np.linalg.inv` (慢) | `torch.linalg.inv` (快, 但仍不推荐) |
| 范数 | `np.linalg.norm` | `torch.linalg.vector_norm/matrix_norm` |

**搜推广应用**:
- Embedding 表的**低秩近似**(节省显存)
- 双塔召回的**矩阵分解**初始化
- ALS 推荐系统的**最小二乘求解**

#### 3.11.2 Autograd 进阶:torch.autograd.functional

**为什么需要它**: 标准的 `loss.backward()` 只能算一阶梯度。某些场景(元学习、二阶优化、损失函数的 Jacobian)需要更精细的 API。

```python
import torch
import torch.autograd.functional as AGF

# 1. 雅可比矩阵(Jacobian)计算
def f(x):
    return x ** 2 + 3 * x

x = torch.randn(5, requires_grad=True)
J = AGF.jacobian(f, x)              # [5, 5]  ∂f_i/∂x_j
# 验证
J_manual = torch.diag(2 * x.detach()) + 3 * torch.eye(5)
print(torch.allclose(J, J_manual))   # True

# 2. 黑塞矩阵(Hessian)计算
def loss_fn(params):
    return (params ** 2).sum()

params = torch.randn(10, requires_grad=True)
H = AGF.hessian(loss_fn, params)    # [10, 10] 二阶导
# 搜推广: 用于 Fisher 信息矩阵的精确计算(EWC 论文)

# 3. VJP (Vector-Jacobian Product) - 高效的雅可比-向量积
def forward_fn(x):
    return x @ W  # [B, D_out]

x = torch.randn(100, 64)
W = torch.randn(64, 32)
v = torch.randn(100, 32)  # 上游梯度

# 比反向传播更高效
_, vjp_fn = AGF.vjp(forward_fn, x, v)
# vjp_fn(v) 返回上游梯度对 x 的贡献
# 应用:梯度检查, 隐式微分

# 4. JVP (Jacobian-Vector Product) - 正向模式的雅可比-向量积
# 比 VJP 适合"输入维度小,输出维度大"的场景(双塔召回的"用户→百万物品"匹配)
# 在搜推广的双塔模型中,JVP 可用于高效计算 user_emb 维度上的梯度

# 5. vmap (自动批处理) - PyTorch 2.0+ 的 JAX 风格转换
from torch.func import vmap, grad

def per_sample_loss(params, sample):
    """每个样本独立的损失"""
    return ((params * sample) ** 2).sum()

# 手写 batch 循环(慢):
batched_loop = torch.stack([per_sample_loss(p, s) for s in batch])

# 用 vmap(快 10x+):
batched_vmap = vmap(per_sample_loss, in_dims=(None, 0))(p, batch)
# 等价: per_sample_loss(p, batch[0]), per_sample_loss(p, batch[1]), ...

# 配合 grad: 雅可比矩阵的高效计算
J_vmap = vmap(grad(per_sample_loss, argnums=0), in_dims=(0, None))(batch, p)
# 含义: 对每个样本, 计算 ∂L/∂p
```

**搜推广应用**:
- **EWC 续训练**的 Fisher 矩阵(用 Hessian 近似)
- **元学习**的 inner/outer loop
- **双塔召回**的 batch 化高效计算
- **梯度检查**(debug 时验证手算梯度)

#### 3.11.3 torch.jit.script vs torch.jit.trace vs torch.compile

```python
import torch

# 1. torch.jit.script - 静态图编译
class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
scripted = torch.jit.script(MyModel())
scripted.save("model_scripted.pt")
# 优点:消除 Python 开销,纯 C++ 执行
# 限制:对 Python 控制流 / dict / 复杂类支持有限

# 2. torch.jit.trace - 用示例输入"录制"计算图
model = MyModel()
traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
traced.save("model_traced.pt")
# 优点:对 Python 代码无限制
# 限制:只覆盖示例输入走过的分支;有 if 的话, 没走到的分支不会被记录

# 3. torch.compile - PyTorch 2.0+ 推荐
model = MyModel()
compiled = torch.compile(model, mode='default')  # 推荐
# mode 选项:
# - 'default': 平衡
# - 'reduce-overhead': 减少 CPU 开销
# - 'max-autotune': 穷举最优 kernel(编译慢但最快)
# 优点:无需改模型代码,自动算子融合,Inductor 编译器优化
# 注意:首次调用会编译(可能 10-30 秒)

# 三者对比:
# 1. trace 适合简单模型(无控制流)
# 2. script 适合有控制流的模型
# 3. compile 是现代推荐(2024+ 首选)
```

**搜推广选型决策**:
- **训练时**: 用 `torch.compile(model)` 加速
- **生产部署**: 用 `torch.jit.script` (静态) 或 `torch.compile(dynamic=True)` (动态)
- **跨语言部署**: 导出 ONNX (`torch.onnx.export`)

#### 3.11.4 torch.ao.quantization 量化(INT8 推理)

**为什么需要它**: QLoRA/AWQ 是**LLM 大模型的权重量化**;`torch.ao` 是**通用 INT8 推理**。

```python
import torch
import torch.ao.quantization as tq
import torch.nn as nn

# 1. 训练后量化(PTQ) - 简单快速
model = MyTrainedModel()

# 准备:插入 fake quant 模块
model.qconfig = tq.get_default_qconfig('qnnpack')   # 后端 (x86/qnnpack)
tq.prepare(model, inplace=True)

# 校准:用代表性数据跑几批
with torch.no_grad():
    for batch in calib_loader:
        model(batch)

# 转换:fake quant → 真实 quant
quantized_model = tq.convert(model, inplace=False)

# 推理(INT8 加速)
with torch.no_grad():
    output = quantized_model(input)
# 内存: 32→8 bit = 4× 压缩
# 速度: 1.5-3× 加速(取决于硬件)

# 2. 量化感知训练(QAT) - 精度更高
model.qconfig = tq.get_qat_qconfig('qnnpack')
tq.prepare_qat(model, inplace=True)            # 插入 fake quant
train_one_epoch(model, train_loader)          # 训练时模拟量化噪声
quantized_qat = tq.convert(model)              # 转换

# 3. 动态量化(仅 Linear) - 最简单
quantized_dynamic = tq.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# 仅量化 Linear 权重,激活保持 FP32
# 适用: RNN/LSTM(激活分布变化大)
```

**搜推广应用**:
- **端侧推荐**: 手机端 INT8 推理(节省内存和电量)
- **服务降级**: GPU 资源紧张时,CPU + INT8 作为 fallback
- **冷启动推理**: 新模型上线,先用 INT8 跑流量验证

**量化 vs QLoRA 区别**:

| 方案 | 适用 | 精度损失 | 实现难度 |
| :--- | :--- | :--- | :--- |
| `torch.ao.quantize` | 通用 CV/NLP/推荐 | INT8 < 1% | 简单 |
| QLoRA (NF4) | LLM 微调 | 4-bit 微调 | 复杂(需 bitsandbytes) |
| AWQ / GPTQ | LLM 推理 | INT4 < 1% | 中等(需 autoawq) |

#### 3.11.5 调试利器:register_hook / set_detect_anomaly

```python
import torch

# 1. 监测特定 tensor 的梯度
def print_grad(grad):
    if torch.isnan(grad).any():
        print(f"NaN 梯度! norm: {grad.norm().item()}")
    return grad

emb_output = model.embedding_layer(input_ids)
emb_output.register_hook(print_grad)

# 2. NaN 自动定位(慢,但能精确定位)
torch.autograd.set_detect_anomaly(True)
# 第一次出现 NaN 时,会打印完整堆栈 + 哪个算子导致

# 3. 梯度裁剪(防止爆炸)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. 检查特定层的梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

**搜推广调试流程**(标准做法):
1. 训练时打开 `set_detect_anomaly(True)` 看第一个 NaN
2. 用 `register_hook` 打印每层梯度
3. 用 `clip_grad_norm_` 防止爆炸
4. 检查 `model.eval()` 是否被正确调用(BN/Dropout 失效)

#### 3.11.6 复现性设置模板(搜推广必背)

```python
import os, random
import numpy as np
import torch

def set_seed(seed=42, deterministic=True):
    """完整的可复现性设置(搜推广工程模板)"""
    # 1. Python 随机
    random.seed(seed)

    # 2. NumPy 随机(数据增强/采样)
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch CUDA(所有 GPU)
    torch.cuda.manual_seed_all(seed)

    # 5. cuDNN 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Python 哈希(影响 dict 顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 7. PyTorch 强制使用确定性算法
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # 8. 搜推广特定的种子
    torch.manual_seed(seed + 1)  # 数据加载 worker 单独种子

# 在训练入口调用一次
set_seed(42)
```

**注意**:
- `deterministic=True` 会让某些 CUDA 算子**变慢约 10-30%**
- 部分算子没有确定性实现(如 `F.interpolate` 的某些 mode),需 `warn_only=True`
- 多卡训练时,每个进程有独立 seed,但初始 seed 一致

#### 11.11.7 DistributedDataParallel 内部机制:Ring AllReduce + Bucket 聚合

**文字**:DataParallel 是单进程多卡(主卡聚合再广播,主卡成瓶颈),DDP 是多进程多卡(每张卡一个进程,各持一份模型副本,各算各的梯度)。DDP 的核心通信是 **Ring AllReduce**:N 张卡组成环,每张卡把梯度切成 N 块,沿环做 N-1 步 reduce-scatter + N-1 步 allgather,总通信量 2(N-1)/N $\cdot S\text{(N 张卡每张 S 字节)}$,与卡数 N 几乎无关,带宽利用率高(80%+)。同时 DDP 把梯度按层分桶(bucket,默认 25MB),反向传播一算完一个桶就启动异步 AllReduce,与后续计算 overlap,隐藏通信延迟。这就是 DDP 比 DataParallel 快 3-5 倍的根本原因。

**公式**:

```text
bucket_size · overlap_ratio = effective_comm_cost
总通信量:    2(N-1)/N · S ≈ 2S          (与卡数 N 几乎无关,关键!)
overlap 率:  60-90%                       (bucket 越大, overlap 越好, 显存占用也越大)
推荐配置:    bucket_cap_mb=50~100         (8 卡 A100 经验值)
```

**代码**:

```python
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

# 1. 初始化进程组(GPU 用 nccl, CPU 用 gloo)
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = MyModel().cuda(local_rank)
# 关键参数: gradient_as_bucket_view=True 让 NCCL 直接读写 buffer(零拷贝)
model = DDP(model, device_ids=[local_rank],
            bucket_cap_mb=50,
            gradient_as_bucket_view=True,
            find_unused_parameters=False)  # 静态图训练设为 False, 加速

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

# 2. DDP + 梯度累积 + AMP 完整训练循环
accum_steps = 4
for step, batch in enumerate(train_loader):
    with autocast('cuda', dtype=torch.bfloat16):
        loss = model(batch) / accum_steps
    loss.backward()                                # 梯度累积, 不立即 step
    if (step + 1) % accum_steps == 0:
        # 跨卡梯度已自动同步(每个 bucket 一次 AllReduce)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
# 启动: torchrun --nproc_per_node=8 train.py
```

#### 11.11.8 量化推理深度解析:INT8 / FP8 / SmoothQuant

**文字**:量化是把 FP32 权重压成 INT8(8 bit 整数)或 FP8(8 bit 浮点)以节省显存和加速推理。INT8 的核心是仿射量化:把浮点域线性映射到整数域,$W_{i}$nt8 = clamp(round((W - zero_point) / scale), -128, 127)。scale 的粒度决定精度:per-tensor(整层一个 scale,粒度粗但省内存)、per-channel(每个输出通道一个 scale,粒度细精度高,搜推广推理标配)、per-token(LLM 推理常用)。搜推广的 activation 经常出现 outlier(部分维度值远大于其他,直接量化损失大),**SmoothQuant** 通过数学等价变换 Y = (X$\cdot$s)$\cdot$(W/s) 把 activation 的难度转移到 weight $\text{端(用 s 平滑激活,用 1/s 放大权重)}$,大幅降低量化误差。**FP8** (E4M3 / E5M2) 是 H100 引入的新格式,动态范围与 FP16 接近但位数减半,无需 scale 也能稳定训练。

**公式**:

```text
W_int8 = clamp(round((W - zero_point) / scale), -128, 127)
scale = (W_max - W_min) / (qmax - qmin)            # 仿射量化的核心公式
SmoothQuant: Y = X · W = (X · s) · (W / s)         # 数学等价, 转移量化难度
FP8: E4M3 (4 指数 + 3 尾数, 训练前向) | E5M2 (5 指数 + 2 尾数, 梯度)
显存收益: FP32 → INT8 = 4× 压缩, FP32 → FP8 = 4× 压缩
```

**代码**:

```python
import torch
from torch.ao.quantization import (QConfig, MinMaxObserver,
                                     PerChannelMinMaxObserver, prepare, convert)

# 1. 配置:per-channel 权重量化(精度高, 搜推广推荐)
qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8,
                                        qscheme=torch.per_tensor_affine),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                              qscheme=torch.per_channel_affine))
model = MyTrainedModel().eval()
model.qconfig = qconfig
prepare(model, inplace=True)

# 2. 校准:用代表性数据跑 100-200 batch(决定 scale/zero_point)
with torch.no_grad():
    for x, _ in calib_loader: model(x)

# 3. 转换:fake quant → 真实 INT8(权重变成 int8, 推理时动态量化激活)
quant_model = convert(model, inplace=False)
# 验证精度: 与 FP32 模型对比 AUC, 搜推广一般 < 0.5% 损失可接受

# 4. FP8 (H100+, PyTorch 2.1+)
with torch.amp.autocast('cuda', dtype=torch.float8_e4m3fn):
    out = model(x)  # 直接 FP8 推理, 无需校准
```

#### 11.11.9 梯度检查点 (Gradient Checkpointing):用时间换空间

**文字**:反向传播需要保存前向的所有中间激活(activation)用于链式法则。L 层 Transformer,每层激活占 B$\cdot$T$\cdot$D$\cdot$4 字节,总显存 $O(L·B·T·D)$,层数深时显存爆炸。**Gradient Checkpointing** 的核心思想:前向时只保存少量"检查点"激活(每 √L 层一个),其余中间激活用完就释放;反向时从最近检查点重新跑前向(用时间换空间),重新计算这些层的激活再算梯度。显存从 $O(L)$ 降到 $O(√L)$,L=64 时省 8 倍,代价是约 30% 额外前向计算。搜推广中常对 Transformer 的深层做 checkpoint,浅层不算(便宜)。

**公式**:

```text
无 checkpoint:  memory = O(L · B · T · D)
有 checkpoint:  memory = O(√L · B · T · D)
额外计算:       +30% 前向时间(重计算被释放的激活)
最优策略:       checkpoint 平方根层(L=64 → 每 8 层设 1 个 checkpoint)
```

**代码**:

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 1. 单层包装(use_reentrant=False 是 PyTorch 2.1+ 推荐)
class CheckpointedBlock(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        return x

# 2. 搜推广实战:只对深层做 checkpoint,浅层不浪费
class SASRec(nn.Module):
    def __init__(self, n_layers=8):
        super().__init__()
        self.early = nn.ModuleList([TransformerBlock(256) for _ in range(2)])
        self.late  = nn.ModuleList([TransformerBlock(256) for _ in range(n_layers-2)])
    def forward(self, x):
        for blk in self.early: x = blk(x)                            # 浅层不 checkpoint
        for blk in self.late:  x = checkpoint(blk, x, use_reentrant=False)  # 深层做
        return x
# 收益: 8 层 SASRec 显存从 ~6GB 降到 ~3GB, 速度下降约 15%
```

---

### 11.13 进阶实战专题(原书未深入覆盖)

> 本节是第 11 章的"加餐篇",挑了三个工程上"会用、但原书没讲透"的主题:
> - **11.13.1** Flash Attention 的严格数学推导(Online Softmax + Tiling + IO 复杂度)
> - **11.13.2** QAT vs PTQ 全景对比,把量化从"推理"延伸到"训练"
> - **11.13.3** PyTorch Profiler 实战,一套"找瓶颈→优化→复测"的工程模板
>
> 三个主题彼此独立,可按需阅读。

#### 11.13.1 Flash Attention 完整推导:Online Softmax 与 Tiling

**文字**:11.6.8.10 讲了 Flash Attention 的直觉("避免物化 N×N 矩阵"),但没给严格推导。本节补完三件事:① Online Softmax 的递推关系(为什么分块后还能保持数值一致);② Tiling 的数学依据(分块大小的最优选取);③ HBM IO 复杂度的精确计算(Dao et al. 2022 的核心结论)。这套推导也是理解 **Flash Attention-2 / 3**、**xformers**、**FlexAttention** 共同的理论基础。

**公式**——从标准 Attention 出发:

$$
\text{标准 softmax}: \quad m = \max_j s_j, \quad p_j = \frac{e^{s_j - m}}{\sum_k e^{s_k - m}}, \quad o = \sum_j p_j v_j
$$

**问题**:要算 $o$,必须先看到所有 $s_j$ 才能确定 $m$ 和分母 $\sum e^{s_k - m}$——这意味着 softmax 不可"流式"做,必须先物化整个 $N \times N$ 矩阵到 HBM。

**Online Softmax 的递推**(Milakov & Gimelshein 2018,Flash Attention 的关键灵感):

把 $K$ 沿序列维切成 $T_c$ 块 $B_1, B_2, \ldots, B_{T_c}$,处理到第 $i$ 块时维护三个标量 $(m_i, \ell_i, o_i)$:

$$
\begin{aligned}
m_i &= \max\!\Big(m_{i-1},\ \max_{j \in B_i} s_{ij}\Big) \\
\ell_i &= \ell_{i-1} \cdot e^{m_{i-1} - m_i} \;+\; \sum_{j \in B_i} e^{s_{ij} - m_i} \\
o_i &= o_{i-1} \cdot \frac{\ell_{i-1} \cdot e^{m_{i-1} - m_i}}{\ell_i} \;+\; \frac{1}{\ell_i}\sum_{j \in B_i} e^{s_{ij} - m_i} \cdot v_{ij}
\end{aligned}
$$

**数值等价性证明**(归纳法):
- **基例** $i=1$:$m_1 = \max_{j \in B_1} s_j$,$\ell_1 = \sum_{j \in B_1} e^{s_j - m_1}$,$o_1 = \sum_{j \in B_1} p_j v_j$,退化为标准 softmax ✓
- **归纳步** $i \to i+1$:$m_{i+1}$ 是全局 max,旧贡献 $o_i$ 需乘 $e^{m_i - m_{i+1}}$(因为旧 softmax 的分母基于旧 max $m_i$,换新 max 后所有项都要 rescale),新贡献按新 max 归一化;$\ell_{i+1}$ 同理 ✓
- **极限** $i = T_c$:$o_{T_c}$ 严格等于一次性 softmax 的结果 ✓

**HBM IO 复杂度**(Dao et al. 2022 Theorem 1):

设序列长 $N$,头维 $d$,SRAM 大小 $M$(A100 是 192 KB/SM)。分块大小 $B_r$($Q$ 块)、$B_c$($K,V$ 块)受 $B_r d + 2 B_c d \le M$ 约束。最优取 $B_r = B_c = \Theta(M^{1/2} d^{-1/2} M^{1/2}) = \Theta(M/d)$(原文有更细的讨论,这里给直觉):

$$
\text{标准 Attention HBM IO}: \Theta(N^2 + N d) \quad(\text{读写注意力矩阵})
$$

$$
\text{Flash Attention HBM IO}: \Theta\!\left(\frac{N^2 d^2}{M}\right) \quad(\text{不再读写 } N^2 \text{ 中间矩阵})
$$

直觉:**$d$ 远小于 $N$ 时,标准是 $O(N^2)$,Flash 是 $O(N^2)$ 但常数从 1 降到 $d^2 / M \approx 1/10$**(以 $d=64, M=100\text{KB}$ 估算);**真正的胜利是显存**——**peak memory 从 $O(N^2)$ 降到 $O(N)$**,长序列(8K/32K/128K)训练成为可能,这就是 HSTU/GenRec 能跑超长序列的根本原因。

**代码**——手写一个"教学版" Flash Attention(用 PyTorch 模拟 SRAM + HBM):

```python
import torch
import math

def flash_attention_forward(Q, K, V, block_q=64, block_k=64):
    """
    Q, K, V: [B, H, N, d]   标准形状
    返回 out: [B, H, N, d], 与 torch 实现数值一致(误差 < 1e-5)
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # 1. 切 Q 为 B_r 大小的块(外层循环)
    out = torch.zeros_like(Q)
    L   = torch.zeros(B, H, N, device=Q.device)   # log-sum-exp, 用于反向
    M   = torch.full((B, H, N), float('-inf'), device=Q.device)

    for i in range(0, N, block_q):
        Qi = Q[:, :, i:i+block_q, :] * scale               # 加载到 SRAM
        oi = torch.zeros(B, H, block_q, d, device=Q.device)
        li = torch.zeros(B, H, block_q, device=Q.device)
        mi = torch.full((B, H, block_q), float('-inf'), device=Q.device)

        for j in range(0, N, block_k):
            Kj = K[:, :, j:j+block_k, :]                    # K 块加载
            Vj = V[:, :, j:j+block_k, :]
            Sij = Qi @ Kj.transpose(-2, -1)                 # [B,H,Br,Bc], SRAM 中

            # —— Online Softmax 更新 ——
            mi_new = torch.maximum(mi, Sij.amax(dim=-1))    # 新 max
            Pij    = torch.exp(Sij - mi_new.unsqueeze(-1))  # 未归一化概率
            li_new = li * torch.exp(mi - mi_new) + Pij.sum(dim=-1)
            oi     = oi * (li * torch.exp(mi - mi_new)).unsqueeze(-1) / li_new.unsqueeze(-1) \
                     + (Pij @ Vj) / li_new.unsqueeze(-1)
            mi, li = mi_new, li_new

        out[:, :, i:i+block_q, :] = oi
        M[:, :, i:i+block_q]      = mi
        L[:, :, i:i+block_q]      = li + torch.log(li)     # = mi + log(li)
    return out, M, L

# ——— 验证与标准实现的数值一致性 ———
B, H, N, d = 1, 2, 128, 32
torch.manual_seed(0)
Q = torch.randn(B, H, N, d, device='cuda')
K = torch.randn(B, H, N, d, device='cuda')
V = torch.randn(B, H, N, d, device='cuda')

out_std = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
out_flash, _, _ = flash_attention_forward(Q, K, V, block_q=32, block_k=32)
print(f"max |out_std - out_flash| = {(out_std - out_flash).abs().max().item():.2e}")
# 输出: max |out_std - out_flash| = 1.19e-06  (FP32 累积误差)
```

**工程落地**——生产永远用 `flash_attn` 官方库,不要自己写:

```python
# 1. 优先用 PyTorch 2.2+ 内置(自动 dispatch 到 flash_attn / mem_efficient)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

# 2. 需要更细控制(可变 mask、attention bias)时,直接调 flash_attn
# pip install flash-attn==2.5.8  # 与 CUDA/PyTorch 版本严格对齐!
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)

# 3. 搜推广 HSTU/GenRec 的典型配置
#    block_size 自动选择;head_dim 必须 ∈ {64, 128, 256};不支持 fp32
```

**易混淆点**:
- **"$O(N)$ 显存"≠"$O(N)$ 计算"**:Flash Attention 的 FLOPs 与标准完全一致(甚至略多 1 次 rescale),**省的是显存和 HBM IO,不是 FLOPs**。
- **不能直接用于 FP32**:`flash_attn_func` 只支持 FP16/BF16;FP32 训练要 autocast 转 BF16 再调用。
- **不传 mask**:Flash Attention 的 causal mask 是硬编码的(三角结构);任意 mask 要用 `flash_attn_varlen_func` 或 `xformers.memory_efficient_attention`。
- **head_dim 限制**:head_dim 必须是 8 的倍数且 ≤ 256,搜推广 Embedding dim 64/128 时 8 头正好,没问题;若 head_dim = 100 这种"奇怪数"会失败。

---

#### 11.13.2 INT8 量化训练详解:QAT vs PTQ 全景对比

**文字**:11.11.8 讲了**推理**侧的 INT8 量化(PTQ,Post-Training Quantization),本节补完**训练**侧的 QAT(Quantization-Aware Training)。QAT 在训练时把"假量化"(fake quant)节点插入计算图,前向用 quant→dequant 后的"伪量化"值算,反向通过 **STE**(Straight-Through Estimator)把梯度原样回传——模型学到的权重自动适应量化误差,精度损失通常 < 0.1%。PTQ 则训练完直接量化(仅需几百 batch 校准),快但损失 0.5–2%。**搜推广实战选型**:小模型(< 1B)优先 PTQ(快)、大模型或精度敏感场景(CTR 主力 / 召回主模型)用 QAT 微调几十 epoch 兜底。

**公式**——仿射量化(INT8 对称 / 非对称):

$$
x_q = \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{x - z}{s}\right),\ -2^{b-1},\ 2^{b-1}-1\right), \quad \hat{x} = s \cdot x_q + z
$$

其中 $s$ 是 scale,$z$ 是 zero-point,$b=8$ 时量化范围 $[-128, 127]$。**Fake Quant**(训练时模拟,推理时移除):

$$
\hat{x} = s \cdot \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{x - z}{s}\right),\ -128,\ 127\right) + z
$$

前向用 $\hat{x}$ 算(引入量化误差,让模型"看到"这个误差),反向时 **$\partial \mathrm{round}(x) / \partial x = 1$**(STE 近似,实际 round 不可微),其余反传照旧:

$$
\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbb{1}\!\left[x \in (z - 128s,\ z + 127s)\right]
$$

**注意**:STE 把 clamp 之外的梯度截断为 0(避免 outlier 推动权重乱跑),这正是 QAT 能稳的关键。

**PTQ vs QAT 全景对比表**:

| 维度 | PTQ (训练后量化) | QAT (量化感知训练) |
| :--- | :--- | :--- |
| **何时量化** | 训练完之后 | 训练过程中同步 |
| **需要数据** | 校准集(100–500 batch) | 完整训练集 + 几十 epoch 微调 |
| **精度损失** | 0.5–2%(activation outlier 时更差) | < 0.1%(模型主动适配) |
| **耗时** | 分钟级 | 与正常训练相当(每步 +5–10%) |
| **适用场景** | 小模型、CV/简单 NLP、对 AUC 容忍 | 大模型、搜推广主力、精度红线场景 |
| **实现成本** | 低(`torch.ao.quantization` 几行) | 中(需改 model + 微调 pipeline) |
| **搜推广经验** | Embedding 旁路 + 校准 → 首选 | AUC 红线 < 0.5% 时兜底 |

**代码**——PyTorch 2.x 的完整 QAT 流程:

```python
import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig, MinMaxObserver, PerChannelMinMaxObserver,
    prepare_qat, convert, get_default_qat_qconfig
)

# ========== 1. 准备:把普通 BN 换成 BNReLU 融合(quant 友好) ==========
class QuantReadyMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))

# 融合 BN+ReLU(必须在 prepare_qat 之前)
model_fp32 = QuantReadyMLP(128, 256, 1).train()
model_fp32.fuse_model_eval = lambda: None  # 占位
# 实际工程用 torch.ao.quantization.fuse_modules
from torch.ao.quantization import fuse_modules
fused = fuse_modules(model_fp32, `'bn1', 'relu'`)

# ========== 2. 配置 QAT qconfig(per-channel weight,搜推广标配) ==========
qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine),
)
fused.qconfig = qconfig

# ========== 3. 准备 QAT 模型(插入 fake quant 节点) ==========
model_qat = prepare_qat(fused, inplace=True)

# ========== 4. 微调(正常训练循环,只是 model 内部多了 fake quant) ==========
optimizer = torch.optim.AdamW(model_qat.parameters(), lr=1e-4)
for epoch in range(5):                          # QAT 微调 epoch 少
    for x, y in train_loader:
        out = model_qat(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# 5. 训练完冻结 BN stats、关闭 fake quant,转成真 INT8
model_qat.eval()
model_int8 = convert(model_qat.eval(), inplace=False)   # 权重变 int8,推理时激活动态 quant

# 6. 验证精度
with torch.no_grad():
    p_fp32 = torch.sigmoid(model_fp32(x_calib)).numpy()
    p_int8 = torch.sigmoid(model_int8(x_calib)).numpy()
auc_fp32 = roc_auc_score(y_calib, p_fp32)
auc_int8 = roc_auc_score(y_calib, p_int8)
print(f"FP32 AUC = {auc_fp32:.4f}, INT8 AUC = {auc_int8:.4f}, Δ = {auc_fp32-auc_int8:+.4f}")
# 典型: Δ ∈ [-0.001, -0.003], 搜推广可接受
```

**代码**——QAT 训练时的"反直觉"陷阱:

```python
# 陷阱 1:model.train() 与 model.eval() 行为不同
model_qat.train()    # fake quant 启用,BN 用 batch stats
model_qat.eval()     # fake quant 仍启用!但 BN 用 running stats(不要再 train)

# 陷阱 2:QAT 的学习率要比正常训练小 10–100 倍(微调阶段,别再"学新东西")
# 搜推广经验: 5–20 epoch, lr=1e-5 ~ 1e-4(原训练的 1/10)

# 陷阱 3:Embedding 层一般不量化(精度损失大,体积也未必是瓶颈)
# 解决: nn.Embedding.from_pretrained(emb, freeze=True) 直接保留 FP32
#       或者用 bitsandbytes 的 8-bit Adam 优化器省优化器状态
```

**与 11.11.8 的衔接**:11.11.8 讲的"per-tensor / per-channel"对应**推理时**的量化粒度;QAT 的 fake quant 也分粒度,**per-channel 权重 + per-tensor 激活是搜推广默认组合**。再高级一档是 **SmoothQuant**(用 $Y = (X \cdot s) \cdot (W / s)$ 把 activation 的难度转移到 weight,见 11.11.8 文字),可与 QAT 正交叠加用。

---

#### 11.13.3 PyTorch Profiler 实战:从 profile 到优化的完整流程

**文字**:Profiler 是 PyTorch 工程的"X 光机",能告诉你每个 GPU kernel 花了多少时间、CPU 是不是卡在数据加载、显存峰值在哪一步。一套规范的 profile 流程通常包含 5 步:① 准备环境(固定随机种子、预热 3–5 步);② 用 `torch.profiler` 采 10–20 step;③ 用 `kineto` 表格 / TensorBoard / Chrome trace 可视化;④ 定位瓶颈(数据加载?kernel launch overhead?某个 matmul 慢?);⑤ 优化 + 复测。**搜推广训练最容易踩的三个瓶颈**:**DataLoader worker 不够**(CPU-bound)、**小 tensor 频繁 launch**(CPU-bound kernel launch overhead)、**attention 未用 flash**(IO-bound,长序列才显著)。

**公式**——关键性能指标:

$$
\begin{aligned}
\text{step time} &= t_{\text{forward}} + t_{\text{backward}} + t_{\text{optimizer}} + t_{\text{io wait}} \\
\text{GPU utilization} &= \frac{\sum t_{\text{kernel}}}{\text{step time}} \quad(\text{健康值} > 70\%) \\
\text{瓶颈类型} &=
\begin{cases}
t_{\text{io wait}} \gg 0, & \text{DataLoader 太慢 → 加 worker} \\
t_{\text{kernel}} \ll \text{step time}, & \text{CPU launch overhead → 合并小算子 / torch.compile} \\
\text{单个 kernel 占比} > 30\%, & \text{热点算子 → 换 flash / cuBLAS / 自定义}
\end{cases}
\end{aligned}
$$

**代码**——标准 Profiling 模板(10 行,塞进训练循环即可):

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

# —— 1. schedule: warmup 2 步、active 5 步、repeat 1 次(共 7 step 数据)——
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=5, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log/profiler'),  # tb 可视化
    record_shapes=True,                # 记录每算子的输入 shape
    profile_memory=True,               # 记录显存(找 OOM 杀手)
    with_stack=True,                   # 记录 Python 调用栈
)

# —— 2. 训练循环里包一层 ——
model.train()
prof.start()
for step, batch in enumerate(train_loader):
    out = model(batch)
    loss = criterion(out, batch.y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    prof.step()                        # 通知 profiler "一步结束"
    if step >= 7: break                # schedule 控制总步数
prof.stop()
```

**代码**——三大常见瓶颈的诊断与修复:

```python
# ——— 瓶颈 1:DataLoader 太慢(CPU-bound, GPU 空转) ———
# 现象: profiler 表格里 cuda_time 很低, 但 self_cuda_time/host_time 比例 < 30%
# 诊断: 打印 batch 加载时间
import time
t0 = time.time(); next(iter(train_loader)); print(f"1 batch 加载: {time.time()-t0:.3f}s")
# 修复:
train_loader = DataLoader(ds, batch_size=B,
    num_workers=8,        # 经验: 2 * num_GPU
    pin_memory=True,      # 锁页内存, 加速 H2D 拷贝
    prefetch_factor=4,    # 预取 batch 数
    persistent_workers=True)

# ——— 瓶颈 2:小 tensor 频繁 launch(常见于搜推广的 lookup + scatter) ———
# 现象: profiler 里 kernel 数 > 1000, 但每个 kernel 平均耗时 < 10μs
# 诊断: print(prof.key_averages().table(sort_by="count", row_limit=10))
# 修复:
#   (a) 合并 Embedding 查表: torch.nn.EmbeddingBag 或自己 concat
#   (b) torch.compile(model) —— Inductor 会自动 fuse
#   (c) CUDA Graphs 捕获固定输入的图: 适合推理 / 固定 shape 训练

# ——— 瓶颈 3:attention 占大头(长序列) ———
# 现象: aten::scaled_dot_product_attention 占比 > 30%
# 诊断: 检查 sdp_kernel 当前选了哪个 backend
print(torch.backends.cuda.flash_sdp_enabled(),  # True = 用了 Flash
      torch.backends.cuda.mem_efficient_sdp_enabled(),
      torch.backends.cuda.math_sdp_enabled())
# 修复: 长序列 T > 1024 时强制 Flash
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**代码**——显存剖析(找 OOM 真凶):

```python
import torch

# 方案 A:memory snapshot(可视化推荐,Chrome 打开即可)
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... 跑一个 OOM 步骤 ...
torch.cuda.memory._dump_snapshot("oom.pickle")
# Chrome 打开 chrome://tracing, load → oom.pickle 即可看到每行显存分配

# 方案 B:实时刻画(线上监控)
print(f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
      f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
      f"max_allocated={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
# 搜推广常见问题: max_allocated 远小于 reserved → 显存碎片化
# 解决: torch.cuda.empty_cache() + 减少并行流 + 启用 expandable_segments
#      (PyTorch 2.1+: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
```

**搜推广 profile checklist**(背下来):

| 步骤 | 检查项 | 健康标准 |
| :--- | :--- | :--- |
| 1 | GPU 利用率 `nvidia-smi` | > 70% |
| 2 | Profiler `self_cuda_time` / total | > 60%(其余是 CPU overhead) |
| 3 | Top-3 kernel 累计占比 | < 70%(没有单点霸榜) |
| 4 | DataLoader `num_workers` | ≥ 2 × GPU 数 |
| 5 | `pin_memory=True` | 必开 |
| 6 | Flash Attention | 序列 > 1024 时启用 |
| 7 | 混合精度 (BF16) | A100/H100 必开 |
| 8 | `torch.compile` | 静态图 / 推理建议开 |
| 9 | 显存 `max_allocated` / 卡容量 | < 80%(留 batch size 弹性) |
| 10 | 复测同 seed 同 step 数 | 优化后 step time 下降 > 10% 才算有效 |

**与 11.11.5 调试的衔接**:Profiler 找瓶颈,**`register_hook` + `set_detect_anomaly` 找数值问题**(梯度爆炸、NaN、loss 抖动)。两者互补,Profile 工具是"宏观体检",hook 是"微观活检"。

---

> **本节配套参考**: [[第12章-PyTorch进阶函数速查]] 包含 100+ 完整函数速查(签名+示例+陷阱)
> **本节配套理论**: [[第21章-信息论深层次理解]] 涵盖信息几何、互信息下界、信息瓶颈等深层次话题

---

> 如果前面 3 章(第 9/10/11 章)是"零件",那第 14 章就是"组装"——把 Python 数据结构、数据处理 Pipeline、PyTorch Tensor 和 Embedding 全部串起来,构建一个完整、可运行的搜推广训练数据生成流程。

---

### 11.12 分布式训练三件套:DDP / FSDP / DeepSpeed 选型对比

> **为什么单独立这一节?** 11.11.7 详细拆解了 DDP 的 Ring AllReduce 机制,但当模型 + 优化器状态超过单卡显存(搜推广大模型的 Embedding 表常达百 GB)时,DDP 直接 OOM。本节把 DDP / FSDP / DeepSpeed 放在同一张表上对比,给出"模型多大 → 选哪个"的决策树。

#### 11.12.1 三者定位与适用规模对比

**文字**:三者的本质区别在于**参数、梯度、优化器状态**各放在哪里。**DDP** 每张卡都复制完整模型,只把梯度按 bucket 切到多卡做 AllReduce;**FSDP** (Fully Sharded Data Parallel) 进一步把参数、梯度、优化器状态都分片(shard),前向/反向时用 AllGather 临时重建当前层完整参数;**DeepSpeed ZeRO** 是微软等价方案,ZeRO-1/2/3 与 FSDP 的 `OPTIM_STATE / SHARD_GRAD / FULL_SHARD` 模式一一对应。节省的显存 $\propto$ 卡数 $N$,代价是通信量增加。

| 方案 | 切分粒度 | 单卡显存(模型占用) | 适用模型规模 | 每步通信量 |
| :--- | :--- | :--- | :--- | :--- |
| **DDP** | 仅梯度 bucket | $M$ | ≤ 7B(AMP) | $2M$ (AllReduce) |
| **FSDP / ZeRO-1** | Optimizer State | $M + O/N$ | 7B–13B | $2M$ |
| **FSDP / ZeRO-2** | + Gradients | $(M+G+O)/N$ 近似 $M/N + O/N$ | 13B–30B | $2M$ |
| **FSDP / ZeRO-3** | + Parameters | $(M+G+O)/N$ | 30B–70B+ | $3M$ (AllGather + ReduceScatter) |
| **DeepSpeed + 3D** | TP + PP + ZeRO-3 | $(M+G+O)/(N \cdot T)$ | 70B+ LLM | 含 TP/PP 额外通信 |

> 表注:$M$=参数,$G$=梯度,$O$=优化器状态;$T$=TP 切分数。FP32 训练 Adam 优化器 $O = 8M$ (FP32 主参数 + 2×FP32 动量),因此 $M+G+O \approx 14M$。

#### 11.12.2 通信与显存的数学对比

**公式**:

$$
\text{DDP 单卡显存} = M + G + O = 2M + 4M + 8M = 14M \quad (\text{FP32, Adam})
$$

$$
\text{FSDP FULL\_SHARD 单卡显存} = \frac{M + G + O}{N} = \frac{14M}{N}
$$

$$
\text{FSDP 总通信量} = 3M \quad (\text{前向 AllGather} + \text{反向 AllGather} + \text{ReduceScatter})
$$

$$
\text{加速比上限 (Amdahl)} \leq 1 - \frac{T_{\text{comm}}}{T_{\text{comp}} + T_{\text{comm}}}
$$

DeepSpeed ZeRO-3 与 FSDP FULL_SHARD 在数学上**完全等价**,只多支持三项能力:(a) **CPU Offload**——把 $O$ 卸载到 CPU 内存,单卡可训练参数量再翻倍;(b) **NVMe Offload**——卸载到本地 SSD,适合千亿参数;(c) **3D Parallel**——张量并行(TP,层内切分)+ 流水线并行(PP,层间切分)+ 数据并行(DP)混合,Hopper 时代 70B+ LLM 标配。

#### 11.12.3 选型决策树 + 代码骨架

**文字**:面对一个新模型,先问三个问题:(1) **模型 + 优化器 + 激活能否塞进单卡?** 是 → 不需要分布式,跳过本节。(2) 装不下但**只需多卡复制**就能跑?→ DDP 即可;8 卡 A100 上 DDP 配合 AMP 能跑到 7B 模型。(3) 单卡连模型都放不下?→ FSDP FULL_SHARD + Activation Checkpointing;若还爆,叠加 CPU Offload。(4) 70B+ LLM?→ FSDP ZeRO-3 + Activation Checkpointing + CPU Offload,或 DeepSpeed + 3D Parallel。

**代码 1**(FSDP 实战骨架,推荐主选):

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 1. FSDP 包装策略:按 Transformer Block 分片(粒度粗,通信少)
fsdp_cfg = dict(
    mixed_precision=MixedPrecision(           # BF16 计算, FP32 主参数
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取下一层参数
    cpu_offload=True,        # 优化器状态 → CPU 内存, 显存不够时必开
    limit_all_gathers=True,  # 限制并发 AllGather 数量, 防 OOM
)

# 2. 按层包装(只对 TransformerBlock 做 FSDP, Embedding 留在 rank 0)
model = FSDP(model,
             auto_wrap_policy=transformer_auto_wrap_policy(
                 transformer_layer_cls={TransformerBlock}),
             use_orig_params=True,            # 让 optimizer 看到原始参数
             **fsdp_cfg)
# 3. 启动: torchrun --nproc_per_node=8 train.py
```

**代码 2**(DeepSpeed 等价方案,适合跨节点 + Offload):

```json
// ds_config.json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,                    // 等价于 FSDP FULL_SHARD
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,          // 通信与计算流水
    "contiguous_gradients": true
  }
}
```

```python
import deepspeed
# 一行初始化,等价于上面的 FSDP 包装
model, optimizer, _, scheduler = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config="ds_config.json")
# 启动: deepspeed --num_gpus=8 train.py
```

**经验法则**:8 卡内、能上 FSDP 就上 FSDP(PyTorch 原生、调试友好、与 `torch.compile` 兼容);跨节点(16+ 卡) + 需要 CPU/NVMe Offload 选 **DeepSpeed**;3D Parallel(TP+PP+DP)是 70B+ LLM 标配,工程复杂度指数级上升,搜推广场景**一般用不到**——搜推广模型的瓶颈通常在 Embedding 表(几十 GB 查表),不是 Transformer 参数量,优先用 **ZeRO-3 + Embedding 分片** 而非 TP。

---

## 11.14 神经网络模块化:Sequential/ModuleList/ModuleDict 实战对比

### 11.14.1 三种容器类对比

| 容器 | 何时用 | 优点 | 缺点 | 典型场景 |
| :--- | :--- | :--- | :--- | :--- |
| `nn.Sequential` | 层按顺序执行,无分支 | 简洁,可直接 forward | 无法跳过层/分支 | 简单 MLP、CNN |
| `nn.ModuleList` | 任意结构,需要索引 | 灵活,可遍历 | 不会自动 forward | 多专家 (MoE)、可配置层数 |
| `nn.ModuleDict` | 任意结构,需要名字 | 可按名字访问 | 同 ModuleList | 多任务塔 (MMoE Tower) |

```python
import torch
import torch.nn as nn

# ===== nn.Sequential: 简单堆叠 =====
seq_model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
# 直接调用
out = seq_model(torch.randn(32, 128))  # [32, 10]

# ===== nn.ModuleList: 动态层数 =====
class DynamicMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(len(layer_dims) - 1)
        ])
        self.activations = nn.ModuleList([
            nn.ReLU() for _ in range(len(layer_dims) - 2)
        ] + [nn.Identity()])  # 最后一层不加激活

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

# ===== nn.ModuleDict: 多任务多塔 =====
class MultiTowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU()
        )
        # 多个任务塔,按名字访问
        self.towers = nn.ModuleDict({
            'ctr': nn.Linear(64, 1),
            'cvr': nn.Linear(64, 1),
            'like': nn.Linear(64, 1),
        })

    def forward(self, x):
        feat = self.shared_backbone(x)
        return {name: tower(feat) for name, tower in self.towers.items()}
```

### 11.14.2 钩子 (Hooks) 实战:中间层特征提取

```python
class FeatureExtractor:
    """
    用 forward hook 提取任意中间层特征。
    """
    def __init__(self, model, layer_names):
        self.model = model
        self.features = {}
        self.handles = []

        for name in layer_names:
            handle = self._register_hook(name)
            self.handles.append(handle)

    def _register_hook(self, name):
        def hook(module, input, output):
            # 关闭梯度
            self.features[name] = output.detach()
        # 找到层
        for n, m in self.model.named_modules():
            if n == name:
                return m.register_forward_hook(hook)
        raise ValueError(f"Layer {name} not found")

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, x):
        self.model(x)
        return self.features


# 使用:提取 hidden layer 表征
extractor = FeatureExtractor(model, ['encoder.layer.5', 'encoder.layer.11'])
features = extractor(input_ids)
print(features['encoder.layer.5'].shape)  # [B, L, D]
extractor.remove()
```

### 11.14.3 PyTorch 常见报错 + 解决方案速查

| 报错 | 原因 | 解决方案 |
| :--- | :--- | :--- |
| `RuntimeError: Expected all tensors to be on the same device` | 模型和输入在不同 device | `model.to(device); data.to(device)` |
| `RuntimeError: shape mismatch` | 矩阵维度不对 | 用 `print(x.shape)` 逐步排查 |
| `RuntimeError: CUDA out of memory` | 显存不足 | 降 batch, 用 checkpoint, 混合精度 |
| `RuntimeError: Trying to backward through the graph a second time` | 用了同一个计算图两次 | `loss.backward(retain_graph=False)` 或 detach |
| `TypeError: 'Tensor' object is not callable` | 把张量当函数调 | 检查变量名覆盖 (e.g., `x = nn.Linear` 然后 `x(...)`) |
| `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | 用了 `torch.no_grad` 或 `tensor.detach()` | 把 detach 后的张量拿去 backward 会失败 |
| `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)` | broadcast 失败 | 显式 `unsqueeze` 或 `expand` |
| `RuntimeError: index out of range in self` | Embedding 越界 | 加 clamp 或 mask OOV |
| `KeyError` in DataLoader | collate_fn 漏字段 | 检查 dict key 一致性 |
| `ValueError: too many values to unpack` | 忘了某个返回值 | 检查模型 forward 返回值数量 |

### 11.14.4 性能分析:torch.profiler 完整示例

```python
from torch.profiler import profile, record_function, ProfilerActivity

model = nn.Transformer(...).cuda()
input = torch.randint(0, 1000, (32, 50)).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        for _ in range(10):
            out = model(input)

# 打印
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 保存 Chrome trace (可视化)
prof.export_chrome_trace("trace.json")
# 用 chrome://tracing 或 https://ui.perfetto.dev 打开

# 按 op 类型查看
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="self_cuda_memory_usage", row_limit=10
))
```

### 11.14.5 配套章节

- `./第12章-PyTorch进阶函数速查.md` — 各种函数参考
- `./第10章-数据处理.md` — DataLoader 工程
- `./第15章-训练流程与工程实践.md` — 训练 pipeline 对接
- `./第17章-高级工程实践.md` — 部署侧的 PyTorch



## 深度扩展 · 多视角系统展开(续)

### 110. 第 11 章:PyTorch 基础 - 工程视角 视角

**核心议题**: 本节从工程视角视角切入,系统展开 "第 11 章:PyTorch 基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:PyTorch 基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工程视角视角的展开**

从工程视角视角出发,我们可以把 "第 11 章:PyTorch 基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调工程视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 11 章:PyTorch 基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"工程视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 11 章:PyTorch 基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:PyTorch 基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:PyTorch 基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:PyTorch 基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工程视角视角切入,系统展开了 "第 11 章:PyTorch 基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 11 章:PyTorch 基础 - 评估视角 视角

**核心议题**: 本节从评估视角视角切入,系统展开 "第 11 章:PyTorch 基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:PyTorch 基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 评估视角视角的展开**

从评估视角视角出发,我们可以把 "第 11 章:PyTorch 基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调评估视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 11 章:PyTorch 基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"评估视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 11 章:PyTorch 基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:PyTorch 基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:PyTorch 基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:PyTorch 基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从评估视角视角切入,系统展开了 "第 11 章:PyTorch 基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 11 章:PyTorch 基础 - 工业落地 视角

**核心议题**: 本节从工业落地视角切入,系统展开 "第 11 章:PyTorch 基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:PyTorch 基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工业落地视角的展开**

从工业落地视角出发,我们可以把 "第 11 章:PyTorch 基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调工业落地视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 11 章:PyTorch 基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"工业落地" 视角,但落到代码层面是类似的骨架:

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

评估 "第 11 章:PyTorch 基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:PyTorch 基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:PyTorch 基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:PyTorch 基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工业落地视角切入,系统展开了 "第 11 章:PyTorch 基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 11 章:PyTorch 基础 - 前沿趋势 视角

**核心议题**: 本节从前沿趋势视角切入,系统展开 "第 11 章:PyTorch 基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:PyTorch 基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 前沿趋势视角的展开**

从前沿趋势视角出发,我们可以把 "第 11 章:PyTorch 基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调前沿趋势视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 11 章:PyTorch 基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"前沿趋势" 视角,但落到代码层面是类似的骨架:

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

评估 "第 11 章:PyTorch 基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:PyTorch 基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:PyTorch 基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:PyTorch 基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从前沿趋势视角切入,系统展开了 "第 11 章:PyTorch 基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 11 章:PyTorch 基础 - 召回视角 视角

**核心议题**: 本节从召回视角视角切入,系统展开 "第 11 章:PyTorch 基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:PyTorch 基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 召回视角视角的展开**

从召回视角视角出发,我们可以把 "第 11 章:PyTorch 基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调召回视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 11 章:PyTorch 基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"召回视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 11 章:PyTorch 基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:PyTorch 基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:PyTorch 基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:PyTorch 基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从召回视角视角切入,系统展开了 "第 11 章:PyTorch 基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



