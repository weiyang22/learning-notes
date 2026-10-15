---
title: "第05章-01-Self-Attention"
tags:
  - 深度学习
  - Self
  - Attention
description: "本章要点：Self-Attention (自注意力) 是 Transformer 的核心机制 (Vaswani et al., 2017)."
chapter_id: 深度学习手撕-05.01
last_updated: 2026-07-14
---

# 第05章-01-Self-Attention

> 放点积注意力)

> ⭐ **本章 (Transformer 架构) 是 LLM 面试的核心**, Self-Attention 是其中最最核心

---

## 1. 内容介绍

**Self-Attention** (自注意力) 是 Transformer 的核心机制 (Vaswani et al., 2017).

给定一个序列, Self-Attention 让每个位置能"看到"序列中所有其他位置, 并根据相关性加权聚合.

**公式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## 2. 为什么这么做

**历史背景**:
- RNN/LSTM 处理序列: 顺序计算, 无法并行, 长距离依赖丢失
- CNN 处理序列: 局部感受野, 难捕获全局
- Attention (Bahdanau 2014): 用于 seq2seq, 但只在 decoder 端
- **Self-Attention (2017)**: 让序列内部任意两位置直接交互, 突破 RNN 局限

**核心问题**:
- 如何让序列中任意两个位置直接建立关系?
- 如何并行计算 (替代 RNN 的顺序依赖)?
- 如何捕获长距离依赖 (替代 LSTM 的门控)?

---

## 3. 这么做的原因 (QKV 设计的动机)

**Q (Query) / K (Key) / V (Value) 的含义**:
- 类比数据库检索: Q = 查询, K = 索引, V = 内容
- QK^T 算"查询"和"索引"的相关性
- 乘 V 加权求和得到"查询的结果"

**为什么用 3 个矩阵而不是 1 个?**
- 让 Q 和 K 解耦, 增加表达能力
- Q/K 可以是不同子空间, 捕获不同相关性
- 多头机制需要不同子空间 (后面讲)

---

## 4. 解决了什么问题

- **并行计算**: Self-Attention 一次前向就是 $O(n^2 d)$ 矩阵乘法, GPU 友好
- **长距离依赖**: 任意两位置直接连接, 距离无关
- **可解释性**: 注意力权重可视化, 看到模型关注哪些位置

---

## 5. 意义

- 2017 年 Attention Is All You Need 论文
- 直接催生了 GPT / BERT / T5 / LLaMA
- 2023+ 的 LLM 时代, Self-Attention 是基础

---

## 6. 公式表达

**缩放点积注意力 (Scaled Dot-Product Attention)**:

```text
Q: (n, d_k)     # n 个 token, 每个 d_k 维
K: (n, d_k)
V: (n, d_v)

score = Q @ K^T / √d_k    # (n, n)
attn = softmax(score)        # (n, n)
out = attn @ V              # (n, d_v)
```

**为什么除以 $\sqrt{d_k}$?**
- QK^T 的方差是 $d_k$, 即 $O(d_k)$
- softmax 在大值时梯度小, 导致梯度消失
- 除以 $\sqrt{d_k}$ 让方差归一化, 训练稳定

**Mask (Decoder 用的 Causal Mask)**:
- 第 $i$ 个位置只能看 $\leq i$ 的位置
- mask = 上三角 $-\infty$, 加到 score 上

---

## 7. 题目设计

**为什么这么考**: Self-Attention 是 LLM 面试第一必问:
- 公式必须能默写
- QKV 来源要能讲清
- 为什么缩放要能解释
- 复杂度要能算
- 实现 (NumPy) 要能写

**面试陷阱**: 写不对 mask, 不懂因果掩码, 不知道缩放原因.

---

## 8. 题目

### 题目 1: 手写 Self-Attention (NumPy)

```python
def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, n, d_k)
    mask: (n, n) 或 (batch, n, n), True 表示屏蔽
    返回: (batch, n, d_v)
    """
    pass
```

### 题目 2: 实现 Causal Self-Attention (带 mask)

```python
def causal_self_attention(Q, K, V):
    """
    Decoder 用, 每个位置只能看左侧
    """
    pass
```

### 题目 3: 解释为什么除以 √d_k

```text
不缩放 vs 缩放 的对比, 用 softmax 梯度推导
```

### 题目 4: 计算 Self-Attention 的复杂度

```text
时间: O(n² d)
空间: O(n² + nd)
```

---

## 9. 答案

### 答案 1: Self-Attention (NumPy)

```python
import numpy as np

def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, n, d_k)
    """
    d_k = Q.shape[-1]
    # 1. 相似度: (batch, n, d_k) @ (batch, d_k, n) → (batch, n, n)
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)
    # 2. Mask (可选)
    if mask is not None:
        scores = scores + mask * (-1e9)  # 屏蔽位置得 -inf
    # 3. Softmax
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    # 4. 加权求和: (batch, n, n) @ (batch, n, d_v) → (batch, n, d_v)
    return attn @ V
```

### 答案 2: Causal Self-Attention

```python
def causal_self_attention(Q, K, V):
    """Decoder 用"""
    n = Q.shape[-2]
    # 上三角 mask: 第 i 行只允许看 ≤ i 的列
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)  # 上三角为 True
    return self_attention(Q, K, V, mask=mask)
```

**直觉**: 第 $i$ 个 token 的注意力分数, 第 $j > i$ 列被 mask 掉, 相当于"看不到未来".

### 答案 3: 为什么除以 √d_k (推导)

```text
设 Q, K 的每个分量独立, 均值 0, 方差 1.
则 (Q · K) 的每个分量 = Σ_i Q_i K_i
方差 = Σ Var(Q_i K_i) = Σ E[Q_i²]E[K_i²] = d_k

所以 (Q · K) / √d_k 方差归一化 → 1.
否则方差 d_k, softmax 集中, 梯度消失.
```

**实验**: 假设 $d_k=64$, 不缩放, QK^T 方差 64, softmax 后最大概率 ≈ 1, 梯度 ≈ 0. 缩放后方差 1, 正常训练.

### 答案 4: 复杂度

| 维度 | 复杂度 |
|---|---|
| Q @ K^T | $O(n^2 d_k)$ |
| Softmax | $O(n^2)$ |
| attn @ V | $O(n^2 d_v)$ |
| **总时间** | $O(n^2 d)$ (n 主导) |
| **空间** | $O(n^2)$ (注意力矩阵) |

**对比**:
- RNN: $O(n \cdot d^2)$ 时间, 顺序, **不能并行**
- CNN: $O(\log n \cdot d^2)$ 时间, 局部感受野
- **Self-Attention: $O(1)$ 距离, 完全并行**

---

---

## 10. PyTorch 版(高效实现要点)

```python
import torch, torch.nn.functional as F

def sdpa(q, k, v, causal=False):
    # q,k,v: (B, H, T, d)
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d**0.5        # (B,H,T,T)
    if causal:
        T = q.size(-2)
        m = torch.triu(torch.ones(T, T, device=q.device), 1).bool()
        scores = scores.masked_fill(m, float('-inf'))
    attn = scores.softmax(-1)
    return attn @ v

# 生产直接用融合内核 (走 FlashAttention 后端, 省显存提速):
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

> `F.scaled_dot_product_attention` 会自动选 FlashAttention/memory-efficient 后端,是训练/推理的默认选择(见 [6.4 Flash Attention](../第06章-LLM核心技术/第06章-04-Flash-Attention.md))。

---

## 11. 深入: 三种 mask 与它们的用途

| mask | 形状/作用 | 场景 |
| :--- | :--- | :--- |
| **Causal mask** | 上三角 $-\infty$,看不到未来 | Decoder/GPT 自回归 |
| **Padding mask** | 屏蔽 `<pad>` 位置 | 变长 batch 对齐 |
| **双向(无 mask)** | 全可见 | Encoder/BERT |

Padding + causal 常同时存在(合并成一个 mask 加到 scores)。忘记 padding mask 会让模型注意到无意义的填充位。

---

## 12. 深入: 为什么 softmax 大值梯度会消失(补齐推导)

softmax 输出 $s$,雅可比 $\partial s_i/\partial z_j = s_i(\delta_{ij}-s_j)$。当某 $s_i\to1$(分布极尖),该行雅可比 $\to0$ → 上游梯度传不回去。QK^T 方差为 $d_k$ 时 logits 尺度大,softmax 易饱和;除以 $\sqrt{d_k}$ 把方差压回 1,避免饱和。这是缩放的**梯度视角**(第3节是方差视角,两者互补)。

---

## 13. 深入: Attention 是"可微的软检索"

把注意力理解为**软字典查询**:query 和每个 key 算相似度 → softmax 归一化成权重 → 对 value 加权求和。相比硬检索(取 top-1),软检索处处可微,能端到端训练。这个视角贯通 RAG 检索、记忆网络、Pointer Network(见 [Agent技术第02章](../../Agent技术/第02章-RAG检索增强生成.md))。

---

## 14. 深入: 与 MHA / GQA / MLA 的接续

单头 Self-Attention 只在一个子空间建模关系。多头把它并行到多个子空间(见 [5.2 多头注意力](第05章-02-多头注意力.md));推理时 KV Cache 成瓶颈,催生 MQA/GQA/MLA(见 [LLM发展第04章](../../LLM发展/第04章-注意力与效率演进.md))。这条线是"表达力 → 效率"的完整演进。

---

## 15. 面试高频(补充)

1. 三种 mask 分别用于? → causal(自回归)/padding(变长)/无(双向)。
2. 缩放的两个视角? → 方差归一化 + 防 softmax 饱和梯度消失。
3. 为什么 Attention 能并行而 RNN 不能? → 无顺序依赖,一次矩阵乘。
4. 生产里用什么实现? → `F.scaled_dot_product_attention`(FlashAttention 后端)。

---

> **本节版本**: v2.0 (2026-07-16, 加厚: PyTorch/mask/梯度推导/软检索/演进)
> **配套**: Ch05 Transformer 架构目录
> **下一节**: [5.2 - Multi-Head Attention](第05章-02-多头注意力.md)

---

*Self-Attention 是 LLM 的"原子". 不会, 别说自己懂 LLM.*

---

## 16. Self-Attention 的完整数学推导

### 16.1 从输入到 Q、K、V

给定 $X\in\mathbb{R}^{B\times T\times d_{model}}$，单头投影为：

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$$

其中 $W_Q,W_K\in\mathbb{R}^{d_{model}\times d_k}$，$W_V\in\mathbb{R}^{d_{model}\times d_v}$。自注意力的 Q、K、V 来自同一序列；交叉注意力的 Q 来自目标序列，K、V 来自源序列。

三个投影不是冗余复制：Q 决定当前位置想检索什么，K 决定每个位置如何被匹配，V 决定被聚合的内容。

### 16.2 点积分数的概率尺度

对一个 query $q$ 与 key $k$，未缩放分数为 $s=q^Tk=\sum_{r=1}^{d_k}q_rk_r$。若各分量独立、均值为 0、方差为 1，则：

$$E[s]=0,\qquad Var(s)=\sum_{r=1}^{d_k}Var(q_rk_r)=d_k$$

因此标准差随 $\sqrt{d_k}$ 增长。使用 $s/\sqrt{d_k}$ 后方差回到 1，使 softmax logits 在初始化阶段保持稳定尺度。

若 Q、K 分量方差分别为 $\sigma_q^2,\sigma_k^2$，则缩放后点积方差为 $\sigma_q^2\sigma_k^2$；所以初始化与归一化仍然重要。

### 16.3 Softmax 的 Jacobian

令 $a_i=e^{s_i}/\sum_j e^{s_j}$，则：

$$\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j)$$

当某个 $a_i\to1$ 时，各项梯度趋近 0；未缩放的大方差 logits 会更快进入该饱和区。缩放既是方差控制，也是梯度流控制。

### 16.4 掩码的数学语义

causal mask 定义允许集合 $\mathcal{A}_i=\{j\mid j\le i\}$，掩码后概率为：

$$a_{ij}=\frac{\exp(s_{ij})\mathbf{1}[j\in\mathcal{A}_i]}{\sum_{r\in\mathcal{A}_i}\exp(s_{ir})}$$

实现上通常把不允许位置加 $-\infty$。若某一整行都被屏蔽，softmax 会出现未定义的 $0/0$ 或 NaN；构造 padding mask 时必须保证每个有效 query 至少有一个可见 key。

### 16.5 数值稳定 Softmax

$$softmax(s)_i=\frac{e^{s_i-m}}{\sum_j e^{s_j-m}},\qquad m=\max_j s_j$$

减去行最大值不改变概率，却避免指数上溢。混合精度下应优先使用框架融合实现，不要手写 `exp/sum` 替代稳定内核。

### 16.6 对 V、Q、K 的梯度

记 $S=QK^T/\sqrt{d_k}$、$A=softmax(S)$、$O=AV$，上游梯度为 $G=\partial L/\partial O$，则：

$$\frac{\partial L}{\partial V}=A^TG,\qquad \frac{\partial L}{\partial A}=GV^T$$

对每一行 softmax，$\partial L/\partial S=(diag(a)-aa^T)(\partial L/\partial a)$。再由矩阵乘法得到：

$$\frac{\partial L}{\partial Q}=\frac{\partial L}{\partial S}K/\sqrt{d_k},\qquad \frac{\partial L}{\partial K}=(\frac{\partial L}{\partial S})^TQ/\sqrt{d_k}$$

### 16.7 置换等变性与位置编码

若对 token 使用同一个置换矩阵 $P$，且没有位置编码，则 $Attention(PX)=P Attention(X)$。注意力知道 token 内容，却不知道原始顺序。

绝对位置嵌入、相对位置偏置、RoPE、ALiBi 都是在打破这种顺序不辨识性，但它们把位置信息注入分数或表示的方式不同。

### 16.8 核回归与软检索视角

对固定 query，输出 $o_i=\sum_j a_{ij}v_j$ 是 value 的凸组合；softmax 点积定义了数据依赖核。它可视为可学习的 Nadaraya-Watson 核回归或可微软检索。

### 16.9 复杂度分解

投影 QKV 的复杂度约为 $O(BTd_{model}(2d_k+d_v))$；分数和加权聚合分别为 $O(BT^2d_k)$ 与 $O(BT^2d_v)$。

显式 materialize 注意力矩阵需要 $O(BHT^2)$ 存储。FlashAttention 不改变精确注意力的渐近计算量，而是通过分块与在线 softmax 避免把完整矩阵写回 HBM。

## 17. PyTorch 参考实现与等价性测试

```python
import torch
from torch import nn
from torch.nn import functional as F

def manual_sdpa(q, k, v, *, causal=False, allowed_mask=None):
    """q/k/v: (B, H, T, D); allowed_mask 中 True 表示允许参与注意力。"""
    scale = q.size(-1) ** -0.5
    scores = q @ k.transpose(-2, -1) * scale
    if causal:
        q_len, k_len = q.size(-2), k.size(-2)
        assert q_len == k_len, "本教学版本只处理方形 causal attention"
        allowed = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    if allowed_mask is not None:
        scores = scores.masked_fill(~allowed_mask, float("-inf"))
    probs = scores.softmax(dim=-1)
    return probs @ v, probs

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, *, causal=True):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(y)

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_states, memory_states):
        batch, q_len, width = query_states.shape
        k_len = memory_states.size(1)
        q = self.q_proj(query_states)
        k, v = self.kv_proj(memory_states).chunk(2, dim=-1)
        q = q.view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(batch, q_len, width)
        return self.out_proj(y)

@torch.no_grad()
def demo():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    manual, probs = manual_sdpa(q, k, v, causal=True)
    fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(manual, fused, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))
    module = SelfAttention(d_model=64, n_heads=4).eval()
    assert module(torch.randn(2, 8, 64)).shape == (2, 8, 64)
    cross = CrossAttention(d_model=64, n_heads=4).eval()
    assert cross(torch.randn(2, 5, 64), torch.randn(2, 9, 64)).shape == (2, 5, 64)

if __name__ == "__main__":
    demo()
```

### 17.1 PyTorch API 语义

- `F.scaled_dot_product_attention` 的布尔 `attn_mask` 中 `True` 表示该位置参与注意力；这与常见 `masked_fill(mask, -inf)` 中 `True` 表示屏蔽的约定相反。
- 使用 `is_causal=True` 时不要再传冲突的显式 mask；变长序列需要先确认 mask 的形状与广播语义。
- 该函数会按设备、dtype、shape 与后端设置选择 Flash、memory-efficient 或 math 实现；不能仅凭函数名断言一定使用某个内核。
- dropout 是否生效由 `dropout_p` 决定，不会自动读取模块训练态；模块封装应在 eval 时显式传 `0.0`。
- GQA 的 query heads 必须能被 KV heads 整除，且 K、V head 数相同；支持范围还受后端、设备和张量类型限制。

## 18. 论文谱系与 Self-Attention 对应

### 18.1 Neural Machine Translation by Jointly Learning to Align and Translate

- 论文：**Neural Machine Translation by Jointly Learning to Align and Translate**（Bahdanau et al., 2014）。
- 方法对应：加性注意力与软对齐。
- 工业对应：机器翻译对齐。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.2 Effective Approaches to Attention-based NMT

- 论文：**Effective Approaches to Attention-based NMT**（Luong et al., 2015）。
- 方法对应：点积与乘性注意力。
- 工业对应：高效序列对齐。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.3 Attention Is All You Need

- 论文：**Attention Is All You Need**（Vaswani et al., 2017）。
- 方法对应：缩放点积注意力与多头机制。
- 工业对应：Transformer 基础。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.4 BERT

- 论文：**BERT**（Devlin et al., 2018）。
- 方法对应：无 causal mask 的双向自注意力。
- 工业对应：文本理解与检索。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.5 GPT

- 论文：**GPT**（Radford et al., 2018）。
- 方法对应：causal decoder-only 自注意力。
- 工业对应：自回归生成。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.6 Transformer-XL

- 论文：**Transformer-XL**（Dai et al., 2019）。
- 方法对应：片段级记忆与相对位置。
- 工业对应：长文本建模。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.7 XLNet

- 论文：**XLNet**（Yang et al., 2019）。
- 方法对应：排列语言建模与双流注意力。
- 工业对应：预训练表示。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.8 Sparse Transformer

- 论文：**Sparse Transformer**（Child et al., 2019）。
- 方法对应：稀疏连接降低长序列成本。
- 工业对应：长上下文生成。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.9 Reformer

- 论文：**Reformer**（Kitaev et al., 2020）。
- 方法对应：LSH attention 与可逆层。
- 工业对应：内存受限训练。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.10 Longformer

- 论文：**Longformer**（Beltagy et al., 2020）。
- 方法对应：滑窗局部加全局 token。
- 工业对应：长文档理解。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.11 BigBird

- 论文：**BigBird**（Zaheer et al., 2020）。
- 方法对应：随机、局部、全局稀疏模式。
- 工业对应：长文档 NLP。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.12 Linformer

- 论文：**Linformer**（Wang et al., 2020）。
- 方法对应：序列维低秩投影。
- 工业对应：线性复杂度近似。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.13 Performer

- 论文：**Performer**（Choromanski et al., 2020）。
- 方法对应：随机特征近似 softmax kernel。
- 工业对应：线性注意力。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.14 RoFormer

- 论文：**RoFormer**（Su et al., 2021）。
- 方法对应：旋转位置编码进入 QK 点积。
- 工业对应：现代 LLM 位置表示。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.15 ALiBi

- 论文：**ALiBi**（Press et al., 2021）。
- 方法对应：按相对距离加入线性偏置。
- 工业对应：长度外推。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.16 Talking-Heads Attention

- 论文：**Talking-Heads Attention**（Shazeer et al., 2020）。
- 方法对应：跨 head 混合 logits 与权重。
- 工业对应：注意力表达增强。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.17 Multi-Query Attention

- 论文：**Multi-Query Attention**（Shazeer, 2019）。
- 方法对应：多 Q head 共享单组 K/V。
- 工业对应：降低解码 KV 带宽。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.18 GQA

- 论文：**GQA**（Ainslie et al., 2023）。
- 方法对应：多组共享 K/V 折中质量与效率。
- 工业对应：LLM 在线推理。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.19 FlashAttention

- 论文：**FlashAttention**（Dao et al., 2022）。
- 方法对应：分块与在线 softmax 的 IO-aware 精确算法。
- 工业对应：训练和长上下文推理。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.20 FlashAttention-2

- 论文：**FlashAttention-2**（Dao, 2023）。
- 方法对应：减少非矩阵运算并改善并行划分。
- 工业对应：GPU 吞吐优化。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.21 FlashAttention-3

- 论文：**FlashAttention-3**（Shah et al., 2024）。
- 方法对应：面向 Hopper 的异步与低精度优化。
- 工业对应：H100 训练推理。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.22 PagedAttention

- 论文：**PagedAttention**（Kwon et al., 2023）。
- 方法对应：分页管理 KV Cache。
- 工业对应：连续批处理服务。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.23 Mistral 7B

- 论文：**Mistral 7B**（Jiang et al., 2023）。
- 方法对应：GQA 与滑动窗口注意力。
- 工业对应：高效开放 LLM。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.24 LLaMA

- 论文：**LLaMA**（Touvron et al., 2023）。
- 方法对应：RoPE、Pre-Norm 与 causal attention。
- 工业对应：基础模型。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.25 Grouped-Query Attention

- 论文：**Grouped-Query Attention**（Ainslie et al., 2023）。
- 方法对应：从 MHA 上采样/转换到 GQA。
- 工业对应：低成本模型改造。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.26 DeepSeek-V2

- 论文：**DeepSeek-V2**（DeepSeek-AI, 2024）。
- 方法对应：MLA 压缩 KV 表示。
- 工业对应：高吞吐 LLM 服务。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.27 Ring Attention

- 论文：**Ring Attention**（Liu et al., 2023）。
- 方法对应：跨设备环形传递 KV 块。
- 工业对应：超长上下文训练。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.28 StreamingLLM

- 论文：**StreamingLLM**（Xiao et al., 2023）。
- 方法对应：attention sink 与滑动缓存。
- 工业对应：流式长文本生成。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.29 Attention Sinks in Transformers

- 论文：**Attention Sinks in Transformers**（相关后续研究）。
- 方法对应：分析前部 token 吸收注意力质量。
- 工业对应：稳定流式推理。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

### 18.30 Transformers are RNNs

- 论文：**Transformers are RNNs**（Katharopoulos et al., 2020）。
- 方法对应：核技巧重排线性注意力。
- 工业对应：长序列近似。
- 阅读动作：核对注意力连接模式、复杂度、位置方案、训练设置和消融。
- 复现动作：先与稠密 FP32 基线对齐，再比较显存、吞吐和误差。
- 判断边界：近似或稀疏方法的渐近优势不等于短序列上的真实收益。

## 19. 工业案例拆解

### 19.1 案例 01

- 业务场景：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.2 案例 02

- 业务场景：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.3 案例 03

- 业务场景：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.4 案例 04

- 业务场景：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.5 案例 05

- 业务场景：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.6 案例 06

- 业务场景：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.7 案例 07

- 业务场景：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.8 案例 08

- 业务场景：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.9 案例 09

- 业务场景：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.10 案例 10

- 业务场景：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.11 案例 11

- 业务场景：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.12 案例 12

- 业务场景：视频理解：时空 token 数巨大，分解空间与时间 attention。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.13 案例 13

- 业务场景：语音识别：流式 chunk attention 控制右上下文与实时率。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.14 案例 14

- 业务场景：时间序列预测：causal mask、滚动验证与未来信息泄漏检查。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.15 案例 15

- 业务场景：多模态模型：图像 token 与文本 token 的 cross/self attention 布局。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.16 案例 16

- 业务场景：长文档问答：稀疏/滑窗 attention 与证据召回率联合优化。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.17 案例 17

- 业务场景：批量离线生成：大 batch prefill 提升算力利用，解码做动态调度。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.18 案例 18

- 业务场景：高并发 API：Paged KV、请求抢占、前缀缓存和租户隔离。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.19 案例 19

- 业务场景：安全审核模型：双向 attention 分类并关注长尾违规样本召回。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

### 19.20 案例 20

- 业务场景：模型可解释分析：attention map 只能作相关信号，不能直接当因果解释。
- 数据约束：定义时间边界、padding、截断、特殊 token 和隐私隔离。
- 模型约束：明确双向、causal、cross、滑窗或稀疏连接模式。
- 数值验收：FP32 基线与融合、AMP、量化结果逐层和端到端对齐。
- 性能验收：分开报告 prefill 与 decode 的吞吐、median、p95、p99 和显存。
- 质量验收：离线指标、人工评测、失败样本与在线业务指标共同决策。
- 上线策略：灰度、限流、监控、告警、缓存隔离、版本记录和快速回滚。

## 20. Self-Attention 专题知识卡

### 20.1 Q 的检索语义

- 一句话定义：围绕 **Q 的检索语义** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：Neural Machine Translation by Jointly Learning to Align and Translate。
- 工业映射：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.2 K 的索引语义

- 一句话定义：围绕 **K 的索引语义** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：Effective Approaches to Attention-based NMT。
- 工业映射：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.3 V 的内容语义

- 一句话定义：围绕 **V 的内容语义** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：Attention Is All You Need。
- 工业映射：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.4 QKV 联合投影

- 一句话定义：围绕 **QKV 联合投影** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：BERT。
- 工业映射：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.5 点积相似度

- 一句话定义：围绕 **点积相似度** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：GPT。
- 工业映射：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.6 sqrt(d_k) 缩放

- 一句话定义：围绕 **sqrt(d_k) 缩放** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：Transformer-XL。
- 工业映射：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.7 Softmax Jacobian

- 一句话定义：围绕 **Softmax Jacobian** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：XLNet。
- 工业映射：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.8 温度参数

- 一句话定义：围绕 **温度参数** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：Sparse Transformer。
- 工业映射：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.9 数值稳定 Softmax

- 一句话定义：围绕 **数值稳定 Softmax** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：Reformer。
- 工业映射：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.10 Causal Mask

- 一句话定义：围绕 **Causal Mask** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：Longformer。
- 工业映射：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.11 Padding Mask

- 一句话定义：围绕 **Padding Mask** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：BigBird。
- 工业映射：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.12 Additive Mask

- 一句话定义：围绕 **Additive Mask** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：Linformer。
- 工业映射：视频理解：时空 token 数巨大，分解空间与时间 attention。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.13 Boolean Mask

- 一句话定义：围绕 **Boolean Mask** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：Performer。
- 工业映射：语音识别：流式 chunk attention 控制右上下文与实时率。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.14 全屏蔽行

- 一句话定义：围绕 **全屏蔽行** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：RoFormer。
- 工业映射：时间序列预测：causal mask、滚动验证与未来信息泄漏检查。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.15 Attention Dropout

- 一句话定义：围绕 **Attention Dropout** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：ALiBi。
- 工业映射：多模态模型：图像 token 与文本 token 的 cross/self attention 布局。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.16 输出投影

- 一句话定义：围绕 **输出投影** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：Talking-Heads Attention。
- 工业映射：长文档问答：稀疏/滑窗 attention 与证据召回率联合优化。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.17 单头注意力

- 一句话定义：围绕 **单头注意力** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：Multi-Query Attention。
- 工业映射：批量离线生成：大 batch prefill 提升算力利用，解码做动态调度。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.18 多头注意力

- 一句话定义：围绕 **多头注意力** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：GQA。
- 工业映射：高并发 API：Paged KV、请求抢占、前缀缓存和租户隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.19 MQA

- 一句话定义：围绕 **MQA** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：FlashAttention。
- 工业映射：安全审核模型：双向 attention 分类并关注长尾违规样本召回。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.20 GQA

- 一句话定义：围绕 **GQA** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：FlashAttention-2。
- 工业映射：模型可解释分析：attention map 只能作相关信号，不能直接当因果解释。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.21 Head Dimension

- 一句话定义：围绕 **Head Dimension** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：FlashAttention-3。
- 工业映射：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.22 Head 拼接

- 一句话定义：围绕 **Head 拼接** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：PagedAttention。
- 工业映射：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.23 Head 冗余

- 一句话定义：围绕 **Head 冗余** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：Mistral 7B。
- 工业映射：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.24 Head Pruning

- 一句话定义：围绕 **Head Pruning** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：LLaMA。
- 工业映射：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.25 Self-Attention

- 一句话定义：围绕 **Self-Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：Grouped-Query Attention。
- 工业映射：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.26 Cross-Attention

- 一句话定义：围绕 **Cross-Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：DeepSeek-V2。
- 工业映射：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.27 Encoder Attention

- 一句话定义：围绕 **Encoder Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：Ring Attention。
- 工业映射：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.28 Decoder Attention

- 一句话定义：围绕 **Decoder Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：StreamingLLM。
- 工业映射：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.29 双向可见性

- 一句话定义：围绕 **双向可见性** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：Attention Sinks in Transformers。
- 工业映射：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.30 因果可见性

- 一句话定义：围绕 **因果可见性** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：Transformers are RNNs。
- 工业映射：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.31 前缀语言模型

- 一句话定义：围绕 **前缀语言模型** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：Neural Machine Translation by Jointly Learning to Align and Translate。
- 工业映射：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.32 Packed Sequence Mask

- 一句话定义：围绕 **Packed Sequence Mask** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：Effective Approaches to Attention-based NMT。
- 工业映射：视频理解：时空 token 数巨大，分解空间与时间 attention。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.33 绝对位置嵌入

- 一句话定义：围绕 **绝对位置嵌入** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：Attention Is All You Need。
- 工业映射：语音识别：流式 chunk attention 控制右上下文与实时率。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.34 相对位置偏置

- 一句话定义：围绕 **相对位置偏置** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：BERT。
- 工业映射：时间序列预测：causal mask、滚动验证与未来信息泄漏检查。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.35 RoPE

- 一句话定义：围绕 **RoPE** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：GPT。
- 工业映射：多模态模型：图像 token 与文本 token 的 cross/self attention 布局。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.36 ALiBi

- 一句话定义：围绕 **ALiBi** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：Transformer-XL。
- 工业映射：长文档问答：稀疏/滑窗 attention 与证据召回率联合优化。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.37 位置外推

- 一句话定义：围绕 **位置外推** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：XLNet。
- 工业映射：批量离线生成：大 batch prefill 提升算力利用，解码做动态调度。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.38 置换等变性

- 一句话定义：围绕 **置换等变性** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：Sparse Transformer。
- 工业映射：高并发 API：Paged KV、请求抢占、前缀缓存和租户隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.39 Attention Kernel

- 一句话定义：围绕 **Attention Kernel** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：Reformer。
- 工业映射：安全审核模型：双向 attention 分类并关注长尾违规样本召回。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.40 核回归视角

- 一句话定义：围绕 **核回归视角** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：Longformer。
- 工业映射：模型可解释分析：attention map 只能作相关信号，不能直接当因果解释。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.41 注意力熵

- 一句话定义：围绕 **注意力熵** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：BigBird。
- 工业映射：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.42 Attention Sink

- 一句话定义：围绕 **Attention Sink** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：Linformer。
- 工业映射：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.43 Induction Head

- 一句话定义：围绕 **Induction Head** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：Performer。
- 工业映射：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.44 Retrieval Head

- 一句话定义：围绕 **Retrieval Head** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：RoFormer。
- 工业映射：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.45 QK Normalization

- 一句话定义：围绕 **QK Normalization** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：ALiBi。
- 工业映射：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.46 Logit Capping

- 一句话定义：围绕 **Logit Capping** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：Talking-Heads Attention。
- 工业映射：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.47 初始化尺度

- 一句话定义：围绕 **初始化尺度** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：Multi-Query Attention。
- 工业映射：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.48 残差尺度

- 一句话定义：围绕 **残差尺度** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：GQA。
- 工业映射：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.49 Pre-Norm

- 一句话定义：围绕 **Pre-Norm** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：FlashAttention。
- 工业映射：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.50 Post-Norm

- 一句话定义：围绕 **Post-Norm** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：FlashAttention-2。
- 工业映射：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.51 RMSNorm 前置

- 一句话定义：围绕 **RMSNorm 前置** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：FlashAttention-3。
- 工业映射：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.52 SwiGLU 接续

- 一句话定义：围绕 **SwiGLU 接续** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：PagedAttention。
- 工业映射：视频理解：时空 token 数巨大，分解空间与时间 attention。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.53 KV Cache

- 一句话定义：围绕 **KV Cache** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：Mistral 7B。
- 工业映射：语音识别：流式 chunk attention 控制右上下文与实时率。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.54 Prefill

- 一句话定义：围绕 **Prefill** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：LLaMA。
- 工业映射：时间序列预测：causal mask、滚动验证与未来信息泄漏检查。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.55 Decode

- 一句话定义：围绕 **Decode** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：Grouped-Query Attention。
- 工业映射：多模态模型：图像 token 与文本 token 的 cross/self attention 布局。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.56 Cache 追加

- 一句话定义：围绕 **Cache 追加** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：DeepSeek-V2。
- 工业映射：长文档问答：稀疏/滑窗 attention 与证据召回率联合优化。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.57 Cache 分页

- 一句话定义：围绕 **Cache 分页** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：Ring Attention。
- 工业映射：批量离线生成：大 batch prefill 提升算力利用，解码做动态调度。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.58 Prefix Cache

- 一句话定义：围绕 **Prefix Cache** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：StreamingLLM。
- 工业映射：高并发 API：Paged KV、请求抢占、前缀缓存和租户隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.59 Continuous Batching

- 一句话定义：围绕 **Continuous Batching** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：Attention Sinks in Transformers。
- 工业映射：安全审核模型：双向 attention 分类并关注长尾违规样本召回。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.60 请求抢占

- 一句话定义：围绕 **请求抢占** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：Transformers are RNNs。
- 工业映射：模型可解释分析：attention map 只能作相关信号，不能直接当因果解释。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.61 Sliding Window

- 一句话定义：围绕 **Sliding Window** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：Neural Machine Translation by Jointly Learning to Align and Translate。
- 工业映射：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.62 Local Attention

- 一句话定义：围绕 **Local Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：Effective Approaches to Attention-based NMT。
- 工业映射：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.63 Global Token

- 一句话定义：围绕 **Global Token** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：Attention Is All You Need。
- 工业映射：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.64 Block Sparse

- 一句话定义：围绕 **Block Sparse** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：BERT。
- 工业映射：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.65 Random Sparse

- 一句话定义：围绕 **Random Sparse** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：GPT。
- 工业映射：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.66 Linear Attention

- 一句话定义：围绕 **Linear Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：Transformer-XL。
- 工业映射：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.67 Performer

- 一句话定义：围绕 **Performer** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：XLNet。
- 工业映射：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.68 Linformer

- 一句话定义：围绕 **Linformer** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：Sparse Transformer。
- 工业映射：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.69 Longformer

- 一句话定义：围绕 **Longformer** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：Reformer。
- 工业映射：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.70 BigBird

- 一句话定义：围绕 **BigBird** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：Longformer。
- 工业映射：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.71 Reformer

- 一句话定义：围绕 **Reformer** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：BigBird。
- 工业映射：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.72 Ring Attention

- 一句话定义：围绕 **Ring Attention** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：Linformer。
- 工业映射：视频理解：时空 token 数巨大，分解空间与时间 attention。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.73 FlashAttention

- 一句话定义：围绕 **FlashAttention** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：Performer。
- 工业映射：语音识别：流式 chunk attention 控制右上下文与实时率。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.74 Online Softmax

- 一句话定义：围绕 **Online Softmax** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：RoFormer。
- 工业映射：时间序列预测：causal mask、滚动验证与未来信息泄漏检查。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.75 Tiling

- 一句话定义：围绕 **Tiling** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：ALiBi。
- 工业映射：多模态模型：图像 token 与文本 token 的 cross/self attention 布局。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.76 HBM 访存

- 一句话定义：围绕 **HBM 访存** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：Talking-Heads Attention。
- 工业映射：长文档问答：稀疏/滑窗 attention 与证据召回率联合优化。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.77 算术强度

- 一句话定义：围绕 **算术强度** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：Multi-Query Attention。
- 工业映射：批量离线生成：大 batch prefill 提升算力利用，解码做动态调度。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.78 融合 QKV

- 一句话定义：围绕 **融合 QKV** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：GQA。
- 工业映射：高并发 API：Paged KV、请求抢占、前缀缓存和租户隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.79 张量布局

- 一句话定义：围绕 **张量布局** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：FlashAttention。
- 工业映射：安全审核模型：双向 attention 分类并关注长尾违规样本召回。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.80 Contiguous Copy

- 一句话定义：围绕 **Contiguous Copy** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：FlashAttention-2。
- 工业映射：模型可解释分析：attention map 只能作相关信号，不能直接当因果解释。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.81 AMP

- 一句话定义：围绕 **AMP** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：FlashAttention-3。
- 工业映射：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.82 BF16

- 一句话定义：围绕 **BF16** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：PagedAttention。
- 工业映射：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.83 FP16

- 一句话定义：围绕 **FP16** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：Mistral 7B。
- 工业映射：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.84 TF32

- 一句话定义：围绕 **TF32** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：LLaMA。
- 工业映射：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.85 量化 QKV

- 一句话定义：围绕 **量化 QKV** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：Grouped-Query Attention。
- 工业映射：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.86 量化 KV Cache

- 一句话定义：围绕 **量化 KV Cache** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：DeepSeek-V2。
- 工业映射：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.87 梯度检查点

- 一句话定义：围绕 **梯度检查点** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：Ring Attention。
- 工业映射：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.88 序列并行

- 一句话定义：围绕 **序列并行** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：StreamingLLM。
- 工业映射：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.89 张量并行

- 一句话定义：围绕 **张量并行** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：Attention Sinks in Transformers。
- 工业映射：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.90 上下文并行

- 一句话定义：围绕 **上下文并行** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：Transformers are RNNs。
- 工业映射：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.91 编译优化

- 一句话定义：围绕 **编译优化** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：Neural Machine Translation by Jointly Learning to Align and Translate。
- 工业映射：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.92 CUDA Graph

- 一句话定义：围绕 **CUDA Graph** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：Effective Approaches to Attention-based NMT。
- 工业映射：视频理解：时空 token 数巨大，分解空间与时间 attention。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.93 ONNX 导出

- 一句话定义：围绕 **ONNX 导出** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：Attention Is All You Need。
- 工业映射：语音识别：流式 chunk attention 控制右上下文与实时率。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.94 后端选择

- 一句话定义：围绕 **后端选择** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：BERT。
- 工业映射：时间序列预测：causal mask、滚动验证与未来信息泄漏检查。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.95 确定性复现

- 一句话定义：围绕 **确定性复现** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：GPT。
- 工业映射：多模态模型：图像 token 与文本 token 的 cross/self attention 布局。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.96 Profiler

- 一句话定义：围绕 **Profiler** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：Transformer-XL。
- 工业映射：长文档问答：稀疏/滑窗 attention 与证据召回率联合优化。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.97 形状断言

- 一句话定义：围绕 **形状断言** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：XLNet。
- 工业映射：批量离线生成：大 batch prefill 提升算力利用，解码做动态调度。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.98 梯度断言

- 一句话定义：围绕 **梯度断言** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：Sparse Transformer。
- 工业映射：高并发 API：Paged KV、请求抢占、前缀缓存和租户隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.99 Mask 单测

- 一句话定义：围绕 **Mask 单测** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：Reformer。
- 工业映射：安全审核模型：双向 attention 分类并关注长尾违规样本召回。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.100 因果泄漏测试

- 一句话定义：围绕 **因果泄漏测试** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：Longformer。
- 工业映射：模型可解释分析：attention map 只能作相关信号，不能直接当因果解释。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.101 基线对齐

- 一句话定义：围绕 **基线对齐** 明确数学语义、张量契约和工程边界。
- 核心公式：$A=softmax(S)$。
- PyTorch 映射：`Tensor.view`。
- 论文坐标：BigBird。
- 工业映射：搜索查询编码：双向 attention 编码短查询，关注召回质量与批量吞吐。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.102 长序列压力测试

- 一句话定义：围绕 **长序列压力测试** 明确数学语义、张量契约和工程边界。
- 核心公式：$O=AV$。
- PyTorch 映射：`masked_fill`。
- 论文坐标：Linformer。
- 工业映射：广告序列建模：用户行为因果 attention，严格避免使用未来行为。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.103 尾延迟

- 一句话定义：围绕 **尾延迟** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial V=A^TG$。
- PyTorch 映射：`torch.tril`。
- 论文坐标：Performer。
- 工业映射：推荐长历史：滑动窗口、压缩记忆或分层 attention 控制序列成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.104 显存水位

- 一句话定义：围绕 **显存水位** 明确数学语义、张量契约和工程边界。
- 核心公式：$\partial Q=(\partial S)K/\sqrt d$。
- PyTorch 映射：`torch.softmax`。
- 论文坐标：RoFormer。
- 工业映射：机器翻译：encoder 双向、decoder causal、cross-attention 对齐源句。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.105 Attention 可视化

- 一句话定义：围绕 **Attention 可视化** 明确数学语义、张量契约和工程边界。
- 核心公式：$O(T^2d)$。
- PyTorch 映射：`torch.testing.assert_close`。
- 论文坐标：ALiBi。
- 工业映射：文档摘要：长上下文 attention 与输出事实一致性共同验收。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.106 可解释性边界

- 一句话定义：围绕 **可解释性边界** 明确数学语义、张量契约和工程边界。
- 核心公式：$M=O(HT^2)$。
- PyTorch 映射：`torch.profiler`。
- 论文坐标：Talking-Heads Attention。
- 工业映射：RAG 生成：attention 不能替代检索质量，需验证引用与证据覆盖。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.107 数据泄漏

- 一句话定义：围绕 **数据泄漏** 明确数学语义、张量契约和工程边界。
- 核心公式：$a_i=e^{s_i-m}/\sum_j e^{s_j-m}$。
- PyTorch 映射：`torch.compile`。
- 论文坐标：Multi-Query Attention。
- 工业映射：在线聊天：KV Cache、连续批处理与 p99 解码延迟是核心指标。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.108 在线漂移

- 一句话定义：围绕 **在线漂移** 明确数学语义、张量契约和工程边界。
- 核心公式：$p(x)=\prod_t p(x_t|x_{<t})$。
- PyTorch 映射：`torch.amp.autocast`。
- 论文坐标：GQA。
- 工业映射：代码补全：causal mask 与文件级长上下文，关注首 token 和逐 token 延迟。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.109 多租户隔离

- 一句话定义：围绕 **多租户隔离** 明确数学语义、张量契约和工程边界。
- 核心公式：$Q=XW_Q$。
- PyTorch 映射：`F.scaled_dot_product_attention`。
- 论文坐标：FlashAttention。
- 工业映射：多轮客服：会话截断、系统提示保护、缓存复用与隐私隔离。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.110 缓存回滚

- 一句话定义：围绕 **缓存回滚** 明确数学语义、张量契约和工程边界。
- 核心公式：$K=XW_K$。
- PyTorch 映射：`nn.Linear`。
- 论文坐标：FlashAttention-2。
- 工业映射：端侧小模型：GQA/MQA、INT4 与固定内存预算共同设计。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.111 论文复现

- 一句话定义：围绕 **论文复现** 明确数学语义、张量契约和工程边界。
- 核心公式：$V=XW_V$。
- PyTorch 映射：`Tensor.transpose`。
- 论文坐标：FlashAttention-3。
- 工业映射：视觉 Transformer：patch token 的全局 attention 与高分辨率平方成本。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

### 20.112 工业验收

- 一句话定义：围绕 **工业验收** 明确数学语义、张量契约和工程边界。
- 核心公式：$S=QK^T/\sqrt d$。
- PyTorch 映射：`Tensor.contiguous`。
- 论文坐标：PagedAttention。
- 工业映射：视频理解：时空 token 数巨大，分解空间与时间 attention。
- Shape 检查：记录 `B, H, T_q, T_k, D`，逐步核对 transpose、广播与合头。
- Mask 检查：明确 True 是允许还是屏蔽，并覆盖 causal、padding 与组合场景。
- 数值检查：减最大值、防全屏蔽行、监控 NaN/Inf 与低精度误差。
- 梯度检查：确认 Q、K、V、投影矩阵梯度存在且有限。
- 复杂度检查：区分 QKV 投影、分数矩阵、AV 聚合、缓存与通信成本。
- PyTorch 检查：优先原生融合 SDPA，手写版本只用于理解和对照。
- 训练检查：dropout、checkpoint、AMP 与并行策略必须进入配置记录。
- 推理检查：区分 prefill 和 decode，记录 KV Cache 大小与生命周期。
- 性能检查：目标硬件预热后同步计时，报告吞吐和 median/p95/p99。
- 论文检查：核对连接模式、位置编码、近似误差、数据集与硬件。
- 解释检查：attention 权重是模型内部相关信号，不自动构成因果解释。
- 安全检查：causal 泄漏、padding 泄漏、跨请求缓存串扰必须有测试。
- 上线检查：灰度、监控、告警、版本化、容量保护和回滚路径齐全。
- 面试表达：公式 → 缩放原因 → mask → 复杂度 → PyTorch → 工业瓶颈。

## 21. 手撕、证明与工程练习

## 22. 结论

Self-Attention 的核心不是背一行公式，而是理解 QKV 投影、尺度控制、mask 语义、softmax 数值、梯度路径和二次复杂度如何共同决定模型行为。

生产实现优先使用 PyTorch 原生融合 SDPA；任何长上下文或稀疏近似方案都必须同时与稠密基线比较质量、显存、吞吐和尾延迟。
