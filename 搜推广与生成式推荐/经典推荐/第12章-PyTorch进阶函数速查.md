---
title: PyTorch 常用函数大全
tags:
  - PyTorch
  - 函数速查
  - 搜推广
  - 深度学习
description: 100+ 核心 PyTorch 函数的完整速查手册,涵盖 Tensor/Autograd/分布式/量化/性能分析等
last_updated: 2026-07-14
status: ✅ v1.0
---

# PyTorch 常用函数大全(100+ 核心函数)

> **使用说明**: 本文档是搜推广/PyTorch 工程的速查手册。每个函数给出签名、作用、参数说明、示例和常见陷阱。所有代码均可在 PyTorch ≥ 2.0 环境下直接运行。
>
> **配套章节**: 详细理论推导见 [[第11章-PyTorch基础]] 和 [[第17章-高级工程实践]]

## 目录

- [§1 Tensor 创建与基础](#1-tensor-创建与基础)
- [§2 Tensor 形状变换](#2-tensor-形状变换)
- [§3 索引与选择](#3-索引与选择)
- [§4 归约运算](#4-归约运算)
- [§5 矩阵运算与 linalg](#5-矩阵运算与-linalg)
- [§6 激活函数](#6-激活函数)
- [§7 损失函数](#7-损失函数)
- [§8 Autograd](#8-autograd)
- [§9 nn.Module 基础](#9-nnmodule-基础)
- [§10 优化器与学习率调度](#10-优化器与学习率调度)
- [§11 DataLoader](#11-dataloader)
- [§12 分布式训练](#12-分布式训练)
- [§13 混合精度训练](#13-混合精度训练)
- [§14 性能分析](#14-性能分析)
- [§15 显存管理](#15-显存管理)
- [§16 JIT 编译](#16-jit-编译)
- [§17 量化](#17-量化)
- [§18 ONNX 与部署](#18-onnx-与部署)
- [§19 随机性控制](#19-随机性控制)
- [§20 检查点与序列化](#20-检查点与序列化)
- [§21 易混淆函数对比速查](#21-易混淆函数对比速查)
- [§22 散射 / 索引 / 稀疏更新 API 速查](#22-散射--索引--稀疏更新-api-速查)
- [§23 函数易混淆对比表](#23-函数易混淆对比表)
- [§24 torch.compile 调优与降级实战](#24-torchcompile-调优与降级实战)

---

## 1. Tensor 创建与基础

### torch.tensor

```python
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

**作用**: 从 Python list / NumPy array / 标量创建 Tensor。

**参数**:
- `data` (list/array/scalar): 数据源
- `dtype` (torch.dtype, 默认 None): 强制类型
- `device` (str): `'cpu'` / `'cuda'` / `'cuda:0'`

**示例**:
```python
x = torch.tensor([1, 2, 3])            # int64
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
x = torch.tensor(np_arr, device='cuda')
```

**注意**:
- 频繁创建小 Tensor 会触发大量内存分配,**搜推广大 batch 训练务必复用 buffer**
- `torch.tensor(np_arr)` **共享内存**;若不希望共享,使用 `torch.tensor(np_arr.copy())`

### torch.zeros / torch.ones / torch.empty

```python
torch.zeros(*size, dtype=None, device=None, requires_grad=False)
torch.ones(*size, ...)
torch.empty(*size, ...)  # 不初始化,可能含垃圾值
```

**作用**: 创建全 0/全 1/未初始化的 Tensor。`empty` 比 `zeros` **快约 2 倍**(不写 0)。

**示例**:
```python
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3, dtype=torch.float32)
empty = torch.empty(1024, 64)         # 注意:内容未定义!
```

**注意**:
- `torch.empty` 用于**会被立即覆写**的 buffer(如 `out=` 参数)
- 任何 "被读到"的 `empty` 都会带来未定义行为,务必先 `fill_` 或手动赋值

### torch.full

```python
torch.full(size, fill_value, dtype=None, device=None, requires_grad=False)
```

**作用**: 用常量填充 Tensor。

**示例**:
```python
mask = torch.full((1024, 50), -1e9)  # 注意力 mask 填充
padding = torch.full_like(seq_ids, pad_id)
```

### torch.arange / torch.linspace

```python
torch.arange(start=0, end, step=1, dtype=None, device=None)
torch.linspace(start, end, steps, dtype=None, device=None)
```

**作用**: 创建等差序列。

**示例**:
```python
# RoPE 预计算
inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

# 位置编码
positions = torch.arange(0, max_seq_len).unsqueeze(0)  # [1, T]

# 在 [0, 1] 区间均匀采样 100 个点
x = torch.linspace(0, 1, 100)
```

### torch.eye

```python
torch.eye(n, m=None, dtype=None, device=None, requires_grad=False)
```

**作用**: 创建单位矩阵(或矩形对角矩阵)。

**示例**:
```python
I = torch.eye(d)            # [d, d] 单位矩阵
# RoPE 中构造 [2, 2] 旋转矩阵
rot = torch.eye(2) * cos + torch.tensor([[0, -1], [1, 0]]) * sin
```

### torch.randn / torch.rand / torch.randint

```python
torch.randn(*size, dtype=None, device=None, requires_grad=False)
torch.rand(*size, ...)   # 均匀 [0, 1)
torch.randint(low, high, size, dtype=None, device=None)
```

**作用**: 创建随机 Tensor。

**示例**:
```python
weights = torch.randn(64, 32) * 0.01   # 初始化,小方差
mask = torch.rand(1024, 50) < 0.1     # 10% dropout mask
indices = torch.randint(0, 1000, (B,))  # 随机负采样
```

**注意**:
- 默认 dtype 是 `torch.float32`
- **reproducibility**: 用前必先 `torch.manual_seed(seed)`

### torch.from_numpy

```python
torch.from_numpy(ndarray)
```

**作用**: NumPy 数组 → Tensor(**共享内存**,零拷贝)。

**示例**:
```python
np_arr = np.array([1, 2, 3])
t = torch.from_numpy(np_arr)        # 共享内存
np_arr[0] = 99                       # t[0] 也会变成 99!
```

**注意**:
- `from_numpy` 不支持 GPU;若要 GPU 版的 numpy,使用 `torch.from_numpy(arr).to('cuda')`
- NumPy 必须为 `np.ndarray`,不能是 list

### torch.as_tensor

```python
torch.as_tensor(data, dtype=None, device=None)
```

**作用**: 智能转换(优先 zero-copy,做不到再 copy)。

**示例**:
```python
t = torch.as_tensor(some_list)          # 内部判断是否可 zero-copy
t = torch.as_tensor(tensor_on_gpu)     # 直接返回引用,不 copy
```

### torch.tensor 的零拷贝替代

```python
torch.as_strided(size, stride, storage_offset=0)
```

**作用**: 通过 stride 创建一个零拷贝的视图(view)。

**示例**:
```python
# 滑动窗口零拷贝视图 (避免 O(N×W) 的内存)
x = torch.arange(10).float()                       # [10]
windows = x.as_strided(size=(8, 5), stride=(1, 1))  # [8, 5]
# windows[i] = x[i:i+5]  完全零拷贝!

# 等价但更慢的代码:
windows_loop = torch.stack([x[i:i+5] for i in range(8)])  # 8 倍内存
```

**注意**:
- `as_strided` **不检查边界**!传错 size/stride 会导致非法内存访问
- 用于高级优化(滑动窗口、卷积 im2col);日常**不建议**使用

---

## 2. Tensor 形状变换

### torch.view

```python
tensor.view(*shape)
```

**作用**: 返回**新视图**(共享内存,只改 shape/stride)。

**示例**:
```python
x = torch.randn(4, 6)
y = x.view(2, 12)              # [4, 6] → [2, 12]  共享内存
```

**注意**:
- 要求原 tensor **内存连续**(`is_contiguous() == True`)
- 不连续时报错 `RuntimeError: view size is not compatible with...`

### torch.reshape

```python
tensor.reshape(*shape)
```

**作用**: 同 `view`,**但不要求连续**——不连续时会自动 `.contiguous()`。

**示例**:
```python
x = torch.randn(2, 3, 4)
y = x.permute(2, 0, 1).reshape(8, 3)  # 自动 contiguous
```

**注意**:
- `reshape` 是**安全版本**的 `view`;默认就用 `reshape` 即可

### tensor.transpose / torch.transpose

```python
tensor.transpose(dim0, dim1)
```

**作用**: 交换两个维度。**O(1)**(只改 stride,不复制)。

**示例**:
```python
x = torch.randn(2, 3, 4)            # [B, T, D]
y = x.transpose(1, 2)                # [2, 4, 3] = [B, D, T]
```

**注意**:
- transpose 后**不连续**;`view` 会报错,用 `reshape` 或 `.contiguous()`

### tensor.permute

```python
tensor.permute(*dims)
```

**作用**: 任意重排维度(`transpose` 的超集)。

**示例**:
```python
x = torch.randn(2, 3, 4, 5)         # [B, C, H, W]
y = x.permute(0, 2, 3, 1)            # [B, H, W, C]  通道维放到最后
```

### tensor.squeeze / tensor.unsqueeze

```python
tensor.squeeze(dim=None)   # dim=None: 删所有 size=1 的维
tensor.unsqueeze(dim)        # 在 dim 位置插入 size=1 的维
```

**作用**: 处理 size=1 的维度。

**示例**:
```python
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()                  # [3, 4]  删所有
z = x.unsqueeze(0)               # [1, 1, 3, 1, 4]
# 多头注意力的标准操作
q = q.view(B, T, n_heads, head_dim).transpose(1, 2)  # [B, n_heads, T, head_dim]
```

### tensor.flatten / tensor.unflatten

```python
tensor.flatten(start_dim=0, end_dim=-1)
tensor.unflatten(dim, sizes)
```

**作用**: 展平 / 反展平。

**示例**:
```python
x = torch.randn(2, 3, 4)
flat = x.flatten(start_dim=1)         # [2, 12]
back = flat.unflatten(1, (3, 4))     # [2, 3, 4]
```

### torch.cat / torch.stack

```python
torch.cat(tensors, dim=0, out=None)        # 沿已有维拼接
torch.stack(tensors, dim=0, out=None)      # 新增一个维堆叠
```

**作用**: 拼接/堆叠。

**示例**:
```python
a, b = torch.randn(2, 5), torch.randn(3, 5)
c = torch.cat([a, b], dim=0)              # [5, 5]

x, y = torch.randn(2, 3), torch.randn(2, 3)
z = torch.stack([x, y], dim=1)             # [2, 2, 3]  多了一维
```

**注意**:
- `cat` 要求除拼接维外**其他维 size 一致**
- `stack` 要求**所有维 size 一致**;本质是 `cat + unsqueeze`

### torch.split / torch.chunk

```python
torch.split(tensor, split_size_or_sections, dim=0)
torch.chunk(tensor, chunks, dim=0)
```

**作用**: 把 tensor 切成块。

**示例**:
```python
x = torch.randn(10, 64)
parts = torch.split(x, 3, dim=0)        # [3,3,3,1] 各部分
chunks = torch.chunk(x, 3, dim=0)        # 平均 3 份(最后一份可能小)

# 多头 QKV 合并投影分拆
qkv = self.W_qkv(x).view(B, T, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)                  # 各 [B, n_heads, T, head_dim]
```

### torch.unbind

```python
torch.unbind(tensor, dim=0)
```

**作用**: 移除指定维,返回 tuple。

**示例**:
```python
x = torch.randn(3, 4, 5)
a, b, c = torch.unbind(x, dim=0)         # 各 [4, 5]
```

### tensor.expand / tensor.repeat

```python
tensor.expand(*sizes)    # 零拷贝广播(只允许 size=1 的 dim 扩)
tensor.repeat(*sizes)    # 实际复制数据
```

**作用**: 维度扩展。

**示例**:
```python
x = torch.randn(1, 3, 1)               # [1, 3, 1]
y = x.expand(2, 3, 4)                  # [2, 3, 4] 零拷贝!
z = x.repeat(2, 1, 4)                  # [2, 3, 4] 实际复制

# GQA 中的 K/V 头广播
K = K.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)  # 零拷贝
```

**注意**:
- `expand` 的输出**与原 tensor 共享内存**,只允许 size=1 的维扩
- `repeat` 会**真复制**,占更多内存

### tensor.contiguous

```python
tensor.contiguous(memory_format=torch.contiguous_format)
```

**作用**: 返回连续内存的 tensor(若已连续则直接返回自身)。

**示例**:
```python
x = torch.randn(2, 3).transpose(0, 1)  # 不连续
y = x.contiguous()                       # 复制一份连续的
z = y.view(6)                            # 现在可以 view
```

**注意**:
- `contiguous()` **涉及 O(n) 内存拷贝**,在循环中调用是性能杀手
- 训练循环内若需要 `contiguous`,应**在循环外**做一次

### torch.broadcast_to

```python
torch.broadcast_to(tensor, size)
```

**作用**: 显式广播到目标 shape(零拷贝)。

**示例**:
```python
x = torch.tensor([[1], [2], [3]])    # [3, 1]
y = torch.broadcast_to(x, (3, 4))     # [3, 4] 零拷贝
# 等价: y = x.expand(3, 4)
```

**注意**:
- 与 `expand` 几乎相同,只是 `expand` 是方法、`broadcast_to` 是函数
- 实际工程中 `.unsqueeze + .expand` 更常用

---

## 3. 索引与选择

### 基本索引

```python
tensor[i, j, k]            # 单元素
tensor[i]                   # 第 i 个 slice
tensor[i:j]                 # 切片
tensor[:, j]                # 所有行的第 j 列
tensor[..., k]              # 等价于 :, :, k
```

**注意**:
- **基本索引返回视图**(`view`),不复制数据
- 区别于**高级索引**(`tensor[[0, 2], [1, 3]]`),后者**复制**数据

### 布尔索引

```python
tensor[mask]                # 1D 输出,只保留 mask=True 位置
```

**示例**:
```python
x = torch.arange(10)
mask = x > 5
y = x[mask]                 # tensor([6, 7, 8, 9])

# 搜推广: 过滤 padding
seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
mask = seq != 0              # [B, T]
valid = seq[mask]            # 一维 tensor,包含所有非零 ID
```

**注意**:
- 输出**丢失原始 shape**,变成 1D
- 搜推广里**慎用**——通常需要保留 shape,用 `masked_fill` / `masked_select` 更合适

### torch.gather

```python
torch.gather(input, dim, index, *, sparse_grad=False)
```

**作用**: 沿 dim 按 index 取出对应位置的值。

**示例**:
```python
# 场景:从 logits [B, V] 取出 top-k 位置的分数
logits = torch.randn(B, V)
topk_idx = logits.topk(K, dim=-1).indices              # [B, K]
topk_scores = logits.gather(dim=1, index=topk_idx)     # [B, K]

# 场景:取序列每个位置的特定 ID
seq = torch.tensor([[10, 20, 30], [40, 50, 60]])    # [B, T]
idx = torch.tensor([[0, 2], [1, 0]])                  # [B, 2]
selected = seq.gather(dim=1, index=idx)               # [10, 30], [50, 40]
```

**注意**:
- `index` 的 shape 必须**与输出相同**;`dim` 上 size 一致
- 与 `tensor[index]` 高级索引**不同**——`gather` 沿指定维取

### torch.index_select

```python
torch.index_select(input, dim, index)
```

**作用**: 沿 dim 取出 index 指定的 slice(类似 `tensor[index]` 但显式指定维)。

**示例**:
```python
# 从 Embedding 表中选若干行
emb = nn.Embedding(1000, 64)
selected_ids = torch.tensor([0, 100, 999, 500])
selected = emb.weight.index_select(dim=0, index=selected_ids)  # [4, 64]
# 等价于 selected = emb.weight[selected_ids] 但语义更明确
```

### torch.masked_select

```python
torch.masked_select(input, mask, *, out=None)
```

**作用**: 取出 mask=True 位置的所有元素(1D 输出)。

**示例**:
```python
scores = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])
mask = scores > 0.3
high = scores.masked_select(mask)         # tensor([0.5, 0.8])
```

**注意**:
- 输出**总是 1D**
- 区别于布尔索引 `x[mask]`,但结果相同

### tensor.masked_fill / tensor.masked_fill_

```python
tensor.masked_fill(mask, value)
tensor.masked_fill_(mask, value)     # 原地操作
```

**作用**: 把 mask=True 位置填成 value(常用 -inf 屏蔽 attention)。

**示例**:
```python
# 注意力 mask
scores = torch.randn(B, n_heads, T, T)
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float('-inf'))

# Padding mask
seq_mask = (seq == pad_id)               # True = padding
scores = scores.masked_fill(seq_mask.unsqueeze(1).unsqueeze(2), -1e9)
```

**注意**:
- `mask` 与 `tensor` 形状必须**可广播**
- `_` 后缀是原地操作,会修改原 tensor

### torch.where

```python
torch.where(condition, input, other)
```

**作用**: 按 condition 选择 input 或 other。

**示例**:
```python
# ReLU 替代
y = torch.where(x > 0, x, torch.zeros_like(x))

# Padding 位置填 -1e9
attn = torch.where(pad_mask.unsqueeze(1), torch.full_like(scores, -1e9), scores)
```

### torch.nonzero / torch.argwhere

```python
torch.nonzero(input, *, as_tuple=False)
```

**作用**: 返回非零元素的索引。

**示例**:
```python
x = torch.tensor([1, 0, 2, 0, 3])
idx = torch.nonzero(x)                # [[0], [2], [4]]
# 等价: idx = torch.argwhere(x)
```

---

## 4. 归约运算

### tensor.sum / tensor.mean

```python
tensor.sum(dim=None, keepdim=False, dtype=None)
tensor.mean(dim=None, keepdim=False)
```

**作用**: 求和/均值。

**示例**:
```python
x = torch.randn(B, T, D)
x.sum(dim=1)              # [B, D] 沿 T 求和
x.mean(dim=-1)            # [B, T] 沿 D 求均值

# 搜推广:变长序列的 masked mean
mask = (seq != 0).unsqueeze(-1).float()  # [B, T, 1]
pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
```

### tensor.max / tensor.min / tensor.argmax / tensor.argmin

```python
tensor.max(dim=None, keepdim=False)       # 返回 (values, indices)
tensor.argmax(dim=None, keepdim=False)    # 只返回 indices
```

**作用**: 求最大/最小值及其位置。

**示例**:
```python
logits = torch.randn(B, V)             # [B, V]
pred_class = logits.argmax(dim=-1)      # [B]  预测类

# top-k 的 logits
topk_scores, topk_indices = logits.topk(K, dim=-1)

# 序列中每行的最大值
x = torch.randn(B, T)
max_per_seq, max_idx = x.max(dim=1)    # 各 [B]
```

### tensor.topk / tensor.kthvalue

```python
tensor.topk(k, dim=-1, largest=True, sorted=True)
```

**作用**: 取 top-k 大的元素。

**示例**:
```python
logits = torch.randn(B, V)
top10_scores, top10_ids = logits.topk(10, dim=-1)  # 各 [B, 10]
```

### torch.unique

```python
torch.unique(input, sorted=True, return_inverse=False, return_counts=False)
```

**作用**: 去重。

**示例**:
```python
item_ids = torch.tensor([1, 2, 2, 3, 1, 4])
unique_ids = torch.unique(item_ids)               # [1, 2, 3, 4]
unique_ids, counts = torch.unique(item_ids, return_counts=True)
# ([1,2,3,4], [2,2,1,1])
```

### torch.bincount

```python
torch.bincount(input, weights=None, minlength=0)
```

**作用**: 统计非负整数中每个值出现次数(类似直方图)。

**示例**:
```python
ids = torch.tensor([0, 1, 2, 0, 1, 0, 3])
counts = torch.bincount(ids)          # [3, 2, 1, 1]
# 搜推广:统计各物品的曝光次数,用于计算负采样权重
```

### tensor.cumsum / tensor.cumprod

```python
tensor.cumsum(dim=0)
tensor.cumprod(dim=0)
```

**作用**: 累积求和/累积乘积。

**示例**:
```python
# EmbeddingBag 的 offsets
lengths = torch.tensor([3, 5, 2, 4])
offsets = torch.cat([torch.tensor([0]), lengths.cumsum(0)[:-1]])
# offsets: [0, 3, 8, 10]
```

### tensor.diff

```python
tensor.diff(n=1, dim=-1)
```

**作用**: 沿 dim 计算相邻差分。

**示例**:
```python
timestamps = torch.tensor([100, 150, 200, 250, 300])
gaps = torch.diff(timestamps)         # [50, 50, 50, 50]
# 搜推广:计算用户行为的时间间隔
```

### torch.std / torch.var

```python
torch.std(input, dim=None, unbiased=True, keepdim=False)
torch.var(input, dim=None, unbiased=True)
```

**作用**: 标准差/方差。`unbiased=True` 用 N-1(默认),`False` 用 N。

**示例**:
```python
x = torch.randn(1000)
std = x.std(unbiased=True)            # 样本标准差
```

### torch.norm

```python
torch.norm(input, p=2, dim=None, keepdim=False)
```

**作用**: 计算 L^p 范数。

**示例**:
```python
x = torch.randn(10, 64)
norms = x.norm(p=2, dim=-1)          # [10]  每行的 L2 范数
# L2 归一化
x_normalized = x / x.norm(p=2, dim=-1, keepdim=True)
# 等价: F.normalize(x, p=2, dim=-1)
```

---

## 5. 矩阵运算与 linalg

### torch.matmul

```python
torch.matmul(input, other, *, out=None)
```

**作用**: 通用矩阵乘法(支持广播、batch、稀疏)。

**示例**:
```python
# 2D matmul
y = torch.matmul(x, w)              # [B, D_in] @ [D_in, D_out]

# batched matmul
q = torch.randn(B, n_heads, T, d_k)
k = torch.randn(B, n_heads, T, d_k)
attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # [B, n_heads, T, T]
```

### torch.mm / torch.bmm

```python
torch.mm(input, mat)         # 严格 2D
torch.bmm(input, mat2)        # 严格 3D
```

**注意**:
- `mm` 拒绝 3D 输入,会报错
- `bmm` 拒绝非 3D 输入
- 通用场景**直接用 `matmul`** 更省心

### torch.einsum

```python
torch.einsum(equation, *operands)
```

**作用**: 爱因斯坦求和约定,用字符串描述复杂的乘积/收缩/广播。

**示例**:
```python
# 双塔内积
score = torch.einsum('bd,bd->b', user_emb, item_emb)

# 注意力 Q K^T
attn = torch.einsum('bntd,bnsd->bnts', Q, K)

# 带偏置的线性
y = torch.einsum('bd,dh->bh', x, w) + b
```

**注意**:
- **可读性极强**,但首次接触需要适应
- 在 LLM 时代已成主流(替换大量 bmm+permute+reshape)

### torch.dot / torch.cross

```python
torch.dot(input, other)            # 1D 点积
torch.cross(input, other, dim=-1)  # 3D 叉积
```

### torch.t / tensor.T

```python
tensor.t()                # 等价于 tensor.transpose(0, 1),仅对 2D
tensor.T                  # 反转所有维(慎用)
```

### torch.linalg 模块

```python
torch.linalg.svd(A, full_matrices=True)            # 奇异值分解
torch.linalg.qr(A, mode='reduced')                  # QR 分解
torch.linalg.cholesky(A)                          # Cholesky 分解(对称正定)
torch.linalg.inv(A)                                # 矩阵求逆(慢,优先用 solve)
torch.linalg.solve(A, B)                          # 求解 AX = B(优先用此)
torch.linalg.eigvals(A)                           # 特征值
torch.linalg.eigh(A)                              # 对称阵的特征值分解
torch.linalg.matrix_norm(A, ord='fro')            # Frobenius 范数
torch.linalg.vector_norm(x, ord=2)                # 向量 L2 范数(等同 torch.norm)
torch.linalg.det(A)                                # 行列式
torch.linalg.matrix_rank(A)                       # 矩阵秩
torch.linalg.pinv(A)                              # 伪逆
torch.linalg.slogdet(A)                           # 符号 + log 行列式(数值稳定)
```

**示例**:
```python
# SVD 推荐: 用于 Embedding 表的低秩近似
emb = nn.Embedding(10000, 64).weight  # [10000, 64]
U, S, V = torch.linalg.svd(emb, full_matrices=False)  # U: [10000, 64], S: [64]
# 取 top-r 秩
r = 32
emb_lowrank = U[:, :r] @ torch.diag(S[:r]) @ V[:r, :]  # [10000, 64] 低秩近似
```

**注意**:
- 优先用 `torch.linalg.solve(A, B)` 而非 `torch.linalg.inv(A) @ B`,数值更稳定、速度更快
- 矩阵求逆是 **O(n³)**,且对病态矩阵不稳定;实际工程**几乎不用**

### tensor.trace / torch.trace

```python
torch.trace(input)        # 对角线元素之和
```

**示例**:
```python
A = torch.eye(3)
torch.trace(A)              # 3.0
```

---

## 6. 激活函数

### torch.relu / F.relu

```python
F.relu(input, inplace=False)
```

**作用**: ReLU(x) = max(0, x)。

**示例**:
```python
x = torch.randn(5)
y = F.relu(x)
```

**注意**:
- `inplace=True` 会修改 input,**可能导致 autograd 报错**(梯度计算时 input 已被覆盖)
- 推荐用 `inplace=False`(默认)

### F.leaky_relu / F.prelu

```python
F.leaky_relu(input, negative_slope=0.01, inplace=False)
F.prelu(input, weight)        # weight 是可学习参数
```

**作用**: 负值有小斜率,避免 Dead ReLU。

### F.gelu

```python
F.gelu(input, approximate='none')     # 'none' 精确,'tanh' 近似
```

**作用**: $x \cdot \Phi(x)$,BERT/GPT 默认。

**示例**:
```python
x = torch.tensor([1.0, -1.0, 0.0])
y = F.gelu(x)         # 近似 [0.841, -0.159, 0]
```

### F.silu / F.silu

```python
F.silu(input, inplace=False)       # 即 Swish
```

**作用**: $x \cdot \sigma(x)$。SwiGLU 的基础。

### F.mish

```python
F.mish(input, inplace=False)
```

**作用**: $x \cdot \tanh(\text{softplus}(x))$。某些视觉模型用。

### F.softplus

```python
F.softplus(input, beta=1, threshold=20)
```

**作用**: $\log(1 + e^x)$,ReLU 的平滑版。

### F.sigmoid

```python
torch.sigmoid(input)         # 也可用 F.sigmoid
```

**作用**: $\sigma(x) = 1/(1+e^x)$。

**注意**:
- 推荐**直接用 `F.binary_cross_entropy_with_logits`**,数值更稳定

### F.tanh

```python
torch.tanh(input)
```

### F.softmax / F.log_softmax

```python
F.softmax(input, dim)
F.log_softmax(input, dim)        # 等价于 log(softmax),数值稳定
```

**作用**: 沿 dim 归一化到概率分布。

**注意**:
- 搜推广的 next-item 预测**必用 log_softmax 或 cross_entropy**;**永远不要**自己 `log(softmax(x))`

### F.dropout

```python
F.dropout(input, p=0.5, training=True, inplace=False)
```

**作用**: 训练时随机置零(乘以 1/(1-p) 保持期望)。

**注意**:
- 推荐用 `nn.Dropout`(模块化);`F.dropout` 较少用
- 推理时**必须**设 `training=False`

---

## 7. 损失函数

### F.binary_cross_entropy_with_logits

```python
F.binary_cross_entropy_with_logits(input, target, pos_weight=None, reduction='mean')
```

**作用**: CTR 二分类标准损失。**内部做 sigmoid + BCE**,数值稳定。

**示例**:
```python
logits = model(user_features)         # [B]
labels = torch.randint(0, 2, (B,))
loss = F.binary_cross_entropy_with_logits(logits, labels.float())

# 正负样本不平衡
pos_weight = torch.tensor([99.0])     # 负/正比
loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
```

### F.cross_entropy

```python
F.cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean')
```

**作用**: 多分类标准损失 = `log_softmax + nll_loss`。

**示例**:
```python
logits = model(features)              # [B, V]
target = torch.randint(0, V, (B,))
loss = F.cross_entropy(logits, target)

# next-item 预测 + 忽略 padding
loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), ignore_index=0)
```

### F.mse_loss / F.l1_loss

```python
F.mse_loss(input, target, reduction='mean')
F.l1_loss(input, target)
```

**作用**: 均方误差/平均绝对值误差,用于回归任务。

### F.smooth_l1_loss / F.huber_loss

```python
F.smooth_l1_loss(input, target, beta=1.0, reduction='mean')
F.huber_loss(input, target, delta=1.0, reduction='mean')
```

**作用**: 在 $|x| < \delta$ 时用 L2,否则用 L1,对异常值鲁棒。

### F.triplet_margin_loss

```python
F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2)
```

**作用**: 三元组损失,让 anchor 离 positive 近、离 negative 远。

### F.margin_ranking_loss

```python
F.margin_ranking_loss(input1, input2, target, margin=0.0)
```

**作用**: BPR 排序损失。

### F.kl_div

```python
F.kl_div(input, target, reduction='batchmean', log_target=False)
```

**作用**: KL 散度,**注意 input 是 log 概率**!

**示例**:
```python
# 蒸馏
log_student = F.log_softmax(student_logits, dim=-1)
prob_teacher = F.softmax(teacher_logits / T, dim=-1)
loss = F.kl_div(log_student, prob_teacher, reduction='batchmean') * (T ** 2)
```

**注意**:
- input **必须是 log 概率**(不是 logits,也不是 softmax!)
- `log_target=False` 时 target 必须是普通概率

### F.cosine_embedding_loss

```python
F.cosine_embedding_loss(input1, input2, target, margin=0, reduction='mean')
```

**作用**: 余弦相似度损失。`target=1` 让相似度趋近 1,`target=-1` 让其趋近 -1。

---

## 8. Autograd

### torch.no_grad / torch.inference_mode

```python
with torch.no_grad():
    output = model(x)

with torch.inference_mode():     # 更快,更严格
    output = model(x)
```

**作用**: 关闭梯度计算。`inference_mode` 比 `no_grad` 再快约 10%。

**注意**:
- 验证/推理时**必须**用,否则会构建计算图,显存爆炸
- `inference_mode` 内的 tensor 不能拿出来再做需要梯度的运算

### torch.enable_grad

```python
with torch.enable_grad():
    x = compute()       # 在 @torch.no_grad 上下文中临时开启梯度
```

**场景**: 模型前向时用 `no_grad` 提效,但需要在局部子模块中计算梯度。

### torch.autograd.grad

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False)
```

**作用**: 计算 outputs 对 inputs 的梯度,返回 tuple 而**不累加到 .grad**。

**示例**:
```python
x = torch.randn(4, 8, requires_grad=True)
y = x.sum()
g = torch.autograd.grad(y, x)       # g 是 (tensor,) 形状同 x
g = torch.autograd.grad(y, x, create_graph=True)[0]  # 允许二阶求导
```

### torch.autograd.backward

```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None)
```

**作用**: 沿计算图反向传播(等价于 `tensor.backward()`)。

### torch.autograd.functional

```python
torch.autograd.functional.jacobian(func, inputs, create_graph=False)
torch.autograd.functional.hessian(func, inputs, create_graph=False)
torch.autograd.functional.vjp(func, inputs, v)         # 向量-雅可比积
torch.autograd.functional.jvp(func, inputs, v)         # 雅可比-向量积
torch.autograd.functional.vmap(func, in_dims)          # 自动批处理
```

**作用**: 函数式 API,显式计算 Jacobian / Hessian / VJP / JVP。

**示例**:
```python
def f(x):
    return x ** 2

x = torch.randn(3, requires_grad=True)
J = torch.autograd.functional.jacobian(f, x)        # [3, 3] 雅可比矩阵

# vmap: 自动将 f 应用到 batch
def per_sample_loss(params, data):
    return (params * data).sum()
batched_loss = torch.func.vmap(per_sample_loss, in_dims=(None, 0))
```

**注意**:
- 搜推广里**很少用**,但元学习、second-order 优化需要
- `vmap` 是 PyTorch 2.0+ 的 JAX-style 转换,比手写 batch 循环快

### tensor.requires_grad_ / tensor.detach / tensor.grad

```python
tensor.requires_grad_(True)         # 原地开启梯度
tensor.detach()                     # 切断计算图(返回新 tensor)
tensor.grad                         # 已累积的梯度
tensor.grad_fn                      # 产生该 tensor 的 grad_fn
```

**注意**:
- `tensor.detach()` 是**关键**:在蒸馏、特征提取、teacher-student 中**必须**用
- 不 detach 会导致**梯度流到不需要更新的模块**

### tensor.register_hook

```python
tensor.register_hook(hook_fn)        # 反向时调用 hook(可读/修改梯度)
```

**示例**:
```python
# 打印特定层的梯度
def print_grad(grad):
    print(f"grad norm: {grad.norm().item():.4f}")
    return grad
emb_output.register_hook(print_grad)

# 裁剪特定层的梯度
def clip_grad(grad):
    return torch.clamp(grad, -0.1, 0.1)
model.user_embedding.weight.register_hook(clip_grad)
```

### torch.autograd.Function (自定义算子)

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向计算
        ctx.save_for_backward(input)
        return some_output

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播
        input, = ctx.saved_tensors
        return grad_for_input
```

**作用**: 自定义算子(实现 straight-through estimator 等)。

**示例**: Straight-Through Estimator (用于量化、VQ-VAE)

```python
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()  # 前向: 量化到 {-1, 1}

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # 反向: 假装是恒等函数

# 使用
x = torch.randn(4, requires_grad=True)
quantized = STE.apply(x)   # 前向量化,但梯度可传
```

---

## 9. nn.Module 基础

### nn.Linear

```python
nn.Linear(in_features, out_features, bias=True)
```

**作用**: 全连接层。`y = xW^T + b`。

**参数**:
- `in_features` (int): 输入维度
- `out_features` (int): 输出维度
- `bias` (bool): 是否用偏置

**注意**:
- 权重 `W` 形状是 `[out_features, in_features]`(**不是反过来**)

### nn.Conv1d / nn.Conv2d / nn.Conv3d

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
```

**作用**: 卷积层。搜推广中**主要用于文本特征**(1D conv over embedding)。

### 归一化层

```python
nn.BatchNorm1d(num_features)        # 沿 batch 维归一化
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)     # 沿特征维归一化(Transformer 标配)
nn.GroupNorm(num_groups, num_channels)  # 小 batch 时的折衷
nn.RMSNorm(dim, eps=1e-6)          # LLaMA 标配
```

**选择决策**:
- CNN → BatchNorm
- Transformer/RNN → LayerNorm
- 小 batch + CNN → GroupNorm
- LLM / GenRec → RMSNorm

**详细对比见**: [[第11章-PyTorch基础]] §3.6.2

### nn.Embedding / nn.EmbeddingBag

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
nn.EmbeddingBag(num_embeddings, embedding_dim, mode='mean')
```

**作用**: 查表。`EmbeddingBag` 把查表 + 池化合并为单一算子(更高效)。

### nn.Dropout / nn.Dropout1d/2d/3d

```python
nn.Dropout(p=0.5, inplace=False)
```

**注意**:
- `nn.Dropout` 在 `model.eval()` 时**自动失效**
- 训练时 `p=0.1` 意味着 10% 神经元被置零

### nn.MultiheadAttention

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=False)
```

**作用**: 标准多头注意力。

**示例**:
```python
attn = nn.MultiheadAttention(d_model=512, num_heads=8, batch_first=True)
x = torch.randn(32, 50, 512)
output, weights = attn(x, x, x)     # 自注意力
```

**注意**:
- PyTorch 2.0+ 推荐用 `F.scaled_dot_product_attention`(自动选 Flash Attention)

### F.scaled_dot_product_attention (SDPA)

```python
F.scaled_dot_product_attention(query, key, value,
                                attn_mask=None,
                                dropout_p=0.0,
                                is_causal=False,
                                scale=None)
```

> **深度补充:三个 backend 的选型与 causal/dropout 写法**

**文字**: PyTorch 2.0 引入的 `F.scaled_dot_product_attention` 把 `\mathrm{softmax}(QK^{T}/√d)V` 一条链路封装成一个 op,内部按硬件和 shape 选 backend。`flash` 对应 FlashAttention / FlashAttention-2,把 QKV 分块载入 SRAM、逐块算完 $\mathrm{softmax}$ 再写回 HBM,显存从 $O(N²)$ 降到 $O(N)$,长序列(>2k)训练吞吐量翻倍;`mem_efficient` 是 xFormers 路径,适合不支持 flash 的旧卡或 head_dim 不整除 64 的场景;`math` 是朴素实现,慢但能做任意 mask(嵌套、block-local、prefix-LM)。`is_causal=True` 会构造下三角 mask,等价于 mask 写法但更省显存;dropout 直接传 `dropout_p` 参数,框架在 $\mathrm{softmax}$ 之后应用,比手动写 `attn.masked_fill + dropout` 高效,因为 mask 的 $\mathrm{softmax}$ 会被融合进 kernel。生产环境推荐显式指定 backend 而不是依赖自动选型,跨硬件部署时行为可控。

**公式**:
```text
score = Q K^T / √d_k  + mask          # 缩放点积
attn  = softmax(score) · dropout_p    # 概率分布
output = attn · V                      # 加权求和
```

**代码**:
```python
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# 显式选 backend,跨硬件部署更可控
def attn(q, k, v, causal=True, dropout_p=0.0):
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,    # 显式 causal 时传 None
            dropout_p=dropout_p if model.training else 0.0,
            is_causal=causal,           # 比手写 mask 更快
            scale=None,                 # None → 自动 1/√d
        )

# 决策代码:按 head_dim 和 seq_len 选 backend
def pick_backend(head_dim, seq_len, dtype):
    if head_dim % 64 == 0 and seq_len > 1024 and dtype in (torch.float16, torch.bfloat16):
        return SDPBackend.FLASH_ATTENTION
    if seq_len > 512:
        return SDPBackend.EFFICIENT_ATTENTION
    return SDPBackend.MATH
```

### nn.Transformer / nn.TransformerEncoder / Decoder

```python
nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
```

**作用**: 完整的 Transformer Encoder,堆叠多层 EncoderLayer。

### nn.LSTM / nn.GRU / nn.RNN

```python
nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=False)
```

**作用**: RNN 系列。搜推广里**越来越少用**,大多被 Transformer 取代。

### nn.Sequential / nn.ModuleList / nn.ModuleDict

```python
nn.Sequential(layer1, layer2, ...)
nn.ModuleList([layer1, layer2, ...])
nn.ModuleDict({'name1': layer1, 'name2': layer2})
```

**注意**:
- 简单的顺序连接用 `Sequential`
- 需要灵活 forward 逻辑(如 MoE)用 `ModuleList` + 手动循环
- 用 **普通 list** 保存子模块是**错的**——`parameters()` 不会找到!

### nn.init (初始化)

```python
nn.init.xavier_uniform_(tensor, gain=1.0)
nn.init.xavier_normal_(tensor, gain=1.0)
nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
nn.init.normal_(tensor, mean=0, std=1)
nn.init.uniform_(tensor, a=0, b=1)
nn.init.zeros_(tensor)
nn.init.constant_(tensor, val)
```

**示例**:
```python
# 推荐的标准初始化
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 小方差初始化
```

---

## 10. 优化器与学习率调度

### torch.optim.SGD

```python
torch.optim.SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

**作用**: 随机梯度下降 + 动量。

**注意**:
- `momentum=0.9` + `nesterov=True` 是常用配置
- 搜推广的 Embedding 用 SGD **很少**(Adam 系更普遍)

### torch.optim.Adam

```python
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
```

**作用**: 自适应学习率,主流默认。

**注意**:
- **搜推广中 Embedding 用 Adam 时务必 `weight_decay=0`**(否则稀疏更新代价巨大)

### torch.optim.AdamW

```python
torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
```

**作用**: 改进版 Adam,**解耦权重衰减**。

**注意**:
- 2024 年**所有主流模型**(LLaMA/BERT/ViT)的标准选择
- `weight_decay=0.1` 是 LLaMA 默认

### torch.optim.SparseAdam

```python
torch.optim.SparseAdam(params, lr=1e-2)
```

**作用**: 稀疏 Adam,**只更新被索引的 Embedding 行**。

**注意**:
- 搜推广 Embedding 表特别大时**必用**,否则内存爆炸
- 限制:`weight_decay` 必须为 0

### torch.optim.RMSprop / Adagrad / Adadelta

```python
torch.optim.RMSprop(params, lr=1e-2, alpha=0.99, weight_decay=0)
torch.optim.Adagrad(params, lr=1e-2)
torch.optim.Adadelta(params, lr=1.0, rho=0.9)
```

### torch.optim.LBFGS

```python
torch.optim.LBFGS(params, lr=1.0, max_iter=20)
```

**作用**: 二阶优化(用 Hessian 近似),收敛快但内存大。

**注意**:
- 搜推广**几乎不用**(batch 大、参数多)
- 小模型/科学计算场景用

### torch.optim.lr_scheduler

```python
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0)
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps)
```

**注意**:
- Transformer/GenRec 标准组合:**Warmup + Cosine**
- 搜推广里**不要**用 `ReduceLROnPlateau` 监控 AUC(可能错过最佳)

---

## 11. DataLoader

### torch.utils.data.Dataset

```python
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]   # 返回单条样本
```

**作用**: PyTorch Dataset 协议。**必须**实现 `__len__` 和 `__getitem__`。

**详细见**: `第10章-数据处理.md` §2.7.1

### torch.utils.data.IterableDataset

```python
class StreamDataset(IterableDataset):
    def __iter__(self):
        for item in stream:
            yield item
```

**作用**: 迭代式 Dataset,用于**流式数据**(Kafka/Flink)。

**注意**:
- 不支持 `__len__`、不与 `Sampler` 兼容
- 多 worker 时**每个 worker 独立迭代**(可能重复)

### torch.utils.data.DataLoader

```python
torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, sampler=None,
    num_workers=0, collate_fn=None, pin_memory=False,
    drop_last=False, timeout=0, prefetch_factor=2,
    persistent_workers=False
)
```

**参数关键**:
- `num_workers`: 数据加载子进程数(通常 = CPU 核数 / 2)
- `pin_memory`: 把数据锁在 CPU 内存(加速 CPU→GPU 传输)
- `persistent_workers`: 跨 epoch 保留 worker(避免反复 fork)
- `drop_last`: 丢掉最后一个不完整 batch(避免 batch size 抖动)
- `collate_fn`: 自定义 batch 组装函数(处理变长序列)

### torch.utils.data.distributed.DistributedSampler

```python
torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=None, rank=None, shuffle=True
)
```

**作用**: 在 DDP 多卡训练中,**每张卡取不同子集**。

**使用**:
```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, sampler=sampler, ...)
for epoch in range(epochs):
    sampler.set_epoch(epoch)   # 关键: 每个 epoch 不同 shuffle
    for batch in loader:
        ...
```

---

## 12. 分布式训练

### torch.distributed.init_process_group

```python
torch.distributed.init_process_group(
    backend='nccl',          # GPU: nccl; CPU: gloo
    init_method='env://',   # 从环境变量读 rank/world_size/master_addr
    world_size=-1,           # 自动检测
    rank=-1
)
```

**作用**: 初始化分布式进程组。

**注意**:
- GPU 训练**必须**用 `nccl` backend
- 通常配合 `torchrun` 自动初始化

### torch.distributed.all_reduce / all_gather / broadcast / barrier

```python
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM)   # 求和/求平均
torch.distributed.all_gather(tensor_list, tensor)        # 收集到所有卡
torch.distributed.broadcast(tensor, src=0)                # 从 src 广播
torch.distributed.barrier()                              # 同步所有卡
```

**示例**: 计算全局 InfoNCE 负样本
```python
# 收集所有 GPU 的 embeddings
local_emb = model.encode(batch)             # [B, D]
all_embs = [torch.zeros_like(local_emb) for _ in range(world_size)]
torch.distributed.all_gather(all_embs, local_emb)
global_emb = torch.cat(all_embs, dim=0)     # [B*world_size, D]
# 用 global_emb 做负样本,比单卡多 world_size 倍负样本
```

### torch.nn.parallel.DistributedDataParallel

```python
torch.nn.parallel.DistributedDataParallel(
    module, device_ids=None, output_device=None,
    find_unused_parameters=False, gradient_as_bucket_view=True
)
```

**作用**: 标准的分布式数据并行训练。

**使用**:
```python
model = DDP(model, device_ids=[local_rank])
```

### torch.distributed.fsdp

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch

model = FSDP(
    model,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, ...),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device()
)
```

**作用**: 完全分片数据并行——N 卡训练时,每张卡只存 1/N 参数,显存省 N 倍。

### torchrun

```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
         --master_addr=master.example.com --master_port=29500 \
         train.py
```

**作用**: 启动分布式训练(替代旧的 `python -m torch.distributed.launch`)。

---

## 13. 混合精度训练

> **深度补充:为什么需要 autocast + GradScaler**

**文字**: fp16 把 32 位浮点压成 16 位,数值范围从 [1.4e-45, 3.4e38] 缩小到 [5.96e-8, 65504]。训练时梯度更新量常比前向输出小几个数量级,直接用 fp16 表示会下溢成 0,模型再也学不到东西。Autocast 的解决思路很优雅:按 op 类别自动选 dtype,矩阵乘、卷积走 fp16(快且累加在 fp32),归约类、$\mathrm{softmax}$、batch_norm 保留 fp32(防溢出)。GradScaler 在 loss 端乘一个 scale_factor(初始常为 2^16),反向时梯度同步放大,优化器看到的梯度"上移"出 fp16 危险区;step 之前再 unscale 回原量级。如果某个 step 检测到 $\infty$/$\mathrm{NaN}$,scaler 自动跳过这次更新并把 scale 减半,等梯度稳了再翻倍拉回。fp32 训练一天的开销,bf16 直接砍半,显存再省三成,代价是逐 op 调度的几十微秒开销和"为某些 op 标 dtype"的规则表维护。

**公式**:
```text
scaled_loss = loss · scale_factor        # 初始 2^16 ≈ 65536
grad_scaled = backward(scaled_loss)      # 等价 grad × scale_factor
param ← param − lr · grad_scaled / scale_factor
```

**代码**:
```python
scaler = torch.amp.GradScaler('cuda')
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        # bf16 无需 scaler,fp16 才需要;dtype 选错会触发 NaN
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)            # 把梯度除回原量级
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                # inf/nan 时跳过
    scaler.update()                       # 自适应 scale
```

### torch.amp.autocast

```python
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = loss_fn(output, target)
loss.backward()
```

**作用**: 自动混合精度——autocast 块内自动用 BF16/FP16。

**dtype 选择**:
- `torch.bfloat16`: **推荐**!范围与 FP32 相同,无需 GradScaler
- `torch.float16`: 范围窄,需要 GradScaler

### torch.amp.GradScaler

```python
scaler = torch.amp.GradScaler('cuda')

for batch in loader:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()       # 梯度放大
    scaler.unscale_(optimizer)            # 取消缩放
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                # 仅当梯度无 inf/nan 时更新
    scaler.update()                       # 调整 scale 因子
```

**注意**:
- **BF16 不需要 GradScaler**!只在 FP16 时用
- 推荐用 `torch.amp.GradScaler('cuda')`(新 API)而非 `torch.cuda.amp.GradScaler`(旧)

> **深度补充:GradScaler 三个方法的内部顺序与自适应 scale**

**文字**: GradScaler 的三个方法必须按 `get_scale \to step \to update` 顺序调用,否则要么重复放大,要么跳过 $\infty$/$\mathrm{NaN}$ 检查。`get_scale()` 返回当前 scale_factor,主要供日志和调试;`step(optimizer)` 做两件事,先把缩放后的梯度 unscale 回原量级,然后检查 $\infty$/$\mathrm{NaN}$,只要任意一张量含 $\infty$ 或 $\mathrm{NaN}$ 就跳过 optimizer.step(对应本次 batch 整个丢弃),否则正常更新;`update()` 在 step 之后被调,根据"连续 N 次 step 都成功或失败"自适应调整 scale,连续成功 2000 次翻倍(直到上限 2^24),遇到一次 $\infty$/$\mathrm{NaN}$ 减半。这套机制让 scale 始终贴着"梯度上界"浮动,既不浪费 fp16 精度,也不会因为 scale 太小而下溢。scaler 状态本身是 CPU 上的 Python 对象,跨 device / 跨进程同步需要走 DDP 的 bucket 机制(配合 `model.register_post_accumulate_grad_hook`)。

**公式**:
```text
scale = scale_factor · loss            # 放大后的 loss
grad  = backward(scale)                # 梯度同步放大
has_inf_nan = any(isinf(g) or isnan(g) for g in grads)
if has_inf_nan: scale_factor ← scale_factor / 2; skip step
else:           scale_factor ← min(scale_factor · 2, 2^24)
```

**代码**:
```python
scaler = torch.amp.GradScaler('cuda', init_scale=2**16,
                              growth_interval=2000)
for epoch in range(epochs):
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)              # inf/nan → 静默跳过
        scaler.update()                     # 自适应 scale_factor
        if scaler.get_scale() < 1.0:        # 极端情况下强制提升
            print("scale 已塌缩, 跳过本 batch")
```

### torch.set_float32_matmul_precision

```python
torch.set_float32_matmul_precision('high')     # 'highest' / 'high' / 'medium'
```

**作用**: 控制 FP32 矩阵乘法的精度。

- `'medium'`: 用 BF16 模拟 FP32,最快但有微小误差
- `'high'`: 默认
- `'highest'`: 最高精度,最慢

---

## 14. 性能分析

### torch.profiler.profile

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()

# 输出表格
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome trace(在 chrome://tracing 可视化)
prof.export_chrome_trace("trace.json")
```

**schedule 参数**:
- `wait=N`: 前 N 步不记录(等待 GPU 稳定)
- `warmup=N`: 接下来 N 步 warmup(不记录)
- `active=N`: 记录 N 步
- `repeat=N`: 重复上述循环 N 次

### torch.profiler.record_function

```python
with torch.profiler.record_function("model_forward"):
    output = model(input)
```

**作用**: 在 profiler 输出中**标记**特定代码段(便于定位瓶颈)。

### torch.cuda.synchronize

```python
torch.cuda.synchronize()       # 等待所有 GPU 操作完成
```

**注意**:
- **不调用 `synchronize` 时**,CPU/GPU 是**异步**的——CPU 提交 kernel 后立即返回
- profiler 想看到准确 GPU 时间,**必须** `synchronize`
- 训练循环中**不要**滥用(会拖慢)

---

## 15. 显存管理

### torch.cuda.memory_allocated / memory_reserved

```python
allocated = torch.cuda.memory_allocated() / 1e9     # GB
reserved = torch.cuda.memory_reserved() / 1e9       # GB
print(f"已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")
```

**区别**:
- `memory_allocated`: tensor **实际占用的**显存
- `memory_reserved`: PyTorch 分配器**从 GPU 申请的**总显存(含缓存)
- 若 `reserved >> allocated` → **碎片化**

### torch.cuda.max_memory_allocated / max_memory_reserved

```python
torch.cuda.reset_peak_memory_stats()
# ... 跑训练 ...
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"峰值显存: {peak:.2f} GB")
```

**作用**: 获取 GPU 显存**峰值**(用于 OOM 排查)。

### torch.cuda.empty_cache

```python
torch.cuda.empty_cache()     # 释放 allocator 缓存的空块
```

**注意**:
- 不会释放**被 tensor 占用的**显存,只释放**未使用**的缓存
- **正常训练不需要**调用,PyTorch 自动管理
- 仅在 OOM 调试时**急救用**

### torch.cuda.memory_summary

```python
print(torch.cuda.memory_summary())
```

**作用**: 输出当前显存分配器状态(详细报告)。

### torch.cuda.memory._record_memory_history

```python
torch.cuda.memory._record_memory_history()
# ... 跑训练 ...
torch.cuda.memory._dump_snapshot("snapshot.pickle")
# 在 https://pytorch.org/memory_viz 上传查看可视化
```

### torch.utils.checkpoint.checkpoint (梯度检查点)

```python
torch.utils.checkpoint.checkpoint(function, *args,
                                   use_reentrant=False,
                                   preserve_rng_state=True)
```

> **深度补充:为什么需要 gradient checkpoint**

**文字**: 反向传播需要存所有中间激活用于链式法则,一个 L 层 Transformer 显存占用大致是 `O(L · d · b · s)`,深模型直接 OOM。`torch.utils.checkpoint.checkpoint` 用"时间换空间":前向时不存中间激活,只留输入;反向时重新跑一遍前向算中间值,显存从 $O(L)$ 降到 $O(√L)$,代价是前向被多算一次(约 33% 时间开销)。`use_reentrant` 参数决定实现路径,True 走旧的 autograd hook 方案(对 `torch.autograd` 兼容性更好,支持带 hook 的参数),False 走新的 `save_on_cpu` + 重计算路径(支持 DDP、FSDP、activation offload),2.0 之后推荐 False。`checkpoint` 函数与 `nn.Module` 配合使用时,通常会用 `functools.partial` 包装,或者直接用 `checkpoint_sequential` 处理整组层,粗粒度分段比逐层 wrap 的 Python overhead 更低。

**公式**:
```text
mem_eager    = O(L · N_act)            # 全部存中间激活
mem_ckpt     = O(√(L · N_act))         # 分段重算
time_ckpt    = 1.33 × time_eager       # 前向多算一次
```

**代码**:
```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.nn as nn

class DeepTransformer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
    def forward(self, x):
        # 方式 1:每层都 checkpoint(粒度细,灵活)
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
        # 方式 2:整组 sequential(粒度粗,overhead 低)
        # return checkpoint_sequential(self.layers, segments=4, input=x, use_reentrant=False)
```

---

## 16. JIT 编译

### torch.jit.script

```python
import torch

class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

scripted = torch.jit.script(MyModel())
scripted.save("model_scripted.pt")     # 序列化

# 加载 + 推理
loaded = torch.jit.load("model_scripted.pt")
out = loaded(x)
```

**作用**: 把 Python 代码**追踪**为 TorchScript 静态图(消除 Python 开销)。

**注意**:
- `script` 对**控制流**(if/for)有要求
- 涉及 Python `dict` / 复杂类时**可能失败**
- **推荐**先用 `torch.compile`(更现代)

### torch.jit.trace

```python
traced = torch.jit.trace(model, example_input)
traced.save("model_traced.pt")
```

**作用**: 用**示例输入**跑一次,记录所有操作,生成静态图。

**注意**:
- `trace` 比 `script` **限制更少**(支持任意 Python 代码)
- 但**只覆盖**示例输入走过的路径(if 分支没走到就不被记录)
- **2024 年后推荐用 `torch.compile`**,基本替代了 trace

### torch.compile

```python
model = torch.compile(model, mode='default')    # 推荐大多数场景
model = torch.compile(model, mode='reduce-overhead')   # 小 batch 推理
model = torch.compile(model, mode='max-autotune')     # 编译慢但最快
```

> **深度补充:torch.compile 流水线与 mode 之间的取舍**

**文字**: `torch.compile` 在 PyTorch 2.0 之后基本替代了 TorchScript,它走 Dynamo + AOTAutograd + Inductor 三段式流水线。第一步把 Python 字节码转成 FX $\text{图(支持动态 shape、张量依赖控制流)}$,第二步做反向图生成(死代码消除、算子融合),第三步交给 Inductor 把算子下放到 Triton/CUDA。`mode` 决定 Inductor 的优化预算:`default` 跑融合和内存规划,适合大多数场景;`reduce-overhead` 用 CUDA Graphs 消除 Python 与 kernel launch 的中间开销,小 batch 推理收益最大;`max-autotune` 会在编译时对每个点积、归一化做 Triton 模板穷举(可能编译几十分钟),但稳态吞吐能比 default 再高 10% 到 30%。`dynamic=True` 会让图按符号 shape 编译,序列长度不固定时(搜推广大模型生成式召回常见)能避免每次 padding 重新编译。

**公式**:
```text
fx_graph     = Dynamo.trace(model, example_input)   # Python → FX
aot_graph    = AOTAutograd(fx_graph)                # 反向 + 融合
output_code  = Inductor(aot_graph, target=GPU)      # Triton/CUDA
```

**代码**:
```python
import torch, time
model = MyTransformer().cuda()
opt = torch.compile(model, mode='reduce-overhead', dynamic=True)

# 第一次 forward 触发编译(可能数秒到数分钟),之后被缓存
@torch.inference_mode()
def serve(x):
    return opt(x)            # 稳态 latency 比 eager 降低 20% 左右

# 基准:对比 compile vs eager
for _ in range(50): _ = serve(x)   # 预热
t0 = time.perf_counter()
for _ in range(200): _ = serve(x)
print(f"compile: {(time.perf_counter()-t0)/200*1e3:.2f} ms/iter")
```

**作用**: PyTorch 2.0+ 的 JIT 编译器,**无需改模型代码**。

**mode 说明**:
- `default`: 平衡
- `reduce-overhead`: 减少 CPU 开销
- `max-autotune`: 穷举最优 kernel(编译很慢)

**详细见**: [[第11章-PyTorch基础]] §3.9

---

## 17. 量化

### torch.quantize_per_tensor

```python
torch.quantize_per_tensor(input, scale, zero_point, dtype)
```

**作用**: 把 FP32 Tensor **逐张量**量化到 INT8/INT16。

**示例**:
```python
x = torch.randn(100)
scale = 0.01
zero_point = 128
x_int8 = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
# 32× 压缩
```

### torch.quantize_per_channel

```python
torch.quantize_per_channel(input, scales, zero_points, axis, dtype)
```

**作用**: **逐通道**量化(精度更高,搜推广 Embedding 量化推荐)。

### torch.dequantize

```python
x_fp = torch.dequantize(x_int8)     # 还原为 FP32
```

### torch.ao.quantization (PyTorch 原生量化 API)

```python
import torch.ao.quantization as tq

# 1. 准备: 在模型上插 fake quant
model.qconfig = tq.get_default_qconfig('qnnpack')      # 后端
tq.prepare_qat(model, ...)                            # 训练时量化

# 2. 训练(QAT)
train_one_epoch(model, ...)

# 3. 转换: fake quant → 真实 quant
tq.convert(model, ...)

# 4. 推理(INT8 加速)
with torch.no_grad():
    output = model(input)
```

**注意**:
- 搜推广的**端侧推理**必用
- 与 GPTQ/AWQ(LLM 量化)不同——这是**通用 CV/NLP 量化**
- 详细见: https://pytorch.org/docs/stable/quantization.html

### torch.ao.quantization.quantize_dynamic

```python
quantized_model = tq.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

**作用**: **动态量化**——只量化 Linear,权重 FP32 → INT8,**激活**保持 FP32。

**适用**: 推理时激活分布变化大的场景(LSTM/RNN 较常用)。

---

## 18. ONNX 与部署

### torch.onnx.export

```python
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=14
)
```

**参数**:
- `dynamic_axes`: 哪些维度是动态的(用 `batch` / `seq` 等命名)
- `opset_version`: ONNX 算子集版本(越高支持越多算子)

**注意**:
- **必须**能用示例输入 forward 一次——避免控制流、字典、动态 shape
- 推荐先用 `torch.onnx.export(..., do_constant_folding=True)` 优化

### torch.onnx.load

```python
model_onnx = torch.onnx.load("model.onnx")
# 在 PyTorch 中推理
torch.onnx.checker.check_model(model_onnx)   # 校验合法性
```

### torch.onnx.runtime

```python
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
outputs = sess.run(None, {'input': input_data.numpy()})
```

**注意**:
- `onnxruntime` 需 `pip install onnxruntime-gpu`(GPU 版)
- 支持更多 backend(DirectML/TensorRT/CANN),部署时常用

---

## 19. 随机性控制

### torch.manual_seed

```python
torch.manual_seed(42)        # 同时设置 CPU 和 CUDA 随机种子
```

**作用**: 设置 PyTorch 全局随机种子,保证**可复现**。

### torch.cuda.manual_seed / manual_seed_all

```python
torch.cuda.manual_seed(42)         # 当前 GPU
torch.cuda.manual_seed_all(42)     # 所有 GPU
```

### torch.use_deterministic_algorithms

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

**作用**: 强制使用**确定性算法**(避免某些 CUDA 算法的不确定性)。

**注意**:
- 可能**降低性能**(约 10-30%)
- 部分算子没有确定性实现,会**报错**
- `warn_only=True` 让无法确定的算子只 warn,不报错

### 完整可复现设置模板

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

---

## 20. 检查点与序列化

### torch.save / torch.load

```python
# 保存
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'best_metric': best_metric,
}, 'checkpoint.pt')

# 加载
ckpt = torch.load('checkpoint.pt', map_location='cuda')
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1
```

**注意**:
- `torch.save` 基于 **pickle**,可能**执行任意代码**——不要加载不可信来源的 checkpoint
- 推荐用 **safetensors** 替代

### safetensors (推荐)

```python
from safetensors.torch import save_file, load_file

# 保存
save_file(model.state_dict(), 'model.safetensors')

# 加载
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)

# 懒加载(只读取需要的 tensor)
from safetensors import safe_open
with safe_open('model.safetensors', framework='pt') as f:
    key = list(f.keys())[0]    # 第一个 key
    tensor = f.get_tensor(key)
```

**优势**:
- **无 pickle 风险**(纯数据格式)
- **内存映射**(懒加载,适合大模型)
- 加载速度比 `torch.save` 快

### state_dict 模式

```python
# 模型权重
model.state_dict()                 # OrderedDict,所有 parameters 和 buffers
model.load_state_dict(sd)          # 严格匹配
model.load_state_dict(sd, strict=False)   # 允许部分缺失(迁移学习)

# 保存/恢复完整训练状态
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),         # AMP 必加
    'epoch': epoch,
    'rng_state': torch.get_rng_state(),   # 完整可复现
    'cuda_rng_state': torch.cuda.get_rng_state_all(),
}, 'checkpoint.pt')
```

---

## 附录:速查表 (Cheat Sheet)

| 类别 | 关键 API | 一句话 |
| :--- | :--- | :--- |
| 基础 | `torch.tensor`, `torch.zeros`, `torch.empty` | 创建 Tensor |
| 形状 | `view`, `reshape`, `transpose`, `permute` | 改变形状 |
| 形状 | `squeeze`, `unsqueeze`, `flatten`, `unflatten` | 处理 size=1 维 |
| 拼接 | `cat`, `stack`, `split`, `chunk`, `unbind` | 切分/拼接 |
| 索引 | `gather`, `masked_fill`, `where`, `nonzero` | 索引/选择 |
| 归约 | `sum`, `mean`, `max`, `argmax`, `topk` | 归约运算 |
| 归约 | `norm`, `std`, `var`, `cumsum`, `bincount` | 统计 |
| 矩阵 | `matmul`, `mm`, `bmm`, `einsum` | 矩阵乘法 |
| 矩阵 | `torch.linalg.{svd,qr,inv,solve}` | 线性代数 |
| 激活 | `F.relu`, `F.gelu`, `F.silu`, `F.softmax` | 激活函数 |
| 损失 | `F.cross_entropy`, `F.binary_cross_entropy_with_logits` | 分类损失 |
| 损失 | `F.mse_loss`, `F.smooth_l1_loss`, `F.triplet_margin_loss` | 回归/Ranking 损失 |
| Autograd | `torch.no_grad`, `torch.enable_grad`, `tensor.backward` | 梯度控制 |
| Autograd | `torch.autograd.grad`, `register_hook`, `torch.autograd.Function` | 高级 Autograd |
| nn | `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm` | 基础层 |
| nn | `nn.Embedding`, `nn.EmbeddingBag`, `nn.MultiheadAttention` | 推荐/注意力 |
| nn | `nn.Sequential`, `nn.ModuleList`, `nn.Dropout` | 容器/正则化 |
| 优化 | `optim.SGD`, `optim.Adam`, `optim.AdamW` | 优化器 |
| 优化 | `optim.lr_scheduler.{StepLR, CosineAnnealingLR, OneCycleLR}` | LR 调度 |
| 数据 | `DataLoader`, `Dataset`, `IterableDataset` | 数据加载 |
| 数据 | `DistributedSampler`, `collate_fn` | 分布式/批处理 |
| 分布式 | `DistributedDataParallel`, `torch.distributed` | 分布式训练 |
| 分布式 | `torchrun` (启动器) | 启动命令 |
| 混合精度 | `torch.amp.autocast`, `GradScaler` | AMP |
| 性能 | `torch.profiler.profile` | 性能分析 |
| 显存 | `torch.cuda.memory_allocated` | 显存监控 |
| 编译 | `torch.compile`, `torch.jit.script` | JIT |
| 量化 | `torch.ao.quantization.quantize` | PTQ/QAT |
| 部署 | `torch.onnx.export`, `onnxruntime` | ONNX 导出/部署 |
| 复现 | `torch.manual_seed`, `use_deterministic_algorithms` | 复现性 |
| 保存 | `torch.save`, `safetensors.save_file` | 检查点 |

---

> **本文件配套章节**:
> - `第02章-深度学习数学与理论基础.md` — 数学基础
> - [[第11章-PyTorch基础]] — PyTorch 基础(4319 行最完整)
> - `第15章-训练流程与工程实践` — 训练流程
> - [[第17章-高级工程实践]] — 高级主题

---

## 22. 散射 / 索引 / 稀疏更新 API 速查

> **本节定位**: 第 3 节「索引与选择」只覆盖了 `gather / index_select / masked_select / masked_fill / where / nonzero`。搜推广里**另外一组高频 API**——`scatter / scatter_add / index_add / index_put`——处理"按 index 把值写回 / 累加"语义,正是 Embedding 梯度累加、序列填充、beam search 投票、cross-entropy label smoothing 的核心。本节把它们一次性补齐。
>
> 注: §21 已给出 scatter / index_put / gather / view / reshape 等的"选型决策表",本节聚焦**完整签名、autograd 兼容性、生产踩坑**。

### torch.scatter / tensor.scatter_

```python
torch.scatter(input, dim, index, src)        # src → input
torch.scatter(input, dim, index, value)      # value 标量
tensor.scatter_(dim, index, src/value)
```

**作用**: 沿 `dim` 用 `index` 指定的坐标把 `src` 的元素**写回** `input`(`scatter_` 是原地版)。

**示例**:把每个样本的 top-1 类别位置设成 1,其余位置设成 0(one-hot 散射)

```python
B, V = 4, 10
pred = torch.tensor([[2], [5], [0], [7]])          # [B, 1]
one_hot = torch.zeros(B, V).scatter_(dim=1, index=pred, value=1.0)
# one_hot[i, pred[i]] = 1.0
```

**注意**:
- `index.shape == out.shape`,且 `out.shape[d] == input.shape[d] (d ≠ dim)`,`out.shape[dim] == index.shape[dim]`
- `src.shape == index.shape`
- **不要在 autograd 链里对 leaf tensor 做 `scatter_`**——若必须,用 `index_copy_` 或 `index_put_`

### torch.scatter_add / tensor.scatter_add_

```python
torch.scatter_add(input, dim, index, src)
tensor.scatter_add_(dim, index, src)
```

**作用**: 沿 `dim` 把 `src` **累加**到 `input` 的 `index` 指定位置(重复 index 自动求和)。

**示例**:把每个 item_id 在 batch 内被命中的次数累加到一个全局直方图

```python
# 搜推广:统计本 batch 内每个 item 的曝光次数
item_ids = torch.tensor([[3, 5, 7], [3, 2, 5]])   # [B, T]
flat = item_ids.flatten()                          # [B*T]
V = 10
counts = torch.zeros(V).scatter_add_(0, flat, torch.ones_like(flat).float())
# counts[k] = item k 出现次数
```

**注意**:
- `scatter_add_` 是**仅有的几种支持 autograd 反向的原地累加 API**之一,常用于自定义 embedding 梯度
- **CUDA 后端无锁**,并发写同一 index 由实现定义——不要在多流并发场景用

### torch.index_add / tensor.index_add_

```python
torch.index_add(input, dim, index, source, *, alpha=1)
tensor.index_add_(dim, index, source, alpha=1)
```

**作用**: 把 `source` 沿 `dim` 加到 `input` 的 `index` 位置(`alpha` 是 source 的系数,等价于 `out[index] += alpha * source`)。

**示例**:把多个样本的 item 嵌入累加到对应 item 槽位

```python
# 全局 item 表上累加 embedding
item_table = torch.zeros(V, D)                          # [V, D]
hits = torch.randn(T, D)                                 # [T, D]
ids  = torch.tensor([0, 5, 0, 3, 5, 2])                  # [T]
item_table.index_add_(0, ids, hits)                      # 重复 id 累加
```

**对比**:

| 维度 | `scatter_add_` | `index_add_` | 累加项来源 |
| :--- | :--- | :--- | :--- |
| 输入类型 | 仅 tensor | tensor 或 `(int, Tensor)` | src / source |
| 系数 | 固定 1.0 | `alpha` | 参数 |

### torch.index_put / tensor.index_put_

```python
torch.index_put(input, indices, values, accumulate=False)
tensor.index_put_(indices, values, accumulate=False)     # 原地
```

**作用**: 通用版"按多维 index 写回"——`indices` 是**元组**(每个维一个 index tensor),覆盖 scatter 全部功能 + 非连续 index。

**示例**:把序列对应位置填充 padding id

```python
seq = torch.zeros(B, T, dtype=torch.long)               # [B, T]
rows = torch.arange(B).unsqueeze(1)                      # [B, 1]
cols = torch.tensor(`1, 3, 5`)                         # [B, 3]
seq.index_put_((rows, cols), torch.tensor(`10, 20, 30`))
# seq[0,1]=10, seq[0,3]=20, seq[0,5]=30
```

**注意**:
- `accumulate=True` 等价 `scatter_add`——**比手动 gather→add→scatter 快**
- `indices` 是 tuple,顺序对应 `dim`
- 高级索引/写回会**触发 `.contiguous()`**,大 tensor 上有 O(N) 开销

### torch.gather vs torch.scatter vs torch.index_put

| 语义 | 读 (gather 类) | 写 (scatter 类) | 写+累加 (index_put/scatter_add) |
| :--- | :--- | :--- | :--- |
| 单维 | `torch.gather` | `torch.scatter` | `torch.scatter_add` |
| 多维 | `tensor[indices]` | `tensor.index_put_(indices, v, accumulate=False)` | `tensor.index_put_(indices, v, accumulate=True)` |
| 典型场景 | 取 top-k logits | one-hot 编码 | Embedding 梯度累加 |

**搜推广最常见组合**:
```python
# 1. gather: 取正样本的预测分数
pos_score = logits.gather(1, pos_id)            # [B, 1]

# 2. scatter: 构造 label smoothing 软标签
soft_label = torch.full((B, V), eps / V)
soft_label.scatter_(1, target, 1 - eps + eps / V)

# 3. scatter_add / index_add_: 统计 batch 内 item 曝光
item_count.scatter_add_(0, flat_ids, torch.ones_like(flat_ids).float())
```

### torch.segment_reduce / torch_scatter 替代

PyTorch 1.7+ 在 `torch_scatter` 之外提供官方 `torch.seg_reduce`(实验性,需 nightly):

```python
# segment_mean: 同一 group 内取平均
group_ids = torch.tensor([0, 0, 1, 1, 1, 2])    # 6 个点 → 3 个组
values    = torch.tensor([1., 2., 3., 4., 5., 6.])
out = torch.seg_reduce(values, reduce="mean", lengths=torch.tensor([2, 3, 1]))
# tensor([1.5, 4.0, 6.0])   等价 groupby(group_ids).mean
```

**注意**:
- `torch.seg_reduce` 仅 nightly,生产推荐用 [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) 第三方库
- 搜推广里**用户行为序列按 user 聚合**、**物品按类别聚合**都用此模式

### tensor.scatter_reduce_ (PyTorch 1.12+)

```python
tensor.scatter_reduce_(dim, index, src, reduce, *, include_self=True)
# reduce: 'sum' | 'prod' | 'mean' | 'amax' | 'amin'
```

**作用**: `scatter_add` 的**通用版**,支持 sum / mean / max / min / prod。

**示例**:对每个 user 的所有 item embedding 取 max-pooling(替代复杂的 split + stack)

```python
# 假设 padding 已经 mask 成 -inf
user_emb.scatter_reduce_(0, user_ids, item_emb, reduce='amax', include_self=False)
# 等价 segmented max-pooling
```

**陷阱**:
- `include_self=True` 时 output 会被当作"self"参与计算(影响 mean/max)
- CUDA 路径对 `reduce='mean'` 性能较差,大 batch 建议手动用 sum / count 替代

### 内存布局陷阱(踩坑实录)

| 错误写法 | 正确写法 | 现象 |
| :--- | :--- | :--- |
| `x[rows, cols] = val` | `x.index_put_((rows, cols), val)` | 高维赋值会触发 `RuntimeError: at::indexPut` |
| `x[idx] += src`(in-place) | `x.index_add_(0, idx, src)` | 重复 idx 时 `+=` 只写最后一次,`index_add_` 累加 |
| `emb[ids] = grad` | `emb.index_put_(ids, grad, accumulate=True)` | 高级索引赋值是 scatter,不是 gather |
| 跨 device 写回 `x[i] = y` | 显式 `x[i.to(x.device)] = y.to(x.device)` | `x` 在 GPU,`i`/`y` 在 CPU 会 raise |

---

## 23. 函数易混淆对比表

> **本节定位**: 速查手册最后一公里——把"看似相同、实际不同"的 API 摆到同一张表里。覆盖搜推广训练链路**真实踩过坑**的 12 组对比。每组给出**选型决策**和**典型反例**。
>
> 注: §21 按 API 族(scatter / einsum / gather / view / cat / bmm)组织对比;本节按**使用场景**组织(eval/no_grad、item/numpy、register_buffer 等),两者互为补充。

### 1. `view` vs `reshape` vs `contiguous().view()`

| API | 是否要求连续 | 行为 | 共享内存? |
| :--- | :--- | :--- | :--- |
| `tensor.view(*shape)` | **必须** | 改 stride,失败抛错 | 是 |
| `tensor.reshape(*shape)` | 否 | 连续则 view,否则 copy 后 view | **不保证** |
| `tensor.contiguous().view(*shape)` | 强转 | 先 copy 再 view | 否(新 tensor) |

**踩坑**:对 `permute` / `transpose` 结果调 `view` 必报错。`reshape` 在 batch 小可接受,大 batch 建议显式 `contiguous().view()` 让 copy 行为可观察。

**反例**:
```python
# WRONG: permute 后的 view
x = torch.randn(2, 3, 4)
x.permute(0, 2, 1).view(2, 12)   # RuntimeError
# RIGHT:
x.permute(0, 2, 1).reshape(2, 12)
# 或
x.permute(0, 2, 1).contiguous().view(2, 12)
```

### 2. `torch.gather` vs `tensor[...]` vs `torch.index_select`

| 场景 | 推荐 API | 说明 |
| :--- | :--- | :--- |
| 沿**单维**取若干行 / 列 | `index_select` | 显式 `dim`,返回 slice |
| 沿**单维**按**任意 index**取(每行不同) | `gather` | `index` 形状 = 输出 |
| 按**多维** index 同时取 | `tensor[rows, cols]` | 高级索引,行为最直白 |
| 按**布尔 mask** 取 | `tensor[mask]` / `masked_select` | 高级索引,1D 输出 |

**反例**:
```python
# WRONG: 用 gather 取整行(能跑但语义错乱)
row0 = x.gather(0, torch.tensor([0]*B))  # 实际拿到的是每列 index=0 的元素

# RIGHT: 用 index_select
row0 = x.index_select(0, torch.tensor([0]))
```

### 3. `scatter_` vs `index_put_` vs `index_add_` vs `scatter_add_`

见 §21 末尾表格,这里再补**反例**:
```python
# WRONG: 高级索引 + in-place add
x[ids] += grad              # 重复 id 会被后写覆盖,丢梯度!
# RIGHT:
x.index_add_(0, ids, grad)  # 累加语义
# 或:
x.scatter_add_(0, ids, grad)
```

### 4. `expand` vs `repeat` vs `broadcast_to`

| API | 是否复制 | size=1 维限制 | 内存 |
| :--- | :--- | :--- | :--- |
| `expand` | 否(零拷贝广播) | 仅 size=1 维可扩 | O(1) |
| `repeat` | 是 | 无限制 | O(∏sizes) |
| `broadcast_to` | 否 | 同 expand | O(1) |

**选型**:
- 临时把 `[1, D]` 变成 `[B, D]` → `expand`(零拷贝)
- 需要独立修改 / 写入 → `repeat`(真复制)
- 把 `[B]` 加到 `[B, D]` → 直接利用 PyTorch 隐式广播,**不需要**显式 expand

### 5. `tensor.clone()` vs `tensor.detach()` vs `tensor.clone().detach()`

| API | 数据相同? | 共享内存? | 需要梯度? | 何时用 |
| :--- | :--- | :--- | :--- | :--- |
| `clone()` | 复制 | 否 | **继承** | 需要独立副本 + 反向 |
| `detach()` | 引用 | **是** | **关闭** | 切计算图,内存共享 |
| `clone().detach()` | 复制 | 否 | 关闭 | 复制且切图(最常用) |
| `contiguous()` | 复制 | 否 | 继承 | 改内存布局 |

**搜推广典型**:
```python
# 蒸馏: 让 student 走 teacher 的特征但不更新 teacher
with torch.no_grad():                # 或 teacher.eval() + detach
    teacher_feat = teacher(x).detach()

# 1. detach() 即可
# 2. 如果 teacher_feat 还要再做 in-place 修改 → clone().detach()
```

### 6. `F.softmax` vs `F.log_softmax` vs `log(F.softmax(x))`

| 写法 | 数值稳定性 | 速度 |
| :--- | :--- | :--- |
| `F.softmax(x, dim)` | 一般(可能上溢) | 最快 |
| `F.log_softmax(x, dim)` | **好**(内部 log-sum-exp) | 快 |
| `torch.log(F.softmax(x, dim))` | 差(可能 -inf) | 慢 |
| `F.cross_entropy(x, target)` | **最好** | 略快 |

**规则**:算 cross-entropy 永远用 `F.cross_entropy`;算概率对数永远用 `F.log_softmax`;**永远不要** `log(softmax(x))`。

### 7. `model.eval()` vs `torch.no_grad()` vs `torch.inference_mode()`

| 场景 | 推荐 |
| :--- | :--- |
| 训练时验证 AUC | `model.eval() + torch.no_grad()` |
| 生产推理 | `model.eval() + torch.inference_mode()` |
| 写激活梯度图 | `model.eval() + torch.enable_grad()` |

**区别**:
- `model.eval()` 切换 BN/Dropout 行为,**不**关梯度
- `torch.no_grad()` 关梯度(节省显存)但 tensor 还能放进计算图
- `torch.inference_mode()` 关梯度 + view tracking,**不能**再 `backward()`,但**最快**

**反例**:
```python
# WRONG: 验证时漏掉 model.eval(),BN 用 running stats 跑不动
model.train()                      # 默认
with torch.no_grad():
    val_loss = criterion(model(x_val), y_val)  # BN 用 batch stats,验证指标波动

# RIGHT:
model.eval()
with torch.no_grad():
    val_loss = criterion(model(x_val), y_val)
```

### 8. `nn.Parameter` vs `register_buffer` vs `state_dict`

| 机制 | 是否训练 | 是否进 `state_dict` | 是否被 `optimizer` 更新 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `nn.Parameter` | **是** | 是 | 是 | 权重、bias |
| `register_buffer` | 否 | 是 | 否 | BN running mean/var,位置编码,EMA |
| 普通 `tensor` (self.x) | 否 | **否** | 否 | 临时变量(常量) |

**踩坑**:
```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        # WRONG: 不会被 .to(device) 跟随
        self.pe = torch.randn(50, 64)

        # RIGHT:
        self.register_buffer('pe', torch.randn(50, 64))

        # 加持久化标记
        self.register_buffer('pe', torch.randn(50, 64), persistent=True)
```

### 9. `F.relu(inplace=True)` vs `out=` 参数 vs `tensor.relu_()`

| 写法 | 修改原 tensor? | 适用 |
| :--- | :--- | :--- |
| `F.relu(x, inplace=True)` | **是** | autograd 容易炸(BN 复用显存场景) |
| `torch.relu(x, out=y)` | 否,写入 y | 想换 buffer,out 与 x 形状一致 |
| `x.relu_()` | 是(原地) | 同 F.relu inplace |

**反例**:
```python
# WRONG: 推理时 inplace ReLU + 后续 fallback 路径
def forward(self, x):
    h = F.relu(self.fc1(x), inplace=True)
    if h.sum() < 0:                 # 报错: inplace 改了 h,sum 读到脏值
        return self.fc2(h)
    return h
# RIGHT: 不要 inplace + 后续读
```

### 10. `F.dropout(p=0.5)` vs `nn.Dropout(p=0.5)` vs `tensor.dropout_()`

| 写法 | 自动随 `model.eval()` 失效? | 适用 |
| :--- | :--- | :--- |
| `nn.Dropout(p)` | **是** | 模块化标准用法 |
| `F.dropout(x, p, training=...)` | 需手动传 `training` | 灵活控制 |
| `x.dropout_(p)` | **不失效** | 调试/特殊场景 |

### 11. `torch.tensor.item()` vs `.tolist()` vs `.numpy()` vs `.cpu().numpy()`

| 目标类型 | API | 注意 |
| :--- | :--- | :--- |
| Python 标量 | `tensor.item()` | 仅 0-d / 单元素 |
| Python list | `tensor.tolist()` | 自动跨 device 转 CPU |
| NumPy | `tensor.numpy()` | **仅 CPU tensor**;GPU tensor 先 `.cpu()` |
| 跨框架共享 | `tensor.__dlpack__()` / `from_dlpack` | 零拷贝,支持 GPU |

**反例**:
```python
# WRONG: GPU tensor 直接 numpy
x = torch.randn(3, device='cuda')
x.numpy()                 # TypeError

# RIGHT:
x.cpu().numpy()
# 或零拷贝
np.from_dlpack(x)         # 支持 GPU
```

### 12. `torch.cuda.synchronize()` vs `torch.cuda.stream()` vs `torch.cuda.Event`

| 需求 | API | 开销 |
| :--- | :--- | :--- |
| 等待所有 GPU 完成 | `torch.cuda.synchronize()` | 全局同步,慢 |
| 测单 kernel 时延 | `torch.cuda.Event(enable_timing=True)` | 轻量 |
| 跨 stream 同步 | `event.wait(stream)` / `stream.wait_event(event)` | 轻量 |
| 异步重叠计算 | `torch.cuda.stream(s)` 上下文 | 零 |

**反例**:
```python
# WRONG: 训练循环里每步 synchronize
for batch in loader:
    loss = train_step(batch)
    torch.cuda.synchronize()   # 强制同步,毁掉流水线
    print(loss.item())         # .item() 自带同步,不要重复

# RIGHT:
for batch in loader:
    loss = train_step(batch)
print(loss.item())              # 只在需要 log 时同步一次
```

---

## 24. torch.compile 调优与降级实战

> **本节定位**: 第 16 节给出了 `torch.compile` 的概览和 mode 选型。本节展开**生产环境真正需要的细节**:graph break 调试、dynamic shape、`mark_dynamic` 显式声明、自定义 backend、Inductor 调优、与 DDP / FSDP 协作、常见回退路径。

### 24.1 三段式流水线回顾

```text
Python 字节码 ──Dynamo──> FX Graph ──AOTAutograd──> 反向 FX Graph ──Inductor──> Triton/CUDA
```

任何一段失败,`torch.compile` 会**退回 eager 模式**(并在第一次调用时给 warning)。要让 compile 真正吃满性能,必须**消除 graph break**。

### 24.2 关闭 graph break 的排查流程

```python
import torch._dynamo as dynamo

# 1. 详细日志
dynamo.config.verbose = True
dynamo.config.log_level = logging.DEBUG

# 2. 把 graph break 升级为 error
dynamo.config.suppress_errors = False   # 任何 break 立刻抛错

# 3. 报告所有 break
from torch._dynamo.utils import counters
print(dynamo.utils.counters)            # 跑一次后看 counters
```

**最常见 break 原因**:
1. 用了 Python 字典 / set 做 key
2. 数据依赖的 `if` 分支
3. `for` 循环上限是 Python int 而非 tensor
4. 调用了未追踪的第三方 C 库

**修复模式**:
```python
# WRONG: 数据依赖分支
def f(x, flag: torch.Tensor):
    if flag.item():                     # .item() 触发同步,break
        return x * 2
    return x + 1

# RIGHT: 编译时分支
def f(x, flag: torch.Tensor):
    if flag:                            # tensor truthy,无 break
        return x * 2
    return x + 1
```

### 24.3 dynamic shape 与 mark_dynamic

搜推广序列长度**几乎从不变**——batch 内 padding 到 max_len,生产时单条 query 长度从 1 到 1024 不等。`dynamic=True` 编译耗时从分钟级到秒级,但稳态快很多。

```python
model = torch.compile(model, dynamic=True)
# 等价于告诉 Dynamo:shape 维度按符号编译
```

**显式声明**比全局 `dynamic=True` 更好:
```python
# 只把 seq_len 维标 dynamic,batch 维仍可静态
torch._dynamo.mark_dynamic(seq, 1)      # 标第 1 维为 dynamic
```

**陷阱**:`dynamic=True` 会**关闭部分算子的 shape 专属优化**(某些 kernel 是为具体 shape 调的)。如果生产流量 shape 集中在 2~3 个值,建议预编译几个具体 shape,而不是用 dynamic。

### 24.4 mode 详解:default / reduce-overhead / max-autotune

| mode | 适用 | 收益 | 代价 |
| :--- | :--- | :--- | :--- |
| `default` | 通用 | 算子融合、内存规划,提速 20-40% | 首次编译 1-30s |
| `reduce-overhead` | 小 batch 推理 / serving | CUDA Graphs 消除 launch overhead,Latency -30% | **显存占用略增** |
| `max-autotune` | 稳态吞吐优先 | 自动选最优 Triton 模板 / cuBLAS config | **编译 5-60 分钟** |
| `max-autotune-no-cudagraphs` | 显存紧张 + 大模型 | 跳过 CUDA Graph,稳态吞吐高且省显存 | 编译仍很慢 |

**推荐组合**:
```python
# 训练: default + 偶尔 reduce-overhead
train_model = torch.compile(model, mode='default')

# 推理服务: reduce-overhead 几乎总是对的
serve_model = torch.compile(model, mode='reduce-overhead')

# 离线 batch 评分: max-autotune 长期受益
batch_model = torch.compile(model, mode='max-autotune')
```

### 24.5 与 cudnn.benchmark / TF32 协同

```python
torch.backends.cudnn.benchmark = True         # 让 cuDNN 选最快卷积算法
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 加速 matmul
torch.set_float32_matmul_precision('high')    # 同上,新 API
```

这三项**对 `torch.compile` 透明**——Inductor 仍会接管融合算子,但底层 kernel 选择会从 cudnn.benchmark 的 cache 受益。

### 24.6 与 DDP / FSDP 协作的坑

| 场景 | 问题 | 解决 |
| :--- | :--- | :--- |
| DDP + compile | 第一次各 rank 独立编译,慢 | `torch._dynamo.config.cache_size_limit = 64` |
| FSDP + compile | reduce bucket 触发 graph break | 用 `use_orig_params=True` + `limit_all_gathers=True` |
| 编译后 backward 慢 | AOTAutograd 重算 | 设置 `torch._dynamo.config.optimize_ddp = True`(实验性) |
| 检查点加载失败 | Dynamo 0.x 编译的图与 2.x 不兼容 | 删 `~/.cache/torch_inductor` 和 `~/.triton` |

### 24.7 自定义 backend / Inductor pass

```python
# 注册自定义 backend(给 torch.compile 调用)
def my_backend(gm, example_inputs, options):
    # 在这里可以走自己的编译器
    from torch._inductor import compile
    return compile(gm, example_inputs, options)

torch.compile(model, backend=my_backend)

# 注册自定义 Inductor lowering(替换某个算子的实现)
from torch._inductor.lowering import register_lowering
@register_lowering(torch.ops.aten.scaled_dot_product_attention)
def my_sdpa(*args, **kwargs):
    return my_custom_kernel(*args, **kwargs)
```

**典型场景**:搜推广里有自定义 TopK 取负采样,可以直接走 Inductor 的 `register_lowering`,让 compile 把自定义 op 融合进 Triton kernel。

### 24.8 性能对比实验模板

```python
import torch, time

model = MyModel().cuda()
opt_model = torch.compile(model, mode='default')

x = torch.randn(64, 50, 64, device='cuda')

# 1. 预热(触发编译)
for _ in range(20): _ = opt_model(x)
torch.cuda.synchronize()

# 2. 计时 eager
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(200): _ = model(x)
torch.cuda.synchronize(); eager_ms = (time.perf_counter() - t0) / 200 * 1e3

# 3. 计时 compile
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(200): _ = opt_model(x)
torch.cuda.synchronize(); comp_ms = (time.perf_counter() - t0) / 200 * 1e3

print(f"eager:   {eager_ms:.2f} ms/iter")
print(f"compile: {comp_ms:.2f} ms/iter  speedup={eager_ms/comp_ms:.2f}x")
```

**经验值**(B=64, T=50, D=64, 单层 Transformer):
- `default`: 提速 1.3-1.8×
- `reduce-overhead`: 提速 1.5-2.2×(尤其小 batch)
- `max-autotune`: 提速 1.8-2.8×(但 5-30 分钟编译)

### 24.9 工业级 checklist(推荐流程)

1. **基线**:eager 模式 + cudnn.benchmark 跑通,记录 latency / GPU util
2. **零改动的 compile**:`torch.compile(model)` 直接包,跑同 workload
3. **若提速 < 1.2×**:打开 `dynamo.config.verbose`,看 graph break
4. **逐个修 break**:用 `torch._dynamo.allow_in_graph` 把无关紧要的 Python 函数包起来
5. **加 `mark_dynamic`**:对真实生产 shape 分布声明 dynamic
6. **尝试 `mode='reduce-overhead'`**:小 batch 推理收益最大
7. **生产部署前**:预热 N 个代表 shape,落盘 inductor cache
8. **DDP / FSDP**:`cache_size_limit=64` + 删 `~/.cache/torch_inductor` 验证无版本冲突

**回退策略**:
- compile 报错 → `torch._dynamo.reset()` + 重新加载模型
- 性能倒退 → 检查是否开了不必要的 `dynamic` 或 `max-autotune`
- 显存膨胀 → 改用 `max-autotune-no-cudagraphs` 或关闭 compile

### 24.10 真实踩坑记录

| 现象 | 根因 | 解决 |
| :--- | :--- | :--- |
| compile 后变慢 1.5× | 数据依赖控制流触发 fallback | `torch._dynamo.config.suppress_errors=False` 排查 |
| 训练 loss NaN | `loss` 含 Python 标量运算 | 把 `loss = pred.float().mean()` 中的 `.float()` 移到 `pred` |
| 多卡训练 hang | Dynamo 编译产物没广播 | `torch._dynamo.config.cache_size_limit = 64` + 各 rank 同步编译 |
| 部署时首请求 30s+ | 第一次 forward 触发编译 | 启动后用 fake input 预热 10 次 |
| 显存增长 2GB | `reduce-overhead` 开了 CUDA Graph | 改用 `max-autotune-no-cudagraphs` |

---

## 附录(续):新增章节速查行

| 类别 | 关键 API | 一句话 |
| :--- | :--- | :--- |
| 散射 | `scatter_`, `scatter_add_`, `scatter_reduce_` | 按 index 写回 / 累加 |
| 索引写 | `index_add_`, `index_put_` | 多维 index 写回,支持 `accumulate` |
| 散点聚合 | `torch.seg_reduce` | segment mean/sum/max |
| 易混淆 | `view/reshape/contiguous().view` | 形状变换三件套 |
| 易混淆 | `gather/index_select/tensor[...]` | 索引三件套 |
| 易混淆 | `clone/detach/contiguous` | 副本/切图/重布局 |
| 易混淆 | `softmax/log_softmax/log(softmax)` | 数值稳定三件套 |
| 易混淆 | `eval/no_grad/inference_mode` | 验证三件套 |
| 易混淆 | `Parameter/buffer/普通 tensor` | 状态管理三件套 |
| 易混淆 | `inplace/out=/relu_` | 原地操作三件套 |
| 易混淆 | `Dropout/F.dropout/dropout_` | 随机失活三件套 |
| 易混淆 | `item/tolist/numpy/from_dlpack` | Python 互转四件套 |
| 易混淆 | `synchronize/Event/Stream` | GPU 同步三件套 |
| 编译 | `torch.compile(mode, dynamic, backend)` | 主入口 |
| 编译 | `dynamo.config.verbose / suppress_errors` | 调试 |
| 编译 | `mark_dynamic` | 显式 dynamic shape |
| 编译 | `dynamo.allow_in_graph(fn)` | 把 Python 函数强行 inlining |
| 编译 | `inductor.compile`, `register_lowering` | 自定义 backend / op |

---

> **新增章节配套**:
> - [[第16章-生成式推荐与小型LLM设计模式]] — 配合 `torch.compile` 的生产实践
> - [[第17章-高级工程实践]] — FSDP、quantization、ONNX 配合 compile
> - [[第19章-知识蒸馏]] — `detach / clone().detach()` 模式在蒸馏中的角色

---

## 21. 易混淆函数对比速查

本节聚焦搜推广/深度学习工程中**最易写错**的 API 簇,按"对照表 + 最小可运行示例 + 陷阱"组织,补 §1-§20 的未覆盖盲区。

### 21.1 scatter / scatter_add / scatter_reduce / index_put 怎么选

四者都把 `src` 按 `index` 写入 `self` 的指定位置,关键区别在于**写入语义**:

| 函数 | 重复 index 行为 | 返回 | 典型场景 |
| :--- | :--- | :--- | :--- |
| `tensor.scatter(dim, index, src)` | **未定义**(任取其一) | 新 Tensor | one-hot 嵌入查表反向 |
| `tensor.scatter_add(dim, index, src)` | **累加** | 新 Tensor | 词袋/统计:多个 doc 落入同一桶 |
| `tensor.scatter_reduce_(dim, index, src, reduce, *, include_self=True)` | 累加/取 max/取 min | in-place | AdamW 的 sparse 梯度、Embedding 梯度累加 |
| `tensor.index_put_(indices, values, accumulate=False)` | 默认覆盖,`accumulate=True` 累加 | in-place | 通用稀疏写入(scatter 的超集) |

**最小示例**(假设 `output[B, N]` 是 user×item 评分矩阵,把 5 条 (user,item,rating) 三元组写进去):

```python
B, N = 3, 4
out = torch.zeros(B, N)
u_idx = torch.tensor([0, 1, 0, 2, 1])     # 5 个 user
i_idx = torch.tensor([2, 0, 2, 3, 0])     # 5 个 item
score = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# (1) scatter: 重复 index 行为未定义,实际谁后写谁赢(类似覆盖)
out1 = torch.zeros(B, N)
out1.scatter_(1, i_idx.unsqueeze(1).expand(5, N), score.unsqueeze(1).expand(5, N))  # 不推荐
# 等价且更清晰:
out1 = torch.zeros(B, N)
out1[u_idx, i_idx] = score                       # 但同样未定义顺序!

# (2) scatter_add: 重复 index 自动累加(正确实现稀疏矩阵 +)
out2 = torch.zeros(B, N)
out2.scatter_add_(1, i_idx.unsqueeze(1).expand(5, N), score.unsqueeze(1).expand(5, N))
# 关键:必须把 index 扩展到与 score 同形状,scatter_add 不做广播

# (3) scatter_reduce_: PyTorch 2.0+,in-place + 显式 reduce
out3 = torch.zeros(B, N)
out3.scatter_reduce_(1, i_idx.unsqueeze(1).expand(5, N), score.unsqueeze(1).expand(5, N),
                     reduce='sum', include_self=True)

# (4) index_put_: API 更直观,自动广播,推荐
out4 = torch.zeros(B, N)
out4.index_put_((u_idx, i_idx), score, accumulate=True)
# accumulate=False(默认)等价于 scatter,True 等价于 scatter_add
```

**陷阱**:
- `scatter_*` 的 `index` **形状必须与 `src` 一致**;`src` 比 `index` 长会被截断,比 `index` 短会报 `RuntimeError`
- `index_put_` 的 `indices` 是**元组**(每维一个 1D 长 Tensor),**不是**单个多维 Tensor——与 `scatter` 签名不一样,经常写错
- `scatter_reduce_` 的 `include_self=False` 表示"写入前先把 self 对应位置清零",**注意与 `accumulate=True` 互斥**语义

### 21.2 einsum 写法速查(Einstein 求和)

`torch.einsum(equation, *ops)` 用字符串描述张量运算,核心是**输入箭头左侧是输入维度标签,右侧是输出维度标签**;出现一次的标签**求和消元**(reduction),出现两次以上的标签**保留**。

| 运算 | 写法 | 等价 API |
| :--- | :--- | :--- |
| 向量内积 | `'i,i->'` | `(x*y).sum()` |
| 向量逐元素乘 | `'i,i->i'` | `x * y` |
| 矩阵乘向量 | `'ij,j->i'` | `M @ v` |
| 矩阵乘 | `'ij,jk->ik'` | `M @ N` |
| 批量矩阵乘 | `'bij,bjk->bik'` | `torch.bmm(A, B)` |
| 转置 | `'ij->ji'` | `M.t()` |
| 对角求和(迹) | `'ii->'` | `torch.trace(M)` |
| 外积 | `'i,j->ij'` | `torch.outer(x, y)` |
| 注意力分数 | `'bnd,bmd->bnm'` | `Q @ K.transpose(-1,-2)` |
| 双塔召回打分 | `'bd,nd->bn'` | `user @ item.T` |
| 特征二阶交叉 | `'bf,gf->bgf'` | `user.unsqueeze(2) * item.unsqueeze(1)` |
| 多头 attention 输出 | `'bnh,hnd->bnd'` | `(attn @ V).transpose(1,2).reshape(B,N,d)` |

**示例(双塔召回 + 注意力)**:
```python
Q = torch.randn(2, 10, 8, 16)   # [B, N, H, d_k]
K = torch.randn(2, 12, 8, 16)   # [B, M, H, d_k]
V = torch.randn(2, 12, 8, 16)
# 1) QK^T:[B, N, H, d_k] x [B, M, H, d_k] -> [B, H, N, M]
attn = torch.einsum('bnhd,bmhd->bhnm', Q, K) / (16 ** 0.5)
# 2) 注意力加权 V:[B, H, N, M] x [B, M, H, d_k] -> [B, N, H, d_k]
out = torch.einsum('bhnm,bmhd->bnhd', F.softmax(attn, dim=-1), V)
```

**性能提示**:
- einsum 会**走最优 BLAS 路径**(`ij,jk->ik` 等价且往往快于 `mm`)
- **避免**出现 `...` 省略号带来的额外 dispatch 开销
- 超大维度建议显式 `permute` 后用 `bmm`,比 einsum 更易被 Inductor 优化

### 21.3 gather / index_select / masked_select / take 怎么选

四个"按 index 取值"API 的核心差异在**index 的形状语义**:

| 函数 | index 形状 | 沿哪一维 | 输出形状 | 是否要求 index 在 dim 内单调 |
| :--- | :--- | :--- | :--- | :--- |
| `tensor.gather(dim, index)` | 与 `tensor` 同形状 | 指定 dim | 与 `index` 同形状 | 否 |
| `tensor.index_select(dim, index)` | 1D | 指定 dim | 在该维替换为 `len(index)` | 否 |
| `tensor.masked_select(mask)` | 与 `tensor` 同形状的 bool | 无 | 1D,`mask.sum()` 长 | — |
| `tensor.take(indices)` | 1D,视为**扁平化后**的索引 | 无 | 1D,`len(indices)` | 否 |

**最小示例**:
```python
x = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])              # [2, 3]

# gather: 每个位置独立查(典型:topk 索引还原分数)
idx = torch.tensor([[2, 0], [1, 2]])          # [2, 2]
torch.gather(x, 1, idx)
# -> tensor([[30, 10], [50, 60]])

# index_select: 沿 dim 抽取若干行/列
torch.index_select(x, 0, torch.tensor([1, 0]))  # 取第 1、0 行 -> [[40,50,60],[10,20,30]]

# masked_select: 拍平成 1D
mask = x > 25
torch.masked_select(x, mask)                  # tensor([30, 40, 50, 60])

# take: 把 x 视为扁平 1D
torch.take(x, torch.tensor([0, 4]))           # tensor([10, 50])
```

**陷阱**:
- `topk` 返回的索引**只能用 `gather` 还原**;用 `index_select` 只能取整行,不能按列
- `index_select` 的 index 必须是 `torch.long`;`masked_select` 的 mask 必须是 `torch.bool`
- `take` 等价于 `x.flatten().index_select(0, idx)`,在 embedding 反查(词 → 多个位置)时常用

### 21.4 view / reshape / flatten / squeeze

都改变"逻辑形状",区别在**是否要求内存连续**和**是否可逆**:

| 函数 | 要求连续 | 拷贝? | 返回类型 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `view` | 是 | 否(零拷贝) | view | 已知 contiguous 时的首选 |
| `reshape` | 否 | **可能**拷贝(不连续时) | view 或新 Tensor | 安全默认 |
| `flatten(input, start_dim=0, end_dim=-1)` | 否 | 可能拷贝 | view 或新 Tensor | 展平多个连续维 |
| `unflatten(input, dim, sizes)` | 否 | 可能拷贝 | view | 把某维拆成多维(embed 的 [B,N,d] → [B,N,H,d_k]) |

```python
x = torch.randn(2, 3, 4)
x.flatten(1).shape                     # torch.Size([2, 12])  ← 跳过 batch 维
y = x.flatten(1).unflatten(1, (3, 4))   # 还原成 [2, 3, 4]
```

**陷阱**:
- `view(-1)` 等价于 `flatten(0)`,会把 batch 维也展平,**写错会导致 cross-entropy 输入 shape 对不上**
- `transpose` / `permute` 后**一定不连续**,`view` 会报错——必须先 `contiguous()` 或直接用 `reshape`

### 21.5 cat / stack / concat / chunk / split

| 函数 | 维度关系 | 新增维? | 用途 |
| :--- | :--- | :--- | :--- |
| `torch.cat(tensors, dim)` | **同形状**(除 dim) | 否 | 拼接样本/序列 |
| `torch.stack(tensors, dim)` | **完全同形** | **是**(在 dim 处插入新维) | 堆成 batch / 序列 |
| `torch.concat` | **别名** | 同 cat | 1.13+ 推荐 cat 即可 |
| `torch.chunk(tensor, chunks, dim)` | 平均切 | 否 | 切固定块数(可能不等分) |
| `torch.split(tensor, split_size_or_sections, dim)` | 按长度切 | 否 | 精确控制每段长度 |

```python
a = torch.randn(2, 3); b = torch.randn(4, 3)
torch.cat([a, b], dim=0).shape           # [6, 3]   拼样本
torch.stack([a, b[:2]], dim=0).shape     # 报错: 形状不一致
torch.chunk(torch.randn(10, 3), 3, dim=0)  # 3 段: [4,3], [3,3], [3,3]
torch.split(torch.randn(10, 3), [2, 3, 5], dim=0)  # 精确切分
```

**陷阱**:
- `stack` 要求所有 tensor **完全同形**;`cat` 只要求在 `dim` 之外同形
- `chunk(n)` 不能保证每段相等(末尾段可能短),要等分用 `torch.chunk` 配合 `split`,或直接用 `split` 给定长度

### 21.6 bmm / matmul / einsum('bij,bjk->bik')

三个 API 都能算 batched matmul,选型建议:

| 写法 | 是否 broadcast | 速度 | 备注 |
| :--- | :--- | :--- | :--- |
| `torch.bmm(A, B)` | 否(要求 A,B 同 batch 维) | 最快 | A,B 必为 3D,严格同 batch |
| `torch.matmul(A, B)` | **是**(自动 broadcast 前导维) | 略慢 | 推荐默认,支持 4D+ 高阶张量 |
| `torch.einsum('bij,bjk->bik', A, B)` | 否 | 与 bmm 相当 | 可读性最好,可与维度置换组合 |

```python
# matmul 的广播魔法:任意前导维自动 broadcast
A = torch.randn(2, 1, 5, 4)   # [B, H, N, d]
B = torch.randn(1, 8, 4, 6)   # [B, H, d, M]
C = A @ B                       # [2, 8, 5, 6]  ← broadcast 到 [2, 8, ...]
```

**陷阱**:
- `bmm` 严格 3D,4D 会报错;多头 attention reshape 出来的 4D 张量**必须**用 `matmul` 或 `@`
- `einsum` 在维度很多时可读性下降,这时拆成 `permute` + `bmm` + `permute` 反而更清晰

---

### 21.7 搜推广最易写错的 10 组函数(代码评审速查表)

搜推广模型代码评审中,90% 的 bug 出自以下 10 组函数。理解差异比记住 API 更重要。

| # | 误区 | 正解 | 关键差异 |
| :--- | :--- | :--- | :--- |
| 1 | `scatter` 当 `index_put` 用 | 写: `x[idx] = val` → 用 `index_put`;按 index 聚合 → 用 `scatter_reduce` | `scatter` 只做"散射写入",不能直接当 setter |
| 2 | `einsum('bnd,btd->btn', q, k)` 当矩阵乘 | Q·Kᵀ → `torch.bmm(q, k.transpose(-1,-2))` 或 `einsum` 即可 | einsum 不快,只为可读;矩阵乘走 cuBLAS |
| 3 | `matmul` 当 `bmm` 用 | 2-D 用 `matmul`;3-D batch 用 `bmm` 更显式 | 行为等价,但 `bmm` 拒绝 broadcast,易抓 shape bug |
| 4 | `view` 一步到位 | 非连续 tensor 先 `.contiguous().view(...)` 或直接 `reshape` | `view` 要求内存连续,`reshape` 内部自动 copy |
| 5 | 一上来就 `.contiguous()` | 仅在 `view/reshape/transpose` 后送入某些 kernel(Conv1d)时需要 | 多余的 contiguous 是隐形显存杀手 |
| 6 | `stack` 拼 batch | 已有 batch 维要合并 → `cat`;新建维度才算 `stack` | `cat` 不增维,`stack` 必增一维 |
| 7 | `F.softmax` 当 `torch.softmax` 用 | 二者完全等价,选其一保持团队一致 | 区别仅在命名空间(函数式 vs 顶层) |
| 8 | `argmax` 拿值和索引 | 只要值用 `max(dim)`,只要索引用 `argmax`,两个都要用 `max(dim).indices` | `argmax` 只返索引,不带值 |
| 9 | `nn.Linear` 当 `F.linear` 用 | 带 bias 的可学习层用 `nn.Linear`;动态 weight(LoRA、MoE 路由)用 `F.linear(x, w, b)` | `F.linear` 无 Parameter 包装,不占 optimizer 状态 |
| 10 | `no_grad` 跑推理 | 纯推理、不需要 autograd 追踪 → 用 `inference_mode` 更快更省显存 | `inference_mode` 关闭 view-tracking,断言更严 |

**最小可运行示例**:

```python
import torch
import torch.nn.functional as F

# 4) view vs reshape
x = torch.arange(12).reshape(3, 4).t()  # 非连续
try: x.view(12)         # RuntimeError
except RuntimeError as e: print("view fail:", str(e)[:30])
y = x.reshape(12)       # OK,内部自动 contiguous
assert y.shape == (12,)

# 6) cat vs stack
a, b = torch.zeros(2, 3), torch.ones(2, 3)
torch.cat([a, b], dim=0).shape   # (4, 3)  — 合并 batch
torch.stack([a, b], dim=0).shape # (2, 2, 3) — 新增 dim

# 8) argmax vs max(dim)
logits = torch.randn(4, 10)
idx = logits.argmax(dim=1)              # 索引
val, _ = logits.max(dim=1)              # 值
val_idx, _ = logits.max(dim=1)          # 值(索引用 _ 丢掉)

# 10) no_grad vs inference_mode
with torch.inference_mode():            # 推荐用于推理
    y = model(x)
```

**避坑口诀**:能 `reshape` 别 `view`、能 `cat` 别 `stack`、能 `bmm` 别 `matmul`、能 `inference_mode` 别 `no_grad`、能 `F.linear` 别 `nn.Linear`(动态权重组装场景)。



## 深度扩展 · 多视角系统展开(续)

### 110. PyTorch 常用函数大全 - 工程视角 视角

**核心议题**: 本节从工程视角视角切入,系统展开 "PyTorch 常用函数大全" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PyTorch 常用函数大全" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工程视角视角的展开**

从工程视角视角出发,我们可以把 "PyTorch 常用函数大全" 拆解为以下几个层次:

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

信息论为"PyTorch 常用函数大全" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "PyTorch 常用函数大全" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PyTorch 常用函数大全" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PyTorch 常用函数大全" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PyTorch 常用函数大全" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工程视角视角切入,系统展开了 "PyTorch 常用函数大全" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. PyTorch 常用函数大全 - 评估视角 视角

**核心议题**: 本节从评估视角视角切入,系统展开 "PyTorch 常用函数大全" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PyTorch 常用函数大全" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 评估视角视角的展开**

从评估视角视角出发,我们可以把 "PyTorch 常用函数大全" 拆解为以下几个层次:

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

信息论为"PyTorch 常用函数大全" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "PyTorch 常用函数大全" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PyTorch 常用函数大全" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PyTorch 常用函数大全" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PyTorch 常用函数大全" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从评估视角视角切入,系统展开了 "PyTorch 常用函数大全" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. PyTorch 常用函数大全 - 工业落地 视角

**核心议题**: 本节从工业落地视角切入,系统展开 "PyTorch 常用函数大全" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PyTorch 常用函数大全" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工业落地视角的展开**

从工业落地视角出发,我们可以把 "PyTorch 常用函数大全" 拆解为以下几个层次:

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

信息论为"PyTorch 常用函数大全" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "PyTorch 常用函数大全" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PyTorch 常用函数大全" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PyTorch 常用函数大全" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PyTorch 常用函数大全" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工业落地视角切入,系统展开了 "PyTorch 常用函数大全" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. PyTorch 常用函数大全 - 前沿趋势 视角

**核心议题**: 本节从前沿趋势视角切入,系统展开 "PyTorch 常用函数大全" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PyTorch 常用函数大全" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 前沿趋势视角的展开**

从前沿趋势视角出发,我们可以把 "PyTorch 常用函数大全" 拆解为以下几个层次:

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

信息论为"PyTorch 常用函数大全" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "PyTorch 常用函数大全" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PyTorch 常用函数大全" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PyTorch 常用函数大全" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PyTorch 常用函数大全" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从前沿趋势视角切入,系统展开了 "PyTorch 常用函数大全" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. PyTorch 常用函数大全 - 召回视角 视角

**核心议题**: 本节从召回视角视角切入,系统展开 "PyTorch 常用函数大全" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PyTorch 常用函数大全" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 召回视角视角的展开**

从召回视角视角出发,我们可以把 "PyTorch 常用函数大全" 拆解为以下几个层次:

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

信息论为"PyTorch 常用函数大全" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "PyTorch 常用函数大全" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PyTorch 常用函数大全" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PyTorch 常用函数大全" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PyTorch 常用函数大全" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从召回视角视角切入,系统展开了 "PyTorch 常用函数大全" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

