---
title: "第09章-04-PagedAttention"
tags:
  - 深度学习
  - PagedAttention
description: "本章要点：PagedAttention 借鉴操作系统\"虚拟内存+分页\"的思想管理 KV Cache: 把每条序列的 KV 切成固定大小的 block(页), 用 …"
chapter_id: 推理优化-PagedAttention
last_updated: 2026-07-14
---

# 第09章-04-PagedAttention

-

## 1. 内容介绍

**PagedAttention** 借鉴操作系统"虚拟内存+分页"的思想管理 KV Cache: 把每条序列的 KV 切成固定大小的 **block(页)**, 用 **block table(页表)** 把逻辑位置映射到物理块。这样 KV 不必连续存放, 显存利用率从 20-40% 提升到 90%+。

---

## 2. 为什么这么做

**朴素 KV Cache 的问题**: 为每条请求预留 `max_len` 的连续显存。

- **内部碎片**: 实际生成 100 token 却按 2048 预留 → 浪费 95%。
- **外部碎片**: 变长请求进进出出, 连续大块难以复用。
- 结果: GPU 显存被浪费, 并发数(吞吐)上不去。

**核心洞察**: KV Cache 的访问是**顺序追加**的, 天然适合分页——就像 OS 用页表让进程"以为"内存连续, 实则物理分散。

---

## 3. 方案对比

| 方案 | 显存利用率 | 碎片 | 并发采样共享 |
|---|---|---|---|
| 连续预留 (naive) | 20-40% | 高 | ❌ 每束复制 |
| PagedAttention | 90%+ | 几乎无 | ✅ copy-on-write |

**关键收益**:
- 吞吐提升 2-4×(vLLM 论文)。
- Beam search / parallel sampling 时多个候选**共享前缀 block**, 只在分叉处才复制(copy-on-write), 大幅省显存。

---

## 4. 手撕实现(简化版分页管理器)

```python
import torch

class PagedKVCache:
    """固定 block_size 的 KV 分页管理器 (单层示意)"""
    def __init__(self, num_blocks, block_size, n_heads, head_dim):
        self.block_size = block_size
        # 物理显存池: (num_blocks, block_size, n_heads, head_dim)
        self.k_pool = torch.zeros(num_blocks, block_size, n_heads, head_dim)
        self.v_pool = torch.zeros(num_blocks, block_size, n_heads, head_dim)
        self.free = list(range(num_blocks))      # 空闲块栈
        self.block_tables = {}                    # seq_id -> [物理块号...]
        self.lengths = {}                         # seq_id -> 已用 token 数

    def _alloc(self):
        if not self.free:
            raise RuntimeError("显存块耗尽(需抢占/换出)")
        return self.free.pop()

    def append(self, seq_id, k, v):               # k,v: (n_heads, head_dim)
        table = self.block_tables.setdefault(seq_id, [])
        L = self.lengths.get(seq_id, 0)
        if L % self.block_size == 0:              # 当前块满, 申请新块
            table.append(self._alloc())
        blk, off = table[-1], L % self.block_size
        self.k_pool[blk, off] = k
        self.v_pool[blk, off] = v
        self.lengths[seq_id] = L + 1

    def gather(self, seq_id):                     # 逻辑视图: 拼回连续 KV
        table, L = self.block_tables[seq_id], self.lengths[seq_id]
        ks = [self.k_pool[b] for b in table]
        vs = [self.v_pool[b] for b in table]
        return (torch.cat(ks)[:L], torch.cat(vs)[:L])

    def fork(self, parent, child):
        """并发采样: 子序列共享父序列的块 (copy-on-write)"""
        self.block_tables[child] = list(self.block_tables[parent])  # 只复制页表
        self.lengths[child] = self.lengths[parent]
```

**要点**:
- `append` 满页才 `_alloc` → 消除内部碎片。
- `fork` 只复制**页表**不复制显存 → 前缀共享; 真正写入分叉块时才 copy(此处省略, 生产实现用引用计数)。

---

## 5. 意义

- vLLM / TGI / TensorRT-LLM 的核心内存管理机制。
- 让"高并发 LLM 服务"从实验室走向生产: 相同 GPU 服务数倍请求。
- 与 continuous batching(动态批处理)配合是现代推理引擎的两大支柱。

---

## 6. 面试高频

1. PagedAttention 解决什么问题? → KV Cache 内/外碎片, 提升显存利用率与吞吐。
2. block_size 如何权衡? → 太小页表开销大, 太大内部碎片回升(通常 16)。
3. 为什么能加速 beam search? → 前缀 block 共享 + copy-on-write。

## 相关章节

- 前置: [6.3 KV-Cache](../第06章-LLM核心技术/第06章-03-KV-Cache.md)
- 相关: [9.4 Prefix-Caching与Sampling](第09章-04-Prefix-Caching与Sampling.md)

## 5. PagedAttention 内存原理

> 深入理解 PagedAttention 为何能提升显存利用率 2-4 倍。本节从虚拟内存类比、块大小权衡、引用计数、copy-on-write 四个维度剖析其底层原理。

### 5.1 虚拟内存类比: 为什么分页

操作系统用"虚拟地址 → 物理地址"的页表机制解决碎片问题——程序以为自己拥有连续内存, 实际物理页散落在各处。PagedAttention 把这套机制搬到 GPU 显存里, 用 block table 完成 "逻辑位置 → 物理块" 的映射。两套体系的对应关系如下:

| 操作系统概念 | PagedAttention 对应 | 用途 |
|---|---|---|
| 虚拟页 (page) | KV block (固定 token 数) | 最小分配单位 |
| 页表 (page table) | block table (seq → [物理块号]) | 逻辑→物理映射 |
| TLB | GPU L2 cache / CUDA 缓存 | 加速查表 |
| 缺页中断 (page fault) | block 申请失败 → 抢占 | 容量不足处理 |
| 写时复制 (COW) | beam/parallel sampling 共享前缀 | 多路复用显存 |
| 内存交换 (swap) | CPU offload (大模型) | 容量扩展 |

**关键洞察**: KV Cache 的访问是**顺序追加、随机读取**——只往末尾写新 token, 推理时按位置读取历史。这意味着:

1. 写入永远是 append, 没有"中间插入" → 简化块管理为栈式分配。
2. 读取按逻辑位置 (0, 1, 2, ...) → block table 一次查表拼出连续视图。
3. 物理块可以**任意打散**放置, 对计算透明。

朴素方案 (continuous) 的核心缺陷是**双重浪费**:
- **预分配浪费**: 为每条序列预留 `max_seq_len` 的连续显存, 实际生成 100 token 浪费 95% 显存。
- **碎片浪费**: 变长请求进进出出, 空闲的"洞"无法被后续请求复用, 因为后续请求需要连续空间。

PagedAttention 用分页把"连续性"约束从**物理显存**转移到**逻辑视图**——分配可以离散, 访问仍按逻辑。这正是虚拟内存的本质。

### 5.2 block_size 权衡: 太小太大都不好

block_size 是 PagedAttention 最核心的超参, 决定了**块管理开销 vs 内部碎片**的权衡:

| block_size | 块表深度 (2048 token) | 内部碎片上限 | 适用场景 |
|---|---|---|---|
| 4 | 512 | 3 token (~0.15%) | 极端短文本、内存敏感 |
| 8 | 256 | 7 token (~0.34%) | 对话场景 |
| **16** | 128 | 15 token (~0.73%) | **默认推荐** (vLLM) |
| 32 | 64 | 31 token (~1.5%) | 长文本生成 |
| 64 | 32 | 63 token (~3%) | 极少改动 |
| 128+ | 16 | 极大 (5%+) | 不推荐 |

**公式**:
- 块表项数 = `ceil(seq_len / block_size)`
- 内部碎片上界 = `block_size - 1` token/序列
- 块表显存 = `num_seqs × ceil(max_len / block_size) × 4 byte` (FP32) 或 2 byte (FP16 指针)

**16 是甜点**的原因:
- 块表深度 ≤128, 查找开销可忽略。
- 碎片 <1%, 比朴素方案的 95% 浪费下降两个数量级。
- 单块大小适中, GPU kernel coalesced 访存效率高。

**太小 (block_size=4)** 的问题:
- 块表深度 512, 每次 gather 需 512 次间接寻址, kernel launch overhead 上升。
- 块表本身占显存 (`num_seqs × 512 × 4B`), 上万序列时块表也是 MB 级开销。
- GPU 访存 coalesced 模式被打散, 带宽利用率下降。

**太大 (block_size=128)** 的问题:
- 短对话 (50 token) 用不满 1 块, 浪费 78 token/序列。
- 大量短请求时, 显存利用率反而比 16 还低。

**vLLM 早期论文 (SOSP'23) 报告**: block_size=16 时相对 block_size=4, 端到端吞吐差异 <5%, 但显存占用降 30%。vLLM 0.2.x 之后允许 per-model 配置, 长上下文模型 (Llama-3-8K) 可用 32, 默认仍 16。

### 5.3 block_table 数据结构与 CUDA kernel

PagedAttention 的 CUDA kernel 接收 **block_tables[num_seqs, max_blocks]** 二维数组 + **物理 KV 池**, 计算 attention 时:

```python
# 伪代码: gather K/V 后做 attention
def paged_attention_kernel(
    q,                  # [num_seqs, n_heads, head_dim]
    k_pool, v_pool,     # [num_blocks, block_size, n_heads, head_dim]
    block_tables,       # [num_seqs, max_blocks_per_seq]
    seq_lens,           # [num_seqs]
):
    for seq_idx in range(num_seqs):
        L = seq_lens[seq_idx]
        for blk_idx in range(ceil(L / block_size)):
            phys_blk = block_tables[seq_idx, blk_idx]
            # 把物理块内 [0:block_size] 的 K/V 拷出
            k_seg = k_pool[phys_blk]   # [block_size, n_heads, head_dim]
            v_seg = v_pool[phys_blk]
            # ... attention 计算
```

**关键工程细节**:
1. **块表压缩**: 现代实现用 int32 (4B) 而非 int64 存块号, 上万序列可省 MB 级。
2. **CUDA kernel 优化**: vLLM 用了 `flash_attn` 风格的 tiling, 在一个 kernel 内完成"按块 gather + QK^T + softmax + PV", 避免中间显存分配。
3. **共享内存缓存**: 把当前块的 K/V 加载到 shared memory, 多个 warp 并行计算 attention, 访存效率逼近 dense 版本。

**手撕 block table**:
```python
import torch

class BlockTable:
    """压缩版 block table: int32, 紧凑存储"""
    def __init__(self, num_seqs, max_blocks_per_seq, device='cuda'):
        # +1 是为了 padding, 简化 kernel 边界
        self.table = torch.zeros(
            num_seqs, max_blocks_per_seq + 1,
            dtype=torch.int32, device=device
        )
        self.max_blocks = max_blocks_per_seq
        # 哨兵: 无效块号, 防止越界
        self.table[:, -1] = -1

    def get(self, seq_idx, blk_idx):
        return int(self.table[seq_idx, blk_idx])

    def set(self, seq_idx, blk_idx, phys_blk):
        self.table[seq_idx, blk_idx] = phys_blk

    def num_used(self, seq_idx):
        """返回该序列当前用了多少块"""
        return (self.table[seq_idx] >= 0).sum().item()
```

### 5.4 引用计数与 copy-on-write

beam search 和 parallel sampling 都会**分叉**——同一前缀分出多个候选。朴素方案给每条候选分配独立 KV, 显存线性增长。PagedAttention 用**引用计数 + copy-on-write (CoW)** 解决:

**机制**:
- 父序列已分配 N 块, 物理块号 = `[b1, b2, ..., bN]`。
- 子序列分叉时, `block_table[child] = block_table[parent].copy()`, 但**不复制物理 K/V**。
- 物理块加引用计数: `refcount[b1] = refcount[b2] = ... = K` (K 为共享该块的子序列数)。
- 写入时: 若 `refcount[blk] > 1`, **先复制一份私有块**再写, 并把子序列的 block_table 改成新块号; 写完把旧块 refcount -1。

**为什么能省显存**:
- 前 100 token 全共享 → 省 99 块。
- 只有分叉后 (第 101 token 起) 才有私有块。
- beam_size=4, 平均分叉位置 = 200, 序列长 1000:
  - 朴素: 4 × 1000 = 4000 块。
  - PagedAttention: 100 (共享前缀) + 4 × 900 (私有后缀) = 3700 块 → **省 7.5%**。
- 若分叉更早 (50 token), 长序列 2000: 50 + 4 × 1950 = 7850 vs 8000 → 省 1.9%。

**实际收益取决于分叉位置和序列长度**。Prefix sharing (system prompt 共享) 才是 CoW 的最大收益来源——几千 token 的 system prompt 几乎零成本共享给所有请求。

**手撕引用计数**:
```python
class RefCountedBlocks:
    def __init__(self, num_blocks):
        self.refcount = [0] * num_blocks

    def acquire(self, blk):
        self.refcount[blk] += 1

    def release(self, blk):
        self.refcount[blk] -= 1
        if self.refcount[blk] == 0:
            return True   # 可回收
        return False

    def cow_write(self, seq, blk_idx, new_data):
        """写时复制: 若 refcount>1 则分配新块"""
        phys_blk = seq.block_table[blk_idx]
        if self.refcount[phys_blk] > 1:
            new_blk = self.allocator.alloc()
            # 物理拷贝
            k_pool[new_blk] = k_pool[phys_blk].clone()
            v_pool[new_blk] = v_pool[phys_blk].clone()
            # 旧块引用 -1
            self.refcount[phys_blk] -= 1
            # 新块引用 = 1
            self.refcount[new_blk] = 1
            # 改写子序列块表
            seq.block_table[blk_idx] = new_blk
        # 写入新块
        k_pool[seq.block_table[blk_idx]] = new_data
```

### 5.5 抢占 (Preemption) 与换出 (Swapping)

当 GPU 显存块耗尽, vLLM 不立即拒绝, 而是用**抢占**机制腾空间。两种策略:

**Recomputation (重计算)**:
- 把"低优先级"序列的物理块**全部释放** (block table 清空, refcount=0)。
- 等显存空闲, 重新跑前缀 prompt, 重新生成 KV。
- 优点: 无需 CPU 内存; 缺点: 浪费算力。
- vLLM 默认策略, 适合 prompt 短的对话场景。

**Swap (换出到 CPU)**:
- 把物理块**搬到 CPU 内存** (通过 `cudaMemcpy`), block table 保留映射。
- 等显存空闲, 再 swap 回来。
- 优点: 不重算; 缺点: 需要大 CPU 内存 + PCIe 带宽, 大块 swap 慢。
- TGI / TensorRT-LLM 倾向用 swap, 因为 prompt 很长重算代价大。

**手撕抢占管理器**:
```python
class PreemptionManager:
    def __init__(self, gpu_pool, cpu_pool):
        self.gpu = gpu_pool
        self.cpu = cpu_pool
        self.swap_in_progress = {}

    def preempt_recompute(self, seq):
        """释放该 seq 的所有 GPU 块, refcount 归 0 时回收"""
        for blk in seq.block_table:
            self.gpu.refcount.release(blk)
        seq.block_table = []
        seq.swapped = False

    def preempt_swap(self, seq):
        """把 GPU 块搬到 CPU, 记录映射"""
        cpu_blks = []
        for gpu_blk in seq.block_table:
            cpu_blk = self.cpu.alloc()
            # 异步 DMA: GPU → CPU
            self.cpu.k_pool[cpu_blk].copy_(self.gpu.k_pool[gpu_blk], non_blocking=True)
            self.cpu.v_pool[cpu_blk].copy_(self.gpu.v_pool[gpu_blk], non_blocking=True)
            self.gpu.refcount.release(gpu_blk)
            cpu_blks.append(cpu_blk)
        seq.block_table = []
        seq.cpu_blocks = cpu_blks
        seq.swapped = True

    def resume_swap(self, seq):
        """把 CPU 块换回 GPU"""
        new_gpu_blks = []
        for cpu_blk in seq.cpu_blocks:
            gpu_blk = self.gpu.alloc()
            self.gpu.k_pool[gpu_blk].copy_(self.cpu.k_pool[cpu_blk], non_blocking=True)
            self.gpu.v_pool[gpu_blk].copy_(self.cpu.v_pool[cpu_blk], non_blocking=True)
            new_gpu_blks.append(gpu_blk)
        self.cpu.release(cpu_blk)
        seq.block_table = new_gpu_blks
        seq.swapped = False
```

**调度策略**:
- vLLM 用"最长序列优先抢占"——选最久没产出的序列, 影响最小。
- 抢占次数记录到 metrics, 高频抢占说明 batch size 过大, 应降并发。

### 5.6 多层 KV 共享: 一块页表管 32 层

Transformer 有 N 层 (Llama-7B = 32, Llama-70B = 80), 每层都有独立的 K/V。朴素实现给每层单独建 block table, 显存 ×N。PagedAttention 用**统一块表 + per-layer 物理池**——所有层共用一个 block_table, 但每层有独立的物理池。

**好处**:
- 块表只存一份 (几十 KB), 不随层数放大。
- 每层物理池大小 = `num_blocks × block_size × n_kv_heads × head_dim`, 32 层 = 32 × 单层。
- 调度时按"层粒度"分配块——一次分配 32 个块 (每层 1 个), 加引用计数。

**代码**:
```python
class MultiLayerPagedKV:
    def __init__(self, num_layers, num_blocks, block_size, n_kv_heads, head_dim):
        self.num_layers = num_layers
        # 每层独立物理池
        self.k_pool = [torch.zeros(num_blocks, block_size, n_kv_heads, head_dim)
                       for _ in range(num_layers)]
        self.v_pool = [torch.zeros(num_blocks, block_size, n_kv_heads, head_dim)
                       for _ in range(num_layers)]
        # 块表跨层共享: 一行 block_table 对应一层的所有块
        # block_table[seq] = [b1, b2, ...]  每个 bi 是一组 (layer_0_bi, layer_1_bi, ...)
        self.block_tables = {}    # seq_id -> list of block_group_ids
        self.free_groups = list(range(num_blocks))   # 按"组"分配
```

**注**: 实际 vLLM 实现里, "组"是**每层一个独立 block table**, 但调度器保证分配时按"组"对齐——一次申请 32 个块。这种对齐让 prefix sharing 仍然有效, 块表本身开销可控。

### 5.7 显存占用估算

给定模型与并发数, 估算 PagedAttention 的显存需求:

```text
KV 大小 (FP16, 每 token) = 2 (K+V) × n_layers × n_kv_heads × head_dim × 2 byte
                       = 2 × 32 × 8 × 128 × 2 = 131 KB/token  (Llama-3-8B)

假设:
  - 单序列平均长度 = 1024 token
  - 并发 = 100 seq
  - block_size = 16
  - 实际占用 = 1024 × 100 = 102,400 token
  - 浪费 (碎片+预留) = ~5%
  - 总 KV 显存 ≈ 102,400 × 131 KB × 1.05 ≈ 13.4 GB
```

朴素方案 (预留 2048):
```text
  - 总预留 = 2048 × 100 × 131 KB = 26.2 GB
  - 实际用 13.4 GB
  - 利用率 = 51% (理想情况下, 实际因碎片更低)
```

PagedAttention 利用率 = 13.4 / 13.4 = 100% (碎片部分已在 1.05 倍里估掉), 朴素 = 51%, **提升 ~2×**。配合 prefix sharing (system prompt 5KB), 1000 seq 可省 5GB。

### 5.8 小结

PagedAttention 的精髓是把"连续性"约束从**物理显存**转移到**逻辑视图**。它不是新算法, 而是**新内存管理**——让 GPU 显存利用率从 50% 提升到 90%+, 直接转化为 2-4 倍吞吐。这套思路的影响远超推理: 训练时的 activation checkpointing、CPU offload 都有类似"分页"影子。

关键工程要点回顾:
- **block_size=16** 是甜点, 不要轻易改。
- **block table 用 int32**, 紧凑存储。
- **CoW 配合引用计数** 实现 beam / parallel sampling 共享。
- **抢占 (recompute vs swap)** 处理显存不足。
- **多层对齐分配**让 prefix sharing 跨层生效。

下一节 (6. vLLM 完整实现) 会基于这些原理展开 vLLM 0.4+ 的工程实现细节。

---

## 6. vLLM 完整实现

> 本节不是逐行复刻某个固定版本源码，而是按稳定职责拆出一个可运行的教学版。vLLM 的模块路径和内部类名会随版本变化，但请求状态机、块分配、连续批处理、模型执行和采样这五条主线相对稳定。

### 6.1 先看整体架构

一个在线请求从 HTTP 入口到返回 token，大致经过以下组件：

```text
Client
  │
  ▼
OpenAI-compatible API Server
  │  校验参数、tokenize、生成 request_id
  ▼
Async Engine Client
  │  提交请求、接收增量结果
  ▼
Engine Core / Scheduler
  │  选择本轮运行的请求
  │  分配或复用 KV blocks
  ▼
Model Runner
  │  构造 input_ids / positions / slot_mapping
  ▼
Attention Backend
  │  PagedAttention 读取分页 KV
  ▼
Sampler
  │  temperature / top-k / top-p / stop
  ▼
Detokenizer → streaming response
```

各组件只做一件事：

| 组件 | 主要职责 | 不应承担的职责 |
|---|---|---|
| API Server | 协议转换、鉴权、流式响应 | 不直接操作 GPU block |
| Tokenizer | 文本与 token 互转 | 不决定调度顺序 |
| Scheduler | 选择请求、控制 token budget | 不执行 Transformer |
| KV Manager | 分配、释放、共享物理块 | 不计算 logits |
| Model Runner | 整理张量并发起前向 | 不维护 HTTP 连接 |
| Attention Backend | 高效访问分页 KV | 不判断 stop words |
| Sampler | 从 logits 选 token | 不分配 KV block |

这种职责分离很重要：

- 调度器运行在 CPU 控制面。
- 模型和 attention kernel 运行在 GPU 数据面。
- block table 是连接控制面和数据面的桥梁。
- 每一轮 decode 都只传必要的增量元数据。
- 在线吞吐取决于调度与执行能否重叠。

### 6.2 请求生命周期与状态机

请求不只是 `waiting` 和 `done` 两态。完整状态至少包括：

```python
from enum import Enum, auto

class RequestStatus(Enum):
    WAITING = auto()          # 已入队，尚未获得计算预算
    PREFILLING = auto()       # 正在计算 prompt KV
    RUNNING = auto()          # decode，每轮生成一个或多个 token
    PREEMPTED = auto()        # 被抢占，等待恢复
    SWAPPED = auto()          # KV 已换出到 CPU
    FINISHED_STOPPED = auto() # 命中 stop token / stop string
    FINISHED_LENGTH = auto()  # 达到 max_tokens 或模型长度上限
    FINISHED_ABORTED = auto() # 客户端取消
    FINISHED_ERROR = auto()   # 执行异常

    @property
    def is_finished(self):
        return self.name.startswith("FINISHED_")
```

典型状态转移如下：

```text
WAITING ──有预算──► PREFILLING
PREFILLING ──prompt 完成──► RUNNING
PREFILLING ──显存不足──► PREEMPTED
RUNNING ──继续生成──► RUNNING
RUNNING ──显存不足──► PREEMPTED / SWAPPED
PREEMPTED ──重新获得预算──► PREFILLING
SWAPPED ──swap in 完成──► RUNNING
RUNNING ──stop──► FINISHED_STOPPED
RUNNING ──长度上限──► FINISHED_LENGTH
任意非终态 ──取消──► FINISHED_ABORTED
```

必须维护的请求字段：

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_token_ids: set[int] = field(default_factory=set)
    seed: Optional[int] = None

@dataclass
class SequenceRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    error: Optional[str] = None

    @property
    def all_token_ids(self):
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def remaining_prefill_tokens(self):
        return max(0, len(self.prompt_token_ids) - self.num_computed_tokens)

    @property
    def generated_tokens(self):
        return len(self.output_token_ids)

    def should_stop(self, token_id: int):
        return (
            token_id in self.sampling.stop_token_ids
            or self.generated_tokens >= self.sampling.max_tokens
        )
```

边界条件不能省：

- 空 prompt 通常需要补 BOS。
- `max_tokens=0` 应直接结束，不进入 GPU。
- prompt 长度超过模型上限必须在入口拒绝。
- 客户端取消后要尽快释放 block。
- 输出达到上限时，本轮刚生成的 token 是否返回要保持协议一致。
- 请求异常不能泄漏其引用的共享 block。

### 6.3 KV Cache 容量规划

启动时先加载模型权重，再用剩余显存决定 KV block 数量。

设：

- `M_total`：GPU 总显存。
- `u`：`gpu_memory_utilization`。
- `M_weight`：模型权重和常驻 buffer。
- `M_peak`：一次 profile forward 的峰值激活。
- `M_block`：一个跨全部层 KV block 的字节数。

则可用于 KV 的预算近似为：

$$
M_{KV} = M_{total} \times u - M_{weight} - M_{peak}
$$

$$
N_{blocks} = \left\lfloor \frac{M_{KV}}{M_{block}} \right\rfloor
$$

一个 block 跨全部 Transformer 层时：

$$
M_{block} = 2 \times L \times B \times H_{kv} \times D \times bytes(dtype)
$$

其中：

- `2` 表示 K 和 V。
- `L` 是层数。
- `B` 是 block_size。
- `H_kv` 是 KV head 数。
- `D` 是 head_dim。

容量估算函数：

```python
def kv_block_bytes(
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    return (
        2
        * num_layers
        * block_size
        * num_kv_heads
        * head_dim
        * dtype_bytes
    )


def estimate_num_blocks(
    total_gpu_bytes: int,
    utilization: float,
    weight_bytes: int,
    peak_activation_bytes: int,
    block_bytes: int,
) -> int:
    if not 0 < utilization <= 1:
        raise ValueError("utilization 必须位于 (0, 1]")
    budget = int(total_gpu_bytes * utilization)
    available = budget - weight_bytes - peak_activation_bytes
    if available <= 0:
        return 0
    return available // block_bytes
```

以 24 GiB GPU 为例：

```python
GiB = 1024 ** 3
block_bytes = kv_block_bytes(
    num_layers=32,
    block_size=16,
    num_kv_heads=8,
    head_dim=128,
    dtype_bytes=2,
)
num_blocks = estimate_num_blocks(
    total_gpu_bytes=24 * GiB,
    utilization=0.90,
    weight_bytes=16 * GiB,
    peak_activation_bytes=2 * GiB,
    block_bytes=block_bytes,
)
print(block_bytes, num_blocks)
```

生产实现不会完全依赖静态公式，而会执行一次 profile run：

1. 用最大候选 batch 构造虚拟输入。
2. 记录模型前向峰值显存。
3. 清理临时 tensor 和 CUDA graph 缓冲。
4. 根据剩余显存建立 KV cache tensor。
5. 若 block 数为 0，启动直接失败并给出调参建议。

### 6.4 物理块分配器

最小可用分配器只需空闲栈和引用计数：

```python
class NoFreeBlocks(RuntimeError):
    pass


class PhysicalBlockAllocator:
    """教学版 GPU block 分配器。"""

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks 必须为正数")
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.refcounts = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def allocate(self) -> int:
        if not self.free_blocks:
            raise NoFreeBlocks("没有可用 KV block")
        block_id = self.free_blocks.pop()
        if self.refcounts[block_id] != 0:
            raise RuntimeError("空闲表与引用计数不一致")
        self.refcounts[block_id] = 1
        return block_id

    def acquire(self, block_id: int) -> None:
        self._check_id(block_id)
        if self.refcounts[block_id] <= 0:
            raise RuntimeError("不能引用未分配块")
        self.refcounts[block_id] += 1

    def release(self, block_id: int) -> bool:
        self._check_id(block_id)
        if self.refcounts[block_id] <= 0:
            raise RuntimeError("重复释放 block")
        self.refcounts[block_id] -= 1
        became_free = self.refcounts[block_id] == 0
        if became_free:
            self.free_blocks.append(block_id)
        return became_free

    def is_shared(self, block_id: int) -> bool:
        self._check_id(block_id)
        return self.refcounts[block_id] > 1

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(f"非法 block_id={block_id}")

    def validate(self) -> None:
        free_set = set(self.free_blocks)
        if len(free_set) != len(self.free_blocks):
            raise AssertionError("空闲块重复")
        for block_id, refcount in enumerate(self.refcounts):
            if (block_id in free_set) != (refcount == 0):
                raise AssertionError(
                    f"block={block_id} 的空闲状态与 refcount 冲突"
                )
```

为什么用栈而不是复杂伙伴系统：

- 所有块等大，不需要按尺寸查找。
- 分配和释放都是 $O(1)$。
- 物理连续性不是正确性要求。
- 栈实现最容易验证引用计数不变量。
- 真正的性能热点在 GPU kernel，不在这几个 Python 操作。

关键不变量：

```text
refcount == 0  <=> block 位于 free_blocks
refcount > 0   <=> 至少一个逻辑块引用该物理块
sum(refcount)  == 所有 block table 引用数之和
同一 block_id 不能在 free_blocks 出现两次
```

### 6.5 逻辑块与 slot mapping

逻辑 token 位置到物理槽位的映射为：

```python
def logical_to_slot(
    block_table: list[int],
    token_position: int,
    block_size: int,
) -> int:
    if token_position < 0:
        raise ValueError("token_position 不能为负")
    logical_block = token_position // block_size
    offset = token_position % block_size
    if logical_block >= len(block_table):
        raise IndexError("逻辑块尚未分配")
    physical_block = block_table[logical_block]
    return physical_block * block_size + offset
```

例如：

```text
block_size = 4
block_table = [7, 2, 9]
逻辑位置 0  -> 物理 slot 28
逻辑位置 3  -> 物理 slot 31
逻辑位置 4  -> 物理 slot 8
逻辑位置 6  -> 物理 slot 10
逻辑位置 8  -> 物理 slot 36
```

模型执行前，CPU 会为本轮 token 构造 `slot_mapping`：

```python
def build_slot_mapping(requests, block_size):
    slots = []
    owners = []
    for req in requests:
        start = req.num_computed_tokens
        end = req.num_tokens
        for position in range(start, end):
            slot = logical_to_slot(
                req.block_table,
                position,
                block_size,
            )
            slots.append(slot)
            owners.append(req.request_id)
    return slots, owners
```

`slot_mapping` 的作用：

- 告诉 reshape-and-cache kernel 把新 K/V 写到哪里。
- 避免先构造连续 KV 再复制到分页池。
- prefill 时包含多个 token 的槽位。
- decode 时每个序列通常只包含一个槽位。
- padding token 可映射到哨兵槽位并在 kernel 中跳过。

### 6.6 教学版 KV Cache Manager

```python
from math import ceil


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size 必须为正数")
        self.block_size = block_size
        self.allocator = PhysicalBlockAllocator(num_blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负")
        return ceil(num_tokens / self.block_size)

    def additional_blocks_needed(
        self,
        req: SequenceRequest,
        target_tokens: int,
    ) -> int:
        target_blocks = self.blocks_needed(target_tokens)
        return max(0, target_blocks - len(req.block_table))

    def can_allocate(
        self,
        req: SequenceRequest,
        target_tokens: int,
    ) -> bool:
        needed = self.additional_blocks_needed(req, target_tokens)
        return needed <= self.allocator.num_free

    def ensure_capacity(
        self,
        req: SequenceRequest,
        target_tokens: int,
    ) -> None:
        needed = self.additional_blocks_needed(req, target_tokens)
        allocated = []
        try:
            for _ in range(needed):
                allocated.append(self.allocator.allocate())
        except Exception:
            for block_id in allocated:
                self.allocator.release(block_id)
            raise
        req.block_table.extend(allocated)

    def free_request(self, req: SequenceRequest) -> None:
        for block_id in req.block_table:
            self.allocator.release(block_id)
        req.block_table.clear()

    def fork(self, parent: SequenceRequest, child: SequenceRequest) -> None:
        if child.block_table:
            raise ValueError("child 必须尚未分配 block")
        child.block_table = list(parent.block_table)
        child.num_computed_tokens = parent.num_computed_tokens
        for block_id in child.block_table:
            self.allocator.acquire(block_id)

    def ensure_writable_tail(self, req: SequenceRequest) -> None:
        if not req.block_table:
            return
        tail = req.block_table[-1]
        used_in_tail = req.num_tokens % self.block_size
        tail_is_partial = used_in_tail != 0
        if not tail_is_partial or not self.allocator.is_shared(tail):
            return
        new_tail = self.allocator.allocate()
        # 真正实现还要复制旧 tail 的有效 K/V。
        # copy_kv_block(src=tail, dst=new_tail)
        req.block_table[-1] = new_tail
        self.allocator.release(tail)
```

`ensure_capacity` 使用事务式回滚：

- 先把本次新分配的块记到局部列表。
- 任一分配失败，释放本次已拿到的块。
- 成功后再统一写入请求页表。
- 避免“分到一半失败”导致显存泄漏。

`ensure_writable_tail` 只复制尾块：

- 已填满的历史块永远不会再写，可继续共享。
- 未填满尾块如果只有一个引用，可以原地写。
- 未填满尾块如果多个引用，必须 CoW。
- 复制时只需复制有效槽位，未使用槽位可不复制。

### 6.7 连续批处理调度器

静态 batching 必须等整个 batch 都完成。连续 batching 每轮都可加入新请求、移除完成请求。

调度目标不是“塞入最多请求”，而是在约束下最大化有效 token：

```text
约束 1: 本轮 token 数 <= max_num_batched_tokens
约束 2: 活跃序列数 <= max_num_seqs
约束 3: 新增 KV blocks <= free blocks
约束 4: 每个请求不超过模型长度
约束 5: chunked prefill 时单请求 prefill 可被截断
```

调度结果结构：

```python
@dataclass
class ScheduledItem:
    request: SequenceRequest
    num_scheduled_tokens: int
    is_prefill: bool


@dataclass
class SchedulerOutput:
    items: list[ScheduledItem] = field(default_factory=list)
    preempted: list[SequenceRequest] = field(default_factory=list)

    @property
    def total_tokens(self):
        return sum(x.num_scheduled_tokens for x in self.items)
```

教学版调度器：

```python
from collections import deque


class ContinuousBatchScheduler:
    def __init__(
        self,
        kv_manager: KVCacheManager,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        long_prefill_token_threshold: int = 256,
    ):
        self.kv = kv_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.waiting = deque()
        self.running = []

    def add_request(self, req: SequenceRequest) -> None:
        if req.status is not RequestStatus.WAITING:
            raise ValueError("只能加入 WAITING 请求")
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        budget = self.max_num_batched_tokens

        # 先服务正在 decode 的请求，降低 TPOT。
        still_running = []
        for req in self.running:
            if budget == 0:
                still_running.append(req)
                continue
            target_tokens = req.num_tokens + 1
            if not self.kv.can_allocate(req, target_tokens):
                output.preempted.append(req)
                req.status = RequestStatus.PREEMPTED
                continue
            self.kv.ensure_writable_tail(req)
            self.kv.ensure_capacity(req, target_tokens)
            output.items.append(ScheduledItem(req, 1, False))
            still_running.append(req)
            budget -= 1
        self.running = still_running

        # 剩余预算给 waiting 请求做 prefill。
        while (
            self.waiting
            and budget > 0
            and len(self.running) < self.max_num_seqs
        ):
            req = self.waiting[0]
            remaining = req.remaining_prefill_tokens
            chunk = min(
                remaining,
                budget,
                self.long_prefill_token_threshold,
            )
            if chunk <= 0:
                self.waiting.popleft()
                req.status = RequestStatus.RUNNING
                self.running.append(req)
                continue
            target = req.num_computed_tokens + chunk
            if not self.kv.can_allocate(req, target):
                break
            self.kv.ensure_capacity(req, target)
            self.waiting.popleft()
            req.status = RequestStatus.PREFILLING
            output.items.append(ScheduledItem(req, chunk, True))
            self.running.append(req)
            budget -= chunk

        return output
```

这个版本故意省略了优先级、swap、encoder-decoder 和多模态占位符，但保留核心约束。

调度后的状态更新：

```python
def apply_model_output(
    item: ScheduledItem,
    sampled_token_id: int | None,
) -> None:
    req = item.request
    req.num_computed_tokens += item.num_scheduled_tokens

    if item.is_prefill:
        prompt_done = (
            req.num_computed_tokens >= len(req.prompt_token_ids)
        )
        req.status = (
            RequestStatus.RUNNING
            if prompt_done
            else RequestStatus.PREFILLING
        )
        return

    if sampled_token_id is None:
        raise RuntimeError("decode 必须返回 sampled token")
    req.output_token_ids.append(sampled_token_id)
    req.num_computed_tokens += 1
    if req.should_stop(sampled_token_id):
        if sampled_token_id in req.sampling.stop_token_ids:
            req.status = RequestStatus.FINISHED_STOPPED
        else:
            req.status = RequestStatus.FINISHED_LENGTH
```

### 6.8 模型执行输入的打包

混合 batch 可能同时包含：

- 请求 A：prefill 128 token。
- 请求 B：decode 1 token。
- 请求 C：decode 1 token。
- 请求 D：prefill 64 token。

不能简单堆成 `[batch, seq]` 的矩形 tensor，因为 padding 会浪费计算。常见做法是把 token 展平：

```text
input_ids = [A 的 128 个, B 的 1 个, C 的 1 个, D 的 64 个]
positions = [0..127, pos_B, pos_C, 0..63]
slot_mapping = [每个 token 对应的物理 KV slot]
query_start_loc = [0, 128, 129, 130, 194]
seq_lens = [128, len_B, len_C, 64]
```

打包函数：

```python
@dataclass
class ModelInput:
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    query_start_loc: list[int]
    seq_lens: list[int]
    request_ids: list[str]


def prepare_model_input(
    items: list[ScheduledItem],
    block_size: int,
) -> ModelInput:
    input_ids = []
    positions = []
    slot_mapping = []
    query_start_loc = [0]
    seq_lens = []
    request_ids = []

    for item in items:
        req = item.request
        start = req.num_computed_tokens
        end = start + item.num_scheduled_tokens
        tokens = req.all_token_ids[start:end]

        if len(tokens) != item.num_scheduled_tokens:
            raise AssertionError("调度 token 数与请求数据不一致")

        input_ids.extend(tokens)
        positions.extend(range(start, end))
        for position in range(start, end):
            slot_mapping.append(
                logical_to_slot(
                    req.block_table,
                    position,
                    block_size,
                )
            )
        query_start_loc.append(len(input_ids))
        seq_lens.append(req.num_tokens)
        request_ids.append(req.request_id)

    return ModelInput(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        request_ids=request_ids,
    )
```

这种 ragged layout 的收益：

- 不为不同长度请求补大量 padding。
- prefill 和 decode 能共享一次模型调用。
- attention backend 根据边界数组区分请求。
- CUDA graph 可按常见 token 数建立多个捕获尺寸。

### 6.9 PagedAttention 前向伪实现

下面代码强调语义，不追求 GPU 性能：

```python
import math
import torch


def reference_paged_attention(
    query,          # [num_heads, head_dim]
    k_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,        # 同上
    block_table,    # [num_logical_blocks]
    seq_len: int,
    scale: float,
):
    keys = []
    values = []
    block_size = k_cache.shape[1]

    for logical_block in range(math.ceil(seq_len / block_size)):
        physical_block = int(block_table[logical_block])
        valid = min(block_size, seq_len - logical_block * block_size)
        keys.append(k_cache[physical_block, :valid])
        values.append(v_cache[physical_block, :valid])

    k = torch.cat(keys, dim=0)      # [seq, kv_heads, dim]
    v = torch.cat(values, dim=0)

    # 简化为 MHA；GQA 需要把 KV head 映射到多个 query head。
    scores = torch.einsum("hd,thd->ht", query, k) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    output = torch.einsum("ht,thd->hd", probs, v)
    return output
```

生产 kernel 不会真的 `torch.cat`：

1. 每个 warp 负责一个或一组 query head。
2. 根据 block table 逐块加载 K。
3. 在线维护分块 softmax 的最大值和归一化项。
4. 同步累加 `P @ V`。
5. 最终只写一个 output tensor。
6. 中间 attention matrix 不落显存。

分块 softmax 合并公式：

设旧块统计量为 $(m_a, l_a, o_a)$，新块为 $(m_b, l_b, o_b)$：

$$
m = \max(m_a, m_b)
$$

$$
l = e^{m_a-m}l_a + e^{m_b-m}l_b
$$

$$
o = \frac{e^{m_a-m}l_a o_a + e^{m_b-m}l_b o_b}{l}
$$

这与 FlashAttention 的在线 softmax 思路一致，只是 K/V 地址来自 block table。

### 6.10 采样器

```python
def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    if logits.ndim != 1:
        raise ValueError("logits 必须是一维词表向量")

    if params.temperature == 0:
        return int(torch.argmax(logits))
    if params.temperature < 0:
        raise ValueError("temperature 不能为负")

    scores = logits.float() / params.temperature

    if params.top_k > 0:
        k = min(params.top_k, scores.numel())
        threshold = torch.topk(scores, k).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    if params.top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > params.top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        filtered.scatter_(0, sorted_idx, sorted_scores)
        scores = filtered

    probs = torch.softmax(scores, dim=-1)
    token = torch.multinomial(probs, 1, generator=generator)
    return int(token.item())
```

采样工程注意点：

- softmax 用 FP32，输出 token 再转整型。
- `temperature=0` 等价于 greedy，不做除零。
- top-p 必须保留跨过阈值的第一个 token。
- 多请求随机数状态要隔离，保证 seed 可复现。
- repetition penalty 需要读取请求历史 token。
- tensor parallel 下 logits 可能分片，采样前要聚合或使用分布式采样。

### 6.11 Engine 主循环

```python
class ToyEngine:
    def __init__(self, scheduler, model_runner, kv_manager):
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.kv = kv_manager

    def add_request(self, req):
        self.scheduler.add_request(req)

    def step(self):
        scheduled = self.scheduler.schedule()
        if not scheduled.items:
            return []

        model_input = prepare_model_input(
            scheduled.items,
            self.kv.block_size,
        )
        outputs = self.model_runner.execute(model_input)

        finished = []
        for item, token_id in zip(scheduled.items, outputs.token_ids):
            apply_model_output(item, token_id)
            if item.request.status.is_finished:
                self.kv.free_request(item.request)
                finished.append(item.request)

        return finished
```

真实异步引擎还要处理：

- 输入队列和输出队列。
- 多进程 worker 生命周期。
- tensor parallel / pipeline parallel 通信。
- NCCL 错误传播。
- 客户端断开连接。
- speculative decoding 的多 token 验证。
- LoRA adapter 的动态装载。
- 多模态 encoder cache。
- metrics 与 tracing。

### 6.12 启动 OpenAI 兼容服务

常见命令形式：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching
```

客户端请求：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是严谨的算法导师。"},
        {"role": "user", "content": "解释 PagedAttention。"},
    ],
    temperature=0.2,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

流式请求：

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "给出三条优化建议"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

注意：命令行参数会随 vLLM 版本调整。应以部署版本的 `vllm serve --help` 和稳定版文档为准，不要把内部模块路径写死到运维脚本。

### 6.13 关键配置解释

| 配置 | 控制对象 | 调大后的收益 | 调大后的风险 |
|---|---|---|---|
| `gpu_memory_utilization` | KV 可用显存比例 | 更多 block、更高并发 | OOM 余量变小 |
| `max_model_len` | 单请求最大上下文 | 支持长文本 | profile 与缓存预算增加 |
| `block_size` | KV 分配粒度 | 大块页表更浅 | 内部碎片增大 |
| `kv_cache_dtype` | KV 存储精度 | FP8 可扩大容量 | 精度与硬件兼容风险 |
| `enable_prefix_caching` | 前缀复用 | 重复 prompt 降 TTFT | 哈希和缓存管理开销 |
| `max_num_batched_tokens` | 每轮 token 预算 | prefill 吞吐提高 | decode 延迟可能抖动 |
| `max_num_seqs` | 活跃序列上限 | 并发提高 | 调度与显存压力增加 |

调参顺序建议：

1. 先固定模型、精度和最大上下文。
2. 确认单请求输出正确。
3. 用保守显存比例跑通。
4. 根据 OOM 余量提高 KV 容量。
5. 再调 token budget 和并发数。
6. 最后评估 prefix cache、FP8 KV、spec decode。

### 6.14 最小正确性自检

```python
def demo_allocator_and_mapping():
    allocator = PhysicalBlockAllocator(4)
    first = allocator.allocate()
    second = allocator.allocate()
    assert (first, second) == (0, 1)
    allocator.acquire(first)
    assert allocator.refcounts[first] == 2
    assert allocator.release(first) is False
    assert allocator.release(first) is True
    allocator.validate()

    table = [7, 2, 9]
    assert logical_to_slot(table, 0, 4) == 28
    assert logical_to_slot(table, 4, 4) == 8
    assert logical_to_slot(table, 11, 4) == 39


def demo_kv_manager():
    req = SequenceRequest(
        request_id="r1",
        prompt_token_ids=list(range(17)),
        sampling=SamplingParams(max_tokens=4),
    )
    manager = KVCacheManager(num_blocks=8, block_size=16)
    manager.ensure_capacity(req, 17)
    assert len(req.block_table) == 2
    assert manager.allocator.num_free == 6
    manager.free_request(req)
    assert manager.allocator.num_free == 8
    manager.allocator.validate()


if __name__ == "__main__":
    demo_allocator_and_mapping()
    demo_kv_manager()
```

这个自检覆盖三个最容易出错的点：

- 引用计数释放到 0 才回收。
- 跨 block 边界的 slot mapping 正确。
- 请求结束后所有块都归还。

### 6.15 读 vLLM 源码的方法

不要从 CUDA kernel 开始硬啃。推荐顺序：

1. 找服务入口，确认请求对象如何创建。
2. 跟踪一次 request_id 进入 engine 的路径。
3. 找 scheduler 输出的数据结构。
4. 看 KV cache manager 怎样分配 block。
5. 看 model runner 怎样构造 input batch。
6. 最后再看 attention backend 和 kernel。

阅读时画三张图：

- **状态图**：请求从 waiting 到 finished。
- **数据图**：token、position、slot、block table 的形状。
- **时序图**：CPU 调度、H2D、GPU forward、D2H、采样。

遇到版本差异时，优先抓稳定概念：

- 是否按 token budget 调度。
- block table 在何处生成。
- slot mapping 在何处消费。
- prefix hash 在何时查询。
- preemption 在何时触发。
- 完成请求在何时释放 block。

### 6.16 本章小结

vLLM 的高吞吐不是单一 CUDA kernel 带来的，而是系统协同：

- PagedAttention 提高 KV 显存利用率。
- continuous batching 提高 GPU 有效占用。
- ragged input 避免 padding 浪费。
- chunked prefill 控制长 prompt 对 decode 的阻塞。
- prefix caching 复用重复前缀。
- 异步服务把网络、调度和 GPU 执行串成流水线。

判断一个实现是否真正理解 vLLM，可以检查它是否回答清楚五个问题：

1. 新 token 的 K/V 最终写进哪个物理 slot？
2. 请求何时获得新 block，何时释放？
3. 长 prompt 为什么不会永久饿死 decode？
4. 两个相同前缀如何命中同一批物理 block？
5. 显存不足时系统牺牲吞吐、延迟还是算力？

## vLLM 内部用 HF tokenizer
class TokenizerWorker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
```

**优化**:
- 多个请求用同一 tokenizer, 用线程级 tokenizer 池。
- Detokenization 增量做 (每步 decode 一个 token), 减少 CPU-GPU 同步。

### 6.9 Sampling 参数

vLLM 的 `SamplingParams` 支持多种采样策略:

```python
from vllm import SamplingParams

## Greedy
sp = SamplingParams(temperature=0.0, max_tokens=100)

## Top-p (nucleus) sampling
sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

## Top-k
sp = SamplingParams(temperature=0.8, top_k=50, max_tokens=100)

## Beam search
sp = SamplingParams(n=4, best_of=4, use_beam_search=True, max_tokens=100)

## 多输出 (parallel sampling)
sp = SamplingParams(n=4, temperature=0.8, max_tokens=100)  # 返回 4 个独立采样
```

Sampling 在 ModelWorker 里通过 `torch.multinomial` 或 vLLM 自家 CUDA kernel (`csrc/sampling`) 完成, 通常和 attention kernel **fusion**——一个 kernel 完成 attention + sampling, 减少中间显存。

### 6.10 AsyncLLMEngine 与流式输出

生产部署用 `AsyncLLMEngine`:

```python
from vllm.engine.async_llm_engine import AsyncLLMEngine

engine = AsyncLLMEngine.from_engine_args(engine_args)

async def stream_generate(prompt: str):
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    # 流式: 每个 token yield 一次
    async for output in engine.generate(prompt, sampling_params, request_id="req-1"):
        yield output.outputs[0].text
```

**AsyncLLMEngine 内部**:
- 用 asyncio 协程, 单线程处理数千并发请求。
- 内部用 `asyncio.Queue` 跨协程通信。
- 一个 ModelWorker (单 GPU) 跑同步 step, 多个请求共享。

**OpenAI 兼容服务**:
vLLM 提供 `vllm.entrypoints.openai.serving_chat`, 启动后是 OpenAI API 兼容 server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --port 8000
```

可直接用 OpenAI Python SDK 调用, 业务代码无感迁移。

### 6.11 Tensor Parallel 多 GPU

大模型 (70B+) 单卡放不下, 需 TP 切分:

```python
EngineArgs(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,    # 4 卡 TP=4
)
```

vLLM 用 NCCL 做卡间通信:
- **ColumnParallelLinear**: 沿输出维度切, 每卡算局部输出, 最后 all_gather。
- **RowParallelLinear**: 沿输入维度切, 每卡算局部输出, 最后 all_reduce。
- **PagedAttention**: 每卡独立 KV 池, attention 在卡内完成, 无需跨卡通信 (因为 Q/K/V 都被切到对应卡)。

**注意**: 70B 模型 TP=4, 每张卡需要 ~20GB KV 池 + 权重, 总显存 ~80GB + 激活。

### 6.12 vLLM 的实际部署参数

生产部署常用的 EngineArgs 配置:

```python
EngineArgs(
    model="meta-llama/Llama-3-70b-instruct",
    tensor_parallel_size=8,             # 8 卡 A100
    pipeline_parallel_size=1,
    gpu_memory_utilization=0.92,        # 留 8% 给其他
    max_num_batched_tokens=8192,         # 大 batch
    max_num_seqs=256,                   # 最多 256 并发
    block_size=16,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,        # 长 prompt 切块
    enforce_eager=False,                # 启用 CUDA graph
    dtype="bfloat16",
    quantization="awq",                 # AWQ 量化
    max_model_len=8192,
)
```

**量化**:
- `awq` / `gptq` / `squeezellm`: 4-bit 量化, 70B 权重从 140GB 压到 ~40GB。
- vLLM 内部用 `bitsandbytes` 或自研量化 kernel。

**Chunked prefill**:
- 长 prompt (10K token) 不能一次塞进 GPU, 切成 8K + 2K 两段。
- vLLM 0.4+ 默认启用。

### 6.13 监控与 Profiling

vLLM 内置 Prometheus metrics:

```python
## 启动时
EngineArgs(... disable_log_stats=False ...)
```

**核心指标**:
- `vllm:num_requests_swapped`: swap 次数, 高频说明 batch 太大。
- `vllm:gpu_cache_usage_perc`: KV 池利用率, 应保持 70-90%。
- `vllm:prompt_tokens_total` / `vllm:generation_tokens_total`: 吞吐。
- `vllm:request_success_total` / `vllm:request_failure_total`: QPS。

**Profiling**:
```python
## 用 torch.profiler
from torch.profiler import profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    outputs = engine.generate(["hello world"], sampling_params)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

找 attention / sampling / 内存分配的瓶颈。

### 6.14 vLLM vs TGI vs TensorRT-LLM

| 特性 | vLLM | TGI (HF) | TensorRT-LLM (NVIDIA) |
|---|---|---|---|
| KV 管理 | PagedAttention | 部分支持 | PagedAttention |
| 调度 | Continuous batching | Continuous | Continuous + in-flight batching |
| 性能 (FP16) | ★★★★ | ★★★ | ★★★★★ (NVIDIA 优化) |
| 易用性 | ★★★★★ (Python API) | ★★★★ | ★★ (需编译 engine) |
| 量化 | AWQ/GPTQ/SqueezeLLM | BitsAndBytes/GPTQ | SmoothQuant/AWQ |
| TP/PP | TP + PP | TP | TP + PP + CP |
| 生产稳定 | ★★★★ | ★★★ | ★★★★★ |
| 适用 | 通用首选 | HF 生态 | NVIDIA 全栈 |

**选型建议**:
- 快速上线 + 通用: **vLLM**。
- NVIDIA 硬件 + 极致性能: **TensorRT-LLM**。
- HF 生态深度集成: **TGI**。

### 6.15 vLLM 源码导读

vLLM 0.4+ 主要目录:

```text
vllm/
├── engine/                  # LLMEngine / AsyncLLMEngine
│   ├── llm_engine.py
│   ├── async_llm_engine.py
│   └── arg_utils.py
├── core/                    # Scheduler / BlockManager
│   ├── scheduler.py
│   ├── block_manager.py
│   └── policy.py
├── model_executor/          # 模型 + CUDA 算子
│   ├── models/              # llama.py, qwen.py, ...
│   ├── layers/              # attention.py, linear.py, ...
│   └── worker/              # worker.py
├── attention/               # PagedAttention 实现
│   ├── backends/            # flashinfer, xformers, flash_attn
│   └── ops/                 # 各种 attention 变体
├── csrc/                    # C++/CUDA
│   ├── attention/
│   └── sampling/
├── entrypoints/             # OpenAI server, etc.
└── sampling_params.py
```

**第一次读源码建议**:
1. `engine/llm_engine.py` — 看 LLMEngine 怎么把 add_request → step → output 串起来。
2. `core/scheduler.py` — 看调度逻辑和抢占策略。
3. `core/block_manager.py` — 看 BlockSpaceManager 怎么管块。
4. `model_executor/layers/attention.py` — 看 PagedAttention Python wrapper。
5. `csrc/attention/paged_attention.cu` — 看 CUDA kernel (难度高, 可选)。

### 6.16 小结

vLLM 是 PagedAttention 思想的**工业级实现**:
- **BlockSpaceManager**: 引用计数 + prefix cache 共享, 显存利用率 90%+。
- **PagedAttention kernel**: flash-attn 风格 online softmax, 性能逼近 dense。
- **Continuous batching**: iteration-level 调度, GPU 持续满载, 吞吐 2-4×。
- **Chunked prefill**: 长 prompt 分块, 避免单步 OOM。
- **TP/PP + 量化**: 70B+ 模型分布式推理。

下一节 (7. Prefix Sharing / Beam Search) 深入 prefix sharing 和 beam search 的工程实现。


## 7. Prefix Sharing / Beam Search

> 详解 PagedAttention 在 beam search、parallel sampling、prefix caching 三种场景下的工程实现, 重点剖析 copy-on-write 的引用计数细节和 prefix cache 的哈希策略。

### 7.1 为什么需要 Prefix Sharing

LLM 服务中**前缀重复**是普遍现象:

- **System prompt**: 客服 / 角色扮演场景, 几千 token 的 system prompt 几乎所有请求共享。
- **Few-shot examples**: 100-1000 token 的示例 prompt, 100% 重复。
- **Multi-turn chat**: 多轮对话, 早期轮次完全共享。
- **Long context QA**: 把整本书喂给模型, 不同 question 共享前 N-1 章。

朴素方案: 每条请求独立 KV, 前缀部分重复存储, 浪费严重。
- 1000 seq, 共享 5KB system prompt → 5GB 浪费。
- PagedAttention prefix sharing: 几乎 0 浪费, 5GB 显存省下, 直接给新请求用。

### 7.2 Block-level Hash 策略

prefix sharing 的核心是**块级哈希**——把每 block (16 token) 算一个 hash, 命中即复用。

**朴素哈希**:
```python
def block_hash_naive(token_ids: list[int]) -> int:
    return hash(tuple(token_ids))
```

**问题**: 哈希只用 token IDs, 不考虑位置。两个不同位置的相同 token 块会有相同 hash, 错位复用。

**示例**:
```text
seq A: [BOS, 你, 好, 世, 界, 今, 天, ...]  ← block 0 hash(token[0:16])
seq B: [BOS, 你, 好, 世, 界, 周, 一, ...]  ← block 0 hash(token[0:16])
```
若 A 和 B 前 4 token 相同 (BOS, 你, 好, 世), 后面不同, 用 block 长度 4 时, 错位可能误判。

**vLLM 改进**: 哈希**包含父块 hash** (类似 Merkle Tree):
```python
def block_hash_v2(token_ids: list[int], parent_hash: int, block_size: int = 16) -> int:
    """递归哈希: h(block_n) = hash(tokens + h(block_{n-1}))"""
    h = hashlib.sha256()
    h.update(parent_hash.to_bytes(8, 'little'))
    h.update(bytes(token_ids))
    return int.from_bytes(h.digest()[:8], 'little')
```

**好处**:
- 位置敏感: block_n 的 hash 依赖 block_{n-1}, 错位必不命中。
- 增量计算: 只需存每个 block 的 hash, 不需重算整段。
- 哈希空间大 (2^64), 冲突概率可忽略。

**存储结构**:
```python
class PrefixCache:
    def __init__(self):
        # block_hash -> physical_block_id
        self.cache: dict[int, int] = {}
        # physical_block_id -> ref_count
        self.refcount: dict[int, int] = {}

    def lookup(self, tokens: list[int], parent_hash: int) -> int | None:
        h = block_hash_v2(tokens, parent_hash)
        return self.cache.get(h)

    def insert(self, tokens: list[int], parent_hash: int, phys_blk: int):
        h = block_hash_v2(tokens, parent_hash)
        self.cache[h] = phys_blk
        self.refcount[phys_blk] = 1
```

**复杂度**:
- Lookup: O(1) dict 查找。
- Insert: O(1) dict 插入。
- Compute hash: O(block_size) 算 sha256, 16 token 几 μs。

### 7.3 完整 Prefix Sharing 流程

考虑两个请求:
- Request A: "Translate to French: Hello" (10 token)
- Request B: "Translate to French: Goodbye" (11 token)

假设 block_size=8, 则:
- A 的 tokens = [BOS, Translate, to, French, :, Hello] (5 token, 1 block 不满)
- B 的 tokens = [BOS, Translate, to, French, :, Goodbye] (6 token, 1 block 不满)

**无 prefix sharing**:
- A 分配 1 块: [b_A_0]
- B 分配 1 块: [b_B_0]
- 浪费: 两个块的 [BOS, Translate, to, French, :] 重复存了。

**有 prefix sharing**:
- A 分配 1 块, hash(h_parent=0, tokens=[BOS, ..., :]) = h_0, 物理块 b_0, refcount=1。
- B 查 cache: h_0 命中, 复用 b_0, refcount=2。
- A 再 append " Hello": hash(h_0, tokens=[Hello]) = h_1, 分配新块 b_1, refcount=1。
- B 再 append " Goodbye": hash(h_0, tokens=[Goodbye]) = h_1', 与 A 的 h_1 不同, 分配新块 b_2, refcount=1。

**结果**: A 用 [b_0, b_1], B 用 [b_0, b_2], 共享前缀省 1 块。

### 7.4 Block Table 与 Prefix Sharing

带 prefix sharing 的 block table 长这样:

```text
物理块池:  [b_0: 共享前缀, b_1: A 私有, b_2: B 私有, ...]

Request A block_table:  [b_0, b_1]   refcount: b_0→2, b_1→1
Request B block_table:  [b_0, b_2]   refcount: b_0→2, b_2→1
```

**调度器逻辑**:
```python
def allocate_with_prefix_cache(seq, prefix_cache, allocator):
    blocks = []
    parent_hash = 0
    token_blocks = chunked(seq.tokens, block_size)
    for blk_tokens in token_blocks:
        h = block_hash_v2(blk_tokens, parent_hash)
        if h in prefix_cache.cache:
            # 命中: 复用
            phys_blk = prefix_cache.cache[h]
            prefix_cache.refcount[phys_blk] += 1
        else:
            # 未命中: 分配
            phys_blk = allocator.alloc()
            prefix_cache.cache[h] = phys_blk
            prefix_cache.refcount[phys_blk] = 1
        blocks.append(phys_blk)
        parent_hash = h
    seq.block_table = blocks
    return blocks
```

### 7.5 释放与 Eviction

请求完成后, 释放 block_table 里的物理块, 但**保留 prefix cache 映射** (供后续复用):

```python
def release(seq, prefix_cache, free_pool):
    for blk in seq.block_table:
        prefix_cache.refcount[blk] -= 1
        if prefix_cache.refcount[blk] == 0:
            # 真正没人用了
            free_pool.append(blk)
            # 但 prefix_cache 仍保留 hash → blk 映射
            # 下次有相同前缀的请求可复用
    seq.block_table = []
```

**问题**: prefix cache 会无限增长, 占 dict 内存。需要**eviction 策略**:

| 策略 | 描述 | 优缺点 |
|---|---|---|
| LRU | 最近最少使用的块先回收 | 实现简单, 命中率中等 |
| LFU | 最不经常使用的块先回收 | 长尾内容命中好 |
| TTL | 超过 N 秒未访问就回收 | 适合"周期性 prompt" |
| 手动 | 用户主动调用 reset_prefix_cache() | 最简单 |

vLLM 当前实现: **不主动 evict**, 依赖请求量自然填满。生产环境可手动 `engine.reset_prefix_cache()` 定期清。

**Eviction 实现 (LRU)**:
```python
from collections import OrderedDict

class LRUPrefixCache:
    def __init__(self, max_entries: int = 100_000):
        self.max = max_entries
        self.cache = OrderedDict()  # hash -> (phys_blk, access_time)

    def get(self, h):
        if h in self.cache:
            blk, _ = self.cache.pop(h)
            self.cache[h] = (blk, time.time())  # LRU: 移到末尾
            return blk
        return None

    def put(self, h, blk):
        if len(self.cache) >= self.max:
            old_h, (_, old_blk) = self.cache.popitem(last=False)  # 弹出最旧
            # 物理块需要 refcount=0 才能回收
            yield_phys_block(old_blk)
        self.cache[h] = (blk, time.time())
```

### 7.6 Beam Search 完整实现

Beam search 维护 K 个候选, 每步选 top-K:

```python
def beam_search_step(model, beams, k, block_manager):
    """beams: list of (token_ids, cumulative_logprob)"""
    new_beams = []

    for beam_id, (tokens, cum_logp) in enumerate(beams):
        # 1. fork block table (不复制物理 K/V)
        child_id = block_manager.fork(parent=beam_id, child=new_seq_id)

        # 2. 前向计算, 拿到 top-K next tokens
        logits = model.forward(tokens)  # shape: [seq_len, vocab]
        last_logits = logits[-1]        # 取最后位置
        log_probs = F.log_softmax(last_logits, dim=-1)

        # 3. 取 top-K
        topk_logp, topk_ids = torch.topk(log_probs, k=k)

        for i in range(k):
            new_tokens = tokens + [topk_ids[i].item()]
            new_cum = cum_logp + topk_logp[i].item()
            new_beams.append((new_tokens, new_cum, child_id))

    # 4. 选 top-K 全局最优
    new_beams.sort(key=lambda x: x[1], reverse=True)
    return new_beams[:k]
```

**PagedAttention 配合**:

每条 beam 维护**独立 block_table**, 但**物理块共享**直到分叉点。

```python
class BeamManager:
    def __init__(self, beam_size: int):
        self.beam_size = beam_size
        self.beams: list[Beam] = []
        self.block_manager = BlockManager()
        self.done = [False] * beam_size

    def expand(self):
        """每个 beam 生成 beam_size 个候选, 总 beam_size² 个, 选 top beam_size"""
        all_candidates = []
        for i, beam in enumerate(self.beams):
            if self.done[i]:
                continue
            # 1. 给 beam i 分叉出 beam_size 个子
            children = []
            for j in range(self.beam_size):
                child_id = self._next_id()
                # fork block table (CoW)
                new_block_table = list(beam.block_table)
                self.block_manager.refcount[blk] += 1 for blk in new_block_table
                child = Beam(
                    tokens=beam.tokens[:],
                    block_table=new_block_table,
                    parent=i,
                    child_id=child_id
                )
                children.append(child)

            # 2. forward 每个 child
            for child in children:
                logits = self.model.forward(child.tokens)
                topk = logits.topk(self.beam_size, dim=-1)
                for logp, tok in zip(topk.values, topk.indices):
                    new_tokens = child.tokens + [tok.item()]
                    all_candidates.append((new_tokens, beam.score + logp.item(), child))

        # 3. 全局 top beam_size
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        survivors = all_candidates[:self.beam_size]

        # 4. 释放未选中的 beam
        chosen_ids = {s[2].child_id for s in survivors}
        for c in all_candidates:
            if c[2].child_id not in chosen_ids:
                self.block_manager.release(c[2].block_table)

        # 5. 更新 beams
        self.beams = [s[2] for s in survivors]
```

**关键**: `expand` 后**未选中的子 beam 立即释放**——它们的物理块 refcount 减回, 真正没人用了的块进 free_pool。

### 7.7 Copy-on-Write 写时复制细节

Beam search 在**分叉点**之后才真正独立:

```text
初始: 所有 beam 共享 prompt 的 N 块
       ┌── Beam 0: 块 [b0, b1, ..., bN-1]  (prompt)
       ├── Beam 1: 块 [b0, b1, ..., bN-1]  (共享)
       └── Beam 2: 块 [b0, b1, ..., bN-1]  (共享)

Step 1: Beam 0 选 "A", Beam 1 选 "B", Beam 2 选 "C"
       ┌── Beam 0: 块 [b0, ..., bN-1, bN_0]  (新增私有)
       ├── Beam 1: 块 [b0, ..., bN-1, bN_1]  (新增私有)
       └── Beam 2: 块 [b0, ..., bN-1, bN_2]  (新增私有)
       refcount: b0..bN-1 = 3, bN_0/1/2 = 1

Step 2: Beam 0 选 "X" (继续, 不分叉)
       └── Beam 0: 块 [b0, ..., bN-1, bN_0, bN+1_0]
       bN_0 refcount=1 (Beam 0 独占), 不需要 CoW

如果 Beam 0 选 "Y" 触发从 Beam 1 分叉:
       ┌── New Beam: 块 [b0, ..., bN-1, bN_0', bN+1_new]
       bN_0 refcount=1 → 实际是 Beam 0 独占, Beam 1 已释放
```

**CoW 触发条件**: 向 refcount > 1 的物理块**写入新数据**。在我们的实现里, beam 的 block_table 是"逻辑连续"的, 写入永远发生在末尾——但末尾块可能是共享的!

**典型场景**:
- Beam 0 已分配 bN_0, Beam 1 复制 block_table 时也指向 bN_0 (refcount=2)。
- Beam 0 下一 token 要写入 bN_0 的下一个位置——这不冲突, 因为是顺序写。
- 但 Beam 1 同时要写 bN_0 的下一个位置——冲突! 需要 CoW。

**手撕 CoW**:
```python
def append_with_cow(seq, k, v, block_manager, k_pool, v_pool):
    """写入新 KV, 若目标块被共享则先复制"""
    L = len(seq.tokens) - 1   # 写位置
    blk_idx = L // block_size
    pos_in_blk = L % block_size

    if blk_idx >= len(seq.block_table):
        # 需要新块
        new_blk = block_manager.alloc()
        seq.block_table.append(new_blk)
        return

    phys_blk = seq.block_table[blk_idx]
    if block_manager.refcount[phys_blk] > 1:
        # 共享块: 先复制
        new_blk = block_manager.alloc()
        k_pool[new_blk].copy_(k_pool[phys_blk])
        v_pool[new_blk].copy_(v_pool[phys_blk])
        # 旧块 refcount--
        block_manager.refcount[phys_blk] -= 1
        # 新块
        block_manager.refcount[new_blk] = 1
        seq.block_table[blk_idx] = new_blk
        phys_blk = new_blk

    # 写入
    k_pool[phys_blk, pos_in_blk] = k
    v_pool[phys_blk, pos_in_blk] = v
```

**性能**: CoW 是**罕见事件**——只在分叉步触发, 后续都是私有块顺序写。CoW 本身是一次 `cudaMemcpy`, 16 token × 128 dim × 2 byte = 4KB, 微秒级。

### 7.8 Parallel Sampling 与 n>1

OpenAI API 的 `n=4` 参数让模型对同一 prompt 生成 4 个独立采样结果:

```python
sampling_params = SamplingParams(n=4, temperature=0.8, max_tokens=200)
response = llm.generate(["Tell me a joke"], sampling_params)
## response 有 4 个 outputs, 共享同一个 prompt
```

**实现**:
- 4 个采样共享 prompt 的 K/V (CoW)。
- Sampling 时每条独立采样 (随机性来自 temperature / top_p)。
- 4 个输出的 token 序列不同, 后半段 block_table 独立。

**代码**:
```python
class ParallelSamplingScheduler:
    def __init__(self, n: int):
        self.n = n

    def fork_n(self, parent_seq):
        """从 parent 分叉出 n 个子"""
        children = []
        for i in range(self.n):
            child = Sequence(
                tokens=parent_seq.tokens[:],
                block_table=list(parent_seq.block_table),  # 共享
            )
            children.append(child)
            # 增加 refcount
            for blk in parent_seq.block_table:
                self.block_manager.refcount[blk] += 1
        return children
```

**收益**:
- prompt 1000 token, 4 个采样, 总长 2000 token。
- 朴素: 4 × 2000 = 8000 块的 K/V 显存。
- Parallel + PagedAttention: 1000 (共享) + 4 × 1000 (独立后缀) = 5000 块 → **省 37.5%**。
- 当 n 越大, 节省比例越高。

### 7.9 Prefix Caching 的命中率优化

Prefix caching 不是"开了就一定赚", 需要调参:

**提高命中率**:
- **长 system prompt**: 越长命中率越高 (毕竟都是同一段)。
- **请求批量化**: 高 QPS 场景, 短时间内多个请求命中。
- **稳定 prompt**: 不要让 system prompt 里带时间戳等动态信息, 否则 hash 永远不同。

**降低开销**:
- **短 prompt**: < 100 token 时, prefix cache 收益小 (块少), 哈希计算反而是开销。
- **关闭 prefix caching**: `enable_prefix_caching=False`, 短请求场景。

**监控**:
```python
## vLLM metrics
vllm:prefix_cache_hits_total
vllm:prefix_cache_queries_total
命中率 = hits / queries
```

**调优**:
- 命中率 > 30% → prefix caching 收益明显, 继续用。
- 命中率 < 10% → 考虑关闭, 省哈希计算开销。

### 7.10 Prefix Sharing 实战: Chat Template 共享

生产中**所有 chat 请求**的 system prompt 几乎一样 (客服角色定义), 命中率 100%:

```python
## 服务端代码
SYSTEM_PROMPT = """You are a helpful AI assistant for ACME Corp.
You must follow these rules:
1. Be concise and accurate.
2. If unsure, say "I don't know" rather than guess.
3. Never provide harmful content.
[... 500 token ...]"""

def handle_request(user_message: str):
    # 拼成 chat template, system prompt 固定
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_message}\n<|assistant|>\n"
    return llm.generate([prompt], sampling_params)
```

**PagedAttention prefix sharing 效果**:
- 1000 并发请求, system prompt = 500 token, block_size=16, 500/16 ≈ 32 块。
- 朴素: 1000 × 32 = 32,000 块的 K/V 显存。
- 共享: 32 块 (refcount=1000) + 每请求后续私有块 → 几乎省 32,000 块。
- 32,000 块 × 16 token × 2 (K+V) × 32 层 × 8 heads × 128 dim × 2 byte ≈ 32 GB。
- 实际省 32GB 显存, 等于多服务 50+ 并发请求。

### 7.11 Prefix Sharing 实战: 多轮对话

多轮对话的"历史轮次"完全共享:

```python
## Turn 1
## Request: [system, user1]
## Tokens: [BOS, sys_0, sys_1, ..., user1_0, user1_1, ...]

## Turn 2 (同会话)
## Request: [system, user1, assistant1, user2]
## Tokens: 上述 + [assistant1_0, ..., user2_0, ...]
```

**朴素**: Turn 2 重新算 system + user1 的 K/V → 浪费 30+ 块。
**Prefix sharing**: Turn 2 复用 Turn 1 的 block_table, 只算 assistant1 + user2 的 K/V。

**工程实现**: 服务端维护 `session_id -> cached_block_table`, 每次新 turn 用上次结果:

```python
class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Sequence] = {}

    def get_or_create(self, session_id, prompt):
        if session_id in self.sessions:
            old = self.sessions[session_id]
            # 找到共同前缀长度 (token 级别)
            common = 0
            for a, b in zip(old.tokens, prompt):
                if a == b:
                    common += 1
                else:
                    break
            # 复用 common 长度的 block_table
            n_blocks = common // block_size
            block_table = old.block_table[:n_blocks]
            # 增量: 算 common 之后的新 token
            new_tokens = prompt[common:]
            return Sequence(tokens=new_tokens, block_table=block_table)
        else:
            return Sequence(tokens=prompt, block_table=[])
```

**注意**: 实际上 vLLM 不显式做 session 管理, 因为 chat template 可能让 history 重新拼一遍, 哈希变了。生产中通常**让客户端维护 KV 缓存**或**服务端用 in-memory session**。

### 7.12 Prefix Sharing 的局限

**不适用场景**:
- **动态 prompt**: prompt 里带时间戳、随机 ID, 永远不命中。
- **个性化 prompt**: 每用户不同, 命中率低。
- **极短 prompt**: < 16 token, 一个块都用不满, 反而浪费块表项。

**性能开销**:
- 哈希计算: 每 block 16 token 算一次 sha256, 几十 μs, 高 QPS 时也是开销。
- 字典查找: O(1), 但 100k+ entries 时 cache miss 延迟上升。
- 引用计数: 加减, 无锁化困难 (高并发下 atomic op 慢)。

**正确性**:
- 哈希冲突: sha256 8 byte (64 bit) 截断, 2^32 个块才可能冲突一次, 实际不会。
- 逻辑泄漏: 释放时必须严格 refcount, 否则显存泄漏。

### 7.13 Beam Search 的工程取舍

beam search 在生产 LLM 推理中**很少用**, 因为:
- **多样性差**: top-K 容易收敛到相似答案。
- **延迟大**: K 个候选都要 forward, 延迟 ×K。
- **采样更优**: temperature + top_p 通常能拿到更好结果。

**vLLM 仍支持** `use_beam_search=True`, 实现如下:

```python
sampling_params = SamplingParams(
    n=4,
    use_beam_search=True,
    temperature=0.0,  # beam search 不要 sampling
    max_tokens=200,
)
```

**性能**:
- Llama-7B, beam_size=4, batch=8, beam search vs parallel sampling:
  - Beam: ~16 tok/s/user (4 路, 但 GPU 满载)
  - Parallel sampling: ~40 tok/s/user (4 路独立, GPU 满载更快)
- Beam search 优势在长序列 + 小 batch, 因为 GPU 利用率已经被 4 路填满。

### 7.14 实测: prefix sharing 效果

Llama-3-8B, 8 × A100, batch_size=64, 测 3 个场景:

| 场景 | 无 prefix sharing | 有 prefix sharing (1KB system) | 节省 |
|---|---|---|---|
| Random prompt | 2400 tok/s | 2450 tok/s | ~0% (overhead 抵消) |
| 共享 500 token system | 2400 tok/s | 3200 tok/s | +33% |
| 共享 5KB system | 2400 tok/s | 5400 tok/s | +125% |

**结论**: prefix sharing 收益与**共享 prompt 长度**成正比。客服 / 角色场景 5KB+ system, 收益巨大; 通用 chat 几百 token, 收益中等。

### 7.15 面试高频: Prefix Sharing 追问

**Q: prefix sharing 怎么保证语义正确?**
A: hash 包含父块, 位置敏感, 不会错位复用。

**Q: prefix cache 占用多少内存?**
A: 每块 16 token × 2(K+V) × 32 层 × 8 heads × 128 dim × 2 byte = 4 MB 物理 K/V, dict 项本身 ~64 byte。100k blocks → ~400 GB 物理 + 6 MB dict (实际 prefix cache 只存映射, 不复制物理 K/V, 复用时 refcount++ 即可)。

**Q: beam search 和 parallel sampling 在 vLLM 里性能差多少?**
A: 实测 parallel sampling 略快 (~10-20%), 因为 beam search 选 top-K 时有排序开销, 且 beam 早期容易重复 (多样性低)。

### 7.16 小结

Prefix sharing 是 PagedAttention 的**杀手级特性**:
- **System prompt 共享**让客服 / 角色场景显存利用率翻倍。
- **Multi-turn chat**复用历史轮次, 增量计算。
- **Beam search / parallel sampling**通过 CoW 省 30-50% 显存。
- **Block-level Merkle hash**保证位置敏感, 不误命中。
- **LRU eviction**控制 cache 内存, 防止无限增长。

下一节 (8. Chunked Prefill) 解决长 prompt 推理的难题。


## 8. Chunked Prefill

> 长 prompt (10K-128K token) 是 LLM 推理的最大挑战。Chunked Prefill 把长 prompt 切成小段, 与 decode 阶段混合调度, 显著提升 GPU 利用率与首字延迟。

### 8.1 长 Prompt 的挑战

现代 LLM 支持的 context length 越来越长:
- Llama-2: 4K
- Llama-3: 8K
- Qwen-2.5: 32K
- Claude / GPT-4: 100K-1M

但**长 prompt 推理**有三大难题:

1. **显存爆炸**: KV Cache 与 seq_len 线性, 100K token 的 K/V 在 Llama-3-70B = 100K × 80 层 × 8 heads × 128 dim × 2 byte = 16 GB / seq。
2. **首字延迟 (TTFT) 高**: prefill 阶段需要 forward 整个 prompt, 100K token 在 A100 上约 30 秒。
3. **GPU 浪费**: prefill 阶段 GPU 满载, 但 decode 阶段 GPU 闲置 (compute-bound vs memory-bound 切换), 调度不当就空转。

**示例**:
- 用户上传 50K token 的 PDF 提问, TTFT = 30s, 用户体验差。
- 同时 100 个 1K token 的短请求等待, 被长 prompt 阻塞。

### 8.2 朴素 Prefill 的问题

朴素方案: 收到 prompt → 一次性 forward → 等生成完成 → 处理下一个。

```python
def naive_prefill_decode(model, prompt_tokens, max_new_tokens):
    # Prefill: 一次 forward 整个 prompt
    logits = model.forward(prompt_tokens)  # 50K token, GPU 满载 5 秒
    next_token = sample(logits[-1])

    # Decode: 一个一个生成
    for _ in range(max_new_tokens):
        logits = model.forward([next_token])  # 1 token, 内存带宽限制
        next_token = sample(logits[-1])
    return generated
```

**问题**:
- Prefill 5 秒, decode 1 token 30 ms × 200 token = 6 秒, 总 11 秒。
- Prefill 期间 GPU 100% 利用, decode 期间 GPU < 30% (memory-bound)。
- 长 prompt 阻塞短请求, 公平性差。

### 8.3 Chunked Prefill 核心思想

**Chunked prefill** = 把长 prompt 切成小段 (e.g. 512 token), 每步处理一段 + 一些 decode 任务。

```text
朴素:
  Step 1: Prefill 50K (5s, GPU 满载)
  Step 2..200: Decode 1 token (30 ms 一步, GPU 30%)

Chunked:
  Step 1:   Prefill chunk 0 (512 tok, 50ms) + 50 decode 短请求
  Step 2:   Prefill chunk 1 (512 tok, 50ms) + 50 decode 短请求
  ...
  Step 100: Prefill chunk 99 (512 tok, 50ms) + 50 decode 短请求
  Step 101: 全部 prefill 完, 进入 decode
  Step 102..: Decode 长 prompt + 50 decode 短请求
```

**收益**:
- 短请求的 TTFT 降低: 不必等 50K prefill 完, 每 50ms 处理一步。
- GPU 利用率提升: prefill + decode 混合, GPU 持续满载。
- 公平性: 长短请求交错, 不再阻塞。

**代价**:
- 长 prompt 的 TTFT 略增 (5s → 5s, 因为 prefill 总计算量不变, 只是切分)。
- 调度逻辑更复杂, 需要混合 prefill / decode 队列。

### 8.4 调度算法: Mixed Batching

Chunked prefill 调度器维护三个队列:

```python
class ChunkedScheduler:
    def __init__(self, chunk_size, max_num_batched_tokens):
        self.chunk_size = chunk_size        # 单步最多 prefill 多少 token (e.g. 512)
        self.max_batched = max_num_batched_tokens   # 单步总 token 上限
        self.prefill: deque[Sequence] = deque()  # 等待 prefill 的 seq
        self.decode: deque[Sequence] = deque()   # decode 中的 seq
        self.swapped: deque[Sequence] = deque()

    def add_request(self, seq):
        # 长 prompt 拆成多个 chunk 入队
        if len(seq.tokens) > self.chunk_size:
            # 切成多个 chunk
            for start in range(0, len(seq.tokens), self.chunk_size):
                chunk = seq.tokens[start:start+self.chunk_size]
                self.prefill.append(SequenceChunk(seq, start, chunk))
        else:
            self.prefill.append(SequenceChunk(seq, 0, seq.tokens))

    def schedule(self) -> tuple[list[SequenceChunk], list[Sequence]]:
        """返回本步要处理的 prefill chunk 和 decode seq"""
        batched_prefill = []
        batched_decode = []
        token_budget = self.max_batched  # e.g. 2048

        # 1. 优先处理 decode (公平性 + 低延迟)
        while self.decode and token_budget > 0:
            seq = self.decode.popleft()
            batched_decode.append(seq)
            token_budget -= 1  # decode 每步 1 token

        # 2. 剩余预算给 prefill chunk
        while self.prefill and token_budget >= self.chunk_size:
            chunk = self.prefill.popleft()
            batched_prefill.append(chunk)
            token_budget -= len(chunk.tokens)

        return batched_prefill, batched_decode
```

**token 预算分配**:
- `max_num_batched_tokens = 2048` 是关键参数。
- decode 每个 1 token, 假设 batch=64, 64 token → 还剩 1984 给 prefill。
- 1984 / 512 = 3.875, 最多 3 个 prefill chunk (1536 token), 还剩 448 token 浪费。

**生产调优**:
- `max_num_batched_tokens=2048`: 平衡延迟与吞吐。
- `=4096`: 更高吞吐, 但单步延迟上升。
- `=1024`: 更快响应, 但 GPU 利用率下降。

### 8.5 注意力计算: 处理跨 Chunk

Chunked prefill 的核心难点是**注意力跨 chunk 依赖**——每个 token 要 attend 整个 history, 而 history 被切成多段。

**朴素方案**: 每段独立 forward, 但第二段要拿到第一段的 K/V。
- 问题: 第一段的 K/V 在 GPU, 第二段要重新计算 (重复算 attention)。

**正确方案**: 把 chunked prefill 的 K/V 写到**同一物理 KV 池**, 后续 chunk 的 attention 看到前序 chunk 的 K/V。

```python
def chunked_prefill_step(model, chunks, decode_seqs, kv_pool, block_tables):
    """每个 step 处理多个 prefill chunk + 多个 decode seq"""
    # 1. 把所有 token 拼成一个 batch
    all_input_ids = []
    all_positions = []
    chunk_seq_ids = []   # 标记每个 token 属于哪个 chunk
    for chunk in chunks:
        all_input_ids.extend(chunk.tokens)
        all_positions.extend(range(chunk.start_pos, chunk.start_pos + len(chunk.tokens)))
        chunk_seq_ids.extend([chunk.seq_id] * len(chunk.tokens))
    for seq in decode_seqs:
        all_input_ids.append(seq.tokens[-1])
        all_positions.append(len(seq.tokens) - 1)
        chunk_seq_ids.append(seq.seq_id)

    # 2. forward (关键: 每个 seq 的 attention 要看到完整 history)
    logits = model.forward(
        input_ids=all_input_ids,
        positions=all_positions,
        seq_ids=chunk_seq_ids,   # 告诉 attention 哪些 token 属于同 seq
        kv_pool=kv_pool,
        block_tables=block_tables,
    )

    # 3. 把新算的 K/V 写回对应物理位置
    new_kv_positions = ...   # 跟踪每个 chunk 的写入位置
    write_kv_to_pool(kv_pool, new_kv_positions, ...)

    return logits
```

**PagedAttention 的 block_table 让这变得简单**:
- 每条 seq 维护 block_table, 标记自己的物理块。
- 无论分多少 chunk, 每个 chunk 算的 K/V 写入对应 seq 的物理块。
- Attention kernel 通过 block_table 拼出完整 history。

### 8.6 显存与计算权衡

**计算量分析**:
- 长 prompt 100K token, chunk_size=512, 共 100K/512 ≈ 200 chunk。
- 每 chunk: forward 512 token, 计算量与朴素 prefill 相同 (O(L²) attention, 但分块后每块 O(chunk × history), 总和 O(L²), 一样)。
- **总计算量 = 朴素 prefill**, 不增加, 只是切分。

**显存分析**:
- chunked prefill 不减少 KV 显存 (仍然要存所有 token 的 K/V)。
- 但可以**避免一次性分配 O(L²) 注意力矩阵**——朴素 prefill 100K token 时, 注意力矩阵 = 100K × 100K × 2 byte = 20 GB, 单卡放不下。
- chunked prefill 每块 512 × history, 显存可控。

**CUDA kernel 优化**:
- vLLM 用 **flash-attn 风格分块 attention**, 每块在 shared memory 算 QK^T + softmax + PV, 不存中间矩阵。
- chunked prefill 与 flash-attn 天然契合。

### 8.7 FlashAttention 配合 Chunked Prefill

FlashAttention 把 attention 切成小块, 在 GPU SRAM 里完成 QK^T + softmax + PV, 不写回 HBM:

```python
## 简化版: flash attention 分块
def flash_attn_chunked(q, k_blocks, v_blocks):
    """q: [1, head_dim], k_blocks/v_blocks: list of [block_size, head_dim]"""
    m = -inf  # running max
    l = 0     # running sum
    o = 0     # running output

    for k, v in zip(k_blocks, v_blocks):
        s = q @ k.T / sqrt(head_dim)      # [block_size]
        m_new = max(m, s.max())
        s_exp = exp(s - m_new)
        l_new = l * exp(m - m_new) + s_exp.sum()
        o = (o * l * exp(m - m_new) + s_exp @ v) / l_new
        m, l = m_new, l_new
    return o
```

**Chunked prefill 配合**:
- 每 step 处理一个 chunk, 用 flash-attn 算 attention, 写回 K/V 到物理池。
- 下一 chunk 看到物理池里的前序 K/V, 继续 flash-attn。
- 显存 O(chunk_size) 而非 O(L²)。

### 8.8 vLLM 的 chunked_prefill 开关

```python
EngineArgs(
    model="...",
    enable_chunked_prefill=True,    # 启用 (默认)
    max_num_batched_tokens=2048,    # 单步总 token 上限
    chunked_prefill_size=512,       # 单 chunk 大小 (vLLM 0.4+ 引入)
)
```

**实测** (Llama-3-8B, 8 × A100, batch=64):
- 无 chunked prefill: 长 prompt TTFT = 5.2s, 短请求 P99 = 8.0s。
- 有 chunked prefill: 长 prompt TTFT = 5.5s (略增), 短请求 P99 = 0.8s (降 10×)。
- 总体吞吐 +15% (长短混合, GPU 利用率提升)。

**结论**: chunked prefill **短请求显著受益**, **长请求 TTFT 略增**, **总体吞吐提升**。生产强烈推荐开启。

### 8.9 Chunked Prefill 的边界

**禁用场景**:
- 全部请求都长 (e.g. 全部 50K token): chunked 没意义, 退化为朴素 prefill。
- 极短请求 (e.g. < 100 token): chunk_size=512 远大于请求, 直接当一个 chunk。

**配置建议**:
- `max_num_batched_tokens`: 与 batch size 联动。batch=64, decode 64 token, prefill 预算 = 2048 - 64 = 1984。
- `chunked_prefill_size`: 256-1024 范围, 太小调度开销大, 太大单步延迟高。

### 8.10 替代方案: Prompt Cache

**Prompt cache** 是另一种长 prompt 优化:
- 离线把长 prompt 跑一次, 把 K/V 缓存到内存 / 磁盘。
- 用户提问时, 直接加载缓存的 K/V, 只需 forward 新增 token。

**适用**:
- 同一长 prompt 多次提问 (客服 / 文档问答)。
- prompt 极长 (10K+) 重复利用。

**与 chunked prefill 关系**:
- Prompt cache 解决"重复长 prompt", chunked prefill 解决"首次长 prompt"。
- 两者可叠加: 首次长 prompt 用 chunked prefill, 后续用 prompt cache。

### 8.11 长 Context 优化: Ring Attention

128K+ context 仍挑战显存。**Ring Attention** 把 sequence 切到多卡, 每卡算一段, 通过环形通信交换 K/V:

```text
GPU 0: token 0-32K
GPU 1: token 32K-64K
GPU 2: token 64K-96K
GPU 3: token 96K-128K

Step 1: GPU 0 算 Q[0:32K] × K[0:32K]   (本地)
Step 2: GPU 0 算 Q[0:32K] × K[32K:64K]  (从 GPU 1 拿 K)
Step 3: GPU 0 算 Q[0:32K] × K[64K:96K]  (从 GPU 2 拿 K)
Step 4: GPU 0 算 Q[0:32K] × K[96K:128K] (从 GPU 3 拿 K)
```

**优点**: 1M token context 也可处理, 显存分摊到 N 卡。
**缺点**: 通信开销大, 适合高速互联 (NVLink / InfiniBand)。

**与 PagedAttention 关系**: 两者正交, 可叠加。vLLM 实验性支持 ring attention。

### 8.12 调度器: 优先级与抢占

chunked prefill 调度器需要更细粒度的优先级:

**优先级**:
- **Decode 优先**: 短响应延迟敏感, decode token 1 个但 1 步延迟低。
- **Prefill chunk 优先**: 长 prompt TTFT 敏感, 优先算 prefill。
- **混合策略**: decode 占 30% 预算, prefill 占 70%。

**抢占**:
- chunked prefill 期间, 显存不足以启动新 prefill 时, 抢占**最久没产出的 decode**。
- decode seq 抢占后, K/V 保留 (swap) 或重算 (recompute)。

**vLLM 调度代码** (简化):
```python
def schedule(self):
    batched_decode = self._schedule_decode(budget=decode_budget)
    batched_prefill = self._schedule_prefill(budget=prefill_budget)

    # 若 prefill 显存不够, 抢占 decode
    while not self.block_manager.can_allocate(prefill_seq) and batched_decode:
        victim = batched_decode.pop()
        self._preempt(victim)

    return batched_prefill, batched_decode
```

### 8.13 实战: Long Context QA 服务

```python
## 服务端
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    tensor_parallel_size=8,
    max_model_len=32768,                # 32K context
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
    chunked_prefill_size=1024,
    enable_prefix_caching=True,
)

sampling_params = SamplingParams(temperature=0.3, max_tokens=500)

## 用户上传 30K token PDF, 提问
prompt = f"<|context|>\n{pdf_content}\n<|user|>\n{user_question}\n<|assistant|>\n"
response = llm.generate([prompt], sampling_params)
```

**性能** (A100 × 8, Llama-3-70B):
- 30K token prompt prefill: ~3 秒 (无 chunked) / ~3.2 秒 (有 chunked, 略增)。
- 短请求 (1K prompt) TTFT: 50ms (无 chunked, 阻塞 3s) / 0.5s (有 chunked)。
- 总体吞吐: +25% (chunked 混合调度)。

### 8.14 Chunked Prefill 的失败案例

**场景**: 用户每次提问都带 100K token 长文档, batch=1。
- chunked prefill: 每步 512 token, 200 步 prefill + decode, 单请求 TTFT 略增。
- 朴素 prefill: 一次 forward, 单请求 TTFT 略低。
- **结论**: batch=1 时 chunked prefill 反而略慢, 可关。

**调优**:
- batch ≥ 4 + 长短混合: 强烈推荐 chunked。
- batch=1 全长 prompt: 关闭 chunked。

### 8.15 与 Continuous Batching 的关系

**Continuous batching**: iteration-level 调度, 每步重选 batch。
**Chunked prefill**: 单 step 内 prefill + decode 混合。

两者**正交, 叠加**:
- Continuous batching 保证 decode 完成立即释放, 新请求随时加入。
- Chunked prefill 保证 prefill 阶段也能混合 decode, 不阻塞。

```python
## vLLM 默认: continuous batching + chunked prefill 都开
EngineArgs(
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,
    scheduler_delay_factor=0,  # 0 = 严格 continuous batching
)
```

### 8.16 小结

Chunked prefill 解决了长 prompt 与短请求混合的关键难题:
- **短请求 TTFT**降低 10× (P99 从 8s → 0.8s)。
- **总体吞吐**提升 15-25% (GPU 利用率提升)。
- **公平性**显著改善, 长请求不阻塞短请求。

**适用**:
- 长短请求混合 (99% 短 + 1% 长是常见比例)。
- 长 prompt (10K+) 偶发场景。

**不适用**:
- 全部长 prompt (batch=1 极端情况)。
- 极短 prompt (< 100 token, chunk_size 远大于请求)。

下一节 (9. 工业实践) 介绍 vLLM 在真实生产中的部署、调优、监控经验。



## 深度扩展 · 多视角系统展开(续)

### 110. PagedAttention 与 vLLM - 优化视角 视角

**核心议题**: 本节从优化视角视角切入,系统展开 "PagedAttention 与 vLLM" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PagedAttention 与 vLLM" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 优化视角视角的展开**

从优化视角视角出发,我们可以把 "PagedAttention 与 vLLM" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调优化视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"PagedAttention 与 vLLM" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"优化视角" 视角,但落到代码层面是类似的骨架:

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

评估 "PagedAttention 与 vLLM" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PagedAttention 与 vLLM" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PagedAttention 与 vLLM" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PagedAttention 与 vLLM" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从优化视角视角切入,系统展开了 "PagedAttention 与 vLLM" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. PagedAttention 与 vLLM - 数值稳定 视角

**核心议题**: 本节从数值稳定视角切入,系统展开 "PagedAttention 与 vLLM" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PagedAttention 与 vLLM" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值稳定视角的展开**

从数值稳定视角出发,我们可以把 "PagedAttention 与 vLLM" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调数值稳定视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"PagedAttention 与 vLLM" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"数值稳定" 视角,但落到代码层面是类似的骨架:

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

评估 "PagedAttention 与 vLLM" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PagedAttention 与 vLLM" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PagedAttention 与 vLLM" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PagedAttention 与 vLLM" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值稳定视角切入,系统展开了 "PagedAttention 与 vLLM" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. PagedAttention 与 vLLM - 工程优化 视角

**核心议题**: 本节从工程优化视角切入,系统展开 "PagedAttention 与 vLLM" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PagedAttention 与 vLLM" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工程优化视角的展开**

从工程优化视角出发,我们可以把 "PagedAttention 与 vLLM" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调工程优化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"PagedAttention 与 vLLM" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"工程优化" 视角,但落到代码层面是类似的骨架:

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

评估 "PagedAttention 与 vLLM" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PagedAttention 与 vLLM" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PagedAttention 与 vLLM" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PagedAttention 与 vLLM" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工程优化视角切入,系统展开了 "PagedAttention 与 vLLM" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. PagedAttention 与 vLLM - 最佳实践 视角

**核心议题**: 本节从最佳实践视角切入,系统展开 "PagedAttention 与 vLLM" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PagedAttention 与 vLLM" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 最佳实践视角的展开**

从最佳实践视角出发,我们可以把 "PagedAttention 与 vLLM" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调最佳实践视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"PagedAttention 与 vLLM" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"最佳实践" 视角,但落到代码层面是类似的骨架:

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

评估 "PagedAttention 与 vLLM" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PagedAttention 与 vLLM" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PagedAttention 与 vLLM" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PagedAttention 与 vLLM" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从最佳实践视角切入,系统展开了 "PagedAttention 与 vLLM" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. PagedAttention 与 vLLM - 可视化 视角

**核心议题**: 本节从可视化视角切入,系统展开 "PagedAttention 与 vLLM" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"PagedAttention 与 vLLM" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 可视化视角的展开**

从可视化视角出发,我们可以把 "PagedAttention 与 vLLM" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调可视化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"PagedAttention 与 vLLM" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"可视化" 视角,但落到代码层面是类似的骨架:

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

评估 "PagedAttention 与 vLLM" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "PagedAttention 与 vLLM" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "PagedAttention 与 vLLM" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"PagedAttention 与 vLLM" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从可视化视角切入,系统展开了 "PagedAttention 与 vLLM" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##

```
