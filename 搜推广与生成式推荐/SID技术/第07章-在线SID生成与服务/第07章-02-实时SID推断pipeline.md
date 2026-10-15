---
title: 2-实时SID推断pipeline
tags:
  - 方向
  - SID
description: 物品入库 → 离线算 SID → 存到 KV → 在线直接查...
chapter_id: 7.2
last_updated: 2026-07-14
status: ✅ v1.0

---

# 2. 实时 SID 推断 pipeline

> **本章定位**: 在线 encoder 的设计与实现
> **预计时长**: 60 分钟

---

## 2.1 三种架构

### 架构 1: 全离线

```text
物品入库 → 离线算 SID → 存到 KV → 在线直接查
```

时延: 0 (查表)
局限: 不支持 QuaSID

### 架构 2: 混合

```text
物品入库 → 离线算内容 SID → 存
请求时 → 在线算 query-aware 补充
```

时延: 1-5ms
灵活: 中

### 架构 3: 全在线

```text
请求时 → 在线 encoder + RQ-VAE → SID
```

时延: 5-20ms
灵活: 高

---

## 2.2 在线 Encoder 优化

```python
# 用 TorchScript / ONNX 加速
import torch
model = ...  # 训练好的 encoder
scripted = torch.jit.script(model)
scripted.save("encoder.pt")

# 或用 TensorRT
# trtexec --onnx=encoder.onnx --saveEngine=encoder.engine
```

---

## 2.3 批量推断

```python
# 单次 1ms, 但 100 个物品串行 = 100ms → 不行
# 批量并行: 100 个物品 1ms 完成

def batch_sid_encode(model, items, batch_size=64):
    sids = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        sids.append(model.encode(batch))
    return torch.cat(sids)
```

---

## 2.4 三种架构如何选(决策表)

| 维度 | 全离线 | 混合 | 全在线 |
| :--- | :--- | :--- | :--- |
| 时延 | 0(查表) | 1-5ms | 5-20ms |
| query-aware | ❌ | 部分 | ✅ |
| 适合物料 | item(海量、稳定) | item+少量动态 | query/user(少量、多变) |
| 工程复杂度 | 低 | 中 | 高 |

**黄金法则**:**item 走离线(海量但稳定,预计算最划算),query/user 走在线**(少量但动态,必须实时算)。这就是"混合架构"成为主流的原因——把计算花在真正需要实时的地方。

---

## 2.5 在线 Encoder 的工程优化清单

在线 encoder 是时延瓶颈,优化手段:

| 手段 | 收益 |
| :--- | :--- |
| **TorchScript/ONNX/TensorRT** | 图优化 + 算子融合,2-5× |
| **量化(INT8)** | encoder 更小更快(见 [deeplearning 9.1](../../../算法与AI/深度学习手撕/第09章-推理优化/第09章-01-量化.md)) |
| **蒸馏小 encoder** | 用小模型逼近大 encoder |
| **动态批处理** | 攒窗口合批,摊薄开销 |
| **结果缓存** | 热门 query 的 SID 缓存,命中即 0 计算 |

```python
# 缓存 + 动态批的伪代码
def online_sid(query):
    if (sid := lru_cache.get(query)) is not None:
        return sid                          # 命中缓存
    sid = batched_encoder.submit(query)     # 进入动态批队列
    lru_cache.put(query, sid)
    return sid
```

---

## 2.6 一致性:离线与在线 encoder 必须同源

若 item SID 离线用模型 A 算、query SID 在线用模型 B 算,两者不在同一语义空间 → 检索全乱。必须:
- **同一个 encoder + 同一套码本**用于离线和在线。
- 版本升级时离线在线**同步切换**(见 [09-03 版本切换](../第09章-DLRM生产环境SID实战/第09章-03-版本切换与AB测试.md))。
- 这与搜推广的"离线在线一致性"是同一问题(见 [工程能力第05章](../../../基础/工程基础/工程能力/第05章-框架流程与工程架构设计.md))。

---

## 2.7 下一步

[3-ANN索引与SID检索.md](第07章-03-ANN索引与SID检索.md)

<!-- v2-batch-ad-expanded -->

## 深度扩展：从数学到生产闭环

> **专题焦点**：实时 SID 推断、动态批处理、缓存与版本一致性
> **阅读法**：A/B 理论 → C 运行代码 → D/E 对照 → F 实验卡。

## A. 统一符号、数学推导与工程含义

本节把 RQ-VAE、冷启动、位置编码、ANN 与实时推断放在同一套符号下。每个公式都对应可观测指标，不把数学当装饰。

| 符号 | 定义 | 形状 | 必须监控 |
| :--- | :--- | :--- | :--- |
| $x$ | 内容/行为/多模态特征 | $[B,d_x]$ | 缺失率、版本、漂移 |
| $z=f_\theta(x)$ | 连续语义表征 | $[B,D]$ | 范数、对齐、均匀性 |
| $E^{(\ell)}$ | 第 $\ell$ 层码本 | $[K_\ell,D]$ | 利用率、死码、版本 |
| $s=(k_1,\ldots,k_L)$ | SID | $[B,L]$ | 碰撞率、前缀纯度 |
| $q(z)$ | 量化重建向量 | $[B,D]$ | MSE、下游 Recall |
| $p$ | 行为时间位置 | $[T]$ | 外推长度、mask |
| $\ell$ | SID 层级位置 | $[L]$ | 粗细语义、后层熵 |

### A.1 RQ-VAE 残差量化推导

令 encoder 输出 $z=f_\theta(x)$，初始残差 $r^{(0)}=z$。第 $\ell$ 层执行：

$$
k_\ell=\arg\min_{k\in[1,K_\ell]}\|r^{(\ell-1)}-E_k^{(\ell)}\|_2^2,
\qquad r^{(\ell)}=r^{(\ell-1)}-E_{k_\ell}^{(\ell)}.
$$

量化重建为：

$$
q(z)=\sum_{\ell=1}^{L}E_{k_\ell}^{(\ell)},\qquad z=q(z)+r^{(L)}.
$$

最后一个等式来自递推的望远镜求和。它解释了“后层修正前层误差”的含义：第 1 层不是标签类目，而是对当前表征分布的粗近似；后续层只看剩余误差。若数据没有层次结构，RQ 层也不会自动对应“类目、品牌、型号”。

距离计算展开为：

$$
\|r-e_k\|_2^2=\|r\|_2^2+\|e_k\|_2^2-2r^\top e_k.
$$

因此实现应使用矩阵乘法，而不是显式创建 `[B,K,D]` 差值张量。逐层贪心复杂度是 $O(BD\sum_\ell K_\ell)$；全局组合搜索是 $O(BD\prod_\ell K_\ell)$，后者在工业规模不可行。

### A.2 损失、直通估计与码本更新

常用目标为：

$$
\mathcal L=\|x-g_\phi(q(z))\|_2^2+
\sum_{\ell=1}^{L}\left(
\|\operatorname{sg}[r^{(\ell-1)}]-e_{k_\ell}\|_2^2+
\beta\|r^{(\ell-1)}-\operatorname{sg}[e_{k_\ell}]\|_2^2\right).
$$

`argmin` 不可微，encoder 使用直通估计 $z_{st}=z+\operatorname{sg}[q(z)-z]$。前向是量化结果，反向把梯度近似传给连续表征。码本既可用梯度更新，也可用 EMA：

$$
N_k\leftarrow\gamma N_k+(1-\gamma)n_k,\quad
m_k\leftarrow\gamma m_k+(1-\gamma)\sum_{i:k_i=k}r_i,\quad
e_k\leftarrow m_k/(N_k+\epsilon).
$$

EMA 降低稀疏命中方差，但不能自动复活从未被选择的死码。生产中要保存逐层命中直方图，并从高误差残差池重置长期低频码。

### A.3 深度、宽度、有效容量与率失真

理论容量 $C=\prod_\ell K_\ell$ 只是上限。若第 $\ell$ 层码字概率为 $p_\ell(k)$，独立近似下有效容量是：

$$
C_{eff}=\exp\left(\sum_{\ell=1}^{L}H(p_\ell)\right),\quad
H(p_\ell)=-\sum_k p_\ell(k)\log p_\ell(k).
$$

当某层 90% 样本命中同一码字，扩大 $K$ 不会自动增加信息。必须联合报告：逐层 residual MSE、perplexity、top-1 code share、死码率、联合 SID 碰撞率和下游 Recall。

码长近似 $R=\sum_\ell\lceil\log_2K_\ell\rceil$ bit。率失真问题是在固定 $R$ 下最小化 $D(R)=\mathbb E\|z-q(z)\|^2$。深而窄通常带来清晰残差分工，但串行层更多；浅而宽层少，却增加单层搜索和稀疏命中风险。

### A.4 冷启动：偏差—方差与经验贝叶斯

随机 ID 对零行为物品没有观测，后验接近随机先验。内容 SID 使用 $z=f_\theta(c)$，共享 encoder 将老物品学到的条件结构迁移到新品。误差可分为：

$$
\mathbb E\|\hat y-y\|^2=\|\mathbb E\hat y-y\|^2+\operatorname{Tr}\operatorname{Var}(\hat y)+\sigma^2.
$$

内容分支降低零行为时的方差，但会引入内容与消费意图不一致的偏差。一个可解释融合是：

$$
\hat y_i=\lambda(n_i)y_i^{beh}+[1-\lambda(n_i)]y_i^{content},
\qquad \lambda(n)=\frac{n}{n+\tau}.
$$

$n=0$ 时只依赖内容；行为积累后平滑转向行为表征。$\tau$ 应按品类、曝光机会和反馈噪声选择。评估必须按 item age、行为计数、内容缺失和域偏移切片，否则老物品流量会掩盖新品失败。

统一空间不能只做正样本对齐。常用两项为：

$$
\mathcal L_{align}=\mathbb E\|u-i^+\|_2^2,
\quad \mathcal L_{uniform}=\log\mathbb E_{a,b}\exp(-t\|h_a-h_b\|_2^2).
$$

只压对齐会允许所有向量坍塌；均匀性把样本铺到单位球面。还要检查各模态、语言、品类、新旧物品的局部均匀性。

### A.5 位置编码：时间轴不等于 SID 层级轴

SID $(k_1,\ldots,k_L)$ 的 $\ell$ 是码本层级位置；用户历史 $(s_1,\ldots,s_T)$ 的 $p$ 是行为时间位置。建议显式组合：

$$
h_{p,\ell}=E_{code}(k_{p,\ell})+E_{level}(\ell)+E_{time}(p)+E_{type}(item).
$$

把两轴压成一个绝对位置会混淆“下一个行为”和“下一个残差层”。若展平为 $T\times L$ token，mask 必须与训练目标一致：item 间遵循时间因果，item 内可双向或粗到细因果，不能看未来 item。

正弦编码：

$$
PE(p,2j)=\sin(p/10000^{2j/D}),\quad PE(p,2j+1)=\cos(p/10000^{2j/D}).
$$

RoPE 将通道对旋转，并满足
$(R_{\theta_p}q)^\top(R_{\theta_t}k)=q^\top R_{\theta_{t-p}}k$，使注意力依赖相对位移。推荐先以“时间轴 RoPE + 层级 learned embedding”为强基线；只有消融证据充分时才给层级轴也用 RoPE。

### A.6 ANN、度量一致性与时延预算

L2 归一化后，余弦、内积与 L2 排序等价：

$$
\|a-b\|_2^2=2-2a^\top b.
$$

训练、建库、查询必须统一归一化与距离度量。在线预算写为：

$$
T_{e2e}=T_{queue}+T_{feature}+T_{encoder}+T_{RQ}+T_{ANN}+T_{post}+T_{network}.
$$

动态批降低单位计算成本，却增加排队。必须给等待窗口硬上限。用连续向量 Flat 隔离量化误差，用 SID 重建向量 Flat 隔离 ANN 误差，再测线上新鲜度和策略过滤。

| 索引 | 训练 | 增量 | 内存 | 优点 | 主要风险 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Flat | 无 | 易 | 高 | 精确基线 | 线性扫描 |
| IVF-Flat | coarse k-means | 中 | 中 | 参数直观 | nprobe 不足 |
| HNSW | 无聚类 | 易 | 高 | 高召回低时延 | 图内存大 |
| IVF-PQ | 多阶段训练 | 难 | 低 | 十亿级压缩 | 二次量化误差 |
| SID 前缀桶 | 稳定码本 | 易 | 低 | 利用层次结构 | 热桶倾斜 |

## B. 本文件专题：实时 SID 推断、动态批处理、缓存与版本一致性

**核心问题**：如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源。

### B.1 从业务目标反推技术选择

先写新品、长尾、头部、跨域、内容缺失和版本迁移六类切片，再定模型。总体 Recall 上升不能证明新品受益；总体 P99 合格也不能证明峰值动态批不超时。本专题主指标为：P50/P95/P99 时延、动态批大小、缓存命中率、版本错配率、超时降级率。

### B.2 五个可证伪假设

1. 增加 RQ 深度会降低重建误差，但后层困惑度接近 1 时不再提升检索。
2. 内容 SID 提高零行为覆盖，但内容域偏移会让校准恶化。
3. 归一化一致时，ANN 与量化误差可分开；不一致时二者混淆。
4. 时间位置与 SID 层级分轴建模比简单展平更稳定。
5. 动态批只在队列等待受控时改善吞吐。

### B.3 误差预算与隔离

$$
\Delta Rpprox\Delta R_{encoder}+\Delta R_{RQ}+\Delta R_{ANN}+\Delta R_{freshness}+\Delta R_{policy}.
$$

用 Flat 隔离 ANN，用连续向量隔离 RQ，固定快照隔离 freshness，关闭业务过滤隔离 policy。一次只替换一个组件。

| 层次 | 观测 | 失败 | 首选动作 |
| :--- | :--- | :--- | :--- |
| Encoder | 范数、漂移、间隔 | 坍塌/偏移 | 比较分桶直方图 |
| RQ | MSE、熵、碰撞 | 热码/后层无信息 | 查看逐层残差 |
| ANN | Recall、P99、内存 | 近似损失 | 对照 Flat 扫参数 |
| 服务 | 队列、批、超时 | P99 越线 | span 拆分与限时 |
| 冷启动 | age/count | 新品失败 | 检查缺失与域偏移 |
| 位置 | 长度、注意力熵 | 外推崩溃 | 对照双轴方案 |

### B.4 数据与版本契约

item 记录至少包含 `item_id`、`feature_version`、`encoder_version`、`codebook_version`、`sid`、`embedding_norm`、`created_at`。索引快照再记录 `index_version`、距离度量、归一化标志和数据截止时间。服务响应及采样日志携带版本；回滚切不可变指针，不在事故中临时全量重算。

## C. 完整实现：PyTorch RQ-VAE、Faiss ANN 与实时动态批

代码可放入单个 `.py` 运行。核心学习依赖为 `torch`、`numpy`；ANN 需 `faiss-cpu` 或 GPU 版。版本字段、归一化和超时边界没有省略。

```python
from __future__ import annotations
import asyncio
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def squared_l2(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """x:[B,D], codebook:[K,D] -> [B,K]，不创建 [B,K,D]。"""
    return (x.square().sum(1, keepdim=True)
            + codebook.square().sum(1).unsqueeze(0)
            - 2.0 * x @ codebook.t()).clamp_min_(0.0)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, levels: int = 5,
                 codes: int = 256, beta: float = 0.25):
        super().__init__()
        if dim <= 0 or levels <= 0 or codes <= 1:
            raise ValueError("invalid RQ dimensions")
        self.dim, self.levels, self.codes, self.beta = dim, levels, codes, beta
        self.codebooks = nn.Parameter(
            torch.randn(levels, codes, dim) / math.sqrt(dim))

    def forward(self, z: torch.Tensor):
        if z.ndim != 2 or z.size(1) != self.dim:
            raise ValueError(f"expected [B,{self.dim}]")
        residual = z
        quantized = torch.zeros_like(z)
        ids, residual_norms = [], [residual.detach().norm(dim=1).mean()]
        code_loss, commit_loss = z.new_zeros(()), z.new_zeros(())
        for level, codebook in enumerate(self.codebooks):
            idx = squared_l2(residual, codebook).argmin(dim=1)
            chosen = F.embedding(idx, codebook)
            ids.append(idx)
            code_loss = code_loss + F.mse_loss(chosen, residual.detach())
            commit_loss = commit_loss + F.mse_loss(residual, chosen.detach())
            quantized = quantized + chosen
            residual = residual - chosen
            residual_norms.append(residual.detach().norm(dim=1).mean())
        sid = torch.stack(ids, dim=1)
        straight_through = z + (quantized - z).detach()
        stats = {"code_loss": code_loss.detach(),
                 "commit_loss": commit_loss.detach(),
                 "residual_norms": torch.stack(residual_norms)}
        return straight_through, sid, code_loss + self.beta * commit_loss, stats

    @torch.inference_mode()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)[1]

    @torch.inference_mode()
    def decode(self, sid: torch.Tensor) -> torch.Tensor:
        if sid.ndim != 2 or sid.size(1) != self.levels:
            raise ValueError("sid must be [B,levels]")
        return torch.stack([F.embedding(sid[:, i], self.codebooks[i])
                            for i in range(self.levels)]).sum(0)


class RQVAE(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256,
                 latent: int = 128, levels: int = 5, codes: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                     nn.LayerNorm(hidden), nn.Linear(hidden, latent))
        self.quantizer = ResidualVectorQuantizer(latent, levels, codes)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                     nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor):
        z = F.normalize(self.encoder(x), dim=-1)
        zq, sid, vq_loss, stats = self.quantizer(z)
        recon = self.decoder(zq)
        rec_loss = F.mse_loss(recon, x)
        stats["reconstruction_loss"] = rec_loss.detach()
        return recon, sid, rec_loss + vq_loss, stats

    @torch.inference_mode()
    def encode(self, x: torch.Tensor):
        z = F.normalize(self.encoder(x), dim=-1)
        sid = self.quantizer.encode(z)
        vector = F.normalize(self.quantizer.decode(sid), dim=-1)
        return sid, vector


def train_step(model, optimizer, batch):
    model.train()
    _, _, loss, _ = model(batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


class FaissSIDIndex:
    """归一化内积 HNSW，外部 item_id 不依赖行号。"""
    def __init__(self, dim: int, m: int = 32, ef_search: int = 64):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("install faiss-cpu or faiss-gpu") from exc
        self.faiss, self.dim = faiss, dim
        base = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = 200
        base.hnsw.efSearch = ef_search
        self.index = faiss.IndexIDMap2(base)

    @staticmethod
    def normalized(x):
        x = np.ascontiguousarray(x, dtype="float32").copy()
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("zero vector cannot be normalized")
        return x / norms

    def add(self, vectors, item_ids):
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("vector shape mismatch")
        ids = np.ascontiguousarray(item_ids, dtype="int64")
        if ids.shape != (vectors.shape[0],):
            raise ValueError("item_ids shape mismatch")
        self.index.add_with_ids(self.normalized(vectors), ids)

    def search(self, queries, topk=100):
        scores, ids = self.index.search(self.normalized(queries), topk)
        return ids, scores

    def save(self, path):
        self.faiss.write_index(self.index, path)


@dataclass
class Request:
    feature: torch.Tensor
    deadline: float
    future: asyncio.Future


class AsyncSIDBatcher:
    """实时推断：最大批和最大等待同时受限。"""
    def __init__(self, model, max_batch=64, max_wait_ms=2.0, device="cpu"):
        self.model = model.eval().to(device)
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.device = torch.device(device)
        self.queue = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None

    async def start(self):
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def close(self):
        if self.worker is not None:
            await self.queue.put(None)
            await self.worker
            self.worker = None

    async def infer(self, feature, timeout_ms=20.0):
        if self.worker is None:
            raise RuntimeError("call start() first")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(Request(feature.detach().cpu(),
                                     loop.time() + timeout_ms / 1000.0,
                                     future))
        return await asyncio.wait_for(future, timeout_ms / 1000.0)

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            if first is None:
                return
            batch = [first]
            cutoff = min(first.deadline, loop.time() + self.max_wait)
            while len(batch) < self.max_batch:
                remaining = cutoff - loop.time()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is None:
                    await self.queue.put(None)
                    break
                batch.append(nxt)
            now = loop.time()
            live = [r for r in batch if r.deadline > now and not r.future.done()]
            live_ids = {id(r) for r in live}
            for r in batch:
                if id(r) not in live_ids and not r.future.done():
                    r.future.set_exception(asyncio.TimeoutError("expired in queue"))
            if not live:
                continue
            features = torch.stack([r.feature for r in live]).to(self.device)
            with torch.inference_mode():
                sids, vectors = self.model.encode(features)
            for row, request in enumerate(live):
                if not request.future.done():
                    request.future.set_result((sids[row].cpu(), vectors[row].cpu()))


async def smoke_test():
    torch.manual_seed(7)
    model = RQVAE(32, hidden=64, latent=16, levels=4, codes=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randn(48, 32)
    assert math.isfinite(train_step(model, optimizer, batch))
    sid, vector = model.encode(batch[:3])
    assert sid.shape == (3, 4) and vector.shape == (3, 16)
    batcher = AsyncSIDBatcher(model, max_batch=8, max_wait_ms=1.0)
    await batcher.start()
    out = await asyncio.gather(*(batcher.infer(batch[i], 1000.0) for i in range(8)))
    await batcher.close()
    assert len(out) == 8 and out[0][0].numel() == 4
    print("smoke test passed", sid.tolist())


if __name__ == "__main__":
    asyncio.run(smoke_test())
```

### C.1 Faiss 建库与查询

```python
item_features = torch.randn(10_000, 32)
with torch.inference_mode():
    item_sids, item_vectors = model.encode(item_features)
ann = FaissSIDIndex(item_vectors.size(1), m=32, ef_search=80)
ann.add(item_vectors.numpy(), np.arange(10_000, dtype="int64"))
with torch.inference_mode():
    query_sids, query_vectors = model.encode(torch.randn(4, 32))
ids, scores = ann.search(query_vectors.numpy(), topk=20)
assert ids.shape == scores.shape == (4, 20)
```

该实现是完整学习闭环，不伪装成完整平台。生产还要接特征服务、认证、限流、指标、双版本读和快照校验；这些应在真实接口确定后接入，而不是提前制造抽象。

## D. 工业案例：事实边界与可迁移原则

> YouTube、Alibaba、Spotify 的公开系统不等于都使用本教程这套 SID。以下基于公开的多阶段推荐、层次检索与向量检索思想做 SID 映射，不把推演冒充厂商内部事实。

| 案例 | 公开启发 | SID 映射 | 主指标 | 不可照搬 |
| :--- | :--- | :--- | :--- | :--- |
| YouTube | candidate generation + ranking、多目标、新鲜度 | SID/ANN 召回，排序保留完整上下文 | 候选 Recall、观看时长、freshness | 流量与目标权重 |
| Alibaba | 电商多阶段检索、层次结构、规则过滤 | SID 前缀桶粗召回，ANN/排序细化 | 新品覆盖、GMV 护栏、热桶倾斜 | 类目、库存、活动规则 |
| Spotify | 向量近邻、内容与协同信号融合 | 音频/文本生成新品 SID，行为积累后融合 | save/skip、长尾覆盖、延迟 | 音乐反馈不等于点击 |

### D.1 YouTube 式两阶段候选与排序

第一阶段要高覆盖和硬时延，可取连续向量 ANN、SID 前缀桶或二者并集；第二阶段再处理丰富上下文与多目标。量化模型负责结构和压缩，不应承担全部业务目标。实验同时保留连续 Flat、连续 ANN、SID 重建 Flat、SID ANN 四条线，才能分清误差来源。

### D.2 Alibaba 式层次检索与新品治理

电商类目天然分层，但 RQ 每层不会自动对齐人工类目。应测前缀与类目的 NMI、条件熵、桶大小长尾和热前缀流量。新品先凭内容 SID 获得候选资格，再经过库存、地域、合规和活动过滤；新品覆盖分母应是“有曝光机会的可售新品”。

### D.3 Spotify 式内容到协同平滑迁移

新歌曲先用音频、文本、作者元数据生成内容 SID。播放、收藏、跳过信号积累后，用 $\lambda(n)$ 逐步融合协同表征。playlist 的顺序放在时间位置轴，单曲 SID 的粗细层级放在 level 轴。skip 的噪声很高，应按反馈置信度折算有效行为计数。

### D.4 共同上线纪律

1. 先有可复现 Flat 基线，再调 ANN。
2. 新品、长尾、头部、跨域分别报告。
3. 模型、码本、索引记录不可变版本。
4. 线上实验预登记主指标、护栏、停止条件。
5. 厂商案例只启发约束，不替代本业务证据。

## E. 方案对比与验收表

### E.1 RQ 配置

| 方案 | $L$ | $K$ | 优势 | 风险 | 适用 |
| :--- | ---: | ---: | :--- | :--- | :--- |
| 浅而宽 | 2-3 | 1024-4096 | 层少 | 单层搜索贵、稀疏 | 低层级需求 |
| 深而窄 | 6-8 | 128-256 | 残差细化、前缀清晰 | 串行、后层无信息 | 层次召回 |
| 均衡 | 4-6 | 256-1024 | 成本质量折中 | 仍需看有效熵 | 常规起点 |
| 非均匀 | 各层不同 | 各层不同 | 容量给高熵层 | 版本复杂 | 有逐层证据后 |

### E.2 冷启动

| 方法 | 零行为 | 随行为更新 | 可解释 | 主要偏差 |
| :--- | :---: | :---: | :---: | :--- |
| 随机 ID | 否 | 是 | 低 | 新品高方差 |
| 纯内容 | 是 | 否 | 中 | 意图错位 |
| 内容 SID | 是 | 可重算 | 高 | 量化与漂移 |
| 贝叶斯融合 | 是 | 是 | 高 | $\tau$ 与噪声 |
| 多任务统一空间 | 是 | 是 | 中 | 目标冲突 |

### E.3 位置编码

| 编码 | 相对位置 | 外推 | 参数 | SID 建议 |
| :--- | :---: | :---: | :---: | :--- |
| Learned absolute | 否 | 弱 | 有 | 固定短层级 |
| Sinusoidal | 间接 | 中 | 无 | 教学基线 |
| RoPE | 是 | 中 | 无 | 时间轴强基线 |
| ALiBi | 是 | 强 | 少 | 长历史对照 |
| level embedding | 层级显式 | 不涉超长 | 有 | 层级轴首选 |
| 双轴方案 | 是 | 依时间轴 | 少 | 消融后采用 |

### E.4 ANN 与在线架构

| 方案 | 质量 | 内存 | 更新 | P99 风险 | 用法 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Flat | 精确 | 高 | 易 | 大库高 | 基线 |
| HNSW | 高 | 高 | 易 | 中 | 低时延召回 |
| IVF-Flat | 可调 | 中 | 中 | 中 | 大规模 |
| IVF-PQ | 中 | 低 | 难 | 中 | 内存受限 |
| 全离线 SID | 稳定 | KV 成本 | 批次 | 低 | item |
| 混合推断 | 高 | 中 | 灵活 | 中 | item 离线、query 在线 |
| 全在线 | 最新 | 中 | 灵活 | 高 | 小流量 query-aware |

### E.5 发布门槛

| 阶段 | 必须通过 | 否决条件 |
| :--- | :--- | :--- |
| 数据 | 快照与版本齐全 | 数据截止不明 |
| RQ | MSE、困惑度、碰撞达标 | 长期单码垄断 |
| ANN | 相对 Flat Recall 达标 | 无精确基线 |
| 冷启动 | age/count 切片增益 | 只看总体 |
| 服务 | 峰值 P99、错误率达标 | 等批无上限 |
| 发布 | 双读、灰度、指针回滚 | 版本不可追溯 |

## F. 可复现实验与设计审查图谱

以下不是虚构线上结果，而是可执行实验卡。阈值是示例门槛，上线前必须由业务基线替换。每张卡改变规模、RQ、索引、冷启动切片和位置方案。

### F.1 场景 001：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=1024, D=256`；码长 `50 bit`，`log10(理论容量)=15.05`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=8`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.2 场景 002：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=512, D=192`；码长 `63 bit`，`log10(理论容量)=18.96`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=12`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.3 场景 003：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=256, D=128`；码长 `32 bit`，`log10(理论容量)=9.63`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=16`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.4 场景 004：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=128, D=96`；码长 `42 bit`，`log10(理论容量)=12.64`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=20`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.5 场景 005：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=1024, D=64`；码长 `30 bit`，`log10(理论容量)=9.03`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=24`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.6 场景 006：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=512, D=256`；码长 `45 bit`，`log10(理论容量)=13.55`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=28`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.7 场景 007：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=256, D=192`；码长 `56 bit`，`log10(理论容量)=16.86`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=32`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.8 场景 008：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=128, D=128`；码长 `28 bit`，`log10(理论容量)=8.43`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=36`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.9 场景 009：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=1024, D=96`；码长 `60 bit`，`log10(理论容量)=18.06`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=40`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.10 场景 010：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=512, D=64`；码长 `27 bit`，`log10(理论容量)=8.13`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=44`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.11 场景 011：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=256, D=256`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=48`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.12 场景 012：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=128, D=192`；码长 `49 bit`，`log10(理论容量)=14.75`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=52`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.13 场景 013：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=1024, D=128`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=56`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.14 场景 014：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=512, D=96`；码长 `54 bit`，`log10(理论容量)=16.26`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=60`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.15 场景 015：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=256, D=64`；码长 `24 bit`，`log10(理论容量)=7.22`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=64`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.16 场景 016：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=128, D=256`；码长 `35 bit`，`log10(理论容量)=10.54`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=4`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.17 场景 017：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=1024, D=192`；码长 `70 bit`，`log10(理论容量)=21.07`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=8`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.18 场景 018：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=512, D=128`；码长 `36 bit`，`log10(理论容量)=10.84`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=12`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.19 场景 019：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=256, D=96`；码长 `48 bit`，`log10(理论容量)=14.45`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=16`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.20 场景 020：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=128, D=64`；码长 `21 bit`，`log10(理论容量)=6.32`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=20`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.21 场景 021：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=1024, D=256`；码长 `50 bit`，`log10(理论容量)=15.05`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=24`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.22 场景 022：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=512, D=192`；码长 `63 bit`，`log10(理论容量)=18.96`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=28`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.23 场景 023：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=256, D=128`；码长 `32 bit`，`log10(理论容量)=9.63`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=32`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.24 场景 024：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=128, D=96`；码长 `42 bit`，`log10(理论容量)=12.64`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=36`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.25 场景 025：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=1024, D=64`；码长 `30 bit`，`log10(理论容量)=9.03`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=40`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.26 场景 026：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=512, D=256`；码长 `45 bit`，`log10(理论容量)=13.55`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=44`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.27 场景 027：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=256, D=192`；码长 `56 bit`，`log10(理论容量)=16.86`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=48`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.28 场景 028：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=128, D=128`；码长 `28 bit`，`log10(理论容量)=8.43`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=52`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.29 场景 029：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=1024, D=96`；码长 `60 bit`，`log10(理论容量)=18.06`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=56`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.30 场景 030：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=512, D=64`；码长 `27 bit`，`log10(理论容量)=8.13`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=60`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.31 场景 031：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=256, D=256`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=64`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.32 场景 032：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=128, D=192`；码长 `49 bit`，`log10(理论容量)=14.75`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=4`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.33 场景 033：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=1024, D=128`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=8`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.34 场景 034：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=512, D=96`；码长 `54 bit`，`log10(理论容量)=16.26`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=12`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.35 场景 035：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=256, D=64`；码长 `24 bit`，`log10(理论容量)=7.22`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=16`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.36 场景 036：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=128, D=256`；码长 `35 bit`，`log10(理论容量)=10.54`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=20`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.37 场景 037：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=1024, D=192`；码长 `70 bit`，`log10(理论容量)=21.07`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=24`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.38 场景 038：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=512, D=128`；码长 `36 bit`，`log10(理论容量)=10.84`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=28`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.39 场景 039：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=256, D=96`；码长 `48 bit`，`log10(理论容量)=14.45`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=32`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.40 场景 040：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=128, D=64`；码长 `21 bit`，`log10(理论容量)=6.32`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=36`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.41 场景 041：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=1024, D=256`；码长 `50 bit`，`log10(理论容量)=15.05`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=40`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.42 场景 042：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=512, D=192`；码长 `63 bit`，`log10(理论容量)=18.96`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=44`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.43 场景 043：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=256, D=128`；码长 `32 bit`，`log10(理论容量)=9.63`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=48`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.44 场景 044：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=128, D=96`；码长 `42 bit`，`log10(理论容量)=12.64`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=52`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.45 场景 045：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=1024, D=64`；码长 `30 bit`，`log10(理论容量)=9.03`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=56`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.46 场景 046：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=512, D=256`；码长 `45 bit`，`log10(理论容量)=13.55`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=60`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.47 场景 047：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=256, D=192`；码长 `56 bit`，`log10(理论容量)=16.86`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=64`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.48 场景 048：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=128, D=128`；码长 `28 bit`，`log10(理论容量)=8.43`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=4`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.49 场景 049：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=1024, D=96`；码长 `60 bit`，`log10(理论容量)=18.06`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=8`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.50 场景 050：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=512, D=64`；码长 `27 bit`，`log10(理论容量)=8.13`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=12`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.51 场景 051：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=256, D=256`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=16`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.52 场景 052：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=128, D=192`；码长 `49 bit`，`log10(理论容量)=14.75`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=20`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.53 场景 053：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=1024, D=128`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=24`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.54 场景 054：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=512, D=96`；码长 `54 bit`，`log10(理论容量)=16.26`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=28`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.55 场景 055：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=256, D=64`；码长 `24 bit`，`log10(理论容量)=7.22`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=32`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.56 场景 056：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=128, D=256`；码长 `35 bit`，`log10(理论容量)=10.54`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=36`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.57 场景 057：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=1024, D=192`；码长 `70 bit`，`log10(理论容量)=21.07`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=40`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.58 场景 058：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=512, D=128`；码长 `36 bit`，`log10(理论容量)=10.84`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=44`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.59 场景 059：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=256, D=96`；码长 `48 bit`，`log10(理论容量)=14.45`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=48`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.60 场景 060：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=128, D=64`；码长 `21 bit`，`log10(理论容量)=6.32`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=52`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.61 场景 061：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=1024, D=256`；码长 `50 bit`，`log10(理论容量)=15.05`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=56`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.62 场景 062：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=512, D=192`；码长 `63 bit`，`log10(理论容量)=18.96`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=60`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.63 场景 063：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=256, D=128`；码长 `32 bit`，`log10(理论容量)=9.63`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=64`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.64 场景 064：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=128, D=96`；码长 `42 bit`，`log10(理论容量)=12.64`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=4`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.65 场景 065：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=1024, D=64`；码长 `30 bit`，`log10(理论容量)=9.03`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=8`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.66 场景 066：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=512, D=256`；码长 `45 bit`，`log10(理论容量)=13.55`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=12`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.67 场景 067：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=256, D=192`；码长 `56 bit`，`log10(理论容量)=16.86`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=16`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.68 场景 068：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=128, D=128`；码长 `28 bit`，`log10(理论容量)=8.43`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=20`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.69 场景 069：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=1024, D=96`；码长 `60 bit`，`log10(理论容量)=18.06`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=24`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.70 场景 070：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=512, D=64`；码长 `27 bit`，`log10(理论容量)=8.13`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=28`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.71 场景 071：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=256, D=256`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=32`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.72 场景 072：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=128, D=192`；码长 `49 bit`，`log10(理论容量)=14.75`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=36`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.73 场景 073：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=1024, D=128`；码长 `40 bit`，`log10(理论容量)=12.04`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=40`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.74 场景 074：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=512, D=96`；码长 `54 bit`，`log10(理论容量)=16.26`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=44`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.75 场景 075：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=256, D=64`；码长 `24 bit`，`log10(理论容量)=7.22`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=48`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.76 场景 076：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=128, D=256`；码长 `35 bit`，`log10(理论容量)=10.54`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=52`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.77 场景 077：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=1024, D=192`；码长 `70 bit`，`log10(理论容量)=21.07`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=56`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.78 场景 078：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=512, D=128`；码长 `36 bit`，`log10(理论容量)=10.84`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=60`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.79 场景 079：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=256, D=96`；码长 `48 bit`，`log10(理论容量)=14.45`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=64`，HNSW `efSearch=128`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.80 场景 080：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=128, D=64`；码长 `21 bit`，`log10(理论容量)=6.32`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=4`，HNSW `efSearch=144`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.81 场景 081：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=1024, D=256`；码长 `50 bit`，`log10(理论容量)=15.05`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=8`，HNSW `efSearch=160`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.82 场景 082：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=512, D=192`；码长 `63 bit`，`log10(理论容量)=18.96`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=12`，HNSW `efSearch=176`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.83 场景 083：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=256, D=128`；码长 `32 bit`，`log10(理论容量)=9.63`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=16`，HNSW `efSearch=192`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.84 场景 084：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=128, D=96`；码长 `42 bit`，`log10(理论容量)=12.64`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=20`，HNSW `efSearch=16`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.85 场景 085：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=1024, D=64`；码长 `30 bit`，`log10(理论容量)=9.03`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=24`，HNSW `efSearch=32`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.86 场景 086：实时 SID 推断、动态批处理、缓存与版本一致性 / 1-5 次反馈

- **问题**：目录 `100,000`、主指标 `P50/P95/P99 时延` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `1-5 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=5, K=512, D=256`；码长 `45 bit`，`log10(理论容量)=13.55`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+10)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`block-causal RoPE`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-Flat`；IVF `nprobe=28`，HNSW `efSearch=48`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `15ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`P50/P95/P99 时延`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`动态批大小`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.87 场景 087：实时 SID 推断、动态批处理、缓存与版本一致性 / 6-50 次反馈

- **问题**：目录 `1,000,000`、主指标 `动态批大小` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `6-50 次反馈`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=7, K=256, D=192`；码长 `56 bit`，`log10(理论容量)=16.86`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+20)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-ALiBi + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`SID 前缀桶 + HNSW`；IVF `nprobe=32`，HNSW `efSearch=64`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `20ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`动态批大小`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`缓存命中率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.88 场景 088：实时 SID 推断、动态批处理、缓存与版本一致性 / 稳定老品

- **问题**：目录 `10,000,000`、主指标 `缓存命中率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `稳定老品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=4, K=128, D=128`；码长 `28 bit`，`log10(理论容量)=8.43`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+50)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`无时间位置消融`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`HNSW`；IVF `nprobe=36`，HNSW `efSearch=80`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `30ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`缓存命中率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`版本错配率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.89 场景 089：实时 SID 推断、动态批处理、缓存与版本一致性 / 跨域迁移品

- **问题**：目录 `100,000,000`、主指标 `版本错配率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `跨域迁移品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=6, K=1024, D=96`；码长 `60 bit`，`log10(理论容量)=18.06`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+100)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`sinusoidal 双轴`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`IVF-PQ`；IVF `nprobe=40`，HNSW `efSearch=96`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `50ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`版本错配率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`超时降级率`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

### F.90 场景 090：实时 SID 推断、动态批处理、缓存与版本一致性 / 0 行为新品

- **问题**：目录 `1,000,000,000`、主指标 `超时降级率` 下检验“如何在 P99 预算内完成 encoder、残差量化、ANN 检索并保证离在线同源”。
- **切片**：单报 `0 行为新品`，再按品类、语言、内容缺失、item age、曝光机会分桶；禁止总体均值覆盖。
- **RQ**：`L=3, K=512, D=64`；码长 `27 bit`，`log10(理论容量)=8.13`，验收看有效熵。
- **RQ 诊断**：逐层 residual MSE、perplexity、top-1 占比、死码率、联合碰撞；后层熵≈0 就停止加深。
- **冷启动**：`lambda(n)=n/(n+5)`；n=0 只用内容，高噪反馈折算有效计数。
- **位置**：`time-RoPE + level-embedding`；时间轴与 level 轴独立，消融保持参数量和 mask 一致。
- **索引**：`Flat-IP 基线`；IVF `nprobe=44`，HNSW `efSearch=112`；建库、查询同样 L2 归一化。
- **精确基线**：固定快照跑 Flat；ANN Recall 定义 `|ANN_K ∩ Flat_K| / K`。
- **时延**：端到端 P99 示例门槛 `10ms`；拆 queue/feature/encoder/RQ/ANN/post/network。
- **动态批**：最大批 64，等待不超过预算 10%；到期请求入模前失败或降级，不能无限排队。
- **主指标**：`超时降级率`；阈值离线预登记，线上不得看结果后换口径。
- **护栏**：`P50/P95/P99 时延`、错误率、超时率、热门偏置、新品覆盖；任一越线即回滚。
- **隔离**：连续 Flat 正常而 SID Flat 降，查 RQ；SID Flat 正常而 ANN 降，查索引。
- **根因顺序**：数据版本 → 归一化/距离 → encoder → RQ 热码 → ANN → 业务过滤。
- **修复**：一次只改一个变量；死码用高误差残差重置，索引扫参，域偏移重训内容 encoder。
- **发布**：影子建库，1%/5%/20% 灰度；模型、码本、索引绑定版本，回滚只切指针。
- **审计**：保存配置、种子、截止时间、model hash、Flat/ANN 邻居、逐层 SID 直方图。
- **交叉阅读**：[附录D-SID与位置编码](../附录D-SID与位置编码.md)。
- **通过**：主指标达标、护栏不过线、两次独立快照方向一致；否则结论是“证据不足”。

## G. 跨章节学习闭环

按“表示 → 冷启动 → 位置 → 在线 → 约束/评估”顺序复习：

1. [第03章-02-RQ-VAE的训练与推理](../第03章-RQ-VAE与残差量化/第03章-02-RQ-VAE的训练与推理.md)：把局部结论放回完整 SID 链路。
2. [第05章-附录-统一空间实现](../第05章-统一语义空间与冷启动/第05章-附录-统一空间实现.md)：把局部结论放回完整 SID 链路。
3. [第07章-03-ANN索引与SID检索](第07章-03-ANN索引与SID检索.md)：把局部结论放回完整 SID 链路。
4. [第07章-04-SID版本管理与回滚](第07章-04-SID版本管理与回滚.md)：把局部结论放回完整 SID 链路。
5. [附录D-SID与位置编码](../附录D-SID与位置编码.md)：把局部结论放回完整 SID 链路。

### G.1 口述检查

- 从 $r^{(0)}=z$ 推到 $z=q(z)+r^{(L)}$，解释逐层贪心为何可上线。
- 说明 $K^L$ 与有效容量的差别，列出三个码本健康指标。
- 用偏差—方差和经验贝叶斯解释新品受益与失效条件。
- 区分 SID level 与行为时间位置，并画出无未来泄漏的 mask。
- 用 Flat 隔离 ANN 误差，用连续向量隔离量化误差。
- 给出模型、码本、索引的版本绑定与回滚步骤。

### G.2 最终原则

SID 不是神奇 ID，而是连续表征、离散压缩、层次结构、检索系统和在线治理的共同接口。数学目标、数据切片、索引度量与服务版本一致时，收益才可复现。
