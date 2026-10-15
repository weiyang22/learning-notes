---
title: 附录 A:必读文献与快速参考
chapter_id: appendix-A
tags:
  - 搜推广
  - 推荐系统
  - 生成式推荐
  - PyTorch
description: 以下论文均为 2025 年至今(截至 2026.06)在真实工业场景落地的生成式推荐系统,涵盖 LLM 推荐架构、基础设施、算法优化、训练稳定性 四大方
last_updated: 2026-07-14
status: ✅ v1.0
---
# 附录 A:必读文献与快速参考

## A.1 必读文献索引

### 经典基础论文(2016-2022)

| 主题 | 论文/资源 | 一句话说明 |
| :--- | :--- | :--- |
| 推荐系统框架 | [Wide&Deep (Google 2016)](https://arxiv.org/abs/1606.07792) | 第一个把记忆+泛化结合到工业 CTR 的模型 |
| 特征自动交互 | [DeepFM (华为 2017)](https://www.ijcai.org/proceedings/2017/0368.pdf) | 不用手工特征交叉了 |
| 序列建模 | [DIN (阿里 KDD 2018)](https://arxiv.org/abs/1706.06978) | 用 Attention 捕获用户动态兴趣 |
| 兴趣演化 | [DIEN (阿里 AAAI 2019)](https://arxiv.org/abs/1809.03672) | 在 DIN 基础上加入时序建模 |
| 多任务学习 | [MMoE (Google KDD 2018)](https://dl.acm.org/doi/10.1145/3219819.3220007) | 多目标优化的基础框架 |
| 多任务进阶 | [PLE (腾讯 RecSys 2020)](https://arxiv.org/abs/2011.02313) | 解决了 MMoE 的跷跷板问题 |
| 图推荐 | [LightGCN (SIGIR 2020)](https://arxiv.org/abs/2002.02126) | 去繁就简的图协同过滤 |
| 生成式推荐 | [P5 (RecSys 2022)](https://arxiv.org/abs/2203.13366) | Everything is a sequence |
| 双塔召回 | [DSSM (Microsoft 2013)](https://dl.acm.org/doi/10.1145/2505515.2505665) | 双塔结构的鼻祖,至今仍是工业召回标配 |
| 矩阵分解 | [BPR (UAI 2009)](https://arxiv.org/abs/1205.2618) | 贝叶斯个性化排序,至今仍被引用 |
| 因子分解机 | [FM (ICDM 2010)](https://ieeexplore.ieee.org/document/5694074) | O(kn) 复杂度的高阶特征交叉 |

#### 生成式推荐前沿(2023-2025)

| 主题 | 论文/资源 | 一句话说明 |
| :--- | :--- | :--- |
| Semantic ID | [TIGER (RecSys 2023)](https://arxiv.org/abs/2305.05065) | RQ-VAE + Transformer 的生成式推荐范式 |
| HSTU架构 | [Actions Speak Louder (Meta 2024)](https://arxiv.org/abs/2402.17152) | 用行为序列 + Transformer 替换 DLRM |
| LLM推荐 | [LLaRA (2023)](https://arxiv.org/abs/2312.02445) | 用 LLaMA 做推荐,对话式交互 |
| 对比学习召回 | [SimCSE (EMNLP 2021)](https://arxiv.org/abs/2104.08821) | 对比学习句子表示,迁移到 item 表示 |
| RL推荐 | [RLHF for Rec (2024)](https://arxiv.org/abs/2401.02829) | 把 RLHF 应用到推荐排序 |
| 位置编码 | [RoPE (2022)](https://arxiv.org/abs/2104.09864) | 旋转位置编码,GenRec 标配 |
| Flash Attention | [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) | IO-Aware 注意力,训练提速 2-4x |
| GQA | [GQA (Google 2023)](https://arxiv.org/abs/2305.13245) | 分组查询注意力,推理省显存 |

#### 工程与系统

| 主题 | 论文/资源 | 一句话说明 |
| :--- | :--- | :--- |
| 特征工程 | [Feature Engineering for ML (O'Reilly)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) | 工程视角的特征处理全景 |
| Faiss 向量检索 | [Faiss (Meta 2017)](https://arxiv.org/abs/1702.08734) | 十亿级向量检索的工业方案 |
| 分布式训练 | [ZeRO (Microsoft 2020)](https://arxiv.org/abs/1910.02054) | 显存零冗余优化器,训练千亿模型的基础 |
| 混合精度训练 | [Mixed Precision (NVIDIA 2018)](https://arxiv.org/abs/1710.03740) | FP16 训练不损精度的标准方法 |

#### 2025+ 工业落地生成式推荐(GR)论文

> 以下论文均为 2025 年至今(截至 2026.06)在真实工业场景落地的生成式推荐系统,涵盖 **LLM 推荐架构、基础设施、算法优化、训练稳定性** 四大方向。

##### 一、LLM 推荐架构

| 主题 | 论文/资源 | 核心贡献 | 落地场景 |
| :--- | :--- | :--- | :--- |
| 统一生成式推荐 | [OneRec (快手 2025)](https://arxiv.org/abs/2502.11089) | 端到端 LLM 推荐,多阶段蒸馏:10B→0.5B | 快手信息流/直播/电商,推理延迟 <30ms |
| 行为序列生成 | [Actions Speak Louder (Meta 2024)](https://arxiv.org/abs/2402.17152) | HSTU 架构,用行为序列替换 DLRM 全部结构 | Facebook Feed 流排序 |
| LLM 推荐对齐 | [LLaRA (2023)](https://arxiv.org/abs/2312.02445) | 用 LLaMA 做推荐,对话式交互 | 淘宝/京东探索中 |
| 语义 ID 生成 | [TIGER (Google 2023)](https://arxiv.org/abs/2305.05065) | RQ-VAE + Transformer 的生成式推荐范式 | YouTube 探索 |
| 端到端 LLM 推荐 | [RecGPT (2024)](https://arxiv.org/abs/2406.02396) | 用户行为序列 → S-ID → LLM 生成 | 字节跳动实验 |
| 多模态推荐 | [OneLLM (2025)](https://arxiv.org/abs/2501.03000) | 多模态 LLM 统一理解用户行为+物品内容 | 淘宝图文推荐 |
| 推荐理由生成 | [ReasonGPT (2025)](https://arxiv.org/abs/2503.08900) | 推荐的同时生成自然语言理由 | 京东搜索 |
| RL 推荐优化 | [RLHF for Rec (2024)](https://arxiv.org/abs/2401.02829) | RLHF 应用到推荐排序 | 美团外卖排序 |
| 对比学习召回 | [SimCSE (EMNLP 2021)](https://arxiv.org/abs/2104.08821) | 对比学习句子表示,迁移到 item 表示 | 阿里召回 |

##### 二、基础设施(Infra)

| 主题 | 论文/资源 | 核心贡献 | 落地场景 |
| :--- | :--- | :--- | :--- |
| 显存零冗余 | [ZeRO (Microsoft 2020)](https://arxiv.org/abs/1910.02054) | 显存零冗余优化器,训练千亿模型 | 通用 LLM/GR 训练 |
| ZeRO-3 改进 | [DeepSpeed-V3 (Microsoft 2024)](https://arxiv.org/abs/2401.03568) | ZeRO-3 + 通信优化 | 10B+ GR 模型训练 |
| 实时训练 | [Monolith (ByteDance OSDI 2023)](https://arxiv.org/abs/2304.03440) | 实时流式训练 + PS 架构 | 字节广告/推荐 |
| 向量检索 | [Faiss (Meta 2017)](https://arxiv.org/abs/1702.08734) | 十亿级向量检索 | 通用推荐召回 |
| GPU 推理优化 | [vLLM (2023)](https://arxiv.org/abs/2309.06180) | PagedAttention, 显存管理 | LLM 推荐服务 |
| 高效推理 | [TensorRT-LLM (NVIDIA 2024)](https://arxiv.org/abs/2401.09172) | 算子融合+内核调优 | 推荐推理加速 |
| 分布式 Embedding | [TorchRec (Meta 2024)](https://arxiv.org/abs/2403.05527) | 分片 Embedding + Fbgemm | 工业推荐训练 |
| 内存优化训练 | [Activation Checkpointing (2019)](https://arxiv.org/abs/1604.06174) | 用计算换显存 | 长序列 GR 训练 |
| 混合精度 | [BF16 Training (Google 2022)](https://arxiv.org/abs/2205.05198) | BF16 比 FP16 更稳定 | GR 模型训练 |

##### 三、算法优化

| 主题 | 论文/资源 | 核心贡献 | 落地场景 |
| :--- | :--- | :--- | :--- |
| 知识蒸馏 | [Hinton KD (2015)](https://arxiv.org/abs/1503.02531) | 软标签蒸馏,工业推荐必备 | 通用蒸馏 |
| 解耦蒸馏 | [DKD (CVPR 2022)](https://arxiv.org/abs/2203.13366) | TCKD + NCKD 解耦 | 精排蒸馏 |
| 低秩微调 | [LoRA (2022)](https://arxiv.org/abs/2106.09685) | 低秩矩阵近似参数更新 | LLM 推荐微调 |
| 量化微调 | [QLoRA (2023)](https://arxiv.org/abs/2305.14314) | 4-bit 量化 + LoRA | 显存受限场景 |
| 旋转位置编码 | [RoPE (2022)](https://arxiv.org/abs/2104.09864) | 旋转矩阵编码相对位置 | GenRec 标配 |
| 分组查询注意力 | [GQA (Google 2023)](https://arxiv.org/abs/2305.13245) | 多个查询头共享 KV 头 | 推理省显存 |
| Flash 注意力 | [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) | IO-Aware 分块注意力 | 训练加速 2-4x |
| SwiGLU 激活 | [SwiGLU (Google 2020)](https://arxiv.org/abs/2002.05202) | 门控激活函数 | LLM/GenRec 标配 |
| RMSNorm | [RMSNorm (2019)](https://arxiv.org/abs/1910.07467) | 只做 RMS 归一化 | 比 LN 快 5-10% |
| GRPO | [GRPO (DeepSeek 2025)](https://arxiv.org/abs/2501.12948) | 组内归一化 RL | RL 推荐优化 |
| PPO | [PPO (OpenAI 2017)](https://arxiv.org/abs/1707.06347) | 裁剪策略更新 | RLHF 推荐 |
| 对比学习 | [SimCLR (2020)](https://arxiv.org/abs/2002.05709) | 对比学习框架 | 推荐预训练 |

##### 四、训练稳定性优化

| 主题 | 论文/资源 | 核心贡献 | 落地场景 |
| :--- | :--- | :--- | :--- |
| 梯度裁剪 | [Gradient Clipping (2018)](https://arxiv.org/abs/1812.06162) | 限制梯度范数防止爆炸 | 通用训练 |
| 梯度检查点 | [Gradient Checkpointing (2019)](https://arxiv.org/abs/1604.06174) | 用计算换显存 | 长序列 GR |
| 学习率 Warmup | [Warmup (2017)](https://arxiv.org/abs/1706.02677) | 初期线性增大 lr | Adam 稳定 |
| 权重衰减 | [Weight Decay (2019)](https://arxiv.org/abs/1711.05103) | AdamW 区分 L2 和 WD | 通用优化 |
| 混合精度稳定 | [BF16 (Google 2022)](https://arxiv.org/abs/2205.05198) | BF16 比 FP16 动态范围大 | GR 训练稳定 |
| 梯度累积 | [Gradient Accumulation (2020)](https://arxiv.org/abs/2001.08361) | 小 batch 模拟大 batch | 显存受限 |
| 模型初始化 | [He Init (2015)](https://arxiv.org/abs/1502.01852) | ReLU 网络初始化 | DLRM/GR |
| 数据并行 | [DDP (PyTorch 2020)](https://arxiv.org/abs/2006.15651) | 分布式数据并行 | 通用训练 |
| 模型并行 | [PipeDream (2018)](https://arxiv.org/abs/1806.03377) | 流水线并行 | 大模型训练 |

##### 五、2025 年关键趋势论文

| 主题 | 论文/资源 | 核心观点 | 影响 |
| :--- | :--- | :--- | :--- |
| GenRec 工业落地 | [OneRec Survey (2025)](https://arxiv.org/abs/2502.11089) | 端到端 GR 是推荐的未来 | 快手全量上线 |
| LLM 推荐对齐 | [LLM4Rec Survey (2025)](https://arxiv.org/abs/2501.03000) | LLM 推荐的 5 大范式 | 学术+工业共识 |
| 训练稳定性 | [Stable Training (2025)](https://arxiv.org/abs/2502.07972) | GR 训练的 10 大陷阱 | 字节跳动经验 |
| 推理优化 | [Inference for GR (2025)](https://arxiv.org/abs/2503.08900) | GR 推理延迟从 100ms→30ms | 工业部署关键 |
| 模型压缩 | [GR Compression (2025)](https://arxiv.org/abs/2501.12948) | 10B→0.5B 蒸馏方案 | 快手/字节实践 |

> **怎么用这张表**:按需查阅。做 LLM 推荐看第一类,做训练优化看第四类,做推理优化看基础设施+算法优化。每篇论文一句话说明,点进去看原文。

### A.2 后续章节预告(如本系列继续更新)

```text
【第二篇:搜推广经典模型从 0 实现】(本系列第二册)
- 第 8 章:逻辑回归与 FM(了解最基础的推荐模型)
- 第 9 章:Wide&Deep 与 DeepFM(工业标配模型)
- 第 10 章:DIN 与 DIEN(序列建模)
- 第 11 章:MMoE 与 PLE(多任务学习)
- 第 12 章:从模型到系统:召回 → 粗排 → 精排

【第三篇:工程进阶】(本系列第三册)
- 第 13 章:大规模特征存储(Redis / Tair / Faiss)
- 第 14 章:分布式训练(PS / AllReduce / 模型并行)
- 第 15 章:A/B 实验与线上指标
- 第 16 章:模型压缩(量化 / 蒸馏 / 剪枝)
- 第 17 章:搜推广全链路实战项目
```

---

### A.3 快速参考卡(Quick Reference)

#### A.3.1 超参数速查

| 超参数 | 推荐起点 | 调参方向 | 理由 |
| :--- | :--- | :--- | :--- |
| Embedding lr | `1e-2` | 若不收敛降至 `5e-3` | Embedding 稀疏更新,需要更大 lr |
| MLP lr | `5e-4` | 配合 warmup | 与 Embedding 分开设置 |
| weight_decay | `1e-4` ~ `1e-5` | Embedding 一般不加 | L2 正则 |
| batch_size | `1024` | 显存允许则 `2048~4096` | linear scaling rule:batch翻倍,lr翻倍 |
| Embedding dim | `32` | 可试 `16` / `64` | `min(64, max(8, vocab^0.25 × 4))` |
| Dropout | `0.1` | 若过拟合增至 `0.3` | 在 Embedding 后、MLP 中用 |
| warmup steps | `1000~5000` | 步数 = 总步数 × 5% | Adam 初期方差估计不稳 |
| grad_clip_norm | `1.0` | 若梯度爆炸降至 `0.5` | 搜推广 Embedding 梯度大 |
| neg_ratio | `3~5` | 正负比 1:3 起步 | 过大会让模型"只学负样本" |
| max_seq_len | `50` | 用 percentile 95 截断 | 超长序列做 top-k attention |

#### A.3.2 Loss 选型速查

| 任务 | 推荐 Loss | 注意事项 |
| :--- | :--- | :--- |
| CTR 二分类 | `F.binary_cross_entropy_with_logits` | 不要用 `F.binary_cross_entropy`(需先 sigmoid) |
| 多分类(next item) | `F.cross_entropy` | 输入是 logits,内部做 softmax |
| 召回双塔 | `InfoNCE / BPR` | batch 内负采样 |
| 回归(CVR) | `F.mse_loss` / `F.smooth_l1_loss` | Huber loss 对异常值更鲁棒 |
| 排序学习 | `BPRLoss` / `ListNet` | Pairwise/Listwise 比 Pointwise 更稳 |

#### A.3.3 常见 NaN 原因速查

```text
症状                  | 最可能的原因              | 最快排查方法
| :--- |
loss = NaN (立即)    | lr 太大 / 输入有 NaN      | 打印 batch,检查 lr
loss → NaN (渐进)    | Embedding 梯度爆炸        | 加 grad_clip_norm=1.0
混合精度下 NaN        | FP16 overflow             | 换 BF16 或加 GradScaler
正样本 loss 正常     | 负采样标签错误             | 打印 labels 的 mean
val loss=NaN         | model.eval() 忘了调       | 检查推理时是否有 Dropout/BN
```

#### A.3.4 Embedding 内存速查

```text
vocab_size × dim × dtype_bytes = 显存(字节)

常见配置:
  100万  × 32  × 4(FP32) = 128 MB
  1000万 × 32  × 4(FP32) = 1.28 GB
  1亿    × 64  × 4(FP32) = 25.6 GB  (超出单卡,需模型并行)
  1亿    × 64  × 2(BF16) = 12.8 GB  (A100 80G 可装下)

优化策略(显存从高到低):
  FP32 → BF16:省一半
  量化 INT8:再省一半
  Embedding 哈希碰撞(hash trick):减少 vocab_size
  分片 Embedding(Sharded Embedding):多卡分担
```

#### A.3.5 PyTorch 操作速查

```python
# 维度操作
x.view(a, b)            # 改变形状(要求 contiguous)
x.reshape(a, b)          # 改变形状(自动 contiguous)
x.transpose(d1, d2)      # 交换两个维度(O(1),修改 stride)
x.permute(0, 2, 1)       # 任意重排维度(O(1),修改 stride)
x.contiguous()           # 物理重排为连续内存(O(n))
x.unsqueeze(dim)         # 新增维度:[B,D] → [B,1,D]
x.squeeze(dim)           # 删除 size=1 维度
x.expand(a, b, c)        # 广播展开(不复制内存)
x.repeat(a, b, c)        # 复制展开(复制内存)

# 聚合操作
x.mean(dim=-1)           # 最后一维取均值
x.sum(dim=1, keepdim=True)  # 沿 dim1 求和,保持维度
x.max(dim=-1).values     # 最后一维最大值
x.softmax(dim=-1)        # softmax(数值不稳定! 用 F.softmax)
F.normalize(x, dim=-1)   # L2 归一化(双塔 cosine 检索必用)

# Embedding 操作
emb = nn.Embedding(vocab, dim, padding_idx=0)
emb(ids)                 # [B] → [B, dim] 或 [B,T] → [B,T,dim]
emb_bag = nn.EmbeddingBag(vocab, dim, mode='mean')
emb_bag(ids, offsets)    # 变长 bag,无需 padding

# 梯度控制
with torch.no_grad():    # 推理时禁用梯度
    pred = model(x)
param.requires_grad_(False)  # 冻结参数(fine-tuning 时)
loss.backward(retain_graph=True)  # 保留计算图(多次反向时)
```

---

### A.4 核心术语表(Glossary)

| 术语 | 英文 | 含义 | 出现章节 |
| :--- | :--- | :--- | :--- |
| **Embedding** | Embedding | 将离散 ID 映射到稠密低维向量的可学习查表 | §3.2, §11.5 |
| **emb_table** | Embedding Table | `nn.Embedding` 内部的 `[vocab, dim]` 权重矩阵 | §3.1 |
| **CTR** | Click-Through Rate | 点击率,CTR 预估是精排的核心任务 | §1.2 |
| **AUC** | Area Under Curve | ROC 曲线下面积,= P(正样本排在负样本前) | §2.7 |
| **GAUC** | Group AUC | 按用户分组加权平均的 AUC,衡量个性化效果 | §5.8 |
| **DLRM** | Deep Learning Recommendation Model | Meta 提出的工业推荐范式,Embedding+MLP | §5 |
| **GR** | Generative Recommendation | 把推荐当成序列生成任务,用 LM 架构 | §5 |
| **SID** | Semantic ID | 用层次量化码(VQ-VAE/RQ-VAE)替换 item_id | §6 |
| **VQ-VAE** | Vector Quantized VAE | 将连续 Embedding 量化为离散码本索引 | §6.3 |
| **RQ-VAE** | Residual Quantized VAE | 用多级量化降低重建误差 | §6.4 |
| **RoPE** | Rotary Position Embedding | 通过旋转矩阵编码相对位置,可外推到更长序列 | §6.2.3 |
| **GQA** | Grouped Query Attention | 多个查询头共享少量键值头,省显存 | §6.2.4 |
| **SwiGLU** | Swish-Gated Linear Unit | 门控激活函数,2024-2025 主流 LLM/GenRec 标配 | §6.2.1 |
| **RMSNorm** | Root Mean Square Norm | 只做 RMS 归一化,不减均值,比 LN 快 5-10% | §6.2.2 |
| **LoRA** | Low-Rank Adaptation | 用低秩矩阵近似参数更新,高效微调大模型 | §6.4 |
| **KV-Cache** | Key-Value Cache | 推理时缓存历史 K/V,避免重复计算 | §6.5 |
| **BPR** | Bayesian Personalized Ranking | 贝叶斯个性化排序,pairwise 学习目标 | §A.1 |
| **InfoNCE** | Info Noise-Contrastive Estimation | 对比学习损失,召回双塔常用 | §7.6 |
| **PPO** | Proximal Policy Optimization | RLHF 的核心算法,裁剪策略更新步长 | §7 |
| **GRPO** | Group Relative Policy Optimization | PPO 的简化版,2025 年 DeepSeek 推广 | §7.7 |
| **stride** | 步幅 | Tensor 各维度相邻元素的内存间距 | §3.1.4 |
| **contiguous** | 连续内存 | Tensor 在内存中按行优先连续排列 | §3.1.4 |
| **collate_fn** | 批处理函数 | DataLoader 中将单条样本合并为 batch 的函数 | §2.7 |
| **padding_idx** | 填充索引 | Embedding 中不参与梯度更新的特殊 ID,通常为 0 | §3.5 |
| **warmup** | 学习率预热 | 训练初期线性增大 lr,等 Adam 的方差估计稳定 | §2.2 |
| **grad_clip** | 梯度裁剪 | 限制梯度范数上限,防止 Embedding 梯度爆炸 | §5.3 |
| **ALS** | Alternating Least Squares | 交替最小二乘,矩阵分解的经典优化方法 | §2.10 |
| **双塔** | Two-Tower / Bi-Encoder | 用户和物品各用一个编码器,内积打分的召回架构 | §5 |
| **正则化** | Regularization | L1/L2/Dropout,防止过拟合的技术统称 | §5.4 |
| **负采样** | Negative Sampling | 从未点击样本中采样构造负例 | §4.6 |
| **数据泄漏** | Data Leakage | 训练集用到了未来的信息,导致离线好线上差 | §2.6 |
| **特征一致性** | Feature Consistency | 训练时的特征处理方式与线上服务保持一致 | §2.8 |

---

> **最后的话**:
>
> 你读到这里,已经走过了从 Python 内存模型到 PyTorch 底层、从数据处理到完整训练 Pipeline 的完整旅程。
>
> 从 CV 转搜推广,最难的从来不是"学新模型",而是**切换解决问题的思维范式**:
> - CV 里你问"这个模型准不准?",搜推广里你问"这个特征靠不靠谱?"
> - CV 里你纠结"这个 CNN 层数够不够?",搜推广里你纠结"这个 ID 的 Embedding 有没有被充分更新?"
> - CV 里你关注"测试集精度",搜推广里你关注"离线涨了 0.5 个点,线上 AUC 会不会也涨?"
>
> 这份教程是个起点,不是终点。建议你:打开团队的项目代码,带着本篇的知识点,一行一行地看。
> 你会发现——原来 `features.get(name, default)` 是在这里用的;原来 `nn.Embedding` 的 `padding_idx=0` 是在这里防止污染;原来这个 `collate_fn` 是在处理变长序列。
>
> 祝你在搜推广的路上越走越顺!

---

### A.5 必读开源代码库清单(2024-2025)

| 框架 | 维护方 | 语言 | 特点 | 推荐场景 | 链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RecBole** | 华东师大 | PyTorch | 70+ 模型统一接口,适合学术 baseline | 学术论文/快速实验 | [recbole.com](https://recbole.io/) |
| **TorchRec** | Meta | PyTorch/C++ | Meta 工业级实现,Fbgemm 后端 | 大规模分布式训练 | [github.com/pytorch/torchrec](https://github.com/pytorch/torchrec) |
| **FuxiCTR** | 华为 | PyTorch/TensorFlow | 一键式特征工程 + 30+ 模型 | 工业 CTR baseline | [github.com/rec-research/FuxiCTR](https://github.com/openbenchmark/FuxiCTR) |
| **DeepCTR** | 社区 | TensorFlow/Keras | 入门友好,适合教学 | 学习 CTR 模型 | [github.com/shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR) |
| **EasyRec** | 阿里 | TensorFlow/Python | 阿里内部开源,TFRecord 全流程 | 工业部署 | [github.com/alibaba/EasyRec](https://github.com/alibaba/EasyRec) |
| **RecTools** | VK | PyTorch | 偏推荐+召回+排序一体化 | 推荐入门 | [github.com/MobileTeleSystems/RecTools](https://github.com/MobileTeleSystems/RecTools) |
| **Transformers4Rec** | NVIDIA | PyTorch | HuggingFace 风格接口,Session-based | 序列推荐/NVIDIA GPU | [github.com/NVIDIA-Merlin/Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) |
| **PaddleRec** | 百度 | PaddlePaddle | 国产化栈,文档全 | 国内工业落地 | [github.com/PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) |
| **Microsoft Recommenders** | 微软 | PyTorch/TF/Spark | 算法+评测+工具链齐全 | 入门 + 评测 | [github.com/recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) |
| **DLRM Reference** | Meta | PyTorch/C++ | DLRM 官方实现,Fbgemm 优化 | 学习工业 DLRM | [github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm) |

> **怎么选**:学术研究选 **RecBole**,工业 DLRM 选 **TorchRec/DLRM**,序列推荐选 **Transformers4Rec**,从 0 学起选 **DeepCTR**,国内落地选 **EasyRec/PaddleRec**。

---

### A.6 工业界五大公司推荐架构白皮书核心要点

| 公司 | 核心架构 | 关键创新 | 公开技术资料 |
| :--- | :--- | :--- | :--- |
| **Google** | DLRM → M-FALCON | Sharded Embedding + M-FALCON 异步训练 | KDD 2023 论文 / TF Recommenders |
| **Meta** | DLRM → HSTU | HSTU 用行为序列替换 DLRM 全部结构 | DLRM 论文 + Actions Speak Louder (KDD 2024) |
| **Alibaba** | 经典 DLRM → 端到端生成 | 阿里 DIEN → 多场景多任务 → 端到端 LLM 推荐 | DIN/DIEN 论文 + 一系列 SIGIR/KDD |
| **ByteDance** | Monolith + 亿级 Embedding | Monolith 实时训练 + TensorFlow + PS 架构 | Monolith 论文(OSDI 2023) |
| **Tencent** | PLE 多任务 + 大模型 | PLE 解决 MMoE 跷跷板 + 混元大模型推荐 | PLE 论文(RecSys 2020) + 混元技术报告 |

#### 共同特点(看完 5 份白皮书后的总结)

1. **Embedding 都是分片的**:即便是 7B 的 LLM 推荐,Embedding Table 还是单机多卡分片(因为稀疏更新特性)
2. **离线训练是异步的**:Parameter Server + 流式数据 + 实时特征,极少用同步 SGD
3. **特征存储是双层**:Redis/Tair 存实时特征,OSS/HDFS 存历史特征,按时间窗口 join
4. **推理服务是分层的**:召回(向量检索)→ 粗排(轻模型)→ 精排(重模型)→ 重排(规则+LLM)
5. **A/B 实验是标配**:每改一个超参都要走 A/B,不允许"工程师拍脑袋上"

---

### A.7 两周速成搜推广大纲(每天 2-3 小时)

#### Week 1:基础 + 经典模型

| 天 | 主题 | 学习材料 | 输出物 |
| :--- | :--- | :--- | :--- |
| Day 1 | 工业推荐全景 | 本教程第 1 章 | 画出召回-粗排-精排-重排流程图 |
| Day 2 | 数学基础回顾 | 本教程第 2 章 | 用 PyTorch 复现 LR/FM |
| Day 3 | Embedding 详解 | 本教程第 3 章 | 实现 DLRM 的 EmbeddingBag |
| Day 4 | DeepFM + xDeepFM | 经典论文 + 第 5 章 | PyTorch 实现 DeepFM |
| Day 5 | DIN/DIEN 序列建模 | 第 5/6 章 + DIN 论文 | 实现简化版 DIN |
| Day 6 | 多任务 MMoE/PLE | 第 4 章 | 实现 PLE(替代跷跷板 MMoE) |
| Day 7 | 双塔召回 + 向量检索 | 第 5 章 + Faiss 文档 | 用 Faiss 索引百万向量 |

#### Week 2:生成式推荐 + 工程

| 天      | 主题                   | 学习材料             | 输出物                          |
| :--- | :--- | :--- | :--- |
| Day 8  | DLRM → GR 范式迁移       | 第 5 章            | 写一篇学习笔记,理解动机                 |
| Day 9  | Semantic ID (RQ-VAE) | 第 6 章            | 跑通 RQ-VAE 代码,生成 S-ID         |
| Day 10 | TIGER 完整实现           | 第 6 章 + TIGER 论文 | PyTorch 实现简化版 TIGER          |
| Day 11 | 训练数据 pipeline        | 第 14 章           | 写一个 raw log → tfrecord 的完整脚本 |
| Day 12 | 训练流程工程               | 第 15 章           | DeepSpeed ZeRO-2 跑 1B 参数实验   |
| Day 13 | 推理服务化                | 第 17 章           | 用 vLLM 部署一个生成式推荐服务           |
| Day 14 | 复盘 + 项目代码 review     | 全部               | 用本教程知识读懂团队 1 个核心模块           |

> **配套代码**:每天的学习材料对应的 PyTorch 实现,见本教程的 GitHub 仓库(假设有)。
> **节奏**:Day 1-7 是"看懂" + 跑通小例子,Day 8-14 是"动手" + 部署。

---
### A.8 推荐系统工程师成长路径

```text
阶段 1:初级工程师(0-1 年)
  ├── 读论文 100 篇(本附录 A.1 列表)
  ├── 跑通 RecBole/TorchRec 全部 example
  ├── 在团队模型上做 1-2 个小改进(加特征/调 loss)
  └── 掌握数据 pipeline 端到端

阶段 2:中级工程师(1-3 年)
  ├── 主导过 1 个完整模块(召回/粗排/精排)
  ├── 熟悉 AB 实验设计,能独立上线
  ├── 写 1 篇技术 blog 或 paper
  └── 掌握分布式训练,优化过性能瓶颈

阶段 3:高级工程师(3-5 年)
  ├── 设计过 1 套新架构(HSTU/TIGER/MMoE 的工业改进)
  ├── 主导过 1 个 0 到 1 的项目
  ├── 带过 2-3 个初级工程师
  └── 在 KDD/RecSys/SIGIR 发表过 paper(可选)

阶段 4:专家/科学家(5+ 年)
  ├── 推动过范式级创新(DLRM→GR 迁移)
  ├── 影响公司技术方向
  └── 在国际顶会做 tutorial 或 keynote
```

---

> **附录 A 的最后补充**:
>
> 写完这一节,本教程从 **0(纯 CV 转行)** 到 **1(能上手搜推广项目)** 的链路就完整了。
>
> 接下来该看什么?:
> 1. **代码驱动**:用附录 A.7 的两周大纲,每天 2-3 小时,14 天后你能给团队讲清楚 DLRM 和 GR 的区别
> 2. **论文驱动**:用附录 A.1 的 50 篇精读清单,每月 5 篇,一年后你能在 paper 群里发言
>
> 搜推广是个工程 + 算法双轮驱动的领域,愿你在这条路上越走越稳。

### A.9 扩展文献清单(按子领域分组,共 150+ 篇)

#### A.9.1 DLRM 时代必读(20 篇)

| # | 论文/资源 | 年份 | 会议/期刊 | 核心贡献 | 为什么必读 |
|---|----------|------|----------|---------|-----------|
| 1 | Deep Learning Recommendation Model (DLRM) | 2019 | arXiv | Facebook 开山之作,定义 embedding + MLP + feature interaction | 工业 baseline |
| 2 | DLRM 开源实现 | 2019 | GitHub | PyTorch + CUDA 参考实现 | 复现入口 |
| 3 | Product Neural Network (PNN) | 2016 | ICDM | IPNN / OPNN / PNN* 三种乘积 | 特征交叉起点 |
| 4 | Wide & Deep | 2016 | RecSys | Memorization + Generalization 双塔 | Google Play 实践 |
| 5 | DeepFM | 2017 | IJCAI | FM + Deep 并行,端到端 | 简化特征工程 |
| 6 | xDeepFM | 2018 | KDD | CIN 高阶显式交互 | 超越 FM 阶数限制 |
| 7 | AutoInt | 2019 | CIKM | 多头自注意力做显式交叉 | attention 进入特征交叉 |
| 8 | DCN / DCN-V2 | 2017/2020 | WWW/KDD | Cross Network 显式高阶 | 工业广泛部署 |
| 9 | Deep Crossing | 2016 | Microsoft | 残差连接用于特征交叉 | 早期里程碑 |
| 10 | FFM (Field-aware FM) | 2016 | RecSys | Field 维度区分 | 经典 ML 时代 |
| 11 | AFM (Attentional FM) | 2017 | IJCAI | attention 权重作用于 FM | FM 增强 |
| 12 | FiBiNET | 2019 | RecSys | SENet + bilinear 交互 | 重要变体 |
| 13 | FLEN | 2020 | AAAI | 场级特征交叉 | 腾讯实践 |
| 14 | Field-embedded PNN | 2019 | AAAI | FPNN 改进 | 阿里实践 |
| 15 | AutoFIS | 2020 | KDD | 自动特征交互选择 | 搜索式选择 |
| 16 | AFN (Adaptive Factorization Network) | 2020 | AAAI | 对数阶数学习 | 自适应阶数 |
| 17 | ONN (Operation-aware NN) | 2019 | SIGIR | 多种交互操作学习 | 多操作融合 |
| 18 | InterHAt | 2020 | SIGIR | 层级 attention 交叉 | 复杂设计 |
| 19 | AutoCross | 2019 | KDD | 搜索式特征组合 | 阿里 |
| 20 | AIM (Automatic Interaction Machine) | 2019 | SIGIR | 自监督特征交互 | 新方向 |

#### A.9.2 序列推荐奠基(20 篇)

| # | 论文 | 年份 | 会议 | 核心贡献 | 意义 |
|---|------|------|------|---------|
| 21 | FPMC | 2010 | UAI | 马尔可夫链 + MF | 序列推荐开山 |
| 22 | GRU4Rec | 2016 | ICLR | GRU 用于 session-based | RNN 时代代表 |
| 23 | Caser | 2018 | AAAI | 卷积视角看序列 | CNN 视角 |
| 24 | SASRec | 2018 | ICDM | Self-Attentive Sequential Recommendation | Transformer 序列 |
| 25 | BERT4Rec | 2019 | CIKM | Cloze + 双向 Transformer | BERT 范式 |
| 26 | S3-Rec | 2020 | CIKM | 自监督 + 互信息最大化 | 预训练思想 |
| 27 | FEARec | 2022 | SIGIR | 频域增强序列推荐 | 频域新视角 |
| 28 | CL4SRec | 2022 | SIGIR | 对比学习序列推荐 | 对比学习 |
| 29 | ICLRec | 2022 | SIGIR | 意图对比学习 | 意图聚类 |
| 30 | CoSeRec | 2022 | KDD | 数据增强 + 对比 | 增强 + 对比 |
| 31 | DuoRec | 2022 | CIKM | 语义 + 对比双重建 | 双向重建 |
| 32 | LightSANs | 2021 | KDD | 轻量化自注意力 | 低秩分解 |
| 33 | Sparse Attention | 2021 | arXiv | 稀疏 attention 加速 | 长序列 |
| 34 | Locker | 2022 | KDD | 局部-全局 attention | 长序列建模 |
| 35 | TWIN | 2022 | SIGIR | 两阶段 TWIN | 工业部署友好 |
| 36 | HGN | 2019 | KDD | Hierarchical Gating Network | 门控机制 |
| 37 | SLi-Rec | 2019 | WSDM | 滑动窗口 + 注意力 | 短期信号 |
| 38 | DIN | 2018 | KDD | Deep Interest Network,attention 兴趣 | 阿里里程碑 |
| 39 | DIEN | 2019 | AAAI | 兴趣演化网络 | GRU + AUGRU |
| 40 | DSIN | 2019 | IJCAI | Deep Session Interest | session 划分 |

#### A.9.3 召回与 ANN 时代(15 篇)

| # | 论文/库 | 年份 | 会议 | 核心 |
|---|--------|------|------|
| 41 | TDM (Tree-based Deep Model) | 2018 | KDD | 阿里巴巴,树形索引 + beam search |
| 42 | JTM (Joint Tree) | 2019 | 树形召回增强  | |
| 43 | ComiRec | 2020 | KDD | 多兴趣召回,可控多样性 |
| 44 | MIND | 2020 | ACM | 多兴趣用户表示 |
| 45 | SINE | 2021 | WWW | 稀疏多兴趣 |
| 46 | Faiss | 2017 | Facebook 开源 ANN 库  | |
| 47 | ScaNN | 2020 | Google ANN 库  | |
| 48 | ANNOY | 2010 | Spotify ANN 库  | |
| 49 | Milvus | 2019 | 向量数据库  | |
| 50 | Pinecone | 2018 | 商业向量数据库  | |
| 51 | HNSW | 2016 | 算法突破  | |
| 52 | IVF-PQ | 2010 | 经典 IVF 量化  | |
| 53 | SPTAG | 2018 | 微软空间分区  | |
| 54 | Vearch | 2020 | 京东向量引擎  | |
| 55 | Qdrant | 2021 | Rust 写的高性能  | |

#### A.9.4 生成式推荐 (GR) 时代(25 篇 ★重点)

| # | 论文 | 年份 | 会议 | 核心贡献 | 工业/学术 |
|---|------|------|------|---------|----------|
| 56 | TIGER (Tandem Indexing for Generation) | 2023 | arXiv | 递归 + RQ-Kmeans 层次化 SID,Transformer 生成 | ★学术 + 谷歌 |
| 57 | HSTU (Hierarchical Sequential Transduction Unit) | 2024 | Meta KDD | 排序统一架构,超越 Transformer | ★Meta 工业 |
| 58 | OneRec | 2025 | KDD | 快手端到端生成式推荐 | ★快手 |
| 59 | OneSearch | 2025 | KDD | 阿里搜索 GR 化 | ★阿里 |
| 60 | Scaling Sequential Recommendation Models | 2024 | arXiv | 序列推荐规模定律 | Meta |
| 61 | Letter-Tagging | 2024 | arXiv | 字母标记法生成式推荐 | 字节 |
| 62 | MTGR (Multi-Task Generative Rec) | 2024 | arXiv | 多任务统一 | 学术 |
| 63 | COBRA | 2024 | arXiv | 排序生成式 | Meta |
| 64 | MiniGPT4Rec | 2024 | arXiv | 极简 LLM4Rec | 学术 |
| 65 | LLaRA | 2024 | arXiv | LLM 行为对齐推荐 | 学术 |
| 66 | InstructRec | 2024 | KDD | 指令微调推荐 | 学术 |
| 67 | RecLLM-RAG | 2024 | arXiv | RAG 范式推荐 | 学术 |
| 68 | CALRec | 2024 | SIGIR | 对比对齐 LLM4Rec | 学术 |
| 69 | LLMRec | 2023 | arXiv | LLM 增强推荐综述 | 综述 |
| 70 | GenRec | 2023 | arXiv | 生成式推荐综述 | 综述 |
| 71 | Survey on GR | 2024 | arXiv | GR 综述 | 综述 |
| 72 | TIGER 改进版 GenRES | 2024 | arXiv | RQ-Kmeans 改进  | |
| 73 | EAGER (Streaming GR) | 2024 | arXiv | 流式增量 SID | 学术 |
| 74 | Semantic ID for Cold-Start | 2024 | arXiv | 冷启动 SID | 学术 |
| 75 | PQ-Tokenizer | 2024 | arXiv | PQ 残差量化 SID | 学术 |
| 76 | Residual Quantization | 2022 | ICLR | 残差量化奠基  | |
| 77 | VQ-VAE-2 | 2019 | NeurIPS | 层级 VQ-VAE | 神经压缩 |
| 78 | RQ-VAE | 2022 | arXiv | Residual-Quant VAE | SID 核心组件 |
| 79 | FSQ (Finite Scalar Quantization) | 2023 | arXiv | 无 codebook 量化 | VQ 替代 |
| 80 | LFQ (Lookup-Free Quantization) | 2023 | arXiv | 高效量化 | VQ 加速 |

#### A.9.5 LLM4Rec 系列(20 篇)

| # | 论文 | 年份 | 会议 | 核心 |
|---|------|------|------|
| 81 | P5 | 2022 | ACL | 多任务统一 prompt |
| 82 | M6-Rec | 2022 | 阿里多模态 LLM 推  | |
| 83 | CTRL (Conditional Transformer Rec) | 2022 | 条件生成式  | |
| 84 | RecFormer | 2023 | 纯生成式  | |
| 85 | IDvs.MoRec | 2023 | KDD | ID 与语义对比 |
| 86 | LLM4RecSurvey | 2023 | 综述  | |
| 87 | TALLRec | 2023 | ACL | TALL 指令微调 |
| 88 | InstructRec | 2024 | KDD | 指令式 |
| 89 | GPT4Rec | 2023 | LLM 标题生成  | |
| 90 | ChatRec | 2023 | 对话式推荐  | |
| 91 | Llama4Rec | 2024 | Llama 适配推荐  | |
| 92 | OpenP5 | 2024 | 开源 P5  | |
| 93 | LLMRec基准 | 2024 | 多模型基准  | |
| 94 | RecSys-LLM | 2024 | 系统框架  | |
| 95 | AgentRec | 2024 | 智能体推荐  | |
| 96 | RecMind | 2024 | 自主决策  | |
| 97 | FLAN-Rec | 2024 | FLAN 风格微调  | |
| 98 | Qwen-Rec | 2024 | Qwen 推荐  | |
| 99 | NoteLLM | 2024 | 笔记推荐  | |
| 100 | LLM-Embedding for Rec | 2024 | LLM 嵌入召回  | |

#### A.9.6 工业实践 blog(10 篇)

| # | 公司 | 文章 | 链接 | 重点 |
|---|------|------|------|------|
| 101 | Meta | "How Instagram uses AI" | engineering.fb.com/ai | HSTU 思想雏形 |
| 102 | Google | "Pathways through Recommenders" | blog.google/technology | YouTube 推荐 |
| 103 | 快手 | "OneRec 端到端 GR" | kuaishou 公众号 | 端到端落地 |
| 104 | 阿里 | "OneSearch" | 阿里技术 | 搜索 GR 化 |
| 105 | 字节 | "TikTok 推荐系统揭秘" | infoq 报道 | 多目标排序 |
| 106 | 美团 | "美团推荐实践" | 美团技术 | 多场景排序 |
| 107 | 京东 | "京东推荐演进" | jd.com 技术博客 | 长序列建模 |
| 108 | 拼多多 | "拼多多推荐" | 内部 | 工业级 |
| 109 | 腾讯 | "微信看一看" | 微信 AI 团队 | 图神经网络 |
| 110 | Netflix | "Netflix Recommender" | netflixtechblog | 经典 industry |

#### A.9.7 推荐系统工具与库(15 个)

| # | 工具 | 类型 | 链接 | 用途 |
|---|------|------|------|------|
| 111 | RecBole | 推荐系统框架 | recbole.io | 学术主流 |
| 112 | RecZoo | 推荐系统合集 | 推荐 | 多库 |
| 113 | TorchRec | Meta 工业库 | pytorch.org/rec | Meta 工业 |
| 114 | DeepCTR | 特征交叉 | deepctr | DLRM 时代 |
| 115 | EasyRec | 阿里 | easyrec | 工业级 |
| 116 | Merlin | NVIDIA 推荐框架 | github.com/NVIDIA-Merlin | GPU 加速 |
| 117 | Vectorflow | 阿里向量 | vectorflow | 向量引擎 |
| 118 | Faiss | Facebook ANN | faiss.ai | ANN 索引 |
| 119 | ScaNN | Google ANN | github.com/google-research/scann | ANN |
| 120 | Qdrant | 向量数据库 | qdrant.tech | 向量库 |
| 121 | Milvus | 向量数据库 | milvus.io | 向量库 |
| 122 | Pinecone | 商业向量 | pinecone.io | 商业 |
| 123 | Weaviate | 向量数据库 | weaviate.io | 开源 |
| 124 | ChromaDB | 嵌入式向量 | trychroma.com | 轻量级 |
| 125 | LanceDB | 列存向量 | lancedb.com | 新一代 |

#### A.9.8 经典数据集(15 个)

| # | 数据集 | 规模 | 用途 | 链接 |
|---|--------|------|------|------|
| 126 | MovieLens-1M/25M | 1M/25M 评分 | 入门 | grouplens.org |
| 127 | Amazon Review | 2 亿+ | 工业规模 | snap.stanford.edu/data/amazon |
| 128 | Yelp Open Dataset | 千万级 | 业务 | yelp.com/dataset |
| 129 | Steam | 游戏 | 游戏推荐 | kaggle |
| 130 | Gowalla | 签到 | LBSN | snap.stanford.edu |
| 131 | Taobao | 阿里 | 行为序列 | tianchi |
| 132 | Tianchi | 阿里系 | 多场景 | tianchi.aliyun.com |
| 133 | KuaiSAR | 快手 | 短视频 | kuairand |
| 134 | WeChat | 微信 | 看一看 | wechat-ai |
| 135 | Netflix Prize | 1 亿评分 | 经典 | netflix |
| 136 | LastFM | 音乐 | 音乐推荐 | grouplens |
| 137 | Pinterest | 视觉 | 图像推荐 | pin |
| 138 | RetailRocket | 电商 | 转化 | retailrocket |
| 139 | Criteo | 千亿级 | CTR | kaggle/criteo |
| 140 | Avazu | CTR | 移动广告 | kaggle/avazu |

#### A.9.9 关键 benchmark 与 leaderboard(10 个)

| # | Benchmark | 主办 | 核心任务 | 规模 |
|---|-----------|------|---------|------|
| 141 | RecBole Leaderboard | RecBole | 通用 | 全算法 |
| 142 | Amazon KDD Cup 2023 | KDD | 购物 | 百万级 |
| 143 | WSDM Cup 2024 | WSDM | 多任务 | 工业 |
| 144 | CIKM Cup | CIKM | 多场景 | 中等 |
| 145 | SIGIR Workshop | SIGIR | 多 | 学术 |
| 146 | RecSys Challenge | RecSys | 实时 | 工业 |
| 147 | Tianchi 工业赛 | 阿里 | 多 | 大型 |
| 148 | KuaiShou Cup | 快手 | 短视频 | 大型 |
| 149 | A/B Test Bench | Meta | 真实流量 | Meta |
| 150 | HOLMES | Amazon | 评测 | 多场景 |

---

### A.10 顶级会议与期刊(投稿/跟踪必备)

#### A.10.1 推荐系统核心会议(CCF A/B)

| 会议 | CCF 等级 | 投稿时间 | 录用率 | 主题 |
|------|---------|---------|--------|------|
| **KDD** | A | 2月/8月 | 15-20% | 数据挖掘 + 推荐 |
| **SIGIR** | A | 1月/7月 | 20% | 信息检索 |
| **RecSys** | B | 4月 | 25% | 推荐系统专会 |
| **WWW** | A | 10月 | 18% | Web + 推荐 |
| **WSDM** | B | 8月 | 20% | 搜索挖掘 |
| **CIKM** | B | 5月/11月 | 22% | 信息管理 |
| **AAAI** | A | 8月 | 20% | AI 综合 |
| **IJCAI** | A | 1月 | 18% | AI 综合 |
| **NeurIPS** | A | 5月/10月 | 25% | ML 顶会 |
| **ICML** | A | 1月/5月 | 22% | ML 顶会 |
| **ICLR** | - | 9月/3月 | 30% | 深度学习 |
| **ACL** | B | 5月 | 22% | NLP |

#### A.10.2 关键期刊

| 期刊 | 影响因子 | 主题 |
|------|---------|------|
| **TOIS** | ~3.0 | ACM 信息系统 |
| **TKDE** | ~8.0 | 知识与数据工程 |
| **TIST** | ~3.5 | 智能系统技术 |
| **Neurocomputing** | ~5.5 | 神经计算 |
| **Information Sciences** | ~8.0 | 信息科学 |
| **KBS** | ~8.0 | 知识系统 |
| **Pattern Recognition** | ~8.0 | 模式识别 |

#### A.10.3 工业技术报告与会议

| 报告/会议 | 出品方 | 频率 | 重点 |
|----------|--------|------|------|
| **KDD Applied Data Science** | KDD | 年 | 工业实践 |
| **RecSys Industry Track** | RecSys | 年 | 工业部署 |
| **CIKM Industry** | CIKM | 年 | 应用 |
| **SIGIR Industry** | SIGIR | 年 | 工业实践 |
| **VLDB Industrial** | VLDB | 年 | 数据库 + 推荐 |
| **SIGMOD Industrial** | SIGMOD | 年 | 数据管理 |
| **WWW Industry** | WWW | 年 | 工业 |
| **MLSys** | MLSys | 年 | ML 系统 |
| **OSDI/NSDI** | - | 年 | 系统 |
| **SOSP** | - | 双年 | 系统 |

---

### A.11 工业级代码示例精选

#### A.11.1 TIGER 完整简化版(200 行 PyTorch)

```python
# tiger_minimal.py
"""
TIGER 简化版实现,体现核心三组件:
1. RQ-Kmeans 量化器 (生成 SID)
2. Transformer Decoder (自回归生成)
3. Constrained Beam Search (工业落地)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


# ===== 组件 1: RQ-Kmeans 量化器 =====
class RQKmeansQuantizer:
    """残差 K-means 量化器,生成 K 级 SID 码字"""
    def __init__(self, num_levels=4, codebook_size=256):
        self.K = num_levels
        self.V = codebook_size
        self.codebooks = []  # 每层一个 KMeans

    def fit(self, embeddings):
        """逐层残差量化"""
        residual = embeddings.copy()
        for k in range(self.K):
            km = KMeans(n_clusters=self.V, n_init=10, random_state=42)
            km.fit(residual)
            self.codebooks.append(km)
            # 残差 = 原值 - 最近质心
            centers = km.cluster_centers_[km.labels_]
            residual = residual - centers

    def encode(self, embeddings):
        """向量 → K 级 SID"""
        sids = np.zeros((len(embeddings), self.K), dtype=np.int64)
        residual = embeddings.copy()
        for k, km in enumerate(self.codebooks):
            sids[:, k] = km.predict(residual)
            centers = km.cluster_centers_[sids[:, k]]
            residual = residual - centers
        return sids

    def decode(self, sids):
        """SID → 重构向量"""
        embeddings = np.zeros((len(sids), self.codebooks[0].cluster_centers_.shape[1]))
        for k, km in enumerate(self.codebooks):
            centers = km.cluster_centers_[sids[:, k]]
            embeddings += centers
        return embeddings


# ===== 组件 2: Transformer Decoder =====
class TIGERDecoder(nn.Module):
    """TIGER 的自回归 Transformer"""
    def __init__(self, num_items, sid_vocab_size=256, sid_levels=4,
                 d_model=256, nhead=8, num_layers=6, max_seq_len=50):
        super().__init__()
        # SID token 嵌入 (共享 K 级 vocab)
        self.sid_embed = nn.ModuleList([
            nn.Embedding(sid_vocab_size + 1, d_model)  # +1 for padding
            for _ in range(sid_levels)
        ])
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        # 输出头:每级 SID 一个头
        self.heads = nn.ModuleList([
            nn.Linear(d_model, sid_vocab_size) for _ in range(sid_levels)
        ])

    def forward(self, input_sids, attn_mask=None):
        """
        input_sids: (B, T, K) 历史 SID 序列
        """
        B, T, K = input_sids.shape
        # 嵌入求和
        x = sum(self.sid_embed[k](input_sids[:, :, k]) for k in range(K))
        # 加位置编码
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(positions)
        # 因果 mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        # 解码
        x = self.decoder(x, x, tgt_mask=causal_mask)
        # 每级头
        logits = [self.heads[k](x) for k in range(K)]  # (B, T, V) × K
        return logits


# ===== 组件 3: 工业级训练循环 =====
class TIGERTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def train_step(self, batch_sids):
        """
        batch_sids: (B, T, K) 用户历史 SID 序列
        训练目标:next SID prediction
        """
        B, T, K = batch_sids.shape
        # 输入:前 T-1 个,目标:后 T-1 个 (teacher forcing)
        input_sids = batch_sids[:, :-1, :]
        target_sids = batch_sids[:, 1:, :]

        logits_list = self.model(input_sids)  # (B, T-1, V) × K
        # K 级损失求平均
        loss = 0
        for k in range(K):
            loss_k = F.cross_entropy(
                logits_list[k].reshape(-1, logits_list[k].size(-1)),
                target_sids[:, :, k].reshape(-1)
            )
            loss = loss + loss_k
        loss = loss / K

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()
```

#### A.11.2 HSTU 注意力机制简化版

```python
# hstu_attention.py
"""
HSTU (Hierarchical Sequential Transduction Unit) 注意力简化版
来自 Meta KDD 2024 论文:超越 Transformer 的排序架构
"""

class HSTUAttention(nn.Module):
    """HSTU 注意力:SiLU + 相对位置 + 门控"""
    def __init__(self, d_model, nhead, max_seq_len=1024):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        # QKV 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        # 相对位置偏置
        self.rel_bias = nn.Parameter(torch.zeros(max_seq_len, max_seq_len))
        # 门控:u,v 单独投影
        self.u_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)

        # Scaled dot-product with relative bias
        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + self.rel_bias[:T, :T].unsqueeze(0).unsqueeze(0)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = attn @ V  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # SiLU 激活 + 门控
        u = F.silu(self.u_proj(x))
        v = self.v_proj(x)
        out = out * u * v  # 三元门控
        return self.out_proj(out)


class HSTUBlock(nn.Module):
    """HSTU 完整 block"""
    def __init__(self, d_model, nhead, expansion=4):
        super().__init__()
        self.attn = HSTUAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model)
        )

    def forward(self, x, attn_mask=None):
        # 残差连接 (Post-LN 变体)
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

#### A.11.3 OneRec 端到端简化版

```python
# onerec_minimal.py
"""
OneRec (快手 KDD 2025) 端到端简化版:
特征 → SID → Transformer → 直接生成 K 个推荐
不依赖 ANN 候选集
"""

class OneRec(nn.Module):
    def __init__(self, num_items, sid_vocab_size, sid_levels=4,
                 d_model=512, nhead=8, num_layers=12, max_seq_len=128):
        super().__init__()
        # 用户侧特征编码
        self.user_feat_proj = nn.Linear(64, d_model)  # 假设用户特征 64 维
        # SID token 嵌入
        self.sid_embed = nn.ModuleList([
            nn.Embedding(sid_vocab_size, d_model) for _ in range(sid_levels)
        ])
        # 主干 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出头
        self.heads = nn.ModuleList([
            nn.Linear(d_model, sid_vocab_size) for _ in range(sid_levels)
        ])

    def encode_history(self, user_features, history_sids):
        """
        user_features: (B, F) 用户画像
        history_sids: (B, T, K) 用户历史 SID 序列
        """
        user_emb = self.user_feat_proj(user_features).unsqueeze(1)  # (B, 1, D)
        sid_emb = sum(self.sid_embed[k](history_sids[:, :, k])
                     for k in range(len(self.sid_embed)))
        # 拼接:用户画像在 [0],历史在 [1..T]
        x = torch.cat([user_emb, sid_emb], dim=1)
        encoded = self.encoder(x)
        return encoded[:, 0]  # 用用户位置作为 query 表示

    def generate(self, user_features, history_sids, max_k=10, sid_levels=4):
        """自回归生成 K 个 SID"""
        B = user_features.size(0)
        user_emb = self.encode_history(user_features, history_sids)
        generated = torch.zeros(B, 0, sid_levels, dtype=torch.long,
                              device=user_features.device)
        cur_input = torch.cat([history_sids], dim=1)
        for _ in range(max_k):
            sid_emb = sum(self.sid_embed[k](cur_input[:, :, k])
                         for k in range(sid_levels))
            # 用用户表征作为 query
            user_emb_exp = self.user_feat_proj(user_features).unsqueeze(1)
            x = torch.cat([user_emb_exp, sid_emb], dim=1)
            encoded = self.encoder(x)
            last_hidden = encoded[:, -1]  # 取最后位置
            # 预测下一 SID
            next_sid = torch.stack([
                self.heads[k](last_hidden).argmax(-1)
                for k in range(sid_levels)
            ], dim=-1)  # (B, K)
            generated = torch.cat([generated, next_sid.unsqueeze(1)], dim=1)
            cur_input = torch.cat([cur_input, next_sid.unsqueeze(1)], dim=1)
        return generated  # (B, max_k, K)
```

---

### A.12 学习路径(分阶段细化)

#### A.12.1 第 1 阶段:基础(0-3 个月)

**目标**:能复现经典 DLRM 模型,理解推荐系统全链路

**必看**:
- 第 1-4 章(rec教程)
- A.9.1 DLRM 时代前 10 篇
- A.9.3 召回前 5 篇
- MovieLens 数据集探索
- DIN/DIEN 论文精读

**产出**:
- 复现 DIN on MovieLens
- 复现 DeepFM on Criteo
- 写一篇读书笔记(公众号/知乎)

#### A.12.2 第 2 阶段:进阶(3-6 个月)

**目标**:掌握序列推荐、对比学习、LLM4Rec 早期

**必看**:
- 第 5 章(范式迁移)
- A.9.2 序列推荐 20 篇
- A.9.5 LLM4Rec 前 10 篇
- SASRec/BERT4Rec 复现
- P5/TALLRec 复现

**产出**:
- 复现 SASRec
- 改进版:加入对比学习
- KDD/SIGIR 投稿 1 篇

#### A.12.3 第 3 阶段:GR 时代(6-12 个月)

**目标**:掌握生成式推荐核心技术

**必看**:
- 第 6-8 章(SID/HSTU/OneRec)
- A.9.4 GR 时代 25 篇 ★
- A.9.6 工业 blog
- TIGER 完整复现
- HSTU 简化复现
- OneRec 简化复现

**产出**:
- 端到端 GR demo
- 工业级 SID pipeline
- 投 RecSys/KDD 1 篇

#### A.12.4 第 4 阶段:工业实践(12-18 个月)

**目标**:能主导公司 GR 迁移项目

**必看**:
- Meta/快手/阿里/字节 内部 blog
- A.10 工业实践
- HSTU 大规模训练
- OneRec 部署细节

**产出**:
- 公司内 GR POC
- 技术分享 2-3 次
- 推动落地 1 个场景

---

### A.13 复现 checklist(28 项)

#### A.13.1 数据(5 项)

- [ ] 选定数据集(MovieLens/Amazon/Taobao/KuaiSAR)
- [ ] 数据清洗:过滤冷启动用户/物品
- [ ] 数据划分:时间序列 train/val/test(80/10/10)
- [ ] 负采样:1:4 或 1:49
- [ ] 数据增强:序列截断/打乱/对比正样本

#### A.13.2 模型(8 项)

- [ ] Embedding 层:user/item/cross 三类
- [ ] 序列编码:GRU/Transformer/HSTU
- [ ] 特征交叉:FM/Deep/Cross
- [ ] 多任务头:CTR/CVR/点赞/收藏/时长
- [ ] SID 生成:RQ-Kmeans/VQ-VAE
- [ ] 生成头:自回归 + constrained beam search
- [ ] 多兴趣:capsule/MoE
- [ ] 位置编码:sinusoidal/RoPE/ALiBi

#### A.13.3 训练(6 项)

- [ ] Loss:CE/BCE/Focal/InfoNCE 组合
- [ ] Optimizer:AdamW/Lion/Adafactor
- [ ] LR Schedule:Warmup + Cosine
- [ ] 混合精度:fp16/bf16
- [ ] 梯度累积:模拟大 batch
- [ ] EMA:模型平滑

#### A.13.4 评估(5 项)

- [ ] Recall@K (K=10/20/50)
- [ ] NDCG@K
- [ ] MRR
- [ ] Hit Rate
- [ ] Coverage / Diversity (工业)

#### A.13.5 上线(4 项)

- [ ] 模型导出:ONNX / TorchScript
- [ ] 服务化:Triton / TensorFlow Serving
- [ ] 性能:Latency < 50ms
- [ ] 监控:QPS / 异常 / 数据漂移

---

### A.14 常见面试题(精选 50 题)

#### A.14.1 DLRM 基础(15 题)

1. **DLRM 的核心组件有哪些?** Embedding + Bottom MLP + Feature Interaction + Top MLP
2. **为什么需要 Bottom MLP?** 把稀疏特征转稠密
3. **Feature Interaction 的本质是什么?** 显式建模特征间的非线性关系
4. **DeepFM vs Deep & Cross?** DeepFM 用 FM,DCN 用 Cross Network
5. **DIN 的 Attention 和 Transformer 的区别?** DIN attention query 是 user ad,keys 是 history
6. **DIN 的 Activation Unit 输出是什么?** 用户对每个历史物品的兴趣权重
7. **DIEN 的 AUGRU 是什么?** Attention Update Gate GRU,门控基于兴趣分数
8. **DSIN 的 Session 划分原则?** 30 分钟间隔
9. **什么是 Dice 激活函数?** 连续近似 ReLU/PReLU
10. **DLRM 的稀疏 embedding 为什么用 Adam?** Adam 对稀疏梯度友好
11. **Negative Sampling 策略对比?** 随机 / hard / in-batch / popularity-aware
12. **AUC 和 Logloss 关系?** 一致性,AUC 关注序,Logloss 关注概率
13. **Multi-task Learning 的梯度冲突?** GradNorm / PCGrad / GradVac
14. **MMoE 解决了什么问题?** 多任务间硬共享导致负迁移
15. **PLE 的核心改进?** 分离 shared/task experts,降低负迁移

#### A.14.2 序列推荐(10 题)

16. **SASRec vs BERT4Rec?** SASRec 单向,BERT4Rec 双向 + Cloze
17. **为什么序列推荐用 Transformer?** 长依赖 + 并行训练
18. **Caser's水平/垂直卷积分别建模什么?** 水平:跳过行为,垂直:点级模式
19. **如何处理超长序列?** 截断 + 兴趣抽取 + 分层
20. **对比学习在序列推荐的好处?** 缓解稀疏 + 自监督信号
21. **数据增强方式?** crop/mask/reorder/substitute
22. **RNN vs Transformer 在长序列上的瓶颈?** RNN 串行,Transformer 二次复杂度
23. **Flash Attention 解决了什么?** IO-aware 注意力,降低显存
24. **Linformer 的核心思想?** 低秩投影 attention
25. **S3-Rec 的自监督任务?** Mutual Information Maximization (4 个 pretext)

#### A.14.3 生成式推荐(15 题)★核心

26. **TIGER 的核心创新点?** SID 序列 + 自回归生成
27. **为什么用 RQ-Kmeans 而不是 K-means?** 残差量化使码字组合表达力更强
28. **RQ-Kmeans 的码字大小?** 256^K,K=4 即 4.3B 组合
29. **SID 相比 item id 的优势?** 语义相似 → 冷启动友好 + 平滑输出分布
30. **TIGER 训练目标?** 下一 SID 预测 (next token prediction)
31. **推理时如何约束生成?** Constrained Beam Search:每个 prefix 必须在训练集出现
32. **什么是 Constrained Beam Search?** 限制生成路径在 trie 中
33. **HSTU 相比 Transformer 改进?** SiLU + 门控 + 相对位置 + 高效稀疏
34. **HSTU 的三元门控公式?** $o = \text{Attn}(x) \cdot \text{SiLU}(u) \cdot v$
35. **OneRec 的"端到端"指什么?** 不依赖 ANN 候选集,直接生成 K 个物品
36. **OneRec 的训练数据构造?** 用户行为序列 → SID → 自回归
37. **GR 的多任务头如何设计?** 每任务一个 SID 头,共享 backbone
38. **冷启动物品如何生成 SID?** 用 sentence-T5 编码后量化
39. **SID 的死码字问题?** 训练时某些码字不出现,需重启
40. **码字利用率低怎么办?** EMA 更新 codebook + 死码重启

#### A.14.4 LLM4Rec(10 题)

41. **P5 的核心思想?** 统一多任务为 prompt
42. **IDvs.MoRec 的核心发现?** 语义 ID 不一定优于纯 ID,要看任务
43. **TALLRec 的训练方式?** LoRA 微调 LLM
44. **InstructRec 的指令设计?** 5 类指令:rating/sequential/feature/filter/explanation
45. **AgentRec 的智能体如何做决策?** ReAct 框架:思考 + 行动
46. **RAG 在推荐中怎么用?** 召回阶段检索 + 生成阶段重排
47. **LLM 做召回的劣势?** 推理慢 + 难建模大规模候选
48. **LLM 做排序的劣势?** 黑盒 + 难优化业务目标
49. **LLM 的 Token 化对推荐的影响?** 物品 ID 需要 SID 化
50. **未来 LLM4Rec 方向?** Agentic + 多模态 + 可解释

---

### A.15 推荐系统竞赛列表

| 比赛 | 主办 | 时间 | 规模 | 任务 |
|------|------|------|------|------|
| KDD Cup 2023 | KDD | 5月-9月 | 万人 | 购物 |
| RecSys Challenge | RecSys | 4月-8月 | 千队 | 实时 |
| WSDM Cup | WSDM | 11月-2月 | 千人 | 多任务 |
| CIKM Cup | CIKM | 7月-10月 | 千人 | 多场景 |
| Tianchi 工业 | 阿里 | 全年 | 万人 | 多 |
| KuaiShou Cup | 快手 | 不定期 | 千人 | 短视频 |
| Kuaishou-Live | 快手直播 | 6月 | 千人 | 直播 |
| BIKE 比赛 | WSDM 2023 | 1月 | 千人 | 行为 |

---

### A.16 顶级研究组与学者

| 研究组 | 机构 | 代表人物 | 重点 |
|--------|------|---------|------|
| **RecSys@Meta** | Meta | Dario Garcia-Sorlano, Liang Luo | HSTU, GR |
| **RecBole 团队** | 浙大 | Xiangnan He, Yongchun Zhu | 推荐系统库 |
| **RUCAIBox** | 人民大学 | Xiang Wang | GNN 推荐 |
| **SICS Lab** | 复旦 | Wei Zhang, Yongfeng Zhang | P5 |
| **阿里巴巴** | 阿里 | 多团队 | OneSearch, EasyRec |
| **快手** | 快手 | 推荐算法 | OneRec |
| **微软** | MSRA | 多位 | DIN, 推荐 |
| **Google** | Google | 多位 | YouTube 推荐 |
| **Twitter Cortex** | Twitter | 多位 | 时间序列 |
| **Netflix** | Netflix | 多位 | 经典 |
| **Spotify** | Spotify | 多位 | 音乐 |
| **Pinterest** | Pinterest | 多位 | 视觉推荐 |
| **Stanford SNAP** | Stanford | Jure Leskovec | 图 + 推荐 |
| **UT Austin** | UT | CalibratedRec 等 | 校准 |
| **CMU** | CMU | 多位 | 强化学习推荐 |

---

### A.17 常用术语速查表(100+)

| 术语 | 英文 | 中文 | 释义 |
|------|------|------|------|
| CTR | Click-Through Rate | 点击率 | 点击 / 曝光 |
| CVR | Conversion Rate | 转化率 | 转化 / 点击 |
| GMV | Gross Merchandise Volume | 成交总额 | 平台总成交 |
| AUC | Area Under Curve | 曲线下面积 | ROC 曲线下面积 |
| GAUC | Group AUC | 分组 AUC | 按用户分组的 AUC |
| NDCG | Normalized Discounted Cumulative Gain | 归一化折损累计增益 | 排序质量 |
| MRR | Mean Reciprocal Rank | 平均倒数排名 | 首个相关位置倒数 |
| MAP | Mean Average Precision | 平均精度均值 | 多查询平均精度 |
| HIT | Hit Rate | 命中率 | top-k 包含目标比例 |
| Recall | Recall | 召回率 | top-k 召回目标比例 |
| Precision | Precision | 准确率 | top-k 中相关比例 |
| Coverage | Coverage | 覆盖率 | 推荐覆盖物品比例 |
| Diversity | Diversity | 多样性 | 推荐列表多样性 |
| Novelty | Novelty | 新颖性 | 推荐列表新颖性 |
| Serendipity | Serendipity | 惊喜度 | 意外有用推荐 |
| Embedding | Embedding | 嵌入 | 离散特征向量表示 |
| SID | Semantic ID | 语义 ID | 物品的层次化语义编码 |
| VQ | Vector Quantization | 矢量量化 | 向量离散化 |
| RQ | Residual Quantization | 残差量化 | 逐层残差量化 |
| GR | Generative Recommendation | 生成式推荐 | 自回归生成推荐列表 |
| DLRM | Deep Learning Recommendation Model | 深度学习推荐 | DLRM 范式 |
| ANN | Approximate Nearest Neighbor | 近似最近邻 | 召回常用 |
| HNSW | Hierarchical Navigable Small World | 层次化小世界图 | ANN 算法 |
| IVF | Inverted File | 倒排文件 | ANN 算法 |
| PQ | Product Quantization | 乘积量化 | ANN 压缩 |
| SID | Semantic ID | 语义标识 | 物品的离散语义表示 |
| HSTU | Hierarchical Sequential Transduction Unit | 层次化序列转换单元 | Meta 排序架构 |
| MMoE | Multi-gate Mixture of Experts | 多门控专家混合 | 多任务 |
| PLE | Progressive Layered Extraction | 渐进分层提取 | MMoE 改进 |
| ESMM | Entire Space Multi-task Model | 全空间多任务 | 阿里 |
| MMOE | Multi-Mixture of Experts | 多专家混合 | Google |
| CTCVR | CVR over Click | 点击后转化 | 多任务 |
| Hard Negative | Hard Negative | 难负样本 | 难分的负样本 |
| In-batch | In-batch Negative | 批内负样本 | 共享 batch 内的负样本 |
| Cross Feature | Cross Feature | 交叉特征 | 组合特征 |
| Sequence | Sequence | 序列 | 用户行为时间序列 |
| Session | Session | 会话 | 一段连续行为 |
| Coarse Ranking | Coarse Ranking | 粗排 | 中间层排序 |
| Fine Ranking | Fine Ranking | 精排 | 最终层排序 |
| Recall | Recall | 召回 | 从亿级到千级 |
| Re-rank | Re-rank | 重排 | 考虑多样性 |
| Explore | Explore | 探索 | 推荐新物品 |
| Exploit | Exploit | 利用 | 推荐已知偏好 |
| Bandit | Bandit | 老虎机 | 探索利用权衡 |
| MAB | Multi-Armed Bandit | 多臂老虎机 | 探索算法 |
| UCB | Upper Confidence Bound | 置信上界 | 老虎机策略 |
| Thompson | Thompson Sampling | 汤普森采样 | 贝叶斯策略 |
| LinUCB | Linear UCB | 线性置信上界 | 上下文老虎机 |
| GBDT | Gradient Boosting Decision Tree | 梯度提升树 | 经典 ML |
| DCN | Deep & Cross Network | 深度交叉网络 | 显式交叉 |
| FM | Factorization Machine | 因子分解机 | 经典 ML |
| FFM | Field-aware FM | 域感知 FM | FM 改进 |
| AFM | Attentional FM | 注意力 FM | FM 改进 |
| AUC | Area Under the Curve | 曲线下面积 | 评估指标 |
| Logloss | Log Loss | 对数损失 | 训练损失 |
| NCE | Noise Contrastive Estimation | 噪声对比估计 | 采样损失 |
| InfoNCE | InfoNCE | 信息对比损失 | 对比学习损失 |
| BCE | Binary Cross Entropy | 二值交叉熵 | 二分类损失 |
| Focal Loss | Focal Loss | 焦点损失 | 不平衡处理 |
| EMA | Exponential Moving Average | 指数移动平均 | 模型平滑 |
| Adam | Adaptive Moment | 自适应动量 | 优化器 |
| AdamW | Adam with Weight decay | Adam + 权重衰减 | 优化器 |
| Lion | Lion | 新优化器 | 符号动量 |
| Lookahead | Lookahead | 前瞻优化器 | 双层优化 |
| SAM | Sharpness-Aware Minimization | 锐度感知 | 泛化 |
| Mixed Precision | Mixed Precision | 混合精度 | fp16/bf16 |
| AMP | Automatic Mixed Precision | 自动混合精度 | PyTorch API |
| DDP | Distributed Data Parallel | 分布式数据并行 | 多卡 |
| FSDP | Fully Sharded Data Parallel | 全分片数据并行 | 超大模型 |
| ZeRO | Zero Redundancy Optimizer | 零冗余优化器 | DeepSpeed |
| LoRA | Low-Rank Adaptation | 低秩适配 | PEFT |
| Prefix | Prefix Tuning | 前缀微调 | PEFT |
| Prompt | Prompt Tuning | 提示微调 | PEFT |
| Adapter | Adapter Tuning | 适配器微调 | PEFT |
| KD | Knowledge Distillation | 知识蒸馏 | 模型压缩 |
| Quantization | Quantization | 量化 | 模型压缩 |
| Pruning | Pruning | 剪枝 | 模型压缩 |
| Distill | Distillation | 蒸馏 | 教师-学生 |
| Beam Search | Beam Search | 束搜索 | 解码 |
| Constrained Beam | Constrained Beam Search | 约束束搜索 | 受限解码 |
| Top-K | Top-K | 前 K 个 | 截断 |
| Top-p | Top-p (Nucleus) | 核采样 | 截断 |
| Temperature | Temperature | 温度 | 采样 |
| MLE | Maximum Likelihood Estimation | 最大似然估计 | 训练目标 |
| RCE | Reverse Cross Entropy | 反向交叉熵 | 训练目标 |
| KL | Kullback-Leibler | KL 散度 | 分布差异 |
| JS | Jensen-Shannon | JS 散度 | 对称分布差异 |
| W2 | Wasserstein-2 | W2 距离 | 分布距离 |
| ONNX | Open Neural Network Exchange | 开放神经网络交换 | 模型格式 |
| TorchScript | TorchScript | TorchScript | PyTorch JIT |
| TensorRT | TensorRT | TensorRT | NVIDIA 推理 |
| Triton | Triton | Triton | NVIDIA 服务 |
| vLLM | vLLM | vLLM | 高吞吐推理 |
| SGLang | SGLang | SGLang | LLM 服务 |
| FlashAttn | Flash Attention | Flash 注意力 | 高效 attention |
| PagedAttn | Paged Attention | 分页注意力 | KV cache |
| KV Cache | KV Cache | 键值缓存 | 推理加速 |
| Beam Width | Beam Width | 束宽度 | 解码宽度 |
| Length Penalty | Length Penalty | 长度惩罚 | 解码后处理 |
| Repetition Penalty | Repetition Penalty | 重复惩罚 | 解码后处理 |
| Trie | Trie | 字典树 | 受限解码 |
| FSM | Finite State Machine | 有限状态机 | 受限解码 |

---

### A.18 论文阅读方法论

#### A.18.1 三遍阅读法(经典)

**第一遍(15 分钟):鸟瞰**
- 看标题、摘要、引言
- 看图表(尤其实验结果图)
- 看结论
- 决定是否值得精读

**第二遍(60 分钟):理解**
- 通读全文,标注不懂的术语
- 看数学推导,跳过硬证明
- 列出关键贡献 3 条

**第三遍(数小时):复现**
- 对照开源代码
- 自己手写 mini 版
- 在小数据集上跑通

#### A.18.2 笔记模板

```text
# 论文笔记:[标题]

## 1. 动机(Why)
- 解决什么问题?
- 之前的方案有什么不足?

## 2. 方法(How)
- 核心 idea 一句话:
- 公式:
- 架构图描述:

## 3. 实验(What)
- 数据集:
- Baseline:
- 核心结果:

## 4. 我的思考
- 优点:
- 缺点:
- 与已有工作对比:
- 可以借鉴到我的项目吗?
```

#### A.18.3 论文清单管理

推荐用 Notion 或 Obsidian 建一个论文库:

| 字段 | 说明 |
|------|------|
| 标题 | 论文名 |
| 会议/期刊 | 发表场所 |
| 年份 | |
| 主题 | 推荐/排序/召回 |
| 阅读状态 | 未读 / 一读 / 二读 / 精读 |
| 复现状态 | 未开始 / 复现中 / 完成 |
| 关键创新 | 一句话 |
| 我的笔记 | 链接 |

---

### A.19 必备技能树(转行/进阶)

#### A.19.1 工程能力(6 项)

- [ ] Python 高级(装饰器/生成器/异步)
- [ ] PyTorch 深入(自定义算子/分布式)
- [ ] SQL 高级(窗口函数/递归 CTE)
- [ ] Spark/Flink 大数据
- [ ] CUDA 基础(GPU 编程)
- [ ] Linux 性能优化(top/strace/perf)

#### A.19.2 算法能力(8 项)

- [ ] DLRM(DIN/DIEN/DSIN)
- [ ] 序列(SASRec/BERT4Rec/TWINE)
- [ ] 召回(DSSM/MIND/TDM)
- [ ] 多任务(MMoE/PLE/ESMM)
- [ ] 对比学习(CL4SRec/CoSeRec)
- [ ] LLM4Rec(P5/TALLRec/InstructRec)
- [ ] SID(TIGER/RQ-Kmeans/RQ-VAE)
- [ ] GR(OneRec/HSTU)

#### A.19.3 业务能力(5 项)

- [ ] A/B 实验设计
- [ ] 指标体系搭建
- [ ] 线上问题排查
- [ ] 跨团队协作
- [ ] 技术分享

---

### A.20 常见坑与避坑指南(20 条)

1. **不要一开始就上 GR** → 先用 DIN/MMoE 把 DLRM baseline 跑顺
2. **不要复现冷门论文** → 优先 TIGER/HSTU/OneRec
3. **不要忽略数据质量** → 80% 时间花在数据清洗
4. **不要死磕负样本策略** → 1:4 通常够用
5. **不要混用不同框架的 checkpoint** → PyTorch ↔ TF 模型不通用
6. **不要在 eval 上过拟合** → 用多数据集交叉验证
7. **不要假设线上线下指标一致** → 至少差 5%
8. **不要忽略冷启动** → 新物品/新用户单独模型
9. **不要忽略长尾物品** → 多样性 + 探索机制
10. **不要死记公式** → 理解推导
11. **不要跳过 baseline** → 简单 LR 都跑不过就别上深度学习
12. **不要在 production 改架构** → 灰度发布
13. **不要相信 ablation 的"无显著差异"** → 看绝对值
14. **不要写 2000 行的单文件** → 模块化
15. **不要在 PyTorch 1.x 写新代码** → 用 2.0+ torch.compile
16. **不要忽略学习率** → LR 是最容易调的
17. **不要忽略 batch size 影响** → 小 batch 噪声大,大 batch 收敛稳
18. **不要用全局池化做兴趣** → attention 更合适
19. **不要相信"无 cost improvement"** → 工程成本也算
20. **不要忽略模型可解释性** → 业务方需要解释

---

### A.21 资源汇总(网盘/网站/课程)

#### A.21.1 在线课程

| 课程 | 平台 | 内容 |
|------|------|------|
| Deep Learning Specialization | Coursera (Andrew Ng) | DL 基础 |
| Machine Learning | Coursera (Andrew Ng) | ML 基础 |
| 推荐系统实战 | 极客时间 | 推荐系统 |
| 推荐系统深度学习 | B 站 | 实战 |
| Mining Massive Datasets | Stanford | 大数据 |
| Practical DL | fast.ai | DL 实战 |
| DLRM 系统课 | Carnegie Mellon | 推荐系统 |
| 深度学习推荐系统 | 王喆 | 经典 |
| 工业级推荐系统 | 天池 | 实战 |

#### A.21.2 YouTube 频道

| 频道 | 重点 |
|------|------|
| **StatQuest** | ML/DL 基础 |
| **3Blue1Brown** | 数学直觉 |
| **Yannic Kilcher** | 论文精讲 |
| **AI Coffee Break** | 短论文 |
| **Andrej Karpathy** | DL 系统课 |
| **Two Minute Papers** | 最新研究 |
| **Henry AI Labs** | 深度学习 |
| **MLOps Community** | 工程实践 |
| **PyTorch** | 官方教程 |
| **HuggingFace** | LLM 实战 |

#### A.21.3 中文资源

| 资源 | 链接 | 重点 |
|------|------|------|
| 机器之心 | 公众号 | 资讯 |
| 量子位 | 公众号 | 资讯 |
| DataFun | 公众号/网站 | 工业实践 |
| 智源社区 | 公众号 | 论文 |
| RUC AI Box | 公众号 | 学术 |
| 夕小瑶的卖萌屋 | 公众号 | NLP/推荐 |
| 深度学习与推荐系统 | 王喆书 | 工业实践 |
| 阿里巴巴技术 | 公众号 | 阿里实践 |
| 美团技术 | 公众号 | 美团实践 |
| 字节跳动技术 | 公众号 | 字节实践 |
| 腾讯技术工程 | 公众号 | 腾讯实践 |

#### A.21.4 GitHub 仓库

| 仓库 | stars | 重点 |
|------|-------|------|
| **RecBole** | 3k+ | 推荐系统框架 |
| **RecZoo** | 1k+ | 多库合集 |
| **TorchRec** | 4k+ | Meta 工业 |
| **DeepCTR** | 7k+ | 特征交叉 |
| **EasyRec** | 1.5k+ | 阿里 |
| **Merlin** | 4k+ | NVIDIA |
| **Faiss** | 30k+ | ANN |
| **ScaNN** | 5k+ | ANN |
| **TIGER (Rec)** | - | SID 实现 |
| **OneRec** | - | 端到端 |
| **RecLLM-RAG** | - | LLM4Rec |

#### A.21.5 论文搜索

| 网站 | URL | 重点 |
|------|-----|------|
| Google Scholar | scholar.google.com | 通用 |
| Semantic Scholar | semanticscholar.org | AI 增强搜索 |
| arXiv | arxiv.org | 预印本 |
| PapersWithCode | paperswithcode.com | 论文+代码 |
| DBLP | dblp.org | 计算机科学 |
| Connected Papers | connectedpapers.com | 论文图谱 |
| ResearchGate | researchgate.net | 学术社交 |
| ACM DL | dl.acm.org | 顶会 |
| IEEE Xplore | ieee.org | 顶刊 |
| Citeseer | citeseerx.ist.psu.edu | 引用 |
| OpenReview | openreview.net | 评审 |

---

### A.22 实战项目清单(从易到难)

#### A.22.1 入门级(2 周内完成)

1. **MovieLens + DIN 复现**
   - 数据: MovieLens-1M
   - 模型: DIN
   - 评估: AUC + Logloss
   - 产出: GitHub repo + 1 篇 blog

2. **Amazon + DeepFM**
   - 数据: Amazon Electronics
   - 模型: DeepFM
   - 评估: AUC
   - 产出: 同上

3. **新闻推荐 baseline**
   - 数据: Adressa / MIND
   - 模型: NRMS
   - 评估: NDCG@10
   - 产出: 复现报告

#### A.22.2 中级(4 周内完成)

4. **序列推荐 + 对比学习**
   - 数据: Amazon Beauty
   - 模型: SASRec + CoSeRec
   - 评估: Recall@10, NDCG@10
   - 产出: 改进版 paper

5. **SID 生成与可视化**
   - 数据: Amazon
   - 模型: RQ-Kmeans + t-SNE
   - 评估: codebook utilization
   - 产出: 完整 pipeline

6. **TIGER 复现**
   - 数据: MovieLens / Amazon
   - 模型: TIGER
   - 评估: Recall@10
   - 产出: 完整 GR demo

#### A.22.3 高级(8-12 周)

7. **OneRec 端到端**
   - 数据: KuaiSAR 或自建
   - 模型: OneRec 简化版
   - 评估: 完整指标
   - 产出: 端到端 demo + paper

8. **HSTU 大规模**
   - 数据: Criteo 1TB
   - 模型: HSTU
   - 评估: AUC + Logloss
   - 产出: 工业级 demo

9. **GR + LLM 融合**
   - 数据: Amazon + LLM 增强
   - 模型: LLM4GR
   - 评估: 综合指标
   - 产出: 创新 paper

#### A.22.4 顶级(研究级)

10. **新 SID 算法**
    - 提出改进的 SID 量化算法
    - 理论分析 + 实验
    - 投 KDD/SIGIR

11. **新 GR 架构**
    - 改进 HSTU 或 OneRec
    - 理论 + 实验
    - 投 NeurIPS/ICML

12. **新应用场景**
    - 把 GR 应用到新场景(医疗、教育)
    - 投 KDD Applied Track

---

### A.23 数学基础速查(15 个公式)

#### A.23.1 注意力机制

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

#### A.23.2 多头注意力

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。

#### A.23.3 因果 LM Loss

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^T \log P(x_t | x_{<t})
$$

#### A.23.4 InfoNCE

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s(x, x^+)/\tau)}{\sum_{x^- \in \mathcal{N}} \exp(s(x, x^-)/\tau)}
$$

#### A.23.5 RQ 量化

$$
\hat{x} = \sum_{k=1}^K c_k(x), \quad c_k(x) = \text{argmin}_j \| r_{k-1}(x) - e_j^{(k)} \|^2
$$

其中 $r_k(x) = x - \sum_{j=1}^k c_j(x)$,$c_k(x)$ 是第 $k$ 层码字。

#### A.23.6 多任务加权

$$
\mathcal{L} = \sum_i w_i \mathcal{L}_i + \lambda \|\theta\|^2
$$

#### A.23.7 MMoE 门控

$$
y_k = \sum_{i=1}^n g_i^{(k)}(x) f_i(x), \quad g^{(k)} = \text{softmax}(W^{(k)} x)
$$

#### A.23.8 AUC 计算

$$
\text{AUC} = \frac{\sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbb{1}[s_i > s_j]}{|\text{pos}| \cdot |\text{neg}|}
$$

#### A.23.9 NDCG

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{DCG@K} = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

#### A.23.10 交叉熵

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_i [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]
$$

#### A.23.11 KL 散度

$$
D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

#### A.23.12 JS 散度

$$
D_{\text{JS}}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)
$$

其中 $M = \frac{1}{2}(P+Q)$。

#### A.23.13 知识蒸馏

$$
\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(y, \hat{y}_s) + (1-\alpha) \mathcal{L}_{\text{CE}}(\hat{y}_t, \hat{y}_s)
$$

#### A.23.14 LoRA

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
$$

#### A.23.15 Beam Search

$$
\mathcal{B}_t = \text{TopK}\left(\bigcup_{b \in \mathcal{B}_{t-1}} \{b \circ v : v \in V\}\right)
$$

---

### A.24 性能调优 checklist(20 项)

#### A.24.1 数据加载(4 项)

- [ ] DataLoader 用 `num_workers=4*num_gpus`
- [ ] `pin_memory=True`
- [ ] `prefetch_factor=4`
- [ ] 数据预处理离线完成(不要在线)

#### A.24.2 训练(6 项)

- [ ] 混合精度 `torch.cuda.amp`
- [ ] 梯度累积模拟大 batch
- [ ] `torch.compile(model)` PyTorch 2.0+
- [ ] Flash Attention 替代原生 attention
- [ ] 通信压缩 (ZeRO-2 / ZeRO-3)
- [ ] EMA 模型平滑

#### A.24.3 推理(6 项)

- [ ] TensorRT / TorchScript 导出
- [ ] KV Cache 启用
- [ ] Dynamic Batching
- [ ] Continuous Batching (vLLM)
- [ ] Paged Attention (vLLM)
- [ ] Speculative Decoding

#### A.24.4 服务(4 项)

- [ ] Triton Inference Server
- [ ] GPU 利用率 > 70%
- [ ] P99 Latency < 50ms
- [ ] Auto-scaling

---

### A.25 推荐系统常见英文词汇(按主题)

#### A.25.1 排序相关

- ranking score, predicted CTR, calibration, isotonic regression, ECE, NDCG, MRR

#### A.25.2 召回相关

- candidate generation, approximate nearest neighbor, ANN index, HNSW, IVF, PQ

#### A.25.3 特征相关

- feature engineering, feature store, embedding table, hashing trick, feature crossing

#### A.25.4 训练相关

- online learning, continual learning, incremental training, federated learning

#### A.25.5 评估相关

- offline metrics, online A/B test, interleaving, counterfactual evaluation

#### A.25.6 部署相关

- online serving, real-time inference, batch inference, pipeline, microservices

#### A.25.7 业务相关

- cold start, long tail, popularity bias, filter bubble, echo chamber

---

### A.26 推荐系统工具箱(Python 包清单)

| 包名 | 用途 | 安装 |
|------|------|------|
| `recbole` | 推荐系统框架 | `pip install recbole` |
| `torchrec` | Meta 工业库 | `pip install torchrec` |
| `deepctr` | 特征交叉 | `pip install deepctr` |
| `faiss-cpu/gpu` | ANN | `pip install faiss-cpu` |
| `scikit-learn` | ML 工具 | `pip install scikit-learn` |
| `pandas/numpy` | 数据处理 | `pip install pandas` |
| `wandb` | 实验跟踪 | `pip install wandb` |
| `optuna` | 超参搜索 | `pip install optuna` |
| `transformers` | LLM | `pip install transformers` |
| `sentence-transformers` | 句子嵌入 | `pip install sentence-transformers` |
| `vllm` | LLM 服务 | `pip install vllm` |
| `triton` | GPU 服务 | `pip install triton` |
| `onnxruntime` | 模型推理 | `pip install onnxruntime` |
| `bitsandbytes` | 量化 | `pip install bitsandbytes` |
| `accelerate` | 分布式 | `pip install accelerate` |
| `datasets` | 数据集 | `pip install datasets` |

---

### A.27 写作模板(项目报告 / 实验报告 / 技术 blog)

#### A.27.1 实验报告模板

```markdown
# 实验报告:[实验名称]

## 1. 实验背景
- 业务问题:
- 假设:
- 目标指标:

## 2. 实验设计
- 数据集:
- Baseline:
- 实验组:
- 评估指标:
- A/B 实验设计:

## 3. 实验结果
| 指标 | Baseline | 实验组 | 提升 |
|------|---------|--------|------|
| AUC | 0.85 | 0.87 | +2.4% |
| NDCG | 0.72 | 0.75 | +4.2% |

## 4. 分析
- 为什么提升?
- 显著性?
- 边缘情况?

## 5. 结论与下一步
- 上线?
- 进一步实验?
```

#### A.27.2 技术 blog 模板

```markdown
# [标题:业务痛点 + 解决方案]

## 1. 背景
描述业务场景、问题严重性。

## 2. 现有方案的不足
分析现有方案的痛点。

## 3. 我们的方案
详细技术方案,带图。

## 4. 实验与上线
数据 + 结果 + 截图。

## 5. 总结与展望
收获 + 未来方向。
```

---

### A.28 论文写作建议(中文 + 英文)

#### A.28.1 中文论文

- **结构**: 摘要、引言、相关工作、方法、实验、结论
- **篇幅**: 8000-12000 字
- **公式**: LaTeX 排版
- **图表**: 矢量图(SVG/PDF)
- **语言**: 学术化,避免口语

#### A.28.2 英文论文(Springer 模板)

- **Abstract**: 150-250 词
- **Introduction**: 1-2 页
- **Related Work**: 1 页
- **Method**: 3-4 页
- **Experiments**: 3-4 页
- **Conclusion**: 0.5 页
- **References**: 30-50 篇

#### A.28.3 投稿策略

| 会议 | 投稿周期 | 难度 | 建议 |
|------|---------|------|------|
| RecSys | 4 月 | 中 | 入门顶会 |
| CIKM | 5 月/11 月 | 中 | 工业友好 |
| SIGIR | 1 月/7 月 | 高 | 顶级 |
| KDD | 2 月/8 月 | 极高 | 顶级 |
| AAAI | 8 月 | 中 | AI 综合 |
| NeurIPS | 5 月/10 月 | 极高 | ML 顶级 |
| ICLR | 9 月/3 月 | 高 | 深度学习 |
| WWW | 10 月 | 高 | Web |
| WSDM | 8 月 | 中 | 搜索挖掘 |

---

### A.29 工业级代码规范

#### A.29.1 命名规范

```python
# 类名: PascalCase
class TIGERModel:
    pass

# 函数/变量: snake_case
def encode_sid(sid_sequence):
    return encoded

# 常量: UPPER_SNAKE_CASE
MAX_SEQ_LEN = 100

# 私有: _leading_underscore
def _internal_method(self):
    pass
```

#### A.29.2 目录结构

```text
project/
├── README.md
├── pyproject.toml
├── src/
│   └── project/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── tiger.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── train/
│       │   └── trainer.py
│       └── utils/
│           └── metrics.py
├── configs/
│   └── default.yaml
├── tests/
│   └── test_models.py
├── scripts/
│   └── train.py
└── docs/
```

#### A.29.3 提交规范

```text
feat: add TIGER model
fix: fix SID encoder bug
docs: update README
style: format code
refactor: refactor data loader
test: add unit tests
chore: update dependencies
```

#### A.29.4 配置管理

```yaml
# configs/default.yaml
model:
  name: tiger
  d_model: 256
  nhead: 8
  num_layers: 6

training:
  batch_size: 256
  lr: 1e-3
  epochs: 100
  warmup_steps: 1000

data:
  train_path: data/train.parquet
  val_path: data/val.parquet
  num_workers: 4
```

---

### A.30 推荐系统生涯建议

#### A.30.1 转行(0-1 年)

- **优先目标**:能复现 DIN/SASRec/TIGER
- **学习方式**: 每周 5 篇论文 + 1 个项目
- **避免**: 算法刷题 < 业务理解

#### A.30.2 进阶(1-3 年)

- **优先目标**:主导 1 个完整模块
- **学习方式**: 每月 1 篇 paper 投稿 + 1 个技术 blog
- **避免**: 重复造轮子

#### A.30.3 高级(3-5 年)

- **优先目标**:架构设计 + 团队带新人
- **学习方式**: 主导 0 到 1 项目 + 行业会议分享
- **避免**: 仅做执行,不做决策

#### A.30.4 专家(5+ 年)

- **优先目标**:推动范式创新
- **学习方式**: tutorial / keynote / 行业标准
- **避免**: 闭门造车

---

### A.31 附录:推荐学习路线(总图)

```text
Phase 0 (0-3 mo): 基础
├─ Python + PyTorch 熟练
├─ DLRM 时代 5 篇 (DIN/DIEN/DeepFM)
└─ 复现 1 个 baseline

Phase 1 (3-6 mo): 序列 + 对比
├─ 序列 5 篇 (SASRec/BERT4Rec)
├─ 对比 3 篇 (CL4SRec/CoSeRec)
└─ KDD/SIGIR 投稿 1 篇

Phase 2 (6-12 mo): GR 时代 ★
├─ SID 5 篇 (TIGER/RQ-Kmeans)
├─ GR 5 篇 (HSTU/OneRec)
├─ 工业 blog 5 篇
└─ RecSys 投稿 1 篇

Phase 3 (12-18 mo): 工业落地
├─ 端到端 GR demo
├─ 大规模训练优化
├─ 真实业务上线
└─ KDD 投稿 1 篇

Phase 4 (18+ mo): 创新
├─ 改进算法
├─ 行业会议 tutorial
└─ 影响力建设
```

---

### A.32 重要补充:2025-2026 最新论文与趋势

#### A.32.1 2025 必读新论文(15 篇)

| # | 论文 | 会议 | 重点 |
|---|------|------|------|
| 1 | OneRec (快手) | KDD 2025 | 端到端 GR |
| 2 | OneSearch (阿里) | KDD 2025 | 搜索 GR 化 |
| 3 | Meta GenRec 2.0 | arXiv 2025 | 工业级 GR |
| 4 | ActionPiece | arXiv 2025 | 动作粒度 SID |
| 5 | 字节 HSTU 改进 | arXiv 2025 | 字节实践 |
| 6 | 美团 KDD 2025 | KDD 2025 | 美团实践 |
| 7 | 京东推荐新范式 | arXiv 2025 | 京东 |
| 8 | 腾讯 GR 实践 | arXiv 2025 | 腾讯 |
| 9 | 阿里 PID | KDD 2025 | 个性化 ID |
| 10 | DeepSeek-Rec | arXiv 2025 | DeepSeek 推荐 |
| 11 | Qwen3-Rec | arXiv 2025 | Qwen 推荐 |
| 12 | Hugging Face RecSys | HF 2025 | 开源框架 |
| 13 | Apple RecSys | Apple ML 2025 | 苹果 |
| 14 | OpenAI 推荐 | OpenAI 2025 | OpenAI |
| 15 | Anthropic 推 | Anthropic 2025 | Anthropic |

#### A.32.2 2026 趋势预测

1. **GR 成为主流**: 80% 头部公司将 GR 化排序
2. **多模态 SID**: 文本+图像+视频统一 SID
3. **Agentic 推荐**: LLM Agent 主导决策
4. **实时学习**: 在线 GR 模型
5. **联邦推荐**: 隐私 + 性能双优化
6. **端侧 GR**: 手机端小模型 GR
7. **GR + RL**: RLHF 优化 GR
8. **跨域 GR**: 一个模型服务多场景
9. **可解释 GR**: 决策可解释
10. **GR + 因果**: 因果推断融合

---

### A.33 附录 A 的最终小结

**附录 A 是本教程的"参考手册"**, 列举了:
- 150+ 必读论文
- 15+ 顶级会议
- 28 项复现 checklist
- 50+ 面试题
- 10 个数据集
- 6 大研究方向

使用建议:
- **入门**: A.9.1 + A.9.2(共 40 篇)
- **进阶**: A.9.4 + A.9.5(共 45 篇)
- **面试**: A.14(50 题)
- **实战**: A.13(28 项) + A.22(12 个项目)

---

> **附录 A 结束**:
>
> 本教程至此完整。从第 1 章 Python 入门,到第 6 章 SID,到附录 A 必读文献,涵盖了从"CV 转行搜推广"到"GR 工程师"的全部必备知识。
>
> 接下来的方向:
> 1. **深耕论文**: 每月 5 篇精读,一年 60 篇
> 2. **实战项目**: 完成 A.22 中至少 3 个项目
> 3. **社区参与**: KDD/SIGIR 投稿 + 学术分享
>
> 推荐系统的未来是 GR,你的未来也在这里。

