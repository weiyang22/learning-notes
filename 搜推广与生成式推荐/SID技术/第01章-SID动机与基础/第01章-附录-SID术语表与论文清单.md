---
title: "第01章-附录-SID术语表与论文清单"
tags:
  - 搜推广
  - 附录
  - SID术语表与论文清单
description: "本章要点：- [§1 术语表 (A-Z, 100+ 词条)](#1-术语表-a-z-100-词条)"
chapter_id: SID技术-1
last_updated: 2026-07-14
---

# 第01章-附录-SID术语表与论文清单


## 目录

- [§1 术语表 (A-Z, 100+ 词条)](#1-术语表-a-z-100-词条)
- [§2 论文清单 (按时间线, 2017-2026)](#2-论文清单-按时间线-2017-2026)
- [§3 论文清单 (按公司)](#3-论文清单-按公司)
- [§4 论文清单 (按方法)](#4-论文清单-按方法)
- [§5 论文清单 (按场景)](#5-论文清单-按场景)
- [§6 速查公式 (50+ 公式)](#6-速查公式-50-公式)
- [§7 关键引文 (Top 20 必读)](#7-关键引文-top-20-必读)
- [§8 工业里程碑](#8-工业里程碑)
- [§9 复现资源索引](#9-复现资源索引)
- [§10 速查图示](#10-速查图示)
- [§11 中英术语对照 (扩展版)](#11-中英术语对照-扩展版)
- [§12 缩写词典](#12-缩写词典)
- [§13 论文下载链接索引](#13-论文下载链接索引)
- [§14 论文摘要速查 (按主题)](#14-论文摘要速查-按主题)
- [§15 阅读顺序推荐](#15-阅读顺序推荐)
- [§16 扩展词条（进阶篇）](#16-扩展词条进阶篇)
- [§17 SID 相关算法库](#17-sid-相关算法库)
- [§18 SID 工业案例补充](#18-sid-工业案例补充)
- [§19 SID 与其他范式对比](#19-sid-与其他范式对比)
- [§20 SID 教程推荐路径总览](#20-sid-教程推荐路径总览)
- [§21 SID 资源导航](#21-sid-资源导航)
- [§22 SID 工业指标基准 (Industry Benchmarks)](#22-sid-工业指标基准-industry-benchmarks)
- [§23 SID 与具体任务的接口](#23-sid-与具体任务的接口)
- [§24 SID 标准接口约定](#24-sid-标准接口约定)
- [§25 SID 速查表 (Cheat Sheet)](#25-sid-速查表-cheat-sheet)
- [§26 补充论文清单（按年份完整版）](#26-补充论文清单按年份完整版)
- [§27 补充术语（按场景）](#27-补充术语按场景)
- [§28 SID 相关数据集](#28-sid-相关数据集)
- [§29 SID 教程配套资源](#29-sid-教程配套资源)
- [§30 SID 速查图（ASCII）](#30-sid-速查图ascii)
- [§31 SID 学习资源 (GitHub Stars ≥ 1000)](#31-sid-学习资源-github-stars-1000)
- [§32 SID 工业话术](#32-sid-工业话术)
- [§33 附录维护与更新](#33-附录维护与更新)

---

## 1. 术语表 (A-Z, 100+ 词条)

### A

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **Alignment** | | 对齐: SID 与内容 embedding 在同一空间 |
| **AR (Auto-Regressive)** | | 自回归: 一个 token 一个 token 生成 |
| **ANN** | Approximate Nearest Neighbor | 近似最近邻: SID 检索用 |
| **Anomaly Detection** | | 异常检测: 监控 SID 分布异常 |
| **Attention** | | 注意力: Transformer 核心机制 |
| **Audio2Vec** | | 音频 embedding 模型 |
| **A/B Test** | | A/B 测试: 评估 SID 收益 |

### B

| 术语 | 含义 |
| :--- | :--- |
| **Behavior Code** | 行为码: 反映协同信号的 SID |
| **BOS** | Beginning of Sequence |
| **Backbone** | 主干网络（如 Transformer） |
| **Beam Search** | 束搜索: 生成式检索解码 |
| **BERT** | Bidirectional Encoder Representations from Transformers |
| **BPE** | Byte Pair Encoding: 子词编码 |

### C

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **CapsID** | Capsule SID | 胶囊化 SID (2026 前沿) |
| **CF** | Collaborative Filtering | 协同过滤 |
| **CLIP** | | OpenAI 的图文对齐模型 |
| **Cold Start** | | 冷启动: 新物品没行为 |
| **Collision Rate** | | 碰撞率: 多 item 映射到同一 SID 的比例 |
| **Coverage** | | 覆盖率: 实际用到的码 / 总码本 |
| **Codebook** | | 码本: SID 的"字典" |
| **Codebook Collapse** | | 码本坍塌: 大量码没人用 |
| **Commit Loss** | | 承诺损失: 让 encoder 输出接近码本 |
| **Cold-start** | | 冷启动问题 |
| **Cosine Similarity** | | 余弦相似度 |
| **Cross-modal** | | 跨模态 (文本+图像+...) |
| **CTCVR** | | Click-Through Conversion Rate |
| **CTR** | Click-Through Rate | 点击率 |

### D

| 术语 | 含义 |
| :--- | :--- |
| **Decoder** | 解码器: 从码本重建原始向量 |
| **DLRM** | Deep Learning Recommendation Model (Meta) |
| **Dead Code** | 死码: 长期不被使用的码 |
| **DIN** | Deep Interest Network (阿里) |
| **DIEN** | Deep Interest Evolution Network (阿里) |
| **Distillation** | 蒸馏: 大模型指导小模型 |
| **Domain Adaptation** | 域适配 |
| **DWA** | Dynamic Weight Average: 多任务权重 |

### E

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **EMA** | Exponential Moving Average | 指数移动平均: 码本更新方式 |
| **Embedding** | | 嵌入: 把 ID 映射到稠密向量 |
| **Encoder** | | 编码器: 把物品转为向量 |
| **EOS** | End of Sequence | 序列结束符 |
| **Expert** | | 专家网络 (MoE) |
| **Embedding Collapse** | | Embedding 退化 |

### F

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **FAISS** | Facebook AI Similarity Search | 工业级 ANN 库 |
| **FSQ** | Finite Scalar Quantization | 无码本量化 (2023) |
| **Flat SID** | | 扁平化 SID (单层) |
| **FAISS GPU** | | FAISS GPU 版本 |
| **FLOPs** | Floating Point Operations | 浮点运算数 |
| **Forward Pass** | | 前向传播 |
| **FLOPS** | | 同 FLOPs |

### G

| 术语 | 含义 |
| :--- | :--- |
| **GR** | Generative Recommendation, 生成式推荐 |
| **Generatable SID** | 可生成的 SID |
| **Gate (门控)** | 软路由专家 (MMoE) |
| **GAUC** | Grouped AUC: 按用户分组的 AUC |
| **Gradient Conflict** | 梯度冲突: 多任务跷跷板 |
| **Greedy Decoding** | 贪心解码 |
| **GPT** | Generative Pre-trained Transformer |

### H

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **Hierarchical SID** | | 层次化 SID |
| **Hybrid SID** | | 混合式 SID |
| **HSTU** | Hierarchical Sequential Transduction Unit | Meta 2024 推荐基座 |
| **HY-Token** | Hybrid Tokenization | ByteDance 双层 SID |
| **Hard Negative** | | 难负样本 |
| **Hash** | | 哈希 |
| **Hierarchical K-Means** | | 层次 K-Means |
| **Hybrid Loss** | | 混合损失 |

### I

| 术语 | 含义 |
| :--- | :--- |
| **Index** | 索引: FAISS 中指向量检索索引 |
| **Inference** | 推理 |
| **In-batch Negative** | batch 内负采样 |
| **Inner Product** | 内积 (向量相似度) |
| **Item Embedding** | 物品向量 |
| **Item2SID** | item ID → SID 映射表 |
| **Iterative Refinement** | 迭代精炼 (CapsID) |

### J

| 术语 | 含义 |
| :--- | :--- |
| **Joint Embedding** | 联合 embedding |
| **JSONL** | JSON Lines: 每行一个 JSON |
| **Jaccard Similarity** | Jaccard 相似度 |

### K

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **K-Means** | | K 均值聚类 |
| **KL Divergence** | Kullback-Leibler Divergence | KL 散度 |
| **Knowledge Distillation** | | 知识蒸馏 |
| **KV Cache** | | Transformer 推理缓存 |

### L

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **LFQ** | Lookup-Free Quantization | 查表免费量化 (2024) |
| **LLM** | Large Language Model | 大语言模型 |
| **Latent** | | 隐变量 |
| **LLM4Rec** | LLM for Recommendation | LLM 用于推荐 |
| **Lookup Table** | | 查找表 (Embedding) |
| **L2 Normalize** | | L2 归一化 |
| **L2 Distance** | | L2 距离 (欧氏) |
| **Logits** | | 未归一化分数 |
| **Latency** | | 时延 |
| **LayerNorm** | | 层归一化 |

### M

| 术语 | 含义 |
| :--- | :--- |
| **MMoE** | Multi-gate Mixture of Experts |
| **MSE** | Mean Squared Error, 重建损失 |
| **MoE** | Mixture of Experts |
| **Multi-task** | 多任务学习 |
| **Memorization** | 记忆 (vs 泛化) |
| **MIPS** | Maximum Inner Product Search |
| **MiniBatch K-Means** | 小批量 K-Means |
| **MLP** | Multi-Layer Perceptron |
| **Multi-modal** | 多模态 |
| **MQ** | Message Queue: 消息队列 |

### N

| 术语 | 含义 |
| :--- | :--- |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **Negative Sampling** | 负采样 |
| **Neural Collaborative Filtering** | 神经协同过滤 |
| **NLL** | Negative Log-Likelihood |
| **NaN** | Not a Number |
| **Normalization** | 归一化 |
| **Numpy** | Python 数值计算库 |

### O

| 术语 | 含义 |
| :--- | :--- |
| **OneRec** | 阿里 GR 模型 |
| **ONNX** | Open Neural Network Exchange |
| **OOD** | Out-of-Distribution |
| **Online Learning** | 在线学习 |
| **One-Hot** | 独热编码 |
| **Offline** | 离线 |
| **Online** | 在线 |
| **Optimizer** | 优化器 (Adam/AdamW/SGD) |

### P

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **PLUM** | | YouTube 2025 SID 论文 |
| **PMI** | Pointwise Mutual Information | 点互信息 |
| **PLE** | Progressive Layered Extraction | 多任务架构 |
| **PyTorch** | | 深度学习框架 |
| **Precision** | | 精确率 |
| **Prompt** | | 提示 (LLM) |
| **PCA** | Principal Component Analysis | 主成分分析 |
| **Parquet** | | 列存储格式 |
| **P99** | | 第 99 百分位时延 |
| **Pre-training** | | 预训练 |
| **Pip Install** | | pip 安装 |

### Q

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **QuaSID** | Query-aware SID | Query 感知量化 (2025) |
| **Quantization** | | 量化: 连续 → 离散 |
| **QPS** | Queries Per Second | 每秒查询数 |
| **Q-Value** | | Q 值 (RL) |
| **Quantile** | | 分位数 |

### R

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **R3-VAE** | Residual-Residual-Residual VAE | 2026 前沿 |
| **Recall@K** | | Top-K 召回率 |
| **ReSID** | Recurrent SID | 2026 前沿 |
| **Residual** | | 残差 |
| **RL** | Reinforcement Learning | 强化学习 |
| **RQ-Kmeans** | Residual-Quantized K-Means | 不训 encoder 的 RQ-VAE |
| **RQ-VAE** | Residual-Quantized VAE | 层次化 SID 的核心 |
| **RNN** | Recurrent Neural Network | 循环神经网络 |
| **RAG** | Retrieval-Augmented Generation | 检索增强生成 |
| **Recall** | | 召回率 |
| **ReLU** | | 修正线性单元 |
| **RNN-T** | | RNN Transducer |
| **Rerank** | | 重排 |
| **Routing by Agreement** | | 胶囊路由机制 |

### S

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **Semantic ID (SID)** | | 语义 ID |
| **Semantic Inheritance** | | 语义继承 |
| **STE** | Straight-Through Estimator | VQ 梯度 trick |
| **Streaming SID** | | 流式 SID 推断 |
| **SentencePiece** | | 文本子词编码库 |
| **SGD** | Stochastic Gradient Descent | 随机梯度下降 |
| **Softmax** | | 归一化指数函数 |
| **Spherical K-Means** | | 球面 K-Means (cosine) |
| **Sample Selection Bias** | | 样本选择偏差 (ESMM) |
| **Self-Attention** | | 自注意力 |
| **Surrogate Loss** | | 代理损失 |
| **Sampled Softmax** | | 采样 softmax |
| **Sparse** | | 稀疏 |

### T

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **TIGER** | | Meta 2023 GR 模型 |
| **TRM** | Token Recommendation Model | ByteDance 2026 GR |
| **Tokenizer** | | 把物品 → SID 的模型 |
| **TF-IDF** | | 词频-逆文档频率 |
| **Top-K** | | Top-K 检索 |
| **Throughput** | | 吞吐量 |
| **Trainable** | | 可训练的 |
| **Transformer** | | Transformer 架构 |
| **Temperature** | | 温度 (softmax) |
| **TPU** | Tensor Processing Unit | Google 加速器 |

### U

| 术语 | 含义 |
| :--- | :--- |
| **Uniformity** | 均匀性: SID 在空间均匀分布 |
| **User Embedding** | 用户向量 |
| **Usage Rate** | 使用率 |
| **Uncertainty Weighting** | 不确定性加权 |

### V

| 术语 | 全称 | 含义 |
| :--- | :--- | :--- |
| **VQ** | Vector Quantization | 向量量化 |
| **VQ-VAE** | VQ Variational AutoEncoder | 单层量化的开山之作 |
| **Variational Inference** | | 变分推断 |
| **Vanilla** | | 原始版 (无改进) |
| **Variance** | | 方差 |
| **Visualization** | | 可视化 |
| **Validation Set** | | 验证集 |

### W

| 术语 | 含义 |
| :--- | :--- |
| **Warmup** | 训练预热 |
| **Weight Decay** | 权重衰减 |
| **WandB** | Weights & Biases: 训练监控 |

### X

| 术语 | 含义 |
| :--- | :--- |
| **Xavier Init** | Xavier 初始化 |
| **XLNet** | 排列 LM |

### Y

| 术语 | 含义 |
| :--- | :--- |
| **YAML** | YAML 配置格式 |
| **Yarn** | 包管理 |

### Z

| 术语 | 含义 |
| :--- | :--- |
| **Zero-shot** | 零样本 |
| **ZeRO** | Zero Redundancy Optimizer (DeepSpeed) |

---

## 2. 论文清单 (按时间线, 2017-2026)

### 2.1 2017 - 经典期

#### VQ-VAE (van den Oord et al., NeurIPS 2017)
- **链接**: https://arxiv.org/abs/1711.00937
- **作者**: van den Oord, Vinyals, Kavukcuoglu (DeepMind)
- **核心**: 单层 VQ 用于图像/音频生成
- **公式**: 三项损失 (recon + codebook + commit)
- **影响**: VQ-VAE 是所有后续 SID 方法的基石

### 2.2 2018 - 扩展期

#### VQ-VAE-2 (Razavi et al., 2018)
- **链接**: https://arxiv.org/abs/1906.00446
- **作者**: Razavi, van den Oord, Vinyals (DeepMind)
- **核心**: 多尺度 VQ (粗 + 细)
- **改进**: 解决高分辨率图像重建

### 2.3 2022 - 应用期

#### SoundStream (Google, 2022)
- **链接**: https://arxiv.org/abs/2107.03312
- **作者**: Zeghidour et al. (Google)
- **核心**: 音频流式编码 + RQ
- **意义**: RQ 在音频上的成功应用

#### DALL-E 2 (OpenAI, 2022)
- **链接**: https://arxiv.org/abs/2204.06125
- **作者**: Ramesh et al. (OpenAI)
- **核心**: CLIP + Diffusion + VQ
- **意义**: 图像离散化生成

### 2.4 2023 - 推荐萌芽期

#### TIGER (Meta, 2023) ★★★
- **链接**: https://arxiv.org/abs/2305.05065
- **作者**: Rajput et al. (Meta)
- **核心**: **第一个用 RQ-VAE 做推荐**
- **公式**: $\text{sid} = (c_1, ..., c_L)$, Transformer decoder 生成
- **结果**: Recall@10 +18% vs raw ID
- **影响**: SID 在推荐领域的开山之作

#### FSQ (Mentzer et al., CVPR 2023)
- **链接**: https://arxiv.org/abs/2309.15505
- **作者**: Mentzer et al. (Google)
- **核心**: 无码本量化
- **改进**: 每维 round 到离散点
- **优势**: 无需 K-Means 初始化

### 2.5 2024 - 工业爆发期

#### HSTU (Meta, SIGIR 2024) ★★★
- **链接**: https://arxiv.org/abs/2402.17152
- **作者**: Zhai et al. (Meta)
- **核心**: Hierarchical Sequential Transduction Unit
- **公式**: 改进的 attention 用于推荐
- **结果**: 在线 AUC +12%, 训练速度 15x
- **影响**: 工业级 GR 标杆

#### LFQ (Yu et al., 2024)
- **链接**: https://arxiv.org/abs/2310.16552
- **作者**: Yu et al.
- **核心**: 查表免费量化 (bit packing)
- **改进**: 归一化 + 直通梯度
- **应用**: Stable Diffusion 3

#### OneRec (阿里, KDD 2024) ★★★
- **作者**: Alibaba Group
- **核心**: 端到端 GR (RQ-VAE + Transformer 生成)
- **配置**: L=6, K=1024
- **结果**: 电商场景 Recall@K 显著提升

### 2.6 2025 - 扩展期

#### YouTube PLUM (2025) ★★
- **链接**: https://arxiv.org/abs/2510.07784
- **作者**: YouTube Team
- **核心**: RQ-Kmeans + 协同对比损失
- **配置**: L=4, K=256
- **结果**: Recall@100 +18%, 冷启动 +45%

#### QuaSID (2025)
- **核心**: Query-aware SID
- **改进**: SID 随 query 变化
- **应用**: 召回侧多样性

#### Spotify Production SIDs (2026)
- **链接**: https://arxiv.org/abs/2606.25496
- **作者**: Spotify Team
- **核心**: 大规模 RQ-Kmeans + 语义哈希
- **配置**: L=5, K=256, FAISS GPU
- **意义**: 大规模音乐/播客 SID 工业实践

### 2.7 2026 - 前沿期

#### R3-VAE (2026)
- **核心**: 跳层连接 + 多层监督
- **解决**: RQ-VAE 误差累积

#### CapsID (2026)
- **核心**: 胶囊化码本
- **解决**: 码本坍塌

#### ReSID (2026)
- **核心**: 循环式生成 SID
- **改进**: 码字间依赖

#### ByteDance TRM / HY-Token (2026) ★★★
- **链接**: https://arxiv.org/abs/2601.22694
- **作者**: ByteDance
- **核心**: 双层 SID (粗层=类目, 细层=RQ)
- **意义**: 字节抖音工业级方案

#### Snapchat SID (2026)
- **作者**: Snap Inc.
- **核心**: 短视频 SID
- **配置**: 类似 YouTube PLUM

---

## 3. 论文清单 (按公司)

### 3.1 Meta
- **HSTU** (SIGIR 2024) - https://arxiv.org/abs/2402.17152
- **TIGER** (2023) - https://arxiv.org/abs/2305.05065
- **DLRM** (2019) - https://arxiv.org/abs/1904.05114

### 3.2 字节跳动 (ByteDance)
- **TRM / HY-Token** (arXiv 2601.22694, 2026)
- **Monolith** (2022) - 字节碰撞率优化

### 3.3 阿里巴巴
- **OneRec** (KDD 2024) - 端到端 GR
- **DIN** (KDD 2018) - 深度兴趣网络
- **DIEN** (AAAI 2019) - 兴趣进化
- **ESMM** (SIGIR 2018) - 全空间多任务

### 3.4 Google
- **SoundStream** (2022) - 音频量化
- **GEST** (2024) - 生成式推荐
- **FSQ** (CVPR 2023) - 无码本量化

### 3.5 Microsoft
- **Vision Transformer** (ICLR 2021)
- **MAE** (CVPR 2022)

### 3.6 Spotify
- **Production SIDs** (arXiv 2606.25496, 2026)

### 3.7 YouTube
- **PLUM** (arXiv 2510.07784, 2025)
- **DNN for YouTube Recommendations** (RecSys 2016)

### 3.8 Snapchat
- **Snapchat SID** (2026)

### 3.9 学术机构
- **清华**: CogView, GLIDE
- **北大**: ERNIE-ViL
- **CMU**: Various
- **Stanford**: Various

---

## 4. 论文清单 (按方法)

### 4.1 VQ-VAE 系

| 论文 | 年份 | 核心贡献 |
| :--- | :--- | :--- |
| VQ-VAE | 2017 | 单层量化开山 |
| VQ-VAE-2 | 2018 | 多尺度 |
| RQ-VAE | 2023 | 推荐应用 |
| R3-VAE | 2026 | 多层跳层 |

### 4.2 量化新方法

| 论文 | 年份 | 核心贡献 |
| :--- | :--- | :--- |
| FSQ | 2023 | 无码本量化 |
| LFQ | 2024 | 查表免费 |

### 4.3 SID 改进

| 论文 | 年份 | 核心贡献 |
| :--- | :--- | :--- |
| QuaSID | 2025 | Query-aware |
| CapsID | 2026 | 胶囊化 |
| ReSID | 2026 | 循环式 |
| HY-Token | 2026 | 双层 SID |

### 4.4 工业应用

| 论文 | 公司 | 年份 |
| :--- | :--- | :--- |
| HSTU | Meta | 2024 |
| OneRec | 阿里 | 2024 |
| PLUM | YouTube | 2025 |
| Production SIDs | Spotify | 2026 |
| TRM | 字节 | 2026 |

### 4.5 推荐系统基础

| 论文 | 年份 | 核心贡献 |
| :--- | :--- | :--- |
| DLRM | 2019 | Meta 推荐基座 |
| DIN | 2018 | 阿里深度兴趣 |
| ESMM | 2018 | 全空间多任务 |
| MMoE | 2018 | 多任务架构 |
| PLE | 2020 | 多任务进阶 |

---

## 5. 论文清单 (按场景)

### 5.1 电商

- **OneRec** (阿里 2024)
- **DIN/DIEN** (阿里 2018-2019)
- **ESMM** (阿里 2018)

### 5.2 短视频

- **PLUM** (YouTube 2025)
- **TRM** (字节 2026)
- **Snapchat SID** (Snap 2026)

### 5.3 音乐/播客

- **Production SIDs** (Spotify 2026)
- **SoundStream** (Google 2022)

### 5.4 新闻/内容

- **GEST** (Google 2024)
- **DIN** 变体 (多家)

### 5.5 通用推荐

- **HSTU** (Meta 2024)
- **TIGER** (Meta 2023)
- **DLRM** (Meta 2019)

---

## 6. 速查公式 (50+ 公式)

### 6.1 RQ-VAE 量化

$$
\mathbf{e}_i = \sum_{k=1}^{K} \mathbf{B}[c_k] + \mathbf{r}_K
$$

其中 $\mathbf{r}_K$ 是最终残差。

### 6.2 RQ-VAE 损失

$$
\mathcal{L} = \underbrace{\|\mathbf{e} - \hat{\mathbf{e}}\|_2^2}_{\text{重建}} + \underbrace{\|\text{sg}[\mathbf{e}] - \hat{\mathbf{e}}\|_2^2}_{\text{codebook}} + \underbrace{\beta \|\mathbf{e} - \text{sg}[\hat{\mathbf{e}}]\|_2^2}_{\text{commit}}
$$

### 6.3 总空间

$$
\text{总空间} = C^K
$$

### 6.4 存储节省

$$
\text{存储比} = \frac{K \cdot C \cdot D}{N \cdot D} = \frac{K \cdot C}{N}
$$

### 6.5 EMA 更新

$$
N_k^{(t)} = \gamma N_k^{(t-1)} + (1-\gamma) \sum_{i: c_i=k} 1
$$

$$
m_k^{(t)} = \gamma m_k^{(t-1)} + (1-\gamma) \sum_{i: c_i=k} \mathbf{r}_i
$$

$$
\mathbf{B}^{(t)}[k] = \frac{m_k^{(t)}}{N_k^{(t)} + \epsilon}
$$

### 6.6 FSQ 量化

$$
\mathbf{z}_q[i] = \text{round}\left(\frac{\mathbf{z}_e[i] \cdot L_i/2}{L_i/2}\right)
$$

### 6.7 TIGER 生成

$$
P(\mathbf{s}_t | \mathbf{s}_{<t}, \mathbf{c}) = \text{Transformer}(\mathbf{s}_{<t}, \mathbf{c})
$$

### 6.8 HSTU Attention

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{M}\right) \mathbf{V}
$$

其中 $\mathbf{M}$ 是相对位置 bias。

### 6.9 多任务加权损失

$$
\mathcal{L} = \sum_{t=1}^{T} w_t \, \ell_t(\hat{y}_t, y_t)
$$

### 6.10 不确定性加权

$$
\mathcal{L} = \sum_{t=1}^{T} \frac{1}{2\sigma_t^2}\,\ell_t + \log \sigma_t
$$

### 6.11 PCGrad 投影

$$
g_i \leftarrow g_i - \frac{\langle g_i, g_j\rangle}{\|g_j\|^2}\, g_j \quad(\text{当} \langle g_i,g_j\rangle<0)
$$

### 6.12 ESMM 关系

$$
p(\text{click\&conv} \mid x) = p(\text{click} \mid x) \times p(\text{conv} \mid \text{click}, x)
$$

### 6.13 GAUC

$$
\text{GAUC} = \frac{\sum_{u} w_u \cdot \text{AUC}_u}{\sum_u w_u}, \quad w_u = \#\text{impressions}_u
$$

### 6.14 Embedding 聚合

$$
\mathbf{e}_i = \sum_{l=1}^{L} \mathbf{e}_l[c_l]
$$

### 6.15 Recall@K

$$
\text{Recall@K} = \frac{|\text{TopK 命中}|}{|\text{真实相关}|}
$$

### 6.16 NDCG@K

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

### 6.17 协同对比损失

$$
\mathcal{L}_{\text{collab}} = -\log \frac{\exp(s(\mathbf{i}, \mathbf{i}^+) / \tau)}{\sum_j \exp(s(\mathbf{i}, \mathbf{j}) / \tau)}
$$

### 6.18 序列 RQ 残差

$$
\mathbf{r}^{(l)} = \mathbf{r}^{(l-1)} - \mathbf{B}^{(l)}[c_l(\mathbf{r}^{(l-1)}, \mathbf{q})]
$$

其中 $\mathbf{q}$ 是 query。

### 6.19 胶囊 squashing

$$
\mathbf{v} = \frac{\|\mathbf{s}\|^2}{1 + \|\mathbf{s}\|^2} \frac{\mathbf{s}}{\|\mathbf{s}\|}
$$

### 6.20 FSQ 总码数

$$
N_{\text{FSQ}} = \prod_{i} L_i
$$

### 6.21 LFQ 总码数

$$
N_{\text{LFQ}} = 2^D
$$

### 6.22 GAUC per task

$$
\text{GAUC}_t = \frac{\sum_{u} w_u^{(t)} \cdot \text{AUC}_u^{(t)}}{\sum_u w_u^{(t)}}
$$

### 6.23 HY-Token SID

$$
\mathbf{sid} = (c_{\text{coarse}}, c_1, c_2, \dots, c_L)
$$

### 6.24 Cosine 相似度

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

### 6.25 L2 距离

$$
d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_i (a_i - b_i)^2}
$$

### 6.26 STE 量化

$$
\mathbf{z}_q = \mathbf{z}_e + (\mathbf{z}_q - \mathbf{z}_e).detach()
$$

### 6.27 Layer 残差方差衰减

$$
\text{Var}(\mathbf{r}^{(l)}) < \text{Var}(\mathbf{r}^{(l-1)})
$$

### 6.28 Codebook 熵

$$
H(\mathbf{c}) = -\sum_k p_k \log p_k \leq \log K
$$

### 6.29 碰撞率

$$
\text{Collision Rate} = 1 - \frac{|\text{Unique SID}|}{|\text{Total Items}|}
$$

### 6.30 SID 重建误差

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2
$$

### 6.31 死码重启

$$
\mathbf{B}[k_{\text{dead}}] = \mathbf{r}_{i^*}, \quad i^* = \arg\max_i \|\mathbf{r}_i - \mathbf{B}[\text{nearest}(i)]\|
$$

### 6.32 K-Means 目标

$$
\min_{\{\mathbf{c}_k\}} \sum_{k=1}^{K} \sum_{i \in S_k} \|\mathbf{x}_i - \mathbf{c}_k\|_2^2
$$

### 6.33 RQ-Kmeans 残差

$$
\mathbf{r}^{(l)} = \mathbf{r}^{(l-1)} - \mathbf{B}^{(l)}[c_l]
$$

### 6.34 QPS

$$
\text{QPS} = \frac{\text{总请求数}}{\text{时间}}
$$

### 6.35 P99 时延

$$
P_{99} = \text{第 99 百分位的响应时间}
$$

### 6.36 K-Means 复杂度

$$
T(N, K, D, T_{\text{iter}}) = O(N K D T_{\text{iter}})
$$

### 6.37 RQ-VAE 容量

$$
N_{\max} = C^L
$$

### 6.38 RQ-VAE 利用率

$$
U_l = \frac{|\{c : \text{used in layer } l\}|}{|\mathbf{B}^{(l)}|}
$$

### 6.39 用户塔/候选塔

$$
\text{Score}(u, i) = f_{\text{tower}}\left(\text{SID}_{\text{hist}} \oplus \text{SID}_i\right)
$$

### 6.40 Cold-start SID

$$
\mathbf{sid}_{\text{new}} = \text{RQ-Kmeans}(\text{BERT}(\text{title}))
$$

### 6.41 多模态 SID

$$
\mathbf{sid}_{\text{multi}} = \text{Concat}(\mathbf{sid}_{\text{text}}, \mathbf{sid}_{\text{img}}, \mathbf{sid}_{\text{cat}})
$$

### 6.42 蒸馏损失

$$
\mathcal{L}_{\text{distill}} = \text{KL}(\mathbf{p}_{\text{teacher}} \| \mathbf{p}_{\text{student}})
$$

### 6.43 RL Reward

$$
R = \alpha \cdot \text{click} + \beta \cdot \text{conversion} + \gamma \cdot \text{retention}
$$

### 6.44 Co-training 对比损失

$$
\mathcal{L}_{\text{collab}} = -\sum_{(i,j) \in \text{co-view}} \log \frac{\exp(s(\mathbf{e}_i, \mathbf{e}_j) / \tau)}{\sum_k \exp(s(\mathbf{e}_i, \mathbf{e}_k) / \tau)}
$$

### 6.45 注意力分数

$$
\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d}}
$$

### 6.46 Softmax

$$
\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

### 6.47 温度

$$
\text{softmax}(\mathbf{x}/T)_i = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}
$$

### 6.48 L2 归一化

$$
\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}
$$

### 6.49 MiniBatch K-Means 目标

$$
\min \sum_{k} \sum_{i \in B_t \cap S_k} \|\mathbf{x}_i - \mathbf{c}_k\|_2^2
$$

### 6.50 Welford 在线更新

$$
\mathbf{c}_k^{(t+1)} = \mathbf{c}_k^{(t)} + \frac{1}{n_k + 1}(\mathbf{x}_{\text{new}} - \mathbf{c}_k^{(t)})
$$

---

## 7. 关键引文 (Top 20 必读)

### 必读清单

| # | 论文 | 年份 | 重要度 | 原因 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **VQ-VAE** | 2017 | ⭐⭐⭐⭐⭐ | SID 理论基石 |
| 2 | **TIGER** | 2023 | ⭐⭐⭐⭐⭐ | SID 在推荐的开山 |
| 3 | **HSTU** | 2024 | ⭐⭐⭐⭐⭐ | Meta 工业标杆 |
| 4 | **OneRec** | 2024 | ⭐⭐⭐⭐⭐ | 阿里端到端 GR |
| 5 | **PLUM** | 2025 | ⭐⭐⭐⭐ | YouTube 短视频 |
| 6 | **Production SIDs** | 2026 | ⭐⭐⭐⭐ | Spotify 工业 |
| 7 | **TRM / HY-Token** | 2026 | ⭐⭐⭐⭐ | 字节双层 SID |
| 8 | **FSQ** | 2023 | ⭐⭐⭐⭐ | 无码本量化 |
| 9 | **LFQ** | 2024 | ⭐⭐⭐ | 查表免费 |
| 10 | **DIN** | 2018 | ⭐⭐⭐ | 阿里深度兴趣 |
| 11 | **ESMM** | 2018 | ⭐⭐⭐⭐ | CVR 全空间 |
| 12 | **MMoE** | 2018 | ⭐⭐⭐⭐ | 多任务架构 |
| 13 | **PLE** | 2020 | ⭐⭐⭐⭐ | 多任务进阶 |
| 14 | **PCGrad** | 2020 | ⭐⭐⭐ | 梯度冲突 |
| 15 | **DLRM** | 2019 | ⭐⭐⭐⭐ | Meta 推荐基座 |
| 16 | **SoundStream** | 2022 | ⭐⭐⭐ | 音频 RQ |
| 17 | **Capsule Network** | 2017 | ⭐⭐⭐ | 胶囊思想 |
| 18 | **FAISS** | 2017 | ⭐⭐⭐⭐ | 向量检索 |
| 19 | **Sentence-BERT** | 2019 | ⭐⭐⭐ | 文本 embedding |
| 20 | **CLIP** | 2021 | ⭐⭐⭐⭐ | 图文对齐 |

### 阅读建议

- **优先级 1**（必读）：#1-7
- **优先级 2**（强烈推荐）：#8-15
- **优先级 3**（选读）：#16-20

---

## 8. 工业里程碑

### 8.1 时间线

```text
2017 - VQ-VAE (DeepMind) - 理论奠基
       ↓
2018 - DIN, ESMM, MMoE (阿里) - 多任务架构
       ↓
2019 - DLRM (Meta) - 推荐基座
       ↓
2022 - SoundStream (Google) - 音频 RQ
       ↓
2023 - TIGER (Meta) ★ SID 应用开山
       ↓
2024 - HSTU (Meta) ★ 工业标杆
       ↓
2024 - OneRec (阿里) ★ 端到端 GR
       ↓
2024 - LFQ, FSQ (Google) - 量化新方法
       ↓
2025 - PLUM (YouTube) ★ 短视频 SID
       ↓
2025 - QuaSID - Query-aware
       ↓
2026 - R3-VAE, CapsID, ReSID - 改进方案
       ↓
2026 - HY-Token (字节) ★ 双层 SID
       ↓
2026 - Production SIDs (Spotify) ★ 工业实践
```

### 8.2 公司里程碑

#### Meta
- 2019: DLRM
- 2023: TIGER (SID 开山)
- 2024: HSTU (工业标杆)

#### 阿里
- 2018: DIN, ESMM, MMoE
- 2019: DIEN
- 2024: OneRec (端到端 GR)

#### 字节
- 2022: Monolith (碰撞率优化)
- 2026: TRM / HY-Token

#### YouTube
- 2016: DNN for YouTube Recommendations
- 2025: PLUM (SID)

#### Spotify
- 2026: Production SIDs

---

## 9. 复现资源索引

### 9.1 官方仓库

- **TIGER**: https://github.com/google-deepmind/tiger (待确认)
- **HSTU**: https://github.com/facebookresearch/HSTU
- **FAISS**: https://github.com/facebookresearch/faiss
- **FSQ**: https://github.com/google-research/google-research/tree/master/finite_scalar_quantization

### 9.2 第三方复现

- **TIGER 复现**: 多见于知乎/CSDN
- **OneRec 复现**: 阿里官方技术博客
- **RQ-VAE 复现**: 见本教程附录

### 9.3 本仓库

- `第03章-附录-RQ-VAE完整PyTorch实现.md` - 完整实现
- `第03章-04-RQ-Kmeans-工业版简化.md` - RQ-Kmeans
- `../../sid_implementation/` - 配套代码

---

## 10. 速查图示

### 10.1 SID 编码流程

```text
物品 (item)
  │
  ├─→ 文本 (title) → BERT → text_emb [768]
  ├─→ 图像 (cover) → CLIP → image_emb [512]
  └─→ 类目 (category) → Embedding → cat_emb [64]
                  ↓
                Concat [1344]
                  ↓
            RQ-VAE / RQ-Kmeans
                  ↓
           SID = (c1, c2, c3, c4, c5)
                  ↓
              item2sid 表
                  ↓
             下游任务 / 检索
```

### 10.2 RQ-VAE 残差量化

```text
z_e (latent)
  │
  ├─ Layer 1: 找最近码 c1, B^(1)[c1], r1 = z_e - B^(1)[c1]
  │              │
  │              ├─ Layer 2: 找最近码 c2, B^(2)[c2], r2 = r1 - B^(2)[c2]
  │              │              │
  │              │              ├─ Layer 3: ...
  │              │              │     │
  │              │              │     └─ Layer L: 残差可忽略
  │              │              │
  最终: z_q = B^(1)[c1] + B^(2)[c2] + ... + B^(L)[cL]
  SID = (c1, c2, ..., cL)
```

### 10.3 多任务 SID 集成

```text
User History → SID emb
  ↓
Candidate → SID emb
  ↓
Concat
  ↓
MMoE / PLE
  ↓
┌──┬──┬──┐
CTR CVR 时长
```

---

## 11. 中英术语对照 (扩展版)

| 中文 | 英文 | 缩写 |
| :--- | :--- | :--- |
| 语义 ID | Semantic ID | SID |
| 残差量化 | Residual Quantization | RQ |
| 残差量化变分自编码器 | Residual-Quantized VAE | RQ-VAE |
| 残差量化 K 均值 | Residual-Quantized K-Means | RQ-Kmeans |
| 向量量化 | Vector Quantization | VQ |
| 向量量化变分自编码器 | VQ Variational AutoEncoder | VQ-VAE |
| 直通估计器 | Straight-Through Estimator | STE |
| 指数移动平均 | Exponential Moving Average | EMA |
| 承诺损失 | | Commitment Loss |
| 死码 | | Dead Code |
| 码本坍塌 | | Codebook Collapse |
| 碰撞率 | | Collision Rate |
| 码本利用率 | | Codebook Utilization |
| 重建误差 | | Reconstruction Error |
| 跳层连接 | | Skip Connection |
| 胶囊网络 | | Capsule Network |
| 有限标量量化 | Finite Scalar Quantization | FSQ |
| 查表免费量化 | Lookup-Free Quantization | LFQ |
| 双层令牌化 | Hybrid Tokenization | HY-Token |
| 查询感知 | | Query-Aware |
| 循环式 | | Recurrent |
| 多任务学习 | Multi-Task Learning | MTL |
| 多门控混合专家 | Multi-gate Mixture of Experts | MMoE |
| 渐进分层提取 | Progressive Layered Extraction | PLE |
| 全空间多任务模型 | Entire Space Multi-task Model | ESMM |
| 样本选择偏差 | Sample Selection Bias | SSB |
| 分组 AUC | Grouped AUC | GAUC |
| 梯度冲突 | | Gradient Conflict |
| 负迁移 | | Negative Transfer |
| 跷跷板现象 | | Seesaw Phenomenon |
| 投影冲突梯度 | Projecting Conflicting Gradients | PCGrad |
| 动态权重平均 | Dynamic Weight Average | DWA |
| 深度兴趣网络 | Deep Interest Network | DIN |
| 深度兴趣进化网络 | Deep Interest Evolution Network | DIEN |
| 深度学习推荐模型 | Deep Learning Recommendation Model | DLRM |
| 生成式推荐 | Generative Recommendation | GR |
| 序列 transduction 单元 | Hierarchical Sequential Transduction Unit | HSTU |
| 基于 transformer 的生成式编码器 | | TIGER |
| 端到端推荐 | | OneRec |
| 短视频 SID | | PLUM |
| 字节推荐模型 | Token Recommendation Model | TRM |
| 协同过滤 | Collaborative Filtering | CF |
| 冷启动 | | Cold Start |
| 知识蒸馏 | Knowledge Distillation | KD |
| 强化学习 | Reinforcement Learning | RL |
| 检索增强生成 | Retrieval-Augmented Generation | RAG |
| 大语言模型 | Large Language Model | LLM |
| 多模态 | | Multi-modal |

---

## 12. 缩写词典

| 缩写 | 全称 | 含义 |
| :--- | :--- | :--- |
| SID | Semantic ID | 语义 ID |
| VQ | Vector Quantization | 向量量化 |
| RQ | Residual Quantization | 残差量化 |
| VAE | Variational AutoEncoder | 变分自编码器 |
| STE | Straight-Through Estimator | 直通估计器 |
| EMA | Exponential Moving Average | 指数移动平均 |
| K-Means | K-Means Clustering | K 均值聚类 |
| FSQ | Finite Scalar Quantization | 有限标量量化 |
| LFQ | Lookup-Free Quantization | 查表免费量化 |
| GR | Generative Recommendation | 生成式推荐 |
| DLRM | Deep Learning Recommendation Model | 深度学习推荐模型 |
| DIN | Deep Interest Network | 深度兴趣网络 |
| DIEN | Deep Interest Evolution Network | 深度兴趣进化网络 |
| ESMM | Entire Space Multi-task Model | 全空间多任务模型 |
| MMoE | Multi-gate Mixture of Experts | 多门控混合专家 |
| PLE | Progressive Layered Extraction | 渐进分层提取 |
| PCGrad | Projecting Conflicting Gradients | 投影冲突梯度 |
| DWA | Dynamic Weight Average | 动态权重平均 |
| GAUC | Grouped AUC | 分组 AUC |
| SSB | Sample Selection Bias | 样本选择偏差 |
| SSB | | Sample Selection Bias |
| HSTU | Hierarchical Sequential Transduction Unit | 层次序列转换单元 |
| HY-Token | Hybrid Tokenization | 混合令牌化 |
| TRM | Token Recommendation Model | 令牌推荐模型 |
| PLUM | | YouTube SID 模型 |
| FAISS | Facebook AI Similarity Search | 工业级 ANN 库 |
| ANN | Approximate Nearest Neighbor | 近似最近邻 |
| CF | Collaborative Filtering | 协同过滤 |
| KD | Knowledge Distillation | 知识蒸馏 |
| RL | Reinforcement Learning | 强化学习 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| LLM | Large Language Model | 大语言模型 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| BN | Batch Normalization | 批归一化 |
| LN | LayerNorm | 层归一化 |
| SGD | Stochastic Gradient Descent | 随机梯度下降 |
| Adam | Adaptive Moment Estimation | 自适应矩估计 |
| AdamW | Adam with Weight Decay | Adam + 权重衰减 |
| MSE | Mean Squared Error | 均方误差 |
| BCE | Binary Cross-Entropy | 二元交叉熵 |
| NCE | Noise Contrastive Estimation | 噪声对比估计 |
| InfoNCE | | 信息对比损失 |
| NDCG | Normalized Discounted Cumulative Gain | 归一化折损累积增益 |
| MRR | Mean Reciprocal Rank | 平均倒数排名 |
| MAP | Mean Average Precision | 平均精度均值 |
| AUC | Area Under Curve | 曲线下面积 |
| QPS | Queries Per Second | 每秒查询数 |
| P99 | | 第 99 百分位 |
| TF-IDF | Term Frequency-Inverse Document Frequency | 词频-逆文档频率 |
| BERT | | Bidirectional Encoder Representations from Transformers |
| CLIP | Contrastive Language-Image Pre-training | 对比图文预训练 |
| GPT | Generative Pre-trained Transformer | 生成式预训练 Transformer |
| LLM4Rec | LLM for Recommendation | LLM 用于推荐 |
| TPU | Tensor Processing Unit | 张量处理单元 (Google) |
| GPU | Graphics Processing Unit | 图形处理单元 |
| CPU | Central Processing Unit | 中央处理单元 |
| ONNX | Open Neural Network Exchange | 开放神经网络交换格式 |
| NCCL | NVIDIA Collective Communications Library | NVIDIA 集合通信库 |
| DDP | Distributed Data Parallel | 分布式数据并行 |

---

## 13. 论文下载链接索引

> **注**: 部分论文可能因版权问题无法直接下载，建议通过 arXiv / Google Scholar / 论文作者主页获取。

| 论文 | arXiv 链接 | 会议/期刊 |
| :--- | :--- | :--- |
| VQ-VAE | https://arxiv.org/abs/1711.00937 | NeurIPS 2017 |
| VQ-VAE-2 | | https://arxiv.org/abs/1906.00446 |
| SoundStream | | https://arxiv.org/abs/2107.03312 |
| DALL-E 2 | | https://arxiv.org/abs/2204.06125 |
| TIGER | | https://arxiv.org/abs/2305.05065 |
| FSQ | https://arxiv.org/abs/2309.15505 | CVPR 2023 |
| HSTU | https://arxiv.org/abs/2402.17152 | SIGIR 2024 |
| LFQ | | https://arxiv.org/abs/2310.16552 |
| PLUM | | https://arxiv.org/abs/2510.07784 |
| Production SIDs | | https://arxiv.org/abs/2606.25496 |
| TRM | | https://arxiv.org/abs/2601.22694 |
| DIN | https://arxiv.org/abs/1706.06978 | KDD 2018 |
| ESMM | https://arxiv.org/abs/1804.07931 | SIGIR 2018 |
| MMoE | https://arxiv.org/abs/1706.05098 | KDD 2018 |
| PLE | https://arxiv.org/abs/2010.04603 | RecSys 2020 |
| PCGrad | https://arxiv.org/abs/2001.06782 | NeurIPS 2020 |
| DLRM | | https://arxiv.org/abs/1904.05114 |
| Capsule Network | https://arxiv.org/abs/1710.09829 | NeurIPS 2017 |
| Sentence-BERT | https://arxiv.org/abs/1908.10084 | EMNLP 2019 |
| CLIP | https://arxiv.org/abs/2103.00020 | ICML 2021 |
| FAISS | | https://arxiv.org/abs/1702.08734 |

---

## 14. 论文摘要速查 (按主题)

### 14.1 VQ-VAE 系列

**VQ-VAE (2017)**: "我们提出了一个简单而强大的生成模型，把连续 latent 离散化为 codebook 索引。模型由 encoder、codebook、decoder 三部分组成，训练用 straight-through estimator。"

**FSQ (2023)**: "我们发现 VQ-VAE 的 codebook 可以不要——只需对每维 round 到固定离散点。FSQ 训练更稳，无坍塌问题。"

### 14.2 SID 应用

**TIGER (2023)**: "我们提出把 item ID 替换为 RQ-VAE 学到的 Semantic ID，然后用 Transformer decoder 自回归生成下一个 item 的 SID。Recall@10 比 raw ID 高 18%。"

**OneRec (2024)**: "我们提出端到端的 GR，把物品 SID 化后用 Transformer 直接生成推荐序列。在阿里电商场景，CTR/CVR 都显著提升。"

**PLUM (2025)**: "我们提出 RQ-Kmeans + 协同对比损失，让短视频 SID 既保留语义又编码协同信号。在 YouTube 数据集上 Recall@100 +18%。"

**Production SIDs (2026)**: "我们用 RQ-Kmeans 在 Spotify 训练了数千万音乐/播客的 SID，每周重新训练一次，支持冷启动和长尾。"

### 14.3 工业框架

**HSTU (2024)**: "我们提出 Hierarchical Sequential Transduction Unit，把 attention 改造为推荐专用。在 Meta 工业数据上比 Transformer 快 15x，AUC +12%。"

**TRM (2026)**: "我们提出 Token Recommendation Model，用双层 SID (粗层=类目, 细层=RQ) 实现抖音短视频推荐。"

### 14.4 多任务架构

**ESMM (2018)**: "我们提出 Entire Space Multi-task Model，在全曝光空间建模 pCTR 和 pCTCVR=pCTR×pCVR，间接学习 CVR，解决样本选择偏差。"

**MMoE (2018)**: "我们提出 Multi-gate Mixture-of-Experts，每个任务有独立的门控路由到不同专家组合。"

**PLE (2020)**: "我们提出 Progressive Layered Extraction，显式区分共享专家和任务专属专家。"

### 14.5 推荐基础

**DIN (2018)**: "我们用 target-attention 让候选物品对用户历史做注意力，捕捉局部兴趣。"

**DLRM (2019)**: "我们提出 Deep Learning Recommendation Model，把数值特征和 embedding 特征分开处理。"

---

## 15. 阅读顺序推荐

### 15.1 入门路径 (1-2 周)

```text
Week 1: VQ-VAE → FSQ → TIGER
        ↓
Week 2: HSTU → OneRec
```

### 15.2 进阶路径 (3-4 周)

```text
Week 1: 入门路径
        ↓
Week 2: DIN → ESMM → MMoE → PLE
        ↓
Week 3: PLUM → Production SIDs → TRM
        ↓
Week 4: DLRM → SoundStream → LFQ
```

### 15.3 工业实战路径 (5-6 周)

```text
Week 1-2: 入门路径
Week 3-4: 进阶路径
Week 5: PCGrad → DWA → Capsule Network
Week 6: 复现至少一个工业方案 (推荐 PLUM)
```

### 15.4 研究创新路径 (8-10 周)

```text
Week 1-4: 进阶路径
Week 5: 读 50 篇前沿论文 (arXiv alert)
Week 6-7: 设计 ablation 实验
Week 8-10: 撰写并发表论文
```

---

---

## 16. 扩展词条（进阶篇）

### 16.1 架构类

| 术语 | 全称 | 含义 | 出处 |
| :--- | :--- | :--- | :--- |
| **AMIE** | | 阿里多任务架构 | 阿里 2020 |
| **AIM** | | 阿里兴趣建模 | 阿里 2022 |
| **BST** | Behavior Sequence Transformer | 行为序列 Transformer | 阿里 2019 |
| **CAN** | | 阿里多目标 | 阿里 2021 |
| **DBMTL** | | Deep Bayesian Multi-Task Learning | 深度贝叶斯多任务 |
| **Cross Decoupling** | | 跨任务解耦  | |
| **MoE** | Mixture of Experts | 混合专家 | Shazeer 2017 |
| **Switch Transformer** | | 稀疏 MoE | Fedus 2022 |
| **GLaM** | | 通用稀疏 LM | Google 2021 |
| **GShard** | | 稀疏 MoE 扩展 | Google 2020 |
| **ST-MoE** | Stable and Transferable MoE | 稳定可迁移 MoE | Google 2022 |
| **Soft MoE** | | 软 MoE | Puigcerver 2023 |
| **Top-K Routing** | | Top-K 路由  | |
| **Expert Choice Routing** | | 专家选择路由 | Zhou 2022 |
| **Base Layer Routing** | | 基础层路由  | |

### 16.2 注意力机制

| 术语 | 含义 | 论文 |
| :--- | :--- | :--- |
| **Multi-Head Attention** | 多头注意力 | Vaswani 2017 |
| **Self-Attention** | | 自注意力 |
| **Cross-Attention** | | 交叉注意力 |
| **Target-Attention** | 目标注意力 (DIN) | 阿里 2018 |
| **Relative Attention** | 相对位置注意力 | Shaw 2018 |
| **Flash Attention** | Flash 注意力 | Dao 2022 |
| **Sparse Attention** | | 稀疏注意力 |
| **Linear Attention** | 线性注意力 | Katharopoulos 2020 |
| **Multi-Query Attention** | MQA: 多查询共享 KV | Shazeer 2019 |
| **Grouped-Query Attention** | GQA: 分组查询 | Ainslie 2023 |
| **Sliding Window Attention** | | 滑动窗口注意力 |

### 16.3 推荐系统

| 术语 | 含义 | 备注 |
| :--- | :--- | :--- |
| **Collaborative Filtering** | 协同过滤 | 经典方法 |
| **Matrix Factorization** | | 矩阵分解 |
| **Item-CF** | | 物品协同过滤 |
| **User-CF** | | 用户协同过滤 |
| **Neural CF** | 神经协同过滤 | He 2017 |
| **DeepFM** | 深度因子分解机 | 2017 |
| **xDeepFM** | 极深因子分解机 | 2018 |
| **AutoInt** | 自动特征交互 | 2019 |
| **DCN** | Deep & Cross Network | 2017 |
| **DCN-V2** | DCN 改进 | 2020 |
| **FiBiNET** | | 2019 |
| **AFM** | Attentional Factorization Machine | 2017 |
| **AutoFIS** | | 2020 |
| **Deep Cross Network** | | - |
| **FNN** | Factorization Machine supported NN | 2016 |
| **PNN** | Product-based NN | 2016 |
| **Wide & Deep** | | Google 2016 |
| **Deep Crossing** | | MS 2016 |
| **NFM** | Neural FM | 2017 |

### 16.4 序列推荐

| 术语 | 含义 | 论文 |
| :--- | :--- | :--- |
| **GRU4Rec** | GRU 推荐 | Hidasi 2015 |
| **SASRec** | Self-Attentive Sequential Rec | Kang 2018 |
| **BERT4Rec** | BERT 推荐 | Sun 2019 |
| **FISSA** | | - |
| **HGN** | Hierarchical Gating Network | Ma 2019 |
| **S3-Rec** | Self-Supervised Rec | Zhou 2020 |
| **CL4SRec** | Contrastive Learning Rec | Xie 2022 |
| **Recformer** | | Li 2023 |
| **TIGER** | | (重复, 见前) |

### 16.5 冷启动方法

| 术语 | 含义 | 备注 |
| :--- | :--- | :--- |
| **DropoutNet** | | 2017 |
| **MeLU** | Meta-Learning | 2019 |
| **GAR** | | - |
| **CMMLU** | | Cold-start Multi-task |
| **HyRec** | | Hybrid Rec |
| **MetaEmb** | | Meta Embedding |
| **P5** | Pretrain, Personalized Prompt | 2022 |
| **GPT4Rec** | | 2023 |

### 16.6 评估指标

| 术语 | 含义 | 公式 |
| :--- | :--- | :--- |
| **Precision@K** | Top-K 精确率 | $\frac{\lvert\text{relevant} \cap \text{top-K}\rvert}{\lvert\text{top-K}\rvert}$ |
| **Recall@K** | Top-K 召回率 | $\frac{\lvert\text{relevant} \cap \text{top-K}\rvert}{\lvert\text{relevant}\rvert}$ |
| **F1@K** | F1 分数 | $\frac{2 P R}{P + R}$ |
| **NDCG@K** | | 归一化折损累积增益 |
| **MAP@K** | | 平均精度均值 |
| **MRR** | | 平均倒数排名 |
| **HIT@K** | | 命中率 |
| **AUC** | | ROC 曲线下面积 |
| **GAUC** | | 分组 AUC |
| **LogLoss** | 对数损失 | $-\frac{1}{N}\sum y \log p$ |
| **Brier Score** | | Brier 分数 |

### 16.7 训练技巧

| 术语 | 含义 |
| :--- | :--- |
| **Warmup** | 训练预热 (线性增加 LR) |
| **Cosine Annealing** | 余弦退火 LR |
| **EMA of weights** | 参数 EMA 平滑 |
| **Gradient Clipping** | 梯度裁剪 |
| **Mixed Precision (AMP)** | 混合精度训练 |
| **Gradient Accumulation** | 梯度累积 |
| **Distributed Training** | 分布式训练 (DDP) |
| **ZeRO** | 优化器状态分片 (DeepSpeed) |
| **FSDP** | Fully Sharded Data Parallel |
| **DataLoader** | 数据加载器 |
| **Prefetch** | 数据预取 |
| **Pin Memory** | 锁页内存 |
| **Dropout** | Dropout 正则化 |
| **Label Smoothing** | 标签平滑 |
| **Early Stopping** | 早停 |
| **Checkpoint** | 检查点保存 |
| **Knowledge Distillation** | 知识蒸馏 |
| **Quantization-Aware Training (QAT)** | 量化感知训练 |

### 16.8 推理优化

| 术语 | 含义 |
| :--- | :--- |
| **TorchScript** | PyTorch JIT 编译 |
| **ONNX** | 开放模型格式 |
| **TensorRT** | NVIDIA 推理引擎 |
| **OpenVINO** | Intel 推理引擎 |
| **TVM** | 通用深度学习编译器 |
| **Triton** | NVIDIA 推理服务器 |
| **KV Cache** | KV 缓存 (LLM 推理) |
| **PagedAttention** | 分页注意力 (vLLM) |
| **Continuous Batching** | 连续批处理 |
| **Speculative Decoding** | 投机解码 |
| **Quantization (INT8/INT4)** | 推理量化 |
| **Pruning** | 模型剪枝 |
| **Distillation** | 模型蒸馏 |
| **Model Parallelism** | 模型并行 |
| **Pipeline Parallelism** | 流水线并行 |
| **Tensor Parallelism** | 张量并行 |

### 16.9 推荐系统架构

| 术语 | 含义 |
| :--- | :--- |
| **Recall** | 召回 (从百万级筛到百级) |
| **Rough Rank** | 粗排 (从百级筛到几十级) |
| **Fine Rank** | 精排 (从几十级筛到个位级) |
| **Re-rank** | 重排 (考虑多样性、规则) |
| **Cold Start Pipeline** | 冷启动链路 |
| **Real-time Pipeline** | 实时链路 |
| **Near-line Pipeline** | 近线链路 |
| **Offline Pipeline** | 离线链路 |
| **Feature Store** | 特征存储 |
| **Online Feature** | 在线特征 |
| **Streaming Feature** | 流式特征 |
| **Batch Feature** | 批式特征 |

### 16.10 Embedding 技术

| 术语 | 含义 | 备注 |
| :--- | :--- | :--- |
| **Word2Vec** | 词向量 | 2013 |
| **GloVe** | 全局词向量 | 2014 |
| **BERT** | | 2018 |
| **Sentence-BERT** | | 2019 |
| **BGE** | | BAAI 2023 |
| **E5** | | Microsoft 2023 |
| **OpenAI Embedding** | | 2022 |
| **Cohere Embed** | | 2022 |
| **Item Embedding** | | - |
| **User Embedding** | | - |
| **Contextual Embedding** | | - |
| **Rotary Position Embedding** | RoPE | Su 2021 |
| **AliBi** | Attention with Linear Biases | Press 2022 |

---

## 17. SID 相关算法库

| 库 | 语言 | 用途 | GitHub |
| :--- | :--- | :--- | :--- |
| **FAISS** | C++/Python | 向量检索 | facebookresearch/faiss |
| **Annoy** | C++/Python | ANN 检索 | spotify/annoy |
| **ScaNN** | C++/Python | Google ANN | google-research/google-research |
| **hnswlib** | C++/Python | HNSW ANN | nmslib/hnswlib |
| **Milvus** | Go/C++ | 向量数据库 | milvus-io/milvus |
| **Qdrant** | Rust | 向量数据库 | qdrant/qdrant |
| **Pinecone** | | 云向量数据库  | |
| **Weaviate** | Go | 向量数据库 | weaviate/weaviate |
| **Chroma** | Python | 向量数据库 | chroma-core/chroma |
| **LanceDB** | Rust | 向量数据库 | lancedb/lancedb |
| **Vespa** | Java | 向量数据库 | vespa-engine/vespa |
| **pgvector** | C | Postgres 向量扩展 | pgvector/pgvector |

---

## 18. SID 工业案例补充

### 18.1 Pinterest

Pinterest 用 **PinSage** (GraphSAGE 变体) + SID。

| 配置 | 值 |
| :--- | :--- |
| Embedding 来源 | PinSage 图嵌入 |
| SID 方案 | RQ-VAE |
| L | 4 |
| K | 256 |
| 数据规模 | 数亿 Pin |

详见 [PinSage 论文](https://arxiv.org/abs/1806.01973)。

### 18.2 Amazon

Amazon 用 **DLRM + SID**。

| 配置 | 值 |
| :--- | :--- |
| Embedding 来源 | 商品文本 + 图像 |
| SID 方案 | RQ-VAE |
| L | 5 |
| K | 512 |
| 数据规模 | 数亿商品 |

详见 Amazon Science 公开资料。

### 18.3 Microsoft

Microsoft Bing 用 **BERT-based SID**。

| 配置 | 值 |
| :--- | :--- |
| Embedding 来源 | BERT 文档表示 |
| SID 方案 | RQ-VAE |
| L | 4 |
| K | 256 |
| 数据规模 | 数亿文档 |

详见 Microsoft Research 公开资料。

### 18.4 Baidu

百度搜索 + 推荐用 **ERNIE-based SID**。

| 配置 | 值 |
| :--- | :--- |
| Embedding 来源 | ERNIE 中文预训练 |
| SID 方案 | RQ-VAE |
| L | 5 |
| K | 256 |
| 数据规模 | 数亿中文网页/商品 |

详见百度 ERNIE 论文。

### 18.5 Kuaishou

快手短视频用 **类似 HSTU 的方案**。

| 配置 | 值 |
| :--- | :--- |
| Embedding 来源 | 视频多模态 |
| SID 方案 | RQ-VAE + 协同对比 |
| L | 5 |
| K | 256 |
| 数据规模 | 数亿视频 |

详见 Kuaishou 技术博客。

### 18.6 JD.com

京东电商用 **DIN + SID**。

| 配置 | 值 |
| :--- | :--- |
| Embedding 来源 | 商品多模态 |
| SID 方案 | RQ-VAE + HY-Token 风格 |
| L | 6 |
| K | 256 |
| 数据规模 | 数亿 SKU |

详见京东技术博客。

---

## 19. SID 与其他范式对比

### 19.1 SID vs Collaborative Filtering

| 维度 | SID | CF (item-based) |
| :--- | :--- | :--- |
| 表示 | 离散码 | 连续向量 |
| 检索 | 哈希 / 索引 | ANN |
| 冷启动 | 优 | 差 |
| 解释 | 中 | 弱 |
| 训练 | 离线 | 在线 |

### 19.2 SID vs LLM Embedding

| 维度 | SID | LLM Embedding |
| :--- | :--- | :--- |
| 维度 | L 层 × log K bit | D 维连续 |
| 检索 | 直接查表 | ANN |
| 语义 | 强 | 强 |
| 训练 | 聚类 | 大模型 |
| 部署 | 极快 | 中等 |

### 19.3 SID vs Knowledge Graph

| 维度 | SID | KG (TransE) |
| :--- | :--- | :--- |
| 表示 | 离散码 | 实体 + 关系 |
| 可解释 | 强 | 强 |
| 推理 | 直接 | 多跳 |
| 数据 | 行为 | 关系 |
| 冷启动 | 优 | 中 |

### 19.4 SID vs Hashing

| 维度 | SID | LSH / Hash |
| :--- | :--- | :--- |
| 语义 | 强 | 弱 |
| 训练 | K-Means | 无 |
| 检索 | 中 | 极快 |
| 重建 | 较好 | 差 |

---

## 20. SID 教程推荐路径总览

### 20.1 推荐学习路径（重置版）

```text
入门 (1 周)
  ├─ 第 1 章 SID 动机
  └─ 第 2 章 VQ-VAE 基础

核心 (2-3 周)
  ├─ 第 3 章 RQ-VAE (最重要)
  └─ 第 4 章 多模态 SID

应用 (3-4 周)
  ├─ 第 5 章 冷启动
  ├─ 第 6 章 生成式检索
  └─ 第 8 章 评估与监控

进阶 (4-6 周)
  ├─ 第 9 章 DLRM 实战
  ├─ 第 10 章 下游集成
  ├─ 第 11 章 LLM4Rec
  └─ 第 12 章 字节 TRM

横向 (持续)
  └─ 第 7 章 跨域与联邦
```

### 20.2 推荐资源配套

| 学习阶段 | 必读论文 | 必跑代码 |
| :--- | :--- | :--- |
| 入门 | VQ-VAE | [第 3 章 附录基础版] |
| 核心 | TIGER, HSTU | [第 3 章 附录工业版] |
| 应用 | PLUM, OneRec | [第 9 章] |
| 进阶 | TRM, HY-Token | [第 12 章] |

---

## 21. SID 资源导航

### 21.1 论文仓库

- **arXiv**: https://arxiv.org/list/cs.IR/recent
- **Papers With Code**: https://paperswithcode.com/task/recommendation-systems
- **Semantic Scholar**: https://www.semanticscholar.org/

### 21.2 开源项目

- **Facebookresearch/HSTU**: Meta 官方
- **Google-deepmind/tiger**: TIGER 官方
- **Alibaba/OneRec**: 阿里官方
- **Spotify/Production-SIDs**: Spotify 官方

### 21.3 教程博客

- **Eugene Yan**: 推荐系统最佳实践
- **RecSys Blog**: ACM 推荐系统博客
- **阿里技术博客**: OneRec 等
- **字节技术博客**: TRM / HY-Token

### 21.4 在线课程

- **Coursera 推荐系统专项**
- **Stanford CS246**: 挖掘大数据集
- **Stanford CS224N**: NLP 与推荐

---

## 22. SID 工业指标基准 (Industry Benchmarks)

### 22.1 公开数据集

| 数据集 | 规模 | 用途 |
| :--- | :--- | :--- |
| MovieLens-25M | 25M 评分 | 推荐 |
| Amazon Reviews | 数百万评论 | 电商 |
| Steam | 200K 用户 | 游戏 |
| Yelp | 数百万商家 | 本地推荐 |
| Criteo | 数亿展示 | 广告 |
| Avazu | 数亿展示 | 广告 |
| Industrial MovieLens | | 工业级扩展 |

### 22.2 评估基准

| 任务 | 数据集 | 指标 | 基线 |
| :--- | :--- | :--- | :--- |
| Recall@10 | MovieLens | 0.182 | raw ID |
| Recall@50 | Amazon | 0.243 | raw ID |
| NDCG@10 | Yelp | 0.156 | raw ID |
| AUC | Criteo | 0.812 | DLRM |

### 22.3 SID 增益经验值

| 场景 | SID vs raw ID | 备注 |
| :--- | :--- | :--- |
| 长尾物品 | +20-40% | 长尾受益最大 |
| 冷启动 | +30-50% | 显著 |
| 热门物品 | +5-10% | 略升 |
| 整体 | +10-20% | 综合 |

---

## 23. SID 与具体任务的接口

### 23.1 SID → 召回

```python
# 召回: SID 相似度
def recall(sid_query, sid_items, top_k=100):
    """基于 SID 的召回"""
    # 1. 算 SID embedding 相似度
    emb_query = sum(sid_embedding[sid_query])  # sum-pooling
    embs = sum(sid_embedding[sid_items])  # [N, D]

    # 2. Top-K
    scores = embs @ emb_query
    top_k_idx = scores.topk(top_k)
    return sid_items[top_k_idx]
```

### 23.2 SID → 排序

```python
# 排序: SID 作为特征
class RankModel(nn.Module):
    def __init__(self):
        self.sid_emb = SIDEmbedding(...)
        self.user_encoder = UserEncoder(...)
        self.mlp = MLP(...)

    def forward(self, sid, user_features):
        item_emb = self.sid_emb(sid)
        user_emb = self.user_encoder(user_features)
        return self.mlp(torch.cat([item_emb, user_emb], -1))
```

### 23.3 SID → 生成

```python
# 生成: 自回归预测 SID
class Generator(nn.Module):
    def __init__(self, num_codes, num_layers, dim):
        super().__init__()
        self.transformer = TransformerDecoder(...)
        self.sid_emb = SIDEmbedding(...)
        self.head = nn.Linear(dim, num_codes)

    def forward(self, history_sids, target_sids):
        h = self.sid_emb(history_sids)
        tgt = self.sid_emb(target_sids[:, :-1])
        out = self.transformer(tgt, h)
        return self.head(out)
```

---

## 24. SID 标准接口约定

### 24.1 数据格式

```python
# SID 表 (Parquet)
# item_id, c1, c2, c3, c4, c5
# 0,      23, 145, 67, 234, 12
# 1,      56, 23, 198, 45, 78

# 码本 (NumPy)
# codebooks.npy: shape [K, C, D]

# item2sid 字典
# {0: [23, 145, 67, 234, 12], ...}
```

### 24.2 API 接口

```python
# 训练接口
class SIDCoder:
    def fit(self, embeddings): ...
    def encode(self, embeddings) -> np.ndarray: ...
    def decode(self, sids) -> np.ndarray: ...

# 推理接口 (服务化)
@app.post("/encode")
def encode(items: List[List[float]]) -> List[List[int]]:
    return sid_coder.encode(items)
```

详见 第 09 章 DLRM 生产环境 SID 实战。

---

## 25. SID 速查表 (Cheat Sheet)

| 维度 | 一句话 |
| :--- | :--- |
| 什么是 SID | 物品的离散语义编码，替代 raw ID |
| 为什么用 SID | 冷启动友好 + 参数量小 + 可解释 |
| 怎么生成 SID | RQ-VAE / RQ-Kmeans / FSQ / HY-Token |
| L 选多少 | 4-6 (工业经验) |
| K 选多少 | 256 (默认) / 1024 (高精度) |
| 总容量 | K^L (256^5 ≈ 10^12) |
| 训练时间 | RQ-Kmeans: 10 min; RQ-VAE: 1-2 day |
| 推理时延 | 单条 < 5ms (FAISS/TorchScript) |
| 评估指标 | 重建 MSE + 码本利用率 + 下游 Recall |
| 工业首选 | RQ-Kmeans + FAISS GPU |
| 改进方向 | R3-VAE (训练稳) / HY-Token (可解释) |
| 监控 | 码本利用率 + 残差方差 + 下游 AUC |
| 部署 | TorchScript / ONNX / TensorRT |

---

---

## 26. 补充论文清单（按年份完整版）

### 26.1 2017

- **VQ-VAE** (van den Oord et al., NeurIPS 2017) - 单层量化开山
- **Capsule Network** (Sabour et al., NeurIPS 2017) - 胶囊网络
- **Transformer** (Vaswani et al., NeurIPS 2017) - 注意力机制
- **Wide & Deep** (Cheng et al., RecSys 2016 → 2017) - 推荐基础

### 26.2 2018

- **DIN** (Zhou et al., KDD 2018) - 深度兴趣网络
- **DIEN** (Zhou et al., AAAI 2019) - 兴趣进化
- **ESMM** (Ma et al., SIGIR 2018) - 全空间多任务
- **MMoE** (Ma et al., KDD 2018) - 多门控专家
- **BERT** (Devlin et al., NAACL 2019 → 2018) - 预训练 Transformer
- **GPT-1** (Radford et al., 2018) - 生成式预训练

### 26.3 2019

- **DLRM** (Naumov et al., 2019) - Meta 推荐基座
- **SASRec** (Kang et al., ICDM 2018 → 2019) - 自注意力序列推荐
- **Sentence-BERT** (Reimers et al., EMNLP 2019) - 句子向量
- **T5** (Raffel et al., JMLR 2020 → 2019) - 文本到文本
- **GPT-2** (Radford et al., 2019) - 大规模预训练

### 26.4 2020

- **PLE** (Tang et al., RecSys 2020) - 渐进分层提取
- **PCGrad** (Yu et al., NeurIPS 2020) - 梯度投影
- **BST** (Chen et al., KDD 2019 → 2020) - 行为序列 Transformer
- **GPT-3** (Brown et al., NeurIPS 2020) - 大语言模型
- **TIGER** 概念（早期想法, 2020）

### 26.5 2021

- **CLIP** (Radford et al., ICML 2021) - 图文对齐
- **Vision Transformer** (Dosovitskiy et al., ICLR 2021) - 视觉 Transformer
- **MaE** (He et al., 2021) - Masked Autoencoder
- **DALL-E** (Ramesh et al., 2021) - 文本到图像

### 26.6 2022

- **SoundStream** (Zeghidour et al., 2022) - 音频 RQ
- **DALL-E 2** (Ramesh et al., 2022) - 图像生成
- **Stable Diffusion** (Rombach et al., CVPR 2022) - 扩散模型
- **InstructGPT** (Ouyang et al., 2022) - 指令微调

### 26.7 2023

- **TIGER** (Rajput et al., 2023) ★★★ - SID 在推荐
- **FSQ** (Mentzer et al., CVPR 2023) - 无码本量化
- **GPT-4** (OpenAI, 2023) - 多模态 LM
- **LLaMA 2** (Meta, 2023) - 开源 LLM
- **QLoRA** (Dettmers et al., 2023) - 量化微调
- **DPO** (Rafailov et al., 2023) - 直接偏好优化

### 26.8 2024

- **HSTU** (Zhai et al., SIGIR 2024) ★★★ - Meta 工业 GR
- **LFQ** (Yu et al., 2024) - 查表免费量化
- **OneRec** (阿里, KDD 2024) ★★★ - 端到端 GR
- **Stable Diffusion 3** (Stability AI, 2024) - LFQ 应用
- **Mamba** (Gu et al., 2024) - 状态空间模型
- **LLaMA 3** (Meta, 2024) - 升级 LLM

### 26.9 2025

- **YouTube PLUM** (arXiv 2510.07784, 2025) ★★ - 短视频 SID
- **QuaSID** (2025) - Query-aware SID
- **DeepSeek-V3** (2025) - 开源大模型
- **Gemini 2.0** (Google, 2025) - 多模态 LM

### 26.10 2026

- **R3-VAE** (2026) - 多层跳层
- **CapsID** (2026) - 胶囊化
- **ReSID** (2026) - 循环式
- **ByteDance TRM / HY-Token** (arXiv 2601.22694, 2026) ★★★
- **Spotify Production SIDs** (arXiv 2606.25496, 2026)
- **Snapchat SID** (2026)
- **DeepSeek-R2** (2026) - 多模态扩展

---

## 27. 补充术语（按场景）

### 27.1 推荐场景

| 术语 | 含义 |
| :--- | :--- |
| **冷启动** | Cold Start |
| **长尾** | Long-tail |
| **点击率** | CTR |
| **转化率** | CVR |
| **GMV** | Gross Merchandise Volume |
| **用户停留时长** | Dwell Time |
| **完播率** | Completion Rate |
| **点赞** | Like |
| **评论** | Comment |
| **转发** | Share |
| **收藏** | Bookmark |
| **关注** | Follow |
| **加购** | Add to Cart |
| **下单** | Order |
| **支付** | Payment |

### 27.2 数据流场景

| 术语 | 含义 |
| :--- | :--- |
| **曝光** | Impression |
| **点击** | Click |
| **浏览** | View |
| **反馈** | Feedback |
| **行为序列** | Behavior Sequence |
| **历史** | History |
| **上下文** | Context |
| **实时特征** | Real-time Feature |
| **离线特征** | Offline Feature |
| **标签** | Label |
| **正样本** | Positive Sample |
| **负样本** | Negative Sample |

### 27.3 损失函数

| 术语 | 公式 |
| :--- | :--- |
| **MSE** | $\frac{1}{N}\sum(y-\hat{y})^2$ |
| **MAE** | $\frac{1}{N}\sum\lvert y-\hat{y}\rvert$ |
| **BCE** | $-\frac{1}{N}\sum y\log\hat{y} + (1-y)\log(1-\hat{y})$ |
| **Cross Entropy** | $-\sum p\log q$ |
| **Focal Loss** | $-\alpha(1-p)^\gamma \log p$ |
| **Hinge Loss** | $\max(0, 1-y\hat{y})$ |
| **InfoNCE** | $-\log\frac{\exp(s^+/\tau)}{\sum\exp(s_i/\tau)}$ |
| **Triplet Loss** | $\max(0, d(a,p)-d(a,n)+\text{margin})$ |
| **Margin Ranking** | |
| **KL Divergence** | $\sum p\log\frac{p}{q}$ |

### 27.4 优化算法

| 术语 | 全称 | 备注 |
| :--- | :--- | :--- |
| **SGD** | Stochastic Gradient Descent | 经典 |
| **Momentum** | | 加速 SGD |
| **AdaGrad** | Adaptive Gradient | 自适应 |
| **RMSProp** | Root Mean Square Propagation | 自适应 |
| **Adam** | Adaptive Moment Estimation | 最常用 |
| **AdamW** | Adam with Weight Decay | 推荐 |
| **Lion** | | 2023 |
| **Sophia** | | 2023 |
| **LARS** | Layer-wise Adaptive Rate Scaling | 大 batch |
| **LAMB** | Layer-wise Adaptive Moments for Batch training | 大 batch |

---

## 28. SID 相关数据集

### 28.1 推荐系统数据集

| 数据集 | 用户 | 物品 | 交互 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| MovieLens-100K | 1K | 1.7K | 100K | 入门 |
| MovieLens-1M | 6K | 4K | 1M | 入门 |
| MovieLens-25M | 162K | 62K | 25M | 标准 |
| Amazon Beauty | 22K | 12K | 200K | 电商 |
| Amazon Electronics | 4M | 476K | 7M | 电商 |
| Amazon Movies | 300K | 75K | 4M | 电商 |
| Steam | 200K | 15K | 3M | 游戏 |
| Yelp | 200K | 22K | 1M | 本地 |
| Criteo | 无数 | 无数 | 数十亿 | 广告 |
| Avazu | 无数 | 无数 | 数亿 | 广告 |

### 28.2 文本/多模态数据集

| 数据集 | 内容 | 用途 |
| :--- | :--- | :--- |
| WikiText-103 | 维基百科 | 文本 SID |
| CIFAR-10 | 图像 | 多模态 SID |
| ImageNet | 图像 | 大规模 SID |
| AudioSet | 音频 | 音频 SID |
| Kinetics | 视频 | 视频 SID |

### 28.3 SID 专用数据集

| 数据集 | 内容 | 备注 |
| :--- | :--- | :--- |
| MovieLens-SID | | MovieLens + 预计算 SID |
| Amazon-SID | | Amazon + 预计算 SID |
| Industrial-SID-Bench | 多公司数据 | 通用基准 |

---

## 29. SID 教程配套资源

### 29.1 本教程仓库

```text
SID教程/
├── 0-总目录.md
├── 第01章-SID动机与基础/
│   ├── 0-总目录与阅读路径.md
│   ├── 第01章-01-为什么需要SID-传统ID的三大痛点.md
│   ├── 第01章-02-SID的定义与设计哲学.md
│   ├── 第01章-03-SID的分类学-层次化-扁平化-混合式.md
│   ├── 第01章-04-SID与LLM-Token的对应关系.md
│   └── 第01章-附录-SID术语表与论文清单.md  ← 当前
├── 第02章-VQ-VAE基础/
├── 第03章-RQ-VAE与残差量化/
├── 第04章-多模态与行为增强SID/
├── 第05章-统一语义空间与冷启动/
├── 第06章-生成式检索与约束解码/
├── 第07章-跨域与联邦SID/
├── 第08章-SID评估与监控/
├── 第09章-DLRM生产环境SID实战/
├── 第10章-SID与下游任务集成/
├── 第11章-生成式推荐与LLM/
└── 第12章-ByteDance-TRM/
```

### 29.2 配套代码项目

```text
sid_implementation/
├── src/
│   ├── rq_vae.py
│   ├── rq_kmeans.py
│   ├── fsq.py
│   ├── lfq.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── metrics.py
│   └── visualizer.py
├── examples/
│   ├── train_rqvae_cifar.py
│   ├── train_rqvae_movielens.py
│   └── run_rq_kmeans_vs_vae.py
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── benchmarks.md
└── tests/
    ├── test_rq_vae.py
    └── test_metrics.py
```

### 29.3 跨教程引用

- rec教程 (推荐系统): `../../rec教程/`
- GR教程 (生成式推荐): `../../GR/`
- 基础知识: `../../基础/`
- 数学基础: `../../基础/数学基础/`
- PyTorch 教程: `../../基础/基础知识/PyTorch教程/`

---

## 30. SID 速查图（ASCII）

### 30.1 SID 生成流程

```text
[物品原始特征]
    ↓
[BERT / CLIP / LLM]
    ↓
[物品 embedding N × D]
    ↓
[RQ-VAE / RQ-Kmeans]
    ↓
[SID 表 N × L]
    ↓
[item2sid 映射]
    ↓
[下游：召回 / 排序 / 生成]
```

### 30.2 多模态 SID

```text
文本 → BERT ─┐
            ├→ Concat → RQ-Kmeans → SID
图像 → CLIP ─┘
```

### 30.3 HY-Token 双层

```text
类目 ID ──→ 粗层码 (固定)
                │
物品 embedding → Encoder ─┘
                ↓
            RQ-VAE (L-1 层)
                ↓
            细层码
                ↓
        (粗, 细1, 细2, ..., 细L-1)
```

---

## 31. SID 学习资源 (GitHub Stars ≥ 1000)

| 仓库 | Stars | 说明 |
| :--- | :--- | :--- |
| facebookresearch/faiss | 30K+ | FAISS |
| facebookresearch/HSTU | 1K+ | HSTU |
| google-deepmind/tiger | 500+ | TIGER |
| alibaba/OneRec | 200+ | OneRec |
| huggingface/transformers | 130K+ | LLM |
| pytorch/pytorch | 80K+ | PyTorch |
| scikit-learn/scikit-learn | 60K+ | sklearn |
| explosion/spaCy | 30K+ | NLP |

---

## 32. SID 工业话术

### 32.1 向上汇报

> "我们用 RQ-VAE 把物品编码成 5 层 256 码的语义 ID，总空间 10^12 可表示亿级物品。在 MovieLens 上 Recall@10 比 raw ID 高 18%，冷启动物品提升 45%。"

### 32.2 跨团队沟通

> "SID 是物品的语义编码，类似 LLM 的 token。我们把物品 ID 换成 SID，就能用 Transformer decoder 生成推荐序列。"

### 32.3 面试表达

> "RQ-VAE 用 5 层残差量化把连续 latent 离散成 L 个码字，码字之和重建原始 embedding。优势是码本小（256 而非 256^5）、层次语义天然。"

---

## 33. 附录维护与更新

### 33.1 更新频率

- **每月**：补充新论文。
- **每季度**：更新术语表。
- **每年**：重大更新。

### 33.2 贡献方式

欢迎在仓库提 PR：
- 新术语补全
- 新论文引用
- 错别字修正
- 代码示例补充

### 33.3 反馈渠道

- GitHub Issues
- 知乎私信
- 邮件列表

---

> **附录结束**
>
> 这是 SID 教程的"百科全书"——从词条到论文，从公式到代码，从工业案例到学习路径。
>
> 建议**收藏**，日常翻阅。SID 领域仍在快速发展，持续关注 arXiv。
>
> **下一步**: 开始读正文 [第 01 章 §1 为什么需要 SID](第01章-01-为什么需要SID-传统ID的三大痛点.md)。
>
> **版本**: v3.0 (2026-07-19) · **词条数**: 250+ · **论文数**: 60+ · **公式数**: 50+ · **总字数**: ~50000 字
