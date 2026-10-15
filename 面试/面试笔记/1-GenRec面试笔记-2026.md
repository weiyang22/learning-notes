---
title: GenRec 面试笔记 2026
tags:
  - GenRec
  - 生成式推荐
  - 推荐算法
  - 面试
  - HSTU
  - OneRec
  - MTGR
  - SID
  - TIGER
  - 2024-2026
description: "定位: 2024-2026 GenRec / 推荐系统 算法工程师面试笔记 范围: 字节 / 阿里 / 腾讯 / 美团 / 京东 / 小红书 / 快手 / 拼多 2024-2026 GenRec / Recommendation System 算法工程师面试笔记。 聚焦 HSTU / OneRec / MTGR / TIGER / SID / Scaling 等最新趋势， 涵盖字节/阿里/腾讯/美团/京东/小红书/快手 推荐算法面试真题 + 50+ 经典模型题 + 系统设计题。"
last_updated: 2026-07-23
status: ✅ v1.0
---

# GenRec 面试笔记 2026

> **定位**: 2024-2026 GenRec / 推荐系统 算法工程师面试笔记
> **范围**: 字节 / 阿里 / 腾讯 / 美团 / 京东 / 小红书 / 快手 / 拼多多 推荐算法
> **重点**: HSTU / OneRec / MTGR / TIGER / SID / Scaling Law / 24 篇 GenRec 论文
> **风格**: 公式 + ASCII 图 + 表格 + 真题 + 参考答案

---

## 0. 一句话总结

> **GR (Generative Recommendation) 已经从 2024 年 Meta HSTU 的"ChatGPT 时刻"预言, 演化到 2025-2026 年快手 OneRec 系列的端到端替代级联架构, 美团 MTGR / 阿里 NEZHA / 京东 OxygenREC / 腾讯 GPR 全面落地——推荐系统的范式革命正在发生, 这是面试的最高优先级话题。**

---

## 1. 推荐算法面试概览 (2024-2026 趋势)

### 1.1 大厂推荐算法面试侧重点

| 公司 | 推荐算法面试侧重 | 典型提问 | 难度 |
| :--- | :--- | :--- | :---: |
| **字节** (抖音/TikTok) | 业务驱动 + 工程链路 + 召回粗排精排细节 + 长序列建模 + RankMixer | "设计抖音推荐流水线" / "MMoE 跷跷板现象" / "DIN 注意力" / "特征一致性" | ⭐⭐⭐⭐⭐ |
| **阿里** (淘宝/闲鱼/Alimama) | 多目标 (ESMM/MMoE/PLE) + 业务理解 + RL/精排 + URM/NEZHA/ReaSeq | "ESMM 怎么解决 CVR 偏差" / "MMoE 跷跷板" / "GRPO vs PPO" | ⭐⭐⭐⭐⭐ |
| **腾讯** (微信视频号/看点) | 广告 ROI + GPR + HEPO + 多场景建模 + Bandit | "GPR 的 HHD 解码器" / "HEPO 算法" / "如何做冷启动" | ⭐⭐⭐⭐⭐ |
| **美团** (外卖/到店) | MTGR + HSTU + Group LayerNorm + Scaling Law + 检索 / 排序 | "MTGR 与 HSTU 的差异" / "BM25 vs TF-IDF" / "GRPO 损失" | ⭐⭐⭐⭐ |
| **京东** (首页/频道) | OxygenREC + 快慢思考 + SA-GCPO + 指令跟随 + 电商多场景 | "指令跟随式 GR 怎么做" / "多场景建模" / "CTR 预估模型" | ⭐⭐⭐⭐ |
| **小红书** (笔记/视频) | 多模态 + 内容理解 + 冷启动 + SAGE | "多模态 embedding" / "冷启动策略" / "如何做内容召回" | ⭐⭐⭐⭐ |
| **快手** (主 APP/极速版) | OneRec 系列 + 端到端 GR + ECPO/GBPO + IPA + 会话级生成 | "OneRec 端到端架构" / "Lazy Decoder 为什么减 94% 计算" / "ECPO vs GRPO" | ⭐⭐⭐⭐⭐ |
| **拼多多** | 性价比推荐 + CTR/CVR + 多目标 | "ESMM 应用" / "MMoE" / "双塔召回" | ⭐⭐⭐ |
| **百度** (搜索+信息流) | GRAB + 搜索推荐融合 + LLM 启发模型 | "GRAB CamA 注意力" / "搜索排序架构" | ⭐⭐⭐⭐ |
| **华为** | RelayGR + 昇腾 NPU + 跨阶段推理 | "RelayGR 推理优化" / "国产 NPU 适配" | ⭐⭐⭐ |

### 1.2 2025-2026 最新趋势 (面试必备)

1. **GR 范式革命**:
   - 2024 Q1 Meta HSTU: 首次验证推荐 Scaling Law
   - 2024 Q4 阿里 NEZHA: 推测解码 + 搜索广告落地
   - 2025 Q1 快手 OneRec-V1: 端到端 GR 替代级联架构
   - 2025 Q2 美团 MTGR: 保留交叉特征的混合架构
   - 2025 Q3 腾讯 GPR: 广告 One-Model 范式
   - 2025 Q4 京东 OxygenREC: 指令跟随 + 多场景
   - 2026 Q1 多家公司: 测试时间扩展 (TTS) + 过程奖励 (PROMISE)
2. **SID (Semantic ID) 标准化**:
   - RQ-VAE → RQ-KMeans → OPQ parallel codebook (阿里)
   - 层次化码本 + 渐进式掩码 (Google PLUM)
   - 均匀语义映射 + Sinkhorn 求解 (LC-Rec)
3. **LLM × Rec 融合**:
   - LLM 作为通用检索器 (URM)
   - LLM 作为奖励模型 + RLHF/GRPO 对齐
   - 显式推理 (OneRec-Think, ReaSeq)
4. **Scaling Law 在推荐上验证**:
   - HSTU 1.5T 参数: 在线 +12.4%
   - OneRec-V2 8B 参数: 严格遵循 Scaling Law
   - MTGR-large: 137.87 GFLOPs 单样本 (160x DLRM)
5. **多模态 GR**:
   - 文本/图像/行为 SID 统一 (SAGE, PinRec)
   - QARM V2: 多模态序列对齐
   - 字节 MERGE: 流式物品索引

### 1.3 2026 推荐算法面试 P6/P7 必备知识点

| 知识模块 | 必备程度 | 重点内容 |
| :--- | :---: | :--- |
| DLRM 架构 | ⭐⭐⭐⭐⭐ | EmbeddingBag + Bottom MLP + Interaction + Top MLP |
| 召回-粗排-精排-重排 | ⭐⭐⭐⭐⭐ | 各阶段目标 + 延迟 + 模型选择 |
| DIN / DIEN / DSIN | ⭐⭐⭐⭐⭐ | Attention 机制 + 兴趣演化 + Session |
| MMoE / PLE / ESMM / SARNet | ⭐⭐⭐⭐⭐ | 多任务冲突 + 全空间建模 + CVR 偏差 |
| DeepFM / xDeepFM / DCN | ⭐⭐⭐⭐ | 特征交叉 |
| SASRec / BERT4Rec | ⭐⭐⭐⭐⭐ | 序列建模基础 |
| **HSTU** | ⭐⭐⭐⭐⭐ | Pointwise Aggregated Attention + Scaling Law + M-FALCON |
| **OneRec 系列** | ⭐⭐⭐⭐⭐ | Encoder-Decoder → Lazy Decoder + ECPO/GBPO + IPA |
| **MTGR** | ⭐⭐⭐⭐⭐ | 混合架构 + Group LayerNorm + 动态混合掩码 |
| **SID / RQ-VAE / TIGER** | ⭐⭐⭐⭐⭐ | 残差量化 + 语义对齐 + Beam Search |
| 推理优化 (KV-Cache/Flash) | ⭐⭐⭐⭐⭐ | KV-Cache + PagedAttention + Flash Attention |
| **GRPO / PPO / DPO** | ⭐⭐⭐⭐⭐ | 强化学习对齐 |
| RL + Rec (Bandit/Contextual) | ⭐⭐⭐⭐ | 探索利用 + 冷启动 |
| 位置编码 (RoPE/ALiBi) | ⭐⭐⭐⭐ | 长序列外推 |
| Online Learning (FTRL) | ⭐⭐⭐⭐ | 增量训练 |
| A/B 实验 + 因果推断 | ⭐⭐⭐⭐ | IPS / CUPED |
| 向量检索 (Faiss/HNSW) | ⭐⭐⭐⭐ | ANN 算法 |
| 系统设计 (短视频/电商/广告) | ⭐⭐⭐⭐⭐ | 完整方案 |

---

## 2. 推荐系统基础

### 2.1 推荐系统四阶段 Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│                       推荐系统架构                              │
│                                                              │
│   用户请求                                                     │
│     ↓                                                        │
│   ┌──────────────┐                                           │
│   │ 召回 Recall  │   100万+ 物品 → 1000-5000 候选              │
│   │  多路并行     │   - 协同过滤 (i2i/u2i)                      │
│   │              │   - 向量召回 (双塔 + ANN)                   │
│   │              │   - 规则召回 (热门/标签)                     │
│   │              │   - 关注/订阅                              │
│   └──────┬───────┘                                           │
│          ↓                                                   │
│   ┌──────────────┐                                           │
│   │ 粗排 Pre-Rank│   1000-5000 → 200-500                     │
│   │  轻量模型     │   - COLD / 知识蒸馏 / 双塔                   │
│   │  低延迟       │   - 10ms 内完成                            │
│   └──────┬───────┘                                           │
│          ↓                                                   │
│   ┌──────────────┐                                           │
│   │  精排 Rank   │   200-500 → 50-100                        │
│   │  复杂模型     │   - DIN / DIEN / MMoE / DeepFM             │
│   │  多目标       │   - CTR / CVR / 时长 / 互动                │
│   │  实时特征     │   - 实时用户行为序列                        │
│   └──────┬───────┘                                           │
│          ↓                                                   │
│   ┌──────────────┐                                           │
│   │ 重排 Re-Rank │   50-100 → 10-20 最终曝光                  │
│   │  业务规则     │   - 多样性 (MMR / DPP)                     │
│   │  探索利用     │   - 去重 / 规则                            │
│   │  打散         │   - 新鲜度提权                             │
│   └──────┬───────┘                                           │
│          ↓                                                   │
│       曝光 / 返回                                             │
└─────────────────────────────────────────────────────────────┘
```

**为什么需要分层?**
- 精排模型对每个候选都要算一次, 万级候选 → 几万次 DNN, 延迟扛不住
- 召回阶段用轻量、可索引的方式 (双塔 + ANN) 快速从全库缩到上千
- 分层是算力与精度的工程权衡, 单模型无法兼顾

### 2.2 DLRM 架构 (Meta 2019)

```text
┌────────────────────────────────────────┐
│              DLRM 架构                   │
│                                         │
│   Sparse Features (categorical)          │
│   ┌─────────────┬──────────────┐        │
│   │ user_id     │ item_id      │        │
│   │ country     │ category     │        │
│   │ age         │ price        │        │
│   └─────────────┴──────────────┘        │
│         ↓ EmbeddingBag (变长池化)        │
│   ┌─────────────┬──────────────┐        │
│   │  e_u (稠密)  │  e_i (稠密)   │        │
│   └─────────────┴──────────────┘        │
│                                         │
│   Dense Features (数值)                  │
│   ┌─────────────────────────────┐       │
│   │ age, price, ctr, ...        │       │
│   └─────────────────────────────┘       │
│         ↓ Bottom MLP                     │
│   ┌─────────────────────────────┐       │
│   │  d (稠密)                   │       │
│   └─────────────────────────────┘       │
│                                         │
│   Feature Interaction (点积交互)          │
│   ┌─────────────────────────────┐       │
│   │  e_u ⊙ e_i  (二阶交叉)       │       │
│   │  + e_u · d                  │       │
│   └─────────────────────────────┘       │
│         ↓                               │
│   ┌─────────────────────────────┐       │
│   │  Top MLP → pCTR             │       │
│   └─────────────────────────────┘       │
└────────────────────────────────────────┘
```

**核心组件**:
1. **EmbeddingBag**: 处理变长稀疏特征 (如用户历史物品列表), 输出固定维度
2. **Bottom MLP**: 处理 dense 数值特征
3. **Feature Interaction**: 两两特征点积, 模拟二阶交叉 (替代 FM 的高效实现)
4. **Top MLP**: 最终打分

### 2.3 协同过滤与矩阵分解

#### 2.3.1 User-CF vs Item-CF

| 维度 | User-CF | Item-CF |
| :--- | :--- | :--- |
| 核心思想 | 相似用户喜欢的物品 | 与历史物品相似的物品 |
| 适合场景 | 用户数 << 物品数 (新闻) | 物品数 << 用户数 (电商) |
| 冷启动 | 新物品友好 | 新用户友好 |
| 计算 | 用户-用户相似度矩阵 | 物品-物品相似度矩阵 |
| 实时性 | 用户行为更新实时影响 | 物品更新实时影响 |

#### 2.3.2 矩阵分解 (MF)

```python
# 简化版 MF
class MF(nn.Module):
    def __init__(self, n_user, n_item, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_user, dim)
        self.item_emb = nn.Embedding(n_item, dim)

    def forward(self, user, item):
        u = self.user_emb(user)   # (B, dim)
        i = self.item_emb(item)   # (B, dim)
        return (u * i).sum(dim=-1)  # 内积
```

#### 2.3.3 BPR Loss

```python
def bpr_loss(pos_score, neg_score):
    """
    Bayesian Personalized Ranking
    让正样本打分 > 负样本打分
    """
    return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

# 与 CE 的区别:
# BPR: pairwise, 不需要绝对分值, 只关心相对顺序
# CE: pointwise, 需要预测精确概率
```

### 2.4 基础 50 面试题

> 以下题目按难度排列, ⭐ = 易, ⭐⭐⭐⭐⭐ = 难

#### Q1. AUC 的物理意义是什么? 为什么不用准确率?
- **难度**: ⭐⭐
- **答案**:
  - AUC = 随机一对 (正, 负) 样本, 正样本排在负样本前的概率
  - 不依赖阈值, 适合样本不平衡场景 (CTR 通常 < 10%)
  - 准确率在不平衡数据下会失真 (全预测负样本就有 90% 准确率)
- **追问**: GAUC 是什么? = Group AUC, 用户内 AUC 加权平均
- **来源**: 牛客网

#### Q2. 为什么推荐系统用 Log Loss 不用 MSE?
- **难度**: ⭐⭐⭐
- **答案**:
  - **梯度性质**: Log Loss + Sigmoid 梯度 = `p - y` (线性), MSE 梯度 = `(p-y)·σ'(p)` (含 sigmoid 导数, 容易饱和)
  - **概率校准**: Log Loss 输出天然是概率, MSE 输出可解释性差
  - **理论依据**: Log Loss 是极大似然估计的最优损失
- **追问**: 回归场景下用什么? MSE / Huber Loss
- **来源**: 字节跳动 一面

#### Q3. FM vs DeepFM 的区别?
- **难度**: ⭐⭐
- **答案**:
  - FM: 显式二阶交叉 `w_ij * x_i * x_j`
  - DeepFM: FM (低阶) + DNN (高阶), 并行结构共享 embedding
- **追问**: FM 复杂度? O(kn), vs 两两交叉 O(kn²)
- **来源**: 经典题

#### Q4. Embedding 维度怎么选?
- **难度**: ⭐⭐⭐
- **答案**:
  - 经验公式: `min(64, max(8, vocab^0.25 * 4))`
  - 过大: 过拟合, 显存爆炸, 训练慢
  - 过小: 表达不足, 欠拟合
  - 实战: 64 / 128 / 256, 看 AUC 和验证集表现
- **追问**: 1 亿物品, 64 维需要多少显存? 1e8 * 64 * 4B = 25.6 GB

#### Q5. 召回为什么用近似检索 (ANN)?
- **难度**: ⭐⭐
- **答案**:
  - 物品库百万到十亿级, 精确 KNN O(N) 不可行
  - Faiss / Milvus / HNSW 做 ANN, 复杂度 O(log N) ~ O(N^0.5)
  - 召回阶段对精度容忍度高 (候选 1000, 真正曝光 10), 但对速度要求极高
- **追问**: HNSW 是什么? Hierarchical Navigable Small World, 图索引

#### Q6. 召回和排序的差异?
- **难度**: ⭐⭐
- **答案**:
  - **召回**: 减少候选数到 1000 以内, 模型简单, 特征少, 关键是快
  - **排序**: 追求精确, 模型复杂, 特征多, 关键是准
  - 召回率 (Recall) vs 精确率 (Precision) 的权衡
- **追问**: 召回阶段的负样本怎么采? 精排呢? → 召回: 全局负采样; 精排: 用户已曝光未点击

#### Q7. 双塔模型为什么不能精排?
- **难度**: ⭐⭐⭐
- **答案**:
  - User Tower 和 Item Tower 独立编码, 只在最后算内积
  - 丢失 user × item 的 cross 特征
  - 无法做 target-aware attention (DIN 那种)
- **追问**: 双塔怎么改进? → 引入 side info, 浅层 cross, 或者用 PolyEncoder

#### Q8. 推荐系统冷启动怎么办?
- **难度**: ⭐⭐⭐
- **答案**:
  - **新用户**: 注册信息 (年龄/性别/位置), 热门兜底, 多样化试探, Bandit 快速探索
  - **新物品**: 侧信息 (类目/品牌/价格/封面), 元学习, 探索流量, 内容相似召回
  - **新场景**: 迁移学习 (从老场景 Fine-tune), 多任务学习共享底层
- **追问**: E&E 怎么做? → Thompson Sampling / UCB / 探索流量比例

#### Q9. 推荐系统位置偏差怎么解?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **训练时**: 加位置 embedding, 让模型看到位置信息
  - **推理时**: 剥离位置 embedding
  - **因果方法**: IPS 逆倾向加权
  - **数据方法**: 随机化位置 (小流量实验)
- **追问**: 还知道哪些 bias? → selection bias, exposure bias, conformity bias, popularity bias

#### Q10. 协同过滤相似度怎么算? 极端情况怎么处理?
- **难度**: ⭐⭐⭐
- **答案**:
  - **余弦相似度**: `sim(u,v) = u·v / (|u||v|)`, 不考虑评分绝对值
  - **皮尔逊**: 减去均值, 考虑用户评分偏差
  - **Jaccard**: 只看共同行为集合
  - **极端**: 热门物品让所有相似度都高 → 加 IDF 权重 (惩罚热门)
- **追问**: 用户很多怎么算物品间相似度? → 离线分布式计算 (Spark), 或只用物品 embedding (向量召回)

#### Q11. 向量检索 IVF-PQ 的原理?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **IVF** (Inverted File): 把向量空间划成 nlist 个 Voronoi 单元, 检索时只查最近的 nprobe 个单元
  - **PQ** (Product Quantization): 把向量切成 m 段, 每段聚类, 用 m 个 cluster id 表示向量, 压缩比 = 32 / (m * 4)
  - **IVF-PQ**: 先 IVF 缩小范围, 再 PQ 加速距离计算, 适合百万到亿级
- **追问**: HNSW vs IVF-PQ? HNSW 精度高, 内存大; IVF-PQ 压缩比高, 精度略低

#### Q12. LR 的推导? 为什么用 Sigmoid?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - LR: `p(y=1|x) = σ(w·x + b) = 1 / (1 + e^(-w·x-b))`
  - 推导: 极大似然 → 交叉熵损失 → 梯度 = `(p-y)x`
  - Sigmoid: 把线性输出映射到 (0,1) 概率空间
- **追问**: LR vs GBDT? LR 线性, 可解释, 在线学习方便; GBDT 非线性, 精度高, 不能在线更新

#### Q13. 用户行为序列怎么用?
- **难度**: ⭐⭐
- **答案**:
  - **Pooling**: mean/sum/max, 简单但丢失顺序
  - **DIN**: target-aware attention, 跟候选物品算权重
  - **DIEN**: 加 GRU 建模兴趣演化
  - **SASRec/BERT4Rec**: Self-attention 建模序列
- **追问**: 长序列怎么办? → 压缩 (TWIn/LONGER), 检索 (SIM), 预训练表征

#### Q14. 召回通道有哪些类型?
- **难度**: ⭐⭐
- **答案**:
  - **协同过滤**: i2i (物品相似), u2i (用户向量)
  - **向量召回**: 双塔 → ANN
  - **规则召回**: 热门, 标签, 作者关注
  - **图召回**: GraphSAGE, PinSage
  - **树召回**: Yandex 用过, 大规模
- **追问**: 多路召回怎么融合? → 各路取 TopN, 加权打分, 或者粗排再排

#### Q15. 如何保证线上线下特征一致性?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **统一特征 SDK**: 离线训练和在线推理用同一个特征计算代码
  - **特征平台**: 集中存储, Redis 在线拉取
  - **样本回灌**: 训练时用实际线上拉到的特征
  - **监控**: 特征分布监控 (PSI / KS)
- **追问**: 离线涨了线上没涨, 可能原因? → 特征不一致 / 数据穿越 / 流量倾斜 / 长尾差

#### Q16. Embedding 显存爆炸怎么办?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **分片**: TBE (Tencent Basic Embedding), 每张卡存部分
  - **量化**: INT8 4x, INT4 8x
  - **哈希**: Hash Embedding, 减少冲突
  - **稀疏更新**: 只更新出现过的 embedding, 节省 IO
  - **共享**: 多个特征共享同一个 embedding
- **追问**: 10亿 物品 * 64 维 * 4 字节 = 25.6 GB, 单机怎么放? → 分片 + 量化

#### Q17. 用户和物品协同过滤各适合什么场景?
- **难度**: ⭐⭐
- **答案**:
  - **Item-CF**: 用户远多于物品, 物品数稳定, 适合物品少的 (电商)
  - **User-CF**: 用户稳定, 物品快速增加, 适合用户少的 (新闻, 短视频)
  - **冷启动**: 新物品用 User-CF (用相似用户偏好推), 新用户用 Item-CF (用物品相似推)
- **追问**: 抖音用哪种? → 主要是向量召回 (双塔) + 关注/订阅召回 (i2i)

#### Q18. CTR 预估模型的发展史?
- **难度**: ⭐⭐
- **答案**:
  - **LR**: 线性, 2010 前
  - **FM/FFM**: 二阶交叉, 2010-2015
  - **GBDT+LR**: Facebook 2014, 树模型筛特征
  - **Wide & Deep**: Google 2016, 记忆+泛化
  - **DeepFM**: 2017, FM + DNN 并行
  - **DIN/DIEN**: 阿里 2017-2018, attention
  - **MMoE/PLE/ESMM**: 多任务
  - **DCN/DCN-v2**: 显式高阶交叉
  - **HSTU**: Meta 2024, Transformer-based
  - **OneRec/MTGR**: 端到端 GR 2025
- **追问**: 你最看好哪个? 为什么?

#### Q19. 如何处理推荐系统的反馈延迟?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **延迟反馈建模**: 转化延迟分布 (Hazard 模型), 给延迟样本加权
  - **重要性采样**: 等到反馈再训练, 但增加存储成本
  - **近似**: 设置截止窗口 (点击 24h, 转化 7d), 之后标记为负样本
  - **RL 方法**: 用观看时长 / 短期互动作为即时反馈
- **追问**: OneRec-V2 怎么做? → 时长感知奖励塑形 (Duration-Aware Reward Shaping)

#### Q20. 召回评估指标有哪些?
- **难度**: ⭐⭐
- **答案**:
  - **Recall@K**: Top-K 中包含真实正例的比例
  - **Hit Rate@K**: 至少命中一个
  - **NDCG@K**: Normalized Discounted Cumulative Gain, 位置加权
  - **MRR**: 第一个正例的倒数排名
  - **覆盖率**: 召回物品占全库的比例
- **追问**: 召回 vs 精排的指标区别? → 召回看 Recall@K, 精排看 AUC/GAUC

#### Q21. AUC 和 GAUC 的区别? 为什么上线也要看 AUC?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **AUC**: 全局, 对所有正负样本
  - **GAUC**: 用户内 AUC 加权平均, `GAUC = Σ n_u * AUC_u / Σ n_u`
  - GAUC 反映个体级排序质量, 更符合用户体验
  - 但线上也要看 AUC, 因为 GAUC 在极端情况下 (单用户全正或全负) 不稳定
- **追问**: 抖音推荐用什么指标? → App Stay Time, LT7, 完播率

#### Q22. L1 vs L2 正则化的区别?
- **难度**: ⭐⭐
- **答案**:
  - **L1**: `||w||_1`, 产生稀疏解 (很多 0), 适合特征选择
  - **L2**: `||w||_2²`, 平滑, 防止过拟合, 所有权重都小
  - **L1 不可微** 在 0 点, 用次梯度
- **追问**: ElasticNet? → L1 + L2

#### Q23. SGD / BGD / MBGD 的区别?
- **难度**: ⭐⭐
- **答案**:
  - **BGD** (Batch): 全量数据算梯度, 收敛稳定, 但慢
  - **SGD**: 随机一个样本, 快但噪声大
  - **MBGD** (Mini-batch): 折中, 工业界默认 256/512/1024
  - **Adam**: 在 MBGD 上加自适应学习率

#### Q24. 推荐系统的数据穿越怎么检测?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **特征时间戳**: 每个特征带 timestamp, 训练时只能用 ≤ 当前 label 时间的特征
  - **样本 shuffle 顺序**: 按时间切分 train/val/test, 不能随机 shuffle
  - **Gap 检查**: 训练样本 label 时间 vs 验证集第一行时间是否有 gap
  - **特征时延**: 在线拉特征有时间窗口, 离线要还原
- **追问**: 离线涨了线上没涨怎么排查? → 先看数据穿越

#### Q25. 推荐系统的负采样策略?
- **难度**: ⭐⭐⭐
- **答案**:
  - **随机负采样**: 全局随机, 简单但太"简单"
  - **Popularity 负采样**: 按物品热度采样, 更现实
  - **Hard Negative**: 容易被分错的, 提升训练效率 (in-batch negatives 也算)
  - **曝光未点击**: 用户真实负反馈 (精排阶段用)
- **追问**: 召回和精排的负采样区别? → 召回用全局随机, 精排用曝光未点击

#### Q26. BGD, MBGD, SGD 公式?
- **难度**: ⭐⭐
- **答案**:
  - BGD: `w = w - η * ∇L(w; 全量数据)`
  - SGD: `w = w - η * ∇L(w; 一个样本)`
  - MBGD: `w = w - η * ∇L(w; 一个 batch)`
  - 收敛速度: BGD < MBGD < SGD (per step), 总收敛时间相反

#### Q27. Adam vs AdamW?
- **难度**: ⭐⭐
- **答案**:
  - **Adam**: weight decay 加在梯度上, 与 L2 等价 (只对标准损失成立)
  - **AdamW**: weight decay 从梯度分离, 单独衰减权重, 更合理的正则化
  - 现代 LLM 训练默认 AdamW

#### Q28. 推荐系统的多任务学习 Loss 怎么加权?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **手动加权**: 经验调, 简单但费时
  - **Uncertainty Weighting** (Kendall 2018): 每个任务学一个 σ, 自动平衡
  - **GradNorm**: 根据梯度大小动态调权
  - **PCGrad** (Stanford): 梯度冲突时投影, 缓解负迁移
  - **DWA** (Dynamic Weight Averaging): 根据 loss 下降速度加权
- **追问**: 实际工程中用什么? → 多数用手动调, 离线 GAUC 监控

#### Q29. 工业界推荐系统的 AB 实验怎么做?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **流量分桶**: `hash(user_id) % 10000 → bucket`
  - **互斥实验**: 同层只能进一个
  - **分层实验**: 不同层独立 (召回/粗排/精排/UI 各一层)
  - **流量比例**: 实验组 5% / 对照 95% (小流量) → 10% → 50% → 100%
  - **指标监控**: 短 (CTR/CVR), 中 (时长), 长 (留存)
- **追问**: 显著性检验? → t 检验, 卡方, CUPED 方差缩减

#### Q30. 推荐系统的位置 embedding 怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - 训练时把位置信息 (1-20) 作为 embedding 加入
  - 推理时把位置 embedding 置零或忽略
  - 训练加 Dropout 防止对位置过拟合
- **追问**: 还有其他位置偏差方法? → 位置作为 side info, 让模型自行学习

#### Q31. 过拟合怎么解决?
- **难度**: ⭐⭐
- **答案**:
  - 数据: 更多数据 / 数据增强
  - 模型: Dropout / BN / 更小模型 / 简化结构
  - 正则: L1/L2 / Early Stopping
  - 集成: Bagging / Boosting
- **追问**: Dropout 在推理时怎么处理? → 不使用 (model.eval()), 训练时 scale 1/(1-p) 或推理时 * (1-p)

#### Q32. 推荐系统的冷启动 embedding 怎么处理?
- **难度**: ⭐⭐⭐
- **答案**:
  - **新物品**: 用内容 embedding (文本/图像) 初始化, 而非随机
  - **新用户**: 用注册信息 + 行为 embedding 平均
  - **元学习**: MAML 快速适应
  - **少样本学习**: Prototypical Networks
- **追问**: 字节 MERGE 怎么做? → 流式物品索引, 实时生成内容 embedding

#### Q33. 推荐系统的曝光去重怎么实现?
- **难度**: ⭐⭐⭐
- **答案**:
  - **倒排索引**: 用户 → 最近曝光物品集合
  - **Bloom Filter**: 快速判断是否曝光过 (允许小概率误判)
  - **Embedding 检索**: 用 hash 快速查重
- **追问**: 多设备同步? → 中心化曝光记录, 跨设备同步

#### Q34. 推荐系统怎么做长尾物品的推荐?
- **难度**: ⭐⭐⭐
- **答案**:
  - **数据**: 长尾物品降采样 + 热门物品降采样
  - **模型**: 物品侧 embedding 加热度因子
  - **流量**: 探索流量 (5-10%)
  - **重排**: 多样性约束 (MMR), 给长尾曝光机会
- **追问**: 怎么平衡热门和长尾? → 多目标融合, 长尾物品的召回分加权重

#### Q35. 推荐系统如何处理数据稀疏?
- **难度**: ⭐⭐⭐
- **答案**:
  - **特征工程**: 交叉特征 / 统计特征
  - **模型**: GBDT 树模型 / FM 类
  - **迁移学习**: 从其他任务借力 (CTR 数据多, CVR 借 CTR 的 embedding)
  - **ESMM**: 全空间建模, CVR 通过 pCTR × pCVR 学习
- **追问**: CVR 稀疏怎么解决? → ESMM 是经典方案

#### Q36. 推荐系统的实时特征怎么用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **Flume/Kafka**: 行为日志实时流
  - **Flink/Spark Streaming**: 实时特征计算
  - **Redis**: 在线存储, 毫秒级延迟
  - **特征平台**: 离线/在线统一计算
- **追问**: 实时特征失效怎么办? → 设置 TTL, 降级用离线特征

#### Q37. 推荐系统的用户兴趣建模怎么演进?
- **难度**: ⭐⭐⭐
- **答案**:
  - **长期 + 短期**: 分别建模 (长期: 30 天, 短期: 1 小时)
  - **Gating**: 长短期融合门
  - **DIEN**: 兴趣演化 GRU
  - **多模态**: 文本/图像/视频 embedding 融合
- **追问**: HSTU 怎么建模? → 把所有特征 token 化, 用 Pointwise Aggregated Attention

#### Q38. 推荐系统怎么做探索与利用 (EE)?
- **难度**: ⭐⭐⭐
- **答案**:
  - **Bandit 算法**: UCB, Thompson Sampling
  - **Contextual Bandit**: LinUCB, 加入上下文
  - **强化学习**: DQN/PPO/GRPO
  - **混合**: 90% 利用 + 10% 探索
- **追问**: 抖音怎么平衡? → Bandit + 内容标签 + 多样性约束

#### Q39. 推荐系统的强化学习怎么用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **State**: 用户历史 + 上下文
  - **Action**: 推荐物品
  - **Reward**: 点击/时长/转化
  - **算法**: DQN, PPO, GRPO, ECPO
- **追问**: GRPO 和 PPO 的区别? → GRPO 不要 critic, 组内归一化估计 advantage, 省显存

#### Q40. 推荐系统怎么做特征选择?
- **难度**: ⭐⭐⭐
- **答案**:
  - **过滤法**: 方差 / 互信息 / 卡方
  - **包装法**: RFE / 序列选择
  - **嵌入法**: L1 正则 / 树模型特征重要性
  - **业务理解**: 人工剔除明显无效特征
- **追问**: 树模型怎么做特征重要性? → Gain / Split / Coverage

#### Q41. 推荐系统的样本构造怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - **正样本**: 用户点击/转化/完播
  - **负样本**: 用户曝光未点击 (精排) / 全局随机 (召回)
  - **采样比例**: 1:5, 1:10, 根据数据量调
  - **时间窗口**: 训练样本时间范围, 避免穿越
- **追问**: 为什么要控制正负样本比? → 太负样本多, 模型只学"不点击"

#### Q42. 推荐系统的效果评估怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - **离线**: AUC, GAUC, NDCG, Recall@K, Hit Rate@K
  - **在线**: CTR, CVR, 时长, 留存, GMV
  - **A/B 实验**: 小流量验证
  - **长线**: 7 日/30 日留存, 月活
- **追问**: 离线涨了线上没涨原因? → 数据穿越 / 特征不一致 / 流量倾斜 / 长尾差

#### Q43. 推荐系统的特征监控怎么做?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **分布监控**: PSI (Population Stability Index), KS 检验
  - **缺失率监控**: 实时告警 (超过 5% 告警)
  - **值域监控**: 数值特征的最大最小值
  - **预测分布**: 模型输出是否偏移
- **追问**: PSI 阈值? → > 0.1 黄色, > 0.2 红色

#### Q44. 推荐系统的模型监控怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - **指标监控**: 实时 AUC (困难), 业务指标
  - **分布监控**: 预测值分布 vs 历史
  - **A/B 实验**: 实验组 vs 对照组指标差
  - **性能监控**: QPS, P99 延迟, 错误率
- **追问**: 模型漂移怎么发现? → PSI + 在线指标

#### Q45. 推荐系统的 Embedding 怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - **初始化**: 随机 or 预训练
  - **训练**: end2end (梯度反传) / 分阶段 (w2v, Item2Vec)
  - **存储**: Hash Embedding, Quantization
  - **服务**: 离线训练, 在线查表
- **追问**: Item2Vec 怎么做? → 跟 Word2Vec 一样, 把用户行为序列当作句子

#### Q46. 推荐系统的双塔模型怎么训练?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **正样本**: 用户点击的物品
  - **负样本**: 全局随机 (in-batch 也算)
  - **Loss**: InfoNCE / 温度缩放 softmax
  - **温度**: τ 越小越聚焦难样本
- **追问**: 温度系数怎么选? → 0.05-0.1, 看 Recall@K 表现

#### Q47. 推荐系统的 ANN 算法对比?
- **难度**: ⭐⭐⭐
- **答案**:
  - **Flat**: 暴力, 100万以下
  - **IVF**: 倒排, 1 亿级
  - **HNSW**: 图索引, 精度高, 内存大
  - **PQ**: 乘积量化, 压缩比高
  - **IVF-PQ**: 工业界主流 (Faiss)
- **追问**: 内存限制下怎么选? → IVF-PQ + HNSW 混合

#### Q48. 推荐系统的 Item2Vec 怎么做?
- **难度**: ⭐⭐
- **答案**:
  - 把用户行为序列当作句子, 物品当作词
  - 用 Skip-gram (Word2Vec) 训练
  - 行为相似的物品 embedding 接近
  - 用于 i2i 召回
- **追问**: 和 Graph Embedding 的区别? → Item2Vec 序列建模, Graph Embedding 图结构建模

#### Q49. 推荐系统的 GNN 怎么做?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **构建图**: 用户-物品二部图
  - **消息传递**: PinSage, GraphSAGE
  - **采样邻居**: 固定数量, 控制复杂度
  - **应用**: 召回 / 特征工程
- **追问**: 工业界怎么落地? → PinSage 在 Pinterest, LightGCN 简化版

#### Q50. 推荐系统的核心能力是哪些?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  1. **业务理解**: 把业务问题转化为数学问题
  2. **特征工程**: 从原始数据提取有效信号
  3. **模型选择**: 根据场景选合适的模型 (DLRM / DIN / GR)
  4. **工程能力**: 训练 / 推理 / 部署全链路
  5. **AB 实验**: 用数据说话, 量化收益
  6. **持续迭代**: 监控 + 反馈 + 优化
- **追问**: 你最擅长哪个? → (根据个人经历答)

---

## 3. 深度学习推荐模型

### 3.1 Wide & Deep (Google 2016)

```text
┌─────────────────────────────────────────┐
│         Wide & Deep 架构                  │
│                                          │
│   输入: User + Context + Item             │
│     ↓                                    │
│   ┌─────────┐    ┌─────────┐              │
│   │  Wide   │    │  Deep   │              │
│   │ (LR)    │    │ (DNN)   │              │
│   │ 记忆能力 │    │ 泛化能力 │              │
│   │ 交叉特征 │    │ Embedding│              │
│   └────┬────┘    └────┬────┘              │
│        └──────┬───────┘                  │
│              ↓                           │
│         Concat → pCTR                    │
└─────────────────────────────────────────┘
```

- **Wide**: 线性, 记忆 (Memorization), 直接学习特征交叉
- **Deep**: DNN, 泛化 (Generalization), Embedding 相似性
- **缺点**: Wide 部分需要人工设计交叉特征

### 3.2 DeepFM (2017)

```text
┌─────────────────────────────────────────┐
│          DeepFM 架构                      │
│                                          │
│   输入: 各类特征                           │
│     ↓                                    │
│   ┌─────────┐    ┌─────────┐              │
│   │   FM    │    │  Deep   │              │
│   │ (二阶)   │    │ (DNN)   │              │
│   │ 共享Emb │    │ 共享Emb │              │
│   └────┬────┘    └────┬────┘              │
│        └──────┬───────┘                  │
│              ↓                           │
│         Concat → pCTR                    │
└─────────────────────────────────────────┘
```

- **改进**: 共享 Embedding, FM 显式二阶 + DNN 隐式高阶
- **优点**: 端到端, 无需人工交叉特征

### 3.3 DIN (Deep Interest Network, 阿里 2017)

```text
┌──────────────────────────────────────────────────────┐
│                DIN 架构                                │
│                                                       │
│   用户历史行为序列 [item1, item2, ..., itemN]           │
│     ↓                                                 │
│   [e1, e2, ..., eN]  (Embedding)                      │
│                                                       │
│   候选物品: e_target                                   │
│                                                       │
│   对每个历史 item: Attention Weight =                   │
│     f(e_i, e_target) = MLP(e_i ⊕ e_target ⊕ e_i*e_target)  │
│                                                       │
│   加权聚合:                                            │
│     user_repr = Σ_i weight_i * e_i                    │
│                                                       │
│   Concat [user_repr, e_target, user_profile] → MLP → pCTR
└──────────────────────────────────────────────────────┘
```

**核心公式**:
```python
def din_attention(query, keys, values):
    """
    query: 候选物品 (B, d)
    keys: 历史行为 (B, N, d)
    values: keys 本身
    """
    # 激活单元
    attn_weights = mlp(
        torch.cat([
            query.unsqueeze(1).expand_as(keys),  # (B, N, d)
            keys,
            query.unsqueeze(1) * keys  # 外积
        ], dim=-1)
    ).squeeze(-1)  # (B, N)
    # 注意: 这里不要 softmax, DIN 用原始权重
    return (attn_weights.unsqueeze(-1) * values).sum(dim=1)
```

**DIN vs Transformer Self-Attention**:
- **DIN**: target 作为 query, attend 历史 (target-aware)
- **Self-Attention**: 历史内部互相 attend
- **DIN 没有 softmax**: 用原始权重, 反映兴趣强度

### 3.4 DIEN (Deep Interest Evolution Network, 阿里 2018)

```text
┌──────────────────────────────────────────────────────┐
│                 DIEN 架构                              │
│                                                       │
│   输入: 用户历史行为序列                                │
│     ↓                                                 │
│   行为序列层: [e1, e2, ..., eN]                       │
│     ↓                                                 │
│   兴趣抽取层 (GRU): h1, h2, ..., hN                   │
│     ↓                                                 │
│   兴趣进化层 (AUGRU):                                  │
│     - 计算 attention (候选 vs h_i)                     │
│     - 用 attention 更新 GRU gate                       │
│     - 得到最终兴趣表征 h_N'                            │
│     ↓                                                 │
│   Concat [h_N', e_target, ...] → MLP → pCTR          │
└──────────────────────────────────────────────────────┘
```

**AUGRU 公式**:
```python
def augru_step(gru_cell, h_prev, x_t, attention_weight):
    # 标准 GRU
    z = sigmoid(W_z @ [h_prev, x_t])
    r = sigmoid(W_r @ [h_prev, x_t])
    h_tilde = tanh(W @ [r * h_prev, x_t])
    # 用 attention 更新
    h_t = (1 - attention_weight * z) * h_prev + attention_weight * z * h_tilde
    return h_t
```

### 3.5 DCN / DCN-v2 (Deep & Cross Network)

```text
┌─────────────────────────────────────────┐
│         DCN-v2 架构                      │
│                                          │
│   输入: x0 (Embedding 拼接)              │
│     ↓                                    │
│   Cross Network (多层):                  │
│     x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l │
│     ↓                                    │
│   Deep Network (DNN):                    │
│     h = MLP(x0)                          │
│     ↓                                    │
│   Concat [cross_outputs, h] → pCTR       │
└─────────────────────────────────────────┘
```

**显式高阶交叉**: 不像 DNN 隐式学习, DCN 显式做特征交叉

### 3.6 xDeepFM (2018)

- **改进**: 用 CIN (Compressed Interaction Network) 替代 DCN
- **向量级交叉**: DCN 是 bit-wise, CIN 是 vector-wise
- **复杂度**: O(n² × H × D), 比 DCN 高

### 3.7 深度学习模型 50 面试题

#### Q1. DIN 的 Attention 公式?
- **难度**: ⭐⭐⭐
- **答案**:
  - 激活单元: `f(e_i, e_a) = MLP(e_i ⊕ e_a ⊕ e_i ⊗ e_a)` (外积拼接)
  - 输出权重, 无 softmax
  - 聚合: `user_repr = Σ weight_i * e_i`
- **追问**: 为什么不用 softmax? → DIN 要保留兴趣强度, softmax 会归一化掉

#### Q2. DIN 和 Transformer 的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - DIN: target-aware, query 是候选物品
  - Transformer Self-Attention: 历史内部互相 attend
  - DIN 没有 softmax, 保留兴趣强度

#### Q3. DIEN 的两层 GRU 各自的作用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 兴趣抽取层 (GRU): 建模兴趣的时序演化, 输出 h_t
  - 兴趣进化层 (AUGRU): 用 attention 调制 GRU 更新门, 强化相关兴趣
- **追问**: AUGRU vs AGRU? → AUGRU 是 attention * update gate, AGRU 只用 attention 更新 hidden state

#### Q4. Wide & Deep 的 Wide 部分为什么需要人工特征?
- **难度**: ⭐⭐⭐
- **答案**:
  - Wide 是线性 LR, 没有 embedding
  - 需要显式的交叉特征 (如 `user_installed_app × app_category`)
  - 优点: 可解释, 记忆强
  - 缺点: 依赖人工经验, 不能泛化
- **追问**: 怎么改进? → DeepFM 让 Deep 部分共享 Embedding, 自动学习交叉

#### Q5. DCN 和 xDeepFM 的区别?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - DCN: bit-wise 交叉, `x_{l+1} = x0 ⊙ W_l x_l`
  - xDeepFM: vector-wise 交叉, CIN 用 Hadamard 积
  - xDeepFM 表达力更强, 但复杂度更高
- **追问**: 实际工程用哪个? → DCN-v2 较多 (简单, 效果好), xDeepFM 较少

#### Q6. 深度模型的特征交叉自动学习 vs 人工设计?
- **难度**: ⭐⭐⭐
- **答案**:
  - 人工: 可解释, 记忆强, 但依赖经验
  - 自动: 泛化强, 表达力高, 但难解释
  - 实战: 两者结合 (DeepFM 风格)
- **追问**: DLRM 用点积交互的好处? → 参数少, 计算快, 模拟二阶

#### Q7. MMoE 的跷跷板现象是什么? 怎么解决?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 多个 Expert 被多个 Tower 共享
  - 任务 A 优化时, 改了某个 Expert 的权重, 可能伤害任务 B
  - 梯度冲突, 负迁移
- **追问**: PLE 怎么解? → 任务专属 Expert 隔离, 共享 Expert 减少

#### Q8. ESMM 怎么解决 CVR 样本选择偏差?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **问题**: CVR 只在点击样本上训练, 但预测是全空间
  - **解法**: 联合训练 pCTR 和 pCTCVR, `pCVR = pCTCVR / pCTR`
  - 两者都在全样本空间训练, 解决偏差
  - 同时缓解 CVR 数据稀疏问题 (CTR 任务辅助)
- **追问**: 为什么用乘法不用除法? → 除法 pCTR 小, 数值不稳定; 乘法直接优化目标

#### Q9. 怎么选 CTR 模型?
- **难度**: ⭐⭐⭐
- **答案**:
  - **冷启动/小数据**: LR / FM / GBDT+LR
  - **大数据 + 复杂特征**: DeepFM / DCN
  - **用户行为序列**: DIN / DIEN
  - **多目标**: MMoE / PLE / ESMM
  - **GR 时代**: HSTU / OneRec / MTGR
- **追问**: 抖音用什么? → DIN + DIEN 早期, 现在部分场景用 OneRec

#### Q10. 为什么用 DNN 处理高阶特征交叉?
- **难度**: ⭐⭐⭐
- **答案**:
  - FM 只到二阶, 三阶以上参数爆炸 (O(kn³))
  - DNN 隐式学习高阶交叉, 不显式构造
  - 通过 embedding + MLP 自然拟合
- **追问**: DNN 有什么问题? → 难解释, 训练慢, 需要大量数据

#### Q11. BN vs LayerNorm 在推荐模型中?
- **难度**: ⭐⭐⭐
- **答案**:
  - BN: 批次维度归一化, CNN 友好
  - LayerNorm: 单样本归一化, Transformer/MLP 友好
  - 推荐中: 输入特征维度差异大, 用 LayerNorm 更稳定
- **追问**: MTGR 用了什么? → Group LayerNorm (按 Token 类型分组)

#### Q12. 推荐模型的 Dropout 怎么设置?
- **难度**: ⭐⭐
- **答案**:
  - Embedding 层: 0 (太大会破坏语义)
  - MLP 隐层: 0.2-0.5
  - 注意力权重: 0.1
  - 训练时打开, 推理时关闭
- **追问**: 训练和推理的区别? → model.eval() 关闭 Dropout/BN

#### Q13. 推荐模型的激活函数选择?
- **难度**: ⭐⭐
- **答案**:
  - 隐层: ReLU (默认), GeLU (Transformer), SwiGLU (LLM)
  - 输出: Sigmoid (二分类 CTR), Softmax (多分类)
  - 注意力分数: Softmax (传统), 原始 (DIN)
- **追问**: ReLU 的问题? → Dead ReLU (负数永远不更新), 用 LeakyReLU / GeLU

#### Q14. 推荐模型的位置编码怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - DIN: target-aware attention, 无显式位置编码
  - SASRec: 绝对位置 embedding
  - Transformer: Sinusoidal / RoPE / ALiBi
  - HSTU: 相对注意力偏置 (时间衰减)
- **追问**: 推荐的位置编码和 NLP 的区别? → 推荐是时间序列, 用时间差编码更自然

#### Q15. 推荐模型如何处理数值特征?
- **难度**: ⭐⭐
- **答案**:
  - **归一化**: MinMax / Z-Score
  - **分桶**: 等频 / 等距 / 决策树分桶
  - **Embedding**: 先分桶, 再过 embedding
  - **直接拼接**: 简单场景
- **追问**: 为什么要分桶? → 减少噪声, 让模型学到非线性

#### Q16. 推荐模型如何处理类别特征?
- **难度**: ⭐⭐
- **答案**:
  - **Hash**: 减少 vocab 大小, 但有冲突
  - **Embedding**: 主流, 学习稠密表示
  - **One-hot**: 维度爆炸, 不推荐
  - **Target Encoding**: 用目标统计代替类别
- **追问**: 高基数类别 (1亿物品) 怎么办? → Hash + EmbeddingBag

#### Q17. 推荐模型如何处理序列特征?
- **难度**: ⭐⭐⭐
- **答案**:
  - **Pooling**: mean/sum/max (简单)
  - **DIN**: target-aware attention
  - **DIEN**: GRU + attention
  - **SASRec/BERT4Rec**: Self-Attention
  - **HSTU**: Pointwise Aggregated Attention (统一架构)
- **追问**: 长序列怎么办? → 压缩 (TWIn/LONGER), 检索 (SIM)

#### Q18. 推荐模型如何处理多模态特征?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **CLIP/SigLIP**: 文本-图像对齐, 输出统一向量
  - **VideoCLIP**: 视频-文本对齐
  - **BERT/RoBERTa**: 文本 embedding
  - **拼接**: 多模态 embedding 拼接后送 DNN
  - **Token Mixer**: 不同模态当不同 token, 用 Transformer
- **追问**: QARM V2 怎么做? → 推理用户序列多模态对齐

#### Q19. 推荐模型的损失函数?
- **难度**: ⭐⭐⭐
- **答案**:
  - **点对点**: BCE (二分类), CE (多分类)
  - **成对**: BPR (Bayesian Personalized Ranking)
  - **列表**: ListNet, ListMLE, ApproxNDCG
  - **InfoNCE**: 对比学习, 双塔
- **追问**: GRPO Loss? → 同 prompt 采样 N 个, 用相对优势, 不需要 critic

#### Q20. 推荐模型的优化器选择?
- **难度**: ⭐⭐
- **答案**:
  - **Adam**: 默认, 适应稀疏梯度
  - **AdamW**: weight decay 分离, LLM 训练
  - **FTRL**: 在线学习, 稀疏性
  - **LAMB**: 大 batch 训练
- **追问**: 推荐系统的稀疏 embedding 用什么? → Adagrad / Sparse Adam

#### Q21. 推荐模型如何训练?
- **难度**: ⭐⭐⭐
- **答案**:
  - **离线批训练**: 全量数据, GPU 集群
  - **在线增量**: FTRL, 实时更新
  - **异步训练**: Parameter Server 架构
  - **分布式**: DDP / FSDP / DeepSpeed
- **追问**: 字节 PS 架构? → 内部使用 BytePS / ByteMLP

#### Q22. 推荐模型如何推理?
- **难度**: ⭐⭐⭐
- **答案**:
  - **TF Serving / Triton**: 工业级服务
  - **ONNX / TensorRT**: 模型加速
  - **量化**: INT8 / INT4
  - **批处理**: dynamic batching
- **追问**: GR 模型推理怎么做? → Beam Search + Prefix Tree

#### Q23. 推荐模型的蒸馏怎么做?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **Logit KD**: 教师模型 soft target 指导学生
  - **Feature KD**: 中间层特征对齐
  - **Relation KD**: 样本间关系
  - **应用**: 精排蒸馏到粗排, 大模型蒸馏到小模型
- **追问**: 温度怎么选? → 3-5, 看下游任务表现

#### Q24. 推荐模型的特征重要性怎么评估?
- **难度**: ⭐⭐⭐
- **答案**:
  - **消融实验**: 去掉某特征, 看 AUC 变化
  - **SHAP**: 解释每个样本的特征贡献
  - **树模型**: 特征重要性 (Gain/Split/Coverage)
  - **Permutation**: 打乱特征, 看效果
- **追问**: 工业界常用? → 消融 + 树模型 + 业务方反馈

#### Q25. 推荐模型的过拟合怎么判断?
- **难度**: ⭐⭐
- **答案**:
  - **训练 AUC 高, 验证 AUC 低**: 过拟合
  - **训练 Loss 下降, 验证 Loss 上升**: 过拟合
  - **Early Stopping**: 监控验证 Loss
  - **学习曲线**: 训练 vs 验证
- **追问**: 推荐系统怎么防过拟合? → Dropout, L2, Early Stopping, 数据增强

#### Q26. 推荐模型如何在线学习?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **FTRL**: 在线 LR, 稀疏性
  - **Streaming SGD**: 实时 mini-batch
  - **Bandit**: 探索利用
  - **挑战**: 漂移检测, 异常数据
- **追问**: 字节怎么做? → 内核 PS, 实时更新 embedding

#### Q27. 推荐模型的 A/B 实验怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - **流量分桶**: hash(user_id) 分桶
  - **小流量**: 1-5%, 验证
  - **中流量**: 10-30%, 放量
  - **全量**: 100%
  - **监控**: 短/中/长指标
- **追问**: 多变量? → 多层实验, 互斥分层

#### Q28. 推荐模型的效果评估指标?
- **难度**: ⭐⭐
- **答案**:
  - **离线**: AUC, GAUC, NDCG, Recall@K, Hit Rate@K
  - **在线**: CTR, CVR, 时长, 留存, GMV
  - **A/B**: 显著性检验
  - **长线**: LT7, LT30
- **追问**: 抖音用什么? → App Stay Time, LT7, 完播率, 互动率

#### Q29. 推荐模型如何冷启动?
- **难度**: ⭐⭐⭐
- **答案**:
  - **新用户**: 注册信息 + 热门 + Bandit
  - **新物品**: 内容 embedding + 探索流量
  - **新场景**: 迁移学习 + 多任务
- **追问**: 字节新视频怎么冷启动? → 内容 embedding + 相似视频 + 探索流量

#### Q30. 推荐模型如何做多任务?
- **难度**: ⭐⭐⭐
- **答案**:
  - **Shared-Bottom**: 共享底层
  - **MMoE**: 多专家 + 多门控
  - **PLE**: 共享 + 任务专属
  - **ESMM**: 概率关系建模
  - **SARNet**: 任务自适应
- **追问**: 跷跷板现象? → MMoE 有, PLE 缓解

#### Q31. 推荐模型如何做长序列?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **TWIn**: Twin Transformer, 分块降复杂度
  - **LONGER**: 局部 + 全局 attention
  - **STCA**: 稀疏 + 聚类 attention
  - **LASER**: 长期表征压缩
  - **HSTU**: Pointwise Aggregated Attention (工业界)
- **追问**: HSTU 8192 长度比 FA2 快多少? → 5.3-15.2x

#### Q32. 推荐模型如何做多场景?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **STAR**: 星形拓扑, 共享 + 场景专属
  - **PEPNet**: 参数化专家网络
  - **OxygenREC**: 场景指令 + SA-GCPO
  - **MTGR**: 混合架构保留交叉
- **追问**: 多场景如何平衡? → 场景独立 loss + 共享 loss 加权

#### Q33. 推荐模型如何做端到端?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - **OneRec**: Encoder-Decoder + 偏好对齐
  - **OneRec-V2**: Lazy Decoder-Only + 真实反馈 RL
  - **GPR**: HHD 双解码器 + MTP/VAFT/HEPO
  - **优势**: 避免误差传播, 全局最优
  - **挑战**: 计算成本高, 工程复杂
- **追问**: 字节做端到端了吗? → 部分场景 (搜索排序), 还在演进

#### Q34. 推荐模型如何做 Scaling Law?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - **HSTU**: 1.5T 参数, 在线 +12.4%, 严格 Scaling
  - **OneRec-V2**: 8B 参数, 严格遵循幂律
  - **MTGR**: 137.87 GFLOPs (160x DLRM)
  - **训练成本**: 与 DLRM 持平 (优化后)
  - **推理成本**: 比 DLRM 节省 12% (MTGR)
- **追问**: Scaling Law 在推荐上为什么成立? → 用户行为信号丰富, 类似文本

#### Q35. 推荐模型与 LLM 的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - **LLM**: 通用语言理解, 大规模预训练
  - **推荐模型**: 用户/物品表征, 多目标优化
  - **融合**: URM (LLM 作为检索器), OneRec-Think (注入推理)
- **追问**: LLM 能否替代推荐系统? → 部分场景可以 (冷启动, 长尾), 但延迟和成本是瓶颈

#### Q36. 推荐模型的 attention 复杂度?
- **难度**: ⭐⭐⭐
- **答案**:
  - Self-Attention: O(N²d), N 序列长, d 维度
  - Flash Attention: IO-aware, 实际运行快
  - Longformer: 稀疏 O(N)
  - Linear Attention: O(N), 核函数近似
  - HSTU: Pointwise Aggregated, 线性
- **追问**: 为什么推荐系统关注效率? → 工业 QPS 高, 延迟敏感

#### Q37. 推荐模型的训练加速方法?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **混合精度**: FP16/BF16, 节省显存
  - **分布式**: DDP / FSDP / DeepSpeed
  - **Gradient Checkpointing**: 重计算, 换显存
  - **算子融合**: Torch.compile, Triton
  - **稀疏更新**: 只更新出现过的 embedding
- **追问**: MTGR 怎么优化? → Fused Cutlass-based HSTU kernel, 2-3x 加速

#### Q38. 推荐模型的推理加速方法?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **量化**: INT8 / INT4 / 二值化
  - **剪枝**: 结构化 / 非结构化
  - **蒸馏**: 大模型 → 小模型
  - **KV-Cache**: 缓存历史 K/V
  - **PagedAttention**: 分页管理
  - **M-FALCON**: 微批 + 缓存 (HSTU)
- **追问**: GR 模型怎么推理? → Beam Search + Prefix Tree

#### Q39. 推荐模型如何处理 OOV (Out-Of-Vocabulary)?
- **难度**: ⭐⭐
- **答案**:
  - **Hash Embedding**: 哈希到固定桶, 新物品分到现有桶
  - **Item2Vec 用内容**: 新物品用文本 embedding
  - **冷启动策略**: 探索流量
- **追问**: 字节怎么做? → MERGE 流式物品索引

#### Q40. 推荐模型的样本不平衡怎么办?
- **难度**: ⭐⭐⭐
- **答案**:
  - **负采样**: 1:5 或 1:10
  - **Focal Loss**: 难样本加权
  - **Class Weight**: 损失加权
  - **过采样**: SMOTE
- **追问**: 召回和精排的处理? → 召回负采样, 精排曝光未点击

#### Q41. 推荐模型如何做增量训练?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **在线 FTRL**: 每条样本更新
  - **小时级**: 每小时拉新数据重训
  - **天级**: 每天全量重训
  - **挑战**: catastrophic forgetting, 用 EWC / 知识蒸馏
- **追问**: 字节怎么做? → PS 增量更新 embedding

#### Q42. 推荐模型如何做迁移学习?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **预训练 + 微调**: BERT 类
  - **多任务学习**: 共享底层
  - **跨场景迁移**: 旧场景训练, 新场景微调
  - **Meta-Learning**: MAML 快速适应
- **追问**: OneRec-Think 怎么做? → Qwen-8B 预训练 + 多任务微调

#### Q43. 推荐模型如何评估多样性?
- **难度**: ⭐⭐⭐
- **答案**:
  - **类别覆盖**: 推荐的物品类目分布
  - **ILS** (Intra-List Similarity): 列表内相似度, 越低越好
  - **NDCG@K**: 不仅看相关性, 也看多样性
  - **MMR**: 重排时直接优化相关 + 多样
- **追问**: 抖音怎么保证多样性? → 同作者/同主题打散

#### Q44. 推荐模型的因果推断怎么用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **IPS**: 逆倾向加权, 纠正位置偏差
  - **Uplift**: 因果效应, 营销
  - **DML**: Double Machine Learning
  - **Causal Tree**: 因果树模型
- **追问**: OneRec 怎么解决? → ECPO/GBPO 算法

#### Q45. 推荐模型如何处理探索 (Exploration)?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **ε-greedy**: 10% 随机探索
  - **UCB**: 基于置信上界
  - **Thompson Sampling**: 贝叶斯采样
  - **LinUCB**: Contextual Bandit
- **追问**: 抖音怎么探索? → 内容标签 + Bandit + 多样性约束

#### Q46. 推荐模型的样本构造技巧?
- **难度**: ⭐⭐⭐
- **答案**:
  - **Skip-above**: 跳过已点击同物品
  - **负采样权重**: 按热度, 提升难样本
  - **时间窗口**: 控制训练样本时间范围
  - **去重**: 用户内去重, 防穿越
- **追问**: OneRec 怎么处理? → Session-wise List Generation, 构造高质量会话

#### Q47. 推荐模型的位置偏差建模?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **位置 embedding**: 训练时加位置, 推理时去掉
  - **IPS**: 逆倾向加权, 估计无偏 AUC
  - **PAL**: Position-Aware Learning
  - **实际**: 模型学位置 → 推理时对齐
- **追问**: OneRec 怎么解? → 不再分位置, 直接生成 top-K

#### Q48. 推荐模型的特征穿越?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - **数据穿越**: 训练时用到未来信息, 评估虚高
  - **检测**: 特征时间戳, 样本时序
  - **避免**: 严格按时间切分, 不用未来特征
- **追问**: 怎么发现? → 离线涨, 线上没涨, 八成是穿越

#### Q49. 推荐模型的长尾物品如何曝光?
- **难度**: ⭐⭐⭐
- **答案**:
  - **下采样**: 热门物品负采样
  - **加权重**: 长尾物品召回分加权
  - **探索流量**: 5-10% 给长尾
  - **重排**: 多样性约束
- **追问**: 字节抖音? → 同 + 流量分成 (5% 新视频流量池)

#### Q50. 推荐模型如何与 LLM 结合?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - **URM**: LLM 作为通用检索器
  - **OneRec-Think**: LLM 显式推理
  - **ReaSeq**: CoT 多智能体
  - **LC-Rec**: LLM + SID 协同语义
  - **挑战**: 推理延迟, 工程成本
- **追问**: 字节豆包怎么用? → 部分场景辅助内容理解

---

## 4. 多任务学习

### 4.1 多任务学习为什么需要?

推荐系统通常要同时优化多个目标:
- **CTR**: 点击率
- **CVR**: 转化率 (点击后)
- **完播率**: 视频看完
- **时长**: 观看时长
- **互动**: 点赞, 评论, 分享

不同目标之间有相关性也有冲突, 多任务学习要平衡:
- **正迁移**: 一个任务提升带来另一个提升
- **负迁移**: 跷跷板现象

### 4.2 Shared-Bottom (最简单)

```text
┌────────────────────────────────────────┐
│         Shared-Bottom                   │
│                                         │
│   输入: 共享特征                         │
│     ↓                                   │
│   Shared Layer (MLP)                    │
│     ↓                                   │
│   ┌─────────┬─────────┬─────────┐       │
│   │ Tower A │ Tower B │ Tower C │       │
│   │  CTR    │  CVR    │  Time   │       │
│   └────┬────┴────┬────┴────┬────┘       │
│        ↓         ↓         ↓             │
│     pCTR       pCVR      pTime           │
└────────────────────────────────────────┘
```

**问题**: 任务冲突时, 共享层被互相拖累

### 4.3 MMoE (Multi-gate Mixture-of-Experts)

```text
┌────────────────────────────────────────┐
│              MMoE                       │
│                                         │
│   输入: x                                │
│     ↓                                   │
│   ┌─────────────────────────────────┐   │
│   │   Expert Pool                   │   │
│   │  E1, E2, E3, E4, E5, E6, E7, E8 │   │
│   └─────────────────────────────────┘   │
│       ↑    ↑    ↑                       │
│       │    │    │                       │
│   ┌────┐ ┌────┐ ┌────┐                  │
│   │Gate│ │Gate│ │Gate│ (每个任务一个)    │
│   │  A │ │  B │ │  C │                  │
│   └──┬─┘ └──┬─┘ └──┬─┘                  │
│      ↓     ↓     ↓                      │
│   TowerA TowerB TowerC                  │
└────────────────────────────────────────┘
```

**Gate 公式**:
```python
def mmoe_layer(x, experts, gates):
    """
    x: (B, d_in)
    experts: list of nn.Module, each (B, d_out)
    gates: list of nn.Module, each outputs softmax weights (B, n_experts)
    """
    expert_outs = torch.stack([e(x) for e in experts], dim=1)  # (B, n_experts, d_out)
    out = []
    for gate in gates:
        weights = gate(x)  # (B, n_experts)
        weighted = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        out.append(weighted)
    return out  # list of (B, d_out)
```

**优势**: 软共享, 缓解任务冲突
**问题**: 仍有跷跷板, 因为 Experts 共享

### 4.4 PLE (Progressive Layered Extraction)

```text
┌────────────────────────────────────────┐
│             PLE (单层)                   │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │ Shared Experts (3-4个)           │   │
│   └─────────────────────────────────┘   │
│       ↑ ↑ ↑                             │
│       │ │ │                             │
│   ┌────┐ ┌────┐ ┌────┐                  │
│   │Gate│ │Gate│ │Gate│                  │
│   │  A │ │  B │ │  C │                  │
│   └──┬─┘ └──┬─┘ └──┬─┘                  │
│      ↓     ↓     ↓                      │
│   ┌─────────────────────────────────┐   │
│   │  任务 A Experts (specific)      │   │
│   │  任务 B Experts (specific)      │   │
│   │  任务 C Experts (specific)      │   │
│   └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

**核心改进**: 任务专属 Expert + 共享 Expert, 显式分离
**渐进式**: 多层堆叠, 逐层提取

### 4.5 ESMM (Entire Space Multi-task Model)

```text
┌────────────────────────────────────────┐
│              ESMM                       │
│                                         │
│   输入: x                                │
│     ↓                                   │
│   Shared Embedding Layer                │
│     ↓                                   │
│   ┌─────────────────┐                   │
│   │   CTR Tower     │ → pCTR            │
│   └─────────────────┘                   │
│   ┌─────────────────┐                   │
│   │   CTCVR Tower   │ → pCTCVR          │
│   └─────────────────┘                   │
│                                         │
│   pCVR = pCTCVR / pCTR                  │
│                                         │
│   Loss = L(CTR) + L(CTCVR)              │
│   (用乘法避免 pCTR 小导致不稳定)        │
└────────────────────────────────────────┘
```

**创新**:
1. **全空间建模**: CVR 通过 pCTCVR / pCTR 学习, 不需要 CVR 样本
2. **解决样本选择偏差**: 不只在点击样本上训练
3. **解决数据稀疏**: CTR 任务辅助 CVR

### 4.6 多任务 30 面试题

#### Q1. MMoE 为什么有跷跷板现象?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 多个 Expert 被多个 Tower 共享
  - 任务 A 优化时, 改了某个 Expert 的权重, 可能伤害任务 B
  - 梯度冲突, 负迁移
- **追问**: PLE 怎么解? → 任务专属 Expert 隔离, 共享 Expert 减少

#### Q2. ESMM 怎么解决 CVR 样本选择偏差?
- **难度**: ⭐⭐⭐⭐
- **答案**: (见 4.5 节)

#### Q3. 多任务 Loss 怎么加权?
- **难度**: ⭐⭐⭐⭐
- **答案**: (见 Q28 of 基础)
- **追问**: ESMM 怎么加权? → L(CTR) + L(CTCVR), 默认 1:1

#### Q4. PLE 的渐进式是什么意思?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 多层 (3-4 层) 堆叠, 每层提取更抽象的特征
  - 浅层: 共享 Expert 提取通用特征
  - 深层: 任务专属 Expert 提取专有特征
- **追问**: 和 MMoE 的本质区别? → 显式任务隔离

#### Q5. 多任务的负迁移怎么发现?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 单独训练某任务 AUC 高, 加到多任务 AUC 低
  - 任务相关性低时容易出现
  - **解法**: PLE, 任务相关性度量, 软共享
- **追问**: 任务相关性怎么度量? → 训练后看预测分布的 KL / 互信息

#### Q6. 多任务学习怎么落地?
- **难度**: ⭐⭐⭐
- **答案**:
  - 选择任务: 业务相关 + 数据量足够
  - 设计架构: MMoE / PLE / ESMM
  - 调权: 离线 AUC + 在线 AB
  - 监控: 各任务指标独立看
- **追问**: 美团怎么落地? → AITM (广告), 借鉴 PLE

#### Q7. 跨场景迁移和多任务的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - **多任务**: 同一用户群, 不同任务
  - **多场景**: 不同用户群/不同上下文, 同一任务
  - **迁移**: 不同领域 (电商 → 新闻)
- **追问**: 字节怎么做? → 多场景 + 多任务 + 迁移混合用

#### Q8. 多任务学习与多标签学习的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - **多任务**: 每个任务有独立 loss, 不同输出头
  - **多标签**: 一个样本多个标签, BCE 多输出
  - 本质相同, 多任务更强调共享

#### Q9. ESMM 训练时为什么用乘法不用除法?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 除法: `pCVR = pCTCVR / pCTR`, pCTR 很小, 数值不稳定
  - 乘法: `pCTCVR = pCTR × pCVR`, 直接优化目标
  - 在线预测时再除
- **追问**: 怎么避免 pCTCVR > pCTR? → 不输出 pCTCVR, 只用 pCVR

#### Q10. MMoE 的 Gate 怎么初始化?
- **难度**: ⭐⭐
- **答案**: 默认 softmax 初始化, 均匀分布
- **追问**: 怎么防止 Gate 退化? → 加 entropy 正则

#### Q11. 多任务的 Loss 加权, 训练 Loss 应该怎么对齐?
- **难度**: ⭐⭐⭐
- **答案**: 不同任务 loss 量级差异大, 难对齐
- **解法**: 标准化到 [0, 1] 或用 log 变换
- **追问**: Uncertainty Weighting? → 学 σ, 自动平衡

#### Q12. 多任务学习可以缓解样本不平衡吗?
- **难度**: ⭐⭐⭐
- **答案**: 可以, 辅助任务提供额外监督信号
- 例子: ESMM 用 CTR 数据辅助 CVR, 缓解 CVR 稀疏

#### Q13. 多任务学习在冷启动的作用?
- **难度**: ⭐⭐⭐
- **答案**:
  - 老任务数据多, 共享 embedding 给新任务
  - 新任务快速 cold-start
  - 类似迁移学习

#### Q14. MMoE 和 PLE 的参数量对比?
- **难度**: ⭐⭐⭐
- **答案**:
  - MMoE: K experts + K gates × N tasks
  - PLE: 共享 experts + 任务专属 experts + gates
  - PLE 参数量更多, 但效果更好
- **追问**: 显存限制下怎么选? → 减少 expert 数, 用 MMoE

#### Q15. ESMM 的在线推理?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线只输出 pCVR (用于排序)
  - pCTR 和 pCTCVR 是辅助训练
  - 推理时直接用 pCVR 排序
- **追问**: 为什么 pCTCVR 不能用? → pCTCVR 是点击后转化, 排序目标是 CVR

#### Q16. 多任务如何选择 Tower 网络?
- **难度**: ⭐⭐⭐
- **答案**:
  - 默认 2-3 层 MLP, 隐层 128-256
  - 复杂场景用 DIN/Attention
  - 简单场景用 LR
- **追问**: 共享底层, Tower 一定要独立? → 是, 防止梯度冲突

#### Q17. 多任务训练 Batch Size 怎么设置?
- **难度**: ⭐⭐
- **答案**: 每个任务独立 batch, 或共享 batch
- 共享: 数据对齐, 但 batch 受限
- 独立: 灵活, 但样本不对齐

#### Q18. 多任务的样本构造?
- **难度**: ⭐⭐⭐
- **答案**:
  - 共享样本: 同时有所有任务的标签
  - 独立样本: 各任务用自己的数据
  - ESMM 用全空间样本
- **追问**: 如何处理缺失标签? → mask 掉

#### Q19. 多任务学习可以做什么任务?
- **难度**: ⭐⭐
- **答案**: CTR, CVR, 完播, 时长, 互动, 留存, GMV, 等等
- **追问**: 抖音推荐的任务? → 完播, 点赞, 评论, 转发, 关注, 时长

#### Q20. SARNet 是什么?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - SARNet: Self-supervised Auxiliary Reasoning Network
  - 用辅助自监督任务提升主任务
  - 类似 BERT 的 masked 预测

#### Q21. MMoE 的 Gate 输入是 x 还是 Expert 输出?
- **难度**: ⭐⭐⭐
- **答案**: Gate 输入是 x (原始输入), 不依赖 Expert 输出
- **追问**: 为什么? → 让 Gate 学习"哪个 Expert 对当前输入重要"

#### Q22. 多任务学习如何处理新加任务?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 冻结旧任务, 训练新 Tower
  - 渐进式: 新任务加入, 逐步解冻
  - 知识蒸馏: 旧模型指导新模型
- **追问**: 字节怎么做? → 同上, 持续迭代

#### Q23. 多任务学习在线推理延迟?
- **难度**: ⭐⭐⭐
- **答案**:
  - 多塔增加少量计算 (2-3 个 MLP)
  - 通常不影响延迟
  - 瓶颈在共享层
- **追问**: 多塔怎么并行? → 一份输入, 多塔并行

#### Q24. 多任务学习可以减少 embedding 冗余吗?
- **难度**: ⭐⭐
- **答案**: 是, 共享 embedding 减少参数量
- 例子: 用户特征共享, 多个任务都用

#### Q25. 多任务学习可以做 Ranking 吗?
- **难度**: ⭐⭐⭐
- **答案**: 是, 精排层就是多目标排序
- **追问**: 多目标排序怎么融合? → 加权 (w_CTR × pCTR + w_CVR × pCVR)

#### Q26. ESMM 假设 pCTR 和 pCVR 独立, 这个假设成立吗?
- **难度**: ⭐⭐⭐⭐
- **答案**: 不严格成立, 但实践中效果不错
- 严格建模需要更复杂模型 (因果推断)

#### Q27. 多任务学习可以加正则吗?
- **难度**: ⭐⭐
- **答案**: 是, 任务相关性正则, 防止负迁移
- **追问**: 怎么加? → 任务间 KL 散度 / 互信息正则

#### Q28. 多任务学习中如何做注意力?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - MMoE 用 Gate 作为软注意力
  - MTAN 用注意力网络动态加权
  - Attention-based Sharing
- **追问**: MTAN? → Multi-Task Attention Network

#### Q29. 多任务可以解决冷启动吗?
- **难度**: ⭐⭐⭐
- **答案**: 部分可以, 借力其他任务
- 例子: CVR 借 CTR 的 embedding

#### Q30. 多任务学习与端到端 GR 的关系?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 多任务 + GR: OneRec 多目标优化 (CTR/CVR/时长)
  - OneRec-V2: 用 RL 平衡多目标
  - ECPO/GBPO 算法
- **追问**: 字节怎么做? → OneRec 已经统一多目标

---

## 5. 序列建模与长序列

### 5.1 DIN / DIEN / DSIN

见 3.3 节

### 5.2 SASRec (Self-Attentive Sequential Recommendation)

```text
┌────────────────────────────────────────┐
│         SASRec 架构                      │
│                                         │
│   输入: 用户历史 [i1, i2, ..., iN]       │
│     ↓                                   │
│   Embedding: [e1, e2, ..., eN]          │
│     + Position Embedding                │
│     ↓                                   │
│   ┌─────────────────────────────┐      │
│   │  Multi-Head Self-Attention  │      │
│   │  (Causal Mask: 看不到未来)    │      │
│   └─────────────────────────────┘      │
│     ↓ × L 层                           │
│   Point-Wise FFN                       │
│     ↓                                   │
│   [h1, h2, ..., hN]  (各位置隐状态)     │
│     ↓                                   │
│   取最后 hN → MLP → p(next_item)       │
└────────────────────────────────────────┘
```

**核心**:
- 单向 Self-Attention (Causal Mask)
- 位置 Embedding
- 与 GPT 类似的 Decoder-only

### 5.3 BERT4Rec

```text
┌────────────────────────────────────────┐
│        BERT4Rec 架构                     │
│                                         │
│   输入: 用户历史 (部分 mask)              │
│     ↓                                   │
│   Embedding + Position                  │
│     ↓                                   │
│   ┌─────────────────────────────┐      │
│   │  Multi-Head Self-Attention  │      │
│   │  (Bidirectional, 全可见)     │      │
│   └─────────────────────────────┘      │
│     ↓ × L 层                           │
│   [h1, h2, ..., hN]                     │
│     ↓                                   │
│   Mask 位置 → p(masked_item)            │
│                                         │
│   训练: Cloze Task (类似 BERT)          │
│   推理: 末尾追加 [MASK], 预测           │
└────────────────────────────────────────┘
```

**核心**: 双向 Self-Attention + Masked Item Prediction (Cloze)

### 5.4 长序列建模

#### 5.4.1 SIM (Search-based Interest Model)

- **核心**: 用高效检索找到 Top-K 兴趣子序列
- **优势**: 不损失长序列信息
- **应用**: 阿里, 抖音

#### 5.4.2 TWIn (Twin Transformer)

- **核心**: 双塔, 一塔看全局, 一塔看局部
- **优势**: 复杂度从 O(N²) 降到 O(N)
- **应用**: 阿里

#### 5.4.3 LONGER

- **核心**: 局部 Global + Long-term 兴趣
- **优势**: 兼顾长短期兴趣

#### 5.4.4 STCA (阿里 2023)

- **核心**: Sparse + Clustering Attention
- **优势**: O(N²) → O(N log N)

#### 5.4.5 LASER (阿里 2024)

- **核心**: Long-term Aggregated SEquence Representation
- **优势**: 把长序列压缩成少量 token

#### 5.4.6 HSTU (Meta 2024) - 见第 8 章

### 5.5 序列建模 30 面试题

#### Q1. SASRec 和 BERT4Rec 的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - SASRec: Causal Mask, 单向
  - BERT4Rec: Bidirectional + Masked Prediction
  - BERT4Rec 更强但训练任务特殊
- **追问**: 工业界用哪个? → 多数用 SASRec, 简单高效

#### Q2. SASRec 怎么避免未来信息泄漏?
- **难度**: ⭐⭐⭐
- **答案**: Causal Mask (下三角), 位置 t 只能看 ≤ t 的历史
- **追问**: BERT4Rec 怎么训练? → Mask 掉中间 item, 用上下文预测

#### Q3. DIN 的 target-aware 注意力为什么有效?
- **难度**: ⭐⭐⭐
- **答案**: 不同候选物品激活不同历史行为, 兴趣表征更精确
- **追问**: 怎么改进? → DIEN 加 GRU 建模演化, DSIN 加 Session

#### Q4. DIEN 的两层 GRU 怎么配合?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 第一层: 兴趣抽取, GRU 输出 h_t
  - 第二层: AUGRU, 用 attention 调制更新门
- **追问**: 为什么需要两层? → 抽取 + 进化, 两个不同抽象

#### Q5. 长序列建模的核心挑战?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 计算复杂度: O(N²) 太大
  - 信息冗余: 用户行为大量重复
  - 兴趣演化: 长程依赖难捕捉
- **追问**: 怎么解? → SIM 检索 + HSTU 高效注意力

#### Q6. 推荐系统的序列长度一般多长?
- **难度**: ⭐⭐
- **答案**:
  - 短期: 50-100 (DIN/DIEN)
  - 中期: 100-500 (SASRec)
  - 长期: 1000+ (HSTU 8192)
  - 超长: 10000+ (研究阶段)
- **追问**: 字节抖音? → 一般 1024-4096

#### Q7. 序列建模的 attention 复杂度?
- **难度**: ⭐⭐⭐
- **答案**:
  - 标准: O(N²d)
  - Flash: IO 优化, 实际快
  - 稀疏: O(N log N) (Longformer)
  - 线性: O(N), 核函数近似
  - HSTU: O(N^0.87d) 接近线性

#### Q8. 序列建模的位置编码?
- **难度**: ⭐⭐⭐
- **答案**:
  - 绝对位置: 标准 Transformer
  - 相对位置: 时间差 (推荐更自然)
  - RoPE: 旋转, LLM 友好
  - ALiBi: 线性偏置, 外推好
  - HSTU: 相对注意力偏置
- **追问**: 推荐系统用什么? → 时间差编码 (HSTU)

#### Q9. 序列建模的负样本构造?
- **难度**: ⭐⭐
- **答案**:
  - 随机负采样
  - 曝光未点击 (精排)
  - in-batch negative (双塔)
  - Hard negative (难负样本)

#### Q10. 序列建模的冷启动?
- **难度**: ⭐⭐⭐
- **答案**:
  - 新用户: 短序列, 用热门/规则兜底
  - 新物品: 内容 embedding 初始化
  - 元学习: 快速适应

#### Q11. 序列建模的位置偏差?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 推荐中位置 = 展示位置, 有偏差
  - 处理: 训练时加位置 embedding, 推理时去掉
  - 或 IPS 校正
- **追问**: OneRec 怎么解? → 直接生成 Top-K, 没有位置概念

#### Q12. 序列建模如何处理重复行为?
- **难度**: ⭐⭐⭐
- **答案**:
  - 去重: 保留最后一次
  - 加权: 多次行为加权
  - 时间衰减: 最近权重高
- **追问**: DIN 怎么处理? → attention 自动学

#### Q13. 序列建模 vs 集合建模?
- **难度**: ⭐⭐⭐
- **答案**:
  - **集合**: 无序, mean/sum pooling
  - **序列**: 有序, 考虑先后顺序
  - 推荐: 序列更重要 (用户兴趣演化)

#### Q14. 序列建模如何并行训练?
- **难度**: ⭐⭐⭐
- **答案**:
  - Padding: 短序列 pad 到统一长度
  - Bucketing: 按长度分组, 减少 padding
  - Dynamic batching: 同 batch 相似长度
- **追问**: HSTU 怎么做? → Stochastic Length, 随机采样长度

#### Q15. 序列建模的评估?
- **难度**: ⭐⭐
- **答案**:
  - Recall@K, Hit Rate@K
  - NDCG@K
  - MRR
  - 与精排联合看整体效果

#### Q16. 序列建模与多任务结合?
- **难度**: ⭐⭐⭐
- **答案**:
  - 序列表征 + 多任务 Tower
  - 不同任务用不同序列长度
- **追问**: MTGR 怎么用? → 多序列统一编码 (行为 + 交叉特征)

#### Q17. 序列建模与强化学习结合?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - State: 序列表征
  - Action: 推荐物品
  - Reward: 点击/时长
- **追问**: OneRec 怎么用? → ECPO/GBPO

#### Q18. 序列建模可以离线训练多久?
- **难度**: ⭐⭐
- **答案**: 通常 7-30 天数据, 训练 1-7 天
- **追问**: 字节? → 看数据量, 一般 1-2 周

#### Q19. 序列建模如何评估 embedding?
- **难度**: ⭐⭐⭐
- **答案**:
  - 下游任务 AUC 提升
  - Embedding 相似度可视化 (t-SNE)
  - 物品相似度检索召回率
- **追问**: Item2Vec 怎么评估? → 人工看相似物品是否合理

#### Q20. 序列建模的 GPU 显存占用?
- **难度**: ⭐⭐⭐
- **答案**:
  - 8192 长度 × 64 dim × 4 字节 = 2 MB / 样本
  - batch=128, ~256 MB
  - 加上 attention 中间变量, GB 级
- **追问**: HSTU 怎么优化? → Pointwise Aggregated, 内存减半

#### Q21. 序列建模可以做召回吗?
- **难度**: ⭐⭐⭐
- **答案**: 可以, 但召回通常用双塔
- 序列建模更多用于精排
- **追问**: OneRec 怎么召回? → 自回归生成 SID, Beam Search

#### Q22. SASRec 的位置编码有什么问题?
- **难度**: ⭐⭐⭐
- **答案**:
  - 训练时位置 1-N, 测试时可能更长
  - 长度外推问题
- **追问**: 怎么解? → RoPE / ALiBi / HSTU 相对位置

#### Q23. DIEN 的负样本如何构造?
- **难度**: ⭐⭐
- **答案**: 同 DIN, 曝光未点击或随机负采样

#### Q24. 序列建模的样本时间窗口?
- **难度**: ⭐⭐
- **答案**: 通常 30-90 天, 看业务
- 抖音: 短期 7 天, 长期 30 天

#### Q25. 序列建模如何做长期兴趣?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 长期兴趣 token 压缩 (LASER)
  - 检索 (SIM)
  - 长期 + 短期分开建模
- **追问**: HSTU 怎么做? → 8192 长度, 统一编码

#### Q26. 序列建模的 dropout 怎么设置?
- **难度**: ⭐⭐
- **答案**:
  - Embedding: 0 (重要)
  - Attention: 0.1
  - FFN: 0.1-0.2
- **追问**: 太多了会怎样? → 训练不收敛

#### Q27. 序列建模的 layer 数?
- **难度**: ⭐⭐
- **答案**: 2-4 层 (SASRec), 6-12 层 (BERT4Rec)
- **追问**: 太深会怎样? → 过拟合, 训练慢

#### Q28. 序列建模如何预训练?
- **难度**: ⭐⭐⭐
- **答案**:
  - Masked Item Prediction (BERT4Rec)
  - Next Item Prediction (SASRec)
  - 对比学习 (Contrastive)
  - 跨域对齐 (LC-Rec)
- **追问**: 字节怎么做? → 内部用 OneRec Foundation 预训练

#### Q29. 序列建模的注意力可视化?
- **难度**: ⭐⭐
- **答案**: 把 attention 权重画出来, 看模型关注哪些历史
- **追问**: 怎么解释? → 兴趣相关行为权重高

#### Q30. 序列建模可以做生成式吗?
- **难度**: ⭐⭐⭐⭐
- **答案**: 可以, OneRec / HSTU / MTGR 都是
- **追问**: 与判别式序列建模的区别? → 直接生成下一个 item, 而非打分

---

## 6. 特征工程

### 6.1 类别特征

#### 6.1.1 处理方法

- **One-hot**: 维度爆炸, 不推荐
- **Hash**: 减少 vocab, 有冲突
- **Embedding**: 主流, 学习稠密表示
- **Target Encoding**: 用目标统计代替类别
- **EmbeddingBag**: 变长特征池化

```python
# Hash Embedding
class HashEmbedding(nn.Module):
    def __init__(self, n_hash, dim):
        super().__init__()
        self.emb = nn.Embedding(n_hash, dim)
    def forward(self, x):
        # x: 任意整数 ID, hash 到 n_hash 桶
        x_hash = (x * 2654435761) % self.n_hash
        return self.emb(x_hash)
```

### 6.2 数值特征

#### 6.2.1 处理方法

- **MinMax 归一化**: `(x - min) / (max - min)`
- **Z-Score**: `(x - μ) / σ`
- **分桶**: 等频 / 等距 / 决策树
- **对数变换**: 处理长尾 (e.g. 时长)

```python
# 分桶
def bucketize(x, n_bins=10, method='quantile'):
    if method == 'quantile':
        bins = np.quantile(x, np.linspace(0, 1, n_bins+1))
    elif method == 'uniform':
        bins = np.linspace(x.min(), x.max(), n_bins+1)
    return np.digitize(x, bins)
```

### 6.3 序列特征

- **Pooling**: mean/sum/max
- **DIN/DIEN**: attention
- **截断**: 取最近 N 个
- **多尺度**: 7天 / 30天 / 90天 分别建模

### 6.4 多模态特征

- **文本**: BERT/RoBERTa embedding
- **图像**: CLIP/SigLIP/ViT embedding
- **视频**: VideoCLIP / TimeSformer
- **音频**: Wav2Vec / Whisper

### 6.5 特征工程 30 面试题

#### Q1. 为什么推荐系统要做特征工程?
- **难度**: ⭐⭐
- **答案**:
  - 模型假设: 好的特征能弥补模型不足
  - 业务理解: 业务特征提升效果
  - 跨域知识: 迁移特征

#### Q2. 类别特征如何处理?
- **难度**: ⭐⭐
- **答案**: (见 6.1)

#### Q3. 数值特征为什么要归一化?
- **难度**: ⭐⭐
- **答案**:
  - 加速收敛
  - 防止数值特征主导
  - 让所有特征在同一尺度
- **追问**: 树模型需要吗? → 不需要, 树对尺度不敏感

#### Q4. 为什么要分桶?
- **难度**: ⭐⭐
- **答案**:
  - 减少噪声
  - 非线性建模
  - 处理长尾分布
- **追问**: 分桶数怎么选? → 10-100, 验证 AUC 决定

#### Q5. 类别基数太大怎么办?
- **难度**: ⭐⭐⭐
- **答案**:
  - Hash Embedding
  - 截断长尾 (低频物品统一桶)
  - 共享 Embedding (类似物品)
- **追问**: 字节物品 ID? → 1 亿+, 用 Hash + Item2Vec

#### Q6. 实时特征怎么用?
- **难度**: ⭐⭐⭐
- **答案**:
  - Kafka/Flink 流式计算
  - Redis 在线存储
  - TTL 控制失效时间
  - 离线/在线特征一致性
- **追问**: 抖音实时特征? → 几百维, 50ms 拉取

#### Q7. 特征穿越怎么避免?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 严格时间戳
  - 不用未来特征
  - 离线/在线特征对齐

#### Q8. 特征选择方法?
- **难度**: ⭐⭐
- **答案**: (见基础 Q40)
- **追问**: 推荐系统常用? → 树模型 + 消融 + 业务经验

#### Q9. 文本特征如何表示?
- **难度**: ⭐⭐
- **答案**:
  - TF-IDF
  - Word2Vec / Item2Vec
  - BERT embedding
  - 大模型 embedding (Qwen/DeepSeek)

#### Q10. 图像特征如何表示?
- **难度**: ⭐⭐
- **答案**:
  - CNN (ResNet/ViT)
  - CLIP (文本-图像对齐)
  - 内容 embedding 用于召回

#### Q11. 时间特征怎么用?
- **难度**: ⭐⭐
- **答案**:
  - 小时 / 周几 / 月份 / 节假日
  - 时间差 (距上次行为)
  - 周期性编码 (sin/cos)
- **追问**: 抖音用什么? → 发布时间 + 观看时间 + 时长

#### Q12. 用户画像特征?
- **难度**: ⭐⭐
- **答案**:
  - 人口属性: 年龄 / 性别 / 地域
  - 兴趣标签: 类目偏好 / 关键词
  - 行为统计: 30天活跃天数 / 平均时长
- **追问**: 实时更新? → Kafka + Flink, 秒级

#### Q13. 物品画像特征?
- **难度**: ⭐⭐
- **答案**:
  - 类目 / 品牌 / 价格 / 标签
  - 内容 embedding (文本/图像/视频)
  - 统计: 曝光 / 点击 / 转化
- **追问**: 新物品? → 内容 embedding 初始化

#### Q14. 上下文特征?
- **难度**: ⭐⭐
- **答案**:
  - 时间 / 地点 / 设备 / 网络
  - 刷新场景 / 滑动深度

#### Q15. 交叉特征?
- **难度**: ⭐⭐⭐
- **答案**:
  - 手工交叉: user_age × item_category
  - 自动交叉: FM / DCN / DeepFM
  - 隐式交叉: DNN

#### Q16. 高基数类别特征怎么训练?
- **难度**: ⭐⭐⭐
- **答案**:
  - EmbeddingBag (变长池化)
  - Hash Embedding
  - 共享 Embedding
- **追问**: 显存? → 1亿物品 * 64 维 = 25 GB, 分片 + 量化

#### Q17. 特征监控怎么做?
- **难度**: ⭐⭐⭐
- **答案**: PSI, KS, 缺失率, 值域
- **追问**: PSI 阈值? → >0.1 警告, >0.2 报警

#### Q18. 特征降维方法?
- **难度**: ⭐⭐
- **答案**:
  - PCA
  - 树模型重要性排序
  - 业务人工剔除
- **追问**: AutoML? → 自动特征工程 (研究)

#### Q19. 离线训练特征 vs 在线特征不一致怎么办?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 统一特征 SDK
  - 特征平台 (Feast, Tecton)
  - 离线用真实线上特征
- **追问**: 字节 Feature Store? → 自研, 内部叫 "特征中心"

#### Q20. 特征存储格式?
- **难度**: ⭐⭐
- **答案**:
  - Parquet (列存, 离线训练)
  - TFRecord (TF 专用)
  - Arrow (跨语言)
  - Redis Hash (在线 KV)
- **追问**: 大规模特征? → HBase + Redis 二级缓存

#### Q21. 特征归一化的标准方法?
- **难度**: ⭐⭐
- **答案**:
  - MinMax: `[0, 1]`
  - Z-Score: N(0, 1)
  - Robust: 中位数 + IQR
- **追问**: 深度模型? → BN / LN 自带归一化

#### Q22. 特征的 embedding 怎么预训练?
- **难度**: ⭐⭐⭐
- **答案**:
  - Word2Vec / Item2Vec
  - BERT
  - 跨域对齐 (CLIP)
- **追问**: 字节? → 内部预训练 Embedding 服务

#### Q23. 缺失值怎么填?
- **难度**: ⭐⭐
- **答案**:
  - 默认值 (0/-1/均值)
  - 缺失指示特征 (是否缺失)
  - 模型预测填
- **追问**: 推荐? → 默认值 + 缺失指示

#### Q24. 异常值怎么处理?
- **难度**: ⭐⭐
- **答案**:
  - 截断 (Winsorize)
  - log 变换
  - 单独分桶
- **追问**: 异常值检测? → 3σ / IQR / 业务阈值

#### Q25. 类别特征的 Target Encoding 怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - `enc(c) = mean(target | item = c)`
  - 加贝叶斯平滑, 防止过拟合
- **追问**: 泄漏风险? → 用 K-fold Target Encoding

#### Q26. 文本特征的多语言处理?
- **难度**: ⭐⭐⭐
- **答案**:
  - mBERT (100+ 语言)
  - XLM-RoBERTa
  - 翻译到统一语言
- **追问**: 字节国际版? → XLM-R + 翻译

#### Q27. 序列特征如何采样?
- **难度**: ⭐⭐⭐
- **答案**:
  - 固定长度截断
  - 随机长度 (HSTU Stochastic Length)
  - 重要性采样
- **追问**: 为什么随机长度? → 减少偏差, 提升泛化

#### Q28. 特征可视化怎么做?
- **难度**: ⭐⭐
- **答案**:
  - Embedding t-SNE / UMAP
  - 特征分布直方图
  - 注意力权重图

#### Q29. 类别特征的稀有值怎么处理?
- **难度**: ⭐⭐
- **答案**:
  - 替换为 "Other"
  - 共享 Embedding
  - 降采样
- **追问**: 阈值? → 频次 < 100 视为稀有

#### Q30. 特征工程在 GR 中还需要吗?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 部分需要: 交叉特征 (MTGR 保留)
  - 部分自动: HSTU 自动编码
  - 业务特征仍是必要
- **追问**: 字节 OneRec? → 简化特征工程, 靠模型自动

---

## 7. Semantic ID 与生成式检索

### 7.1 SID 基础 (RQ-VAE / RQ-KMeans / 层次 KMeans)

#### 7.1.1 为什么需要 SID?

传统推荐系统用原子 ID (如视频 ID), 缺点:
1. **冷启动**: 新物品没 ID embedding
2. **语义缺失**: ID 不携带语义
3. **扩展性**: 亿级物品, softmax 计算爆炸

GR 范式需要把物品表示成**离散 token 序列**, 即 SID:
- 物品 → Encoder → 向量 → VQ → Token 序列
- 序列长度 4-8, 词表 256-1024

#### 7.1.2 RQ-VAE (Residual Quantization VAE)

```text
┌──────────────────────────────────────────────┐
│            RQ-VAE 架构                        │
│                                               │
│   Item Content (文本/图像)                     │
│     ↓                                         │
│   ┌─────────┐                                 │
│   │ Encoder │                                 │
│   └────┬────┘                                 │
│        ↓                                      │
│   z0 (连续向量)                                │
│        ↓                                      │
│   ┌─────────────────────────────────┐         │
│   │ Level 1:                        │         │
│   │  c0 = argmin_k ||z0 - e_k||    │         │
│   │  residual_1 = z0 - e_{c0}      │         │
│   └────┬────────────────────────────┘         │
│        ↓                                      │
│   ┌─────────────────────────────────┐         │
│   │ Level 2:                        │         │
│   │  c1 = argmin_k ||residual_1 - e_k|| │
│   │  residual_2 = residual_1 - e_{c1} │
│   └────┬────────────────────────────┘         │
│        ↓                                      │
│   ... (3-4 层)                                │
│                                               │
│   SID = (c0, c1, c2, c3)  例: (12, 245, 67)  │
│        ↓                                      │
│   ┌─────────┐                                 │
│   │ Decoder │ (重建)                          │
│   └─────────┘                                 │
└──────────────────────────────────────────────┘
```

**核心思想**:
1. **残差量化**: 每级量化后用残差继续, 降低重建误差
2. **多级码本**: 每级独立 K-Means 聚类, 码本大小 256-1024
3. **比 VQ-VAE 精细**: 同样的 token 数量, 表达力更强

**训练**:
```python
def rq_vae_forward(encoder, decoder, codebooks, x):
    """
    codebooks: list of (K, d) tensors, 多个码本
    """
    z = encoder(x)  # (B, d)
    sids = []
    residual = z
    for codebook in codebooks:
        # 找最近邻
        distances = ((residual.unsqueeze(1) - codebook.unsqueeze(0)) ** 2).sum(-1)
        c = distances.argmin(dim=1)  # (B,)
        sids.append(c)
        # 计算残差
        quantized = codebook[c]  # (B, d)
        residual = residual - quantized
    # 重建
    z_quantized = z - residual  # 或 sum(codebook[c_l] for l)
    x_hat = decoder(z_quantized)
    return sids, x_hat
```

#### 7.1.3 RQ-KMeans (层次 KMeans)

- **简化版**: 不要 VAE, 直接对物品 embedding 做层次 KMeans
- **步骤**:
  1. KMeans 聚类 → Level 1
  2. 每类内 KMeans → Level 2
  3. 直到深度 L
- **优点**: 简单, 训练快
- **缺点**: 重建误差大, 表达力弱

#### 7.1.4 层次 KMeans

```text
Item Embedding
    ↓ KMeans (K=256)
Level 1: 256 个 cluster
    ↓ 每 cluster 内 KMeans (K=256)
Level 2: 256² = 65536 个子 cluster
    ↓
SID = (level1_id, level2_id, level3_id, level4_id)
```

### 7.2 TIGER / LC-Rec / ColaRec

#### 7.2.1 TIGER (Google 2023)

**核心**: 第一个工业级 GR 推荐系统
- **架构**: Encoder-Decoder Transformer
- **SID**: RQ-VAE 生成
- **训练**: Next Token Prediction
- **推理**: Beam Search 生成 Top-K

```text
┌────────────────────────────────────────┐
│          TIGER 架构                      │
│                                         │
│   输入: 用户历史 SID 序列                 │
│     ↓                                   │
│   ┌─────────────────────────────────┐   │
│   │   Transformer Encoder-Decoder   │   │
│   │   (类似 T5 / BART)              │   │
│   └─────────────────────────────────┘   │
│     ↓                                   │
│   生成下一个 SID 序列                     │
│     ↓                                   │
│   Beam Search                           │
│     ↓                                   │
│   Top-K 候选物品                          │
└────────────────────────────────────────┘
```

**贡献**:
1. 第一次把生成式检索用到推荐
2. 解决冷启动: 新物品可立刻被生成
3. 多样性: Beam Search 自然产生多样推荐

**实验结果**: Beauty 数据集 NDCG@5 比 SASRec +29%

#### 7.2.2 LC-Rec (ICDE 2024)

**核心**: LLM + SID 协同语义
- **问题**: LLM 的语义空间和推荐系统的协同语义有 gap
- **方案**:
  1. RQ-VAE + 均匀语义映射 (Sinkhorn) 解决 ID 冲突
  2. 多任务对齐训练 (3 层对齐)
- **对齐任务**:
  - 序列物品预测
  - 显式索引-语言对齐 (index2text, text2index)
  - 推荐导向的隐式对齐 (基于意图, 个性化偏好)

```python
# 均匀语义映射 (避免冲突)
def uniform_semantic_mapping(embeddings, n_codes, tau=0.05):
    # 转化为最优传输问题, Sinkhorn 求解
    cost = -torch.matmul(embeddings, codebook.t())  # (N, K)
    # Sinkhorn-Knopp iteration
    transport = sinkhorn(cost / tau)
    return transport.argmax(dim=1)  # 分配到 codebook
```

#### 7.2.3 ColaRec

- **核心**: Collaborative Language Model
- **改进**: 把协同信号注入 LLM
- **应用**: 进一步提升 LC-Rec

### 7.3 训练与推理

#### 7.3.1 训练

```python
def train_gr_model(model, user_history_sids, target_sids, optimizer):
    """
    user_history_sids: (B, L_h) 用户历史的 SID 序列
    target_sids: (B, L_t) 目标物品的 SID 序列
    """
    logits = model(user_history_sids, target_sids[:, :-1])  # 自回归
    loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                          target_sids[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

#### 7.3.2 推理 (Beam Search)

```python
def beam_search(model, user_history_sids, beam_size=10, max_len=4):
    beams = [(0.0, [])]  # (log_prob, sid_sequence)
    for step in range(max_len):
        all_candidates = []
        for log_prob, seq in beams:
            input_ids = user_history_sids + seq
            logits = model(input_ids)[:, -1, :]  # next token
            topk_logit, topk_idx = logits.topk(beam_size)
            for log_p, idx in zip(topk_logit, topk_idx):
                all_candidates.append((log_prob + log_p.item(), seq + [idx.item()]))
        beams = sorted(all_candidates, key=lambda x: -x[0])[:beam_size]
    return beams
```

### 7.4 SID 30 面试题

#### Q1. 什么是 Semantic ID (SID)?
- **难度**: ⭐⭐
- **答案**: 物品的离散 token 序列, 用 RQ-VAE 等方法生成, 代替原子 ID

#### Q2. 为什么要用 SID?
- **难度**: ⭐⭐⭐
- **答案**:
  - 冷启动: 新物品立刻可用
  - 语义: SID 携带内容信息
  - 词汇表: 词表小 (256⁴ = 4B), 远小于 1 亿物品
  - 可解码: 反向映射回物品

#### Q3. RQ-VAE vs VQ-VAE?
- **难度**: ⭐⭐⭐
- **答案**:
  - VQ-VAE: 单级量化, 重建误差大
  - RQ-VAE: 多级残差, 表达力强
- **追问**: 实际工程? → RQ-VAE 为主流

#### Q4. RQ-VAE 如何训练?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 编码器: 内容 (文本/图像) → 向量
  - 残差量化: 多级 VQ
  - 解码器: 重建
  - Loss = 重建 Loss + Commitment Loss
- **追问**: Commitment Loss 是什么? → 鼓励 encoder 输出靠近码本

#### Q5. 层次 KMeans vs RQ-VAE?
- **难度**: ⭐⭐⭐
- **答案**:
  - 层次 KMeans: 简单, 无 VAE, 快速
  - RQ-VAE: 端到端学习, 表达力强
- **追问**: 工业界用哪个? → RQ-KMeans (快) 或 RQ-VAE (精度)

#### Q6. SID 长度怎么选?
- **难度**: ⭐⭐⭐
- **答案**:
  - 太短 (2-3): 表达力不足, ID 冲突
  - 太长 (8+): 序列太长, 训练慢
  - 经验: 4-6 (256⁴ = 4B 物品, 256⁶ = 2.8 万亿)
- **追问**: 字节? → 4 层, 256 码本

#### Q7. SID 的 ID 冲突怎么办?
- **难度**: ⭐⭐⭐
- **答案**:
  - 增加层级 (TIGER 方法)
  - 均匀语义映射 (LC-Rec)
  - 末层用分类器 (字节)

#### Q8. SID 词表大小怎么选?
- **难度**: ⭐⭐
- **答案**:
  - 太小: 表达力不足, 冲突多
  - 太大: 训练慢, 内存大
  - 经验: 256 / 512 / 1024
- **追问**: 字节? → 256

#### Q9. SID 的重建率怎么评估?
- **难度**: ⭐⭐
- **答案**:
  - 用 SID → decoder 重建物品 embedding
  - 与原 embedding 算相似度
  - 召回率 (Hit Rate): 是否能找回原物品
- **追问**: PLUM 的重建率? → 96.7%

#### Q10. TIGER 怎么训练?
- **难度**: ⭐⭐⭐
- **答案**:
  - 离线: 用 Sentence-T5 编码内容
  - RQ-VAE 生成 SID
  - Transformer 训练 NTP
  - 在线: Beam Search 生成
- **追问**: Sentence-T5? → 文本 embedding

#### Q11. TIGER 的优势?
- **难度**: ⭐⭐
- **答案**:
  - 无需候选集
  - 冷启动: 新物品立刻可用
  - 多样性: Beam Search 产生多样结果
  - 端到端

#### Q12. LC-Rec 怎么解决协同-语言语义鸿沟?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 三层对齐任务:
    1. 序列物品预测
    2. 显式索引-语言对齐
    3. 隐式推荐对齐
- **追问**: 实验结果? → SOTA on Amazon datasets

#### Q13. OneRec 怎么用 SID?
- **难度**: ⭐⭐⭐
- **答案**:
  - 用 RQ-VAE 把快手视频编码为 SID
  - 训练: 用户行为 SID 序列 → 目标 SID
  - 推理: 自回归生成 SID, Beam Search
- **追问**: 训练数据? → Session-wise List Generation

#### Q14. SID 训练需要多少数据?
- **难度**: ⭐⭐
- **答案**:
  - RQ-VAE: 几万到几百万物品 embedding
  - GR 模型: 几十亿样本
- **追问**: 字节? → 内部全量

#### Q15. SID 的生成质量怎么评估?
- **难度**: ⭐⭐
- **答案**:
  - 重建率 (Hit Rate)
  - 唯一性 (Uniqueness): 没有重复 ID
  - 语义一致性: 相似物品 SID 前缀相似
- **追问**: PLUM 唯一性? → 96.7%

#### Q16. 为什么要约束解码 (Constrained Beam Search)?
- **难度**: ⭐⭐⭐
- **答案**:
  - 生成的可能不是合法 SID (词表外)
  - 约束到合法 SID 集合
  - 用 Trie 树加速
- **追问**: 字节怎么做? → Prefix Tree + Beam Search

#### Q17. SID 的 Vocabulary 怎么合并到 LLM?
- **难度**: ⭐⭐⭐
- **答案**:
  - LLM 原 vocab + SID 新 tokens
  - 初始化新 embedding (随机或内容平均)
  - 微调整个模型
- **追问**: OneRec-Foundation? → 1.7B/8B, 基于 Qwen3

#### Q18. GR 推理延迟?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Beam Search: 几十毫秒
  - KV-Cache: 必用, 否则更慢
  - vLLM 加速: 几毫秒
- **追问**: OneRec-V2? → 36ms (L20 GPU)

#### Q19. SID 与原子 ID 的混合使用?
- **难度**: ⭐⭐⭐
- **答案**:
  - HY-Token: SID + 类别 ID + 文本
  - 字节 MERGE: 流式物品索引
- **追问**: 字节? → MERGE 部署于搜索引擎

#### Q20. RQ-VAE 的训练 Loss?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 重建 Loss: ||x - x_hat||²
  - Commitment Loss: ||z_e(x) - sg[c]||² (encoder 接近码本)
  - Codebook Loss: ||sg[z_e(x)] - c||² (码本追 encoder)
  - Total = 重建 + α(Commitment + Codebook)
- **追问**: sg 是 stop-gradient

#### Q21. SID 在线怎么服务?
- **难度**: ⭐⭐⭐
- **答案**:
  - 物品 SID 预计算, 存 Redis
  - 在线: Beam Search 生成 SID → 查 Redis 找物品
  - 缓存: 热门 SID 放 L1
- **追问**: 字节? → 自研服务

#### Q22. 冷启动 SID 怎么处理?
- **难度**: ⭐⭐⭐
- **答案**:
  - 新物品: 用内容 embedding 实时生成 SID
  - 在线无延迟: SID 生成 < 1ms
- **追问**: 字节 MERGE? → 流式生成

#### Q23. SID 长度不固定怎么办?
- **难度**: ⭐⭐
- **答案**:
  - 通常固定 4-6 层
  - 不固定: 用 [EOS] token 终止
  - 工业界固定为主
- **追问**: 字节? → 固定 4 层

#### Q24. 怎么验证 SID 质量?
- **难度**: ⭐⭐
- **答案**:
  - 重建率: 重构物品 embedding
  - 召回率: Top-K 召回原物品
  - 多样性: 生成的 SID 是否多样
- **追问**: 离线指标 → 在线指标的 gap?

#### Q25. SID 训练需要 GPU 多久?
- **难度**: ⭐⭐
- **答案**:
  - 10万物品 RQ-VAE: 单 GPU 几小时
  - GR 模型: 几百 GPU 几天
- **追问**: 字节? → 内部有完整 pipeline

#### Q26. 多个 SID 编码器怎么选?
- **难度**: ⭐⭐⭐
- **答案**:
  - 文本: Sentence-T5 / BERT
  - 图像: CLIP / ViT
  - 多模态: CLIP / Qwen-VL
  - 行为: SASRec
- **追问**: 字节? → 自研多模态编码器

#### Q27. SID 的语义保持性?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 相似物品应该有相似 SID 前缀
  - 评估: 物品相似度 vs SID 相似度相关性
  - 训练时加语义对齐 Loss
- **追问**: PLUM 怎么做的? → 渐进式掩码

#### Q28. SID 与 item embedding 的对比?
- **难度**: ⭐⭐⭐
- **答案**:
  - item embedding: 连续稠密, 难解释
  - SID: 离散 token, 可解码, 可解释
  - SID 训练需要额外的量化步骤
- **追问**: 字节为什么用 SID? → 端到端生成, 冷启动

#### Q29. GR 模型需要 KV-Cache 吗?
- **难度**: ⭐⭐⭐
- **答案**: 是, 推理时缓存历史 K/V, 避免重算
- **追问**: vLLM 的 PagedAttention? → 分页管理 KV

#### Q30. GR 的未来方向?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 扩散生成 (阿里 LLaDA-Rec)
  - 测试时间扩展 (PROMISE)
  - 显式推理 (OneRec-Think)
  - 多模态 SID (QARM V2)

---  - 多模态 SID (QARM V2)


## 8. HSTU 与生成式推荐 (重点, 最新)

### 8.1 HSTU 架构与原理 (Meta ICML 2024)

**论文**: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
**作者**: Jiaqi Zhai 等 (Meta)
**贡献**: 1.5T 参数, 在线 A/B +12.4%, 首次验证推荐 Scaling Law

#### 8.1.1 核心思想

**问题**: 传统 DLRM 扩展性差, 不能随计算量增加提升效果

**洞察**:
- 推荐任务 = 序列转导 (Sequential Transduction)
- DLRM 异构特征可统一序列化
- 借鉴 Transformer 但改造 attention

#### 8.1.2 HSTU 架构

```text
┌──────────────────────────────────────────────┐
│              HSTU 单层                         │
│                                               │
│   Input: x ∈ R^{N×d}                          │
│     ↓                                         │
│   ┌─────────────────────────────────┐         │
│   │ Pointwise Projection             │         │
│   │ U(X) = X W_U + b_U              │         │
│   │ V(X) = X W_V + b_V              │         │
│   │ Q(X) = X W_Q + b_Q              │         │
│   │ K(X) = X W_K + b_K              │         │
│   └────┬───────────────────────┬────┘         │
│        ↓                       ↓              │
│   ┌─────────────────────────────────┐         │
│   │ Spatial Aggregation              │         │
│   │ (Pointwise Aggregated Attention) │         │
│   │ A(X)V(X) = φ(A(X)) ⊙ V(X)      │         │
│   │ A(X) = Q(X) K(X)^T / √d        │         │
│   │ φ(x) = sigmoid(x)  (NOT softmax)│         │
│   └────┬────────────────────────────┘         │
│        ↓                                      │
│   ┌─────────────────────────────────┐         │
│   │ Pointwise Transformation          │         │
│   │ Norm(A(X)V(X)) ⊙ U(X)          │         │
│   │ (Gated MLP, 元素-wise gating)    │         │
│   └────┬────────────────────────────┘         │
│        ↓                                      │
│   Output + Residual                           │
└──────────────────────────────────────────────┘
```

#### 8.1.3 与 Transformer 的关键差异

| 维度 | Transformer | HSTU |
| :--- | :--- | :--- |
| **Attention** | Softmax | Pointwise (Sigmoid) |
| **归一化方式** | 全局归一 | 逐元素 (局部) |
| **MLP 整合** | 独立 FFN | 内嵌在 Attention 后 (Gated MLP) |
| **输入处理** | Token + Position Embedding | 序列化所有特征 (无 Position) |
| **稀疏利用** | 通用 | Stochastic Length + Grouped GEMM |
| **效率** | FlashAttention2 | 5.3-15.2x faster on 8K length |

#### 8.1.4 Pointwise Aggregated Attention

```python
def pointwise_aggregated_attention(Q, K, V):
    """
    Q, K, V: (B, N, d)
    与 Transformer Self-Attention 关键差异:
    - 用 sigmoid 而非 softmax (每元素独立)
    - 不需要 mask (推荐场景)
    """
    A = Q @ K.transpose(-1, -2) / (d ** 0.5)  # (B, N, N)
    A = torch.sigmoid(A)  # 不是 softmax!
    # 归一化: 与 softmax 不同, 每个元素独立
    out = A @ V  # (B, N, d)
    return out
```

**为什么用 sigmoid 而不是 softmax?**
1. **兴趣强度**: sigmoid 保留每个元素的强度, softmax 归一化损失信息
2. **非静态词表**: softmax 假设静态词表, 推荐是流式的
3. **多任务**: 既要预测点击概率 (强度), 又要排序 (相对)

#### 8.1.5 训练任务

HSTU 设计 4 个预训练任务:
1. **Next Action Prediction**: 预测下一个用户动作 (点击/停留)
2. **Next Content Prediction**: 预测下一个内容 (item)
3. **Engagement Intensity**: 预测参与强度 (停留时长)
4. **Action Interleave**: 多动作交错预测

### 8.2 1.5T 参数规模化

#### 8.2.1 Scaling Law 验证

- **数据**: 1000 亿样本
- **计算**: 跨 3 个数量级
- **结论**: 严格遵循 LLM 的幂律 Scaling Law
  - 模型质量 ∝ (训练计算量)^α
  - α ≈ 0.05 (类似 LLM)
- **最大模型**: 8192 序列长度 × 1024 维度 × 24 层 HSTU
- **计算量**: 接近 GPT-3 / LLaMA-2

#### 8.2.2 在线结果

- **A/B 测试**: +12.4% 主指标
- **模型大小**: 1.5T 参数
- **部署**: 多 surface, 数十亿用户
- **推理吞吐**: DLRM 同等预算下 285x 模型复杂度

### 8.3 M-FALCON 推理优化

**全名**: Microbatched-Fast Attention Leveraging Cacheable OperatioNs

#### 8.3.1 核心思想

- **问题**: 排序阶段要算 m 个候选 (数千到数万), 每次 attention O(n²d)
- **解法**: 微批 + 缓存
- **复杂度**: O(bmn²d) → O(n²d)

#### 8.3.2 算法细节

```text
Input:
  - 用户序列: n
  - 候选数: m
  - 微批大小: bm

For each request:
  1. 计算用户序列编码一次 (Encoder Cache)
  2. 把候选分成 ⌈m/bm⌉ 微批
  3. 每个微批的 attention 计算复用相同的 Q
  4. KV-Cache 在 forward pass 之间共享
```

#### 8.3.3 效果

- **复杂模型**: 比 DLRM 大 285x, 推理吞吐高 1.5x-3x
- **延迟**: 1024 候选时吞吐 +1.50x, 16384 候选时 +2.99x
- **HBM 用量**: 节省 50%

### 8.4 OneRec 系列 (快手)

#### 8.4.1 OneRec-V1 (2025 Q1)

**论文**: "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment" (arXiv 2502.18965)

**贡献**:
1. 端到端 GR 替代级联架构
2. ECPO 算法 (Early Clipped GRPO)
3. IPA 策略 (Iterative Preference Alignment)

**架构**:
```text
┌────────────────────────────────────────────┐
│             OneRec-V1 架构                  │
│                                             │
│   用户多尺度特征 (序列 + 画像 + 上下文)       │
│     ↓                                       │
│   ┌─────────────────────────────┐           │
│   │   Encoder                   │           │
│   │   (多尺度 Transformer)       │           │
│   └────┬────────────────────────┘           │
│        ↓                                    │
│   用户表征                                   │
│        ↓                                    │
│   ┌─────────────────────────────┐           │
│   │   Decoder                   │           │
│   │   (自回归生成 SID 序列)      │           │
│   └────┬────────────────────────┘           │
│        ↓                                    │
│   Session-wise List Generation             │
└────────────────────────────────────────────┘
```

**训练三阶段**:
1. **监督学习 (NTP)**: Next Token Prediction 拟合历史曝光分布
2. **奖励建模 (RM)**: 个性化 P-Score 模型
3. **强化学习 (ECPO)**: 偏好对齐

**ECPO 算法**:
```python
def ecpo_loss(logits, advantages, eps_clip=0.2, delta=0.5):
    """
    Early Clipped GRPO: 对负优势样本预先裁剪
    """
    ratio = compute_importance_sampling_ratio(logits)
    # 对负优势样本, 限制最大比率
    ratio_clipped = torch.where(
        advantages < 0,
        torch.min(ratio, delta),  # 关键: 限制负样本比率
        torch.clamp(ratio, 1-eps_clip, 1+eps_clip)
    )
    loss = -(advantages * ratio_clipped).mean()
    return loss
```

**在线结果**:
- **Total Watch Time**: +1.68%
- **Average View Duration**: +6.56%
- **部署规模**: 4 亿 DAU
- **模型**: OneRec-1B

#### 8.4.2 OneRec-V2 (2025 Q3)

**论文**: "OneRec-V2: Towards a Mature Generative Recommender" (arXiv 2508.20900)

**改进**:
1. **Lazy Decoder-Only 架构**: 计算减 94%, 训练资源减 90%
2. **真实用户反馈 RL**: 用播放时长作为奖励
3. **GBPO 算法**: Gradient-Bounded Policy Optimization
4. **Scaling**: 0.5B → 8B 参数

**Lazy Decoder-Only 架构**:
```text
┌────────────────────────────────────────┐
│       Lazy Decoder-Only 架构           │
│                                         │
│   输入: 用户序列 + 上下文                 │
│     ↓                                   │
│   ┌─────────────────────────────────┐   │
│   │   Decoder-Only Transformer      │   │
│   │   (无 Encoder)                   │   │
│   │   简化 Cross-Attention          │   │
│   │   (移除 K/V projection)          │   │
│   └─────────────────────────────────┘   │
│     ↓                                   │
│   直接生成 SID 序列                       │
└────────────────────────────────────────┘
```

**关键设计**:
- 移除 Encoder, 简化 Cross-Attention
- 所有计算在 Decoder
- 8B 参数仍可扩展

**GBPO 算法**:
```python
def gbpo_loss(logits, advantages, bce_logits, bce_labels):
    """
    Gradient-Bounded Policy Optimization
    用 BCE 损失界定 RL 梯度
    """
    rl_grad = compute_policy_gradient(logits, advantages)
    bce_grad = compute_bce_gradient(bce_logits, bce_labels)
    # 用 BCE 梯度界定 RL 梯度, 防止梯度爆炸
    bounded_grad = torch.clamp(rl_grad, -bce_grad.abs(), bce_grad.abs())
    return bounded_grad
```

**时长感知奖励塑形 (Duration-Aware Reward Shaping)**:
```python
def duration_aware_reward(raw_watch_time, video_duration):
    """
    解决长视频天然倾向于累积更长的播放时长的偏差
    """
    # 完整观看 bonus
    completion = raw_watch_time / video_duration
    reward = (1 - 0.5 * completion) * raw_watch_time  # 短看完高 reward
    return reward
```

**在线结果 (与 V1 对比)**:
- **App Stay Time**: Kuaishou +0.467%, Kuaishou Lite +0.741%
- **LT7**: 显著提升
- **多目标**: 无跷跷板
- **部署**: 400M DAU, 5% 流量

#### 8.4.3 OneRec-Think (2025 Q4)

**论文**: "OneRec-Think: In-Text Reasoning for Generative Recommendation" (arXiv 2510.11639)

**核心**: 把 LLM 的显式推理能力注入 GR

**三个组件**:
1. **Itemic Alignment**: 项目语义对齐 (推荐知识 + LLM 语言)
2. **Reasoning Activation**: 推理激活 (SFT 提取推理路径)
3. **Think Ahead**: 工业部署架构

**训练任务** (4 个):
1. Next Token Prediction (基座)
2. Item-Text Alignment (跨模态对齐)
3. Reasoning Distillation (推理路径蒸馏)
4. GRPO 强化学习 (推理奖励)

**Rollout-Beam Reward**: 解决传统奖励稀疏
- 生成 N 个 rollout + M 个 beam search
- 联合评分
- 给 RL 提供密集信号

**在线结果**:
- App Stay Time: +0.159%
- 公共基准 SOTA

#### 8.4.4 OneRec-Foundation (OpenOneRec, 2025 Q4)

**论文**: "OpenOneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment" (arXiv 2512.24762)

**核心**: 开源 OneRec, 1.7B/8B 两个规模

**架构**:
- 基于 Qwen3
- Co-pretraining
- 用于推荐任务微调
- 开源推动 GR 研究

#### 8.4.5 OneMall (2026 Q1)

**论文**: "OneMall: Unified E-commerce Recommendation" (arXiv 2601.21770)

**核心**: 电商多场景统一
- 商品卡 / 短视频 / 直播
- 单一 GR 模型
- 场景指令控制

### 8.5 MTGR (美团)

**论文**: "MTGR: Meituan Generative Recommendation" (arXiv 2505.18654)

**核心**: 借鉴 HSTU 架构, 但保留 DLRM 的交叉特征

**关键发现**:
> 直接采用生成式方法放弃交叉特征会严重降低性能, 扩大规模无法弥补

#### 8.5.1 架构

```text
┌────────────────────────────────────────┐
│             MTGR 架构                   │
│                                         │
│   输入: 全部 DLRM 特征 (Token 化)        │
│     ↓                                   │
│   Group LayerNorm (按特征类型分组)       │
│     ↓                                   │
│   ┌─────────────────────────────────┐   │
│   │   HSTU 多层                    │   │
│   │   + 动态混合掩码                │   │
│   │   + 交叉特征保留                │   │
│   └─────────────────────────────────┘   │
│     ↓                                   │
│   ┌─────────────────────────────────┐   │
│   │   Pred Tower (Ranking Loss)    │   │
│   │   (类似 DLRM 精排)              │   │
│   └─────────────────────────────────┘   │
│     ↓                                   │
│   pCTR / pCVR (判别式, 不是 GR)        │
└────────────────────────────────────────┘
```

#### 8.5.2 三个关键技术

1. **Group LayerNorm**: 按 Token 类型分组 (用户/物品/上下文/交叉)
2. **动态混合掩码**: 处理不同语义空间的 Token
3. **保留交叉特征**: 沿用 DLRM 特征体系

#### 8.5.3 训练引擎

**MTGR-Training**: 基于 TorchRec 的高性能分布式训练框架
- 支持千亿参数模型
- 100+ GFLOPs/example 计算量
- Fused Cutlass-based HSTU kernel (Triton 版本 2-3x 加速)

#### 8.5.4 在线结果 (美团外卖, 2025 Q1 全量)

| 模型 | 复杂度 (GFLOPs) | 离在线效果 |
| :--- | :---: | :--- |
| DLRM base | 0.86 | 基线 |
| MTGR-small | 5+ | 显著超 DLRM |
| MTGR-middle | 20+ | 显著超 small |
| MTGR-large | 55.76 | **外卖首页订单量 +1.22%, PV_CTR +1.31%, 离线 CTCVR GAUC +2.88pp** |
| MTGR-xl (实验) | 137.87 | 最高, 推理限制未上线 |

**资源对比**:
- 训练资源: 与 DLRM 持平
- 推理资源: 比 DLRM 节省 12%

### 8.6 GPR HEPO (腾讯)

**论文**: "GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation" (arXiv 2511.10138)

**核心**: 广告场景 One-Model 范式, 部署在微信视频号广告

#### 8.6.1 三大创新

1. **统一表示 (Unified Schema)**:
   - U/O/E/I-Token 四种统一输入模式
   - RQ-KMeans+ 词元化
   - 广告 + 有机内容在同一 SID 空间

2. **异构分层解码器 HHD (Heterogeneous Hierarchical Decoder)**:
   - HSD (Heterogeneous Sequence-wise Decoder): 用户意图深度理解
   - PTD (Progressive Token-wise Decoder): "思考-精炼-生成"范式
   - HTE (Hierarchical Token-wise Evaluator): 价值估计

3. **多阶段联合训练**:
   - MTP (Multi-Token Prediction): 预训练捕获多兴趣
   - VAFT (Value-Aware Fine-Tuning): eCPM 感知
   - HEPO (Hierarchy Enhanced Policy Optimization): RL 探索

#### 8.6.2 HEPO 算法

```python
def hepo_loss(logits, advantages, hierarchy_rewards):
    """
    分层过程奖励 + Anticipatory Request Rehearsal (ARR)
    """
    policy_grad = compute_policy_gradient(logits, advantages)
    # 分层过程奖励 (层次化 advantage)
    hier_adv = compute_hierarchical_advantage(hierarchy_rewards)
    # ARR: 模拟预期请求, 防止分布偏移
    arr_term = anticipatory_request_rehearsal(logits)
    return -(policy_grad * hier_adv).mean() + arr_term
```

#### 8.6.3 在线结果 (微信视频号广告)

- **GMV**: 显著提升
- **CTCVR**: 显著提升
- **多目标**: 平衡优化

### 8.7 OxygenREC SA-GCPO (京东)

**论文**: "OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation" (arXiv 2512.22386)

**核心**: 指令跟随式 GR, 部署在京东 App 多场景

#### 8.7.1 快慢思考架构

```text
┌──────────────────────────────────────────────┐
│           OxygenREC 快慢思考架构               │
│                                               │
│   Slow Thinking (近线 LLM Pipeline)           │
│   ┌─────────────────────────────────┐         │
│   │   上下文推理指令生成             │         │
│   │   (Contextual Reasoning)        │         │
│   │   - 用户行为 + 时空上下文        │         │
│   │   - 个性化特征 + 世界知识        │         │
│   └────┬────────────────────────────┘         │
│        ↓ (KV 缓存, 零在线 LLM 调用)          │
│   ┌─────────────────────────────────┐         │
│   │   Fast Thinking                  │         │
│   │   高效 Encoder-Decoder          │         │
│   │   + 指令提示 (I_s, I_r)         │         │
│   │   + Q2I 对齐损失               │         │
│   │   + IGR 指令引导检索            │         │
│   └────┬────────────────────────────┘         │
│        ↓                                      │
│   自回归生成 SID 序列                          │
└──────────────────────────────────────────────┘
```

#### 8.7.2 SA-GCPO 算法

```python
def sa_gcpo_loss(logits, advantages, reward_scores, threshold=0.5):
    """
    Soft Adaptive Group Clip Policy Optimization
    用自适应门控函数替代硬截断
    """
    # 用真实用户反馈奖励作为阈值, 区分正负 advantage
    pos_mask = advantages > threshold
    neg_mask = advantages < -threshold
    ratio = compute_importance_sampling_ratio(logits)
    # 自适应裁剪权重
    soft_weight = adaptive_gate_function(reward_scores)
    bounded_ratio = torch.where(
        pos_mask,
        torch.clamp(ratio, 1-eps_clip, 1+eps_clip),
        ratio * soft_weight  # 自适应
    )
    return -(advantages * bounded_ratio).mean()
```

#### 8.7.3 在线结果 (京东 App 6 大场景)

| 场景 | 提升 |
| :--- | :--- |
| 首页导流 (1, 2) | GMV +4.52% ~ +8.40% |
| 频道浏览 (3, 4) | 订单量 +8.03% |
| 结算路径 (5, 6) | GMV +11.80% (强购买意图) |
| **统一 vs 独立模型** | 全面超越独立微调基线 |
| **MFU** | 40% (128× H800) |

### 8.8 TokenMixer-Large 等规模化架构 (24 篇论文系列)

#### 8.8.1 TokenMixer-Large (字节跳动)

**核心**: 大规模 Token Mixer 架构, 替代 Self-Attention
- **效率**: 8192 长度比 SA 快 2-3x
- **设计**: 基于线性注意力 + 门控

#### 8.8.2 OneTrans (字节跳动)

- **单架构多任务**: 一套 Transformer 处理召回 + 排序 + 重排
- **共享底层**: 减少工程复杂度

#### 8.8.3 STCA (阿里)

- **Sparse + Clustering Attention**
- 复杂度 O(N²) → O(N log N)
- 8192 长度可用

#### 8.8.4 EST, SORT, UniScale (字节/阿里)

- **EST** (Efficient Sequence Transformer): 字节
- **SORT** (Sparse Optimized Ranking Transformer): 字节
- **UniScale** (Unified Scaling): 字节

#### 8.8.5 HyFormer (混合架构)

- 混合 Self-Attention + Linear Attention
- 早期层用线性, 后期层用完整
- 平衡效率与表达力

#### 8.8.6 STORE / FAT / HHFT

- **STORE**: 字节, Streaming 优化
- **FAT**: 阿里, Feature-Aware Transformer
- **HHFT**: 阿里, Hybrid Hierarchical Fine-Tuning

#### 8.8.7 LFM4Ads / SMES / LASER

- **LFM4Ads**: 字节, 广告低频记忆
- **SMES**: 阿里, Sparse MoE for Sequences
- **LASER**: 阿里, 长期兴趣压缩

### 8.9 HSTU / OneRec / MTGR 50 面试题

#### Q1. HSTU 的核心创新是什么?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 1. Pointwise Aggregated Attention (Sigmoid 而非 Softmax)
  - 2. 序列转导统一 DLRM 异构特征
  - 3. 1.5T 参数验证 Scaling Law
  - 4. M-FALCON 推理加速
- **追问**: 为什么用 sigmoid? → 保留兴趣强度 + 流式词表 + 多任务

#### Q2. HSTU 与 Transformer 的关键区别?
- **难度**: ⭐⭐⭐⭐
- **答案**: (见 8.1.3 表)
- **追问**: 哪个更重要? → Sigmoid attention (突破性)

#### Q3. M-FALCON 怎么加速推理?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 微批 + 缓存, O(bmn²d) → O(n²d)
- **追问**: 能完全消除 m 的影响吗? → 不能, 但常数级别

#### Q4. OneRec 怎么替代级联架构?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 端到端 GR, 单模型输出 session
  - 不再分召回/粗排/精排/重排
  - 避免误差传播, 全局最优
- **追问**: 计算量增加多少? → 训练持平, 推理持平或更优

#### Q5. OneRec-V2 的 Lazy Decoder 优势?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 计算减 94%
  - 训练资源减 90%
  - 支持 16x 模型规模 (0.5B → 8B)
  - 严格遵循 Scaling Law
- **追问**: 怎么做到的? → 移除 Encoder, 简化 Cross-Attention

#### Q6. OneRec 的训练流程?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  1. 监督学习 (NTP): 拟合历史曝光分布
  2. 奖励建模 (RM): 学习个性化 P-Score
  3. RL (ECPO/GBPO): 偏好对齐
- **追问**: 为什么需要 RM? → 解决历史数据天花板

#### Q7. ECPO vs GRPO 的区别?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - GRPO: 标准算法, 组内归一化
  - ECPO: 对负优势样本预先裁剪, 限制比率
  - 防止负优势梯度爆炸
- **追问**: δ 参数怎么选? → 0.3-0.5, 实验决定

#### Q8. GBPO vs ECPO 的区别?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - ECPO: 用硬裁剪
  - GBPO: 用 BCE 梯度界定 RL 梯度, 更平滑
  - GBPO 解决 "策略比率 = 1 但仍梯度爆炸" 的问题
- **追问**: GBPO 优势? → 完整样本利用 + 有界梯度稳定化

#### Q9. MTGR 与 HSTU 的关键差异?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - HSTU: 端到端 GR, 放弃交叉特征
  - MTGR: 借鉴 HSTU 架构, 但保留交叉特征
  - MTGR 用判别式 loss, 不是自回归生成 SID
- **追问**: MTGR 为什么不用 GR? → 交叉特征在外卖业务关键, 扩大规模无法弥补

#### Q10. MTGR 的 Group LayerNorm 作用?
- **难度**: ⭐⭐⭐⭐
- **答案**: 按 Token 类型分组归一化
- 不同语义空间 Token 对齐
- 提升深层次训练稳定性
- **追问**: 怎么分组? → 用户/物品/上下文/交叉特征各一组

#### Q11. GPR 的 HHD 解码器?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: (见 8.6.1)
- HSD: 用户意图
- PTD: "思考-精炼-生成"
- HTE: 价值估计
- **追问**: 为什么三层? → 解耦意图理解和生成, 灵活

#### Q12. HEPO 算法?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 分层过程奖励 + Anticipatory Request Rehearsal (ARR)
- **追问**: ARR 是什么? → 模拟预期请求, 防止分布偏移

#### Q13. OxygenREC 的快慢思考?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - Slow: 近线 LLM 生成推理指令 (KV 缓存)
  - Fast: 实时 Encoder-Decoder 执行
- **追问**: 为什么这样做? → LLM 延迟太高, 但指令可以批量生成

#### Q14. SA-GCPO 与 GRPO 的区别?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - GRPO: 硬截断
  - SA-GCPO: 自适应门控函数 + 真实反馈阈值
  - 解决多任务多场景下策略学习稳定性
- **追问**: 阈值怎么定? → 用真实用户奖励分数

#### Q15. HSTU 的 Scaling Law 结论?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 模型质量 ∝ (训练计算量)^0.05, 严格遵循幂律
- **追问**: 推荐 Scaling Law 与 LLM 一样吗? → 类似, 但序列长度更重要

#### Q16. OneRec-V2 在 Kuaishou Lite 上效果?
- **难度**: ⭐⭐⭐
- **答案**: App Stay Time +0.741%
- **追问**: 主 feed? → +0.467%

#### Q17. OneRec-V2 的 GBPO 优势?
- **难度**: ⭐⭐⭐⭐
- **答案**: 保留所有样本的梯度, 不裁剪
- 用 BCE 梯度界定 RL 梯度
- 训练稳定 + 多样化探索
- **追问**: 为什么用 BCE? → BCE 梯度天然有界, 稳定

#### Q18. OneRec-Think 的 Itemic Alignment?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 推荐项目 + LLM 语言对齐
- 多任务预训练 (4 个互补任务)
- 把推荐知识注入 LLM 语言空间
- **追问**: 与 LC-Rec 区别? → 更系统, 工业部署优化

#### Q19. GPR 如何处理广告的特殊性?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - VAFT (Value-Aware): eCPM 感知
  - 多 token SID (广告+内容)
  - HTE 价值估计
- **追问**: 广告 CTR 预估怎么做? → HTE 层

#### Q20. MTGR 的训练引擎?
- **难度**: ⭐⭐⭐⭐
- **答案**: 基于 TorchRec, Fused Cutlass-based HSTU kernel
- **追问**: 为什么手写 kernel? → Triton 版本效率不够

#### Q21. HSTU 的 4 个训练任务?
- **难度**: ⭐⭐⭐⭐
- **答案**: Next Action / Next Content / Engagement Intensity / Action Interleave
- **追问**: 为什么多任务? → 推荐任务多目标

#### Q22. OneRec 的 Session-wise List Generation?
- **难度**: ⭐⭐⭐⭐
- **答案**: 训练数据用 session, 而非单个 item
- 保持推荐上下文连贯性
- point-wise 不平衡
- **追问**: 怎么构造 session? → 同一时间段连续行为

#### Q23. GPR 的统一表示 (Unified Schema)?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: U/O/E/I-Token (用户/有机/环境/物品)
- RQ-KMeans+ 词元化
- 广告 + 有机内容在同一空间
- **追问**: 为什么混合广告和有机? → 用户兴趣融合

#### Q24. HSTU vs SASRec 实验对比?
- **难度**: ⭐⭐⭐
- **答案**: HR@10 +8.6%, NDCG@10 +7.3%
- HSTU-large: HR@10 +15.5%, NDCG@10 +18.1%
- **追问**: 训练效率? → 5.3-15.2x faster than FlashAttention2

#### Q25. OneRec-V2 的 8B 模型能力?
- **难度**: ⭐⭐⭐⭐
- **答案**: 严格遵循 Scaling Law
- 0.1B → 8B 都能工作
- loss ∝ N^-α (幂律)
- **追问**: 还能更大吗? → 理论上可以, 工程挑战

#### Q26. OxygenREC 的多场景如何统一?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 场景信息 → 指令化
  - 统一奖励映射服务
  - SA-GCPO 算法对齐多业务目标
  - "train-once-deploy-everywhere"
- **追问**: 为什么能统一? → 指令机制把场景作为条件输入

#### Q27. M-FALCON 的微批大小怎么选?
- **难度**: ⭐⭐⭐⭐
- **答案**: bm 相对于 n 是小常数
- 经验: 16-64
- **追问**: 太小? → 微批过多, 调度开销大

#### Q28. OneRec-Think 怎么用 GRPO 训练?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 推理路径生成 + 物品生成
  - Rollout-Beam 奖励
  - GRPO 算法
- **追问**: 为什么不用 PPO? → 太复杂, 推理场景不需要 critic

#### Q29. HSTU 的 Stochastic Length (SL)?
- **难度**: ⭐⭐⭐⭐
- **答案**: 训练时随机采样序列长度
- 算法性增加稀疏性
- 复杂度 O(N^αd), α < 1
- **追问**: α 大约多少? → 0.87 (接近线性)

#### Q30. OneRec 的 ECPO 公式?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: (见 8.4.1 代码)
- 对负优势预先裁剪 (ratio ≤ δ)
- 对正优势标准 PPO 裁剪
- **追问**: 为什么负优势要特殊处理? → 负样本比率容易爆炸

#### Q31. MTGR-large 的 FLOPs?
- **难度**: ⭐⭐⭐
- **答案**: 55.76 GFLOPs/sample
- DLRM base: 0.86 GFLOPs/sample
- 比例: 65x
- **追问**: 实验用 137.87 GFLOPs (160x) 模型, 推理限制未上线

#### Q32. GPR 的 MTP 训练?
- **难度**: ⭐⭐⭐⭐
- **答案**: Multi-Token Prediction
- 预训练同时预测多个未来 token
- 捕获多兴趣模式
- **追问**: 与 NTP 区别? → NTP 一次一个, MTP 一次多个

#### Q33. OxygenREC 的指令引导检索 (IGR)?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 用指令作为 query
  - 从用户长期历史中检索最相关部分
  - 过滤无关历史噪声
  - 模型专注当前意图
- **追问**: 怎么确保检索质量? → Q2I 对齐损失

#### Q34. HSTU 与 OneRec 的本质区别?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - HSTU: 通用 GR 架构, 1.5T 参数, 学术研究 + 工业
  - OneRec: 端到端替代级联, 1-8B, 工业落地
  - OneRec 借鉴 HSTU 架构, 但加上 RL 对齐
- **追问**: 哪个更实用? → OneRec (已大规模落地)

#### Q35. OneRec-Think 的 Think-Ahead 架构?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - 显式推理 + 物品生成
  - 推理路径作为中间步骤
  - 推理激活 + 偏好推理
- **追问**: 为什么需要推理? → 复杂用户意图需要演绎

#### Q36. MTGR 的动态混合掩码?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 处理不同序列 (用户行为 vs 物品) 的 attention mask
- 同语义空间内 attend
- 跨语义空间隔离
- **追问**: 怎么动态? → 训练时随机切换, 提升泛化

#### Q37. GPR 的 HEPO 怎么探索?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - ARR (Anticipatory Request Rehearsal): 模拟预期请求
  - 仿真环境中探索超越日志曝光
  - 发现未充分服务的高价值候选
- **追问**: 仿真环境怎么建? → 历史数据 + CTR 模型

#### Q38. OxygenREC 的训练框架?
- **难度**: ⭐⭐⭐⭐
- **答案**: Oxygen 9N-LLM
- 融合稀疏嵌入引擎 + LLM 稠密训练
- 128× H800, 40% MFU
- **追问**: 推理框架? → xLLM (xSchedule + xAttention + xBeam)

#### Q39. HSTU 的 M-FALCON 部署?
- **难度**: ⭐⭐⭐⭐
- **答案**: Meta 多 surface 部署
- 285x 模型复杂度, 1.5-3x 吞吐
- **追问**: 推理硬件? → H100 GPU 集群

#### Q40. OneRec 的 IPA 策略?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: Iterative Preference Alignment
- 用预训练 RM 评分, 排序 sampled responses
- 选 best-chosen 和 worst-rejected
- 迭代训练
- **追问**: 为什么迭代? → 单次 DPO 效果有限

#### Q41. MTGR 的离线效果?
- **难度**: ⭐⭐⭐
- **答案**: CTCVR GAUC +2.88pp (近 2 年最大)
- 不同尺寸模型都超 DLRM
- **追问**: 训练数据? → 2024.10 之后, 半年的数据

#### Q42. GPR 的 HTE 价值估计?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: Hierarchical Token-wise Evaluator
- 整合 eCPM 价值估计
- 与生成解耦
- **追问**: eCPM 怎么算? → bid × pCTR × pCVR × 1000

#### Q43. OneRec-V2 推理延迟?
- **难度**: ⭐⭐⭐
- **答案**: 36ms (L20 GPU)
- MFU 62%
- beam size 512
- context length 3000
- **追问**: 为什么这么快? → Lazy Decoder + 优化

#### Q44. OxygenREC 的场景指令?
- **难度**: ⭐⭐⭐⭐
- **答案**: 场景信息 (首页/购物车) + 触发物品 → 场景指令
- 作为模型条件输入
- 业务规则用 prefix-constrained beam search
- **追问**: 怎么约束? → Trie 树 + 业务规则

#### Q45. HSTU 的 Gated MLP 与标准 FFN?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 标准 FFN: `W2·σ(W1·x + b1)` 独立模块
  - Gated MLP: `Norm(A(X)V(X)) ⊙ U(X)` 元素-wise gating
  - 节省参数, 增强表达力
- **追问**: 门控为什么用 sigmoid? → 稀疏激活

#### Q46. MTGR 的剪枝优化?
- **难度**: ⭐⭐⭐
- **答案**: 训练成本与 DLRM 持平
- 随机长度采样策略
- 单样本 FLOPs 提升 65x, 但成本持平
- **追问**: 怎么做到的? → 序列压缩 + kernel 优化

#### Q47. OneRec 的格式奖励?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 合法性检查 (生成合法 SID 序列)
  - 合法样本优势 = 1
  - 非法样本丢弃
- **追问**: 为什么要合法性检查? → 防止生成无效物品

#### Q48. GPR 的 eCPM 感知微调?
- **难度**: ⭐⭐⭐⭐
- **答案**: VAFT (Value-Aware Fine-Tuning)
- 对高价值物品重新加权
- 与业务优先级对齐
- **追问**: 权重怎么设? → 用历史 eCPM 分布

#### Q49. HSTU 是 generative 还是 discriminative?
- **难度**: ⭐⭐⭐⭐
- **答案**: 都可以
  - Generative: 预测下一个 item/action
  - Discriminative: 训练后冻结, 做表征
- **追问**: HSTU 论文用哪种? → Generative (ranking + retrieval 都用)

#### Q50. OneRec-V2 的模型对比 (V1 vs V2)?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  | 维度 | V1 | V2 |
  | :--- | :--- | :--- |
  | 架构 | Encoder-Decoder | Lazy Decoder-Only |
  | 计算 | 100% | 6% |
  | 训练资源 | 100% | 10% |
  | 模型规模 | 1B | 8B |
  | RL | ECPO + RM | GBPO + 真实反馈 |
- **追问**: V2 比 V1 强多少? → +0.467% ~ +0.741% App Stay Time

---

## 9. 大规模架构

### 9.1 Token Mixer vs Attention

**Token Mixer 架构**: 字节跳动 TokenMixer-Large 等

**核心思想**:
- Self-Attention O(N²d) 计算密集
- Token Mixer 用更轻量的方式混合 token 信息
- 线性注意力 (Linear Attention): 核函数近似
- 多层感知机 (MLP-Mixer): 通道混合 + 空间混合分离

### 9.2 异构 Transformer

**异构性来源**:
- 多种特征类型 (用户/物品/上下文/交叉)
- 不同序列长度
- 不同模态 (文本/图像/视频)

**异构 Transformer 设计**:
- 不同 head 处理不同特征
- 不同层用不同 attention (局部 + 全局)
- MoE 路由不同专家

### 9.3 稀疏激活 (MoE / Memory)

**MoE (Mixture of Experts)**:
- 多个专家网络
- Gate 路由激活少数专家
- 推理 FLOPs 不变, 但参数量大幅增加

**Memory-based**:
- 外部记忆存储历史信息
- Attention 读记忆

### 9.4 大规模架构 30 面试题

#### Q1. Token Mixer 是什么?
- **难度**: ⭐⭐⭐
- **答案**: 替代 Self-Attention 的 token 混合机制
- 包括: Linear Attention, MLP-Mixer, Hyena, Mamba
- **追问**: 字节 TokenMixer-Large 用了什么? → 线性注意力 + 门控

#### Q2. 异构 Transformer 怎么设计?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 不同 head 关注不同特征
  - 不同层用不同 attention 模式
  - MoE 路由不同专家
- **追问**: 推荐系统为什么要异构? → 多特征类型, 多序列长度

#### Q3. MoE 在推荐系统怎么用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - MMoE: 多专家 + 多门控
  - PLE: 共享 + 任务专属
  - 字节 MERGE: 流式物品索引
- **追问**: MoE 推理开销? → 只激活少数专家, FLOPs 持平

#### Q4. 如何选择 Attention 机制?
- **难度**: ⭐⭐⭐
- **答案**:
  - 短序列 (< 100): 标准 Self-Attention
  - 长序列 (> 1000): Flash, Linear, HSTU
  - 超长 (> 10000): RAG, 长上下文模型
- **追问**: 推荐系统一般多长? → 短期 100, 长期 8192+

#### Q5. Linear Attention 的原理?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 用核函数近似 softmax
  - `softmax(QK^T) V ≈ φ(Q) (φ(K)^T V)`
  - 复杂度 O(N) 而非 O(N²)
- **追问**: 精度损失? → 近似, 但实际差距小

#### Q6. 异构注意力机制?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 局部 (Sliding Window) + 全局 (Dilated)
  - 不同层不同 pattern
  - 字节 HyFormer 混合 SA + Linear
- **追问**: 为什么混合? → 局部模式 + 全局依赖

#### Q7. MoE 的负载均衡?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 加 auxiliary loss 让专家使用均匀
  - top-k gating 限制专家数
  - Expert Capacity 限制
- **追问**: 不均衡会怎样? → 部分专家过载, 训练崩溃

#### Q8. 推荐系统需要多深的 Transformer?
- **难度**: ⭐⭐⭐
- **答案**:
  - HSTU: 24 层
  - OneRec-V2: 类似 LLM 规模
  - 推荐浅层 (DIN, DIEN): 3-6 层
- **追问**: 太深会过拟合吗? → 数据足够时不会

#### Q9. 推荐系统需要多少注意力头?
- **难度**: ⭐⭐⭐
- **答案**:
  - 标准: 8-16
  - HSTU: 类似
  - OneRec: Qwen3 风格 (GQA)
- **追问**: 头数与表达力? → 多头捕获不同子空间

#### Q10. Flash Attention 原理?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 分块 softmax, IO-aware
  - 减少 HBM 读写
  - 理论 FLOPs 不变, 实际加速
- **追问**: 实际加速多少? → 5-10x

#### Q11. 长序列建模的稀疏性利用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Sliding Window: 只看邻近
  - Global Tokens: 少量全局 token
  - HSTU Stochastic Length: 随机长度, 增加稀疏
- **追问**: HSTU 复杂度? → O(N^0.87d)

#### Q12. 异构架构如何处理不同模态?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 多模态 Token 拼接
  - 不同模态用不同专家
  - 跨模态 attention
- **追问**: 字节 QARM V2? → 推理用户序列多模态对齐

#### Q13. MoE vs Dense 的训练差异?
- **难度**: ⭐⭐⭐
- **答案**:
  - Dense: 全部参数更新
  - MoE: 路由 + 选定专家更新
  - 训练 FLOPs: MoE 更小 (只激活部分)
- **追问**: 显存呢? → MoE 全量显存 (所有专家加载)

#### Q14. 推荐系统的特征交互架构?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - DLRM: 点积交互
  - DCN: 显式交叉
  - HSTU: 全部序列化, 让模型自己学
  - MTGR: 混合, 保留部分交叉
- **追问**: 哪种最好? → 取决于业务, 电商交叉重要

#### Q15. Token 长度对显存的影响?
- **难度**: ⭐⭐⭐
- **答案**: 8192 长度 × 64 dim × 4 字节 = 2 MB/样本
- batch=128 → 256 MB
- attention 中间变量更大
- **追问**: 如何优化? → Flash Attention, KV-Cache

#### Q16. 多任务多场景架构?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - MMoE / PLE: 多任务
  - STAR / PEPNet: 多场景
  - OxygenREC: 指令化统一
  - OneRec: 端到端统一
- **追问**: 哪种最优? → 没有银弹

#### Q17. 大模型在推荐系统怎么用?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 作为特征提取器 (Qwen-VL 多模态)
  - 作为通用检索器 (URM)
  - 作为决策器 (OneRec-Think)
- **追问**: 计算成本? → 大, 需要推理优化

#### Q18. 异构 Transformer 的并行训练?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - DDP: 同步数据并行
  - FSDP: 参数分片
  - 不同 head 独立计算, 后融合
- **追问**: 通信开销? → 大模型通信多

#### Q19. 怎么平衡 Attention 计算和 MLP 计算?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 短序列: MLP 主导
  - 长序列: Attention 主导
  - 平衡点: 序列长度 ~ 维度
- **追问**: 推荐系统序列长度? → 短期 100, 长期 8192

#### Q20. 异构 Transformer 的工程实现?
- **难度**: ⭐⭐⭐
- **答案**:
  - 不同层用不同 CUDA kernel
  - 自定义 Triton kernel
  - Fused Operator
- **追问**: HSTU 怎么实现? → Fused Cutlass kernel

#### Q21. 推荐系统如何用 Transformer 的并行性?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Sequence Parallel: 长序列切分到多卡
  - Tensor Parallel: 大矩阵切分
  - Pipeline Parallel: 层间切分
- **追问**: HSTU 怎么做? → Context Parallelism Scaling

#### Q22. 注意力机制的复杂度对比?
- **难度**: ⭐⭐⭐
- **答案**:
  | 类型 | 复杂度 |
  | :--- | :--- |
  | Self-Attention | O(N²d) |
  | Flash | O(N²d) 但 IO 优化 |
  | Linear | O(Nd) |
  | Sparse (Longformer) | O(N√N) |
  | HSTU Pointwise | O(N^0.87d) |
- **追问**: 哪种最快? → Linear, 但精度略差

#### Q23. 多模态 Transformer 的挑战?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 不同模态的 token 化
  - 跨模态对齐 (CLIP)
  - 模态缺失处理
- **追问**: 推荐系统多模态? → 视频/图像/文本 SID 统一

#### Q24. 怎么评估大规模架构的效果?
- **难度**: ⭐⭐⭐
- **答案**:
  - 离线 AUC, NDCG
  - 在线 AB 指标
  - 推理延迟 (QPS)
  - 训练成本 (GPU 小时)
- **追问**: Scaling Law? → 看 loss vs 计算量曲线

#### Q25. 大模型架构如何选择?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 业务规模: 小 → 简单架构
  - 延迟要求: 高 → 优化 attention
  - 多模态: 异构融合
  - 多任务: MoE
- **追问**: 推荐系统推荐? → HSTU (Meta) / OneRec (快手)

#### Q26. 推荐系统的混合架构?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - MTGR: HSTU 架构 + 交叉特征
  - LUM (阿里): 预训练 + 微调
  - HLLM (字节): Item/User 双塔 LLM
- **追问**: 哪种效果最好? → 取决于业务, 阿里 LUM 保留交叉

#### Q27. 注意力机制的内存优化?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Flash Attention: IO 优化
  - KV-Cache: 缓存历史
  - PagedAttention (vLLM): 分页管理
  - M-FALCON: 微批 + 缓存
- **追问**: PagedAttention 优势? → 减少碎片, 提升 batch 吞吐 10-20x

#### Q28. 推荐系统的 GPU 利用率?
- **难度**: ⭐⭐⭐
- **答案**:
  - 经典 DLRM: 30-50% (稀疏 + IO)
  - HSTU: 50-70%
  - OneRec-V2: 62% (L20 GPU)
- **追问**: 为什么 DLRM 利用率低? → Embedding 查表是 IO 密集

#### Q29. 推荐系统的训练加速技巧?
- **难度**: ⭐⭐⭐
- **答案**:
  - 混合精度 (BF16/FP16)
  - 梯度累积
  - 分布式训练
  - Embedding 分片
- **追问**: 字节怎么做的? → 自研训练框架

#### Q30. 推荐系统如何支持亿级物品?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Embedding 分片
  - 量化 (INT8/INT4)
  - 哈希碰撞
  - SID 离散化
  - 稀疏更新
- **追问**: SID 后呢? → 词表小, 可用 LLM 训练

---

## 10. Embedding 与向量检索

### 10.1 Item2Vec / Word2Vec

**原理**: 把用户行为序列当作"句子", 物品当作"词", 用 Skip-gram 训练
- 行为相似的物品 embedding 接近
- 用于 i2i 召回

```python
class Item2Vec(nn.Module):
    def __init__(self, n_items, dim):
        super().__init__()
        self.in_emb = nn.Embedding(n_items, dim)
        self.out_emb = nn.Embedding(n_items, dim)

    def forward(self, center_item, context_items, neg_items):
        # Skip-gram + Negative Sampling
        center = self.in_emb(center_item)  # (B, dim)
        pos = self.out_emb(context_items)  # (B, K, dim)
        neg = self.out_emb(neg_items)      # (B, K_neg, dim)
        pos_score = (center.unsqueeze(1) * pos).sum(-1)
        neg_score = (center.unsqueeze(1) * neg).sum(-1)
        loss = -torch.log(torch.sigmoid(pos_score)).mean() \
               -torch.log(torch.sigmoid(-neg_score)).mean()
        return loss
```

### 10.2 向量检索 (Faiss / Milvus / HNSW)

#### 10.2.1 ANN 算法对比

| 算法 | 原理 | 复杂度 | 适用规模 |
| :--- | :--- | :--- | :--- |
| Flat | 暴力遍历 | O(N) | < 100万 |
| IVF | 倒排索引 | O(N/nlist) | 100万-1亿 |
| HNSW | 图索引 | O(log N) | 100万-10亿 |
| PQ | 乘积量化 | O(m) | 内存受限 |
| IVF-PQ | IVF + PQ | O(N/nlist) | 亿级, 工业主流 |

#### 10.2.2 HNSW

```python
# Faiss HNSW
index = faiss.IndexHNSWFlat(dim, 32)  # 32 = M (每节点连接数)
index.hnsw.efConstruction = 40
index.add(vectors)
index.hnsw.efSearch = 16  # 查询时探索数
distances, indices = index.search(query, k=10)
```

#### 10.2.3 IVF-PQ

```python
# Faiss IVF-PQ
nlist = 4096   # 簇数
m = 8          # PQ 段数
nbits = 8      # 每段 bits
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
index.train(vectors)
index.add(vectors)
index.nprobe = 64  # 查询簇数
distances, indices = index.search(query, k=10)
```

### 10.3 Embedding 30 面试题

#### Q1. Item2Vec 和 Word2Vec 的区别?
- **难度**: ⭐⭐
- **答案**: 几乎一样, Item2Vec 是把"物品"当作"词", 训练 Skip-gram
- **追问**: 行为序列怎么定义? → 用户连续点击的物品序列

#### Q2. 推荐系统为何需要 ANN?
- **难度**: ⭐⭐
- **答案**:
  - 物品库百万到十亿级, 精确 KNN O(N) 不可行
  - ANN 牺牲精度换速度
  - 召回阶段对精度容忍度高
- **追问**: 工业级常用? → Faiss IVF-PQ + HNSW 混合

#### Q3. HNSW 的原理?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Hierarchical Navigable Small World
  - 多层图, 每层是 NSW (小世界)
  - 上层稀疏, 快速定位
  - 底层稠密, 精确搜索
- **追问**: 为什么快? → 跳表式定位 + 局部搜索

#### Q4. IVF-PQ 的内存优化?
- **难度**: ⭐⭐⭐⭐
- **答案**: PQ 把向量切成 m 段, 每段用 256 个聚类中心
- 压缩比: 32 字节 → m 字节
- 例: m=8, 压缩 4x
- **追问**: 精度损失? → 重建误差 < 1%

#### Q5. 向量归一化 (L2 Norm) 的影响?
- **难度**: ⭐⭐⭐
- **答案**:
  - L2 归一化后, 内积 = 余弦相似度
  - 不归一化, 内积与向量范数有关
  - 大多数工业系统归一化
- **追问**: 何时不归一化? → 当向量范数有意义时 (e.g. 置信度)

#### Q6. Embedding 维度怎么选?
- **难度**: ⭐⭐⭐
- **答案**: (见基础 Q4)
- **追问**: 维度翻倍召回效果一定好? → 不一定, 边际效益递减

#### Q7. 向量检索的 GPU 加速?
- **难度**: ⭐⭐⭐
- **答案**:
  - Faiss GPU Index
  - Milvus GPU 模式
  - ScaNN (Google)
- **追问**: 工业部署? → Faiss CPU, 简单稳定

#### Q8. Embedding 衰减 (embedding decay)?
- **难度**: ⭐⭐⭐
- **答案**:
  - 训练好的 embedding 随时间漂移
  - 监控 PSI, 重训或增量更新
- **追问**: 字节? → 增量更新

#### Q9. Embedding 共享策略?
- **难度**: ⭐⭐
- **答案**:
  - 多特征共享 (用户画像 vs 行为)
  - 多场景共享 (不同业务)
  - 减少参数量
- **追问**: 共享有什么风险? → 一个任务破坏共享, 影响其他任务

#### Q10. 双塔召回的负样本构造?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 全局随机负采样
  - In-batch Negatives (其他样本的正样本)
  - Hard Negative (相似但未点击)
- **追问**: 温度系数? → 0.05-0.1, 影响分布锐度

#### Q11. 向量检索的评估指标?
- **难度**: ⭐⭐
- **答案**:
  - Recall@K: Top-K 包含真实正例比例
  - QPS: 每秒查询数
  - 延迟 (P50/P99)
- **追问**: 工业级 Recall? → > 95%

#### Q12. Embedding 如何做 AB 测试?
- **难度**: ⭐⭐⭐
- **答案**:
  - 双塔模型对比 (旧 vs 新)
  - 召回率 + CTR 双指标
  - 小流量验证
- **追问**: 怎么确保公平? → 同一流量桶

#### Q13. Embedding 量化方法?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - INT8: 4x 压缩, 精度损失 < 1%
  - INT4: 8x 压缩, 精度损失 ~2%
  - PQ: 4-32x 压缩
- **追问**: GPTQ vs AWQ? → LLM 量化常用, 推荐不常用

#### Q14. 向量检索的工业实践?
- **难度**: ⭐⭐⭐
- **答案**:
  - Faiss IVF-PQ 主力
  - HNSW 辅助高精度场景
  - Milvus 分布式
  - 字节自研 Vearch
- **追问**: 字节? → Vearch + 自研

#### Q15. Embedding 冷启动?
- **难度**: ⭐⭐⭐
- **答案**:
  - 新物品: 用内容 embedding 初始化
  - 新用户: 用注册信息 + 行为 embedding 平均
- **追问**: 元学习? → MAML, 快速适应

#### Q16. Embedding 蒸馏?
- **难度**: ⭐⭐⭐
- **答案**:
  - 教师模型 → 学生模型
  - 蒸馏后维度/参数量降低
  - 适合边缘部署
- **追问**: 推荐系统? → 精排蒸馏到粗排

#### Q17. 双塔模型的距离度量?
- **难度**: ⭐⭐⭐
- **答案**:
  - Cosine: 归一化后内积
  - Inner Product: 直接内积
  - L2: 欧氏距离
- **追问**: 推荐系统用哪种? → Inner Product (效率高)

#### Q18. 向量召回与文本检索的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - 文本检索: BM25 + 向量混合
  - 推荐: 主要向量检索
  - 共同点: ANN 算法
- **追问**: 字节搜索? → BM25 + 向量混合 (Farewell to Item IDs)

#### Q19. Embedding 监控怎么做?
- **难度**: ⭐⭐⭐
- **答案**:
  - PSI 分布漂移
  - 召回率监控
  - 异常检测
- **追问**: PSI 阈值? → >0.1 警告, >0.2 报警

#### Q20. Item2Vec 的训练数据怎么构造?
- **难度**: ⭐⭐
- **答案**:
  - 用户连续点击行为 → 序列
  - 滑动窗口构造 (center, context)
  - 负采样 (热度)
- **追问**: 时间窗口多长? → 7-30 天

#### Q21. 多模态 Embedding 怎么融合?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 文本/图像各自编码
  - 投影到统一空间 (CLIP)
  - 拼接送 DNN
- **追问**: 字节? → QARM V2, 推理用户序列多模态对齐

#### Q22. Embedding Bag 是什么?
- **难度**: ⭐⭐⭐
- **答案**: PyTorch 的 EmbeddingBag, 处理变长稀疏特征
- 支持 mean / sum / max 池化
- **追问**: vs Embedding? → EmbeddingBag 直接池化, 减少维度

#### Q23. 向量检索的瓶颈?
- **难度**: ⭐⭐⭐
- **答案**:
  - 内存: 亿级向量, GB-TB 级
  - IO: 大数据吞吐
  - 延迟: QPS 要求
- **追问**: 怎么优化? → 量化 + 分片 + GPU

#### Q24. Embedding 预训练 vs 端到端训练?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 预训练: Item2Vec, BERT
  - 端到端: 与主任务一起训练
  - 预训练快但可能错位; 端到端精确但慢
- **追问**: 推荐系统? → 多数端到端

#### Q25. Item2Vec 训练的 batch 设置?
- **难度**: ⭐⭐
- **答案**: 中心词 batch + 上下文 batch + 负采样 batch
- batch_size = 1024-8192
- **追问**: 训练多久? → 亿级样本几小时

#### Q26. 向量召回的索引怎么更新?
- **难度**: ⭐⭐⭐
- **答案**:
  - 全量重建 (天级)
  - 增量更新 (小时级)
  - 在线 embedding + Redis 缓存
- **追问**: 字节? → Vearch 流式

#### Q27. Embedding 相似度可视化?
- **难度**: ⭐⭐
- **答案**: t-SNE / UMAP 降维
- 聚类检查
- **追问**: 怎么解释? → 看相似物品是否合理

#### Q28. 物品 ID Embedding 怎么更新?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线训练 (FTRL)
  - 离线增量 (小时/天)
  - 全量重训 (周/月)
- **追问**: 在线更新风险? → 漂移, 监控

#### Q29. 向量检索 + 规则召回融合?
- **难度**: ⭐⭐⭐
- **答案**:
  - 各路召回取 TopN
  - 加权融合分数
  - 粗排再统一排序
- **追问**: 权重怎么定? → 业务经验 + AB

#### Q30. Embedding 服务的 QPS 要求?
- **难度**: ⭐⭐
- **答案**:
  - 字节: 数十万 QPS
  - 美团: 十万级
  - 中小厂: 千级
- **追问**: 怎么保证? → 缓存 + 分片 + GPU

---

## 11. 在线学习与增量训练

### 11.1 FTRL

**全称**: Follow The Regularized Leader

**核心**: 在线 LR + L1 正则 + L2 正则, 保持稀疏性

```python
# FTRL 更新
for sample in stream:
    grad = compute_gradient(sample)
    z += grad
    n += grad ** 2
    sigma = (sqrt(n) - sqrt(n_prev)) / alpha
    w = argmin(w^T z + (1/2) * sigma * ||w||² + λ1 * ||w||_1)
```

**优点**: 稀疏, 适合在线学习
**缺点**: 单步更新, 不易并行

### 11.2 增量更新

**方式**:
- **小时级**: 拉新数据, 增量训练
- **天级**: 全量重训
- **周级**: 全量重训 + 微调

**挑战**: catastrophic forgetting
- EWC (Elastic Weight Consolidation)
- 知识蒸馏
- LoRA 微调

### 11.3 在线学习 30 面试题

#### Q1. 在线学习的优势?
- **难度**: ⭐⭐
- **答案**:
  - 实时适应用户行为变化
  - 解决数据漂移
  - 不需要全量重训
- **追问**: 工业界为什么不用? → 工程复杂, 不易并行

#### Q2. FTRL 与 SGD 的区别?
- **难度**: ⭐⭐⭐
- **答案**:
  - SGD: 标准梯度下降
  - FTRL: 在线 LR + L1 稀疏
  - FTRL 保持稀疏 embedding
- **追问**: 工业界为什么用 FTRL? → 在线学习 + 稀疏性

#### Q3. 在线学习的工程挑战?
- **难度**: ⭐⭐⭐
- **答案**:
  - 流式数据处理 (Kafka)
  - 在线特征拉取 (Redis)
  - 实时模型服务
  - 异常检测
- **追问**: 字节? → 内部 PS 系统

#### Q4. 增量训练 vs 在线学习?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线学习: 每条样本更新
  - 增量训练: 批次数据更新
  - 在线更实时, 但工程复杂
- **追问**: 推荐系统主流? → 增量训练 (小时/天)

#### Q5. 如何解决 catastrophic forgetting?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - EWC: 重要参数加约束
  - 知识蒸馏: 旧模型指导
  - 经验回放: 保留旧数据
  - LoRA: 低秩微调
- **追问**: 推荐系统? → 知识蒸馏 + 经验回放

#### Q6. FTRL 的稀疏性原理?
- **难度**: ⭐⭐⭐⭐
- **答案**: L1 正则化在梯度更新中产生精确 0
- 与 SGD 不同, SGD 不会精确 0
- **追问**: 为什么推荐系统需要稀疏? → 亿级 embedding, 节省内存

#### Q7. 在线学习如何做 AB 测试?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线模型 vs 离线模型, 同一流量桶
  - 监控核心指标差异
  - 异常告警
- **追问**: 风险? → 在线异常会立刻影响所有用户

#### Q8. 增量训练的数据量怎么定?
- **难度**: ⭐⭐
- **答案**:
  - 天级: 几亿到几十亿样本
  - 小时级: 几千万
  - 看业务规模
- **追问**: 字节? → 内部 PB 级

#### Q9. 在线学习与 A/B 实验的关系?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线学习持续优化模型
  - A/B 验证新模型 vs 旧模型
  - 通过验证后切流量
- **追问**: 字节? → AB 平台 + 在线学习集成

#### Q10. 在线学习如何监控漂移?
- **难度**: ⭐⭐⭐
- **答案**:
  - 特征分布 (PSI)
  - 预测分布
  - 在线指标 (CTR)
  - 异常检测
- **追问**: 漂移阈值? → PSI > 0.2 报警

#### Q11. 在线 LR vs 离线 LR?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线 LR: FTRL, 单步更新
  - 离线 LR: 批量梯度下降
  - 在线实时, 离线稳定
- **追问**: 推荐系统? → 在线 FTRL (稀疏) + 离线深度

#### Q12. 在线学习对推荐多样性的影响?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线学习可能过拟合热点
  - 需要约束 (正则化, 探索流量)
- **追问**: 怎么平衡? → 多样性约束

#### Q13. 在线学习的回滚机制?
- **难度**: ⭐⭐⭐
- **答案**:
  - 保存模型快照 (天级)
  - 监控发现异常, 回滚到上一版本
- **追问**: 字节? → 蓝绿发布

#### Q14. 在线学习如何处理冷启动?
- **难度**: ⭐⭐⭐
- **答案**:
  - 探索流量给新物品/用户
  - 在线学习快速反映新行为
- **追问**: 需要多久? → 分钟级

#### Q15. 在线 FTRL 的工程实现?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 参数服务器 (PS) 架构
  - Worker 计算梯度
  - Server 聚合 + FTRL 更新
- **追问**: 字节? → BytePS, 自研

#### Q16. 在线学习对召回的影响?
- **难度**: ⭐⭐⭐
- **答案**:
  - 召回需要重新建索引
  - 在线索引更新 (流式)
- **追问**: 字节? → Vearch 流式索引

#### Q17. 在线学习与冷启动的关系?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线学习对新物品友好 (立即反映)
  - 但需要探索流量配合
- **追问**: 探索流量比例? → 5-10%

#### Q18. 增量训练需要重置优化器状态吗?
- **难度**: ⭐⭐⭐
- **答案**:
  - Adam 的 1st/2nd moment 需要重置
  - 否则可能不收敛
- **追问**: 字节? → 增量训练时 warm restart

#### Q19. 在线学习如何与深度模型结合?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - Embedding 层在线更新 (FTRL)
  - DNN 离线训练
  - 混合架构
- **追问**: 字节? → 内部 PS + DNN 混合

#### Q20. 在线学习可以做多目标吗?
- **难度**: ⭐⭐⭐
- **答案**: 是, 各任务独立在线更新
- 但任务冲突需要平衡
- **追问**: 字节? → 多任务 PS

#### Q21. 在线学习的 QPS 要求?
- **难度**: ⭐⭐
- **答案**: 字节级别, 几十亿次/天
- 单参数更新 QPS
- **追问**: 工程? → PS 高并发

#### Q22. 在线学习的特征实时性?
- **难度**: ⭐⭐⭐
- **答案**:
  - 实时特征 (毫秒级)
  - 准实时 (秒级)
  - 离线特征 (天级)
- **追问**: 推荐系统? → 实时 + 准实时 + 离线混合

#### Q23. 在线学习如何评估效果?
- **难度**: ⭐⭐⭐
- **答案**:
  - 短期: 实时 CTR/CVR
  - 中期: 时长/互动
  - 长期: 留存/GMV
- **追问**: 评估周期? → 7-30 天

#### Q24. FTRL 的 α 参数?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - α: 学习率缩放
  - λ1: L1 正则
  - λ2: L2 正则
- **追问**: 怎么调? → 网格搜索 + 业务经验

#### Q25. 在线学习与离线训练的切换?
- **难度**: ⭐⭐⭐
- **答案**:
  - 在线发现问题, 切离线训练
  - 离线训练验证, 切在线
- **追问**: 字节? → 双轨, 互为 backup

#### Q26. 在线学习可以做序列模型吗?
- **难度**: ⭐⭐⭐⭐
- **答案**: 困难, 序列模型需要历史
- 实际: 用 streaming GRU 简化
- **追问**: 字节? → DIN 在线简化版

#### Q27. 在线学习与端到端 GR 的关系?
- **难度**: ⭐⭐⭐⭐⭐
- **答案**:
  - OneRec-V2 用真实用户反馈 RL (近线)
  - 不是真正的在线学习, 是 batch 反馈
  - 在线 LLM 生成 SID
- **追问**: 未来? → 在线 LLM + 在线 GR

#### Q28. 在线学习的内存占用?
- **难度**: ⭐⭐⭐
- **答案**:
  - Embedding: 主内存
  - 优化器状态 (Adam): 8x
  - 模型: 主要
- **追问**: 字节? → PS 分片

#### Q29. 增量训练的 batch size?
- **难度**: ⭐⭐
- **答案**: 通常与全量相同
- 但 learning rate 调整
- **追问**: 字节? → 根据增量数据规模

#### Q30. 在线学习的未来方向?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 在线 LLM (模型自更新)
  - 在线 GR (流式生成)
  - 在线强化学习
- **追问**: 字节 OneRec? → 已经在做

---

## 12. A/B 测试与评估

### 12.1 A/B 实验基础

**核心**: 同层用户分桶, 实验组 vs 对照组, 比较核心指标

```python
# 分桶
def assign_bucket(user_id, n_buckets=10000):
    return hash(user_id) % n_buckets

# 实验配置
control_buckets = list(range(0, 5000))    # 50% 对照
experiment_buckets = list(range(5000, 10000))  # 50% 实验

# 指标监控
def monitor_metrics(experiment_name):
    ctr_exp = get_ctr(experiment_buckets)
    ctr_ctrl = get_ctr(control_buckets)
    p_value = ttest(ctr_exp, ctr_ctrl)
    return ctr_exp - ctr_ctrl, p_value
```

### 12.2 显著性检验

- **t 检验**: 连续指标
- **卡方检验**: 离散指标
- **序检验**: 多层实验
- **CUPED**: 方差缩减

### 12.3 A/B 测试 20 面试题

#### Q1. A/B 测试的流程?
- **难度**: ⭐⭐
- **答案**:
  1. 假设 (H0/H1)
  2. 确定指标 (OEC + 护栏)
  3. 流量分桶 (实验 vs 对照)
  4. 小流量验证 (1-5%)
  5. 显著性检验
  6. 全量上线
- **追问**: 显著性检验? → t 检验, p < 0.05

#### Q2. 流量分桶的方法?
- **难度**: ⭐⭐
- **答案**:
  - hash(user_id) % 10000
  - 用户稳定, 跨实验不串
- **追问**: 互斥实验? → 同层只能进一个

#### Q3. A/B 测试的指标怎么选?
- **难度**: ⭐⭐⭐
- **答案**:
  - OEC (Overall Evaluation Criterion): 北极星指标
  - 护栏指标: 不能下降的 (e.g. 负反馈率)
  - 短/中/长指标
- **追问**: 抖音 OEC? → App Stay Time, LT7

#### Q4. 显著性检验的方法?
- **难度**: ⭐⭐⭐
- **答案**:
  - t 检验 (连续): 评估均值差异
  - 卡方检验 (离散): 评估比例差异
  - 方差分析 (多组)
- **追问**: 样本量? → power analysis, α=0.05, power=0.8

#### Q5. A/B 实验的辛普森悖论?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 分组看效果相反, 整体看正向
  - 推荐系统常见 (不同用户群效果不同)
- **追问**: 怎么避免? → 分人群看指标

#### Q6. A/B 实验的流量倾斜?
- **难度**: ⭐⭐⭐
- **答案**:
  - 新模型效果更好, 用户向新模型迁移
  - 严重时, 实验组效果虚高
- **追问**: 怎么避免? → 锁定实验组用户, 强制分桶

#### Q7. 多变量实验怎么做?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 因子设计 (Factorial Design): 多个变量同时实验
  - 分层实验: 召回/粗排/精排/UI 各一层
- **追问**: 字节? → 实验平台分层

#### Q8. 长效实验?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 新鲜效应 (Novelty): 用户对新鲜好奇, 效果短期高估
  - 学习效应 (Learning): 用户逐渐适应, 效果长期变化
  - 季节性: 不同时段效果不同
- **追问**: 字节? → 7 日 / 30 日观察

#### Q9. CUPED 方差缩减?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 用历史协变量减少方差
  - Y_i_adjusted = Y_i - θ * X_i
  - θ = cov(Y, X) / var(X)
- **追问**: 字节? → 实验平台内置

#### Q10. A/B 实验怎么做转化漏斗?
- **难度**: ⭐⭐⭐
- **答案**: 曝光 → 点击 → 转化, 每步单独看
- 转化漏斗某步下降, 可能有问题
- **追问**: 字节? → 实验报告自动生成

#### Q11. 互斥实验 vs 分层实验?
- **难度**: ⭐⭐⭐
- **答案**:
  - 互斥: 同层只能进一个实验, 避免干扰
  - 分层: 不同层独立实验, 互不干扰
- **追问**: 字节? → 多层实验平台

#### Q12. A/B 实验的护栏指标?
- **难度**: ⭐⭐⭐
- **答案**:
  - 不能下降的指标: 负反馈率, 投诉率
  - 性能指标: 延迟, 错误率
- **追问**: 护栏告警? → 实时监控

#### Q13. 怎么设计一个推荐系统的 A/B 实验?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  1. 假设: 新模型提升 CTR
  2. 指标: CTR, 时长, 互动率
  3. 流量: 5% 实验
  4. 周期: 7 天
  5. 显著性: p < 0.05
- **追问**: 样本量计算? → power analysis

#### Q14. Switchback 实验?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 不按用户分, 按时间分
  - 1 小时对照, 1 小时实验
  - 适合网络效应场景
- **追问**: 字节? → 抖音/快手某些实验

#### Q15. 实验效果的评估时间多长?
- **难度**: ⭐⭐⭐
- **答案**:
  - 短指标 (CTR): 1-3 天
  - 中指标 (时长): 3-7 天
  - 长指标 (留存): 7-30 天
- **追问**: 全量时间? → 看稳定性

#### Q16. 怎么验证 A/B 实验结论可靠?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 显著性检验
  - 多天观察
  - 多指标交叉验证
  - 反转检查 (翻转实验组对照)
- **追问**: 字节? → 实验平台自动检查

#### Q17. A/B 实验与灰度发布?
- **难度**: ⭐⭐⭐
- **答案**:
  - A/B: 实验组对照, 验证效果
  - 灰度: 逐步放量, 监控风险
- **追问**: 字节? → 内部发布平台

#### Q18. A/B 实验的样本污染?
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 用户跨实验串桶
  - 老用户新模型, 数据混乱
- **追问**: 怎么避免? → hash 分桶 + 用户锁定

#### Q19. 怎么判断 A/B 实验结果?
- **难度**: ⭐⭐⭐
- **答案**:
  - p 值 < 0.05
  - 效果大小 (effect size) > 业务阈值
  - 多指标一致
- **追问**: 字节? → 0.1% CTR 提升也算显著

#### Q20. A/B 实验的伦理?
- **难度**: ⭐⭐
- **答案**:
  - 用户知情同意
  - 数据隐私保护
  - 不损害用户体验
- **追问**: 推荐系统? → 用户协议 + 数据合规

---

## 13. 推荐系统系统设计

### 13.1 设计一个短视频推荐系统 (抖音/TikTok)

#### 13.1.1 需求分析

**功能需求**:
- 用户打开 APP, 推荐 10-20 个视频
- 实时响应 (< 200ms)
- 个性化推荐 (每个用户不同)
- 多目标: 完播/点赞/评论/转发/关注

**非功能需求**:
- QPS: 10 万+ (千万级 DAU)
- 延迟: < 200ms
- 数据量: 亿级视频, 百亿级用户行为
- 7×24 在线服务

#### 13.1.2 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│           短视频推荐系统完整架构                               │
│                                                              │
│   ┌────────────────────────────────────────────────────┐   │
│   │   召回层 (Recall)                                    │   │
│   │   多路并行:                                          │   │
│   │   - 双塔向量召回 (User/Video embedding → ANN)       │   │
│   │   - 协同过滤 (i2i/u2i)                              │   │
│   │   - 关注/订阅 (i2i)                                 │   │
│   │   - 热门/标签召回 (规则)                            │   │
│   │   - 探索流量 (Bandit)                                │   │
│   │   输出: ~1000 候选                                    │   │
│   └────────────────────┬───────────────────────────────┘   │
│                        ↓                                     │
│   ┌────────────────────────────────────────────────────┐   │
│   │   粗排 (Pre-Rank)                                    │   │
│   │   轻量模型: 双塔 / GBDT / COLD / 蒸馏                 │   │
│   │   目标: 快速截断, 控制规模                            │   │
│   │   输出: ~200-500 候选                                 │   │
│   └────────────────────┬───────────────────────────────┘   │
│                        ↓                                     │
│   ┌────────────────────────────────────────────────────┐   │
│   │   精排 (Rank)                                        │   │
│   │   复杂模型: DIN / DIEN / HSTU / OneRec              │   │
│   │   多目标: 完播/点赞/评论/时长                        │   │
│   │   实时特征: 近期行为序列 (实时拉)                    │   │
│   │   输出: ~50-100 候选                                 │   │
│   └────────────────────┬───────────────────────────────┘   │
│                        ↓                                     │
│   ┌────────────────────────────────────────────────────┐   │
│   │   重排 (Re-Rank)                                     │   │
│   │   - 多样性 (MMR / DPP)                              │   │
│   │   - 去重 (曝光过滤)                                  │   │
│   │   - 新鲜度提权                                        │   │
│   │   - 规则 (运营, 版权, 负反馈)                       │   │
│   │   - 探索流量 (EE)                                    │   │
│   │   输出: ~10-20 最终曝光                              │   │
│   └────────────────────┬───────────────────────────────┘   │
│                        ↓                                     │
│   ┌────────────────────────────────────────────────────┐   │
│   │   反馈闭环                                            │   │
│   │   用户行为 → 日志 → 实时流 (Kafka/Flink)            │   │
│   │   → 离线训练样本 → 模型更新                           │   │
│   │   → 实时特征 (Redis) 更新                            │   │
│   └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 13.1.3 关键设计点

1. **召回多路并行**:
   - 双塔向量召回 (主, 50%)
   - i2i 协同过滤 (30%)
   - 关注/热门 (20%)
   - 探索 (5-10%)

2. **实时特征**:
   - 用户最近 N 次行为 (滑动窗口)
   - 视频热度 (实时)
   - 上下文 (时间/设备)

3. **多目标融合**:
   - 精排层输出多个目标
   - 加权融合: `score = w1*完播 + w2*点赞 + w3*时长 + ...`
   - 权重通过 AB 实验调优

4. **探索与利用**:
   - UCB / Thompson Sampling
   - 5-10% 流量给新内容
   - 新视频冷启动用 Bandit

5. **A/B 实验**:
   - 流量分桶 (hash)
   - 短/中/长指标监控
   - 全量上线

#### 13.1.4 技术选型

| 模块 | 技术 |
| :--- | :--- |
| 数据存储 | HDFS / HBase / Redis / Cassandra |
| 实时流 | Kafka / Flink / Spark Streaming |
| 离线训练 | Spark / PyTorch / TensorFlow |
| 在线服务 | 自研 (字节) / TF Serving / Triton |
| 向量检索 | Faiss / Milvus / Vearch (字节) |
| 特征平台 | 自研 / Feast / Tecton |
| 监控 | Prometheus / Grafana / ELK |
| AB 实验 | 自研平台 |

### 13.2 设计一个电商推荐 (淘宝)

#### 13.2.1 业务特点

- **GMV 导向**: 转化率 (CVR) 比 CTR 更重要
- **多场景**: 首页 / 详情页 / 购物车 / 搜索
- **多目标**: CTR + CVR + GMV + 收藏

#### 13.2.2 关键差异

- **MTGR**: 保留交叉特征, 不全 GR
- **ESMM**: 全空间建模 CTR + CVR
- **长期价值**: GMV > 点击

#### 13.2.3 架构

```text
用户请求
   ↓
召回 (i2i + u2i + 向量 + 搜索)
   ↓
粗排 (COLD / 蒸馏)
   ↓
精排 (ESMM + 多任务 + 时长)
   ↓
重排 (多样性 + 业务规则)
   ↓
曝光
```

### 13.3 设计一个新闻推荐 (今日头条)

#### 13.3.1 业务特点

- **强时效**: 新闻很快过期
- **内容理解**: NLP 提取关键词
- **冷启动严重**: 每天大量新内容
- **去重**: 同一事件多篇报道

#### 13.3.2 关键设计

- **实时内容 embedding**: 内容入库即生成 SID
- **新内容曝光**: 给 5-10% 探索流量
- **主题去重**: 同一事件只显示一篇
- **时效衰减**: 老新闻降权

### 13.4 设计一个广告系统 (CTR + CVR)

#### 13.4.1 业务特点

- **Bid × pCTR × pCVR × 1000 = eCPM**
- **DSP + SSP + ADX**
- **实时竞价 (RTB)**: 100ms 决策

#### 13.4.2 关键模块

- **CTR 预估**: GBDT / DeepFM / DCN
- **CVR 预估**: ESMM / 多任务
- **召回**: 双塔 + 关键词
- **粗排**: 轻量模型
- **精排**: pCTR × pCVR × Bid

#### 13.4.3 腾讯 GPR

- **One-Model**: 端到端 GR
- **HEPO**: 分层过程奖励
- **HHD 解码器**: 解耦意图和生成

### 13.5 完整方案模板

```text
# 推荐系统设计方案

## 1. 业务目标
- OEC: xxx
- 护栏指标: xxx

## 2. 流量与性能
- DAU: xxx
- QPS: xxx
- 延迟: < xxx ms
- 数据量: xxx

## 3. 架构 (Pipeline)
- 召回 (多路并行) → 粗排 → 精排 → 重排

## 4. 召回设计
- 协同过滤
- 向量召回 (双塔 + ANN)
- 规则召回
- 探索流量

## 5. 粗排设计
- 轻量模型 (双塔 / GBDT / 蒸馏)
- 目标: 快速截断

## 6. 精排设计
- 模型选择: DIN / DIEN / MMoE / HSTU / OneRec
- 多目标: CTR / CVR / 时长 / 互动
- 实时特征: 用户行为序列
- 损失函数: CE / BPR / 多任务

## 7. 重排设计
- 多样性 (MMR / DPP)
- 去重 / 打散
- 业务规则
- 探索流量

## 8. 特征工程
- 用户画像
- 物品画像
- 上下文
- 实时行为
- 多模态 (文本/图像)

## 9. 模型训练
- 离线训练 (Spark / PyTorch)
- 在线增量 (FTRL / 蒸馏)
- 分布式 (DDP / FSDP / PS)

## 10. 在线服务
- 模型服务 (TF Serving / Triton / 自研)
- 向量检索 (Faiss / Milvus)
- 特征服务 (Redis / 自研)
- 缓存 + 降级

## 11. 评估
- 离线: AUC, NDCG
- 在线: AB 实验, 核心指标
- 监控: 性能 + 业务

## 12. 工程挑战
- 数据漂移
- 冷启动
- 长尾
- 探索利用
- 多样性
```

---

## 14. 字节 / 阿里 / 腾讯 / 美团 推荐算法面试真题

> 50 道真题, 按公司分类, 含参考思路

### 字节跳动 (15 道)

#### Q1. 设计抖音首页推荐系统 (字节三面)
- **难度**: ⭐⭐⭐⭐⭐
- **思路**:
  - 漏斗架构: 召回 (多路) → 粗排 → 精排 (DIN + 多目标) → 重排
  - 召回: 双塔 + i2i + 关注 + 热门 + 探索
  - 精排: 多目标融合 (完播/点赞/评论/时长)
  - 重排: 多样性 + 业务规则
  - 200ms 延迟: 大量缓存 + 蒸馏 + GPU 推理
- **追问**:
  - 新视频冷启动? → 内容 embedding + 探索流量
  - 位置偏差? → IPS + 训练加位置特征
  - 多样性怎么保证? → MMR / 同作者/主题打散

#### Q2. MMoE 的跷跷板现象 (字节一面)
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 多个 Expert 被共享, 任务梯度冲突
  - 解决: PLE 任务专属 Expert, 或 GradNorm 动态加权
- **追问**: ESMM 怎么用? → 概率关系建模, 联合训练 CTR/CVR

#### Q3. DIN 的 Attention 公式 (字节一面)
- **难度**: ⭐⭐⭐
- **答案**: `weight_i = MLP(e_i ⊕ e_target ⊕ e_i*e_target)`, 加权聚合
- **追问**: 为什么不用 softmax? → DIN 要保留兴趣强度

#### Q4. AUC 和 GAUC 的区别 (字节一面)
- **难度**: ⭐⭐⭐
- **答案**: AUC 全局, GAUC 用户内 AUC 加权
- **追问**: GAUC 局限性? → 单用户全正或全负不稳定

#### Q5. 召回为什么用近似检索 (字节一面)
- **难度**: ⭐⭐
- **答案**: 亿级物品库, 精确 KNN O(N) 不可行, ANN O(log N)
- **追问**: HNSW 原理? → 多层图索引

#### Q6. ROC 和 PR 曲线 (字节 TikTok 一面)
- **难度**: ⭐⭐⭐
- **答案**:
  - ROC: FPR vs TPR, 对类别不均衡鲁棒
  - PR: Precision vs Recall, 适合类别极度不平衡
- **追问**: 推荐系统用哪个? → 都用, 主看 AUC

#### Q7. 多场景建模怎么做 (字节抖音二面)
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - STAR, PEPNet, OxygenREC
  - 场景信息 → 指令, 统一架构
- **追问**: 字节怎么做? → 内部 RankMixer 多场景

#### Q8. RankMixer 的结构原理 (字节抖音二面)
- **难度**: ⭐⭐⭐⭐
- **答案**: Token 机制 + 多尺度注意力 + 场景 block
- **追问**: 与 DIN 区别? → RankMixer 显式 token, 更灵活

#### Q9. Position Bias 怎么判断 (字节一面)
- **难度**: ⭐⭐⭐
- **答案**: 看同一 item 在不同位置的 CTR 差异
- **追问**: 其他 bias? → selection, exposure, conformity, popularity

#### Q10. 召回和排序的负样本构造 (字节一面)
- **难度**: ⭐⭐⭐
- **答案**:
  - 召回: 全局随机
  - 精排: 曝光未点击
- **追问**: 比例? → 1:5 到 1:10

#### Q11. 长序列建模长度与方案 (字节二面)
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 短期: 100-200, DIN/DIEN
  - 长期: 1024-8192, HSTU
  - 超长: SIM 检索
- **追问**: HSTU 怎么高效? → Pointwise Aggregated

#### Q12. 特征时序不一致怎么办 (字节二面)
- **难度**: ⭐⭐⭐
- **答案**: 严格时间戳, 离线用真实线上特征
- **追问**: 监控 PSI? → >0.1 警告, >0.2 报警

#### Q13. 双塔模型优缺点 (字节)
- **难度**: ⭐⭐⭐
- **答案**:
  - 优: 可预建索引, 检索快
  - 缺: 丢失 cross 特征
- **追问**: 改进? → PolyEncoder / 浅层 cross

#### Q14. 推荐系统冷启动策略 (字节)
- **难度**: ⭐⭐⭐
- **答案**:
  - 新用户: 注册信息 + 热门 + Bandit
  - 新物品: 内容 embedding + 探索流量
- **追问**: 字节新视频? → 内容 embedding + 相似召回

#### Q15. 过拟合怎么解决 (字节二面)
- **难度**: ⭐⭐
- **答案**: Dropout, L2, Early Stopping, 数据增强
- **追问**: 推荐系统? → 同上 + 业务正则

### 阿里 (15 道)

#### Q16. ESMM 怎么解决 CVR 偏差 (阿里)
- **难度**: ⭐⭐⭐⭐
- **答案**: 全空间联合训练 CTR + CTCVR, pCVR = pCTCVR/pCTR
- **追问**: 为什么用乘法? → 除法数值不稳定

#### Q17. DIN 模型细节 (阿里)
- **难度**: ⭐⭐⭐⭐
- **答案**: Target-aware attention, 激活单元, 无 softmax
- **追问**: 工业部署? → TensorFlow + PS

#### Q18. DIEN 的两层 GRU (阿里)
- **难度**: ⭐⭐⭐⭐
- **答案**: 兴趣抽取 GRU + 兴趣进化 AUGRU
- **追问**: AUGRU 公式? → attention 调制 update gate

#### Q19. SIM 长序列模型 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: Search-based Interest, 检索 Top-K 兴趣子序列
- **追问**: 为什么不用全长? → 计算成本太高

#### Q20. URM 与传统双塔的区别 (阿里 Alimama)
- **难度**: ⭐⭐⭐⭐
- **答案**:
  - 双塔: 独立编码, 最后内积
  - URM: LLM 作为通用检索器, 文本查询
- **追问**: URM 用什么 LLM? → Qwen-7B

#### Q21. NEZHA 推测解码 (阿里)
- **难度**: ⭐⭐⭐⭐
- **答案**: 内嵌轻量级 autoregressive draft head, 推测解码
- **追问**: 为什么零牺牲? → 不需要外挂独立 draft model

#### Q22. ReaSeq 多智能体 CoT (阿里 TaoRank)
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 多智能体协作蒸馏产品知识 + 扩散 LLM 推理
- **追问**: CoT 推理为什么有效? → 显式思考过程

#### Q23. 推荐系统的 Q1 增长目标 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: 用户活跃度, GMV, 留存
- **追问**: 不同业务? → 淘宝 GMV, 闲鱼 DAU

#### Q24. MMoE 与 PLE 的参数量 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: MMoE 轻, PLE 重但效果好
- **追问**: 显存限制? → 用 MMoE

#### Q25. AB 实验怎么做显著性检验 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: t 检验, 卡方, CUPED 方差缩减
- **追问**: 样本量? → power analysis

#### Q26. 如何保证线上线下一致性 (阿里)
- **难度**: ⭐⭐⭐⭐
- **答案**: 统一特征 SDK, 特征平台, Redis 实时拉取
- **追问**: 字节? → 自研 Feature Store

#### Q27. 推荐系统怎么做排序模型迭代 (阿里)
- **难度**: ⭐⭐⭐⭐
- **答案**:
  1. 离线 AUC 提升
  2. 小流量 AB
  3. 全量上线
  4. 监控核心指标
- **追问**: 多目标? → 多任务联合训练 + 单独监控

#### Q28. ESMM 在线推理 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: 只输出 pCVR, 排序用
- **追问**: pCTCVR? → 不直接用, 只用于训练

#### Q29. 阿里推荐系统怎么做冷启动 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: ESMM 借 CTR 数据, 内容 embedding
- **追问**: 字节? → 内容 embedding + 探索流量

#### Q30. 推荐系统的多目标融合 (阿里)
- **难度**: ⭐⭐⭐
- **答案**: 加权融合 score = w1*p1 + w2*p2 + ...
- **追问**: 权重怎么调? → AB 实验 + 业务经验

### 腾讯 (10 道)

#### Q31. GPR 的 HHD 解码器 (腾讯)
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: HSD (用户意图) + PTD (思考-精炼-生成) + HTE (价值估计)
- **追问**: 为什么三层? → 解耦意图理解和生成

#### Q32. HEPO 算法细节 (腾讯)
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: 分层过程奖励 + Anticipatory Request Rehearsal
- **追问**: ARR 是什么? → 模拟预期请求

#### Q33. 微信视频号广告推荐 (腾讯)
- **难度**: ⭐⭐⭐⭐
- **答案**: GPR One-Model, 统一广告+有机内容
- **追问**: 业务收益? → GMV 显著提升

#### Q34. 多场景建模怎么做 (腾讯)
- **难度**: ⭐⭐⭐⭐
- **答案**: STAR / PEPNet / 场景指令化
- **追问**: 腾讯用什么? → GPR 场景指令

#### Q35. 推荐系统冷启动 (腾讯)
- **难度**: ⭐⭐⭐
- **答案**: 内容 + 探索流量 + 元学习
- **追问**: 微信视频号? → 内容 + 关注 + 朋友圈

#### Q36. 推荐系统反馈延迟 (腾讯)
- **难度**: ⭐⭐⭐⭐
- **答案**: 延迟反馈建模, 重要性加权
- **追问**: 实际? → 截止窗口 + 部分反馈

#### Q37. 推荐系统怎么做多目标 (腾讯)
- **难度**: ⭐⭐⭐
- **答案**: MMoE / PLE / ESMM
- **追问**: 腾讯? → GPR + 多 token SID

#### Q38. 广告 CTR 怎么做 (腾讯)
- **难度**: ⭐⭐⭐
- **答案**: GBDT / DeepFM / DCN
- **追问**: GPR 怎么做? → HHD + MTP + HEPO

#### Q39. 推荐系统怎么做强化学习 (腾讯)
- **难度**: ⭐⭐⭐⭐
- **答案**: DQN / PPO / GRPO
- **追问**: HEPO 与 GRPO 区别? → 分层 + ARR

#### Q40. 推荐系统位置偏差 (腾讯)
- **难度**: ⭐⭐⭐
- **答案**: 位置 embedding + IPS
- **追问**: 字节? → IPS + 训练加位置

### 美团 (10 道)

#### Q41. MTGR 与 HSTU 的差异 (美团)
- **难度**: ⭐⭐⭐⭐⭐
- **答案**: HSTU 全 GR, MTGR 保留交叉特征, 判别式 loss
- **追问**: 为什么? → 外卖业务交叉特征关键

#### Q42. Group LayerNorm 作用 (美团)
- **难度**: ⭐⭐⭐⭐
- **答案**: 按 Token 类型分组, 不同语义空间对齐
- **追问**: 怎么分组? → 用户/物品/上下文/交叉

#### Q43. MTGR 训练引擎 (美团)
- **难度**: ⭐⭐⭐⭐
- **答案**: TorchRec + Fused Cutlass kernel
- **追问**: 为什么手写 kernel? → Triton 效率不够

#### Q44. 美团外卖推荐特点 (美团)
- **难度**: ⭐⭐⭐⭐
- **答案**: 低点击率高复购, 交叉特征关键
- **追问**: MTGR 怎么解决? → 保留交叉特征

#### Q45. 推荐系统长序列 (美团)
- **难度**: ⭐⭐⭐
- **答案**: SIM / STCA / LASER
- **追问**: 美团? → 同上 + HSTU

#### Q46. 美团推荐怎么做冷启动 (美团)
- **难度**: ⭐⭐⭐
- **答案**: 内容 + 探索 + 业务规则
- **追问**: 新商家? → 商家画像 + 同类推荐

#### Q47. 推荐系统 A/B 实验 (美团)
- **难度**: ⭐⭐⭐
- **答案**: 流量分桶 + 显著性 + 多指标
- **追问**: 美团? → 内部实验平台

#### Q48. 推荐系统多目标 (美团)
- **难度**: ⭐⭐⭐
- **答案**: AITM (借鉴 PLE)
- **追问**: 业务? → 外卖 CTR + CVR + GMV

#### Q49. 美团搜索推荐 (美团一面)
- **难度**: ⭐⭐⭐
- **答案**: BM25 + 向量召回 + 重排
- **追问**: BM25 vs TF-IDF? → BM25 长度归一化

#### Q50. GRPO vs PPO (美团一面)
- **难度**: ⭐⭐⭐
- **答案**: GRPO 不要 critic, 组内归一化, 更省显存
- **追问**: 美团推荐? → RL 探索利用

---

## 15. 行为面试

### 15.1 STAR 模板

**S (Situation)**: 业务背景 (3 句)
**T (Task)**: 我的职责 (2 句)
**A (Action)**: 我做了什么 (5-7 句, 具体)
**R (Result)**: 量化结果 (3 句)

### 15.2 推荐算法工程师常见问题

#### Q1. 介绍一下你自己
- **建议**:
  - 1 分钟: 教育背景 + 主要工作
  - 2 分钟: 关键技术 + 业务成果
  - 突出与目标岗位的匹配

#### Q2. 最有挑战的项目?
- **建议**:
  - 用 STAR
  - 具体数字 (性能提升, 业务指标)
  - 你的角色 (不是团队的)

#### Q3. 为什么离开上一家公司?
- **建议**:
  - 正面理由: 学习 / 成长 / 业务匹配
  - 不要抱怨老东家

#### Q4. 5 年职业规划?
- **建议**:
  - 技术深度 + 业务广度
  - 与公司成长路径匹配
  - 具体可执行

#### Q5. 你的优缺点?
- **优点**:
  - 与岗位匹配的技术能力
  - 强项 + 例子
- **缺点**:
  - 真实但不致命
  - 改进计划

#### Q6. 你最不喜欢的同事类型?
- **建议**: 务实, 不要虚伪
- 例如: 不喜欢不分享的人

#### Q7. 为什么选我们公司?
- **建议**:
  - 业务: 与你的经验匹配
  - 技术: 你感兴趣的方向
  - 文化: 与你的价值观契合

#### Q8. 期望薪资?
- **建议**:
  - 市场行情 (BOSS, 脉脉)
  - 当前薪资 + 涨幅
  - 不要狮子大开口

#### Q9. 你有什么问题想问?
- **5 个加分问题**:
  1. 团队当前最大的技术挑战是什么?
  2. 这个岗位 3 个月的成长路径?
  3. 团队的代码 review / 知识分享机制?
  4. 业务的核心指标是什么? 我能贡献什么?
  5. 团队对前沿技术 (LLM / GR) 的态度?

---

## 附录 A: 24 篇 GenRec 论文速查表

| 论文 | 工业方 | 核心创新 |
| :--- | :--- | :--- |
| HSTU | Meta | 序列转导, Scaling Law |
| OneRec-V1 | 快手 | 端到端 GR 替代级联, ECPO |
| OneRec-V2 | 快手 | Lazy Decoder, GBPO |
| OneRec-Think | 快手 | 显式推理 |
| OpenOneRec | 快手 | 开源, Qwen3 |
| OneMall | 快手 | 电商多场景 |
| OneLive | 快手 | 直播 |
| OneSearch | 快手 | 搜索统一 |
| MTGR | 美团 | HSTU + 交叉特征 |
| OxygenREC | 京东 | 指令跟随, 快慢思考 |
| GPR | 腾讯 | 广告 One-Model, HHD |
| S-GRec | 腾讯 | 个性化语义 |
| HiGR | 腾讯 | 层级 slate |
| NEZHA | 阿里 | 推测解码 |
| URM | 阿里 | LLM 检索 |
| ReaSeq | 阿里 | CoT + 扩散 |
| REG4Rec | 阿里 | 推理增强 |
| RankGR | 阿里 | Listwise DPO |
| Masked Diffusion GR | 阿里 | 扩散生成 |
| PLUM | Google | LLM 持续预训练 |
| TIGER | Google | 第一个 GR |
| PinRec | Pinterest | 多 token 连续 |
| KuaiFormer | 快手 | 检索 Transformer |
| GRAB | 百度 | LLM 启发 CTR |
| LLaDA-Rec | Snap | 离散扩散 |
| RelayGR | 华为 | 跨阶段推理 |
| Sage | 小红书 | 学术验证 |
| Climber-Pilot | 网易 | 非近视指令 |
| FLAME | 网易 | GR 高效服务 |
| Farewell to Item IDs | 字节 | 搜索引擎 |
| MERGE | 字节 | 流式物品索引 |
| Cadet | LinkedIn | 广告 CTR |

## 附录 B: 关键论文链接

- HSTU: arxiv.org/abs/2402.17152
- OneRec: arxiv.org/abs/2502.18965
- OneRec-V2: arxiv.org/abs/2508.20900
- OneRec-Think: arxiv.org/abs/2510.11639
- MTGR: arxiv.org/abs/2505.18654
- GPR: arxiv.org/abs/2511.10138
- OxygenREC: arxiv.org/html/2512.22386v1
- TIGER: arxiv.org/abs/2305.05065
- LC-Rec: github.com/zhengbw0324/LC-Rec
- GRAB: 即将公布
- GR Survey: blog.recsys-frontier.com

## 附录 C: 内部参考

- rec教程/第05章-DLRM到GR范式迁移.md
- rec教程/附录E-面试与Q&A速查.md (50 问速查)
- GR/第09章-规模化排序与新架构/deep-dives/ (24 篇论文分析)
- SID教程/ 全部
- RL/4-RL在GenRec中的经典方法.md (2447 行 RL 经典)

## 附录 D: 速查表

### D.1 推荐系统 Pipeline

```text
召回 → 粗排 → 精排 → 重排
↓     ↓     ↓     ↓
1000  200   50    10
```

### D.2 GR 主流论文时间线

```text
2023 TIGER (Google)
2024 HSTU (Meta) - Scaling Law
2024 LC-Rec (ICDE) - LLM + SID
2025 OneRec-V1/V2/Think (快手)
2025 MTGR (美团)
2025 GPR (腾讯)
2025 OxygenREC (京东)
2025 NEZHA/URM/ReaSeq (阿里)
2026 OpenOneRec, OneMall (开源)
```

### D.3 关键缩写

- DLRM: Deep Learning Recommendation Model
- GR: Generative Recommendation
- SID: Semantic ID
- VQ-VAE: Vector Quantized VAE
- RQ-VAE: Residual Quantized VAE
- InfoNCE: Information Noise Contrastive Estimation
- NTP: Next Token Prediction
- MMoE: Multi-gate Mixture-of-Experts
- PLE: Progressive Layered Extraction
- ESMM: Entire Space Multi-task Model
- HSTU: Hierarchical Sequential Transduction Unit
- ECPO: Early Clipped Policy Optimization
- GBPO: Gradient-Bounded Policy Optimization
- GRPO: Group Relative Policy Optimization
- DPO: Direct Preference Optimization
- PPO: Proximal Policy Optimization

---

> **版本**: v1.0 (2026-07-23) · **维护**: 自维护
>
> *Generated with comprehensive research from Meta HSTU paper, OneRec series, MTGR, OxygenREC, GPR, TIGER, LC-Rec papers + 字节/阿里/腾讯/美团/京东/小红书面试真题*
>
> *祝面试顺利!*
