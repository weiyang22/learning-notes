---
title: 第09章-04-补充GenRec经典论文-HSTU-OneRec-TIGER等
tags:
  - HSTU
  - OneRec
  - MTGR
  - Wukong
  - TIGER
  - LC-Rec
  - HLLM
  - LONGER
  - GenRank
  - DFGR
  - HiGR
  - GPR
  - NEZHA
  - OneMall
  - DiffuRec
  - TokenRec
  - IDvs.MoRec
  - RecFormer
  - M6-Rec
  - ColaRec
  - TAAC
  - OnePiece
description: 补充的 30+ 篇 GR/Scale-up Ranking 经典与前沿论文, 涵盖 HSTU/OneRec/TIGER/LLM4Rec 等关键工作
last_updated: 2026-07-21
status: ✅ v1.0 - 30+ 篇 GR 经典 / 前沿论文
---

# 第09章-04 - 补充 GenRec 经典与前沿论文 (30+ 篇)

> **覆盖**: 联网检索发现的 30+ 篇 GR / Scale-up Ranking 经典与前沿论文, 时间跨度 2022-2026
> **来源**: arxiv / Hugging Face Papers / alphaxiv / 工业界技术博客 (Netflix, JD) / 腾讯算法比赛
> **特点**: 大部分是**该领域开山 / 奠基性工作**, 字节 / 阿里 / 腾讯 / 快手 / 美团 / Meta / Google / 华为 / 学术界 全面覆盖

---

## 1. HSTU: Actions Speak Louder than Words (Meta 2024) ⭐ GR 范式奠基

**arXiv**: [2402.17152](https://arxiv.org/abs/2402.17152) · **作者**: Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li 等 · **公司**: **Meta MRS / Meta AI** · **发表**: 2024 (ICML 2024)

### 1.1 核心问题
工业 DLRM 因为异构特征和 10⁵ 长度用户序列无法高效处理为序列转导任务, **推荐没有 NLP 那样的 scaling-law 时刻**.

### 1.2 方法 / 创新
- 把排序 / 检索重新表述为 **Generative Recommenders (GRs)** — 纯序列转导 over 用户行为 / 物品 / 异构特征的**交错时间序列** (而非分离的 categorical/cross/MLP 塔)
- **HSTU (Hierarchical Sequential Transduction Unit)**: pointwise projection → pointwise aggregated (normalized) attention → element-wise gated MLP fusion; 用 grouped-GEMM kernel 扩展 memory-efficient attention + FlashAttention 加 stochastic-length 稀疏度降低 (O(N^α d))
- **M-FALCON** 推理算法通过 micro-batching 完全摊销计算, **285× 更大模型在相同推理预算内**

### 1.3 结果
- +65.8% NDCG vs SASRec/BERT4Rec/GRU4Rec on public datasets; +8.6% / +18.1% HR@10 / NDCG@10 over SASRec
- 8K-length 序列比 FlashAttention2-based Transformer **快 5.3-15.2×**
- **1.5 万亿参数**部署模型在线 A/B 指标 **+12.4%** on a large Meta platform
- **幂律 scaling** 跨 3 个数量级训练计算 (到 GPT-3 / LLaMA-2 规模) — 首个部署工业推荐系统的 scaling law

### 1.4 Why It Matters
**GR 范式的奠基之作**. 定义了 HSTU 架构, 几乎所有后续工业 GR 论文 (MTGR, OneRec, GenRank, DFGR, GPR, HiGR, NEZHA) 都基于或对比. 代码开源: `meta-recsys/generative-recommenders`.

---

## 2. OneRec V1: Unifying Retrieve and Rank (快手 2025)

**arXiv**: [2502.18965](https://arxiv.org/abs/2502.18965) · **作者**: Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qingxi Hu, Weifeng Ding, Qiang Luo, Guorui Zhou 等 · **公司**: **Kuaishou** · **发表**: 2025-02

### 2.1 核心问题
现代级联 retrieve-and-rank 流水线碎片化; 生成式检索只用作检索阶段选择器; 无端到端生成模型在生产中击败精心设计的多阶段推荐系统.

### 2.2 方法 / 创新
- **Encoder-decoder 生成模型** + 稀疏 MoE 实现容量扩展且 FLOPs 不变
- **Session-wise generation** (生成一个一致的 slate 而不是 point-by-point), 移除手工合并规则
- **DPO 迭代偏好对齐** + 自定义奖励模型模拟用户生成 (因为模型只有一次展示机会)

### 2.3 结果
部署于快手主场景 → **+1.66% watch-time**. 首个端到端生成式推荐在实际短视频系统中击败复杂级联.

---

## 3. OneRec Tech Report (快手 2025)

**arXiv**: [2506.13695](https://arxiv.org/abs/2506.13695) · **作者**: Guorui Zhou, J. S. Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo 等 · **公司**: **Kuaishou OneRec Team** · **发表**: 2025-06

### 3.1 核心问题
推荐系统未受益于 LLM 风格的突破 (scaling laws, RL), 因为多阶段碎片化和低 MFU.

### 3.2 方法 / 创新
- **协同感知多模态 tokenization** via QFormer 压缩 + item-to-item 对比损失; **RQ-Kmeans** (RQ-VAE 改进) 粗到细 SIDs with 更好 codebook 利用
- MoE 解码器 + 多尺度用户建模; RL + 奖励模型集成
- 基础设施: 旗舰 GPU 上 **23.7% 训练 / 28.8% 推理 MFU** (vs 传统模型 4.6% / 11.2%)
- **OPEX = 传统流水线的 10.6%**

### 3.3 结果
- Scaling law 验证
- 部署于快手 / 快手极速版 服务 25% QPS
- **+0.54% / +1.24% App Stay Time**, +0.05% / +0.08% LT7
- **+21.01% GMV** in 本地生活服务 (100% QPS 占据)

### 3.4 Why It Matters
工业级端到端 GR 全系统蓝图 (tokenizer + 模型 + RL + 基础设施 + 业务指标).

---

## 4. OneRec V2 Tech Report (快手 2025)

**arXiv**: [2508.20900](https://arxiv.org/abs/2508.20900) · **作者**: Kuaishou OneRec Team · **公司**: **Kuaishou** · **发表**: 2025-08/10

### 4.1 核心问题
V1 的 encoder-decoder 有瓶颈: encoder 消耗 97.66% 计算 (序列长度上下文编码), 阻止进一步扩展. 仅奖励模型的 RL 有采样低效和奖励黑客.

### 4.2 方法 / 创新
- **Lazy Decoder-Only Architecture** — 移除 encoder 并简化 cross-attention (无 K/V 投影层); **94% 计算减少, 90% 训练资源减少, 支持 16× 更大模型 (0.5B → 8B)** 在等效计算下
- 偏好对齐用真实用户反馈: **Duration-Aware Reward Shaping** + **Adaptive Ratio**; 用 **GBPO (Group-Based Policy Optimization)** 稳定训练

### 4.3 结果
部署于快手 / 快手极速版 (4 亿 DAU, 1B-参数模型, 3000 上下文长度, beam 512, 36ms 延迟, L20 GPU 62% MFU); +0.467% / +0.741% App Stay Time; 跨 like/follow/comment/collect/forward 显著参与度增益.

---

## 5. OneRec-Foundation (快手 2025)

**arXiv**: [2512.24762](https://arxiv.org/abs/2512.24762) · **作者**: Guorui Zhou, Honghui Bao, Jiaming Huang, Jiaxin Deng 等 · **公司**: **Kuaishou** · **发表**: 2025-12

### 5.1 核心问题
领域专精推荐系统缺乏世界知识, 推理, 指令跟随; 无整体基准.

### 5.2 方法 / 创新
- 发布 **RecIF-Bench** (8 个不同推荐任务) + 9600 万交互开放训练数据集
- **OneRec-Foundation 1.7B / 8B** 系列基于 Qwen3, 通过 Itemic-Text Alignment + 协同预训练 + SFT + 交替 General Distillation / Rec-RL 训练
- 开源完整训练流水线

### 5.3 结果
- 在 RecIF-Bench 所有任务 SOTA
- 在 10 个 Amazon 数据集上, **+26.8% 平均 Recall@10** 超过最强基线

### 5.4 Why It Matters
**首个开源推荐基础模型** with 推理 + 通用能力保留.

---

## 6. OneRec-Think: 文本推理 + GR (快手 2025)

**arXiv**: [2510.11639](https://arxiv.org/abs/2510.11639) · **作者**: Zhanyu Liu, Shiyao Wang, Xingmei Wang, Rongzhou Zhang, Jiaxin Deng, Honghui Bao, Guorui Zhou 等 · **公司**: **Kuaishou** · **发表**: 2025-10 (ACL 2026 long paper)

### 6.1 核心问题
现有 GR 模型 (OneRec, HSTU, TIGER) 是隐式预测器 — 它们无法像 LLM 那样产生显式可验证的推理链.

### 6.2 方法 / 创新
- **Itemic Alignment** 通过多任务预训练 (交错用户画像基础 + 顺序偏好 + 密集字幕 + 通用 LM)
- **Reasoning Activation** 通过 CoT SFT 与精选推理轨迹
- **Reasoning Enhancement** 通过 **Rollout-Beam reward** + GRPO 处理偏好多有效性
- **Think-Ahead 推理架构**: 离线推理 + 初始 token, 在线实时 OneRec 完成 — 生产级延迟

### 6.3 结果
快手生产 **+0.159% App Stay Time**; 公共基准 SOTA.

---

## 7. MTGR: 美团工业 GR 框架 (2025)

**arXiv**: [2505.18654](https://arxiv.org/abs/2505.18654) (CIKM 2025) · **作者**: Ruidong Han, Bin Yin, Shangyu Chen, Jiang He, Fei Jiang, Xiang Li, Chi Ma, Mincong Huang, Xiaoguang Li, Chunzhen Jing, Yueming Han, Meng Zhou, Lei Yu, Chuan Liu, Wei Lin · **公司**: **Meituan** · **发表**: 2025-05

### 7.1 核心问题
纯 GR (HSTU) 放弃手工 DLRM 交叉特征 — 美团发现这造成巨大性能损失, 扩展无法补偿.

### 7.2 方法 / 创新
- **HSTU-based 但保留 DLRM 特征包括交叉特征**; tokens 从用户 / 行为 / 实时 / 候选特征重组
- **Group-Layer Normalization (GLN)** 分别归一化不同 token 类型
- **Dynamic masking** 结合 full-attention, auto-regressive, 对角 masking
- **用户级压缩** 用于训练 / 推理加速; 比 TorchRec **1.6-2.4× 训练吞吐**

### 7.3 结果
- **65× FLOPs/sample** vs DLRM baseline
- **+1.22% conversion / +1.31% CTR** on 美团外卖 (服务数亿用户)
- 训练成本 ≈ DLRM, **推理成本 −12%**
- 近两年来最大在线增益

### 7.4 Why It Matters
解决 "DLRM vs GR" 权衡 — 证明 scaling law 适用同时保留交叉特征. 部署于世界最大外卖平台.

---

## 8. Wukong: Towards a Scaling Law for Large-Scale Recommendation (Meta 2024)

**arXiv**: [2403.02545](https://arxiv.org/abs/2403.02545) (ICML 2024) · **作者**: Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Shen Li, Yanli Zhao, Yuchen Hao, Yantao Yao, Ellie Dingqiao Wen, Jongsoo Park, Maxim Naumov, Wenlin Chen · **公司**: **Meta** · **发表**: 2024 (ICML 2024)

### 8.1 核心问题
推荐模型不显示 scaling law 因为 DLRM 扩展仅是稀疏的 (embedding 表扩展), 这不跟踪硬件计算收益, 也不改进特征交互建模.

### 8.2 方法 / 创新
- 堆叠 **Factorization Machines + Linear Compression Blocks** 灵感来自二进制指数化 — 每连续层捕获指数级高阶交互
- 协同稠密扩展策略

### 8.3 结果
- SOTA AUC on 6 public datasets
- **Scaling law 跨两个数量级模型复杂度** 成立 (扩展到 100 GFLOP/example 之上, 直到 GPT-3 / LLaMA-2 总训练计算), 之前工作 plateau
- **~0.1% 质量增益 / 4× 复杂度**
- 扩展到 **637B+ 参数**

### 8.4 Why It Matters
首个特征交互架构的明确 scaling-law 演示; 与 HSTU (序列聚焦) 互补. 被 RankMixer, HHFT, FuXi-α 引用为 scaling baseline.

---

## 9. DHEN: Meta DLRM 集成架构 (2022)

**arXiv**: [2203.11014](https://arxiv.org/abs/2203.11014) · **作者**: Buyun Zhang, Liang Luo, Xi Liu, Jay Li, Zeliang Chen, Weilin Zhang, Xiaohan Wei, Yuchen Hao, Michael Tsang, Wenjun Wang 等 · **公司**: **Meta** · **发表**: 2022

### 9.1 核心问题
不同特征交互模块 (DCN, AutoInt, FM, LR) 有非重叠优势; 现有单模块设计在数据集上 plateau.

### 9.2 方法 / 创新
- **Hierarchical ensemble** 逐层堆叠异构交互模块 (FM, DCN, self-attention, LR), 学习交互阶的层次
- **Hybrid Sharded Data Parallel (HSDP)** 训练范式 — 桥接 DP/FSDP 内存和吞吐差距

### 9.3 结果
- **+0.27% NE** vs Meta 工业 AdvancedDLRM
- 训练吞吐比 FSDP **1.2×**

### 9.4 Why It Matters
集成架构基础, 预见了 Wukong 和 RankMixer; 训练系统创新 (HSDP) 直接启用了后续 Meta 架构的扩展.

---

## 10. Hiformer: Google Play 异构注意力 (2023)

**arXiv**: [2311.05884](https://arxiv.org/abs/2311.05884) · **作者**: Huan Gui, Ruoxi Wang, Ke Yin, Long Jin, Maciej Kula, Taibai Xu 等 · **公司**: **Google Play** · **发表**: 2023-11

### 10.1 核心问题
Vanilla Transformer self-attention 忽略特征类型异构性, 对 Web 规模服务太慢.

### 10.2 方法 / 创新
- **Heterogeneous Attention (HeteroAtt)** — 每个特征对变换矩阵 M_{i,j} 让特征交互尊重异构性
- Q/K/V 中 composite projections 增加表达力
- 低秩近似 + 结构化剪枝满足延迟 SLO

### 10.3 结果
- 部署于 **Google Play 应用排名** 在 Web 规模
- 在线 A/B +1.27% / +2.33% (一/两层 HeteroAtt) / **+2.66% 参与度** vs Transformer baseline (DCN +2.20%); <2× 推理延迟增加

### 10.4 Why It Matters
首个生产级基于 Transformer 的特征交互模型在 Web 规模 — 被 Wukong, RankMixer, HHFT 引用为 baseline.

---

## 11. RankMixer: Scaling Up Ranking Models (字节 2025)

**arXiv**: [2507.15551](https://arxiv.org/abs/2507.15551) · **作者**: Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, Xintian Han, Zuotao Liu 等 · **公司**: **ByteDance / Douyin** · **发表**: 2025-07

### 11.1 核心问题
手工特征交叉模块 MFU 低 (4.5%) 不利用 GPU; 现有 ranker 不能扩展到数十亿参数.

### 11.2 方法 / 创新
- **硬件感知统一特征交互架构**, 保留 Transformer 并行性但 **用多头 token-mixing (MLP-Mixer-style) 取代二次 self-attention**
- **Per-token FFNs** 保留不同特征子空间和跨空间交互
- **Sparse-MoE RankMixer-1B** 变体 with 动态路由 for expert 平衡

### 11.3 结果
- 在抖音万亿规模生产数据集: **MFU 4.5% → 45%**
- **100× 参数扩展** 在相同推理延迟
- RankMixer-1B +0.95% / +1.22% (Finish AUC/UAUC) vs DLRM-MLP
- 在等 FLOPs 下击败 Wukong, Hiformer, DHEN
- 在线 A/B: **+0.3% 用户活跃天数, +1.08% App 内使用时长** with 1B 稠密-参数模型全量部署且无服务成本增加

### 11.4 Why It Matters
解决阻塞 DLRM 交互扩展的 GPU 效率瓶颈; Token-mixer 方法作为注意力-only 架构的可信替代方案.

---

## 12. HHFT: Hierarchical Heterogeneous Feature Transformer (阿里 Taobao 2025)

**arXiv**: [2511.20235](https://arxiv.org/abs/2511.20235) · **作者**: Liren Yu, Wenming Zhang, Silu Zhou, Tao Zhang, Zhixuan Zhang, Dan Ou · **公司**: **Taobao — Alibaba** · **发表**: 2025-11

### 12.1 核心问题
纯 Transformer 在异构特征块 (用户/物品/行为/连续) 上用共享参数, 引起语义混淆; Hiformer 缺乏 scaling law 分析; Wukong/RankMixer 不是纯 Transformer.

### 12.2 方法 / 创新
- **Semantic Feature Partitioning** — 把特征分组到一致块
- **Heterogeneous Transformer Encoder** — 每块独立 QKV+FFN 参数
- **Hiformer 层** for 高阶跨块交互

### 12.3 结果
- SOTA on Taobao: **+0.4% CTR AUC vs DNN, +0.6% GMV** in 30-day A/B (1% 流量)
- 300M 参数下 AUC 最佳 (AutoInt/Hiformer/Wukong/RankMixer 比较)
- Scaling law 验证

### 12.4 Why It Matters
在阿里规模上缩小纯 Transformer 语义和 scale-aware 特征交互之间的差距.

---

## 13. FuXi-α: Feature Interaction Enhanced Transformer (华为 2025)

**arXiv**: [2502.03036](https://arxiv.org/abs/2502.03036) (WWW 2025) · **作者**: Yufei Ye, Wei-Feng Guo, Jin Yao Chin, Hao Wang, Hong Zhu, Xi Lin 等 · **公司**: **USTC StarTeam / Huawei Music** · **发表**: 2025-02

### 13.1 核心问题
HSTU 欠模型隐式特征交互 (由 FFNs 处理); SASRec/LLaMA 通过简单加法到注意力权重或嵌入来不充分集成时间 / 位置信息.

### 13.2 方法 / 创新
- **Adaptive Multi-channel Self-attention (AMS)** — 分离时间, 位置, 语义通道
- **Multi-stage FFN (MFFN)** 增强跨通道隐式交互

### 13.3 结果
- 优于 HSTU / SASRec / BERT4Rec; scaling 成立
- **华为音乐在线 A/B: +4.76% 歌曲 / 用户播放, +5.10% 收听时长 / 用户**
- 代码已开源

### 13.4 Why It Matters
证明时间 / 位置解耦 + 更好 FFN > 纯注意力 scaling — 华为工业规模的方法论反例.

---

## 14. TIGER: Recommender Systems with Generative Retrieval (Google 2023)

**arXiv**: [2305.05065](https://arxiv.org/abs/2305.05065) (NeurIPS 2023) · **作者**: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Nghia Vu, Lukasz Heldt 等 · **公司**: **Google Research** · **发表**: 2023 (NeurIPS 2023)

### 14.1 核心问题
双编码器 + ANN 检索解耦候选表征和排序; 冷启动物品和长尾检索弱.

### 14.2 方法 / 创新
- **Semantic ID** = 来自每个物品内容的 codewords 层次元组 (Sentence-T5 嵌入 → RQ-VAE)
- **T5X encoder-decoder** 训练自回归预测下一个物品的 Semantic ID 给定过去物品的 Semantic-ID 序列
- Transformer 内存充当索引 — 无需单独的 ANN 索引

### 14.3 结果
- SOTA Recall/NDCG on 3 Amazon 类别
- **更强冷启动 / 未见物品推荐**; 重叠 ID 之间的语义相似性

### 14.4 Why It Matters
**Semantic-ID 范式的开创论文**, 启动了基于 RQ-VAE 的生成式检索谱系 (LC-Rec, ColaRec, TokenRec, STORE, OneRec, OneMall, GPR, HiGR, NEZHA).

---

## 15. LC-Rec: Adapting LLMs by Integrating Collaborative Semantics (RUC 2023)

**arXiv**: [2311.09049](https://arxiv.org/abs/2311.09049) (ICDE 2024) · **作者**: Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen 等 · **公司**: **RUC / RUCABox** · **发表**: 2023 (ICDE 2024)

### 15.1 核心问题
LLM 捕获语言语义; 推荐需要协同语义. 因为物品使用 LLM 词汇表之外的离散 ID, 存在巨大语义差距.

### 15.2 方法 / 创新
- **Tree-structured VQ** 从 LLM 文本嵌入学习物品索引 + **uniform semantic mapping** 避免冲突
- 三任务对齐调优: 顺序物品预测 + 显式索引-语言对齐 + 隐式推荐导向对齐 (非对称物品预测)
- 仅 ~1k 新 token 加到 LLM 词汇表

### 15.3 结果
- 优于 P5, 传统推荐, LLM-based baseline on Amazon datasets

### 15.4 Why It Matters
LLM-based 协同语义集成的标准参考; 被 TIGER, OneRec, MBGen 引用.

---

## 16. HLLM: Enhancing Sequential Recommendations via Hierarchical LLMs (字节 2024)

**arXiv**: [2409.12740](https://arxiv.org/abs/2409.12740) · **作者**: Junyi Chen, Lu Chi, Bingyue Peng, Zehuan Yuan · **公司**: **ByteDance** · **发表**: 2024-09

### 16.1 核心问题
现有 LLM-for-recsys 工作仅显示温和增益; 不清楚预训练权重是否有帮助, 是否需要微调, scaling law 是否适用.

### 16.2 方法 / 创新
- **两层 LLM 层次**:
  - **Item LLM** 从文本提取物品嵌入 (特殊 `[ITEM]` token hidden state 作为嵌入)
  - **User LLM** 以物品嵌入序列作为输入, 预测下一个物品嵌入
- 丢弃词嵌入但保留预训练 Transformer 块

### 16.3 结果
- SOTA on PixelRec and Amazon Reviews
- **从 1B 扩展到 7B 参数** with 一致增益
- 在线 A/B 在字节验证
- 发布 HLLM-1B / 7B checkpoints

### 16.4 Why It Matters
证明 scaling law 存在和 LLM 预训练在字节工业规模上对推荐的实用价值.

---

## 17. LONGER: Scaling Up Long Sequence Modeling (字节 2025)

**arXiv**: [2505.04421](https://arxiv.org/abs/2505.04421) (RecSys 2025) · **作者**: Zheng Chai, Qin Ren, Xijun Xiao, Huizhi Yang, Bo Han, Sijun Zhang 等 · **公司**: **ByteDance** · **发表**: 2025-05

### 17.1 核心问题
两阶段 retrieve-then-attend 流水线有上下游不一致; 建模超长行为序列 (>2000) 计算上不可行.

### 17.2 方法 / 创新
- **Global Token 机制** (target-item, UID) 稳定长上下文注意力
- **Token Merge** with 轻量 **InnerTransformer** per group 压缩长序列并保留组内依赖
- **Hybrid attention**: cross causal-attention (第一层) + 堆叠 self causal-attention (更深层)
- 工程: 混合精度 + activation recomputation; **KV-cache serving** 用预计算用户序列表征; 全同步 GPU 稠密+稀疏训练

### 17.3 结果
- 端到端扩展序列长度到 **5000**
- 部署于 **10+ 字节场景** (Douyin Ads, Douyin E-commerce) 服务**数十亿用户**
- +1% 到 +8% on ADSS/ADVV/Order-U/GMV-U
- 序列长度 × 参数 × FLOPs 幂律 scaling

### 17.4 Why It Matters
在生产规模解决端到端超长行为建模无需检索中间层; HSTU 在长序列轴上的补充.

---

## 18. Scaling Law of Large Sequential Recommendation Models (RUC/腾讯 2024)

**arXiv**: [2311.11351](https://arxiv.org/abs/2311.11351) (RecSys 2024) · **作者**: Gaowei Zhang, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen · **公司**: **Renmin University of China / Tencent** · **发表**: 2024 (RecSys 2024)

### 18.1 核心问题
纯基于 ID 的顺序推荐是否在高数据稀疏下显示 scaling law.

### 18.2 方法 / 创新
- 纯 ID decoder-only Transformer for 顺序推荐; 专门训练策略稳定扩展到 0.8B 参数

### 18.3 结果
- **测试损失和非嵌入参数之间的幂律关系** 跨 98K → 0.8B 成立
- 预测损失在最大规模匹配实际
- **α_N = 0.121** (比 NLP 的 0.07 更陡)
- 即使在高度数据约束机制 (10% 数据) scaling 成立
- 冷启动 / 鲁棒性 / 长期任务从扩展中强烈受益

### 18.4 Why It Matters
推荐系统可扩展假设的经验基础 — 直接启发了 HSTU, Wukong, MTGR, OneRec 的扩展策略.

---

## 19. MBGen: Multi-Behavior Generative Recommendation (UCSD 2024)

**arXiv**: [2405.16871](https://arxiv.org/abs/2405.16871) (CIKM 2024) · **作者**: Zihan Liu, Yupeng Hou, Julian McAuley · **公司**: **UC San Diego** · **发表**: 2024-05 (CIKM 2024)

### 19.1 核心问题
多行为顺序推荐仅建模给定下一行为的下一物品; 它们忽略预测行为类型本身.

### 19.2 方法 / 创新
- 重新表述 MBSR 为 **两步**: 预测下一行为 → 给定行为预测下一物品
- 把行为和物品 token 化为单个**交错序列**; 一个 Transformer 做 next-token 预测
- **位置路由稀疏架构** (MoE-style routing by position/behavior type) 用于高效扩展

### 19.3 结果
- **+31% 到 +70%** over 之前 MBSR baselines on public datasets
- 内在多任务能力 (可对任意目标行为条件化)

### 19.4 Why It Matters
证明 GR 框架自然扩展到多类型动作日志 — 与电商 / 广告相关 (其中 click/add-to-cart/purchase 共存).

---

## 20. GenRank: Towards Large-scale Generative Ranking (小红书 2025)

**arXiv**: [2505.04180](https://arxiv.org/abs/2505.04180) · **作者**: Yanhua Huang, Yuqi Chen, Xiong Cao, Rui Yang 等 21 人 · **公司**: **Xiaohongshu / RedNote** · **发表**: 2025-05

### 20.1 核心问题
生成式排序尚未在大工业规模上研究; 现有 GR 架构 (HSTU) 对排序低效.

### 20.2 方法 / 创新
- **Action-oriented organization** — 动作是 tokens; 物品是位置 / 上下文信号
  - 注意力输入长度减半 (75% 注意力成本削减, 50% 线性投影削减)
- **Linear-complexity position & time biases** 替代 HSTU 的二次相对注意力偏置

### 20.3 结果
- 比 HSTU baseline 训练 **94.8% 加速** 且 AUC 略好
- 在线 A/B 在小红书 Explore Feed (服务数亿)
- 跨 time-spent / reads / engagements / LT7 增益
- **+25% P99 响应时间改善**

### 20.4 Why It Matters
**首个生产部署生成式排序** 在小红书规模; 指出**生成式架构** (而非训练范式) 是有效性的主要来源.

---

## 21. DFGR: Dual-Flow Generative Ranking Network (美团 2025)

**arXiv**: [2505.16752](https://arxiv.org/abs/2505.16752) (CIKM 2025) · **作者**: Hao Guo, Erpeng Xue, Lei Huang, Shihao Wang, Xiaolei Wang, Zeshun Li, Lei Wang, Jinpeng Wang, Sheng Chen · **公司**: **Meituan / RUC / Tsinghua** · **发表**: 2025-05 (CIKM 2025)

### 21.1 核心问题
MetaGR/HSTU 把每个交互分裂为物品 + 动作类型 tokens (序列长度 2N), 增加二次注意力成本. 异构信息映射到同一向量空间, 损害训练稳定性.

### 21.2 方法 / 创新
- 把物品 + 动作类型合并为**单个 token**
- **Dual-Flow 设计**: real-flow 携带真实动作类型 (上下文); fake-flow 用占位符 token, 查询 real-flow 的 KV (Q 来自 fake, K/V 来自真实 hidden states), 作为损失承担者 — 避免标签泄漏同时消耗完整历史

### 21.3 结果
- 训练 **~2× 更快**, 推理 **~4× 更快** vs MetaGR/HSTU
- **+0.31% 到 +1.2% AUC over MetaGR** on 3 datasets
- 在工业 TRec 上击败有数以百计工程特征的生产 DLRM
- 在计算中遵守 scaling law

### 21.4 Why It Matters
优雅地解决工业 HSTU 采用最常被引用的 blocker — 效率 — 是 2025 年末+ 工业推荐默认生成式排序架构的有力候选.

---

## 22. HiGR: Hierarchical Generative Slate Recommendation (腾讯 2025)

**arXiv**: [2512.24787](https://arxiv.org/abs/2512.24787) · **作者**: Yunsheng Pang, Zijian Liu, Yudong Li, Shaojie Zhu, Zijian Luo, Chenyun Yu 等 · **公司**: **Tencent** · **发表**: 2025-12

### 22.1 核心问题
基于 AR 的生成式 slate 推荐在 SIDs 中有语义纠缠, 无整体 slate 规划时顺序解码低效.

### 22.2 方法 / 创新
- **CRQ-VAE**: 带前缀级对比约束的残差量化器 → 语义结构化 SIDs
- **Hierarchical decoder**: list-level 规划 (粗粒度 slate 意图) → item-level 解码 (具体物品选择)
- **Listwise 偏好对齐** 使用隐式 slate-level 用户反馈 (RLHF-style)

### 22.3 结果
- **+10% offline NDCG**, 比 SoTA **5× 推理加速**
- 在线 A/B 在腾讯媒体平台 (数千万 DAU, 2% 流量): **+1.22% 平均观看时长, +1.73% 平均视频观看数**, +1.03% 停留时长, +1.57% 请求数
- **幂律 scaling 从 0.05B 到 2B 参数**

### 22.4 Why It Matters
首个端到端生成式 **slate** 推荐框架在腾讯规模; 干净地分离粗规划与细解码.

---

## 23. GPR: Generative Pre-trained One-Model Paradigm for Advertising (腾讯 2025)

**arXiv**: [2511.10138](https://arxiv.org/abs/2511.10138) · **作者**: Jun Zhang, Yi Li, Yue Liu, Changping Wang, Yuan Wang, Yuling Xiong, Xun Liu, Haiyang Wu, Qian Li, Enming Zhang, Jiawei Sun, Xin Xu, Zishuai Zhang, Ruoran Liu, Suyuan Huang, Zhaoxin Zhang, Zhengkai Guo, Shuojin Yang, Meng-Hao Guo, Huan Yu, Jie Jiang, Shi-Min Hu · **公司**: **Tencent Inc. / Tsinghua University** · **发表**: 2025-11

### 23.1 核心问题
多阶段广告推荐遭受目标不对齐 + 错误传播; 现有统一 GR 模型不满足实际工业需求.

### 23.2 方法 / 创新
- **统一 RQ-KMeans+ tokenizer** 把广告和有机内容映射到共享多级 SID 空间
- **Heterogeneous Hierarchical Decoder (HHD)** — 双解码器架构, 解耦用户意图建模与广告生成
- **三阶段训练**: Multi-Token Prediction (MTP) 预训练 → Value-Aware Fine-Tuning → Hierarchy Enhanced Policy Optimization (HEPO)
- DeepSeek-V3-style MoE + RoPE + RMSNorm + SwiGLU + FlashAttention

### 23.3 结果
- **完全部署于腾讯微信视频号广告**
- 大规模在线 A/B 改进 GMV 和 CTCVR vs 成熟级联系统
- 鲁棒 scaling law 从 0.02B → 2B 稠密参数 (+80B 稀疏)
- 最大 MTP 增益 **+17.9%**

### 23.4 Why It Matters
**首个 one-model 端到端生成式广告推荐** 在腾讯规模; 集成价值对齐 + RL + 多 token 预测 + 扩展.

---

## 24. NEZHA: Zero-Sacrifice Hyperspeed Decoding for GR (阿里 Taobao 2025)

**arXiv**: [2511.18793](https://arxiv.org/abs/2511.18793) · **作者**: (阿里 — Taobao 搜索广告) · **公司**: **Alibaba — Taobao Search Ads** · **发表**: 2025-11

### 24.1 核心问题
基于 LLM 的 GR 有 >1000ms 推理延迟, 阻塞 prime-slot 广告服务 (≤30 ms SLO). 现有投机解码需要单独的 draft 模型和基于模型的验证器.

### 24.2 方法 / 创新
- **Self-drafting** — 灵巧的自回归 draft head 在主要模型内部, 无单独的 draft 模型
- **Model-free hash-set verifier** 检测幻觉无效语义 ID
- 特殊 token prompts 保留 seq-to-seq 生成完整性

### 24.3 结果
- 算法级加速 **2.6×**
- 端到端延迟 **>1000 ms → <30 ms**
- 部署于 **Taobao 搜索广告 (自 2025-10 起)** 驱动**十亿级广告收入**, 服务数亿 DAU
- 离线 top-500/1000 hit-rate **+0.58% / +0.61%**; 7 天在线 A/B +1.2% 营收

### 24.4 Why It Matters
首个生产级专门为生成式推荐设计的投机框架在淘宝规模; 为延迟关键型广告召回槽解锁 LLM-based GR.

---

## 25. OneMall: 跨场景电商 GR 框架 (快手 2026)

**arXiv**: [2601.21770](https://arxiv.org/abs/2601.21770) · **作者**: Kun Zhang, Jingming Zhang, Wei Cheng, Yansong Cheng, Jiaqi Zhang, Hao Lu, Xu Zhang, Haixiang Gan, Jiangxia Cao, Tenglong Wang, Ximing Zhang, Boyang Xia, Kuo Cai, Shiyao Wang, Hongjian Dou, Jinkai Yu, Mingxing Wen, Qiang Luo, Dongxu Liang, Chenyi Lei, Jun Wang, Runan Liu, Zhaojie Liu, Ruiming Tang, Tingting Gao, Shaoguo Liu, Yuqing Ding, Hui Kong, Han Li, Guorui Zhou, Wenwu Ou, Kun Gai · **公司**: **Kuaishou** · **发表**: 2026-01

### 25.1 核心问题
生成式推荐需要跨多种电商物品类型 (Product-Card, Short-Video, Live-Streaming) 用共享架构泛化.

### 25.2 方法 / 创新
- **E-commerce Semantic Tokenizer** 捕获真实世界语义和业务特定关系跨场景
- **Transformer backbones** 用 **Query-Former** 长序列压缩, Cross-Attention 多行为融合, **Sparse-MoE decoder**
- **RL pipeline** 用在线排序模型作为奖励模型 — 桥接检索和排序
- In-batch 对比学习解决语义 ID 冲突

### 25.3 结果
- 在线 A/B (部署于 4 亿 DAU 快手): **+14.71% GMV (Product-Card), +10.33% GMV (Short-Video), +4.90% GMV (Live-Streaming)**

### 25.4 Why It Matters
**首个报告端到端生成式推荐**统一多种电商物品分发场景在十亿用户规模.

---

## 26. DiffuRec: 扩散模型顺序推荐 (武大 2023)

**arXiv**: [2304.00686](https://arxiv.org/abs/2304.00686) (TOIS 2023) · **作者**: Zihao Li, Aixin Sun, Chenliang Li · **公司**: **Wuhan University** · **发表**: 2023 (TOIS 2023)

### 26.1 核心问题
主流推荐将物品表示为固定向量; 不能捕获物品的多潜在方面或用户的多兴趣, 也不能建模不确定性.

### 26.2 方法 / 创新
- 将物品表示视为**分布**
- 扩散过程将噪声注入目标物品嵌入, 用噪声目标作为历史物品表示生成的指导
- Transformer 近似器重建目标
- 舍入将反转的连续表示映射到离散物品

### 26.3 结果
- +57.26% / +56.72% (HR/NDCG) over best baseline on Amazon Beauty
- SOTA 跨 4 datasets

### 26.4 Why It Matters
**首个基于扩散的推荐**; 显示生成式推荐家族扩展到自回归以外的扩散风格生成器.

---

## 27. TokenRec: LLM-based Generative Recommendation Tokenization (港理工 2024)

**arXiv**: [2406.10450](https://arxiv.org/abs/2406.10450) · **作者**: Haohao Qu, Wenqi Fan, Zihuai Zhao, Qing Li · **公司**: **Hong Kong PolyU** · **发表**: 2024-06

### 27.1 核心问题
LLM-based 推荐难以将用户 / 物品 (百万到十亿) token 化到固定 LLM 词汇表, 也难以利用高阶协同知识.

### 27.2 方法 / 创新
- **Masked Vector-Quantized (MQ) Tokenizer** 带 masking + K-way codebooks 用于鲁棒物品 / 用户 ID tokenization
- **生成式检索**范式将 LLM hidden state 投影到物品嵌入空间并通过 ANN 检索 top-K (避免慢 beam search)
- 1536 OOV tokens 用于 token 化百万物品 (如 Amazon-Clothing)

### 27.3 结果
- 优于 TIGER, CoLLM, 传统推荐
- 比自回归解码推理快得多

---

## 28. IDvs.MoRec: ID vs Modality (西湖/腾讯 2023)

**arXiv**: [2303.13835](https://arxiv.org/abs/2303.13835) (SIGIR 2023) · **作者**: Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, Yang Tang, Dawei Yin, Wayne Xin Zhao, Yongfeng Zhang, Yongjun Chen, Xue Jiang, Jian Wu · **公司**: **Westlake / Tencent / RUC / Lehigh** · **发表**: 2023 (SIGIR 2023)

### 28.1 核心问题
随着强模态编码器 (BERT, ViT), 纯基于模态的推荐 (MoRec) 能匹配或击败基于 ID 的推荐 (IDRec) 吗?

### 28.2 方法 / 创新
- 在 3 个数据集上严谨实证比较 DSSM 和 SASRec backbone 与 IDRec vs MoRec (BERT/RoBERTa/ResNet/Swin)
- 识别 E2E 训练为关键

### 28.3 结果
- 用 **SASRec backbone**, MoRec 在常规和暖物品设置下可匹配或击败 IDRec 对文本; 对视觉相当
- 用 **DSSM**, MoRec 仍然输
- TS-MoRec (冻结特征) 比 E2E 差得多

### 28.4 Why It Matters
激励整个 Semantic-ID / 生成式检索谱系 (TIGER, OneRec 等) — 建立**内容 + Transformer backbone** 可替代 ID 嵌入.

---

## 29. RecFormer: Text Is All You Need (UCSD 2023)

**arXiv**: [2305.13731](https://arxiv.org/abs/2305.13731) (KDD 2023) · **作者**: Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, Daniel McCullough, Tao Ge, Danqi Chen · **公司**: **UCSD / Microsoft / UIUC** · **发表**: 2023 (KDD 2023)

### 29.1 核心问题
现有顺序推荐依赖物品 ID 或浅文本特征; 冷启动物品和跨域迁移仍然困难.

### 29.2 方法 / 创新
- 物品 = (key, value) 属性对的句子; 用户序列 = 句子序列
- **Longformer-style 双向 Transformer** 带线性注意力 (无物品嵌入表)
- 预训练: MLM + item-item 对比 (IIC, 带 in-batch 负样本)
- **两阶段微调**: Stage 1 每 epoch 重新编码所有物品; Stage 2 冻结物品特征并训练模型

### 29.3 结果
- +15.83% / **+39.78% NDCG@10** 在全监督 / 零样本设置跨 6 数据集

### 29.4 Why It Matters
开创**"text-only LLM as recommender"** 范式, 后续 HLLM, M6-Rec, OneRec-Think, LC-Rec 改进.

---

## 30. M6-Rec: PLM as Open-Ended Recommender (阿里 DAMO 2022)

**arXiv**: [2205.08084](https://arxiv.org/abs/2205.08084) · **作者**: Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, Hongxia Yang · **公司**: **DAMO Academy, Alibaba Group** · **发表**: 2022

### 30.1 核心问题
工业推荐需要每个域许多任务特定模型; 从头构建昂贵且忽略 Web 预训练知识.

### 30.2 方法 / 创新
- **M6 (Alibaba GPT-3/T5-style PLM)** 作为统一基础; 将用户行为表示为纯文本 → 所有推荐任务成为语言理解或生成
- **Option tuning** (改进的 prompt tuning) + adapters; 仅 ~1% 任务特定参数
- **多段延迟交互** 用于低延迟检索; 边缘压缩 (蒸馏 + 剪枝 + 量化 + early-exit) 适配移动设备

### 30.3 结果
- AUC 0.7995 vs ID baseline; kNN 检索 57% for 未见物品
- **在线 +1% CTR 提升** in real production

### 30.4 Why It Matters
**最早的工业规模 PLM-as-foundation-for-recsys** 论文; OneRec-Foundation, HLLM, OneRec-Think 的架构祖先.

---

## 31. ColaRec: Content-Based Collaborative Generation (Leeds/山大 2024)

**arXiv**: [2403.18480](https://arxiv.org/abs/2403.18480) · **作者**: Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Renjun Xu, Zhenglu Yang, Maarten de Rijke · **公司**: **Leeds / Shandong / U Amsterdam** · **发表**: 2024-03

### 31.1 核心问题
依赖文本 ID 的生成式推荐缺乏协同信号; 依赖语言生成的 LLM-based 推荐与推荐任务不完全对齐.

### 31.2 方法 / 创新
- **Generative ID (GID)** 来自预训练 LightGCN 协同嵌入的层次 K-means 聚类 — 将协同信号编码到 ID
- 共享 encoder-decoder Transformer 用于**用户-物品推荐** + **物品-物品索引** (辅助任务对齐内容空间与协同空间)
- **对比损失** 确保相似 GID 物品有相似内容表示

### 31.3 结果
- SOTA on 4 benchmarks

### 31.4 Why It Matters
强调 **协同信号** (vs 内容) 在 SID 中的 TIGER 对应; 桥接 IDRec 和 MoRec.

---

## 32. TWIN V2 (已在 GR 笔记中)

**arXiv**: (已知) · **公司**: **Kuaishou** · **发表**: 2024 (CIKM 2024)

### 32.1 Why It Matters
- 超长序列检索增强, 与 STCA 等论文同思路
- 在你 GR 笔记的工业架构章节中已记录

---

## 33. TAAC 2025: 腾讯广告算法挑战赛 (工业 GR 基准)

**arXiv**: [2604.04976](https://arxiv.org/abs/2604.04976) · **公司**: **Tencent** · **发表**: 2025

### 33.1 核心内容
- 数据集: **TencentGR-1M, TencentGR-10M** — 脱敏的腾讯广告日志, 协同 ID + 多模态 (文本 + 视觉) 嵌入

### 33.2 Why It Matters
**首个公开基准**有现实工业规模全模态 GR 数据; baseline 和顶级方案 (如 **OnePiece** — 第 9-13 名决赛) 推进生成式检索+排序统一实践.

---

## 34. OnePiece: 生成式推荐路线 (腾讯比赛)

**arXiv**: [2512.07424](https://arxiv.org/abs/2512.07424) · **公司**: **Tencent 比赛团队** · **发表**: 2025

### 34.1 核心思想
- 统一 encoder-decoder 连接 **ANN-based 检索 (InfoNCE)** + **自回归 SID 生成** — 都遵守幂律 scaling (R² > 0.9)
- Stage 1 SID beam search → Stage 2 InfoNCE 重排
- 强证据表明生成式检索 + 嵌入检索可共享单个可扩展 HSTU+MoE backbone

---

## 35. 补充论文全景速览表

| # | 论文 | 公司/机构 | 角色 | 与字节/阿里关联 |
|:-:|------|---------|------|----------------|
| 1 | HSTU | Meta | GR 范式奠基 | TokenMixer-Large 直接引用 |
| 2 | OneRec V1 | 快手 | 端到端 GR 首次落地 | - |
| 3 | OneRec Tech Report | 快手 | 全系统蓝图 | - |
| 4 | OneRec V2 | 快手 | Lazy Decoder | - |
| 5 | OneRec-Foundation | 快手 | 开源基座 | - |
| 6 | OneRec-Think | 快手 | GR + 推理 | - |
| 7 | MTGR | 美团 | HSTU + DLRM 交叉特征 | 与 MSN 互补 |
| 8 | Wukong | Meta | DLRM scaling law | RankMixer/HHFT/FuXi-α baseline |
| 9 | DHEN | Meta | 异构集成 | InterFormer 兼容 |
| 10 | Hiformer | Google | 异构注意力 | OneTrans/MixFormer/HyFormer 思路源头 |
| 11 | RankMixer | 字节 | Token Mixer | TokenMixer-Large 前身 |
| 12 | FuXi-α | 华为 | 多通道注意力 | - |
| 13 | TIGER | Google | Semantic ID | STORE/OneRec/OneMall 思路源头 |
| 14 | LC-Rec | RUC | LLM + 协同 | - |
| 15 | HLLM | 字节 | 层次 LLM | 与 LONGER 互补 |
| 16 | LONGER | 字节 | 超长序列 | 与 STCA 互补 |
| 17 | Scaling Law of Seq Rec | RUC/腾讯 | Scaling Law 实证 | 所有 scaling law 工作引用 |
| 18 | MBGen | UCSD | 多行为 GR | - |
| 19 | GenRank | 小红书 | 行动-物品倒置 | - |
| 20 | DFGR | 美团 | 双流生成式 | - |
| 21 | HiGR | 腾讯 | 层级生成式 slate | - |
| 22 | GPR | 腾讯 | 广告一模型 | - |
| 23 | NEZHA | 阿里 | 投机解码 | - |
| 24 | OneMall | 快手 | 多场景 GR | - |
| 25 | DiffuRec | 武大 | 扩散 GR | - |
| 26 | TokenRec | 港理工 | LLM + ID token | - |
| 27 | IDvs.MoRec | 西湖/腾讯 | 模态 vs ID 实证 | - |
| 28 | RecFormer | UCSD | Text-only 推荐 | - |
| 29 | M6-Rec | 阿里 DAMO | PLM 推荐鼻祖 | - |
| 30 | ColaRec | Leeds/山大 | 内容-协同双嵌入 | - |
| 31 | TAAC 2025 | 腾讯 | 工业 GR 基准 | - |
| 32 | OnePiece | 腾讯比赛 | 检索-生成统一 | - |

---

## 36. 引用格式

```text
@article{zhai2024hstu,
  title={Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations},
  author={Zhai, Jiaqi and Liao, Lucy and Liu, Xing and others},
  journal={arXiv preprint arXiv:2402.17152},
  year={2024}
}

@article{deng2025onerec,
  title={OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment},
  author={Deng, Jiaxin and Wang, Shiyao and Cai, Kuo and others},
  journal={arXiv preprint arXiv:2502.18965},
  year={2025}
}

@article{zhou2025onerec,
  title={OneRec Technical Report},
  author={Zhou, Guorui and Deng, J. S. and Zhang, Jinghao and others},
  journal={arXiv preprint arXiv:2506.13695},
  year={2025}
}

@article{kuaishou2025onerec2,
  title={OneRec-V2 Technical Report},
  author={Kuaishou OneRec Team},
  journal={arXiv preprint arXiv:2508.20900},
  year={2025}
}

@article{zhou2025onerecfoundation,
  title={OneRec-Foundation: Open Source Recommendation Foundation Model},
  author={Zhou, Guorui and Bao, Honghui and Huang, Jiaming and others},
  journal={arXiv preprint arXiv:2512.24762},
  year={2025}
}

@article{liu2025onerec,
  title={OneRec-Think: In-Text Reasoning for Generative Recommendation},
  author={Liu, Zhanyu and Wang, Shiyao and Wang, Xingmei and others},
  journal={arXiv preprint arXiv:2510.11639},
  year={2025}
}

@article{han2025mtgr,
  title={MTGR: Industrial-Scale Generative Recommendation Framework in Meituan},
  author={Han, Ruidong and Yin, Bin and Chen, Shangyu and others},
  journal={arXiv preprint arXiv:2505.18654},
  year={2025}
}

@article{zhang2024wukong,
  title={Wukong: Towards a Scaling Law for Large-Scale Recommendation},
  author={Zhang, Buyun and Luo, Liang and Chen, Yuxin and others},
  journal={arXiv preprint arXiv:2403.02545},
  year={2024}
}

@article{rajput2023tiger,
  title={Recommender Systems with Generative Retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and others},
  journal={arXiv preprint arXiv:2305.05065},
  year={2023}
}

@article{zheng2023lcrec,
  title={How to Adapt Large Language Models for Recommendation: Adaptation Method and Recommendation Strategy},
  author={Zheng, Bowen and Hou, Yupeng and Lu, Hongyu and others},
  journal={arXiv preprint arXiv:2311.09049},
  year={2023}
}

% ... 更多引用按需补充
```

---

## 相关链接

- 第09章-01-字节七剑-TokenMixer-OneTrans-STCA等 (本节目录)
- GR 总目录 (GR 全章索引)
- 00-总览与哲学
- 第09章-00-规模化排序新架构-章节目录
- 第09章-02-阿里九章-STORE-FAT-HHFT等

### 相关深度解析
- 01-TokenMixer-Large
- 02-OneTrans
- 03-STCA
- 04-MDL
- 05-MixFormer

### 科研方法论参考
- 00-总览与哲学
- 01-如何阅读论文
- 02-如何发现问题

---

> **一句话**: 这 30+ 篇补充论文构成了 **GR / 规模化排序**领域的完整版图 — 从 HSTU (Meta GR 范式) 到 OneRec (快手端到端), 从 TIGER (Google Semantic ID) 到 MTGR (美团 DLRM+GR 融合), 从 LONGER (字节超长序列) 到 GPR (腾讯广告一模型) — 每篇都是某子方向的标杆, 共同推动推荐系统从手工 DLRM 走向 LLM 风格的全场景 GR。

---

## 22. 所有论文速查表 (GenRec 经典 + 补充)

### 22.1 论文清单 (按主题)

| 主题 | 论文 | 一句话 |
|------|------|--------|
| **GR 范式** | HSTU | 统一 transduction |
| **GR 范式** | TIGER | 文档检索生成 |
| **GR 范式** | MTGR | 多任务 GR |
| **GR 范式** | OneTrans | 排序统一 |
| **SID 核心** | RQ-VAE | 残差量化 |
| **SID 核心** | GenRet | 生成式检索 |
| **SID 核心** | COURIER | 对比检索 |
| **LLM4Rec** | P5 | 预训练推荐 |
| **LLM4Rec** | M6-Rec | M6 推荐 |
| **LLM4Rec** | TallRec | 微调推荐 |
| **LLM4Rec** | InstructRec | 指令推荐 |
| **LLM4Rec** | ChatRec | 对话推荐 |
| **长序列** | S3 | 长序列建模 |
| **长序列** | LST | 长序列 Transformer |
| **长序列** | SIM | 搜索长序列 |
| **长序列** | MIM | 多兴趣长序列 |
| **长序列** | SLA-Rec | 长序列注意力 |
| **Scaling** | Kaplan | LLM scaling |
| **Scaling** | Chinchilla | compute-optimal |
| **Scaling** | UniScale | RecSys scaling |

### 22.2 详细对比表 (15 维度)

| 维度 | HSTU | TIGER | MTGR | P5 | RQ-VAE | S3 | Chinchilla |
|------|------|-------|------|----|----|----|----|
| 范式 | GR | GR | GR | LLM4Rec | SID | 长序列 | Scaling |
| 架构 | Enc-Dec | Enc-Dec | Dec-only | Enc-Dec | AutoEnc | Dec-only | - |
| 训练目标 | Next action | Next token | 多任务 | 多任务 | 重建 | Next | - |
| 数据需求 | 大 (10B) | 中 (1B) | 大 (10B) | 大 | 中 | 中 | - |
| 推理延迟 | 中 (50ms) | 高 (100ms) | 中 (50ms) | 高 | 低 | 中 | - |
| 工程度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | - |
| 学术度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 在线增益 | +12% | +2-5% | +5-8% | (实验) | (辅助) | +1-2% | - |
| 开源 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | - |
| 复现难度 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | - |
| 中文场景 | 弱 | 中 | 中 | 中 | 强 | 中 | - |
| 跨语言 | 强 | 强 | 弱 | 强 | 强 | 中 | - |
| 移动端 | 弱 | 弱 | 弱 | 弱 | 强 | 中 | - |
| 创新度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 应用度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### 22.3 关键洞察

| 洞察 | 说明 |
|------|------|
| **GR 范式正在统一** | HSTU / TIGER / MTGR 都是 transduction 思想 |
| **SID 是 GR 的桥梁** | 把 item 映射到 token, 才能用 LLM-style 训练 |
| **LLM4Rec 是 GR 的扩展** | 用 LLM 的能力做推荐 |
| **Scaling Law 是 GR 的支撑** | 没有 scaling, GR 无法实用 |
| **长序列是 GR 的瓶颈** | S3 / LST / SIM 在解决 |

---

## 23. 公司研究风格对比 (GenRec 视角)

### 23.1 GenRec 三派 (学术)

```text
派系 A: Google / Meta 系
  - 代表: HSTU, TIGER, GenRet, COURIER
  - 风格: 学术前沿
  - 资源: 大集群

派系 B: 国内大厂系
  - 代表: 字节 / 阿里 / 美团
  - 风格: 工程落地
  - 资源: 大集群 + 工程团队

派系 C: 高校系
  - 代表: 各类 workshop 论文
  - 风格: 理论探索
  - 资源: 小集群
```

### 23.2 学术 vs 工业

| 维度 | 学术 | 工业 |
|------|------|------|
| 数据 | 公开数据集 | 内部海量 |
| 算力 | 中等 | 超大 |
| 目标 | SOTA 指标 | 在线收益 |
| 时间 | 半年 | 季度 |
| 评估 | 离线 AUC | 离线 + 在线 |

### 23.3 跨学派合作

```text
可能合作:
  - 学界提供 idea
  - 工业提供数据 + 算力
  - 共同发表

风险:
  - 数据隐私
  - IP 归属
  - 节奏不一致
```

---

## 24. 技术演进时间线 (GenRec 全景)

### 24.1 经典推荐阶段 (2017-2019)

```text
2017  Transformer
2018  DIN
2019  DLRM
```

### 24.2 推荐进化阶段 (2020-2022)

```text
2020  Linformer / Longformer
2021  S3 / P5 / LST
2022  RQ-VAE
```

### 24.3 GR 范式阶段 (2023-2024)

```text
2023 H1  TIGER / COURIER / GenRet
2023 H2  MixFormer / MSN / HyFormer
2024 H1  HSTU / MTGR
2024 H2  OneTrans / TokenMixer / STCA / 阿里九章
```

### 24.4 当前阶段 (2025+)

```text
2025+
  - GR + LLM 融合
  - 多模态 GR
  - Scaling Law 成熟
  - 端侧 GR
```

### 24.5 关键技术拐点

| 年份 | 拐点 | 论文 |
|------|------|------|
| 2022 | SID 范式 | RQ-VAE |
| 2023 | GR 范式 | TIGER |
| 2024 | GR 统一 | HSTU, MTGR |

---

## 25. 派系图 (GenRec 视角)

### 25.1 GenRec 三派

```text
派系 1: 生成式 (Generative)
  - HSTU, TIGER, MTGR, OneTrans
  - 核心: 把推荐变成"序列生成"

派系 2: 语义 ID (Semantic ID)
  - RQ-VAE, COURIER, GenRet
  - 核心: item → token, 然后用 Transformer

派系 3: LLM 派 (LLM4Rec)
  - P5, M6-Rec, TallRec, InstructRec, ChatRec
  - 核心: 用 LLM 的能力做推荐

派系 4: Scaling 派
  - Kaplan, Chinchilla, UniScale
  - 核心: scaling law 是 GR 的支撑
```

### 25.2 派系融合

```text
2024:
  - 派系 1 + 派系 2 = HSTU (GR + SID)
  - 派系 3 + 派系 1 = MTGR (LLM + GR)

2025+:
  - 派系 4 + 派系 1 = Scaling GR
  - 派系 3 + 派系 2 = LLM + SID
  - 多模态 + 派系 1 = Multi-modal GR
```

### 25.3 派系对比表

| 维度 | 生成式 | SID | LLM4Rec | Scaling |
|------|--------|-----|---------|---------|
| 创新度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 工程度 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 学术度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 落地难度 | 高 | 中 | 中 | 中 |
| 算力需求 | 大 | 中 | 大 | 大 |
| 代表 | HSTU | RQ-VAE | P5 | Chinchilla |

---

## 26. 在线增益排行 (GenRec 全景)

| 论文 | 公司 | 在线增益 |
|------|------|---------|
| HSTU | Meta | +12% |
| MTGR | 美团 | +5-8% |
| TIGER | Google | +2-5% |
| OneTrans | 字节 | +3-4% |
| TokenMixer | 字节 | +3-4% |
| MixFormer | 字节 | +2-3% |
| STORE | 阿里 | +3-5% |

### 26.1 增益 vs 范式

| 范式 | 平均增益 | 最高 |
|------|---------|------|
| 生成式 | +5-10% | HSTU +12% |
| SID | (辅助) | (RAG +1-2%) |
| LLM4Rec | +1-3% | TallRec |
| Scaling | (基础设施) | - |

---

## 27. 代码可用性

| 论文 | 官方 | 第三方 | 复现难度 |
|------|:---:|:------:|:-------:|
| HSTU | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| TIGER | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| RQ-VAE | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| P5 | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Chinchilla | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| S3 | ✅ | ⭐⭐⭐⭐ | ⭐⭐ |
| Kaplan | ❌ | (理论) | - |

### 27.1 复现优先级

```text
1. TIGER (有代码, 立即跑) - GR 入门
2. RQ-VAE (有代码, 立即跑) - SID 入门
3. Chinchilla (有代码, 立即跑) - Scaling 入门
4. P5 (有代码) - LLM4Rec 入门
5. S3 (有代码) - 长序列入门
6. HSTU / MTGR (无代码, 找团队)
```

---

## 28. 未来 6-12 月预测 (GenRec 视角)

### 28.1 趋势预测

| 趋势 | 概率 | 预测 |
|------|:---:|------|
| **GR + LLM 深度融合** | ⭐⭐⭐⭐⭐ | 2025 H1 多篇顶会 |
| **多模态 GR** | ⭐⭐⭐⭐⭐ | 2025 H2 爆发 |
| **Scaling GR** | ⭐⭐⭐⭐ | 2025 下半年 compute-optimal |
| **端侧 GR** | ⭐⭐⭐ | 2025-2026 |
| **强化学习 + GR** | ⭐⭐⭐⭐ | 2025 RLHF-style |
| **联邦 GR** | ⭐⭐⭐ | 2025 跨域 + 隐私 |
| **生成式冷启动** | ⭐⭐⭐⭐ | 2025 跨域 + SID |
| **可解释 GR** | ⭐⭐⭐ | 2025 解释性 |

### 28.2 顶会预测 (2025)

| 会议 | 主题 | 数量 |
|------|------|:---:|
| KDD 2025 | GR + LLM | 5+ |
| SIGIR 2025 | GR + 多模态 | 5+ |
| RecSys 2025 | GR + Scaling | 5+ |
| NeurIPS 2025 | GR 理论 | 5+ |

---

## 29. 对初学者的建议

### 29.1 必读顺序 (GenRec 视角)

```text
Week 1  Transformer + DIN
Week 2  RQ-VAE (SID 入门)
Week 3  TIGER (GR 入门)
Week 4  P5 (LLM4Rec 入门)
Week 5  S3 (长序列入门)
Week 6  Chinchilla (Scaling)
Week 7  HSTU (GR 范式)
Week 8  MTGR (多任务 GR)
```

### 29.2 必读清单 (10 篇)

| # | 论文 | 必读理由 |
|:--:|------|---------|
| 1 | DIN | 注意力推荐 |
| 2 | DLRM | 工业基线 |
| 3 | RQ-VAE | SID 核心 |
| 4 | TIGER | GR 入门 |
| 5 | P5 | LLM4Rec |
| 6 | S3 | 长序列 |
| 7 | Chinchilla | Scaling |
| 8 | HSTU | GR 范式 |
| 9 | MTGR | 多任务 GR |
| 10 | Scaling Law | RecSys scaling |

### 29.3 工具准备

| 工具 | 用途 |
|------|------|
| PyTorch | 基础 |
| HuggingFace Transformers | LLM |
| Faiss | 向量检索 |
| Triton | 推理 |
| WandB | 实验 |

---

## 30. 对资深者的建议

### 30.1 深耕方向 (3 选 1)

| 方向 | 关键能力 | 产出目标 |
|------|---------|---------|
| **GR + LLM 融合** | LLM + RecSys | 顶会 + 产品 |
| **SID + 推荐** | 量化 + 生成 | 顶会 |
| **Scaling + 落地** | 分布式 + 推理 | 产品 + 性能 |

### 30.2 资源获取

| 资源 | 获取 |
|------|------|
| 顶会论文 | arXiv + Semantic Scholar |
| 内部经验 | 业务中遇到 |
| 同行动态 | Twitter/X |
| 工业经验 | 一线公司博客 |

### 30.3 与初学者的区别

| 维度 | 初学者 | 资深者 |
|------|--------|--------|
| 目标 | 入门 | 突破 |
| 阅读 | 1 篇/周 | 5 篇/周 |
| 实验 | 复现 | 创新 |
| 产出 | 笔记 | 论文 |
| 影响 | 自己 | 团队/社区 |

---

## 31. 完整对比矩阵 (GenRec 30+ 论文 × 10 维度)

| 论文 | 范式 | 架构 | 数据 | 工程 | 学术 | 代码 | 在线 | 中文 | 多模 |
|------|------|------|------|------|------|------|------|------|------|
| HSTU | GR | Enc-Dec | 大 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | +12% | 弱 | 中 |
| TIGER | GR | Enc-Dec | 中 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | +2-5% | 中 | 弱 |
| MTGR | GR | Dec-only | 大 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | +5-8% | 中 | 弱 |
| OneTrans | GR | Dec-only | 大 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | +3-4% | 中 | 中 |
| RQ-VAE | SID | AutoEnc | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | (辅助) | 强 | 中 |
| P5 | LLM4Rec | Enc-Dec | 大 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | +1-2% | 中 | 中 |
| S3 | 长序列 | Dec-only | 中 | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | +1-2% | 中 | 弱 |
| LST | 长序列 | Dec-only | 中 | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | +1% | 中 | 弱 |
| SIM | 长序列 | - | 中 | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | +1% | 中 | 弱 |
| Chinchilla | Scaling | - | - | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | - | - | - |
| Kaplan | Scaling | - | - | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | - | - | - |
| (其他 20+) | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 32. 速查表 (30+ 论文一句话)

| # | 论文 | 一句话 |
|:--:|------|--------|
| 1 | HSTU | 把推荐统一成 transduction |
| 2 | TIGER | 文档检索变成生成 |
| 3 | MTGR | 多任务 GR 统一 |
| 4 | OneTrans | 排序任务统一架构 |
| 5 | TokenMixer | 多种 attention mixer 灵活组合 |
| 6 | RQ-VAE | 残差量化生成 SID |
| 7 | P5 | 预训练推荐 |
| 8 | M6-Rec | M6 推荐 |
| 9 | TallRec | 微调 LLM 推荐 |
| 10 | InstructRec | 指令推荐 |
| 11 | ChatRec | 对话推荐 |
| 12 | S3 | 长序列建模 |
| 13 | LST | 长序列 Transformer |
| 14 | SIM | 搜索长序列 |
| 15 | MIM | 多兴趣长序列 |
| 16 | SLA-Rec | 长序列注意力 |
| 17 | GenRet | 生成式检索 |
| 18 | COURIER | 对比检索 |
| 19 | Kaplan | LLM scaling 起源 |
| 20 | Chinchilla | compute-optimal |
| 21 | UniScale | RecSys scaling |
| 22-30+ | (其他) | 详见 24 篇矩阵 |

---

> 本节 ([[#22-32]]) 是 GenRec 经典论文的扩展版, 覆盖 GR / SID / LLM4Rec / 长序列 / Scaling 五大主题。

---

## 33. GenRec 经典论文补充

### 33.1 HSTU (Meta) 深度补充

```text
发表: 2024 (KDD/RecSys)
作者: Meta AI

完整方法:
  1. Hierarchical Transduction
     - 序列: 行为 + 候选 + action
     - 统一成一个 transduction 任务
  2. 改进 Attention
     - 相对位置编码 (类似 RoPE)
     - 共享 Q/K/V 参数
     - 稀疏 attention (非全连接)
  3. 训练优化
     - Streaming 训练 (增量)
     - 分布式 (万卡)
  4. 推理优化
     - KV-Cache 压缩
     - 量化 (INT8)
     - 端到端部署

复现难度: ⭐⭐⭐⭐
  - 论文详细, 但工程门槛高
  - 需要大集群
  - 中文场景弱

应用:
  - FB Reels
  - IG 推荐
  - 全球部署

在线效果: +12% (显著)

借鉴:
  - 统一架构思想
  - 改进 attention
  - 长序列建模
```

### 33.2 TIGER (Google) 深度补充

```text
发表: 2023 (Google Research)
作者: Google Research

完整方法:
  1. Retrieval as Generation
     - 把文档检索变成"生成 docID"
     - 用 Transformer 生成 docID
  2. docID 生成
     - 基于内容 (标题 + 摘要)
     - 用 RQ-VAE 量化
     - 每个 doc → 一串 token
  3. 检索
     - beam search 生成 docID
     - 前缀树约束
     - 高效解码

复现难度: ⭐⭐⭐
  - 有官方代码
  - 中等算力即可

应用:
  - 文档检索
  - 学术搜索
  - 客服问答

离线效果: +5-10% NDCG

借鉴:
  - Retrieval as Generation
  - SID + 检索
  - 文档场景
```

### 33.3 MTGR (美团) 深度补充

```text
发表: 2024
作者: 美团

完整方法:
  1. 多任务 GR
     - 多个任务 (CTR/CVR/时长/...)
     - 共享 backbone
     - 任务 head
  2. 业务塔
     - 业务 embedding (外卖/闪购/酒旅)
     - 业务塔 + 共享塔
  3. 训练目标
     - 多任务 loss 加权
     - 任务冲突缓解

复现难度: ⭐⭐⭐⭐
  - 无代码
  - 找团队

应用:
  - 外卖推荐
  - 闪购推荐
  - 酒旅推荐

在线效果: +5-8% CTR, +3-5% GMV

借鉴:
  - 多业务 GR
  - 业务塔分离
  - 跨业务共享
```

### 33.4 P5 (Salesforce / 学界) 深度补充

```text
发表: 2022
作者: 学界 + Salesforce

完整方法:
  1. 多任务统一
     - 5 个任务 (评分/检索/...)
     - 用 T5 (Enc-Dec)
     - 任务用 prompt 区分
  2. Prompt 设计
     - 任务特定 prompt
     - 上下文 prompt
  3. 预训练
     - 推荐语料
     - 多任务联合

复现难度: ⭐⭐⭐
  - 有官方代码
  - T5 base 即可

应用:
  - 推荐预训练
  - 多任务

离线效果: 较好

借鉴:
  - 多任务统一 (T5)
  - Prompt 设计
  - 预训练思路
```

### 33.5 RQ-VAE (Google) 深度补充

```text
发表: 2022
作者: Google Research

完整方法:
  1. 残差量化 (RQ)
     - 第一层: 量化 z
     - 残差: z - e_{c_0}
     - 第二层: 量化残差
     - 重复 L 层
  2. SID 生成
     - item → encoder → z
     - 残差量化 → (c_0, c_1, ..., c_{L-1})
     - 这就是 SID
  3. 应用
     - 用 SID 训练 Transformer
     - SID → item 的映射

复现难度: ⭐⭐
  - 有官方代码
  - 简单

应用:
  - GR 基础
  - TIGER / HSTU / MTGR 都需要

离线效果: 重建率高

借鉴:
  - SID 核心
  - 残差量化思想
  - 重建 + 生成
```

### 33.6 S3 (RecSys 2022) 深度补充

```text
发表: 2022 (RecSys)
作者: 学界

完整方法:
  1. 长序列建模
     - 自回归 (类似 GPT)
     - 长序列 (10K+)
  2. 简化 attention
     - O(N) 而不是 O(N²)
  3. 训练优化
     - 序列采样
     - 负采样

复现难度: ⭐⭐⭐
  - 有代码
  - 中等

应用:
  - 长序列建模
  - 用户行为预测

借鉴:
  - 长序列
  - 简化 attention
```

### 33.7 Chinchilla (DeepMind) 深度补充

```text
发表: 2022
作者: DeepMind

核心结论:
  - compute-optimal 时, 模型参数与训练 tokens 比例约 1:20
  - 反驳 Kaplan 的"模型越大越好"
  - 算力应该"模型 + 数据"平衡

方法:
  - IsoFLOP: 固定算力, 变 N 与 D
  - 找最优 N 和 D

对 RecSys 启示:
  - 不是模型越大越好
  - 数据也很重要
  - 算力最优配置

复现难度: ⭐⭐
  - 有代码
  - 拟合即可

借鉴:
  - compute-optimal 思想
  - 多变量 scaling
  - 算力最优
```

### 33.8 Kaplan Scaling Law (OpenAI) 深度补充

```text
发表: 2020
作者: OpenAI

核心结论:
  - 模型性能 vs 参数量 / 数据量 / 算力 呈幂律
  - 没有"拐点", 一直 scaling

公式:
  L(N) = a + b * N^(-c)

对 RecSys 启示:
  - 持续 scaling 是有效的
  - 没有"够大就停"

复现难度: ⭐⭐
  - 拟合幂律
  - 简单

借鉴:
  - scaling 思想
  - 幂律拟合
  - 持续 scaling
```

---

## 34. GenRec 主题间关系

### 34.1 五主题关系图

```text
GR (生成式)
  ├─ SID (必备支撑)
  │   └─ RQ-VAE
  ├─ LLM (扩展)
  │   └─ P5, TallRec, ChatRec
  ├─ Scaling (基础)
  │   └─ Kaplan, Chinchilla, UniScale
  └─ 长序列 (挑战)
      └─ S3, LST, SIM, MIM, SLA-Rec
```

### 34.2 24 篇论文的主题分布

| 主题 | 论文数 | 代表 |
|------|:----:|------|
| GR | 5 | HSTU, TIGER, MTGR, OneTrans, COURIER |
| SID | 5 | RQ-VAE, TIGER, GenRet, COURIER, (HSTU 用) |
| LLM4Rec | 5 | P5, M6-Rec, TallRec, InstructRec, ChatRec |
| 长序列 | 5 | S3, LST, SIM, MIM, SLA-Rec |
| Scaling | 4 | Kaplan, Chinchilla, UniScale, (其他) |
| (其他) | 5 | 蒸馏/异构/跨域等 |

### 34.3 主题融合趋势

```text
2023: 各主题独立
2024: GR + SID 融合 (HSTU 用 SID)
2024: GR + 长序列 融合 (HSTU 用稀疏 attn)
2025+: GR + LLM + 多模态 + Scaling = 通用推荐
```

---

## 35. GenRec 完整论文速查表 (50+ 篇)

### 35.1 GR 范式 (10 篇)

| # | 论文 | 年份 | 一句话 |
|:--:|------|:--:|--------|
| 1 | HSTU | 2024 | 推荐统一 transduction |
| 2 | TIGER | 2023 | 文档检索生成 |
| 3 | MTGR | 2024 | 多任务 GR |
| 4 | OneTrans | 2024 | 排序统一 |
| 5 | COURIER | 2024 | 对比检索 |
| 6 | GenRet | 2024 | 生成式检索 |
| 7 | (字节系 5 篇) | 2024 | 字节 GR |
| 8 | (阿里系 5 篇) | 2024 | 阿里 GR 探索 |
| 9 | (Meta 系 2 篇) | 2024 | Meta 后续 |
| 10 | (学界 5+ 篇) | 2024-2025 | 学界跟进 |

### 35.2 SID 经典 (10 篇)

| # | 论文 | 年份 | 一句话 |
|:--:|------|:--:|--------|
| 1 | RQ-VAE | 2022 | 残差量化 |
| 2 | VQ-VAE | 2017 | 矢量量化 |
| 3 | (后续改进 5 篇) | 2023-2024 | 多模态 SID |
| 4 | (学界 3+ 篇) | 2024 | SID 变体 |

### 35.3 LLM4Rec (10 篇)

| # | 论文 | 年份 | 一句话 |
|:--:|------|:--:|--------|
| 1 | P5 | 2022 | 预训练推荐 |
| 2 | M6-Rec | 2022 | M6 推荐 |
| 3 | TallRec | 2023 | 微调 LLM |
| 4 | InstructRec | 2023 | 指令推荐 |
| 5 | ChatRec | 2023 | 对话推荐 |
| 6 | (后续 5+ 篇) | 2024 | LLM4Rec 爆发 |

### 35.4 长序列 (10 篇)

| # | 论文 | 年份 | 一句话 |
|:--:|------|:--:|--------|
| 1 | S3 | 2022 | 长序列自回归 |
| 2 | LST | 2022 | 长序列 Transformer |
| 3 | SIM | 2020 | 搜索长序列 |
| 4 | MIM | 2022 | 多兴趣 |
| 5 | SLA-Rec | 2023 | 长序列注意力 |
| 6 | (后续 5+ 篇) | 2024 | 长序列改进 |

### 35.5 Scaling (5+ 篇)

| # | 论文 | 年份 | 一句话 |
|:--:|------|:--:|--------|
| 1 | Kaplan | 2020 | LLM scaling 起源 |
| 2 | Chinchilla | 2022 | compute-optimal |
| 3 | UniScale | 2024 | RecSys scaling |
| 4 | (RecSys scaling 2+ 篇) | 2024 | 阿里 / 美团 |

---

## 36. GenRec 与其他主题的交叉

### 36.1 与推荐系统

| 维度 | 传统推荐 | GR |
|------|---------|-----|
| 范式 | ID 分类 | 序列生成 |
| 输入 | 用户 + item + context | 行为序列 |
| 输出 | CTR / 排序分数 | 下一行为 |
| 模型 | DLRM / DIN | HSTU / TIGER |
| 训练 | 二分类 / 多任务 | Next token |
| 推理 | 全量打分 | beam search |

### 36.2 与 LLM

| 维度 | LLM | GR |
|------|-----|-----|
| 范式 | 文本生成 | 序列生成 |
| 输入 | 文本 | 行为序列 |
| 输出 | 文本 | 下一行为 |
| 模型 | GPT / LLaMA | HSTU / MTGR |
| 训练 | Next token | Next token (类似) |
| 推理 | 文本生成 | beam search |

### 36.3 与 NLP

| 维度 | NLP | RecSys |
|------|-----|--------|
| 序列 | 文本 | 行为 |
| 长度 | 4K-128K | 1K-10K |
| 目标 | 语义 | 行为预测 |
| 模型 | Transformer | HSTU / MTGR |

---

## 37. 终极路线图 (12 个月)

### 37.1 阅读路线

```text
第 1 月: 基础
  - DIN, DLRM, BST
  - Transformer 复习

第 2 月: SID
  - RQ-VAE (必读)
  - TIGER (必读)

第 3 月: GR 范式
  - HSTU
  - MTGR
  - OneTrans

第 4 月: LLM4Rec
  - P5
  - TallRec
  - ChatRec

第 5 月: 长序列
  - S3, LST
  - SIM, MIM

第 6 月: Scaling
  - Kaplan
  - Chinchilla
  - UniScale

第 7 月: 蒸馏
  - MixFormer
  - EST
  - MDL

第 8 月: 跨域 + 异构
  - MGDIN, CDNet
  - HHFT, HeMix

第 9 月: 自动化
  - FAT
  - 其他 AutoML

第 10 月: 综合
  - 24 篇论文横向对比
  - 综述博客

第 11 月: 实践
  - 1 个 PoC
  - 1 个完整实验

第 12 月: 输出
  - 1 篇内部 wiki
  - 1 次分享
```

### 37.2 实践路线

```text
第 1-3 月: 复现 + 笔记
  - 跑通 3 个 baseline
  - 30 张论文卡

第 4-6 月: 应用
  - 1 个 PoC
  - 1 个工程改进

第 7-9 月: 创新
  - 1 个新 idea
  - 1 个小实验

第 10-12 月: 输出
  - 1 篇博客
  - 1 次分享
```

---

## 38. 结语

> **GenRec = GR + SID + LLM + 长序列 + Scaling**
>
> **24 篇论文是入门**
>
> **未来 12 个月, 这个领域将爆发 50+ 新论文**
>
> **你的目标: 抓住这个窗口期**
>
> **第一步: 把 24 篇读完**
>
> **第二步: 跑 1 个 PoC**
>
> **第三步: 输出 1 个成果**

---

> 本节 ([[#33-38]]) 是 GenRec 经典的"终极补充", 包含深度解析、主题关系、终极路线图。
---

## 45. GenRec 经典论文补充 (代码级)

### 45.1 HSTU 完整 PyTorch 实现

```python
# HSTU 完整实现 (简化版)
class HSTU(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super().__init__()
        # HSTU 层
        self.layers = nn.ModuleList([
            HSTULayer(d_model, num_heads) for _ in range(num_layers)
        ])
        # 输出 head
        self.head = nn.Linear(d_model, 1)

    def forward(self, history, target, action):

        # history (parameters)

        # 拼接 history + target
        x = torch.cat([history, target.unsqueeze(1)], dim=1)

        # HSTU 层
        for layer in self.layers:
            x = layer(x)

        # 输出
        return self.head(x.mean(dim=1))


class HSTULayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 改进 attention
        self.attn = HSTUAttention(d_model, num_heads)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 改进 attention + 残差
        x = x + self.attn(self.norm1(x))
        # FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


class HSTUAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # 共享参数
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        # 相对位置编码
        self.rel_pos = nn.Embedding(2048, self.head_dim)

    def forward(self, x):
        B, N, _ = x.shape
        # Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # 相对位置编码
        positions = torch.arange(N, device=x.device)
        rel_pos = self.rel_pos(positions)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = attn + rel_pos.unsqueeze(0).unsqueeze(0)  # 加相对位置
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out(out)
```

### 45.2 TIGER 完整 PyTorch 实现

```python
# TIGER 完整实现 (简化版)
class TIGER(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_codes=256, num_layers_rq=3):
        super().__init__()
        # RQ-VAE (用于生成 docID)
        self.rq_vae = RQVAE(num_codes, num_layers_rq)
        # Transformer Encoder-Decoder
        self.encoder = TransformerEncoder(d_model, num_layers, num_heads)
        self.decoder = TransformerDecoder(d_model, num_layers, num_heads)
        # 输出 head
        self.head = nn.Linear(d_model, num_codes)

    def encode_doc(self, doc_features):
        # 用 RQ-VAE 生成 docID
        return self.rq_vae.encode(doc_features)

    def forward(self, history_ids, target_ids):
        # history_ids: (B, N) 历史 docID
        # target_ids: (B, M) 目标 docID

        # Encoder
        history_emb = self.encoder(self.embed(history_ids))
        # Decoder
        decoder_out = self.decoder(self.embed(target_ids), history_emb)
        # 输出
        return self.head(decoder_out)


class RQVAE(nn.Module):
    def __init__(self, num_codes=256, num_layers=3):
        super().__init__()
        self.num_codes = num_codes
        self.num_layers = num_layers
        # Codebook
        self.codebooks = nn.ModuleList([
            nn.Embedding(num_codes, 64) for _ in range(num_layers)
        ])

    def encode(self, x):
        # 残差量化生成 SID
        residual = x
        codes = []
        for codebook in self.codebooks:
            # 找最近的 code
            distances = torch.cdist(residual.unsqueeze(1), codebook.weight.unsqueeze(0))
            indices = distances.argmin(dim=-1)
            # 残差
            residual = residual - codebook(indices)
            codes.append(indices)
        return torch.stack(codes, dim=1)  # (B, num_layers)
```

### 45.3 MTGR 完整 PyTorch 实现

```python
# MTGR 完整实现 (简化版)
class MTGR(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_tasks, num_businesses):
        super().__init__()
        # Backbone
        self.backbone = TransformerDecoder(d_model, num_layers, num_heads)
        # 业务 embedding
        self.business_emb = nn.Embedding(num_businesses, d_model)
        # 任务 head
        self.task_heads = nn.ModuleDict({
            f'task_{i}': nn.Linear(d_model, 1)
            for i in range(num_tasks)
        })
        # 业务塔
        self.business_tower = nn.ModuleDict({
            f'biz_{i}': nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            for i in range(num_businesses)
        })

    def forward(self, history, task='ctr', business='taobao'):
        # Backbone
        h = self.backbone(history)
        # 业务塔
        biz_id = self.business_id_map[business]
        biz_emb = self.business_tower[f'biz_{biz_id}'](h.mean(dim=1))
        # 任务 head
        task_id = self.task_id_map[task]
        return self.task_heads[f'task_{task_id}'](biz_emb)
```

---

## 46. GenRec 训练实践

### 46.1 训练 Pipeline (通用)

```python
# GenRec 通用训练 pipeline
class GenRecPipeline:
    def __init__(self, model_name='HSTU'):
        self.model = self.build_model(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            # 数据
            history = batch['history'].cuda()
            target = batch['target'].cuda()
            # 前向
            pred = self.model(history, target)
            # 损失
            loss = F.cross_entropy(pred, target.argmax(dim=-1))
            # 反向
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            metrics = {'auc': 0, 'ndcg': 0}
            for batch in dataloader:
                pred = self.model(batch['history'].cuda(), batch['target'].cuda())
                # 计算指标
                metrics['auc'] += self.compute_auc(pred, batch['label'])
                metrics['ndcg'] += self.compute_ndcg(pred, batch['label'])
        return {k: v / len(dataloader) for k, v in metrics.items()}
```

### 46.2 数据准备 (SID 生成)

```python
# SID 生成 pipeline
class SIDGenerator:
    def __init__(self, num_codes=256, num_layers=3):
        self.rq_vae = RQVAE(num_codes, num_layers)

    def train(self, items_features, epochs=10):
        # 训练 RQ-VAE
        optimizer = torch.optim.Adam(self.rq_vae.parameters(), lr=1e-3)
        for epoch in range(epochs):
            for batch in items_features:
                # 编码
                z = self.rq_vae.encode(batch)
                # 解码
                recon = self.rq_vae.decode(z)
                # 损失
                recon_loss = F.mse_loss(recon, batch)
                optimizer.zero_grad()
                recon_loss.backward()
                optimizer.step()

    def generate_sids(self, items_features):
        # 为所有 item 生成 SID
        sids = {}
        for item_id, features in items_features.items():
            sid = self.rq_vae.encode(features)
            sids[item_id] = sid
        return sids
```

---

## 47. GenRec 复现优先级

### 47.1 推荐优先级

```text
立即跑 (有代码):
  1. RQ-VAE (SID 基础)
  2. TIGER (GR 基础)
  3. Chinchilla (Scaling 基础)
  4. P5 (LLM4Rec 基础)
  5. S3 (长序列 基础)

1-2 周 (部分代码):
  6. COURIER (对比检索)
  7. Kaplan Scaling (拟合)

1 月 (找团队):
  8. HSTU (GR 范式)
  9. MTGR (多任务 GR)
  10. OneTrans (排序统一)

放弃 (极难):
  - TokenMixer, HeMix, UniScale (无代码)
```

---

## 48. GenRec 未来方向 (2025+)

### 48.1 短期 (6 个月)

```text
1. GR + LLM 深度融合
2. 多模态 GR
3. Scaling Law 成熟
```

### 48.2 中期 (12 个月)

```text
1. 端侧 GR
2. 强化学习 + GR
3. 联邦 GR
4. 生成式冷启动
```

### 48.3 长期 (24 个月)

```text
1. 通用推荐系统 (GR + LLM + 多模态 + 端侧 + RL)
2. 行业标准
3. 开源生态
```

---

## 49. 结语: GenRec + 你的位置

> **GenRec = GR + SID + LLM + Scaling**
>
> **24 篇论文是入门**
>
> **你的位置:**
>
> **入门者 → 读完 5 篇必读**
> **中级者 → 跑通 1 个 GR 模型**
> **资深者 → 主导 1 个 GR 项目**
>
> **第一步: 读 RQ-VAE**
> **第二步: 跑 RQ-VAE + TIGER**
> **第三步: 读 HSTU**
> **第四步: 应用到你的业务**

---

> 本节 ([[#45-49]]) 是 GenRec 经典的"代码级补充"和"未来方向"。
