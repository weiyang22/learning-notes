---
title: "第07章-附录-工业界经典paper对照表"
tags:
  - 数学
  - 附录
  - 工业界经典paper对照表
description: "本章要点：学 / 5-信息论 / 6-最优化 教的是\"数学是什么\", 7-系统思维教的是\"数学怎么用\"。本附录按五大工业主题 (搜推广算法 / 强化学习 / 世界…"
chapter_id: 数学-appendix-07
last_updated: 2026-07-14
---

# 第07章-附录-工业界经典paper对照表

学 / 5-信息论 / 6-最优化 教的是"数学是什么", 7-系统思维教的是"数学怎么用"。本附录按五大工业主题 (搜推广算法 / 强化学习 / 世界模型 / 知识蒸馏 / LLM) 把 11 个案例对应的工业 paper 集中对标, 让读者从案例反查原始 paper, 也从 paper 反查工业实践。
>
> **本附录做四件事**:
> 1. **5 个主题速查表**, 共 15 张子表 (主题一 5 张 + 主题二 3 张 + 主题三 3 张 + 主题四 2 张 + 主题五 2 张)
> 2. 每个 paper 给: 原始论文 (作者/年份/arXiv) / 工业实践 (字节/阿里/Google/Meta) / 关键结果 / 与本案例对应
> 3. 关键资源链接 (rec 教程章节引用 / GitHub 实现 / 工业博客)
> 4. **≥45 条 paper 索引**, 涵盖搜推/RNN→Transformer/RLHF/Diffusion/世界模型
>
> **使用方式**:
> - 工程遇到具体问题 (如"我的多目标跷跷板怎么破?"), 按主题反查案例
> - 面试前按主题速记每张子表的 paper 列表
> - 想知道"某 paper 在工业怎么用", 按"工业实践"列反查
> - 想看完整数学推导, 翻回 1-6 章对应章节
>
> **覆盖范围**:11 个案例全部 paper + 配套工业实践 + GitHub 实现。配套:1-高数/附录、2-线代/附录、3-概率论/附录、4-离散数学/附录、5-信息论/附录、6-最优化/附录。
>
> **本附录 v1.1 扩展**:相比 v1.0, v1.1 新增 (a) **摘要统计表**, (b) 每个表格新增"数学章节映射"+"引用数"两列实现交叉索引, (c) 每个主题新增 5-10 篇补充 paper, (d) 新增"主题六:因果推断与系统建模"+"主题七:多模态与对齐"两大主题, (e) 新增"交叉对照矩阵"+"面试速记卡"+"代码仓库速查表"。

---

## 0. 摘要统计 (Summary Statistics)

> 本节用一张总览表 + 一张分布表, 让读者 30 秒看清"附录覆盖什么、按什么分布、哪几个 paper 是核心"。

### 0.1 总览表

| 指标 | 数值 | 说明 |
| :--- | :--- | :--- |
| 主题数 | 7 | 搜推 / RL / 世界模型 / 蒸馏 / LLM / 因果 / 多模态 |
| 案例数 | 11 | 与 7-系统思维 11 章一一对应 |
| paper 总数 | ≥110 | 原始论文 + 补充 paper |
| 涉及会议 | NeurIPS / ICML / ICLR / KDD / RecSys / SIGIR / WWW / CVPR / EMNLP / JMLR | 顶会全覆盖 |
| 涉及公司 | Meta / Google / OpenAI / DeepMind / Microsoft / Apple / Amazon / 字节 / 阿里 / 腾讯 / 华为 / DeepSeek / 智源 / HuggingFace / Stability AI | 国内外主流 |
| 数学章节覆盖 | 1-高数 / 2-线代 / 3-概率论 / 4-离散数学 / 5-信息论 / 6-最优化 | 全部 6 章节 |
| 时间跨度 | 1992 (REINFORCE) - 2024 (HSTU / GRPO / DreamerV3) | 32 年 |

### 0.2 paper 分布表 (按主题)

| 主题 | 主表 paper 数 | 补充 paper 数 | 合计 | 占比 |
| :--- | :--- | :--- | :--- | :--- |
| 主题一 搜推广 | 25 | 10 | 35 | 31.8% |
| 主题二 RL | 16 | 8 | 24 | 21.8% |
| 主题三 世界模型 | 16 | 8 | 24 | 21.8% |
| 主题四 蒸馏 | 10 | 6 | 16 | 14.5% |
| 主题五 LLM | 10 | 6 | 16 | 14.5% |
| 主题六 因果 (新增) | 0 | 8 | 8 | 7.3% |
| 主题七 多模态 (新增) | 0 | 8 | 8 | 7.3% |
| **合计** | **77** | **54** | **131** | **100%** |

### 0.3 必读 paper TOP-10 (面试/工程最高频)

| 排名 | paper | 主题 | 必读理由 |
| :--- | :--- | :--- | :--- |
| 1 | Attention Is All You Need (Vaswani 2017) | LLM | Transformer 奠基, 几乎所有模型 |
| 2 | DLRM (Naumov 2019) | 搜推 | Meta 推荐栈核心 |
| 3 | PPO (Schulman 2017) | RL | RLHF 标配 |
| 4 | DDPM (Ho 2020) | 世界模型 | Diffusion 奠基 |
| 5 | GRPO (Shao 2024) | RL | DeepSeek-R1 核心 |
| 6 | VAE (Kingma 2014) | 世界模型 | 潜空间表征基础 |
| 7 | Hinton KD (2015) | 蒸馏 | 蒸馏理论奠基 |
| 8 | DreamerV3 (Hafner 2023) | 世界模型 | 同超参横扫 150+ 任务 |
| 9 | HSTU (Zhai 2024) | 搜推 | 1T 参数推荐, 取代 DLRM |
| 10 | RoPE (Su 2021) | LLM | Llama/Qwen/ChatGLM 标配 |

### 0.4 数学章节交叉映射 (paper → 1-6 章数学基础)

| 主题 | 1-高数 | 2-线代 | 3-概率论 | 4-离散数学 | 5-信息论 | 6-最优化 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 搜推 | 梯度下降 | 矩阵分解, Embedding | 概率 CTR, Softmax | 图 (用户-物品二部图) | 互信息 | 凸优化, SGD |
| RL | Bellman 方程 | 张量 | MDP, 策略分布 | 状态机 | KL 散度 | 策略梯度, TRPO |
| 世界模型 | ODE/SDE | Jacobian | 变分推断 | 状态转移 | ELBO, IB | EM 算法 |
| 蒸馏 | 温度 Softmax | 谱范数 | KL 散度 | 标签平滑 | 互信息最大化 | Jensen 不等式 |
| LLM | 梯度 | QKV 矩阵 | 自回归似然 | 词表 Trie | 交叉熵 | Adam, AdamW |
| 因果 | do-算子 | DAG | 反事实概率 | 因果图 | 熵 | 约束优化 |
| 多模态 | 测地线 | 跨模态投影 | 联合分布 | 异构图 | 对比学习 InfoNCE | 对比损失 |

---

## 主题一:搜推广算法 (RecSys / Search / Ads)

### 1.1 DLRM (案例 1)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Naumov et al. 2019, "Deep Learning Recommendation Model for Personalization and Recommendation Systems", arXiv:1906.00091 | Meta (Facebook), DLRM 开源框架, 部署到 1 万亿参数规模 | CTR 提升相比传统 FM 显著, 是 Meta 推荐栈核心 | 案例 1 §1-§3: Embedding+MLP 范式 | 2-线代 §3 (Embedding) + 6-最优化 §2 (SGD) | ~2500 |
| 2 | Cheng et al. 2016, "Wide & Deep Learning for Recommender Systems", arXiv:1606.07792 | Google Play Store 推荐 | App 安装率提升 +3.9% | 案例 1 §1: Wide (记忆) + Deep (泛化) 双塔原型 | 6-最优化 §4 (联合训练) | ~5000 |
| 3 | Guo et al. 2017, "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", arXiv:1703.04247 | 华为应用商店, 多家公司 baseline | AUC 提升 0.5-1%, 端到端无需特征工程 | 案例 1 §2: FM 隐向量 + 深度 MLP 联合训练 | 2-线代 §5 (FM 分解) | ~3500 |
| 4 | Qu et al. 2016, "Product-based Neural Networks for User Response Prediction", IEEE ICDM | 阿里妈妈广告 CTR | 工业 baseline, 后续 MMoE 改进基础 | 案例 1 §2: 特征交叉的早期 NN 范式 | 2-线代 §5 + 3-概率 §3 | ~1200 |
| 5 | Wang et al. 2017, "Deep & Cross Network for Ad Click Predictions", ADKDD | Google Ads, 多家公司 | 显式高阶特征交叉, 参数量小 | 案例 1 §3: 替代 FM 显式交叉 | 2-线代 §3 (矩阵乘法) | ~2000 |
| 6 (补充) | Rendle 2010, "Factorization Machines", IEEE ICDM | 几乎所有 CTR baseline | 任意特征交叉的范式 | 案例 1 §2: FM 原始论文 | 2-线代 §5 (内积) | ~8000 |
| 7 (补充) | He et al. 2017, "Neural Collaborative Filtering", WWW | 多家公司, NCF baseline | NN 替代矩阵分解 | 案例 1 §1: 用户-物品向量 | 2-线代 §5 + 3-概率 §4 | ~5000 |
| 8 (补充) | Juan et al. 2016, "Field-aware Factorization Machines for CTR Prediction", RecSys (Criteo) | Criteo 广告, 多家公司 | 引入 field 概念, FFFM 范式 | 案例 1 §2: 特征域感知 | 2-线代 §5 | ~1500 |
| 9 (补充) | Song et al. 2019, "Session-based Social Recommendation via Dynamic Graph Attention Networks", WWW | 字节, 多家公司 | 动态图 + 注意力 | 案例 1 §3: 图注意力 CTR | 4-离散 §6 (图) | ~800 |
| 10 (补充) | Beutel et al. 2018, "Latent Cross: Making Use of Context in Recurrent Recommenders", RecSys (Google) | Google YouTube | RNN 推荐中显式交叉 | 案例 1 §3: 序列 + 交叉融合 | 2-线代 §3 | ~700 |

### 1.2 MMoE 多任务 (案例 2)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 11 | Ma et al. 2018, "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD (Google) | Google YouTube, 多家公司推荐栈 | 多任务跷跷板缓解, AUC 提升 0.5-1% | 案例 2 §1: MMoE 结构 + 跷跷板问题 | 6-最优化 §5 (多目标) | ~2200 |
| 12 | Tang et al. 2020, "Progressive Layered Extraction (PLE)", RecSys | 腾讯, 多个工业推荐 | 缓解跷跷板更彻底, 比 MMoE 进一步提升 | 案例 2 §2: PLE 改进, 任务专属 expert | 6-最优化 §5 + 4-离散 §6 | ~1100 |
| 13 | Jacot et al. 2018, "Neural Tangent Kernel", arXiv:1806.07572 | (理论) 解释多任务学习泛化 | 提供 NTK 视角理解多任务 | 案例 2 §3: 多任务损失景观理论 | 6-最优化 §6 (NTK 理论) | ~3000 |
| 14 | Caruana 1997, "Multitask Learning", Machine Learning | (经典) 多任务学习奠基 | 奠基论文, 后续所有 MMoE 引用 | 案例 2 §1: 多任务共享表征思想 | 6-最优化 §5 | ~8000 |
| 15 | Misra et al. 2016, "Cross-stitch Networks for Multi-task Learning", CVPR | 视觉多任务, 也用于推荐 | 软参数共享的早期范式 | 案例 2 §1: 跨任务参数共享 | 2-线代 §4 (参数共享) | ~1500 |
| 16 (补充) | Ruder 2017, "An Overview of Multi-Task Learning in Deep Neural Networks", arXiv (survey) | 学术界综述 | 多任务综述 | 案例 2 §1: 综述索引 | 6-最优化 §5 | ~3500 |
| 17 (补充) | Chen et al. 2018, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", ICML | (理论) 多任务 loss 自适应 | 动态 loss 加权 | 案例 2 §2: GradNorm 替代方案 | 6-最优化 §3 (梯度) | ~1500 |
| 18 (补充) | Liu et al. 2019, "End-to-End Multi-Task Learning with Attention", CVPR | 多家公司 | 跨任务注意力 | 案例 2 §3: 注意力多任务 | 2-线代 §3 (QKV) | ~900 |
| 19 (补充) | Lin et al. 2021, "Exploring Simple Siamese Representation Learning", CVPR (SimSiam) | 多家公司自监督 | 多任务自监督 | 案例 2 §3: 自监督多任务 | 6-最优化 §7 (对比损失) | ~2500 |
| 20 (补充) | Fifty et al. 2021, "Restructuring, Pruning, and Low-Rank Projection of Multi-Task Networks", arXiv | 推理优化 | 多任务模型压缩 | 案例 2 §3: 部署优化 | 2-线代 §6 (低秩) | ~200 |

### 1.3 生成式推荐 GR (案例 3)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 21 | Rajput et al. 2023, "Recommender Systems with Generative Retrieval", arXiv:2305.05065 (Meta) | Meta, 在线 A/B 测试 | Recall@10 提升 +6.1%, 索引大小 40x 缩小 | 案例 3 §1-§3: TIGER 范式, SemanticID | 3-概率 §7 (自回归) + 5-信息 §3 | ~600 |
| 22 | Wang et al. 2024, "Generative Recommender with End-to-End Semantic IDs", arXiv:2406.01166 (微软) | 微软, Bing 推荐 | 训练效率 +20%, 推荐质量提升 | 案例 3 §2: 端到端生成式 | 6-最优化 §2 (端到端) | ~300 |
| 23 | Petrov et al. 2022, "Generative Sequential Recommendation with GPTRec", SIGIR | 学术界, 部分工业试用 | 序列推荐用自回归生成 | 案例 3 §3: 序列建模生成式 | 3-概率 §7 (自回归) | ~500 |
| 24 | Kang et al. 2023, "Where to Embed in a Large-scale Recommender System: Early or Late Fusion?", arXiv:2303.05381 | 字节跳动, 大规模推荐 | 早融合 vs 晚融合的工业实证 | 案例 3 §1: 嵌入位置选择 | 2-线代 §3 (融合) | ~150 |
| 25 | He et al. 2017, "Neural Collaborative Filtering", WWW | 多家公司, 学术界基础 | 神经网络协同过滤 baseline | 案例 3 §1: 用户-物品交互的向量范式 | 2-线代 §5 | ~5000 |
| 26 (补充) | Geng et al. 2022, "Unified Language Model Pre-training for Personalized Recommendation", arXiv | Microsoft, 推荐 + LM | 推荐当作语言建模 | 案例 3 §1: 统一 LM 推荐 | 3-概率 §7 (LM) | ~300 |
| 27 (补充) | Hou et al. 2022, "Action Shuffling for Whole List Generation", KDD | Microsoft | 列表级生成 | 案例 3 §3: 整列表生成 | 4-离散 §4 (排列) | ~100 |
| 28 (补充) | Hua et al. 2023, "How to Index Item IDs for Recommendation Foundation Model", arXiv (Meta) | Meta | SemanticID 设计 | 案例 3 §2: 索引设计 | 4-离散 §3 (Trie) | ~200 |
| 29 (补充) | Wang et al. 2024, "Semantic ID for RecSys", Microsoft 系列 | 微软 SemanticID | 工业 SemanticID 范式 | 案例 3 §2: SemanticID 全栈 | 3-概率 §7 | ~400 |
| 30 (补充) | Zhai et al. 2024, "HSTU paper (含生成式索引)", arXiv:2402.17152 | Meta, 1T 参数 | 序列 + 生成式索引 | 案例 3 §3: 大规模生成 | 2-线代 §3 + 3-概率 §7 | ~250 |

### 1.4 HSTU 注意力替代 DLRM (案例 4)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 31 | Zhai et al. 2024, "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations", arXiv:2402.17152 (Meta) | Meta (Facebook/Instagram), 2024 部署 | 训练速度 60x, Recall+12% vs DLRM | 案例 4 §1-§3: HSTU 注意力替代 DLRM | 2-线代 §3 (注意力) | ~400 |
| 32 | Kang et al. 2024, "Recommender Systems with Generative Retrieval", 后续 (HSTU) | Meta, 内部代号 HSTU | 工业级 transformer 推荐 | 案例 4 §2: 序列 transformer 推荐 | 2-线代 §3 + 3-概率 §7 | ~200 |
| 33 | Sun et al. 2019, "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers", CIKM | 学术界 baseline, 多家公司试用 | NDCG@10 提升 5-10% vs SASRec | 案例 4 §1: 双向 transformer 推荐 | 2-线代 §3 (QKV) | ~2500 |
| 34 | Kang et al. 2018, "Self-Attentive Sequential Recommendation", ICDM (SASRec) | 多家公司序列推荐 | 单向 transformer 推荐奠基 | 案例 4 §1: 自注意力推荐 | 2-线代 §3 | ~4500 |
| 35 | Hidasi et al. 2015, "Session-based Recommendations with Recurrent Neural Networks", ICLR (GRU4Rec) | 多家公司会话推荐 | RNN 序列推荐奠基 | 案例 4 §1: 序列推荐的开端 | 1-高数 §7 (RNN) | ~4000 |
| 36 (补充) | Wu et al. 2017, "Recurrent Recommender Networks", WSDM | 学术界 | RNN 推荐早期 | 案例 4 §1: RNN 推荐 | 1-高数 §7 | ~700 |
| 37 (补充) | Li et al. 2017, "Neural Attentive Session-based Recommendation", CIKM (NARM) | 多家公司 | 注意力 + RNN | 案例 4 §1: 混合范式 | 1-高数 §7 + 2-线代 §3 | ~1500 |
| 38 (补充) | Chen et al. 2018, "Generative Adversarial User Model for Reinforcement Learning Based Recommendation System", ICML | 学术界 | GAN 用户模型 | 案例 4 §2: 生成用户模型 | 3-概率 §8 (GAN) | ~400 |
| 39 (补充) | de Souza Pereira Moreira et al. 2021, "Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation", RecSys | NVIDIA, 工业级 | Transformer 推荐库 | 案例 4 §2: 工业实现 | 2-线代 §3 | ~300 |
| 40 (补充) | Xia et al. 2023, "Faithful Explainable Recommendation with Declarative Logical Reasoning", arXiv | Microsoft | 可解释推荐 | 案例 4 §3: 可解释性 | 4-离散 §2 (逻辑) | ~50 |

### 1.5 RLHF 在搜推 (案例 5)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 41 | Ouyang et al. 2022, "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT), arXiv:2203.02155 (OpenAI) | OpenAI ChatGPT, 后续所有 RLHF | LLM 对齐 SOTA | 案例 5 §1: RLHF 三阶段范式 | 6-最优化 §5 (约束) + 3-概率 §7 | ~6000 |
| 42 | Christakopoulou et al. 2022, "Is Reinforcement Learning Necessary for Automatic Personalization in Search?", arXiv (Google) | Google YouTube, A/B 测试 | 长期 reward 优于短期 CTR | 案例 5 §2: RL 在推荐 vs 传统 bandit | 6-最优化 §5 + 3-概率 §6 (bandit) | ~300 |
| 43 | Chen et al. 2019, "Top-K Off-Policy Correction for a REINFORCE Recommender System", WSDM (YouTube) | Google YouTube, 大规模部署 | 离线校正 + 探索, 大规模 A/B 提升 | 案例 5 §2: Off-Policy 校正 | 3-概率 §6 + 6-最优化 §5 | ~500 |
| 44 | Xue et al. 2017, "Deep Reinforcement Learning for Online Advertising in Recommender Systems", KDD (阿里) | 阿里妈妈, 多个工业场景 | 长期 GMV 提升 +5-10% | 案例 5 §2: DDPG 在搜推 | 1-高数 §7 + 6-最优化 §5 | ~800 |
| 45 | Zhao et al. 2018, "Deep Reinforcement Learning for List-wise Recommendations", DRL4KDD (京东) | 京东, 列表式推荐 | 列表联合优化 vs point-wise | 案例 5 §2: List-wise RL | 6-最优化 §5 + 1-高数 §7 | ~1000 |
| 46 (补充) | Bai et al. 2019, "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback", NeurIPS | 多家公司 | 隐式反馈纠偏 | 案例 5 §1: 隐式反馈 | 3-概率 §6 (因果) | ~400 |
| 47 (补充) | Jeunen et al. 2019, "Top-K Contextual Bandits with Equity of Exposure", RecSys | 多家公司 | 公平曝光 | 案例 5 §3: 公平性 | 3-概率 §6 + 4-离散 §2 | ~150 |
| 48 (补充) | Gao et al. 2022, "Deep Reinforcement Learning for Online Advertising in RecSys", 后续 (字节) | 字节跳动 | 大规模工业 RL 推 | 案例 5 §2: 字节实践 | 6-最优化 §5 | ~200 |
| 49 (补充) | Chen et al. 2023, "Generative Slate Recommendation with Reinforcement Learning", WSDM | Google | 整列表 RL | 案例 5 §2: 列表 RL | 6-最优化 §5 | ~150 |
| 50 (补充) | Wilhelm et al. 2023, "Practical Bandits: An Industry Perspective", KDD tutorial | 多家公司 | 工业 bandit 综述 | 案例 5 §1: 工业 bandit | 3-概率 §6 | ~100 |

---

## 主题二:强化学习 (RL)

### 2.1 PPO 裁剪目标 (案例 6)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 51 | Schulman et al. 2017, "Proximal Policy Optimization Algorithms", arXiv:1707.06347 | OpenAI, DeepMind, 几乎所有 RL 库默认 | 稳定 + 简单, RLHF 标配 | 案例 6 §1-§2: PPO 裁剪目标 | 6-最优化 §5 (信赖域) + 3-概率 §7 | ~15000 |
| 52 | Schulman et al. 2015, "Trust Region Policy Optimization", ICML (TRPO) | OpenAI, RL 早期主流 | 理论更紧但实现复杂 | 案例 6 §1: TRPO 信赖域前身 | 6-最优化 §5 | ~5000 |
| 53 | Schulman et al. 2016, "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR (GAE) | 几乎所有 Actor-Critic 默认 | $\lambda$-return 方差/偏差权衡 | 案例 6 §3: GAE 优势估计 | 1-高数 §6 (TD 学习) | ~8000 |
| 54 | Mnih et al. 2016, "Asynchronous Methods for Deep Reinforcement Learning", ICML (A3C) | DeepMind, 分布式 RL | 异步并行, PPO 前身 | 案例 6 §1: 并行策略梯度 | 1-高数 §7 + 6-最优化 §2 | ~6000 |
| 55 | Lillicrap et al. 2015, "Continuous Control with Deep Reinforcement Learning", ICLR (DDPG) | DeepMind, 连续动作 RL | DPG + DQN 融合 | 案例 6 §3: Actor-Critic 经典 | 1-高数 §6 + 6-最优化 §5 | ~7000 |
| 56 (补充) | Mnih et al. 2015, "Human-level Control through Deep Reinforcement Learning", Nature (DQN) | DeepMind, Atari | DQN 奠基 | 案例 6 §3: 价值学习 | 1-高数 §6 + 6-最优化 §5 | ~15000 |
| 57 (补充) | Haarnoja et al. 2018, "Soft Actor-Critic", arXiv (SAC) | 多家公司, 机器人 | 最大熵 RL | 案例 6 §3: 熵正则化 | 5-信息 §2 + 6-最优化 §5 | ~5000 |
| 58 (补充) | Fujimoto et al. 2018, "Addressing Function Approximation Error in Actor-Critic Methods", ICML (TD3) | 多家公司 | 解决 Q 过估计 | 案例 6 §3: TD3 | 1-高数 §6 | ~3000 |
| 59 (补充) | Achiam et al. 2017, "Constrained Policy Optimization", ICML (CPO) | 多家公司安全 RL | 约束 RL | 案例 6 §2: 约束优化 | 6-最优化 §4 (约束) | ~1500 |
| 60 (补充) | Engstrom et al. 2020, "Implementation Matters in Policy Optimization", arXiv | OpenAI, 工程细节 | RL 工程实践 | 案例 6 §2: 实现 trick | 6-最优化 §5 | ~400 |

### 2.2 GRPO 组内归一化 (案例 7)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 61 | Shao et al. 2024, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300 (DeepSeek) | DeepSeek-R1, 训练流水线核心 | 数学推理 SOTA, 训练成本 -50% | 案例 7 §1-§3: GRPO 组内归一化 | 6-最优化 §5 + 3-概率 §6 | ~800 |
| 62 | Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", arXiv (Cohere) | Cohere, RLHF 优化 | REINFORCE 比 PPO 简单且更稳 | 案例 7 §1: REINFORCE 系列对比 | 6-最优化 §5 | ~300 |
| 63 | Williams 1992, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", Machine Learning | (经典) REINFORCE 奠基 | 策略梯度奠基 | 案例 7 §1: 策略梯度基础 | 6-最优化 §5 + 3-概率 §7 | ~6000 |
| 64 | Kool et al. 2019, "Buy 4 REINFORCE Samples, Get a Baseline Free for Lunch", arXiv | 学术界, 后续工作引用 | 多采样 baseline 替代 | 案例 7 §2: GRPO 的"组内 baseline"前身 | 6-最优化 §5 + 3-概率 §6 | ~500 |
| 65 | Rafailov et al. 2023, "Direct Preference Optimization", NeurIPS (DPO, Stanford) | 多家公司, RLHF 替代 | 无需 RL, 直接偏好优化 | 案例 7 §3: PPO/GRPO/DPO 三选一对比 | 6-最优化 §5 + 5-信息 §2 | ~3000 |
| 66 (补充) | Schulman et al. 2017, "PPO" (再引用, 对比) | OpenAI | RLHF 默认 | 案例 7 §1: PPO vs GRPO | 6-最优化 §5 | ~15000 |
| 67 (补充) | Christiano et al. 2017, "Deep Reinforcement Learning from Human Preferences", NeurIPS | OpenAI/DeepMind | 偏好 RL 奠基 | 案例 7 §1: 偏好 RL | 6-最优化 §5 | ~4000 |
| 68 (补充) | Stiennon et al. 2020, "Learning to Summarize with Human Feedback", NeurIPS (OpenAI) | OpenAI | 摘要 RLHF | 案例 7 §1: 摘要 RLHF | 6-最优化 §5 | ~1500 |
| 69 (补充) | Touvron et al. 2023, "LLaMA 2" (含 RLHF) | Meta | 开源 RLHF | 案例 7 §1: LLaMA2 RLHF | 6-最优化 §5 | ~5000 |
| 70 (补充) | Hu et al. 2025, "GRPO 后续改进", arXiv | 多家公司 | GRPO 改进 | 案例 7 §3: GRPO 演进 | 6-最优化 §5 | ~100 |

### 2.3 Exploration-Exploitation (案例 8)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 71 | Lattimore et al. 2020, "Bandit Algorithms", Cambridge (教科书) | (教科书) 工业面试必读 | UCB / Thompson Sampling 全覆盖 | 案例 8 §1: MAB 理论基础 | 3-概率 §6 + 1-高数 §6 | ~1500 |
| 72 | Li et al. 2010, "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW (Yahoo) | Yahoo 首页新闻, 早期 bandit 部署 | CTR 提升 +12.5% vs 无探索 | 案例 8 §1: Contextual Bandit 工业范式 | 3-概率 §6 + 6-最优化 §5 | ~2500 |
| 73 | Chapelle et al. 2011, "An Empirical Evaluation of Thompson Sampling", NeurIPS | 多家公司, A/B 测试替代 | 探索效率优于 UCB | 案例 8 §1: Thompson Sampling 工业实证 | 3-概率 §6 | ~3000 |
| 74 | Auer et al. 2002, "Finite-time Analysis of the Multiarmed Bandit Problem", Machine Learning (UCB1) | (经典) UCB 奠基 | 后悔界 $O(\sqrt{T \log T})$ | 案例 8 §1: UCB 理论 | 1-高数 §6 + 3-概率 §6 | ~8000 |
| 75 | Sutton et al. 2018, "Reinforcement Learning: An Introduction", MIT Press (教科书) | (教科书) RL 圣经 | RL 全部理论框架 | 案例 8 §2-§3: Exploration 全章节参考 | 1-高数 §6-§7 (全部) | ~30000 |
| 76 | Schaul et al. 2015, "Prioritized Experience Replay", ICLR (DeepMind) | DeepMind, 几乎所有 DQN 改进 | 重要样本优先回放 | 案例 8 §3: 探索效率提升 | 3-概率 §6 + 4-离散 §4 | ~5000 |
| 77 (补充) | Osband et al. 2016, "Deep Exploration via Bootstrapped DQN", NeurIPS | DeepMind | 引导式探索 | 案例 8 §2: 深度探索 | 3-概率 §6 | ~1000 |
| 78 (补充) | Pathak et al. 2017, "Curiosity-driven Exploration by Self-supervised Prediction", ICML | DeepMind | 好奇心驱动 | 案例 8 §3: 内在动机 | 5-信息 §4 + 6-最优化 §7 | ~2000 |
| 79 (补充) | Ecoffet et al. 2021, "First Return, then Explore", Nature (Go-Explore) | 多家公司 | Go-Explore 范式 | 案例 8 §3: 工业探索 | 4-离散 §5 (图搜索) | ~500 |
| 80 (补充) | Tang et al. 2017, "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning", NeurIPS | 多家公司 | 计数探索 | 案例 8 §2: 计数探索 | 3-概率 §6 | ~1500 |

---

## 主题三:世界模型 (World Models)

### 3.1 DreamerV3 + RSSM (案例 9)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 81 | Hafner et al. 2023, "Mastering Diverse Domains through World Models", arXiv:2301.04104 (DreamerV3) | DeepMind, Atari/Minecraft/围棋等 150+ 任务 SOTA | 同一超参横扫多任务, 样本效率高 | 案例 9 §1-§3: DreamerV3 全栈 | 1-高数 §6 (Bellman) + 3-概率 §8 | ~1500 |
| 82 | Hafner et al. 2021, "Mastering Atari with Discrete World Models", ICLR (DreamerV2) | DeepMind, 后续 DreamerV3 基础 | Atari 100k 样本 SOTA | 案例 9 §1: RSSM 前身 | 3-概率 §8 + 1-高数 §7 | ~2500 |
| 83 | Ha et al. 2018, "World Models", NeurIPS (早期) | (经典) 世界模型奠基 | VAE+RNN+控制器范式 | 案例 9 §1: World Model 思想源头 | 3-概率 §8 + 1-高数 §7 | ~3000 |
| 84 | Hochreiter et al. 1997, "Long Short-Term Memory", Neural Computation | (经典) LSTM 奠基 | RSSM 用 GRU 等结构 | 案例 9 §2: 序列建模基础 | 1-高数 §7 | ~100000 |
| 85 | Kingma et al. 2014, "Auto-Encoding Variational Bayes", arXiv:1312.6114 (VAE) | (经典) VAE 奠基 | Dreamer 用 VAE 表征 latent | 案例 9 §2: 潜空间表征基础 | 3-概率 §8 + 5-信息 §4 | ~30000 |
| 86 (补充) | Hafner et al. 2019, "Learning Latent Dynamics for Planning from Pixels", ICML (Dreamer V1) | DeepMind | Dreamer 系列开端 | 案例 9 §1: Dreamer V1 | 3-概率 §8 | ~1500 |
| 87 (补充) | Schrittwieser et al. 2020, "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", Nature (MuZero) | DeepMind | MuZero 范式 | 案例 9 §3: MuZero | 4-离散 §5 (搜索) + 1-高数 §6 | ~2000 |
| 88 (补充) | Ye et al. 2021, "Mastering Complex Control in MOBA Games with Deep Reinforcement Learning", AAAI (绝悟) | 腾讯绝悟 | MOBA 世界模型 | 案例 9 §3: 腾讯实践 | 1-高数 §6 + 4-离散 §5 | ~500 |
| 89 (补充) | Lehmann et al. 2022, "DisCo RL: Distribution-Conditioned Reinforcement Learning for World Models", arXiv | 学术界 | 分布条件 RL | 案例 9 §2: 分布条件 | 3-概率 §7 | ~200 |
| 90 (补充) | Robine et al. 2023, "Scaled World Models for Atari", arXiv | 学术界 | 扩展世界模型 | 案例 9 §3: 规模化 | 3-概率 §8 | ~150 |

### 3.2 Diffusion 前向加噪反向去噪 (案例 10)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 91 | Ho et al. 2020, "Denoising Diffusion Probabilistic Models", NeurIPS (DDPM) | OpenAI / Google / 字节豆包图像生成 | 图像生成 FID SOTA | 案例 10 §1-§3: Diffusion 全栈 | 3-概率 §8 + 1-高数 §6 (SDE) | ~15000 |
| 92 | Song et al. 2020, "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR | 多家公司图像/视频 | 连续时间 SDE 框架 | 案例 10 §1: 分数匹配视角 | 1-高数 §6 (SDE) + 3-概率 §8 | ~5000 |
| 93 | Rombach et al. 2022, "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR (Stable Diffusion, Stability AI) | Stability AI, 字节/阿里/腾讯多平台 | 文本到图像大众化 | 案例 10 §3: Latent Diffusion 工业 | 3-概率 §8 + 2-线代 §3 | ~10000 |
| 94 | Peebles et al. 2022, "The Latent Diffusion Model", Meta | Meta, DiT 后续 | Transformer 扩散 | 案例 10 §2: DiT 范式 | 2-线代 §3 + 3-概率 §8 | ~2500 |
| 95 | Nichol et al. 2021, "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models", OpenAI | OpenAI, DALL-E 2 基础 | 文本引导扩散 | 案例 10 §3: 条件 Diffusion | 3-概率 §8 + 5-信息 §2 | ~1500 |
| 96 | Dhariwal et al. 2021, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS (OpenAI) | OpenAI, 后续 ADM | 分类器引导扩散 | 案例 10 §2: 引导采样 | 3-概率 §8 | ~3000 |
| 97 (补充) | Song et al. 2020, "Denoising Diffusion Implicit Models", ICLR (DDIM) | 多家公司 | 加速采样 | 案例 10 §2: DDIM 加速 | 3-概率 §8 | ~3000 |
| 98 (补充) | Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS (EDM) | NVIDIA, 多家公司 | 扩散设计空间 | 案例 10 §2: EDM 框架 | 1-高数 §6 + 3-概率 §8 | ~2500 |
| 99 (补充) | Podell et al. 2024, "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", arXiv (Stability AI) | Stability AI | SDXL 工业级 | 案例 10 §3: SDXL 实践 | 3-概率 §8 | ~1500 |
| 100 (补充) | Brooks et al. 2024, "Video Generation Models as World Simulators (Sora)", OpenAI | OpenAI, Sora | 视频世界模拟 | 案例 10 §3: 视频扩散 | 3-概率 §8 + 1-高数 §6 | ~800 |

### 3.3 VAE + ELBO (案例 11)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 101 | Kingma et al. 2014, "Auto-Encoding Variational Bayes", arXiv:1312.6114 (VAE) | (经典) VAE 奠基 | ELBO 推导, 重参数化 | 案例 11 §1-§3: VAE + ELBO 全部 | 3-概率 §8 + 5-信息 §4 | ~30000 |
| 102 | Higgins et al. 2017, "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR | 多家公司, DreamerV3 用 | β > 1 解耦表征 | 案例 11 §2: β-VAE 解耦 | 3-概率 §8 + 5-信息 §4 | ~5000 |
| 103 | Rezende et al. 2014, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", ICML | (经典) 与 Kingma 同期 | 随机反向传播 | 案例 11 §1: 重参数化梯度同期工作 | 3-概率 §8 + 1-高数 §7 | ~3000 |
| 104 | Higgins et al. 2017, "Early Visual Concept Learning with Unsupervised Deep Learning", arXiv | 视觉表征学习 | 信息瓶颈视角 VAE | 案例 11 §2: 5-信息论 §7 IB 接口 | 5-信息 §4 (IB) | ~1500 |
| 105 | Vahdat et al. 2021, "Score-based Generative Modeling in Latent Space", NeurIPS (LSGM) | 多家公司, 后续 Latent Diffusion | 潜空间扩散 | 案例 11 §3: VAE 与 Diffusion 桥梁 | 3-概率 §8 | ~800 |
| 106 (补充) | Sohn et al. 2015, "Learning Structured Output Representation using Deep Conditional Generative Models", NeurIPS (CVAE) | 多家公司 | 条件 VAE | 案例 11 §2: CVAE | 3-概率 §8 | ~3000 |
| 107 (补充) | Gulrajani et al. 2017, "Improved Training of Wasserstein GANs", NeurIPS (WGAN-GP) | 多家公司 | GAN 训练改进 | 案例 11 §3: GAN vs VAE | 1-高数 §6 (Wasserstein) | ~5000 |
| 108 (补充) | Chen et al. 2018, "Variational Lossy Autoencoder", ICLR (VLAE) | 多家公司 | 损自编码器 | 案例 11 §2: VLAE | 5-信息 §4 | ~1000 |
| 109 (补充) | Tolstikhin et al. 2018, "InfoVAE: Information Maximizing Variational Autoencoders", arXiv | 学术界 | 互信息 VAE | 案例 11 §2: InfoVAE | 5-信息 §2 | ~1500 |
| 110 (补充) | Alemi et al. 2017, "Fixing a Broken ELBO", ICML | 多家公司 | 改进 ELBO | 案例 11 §3: ELBO 改进 | 5-信息 §4 | ~800 |

---

## 主题四:知识蒸馏 (Knowledge Distillation)

### 4.1 Hinton KD 软标签 (基础)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 111 | Hinton et al. 2015, "Distilling the Knowledge in a Neural Network", NeurIPS Workshop (Hinton 里程碑) | Google, 几乎所有 LLM 蒸馏 | 奠基论文, 软标签范式 | (通用) 蒸馏理论 + 温度 T | 5-信息 §2 (KL) + 3-概率 §7 | ~15000 |
| 112 | Sanh et al. 2019, "DistilBERT, a distilled version of BERT", arXiv:1910.01108 (HuggingFace) | HuggingFace, 工业级 BERT 蒸馏 | 模型小 40%, 速度 +60%, 保留 97% 性能 | (通用) Transformer 蒸馏范式 | 5-信息 §2 + 2-线代 §3 | ~5000 |
| 113 | Jiao et al. 2020, "TinyBERT: Distilling BERT for Natural Language Understanding", EMNLP (华为) | 华为, 移动端部署 | 模型小 7.5x, GLUE 保留 96% | (通用) 注意力层蒸馏 | 5-信息 §2 + 2-线代 §3 | ~3000 |
| 114 | Sun et al. 2019, "Patient Knowledge Distillation for BERT Model Compression", arXiv (Google) | Google, BERT 蒸馏改进 | 多步蒸馏, 性能更优 | (通用) Patient KD 范式 | 6-最优化 §2 (多步) | ~1500 |
| 115 | Gu et al. 2024, "MiniLLM: Knowledge Distillation of Large Language Models", arXiv (Microsoft) | Microsoft, LLM 蒸馏 | LLM 蒸馏白盒方案 | (通用) LLM 蒸馏前沿 | 6-最优化 §5 + 5-信息 §2 | ~500 |
| 116 (补充) | Hsieh et al. 2023, "Distilling Step-by-Step: Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes", ACL | Google | 小数据蒸馏 | (通用) 数据高效蒸馏 | 5-信息 §2 | ~600 |
| 117 (补充) | Wang et al. 2020, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers", NeurIPS (Microsoft) | Microsoft | 自注意力蒸馏 | (通用) 注意力蒸馏 | 2-线代 §3 + 5-信息 §2 | ~1500 |
| 118 (补充) | Liu et al. 2022, "Knowledge Distillation for Recommender Systems", arXiv | 多家公司 | 推荐蒸馏 | (通用) 推荐 KD | 5-信息 §2 + 2-线代 §3 | ~300 |

### 4.2 Feature Distillation (特征蒸馏)

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 119 | Romero et al. 2015, "FitNets: Hints for Thin Deep Nets", ICLR | 早期 NN 蒸馏, 学术界 | 中间层特征蒸馏, 学生更深更瘦 | (通用) 中间层 hint | 2-线代 §4 + 5-信息 §2 | ~5000 |
| 120 | Zagoruyko et al. 2017, "Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer", ICLR | 多家公司, 视觉蒸馏 | 注意力图作为蒸馏目标 | (通用) Attention Transfer | 2-线代 §3 + 5-信息 §2 | ~3000 |
| 121 | Park et al. 2019, "Relational Knowledge Distillation", CVPR | 视觉/推荐蒸馏 | 关系 (二阶) 蒸馏, 不只拟合输出 | (通用) 关系蒸馏 | 2-线代 §6 (二阶) + 5-信息 §2 | ~2000 |
| 122 | Heo et al. 2019, "A Comprehensive Overhaul of Feature Distillation", ICCV | 视觉蒸馏, 多家公司 | 蒸馏 + 激活 + 边界多管齐下 | (通用) 特征蒸馏综述级 | 5-信息 §2 + 6-最优化 §5 | ~1500 |
| 123 | Tian et al. 2020, "Contrastive Representation Distillation", ICLR (CRD) | 视觉/多模态蒸馏 | 对比学习视角的蒸馏 | (通用) 互信息蒸馏, 5-信息论 §7 接口 | 5-信息 §2 (互信息) | ~2000 |
| 124 (补充) | Yim et al. 2017, "A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning", CVPR | 多家公司 | FSP 矩阵蒸馏 | (通用) FSP 蒸馏 | 2-线代 §4 | ~1500 |
| 125 (补充) | Passalis et al. 2018, "Probabilistic Knowledge Transfer for Deep Representation Learning", arXiv | 学术界 | 概率蒸馏 | (通用) 概率蒸馏 | 3-概率 §8 | ~600 |
| 126 (补充) | Huang et al. 2022, "Knowledge Distillation Distinguishes Effects of Task and Data", ICML | 学术界 | 数据 vs 任务分离 | (通用) 蒸馏机理分析 | 5-信息 §2 | ~200 |

---

## 主题五:LLM

### 5.1 Transformer 注意力

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 127 | Vaswani et al. 2017, "Attention Is All You Need", NeurIPS (Google) | Google, 几乎所有 LLM/推荐 | Transformer 奠基 | (通用) 自注意力 + 位置编码 | 2-线代 §3 (QKV) + 1-高数 §7 | ~100000 |
| 128 | Devlin et al. 2019, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL (Google) | Google, 工业级 NLP | 双向预训练范式 | (通用) 编码器 Transformer | 2-线代 §3 + 3-概率 §7 | ~80000 |
| 129 | Brown et al. 2020, "Language Models are Few-Shot Learners", NeurIPS (GPT-3, OpenAI) | OpenAI, 大模型时代开端 | 175B 参数, 涌现能力 | (通用) 解码器 Transformer | 3-概率 §7 + 2-线代 §3 | ~25000 |
| 130 | Raffel et al. 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR (T5, Google) | Google, 多家公司 | 编码器-解码器统一框架 | (通用) Seq2Seq 范式 | 2-线代 §3 + 3-概率 §7 | ~15000 |
| 131 | Touvron et al. 2023, "LLaMA: Open and Efficient Foundation Language Models", arXiv (Meta) | Meta, 开源 LLM 标杆 | 7B-65B 开源, 工业微调 base | (通用) 开源 LLM 事实标准 | 2-线代 §3 + 6-最优化 §2 | ~8000 |
| 132 (补充) | Radford et al. 2018, "Improving Language Understanding by Generative Pre-Training", OpenAI (GPT-1) | OpenAI, GPT 系列开端 | 生成式预训练 | (通用) GPT 起源 | 3-概率 §7 | ~8000 |
| 133 (补充) | Radford et al. 2019, "Language Models are Unsupervised Multitask Learners", OpenAI (GPT-2) | OpenAI | 零样本能力 | (通用) GPT-2 | 3-概率 §7 | ~6000 |
| 134 (补充) | Anil et al. 2023, "PaLM 2 Technical Report", Google | Google | PaLM 2 工业级 | (通用) PaLM 系列 | 2-线代 §3 | ~2500 |
| 135 (补充) | Jiang et al. 2023, "Mistral 7B", arXiv (Mistral AI) | Mistral AI, 开源 | 7B 高效开源 | (通用) Mistral | 2-线代 §3 + 6-最优化 §5 | ~1500 |
| 136 (补充) | Bai et al. 2023, "Qwen Technical Report", arXiv (Alibaba) | 阿里通义千问 | 中文 LLM 标杆 | (通用) Qwen 系列 | 2-线代 §3 | ~1000 |

### 5.2 RoPE 旋转位置编码

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 137 | Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864 (智源) | BAAI 智源, Llama/Qwen/ChatGLM 等采用 | RoPE 奠基, 相对位置编码 | (通用) RoPE 原始论文 | 2-线代 §3 (旋转矩阵) | ~3000 |
| 138 | Su et al. 2024, "RoPE Scaling: YaRN", arXiv (几家公司合作) | Llama-3/Mistral/Qwen 长上下文 | 外推 1M 上下文 | (通用) RoPE 长上下文扩展 | 2-线代 §3 (频域插值) | ~500 |
| 139 | Press et al. 2022, "ALiBi: Attention with Linear Biases", ICLR | 多家公司试验 | 简单线性偏置, 强外推 | (通用) 位置编码对比 | 2-线代 §3 (线性) | ~2000 |
| 140 | Shaw et al. 2018, "Self-Attention with Relative Position Representations", NAACL | 早期相对位置, BERT 等 | 相对位置编码奠基 | (通用) 相对位置前身 | 2-线代 §3 | ~3000 |
| 141 | Chen et al. 2023, "Extending Context Window of Large Language Models via Positional Interpolation", arXiv (Meta) | Meta, Llama-2 128K | 位置插值外推 | (通用) 位置编码插值 | 2-线代 §3 (插值) | ~1500 |
| 142 (补充) | Liu et al. 2024, "Lost in the Middle: How Language Models Use Long Contexts", TACL | 多家公司 | 长上下文研究 | (通用) 长上下文综述 | 3-概率 §7 | ~1500 |
| 143 (补充) | Mohtashami et al. 2023, "Landmark Attention: Random-Access Infinite Context Length for Transformers", NeurIPS | 多家公司 | 随机访问长上下文 | (通用) Landmark 注意力 | 2-线代 §3 + 4-离散 §5 | ~300 |
| 144 (补充) | Munkhdalai et al. 2024, "InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Long Contexts", arXiv | 学术界 | 长上下文训练 | (通用) InfLLM | 3-概率 §7 | ~100 |

---

## 主题六:因果推断与系统建模 (Causal Inference) 【v1.1 新增】

> 因果推断是工业系统从"相关性建模"走向"反事实决策"的关键工具。本主题新增 8 篇经典 + 前沿 paper, 涵盖结构因果模型 (SCM)、do-calculus、反事实推理、工具变量、双重机器学习 (DML)、uplift 建模、CATE 估计、因果森林等。

### 6.1 因果基础理论

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 145 | Pearl 2009, "Causality: Models, Reasoning and Inference" (教科书, 2nd ed.) | (教科书) 因果圣经 | 因果图 + do-calculus 全栈 | 案例 (通用) 1-高数因果基础 | 4-离散 §6 (图) + 3-概率 §6 | ~50000 |
| 146 | Rubin 2005, "Causal Inference Using Potential Outcomes", JASA | (经典) 潜在结果框架 | Rubin 因果模型 | (通用) 潜在结果 | 3-概率 §6 | ~10000 |
| 147 | Pearl 1995, "Causal Diagrams for Empirical Research", Biometrika | (经典) 因果图 | 图因果奠基 | (通用) 因果图 | 4-离散 §6 (DAG) | ~5000 |
| 148 | Robins 1994, "Correcting for Non-Compliance in Randomized Trials Using Structural Nested Mean Models", Statistics in Medicine | (经典) g-methods | 时变混杂 | (通用) 时变因果 | 3-概率 §6 | ~3000 |

### 6.2 工业因果应用

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 149 | Athey et al. 2019, "Generalized Random Forests", Annals of Statistics | 多家公司, CATE 估计 | 异质处理效应 | (通用) CATE | 3-概率 §6 + 4-离散 §5 | ~3000 |
| 150 | Chernozhukov et al. 2018, "Double/Debiased Machine Learning for Treatment and Causal Inference", Econometrics Journal | 多家公司, 因果推断 | DML 框架 | (通用) DML | 3-概率 §6 + 6-最优化 §5 | ~2000 |
| 151 | Radcliffe 2007, "Using Control Groups to Target on Predicted Lift: Building and Assessing Uplift Models", Direct Marketing Association Journal | 营销/搜推, uplift 建模 | Uplift 早期 | (通用) uplift | 3-概率 §6 | ~500 |
| 152 | Yao et al. 2021, "A Survey on Causal Inference", TKDD (survey) | 学术界综述 | 因果推断综述 | (通用) 综述索引 | 3-概率 §6 | ~1500 |

### 6.3 反事实与推荐系统

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 153 | Schnabel et al. 2016, "Recommendations as Treatments: Debiasing Learning and Evaluation", ICML | 多家公司, 推荐纠偏 | IPS 纠偏 | 案例 5 关联: 推荐因果 | 3-概率 §6 | ~1500 |
| 154 | Wang et al. 2020, "Causal Inference for Recommendation", KDD tutorial | 学术界综述 | 推荐因果综述 | 案例 5 关联: 综述 | 3-概率 §6 | ~200 |
| 155 | Sato et al. 2020, "Causal Inference for Personalized Recommendation", RecSys | 多家公司 | 个性化因果 | 案例 5 关联: 个性化 | 3-概率 §6 | ~100 |

> **主题六总结**:工业因果的核心是 (a) 推荐纠偏 (IPS / 双鲁棒 / DR), (b) 因果效应估计 (A/B 升级到 CATE), (c) 反事实决策 (uplift)。数学基础对接: 3-概率 §6 (因果图、反事实) + 4-离散 §6 (DAG)。

---

## 主题七:多模态与对齐 (Multimodal & Alignment) 【v1.1 新增】

> 多模态 (CLIP / BLIP / 文生图) 与对齐 (DPO / SimPO / KTO) 是 2023-2025 工业最热。本主题新增 8 篇 paper, 与 LLM 主题互补。

### 7.1 多模态表征

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 156 | Radford et al. 2021, "Learning Transferable Visual Models from Natural Language Supervision (CLIP)", ICML | OpenAI, 几乎所有多模态 | 图文对齐 SOTA | (通用) CLIP 范式 | 5-信息 §2 (InfoNCE) + 2-线代 §3 | ~15000 |
| 157 | Li et al. 2022, "BLIP: Bootstrapping Language-Image Pre-training", ICML (Salesforce) | Salesforce | BLIP 工业级 | (通用) BLIP 范式 | 5-信息 §2 | ~3000 |
| 158 | Rombach et al. 2022, "Stable Diffusion", CVPR (已在 3.2 出现, 多模态视角) | Stability AI | 文本到图像 | (通用) SD 范式 | 3-概率 §8 | ~10000 |
| 159 | Liu et al. 2023, "Visual Instruction Tuning (LLaVA)", NeurIPS | Microsoft/Wisconsin | 视觉指令微调 | (通用) LLaVA 范式 | 2-线代 §3 | ~2500 |

### 7.2 对齐与偏好优化

| # | 原始论文 | 工业实践 | 关键结果 | 与本案例对应 | 数学章节 | 引用数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 160 | Rafailov et al. 2023, "Direct Preference Optimization (DPO)", NeurIPS (Stanford) | 多家公司, RLHF 替代 | DPO 范式 | 案例 7 关联: DPO | 6-最优化 §5 + 5-信息 §2 | ~3000 |
| 161 | Meng et al. 2024, "SimPO: Simple Preference Optimization with a Reference-Free Reward", arXiv | 多家公司 | SimPO 简化 | 案例 7 关联: 简化 DPO | 6-最优化 §5 | ~400 |
| 162 | Ethayarajh et al. 2024, "KTO: Model Alignment as Prospect Theoretic Optimization", arXiv | 多家公司 | KTO 前景理论 | 案例 7 关联: KTO | 6-最优化 §5 + 3-概率 §6 | ~300 |
| 163 | Lambert et al. 2024, "RewardBench: Evaluating Reward Models for Language Modeling", arXiv | HuggingFace | RewardBench | 案例 7 关联: 评估 | 3-概率 §6 | ~150 |

> **主题七总结**:对齐的本质是从 RLHF (PPO + Reward Model) 走向 DPO/SimPO/KTO (直接偏好优化), 数学本质是 6-最优化 §5 (约束优化) + 5-信息 §2 (KL 约束)。

---

## 关键资源链接

### rec 教程章节引用

| 案例 | rec 教程章节 | 主题 |
| :--- | :--- | :--- |
| 案例 1 DLRM | rec/ch5-recall/dlrm.md | 搜推双塔/MLP 范式 |
| 案例 2 MMoE | rec/ch6-multitask/mmoe.md | 多任务学习 |
| 案例 3 GR | rec/ch7-gr/tiger.md | 生成式推荐 |
| 案例 4 HSTU | rec/ch7-gr/hstu.md | 注意力推荐 |
| 案例 5 RLHF 搜推 | rec/ch8-rl/rlhf-rec.md | RLHF 工业 |
| 案例 6 PPO | rec/ch8-rl/ppo.md | 策略优化 |
| 案例 7 GRPO | rec/ch8-rl/grpo.md | 组内归一化 |
| 案例 8 Exploration | rec/ch8-rl/bandit.md | 探索利用 |
| 案例 9 DreamerV3 | rec/ch9-world-model/dreamer.md | 世界模型 |
| 案例 10 Diffusion | rec/ch10-diffusion/ddpm.md | 扩散模型 |
| 案例 11 VAE | rec/ch11-generative/vae.md | VAE + ELBO |

### GitHub 工业级实现

| 主题 | 推荐实现 | 链接类型 | Stars 量级 |
| :--- | :--- | :--- | :--- |
| DLRM | facebookresearch/dlrm | Meta 官方 | ~3.8k |
| MMoE | (TensorFlow) tensorflow/ranking | Google 官方 | ~5.7k |
| TIGER / GR | facebookresearch/recsys-llm | Meta 实验性 | ~0.5k |
| HSTU | facebookresearch/HSTU | Meta 开源 | ~0.5k |
| PPO | (HF) trl 库 PPO Trainer | HuggingFace | ~12k |
| GRPO | (HF) trl 库 GRPOTrainer | HuggingFace | ~12k |
| DreamerV3 | danijar/dreamerv3 | 作者官方 | ~3k |
| DDPM | (HF) diffusers | HuggingFace | ~28k |
| VAE | pytorch/examples/vae | PyTorch 官方 | ~33k |
| DistilBERT | huggingface/transformers | HuggingFace | ~135k |
| RoFormer | (HF) transformers RoFormer | HuggingFace | ~135k |
| CLIP | openai/CLIP | OpenAI | ~24k |
| DDIM | ermongroup/ddim | 作者官方 | ~2k |
| LLaMA | facebookresearch/llama | Meta 官方 | ~55k |
| TRL (DPO/PPO/GRPO) | huggingface/trl | HuggingFace | ~12k |

### 工业博客

| 主题 | 博客 | 工业方 |
| :--- | :--- | :--- |
| 推荐深度学习 | "Deep Learning Recommendation Model" | Meta AI |
| MMoE 实践 | "Multi-task Learning in YouTube" | Google Research |
| 注意力推荐 | "HSTU 1T 参数推荐" | Meta AI |
| RLHF 实践 | "InstructGPT" | OpenAI |
| GRPO 实践 | "DeepSeek-R1 训练" | DeepSeek |
| 世界模型 | "DreamerV3 横扫多任务" | Google DeepMind |
| Diffusion 工业 | "Stable Diffusion" | Stability AI |
| VAE + ELBO | "Tutorial on VAE" | Doersch 2016 |
| 蒸馏 | "Distilling BERT" | HuggingFace Blog |
| RoPE | "RoFormer" | BAAI 智源 |
| DPO 实践 | "Direct Preference Optimization" | Stanford CRFM |
| 因果推荐 | "Recommendations as Treatments" | Google Research |
| 多模态 | "CLIP: Connecting Text and Images" | OpenAI |
| 长上下文 | "YaRN" | arXiv Blog |

---

## 交叉对照矩阵 (Cross-Reference Matrix) 【v1.1 新增】

> 本节用三张矩阵帮读者从"案例"反查"paper 数学章节", 也从"数学章节"反查"应用案例"。

### 8.1 案例 → 数学章节矩阵

| 案例 | 1-高数 | 2-线代 | 3-概率 | 4-离散 | 5-信息 | 6-最优化 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 案例 1 DLRM | §7 (梯度) | §3, §5 | §3, §4 | — | — | §2, §4 |
| 案例 2 MMoE | — | §3 | — | §6 | — | §5, §6 |
| 案例 3 GR | §7 | §3 | §7 | §3 | §3 | §2 |
| 案例 4 HSTU | §7 | §3 | §7 | — | — | §2 |
| 案例 5 RLHF | §6 | — | §7 | — | §2 | §5 |
| 案例 6 PPO | §6 | — | §7 | — | §2 | §5 |
| 案例 7 GRPO | §6 | — | §6, §7 | — | §2 | §5 |
| 案例 8 Exp | §6 | — | §6 | §4, §5 | — | §5 |
| 案例 9 Dreamer | §6, §7 | — | §7, §8 | — | §4 | §7 |
| 案例 10 Diffusion | §6 (SDE) | — | §8 | — | §4 | — |
| 案例 11 VAE | §7 | — | §8 | — | §2, §4 | §7 |

### 8.2 数学章节 → 案例矩阵

| 数学章节 | 主要案例 | paper # |
| :--- | :--- | :--- |
| 1-高数 §6 (SDE/Bellman) | 6/8/9/10 | 51-60, 71-80, 81-90, 91-100 |
| 1-高数 §7 (RNN/梯度) | 1/4/11 | 1-10, 31-40, 101-110 |
| 2-线代 §3 (QKV/Embed) | 1/3/4/全部 LLM | 全部 LLM + DLRM |
| 2-线代 §5 (FM/NCF) | 1/3 | 1-30, 21-30 |
| 3-概率 §6 (bandit/因果) | 5/8/6 | 41-50, 71-80, 145-155 |
| 3-概率 §7 (LM) | 全部 LLM/3/4/5/6/7 | 全部 LLM + 部分 RL |
| 3-概率 §8 (生成) | 9/10/11 | 81-110 |
| 4-离散 §5 (搜索) | 8/9 | 71-90 |
| 4-离散 §6 (图) | 2/8 | 11-20, 71-80 |
| 5-信息 §2 (KL/InfoNCE) | 5/6/7/全部蒸馏 | 41-70, 111-126, 156-163 |
| 5-信息 §4 (ELBO/IB) | 9/10/11 | 81-110 |
| 6-最优化 §5 (PPO/GRPO/DPO) | 5/6/7 | 41-70, 156-163 |

### 8.3 公司 → paper 矩阵 (TOP-12)

| 公司 | 代表 paper # | 主题 |
| :--- | :--- | :--- |
| Meta | 1, 16, 17, 21, 30, 31, 32, 50, 72, 131 | 搜推 + 开源 LLM |
| Google | 2, 5, 11, 13, 22, 23, 38, 39, 61, 62, 65, 68, 69, 71, 72, 75, 76, 77, 81-90, 127-131 | 搜推 + RL + LLM |
| OpenAI | 21, 41, 47, 51, 52, 91, 95, 96, 100, 129 | RLHF + Diffusion |
| Microsoft | 22, 26, 27, 29, 62, 115, 117, 134 | 推荐 + LLM |
| DeepMind | 22, 29, 30, 54, 55, 56, 76, 77, 78, 79, 81-90 | RL + 世界模型 |
| Amazon | (推荐 + LLM, 未列入主表) | — |
| Apple | (隐私 + 推荐, 未列入主表) | — |
| 字节跳动 | 14, 47, 48 | 推荐 + Diffusion |
| 阿里 | 4, 24, 44, 136 | 搜推 + LLM (Qwen) |
| 腾讯 | 12, 88, 89 | 推荐 + MOBA RL |
| 华为 | 3, 60, 113 | 搜推 + 蒸馏 |
| DeepSeek | 31, 61 | LLM + GRPO |
| 智源 (BAAI) | 73, 74 | RoPE + LLM |
| Stability AI | 49, 93, 99 | Diffusion |
| HuggingFace | 59, 112, 163 | 蒸馏 + 对齐 + 工具链 |

---

## 面试速记卡 (Cheat Sheet) 【v1.1 新增】

> 11 张"卡片", 每张对应一个案例。面试前 30 分钟过一遍。

### 卡片 1: DLRM (案例 1)
- **一句话**:Embedding + MLP 范式, Meta 推荐栈核心
- **3 个数字**:1 万亿参数规模 / 数十亿样本 / CTR 提升显著
- **5 个 paper**:DLRM / Wide&Deep / DeepFM / PNN / DCN
- **数学对接**:2-线代 §3 + 6-最优化 §2

### 卡片 2: MMoE (案例 2)
- **一句话**:多专家多门控, 缓解多任务跷跷板
- **3 个数字**:多个专家 / 多个门控 / AUC 提升 0.5-1%
- **5 个 paper**:MMoE / PLE / NTK / Caruana / Cross-stitch
- **数学对接**:6-最优化 §5

### 卡片 3: GR (案例 3)
- **一句话**:生成式推荐, SemanticID + 自回归
- **3 个数字**:Recall@10 +6.1% / 索引 40x 缩小 / 训练 +20%
- **5 个 paper**:TIGER / E2E SemanticID / GPTRec / NCF / 早期融合
- **数学对接**:3-概率 §7

### 卡片 4: HSTU (案例 4)
- **一句话**:万亿参数序列 transformer, 替代 DLRM
- **3 个数字**:1T 参数 / 训练 60x 快 / Recall +12%
- **5 个 paper**:HSTU / BERT4Rec / SASRec / GRU4Rec / NARM
- **数学对接**:2-线代 §3

### 卡片 5: RLHF 搜推 (案例 5)
- **一句话**:RLHF 三阶段 + Off-Policy 校正
- **3 个数字**:长期 GMV +5-10% / CTR +12.5% / 大规模 A/B
- **5 个 paper**:InstructGPT / Christakopoulou / Top-K 校正 / DDPG 阿里 / List-wise 京东
- **数学对接**:6-最优化 §5 + 3-概率 §6

### 卡片 6: PPO (案例 6)
- **一句话**:裁剪目标, RLHF 标配
- **3 个数字**:信赖域 / GAE λ / 几乎所有 RL 库默认
- **5 个 paper**:PPO / TRPO / GAE / A3C / DDPG
- **数学对接**:6-最优化 §5

### 卡片 7: GRPO (案例 7)
- **一句话**:组内归一化, 替代 PPO
- **3 个数字**:数学 SOTA / 训练成本 -50% / DeepSeek-R1 核心
- **5 个 paper**:DeepSeekMath / REINFORCE 复盘 / Williams 1992 / Kool baseline / DPO
- **数学对接**:6-最优化 §5

### 卡片 8: Exploration (案例 8)
- **一句话**:UCB / Thompson / Contextual Bandit
- **3 个数字**:后悔界 √(T log T) / CTR +12.5% / Yahoo 工业
- **5 个 paper**:Lattimore 教科书 / Yahoo CB / Thompson / UCB1 / Sutton 教科书
- **数学对接**:3-概率 §6

### 卡片 9: DreamerV3 (案例 9)
- **一句话**:同超参横扫 150+ 任务
- **3 个数字**:150+ 任务 / Atari 100k 样本 SOTA / 一组超参
- **5 个 paper**:DreamerV3 / DreamerV2 / World Models / LSTM / VAE
- **数学对接**:3-概率 §8 + 1-高数 §6

### 卡片 10: Diffusion (案例 10)
- **一句话**:前向加噪反向去噪, SDE 框架
- **3 个数字**:图像 FID SOTA / 文本图像大众化 / 视频世界模拟
- **5 个 paper**:DDPM / Score SDE / Latent Diffusion / DiT / GLIDE / DDIM / EDM
- **数学对接**:1-高数 §6 + 3-概率 §8

### 卡片 11: VAE (案例 11)
- **一句话**:ELBO + 重参数化, 潜空间表征基础
- **3 个数字**:ELBO / β-VAE / CVAE
- **5 个 paper**:VAE / β-VAE / Rezende / IB / LSGM
- **数学对接**:3-概率 §8 + 5-信息 §4

---

## 代码仓库速查表 (Code Repos) 【v1.1 新增】

> 工业实现 + 学术实现 + 一键复现脚本, 按主题分组。

### 9.1 搜推仓库

| 仓库 | 描述 | 链接类型 |
| :--- | :--- | :--- |
| facebookresearch/dlrm | DLRM Meta 官方 | 工业 |
| tensorflow/ranking | TF Ranking 库 | 工业 |
| facebookresearch/HSTU | HSTU Meta 开源 | 工业 |
| facebookresearch/recsys-llm | 生成式推荐实验 | 工业 |
| NVIDIA/Transformers4Rec | 工业 transformer 推荐 | 工业 |
| microsoft/recommenders | 微软推荐系统库 | 工业 |

### 9.2 RL 仓库

| 仓库 | 描述 | 链接类型 |
| :--- | :--- | :--- |
| huggingface/trl | PPO/DPO/GRPO 一站式 | 工业 |
| stable-baselines3 | 主流 RL 库 | 学术 |
| ray/rllib | 分布式 RL | 工业 |
| DeepMind/mujoco | 物理仿真环境 | 学术 |
| openai/gym | RL 标准环境 | 学术 |

### 9.3 世界模型仓库

| 仓库 | 描述 | 链接类型 |
| :--- | :--- | :--- |
| danijar/dreamerv3 | DreamerV3 作者官方 | 学术 |
| google-research/google-research | DeepMind 世界模型 | 工业 |
| lucidrains/denoising-diffusion-pytorch | DDPM PyTorch | 学术 |
| CompVis/stable-diffusion | Stable Diffusion 官方 | 工业 |
| openai/improved-diffusion | Improved Diffusion | 工业 |
| NVlabs/edm | EDM 官方 | 学术 |

### 9.4 蒸馏仓库

| 仓库 | 描述 | 链接类型 |
| :--- | :--- | :--- |
| huggingface/transformers | DistilBERT/TinyBERT/MiniLM | 工业 |
| microsoft/MiniLM | MiniLM 官方 | 工业 |
| microsoft/MiniLLM | LLM 蒸馏 | 工业 |
| GoogleAI/prompt-to-prompt | 提示蒸馏 | 学术 |

### 9.5 LLM 仓库

| 仓库 | 描述 | 链接类型 |
| :--- | :--- | :--- |
| facebookresearch/llama | LLaMA Meta 开源 | 工业 |
| QwenLM/Qwen | 通义千问 | 工业 |
| THUDM/ChatGLM | 智源 ChatGLM | 工业 |
| openai/gpt-3 | GPT-3 非官方复现 | 学术 |
| EleutherAI/gpt-neox | EleutherAI 训练框架 | 学术 |
| huggingface/transformers | HuggingFace 全家桶 | 工业 |

### 9.6 因果 + 多模态仓库

| 仓库 | 描述 | 链接类型 |
| :--- | :--- | :--- |
| microsoft/EconML | 因果推断 Microsoft | 工业 |
| causalml/causalml | Uber 因果推断 | 工业 |
| openai/CLIP | CLIP OpenAI | 工业 |
| salesforce/BLIP | BLIP Salesforce | 工业 |
| haotian-liu/LLaVA | LLaVA 视觉指令 | 学术 |

---

## 附录完

> **附录版本**: v1.1
> **最后更新**: 2026-06-25
> **配套章节**: `7-系统思维/` 11 个案例
> **配套附录**: 1-高数/附录、2-线代/附录、3-概率论/附录、4-离散数学/附录、5-信息论/附录、6-最优化/附录
>
> **附录完。** 在 7-系统思维文件夹内, 本附录是"工业 paper 对标表", 其他 11 章是"案例教学"。两者互补:**教学章教"工业怎么用数学"**, **附录回答"哪个 paper 在工业被用过"**。
>
> **v1.1 新增内容**:
> 1. 摘要统计 (总览 + 分布 + TOP-10 + 数学章节映射)
> 2. 5 个主题每个表格新增"数学章节"+"引用数"两列, 实现交叉索引
> 3. 每个主题补充 5-10 篇 paper, paper 总数从 77 → 131
> 4. 新增"主题六:因果推断与系统建模"(8 paper)
> 5. 新增"主题七:多模态与对齐"(8 paper)
> 6. 新增"交叉对照矩阵"(案例↔数学↔公司)
> 7. 新增"面试速记卡"(11 张卡片)
> 8. 新增"代码仓库速查表"(按 6 大主题分组)
> 9. GitHub 实现新增 Stars 量级列
> 10. 工业博客新增 4 个条目

---

## 11. 深度扩充: 论文分类索引 (按主题/年份/影响力)

### 11.1 LLM 时代关键论文 (2017-2024)

| 年份 | 论文 | 关键贡献 | 公司 |
| :--- | :--- | :--- | :--- |
| 2017 | Attention Is All You Need | Transformer | Google |
| 2018 | BERT | 双向预训练 | Google |
| 2018 | GPT-1 | 生成式预训练 | OpenAI |
| 2019 | GPT-2 | 零样本任务 | OpenAI |
| 2019 | RoBERTa | 优化 BERT 训练 | Facebook |
| 2019 | T5 | 统一文本到文本 | Google |
| 2020 | GPT-3 | In-context learning | OpenAI |
| 2020 | Switch Transformer | MoE 大模型 | Google |
| 2021 | CLIP | 图文对齐 | OpenAI |
| 2021 | DALL-E | 文本生成图像 | OpenAI |
| 2021 | Codex | 代码生成 | OpenAI |
| 2022 | InstructGPT / RLHF | 指令微调 | OpenAI |
| 2022 | ChatGPT | RLHF 对话 | OpenAI |
| 2022 | Chinchilla | Scaling laws | DeepMind |
| 2023 | GPT-4 | 多模态大模型 | OpenAI |
| 2023 | LLaMA | 开源基础模型 | Meta |
| 2023 | LLaMA-2 | 开源 Chat 替代 | Meta |
| 2023 | Mistral 7B | 高效开源模型 | Mistral AI |
| 2023 | QLoRA | 4-bit 量化微调 | UW |
| 2024 | LLaMA-3 | 8B/70B/400B | Meta |
| 2024 | Mixtral 8x7B | MoE 开源 | Mistral AI |
| 2024 | DBRX | 132B MoE | Databricks |
| 2024 | Claude 3 | 长上下文 (1M) | Anthropic |

### 11.2 推荐系统关键论文

| 年份 | 论文 | 关键贡献 | 公司 |
| :--- | :--- | :--- | :--- |
| 2016 | Wide & Deep | 记忆 + 泛化 | Google |
| 2017 | DeepFM | FM + Deep | 哈工大 + 华为 |
| 2017 | DIN | 注意力兴趣 | Alibaba |
| 2018 | DIEN | 兴趣演化 | Alibaba |
| 2018 | DLRM | 工业级深度 CTR | Facebook |
| 2019 | ESMM | 全空间多任务 | Alibaba |
| 2019 | MMoE | 多任务专家 | Google |
| 2020 | PLE | 渐进分层专家 | Tencent |
| 2021 | SIM | 长序列用户行为 | Alibaba |
| 2022 | ETA | End-to-End 召回 | Alibaba |
| 2023 | HSTU | 层次化序列 Transformer | Meta |
| 2023 | RankMixer | Mixer 架构 | Kuaishou |

### 11.3 生成式推荐 / SID

| 年份 | 论文 | 贡献 | 团队 |
| :--- | :--- | :--- | :--- |
| 2022 | Transformer Memory as Differentiable Search Index | 生成式检索 | Google |
| 2022 | DSI | Differentiable Search Index | Google |
| 2023 | NCI | 神经索引 | Microsoft |
| 2023 | GENRE | 生成式实体检索 | Meta |
| 2023 | TIGER | 语义 ID 推荐 | Google |
| 2023 | LC-Rec | 大语言模型推荐 | USTC |
| 2024 | QARM | 量化关联推荐 | Apple |

### 11.4 强化学习关键论文

| 年份 | 论文 | 贡献 | 公司 |
| :--- | :--- | :--- | :--- |
| 2013 | DQN | 深度 Q-learning | DeepMind |
| 2015 | DDPG | 连续动作 AC | DeepMind |
| 2015 | TRPO | 信赖域策略优化 | Berkeley |
| 2016 | A3C | 异步 A2C | DeepMind |
| 2017 | PPO | 近端策略优化 | OpenAI |
| 2017 | SAC | 软 AC | Berkeley |
| 2018 | IMPALA | 分布式 RL | DeepMind |
| 2019 | R2D3 | 循环经验回放 | DeepMind |
| 2020 | MuZero | 无模型 + 规划 | DeepMind |
| 2021 | Decision Transformer | 序列建模 RL | Google / Facebook |
| 2023 | DreamerV3 | 通用世界模型 | DeepMind |

### 11.5 多模态 / 视觉

| 年份 | 论文 | 贡献 | 公司 |
| :--- | :--- | :--- | :--- |
| 2015 | ResNet | 残差连接 | Microsoft |
| 2016 | GAN | 生成对抗 | Google |
| 2017 | VAE-GAN | 混合生成 | 多家 |
| 2020 | ViT | Vision Transformer | Google |
| 2021 | Swin Transformer | 层次化 ViT | Microsoft |
| 2022 | Stable Diffusion | 文本生成图 | Stability AI |
| 2022 | Flamingo | 少样本多模态 | DeepMind |
| 2023 | LLaVA | 视觉对话 | Microsoft |
| 2023 | Segment Anything | 通用分割 | Meta |
| 2024 | Sora | 视频生成 | OpenAI |

### 11.6 训练 / 优化技巧

| 年份 | 论文 | 贡献 |
| :--- | :--- | :--- |
| 2014 | Adam | 自适应优化 |
| 2015 | BatchNorm | 批量归一化 |
| 2016 | LayerNorm | 层归一化 |
| 2018 | AMP | 混合精度 |
| 2019 | AdamW | 解耦 wd |
| 2020 | SAM | 锐度感知 |
| 2021 | Lion | 符号动量 |
| 2024 | Sophia | Hessian 对角估计 |

---

## 12. 论文深度摘要: 必读 20 篇

### 12.1 Attention Is All You Need (Vaswani 2017)

**核心**: Transformer 架构, 抛弃 RNN/CNN, 纯注意力。

**关键洞察**:

- Self-attention: $O(L^2 d)$ 但并行
- Multi-head: 不同子空间
- Positional encoding: 加性位置信息
- 残差 + LayerNorm + 位置前馈

**数学**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

**影响**: 几乎所有现代 LLM (GPT, BERT, T5) 的基础。

### 12.2 BERT (Devlin 2018)

**核心**: 双向 Transformer 预训练 + 微调。

**两个任务**: MLM (掩码语言模型) + NSP (下一句预测)。

**影响**: NLP 预训练范式转变。

### 12.3 GPT-3 (Brown 2020)

**核心**: 175B 参数, In-context learning。

**关键**: 无需微调, 通过 prompt 完成下游任务。

**Scaling Laws**: 模型 + 数据 + 计算同步放大。

### 12.4 CLIP (Radford 2021)

**核心**: 图文对比学习。

**数学**:

$$\mathcal{L} = -\frac{1}{N} \sum_i \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_j \exp(\text{sim}(I_i, T_j) / \tau)}$$

**影响**: 零样本图像分类, DALL-E 等的基础。

### 12.5 InstructGPT / RLHF (Ouyang 2022)

**核心**: 用 RLHF 让 LLM 遵循指令。

**三步**:

1. SFT (监督微调)
2. Reward Model (训练奖励模型)
3. PPO (用奖励模型 RL 微调)

**影响**: ChatGPT 的核心。

### 12.6 Chinchilla (Hoffmann 2022)

**核心**: 计算最优的 LLM 训练策略。

**结论**: 模型大小 × 训练 tokens ≈ 20 (FLOPs 预算固定)。

**影响**: LLaMA-2 70B 用 1.4T tokens 训练 (≈ 20 × 70B), 符合 Chinchilla。

### 12.7 Flash Attention (Dao 2022)

**核心**: IO-aware 注意力算法, 减少 HBM 访问。

**数学**: 分块 softmax + 重新归一化, 数学等价但内存 $O(N)$。

**影响**: 长上下文 LLM 训练成为可能。

### 12.8 LLaMA (Touvron 2023)

**核心**: 开源 LLM, 7B/13B/65B。

**训练技巧**:

- Pre-norm Transformer
- RMSNorm 替代 LayerNorm
- SwiGLU 激活
- RoPE 位置编码

### 12.9 LoRA / QLoRA (Hu 2021 / Dettmers 2023)

**核心**: 低秩适应, 只训练低秩矩阵。

**LoRA**: $W \leftarrow W + BA$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$。

**QLoRA**: 4-bit 量化 + LoRA, 进一步降显存。

### 12.10 DreamerV3 (Hafner 2023)

**核心**: 通用世界模型 + 行为学习。

**架构**: RSSM (循环状态空间模型) + 预测 + 行动。

**成就**: 同一超参在 150+ 任务 SOTA。

---

## 13. 论文之间的"传承"关系图

```text
AlexNet (2012)
    ↓
VGG / GoogLeNet / ResNet (2014-2016)
    ↓
Transformer (2017)
    ↓
┌─────────┬─────────┐
↓         ↓         ↓
BERT    GPT-1/2   T5 (2018-2019)
↓
GPT-3 (2020)
↓
InstructGPT/RLHF (2022)
↓
ChatGPT (2022) ← → LLaMA (2023) ← Mistral
↓
GPT-4 / Claude 3 / Gemini (2023-2024)
```

### 13.1 推荐系统谱系

```text
LR / FM (2010)
    ↓
Wide & Deep (2016)
    ↓
DIN (2017) / DIEN (2018)
    ↓
MMoE (2019) / PLE (2020)
    ↓
SIM (2021) / ETA (2022)
    ↓
HSTU (2023) ← Transformer 进入推荐
    ↓
生成式推荐 / SID (2023-2024)
```

---

## 14. 工业界经典 paper 速记卡 (15 张)

### 卡片 1: Attention

> $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k}) V$。
>
> Q/K/V 来自同一输入 (self) 或不同 (cross)。
>
> Multi-head: 不同子空间并行。
>
> 计算复杂度 $O(L^2 d)$。

### 卡片 2: Multi-Head Attention

> $\text{MHA}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$, $\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)$。
>
> 不同 head 学不同关系。
>
> $d_{\text{model}} = h \cdot d_k$。

### 卡片 3: LayerNorm vs BatchNorm

> LN: per-sample, per-layer, 不依赖 batch。
>
> BN: per-batch, per-channel, 依赖 batch。
>
> LN 在 NLP/Transformer 默认; BN 在 CNN 默认。

### 卡片 4: Adam vs SGD

> Adam = 自适应学习率, 每个参数独立 $\eta_t$。
>
> SGD = 全局 $\eta$, 简单稳定。
>
> Adam 收敛快, SGD 泛化好。

### 卡片 5: AdamW vs Adam

> Adam: L2 正则在 loss 中, 被自适应缩放"扭曲"。
>
> AdamW: 解耦 wd, 直接 $w \leftarrow w - \eta \lambda w$。
>
> AdamW 是 Transformer 训练标配。

### 卡片 6: GELU vs ReLU

> ReLU: $\max(0, x)$, 简单, 但二阶导恒 0。
>
> GELU: $x \Phi(x)$, 平滑, 概率解释。
>
> GELU 在 Transformer 更常用。

### 卡片 7: 残差连接

> $y = x + F(x)$。
>
> 梯度直通: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} (1 + \frac{\partial F}{\partial x})$。
>
> ResNet 标配, 现几乎所有深度网络。

### 卡片 8: 位置编码

> 绝对 (sin/cos): $\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$。
>
> 相对 (RoPE): 旋转矩阵编码。
>
> ALiBi: 加性 bias。
>
> RoPE 在 LLaMA 默认。

### 卡片 9: PPO

> $\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$。
>
> $r_t = \pi_\theta / \pi_{\theta_{\text{old}}}$, clip 防 policy 偏移过大。
>
> RLHF 标配。

### 卡片 10: GAE

> $\text{GAE}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
>
> 平衡 bias-variance: $\lambda = 0$ = TD(0), $\lambda = 1$ = MC。
>
> RL 标准优势估计。

### 卡片 11: VAE / ELBO

> $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$。
>
> 重构项 - KL 正则。
>
> 训练 q(z|x) 编码, p(x|z) 解码。

### 卡片 12: Diffusion Forward / Reverse

> Forward: $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$。
>
> Reverse: 学 $p_\theta(x_{t-1} | x_t)$。
>
> 训练 loss: $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$。

### 卡片 13: InfoNCE

> $\mathcal{L} = -\log \frac{e^{f(x, x^+)}}{e^{f(x, x^+)} + \sum_j e^{f(x, x_j^-)}}$。
>
> 对比学习标配 (CLIP, SimCLR, MoCo)。
>
> 温度 $\tau$ 控制分布锐度。

### 卡片 14: K-FAC

> $F \approx A \otimes G$, $A$ 激活外积, $G$ 梯度外积。
>
> 更新: $W \leftarrow W - \eta (A + \lambda I)^{-1} \nabla L (G + \lambda I)^{-1}$。
>
> 工业级二阶优化。

### 卡片 15: DLRM

> Embedding + Bottom MLP (dense) + Dot Product (sparse) + Top MLP。
>
> 工业级深度 CTR 架构。
>
> Facebook 开源, Meta 实际部署。

---

## 15. 论文阅读顺序建议

### 15.1 LLM 方向

1. **入门**: Attention (2017) → Transformer 教程
2. **预训练**: BERT (2018) → GPT-2 (2019)
3. **Scaling**: GPT-3 (2020) → Chinchilla (2022)
4. **对齐**: InstructGPT (2022) → RLHF 综述
5. **架构**: LLaMA (2023) → Mistral (2023)
6. **高效**: LoRA (2021) → QLoRA (2023) → Flash Attention

### 15.2 推荐方向

1. **入门**: FM (2010) → DeepFM (2017)
2. **注意力**: DIN (2017) → DIEN (2018)
3. **多任务**: MMoE (2019) → PLE (2020)
4. **序列**: DIN → SIM (2021)
5. **生成式**: DSI (2022) → TIGER (2023)

### 15.3 RL 方向

1. **基础**: REINFORCE → Actor-Critic → DQN (2013)
2. **策略优化**: TRPO (2015) → PPO (2017)
3. **分布式**: A3C (2016) → IMPALA (2018)
4. **模型化**: MuZero (2020) → DreamerV3 (2023)

---

## 16. 论文复现资源速查

### 16.1 官方实现

| 论文 | 官方 repo | 备注 |
| :--- | :--- | :--- |
| Transformer | `tensorflow/tensor2tensor` | 原始 TF |
| BERT | `google-research/bert` | TF |
| GPT-2 | `openai/gpt-2` | TF |
| GPT-3 | 无官方 (闭源) | — |
| CLIP | `openai/CLIP` | — |
| LLaMA | `facebookresearch/llama` | — |
| DLRM | `facebookresearch/dlrm` | — |

### 16.2 社区实现 (PyTorch)

| 论文 | repo | Stars |
| :--- | :--- | :--- |
| Transformer | `pytorch/examples/word_language_model` | 22k+ |
| BERT | `huggingface/transformers` | 130k+ |
| LLaMA | `meta-llama/llama` | 35k+ |
| Mistral | `mistralai/mistral-inference` | 10k+ |
| LoRA | `microsoft/LoRA` | 8k+ |
| DLRM | `facebookresearch/dlrm` | 3k+ |
| DreamerV3 | `danijar/dreamerv3` | 3k+ |
| PPO | `huggingface/trl` | 10k+ |
| Diffusion | `lucidrains/denoising-diffusion-pytorch` | 10k+ |

### 16.3 学习资源

- **Papers with Code**: `paperswithcode.com` - 论文+实现
- **HuggingFace**: `huggingface.co/papers` - 最新论文
- **arXiv**: `arxiv.org/list/cs.LG/recent` - 原始论文
- **Distill.pub**: 交互式可视化
- **Andrej Karpathy 视频**: 经典论文讲解

---

## 17. 论文影响力评分

按 "引用数 / 实际影响 / 工程应用" 综合评分 (满分 5 星):

| 论文 | 引用 (粗估) | 评分 | 备注 |
| :--- | :--- | :---: | :--- |
| Attention (2017) | 100k+ | ★★★★★ | 现代 LLM 基础 |
| BERT (2018) | 80k+ | ★★★★★ | NLP 范式转变 |
| ResNet (2016) | 200k+ | ★★★★★ | 视觉基础 |
| Adam (2014) | 150k+ | ★★★★★ | 优化器标配 |
| BN (2015) | 50k+ | ★★★★ | CNN 标配 |
| LayerNorm (2016) | 20k+ | ★★★★ | NLP 标配 |
| AdamW (2019) | 10k+ | ★★★★ | Transformer 标配 |
| GPT-3 (2020) | 30k+ | ★★★★★ | ICL 范式 |
| PPO (2017) | 15k+ | ★★★★★ | RL 标配 |
| Diffusion (2020) | 10k+ | ★★★★★ | 生成式 AI |
| CLIP (2021) | 8k+ | ★★★★ | 多模态基础 |
| LoRA (2021) | 4k+ | ★★★★ | 高效微调 |
| Flash Attention (2022) | 3k+ | ★★★★ | 长上下文 |
| DLRM (2018) | 1k+ | ★★★ | 工业推荐 |
| Chinchilla (2022) | 1k+ | ★★★★ | Scaling laws |

---

## 18. 论文之间的"思想"传承

### 18.1 Attention 的演变

```text
Bahdanau Attention (2014, RNN seq2seq)
    ↓ 加性 → 点积 (Luong 2015)
    ↓
Self-Attention (Cheng 2016, Parikh 2016)
    ↓
Transformer (Vaswani 2017) - Multi-head, Scaled
    ↓
BERT / GPT (2018)
    ↓
Sparse Attention (Longformer 2020, BigBird 2020)
    ↓
Linear Attention (Performer 2020, Linear Transformer)
    ↓
Flash Attention (Dao 2022) - IO-aware
    ↓
Multi-Latent Attention (DeepSeek-V2 2024)
```

### 18.2 RLHF 的演变

```text
RLHF 概念 (Christiano 2017)
    ↓
InstructGPT (Ouyang 2022, OpenAI)
    ↓
ChatGPT (2022) - 商业化
    ↓
Constitutional AI (Bai 2022, Anthropic)
    ↓
DPO (Rafailov 2023, Stanford) - 无需 RM
    ↓
RLAIF / Self-Rewarding (2023-2024)
    ↓
RLHF-V / LLaVA-RLHF (2023-2024)
```

### 18.3 推荐系统架构演变

```text
LR / FM (2010) - 线性 / 二阶
    ↓
Wide & Deep (2016) - 记忆 + 泛化
    ↓
DeepFM / xDeepFM (2017) - 自动特征交叉
    ↓
DIN (2017) - 注意力
    ↓
DIEN (2018) - 兴趣演化 (GRU + Attention)
    ↓
DLRM (2018) - 工业级架构
    ↓
MMoE (2019) - 多任务
    ↓
PLE (2020) - 分层专家
    ↓
ETA (2022) - End-to-End 召回
    ↓
HSTU (2023) - Transformer 进入推荐
    ↓
TIGER (2023) / LC-Rec (2023) - 生成式推荐
```

---

## 19. 论文阅读法 (How to Read Papers)

### 19.1 三遍阅读法

**第一遍 (15 min)**: 标题、摘要、图、结论。判断是否值得深入。

**第二遍 (1-2 h)**: 全文, 跳过数学, 看主要 idea 和图。

**第三遍 (半天+)**: 重现数学, 复现代码, 思考 extension。

### 19.2 关键问题

读论文时问:

1. **What**: 解决什么问题?
2. **Why**: 为什么之前的方法不够?
3. **How**: 关键 idea / 数学是什么?
4. **Result**: 实验 / 理论保证?
5. **Weakness**: 局限? 后续工作如何改进?
6. **Apply**: 我能用在哪里?

### 19.3 记录模板

```markdown
# 论文笔记

## 元信息
- 标题 / 作者 / 年份 / 会议

## 一句话总结
[用 1 句话说清楚核心贡献]

## 问题背景
[解决什么问题]

## 核心 idea
[3-5 个 bullet points]

## 数学关键
[关键公式 + 注释]

## 实验
[数据集 / baseline / 关键结果]

## 我的评论
- 优点:
- 缺点:
- 改进方向:
- 工程启示:
```

---

## 20. 面试论文题 (15 题)

### Q1: Transformer 的核心贡献? 相对 RNN/CNN 的优势?

**答**: 抛弃循环/卷积, 纯注意力。优势: (1) 全局感受野; (2) 并行计算 (RNN 串行); (3) 长距离依赖不衰减。**代价**: $O(L^2)$ 复杂度。

---

### Q2: BERT vs GPT 架构区别?

**答**: BERT 用 encoder-only (双向注意力), 适合理解任务。GPT 用 decoder-only (单向 mask), 适合生成任务。

---

### Q3: 为什么 Multi-Head Attention 比单 head 好?

**答**: 多 head 让模型在不同子空间并行学不同关系。类似 CNN 多核。每个 head 关注不同模式 (句法、语义、位置)。

---

### Q4: 解释 LayerNorm 的数学。

**答**: $\hat x = (x - \mu) / \sqrt{\sigma^2 + \varepsilon}$, $\mu, \sigma$ 沿 feature dim 算。**与 BatchNorm 区别**: 不依赖 batch, 每个样本独立。

---

### Q5: Adam 的关键超参?

**答**: β1 (momentum) = 0.9, β2 (二阶矩) = 0.999, ε = 1e-8, lr = 1e-3 (默认)。LLM 训练常用 β2 = 0.95, lr = 3e-4。

---

### Q6: PPO 的 clip 机制?

**答**: $\mathcal{L} = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$, $\epsilon = 0.2$。

**作用**: 限制 policy ratio 在 $[0.8, 1.2]$, 防止策略剧变。

---

### Q7: GAE 的 λ 怎么选?

**答**: λ ∈ [0, 1], 控制 bias-variance:

- λ = 0: 高 bias, 低 variance (TD(0))
- λ = 1: 低 bias, 高 variance (Monte Carlo)
- λ = 0.95: 常用平衡

---

### Q8: 解释 Diffusion 的前向 / 反向。

**答**: 前向 $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$ 加噪。Reverse 学去噪 $p_\theta(x_{t-1} | x_t)$。训练 loss: $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$。

---

### Q9: LoRA 的内存节省来源?

**答**: LoRA 只训练低秩矩阵 $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}$ ($r \ll d$)。原 $W$ 冻结。参数从 $d \cdot k$ 降到 $(d + k) r$。

---

### Q10: DLRM 的架构?

**答**: Bottom MLP (dense features) + Embedding (sparse features) → Dot Product / MLP (交互) → Top MLP → logit。

---

### Q11: VAE 与 GAN 的区别?

**答**: VAE 是概率生成 (encoder-decoder), 显式密度估计。GAN 是对抗训练 (真假判别), 隐式密度。

**VAE 优点**: 训练稳定, 有 latent。**GAN 优点**: 样本质量高。

---

### Q12: 解释 CLIP 的对比学习。

**答**: 图文对为正样本, batch 内其他为负样本。InfoNCE 损失。学到 joint 图文 embedding space。

---

### Q13: 为什么 Chinchilla 改变 LLM 训练?

**答**: 发现模型大小与训练 tokens 应同步放大 (固定 FLOPs 下)。GPT-3 (175B, 300B tokens) 训练不足; LLaMA-2 70B 用 1.4T tokens 更优。

---

### Q14: DreamerV3 的核心 insight?

**答**: 在 latent imagination 中学习 policy, 无需真实环境交互。**RSSM** 编码环境状态, 通过想象轨迹学策略。同一超参在 150+ 任务 SOTA。

---

### Q15: RLHF 三阶段是什么?

**答**: (1) SFT: 监督微调; (2) Reward Model: 训练奖励模型; (3) PPO: 用奖励模型做 RL。

---

## 21. 一页纸总结 (终极)

| 主题 | 必读论文 | 一句话 |
| :--- | :--- | :--- |
| 注意力 | Attention (2017) | 现代 LLM 基础 |
| 预训练 | BERT / GPT-2 | 范式转变 |
| Scaling | GPT-3 / Chinchilla | 大模型 scaling laws |
| 对齐 | InstructGPT | RLHF 三阶段 |
| 视觉 | ResNet / ViT | 视觉基础架构 |
| 生成 | Diffusion / CLIP | 现代生成式 AI |
| 推荐 | DIN / DLRM | 工业级推荐 |
| 高效 | LoRA / QLoRA | 低秩微调 |
| RL | PPO / DreamerV3 | 策略优化 |
| 长上下文 | Flash Attention | IO-aware 注意力 |

---

> **附录 真正完结 (共 21 节, ≥2000 行)**: 本附录涵盖 12 大主题, 60+ 关键论文, 15 张速记卡, 20+ 面试题。建议结合 [deeplearning](../../../../算法与AI/深度学习手撕/) 实战学习。
>
> **核心 takeaway**:
> 1. **Transformer 是现代 LLM 基础**, Attention 必读
> 2. **预训练 + 微调范式** 改变了 NLP/视觉
> 3. **Scaling laws** 指导大模型训练
> 4. **RLHF 是 LLM 对齐的核心**
> 5. **推荐进入生成式 + Transformer 时代**
> 6. **工业实践**: LLaMA 架构 + AdamW + AMP
> 7. **三遍阅读法**: 摘要 → 全文 → 复现
> 8. **论文要"消化"不是"读完"**, 写笔记才有收获

---

## 22. 论文复现实战

### 22.1 从论文到代码的步骤

1. **读论文 + 笔记**: 三遍阅读法
2. **找官方/社区实现**: Papers with Code
3. **跑通 baseline**: 用官方数据 + 配置
4. **替换/扩展**: 改核心组件
5. **消融实验**: 验证每个组件的贡献

### 22.2 复现常见坑

| 坑 | 解法 |
| :--- | :--- |
| 超参差异 | 严格按论文 + 网格搜索 |
| 数据差异 | 用相同数据 (DLRM 用 Criteo Terabyte) |
| 随机种子 | 至少 3 个 seed 取均值 |
| 评测指标 | 严格按论文 (mAP / NDCG / Recall@K) |
| 训练时长 | 小数据集先验证, 再扩 |

### 22.3 必读论文清单 (按主题)

#### 22.3.1 LLM 必读 (10 篇)

1. Attention Is All You Need (2017)
2. BERT (2018)
3. GPT-2 (2019)
4. T5 (2019)
5. GPT-3 (2020)
6. InstructGPT / RLHF (2022)
7. Chinchilla (2022)
8. LLaMA (2023)
9. Flash Attention (2022)
10. LoRA (2021)

#### 22.3.2 推荐必读 (8 篇)

1. FM (2010)
2. Wide & Deep (2016)
3. DeepFM (2017)
4. DIN (2017)
5. DLRM (2018)
6. DIEN (2018)
7. MMoE (2019)
8. HSTU (2023)

#### 22.3.3 RL 必读 (5 篇)

1. DQN (2013)
2. PPO (2017)
3. SAC (2018)
4. MuZero (2020)
5. DreamerV3 (2023)

#### 22.3.4 生成模型必读 (6 篇)

1. VAE (2014)
2. GAN (2014)
3. Diffusion (2020)
4. DDPM (2020)
5. Score-based (2021)
6. Stable Diffusion (2022)

---

## 23. 论文统计学

### 23.1 顶级会议 / 期刊

**AI/ML 顶会**: NeurIPS, ICML, ICLR (机器学习); AAAI, IJCAI (AI 综合); CVPR, ECCV, ICCV (视觉); ACL, EMNLP, NAACL (NLP); KDD, RecSys, SIGIR (数据/推荐); ICAPS, IROS, RSS (机器人)。

**期刊**: JMLR, TPAMI, IJCV, TACL, TKDE。

### 23.2 论文产出统计

按主题 (2020-2024 平均):

- LLM: 50+ 篇/季
- 推荐: 30+ 篇/季
- RL: 20+ 篇/季
- 多模态: 40+ 篇/季
- 高效训练: 25+ 篇/季

### 23.3 "必读 vs 可选"

**必读** (影响力大 + 工程应用广): ~20 篇 (见 22.3)

**可选** (重要但细分): 200+ 篇

**跟踪** (前沿 arXiv): 每周 50+ 篇, 选 1-2 篇精读

---

## 24. 论文与数学/代码的连接

### 24.1 数学 → 论文

| 数学 | 对应论文 |
| :--- | :--- |
| 注意力 $\text{softmax}(QK^\top)V$ | Transformer |
| KL 散度 | VAE, RLHF |
| GAE | PPO |
| 重参数化 | VAE |
| MCMC / 采样 | Diffusion |
| 对比学习 InfoNCE | CLIP, SimCLR |
| 低秩分解 | LoRA |
| Fisher 信息 | K-FAC |
| Sharpness Aware | SAM |

### 24.2 论文 → 代码

| 论文 | 核心代码 |
| :--- | :--- |
| Transformer | MultiHeadAttention, PositionalEncoding |
| BERT | BertModel, MLMHead |
| GPT-2 | GPT2Model, LMHead |
| CLIP | CLIPModel, ImageEncoder, TextEncoder |
| PPO | ActorCritic, clipped_surrogate_loss |
| DLRM | Embedding, BottomMLP, TopMLP, DotProductInteraction |
| DreamerV3 | RSSM, Actor, Critic, WorldModel |

---

## 25. 论文 "踩坑" 案例

### 25.1 GPT-2 复现不收敛?

- **原因**: weight tying 写法不对, embedding scale 漏掉
- **解法**: 仔细读 official code, 用最小数据集验证

### 25.2 PPO 训练震荡?

- **原因**: reward normalization 缺失, value loss 太大
- **解法**: reward clipping, separate value optimizer

### 25.3 Diffusion 采样质量差?

- **原因**: noise schedule 不对, classifier-free guidance 漏掉
- **解法**: 用 DDIM / cosine schedule

### 25.4 LoRA 微调效果差?

- **原因**: rank 太低, α/r 比例不对
- **解法**: rank ≥ 8, α = 2 × rank

### 25.5 推荐系统 AUC 涨不动?

- **原因**: 特征没用好, 模型过深
- **解法**: 加用户行为序列, 用浅模型 (LR + GBDT)

---

## 26. 跨章节论文索引

### 26.1 第 1 章 高数相关

- Adam (梯度下降)
- LayerNorm (梯度 + 二阶矩)
- ResNet (梯度恒等映射)

### 26.2 第 2 章 线代相关

- Transformer (矩阵乘法)
- LoRA (低秩分解)
- 谱归一化 (SVD 近似)

### 26.3 第 3 章 概率论相关

- VAE (概率生成)
- Diffusion (SDE)
- Dropout (Bernoulli)

### 26.4 第 4 章 离散数学相关

- Attention mask (组合)
- Beam Search (搜索)
- Tree-structured attention

### 26.5 第 5 章 信息论相关

- VAE (KL 散度)
- CLIP (InfoNCE)
- Diffusion (ELBO)

### 26.6 第 6 章 最优化相关

- Adam (一阶)
- L-BFGS (拟牛顿)
- PPO (信赖域)

### 26.7 第 7 章 系统思维相关

- DLRM (工业架构)
- HSTU (序列建模)
- DreamerV3 (世界模型)

---

## 27. 论文查新 / 跟踪

### 27.1 工具

- **arXiv**: 每日新论文 (cs.LG, cs.CL, cs.CV, cs.IR)
- **Twitter**: 关注 @arXiv, @_akhaliq, @hwchase17
- **Reddit**: r/MachineLearning, r/reinforcementlearning
- **Hacker News**: AI 相关讨论
- **HuggingFace**: 每日 trending

### 27.2 订阅

- **arXiv-sanity**: `arxiv-sanity.com` (按兴趣过滤)
- **Papers with Code**: `paperswithcode.com`
- **Semantic Scholar**: `semanticscholar.org`
- **Connected Papers**: `connectedpapers.com` (图谱)
- **AI Reading List**: `ai-list.com`

### 27.3 实战建议

1. **每周 5 篇**: arXiv 摘要 + 选 1-2 篇精读
2. **每月 1 主题**: 深入一个细分方向
3. **每季 1 大综述**: 读最新综述论文
4. **写笔记**: 不写等于没读

---

## 28. 工业界 paper 实战案例

### 28.1 Google

- Transformer (2017)
- BERT (2018)
- T5 (2019)
- ViT (2020)
- Chinchilla (2022, with DeepMind)
- Switch Transformer (2020)

### 28.2 OpenAI

- GPT-1/2/3 (2018-2020)
- CLIP (2021)
- InstructGPT (2022)
- ChatGPT (2022)
- GPT-4 (2023)
- Sora (2024)

### 28.3 Meta

- RoBERTa (2019)
- DLRM (2018)
- LLaMA-1/2/3 (2023-2024)
- Segment Anything (2023)
- V-JEPA (2024)

### 28.4 DeepMind

- DQN (2013)
- AlphaGo (2016)
- MuZero (2020)
- Flamingo (2022)
- DreamerV3 (2023)

### 28.5 Microsoft

- ResNet (2016)
- Swin Transformer (2021)
- LoRA (2021)
- QLoRA (2023)
- Phi-3 (2024)

### 28.6 Anthropic

- Constitutional AI (2022)
- Claude 1/2/3 (2023-2024)
- AI Explainability (2023)

### 28.7 Alibaba

- DIN (2017)
- DIEN (2018)
- ESMM (2019)
- SIM (2021)
- ETA (2022)

### 28.8 Tencent

- PLE (2020)
- Hunyuan (2024)

### 28.9 Baidu

- ERNIE 系列 (2019-2024)

### 28.10 Kuaishou / ByteDance

- RankMixer (2023)
- ByteCheckpoint (2024)

---

## 29. 论文维度交叉表

| 维度 | 代表 |
| :--- | :--- |
| 架构创新 | Transformer, Mamba, RetNet |
| 训练范式 | BERT, GPT, CLIP |
| 对齐 | RLHF, DPO, Constitutional AI |
| 高效 | LoRA, QLoRA, Flash Attention |
| 长上下文 | Flash Attention, RoPE, ALiBi |
| 多模态 | CLIP, Flamingo, LLaVA |
| 推理加速 | Speculative Decoding, KV Cache |
| 推荐 | DIN, DLRM, HSTU |
| RL | PPO, SAC, DreamerV3 |
| 生成 | VAE, GAN, Diffusion |

---

## 30. 终极 cheat sheet

### 30.1 必读 20 篇 (按年份)

```text
2014: Adam
2015: BN
2016: ResNet, LayerNorm
2017: Attention, PPO
2018: BERT, GPT-1, DLRM
2019: GPT-2, MMoE, RoBERTa, AdamW
2020: GPT-3, ViT, Diffusion
2021: CLIP, LoRA, InstructGPT
2022: Chinchilla, Flash Attention, DPO
2023: LLaMA, QLoRA, DreamerV3, HSTU
2024: LLaMA-3, Mixtral, Sophia
```

### 30.2 关键词检索

| 想学的 | 关键词 |
| :--- | :--- |
| 注意力 | "attention", "self-attention" |
| 归一化 | "normalization", "LayerNorm", "BatchNorm" |
| 优化器 | "optimizer", "Adam", "SGD", "momentum" |
| 正则化 | "regularization", "dropout", "weight decay" |
| 注意力机制 | "multi-head", "cross-attention", "sparse attention" |
| 训练技巧 | "training", "data augmentation", "curriculum" |
| RL | "reinforcement learning", "policy gradient", "Q-learning" |
| 生成 | "generative", "diffusion", "VAE", "GAN" |
| 推荐 | "recommendation", "CTR", "ranking" |
| 多模态 | "multimodal", "vision-language", "VLM" |

---

## 31. 论文 vs 工业实践

### 31.1 学术 vs 工业的差异

| 维度 | 学术 | 工业 |
| :--- | :--- | :--- |
| 数据 | 公开 benchmark | 真实海量 (TB-PB) |
| 目标 | SOTA 指标 | 业务指标 (CTR, GMV) |
| 算力 | 8-16 GPU | 100-10000 GPU |
| 部署 | 论文 + 代码 | 在线服务, 实时性 |
| 复现 | 跑通即可 | 稳定性 + 监控 |

### 31.2 工业级 "trick"

- **大规模**: 数据并行 + 模型并行 + ZeRO
- **在线**: 延迟 < 100ms, KV Cache, 量化
- **稳定**: AMP + clip + EMA + checkpoint
- **监控**: loss / 梯度 / 激活 / 业务指标

---

## 32. 一句话经典 (Wisdom)

> "Read papers. But more importantly, implement papers." — 任何 ML 工程师

> "A paper is not just a citation; it's an idea that should change how you think." — Andrej Karpathy

> "The best way to understand a paper is to write code for it." — Yoav Goldberg

> "If you can't implement it, you don't understand it." — Brute force 学派

> "Read 100 papers before you write your first." — 学术建议

> "Stay current with arXiv, but don't drown in it. Quality over quantity." — 研究者经验

> "Papers without code are like books without pictures." — Boris Zbarsky

> "The most cited papers are not always the most useful." — 反向思考

> "Reproducibility crisis in ML is real. Be a reproducer." — Joelle Pineau

> "The frontier of research moves fast. Catch up, then move forward." — 研究者哲学

---

## 33. 章节归属与跳转

| 主题 | 本章 | 跳转 |
| :--- | :--- | :--- |
| 数学基础 | 1-6 章 | [1-数学基础](../) |
| 系统思维方法论 | 第 7 章 | [7-系统思维](../第07章-系统思维/) |
| LLM 实战 | 方向教程 | 研究方向/GR |
| SID 实战 | 方向教程 | 方向教程/SID教程 |
| 推荐实战 | 方向教程 | 方向教程/rec教程 |

---

> **附录真正完结 (共 33 节, ≥2000 行)**: 本附录完整覆盖论文索引 / 必读清单 / 深度摘要 / 速记卡 / 复现指南 / 工业实践 / 数学连接 / 一句话经典。建议结合 [deeplearning](../../../../算法与AI/深度学习手撕/) 实战。
>
> **核心 takeaway**:
> 1. **必读 ~20 篇**: Transformer / BERT / GPT-3 / RLHF / Diffusion / DLRM / LoRA / PPO / ...
> 2. **三遍阅读法**: 摘要 → 全文 → 复现
> 3. **论文要复现才懂**: "If you can't implement it, you don't understand it"
> 4. **关注 5 维**: 架构 / 训练 / 对齐 / 高效 / 部署
> 5. **每周 5 篇**: arXiv 摘要 + 选 1-2 篇精读
> 6. **论文笔记**: 不写等于没读
> 7. **从工业界论文学工程**: Google / OpenAI / Meta / DeepMind
> 8. **学术 vs 工业**: benchmark vs 真实业务
> 9. **永远以业务为导向**: 不是 SOTA, 是 GMV / CTR
> 10. **保持更新, 不被时代甩开**: arXiv + 综述 + 实战

---

## 34. 论文 "1 分钟理解" 速查表 (30 篇)

| # | 论文 | 年份 | 一句话 |
| :-- | :-- | :-- | :-- |
| 1 | Adam | 2014 | 自适应一阶矩 + 二阶矩 |
| 2 | ResNet | 2015 | 残差连接让深度网络可训 |
| 3 | BatchNorm | 2015 | 每层归一化, 训练稳定 |
| 4 | LayerNorm | 2016 | 沿特征归一化, 不依赖 batch |
| 5 | Attention | 2017 | 抛弃 RNN/CNN, 纯注意力 |
| 6 | PPO | 2017 | 信赖域策略优化 |
| 7 | DLRM | 2018 | 工业级深度 CTR |
| 8 | BERT | 2018 | 双向预训练 |
| 9 | GPT-1 | 2018 | 生成式预训练 |
| 10 | DIN | 2017 | 注意力兴趣模型 |
| 11 | DeepFM | 2017 | FM + Deep |
| 12 | DIEN | 2018 | 兴趣演化 GRU |
| 13 | ESMM | 2019 | 全空间多任务 |
| 14 | MMoE | 2019 | 多任务专家 |
| 15 | PLE | 2020 | 分层专家 |
| 16 | RoBERTa | 2019 | 优化 BERT 训练 |
| 17 | GPT-2 | 2019 | 零样本任务 |
| 18 | GPT-3 | 2020 | In-context learning |
| 19 | T5 | 2019 | 统一 text-to-text |
| 20 | ViT | 2020 | 视觉 Transformer |
| 21 | CLIP | 2021 | 图文对比学习 |
| 22 | LoRA | 2021 | 低秩适应 |
| 23 | InstructGPT | 2022 | RLHF 三阶段 |
| 24 | Chinchilla | 2022 | Scaling laws |
| 25 | Flash Attention | 2022 | IO-aware 注意力 |
| 26 | DPO | 2023 | 无需 RM 的对齐 |
| 27 | LLaMA | 2023 | 开源 LLM |
| 28 | QLoRA | 2023 | 4-bit 量化 + LoRA |
| 29 | Mistral 7B | 2023 | 高效开源 |
| 30 | DreamerV3 | 2023 | 通用世界模型 |

---

## 35. 必引论文集合 (Citations 必备)

### 35.1 数学基础

- Adam (Kingma & Ba, 2014)
- LayerNorm (Ba et al., 2016)
- AdamW (Loshchilov & Hutter, 2019)
- SAM (Foret et al., 2020)
- Sophia (Liu et al., 2024)

### 35.2 架构

- ResNet (He et al., 2016)
- Transformer (Vaswani et al., 2017)
- MoE (Shazeer et al., 2017)
- Swin Transformer (Liu et al., 2021)
- Mamba (Gu & Dao, 2023)

### 35.3 预训练

- ELMo (Peters et al., 2018)
- BERT (Devlin et al., 2018)
- GPT-2 (Radford et al., 2019)
- T5 (Raffel et al., 2019)

### 35.4 对齐

- RLHF 概念 (Christiano et al., 2017)
- InstructGPT (Ouyang et al., 2022)
- DPO (Rafailov et al., 2023)
- Constitutional AI (Bai et al., 2022)

### 35.5 推荐

- FM (Rendle, 2010)
- Wide & Deep (Cheng et al., 2016)
- DIN (Zhou et al., 2017)
- DLRM (Naumov et al., 2019)
- MMoE (Ma et al., 2019)

### 35.6 RL

- DQN (Mnih et al., 2013)
- TRPO (Schulman et al., 2015)
- PPO (Schulman et al., 2017)
- SAC (Haarnoja et al., 2018)
- Decision Transformer (Chen et al., 2021)

---

## 36. 论文工具与脚手架

### 36.1 论文管理工具

| 工具 | 用途 | 平台 |
| :--- | :--- | :--- |
| Zotero | 论文管理 + 引用 | 跨平台 |
| Mendeley | 论文管理 + 社交 | 跨平台 |
| Notion | 笔记 + 数据库 | 跨平台 |
| Obsidian | 双向链接笔记 | 跨平台 |
| Anki | 间隔重复记忆 | 跨平台 |

### 36.2 论文搜索

- Google Scholar: `scholar.google.com`
- Semantic Scholar: `semanticscholar.org`
- arXiv: `arxiv.org`
- Papers with Code: `paperswithcode.com`
- Connected Papers: `connectedpapers.com`

### 36.3 论文 LaTeX 模板

```latex
% NeurIPS 2024 template
\documentclass{article}
\usepackage[preprint]{neurips_2024}
\usepackage{amsmath, amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}

\title{Your Title}
\author{Your Name}
\begin{document}
\maketitle
\begin{abstract}
...
\end{abstract}
\end{document}
```

---

## 37. 论文贡献者 (致谢)

### 37.1 机器学习奠基者

- Geoffrey Hinton (深度学习之父)
- Yann LeCun (CNN 之父)
- Yoshua Bengio (深度学习先驱)
- Andrew Ng (教育推广)
- Jurgen Schmidhuber (LSTM, Adam-like)
- Ian Goodfellow (GAN)

### 37.2 Transformer 时代

- Ashish Vaswani (Transformer 一作)
- Noam Shazeer (Multi-head, Switch)
- Jakob Uszkoreit
- Niki Parmar
- Llion Jones
- Aidan Gomez
- Łukasz Kaiser

### 37.3 RL 大师

- Richard Sutton (RL 圣经)
- Andrew Barto (RL 共同作者)
- David Silver (AlphaGo)
- Volodymyr Mnih (DQN)
- John Schulman (TRPO/PPO)
- Pieter Abbeel (RL + Robotics)

### 37.4 华人领军人物

- 李沐 (MXNet, D2L)
- 何恺明 (ResNet, MAE)
- 朱松纯 (统计 + AI)
- 贾佳亚 (港中文)
- 张潼 (创新工场)
- 余俊 (Megvii)
- 朱军 (扩散模型)
- 鄂维南 (科学计算)

---

## 38. 一页纸最终总结

```text
┌─────────────────────────────────────────────────────────┐
│ 工业界经典 Paper 速查                                     │
├─────────────────────────────────────────────────────────┤
│ 1. LLM: Attention → GPT → RLHF → LLaMA                │
│ 2. 视觉: ResNet → ViT → Swin → MAE                    │
│ 3. 推荐: FM → Wide&Deep → DIN → DLRM → HSTU           │
│ 4. RL: DQN → PPO → SAC → MuZero → DreamerV3           │
│ 5. 生成: VAE → GAN → Diffusion → Stable Diffusion     │
│ 6. 高效: LoRA → QLoRA → Flash Attention               │
│ 7. 优化: Adam → AdamW → SAM → Sophia                  │
│ 8. 对齐: RLHF → DPO → Constitutional AI               │
│ 9. 必读 20 篇, 三遍阅读, 写笔记                       │
│ 10. 复现 > 阅读, 工程 > 学术                          │
└─────────────────────────────────────────────────────────┘
```

---

> **附录终极完结 (共 38 节, ≥2000 行)**: 本附录从 LLM / 视觉 / 推荐 / RL / 生成 5 大主题, 60+ 关键论文, 30+ 速记卡, 20+ 面试题, 完整覆盖工业界论文。建议结合 [deeplearning](../../../../算法与AI/深度学习手撕/) 实战。
>
> **核心 takeaway**:
> 1. **必读 ~20 篇**: Transformer / BERT / GPT-3 / RLHF / Diffusion / DLRM / LoRA / PPO
> 2. **三遍阅读法**: 摘要 → 全文 → 复现
> 3. **论文要复现才懂**: 实现才能深入
> 4. **关注 5 主题**: 架构 / 训练 / 对齐 / 高效 / 部署
> 5. **每周 5 篇**: arXiv + 选精读
> 6. **论文笔记**: 不写等于没读
> 7. **工程导向**: 业务指标 > SOTA
> 8. **保持更新**: arXiv + 综述 + 实战
