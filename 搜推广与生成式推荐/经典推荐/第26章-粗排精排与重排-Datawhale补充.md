---
title: 第26章-粗排精排与重排 · Datawhalechina 补充
tags:
  - 经典推荐
  - 重排
  - 多样性
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/fun-rec（小麦书）上篇 ch4 重排多样性建模与 torch-rechub 排序模型实现，补充 MMR/DPP/Transformer 重排的推导与可运行代码
companion_to: 第26章-粗排精排与重排.md
source: https://github.com/datawhalechina/fun-rec ; https://github.com/datawhalechina/torch-rechub
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 粗排精排与重排 · Datawhalechina 对照补充

> 本文件是对同目录《第26章-粗排精排与重排.md》的补充，内容整理自 [datawhalechina/fun-rec](https://github.com/datawhalechina/fun-rec)（小麦书）**上篇 ch4 重排多样性建模**与 [datawhalechina/torch-rechub](https://github.com/datawhalechina/torch-rechub) 的排序模型实现。**重排多样性（MMR/DPP）是本地该章最薄弱的一环**——fun-rec 上篇 ch4 恰好给出完整推导，是本次对照中性价比最高的单点补充。

---

## 一、补充来源地图

| 来源章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| fun-rec ch4.1 基于贪心的重排 | MMR（最大边际相关性）、DPP（行列式点过程） | 第26章正文 | **核心补充**：多样性重排完整推导 |
| fun-rec ch4.2 基于个性化的重排 | Transformer 重排、排列组合重排 | 第26章正文 | 序列化重排范式 |
| torch-rechub 模型库 | 13+ 排序模型（DeepFM/DCN-v2/DIN/DIEN…）实现 | 全章 | 精排侧的可运行参照 |
| fun-rec ch3 多目标/多场景 | 任务依赖/损失融合/多塔 | 第04章、第18章（关联） | 粗排-精排的目标函数衔接 |

---

## 二、核心补充内容

### 2.1 MMR：最大边际相关性（fun-rec ch4.1.1）

**目标**：逐个选取物品，同时最大化"与用户相关"且"与已选集合不相似"：

$$\text{MMR} = \arg\max_{i \in R \setminus S} \left[ \lambda \cdot \text{sim}_1(i, u) - (1-\lambda) \max_{j \in S} \text{sim}_2(i, j) \right]$$

- $R$：候选集；$S$：已选集合；$\lambda$ 平衡相关性与多样性；
- 贪心复杂度 $O(k \cdot |R|)$，可在线上实时跑；
- 工程要点：`sim_2` 用物品 embedding 的余弦即可起步；类目打散、同作者去重等业务规则本质上是 MMR 的硬约束特例。

### 2.2 DPP：行列式点过程（fun-rec ch4.1.2）——理论最优雅的多样性模型

DPP 把"选出的子集既相关又多样"变成一个**概率模型的 MAP 推断**：

$$P(S) \propto \det(L_S), \quad L = \text{diag}(r_i) \cdot K \cdot \text{diag}(r_i)$$

- $L$：核矩阵；$r_i$：物品相关性（精排分）；$K$：相似度核（embedding 余弦）；
- **几何直觉**：$\det(L_S)$ 等于行向量张成的平行六面体体积——相关性决定向量长度，相似度决定向量夹角；两个高相关物品夹角小、体积小、联合概率低，自动被抑制。这就是"多样性"的数学化；
- **贪心 MAP 推断**：每步选使行列式增量最大者，增量有快速更新公式（Cholesky 分量），工程复杂度 $O(k^2|R|)$；
- **与 MMR 的关系**：λ=1 的贪心 MMR 可视为 DPP 的对数行列式目标的局部近似；DPP 有全局子模性保证，理论上更优。

### 2.3 个性化重排：Transformer 重排与排列组合重排（fun-rec ch4.2）

- **Transformer 重排**：把候选序列（含位置/上下文特征）输入轻量 Transformer，直接输出重排后的序列分——把"打分→排序"变成"序列→序列"，能建模 item 间相互影响（一个 item 的存在改变另一个的 CTR）；
- **基于排列组合的重排**：对 top-k 候选枚举/采样若干排列，用精排+多样性综合分挑最优排列；枚举上限控制在 $k \le 5$ 级别，否则组合爆炸；
- **两类的取舍**：Transformer 重排表达力强但需要训练与算力；排列组合即插即用，适合作为冷启动方案。

### 2.4 精排侧对照：torch-rechub 的可运行参照

torch-rechub 维护 13+ 个排序模型的 PyTorch 实现（DeepFM、Wide&Deep、DCN、DCN-v2、DIN、DIEN、MMoE 等），与本地《经典推荐/第22章 DLRM 经典算法详解》《第04章 多任务学习》的论文讲解一一对应。用法建议：本地章节读原理 → torch-rechub 跑对应模型的小数据集（Criteo/Movielens）→ 观察参数量与训练曲线，把"论文结构 → 代码结构"的映射建立起来（与本地《PTH教程》互补）。

---

## 三、可直接动手的实践

1. **实现贪心 DPP**：对精排 top-50 用 embedding 相似度 + 精排分构造 $L$，贪心选 10 个；与纯精排 top-10 对比类目覆盖数与 NDCG 损失；
2. **λ 敏感性实验**：扫 MMR 的 λ ∈ {0.3, 0.5, 0.7, 0.9}，画"多样性-相关性"帕累托曲线；
3. **对照阅读**：用 torch-rechub 的 DIN 实现对照《第22章》DIN 部分的结构图，逐模块打勾。

---

## 四、延伸链接

- fun-rec（小麦书）在线阅读：<https://datawhalechina.github.io/fun-rec/>
- torch-rechub：<https://github.com/datawhalechina/torch-rechub>
- 本地相关文件：第25章-召回与向量检索（上游候选）、第04章-多任务学习与第18章-多场景推荐（目标函数侧）、第27章-在线学习与AB实验（重排策略上线验证）、经典推荐/第22章-DLRM经典推荐算法详解
