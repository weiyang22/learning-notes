---
title: 第11章-Scaling-up-规模化扩展 · Datawhalechina 补充
tags:
  - LLM
  - 分布式训练
  - Scaling
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/so-large-lm ch08 分布式训练与 diy-llm（MoE/缩放定律），补充数据/模型/流水线并行的系统设计与 Scaling Law 实证
companion_to: 第11章-Scaling-up-规模化扩展.md
source: https://github.com/datawhalechina/so-large-lm ; https://github.com/datawhalechina/diy-llm
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Scaling-up 规模化扩展 · Datawhalechina 对照补充

> 本文件是对同目录《第11章-Scaling-up-规模化扩展.md》的补充，内容整理自 [so-large-lm](https://github.com/datawhalechina/so-large-lm)（大模型基础，CS324 风格）**ch08 分布式训练**与 [diy-llm](https://github.com/datawhalechina/diy-llm)（覆盖预训练数据/Tokenizer/Transformer/MoE/分布式/Scaling Laws 的渐进作业）。原文档讲 Scaling 的全景叙事；so-large-lm ch08 的增量是**并行策略的系统性设计**（数据/模型/流水线/混合），diy-llm 的增量是**Scaling Laws 的作业级实证**。

---

## 一、补充来源地图

| 来源章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| so-large-lm ch08.1 数据并行 | DDP/ZeRO 1-2-3 | 第11章正文 | ZeRO 分档的显存账本 |
| so-large-lm ch08.2 模型并行 | 张量并行（Megatron 切分） | 第11章正文 | **核心补充**：切分位置的通信分析 |
| so-large-lm ch08.3 流水线并行 | GPipe/PipeDream、bubble 分析 | 第11章正文 | **核心补充**：气泡率公式 |
| so-large-lm ch08.4 混合策略 | 3D 并行 + 优化器状态分片 | 第11章正文 | 工业组合公式 |
| diy-llm 作业 | Scaling Laws 实证、MoE | 第11章正文 | 动手拟合缩放曲线 |

---

## 二、核心补充内容

### 2.1 并行策略的显存-通信账本（so-large-lm ch08）

so-large-lm ch08 用统一的"显存账本 + 通信代价"框架比较各并行策略，建议并入原文档的选型表：

| 策略 | 切分对象 | 通信 | 显存收益 | 关键代价 |
| :--- | :--- | :--- | :--- | :--- |
| 数据并行（DDP） | batch | 反向梯度 AllReduce | 无（仅梯度分片后为 ZeRO-1+） | 通信量 ∝ 参数量/步 |
| ZeRO-1/2/3 | 优化器状态/梯度/参数 | 分片通信换显存 | 参数+状态最多降 N 倍 | ZeRO-3 前向需参数 gather |
| 张量并行（TP） | 单层权重矩阵 | 前后向各 2 次 AllReduce | 层内显存 | **通信每层都发生，只适合卡内/NVLink 域** |
| 流水线并行（PP） | 层间 | 仅阶段边界 P2P | 层数维度 | 气泡（bubble） |
| 序列并行/上下文并行 | 序列维 | Attention/AllGather | 长上下文关键 | 实现复杂 |

**Megatron 切分的核心结论**（ch08.2）：MLP 按列切 W1、按行切 W2 时，前向只需一次 AllReduce；Attention 的 QKV 按头切——**切分位置决定通信次数**，这是"为什么 TP 要放在 NVLink 域内"的根因。

**流水线气泡率公式**（ch08.3）：micro-batch 数 m、阶段数 p 时，GPipe 气泡率 $= \frac{p-1}{m+p-1}$；工程上取 $m \ge 4p$ 把气泡压到 <20%，配 1F1B 调度把激活显存从 O(m) 压到 O(p)。

### 2.2 3D 并行的组合公式（so-large-lm ch08.4）

工业组合的经验公式：**TP 放卡内（NVLink）→ PP 放节点间 → DP 包裹全局**；dp_size = 总卡数 / (tp_size × pp_size)。so-large-lm 给出 175B 模型的示例配比推导，可直接作为原文档选型章的算例。

### 2.3 Scaling Laws 的作业级实证（diy-llm）

diy-llm 把 Scaling Law 做成渐进作业，正好补原文档"讲结论、缺实证"的部分：

1. **Chinchilla 复算**：固定算力预算 C ≈ 6ND，扫多组 (N, D) 拟合 $L(N,D) = E + A/N^\alpha + B/D^\beta$，验证"参数:数据 ≈ 1:20 token"的结论边界；
2. **小模型外推**：用 ≤100M 模型的 scaling 曲线外推大模型 loss——这是"为什么工业界敢押注大模型"的方法论本身；
3. **MoE 的 scaling 例外**：MoE 的有效参数 ≈ 激活参数，diy-llm 作业里对比相同激活参数下 dense/MoE 的曲线（衔接本地《手撕/第05章-06 MOE》与《Mamba》目录）。

---

## 三、可直接动手的实践

1. 用 2.1 的账本对 7B 模型算一笔：fp16 参数 + AdamW 状态（fp32 参数/动量/方差）的显存，分别在 ZeRO-1/2/3 × 8 卡下各是多少；
2. 按 2.2 公式为一个 70B 模型设计 3D 并行配比（64 卡 8 节点），写出 tp/pp/dp 的取值与理由；
3. 用 diy-llm 的小模型作业拟合一条自己的 scaling 曲线（≥4 个模型规模），并与 Chinchilla 结论对照。

---

## 四、延伸链接

- so-large-lm 在线阅读：<https://github.com/datawhalechina/so-large-lm>
- diy-llm：<https://github.com/datawhalechina/diy-llm>
- 本地相关文件：第07章-LLM训练全流程实务（单机到多机实操）、第10章-Model-Infra（算子层）、基础/工程基础/工程能力/第03章-参数量与显存预估（显存账本基础）、手撕/第05章-06-MOE
