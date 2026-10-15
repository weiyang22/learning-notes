---
title: 第07章-LLM训练全流程实务 · Datawhalechina 补充
tags:
  - LLM
  - 训练实务
  - Transformers
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm 第六章（Transformers 框架训练实践），补充 Trainer 实战、超参调优与训练产物的框架级细节
companion_to: 第07章-LLM训练全流程实务.md
source: https://github.com/datawhalechina/happy-llm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# LLM 训练全流程实务 · Datawhalechina 对照补充

> 本文件是对同目录《第07章-LLM训练全流程实务.md》的补充，内容整理自 [happy-llm](https://github.com/datawhalechina/happy-llm) **第六章 Transformers 框架实战**（Trainer/TrainingArguments、Train-on-215M 全流程）。原文档讲训练全流程的方法论；happy-llm ch6 的增量是**框架级实操**——从数据集封装到 Trainer 配置到训练后检查的每一步代码，且与附录 D 的 215M 复现是同一套工程。

---

## 一、补充来源地图

| happy-llm ch6 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| 数据集封装 | Dataset/预处理/打包分块 | 数据准备节 | **核心补充**：token 流打包的正确姿势 |
| Trainer 与 TrainingArguments | 超参、调度、保存策略 | 训练实务节 | **核心补充**：参数逐项讲义 |
| 215M 全流程 | 与附录D 联动 | 全流程节 | 端到端参照（本库附录D 补充的姊妹） |
| SWIFT 实战 | 微调框架（后训练） | 第09章（关联） | 框架化微调 |

---

## 二、核心补充内容

### 2.1 数据打包的框架级细节（happy-llm ch6）

预训练数据进 Trainer 前的处理链，几个容易错的点：

1. **流式拼接 + 定长切块**：文档间加 EOS 拼接成长 token 流 → 按 `seq_len` 硬切（相邻块无重叠）——每块都是完整训练样本；`attention_mask` 全 1（拼接式），若用 padding 式则要配 `labels=-100` 屏蔽；
2. **labels = input_ids 的移位**：语言建模头内部移位（因果移位在 forward 里做），自己实现时别双重移位；
3. **分块边界的文档截断**：长文档被切断是正常且必要的（覆盖完整语料），但**文档间必须 EOS 分隔**，否则模型学到跨文档接续的伪相关。

### 2.2 TrainingArguments 关键项速查（happy-llm ch6）

| 参数组 | 关键项 | 实务建议 |
| :--- | :--- | :--- |
| 优化 | lr / warmup / scheduler | 预训练 cosine + 3% warmup；微调 linear 短 warmup |
| 批量 | per_device × grad_accum | 有效 batch = 二者乘积 × 卡数，先定有效 batch 再反推 |
| 精度 | bf16 / gradient_checkpointing | 显存紧张先开 checkpointing（约省 40% 激活，慢 ~30%） |
| 保存 | save_strategy / save_total_limit | 与《PTH 补充》七件套衔接；断点必含 optimizer/scheduler/rng |
| 评估 | eval_strategy / logging_steps | eval loss 与 train loss 分开看斜率（过拟合信号） |

### 2.3 训练后的标准检查单（happy-llm 215M 流程收尾）

训练跑完 ≠ 训练成功，happy-llm 的收尾检查可直接抄：

1. loss 曲线平滑性（毛刺 = lr 过大或数据脏）；
2. 生成冒烟：固定 prompt 采样，检查标点/重复/语种是否正常；
3. 权重健康：逐层范数无 NaN/异常（衔接《PTH 补充》2.3）；
4. 小基准回归：用固定评测集对比上次 checkpoint（防"训崩但 loss 没炸"）。

---

## 三、可直接动手的实践

1. 用 happy-llm ch6 代码在单卡跑完 215M 级小模型的预训练（或其 1/4 规模），全流程走通 2.3 检查单；
2. 故意制造两个错：数据无 EOS 拼接、双重移位 labels，观察 loss 与生成质量的变化并记录；
3. 做一次断点续训演练：中途 kill，从最近 checkpoint 续训，验证 loss 曲线连续。

---

## 四、延伸链接

- happy-llm ch6：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter6>
- 本地相关文件：LLM发展/附录D-从零构建LLM-Datawhale补充.md（215M 复现姊妹篇）、第08章-训练监控与权重续训（监控侧）、第09章-SFT等后训练教程（SWIFT 衔接）、搜推广与生成式推荐/PTH教程/第01章补充（checkpoint 工程）
