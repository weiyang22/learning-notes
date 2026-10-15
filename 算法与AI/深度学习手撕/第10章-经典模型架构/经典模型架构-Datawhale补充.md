---
title: 经典模型架构 · Datawhalechina 补充
tags:
  - 经典架构
  - ViT
  - LLaMA
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/thorough-pytorch 第十章（ResNet/ViT/Swin 源码解读）与 happy-llm（LLaMA/mini-GPT），补充跨模态经典主干的源码级对照
companion_to: 第10章-经典模型架构/
source: https://github.com/datawhalechina/thorough-pytorch ; https://github.com/datawhalechina/happy-llm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 经典模型架构 · Datawhalechina 对照补充

> 本文件是对同目录《第10章-经典模型架构》（经典架构对比 / LLaMA 完整实现 / 训练 mini-GPT 三节）的补充，内容整理自 [thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch) **第十章 经典网络源码解读**（ResNet/ViT/Swin 等逐行讲解）与 [happy-llm](https://github.com/datawhalechina/happy-llm)（LLaMA/mini-GPT 训练线）。本地三节偏 LLM 侧；thorough-pytorch 补**视觉主干**，使"经典架构"的版图完整。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地节 | 补充价值 |
| :--- | :--- | :--- | :--- |
| thorough-pytorch 第十章 | ResNet（残差）/ ViT（patch 化）/ Swin（层次化窗口）源码 | 10-01 对比 | **核心补充**：CV 主干源码 |
| happy-llm ch3~5 | LLaMA / mini-GPT 训练 | 10-02/03 | 训练侧衔接（《第07章补充》互引） |
| d2l-ai-solutions-manual | 经典网络习题 | 10-01 | 自查 |

---

## 二、核心补充内容

### 2.1 跨模态主干的"同一张图"（thorough-pytorch 第十章 × 本地 10-01）

ResNet → ViT → Swin → LLaMA 的主线其实是同一条：**"残差 + 归一化 + 注意力/卷积的堆叠"**，差异只在三处设计选择：

| 设计维度 | ResNet (2015) | ViT (2021) | Swin (2021) | LLaMA (2023) |
| :--- | :--- | :--- | :--- | :--- |
| 基本算子 | 卷积 | 全局自注意力 | 窗口注意力（+移位） | 因果自注意力 |
| 层次化 | 步长下采样 | 无（纯序列） | patch merging 逐级降分辨率 | 无（恒定分辨率 token 流） |
| 归一化 | BN | LN | LN | **RMSNorm（Pre）** |
| 位置信息 | 卷积天然局部 | 可学习绝对 PE | 相对位置 bias | **RoPE** |

**读源码的任务法**：每读一个主干，只回答三个问题——张量形状从进到出怎么变（尤其下采样/patch 化的位置）、残差分支里是什么、位置信息从哪来。thorough-pytorch 第十章的逐行讲解可按此法压缩。

### 2.2 LLaMA 完整实现的自检锚点（本地 10-02 × happy-llm ch3）

本地已给 LLaMA 实现与《附录 D》215M；自检锚点补三个高频 bug 位（happy-llm 社区常见踩坑）：

1. **RoPE 的应用位置**：应在 QK 进 attention 前分别施加（不是对拼接后的 QKV）；
2. **GQA 的 KV 复制**：`repeat_kv` 的维度对齐（head 维 repeat，别错成 batch 维广播）；
3. **causal mask 与 padding mask 的合成**：两个布尔掩码合并时 `&` 逐元素（防广播形状错误）。

### 2.3 mini-GPT 训练循环的教学价值（本地 10-03）

训练 mini-GPT 是"最小可信产线"：数据打包（与《第07章补充》2.1 同规）→ 训练 → 生成采样（temperature/top-k）→ loss 曲线判读。教学增量：把《第07章补充》的检查单在 100M 级上全部走一遍，成本极低但覆盖预训练全流程——是进入更大规模训练前的最后一道演练。

---

## 三、可直接动手的实践

1. 按 2.1 的三问法读 thorough-pytorch 第十章的 ViT 源码，画出 patch-embedding → 分类头的形状流转图；
2. 用 2.2 三个锚点审计自己的 LLaMA 实现（或附录 D 的 215M），修复任一处问题并写回归测试；
3. 完成 mini-GPT 训练一次，交付"loss 曲线 + 采样样例 + 检查单勾选记录"三件套。

---

## 四、延伸链接

- thorough-pytorch 第十章：<https://datawhalechina.github.io/thorough-pytorch/>
- happy-llm：<https://github.com/datawhalechina/happy-llm>
- 本地相关文件：第10章三节正文、第02章-架构演进-Datawhale补充.md（diff 表）、LLM发展/附录D-从零构建LLM-Datawhale补充.md、第07章-LLM训练全流程实务-Datawhale补充.md、手撕/手撕基础模块第01-04章-Datawhale补充.md
