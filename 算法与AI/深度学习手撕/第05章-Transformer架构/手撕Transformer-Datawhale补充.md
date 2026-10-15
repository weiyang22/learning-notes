---
title: 第05章-Transformer架构 · Datawhalechina 补充
tags:
  - 深度学习
  - 手撕
  - Transformer
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm 第二章（手把手搭 Transformer）、thorough-pytorch 第十章（源码解读）与 leedl-tutorial，补充手撕 Transformer 的逐模块代码参照与作业
companion_to: 第05章-Transformer架构/（整章 10 个小节）
source: https://github.com/datawhalechina/happy-llm ; https://github.com/datawhalechina/thorough-pytorch ; https://github.com/datawhalechina/leedl-tutorial
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 手撕 Transformer · Datawhalechina 对照补充

> 本文件是对同目录《第05章-Transformer架构》整章（Self-Attention / 多头注意力 / 位置编码 / LayerNorm / 完整 Block / MoE / 训练稳定性 / Linear-Attention / Mamba 与 SSM / GroupNorm 共 10 节）的补充。三份 datawhale 资源的增量：**happy-llm ch2 的"手把手搭 Transformer"完整代码**、**thorough-pytorch 第十章的经典模型源码逐行解读**、**leedl-tutorial 的 Self-attention 作业**（动手机会）。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地小节 | 补充价值 |
| :--- | :--- | :--- | :--- |
| happy-llm ch2 | 从注意力到完整 Transformer 的搭建全过程 | 01~05 节 | **核心补充**：逐模块可运行代码 |
| thorough-pytorch 第十章 | ResNet/Swin/ViT 等源码逐行解读 | 03~05 节（架构侧） | 工业实现 vs 论文实现的差异 |
| leedl-tutorial 作业 | Self-attention 相关作业（预测天气等） | 01~02 节 | 动手作业 |
| d2l-ai-solutions-manual | 《动手学深度学习》习题解答 | 01~07 节 | 习题解答参照 |
| hack-rnns / hand-bert | 用线性层从零搭 NLP 网络 / 手搭 BERT | 01~05 节 | 更"裸"的实现视角 |

---

## 二、核心补充内容

### 2.1 手撕路线图（happy-llm ch2 的实现顺序）

happy-llm 给出的搭 Transformer 顺序，与本地 01→05 节的顺序互补（它强调"自底向上、每步可测"）：

```text
① Scaled Dot-Product Attention（单头，先对齐手算结果）
② Multi-Head Attention（reshape 的正确性：head 维与 seq 维不能搞反）
③ 位置编码（正弦式 vs 可学习式：各写一版并对比外推能力）
④ LayerNorm + FFN + 残差（Pre-LN vs Post-LN 两种 Block 顺序）
⑤ 完整 Encoder/Decoder Block + 掩码（padding mask 与 causal mask 分开测）
⑥ 组装并过拟合一个小语料（train loss → 0，验证实现正确性）
```

**自检锚点**（happy-llm 强调，本地各节可对照）：每写完一个模块用**数值测试**验证——如单头注意力对"全同 query"应输出值的均匀加权；causal mask 下位置 i 的 attention 权重对 j>i 必须为 0。

### 2.2 论文实现 vs 工业实现的差异点（thorough-pytorch 第十章）

本地小节按论文结构讲；thorough-pytorch 的源码解读补充工业落地的常见改动，面试高频：

| 位置 | 论文版 | 工业版（ViT/Swin 等） |
| :--- | :--- | :--- |
| Norm 位置 | Post-LN | Pre-LN（训练更稳，大模型标配） |
| 激活 | ReLU | GELU / SwiGLU |
| 位置编码 | 正弦 | RoPE / 可学习绝对位置（ViT） |
| Attention | 全局 | 窗口/线性/FlashAttention（IO 感知） |
| Bias | 有 | QKV 常去 bias（LLaMA 系） |

### 2.3 Linear-Attention 与 Mamba 的学习入口（对应本地 08/09 节）

本地 08/09 节讲原理；动手侧的补充入口：diy-llm 的 MoE/架构作业与 LLM发展/Mamba 目录（本地已有）互为参照；手撕目标建议定为"用 chunk 化递推实现一个最小线性注意力并对比 softmax attention 在 4k 序列上的显存-速度"——这是把 08/09 节从"看懂"推进到"写过"的最短路径。

---

## 三、可直接动手的实践

1. 按 2.1 顺序手写完整 Transformer（禁用 nn.Transformer），每模块配数值自检；最后在 toy 语料上过拟合验证；
2. 完成 leedl-tutorial 的 Self-attention 作业（天气预测），体验 attention 权重的可解释性分析；
3. 用 2.2 差异表逐项检查自己的实现与某开源实现（如 tiny-universe 或 LLaMA）的差异，写一页"论文→工业"对照笔记。

---

## 四、延伸链接

- happy-llm ch2：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter2>
- thorough-pytorch 第十章：<https://datawhalechina.github.io/thorough-pytorch/>
- leedl-tutorial：<https://github.com/datawhalechina/leedl-tutorial>
- tiny-universe（大模型白盒子）：<https://github.com/datawhalechina/tiny-universe>
- 本地相关文件：LLM发展/附录D-从零构建LLM-Datawhale补充.md（LLaMA2 组件级衔接）、第05章-09-Mamba与SSM（与 LLM发展/Mamba 目录互链）、手撕第01~04章（前置模块）
