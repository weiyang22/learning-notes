---
title: 第03章-位置编码与长上下文演进 · Datawhalechina 补充
tags:
  - LLM
  - RoPE
  - 长上下文
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm ch2 与 so-large-lm，补充 RoPE 的复数推导细节与长上下文扩展方法分类（外推/插值/NTK/YaRN）
companion_to: 第03章-位置编码与长上下文演进.md
source: https://github.com/datawhalechina/happy-llm ; https://github.com/datawhalechina/so-large-lm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 位置编码与长上下文 · Datawhalechina 对照补充

> 本文件是对同目录《第03章-位置编码与长上下文演进.md》的补充，内容整理自 [happy-llm](https://github.com/datawhalechina/happy-llm) ch2 的 RoPE 推导与 [so-large-lm](https://github.com/datawhalechina/so-large-lm) 的长上下文讨论。原文档讲演进谱系；两份资料补**数学细节与扩展方法的统一分类**。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| happy-llm ch2 RoPE | 复数视角的旋转推导 | 位置编码节 | **核心补充**：从二维旋转到多头推广 |
| so-large-lm 长上下文 | 外推问题与扩展方法 | 长上下文节 | **核心补充**：外推/插值二分法 |

---

## 二、核心补充内容

### 2.1 RoPE 的推导主线（happy-llm ch2）

RoPE 的目标：找一个位置相关的变换，使 **attention 内积只依赖相对位置**。推导主线（面试可白板版）：

1. **二维出发**：把 q、k 的每两维视为复数，位置 m 的变换是旋转 $q_m = q \cdot e^{im\theta}$；内积 $\langle q_m, k_n\rangle$ 的实部只含 $(m-n)\theta$——**旋转实现相对位置**；
2. **推广到 d 维**：不同维度对用不同频率 $\theta_i = 10000^{-2i/d}$（低维高频、高维低频，形成"钟表齿轮"结构）；
3. **实现等价**：旋转可写成对 q 的稀疏块乘（RoPE 缓存 cos/sin 表，逐元素乘），这就是代码里的 `apply_rotary_pos_emb`；
4. **性质**：相对性（内积仅依赖 m−n）、远程衰减（加线性偏置可强化）、与因果掩码正交。

### 2.2 长上下文扩展的统一分类（so-large-lm）

训练长度有限时扩展上下文的两条路线，方法都可挂进这个二分法：

| 路线 | 思想 | 代表 | 代价 |
| :--- | :--- | :--- | :--- |
| **外推（Extrapolation）** | 改注意力结构使超长位置"定义良好" | ALiBi（线性距离偏置）、滑动窗口 | 位置语义改变，需对齐预训练分布 |
| **插值（Interpolation）** | 把长位置压回训练区间 | Position Interpolation（PI）、**NTK-aware**（改 base）、**YaRN**（PI+NTK 按频率混合 + 温度） | 需少量长文微调恢复分辨率 |

**工程含义**：本地第03章的演进表可加"扩展操作"一列——base 缩放（NTK）是最常用的零训练扩展（把 θ 的 base 从 10000 调大，等价于频率整体下移）；YaRN 是当前开源长上下文的事实标准；KV cache 显存随长度线性涨，长上下文必须配 GQA/MLA（衔接《第02章补充》diff 表）与《第04章》效率演进。

### 2.3 面试问答链（自检）

```text
Q: RoPE 为什么用旋转而不是加？
A: 旋转在 attention 内积中天然产生相对位置项；绝对加法做不到纯相对。
Q: 为什么改 base 能扩长上下文？
A: 压低高频、保住低频齿轮结构——PI 是线性压缩所有频率，NTK 只压低频（外推高频），
   YaRN 按频率插值并配温度修正注意力熵。
```

---

## 三、可直接动手的实践

1. 用复数形式推导二维 RoPE 的内积相对性，并实现 `apply_rotary_pos_emb` 与 `nn.RoPE` 参考实现数值对齐；
2. 同一模型分别做 PI 与 NTK 扩展（不改权重），在 2k 训练长度外测 perplexity 曲线，比较两条路线的外推斜率；
3. 复现"远程衰减"实验：不同层的 attention 权重随相对距离的衰减谱（默认 vs +线性偏置）。

---

## 四、延伸链接

- happy-llm ch2：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter2>
- so-large-lm：<https://github.com/datawhalechina/so-large-lm>
- 本地相关文件：第03章正文（演进谱系）、第04章-注意力与效率演进（KV cache 侧）、手撕/第05章-03-位置编码、第02章-架构演进-Datawhale补充.md（diff 表）、LLM发展/附录D（215M 的 RoPE 实现）
