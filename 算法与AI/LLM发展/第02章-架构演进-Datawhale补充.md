---
title: 第02章-架构演进 · Datawhalechina 补充
tags:
  - LLM
  - 架构
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm 第二~三章（Transformer 与主流 LLM 主干），补充从原始 Transformer 到现代 LLM 主干的组件演进链
companion_to: 第02章-架构演进.md
source: https://github.com/datawhalechina/happy-llm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 架构演进 · Datawhalechina 对照补充

> 本文件是对同目录《第02章-架构演进.md》的补充，内容整理自 [happy-llm](https://github.com/datawhalechina/happy-llm) **第二章 Transformer 架构**与**第三章 LLM 主干模型**（LLaMA/Mistral/MiniMind 类）。原文档讲演进谱系；happy-llm 的增量是**每个组件变更的"代码级 diff"**——现代 LLM 主干相对原始 Transformer 的每一处修改及其动机。

---

## 一、补充来源地图

| happy-llm 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch2 Transformer | Attention/FFN/Norm/位置编码基础 | 架构演进早期段 | 基线锚点 |
| ch3 LLaMA 系 | RMSNorm/RoPE/SwiGLU/GQA/去 bias | 演进中后段 | **核心补充**：组件级 diff 清单 |
| ch3 MiniMind 系 | 小模型主干裁剪 | 演进谱系 | 小模型的取舍逻辑 |
| 附录D（本库） | 215M 复现 | （关联） | 代码落地 |

---

## 二、核心补充内容

### 2.1 原始 Transformer → 现代 LLM 主干的组件 diff 清单（happy-llm ch3）

这张"改动表"是面试与手撕的高频考点，建议并入原文档的演进主线：

| 组件 | 原始 Transformer (2017) | 现代 LLM 主干 (LLaMA 系) | 为什么改 |
| :--- | :--- | :--- | :--- |
| Norm | LayerNorm（Post-LN） | **RMSNorm**（Pre-LN 位置） | 去均值分支省算力；Pre-LN 训练更稳（深栈不塌） |
| 位置编码 | 正弦绝对 | **RoPE** 旋转位置 | 相对位置+线性外推性好，配合 NTK 缩放支持长上下文 |
| 激活 | ReLU | **SwiGLU**（门控线性单元） | 门控提升同参数量表达力 |
| 注意力 | MHA | **GQA/MQA** | KV cache 显存降 N 倍，质量近 MHA |
| Bias | QKV/FFN 全 bias | **全部去 bias** | 大规模训练下无增益、省参数 |
| 词表绑定 | input/output embedding 共享 | 大词表时**解绑** | 词表大时共享强制两者同维，解绑更灵活 |

**记忆法**：改动全部围绕"**训练稳定性 + 推理效率（KV cache）**"两条主线——每处 diff 都能归到这两条之一。

### 2.2 Encoder-only / Decoder-only / Enc-Dec 的现代分工（happy-llm ch2~3）

原文档有谱系；happy-llm 补充了"为什么今天 Decoder-only 一统天下"的技术论证，可直接引用：

1. **因果掩码 = 天然训练信号**：每位置都出 loss（Bert 式只训 15% mask 位），数据效率高；
2. **零样本/少样本的涌现**：自回归条件生成天然支持 in-context learning；
3. **工程统一**：一条管线服务预训练/SFT/对齐/推理，Encoder-only 的双向注意力优势被_embedding 检索类专门模型（BGE 类）分流。

### 2.3 小模型主干的裁剪逻辑（MiniMind 系）

小模型（本地附录 D 的 215M、MiniMind 26M 级）的配置取舍与"缩"的方向：d_model 与层数同步缩（保持深度/宽度比）、词表适度（64k 而非 150k，减少 embedding 稀疏）、head 数保持能整除 d_model——这些取舍在《手撕Transformer 补充》与《附录D 补充》里以代码形式出现，此处补"为什么这样配"的设计论。

---

## 三、可直接动手的实践

1. 把 2.1 的 diff 表逐项落到代码：从《手撕/第05章》的原始实现出发，改出 LLaMA 风格主干（四件套 + 去 bias），diff 一次训练冒烟；
2. 做 GQA 消融：MHA vs GQA(4组) 在同配置下的 KV cache 体积与生成速度对比；
3. 用 2.2 的三条论证检查自己是否能把"为什么 Decoder-only"讲成 60 秒面试答案。

---

## 四、延伸链接

- happy-llm ch2/ch3：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter2>
- 本地相关文件：第03章-位置编码与长上下文演进、第04章-注意力与效率演进、手撕/第05章-Transformer架构-Datawhale补充（代码侧）、附录D-从零构建LLM-Datawhale补充（215M 复现）、tiny-universe（手搓主线）
