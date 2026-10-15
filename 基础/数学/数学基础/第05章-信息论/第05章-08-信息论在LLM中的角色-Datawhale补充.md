---
title: 第05章-08-信息论在LLM中的角色 · Datawhalechina 补充
tags:
  - 数学
  - 信息论
  - LLM
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/so-large-lm（CS324 风格），补充语言模型的极大似然=最小交叉熵的严格视角与信息论度量在 LLM 评估中的应用
companion_to: 第05章-08-信息论在LLM中的角色.md
source: https://github.com/datawhalechina/so-large-lm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 信息论在 LLM 中的角色 · Datawhalechina 对照补充

> 本文件是对同目录《第05章-08-信息论在LLM中的角色.md》的补充，内容整理自 [so-large-lm](https://github.com/datawhalechina/so-large-lm)（大模型基础，Stanford CS324 风格）的建模与评估章节。原文档从信息论原理到 LLM 的映射已经完整；so-large-lm 的增量是**训练目标的严格表述**（极大似然 = 最小 KL = 最小交叉熵）与**评估度量的信息论解释**（perplexity 的正确用法）。

---

## 一、补充来源地图

| so-large-lm 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- |
| 语言模型任务定义（自回归分解） | 第05章-05（关联） | 交叉熵目标的推导起点 |
| 训练目标：极大似然 ↔ 最小交叉熵 ↔ 最小 KL | 本章正文 | **核心补充**：三个目标等价的严格表述 |
| 评估：perplexity 与其误用 | 本章正文 | **核心补充**：PPL 可比性条件 |
| Scaling 与 loss | 第11章（关联） | 交叉熵作为 Scaling Law 的因变量 |

---

## 二、核心补充内容

### 2.1 三个训练目标的等价链（so-large-lm 建模章节）

对自回归语言模型，以下三句话是**同一件事**，链条值得背下来：

```text
极大似然：max_θ Σ log p_θ(x_i | x_<i)
  ⟺ 最小 NLL：min_θ Σ -log p_θ(x_i | x_<i)
  ⟺ 最小交叉熵：H(p_data, p_θ) = H(p_data) + KL(p_data ‖ p_θ)
                （H(p_data) 与 θ 无关，故等价于最小 KL）
```

**推论**：训练 loss 的下界是数据的真实熵——loss 不会降到 0（除非数据可被模型完美预测），"loss 下降变慢"的部分原因是逼近数据熵底。这个视角解释了本地《第06章-05 交叉熵·困惑度·NLL》的现象学。

### 2.2 Perplexity 的可比性条件（so-large-lm 评估章节）

PPL = exp(平均 NLL) 有严格的**可比性前提**，误用清单：

1. **同词表同分词**：BPE 词表不同的两个模型，PPL 不可直接比（token 化改变序列熵）；
2. **同数据分布**：领域切换（新闻→代码）下 PPL 差异混合了"模型能力"与"数据熵差异"；
3. **同上下文窗口协议**： sliding window vs 全文上下文会影响条件长度；
4. **PPL 低 ≠ 下游强**：PPL 与下游任务的相关性是非线性的（scaling 文献的结论），选模型要看任务指标——这解释了为什么现代评估转向基准测试而非裸 PPL。

### 2.3 信息论度量在 LLM 系统中的延伸应用

把本章度量挂到 LLM 系统的三个位置：

| 度量 | 系统位置 | 用途 |
| :--- | :--- | :--- |
| 条件熵 / 每 token NLL | 推理日志 | 逐 token 困惑度定位"模型意外"的位置（数据质量分析） |
| KL 散度 | RLHF/GRPO | 策略相对参考策略的漂移预算（见第05章对齐补充文件） |
| 互信息 | 表征分析 | probing：表征保留了多少输入信息（线性探针的信息论口径） |

---

## 三、可直接动手的实践

1. 用等价链（2.1）手推：固定数据集上"最小化交叉熵损失"的训练曲线渐近值 ≈ 数据的一阶熵估计（用 n-gram 模型估计对照）；
2. 复现一个 PPL 误用案例：同模型在两种分词下的 PPL 差异，量化"分词不同导致不可比"；
3. 抽取一段真实文本的逐 token NLL 热力图，定位高困惑 token，判断是知识缺失还是数据噪声。

---

## 四、延伸链接

- so-large-lm：<https://github.com/datawhalechina/so-large-lm>
- 本地相关文件：第05章-05/06（交叉熵与 PPL 原理）、第05章-07 信息瓶颈（InfoNCE 衔接）、LLM发展/第05章-对齐技术演进-Datawhale补充.md（KL 预算的应用）、LLM发展/第11章-Scaling 补充（交叉熵作为 scaling 因变量）
