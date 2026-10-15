---
title: 第01章-LLM发展历程总览 · Datawhalechina 补充
tags:
  - LLM
  - 发展史
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/so-large-lm ch1 与 happy-llm ch1，补充 LLM 三个时代的划分论证与开源模型谱系坐标
companion_to: 第01章-LLM发展历程总览.md
source: https://github.com/datawhalechina/so-large-lm ; https://github.com/datawhalechina/happy-llm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# LLM 发展历程总览 · Datawhalechina 对照补充

> 本文件是对同目录《第01章-LLM发展历程总览.md》的补充，内容整理自 [so-large-lm](https://github.com/datawhalechina/so-large-lm) **ch1 大模型概述**与 [happy-llm](https://github.com/datawhalechina/happy-llm) **ch1 LLM 综述**。原文档按时间轴罗列里程碑；两份综述的增量是**时代划分的论证**（为什么分三个时代、每个时代的"主导矛盾"是什么）与**开源谱系的坐标系**。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| so-large-lm ch1 | LLM 定义、资源收集、时代划分 | 总览节 | **核心补充**：三时代框架 |
| happy-llm ch1 | LLM 发展综述（BERT→GPT→ChatGPT→开源） | 总览节 | **核心补充**：两条路线的合流叙事 |
| tiny-universe | 手作系入口 | （关联） | 学完历史动手的衔接 |

---

## 二、核心补充内容

### 2.1 三个时代与各自的主导矛盾（so-large-lm ch1）

时间轴罗列容易"看过就忘"；用"每时代解决什么主导矛盾"记忆：

| 时代 | 时间 | 主导矛盾 | 代表突破 |
| :--- | :--- | :--- | :--- |
| 预训练架构时代 | 2018~2022 | **架构之争**（Encoder/Decoder/Enc-Dec） | BERT（理解）、GPT-3（少样本涌现）、T5（统一） |
| 对齐时代 | 2022~2024 | **能力→可用**（对齐指令与人类偏好） | InstructGPT/ChatGPT（RLHF）、对话格式范式 |
| 开源与推理时代 | 2024~ | **成本与能力边界**（开源复现、推理范式） | LLaMA 系开源生态、o1/R1 测试时扩展、MoE 降本 |

**叙事用法**：面试讲"LLM 历史"时按三时代讲主导矛盾（每个时代 1 个矛盾 + 2 个代表），比背年份有结构感得多。

### 2.2 两条路线的合流叙事（happy-llm ch1）

BERT（自编码/理解）与 GPT（自回归/生成）两条路线的分与合：

- 分：2018~2020 两派各自优化（BERT 系刷理解基准、GPT 系堆生成规模）；
- 合：指令微调 + RLHF 把"生成模型"变成"通用助手"，**Decoder-only 胜出**（论证见《第02章补充》2.2）；Encoder 路线并未消亡，而是分流为专门的 Embedding/检索模型（BGE 类，衔接《RAG 补充》检索器）；
- 这条"分→合→分流"叙事正好解释了本库章节结构（LLM 主干 + 检索模型 + 生成式推荐三条线）的存在理由。

### 2.3 开源模型谱系坐标（so-large-lm 资源收集）

开源系的谱系坐标（本地附录 B 的 50+ 模型可挂到这张图上）：

```text
LLaMA (2023.2) ──┬─ LLaMA2 (2023.7, 对齐版) ──┬─ LLaMA3/3.1/3.2/4 (长上下文/多模态)
                 └─ 衍生: Alpaca/Vicuna (指令微调始祖)
Qwen 系 (2023~) ── 多尺寸全家桶 + Qwen2.5/QwQ(推理) ── 中文开源主线
DeepSeek 系 ───── MoE + MLA + R1(RLVR 推理) ── 成本革命主线
GLM/InternLM/Mistral ── 各具特色的多模态/长文本/推理变体
```

读谱系的方法：每条线问三个问题——**架构上改了什么（对照《第02章补充》diff 表）、数据上做了什么、训练上用了什么范式（对照《第05/06/09章补充》）**。

---

## 三、可直接动手的实践

1. 把本地附录 B 的 50+ 开源模型按 2.3 的四条主线归类，标出没归进去的模型并说明其特色；
2. 用三时代框架写一份 500 字的"LLM 史"口头稿并录音自测；
3. 任选两个时代的交界论文（如 InstructGPT 或 LLaMA1）精读，验证"主导矛盾"的判断是否站得住。

---

## 四、延伸链接

- so-large-lm ch1：<https://github.com/datawhalechina/so-large-lm>
- happy-llm ch1：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter1>
- 本地相关文件：第01章正文、第02章-架构演进-Datawhale补充.md、附录B-开源大模型实战-Datawhale补充.md、tiny-universe（动手衔接）
