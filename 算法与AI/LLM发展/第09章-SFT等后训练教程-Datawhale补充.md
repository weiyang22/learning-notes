---
title: 第09章-SFT等后训练教程 · Datawhalechina 补充
tags:
  - LLM
  - 后训练
  - SFT
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/post-training-of-llms 与 SWIFT/LLaMA-Factory 实践，补充后训练课程体系、数据工程与训练框架选型
companion_to: 第09章-SFT等后训练教程.md
source: https://github.com/datawhalechina/post-training-of-llms ; https://github.com/datawhalechina/self-llm
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# SFT 等后训练 · Datawhalechina 对照补充

> 本文件是对同目录《第09章-SFT等后训练教程.md》的补充，内容整理自 [post-training-of-llms](https://github.com/datawhalechina/post-training-of-llms)（Llama Creator 的后训练课程中译：SFT/DPO/评估）与 [self-llm](https://github.com/datawhalechina/self-llm) 的 SWIFT/LLaMA-Factory 实战章节。原文档讲 SFT 方法与数据规范；post-training 课程把后训练做成**课程体系**（含数据构造、训练、评估的闭环作业），正好补"工程闭环"。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| post-training-of-llms | SFT 数据构造 → LoRA/SFT → 评估闭环 | 第09章正文 | **核心补充**：课程级作业链 |
| self-llm（SWIFT） | ms-swift 框架实战 | 第09章正文 | **核心补充**：框架化微调 |
| self-llm（LLaMA-Factory） | WebUI/CLI 双模式 | 第09章正文 | 低代码路线 |
| happy-llm ch7 | 后训练任务分类（SFT/对齐/知识编辑） | 第09章正文 | 分类学 |

---

## 二、核心补充内容

### 2.1 SFT 数据构造的课程级方法（post-training-of-llms）

post-training 课程把"SFT 效果差"归因到数据，给出构造链条：

1. **种子来源**：人工撰写（少量精）→ 强模型蒸馏（大量粗）→ 现有数据集改造（Alpaca/ShareGPT 格式清洗）；
2. **质量过滤三道闸**：规则闸（长度/语种/格式）→ 模型闸（用打分模型筛）→ 去重闸（MinHash/语义去重）；
3. **多样性配比**：任务类型（问答/写作/代码/推理）× 难度 × 领域的三维配比表——数据配比决定模型"性格"，比超参影响更大；
4. **格式规范**：对话模板与 chat template 严格一致（角色标记错误是 SFT 后胡言的首因，与 daily-interview 题库 Q5 呼应）。

### 2.2 训练框架选型：SWIFT vs LLaMA-Factory vs 裸 Trainer

| 维度 | ms-swift | LLaMA-Factory | 裸 transformers Trainer |
| :--- | :--- | :--- | :--- |
| 上手 | CLI 丰富，魔搭生态 | **WebUI 最友好** | 全手动 |
| 多模态/Agent 微调 | 支持面最广 | 主流 LLM | 自己接 |
| 可控性 | 中（配置驱动） | 中 | **高** |
| 适用 | 生产化微调流水线 | 快速实验/教学 | 研究/定制 |

选型结论：教学与快速验证用 LLaMA-Factory WebUI；生产流水线用 SWIFT（数据-训练-部署一体）；需要新方法/新结构时回裸 Trainer（与《第07章补充》的 Trainer 细节衔接）。

### 2.3 后训练的评估闭环（post-training 课程收尾）

SFT/DPO 后的评估三层（与《第08章评估补充》Agent 侧指标互补）：

1. **自动基准**：MMLU/CEval 类知识集（防灾难遗忘）+ 任务专属集；
2. **LLM-as-judge**：与基座模型对比胜率（注意 judge 偏差控制）；
3. **人工抽检**：固定 50~100 条金标准集，每次迭代盲评——自动分数与人工分数的相关性要定期校准。

---

## 三、可直接动手的实践

1. 按 2.1 构造 500 条领域 SFT 数据（含三维配比表），用 LLaMA-Factory LoRA 微调 7B，前后对比任务专属集；
2. 故意混入 10% 角色标记错误的数据，观察训练后模型的对话格式崩坏，验证数据闸的必要性；
3. 建立评估闭环：跑一次"自动基准 + LLM-judge + 人工 50 条"三层评估，记录三层结论是否一致。

---

## 四、延伸链接

- post-training-of-llms：<https://github.com/datawhalechina/post-training-of-llms>
- self-llm：<https://github.com/datawhalechina/self-llm>
- 本地相关文件：第05章-对齐技术演进-Datawhale补充.md（DPO/GRPO 后续）、第07章-LLM训练全流程实务-Datawhale补充.md（Trainer 细节）、附录B-开源大模型实战-Datawhale补充.md（框架与显存门槛）、第08章-训练监控与权重续训
