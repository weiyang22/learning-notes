---
title: 第06章-推理能力演进 · Datawhalechina 补充
tags:
  - LLM
  - 推理
  - RLVR
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm ch8 与 tiny-universe，补充 CoT→Self-Consistency→ToT→RLVR→测试时扩展的推理演进线及其训练配方
companion_to: 第06章-推理能力演进.md
source: https://github.com/datawhalechina/happy-llm ; https://github.com/datawhalechina/tiny-universe
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 推理能力演进 · Datawhalechina 对照补充

> 本文件是对同目录《第06章-推理能力演进.md》的补充，内容整理自 [happy-llm](https://github.com/datawhalechina/happy-llm) ch8（RLVR/可验证奖励）与 [tiny-universe](https://github.com/datawhalechina/tiny-universe)（RAG/Agent/评估手作）。原文档讲推理技术演进；本补充把"提示工程推理"与"训练推理"两条线收拢，并给出 RLVR 的最小复现路径。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| happy-llm ch8 | RLVR：数学/代码题的可验证奖励 RL | 推理演进后段 | **核心补充**：R1 式训练配方 |
| tiny-universe | CoT/Self-Consistency 的手作实现 | 推理演进前段 | **核心补充**：最小代码 |
| 本地《第05章对齐补充》 | GRPO 训练实战 | （关联） | 训练侧细节不重复 |

---

## 二、核心补充内容

### 2.1 推理演进的两条正交线（收拢视角）

本地第06章按时间罗列；用"干预点"重新分类后演进线更清晰：

```text
推理时干预（不训模型）：CoT（分步提示）→ Self-Consistency（多路采样投票）
  → ToT（树搜索 + 评估剪枝）→ 测试时扩展 o1 类（长思考预算 + 过程验证）
训练时干预（改模型）：SFT on 轨迹（蒸馏长思考）→ RLVR（可验证奖励强化）
  （R1 类：R1-Zero 纯 RLVR 涌现反思；R1 常规 = 冷启动 SFT + RLVR 两阶段）
```

**两线的交换律**：测试时扩展与 RLVR 可叠加——RLVR 提升采样质量基座，测试时扩展放大基座；这也是"推理成本从训练侧转移到推理侧"的产业叙事（衔接《第11章 Scaling 补充》2.3 的外推视角）。

### 2.2 RLVR 的最小配方（happy-llm ch8 × 第05章补充）

RLVR 与 RLHF 的关键差异是**奖励的来源**：人工/模型打分 → **程序化可验证**（数学答案比对、代码单测通过）。最小复现路径：

1. 数据：带标准答案的数学题（GSM8K 级）或带单测的代码题；
2. 奖励函数：`reward = f(最终答案正确) + 过程格式奖励`——**验证器是系统的灵魂**，验证器漏洞会被策略利用（reward hacking 的教科书来源）；
3. 训练：GRPO（组采样 + 组内归一化优势，衔接《第05章对齐补充》2.1 的工程点：全对/全错 prompt 筛除）；
4. 观察指标：答案正确率、平均思考长度（自发涌现变长是 R1-Zero 的标志现象）、语言混杂率（需语言一致性奖励压制）。

### 2.3 tiny-universe 的手作锚点

tiny-universe 用最小代码实现 CoT 与 Self-Consistency 的采样-投票循环，适合作为本章的"先写后读"入口：同一个数学题集上对比直接回答 / CoT / CoT+自一致性（n=8）三条基线的准确率-成本曲线——这张曲线是理解"测试时扩展为什么值得"的实证起点。

---

## 三、可直接动手的实践

1. 跑 tiny-universe 的 CoT/Self-Consistency 基线，产出"准确率 vs 平均 token 成本"曲线；
2. 按 2.2 配方搭最小 RLVR（小模型 + GSM8K 子集 + GRPO），观察思考长度是否随训练自发增长；
3. 做一次 reward hacking 复现：故意放宽答案比对（如只匹配数字），记录策略学会的"作弊"模式。

---

## 四、延伸链接

- happy-llm ch8：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter8>
- tiny-universe：<https://github.com/datawhalechina/tiny-universe>
- 本地相关文件：第06章正文（演进谱系）、LLM发展/第05章-对齐技术演进-Datawhale补充.md（GRPO 工程）、搜推广与生成式推荐/RL/RL/RL教程对照-Datawhale补充.md（PPO 前置）、面试/面试笔记/0-LLM面试笔记-2026-Datawhale补充.md（推理题源）
