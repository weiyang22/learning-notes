---
title: 第05章-对齐技术演进 · Datawhalechina 补充
tags:
  - LLM
  - 对齐
  - RLHF
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm 第八章（GRPO/OPD/Search-R1/ReTool）与 hello-agents 第十一章（Agentic-RL），补充从 SFT 到 GRPO 的训练实战与 Agentic 对齐前沿
companion_to: 第05章-对齐技术演进.md
source: https://github.com/datawhalechina/happy-llm ; https://github.com/datawhalechina/hello-agents
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 对齐技术演进 · Datawhalechina 对照补充

> 本文件是对同目录《第05章-对齐技术演进.md》的补充，内容整理自 [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) **第八章 Agentic-RL（GRPO/OPD/Search-R1/ReTool）**与 [hello-agents](https://github.com/datawhalechina/hello-agents) **第十一章 Agentic-RL（从 SFT 到 GRPO 的训练实战）**。原文档覆盖对齐技术谱系（RLHF/DPO 等）；两份 datawhale 资源的增量是**训练实操侧**：GRPO 的完整训练流程、可复现代码、以及 2025 年的 Agentic 对齐范式。

---

## 一、补充来源地图

| 来源章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| happy-llm ch8 | GRPO、OPD、Search-R1、ReTool | 第05章正文 | **核心补充**：GRPO 训练实战 |
| hello-agents ch11 | 从 SFT 到 GRPO 全流程 | 第05章正文 | SFT→RM→RL 三阶段的可运行管线 |
| happy-llm ch6 | Pretrain/SFT/LoRA（前置） | 第09章（关联） | 对齐前的 SFT 地基 |
| 数学/第07章-07 GRPO 案例本地文件 | 组内归一化的数学 | 本章正文 | 数学侧 ↔ 工程侧互引 |

---

## 二、核心补充内容

### 2.1 GRPO 训练实战的关键工程点（happy-llm ch8）

本地已有 GRPO 的数学（组内归一化优势 $\hat{A}_i=\frac{r_i-\mathrm{mean}(\{r\})}{\mathrm{std}(\{r\})}$）；happy-llm ch8 补的是**把它跑起来**的工程细节：

1. **组采样（group sampling）**：同一 prompt 采 G 个回答（典型 G=4~16）——显存预算 = G × rollout 长度，是 GRPO 最大的资源项；
2. **无价值网络的优势来源**：以组内奖励统计替代 critic，省下 RM/Value 模型的显存，但要求**奖励可量化且组内有区分度**——若组内奖励全相同（全对/全错），优势为 0，该 prompt 无梯度，因此 prompt 难度筛选（过滤全对/全错）是数据侧关键工序；
3. **KL 正则的位置**：GRPO 把参考策略 KL 加进损失（而非奖励），系数 β 的调参影响"能力提升 vs 语言漂移"的平衡；
4. **裁剪与组归一化的相互作用**：clip 范围在归一化优势下需重新校准（典型 ε=0.2 起步），过裁剪会吞掉组内差异信号。

### 2.2 从 SFT 到 GRPO 的完整管线（hello-agents ch11）

hello-agents 给出 Agent 场景的三阶段管线，与原文档的算法谱系互补：

```text
SFT（蒸馏专家轨迹，建立格式与工具调用习惯）
  → RM/规则奖励（轨迹级打分：任务完成率、工具正确性、步数成本）
  → GRPO/RL（在可验证奖励上提升多步决策）
```

工程要点：Agent RL 的奖励是**稀疏+延迟**的，hello-agents 建议先用过程奖励 shaping（步数惩罚、工具错误惩罚）把信号变密，再逐步退火到纯结果奖励。

### 2.3 两条 Agentic 对齐新范式（happy-llm ch8 后半）

- **Search-R1**：把"是否调用检索、何时检索"作为 RL 动作训练——对齐对象从"回答质量"扩展到"信息获取策略"；
- **ReTool（Coding Agent-RL）**：代码解释器调用的 RL 训练，奖励 = 程序执行结果正确性——可验证奖励（verifiable reward）路线的代表，也是 OPD（on-policy distillation，从强模型在线蒸馏）之外的另一条能力获取路径。

**对原文档的价值**：这两条把"对齐技术演进"的时间轴从 RLHF→DPO 延伸到 **RLVR（可验证奖励 RL）/ Agentic-RL**，是 2025~2026 对齐的主叙事。

---

## 三、可直接动手的实践

1. 用 happy-llm ch8 的 GRPO 代码在小模型（0.5~1.5B）上跑数学题 RLVR，记录组内奖励分布与"全对/全错 prompt 占比"随训练的变化；
2. 实现 prompt 难度筛选器（过滤零方差组），对比筛选前后的训练效率；
3. 把本地《数学/第07章-07》的 GRPO 推导与 happy-llm 代码逐项对齐（优势归一化 → clip → KL），形成"公式↔代码"对照表。

---

## 四、延伸链接

- happy-llm ch8：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter8>
- hello-agents ch11：<https://github.com/datawhalechina/hello-agents/tree/main/docs/chapter11>
- 作者维护的 agentic-rl-lab（更新更频繁）：<https://github.com/KMnO4-zx/agentic-rl-lab>
- 本地相关文件：第09章-SFT等后训练教程（前置 SFT）、第06章-推理能力演进（RLVR 与测试时扩展交叉）、数学/第07章-05/06/07（数学案例）、搜推广与生成式推荐/RL/RL/RL教程对照-Datawhale补充.md（PPO 前置）
