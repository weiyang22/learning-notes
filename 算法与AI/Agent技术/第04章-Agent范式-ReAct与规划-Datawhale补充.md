---
title: 第04章-Agent范式-ReAct与规划 · Datawhalechina 补充
tags:
  - Agent
  - ReAct
  - 规划
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/hello-agents 第四章（范式原理与手写实现），补充 ReAct/Plan-and-Solve/Reflection 三范式的代码骨架与选型判据
companion_to: 第04章-Agent范式-ReAct与规划.md
source: https://github.com/datawhalechina/hello-agents
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Agent 范式：ReAct 与规划 · Datawhalechina 对照补充

> 本文件是对同目录《第04章-Agent范式-ReAct与规划.md》的补充，内容整理自 [hello-agents](https://github.com/datawhalechina/hello-agents) **第四章 范式：原理与实现**。原文档讲范式谱系与论文脉络；hello-agents ch4 的增量是**三种范式的手写代码骨架**（不依赖框架，几十行可跑），这是"看懂范式"到"能改范式"的分水岭。

---

## 一、补充来源地图

| hello-agents ch4 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ReAct 实现 | Thought→Action→Observation 循环代码 | 范式谱系节 | **核心补充**：最小可跑循环 |
| Plan-and-Solve | 先全量规划再逐步执行 | 范式谱系节 | **核心补充**：规划-执行解耦骨架 |
| Reflection | 执行-反思-重试循环 | 范式谱系节 | **核心补充**：自我修正的触发与终止 |
| 框架实现对照 | ch13/14 用自研框架复现 | 第05章（关联） | 裸写 → 框架的迁移 |

---

## 二、核心补充内容

### 2.1 三范式代码骨架与差异定位（hello-agents ch4）

三范式的本质差异在**决策时机**：

```python
# ReAct：边想边做（交错决策）
while not done:
    thought, action = llm(prompt_with(history))
    observation = env.execute(action)
    history += (thought, action, observation)

# Plan-and-Solve：先想完再做（两阶段决策）
plan = llm(plan_prompt(task))            # 一次性产出步骤序列
for step in plan:                        # 逐步执行
    result = agent.execute(step, context)

# Reflection：做完再想（自我修正外环）
for attempt in range(max_attempts):
    result = solve(task)                 # 内层可为 ReAct/Plan-and-Solve
    critique = llm(reflect_prompt(task, result))
    if critique.passed: break
    task = refine(task, critique)
```

| 维度 | ReAct | Plan-and-Solve | Reflection |
| :--- | :--- | :--- | :--- |
| 决策时机 | 步级交错 | 事前全量 | 事后修正 |
| 长任务适用 | 中（上下文增长） | **好**（规划即记忆） | 好（可多轮） |
| 错误恢复 | 弱（链式中断） | 中（重规划） | **强** |
| token 成本 | 高（历史长） | 低 | 最高（多轮） |

### 2.2 范式组合的工程形态

hello-agents 强调范式是**可组合的原语**而非互斥选项，生产常见组合：

- **外层 Reflection + 内层 Plan-and-Solve**：先规划、执行、再反思整体——反思粒度是"计划级"而非"步级"（步级反思成本过高）；
- **ReAct + 预算控制**：步数上限 + 失败 n 次后切换为 Plan-and-Solve 重规划（降级链，与《第08章补充》2.2 的事故模式呼应）；
- **规划固化**：Plan-and-Solve 的 plan 存为结构化对象（可持久化/人工审核/断点续跑）——这是它比 ReAct 更"工程化"的根本原因。

### 2.3 选型判据表（可直接并入原文档）

| 判据 | 推荐 |
| :--- | :--- |
| 任务可预先分解、步骤清晰 | Plan-and-Solve |
| 环境反馈决定下一步（搜索/操作） | ReAct |
| 质量要求高、可承受多轮成本 | Reflection 外环 |
| 需要人工审核计划 | Plan-and-Solve（plan 结构化） |
| 上下文预算紧张 | Plan-and-Solve（历史短） |

---

## 三、可直接动手的实践

1. 按 2.1 骨架裸写三范式（各 <50 行，无框架），在同一任务（如两步计算问答）上对比步数/token/成功率；
2. 实现"Reflection + Plan-and-Solve"组合：反思只针对计划级失败，测量其相对纯 Plan-and-Solve 的提升与成本；
3. 给 ReAct 加预算控制降级链（2.2），压测长任务下的成本曲线。

---

## 四、延伸链接

- hello-agents ch4：<https://github.com/datawhalechina/hello-agents/tree/main/docs/chapter4>
- 本地相关文件：第05章-Agent工作流与编排（工程化衔接）、附录E-从零构建智能体-Datawhale补充.md（自研框架）、第08章-Agent评估与生产落地-Datawhale补充.md（降级与事故模式）、LLM发展/第06章-推理能力演进（CoT/ToT 的模型侧视角）
