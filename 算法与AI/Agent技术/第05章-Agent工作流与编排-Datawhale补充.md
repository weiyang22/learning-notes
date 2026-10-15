---
title: 第05章-Agent工作流与编排 · Datawhalechina 补充
tags:
  - Agent
  - 工作流
  - 编排
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/hello-agents 第五章（Agentic 工作流）与 easy-vibe 工作流篇，补充工作流设计模式、人机协同与 Agent 系统设计原则
companion_to: 第05章-Agent工作流与编排.md
source: https://github.com/datawhalechina/hello-agents ; https://github.com/datawhalechina/easy-vibe
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Agent 工作流与编排 · Datawhalechina 对照补充

> 本文件是对同目录《第05章-Agent工作流与编排.md》的补充，内容整理自 [hello-agents](https://github.com/datawhalechina/hello-agents) **第五章 Agentic 工作流**（Anthropic《Building Effective Agents》中译方法论）与 [easy-vibe](https://github.com/datawhalechina/easy-vibe) 的工作流篇。原文档讲编排框架；两份资料补的是**设计模式层**——什么时候该用工作流、什么时候该用自主 Agent，这是编排选型的第一问题。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| hello-agents ch5（Anthropic 方法论） | 提示链/路由/并行/评估-优化/编排-工人 五模式 | 编排节 | **核心补充**：模式库 |
| hello-agents ch5 设计原则 | 简单性/透明性/工具精心设计三原则 | 编排节 | **核心补充**：反过度工程 |
| easy-vibe 工作流篇 | 长任务四机制、Agent Teams | 工作流节（关联） | 编码场景特化（工具手册已有） |
| 本地第06章 | 多智能体协作 | （关联） | 多智能体 vs 工作流辨析 |

---

## 二、核心补充内容

### 2.1 五种工作流模式的适用判据（Anthropic 方法论，hello-agents ch5）

Anthropic 的五模式 + 判据，比"上框架"更接近本质：

| 模式 | 机制 | 用 | 不用 |
| :--- | :--- | :--- | :--- |
| 提示链 Prompt Chaining | 顺序分解，步间可加程序检查点 | 任务可清晰分步 | 步骤间强耦合无法切 |
| 路由 Routing | 先分类再分流 | 输入类型异质（客服/检索） | 类型边界模糊 |
| 并行化 Parallelization | 切分同时跑 + 聚合 | 独立子任务/多数表决 | 子任务有依赖 |
| 评估-优化 Evaluator-Optimizer | 生成-评审双角色循环 | 有清晰评审标准 | 评审标准模糊（评审同样不可靠） |
| 编排者-工人 Orchestrator-Workers | 动态分解+分发+综合 | 子任务不可预知 | 固定流程（用提示链即可） |

**核心纪律（三原则）**：找最简单的方案、保持步骤透明可调试、把工程精力花在工具定义上（工具描述质量对成功率的贡献常大于换模型）——反"一上来就多智能体框架"。

### 2.2 工作流 vs 自主 Agent vs 多智能体的三层辨析

本地三章（05/06/04）的关系用一张决策图收拢：

```text
任务可预先完全定义？ ──是──→ 工作流（提示链/路由，最可控最便宜）
        │否
有可靠评审标准可循环？ ──是──→ 单 Agent + Reflection（自主但有界）
        │否
子任务天然并行/需要专长分工？ ──是──→ 多智能体（编排者-工人/辩论）
```

**成本与可靠性沿箭头递减**——每往右一级，token 成本与失败模式都显著增加；这与《第08章评估补充》的事故模式清单互为因果。

### 2.3 人机协同与检查点设计

hello-agents ch5 与 easy-vibe 共同强调的工程件（本地第05章正文可补）：

1. **人在环的位置**：计划审核（Agent 提 plan 人批准后执行，Plan-and-Solve 的工程化）优于逐轮审核（打断心流也不现实）；
2. **检查点持久化**：长工作流的中间产物落盘（结构化 JSON + 版本），失败从检查点续跑而非重来——easy-vibe 长任务四机制的通用化；
3. **可观测性内建**：每步的输入/输出/token/耗时落日志，是后续任何优化的前提（衔接《5-监控》）。

---

## 三、可直接动手的实践

1. 用 2.1 五模式改造一个现有 Agent 任务：先判定该用哪档（2.2 决策图），再实现，对比改造前后的成本与成功率；
2. 给长工作流加检查点续跑，人为中断一次验证恢复正确性；
3. 做"工具描述质量"消融：同一 Agent 只改工具 docstring 的详略，测调用成功率变化。

---

## 四、延伸链接

- hello-agents ch5：<https://github.com/datawhalechina/hello-agents/tree/main/docs/chapter5>
- easy-vibe：<https://github.com/datawhalechina/easy-vibe>
- 本地相关文件：第04章范式补充（单 Agent 内环）、第06章多智能体补充（协作层）、工具手册/第02章-Vibe-Coding补充（编码场景工作流）、5-监控与可观测性
