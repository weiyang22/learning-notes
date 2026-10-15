---
title: 附录E-从零构建智能体 · Datawhalechina 补充
tags:
  - Agent
  - 自研框架
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/hello-agents 第七章自研框架与实战案例章，补充 HelloAgents 设计哲学、案例项目清单与"从零到框架"的迁移路径
companion_to: 附录E-从零构建智能体.md
source: https://github.com/datawhalechina/hello-agents
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 从零构建智能体 · Datawhalechina 对照补充

> 本文件是对同目录《附录E-从零构建智能体.md》的补充，内容整理自 [hello-agents](https://github.com/datawhalechina/hello-agents) **第七章 自研智能体框架**与**第十三章/十四章 实战案例**。原文档（ReAct 原理 + LangChain/CrewAI/AutoGen 代码）讲"用框架"；hello-agents 的路线是"**先造轮子再上车**"——自研 200 行级框架 HelloAgents 把 Agent 的本质收敛到最小，这是理解一切框架抽象的锚点。

---

## 一、补充来源地图

| hello-agents 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch7 自研框架 | HelloAgents：最小内核（LLM+记忆+工具+执行循环） | 附录E 正文 | **核心补充**：Agent 的最小完备集 |
| ch13/14 实战案例 | 智能客服 / 数据分析 / 自主研究等多个端到端项目 | 附录E 正文 | **核心补充**：案例脚手架 |
| 设计哲学 | "为什么抽象这么少" | 附录E（设计观） | 对框架臃肿的免疫 |
| Extra 系列 | 面试题/踩坑/自进化 | 《第08章补充》（关联） | 配套生态 |

---

## 二、核心补充内容

### 2.1 Agent 的最小完备集（HelloAgents 设计哲学）

HelloAgents 把智能体收敛为四个可插拔组件，任何框架都能映射回这套骨架：

```text
LLM 接口（统一 provider）          ← 决策源
记忆（短对话历史 + 长期向量存储）    ← 状态
工具系统（注册/描述/调用/结果回填）  ← 能力
执行循环（ReAct 式 while + 终止条件） ← 控制流
```

**教学价值**：读完 200 行内核后，看 LangChain 的 Chain/Agent/Tool 抽象，每一层都能回答"它在最小完备集的哪里、加了什么"。反向练习：给任一框架的问题（黑盒行为/调试困难）在骨架上定位根因。

**与本地工具章的衔接**：骨架里的工具系统升级为 MCP Client（标准协议）、执行循环升级为范式组合（《第04章补充》）——最小完备集是所有上层抽象的"汇编层"。

### 2.2 实战案例清单与复现顺序（hello-agents 案例）

案例按"能力阶梯"排列，建议作为附录 E 的实践附录：

| 案例类型 | 核心训练点 | 建议顺位 |
| :--- | :--- | :--- |
| 智能客服（FAQ+工单） | 工具调用 + 会话记忆 | ① 先做 |
| 数据分析智能体 | 代码执行 + 结果解读 | ② |
| 自主研究/调研智能体 | 长任务规划 + 报告生成 | ③ |
| 多智能体协作项目 | 编排 + 通信（衔接第06章） | ④ 最后 |

每个案例的固定复盘三问：失败案例集中在哪层（LLM/记忆/工具/循环）？哪个组件该升级？升级后回归测试怎么设计？

### 2.3 "从零到框架"的迁移路径

```text
附录E 正文（LangChain/CrewAI/AutoGen 用法）
  ← 先跑通
hello-agents ch7（裸写内核）
  ← 理解抽象
用自研内核重做一个附录E案例
  ← 验证理解
回到生产框架，带着内核心智做工程决策
  ← 落地
```

---

## 三、可直接动手的实践

1. 跟 ch7 写出 HelloAgents 内核（四组件 + ReAct 循环，目标 <300 行），并在内核上复刻附录 E 的客服案例；
2. 把内核的工具系统改造为 MCP Client（衔接《第03章补充》），对比改造前后的工具接入成本；
3. 按案例阶梯做完 ①② 两个项目，每项目写"失败案例分析 + 组件升级决策"复盘。

---

## 四、延伸链接

- hello-agents ch7：<https://github.com/datawhalechina/hello-agents/tree/main/docs/chapter7>
- 本地相关文件：附录E 正文（框架用法）、第04章-Agent范式-ReAct与规划-Datawhale补充.md（范式骨架）、第03章-工具调用与Function-Calling-Datawhale补充.md（协议层）、第06章-多智能体系统-Datawhale补充.md（协作层）
