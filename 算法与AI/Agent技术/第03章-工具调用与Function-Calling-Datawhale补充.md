---
title: 第03章-工具调用与Function-Calling · Datawhalechina 补充
tags:
  - Agent
  - MCP
  - 协议
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/hello-agents 第十章（MCP/A2A/ANP 协议解析）与 Extra05（Agent Skills 与 MCP 对比），补充工具调用之上的协议层体系
companion_to: 第03章-工具调用与Function-Calling.md
source: https://github.com/datawhalechina/hello-agents
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 工具调用与 Function Calling · Datawhalechina 对照补充

> 本文件是对同目录《第03章-工具调用与Function-Calling.md》的补充，内容整理自 [hello-agents](https://github.com/datawhalechina/hello-agents) **第十章 智能体通信协议（MCP/A2A/ANP）**与社区精选 **Extra05（Agent Skills 与 MCP 对比解读）**。原文档讲单模型内的工具调用机制；hello-agents ch10 的增量是**协议层**——当工具、智能体、网络跨越不同组织与运行时，Function Calling 之上需要标准协议栈。

---

## 一、补充来源地图

| hello-agents 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch10 MCP | Model Context Protocol：资源/工具/提示三原语 | 第03章正文 | **核心补充**：协议视角 |
| ch10 A2A | Agent-to-Agent 协议：跨智能体协作 | 第06章（关联） | 多智能体通信标准 |
| ch10 ANP | 开放网络智能体协议 | —（本地缺） | 去中心化方向 |
| Extra05 | Agent Skills vs MCP 的分工对比 | 第02章 Vibe 手册（关联） | 何时用 Skill、何时用 MCP |

---

## 二、核心补充内容

### 2.1 Function Calling → MCP 的层级关系（hello-agents ch10）

原文档的 Function Calling 是"模型 ↔ 工具"的**点对点**机制；MCP 把它标准化为**客户端-服务器**架构，解决 N×M 集成问题：

```text
Function Calling：每个应用为每个工具写一遍集成（N 应用 × M 工具）
MCP：工具方实现一次 MCP Server（工具/资源/提示词三原语）
     应用方实现一次 MCP Client（发现 + 调用）
     → N + M
```

MCP Server 的三原语与其在本地章节的对应：

| MCP 原语 | 作用 | 本地对应 |
| :--- | :--- | :--- |
| Tools | 可执行操作（带 JSON Schema） | 第03章 Function Calling 的工具定义 |
| Resources | 可读上下文（文件/DB 记录） | 第07章 记忆与上下文的"选"操作 |
| Prompts | 服务端预置的提示模板 | 第02章 Vibe 手册的 Skills 触发 |

### 2.2 A2A：智能体间协作的协议面（ch10）

MCP 解决"应用↔工具"，A2A 解决"智能体↔智能体"：

- **Agent Card**：智能体的自描述（能力/端点/认证）——对方如何发现你能干什么；
- **任务生命周期**：任务下发 → 状态流转（submitted/working/completed）→ 产物交付；
- 与 MCP 的**正交性**：一个智能体可以同时是 MCP Client（用工具）和 A2A Server（被其他智能体协作）——这张"双栈图"是 ch10 的核心心智模型。

### 2.3 何时用 Skill、何时用 MCP（Extra05 的判据）

Extra05 给出的分工判据，可直接并入原文档的工程决策节：

| 判据 | 选 Agent Skill | 选 MCP |
| :--- | :--- | :--- |
| 内容形态 | 方法论/流程/代码片段（文本知识） | 需要进程/网络/外部系统访问 |
| 生命周期 | 随仓库/会话携带 | 常驻服务、多应用复用 |
| 权限 | 无独立权限 | 需要独立认证与审计 |
| 典型例子 | "如何写好测试的流程" | GitHub/数据库/浏览器操作 |

经验法则：**知识以 Skill 沉淀，能力以 MCP 暴露**；两者组合时 Skill 负责教 agent "何时/怎么用"，MCP 负责"真的做到"。

---

## 三、可直接动手的实践

1. 把本地一个常用脚本封装成最小 MCP Server（仅 Tools 原语），在 OpenCode/Claude Code 中接入，对比 Function Calling 的注册差异；
2. 画出自己工作流的"双栈图"：哪些环节走 Skill（知识）、哪些走 MCP（能力）、哪些未来走 A2A（协作）；
3. 用 MCP Inspector 检查自己的 Server 的 schema 是否自描述完备（工具描述质量决定模型调用成功率）。

---

## 四、延伸链接

- hello-agents ch10：<https://github.com/datawhalechina/hello-agents/tree/main/docs/chapter10>
- Extra05：<https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra05-AgentSkills解读.md>
- 本地相关文件：第02章-Vibe-Coding全栈配置手册-Datawhale补充.md（MCP 配置实战）、第05章-Agent工作流与编排、第06章-多智能体系统（A2A 应用侧）
