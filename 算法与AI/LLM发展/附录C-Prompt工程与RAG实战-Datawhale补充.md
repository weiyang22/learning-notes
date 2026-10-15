---
title: 附录C-Prompt工程与RAG实战 · Datawhalechina 补充
tags:
  - LLM
  - Prompt
  - RAG
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/llm-cookbook（吴恩达课程中译）与 all-in-rag，补充提示工程课程的逐课索引与 RAG 实战衔接
companion_to: 附录C-Prompt工程与RAG实战.md
source: https://github.com/datawhalechina/llm-cookbook ; https://github.com/datawhalechina/all-in-rag
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Prompt 工程与 RAG 实战 · Datawhalechina 对照补充

> 本文件是对同目录《附录C-Prompt工程与RAG实战.md》的补充，内容整理自 [llm-cookbook](https://github.com/datawhalechina/llm-cookbook)（吴恩达大模型系列课程中文版，25k stars）与 [all-in-rag](https://github.com/datawhalechina/all-in-rag)（本库 Agent 第02章已有 RAG 深度补充，本文件聚焦**提示工程课程线**）。llm-cookbook 的差异化价值：吴恩达课程的**逐课实操 Notebook 中译**——每课一个小任务，是"会调 API"到"会设计提示"的练习场。

---

## 一、补充来源地图

| llm-cookbook 课程 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ChatGPT Prompt Engineering | 迭代式提示开发、分隔符、结构化输出 | Prompt 高级技巧节 | **核心补充**：迭代方法论而非一次成型 |
| Building Systems with ChatGPT | 分类/审核/链条式处理/评估 | Prompt 技巧节 | 把提示组织成系统的工作流 |
| LangChain 系列课程 | Chain/Agent/记忆/RAG 组件化 | LangChain 节 | 框架抽象与裸写的对照 |
| Finetuning LLMs | 指令微调数据构建 | 与第09章衔接 | SFT 数据侧视角 |
| all-in-rag ch1~ch6 | RAG 数据/索引/检索/评估全栈 | RAG 全栈节 | 详见《Agent技术/第02章》补充文件 |

---

## 二、核心补充内容

### 2.1 吴恩达式提示开发方法论（Prompt Engineering 课）

与本地附录 C"技巧清单"互补的是**开发流程**：

1. **一次一个指令，先写最短版**：给任务 → 看输出哪里不对 → 只针对缺陷改提示 → 重复。技巧清单是"武器库"，迭代流程是"打法"；
2. **明确输出结构契约**：用分隔符（``` 或 ###）隔离输入、要求 JSON/表格等结构化输出、指定字段名——下游系统的稳定性来自提示的契约性；
3. **给模型"思考时间"**：要求先列步骤再给结论（与 CoT 一致）；对复杂任务用"完成 X 之前先做 Y"的顺序指令；
4. **系统性评估变体**：改提示后在**固定测试集**上对比，不靠单例感觉——这套纪律与本库《工程与职业/3-AB测试与因果推断》一致。

### 2.2 从提示到系统（Building Systems 课的链条设计）

Building Systems 课程演示了把单条提示扩展成"提示链"：输入分类器 →（分类路由）→ 各类处理提示 → 审核链（安全/合规检查）→ 输出组装。两个可迁移模式：

- **审核链前置**：用户输入先过 moderation，再进业务提示——成本极低但挡掉大部分脏输入；
- **分类路由**：让小模型/简单提示做意图分类，再路由到对应的重提示——与《Agent技术/第02章补充》的查询路由、RAG 的智能路由是同一模式在不同层的应用。

### 2.3 LangChain 的抽象取舍（LangChain 系列课）

课程展示了 LangChain 的 PromptTemplate/Chain/Agent 抽象。取舍判断（可并入附录 C 的工程建议）：

- 原型期：LangChain 快（组件即插即用）；
- 生产期：链路要可调试、可观测——裸写提示 + 自建编排更可控，或只用 LangChain 的个别组件；
- 这个"框架 vs 裸写"的张力与本库《Agent技术/第05章》的编排对比一致，附录 C 可互引。

---

## 三、可直接动手的实践

1. 按 2.1 的迭代流程对一个真实任务做 5 轮提示迭代，每轮记录"改了什么 + 测试集指标变化"；
2. 用分类路由模式改造一个入口提示：先分类（FAQ/检索/计算/闲聊）再走对应分支，测分类准确率与端到端满意度；
3. 同一 RAG 问答题分别用"裸提示"与 LangChain 组件实现，对比 tokens 消耗与调试便利性。

---

## 四、延伸链接

- llm-cookbook：<https://github.com/datawhalechina/llm-cookbook>
- all-in-rag：<https://datawhalechina.github.io/all-in-rag/>
- 本地相关文件：算法与AI/Agent技术/第02章-RAG检索增强生成-Datawhale补充.md（RAG 深度）、第09章-SFT等后训练教程（Finetuning 课衔接）、第04章-Agent范式（ReAct 原理侧）
