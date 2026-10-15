---
title: 0-LLM面试笔记-2026 · Datawhalechina 补充
tags:
  - 面试
  - LLM
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/daily-interview LLM 题库与 hello-agents Extra01 Agent 面试题，补充题源索引、题域分布与刷题方案
companion_to: 0-LLM面试笔记-2026.md
source: https://github.com/datawhalechina/daily-interview ; https://github.com/datawhalechina/hello-agents
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# LLM 面试笔记 · Datawhalechina 对照补充

> 本文件是对同目录《0-LLM面试笔记-2026.md》的补充，内容整理自 [daily-interview](https://github.com/datawhalechina/daily-interview)（Datawhale 成员面经库）的 `docs/04-ai-algorithms/llm/llm-interview-questions.md`（大模型面试题小结，**带参考答案的 Q&A 题库**）与 [hello-agents](https://github.com/datawhalechina/hello-agents) **Extra01（Agent 面试题总结 + 参考答案）**。两份题库与本笔记互补：笔记偏体系梳理，题库偏逐题演练。

---

## 一、补充来源地图

| 来源 | 形态 | 题域 | 补充价值 |
| :--- | :--- | :--- | :--- |
| daily-interview `04-ai-algorithms/llm` | **Q&A 带答案**（Q1、Q2…逐题作答） | 架构对比/注意力/上下文/分词/角色标记/词嵌入/知识遗忘等 | **核心补充**：可直接对答案 |
| daily-interview `04-ai-algorithms` 其他目录 | 面经 + 题目 | ML/CV/NLP/推荐/开发 | 通用算法岗题源 |
| daily-interview `03-interview-experience` | 经验帖 | 各公司流程与真题 | 面试节奏参考 |
| hello-agents Extra01 | 题目 + 独立参考答案文件 | Agent 专题（框架/记忆/多智能体/评估） | **核心补充**：Agent 岗专用 |

---

## 二、核心补充内容

### 2.1 daily-interview LLM 题库的题域分布与使用方式

daily-interview 的 LLM 题库按**概念块**组织，每个概念块对应本地笔记的一章，可以按块对练：

| 题库概念块（示例题） | 对应本地笔记章节 | 对练方式 |
| :--- | :--- | :--- |
| 架构对比（BERT/GPT/Encoder-Decoder 优缺点） | 第02章-架构演进 | 先自答 → 对答案 → 补笔记 |
| 自注意力与长依赖（vs RNN） | 手撕/第05章-01 | 白板推导 QK 注意力 |
| 上下文长度（为何是输入+输出总和） | 第03章-位置编码与长上下文 | 追问链：KV cache / 截断策略 |
| 分词器（vs 传统中文分词、唯一性） | 附录A-NLP基础 | 现场切一个例句 |
| 角色标记（用户/AI 如何区分） | 第09章-SFT（对话模板） | 写出 chat template |
| 静态词嵌入 vs 上下文嵌入 | 附录A-NLP基础 | 追问 word2vec 线性关系 |
| 知识遗忘改哪里（注意力层 vs FFN） | 第02章-架构演进 | 追问模型编辑/知识神经元 |

**刷题闭环**：自答 → 对题库答案 → 把"答案里有而笔记没有的点"回填笔记（这是本补充最重要的动作——题库答案来自一线面经，常有本地笔记未覆盖的口径）。

### 2.2 hello-agents Extra01：Agent 专题题域

Extra01 的 Agent 题域与本笔记的对应关系（答案在独立文件，适合先答后对）：

| Extra01 题域 | 本地笔记/章节 |
| :--- | :--- |
| Agent 范式选型（ReAct/Plan-and-Solve/Reflection） | Agent技术/第04章 |
| 记忆与上下文设计 | Agent技术/第07章 + 其 Datawhale 补充 |
| 多智能体分工与通信（MCP/A2A） | Agent技术/第03/06章 + 第03章补充 |
| 评估方法（三层指标/LLM-as-judge） | Agent技术/第08章 + 其 Datawhale 补充 |
| Agentic-RL（SFT→GRPO） | LLM发展/第05章 + 其 Datawhale 补充 |

### 2.3 与 GenRec 面试笔记的联动题源

《1-GenRec面试笔记-2026》的题源在 datawhale 侧对应 fun-rec 下篇 ch5~ch8（Tokenizer/HSTU/OneRec/PLUM/OneRec-Think）——GenRec 题的标准答案口径可按 fun-rec 章节校对，与本地 GR 第08章补充文件、SID Tokenizer 补充文件的内容一一对应。

---

## 三、可直接动手的实践

1. 建立"三栏对练表"（题目 / 我的答案 / 题库答案差异点），先跑完 daily-interview LLM 题库的前 20 题；
2. 用 Extra01 做 Agent 专题模拟面（先看题不看答案），错题回跳章节；
3. 每周把对练中发现的笔记缺口回填《0-LLM面试笔记-2026》，保持笔记为"终稿答案"层。

---

## 四、延伸链接

- daily-interview：<https://github.com/datawhalechina/daily-interview>
- hello-agents Extra01：<https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra01-面试问题总结.md>
- 本地相关文件：1-GenRec面试笔记-2026（题源联动 fun-rec）、2-算法工程师通用面试-2026（daily-interview 其他目录）、3-公司面试真题与经验-2026（经验帖）
