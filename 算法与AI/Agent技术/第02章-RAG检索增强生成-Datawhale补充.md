---
title: 第02章-RAG检索增强生成 · Datawhalechina 补充
tags:
  - Agent
  - RAG
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/all-in-rag 全 10 章，补充混合检索、查询构建、Text2SQL、RAG 评估体系与 KG-RAG/GraphRAG 实战
companion_to: 第02章-RAG检索增强生成.md
source: https://github.com/datawhalechina/all-in-rag
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# RAG · Datawhalechina 对照补充

> 本文件是对同目录《第02章-RAG检索增强生成.md》的补充，内容整理自 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)（RAG 技术全栈指南，11k+ stars）全部 10 章。原文档从 Agent 视角讲 RAG 在系统中的位置；all-in-rag 的差异化价值在**检索与评估的深度**：混合检索、查询重构、Text2SQL、评估指标体系、GraphRAG 双实战项目——这五块是原文档相对薄弱的系统化部分。

---

## 一、补充来源地图

| all-in-rag 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch1 解锁 RAG + 四步构建 | 最小可用 RAG | 第02章正文 | 快速基线，用于对照性能 |
| ch2 数据准备 | 数据加载、文本分块策略 | 第02章正文 | 分块策略（fixed/recursive/semantic）系统化 |
| ch3 索引构建 | 向量嵌入、多模态嵌入、Milvus、索引优化 | 第02章正文 | **HNSW/IVF 参数调优**与多模态嵌入 |
| ch4 检索优化 | 混合检索、查询构建、Text2SQL、查询重构、高级检索 | 第02章正文 | **核心补充** |
| ch5 生成集成 | 格式化生成、结构化输出 | 第02章正文 | JSON Schema 约束解码 |
| ch6 RAG 系统评估 | 评估方法论、评估工具 | 第08章（Agent 评估） | **核心补充**：RAG 专项指标 |
| ch7 高级架构 | 基于知识图谱的 RAG | —（本地缺） | KG-RAG |
| ch8~ch9 实战项目 | "今天吃什么"全栈项目 + GraphRAG 优化 | —（本地缺） | 端到端工程 + 智能查询路由 |

---

## 二、核心补充内容

### 2.1 混合检索：稠密 + 稀疏（all-in-rag ch4.1）

原文档讲向量检索为主；all-in-rag 的关键补充是**单路检索都有系统性盲区**：

| 检索路 | 强项 | 盲区 |
| :--- | :--- | :--- |
| 稠密（向量） | 语义泛化、同义改写 | 精确词（产品型号、人名、错误码）易被"语义平滑"吞掉 |
| 稀疏（BM25/关键词） | 精确词、术语、ID 匹配 | 无语义泛化 |
| 融合 | 互补 | 需要分数归一化 |

**RRF（Reciprocal Rank Fusion）融合公式**——两路结果无需校准分数即可融合：

$$\text{score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \quad k \approx 60$$

再叠加**重排序模型（cross-encoder rerank）**：召回 top-50 → rerank 取 top-5。all-in-rag 的实测结论：混合检索 + rerank 在中文 QA 基线上通常比纯向量检索高 5~15 个百分点的 hit rate。

### 2.2 查询构建与重构（all-in-rag ch4.2/4.4）

用户原始 query 直接检索往往不是最优，标准加工链：

```text
原始 query
  → 意图分类（问答 / 检索 / 计算 / 闲聊）
  → 查询改写（指代消解："它的价格" → "iPhone 15 的价格"）
  → 查询扩展（HyDE：先让 LLM 生成假设答案，用假设答案的向量去检索）
  → 查询分解（复合问题拆成多个子查询，各自检索后合并）
  → 路由（不同意图走不同索引：向量库 / SQL / KG）
```

其中 **HyDE** 与**查询分解**是原文档未覆盖的两个高性价比技巧。

### 2.3 Text2SQL：结构化数据的检索（all-in-rag ch4.3）

企业数据大头在数据库而非文档，all-in-rag 把 Text2SQL 作为 RAG 的一等公民：

- **Schema 注入**：把表结构、样例行、枚举值写进 prompt，LLM 生成 SQL；
- **防幻觉三件套**：只允许 SELECT、SQL 语法校验、执行前 EXPLAIN；
- **失败回退**：SQL 生成失败 → 降级为文档检索并明确告知用户数据边界。

### 2.4 RAG 评估体系（all-in-rag ch6）——原文档最大缺口

RAG 的评估要**分环节**，不能只看最终答案对不对：

| 环节 | 指标 | 含义 |
| :--- | :--- | :--- |
| 检索 | Hit Rate / MRR / NDCG@k | 正确文档有没有被召回、排名如何 |
| 检索-生成衔接 | Context Relevance | 召回的上下文与问题的相关度 |
| 生成 | Faithfulness（忠实度） | 答案是否只基于上下文（衡量幻觉） |
| 生成 | Answer Relevance | 答案是否回应了问题 |
| 端到端 | Correctness | 与标注答案的一致性（LLM-as-judge） |

工具层面 all-in-rag 演示了 **RAGAS** 的用法：构造 `question / answer / contexts / ground_truth` 四元组数据集，一键产出上述指标。**实践建议**：任何 RAG 改动（换 embedding、调 chunk、加 rerank）都必须在这套指标上过回归，否则单例好坏不可信。

### 2.5 进阶架构：KG-RAG 与智能路由（all-in-rag ch7/ch9）

- **KG-RAG**：实体-关系三元组入库（Neo4j），多跳问题（"A 的供应商的 CEO"）用图查询补向量检索的短板；
- **GraphRAG 项目实战**（ch9）：图数据建模 → Milvus + 图双索引 → **智能查询路由**（分类器决定 query 走向量检索、图检索还是混合）——路由思想可直接迁移到 Agent 的工具选择。

---

## 三、可直接动手的实践

1. **给本地 RAG 加评估**：用 RAGAS 对现有问答链路跑一次基线（hit rate / faithfulness / answer relevance），把基线数字记入实验日志；
2. **混合检索对照**：同一批数据上对比"纯向量" vs "向量+BM25+RRF" vs "再加 rerank"三档的 MRR；
3. **查询重构实验**：对 10 个带指代的真实 query 测试改写前后的召回差异。

---

## 四、延伸链接

- all-in-rag 在线阅读：<https://datawhalechina.github.io/all-in-rag/>
- 实战项目（What-to-eat-today）：<https://github.com/FutureUnreal/What-to-eat-today>
- 本地相关文件：第07章-记忆与上下文工程（记忆与 RAG 的分层）、第08章-Agent评估与生产落地（把 RAG 指标并入 Agent 评估）、附录C-Prompt工程与RAG实战
