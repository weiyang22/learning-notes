---
title: 附录C Prompt 工程与 RAG 实战
tags:
  - Prompt Engineering
  - RAG
  - Few-shot
  - CoT
  - LangChain
  - LlamaIndex
  - 向量检索
  - ReRank
description: Prompt 高级技巧 (Few-shot/CoT/ReAct) + RAG 全栈. 整合 llm-cookbook + all-in-rag 内容
last_updated: 2026-09-11
status: ✅ v1.0 - 整合 llm-cookbook + all-in-rag
chapter_id: appendix-c
---

# 附录C Prompt 工程与 RAG 实战

> 本附录整合自 [datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook) (Prompt Engineering 教程) 与 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) (RAG 全栈指南).

---

## 1. Prompt 工程基础

### 1.1 什么是 Prompt Engineering

**Prompt** 是给 LLM 的输入文本, 决定模型输出. **Prompt Engineering** 是设计高效 prompt 的系统方法.

**两大原则**:
1. **写清晰的指令**: 具体、明确、给出示例
2. **给模型思考时间**: CoT、ReAct 等技巧让模型逐步推理

### 1.2 基础原则

来自 OpenAI Cookbook 与吴恩达课程:

| 原则 | 说明 |
| :--- | :--- |
| **使用分隔符** | 用 `"""` 或 `###` 分隔指令和输入 |
| **结构化输出** | 要求 JSON / 表格输出 |
| **检查条件** | "如果不知道, 回答'未知'" |
| **Few-shot** | 给几个示例 |
| **指定任务** | "翻译" / "总结" / "分类" 明确 |

---

## 2. Prompt 高级技巧

### 2.1 Zero-Shot Prompting

无示例直接问.

```python
prompt = "将以下文本分类为正面或负面情感: '这个产品太糟糕了, 完全不能用!'"
# 输出: 负面
```

### 2.2 Few-Shot Prompting ⭐

给几个示例, 模型学习模式.

```python
prompt = """
将英文翻译成法语:

sea → mer
sky → ciel
star → étoile
book →
"""
# 输出: livre
```

**最佳实践**:
- 示例数量: 3-5 个
- 示例要有代表性
- 顺序固定 (LLM 对顺序敏感)

### 2.3 Chain-of-Thought (CoT) ⭐⭐ 核心技巧

让模型"一步步思考", 大幅提升推理能力.

**Zero-shot CoT**: 加一句 "Let's think step by step"

```python
prompt = """
我买了一个20元的商品, 打折15%, 加5元税, 最终多少钱?
Let's think step by step.
"""
```

**Few-shot CoT**: 给带推理过程的示例

```python
prompt = """
Q: 一个停车场有 3 辆汽车, 又来了 2 辆, 一共几辆?
A: 原来有 3 辆, 又来了 2 辆, 所以 3 + 2 = 5 辆. 答案是 5.

Q: 我有 23 个苹果, 给了 5 个给朋友, 又买了 8 个, 一共几个?
A: 原来有 23 个, 给出 5 个剩 18 个, 又买了 8 个, 所以 18 + 8 = 26 个. 答案是 26.

Q: 服务器有 50 个请求, 完成 30 个后又收到 15 个, 还剩多少待处理?
A:
"""
```

**Zero-shot CoT 触发提示**:
- "Let's think step by step"
- "Let's work this out step by step"
- "First, let's think about this carefully"
- "Before answering, explain your reasoning"

### 2.4 Self-Consistency (自一致性)

多次采样 + 投票, 提升推理可靠性.

```python
from collections import Counter

prompt = "..."  # CoT prompt
answers = []
for _ in range(7):  # 多次采样
    response = llm.generate(prompt, temperature=0.7)
    # 提取最终答案
    final = extract_answer(response)
    answers.append(final)

# 多数投票
final_answer = Counter(answers).most_common(1)[0][0]
```

论文: Wang et al., 2022, *Self-Consistency Improves Chain of Thought Reasoning*.

### 2.5 ReAct (Reasoning + Acting) ⭐ Agent 基础

LLM 交替进行推理和行动.

```python
prompt = """
回答问题: 哥伦比亚 2023 年人均 GDP 是多少?

Thought 1: 我需要查哥伦比亚的 GDP 数据.
Action 1: Search["Colombia GDP 2023"]
Observation 1: 哥伦比亚 2023 年 GDP 总额 3635 亿美元, 人口约 5200 万.
Thought 2: 用 GDP 除以人口.
Action 2: Calculate["363.5 / 52"]
Observation 2: 约 7000 美元.
Thought 3: 我得到了答案.
Action 3: Finish["约 7000 美元"]
"""
```

详见 [算法与AI/../Agent技术/第04章-Agent范式-ReAct与规划](../Agent技术/第04章-Agent范式-ReAct与规划.md).

### 2.6 Tree of Thoughts (ToT)

多路径探索 + 评估回溯. 适合复杂问题.

```python
def tree_of_thoughts(question, n_paths=3, depth=3):
    # 1. 生成多条思路
    thoughts = generate_n(question, n_paths)

    # 2. 评估每条
    scores = [evaluate(q + t) for t in thoughts]

    # 3. 选择最佳, 继续展开
    best = thoughts[argmax(scores)]
    next_thoughts = expand(q + best, n_paths)

    # 4. 重复到 depth 层
    # ...

    return best
```

### 2.7 ReAct + CoT 高级组合

```python
prompt = """
# 工具
你有两个工具:
1. Search(query): 搜索网页
2. Calculator(expr): 计算数学表达式

# 格式
Question: [用户问题]
Thought 1: [你的思考]
Action 1: [工具调用]
Observation 1: [工具结果]
Thought 2: [下一步思考]
Action 2: [下一步工具]
Observation 2: [结果]
...
Thought N: [最终思考]
Final Answer: [答案]

# 开始
Question: 法国的首都是什么? 那里的人口是多少?
Thought 1:
"""
```

---

## 3. 提示词模板库

### 3.1 总结

```python
template = """
请总结以下文本, 要求:
1. 不超过 100 字
2. 保留关键数字
3. 用列表呈现要点

文本:
{text}

总结:
"""
```

### 3.2 翻译

```python
template = """
将以下 {source_lang} 翻译成 {target_lang}, 要求:
1. 保留专业术语
2. 翻译地道自然
3. 输出仅译文, 无其他内容

原文:
{text}

译文:
"""
```

### 3.3 信息抽取

```python
template = """
从以下文本中抽取所有:
- 人名
- 地名
- 组织
- 时间
- 金额

输出 JSON 格式: {"人名": [], "地名": [], ...}

文本:
{text}
"""
```

### 3.4 分类

```python
template = """
将以下文本分类到最合适的类别:
类别: 体育, 政治, 科技, 娱乐, 其他

如果不确定, 输出 "其他"。

文本: {text}

类别:
"""
```

### 3.5 代码审查

```python
template = """
审查以下代码, 找出:
1. 潜在 Bug
2. 性能问题
3. 安全漏洞
4. 风格问题

输出 Markdown 格式报告。

代码:
```python
{code}
```
"""
```

### 3.6 角色扮演

```python
template = """
你是 {role}, 具有 {years} 年经验。
你的专长是 {expertise}。
回答风格: {style}。

问题: {question}
"""
```

---

## 4. RAG 基础

### 4.1 什么是 RAG

**Retrieval-Augmented Generation (RAG)** = 检索 + 生成. 通过从外部知识库检索相关信息, 增强 LLM 的回答.

```text
用户问题 → 检索相关文档 → 拼接到 Prompt → LLM 生成回答
       ↓              ↓                ↓              ↓
    "X 是什么?"   [Doc1, Doc2, Doc3]  "基于以下资料..."  "X 是..."
```

### 4.2 为什么需要 RAG

| 问题 | LLM 自身 | LLM + RAG |
| :--- | :--- | :--- |
| 知识陈旧 | 训练截止后的新事件不知 | 实时检索 |
| 幻觉 | 编造事实 | 基于真实文档 |
| 私有数据 | 无法访问 | 检索内部文档 |
| 可解释性 | 黑盒 | 给出参考来源 |

### 4.3 RAG 工作流

```python
# 1. 离线索引
documents = ["doc1.txt", "doc2.txt", ...]
chunks = split_documents(documents)  # 切块
embeddings = embed(chunks)  # 向量化
vector_db.store(chunks, embeddings)  # 入库

# 2. 在线查询
query = "用户问题"
query_emb = embed(query)
relevant_chunks = vector_db.search(query_emb, top_k=5)  # 检索

prompt = f"""
基于以下资料回答:
{relevant_chunks}

问题: {query}
"""
answer = llm.generate(prompt)
```

---

## 5. RAG 全栈实战

### 5.1 文档处理与切块

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

# 2. 切块 (关键参数: chunk_size, overlap)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块 500 字符
    chunk_overlap=50,       # 重叠 50 字符 (保持上下文)
    separators=["\n\n", "\n", "。", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"切块数: {len(chunks)}")
```

### 5.2 Embedding 模型

```python
from sentence_transformers import SentenceTransformer

# 中文 Embedding
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 英文 Embedding
# model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# 编码
embeddings = model.encode(["文档1的内容", "文档2的内容"])
print(embeddings.shape)  # (2, 512)
```

主流 Embedding 模型对比:

| 模型 | 维度 | 语言 | 特点 |
| :--- | :--- | :--- | :--- |
| **bge-small-zh-v1.5** | 512 | 中 | 轻量中文 |
| **bge-large-zh-v1.5** | 1024 | 中 | 高质量中文 |
| **bge-m3** | 1024 | 多 | 多语言长文本 |
| **text-embedding-3-large** | 3072 | 多 | OpenAI |
| **Cohere embed-v3** | 1024 | 多 | 商用高质量 |

### 5.3 向量数据库

```python
import chromadb

# Chroma (本地嵌入式, 适合开发)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")

# 添加文档
collection.add(
    documents=["文档内容1", "文档内容2"],
    metadatas=[{"source": "wiki"}, {"source": "book"}],
    ids=["id1", "id2"]
)

# 查询
results = collection.query(
    query_texts=["用户问题"],
    n_results=3
)
for doc, dist in zip(results['documents'][0], results['distances'][0]):
    print(f"距离: {dist:.3f} | {doc[:50]}")
```

其他主流向量库:

| 名称 | 类型 | 特点 |
| :--- | :--- | :--- |
| **FAISS** | 嵌入式 | Meta 出品, 高性能 |
| **Chroma** | 嵌入式 | 简单易用 |
| **Milvus** | 服务化 | 亿级向量, 生产级 |
| **Qdrant** | 服务化 | Rust 写, 性能优 |
| **Weaviate** | 服务化 | 模块化 |
| **pgvector** | PG 扩展 | 用 Postgres |

### 5.4 完整 RAG Pipeline

```python
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'}
)

# 2. 加载向量库
vectorstore = FAISS.load_local("./faiss_index", embeddings)

# 3. LLM (本地 Ollama)
llm = Ollama(model="qwen2.5:7b")

# 4. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 把所有文档塞进 prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 5. 问答
result = qa_chain({"query": "Datawhale 是什么组织?"})
print(f"答案: {result['result']}")
print(f"参考来源:")
for doc in result['source_documents']:
    print(f"  - {doc.metadata.get('source', '未知')}")
```

---

## 6. RAG 进阶

### 6.1 ReRank (重排序)

两阶段检索: 先向量检索 top-K, 再用 cross-encoder 重排序.

```python
from sentence_transformers import CrossEncoder

# 加载重排模型
reranker = CrossEncoder('BAAI/bge-reranker-large')

# 1. 粗排: 向量检索 top 50
candidates = vectorstore.similarity_search(query, k=50)

# 2. 精排: Cross-encoder 重排
pairs = [(query, doc.page_content) for doc in candidates]
scores = reranker.predict(pairs)

# 3. 按分数排序, 取 top 5
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
top_docs = [doc for doc, score in ranked]
```

### 6.2 HyDE (Hypothetical Document Embeddings)

让 LLM 先"假想"一个理想答案, 然后用这个假想答案去检索.

```python
# 1. LLM 生成假想答案
hyde_prompt = f"请简洁回答: {query}"
hypothetical = llm.generate(hyde_prompt)

# 2. 用假想答案的 embedding 去检索
query_emb = embed_model.encode([hypothetical])
results = vectorstore.similarity_search_by_vector(query_emb[0], k=5)
```

### 6.3 Multi-Query

LLM 生成多个查询变体, 提高召回.

```python
multi_query_prompt = """
基于原问题, 生成 5 个不同角度的改写问题:

原问题: {query}

5 个改写:
1.
"""

queries = llm.generate(multi_query_prompt).split("\n")
all_docs = []
for q in queries:
    docs = vectorstore.similarity_search(q, k=2)
    all_docs.extend(docs)

# 去重 + 取 top K
unique_docs = list({d.page_content: d for d in all_docs}.values())[:5]
```

### 6.4 GraphRAG (微软)

用知识图谱增强 RAG.

```python
from graphrag import GraphRAG

# 1. 构建图谱
graph = GraphRAG("./documents")
graph.build_graph()

# 2. 查询 (使用社区检测和摘要)
result = graph.query("这个问题涉及哪些实体? 它们的关联是什么?")
```

### 6.5 评估 RAG 系统

```python
from ragas import evaluate

# 评估指标
result = evaluate(
    dataset,
    metrics=[
        "faithfulness",     # 答案是否基于检索内容
        "answer_relevancy",  # 答案是否相关
        "context_precision", # 检索的精确度
        "context_recall",    # 检索的召回率
    ]
)
print(result)
```

---

## 7. LangChain 高级用法

### 7.1 Memory (对话记忆)

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

chain.predict(input="你好")
# '你好! 有什么可以帮助你的?'

chain.predict(input="1+1=?")
# '1+1=2'

chain.predict(input="刚才那个问题呢?")
# '你刚才问的是 1+1=? 答案是 2.'
```

### 7.2 Agent

```python
from langchain.agents import load_tools, initialize_agent

# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化 Agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

agent.run("特斯拉现在的股价是多少? 涨了百分之几?")
```

### 7.3 Custom Tool

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    # 实际调用 API
    return f"{city} 天气晴, 25°C"

# 注册到 Agent
agent = initialize_agent([get_weather, ...], llm, ...)
```

---

## 8. LlamaIndex 简介

LlamaIndex 是另一个主流 RAG 框架, 特点:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引 (一行)
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("Datawhale 是什么?")
print(response)
```

**LangChain vs LlamaIndex**:
- **LangChain**: 更通用, Agent 工具强, 灵活
- **LlamaIndex**: 更专注 RAG, 数据连接器多, 索引能力强

---

## 9. RAG 评估 (深入)

### 9.1 评估数据集

需要:
- **问题**: 测试查询
- **标准答案**: Ground truth
- **上下文**: 检索到的文档

### 9.2 评估指标

| 指标 | 含义 | 计算 |
| :--- | :--- | :--- |
| **Context Precision** | 检索的精确度 | 检索到的相关文档 / 总检索数 |
| **Context Recall** | 检索的召回率 | 检索到的相关文档 / 全部相关文档 |
| **Faithfulness** | 答案是否基于检索 | LLM-as-judge |
| **Answer Relevancy** | 答案相关度 | LLM-as-judge |

### 9.3 RAGAS 自动评估

```python
from datasets import Dataset
from ragas import evaluate

# 准备数据
data = {
    "question": ["Datawhale 是什么?", "什么是 RAG?"],
    "answer": ["Datawhale 是 AI 开源组织...", "RAG 是..."],
    "contexts": [["Datawhale 是一个..."], ["RAG 检索增强..."]],
    "ground_truth": ["Datawhale 是..."],
}

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=["faithfulness", "context_recall", "answer_relevancy"])
print(result)
```

---

## 10. 高级 RAG 模式对比

| 模式 | 优点 | 缺点 | 适用 |
| :--- | :--- | :--- | :--- |
| **Naive RAG** | 简单 | 召回不准 | 原型 |
| **ReRank** | 精确 | 慢 | 高精度需求 |
| **HyDE** | 处理抽象问题 | 依赖 LLM | 长尾查询 |
| **Multi-Query** | 召回高 | 多次 LLM 调用 | 复杂问题 |
| **GraphRAG** | 全局关系 | 构建慢 | 实体关系查询 |
| **Self-RAG** | 自适应检索 | 慢 | 高质量 |
| **Corrective RAG** | 自我纠错 | 复杂 | 容错场景 |

---

## 11. 常见问题与最佳实践

### Q1: 检索不到相关内容

**解决**:
- 调小 `chunk_size`, 增加切块粒度
- 增加 `top_k`, 检索更多候选
- 用 ReRank 精排
- 优化 Embedding 模型 (用更大的或领域微调)
- 检查文档质量 (OCR、解析是否正确)

### Q2: 回答质量差

**解决**:
- 优化 Prompt (明确指令, 加引用要求)
- 改用更强的 LLM
- 加入 CoT 或 Self-Consistency
- 用 ReRank 提高上下文质量

### Q3: 速度慢

**解决**:
- 用更快的 Embedding 模型
- 减小 `top_k`
- 用缓存 (避免重复检索)
- 用 vLLM / TGI 服务化

### Q4: 成本高

**解决**:
- 用更小的 Embedding
- 用更便宜的 LLM (Qwen2.5-7B 替代 GPT-4)
- 缓存常见查询
- 监控 token 使用

---

## 12. RAG vs Fine-tuning vs Prompt Engineering

| 方法 | 适用场景 | 成本 | 时效性 |
| :--- | :--- | :--- | :--- |
| **Prompt Engineering** | 简单任务, LLM 已具备能力 | 极低 | 即时 |
| **RAG** | 需要最新/私有知识 | 中 (向量库) | 实时 |
| **Fine-tuning** | 需要特定风格/输出格式 | 高 (GPU + 数据) | 离线 |

**最佳实践**: **RAG + Fine-tuning + Prompt Engineering 组合使用**.

---

## 13. 参考资源

### Datawhale 教程
- [datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook) — Prompt Engineering 经典教程
- [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) — RAG 全栈指南
- [datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe) — 应用开发
- [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) — 从零开始构建智能体

### 经典论文
1. **RAG**: Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2. **Self-RAG**: Asai et al., 2023. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511.
3. **HyDE**: Gao et al., 2022. *Precise Zero-Shot Dense Retrieval without Relevance Labels*. arXiv:2212.10496.
4. **GraphRAG**: Edge et al., 2024. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130.

### 工具与框架
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [RAGAS](https://docs.ragas.io/) — RAG 评估
- [ChromaDB](https://www.trychroma.com/)
- [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding)

---

> 本附录覆盖了 Prompt Engineering 核心技巧和 RAG 全栈实战. 配合 [算法与AI/../Agent技术/第02章-RAG检索增强生成](../Agent技术/第02章-RAG检索增强生成.md) 阅读效果更佳.
