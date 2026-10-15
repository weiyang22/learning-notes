---
title: 附录A NLP 基础与文本表示
tags:
  - NLP
  - 文本表示
  - Word2Vec
  - BERT
  - Tokenization
  - 词性标注
  - 命名实体识别
description: NLP 基础概念与文本表示演进. 来自 happy-llm/llm-cookbook 整合. 中英文分词、子词切分、词性标注、文本分类、NER、关系抽取、摘要、翻译、问答 + 文本表示从 one-hot 到 BERT
last_updated: 2026-09-11
status: ✅ v1.0 - 整合 happy-llm 与 llm-cookbook
chapter_id: appendix-a
---

# 附录A NLP 基础与文本表示

> 本附录整合自 [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) 第 1 章 NLP 基础概念 与 [llm-cookbook](https://github.com/datawhalechina/llm-cookbook) 的核心要点. 作为 LLM 发展的前置知识.

---

## 1. 什么是 NLP

**自然语言处理 (Natural Language Processing, NLP)** 是人工智能的一个分支, 旨在让计算机理解、解释和生成人类语言. 它结合了计算机科学、语言学、心理学等多个学科.

NLP 的核心任务包括:

| 任务 | 输入 | 输出 | 典型应用 |
| :--- | :--- | :--- | :--- |
| **中文分词** | 中文句子 | 词序列 | 中文处理基础 |
| **子词切分** | 任意文本 | subword 序列 | 解决 OOV 问题 |
| **词性标注** | 句子 | (词, 词性) 序列 | 句法分析前处理 |
| **文本分类** | 文本 | 类别标签 | 情感分析、垃圾邮件 |
| **命名实体识别 NER** | 句子 | (实体, 类型) | 信息抽取 |
| **关系抽取** | 句子 | (头实体, 关系, 尾实体) | 知识图谱 |
| **文本摘要** | 长文本 | 短摘要 | 新闻摘要 |
| **机器翻译** | 源语言文本 | 目标语言文本 | 跨语言交流 |
| **自动问答** | 问题 + 上下文 | 答案 | 客服、知识库 |

---

## 2. NLP 发展历程 (简史)

```text
1940s-1960s  图灵测试 (1950) · 乔姆斯基生成语法 · 规则基础机器翻译
1970s-1990s  符号主义 vs 统计方法 · HMM · 决策树 · 统计机器翻译
2000s        SVM · 浅层神经网络 · CRF (条件随机场)
2013         Word2Vec · 词向量时代开启 (Tomas Mikolov)
2014-2015    Seq2Seq + Attention · 神经机器翻译
2017         Transformer 诞生 (Vaswani et al.) · 奠基现代 LLM
2018         BERT (Masked LM) · GPT-1 (自回归) · 预训练-微调范式
2019         GPT-2 · 零样本学习 · 规模即能力
2020         GPT-3 (175B) · Scaling Law · In-Context Learning
2022         InstructGPT · RLHF · ChatGPT 破圈
2023         GPT-4 · LLaMA 开源 · 多模态
2024         DeepSeek-V3 (MoE+MLA) · DPO · o1 推理模型
2025-2026    R1 / GRPO · Agent · Test-Time Compute Scaling
```

---

## 3. NLP 核心任务详解

### 3.1 中文分词 (Chinese Word Segmentation, CWS)

中文没有空格分隔, 必须先切词. 例:

```text
英文输入: The cat sits on the mat.
英文切割:  [The | cat | sits | on | the | mat]

中文输入: 今天天气真好，适合出去游玩.
中文切割:  ["今天", "天气", "真", "好", "，", "适合", "出去", "游玩", "。"]
```

**错误示例**:

```text
输入: 雍和宫的荷花开的很好。

正确切割: 雍和宫 | 的 | 荷花 | 开 | 的 | 很 | 好 | 。
错误切割: 雍 | 和 | 宫的 | 荷花 | 开的 | 很好 | 。  ← 地名被拆散
```

主要算法:
- **基于词典**: 最大匹配法 (MM, RMM)
- **基于统计**: HMM, CRF, 感知机
- **基于深度学习**: BiLSTM-CRF, BERT-CRF
- **基于大模型**: 直接用 LLM 做分词 (prompt-based)

### 3.2 子词切分 (Subword Segmentation)

解决 OOV (Out-of-Vocabulary) 问题. 把罕见词分解为已知子词.

| 算法 | 训练方式 | 特点 |
| :--- | :--- | :--- |
| **BPE** (Byte Pair Encoding) | 迭代合并最高频字符对 | GPT-2, RoBERTa |
| **WordPiece** | 似然最大化的合并 | BERT |
| **Unigram LM** | 反向: 删减字符 | XLNet, ALBERT |
| **SentencePiece** | 上述任一 + 跨语言 | T5, mBART, LLaMA |

例:

```text
输入: unhappiness
不使用子词切分: ["unhappiness"]      (OOV)
使用 BPE:        ["un", "happi", "ness"] (已知子词组合)
```

**LLM 的 token 化**: 主流 LLM 使用 BPE/SentencePiece, vocab 大小 32K-100K.

### 3.3 词性标注 (POS Tagging)

为每个词分配语法角色. 例:

```text
She    is     playing   the   guitar   in    the   park  .
PRP    VBZ    VBG       DT    NN      IN    DT    NN    .
代词   动词   现在分词  冠词  名词    介词  冠词  名词  标点
```

模型演进: HMM → CRF → BiLSTM-CRF → BERT-CRF → LLM (zero-shot)

### 3.4 文本分类 (Text Classification)

应用最广的任务之一. 模型演进:

| 时代 | 模型 | 特点 |
| :--- | :--- | :--- |
| 传统 | Naive Bayes, SVM | 词袋 + TF-IDF |
| 深度学习 | TextCNN, BiLSTM+Attention | 端到端 |
| 预训练 | BERT fine-tune | 迁移学习 |
| Prompt | SetFit, LLM zero/few-shot | 小样本 |

### 3.5 命名实体识别 (NER)

```text
输入: 李雷和韩梅梅是北京市海淀区的居民，他们计划在2024年4月7日去上海旅行。

输出: [
  ("李雷", "人名"),
  ("韩梅梅", "人名"),
  ("北京市海淀区", "地名"),
  ("2024年4月7日", "日期"),
  ("上海", "地名")
]
```

应用: 信息抽取、知识图谱构建、问答系统.

### 3.6 关系抽取 (Relation Extraction)

```text
输入: 比尔·盖茨是微软公司的创始人。
输出: [("比尔·盖茨", "创始人", "微软公司")]
```

### 3.7 文本摘要

- **抽取式 (Extractive)**: 直接选原文中重要句子
- **生成式 (Abstractive)**: 重新组织语言生成新摘要 (LLM 主流方式)

### 3.8 机器翻译

从规则 → 统计 (SMT) → 神经 (NMT) → 大模型 (GPT-4, Claude).

### 3.9 自动问答

三种类型:
- **检索式**: 从大量文本中检索答案
- **知识库**: 基于结构化知识 (KGQA)
- **生成式**: LLM 直接生成 (RAG 增强)

---

## 4. 文本表示的发展

文本表示是 NLP 的核心: 把文字变成计算机能处理的向量.

### 4.1 One-Hot 编码

```text
"雍和宫的荷花很美"
词汇表大小: 16384

vector = [0, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ..., 1, 0, ...]
#                    ↑          ↑          ↑          ↑          ↑
#      16384 维中只有 5 个位置为 1, 其余 16379 个位置为 0
# 实际有效维度: 仅 5 维 (非零维度)
# 稀疏率: (16384-5)/16384 ≈ 99.97%
```

**问题**: 维度灾难 + 语义鸿沟 (猫/狗是正交的,无法表达相似性)

### 4.2 N-gram 语言模型

基于马尔可夫假设: 一个词的出现概率只依赖前 N-1 个词.

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_{i-N+1}, \ldots, w_{i-1})$$

- **Unigram (N=1)**: $P(w_i)$
- **Bigram (N=2)**: $P(w_i | w_{i-1})$
- **Trigram (N=3)**: $P(w_i | w_{i-2}, w_{i-1})$

**问题**: 数据稀疏 (OOV), 难以泛化.

### 4.3 Word2Vec (2013) ⭐ 革命性

由 Tomas Mikolov 提出, **第一个真正实用**的词向量方法. 两种架构:

**CBOW (Continuous Bag-of-Words)**: 用上下文词预测中心词

```python
# 输入层: 上下文词的 one-hot (窗口大小 2C)
# 投影层: 平均或拼接 → 隐藏层 (d 维)
# 输出层: 中心词的 softmax 概率分布

# 损失: -log P(w_center | w_context)
```

**Skip-Gram**: 用中心词预测上下文词 (大数据集更优)

```python
# 输入层: 中心词 one-hot
# 投影层: d 维向量
# 输出层: 上下文每个词的 softmax
```

训练技巧: **Negative Sampling** (负采样) + **Hierarchical Softmax** 让训练可行.

Word2Vec 的核心贡献: **语义关系可以用向量运算表达**!

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

### 4.4 GloVe (Global Vectors, 2014)

结合全局矩阵分解 (LSA) 和局部上下文 (Word2Vec). 共现矩阵 + 加权最小二乘.

### 4.5 ELMo (2018) ⭐ 从静态到动态

ELMo = **Embeddings from Language Models**. 关键创新: **同一词在不同上下文有不同向量**.

```python
# ELMo 用双向 LSTM (BiLM) 预训练
# 下游任务时, 把每一层的隐状态加权求和

ELMo_vec = s_task * h_LM_1 + s_task * h_LM_2 + ... + s_task * h_LM_L
```

**优势**: 解决一词多义. "bank" 在 "river bank" 和 "bank account" 中向量不同.

### 4.6 Transformer 时代 (2017-) ⭐ 现代主流

详见 [第02章-架构演进](第02章-架构演进.md).

**Self-Attention 替代 RNN 的核心优势**:
- **并行计算**: O(1) 路径长度 (vs RNN O(n))
- **长距离依赖**: 直接建模任意位置关系
- **可扩展性**: 训练效率大幅提升

---

## 5. 预训练范式演进

| 范式 | 代表 | 训练目标 | 典型应用 |
| :--- | :--- | :--- | :--- |
| **Encoder-Only** | BERT, RoBERTa | MLM + NSP | 理解任务 (分类/QA) |
| **Decoder-Only** | GPT, LLaMA | Next Token Prediction | 生成任务 (对话/创作) |
| **Encoder-Decoder** | T5, BART | Span Corruption | 翻译/摘要 |

详见 [第03章-LLM发展历程总览](第01章-LLM发展历程总览.md).

---

## 6. LLM 时代的 NLP

### 6.1 In-Context Learning

GPT-3 引入的范式: **不需要 fine-tune, 只需在 prompt 里给几个示例**.

```python
prompt = """
把英文翻译成中文:
sea → 海
sky → 天
star →
"""
# LLM 输出: 星
```

### 6.2 Prompt Engineering

来自 [llm-cookbook](https://github.com/datawhalechina/llm-cookbook) 的核心技巧:

| 技巧 | 说明 | 例子 |
| :--- | :--- | :--- |
| **Zero-shot** | 无示例直接问 | "翻译: hello →" |
| **Few-shot** | 给几个示例 | (上面 In-Context) |
| **CoT (Chain-of-Thought)** | 让模型"逐步思考" | "一步步算: 23×17 = ?" |
| **ReAct** | 推理 + 行动 (工具调用) | Agent 技术 |
| **Role Prompting** | 指定角色 | "你是一位资深数学老师" |

### 6.3 RAG (Retrieval-Augmented Generation)

```python
# 1. 用户提问
query = "Datawhale 是什么?"

# 2. 检索相关文档
relevant_docs = vector_db.search(query, top_k=3)

# 3. 构造 prompt
prompt = f"""基于以下资料回答问题:
{relevant_docs}

问题: {query}
"""

# 4. LLM 生成答案
answer = llm.generate(prompt)
```

详见 [../Agent技术/第02章-RAG检索增强生成](../Agent技术/第02章-RAG检索增强生成.md).

---

## 7. 实践: 从零实现简易分词器

```python
import re
from collections import Counter

class SimpleTokenizer:
    """BPE 算法简化版"""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # 字符对 → 合并规则

    def train(self, corpus):
        """从语料训练 BPE merges"""
        # 初始化词汇表 (每个字符)
        vocab = Counter()
        for word in corpus:
            # 单词拆成字符序列, 加结束符 </w>
            vocab[' '.join(list(word)) + ' </w>'] += 1

        # 迭代合并
        for _ in range(self.vocab_size - len(set(' '.join(vocab)))):
            # 统计字符对频率
            pairs = Counter()
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq

            if not pairs:
                break

            # 合并最高频字符对
            best = max(pairs, key=pairs.get)
            self.merges[best] = len(self.merges)

            # 应用合并
            new_vocab = {}
            bigram = ' '.join(best)
            for word, freq in vocab.items():
                new_word = word.replace(bigram, ''.join(best))
                new_vocab[new_word] = freq
            vocab = new_vocab

        self.vocab = set()
        for word in vocab:
            self.vocab.update(word.split())

    def encode(self, text):
        """编码文本为 token IDs"""
        tokens = []
        for word in text.split():
            # 应用 merges
            word_tokens = list(word) + ['</w>']
            for merge, _ in sorted(self.merges.items(), key=lambda x: x[1]):
                bigram = ' '.join(merge)
                word = word.replace(''.join(merge), bigram)
            tokens.extend(word.split())
        return tokens

    def decode(self, tokens):
        """token IDs 解码为文本"""
        return ''.join(tokens).replace('</w>', ' ').strip()


# 使用示例
corpus = ["hello world", "hello python", "hello datawhale"]
tokenizer = SimpleTokenizer(vocab_size=50)
tokenizer.train(corpus)
print(tokenizer.encode("hello world"))
# ['h', 'e', 'l', 'lo', '</w>', 'w', 'o', 'r', 'l', 'd', '</w>']
```

---

## 8. 实践: 从零实现简易文本分类

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    """基于平均词向量的简易文本分类器"""

    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token ids
        # 平均词向量
        emb = self.embedding(x).mean(dim=1)  # (batch, embed_dim)
        return self.fc(emb)  # (batch, num_classes)


# 训练示例
model = TextClassifier(vocab_size=10000, embed_dim=128, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 假设 1000 个样本
for epoch in range(10):
    for batch in dataloader:  # dataloader 自己定义
        inputs, labels = batch['text'], batch['label']
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 9. 实践: 使用 Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModel

# 加载预训练模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 文本分类
text = "今天天气真好"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# outputs.last_hidden_state: (1, seq_len, 768) 最后一层隐藏状态
# outputs.pooler_output: (1, 768) [CLS] token 的池化表示
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(cls_embedding.shape)
# torch.Size([1, 768])
```

---

## 10. 常见面试题

### Q1: One-Hot 编码的优缺点?

**答**:
- **优点**: 简单, 无需学习
- **缺点**: 维度灾难 (词表大小), 语义鸿沟 (向量正交无法表达相似性), 稀疏

### Q2: Word2Vec 的 CBOW 和 Skip-Gram 区别?

**答**:
- **CBOW**: 用上下文预测中心词 (训练快, 小数据集好)
- **Skip-Gram**: 用中心词预测上下文 (大语料表现更好, 低频词效果好)

### Q3: 为什么 ELMo 能解决一词多义?

**答**: ELMo 根据上下文动态生成词向量. 同一个 "bank" 在 "river bank" 和 "bank account" 中, BiLM 生成的隐状态不同, 通过加权求和得到不同的向量.

### Q4: BPE 和 WordPiece 的区别?

**答**:
- **BPE**: 合并频率最高的字符对 (数据驱动)
- **WordPiece**: 合并能最大化训练数据似然的字符对 (语言模型驱动)
- 实际效果接近, 选择主要看框架惯例

### Q5: 大模型时代, NLP 还需要分词吗?

**答**: 仍需要, 但更精细 (subword). 主流 LLM (LLaMA, Qwen) 都使用 BPE/SentencePiece. 优势:
- 固定 vocab (32K-100K)
- 无 OOV
- 多语言支持
- 字符级回退 (BPE 通常用 byte-level)

---

## 11. 关键公式汇总

$$\text{One-Hot: } x_i \in \mathbb{R}^{|V|}, \quad x_i[j] = \begin{cases} 1 & j = i \\ 0 & j \neq i \end{cases}$$

$$P(w_i | w_{i-N+1}, \ldots, w_{i-1}) = \frac{P(w_{i-N+1}, \ldots, w_i)}{P(w_{i-N+1}, \ldots, w_{i-1})}$$

$$\text{Word2Vec (Skip-Gram): } \max \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

$$\text{BPE 合并: } \text{merge}(a, b) \leftarrow \arg\max_{(a,b)} \text{freq}(ab)$$

$$\text{ELMo: } \text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j h_{k,j}^{LM}$$

---

## 12. 参考资源

- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) — 第 1 章 NLP 基础概念 (本附录主要来源)
- [datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook) — Prompt Engineering 教程
- [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) — 开源大模型食用指南
- [datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe) — 动手学大模型应用开发

### 经典论文

1. **Word2Vec**: Mikolov et al., 2013. *Distributed Representations of Words and Phrases and their Compositionality*. arXiv:1310.4546.
2. **ELMo**: Peters et al., 2018. *Deep contextualized word representations*. arXiv:1802.05365.
3. **Transformer**: Vaswani et al., 2017. *Attention Is All You Need*. arXiv:1706.03762.
4. **BERT**: Devlin et al., 2019. *Pre-training of Deep Bidirectional Transformers*. arXiv:1810.04805.
5. **GPT-3**: Brown et al., 2020. *Language Models are Few-Shot Learners*. arXiv:2005.14165.

---

> 本附录补足了 LLM 发展所需的前置 NLP 基础. 后续附录将整合 self-llm 的 LLaMA/Qwen 等开源模型实战,以及 llm-cookbook 的 Prompt Engineering.
