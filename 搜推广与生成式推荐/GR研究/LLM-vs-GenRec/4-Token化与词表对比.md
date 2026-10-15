---
title: 4-LLM vs GenRec Token化与词表对比
tags:
  - LLM
  - GenRec
  - Tokenization
  - SID
  - BPE
  - WordPiece
  - RQ-VAE
  - 词表
  - OOV
  - 冷启动
  - Embedding
  - 对比
description: LLM 与 GenRec 在 Tokenization、词表构造、OOV、冷启动、Embedding 等方面的系统性对比
last_updated: 2026-07-25
status: ✅ v1.0 - Token化与词表对比
---

# 4-LLM vs GenRec Token化与词表对比

> [!info] **本章定位**: 从"输入表征"这一最底层环节出发,系统比较 LLM(自然语言)与 GenRec(用户行为/物品)两种范式如何把"原始世界"切成"模型可吃的离散单元",以及这种切分如何决定后续训练目标、推理效率、扩展性。
>
> [!tip] **阅读建议**: 本章适合已经掌握 Transformer 基础、RQ-VAE / VQ-VAE 概念、SID 教程前 3 章、rec教程第03章 Embedding 的读者。如果第一次接触 SID,先看 [[搜推广与生成式推荐/经典推荐/第03章-Embedding与数据组织]] 和 `../../基础/数学基础/第03章-概率论/0-总目录与阅读路径.md`。
>
> [!info] **与本系列其他章节的关系**: 上一篇 [[3-数据与训练对比]], 下一篇 `5-训练目标与损失对比`。

---

## 目录

- [0. 一句话总结](#0-一句话总结)
- [1. LLM 的 Tokenization](#1-llm-的-tokenization)
  - [1.1 BPE (Byte Pair Encoding)](#11-bpe-byte-pair-encoding)
  - [1.2 WordPiece](#12-wordpiece)
  - [1.3 SentencePiece](#13-sentencepiece)
  - [1.4 tiktoken (OpenAI)](#14-tiktoken-openai)
  - [1.5 Llama 3 Tokenizer](#15-llama-3-tokenizer)
  - [1.6 Qwen Tokenizer](#16-qwen-tokenizer)
- [2. GenRec 的 Tokenization](#2-genrec-的-tokenization)
  - [2.1 Semantic ID (SID) 总览](#21-semantic-id-sid-总览)
  - [2.2 RQ-VAE (TIGER)](#22-rq-vae-tiger)
  - [2.3 RQ-KMeans (阿里 / 快手)](#23-rq-kmeans-阿里--快手)
  - [2.4 LightGCN-based SID](#24-lightgcn-based-sid)
  - [2.5 物品 ID 直接 (OneRec V1)](#25-物品-id-直接-onerec-v1)
  - [2.6 OneRec V2 的 SID 设计](#26-onerec-v2-的-sid-设计)
- [3. 词表大小对比](#3-词表大小对比)
  - [3.1 LLM: 32K–256K](#31-llm-32k256k)
  - [3.2 GenRec: 物品数 (百万到十亿)](#32-genrec-物品数-百万到十亿)
- [4. 词表构造对比](#4-词表构造对比)
  - [4.1 LLM: 统计频率驱动](#41-llm-统计频率驱动)
  - [4.2 GenRec: 聚类 + 量化](#42-genrec-聚类--量化)
- [5. OOV (Out-of-Vocabulary) 处理](#5-oov-out-of-vocabulary-处理)
- [6. 新物品处理 (冷启动)](#6-新物品处理-冷启动)
- [7. Token Embedding 学习对比](#7-token-embedding-学习对比)
- [8. 共享 Embedding 对比](#8-共享-embedding-对比)
- [9. 数学对比](#9-数学对比)
- [10. 实际案例对比](#10-实际案例对比)
- [11. 优劣对比](#11-优劣对比)
- [12. 未来融合](#12-未来融合)
- [附录 A · 速查表](#附录-a--速查表)
- [附录 B · 参考与链接](#附录-b--参考与链接)

---

## 0. 一句话总结

> **LLM 的 Tokenization 是"统计频率 + 字符/字节"驱动的扁平离散化,目标是覆盖自然语言字符序列;GenRec 的 Tokenization 是"语义聚类 + 量化"驱动的层次离散化,目标是压缩物品空间并暴露语义结构。两类方法的差异决定了训练目标、推理方式、冷启动行为和扩展性**。

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| 输入域 | 文本字符流 | 用户行为序列 / 物品原始特征 |
| 基本单元 | 子词 (subword),来自字节合并 | Semantic ID,来自聚类 + 量化 |
| 词表构造 | 统计语料 + 频率合并 | 内容向量 → KMeans → 残差量化 |
| 词表大小 | 32K – 256K | 4K – 64K(粗)+ 物品数(细) |
| OOV 处理 | Byte 回退 / 子词拆分 | 重新跑 Encoder 编码;无天然 Byte 回退 |
| 新物品 | 自动可拆解,立即可 tokenize | 必须先编码得到 SID,才能加入词表 |
| Embedding 共享 | 全部 token 共享一张大表 | SID 与物品 ID 可共享 / 不共享 |
| 层次化 | 字节 → 子词 → 词(隐式) | 多级 RQ-VAE / RQ-KMeans |
| 训练目标耦合度 | 与目标解耦(BERT/ GPT 都用相同 tokenizer) | 与目标强耦合(SID 决定召回质量) |

下面正式进入 12 节正文。

---

## 1. LLM 的 Tokenization

LLM 把"自然语言文本"切成"离散子词序列"。这一步看似简单,实际决定了:

1. **模型能否覆盖罕见词、生僻字、emoji、代码片段**;
2. **同义/近义词是否落在相近 token**(一般不强求,LLM 用 embedding 自学);
3. **词表大小与 embedding 参数量的线性关系**(32K × 4K = 128M 参数);
4. **推理时 token 数与延迟的强相关**(每 token 一次前向);
5. **多语言/多模态扩展的边界**(是否需要回退到 byte)。

下面分别介绍 6 种主流 Tokenizer。

### 1.1 BPE (Byte Pair Encoding)

**BPE** 是 1994 年由 Sennrich 等人(为机器翻译)引入 NLP 的一种算法,后被 GPT-2/3/4 改造为 Byte-Level BPE,沿用至今。它是 LLM 历史上最长久的子词方案。

#### 1.1.1 算法核心

BPE 通过**反复合并语料中最高频的相邻 token 对**来构造词表,基本流程:

```text
输入:  训练语料 C,目标词表大小 V
初始化: 字符集 Σ
循环:
    统计所有相邻 token 对 (x, y) 的频率 freq(x, y)
    (x*, y*) = argmax freq(x, y)
    把 (x*, y*) 加入词表(作为一个新 token)
    把语料中所有 (x*, y*) 替换为新 token
直至: 词表大小 == V
```

**Byte-Level BPE**(GPT-2 起) 在此基础上再加一层:**初始字符集固定为 256 个 UTF-8 字节**。这样任何字符串都能用有限字符 tokenize,不需要单独的 "UNK"。

#### 1.1.2 数学描述

设第 $t$ 步的合并规则为

$$
(x_t, y_t) = \arg\max_{(x,y) \in \mathcal{P}_t} f_t(x, y)
$$

其中 $\mathcal{P}_t$ 是当前所有相邻对集合, $f_t(x,y)$ 是频次。合并后

$$
\Sigma_{t+1} = \Sigma_t \cup \{z_t\}, \quad z_t = x_t \oplus y_t
$$

> 符号说明: $z_t$ 是新 token, $\oplus$ 表示拼接。 $| \Sigma_T | = V$ 时停止。

#### 1.1.3 PyTorch 伪代码 (训练 BPE)

```python
import collections

def get_pairs(word):
    """返回 (prev, curr) 二元组集合"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def train_bpe(corpus, vocab_size, num_merges):
    """简化版 BPE 训练"""
    # 1. 初始化词表为字符集
    vocab = collections.Counter()
    for word, freq in corpus.items():
        vocab[" ".join(list(word))] += freq

    merges = []
    for i in range(num_merges):
        # 2. 统计相邻对频率
        pairs = collections.Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        # 3. 取最高频对
        best = max(pairs, key=pairs.get)
        # 4. 合并到词表
        merged = "".join(best)
        merges.append(best)
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = word.replace(" ".join(best), merged)
            new_vocab[new_word] = freq
        vocab = new_vocab
    return merges, vocab
```

#### 1.1.4 BPE 的优缺点

| 维度 | 优点 | 缺点 |
| :--- | :--- | :--- |
| 实现复杂度 | 极简,O(N·M) | 极简,但工业级需工程优化 |
| 词表覆盖 | Byte-level 后 100% 覆盖 | 极稀有 Unicode 仍被拆得很碎 |
| 多语言 | 各语种共享 256 字节 | 高资源语言 token 短,低资源语言 token 长 |
| 语义信息 | 不引入(后续由 LLM 学习) | 完全无先验 |
| OOV | 不存在(都能 tokenize) | 几乎没有真正的"UNK"概念 |
| 与 embedding 关系 | 一一对应 | 高频词占多 token, 浪费 |

#### 1.1.5 BPE 在主流模型中的使用

| 模型 | BPE 变体 | 词表大小 | 备注 |
| :--- | :--- | :---: | :--- |
| GPT-2 | Byte-level BPE | 50,257 | 经典实现 |
| GPT-3 | Byte-level BPE | 50,257 | 同 GPT-2 |
| GPT-4 | o200k_base | 200,000 | 优化多语言/代码 |
| Llama 1/2 | SentencePiece BPE | 32,000 | SentencePiece 实现 |
| Llama 3 | tiktoken BPE | 128,000 | 更高压缩率 |

### 1.2 WordPiece

**WordPiece** 是 Google 在 BERT 时代提出的子词算法。它和 BPE 几乎相同,但**合并准则不是最高频,而是"加入新 token 后能最大化语料 likelihood"**。

#### 1.2.1 合并准则

设当前模型是把序列当作字符独立出现概率的 unigram 语言模型。加入新合并 $z = x \oplus y$ 前后,语料 log-likelihood 的变化为

$$
\Delta L(z) = \log P(z) - \log P(x) - \log P(y)
$$

= 出现 $z$ 的次数 × [该频次的边际增益]。

实务上等价于

$$
\text{score}(x, y) = \frac{f(x,y)}{f(x) \cdot f(y)}
$$

即该 pair 在"独立出现"的预期次数上的倍数。倍数越大,说明 $x,y$ 共现越"非偶然"。

#### 1.2.2 与 BPE 的对比

| 维度 | BPE | WordPiece |
| :--- | :--- | :--- |
| 合并准则 | 频次最高 | likelihood 增益最大 |
| 计算成本 | 低(只需频次) | 中(需要语料频次) |
| 实现 | 简单 | 稍复杂 |
| 效果 | 高频词好 | 略优(更稳健) |
| 代表模型 | GPT 系列 | BERT、DistilBERT、MobileBERT |

> **pyonote**: WordPiece 在中文场景下表现一般,因为中文"字符"已经被切分得很细,Jieba / 字符级都是主流方案,BERT-base-chinese 即用字符级。

#### 1.2.3 PyTorch 伪代码 (WordPiece 训练片段)

```python
def wordpiece_score(pair_freq, x_freq, y_freq):
    """与 BPE 唯一不同的就是合并准则"""
    return pair_freq / (x_freq * y_freq)

def train_wordpiece(corpus, vocab_size):
    pairs = collections.Counter()
    for word, freq in corpus.items():
        for pair in get_pairs(word):
            pairs[pair] += freq
    merges = []
    while len(merges) < vocab_size:
        scores = {
            pair: wordpiece_score(f, x_freq[pair[0]], y_freq[pair[1]])
            for pair, f in pairs.items()
        }
        best = max(scores, key=scores.get)
        merges.append(best)
        # 合并逻辑同 BPE
        ...
    return merges
```

### 1.3 SentencePiece

**SentencePiece** 是 Google 2018 年开源的"语言无关" tokenizer。与 BPE/WordPiece 不同,它**直接在原始文本(而非预分词)上工作**,因此无需依赖任何特定语言的分词器。

#### 1.3.1 核心思想

- **不预设分词**:把文本看作 UTF-8 字节流(或 Unicode 字符流)
- **支持 BPE / Unigram 两种算法**
- **自带 Normalization**(NFKC、小写化等)

```text
原文: "Hello, world!"
    ↓ normalize (NFKC + 小写)
    ↓ [可选] 按元数据拆分(如 ChatML 模板)
    ↓ BPE / Unigram 合并
tokens: ["<s>", "▁hello", ",", "▁world", "!", "</s>"]
```

#### 1.3.2 Unigram LM 模式

SentencePiece 还提供**Unigram Language Model** 模式,与 BPE 完全相反:

- 从大词表开始
- 每次删除"去掉后 loss 增长最小"的 token
- 直到词表达到目标大小

> Unigram 模式在多语言模型中很常见(如 XLM-R、mBART),效果通常优于 BPE。

#### 1.3.3 SentencePiece 训练 (命令行)

```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=m \
  --vocab_size=32000 \
  --model_type=bpe \
  --character_coverage=0.9995 \
  --byte_fallback=true \
  --input_sentence_size=5000000 \
  --shuffle_input_sentence=true
```

#### 1.3.4 Llama 2 词表构成

Llama 2 的 SentencePiece 词表包含:

| 类别 | 数量 |
| :--- | ---: |
| 256 字节 (byte) | 256 |
| 数字 / 标点 | ~200 |
| 中文 / 日文 / 韩文字符 | ~6,000 |
| 拉丁语字符组合 | ~25,000 |
| 特殊 token (`<s>`, `</s>`, `<unk>`) | 3 |
| **总计** | **32,000** |

### 1.4 tiktoken (OpenAI)

**tiktoken** 是 OpenAI 在 2023 年开源的高性能 BPE 库,用 Rust 实现。它是 GPT-4 / GPT-4o / o1 / Sora 文生视频等模型使用的官方 tokenizer。

#### 1.4.1 关键设计

- **纯 BPE**(无 WordPiece / Unigram)
- **多套词表**: `cl100k_base`(GPT-4)、`o200k_base`(GPT-4o)、`p50k_base`(Codex)、`r50k_base`(GPT-3)
- **正则化预处理**:用 `gpt2_split_pattern`(把 contractions、字母、数字、标点等切开)

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(tokens)          # [9906, 11, 15339, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

#### 1.4.2 o200k_base 与多语言改进

| 词表 | 模型 | 大小 | 多语言压缩率 |
| :--- | :--- | :---: | :---: |
| cl100k_base | GPT-4 | 100,256 | 1.0× (基线) |
| o200k_base | GPT-4o | 200,019 | 1.4× (更短) |
| o200k_harmony | o1 / o3 | 200,096 | 同 o200k + 工具调用 |

> GPT-4o 在中文上的 token 数量平均比 GPT-3.5 少 33%,这是词表扩展的直接收益。

#### 1.4.3 与 HuggingFace `tokenizers` 的关系

| 维度 | tiktoken | tokenizers (HF) |
| :--- | :--- | :--- |
| 速度 | 极快(Rust + 缓存) | 快(Rust) |
| 算法 | 仅 BPE | BPE / WordPiece / Unigram |
| 训练 | 不支持 | 支持 |
| 多模态 | 否 | 否 |
| 主要用户 | OpenAI | HuggingFace 生态 |

### 1.5 Llama 3 Tokenizer

Meta 在 Llama 3 中把 tokenizer 从 Llama 2 的 SentencePiece BPE(32K)升级为 **tiktoken 风格的 BPE,词表扩到 128K**。

#### 1.5.1 设计动机

- **更高压缩率**:平均每 token 编码更多字符 → 训练更快 / 推理更长上下文
- **更好的多语言支持**:128K 词表能容纳大量非英语 token
- **保留数字一致性**:每个数字 0–9 一个 token(与 Llama 2 一致),便于数学任务

#### 1.5.2 Llama 3 vs Llama 2 对比

| 维度 | Llama 2 | Llama 3 |
| :--- | :--- | :--- |
| 算法 | SentencePiece BPE | tiktoken BPE |
| 词表大小 | 32,000 | 128,256 |
| 多语言 | 一般 | 显著增强 |
| 数字 | 0–9 独立 token | 同 |
| 中文压缩率(对比 GPT-3.5) | 1.5× | 0.7× |
| 训练工具 | SentencePiece | tiktoken + 自研 |

#### 1.5.3 实践意义

- **短 prompt 更短**:同样的"hello world"在 Llama 2 占 2 tokens,Llama 3 可能占 1 token
- **多语言对话**:中文 token 长度缩短约 30%,直接降低 GPU 时延
- **多模态扩展**:Llama 3.1 / 3.2 仍用同词表,把图像切成 visual token 走单独编码器

### 1.6 Qwen Tokenizer

阿里 Qwen 系列(通义千问)对中文场景做了极致优化。

#### 1.6.1 词表构成

| 版本 | 词表大小 | 关键设计 |
| :--- | :---: | :--- |
| Qwen | 151,851 | 中文 + 英文 + 代码 |
| Qwen2 | 151,851 | 进一步合并高频组合 |
| Qwen2.5 | 151,851 | 同词表,改进训练 |
| Qwen3 | 151,851 | 同词表,扩展语言 |

> 与 Llama 3 选 128K 不同,Qwen 选了 ~152K,中文占比显著更高。

#### 1.6.2 中文优化技巧

- **大量 Bigram/Trigram 中文片段**(如 "中国", "一个", "我们")
- **成语/四字词组合**(如 "一如既往", "不可思议")
- **数字 + 量词组合**(如 "1个", "10万", "50%")
- **保留 byte 回退**:少量生僻字拆字节

#### 1.6.3 Qwen vs Llama 3 中文压缩率(实证)

```text
文本: "今天天气不错,我们去公园散步。"
Qwen2 编码: ["今天", "天气", "不错", "，", "我们", "去", "公园", "散步", "。"]   → 9 tokens
Llama 3 编码: ["今","天","天","气","不","错","，","我","们","去","公","园","散","步","。"]  → 15 tokens
```

> 在中文场景,Qwen 比 Llama 3 平均短 30–40%,直接影响推理成本与速度。

#### 1.6.4 Qwen tokenizer 的多模态扩展

- **Qwen-VL**:在词表外引入 `<image>`、`<|im_start|>`、`<|im_end|>` 等特殊 token
- **位置编码**:视觉 token 单独计算 RoPE 索引

---

## 2. GenRec 的 Tokenization

GenRec 的"token"是**用户行为序列中的"下一个物品"**,因此核心问题是:

> **如何把"亿万个物品"用"有限 token 序列"表达?**

答案有三条技术路线:

1. **直接用物品 ID**(OneRec V1);
2. **用语义聚类 + 量化的产物作为 Semantic ID**(TIGER / RQ-VAE);
3. **混合路线**:部分场景用 SID,部分场景用品类 ID 或属性 ID。

下面逐项展开。

### 2.1 Semantic ID (SID) 总览

**Semantic ID (SID)** 是 GenRec 时代最核心的离散化技术。它把"物品的稠密表示"量化成一个或多个**层次化的离散 token 序列**,例如:

```text
物品 123456789 → [3, 27, 89]   # 三级 SID
物品 987654321 → [3, 27, 92]   # 兄弟 SID,前两级相同
```

这种设计带来三个好处:

1. **表达力**:`vocab_0 × vocab_1 × vocab_2` 个物品可被 `|vocab_i|^(层次数)` 组合覆盖(几何级)
2. **语义结构**:相近物品 SID 前缀相同,Transformer 易学
3. **可生成**:`[query; history; SID_prefix]` → 续写下一个 SID token,再用 TRIE 检索物品

#### 2.1.1 SID 的统一数学定义

给定物品 $i$,其语义表示为 $\mathbf{z}_i \in \mathbb{R}^d$。层级数为 $L$,第 $k$ 级码本为 $\mathcal{C}_k = \{\mathbf{e}_k^{(1)}, \dots, \mathbf{e}_k^{(K)}\}$,则 SID 生成过程为:

$$
c_k = \arg\min_{j \in [K]} \| \mathbf{r}_{k-1} - \mathbf{e}_k^{(j)} \|_2^2, \quad
\mathbf{r}_k = \mathbf{r}_{k-1} - \mathbf{e}_k^{(c_k)}
$$

其中 $\mathbf{r}_0 = \mathbf{z}_i$, $c_k \in [K]$ 是第 $k$ 层 token。最终 SID 为元组 $(c_1, c_2, \dots, c_L)$。

#### 2.1.2 SID 与 LLM Token 的本质区别

| 维度 | LLM Token | GenRec SID |
| :--- | :--- | :--- |
| 编码函数 | 查表(确定) | 编码器 + 量化(需训练) |
| 词汇 | 字符 / 子词 | 语义聚类中心 |
| 覆盖率 | 100%(字节回退) | 仅覆盖 Encoder 训练过的物品 |
| 序列结构 | 1D 线性 | 1D 序列 / 多级并行(看实现) |
| 损失重建 | 无 | Encoder-Decoder 重构误差 |
| 与内容耦合 | 松散 | 紧密 |

### 2.2 RQ-VAE (TIGER)

**RQ-VAE**(Residual-Quantized VAE) 是 TIGER 论文的核心技术,由 Google 2023 年提出。它把"物品 → 文本/图像 → 连续向量 → 多级离散 token"做成一个端到端可训练系统。

#### 2.2.1 整体架构

```text
                        ┌───────────────┐
                        │ 物品内容       │
                        │ (标题/封面/类目) │
                        └───────┬───────┘
                                ▼
                     ┌─────────────────────┐
                     │  Encoder (文本 / 视觉) │
                     │     E_θ              │
                     └─────────┬───────────┘
                              z_0 (d 维向量)
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Residual Quantization (RQ)    │
              │   c_1 = argmin ||z_0 - e_j||    │
              │   r_1 = z_0 - e_{c_1}          │
              │   c_2 = argmin ||r_1 - e_j||    │
              │   r_2 = r_1 - e_{c_2}          │
              │   ...                            │
              └───────────────┬────────────────┘
                              ▼
                       SID = (c_1, c_2, c_3)
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Decoder D_θ                    │
              │ 重建出文本/视觉内容              │
              └───────────────┬────────────────┘
                              ▼
                      重建损失 ||D(E(x)) - x||
```

#### 2.2.2 RQ 的数学公式

设第 $k$ 层码本为 $\mathcal{E}_k = \{\mathbf{e}_k^{(1)}, \dots, \mathbf{e}_k^{(K)}\}$,残差向量 $\mathbf{r}_k$ 满足:

$$
\mathbf{r}_0 = \mathbf{z}, \quad
c_k = \arg\min_{j} \| \mathbf{r}_{k-1} - \mathbf{e}_k^{(j)} \|_2^2, \quad
\mathbf{r}_k = \mathbf{r}_{k-1} - \mathbf{e}_k^{(c_k)}
$$

重建误差:

$$
\mathcal{L}_{\text{RQ-VAE}} = \| \mathbf{x} - D_\theta(\hat{\mathbf{z}}) \|_2^2 + \beta \sum_k \| \text{sg}[\mathbf{r}_{k-1}] - \mathbf{e}_k^{(c_k)} \|_2^2 + \| \mathbf{r}_{k-1} - \text{sg}[\mathbf{e}_k^{(c_k)}] \|_2^2
$$

> 后两项是 **VQ 损失**:commitment loss + codebook loss,来自 VQ-VAE(van den Oord 2017)。

#### 2.2.3 TIGER 端到端训练

```python
import torch
import torch.nn as nn

class RQVAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x):
        return self.net(x)

class ResidualQuantizer(nn.Module):
    def __init__(self, latent_dim, num_levels, codebook_size):
        super().__init__()
        # 多层码本
        self.codebooks = nn.ParameterList([
            nn.Embedding(codebook_size, latent_dim)
            for _ in range(num_levels)
        ])
        self.num_levels = num_levels

    def forward(self, z):
        residual = z
        codes = []
        for cb in self.codebooks:
            # (B, latent_dim) @ (latent_dim, codebook_size)
            dist = torch.cdist(residual.unsqueeze(1), cb.weight.unsqueeze(0))
            code = dist.argmin(dim=-1)               # (B, 1)
            codes.append(code)
            quantized = cb(code).squeeze(1)          # (B, latent_dim)
            residual = residual - quantized
        return torch.cat(codes, dim=-1), residual    # (B, num_levels), (B, latent_dim)

class RQVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim,
                 num_levels=3, codebook_size=256):
        super().__init__()
        self.encoder = RQVAEEncoder(in_dim, hidden_dim, latent_dim)
        self.quantizer = ResidualQuantizer(latent_dim, num_levels, codebook_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        codes, residual = self.quantizer(z)
        x_hat = self.decoder(residual)
        return x_hat, codes
```

#### 2.2.4 RQ-VAE 的优劣

| 维度 | 优势 | 劣势 |
| :--- | :--- | :--- |
| 表达力 | 多层组合 → 几何级物品 | 重建误差累积 |
| 训练 | 端到端 | 码本利用率不均(有 dead code) |
| 服务 | 物品编码一次,cache 住 | 新物品必须重跑 Encoder |
| 推理 | 续写 SID token 即可 | 需 TRIE 约束解码 |
| 工业部署 | 成熟(TIGER / COURIER 都用) | Encoder 服务化是单点 |

### 2.3 RQ-KMeans (阿里 / 快手)

**RQ-KMeans** 是阿里(淘宝、天猫)、快手等公司在大规模工业场景中常用的 SID 方案,与 RQ-VAE 的区别:**用 KMeans 替代 VQ-VAE 的 Encoder-Decoder**,直接在"物品 embedding"上做层次聚类。

#### 2.3.1 为什么用 KMeans 而不是 VAE?

| 维度 | RQ-VAE | RQ-KMeans |
| :--- | :--- | :--- |
| Encoder 训练 | 需训练 VAE | 直接用物品 embedding |
| 聚类初始化 | VAE 隐空间 | 现成 item embedding |
| 计算成本 | 重训 VAE | KMeans 几小时 |
| 端到端 | 是 | 否(解耦) |
| 适配新物品 | 需 Encoder | 重新聚类即可 |
| 工业实践 | TIGER, COURIER | 阿里 OneRec V2, 快手 |

#### 2.3.2 RQ-KMeans 算法流程

```text
输入: 物品 embedding {e_1, ..., e_N},  层次数 L, 每层码本 K_l
输出: 每个物品的多级 SID {c_1, ..., c_L}

for l in 1..L:
    # 1. KMeans 聚类
    cluster_ids = KMeans(K_l).fit_predict(current_residuals)
    # 2. 取每个簇的 centroid
    centroids = compute_centroids(current_residuals, cluster_ids)
    # 3. 计算残差
    current_residuals = current_residuals - centroids[cluster_ids]
    # 4. 记录第 l 层 ID
    codes[l] = cluster_ids
```

#### 2.3.3 数学描述

$$
\mathbf{r}_0 = \mathbf{e}_i, \quad
\mu_{l,j} = \frac{1}{|\mathcal{C}_{l,j}|} \sum_{i \in \mathcal{C}_{l,j}} \mathbf{r}_{l-1,i}, \quad
c_{l,i} = \arg\min_j \| \mathbf{r}_{l-1,i} - \mu_{l,j} \|_2^2, \quad
\mathbf{r}_{l,i} = \mathbf{r}_{l-1,i} - \mu_{l,c_{l,i}}
$$

#### 2.3.4 阿里 OneRec V2 的 SID 设计要点

- **3 级 SID**:每级 1000 类,组合后 $10^9$ 个 ID,可覆盖 10 亿物品
- **结合类目信息**:第一级固定为类目中心,后两级用 KMeans
- **冷启动**:新物品直接走一遍 KMeans,分钟级可获得 SID
- **跨域共享**:类目编码共用 → SID 第一级天然跨域

#### 2.3.5 快手实践

- **4 级 SID**:第一级 2K,后三级各 1K → 2 × 10^9
- **多模态融合**:封面图 + 标题 + 标签 → 统一 embedding → KMeans

### 2.4 LightGCN-based SID

**LightGCN**(He et al. 2020)是推荐系统最经典的图卷积模型。一些 GenRec 工作把它作为 Encoder,把"物品的二部图结构"编码到 embedding,再做 RQ。

#### 2.4.1 流程

```text
用户-物品交互图 G = (U ∪ V, E)
        │
        ▼
LightGCN 多层传播
H^(l+1) = D^{-1/2} A D^{-1/2} H^(l)      ← 简化 GCN
        │
        ▼
物品 embedding e_i = average(H^(0..L))
        │
        ▼
RQ-KMeans 得到 SID
```

#### 2.4.2 与 RQ-VAE 的本质区别

| 维度 | RQ-VAE (TIGER) | LightGCN-SID |
| :--- | :--- | :--- |
| 编码源 | 内容(文本/图像) | 协同过滤图 |
| 依赖内容特征 | 是 | 否 |
| 冷启动 | 需内容 embedding | 需交互数据 |
| 解释性 | 强(语义) | 弱(纯结构) |
| 实际效果 | 长尾更好 | 头物品更准 |

### 2.5 物品 ID 直接 (OneRec V1)

**OneRec V1**(阿里 2025)是国内第一个端到端生成式推荐的工作。它的 tokenization 是**最朴素也最反直觉**的:

> **直接把每个物品 ID 当作一个 token**。

#### 2.5.1 为什么可以这么做?

| 条件 | OneRec V1 满足? |
| :--- | :---: |
| 物品量可控(百万级) | ✅ |
| 接受大 embedding 表 | ✅(H800 显存够) |
| 训练数据足够 | ✅(千亿样本) |
| 检索效率可优化 | ✅(TRIE + Beam) |
| 冷启动不重要 | ✅(电商场景,新物品可走老通道) |

#### 2.5.2 词表构造

```text
物品库 V = {i_1, ..., i_N}      N = 5 × 10^6  (电商)
特殊 token: [PAD], [BOS], [EOS], [MASK]
词表大小: 5,000,000 + 4
```

embedding 表大小 = $5 \times 10^6 \times 128$ = $640$ MB(FP16,只是 128 维),可控。

#### 2.5.3 训练目标

$$
\mathcal{L} = -\log P(i_t | i_1, i_2, \dots, i_{t-1}; \theta)
$$

其中 $i_t$ 是**下一个物品**,直接做交叉熵。

> 这里直接套用 LLM 的 Next-Token Prediction 范式,只不过"token"是物品 ID。

#### 2.5.4 与 LLM 的本质相似

| 维度 | LLM | OneRec V1 |
| :--- | :--- | :--- |
| 词表大小 | 32K – 256K | 1M – 10M |
| 序列长度 | 1K – 1M | 100 – 1000 |
| 单 token embedding 维度 | 1024 – 4096 | 64 – 256 |
| 推理方式 | 自回归 | 自回归 + TRIE 约束 |
| 训练目标 | NTP | NIP(下一物品) |

#### 2.5.5 OneRec V1 的优点

- **简单直接**:不需要单独的 Encoder,不需要聚类
- **个性化强**:每个物品独立 token,完全保留个性化
- **可扩展**:加新物品 = 加一行 embedding

#### 2.5.6 OneRec V1 的缺点

- **词表巨大**:5M token,embedding 表 640MB+
- **稀疏**:长尾物品几乎学不到,需要额外损失(BPR / In-batch Neg)
- **无共享**:相近物品没有参数共享,泛化差

### 2.6 OneRec V2 的 SID 设计

阿里在 OneRec V2(2025)上做了**根本性升级**:抛弃直接 ID,改用 **RQ-KMeans 生成的 SID**。

#### 2.6.1 升级动机

| OneRec V1 问题 | V2 解决方案 |
| :--- | :--- |
| 词表大(5M) | 3 级 SID,词表 1K×3 = 10^9 容量 |
| 长尾物品学不到 | 同类物品共享 token,共享梯度 |
| 无语义结构 | SID 前缀即类目,Transformer 易学 |
| 推理慢 | SID 续写 → 候选物品集小 |

#### 2.6.2 V2 SID 设计细节

```text
物品  i
   │
   ▼
文本 embedding (Qwen2.5-7B 编码) → e_i ∈ R^4096
   │
   ▼
类目分类 (一级固定为类目 ID)
   │
   ▼
KMeans on residual (二、三级)
   │
   ▼
SID = (cat_id, k1_id, k2_id)
```

#### 2.6.3 V2 的训练目标

```python
# 同时监督 3 个 token 的预测
loss = (
    F.cross_entropy(logits_level1, codes[:, 0]) +
    F.cross_entropy(logits_level2, codes[:, 1]) +
    F.cross_entropy(logits_level3, codes[:, 2])
)
```

> 这是**多任务 NTP**,详见 `5-训练目标与损失对比`。

#### 2.6.4 V2 关键工程点

- **TRIE 约束解码**:续写时只允许同前缀物品
- **Beam Search + Cache**:平均展开 < 100 个候选
- **冷启动**:新物品直接走一遍 Qwen + KMeans,几秒上线

#### 2.6.5 V2 的效果

| 指标 | V1(直接 ID) | V2(RQ-KMeans SID) |
| :--- | :---: | :---: |
| Hit@10 | 0.321 | 0.358 |
| NDCG@10 | 0.184 | 0.207 |
| 长尾 Hit@10 | 0.082 | 0.146 |
| 推理延迟 | 38ms | 22ms |

> V2 不仅全面超过 V1,而且**长尾提升最显著**(因为 SID 让冷门物品"借光"头部)。

---

## 3. 词表大小对比

词表大小决定了:

- embedding 表参数量
- 输出层 softmax 计算量
- 推理时的 search space 大小

| 维度 | LLM | GenRec |
| :--- | :---: | :---: |
| 典型范围 | 32K – 256K | 4K – 10M+ |
| 增长动机 | 多语言 / 多模态 | 覆盖物品库 |
| 单 token 维度 | 1024 – 4096 | 64 – 256 |
| embedding 总参数量 | 30M – 1G | 1M – 2G |

### 3.1 LLM: 32K – 256K

#### 3.1.1 词表大小的工程权衡

| 词表大小 | 多语言 | 模型 | 备注 |
| :---: | :--- | :--- | :--- |
| 8K | 英文为主 | Llama 1(早期实验) | 已淘汰 |
| 32K | 中等 | Llama 1 / 2 | 中文压缩率差 |
| 64K | 良好 | Mistral | 法国 Mistral AI |
| 100K | 优秀 | GPT-4 | 多语言增强 |
| 128K | 极佳 | Llama 3 | 多语言 + 长上下文 |
| 200K | SOTA | GPT-4o / o1 | 多语言 + 多模态 |

#### 3.1.2 词表大小的极限

> **不可无限扩大**:词表扩到 1M,即使 dim=128,embedding 表也 128M 参数;而且 softmax 在 1M 类上的计算非常昂贵。

实践中:

- **50K – 200K** 是主流 LLM 的甜区
- 更大的词表只在"必须覆盖极稀有 token"时使用

#### 3.1.3 词表与计算量的关系

```text
embedding lookup:    O(seq_len × dim)
softmax over vocab:  O(seq_len × vocab_size × dim)   ← 主导项
```

> 推理时 softmax over vocab 是主要瓶颈。词表 200K vs 50K,最后一层慢 4 倍。

### 3.2 GenRec: 物品数 (百万到十亿)

#### 3.2.1 三种典型规模

| 规模 | 代表 | 词表设计 | 备注 |
| :--- | :--- | :--- | :--- |
| 百万级 | 淘宝 OneRec V1 | 5M 物品 ID | 简单直接 |
| 千万级 | 抖音电商 | 10M + SID | 头物品 + SID 混合 |
| 亿级 | Amazon | 100M 物品 + SID | 多级 SID 必备 |
| 十亿级 | Walmart | 1B 物品 + 4 级 SID | 工业极限 |

#### 3.2.2 物品 ID vs SID 的等效容量

| 设计 | 词表 | 组合容量 |
| :--- | :--- | :--- |
| 直接 ID | 1M token | 1M |
| 2 级 SID 256×256 | 256 token / 级 | 256² = 65,536 |
| 3 级 SID 256×256×256 | 256 token / 级 | 256³ ≈ 16M |
| 3 级 SID 1K×1K×1K | 1K token / 级 | 10⁹ |
| 4 级 SID 1K×1K×1K×1K | 1K token / 级 | 10¹² |

> 一句话:**多级 SID 用"小词表 × 序列长度"换"大词表 × 单 token"**。

#### 3.2.3 实际生产规模

```text
阿里 OneRec V2:  10 亿物品 × 3 级 SID(1000³)
字节 HSTU:      30 亿行为 × item ID embedding
美团 MTGR:      5 亿商家 × 2 级 SID
腾讯太极:      100 亿 SKU × 4 级 SID
```

---

## 4. 词表构造对比

| 阶段 | LLM | GenRec |
| :--- | :--- | :--- |
| 数据来源 | 文本语料(网页/书/对话) | 物品内容 / 协同图 |
| 构造算法 | 频次合并 / 频次合并 + likelihood | KMeans / RQ-VAE |
| 是否需要 Encoder | 否(直接统计) | 是(Encoder 必须先训) |
| 是否可微 | 否(查表) | 是(Encoder + Decoder 联合训练) |
| 训练一次 | 静态 | 可重新生成(SID 重训) |

### 4.1 LLM: 统计频率驱动

LLM 词表是**离线一次性构造**,流程:

```text
1. 收集原始语料 corpus (GB ~ TB)
2. normalize (NFKC, 小写化, 去重)
3. 字符切分 (BPE: 用字节; WordPiece: 需预分词)
4. 反复合并最高频对
5. 输出 merges + vocab
```

#### 4.1.1 字符覆盖率 (character_coverage)

SentencePiece 有一个关键参数 `character_coverage`:

```text
--character_coverage=0.9995   # 99.95% 的字符纳入基础字符集
--character_coverage=0.9999   # 99.99%,中文常用
--character_coverage=1.0      # 100%,emoji/罕见字符也覆盖
```

`character_coverage < 1` 时,稀有字符会被替换为 `<UNK>`(实际上 SentencePiece 默认用 byte 回退)。

#### 4.1.2 训练数据量与词表质量

```text
小语料 (1GB):
  → 词表过拟合
  → 大量 byte token
大语料 (1TB):
  → 词表稳健
  → 高频词组占主导
```

经验法则:**语料 ≥ 100×词表大小(按 token 数)** 即可得到高质量词表。

### 4.2 GenRec: 聚类 + 量化

GenRec 词表(以 SID 为例)的构造流程:

```text
1. 收集物品原始特征 corpus
2. Encoder 编码 (内容 / 协同 / 多模态)  → Z ∈ R^{N×d}
3. (可选) PCA / whitening / normalize
4. 层次聚类 (KMeans 多层)
5. 输出 codebook + 每物品的 SID
```

#### 4.2.1 Encoder 选择

| Encoder | 输出 | 适用场景 |
| :--- | :--- | :--- |
| Qwen2.5-7B | 4096 维文本向量 | 文本主导 |
| CLIP ViT-L/14 | 768 维多模态向量 | 图文混合 |
| BERT-base | 768 维文本向量 | 英文场景 |
| LightGCN | 64 维协同向量 | 无内容/纯协同 |
| 自研双塔 | 128 维专属向量 | 工业极致 |

#### 4.2.2 聚类参数

```python
from sklearn.cluster import KMeans

# 第一级 KMeans:大粒度
km1 = KMeans(n_clusters=1000, n_init=10, max_iter=300)
codes_l1 = km1.fit_predict(item_embeddings)

# 残差
residuals = item_embeddings - km1.cluster_centers_[codes_l1]

# 第二级 KMeans:中粒度
km2 = KMeans(n_clusters=1000, n_init=10, max_iter=300)
codes_l2 = km2.fit_predict(residuals)

# 残差
residuals = residuals - km2.cluster_centers_[codes_l2]

# 第三级 KMeans:细粒度
km3 = KMeans(n_clusters=1000, n_init=10, max_iter=300)
codes_l3 = km3.fit_predict(residuals)
```

#### 4.2.3 KMeans 的失效场景

- **极度不均衡物品**:头部物品多,长尾少 → KMeans 簇大小差 100×
- **冷启动新物品**:不在 KMeans 输入中 → 重新聚类
- **Encoder 漂移**:Encoder 在线更新 → SID 不稳定

> 工业解法:**定期重聚类**(每 1 – 7 天)+ **固定类目中心作为第一级**。

---

## 5. OOV (Out-of-Vocabulary) 处理

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| 是否存在 OOV | 否(byte 回退) | 是(物品不在编码器训练集) |
| OOV 原因 | — | 新物品 / Encoder 漂移 / 数据缺失 |
| 应对 | 拆字节 | 重新 Encoder / 临时用品类 ID |
| 损失大小 | 几乎为零 | 显著(冷启动) |

### 5.1 LLM: Byte 回退天然无 OOV

#### 5.1.1 Byte-Level BPE 的工作原理

GPT-2 起的做法:

```text
"中文 emoji 😄"
   ↓ UTF-8 编码
b'\xe4\xb8\xad\xe6\x96\x87 ... '
   ↓ 字节级 BPE
[0xE4, 0xB8, 0xAD, 0xE6, 0x96, 0x87, ..., 0xF0, 0x9F, 0x98, 0x84]
   ↓ BPE 合并
[<unk_word>, ..., <emoji_combine>]
```

256 个字节保证 100% 覆盖。任何 Unicode 字符都能拆。

#### 5.1.2 实测多语言压缩率

| 词表 | 英文压缩率 | 中文压缩率 | 阿拉伯文压缩率 |
| :--- | :---: | :---: | :---: |
| GPT-2 (50K) | 1.0× | 2.5× | 2.0× |
| GPT-4 (100K) | 0.8× | 1.4× | 1.6× |
| Qwen2 (152K) | 1.1× | 0.9× | 1.3× |

> 词表越大、字节回退越多,压缩率分布越均衡。

### 5.2 GenRec: 新物品 = 重新编码

#### 5.2.1 冷启动物品的 SID 生成

```python
def gen_sid_for_new_item(item_content):
    # 1. 内容编码
    embedding = encoder.encode(item_content)         # (d,)

    # 2. KMeans 找最近簇
    residual = embedding
    codes = []
    for km in [km1, km2, km3]:
        code = km.predict(residual.reshape(1, -1))[0]
        codes.append(code)
        residual = residual - km.cluster_centers_[code]

    return codes   # [c1, c2, c3]
```

#### 5.2.2 没有 Encoder 时的兜底

| 场景 | 兜底策略 |
| :--- | :--- |
| 完全没有内容 | 用类目 ID + 0 填充后两级 |
| Encoder 服务挂了 | 走老物品的 SID 平均 |
| KMeans 还没聚类 | 用全局 SID `[0, 0, 0]` |
| 新平台冷启动 | 用文本 embedding + KMeans |

#### 5.2.3 与 LLM 的本质区别

> **LLM 的 OOV 是字符级,代价是 token 变长;GenRec 的 OOV 是物品级,代价是必须经过完整 Encoder 链路,可能延迟几分钟到几小时**。

---

## 6. 新物品处理 (冷启动)

冷启动是 GenRec 的头号难题。LLM 没有"新物品",但有"新概念",对比看:

| 维度 | LLM 新概念 | GenRec 新物品 |
| :--- | :--- | :--- |
| 出现频率 | 罕见词 | 新上架商品 |
| 表征来源 | 子词拼接 | 内容 / 行为 |
| 接入时间 | 即时 | 分钟 ~ 天 |
| 表现差距 | 与旧词同 | 通常 30–50% 差 |

### 6.1 GenRec 冷启动的 SID 路线

#### 6.1.1 全冷启动(无任何行为)

```text
策略: 内容 embedding + KMeans → SID
依赖: 文本 / 图像 / 类目 / 标签
效果: 中等(优于随机,劣于热物品)
时间: 秒级
```

#### 6.1.2 弱冷启动(< 10 次曝光)

```text
策略: 行为相似度检索 + SID 平均
依赖: 相似物品的 SID
效果: 接近热物品的 60%
时间: 分钟级
```

#### 6.1.3 强冷启动(> 100 次曝光)

```text
策略: 重训 Encoder(增量) / 聚类重做
依赖: 全量训练数据
效果: 与热物品接近
时间: 小时 ~ 天
```

### 6.2 阿里 OneRec V2 的冷启动管线

```text
新商品上架
   ↓
   Qwen2.5-7B 编码标题 + 类目 + 标签
   ↓
   取 e_i ∈ R^4096
   ↓
   第一级: 类目字典查表 → c_1
   ↓
   第二级: KMeans 找最近簇 → c_2
   ↓
   第三级: 残差 KMeans → c_3
   ↓
   SID = (c_1, c_2, c_3)
   ↓
   立即入候选池(走 TRIE 约束解码)
   ↓
   累积行为后重新微调 SID
```

### 6.3 LLM 的"新概念" vs GenRec 的"新物品"

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| 新单位 | 罕见词 | 新商品 |
| 处理时间 | 即时(子词拼接) | 分钟级(KMeans) |
| 实际表现 | 略低于常见词 | 显著低于热物品 |
| 长期学习 | 训练数据多了自动学 | 重聚类 + 重训 |
| 优化方向 | 扩大词表 + byte 回退 | 多模态 + 跨域 SID |

---

## 7. Token Embedding 学习对比

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| Embedding 初始化 | 随机 | 随机 / 来自 Encoder |
| 是否共享输入输出 | 是(tied) | 否(常见) |
| 维度选择 | 越大越好(2048+) | 64 – 256 |
| 训练目标耦合 | 与 LM 目标耦合 | 与召回/排序目标耦合 |
| 与预训练关系 | 不可换词表(否则重训) | 可换 SID(Encoder 重训即可) |

### 7.1 LLM Token Embedding

```python
class LLMEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # 输入 embedding
        self.input_embed = nn.Embedding(vocab_size, d_model)
        # 输出 embedding (与输入共享)
        self.output_embed = self.input_embed  # tied
        # 缩放因子(深度模型常用)
        self.scale = d_model ** 0.5

    def forward(self, x):
        return self.input_embed(x) * self.scale
```

#### 7.1.1 Tied Embedding 的好处

- **参数减半**(GPT-3 175B 用 tied 节省 90G 显存)
- **语义一致性**:输入输出共享表征空间

#### 7.1.2 Tied 的代价

- 训练初期不稳定(共享梯度)
- 某些模型(DeepSeek)放开 tied 效果更好

### 7.2 GenRec SID Embedding

```python
class SIDEmbedding(nn.Module):
    def __init__(self, num_levels, codebook_size, d_model):
        super().__init__()
        # 每一级一个独立 embedding
        self.level_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model)
            for _ in range(num_levels)
        ])
        # 位置编码(可选)
        self.pos_emb = nn.Embedding(num_levels, d_model)

    def forward(self, codes):
        # codes: (B, num_levels)
        embs = []
        for lvl, embed in enumerate(self.level_embeds):
            e = embed(codes[:, lvl])                  # (B, d_model)
            e = e + self.pos_emb(torch.tensor(lvl))   # 加位置
            embs.append(e)
        return torch.stack(embs, dim=1)                # (B, num_levels, d_model)
```

#### 7.2.1 多级 Embedding 是否独立?

| 方案 | 参数共享 | 效果 | 备注 |
| :--- | :--- | :--- | :--- |
| 完全独立 | 否 | 最好 | 参数量 × 层次数 |
| 第一级共享类目 | 第一级独立 | 中等 | 工业常用 |
| 完全共享 | 是 | 差 | 层级信息丢失 |

#### 7.2.2 Embedding 维度选择

```text
LLM: dim 随模型规模 → 1024 (1B), 2048 (7B), 4096 (70B)
GenRec: dim 固定较小 → 64 (双塔), 128 (OneRec V2), 256 (OneRec V1)
```

> GenRec 维度小是因为序列短(100-1000)且要快推理。

### 7.3 输入/输出解耦 vs 共享

| 场景 | 输入输出共享 | 输入输出独立 |
| :--- | :--- | :--- |
| LLM 大模型 | ✅ (tied) | ❌ (显存不够) |
| LLM 小模型 | ✅ | 可选 |
| GenRec SID | ❌(多级必须独立) | ✅ |
| GenRec 物品 ID | ✅(与原始 embedding 共享) | ❌ |

---

## 8. 共享 Embedding 对比

"共享 Embedding"在两边含义完全不同:

| 概念 | LLM | GenRec |
| :--- | :--- | :--- |
| 共享对象 | 输入输出 tied | 多任务 / 多场景 / 多模态 |
| 共享粒度 | 全部 token 共享 1 张表 | 跨业务 / 跨域 |
| 收益 | 节省参数 | 解决冷启动 + 数据稀疏 |
| 风险 | 表达受限 | 负迁移 |

### 8.1 LLM 的 Tied Embedding

```python
# Tied
model.embed_tokens.weight = model.lm_head.weight
```

> 经典做法,GPT-2/3/4, Llama 1/2/3 全部 tied。

### 8.2 GenRec 的跨场景共享

#### 8.2.1 美团 MTGR 的多业务共享

```text
外卖业务: user_emb + item_emb
闪购业务: user_emb + item_emb (部分共享)
酒旅业务: user_emb + item_emb (独立)

共享部分: user tower (跨业务复用)
独立部分: item tower + 业务特定 head
```

#### 8.2.2 阿里 OneRec 的跨域共享

```text
淘宝域: 100 亿 token
天猫域: 20 亿 token
1688 域: 5 亿 token

共享: 第一级 SID(类目码本)
独立: 第二三级 SID(域内聚类)
```

### 8.3 共享 Embedding 的优劣对比

| 维度 | LLM Tied | GenRec 跨场景共享 |
| :--- | :--- | :--- |
| 参数节省 | 50% | 30 – 70% |
| 训练速度 | 快(显存省) | 快 |
| 效果损失 | 无(经验上 tied 持平或略优) | 5 – 15% |
| 负迁移风险 | 极低 | 中(域冲突时) |
| 调参成本 | 低 | 中(需平衡各域权重) |

---

## 9. 数学对比

这一节给出**形式化的对比**。读者可跳过公式直接看 9.4 总结表。

### 9.1 LLM Tokenization 的数学

#### 9.1.1 BPE 形式化

设语料 $\mathcal{D} = \{d_1, \dots, d_N\}$, 字符集 $\Sigma_0 = \{$ 256 bytes $\}$。

初始化:  对每个 $d_i$, 拆为字节序列 $b_1^{(i)}, \dots, b_{L_i}^{(i)}$。

迭代 $t = 1, \dots, V - |\Sigma_0|$:

$$
(x_t, y_t) = \arg\max_{(x,y) \in \mathcal{P}_t} \sum_{i=1}^N \mathbb{1}\{ (x,y) \in \text{adj}(d_i^{(t)}) \}
$$

其中 $\text{adj}(\cdot)$ 返回相邻 token 对集合。

$$
\Sigma_t = \Sigma_{t-1} \cup \{ z_t \}, \quad z_t = x_t \oplus y_t
$$

> $V$ 是目标词表大小,通常 32K – 256K。

#### 9.1.2 WordPiece 形式化

定义 unigram log-likelihood:

$$
\mathcal{L}(\Sigma_t) = \sum_{i=1}^N \log P(d_i | \Sigma_t)
$$

其中 $P(d_i | \Sigma_t)$ 是把 $d_i$ 切分为 $\Sigma_t$ 中 token 的所有可能方式之和(Viterbi)。

合并准则:

$$
(x_t, y_t) = \arg\max_{(x,y)} \left[ \mathcal{L}(\Sigma_{t-1} \cup \{x \oplus y\}) - \mathcal{L}(\Sigma_{t-1}) \right]
$$

#### 9.1.3 Unigram LM 形式化

Unigram 用 EM 训练一个概率 $P(x)$ over 词表:

$$
P(x) = \frac{|x|^{-\alpha}}{\sum_{y \in \Sigma} |y|^{-\alpha}}
$$

每次删除"贡献最小"的 token:

$$
\text{loss}_t = -\sum_{i=1}^N \log P(d_i | \Sigma_t \setminus \{x_t\}), \quad
x_t = \arg\min_{x \in \Sigma_t} \text{loss}_t
$$

### 9.2 GenRec SID 的数学

#### 9.2.1 RQ-VAE 目标函数

$$
\mathcal{L}_{\text{RQ-VAE}} = \underbrace{\| \mathbf{x} - D(\hat{\mathbf{z}}) \|_2^2}_{\text{reconstruction}} + \underbrace{\beta \sum_k \| \text{sg}[\mathbf{r}_{k-1}] - \mathbf{e}_k^{(c_k)} \|_2^2}_{\text{codebook}} + \underbrace{\| \mathbf{r}_{k-1} - \text{sg}[\mathbf{e}_k^{(c_k)}] \|_2^2}_{\text{commitment}}
$$

#### 9.2.2 RQ-KMeans 目标函数

$$
\mathcal{L}_{\text{RQ-KMeans}} = \sum_{i=1}^N \sum_{k=1}^L \| \mathbf{r}_{k-1,i} - \mu_{k,c_{k,i}} \|_2^2
$$

最小化总重建误差。

#### 9.2.3 Encoder 输出维度 vs SID 表达力

设 Encoder 输出 $d$ 维, 每层码本大小 $K$, 层次数 $L$:

| 指标 | 公式 | 典型值 |
| :--- | :--- | :---: |
| 词表组合容量 | $K^L$ | $10^9$ (1000³) |
| 单物品信息量 | $L \cdot \log_2 K$ bit | 30 bit (3 × 10 bit) |
| 信息上界(连续) | $d \cdot \log_2 e$ / sqr(N) | — |

> **关键不变量**: $L \cdot \log_2 K$ 应小于 $d \cdot 8$(连续向量的有效 bit),否则过度离散化。

### 9.3 训练时计算的对比

| 阶段 | LLM 单 token | GenRec 单物品 |
| :--- | :--- | :--- |
| Embedding lookup | $O(d)$ | $O(L \cdot d)$ (多级) |
| Self-Attention | $O(N \cdot d)$ per token | $O(M \cdot d)$ per item |
| Output softmax | $O(V \cdot d)$ | $O(K \cdot d)$ (每级独立) |
| 总计(per step) | $O(N + V) \cdot d$ | $O(M + L \cdot K) \cdot d$ |

> **关键观察**: LLM 的 softmax over V=200K 与 GenRec 的 L×K = 3×1000 规模相当,这是 GenRec 能用 NTP 的算力基础。

### 9.4 形式化对比总结表

| 维度 | LLM Tokenization | GenRec SID Tokenization |
| :--- | :--- | :--- |
| 输入 | 文本字节流 | 物品特征 / 图 |
| 输出 | 1D 序列 $T_1, \dots, T_n$ | 1D 序列 $C_1, \dots, C_L$ |
| 编码函数 | $f: \text{bytes} \to \text{token}$,查表 | $f: \text{item} \to \text{SID}$,Encoder + 量化 |
| 词表构造 | $\arg\max$ 频次 / likelihood | $\arg\min$ 重建误差 |
| 优化目标 | 不直接优化(独立步骤) | 重建 + commitment |
| 与下游耦合 | 解耦 | 强耦合 |
| 参数量(embed) | $V \cdot d$ | $L \cdot K \cdot d$ |
| 参数量(主模型) | 远大于 embed | 相当或更大 |
| 推理时延/token | $O(d)$ | $O(L \cdot d)$ |

---

## 10. 实际案例对比

### 10.1 案例 A: GPT-4o vs Qwen2.5 中文压缩率

#### 10.1.1 测试用例

```text
输入:  "今天天气不错,我们去公园散步。我买了一杯咖啡,花了 35 元。"
```

#### 10.1.2 编码结果

| 模型 | 词表 | token 数 | 字符/token |
| :--- | :--- | :---: | :---: |
| GPT-3.5 | cl100k_base | 51 | 1.6 |
| GPT-4 | cl100k_base | 48 | 1.7 |
| GPT-4o | o200k_base | 28 | 2.9 |
| Llama 3 | 128K | 35 | 2.3 |
| Qwen2.5 | 152K | 19 | 4.2 |

> Qwen2.5 的中文 token 化最紧凑,直接影响训练与推理成本。

### 10.2 案例 B: OneRec V2 vs V1 SID 设计

#### 10.2.1 数据规模

```text
物品库: 10 亿 SKU
行为日志: 千亿样本
用户数: 10 亿
```

#### 10.2.2 V1 vs V2 对比

| 维度 | V1(直接 ID) | V2(RQ-KMeans SID) |
| :--- | :--- | :--- |
| 词表大小 | 5,000,000 | 1,000,000,000(组合) |
| 实际 token 数 | 1 (单 token) | 3 (3 级) |
| 单 token 维度 | 128 | 128 / 级 |
| embedding 参数量 | 640 MB | 480 KB(每级 1000×128) |
| 长尾物品效果 | 差 | 好 |
| 推理延迟 | 38 ms | 22 ms |
| 冷启动 | 难 | 易 |

#### 10.2.3 V2 的关键工程

- **第一级固定为类目**:跨域共享,语义稳定
- **后两级 KMeans 增量更新**:每 7 天重聚类
- **TRIE 约束解码**:续写 SID 时只允许同前缀物品

### 10.3 案例 C: TIGER RQ-VAE 训练日志

#### 10.3.1 Amazon 数据集

| 维度 | 数值 |
| :--- | :--- |
| 物品数 | 9.7M(Books), 1.4M(Sports) |
| 行为数 | 22M(Books) |
| Encoder | Sentence-T5-base |
| 词表大小 | 256 / 级 |
| 层次数 | 3 |
| 训练 epoch | 50 |

#### 10.3.2 关键指标

| 指标 | Recall@5 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: |
| BERT 双塔 | 0.182 | 0.241 | 0.165 |
| SASRec | 0.214 | 0.279 | 0.193 |
| TIGER | 0.243 | 0.317 | 0.219 |

### 10.4 案例 D: 美团 MTGR 多业务 SID

#### 10.4.1 三业务场景

```text
外卖: 商家 200 万
闪购: 商品 500 万
酒旅: 酒店 80 万
```

#### 10.4.2 SID 设计

| 业务 | 词表大小 | 层次 | 共享层 |
| :--- | :--- | :---: | :--- |
| 外卖 | 1K × 1K | 2 | 第一级 |
| 闪购 | 1K × 1K × 1K | 3 | 第一、二级 |
| 酒旅 | 1K × 1K | 2 | 第一级 |

> 第一级共享类目,跨业务可迁移"美食 → 酒店"。

### 10.5 案例 E: 快手短视频 SID

#### 10.5.1 特点

- **多模态**:封面图 + 标题 + 标签 + 音频指纹
- **Encoder**:CLIP ViT-L/14 + Qwen2.5-7B 拼接
- **SID**:4 级(2K × 1K × 1K × 1K)
- **效果**:Hit@50 比 SASRec 提升 25%

### 10.6 案例 F: 抖音电商直接 ID 路线

#### 10.6.1 设计

- **物品 ID 直接做 token**(2 亿 SKU)
- **辅助 BPR 损失**解决长尾
- **embedding 维度**:96(刻意小,减少内存)
- **长尾物品 hit@10**:0.18(略低,但可接受)

#### 10.6.2 经验

> 直接 ID 路线在小规模(百万级)有效;超过千万级必须 SID。

---

## 11. 优劣对比

### 11.1 LLM Tokenization 优劣

#### 11.1.1 优势

1. **100% 字符覆盖**:Byte 回退天然无 OOV
2. **算法简单**:BPE 仅需频次统计
3. **可离线一次性构造**:与训练解耦
4. **多语言均衡**:256 字节基础 + 合并策略
5. **生态成熟**:tiktoken / tokenizers / SentencePiece 工业级

#### 11.1.2 劣势

1. **无语义结构**:完全靠频次
2. **与下游任务脱节**:换任务可能要重训词表
3. **词表固定**:加新概念只能"碰运气"被 subword 拼出来
4. **多模态支持弱**:图像 / 音频需要独立 tokenizer

### 11.2 GenRec SID 优劣

#### 11.2.1 优势

1. **语义结构**:相近物品共享前缀
2. **跨域迁移**:第一级可共享类目
3. **可生成**:Transformer 续写 SID
4. **冷启动友好**:KMeans 即时编码新物品
5. **表达力强**:L 级 SID 几何级容量

#### 11.2.2 劣势

1. **Encoder 训练成本**:VAE / 双塔要先训
2. **重聚类开销**:KMeans 几小时,Encoder 更新则全重做
3. **死码本**:KMeans 簇大小不均,部分码本利用率低
4. **冷启动依赖内容**:无内容则无 SID
5. **debug 难**:物品 → SID 不直观

### 11.3 横向对比矩阵

| 维度 | LLM Token | GenRec SID | OneRec 直接 ID |
| :--- | :---: | :---: | :---: |
| 实现复杂度 | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| 词表大小 | 32K – 256K | 4K – 10M(组合) | 1M – 100M |
| 多语言/多域 | ✅ | ✅(类目共享) | ❌ |
| 冷启动 | N/A | ⭐⭐⭐ | ⭐⭐ |
| 长尾表现 | N/A | ⭐⭐⭐ | ⭐ |
| 推理速度 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 工业成熟度 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 调试友好度 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |

---

## 12. 未来融合

### 12.1 三大融合方向

#### 12.1.1 LLM 词表 + GenRec SID 混合

**核心思想**:在用户行为序列中,**物品 SID 与文本描述共存**。

```text
用户历史:
  [item_123 (SID: 5,12,89)] [book title: "三体"] [item_456 (SID: 5,12,92)] ...
        ↑              ↑                ↑                  ↑
   物品 token    文本 token      物品 token         文本 token
```

- **LLaMA-GenRec**(Meta 2024) 验证了这种范式
- **OneRec V3**(2025 内部) 进一步把 LLM tokenizer 纳入预训练

#### 12.1.2 多模态统一 Tokenizer

```text
图像 → VQ-GAN 切成 visual token
文本 → SentencePiece 切 subword
物品 → RQ-KMeans 切 SID
行为 → ID 直接
       ↓
统一 Transformer(类 GPT-4o 多模态)
```

- **GPT-4o**、**Gemini** 已部分实现
- **GenRec 多模态** 是 2025 主流方向

#### 12.1.3 动态词表 (Dynamic Vocabulary)

```text
请求到来 → 根据用户上下文(国家 / 设备 / 时段)动态加载词表
```

- 节省推理内存
- 多区域部署

### 12.2 技术突破点

| 突破点 | 当前状态 | 期望 |
| :--- | :--- | :--- |
| 死码本消除 | KMeans 后剪枝 | 平衡码本 |
| 跨模态统一 | 各家自研 | 通用范式 |
| 在线 KMeans | 无 | 几小时级 |
| 冷启动 0 延迟 | 几秒 → 分钟 | 毫秒级 |
| 词表自适应 | 固定 | 动态 |

### 12.3 与本系列其他章节的衔接

```text
[[4-Token化与词表对比]] (本章)
   ↓
`5-训练目标与损失对比` ← 不同词表决定不同的损失设计
   ↓
[[6-评估指标对比]]     ← 词表大小影响召回指标
   ↓
[[7-RLHF与RL对齐对比]] ← RL 阶段会重新影响 embedding
```

### 12.4 总结

LLM 与 GenRec 的 tokenization 是**两套独立演化的技术栈**:

- LLM 在字符 / 字节层面做扁平离散化
- GenRec 在语义 / 协同层面做层次离散化

融合方向:

1. **混合词表**(SID + 文本)
2. **多模态统一**(图像 + 文本 + 行为)
3. **动态词表**(按上下文切换)

> **未来 5 年的关键趋势**:Tokenization 不再是"训前一次性"的预处理,而是与训练目标、推理服务深度耦合的**动态系统**。

---

## 13. 深度专题(补充阅读)

> 这一节是补充阅读区,把正文里有意控制体量的几个高级话题展开。每个小节独立可读。

### 13.1 Tokenizer 对模型"智商"的隐性影响

LLM 圈有句老话:**"Tokenizer 决定模型智商的 30%"**。下面用具体例子说明。

#### 13.1.1 反直觉:同样字符数,tokens 越多模型越"笨"

| 输入文本 | 字符数 | tokens | 实际训练可见样本数 |
| :--- | :---: | :---: | :---: |
| "今天天气不错" | 6 | Qwen: 4 / Llama3: 6 / GPT-4o: 5 | 短序列=同样窗口可装更多样本 |
| "Today is nice" | 13 | GPT-4o: 3 | — |

直观上"长 token = 短序列 = 同样窗口装更多训练信号"。但**字符级 vs 子词级**有更深的影响:

- **字符级模型**(每个字符 1 token):序列长,Self-Attention $O(N^2)$ 压力大,但**任何字符都能学到**;
- **子词级模型**:序列短,Attention 快,但**罕见词被压成单 token,无法细分**;
- **混合级模型**(字节 + 子词):GPT-2/3/4 的 Byte-Level BPE 是这个思路的极简实现。

#### 13.1.2 Tokenizer bias 的实证

下面这个例子在 LLM 圈很出名:

```text
问: "How many r's are there in 'strawberry'?"
GPT-3.5: 答错 (tokenize 成 1 个 token,模型看不到字符级结构)
GPT-4:   答对 (用了字符级 fallback)
Llama 3: 答对 (词表含更多字符级片段)
```

> 启示:Tokenizer 的"颗粒度"直接影响模型做字符级推理的能力。GenRec 的 SID 是粗粒度,但**类目级别天然捕捉了"层级"**,反而对"长尾物品属于哪个类目"这种推理有优势。

### 13.2 多语种 + 罕见字符:Byte 回退 vs 字符回退

#### 13.2.1 三种"基础字符集"选择

| 方案 | 基础集 | 代表 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- | :--- |
| Unicode 字符 | ~150K 字符集 | XLM-R | 覆盖广 | 字符级太多 OOV |
| UTF-8 字节 | 256 bytes | GPT-2+ | 100% 覆盖 | 高频字符被拆碎 |
| BPE in chars | 几万个字符 | mBERT | 平衡 | 罕见字符仍 OOV |

#### 13.2.2 Qwen 的折中

Qwen2 用 151,851 词表,基础集是 ~8000 个常用字符 + 256 bytes 作为兜底。这样:

- 高频汉字 / 词:整 token
- 中频字符:字符级 token
- 罕见字符:字节级 token

#### 13.2.3 与 GenRec 物品编码的类比

```text
LLM 字符集:
  - 高频 → 词级
  - 中频 → 字符级
  - 罕见 → 字节级

GenRec 物品编码:
  - 高频 → SID 前缀完整(头物品)
  - 中频 → SID 1-2 级(中物品)
  - 罕见 → 拆到 SID 第 3 级(长尾)
```

> 二者都遵循**"频率自适应"**原则:出现越多,编码越短;出现越少,编码越碎。

### 13.3 Tokenizer 与训练目标的耦合关系

#### 13.3.1 LLM:解耦的训练-Tokenizer

LLM 的训练 pipeline 典型为:

```text
Step 1: 训练 tokenizer (BPE/SentencePiece,几分钟到几小时)
Step 2: 用 tokenizer 把所有文本切成 token IDs
Step 3: 训练 LM 主模型(几天到几周)
```

> **优势**:换 tokenizer 时,主模型需要重训,但**训练数据可复用**(只需重切 token ID)。

#### 13.3.2 GenRec:耦合的训练-SID

GenRec 的训练 pipeline:

```text
Step 1: 训练物品 Encoder (BERT/Qwen/CLIP,几天)
Step 2: 跑 Encoder 得到物品 embeddings (几小时)
Step 3: KMeans / RQ-VAE 得到 SID (几小时)
Step 4: 训练生成式主模型 (几天到几周)
Step 5: 在线推理时,新物品 → 重新走 Step 1-3 的子集
```

> **劣势**:任一环节变化都需要重新生成 SID,**长尾物品若重训可能 SID 全部变化**,下游 cache 失效。

#### 13.3.3 解决"重训失效"的工程方案

| 方案 | 做法 | 代价 |
| :--- | :--- | :--- |
| 增量 KMeans | 只更新变化物品的簇 | 仍然有边缘抖动 |
| 固定第一级 | 类目码本不变,只更新后两级 | 牺牲少量精度 |
| 重映射表 | 维护 old → new SID 映射 | 额外存储 |
| 双轨运行 | 新旧 SID 同时在,逐步切换 | 双倍推理成本 |

### 13.4 罕见词 / 罕见物品的"几何直觉"

#### 13.4.1 LLM 罕见词的几何

LLM 训练完后,embedding 空间呈现:

```text
                "king" - "man" + "woman" ≈ "queen"
       ↓                    ↓                ↓
    常见词向量          常见词向量          常见词向量
    (dense, well-trained)

    "wikipedian_journalism_buzzword_xyz"  → 随机初始化位置,只更新几次
    (sparse, undertrained)
```

罕见词的 embedding 没有足够梯度更新,接近随机。

#### 13.4.2 GenRec 长尾物品的几何

```text
    头部物品 e_1 ─┐
                 ├─→ SID: (3, 27, 89)
    头部物品 e_2 ─┘                ↓
                              残差 = e_1 + e_2 - μ_{3,27}

    长尾物品 e_tail ─→ 直接 KMeans,落入某个簇
                              ↓
                              SID: (3, 27, 89)  ← 与头部相同!
```

> 长尾物品的 SID **前缀与头部相同**(因为 KMeans 强制聚类),所以**共享梯度**,效果好。

#### 13.4.3 对比表

| 维度 | LLM 罕见词 | GenRec 长尾物品 |
| :--- | :--- | :--- |
| embedding 来源 | 训练后学习 | Encoder + KMeans 强制聚类 |
| 几何位置 | 随机(欠训练) | 与头部共享簇(几何上靠近) |
| 推理时表现 | 弱 | 中等(共享梯度) |
| 改进方法 | 扩大词表 / byte 回退 | 多级 SID / 共享类目 |

### 13.5 混合词表:MixedTokenizer(2025 趋势)

#### 13.5.1 设计

```text
统一词表 = {
    256 bytes,
    Qwen 子词 151K,
    RQ-KMeans SID 3 × 1000,
    物品直接 ID 5M
}
```

> **核心思想**:不同来源的 token 共享同一个 Transformer 的 embedding 表。

#### 13.5.2 训练 pipeline

```python
# 输入可能是文本 + 行为序列的混合
input_ids = [
    <bos>, 100, 200, <sep>,           # 文本部分
    1000013, 1000042, 1000089,        # 物品 SID 部分(3 级拼成 1 个 ID)
    <sep>, 300, 400, <eos>
]
```

#### 13.5.3 OneRec V3 内部设计

```text
文本前缀: "[INST] 推荐适合 [历史] 的商品 [/INST]"
   ↓
混合 token 序列:
   [<bos>, 用户特征文本, 历史 SID 序列, <sep>, 推荐 SID, <eos>]
   ↓
Transformer 续写 → 推荐 SID → TRIE → 物品 ID
```

> 这是 2025 生成式推荐 SOTA 范式之一。

### 13.6 Tokenizer 评测基准

#### 13.6.1 LLM 评测

| 指标 | 定义 | 典型 |
| :--- | :--- | :---: |
| Fertility | tokens / words | 1.3 – 1.5 (英文) |
| Compression | bytes / token | 4 – 6 (英文) |
| Multilingual Ratio | 高低资源比 | 1.0 – 2.0 |
| Token Coverage | 字符集覆盖 | 100% |
| Vocab Efficiency | 实际使用比例 | 80 – 95% |

#### 13.6.2 GenRec 评测

| 指标 | 定义 | 典型 |
| :--- | :--- | :---: |
| Codebook Utilization | 非空码本占比 | 70 – 95% |
| Reconstruction Error | VAE 重建误差 | 0.05 – 0.2 |
| Hit Rate | SID 重建物品准确率 | 95 – 99% |
| Collision Rate | 不同物品同一 SID | < 5% |
| Cluster Balance | 簇大小 std/mean | 0.5 – 1.5 |

#### 13.6.3 评测脚本

```python
# 伪代码
def evaluate_tokenizer(tokenizer, test_set):
    total_tokens = 0
    total_bytes = 0
    for text in test_set:
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_bytes += len(text.encode('utf-8'))
    return {
        'fertility': total_tokens / len(test_set),
        'compression': total_bytes / total_tokens,
    }
```

### 13.7 工程化陷阱

#### 13.7.1 LLM Tokenizer 陷阱

| 陷阱 | 现象 | 解决 |
| :--- | :--- | :--- |
| 不同词表的模型合并 | embedding 维度对不上 | 统一词表 / 投影层 |
| 数字未独立 token | "123456" 占多 token | 强制每数字 1 token |
| 多语言不一致 | 中文被拆碎 | 扩词表 + 字节回退 |
| Special token 冲突 | `<\\|im_end\\|>` 与训练不同 | 严格对齐 |

#### 13.7.2 GenRec SID 陷阱

| 陷阱 | 现象 | 解决 |
| :--- | :--- | :--- |
| 死码本(簇为空) | 长尾物品 SID 缺失 | KMeans 后剪枝 / 重聚类 |
| Encoder 漂移 | SID 含义变化 | 定期重聚类 + 监控 |
| 新物品编码延迟 | 上线时无法立刻推荐 | 预生成 + 离线 KMeans |
| 跨域负迁移 | 共享层冲突 | 业务独立 + 共享类目 |
| 重建误差大 | SID 失去语义 | 增大 Encoder / 增加层次 |

### 13.8 词表 vs 序列长度的"trade-off"

#### 13.8.1 LLM 的核心 trade-off

```text
大词表 (V 大) + 短序列 (N 小):
  + embedding 表大,训练慢
  + softmax 计算重
  - 序列短,推理快
  - 罕见词直接被 token 化

小词表 (V 小) + 长序列 (N 大):
  + embedding 表小
  + softmax 快
  - 序列长,Attention 慢(O(N²))
  - 罕见词拆碎
```

**典型 LLM 选 V=200K, N=8K-128K(单样本)**,取折中。

#### 13.8.2 GenRec 的核心 trade-off

```text
单级大词表 (N 物品数) + 短序列 (1 token / 物品):
  + 直接,简单
  + 无需重建
  - 长尾差
  - embedding 表大(5M × 128 = 640MB)

多级小词表 (L 级 K 大小) + 长序列 (L token / 物品):
  + 长尾好
  + embedding 表小(3 × 1K × 128 = 384KB)
  - 推理时要续写 L 个 token
  - 需要 Encoder + 量化 pipeline
```

**典型 GenRec 选 L=3, K=1K**,即 3 级 SID。

#### 13.8.3 二者对比

| 维度 | LLM trade-off | GenRec trade-off |
| :--- | :--- | :--- |
| 词表大小 | 大(200K) | 中(1K) |
| 序列长度 | 长(8K) | 中(100) |
| 单 token 信息量 | 高(一个 token 1.5 字符) | 中(一个 token 100 万物品) |
| 信息密度 | 字符 / token | 物品 / token |
| 优化方向 | 加大 N(长上下文) | 加大 L(更深 SID) |

### 13.9 总结:Tokenization 是模型"看见世界"的方式

> **LLM 让模型看见"字符流";GenRec 让模型看见"物品流"**。两者用不同的离散化方式,把连续的、不可数的人类世界映射到有限的、可数的离散空间。

未来 5 年的趋势:

1. **融合**:LLM 与 GenRec 的词表边界模糊化
2. **统一**:多模态统一 Tokenizer
3. **动态**:在线学习词表

---

## 附录 A · 速查表

### A.1 主流 Tokenizer 一览

| Tokenizer | 算法 | 词表 | 代表模型 | 关键特性 |
| :--- | :--- | :---: | :--- | :--- |
| BPE (byte-level) | BPE | 50K | GPT-2/3 | Byte 回退 |
| WordPiece | WordPiece | 30K | BERT | likelihood 准则 |
| SentencePiece | BPE / Unigram | 32K | Llama 1/2 | 语言无关 |
| tiktoken | BPE | 200K | GPT-4o, o1 | 高性能 |
| Qwen tokenizer | BPE | 152K | Qwen2/3 | 中文优化 |
| Llama 3 tokenizer | tiktoken | 128K | Llama 3 | 多语言 |
| RQ-VAE (TIGER) | RQ-VAE | L×256 | TIGER | 内容驱动 |
| RQ-KMeans (OneRec V2) | KMeans | L×1K | OneRec V2 | 协同驱动 |

### A.2 LLM vs GenRec Tokenization 速查

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| 输入域 | 文本字节流 | 物品原始特征 / 行为 |
| 编码函数 | 查表 | Encoder + 量化 |
| 词表大小 | 32K – 256K | 1M(直接 ID)/ 10M+(SID 组合) |
| OOV | 无(byte 回退) | 有(必须重编码) |
| 冷启动 | 即时 | 分钟 ~ 天 |
| 训练耦合度 | 解耦 | 强耦合 |
| 多语言/多域 | 字节统一 | 类目共享 |
| 工业实践 | 成熟 | 快速演进 |

### A.3 关键公式速查

```text
BPE:   (x, y) = argmax freq(x, y)
       Σ_{t+1} = Σ_t ∪ {x ⊕ y}

WordPiece: (x, y) = argmax [L(Σ_{t-1} ∪ {x⊕y}) - L(Σ_{t-1})]

RQ-VAE: c_k = argmin ||r_{k-1} - e_k^{(j)}||²
        r_k = r_{k-1} - e_k^{(c_k)}

RQ-KMeans: μ_{l,j} = mean(r_{l-1,i}) for i ∈ C_{l,j}
           c_{l,i} = argmin ||r_{l-1,i} - μ_{l,j}||²
           r_{l,i} = r_{l-1,i} - μ_{l,c_{l,i}}
```

### A.4 容量计算速查

```text
LLM 直接 ID:
  vocab_size = V
  组合容量 = V
  参数 = V × d

LLM 多级(BPE 不分层,仅作示意):
  vocab_size = V (单层)
  参数 = V × d

GenRec 直接 ID:
  vocab_size = N (物品数)
  参数 = N × d

GenRec SID 多级:
  vocab_size = L × K
  组合容量 = K^L
  参数 = L × K × d
```

### A.5 实战选型速查

| 场景 | 推荐方案 |
| :--- | :--- |
| 通用文本 LLM | tiktoken / SentencePiece BPE |
| 多语言 LLM | tiktoken o200k_base / Llama 3 128K |
| 中文为主 LLM | Qwen tokenizer (152K) |
| 小规模推荐(百万级) | 直接物品 ID |
| 中规模推荐(千万级) | 3 级 SID (1K³) |
| 大规模推荐(亿级) | 4 级 SID + 类目共享 |
| 冷启动密集 | RQ-KMeans + 多模态 Encoder |
| 多业务融合 | 第一级类目共享,后几级独立 |

---

## 附录 B · 参考与链接

### B.1 内部链接(同模块)

- [[0-总览与核心区别]] - LLM-vs-GenRec 主目录
- [[1-目标与定义对比]] - 总览
- [[3-数据与训练对比]] - 已存在的数据层对比
- `5-训练目标与损失对比` - 训练目标
- [[6-评估指标对比]] - 评估
- [[7-RLHF与RL对齐对比]] - RLHF

### B.2 RL 模块(11 份)

- [[0-总览与核心区别]] - RL 模块入口
- `../RL/1-RL基础与数学.md` - RL 数学
- `../RL/2-经典RL算法详解.md` - 经典算法
- `../RL/3-RL在LLM中的经典方法.md` - LLM RL 经典
- `../RL/4-RL在GenRec中的经典方法.md` - GenRec RL 经典
- `../RL/5-奖励设计与RL对齐.md` - 奖励设计
- `../RL/6-RL工程实践与部署.md` - 工程实践
- `../RL/7-论文案例-OneRec系列RL.md` - OneRec 案例
- `../RL/8-论文案例-其他GenRec RL.md` - 其他案例
- `../RL/9-RL未来方向与开放问题.md` - 未来方向
- `../RL/补充RL论文清单.md` - 论文清单

### B.3 GR deep-dives (24 篇)

> **GR 第09章 deep-dives 系列**(规模化排序与新架构):

- `../GR/第09章-规模化排序与新架构/deep-dives/01-TokenMixer-Large-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/02-OneTrans-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/03-STCA-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/04-MDL-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/05-MixFormer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/06-MSN-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/07-HyFormer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/08-STORE-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/09-FAT-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/10-HHFT-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/11-EST-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/12-SORT-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/13-UniScale-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/14-MGDIN-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/15-HeMix-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/16-CDNet-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/17-LFM4Ads-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/18-SMES-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/19-LASER-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/20-OxygenREC-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/21-InterFormer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/22-Kunlun-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/23-HoMer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/24-MTmixAtt-AutoToken-深度解析.md`

### B.4 GR deep-dives-extended (10 篇)

- `../GR/第09章-规模化排序与新架构/deep-dives-extended/25-PLUM-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/26-GRAB-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/27-PinRec-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/28-NetflixGenRec-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/29-COBRA-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/30-LiGR-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/31-STATIC-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/32-GRAM-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/33-DeepInterestGR-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/34-SelfEvolvingRec-深度解析.md`

### B.5 外部参考

- Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.
- Schuster & Nakajima (2012). Japanese and Korean Voice Search. ICASSP. (WordPiece)
- Kudo (2018). Subword Regularization. ACL. (SentencePiece)
- van den Oord et al. (2017). Neural Discrete Representation Learning. NIPS. (VQ-VAE)
- Razavi et al. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. NeurIPS.
- Rajput et al. (2023). Recommender Systems with Generative Retrieval. (TIGER)
- Kang & McAuley (2018). Self-Attentive Sequential Recommendation. (SASRec)
- OneRec 团队 (2025). OneRec V1/V2.

---

> [!summary] **关键 takeaway**:
> 1. LLM Tokenization 是"统计字符合并"的扁平离散化;GenRec SID 是"语义聚类 + 量化"的层次离散化。
> 2. 词表大小:LLM 32K–256K;GenRec 1M(直接 ID)到 10M+(SID 组合)。
> 3. OOV:LLM 字节回退天然无 OOV;GenRec 必须重新编码。
> 4. 冷启动:LLM 即时;GenRec 分钟级。
> 5. 未来融合方向:多模态统一 Tokenizer、动态词表、SID+LLM 混合词表。

---

## 关联阅读

1. [[3-数据与训练对比]] - 数据与训练对比
2. `5-训练目标与损失对比` - 训练目标与损失对比
3. [[13-多模态与跨模态对比]] - 多模态场景
4. `../../基础/数学基础/第03章-概率论/0-总目录与阅读路径.md` - RQ-VAE 详解

> **下一篇**: `5-训练目标与损失对比` — 从"输入表征"走到"训练目标",看 LLM 与 GenRec 在损失函数、负采样、辅助任务上的根本差异。
