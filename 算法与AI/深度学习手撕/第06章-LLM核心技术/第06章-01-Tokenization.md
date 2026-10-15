---
title: "第06章-01-Tokenization"
tags:
  - 深度学习
  - Tokenization
description: "本章要点：chapterid: 06.01"
chapter_id: 深度学习手撕-06.01
last_updated: 2026-07-14
---

# 第06章-01-Tokenization

LM 第一步
chapter_id: 06.01
last_updated: 2026-07-14
status: ✅ v1.0

---

## 6.1 Tokenization (BPE / BBPE / WordPiece / SentencePiece)

---

## 1. 内容介绍

**Tokenization** 把文本切成模型能理解的"基本单元" (token). 是 LLM 的**第一步**.

LLM 时代, 主流是 **BBPE** (Byte-level BPE), GPT/LLaMA/Qwen 都用.

---

## 2. 为什么这么做

**问题**:
- 直接用字符: 序列长, 表达力差
- 直接用词: 词表大 (50万+), OOV 问题
- 子词 (Subword): 平衡词表大小和序列长度

---

## 3. 这么做的原因 (方案对比)

| 方案 | 描述 | 词表大小 | OOV | 代表 |
|---|---|---|---|---|
| 字符 | 每个字符一个 token | 26 + 标点 | 无 | 早期 NLP |
| 词 | 每个词一个 token | 50万+ | 有 | 传统 ML |
| BPE | 频次合并字符 | 1-5万 | 无 | GPT-2 |
| WordPiece | 概率合并 | 3万 | 无 | BERT |
| **BBPE** | 字节级 BPE | 256 字符基础 | **全无** | **GPT-4, LLaMA** |
| SentencePiece | 语言无关 (BPE/Unigram) | 灵活 | 无 | LLaMA, T5 |

**为什么 LLM 都用 BBPE?**
- 以 **字节 (256)** 为基础, 任何 UTF-8 字符都能编码
- 无 OOV (Out-of-Vocabulary)
- 词表小 (32K-128K) 但语义丰富

---

## 4. 解决了什么问题

- **OOV 问题**: 任何字符都能表示
- **多语言**: 同一 tokenizer 处理 100+ 语言
- **词表 vs 序列长度平衡**: 32K 词表, 平均每词 ~4 字符

---

## 5. 意义

- LLM 第一步, 词表选择决定模型上限
- 主流: BBPE (OpenAI, Meta, Mistral) / SentencePiece (Google, LLaMA)

---

## 6. 公式表达

```text
BPE 训练:
  1. 初始化: 每个词拆成字符
  2. 循环 K 次:
     a. 统计相邻字符对频次
     b. 合并频次最高的字符对
     c. 加入词表
  3. 输出: 词表 + 合并规则

BBPE 训练:
  同 BPE, 但以字节 (256) 为基础
  不需要预分词
```

---

## 7. 题目设计

**为什么这么考**: Tokenization 是 LLM 面试必问:
- BPE 训练流程
- BBPE 与 BPE 区别
- 实现简单的 BPE
- 词表大小 vs 序列长度

**面试陷阱**: 不理解 BBPE 怎么"无 OOV"; 不理解训练 vs 推理区别.

---

## 8. 题目

### 题目 1: 实现 BPE 训练

```python
def bpe_train(corpus, vocab_size):
    """
    输入: 语料 (list of strings), 词表大小
    输出: 词表 (dict: token -> id), 合并规则
    """
    pass
```

### 题目 2: 实现 BPE 编码/解码

```python
def bpe_encode(text, vocab, merges):
    """把文本编码为 token id 序列"""
    pass

def bpe_decode(token_ids, vocab):
    """把 token id 序列还原为文本"""
    pass
```

### 题目 3: 实现简易 BBPE 训练

```python
def bbpe_train(text_bytes, vocab_size):
    """
    字节级 BPE 训练
    text_bytes: 文本的字节序列
    返回: 词表 + 合并规则
    """
    pass
```

---

## 9. 答案

### 答案 1: BPE 训练

```python
from collections import defaultdict, Counter

def get_pairs(word_ids):
    """统计相邻字符对"""
    pairs = Counter()
    for word in word_ids:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def bpe_train(corpus, vocab_size):
    """
    BPE 训练
    """
    # 1. 词频统计
    word_freqs = Counter(corpus)
    # 2. 初始化: 每个词拆成字符 (加 </w> 标记词尾)
    vocab = {'<unk>': 0, '<pad>': 1, '</w>': 2}
    word_ids = []
    for word in word_freqs:
        chars = list(word) + ['</w>']
        word_ids.append(chars)
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)

    merges = []  # (a, b) → c

    # 3. 循环合并
    while len(vocab) < vocab_size:
        pairs = get_pairs(word_ids)
        if not pairs:
            break
        # 找最频繁对
        best = max(pairs, key=pairs.get)
        # 合并
        new_token = ''.join(best)
        vocab[new_token] = len(vocab)
        merges.append(best)
        # 更新 word_ids
        new_word_ids = []
        for word in word_ids:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_ids.append(new_word)
        word_ids = new_word_ids

    return vocab, merges
```

### 答案 2: BPE 编码/解码

```python
def bpe_encode(text, vocab, merges):
    """BPE 编码: 文本 → token id 序列"""
    # 1. 预分词 (按空格)
    words = text.split(' ')
    # 2. 对每个词应用 BPE 合并规则
    token_ids = []
    for word in words:
        subwords = list(word) + ['</w>']
        # 按 merges 顺序应用
        for a, b in merges:
            i = 0
            new_sub = []
            while i < len(subwords):
                if i < len(subwords) - 1 and subwords[i] == a and subwords[i+1] == b:
                    new_sub.append(a + b)
                    i += 2
                else:
                    new_sub.append(subwords[i])
                    i += 1
            subwords = new_sub
        # 转为 id
        for sw in subwords:
            token_ids.append(vocab.get(sw, vocab['<unk>']))
    return token_ids

def bpe_decode(token_ids, vocab):
    """BPE 解码: token id 序列 → 文本"""
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = [inv_vocab.get(tid, '<unk>') for tid in token_ids]
    # 合并 + 去 </w>
    text = ''.join(tokens).replace('</w>', ' ')
    return text.strip()
```

### 答案 3: BBPE 训练 (GPT-2 风格)

```python
def get_pairs(word_ids):
    pairs = Counter()
    for word in word_ids:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def bbpe_train(text_bytes, vocab_size):
    """
    BBPE 训练
    text_bytes: 一个长文本的字节序列
    """
    # 1. 字节级初始化
    word_ids = [[b] for b in text_bytes]  # 每个字节是一个"词"
    vocab = {bytes([b]): i for i, b in range(256)}
    merges = []

    # 2. 循环合并
    while len(vocab) < vocab_size:
        pairs = get_pairs(word_ids)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new_token = best[0] + best[1]
        vocab[new_token] = len(vocab)
        merges.append(best)
        # 更新
        new_word_ids = []
        for word in word_ids:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_ids.append(new_word)
        word_ids = new_word_ids

    return vocab, merges
```

**复杂度**: 训练 $O(K \cdot N)$, K 是 vocab_size, N 是语料大小.

---

---

## 深入: 三种子词算法对比(BPE / WordPiece / Unigram)

| 算法 | 合并/切分准则 | 代表 |
| :--- | :--- | :--- |
| **BPE** | 贪心合并**最高频**相邻对 | GPT-2、RoBERTa |
| **WordPiece** | 合并使**语言模型似然**最大的对 | BERT |
| **Unigram** | 从大词表**剪枝**,保留似然损失最小的子词 | T5、SentencePiece |

BPE 频率驱动、简单;WordPiece 似然驱动;Unigram 概率化、能给一个词多种切分(利于正则)。现代 LLM 多用 **BBPE**(字节级 BPE):在 UTF-8 字节上做 BPE,**无 OOV**、天然支持多语和 emoji。

---

## 深入: Tokenization 为什么影响模型能力

分词不是无害的预处理,它直接影响下游:
- **数字**:"1234" 被切成 "12","34" → 算术能力受损。现代模型常**逐位数字**分词。
- **代码**:缩进/符号切分不当 → 代码能力下降;需专门覆盖。
- **多语公平性**:中文/小语种若字节化,同样内容 token 数远多于英文 → 上下文更贵、更慢(token 税)。
- **词表大小权衡**:大词表压缩率高(序列短)但 embedding/softmax 变大;小词表反之。常 32K-256K。

> "Glitch token"(如训练极少的词)会导致模型异常输出——分词器与训练语料错配的典型症状。

---

## 深入: 为什么 tokenization 是"最被低估的一环"

许多 LLM 的怪异行为(不会数字母、算术错、某些语言差)根源在分词而非模型。趋势:
- **数字逐位** + 代码专门处理已成标配。
- 探索 **byte-level / tokenizer-free**(如 MambaByte、BLT)彻底去掉分词器,但序列变长、成本高,尚未主流。
- 分词器一旦定下,**难以更改**(改了要重训),所以预训练前需慎重设计(见 [LLM发展第07章](../../LLM发展/第07章-LLM训练全流程实务.md))。

---

## 面试高频(补充)

1. BPE/WordPiece/Unigram 区别? → 频率合并 / 似然合并 / 剪枝概率化。
2. 为什么 LLM 用字节级 BPE? → 无 OOV,支持多语/emoji/代码。
3. 分词如何影响算术能力? → 数字被任意切分;逐位分词可缓解。
4. 大 vs 小词表权衡? → 压缩率(序列长)vs embedding/softmax 大小。

---

> **本节版本**: v2.0 (2026-07-16, 加厚: 三算法对比/对能力的影响/被低估的一环)
> **配套**: Ch06 LLM 核心技术目录
> **下一节**: [6.2 - RoPE (旋转位置编码) 深入](第06章-02-旋转位置编码.md)

---

*BBPE = LLM 时代的"通用语". 无 OOV 是它的最大优势.*

---

## 10. BPE 算法详解

### 10.1 学习目标

完成本节后，应能解释 BPE 的训练目标、合并过程与推理过程。

应能区分“字符级 BPE”“字节级 BPE”和“带预分词的 BPE”。

应能手算一个小语料上的前若干次合并。

应能写出可运行、可复现、不会跨词边界误合并的实现。

应能分析词表大小、序列长度、训练成本之间的权衡。

### 10.2 记号约定

设训练语料为 $D=\{x_1,x_2,\ldots,x_N\}$。

设初始符号集合为 $V_0$。

字符级 BPE 的 $V_0$ 通常是语料中出现过的字符集合。

字节级 BPE 的 $V_0$ 固定包含 $0,1,\ldots,255$ 共 256 个字节。

设第 $t$ 轮合并后的词表为 $V_t$。

设第 $t$ 轮待合并的符号对为 $(a_t,b_t)$。

设新符号为 $c_t=a_tb_t$。

设合并规则序列为 $M=[(a_1,b_1),\ldots,(a_K,b_K)]$。

规则的先后顺序属于 tokenizer 模型的一部分，不能随意重排。

### 10.3 BPE 的核心思想

BPE 最初是一种数据压缩算法。

在子词学习中，它把高频相邻符号对反复替换为新符号。

每次合并都让词表增加一个 token。

高频片段会逐渐成为完整 token。

低频词仍可退化成更小的子词、字符或字节。

因此 BPE 同时获得有限词表和开放词汇表示能力。

### 10.4 频次目标

第 $t$ 轮统计相邻符号对频次：

$$
C_t(a,b)=\sum_{x\in D}\sum_{i=1}^{|x|-1}
\mathbf{1}[x_i=a\land x_{i+1}=b].
$$

选择最高频符号对：

$$
(a_t,b_t)=\arg\max_{(a,b)}C_t(a,b).
$$

随后执行全局、不重叠替换：

$$
\ldots,a_t,b_t,\ldots\quad\Rightarrow\quad\ldots,c_t,\ldots
$$

若多个符号对频次相同，必须定义确定性的 tie-break 规则。

常见规则是按符号 ID 或字典序选择最小者。

确定性对于训练复现、跨机器校验和模型发布都很重要。

### 10.5 语料频率不能丢

若预分词后得到词频表 $F(w)$，则符号对频次应带权统计。

正确公式为：

$$
C_t(a,b)=\sum_w F(w)\cdot n_w(a,b),
$$

其中 $n_w(a,b)$ 是符号对在词 $w$ 当前切分中的出现次数。

例如 `low` 出现 100 次，不能只按一个样本计数。

教学实现最容易遗漏这个权重。

遗漏权重会让稀有词与高频词拥有相同影响力。

最终合并规则会明显偏离真实语料分布。

### 10.6 手算案例

考虑词频表：

| 词 | 频次 |
|---|---:|
| `low` | 5 |
| `lower` | 2 |
| `newest` | 6 |
| `widest` | 3 |

加入词尾标记 `</w>` 后：

```text
l o w </w>                 × 5
l o w e r </w>             × 2
n e w e s t </w>           × 6
w i d e s t </w>           × 3
```

初始时 `(e,s)` 的频次是 $6+3=9$。

初始时 `(s,t)` 的频次也是 $6+3=9$。

若 tie-break 选择 `(e,s)`，得到新符号 `es`。

第一轮结果：

```text
l o w </w>                 × 5
l o w e r </w>             × 2
n e w es t </w>            × 6
w i d es t </w>            × 3
```

第二轮 `(es,t)` 的频次为 9。

合并后得到 `est`。

```text
l o w </w>                 × 5
l o w e r </w>             × 2
n e w est </w>             × 6
w i d est </w>             × 3
```

第三轮 `(est,</w>)` 的频次为 9。

合并后得到 `est</w>`。

此时 `newest` 与 `widest` 共享高频后缀 token。

这展示了子词模型对词法结构的自动发现。

注意：BPE 并不知道“后缀”这个语言学概念。

它只是从局部共现频率中学到该片段。

### 10.7 训练与编码不是同一件事

训练阶段从语料中学习有序合并规则。

编码阶段固定使用已有规则，不再重新统计频次。

训练阶段的输入通常是大规模语料。

编码阶段的输入通常是单条或一个 batch 文本。

训练结果至少包含词表和合并规则。

生产系统还会保存归一化器、预分词器和特殊 token 配置。

只保存 token 到 ID 的映射是不够的。

没有合并顺序，同一文本可能出现不同切分。

### 10.8 朴素训练实现

下面实现保留词频，且不跨预分词边界合并。

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

SymbolSeq = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


@dataclass(frozen=True)
class BPEModel:
    vocab: dict[bytes, int]
    merges: tuple[Pair, ...]


def split_to_bytes(word: str) -> SymbolSeq:
    """一个预分词单元转为单字节符号序列。"""
    return tuple(bytes([b]) for b in word.encode("utf-8"))


def count_weighted_pairs(
    words: dict[SymbolSeq, int],
) -> Counter[Pair]:
    """按词频加权统计相邻符号对。"""
    counts: Counter[Pair] = Counter()
    for symbols, frequency in words.items():
        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += frequency
    return counts


def merge_pair(symbols: SymbolSeq, pair: Pair) -> SymbolSeq:
    """从左到右执行不重叠合并。"""
    output: list[bytes] = []
    i = 0
    while i < len(symbols):
        if (
            i + 1 < len(symbols)
            and symbols[i] == pair[0]
            and symbols[i + 1] == pair[1]
        ):
            output.append(pair[0] + pair[1])
            i += 2
        else:
            output.append(symbols[i])
            i += 1
    return tuple(output)


def train_byte_bpe(
    corpus: Iterable[str],
    vocab_size: int,
) -> BPEModel:
    """教学版 byte-level BPE；按空白预分词。"""
    if vocab_size < 256:
        raise ValueError("byte-level BPE 的 vocab_size 不能小于 256")

    raw_words: Counter[str] = Counter()
    for text in corpus:
        raw_words.update(text.split())

    words: dict[SymbolSeq, int] = {
        split_to_bytes(word): frequency
        for word, frequency in raw_words.items()
    }

    vocab: dict[bytes, int] = {
        bytes([value]): value
        for value in range(256)
    }
    merges: list[Pair] = []

    while len(vocab) < vocab_size:
        pair_counts = count_weighted_pairs(words)
        if not pair_counts:
            break

        # 先按频次最大，再按 bytes 字典序最小，保证可复现。
        best_pair = min(
            pair_counts,
            key=lambda pair: (-pair_counts[pair], pair),
        )
        new_symbol = best_pair[0] + best_pair[1]

        if new_symbol in vocab:
            break

        vocab[new_symbol] = len(vocab)
        merges.append(best_pair)
        words = {
            merge_pair(symbols, best_pair): frequency
            for symbols, frequency in words.items()
        }

    return BPEModel(vocab=vocab, merges=tuple(merges))
```

### 10.9 编码实现：按 rank 贪心

将规则顺序转换成 rank：

$$
\operatorname{rank}(a_t,b_t)=t.
$$

编码时反复合并当前序列里 rank 最小的可用符号对。

rank 小表示该规则训练得更早，优先级更高。

```python
def encode_piece(piece: str, model: BPEModel) -> list[int]:
    symbols = list(split_to_bytes(piece))
    ranks = {pair: rank for rank, pair in enumerate(model.merges)}

    while len(symbols) >= 2:
        candidates = [
            (ranks[(left, right)], i, (left, right))
            for i, (left, right) in enumerate(zip(symbols, symbols[1:]))
            if (left, right) in ranks
        ]
        if not candidates:
            break

        _, _, best_pair = min(candidates)
        symbols = list(merge_pair(tuple(symbols), best_pair))

    return [model.vocab[symbol] for symbol in symbols]


def encode_text(text: str, model: BPEModel) -> list[int]:
    ids: list[int] = []
    for piece in text.split():
        ids.extend(encode_piece(piece, model))
    return ids


def decode_ids(ids: list[int], model: BPEModel) -> str:
    inverse_vocab = {idx: token for token, idx in model.vocab.items()}
    raw = b"".join(inverse_vocab[idx] for idx in ids)
    return raw.decode("utf-8", errors="strict")
```

教学实现使用线性扫描候选对。

真实 tokenizer 会使用更高效的数据结构和批量处理。

编码器还必须恢复空格信息，不能简单用 `split()` 后丢失分隔符。

GPT-2 风格方案会在预分词阶段把空格编码进 token 表示。

SentencePiece 常使用 `▁` 表示词边界或空白。

### 10.10 最小可运行检查

```python
def demo_bpe() -> None:
    corpus = [
        "low lower lowest",
        "new newer newest",
        "low low low",
    ]
    model = train_byte_bpe(corpus, vocab_size=280)

    assert len(model.vocab) <= 280
    assert len(model.vocab) >= 256
    assert len(model.merges) == len(model.vocab) - 256

    token_ids = encode_piece("lowest", model)
    recovered = decode_ids(token_ids, model)
    assert recovered == "lowest"


if __name__ == "__main__":
    demo_bpe()
```

这个检查覆盖了词表增长、规则数量和 UTF-8 往返。

若要覆盖中文，应再加入 `你好🙂` 的 round-trip 检查。

### 10.11 为什么全局替换必须不重叠

对序列 `a a a` 合并 `(a,a)`。

从左到右不重叠替换得到 `aa a`。

不能同时把位置 `(0,1)` 与 `(1,2)` 都合并。

因为中间的 `a` 不能同时属于两个输出 token。

因此单轮合并的输出长度是确定的。

长度减少量等于实际不重叠匹配数。

### 10.12 字节级 BPE 的 UTF-8 语义

UTF-8 将 Unicode 字符编码为 1 到 4 个字节。

英文 ASCII 字符通常占 1 字节。

常见中文字符通常占 3 字节。

emoji 通常占 4 字节或更多组合序列。

字节级初始词表可表示任意合法或非法字节序列。

所以编码层面没有 OOV。

但中间 token 可能不是独立合法的 UTF-8 字符串。

例如一个中文字符可能先被拆成三个 byte token。

只有拼接完整字节后才能正确解码。

调试工具不应强制把每个 byte token 单独转成 Unicode。

可用十六进制或可逆映射显示不可打印字节。

### 10.13 BBPE 无 OOV 不等于没有 `<unk>`

有些模型仍保留 `<unk>` 特殊 token。

它可能用于兼容协议或非法归一化结果。

但只要所有输入先转 UTF-8 字节，普通文本无需回退到 `<unk>`。

“词表里是否存在 `<unk>`”与“正常输入是否会产生 OOV”是两个问题。

面试回答应明确区分这两层。

### 10.14 预分词器的作用

BPE 可以直接在完整字节流上学习。

但直接学习可能产生跨词、跨换行甚至跨文档的合并。

预分词器通过边界约束控制可合并范围。

常见边界包括空白、标点、数字和缩写模式。

代码语料还需要考虑缩进、换行和运算符。

预分词规则改变会改变最终 token 分布。

所以预分词器是 tokenizer 模型的一部分。

训练和推理必须使用完全相同的预分词逻辑。

### 10.15 GPT-2 字节到可见字符映射

GPT-2 没有直接把任意 byte 写进文本词表文件。

它先把 256 个 byte 映射到 256 个可显示 Unicode 字符。

这个映射是一一对应的。

因此词表可以存成普通 JSON 字符串。

映射只是序列化层，不改变底层 byte-level 语义。

编码前执行 byte-to-unicode。

解码后执行 unicode-to-byte。

最后才按 UTF-8 还原原始文本。

### 10.16 词表大小与参数量

设隐藏维度为 $d$，词表大小为 $|V|$。

输入 embedding 参数量为：

$$
P_{\text{embed}}=|V|d.
$$

若输出投影不与 embedding 共享，则还需：

$$
P_{\text{lm-head}}=d|V|+|V|.
$$

若权重共享，两个大矩阵可复用。

例如 $d=4096$，词表从 32K 增到 128K。

单个 BF16 embedding 矩阵显存从约 256 MiB 增至约 1 GiB。

大词表降低序列长度，却增加参数、通信和 softmax 成本。

该权衡不能只看压缩率。

### 10.17 PyTorch 参数量计算

```python
import torch
from torch import nn


def embedding_memory_mib(
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = vocab_size * hidden_size * bytes_per_element
    return total_bytes / 1024**2


for vocab_size in (32_000, 64_000, 128_000):
    memory = embedding_memory_mib(vocab_size, 4096)
    print(vocab_size, f"{memory:.1f} MiB")

embedding = nn.Embedding(32_000, 4096, dtype=torch.bfloat16)
assert embedding.weight.numel() == 32_000 * 4096
```

### 10.18 Softmax 成本

每个位置的输出 logits 大小为 $|V|$。

一个 batch 的粗略投影复杂度为：

$$
O(B\cdot L\cdot d\cdot |V|).
$$

其中 $B$ 是 batch size，$L$ 是序列长度。

词表增大可能让 $L$ 下降。

因此总成本取决于压缩率提升能否抵消输出维度扩大。

实际实现还受矩阵乘吞吐、张量并行和采样策略影响。

### 10.19 BPE 编码复杂度

朴素训练每轮全量扫描语料。

若总符号数为 $T$，执行 $K$ 次合并，复杂度约为 $O(KT)$。

每轮重建全部序列也会造成大量内存分配。

工业实现会维护 pair 到出现位置的倒排索引。

还会用优先队列维护最高频 pair。

当局部合并发生时，只更新邻近 pair 的频次。

编码的朴素实现同样可能达到 $O(KL)$。

基于 rank 和局部链表的实现可显著减少重复扫描。

### 10.20 合并规则的确定性

以下因素都会破坏可复现性：

- 同频 pair 使用无序集合迭代顺序。
- 并行归并计数时使用非确定 tie-break。
- 训练语料分片顺序变化。
- 归一化库版本变化。
- 预分词正则表达式变化。
- 特殊 token 插入顺序变化。

发布时应记录 tokenizer 文件哈希。

训练 checkpoint 也应记录 tokenizer 哈希。

推理服务启动时应校验二者一致。

### 10.21 特殊 token

常见特殊 token 包括：

| token | 用途 |
|---|---|
| `<pad>` | batch 对齐 |
| `<bos>` | 序列开始 |
| `<eos>` | 序列结束或文档边界 |
| `<unk>` | 未知符号兼容 |
| `<mask>` | 掩码语言模型 |
| `<sep>` | 段落或字段分隔 |
| `<cls>` | 分类聚合位置 |

特殊 token 通常不参与普通 BPE 合并。

它们应由专门匹配器优先识别。

用户输入中出现相同文本时，是否视为特殊 token 必须有明确策略。

聊天模板经常使用新增控制 token 表示角色边界。

错误处理会造成 prompt 注入边界混乱或训练标签错位。

### 10.22 BPE-dropout

标准 BPE 对同一文本给出确定切分。

BPE-dropout 在训练时随机跳过部分合并规则。

同一词可产生多种粒度的切分。

这相当于对子词边界做数据增强。

设每个可用合并以概率 $p$ 被丢弃。

$p=0$ 时退化为标准 BPE。

$p$ 越大，平均序列通常越长。

推理时通常关闭 dropout，使用确定切分。

### 10.23 正则化示意代码

```python
import random


def encode_with_bpe_dropout(
    piece: str,
    model: BPEModel,
    dropout: float,
    rng: random.Random,
) -> list[int]:
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout 必须位于 [0, 1)")

    symbols = list(split_to_bytes(piece))
    ranks = {pair: rank for rank, pair in enumerate(model.merges)}

    while len(symbols) >= 2:
        candidates = []
        for i, pair in enumerate(zip(symbols, symbols[1:])):
            if pair in ranks and rng.random() >= dropout:
                candidates.append((ranks[pair], i, pair))
        if not candidates:
            break
        _, _, pair = min(candidates)
        symbols = list(merge_pair(tuple(symbols), pair))

    return [model.vocab[symbol] for symbol in symbols]
```

生产实现应避免候选扫描顺序意外改变随机数消耗。

若要求严格复现，应固定随机种子和算法版本。

### 10.24 常见错误清单

错误一：把每个字节包装成独立“词”，导致没有可统计的相邻 pair。

错误二：统计独立词类型，却没有乘词频。

错误三：训练时按一个 tie-break，加载后按另一个 tie-break。

错误四：编码时按字符串最长匹配，而不是按合并 rank。

错误五：把每个 byte token 单独 UTF-8 解码。

错误六：预分词后丢失空格，解码无法还原原文。

错误七：允许普通合并跨过特殊 token 边界。

错误八：词表 ID 重排后没有同步模型 embedding。

错误九：只测试英文，没有测试中文、emoji 和组合字符。

错误十：把 token 数和 Unicode 字符数视为同一概念。

### 10.25 本节练习

1. 手算 `banana bandana` 语料的前三次 BPE 合并。

2. 修改训练函数，使词尾边界也能被可逆编码。

3. 为中文、emoji、空字符串加入 round-trip 测试。

4. 比较 256、300、500 三种词表大小的平均 token 数。

5. 证明一次不重叠合并不会增加序列长度。

6. 解释为什么 longest-match 不一定等价于 merge-rank 编码。

7. 设计一个稳定的同频 pair tie-break 方案。

8. 估算 128K 词表、8192 hidden size 的 BF16 embedding 显存。

---

## 11. SentencePiece

### 11.1 SentencePiece 是什么

SentencePiece 是语言无关的子词训练与编码工具。

它可以直接把原始句子作为输入。

它不要求调用外部、语言相关的空格分词器。

它支持 BPE 和 Unigram 两类主要模型。

它把归一化、空白处理、子词模型和特殊 token 配置统一保存。

因此它不是一种单独的合并算法，而是一套 tokenizer 框架与文件格式。

### 11.2 为什么强调语言无关

英语天然使用空格分词。

中文、日文和泰文的词边界并不总由空格标记。

如果 tokenizer 强依赖英语式预分词，多语言行为会不一致。

SentencePiece 把输入视为 Unicode 字符序列。

空格也被当作普通信息编码。

这让同一套训练流程可用于多种语言。

### 11.3 `▁` 空白标记

SentencePiece 常把空格转写为可见元符号 `▁`。

例如：

```text
原文: Hello world
内部: ▁Hello▁world
```

若切分结果为：

```text
[▁Hello, ▁world]
```

解码时把 `▁` 还原为空格即可。

词首信息因此不会丢失。

前导空格也能参与建模。

连续空格是否保留取决于归一化与配置。

### 11.4 SentencePiece 处理流水线

典型流水线可写成：

$$
\text{raw text}
\xrightarrow{N}
\text{normalized text}
\xrightarrow{E}
\text{escaped text}
\xrightarrow{S}
\text{subword pieces}
\xrightarrow{I}
\text{token ids}.
$$

$N$ 是规范化器。

$E$ 负责空白转义等元符号处理。

$S$ 是 BPE 或 Unigram 分词模型。

$I$ 是 piece 到整数 ID 的映射。

解码按相反方向执行可逆步骤。

归一化通常不是完全可逆的，因此 round-trip 的目标可能是规范化后文本。

### 11.5 Unicode 归一化

视觉相同的字符串可能有不同码点序列。

例如 `é` 可以是单码点 U+00E9。

也可以是 `e` 加组合重音 U+0301。

NFC 会优先组合成预组字符。

NFD 会优先分解为基本字符与组合标记。

NFKC 和 NFKD 还会执行兼容性归一化。

兼容性归一化可能把全角字符、上标等映射为普通形式。

这可降低稀疏性，但可能丢失语义差异。

数学、代码和法律文本尤其需要谨慎。

### 11.6 归一化目标

设原始字符串为 $x$，归一化函数为 $N$。

训练与推理必须满足：

$$
N_{\text{train}}(x)=N_{\text{serve}}(x).
$$

理想情况下还希望归一化幂等：

$$
N(N(x))=N(x).
$$

幂等可防止重复处理不断改变输入。

若归一化不可逆，解码只能恢复 $N(x)$，不能保证恢复原始 $x$。

### 11.7 训练数据格式

SentencePiece 通常读取“一行一个句子”的文本。

空行是否保留要按任务决定。

大规模训练可从语料均匀采样。

采样必须兼顾语言、领域和文档长度。

若英语占 90%，共享词表会自然偏向英语片段。

小语种可能退化成更多字符或字节 token。

这会形成显著的 token tax。

### 11.8 多语言采样温度

设语言 $l$ 的原始样本比例为 $p_l$。

可使用温度重采样：

$$
q_l=\frac{p_l^\alpha}{\sum_j p_j^\alpha}.
$$

当 $0<\alpha<1$ 时，低资源语言被上采样。

当 $\alpha=1$ 时，保持原始比例。

当 $\alpha\to 0$ 时，各语言趋近均匀。

上采样提高小语种词表覆盖。

但过度上采样会牺牲主流语言压缩率。

### 11.9 词表覆盖率与字符覆盖率

SentencePiece 的 `character_coverage` 控制初始字符覆盖。

高覆盖率保留更多低频字符。

低覆盖率会把尾部字符交给 `<unk>` 或 byte fallback。

中文、日文等字符集较大时常使用很高覆盖率。

byte fallback 可进一步确保没有 OOV。

字符覆盖率不是最终 token 覆盖率。

它影响基础字符集合的选择。

### 11.10 BPE 模式与 Unigram 模式

SentencePiece BPE 从小词表开始逐步合并。

SentencePiece Unigram 从大候选词表开始逐步剪枝。

BPE 编码通常由合并规则决定。

Unigram 编码通常用动态规划寻找概率最高的切分。

BPE 更直观，训练与推理都较容易解释。

Unigram 支持自然的多切分采样。

二者都可使用相同的归一化和空白机制。

### 11.11 命令行训练示意

```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=demo \
  --model_type=unigram \
  --vocab_size=32000 \
  --character_coverage=0.9995 \
  --normalization_rule_name=nmt_nfkc \
  --bos_id=1 \
  --eos_id=2 \
  --unk_id=0 \
  --pad_id=3
```

训练通常生成 `demo.model` 与 `demo.vocab`。

`.model` 是推理所需的完整二进制模型。

`.vocab` 更适合人工查看 piece 与 score。

不能只依赖 `.vocab` 重建全部行为。

### 11.12 Python 使用示意

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="demo.model")

text = "大模型 tokenization 很重要。"

pieces = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)
restored = sp.decode(ids)

print(pieces)
print(ids)
print(restored)

assert sp.get_piece_size() == 32_000
assert len(ids) == len(pieces)
```

接口名称可能随绑定版本略有差异。

工程中应锁定版本并保存模型哈希。

### 11.13 特殊 token ID 规划

特殊 ID 应在模型训练前固定。

常见规划示例：

```text
0 -> <unk>
1 -> <s>
2 -> </s>
3 -> <pad>
4 -> <system>
5 -> <user>
6 -> <assistant>
```

不同模型的约定并不统一。

不能假设 `pad_id` 总是 0。

不能假设 BOS 与 EOS 一定自动添加。

编码 API 的 `add_bos`、`add_eos` 行为必须显式测试。

训练 label 的 ignore index 也不应与 pad ID 混淆。

### 11.14 用户自定义符号

`user_defined_symbols` 可把指定字符串作为不可拆分 piece。

适合领域标签、控制标记或结构化占位符。

例如：

```text
<image>
<audio>
<tool_call>
<tool_result>
```

自定义符号不会自动获得安全语义。

服务端仍需验证用户能否直接注入这些控制标记。

若控制 token 与普通文本共用可见字符串，应规定转义策略。

### 11.15 Control symbols 与普通 piece

控制符号通常由应用或模板显式插入。

它们不一定应该从用户普通文本中自动识别。

普通 piece 则由分词算法从文本产生。

两者在模型文件中可能都有固定 ID。

但访问权限和模板语义不同。

这是聊天模型安全边界的一部分。

### 11.16 子词正则化

Unigram 模型可以从多个合理切分中采样。

常用参数包括 `nbest_size` 与 `alpha`。

`nbest_size` 控制候选切分范围。

`alpha` 控制采样分布的尖锐程度。

训练时采样可提升对切分扰动的鲁棒性。

推理和评估时通常使用最优确定切分。

若评估仍随机采样，指标会产生额外方差。

### 11.17 采样公式

设切分候选集合为 $\mathcal{S}(x)$。

每个切分 $s$ 的模型概率为 $P(s)$。

温度采样可写为：

$$
P_\alpha(s\mid x)=
\frac{P(s)^\alpha}
{\sum_{s'\in\mathcal{S}(x)}P(s')^\alpha}.
$$

较大的 $\alpha$ 使分布更集中于高概率切分。

较小的 $\alpha$ 增加多样性。

具体库参数的方向应以实际 API 文档为准并做小样本验证。

### 11.18 byte fallback

若启用 byte fallback，无法由常规 piece 覆盖的字符可退化成字节 token。

这样能处理罕见 Unicode、emoji 新变体和脏数据。

代价是词表中需为 byte 表示保留槽位。

极端输入的序列长度也会增大。

byte fallback 是“兜底机制”，不代表高频语言都应该按字节切分。

应监控实际 fallback 比例。

### 11.19 训练前的数据去重

重复文档会放大其中片段的频次。

tokenizer 会给重复模板、页脚或爬虫噪声分配宝贵词表槽位。

因此 tokenizer 训练语料也应做去重与质量过滤。

常见清理对象包括：

- 重复网页导航。
- Cookie 提示。
- Base64 长串。
- 乱码。
- 自动生成 ID。
- 大量连续标点。
- 隐私数据模板。

tokenizer 语料不必与模型预训练语料逐条一致。

但领域与语言分布应大致匹配。

### 11.20 SentencePiece 模型验收

至少检查以下维度：

- 模型文件能在目标语言绑定中加载。
- 特殊 token ID 与配置一致。
- BOS/EOS 自动添加行为符合预期。
- 空格、换行和制表符处理明确。
- 中文、英文、代码、emoji 都能编码。
- byte fallback 或 `<unk>` 比例可接受。
- 解码得到预期规范化文本。
- 同一输入重复编码结果稳定。
- 多线程编码结果一致。
- 文件哈希在训练和服务端一致。

### 11.21 PyTorch batch 对齐

```python
from dataclasses import dataclass

import torch


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def pad_sequences(
    sequences: list[list[int]],
    pad_id: int,
) -> TokenBatch:
    if not sequences:
        raise ValueError("sequences 不能为空")

    max_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), max_length),
        fill_value=pad_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)

    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence)
        attention_mask[row, :length] = 1

    return TokenBatch(input_ids, attention_mask)
```

padding 方向必须与模型的位置编码、KV cache 和生成实现一致。

decoder-only 模型批量生成常使用左 padding。

训练时也常使用右 padding 或 packed sequence。

不能仅凭模型类型猜测，应查看模型配置和推理实现。

### 11.22 词表导出检查

```python
def inspect_sentencepiece(sp, limit: int = 20) -> None:
    size = sp.get_piece_size()
    print("vocab size:", size)
    for idx in range(min(size, limit)):
        piece = sp.id_to_piece(idx)
        score = sp.get_score(idx)
        print(idx, repr(piece), score)
```

检查低 ID 区域可快速发现特殊 token 排列错误。

检查词表尾部可观察低频字符、byte token 或冗余片段。

### 11.23 常见误区

误区一：SentencePiece 就等于 Unigram。

事实：SentencePiece 也支持 BPE 等模型类型。

误区二：SentencePiece 完全不做预处理。

事实：它内置规范化与空白转义等步骤。

误区三：解码一定逐字符还原原始输入。

事实：不可逆归一化可能改变原文。

误区四：保存 `.vocab` 就足够部署。

事实：完整模型还包含算法和规范化配置。

误区五：多语言共享词表天然公平。

事实：语料比例决定不同语言的 token 效率。

### 11.24 本节练习

1. 比较 NFC 与 NFKC 对数学符号和全角字符的影响。

2. 训练 8K BPE 与 8K Unigram 模型并比较压缩率。

3. 统计五种语言的字符/token 比率。

4. 构造包含连续空格、换行、Tab 的 round-trip 测试。

5. 验证自定义控制 token 是否会被普通文本误触发。

6. 打印词表中 score 最低的 100 个 piece 并分析来源。

7. 估算 byte fallback 为词表带来的固定槽位成本。

8. 设计训练语料的语言温度采样方案。

---

## 12. Unigram Language Model

### 12.1 从“合并”转向“剪枝”

BPE 从基础符号开始不断合并。

Unigram LM 通常从一个较大的候选词表开始。

它为每个候选子词分配概率。

一个句子可有多种合法切分。

模型通过边缘似然评估所有切分。

训练过程逐步删除贡献较小的候选子词。

最终保留目标词表规模。

### 12.2 独立生成假设

设切分 $s=(z_1,z_2,\ldots,z_m)$。

Unigram 假设各 token 独立生成：

$$
P(s)=\prod_{i=1}^{m}p(z_i).
$$

在 log 空间：

$$
\log P(s)=\sum_{i=1}^{m}\log p(z_i).
$$

“Unigram”指的就是这种 token 独立假设。

它不是说文本只含单字符 token。

候选 token 可以是任意长度的子串。

### 12.3 句子边缘似然

同一字符串 $x$ 可能有多个切分。

设所有合法切分集合为 $\mathcal{S}(x)$。

字符串概率为：

$$
P(x)=\sum_{s\in\mathcal{S}(x)}P(s).
$$

语料对数似然为：

$$
\mathcal{L}(D)=\sum_{x\in D}\log P(x).
$$

训练目标是最大化 $\mathcal{L}(D)$。

直接枚举切分数量通常呈指数增长。

但可借助动态规划高效计算。

### 12.4 最优切分与 Viterbi

推理时常寻找概率最大的单个切分：

$$
s^*=\arg\max_{s\in\mathcal{S}(x)}P(s).
$$

在负对数代价下等价于：

$$
s^*=\arg\min_s\sum_{z\in s}-\log p(z).
$$

这是一个最短路径问题。

字符串位置是图节点。

每个可匹配 token 是从起点到终点的一条边。

边权为 $-\log p(z)$。

Viterbi 动态规划可求最短路径。

### 12.5 Viterbi 递推

令 $d[i]$ 表示覆盖前 $i$ 个字符的最小代价。

初始化：

$$
d[0]=0.
$$

递推：

$$
d[j]=\min_{i<j,\;x_{i:j}\in V}
\left(d[i]-\log p(x_{i:j})\right).
$$

同时保存最优前驱位置即可回溯切分。

若不存在可达路径，说明基础字符覆盖不完整。

实际系统会保留字符或 byte fallback 保证可达。

### 12.6 纯 Python Viterbi 实现

```python
from math import inf, log


def unigram_viterbi(
    text: str,
    probabilities: dict[str, float],
) -> tuple[list[str], float]:
    n = len(text)
    best_cost = [inf] * (n + 1)
    previous: list[tuple[int, str] | None] = [None] * (n + 1)
    best_cost[0] = 0.0

    max_piece_length = max(map(len, probabilities), default=0)

    for end in range(1, n + 1):
        start_min = max(0, end - max_piece_length)
        for start in range(start_min, end):
            piece = text[start:end]
            probability = probabilities.get(piece)
            if probability is None or best_cost[start] == inf:
                continue

            candidate = best_cost[start] - log(probability)
            if candidate < best_cost[end]:
                best_cost[end] = candidate
                previous[end] = (start, piece)

    if best_cost[n] == inf:
        raise ValueError("词表无法覆盖输入")

    pieces: list[str] = []
    position = n
    while position > 0:
        state = previous[position]
        assert state is not None
        start, piece = state
        pieces.append(piece)
        position = start

    pieces.reverse()
    return pieces, best_cost[n]
```

### 12.7 Viterbi 示例

```python
probabilities = {
    "北": 0.08,
    "京": 0.08,
    "大": 0.10,
    "学": 0.10,
    "北京": 0.25,
    "大学": 0.28,
    "北京大学": 0.11,
}

pieces, cost = unigram_viterbi("北京大学", probabilities)
print(pieces, cost)
assert "".join(pieces) == "北京大学"
```

最优结果不只由 token 数决定。

一个长 token 概率太低时，两个短 token 可能更优。

这与简单最长匹配不同。

### 12.8 Forward 算法

为了计算所有切分的概率和，使用 forward 动态规划。

令 $\alpha[j]$ 为前 $j$ 个字符的总概率。

初始化：

$$
\alpha[0]=1.
$$

递推：

$$
\alpha[j]=\sum_{i<j,\;x_{i:j}\in V}
\alpha[i]p(x_{i:j}).
$$

最终：

$$
P(x)=\alpha[|x|].
$$

直接在概率空间计算容易下溢。

工业实现使用 log-space 与 log-sum-exp。

### 12.9 LogSumExp

对数域中的加法为：

$$
\log(e^a+e^b)
=m+\log(e^{a-m}+e^{b-m}),
$$

其中 $m=\max(a,b)$。

减去最大值避免指数溢出。

多个值时：

$$
\operatorname{LSE}(a_1,\ldots,a_n)
=m+\log\sum_i e^{a_i-m}.
$$

PyTorch 提供 `torch.logsumexp`。

### 12.10 PyTorch forward loss

```python
import torch


def unigram_log_likelihood(
    text: str,
    pieces: list[str],
    logits: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != (len(pieces),):
        raise ValueError("logits 形状必须与 pieces 一致")

    log_probs = torch.log_softmax(logits, dim=0)
    n = len(text)
    states = [torch.tensor(float("-inf"), device=logits.device)] * (n + 1)
    states[0] = torch.tensor(0.0, device=logits.device)

    for end in range(1, n + 1):
        candidates = []
        for piece_id, piece in enumerate(pieces):
            start = end - len(piece)
            if start >= 0 and text[start:end] == piece:
                candidates.append(states[start] + log_probs[piece_id])
        if candidates:
            states[end] = torch.logsumexp(torch.stack(candidates), dim=0)

    return states[n]
```

该教学代码按 Python 字符位置切片。

真实 SentencePiece 会在规范化后的内部符号序列上工作。

长文本和大词表应使用 trie 降低匹配成本。

### 12.11 梯度解释

边缘似然对某个 token log-probability 的梯度与其后验期望计数相关。

直观上：

某 token 出现在高概率切分中的次数越多，更新越倾向提高它的概率。

某 token 只参与低概率路径，贡献就较小。

这与只保留单一 Viterbi 切分相比利用了更多不确定性。

EM 训练正是利用 expected count 更新参数。

### 12.12 EM 视角

切分 $s$ 是隐变量。

完整数据似然为 $P(x,s)$。

E 步计算后验：

$$
P(s\mid x)=\frac{P(s)}{\sum_{s'}P(s')}.
$$

并得到每个 token 的期望出现次数。

M 步归一化期望计数：

$$
p(z)=\frac{\mathbb{E}[c(z)]}
{\sum_{z'}\mathbb{E}[c(z')]}.
$$

反复迭代直到似然稳定或达到迭代上限。

### 12.13 Forward-Backward

Forward 计算每个前缀的概率。

Backward 计算每个后缀的概率。

若 token $z=x_{i:j}$，其在该位置被使用的后验概率与下式成正比：

$$
\alpha[i]\,p(z)\,\beta[j].
$$

除以整句概率 $P(x)$ 得到归一化后验。

对所有位置和句子求和可得 token 的期望计数。

### 12.14 候选词表初始化

Unigram 不能从只有单字符的目标小词表直接剪枝。

它先构造过完备候选集。

候选可来自：

- 所有高频字符 n-gram。
- 后缀数组统计。
- 初步 BPE 结果。
- 领域词典。
- 必须保留的单字符。
- byte fallback token。

候选集太大会增加训练内存和时间。

候选集太小则限制最终模型上限。

### 12.15 词表剪枝

训练若干轮 EM 后，评估删除每个 piece 的损失增量。

删除 piece 后，包含它的字符串要改用替代切分。

若似然下降很小，说明该 piece 可被其他 piece 充分替代。

每轮通常批量删除一部分低价值 piece。

然后重新执行 EM。

不能一次删除到目标规模，否则概率估计来不及适应。

单字符和必要 byte token 通常不可删除。

### 12.16 剪枝损失直觉

设当前负对数似然为 $L(V)$。

删除 token $z$ 后为 $L(V\setminus\{z\})$。

定义删除代价：

$$
\Delta_z=L(V\setminus\{z\})-L(V).
$$

$\Delta_z$ 小表示删除影响小。

优先删除低 $\Delta_z$ 候选。

实际实现会使用近似，避免对每个 token 完整重训。

### 12.17 为什么必须保留基础符号

若删除某个唯一覆盖字符，部分输入将无合法切分。

这会使 $P(x)=0$，负对数似然变为无穷。

因此字符覆盖集通常被标记为 required pieces。

启用 byte fallback 时，256 个 byte 也构成最终兜底。

可达性比词表压缩更基础。

### 12.18 N-best 切分

除最优路径外，还可计算概率最高的前 $k$ 个切分。

这相当于 DAG 上的 k-shortest paths。

N-best 可用于：

- 训练时子词正则化。
- 分析边界不确定性。
- 调试词表冗余。
- 评估某个词的替代切分。

$k$ 太大会增加编码开销。

### 12.19 随机采样切分

精确采样可基于 forward/backward 概率逐步选择边。

从位置 $i$ 出发，选择 token $z=x_{i:j}$ 的条件概率与下式成正比：

$$
p(z)\beta[j].
$$

归一化所有可用出边后采样。

重复到句尾得到一个完整切分。

加入温度可控制随机程度。

### 12.20 Trie 加速匹配

朴素 Viterbi 对每个终点尝试所有 token，成本较高。

Trie 可从每个起点沿文本向前匹配。

每到一个终止节点就产生一条候选边。

设最长 token 长度为 $M$，文本长度为 $L$。

粗略匹配复杂度可降到 $O(LM)$ 加输出边数。

对大词表尤其重要。

### 12.21 数值稳定实现要点

- 所有路径概率使用 log-space。
- 不可达状态使用 $-\infty$。
- 求和使用 log-sum-exp。
- 概率归一化前减最大 logit。
- 监控 NaN 与无穷值。
- 低概率 piece 可设置下限后重新归一化。
- EM 停止条件使用相对似然变化。

### 12.22 BPE 与 Unigram 对比

| 维度 | BPE | Unigram LM |
|---|---|---|
| 起点 | 小词表 | 大候选词表 |
| 训练方向 | 逐步合并 | 逐步剪枝 |
| 目标 | 高频 pair | 语料似然 |
| 切分 | merge rank | 概率最优路径 |
| 多切分 | 需 BPE-dropout | 原生支持 |
| 推理 | 贪心应用规则 | 动态规划 |
| 可解释性 | 合并历史直观 | 概率模型直观 |

两者都不是绝对更优。

选择取决于语言、工具链、吞吐和训练目标。

### 12.23 一个反例：最长 token 不一定最好

假设：

```text
p("abc") = 0.01
p("a")   = 0.40
p("bc")  = 0.30
```

最长匹配选择 `abc`，概率为 0.01。

切分 `a + bc` 的概率为 $0.4\times0.3=0.12$。

Unigram Viterbi 会选择 `a + bc`。

所以“token 越长越优”不是概率模型的结论。

### 12.24 PyTorch 可微检查

```python
def demo_unigram_gradient() -> None:
    pieces = ["a", "b", "ab"]
    logits = torch.zeros(3, requires_grad=True)

    log_likelihood = unigram_log_likelihood("ab", pieces, logits)
    loss = -log_likelihood
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


demo_unigram_gradient()
```

这个检查验证边缘似然可对 piece logits 反向传播。

### 12.25 常见错误

错误一：只用 Viterbi 最优切分训练，却声称在优化完整边缘似然。

错误二：在概率空间连乘长句，导致数值下溢为零。

错误三：剪掉唯一覆盖某字符的基础 piece。

错误四：把 token score 直接当作归一化概率。

错误五：采样训练开启，评估也忘记关闭。

错误六：候选词表只来自英语空格词，损害无空格语言。

错误七：忽略归一化后字符位置与原始位置的差异。

错误八：认为 Unigram 的 token 一定是单字符。

### 12.26 本节练习

1. 手算 `abab` 在词表 `{a,b,ab}` 下的全部切分。

2. 写出对应的 forward 数组与 Viterbi 数组。

3. 推导 token `ab` 的期望计数。

4. 把 Viterbi 实现改为 byte 序列版本。

5. 使用 trie 加速候选匹配。

6. 实现温度采样并验证随机种子可复现。

7. 比较最优切分 loss 与边缘似然 loss。

8. 设计不可删除 piece 的标记与校验逻辑。

---

## 13. 多模态 Tokenizer

### 13.1 “多模态 tokenizer”有两层含义

第一层是各模态自身的离散化器。

例如文本 BPE、图像 patch、音频 codec token、视频时空 patch。

第二层是把这些表示组织到统一序列中的协议。

例如 `<image_start>`、图像 token、`<image_end>` 与文本 token 的排列。

两层不能混为一谈。

视觉编码器输出连续向量时，也常被口语化称为视觉 token。

它们未必对应离散词表 ID。

### 13.2 统一表示

设文本 token 序列为：

$$
T=(t_1,\ldots,t_{L_t}).
$$

图像 token 或 patch embedding 序列为：

$$
I=(i_1,\ldots,i_{L_i}).
$$

音频序列为：

$$
A=(a_1,\ldots,a_{L_a}).
$$

多模态模型需要定义组合函数：

$$
X=\operatorname{Pack}(T,I,A,\text{control tokens}).
$$

`Pack` 同时决定顺序、边界、位置 ID 和 attention mask。

### 13.3 图像 patch tokenization

给定图像 $x\in\mathbb{R}^{H\times W\times C}$。

使用大小为 $P\times P$ 的不重叠 patch。

patch 数量为：

$$
N=\frac{H}{P}\cdot\frac{W}{P},
$$

假设 $H,W$ 都可被 $P$ 整除。

每个 patch 展平维度为 $P^2C$。

再经线性投影映射到模型隐藏维度 $d$。

### 13.4 Conv2d 等价实现

核大小和步幅都为 $P$ 的卷积可完成 patchify 与线性投影。

```python
import torch
from torch import nn


class PatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 768,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images 应为 [B, C, H, W]")
        _, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("图像尺寸必须能被 patch_size 整除")

        features = self.projection(images)
        return features.flatten(2).transpose(1, 2)
```

输出形状为 `[B, N, d]`。

这些是连续视觉 token embedding。

它们不经过文本词表查表。

### 13.5 Patch tokenizer 检查

```python
def demo_patch_tokenizer() -> None:
    tokenizer = PatchTokenizer(hidden_size=64, patch_size=16)
    images = torch.randn(2, 3, 224, 224)
    tokens = tokenizer(images)
    assert tokens.shape == (2, 196, 64)


demo_patch_tokenizer()
```

$224/16=14$，所以每张图产生 $14\times14=196$ 个 patch token。

### 13.6 动态分辨率

固定缩放到 224×224 会丢失高分辨率细节。

动态分辨率方案允许不同 patch 网格。

常见策略包括：

- 保持宽高比缩放。
- 将大图切成多个 tile。
- 添加全局缩略图。
- 按预算选择网格。
- 对低信息区域做 token pruning。

动态分辨率提高 OCR 与图表理解能力。

但会带来变长序列、batch 对齐和位置编码问题。

### 13.7 图像 token 预算

若每个 tile 产生 $N_p$ 个视觉 token，共有 $K$ 个 tile。

总视觉长度约为：

$$
L_{vision}=K\cdot N_p+L_{control}.
$$

self-attention 朴素复杂度与总长度平方相关：

$$
O((L_{text}+L_{vision})^2d).
$$

所以高分辨率输入不能无限增加 patch。

服务必须设置图像数量、像素和视觉 token 上限。

### 13.8 离散图像 tokenizer

VQ-VAE、VQGAN 等方法把图像编码为离散 codebook ID。

编码器产生连续 latent：

$$
z_e(x)\in\mathbb{R}^{h\times w\times d}.
$$

对每个位置寻找最近 codebook 向量：

$$
k^*=\arg\min_k\|z_e(x)-e_k\|_2^2.
$$

输出 $k^*$ 就是离散视觉 token ID。

解码器再从量化向量重建图像。

### 13.9 向量量化 PyTorch 示例

```python
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, hidden_size: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 3:
            raise ValueError("latents 应为 [B, N, D]")

        codebook = self.codebook.weight
        distances = (
            latents.square().sum(dim=-1, keepdim=True)
            - 2 * latents @ codebook.t()
            + codebook.square().sum(dim=-1)
        )
        token_ids = distances.argmin(dim=-1)
        quantized = self.codebook(token_ids)
        return quantized, token_ids
```

### 13.10 Straight-Through Estimator

`argmin` 不可导。

训练时常使用 straight-through estimator：

$$
z_q^{st}=z_e+\operatorname{sg}(z_q-z_e),
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。

前向值等于 $z_q$。

反向对编码器近似使用恒等梯度。

还需 codebook loss 与 commitment loss 稳定训练。

### 13.11 音频 tokenization

音频可直接切成连续帧特征。

也可通过神经音频 codec 变成离散 token。

常见 codec 使用残差向量量化 RVQ。

第一个 codebook 量化主残差。

后续 codebook 继续量化前一层残差。

每个时间步因此可能对应多个 codebook ID。

### 13.12 RVQ 递推

设输入 latent 为 $z$。

初始残差：

$$
r_0=z.
$$

第 $q$ 层选择：

$$
e_q=Q_q(r_{q-1}).
$$

更新残差：

$$
r_q=r_{q-1}-e_q.
$$

最终量化向量：

$$
\hat z=\sum_{q=1}^{Q}e_q.
$$

增加 codebook 层数通常提高重建质量，也增加 token 率。

### 13.13 音频 token 率

设 codec 帧率为 $f$ 帧/秒。

每帧使用 $Q$ 个 codebook。

若展平为单序列，token 率约为：

$$
R=fQ\quad\text{tokens/s}.
$$

例如 50 Hz、8 个 codebook 会产生约 400 token/s。

一分钟音频约 24K token。

这远高于同长度语音转写文本的 token 数。

端到端语音模型必须精细控制压缩与上下文预算。

### 13.14 音频离散 token 排列

多 codebook token 可按时间优先排列：

```text
(t0,q0), (t0,q1), ..., (t1,q0), (t1,q1), ...
```

也可按 codebook 流分组：

```text
q0 全时间步, q1 全时间步, ...
```

排列影响自回归依赖、并行生成与位置编码。

协议必须与训练目标一致。

### 13.15 视频 tokenization

视频同时具有空间和时间维度。

输入可记为：

$$
x\in\mathbb{R}^{T\times H\times W\times C}.
$$

使用 tubelet 大小 $P_t\times P_h\times P_w$。

token 数为：

$$
N=\frac{T}{P_t}\frac{H}{P_h}\frac{W}{P_w}.
$$

视频 token 数增长非常快。

通常需要降帧、时空池化、关键帧采样或 token 压缩。

### 13.16 视频采样陷阱

均匀采样可能错过短暂关键事件。

只取关键帧可能丢失动作顺序。

按镜头切分可减少重复帧。

长视频还需要时间戳与片段边界 token。

训练和推理的 FPS 不一致会造成分布偏移。

变帧率视频应先明确时间基准。

### 13.17 模态边界 token

统一序列常使用：

```text
<bos>
<user>
请描述这张图片：
<image_start>
<image_patch> × N
<image_end>
<assistant>
```

`<image_patch>` 可能只是占位 ID。

模型前向时对应位置的文本 embedding 会被视觉 embedding 替换。

也有架构通过 cross-attention 单独读取视觉序列。

两种方案的 tokenizer 协议不同。

### 13.18 占位符替换实现

```python
def replace_placeholder_embeddings(
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    mask = input_ids.eq(image_token_id)
    expected = int(mask.sum().item())
    supplied = image_embeddings.numel() // image_embeddings.shape[-1]

    if expected != supplied:
        raise ValueError(
            f"图像占位符数量 {expected} 与视觉 token 数 {supplied} 不一致"
        )

    output = text_embeddings.clone()
    output[mask] = image_embeddings.reshape(-1, image_embeddings.shape[-1])
    return output
```

这段逻辑的关键不是赋值本身，而是严格校验数量。

静默截断会导致图文错位，且很难排查。

### 13.19 位置编码

文本通常是一维顺序。

图像天然具有二维坐标。

视频具有三维时空坐标。

多模态模型可选择：

- 把所有 token 展平成一维位置。
- 对视觉 token 使用二维位置 embedding。
- 对视频使用时间、行、列分解位置。
- 使用多轴 RoPE。
- 通过 cross-attention 保持独立坐标系。

位置方案应与动态分辨率和 padding 策略兼容。

### 13.20 Attention mask

多模态序列不一定使用普通全因果 mask。

一种常见策略是：

视觉输入 token 彼此双向可见。

后续文本生成 token 可看见全部视觉 token 和历史文本。

未来文本仍不可见。

也有模型把视觉编码器与语言解码器分离。

此时视觉侧用双向 attention，文本侧用因果 attention，二者通过 cross-attention 相连。

mask 错误会造成训练泄漏或推理不一致。

### 13.21 Label mask

输入中的图像占位 token 通常不参与文本 next-token loss。

用户 prompt 通常也可被设为 ignore。

仅 assistant 回复位置计算监督损失。

```python
def build_labels(
    input_ids: torch.Tensor,
    supervised_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if input_ids.shape != supervised_mask.shape:
        raise ValueError("input_ids 与 supervised_mask 形状不一致")
    labels = input_ids.clone()
    labels[~supervised_mask] = ignore_index
    return labels
```

### 13.22 模态词表是否共享

文本、图像、音频离散 ID 可以放入一个大整数空间。

也可以各自使用局部 ID，再通过模态 embedding 区分。

共享空间的优点是统一自回归接口。

缺点是词表巨大、采样约束复杂。

局部空间更模块化，但需要额外路由头或预测头。

连续视觉 embedding 则绕过离散视觉词表。

没有一种方案适合所有任务。

### 13.23 受约束采样

若模型同时生成文本与音频 token，当前位置允许的 token 类型可能不同。

可对 logits 添加模态 mask：

$$
\tilde\ell_i=
\begin{cases}
\ell_i,& i\in V_{allowed},\\
-\infty,& i\notin V_{allowed}.
\end{cases}
$$

softmax 后非法 token 概率为 0。

状态机根据控制 token 决定允许集合。

这是协议正确性的关键保障。

### 13.24 PyTorch logits mask

```python
def mask_logits(
    logits: torch.Tensor,
    allowed_ids: torch.Tensor,
) -> torch.Tensor:
    if allowed_ids.dtype != torch.long:
        raise ValueError("allowed_ids 必须是 torch.long")

    masked = torch.full_like(logits, float("-inf"))
    masked.index_copy_(-1, allowed_ids, logits.index_select(-1, allowed_ids))
    return masked
```

若某状态没有任何 allowed ID，softmax 会产生 NaN。

状态机必须在采样前验证非空集合。

### 13.25 多图输入

多张图需要独立边界与映射关系。

例如：

```text
<image_start:0> ... <image_end:0>
文本引用第一张图
<image_start:1> ... <image_end:1>
文本引用第二张图
```

不一定真的为每个索引创建新 token。

也可使用统一边界 token，加外部段 ID 标识图像序号。

服务端应验证占位符数量等于上传图像数量。

还要验证每张图的 patch 数与对应占位区间一致。

### 13.26 OCR 与细粒度视觉 token

OCR 需要保留局部字符细节。

过大 patch 会把多个小字符混在同一 token。

过度缩放会让笔画消失。

常见改进包括高分辨率 tile、区域裁剪和多尺度特征。

但视觉 token 增多会挤占文本上下文。

因此 OCR 任务常是视觉质量与 token 预算的压力测试。

### 13.27 坐标 token

一些模型把边界框坐标离散成 token。

归一化坐标 $x\in[0,1]$ 可量化为：

$$
q(x)=\operatorname{round}(x(K-1)).
$$

再映射为 `<loc_0>` 到 `<loc_{K-1}>`。

反量化为：

$$
\hat x=\frac{q(x)}{K-1}.
$$

量化误差上界约为 $1/(2(K-1))$。

坐标顺序、原点和宽高归一化必须统一。

### 13.28 工具调用与结构 token

多模态聊天往往同时支持工具调用。

图像边界、角色边界和 JSON 工具结构都依赖控制 token。

协议应避免控制 token 在普通用户文本中被误解释。

训练模板、推理模板、日志脱敏器和评估脚本必须共享同一协议。

这也是广义 tokenization 工程的一部分。

### 13.29 多模态测试矩阵

| 测试 | 期望 |
|---|---|
| 纯文本 | 不产生视觉占位符 |
| 单图短文本 | 占位数与视觉 token 数一致 |
| 多图 | 每图边界和顺序正确 |
| 超大图 | 按预算缩放或拒绝 |
| 透明 PNG | 通道转换明确 |
| EXIF 旋转 | 方向被正确处理 |
| 灰度图 | 通道扩展一致 |
| 损坏图片 | 在编码前报错 |
| 空音频 | 拒绝或产生定义好的空表示 |
| 超长音频 | 分段或拒绝 |
| 变帧率视频 | 时间采样可解释 |
| 非法控制 token | 转义或拒绝 |

### 13.30 常见错误

错误一：把视觉连续 embedding ID 与文本离散 token ID 混为一谈。

错误二：图像 patch 数改变后仍使用固定占位符数量。

错误三：只检查总占位数，不检查每张图的对应区间。

错误四：忽略 EXIF 方向，导致视觉内容旋转。

错误五：训练时一种图像缩放，推理时另一种。

错误六：音频 codebook 排列顺序不一致。

错误七：视觉位置错误地参与文本 labels。

错误八：状态机允许在音频区生成任意文本 token。

错误九：无限接收像素导致显存拒绝服务风险。

错误十：多模态模板在客户端和服务端各维护一份并逐渐漂移。

### 13.31 本节练习

1. 计算 448×448 图像在 patch size 14 下的视觉 token 数。

2. 为任意尺寸图像实现 padding 后 patchify。

3. 实现每张图独立占位区间校验。

4. 推导 RVQ 的 token rate 与码率关系。

5. 设计文本/图像/音频统一 ID 空间。

6. 为统一空间实现受约束 logits mask。

7. 构造视觉 token label 泄漏的反例。

8. 比较 flatten 1D 位置与 2D 位置编码的优缺点。

---

## 14. 工业实践：从训练到上线

### 14.1 Tokenizer 是模型 ABI

Tokenizer 定义文本与 token ID 之间的二进制协议。

模型 embedding 的第 $i$ 行对应 tokenizer 的第 $i$ 个 token。

只要 ID 含义改变，模型参数就可能被错误解释。

因此 tokenizer 类似模型的 ABI。

它不是可以随意替换的前处理脚本。

模型版本、tokenizer 版本和聊天模板版本应绑定发布。

### 14.2 最小发布物

一个可部署 tokenizer 至少需要：

- 完整模型文件。
- 特殊 token 映射。
- 归一化配置。
- 预分词配置。
- 聊天模板。
- 最大长度约定。
- 文件哈希。
- 版本号。
- 许可证与来源说明。

若使用外部代码，还要固定依赖版本。

### 14.3 哈希校验

```python
from hashlib import sha256
from pathlib import Path


def file_sha256(path: str) -> str:
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

训练日志记录哈希。

checkpoint metadata 记录哈希。

推理镜像构建时记录哈希。

服务启动时再次校验哈希。

这比只记录文件名可靠。

### 14.4 版本策略

语义版本可按以下原则使用：

- patch：实现修复，但 token ID 与输出完全不变。
- minor：新增兼容 API，不改变已有编码结果。
- major：词表、规则、归一化或模板发生不兼容变化。

只要同一文本的 token ID 变化，通常应视为不兼容升级。

模型也应同步产生新版本。

### 14.5 训练语料配比

tokenizer 训练语料影响词表槽位分配。

应覆盖模型预计服务的语言和领域。

典型类别包括：

- 自然语言网页。
- 书籍与论文。
- 代码。
- 数学公式。
- 表格与 JSON。
- 对话模板。
- 目标小语种。
- emoji 与社交文本。

不需要把全部预训练语料都用于 tokenizer 训练。

高质量、分布匹配的样本通常足够。

### 14.6 文档边界

训练 tokenizer 时不能无意跨文档合并。

模型预训练时也需明确文档边界 token。

若把所有文档直接拼接，末尾与下一篇开头可能产生虚假共现。

可在文档间插入 EOS。

也可在 BPE 统计时禁止跨文档边界。

边界策略应在数据管线中显式实现。

### 14.7 词表大小搜索

不要仅凭行业惯例选 32K 或 128K。

可训练多个候选 tokenizer。

对验证语料比较：

- 平均字符/token。
- P50、P95、P99 token 长度。
- 各语言压缩率。
- 代码压缩率。
- 数字切分模式。
- byte fallback 比例。
- embedding 参数增量。
- 端到端吞吐。

最终选择是模型质量与系统成本的联合决策。

### 14.8 压缩率指标

常用字符压缩率：

$$
R_{char}=\frac{\#\text{Unicode characters}}{\#\text{tokens}}.
$$

常用字节压缩率：

$$
R_{byte}=\frac{\#\text{UTF-8 bytes}}{\#\text{tokens}}.
$$

两者用于多语言比较时含义不同。

中文字符通常占多个 UTF-8 字节。

报告必须注明分母分子定义。

### 14.9 Token tax

设基准语言每语义样本平均 token 数为 $L_{base}$。

目标语言为 $L_l$。

可定义相对 token tax：

$$
\tau_l=\frac{L_l}{L_{base}}-1.
$$

$\tau_l=0.5$ 表示同类内容多消耗约 50% token。

语义等价样本难以严格构造。

可使用平行语料或同任务模板近似比较。

### 14.10 统计脚本

```python
from statistics import mean, median


def tokenization_report(texts: list[str], encode) -> dict[str, float]:
    if not texts:
        raise ValueError("texts 不能为空")

    token_lengths = [len(encode(text)) for text in texts]
    char_lengths = [len(text) for text in texts]
    byte_lengths = [len(text.encode("utf-8")) for text in texts]

    total_tokens = sum(token_lengths)
    if total_tokens == 0:
        raise ValueError("编码结果没有 token")

    ordered = sorted(token_lengths)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))

    return {
        "samples": float(len(texts)),
        "mean_tokens": mean(token_lengths),
        "median_tokens": median(token_lengths),
        "p95_tokens": float(ordered[p95_index]),
        "chars_per_token": sum(char_lengths) / total_tokens,
        "bytes_per_token": sum(byte_lengths) / total_tokens,
    }
```

### 14.11 数字策略

数字切分直接影响算术和表格任务。

需要检查：

```text
0
42
2026
3.14159
-1.2e-5
1,000,000
2026-07-18
+86-13800000000
```

可能策略包括逐位数字、固定三位分组和语料驱动 BPE。

逐位数字规则一致，但序列较长。

大块数字 token 更短，却可能让位值泛化变差。

应通过算术评测与吞吐共同选择。

### 14.12 代码策略

代码 tokenizer 应关注：

- 缩进空格。
- Tab。
- 换行。
- 常见运算符。
- 标识符命名风格。
- Unicode 标识符。
- 长字符串。
- 注释。
- 路径与 URL。
- 十六进制和二进制数字。

若归一化折叠空格，Python 代码语义可能直接改变。

代码场景通常不能照搬自然语言空白归一化。

### 14.13 特殊 token 添加

在已有模型上追加特殊 token 时：

1. 保持所有旧 token ID 不变。
2. 在词表尾部追加新 ID。
3. 扩展输入 embedding。
4. 若 lm head 不共享，也扩展输出矩阵。
5. 初始化新行。
6. 微调让模型学会新 token。
7. 更新模板与生成约束。
8. 保存新模型与新 tokenizer。

只改 tokenizer 不改 embedding 会越界或语义错误。

### 14.14 扩展 embedding

```python
from torch import nn


def extend_embedding(
    old: nn.Embedding,
    new_vocab_size: int,
) -> nn.Embedding:
    old_vocab_size, hidden_size = old.weight.shape
    if new_vocab_size < old_vocab_size:
        raise ValueError("此函数不支持缩小词表")

    new = nn.Embedding(
        new_vocab_size,
        hidden_size,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )

    with torch.no_grad():
        new.weight[:old_vocab_size].copy_(old.weight)
        if new_vocab_size > old_vocab_size:
            mean_row = old.weight.mean(dim=0, keepdim=True)
            new.weight[old_vocab_size:].copy_(mean_row)

    return new
```

均值初始化只是安全基线。

是否更优要通过微调和评估验证。

### 14.15 训练 labels 对齐

自回归训练通常用右移标签：

$$
\text{input}=x_{0:L-1},\quad
\text{target}=x_{1:L}.
$$

聊天模板还需对非 assistant 区域设置 ignore index。

特殊 token 是否参与 loss 必须明确。

EOS 通常应参与监督，否则模型不易学会停止。

PAD 通常不参与监督。

图像占位符通常不作为文本预测目标。

### 14.16 截断策略

简单从右侧截断可能删掉 assistant 标签或答案。

简单从左侧截断可能删掉 system prompt。

多轮对话可按整轮删除最旧消息。

检索增强可优先截断低相关文档。

多模态输入需避免只截掉一半视觉占位区间。

截断应在结构层完成，而不是对最终 ID 数组盲切。

### 14.17 长度估算

API 常在真正编码前估算 token 数。

字符数除以常数只能作为粗略估计。

中文、代码、emoji 和 Base64 的比例差异很大。

计费或强限制必须以真实 tokenizer 结果为准。

前端估算可以偏保守。

服务端必须重新编码并执行权威校验。

### 14.18 缓存

可缓存稳定系统提示词的编码结果。

可缓存固定工具 schema 的 token IDs。

不要缓存包含用户隐私的明文 key，除非缓存隔离和生命周期明确。

缓存 key 至少包含：

- tokenizer 版本。
- 模板版本。
- 原始文本或其安全哈希。
- BOS/EOS 配置。
- 截断参数。

版本缺失会导致升级后复用旧 token IDs。

### 14.19 并发与线程安全

只读 tokenizer 通常可安全并发，但应以具体实现为准。

某些绑定内部有全局线程池。

过度嵌套并行可能造成 CPU 争抢。

批量编码常比逐条提交线程池更高效。

服务启动后避免动态修改词表。

若需要热升级，使用不可变实例和原子路由切换。

### 14.20 性能基准

基准应同时覆盖：

- 短对话。
- 长文档。
- 多语言。
- 代码。
- 大量小 batch。
- 少量大 batch。
- 单线程。
- 多线程。
- 编码。
- 解码。

报告吞吐 tokens/s 和文本 bytes/s。

还应报告 P50、P95、P99 延迟。

只报告平均值会掩盖长尾。

### 14.21 安全边界

tokenizer 也有安全风险：

- 超长输入导致 CPU 消耗。
- 特殊 token 注入破坏聊天边界。
- Unicode 混淆字符绕过过滤。
- 归一化前后审核结果不一致。
- 恶意重复字符串触发最坏复杂度。
- 图像像素炸弹耗尽内存。
- 日志直接输出敏感解码文本。

输入限制应在昂贵处理前执行。

审核策略要明确在原始文本还是规范化文本上运行。

高风险场景可两者都检查。

### 14.22 Unicode 安全

需要覆盖：

- 零宽连接符。
- 零宽非连接符。
- 双向文本控制字符。
- 同形异义字符。
- 组合字符长链。
- 不成对代理项。
- 非字符码点。
- 变体选择符。

不能简单删除所有不可见字符。

它们在部分语言和 emoji 中有合法用途。

策略应按产品风险与语言支持范围制定。

### 14.23 线上监控

建议监控：

- 请求 token 数分布。
- 输入 bytes/token。
- 各语言 token tax。
- `<unk>` 或 byte fallback 比例。
- 截断率。
- 特殊 token 触发率。
- 编码失败率。
- P95 编码延迟。
- tokenizer 版本占比。
- 客户端与服务端计数差异。

分布突变可能来自流量变化，也可能来自版本漂移。

### 14.24 灰度发布

新 tokenizer 通常无法直接给旧模型使用。

若只是实现优化且输出完全一致，可做 shadow 对比。

对同一输入同时运行旧实现和新实现。

逐条比较 token IDs 与解码结果。

达到严格一致后再切换流量。

若词表本身变化，则应作为新模型整体灰度。

### 14.25 Golden cases

保存一组不可变 golden cases：

```text
空字符串
单个空格
连续空格
换行与 Tab
ASCII
简体中文
繁体中文
日文假名
阿拉伯文
组合重音
emoji ZWJ 序列
代码缩进
JSON
URL
超长数字
特殊 token 字面量
```

每个 case 保存期望 IDs 和期望解码结果。

升级依赖或平台时执行严格回归。

### 14.26 Round-trip 属性

若无不可逆归一化，理想属性为：

$$
\operatorname{decode}(\operatorname{encode}(x))=x.
$$

若存在归一化，应改为：

$$
\operatorname{decode}(\operatorname{encode}(x))=N(x).
$$

属性测试可随机生成 Unicode 字符串。

但要限制长度并正确处理非法码点。

### 14.27 ID 稳定性属性

对同一 tokenizer 版本：

$$
\operatorname{encode}(x)\text{ 必须确定且稳定。}
$$

除非明确启用训练期子词采样。

评估、缓存、计费和签名都依赖确定性。

随机模式必须通过不同 API 或显式参数开启。

### 14.28 迁移检查表

1. 冻结旧 tokenizer 产物与哈希。
2. 准备新模型和新 tokenizer。
3. 对齐特殊 token ID。
4. 检查 embedding 与 lm head 尺寸。
5. 重跑 golden cases。
6. 重跑语言压缩率报告。
7. 重跑核心任务评估。
8. 检查聊天模板。
9. 检查截断策略。
10. 检查计费 token 数。
11. 灰度新模型端点。
12. 保留可回滚版本。

### 14.29 工业事故案例：ID 漂移

现象：服务输出突然出现无意义字符。

根因：词表 JSON 被重新排序后按新顺序分配 ID。

模型 embedding 仍按旧 ID 训练。

结果：每个输入 token 查到错误向量。

预防：永远显式保存 ID，不按文件迭代顺序重建。

预防：启动时检查词表哈希和 embedding 行数。

### 14.30 工业事故案例：模板重复 BOS

现象：微调 loss 正常，但生成质量下降。

根因：聊天模板已插入 BOS，编码 API 又自动添加一次 BOS。

训练输入以两个 BOS 开头，推理输入只有一个。

预防：把模板渲染与 special-token 添加做端到端 golden test。

预防：日志同时记录可视化 token 与 ID。

### 14.31 工业事故案例：归一化漂移

现象：某些全角数字和组合字符线上表现异常。

根因：训练使用 NFKC，移动端先做 NFC，服务端又跳过模型规范化器。

预防：只有一个权威归一化实现。

预防：客户端不要假设本地 token 数等于服务端计费结果。

### 14.32 本节练习

1. 为 tokenizer 产物设计 manifest JSON。

2. 实现启动时哈希与词表大小校验。

3. 制作至少 20 条 multilingual golden cases。

4. 比较 32K、64K、128K 词表的端到端训练成本。

5. 设计 tokenizer shadow 流量对比指标。

6. 为特殊 token 注入编写安全测试。

7. 设计长对话的结构化截断算法。

8. 写出 tokenizer 回滚预案。

---

## 15. 面试题、案例与综合自测

### 15.1 基础概念题

#### 问题 1：为什么不直接按词切分？

按词切分会产生巨大词表，并对新词、拼写变化和黏着语产生 OOV。

子词方法用有限词表组合开放词汇，是词表大小与序列长度的折中。

#### 问题 2：为什么不全部按字符切分？

字符词表小且无普通 OOV，但序列很长，高频语义片段不能复用单个表示。

子词能把常见片段压缩成更少 token。

#### 问题 3：BPE 每轮选择什么？

选择当前语料表示中加权频次最高的相邻符号对。

合并后必须更新受影响的 pair 频次。

#### 问题 4：BPE 的训练和编码有什么不同？

训练从语料学习词表与有序合并规则。

编码固定使用这些规则，不再根据单条输入重算频次。

#### 问题 5：BBPE 为什么没有 OOV？

所有输入都可先编码成 0 到 255 的字节序列。

只要 256 个基础 byte 都可表示，任何字节串都有回退路径。

#### 问题 6：byte token 一定是合法字符吗？

不一定。

多字节 UTF-8 字符可能被拆开，单个 token 只含不完整字节片段。

#### 问题 7：WordPiece 与 BPE 的核心差异？

经典 BPE 直接选择最高频 pair。

WordPiece 更强调合并对语言模型似然或相对关联性的改善。

#### 问题 8：SentencePiece 是算法还是工具？

它是一套语言无关 tokenizer 框架。

它可训练 BPE 或 Unigram 等模型，并统一处理规范化与空白。

#### 问题 9：Unigram 为什么需要动态规划？

同一字符串存在指数多种潜在切分。

动态规划可在切分 DAG 上计算最优路径或所有路径概率和。

#### 问题 10：`▁` 有什么作用？

它把空白边界显式编码为普通可见符号。

因此分词和解码可以保留词首与空格信息。

### 15.2 公式推导题

#### 问题 11：写出 BPE pair 频次公式。

$$
C(a,b)=\sum_w F(w)n_w(a,b).
$$

必须乘预分词单元的语料频次 $F(w)$。

#### 问题 12：写出 Unigram 切分概率。

$$
P(s)=\prod_{z\in s}p(z).
$$

log 概率是各 token log 概率之和。

#### 问题 13：写出字符串边缘概率。

$$
P(x)=\sum_{s\in\mathcal{S}(x)}P(s).
$$

它对所有合法切分求和，而不只取 Viterbi 路径。

#### 问题 14：写出 Viterbi 递推。

$$
d[j]=\min_{i<j,x_{i:j}\in V}\left(d[i]-\log p(x_{i:j})\right).
$$

同时保存前驱即可回溯 token 序列。

#### 问题 15：为什么需要 log-sum-exp？

长序列概率连乘会数值下溢。

log-sum-exp 在对数域稳定计算多个路径概率之和。

#### 问题 16：词表扩大对 embedding 参数量的影响？

$$
\Delta P=(|V_{new}|-|V_{old}|)d.
$$

若 lm head 不共享，还要增加相近规模的输出参数。

#### 问题 17：图像 patch 数如何计算？

对不重叠方形 patch：

$$
N=(H/P)(W/P).
$$

前提是尺寸可整除；否则需裁剪或 padding。

#### 问题 18：音频 codec token rate 如何估算？

若帧率为 $f$，每帧 $Q$ 个 codebook，则：

$$
R=fQ\ \text{tokens/s}.
$$

#### 问题 19：多语言温度采样公式？

$$
q_l=\frac{p_l^\alpha}{\sum_jp_j^\alpha}.
$$

$0<\alpha<1$ 会相对提升低资源语言占比。

#### 问题 20：token tax 如何定义？

一种定义是：

$$
\tau_l=L_l/L_{base}-1.
$$

必须说明样本是否语义等价以及 token 长度统计口径。

### 15.3 实现题

#### 问题 21：如何保证 BPE tie-break 可复现？

将排序键定义为 `(-frequency, left_id, right_id)`。

不能依赖哈希表的偶然迭代顺序。

#### 问题 22：如何避免跨词边界合并？

训练时把每个预分词单元保存为独立符号序列。

pair 统计与替换都限制在序列内部。

#### 问题 23：如何保证解码可逆？

保存空格/边界信息，使用一一对应的 byte 映射，并在拼接完整字节后 UTF-8 解码。

若归一化不可逆，只能保证恢复规范化文本。

#### 问题 24：如何测试 tokenizer？

至少测试 golden IDs、round-trip、特殊 token、Unicode、多语言、代码、空白和超长输入。

还要做版本哈希与模型 embedding 尺寸校验。

#### 问题 25：新增特殊 token 后模型要改什么？

在词表尾部追加 ID，扩展 input embedding 和必要的 lm head，并微调新行。

所有旧 ID 必须保持不变。

#### 问题 26：如何实现 Unigram 最优切分？

把字符串位置建成 DAG 节点，把可匹配 piece 建成带负 log 概率的边。

用动态规划求最短路径并回溯。

#### 问题 27：如何实现随机子词切分？

用 forward/backward 得到边的条件概率，按温度调整后从当前位置采样下一条边。

训练固定种子以便必要时复现。

#### 问题 28：如何加速大词表匹配？

用 trie 从每个起点扫描所有可匹配 token。

避免对每个位置遍历完整词表。

#### 问题 29：如何拼接图像 embedding？

先在文本 ID 中放固定占位符，再校验占位符数量与视觉 token 数一致，最后替换对应 embedding。

不能静默截断或广播。

#### 问题 30：如何屏蔽非法模态 token？

按解码状态机生成 allowed ID 集合，把其他 logits 设为 $-\infty$。

采样前校验允许集合非空。

### 15.4 场景分析题

#### 问题 31：中文 token 数明显高于英文，如何排查？

先使用平行语料计算字符/token 与 token tax。

再检查训练语料语言配比、字符覆盖率、byte fallback 比例和词表中的中文片段占比。

#### 问题 32：新版本 tokenizer 吞吐更高，能否直接替换？

只有对全部 golden 与 shadow 样本输出完全相同 token IDs 才能直接替换旧模型前处理。

否则它属于模型协议变更。

#### 问题 33：用户输入 `<assistant>` 怎么办？

普通文本中的可见字符串不应自动获得控制语义。

应转义、分离控制 token 通道或由服务端模板安全插入。

#### 问题 34：为什么同一句话 decode 后空格变化？

可能是归一化折叠空格、预分词丢失空格或 `▁` 恢复逻辑错误。

先明确期望是原文 round-trip 还是规范化文本 round-trip。

#### 问题 35：为什么 tokenizer 有 `<unk>` 仍声称无 OOV？

存在 `<unk>` 可能只是兼容协议。

若普通输入总能回退到 bytes，就不会实际产生未知 token。

#### 问题 36：代码模型为什么保留空白？

缩进、换行和某些空格直接参与语法或可读结构。

折叠空白会改变 Python 等语言的语义。

#### 问题 37：大词表一定更好吗？

不一定。

它缩短序列，却增加 embedding、lm head、通信和 softmax 成本，也可能给噪声片段分配槽位。

#### 问题 38：词表训练需要全部预训练数据吗？

通常不需要。

使用高质量、去重、分布匹配的代表性样本即可估计稳定片段频率。

#### 问题 39：为什么归一化会影响安全？

过滤器若看原文而模型看规范化文本，攻击者可能利用等价字符或不可见字符制造视图差异。

高风险流程应统一规范化并记录审核口径。

#### 问题 40：为什么 tokenizer 难以在预训练后修改？

token ID 与 embedding 行一一绑定。

改变切分会让模型看到不同序列分布，通常需要扩展参数并继续训练甚至重新预训练。

### 15.5 Debug 题

#### 问题 41：所有 BBPE merge 都为空，最可能的原因？

可能把每个 byte 都包装成长度 1 的独立序列，导致序列内部没有相邻 pair。

应让一个预分词单元包含多个 byte 符号。

#### 问题 42：训练结果每次不同，如何定位？

检查同频 pair tie-break、语料遍历顺序、并行计数归并、随机采样种子和依赖版本。

输出前几百条 merge 做差分。

#### 问题 43：中文 decode 报 UTF-8 错误，为什么？

可能对每个 byte token 单独解码，或 token 序列被截断在多字节字符中间。

应先拼接完整 bytes，再执行 UTF-8 解码。

#### 问题 44：模型生成永不停止，tokenizer 可能哪里错？

检查 EOS ID、模板是否插入 EOS、训练 label 是否监督 EOS、生成配置的 eos_token_id 是否一致。

也要检查多个 EOS 别名。

#### 问题 45：padding 后 loss 异常，如何排查？

确认 pad ID、attention mask、label ignore index 和 padding 方向。

pad ID 不等于 ignore index，二者用途不同。

#### 问题 46：多图模型报视觉 token 数不匹配，如何定位？

逐图记录预处理尺寸、tile 数、patch 数和占位区间。

不要只比较 batch 总数。

#### 问题 47：移动端和服务端 token 计费不同，为什么？

可能使用不同 tokenizer 版本、归一化器、模板、BOS/EOS 或截断参数。

服务端应是权威计费方。

#### 问题 48：词表大小正确但模型输出乱码，为什么？

大小一致不代表 ID 语义一致。

应比较 tokenizer 哈希、关键 token ID 和完整 ID 映射。

#### 问题 49：Unigram loss 变成 `inf`，为什么？

某些训练句子没有合法切分，或概率在非 log-space 下溢。

检查基础字符覆盖和 log-sum-exp。

#### 问题 50：子词采样导致评测不稳定，怎么修？

评估和服务端关闭采样，使用确定性的最优切分。

只在训练数据增强路径显式启用采样。

### 15.6 系统设计题

#### 问题 51：设计一个 tokenizer 服务。

核心组件包括不可变模型加载、批量编码、版本路由、输入限制、指标、哈希校验和 golden 回归。

若只是单模型进程内调用，优先直接使用库，不必额外拆微服务。

#### 问题 52：如何支持多个模型？

以模型版本映射到不可变 tokenizer 实例。

请求必须显式携带模型名，缓存 key 也必须包含 tokenizer 版本。

#### 问题 53：如何做 tokenizer 热升级？

后台加载并校验新实例，运行健康检查后原子切换路由。

旧实例保留到在途请求完成，失败则回滚。

#### 问题 54：如何限制资源攻击？

在解析前限制请求字节数、图像像素、音视频时长和样本数。

设置编码超时、并发限制和内存上限。

#### 问题 55：如何监控语言公平性？

按语言分桶监控 tokens/character、tokens/byte、截断率、时延和成本。

使用稳定平行语料做离线回归。

#### 问题 56：如何选择词表大小？

训练多个候选，联合比较压缩率、下游质量、embedding 参数、softmax 成本和端到端吞吐。

不要只优化单一指标。

#### 问题 57：如何设计多模态序列协议？

定义模态边界 token、占位符数量、位置 ID、attention mask、label mask、截断和非法状态处理。

协议需版本化并端到端测试。

#### 问题 58：如何支持工具调用？

由服务端模板插入受保护控制 token，使用语法或状态机约束生成，并把普通用户文本与控制通道分离。

JSON 文本本身不等于安全协议。

#### 问题 59：如何做兼容性测试？

旧实现与新实现对同一大样本 shadow 编码，严格比较 IDs、解码、错误类型和特殊 token 行为。

任何差异都需分类解释。

#### 问题 60：何时考虑 tokenizer-free 模型？

当 byte/patch 级建模带来的鲁棒性或通用性收益足以覆盖更长序列与更高算力成本时。

仍需通过端到端质量和成本实验验证。

### 15.7 快速判断题

61. BPE 只适用于英文。——错误，多语言都可使用。

62. Byte-level BPE 的基础词表至少覆盖 256 个 byte。——正确。

63. BPE merge 规则可任意排序。——错误，顺序决定编码。

64. Unigram 只能输出一种切分。——错误，它可产生 N-best 或采样切分。

65. SentencePiece 一定使用 Unigram。——错误，也可使用 BPE。

66. 大词表总会减少总训练 FLOPs。——错误，还会增加投影和参数成本。

67. `<pad>` ID 一定是 0。——错误，取决于模型配置。

68. EOS 通常应参与自回归训练监督。——正确。

69. 解码可逆意味着归一化一定可逆。——错误。

70. UTF-8 中文字符通常是单字节。——错误，常见中文通常是三字节。

71. 单个 byte token 必然能显示为合法 Unicode。——错误。

72. Tokenizer 可以不随模型 checkpoint 版本化。——错误。

73. 多模态视觉 token 一定是离散 ID。——错误，也可能是连续 embedding。

74. Patch 越小，视觉 token 通常越多。——正确。

75. RVQ codebook 越多，token rate 通常越高。——正确。

76. 图像占位数不匹配时可静默截断。——错误，应立即报错。

77. 所有 logits 都屏蔽后 softmax 仍然正常。——错误，可能产生 NaN。

78. 代码归一化可以安全删除全部空格。——错误。

79. Token 数可以用字符数精确推算。——错误。

80. Tokenizer 文件哈希有助于发现版本漂移。——正确。

### 15.8 综合案例一：设计中英代码共享 tokenizer

需求：支持中文、英文、Python、SQL 和 JSON。

步骤一：收集去重且质量可控的代表语料。

步骤二：按预期线上分布设基础比例。

步骤三：对中文做适度温度上采样，避免 token tax 过高。

步骤四：保留换行、缩进、常用运算符与 JSON 标点。

步骤五：选择 byte fallback 以覆盖罕见字符。

步骤六：训练 32K、64K、96K 三个候选。

步骤七：比较各领域 bytes/token 与 P95 长度。

步骤八：估算 embedding 和 lm head 成本。

步骤九：运行算术、代码补全和多语言下游评测。

步骤十：冻结特殊 token ID 与聊天模板。

验收不能只看平均压缩率。

还要看最差语言、长尾输入与代码语义保真。

### 15.9 综合案例二：给旧模型增加 `<tool_call>`

约束：旧 token ID 不得改变。

方案：在词表尾部追加 `<tool_call>` 与 `<tool_result>`。

扩展 embedding 与未共享的 lm head。

新行可用旧 embedding 均值或相关 token 均值初始化。

构造带工具调用格式的监督数据继续微调。

服务端使用状态机限制控制 token 出现位置。

普通用户文本中的相同字面量必须转义或按普通文本编码。

回归旧任务，确认新增 token 没有破坏普通生成。

发布新模型与新 tokenizer 的绑定版本。

### 15.10 综合案例三：多模态上下文超预算

输入包含 10 张高分辨率图片和长对话。

不能对最终 token ID 数组直接截断。

先按图片重要性与像素预算选择 tile 数。

为每张图保留完整起止边界。

按整轮删除最旧低优先级对话。

保留 system prompt 与当前用户问题。

重新计算视觉与文本 token 总长。

若仍超限，明确拒绝并返回可解释错误。

不可留下半个图像占位区间。

### 15.11 综合案例四：线上 token 数突增

观察：P95 输入 token 数一夜增加 40%。

排查一：按客户端、语言、模型版本分桶。

排查二：比较 bytes/token，判断是文本变长还是压缩率变化。

排查三：检查 tokenizer 与模板版本占比。

排查四：检查是否重复插入工具 schema 或历史消息。

排查五：检查新流量是否包含 Base64、日志或爬虫内容。

排查六：对异常样本做脱敏后离线重放。

修复后加入对应 golden case 与监控告警。

### 15.12 手撕代码任务清单

1. 实现带词频权重的 BPE pair 统计。

2. 实现不重叠 pair 合并。

3. 实现确定性 tie-break。

4. 实现 merge-rank 编码。

5. 实现 bytes round-trip 解码。

6. 实现 Unigram Viterbi。

7. 实现 log-space forward。

8. 实现 N-best 切分。

9. 实现 trie piece 匹配。

10. 实现温度子词采样。

11. 实现 batch padding。

12. 实现 attention mask。

13. 实现 label mask。

14. 实现图像 patch tokenizer。

15. 实现向量量化最近邻。

16. 实现视觉占位 embedding 替换。

17. 实现模态 logits mask。

18. 实现文件 SHA-256。

19. 实现压缩率报告。

20. 实现 golden case 回归器。

### 15.13 终极复习表

| 主题 | 一句话结论 |
|---|---|
| BPE | 从小词表出发，反复合并高频相邻 pair |
| BBPE | 以 256 bytes 兜底，普通文本无 OOV |
| WordPiece | 用似然相关准则选择子词合并 |
| SentencePiece | 统一规范化、空白处理与子词模型的框架 |
| Unigram | 从大候选词表出发，用概率模型剪枝 |
| Viterbi | 求概率最大的单个切分 |
| Forward | 汇总所有合法切分的概率 |
| Subword regularization | 训练时采样多种切分增强鲁棒性 |
| Patch token | 图像局部块的连续或离散表示 |
| Codec token | 音频神经编解码器产生的离散符号 |
| Token tax | 同语义内容在某语言上的额外 token 成本 |
| Special token | 模型协议的一部分，不是普通字符串替换 |
| Golden test | 固定输入对应固定 IDs 与解码结果 |
| Hash | 发现模型与 tokenizer 版本漂移 |
| ABI | token ID 与 embedding 行的稳定绑定 |

### 15.14 最终自测

在不查资料的情况下回答以下问题：

1. 为什么 BPE 必须保存有序 merge rules？

2. 为什么 BBPE 中单 token 可能无法 UTF-8 解码？

3. 为什么词频权重遗漏会改变训练结果？

4. 为什么 SentencePiece 不等于 Unigram？

5. 为什么归一化会破坏严格原文 round-trip？

6. 如何用动态规划计算 Unigram 最优切分？

7. 边缘似然与 Viterbi 似然有什么区别？

8. 为什么 byte fallback 能消除普通 OOV？

9. 词表扩大如何影响 embedding 和 softmax？

10. 如何量化多语言 token tax？

11. 连续视觉 token 与离散视觉 token 有何区别？

12. 图像 patch size 如何影响序列长度？

13. RVQ 为什么每帧产生多个 token？

14. 多模态占位符为什么必须严格校验？

15. 如何防止用户注入聊天控制 token？

16. 为什么 tokenizer 需要与 checkpoint 绑定？

17. 新增特殊 token 后为什么要扩展模型参数？

18. 如何验证一个 tokenizer 实现优化是兼容升级？

19. 线上 token 数突增应看哪些指标？

20. 为什么 tokenizer 是模型能力与系统成本的共同边界？

---

> **扩展小结**：BPE 解决高频片段压缩，SentencePiece 统一语言无关处理，Unigram 用概率模型表达切分不确定性，多模态 tokenizer 把图像、音频和视频纳入序列协议，而工业落地的核心是版本、边界、可逆性与可观测性。

> **实践建议**：先用 golden cases 锁定行为，再比较压缩率与端到端成本；任何 token ID 变化都按模型协议变更处理。
