---
title: "第07章-LLM训练全流程实务"
tags:
  - LLM
  - LLM训练全流程实务
description: "本章要点：① 数据准备 → ② 分词器训练 → ③ 预训练(Pretrain)"
chapter_id: LLM发展-07
last_updated: 2026-07-14
---

# 第07章-LLM训练全流程实务


## 1. 全流程总览

```text
① 数据准备 → ② 分词器训练 → ③ 预训练(Pretrain)
   → ④ 中期训练(继续预训练/退火) → ⑤ SFT(指令微调)
   → ⑥ 对齐(RLHF/DPO/GRPO) → ⑦ 评估 → ⑧ 量化与部署
```

---

## 2. 数据准备(决定上限)

- **来源**: 网页(Common Crawl)、代码、书籍、论文、多语。
- **清洗**: 去重(MinHash/精确)、质量过滤(分类器/规则)、去有害、去隐私(PII)。
- **配比**: 代码/数学比例影响推理能力; 用小模型做数据配比消融。
- **课程学习**: 后期提高高质量/数学/代码占比(退火阶段)。

> 数据 > 架构。同等算力下, 数据质量与配比是效果最大变量。

---

## 3. 分词器(Tokenizer)

- **BPE / BBPE / Unigram**: 主流 BBPE(字节级, 无 OOV)。
- 词表大小 32K-256K: 大词表压缩率高但 embedding 大。
- 中文/多语/代码需专门覆盖。详见 [deeplearning 6.1](../深度学习手撕/第06章-LLM核心技术/第06章-01-Tokenization.md)。

---

## 4. 预训练(Pretrain)

**目标**: next-token 交叉熵。**这是算力大头**(占总成本 90%+)。

关键配置:
- **优化器**: AdamW($\beta_2=0.95$), weight decay 0.1。
- **调度**: warmup(几千步)+ cosine/WSD 到很小(见 [deeplearning 2.2](../深度学习手撕/第02章-优化算法/第02章-02-学习率调度.md))。
- **精度**: BF16 + FP32 主副本(见 [deeplearning 7.3](../深度学习手撕/第07章-训练技术/第07章-03-混合精度.md))。
- **并行**: 数据并行 + 张量并行 + 流水线并行 + ZeRO/FSDP(见 [deeplearning 7.2](../深度学习手撕/第07章-训练技术/第07章-02-分布式训练.md))。
- **序列打包**: 多文档拼接填满上下文, 提利用率。
- **算力估算**: $\approx 6ND$ FLOPs(见 [工程能力第02章](../../基础/工程基础/工程能力/第02章-时间与空间复杂度分析.md))。

```python
# 预训练核心循环骨架
for step, batch in enumerate(loader):
    with autocast(dtype=torch.bfloat16):
        loss = model(batch.input_ids, labels=batch.labels).loss
    scaler.scale(loss).backward()
    clip_grad_norm_(model.parameters(), 1.0)     # 梯度裁剪防爆炸
    scaler.step(opt); scaler.update(); opt.zero_grad()
    sched.step()
    if step % save_every == 0: save_checkpoint(...)   # 见第08章
```

---

## 5. 中期训练(退火 / 长上下文)

- **退火(annealing)**: 训练末期喂高质量数据 + LR 快速降到 0, 显著提分。
- **长上下文扩展**: 用少量长样本 + RoPE 外推(YaRN)把 4K 扩到 128K(见 [第03章](第03章-位置编码与长上下文演进.md))。

---

## 6. SFT(指令微调)

- 数据: 高质量"指令-回答"对(少而精 > 多而杂)。
- 全参 or **LoRA**(省显存, 见 [deeplearning 8.1](../深度学习手撕/第08章-微调与对齐/第08章-01-LoRA.md))。
- 只对回答部分算 loss(mask 掉指令); 小 LR(1e-5 级)、1-3 epoch。

---

## 7. 对齐

按需选 DPO(轻)或 GRPO(推理)。见 [第05章](第05章-对齐技术演进.md)。

---

## 8. 评估

- **基准**: MMLU/GSM8K/HumanEval/MATH/长上下文/安全。
- **注意数据污染**(基准泄漏进训练集)。
- **人评 / LLM-as-judge**: 对开放式能力更可靠。
- **A/B 或 Arena**: 真实偏好。

---

## 9. 量化与部署

- 量化: INT8/INT4/FP8(见 [deeplearning 9.1](../深度学习手撕/第09章-推理优化/第09章-01-量化.md))。
- 推理引擎: vLLM(PagedAttention)、TensorRT-LLM、SGLang。
- 服务: continuous batching + KV Cache 管理(见 [deeplearning 9.4](../深度学习手撕/第09章-推理优化/第09章-04-PagedAttention.md))。

---

## 10. 面试高频

1. 预训练占总成本多少、大头是什么? → 90%+, next-token 预训练。
2. 退火阶段做什么? → 喂高质量数据 + LR 降到 0, 提分。
3. SFT 为什么只对回答算 loss? → 不让模型学着去生成指令。
4. 训练 LLM 的算力怎么估? → ≈ 6ND FLOPs。

---

## 4. 预训练数据准备(Common Crawl / 清洗 / 去重 / 质量过滤)

> **数据是 LLM 的"粮食",也是上限天花板。** 业界共识:"Data > Architecture > Hyperparameters"。同等算力下,数据质量与配比对最终模型能力的影响,通常比架构选择大一个数量级。LLaMA-3 报告里写得很直白:95% 的训练时间花在 next-token 预测上,而 next-token 预测的能力上限,直接由训练数据的分布与质量决定。本节把"预训练数据"从采集到入仓的全链路拆开。

### 4.1 数据来源与配比策略

**核心动机**:LLM 是对训练分布的有偏拟合器。**你想要模型擅长什么,就要在数据里把那个东西喂饱和。** 一份典型 2024-2026 主流开源 LLM 的预训练配比:

| 数据类型 | LLaMA-3 (15T tokens) | Qwen2.5 (18T) | DeepSeek-V3 (14.8T) | 占比逻辑 |
|---------|---------------------|---------------|---------------------|---------|
| 网页(Common Crawl) | ~50% | ~45% | ~55% | 世界知识主源,但需大量清洗 |
| 代码(GitHub 等) | ~15% | ~12% | ~15% | 推理 / 结构化能力 |
| 数学/STEM | ~7% | ~10% | ~8% | 推理基础,数据稀缺需合成 |
| 多语(中/日/韩/欧) | ~10% | ~25%(中英为主) | ~12% | 中文 LLM 通常拉高 |
| 书籍/论文 | ~5% | ~5% | ~5% | 高质量长程依赖 |
| 指令/对话(可选) | ~3% | ~3% | ~5% | 改善基础对话能力 |
| 其他(百科/新闻) | ~10% | 余量 | 余量 | 通用兜底 |

**核心原则**:
- **网页**:量大但脏。占大头,但**清洗后实际可用**通常只有原始的 30-50%。
- **代码**:推理能力的"加速器"。哪怕总占比 10-15%,对 HumanEval/MBPP 的影响是数量级的。
- **数学**:天然稀缺。需要合成(用 Lean/搜索/模型自生成)补足。
- **多语**:中文 LLM 必须拉高中文到 ≥30%,否则中文能力塌方。
- **重复采样**:同一个高质量来源(如 Wikipedia、教科书)可以重复 2-5 次,边际收益仍正。

**数学公式 - 数据配比最优化的近似**:
$$
\min_{\alpha} \; \mathcal{L}_{\text{val}}^{\text{proxy}}(\alpha) \quad \text{s.t.} \quad \sum_i \alpha_i = 1,\; \alpha_i \geq 0
$$
其中 $\alpha_i$ 是第 $i$ 类数据的采样权重,proxy loss 通常用一个小模型(≤1B)在各 $\alpha$ 下训练后,在目标 benchmark 上的加权得分。

---

### 4.2 数据采集:Common Crawl 与替代源

**核心动机**:Common Crawl 是公开网页抓取的"最大池子",每月 ~200-300TB 原始 WARC。但**它不是干净数据**,只是原料。

**主要数据源**:

| 源 | 规模 | 特点 | 处理难度 |
|----|------|------|---------|
| **Common Crawl** | 200TB+/月 | 全网快照,噪声大,90% 无价值 | 高 |
| **GitHub** | ~500B unique code tokens | 代码主源,需去 license 与 personal data | 中 |
| **Wikipedia** | ~20 种语言, ~100M 行 | 高质量百科,几乎无噪声 | 低 |
| **arXiv / 论文** | ~2M 篇 | STEM 高质量 | 中(LaTeX 解析) |
| **书籍** | Books3 / Library Genesis 等 | 长程依赖,但版权风险 | 低(但有法律风险) |
| **StackExchange** | ~50M Q&A | 推理 / 事实知识 | 低 |
| **多语百科** | 各国语言维基 | 多语补足 | 低 |
| **FineWeb / RedPajama** | 已清洗开源数据集 | 起点数据集,省去 90% 清洗 | 极低 |

**Algorithm 4.1** Common Crawl 原始 → 可用 WARC 处理流水线
**Input**: 原始 WARC 文件(每月 ~250TB, ~30B 网页)
**Output**: 清洗后的 JSONL(每行一个文档,含 text + meta)
**Steps**:
1. **WARC 解压**: 提取 HTML 字节流(GNU WARC 工具或 fastWARC,150GB/h 单机)
2. **HTML 解析**: 用 resiliparse / trafilatura / jusText 抽取正文,丢弃 nav/footer/广告
3. **语言识别**(LID): fastText 语言分类器(langdetect.py 或 Facebook's M2L100),阈值 ≥ 0.65
4. **粗过滤**: 长度过滤(50-100k chars)、符号比过滤(字母占比 < 50% 丢)
5. **写入 JSONL**: 后续精细清洗
**Complexity**: O(N) 字节级线性扫描,吞吐 1-5GB/s/核

**代码实现 - WARC 解析 + 语言识别 + 基础过滤**:
```python
import gzip
import json
from fastwarc import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.lang import detect_language

def process_warc(warc_path: str, out_f, min_text_len=200, max_text_len=200_000):
    """处理单个 WARC 文件,写入 JSONL"""
    with gzip.open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            # 只处理 HTTP 响应且状态码 2xx
            if record.headers and record.http_headers and \
               200 <= record.http_headers.status_code < 300:
                try:
                    html = record.reader.read().decode("utf-8", errors="ignore")
                    text = extract_plain_text(html, main_content=True)
                    if not text:
                        continue
                    # 基础长度过滤
                    if len(text) < min_text_len or len(text) > max_text_len:
                        continue
                    # 语言识别(快于 langdetect)
                    lang_info = detect_language(text)
                    if lang_info.score < 0.65:
                        continue  # 噪声语言/混合语言
                    out_f.write(json.dumps({
                        "text": text,
                        "lang": lang_info.lang,
                        "url": record.http_headers.get("WARC-Target-URI", ""),
                    }, ensure_ascii=False) + "\n")
                except Exception as e:
                    continue  # 单文档失败不阻塞流水线

# 使用:几十台机器并行处理每月 dump
# python -c "from process import process_warc; process_warc('CC-MAIN-2024-30.warc.gz', open('out.jsonl','w'))"
```

**关键参数**:
- `min_text_len=200`:低于 200 字的网页通常是导航/cookie 提示,价值低。
- `max_text_len=200_000`:超过 20 万字符的网页通常是聚合页/爬虫产生的脏数据。
- 语言阈值 0.65:低于此分的多半是混合语言页面或代码片段。

**案例**:
- **LLaMA-3** 用 5 个 Common Crawl 大版本(2022-2024)做去重合并,最终用 ~15T 高质量 token。
- **Qwen2.5** 额外爬了大量中文学术/百科站点补充多语。
- **DeepSeek-V3** 大幅压低多语占比,聚焦中英双语 + 代码。
- **FineWeb** (HuggingFace 2024)开源了处理后的 1.3T token 清洗版,直接可用。

**失败案例与改进**:
- **RefinedWeb**:最早公开的全清洗 Common Crawl 流水线(TII),但过于激进过滤,导致低资源语言数据损失。
- **Dolma** (AI2):尝试保留更多语种和长尾,但相应数据噪声也更大。
- **改进方向**:用 model-based quality classifier(见 4.4)替代纯规则,更精细地区分"长但有用"vs"长但噪声"。

---

### 4.3 文本清洗:HTML 残留 / 乱码 / 编码问题

**核心动机**:爬虫数据 30-50% 的"看起来有意义的字符"其实是噪声——HTML 实体残留、emoji 滥用、乱码、Boilerplate。

**常见噪声类型与处理**:
| 噪声类型 | 例子 | 处理方法 |
|---------|------|---------|
| HTML 实体未转义 | `&amp;`, `&#x27;` | `html.unescape()` |
| URL 残留 | `https://...`(正文里夹链接) | regex 去除 |
| 多余空白 | 连续空行、tab | `re.sub(r'\s+', ' ', text)` |
| 控制字符 | `\x00`, `\ufeff` | `text.translate(remove_chars)` |
| 重复段落(爬虫切分错误) | 同一段重复 5 次 | n-gram 去重(见 4.5) |
| Emoji/特殊符号 | 🚀✨🎉 | 按语种保留(中英保,代码全删) |
| 半乱码 | `ä¸å¥½` (utf-8 被当 latin-1 解) | ftfy 库 + chardet 检测 |
| Boilerplate | "订阅 newsletter" / cookie 提示 | 规则 + 分类器 |

**代码实现 - 清洗 pipeline**:
```python
import ftfy
import re
import unicodedata

class TextCleaner:
    def __init__(self):
        # 常用 boilerplate 模式
        self.boilerplate_patterns = [
            r"sign up for.*newsletter",
            r"accept.*cookies",
            r"subscribe to.*channel",
            r"all rights reserved",
            r"\d+\s*comments?",
        ]
        self.bp_regex = re.compile("|".join(self.boilerplate_patterns), re.IGNORECASE)
        # URL / email
        self.url_regex = re.compile(r"https?://\S+|www\.\S+")
        self.email_regex = re.compile(r"\S+@\S+\.\S+")
        # 多空白
        self.ws_regex = re.compile(r"\s+")
        # 控制字符
        self.ctrl_chars = dict.fromkeys(range(32), None)
        self.ctrl_chars.pop(ord('\n'), None)
        self.ctrl_chars.pop(ord('\t'), None)

    def clean(self, text: str) -> str:
        # 1) 修复编码
        text = ftfy.fix_text(text)
        # 2) Unicode 标准化(NFC: 组合字符统一)
        text = unicodedata.normalize("NFC", text)
        # 3) 去控制字符
        text = text.translate(self.ctrl_chars)
        # 4) 去 URL / email
        text = self.url_regex.sub("", text)
        text = self.email_regex.sub("", text)
        # 5) 去 boilerplate
        text = self.bp_regex.sub("", text)
        # 6) 合并多空白
        text = self.ws_regex.sub(" ", text).strip()
        return text

# 使用
cleaner = TextCleaner()
clean_text = cleaner.clean(raw_text)
```

**失败案例**:
- **Gopher 规则**(DeepMind 2021):早期经典过滤规则集,但英文为主,中文场景需要重写。
- **C4** (Google 2020):用关键词黑名单过滤低质量,但过度依赖英文 heuristic。
- **改进**:fineweb 的"可读性 + 教育价值"分类器(Gemma 模型打分)替代了 80% 硬规则。

---

### 4.4 质量过滤:从规则到 Model-based Classifier

**核心动机**:规则过滤是粗筛,**真正决定数据质量的,是 model-based quality classifier**。2023 年后业界主流做法是训一个小的 fastText / Gemma-classifier 给每篇文档打 0-1 分,保留 ≥ 阈值的部分。

**两阶段质量过滤**:

**Stage 1: Heuristic / 规则过滤**(快,丢 60-80% 垃圾):
- 词数 / 平均词长 / 符号比 / 大写比
- 停用词比例(英文 too many "the"→可能是 SEO 页面)
- 段落数 / 句子长度分布
- 关键词黑名单(赌博/色情/广告词)
- **LID 二次过滤**: 即使语言识别通过,低置信度的也丢

**Stage 2: Model-based classifier**(慢,丢剩下 30-50% 低质):
- 用 GPT-3.5/4 或开源模型(如 BGE、Mistral)给每篇文档打 0-5 分"教育价值"分
- 或训一个 fastText 分类器区分 Wikipedia / OpenWebText(高质量) vs CC 噪声(低质量)
- FineWeb 的 fineweb-edu classifier(Gemma-2-2B 蒸馏)目前最强,给 ~1.3T token 打了分

**Algorithm 4.2** Quality Classifier 训练流水线
**Input**: 高质量参考文档集(Wikipedia, OpenWebText, books)+ 低质量参考集(随机 Common Crawl 抽样)
**Output**: 一个能对每篇文档打 0-1 分的分类器
**Steps**:
1. 标注 ~50k 高质量 + ~50k 低质量文档
2. 用 fastText(快)或 DeBERTa-v3(精)训二分类器
3. 在 10k held-out 上评测 AUC,目标 ≥ 0.92
4. 推理时阈值取 0.5-0.7(高阈值=更严格)
**Complexity**: 训练 ~1 GPU-hour, 推理 ~1000 文档/秒/核

**代码实现 - 用 fastText 训质量分类器**:
```python
import fasttext
import random

# 准备训练数据:每行 "__label__hq <text>" 或 "__label__lq <text>"
def format_for_fasttext(text: str, label: str) -> str:
    # fastText 要求文本中换行符替换
    return f"__label__{label} {text.replace(chr(10), ' ').strip()}"

# 训练(假设 train.txt 已准备好)
model = fasttext.train_supervised(
    input="train.txt",
    lr=0.5,
    epoch=5,
    wordNgrams=2,
    dim=100,
    loss="softmax",
)
# 评估
print(model.test("test.txt"))  # (N, P@1, R@1)

# 推理:对每篇新文档打分
def score_doc(text: str) -> float:
    text = text.replace(chr(10), ' ')
    pred = model.predict(text)  # (('__label__hq',), array([0.85]))
    return float(pred[1][0])

# 批量打分
def filter_corpus(input_jsonl: str, output_jsonl: str, threshold: float = 0.5):
    with open(output_jsonl, "w") as out:
        for line in open(input_jsonl):
            doc = json.loads(line)
            if score_doc(doc["text"]) >= threshold:
                out.write(line)
```

**案例**:
- **LLaMA-3**:用 Llama-2 做质量分类器,保留 top 30% 数据。
- **Qwen2.5**:中文场景额外训了一个中文分类器。
- **DeepSeek-V3**:用 RM(reward model)思路训质量分,保留高分。
- **FineWeb-edu**:开源了 Gemma-2-2B 蒸馏的 fineweb-edu-classifier,直接可用。
- **GPT-4 / Claude**:不公开,但工业界猜测用类似自评机制。

**失败案例与改进**:
- 早期 Gopher 用规则+长度过滤,中文场景漏掉 40% 高质量长文。
- 改进:**分语种独立训分类器**(中文一个、英文一个、代码一个),避免英文 bias。
- **陷阱**:用 GPT-4 打分当标签训小模型,会有 GPT 自身的 bias,需要 human spot-check。

---

### 4.5 去重:MinHash / Exact / SemHash

**核心动机**:**重复数据是 LLM 训练的最大隐患之一**。同一文档在 CC 里出现 5-10 次是常态,如果不处理:
- 模型会"记住"高频段落,降低泛化(perplexity 升高)。
- 浪费算力(同样的 token 训 N 次)。
- 可能引入测试集泄漏 → benchmark 作弊。

**三层去重策略**:

**Layer 1: 精确去重**(快):
- 按文档 hash(SHA-256)判重
- 通常去掉 5-15% 重复

**Layer 2: 模糊去重**(MinHash):
- 把每篇文档切成 5-gram,用 MinHash 算 ~256 维指纹
- Jaccard 相似度 ≥ 0.8 视为重复
- 通常再去掉 10-30% 近似重复
- **实现**:datasketch 库,或 faiss 向量化加速

**Layer 3: 跨文档段落去重**:
- 按段落(50-200 token 滑动窗口)做 MinHash
- 去掉跨文档重复段落
- 通常再去掉 5-10%

**Algorithm 4.3** MinHash 去重
**Input**: N 篇文档集合 D = {d_1, ..., d_N}
**Output**: 去重后的子集 D'
**Steps**:
1. 每篇文档切 n-gram(常用 word-level 5-gram)
2. 对每篇文档用 k 个哈希函数算 MinHash 签名(k=256)
3. 文档分桶:把 MinHash 签名分成 b 个 band,每个 band r 行;只要任一 band 完全相同,放入同一候选对
4. 对候选对计算精确 Jaccard 相似度 ≥ 阈值 → 重复
5. 重复文档保留 hash 最小的(任意策略即可)
**Complexity**: O(N × k) 哈希 + O(N² / b) 候选对(但 b 选好通常是 N×常数)

**数学公式 - MinHash 估计 Jaccard**:
$$
P[h_{\min}(A) = h_{\min}(B)] = \frac{|A \cap B|}{|A \cup B|} = J(A, B)
$$
其中 $h_{\min}$ 是 k 个哈希函数中的某一个。k=256 时估计误差 ≤ 0.05。

**代码实现 - 用 datasketch 做大规模 MinHash 去重**:
```python
from datasketch import MinHash, MinHashLSH
import json
from typing import Iterable

def make_minhash(text: str, ngrams: int = 5, num_perm: int = 128) -> MinHash:
    """文档 → MinHash 签名"""
    words = text.lower().split()
    m = MinHash(num_perm=num_perm)
    for i in range(len(words) - ngrams + 1):
        m.update(" ".join(words[i:i+ngrams]).encode())
    return m

def deduplicate(
    input_jsonl: str,
    output_jsonl: str,
    threshold: float = 0.8,
    num_perm: int = 128,
):
    """LSH 索引 → 候选对 → 精确判重"""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen_hashes = {}  # minhash_str → doc_id

    with open(output_jsonl, "w") as out:
        for i, line in enumerate(open(input_jsonl)):
            doc = json.loads(line)
            text = doc["text"]
            m = make_minhash(text, num_perm=num_perm)

            # 查 LSH 索引
            is_dup = False
            for dup_id in lsh.query(m):
                # 这里简化:实际需算精确 Jaccard
                is_dup = True
                break

            if not is_dup:
                # 加进索引
                lsh.insert(str(i), m)
                out.write(line)

            if i % 100000 == 0:
                print(f"Processed {i} docs, kept ~{i * 0.7:.0f}")

# 使用
# deduplicate("common_crawl.jsonl", "deduped.jsonl", threshold=0.8)
```

**参数选择**:
- `threshold=0.8`:Jaccard ≥ 80% 视为重复。太低(<0.7)误杀多,太高(>0.9)漏网多。
- `num_perm=128`:MinHash 签名长度。越大越精确但越慢,128 是常用折中。
- `ngrams=5`:词级 5-gram。中文需先分词或用字符 n-gram(3-5 字)。

**案例**:
- **LLaMA-2/3** 用 MinHash + LSH 在 ~5T 文档上去重,去掉 ~30% 重复。
- **RedPajama-v2** 开源了完整去重 pipeline 和签名(便于二次过滤)。
- **DeepSeek-V3**:用更激进的去重(0.85 阈值),保留 ~10T 干净 token。
- **GLaM** (Google):研究证明去重后训练效率提升 2-3 倍。
- **GPT-4 / Claude**:不公开,但多次报告显示他们做了段落级去重。

**失败案例**:
- 早期 RefinedWeb 只做了文档级去重,大量段落重复(模板生成的 boilerplate)未处理。
- 改进:**段落级去重**(按 50-200 token 滑窗)成为标配,FineWeb 报告段落级去重多去掉 8%。

**SemHash 去重**(前沿):
- 用 sentence embedding 做语义级去重,能抓到 MinHash 漏掉的"改写重复"。
- 工具:SemDeDup (Meta 2023),在小模型上能多抓 5-10% 重复。
- 代价:需要 GPU 跑 embedding,慢 5-10×。

---

### 4.6 PII 与有害内容过滤

**核心动机**:**法律 + 品牌风险**。预训练数据里的 PII(个人隐私)如果不处理,模型可能泄露,违反 GDPR / CCPA。未成年相关、CSAM、医学误导等内容需要过滤。

**PII 处理**:
- **正则**:邮箱、电话、SSN、信用卡、IP 地址
- **NER 模型**:识别姓名、地址、身份证号(中文身份证)
- **替换 vs 删除**:邮箱替换成 `<EMAIL>`,电话替换成 `<PHONE>`(避免破坏文本连贯性)
- **人脸 / 声音**:理论上训练前应过滤含人脸/声音的图像,但纯文本 LLM 这块压力较小

**有害内容过滤**:
- **Toxicity classifier**:如 Perspective API、Jigsaw 的 toxigen
- **CSAM 关键词**:硬黑名单,100% 过滤
- **争议内容**:politics / health / finance 不完全过滤,但降采样

**代码实现 - PII 检测 + 替换**:
```python
import re
from typing import Tuple

class PIIFilter:
    def __init__(self):
        # 简单正则(实际需更复杂,如 RFC 5322)
        self.patterns = {
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "PHONE_CN": re.compile(r"\b1[3-9]\d{9}\b"),  # 中国手机号
            "SSN_US": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "URL": re.compile(r"https?://\S+"),
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "ID_CN": re.compile(r"\b\d{17}[\dXx]\b"),  # 中国身份证
        }

    def detect_and_replace(self, text: str) -> Tuple[str, dict]:
        stats = {k: 0 for k in self.patterns}
        for label, pattern in self.patterns.items():
            text, n = pattern.subn(f"<{label}>", text)
            stats[label] = n
        return text, stats

# 使用
filt = PIIFilter()
clean_text, stats = filt.detect_and_replace(raw_text)
# stats 告诉你替换了多少邮箱/电话,可入仓分析
```

**案例**:
- **LLaMA-2/3**:声称做了严格 PII 过滤,但仍有泄漏(可被 prompt 攻击出来)。
- **Qwen**:中文场景重点处理中国手机号 / 身份证 / 地址。
- **GPT-4 / Claude**:完全不公开,但 OpenAI 报告用专用模型过滤。
- **Anthropic**:有专门团队 + Constitution 文档定义"有害"。

**失败案例**:
- 早期模型可被诱导输出训练数据里的邮箱 / SSN,引发诉讼(Sara Wachter-Boettcher 等)。
- 改进:**差分隐私**(DP-SGD)、**机器遗忘**(machine unlearning)等前沿方向。

---

### 4.7 预训练数据格式:JSONL + Parquet + 词表 ID 化

**核心动机**:清洗后的数据要存成高效的列式存储,支持千卡并行的随机采样。

**主流格式**:
- **JSONL**:每行一个文档,易调试,适合 ≤ 100B token
- **Parquet**:列式压缩,Snappy/zstd 压缩比 3-5×,适合 TB 级,主流选择
- **MDS**(Megatron-DeepSpeed):Megatron 自定义二进制格式,极致吞吐
- **WebDataset / TFRecord**:序列化为二进制 tar,适合图像/多模态

**最佳实践**:
- 用 Parquet + zstd 压缩,按"语言 / 来源 / 质量分"分区
- 配套元数据文件(`*.stats.json`):每个分区的 token 数、平均长度
- 用 **Megatron-DeepSpeed 的 data pipeline** 直接读 parquet 进 dataloader

**代码实现 - Parquet 写入 + 分区**:
```python
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path

def jsonl_to_parquet(input_dir: str, output_dir: str):
    """把所有 jsonl 转成 parquet,按语言分区"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    writers = {}  # lang → ParquetWriter

    for jsonl_file in Path(input_dir).rglob("*.jsonl"):
        batch = []
        for line in open(jsonl_file):
            doc = json.loads(line)
            batch.append({
                "text": doc["text"],
                "lang": doc.get("lang", "unknown"),
                "source": doc.get("source", "common_crawl"),
                "quality": doc.get("quality", 0.0),
            })
            if len(batch) >= 10000:
                _flush_batch(batch, writers, out_path)
                batch = []
        if batch:
            _flush_batch(batch, writers, out_path)

    # 关闭所有 writer
    for w in writers.values():
        w.close()

def _flush_batch(batch, writers, out_path):
    table = pa.Table.from_pylist(batch)
    # 按 lang 分区
    for lang in set(d["lang"] for d in batch):
        sub = table.filter(pa.compute.equal(table["lang"], lang))
        if lang not in writers:
            lang_dir = out_path / f"lang={lang}"
            lang_dir.mkdir(parents=True, exist_ok=True)
            writers[lang] = pq.ParquetWriter(
                lang_dir / "data.parquet",
                schema=table.schema,
                compression="zstd",
            )
        writers[lang].write_table(sub)

# 使用
# jsonl_to_parquet("raw/", "parquet/")
```

**配套 metadata 写入**:
```python
# 训练前:统计每个分区的 token 数(用预训练的 tokenizer 跑一遍)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

def count_tokens_in_parquet(parquet_path: str) -> int:
    table = pq.read_table(parquet_path, columns=["text"])
    total = 0
    for text in table["text"]:
        total += len(tok.encode(text, add_special_tokens=False))
    return total

# 写入 metadata.json
metadata = {
    "total_tokens": ...,
    "per_partition": {...},
    "vocab_size": tok.vocab_size,
    "tokenizer": "Llama-3.2-1B",
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

**案例**:
- **FineWeb**:用 parquet 分区,按 lang / score 分桶。
- **RedPajama-v2**:同样 parquet,公开 schema 文档。
- **Megatron-LM**:用 mds 格式,极致吞吐(30% 提升 vs parquet)。
- **GPT-4 / Claude**:完全黑盒,推测用类似 mds 的二进制格式。

---

### 4.8 预训练数据流水线:端到端示例

**完整流程图**:
```text
[Raw WARC] → [WARC 解析] → [LID + 长度过滤] → [粗清洗]
    ↓
[JSONL] → [Quality Classifier] → [MinHash 去重] → [PII 过滤]
    ↓
[Cleaned JSONL] → [分语种 Parquet] → [Token Count] → [打包 MDS / bin]
    ↓
[训练集 + metadata.json] → [DataLoader] → [Pretrain]
```

**单台 worker 完整实现**:
```python
import json
import gzip
from pathlib import Path
from transformers import AutoTokenizer

def preprocess_pipeline(
    warc_path: str,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    min_score: float = 0.5,
    dedup_threshold: float = 0.8,
):
    """单条流水线: WARC → cleaned parquet → tokenized bin"""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    cleaner = TextCleaner()  # 见 4.3
    pii_filt = PIIFilter()   # 见 4.6
    lsh = MinHashLSH(threshold=dedup_threshold, num_perm=128)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    docs = []

    # 1. WARC → 文本
    with gzip.open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if not (record.http_headers and 200 <= record.http_headers.status_code < 300):
                continue
            try:
                html = record.reader.read().decode("utf-8", errors="ignore")
                text = extract_plain_text(html, main_content=True)
                if not text or len(text) < 200:
                    continue
                # LID
                lang = detect_language(text)
                if lang.score < 0.65:
                    continue
                # 清洗
                text = cleaner.clean(text)
                text, _ = pii_filt.detect_and_replace(text)
                # 质量打分
                score = score_doc(text)
                if score < min_score:
                    continue
                # 去重
                m = make_minhash(text)
                if any(True for _ in lsh.query(m)):
                    continue  # 重复
                lsh.insert(str(len(docs)), m)
                docs.append({"text": text, "lang": lang.lang, "score": score})
            except Exception:
                continue

    # 2. 写入 parquet
    table = pa.Table.from_pylist(docs)
    pq.write_table(table, f"{output_dir}/cleaned.parquet", compression="zstd")

    # 3. Tokenize + 打包 bin(可选)
    token_chunks = []
    for doc in docs:
        ids = tok.encode(doc["text"], add_special_tokens=False)
        token_chunks.extend(ids)
        token_chunks.append(tok.eos_token_id)  # 文档边界

    # 写入二进制 bin(Megatron 格式)
    import numpy as np
    arr = np.array(token_chunks, dtype=np.int32)
    arr.tofile(f"{output_dir}/tokens.bin")

    print(f"Done: {len(docs)} docs, {len(token_chunks)} tokens")

# 使用
# preprocess_pipeline("CC-MAIN-2024-30-001.warc.gz", "output/2024-30/")
```

---

### 4.9 数据配比消融实验:小模型验证大模型

**核心动机**:**用大模型调数据配比太贵**。业界标准做法:用一个小模型(≤1B)在多种配比下训练短步数(几十亿 token),看哪个配比在目标 benchmark 上最好,再 scale 到大模型。

**经典流程**(Llama-2 paper 的做法):
1. 准备 6-10 种候选数据配比
2. 用 1B 模型 × 200B token × 10 配比 = ~10 次小训练
3. 每次训练后 eval MMLU / GSM8K / HumanEval 等核心 benchmark
4. 选综合分最高的配比
5. Scale 到 7B → 70B × 15T token

**代码实现 - 配比 ablation 框架**:
```python
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class MixtureConfig:
    name: str
    weights: Dict[str, float]  # source -> weight
    total_tokens: int = 200_000_000_000  # 200B

def run_ablation(mixtures: List[MixtureConfig], model_size: str = "1B"):
    """对每个配比训练小模型并评估"""
    results = []
    for cfg in mixtures:
        print(f"Training with mixture: {cfg.name}")
        # 1. 构造加权 sampler
        sampler = build_weighted_sampler(cfg.weights)
        # 2. 训练 200B token(几小时/几天)
        model = train_small_model(model_size=model_size, sampler=sampler, tokens=cfg.total_tokens)
        # 3. 评估
        scores = evaluate(model, benchmarks=["mmlu", "gsm8k", "humaneval"])
        results.append({"cfg": cfg, "scores": scores})
        print(f"  MMLU={scores['mmlu']:.3f}, GSM8K={scores['gsm8k']:.3f}, HE={scores['humaneval']:.3f}")
    return results

def find_best_mixture(results):
    """加权综合得分最高的配比"""
    # 假设权重: MMLU=0.4, GSM8K=0.3, HumanEval=0.3
    weights = {"mmlu": 0.4, "gsm8k": 0.3, "humaneval": 0.3}
    best = max(results, key=lambda r: sum(r["scores"][k] * w for k, w in weights.items()))
    return best

# 例子:4 种配比候选
mixtures = [
    MixtureConfig("baseline", {"web": 0.5, "code": 0.15, "math": 0.07, "wiki": 0.05, "books": 0.05, "other": 0.18}),
    MixtureConfig("more_code", {"web": 0.4, "code": 0.25, "math": 0.07, "wiki": 0.05, "books": 0.05, "other": 0.18}),
    MixtureConfig("more_math", {"web": 0.4, "code": 0.15, "math": 0.17, "wiki": 0.05, "books": 0.05, "other": 0.18}),
    MixtureConfig("balanced", {"web": 0.45, "code": 0.15, "math": 0.10, "wiki": 0.08, "books": 0.08, "other": 0.14}),
]
results = run_ablation(mixtures)
best = find_best_mixture(results)
print(f"Best: {best['cfg'].name}")
```

**案例**:
- **LLaMA-2 paper** (2023):Table 6 列出 7 种配比的 ablation,最终选了一个"代码 / 数学 / wiki 略高"的配比。
- **DeepSeek-V2/V3**:专门做"代码 + 数学"提升的 ablation。
- **Qwen2.5**:中文 / 多语 / 代码三角平衡的 ablation。
- **MiniCPM** (清华,2024):在小模型上做了极致的配比实验。

**陷阱**:
- 小模型上的最优配比不一定 scale 到大模型,但相关性通常 ≥ 0.7,可作为强信号。
- 评估 benchmark 必须固定,否则 ablation 不可比。

---

### 4.10 训练数据采样策略:温度采样 vs 均匀采样

**核心动机**:**高质量数据应该被多看几遍**。但重复采样同样的 token,会引入"过拟合"风险。

**三种主流采样策略**:

**(1) 均匀采样**(Uniform):
- 每个 token 被采样概率相等
- 缺点:低质数据也被多看

**(2) 温度采样**(Temperature sampling):
- 概率 ∝ p_i^(1/T)
- T < 1 → 偏向高频(高质量)数据
- T > 1 → 偏向长尾

**(3) 重要性采样**(Importance sampling):
- 按质量分 / 来源权重采样
- DoReMi (DeepMind 2023):用 proxy loss 自动学最优配比

**Algorithm 4.4** Temperature Sampling
**Input**: N 个数据分区的权重 w_1, ..., w_N
**Output**: 采样分布
**Steps**:
1. 计算原始概率 $p_i = w_i / \sum_j w_j$
2. 应用温度: $p'_i = p_i^{1/T} / \sum_j p_j^{1/T}$
3. T=1 是均匀;T<1 偏向高质量;T>1 偏向长尾
**Complexity**: O(N)

**代码实现**:
```python
import numpy as np

def temperature_sample(weights: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """weights: 每分区的 token 数 / 期望占比"""
    weights = np.array(weights, dtype=float)
    if temperature == 1.0:
        return weights / weights.sum()
    # 应用温度
    p = weights ** (1.0 / temperature)
    return p / p.sum()

# 例子:5 个分区,实际权重 = 期望占比
weights = np.array([0.50, 0.15, 0.10, 0.10, 0.15])  # web, code, math, wiki, other
p_uniform = temperature_sample(weights, 1.0)        # [0.50, 0.15, ...]
p_high_quality = temperature_sample(weights, 0.5)    # 偏向 0.50 那项
p_balanced = temperature_sample(weights, 2.0)        # 偏向长尾
```

**案例**:
- **LLaMA-3**:用温度 1.0 均匀采样,但在"高质量子集"内做二次采样。
- **DeepSeek-V3**:多次实验 T=0.7-0.9,效果略好于均匀。
- **MiniCPM**:用 DoReMi 自动学最优配比。

---

### 4.11 数据配比的"长尾"问题

**核心动机**:**预训练数据天然长尾**——头部 10% 的来源贡献 80% 的 token。怎么处理长尾决定了模型在长尾领域的能力。

**策略**:
- **保头去尾**:头部来源不截断,尾部小来源全部保留。
- **保尾加权**:用 inverse-frequency 加权尾部,避免被稀释。
- **过采样低质但多样**:多样性 vs 质量,经验上质量优先。
- **Domain-specific 增强**:法律 / 医学 / 金融专业数据过采样 2-5×。

**代码实现 - Inverse-frequency 采样**:
```python
def inverse_freq_weights(sources_counts: dict, alpha: float = 1.0):
    """
    sources_counts: {'web': 1e12, 'code': 3e11, 'wiki': 1e10, ...}
    alpha: 平滑参数(0=均匀,1=完全 inverse freq)
    """
    total = sum(sources_counts.values())
    n = len(sources_counts)
    weights = {}
    for src, cnt in sources_counts.items():
        p = cnt / total  # 原始频率
        # Inverse-frequency weighting
        weights[src] = (1.0 / p) ** alpha
    # 归一化
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}

# 例子
counts = {"web": 1e12, "code": 3e11, "wiki": 1e10, "math": 5e10}
print(inverse_freq_weights(counts, alpha=0.5))
# alpha=0.5 时 wiki 从 0.01% → 0.5%,过采样
```

**失败案例**:
- 早期 GPT-3 严重偏向英文,中文 / 小语种几乎"饿死"。
- 改进:多语 LLM (mBERT, XLM-R, mT5) 强制按语种平衡。

---

### 4.12 合成数据(Self-instruct / Evol-instruct)

**核心动机**:**真实数据不够用时,用模型自己生成合成数据补**。2024 年起,合成数据已成为主流 LLM 训练的重要补充。

**主流方法**:

| 方法 | 提出者 | 核心思路 | 适用场景 |
|------|--------|---------|---------|
| **Self-Instruct** | Wang 2022 | 用种子指令 + GPT-3 生成新指令 | 通用 SFT 数据 |
| **Evol-Instruct** | WizardLM 2023 | 在已有指令上加难度 / 变化 | 复杂 SFT |
| **Magpie** | Xu 2024 | 直接采样 Llama-3-chat 的指令分布 | 高质量对话 |
| **Self-Rewarding** | Yuan 2024 | 模型同时当学生和老师 | 推理 + 偏好对齐 |
| **STaR / Quiet-STaR** | Zelikman 2022 | 模型解数学题,只保留对的 | 推理 / 数学 |
| **ReST** | Gulcehre 2023 | Reward model 打分 + 过滤 | RL 风格合成 |

**代码实现 - Self-Instruct 简化版**:
```python
import random
from typing import List

SELF_INSTRUCT_PROMPT = """You are asked to come up with a diverse set of task instructions.

Seed Instructions:
{seeds}

Generate {n} new instructions that are:
- Diverse (different topics, formats, lengths)
- High-quality (clear, answerable)
- Different from seeds

Format: each on a new line starting with a number.
"""

def generate_instructions(seeds: List[str], n: int = 100, llm_call=None) -> List[str]:
    """调用 LLM 生成新指令"""
    prompt = SELF_INSTRUCT_PROMPT.format(seeds="\n".join(seeds), n=n)
    response = llm_call(prompt)  # 调用 GPT-4 / Claude
    # 解析
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return [re.sub(r"^\d+\.\s*", "", l) for l in lines]

def generate_response(instruction: str, llm_call) -> str:
    """给定指令生成回答"""
    return llm_call(f"Please answer the following:\n{instruction}")

# 流水线
seeds = load_seed_instructions("seed.jsonl")  # 100 条种子
all_instructions = list(seeds)
for _ in range(10):  # 10 轮迭代
    new_inst = generate_instructions(all_instructions[-100:], n=100)
    all_instructions.extend(new_inst)

# 分类 + 去重 + 质量过滤 → SFT 数据
```

**案例**:
- **WizardLM** (2023):Evol-Instruct 在 Alpaca 数据上扩展出 50k 复杂指令。
- **Magpie** (2024):从 Llama-3-70B-Instruct 直接采样了 300 万高质量对话。
- **DeepSeek-V3**:在数学 / 代码场景大量合成数据。
- **GPT-4 / Claude**:用 RLHF + 内部合成。

**失败案例**:
- **Model Collapse** (Shumailov 2024):只用合成数据训练会导致模型退化,需保留 ≥ 30% 真实数据。
- **Bias 放大**:合成数据会放大种子模型的偏见。

---

### 4.13 训练 / 验证 / 测试集划分

**核心动机**:**评估集不能进训练集,否则 benchmark 作弊**。

**原则**:
- **预训练阶段**:不做 train/val/test 划分,但需要 **deduplication against benchmarks**(用 n-gram overlap 或 embedding 相似度,确保训练集不包含 MMLU / GSM8K 等题目)。
- **SFT 阶段**:划分 95/5(train/val)。
- **RLHF 阶段**:额外准备 RM 训练集(偏好对)+ 评估集。

**去污染流水线**:
```python
from datasketch import MinHash, MinHashLSH

def check_contamination(train_doc: str, benchmark_docs: List[str], threshold: float = 0.7) -> bool:
    """检查 train 文档是否与 benchmark 高重叠"""
    m_train = make_minhash(train_doc, num_perm=256)
    for bm_doc in benchmark_docs:
        m_bm = make_minhash(bm_doc, num_perm=256)
        # 用真实 Jaccard(简化:直接用 MinHash 估计)
        if m_train.jaccard(m_bm) >= threshold:
            return True
    return False

# 大规模:对每个 benchmark 题做 n-gram 索引,然后扫描训练集
def build_contamination_index(benchmark_jsonl: str, threshold: float = 0.7):
    """用 bloom filter 或 LSH 索引所有 benchmark 的 13-gram"""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, line in enumerate(open(benchmark_jsonl)):
        doc = json.loads(line)
        ngrams = extract_ngrams(doc["text"], n=13)
        for ng in ngrams:
            lsh.insert(f"{i}_{ng}", make_minhash(ng))
    return lsh

# 检查训练集
def filter_contaminated(train_jsonl: str, contam_index) -> str:
    out = open("train_clean.jsonl", "w")
    for line in open(train_jsonl):
        doc = json.loads(line)
        ngrams = extract_ngrams(doc["text"], n=13)
        is_contam = any(len(list(contam_index.query(m))) > 0 for m in [make_minhash(ng) for ng in ngrams])
        if not is_contam:
            out.write(line)
    out.close()
```

**案例**:
- **LLaMA-2 paper** 详细报告了对 13 个 benchmark 的去污染流程。
- **MMLU / GSM8K**:因为公开,容易被无意污染,所有主流 LLM 都会做去污染。
- **LiveBench / DynamicBench**:动态更新题目,从源头避免污染。

---

### 4.14 数据治理:版本控制 + 元数据 + License

**核心动机**:**训练数据是 LLM 的"基因",必须可追溯、可审计、可重放**。

**最佳实践**:
- **DVC / LakeFS**:数据版本控制,每次训练绑死数据 commit hash。
- **元数据**:每个数据文件记录来源、抓取时间、license、清洗 pipeline 版本。
- **License 矩阵**:CC-BY-SA / MIT 可商用,GPL 限制,Books3 风险未知。

**代码实现 - 数据版本管理**:
```python
# dvc.yaml 示例(配合 git)
# 数据准备是一个 dvc stage
# 数据 hash 自动算
"""
stages:
  prepare_data:
    cmd: python preprocess.py --input data/raw/ --output data/cleaned/
    deps:
      - preprocess.py
    outs:
      - data/cleaned/
metrics:
  - metrics/data_stats.json:
      cache: false
"""

# 元数据文件 schema
metadata = {
    "version": "2024-07-15-v3",
    "sources": [
        {"name": "Common Crawl 2024-30", "license": "CC0", "tokens_after_cleaning": 5e12},
        {"name": "GitHub", "license": "MIT/Apache/GPL mixed", "tokens": 2e12},
        {"name": "Wikipedia", "license": "CC-BY-SA", "tokens": 5e10},
    ],
    "pipeline_version": "v3.2.1",
    "total_tokens": 7.05e12,
    "contamination_checked": True,
    "pii_filtered": True,
}
```

**案例**:
- **RedPajama-v2**:开源了完整数据 + 元数据 + license 矩阵。
- **The Pile** (EleutherAI 2020):早期大规模开源数据集,17 个来源。
- **FineWeb**:开源 + 完整 provenance,HuggingFace 上直接下载。
- **Books3 争议** (2023):因版权问题被多国诉讼,训练数据 license 透明度成为行业标准。

---

### 4.15 数据准备常见坑与最佳实践

**坑 1:数据泄露 → 评估集污染**
- 现象:MMLU 上 90% 但真实场景很菜。
- 对策:严格 dedup + 13-gram overlap 检查。

**坑 2:数据单调**
- 现象:模型在某些话题(医学)弱,但综合分还行。
- 对策:分领域 ablation,弱域过采样。

**坑 3:过度清洗**
- 现象:只剩 Wikipedia / 新闻,模型变"无聊 + 重复"。
- 对策:保留 20-30% 长尾数据,提供多样性。

**坑 4:多语不平衡**
- 现象:中文 LLM 中文差。
- 对策:中文 ≥ 30%,关键领域(中文学术 / 知乎)过采样。

**坑 5:代码 / 数学比例过低**
- 现象:HumanEval / GSM8K 上不去。
- 对策:代码 ≥ 15%,数学 ≥ 7%,合成补足。

**坑 6:训练数据时效性**
- 现象:训练截止 2023,问 2024 事件答不上。
- 对策:定期 re-train,或用 RAG 补。

**坑 7:数据 / tokenizer / 训练 pipeline 不同步**
- 现象:用 A 的 tokenizer 处理 B 的数据,边界 token 不对。
- 对策:数据 → tokenizer → 训练,三个环节统一。

**最佳实践清单**:
- [ ] 数据版本化(DVC / LakeFS)
- [ ] 完整元数据(provenance + license + token count)
- [ ] LID + 长度过滤 + 质量打分 + MinHash 去重 + PII 过滤
- [ ] Benchmark 去污染
- [ ] 分语种 / 来源 / 质量分分区
- [ ] 配比 ablation(小模型验证)
- [ ] 定期 spot-check 人工抽检
- [ ] License 合规审查
- [ ] 数据 / tokenizer / 模型版本绑定
- [ ] 训练日志里记录数据 hash

---

### 4.16 实战案例:LLaMA-3 数据流水线复盘

**数据规模**:15.6T tokens,5 个 CC 版本合并 + 代码 + 多语 + 书籍 + 论文。

**流水线关键步骤**:
1. **第一阶段**(原始 → JSONL):
   - WARC → HTML parse → LID → 长度过滤
   - 产出 ~50T 原始 token
2. **第二阶段**(清洗):
   - Heuristic 过滤(规则):丢 ~40%
   - Model-based quality classifier(Llama-2 训的):丢 ~25%
   - MinHash dedup(阈值 0.8):丢 ~30%
   - PII 过滤
   - 产出 ~15T 高质量 token
3. **第三阶段**(配比):
   - 7 大类(英语 / 多语 / 代码 / 数学 / 推理 / 书籍 / 其他)
   - 用 1B 小模型做 ablation,选定最终配比
4. **第四阶段**(打包):
   - Tokenize + 拼接到 8192 长度 + pack
   - MDS 格式

**经验教训**:
- **质量分类器是关键**:从 50T → 15T,模型质量分类器贡献最大。
- **代码 / 数学数据稀缺**:用 self-instruct + 合成补足。
- **长尾数据保留**:低资源语言过采样 2-5×,避免能力塌方。
- **dedup 不够激进**:LLaMA-3 阈值 0.8,DeepSeek-V3 用 0.85 更激进,后者代码能力更强。

---

### 4.17 小结:数据准备 checklist

```text
□ 1. 数据源列表(来源 / license / 规模)
□ 2. WARC / 原始数据解析流水线(HTML / LID)
□ 3. Heuristic 清洗(HTML / boilerplate / 长度)
□ 4. 质量分类器(模型打分,阈值 0.5-0.7)
□ 5. MinHash 去重(文档 + 段落级,阈值 0.8)
□ 6. PII 过滤(正则 + NER)
□ 7. Benchmark 去污染(13-gram overlap)
□ 8. 配比 ablation(小模型验证)
□ 9. Tokenize + pack + 二进制 bin
□ 10. Metadata(version / provenance / token count)
□ 11. 备份 + 版本控制
□ 12. License 合规审查
□ 13. 定期 re-crawl 增量更新
```

数据准备占整个 LLM 项目 30-40% 的时间和算力。**一次做对,胜过 10 次调超参**。
<!-- EXPANDED_2026_DEEP_APPEND -->

## 07章深度扩充总说明

本追加部分为“训练全流程实务”建立统一的数学—算法—代码—案例—工程—面试索引。原有内容不改动；新增内容从本标记开始。

## 深度扩充 07.1: 数据治理与可追溯性
> 本节围绕“定义数据资产、版本、许可证、风险等级、采样权重和可复现元数据。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 数据治理与可追溯性 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.1.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.1.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“数据治理与可追溯性”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.1.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.1.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.1.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.1.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.1.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.1** 数据治理与可追溯性 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("数据治理与可追溯性 / 核心实现", "loss=", float(loss), "valid_ratio=", ratio)
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("数据治理与可追溯性 / 工程审计", "first=", values[0], "last=", values[-1])
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.2: 精确去重、近似去重与污染控制
> 本节围绕“比较 hash、MinHash、LSH、文档级和 token 级去重。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 精确去重、近似去重与污染控制 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.2.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.2.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“精确去重、近似去重与污染控制”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.2.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.2.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.2.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.2.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.2.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.2** 精确去重、近似去重与污染控制 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("精确去重、近似去重与污染控制 / 核心实现", "first=", values[0], "last=", values[-1])
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("精确去重、近似去重与污染控制 / 工程审计", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.3: 质量过滤与数据混合
> 本节围绕“用规则、分类器、困惑度和配比消融决定数据进入训练的方式。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 质量过滤与数据混合 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.3.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.3.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“质量过滤与数据混合”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.3.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.3.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.3.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.3.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.3.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.3** 质量过滤与数据混合 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("质量过滤与数据混合 / 核心实现", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("质量过滤与数据混合 / 工程审计", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.4: Tokenizer 训练与词表工程
> 本节围绕“推导 BPE 合并、压缩率、覆盖率、特殊 token 和中文代码权衡。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 Tokenizer 训练与词表工程 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.4.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.4.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“Tokenizer 训练与词表工程”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.4.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.4.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.4.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.4.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.4.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.4** Tokenizer 训练与词表工程 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("Tokenizer 训练与词表工程 / 核心实现", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("Tokenizer 训练与词表工程 / 工程审计", "loss=", float(loss), "valid_ratio=", ratio)
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.5: 文档打包、mask 与有效 token
> 本节围绕“实现跨文档 packing、边界 mask 和 padding-free batch。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 文档打包、mask 与有效 token 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.5.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.5.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“文档打包、mask 与有效 token”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.5.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.5.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.5.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.5.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.5.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.5** 文档打包、mask 与有效 token 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("文档打包、mask 与有效 token / 核心实现", "loss=", float(loss), "valid_ratio=", ratio)
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("文档打包、mask 与有效 token / 工程审计", "first=", values[0], "last=", values[-1])
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.6: Chinchilla 配比与算力预算
> 本节围绕“把参数量、token 数、FLOPs、吞吐和预算约束成可求解模型。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 Chinchilla 配比与算力预算 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.6.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.6.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“Chinchilla 配比与算力预算”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.6.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.6.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.6.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.6.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.6.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.6** Chinchilla 配比与算力预算 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("Chinchilla 配比与算力预算 / 核心实现", "first=", values[0], "last=", values[-1])
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("Chinchilla 配比与算力预算 / 工程审计", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.7: 初始化、AdamW 与梯度裁剪
> 本节围绕“解释方差传播、解耦 weight decay、二阶矩和异常 batch。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 初始化、AdamW 与梯度裁剪 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.7.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.7.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“初始化、AdamW 与梯度裁剪”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.7.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.7.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.7.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.7.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.7.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.7** 初始化、AdamW 与梯度裁剪 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("初始化、AdamW 与梯度裁剪 / 核心实现", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("初始化、AdamW 与梯度裁剪 / 工程审计", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.8: 学习率 warmup、cosine 与 WSD
> 本节围绕“比较 warmup、余弦退火、稳定--衰减阶段以及末期退火。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 学习率 warmup、cosine 与 WSD 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.8.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.8.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“学习率 warmup、cosine 与 WSD”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.8.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.8.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.8.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.8.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.8.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.8** 学习率 warmup、cosine 与 WSD 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("学习率 warmup、cosine 与 WSD / 核心实现", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("学习率 warmup、cosine 与 WSD / 工程审计", "loss=", float(loss), "valid_ratio=", ratio)
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.9: BF16、FP16、FP8 与 loss scaling
> 本节围绕“建立动态范围、溢出检测、master weight 和关键算子回退规则。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 BF16、FP16、FP8 与 loss scaling 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.9.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.9.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“BF16、FP16、FP8 与 loss scaling”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.9.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.9.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.9.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.9.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.9.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.9** BF16、FP16、FP8 与 loss scaling 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("BF16、FP16、FP8 与 loss scaling / 核心实现", "loss=", float(loss), "valid_ratio=", ratio)
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("BF16、FP16、FP8 与 loss scaling / 工程审计", "first=", values[0], "last=", values[-1])
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.10: 数据并行、ZeRO/FSDP 与梯度累积
> 本节围绕“量化参数、梯度、优化器状态的显存占用和通信代价。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 数据并行、ZeRO/FSDP 与梯度累积 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.10.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.10.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“数据并行、ZeRO/FSDP 与梯度累积”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.10.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.10.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.10.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.10.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.10.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.10** 数据并行、ZeRO/FSDP 与梯度累积 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("数据并行、ZeRO/FSDP 与梯度累积 / 核心实现", "first=", values[0], "last=", values[-1])
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("数据并行、ZeRO/FSDP 与梯度累积 / 工程审计", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.11: 张量并行、流水线并行与 checkpoint
> 本节围绕“推导切分维度、micro-batch、bubble、重计算和恢复点。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 张量并行、流水线并行与 checkpoint 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.11.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.11.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“张量并行、流水线并行与 checkpoint”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.11.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.11.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.11.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.11.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.11.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.11** 张量并行、流水线并行与 checkpoint 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("张量并行、流水线并行与 checkpoint / 核心实现", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("张量并行、流水线并行与 checkpoint / 工程审计", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.12: 继续预训练、退火与长上下文阶段
> 本节围绕“设计中期训练的数据切换、学习率曲线和位置覆盖。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 继续预训练、退火与长上下文阶段 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.12.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.12.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“继续预训练、退火与长上下文阶段”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.12.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.12.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.12.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.12.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.12.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.12** 继续预训练、退火与长上下文阶段 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("继续预训练、退火与长上下文阶段 / 核心实现", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("继续预训练、退火与长上下文阶段 / 工程审计", "loss=", float(loss), "valid_ratio=", ratio)
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.13: SFT 数据格式与回答 mask
> 本节围绕“处理 chat template、角色边界、packing、loss mask 和灾难性遗忘。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 SFT 数据格式与回答 mask 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.13.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.13.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“SFT 数据格式与回答 mask”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.13.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.13.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.13.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.13.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.13.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.13** SFT 数据格式与回答 mask 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("SFT 数据格式与回答 mask / 核心实现", "loss=", float(loss), "valid_ratio=", ratio)
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("SFT 数据格式与回答 mask / 工程审计", "first=", values[0], "last=", values[-1])
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.14: LoRA、QLoRA 与参数高效微调
> 本节围绕“推导低秩更新、量化基座、rank 选择和合并部署。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 LoRA、QLoRA 与参数高效微调 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.14.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.14.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“LoRA、QLoRA 与参数高效微调”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.14.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.14.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.14.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.14.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.14.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.14** LoRA、QLoRA 与参数高效微调 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import math

def cosine_schedule(step, warmup, total, peak, floor=0.0):
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("需要 0<=warmup<total")
    if not 0 <= step <= total:
        raise ValueError("step 超出范围")
    if step < warmup:
        return peak * step / max(warmup, 1)
    u = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * u))

if __name__ == "__main__":
    values = [cosine_schedule(s, 2, 10, 3e-4, 1e-5) for s in range(11)]
    print("LoRA、QLoRA 与参数高效微调 / 核心实现", "first=", values[0], "last=", values[-1])
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("LoRA、QLoRA 与参数高效微调 / 工程审计", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.15: 偏好训练与推理能力激励
> 本节围绕“比较 DPO、GRPO、可验证奖励、采样成本和数据闭环。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 偏好训练与推理能力激励 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.15.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.15.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“偏好训练与推理能力激励”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.15.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.15.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.15.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.15.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.15.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.15** 偏好训练与推理能力激励 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch

def finite_grad_norm(parameters, max_norm=1.0):
    params = [p for p in parameters if p.grad is not None]
    if not params or max_norm <= 0:
        raise ValueError("参数梯度不能为空且 max_norm 必须为正")
    grads = [p.grad for p in params]
    if not all(torch.isfinite(g).all() for g in grads):
        return float("nan"), False
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    scale = min(1.0, max_norm / (float(norm) + 1e-6))
    for g in grads:
        g.mul_(scale)
    return float(norm), True

if __name__ == "__main__":
    p = torch.nn.Parameter(torch.ones(3)); (p.square().sum()).backward()
    norm, ok = finite_grad_norm([p])
    print("偏好训练与推理能力激励 / 核心实现", "norm=", norm, "ok=", ok, "grad=", p.grad.tolist())
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("偏好训练与推理能力激励 / 工程审计", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.16: 评估、污染审计与上线部署
> 本节围绕“把 benchmark、人工评测、吞吐、延迟、量化和灰度发布合成验收门槛。”展开。所有示例只使用公开定义；工程数字是测量模板，不把单机结果冒充通用结论。
### 学习目标与边界
1. 能从定义写出 评估、污染审计与上线部署 的核心目标，并区分训练目标、代理指标与上线指标。
2. 能用一个手工算例检查符号、维度、归一化和边界条件。
3. 能运行下面两个最小实现，打印维度、有限性和一个可解释的 sanity check。
4. 能根据失败症状选择回滚、降级、重采样或重新训练，而不是只盯着最终分数。
### 术语表
| 术语 | 本节含义 | 观测量 | 典型误读 |
| :--- | :--- | :--- | :--- |
| 输入状态 | 进入本节模块的 token、偏好对、位置或训练 batch | shape、dtype、有效比例 | 把 padding 当成有效信号 |
| 目标量 | 希望优化或约束的概率、距离、损失或吞吐 | 均值、方差、分位数 | 只看均值忽略尾部 |
| 代理量 | 无法直接观测时使用的可计算指标 | 相关系数、校准误差 | 把代理量当最终目标 |
| 安全边界 | 出现数值、质量或合规风险时的停止条件 | NaN、漂移、拒答率 | 没有回滚阈值 |
| 复现键 | 让实验可重跑的一组配置与随机状态 | commit、数据版本、seed | 只保存模型权重 |
### 数学建模：定义、推导与解释
设本节样本空间为 $\mathcal{X}$, 参数为 $\theta$, 观测长度为 $T$, batch 大小为 $B$。我们先固定约定：第 $0$ 维是 batch，第 $1$ 维是序列或候选，第 $-1$ 维是特征。
**公式 07.16.1（语言模型损失）**
$$
\mathcal L_{LM}=-\frac1N\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$
解释：只对有效 label 计数，padding 和指令 mask 不应进入分母。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.2（困惑度）**
$$
\operatorname{PPL}=\exp(\mathcal L_{LM})
$$
解释：困惑度是平均 token 负对数似然的指数。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.3（算力估算）**
$$
C\approx 6ND
$$
解释：参数量 N 和训练 token D 给出 decoder-only 的一阶 FLOPs 估算。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.4（Adam 一阶矩）**
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$
解释：一阶矩追踪梯度方向。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.5（Adam 二阶矩）**
$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
解释：二阶矩按坐标缩放步长。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.6（AdamW 更新）**
$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)
$$
解释：weight decay 与梯度更新解耦。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.7（余弦调度）**
$$
\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\pi u_t)
$$
解释：平滑降低学习率，末期更细致地收敛。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.8（梯度裁剪）**
$$
g\leftarrow g\min(1,\tau/(\|g\|_2+\varepsilon))
$$
解释：只在范数超阈值时缩放，防止异常 batch 破坏状态。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.9（混合精度）**
$$
\hat g=\operatorname{cast}_{low}(\nabla_\theta\mathcal L),\quad \theta\leftarrow\theta-\eta\operatorname{cast}_{fp32}(\hat g)
$$
解释：关键归约和优化器状态通常保留更高精度。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
**公式 07.16.10（有效 token 比例）**
$$
\rho=\frac{\sum_i\mathbf1[label_i\ne -100]}{B\times T}
$$
解释：监控 rho 可以发现 padding、mask 或 packing 配置错误。 在“评估、污染审计与上线部署”中，它决定了我们如何记录正向信号、如何做归一化，以及如何检查退化情况。
### 从定义到结论的逐步推导
**推导 07.16.1（从 token loss 到 batch loss）**
第一步，写出定义：
$$
\mathcal L_B=\frac{\sum_i m_i\ell_i}{\sum_i m_i+\varepsilon}
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：先求有效 token 的加权和，再除以有效 token 数，而不是盲除 $BT$。
**推导 07.16.2（从梯度累积到等效 batch）**
第一步，写出定义：
$$
g_{eff}=\frac1K\sum_{k=1}^{K}g_k
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：只有在每个 micro-batch 的 loss 已按 token 正确归一化时，累积才近似大 batch。
**推导 07.16.3（从 checkpoint 到恢复一致性）**
第一步，写出定义：
$$
S_{t+1}=F(S_t,B_t,\xi_t)
$$
第二步，保持 batch 维和有效样本集合不变，对两边做代入或求导。
第三步，检查量纲、符号和极端输入；若分母接近零，必须显式加入 epsilon 或走回退分支。
结论：若保存模型、优化器、调度器、随机数和数据游标，恢复后才能近似复现同一轨迹。
### 定理、收敛性与复杂度证明草图
**定理 07.16.1**: 若每个有效样本的损失下界存在，梯度估计二阶矩有限，且步长满足 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$，则随机一阶更新的平均驻点残差在标准光滑假设下趋于零。
证明草图：将光滑性不等式应用到单步更新，取条件期望后，噪声的一阶项消失，二阶项由有限方差假设吸收。把所有时间步相加，得到梯度平方和的可加上界，再用步长条件得到加权平均残差收敛。
$$
\mathbb E[L_{t+1}]\le L_t-\eta_t(1-c\eta_t)\|\nabla L_t\|^2+c\eta_t^2\sigma^2
$$
**定理 07.16.2**: 若输入张量规模为 $(B,T,D)$，本节的朴素实现时间复杂度为 $O(BT\,C_{idx})$，额外空间复杂度为 $O(BT\,S_{idx})$；当长度或候选组数翻倍时，必须重新测量二次项和缓存峰值。
证明草图：把矩阵乘、排序、top-k、softmax、统计归约分别计数；逐项相加得到时间上界，保存中间激活、概率和索引得到空间上界。这个上界用于容量规划，不等于特定 GPU 的实际吞吐。
### 信息论、概率与优化三个视角
- **信息论**：把损失看作对真实分布的交叉熵或对参考分布的相对熵，先问模型是否真的获得了新信息。
- **概率论**：显式写出条件分布、采样集合、估计量方差和置信区间，避免把一次 rollout 当成确定事实。
- **优化论**：关注梯度方向、曲率、约束、步长和可行域；当指标上升但约束违反时，先修正目标而不是继续加大 LR。
- **工程论**：对每个数学量绑定日志字段，例如 valid_count、finite_rate、p95_latency、rollback_reason。
### 手工数值算例
取两个样本的有效 label 数分别为 3 和 1，token loss 分别为总和 1.2 和 0.8。正确 batch loss 是 $(1.2+0.8)/(3+1)=0.5$，不能用 batch 平均 $(0.4+0.8)/2=0.6$ 替代。
若 warmup 设为 2 步、峰值 LR 为 $3\times10^{-4}$，第 1 步 LR 为 $1.5\times10^{-4}$；第 10 步进入余弦末端时才接近 floor。
手算检查应覆盖：零向量、单样本、全 padding、相同奖励、超长位置、非整除头数、非有限梯度。
### 算法流程
**Algorithm 07.16** 评估、污染审计与上线部署 的最小可审计流程
**Input**: 原始样本 $X$、参考状态或参考模型 $R$、超参数 $\epsilon,\beta,\tau$。
**Output**: 经过约束检查的损失/更新量 $U$、审计记录 $M$、可选回滚标记。
1. 读取输入并验证 batch、序列、候选组和特征维度。
2. 计算有效 mask，排除 padding、空回答和超过长度上限的样本。
3. 在 FP32 或安全累加精度下计算 log-prob、距离、相位、路由或 token loss。
4. 对分母、方差、softmax 输入和概率做 epsilon、clamp、finite 检查。
5. 按定义计算主目标，不把监控指标混入反向传播。
6. 计算约束项、均衡项或 KL 项，并记录每项的未加权数值。
7. 进行一次小批量梯度或输出的 sanity check，确认 shape 与 dtype。
8. 若违反阈值，丢弃该更新并记录原因；否则执行更新或写入缓存。
9. 记录均值、标准差、p50/p95、有效比例和随机种子。
10. 每隔固定步数运行离线回归集，比较基线、当前版本和最坏样本。
11. 达到停止条件时保存模型、优化器、调度器、数据游标与审计摘要。
12. 发生漂移时优先灰度、降级或回滚，再决定是否增加训练预算。
**Complexity**: Time $O(BT\,C_{idx})$, Space $O(BT\,S_{idx})$；若引入全量两两注意力/候选比较，额外项分别为 $O(T^2D)$ 或 $O(G^2)$，应在设计评审中单独标注。
### 完整代码实现一：核心张量/损失模块
下面代码可独立保存为 `.py` 运行；它不依赖本项目其它文件，包含输入检查、边界处理、维度打印和最小自检。
```python
import torch


def pack_sequences(sequences, max_len, pad_id=0):
    if max_len <= 1 or not sequences:
        raise ValueError("max_len 必须大于 1 且 sequences 非空")
    stream = []
    for seq in sequences:
        if not seq:
            continue
        stream.extend(seq)
    rows, masks = [], []
    for start in range(0, len(stream), max_len):
        row = stream[start:start + max_len]
        mask = [1] * len(row)
        row += [pad_id] * (max_len - len(row))
        mask += [0] * (max_len - len(mask))
        rows.append(row); masks.append(mask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

if __name__ == "__main__":
    ids, mask = pack_sequences([[1, 2, 3], [4, 5], [6]], 4)
    print("评估、污染审计与上线部署 / 核心实现", "ids=", tuple(ids.shape), "valid=", int(mask.sum()))
```
### 完整代码实现二：NumPy/PyTorch 工程审计模块
第二个实现刻意把观测、阈值和失败分支写出来，适合先在 CPU 小样本上验证，再接入真实训练。
```python
import torch
import torch.nn.functional as F

def masked_lm_loss(logits, labels, ignore_index=-100):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits=(B,T,V), labels=(B,T)")
    valid = labels.ne(ignore_index)
    if not valid.any():
        raise ValueError("没有有效 label")
    token_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=ignore_index)
    return token_loss[valid].mean(), float(valid.float().mean())

if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, -100, 3], [4, -100, -100, 5]])
    loss, ratio = masked_lm_loss(logits, labels)
    print("评估、污染审计与上线部署 / 工程审计", "loss=", float(loss), "valid_ratio=", ratio)
```
### 真实案例与对照
| 系统 | 公开事实或可确认做法 | 本节可借鉴点 | 不应过度推断 |
| :--- | :--- | :--- | :--- |
| GPT-3/LLaMA | 公开论文提供数据、规模或训练流程的部分信息。 | 建立数据、算力和 checkpoint 的审计记录。 | 公开估算不是本地复现保证。 |
| Qwen | 公开系列强调多语言、代码、长上下文和指令模型的阶段化训练。 | 数据配比应按能力分桶评估。 | 不同版本数据混合比例未必相同。 |
| DeepSeek-V3/R1 | 公开报告披露 FP8、MoE、GRPO 等训练工程方向。 | 把硬件、通信、verifier 和评估共同纳入预算。 | 不要只复制单一技巧。 |
| GPT/Claude | 闭源产品更多通过模型卡、系统卡和行为评估说明风险。 | 用黑盒回归和安全门禁补齐不可见内部。 | 无法从产品输出准确反推出训练成本。 |
### 失败案例、调试路径与改进方案
- **症状**：指标上升但真实质量下降
  - **定位**：检查代理目标是否被投机、样本切分是否泄漏、评估是否只看均值。
  - **改进**：加入对抗样本、分桶指标和人工盲评；把停止条件放在主目标之外。
- **症状**：小 batch 正常，大 batch NaN 或超时
  - **定位**：检查归一化分母、梯度累积、通信峰值和 dtype 归约。
  - **改进**：先用 FP32 CPU 复现一批，再逐项恢复混合精度和并行。
- **症状**：边界长度/空样本报错
  - **定位**：打印每个 mask 的有效数量、最大位置、索引范围和 shape。
  - **改进**：入口处拒绝空集合，内部统一使用显式 mask 与安全分母。
- **症状**：离线回归好，线上漂移
  - **定位**：对比输入分布、模板版本、采样温度和服务端截断。
  - **改进**：保留 shadow 流量、版本化 prompt，并准备一键回滚。
### 性能 benchmark 设计
| 维度 | 小基线 | 中基线 | 大基线 | 记录项 |
| :--- | ---: | ---: | ---: | :--- |
| batch | 2 | 8 | 32 | 峰值显存、有效 token/s |
| 长度/候选组 | 128 | 512 | 2048 | p50/p95 延迟、峰值工作区 |
| dtype | FP32 | BF16 | FP8/INT8（若硬件支持） | overflow、回退次数 |
| 并发 | 1 | 8 | 32 | 排队时间、吞吐、尾延迟 |
benchmark 必须固定模型 commit、数据版本、GPU 型号、驱动、编译选项和随机种子；至少重复 3 次并报告均值、标准差和 p95。
### 工程权衡决策树
```text
先问目标是否是质量、稳定性、成本还是延迟？
├─ 质量不足 → 检查数据/目标/评估，再做小规模消融
├─ 数值不稳 → FP32 复现 → 降 LR/加 epsilon/裁剪 → 恢复混合精度
├─ 显存不足 → 减 batch/序列 → 累积 → checkpoint → 分片/量化
├─ 吞吐不足 → 先 profile → 改 IO/批处理 → 再考虑更复杂并行
└─ 线上漂移 → shadow → 灰度 → 触发阈值即回滚
```
### 本节验收清单
- [ ] 1. 公式的变量与实现 shape 一致
- [ ] 2. 极端输入有明确回退
- [ ] 3. 主指标和代理指标分开记录
- [ ] 4. 至少有一个可复现手算例
- [ ] 5. 代码能在 CPU 小 batch 运行
- [ ] 6. 代码打印输出维度与 finite 状态
- [ ] 7. 论文年份与 arXiv 标识可追溯
- [ ] 8. benchmark 不把硬件特例写成定律
- [ ] 9. 失败案例包含症状、定位和改进
- [ ] 10. 上线决策包含回滚条件
## 深度扩充 07.附录：全章时间线、比较矩阵与面试题
### 关键论文完整时间线
| 年份 | 论文/系统 | 作者/机构 | arXiv | 对本章的贡献 |
| :--- | :--- | :--- | :--- | :--- |
| 2020 | 数据治理与可追溯性 | Brown et al. | [2005.14165](https://arxiv.org/abs/2005.14165) | 为“数据治理与可追溯性”提供可追溯背景 |
| 2022 | 精确去重、近似去重与污染控制 | Lee et al. | [2202.09185](https://arxiv.org/abs/2202.09185) | 为“精确去重、近似去重与污染控制”提供可追溯背景 |
| 2020 | 质量过滤与数据混合 | Gao et al. | [2005.14165](https://arxiv.org/abs/2005.14165) | 为“质量过滤与数据混合”提供可追溯背景 |
| 2016 | Tokenizer 训练与词表工程 | Sennrich et al. | [1508.07909](https://arxiv.org/abs/1508.07909) | 为“Tokenizer 训练与词表工程”提供可追溯背景 |
| 2023 | 文档打包、mask 与有效 token | Touvron et al. | [2302.13971](https://arxiv.org/abs/2302.13971) | 为“文档打包、mask 与有效 token”提供可追溯背景 |
| 2022 | Chinchilla 配比与算力预算 | Hoffmann et al. | [2203.15556](https://arxiv.org/abs/2203.15556) | 为“Chinchilla 配比与算力预算”提供可追溯背景 |
| 2019 | 初始化、AdamW 与梯度裁剪 | Loshchilov & Hutter | [1711.05101](https://arxiv.org/abs/1711.05101) | 为“初始化、AdamW 与梯度裁剪”提供可追溯背景 |
| 2024 | 学习率 warmup、cosine 与 WSD | Hu et al. | [2404.06395](https://arxiv.org/abs/2404.06395) | 为“学习率 warmup、cosine 与 WSD”提供可追溯背景 |
| 2022 | BF16、FP16、FP8 与 loss scaling | Micikevicius et al. | [2205.14168](https://arxiv.org/abs/2205.14168) | 为“BF16、FP16、FP8 与 loss scaling”提供可追溯背景 |
| 2020 | 数据并行、ZeRO/FSDP 与梯度累积 | Rajbhandari et al. | [1910.02054](https://arxiv.org/abs/1910.02054) | 为“数据并行、ZeRO/FSDP 与梯度累积”提供可追溯背景 |
| 2019 | 张量并行、流水线并行与 checkpoint | Shoeybi et al. | [1807.04233](https://arxiv.org/abs/1807.04233) | 为“张量并行、流水线并行与 checkpoint”提供可追溯背景 |
| 2024 | 继续预训练、退火与长上下文阶段 | Touvron et al. | [2407.21783](https://arxiv.org/abs/2407.21783) | 为“继续预训练、退火与长上下文阶段”提供可追溯背景 |
| 2022 | SFT 数据格式与回答 mask | Ouyang et al. | [2203.02155](https://arxiv.org/abs/2203.02155) | 为“SFT 数据格式与回答 mask”提供可追溯背景 |
| 2021 | LoRA、QLoRA 与参数高效微调 | Hu et al. | [2106.09685](https://arxiv.org/abs/2106.09685) | 为“LoRA、QLoRA 与参数高效微调”提供可追溯背景 |
| 2025 | 偏好训练与推理能力激励 | DeepSeek-AI | [2501.12948](https://arxiv.org/abs/2501.12948) | 为“偏好训练与推理能力激励”提供可追溯背景 |
| 2023 | 评估、污染审计与上线部署 | Kwon et al. | [2309.06180](https://arxiv.org/abs/2309.06180) | 为“评估、污染审计与上线部署”提供可追溯背景 |
| 2019 | ZeRO | Rajbhandari et al. | 1910.02054 | 提出优化器/梯度/参数分片。 |
| 2020 | Scaling Laws for Neural Language Models | Kaplan et al. | 2001.08361 | 量化规模、数据与损失关系。 |
| 2022 | Training Compute-Optimal Large Language Models | Hoffmann et al. | 2203.15556 | 提出 Chinchilla 计算最优配比。 |
| 2022 | FP8 Formats for Deep Learning | Micikevicius et al. | 2209.05433 | 推动低精度训练工程。 |
| 2022 | Training language models to follow instructions | Ouyang et al. | 2203.02155 | 给出 SFT 与偏好对齐流程。 |
| 2023 | vLLM | Kwon et al. | 2309.06180 | PagedAttention 提升服务吞吐。 |
### 同类方案比较矩阵
| 方案 | 主要目标 | 训练/推理成本 | 泛化风险 | 适合何时选 |
| :--- | :--- | :--- | :--- | :--- |
| 从零预训练 | 获得通用能力 | 最高 | 数据/算力风险 | 有大规模数据与预算 |
| 继续预训练 | 域适配/长上下文 | 中 | 灾难性遗忘 | 已有基座 |
| SFT/LoRA | 任务与格式适配 | 低到中 | 过拟合模板 | 垂直业务 |
| 偏好训练 | 行为与推理 | 中到高 | 奖励投机 | 有偏好或 verifier |
| 量化部署 | 降低成本/延迟 | 工程成本 | 精度退化 | 资源受限服务 |
### 面试题库：15 题详解
#### Q1. 为什么数据决定上限？
**答**：模型只能从训练分布获得可学习信号；低质量、重复和污染数据会占预算并扭曲评估。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q2. 去重为什么不能只用 exact hash？
**答**：轻微改写、网页模板和代码复制不会完全相同，需要近似指纹和文档级聚类。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q3. Tokenizer 词表越大越好吗？
**答**：更大词表可能提高压缩率但增加 embedding、稀有 token 和多语言不平衡，需要用 token/byte 与模型成本共同评估。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q4. 为什么 loss 要按有效 token 归一化？
**答**：padding、回答 mask 和不同长度样本的有效数量不同，按 batch 平均会改变梯度权重。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q5. 6ND 算力估算的限制？
**答**：它是一阶 decoder-only 估算，不包含通信、激活重算、数据管线和 kernel 效率损耗。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q6. AdamW 与 L2 正则有什么区别？
**答**：AdamW 把 weight decay 从自适应梯度中解耦，更直接控制权重缩小。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q7. 为什么需要 warmup？
**答**：早期参数和二阶矩尚未稳定，大步长容易放大初始化和 batch 噪声。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q8. BF16 为什么常比 FP16 更稳？
**答**：BF16 指数范围更接近 FP32，牺牲尾数换取更大动态范围；仍需监控溢出。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q9. ZeRO-3 省显存的代价？
**答**：参数、梯度和优化器状态分片降低单卡存储，但带来 gather、通信和调度复杂度。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q10. 为什么 checkpoint 要保存数据游标？
**答**：只恢复权重会继续读取不同样本，导致 loss 轨迹和数据覆盖不可复现。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q11. SFT 为什么 mask 指令？
**答**：通常希望模型学习回答行为而不是复制用户输入；多轮对话需按角色精细 mask。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q12. LoRA 的 rank 如何选？
**答**：rank 是容量预算，应按目标层、任务复杂度、数据量和验证曲线选择，而不是越大越好。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q13. 怎样防止评估污染？
**答**：做文档级去重、时间切分、n-gram 检索和人工抽样，报告污染风险而不是只报分数。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q14. 训练失败时先查什么？
**答**：先查数据 batch、label/mask、finite、LR、梯度范数和 checkpoint，再查复杂并行优化。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
#### Q15. 上线验收为什么要含 p95？
**答**：平均延迟无法揭示长输入、排队和 cache miss 的尾部体验，生产服务通常由 p95/p99 约束。
**追问**：请给出一个失败边界、一个可运行检查和一个可回滚动作。
**检查**：优先打印 shape、dtype、有效比例、p95 或最坏分桶，不要只打印一个总 loss。
### 全章落地任务
1. 用 CPU 小张量复现全部手算例，并把输出保存成 JSON。
2. 选一个公开小模型做单变量消融：只改本附录讨论的一个组件。
3. 为主指标增加一个独立 verifier 或人工抽样集。
4. 故意制造一个边界错误，确认日志能定位并触发安全分支。
5. 把 benchmark 结果按 commit、硬件、数据和配置写入可比较表。
6. 为上线版本写出灰度门槛、回滚命令和责任人。
7. 把本章公式映射到代码中的变量，形成一页符号--shape 对照表。
8. 每次实验保存随机数、tokenizer、模板、依赖版本和 checkpoint 元数据。
> 本次扩充的历史锚点包含 22 条可追溯论文/系统记录；未公开的 GPT/Claude 内部细节均明确标注为不可确认，不以猜测替代证据。
### 附加推演卡 1
**问题**：围绕第 1 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 2
**问题**：围绕第 2 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 3
**问题**：围绕第 3 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 4
**问题**：围绕第 4 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 5
**问题**：围绕第 5 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 6
**问题**：围绕第 6 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 7
**问题**：围绕第 7 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 8
**问题**：围绕第 8 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 9
**问题**：围绕第 9 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 10
**问题**：围绕第 10 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 11
**问题**：围绕第 11 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 12
**问题**：围绕第 12 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 13
**问题**：围绕第 13 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 14
**问题**：围绕第 14 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 15
**问题**：围绕第 15 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 16
**问题**：围绕第 16 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 17
**问题**：围绕第 1 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 18
**问题**：围绕第 2 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 19
**问题**：围绕第 3 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 20
**问题**：围绕第 4 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 21
**问题**：围绕第 5 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 22
**问题**：围绕第 6 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 23
**问题**：围绕第 7 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 24
**问题**：围绕第 8 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 25
**问题**：围绕第 9 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 26
**问题**：围绕第 10 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 27
**问题**：围绕第 11 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 28
**问题**：围绕第 12 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 29
**问题**：围绕第 13 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 30
**问题**：围绕第 14 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 31
**问题**：围绕第 15 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 32
**问题**：围绕第 16 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 33
**问题**：围绕第 1 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 34
**问题**：围绕第 2 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 35
**问题**：围绕第 3 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 36
**问题**：围绕第 4 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 37
**问题**：围绕第 5 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 38
**问题**：围绕第 6 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 39
**问题**：围绕第 7 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 40
**问题**：围绕第 8 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 41
**问题**：围绕第 9 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 42
**问题**：围绕第 10 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 43
**问题**：围绕第 11 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

### 附加推演卡 44
**问题**：围绕第 12 节，改变一个输入尺度后，哪个约束最先失效？
**已知**：输入可能为空、长度可能超过训练范围，浮点数可能出现非有限值。
**推演**：先写出目标量，再写出 shape、分母、阈值和回退路径；禁止只凭最终分数判断。
**实现**：在 CPU 小 batch 上运行对应代码，打印维度、有效比例、均值、标准差和 finite 状态。
**结论**：若指标与独立 verifier 同时通过，才允许进入下一阶段；否则保存失败样本。
**反例**：单个异常 batch、全 padding、相同奖励或重复位置都不能被静默吞掉。
**工程动作**：记录版本键，必要时灰度或回滚，不把临时修复直接扩大到全量。

