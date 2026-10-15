---
title: 2-Query-Item语义对齐
tags:
  - 方向
  - SID
description: Query tower → query_emb...
chapter_id: 5.2
last_updated: 2026-07-14
status: ✅ v1.0

---

# 2. Query-Item 语义对齐

> **本章定位**: 双塔 vs 端到端 SID 排序
> **预计时长**: 45 分钟

---

## 2.1 双塔架构 (召回)

```text
Query tower → query_emb
Item tower  → item_emb
Similarity  → top-K 候选
```

```python
# 离线算好 item SID
item_sids = model.encode(items)  # [N, K]

# 在线
query_emb = query_tower(query)
# 候选集 SID 与 query 在同一空间
scores = compute_similarity(query_emb, item_sids)
```

**特点**: 召回快,可离线预计算 SID。

---

## 2.2 端到端架构 (排序)

```text
Query + Item (SID) → Transformer → 排序分数
```

```python
# 输入
input_sids = torch.stack([history_sids, candidate_sid], dim=1)  # [B, T, K]
# Transformer
score = rank_model(input_sids)  # [B, 1]
```

**特点**: 排序精准,但需要 query 与 item SID 在同一空间。

---

## 2.3 语义对齐

让 query_emb 与 item_emb 在同一向量空间。

```python
class AlignmentModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(1024, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.item_tower = nn.Sequential(nn.Linear(1024, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, query_emb, item_emb, labels):
        q = self.query_tower(query_emb)
        i = self.item_tower(item_emb)
        # 余弦相似度
        sim = F.cosine_similarity(q, i, dim=-1)  # [B]
        # BCE 损失
        loss = F.binary_cross_entropy_with_logits(sim, labels.float())
        return loss
```

**损失**: 对齐 + 均匀性

```python
def align_uniform_loss(q, i, t=2.0):
    """Alignment + Uniformity"""
    # Alignment: q 和 i 应该接近
    align = (q - i).norm(dim=-1).pow(2).mean()
    # Uniformity: q 应该分散
    uniform = torch.pdist(q, p=2).pow(2).mul(-t).exp().mean().log()
    return align + 0.5 * uniform
```

---

## 2.4 工业实践

| 场景 | 架构 | 原因 |
| :--- | :--- | :--- |
| 召回 | 双塔 | 快,离线预计算 SID |
| 排序 | 端到端 | 精准,需要 SID 互相关联 |
| 搜索 | 端到端 | query-item 强相关 |

---

## 2.5 下一步

[3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)---

## 3. Query-Item 对齐的理论基础

### 3.1 查询与推荐的本质差异

**搜索 vs 推荐**:
- **搜索(Search)**:用户有明确意图(query),需要精确匹配
- **推荐(Recommendation)**:用户无明确意图,需要探索兴趣

**Query 的特点**:
- 主动表达(用户知道想要什么)
- 通常简短(1-5 个词)
- 反映即时意图

**Item 的特点**:
- 被动展现
- 信息丰富(标题、描述、图像)
- 长期属性

**核心挑战**:理解用户意图,从海量商品中找到匹配。

### 3.2 Query 理解

**Query 处理流程**:
1. **分词**:中文分词、英文 tokenize
2. **改写**:同义词、纠错、扩展
3. **意图识别**:导航型、信息型、交易型
4. **类目预测**:用户想买什么类目
5. **实体识别**:品牌、型号、属性

**Query 类型分类**:

| 类型 | 例子 | 特征 |
| :--- | :--- | :--- |
| 品牌型 | "Nike 跑鞋" | 强导航意图 |
| 类目型 | "连衣裙" | 弱导航 |
| 属性型 | "红色 大码" | 强约束 |
| 长尾型 | "iPhone 15 Pro Max 256 远峰蓝" | 精确 |
| 模糊型 | "好看的礼物" | 探索 |

### 3.3 Query-Item 语义对齐

**目标**:把 query 和 item 映射到同一向量空间。

**Query encoder**:
$$
\mathbf{e}_q = f_q(\text{query})
$$

**Item encoder**:
$$
\mathbf{e}_i = f_i(\text{item})
$$

**相似度**:
$$
\text{score}(q, i) = \cos(\mathbf{e}_q, \mathbf{e}_i)
$$

### 3.4 对齐损失详解

**InfoNCE 损失**:
$$
\mathcal{L}_{\text{align}} = -\log \frac{\exp(\cos(\mathbf{e}_q, \mathbf{e}_i^+)/\tau)}{\sum_{j} \exp(\cos(\mathbf{e}_q, \mathbf{e}_i^j)/\tau)}
$$

**多正样本**:
- 一个 query 可能对应多个相关 item
- 用 soft label:
$$
\mathcal{L} = -\sum_{i} y_i \log \frac{\exp(\cos(\mathbf{e}_q, \mathbf{e}_i)/\tau)}{\sum_j \exp(\cos(\mathbf{e}_q, \mathbf{e}_j)/\tau)}
$$
其中 $y_i$ 是相关性标签(0/1/2/3)。

### 3.5 双编码器 vs 交叉编码器

**双编码器(Bi-Encoder)**:
- Query 和 item 独立编码
- 适合大规模检索(离线预计算 item embedding)

**交叉编码器(Cross-Encoder)**:
- Query 和 item 拼接,联合编码
- 精度高,但慢

**混合方案**:
- 召回:双编码器
- 排序:交叉编码器

### 3.6 Query 编码器

**BERT 风格**:
- 输入:[CLS] query [SEP]
- 输出:[CLS] embedding

**Sentence-BERT**:
- Siamese network
- 适合句子级语义

**专用查询编码器**:
- ColBERT:late interaction
- SPLADE:稀疏 + 密集

### 3.7 Item 编码器

**内容编码器**:
- 文本(标题、描述):BERT
- 图像:ViT
- 属性:Embedding

**SID 编码器**:
- 内容 → 连续向量 → RQ-VAE → SID

**协同编码器**(可选):
- 加入用户-商品交互信号
- 提升相关性

### 3.8 工业级实现

**召回阶段**:
- 双编码器
- 离线预计算 10 亿商品 embedding
- FAISS 检索 top-1000

**粗排阶段**:
- 双编码器 + 轻量级交互
- 评分 top-100

**精排阶段**:
- 交叉编码器(双塔+attention)
- 评分 top-10

### 3.9 Query 改写与扩展

**同义词替换**:
- "笔记本" → "笔记本电脑"
- "跑鞋" → "跑步鞋"

**拼写纠错**:
- "iphne" → "iphone"
- "耐克服" → "耐克服"

**Query 扩展**:
- "iPhone" → ["iPhone 15", "iPhone 15 Pro", "Apple 手机"]

**Query 改写代码**:
```python
class QueryRewriter:
    def __init__(self):
        self.synonym_dict = {
            '笔记本': '笔记本电脑',
            '跑鞋': '跑步鞋',
            # ...
        }
        self.spell_checker = SpellChecker()

    def rewrite(self, query):
        # 拼写纠错
        query = self.spell_checker.correct(query)
        # 同义词替换
        for k, v in self.synonym_dict.items():
            query = query.replace(k, v)
        return query
```

### 3.10 难负样本挖掘

**In-batch negatives**:
- Batch 内其他 query 的正样本作为负样本

**跨 batch 难负样本**:
- 用队列维护最近的 item embedding
- 找最相似的作为难负样本

**BM25 难负样本**:
- 用 BM25 找文本相似但语义不相关的商品
- 作为难负样本

**代码**:
```python
def mine_hard_negatives(query_emb, item_embs, k=10):
    """挖掘难负样本"""
    sim = query_emb @ item_embs.T  # [N, M]
    # 排除正样本(标签)
    # 取 top-K 作为难负样本
    hard_neg_idx = sim.topk(k, dim=-1).indices
    return item_embs[hard_neg_idx]
```

### 3.11 多语言 Query-Item 对齐

**场景**:跨境电商,英文 query 搜中文商品。

**方案**:
- 多语言 encoder(XLM-R, mBERT)
- 翻译对数据
- 跨语言对比学习

**多语言对齐损失**:
$$
\mathcal{L}_{\text{multilingual}} = \sum_{(q_l, i)} \mathcal{L}_{\text{align}}(f_l(q_l), f_i(i))
$$

### 3.12 长尾 Query 处理

**问题**:长尾 query 训练样本少。

**方案**:
1. **Query 改写**:把长尾 query 改写为热门 query
2. **类目预测**:用类目信息辅助
3. **元学习**:Meta-learning 学习快速适应
4. **数据增强**:用 LLM 生成同义 query

### 3.13 实时 Query 理解

**在线流程**:
```text
Query 输入 → 分词 → 改写 → 意图识别 → 类目预测 → 编码 → 检索
```

**延迟要求**:< 10ms

**优化**:
- 预加载模型
- 缓存常见 query 结果
- 异步并行

### 3.14 Query-Item 匹配的评估

**离线指标**:
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)

**在线指标**:
- CTR(点击率)
- CVR(转化率)
- GMV
- 跳出率

### 3.15 SID 在 Query-Item 对齐中的作用

**传统 ID 检索的痛点**:
- 新商品无 ID embedding
- 难以 cold-start

**SID 的优势**:
- 新商品根据内容生成 SID
- 自动进入检索空间

**Query → SID**:
- Query 编码为向量
- 与商品 SID 在同一空间
- 直接检索

**Item → SID**:
- 内容 → RQ-VAE → SID
- SID 是商品的"语言"

### 3.16 SID 检索 vs 向量检索

**向量检索**:
- 相似度:cosine / dot product
- 索引:FAISS / Milvus

**SID 检索**:
- 相似度:汉明距离 / 序列匹配
- 索引:Trie 树 / 倒排索引

**对比**:

| 维度 | 向量检索 | SID 检索 |
| :--- | :--- | :--- |
| 检索速度 | 快(FAISS) | 极快(Trie) |
| 索引大小 | 大 | 小 |
| 精度 | 高 | 中(可加 rerank) |
| 冷启动 | 中 | 优秀 |
| 可解释性 | 低 | 中(码本可解释) |

### 3.17 SID 序列匹配

**前缀匹配**:
- 用户 query → SID
- 找 SID 前缀匹配的商品
- 类似 Trie 树检索

**汉明距离**:
- 不同位置数 / 总长度
- 越小越相似

**编辑距离**:
- SID 序列的编辑距离
- 允许插入/删除/替换

**代码**:
```python
def hamming_distance(sid1, sid2):
    """汉明距离"""
    return sum(a != b for a, b in zip(sid1, sid2))

def prefix_match(query_sid, item_sids, prefix_len=2):
    """前缀匹配"""
    matches = []
    q_prefix = query_sid[:prefix_len]
    for sid in item_sids:
        if sid[:prefix_len] == q_prefix:
            matches.append(sid)
    return matches
```

### 3.18 端到端生成式 Query-Item 匹配

**TIGER / HSTU 风格**:
- 用户 query 编码为 SID
- 用 Transformer 生成商品 SID
- 不需要 ANN 检索

**优势**:
- 端到端
- 冷启动友好
- 不需要索引

**劣势**:
- 推理慢
- 训练成本高

### 3.19 SID 的可解释性

**码本可解释**:
- 每个码本索引对应一个语义类别
- 例如:42 = "运动户外-上衣"
- 可视化码本含义

**序列可解释**:
- SID [42, 17, 8, 3] → "运动户外-上衣 / 跑步 / 短袖 / 红色"
- 用户能理解推荐逻辑

**调试友好**:
- 出错时容易定位
- 不像神经网络黑盒

### 3.20 Query-Item 对齐的损失函数设计

**1. 对比损失**:
- InfoNCE
- 适合负样本多的场景

**2. 三元组损失**:
$$
\mathcal{L}_{\text{triplet}} = \max(0, d(q, i^+) - d(q, i^-) + \alpha)
$$

**3. Circle Loss**:
- 更精细的相似度学习
- 适合细粒度检索

**4. PolyLoss**:
- 多项式展开
- 灵活

### 3.21 实战代码

**完整训练流程**:
```python
class QueryItemAlignmentModel(nn.Module):
    def __init__(self, query_encoder, item_encoder, temperature=0.05):
        super().__init__()
        self.query_encoder = query_encoder
        self.item_encoder = item_encoder
        self.temperature = temperature

    def forward(self, queries, items):
        q_emb = self.query_encoder(queries)  # [B, d]
        i_emb = self.item_encoder(items)  # [B, d]
        q_emb = F.normalize(q_emb, dim=-1)
        i_emb = F.normalize(i_emb, dim=-1)
        # 对比损失
        sim = q_emb @ i_emb.T / self.temperature
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        loss = F.cross_entropy(sim, labels)
        return loss
```

**训练**:
```python
def train_alignment(model, train_loader, optimizer, n_epochs=10):
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            queries, items = batch
            loss = model(queries, items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

**评估**:
```python
def evaluate_alignment(model, test_data, item_pool, k=50):
    model.eval()
    with torch.no_grad():
        # 编码所有商品
        all_item_embs = model.item_encoder(item_pool)
        # 评估每个 query
        recalls = []
        for query, pos_items in test_data.items():
            q_emb = model.query_encoder([query])
            scores = q_emb @ all_item_embs.T
            top_k = scores.topk(k).indices
            hits = len(set(top_k.tolist()) & set(pos_items))
            recalls.append(hits / len(pos_items))
    return np.mean(recalls)
```

### 3.22 工业级 Query-Item 系统

**架构**:
```text
Query 输入
  ↓
Query 理解(分词、改写、意图)
  ↓
Query 编码(Encoder)
  ↓
SID 检索 / 向量检索
  ↓
粗排(双塔)
  ↓
精排(交叉编码器)
  ↓
Top-K 推荐
```

**性能指标**:
- P99 延迟:< 30ms
- QPS:> 10 万
- Recall@50:> 0.7
- CTR:> 5%

### 3.23 总结

Query-Item 对齐是搜索 + 推荐的核心。本章深入探讨了:

1. **Query 理解**:分词、改写、意图
2. **对齐损失**:InfoNCE、三元组、Circle
3. **双编码器 vs 交叉编码器**
4. **SID 在对齐中的作用**
5. **检索方式**:向量、Trie、汉明
6. **多语言与长尾**
7. **实战代码**

[3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)---

## 4. 高级话题

### 4.1 Late Interaction 模型

**问题**:双编码器独立编码,丢失交互信号。交叉编码器太慢。

**Late Interaction**(如 ColBERT):
- Query 和 item 各自编码所有 token
- 检索时计算 token 级交互
- 速度:中等(比交叉快很多)
- 精度:接近交叉

**ColBERT 架构**:
1. Query:每个 token 编码为向量 $[\mathbf{q}_1, ..., \mathbf{q}_L]$
2. Item:每个 token 编码为向量 $[\mathbf{i}_1, ..., \mathbf{i}_M]$
3. 相似度:$\sum_j \max_i \mathbf{q}_j^\top \mathbf{i}_i$

### 4.2 SPLADE 稀疏检索

**核心**:稀疏向量,可逆索引,速度快。

**SPLADE 编码**:
- 输入文本
- 输出:稀疏权重向量
- 维度 = 词表大小,只有少量非零

**优势**:
- 速度极快(倒排索引)
- 可解释(权重 = 重要性)
- 适合工业部署

### 4.3 跨模态 Query-Item

**场景**:用户用图像搜商品(拍立淘)。

**方法**:
- 图像 → CNN/ViT embedding
- 与商品 embedding 对齐
- 联合训练

**淘宝拍立淘**:
- 图像检索
- 同款 + 相似款
- CTR 提升 20%

### 4.4 个性化 Query-Item

**场景**:不同用户搜 "笔记本",推荐不同。

**用户上下文**:
- 历史行为
- 人口学属性
- 实时上下文(时间、地点)

**个性化编码**:
- Query + User → 个性化 embedding
- 检索时考虑用户

### 4.5 多任务 Query-Item

**多任务**:
- 召回相关性
- 排序精度
- 多样性
- 新颖性

**多任务训练**:
- 共享底层
- 任务专属头
- GradNorm 自动平衡

### 4.6 因果 Query-Item

**问题**:协同信号捕捉的是相关,不是因果。

**反事实**:
- 用户搜 A,买了 B
- 但 B 是因为 A 的原因吗?

**因果建模**:
- 倾向得分加权(IPW)
- Doubly Robust
- 个性化因果效应

### 4.7 可解释 Query-Item

**用户想了解**:
- 为什么推荐这个商品?
- 它和我的 query 有什么关系?

**可解释方法**:
- 注意力可视化
- 关键短语提取
- 类目映射

### 4.8 实时 Query-Item

**场景**:用户搜索结果需要实时。

**实时流程**:
1. Query 理解(< 5ms)
2. 向量检索(< 20ms)
3. 粗排(< 5ms)
4. 精排(< 10ms)
5. 总延迟 < 50ms

### 4.9 联邦 Query-Item

**场景**:多平台合作,但数据不出域。

**联邦学习**:
- 各平台本地训练
- 共享 embedding
- 中心聚合

**挑战**:
- 通信开销
- 数据异构
- 隐私保护

### 4.10 大模型驱动的 Query-Item

**LLM 增强**:
- Query 改写
- 意图识别
- 商品理解
- 相关性判断

**应用**:
- LLM rerank
- LLM 解释
- LLM 生成 query 扩展

### 4.11 多轮对话式 Query-Item

**场景**:用户多轮对话搜索。

**多轮对话管理**:
- 历史对话跟踪
- 上下文更新
- 代词解析

**代表**:
- 阿里小蜜
- 京东 JIMI
- 微软小冰

### 4.12 跨语言 Query-Item

**场景**:用户用母语搜其他语言商品。

**多语言对齐**:
- mBERT / XLM-R
- 翻译对训练数据
- 跨语言检索

**应用**:
- 跨境电商(AliExpress)
- 国际搜索

### 4.13 Query 理解的高级技术

**1. Query 分类**:
- 导航 / 信息 / 交易
- 不同类别用不同策略

**2. Query 改写**:
- 同义词、纠错、扩展
- LLM 改写

**3. Query 解析**:
- 命名实体识别(NER)
- 属性抽取
- 意图分类

**4. Query 推荐**:
- 给定 query,推荐相似 query
- 提升搜索体验

### 4.14 商品理解的高级技术

**1. 商品属性抽取**:
- 品牌、型号、颜色
- 结构化输出

**2. 商品摘要**:
- LLM 生成简短描述
- 用于展示

**3. 商品分类**:
- 自动分类
- 多级类目

**4. 商品质量评估**:
- 假货检测
- 质量评分

### 4.15 相关性建模

**相关性级别**:

| 级别 | 描述 |
| :--- | :--- |
| 0 | 不相关 |
| 1 | 弱相关 |
| 2 | 中等相关 |
| 3 | 强相关 |
| 4 | 完全相关 |

**学习**:
- 用 5 级评分数据
- Cross-entropy loss
- 或者 ranking loss

### 4.16 召回阶段详解

**多路召回**:
- 文本匹配(BM25)
- 向量检索(SID / Embedding)
- 协同过滤
- 知识图谱
- 热门商品

**合并去重**:
- 多路结果合并
- 去重
- 分数归一化

**工业实践**:
- 通常 5-10 路召回
- 每路 1000 个候选
- 合并后 5000+
- 后续粗排

### 4.17 排序阶段详解

**粗排**:
- 轻量级模型
- 快速过滤
- 1000 → 100

**精排**:
- 复杂模型(DNN)
- 精细打分
- 100 → 10

**重排**:
- 业务规则
- 多样性
- 新颖性
- 10 → 最终

### 4.18 个性化与多样性平衡

**个性化**:用户喜欢什么推什么
**多样性**:推荐要多样,避免信息茧房

**平衡**:
$$
\text{Score} = \alpha \cdot \text{Relevance} + (1-\alpha) \cdot \text{Diversity}
$$

**MMR(Maximal Marginal Relevance)**:
$$
\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1-\lambda) \cdot \max_{d' \in S} \text{Sim}(d, d')
$$

### 4.19 离线评估 vs 在线效果

**离线指标好,在线不一定好**:
- 数据分布偏移
- 用户行为改变
- 系统延迟

**A/B 测试是真理**:
- 离线 → 在线漏斗
- 线上效果 = 真理

### 4.20 推荐系统的可解释性

**用户**:
- "为什么推荐这个?"
- "基于你的搜索历史"

**运营**:
- "这个商品为什么被推荐?"
- "查询相似度 0.8"

**技术**:
- 注意力可视化
- 类目追溯
- 相似度分解

### 4.21 推荐系统的偏差与公平

**常见偏差**:
1. **位置偏差**:用户倾向点靠前的
2. **曝光偏差**:没曝光无数据
3. **流行度偏差**:热门商品更多被推荐
4. **选择偏差**:用户行为非随机

**解决**:
1. **逆倾向加权(IPW)**
2. **位置特征**
3. **流行度正则化**
4. **数据重采样**

### 4.22 推荐系统的鲁棒性

**攻击**:
- 假流量
- 假点击
- 假评论
- 竞争商品打压

**防御**:
- 异常检测
- 行为验证
- 多信号融合
- 联邦学习

### 4.23 推荐系统的演进

**1.0 时代**:协同过滤
**2.0 时代**:深度学习(DNN, DIN)
**3.0 时代**:生成式推荐(TIGER, HSTU)
**4.0 时代**:大模型 + Agent

### 4.24 推荐系统的未来

**短期(1-3 年)**:
- 大模型推荐
- 多模态
- 因果

**中期(3-5 年)**:
- AGI 辅助推荐
- 端到端生成

**长期(5-10 年)**:
- 通用推荐
- 人机协作

---

## 5. 思考题

1. Query 和 Item 的语义鸿沟如何弥合?
2. 双编码器 vs 交叉编码器的权衡?
3. SID 检索 vs 向量检索的选择?
4. 多语言对齐的实现难点?
5. 个性化和多样性如何平衡?

---

## 6. 进阶阅读

### 论文

1. **DSSM** (2013)
2. **ColBERT** (2020)
3. **SPLADE** (2020)
4. **TIGER** (2023)
5. **HSTU** (2024)

### 工业实践

- Google 搜索
- 淘宝拍立淘
- 京东搜索
- 阿里妈妈

---

## 7. 总结

Query-Item 对齐是搜索 + 推荐的核心。本章涵盖了:

1. **理论基础**:对比学习、嵌入空间
2. **Query 理解**:分词、改写、意图
3. **对齐损失**:InfoNCE、三元组
4. **架构**:双编码器、交叉编码器、Late Interaction
5. **SID 应用**:Trie、汉明、序列匹配
6. **多语言、长尾、个性化**
7. **大模型驱动**
8. **工业实践**

[3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)---

## 8. 实战代码合集

### 8.1 ColBERT 实现

```python
class ColBERT(nn.Module):
    """ColBERT: Late Interaction 模型"""
    def __init__(self, vocab_size, dim=128, max_len=512):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.dim = dim
        self.linear = nn.Linear(self.encoder.config.hidden_size, dim)

    def encode(self, input_ids, attention_mask):
        """编码为所有 token 的 embedding"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state  # [B, L, hidden]
        return self.linear(token_embs)  # [B, L, dim]

    def forward(self, q_ids, q_mask, d_ids, d_mask):
        q_emb = self.encode(q_ids, q_mask)  # [B_q, L_q, dim]
        d_emb = self.encode(d_ids, d_mask)  # [B_d, L_d, dim]
        # Late interaction: sum of max similarities
        # q_emb: [B_q, L_q, dim], d_emb: [B_d, L_d, dim]
        # sim: [B_q, B_d, L_q, L_d]
        sim = (q_emb.unsqueeze(1) * d_emb.unsqueeze(0)).sum(dim=-1)
        # Mask
        d_mask_f = d_mask.float().unsqueeze(0).unsqueeze(2)  # [1, B_d, 1, L_d]
        sim = sim.masked_fill(d_mask_f == 0, -1e9)
        # Max over d tokens
        max_sim = sim.max(dim=-1).values  # [B_q, B_d, L_q]
        # Mask q
        q_mask_f = q_mask.float().unsqueeze(1).unsqueeze(-1)  # [B_q, 1, 1, L_q]?
        # Wait, max_sim is [B_q, B_d, L_q], so q_mask is [B_q, L_q] -> [B_q, 1, L_q]
        q_mask_f = q_mask.float().unsqueeze(1)  # [B_q, 1, L_q]
        max_sim = max_sim.masked_fill(q_mask_f == 0, 0)
        # Sum over q tokens
        score = max_sim.sum(dim=-1)  # [B_q, B_d]
        return score
```

### 8.2 SPLADE 实现

```python
class SPLADE(nn.Module):
    """SPLADE: 稀疏检索"""
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(self.encoder.config.hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 投影到词表大小
        logits = self.proj(outputs.last_hidden_state)  # [B, L, V]
        # ReLU + log saturation
        logits = F.relu(logits)
        # Mask padding
        logits = logits * attention_mask.unsqueeze(-1)
        # Max over tokens
        sparse = torch.log(1 + logits).max(dim=1).values  # [B, V]
        return sparse
```

### 8.3 多模态 Query-Item

```python
class MultimodalQueryItemAlignment(nn.Module):
    """多模态 Query-Item 对齐"""
    def __init__(self, text_dim=768, img_dim=1024, emb_dim=256):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('bert-base-chinese')
        self.img_encoder = timm.create_model('vit_base_patch16_224', num_classes=0)
        self.proj_text = nn.Linear(text_dim, emb_dim)
        self.proj_img = nn.Linear(img_dim, emb_dim)
        self.fuse = nn.Linear(emb_dim * 2, emb_dim)

    def encode_query(self, text):
        """编码文本 query"""
        out = self.text_encoder(**text)
        return self.proj_text(out.last_hidden_state[:, 0, :])

    def encode_item(self, image, text):
        """编码多模态商品"""
        img_feat = self.img_encoder(image)
        img_emb = self.proj_img(img_feat)
        text_feat = self.text_encoder(**text).last_hidden_state[:, 0, :]
        text_emb = self.proj_text(text_feat)
        # 融合
        fused = self.fuse(torch.cat([img_emb, text_emb], dim=-1))
        return fused

    def forward(self, query_text, item_image, item_text):
        q = self.encode_query(query_text)
        i = self.encode_item(item_image, item_text)
        q = F.normalize(q, dim=-1)
        i = F.normalize(i, dim=-1)
        sim = q @ i.T
        labels = torch.arange(q.size(0), device=q.device)
        loss = F.cross_entropy(sim, labels)
        return loss
```

### 8.4 SID 检索索引

```python
class SIDTrieIndex:
    """基于 Trie 树的 SID 索引"""
    def __init__(self):
        self.trie = {}  # {prefix: {sid: item_id}}

    def add(self, sid, item_id):
        """添加商品"""
        # 多级前缀索引
        for level in range(1, len(sid) + 1):
            prefix = tuple(sid[:level])
            if prefix not in self.trie:
                self.trie[prefix] = {}
            self.trie[prefix][tuple(sid)] = item_id

    def search_prefix(self, query_sid, prefix_len=2):
        """前缀检索"""
        prefix = tuple(query_sid[:prefix_len])
        if prefix in self.trie:
            return list(self.trie[prefix].values())
        return []

    def search_hamming(self, query_sid, max_distance=2):
        """汉明距离检索"""
        results = []
        for sid_tuple, item_id in self.all_sids.items():
            dist = sum(a != b for a, b in zip(query_sid, sid_tuple))
            if dist <= max_distance:
                results.append((item_id, dist))
        results.sort(key=lambda x: x[1])
        return results
```

### 8.5 端到端生成式检索

```python
class GenerativeQueryItemModel(nn.Module):
    """生成式 Query-Item 检索"""
    def __init__(self, query_encoder, sid_decoder, codebook_size=256, max_sid_len=4):
        super().__init__()
        self.query_encoder = query_encoder
        self.sid_decoder = sid_decoder  # GPT-like
        self.codebook_size = codebook_size
        self.max_sid_len = max_sid_len

    def forward(self, query_ids, target_sids):
        """训练"""
        # 编码 query
        q_emb = self.query_encoder(query_ids)  # [B, d]
        # 解码为 SID
        logits = self.sid_decoder(q_emb)  # [B, L, V]
        # 计算损失
        loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size),
            target_sids.reshape(-1)
        )
        return loss

    @torch.no_grad()
    def generate(self, query_ids, temperature=1.0):
        """生成 SID"""
        q_emb = self.query_encoder(query_ids)
        sids = []
        current = torch.zeros(q_emb.size(0), 1, dtype=torch.long, device=q_emb.device)
        for _ in range(self.max_sid_len):
            logits = self.sid_decoder(q_emb, current)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            sids.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(sids, dim=1)
```

---

## 9. 工业级系统设计

### 9.1 完整架构

```text
┌────────────────────────────────────────┐
│         Query-Item 检索系统              │
├────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  │
│  │ Query 理解    │  │ 商品理解      │  │
│  │ - 分词       │  │ - 文本编码    │  │
│  │ - 改写       │  │ - 图像编码    │  │
│  │ - 意图识别    │  │ - 属性抽取    │  │
│  └──────┬───────┘  └──────┬───────┘  │
│         ↓                  ↓          │
│  ┌──────────────────────────────────┐ │
│  │     召回层(多路)                 │ │
│  │  - 向量检索(FAISS)              │ │
│  │  - SID 检索(Trie)              │ │
│  │  - 文本匹配(BM25)              │ │
│  │  - 协同过滤                   │ │
│  └──────────────┬───────────────────┘ │
│                 ↓                     │
│  ┌──────────────────────────────────┐ │
│  │     粗排层                       │ │
│  │  - 双塔 + 轻量交互              │ │
│  │  - 1000 → 100                  │ │
│  └──────────────┬───────────────────┘ │
│                 ↓                     │
│  ┌──────────────────────────────────┐ │
│  │     精排层                       │ │
│  │  - 交叉编码器                   │ │
│  │  - 多任务                       │ │
│  │  - 100 → 10                    │ │
│  └──────────────┬───────────────────┘ │
│                 ↓                     │
│  ┌──────────────────────────────────┐ │
│  │     重排层                       │ │
│  │  - 多样性                       │ │
│  │  - 新颖性                       │ │
│  │  - 业务规则                     │ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘
```

### 9.2 性能要求

| 阶段 | 延迟 | QPS |
| :--- | :--- | :--- |
| Query 理解 | < 5ms | 10 万 |
| 召回 | < 20ms | 10 万 |
| 粗排 | < 5ms | 10 万 |
| 精排 | < 10ms | 10 万 |
| 重排 | < 5ms | 10 万 |
| **总延迟** | < 50ms | 10 万 |

### 9.3 监控指标

**离线**:
- Recall@K
- NDCG@K
- MRR
- 多样性
- 覆盖率

**在线**:
- 业务:CTR、CVR、GMV
- 系统:QPS、延迟、错误率
- 用户:留存、时长

### 9.4 A/B 测试

**实验设计**:
- 流量分桶
- 实验组 vs 对照组
- 至少 7 天
- 显著性检验

**核心指标**:
- 主指标:CTR / GMV
- 辅助:留存、时长、多样性

### 9.5 部署 Checklist

- [ ] 模型训练完成
- [ ] 离线评估通过
- [ ] 索引建立
- [ ] 服务压测
- [ ] 灰度发布
- [ ] 监控配置
- [ ] 应急回滚

---

## 10. 总结

Query-Item 对齐是搜索 + 推荐的核心。从信息检索到深度学习,从双编码器到生成式检索,技术持续演进。本章系统介绍了:

1. **Query 理解**:分词、改写、意图
2. **对齐损失**:InfoNCE 等
3. **架构**:双编码器、交叉编码器、Late Interaction
4. **SID 应用**:Trie、汉明、序列匹配
5. **多语言、长尾、个性化**
6. **大模型驱动**
7. **工业级系统设计**

**核心洞察**:
- **Query 是用户意图的载体**
- **Item 是商品语义的载体**
- **对齐是匹配的核心**
- **SID 让对齐有了结构**
- **大模型让对齐更智能**

下一步进入 [3-冷启动数学解释.md](第05章-03-冷启动数学解释.md),深入理解冷启动的数学本质。

[3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)---

## 11. 高级训练技巧

### 11.1 课程学习

**思想**:从易到难。

**Query-Item 对齐的课程**:
1. **阶段 1**:完全匹配(完全相同的 query 和 item)
2. **阶段 2**:部分匹配(类目相同,具体不同)
3. **阶段 3**:语义匹配(类目不同但语义相关)
4. **阶段 4**:难负样本(语义相似但不匹配)

### 11.2 数据增强

**Query 增强**:
- 同义词替换
- 回译
- LLM 改写

**Item 增强**:
- 标题改写
- 图像增强
- 属性扰动

```python
class DataAugmentation:
    def augment_query(self, query):
        # 1. 同义词替换
        for word, syn in synonym_dict.items():
            query = query.replace(word, syn)
        # 2. 回译
        if random.random() < 0.3:
            query = self.back_translate(query)
        return query

    def augment_item(self, item):
        # 1. 标题改写
        if random.random() < 0.3:
            item['title'] = self.rewrite_title(item['title'])
        return item
```

### 11.3 对抗训练

**FGSM**:
- 加梯度方向扰动
- 提升鲁棒性

**代码**:
```python
def fgsm_attack(embeddings, epsilon=0.01):
    """FGSM 对抗样本"""
    embeddings.requires_grad_(True)
    loss = compute_loss(embeddings)
    loss.backward()
    perturbation = epsilon * embeddings.grad.sign()
    return embeddings + perturbation
```

### 11.4 知识蒸馏

**教师模型**:大模型 / 集成
**学生模型**:小模型

**蒸馏损失**:
$$
\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{task}} + (1-\alpha) \text{KL}(\text{teacher} \| \text{student})
$$

### 11.5 多任务学习

**任务列表**:
- T1:Query-Item 匹配
- T2:Query 改写
- T3:Item 分类
- T4:Query 意图

**多任务架构**:
```python
class MultiTaskAlignment(nn.Module):
    def __init__(self, n_tasks=4):
        super().__init__()
        self.shared = SharedEncoder()
        self.task_heads = nn.ModuleList([
            TaskHead() for _ in range(n_tasks)
        ])

    def forward(self, x, task_id):
        shared = self.shared(x)
        return self.task_heads[task_id](shared)
```

### 11.6 持续学习

**场景**:新类目、新商品不断出现。

**方法**:
- Elastic Weight Consolidation (EWC)
- Progressive Networks
- Memory Replay

### 11.7 元学习

**MAML**:
- 学习快速适应新任务
- Few-shot 场景

**应用**:
- 长尾类目
- 新平台
- 冷启动场景

### 11.8 自动化机器学习

**AutoML 工具**:
- Optuna:超参搜索
- Ray Tune:分布式
- Determined:实验管理

**应用**:
- 自动特征工程
- 自动模型选择
- 自动调参

---

## 12. 评估方法详解

### 12.1 离线评估

**Recall@K**:
$$
\text{Recall@K} = \frac{\text{检索正确的相关 item 数}}{\text{所有相关 item 数}}
$$

**NDCG@K**:
$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$
$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

**MRR**:
$$
\text{MRR} = \frac{1}{|Q|}\sum_{q} \frac{1}{\text{rank}_q}
$$

**MAP**:
$$
\text{MAP} = \frac{1}{|Q|}\sum_{q} \text{AP}_q
$$

### 12.2 在线评估

**A/B 测试**:
- 主指标:CTR、CVR
- 辅助指标:时长、留存
- 统计显著性

**Interleaving**:
- 快速比较两个模型
- 同一页面混合两个模型结果
- 观察哪个被更多点击

### 12.3 离线 vs 在线

**离线 ≠ 在线**:
- 数据分布偏移
- 用户行为差异
- 系统延迟

**策略**:
- 离线 + A/B 双重验证
- 长期指标关注

---

## 13. 实战项目

### 13.1 项目 1:电商搜索

**数据**:Amazon Product Search
**任务**:Query → 相关商品
**指标**:NDCG@10

### 13.2 项目 2:拍立淘

**数据**:淘宝图像 + 商品
**任务**:图像 → 同款/相似款
**指标**:Recall@50

### 13.3 项目 3:多语言搜索

**数据**:AliExpress
**任务**:跨语言检索
**指标**:MRR

---

## 14. 面试高频问题

### Q1: Query-Item 对齐的核心难点?

**答**:
1. 语义鸿沟:Query 短,Item 长
2. 表达差异:用户口语化,商品专业化
3. 意图模糊:Query 信息有限
4. 实时性:新商品、新 Query 不断出现

### Q2: 双编码器 vs 交叉编码器?

**答**:
- 双编码器:独立编码,快但弱交互
- 交叉编码器:联合编码,慢但强交互
- 工业:召回用双,排序用交叉

### Q3: SID 检索 vs 向量检索?

**答**:
- SID 检索:快、索引小、冷启动友好、可解释
- 向量检索:精度高、灵活、检索质量好
- 选择:召回用 SID 兜底,精排用向量

### Q4: 如何处理长尾 Query?

**答**:
1. Query 改写
2. 类目预测
3. 数据增强
4. 元学习
5. 联邦学习

### Q5: 多语言检索怎么做?

**答**:
1. 多语言 encoder(mBERT / XLM-R)
2. 翻译对训练数据
3. 跨语言对比学习
4. 统一向量空间

### Q6: 如何评估检索质量?

**答**:
- 离线:Recall@K, NDCG@K, MRR
- 在线:CTR, CVR, GMV
- A/B 测试是真理

### Q7: 如何提升相关性?

**答**:
1. 难负样本挖掘
2. 多任务学习
3. 大模型 rerank
4. 多路召回融合
5. 个性化

### Q8: 召回稀疏怎么办?

**答**:
1. 知识图谱补全
2. LLM 生成
3. 联邦学习
4. 多语言共享

---

## 15. 推荐学习路径

### Week 1-2:基础

- DSSM 论文
- 双编码器实现
- MovieLens 练习

### Week 3-4:进阶

- ColBERT 论文
- Late interaction
- 工业级搜索实现

### Week 5-6:高级

- SPLADE
- 生成式检索(TIGER)
- 大模型驱动

### Week 7-8:实战

- 端到端实现
- A/B 测试
- 部署上线

---

## 16. 推荐资源

### 16.1 论文

1. DSSM (2013)
2. ColBERT (2020)
3. SPLADE (2020)
4. SPLADE v2 (2021)
5. ColBERT v2 (2022)
6. TIGER (2023)
7. HSTU (2024)

### 16.2 工业实践

- Google Search
- Amazon Search
- 阿里搜索
- 淘宝拍立淘

### 16.3 学习平台

- Google Research
- HuggingFace
- Sentence-Transformers
- PyTorch Tutorials

### 16.4 比赛

- TREC
- MS MARCO
- BEIR

---

## 17. 结语

Query-Item 对齐是搜索推荐的核心。本章深入探讨了:

1. **理论基础**:对齐损失、嵌入空间
2. **Query 理解**:分词、改写、意图
3. **架构**:双编码器、交叉编码器、Late Interaction
4. **SID 应用**:Trie、汉明、序列匹配
5. **多语言、长尾、个性化**
6. **大模型驱动**
7. **工业实践**

下一步进入 [3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)。

[3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)

---

## 5. 对比学习对齐

### 5.1 为什么需要对比学习

Query 与 Item 的配对数据通常来自点击、加购、收藏和购买行为。
这些行为不是绝对标签，而是带有曝光、位置、价格和用户偏好的观测结果。
对比学习不要求给每个商品建立完整的分类标签，而是直接学习相对关系。
核心问题可以写成：给定 query，正样本的相似度应高于负样本。

设一个 batch 中有 $B$ 对 query-item：

$$
Q = [q_1, q_2, \ldots, q_B], \quad I = [i_1, i_2, \ldots, i_B]
$$

第 $b$ 个 query 与第 $b$ 个 item 是正样本，其余 item 默认作为负样本。
这种 in-batch negative 方式不需要额外维护巨大的负样本表。

### 5.2 温度系数

InfoNCE 中的温度 $au$ 决定 softmax 分布的尖锐程度。

$$
P(i_j|q_b)=\frac{\exp(s(q_b,i_j)/\tau)}{\sum_{k=1}^{B}\exp(s(q_b,i_k)/\tau)}
$$

当 $au$ 较小时，模型更关注最高分样本，梯度更集中。
当 $au$ 较大时，分布更平滑，训练更稳定但区分能力可能不足。
工程上可以固定温度，也可以将温度设为可学习参数。
可学习温度必须限制范围，避免训练后变成过小或过大的异常值。

```python
class LearnableTemperature(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature))
        )

    def forward(self):
        # 通过 clamp 防止数值极端导致 softmax 溢出
        return self.log_temperature.exp().clamp(0.01, 1.0)
```

### 5.3 双向对比损失

只计算 query 到 item 的方向，可能导致 item encoder 受到的约束不足。
更常见的做法是同时计算 query-to-item 与 item-to-query 两个方向。

$$
\mathcal{L}_{q\rightarrow i} = -\frac{1}{B}\sum_b\log P(i_b|q_b)
$$

$$
\mathcal{L}_{i\rightarrow q} = -\frac{1}{B}\sum_b\log P(q_b|i_b)
$$

$$
\mathcal{L}_{bi}=\frac{1}{2}(\mathcal{L}_{q\rightarrow i}+\mathcal{L}_{i\rightarrow q})
$$

双向损失适合 query 与 item 都需要产生高质量 embedding 的场景。
如果 item 侧是固定的商品语义空间，可以只训练 query-to-item 方向。

```python
def symmetric_nce(q, i, temperature=0.07):
    q = F.normalize(q, dim=-1)
    i = F.normalize(i, dim=-1)
    logits = q @ i.T / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss_qi = F.cross_entropy(logits, labels)
    loss_iq = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_qi + loss_iq)
```

### 5.4 多正样本学习

一个 query 往往对应多个有效商品，不能简单地把 batch 中唯一的点击商品当成唯一答案。
例如“夏季白色连衣裙”可能有数百个商品都满足意图。
将其他相关商品误当成负样本会造成 false negative，损害 embedding 空间。

可以使用多正样本 mask：

```python
def multi_positive_loss(logits, positive_mask):
    """logits: [B, N], positive_mask: [B, N]"""
    log_prob = F.log_softmax(logits, dim=-1)
    positive_count = positive_mask.sum(dim=-1).clamp_min(1)
    loss = -(log_prob * positive_mask).sum(dim=-1) / positive_count
    return loss.mean()
```

正样本 mask 可以来自以下来源：

1. 同一 query 在时间窗口内点击过的多个商品。
2. 相同类目、属性和品牌的人工标注商品。
3. 购买或加购商品及其高置信相似商品。
4. 教师模型给出的软相关性分数。

### 5.5 负样本分层

随机负样本容易区分，训练后期提供的梯度很少。
但是直接使用最相似的负样本，又可能把真正相关的商品当作负样本。
因此需要分层构造负样本。

| 负样本类型 | 来源 | 训练作用 |
| :--- | :--- | :--- |
| 随机负样本 | 全商品池随机采样 | 建立基本边界 |
| 曝光未点 | 同一次请求未点击商品 | 接近真实竞争 |
| 类目负样本 | 同类目不同属性 | 学习细粒度差异 |
| BM25 负样本 | 文本高度相似商品 | 学习语义与词面差异 |
| ANN 负样本 | embedding 最近邻 | 提升局部判别能力 |
| 业务负样本 | 违规、缺货、过期商品 | 学习有效性约束 |

负样本比例不应只看数量，还要看难度分布。
一个实用策略是让每个 batch 同时包含随机、曝光和 hard negative。

### 5.6 False Negative 处理

False negative 是检索训练中的常见噪声来源。
一个商品没有被点击，不代表它一定不相关，可能是位置太靠后或用户没有继续浏览。
可以通过时间窗口、用户行为强度和多源日志来降低误判。

常见策略包括：

- 对高相似度负样本降低损失权重。
- 将同类目高质量商品标记为潜在正样本。
- 使用 debiased contrastive loss 修正采样偏差。
- 只将明确被跳过、投诉或无效的商品作为强负样本。
- 对同一 query 的多次曝光进行聚合，而不是逐条处理。

### 5.7 对齐与均匀性

只有 alignment 没有 uniformity，所有 embedding 可能坍缩到空间中的一个小区域。
只有 uniformity 没有 alignment，向量会均匀分散但失去语义关系。
因此需要同时关注正样本距离和整体空间分布。

$$
\mathcal{L}_{align}=\mathbb{E}_{(q,i^+)}\|q-i^+\|^2
$$

$$
\mathcal{L}_{uniform}=\log\mathbb{E}_{x,y}\exp(-t\|x-y\|^2)
$$

$$
\mathcal{L}=\mathcal{L}_{align}+\lambda\mathcal{L}_{uniform}
$$

诊断时可以画出 embedding norm、pairwise cosine 和各维方差。
如果各维方差接近零，通常意味着表示坍缩或输入管道出了问题。

### 5.8 SID 空间中的对比学习

SID 不是普通的离散商品 ID，而是由多个码本位置组成的结构化序列。
对比学习可以同时约束完整 SID 和局部 SID。
完整 SID 用于区分具体商品，前缀 SID 用于表达类目或粗粒度语义。

```python
def sid_hierarchical_loss(query_sid_logits, target_sid, weights=None):
    """对每个 SID 位置计算层级交叉熵。"""
    if weights is None:
        weights = [1.0] * target_sid.size(1)
    losses = []
    for level, weight in enumerate(weights):
        loss = F.cross_entropy(
            query_sid_logits[:, level, :],
            target_sid[:, level]
        )
        losses.append(weight * loss)
    return sum(losses) / sum(weights)
```

早期码本位置通常表示更粗粒度语义，权重可以更高。
后期位置表达细节，权重可以逐渐增加以支持精确匹配。

### 5.9 对比损失的稳定训练

训练前应检查 query、item 和 SID 的长度、掩码、dtype 是否一致。
输入 embedding 建议在计算相似度前做 L2 normalize。
logits 需要考虑 temperature 与数值范围，必要时使用 float32 计算损失。
混合精度训练时，梯度缩放不能掩盖 NaN，必须记录 loss 和梯度范数。

```python
def check_batch(q, i, labels):
    assert q.ndim == 2 and i.ndim == 2
    assert q.size(0) == i.size(0) == labels.size(0)
    assert q.size(1) == i.size(1)
    assert torch.isfinite(q).all()
    assert torch.isfinite(i).all()
```

### 5.10 训练数据配比

数据配比要覆盖头部、腰部和长尾 query。
只使用高频 query 会让模型看起来指标很高，但长尾泛化很差。
推荐按 query 频率分桶，再对每个桶设置采样上限。

一种简单的采样权重为：

$$
w(q)=\frac{1}{(freq(q)+c)^\alpha}
$$

其中 $\alpha$ 控制长尾增强强度，$c$ 防止低频样本权重爆炸。
采样后仍应保留原始曝光分布，用于离线评估的加权校准。

### 5.11 训练失败排查

**现象一：loss 很快降到接近零。**
可能是负样本过于简单、数据泄漏或 batch 内重复配对。
应检查 query 与 item 是否通过 ID 或排序顺序意外泄漏。

**现象二：Recall 很低但训练 loss 正常。**
可能是训练商品池和评估商品池分布不同，或者 SID 版本不一致。
也可能是 top-K 检索索引没有更新到最新 embedding。

**现象三：线上 CTR 下降。**
可能是点击标签受到位置偏差，或者模型过度优化点击而损害成交质量。
需要增加购买、加购、停留和投诉等长期目标。

### 5.12 小结

对比学习对齐的关键不是简单套用 InfoNCE，而是正确处理数据和空间。
应重点关注正负样本定义、temperature、false negative、分布均衡和 SID 层级结构。
当训练目标与线上检索候选池一致时，对比学习才能稳定转化为召回收益。

---

## 6. 双塔模型

### 6.1 双塔的基本结构

双塔模型把 query 与 item 分成两个独立编码器。
Query tower 只看查询和可用的用户上下文。
Item tower 只看商品内容、属性、库存和可离线计算的信息。
两个 tower 输出同维度向量，再通过点积或余弦相似度打分。

```python
class DualTower(nn.Module):
    def __init__(self, query_encoder, item_encoder, dim):
        super().__init__()
        self.query_encoder = query_encoder
        self.item_encoder = item_encoder
        self.query_proj = nn.Linear(dim, dim)
        self.item_proj = nn.Linear(dim, dim)

    def encode_query(self, x):
        return F.normalize(self.query_proj(self.query_encoder(x)), dim=-1)

    def encode_item(self, x):
        return F.normalize(self.item_proj(self.item_encoder(x)), dim=-1)

    def score(self, query, item):
        return (self.encode_query(query) * self.encode_item(item)).sum(-1)
```

### 6.2 为什么双塔适合召回

商品 embedding 可以在离线批处理阶段计算并写入索引。
在线请求只需要编码一次 query，再与索引中的向量做近邻搜索。
编码和检索过程可以横向扩展，不需要对每个 query-item 对重新运行大模型。
因此双塔能够支撑百万到十亿级商品池。

### 6.3 输入特征设计

Query 侧特征包括：

- 原始 query token。
- 纠错和改写后的 query。
- query 长度、语言、设备和入口。
- 用户最近搜索词和短期兴趣。
- 用户地域、时间和场景上下文。

Item 侧特征包括：

- 标题、描述和卖点文本。
- 类目、品牌、价格区间和属性。
- 图片、视频和结构化质量特征。
- 商品生命周期、库存和配送能力。
- 商品的 SID 序列与码本统计。

不能把在线时才知道的未来信息放入 item tower。
例如点击后的行为统计只能在正确的时间切片下构造，不能使用未来窗口数据。

### 6.4 共享编码器与独立编码器

共享底层 Transformer 可以节省参数，并增强 query-item 的词汇对齐。
完全独立的编码器则可以针对两侧的数据特点分别优化。
工业上常见做法是共享 tokenizer 和部分底层层，顶部使用独立 projection head。
如果 query 和 item 的语言差异很大，强行共享全部参数可能限制表达能力。

### 6.5 交互函数

最简单的交互函数是点积：

$$
score(q,i)=q^Ti
$$

余弦相似度会消除向量长度影响：

$$
score(q,i)=\frac{q^Ti}{\|q\|\|i\|}
$$

带偏置的点积可以表达 query 或 item 的先验吸引力：

$$
score(q,i)=q^Ti+b_q+b_i
$$

如果需要轻量级非线性交互，可以拼接 $q$、$i$、$|q-i|$ 和 $q\odot i$。
但交互越复杂，越难使用高效的 ANN 索引。

```python
def pair_features(q, i):
    return torch.cat([q, i, torch.abs(q - i), q * i], dim=-1)
```

### 6.6 训练目标选择

召回模型通常使用以下目标：

1. In-batch InfoNCE：适合大 batch 和大量隐式反馈。
2. Pairwise hinge loss：直接优化正负样本间隔。
3. BPR loss：适合用户偏好排序。
4. Pointwise BCE：适合有较可靠二分类标签的场景。
5. 多任务损失：同时预测点击、加购、购买和停留。

多任务目标可以写成：

$$
\mathcal{L}=\lambda_c\mathcal{L}_{click}+\lambda_a\mathcal{L}_{add}+\lambda_p\mathcal{L}_{purchase}
$$

权重不能只依据 loss 大小决定，应结合业务价值和指标敏感度调节。

### 6.7 用户塔与 Query 塔的区别

搜索场景中，query 是即时意图的主要载体。
推荐场景中，用户历史和上下文通常比一个 query 更重要。
如果把两者混为一个 tower，模型可能用长期兴趣覆盖掉即时需求。
更稳妥的方式是分别生成短期 query embedding 和长期 user embedding。

$$
q_{final}=q_{intent}+\beta u_{long}+\gamma c_{context}
$$

$\beta$ 和 $\gamma$ 可以通过门控网络动态生成。

### 6.8 门控融合

```python
class ContextGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, q, user, context):
        g = self.gate(torch.cat([q, user, context], dim=-1))
        return g * q + (1 - g) * user + context
```

门控值需要监控分布。
如果所有样本的 gate 都接近 0 或 1，说明融合网络可能退化成单一路径。

### 6.9 大 batch 训练

大 batch 能提供更多 in-batch negatives，但会提高显存和通信成本。
可以使用梯度累积、跨卡 all-gather 或 memory queue 扩大负样本池。
all-gather 时要注意不同卡上的梯度是否保留，避免无意中切断 item tower 梯度。

```python
def gather_without_grad(x):
    # 适合只扩大负样本池、不回传其他卡梯度的近似方案
    return torch.cat(all_gather([x.detach()]), dim=0)
```

在质量要求高的训练中，应比较 detach 和完整梯度版本的收益与通信代价。

### 6.10 ANN 索引接口

双塔模型训练完成后，item embedding 需要写入 ANN 索引。
索引构建必须记录模型版本、数据时间、向量维度和归一化方式。
线上 query 的预处理和离线 item 的预处理必须使用兼容版本。

```python
class EmbeddingManifest:
    def __init__(self, model_version, dim, normalized, snapshot):
        self.model_version = model_version
        self.dim = dim
        self.normalized = normalized
        self.snapshot = snapshot

    def validate(self, query_dim, query_normalized):
        assert query_dim == self.dim
        assert query_normalized == self.normalized
```

### 6.11 双塔的主要缺点

独立编码会损失细粒度的 token-to-token 交互。
例如 query 中的“防水”需要与 item 的“IPX8”建立属性级关系，单向量可能难以表达。
双塔还容易受到热门商品和文本长度的影响。
因此通常将双塔用于召回，再由粗排、精排和规则层补足交互。

### 6.12 双塔的改进路径

- 使用多向量表示代替单向量表示。
- 引入属性级辅助任务。
- 将 SID 前缀作为结构化监督。
- 使用 teacher cross-encoder 产生软标签。
- 对不同 query 类型使用不同 projection head。
- 训练时加入线上候选分布中的 hard negatives。

### 6.13 版本兼容与回滚

模型升级时，旧索引和新 query encoder 不能随意混用。
如果不能瞬间重建全量索引，需要支持双版本索引并行服务。
灰度阶段可以按用户或请求 hash 分流，比较新旧模型的召回和延迟。
发现异常时，回滚 query encoder 还不够，必须一起回滚索引版本和特征字典。

### 6.14 小结

双塔的价值是将昂贵的 item 计算前移到离线，将在线计算变成向量近邻搜索。
它的边界是交互能力有限，因此不能把所有排序问题都交给双塔解决。
在 SID 场景中，双塔适合学习 query 到结构化语义空间的入口。

---

## 7. 工业实践

### 7.1 生产系统的分层

一个可上线的 Query-Item 系统通常分为数据层、训练层、索引层、服务层和实验层。
数据层负责日志、商品快照、标签和特征生成。
训练层负责样本构造、模型训练、评估和模型注册。
索引层负责 item embedding、SID 倒排和多版本索引。
服务层负责 query 理解、召回、排序和降级。
实验层负责灰度、A/B、监控和回滚。

### 7.2 数据快照

离线训练必须绑定明确的数据快照时间。
商品内容、库存、价格和行为日志要按照同一时间轴拼接。
训练样本的 label window 不能与 feature window 重叠到未来。
每个样本至少应记录：

- request_id。
- user_id 的脱敏标识。
- query 原文和规范化结果。
- item_id 与 SID 版本。
- 曝光位置、点击和后续行为。
- 样本生成时间与数据分区。

这样才能在指标异常时复现训练数据。

### 7.3 标签构造

点击是最容易获得的信号，但不等价于真实相关性。
购买通常更稀疏，却更接近业务价值。
停留时间可以补充点击后的满意度，但容易受到页面加载影响。
投诉、退款和快速返回等行为可以作为负向信号。

```python
def behavior_weight(event):
    weights = {
        "impression": 0.0,
        "click": 1.0,
        "favorite": 2.0,
        "add_to_cart": 3.0,
        "purchase": 5.0,
        "refund": -3.0,
    }
    return weights.get(event, 0.0)
```

权重只是训练信号，不应直接被解释为概率。

### 7.4 位置偏差

排在前面的商品更容易被看到和点击。
如果直接把点击作为相关性标签，模型会学习位置而不是语义。
可以记录曝光位置、页面深度和可见性，再用倾向得分校正。

$$
\hat{y}=\frac{y}{P(\text{exposure}|position,context)+\epsilon}
$$

生产环境中不应把极端权重直接用于训练，需要裁剪或使用 self-normalized weighting。

### 7.5 多路召回

单一召回源无法覆盖所有 query 类型。
典型召回池包括：

1. SID 前缀召回：对新商品和结构化语义友好。
2. Dense ANN：覆盖同义表达和跨词面语义。
3. BM25：覆盖精确型号、品牌和数字串。
4. 协同过滤：覆盖个性化和隐式偏好。
5. 热门召回：在冷门 query 或系统故障时兜底。
6. 规则召回：满足库存、地域和资质约束。

多路召回结果先去重，再进行分数校准和配额分配。

### 7.6 召回融合

不同召回源的分数不可直接比较。
可以使用分位数归一化、z-score 或基于验证集训练融合模型。
一个简单的加权融合如下：

$$
S(i)=\sum_{m=1}^{M}\alpha_m\cdot norm(S_m(i))
$$

$\alpha_m$ 可以按 query 类型动态调整。
精确型号 query 增加 BM25 权重，模糊意图 query 增加 dense 和 SID 权重。

### 7.7 候选去重

同一商品可能来自多条召回路径，也可能存在多个库存或店铺记录。
去重键通常是 canonical_item_id，而不是展示层 item_id。
去重时要保留最有解释力的来源和最高质量的分数。
如果同一商品来自多个源，可以累计来源特征，而不是简单丢弃重复信息。

```python
def merge_candidates(groups):
    merged = {}
    for source, candidates in groups.items():
        for item in candidates:
            key = item["canonical_id"]
            record = merged.setdefault(key, {"sources": [], "score": 0.0})
            record["sources"].append(source)
            record["score"] = max(record["score"], item["score"])
    return merged
```

### 7.8 粗排与精排

召回结果通常从数千条压缩到数百条，再压缩到最终展示数量。
粗排使用低延迟模型，目标是快速去除明显不相关候选。
精排可以加入 query-item 交叉特征、用户行为和业务约束。
重排阶段处理多样性、去重、库存和运营规则。
每一层都应有独立的候选数量和延迟预算。

### 7.9 延迟预算

总延迟不是各服务平均延迟之和，而更接近关键路径上的高分位延迟。
如果 5 个并行召回服务中任意一个很慢，整体 P99 会被拖高。
建议为每个阶段配置 timeout、fallback 和熔断策略。

| 阶段 | 目标 P99 | 超时动作 |
| :--- | :--- | :--- |
| Query 解析 | 5ms | 使用原 query |
| SID 召回 | 15ms | 返回缓存结果 |
| Dense 召回 | 20ms | 使用 BM25 |
| 粗排 | 8ms | 使用召回分数 |
| 精排 | 15ms | 使用粗排分数 |
| 重排 | 5ms | 使用基础规则 |

### 7.10 缓存策略

可以缓存规范化 query、热门 query 的 SID 结果和稳定的商品 embedding。
缓存 key 必须包含模型版本、地域、设备和必要的业务约束。
库存、价格和合规状态变化时，不能只依赖长 TTL 缓存。
对于短 query，缓存命中率高；对于带用户信息的个性化 query，缓存价值较低。

### 7.11 新商品接入

新商品发布后，内容服务生成文本、图像和属性 embedding。
随后由 SID 编码器生成语义 ID，并写入前缀索引和 ANN 索引。
在索引尚未完成时，可以使用实时内容召回或相似商品迁移策略。
新商品必须经过质量和合规校验，不能因为冷启动而绕过安全规则。

### 7.12 SID 版本管理

SID 码本变化会影响序列含义和索引兼容性。
升级码本时需要记录旧 SID 到新 SID 的映射或同时保留两个版本。
不能只比较整数序列是否相同，还要比较每一层 codebook 的语义定义。

```text
sid_version = v3
codebook_version = cb_2026_07
encoder_checksum = sha256:...
index_snapshot = 2026-07-18T00:00:00Z
```

### 7.13 模型服务

在线服务至少需要暴露以下接口：

- `/encode/query`：返回 query embedding 和解析结果。
- `/retrieve/sid`：执行 SID 前缀或序列检索。
- `/retrieve/vector`：执行 ANN 检索。
- `/rank`：对候选进行排序。
- `/health`：检查模型、索引和依赖状态。

接口输出应包含 trace_id、model_version 和 index_version，便于线上追踪。

### 7.14 监控与告警

监控不能只看平均 CTR。
需要同时关注请求量、空结果率、召回覆盖率、分数分布、SID 分布和延迟。

高优先级告警包括：

- 空结果率突然上升。
- 关键类目 Recall 下降。
- 新商品 SID 生成失败。
- 索引版本滞后超过阈值。
- P99 延迟连续超预算。
- 线上分数全部接近同一常数。
- 某一召回源占比异常升高。

### 7.15 降级设计

外部模型服务不可用时，不能让整个搜索链路失败。
可以按以下顺序降级：精排模型 → 粗排模型 → 召回分数 → 热门和规则结果。
降级结果必须打标，以便排除异常流量对训练数据的污染。

### 7.16 灰度发布

新模型先在离线回放中验证，再进入小流量灰度。
灰度应同时比较相关性、业务、系统和安全指标。
如果只有 CTR 提升而退款或投诉上升，不能判定为成功。
回滚动作需要提前自动化，不能在事故时临时手工拼接命令。

### 7.17 数据与隐私

Query 可能包含姓名、地址、电话或其他敏感信息。
日志应进行脱敏、截断和访问控制。
用户 embedding 不应被当作永久身份标签，也不应无期限保存。
训练和调试数据需要最小权限、审计记录和明确保留周期。

### 7.18 线上问题定位

定位问题时先判断是数据、模型、索引还是服务层。
第一步确认请求是否进入正确版本。
第二步检查 query 解析与候选数量。
第三步抽样检查 SID 和 embedding 是否存在异常。
第四步比较各层分数与上一版本分布。
第五步回放同一 trace，确认是否可复现。

### 7.19 工业落地 Checklist

- [ ] 训练样本有时间切片。
- [ ] 标签定义经过业务确认。
- [ ] hard negative 来源可追踪。
- [ ] query 与 item encoder 版本兼容。
- [ ] SID 码本和索引版本一致。
- [ ] 多路召回有去重和归一化。
- [ ] 每层有独立延迟预算。
- [ ] 空结果和异常分数有告警。
- [ ] 具备灰度和一键回滚。
- [ ] 日志完成脱敏和权限控制。

### 7.20 小结

工业实践的难点不只在模型结构，更在数据时间、索引一致性、服务降级和实验闭环。
一个指标更高但无法复现、无法回滚、无法解释的模型不能算生产系统。
SID 的优势只有嵌入数据、码本、索引和服务协议统一后才能真正体现。

---

## 8. 评估

### 8.1 评估目标

Query-Item 对齐至少有三个层次：语义是否相关、候选是否覆盖、结果是否带来业务价值。
单一指标无法同时衡量这三个层次。
评估应拆成表示层、召回层、排序层和线上业务层。

### 8.2 数据切分

随机切分容易让同一 query 和同一商品同时出现在训练与测试中，导致指标虚高。
推荐使用按时间切分，并额外构造新 query、新 item 和长尾 query 测试集。
如果要评估冷启动，应保证测试 item 在训练窗口中没有行为标签。

```text
train: 2026-01-01 ~ 2026-06-30
valid: 2026-07-01 ~ 2026-07-07
test:  2026-07-08 ~ 2026-07-14
cold_item_test: 发布后 24 小时内的新商品
long_tail_test: 低频 query 分位数 P80 之后
```

### 8.3 Recall@K

Recall@K 衡量相关商品是否进入候选集合。

$$
Recall@K=\frac{|Relevant(q)\cap TopK(q)|}{|Relevant(q)|}
$$

当每个 query 只有一个正样本时，Recall@K 也常被写成 hit rate。
必须明确相关商品集合的定义和时间窗口，否则不同实验不可比较。

### 8.4 Precision@K

Precision@K 衡量返回的前 K 个商品中有多少相关。

$$
Precision@K=\frac{|Relevant(q)\cap TopK(q)|}{K}
$$

召回阶段的 K 通常较大，Precision 较低是正常现象。
不能为了提升召回 Precision 而过度缩小候选池，导致后续排序失去覆盖。

### 8.5 MRR

MRR 只关注第一个相关结果出现的位置。

$$
MRR=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{rank_q}
$$

它适合导航型 query 或用户通常只需要一个答案的场景。
对于多商品购物和探索式搜索，应配合 Recall、NDCG 和多样性指标。

### 8.6 NDCG

NDCG 可以使用多级相关性标签，更适合排序评估。

$$
DCG@K=\sum_{r=1}^{K}\frac{2^{rel_r}-1}{\log_2(r+1)}
$$

$$
NDCG@K=\frac{DCG@K}{IDCG@K}
$$

SID 对齐评估可以同时使用人工相关性、点击强度和购买强度构造 rel。
但不同标签混合时，必须记录每个标签的权重和截断规则。

### 8.7 Coverage

Coverage 衡量系统能够被检索或展示的 item 范围。

$$
Coverage=\frac{|\cup_{q\in Q}TopK(q)|}{|ItemPool|}
$$

Coverage 过低通常意味着热门偏置、索引错误或 embedding 坍缩。
Coverage 过高也不一定好，可能表示相关性约束不足。

### 8.8 新商品覆盖

对 SID 系统，应该单独统计新商品在候选中的出现率。

$$
NewItemCoverage=\frac{\#\text{被召回的新商品}}{\#\text{符合条件的新商品}}
$$

可以按商品年龄分桶：0-1 天、2-7 天、8-30 天和成熟商品。
这样能观察冷启动收益是否只发生在极少数热门新商品上。

### 8.9 长尾分桶评估

按 query 频次、字符长度、语言和类目分别分桶。
头部 query 指标高不能证明系统解决了长尾问题。
每个桶至少报告样本量、Recall@K、MRR、空结果率和置信区间。
当长尾样本量较小时，建议使用 bootstrap 估计区间而不是只看单点指标。

### 8.10 语义检索的人工评估

人工评估需要明确标注准则和冲突处理流程。
至少区分完全相关、部分相关、可替代、不相关和违规结果。
标注员应看到 query、商品核心信息和必要的上下文，但不应看到模型分数。
需要抽检标注一致性，并对分歧样本进行复核。

### 8.11 评估代码

```python
def recall_at_k(retrieved, relevant, k):
    retrieved_k = set(retrieved[:k])
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(retrieved_k & relevant) / len(relevant)


def mean_recall(dataset, k):
    values = [recall_at_k(r, y, k) for r, y in dataset]
    return sum(values) / max(len(values), 1)
```

评估代码必须处理空相关集合、重复结果和无效 item。

### 8.12 分数校准

相似度分数不一定是概率。
如果要把分数用于多路融合或阈值过滤，需要进行 calibration。
可以使用 Platt scaling、isotonic regression 或验证集分桶统计。
校准模型也必须按版本管理，不能只更新主模型而复用旧校准参数。

### 8.13 离线回放

离线回放用历史 query 和商品快照重建完整请求。
回放时必须模拟当时可见的库存、价格、索引和用户状态。
如果直接使用当前商品信息，评估会泄漏未来数据。
回放结果可以用于比较模型，但不能替代真实在线实验。

### 8.14 在线 A/B

A/B 实验需要随机分桶、固定实验单元和预先声明的指标。
实验单元可以是用户、设备或请求，但同一用户应尽量保持分组稳定。
主要指标应对应产品目标，辅助指标用于监控副作用。
实验周期应覆盖工作日和周末，并避免只在单一促销活动期间结论。

### 8.15 显著性与置信区间

提升幅度小于噪声时不能直接宣布胜出。
CTR、CVR 和 GMV 的分布通常不完全服从简单正态分布。
可以使用 bootstrap、分层检验或适合比例数据的统计方法。
同时运行多个指标和多个切片时，要考虑多重比较导致的假阳性。

### 8.16 Interleaving

Interleaving 将两个模型的候选混合到同一个结果页，适合快速比较排序偏好。
它减少了用户间差异，但不适合直接估计长期留存或收入提升。
Interleaving 结果应作为方向判断，最终仍需要正式 A/B 实验。

### 8.17 业务指标

语义对齐模型最终要服务业务和用户体验。
常见指标包括 CTR、加购率、购买率、GMV、搜索后退出率和重复搜索率。
不同业务不能把所有指标压成一个没有解释的总分。
应明确主指标、护栏指标和长期指标。

```text
主指标：有效购买率
护栏指标：投诉率、退款率、空结果率、P99 延迟
长期指标：7 日留存、复购率、用户满意度
```

### 8.18 多样性与新颖性

多个结果都来自同一品牌或同一商品族，相关性可能高但体验单调。
可以统计品牌、类目、价格段和内容主题的覆盖。
新颖性可以通过用户历史未见过的结果比例衡量。
多样性不应简单地随机打散，而应在保持相关性的前提下重排。

### 8.19 公平性评估

热门商家和头部品牌天然拥有更多曝光。
评估时应比较不同商家规模、类目和地区的曝光机会。
公平性约束要结合业务规则，不能为了形式上的均匀而展示低质量或不合规商品。

### 8.20 鲁棒性评估

测试集应加入拼写错误、口语化表达、混合语言、数字型号和极短 query。
还应测试商品标题缺失、图像损坏、属性冲突和 SID 不完整等异常 item。
鲁棒性评估的重点是系统是否优雅降级，而不是所有异常都返回高分。

### 8.21 对抗与安全评估

恶意商家可能通过堆砌关键词让商品进入错误的语义邻域。
系统需要测试关键词重复、无意义符号、伪造品牌和 prompt injection 等输入。
对用户 query 也应防止敏感信息被回显到解释或日志中。

### 8.22 评估报告模板

每次模型评估至少包含：

- 模型、索引、码本和特征版本。
- 训练和测试时间范围。
- 数据量、query 数和 item 数。
- 头部、腰部、长尾和新 item 分桶指标。
- Recall、NDCG、MRR、Coverage 和空结果率。
- P50、P95、P99 延迟。
- 与基线的绝对变化和相对变化。
- 失败样例、风险和下一步实验。

### 8.23 评估反模式

- 只报告一个总体平均值。
- 随机切分造成时间泄漏。
- 用点击当作完美相关性标签。
- 只在训练商品池上评估。
- 只看离线指标不做线上实验。
- 忽略空结果、重复结果和违规结果。
- 用一次异常高峰数据决定模型方向。

### 8.24 小结

评估是从“向量看起来相似”走向“系统真的有效”的桥梁。
必须按数据切分、用户意图、商品生命周期和系统链路拆分指标。
一个可信的结论应能回答：覆盖了谁、改善了什么、代价是什么、是否可复现。

---

## 9. Query 理解

### 9.1 Query 理解的任务

Query 理解把用户输入转化为可检索、可排序和可解释的结构。
它不仅是分词，还包括纠错、实体识别、属性抽取、意图分类和改写。
理解模块输出的结构应被下游模型稳定消费，而不是只返回一段改写文本。

```text
原始 query
  → 规范化
  → 分词与实体识别
  → 意图与类目预测
  → 属性约束
  → 多路改写
  → Query embedding / Query SID
```

### 9.2 规范化

规范化处理大小写、全半角、空格、标点、单位和常见别名。
例如“iphone15”“iPhone 15”和“苹果15”可以共享部分规范化结果。
规范化不能过度合并，因为型号、容量和代际差异可能决定商品是否匹配。
原始 query 必须保留，用于审计、回放和展示。

### 9.3 分词

中文 query 需要同时考虑词典分词和子词切分。
品牌、型号、数字和英文混排不能被普通分词器随意拆散。
分词结果应包含 token 类型和偏移位置，方便属性映射。

```python
def normalize_query(text):
    text = text.strip().lower()
    text = text.replace("　", " ")
    text = " ".join(text.split())
    return text
```

### 9.4 拼写纠错

纠错候选可以来自词典、编辑距离、搜索日志和语言模型。
纠错不能盲目替换低频词，因为低频可能是新品牌或新型号。
高置信纠错可以直接采用，低置信候选应并行保留原 query。

```python
def choose_correction(original, candidates, threshold=0.85):
    if not candidates:
        return original
    best_text, best_score = max(candidates, key=lambda x: x[1])
    return best_text if best_score >= threshold else original
```

### 9.5 同义词和别名

同义词包括口语词、行业词、品牌别名和平台常用表达。
同义词扩展应该保留词源和置信度，避免多个扩展互相冲突。
“手机壳”和“手机套”大概率相近，但“裸机”则可能表达相反的购买约束。

### 9.6 实体识别

常见实体包括品牌、类目、型号、系列、材质、颜色、尺寸、功率和地域。
实体识别结果应包括实体类型、文本 span、标准化值和置信度。

| 实体 | 示例 | 标准化 |
| :--- | :--- | :--- |
| 品牌 | 耐克、Nike | Nike |
| 型号 | 15 Pro Max | iPhone 15 Pro Max |
| 颜色 | 远峰蓝 | blue_titanium |
| 尺寸 | 256G | 256GB |
| 类目 | 跑鞋 | running_shoes |

### 9.7 属性约束

Query 中的属性可能是硬约束，也可能只是偏好。
“必须支持 5G”通常是硬约束，“轻薄笔记本”可能需要软约束和排序偏好。
可以输出两套字段：filter_constraints 与 ranking_preferences。

```json
{
  "entities": [
    {"type": "category", "value": "laptop", "confidence": 0.98},
    {"type": "attribute", "key": "weight", "value": "light", "confidence": 0.72}
  ],
  "hard_filters": [],
  "soft_preferences": ["light_weight"]
}
```

### 9.8 意图分类

常见意图包括导航型、信息型、交易型、比较型和售后型。
意图决定召回范围、展示模板和排序目标。
导航型 query 可能优先品牌官方店，比较型 query 需要多品牌和属性覆盖。
交易型 query 更关注库存、价格和配送能力。

### 9.9 类目预测

类目预测可以使用层级分类器，从一级类目逐级预测到叶子类目。
直接预测叶子类目速度快，但错误一旦发生会完全过滤正确商品。
生产系统应保留 top-n 类目候选，并把类目预测作为软特征参与排序。

### 9.10 Query 改写

改写不是生成一个“更漂亮”的 query，而是生成更适合检索的等价表达。
改写要防止引入原 query 没有表达的品牌、型号或价格条件。
每条改写应保存来源：规则、统计模型、LLM 或用户历史。

```python
class RewriteCandidate:
    def __init__(self, text, source, confidence):
        self.text = text
        self.source = source
        self.confidence = confidence


def retain_candidates(original, candidates, limit=4):
    candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
    output = [original]
    for candidate in candidates:
        if candidate.text not in output and len(output) < limit:
            output.append(candidate.text)
    return output
```

### 9.11 Query 扩展

扩展可以补充同义词、上位类目、相关属性和常见品牌。
扩展数量不能无限增加，否则候选池噪声和延迟都会上升。
应根据 query 类型决定扩展策略：型号 query 少扩展，模糊 query 多扩展。

### 9.12 多轮上下文

对话式搜索需要解析代词、省略和上下文继承。
用户说“那红色的呢”，系统必须知道“那”指向上一轮的商品类目。
上下文状态应有有效期，避免几小时后仍继承旧意图。
涉及价格、地域和配送的约束应明确展示给用户，支持纠正。

### 9.13 个性化理解

同一个 query 在不同用户和场景下可能有不同含义。
个性化只能补充意图，不能覆盖用户明确写出的硬约束。
用户历史信号应按短期、长期和会话内分层。
当用户输入明确型号时，个性化权重应自动降低。

### 9.14 多语言和混合语言

跨境搜索经常出现中文、英文、数字和本地语言混合。
可以在共享多语言空间中编码，也可以生成翻译 query 并行召回。
翻译不能破坏品牌和型号 token，应使用术语表和保护 span。

### 9.15 Query 质量指标

Query 理解模块可以独立评估：

- 纠错准确率。
- 实体识别 precision、recall、F1。
- 意图分类 macro-F1。
- 类目 top-1 和 top-5 accuracy。
- 改写后的 Recall 提升。
- 改写导致的误召回率。
- 解析延迟和缓存命中率。

### 9.16 Query 与 SID 的连接

Query 理解输出的实体和属性可以作为 SID 预测的辅助输入。
粗粒度类目影响 SID 前缀，型号和关键属性影响后续码本位置。
当 Query 解析不确定时，可以生成多个 SID 候选，再由检索和排序共同决定。

$$
P(sid|q)=\sum_z P(sid|z,q)P(z|q)
$$

其中 $z$ 可以表示类目、实体和意图结构。

### 9.17 查询改写的安全边界

LLM 改写必须避免将用户输入变成不相关的营销文案。
需要限制输出长度、词表、实体变更和敏感内容。
改写结果要经过结构化解析和规则校验，不能直接作为硬过滤条件。

```python
def validate_rewrite(original, rewritten, protected_spans):
    if len(rewritten) > 128:
        return False
    for span in protected_spans:
        if span not in rewritten:
            return False
    return True
```

### 9.18 空 query 和异常 query

空 query 可以返回用户最近意图、热门商品或类目入口。
只有标点、随机字符或超长字符串时，应快速返回安全兜底。
恶意 query 不应触发高成本的大模型链路。

### 9.19 Query 理解的线上链路

线上链路必须将解析结果和原 query 同时传给下游。
如果某个解析服务超时，下游应能使用原 query 继续检索。
解析结果应附带版本，以便离线回放时精确复现。

### 9.20 小结

Query 理解的目标是把自然语言输入变成可控的结构化意图。
高质量解析既要提高召回，也要避免过度改写、错误过滤和隐私泄漏。
在 SID 系统中，Query 理解决定了 SID 预测从哪个语义区域开始搜索。

---

## 10. 面试题

### 10.1 什么是 Query-Item 语义对齐

**参考回答**：
Query-Item 语义对齐是将用户查询和商品内容映射到兼容的表示空间，使相关 query-item 的相似度更高，不相关样本的相似度更低。工业系统通常用双塔完成大规模召回，再用交叉编码器或其他交互模型完成精排。SID 提供了结构化、离散化的商品语义表示，可以作为向量检索和生成式检索之间的桥梁。

回答时不要只说“把两个向量做 cosine”，还要说明训练数据、负样本、索引和线上评估。

### 10.2 双塔为什么快

**参考回答**：
双塔将 query 和 item 独立编码。商品向量可以离线预计算并建立 ANN 索引，在线只需要计算一次 query embedding，然后执行近邻检索。因此复杂度不再是对每个 query-item 都运行一次交叉模型。代价是独立编码损失了细粒度交互，所以双塔更适合召回而不是最终排序。

### 10.3 双塔为什么不适合直接精排

独立编码后，query 和 item 的 token 级关系被压缩到单个向量。
“红色 256GB”“不含充电器”“适合窄脚”等细节可能在压缩中丢失。
精排阶段候选数量较少，可以使用 cross-attention 或交叉特征恢复这些关系。

### 10.4 如何选择负样本

**参考回答**：
先使用随机负样本建立基本区分能力，再加入曝光未点、同类目商品、BM25 相似但不相关商品和 ANN 难负样本。要警惕 false negative，尤其是一个 query 对应多个相关 item 时，不能把所有未点击商品都当作强负样本。负样本应覆盖线上真实竞争分布，并按 query 频次和业务场景分层评估。

### 10.5 什么是 in-batch negative

一个 batch 中第 $b$ 个 query 与第 $b$ 个 item 是正配对，其他 item 作为该 query 的负样本。
这样 B 个正配对同时产生约 $B(B-1)$ 个负关系。
优点是实现简单、吞吐高；缺点是 batch 中可能包含 false negative。
如果 batch 内样本高度相似或有多个正样本，必须使用 positive mask。

### 10.6 InfoNCE 的 temperature 有什么作用

temperature 控制 softmax 分布的尖锐程度。
温度小，模型更关注最相似的竞争样本，区分性强但可能训练不稳定。
温度大，梯度更平滑但容易导致区分不足。
应该在验证集上观察 Recall、梯度、embedding 分布和线上延迟，而不是只凭经验设置。

### 10.7 如何判断 embedding 坍缩

可以检查向量范数、各维方差、协方差谱和随机 pair 的 cosine 分布。
如果所有向量彼此相似、各维方差很小，说明表示可能坍缩。
原因包括损失设计不完整、负样本不足、温度异常或输入特征泄漏。
加入 uniformity、增大有效负样本和检查数据通常比盲目加深网络更有效。

### 10.8 SID 与传统 item ID 的差别

传统 item ID 主要是索引符号，不表达商品之间的语义关系。
SID 是由内容或语义量化得到的离散序列，不同位置可以承载层级语义。
因此新商品可以根据内容生成 SID，不必等待大量交互数据。
SID 也会带来码本坍塌、版本兼容和碰撞率等新的工程问题。

### 10.9 SID 如何帮助冷启动

新商品没有点击和购买历史，但通常有标题、属性、图像和类目信息。
内容编码器可以将这些信息量化成 SID，再映射到已有语义空间。
Query 侧也编码到相同空间后，新商品就能被语义召回。
冷启动不是“完全没有数据”，而是从行为信号转向内容和先验信号。

### 10.10 SID 检索和向量检索如何组合

可以让 SID 前缀检索提供结构化候选，让 dense ANN 提供同义和跨词面候选，再由 BM25 补充精确型号匹配。多路结果去重后，使用校准分数融合。对于强约束 query，规则和 BM25 的权重更高；对于模糊 query，SID 和 dense 的权重更高。最终排序模型负责处理用户偏好、库存和业务目标。

### 10.11 如何处理多正样本

构造正样本集合而不是单个 label。
在 InfoNCE 中使用 positive mask，对所有正样本平均 log probability。
也可以按照购买、加购、点击设置不同软权重。
评估时使用多个 relevant item，并报告 Recall 与 NDCG，而不是只看第一个命中。

### 10.12 如何处理长尾 query

**参考回答**：
第一，使用 query 改写、纠错和同义词扩展提高表达覆盖；第二，用类目、实体和属性预测提供结构化先验；第三，使用回译、模板和 LLM 生成进行数据增强；第四，采样时提高长尾权重但保留真实分布；第五，对新 query 单独做时间切分评估。不能只把长尾 query 改写成头部 query，否则会丢失原始细节。

### 10.13 如何做多语言对齐

使用共享多语言 encoder 或分别编码后通过投影层对齐。
训练数据可以来自跨语言 query-item 配对、翻译对和商品多语言描述。
品牌、型号、数字和单位应使用术语保护，避免翻译破坏实体。
评估要按语言对、国家和 query 类型拆分，不能只报告整体平均。

### 10.14 交叉编码器和 Late Interaction 的取舍

交叉编码器把 query 和 item 放到同一个 Transformer 中，交互能力最强，但无法离线预计算所有商品。Late Interaction 保留 token 级向量，检索时执行较轻量的交互，速度和精度处于两者之间。候选池规模决定取舍：大规模召回用双塔，较小候选集可用 Late Interaction 或 cross-encoder。

### 10.15 召回指标和排序指标有什么区别

召回阶段主要看 Recall@K、Coverage、空结果率和延迟，问题是“正确答案有没有进入候选池”。排序阶段主要看 NDCG、MRR、Precision 和业务排序指标，问题是“正确答案是否排在前面”。如果召回 Recall 已经很高但 NDCG 不变，应优化排序；如果 Recall 很低，继续调精排通常没有收益。

### 10.16 离线指标提升但线上下降怎么办

先检查数据切分、标签时间、候选池和线上流量是否一致。
再检查模型是否过度利用位置、热门度或泄漏特征。
比较空结果率、延迟、重复率、库存命中和长期指标。
通过离线回放重现线上请求，确认是相关性问题还是服务链路问题。

### 10.17 如何处理位置偏差

曝光位置影响点击概率，所以不能把点击直接视为无偏标签。
可以记录 position、visibility 和 page depth，使用逆倾向加权或位置特征校正。
还可以通过随机探索流量收集更接近无偏的反馈。
校正后的数据应控制极端权重，避免训练被少数样本支配。

### 10.18 如何设计 A/B 实验

先明确一个主指标和若干护栏指标。
以用户为单位稳定分桶，保证实验组和对照组在时间上覆盖相同流量。
实验前估算样本量和最小可检测差异，实验期间监控 SRM、延迟和异常流量。
实验结束后同时报告绝对提升、相对提升、置信区间和副作用。

### 10.19 线上延迟如何优化

减少在线编码长度，缓存热门 query，批量执行 ANN 请求。
将多路召回并行化，并设置每路独立 timeout。
商品 embedding 离线预计算，精排只处理足够小的候选集。
使用蒸馏、量化和算子融合降低模型耗时。
优化后必须报告 P50、P95 和 P99，而不是只看平均延迟。

### 10.20 如何设计降级链路

当精排超时，使用粗排分数；粗排异常时使用召回分数；召回服务异常时使用缓存、BM25 或热门规则结果。每次降级都记录原因和版本，避免将降级流量混入正常训练标签。降级结果还要遵守库存、合规和地域限制。

### 10.21 如何排查空结果

按顺序检查：

1. 原 query 是否为空或被错误清洗。
2. Query encoder 是否加载正确版本。
3. SID 码本与索引版本是否一致。
4. ANN 索引是否为空或分片不可用。
5. 过滤条件是否过严。
6. 库存和地域服务是否返回错误。
7. 降级召回是否被超时或熔断。

空结果率应按 query 类型和流量来源拆分，否则平均值会掩盖局部故障。

### 10.22 如何识别索引过期

为索引写入 snapshot_time、model_version、codebook_version 和 item_count。
服务每次请求返回或记录当前 index_version。
监控线上请求版本与最新可用版本的差距。
如果 item_count、SID 分布或 embedding norm 突然变化，应暂停发布并回滚。

### 10.23 如何解释 SID

可以解释 SID 每一层对应的码本簇、类目、属性或近邻商品。
解释必须说明置信度和模型版本，不能把码本索引当作绝对业务含义。
如果码本没有稳定语义，应展示“与这些属性和商品相近”而不是虚构自然语言解释。

### 10.24 如何评价一个候选方案

面试中应从四个维度回答：

- 相关性：能否找到真正匹配的商品。
- 覆盖：头部、长尾、新商品是否都能被召回。
- 系统：延迟、QPS、成本和回滚是否可控。
- 业务：点击之外，购买、留存和安全是否改善。

### 10.25 设计题：从零搭建 Query-Item 系统

**回答框架**：

1. 明确场景：搜索还是推荐，是否有明确 query。
2. 定义目标：主指标、护栏指标和延迟预算。
3. 设计数据：曝光日志、行为标签、商品内容和时间切片。
4. 选择召回：BM25、双塔 ANN、SID 和协同过滤多路并行。
5. 设计排序：粗排、精排和多样性重排。
6. 处理冷启动：内容编码、SID 和新商品快速索引。
7. 评估上线：离线分桶、回放、A/B、监控和回滚。

### 10.26 追问：为什么不用一个大模型端到端解决

大模型可以提升 Query 理解和精排，但每次对全量商品执行端到端推理的成本和延迟不可接受。生产系统需要在召回规模、吞吐、可控性、可解释性和故障降级之间平衡。合理方案是让大模型承担高价值、低规模或离线教师角色，而不是无条件替代所有检索组件。

### 10.27 追问：SID 会不会发生碰撞

会。量化空间有限，不同商品可能映射到相同或相近 SID。
需要监控 SID collision rate、同 SID 商品的语义一致性和后续排序可分性。
可以增加 SID 长度、码本大小或引入 residual code，也可以保留 item_id 作为最终唯一标识。
SID 负责语义检索，不应被误当作唯一业务主键。

### 10.28 追问：如何避免 SID 码本坍塌

观察每个 codebook 的使用频率和熵。
对使用率极低的 code 进行重初始化或增加均衡正则。
训练数据需要覆盖长尾语义，不能让头部商品占满所有更新。
同时关注重构误差和检索指标，因为单纯追求均匀使用可能损害语义质量。

### 10.29 追问：如何处理商品下架

索引删除和业务过滤都要支持。
商品下架后应优先从在线过滤层立即阻断，再异步从 ANN 和 SID 索引删除。
缓存结果需要按下架事件失效。
回放和评估时也要使用当时的上下架状态，避免把不可展示商品算成命中。

### 10.30 追问：新 query 没有训练标签怎么办

使用预训练语言模型、词法召回、实体和类目先验获得初始能力。
通过同义 query、模板生成和相似 query 迁移提供弱监督。
线上保留小比例探索流量，收集新 query 的反馈。
之后按时间窗口持续训练，但要防止新流量的噪声快速污染基础模型。

### 10.31 追问：为什么线上 CTR 提升但 GMV 下降

模型可能更偏好容易点击但低转化的商品，也可能展示了低价、低质量或标题党商品。
需要分解点击后的加购、购买、退款和客单价。
将点击作为唯一目标会造成目标错配，应使用多任务或分阶段目标。
同时检查排序是否改变了品牌、库存和价格段分布。

### 10.32 追问：如何做多目标排序

先定义业务目标及其约束关系，再决定加权、级联或约束优化。
一个基础形式是：

$$
S=\alpha R+\beta C+\gamma P+\delta D-\eta Risk
$$

其中 $R$ 是相关性，$C$ 是点击倾向，$P$ 是购买倾向，$D$ 是多样性，Risk 是风险。
权重需要通过离线回放和线上实验校准，不能只由模型 loss 自动决定。

### 10.33 追问：如何向业务解释 embedding

不应只展示一个无法理解的向量。
可以展示 query 的实体、属性、意图、最近邻商品、SID 前缀和召回来源。
对比不同版本在相同 query 上的候选变化。
业务最关心的是“为什么被召回、为什么排在这里、指标是否改善、出现问题能否修复”。

### 10.34 追问：如果只有很小的数据集怎么办

优先使用预训练 encoder 和轻量投影层，减少可训练参数。
采用人工构造的高质量正负样本、数据增强和交叉验证。
不要过早训练复杂的多塔多任务系统。
先验证语义空间和评估流程，再随着日志规模增加引入 hard negative 和在线反馈。

### 10.35 追问：如何保证训练和服务一致

统一 tokenizer、规范化规则、特征字典、模型配置和 SID 版本。
把预处理逻辑作为可版本化组件，并用同一批 golden cases 做离线和在线校验。
上线前比较离线编码与服务编码的 checksum 或数值误差。
服务返回版本信息，发现不一致时优先停止流量而不是继续调参。

### 10.36 面试答题模板

面对 Query-Item 对齐问题，可以按“定义—架构—训练—评估—上线—风险”组织回答。
先说明场景和目标，再解释双塔、SID、ANN 与精排的分工。
接着讲标签、正负样本、损失和难例处理。
最后补充延迟、冷启动、偏差、安全、A/B 和回滚。

```text
定义问题
  → 划分召回与排序
  → 设计 Query/Item 表示
  → 构造正负样本
  → 训练对比目标
  → 评估分桶指标
  → 设计线上链路
  → 监控与降级
```

### 10.37 常见失分点

- 只会背双塔定义，不会解释 ANN 索引。
- 只说 InfoNCE，不讨论 false negative。
- 把未点击商品当作绝对负样本。
- 忽略时间泄漏和位置偏差。
- 只谈 CTR，不谈购买、退款和留存。
- 只谈模型精度，不谈延迟、成本和回滚。
- 把 SID 当作普通 item ID。
- 没有冷启动、新 query 和新 item 的评估方案。

### 10.38 最终复盘清单

1. 我能否说明 Query 和 Item 的信息差异？
2. 我能否解释双塔为什么适合召回？
3. 我能否写出双向 InfoNCE？
4. 我能否识别 false negative？
5. 我能否说明 SID 的层级和冷启动价值？
6. 我能否设计多路召回和候选去重？
7. 我能否区分 Recall 与 NDCG 的责任边界？
8. 我能否处理位置偏差和时间泄漏？
9. 我能否给出延迟预算和降级链路？
10. 我能否解释线上指标与长期业务指标的冲突？

### 10.39 章节总结

Query-Item 语义对齐不是一个孤立的 encoder 训练问题。
它连接了 Query 理解、内容建模、SID 量化、对比学习、ANN 检索、排序、评估和生产服务。
双塔提供规模化召回，交互模型提供精细判断，SID 提供结构化语义和冷启动能力。
真正可靠的系统还必须处理数据偏差、索引版本、延迟预算、隐私安全和实验回滚。

### 10.40 下一步

完成本章后，可以继续学习冷启动的数学解释，并重点关注以下问题：

- 内容信息如何转化为可泛化的 SID。
- 训练时没有行为数据的新商品如何获得质量估计。
- 量化误差如何影响检索边界。
- 语义 SID 与用户行为 SID 如何融合。
- 生成式检索如何在约束解码下保证有效商品。

[3-冷启动数学解释.md](第05章-03-冷启动数学解释.md)

---

## 附：本章新增内容索引

| 章节 | 核心主题 | 适合复习的问题 |
| :--- | :--- | :--- |
| 5 | 对比学习对齐 | 如何构造正负样本与稳定 InfoNCE |
| 6 | 双塔模型 | 为什么独立编码可以支撑大规模召回 |
| 7 | 工业实践 | 如何把模型接入索引、服务和实验系统 |
| 8 | 评估 | 如何拆分离线、在线、业务和风险指标 |
| 9 | Query 理解 | 如何将自然语言转成结构化检索意图 |
| 10 | 面试题 | 如何用工程化框架回答系统设计问题 |

> 本节用于快速定位复习主题，不替代前文的推导、代码和注意事项。
