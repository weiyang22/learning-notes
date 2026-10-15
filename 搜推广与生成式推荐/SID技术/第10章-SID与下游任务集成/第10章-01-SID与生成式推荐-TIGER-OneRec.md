---
title: 1-SID与生成式推荐-TIGER-OneRec
tags:
  - 方向
  - SID
description: class TIGER(nn.Module):...
chapter_id: 10.1
last_updated: 2026-07-14
status: ✅ v1.0

---

# 1. SID 与生成式推荐: TIGER / OneRec

> **本章定位**: SID 在经典 GR 模型中的用法
> **预计时长**: 90 分钟

---

## 1.1 TIGER (Meta 2023)

```python
class TIGER(nn.Module):
    def __init__(self, num_codes=256, num_layers=5, hidden=128, n_heads=4):
        super().__init__()
        # K 个码本的 Embedding
        self.code_embs = nn.ModuleList([nn.Embedding(num_codes, hidden) for _ in range(num_layers)])
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden, n_heads, dim_feedforward=hidden*4)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        # 输出头: 每层一个
        self.heads = nn.ModuleList([nn.Linear(hidden, num_codes) for _ in range(num_layers)])

    def forward(self, history_sids):
        # history_sids: [B, T, K]
        b, t, k = history_sids.shape
        # 嵌入
        embs = [self.code_embs[i](history_sids[..., i]) for i in range(k)]  # K 个 [B, T, D]
        x = torch.stack(embs, dim=2).view(b, t*k, -1)  # [B, T*K, D]
        # Transformer
        x = self.transformer(x, x)
        # 预测下一个 SID
        last = x[:, -k:]  # 最后 K 个
        logits = [self.heads[i](last[:, i]) for i in range(k)]
        return logits  # K 个 [B, num_codes]
```

---

## 1.2 OneRec (阿里 KDD 2024)

**特点**: 端到端 GR,无召回阶段。

```python
class OneRec(nn.Module):
    """端到端生成式推荐"""
    def __init__(self, num_codes=256, num_layers=5, hidden=128):
        super().__init__()
        # SID Embedding
        self.code_embs = nn.ModuleList([nn.Embedding(num_codes, hidden) for _ in range(num_layers)])
        # 编码用户历史
        self.user_encoder = nn.TransformerEncoder(...)
        # 解码
        self.decoder = nn.TransformerDecoder(...)
        # 输出头
        self.heads = nn.ModuleList([nn.Linear(hidden, num_codes) for _ in range(num_layers)])
```

**优势**: 一个模型端到端,无召回-粗排-精排多阶段。

---

## 1.3 训练

```python
# 损失: K 个交叉熵
losses = []
for k in range(num_layers):
    loss = F.cross_entropy(logits[k], target_sids[..., k])
    losses.append(loss)
total_loss = sum(losses)
```

---

## 1.4 自检题

- [ ] TIGER 的核心架构?
- [ ] OneRec 与 TIGER 的区别?

---

## 1.5 下一步

[2-SID与多任务学习.md](第10章-02-SID与多任务学习.md)

## 5. TIGER 架构详解

### 5.0 本节目标

TIGER 的核心思想是把推荐系统中的候选物品表示成一串离散语义 ID。
模型不再直接从固定物品表中预测一个 item index。
模型先读取用户历史中的 SID token。
模型再按自回归方式生成下一个物品的 SID token。
生成完成后，SID 解码器把 token 序列还原成真实物品。
因此，推荐问题被转写为序列建模问题。
这种转写让推荐系统可以复用 Transformer 的建模能力。
这种转写也让物品之间的语义相似关系进入输出空间。
本节先拆解数据表示。
然后拆解序列化方式。
接着分析编码器、解码器和多头输出。
最后讨论训练、推理和工程边界。

### 5.1 从 item ID 到 semantic ID

传统模型通常将每个物品分配一个整数编号。
这个编号只承担索引作用。
item 1001 和 item 1002 的编号相邻并不代表语义相近。
Embedding 可以学习相似性。
但输出空间仍然是巨大的平面分类空间。
物品数量增长时，最后一层线性层也会随之扩大。
SID 把物品映射到多个离散码本。
一个物品可以表示为 `(c1, c2, ..., cK)`。
每个 `ci` 都来自大小为 `V` 的码本。
理论上的组合空间是 `V^K`。
真实物品只占用其中一小部分组合。
码本中的每一层可以承担不同粒度的语义。
第一层通常承载较粗的类别或主题信息。
后续层逐步补充属性、风格和实例差异。
同类物品往往共享前缀。
不同类物品往往在早期 token 上分开。
相似物品可以拥有较长的公共前缀。
这种结构为树形检索和前缀约束提供了基础。

### 5.2 SID 张量的形状

推荐历史可以表示为 `[B, T, K]`。
`B` 表示 batch size。
`T` 表示每个用户的历史长度。
`K` 表示 SID 的码本层数。
最后一维不是普通的连续特征。
最后一维中的每个位置来自一个离散码本。
例如 `K=4` 时，一个物品可能是 `[12, 83, 4, 201]`。
同一物品的四个数字需要按固定顺序解释。
不能在训练和推理时交换码本层的顺序。
padding 需要有独立的 token 约定。
padding 不能与有效 SID code 冲突。
未知物品需要单独的 `UNK` 处理策略。
历史中的重复物品可以保留。
也可以在数据预处理阶段折叠重复行为。
这两种选择会改变序列分布。
推荐时间戳应与 SID 一起保留。
因为相同 SID 在不同时间的意义可能不同。
行为类型也可以作为额外 token。
例如点击、加购、购买可以使用不同的事件 token。

### 5.3 一个样本的序列化

原始行为序列可以写成：

```text
用户 u
  点击 item_a
  停留 item_b
  购买 item_c
  点击 item_d
```

对应的 SID 序列可以写成：

```text
<BOS> c1_a c2_a c3_a c4_a
<SEP> c1_b c2_b c3_b c4_b
<SEP> c1_c c2_c c3_c c4_c
<SEP> c1_d c2_d c3_d c4_d <EOS>
```

`<BOS>` 表示用户序列起点。
`<SEP>` 表示物品边界。
`<EOS>` 表示生成结束。
如果使用固定长度 SID，可以省略显式 `<SEP>`。
省略分隔符时，模型必须知道 `K`。
如果不同物品的 SID 长度可变，就必须使用边界标记。
固定长度可以降低解码复杂度。
可变长度可以表达不同的层级深度。
工程上通常优先采用固定长度。
固定长度更容易进行 batch padding。
固定长度也更容易使用并行 teacher forcing。

### 5.4 TIGER 的整体数据流

输入是用户过去交互过的物品 SID。
每个 SID code 先经过对应码本的 embedding。
同一物品的 K 个 code 得到 K 个向量。
这些向量按照预定义顺序展平。
展平后，一个历史物品对应 K 个 Transformer token。
多个历史物品连接成一个长序列。
位置编码标记 token 的时间和码本位置。
Transformer 读取整段历史。
最后的隐藏状态用于预测目标物品。
目标物品也被拆成 K 个 code。
每个 code 使用一个独立输出头。
所有输出头共享 Transformer 的隐藏表示。
第一个输出头预测粗粒度 code。
第二个输出头预测下一级 code。
后续输出头继续补充细节。
训练时可以一次计算 K 个交叉熵。
推理时可以一次生成一个完整 SID。
也可以逐层生成并进行前缀约束。

### 5.5 词表设计

最简单的做法是每层码本使用相同词表大小。
例如所有层的 code 范围都是 `[0, 255]`。
这样可以用多个相同大小的输出头。
如果各层语义差异很大，也可以使用不同词表大小。
例如第一层使用 128 个 code。
第二层使用 256 个 code。
第三层使用 512 个 code。
不同词表需要不同的 embedding 和输出维度。
共享 code embedding 可以减少参数量。
但共享会弱化不同层的语义角色。
不共享 embedding 通常更容易训练。
输出头是否共享是另一个独立选择。
共享输出头需要增加层类型 embedding。
独立输出头可以直接区分码本位置。

词表必须包含 padding 位置。
损失计算时应忽略 padding target。
特殊 token 应与普通 code 隔离。
不能把 `<EOS>` 当作普通语义 code。
码本重建后需要检查冲突。
如果两个物品映射到了同一个 SID，解码会产生歧义。
可以在离线阶段维护 SID 到 item 的反向索引。
反向索引必须支持一对多诊断。
冲突率是 SID 质量的重要指标。
未覆盖物品也需要被单独统计。
新物品可以通过编码器生成 SID。
也可以暂时进入 fallback item index 空间。

### 5.6 Embedding 层

下面的实现只使用 PyTorch 基础组件。
每一个码本有独立的 embedding 表。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIDEmbedding(nn.Module):
    def __init__(self, codebook_size, num_layers, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.tables = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.layer_embedding = nn.Embedding(num_layers, hidden_size)

    def forward(self, sid):
        # sid: [batch, time, num_layers]
        parts = []
        layer_ids = torch.arange(
            self.num_layers,
            device=sid.device,
        )
        for layer_id, table in enumerate(self.tables):
            value = table(sid[..., layer_id])
            value = value + self.layer_embedding(layer_ids[layer_id])
            parts.append(value)
        return torch.stack(parts, dim=2)
```

`parts` 中每个元素的形状是 `[B, T, D]`。
stack 后的形状是 `[B, T, K, D]`。
码本层 embedding 让模型知道当前 token 来自哪一层。
即使每层使用独立表，层 embedding 仍然有助于表示结构。
如果只使用独立表，可以去掉层 embedding。
去掉后模型仍然可以运行。
但调试时不容易观察层间共享行为。

### 5.7 时间位置与码本位置

普通 Transformer 只有一个线性位置索引。
SID 序列同时包含行为时间位置和码本位置。
第 3 个历史物品的第 2 个 code 与第 2 个历史物品的第 3 个 code 不应混淆。
因此可以使用两类位置 embedding。
一类是行为位置 embedding。
一类是码本位置 embedding。
行为位置对 K 个 code 可以重复。
码本位置在每个物品内部从 0 到 K-1 变化。

```python
class SIDPosition(nn.Module):
    def __init__(self, max_time, num_layers, hidden_size):
        super().__init__()
        self.time = nn.Embedding(max_time, hidden_size)
        self.layer = nn.Embedding(num_layers, hidden_size)

    def forward(self, batch_size, time, num_layers, device):
        t = torch.arange(time, device=device)
        k = torch.arange(num_layers, device=device)
        time_emb = self.time(t)[None, :, None, :]
        layer_emb = self.layer(k)[None, None, :, :]
        return time_emb + layer_emb
```

时间 embedding 的长度必须覆盖截断后的最大历史长度。
如果历史动态变长，超过长度时应先截断。
不要静默使用越界位置。
截断方向要固定。
推荐保留最近行为。
如果业务强调长期兴趣，可以保留摘要 token。
摘要 token 不应与普通行为 token 混用。

### 5.8 Transformer 主干

TIGER 可以采用 encoder-only 主干预测下一项。
也可以采用 encoder-decoder 结构生成目标 SID。
encoder-only 更容易并行训练。
encoder-decoder 更清晰地隔离历史和目标。
严格自回归时需要使用 causal mask。
如果输入仅包含历史，历史内部通常允许双向注意力。
目标 SID 内部生成时必须阻止看到未来 code。

```python
class TIGERBackbone(nn.Module):
    def __init__(self, hidden_size, heads, layers, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x, padding_mask=None):
        return self.encoder(x, src_key_padding_mask=padding_mask)
```

`batch_first=True` 能减少形状转换。
不同 PyTorch 版本对默认布局的支持不同。
实现中应统一输入布局。
`padding_mask=True` 表示被忽略的位置。
mask 的语义必须与所用 API 一致。
错误的 mask 方向会造成静默训练失败。
训练前应使用一个小样本检查注意力可见性。

### 5.9 多输出头

每个 SID 层一个输出头是最直接的实现。

```python
class SIDHeads(nn.Module):
    def __init__(self, hidden_size, num_layers, codebook_size):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size)
            for _ in range(num_layers)
        ])

    def forward(self, hidden):
        # hidden: [B, K, D]
        return torch.stack(
            [head(hidden[:, i]) for i, head in enumerate(self.heads)],
            dim=1,
        )
```

输出形状是 `[B, K, V]`。
目标形状是 `[B, K]`。
可以直接 reshape 后调用交叉熵。

```python
logits = heads(hidden)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

如果不同层权重不同，可以使用加权损失。
例如粗粒度层权重更高。
但权重不能只凭直觉设置。
应先观察每层的熵和准确率。
如果后层始终无法学习，先检查 SID 质量。
不要立即增加模型容量。

### 5.10 层级条件建模

第 i 层 code 的预测可以条件化于前 i-1 层 code。
训练时把真实前缀 embedding 送入解码器。
这种方式是标准 teacher forcing。
推理时使用已经生成的预测前缀。
训练和推理之间存在 exposure bias。
可以使用 scheduled sampling。
但 scheduled sampling 会增加训练不稳定性。
最小实现先使用 teacher forcing。

```python
class PrefixDecoder(nn.Module):
    def __init__(self, codebook_size, num_layers, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(codebook_size, hidden_size)
        self.head = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size)
            for _ in range(num_layers)
        ])

    def forward(self, context, target):
        # context: [B, D]
        # target: [B, K]
        state = context
        outputs = []
        for i in range(target.size(1)):
            outputs.append(self.head[i](state))
            state = state + self.emb(target[:, i])
        return torch.stack(outputs, dim=1)
```

上面的循环只在码本层 K 上循环。
K 通常远小于历史长度 T。
因此它不会成为主要瓶颈。
如果 K 很大，可以改成并行 causal decoder。

### 5.11 目标位置的表示

历史编码得到最后一个用户状态。
目标 SID 的每个位置需要独立的位置表示。
可以将用户状态复制 K 次。
再加上目标层 embedding。

```python
class TargetQueries(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.query = nn.Embedding(num_layers, hidden_size)

    def forward(self, user_state):
        batch = user_state.size(0)
        ids = torch.arange(
            self.query.num_embeddings,
            device=user_state.device,
        )
        return user_state[:, None, :] + self.query(ids)[None, :, :]
```

这种 query 结构适合并行预测 K 个 code。
如果要让第 i 个位置读取已生成前缀，需要额外的 causal attention。
并行预测速度快。
自回归预测表达能力更强。
选择取决于 SID 层间依赖强度。

### 5.12 训练样本构造

一个训练样本由历史和下一个正样本组成。
历史必须严格早于目标发生时间。
同一时间戳的行为需要稳定排序。
如果无法排序，应丢弃有歧义的样本。
样本构造可以使用滑动窗口。
窗口左侧是历史。
窗口右侧第一个物品是目标。
一个长序列可以产生多个训练样本。
但相邻样本高度相关。
需要监控有效样本数而不是原始行数。

```python
class NextItemDataset(torch.utils.data.Dataset):
    def __init__(self, sessions, max_history):
        self.samples = []
        for session in sessions:
            for end in range(1, len(session)):
                start = max(0, end - max_history)
                self.samples.append((session[start:end], session[end]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
```

真实项目中还需要实现 batch collate。
collate 要对历史长度进行 padding。
目标 SID 通常不需要 padding。
用户没有足够历史时可以保留短样本。
也可以设置最小历史长度。
最小长度不应过大，否则冷启动样本被删除。

### 5.13 padding 与 attention mask

历史长度不同是推荐训练的常态。
右 padding 便于使用最后一个有效位置。
左 padding 便于保留最近行为的相对位置。
两者都可以工作。
但取最后状态时必须知道有效长度。
不能直接使用 `x[:, -1]` 处理右 padding。

```python
def last_valid(hidden, lengths):
    index = (lengths - 1).clamp_min(0)
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, index]
```

如果每个物品展开为 K 个 token，长度为 `T*K`。
padding mask 也必须展开到 token 级别。
物品级 mask 不能直接传给 token 级 Transformer。

```python
def expand_history_mask(item_mask, num_layers):
    # item_mask: [B, T], True means padding
    return item_mask.unsqueeze(-1).expand(-1, -1, num_layers).reshape(
        item_mask.size(0), -1
    )
```

### 5.14 训练损失与监控

总损失是各个码本层损失的加权和。
每一层都要单独记录 loss。
每一层都要单独记录 top-1 accuracy。
每一层都要单独记录 top-k accuracy。
完整 SID exact match 也要记录。
prefix match 长度是重要指标。

```python
def sid_loss(logits, target, weights=None):
    # logits: [B, K, V]
    losses = []
    for i in range(logits.size(1)):
        value = F.cross_entropy(logits[:, i], target[:, i])
        losses.append(value)
    values = torch.stack(losses)
    if weights is None:
        return values.mean(), values
    return (values * weights).sum(), values
```

如果第一层 loss 很低而最后一层 loss 很高，可能是正常的层级现象。
如果所有层 loss 都不下降，应检查输入和目标是否错位。
如果训练准确率很高而验证准确率很低，应检查时间泄漏。
如果 exact match 很低但 prefix match 很高，可能是后层码本过细。
如果 prefix match 也低，应优先检查 SID 生成和 tokenization。

### 5.15 负样本与生成式目标

经典分类推荐经常显式采样负样本。
TIGER 的输出空间是 code 空间。
交叉熵天然对其他 code 形成隐式负样本。
但它并不保证物品级负样本充分。
多个物品共享前缀时，后续 code 的区分更重要。
可以增加对比学习辅助目标。
也可以加入 batch 内物品区分损失。
辅助损失必须避免破坏主任务的概率校准。

```python
def cosine_contrastive(a, b, temperature=0.1):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)
```

这段损失适合对齐用户状态与目标物品表示。
如果目标 item 的 SID 已经包含语义，辅助损失通常是可选项。
先做无辅助损失的基线。
只有观察到明确瓶颈时再增加目标。

### 5.16 自回归推理

推理阶段从用户历史得到 context。
第一个输出头生成 `c1`。
第二个输出头根据前缀生成 `c2`。
依次生成直到 `cK`。
然后使用反向索引查找 item。
如果找不到 item，生成结果需要被过滤。

```python
@torch.no_grad()
def generate_sid(model, context, num_layers):
    prefix = []
    for layer_id in range(num_layers):
        logits = model.next_code(context, prefix, layer_id)
        code = logits.argmax(dim=-1)
        prefix.append(code)
    return torch.stack(prefix, dim=1)
```

贪心解码速度快。
贪心解码不一定得到全局最优 SID。
beam search 可以保留多个前缀。
beam width 越大，延迟和显存越高。
业务实时链路通常先使用小 beam。

### 5.17 prefix beam search

beam 的状态包含已生成 prefix 和累计 log probability。
每一层从每个 beam 扩展候选 code。
扩展后保留 top `beam_size` 个状态。
如果候选前缀在 SID trie 中不存在，可以立即剪枝。
如果某个 code 的概率极低，也可以做阈值剪枝。

```python
def topk_beam(beams, logits, beam_size):
    candidates = []
    for prefix, score in beams:
        values, indices = logits.topk(beam_size)
        for value, index in zip(values.tolist(), indices.tolist()):
            candidates.append((prefix + [index], score + value))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:beam_size]
```

真实实现要处理 batch 维度。
真实实现要使用 log softmax 而不是原始 logits。
真实实现要加入重复 item 过滤。
真实实现还要支持业务规则过滤。

### 5.18 SID trie

SID trie 是所有有效 SID 的前缀树。
根节点表示空前缀。
第一级节点表示合法的第一层 code。
第二级节点表示合法的前两层组合。
叶子节点映射到一个或多个 item。

```python
class SIDTrie:
    def __init__(self):
        self.children = {}
        self.items = []

    def insert(self, sid, item_id):
        node = self
        for code in sid:
            node = node.children.setdefault(int(code), SIDTrie())
        node.items.append(item_id)

    def next_codes(self, prefix):
        node = self
        for code in prefix:
            if code not in node.children:
                return []
            node = node.children[code]
        return list(node.children)
```

trie 可以离线构建。
上线前需要版本化。
模型和 trie 必须使用同一份 SID 字典。
版本不一致会导致大量生成结果无法解码。

### 5.19 约束解码的正确性

约束解码不能只限制单个 code 的范围。
单个 code 合法不等于组合合法。
应该在当前 prefix 下查询合法后继。
对非法后继设置负无穷 logit。
至少要保留一个合法候选。
如果当前前缀没有后继，说明 trie 或模型版本有问题。
不要静默放开全部词表。
可以记录 fallback 次数。
fallback 次数突然上升通常是数据发布问题。

### 5.20 生成候选后的去重

同一个用户可能生成重复 item。
需要在候选列表阶段去重。
去重应优先保留分数最高的候选。
不能根据 SID 字符串简单去重后忽略 item alias。
如果一个 SID 对应多个 item，需要再使用库存和业务状态过滤。
过滤顺序会影响最终 top-k。
建议先做合法性检查。
再做 item 解析。
再做可售状态过滤。
最后做去重和多样性重排。

### 5.21 与传统召回的接口

生成式模型不一定替代全部召回。
可以把 TIGER 作为一路召回源。
生成的 SID 映射为候选 item。
再与向量召回、协同召回和规则召回合并。
这样可以降低单模型失效风险。
OneRec 的端到端目标则更激进。
架构选择需要结合业务约束。
如果物品库经常变化，纯生成的维护成本更高。
如果物品语义稳定，纯生成更有机会收益。

### 5.22 复杂度估算

历史 token 数约为 `T*K`。
标准自注意力的计算复杂度约为 `O((T*K)^2*D)`。
K 从 4 增加到 8 时，注意力成本不只是翻倍。
最大历史长度需要结合延迟预算选择。
可以使用分层注意力降低复杂度。
也可以先对每个 item 内的 code 做局部融合。
然后在 item 级进行全局注意力。

```text
item 内融合: O(T * K^2)
item 间注意力: O(T^2)
总成本: O(T*K^2 + T^2)
```

局部融合牺牲部分 token 级交互。
它通常换来更好的延迟和显存。

### 5.23 训练稳定性

SID code 的频次可能高度不均衡。
高频 code 会主导交叉熵。
可以统计每层 code 的频次分布。
极低频 code 可能导致梯度稀疏。
可以在 SID 学习阶段加入频次平衡。
也可以对训练样本做温和重采样。
不要让重采样改变线上真实分布过多。

梯度裁剪可以避免偶发爆炸。
学习率 warmup 可以帮助 Transformer 稳定启动。
LayerNorm 的位置会影响训练行为。
预归一化通常更容易训练深层网络。
混合精度要检查 logits 是否出现 NaN。

### 5.24 训练伪代码

```python
model = TIGERModel(
    codebook_size=256,
    num_sid_layers=5,
    hidden_size=256,
    transformer_layers=6,
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2,
)

for batch in loader:
    history = batch["history_sid"].to(device)
    target = batch["target_sid"].to(device)
    lengths = batch["lengths"].to(device)
    logits = model(history, lengths)
    loss, per_layer = sid_loss(logits, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

训练循环中应该定期保存 checkpoint。
checkpoint 需要包含模型参数。
checkpoint 需要包含优化器状态。
checkpoint 需要包含 SID 版本。
checkpoint 需要包含训练数据切分版本。
恢复训练时不能只加载模型权重而忘记词表。

### 5.25 线上服务结构

线上服务先接收用户 ID 和上下文行为。
特征服务返回最近历史。
SID 服务把 item ID 转成 SID。
TIGER 服务编码历史并生成候选。
解码服务把 SID 映射回 item。
过滤服务删除下架和不可售商品。
重排服务合并其他候选并打分。
最终服务返回推荐列表。

每一步都需要可观测性。
需要记录模型版本。
需要记录 SID 词表版本。
需要记录生成候选数。
需要记录合法 SID 比例。
需要记录解析成功率。
需要记录过滤后剩余数。
需要记录 P50、P95 和 P99 延迟。

### 5.26 失败模式

失败模式一是 SID 冲突。
表现为不同 item 解码到同一 SID。
修复方向是重新训练或增加码本容量。

失败模式二是历史目标泄漏。
表现为离线指标异常高。
修复方向是按时间切分并检查窗口。

失败模式三是词表版本不一致。
表现为线上无法解析新生成的 SID。
修复方向是发布模型和 trie 的原子版本。

失败模式四是后缀坍缩。
表现为多个物品共享相同后层 code。
修复方向是检查层级编码器和重建损失。

失败模式五是生成热门物品过度集中。
表现为覆盖率下降。
修复方向是加入曝光校正和多样性约束。

### 5.27 本节小结

TIGER 把推荐中的 item 预测改写成 SID 序列生成。
核心组件包括 SID embedding、Transformer 主干和多层输出头。
训练时可以使用 K 个交叉熵。
推理时需要考虑 beam search、trie 和业务过滤。
正确的 mask、版本和解码字典与模型结构同样重要。
工程上应先建立可解释的 baseline。
然后再逐步增加层级建模和约束解码。

---

## 6. OneRec 改进点

### 6.0 OneRec 的定位

OneRec 代表更端到端的生成式推荐路线。
它试图减少传统推荐链路中的多阶段接口。
输入是用户和上下文。
输出是一个或多个推荐结果。
中间可以生成语义 ID、兴趣 token 或候选序列。
与普通语言模型一样，OneRec 关注序列概率。
与普通语言模型不同，推荐输出必须满足业务约束。
因此，OneRec 既是生成模型也是决策系统。

### 6.1 为什么需要 OneRec

传统链路通常包括召回、粗排、精排和重排。
每个阶段拥有独立目标。
阶段之间通过候选列表传递信息。
前一阶段漏掉的 item 无法被后一阶段恢复。
多个模型之间还可能出现目标不一致。
召回优化覆盖率。
排序优化点击率或转化率。
重排优化多样性和业务规则。
端到端模型可以共同学习最终目标。
端到端模型也可以利用更长的上下文。

但端到端不等于无成本。
模型需要承担候选空间建模。
模型还需要处理库存和实时策略。
OneRec 的价值取决于整体系统而非单个离线指标。

### 6.2 OneRec 与 TIGER 的对比维度

TIGER 通常强调基于 SID 的生成式检索。
OneRec 更强调端到端推荐路径。
TIGER 可以作为召回模块接入多阶段系统。
OneRec 更可能直接输出最终列表。
TIGER 的输出往往是单个目标 SID。
OneRec 可以生成多个推荐结果。
TIGER 的训练目标通常是 next item。
OneRec 可以联合用户反馈和排序目标。
TIGER 的解码约束相对清晰。
OneRec 需要更完整的业务约束层。

| 维度 | TIGER | OneRec |
| --- | --- | --- |
| 主要目标 | 生成语义 ID | 端到端推荐 |
| 系统位置 | 生成式召回或推荐 | 更接近全链路 |
| 输出形式 | 单个 SID 或候选 | 推荐列表或序列 |
| 训练重点 | next-item likelihood | 统一推荐目标 |
| 约束需求 | SID 合法性 | SID、库存、策略和去重 |
| 落地难点 | 解码和覆盖率 | 延迟、可控性和评估 |

### 6.3 多物品生成

单物品生成适合 next-item 训练。
线上推荐通常需要 top-N 列表。
重复运行单物品生成会增加计算成本。
OneRec 可以直接生成多个物品的 SID 序列。
序列中每个物品之间需要边界 token。
模型可以学习列表内的互补关系。
模型也可以学习多样性。
但列表内生成会引入位置偏置。
第一个结果往往获得更强概率。
训练数据中的展示顺序会影响模型。

```text
<BOS>
SID(item_1) <ITEM_SEP>
SID(item_2) <ITEM_SEP>
SID(item_3) <ITEM_SEP>
<EOS>
```

如果目标是曝光列表，应使用曝光顺序。
如果目标是用户点击列表，应谨慎处理展示偏差。
如果目标是购买结果，可以使用购买时间排序。
不同目标不能无说明地混在一起。

### 6.4 列表级建模

列表级生成可以显式建模互补性。
例如用户喜欢手机后，列表中可以推荐保护壳。
单物品 next-item 模型不一定学习到这种列表结构。
列表模型可以对已生成结果做条件化。
生成第二个 item 时读取第一个 item 的表示。
生成第三个 item 时读取前两个 item 的表示。

列表级损失仍然可以是 token 级交叉熵。
也可以加入列表级覆盖率或多样性奖励。
奖励建模需要稳定的离线反馈。
没有反馈时不应直接优化代理指标。

### 6.5 上下文输入

OneRec 可以接收更多上下文。
上下文包括用户历史。
上下文包括时间和地域。
上下文包括设备和场景。
上下文包括当前页面。
上下文包括搜索词。
上下文包括业务活动。
上下文包括库存和价格状态。

不同上下文可以使用不同 embedding。
类别特征使用 embedding。
连续特征可以使用归一化后线性层。
文本查询可以使用独立编码器。
图像和多模态特征需要额外模块。
初版不必一次接入所有模态。
应先验证最有价值的上下文。

### 6.6 用户表示

用户表示可以来自行为序列编码器。
也可以来自长期画像向量。
还可以由短期兴趣和长期兴趣拼接而成。
短期兴趣关注最近行为。
长期兴趣关注稳定偏好。
两者可以使用不同时间窗口。

```python
class UserContext(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.short_pool = nn.Linear(hidden_size, hidden_size)
        self.long_pool = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, short_state, long_state):
        short_state = self.short_pool(short_state)
        long_state = self.long_pool(long_state)
        gate = torch.sigmoid(self.gate(torch.cat([short_state, long_state], dim=-1)))
        return gate * short_state + (1 - gate) * long_state
```

门控不是必须组件。
简单平均是更容易解释的 baseline。
如果门控没有带来收益，可以删除。

### 6.7 OneRec 的训练目标

训练目标可以分为三类。
第一类是行为预测损失。
第二类是列表排序损失。
第三类是业务和约束辅助损失。

```text
L = λ1 L_token + λ2 L_rank + λ3 L_diversity + λ4 L_constraint
```

`L_token` 保证序列生成能力。
`L_rank` 保证相关性排序。
`L_diversity` 控制列表的相似度。
`L_constraint` 可以惩罚非法输出。
初版建议只使用 `L_token`。
等 token baseline 稳定后再逐项加入辅助目标。
每加一个目标都要记录单独指标。
不能只看总 loss。

### 6.8 排序损失

如果模型输出候选分数，可以使用 pairwise loss。
正样本分数应高于负样本分数。

```python
def pairwise_loss(pos_score, neg_score, margin=0.0):
    return F.relu(margin - pos_score + neg_score).mean()
```

也可以使用 softmax listwise loss。

```python
def listwise_loss(scores, positive_index):
    return F.cross_entropy(scores, positive_index)
```

如果模型只输出生成概率，需要先定义 item 级概率。
一个 SID 的概率可以是所有 code 条件概率的乘积。
在数值上应使用 log probability 求和。

```text
log P(SID) = Σ_i log P(ci | c_<i, context)
```

不能直接乘大量小概率。
直接乘法容易下溢。

### 6.9 生成概率与排序分数

生成概率体现模型对序列的信心。
排序分数还可能需要融合业务特征。
价格、库存和新鲜度不应强行编码成 SID。
可以在解码后做轻量重排。
也可以将业务特征作为生成条件。
两种方案都需要对照实验。

如果生成模型直接输出最终顺序，分数解释更困难。
如果输出候选后再重排，可控性更强。
端到端目标与可控性之间需要权衡。

### 6.10 冷启动改进

纯 SID 生成依赖物品已经拥有 SID。
新 item 没有足够交互时，SID 仍可以由内容生成。
内容特征包括标题、类目、品牌、图像和文本。
内容编码器输出固定维度向量。
量化器将向量映射到 code。

```text
new item text
    ↓
content encoder
    ↓
continuous embedding
    ↓
residual quantizer
    ↓
SID
```

冷启动 item 仍需要进入反向索引。
内容编码器和主模型的版本必须兼容。
新 item 的 SID 质量要单独评估。
不能用成熟商品的指标代表冷启动质量。

### 6.11 长尾 item

热门 item 的曝光次数更多。
模型容易学习成热门推荐器。
长尾 item 的 token 监督较少。
可以使用分层采样增加长尾样本。
也可以使用频次平滑损失。
还可以在解码时增加曝光校正。

```text
adjusted_score(item) = model_score(item) - α log(popularity(item) + 1)
```

曝光校正的 α 必须通过验证集选择。
不能把热度惩罚设置过大。
过大时相关性会明显下降。

### 6.12 用户历史去偏

历史行为不是用户偏好的直接观测。
它受到曝光位置影响。
它受到推荐策略影响。
它受到库存状态影响。
它受到价格和活动影响。
OneRec 训练时如果直接拟合点击，可能复现历史偏差。
需要保存曝光但未点击的数据。
需要区分未曝光与曝光未点击。
两者不能都标记为负样本。

### 6.13 端到端不代表端到端数据

如果数据仍来自旧推荐链路，模型会继承旧链路偏差。
需要记录候选来源。
需要记录每个阶段是否展示。
需要记录用户真正看到的列表。
需要记录位置和曝光时长。
这些日志决定了 OneRec 的上限。
模型架构不能补回缺失的日志。

### 6.14 多目标推荐

点击、停留、加购和购买是不同目标。
可以把行为类型作为事件 token。
也可以使用多任务输出头。

```python
class MultiObjectiveHeads(nn.Module):
    def __init__(self, hidden_size, codebook_size, num_layers):
        super().__init__()
        self.sid = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size)
            for _ in range(num_layers)
        ])
        self.click = nn.Linear(hidden_size, 1)
        self.purchase = nn.Linear(hidden_size, 1)
```

多目标损失需要归一化。
不同任务的损失尺度可能不同。
建议记录每个任务的梯度范数。
如果一个目标完全压制其他目标，需要重新设权重。

### 6.15 约束层的设计

生成模型应该负责相关性。
约束层负责硬业务规则。
硬规则包括下架过滤。
硬规则包括库存过滤。
硬规则包括地域和年龄限制。
硬规则包括用户黑名单。
硬规则包括合规要求。

软规则可以进入重排。
软规则包括多样性。
软规则包括新鲜度。
软规则包括商业价值。
硬规则不能用软分数替代。

### 6.16 约束解码接口

约束解码器应该接收模型 logits。
它根据当前状态返回允许 token 集合。
它不应该直接修改模型参数。
这样便于离线复现。

```python
class Constraint:
    def allowed(self, prefix, context):
        raise NotImplementedError


class ValidSIDConstraint(Constraint):
    def __init__(self, trie):
        self.trie = trie

    def allowed(self, prefix, context):
        return self.trie.next_codes(prefix)
```

实际项目可以用普通函数而不是抽象类。
这里的接口仅用于说明边界。

### 6.17 生成多个候选

可以使用 batch beam search。
每个用户维护多个 beam。
beam 状态需要包含 token、score 和约束状态。
约束状态最好增量更新。
不应每次从根节点重新遍历 prefix。

```text
每个解码步:
  1. 读取 beam hidden state
  2. 计算下一 token logits
  3. 根据 trie 屏蔽非法 token
  4. 展开候选
  5. 保留 top beam
  6. 更新约束状态
```

如果最终输出只需要 10 个 item，beam 不一定需要 100。
beam 太小会造成探索不足。
beam 太大则增加延迟和重复。

### 6.18 多样性策略

生成列表可能包含同类 item。
可以在列表级加入类别覆盖约束。
也可以在重排阶段使用 MMR。

```text
MMR = λ relevance - (1 - λ) max_similarity_to_selected
```

多样性不能完全牺牲相关性。
业务应定义最小相关性阈值。
先过滤不相关候选。
再在剩余候选中做多样性选择。

### 6.19 去重策略

同一 item 的多个 SID alias 必须归一到 item ID。
同一品牌的数量可以设置上限。
同一类目的数量可以设置软上限。
重复 item 的概率可以合并。
列表输出时只保留最高分版本。
去重后不足 top-N 时可以回退到其他召回源。

### 6.20 延迟优化

模型推理延迟由编码和解码共同决定。
历史编码可以缓存。
短时间内重复请求可以复用用户状态。
生成每个 code 的 KV cache 可以复用。
批量请求可以提高 GPU 利用率。

```text
延迟 = 特征读取 + 历史编码 + token 解码 + SID 解析 + 过滤 + 重排
```

优化时不要只测模型 forward。
必须测完整服务链路。
P99 延迟比平均延迟更能反映线上风险。

### 6.21 容错与回退

生成服务超时需要有回退策略。
回退可以使用热门推荐。
也可以使用传统召回结果。
回退策略必须可观测。

```text
if generation_timeout:
    use_cached_candidates
elif decode_failure_rate > threshold:
    use_retrieval_candidates
else:
    use_generated_candidates
```

回退不是失败处理的替代品。
需要继续定位超时和解析失败原因。

### 6.22 OneRec 的可解释性

SID 的前缀可以提供粗粒度解释。
例如模型先生成某个主题 code。
随后生成品牌和属性 code。
可以把这些 code 映射成可读标签。

但 code 不是天然的人类语义。
必须通过统计或内容聚类解释。
不能把 code 编号直接当作业务标签。

### 6.23 训练与发布版本

模型发布需要绑定 SID 字典。
绑定 tokenizer 版本。
绑定特征 schema 版本。
绑定过滤规则版本。
绑定候选后处理版本。

发布单元最好是一个不可变 manifest。
manifest 记录训练数据截止时间。
manifest 记录码本大小和层数。
manifest 记录最大历史长度。
manifest 记录评估指标。

### 6.24 OneRec 适用边界

适合物品量稳定且内容结构清晰的业务。
适合希望统一多阶段目标的场景。
适合有高质量曝光日志的团队。
不适合完全没有可解释回退的关键链路。
不适合物品每日大规模变化而没有增量编码能力的场景。
不适合延迟预算极紧且没有硬件资源的场景。

### 6.25 迁移路径

第一阶段先把 TIGER 作为生成式召回。
第二阶段加入 SID trie 和约束解码。
第三阶段让生成结果参与精排。
第四阶段尝试列表级生成。
第五阶段再评估是否移除部分传统阶段。
每一步都需要保留可回退版本。
不要一次切换全链路。

### 6.26 OneRec 小结

OneRec 的关键不是把更多模块堆进 Transformer。
关键是统一目标、统一表示和统一生成接口。
端到端模型仍然需要数据治理。
仍然需要业务约束。
仍然需要回退策略。
工程收益必须用完整链路指标验证。

---

## 7. 训练流程

### 7.0 训练流程总览

一个可复现的 SID 生成式推荐训练流程包含数据、SID、样本、模型、评估和发布六个环节。
每个环节都需要版本号。
任何一个环节变化都可能改变最终指标。

```text
原始行为日志
  ↓
时间清洗与去重
  ↓
物品特征快照
  ↓
SID 编码器训练
  ↓
物品 SID 字典
  ↓
用户序列样本
  ↓
TIGER / OneRec 训练
  ↓
离线评估
  ↓
约束解码评估
  ↓
灰度发布
```

### 7.1 数据时间边界

训练数据必须有明确截止时间。
验证数据应位于训练之后。
测试数据应位于验证之后。
不能随机打散时间序列后再切分。
随机切分会把未来行为泄漏到训练集。

```text
train: [t0, t1)
valid: [t1, t2)
test:  [t2, t3)
```

用户级切分和时间级切分需要分别说明。
冷启动评估可以额外按物品首次出现时间切分。

### 7.2 原始日志字段

最小字段包括 user_id。
最小字段包括 item_id。
最小字段包括 event_type。
最小字段包括 event_time。
推荐保存 request_id。
推荐保存 page_id。
推荐保存 position。
推荐保存 exposed 标记。
推荐保存业务状态快照。

如果缺少 request_id，曝光归因可能不准确。
如果缺少 position，位置偏差难以建模。
如果缺少业务状态，离线可售过滤无法复现。

### 7.3 日志清洗

先删除缺失 user_id 的行为。
再删除缺失 item_id 的行为。
再检查 event_time 是否可解析。
统一时区。
删除明显超出业务范围的时间。
按 request_id 和 item_id 去重。
同一用户的事件按时间排序。
时间相同的事件按日志到达顺序排序。
无法确定顺序时使用稳定 tie breaker。

```python
def clean_events(events):
    events = events.dropna(subset=["user_id", "item_id", "event_time"])
    events = events.sort_values(["user_id", "event_time", "request_id"])
    events = events.drop_duplicates(
        subset=["user_id", "item_id", "event_time", "event_type"],
    )
    return events
```

### 7.4 行为权重

点击、收藏、加购和购买的意图强度不同。
可以为行为类型设置权重。
权重可以用于样本过滤。
也可以用于损失加权。

```text
view: 1
click: 2
favorite: 3
cart: 4
purchase: 5
```

示例权重不是通用最优值。
需要通过验证集和业务目标选择。
不能把购买行为的数量少误认为信息价值低。

### 7.5 用户序列截断

历史过长会带来计算成本。
常见策略是保留最近 N 个行为。
也可以按行为类型保留重要事件。
例如保留最近 100 个行为和最近 20 个购买。
混合策略会改变位置分布。
必须在训练和线上保持一致。

```python
def truncate(history, max_len):
    return history[-max_len:]
```

简单保留后缀是稳健 baseline。
只有明确需要时再增加复杂采样。

### 7.6 SID 编码器输入

SID 编码器通常读取 item 的多模态特征。
类别特征需要稳定字典。
文本特征需要统一 tokenizer。
图像特征需要固定 backbone 版本。
数值特征要进行归一化。
缺失值需要独立标记。

```text
item_id
  ├── title tokens
  ├── category ids
  ├── brand id
  ├── image embedding
  └── numeric attributes
```

SID 训练数据不能使用未来信息。
例如不能把未来销量作为当前物品编码特征。
价格也需要使用时间快照。

### 7.7 码本数量选择

K 越大，组合空间越大。
K 越大，序列长度也越长。
K 太小会增加 SID 冲突。
K 太大会增加解码延迟。

```text
可用组合空间 = V^K
目标是有效物品数 << V^K
```

设计时需要测实际冲突率。
不能只看理论组合空间。

### 7.8 codebook size 选择

V 影响每层分类难度。
V 太小，早期层不能分开物品。
V 太大，低频 code 增多。
可以先从 128、256、512 做小规模实验。
保持总参数量可比较。

### 7.9 SID 质量指标

重建准确率衡量物品是否能还原。
冲突率衡量不同物品是否碰撞。
覆盖率衡量有多少物品拥有 SID。
平均公共前缀长度衡量层级结构。
码本利用率衡量是否出现坍缩。

```text
collision_rate = collided_items / total_items
coverage = encoded_items / eligible_items
utilization_i = used_codes_i / codebook_size_i
```

这些指标要按物品类别分组。
全局平均可能掩盖长尾类别问题。

### 7.10 SID 字典构建

编码完成后建立 `item_id -> sid` 映射。
同时建立 `sid -> item_id` 反向映射。
反向映射需要检测一对多。
还要记录生效时间。

```python
sid_to_items = {}
for item_id, sid in item_to_sid.items():
    sid_to_items.setdefault(tuple(sid), []).append(item_id)
```

一对多映射不能直接覆盖。
覆盖会导致线上结果不可解释。

### 7.11 生成训练样本

从每个用户序列产生 next-item 样本。
目标是历史之后的第一个有效 item。
被过滤的 item 是否跳过需要统一定义。
如果下架 item 在历史中存在，可以保留历史但不作为目标。

```python
def make_samples(sequence, max_history):
    for target_index in range(1, len(sequence)):
        left = max(0, target_index - max_history)
        yield sequence[left:target_index], sequence[target_index]
```

样本构造必须记录数据版本。

### 7.12 batch collate

collate 负责对历史 SID 做 padding。
历史中每个 item 有 K 个 code。
padding code 可以是特殊值。
embedding 的 padding_idx 应保持一致。

```python
def collate_fn(samples, pad_value=0):
    max_len = max(len(history) for history, _ in samples)
    num_layers = len(samples[0][1])
    history = torch.full(
        (len(samples), max_len, num_layers),
        pad_value,
        dtype=torch.long,
    )
    lengths = []
    targets = []
    for row, (items, target) in enumerate(samples):
        lengths.append(len(items))
        targets.append(target)
        if items:
            history[row, :len(items)] = torch.tensor(items)
    return history, torch.tensor(targets), torch.tensor(lengths)
```

如果 code 0 是合法值，padding 不能使用 0。
应为 padding 预留额外索引。

### 7.13 teacher forcing

teacher forcing 在训练时提供真实目标前缀。
它使不同位置可以并行计算。
每个目标位置都知道正确的前缀。
推理时前缀来自模型预测。
这造成训练推理差异。

```text
训练: [BOS, c1*, c2*, c3*] → 预测 [c1, c2, c3, c4]
推理: [BOS, ĉ1, ĉ2, ĉ3] → 逐步预测
```

先使用 teacher forcing 取得稳定 baseline。

### 7.14 causal mask

生成目标 SID 时第 i 个位置不能读取 i+1 之后的目标。
causal mask 的上三角通常设为不可见。
不同框架对 bool mask 的约定可能不同。
必须使用最小张量单元测试 mask。

```python
def causal_mask(length, device):
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )
```

如果使用 additive mask，则不可见位置通常设为负无穷。
不要同时混用两种语义。

### 7.15 训练循环详解

训练循环的顺序应保持固定。
先读取 batch。
再搬到设备。
再构造 mask。
再 forward。
再计算每层损失。
再反向传播。
再裁剪梯度。
再更新参数。
再更新 scheduler。
最后记录指标。

```python
for step, batch in enumerate(loader):
    history, target, lengths = [x.to(device) for x in batch]
    optimizer.zero_grad(set_to_none=True)
    logits = model(history, lengths)
    loss, layer_losses = compute_loss(logits, target)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    logger.log(loss=loss.item(), grad_norm=float(grad_norm))
```

如果使用梯度累积，应在指定步数后再 optimizer.step。
梯度累积时要正确缩放 loss。

### 7.16 学习率策略

Transformer 通常需要 warmup。
warmup 步数可以按总训练步数比例设置。
学习率过大表现为 loss 震荡。
学习率过小表现为收敛缓慢。

```text
warmup_ratio = 0.01 ~ 0.05
```

这里只是实验起点。
不同 batch size 需要重新校准。

### 7.17 batch size 与 token 数

相比样本数，token 数更能反映计算量。
历史长度变化会导致每个 batch 成本不同。
可以按总 token 数动态组 batch。
简单实现则使用固定 batch size。

```text
batch_tokens ≈ batch_size * max_history * K
```

显存不足时先减少最大历史。
不要立即缩小 hidden size。
历史截断通常对模型结构影响更小。

### 7.18 混合精度

混合精度可以降低显存和提高吞吐。
需要使用 GradScaler 或框架提供的 autocast。
logits 和 loss 计算要检查数值稳定性。

```python
scaler = torch.cuda.amp.GradScaler()
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        logits = model(batch)
        loss = criterion(logits, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

在不支持 CUDA 的环境中应有普通精度回退。

### 7.19 checkpoint

checkpoint 至少保存 model state。
也要保存 optimizer state。
也要保存 scheduler state。
也要保存 global step。
也要保存随机数状态。

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "sid_version": sid_version,
}, path)
```

checkpoint 文件名应包含实验标识。
不要依赖修改时间判断最佳模型。

### 7.20 验证流程

验证时关闭 dropout。
验证时不计算梯度。
验证时使用固定解码策略。
验证时记录 token loss。
验证时记录 SID exact match。
验证时记录 item recall@K。

```python
model.eval()
with torch.no_grad():
    for batch in valid_loader:
        logits = model(batch)
        metrics.update(logits, batch.target)
```

验证数据不能被用于反复调参到过拟合。
需要保留最终测试集。

### 7.21 早停规则

可以根据验证 loss 早停。
也可以根据 item recall 早停。
如果业务目标是点击率，token loss 不是唯一依据。
记录多个指标后选择主指标。
主指标必须在实验开始前定义。

### 7.22 负采样与全量评估

训练时可能无法使用全部物品进行对比。
生成模型通过 code 分类避免了全量 item softmax。
评估时仍要把 SID 解码成 item。
候选字典不能只包含正样本。
全量合法 SID 才能计算真实 recall。

### 7.23 数据并行

数据并行需要每个 worker 读取一致的 SID 字典。
worker seed 要可复现。
shuffle seed 要记录。
分布式 sampler 在每个 epoch 要调用 set_epoch。

```python
sampler.set_epoch(epoch)
```

漏掉这一步会导致不同 epoch 的 shuffle 变化不足。

### 7.24 训练可复现

固定 Python 随机种子。
固定 NumPy 随机种子。
固定 PyTorch 随机种子。
记录 CUDA deterministic 设置。
记录依赖版本。
记录数据 hash。
记录模型超参数。

```python
def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

完全确定性可能降低吞吐。
需要在复现调试和生产训练之间权衡。

### 7.25 小数据过拟合测试

在正式训练前，使用极小数据集过拟合。
如果模型无法记住几十个样本，优先检查代码。
检查目标 shift。
检查 padding mask。
检查 causal mask。
检查损失 shape。
检查 SID 解码。

```text
成功标准:
  训练 loss 接近 0
  token accuracy 接近 100%
  exact match 明显上升
```

这是最便宜也最有效的 sanity check。

### 7.26 训练指标看板

看板应包含训练 loss。
看板应包含验证 loss。
看板应包含每层 loss。
看板应包含每层 accuracy。
看板应包含 exact match。
看板应包含 recall@1、recall@10、recall@50。
看板应包含覆盖率。
看板应包含热门度分布。
看板应包含长尾分组。
看板应包含推理延迟。

### 7.27 错误样本分析

随机抽取预测正确样本。
随机抽取预测错误样本。
抽取高频 item 错误样本。
抽取长尾 item 错误样本。
抽取冷启动 item 错误样本。
抽取 SID 冲突样本。
抽取非法解码样本。

对每类错误记录共同特征。
不要只看总体分数。

### 7.28 消融实验

消融历史长度。
消融 SID 层数。
消融层级 embedding。
消融时间 embedding。
消融 beam search。
消融 trie 约束。
消融多目标损失。
消融流行度校正。

每次只改变一个主要因素。
如果同时改变多项，无法解释收益。

### 7.29 训练阶段的资源估算

模型参数量决定权重显存。
激活量决定训练显存。
历史长度决定注意力显存。
batch token 数决定吞吐。
数据读取决定 GPU 利用率。

```text
训练显存 ≈ 参数 + 梯度 + 优化器状态 + 激活 + 临时张量
```

先测峰值显存。
再决定是否需要梯度检查点。

### 7.30 增量训练

新行为可以按天或按小时增量训练。
增量训练需要避免灾难性遗忘。
可以混入旧样本 replay buffer。
也可以冻结部分 SID 编码器。

```text
new_batch + replay_batch → incremental update
```

SID 字典变更时不能只增量更新推荐模型。
必须同步更新反向索引和 trie。

### 7.31 训练发布门槛

至少要求验证集主指标不低于 baseline。
至少要求非法 SID 比例不高于阈值。
至少要求覆盖率不明显下降。
至少要求 P99 延迟满足预算。
至少要求回退链路可用。
至少要求模型和字典版本一致。

### 7.32 训练流程小结

训练流程的关键是时间正确、SID 正确和评估正确。
先做小数据过拟合。
再做稳定 baseline。
再做系统化消融。
最后才做复杂的端到端目标。

---

## 8. 工业实践(Spotify / 淘宝)

### 8.0 工业实践的阅读方法

论文中的模型通常省略大量系统细节。
工业实践需要同时关注数据、产品和基础设施。
Spotify 与淘宝代表不同内容和交易场景。
Spotify 更强调连续消费和用户会话。
淘宝更强调商品、交易、库存和商业约束。
两者都可以借鉴 SID 与生成式推荐的共同原则。
不能直接复制某个公司的参数。
应该抽象出可迁移的工程方法。

### 8.1 Spotify 场景特点

Spotify 的 item 可以是歌曲、专辑、歌单或播客。
用户行为具有连续播放特征。
一次会话中相邻内容具有强相关性。
用户可能快速跳过某首歌。
用户也可能循环播放某首歌。
播放时长是重要反馈。
收藏、加入歌单和关注艺人是强反馈。

推荐目标不只是点击。
还包括播放开始率。
包括完整播放率。
包括跳过率。
包括会话时长。
包括用户留存。
包括探索与熟悉之间的平衡。

### 8.2 Spotify 的 SID 特征假设

音乐内容拥有稳定的语义属性。
属性包括流派、语言、节奏和艺人。
音频 embedding 可以表示声学风格。
文本 embedding 可以表示歌词和描述。
用户行为序列可以表示短期情绪和场景。
SID 可以融合内容和协同信号。

一个音乐 SID 可以抽象为：

```text
c1: 粗粒度音乐主题
c2: 流派或场景
c3: 艺人群体
c4: 声学风格
c5: 单曲实例
```

这不是固定定义。
真实码本的语义需要通过聚类和 probe 分析。

### 8.3 Spotify 风格的会话样本

```text
历史:
  song_a 完整播放
  song_b 跳过
  song_c 播放 60%
  song_d 收藏
目标:
  song_e
```

行为类型可以作为事件 embedding。
播放进度可以作为连续特征。
跳过行为不应简单等同于负反馈。
因为跳过可能来自上下文变化。

### 8.4 会话级模型

Spotify 场景通常需要实时会话表示。
用户长期画像不能完全替代当前会话。
可以使用双塔结构编码长期和短期兴趣。
再用门控合并两个状态。

```python
class SessionMixer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Linear(hidden * 2, hidden)

    def forward(self, profile, session):
        gate = torch.sigmoid(self.gate(torch.cat([profile, session], -1)))
        return gate * session + (1 - gate) * profile
```

短期状态的权重可以随会话长度变化。
新会话使用长期画像作为主要信息。
会话行为增多后逐步提高短期状态权重。

### 8.5 音乐列表生成

生成式推荐可以输出歌曲序列。
序列中的相邻歌曲需要考虑转场质量。
只优化单曲相关性可能造成列表听感不佳。
可以加入相邻 item 的转场特征。

```text
列表质量 = 单曲相关性 + 转场质量 + 多样性 + 新颖性
```

转场质量可以由历史共听数据估计。
也可以使用音频相似度估计。
两者结合通常更稳健。

### 8.6 Spotify 风格的约束

过滤用户明确不喜欢的内容。
过滤地区不可用内容。
过滤授权过期内容。
过滤重复播放过多的内容。
控制同一艺人的连续出现次数。
控制列表中显式内容的比例。

硬约束应在生成期间或生成后执行。
如果约束会频繁触发，应该反馈到训练数据。

### 8.7 Spotify 风格的探索

生成式模型倾向于选择高概率 item。
高概率通常对应熟悉内容。
探索需要额外机制。
可以给低曝光 item 一个探索预算。
也可以在候选生成后增加新颖性分数。

```text
exploration_score = relevance + β * novelty
```

新颖性不能只用全局低频定义。
用户已经听过的内容也需要单独计算。

### 8.8 负反馈处理

跳过是弱负反馈。
明确 dislike 是强负反馈。
短播放可能是弱负反馈。
播放结束不一定代表强正反馈。
反馈强度应该按行为类型和上下文建模。

### 8.9 淘宝场景特点

淘宝推荐对象通常是商品。
商品包含价格、库存、店铺和促销信息。
商品生命周期变化快。
新品和长尾商品比例高。
用户行为包含浏览、点击、加购、收藏和购买。
目标同时受交易、体验和商业价值影响。

### 8.10 淘宝的 SID 特征假设

商品 SID 可以融合类目、品牌和标题语义。
视觉特征可以表示商品外观。
价格区间可以作为条件特征。
店铺和供应链信息需要谨慎使用。
会随时间变化的字段不能直接作为稳定 SID。

```text
c1: 一级类目
c2: 二级类目
c3: 商品风格
c4: 品牌或属性组合
c5: 商品实例
```

真实商品可以拥有多版本 SID。
例如不同促销状态不必改变语义 SID。
促销状态应通过上下文特征输入。

### 8.11 淘宝的商品生命周期

商品上架后可能没有交互。
内容编码器可以为新品生成初始 SID。
交互积累后可以进行 SID 更新。
SID 更新会影响历史序列解释。
因此需要控制更新频率。

一种策略是冻结稳定前缀。
只更新后层 code。
这样可以保留类目和主题的连续性。
另一种策略是为商品建立 SID 版本。
历史使用旧版本。
线上新请求使用最新版本。
两种策略都需要评估迁移成本。

### 8.12 淘宝的库存与价格

库存是强硬约束。
无库存商品不能进入最终列表。
价格变化不一定需要重新编码 SID。
价格可以作为动态输入。
促销可以作为动态输入。
优惠券状态也可以作为动态输入。

```text
稳定内容 → SID
动态交易状态 → context features
```

把动态价格写进 SID 会造成频繁字典更新。
这通常不是最小可行方案。

### 8.13 淘宝的多目标

点击目标衡量兴趣。
加购目标衡量购买意图。
购买目标衡量最终转化。
GMV 反映商业结果但可能有偏。
退款和投诉反映长期体验风险。

```text
L_total = λ_click L_click
        + λ_cart L_cart
        + λ_buy L_buy
        + λ_quality L_quality
```

权重应与业务阶段匹配。
新产品可以提高点击和探索权重。
成熟交易链路可以提高购买和质量权重。

### 8.14 召回共存

淘宝规模巨大时，单路生成通常不够。
生成式 SID 推荐可以与多路召回并存。
生成模型提供语义和序列相关性。
协同召回提供群体行为信号。
向量召回提供内容相似信号。
规则召回提供活动和运营信号。

候选合并后再统一重排。
必须标记候选来源。
来源特征有助于解释收益。

### 8.15 生成式推荐的灰度方式

先离线回放生成结果。
再进行 shadow traffic。
shadow 阶段不影响真实曝光。
比较生成候选与线上候选的覆盖差异。
再进行小流量 A/B 测试。

```text
离线回放 → shadow → 1% 灰度 → 5% → 25% → 全量
```

每个阶段都需要明确退出条件。
如果非法率或延迟超阈值，自动回退。

### 8.16 工业指标

Spotify 类场景关注播放开始率。
关注跳过率。
关注完整播放率。
关注会话时长。
关注次日和七日留存。

淘宝类场景关注点击率。
关注加购率。
关注购买转化率。
关注 GMV。
关注退款率。
关注用户停留和复购。

两类业务都需要覆盖率。
都需要多样性。
都需要长尾曝光。
都需要 P99 延迟。

### 8.17 特征实时性

用户刚刚点击的内容应快速进入会话状态。
库存变化应快速进入过滤服务。
价格变化应快速进入重排服务。
SID 字典不一定需要同样高频更新。
应把稳定特征和动态特征拆开。

### 8.18 缓存策略

用户长期状态可以缓存。
会话状态可以短 TTL 缓存。
热门 item 的 SID 可以本地缓存。
trie 可以加载到内存。
动态库存不能使用过长 TTL。

```text
静态 SID cache: 长 TTL
用户状态 cache: 中 TTL
库存 cache: 短 TTL
```

缓存命中率和一致性都要监控。

### 8.19 失败回退

Spotify 风格可以回退到最近播放相似列表。
淘宝风格可以回退到热门且有库存列表。
回退结果也需要经过合规过滤。
回退不能绕过用户黑名单。

### 8.20 监控生成分布

监控生成的第一层 code 分布。
监控生成的完整 SID 分布。
监控 item 热度分布。
监控类别覆盖。
监控新鲜度。
监控解析失败。
监控过滤比例。

如果第一层分布突然坍缩，可能是模型退化。
如果解析失败上升，可能是字典版本问题。
如果过滤比例上升，可能是库存或策略变化。

### 8.21 线上日志样例

```json
{
  "request_id": "r-001",
  "model_version": "tiger-2026-07",
  "sid_version": "sid-42",
  "history_length": 30,
  "beam_size": 4,
  "generated": 40,
  "decoded": 36,
  "filtered": 8,
  "returned": 20,
  "latency_ms": 18.4
}
```

日志中不应包含不必要的个人隐私。
用户标识应使用不可逆或受控的内部标识。

### 8.22 业务规则与模型解耦

业务规则变化不应要求立即重训模型。
因此硬过滤和轻量重排应独立部署。
模型只负责提供基础候选和分数。
规则层负责当前策略。
但规则层的强干预也会改变训练分布。
需要定期把线上规则效果反馈到训练。

### 8.23 运营干预

活动商品可以作为额外条件输入。
运营白名单可以作为硬约束或候选注入。
不能让运营配置破坏 SID 合法性。
运营候选也应使用统一 item 解析流程。

### 8.24 新品策略

新品可以使用内容 SID。
新品也可以从相似成熟商品继承候选先验。
新品初期要增加探索曝光。
新品需要单独监控点击和购买。
不能只看全量平均收益。

### 8.25 长尾策略

长尾 item 的 SID 可能有较强语义。
生成模型可以通过前缀泛化发现长尾。
但流行度先验仍可能压制长尾。
可以在候选后处理阶段设置长尾配额。
配额必须有相关性保护。

### 8.26 跨域推荐

用户在音乐场景的偏好和商品场景不同。
SID 表示可以跨域共享部分语义机制。
不应直接共享所有 code embedding。
跨域共享需要域 embedding。

```text
representation = sid_embedding + domain_embedding
```

跨域实验需要防止一个域主导训练。

### 8.27 质量与安全

音乐内容要遵守版权和年龄规则。
商品内容要遵守合规和质量规则。
生成模型不能绕过内容安全服务。
安全过滤应在最终曝光前执行。

### 8.28 组织协作

算法团队维护模型和指标。
数据团队维护日志和样本。
平台团队维护推理服务。
业务团队维护规则和活动。
质量团队维护安全和合规。

接口要写清楚输入、输出和版本。
跨团队问题通常首先表现为版本不一致。

### 8.29 Spotify / 淘宝可迁移经验

稳定语义用 SID 表示。
动态状态用 context 表示。
用户短期行为和长期偏好分开建模。
生成与硬约束解耦。
离线指标和线上指标同时看。
灰度发布必须可回退。
日志必须支持结果追溯。

### 8.30 工业实践小结

Spotify 说明连续会话和列表体验的重要性。
淘宝说明交易约束、商品生命周期和多目标的重要性。
两者共同说明生成式推荐不是单一模型替换。
它需要表示、数据、约束、服务和实验共同配合。

---

## 9. 评估对比

### 9.0 评估原则

生成式推荐不能只看 token accuracy。
最终用户看到的是 item 列表。
因此必须把 SID 预测还原成 item 后再评估。
同时还要评估合法性、覆盖率、多样性和延迟。

### 9.1 评估数据切分

训练集用于拟合参数。
验证集用于选择超参数。
测试集只用于最终报告。
时间切分最符合线上 next-item 场景。

```text
历史窗口来自过去
目标行为来自未来
```

测试集目标不能出现在历史中。
如果 item 在历史中已出现，需要明确是否允许重复推荐。

### 9.2 Token-level 指标

每层 token accuracy 衡量 code 预测正确率。
Top-k token accuracy 衡量候选 code 是否包含正确值。
Token cross entropy 衡量概率质量。
Perplexity 是交叉熵的指数变换。

```text
perplexity = exp(cross_entropy)
```

token 指标适合调试模型。
不适合单独代表推荐效果。

### 9.3 Prefix 指标

prefix@1 衡量前缀正确率。
prefix@k 衡量前缀候选覆盖率。
平均公共前缀长度反映层级预测质量。

```text
prefix_len = max { l : pred[:l] == target[:l] }
```

如果第一层 prefix 很准，模型可能掌握类别兴趣。
如果最后层差，可能是实例区分不足。

### 9.4 SID exact match

exact match 要求 K 个 code 全部正确。
如果 K 增大，exact match 会自然下降。
因此比较不同 K 时要同时报告 prefix 指标。
exact match 只有在 SID 唯一时才等价于 item 命中。

### 9.5 item-level Recall

把预测 SID 解码成 item 后计算 recall@K。
如果一个预测 SID 无法解码，应计入非法或未命中。
不能静默删除非法结果后再计算 top-K。
否则指标会被高估。

```text
Recall@K = 命中的目标数 / 测试目标总数
```

### 9.6 Hit Rate

Hit Rate@K 与单目标 recall 类似。
对每个用户只判断是否命中。
推荐列表场景通常需要同时报告 HR@1、HR@5、HR@10、HR@50。

### 9.7 MRR

MRR 对排名靠前的命中更敏感。

```text
MRR = mean(1 / rank_of_first_hit)
```

如果没有命中，倒数排名记为 0。
MRR 适合单目标 next-item 任务。

### 9.8 NDCG

NDCG 可以处理多相关目标。
如果只有一个目标，NDCG 与排名相关。
多目标场景可以给不同相关等级。
例如购买高于加购，加购高于点击。

```text
DCG@K = Σ_i (2^rel_i - 1) / log2(i + 1)
```

### 9.9 Coverage

coverage 衡量推荐系统触达的不同 item 数量。

```text
coverage = unique_recommended_items / total_eligible_items
```

覆盖率过低说明热门集中。
覆盖率过高也可能说明相关性不足。
因此需要与 recall 一起解释。

### 9.10 Catalog coverage

catalog coverage 按所有可推荐 item 计算。
长尾 coverage 可以单独计算。
新品 coverage 也应单独计算。

```text
long_tail_coverage = recommended_long_tail / eligible_long_tail
```

### 9.11 Diversity

列表内多样性可以用平均 pairwise distance。
相似度可以来自内容 embedding。
也可以来自类别不同程度。

```text
ILD = 1 - average(similarity(item_i, item_j))
```

ILD 越高不一定越好。
如果用户有非常聚焦的兴趣，高多样性可能降低满意度。

### 9.12 Novelty

新颖性衡量推荐 item 对用户是否陌生。
用户过去见过的 item 可以降低 novelty。
全局低频 item 不一定对用户新颖。
需要结合用户历史和曝光历史。

### 9.13 Serendipity

惊喜度同时考虑新颖性和相关性。
推荐一个完全不相关的冷门 item 不算惊喜。
可以使用用户喜欢但历史未见的 item 近似。

### 9.14 Popularity bias

统计推荐 item 的全局流行度。
比较模型与基线的平均流行度。
也可以按流行度分桶报告 recall。

```text
popularity_bucket ∈ {head, torso, tail}
```

如果生成模型只提高 head recall，收益可能不可持续。

### 9.15 Calibration

概率校准衡量预测概率和实际命中率的一致性。
生成概率较高的候选不一定真实命中率更高。
可以按 score 分桶比较预测和实际结果。

### 9.16 Diversity 与 business rules

硬过滤会改变原始模型列表。
应同时报告过滤前和过滤后指标。
过滤后指标更接近线上结果。
过滤比例本身也是质量信号。

### 9.17 解码策略对比

至少比较贪心解码。
比较 beam search。
比较 constrained beam search。
比较 sampling 或 top-p 仅用于探索实验。

```text
greedy: 低延迟，探索弱
beam: 质量稳定，延迟更高
constrained beam: 合法性高，候选可能变窄
sampling: 多样性高，稳定性较弱
```

### 9.18 Beam size 实验

固定模型和输入。
只改变 beam size。
报告 recall、coverage 和延迟。

```text
beam=1, 2, 4, 8
```

通常 beam 增大后 recall 先升后平。
如果 coverage 明显下降，可能是候选排序过度集中。

### 9.19 SID 层数实验

固定理论组合空间规模附近比较不同 K。
报告冲突率。
报告 token length。
报告 exact match。
报告延迟。

K 增大不保证推荐质量提升。
后层 code 可能只增加解码负担。

### 9.20 码本大小实验

比较 V=128、256、512 等设置。
保持隐藏维度和训练步数可比较。
报告每层 code 利用率。
报告低频 code 比例。

### 9.21 历史长度实验

比较短历史和长历史。
按用户活跃度分组。
新用户通常从长历史中受益有限。
高活跃用户可能从更长历史受益。

### 9.22 冷启动评估

按物品首次出现时间构造测试集。
区分新用户冷启动和新物品冷启动。
新用户没有行为时评估内容和热门先验。
新物品没有交互时评估内容 SID。

### 9.23 长尾评估

把 item 按训练期曝光或交互量分桶。
报告每桶 recall。
报告每桶 coverage。
报告每桶曝光变化。

### 9.24 用户分群评估

按历史长度分组。
按活跃度分组。
按地域分组。
按设备分组。
按新老用户分组。
按业务会员等级分组。

全量指标高不代表每个群体都受益。

### 9.25 离线与线上差异

离线 next-item 预测假设目标行为明确。
线上列表曝光会改变用户行为。
离线没有完整模拟位置偏差。
线上还存在库存、延迟和策略约束。
因此离线收益只能作为上线依据的一部分。

### 9.26 A/B 测试

实验需要随机分流。
实验组和对照组应使用相同用户分层。
实验周期要覆盖业务周期。
主要指标需要预注册。
不能只挑选有利指标报告。

```text
primary: 目标转化或用户价值
secondary: CTR、覆盖、留存、延迟
guardrail: 投诉、退款、过滤率、错误率
```

### 9.27 统计显著性

点击率等比例指标需要置信区间。
订单和 GMV 通常方差更大。
不能因为点估计提升就断言收益确定。
应报告样本量和置信区间。

### 9.28 延迟评估

平均延迟不足以代表线上体验。
至少报告 P50、P90、P95 和 P99。
分别测 CPU、GPU 和端到端链路。
还要测冷缓存和热缓存。

### 9.29 资源评估

记录模型参数量。
记录显存占用。
记录 QPS。
记录单请求成本。
记录扩容成本。

```text
cost_per_request = instance_cost / served_requests
```

准确率提升需要与成本提升放在一起比较。

### 9.30 基线选择

最基本基线是热门推荐。
序列任务基线可以是 GRU4Rec 或 SASRec 类模型。
传统候选基线可以是协同过滤。
内容基线可以是向量相似推荐。
多阶段线上系统是最终强基线。

不应只与弱基线比较。

### 9.31 评估表模板

| 模型 | Recall@10 | MRR | Coverage | ILD | P99(ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | - | - | - | - | - |
| Sequence baseline | - | - | - | - | - |
| TIGER greedy | - | - | - | - | - |
| TIGER beam | - | - | - | - | - |
| OneRec constrained | - | - | - | - | - |

表格中的横线不能被当成零。
未测指标应标记为缺失。

### 9.32 错误分析模板

记录用户历史。
记录目标 item。
记录目标 SID。
记录预测 SID。
记录第一个错误层。
记录是否存在合法前缀。
记录是否被业务过滤。
记录是否命中其他相关 item。
记录模型和字典版本。

### 9.33 常见指标误区

把 token accuracy 当作 item recall。
过滤非法结果后再重新凑满 top-K。
随机切分时间序列。
忽略重复推荐。
只报告 head item 指标。
忽略线上规则后的列表。
只比较平均延迟。
只跑一次实验。

### 9.34 评估实现骨架

```python
def recall_at_k(predictions, targets, k):
    hit = 0
    for pred, target in zip(predictions, targets):
        if target in pred[:k]:
            hit += 1
    return hit / max(len(targets), 1)


def mean_reciprocal_rank(predictions, targets):
    values = []
    for pred, target in zip(predictions, targets):
        if target in pred:
            values.append(1.0 / (pred.index(target) + 1))
        else:
            values.append(0.0)
    return sum(values) / max(len(values), 1)
```

真实实现需要处理 item alias 和过滤状态。

### 9.35 评估流水线

先加载固定测试集。
再加载对应 SID 字典。
再运行模型生成。
再执行约束解码。
再执行业务过滤。
再计算 token 指标。
再计算 item 指标。
再计算分布指标。
再输出延迟和资源。

### 9.36 评估报告内容

报告实验配置。
报告数据时间范围。
报告 SID 版本。
报告模型参数。
报告解码策略。
报告过滤规则。
报告主指标。
报告保护指标。
报告分群指标。
报告失败样本。

### 9.37 结论判定

如果 token 指标升高而 item recall 不变，检查 SID 冲突。
如果 recall 升高而 coverage 大幅下降，检查热门偏置。
如果离线升高而线上下降，检查分布和日志偏差。
如果质量升高而 P99 超标，检查解码和缓存。
如果所有指标小幅变化，可能收益低于噪声。

### 9.38 评估小结

评估必须从 token 走到 item，再走到用户和系统。
正确性、覆盖率、多样性、延迟和成本需要同时看。
最强的结果不是单一指标最高。
最强的结果是在目标约束下整体可用。

---

## 10. 面试题

### 10.0 使用方式

面试回答建议先给结论。
再解释表示、模型、训练和推理。
最后补充工程约束和失败模式。
不要只背论文名。
要能说明为什么这样设计。

### 10.1 什么是 SID

**参考回答**：SID 是把物品映射成多个离散语义 code 的表示。
它通常写成 `(c1, c2, ..., cK)`。
与普通 item ID 不同，SID 的不同层可以承载层级语义。
生成式推荐可以把预测 item 转写为预测 SID 序列。

### 10.2 为什么不用普通 item ID

普通 item ID 主要是索引。
相邻编号没有语义关系。
物品数量很大时，item softmax 输出空间大。
SID 提供结构化的离散输出空间。
相似物品可以共享前缀。

### 10.3 SID 的潜在问题是什么

主要问题包括冲突、坍缩、版本变化和解码失败。
如果不同 item 共享完整 SID，就无法唯一还原。
如果码本利用率低，表示容量没有被使用。
如果字典版本不一致，线上解析会失败。

### 10.4 TIGER 的核心结构是什么

TIGER 读取用户历史物品的 SID。
每个 code 经过 embedding。
所有历史 code 展平为 Transformer token 序列。
Transformer 编码用户历史。
输出端为每一个 SID 层设置预测头。
训练时对各层 code 计算交叉熵。

### 10.5 TIGER 为什么要使用多个输出头

每个 SID 层拥有独立的 code 词表和语义角色。
独立输出头可以直接预测对应层。
也可以共享输出头并加入层 embedding。
选择取决于参数量和层间共享程度。

### 10.6 TIGER 的输入形状是什么

常见输入形状是 `[B, T, K]`。
B 是 batch size。
T 是历史 item 数量。
K 是 SID 层数。
embedding 后形状是 `[B, T, K, D]`。
展平后通常是 `[B, T*K, D]`。

### 10.7 为什么不能直接把 `[B,T,K]` 当普通序列

最后一维的 K 个值不是普通连续数值。
它们来自不同码本。
需要分别 lookup embedding。
如果直接把整数值送入线性层，模型会错误理解 code 间的数值距离。

### 10.8 如何加入位置编码

至少可以加入行为时间位置和码本位置。
行为位置表示第几个历史 item。
码本位置表示 item 内的第几层 code。
两类位置相加后送入 Transformer。

### 10.9 历史内部要不要 causal mask

如果只编码已经发生的历史，历史内部可以使用双向注意力。
如果训练任务严格要求自回归重建，则需要 causal mask。
目标 SID 的生成位置必须阻止读取未来 code。
关键是区分历史编码和目标解码。

### 10.10 OneRec 与 TIGER 的区别

TIGER 更常被描述为基于 SID 的生成式检索或 next-item 生成。
OneRec 更强调端到端生成式推荐。
TIGER 可以作为多阶段系统的一路召回。
OneRec 试图统一更多推荐阶段。
OneRec 仍然需要过滤、解码、回退和监控。

### 10.11 生成式推荐如何获得负样本

code 级交叉熵会把词表中的其他 code 作为竞争类别。
因此不必对每个 item 显式构造全量负样本。
但这不代表 item 级负样本完全充分。
可根据需要加入对比或排序辅助目标。

### 10.12 什么是 teacher forcing

训练时在预测第 i 个 code 时输入真实的前 i-1 个 code。
这样可以并行计算多个位置。
推理时只能输入模型已经生成的前缀。
两者差异称为 exposure bias。

### 10.13 exact match 与 item recall 的区别

exact match 要求整条 SID 完全匹配。
item recall 先把预测 SID 解码成 item 再判断命中。
如果 SID 存在冲突，两者不等价。
推荐系统最终应重点关注 item-level 指标。

### 10.14 如何处理非法 SID

构建有效 SID trie。
解码时只允许当前前缀的合法后继。
生成后再检查反向字典。
非法结果应统计并触发告警。
必要时回退到传统召回。
不能静默删除所有非法结果后再计算指标。

### 10.15 为什么需要 trie

词表中每个 code 单独合法，不代表组合合法。
trie 保存所有有效 SID 的前缀关系。
在每一步生成时可以根据当前前缀屏蔽非法 code。
这样既提高合法率，也减少无效搜索。

### 10.16 beam search 和 greedy 的区别

greedy 每一步只保留最高概率 code。
延迟低，实现简单，但可能陷入局部最优。
beam search 保留多个前缀。
质量通常更稳定，但需要更多计算。
实际选择要结合 P99 延迟预算。

### 10.17 生成列表如何去重

先把 SID 解析成 item ID。
不同 SID alias 先归一到同一个 item。
按分数保留最高版本。
再执行业务过滤和多样性规则。
候选不足时从其他召回源补齐。

### 10.18 如何避免热门 item 霸榜

训练中处理曝光偏差。
使用流行度分桶评估。
解码或重排时加入适度流行度校正。
增加长尾探索预算。
设置覆盖率和新颖性保护指标。
不能只用强惩罚压制热门，否则相关性会下降。

### 10.19 如何做冷启动

新用户可以使用热门、上下文和内容特征。
新物品可以通过内容编码器生成 SID。
将新物品加入反向字典。
单独评估新用户和新物品。
不能把全量用户的结果当作冷启动结果。

### 10.20 如何处理动态价格和库存

稳定内容属性进入 SID。
价格、库存、促销等动态字段作为上下文特征或过滤条件。
不应因为每次价格变化都重新生成 SID。
无库存商品在最终曝光前必须过滤。

### 10.21 如何训练 next-item 样本

按用户和时间排序行为。
使用历史窗口预测时间上更晚的下一个 item。
严格按时间切分训练、验证和测试。
相同时间戳需要稳定排序。
避免把未来行为放入历史。

### 10.22 如何验证数据没有泄漏

检查目标时间是否晚于历史最大时间。
检查 item 特征是否来自目标之后。
检查 SID 编码器是否使用未来统计。
检查随机切分是否被错误使用。
对少量样本打印完整时间线人工核验。

### 10.23 为什么先做小数据过拟合

如果模型连几十个样本都记不住，说明实现可能有 bug。
常见 bug 包括目标错位、mask 方向错误、padding 错误和 loss reshape 错误。
小数据测试成本低。
它比直接跑长时间训练更有效。

### 10.24 训练 loss 不下降怎么排查

先检查输入数值范围。
再检查 target 是否使用正确 SID 版本。
再检查 padding 是否污染 loss。
再检查 mask 是否遮住全部有效 token。
再检查学习率和梯度范数。
最后才考虑增加模型容量。

### 10.25 训练准确率高但线上差怎么办

优先怀疑时间泄漏和数据分布变化。
检查训练推理的 teacher forcing 差异。
检查线上 SID 字典版本。
检查库存和业务过滤比例。
检查线上历史长度是否与训练一致。
检查热门偏置和回退比例。

### 10.26 为什么 token accuracy 提升但 recall 不提升

可能是后层 code 更容易预测，但 SID 冲突或解析失败存在。
也可能是模型只提高了热门 token 的预测。
需要查看 item-level recall、非法率和分桶结果。

### 10.27 如何选择 K 和 V

根据物品规模、冲突率、序列长度和延迟选择。
先用小规模网格实验比较 K 与 V。
报告理论容量不够，还要报告真实利用率。
最终选择不是组合空间最大的设置。

### 10.28 K 越大是不是越好

不是。
K 增大可以降低冲突，但会增加 token 长度、参数和解码延迟。
如果后层只是噪声，推荐效果未必提高。
应同时观察冲突率和端到端指标。

### 10.29 为什么每层要单独报告 loss

总 loss 可能掩盖某一层失败。
第一层和最后一层难度通常不同。
分层 loss 可以发现前缀预测和实例区分的瓶颈。

### 10.30 如何设计多目标损失

先确定主业务目标。
再加入点击、购买、质量或多样性辅助目标。
为每个损失设置可解释权重。
监控每个目标的独立指标和梯度。
一次只增加一个主要目标以便消融。

### 10.31 OneRec 真的不需要召回吗

理论上可以减少召回阶段。
工业上通常仍需要回退和候选合并。
如果物品规模、库存变化或延迟约束很强，保留传统召回更稳健。
端到端路线应该通过逐步灰度验证，而不是一次替换。

### 10.32 生成式推荐为什么可能有更好的泛化

SID 让相似物品共享部分前缀。
模型可以把用户对某个主题的偏好泛化到未见过的实例。
普通 item ID 缺少这种显式结构。
但前提是 SID 的语义结构确实可靠。

### 10.33 SID 的语义如何验证

对每层 code 统计类目、品牌和内容属性分布。
训练 probe 预测这些属性。
可视化同前缀物品的 embedding。
观察共享前缀与相似性的关系。
不要只凭 code 编号猜语义。

### 10.34 如何做 SID 冲突诊断

统计 `sid -> items` 的反向映射基数。
按层统计公共前缀长度。
抽取冲突物品的内容和类目。
判断冲突是合理 alias 还是错误碰撞。

### 10.35 如何保证模型和字典一致

使用发布 manifest 绑定 model version 和 sid version。
服务启动时校验二者。
不一致时拒绝加载或回退。
日志中记录两个版本。
发布必须是原子的。

### 10.36 如何做线上回退

设置生成超时阈值。
设置非法解析率阈值。
设置候选不足阈值。
触发时使用缓存或传统召回。
回退结果仍需经过安全和库存过滤。

### 10.37 如何优化生成延迟

缩短历史窗口。
减少 beam width。
使用 KV cache。
批处理请求。
缓存用户状态和静态 SID trie。
减少不必要的 Python 循环。
最终用端到端 P99 评估。

### 10.38 如何评估工业价值

看线上主指标。
看留存、复购或会话质量。
看覆盖率和多样性。
看过滤率和错误率。
看 P99 延迟和资源成本。
不能只看离线 recall。

### 10.39 Spotify 场景最重要的反馈是什么

播放开始、播放时长、跳过、收藏和加入歌单都重要。
不同反馈代表不同强度。
需要结合会话上下文解释跳过。
列表还要考虑相邻歌曲的转场体验。

### 10.40 淘宝场景最重要的约束是什么

库存、商品上下架、价格、地域和合规是重要约束。
商品生命周期快，内容 SID 和动态交易状态应分离。
交易目标还需要结合点击、加购和购买。

### 10.41 生成列表如何保持多样性

在模型分数基础上加入类别、品牌和内容相似度规则。
可以使用 MMR 或分组配额。
先保证相关性，再在可接受候选中增加多样性。

### 10.42 什么是 exposure bias

训练中模型使用真实前缀。
推理中模型使用自己的预测前缀。
错误会在后续位置累积。
可以通过更强的训练策略或约束解码缓解。

### 10.43 什么是 exposure logging bias

只有被展示的 item 才有点击记录。
未展示 item 没有点击不代表用户不喜欢。
如果忽略曝光日志，训练负样本会被错误构造。

### 10.44 什么时候使用对比学习

当用户状态和目标 SID 的表示对齐不足时，可以加入对比目标。
但它不是默认必需组件。
先完成 token 生成 baseline，再通过消融确认增益。

### 10.45 什么时候使用列表级生成

当列表内互补性和多样性是核心目标时可以考虑。
如果当前单 item 生成还不稳定，不应过早切换列表级复杂目标。

### 10.46 怎样回答“端到端一定更好吗”

不一定。
端到端可以减少阶段间目标不一致，但会增加数据、可控性和系统风险。
是否更好要看完整链路指标、成本和回退能力。

### 10.47 如何选择 baseline

至少使用热门、序列模型和当前线上系统。
如果只与热门比较，无法证明工业收益。
如果只与复杂模型比较，无法判断新增复杂度是否值得。

### 10.48 如何解释 P99

P99 表示 99% 请求的延迟不超过该值。
线上用户体验和服务容量更受尾延迟影响。
平均延迟很低时，P99 仍可能导致超时。

### 10.49 如何处理多版本 SID

历史数据记录使用的 SID 版本。
训练时可以统一到快照版本。
线上解码支持兼容窗口。
发布时记录版本映射。
最终要逐步淘汰旧版本。

### 10.50 如何处理物品删除

停止生成已删除 item。
从有效 trie 中移除或标记不可用。
保留历史日志中的旧 item 以便回放。
训练样本是否跳过删除 item 要提前定义。

### 10.51 如何处理物品合并

建立 item alias 映射。
解码后统一到 canonical item。
避免多个 alias 占满同一推荐列表。

### 10.52 推荐系统的主指标怎么选

主指标应与产品目标直接相关。
内容消费可以看有效播放或留存。
电商可以看购买转化、GMV 和长期价值。
CTR 可以是诊断指标，不一定是最终主指标。

### 10.53 如何设计 guardrail

设置延迟上限。
设置错误率上限。
设置投诉、退款或跳过率上限。
设置覆盖率下限。
设置过滤率和回退率上限。

### 10.54 如何处理模型退化

监控 token 分布、候选覆盖和线上主指标。
发现异常后冻结发布。
使用上一稳定模型回退。
再通过版本和数据 diff 定位原因。

### 10.55 如何证明没有过拟合热门 item

报告按流行度分桶的 recall。
报告推荐列表的平均流行度。
报告长尾覆盖率。
报告不同用户群体的结果。
只报告全量 recall 不能证明没有热门偏置。

### 10.56 如何回答“SID 是不是压缩”

SID 可以减少显式 item 分类的输出空间，但它不只是压缩。
它还提供层级结构和语义共享。
如果 SID 只做无结构的哈希，生成泛化收益可能很弱。

### 10.57 如何回答“SID 是否必须可解释”

SID 不要求每个 code 都能直接翻译成人类标签。
但它应该具有可验证的结构。
可以通过属性 probe、聚类和冲突分析验证其有效性。

### 10.58 如何回答“生成模型能否推荐新 item”

如果新 item 具有内容特征，可以先生成内容 SID。
如果没有内容或编码器不支持新分布，只靠交互训练难以冷启动。
还需要把新 SID 注册到解码字典。

### 10.59 如何回答“为什么不用向量召回”

向量召回和 SID 生成不是互斥的。
向量召回擅长近邻搜索。
SID 生成擅长序列条件和结构化解码。
工业系统可以让二者互补。

### 10.60 如何回答“为什么不用大语言模型直接生成 item ID”

普通 item ID 没有语义结构。
大模型可能生成不存在的 ID。
SID trie 可以提供合法前缀约束。
此外推荐系统还有库存、合规和低延迟要求。
因此需要专门的表示和解码层。

### 10.61 设计一个最小可行 TIGER

先固定 SID 字典。
使用最近 N 个历史 item。
分别 embedding K 个 code。
使用一层或两层 Transformer。
使用 K 个线性输出头。
使用 next-item 交叉熵。
使用 greedy 解码。
使用反向字典解析。
先完成离线 recall 和非法率评估。

### 10.62 设计一个最小可行 OneRec

在 TIGER baseline 上增加列表输出。
加入 item boundary token。
加入约束解码。
加入去重和库存过滤。
保留传统召回回退。
只在 shadow 阶段验证列表质量。

### 10.63 面试中的代码题应检查什么

检查输入 shape。
检查 padding mask。
检查 target shape。
检查 logits reshape。
检查 device。
检查 `num_layers` 是否与 SID 最后一维一致。
检查推理是否使用 `torch.no_grad()`。

### 10.64 代码题：计算 SID 损失

```python
def compute_sid_loss(logits, target):
    # logits: [B, K, V]
    # target: [B, K]
    total = logits.new_zeros(())
    for layer in range(logits.size(1)):
        total = total + F.cross_entropy(
            logits[:, layer, :],
            target[:, layer],
        )
    return total / logits.size(1)
```

回答时说明每层损失平均可以避免 K 改变总尺度。

### 10.65 代码题：取最后有效状态

```python
def gather_last(hidden, lengths):
    # hidden: [B, T, D]
    batch = torch.arange(hidden.size(0), device=hidden.device)
    index = (lengths - 1).clamp_min(0)
    return hidden[batch, index]
```

要说明长度是 item 级还是展开后的 token 级。

### 10.66 代码题：生成合法 code

```python
def mask_invalid(logits, allowed):
    mask = torch.full_like(logits, float("-inf"))
    if allowed:
        ids = torch.tensor(allowed, device=logits.device)
        mask[..., ids] = logits[..., ids]
    return mask
```

真实代码还要处理空 allowed 集合和 batch 状态。

### 10.67 代码题：计算 prefix match

```python
def prefix_match(pred, target):
    length = 0
    for a, b in zip(pred, target):
        if a != b:
            break
        length += 1
    return length
```

它可以帮助定位第一个错误层。

### 10.68 系统设计题：服务拆分

回答可以拆为特征服务、模型服务、解码服务、过滤服务和重排服务。
每个服务使用版本化接口。
模型服务不应直接读取易变业务数据库。
过滤服务必须有缓存和回退。
最终日志需要关联 request_id。

### 10.69 系统设计题：如何做灰度

先离线回放。
再 shadow。
再小比例随机实验。
设置自动回退指标。
逐步扩大流量。
实验结束后检查分群和保护指标。

### 10.70 系统设计题：如何处理 QPS

缓存静态 SID trie。
缓存用户状态。
使用批量推理。
限制 beam size。
控制最大历史长度。
按用户活跃度采用不同计算预算。
超时使用稳定回退。

### 10.71 系统设计题：如何处理字典更新

构建新字典并离线校验。
生成兼容 manifest。
预加载新 trie。
模型和字典一起 shadow。
通过原子切换发布。
保留旧版本用于回退。

### 10.72 追问：如果 trie 约束降低 recall

检查 SID 字典覆盖率。
检查模型是否使用同一版本字典。
检查 beam 是否过小。
检查约束是否错误屏蔽了合法路径。
检查目标 item 是否已经下架。
不能直接删除约束来掩盖数据问题。

### 10.73 追问：如果 exact match 很低

先看 prefix match。
再看 item recall。
如果 item recall 高，可能存在 alias 或 SID 等价。
如果 prefix 也低，检查目标 shift、mask 和模型能力。

### 10.74 追问：如果新商品效果差

查看内容 SID 质量。
查看新商品特征缺失率。
查看新商品是否进入 trie。
查看冷启动曝光预算。
单独调研新品流量分布。

### 10.75 追问：如果长尾覆盖提升但购买下降

说明探索可能损害相关性。
降低长尾权重或增加相关性阈值。
按用户分群使用不同探索强度。
把购买、退款和长期留存放入保护指标。

### 10.76 追问：为什么训练时要记录 SID 版本

SID 是标签的一部分。
标签变化会让相同 item 对应不同目标。
不记录版本就无法复现 loss 和线上解码。

### 10.77 追问：为什么不能直接把多个召回结果拼接

拼接会造成重复、排序不一致和来源偏置。
需要统一 item 归一、去重、分数校准和重排。
生成式候选也必须走同一后处理链路。

### 10.78 追问：如何提高可解释性

展示生成 SID 对应的类目、主题或内容属性。
记录每个生成层的概率。
记录候选被过滤的原因。
不要把 code 编号本身当作解释。

### 10.79 追问：为什么不用一个大 softmax

item 数量大时，参数和计算成本高。
大 softmax 也不能利用 item 之间的结构。
SID 把预测拆成多个较小的分类问题。
但拆分会引入冲突和解码问题。
需要完整比较收益与成本。

### 10.80 追问：SID 会不会损失信息

会。
离散量化和固定长度表示可能丢失细节。
因此要评估重建质量、冲突率和下游效果。
如果信息损失造成 recall 下降，需要增加容量或改进编码器。

### 10.81 追问：生成式推荐的最大风险

最大风险包括不可控输出、离线线上偏差、热门坍缩、延迟和版本不一致。
解决方案不是只提高模型规模。
需要表示质量、约束、回退和监控共同处理。

### 10.82 追问：OneRec 为什么仍然需要重排

模型生成的概率不一定包含所有实时商业约束。
重排可以融合库存、价格、活动和多样性。
完全去掉重排需要非常强的上下文建模和约束能力。
工业上通常逐步减少而不是立即删除。

### 10.83 追问：如何判断值得上线

至少满足主指标提升或不下降。
非法率、延迟、成本和保护指标在阈值内。
有明确回退路径。
分群结果没有严重恶化。
模型和数据版本可追溯。

### 10.84 追问：怎样避免回答过度绝对

使用条件化表达。
例如“在 SID 唯一且字典一致时，exact match 才能近似 item 命中”。
说明收益依赖数据、业务和延迟约束。
这比简单说“端到端一定更好”更严谨。

### 10.85 综合题：从零设计 SID 生成推荐

第一步定义业务目标和主指标。
第二步清洗曝光与行为日志。
第三步训练内容和协同结合的 SID 编码器。
第四步检查冲突率、覆盖率和码本利用率。
第五步按时间构造 next-item 样本。
第六步训练 TIGER baseline。
第七步使用反向字典和 trie 约束解码。
第八步加入库存、去重和多样性处理。
第九步与热门、序列模型和当前线上系统比较。
第十步 shadow、灰度并保留回退。

### 10.86 综合题：TIGER 线上效果差的排查顺序

先检查请求和历史是否为空。
再检查 SID 字典版本。
再检查模型输入 shape。
再检查非法 SID 比例。
再检查过滤和回退比例。
再检查候选热度分布。
再检查离线线上时间切分。
最后再调整模型结构。

### 10.87 综合题：为什么模型离线提升但线上下降

可能是离线使用了随机切分。
可能是线上商品状态变化。
可能是训练目标与线上目标不同。
可能是生成延迟导致请求超时。
可能是过滤规则删除了高分候选。
可能是实验流量分布不同。

### 10.88 综合题：如何解释工业落地难点

模型只是链路的一部分。
工业难点通常在数据质量、实时特征、字典同步、约束解码、尾延迟和实验设计。
回答时要同时覆盖算法和系统。

### 10.89 面试答题自检

是否定义了 SID。
是否说清输入形状。
是否说清 embedding 方式。
是否说清 Transformer 的作用。
是否说清多头输出。
是否说清 teacher forcing。
是否说清解码和 trie。
是否说清 item-level 评估。
是否说清线上约束和回退。
是否说清 TIGER 与 OneRec 的边界。

### 10.90 最终记忆卡片

SID：物品的多层离散语义表示。
TIGER：基于 SID 的生成式 next-item 推荐。
OneRec：更端到端的生成式推荐路线。
训练：按时间构造历史到目标。
损失：每个 SID 层的交叉熵。
推理：greedy、beam 或 constrained beam。
约束：trie、库存、合规、去重和多样性。
评估：token、prefix、item、覆盖、延迟和成本。
工业：版本、日志、灰度和回退。
面试：先结论，再结构，最后边界。

### 10.91 全章复习题

- [ ] 能写出 `[B,T,K]` 到 Transformer 输入的变换。
- [ ] 能解释为什么每个 SID 层需要独立输出头。
- [ ] 能说明历史 mask 和目标 causal mask 的区别。
- [ ] 能写出多层 SID 交叉熵。
- [ ] 能解释 teacher forcing 与 exposure bias。
- [ ] 能说明 trie 如何屏蔽非法前缀。
- [ ] 能解释 SID 冲突如何影响 exact match。
- [ ] 能设计时间切分的 next-item 数据集。
- [ ] 能列出 Spotify 场景的反馈信号。
- [ ] 能列出淘宝场景的动态约束。
- [ ] 能设计生成式推荐的回退方案。
- [ ] 能解释为什么 token accuracy 不是最终指标。
- [ ] 能设计 beam size 消融实验。
- [ ] 能说明冷启动 item 如何生成 SID。
- [ ] 能解释为什么动态库存不应写进稳定 SID。
- [ ] 能说出至少五个线上监控指标。
- [ ] 能从数据、模型、解码、服务四层排查问题。
- [ ] 能比较 TIGER、OneRec 和传统多阶段系统。
- [ ] 能说明端到端模型的收益和风险。
- [ ] 能给出上线前的最小验收清单。

### 10.92 章节总结

SID 提供了从物品空间到离散语义空间的桥梁。
TIGER 展示了如何用 Transformer 生成下一个物品的 SID。
OneRec 进一步探索端到端的推荐序列生成。
训练过程需要严格的时间边界和版本治理。
工业应用需要把稳定语义与动态业务状态分开。
评估必须从 token 指标延伸到 item、用户和系统指标。
面试回答需要同时体现算法理解和工程判断。

### 10.93 延伸阅读方向

下一步可以学习 SID 的多任务学习。
下一步可以学习生成式检索的候选约束。
下一步可以学习向量召回与生成式召回融合。
下一步可以学习推荐系统的曝光偏差校正。
下一步可以学习列表级排序和多样性优化。
下一步可以学习在线 A/B 实验和因果评估。
下一步可以学习分布式 Transformer 推理。

### 10.94 实战项目建议

项目一：用公开会话数据训练一个小型 TIGER。
项目二：为商品特征训练一个五层 SID 编码器。
项目三：实现 SID trie 和 constrained beam search。
项目四：比较 greedy 与 beam 的 recall 和延迟。
项目五：增加长尾校正并分析覆盖率变化。
项目六：实现模型、字典和 trie 的 manifest 发布。
项目七：构造 shadow 日志并模拟回退。
项目八：写一份包含分群指标的离线评估报告。

### 10.95 实战验收标准

能够在小数据上过拟合。
能够稳定保存和恢复 checkpoint。
能够把预测 SID 解析为 item。
能够统计非法 SID 比例。
能够实现 top-k recall。
能够比较至少两个解码策略。
能够在字典变更时发现版本不一致。
能够在生成超时时返回回退候选。
能够解释离线指标与线上指标的差异。

### 10.96 终章 checklist

- [ ] 原始日志有用户、物品、事件和时间。
- [ ] 训练、验证、测试按时间划分。
- [ ] SID 字典有正向和反向映射。
- [ ] SID 冲突率已统计。
- [ ] padding code 与合法 code 隔离。
- [ ] 历史和目标的 mask 已验证。
- [ ] 小数据过拟合通过。
- [ ] 每层 loss 和 accuracy 已记录。
- [ ] exact match 与 item recall 已区分。
- [ ] greedy 与 beam 已比较。
- [ ] trie 约束已覆盖非法前缀。
- [ ] 去重和库存过滤已接入。
- [ ] 长尾和冷启动已分组评估。
- [ ] 线上 P99 和回退率已监控。
- [ ] 模型与 SID 版本已绑定。
- [ ] 灰度发布可自动回退。
- [ ] Spotify 类会话指标已理解。
- [ ] 淘宝类交易约束已理解。
- [ ] TIGER 与 OneRec 的边界已理解。
- [ ] 能够独立完成系统设计回答。

### 10.97 结论

生成式推荐的核心不是把推荐结果写成一段文本。
核心是构造可学习、可生成、可约束、可还原的物品表示。
SID 负责提供结构化输出空间。
TIGER 负责把用户历史映射到下一个 SID。
OneRec 负责探索更统一的推荐生成目标。
最终的技术判断必须回到用户价值、系统约束和可验证指标。
