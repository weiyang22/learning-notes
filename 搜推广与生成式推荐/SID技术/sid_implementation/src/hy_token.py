"""
Hy Token 构造 (TRM arXiv 2601.22694 §3.2)

Hy Token = gen-token (RQ 量化的整数序列) + mem-token (BPE 在 gen-token 序列上学习的高阶 n-gram)
用 Wide & Deep 架构融合两者:
  - Deep 侧: MLP 处理 RQ 编码的 5 层整数 (gen-token)
  - Wide 侧: Embedding pool 处理 BPE 学到的 n-gram (mem-token)

动机 (来自 TRM 论文):
    仅用 RQ 编码 (gen-token) 表达物品, 缺乏"哪些 SID 组合在数据中常一起出现"的协同信号。
    用 BPE 在 gen-token 序列上学高频 n-gram, 得到 mem-token, 编码了"组合模式"。
    把二者融合, 既保留单层语义 (Deep), 又捕捉协同信号 (Wide)。

Public API:
    - extract_gen_tokens: RQ-VAE → 物品 → 5 层 gen-token 序列
    - build_mem_tokens_bpe: 标准 BPE, 学高频 n-gram 作为 mem-token 词表
    - build_mem_tokens_topk_grams: ponytail 简化版, 只数 top-k 高频 n-gram
    - WideEmbedding: Wide 侧 embedding (mem-token pool)
    - DeepEmbedding: Deep 侧 embedding (gen-token → MLP)
    - HyTokenFusion: Wide + Deep 融合 (相加)
    - pad_mem_tokens: 把变长 mem-token 序列填充成 (N, T) 矩阵
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


def _lazy_import_encode_to_sid():
    """延迟 import encode_to_sid, 避免循环依赖。

    为什么延迟?
        src.rq_vae 内部可能会 (直接或间接) 引用 src.* 的其他模块,
        而本模块被其他模块引用时如果立即 import rq_vae, 可能触发循环。
        用函数包裹 import 是 Python 社区的常用 trick。
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from src.rq_vae import encode_to_sid
    return encode_to_sid


def extract_gen_tokens(rqvae, embeddings: torch.Tensor,
                       batch_size: int = 1024,
                       device: str = "cpu") -> torch.Tensor:
    """用训练好的 RQ-VAE 把物品 embedding 编码成 gen-token (整数 SID) 序列。

    Args:
        rqvae: 训练好的 src.rq_vae.RQVAE 实例。
        embeddings: (N, input_dim) 物品 embedding。
        batch_size: 切批大小, 防止大 N 一次跑爆显存。
        device: 推理设备。

    Returns:
        (N, num_layers) int64 SID 矩阵。
    """
    encode_to_sid = _lazy_import_encode_to_sid()
    return encode_to_sid(rqvae, embeddings, batch_size=batch_size, device=device)


@dataclass
class BPEResult:
    """BPE 学习结果, 含词表和合并历史。

    字段:
        vocab: {(token_tuple): new_id} 把 n-gram (元组) 映射到新分配的 id
        merges: [(left_pair, right_pair), ...] 合并历史 (论文调试 / 可视化用)
    """
    vocab: Dict[Tuple[int, ...], int]
    merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]


def _pair_frequencies(corpus: List[List[int]]) -> Counter:
    """统计语料中所有相邻 token 对的频率。

    BPE 的核心操作: 每次找频率最高的相邻对, 把它们合并成一个新的 token。

    Args:
        corpus: List[List[int]], 每个内层 list 是一条 token 序列 (如 SID)。

    Returns:
        Counter, key 是 (left, right) 元组, value 是出现次数。
    """
    freq = Counter()
    for seq in corpus:
        # 滑动窗口统计相邻对
        for i in range(len(seq) - 1):
            freq[(seq[i], seq[i + 1])] += 1
    return freq


def _apply_merge(corpus: List[List[int]], pair: Tuple[int, int],
                 new_id: int) -> List[List[int]]:
    """把语料中所有 (pair[0], pair[1]) 相邻对替换为 new_id。

    这是 BPE 的更新步骤: 合并完后, 整个语料的"长度"会缩短,
    下一步统计的相邻对也会跟着变。

    Args:
        corpus: 原始 token 序列列表 (会被复制, 不修改原对象)。
        pair: 要合并的 (left, right) token 对。
        new_id: 合并后的新 token id。

    Returns:
        合并后的新语料。
    """
    new_corpus = []
    for seq in corpus:
        new_seq = []
        i = 0
        while i < len(seq):
            # 同时满足: 不越界 + 当前两个 token 正好是目标对
            if (i + 1 < len(seq)
                    and seq[i] == pair[0]
                    and seq[i + 1] == pair[1]):
                new_seq.append(new_id)
                i += 2  # 跳过两个, 已合并
            else:
                new_seq.append(seq[i])
                i += 1
        new_corpus.append(new_seq)
    return new_corpus


def build_mem_tokens_bpe(corpus: Sequence[Sequence[int]],
                         vocab_size: int = 2_000_000,
                         min_freq: int = 2) -> BPEResult:
    """在整数 token 序列上跑标准 BPE (TRM §3.2 mem-token)。

    算法:
      1. 把所有基本 token 加入词表 (单元素元组)
      2. 重复: 找频率最高的相邻对 → 合并成新 token → 加入词表
      3. 直到词表达到 vocab_size 或没有高频对可合并

    Args:
        corpus: List[Sequence[int]], 每条是 RQ 量化后的 SID 序列。
        vocab_size: 目标词表大小 (论文用 2e7, demo 用更小)。
        min_freq: 频率低于此值的对不合并, 防止过拟合噪声。

    Returns:
        BPEResult 实例。
    """
    corpus = [list(seq) for seq in corpus]

    # 收集所有出现过的基础 token, 排序保证词表 id 稳定
    base_tokens = sorted({tok for seq in corpus for tok in seq})

    vocab: Dict[Tuple[int, ...], int] = {}
    next_id = 0
    for tok in base_tokens:
        # 用元组 (tok,) 作为 key, 与多 token n-gram 统一
        vocab[(tok,)] = next_id
        next_id += 1

    merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    # 持续合并直到词表满 或 没有高频对
    while next_id < vocab_size:
        freq = _pair_frequencies(corpus)
        if not freq:
            break
        pair, count = freq.most_common(1)[0]
        if count < min_freq:
            break
        new_id = next_id
        next_id += 1
        vocab[pair] = new_id
        # 记录合并历史 (调试 / 回溯用)
        merges.append((pair, (pair[0], pair[1])))
        corpus = _apply_merge(corpus, pair, new_id)

    return BPEResult(vocab=vocab, merges=merges)


def encode_with_bpe(corpus: Sequence[Sequence[int]],
                    result: BPEResult) -> List[List[int]]:
    """用学习好的 BPE 词表编码新序列 (贪心最长匹配)。

    算法: 从每个位置尝试匹配最长的 vocab 中的 n-gram (上限 8-gram)。

    Args:
        corpus: 待编码的序列列表。
        result: build_mem_tokens_bpe 返回的词表。

    Returns:
        编码后的 token id 列表。
    """
    vocab = result.vocab
    out: List[List[int]] = []
    for seq in corpus:
        new_seq: List[int] = []
        i = 0
        while i < len(seq):
            best = None
            best_len = 0
            # 从长到短尝试匹配, 命中就用最长的
            # 上限 8 是经验值, 防止极端稀疏的长 n-gram 让编码变慢
            for n in range(min(8, len(seq) - i), 0, -1):
                piece = tuple(seq[i:i + n])
                if piece in vocab:
                    best = piece
                    best_len = n
                    break
            if best is None:
                # 没有任何 n-gram 命中 (理论上单 token 总该命中), 直接保留原 token
                new_seq.append(seq[i])
                i += 1
            else:
                new_seq.append(vocab[best])
                i += best_len
        out.append(new_seq)
    return out


def build_mem_tokens_topk_grams(corpus: Sequence[Sequence[int]],
                                max_gram: int = 3,
                                top_k: int = 100_000,
                                min_freq: int = 2) -> Dict[Tuple[int, ...], int]:
    """Ponytail 简化版 mem-token: 数所有 1..max_gram 元 n-gram, 取 top_k。

    与 BPE 的区别:
      - BPE 是"贪心合并高频对", 学到的是层次化合并树
      - topk 直接枚举所有短 n-gram, 简单但可能冗余

    适用场景: 资源受限 / demo / 不想跑完整 BPE (BPE 在大语料上很慢)。

    Args:
        corpus: token 序列列表。
        max_gram: n-gram 最大长度 (典型 3)。
        top_k: 保留的词表大小。
        min_freq: 过滤低频噪声。

    Returns:
        {n-gram tuple: 新 id} 词表。
    """
    counts: Counter = Counter()
    for seq in corpus:
        n = len(seq)
        # 枚举长度 1..max_gram 的所有 n-gram
        for k in range(1, max_gram + 1):
            if n < k:
                continue
            for i in range(n - k + 1):
                gram = tuple(seq[i:i + k])
                counts[gram] += 1

    # 取频率最高的 top_k, 并过滤 min_freq
    top = [(g, c) for g, c in counts.most_common(top_k) if c >= min_freq]
    # 重新分配 id, 频率高的在前 (id 小)
    return {g: idx for idx, (g, _) in enumerate(top)}


def expand_with_topk_grams(corpus: Sequence[Sequence[int]],
                           mem_vocab: Dict[Tuple[int, ...], int],
                           max_gram: int = 3) -> List[List[int]]:
    """对每个序列, 在末尾追加匹配到的 mem-token id。

    为什么不替换原 token?
        这是 ponytail 简化: 保留原始 gen-token, 在末尾追加 mem-token。
        下游 WideEmbedding 对两部分一起做 sum pool, 不区分来源。

    Args:
        corpus: 原始 token 序列。
        mem_vocab: topk 词表。
        max_gram: 与 build_mem_tokens_topk_grams 一致。

    Returns:
        扩展后的序列列表 (原 token + mem-token 拼接)。
    """
    out: List[List[int]] = []
    for seq in corpus:
        seq_list = list(seq)
        n = len(seq_list)
        # 从 2-gram 开始 (单 token 已经是 gen-token 本身, 没意义重复)
        for k in range(2, max_gram + 1):
            if n < k:
                continue
            for i in range(n - k + 1):
                gram = tuple(seq_list[i:i + k])
                if gram in mem_vocab:
                    seq_list.append(mem_vocab[gram])
        out.append(seq_list)
    return out


class WideEmbedding(nn.Module):
    """Wide 侧: mem-token 序列 → 池化 → (embedding_dim,) 向量。

    论文 §3.2 的设计: mem-token 用 hashing-style 大词表 + sum/mean pool,
    等价于"传统 wide 模型"的稀疏特征交叉, 但用低维 embedding 实现。

    Args:
        vocab_size: mem-token 词表大小 (可能上千万)。
        embedding_dim: 每个 token 的 embedding 维度。
        pool: "sum" 或 "mean"。
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 16,
                 pool: str = "sum"):
        super().__init__()
        # 词表可能很大, 这里用 nn.Embedding 而非哈希桶, 因为 mem-token 已经
        # 是整数 id, 不需要再做特征哈希
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pool = pool

    def forward(self, mem_token_ids: torch.Tensor) -> torch.Tensor:
        """mem_token_ids: (N, T) long → (N, embedding_dim)。"""
        emb = self.embedding(mem_token_ids)
        if self.pool == "sum":
            return emb.sum(dim=1)
        # mean pool: 适合变长序列, 但要小心 padding 0 的影响 (mean 会被稀释)
        return emb.mean(dim=1)


class DeepEmbedding(nn.Module):
    """Deep 侧: gen-token 序列 → token emb → flatten → MLP → (embedding_dim,)。

    论文里 Deep 侧加了 dropout (防止过拟合 RQ 量化引入的噪声)。

    Args:
        gen_vocab_size: RQ codebook 大小 (如 256)。
        num_layers: RQ 层数 (如 5)。
        embedding_dim: 每层 token 的 embedding 维度。
        hidden_dim: MLP 中间层维度。
        dropout: dropout 概率。
    """

    def __init__(self, gen_vocab_size: int, num_layers: int,
                 embedding_dim: int = 32, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(gen_vocab_size, embedding_dim)
        # MLP: 把 num_layers 个 token embedding 拼起来再降维
        # 论文里的"depth" 就体现在这里
        self.mlp = nn.Sequential(
            nn.Linear(num_layers * embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.num_layers = num_layers

    def forward(self, gen_token_ids: torch.Tensor) -> torch.Tensor:
        """(N, num_layers) long → (N, embedding_dim)"""
        emb = self.token_embedding(gen_token_ids)         # (N, L, D)
        flat = emb.view(emb.size(0), -1)                  # (N, L*D)
        return self.mlp(flat)


class HyTokenFusion(nn.Module):
    """Hy Token 最终融合: Deep(gen-token) + Wide(mem-token) 相加。

    Ponytail: 简单 sum fusion, 不做 concat+linear (论文里也允许这种简化)。
    Wide 输出需要投影到 deep_dim 才能相加。
    """

    def __init__(self, gen_vocab_size: int, num_layers: int,
                 mem_vocab_size: int,
                 deep_dim: int = 32, mem_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.deep = DeepEmbedding(gen_vocab_size, num_layers,
                                  embedding_dim=deep_dim, dropout=dropout)
        self.wide = WideEmbedding(mem_vocab_size, embedding_dim=mem_dim)
        # 把 mem_dim 投影到 deep_dim, 让两个分支的输出同维度可相加
        self.wide_proj = nn.Linear(mem_dim, deep_dim)

    def forward(self, gen_token_ids: torch.Tensor,
                mem_token_ids: torch.Tensor) -> torch.Tensor:
        return self.deep(gen_token_ids) + self.wide_proj(self.wide(mem_token_ids))


def pad_mem_tokens(mem_sequences: Sequence[Sequence[int]],
                  pad_id: int = 0) -> torch.Tensor:
    """把变长 mem-token 序列填充成 (N, max_len) 矩阵, 填充值 0。

    神经网络要求 batch 内张量形状一致, 所以变长序列必须 pad。
    0 作为 pad 是约定, 因为词表里通常不会给 0 分配真 token。

    Args:
        mem_sequences: List[List[int]] 变长序列列表。
        pad_id: 填充值, 默认 0。

    Returns:
        (N, max_len) long 张量。
    """
    max_len = max(len(s) for s in mem_sequences)
    out = torch.full((len(mem_sequences), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(mem_sequences):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 8 条玩具 SID 序列, 模拟 5 层 RQ 量化结果
    toy_corpus = [
        [3, 127, 890, 45, 1],
        [3, 127, 890, 45, 7],
        [3, 127, 200, 45, 1],
        [3, 200, 890, 12, 1],
        [3, 127, 200, 45, 1],
        [3, 127, 890, 12, 7],
        [3, 200, 890, 45, 1],
        [10, 20, 30, 40, 50],
    ]

    # 跑一遍 BPE
    result = build_mem_tokens_bpe(toy_corpus, vocab_size=20, min_freq=1)
    print(f"BPE vocab size = {len(result.vocab)}")
    print(f"Top merges: {result.merges[:5]}")
    encoded = encode_with_bpe(toy_corpus, result)
    print(f"First 3 encoded: {encoded[:3]}")

    # 跑一遍简化版 topk
    topk = build_mem_tokens_topk_grams(toy_corpus, max_gram=2, top_k=10, min_freq=1)
    print(f"Top-k 2-gram vocab size = {len(topk)}")
    expanded = expand_with_topk_grams(toy_corpus, topk, max_gram=2)
    print(f"First 3 expanded lengths: {[len(s) for s in expanded[:3]]}")

    # 把 Hy Token 融合模块跑通
    gen_ids = torch.tensor(toy_corpus[:4], dtype=torch.long)
    mem_ids = pad_mem_tokens(expanded[:4])
    fused = HyTokenFusion(gen_vocab_size=1024, num_layers=5, mem_vocab_size=4096)
    out = fused(gen_ids, mem_ids)
    print(f"Hy-Token output shape = {tuple(out.shape)}")
    assert out.shape == (4, 32), f"expected (4, 32), got {out.shape}"
    print("OK")