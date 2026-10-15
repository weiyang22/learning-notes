"""
Example: build Hy Token (gen-token + mem-token) per 方案 3 / 论文 §3.2.

【Hy Token 是什么】
================
ByteDance TRM 论文(arXiv 2601.22694)§3.2 提出的 SID 增强方案:
    - gen-token = 每个物品的"主 SID",由 RQ-VAE 或 RQ-Kmeans 产出,长度 5
    - mem-token = 在 gen-token 序列上跑 BPE,把高频相邻 token 合并成新 token
    - Hy Token = gen-token(generalization)+ mem-token(memorization)
                  通过 Wide & Deep 融合(Wide 走 mem,Deep 走 gen)

【它解决了什么】
================
RQ-VAE 的 5 层 codebook 容量 ≈ 256^5 ≈ 10^12,看似足够,但很多"高阶语义组合"
(比如 [3C, 手机, 苹果, iPhone, 黑色] 这种具体搭配)是 5 元组合的稀有种,
RQ-VAE 的 codebook 学不到。Hy Token 通过 BPE 把这些高频组合记录成新 token,
补回"组合知识"。

【本文件做了什么】
================
    1. 用方案 1 那种带行为融合的 RQ-VAE 训练 500 个候选集,产出 gen-token
       (每个候选集一段 5 元整数序列)
    2. 在这 500 段 gen-token 上跑 BPE,产出 mem-token 表(高频 token 合并)
    3. 把每个候选集的 gen-token 和 mem-token 都融合成 Hy-Token embedding
    4. 同时演示"TopK n-gram"快速版(mem-token 的简化算法)

【运行】
================
    python3 examples/run_hy_token.py

【推荐先读】
================
    - docs/03_hy_token_construction.md(详细方案)
    - src/hy_token.py(Hy Token / BPE 实现)
    - src/build_candidate_sid.py(产生 gen-token 的封装)
"""
import sys
from pathlib import Path

# 把仓库根加到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch

# make_synthetic_dataset:生成 500 物品 + 100 用户 + 8 平均行为的假数据
# CandidateSIDConfig / CandidateSIDBuilder:与方案 1 相同,产 RQ-VAE 与 SID
from src.data_loader import make_synthetic_dataset
from src.build_candidate_sid import (
    CandidateSIDConfig, CandidateSIDBuilder,
)
# build_mem_tokens_bpe       : 标准 BPE 训练,在 token 序列上做合并
# encode_with_bpe            : 把原 token 序列用 BPE 表编码(变成变长 mem-token 序列)
# build_mem_tokens_topk_grams: 简化版,只统计 2-gram / 3-gram 频率,取 TopK
# expand_with_topk_grams     : 用 TopK n-gram 表展开成新的 token 序列
# pad_mem_tokens             : 把变长 mem-token 序列 pad 到定长,便于 batch
# HyTokenFusion              : 把 gen-token + mem-token 融合成 Hy-Token embedding
from src.hy_token import (
    build_mem_tokens_bpe, encode_with_bpe,
    build_mem_tokens_topk_grams, expand_with_topk_grams,
    pad_mem_tokens, HyTokenFusion,
)


def main():
    # ============================================================
    # Step 1: 生成假数据(同方案 1,但换 seed=2 让数据有差异)
    # ============================================================
    # 500 物品,128 维 embedding,每物品平均 8 个行为事件
    ds = make_synthetic_dataset(
        n_candidates=500, n_queries=0,
        emb_dim=128, behaviors_per_item_mean=8, seed=2,
    )

    # ============================================================
    # Step 2: 训练 RQ-VAE 产 gen-token
    # ============================================================
    # 这里的超参与 run_candidate_sid.py 几乎一致
    cfg = CandidateSIDConfig(
        input_dim=128, hidden_dim=128, latent_dim=64,
        num_layers=5, codebook_size=256,
        epochs=15, batch_size=64, lr=1e-3, alpha=0.4,
    )
    builder = CandidateSIDBuilder(cfg)
    builder.fit(ds.candidate_embs, ds.behaviors_per_item)
    sids = builder.encode(ds.candidate_embs, ds.behaviors_per_item)
    # sids.shape = (500, 5),每个元素是 [0, 256) 的整数

    # ============================================================
    # Step 3: 在 gen-token 序列上跑 BPE → mem-token 表
    # ============================================================
    corpus = sids.tolist()   # List[List[int]],每段是 5 元 token 序列
    print(f"Gen-token corpus: {len(corpus)} sequences, "
          f"each length {len(corpus[0])}")

    # 标准 BPE 训练:
    #   - vocab_size=5000 : 限制最终词表大小;vocab 满则停止
    #   - min_freq=2      : 出现 < 2 次的 pair 不合并
    # 这里 vocab_size 故意给得偏小,因为我们只有 500 条 5-gram 序列
    # 生产级(数亿物品)一般 vocab_size = 2*10^7
    bpe = build_mem_tokens_bpe(corpus, vocab_size=5000, min_freq=2)
    print(f"BPE vocab size: {len(bpe.vocab)}")

    # 用 BPE 表编码 corpus → mem_sequences (变长 mem-token 序列)
    mem_sequences = encode_with_bpe(corpus, bpe)
    # 看平均有多少个 mem-token(应该 < 5,因为 BPE 合并了相邻 token)
    print(f"Avg mem-token seq length: "
          f"{sum(len(s) for s in mem_sequences)/len(mem_sequences):.1f}")

    # ============================================================
    # Step 4: 用 HyTokenFusion 把 gen + mem 融合
    # ============================================================
    # HyTokenFusion 内部做的事:
    #   - gen_token → Embedding(256, dim) → 5 个向量
    #   - mem_token → Embedding(|vocab|, dim) → 平均池化
    #   - 拼接 + Linear → Hy Token embedding
    fused = HyTokenFusion(
        gen_vocab_size=cfg.codebook_size,    # 256
        num_layers=cfg.num_layers,           # 5
        mem_vocab_size=max(len(bpe.vocab), 1),
    )
    # 把 numpy 转 torch.LongTensor 才能进 nn.Embedding
    gen_tensor = torch.tensor(corpus, dtype=torch.long)
    # mem_sequences 是变长 list of list,要 pad 到定长
    mem_tensor = pad_mem_tokens(mem_sequences)
    # 前向融合 → shape = (N, D_hy)
    out = fused(gen_tensor, mem_tensor)
    print(f"Hy-Token embedding shape: {tuple(out.shape)}")

    # ============================================================
    # Step 5: 演示"TopK n-gram"快速版(简化版 BPE)
    # ============================================================
    # 完整 BPE 在数亿 5-gram 序列上跑会很慢(几个 CPU 日)
    # TopK n-gram 只统计 2-gram / 3-gram,取频率前 K 个作 vocab
    # 牺牲一点效果换 100x 速度,适合 demo / 早期生产
    topk = build_mem_tokens_topk_grams(corpus, max_gram=3, top_k=500, min_freq=2)
    print(f"Top-k n-gram vocab size: {len(topk)}")

    # ============================================================
    # Step 6: 保存元信息(给下游或训练排序模型用)
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "hy_token_meta.json", "w") as f:
        json.dump({
            "n_items": len(corpus),
            "gen_token_seq_len": len(corpus[0]),
            "bpe_vocab_size": len(bpe.vocab),
            "topk_vocab_size": len(topk),
            "hy_token_dim": out.shape[1],
            # 实际生产还需要把 BPE 表 / TopK 表 / fused 模型权重都持久化
            # 这里只存元信息,demo 简化
        }, f, indent=2)
    print(f"Saved metadata to {out_dir}/hy_token_meta.json")


if __name__ == "__main__":
    main()
