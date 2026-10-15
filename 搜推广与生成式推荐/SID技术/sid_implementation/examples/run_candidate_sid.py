"""
Example: build candidate SID with behavior fusion (方案 1).

【本文件做了什么】
================
演示"方案 1:候选集 SID(含用户行为辅助)"的完整流水线:
    1. 生成 500 个候选集的合成数据(embedding + 用户行为日志)
    2. 用 BehaviorFusion 把用户行为注入候选集 embedding
    3. 训练一个 5 层 RQ-VAE(每层 256 个 codeword)
    4. 对每个候选集产出 SID = [s1, s2, s3, s4, s5]
    5. 把 SID 落到 outputs/candidate_sids.{npy,json}

【运行】
    python3 examples/run_candidate_sid.py

【推荐先读】
    - docs/00_sid_overview_and_principles.md § 4.2 / 4.4(行为融合 / RQ-VAE)
    - docs/01_candidate_sid_with_behavior.md(方案 1 详细方案)
    - src/behavior_fusion.py(行为融合实现)
    - src/build_candidate_sid.py(本例的封装)
    - src/rq_vae.py(RQ-VAE 实现)
"""
import sys
from pathlib import Path

# 把仓库根目录加进 sys.path,这样 `from src.xxx import yyy` 才能找到
# Path(__file__).resolve().parent.parent = .../sid_implementation/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np

# make_synthetic_dataset:生成假数据(候选集 embedding + 用户行为)
# CandidateSIDConfig / CandidateSIDBuilder:封装"embedding 融合 + RQ-VAE + SID"
from src.data_loader import make_synthetic_dataset
from src.build_candidate_sid import (
    CandidateSIDConfig, CandidateSIDBuilder,
)


def main():
    # ============================================================
    # Step 1: 生成合成数据
    # ============================================================
    # n_candidates=500  : 500 个候选物品
    # n_queries=0       : 这是"推荐侧"(无 search query),不需要 query
    # n_users=100       : 100 个用户会产生行为
    # emb_dim=128       : 每个 embedding 是 128 维向量
    # behaviors_per_item_mean=8 : 每个物品平均有 8 次行为(曝光/点击/下载/付费)
    # seed=0            : 固定随机种子,保证每次跑出来的"假数据"一样
    ds = make_synthetic_dataset(
        n_candidates=500,
        n_queries=0,
        n_users=100,
        emb_dim=128,
        behaviors_per_item_mean=8,
        seed=0,
    )
    # 合成数据的对象结构:
    # - ds.candidate_ids    : List[str], 500 个物品 ID
    # - ds.candidate_embs   : np.ndarray (500, 128),每个物品一个 128 维向量
    # - ds.behaviors_per_item: List[List[(user_id, code)]],每个物品的曝光/点击日志
    #                         code ∈ {0=曝光, 1=点击, 2=下载, 3=付费}

    # ============================================================
    # Step 2: 配置并初始化 SID 训练器
    # ============================================================
    # 配置项含义:
    #   input_dim=128  : 输入 embedding 维度 = 上面的 emb_dim
    #   hidden_dim=128 : Encoder / Decoder 的 MLP 隐藏层维度
    #   latent_dim=64  : RQ-VAE 的潜在空间维度(= codeword 维度)
    #                    注意:这里特意用 64,比 128 小,起一个"压缩"作用
    #   num_layers=5   : 5 层残差量化 → SID 长度 = 5
    #   codebook_size=256 : 每层 256 个 codeword → 每层 token ∈ [0, 255]
    #                       组合数 = 256^5 ≈ 1.1e12,远超任何候选集规模
    #   epochs=20      : 训练 20 个 epoch(数据量小,够用)
    #   batch_size=64  : batch 大小
    #   lr=1e-3        : Adam 学习率
    #   alpha=0.4      : 行为信号的注入强度(详见 behavior_fusion.py)
    cfg = CandidateSIDConfig(
        input_dim=128, hidden_dim=128, latent_dim=64,
        num_layers=5, codebook_size=256,
        epochs=20, batch_size=64, lr=1e-3, alpha=0.4,
    )
    # CandidateSIDBuilder = 行为融合 + RQ-VAE 训练 + 编码 三合一的封装
    builder = CandidateSIDBuilder(cfg)

    # ============================================================
    # Step 3: 训练 RQ-VAE
    # ============================================================
    # fit() 内部做了 3 件事:
    #   a. 对每个候选集,把其 embedding 与"用户行为对应的向量"加权融合
    #      (细节见 src/behavior_fusion.py::fuse_candidate_batch)
    #   b. 用融合后的 embedding 训练 RQ-VAE,学到 5 个 codebook
    #      (细节见 src/rq_vae.py::train_rqvae)
    #   c. codebook 走 EMA 更新(指数滑动平均),更稳定
    builder.fit(ds.candidate_embs, ds.behaviors_per_item)

    # ============================================================
    # Step 4: 把所有候选集编码成 SID
    # ============================================================
    # encode() 是"推理"过程:每物品 5 次最近邻查表 → 得到 SID
    # 返回 numpy.ndarray,shape = (N, num_layers) = (500, 5)
    # 数据类型 int64,值域 [0, 256)
    sids = builder.encode(ds.candidate_embs, ds.behaviors_per_item)

    # ============================================================
    # Step 5: 保存结果
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    # 存 npy(给 numpy/后续 Python 处理用)
    np.save(out_dir / "candidate_sids.npy", sids)
    # 存 json(给人看、做 feature store 入库)
    with open(out_dir / "candidate_sids.json", "w") as f:
        # 把 numpy 转 list 才能 JSON 序列化
        json.dump(
            {cid: sid.tolist() for cid, sid in zip(ds.candidate_ids, sids)},
            f, indent=2,
        )

    # ============================================================
    # Step 6: 简单自检
    # ============================================================
    # 唯一 SID 数量:把所有 5 元 tuple 放进 set,看 size
    # 越接近 N(500)说明区分度越好;实际生产 ~70% 后边际收益递减(Snapchat 经验)
    unique_tuples = len({tuple(s.tolist()) for s in sids})
    print(f"Built SID for {len(sids)} candidates")
    print(f"Unique SID tuples: {unique_tuples} / {len(sids)}")
    print(f"Sample: {ds.candidate_ids[0]} -> {sids[0].tolist()}")
    print(f"Saved to {out_dir}/candidate_sids.{{npy,json}}")


if __name__ == "__main__":
    main()
