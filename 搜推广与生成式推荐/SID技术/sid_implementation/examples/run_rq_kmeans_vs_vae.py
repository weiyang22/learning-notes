"""
Compare RQ-VAE vs RQ-Kmeans on the same embeddings.

【为什么做这个对比】
================
TRM 论文(arXiv 2601.22694)§3.2 用的就是 RQ-Kmeans(纯无监督,无 encoder/decoder)。
本项目在用户要求下额外实现了 RQ-VAE 版本。本文件做以下对比:

    ┌──────────────────────────────────────────────────────────┐
    │  对比维度        │  RQ-VAE                       │  RQ-Kmeans  │
    ├──────────────────────────────────────────────────────────┤
    │  训练成本        │  高(20 epoch × encoder+decoder) │  低(3 passes mini-batch) │
    │  重建 MSE        │  更低(学 latent 空间)            │  较高(直接在 emb 空间) │
    │  是否需要 GPU     │  显著受益                     │  CPU 也行                  │
    │  codebook 更新    │  STE + EMA                    │  Sculley 增量更新         │
    │  论文原算法       │  否                           │  是                        │
    │  用户偏好       │  ★(本仓库默认)                │  备选                       │
    └──────────────────────────────────────────────────────────┘

【本文件做了什么】
================
    1. 2000 个候选集 + 100 用户 + 每物品平均 10 行为
    2. 行为融合 (fuse_candidate_batch) 后得到 embs_fused
    3. 用同一份 embs_fused 训练两个 tokenizer:
       (a) RQ-VAE (RQVAE + train_rqvae)
       (b) RQ-Kmeans (fit_rq_kmeans)
    4. 比较:
       - 训练 + 编码耗时
       - 重建 MSE
       - 每个位置上 SID 一致的比例(per-position overlap)
       - 每层 codebook 利用率
    5. 把结果存到 outputs/rq_comparison.json

【运行】
================
    python3 examples/run_rq_kmeans_vs_vae.py

【推荐先读】
================
    - docs/00_sid_overview_and_principles.md § 4.3 / 4.4
    - docs/01 § 4.2(RQ-VAE 架构)
    - src/rq_kmeans.py / src/rq_vae.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch

# make_synthetic_dataset:生成 2000 物品的合成数据(够多才能看出 RQ-VAE 优势)
# RQVAE / RQVAEConfig / train_rqvae / encode_to_sid:RQ-VAE 主体
# fit_rq_kmeans / encode_with_rq_kmeans:RQ-Kmeans 主体
from src.data_loader import make_synthetic_dataset
from src.rq_vae import RQVAE, RQVAEConfig, train_rqvae, encode_to_sid
from src.rq_kmeans import fit_rq_kmeans, encode_with_rq_kmeans


def main():
    # ============================================================
    # Step 1: 生成 2000 物品的合成数据(数据量稍大,显出差异)
    # ============================================================
    ds = make_synthetic_dataset(
        n_candidates=2000, n_queries=0, n_users=200,
        emb_dim=128, behaviors_per_item_mean=10, seed=42,
    )

    # ============================================================
    # Step 2: 行为融合 → 协同 embedding
    # ============================================================
    # fuse_candidate_batch 批量融合:content_emb + α * 行为向量
    # RQ-VAE 和 RQ-Kmeans 都要吃同一份"行为融合后的" embedding 才能公平对比
    from src.behavior_fusion import fuse_candidate_batch
    embs_fused = fuse_candidate_batch(ds.candidate_embs, ds.behaviors_per_item, alpha=0.4)
    embs_t = torch.from_numpy(embs_fused).float()
    # embs_t shape = (2000, 128)

    # ============================================================
    # Step 3: 训练 RQ-VAE 并编码 SID
    # ============================================================
    print("=" * 70)
    print("RQ-VAE")
    print("=" * 70)
    t0 = time.time()
    cfg = RQVAEConfig(input_dim=128, latent_dim=64, num_layers=5,
                      codebook_size=256, hidden_dim=128)
    # 5 层 × 256 = 1.1e12 容量
    vae = RQVAE(cfg)
    # 训练 15 epoch,每个 epoch 把 2000 个样本分 batch_size=128 个 batch
    train_rqvae(vae, embs_t, epochs=15, batch_size=128, lr=1e-3, log_every=5)
    # 训练完后 encode:SID shape (2000, 5)
    sid_vae = encode_to_sid(vae, embs_t)
    t_vae = time.time() - t0

    # ============================================================
    # Step 4: 重建 RQ-VAE 的 embedding 并算 MSE
    # ============================================================
    # 重建路径:orig_emb → encoder → latent_z → quantize → decoder → recon_emb
    # MSE 小 = RQ-VAE 把信息保留了
    with torch.no_grad():
        z = vae.encode_z(embs_t)             # (2000, 64) latent
        z_q = vae.quantizer.decode(sid_vae)  # 把 SID 反量化回 latent
        recon = vae.decode_z(z_q)            # decoder 反向映射回 128 维
        mse_vae = float(((embs_t - recon) ** 2).mean().item())
    print(f"RQ-VAE train+encode time: {t_vae:.1f}s, MSE: {mse_vae:.4f}")

    # ============================================================
    # Step 5: 训练 RQ-Kmeans 并编码 SID
    # ============================================================
    print("=" * 70)
    print("RQ-Kmeans")
    print("=" * 70)
    t0 = time.time()
    # fit_rq_kmeans: 2000 个 embedding × 5 层
    # batch_size=256 比 RQ-VAE 的 128 大,因为 RQ-Kmeans 没 encoder 显存小
    tok = fit_rq_kmeans(embs_t, num_layers=5, codebook_size=256,
                        n_passes=3, batch_size=256, verbose=True)
    sid_kmeans = encode_with_rq_kmeans(tok, embs_t)
    t_kmeans = time.time() - t0

    # ============================================================
    # Step 6: 重建 RQ-Kmeans 的 embedding 并算 MSE
    # ============================================================
    # RQ-Kmeans 没有 encoder/decoder,重建 = 直接累加 5 个 codeword
    with torch.no_grad():
        recon = tok.decode_from_sid(sid_kmeans)
        mse_kmeans = float(((embs_t - recon) ** 2).mean().item())
    print(f"RQ-Kmeans train+encode time: {t_kmeans:.1f}s, MSE: {mse_kmeans:.4f}")

    # ============================================================
    # Step 7: 横向对比,打印关键指标
    # ============================================================
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    # 1. 每个位置上 SID 一致的比例 → 看两种 RQ 是不是学到了相似的"语义剖面"
    overlap = (sid_vae == sid_kmeans).float().mean().item()
    print(f"  Per-position exact match: {overlap*100:.2f}%")
    print(f"  RQ-VAE MSE: {mse_vae:.4f}")
    print(f"  RQ-Kmeans MSE: {mse_kmeans:.4f}")
    # 2. 速度比:RQ-VAE 通常慢 2-10x
    print(f"  Speedup RQ-VAE / RQ-Kmeans: {t_vae/t_kmeans:.1f}x")
    # 3. 每层 codebook 利用率(被用过的 code 数量)
    vae_usage = []
    for layer in range(cfg.num_layers):
        ids = sid_vae[:, layer]
        used = int(torch.unique(ids).numel())
        vae_usage.append(used)
    print(f"  RQ-VAE codebook usage per level: {vae_usage}")
    print(f"  RQ-Kmeans codebook usage per level: {tok.codebook_usage()}")

    # ============================================================
    # Step 8: 保存对比报告(给调参 / 选型用)
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "rq_vae_sids.npy", sid_vae.numpy())
    np.save(out_dir / "rq_kmeans_sids.npy", sid_kmeans.numpy())
    with open(out_dir / "rq_comparison.json", "w") as f:
        json.dump({
            "rq_vae": {"mse": mse_vae, "train_time_sec": t_vae,
                       "shape": list(sid_vae.shape)},
            "rq_kmeans": {"mse": mse_kmeans, "train_time_sec": t_kmeans,
                          "shape": list(sid_kmeans.shape),
                          "codebook_usage": tok.codebook_usage()},
            "overlap_pct": overlap * 100,
            "speedup_vae_over_kmeans": t_vae / t_kmeans,
        }, f, indent=2)
    print(f"Saved comparison to {out_dir}/rq_comparison.json")


if __name__ == "__main__":
    main()
