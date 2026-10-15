"""
Comprehensive SID quality evaluation example.

【为什么这个 demo 存在】
================
光"训出来一个 SID"还不够 —— 你必须知道这个 SID 系统**到底好不好**,质量差在哪里,
具体哪些指标不达标、需不需要重训。本 demo 演示完整的"SID 质量评估 + 增强"流程。

【本文件做了什么】
================
    1. 训 baseline RQ-Kmeans,跑全套 11 个质量指标
    2. 用 HaMR (QuaSID) + NormVar 两个增强项重新调码本,看指标变化
    3. 演示 IntRR(变长 SID):不是每个物品都用满 L 层
    4. 输出 baseline vs 增强后的对比表
    5. 报告是否需要重训(自动决策)

【11 个指标一览(SIDEvaluator.full_report 产出)】
================
    - semantic_cohesion (SC)  : 同 SID 内聚性,QuaSID 验证与 Recall@10 Spearman=0.94
    - codeword_overlap (Oπ)    : DRQ 度量,越小越稳定
    - effective_capacity (K_eff): 实际可分辨的 code 数
    - code_utilization         : 每层被用到的 code 占比
    - ...等

【运行】
================
    python3 examples/run_sid_evaluation.py

【推荐先读】
================
    - docs/05_sid_evaluation_metrics.md(指标详细定义 + 公式 + 评级门槛)
    - src/sid_metrics.py(11 个核心指标实现)
    - src/hamr_loss.py(HaMR 损失)
    - src/quasid.py(NormVar + IntRR 等增强)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# fit_rq_kmeans:baseline RQ-Kmeans
# SIDEvaluator / SIDEvalReport:5 维度报告
# semantic_cohesion / preference_discrimination / codeword_overlap / ...
#   单独函数,你想直接算哪个都行
from src.rq_kmeans import fit_rq_kmeans
from src.sid_metrics import (
    SIDEvaluator, SIDEvalReport,
    semantic_cohesion, preference_discrimination,
    codeword_overlap, effective_capacity,
    code_utilization, code_entropy, code_gini,
    distance_correlation, collision_rate, retention_score,
)
# HaMRLoss : QuaSID 论文(arXiv 2604.11440)增强损失
# IntRRRedistributor:IntRR(arXiv 2602.20704)变长 SID 分配器
from src.hamr_loss import HaMRLoss
from src.quasid import IntRRRedistributor


def make_synthetic_data(N=2000, D=64, K=8, seed=0):
    """
    合成"8 簇 + 行为强度有差异"的假数据:
    - N    : 样本数
    - D    : embedding 维度
    - K    : 真实簇数(用来生成 ground truth SID 分布)
    - seed : 随机种子

    关键设计:
      - centers 是 8 个高斯中心点
      - 每个样本属于某个簇 + 噪声
      - interactions 按"所属簇"成比例 → 模拟"热门簇"的偏好分布

    这样我们就能验证指标 SC 是否能"找到"同簇样本(它应该 ≈ 1)。
    """
    np.random.seed(seed)
    centers = np.random.randn(K, D).astype(np.float32)
    assign = np.random.randint(0, K, N)
    # embs : (N, D),L2 归一化
    embs = centers[assign] + 0.15 * np.random.randn(N, D).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-9)

    # interactions:每个 item 的总交互数,簇间有差异
    interactions = {}
    for i in range(N):
        cluster = assign[i]
        interactions[i] = float(np.random.exponential(1.0) * (1 + (cluster % 5)))
    return embs, assign, interactions


def print_report(label: str, report: SIDEvalReport) -> None:
    """统一格式打印一份 SIDEvalReport,并给出自动重训建议。"""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(report.summary())
    needs, reasons = report.needs_retrain()
    print(f"\nRetrain decision: {needs}")
    for r in reasons:
        print(f"  - {r}")


def main():
    print("=" * 70)
    print("SID QUALITY EVALUATION DEMO")
    print("=" * 70)

    # ============================================================
    # Step 0:生成可解释的假数据
    # ============================================================
    # 2000 样本 × 64 维 × 4 层 × 32 个 codebook entry
    # 数据集故意生成 8 个真实簇,这样 SC 应该能验证通过
    N, D, L, K = 2000, 64, 4, 32
    embs, true_assign, interactions = make_synthetic_data(N, D, K=8)
    print(f"Dataset: N={N}, D={D}, L={L}, K={K}")
    print(f"True clusters: 8, true inter-cluster interaction gap: 5x")

    # ============================================================
    # Step 1:训练 baseline RQ-Kmeans(无任何增强)
    # ============================================================
    print("\n[1] Train baseline RQ-Kmeans")
    torch_em = torch.from_numpy(embs)
    # 训练:2000 个样本 × 4 层 × 32 个 codebook
    # n_passes=3 扫 3 遍数据, batch_size=128
    tok = fit_rq_kmeans(torch_em, num_layers=L, codebook_size=K,
                         n_passes=3, batch_size=128, verbose=False)
    sids = tok.encode_to_sid(torch_em).numpy()  # (2000, 4)
    # 把每层 codebook 提出来给评估器用
    codebooks = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)

    # 先看 3 个最直观的指标
    print(f"  Baseline codebook usage: {[f'{u*100:.1f}%' for u in code_utilization(sids, K)]}")
    print(f"  Baseline collision rate: {collision_rate(sids)*100:.2f}%")
    print(f"  Baseline SC: {semantic_cohesion(embs, sids, sample_size=2000):.4f}")

    # ============================================================
    # Step 2:跑完整的 5 维度 11 指标评估
    # ============================================================
    print("\n[2] Compute full quality report (BASELINE)")
    evalr = SIDEvaluator(codebook_size=K)
    # full_report 返回 SIDEvalReport:包含所有维度 + grade + needs_retrain
    report_baseline = evalr.full_report(embs, sids, interactions, codebooks=codebooks)
    print_report("BASELINE REPORT", report_baseline)

    # ============================================================
    # Step 3:用 HaMR + NormVar 两个增强项调码本
    # ============================================================
    # QuaSID 论文(arXiv 2604.11440)证明 HaMR 损失能把 NDCG@10 提升 +15~+20%
    # NormVar 是辅助正则,防止 codebook 塌缩
    print("\n[3] Apply HaMR loss + NormVar regularizer, retrain")
    import torch.nn.functional as F
    from src.quasid import QuaSIDRating, NormVarRegularizer

    rating = QuaSIDRating()
    norm_reg = NormVarRegularizer()
    hamr = HaMRLoss(margin=0.3, hamming_weight=0.5)

    # 把 baseline 的 codebook 当初始点,允许被梯度更新
    codebooks_hamr = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)
    codebooks_t = torch.from_numpy(codebooks_hamr).clone().requires_grad_(True)
    opt = torch.optim.Adam([codebooks_t], lr=1e-2)  # 只优化 codebook
    embs_t = torch.from_numpy(embs)

    # 手动训练循环:20 个"epoch",其实每个 epoch 都跑"全量编码 + 损失回传"
    for epoch in range(20):
        residual = embs_t.clone()
        layer_codes = []
        for l in range(L):
            cb = codebooks_t[l]
            # 用点积挑最近的 code(等价于 cosine 的单元素版本)
            sim = residual @ cb.t()
            codes = sim.argmax(dim=-1)
            layer_codes.append(codes)
            residual = residual - cb[codes]  # 更新残差
        sids_hamr = torch.stack(layer_codes, dim=1)

        # 损失 = 0.1 * HaMR + 0.01 * NormVar
        hamr_loss = hamr(embs_t, sids_hamr)
        norm_loss = norm_reg(codebooks_t[0]) + norm_reg(codebooks_t[1])
        norm_loss += norm_reg(codebooks_t[2]) + norm_reg(codebooks_t[3])
        loss = 0.1 * hamr_loss + 0.01 * norm_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 再评估一遍
    sids_hamr_np = sids_hamr.detach().numpy()
    codebooks_after = codebooks_t.detach().numpy()
    print_report("AFTER HaMR + NormVar", evalr.full_report(embs, sids_hamr_np, interactions,
                                                       codebooks=codebooks_after))

    # ============================================================
    # Step 4:IntRR 变长 SID
    # ============================================================
    # IntRR(arXiv 2602.20704)思路:
    #   高频物品用 L=4 层(全量表示)
    #   低频物品用 L=1 层(只给最粗的类别)
    # 这样能用更少 token 数表示更大物品池,稀疏存储效率更高
    print("\n[4] Apply IntRR: 1-token-per-item redistribution")
    # 准备 importance:按交互数归一化
    importances = np.array([interactions[i] / max(interactions.values())
                             for i in range(N)])
    redist = IntRRRedistributor(num_layers=L, codebook_size=K)
    # 给每个 item 分配"应该用几层"
    lengths = redist.assign_lengths(importances)
    print(f"  Avg length per item: {lengths.mean():.2f} (out of {L})")
    print(f"  Items with full 4 tokens: {(lengths == L).sum()}/{N}")
    print(f"  Items with 1 token: {(lengths == 1).sum()}/{N}")
    # 用 lengths 把原 SID 重打包(尾部 pad 0)
    packed = redist.pack_sids(sids, lengths)
    print(f"  Packed SIDs: {(packed == 0).sum()} pad tokens (out of {N*L})")

    # ============================================================
    # Step 5:对比表 baseline vs 增强后
    # ============================================================
    print("\n[5] Summary: comparison baseline vs after-enhancement")
    print(f"{'Metric':<32s}  {'Baseline':>10s}  {'After':>10s}  {'Delta':>10s}")
    print("-" * 70)
    # 对每个核心指标,计算 baseline vs after 数值和差值
    for field in ["semantic_cohesion", "codeword_overlap", "effective_capacity",
                  "collision_rate", "retention_score", "code_utilization"]:
        v_base = getattr(report_baseline, field)
        v_after = getattr(evalr.full_report(embs, sids_hamr_np, interactions,
                                            codebooks=codebooks_after), field)
        if isinstance(v_base, list):
            # code_utilization 是 list[4],逐个打印
            v_base_s = "[" + ",".join(f"{x:.2f}" for x in v_base) + "]"
            v_after_s = "[" + ",".join(f"{x:.2f}" for x in v_after) + "]"
            print(f"{field:<32s}  {v_base_s:>10s}  {v_after_s:>10s}")
        else:
            delta = v_after - v_base
            sign = "+" if delta > 0 else ""
            print(f"{field:<32s}  {v_base:>10.4f}  {v_after:>10.4f}  {sign}{delta:>+9.4f}")

    # ============================================================
    # Step 6:把全部报告落盘
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "sid_evaluation_report.json", "w") as f:
        json.dump({
            "baseline": report_baseline.to_dict(),
            "after_enhancement": evalr.full_report(embs, sids_hamr_np, interactions,
                                                  codebooks=codebooks_after).to_dict(),
            "intrr_avg_length": float(lengths.mean()),
            "intrr_pct_full_tokens": float((lengths == L).mean()),
        }, f, indent=2)
    print(f"\nSaved report to {out_dir}/sid_evaluation_report.json")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
