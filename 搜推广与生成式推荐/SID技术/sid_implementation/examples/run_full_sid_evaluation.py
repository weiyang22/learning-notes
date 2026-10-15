"""
SID 码本质量综合评估示例

【为什么这个 demo 存在】
================
评估一个 SID 系统,只算 1-2 个指标远远不够:
    - 只看碰撞率 → 可能 70% 但 SC=0.3(看起来唯一率不错,实际质量稀烂)
    - 只看 SC → 可能 SC=0.95 但碰撞率 80%(下游指标虚增)
必须**多维度交叉看**,才能稳定反映"下游效果"。

本 demo 按 5 维度 14 指标做综合评估:
    1. 收敛性:重建 MSE / 检索保留度 / 距离相关性
    2. 分布性:每层利用率 / 熵 / Gini / 困惑度
    3. 独立性:碰撞率 / L1 前缀碰撞率 / codeword overlap / 有效容量
    4. 语义性:SC(语义内聚) / PD(偏好区分)
    5. 重建性:距离相关性 / 检索保留度 / 量化 MSE

【本文件做了什么】
================
    1. 训 RQ-Kmeans + 训 RQ-VAE,产两套 SID + 两套 codebook
    2. RQ-Kmeans 走标准评估路径
    3. RQ-VAE 走"重写过的"评估路径(因它的重建 = encode+quantize+decode)
    4. 横向打印 8 个核心指标的对比
    5. 输出综合评级(A/B/C/D/F)和重训建议

【运行】
================
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 examples/run_full_sid_evaluation.py

(环境变量是 macOS / Linux 上避免 OpenMP 多线程冲突的标准做法)

【推荐先读】
================
    - docs/05_sid_evaluation_metrics.md(指标定义 + 阈值 + 评级门槛)
    - src/sid_evaluation.py(SIDQualityReport + SIDCodebookEvaluator)
    - src/sid_metrics.py(每个指标的具体计算)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# fit_rq_kmeans:RQ-Kmeans 主体
# RQVAE / RQVAEConfig / train_rqvae / encode_to_sid:RQ-VAE 主体
# SIDCodebookEvaluator:统一的"对 RQ-Kmeans"评估器
from src.rq_kmeans import fit_rq_kmeans
from src.rq_vae import RQVAE, RQVAEConfig, train_rqvae, encode_to_sid
from src.sid_evaluation import SIDCodebookEvaluator


def make_clustered_data(N=1000, D=64, K_clusters=8, noise=0.1, seed=42):
    """
    生成可解释的合成数据:
    - 8 个真实簇 → 我们知道 ground truth SID 应该是什么样
    - 不同簇的 interactions 有差异 → PD 指标可以被验证

    Returns:
        embs          : (N, D) L2-归一化的 embedding
        interactions  : Dict[item_id, 总交互数] — 用于计算 PD
        assign        : 每 item 的真实簇号
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((K_clusters, D)).astype(np.float32)
    assign = rng.integers(0, K_clusters, size=N)
    embs = centers[assign] + noise * rng.standard_normal((N, D)).astype(np.float32)
    # L2-归一化后,cosine 相似度 = 内积
    embs /= np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-9
    # 交互数跟簇成正比,模拟"热门簇"行为更多
    interactions = {i: float((assign[i] + 1) * 10) for i in range(N)}
    return embs, interactions, assign


def _evaluate_vae(evaluator, codebooks, embs, sids, full_recon):
    """
    RQ-VAE 专用评估路径。

    为什么不能直接调 SIDCodebookEvaluator?
        标准评估器假设"重建 = 直接 sum 各层 codeword"(RQ-Kmeans 的语义)
        但 RQ-VAE 是 encoder → latent → quantize → decoder 路径,codeword 在 latent 空间
        所以重建后的 full_recon 必须用 decoder 才能得到同维 embedding

    本函数与标准评估器的区别:
        标准:recon = codebook_lookup_only
        RQ-VAE:recon = decode(quantized_latent)
    """
    from src.sid_metrics import (
        semantic_cohesion, preference_discrimination,
        codeword_overlap, effective_capacity,
        code_utilization, code_entropy, code_gini,
        distance_correlation, collision_rate, retention_score,
    )
    from src.sid_evaluation import (
        SIDQualityReport, _compute_grade, _retrain_decision,
        _prefix_collision_rate,
    )

    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    codebooks = np.asarray(codebooks, dtype=np.float32)
    L, K = codebooks.shape[0], codebooks.shape[1]
    N = embs.shape[0]

    # 跟标准 evaluator 一样建一份空白报告,逐个填充
    report = SIDQualityReport(n_items=N, codebook_size=K, num_layers=L)
    # 收敛性:用全路径(encoder→quant→decoder)的重建 MSE
    report.convergence.reconstruction_mse = float(((embs - full_recon) ** 2).mean())
    report.convergence.retention_score = retention_score(
        embs, sids, codebooks=codebooks, k=10,
        sample_size=min(evaluator.sample_size, 200))
    report.convergence.distance_correlation = distance_correlation(
        embs, sids, sample_size=min(evaluator.sample_size, 200))

    # 分布性
    util = code_utilization(sids, K)
    ent = code_entropy(sids)
    gini = code_gini(sids)
    perp = [float(np.exp(e)) for e in ent]
    report.distribution.utilization_per_level = util
    report.distribution.entropy_per_level = ent
    report.distribution.gini_per_level = gini
    report.distribution.perplexity_per_level = perp

    # 独立性
    report.independence.collision_rate = collision_rate(sids)
    report.independence.prefix_collision_rate = _prefix_collision_rate(sids, level=0)
    sid_counts = {}
    for s in sids:
        key = tuple(s.tolist())
        sid_counts[key] = sid_counts.get(key, 0) + 1
    report.independence.max_items_per_sid = max(sid_counts.values()) if sid_counts else 0
    ov = codeword_overlap(embs, sids, epsilon=evaluator.epsilon)
    report.independence.codeword_overlap = ov
    report.independence.effective_capacity = effective_capacity(K, ov)

    # 语义性
    report.semantic.semantic_cohesion = semantic_cohesion(
        embs, sids, sample_size=min(evaluator.sample_size, 5000))
    report.semantic.preference_discrimination = preference_discrimination(
        sids, evaluator.interaction_lookup,
        sample_pairs=min(evaluator.sample_size, 5000))

    # 重建性(留空,跟 convergence 维度重复)
    report.reconstruction.distance_correlation = report.convergence.distance_correlation
    report.reconstruction.retention_score = report.convergence.retention_score

    # 自动评级 + 重训决策
    report.grade = _compute_grade(report)
    report.needs_retrain, report.retrain_reasons = _retrain_decision(report)
    return report


def main():
    print("=" * 70)
    print("  SID 码本质量综合评估 (5 维度 14 指标)")
    print("=" * 70)

    # ============================================================
    # Step 0:生成数据
    # ============================================================
    N, D = 1000, 64
    embs, interactions, true_clusters = make_clustered_data(N=N, D=D)
    print(f"\n合成数据: N={N}, D={D}, 真实簇数=8")

    # ============================================================
    # Step 1:训 RQ-Kmeans 并评估
    # ============================================================
    print("\n[1] RQ-Kmeans 训练并评估")
    L, K = 4, 32
    tok = fit_rq_kmeans(torch.from_numpy(embs),
                         num_layers=L, codebook_size=K,
                         n_passes=4, batch_size=128, verbose=False)
    sids_km = tok.encode_to_sid(torch.from_numpy(embs)).numpy()
    codebooks_km = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)

    # SIDCodebookEvaluator 内部会算齐 5 维度 14 指标
    evaluator = SIDCodebookEvaluator(
        codebook_size=K,
        interaction_lookup=interactions,
        sample_size=500,
    )
    report_km = evaluator.evaluate(codebooks_km, embs, sids_km)
    print(report_km.summary())

    # ============================================================
    # Step 2:训 RQ-VAE 并评估(走 RQ-VAE 专用重建路径)
    # ============================================================
    print("\n\n[2] RQ-VAE 训练并评估")
    cfg = RQVAEConfig(input_dim=D, hidden_dim=D, latent_dim=D,
                      num_layers=L, codebook_size=K)
    vae = RQVAE(cfg)
    # 训 20 epoch;RQ-VAE 比 RQ-Kmeans 慢
    train_rqvae(vae, torch.from_numpy(embs),
                epochs=20, batch_size=128, lr=1e-3, log_every=100, verbose=False)
    sids_vae = encode_to_sid(vae, torch.from_numpy(embs)).numpy()

    # 全路径重建:encode → quantize → decode
    # 这一步是 RQ-VAE 与 RQ-Kmeans 的本质区别
    vae.eval()
    with torch.no_grad():
        z = vae.encode_z(torch.from_numpy(embs).float())          # (N, D) latent
        z_q = vae.quantizer.decode(torch.from_numpy(sids_vae).long())  # 把 SID 反量化
        sid_recon_vae = vae.decode_z(z_q).numpy()                 # 用 decoder 反向
    codebooks_vae = vae.quantizer.codebooks.detach().cpu().numpy()

    # 复用 evaluator 但走"VAE 专用"评估路径(见上面 _evaluate_vae)
    report_vae = _evaluate_vae(evaluator, codebooks_vae, embs,
                              sids_vae, sid_recon_vae)
    print(report_vae.summary())

    # ============================================================
    # Step 3:横向对比
    # ============================================================
    print("\n\n[3] 对比报告")
    print(f"{'指标':<35s}  {'RQ-Kmeans':>10s}  {'RQ-VAE':>10s}  {'更优':>8s}")
    print("-" * 70)
    # 选 8 个最值得对比的指标,标"更高更好"或"更低更好"
    metrics_for_compare = [
        ("重建 MSE (收敛性)", "convergence.reconstruction_mse", "lower"),
        ("检索保留度 (重建性)", "reconstruction.retention_score", "higher"),
        ("SC 语义内聚 (语义性)", "semantic.semantic_cohesion", "higher"),
        ("平均利用率 (分布性)", "distribution.mean_utilization", "higher"),
        ("平均熵 (分布性)", "distribution.mean_entropy", "higher"),
        ("碰撞率 (独立性)", "independence.collision_rate", "lower"),
        ("Codeword 重叠 Oπ", "independence.codeword_overlap", "lower"),
        ("最大单 SID 物品数", "independence.max_items_per_sid", "lower"),
    ]
    for label, path, better in metrics_for_compare:
        v_km = _get(report_km, path)
        v_vae = _get(report_vae, path)
        v_km_s = _format(v_km)
        v_vae_s = _format(v_vae)
        if isinstance(v_km, float) and isinstance(v_vae, float):
            if better == "lower":
                winner = "Kmeans" if v_km < v_vae else "VAE"
            else:
                winner = "Kmeans" if v_km > v_vae else "VAE"
        else:
            winner = "-"
        print(f"{label:<35s}  {v_km_s:>10s}  {v_vae_s:>10s}  {winner:>8s}")

    print(f"\n综合评级: RQ-Kmeans = {report_km.grade},  RQ-VAE = {report_vae.grade}")
    print(f"需要重训: RQ-Kmeans = {report_km.needs_retrain},  RQ-VAE = {report_vae.needs_retrain}")

    # ============================================================
    # Step 4:落盘
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "full_sid_evaluation.json", "w") as f:
        json.dump({
            "rq_kmeans": report_km.to_dict(),
            "rq_vae": report_vae.to_dict(),
            "comparison": {
                "kmeans_grade": report_km.grade,
                "vae_grade": report_vae.grade,
                "kmeans_needs_retrain": report_km.needs_retrain,
                "vae_needs_retrain": report_vae.needs_retrain,
            },
        }, f, indent=2)
    print(f"\n报告已保存: {out_dir}/full_sid_evaluation.json")

    print("\n" + "=" * 70)
    print("  DEMO 完成")
    print("=" * 70)


def _get(obj, path: str):
    """按点路径拿嵌套字段,例如 'convergence.reconstruction_mse'"""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _format(v) -> str:
    """统一格式输出:float 截短 / 列表原样转 str"""
    if isinstance(v, float):
        if abs(v) > 100:
            return f"{v:.1f}"
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
