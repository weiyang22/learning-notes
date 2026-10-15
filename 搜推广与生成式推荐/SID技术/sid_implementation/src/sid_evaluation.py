"""
SID 码本质量统一评估框架

本模块综合主流 SID 评估方法 (来自 QuaSID, DRQ, semantic-id lib, Eugene Yan 等),
按 5 个维度对训练好的 SID 码本进行全方位评估:

1. **收敛性 (Convergence)**: 训练是否学好 - 重建 MSE / retention
2. **分布性 (Distribution)**: 分布是否均匀 - 利用率 / 熵 / Gini / perplexity
3. **独立性 (Independence)**: 是否过度碰撞 - collision rate / prefix collision
4. **语义性 (Semantics)**: 是否符合用户偏好 - SC / PD / 距离相关
5. **重建性 (Reconstruction)**: 几何保持度 - 距离相关 / O_π / K_eff / 击中率

使用方法:
    >>> evaluator = SIDCodebookEvaluator(codebook_size=256)
    >>> report = evaluator.evaluate(codebooks, item_embeddings, sids)
    >>> print(report.summary())

参考论文:
    - QuaSID (arxiv 2604.11440) - SC, PD, QuaSID 评分
    - DRQ (arxiv 2606.01844) - O_π, K_eff, 诊断
    - Reliability paper (arxiv 2605.25330) - 碰撞感知指标
    - semantic-id PyPI lib - 利用率 / 熵 / 距离相关
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# 让脚本可以直接 python -m src.sid_evaluation 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

# 从单点指标库导入, 这里做"组装"
from src.sid_metrics import (
    semantic_cohesion, preference_discrimination,
    codeword_overlap, effective_capacity,
    code_utilization, code_entropy, code_gini,
    distance_correlation, collision_rate, retention_score,
)


# 类型别名
ArrayLike = Union[np.ndarray, Sequence]


@dataclass
class ConvergenceMetrics:
    """收敛性指标: 训练是否学好。

    字段:
        reconstruction_mse: (N, D) embedding 重建误差, 越低越好
        retention_score: SID 检索 vs 原 embedding 检索的命中率, 越高越好
        distance_correlation: Spearman 相关 (embedding 距离 vs SID Hamming)
    """
    reconstruction_mse: float = 0.0
    retention_score: float = 0.0
    distance_correlation: float = 0.0

    def to_dict(self) -> dict:
        return {
            "reconstruction_mse": self.reconstruction_mse,
            "retention_score": self.retention_score,
            "distance_correlation": self.distance_correlation,
        }


@dataclass
class DistributionMetrics:
    """分布性指标: code 分配是否均匀。

    字段:
        utilization_per_level: 每层利用率 (0~1)
        entropy_per_level: 每层 Shannon 熵 (nats)
        gini_per_level: 每层 Gini 系数 (0~1, 越低越均匀)
        perplexity_per_level: 每层困惑度 (= exp(entropy), 越接近 K 越好)
    """
    utilization_per_level: List[float] = field(default_factory=list)
    entropy_per_level: List[float] = field(default_factory=list)
    gini_per_level: List[float] = field(default_factory=list)
    perplexity_per_level: List[float] = field(default_factory=list)

    @property
    def mean_utilization(self) -> float:
        """平均利用率, 跨所有层。"""
        return float(np.mean(self.utilization_per_level)) if self.utilization_per_level else 0.0

    @property
    def mean_entropy(self) -> float:
        """平均熵, 跨所有层。"""
        return float(np.mean(self.entropy_per_level)) if self.entropy_per_level else 0.0

    def to_dict(self) -> dict:
        return {
            "utilization_per_level": [round(u, 4) for u in self.utilization_per_level],
            "entropy_per_level": [round(e, 4) for e in self.entropy_per_level],
            "gini_per_level": [round(g, 4) for g in self.gini_per_level],
            "perplexity_per_level": [round(p, 4) for p in self.perplexity_per_level],
            "mean_utilization": round(self.mean_utilization, 4),
            "mean_entropy": round(self.mean_entropy, 4),
        }


@dataclass
class IndependenceMetrics:
    """独立性指标: SID 碰撞程度。

    字段:
        collision_rate: 共享 SID 的物品比例, 越低越好
        prefix_collision_rate: L1 前缀碰撞率 (同前缀但完整 SID 不同), 越低越好
        max_items_per_sid: 单 SID 最多物品数 (>5 触发告警)
        codeword_overlap: O_π ∈ [0, 1], 越低越好
        effective_capacity: K_eff ∈ [1, K], 越接近 K 越好
    """
    collision_rate: float = 0.0
    prefix_collision_rate: float = 0.0
    max_items_per_sid: int = 0
    codeword_overlap: float = 0.0
    effective_capacity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "collision_rate": self.collision_rate,
            "prefix_collision_rate": self.prefix_collision_rate,
            "max_items_per_sid": self.max_items_per_sid,
            "codeword_overlap": self.codeword_overlap,
            "effective_capacity": self.effective_capacity,
        }


@dataclass
class SemanticMetrics:
    """语义性指标: SID 是否反映用户偏好。

    字段:
        semantic_cohesion (SC): 同 SID 簇内 cosine 相似度均值, 越高越好
        preference_discrimination (PD): 不同 SID 簇偏好差异, 越高越好
    """
    semantic_cohesion: float = 0.0
    preference_discrimination: float = 0.0

    def to_dict(self) -> dict:
        return {
            "semantic_cohesion": self.semantic_cohesion,
            "preference_discrimination": self.preference_discrimination,
        }


@dataclass
class ReconstructionMetrics:
    """重建性指标: 几何保持度 (与收敛性部分重叠, 专门看下游检索保留)。

    字段:
        distance_correlation: Spearman 相关
        retention_score: HR_sid / HR_orig
    """
    distance_correlation: float = 0.0
    retention_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "distance_correlation": self.distance_correlation,
            "retention_score": self.retention_score,
        }


@dataclass
class SIDQualityReport:
    """综合 SID 质量评估报告。

    字段:
        n_items: 评估物品数
        codebook_size: 每层码本大小 K
        num_layers: SID 层数 L
        convergence / distribution / independence / semantic / reconstruction: 五维度指标
        grade: 综合评级 A/B/C/D/F
        needs_retrain: 是否需要重训
        retrain_reasons: 重训触发原因列表
    """
    n_items: int = 0
    codebook_size: int = 0
    num_layers: int = 0
    convergence: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)
    distribution: DistributionMetrics = field(default_factory=DistributionMetrics)
    independence: IndependenceMetrics = field(default_factory=IndependenceMetrics)
    semantic: SemanticMetrics = field(default_factory=SemanticMetrics)
    reconstruction: ReconstructionMetrics = field(default_factory=ReconstructionMetrics)
    grade: str = "F"
    needs_retrain: bool = True
    retrain_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "codebook_size": self.codebook_size,
            "num_layers": self.num_layers,
            "convergence": self.convergence.to_dict(),
            "distribution": self.distribution.to_dict(),
            "independence": self.independence.to_dict(),
            "semantic": self.semantic.to_dict(),
            "reconstruction": self.reconstruction.to_dict(),
            "grade": self.grade,
            "needs_retrain": self.needs_retrain,
            "retrain_reasons": self.retrain_reasons,
        }

    def summary(self) -> str:
        """人类可读的简明报告 (中文版), 适合贴到 PR / 监控告警里。"""
        lines = [
            "=" * 70,
            f"  SID 码本质量评估报告  (N={self.n_items}, K={self.codebook_size}, L={self.num_layers})",
            "=" * 70,
            f"  综合等级: {self.grade}    需要重训: {self.needs_retrain}",
            "",
            "▌ 收敛性 (训练是否学好)",
            f"  重建 MSE        : {self.convergence.reconstruction_mse:.4f}    (越低越好)",
            f"  检索保留度       : {self.convergence.retention_score:.4f}    (越高越好, 接近 1.0)",
            f"  距离相关性       : {self.convergence.distance_correlation:.4f}    (越高越好)",
            "",
            "▌ 分布性 (code 分布是否均匀)",
            f"  每层利用率       : {[f'{u:.1%}' for u in self.distribution.utilization_per_level]}    (越高越好, >70%)",
            f"  每层 Gini 系数   : {[f'{g:.3f}' for g in self.distribution.gini_per_level]}    (越低越好, <0.3)",
            f"  每层 熵 (nats)    : {[f'{e:.3f}' for e in self.distribution.entropy_per_level]}",
            f"  每层困惑度       : {[f'{p:.1f}' for p in self.distribution.perplexity_per_level]}    (越接近 K 越好)",
            f"  平均利用率       : {self.distribution.mean_utilization:.1%}",
            "",
            "▌ 独立性 (碰撞程度)",
            f"  全 SID 碰撞率    : {self.independence.collision_rate:.2%}    (越低越好, <1%)",
            f"  L1 前缀碰撞率    : {self.independence.prefix_collision_rate:.2%}    (越低越好)",
            f"  单 SID 最大物品数: {self.independence.max_items_per_sid}",
            f"  Codeword 重叠 Oπ : {self.independence.codeword_overlap:.4f}    (越低越好)",
            f"  有效容量 K_eff   : {self.independence.effective_capacity:.1f} / {self.codebook_size}    (越接近 K 越好)",
            "",
            "▌ 语义性 (用户偏好区分度)",
            f"  语义内聚性 SC    : {self.semantic.semantic_cohesion:.4f}    (越高越好, >0.6)",
            f"  偏好区分度 PD    : {self.semantic.preference_discrimination:.4f}    (越高越好, >0.3)",
            "",
            "▌ 重建性 (几何保持)",
            f"  距离相关性       : {self.reconstruction.distance_correlation:.4f}",
            f"  检索保留度       : {self.reconstruction.retention_score:.4f}",
        ]
        if self.needs_retrain:
            lines.append("")
            lines.append("▌ 重训触发原因")
            for r in self.retrain_reasons:
                lines.append(f"  ⚠ {r}")
        return "\n".join(lines)


class SIDCodebookEvaluator:
    """综合 SID 码本质量评估器 (5 维度 14 项指标)。

    Args:
        codebook_size: 每层 codebook 大小 K (例如 256)。
        interaction_lookup: 可选, {item_idx: click_count} 用于计算 PD。
        sample_size: 采样大小 (避免 O(N²) 计算)。
        epsilon: codeword_overlap 检索扰动阈值。
    """

    def __init__(self, codebook_size: int = 256,
                 interaction_lookup: Optional[Dict[int, float]] = None,
                 sample_size: int = 1000,
                 epsilon: float = 0.3):
        self.codebook_size = codebook_size
        self.interaction_lookup = interaction_lookup or {}
        self.sample_size = sample_size
        self.epsilon = epsilon

    def evaluate(self,
                 codebooks: np.ndarray,
                 item_embeddings: ArrayLike,
                 sids: ArrayLike) -> SIDQualityReport:
        """对训练好的 SID 码本进行全维度评估。

        Args:
            codebooks: (L, K, D) 训练好的码本 (numpy array)。
            item_embeddings: (N, D) 物品 embedding。
            sids: (N, L) 整数 SID 序列。

        Returns:
            SIDQualityReport 含 5 维度 14 项指标 + 综合评级 + 重训建议。
        """
        # ----------- 准备数据 -----------
        embs = np.asarray(item_embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        sids = np.asarray(sids, dtype=np.int64)
        if sids.ndim == 1:
            sids = sids.reshape(-1, 1)
        codebooks = np.asarray(codebooks, dtype=np.float32)
        L, K = codebooks.shape[0], codebooks.shape[1]
        N = embs.shape[0]

        report = SIDQualityReport(
            n_items=N,
            codebook_size=K,
            num_layers=L,
        )

        # ----------- 收敛性 -----------
        # 用 codebook 重建 SID 向量, 算 MSE
        sid_recon = np.zeros_like(embs)
        for l in range(L):
            sid_recon += codebooks[l, sids[:, l]]
        mse = float(((embs - sid_recon) ** 2).mean())
        report.convergence.reconstruction_mse = mse
        # retention (SID 检索 vs 原检索的命中率)
        report.convergence.retention_score = retention_score(
            embs, sids, codebooks=codebooks, k=10,
            sample_size=min(self.sample_size, 200))
        # distance correlation
        report.convergence.distance_correlation = distance_correlation(
            embs, sids, sample_size=min(self.sample_size, 200))

        # ----------- 分布性 -----------
        util = code_utilization(sids, K)
        ent = code_entropy(sids)
        gini = code_gini(sids)
        # perplexity = exp(entropy)
        perp = [float(np.exp(e)) for e in ent]
        report.distribution.utilization_per_level = util
        report.distribution.entropy_per_level = ent
        report.distribution.gini_per_level = gini
        report.distribution.perplexity_per_level = perp

        # ----------- 独立性 -----------
        report.independence.collision_rate = collision_rate(sids)
        # L1 前缀碰撞率: 同前缀但完整 SID 不同
        report.independence.prefix_collision_rate = _prefix_collision_rate(sids, level=0)
        # 最大单 SID 物品数
        sid_counts = Counter(map(tuple, sids.tolist()))
        report.independence.max_items_per_sid = max(sid_counts.values()) if sid_counts else 0
        # codeword overlap O_π
        ov = codeword_overlap(embs, sids, epsilon=self.epsilon)
        report.independence.codeword_overlap = ov
        # effective capacity K_eff = K / (1 + (K-1) * O_π)
        report.independence.effective_capacity = effective_capacity(K, ov)

        # ----------- 语义性 -----------
        report.semantic.semantic_cohesion = semantic_cohesion(
            embs, sids, sample_size=min(self.sample_size, 5000))
        report.semantic.preference_discrimination = preference_discrimination(
            sids, self.interaction_lookup,
            sample_pairs=min(self.sample_size, 5000))

        # ----------- 重建性 (与收敛性部分重叠, 专门看下游检索保留) -----------
        report.reconstruction.distance_correlation = report.convergence.distance_correlation
        report.reconstruction.retention_score = report.convergence.retention_score

        # ----------- 综合评级 + 重训决策 -----------
        report.grade = _compute_grade(report)
        report.needs_retrain, report.retrain_reasons = _retrain_decision(report)

        return report


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _prefix_collision_rate(sids: np.ndarray, level: int = 0) -> float:
    """L1 前缀碰撞率: 前 L 层相同但整体 SID 不同的占比。

    衡量"前缀相同但后层不同"的歧义程度,
    高 = 子树高度共享但叶子冲突 → 影响 prefix 检索质量。
    """
    if sids.shape[1] < 2:
        return 0.0
    prefixes = sids[:, :level + 1]
    suffix_cols = list(range(level + 1, sids.shape[1]))
    if not suffix_cols:
        return 0.0
    # 同前缀 → 不同后缀 的映射
    prefix_to_suffixes = {}
    for i in range(len(sids)):
        key = tuple(prefixes[i].tolist())
        suffix = tuple(sids[i, suffix_cols].tolist())
        prefix_to_suffixes.setdefault(key, set()).add(suffix)

    collisions = 0
    total = 0
    # 对每个前缀, 看后缀集合大小 (1 = 无碰撞, >1 = 碰撞)
    for prefixes_set in prefix_to_suffixes.values():
        n = len(prefixes_set)
        if n > 1:
            total += 1
            collisions += 1
    return float(collisions / total) if total > 0 else 0.0


def _compute_grade(report: SIDQualityReport) -> str:
    """综合评级 (A/B/C/D/F)。

    按 5 个维度的关键指标加权打分:
      A: 所有指标在合理阈值内
      F: 至少 3 个维度严重不达标
    """
    score = 0
    total = 0

    # 收敛性 (MSE < 0.05 满分, > 0.5 零分)
    total += 1
    if report.convergence.reconstruction_mse < 0.05:
        score += 1
    elif report.convergence.reconstruction_mse < 0.2:
        score += 0.5

    # 分布性 (平均利用率 > 0.7 满分)
    total += 1
    if report.distribution.mean_utilization > 0.7:
        score += 1
    elif report.distribution.mean_utilization > 0.4:
        score += 0.5

    # 独立性 (碰撞率 < 1% 满分, > 50% 零分)
    total += 1
    if report.independence.collision_rate < 0.01:
        score += 1
    elif report.independence.collision_rate < 0.3:
        score += 0.5

    # 语义性 (SC > 0.6 满分)
    total += 1
    if report.semantic.semantic_cohesion > 0.6:
        score += 1
    elif report.semantic.semantic_cohesion > 0.3:
        score += 0.5

    # 重建性 (retention > 0.7 满分)
    total += 1
    if report.reconstruction.retention_score > 0.7:
        score += 1
    elif report.reconstruction.retention_score > 0.4:
        score += 0.5

    final_score = score / max(total, 1)
    if final_score >= 0.85:
        return "A"
    if final_score >= 0.65:
        return "B"
    if final_score >= 0.45:
        return "C"
    if final_score >= 0.25:
        return "D"
    return "F"


def _retrain_decision(report: SIDQualityReport
                      ) -> Tuple[bool, List[str]]:
    """基于指标阈值判断是否需要重训。

    阈值参考业界主流论文 (QuaSID Table 4, OneRec production guide):
      - 重建 MSE < 0.2 (否则训练未收敛)
      - 平均利用率 > 70%
      - 碰撞率 < 5% (否则下游检索会被虚增)
      - SC > 0.5 (否则 SID 不聚类)
      - Retention > 0.5 (否则 SID 检索保真度差)
      - 有效容量 / K > 50% (否则一半 code 是死的)
    """
    reasons = []
    r = report.convergence
    if r.reconstruction_mse > 0.2:
        reasons.append(f"重建 MSE {r.reconstruction_mse:.3f} > 0.2 (训练未收敛)")
    if r.retention_score < 0.5:
        reasons.append(f"保留度 {r.retention_score:.3f} < 0.5 (SID 检索保真差)")

    d = report.distribution
    if d.mean_utilization < 0.7:
        reasons.append(
            f"平均码本利用率 {d.mean_utilization:.1%} < 70% "
            f"(死 code 过多)"
        )

    i = report.independence
    if i.collision_rate > 0.05:
        reasons.append(
            f"碰撞率 {i.collision_rate:.2%} > 5% (下游指标会被虚增)"
        )
    if i.effective_capacity < report.codebook_size * 0.5:
        K = report.codebook_size
        reasons.append(
            f"有效容量 {i.effective_capacity:.1f} / K={K} < 50% "
            f"(近半数 code 实际无用)"
        )

    s = report.semantic
    if s.semantic_cohesion < 0.5:
        reasons.append(
            f"语义内聚 SC {s.semantic_cohesion:.3f} < 0.5 "
            f"(簇内不一致)"
        )

    if report.codebook_size > 0 and i.codeword_overlap > 0.1:
        reasons.append(
            f"Codeword 重叠 Oπ {i.codeword_overlap:.3f} > 0.1 "
            f"(扰动下不稳定)"
        )

    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# 自测: 演示完整评估流程
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """演示: 训练 RQ-Kmeans 后用综合评估器打分。"""
    import torch
    from src.rq_kmeans import fit_rq_kmeans

    torch.manual_seed(0)
    np.random.seed(0)

    # 生成聚类数据 (8 个真实簇 + 高斯噪声)
    K_clusters, D, K, L = 8, 32, 32, 4
    N = 600
    centers = np.random.randn(K_clusters, D).astype(np.float32)
    assign = np.random.randint(0, K_clusters, N)
    embs = centers[assign] + 0.15 * np.random.randn(N, D).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=-1, keepdims=True)

    # 用真实 cluster_assign 模拟交互次数
    interactions = {i: float((assign[i] + 1) * 5) for i in range(N)}

    # 训练 RQ-Kmeans
    tok = fit_rq_kmeans(torch.from_numpy(embs),
                         num_layers=L, codebook_size=K,
                         n_passes=4, batch_size=128, verbose=False)
    sids = tok.encode_to_sid(torch.from_numpy(embs)).numpy()
    codebooks = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)

    # 综合评估
    evaluator = SIDCodebookEvaluator(
        codebook_size=K,
        interaction_lookup=interactions,
        sample_size=500,
    )
    report = evaluator.evaluate(codebooks, embs, sids)
    print(report.summary())
    print()
    print("=== 完整报告 (dict) ===")
    import json
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))