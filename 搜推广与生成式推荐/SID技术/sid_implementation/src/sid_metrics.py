"""
SID 质量评估指标库

综合主流 SID 评估方法 (来自 QuaSID, DRQ, semantic-id lib, Eugene Yan 等),
提供单点指标 + 一键评估器, 共 14 项指标 + 综合评级 + 重训决策。

按维度分组:
  收敛性 (Convergence):
    - retention_score (HR_sid / HR_orig)
    - distance_correlation (Spearman)
  分布性 (Distribution):
    - code_utilization (每层利用率)
    - code_entropy (每层 Shannon 熵)
    - code_gini (每层 Gini 系数)
    - perplexity = exp(entropy)
  独立性 (Independence):
    - collision_rate (SID 碰撞率)
    - codeword_overlap O_π (DRQ 论文)
    - effective_capacity K_eff (DRQ 论文)
  语义性 (Semantics):
    - semantic_cohesion SC (QuaSID 论文)
    - preference_discrimination PD (QuaSID 论文)
  重建性 (Reconstruction):
    - reconstruction MSE (内部用, 单独函数未暴露)

参考论文:
  - QuaSID (arxiv 2604.11440): SC, PD 指标
  - DRQ (arxiv 2606.01844): O_π, K_eff, 诊断指标
  - Reliability paper (arxiv 2605.25330): 碰撞感知指标
  - semantic-id PyPI lib: utilization / entropy / Gini / distance_corr
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# 让脚本可以直接 python -m src.sid_metrics 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

# scipy 是可选依赖; 缺失时 distance_correlation 返回 NaN 而非崩溃
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# 类型别名: 接受 numpy array 或 list
ArrayLike = Union[np.ndarray, Sequence]


def _to_2d(x: ArrayLike) -> np.ndarray:
    """把任意输入转成 2D float32 array, 1D 视为单行样本。"""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化, 兼容训练发散时的 NaN/Inf。

    兜底: 训练发散 (如 RQ-VAE loss 爆炸) 时 codebook/embedding 可能含 NaN/Inf,
    直接归一化会把 NaN 传播到整张相似度矩阵。先净化, 再用 float64 算范数避免
    大值在 float32 下平方溢出, 零向量保持为零 (范数置 1)。
    """
    # 把 NaN/Inf 替换为 0, 防止传播
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # float64 算范数: float32 在 D=4096 时平方会溢出
    norms = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
    # 零向量保持零: 范数 < 1e-12 视为零
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (x / norms).astype(np.float32)


def semantic_cohesion(embeddings: ArrayLike, sids: ArrayLike,
                       sample_size: int = 50_000,
                       seed: int = 0) -> float:
    """语义内聚性 (Semantic Cohesion, SC): QuaSID 论文指标。

    计算同一 SID 簇内所有物品对的余弦相似度均值。
    反映"同一 SID 对应同一类语义"的程度。

    Args:
        embeddings: (N, D) 物品 embedding。
        sids: (N, L) 整数 SID 序列。
        sample_size: 采样对数上限 (避免 O(N²) 全量计算)。
        seed: 随机种子。

    Returns:
        标量, 范围 [-1, 1]; 越接近 1 越好 (簇内一致)。
    """
    embs = _normalize(_to_2d(embeddings))
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    if embs.shape[0] != sids.shape[0]:
        raise ValueError(f"shape mismatch: embs {embs.shape[0]} vs sids {sids.shape[0]}")

    # 把物品按 SID 分组, 形成 sid → [item_idx...] 的映射
    sid_to_idx: Dict[Tuple[int, ...], List[int]] = {}
    for i, sid in enumerate(sids):
        sid_to_idx.setdefault(tuple(sid.tolist()), []).append(i)

    rng = np.random.default_rng(seed)
    sims = []
    # 计算所有簇内的总对数, 与 sample_size 取最小
    n_pairs = min(sample_size, sum(len(c) * (len(c) - 1) // 2
                                   for c in sid_to_idx.values() if len(c) >= 2))
    pairs_attempted = 0
    # 对每个簇 (SID 相同的物品组), 随机采若干对计算余弦相似度
    for cluster in sid_to_idx.values():
        if len(cluster) < 2 or len(sims) >= n_pairs:
            continue
        for _ in range(min(100, n_pairs - len(sims))):
            # 不放回抽 2 个物品
            i, j = rng.choice(cluster, 2, replace=False)
            sims.append(float(np.dot(embs[i], embs[j])))
            pairs_attempted += 1
    if not sims:
        return 0.0
    return float(np.mean(sims))


def preference_discrimination(sids: ArrayLike,
                               interactions: Dict[int, float],
                               sample_pairs: int = 10_000,
                               seed: int = 0) -> float:
    """偏好区分度 (Preference Discrimination, PD): QuaSID 论文指标。

    衡量不同 SID 簇在用户偏好分布上的差异。
    PD = (1/|C|²) Σ_{c1, c2} |D(c1) - D(c2)|
    其中 D(c) 是簇 c 内所有物品的平均交互次数。

    直观理解: 一个好的 SID 系统应该把"常点击"与"从不点击"的物品分到不同簇。
    PD 越大说明区分用户偏好的能力越强。

    Args:
        sids: (N, L) 整数 SID 序列。
        interactions: {item_idx: interaction_count} 物品索引 → 交互次数。
        sample_pairs: 采样对数 (避免 O(K²))。
        seed: 随机种子。

    Returns:
        标量, 范围 [0, +∞); 越大越好。
    """
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    N = sids.shape[0]
    # 把 interactions 转成数组, 默认 0
    item_pref = np.zeros(N, dtype=np.float32)
    for k, v in interactions.items():
        if 0 <= k < N:
            item_pref[k] = v

    # 按 SID 分组, 算每簇的平均偏好
    sid_to_pref: Dict[Tuple[int, ...], List[float]] = {}
    for i, sid in enumerate(sids):
        sid_to_pref.setdefault(tuple(sid.tolist()), []).append(item_pref[i])
    sid_means = {k: float(np.mean(v)) for k, v in sid_to_pref.items()}
    keys = list(sid_means.keys())
    # 至少需要 2 个簇才有区分度可言
    if len(keys) < 2:
        return 0.0

    # 随机采 sample_pairs 个簇对, 算偏好差绝对值的平均
    rng = np.random.default_rng(seed)
    n_pairs = min(sample_pairs, len(keys) * (len(keys) - 1) // 2)
    diffs = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(keys), 2, replace=False)
        diffs.append(abs(sid_means[keys[i]] - sid_means[keys[j]]))
    return float(np.mean(diffs))


def codeword_overlap(embeddings: ArrayLike, sids: ArrayLike,
                     epsilon: float = 0.05,
                     k_neighbor: int = 10) -> float:
    """Codeword 重叠率 (Overlap, O_π): DRQ 论文诊断指标。

    在 embedding 空间中对每个物品做 ε 邻域检索,
    统计"相似物品是否被分到同一 SID"的比率。

    动机: 一个鲁棒的 SID 系统应当"相似物品 = 相同 SID"。
    如果相邻物品被分到不同 SID, O_π 较高 → 系统对扰动不稳定。

    公式: O_π = E_{i ~ p_π, j ~ p(i, ε)} [I(c(i) = c(j))]

    Args:
        epsilon: cosine 相似度阈值 (1 - ε) 控制邻域大小。
        k_neighbor: 备用参数 (本实现用 epsilon)。

    Returns:
        范围 [0, 1]; 越小越好 (低重叠 = 高区分度)。
    """
    embs = _normalize(_to_2d(embeddings))
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    N = embs.shape[0]
    # 数据太多时采样到 5K, 避免 O(N²) 内存爆炸
    if N > 5000:
        idx = np.random.default_rng(0).choice(N, 5000, replace=False)
        embs = embs[idx]
        sids = sids[idx]
        N = 5000
    # 余弦相似度矩阵 (归一化后点积 = cosine)
    sims = embs @ embs.T
    # 排除对角线 (自身不算邻居)
    np.fill_diagonal(sims, -np.inf)
    threshold = 1.0 - epsilon
    collision_count = 0
    pair_count = 0
    # 对每个物品找 ε 邻域, 看 SID 是否一致
    for i in range(N):
        neighbors = np.where(sims[i] > threshold)[0]
        for j in neighbors:
            if np.array_equal(sids[i], sids[j]):
                collision_count += 1
            pair_count += 1
    if pair_count == 0:
        return 0.0
    return float(collision_count / pair_count)


def effective_capacity(K: int, overlap: float) -> float:
    """有效容量 (Effective Capacity, K_eff): DRQ 论文指标。

    把"碰撞"折算成"实际可分辨的 codebook 容量"。
    公式: K_eff = K / (1 + (K-1) * O_π)

    直观理解:
        O_π = 0 → K_eff = K (全部 code 互不重叠, 理想状态)
        O_π = 1 → K_eff = 1 (全部物品共享同一 SID, 容量坍缩)

    一个 SID 系统哪怕 K 很大, 如果 O_π 很高, 实际可用 code 也很少。
    所以 K_eff 比 K 更能反映真实容量。

    Args:
        K: 码本大小。
        overlap: O_π ∈ [0, 1]。

    Returns:
        范围 [1, K]; 越接近 K 越好。
    """
    if K <= 1:
        return 1.0
    return float(K / (1.0 + (K - 1) * overlap))


def code_utilization(sids: ArrayLike, codebook_size: int) -> List[float]:
    """每层码本利用率 = 已使用 code 数 / 总数。

    范围 [0, 1]; 越大越好 (>70% 健康)。
    """
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    L = sids.shape[1]
    # np.unique 返回不同值, len 就是已使用数
    return [float(len(np.unique(sids[:, l]))) / codebook_size for l in range(L)]


def code_entropy(sids: ArrayLike) -> List[float]:
    """每层 Shannon 熵 (单位 nats)。

    越大 = 分布越均匀。最大可能值 = ln(K) (均匀分布时)。
    """
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    L = sids.shape[1]
    out = []
    for l in range(L):
        # Counter 统计每种 code 的频次
        counts = Counter(sids[:, l].tolist())
        total = sum(counts.values())
        # 概率分布
        p = np.array([c / total for c in counts.values()], dtype=np.float64)
        # -sum(p log p), 加 1e-12 防止 log(0)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        out.append(ent)
    return out


def code_gini(sids: ArrayLike) -> List[float]:
    """每层 Gini 系数。

    衡量分布不均匀度: 0 = 完全均匀, 1 = 全部集中在一个 code。
    越低越好 (< 0.3 健康)。
    """
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    L = sids.shape[1]
    out = []
    for l in range(L):
        counts = Counter(sids[:, l].tolist())
        # 按频次升序排 (Gini 公式要求)
        freqs = np.array(sorted(counts.values()), dtype=np.float64)
        n = len(freqs)
        if n == 0 or freqs.sum() == 0:
            out.append(0.0)
            continue
        cum = np.cumsum(freqs)
        # Gini 公式: G = (2 Σ i·x_i) / (n Σx) - (n+1)/n
        gini = float((2 * np.sum((np.arange(1, n + 1) * freqs))) / (n * freqs.sum()) - (n + 1) / n)
        out.append(gini)
    return out


def distance_correlation(embeddings: ArrayLike, sids: ArrayLike,
                         sample_size: int = 1000,
                         seed: int = 0) -> float:
    """Spearman 相关性: embedding 欧氏距离 vs SID Hamming 距离。

    衡量 SID 距离是否反映 embedding 距离。
    高相关 = SID 检索保真度高 (近邻物品 SID 也近)。

    Args:
        embeddings: (N, D) 物品 embedding。
        sids: (N, L) SID 序列。
        sample_size: 采样数 (避免 O(N²))。
        seed: 随机种子。

    Returns:
        范围 [-1, 1]; 越高越好。
    """
    if not HAS_SCIPY:
        # 没装 scipy 时返回 NaN, 让上层判断
        return float("nan")
    embs = _to_2d(embeddings)
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    N = embs.shape[0]
    if N > sample_size:
        # 采样到 sample_size, 减少 O(N²) 计算量
        idx = np.random.default_rng(seed).choice(N, sample_size, replace=False)
        embs = embs[idx]
        sids = sids[idx]
        N = sample_size
    # embedding 欧氏距离矩阵 (上三角)
    emb_dist = np.sqrt(((embs[:, None] - embs[None, :]) ** 2).sum(-1))
    # SID Hamming 距离: 只要有一位不同就算 1 (0/1 化)
    sid_dist = (sids[:, None] != sids[None, :]).any(-1).astype(np.float32)
    # 提取上三角 (i<j)
    emb_flat = emb_dist[np.triu_indices(N, k=1)]
    sid_flat = sid_dist[np.triu_indices(N, k=1)]
    # 标准差为 0 说明所有样本 SID 全相同, 没有相关性可算
    if np.std(sid_flat) < 1e-9:
        return 0.0
    if np.std(emb_flat) < 1e-9:
        return 0.0
    result = spearmanr(emb_flat, sid_flat)
    # 新版 scipy 返回 Spearmanr 对象, 老版本返回元组
    corr = result.correlation if hasattr(result, "correlation") else result[0]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def collision_rate(sids: ArrayLike) -> float:
    """碰撞率: 与至少一个其他物品共享 SID 的物品比例。

    范围 [0, 1]; 越低越好 (< 1% 健康)。
    """
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    # 每种 SID 的出现次数
    counts = Counter(map(tuple, sids.tolist()))
    N = len(sids)
    # 碰撞物品数 = 所有 count > 1 的 SID 的物品总和
    collided = sum(c for sid, c in counts.items() if c > 1)
    return float(collided / N)


def retention_score(embeddings: ArrayLike, sids: ArrayLike,
                    codebooks: Optional[ArrayLike] = None,
                    k: int = 10, sample_size: int = 200,
                    seed: int = 0) -> float:
    """HR_sid / HR_orig: SID 检索 vs 原 embedding 检索的命中率比值。

    对每个 query (物品 embedding), 找 top-K 邻居, 按两种方式:
      - 原 embedding cosine
      - 解码 SID (各层 codeword 之和) cosine
    算重叠 / K, 取平均。

    Args:
        embeddings: (N, D) 物品 embedding。
        sids: (N, L) SID 序列。
        codebooks: 可选 (L, K, D) 训练好的码本; 不传时退化为 one-hot 近似。
        k: top-K。
        sample_size: 作为 query 的物品数。
        seed: 随机种子。

    Returns:
        范围 [0, 1]; 越接近 1 越好。
    """
    embs = _normalize(_to_2d(embeddings))
    sids = np.asarray(sids, dtype=np.int64)
    if sids.ndim == 1:
        sids = sids.reshape(-1, 1)
    N = embs.shape[0]
    if N > 2000:
        # 采样到 2000 避免 O(N²) 内存爆炸
        idx = np.random.default_rng(seed).choice(N, 2000, replace=False)
        embs_q = embs[idx]
        sids_q = sids[idx]
    else:
        embs_q, sids_q = embs, sids

    # 用 codebook 重建 SID 向量 (每个物品 = Σ_l codebook[l, sid_l])
    if codebooks is not None:
        cb = np.asarray(codebooks, dtype=np.float32)  # (L, K, D)
        sid_recon = np.zeros_like(embs_q)
        for l in range(sids.shape[1]):
            sid_recon += cb[l, sids_q[:, l]]
        sid_recon = _normalize(sid_recon)
    else:
        # 没给 codebook 时的退化: 用 one-hot 平均作为粗糙重建
        sid_recon = embs_q.copy()
        for l in range(sids.shape[1]):
            codebook_eye = np.eye(sids.max() + 1, dtype=np.float32)
            sid_recon = sid_recon + codebook_eye[sids_q[:, l]].mean(axis=1, keepdims=True)
        sid_recon = _normalize(sid_recon)

    # 随机抽 sample_size 个 query
    n_query = min(sample_size, len(embs_q))
    query_idx = np.random.default_rng(seed).choice(len(embs_q), n_query, replace=False)

    # 两两相似度矩阵
    sim_orig = embs_q @ embs_q.T
    sim_sid = sid_recon @ sid_recon.T
    # 排除自身
    np.fill_diagonal(sim_orig, -np.inf)
    np.fill_diagonal(sim_sid, -np.inf)

    ratios = []
    for q in query_idx:
        # argpartition 的 k 必须 <= n-1, 否则报错
        actual_k = min(k, len(embs_q) - 1)
        if actual_k < 1:
            continue
        # 取 top-K 索引, 转 set 求交集
        orig_top = set(np.argpartition(-sim_orig[q], actual_k)[:actual_k].tolist())
        sid_top = set(np.argpartition(-sim_sid[q], actual_k)[:actual_k].tolist())
        overlap = len(orig_top & sid_top)
        ratios.append(overlap / actual_k)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


@dataclass
class SIDEvalReport:
    """完整 SID 评估报告, 14 项指标 + 综合评级 + 重训决策。

    字段按维度分组, 详见 src/sid_evaluation.py 的 5 维度评估报告。
    """
    semantic_cohesion: float = 0.0
    preference_discrimination: float = 0.0
    codeword_overlap: float = 0.0
    effective_capacity: float = 0.0
    code_utilization: List[float] = field(default_factory=list)
    code_entropy: List[float] = field(default_factory=list)
    code_gini: List[float] = field(default_factory=list)
    distance_correlation: float = 0.0
    collision_rate: float = 0.0
    retention_score: float = 0.0
    quantization_mse: float = 0.0

    def to_dict(self) -> dict:
        return {
            "semantic_cohesion": self.semantic_cohesion,
            "preference_discrimination": self.preference_discrimination,
            "codeword_overlap": self.codeword_overlap,
            "effective_capacity": self.effective_capacity,
            "code_utilization": self.code_utilization,
            "code_entropy": self.code_entropy,
            "code_gini": self.code_gini,
            "distance_correlation": self.distance_correlation,
            "collision_rate": self.collision_rate,
            "retention_score": self.retention_score,
            "quantization_mse": self.quantization_mse,
        }

    def summary(self) -> str:
        """人类可读的简明报告 (单行每指标)。"""
        d = self.to_dict()
        lines = ["SID Quality Report", "=" * 40]
        for k, v in d.items():
            if isinstance(v, list):
                v_str = "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
            elif isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            lines.append(f"  {k:32s}: {v_str}")
        return "\n".join(lines)

    def needs_retrain(self,
                      sc_threshold: float = 0.6,
                      pd_threshold: float = 0.3,
                      min_util_threshold: float = 0.7,
                      collision_threshold: float = 0.01,
                      retention_threshold: float = 0.8,
                      keff_ratio_threshold: Optional[float] = None) -> Tuple[bool, List[str]]:
        """基于阈值判断是否需要重训。

        Args:
            sc_threshold: SC 最小可接受值 (默认 0.6)。
            pd_threshold: PD 最小可接受值 (默认 0.3)。
            min_util_threshold: 每层最小利用率 (默认 0.7)。
            collision_threshold: 最大碰撞率 (默认 0.01)。
            retention_threshold: 最小 HR 比值 (默认 0.8)。
            keff_ratio_threshold: 最小 K_eff/K 比值, None 表示跳过此检查。

        Returns:
            (needs_retrain, [reasons]) 元组。
        """
        reasons = []
        if self.semantic_cohesion < sc_threshold:
            reasons.append(f"SC {self.semantic_cohesion:.3f} < {sc_threshold}")
        if self.preference_discrimination < pd_threshold:
            reasons.append(f"PD {self.preference_discrimination:.3f} < {pd_threshold}")
        if self.code_utilization and min(self.code_utilization) < min_util_threshold:
            reasons.append(
                f"min code util {min(self.code_utilization):.2%} < "
                f"{min_util_threshold:.0%}")
        if self.collision_rate > collision_threshold:
            reasons.append(
                f"collision rate {self.collision_rate:.2%} > "
                f"{collision_threshold:.0%}")
        if self.retention_score < retention_threshold:
            reasons.append(
                f"retention {self.retention_score:.3f} < {retention_threshold}")
        if keff_ratio_threshold is not None and self.effective_capacity > 0:
            # 用 effective_capacity 反推 K, 计算 K_eff/K
            K = max(self.effective_capacity * 2, 64)
            actual_ratio = self.effective_capacity / K
            if actual_ratio < keff_ratio_threshold:
                reasons.append(
                    f"K_eff ratio {actual_ratio:.2%} < "
                    f"{keff_ratio_threshold:.0%}")
        return bool(reasons), reasons


class SIDEvaluator:
    """一键评估器: 输入 embedding + SID → 输出完整 SIDEvalReport。

    适合离线批量评估 (训练后调一次)。
    """

    def __init__(self, codebook_size: int = 256):
        self.codebook_size = codebook_size

    def full_report(self,
                    embeddings: ArrayLike,
                    sids: ArrayLike,
                    interactions: Optional[Dict[int, float]] = None,
                    reconstructed: Optional[ArrayLike] = None,
                    codebooks: Optional[ArrayLike] = None,
                    sample_size: int = 1000,
                    epsilon: float = 0.05) -> SIDEvalReport:
        """一次性算齐 14 项指标 + 重建 MSE。

        Args:
            embeddings: (N, D) 物品 embedding。
            sids: (N, L) SID 序列。
            interactions: 可选 {item_idx: count}, 用于 PD。
            reconstructed: 可选 (N, D) 重建向量; 不传则用 embeddings (MSE=0)。
            codebooks: 可选 (L, K, D), retention_score 用。
            sample_size: 采样数 (限制 O(N²) 计算)。
            epsilon: codeword_overlap 的 cosine 阈值 (1-ε)。

        Returns:
            SIDEvalReport。
        """
        sids = np.asarray(sids, dtype=np.int64)
        if sids.ndim == 1:
            sids = sids.reshape(-1, 1)
        embs = _to_2d(embeddings)
        # 重建向量默认等于 embedding (MSE=0)
        if reconstructed is None:
            reconstructed = _to_2d(embeddings)
        mse = float(((embs - reconstructed) ** 2).mean())
        # K 取 SID 中最大值 + 1 与配置的较大值
        K = int(sids.max()) + 1 if sids.size > 0 else self.codebook_size
        K = max(K, self.codebook_size)

        # 逐项计算
        sc = semantic_cohesion(embs, sids, sample_size=sample_size)
        pd = preference_discrimination(sids, interactions or {}, sample_pairs=sample_size)
        overlap = codeword_overlap(embs, sids, epsilon=epsilon)
        util = code_utilization(sids, K)
        ent = code_entropy(sids)
        gini = code_gini(sids)
        dc = distance_correlation(embs, sids, sample_size=sample_size)
        cr = collision_rate(sids)
        rs = retention_score(embs, sids, codebooks=codebooks,
                              k=10, sample_size=sample_size)

        return SIDEvalReport(
            semantic_cohesion=sc,
            preference_discrimination=pd,
            codeword_overlap=overlap,
            effective_capacity=effective_capacity(K, overlap),
            code_utilization=util,
            code_entropy=ent,
            code_gini=gini,
            distance_correlation=dc,
            collision_rate=cr,
            retention_score=rs,
            quantization_mse=mse,
        )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from src.rq_kmeans import fit_rq_kmeans

    np.random.seed(42)
    N, D, L, K = 500, 32, 5, 64

    # 构造一个有 16 个聚类的高斯数据, RQ-Kmeans 应该能学到
    true_clusters = 16
    centers = np.random.randn(true_clusters, D).astype(np.float32) * 0.5
    cluster_assign = np.random.randint(0, true_clusters, size=N)
    embs = centers[cluster_assign] + 0.1 * np.random.randn(N, D).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=-1, keepdims=True)

    tok = fit_rq_kmeans(torch.from_numpy(embs),
                         num_layers=L, codebook_size=K,
                         n_passes=3, batch_size=64, verbose=False)
    sids = tok.encode_to_sid(torch.from_numpy(embs)).numpy()
    codebooks = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)

    # 用真实 cluster_assign 模拟"交互次数" (cluster id 大 = 越受欢迎)
    interactions = {i: float(np.random.exponential(1.0)) * (1 + (cluster_assign[i] % 5))
                     for i in range(N)}

    evalr = SIDEvaluator(codebook_size=K)
    report = evalr.full_report(embs, sids, interactions, codebooks=codebooks)
    print(report.summary())

    needs, reasons = report.needs_retrain()
    print(f"\nNeeds retrain: {needs}")
    for r in reasons:
        print(f"  - {r}")

    # 单点指标速览
    sc = semantic_cohesion(embs, sids, sample_size=2000)
    print(f"\nsemantic_cohesion: {sc:.4f}")
    print(f"codeword_overlap: {codeword_overlap(embs, sids):.4f}")
    print(f"effective_capacity (K=64): {effective_capacity(64, 0.05):.2f}")
    print(f"distance_correlation: {distance_correlation(embs, sids):.4f}")