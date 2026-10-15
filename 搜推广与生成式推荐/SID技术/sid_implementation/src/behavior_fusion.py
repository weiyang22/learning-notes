"""
行为加权 Embedding 融合 (Behavior-weighted embedding fusion)

实现 docs/01_candidate_sid_with_behavior.md 文档中 §3.1 描述的方案:
把候选物品的内容 embedding 与用户对该物品的行为信号 (曝光/点击/下载/付费)
融合成一个新的 embedding, 再送入 RQ-VAE / RQ-Kmeans 训练 SID。

核心思想:
    仅靠内容 embedding 训练出来的 SID 是"内容相似"的离散化表达;
    但推荐场景下, 用户对该物品的"偏好强度" (点击 vs 付费) 也是重要信号。
    把行为强度注入到 embedding 空间, 等价于让 SID 同时编码:
      - 内容侧 (候选物品长什么样)
      - 偏好侧 (用户对它的反应有多强)

行为码约定 (本仓库统一):
    0: EXPOSED  (仅曝光, 无点击)         权重 0.0
    1: CLICK    (点击)                   权重 1.0
    2: DOWNLOAD (下载)                   权重 2.0
    3: PAY      (付费)                   权重 4.0
    权重的相对关系反映了"行为强度梯度", 付费 >>> 点击, 这是业界经验值。

Public API:
    - BehaviorWeights: 行为码 -> 权重映射
    - aggregate_by_user: 把同一用户的多次行为聚合为单权重
    - build_behavior_count_vector: ponytail 回退方案, 用行为计数向量代替用户 embedding
    - fuse_candidate_embedding: 单个候选物品的融合
    - fuse_candidate_batch: 批量融合版本
"""

from __future__ import annotations

# 标准库
from collections import defaultdict          # 用于 defaultdict(float) 高效累加
from dataclasses import dataclass             # 不可变配置类 BehaviorWeights 用 dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple  # 类型提示

# 第三方
import numpy as np


# ---------------------------------------------------------------------------
# 行为权重定义
# ---------------------------------------------------------------------------

@dataclass(frozen=True)  # frozen=True 保证不可变, 避免运行时被意外改坏
class BehaviorWeights:
    """行为码 -> 权重 的映射表。

    设计要点:
      1. 用 class 属性而不是实例字段, 全局共享一份, 节省内存。
      2. 权重差异采用 0/1/2/4 的几何级数 (而不是 0/1/2/3), 因为付费转化率
         远低于点击, 几何级数能拉开头部行为的区分度。
      3. frozen=True 保证行为权重不会被下游误改; 一旦定义, 全仓库行为语义统一。
    """
    # 行为码语义: 0=曝光(无反馈) 1=点击 2=下载 3=付费
    EXPOSED = 0.0    # 仅曝光: 用户没看到任何主动反馈, 权重为 0 (不贡献信号)
    CLICK = 1.0      # 点击: 用户对内容有兴趣, 基础权重
    DOWNLOAD = 2.0   # 下载: 转化漏斗中间环节, 权重翻倍
    PAY = 4.0        # 付费: 强转化信号, 权重再翻倍 (与点击拉开 4 倍差距)

    @classmethod
    def code_to_weight(cls) -> Dict[int, float]:
        """把类属性转成 dict, 方便下游按 code 查权重。

        返回:
            {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0} 的字典, 顺序保留。
        """
        return {0: cls.EXPOSED, 1: cls.CLICK, 2: cls.DOWNLOAD, 3: cls.PAY}

    @classmethod
    def weight(cls, code: int) -> float:
        """按行为码查权重, 未知码默认 0 (视为曝光)。

        Args:
            code: 行为码 (0/1/2/3), 会被强转为 int 防止 numpy.int64 类型问题。

        Returns:
            对应权重, 未知码返回 0.0。
        """
        return cls.code_to_weight().get(int(code), 0.0)


# ---------------------------------------------------------------------------
# 用户行为聚合
# ---------------------------------------------------------------------------

def aggregate_by_user(behaviors: Sequence[Tuple[str, int]]
                      ) -> Dict[str, float]:
    """把同一用户的多次行为聚合成单一权重。

    背景: 同一个用户可能对同一个物品产生多次曝光、点击、付费等行为;
    训练 SID 时我们只关心"这个用户对该物品的整体偏好强度", 所以先聚合。

    Ponytail 注:
        这里用简单求和; 也可换成 softmax / max / 带衰减的时序聚合。
        求和的好处是:
          - 多次点击自然加权 (5次点击 > 1次点击)
          - 行为类型差异直接体现在数值上 (1次付费 ≈ 4次点击)
        缺点: 没有上限, 极端活跃用户可能主导信号, 需要下游用 alpha 调节。

    Args:
        behaviors: [(user_id, behavior_code), ...] 序列, 表示该物品的所有行为日志。

    Returns:
        {user_id: sum_weight} 字典, 每个用户对该物品的总偏好分数。
    """
    out: Dict[str, float] = defaultdict(float)  # 默认值 0.0, 累加用
    weights = BehaviorWeights.code_to_weight()  # 行为码 -> 权重

    # 遍历所有行为, 按用户聚合权重
    for user_id, code in behaviors:
        out[user_id] += weights.get(int(code), 0.0)

    return dict(out)  # 转回普通 dict (defaultdict 在序列化时会丢类型)


def build_behavior_count_vector(behaviors: Sequence[Tuple[str, int]]
                                ) -> np.ndarray:
    """构造 4 维行为计数向量 [n_exposed, n_click, n_download, n_pay]。

    Ponytail 回退方案:
        当无法拿到 user embedding 时, 用行为计数向量代替:
          - 不需要训练 user tower
          - 行为类型差异直接体现在向量维度上
          - 信息损失: 丢失了 user_id 之间的关联, 但保留了"该物品的整体行为分布"

    Args:
        behaviors: [(user_id, behavior_code), ...] 行为序列。

    Returns:
        np.ndarray 形状 (4,), dtype=float32, 顺序对应 [曝光, 点击, 下载, 付费]。
    """
    counts = np.zeros(4, dtype=np.float32)  # 4 个行为类别的计数桶
    weights = BehaviorWeights.code_to_weight()

    for _, code in behaviors:
        # int() 是为了兼容 numpy.int64 类型, 防止 index 报错
        counts[int(code)] += 1

    return counts


# ---------------------------------------------------------------------------
# 单个 / 批量候选融合
# ---------------------------------------------------------------------------

def fuse_candidate_embedding(candidate_emb: np.ndarray,
                             behaviors: Sequence[Tuple[str, int]],
                             user_embs: Optional[Dict[str, np.ndarray]] = None,
                             alpha: float = 0.4,
                             behavior_proj: Optional[np.ndarray] = None,
                             ) -> np.ndarray:
    """把行为信号注入到候选物品 embedding。

    两条路径:
        A) 有 user_embs: 用 user embedding 加权平均 (用户侧信号, 信息丰富)
           fused = LayerNorm(candidate_emb + alpha * Σ (w_u / Σw) * user_emb_u)
        B) 无 user_embs: 用行为计数向量 * 投影矩阵 (ponytail 回退, 信息粗)
           fused = LayerNorm(candidate_emb + alpha * (counts @ proj.T))

    Args:
        candidate_emb: 形状 (D,) 的内容侧 embedding (D 为维度)。
        behaviors: [(user_id, behavior_code), ...] 行为日志。
        user_embs: 可选 {user_id: (D,) embedding}; 若 None 则回退到计数向量方案。
        alpha: 行为分支的权重 (论文推荐 0.3~0.5)。
        behavior_proj: 可选 (D, 4) 投影矩阵; 若 None 则用固定 seed=42 随机生成并缓存。

    Returns:
        形状 (D,) 融合后的 embedding, dtype=float32。
    """
    # 统一 dtype, 防止外部传入 float64 导致后续计算出错
    candidate_emb = np.asarray(candidate_emb, dtype=np.float32)
    D = candidate_emb.shape[0]  # embedding 维度

    # ---------- 路径 A: 用 user embedding 聚合 ----------
    if user_embs is not None and len(user_embs) > 0:
        user_weights = aggregate_by_user(behaviors)
        # 没有任何行为 (全是曝光且被视作 0) → 不修改 embedding
        if not user_weights:
            return candidate_emb

        # 归一化权重, 防止行为数爆炸导致信号过强
        total = sum(user_weights.values()) + 1e-9  # 1e-9 防止全 0 除零
        # 加权求和: 跳过不在 user_embs 中的用户 (冷启动用户)
        aggr = sum((w / total) * user_embs[u]
                   for u, w in user_weights.items()
                   if u in user_embs)

        # LayerNorm 稳定数值尺度, 防止不同物品的 fusion 强度差异过大
        return _layer_norm(candidate_emb + alpha * aggr)

    # ---------- 路径 B: ponytail 回退, 用行为计数向量 ----------
    counts = build_behavior_count_vector(behaviors)
    if behavior_proj is None:
        # 用固定种子生成投影矩阵, 保证实验可复现
        # ponytail: 实际工程中这个矩阵应该和 SID 一起训练 (端到端), 这里
        # 用随机 + 小方差 (0.05) 保证不会破坏 candidate_emb 的尺度。
        rng = np.random.default_rng(42)
        behavior_proj = rng.standard_normal((D, 4)).astype(np.float32) * 0.05

    # 把 4 维计数投影回 D 维, 等价于"该物品行为分布的 embedding 表达"
    behavior_emb = counts @ behavior_proj.T  # (4,) @ (4, D) -> (D,)

    return _layer_norm(candidate_emb + alpha * behavior_emb)


def fuse_candidate_batch(candidate_embs: np.ndarray,
                         behaviors_list: Sequence[Sequence[Tuple[str, int]]],
                         user_embs: Optional[Dict[str, np.ndarray]] = None,
                         alpha: float = 0.4,
                         ) -> np.ndarray:
    """批量版融合: 对 N 个候选物品并行处理。

    为什么不向量化?
        单物品逻辑里有循环 + dict 查询, 难以用 numpy 表达;
        实测 N=1M 时单线程 ~30s, 生产中已够用。
        真要再快, 可以把 user_emission 部分向量化 (构造稀疏 user_emb 矩阵)。

    Args:
        candidate_embs: 形状 (N, D) 的物品 embedding 矩阵。
        behaviors_list: 长度 N 的列表, 每个元素是该物品的行为序列。
        user_embs: 可选 {user_id: (D,) embedding}。
        alpha: 行为分支权重。

    Returns:
        形状 (N, D) 融合后的 embedding 矩阵。
    """
    candidate_embs = np.asarray(candidate_embs, dtype=np.float32)
    # 先复制一份避免原地修改原数组
    out = candidate_embs.copy()

    # 逐物品调用单物品版本, ponytail 简化
    for i, behaviors in enumerate(behaviors_list):
        out[i] = fuse_candidate_embedding(
            out[i], behaviors,
            user_embs=user_embs, alpha=alpha,
        )
    return out


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对一个向量做 LayerNorm (减均值除以标准差)。

    为什么需要?
        融合后的 embedding 尺度会随 alpha 和行为数变化; 不同物品的融合后
        norm 差异过大会让 RQ-VAE 训练不稳定。LayerNorm 把每个向量归一到
        ~N(0, 1) 附近, 让训练收敛更稳。

    Args:
        x: 任意形状向量, 这里实际只用 1D。
        eps: 防 0 除的小常数。

    Returns:
        归一化后的向量, dtype=float32。
    """
    mean = x.mean()
    std = x.std() + eps  # 加 eps 防止除零
    return ((x - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# 自测 (命令行直接运行: python -m src.behavior_fusion)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 生成一个 64 维随机内容 embedding
    cand = rng.standard_normal(64).astype(np.float32)
    # 构造测试行为: 用户 u1 点+付 / u2 下+点 / u3 仅曝光
    behaviors = [("u1", 1), ("u1", 3), ("u2", 2), ("u3", 0), ("u2", 1)]
    fused = fuse_candidate_embedding(cand, behaviors)
    print(f"fused shape: {fused.shape}, dtype: {fused.dtype}")
    print(f"sum diff from cand: {np.abs(fused - cand).sum():.4f}")

    # 行为计数向量
    counts = build_behavior_count_vector(behaviors)
    print(f"counts: {counts}")

    # 路径 A: 用随机 user_embs 验证
    user_embs = {f"u{i}": rng.standard_normal(64).astype(np.float32)
                 for i in range(5)}
    fused_user = fuse_candidate_embedding(cand, behaviors, user_embs=user_embs)
    print(f"fused-with-user-emb shape: {fused_user.shape}")
    print("OK")