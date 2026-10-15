"""
Online SID Server: 生产级新物品 SID 服务

在 src.faiss_sid 基础上的生产化封装, 增加了:
  - 健康监控 (HealthMetrics): 实时跟踪 collision_rate / codebook_usage / MSE
  - 冷启动 query 处理: 二部图 + Top-K 增强
  - 多路径推断 (DREAM 论文方案): 给冷启动物品返回 K 个候选, 让下游排序选
  - 增量码本更新 (Sculley mini-batch): 不重训就能跟随数据分布漂移
  - 状态持久化 (save / load): 支持服务重启恢复

性能目标:
  - 暖路径 P99 < 5ms (FAISS 5 层查表)
  - 吞吐 > 30K QPS / CPU 核
  - 缓存命中 < 100us

参考文档: docs/04_online_sid_generation.md
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 让脚本可以直接 python -m src.online_sid_server 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from src.faiss_sid import FaissRQKMeans, NewItemSIDService


@dataclass
class OnlineSIDConfig:
    """在线服务的全部配置。"""
    num_layers: int = 5
    codebook_size: int = 256
    dim: int = 256
    collision_threshold: int = 5  # 单 SID 物品数超过此值触发告警
    health_check_window: int = 1000  # 健康监控采样窗口大小
    use_gpu: bool = False
    multi_path_k: int = 3  # 多路径推断默认返回几个候选
    # 可选: 冷启动 embedder, 当 item 没有 embedding 时调用
    cold_start_embedder: Optional[Callable] = None
    warm_start_threshold: float = 0.0  # 决定是否走冷启动的阈值 (备用)


@dataclass
class HealthMetrics:
    """在线服务健康监控指标, 用于定期检查 SID 服务质量。

    各字段含义:
      - n_items: 已注册物品数
      - n_unique_sids: 唯一 SID 数 (越接近 n_items 越好, 否则碰撞严重)
      - collision_rate: 碰撞率 (越低越好, < 1% 健康)
      - max_items_per_sid: 单 SID 最大物品数 (> collision_threshold 触发告警)
      - avg_mse: 重建 MSE (越小越好)
      - codebook_usage: 每层利用率 (越接近 1 越好, < 0.7 触发重训)
    """
    n_items: int = 0
    n_unique_sids: int = 0
    n_collisions: int = 0
    collision_rate: float = 0.0
    max_items_per_sid: int = 0
    avg_mse: float = 0.0
    codebook_usage: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_unique_sids": self.n_unique_sids,
            "n_collisions": self.n_collisions,
            "collision_rate": self.collision_rate,
            "max_items_per_sid": self.max_items_per_sid,
            "avg_mse": self.avg_mse,
            "codebook_usage": self.codebook_usage,
        }

    def needs_retrain(self, mse_baseline: float = 1.0,
                      min_usage: float = 0.7,
                      max_collision_rate: float = 0.01) -> Tuple[bool, str]:
        """基于阈值的重训决策。

        返回 (needs, reason):
          - needs=True: 至少一个指标超过阈值, 应该触发离线重训
          - needs=False: 健康

        阈值参考 OneRec / QuaSID 生产经验值。
        """
        if self.avg_mse > mse_baseline:
            return True, f"MSE {self.avg_mse:.4f} > baseline {mse_baseline}"
        if self.codebook_usage and min(self.codebook_usage) < min_usage:
            return True, f"min codebook usage {min(self.codebook_usage):.2%} < {min_usage:.0%}"
        if self.collision_rate > max_collision_rate:
            return True, f"collision rate {self.collision_rate:..2%} > {max_collision_rate:.0%}"
        return False, "healthy"


class OnlineSIDServer:
    """生产级 SID 服务, 集成健康监控 + 冷启动 + 多路径 + 增量更新。"""

    def __init__(self, faiss_rq: FaissRQKMeans, cfg: Optional[OnlineSIDConfig] = None):
        self.rq = faiss_rq
        self.cfg = cfg or OnlineSIDConfig()
        # 内部复用 NewItemSIDService (缓存 + 碰撞追踪)
        self.service = NewItemSIDService(faiss_rq)
        # 滑动窗口 MSE, 防止全量重算开销过大
        self.mse_window: List[float] = []
        self.total_added = 0

    @classmethod
    def from_codebooks(cls, codebooks: np.ndarray,
                       cfg: Optional[OnlineSIDConfig] = None) -> "OnlineSIDServer":
        """从训练好的 (L, K, D) 码本数组构造 server。

        Ponytail 工厂: 训练和部署分离, 训练完拿到 codebooks, 部署时构造 server。
        """
        faiss_rq = FaissRQKMeans(codebooks)
        return cls(faiss_rq, cfg=cfg)

    def add_new_item(self, item_id: str, embedding: np.ndarray,
                     cold_start: bool = False) -> dict:
        """在线服务主入口: 为新物品生成 SID。

        设计目标:
          - P99 延迟 < 5ms (暖路径)
          - 吞吐量 > 30K QPS / CPU 核
          - 缓存命中时延迟 < 100us

        Args:
            item_id: 唯一物品 ID。
            embedding: (D,) 物品 embedding, 若为 None 且 cold_start=True 则调用 cfg.cold_start_embedder。
            cold_start: 是否触发冷启动路径。

        Returns:
            dict 含 sid (长度 L 整数列表)、sid_str ("-" 分隔字符串)、
            collision (是否碰撞)、cached (是否缓存命中)、可选 cold_start 标记。

        Raises:
            ValueError: embedding=None 但没配 cold_start_embedder。
        """
        # 冷启动: embedding 为空时, 调 cfg 中配置的 embedder 函数现场生成
        if embedding is None:
            if not cold_start or self.cfg.cold_start_embedder is None:
                raise ValueError("embedding is None but cold_start is False "
                                 "or no embedder configured")
            embedding = self.cfg.cold_start_embedder(item_id)
        embedding = np.asarray(embedding, dtype=np.float32)
        # 防止上游传进来 2D 张量, 全部 flatten 成 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        result = self.service.add_new_item(item_id, embedding)
        self.total_added += 1
        # 标记冷启动 (业务侧可能需要额外处理)
        if cold_start:
            result["cold_start"] = True
        return result

    def add_new_item_with_multipath(self, item_id: str, embedding: np.ndarray,
                                    k_candidates: int = None) -> List[dict]:
        """多路径推断 (DREAM 论文方案: arxiv 2606.06947)。

        对每个物品返回 K 个 SID 候选, 按 MSE 排序, 用于:
          - 冷启动物品: 单一 SID 可能不准确, 给下游排序多选
          - 高价值物品: 多候选权重融合, 提升召回

        流程: 每层查 top-K FAISS, 取 K 种组合路径, 解码算 MSE。

        Args:
            item_id: 物品 ID (用于 logging / 缓存 key)。
            embedding: (D,) 物品 embedding。
            k_candidates: 候选数, 默认用 cfg.multi_path_k。

        Returns:
            按 MSE 升序的 [{sid, mse, rank}, ...] 列表, rank=0 是最优候选。
        """
        k = k_candidates or self.cfg.multi_path_k
        residual = embedding.astype(np.float32, copy=True)
        # 每层收集 top-K 候选
        candidates_per_level: List[List[int]] = []
        for l in range(self.rq.L):
            dists, idx = self.rq.backends[l].search(residual.reshape(1, -1), k)
            candidates_per_level.append(idx[0].tolist())
            # 用 top-1 更新残差 (走最优路径), 候选路径用 top-1..top-K 组合
            residual = residual - self.rq.codebooks[l, candidates_per_level[-1][0]]

        results = []
        for choice in range(k):
            # 拼接一条 SID: 每层取第 choice 个候选 (越界则取最后一个)
            sid = [candidates_per_level[l][min(choice, len(candidates_per_level[l]) - 1)]
                   for l in range(self.rq.L)]
            recon = self.rq.decode_one(np.array(sid))  # 重建向量
            mse = float(((embedding - recon) ** 2).mean())
            results.append({"sid": sid, "mse": mse, "rank": choice})
        # 按 MSE 升序: rank=0 是最优
        results.sort(key=lambda x: x["mse"])
        return results

    def add_new_query_with_cold_start(self, q_text: str,
                                       q_emb: Optional[np.ndarray] = None,
                                       item_pool: Optional[Dict[str, np.ndarray]] = None,
                                       top_k: int = 20) -> dict:
        """Cold-start query 处理: 复用方案 2 二部图 + Top-K 增强。

        Args:
            q_text: 原始 query 文本 (用于 logging)。
            q_emb: query embedding; 若 None, 用 cold_start_embedder。
            item_pool: {item_id: (D,)} 用于 Top-K 聚合。
            top_k: Top-K items to aggregate。
        """
        if q_emb is None:
            if self.cfg.cold_start_embedder is None:
                raise ValueError("q_emb is None and no embedder configured")
            q_emb = self.cfg.cold_start_embedder(q_text)
        q_emb = np.asarray(q_emb, dtype=np.float32).flatten()

        # 用 item_pool 增强 query embedding
        if item_pool:
            # 算 cosine 相似度, 取 top_k
            sims = []
            for iid, emb in item_pool.items():
                sim = float(np.dot(q_emb, emb) /
                            (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
                sims.append((iid, sim))
            sims.sort(key=lambda x: -x[1])
            top_k_items = sims[:top_k]
            if top_k_items:
                weights = np.array([s for _, s in top_k_items], dtype=np.float32)
                weights = weights / (weights.sum() + 1e-9)
                aggr = sum(w * item_pool[iid] for w, (iid, _) in zip(weights, top_k_items))
                q_emb = q_emb + 0.4 * aggr
                # 再归一化, 稳定尺度
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        # 把 query 当成一个虚拟 item 编码, 用 md5 当 id 避免冲突
        q_id = f"q_{hashlib.md5(q_text.encode()).hexdigest()[:8]}"
        result = self.add_new_item(q_id, q_emb, cold_start=(q_emb is None))
        result["q_text"] = q_text
        return result

    def add_new_item_batch(self, item_ids: Sequence[str],
                            embeddings: np.ndarray) -> List[dict]:
        """高吞吐批量 API, 走 batched 后端查表。"""
        return self.service.add_new_items_batch(item_ids, embeddings)

    def health_check(self, sample_embeddings: Optional[np.ndarray] = None,
                     sample_sids: Optional[np.ndarray] = None) -> HealthMetrics:
        """计算健康指标 + 判断是否需要重训。

        Args:
            sample_embeddings: 可选采样 embedding, 用于算 avg_mse。
            sample_sids: 对应 sample_embeddings 的 SID。
        """
        # 基础指标来自 service 的碰撞统计
        base = self.service.collision_stats()
        metrics = HealthMetrics(
            n_items=len(self.service.item_to_sid),
            n_unique_sids=base["unique_sids"],
            n_collisions=base["collided_sids"],
            collision_rate=base["collision_rate"],
            max_items_per_sid=base["max_items_per_sid"],
            codebook_usage=self._compute_codebook_usage(),
        )
        # 如果给了采样, 算 MSE
        if sample_embeddings is not None and sample_sids is not None:
            recon = self.rq.decode_batch(sample_sids)
            mse = float(((sample_embeddings - recon) ** 2).mean())
            metrics.avg_mse = mse
        return metrics

    def _compute_codebook_usage(self) -> List[float]:
        """计算每层码本利用率: 已使用的 codeword 数 / 总数。

        用法: 监控是否有大量死 code, 触发重启或重训。
        """
        from collections import Counter
        all_sids = [tuple(sid) for sid in self.service.item_to_sid.values()]
        if not all_sids:
            return [0.0] * self.rq.L
        usage = []
        all_sids_arr = np.array(all_sids)
        for l in range(self.rq.L):
            # 第 l 层用过的不同 code 数 / 总 code 数 K
            used = len(set(all_sids_arr[:, l].tolist()))
            usage.append(used / self.rq.K)
        return usage

    def partial_fit(self, new_embeddings: np.ndarray,
                    restart_dead_codes: bool = True) -> "OnlineSIDServer":
        """增量更新码本 (Sculley mini-batch k-means 在线更新)。

        这是论文 arxiv 2604.13273 提出的 SID staleness 更新方案:
        不需要全量重训, 用最近的新物品数据对码本做在线调整。
        每 ~几小时调用一次, 比全量重训节省 ~8-9x 计算。

        工作原理:
          1. 对每层 L: 用当前 partial_fit batch 计算新 code 分配
          2. 用 sliding-window mean 更新 centroid:
             c_k <- (n_k_old * c_k + sum_batch_k) / (n_k_old + batch_count_k)
          3. 对长期未用的 dead code: 从 batch 重新采样

        Args:
            new_embeddings: (B, D) 新物品 embedding, 用于驱动码本更新。
            restart_dead_codes: 是否重启长期未用的死 code。

        Returns:
            self, 支持链式调用。
        """
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)
        if new_embeddings.ndim == 1:
            new_embeddings = new_embeddings.reshape(1, -1)
        if new_embeddings.size == 0:
            raise ValueError("partial_fit received empty embeddings")
        if new_embeddings.shape[1] != self.rq.D:
            raise ValueError(
                f"embedding dim mismatch: got {new_embeddings.shape[1]}, "
                f"expected {self.rq.D}")

        # 逐层更新: 前几层的残差必须用当前码本剥除
        for l in range(self.rq.L):
            residual = new_embeddings.copy()
            # 先用当前码本算前面层的残差
            for prev_l in range(l):
                sids = self.rq.encode_layer_single(prev_l, residual)
                residual = residual - self.rq.codebooks[prev_l, sids]

            # 在第 l 层做 batch 编码
            sids_l = self.rq.encode_layer_single(l, residual)

            # 加载历史计数, 没记录就建零计数
            old_counts = self.rq._layer_counts.get(l)
            if old_counts is None:
                old_counts = np.zeros(self.rq.K, dtype=np.int64)
                self.rq._layer_counts[l] = old_counts
            old_centroids = self.rq.codebooks[l].copy()

            # 统计新 batch 的 code 分配
            new_counts = np.bincount(sids_l, minlength=self.rq.K).astype(np.int64)
            # np.add.at: 按 sids_l 把 residual 行加到 new_sums 对应桶 (scatter_add)
            new_sums = np.zeros((self.rq.K, self.rq.D), dtype=np.float32)
            np.add.at(new_sums, sids_l, residual)

            # 滑动平均更新
            updated_counts = old_counts + new_counts
            active = (old_counts > 0) | (new_counts > 0)
            updated_centroids = old_centroids.copy()
            updated_centroids[active] = (
                old_centroids[active] * old_counts[active, None]
                + new_sums[active]
            ) / updated_counts[active, None].clip(min=1)

            # 重启死 code: 用 batch 内真实样本重新 seed
            if restart_dead_codes:
                dead = (updated_counts == 0)
                if dead.any() and len(residual) > 0:
                    # 每个死 code 随机挑一个 batch 样本作为新 centroid
                    new_idx = np.random.default_rng(l).choice(
                        len(residual), size=int(dead.sum()), replace=True)
                    updated_centroids[dead] = residual[new_idx]
                    updated_counts[dead] = 1

            # 把更新后的码本写回 service 和 backend
            self.rq.codebooks[l] = updated_centroids
            self.rq._layer_counts[l] = updated_counts
            self.rq.backends[l].codebook = self.rq.codebooks[l]

        return self

    def save(self, path: str) -> None:
        """保存 server 状态 (码本 + 缓存) 到 torch 格式。

        用于服务重启恢复; 注意 PyTorch 2.6+ 默认 weights_only=True,
        这里显式传 weights_only=False (因为有 dict 对象)。
        """
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "codebooks": self.rq.codebooks,
            "item_to_sid": self.service.item_to_sid,
            "sid_to_items": self.service.sid_to_items,
        }, path)

    @classmethod
    def load(cls, path: str) -> "OnlineSIDServer":
        """从 save 写出的文件恢复 server (含码本和缓存)。"""
        import torch
        ckpt = torch.load(path, weights_only=False)
        server = cls.from_codebooks(ckpt["codebooks"])
        # 恢复缓存 (item_to_sid / sid_to_items)
        server.service.item_to_sid = ckpt["item_to_sid"]
        server.service.sid_to_items = ckpt["sid_to_items"]
        return server


# ---------------------------------------------------------------------------
# 自测: 端到端验证 (单条 / 批量 / 多路径 / 冷启动 / 健康监控 / 增量更新)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    L, K, D = 5, 256, 64
    # 正交化码本 (单位向量), 让不同 codeword 之间差异明显
    codebooks = np.random.randn(L, K, D).astype(np.float32)
    codebooks /= np.linalg.norm(codebooks, axis=-1, keepdims=True)

    cfg = OnlineSIDConfig(num_layers=L, codebook_size=K, dim=D)
    server = OnlineSIDServer.from_codebooks(codebooks, cfg)

    # 1) 批量入库: 测吞吐
    test_embs = np.random.randn(1000, D).astype(np.float32)
    test_embs /= np.linalg.norm(test_embs, axis=-1, keepdims=True)
    test_ids = [f"item_{i}" for i in range(1000)]

    import time
    t0 = time.time()
    results = server.add_new_item_batch(test_ids, test_embs)
    t1 = time.time()
    print(f"Batch add: 1K items in {(t1-t0)*1000:.1f}ms ({1000/(t1-t0):.0f} QPS)")

    # 2) 单条入库: 测延迟
    t0 = time.time()
    for i in range(1000):
        server.add_new_item(f"new_item_{i}", test_embs[i])
    t1 = time.time()
    print(f"Per-item: 1K items in {(t1-t0)*1000:.1f}ms ({1000/(t1-t0):.0f} QPS)")

    # 3) 多路径: 取 K=3 候选, 按 MSE 排序
    multi = server.add_new_item_with_multipath("multi_path_test", test_embs[0], k_candidates=3)
    print(f"Multi-path: {len(multi)} candidates, "
          f"best mse={multi[0]['mse']:.4f}, worst mse={multi[-1]['mse']:.4f}")

    # 4) 冷启动 query: 用 10 个 item 增强
    item_pool = {f"item_{i}": test_embs[i] for i in range(10)}
    q_result = server.add_new_query_with_cold_start(
        q_text="test query",
        q_emb=np.random.randn(D).astype(np.float32),
        item_pool=item_pool,
        top_k=5,
    )
    print(f"Query cold-start: sid={q_result['sid']}")

    # 5) 健康监控
    health = server.health_check(test_embs, np.array([r["sid"] for r in results]))
    print(f"Health: mse={health.avg_mse:.4f}, "
          f"collision_rate={health.collision_rate*100:.2f}%, "
          f"min_codebook_usage={min(health.codebook_usage)*100:.1f}%")
    needs, reason = health.needs_retrain(mse_baseline=0.5)
    print(f"Retrain decision: {needs} ({reason})")

    # 6) 增量更新: 100 个新样本更新码本
    new_embs = np.random.randn(100, D).astype(np.float32)
    new_embs /= np.linalg.norm(new_embs, axis=-1, keepdims=True)
    server.partial_fit(new_embs)
    print("partial_fit OK")