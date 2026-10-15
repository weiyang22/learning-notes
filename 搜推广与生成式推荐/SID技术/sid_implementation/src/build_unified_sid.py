"""
统一语义空间 SID (方案 2)

实现 docs/01_candidate_sid_with_behavior.md 中描述的方案 2:
为 query 和 item 同时生成 SID, 且让前几层 (shared) 共享同一个码本,
后几层 (individual) 各自一套码本, 从而保证 query 和 item SID 能直接对比。

为什么需要"统一空间"?
    方案 1 只对 item 编码 SID, query 端需要做 KNN 检索 (召回), 效率受限。
    方案 2 让 query 和 item SID 共享前缀 (前几层), 可以:
      - 用前缀直接做倒排索引召回, O(1) 命中
      - 直接计算 query/item SID 的 Hamming 距离做粗排
      - 支持冷启动 query (共享前缀让新 query 也能匹配历史 item SID)

流水线:
    1. 构造二部图 (query, item, behavior) → 行为加权邻接
    2. 增强 query embedding: 用 Top-K item 聚合 (topk_aggregate)
    3. 融合 item embedding: 行为融合 (fuse_candidate_batch)
    4. 训练 UnifiedRQVAE: shared 层 + 个体层
    5. 编码 query SID + item SID (二者共享前 num_shared 层)

Public API:
    - UnifiedSIDBuilder: 端到端 builder
    - build_unified_sid: 一次性 fit + encode 的便捷函数
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# 让脚本可以直接 python -m src.build_unified_sid 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from src.behavior_fusion import fuse_candidate_batch
from src.bipartite_graph import (
    BipartiteGraph, augment_all_queries, topk_aggregate,
)
from src.rq_vae import UnifiedRQVAE, RQVAEConfig


@dataclass
class UnifiedSIDConfig:
    """统一空间 SID 的全部超参。

    关键约束: num_shared + num_individual == num_layers
        论文推荐 2 共享 + 3 个体: query/item SID 前 2 层相同 (语义大类),
        后 3 层各自专属 (细节差异)。
    """
    input_dim: int = 768           # 输入 embedding 维度
    hidden_dim: int = 512          # encoder 中间层维度
    latent_dim: int = 256          # encoder 输出维度
    num_layers: int = 5            # 总层数 (= shared + individual)
    codebook_size: int = 256       # 每层码本 K
    num_shared: int = 2            # 共享层数 (query 和 item 共用同一套码本)
    num_individual: int = 3        # 个体层数 (query 3 层独立 + item 3 层独立)
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    topk: int = 20                 # query 增强时聚合的 Top-K item 数
    alpha_behavior: float = 0.4    # item 行为融合权重
    beta_topk: float = 0.4         # query Top-K 聚合权重
    device: str = "cpu"


class UnifiedSIDBuilder:
    """统一空间 SID 的端到端构建器。"""

    def __init__(self, cfg: Optional[UnifiedSIDConfig] = None):
        self.cfg = cfg or UnifiedSIDConfig()
        # 关键约束: shared + individual 必须等于总层数, 否则模型结构无法拆分
        assert self.cfg.num_shared + self.cfg.num_individual == self.cfg.num_layers
        self.model: Optional[UnifiedRQVAE] = None

    def fit(self,
            query_embs: np.ndarray,
            item_embs: np.ndarray,
            behaviors_list: Sequence[Sequence[Tuple[str, int]]],
            graph: BipartiteGraph,
            query_ids: Optional[Sequence[str]] = None,
            item_ids: Optional[Sequence[str]] = None,
            verbose: bool = True) -> "UnifiedSIDBuilder":
        """训练 UnifiedRQVAE。

        步骤:
            1. 融合 item embedding (行为信号)
            2. 增强 query embedding (Top-K item 聚合)
            3. 构造 UnifiedRQVAE 模型
            4. 训练: query 和 item 共享前 num_shared 层, 个体后 num_individual 层

        Args:
            query_embs: (Q, D) 查询 embedding。
            item_embs: (I, D) 物品 embedding。
            behaviors_list: 长度 I 的列表, 每个物品的行为日志。
            graph: 已构造好的二部图, 用于 query 增强。
            query_ids / item_ids: 可选 id 列表, 用于构造 dict; 不传则用 "q_0/q_1/..."。
            verbose: 是否打印训练日志。

        Returns:
            self, 支持链式调用。
        """
        # 第一步: 融合 item embedding (行为注入)
        item_embs_fused = fuse_candidate_batch(
            item_embs, behaviors_list,
            alpha=self.cfg.alpha_behavior,
        )

        # 第二步: 构造 id 映射 (None 时自动命名)
        if query_ids is None:
            query_ids = [f"q_{i}" for i in range(len(query_embs))]
        if item_ids is None:
            item_ids = [f"i_{i}" for i in range(len(item_embs))]

        # id → embedding 字典, 方便后续按 id 查找
        query_emb_map = {q_id: emb for q_id, emb in zip(query_ids, query_embs)}
        item_emb_map = {i_id: emb for i_id, emb in zip(item_ids, item_embs_fused)}

        # 第三步: 增强 query embedding
        # 用融合后的 item embedding 作为 item_pool, 喂给 bipartite_graph 的增强函数
        augmented = augment_all_queries(
            graph, query_emb_map, item_emb_map,
            top_k=self.cfg.topk, beta=self.cfg.beta_topk,
        )
        # 把 dict 重新堆叠成 (Q, D) 矩阵, 缺失的 query 用原始 embedding 兜底
        q_embs_aug = np.stack(
            [augmented.get(q_id, query_emb_map[q_id]) for q_id in query_ids]
        )

        # 第四步: 构造 UnifiedRQVAE
        rqvae_cfg = RQVAEConfig(
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            latent_dim=self.cfg.latent_dim,
            num_layers=self.cfg.num_layers,
            codebook_size=self.cfg.codebook_size,
        )
        self.model = UnifiedRQVAE(
            rqvae_cfg,
            num_shared=self.cfg.num_shared,
            num_individual=self.cfg.num_individual,
        ).to(self.cfg.device)

        # 第五步: 训练循环
        # 把 numpy 转成 torch 并搬到目标设备
        q_t = torch.from_numpy(q_embs_aug).float().to(self.cfg.device)
        i_t = torch.from_numpy(item_embs_fused).float().to(self.cfg.device)

        # Adam 只优化 encoder/decoder + 个体 quantizer (shared quantizer 在
        # forward 内被 query/item 两路共同更新, 是否走 EMA 取决于 use_ema 配置)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        self.model.train()
        for epoch in range(self.cfg.epochs):
            # 每轮重新打散样本顺序
            perm = torch.randperm(q_t.size(0), device=self.cfg.device)
            epoch_loss = 0.0
            steps = 0
            # mini-batch 循环
            for start in range(0, q_t.size(0), self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                opt.zero_grad()
                # forward 返回 (loss, info_dict)
                loss, info = self.model(q_t[idx], i_t[idx])
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                steps += 1

            # 日志: 控制在 max(epochs//10, 1) 轮打印一次, 避免刷屏
            if verbose and ((epoch + 1) % max(self.cfg.epochs // 10, 1) == 0
                            or epoch == 0):
                print(f"[unified-sid] epoch {epoch+1}/{self.cfg.epochs}  "
                      f"loss={epoch_loss/max(steps,1):.4f}  "
                      f"recon_q={info['recon_q']:.4f}  "
                      f"recon_i={info['recon_i']:.4f}  "
                      f"vq={info['vq_loss']:.4f}")
        self.model.eval()  # 训练完切到推理模式
        return self

    @torch.no_grad()
    def encode_queries(self, query_embs: np.ndarray) -> np.ndarray:
        """把 query embedding 编码成 SID。

        必须先 fit()。

        Args:
            query_embs: (Q, D) query embedding。

        Returns:
            (Q, num_layers) int64 SID 数组。

        Raises:
            RuntimeError: 未训练就调用。
        """
        if self.model is None:
            raise RuntimeError("Call fit() before encode_queries()")
        t = torch.from_numpy(np.asarray(query_embs, dtype=np.float32)).to(self.cfg.device)
        # encode_query_to_sid 内部: encoder → shared quantizer → individual_q
        sids = self.model.encode_query_to_sid(t).cpu().numpy().astype(np.int64)
        return sids

    @torch.no_grad()
    def encode_items(self, item_embs: np.ndarray,
                     behaviors_list: Optional[Sequence[Sequence[Tuple[str, int]]]] = None,
                     ) -> np.ndarray:
        """把 item embedding 编码成 SID。

        与 encode_queries 不同: 这里会再次做行为融合, 因为新物品
        的历史行为可能与训练时不同。

        Args:
            item_embs: (I, D) 物品 embedding。
            behaviors_list: 可选, 长度 I 的行为日志; 不传则跳过融合。

        Returns:
            (I, num_layers) int64 SID 数组。

        Raises:
            RuntimeError: 未训练就调用。
        """
        if self.model is None:
            raise RuntimeError("Call fit() before encode_items()")
        if behaviors_list is not None:
            emb = fuse_candidate_batch(item_embs, behaviors_list,
                                       alpha=self.cfg.alpha_behavior)
        else:
            emb = item_embs
        t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(self.cfg.device)
        sids = self.model.encode_item_to_sid(t).cpu().numpy().astype(np.int64)
        return sids


def build_unified_sid(query_embs: np.ndarray,
                      item_embs: np.ndarray,
                      behaviors_list: Sequence[Sequence[Tuple[str, int]]],
                      logs: Sequence[Tuple[str, str, int]],
                      query_ids: Optional[Sequence[str]] = None,
                      item_ids: Optional[Sequence[str]] = None,
                      cfg: Optional[UnifiedSIDConfig] = None,
                      ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """一键完成 fit + encode, 返回 (query_sids, item_sids) 两个字典。

    Ponytail 风格: 一个函数搞定全部流程。
    与 build_candidate_sid 的区别: 同时返回 query SID, 需要 logs 输入。

    Args:
        query_embs: (Q, D) query embedding。
        item_embs: (I, D) item embedding。
        behaviors_list: 长度 I 的物品行为日志。
        logs: (q_id, i_id, behavior_code) 三元组列表, 用于构造二部图。
        query_ids / item_ids: 可选 id 列表。
        cfg: 可选训练配置。

    Returns:
        ({q_id: [s1..sL]}, {i_id: [s1..sL]}) 两个字典。
    """
    # 第一步: 用 logs 构造二部图 (默认用推荐的 sqrt 归一化)
    graph = BipartiteGraph.from_logs(logs)
    builder = UnifiedSIDBuilder(cfg)

    # 第二步: 训练
    builder.fit(query_embs, item_embs, behaviors_list, graph,
                query_ids=query_ids, item_ids=item_ids)

    # 第三步: 推断
    # 这里再次给 query_ids / item_ids 做默认填充, 与 fit 中的命名一致
    if query_ids is None:
        query_ids = [f"q_{i}" for i in range(len(query_embs))]
    if item_ids is None:
        item_ids = [f"i_{i}" for i in range(len(item_embs))]

    q_sids = builder.encode_queries(query_embs)
    i_sids = builder.encode_items(item_embs, behaviors_list)

    # 转成 dict 方便 JSON 化
    return (
        {q: s.tolist() for q, s in zip(query_ids, q_sids)},
        {i: s.tolist() for i, s in zip(item_ids, i_sids)},
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    Nq, Ni, D = 200, 400, 64  # 200 query + 400 item, 64 维

    # 随机 query / item embedding
    q_embs = rng.standard_normal((Nq, D)).astype(np.float32)
    i_embs = rng.standard_normal((Ni, D)).astype(np.float32)

    # 每个 item 1~20 条行为
    behaviors_list = [
        [(f"u_{rng.integers(0, 50)}", int(rng.integers(0, 4)))
         for _ in range(rng.integers(1, 20))]
        for _ in range(Ni)
    ]

    # 随机生成 2000 条 (q, i, behavior) 日志用于构造二部图
    logs = [
        (f"q_{rng.integers(0, Nq)}", f"i_{rng.integers(0, Ni)}",
         int(rng.integers(0, 4)))
        for _ in range(2000)
    ]

    # 缩小版配置: 让自测在几秒内跑完
    cfg = UnifiedSIDConfig(
        input_dim=D, hidden_dim=64, latent_dim=32,
        num_layers=5, codebook_size=256,
        num_shared=2, num_individual=3,
        epochs=10, batch_size=64, topk=10,
    )

    q_sids, i_sids = build_unified_sid(
        q_embs, i_embs, behaviors_list, logs, cfg=cfg,
    )
    print(f"Q SID samples: {list(q_sids.items())[:2]}")
    print(f"I SID samples: {list(i_sids.items())[:2]}")

    # 验证 shared 层是否真的有共享
    # 取前 num_shared=2 层做集合对比, 应该有显著重叠
    q_shared_keys = set(tuple(s[:2]) for s in q_sids.values())
    i_shared_keys = set(tuple(s[:2]) for s in i_sids.values())
    print(f"Q shared (s1,s2) keys unique: {len(q_shared_keys)}")
    print(f"I shared (s1,s2) keys unique: {len(i_shared_keys)}")
    overlap = q_shared_keys & i_shared_keys
    print(f"Shared (s1,s2) overlap between Q and I: {len(overlap)}")
    print("OK")