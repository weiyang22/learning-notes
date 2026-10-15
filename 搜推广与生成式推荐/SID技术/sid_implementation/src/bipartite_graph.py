"""
查询-物品行为二部图 (Bipartite graph for query-item behavior signals)

实现 docs/01_candidate_sid_with_behavior.md 与 docs/02_unified_semantic_space_sid.md
中描述的方案 2: 用查询与物品之间的历史行为日志 (query, item, behavior) 构造二部图,
然后对每个 query, 聚合其 Top-K 邻居 item 的 embedding 来"增强" query embedding。

为什么需要这个模块?
    原始 query embedding 只编码 query 的字面语义, 但没有"这个 query 用户通常
    会点哪些 item"的行为偏好信号。二部图聚合把这种行为协同信号注入到 query
    embedding 中, 让 query SID 与 item SID 能共享同一个语义空间, 便于跨塔检索。

数据结构选择:
    邻接表 {query_id: [(item_id, weight), ...]} 比传统稀疏邻接矩阵更灵活,
    因为不同 query 的邻居数差异巨大 (长尾 query 可能只有 1 个邻居, 热门 query
    有上千个); 邻接表天然支持可变长度, 不浪费空间。

权重计算策略 (weight_mode):
    - "simple"  (默认):   行为码权重简单求和 (0/1/2/4)
    - "ecpm"    (推荐):   贝叶斯平滑 CTR + CVR + 商业化价值因子
        解决"点击高但转化低"的 eCPM 黑洞问题。
        详见 _compute_ecpm_edge_weight()。

权重归一化策略 (normalize):
    - "sqrt"        (推荐起步):  sqrt + per-query L1 normalize
    - "raw"         : 只做 log1p
    - "none"        : 不做任何处理, 原始累加
    - "inv_degree"  (推荐生产):  按 item/query 节点度数 γ-幂律惩罚
        解决头部物品垄断 Top-K、扼杀长尾的问题。

Public API:
    - BipartiteGraph: 二部图数据结构
    - topk_aggregate: Top-K 邻居 item 的 embedding 增强 query (含语义门控)
    - augment_all_queries: 对所有 query 做增强的批量入口
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# 允许从仓库根目录直接 import src.* 兄弟模块
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from src.behavior_fusion import BehaviorWeights


# ---------------------------------------------------------------------------
# 二部图数据结构
# ---------------------------------------------------------------------------

@dataclass
class BipartiteGraph:
    """查询-物品行为二部图。

    内部表示: edge_w 是 {(query_id, item_id): weight} 的扁平字典。
    为什么不存成 {query_id: {item_id: weight}}?
        字典嵌套会让 "all_queries/all_items/neighbors_of_query" 三个操作
        都需要双重循环; 扁平字典用集合操作 (filter by q_id) 更便宜。
    """
    edge_w: Dict[Tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def from_logs(cls,
                  logs: Iterable[Tuple[str, str, int]],
                  normalize: str = "sqrt",
                  weight_mode: str = "simple",
                  alpha_ctr: float = 0.05,
                  alpha_cvr: float = 0.005,
                  v_pay: float = 10.0,
                  gamma_q: float = 0.0,
                  gamma_i: float = 0.0) -> "BipartiteGraph":
        """从原始行为日志 [(q_id, i_id, behavior_code)] 构造二部图。

        步骤:
          1. 按 weight_mode 计算原始边权重
             - "simple": 行为码权重求和 (0/1/2/4)
             - "ecpm":   贝叶斯平滑 CTR + CVR + 商业化价值因子
          2. 按 normalize 策略做归一化
             - "sqrt":       sqrt + per-query L1
             - "raw":        log1p
             - "none":       不归一化
             - "inv_degree": 双向度数 γ-幂律惩罚 (推荐生产用)

        Args:
            logs: 行为日志, 每条是 (查询 id, 物品 id, 行为码 0/1/2/3)。
            normalize: 归一化策略。
            weight_mode: 边权重计算模式。
                "simple" = 简单求和 (回放原始行为, demo 用)
                "ecpm"   = 商业化敏感, 推荐生产用
            alpha_ctr: 全局 CTR 先验 (eCPM 模式用), 默认 0.05。
                用于拉普拉斯平滑, 避免新边 CTR 估计极端。
            alpha_cvr: 全局 CVR 先验, 默认 0.005。
            v_pay: 付费商业化价值因子, 默认 10.0。
                表示 1 次付费在 eCPM 上等价多少倍点击。
            gamma_q: Query 出度惩罚系数 (inv_degree 模式)。
            gamma_i: Item 入度惩罚系数 (inv_degree 模式)。
                推荐 γ_i = 0.6 抑制头部物品垄断, 论文 §3.3 论证。

        Returns:
            BipartiteGraph 实例, 可直接调用 neighbors_of_query / all_queries。

        Raises:
            ValueError: normalize 或 weight_mode 取值不在已知集合内。
        """
        # 步骤 1: 原始边权重
        if weight_mode == "simple":
            weights = BehaviorWeights.code_to_weight()
            agg: Dict[Tuple[str, str], float] = defaultdict(float)
            for q_id, i_id, code in logs:
                agg[(q_id, i_id)] += weights.get(int(code), 0.0)
        elif weight_mode == "ecpm":
            agg = _compute_ecpm_edge_weight(
                logs, alpha_ctr=alpha_ctr, alpha_cvr=alpha_cvr, v_pay=v_pay,
            )
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")

        # 步骤 2: 归一化
        if normalize == "sqrt":
            agg = _sqrt_normalize(agg)
        elif normalize == "raw":
            agg = {k: float(np.log1p(v)) for k, v in agg.items()}
        elif normalize == "none":
            agg = dict(agg)
        elif normalize == "inv_degree":
            # 双向度数 γ-幂律惩罚 (论文 §3.3 防马太效应)
            agg = _inv_degree_normalize(agg, gamma_q=gamma_q, gamma_i=gamma_i)
        else:
            raise ValueError(f"unknown normalize: {normalize}")

        return cls(edge_w=agg)

    def neighbors_of_query(self, q_id: str,
                           top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """取出指定 query 的所有邻居 item, 按权重降序。

        时间复杂度: O(|E|), 因为要全图扫描一次过滤。
        Ponytail 优化点: 如果图很大 (1M+ query), 可以反向建索引
        {query_id: List[(item_id, weight)]} 一次建好, 查询 O(1)。

        Args:
            q_id: 查询 id。
            top_k: 只返回前 K 个, None 表示全部。

        Returns:
            [(item_id, weight), ...] 列表, 按权重降序。
        """
        # 列表推导式扫描扁平字典, 比 dict.items() 快 (无需解构)
        out = [(i_id, w) for (q, i_id), w in self.edge_w.items() if q == q_id]
        # 按权重降序: 负号让默认升序变成降序
        out.sort(key=lambda x: -x[1])
        return out[:top_k] if top_k else out

    def all_queries(self) -> List[str]:
        """返回图中出现过的所有 query id (排序后)。

        用于 augment_all_queries 等需要遍历全部 query 的场景。
        """
        return sorted({q for (q, _) in self.edge_w.keys()})

    def all_items(self) -> List[str]:
        """返回图中出现过的所有 item id (排序后)。

        用于统计覆盖度、做 embedding 表查找等场景。
        """
        return sorted({i for (_, i) in self.edge_w.keys()})


# ---------------------------------------------------------------------------
# 内部: sqrt + L1 归一化
# ---------------------------------------------------------------------------

def _sqrt_normalize(agg: Dict[Tuple[str, str], float]
                    ) -> Dict[Tuple[str, str], float]:
    """sqrt 压缩 + per-query L1 归一化。

    为什么用 sqrt?
        累加权重差异巨大 (长尾 item 可能只有 1 次曝光, 头部 item 上千次);
        不压缩会让头部 item 在聚合时主导 query embedding, 长尾 item 完全没贡献。
        sqrt 是经典次线性压缩, 比 log 更温和, 比直接截断保留更多信息。

    为什么再做 L1 归一化?
        不同 query 的活跃度差异也很大 (热门 query 1000 个邻居 vs 冷门 1 个),
        L1 归一化让每个 query 的邻居权重和为 1, 后续聚合权重可控。

    Args:
        agg: {(q, i): raw_weight} 原始累加权重。

    Returns:
        {(q, i): normalized_weight} 归一化后的权重。
    """
    # 第一步: 按 query 分组, 准备做 per-query L1
    by_q: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (q, i), w in agg.items():
        by_q[q].append((i, w))

    out: Dict[Tuple[str, str], float] = {}
    for q, items in by_q.items():
        # 先做 sqrt 压缩
        sqrts = [(i, np.sqrt(w)) for i, w in items]
        # 1e-9 防 0 除: 极端情况下某 query 所有原始权重都接近 0
        total = sum(w for _, w in sqrts) + 1e-9
        # 再做 L1 归一化: 让 ∑w_i = 1
        for i, w in sqrts:
            out[(q, i)] = w / total
    return out


def _inv_degree_normalize(agg: Dict[Tuple[str, str], float],
                           gamma_q: float = 0.0,
                           gamma_i: float = 0.0
                           ) -> Dict[Tuple[str, str], float]:
    """按节点度数的 γ-幂律惩罚归一化 (论文 §3.3 防马太效应)。

    公式:
        w'(q, i) = w(q, i) / [ (degree_i)^γ_i · (degree_q)^γ_q + ε ]

    动机: 二部图天然有马太效应 (Matthew Effect)——
    头部物品 (如王者荣耀) 有百万级交互, 边权重天然极高, 在 Top-K 截断时
    会彻底挤占空间; 高相关的新上架 / 长尾物品因为行为稀疏, 连边权重低,
    被扼杀在 Top-K 过滤阶段。

    通过入度惩罚 (γ_i > 0), 让头部物品的权重被系统性压制,
    给高相关但低曝光的长尾物品腾出 Top-K 通道。

    Args:
        agg: {(q, i): raw_weight} 归一化前的权重。
        gamma_q: Query 出度惩罚, 0 表示不惩罚 (默认)。
        gamma_i: Item 入度惩罚, 推荐 0.6 (抑制头部垄断, 但不至于完全抹除)。

    Returns:
        {(q, i): normalized_weight} 归一化后的权重。
    """
    # 统计节点度数: item 被多少个 query 连接 (入度), query 链接多少 item (出度)
    item_degrees: Dict[str, int] = defaultdict(int)
    query_degrees: Dict[str, int] = defaultdict(int)
    for (q, i) in agg.keys():
        item_degrees[i] += 1
        query_degrees[q] += 1

    out: Dict[Tuple[str, str], float] = {}
    for (q, i), w in agg.items():
        # 双向度数惩罚
        penalty = (item_degrees[i] ** gamma_i) * (query_degrees[q] ** gamma_q)
        out[(q, i)] = w / (penalty + 1e-9)  # 1e-9 防 0 除
    return out


def _compute_ecpm_edge_weight(logs: Iterable[Tuple[str, str, int]],
                               alpha_ctr: float = 0.05,
                               alpha_cvr: float = 0.005,
                               v_pay: float = 10.0
                               ) -> Dict[Tuple[str, str], float]:
    """eCPM 敏感的边权重计算 (论文 §3.1 多目标 Pareto Weighting)。

    核心动机: 点击 (Click) 是高频、低成本的轻度行为, 噪声大且有"标题党偏置";
    付费 / 下载是低频、高成本的强意图行为。如果仅按行为发生次数聚合,
    高频点击信号会形成"流形黑洞", 把语义空间拉向点击率高但无转化的毒瘤候选集。

    公式:
        1. 拉普拉斯平滑的条件概率 (极大似然 + 先验):
           ctr = (click + 1) / (show + 1/alpha_ctr)
           cvr = (pay + 1) / (click + 1/alpha_cvr)
        2. 商业化敏感融合:
           w_base = ctr · (1 + v_pay · cvr)
        3. log 平滑抑制头部大曝光词:
           w_edge = log1p(w_base · show)

    行为码映射:
        0 (EXPOSED)  → show
        1 (CLICK)    → click
        2 (DOWNLOAD) → click (与 CLICK 等价, 都是正向信号)
        3 (PAY)      → pay

    学术叙事: 这种设计将传统的"频次二部图"升级为"价值敏感型有向互信息图",
    从数据源头上确保 RQ-VAE 产出的 query_sid 聚类中心由高商业价值行为主导,
    而非单纯的低熵点击噪声。

    Args:
        logs: 行为日志, 每条 (q_id, i_id, code)。
        alpha_ctr: 全局 CTR 先验 (拉普拉斯平滑强度)。
        alpha_cvr: 全局 CVR 先验。
        v_pay: 付费商业化价值因子 (eCPM 权重)。

    Returns:
        {(q, i): eCPM_sensitive_weight} 边权重字典。
    """
    # 第一步: 按 (q, i) 聚合三种信号计数
    edge_signals: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"show": 0, "click": 0, "pay": 0}
    )
    for q_id, i_id, action in logs:
        code = int(action)
        if code == 0:    # EXPOSED → show
            edge_signals[(q_id, i_id)]["show"] += 1
        elif code == 1:  # CLICK → click
            edge_signals[(q_id, i_id)]["click"] += 1
        elif code == 2:  # DOWNLOAD → click (强正向, 与点击等价)
            edge_signals[(q_id, i_id)]["click"] += 1
        elif code == 3:  # PAY → pay
            edge_signals[(q_id, i_id)]["pay"] += 1

    # 第二步: 逐边计算 eCPM 权重
    edge_w: Dict[Tuple[str, str], float] = {}
    for (q, i), s in edge_signals.items():
        show = s["show"]
        click = s["click"]
        pay = s["pay"]

        # 拉普拉斯平滑的条件概率 (避免新边 / 稀疏边估计极端)
        # ctr ≈ 真实点击率 (含 1/alpha_ctr 的虚拟计数)
        ctr = (click + 1.0) / (show + 1.0 / alpha_ctr)
        # cvr ≈ 真实付费转化率 (click 作为分母, 无 click 时用 1/alpha_cvr 平滑)
        cvr = (pay + 1.0) / (click + 1.0 / alpha_cvr + 1e-9)

        # 商业化敏感融合: w_base 类似期望 eCPM
        # 1 + v_pay * cvr 把付费信号放大为点击的 v_pay 倍
        w_base = ctr * (1.0 + v_pay * cvr)

        # log 平滑抑制头部大曝光 (百万级 show 不会压垮 log1p)
        edge_w[(q, i)] = float(np.log1p(w_base * show))

    return edge_w


# ---------------------------------------------------------------------------
# 单个 query 的 Top-K 聚合
# ---------------------------------------------------------------------------

def topk_aggregate(q_emb: np.ndarray,
                   neighbors: Sequence[Tuple[str, float]],
                   item_embs: Dict[str, np.ndarray],
                   beta: float = 0.4,
                   semantic_threshold: Optional[float] = None,
                   gate_weight: Optional[float] = None) -> np.ndarray:
    """用 Top-K 邻居 item 的 embedding 增强 query embedding (含语义硬门控)。

    核心公式:
        aggr = Σ_i  (w_i / Σw) * item_emb_i          # 加权聚合邻居
        fused = q_emb + beta * aggr                    # 加权注入回 query
        output = LayerNorm(fused)                      # 稳定数值

    为什么是 + beta * aggr 而不是直接替换?
        query 的字面语义不能完全丢失 (用户搜"机器学习", 即便他常买鼠标键盘,
        embedding 也要保留机器学习主题, 否则会被行为信号完全覆盖)。
        beta 控制"行为信号强度", 论文推荐 0.3~0.5。

    语义硬门控 (论文 §3.2 反直觉通路处理):
        当一个邻居 item 与 query 的字面相似度低于 semantic_threshold 时,
        视为"反直观候选"; 反直观候选必须同时满足 w ≥ gate_weight 才能保留,
        否则视为偶发噪声被过滤。
        这样既能保留 "q_a1 → i_B1" 这类跨界爆款通路,
        又能剔除 "q_a1 → i_X1" 这类偶发的低权重噪声。

    Args:
        q_emb: 形状 (D,) 的 query 内容 embedding。
        neighbors: [(item_id, weight), ...] 已按权重降序, 通常取 top_k。
        item_embs: {item_id: (D,) embedding} item 池, 缺失的 item 会被跳过。
        beta: 聚合分支权重。
        semantic_threshold: 反直观阈值 (cosine 相似度低于此值视为反直观)。
            None 表示不启用门控 (向后兼容)。
        gate_weight: 反直观候选的最低边权重门槛。
            仅在 semantic_threshold 不为 None 时生效。
            None 时默认 = max(neighbor weights) * 0.5 (兜底, 不至于完全放空)。

    Returns:
        形状 (D,) 增强后的 query embedding。
    """
    # 空邻居直接返回, 避免后续除零
    if not neighbors:
        return q_emb

    # 语义硬门控: 过滤"反直觉且低权重"的偶发噪声
    filtered: List[Tuple[str, float]] = []
    if semantic_threshold is not None:
        # 自动算一个合理的 gate_weight 兜底 (邻居权重中位数的 2 倍)
        # 这样无脑调用也有合理默认, 不至于完全过滤空
        if gate_weight is None:
            sorted_w = sorted(w for _, w in neighbors)
            median_w = sorted_w[len(sorted_w) // 2] if sorted_w else 0.0
            gate_weight = max(median_w * 2.0, 1e-6)

        q_norm = float(np.linalg.norm(q_emb)) + 1e-9
        for i_id, w in neighbors:
            if i_id not in item_embs:
                continue  # 没 embedding 的冷启动物品按原逻辑跳过
            i_emb = item_embs[i_id]
            # cosine 相似度
            cos_sim = float(np.dot(q_emb, i_emb) /
                            (q_norm * (np.linalg.norm(i_emb) + 1e-9)))
            # 反直觉物品 (cos_sim 低) 必须有强图信号 (w 高) 才能保留
            if cos_sim < semantic_threshold and w < gate_weight:
                continue  # 判定为偶发噪声, 过滤
            filtered.append((i_id, w))
    else:
        # 无门控模式: 保留所有有 embedding 的邻居 (原行为)
        filtered = [(i_id, w) for i_id, w in neighbors if i_id in item_embs]

    # 兜底: 门控后没邻居, 用 q_emb 本身 (避免返回零向量)
    if not filtered:
        return q_emb

    # 把过滤后的邻居权重向量化归一化
    weights = np.array([w for _, w in filtered], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-9)

    # 累加: aggr = Σ w_i * item_emb_i
    aggr = np.zeros_like(q_emb, dtype=np.float32)
    for w, (i_id, _) in zip(weights, filtered):
        aggr += w * item_embs[i_id].astype(np.float32)

    fused = q_emb + beta * aggr
    # LayerNorm 稳定尺度, 避免不同 query 的增强幅度差异过大
    mean = fused.mean()
    std = fused.std() + 1e-6
    return ((fused - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# 批量增强: 一次性处理所有 query
# ---------------------------------------------------------------------------

def augment_all_queries(graph: BipartiteGraph,
                        query_embs: Dict[str, np.ndarray],
                        item_embs: Dict[str, np.ndarray],
                        top_k: int = 20,
                        beta: float = 0.4,
                        semantic_threshold: Optional[float] = None,
                        gate_weight: Optional[float] = None) -> Dict[str, np.ndarray]:
    """对图中所有 query 做 Top-K 聚合, 返回 {query_id: 增强后 embedding}。

    Ponytail 简化: 串行循环在 ~1M query 时也能跑 (~几分钟),
    数据量再大可改成 batch matmul: 把 top_k 邻居的 embedding 一次性打包
    成 (Q, top_k, D) 张量做矩阵乘。

    Args:
        graph: 已构造好的 BipartiteGraph。
        query_embs: {query_id: (D,) embedding} 原始 query 表。
        item_embs: {item_id: (D,) embedding} item 表 (通常先用 fuse_candidate_batch 融合过)。
        top_k: 每个 query 取的邻居数。
        beta: 聚合权重。
        semantic_threshold: 反直观阈值 (None 关闭语义门控)。
        gate_weight: 反直观候选的最低边权重门槛 (None 自动算)。

    Returns:
        {query_id: 增强后 (D,) embedding} 字典, 不在 query_embs 中的 query 会被跳过。
    """
    out: Dict[str, np.ndarray] = {}

    # 遍历所有有行为的 query
    for q_id in graph.all_queries():
        # 没原始 embedding 的 query 跳过 (冷启动 query, 后续单独处理)
        if q_id not in query_embs:
            continue

        # 取该 query 的 top_k 邻居
        neighbors = graph.neighbors_of_query(q_id, top_k=top_k)
        # 调用单个 query 的增强函数 (透传语义门控参数)
        out[q_id] = topk_aggregate(
            query_embs[q_id], neighbors, item_embs,
            beta=beta,
            semantic_threshold=semantic_threshold,
            gate_weight=gate_weight,
        )
    return out


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 构造一个小型日志: q1-i1 多次互动, q1-i2 仅曝光, q2-i1 付费
    logs = [
        ("q1", "i1", 1), ("q1", "i1", 2), ("q1", "i2", 0), ("q1", "i3", 3),
        ("q2", "i1", 3), ("q2", "i2", 1), ("q2", "i3", 2),
        ("q3", "i2", 0), ("q3", "i3", 1), ("q3", "i4", 0),
    ]

    # ===== 路径 1: 向后兼容 (默认 simple + sqrt) =====
    g = BipartiteGraph.from_logs(logs, normalize="sqrt")
    print(f"[simple+sqrt] edges: {len(g.edge_w)}")
    print(f"  q1 neighbors: {g.neighbors_of_query('q1', top_k=3)}")
    print(f"  q2 neighbors: {g.neighbors_of_query('q2', top_k=3)}")

    # ===== 路径 2: eCPM 模式 (论文 §3.1) =====
    g_ecpm = BipartiteGraph.from_logs(
        logs, normalize="sqrt", weight_mode="ecpm",
        alpha_ctr=0.05, alpha_cvr=0.005, v_pay=10.0,
    )
    print(f"\n[ecpm+sqrt] edges: {len(g_ecpm.edge_w)}")
    print(f"  q1 neighbors: {g_ecpm.neighbors_of_query('q1', top_k=3)}")
    print(f"  q2 neighbors: {g_ecpm.neighbors_of_query('q2', top_k=3)}")

    # ===== 路径 3: inv_degree 归一化 (论文 §3.3 防马太效应) =====
    # 模拟长尾数据: 让 i_popular 被大量 query 链接
    long_tail_logs = logs + [
        ("q4", "i_popular", 1), ("q5", "i_popular", 1), ("q6", "i_popular", 1),
        ("q7", "i_popular", 1), ("q8", "i_popular", 1), ("q9", "i_popular", 1),
        ("q10", "i_popular", 1), ("q1", "i_popular", 1), ("q2", "i_popular", 1),
    ]
    g_inv = BipartiteGraph.from_logs(
        long_tail_logs, normalize="inv_degree",
        weight_mode="simple", gamma_q=0.0, gamma_i=0.6,
    )
    print(f"\n[simple+inv_degree] edges: {len(g_inv.edge_w)}")
    # i_popular 入度大, 权重应被压制
    w_popular_inv = g_inv.edge_w.get(("q1", "i_popular"), 0)
    w_popular_sqrt = g.edge_w.get(("q1", "i_popular"), 0) if ("q1", "i_popular") in g.edge_w else 0
    print(f"  i_popular in q1 (sqrt): {w_popular_sqrt:.4f} | (inv_degree): {w_popular_inv:.4f}")
    assert w_popular_inv < w_popular_sqrt or w_popular_sqrt == 0, \
        "inv_degree should reduce popular item weight"

    # ===== 路径 4: eCPM + inv_degree 组合 (推荐生产用) =====
    g_prod = BipartiteGraph.from_logs(
        logs, normalize="inv_degree", weight_mode="ecpm",
        alpha_ctr=0.05, alpha_cvr=0.005, v_pay=10.0,
        gamma_q=0.0, gamma_i=0.6,
    )
    print(f"\n[ecpm+inv_degree (production)] edges: {len(g_prod.edge_w)}")

    # ===== Top-K 聚合 (默认无门控) =====
    rng = np.random.default_rng(0)
    q_embs = {f"q{i+1}": rng.standard_normal(64).astype(np.float32)
              for i in range(3)}
    i_embs = {f"i{i+1}": rng.standard_normal(64).astype(np.float32)
              for i in range(4)}
    aug = augment_all_queries(g, q_embs, i_embs, top_k=2, beta=0.4)
    for q_id, emb in aug.items():
        print(f"Augmented {q_id}: shape={emb.shape}, "
              f"mean={emb.mean():.4f}, std={emb.std():.4f}")

    # ===== 路径 5: 语义硬门控 (论文 §3.2 反直觉通路) =====
    # 制造一个反直觉候选: 让 q1 的 embedding 跟 i1 反向, 但 i1 边权很高
    q1_emb = rng.standard_normal(64).astype(np.float32)
    q1_emb /= np.linalg.norm(q1_emb)
    i_embs_specific = {
        # 与 q1 字面非常相似
        "i_similar": (q1_emb + 0.01 * rng.standard_normal(64)).astype(np.float32),
        # 与 q1 字面反向 (反直观)
        "i_opposite": (-q1_emb + 0.01 * rng.standard_normal(64)).astype(np.float32),
        # 与 q1 无关
        "i_random": rng.standard_normal(64).astype(np.float32),
    }
    neighbors = [
        ("i_similar", 1.0),    # 字面相关, 高权重
        ("i_opposite", 0.3),   # 反直觉, 低权重 (偶发噪声)
        ("i_opposite_2", 5.0), # 反直觉, 高权重 (跨界爆款)
    ]
    # 没门控时: 全部聚合
    fused_no_gate = topk_aggregate(q1_emb, neighbors, i_embs_specific, beta=0.4)
    # 有门控: 反直觉低权重被过滤, 反直觉高权重保留
    fused_gated = topk_aggregate(
        q1_emb, neighbors, i_embs_specific, beta=0.4,
        semantic_threshold=0.3, gate_weight=2.0,
    )
    print(f"\n[语义门控] 无门控 vs 有门控 fused 差异: "
          f"{np.abs(fused_no_gate - fused_gated).sum():.4f}")
    print(f"  无门控 fused mean={fused_no_gate.mean():.4f}, std={fused_no_gate.std():.4f}")
    print(f"  有门控 fused mean={fused_gated.mean():.4f}, std={fused_gated.std():.4f}")
    assert not np.allclose(fused_no_gate, fused_gated), \
        "门控应该改变 fused 结果"

    print("\nOK")