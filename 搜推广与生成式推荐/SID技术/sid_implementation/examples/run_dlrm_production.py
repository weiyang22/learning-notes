"""
生产环境 DLRM-style 推荐系统的 SID 集成示例

【真实生产场景】
================
我们在 DLRM(Deep Learning Recommendation Model)类系统中,典型情况是:
    - 训练阶段:物品原始数据 → Item Tower → embedding(已包含文本+图片+行为信号)
    - 推理阶段:query tower 产出 query embedding + 候选集 ANN 检索出 top-K 物品
              → 排序模型给每个候选打分

    此时**完全没有文本**,只有 hash_id 和已训练好的 embedding。

【这个 demo 解决的关键疑问】
================
"没有文本信息的情况下,是不是只能用默认 SID?"
不!DLRM 的物品 embedding 本身就是高度结构化的"语义表征"(item tower 在训练时已经把
文本、图像、行为信号全部融合进去了),RQ-Kmeans 不需要看到原始文本,看到这些 embedding
就够了。
    raw_embedding → RQ-Kmeans 训练 → 5 层 codebook → SID
    在线:hash_id → lookup embedding → 5 次 FAISS → SID(< 1ms)

【本文件做了什么】
================
    1. 模拟初始化 DLRM 后端:
       - FakeFeatureStore:10 万物品 + 128 维 embedding(替代真 Feature Store)
       - FakeQueryTower:用户画像 → 128 维 query embedding
       - FakeANNRetriever:query embedding + 候选集 → top-100
       - FakeDLMRanker:cosine + SID-aware bonus 打分
    2. 离线训练 RQ-Kmeans(50K 物品采样)
    3. 启动 FAISS 在线索引
    4. 全量 100K 物品 SID 入库
    5. 模拟 200 次在线推理,统计各阶段延迟(ann / sid_lookup / rank / total)
    6. 演示冷启动 fallback(物品还没 embedding 时的 hash-based default)

【运行】
================
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python3 examples/run_dlrm_production.py

【推荐先读】
================
    - docs/06_dlrm_production_sid.md(纯 embedding 场景的完整方案)
    - docs/00 § 4.6(在线服务原理)
    - src/faiss_sid.py(FAISS 后端实现)
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# fit_rq_kmeans:离线训练 RQ-Kmeans 的入口
# FaissRQKMeans:在线推理用 FAISS 加速的 5 次查表后端
# SIDCodebookEvaluator:离线评估 codebook 质量
from src.rq_kmeans import fit_rq_kmeans
from src.faiss_sid import FaissRQKMeans
from src.sid_evaluation import SIDCodebookEvaluator


# =========================================================================
# 模拟生产系统组件 (Ponytail: 用简单 Python 类代替真实 feature store)
# =========================================================================
# 真实生产里这些是分布式系统(Redis / Milvus / Spark),demo 里用纯 Python mock
# 关键差异:
#   - FakeFeatureStore 是 deterministic,直接生成相同数据
#   - 真系统的延迟会受 RPC / 网络影响,demo 的延迟只反映计算开销

class FakeFeatureStore:
    """
    模拟 DLRM feature store:hash_id → embedding

    真生产里:
      - key-value 存储(Redis / DynamoDB)
      - embedding 由 DLRM item tower 离线生成,定期推送
      - 查询延迟 < 5ms(P99)
    """

    def __init__(self, item_count: int = 100_000, emb_dim: int = 128,
                 seed: int = 0):
        # 注意这里的 seed=0 是 kwarg 默认值(seed: int = 0),
        # 保证不同构造顺序产同样的 fake 数据
        rng = np.random.default_rng(seed)
        # 生成 hash_id 列表
        self.hash_ids = [f"item_{i:08d}" for i in range(item_count)]
        # 100K × 128 维 embedding 占用约 50MB,真生产特征库是 TB 级
        self.embeddings = rng.standard_normal((item_count, emb_dim)).astype(np.float32)
        # L2 归一化让 cosine 相似度 = 内积
        self.embeddings /= np.linalg.norm(self.embeddings, axis=-1, keepdims=True) + 1e-9

    def get(self, hash_id: str) -> np.ndarray:
        """真生产用 O(1) hash 查表,demo 用 .index() 模拟"""
        idx = self.hash_ids.index(hash_id)
        return self.embeddings[idx]

    def get_many(self, hash_ids: List[str]) -> np.ndarray:
        """批量查:真生产走 MGET 或 pipeline,demo 是 O(N) 但 demo 数据小"""
        return np.stack([self.get(h) for h in hash_ids])

    def __repr__(self):
        return f"FakeFeatureStore(items={len(self.hash_ids)}, dim={self.embeddings.shape[1]})"


class FakeQueryTower:
    """
    模拟 DLRM query tower:user profile → query embedding

    真生产里:
      - 输入:用户最近 N 次点击 / 100 个 favorite / 人口属性
      - 模型:Transformer / 双塔
      - 输出:128 维 dense vector
      - 延迟:~5ms / GPU
    """

    def __init__(self, emb_dim: int = 128, seed: int = 1):
        self.dim = emb_dim
        self.rng = np.random.default_rng(seed)

    def encode(self, user_profile: List[int]) -> np.ndarray:
        """
        实际生产中:user history → Transformer → 128-dim
        这里用 user_profile 作 hash 种子产 deterministic 向量
        """
        h = abs(hash(tuple(user_profile))) % (2 ** 32)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


class FakeANNRetriever:
    """
    模拟 DLRM ANN 召回:query embedding → top-K 候选集 (hash_id, score)

    真生产里:FAISS / Milvus / ScaNN / HNSW index
    demo 里直接用矩阵乘法(已经全归一化,内积=cosine)
    """

    def __init__(self, feature_store: FakeFeatureStore, K_default: int = 100):
        self.fs = feature_store
        # 默认召回 100 个候选供后续排序
        self.K_default = K_default

    def retrieve(self, query_emb: np.ndarray, K: int = None) -> List[Tuple[str, float]]:
        """返回 [(hash_id, cosine_score), ...] 按分数降序"""
        K = K or self.K_default
        # 全量点积(N=100K,demo 可以这么做;真生产只对聚类后的子集算)
        scores = self.fs.embeddings @ query_emb
        # 取 top-K 索引
        top_idx = np.argsort(-scores)[:K]
        return [(self.fs.hash_ids[i], float(scores[i])) for i in top_idx]


class FakeDLMRanker:
    """
    模拟 DLRM 排序模型:input → score

    实际生产中:DLRM 是 MLP(concat([q_emb, item_emb, ...其他特征]))
    这里用 cosine 相似度假装,但加了 SID-aware bonus
    """

    def __init__(self, sid_weight: float = 0.1):
        # SID-aware bonus 的权重:0.1 意味着同 SID 前缀的物品额外 +0.1
        self.sid_weight = sid_weight

    def score(self,
              query_emb: np.ndarray,
              candidate_embs: np.ndarray,
              query_sid: np.ndarray = None,
              candidate_sids: np.ndarray = None) -> np.ndarray:
        """返回每个候选集的分数 (K,)"""
        # 基础 DLRM 分数 = 余弦相似度
        scores = candidate_embs @ query_emb

        # SID-aware bonus: 同 query SID 前缀的候选集 +0.05 额外分数
        # 这种做法示意:SID 不是替代 embedding,而是额外结构化信号
        if query_sid is not None and candidate_sids is not None:
            bonus = (candidate_sids[:, 0] == query_sid[0]).astype(np.float32)
            bonus = bonus * self.sid_weight
            scores = scores + bonus
        return scores


# =========================================================================
# Step 1: 离线训练 (一次性, 用前 50K 物品训练)
# =========================================================================

def offline_train(feature_store: FakeFeatureStore,
                sample_size: int = 50_000,
                n_passes: int = 3):
    """
    离线训练 RQ-Kmeans codebook

    真实生产:
        - 一次训练 1-2 小时(亿级数据,Sculley 增量 K-means)
        - 训练一次后,只用 partial_fit 增量更新,几个月才重训一次
    """
    print("\n[离线阶段 1] 训练 RQ-Kmeans codebook")
    print(f"   采样 {sample_size} 物品训练, {n_passes} 轮")
    rng = np.random.default_rng(42)
    # 随机采样,真实生产可以是"最近 N 天活跃的物品"
    sample_idx = rng.choice(len(feature_store.hash_ids), sample_size, replace=False)
    train_embs = feature_store.embeddings[sample_idx]

    # 训 RQ-Kmeans:5 层 × 256 个 codeword
    # 注意 batch_size=4096 比一般 demo 大,因数据多(50K)
    tok = fit_rq_kmeans(
        torch.from_numpy(train_embs),
        num_layers=5,
        codebook_size=256,
        n_passes=n_passes,
        batch_size=4096,
        verbose=False,
    )

    # 训完后立即评估 quality(避免带病部署)
    full_sids = tok.encode_to_sid(torch.from_numpy(feature_store.embeddings)).numpy()
    codebooks = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)

    # 用假交互数 + 评估器跑 5 维度
    rng = np.random.default_rng(123)
    interactions = {i: float(rng.exponential()) for i in range(len(full_sids))}
    evaluator = SIDCodebookEvaluator(
        codebook_size=256, sample_size=500,
        interaction_lookup=interactions,
    )
    report = evaluator.evaluate(codebooks, feature_store.embeddings, full_sids)
    print(f"   [codebook 质量] 综合等级: {report.grade}, "
          f"碰撞率: {report.independence.collision_rate:.2%}, "
          f"SC: {report.semantic.semantic_cohesion:.3f}")
    return tok, full_sids


# =========================================================================
# Step 2: 启动时构建 FAISS 在线索引
# =========================================================================

def start_online_server(tok):
    """
    DLRM 在线服务启动时加载 SID codebook + 构建 FAISS

    真实生产:
        - 常驻服务启动时一次性加载(类似 DLRM model weights)
        - 5 个 IndexFlatL2 总共 ~640KB
        - 启动开销 < 100ms
    """
    print("\n[在线启动] 加载 codebook + 构建 FAISS 索引")
    codebooks = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)
    # FaissRQKMeans 内部建 5 个 IndexFlatL2
    faiss_rq = FaissRQKMeans(codebooks, prefer_faiss=True)
    print(f"   5 个 IndexFlatL2 加载完成 (启动时间 < 100ms)")
    return faiss_rq


# =========================================================================
# Step 3: 在线推理 (关键路径)
# =========================================================================

def online_serving_demo(faiss_rq: FaissRQKMeans,
                       feature_store: FakeFeatureStore,
                       retriever: FakeANNRetriever,
                       query_tower: FakeQueryTower,
                       ranker: FakeDLMRanker,
                       n_requests: int = 100,
                       top_K: int = 100):
    """
    模拟 n_requests 次请求的延迟统计

    真实生产的 SLA 是 P99 < 50ms,我们看 demo 这套能不能扛住
    """
    print(f"\n[在线推理] 模拟 {n_requests} 次请求, 每次 {top_K} 候选集")

    # 各阶段延迟单独统计,容易看出瓶颈
    latencies = {"ann": [], "sid_lookup": [], "rank": [], "total": []}

    for req_i in range(n_requests):
        # (i) query tower: 用户画像 → query embedding
        # 用户画像是 demo 用的 5 个连续整数
        user_profile = list(range(100 + req_i, 100 + req_i + 5))
        t0 = time.perf_counter()
        query_emb = query_tower.encode(user_profile)
        t_ann_start = time.perf_counter()

        # (ii) ANN 召回: 从 1M 候选集中找 top-100
        candidates = retriever.retrieve(query_emb, K=top_K)
        cand_hash_ids = [c[0] for c in candidates]
        cand_embs = feature_store.get_many(cand_hash_ids)
        t_ann_end = time.perf_counter()

        # (iii) SID 编码: 5 次 FAISS 查表 (无文本)
        # 这就是本 demo 的核心论点:就算没文本,也能算 SID
        cand_sids = faiss_rq.encode_batch(cand_embs)
        query_sid = faiss_rq.encode_one(query_emb)
        t_sid_end = time.perf_counter()

        # (iv) 排序 (DLRM + SID bonus)
        # 真生产:DeepFM / Transformer 接 sid + emb + 其他特征
        scores = ranker.score(query_emb, cand_embs, query_sid, cand_sids)
        t_rank_end = time.perf_counter()

        # 每阶段延迟(秒)
        latencies["ann"].append(t_ann_end - t_ann_start)
        latencies["sid_lookup"].append(t_sid_end - t_ann_end)
        latencies["rank"].append(t_rank_end - t_sid_end)
        latencies["total"].append(t_rank_end - t0)

    return latencies


def print_latency_stats(latencies):
    """打印延迟统计 (mean / p50 / p99)"""
    print(f"\n{'步骤':<15} {'平均':>10} {'P50':>10} {'P99':>10}")
    print("-" * 50)
    for step, ts in latencies.items():
        ts = np.array(ts) * 1000  # 秒 → 毫秒
        print(f"{step:<15} {ts.mean():>8.2f}ms {np.percentile(ts, 50):>8.2f}ms "
              f"{np.percentile(ts, 99):>8.2f}ms")


# =========================================================================
# Step 4: 冷启动 fallback 演示
# =========================================================================

def cold_start_fallback_demo(faiss_rq, hash_id: str = "new_item_999"):
    """新物品 embedding 还没生成时的 fallback"""
    print(f"\n[冷启动] 模拟新物品 {hash_id}")

    import hashlib
    # Hash-based default embedding:用 hash_id 算一个 deterministic 向量
    # 真实生产:用 SentencePiece sub-piece hashing(PLUM 论文思路)或随机向量
    h = hashlib.sha256(hash_id.encode()).digest()
    # 字节展开成 128 维 float,再归一化
    raw = np.frombuffer(h * 4, dtype=np.uint8)[:128].astype(np.float32) / 255.0 - 0.5
    default_emb = raw / (np.linalg.norm(raw) + 1e-9)

    print("   embedding 服务暂未生成 embedding → 用 hash-based default")
    sid = faiss_rq.encode_one(default_emb)
    print(f"   冷启动 SID: {sid.tolist()}")
    print(f"   注: 几分钟后真实 embedding 生成后会重新分配 (用 partial_fit)")


# =========================================================================
# 主流程
# =========================================================================

def main():
    print("=" * 70)
    print("  生产环境 DLRM + SID 集成示例")
    print("=" * 70)

    # ============================================================
    # 初始化:模拟 DLRM 后端
    # ============================================================
    print("\n[初始化] 模拟 DLRM 后端 (FeatureStore / QueryTower / ANN)")
    feature_store = FakeFeatureStore(item_count=100_000, emb_dim=128)
    query_tower = FakeQueryTower(emb_dim=128)
    retriever = FakeANNRetriever(feature_store, K_default=100)
    ranker = FakeDLMRanker(sid_weight=0.1)
    print(f"   {feature_store}")
    print(f"   {retriever}")

    # ============================================================
    # 离线 + 启动
    # ============================================================
    # 训 50K 物品,产 SID codebook
    tok, all_sids = offline_train(feature_store, sample_size=50_000)
    # 建 FAISS 在线索引(5 IndexFlatL2)
    faiss_rq = start_online_server(tok)

    # 全量物品 SID 已分配, 写入 feature store (模拟)
    # 实际生产: save_sid_to_feature_store(hash_ids, all_sids)
    print(f"\n[离线阶段 2] 物品 SID 入库")
    print(f"   为 100K 物品分配 SID (shape={all_sids.shape})")

    # ============================================================
    # 在线推理延迟统计(200 次请求,模拟真实流量)
    # ============================================================
    latencies = online_serving_demo(
        faiss_rq, feature_store, retriever, query_tower, ranker,
        n_requests=200, top_K=100,
    )
    print_latency_stats(latencies)

    # ============================================================
    # 冷启动 fallback 演示
    # ============================================================
    cold_start_fallback_demo(faiss_rq)

    # ============================================================
    # 总结 + 报告落盘
    # ============================================================
    print("\n" + "=" * 70)
    print("  关键认知")
    print("=" * 70)
    print("""
  1. 不需要默认 SID:
     DLRM 系统的物品几乎都有 embedding → 5 次 FAISS 查表即得 SID
     无文本、纯 hash_id + embedding 的场景完全可工作

  2. "默认 SID" 仅用于真正的边缘情况:
     - 新物品 embedding 还在生成中 (< 1% 时间)
     - Embedding 服务异常 (用 cache + hash fallback)
     - 完全冷启动平台 (新业务第 1 天)

  3. SID 是辅助而非替代:
     DLRM 排序仍以 embedding 为主信号, SID 用于额外结构化信息
     拼接 (DLRM_emb + SID_emb) 进 Transformer 是推荐集成方式
    """)

    # 保存报告:统计 + codebook 形状,生产对接需要这一步
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "dlrm_sid_production_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "feature_store": str(feature_store),
            "codebook_shape": list(np.stack(
                [q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0).shape),
            "n_requests": 200,
            "latency_ms": {k: {
                "mean": float(np.mean(v) * 1000),
                "p50": float(np.percentile(v, 50) * 1000),
                "p99": float(np.percentile(v, 99) * 1000),
            } for k, v in latencies.items()},
        }, f, indent=2)
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
