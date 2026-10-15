"""
Online SID Server demo: production scenario with cold start + new items.

【为什么这个 demo 存在】
================
前面几个 demo 都是"离线 batch"流程:
    1. 训好 codebook → 把所有 5000 物品一次性 encode 出来 → 存起来
但真实生产系统需要"在线":
    - 流量高峰每秒新增 1000+ 物品,必须**实时**分配 SID
    - 多数物品已经有 embedding,5 次 FAISS 查表即得 SID(< 1ms)
    - 少数物品还没 embedding(冷启动),需要 fallback 路径
    - 偶尔要 partial_fit 新数据,不能让全量 codebook 滞后于分布漂移

本 demo 模拟"一个真实 DLRM 在线服务"的所有关键路径。

【本文件做了什么】
================
    1. 离线训 RQ-Kmeans(5000 物品 → 5 层 × 256 codeword)
    2. 启动 OnlineSIDServer(自带 FAISS IndexFlatL2)
    3. warm path:5000 existing + 1000 new 带 embedding 的物品(< 1ms/批)
    4. cold path:100 个没 embedding 的物品(用 cold_start_embedder 占位)
    5. multi-path:对模糊物品给 K 个候选 SID(DREAM 论文风格)
    6. cold-start query:用 Top-K 物品聚合算 query SID(二部图)
    7. health_check:重训决策(MSE / 利用率 / 碰撞率)
    8. partial_fit:增量更新 codebook(Sculley 2010 mini-batch)

【运行】
================
    python3 examples/run_online_sid_server.py

【推荐先读】
================
    - docs/00_sid_overview_and_principles.md § 4.6(在线服务)
    - docs/04_online_sid_generation.md(生产级详细方案)
    - src/online_sid_server.py(本例背后的服务实现)
    - src/faiss_sid.py(FAISS 后端)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# fit_rq_kmeans:离线训 RQ-Kmeans
# OnlineSIDServer / OnlineSIDConfig:本 demo 的主角
from src.rq_kmeans import fit_rq_kmeans
from src.online_sid_server import OnlineSIDServer, OnlineSIDConfig


def fake_cold_start_embedder(item_id: str) -> np.ndarray:
    """
    冷启动时的"假 embedder":
    - 输入:item_id 字符串
    - 输出:确定性 64 维向量
    实现:用 hash(item_id) 作种子跑 rng,然后 L2 normalize
    同 item_id 永远出同一个 embedding(可复现、零依赖)。
    这只是 demo 占位,生产里会换成真 LLM/视觉 embedder。
    """
    h = hash(item_id) % (2 ** 32)
    rng = np.random.default_rng(h)
    emb = rng.standard_normal(64).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def main():
    print("=" * 70)
    print("ONLINE SID SERVER DEMO")
    print("=" * 70)

    # ============================================================
    # Step 1:离线训练 RQ-Kmeans(一次性,5K 物品)
    # ============================================================
    print("\n[1] Train RQ-Kmeans codebook on initial 5000 items")
    # 5000 个 L2-归一化随机向量当作假 embedding(实际生产来自 DLRM item tower)
    torch_embeddings = np.random.randn(5000, 64).astype(np.float32)
    torch_embeddings /= np.linalg.norm(torch_embeddings, axis=-1, keepdims=True)
    import torch
    tok = fit_rq_kmeans(torch.from_numpy(torch_embeddings),
                         num_layers=5, codebook_size=256,
                         n_passes=3, batch_size=512, verbose=False)
    # 把每层 codebook 拼成 (L, K, D) 数组给 server
    codebooks = np.stack([q.embeddings.cpu().numpy() for q in tok.rq.levels], axis=0)
    print(f"  Codebook shape: {codebooks.shape}")  # (5, 256, 64)

    # ============================================================
    # Step 2:启动 Online SID Server
    # ============================================================
    cfg = OnlineSIDConfig(
        num_layers=5, codebook_size=256, dim=64,
        cold_start_embedder=fake_cold_start_embedder,  # 没 embedding 时调用
        multi_path_k=3,                                  # 多路径推理给 K 个候选
    )
    # from_codebooks 会在内部建 5 个 FAISS IndexFlatL2(一次 < 100ms)
    server = OnlineSIDServer.from_codebooks(codebooks, cfg)
    print("  Server initialized with FAISS lookup")

    # ============================================================
    # Step 3:Warm path - 处理已有 embedding 的物品
    # ============================================================
    # 这是真实生产的"主路径",绝大多数新增物品都已经有 embedding
    # 单物品 < 1ms,批量走 FAISS 内部并行
    print("\n[2] Process existing items (warm path, 5K items)")
    existing_ids = [f"existing_{i}" for i in range(5000)]
    t0 = time.time()
    server.add_new_item_batch(existing_ids, torch_embeddings)
    elapsed = time.time() - t0
    print(f"  5K items in {elapsed*1000:.1f}ms ({5000/elapsed:.0f} QPS)")

    # ============================================================
    # Step 4:Warm path - 新物品(仍有 embedding)
    # ============================================================
    print("\n[3] Process new items (warm path with embeddings)")
    new_ids = [f"new_{i}" for i in range(1000)]
    new_embs = np.random.randn(1000, 64).astype(np.float32)
    new_embs /= np.linalg.norm(new_embs, axis=-1, keepdims=True)
    t0 = time.time()
    server.add_new_item_batch(new_ids, new_embs)
    elapsed = time.time() - t0
    print(f"  1K items in {elapsed*1000:.1f}ms ({1000/elapsed:.0f} QPS)")

    # ============================================================
    # Step 5:Cold-start - 没 embedding 的物品
    # ============================================================
    # 这种叫"type 3 冷启动"(新物品全无 embedding)
    # server 内部会调用 cold_start_embedder (本 demo 是 fake hash 兜底)
    # 真实生产会通过 LLM/视觉 embedding 服务在 ~50ms 内生成真 embedding
    print("\n[4] Process cold-start items (no embedding available)")
    cold_ids = [f"cold_{i}" for i in range(100)]
    t0 = time.time()
    cold_results = []
    for cid in cold_ids:
        result = server.add_new_item(cid, embedding=None, cold_start=True)
        cold_results.append(result)
    elapsed = time.time() - t0
    print(f"  100 cold-start items in {elapsed*1000:.1f}ms ({100/elapsed:.0f} QPS)")
    print(f"  Sample cold-start SID: {cold_results[0]['sid']}")

    # ============================================================
    # Step 6:多路径推理(multi-path)- DREAM 论文(arXiv 2606.06947)风格
    # ============================================================
    # 当一个物品的 embedding 离两个 code 都很近时,可以保留 K 条候选 SID
    # 让下游排序模型自己选(或者投票)
    print("\n[5] Multi-path inference for ambiguous item")
    sample_emb = new_embs[0]
    multi = server.add_new_item_with_multipath("ambiguous_item", sample_emb, k_candidates=3)
    for i, m in enumerate(multi):
        print(f"  Path {i+1}: sid={m['sid']}, mse={m['mse']:.4f}")

    # ============================================================
    # Step 7:Cold-start query(用二部图 Top-K 增强)
    # ============================================================
    # 真实搜索词的冷启动:
    #   query 是个新 query,自己没 embedding
    #   但日志已经记录了它跟哪些物品共现
    #   → 用 Top-K 物品的 embedding 加权聚合,作为 query 的近似 embedding
    print("\n[6] Cold-start query with Top-K item aggregation")
    item_pool = {f"existing_{i}": torch_embeddings[i] for i in range(50)}
    q_result = server.add_new_query_with_cold_start(
        q_text="iPhone 15 Pro Max case",  # 仅用于日志/debug,不参与计算
        q_emb=np.random.randn(64).astype(np.float32),  # demo 用随机向量
        item_pool=item_pool,        # Top-K 池
        top_k=10,                   # 取行为最重的 10 个 item
    )
    print(f"  Query SID: {q_result['sid']}")

    # ============================================================
    # Step 8:健康检查 + 自动重训决策
    # ============================================================
    # health_check 内部算 5 个指标:
    #   - 平均重建 MSE
    #   - 每层 codebook 利用率
    #   - 碰撞率
    #   - SC / PD(语义性)
    # needs_retrain 用阈值判断要不要重训
    print("\n[7] Health check + retrain decision")
    sample_test_embs = np.random.randn(100, 64).astype(np.float32)
    sample_test_embs /= np.linalg.norm(sample_test_embs, axis=-1, keepdims=True)
    sample_test_sids = server.add_new_item_batch([f"test_{i}" for i in range(100)], sample_test_embs)
    sample_test_sids = np.array([r["sid"] for r in sample_test_sids])
    health = server.health_check(sample_test_embs, sample_test_sids)
    print(f"  Metrics: {json.dumps(health.to_dict(), indent=2)}")
    needs, reason = health.needs_retrain(mse_baseline=0.5, min_usage=0.7, max_collision_rate=0.1)
    print(f"  Retrain decision: {needs} ({reason})")

    # ============================================================
    # Step 9:partial_fit - 增量更新 codebook
    # ============================================================
    # 论文 arXiv 2604.13273 提出"不需要全量重训,新数据走 Sculley 增量更新"
    # 大约 8-9x 计算节省
    print("\n[8] Incremental codebook update (partial_fit)")
    fresh_embs = np.random.randn(500, 64).astype(np.float32)
    fresh_embs /= np.linalg.norm(fresh_embs, axis=-1, keepdims=True)
    server.partial_fit(fresh_embs)
    print("  partial_fit applied to all 5 codebook levels")

    # 重做一次 health_check 看是否好转
    print("\n[9] After partial_fit: health recheck")
    health2 = server.health_check(sample_test_embs, sample_test_sids)
    print(f"  avg_mse={health2.avg_mse:.4f}, "
          f"min_usage={min(health2.codebook_usage)*100:.1f}%")

    # ============================================================
    # Step 10:序列化 server 状态
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    server.save(str(out_dir / "online_sid_server.pt"))
    print(f"\nServer state saved to {out_dir}/online_sid_server.pt")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
