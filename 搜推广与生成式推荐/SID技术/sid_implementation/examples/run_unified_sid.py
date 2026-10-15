"""
Example: build unified semantic space SID for query + item (方案 2).

【方案 2 解决了什么】
================
方案 1 只为"候选集"生成 SID;在搜索/广告场景,我们**也需要为搜索词(query)生成 SID**,
且 query 和 item 应该**在同一个语义空间** —— 这样下游模型才能直接
"对比 query SID 和 item SID" 做事。

传统做法:
    - q_emb → 单独训一个 RQ-VAE
    - i_emb → 单独训一个 RQ-VAE
    - 两套 codebook 完全独立,失去跨域对齐能力

方案 2 的做法(本仓库默认配置):
    - 底层 L_low = 2 层 codebook:由 query / item 共享,捕获"粗粒度类别"
    - 高层 L_high = 3 层 codebook:query / item 各自一套,捕获"细粒度特征"
    - 最终 SID = [shared_s1, shared_s2, q_or_i_s3, q_or_i_s4, q_or_i_s5]
    - 共 5 层,但底层一致确保"同语义类别"对齐

【二部图(Bipartite Graph)在其中的角色】
================
搜索词本身信息密度低,需要"借"相邻候选集增强 embedding:
    1. 用日志 (q, i, behavior_code) 构二部图
    2. 给每个查询找 top-K 行为最重的候选集
    3. 用这 K 个候选集 embedding 加权聚合,补强 q_emb
    4. 增强后的 q_emb 喂给共享 RQ-VAE

【本文件做了什么】
================
    1. 构造合成数据(500 候选 + 200 查询 + 80 用户 + 3000 条日志)
    2. 训练 UnifiedRQVAE(共享 2 层 + 各自 3 层)
    3. 同时产出 query_sids 和 item_sids
    4. 检查 query / item 在底层 codebook 上的重叠度

【运行】
================
    python3 examples/run_unified_sid.py

【推荐先读】
================
    - docs/02_unified_semantic_space_sid.md(详细方案)
    - src/bipartite_graph.py(二部图构建 / Top-K 聚合)
    - src/build_unified_sid.py(本例的封装)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np

# make_synthetic_dataset:这次的配置和方案 1 不一样:
#   - n_queries > 0  : 生成 query 节点和 query embedding
#   - logs_n > 0     : 生成 (q, i, code) 三元组日志(用于构二部图)
from src.data_loader import make_synthetic_dataset
# UnifiedSIDConfig / build_unified_sid:封装"二部图增强 + UnifiedRQVAE"
from src.build_unified_sid import (
    UnifiedSIDConfig, build_unified_sid,
)


def main():
    # ============================================================
    # Step 1: 生成带"query + item + 行为日志"的合成数据集
    # ============================================================
    # n_candidates=500, n_queries=200 : query 数量 < item 数量,符合搜索场景
    # n_users=80 : 生成 q-i 边的用户池
    # emb_dim=128 : 各节点的 embedding 维度
    # behaviors_per_item_mean=8 : 每个 item 平均 8 个行为事件
    # logs_n=3000 : 总共 3000 条 (q, i, code) 日志
    # seed=1       : 固定随机种子让结果可复现
    ds = make_synthetic_dataset(
        n_candidates=500, n_queries=200, n_users=80,
        emb_dim=128, behaviors_per_item_mean=8, logs_n=3000,
        seed=1,
    )

    # ============================================================
    # Step 2: 配置 UnifiedRQVAE
    # ============================================================
    # UnifiedSIDConfig 在 CandidateSIDConfig 基础上加了两个新字段:
    #   - num_shared=2      : 底层共享 codebook 层数(粗粒度对齐)
    #   - num_individual=3  : 高层各 codebook 层数(细粒度个性化)
    # 加上 alpha_behavior / beta_topk 控制行为/二部图注入强度
    cfg = UnifiedSIDConfig(
        input_dim=128, hidden_dim=128, latent_dim=64,
        num_layers=5, codebook_size=256,   # 总 SID 长 5
        num_shared=2, num_individual=3,    # 2 + 3 = 5
        epochs=20, batch_size=64, lr=1e-3,
        topk=10,             # 每个 query 聚合 top-10 候选集增强
        alpha_behavior=0.4,  # 行为信号强度(同方案 1)
        beta_topk=0.4,       # 二部图 Top-K 聚合强度
    )

    # ============================================================
    # Step 3: 一键训练 + 编码
    # ============================================================
    # build_unified_sid 内部完成几件事:
    #   (a) 构二部图 (BipartiteGraph.from_logs)
    #   (b) 对每个 query:用 Top-K 候选集加权聚合作增强
    #   (c) 行为融合(同方案 1)
    #   (d) 训练 UnifiedRQVAE:共享 2 层 + Q 单独 3 层 + I 单独 3 层
    #   (e) 对所有 query 和 item 跑 encode
    q_sids, i_sids = build_unified_sid(
        ds.query_embs, ds.candidate_embs,
        ds.behaviors_per_item, ds.logs,
        query_ids=ds.query_ids, item_ids=ds.candidate_ids,
        cfg=cfg,
    )
    # q_sids : Dict[qid, [s1..s5]]
    # i_sids : Dict[iid, [s1..s5]]

    # ============================================================
    # Step 4: 保存结果
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    # query 和 item 的 SID 各自一个文件
    with open(out_dir / "query_sids.json", "w") as f:
        json.dump(q_sids, f, indent=2)
    with open(out_dir / "item_sids.json", "w") as f:
        json.dump(i_sids, f, indent=2)

    # ============================================================
    # Step 5: 简单检查 - 共享 codebook 的对齐效果
    # ============================================================
    # 提取每个 SID 的前 2 层(共享部分),统计 query/item 各自用了多少独立 code
    # 同时计算 Q ∩ I 用了多少共享 code
    # Q 与 I 共享 code 越多,说明底层"粗粒度语义"对齐得越好
    q_shared = {tuple(s[:2]) for s in q_sids.values()}
    i_shared = {tuple(s[:2]) for s in i_sids.values()}
    print(f"Built Q SID: {len(q_sids)}, I SID: {len(i_sids)}")
    print(f"Q unique shared (s1,s2): {len(q_shared)}")
    print(f"I unique shared (s1,s2): {len(i_shared)}")
    print(f"Overlap Q-I shared codes: {len(q_shared & i_shared)}")
    print(f"Sample Q SID: {list(q_sids.items())[0]}")
    print(f"Sample I SID: {list(i_sids.items())[0]}")
    print(f"Saved to {out_dir}/{{query,item}}_sids.json")


if __name__ == "__main__":
    main()
