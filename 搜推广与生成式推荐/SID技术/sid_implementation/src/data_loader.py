"""
轻量级数据加载器与合成数据生成器

为 SID 流水线提供两件事:
  1. 合成数据生成: 不依赖真实日志就能跑通整个流水线 (单元测试 / demo)
  2. 真实数据加载: 从磁盘 .npy + manifest CSV 读入 LLM embedding

Ponytail 原则:
  - 只有一个合成数据生成器 + 一个 npy 加载器, 不引入 spark/dask/pandas ETL
  - 接口尽量简单: 输入参数直接, 输出 dataclass, 不暴露 pandas DataFrame
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 让脚本可以直接 python -m src.data_loader 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)


@dataclass
class SyntheticDataset:
    """合成数据集, 包含构建 SID 流水线需要的全部数据。

    字段:
        candidate_ids / candidate_embs: 候选物品的 id 和 embedding
        behaviors_per_item: 每个物品对应的行为列表
            (外层 list 按物品顺序, 内层 list 是 [(user_id, code), ...])
        query_ids / query_embs: 可选, 查询的 id 和 embedding (方案 2 需要)
        logs: 可选, 二部图日志 [(q_id, i_id, code), ...] (方案 2 需要)
    """
    candidate_ids: List[str]
    candidate_embs: np.ndarray
    behaviors_per_item: List[List[Tuple[str, int]]]
    query_ids: Optional[List[str]] = None
    query_embs: Optional[np.ndarray] = None
    logs: Optional[List[Tuple[str, str, int]]] = None


def make_synthetic_dataset(n_candidates: int = 1000,
                           n_queries: int = 500,
                           n_users: int = 200,
                           emb_dim: int = 128,
                           behaviors_per_item_mean: int = 10,
                           logs_n: int = 5000,
                           seed: int = 42,
                           ) -> SyntheticDataset:
    """生成可复现的合成数据集, 用于端到端测试。

    数据分布:
      - 物品 embedding: 标准正态 N(0, 1), 模拟 LLM embedding 的零中心分布
      - 行为数: 泊松分布 Poisson(mean), 模拟真实场景中"多数物品行为少, 少数热门"
      - 行为类型: 均匀分布 U{0,1,2,3}, 实际生产中点击率远高于付费, 这里简化为均匀

    Args:
        n_candidates: 候选物品数量。
        n_queries: 查询数量, 设为 0 可以只生成物品 (方案 1 用)。
        n_users: 模拟用户数量, 行为里出现的 user_id 从这个池子采样。
        emb_dim: embedding 维度。
        behaviors_per_item_mean: 平均每物品行为数 (泊松均值)。
        logs_n: 二部图日志条数 (方案 2 用)。
        seed: 随机种子, 保证结果可复现。

    Returns:
        SyntheticDataset 实例, 包含全部字段。
    """
    # 用 numpy 的 Generator 而不是 RandomState, API 更现代且线程安全
    rng = np.random.default_rng(seed)

    # 物品 id 格式: item_00000, item_00001 ... 排序后字典序就是数字序
    candidate_ids = [f"item_{i:05d}" for i in range(n_candidates)]
    candidate_embs = rng.standard_normal((n_candidates, emb_dim)).astype(np.float32)

    # 用户 id
    user_ids = [f"user_{i:04d}" for i in range(n_users)]

    # 为每个物品生成行为: 泊松采样事件数 → 每个事件采样 user + code
    behaviors_per_item: List[List[Tuple[str, int]]] = []
    for _ in range(n_candidates):
        # max(1, ...) 保证每个物品至少有一条行为, 避免完全空物品
        n_events = max(1, int(rng.poisson(behaviors_per_item_mean)))
        beh = []
        for _ in range(n_events):
            u = user_ids[rng.integers(0, n_users)]      # 随机选一个用户
            code = int(rng.integers(0, 4))             # 行为码 0/1/2/3
            beh.append((u, code))
        behaviors_per_item.append(beh)

    # 查询和日志 (方案 2 用), 不需要时跳过
    if n_queries > 0:
        query_ids = [f"q_{i:04d}" for i in range(n_queries)]
        query_embs = rng.standard_normal((n_queries, emb_dim)).astype(np.float32)

        # 二部图日志: 每条随机选 (q, i, code)
        logs: List[Tuple[str, str, int]] = []
        for _ in range(logs_n):
            q = query_ids[rng.integers(0, n_queries)]
            i = candidate_ids[rng.integers(0, n_candidates)]
            code = int(rng.integers(0, 4))
            logs.append((q, i, code))
    else:
        query_ids = None
        query_embs = None
        logs = None

    return SyntheticDataset(
        candidate_ids=candidate_ids,
        candidate_embs=candidate_embs,
        behaviors_per_item=behaviors_per_item,
        query_ids=query_ids,
        query_embs=query_embs,
        logs=logs,
    )


def save_synthetic_dataset(ds: SyntheticDataset, output_dir: str) -> None:
    """把合成数据集序列化到磁盘 (npy + json)。

    为什么用 npy + json 而不是 pickle?
        - npy 是 numpy 原生格式, 跨版本兼容好
        - json 人类可读, 方便排查问题
        - 不依赖 pickle 的安全性

    Args:
        ds: SyntheticDataset 实例。
        output_dir: 输出目录, 不存在会自动创建。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)  # mkdir -p

    # 物品 embedding + id
    np.save(out / "candidate_embs.npy", ds.candidate_embs)
    with open(out / "candidate_ids.json", "w") as f:
        json.dump(ds.candidate_ids, f)

    # 行为日志 (嵌套 list, json 可以直接序列化 tuple 会变 list 所以 OK)
    with open(out / "behaviors_per_item.json", "w") as f:
        json.dump(ds.behaviors_per_item, f)

    # 可选: 查询部分
    if ds.query_embs is not None:
        np.save(out / "query_embs.npy", ds.query_embs)
    if ds.query_ids is not None:
        with open(out / "query_ids.json", "w") as f:
            json.dump(ds.query_ids, f)
    if ds.logs is not None:
        # logs 是 tuple 列表, json.dump 会转成 list, 加载时需转回 tuple
        with open(out / "logs.json", "w") as f:
            json.dump([list(t) for t in ds.logs], f)


def load_synthetic_dataset(input_dir: str) -> SyntheticDataset:
    """读取 save_synthetic_dataset 写出的数据集。

    兼容性: 即使 query_embs / query_ids / logs 不存在, 也能正常加载
    (返回的 SyntheticDataset 对应字段为 None)。

    Args:
        input_dir: 数据集目录, 应包含 candidate_embs.npy 等文件。

    Returns:
        SyntheticDataset 实例。
    """
    p = Path(input_dir)

    # 必读文件
    candidate_embs = np.load(p / "candidate_embs.npy")
    candidate_ids = json.load(open(p / "candidate_ids.json"))
    # 行为: load 时是嵌套 list, 这里做一下格式规整
    raw_behaviors = json.load(open(p / "behaviors_per_item.json"))
    behaviors_per_item = _group_by_item(raw_behaviors, len(candidate_ids))

    # 可选文件: 用 .exists() 探测, 不存在则保持 None
    query_embs = np.load(p / "query_embs.npy") if (p / "query_embs.npy").exists() else None
    query_ids = json.load(open(p / "query_ids.json")) if (p / "query_ids.json").exists() else None
    logs = json.load(open(p / "logs.json")) if (p / "logs.json").exists() else None
    # 把 json 加载的 list 转回 tuple (logs 是 [(q, i, code)] 三元组)
    if logs is not None:
        logs = [(q, i, int(c)) for q, i, c in logs]

    return SyntheticDataset(
        candidate_ids=candidate_ids,
        candidate_embs=candidate_embs,
        behaviors_per_item=behaviors_per_item,
        query_ids=query_ids,
        query_embs=query_embs,
        logs=logs,
    )


def _group_by_item(flat_behaviors: List[Tuple[str, int]],
                   n_items: int) -> List[List[Tuple[str, int]]]:
    """占位函数: 当前 save/load 格式直接是按物品分组的, 不需要重新分组。

    Ponytail: 当前实现是 identity; 留着接口是为了将来支持"扁平行为"格式时
    不用改 load_synthetic_dataset 的签名。

    Args:
        flat_behaviors: 已经是 List[List[(user, code)]] 格式的数据。
        n_items: 物品数量, 当前未用, 保留作接口稳定性。

    Returns:
        原样返回。
    """
    return flat_behaviors


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = make_synthetic_dataset(
        n_candidates=20, n_queries=10,
        behaviors_per_item_mean=5, logs_n=50,
    )
    print(f"candidates: {len(ds.candidate_ids)}, embs: {ds.candidate_embs.shape}")
    print(f"queries: {len(ds.query_ids) if ds.query_ids else 0}")
    print(f"logs: {len(ds.logs) if ds.logs else 0}")
    print(f"first item behaviors (5): {ds.behaviors_per_item[0][:5]}")
    print("OK")