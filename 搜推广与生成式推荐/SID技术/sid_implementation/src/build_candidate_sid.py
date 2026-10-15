"""
候选 SID 端到端构建流水线 (End-to-end candidate SID construction)

实现 docs/01_candidate_sid_with_behavior.md 中描述的方案 1:
对候选物品 (item) 这一侧构建 SID, 把行为信号融合到内容 embedding 中再量化。

完整流水线:
    1. 加载候选物品的 LLM embedding + 历史行为日志
    2. 用 BehaviorFusion 把行为强度注入到物品 embedding (LayerNorm 后)
    3. 用 RQ-VAE 在融合后的 embedding 上做残差量化训练
    4. 训练完成后, 给定任意物品 embedding (含新物品), 输出 [s1..sL] 的 SID

为什么是 RQ-VAE 而不是 RQ-Kmeans?
    RQ-VAE 有 encoder, 可以把高维 LLM embedding 投影到 latent_dim 的"语义子空间";
    这让 codebook 更紧凑, 重建 MSE 更低, 是 Eugene Yan / semantic-id lib 推荐的方案。
    对应的无 encoder 版本见 src.rq_kmeans。

为什么 fit 和 encode 都做融合?
    训练阶段融合 → SID 编码行为语义;
    推理阶段也融合 → 新物品的 embedding 也能利用行为 (如果有历史数据);
    没有行为的新物品则跳过融合, 只用内容 embedding。

Public API:
    - CandidateSIDBuilder: 端到端构建器 (fit + encode + save/load)
    - build_candidate_sid: 一次性 fit + encode 的便捷函数

典型用法:
    >>> builder = CandidateSIDBuilder(input_dim=768, latent_dim=256)
    >>> builder.fit(candidate_embs, behaviors_list, epochs=30)
    >>> sids = builder.encode(candidate_embs, behaviors_list)  # (N, 5) int64
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# 兼容两种调用方式: 作为包内模块 (src.build_candidate_sid) 或脚本直接运行
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# 兄弟模块
from src.behavior_fusion import fuse_candidate_batch
from src.rq_vae import RQVAE, RQVAEConfig, train_rqvae, encode_to_sid


@dataclass
class CandidateSIDConfig:
    """端到端 SID 构建器的全部超参, 集中管理避免函数签名膨胀。

    字段语义:
        input_dim / hidden_dim / latent_dim: encoder 形状 (LLM embedding 通常是 768/1024/4096)
        num_layers: SID 层数, 默认 5 (论文常用)
        codebook_size: 每层码本大小 K, 默认 256 → 单 SID 空间 256^5 ≈ 1.1e12
        epochs / batch_size / lr: 训练超参
        alpha: 行为融合权重 (透传给 fuse_candidate_batch)
        device: "cpu" 或 "cuda"
        use_user_emb: 是否用 user embedding 做融合 (否则用 ponytail 计数向量回退)
    """
    input_dim: int = 768       # LLM embedding 维度 (Qwen-1.5B 是 1536, BERT-768)
    hidden_dim: int = 512      # encoder 中间层维度
    latent_dim: int = 256      # encoder 输出 (量化) 维度
    num_layers: int = 5        # RQ 层数, 等于 SID 长度
    codebook_size: int = 256   # 每层 codebook 大小 K
    epochs: int = 50           # 训练轮数
    batch_size: int = 256      # mini-batch 大小
    lr: float = 1e-3           # Adam 学习率 (只更新 encoder/decoder, codebook 走 EMA)
    alpha: float = 0.4         # 行为分支权重, 透传给 fuse_candidate_batch
    device: str = "cpu"        # 训练设备
    use_user_emb: bool = False # 是否用 user embedding (否则回退到计数向量)


class CandidateSIDBuilder:
    """端到端 builder: 行为融合 → RQ-VAE 训练 → SID 推断。

    状态:
        cfg: 训练配置 (CandidateSIDConfig)
        model: 训练好的 RQVAE 实例 (fit 之后才有)
    """

    def __init__(self, cfg: Optional[CandidateSIDConfig] = None):
        # cfg 为 None 时使用默认值, 方便快速测试
        self.cfg = cfg or CandidateSIDConfig()
        self.model: Optional[RQVAE] = None  # 训练前为 None, 训练后填充

    def fit(self,
            candidate_embs: np.ndarray,
            behaviors_list: Sequence[Sequence[Tuple[str, int]]],
            user_embs: Optional[Dict[str, np.ndarray]] = None,
            verbose: bool = True) -> "CandidateSIDBuilder":
        """训练 RQ-VAE。

        步骤:
            1. 行为融合: 把 (N, D) 内容 embedding + 行为日志 → (N, D) 融合 embedding
            2. 构造 RQ-VAE 模型 (结构由 cfg 决定)
            3. 调用 train_rqvae 训练指定轮数
            4. 把训练好的模型存到 self.model

        Args:
            candidate_embs: (N, D) 物品内容 embedding, D = cfg.input_dim。
            behaviors_list: 长度 N 的列表, 每个元素是该物品的行为序列 [(u_id, code)]。
            user_embs: 可选 {user_id: (D,) embedding}, 用于行为融合路径 A。
            verbose: 是否打印训练日志。

        Returns:
            self, 支持链式调用 builder.fit(...).encode(...).
        """
        # 第一步: 行为融合
        # 根据 use_user_emb 决定走 user_emb 路径还是 ponytail 计数向量路径
        fused = fuse_candidate_batch(
            candidate_embs, behaviors_list,
            user_embs=user_embs if self.cfg.use_user_emb else None,
            alpha=self.cfg.alpha,
        )

        # 第二步: 构造 RQ-VAE 配置
        # RQVAEConfig 字段名与 CandidateSIDConfig 略有不同, 需要显式映射
        rqvae_cfg = RQVAEConfig(
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            latent_dim=self.cfg.latent_dim,
            num_layers=self.cfg.num_layers,
            codebook_size=self.cfg.codebook_size,
        )

        # 第三步: 实例化模型 (尚未训练, 内部已做数据相关初始化 lazy init)
        self.model = RQVAE(rqvae_cfg)

        # numpy → torch float32, RQ-VAE 期望这个 dtype
        embeddings = torch.from_numpy(fused).float()

        # 第四步: 训练 (内部循环 epoch + batch, EMA 更新 codebook)
        train_rqvae(
            self.model, embeddings,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            device=self.cfg.device,
            verbose=verbose,
        )
        return self  # 支持链式调用

    @torch.no_grad()  # 推理不需要梯度, 节省显存 + 提速
    def encode(self,
               candidate_embs: np.ndarray,
               behaviors_list: Optional[Sequence[Sequence[Tuple[str, int]]]] = None,
               user_embs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """把 (N, D) embedding 编码成 (N, L) 整数 SID。

        必须先调用 fit()。

        Args:
            candidate_embs: (N, D) embedding, 可以是新物品的 embedding。
            behaviors_list: 可选行为日志, 传 None 则跳过融合 (冷启动场景)。
            user_embs: 可选 user embedding, 透传给融合函数。

        Returns:
            (N, L) 整数数组, dtype=int64, 每行是 [s1, s2, ..., sL]。

        Raises:
            RuntimeError: 未训练就调用 (model is None)。
        """
        if self.model is None:
            raise RuntimeError("Call fit() before encode()")

        # 如果提供了行为, 同样做融合 (保持训练 / 推理一致)
        # 注意: 推理时的融合与训练必须一致, 否则 SID 分布会偏移
        if behaviors_list is not None:
            emb = fuse_candidate_batch(
                candidate_embs, behaviors_list,
                user_embs=user_embs if self.cfg.use_user_emb else None,
                alpha=self.cfg.alpha,
            )
        else:
            # 冷启动 / 无行为: 直接用内容 embedding
            emb = candidate_embs

        # numpy → torch, encode_to_sid 内部用 batch 切分
        tensor = torch.from_numpy(np.asarray(emb, dtype=np.float32))
        sid = encode_to_sid(
            self.model, tensor,
            batch_size=self.cfg.batch_size,
            device=self.cfg.device,
        )
        return sid.numpy().astype(np.int64)  # 转回 numpy int64

    def save(self, path: str) -> None:
        """保存模型权重 + 配置到 torch 格式文件。

        用途: 训练完成后持久化, 部署时用 load() 恢复。

        Args:
            path: .pt 文件路径, 父目录不存在会自动创建。

        Raises:
            RuntimeError: 尚未训练 (nothing to save)。
        """
        if self.model is None:
            raise RuntimeError("Nothing to save: model not trained")

        # mkdir -p: 父目录不存在时创建
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": self.model.state_dict(),  # 包含 encoder/decoder/codebook
            "config": self.cfg.__dict__,             # dataclass 转 dict 才能 pickle
        }, path)

    def load(self, path: str) -> "CandidateSIDBuilder":
        """从 save() 保存的文件恢复 builder。

        加载后可以直接调用 encode(), 但不能再 fit() (optimizer 状态没保存)。

        Args:
            path: save() 时使用的路径。

        Returns:
            self, 模型设为 eval 模式。
        """
        # map_location="cpu" 保证 GPU 训练的模型也能在 CPU 上加载
        ckpt = torch.load(path, map_location="cpu")
        # 重建配置 (原 cfg 被覆盖, 这就是为什么 save 时存的是 cfg.__dict__)
        self.cfg = CandidateSIDConfig(**ckpt["config"])

        # 重建 RQ-VAE 结构 (必须和训练时一致)
        rqvae_cfg = RQVAEConfig(
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            latent_dim=self.cfg.latent_dim,
            num_layers=self.cfg.num_layers,
            codebook_size=self.cfg.codebook_size,
        )
        self.model = RQVAE(rqvae_cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()  # 切到推理模式 (关闭 dropout 等)
        return self


def build_candidate_sid(candidate_embs: np.ndarray,
                        behaviors_list: Sequence[Sequence[Tuple[str, int]]],
                        candidate_ids: Optional[Sequence[str]] = None,
                        cfg: Optional[CandidateSIDConfig] = None,
                        user_embs: Optional[Dict[str, np.ndarray]] = None,
                        save_path: Optional[str] = None,
                        ) -> Dict[str, List[int]]:
    """一键完成 fit + encode, 返回 {candidate_id: [s1, ..., s5]}。

    Ponytail 风格: 不暴露中间步骤, 一个函数搞定。
    如果需要更细粒度控制 (比如分批训练 / 中途评估), 请直接用 CandidateSIDBuilder。

    Args:
        candidate_embs: (N, D) 物品 embedding。
        behaviors_list: 长度 N 的列表, 每条是该物品的行为日志。
        candidate_ids: 可选长度 N 的 id 列表, 用于构造返回字典的 key;
                       不传则自动用 "item_0", "item_1", ... 命名。
        cfg: 可选训练配置, 不传则用默认 CandidateSIDConfig()。
        user_embs: 可选 {user_id: (D,) embedding}。
        save_path: 可选, 训练好的模型保存路径 (.pt)。

    Returns:
        {candidate_id: [s1, ..., sL]} 字典, 每条 SID 是长度为 L 的整数列表。
    """
    builder = CandidateSIDBuilder(cfg)
    # fit: 训练
    builder.fit(candidate_embs, behaviors_list, user_embs=user_embs)
    # encode: 训练完立即推断, 保证训练 / 推理用同一份融合参数
    sids = builder.encode(candidate_embs, behaviors_list, user_embs=user_embs)

    # 可选持久化
    if save_path is not None:
        builder.save(save_path)

    # 把 (N, L) numpy 数组转成 {id: list} 字典, 方便 JSON 序列化
    if candidate_ids is None:
        # 默认 id: item_0, item_1, ...
        candidate_ids = [f"item_{i}" for i in range(len(sids))]
    return {cid: sid.tolist() for cid, sid in zip(candidate_ids, sids)}


# ---------------------------------------------------------------------------
# 自测: 随机数据走完整流水线
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, D = 300, 64  # 300 个物品, 64 维 embedding (缩小版配置, 跑得快)

    # 随机生成 embedding
    embs = rng.standard_normal((N, D)).astype(np.float32)

    # 为每个物品随机生成 1~20 条行为
    behaviors_list = [
        [(f"u_{rng.integers(0, 50)}", int(rng.integers(0, 4)))
         for _ in range(rng.integers(1, 20))]
        for _ in range(N)
    ]

    # 缩小版配置, 让自测在几秒内跑完
    cfg = CandidateSIDConfig(
        input_dim=D, hidden_dim=64, latent_dim=32,
        num_layers=5, codebook_size=256,
        epochs=10, batch_size=64,
    )
    out = build_candidate_sid(embs, behaviors_list, cfg=cfg)

    print(f"Number of candidates: {len(out)}")
    sample_id = list(out.keys())[0]
    print(f"Sample SID for {sample_id}: {out[sample_id]}")
    # 统计 SID 唯一性: 理想情况碰撞率应该很低 (< 1%)
    unique_count = len(set(map(tuple, out.values())))
    print(f"All SID tuples unique? {unique_count}/{len(out)}")