"""
HaMR 损失: Hamming-guided Margin Repulsion (QuaSID arxiv 2604.11440)

一种即插即用的辅助损失, 用于 RQ-VAE / RQ-Kmeans 训练,
让 codebook 中具有不同 SID 的 codeword 在 embedding 空间被推开,
相同 SID 的 codeword 被拉近。

核心公式 (QuaSID 论文 Eq. 7):
    L_HaMR = -log σ(sim(q, k+) - sim(q, k-))
             + λ · HammingDistance(c+, c-)

    其中:
      - k+ 是与 q 共享 SID 的 codeword (正样本)
      - k- 是与 q SID 不同的 codeword (负样本)
      - sim 是 cosine 相似度
      - HammingDistance 鼓励 SID 序列差异大的样本被推开得更远

效果 (QuaSID Table 3):
    在 Beauty 数据集上 NDCG@10 提升 +15.3%
    在 Toys 数据集上 NDCG@10 提升 +20.1%
    (相对基线 RQ-VAE)

Ponytail 决策:
  - 单文件自包含, 不需要修改 RQ-VAE / RQ-Kmeans 内部代码
  - 作为辅助损失使用: L_total = L_recon + L_vq + hamr_weight * L_HaMR
  - 训练时挂在 RQ-VAE 的 forward 之后, 不改变主训练流程

Public API:
    - HaMRLoss: 损失函数模块 (torch.nn.Module)
    - hamr_loss_rq_vae: 便捷包装, 自动从 model_outputs 抽取 indices
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 让脚本可以直接 python -m src.hamr_loss 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)


class HaMRLoss(nn.Module):
    """Hamming-guided Margin Repulsion 损失。

    Args:
        margin: 正负样本 cosine 相似度之差的目标间隔, 默认 0.2。
            越大 = 把不同 SID 推得越开, 但可能破坏重建。
        hamming_weight: Hamming 排斥项的权重, 默认 1.0。
            控制"SID 差异大 → 空间距离也大"的强度。
        temperature: 对比项的温度系数, 默认 0.07 (CLIP 同款)。
            越小 = 对相似度差异越敏感, 训练越"挑剔"。
    """

    def __init__(self, margin: float = 0.2, hamming_weight: float = 1.0,
                 temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.hamming_weight = hamming_weight
        self.temperature = temperature

    def hamming_distance(self, sids: torch.Tensor) -> torch.Tensor:
        """计算 SID 序列两两之间的归一化 Hamming 距离。

        Hamming 距离 = 序列对应位置不同的元素个数 / 总长度
        这里用归一化值 (除以 L), 让 L=3 和 L=5 的 SID 距离可比。

        Args:
            sids: (B, L) int 张量, 每行是一条 SID。

        Returns:
            (B, B) float 张量, [i, j] 是 sids[i] 和 sids[j] 的归一化 Hamming 距离。
        """
        # 兼容单条输入: (L,) → (1, L)
        if sids.ndim == 1:
            sids = sids.unsqueeze(0)
        B, L = sids.shape
        # 两两比较: (B, 1, L) == (1, B, L) → (B, B, L) bool
        eq = (sids[:, None, :] == sids[None, :, :]).float()
        # 不一致的位置比例 = 1 - 一致的比例
        ham = (1.0 - eq).sum(dim=-1) / L
        return ham

    def forward(self, embeddings: torch.Tensor, sids: torch.Tensor,
                recon_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 HaMR 损失。

        Args:
            embeddings: (B, D) 物品 embedding。
            sids: (B, L) 整数 SID 序列。
            recon_embeddings: 可选 (B, D) 重建 embedding (用于对比原始);
                              若 None, 用 embeddings 作为正样本基线。

        Returns:
            标量 loss tensor。
        """
        # 兼容单条: (D,) → (1, D)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if sids.ndim == 1:
            sids = sids.unsqueeze(0)
        B = embeddings.size(0)
        # batch < 2 没法做对比学习, 直接返回 0 (不算梯度)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # L2 归一化后做点积, 等价于 cosine 相似度
        embs_norm = F.normalize(embeddings, dim=-1)
        # 温度缩放: 让 softmax 更"尖锐", 对正负样本的差异更敏感
        sim = embs_norm @ embs_norm.T / self.temperature              # (B, B)

        # 构造正负样本 mask
        eye = torch.eye(B, device=sim.device, dtype=torch.bool)
        # 正样本 mask: SID 完全相同 (不是自己)
        same_sid = (sids[:, None, :] == sids[None, :, :]).all(dim=-1)  # (B, B)
        same_sid = same_sid & ~eye  # 排除对角线
        # 负样本 mask: SID 完全不同 (不是自己)
        neg_mask = (~same_sid & ~eye).float()

        # 边界情况: 当前 batch 没有正样本 (所有 SID 都不同), 跳过
        if same_sid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        pos_mask = same_sid.float()

        # 加权平均正 / 负样本的相似度分数
        # 1e-9 防 0 除 (极端情况下某 batch 全是不同 SID)
        pos_score = (sim * pos_mask).sum() / (pos_mask.sum() + 1e-9)
        neg_score = (sim * neg_mask).sum() / (neg_mask.sum() + 1e-9)

        # 对比项: 让 pos_score > neg_score + margin
        # log σ(x) 在 x 越大时越接近 0 (loss 小), 越小越负 (loss 大)
        # margin 让正负样本的相似度差距至少 margin
        contrastive = -torch.log(torch.sigmoid(pos_score - neg_score + self.margin))

        # Hamming 排斥项: 负样本对的平均 Hamming 距离, 越大越好
        ham = self.hamming_distance(sids)
        ham_term = (ham * neg_mask).sum() / (neg_mask.sum() + 1e-9)

        return contrastive + self.hamming_weight * ham_term


def hamr_loss_rq_vae(model_outputs: dict, sids: torch.Tensor,
                     embeddings: torch.Tensor,
                     hamr_weight: float = 0.1,
                     **hamr_kwargs) -> torch.Tensor:
    """便捷包装: 把 HaMR 损失挂到 RQ-VAE 训练中。

    使用示例 (在 train_rqvae 内):
        out, indices, vq_loss, info = model(batch)
        total_loss = recon + vq_loss + hamr_weight * HaMRLoss(...)(batch, indices)

    Args:
        model_outputs: RQ-VAE forward 返回的 dict, 需含 'recon' 和 'indices' 键。
        sids: (B, L) int 张量 (这里其实用不到, 接口对齐保留)。
        embeddings: (B, D) 原始输入。
        hamr_weight: 损失权重, 默认 0.1 (经验值, 过大破坏重建)。
        **hamr_kwargs: 透传给 HaMRLoss 的参数 (margin / hamming_weight / temperature)。

    Returns:
        标量 loss tensor。
    """
    hamr = HaMRLoss(**hamr_kwargs)
    indices = model_outputs.get("indices")
    # 没拿到 indices 就跳过 (避免破坏训练)
    if indices is None:
        return torch.tensor(0.0, device=embeddings.device)

    # 兼容 indices 是 (B,) 还是 (B, L) 的情况
    sids_t = torch.as_tensor(indices, dtype=torch.long, device=embeddings.device)
    if sids_t.ndim == 1:
        sids_t = sids_t.unsqueeze(0)
    # batch size 不匹配说明 RQ-VAE 内部出了诡异问题, 跳过
    if sids_t.size(0) != embeddings.size(0):
        return torch.tensor(0.0, device=embeddings.device)
    return hamr_weight * hamr(model_outputs["recon"], sids_t, embeddings)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, L = 64, 32, 3  # batch=64, 32 维, SID 长度 3
    K = 8  # 假设码本大小 8

    # 构造一个"聚类分布"的 embedding: 每个 embedding 是某个中心 + 噪声
    centers = torch.randn(K, D)
    cluster_assign = torch.randint(0, K, (B,))
    embs = centers[cluster_assign] + 0.1 * torch.randn(B, D)
    # SID 第 0 位 = cluster id (理想), 后 2 位随机
    sids = torch.zeros((B, L), dtype=torch.long)
    sids[:, 0] = cluster_assign
    sids[:, 1:] = torch.randint(0, K, (B, L - 1))

    hamr = HaMRLoss()

    # 场景 1: 簇结构明显但 SID 后位随机 (不完美对齐)
    loss1 = hamr(embs, sids).item()
    print(f"HaMR loss (clustered emb, sids[:, 0]=cluster, others random): {loss1:.4f}")

    # 场景 2: SID 全部同簇 (完美对齐, loss 应该最低)
    sids_same = torch.zeros((B, L), dtype=torch.long)
    sids_same[:, 0] = cluster_assign
    sids_same[:, 1:] = cluster_assign.unsqueeze(-1)
    loss2 = hamr(embs, sids_same).item()
    print(f"HaMR loss (all-same-in-cluster sids, emb clustered): {loss2:.4f}")

    # 场景 3: SID 完全随机 (没有结构, loss 应该最高)
    sids_random = torch.randint(0, 256, (B, L))
    loss3 = hamr(embs, sids_random).item()
    print(f"HaMR loss (random sids, no real cluster structure in SID): {loss3:.4f}")

    print(f"\nMargin: {hamr.margin}, hamming_weight: {hamr.hamming_weight}, "
          f"temperature: {hamr.temperature}")
    print("OK")