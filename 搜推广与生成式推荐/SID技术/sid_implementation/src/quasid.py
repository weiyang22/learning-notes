"""
QuaSID 风格增强模块 (arxiv 2604.11440 + 关联论文)

提供三个独立的小工具, 可与 RQ-VAE / RQ-Kmeans 即插即用:

1. QuaSIDRating: 用 dot-product 替代 STE, 让梯度通过 codebook 流动
2. NormVarRegularizer: 防止 codebook 坍缩 (norm 均衡损失)
3. IntRRRedistributor: 层级可变长度 SID (IntRR arxiv 2602.20704)
4. CapsIDRouter: 软路由决定每层是否使用 (CapsID arxiv 2605.05096)

Ponytail 设计: 三个独立小工具, 不搞完整 QuaSID 复现;
                每个工具都是 nn.Module 或 dataclass, 与 RQ-VAE / RQ-Kmeans 解耦。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 让脚本可以直接 python -m src.quasid 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)


class QuaSIDRating(nn.Module):
    """QuaSID 评分机制: 用 dot product 替代 STE, 让梯度流向 codebook。

    传统 VQ-VAE 用 STE (Straight-Through Estimator) 让梯度绕过 codebook:
      quantized = codebook[argmin(distance)], 但 backward 时直接把梯度复制给 encoder
    问题: codebook 永远只能靠 EMA 更新, 训练初期不稳定。

    QuaSID (Eq. 5-6) 改用 soft rating:
      L_rating = -log softmax(sim(r, c) / τ)[c_target]
      通过 dot product 让梯度自然流到 codebook, 训练更稳。

    Args:
        temperature: softmax 温度, 默认 0.1。越小越尖锐 (接近 argmin)。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, residual: torch.Tensor, codebook: torch.Tensor,
                codes: torch.Tensor) -> torch.Tensor:
        """residual: (B, D), codebook: (K, D), codes: (B,)。"""
        K = codebook.size(0)
        # 每个 residual 与所有 codeword 的相似度, 温度缩放
        sim = (residual @ codebook.t()) / self.temperature  # (B, K)
        # target = 选中的 code (one-hot)
        target = F.one_hot(codes, K).float()
        # log_softmax 后 gather 选中的那一位, 取负均值就是 cross-entropy
        return -torch.log_softmax(sim, dim=-1).gather(1, codes.unsqueeze(1)).mean()


class NormVarRegularizer(nn.Module):
    """码本 norm 方差正则化: 防止大部分 codeword 范数偏小, 少数偏大。

    为什么需要?
        VQ-VAE 训练中常见 codebook collapse:
        大多数 codeword 范数很小 (被遗弃), 少数很大 (赢得所有样本)。
        惩罚 norm 的方差可以让 codeword 分布更均衡, 提升利用率。

    使用: reg(codebook) 加到总 loss 上, weight 控制强度。
    """

    def forward(self, codebook: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        # 每个 codeword 的 L2 范数, 然后算方差
        norms = codebook.norm(dim=-1)
        return weight * norms.var()


class IntRRRedistributor:
    """IntRR (arxiv 2602.20704): 层级可变长度的 SID。

    核心思想: 大多数物品只需要 1-2 个 token (短 SID), 少数复杂物品需要全部 5 层 (长 SID)。
    用 item importance (点击率 / 转化率 / 难度) 决定每个物品的 SID 长度。

    优势:
      - 短 SID → 索引空间小, 检索快
      - 长 SID → 复杂物品区分度高
      - 平均长度降低 → 整个倒排索引更紧凑

    Ponytail 简化: 这里接受预计算的 importance 分数, 不做协同信号挖掘。
    """

    def __init__(self, num_layers: int, codebook_size: int,
                 short_threshold: float = 0.5):
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.short_threshold = short_threshold

    def assign_lengths(self, item_importance: np.ndarray) -> np.ndarray:
        """根据重要性给每个物品分配 SID 长度。

        Args:
            item_importance: (N,) 重要性分数, 范围 [0, 1]。

        Returns:
            (N,) int 长度数组, 值域 [1, num_layers]。

        规则:
          - importance > short_threshold → 完整 num_layers
          - 否则 → max(1, num_layers * (1 - importance))
        """
        lengths = np.ones(len(item_importance), dtype=np.int64)
        for i, imp in enumerate(item_importance):
            if imp > self.short_threshold:
                lengths[i] = self.num_layers
            else:
                # 重要性越低, 长度越短; 至少保 1 个 token
                lengths[i] = max(1, int(self.num_layers * (1 - imp)))
        return lengths

    def pack_sids(self, base_sids: np.ndarray, lengths: np.ndarray,
                  pad_value: int = 0) -> np.ndarray:
        """把变长 SID 打包成 (N, num_layers) 矩阵, 未用层用 pad_value 填充。

        Args:
            base_sids: (N, num_layers) 原始完整 SID。
            lengths: (N,) 实际使用的长度。
            pad_value: 填充值, 默认 0 (约定词表 0 不分配真 token)。

        Returns:
            (N, num_layers) 矩阵, 前 lengths[i] 个是真实 code, 后面的 pad。
        """
        N, L = base_sids.shape
        out = np.full_like(base_sids, pad_value)
        for i in range(N):
            # 把前 lengths[i] 个真实 code 复制过去, 后面保持 pad
            out[i, :lengths[i]] = base_sids[i, :lengths[i]]
        return out


class CapsIDRouter:
    """CapsID (arxiv 2605.05096): 软路由决定每层是否使用。

    核心思想: 不是所有物品都需要 5 层 RQ, 根据"每层置信度"动态决定层数。
    置信度 = top-1 与 top-2 codeword 距离之差:
      - 差距大 → 已经很确定, 提前结束 (短 SID)
      - 差距小 → 不确定, 继续下一层 (长 SID)

    效果 (CapsID Table 2): Amazon Reviews 数据集上, 碰撞率从 73.2% 降到 22.1%。

    Args:
        confidence_threshold: 提前停止的 margin 阈值。
        min_length / max_length: SID 长度的上下界。
    """

    def __init__(self, confidence_threshold: float = 0.3,
                 min_length: int = 2, max_length: int = 5):
        self.confidence_threshold = confidence_threshold
        self.min_length = min_length
        self.max_length = max_length

    def decide_lengths(self, distances: np.ndarray) -> np.ndarray:
        """根据每层距离决定每个物品的 SID 长度。

        Args:
            distances: (N, K) 每个物品到所有 codeword 的距离。

        Returns:
            (N,) 长度数组, 值域 [min_length, max_length]。

        逻辑: 对每个物品, 如果 top-2 距离差距 > threshold, 就少用一层。
        """
        N = distances.shape[0]
        # 默认全部用最长
        lengths = np.full(N, self.max_length, dtype=np.int64)
        sorted_dist = np.sort(distances, axis=1)
        # K=1 时没有 top-2, 直接返回默认
        if sorted_dist.shape[1] < 2:
            return lengths
        # margin = (top_2 - top_1) / top_2, 归一化让不同尺度可比
        margin = (sorted_dist[:, 1] - sorted_dist[:, 0]) / (
            sorted_dist[:, 1] + 1e-9)
        for i in range(N):
            # margin 大 = 很确定, 把长度 -1 (至少保留 min_length)
            if margin[i] > self.confidence_threshold and lengths[i] > self.min_length:
                lengths[i] = max(self.min_length, lengths[i] - 1)
        return lengths

    def pack_with_lengths(self, sids: np.ndarray, lengths: np.ndarray,
                          pad_value: int = 0) -> np.ndarray:
        """把变长 SID 打包成定长矩阵, 未用层 pad。"""
        N, L = sids.shape
        out = np.full_like(sids, pad_value)
        for i in range(N):
            out[i, :lengths[i]] = sids[i, :lengths[i]]
        return out


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, K = 32, 64, 256
    residual = torch.randn(B, D)
    codebook = torch.randn(K, D)
    codes = torch.randint(0, K, (B,))

    # 1) QuaSIDRating
    rating = QuaSIDRating()
    print(f"QuaSID rating loss: {rating(residual, codebook, codes).item():.4f}")

    # 2) NormVarRegularizer: 对比"小码本"和"正常码本"的 norm 方差
    reg = NormVarRegularizer()
    cb = torch.randn(K, D) * 0.1  # 小方差码本
    print(f"NormVar loss (small codebook): {reg(cb).item():.4f}")
    cb_uniform = torch.randn(K, D)  # 正常码本
    print(f"NormVar loss (uniform codebook): {reg(cb_uniform).item():.4f}")

    # 3) IntRRRedistributor
    import numpy as np
    redist = IntRRRedistributor(num_layers=5, codebook_size=256)
    importances = np.array([0.1, 0.5, 0.9, 0.3, 0.8])
    lengths = redist.assign_lengths(importances)
    print(f"Item lengths from importances {importances.tolist()}: {lengths.tolist()}")

    base_sids = np.random.randint(0, 256, size=(5, 5))
    packed = redist.pack_sids(base_sids, lengths)
    print(f"Packed SIDs shape: {packed.shape}")
    print("OK")