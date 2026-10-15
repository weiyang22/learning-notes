"""
RQ-Kmeans: 残差量化 + Mini-batch K-means (TRM arXiv 2601.22694 §3.2)

与 RQ-VAE 的区别:
  - 无 encoder / decoder, 直接在原始 embedding 空间做量化
  - 训练用 mini-batch K-means (Sculley 2010), 不是 Adam + EMA
  - 优势: 训练快、不需要 GPU、对分布漂移鲁棒

参考:
  - TRM (arXiv 2601.22694) §3.2 使用 RQ-Kmeans
  - RecTokens (EdoardoBotta) 提供参考 mini-batch 实现

Public API:
    - KMeansQuantizer: 单层 mini-batch K-means (K-means++ init)
    - RQKMeans: L 层残差量化 (KMeansQuantizer 栈)
    - RQKMeansTokenizer: 高层封装 (fit / encode_to_sid / save / load)
    - fit_rq_kmeans / encode_with_rq_kmeans: 便捷函数

Ponytail 笔记:
  - 纯 PyTorch, 无 FAISS 依赖 (FAISS 加速见 src.faiss_sid)
  - Mini-batch K-means 用 online running average, 不是全量 Lloyd's
  - 单文件不搞配置层
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# 让脚本可以直接 python -m src.rq_kmeans 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)


class KMeansQuantizer:
    """单层 mini-batch K-means 量化器。

    不是 nn.Module: codebook 不可训练, 通过 streaming update 拟合。
    避免了 VQ-VAE 的 EMA 复杂度, 更易调参。

    算法:
      1. K-means++ seeding on the first batch seen
      2. Mini-batch running average update:
         n_k += |batch_k|;  c_k = (n_k_old * c_k + sum(batch_k)) / n_k
      3. Empty clusters 保留初始 centroid (安全, 不重新 seed)
    """

    def __init__(self, codebook_size: int, dim: int, *, seed: int = 42):
        self.codebook_size = codebook_size
        self.dim = dim
        self.seed = seed
        # 用 zeros 而不是 random: 让第一次 fit_step 触发 K-means++ init
        self._embeddings = torch.zeros(codebook_size, dim)
        self._counts: Optional[torch.Tensor] = None
        self._fitted = False

    @property
    def embeddings(self) -> torch.Tensor:
        """对外暴露 codebook, 与 RQ-VAE 接口对齐。"""
        return self._embeddings

    def to(self, device) -> "KMeansQuantizer":
        """把 codebook 搬到指定 device (cuda/cpu)。"""
        self._embeddings = self._embeddings.to(device)
        if self._counts is not None:
            self._counts = self._counts.to(device)
        return self

    def fit_step(self, batch: torch.Tensor) -> "KMeansQuantizer":
        """用单个 batch 做一次 mini-batch K-means 更新 (Sculley 2010 算法)。

        Args:
            batch: (B, D) 待拟合的 embedding batch。

        Returns:
            self, 支持链式调用。

        Raises:
            ValueError: 空 batch。
        """
        batch = batch.float().contiguous()  # 强制 contiguous, 否则 .pow 等会慢
        k = self.codebook_size
        device = batch.device
        n = len(batch)
        if n == 0:
            raise ValueError("fit_step received an empty batch")

        if self._counts is None:
            # 第一次调用: 用 K-means++ 算法在当前 batch 上初始化质心
            actual_k = min(k, n)
            # generator 强制在 CPU, 因为 K-means++ 的 multinomial 抽样是 CPU 实现
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            centroids = self._kmeans_pp_init(batch.cpu(), actual_k, generator)
            # 若 batch 比 k 小 (极端情况, 比如早期 demo), 重复采样剩余质心 (防止空 code)
            if actual_k < k:
                extra_idx = torch.randint(0, n, (k - actual_k,), generator=generator)
                centroids = torch.cat([centroids, batch.cpu()[extra_idx]], dim=0)
            self._embeddings = centroids.to(device).float()
            self._counts = torch.zeros(k, dtype=torch.long, device=device)

        # mini-batch 在线更新: 滑动平均形式
        # 公式: c_k <- (n_k_old * c_k + sum_batch_k) / (n_k_old + batch_count_k)
        # 等价于增量 K-means 的移动平均更新 (Lloyd 算法的小批量近似)
        self._embeddings, self._counts = _minibatch_update(
            batch, self._embeddings, self._counts
        )
        self._fitted = True
        return self

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> "QuantizeResult":
        """把 (B, D) 输入量化到最近邻 codeword。

        Returns:
            QuantizeResult 含 codes (B,), quantized (B, D), residuals (B, D)。

        Raises:
            RuntimeError: 未 fit 就调用。
        """
        if not self._fitted:
            raise RuntimeError("KMeansQuantizer must be fit before quantize()")
        dist = self._distance(x, self._embeddings)
        codes = dist.argmin(dim=-1)
        quantized = self._embeddings[codes]  # gather codeword
        residuals = x - quantized  # 残差量化
        return QuantizeResult(codes=codes, quantized=quantized, residuals=residuals)

    @torch.no_grad()
    def restart_dead(self, batch: torch.Tensor,
                     generator: Optional[torch.Generator] = None) -> int:
        """死 code 重启: 把从未被分配的 centroid 重新 seed 到欠覆盖区域。

        没有这一步, 残差量化的深层 codebook 会坍缩——
        少数几个 centroid 赢下所有点, 其余 centroid 永远 count==0,
        深层利用率跌到个位数% (实测 [100%, 9%, 9%, 9%, 9%])。
        这是 mini-batch K-means 的经典失效。

        重启策略 (优于纯随机): 最远点采样——
        把死 code 放到当前离所有活 centroid 最远的样本点上,
        直接去覆盖没人管的区域, 保证它下一 pass 至少能赢下自己 (距离 0)。
        等价 docs 00 §4.3 / 01 §4.2 所述"死 code 重启"。

        Returns:
            本次重启的死 code 数量。
        """
        if not self._fitted or self._counts is None or len(batch) == 0:
            return 0
        dead = self._counts == 0
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        batch = batch.float().to(self._embeddings.device)
        # 每个样本到最近活 centroid 的距离; 死 centroid 不参与竞争
        alive = ~dead
        if not alive.any():
            return 0
        dist = self._distance(batch, self._embeddings[alive])
        min_dist = dist.min(dim=-1).values
        # 取离活 centroid 最远的 n_take 个样本作为新 centroid
        n_take = min(n_dead, len(batch))
        far_idx = torch.topk(min_dist, n_take).indices
        dead_idx = torch.nonzero(dead, as_tuple=False).flatten()[:n_take]
        self._embeddings[dead_idx] = batch[far_idx]
        # 保持 count==0: 下一 pass 的 running-average 更新会把它当作全新 centroid,
        # 用真正分到它的样本均值重新初始化, 避免陈旧 seed 拖偏。
        return n_take

    def _distance(self, x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """欧氏距离平方 ||x - c||², 用数学恒等式避免显式广播。

        ||x - c||² = ||x||² - 2 x·c + ||c||²
        其中 ||x||² (B, 1), ||c||² (K,), x·c (B, K), 广播后得 (B, K)
        """
        return (x.pow(2).sum(-1, keepdim=True)
                - 2 * x @ codebook.t()
                + codebook.pow(2).sum(-1).unsqueeze(0))

    @staticmethod
    def _kmeans_pp_init(data: torch.Tensor, k: int,
                        generator: torch.Generator) -> torch.Tensor:
        """K-means++ 智能初始化: 比随机初始化收敛快得多。

        算法:
          1. 第一步随机选一个点
          2. 后续每步: 选与现有质心距离²成正比的点 (越远越容易选)
          这样初始质心分布更均匀, 避免 Lloyd 收敛到局部最优。
        """
        n = len(data)
        first_idx = torch.randint(0, n, (1,), generator=generator)[0]
        centroids = [data[first_idx]]

        for _ in range(k - 1):
            stacked = torch.stack(centroids)
            # min dim=1 → 每个点到最近质心的距离
            dists = torch.cdist(data, stacked).min(dim=1).values ** 2
            total = dists.sum()
            if total == 0:
                # 所有点都与某质心重合 (退化情况), 随机选
                idx = torch.randint(0, n, (1,), generator=generator)[0]
            else:
                # 按距离²概率采样, multinomial 实现 categorical
                idx = torch.multinomial(dists / total, 1, generator=generator)[0]
            centroids.append(data[idx])
        return torch.stack(centroids)


def _minibatch_update(batch: torch.Tensor,
                      centroids: torch.Tensor,
                      counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """单次 mini-batch K-means 更新 (无梯度, 在原函数已装饰 @no_grad)。

    实现细节:
      - 算距离 → argmin 得 code 分配
      - scatter_add_ 累计每个 code 的样本数和样本和
      - 滑动平均更新 active code, 跳过 batch_count==0 的 code (没动它就不动)
    """
    k, d = centroids.shape
    device = batch.device
    dist = (batch.pow(2).sum(-1, keepdim=True)
            - 2 * batch @ centroids.t()
            + centroids.pow(2).sum(-1).unsqueeze(0))
    assignments = dist.argmin(dim=-1)

    # 准备 scatter 目标: 全零 (K,) 和 (K, D)
    batch_counts = torch.zeros(k, dtype=torch.long, device=device)
    batch_sums = torch.zeros(k, d, dtype=torch.float32, device=device)
    # ones 用于 scatter_add_ 把 1 加到 batch_counts[assignment] 里
    ones = torch.ones(len(batch), dtype=torch.long, device=device)

    # 累计每个 code 的样本数和样本和
    batch_counts.scatter_add_(0, assignments, ones)
    batch_sums.scatter_add_(0, assignments.unsqueeze(1).expand(-1, d), batch)

    # 滑动平均: c_new = (n_old * c_old + sum_batch) / (n_old + batch_count)
    new_counts = counts + batch_counts
    new_centroids = centroids.clone()  # 不动未参与的 code
    active = batch_counts > 0
    if active.any():
        n_old = counts[active].float().unsqueeze(1)
        n_new = new_counts[active].float().unsqueeze(1)
        new_centroids[active] = (
            (n_old * centroids[active] + batch_sums[active]) / n_new
        )
    return new_centroids, new_counts


@dataclass
class QuantizeResult:
    """单层 K-means 量化的输出。

    字段:
        codes: (B,) 每个样本被分配的 code id
        quantized: (B, D) 量化向量 (= codeword[code])
        residuals: (B, D) 残差 (= x - quantized), 传给下一层
    """
    codes: torch.Tensor
    quantized: torch.Tensor
    residuals: torch.Tensor


class RQKMeans:
    """L 层残差量化, 每层一个 KMeansQuantizer。

    工作流程:
      residual = x
      for level in 0..L-1:
        out = level.quantize(residual)
        codes[level] = out.codes
        residual = out.residuals  # = residual - codeword[codes[level]]
      return stack(codes)  # (B, L)
    """

    def __init__(self, num_levels: int, codebook_size: int, dim: int, *,
                 seed: int = 42):
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        # 每层用不同 seed, 保证初始质心不同 (否则多层易坍缩到同一组)
        self.levels: List[KMeansQuantizer] = [
            KMeansQuantizer(codebook_size, dim, seed=seed + i)
            for i in range(num_levels)
        ]

    def to(self, device) -> "RQKMeans":
        """把所有层的 codebook 搬到指定 device。"""
        for q in self.levels:
            q.to(device)
        return self

    def fit_step(self, batch: torch.Tensor) -> "RQKMeans":
        """对一个 batch 做 L 层残差量化训练。

        关键点: 残差在层间传递, 每层只 fit 当层残差。
        """
        with torch.no_grad():
            residual = batch.float()
            for q in self.levels:
                q.fit_step(residual)
                # 更新残差: 当前量化后的部分会被下一层编码
                residual = q.quantize(residual).residuals
        return self

    @torch.no_grad()
    def restart_dead_codes(self, batch: torch.Tensor,
                           generator: Optional[torch.Generator] = None) -> List[int]:
        """逐层重启死 code: 沿残差链把每层未被使用的 centroid 重新 seed。

        在每个训练 pass 之间调用: 用真实残差覆盖坍缩的深层 codebook。
        返回每层重启的数量, 便于日志观测收敛。
        """
        residual = batch.float()
        n_restarted: List[int] = []
        for q in self.levels:
            n_restarted.append(q.restart_dead(residual, generator))
            # 重启后用更新后的 codebook 重算残差
            residual = q.quantize(residual).residuals
        return n_restarted

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> "RQQuantizeResult":
        """L 层残差量化推理。

        Returns:
            RQQuantizeResult:
              codes: (B, L) 整数 SID 矩阵
              quantized: (B, D) 各层 codeword 之和 (重建向量)
              residuals: (B, D) 最终残差 (量化误差)
        """
        all_codes: List[torch.Tensor] = []
        total = torch.zeros_like(x)
        residual = x
        for q in self.levels:
            out = q.quantize(residual)
            all_codes.append(out.codes)
            total = total + out.quantized  # 累加各层 codeword
            residual = out.residuals
        return RQQuantizeResult(
            codes=torch.stack(all_codes, dim=1),
            quantized=total,
            residuals=residual,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode = quantize 的 codes 字段, 接口对齐 RQ-VAE。"""
        return self.quantize(x).codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, D) 重建向量, 即各层 codeword 之和。"""
        out = torch.zeros(codes.size(0), self.dim,
                          device=codes.device, dtype=torch.float32)
        for level_idx, q in enumerate(self.levels):
            out = out + q.embeddings[codes[:, level_idx]]
        return out


@dataclass
class RQQuantizeResult:
    """RQ 量化结果。

    字段:
        codes: (B, L) 整数 SID 矩阵 (主要产物)
        quantized: (B, D) 重建向量 (= Σ codeword[l, codes[:, l]])
        residuals: (B, D) 最终残差 (量化损失)
    """
    codes: torch.Tensor
    quantized: torch.Tensor
    residuals: torch.Tensor


@dataclass
class RQKMeansConfig:
    """高层配置, 与 RQ-VAE 的 RQVAEConfig 接口对齐, 方便切换。"""
    input_dim: int = 768
    num_layers: int = 5
    codebook_size: int = 256
    seed: int = 42
    device: str = "cpu"


class RQKMeansTokenizer:
    """高层 tokenizer, 与 src.rq_vae.RQVAE 接口对齐, 可作为 drop-in 替换。

    Args:
        normalize_residuals: 是否对每层残差做 L2 归一化
            (GRID 默认; 已经归一化的 embedding 用这个能改善 codebook 利用率,
            原始 embedding 用这个反而会损害)。
    """

    def __init__(self, cfg: Optional[RQKMeansConfig] = None,
                 normalize_residuals: bool = False):
        self.cfg = cfg or RQKMeansConfig()
        self.normalize_residuals = normalize_residuals
        # fit 之前为 None
        self.rq: Optional[RQKMeans] = None

    def fit(self, embeddings: torch.Tensor,
            batch_size: int = 256,
            n_passes: int = 3,
            restart_dead_codes: bool = True,
            restart_sample: int = 4096,
            verbose: bool = True) -> "RQKMeansTokenizer":
        """多 pass 训练 RQ-Kmeans tokenizer。

        Args:
            embeddings: (N, input_dim) 训练语料。
            batch_size: mini-batch 大小。
            n_passes: 训练 pass 数 (每 pass 全量过一遍)。
            restart_dead_codes: 是否在每个 pass 间重启死 code。
            restart_sample: 重启采样数 (用多少样本来探测死 code)。
            verbose: 是否打印日志。

        Returns:
            self。
        """
        embs = embeddings.float().to(self.cfg.device)
        self.rq = RQKMeans(
            num_levels=self.cfg.num_layers,
            codebook_size=self.cfg.codebook_size,
            dim=self.cfg.input_dim,
            seed=self.cfg.seed,
        ).to(self.cfg.device)
        # 全局 generator, 用于 K-means 抽样
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)

        # 多 pass: 每 pass 全量过一遍, 中间穿插死 code 重启
        for pass_idx in range(n_passes):
            perm = torch.randperm(embs.size(0), device=self.cfg.device)
            for start in range(0, embs.size(0), batch_size):
                idx = perm[start:start + batch_size]
                batch = embs[idx]
                if self.normalize_residuals:
                    batch = torch.nn.functional.normalize(batch, dim=-1)
                self.rq.fit_step(batch)

            # 每个 pass 后重启死 code, 避免深层 codebook 坍缩 (最后一个 pass 不必)
            n_restarted: List[int] = []
            if restart_dead_codes and pass_idx < n_passes - 1:
                # 采样一批做死 code 探测, 不需要全量
                n = min(restart_sample, embs.size(0))
                sample_idx = torch.randperm(embs.size(0), generator=gen)[:n]
                sample = embs[sample_idx]
                if self.normalize_residuals:
                    sample = torch.nn.functional.normalize(sample, dim=-1)
                n_restarted = self.rq.restart_dead_codes(sample, gen)

            if verbose:
                # 打印每层利用率, 帮助观察坍缩
                used_per_level = self.codebook_usage()
                msg = " | ".join(f"L{l}={u:.1f}%" for l, u in enumerate(used_per_level))
                extra = ""
                if n_restarted and sum(n_restarted) > 0:
                    extra = f"  restarted: {n_restarted}"
                print(f"[rq-kmeans] pass {pass_idx+1}/{n_passes}  usage: {msg}{extra}")
        return self

    @torch.no_grad()
    def encode_to_sid(self, x: torch.Tensor) -> torch.Tensor:
        """(N, input_dim) → (N, num_layers) 整数 SID。"""
        if self.rq is None:
            raise RuntimeError("Call fit() before encode_to_sid()")
        x = x.float().to(self.cfg.device)
        return self.rq.encode(x)

    @torch.no_grad()
    def decode_from_sid(self, sid: torch.Tensor) -> torch.Tensor:
        """(N, num_layers) → (N, input_dim) 重建向量。"""
        if self.rq is None:
            raise RuntimeError("Call fit() before decode_from_sid()")
        return self.rq.decode(sid.to(self.cfg.device))

    def codebook_usage(self) -> List[float]:
        """返回每层码本利用率 (0~100%)。

        用于训练日志观察坍缩; 健康情况应该所有层 > 70%。
        """
        if self.rq is None:
            return [0.0] * self.cfg.num_layers
        out = []
        for q in self.rq.levels:
            if q._counts is None:
                out.append(0.0)
            else:
                # 已使用 code 数 / 总数
                used = int((q._counts > 0).sum().item())
                out.append(100.0 * used / q.codebook_size)
        return out

    def save(self, path: str) -> None:
        """保存 tokenizer 状态 (config + 每层 codebook/counts) 到 torch 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.cfg.__dict__, "rq_state": self._state_dict()}, path)

    def load(self, path: str) -> "RQKMeansTokenizer":
        """从 save 文件恢复 tokenizer。"""
        ckpt = torch.load(path, map_location="cpu")
        self.cfg = RQKMeansConfig(**ckpt["config"])
        self.rq = RQKMeans(
            num_levels=self.cfg.num_layers,
            codebook_size=self.cfg.codebook_size,
            dim=self.cfg.input_dim,
            seed=self.cfg.seed,
        ).to(self.cfg.device)
        self._load_state_dict(ckpt["rq_state"])
        return self

    def _state_dict(self) -> dict:
        return {
            f"level_{i}": {
                "embeddings": q.embeddings.cpu(),
                "counts": (q._counts.cpu() if q._counts is not None else None),
                "fitted": q._fitted,
            }
            for i, q in enumerate(self.rq.levels)
        }

    def _load_state_dict(self, state: dict) -> None:
        for i, q in enumerate(self.rq.levels):
            d = state[f"level_{i}"]
            q._embeddings = d["embeddings"].to(self.rq.levels[0].embeddings.device)
            q._counts = d["counts"]
            q._fitted = d["fitted"]


def fit_rq_kmeans(embeddings: torch.Tensor,
                  num_layers: int = 5,
                  codebook_size: int = 256,
                  n_passes: int = 3,
                  batch_size: int = 256,
                  seed: int = 42,
                  device: str = "cpu",
                  verbose: bool = True) -> RQKMeansTokenizer:
    """便捷函数: 训练 RQ-Kmeans tokenizer 并返回。"""
    cfg = RQKMeansConfig(
        input_dim=embeddings.size(1),
        num_layers=num_layers,
        codebook_size=codebook_size,
        seed=seed,
        device=device,
    )
    tok = RQKMeansTokenizer(cfg)
    tok.fit(embeddings, batch_size=batch_size, n_passes=n_passes, verbose=verbose)
    return tok


def encode_with_rq_kmeans(tokenizer: RQKMeansTokenizer,
                          embeddings: torch.Tensor,
                          batch_size: int = 1024) -> torch.Tensor:
    """便捷函数: 批量编码 embedding → SID, 内部切批处理。"""
    out = []
    for start in range(0, embeddings.size(0), batch_size):
        batch = embeddings[start:start + batch_size]
        out.append(tokenizer.encode_to_sid(batch))
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = RQKMeansConfig(input_dim=64, num_layers=5, codebook_size=256)
    embs = torch.randn(500, 64)
    tok = fit_rq_kmeans(embs, num_layers=5, codebook_size=256,
                         n_passes=3, batch_size=128, verbose=True)
    sid = encode_with_rq_kmeans(tok, embs)
    # 断言形状和值域
    assert sid.shape == (500, 5), f"expected (500, 5), got {sid.shape}"
    assert (sid >= 0).all() and (sid < cfg.codebook_size).all()
    usage = tok.codebook_usage()
    print(f"OK: SID shape={tuple(sid.shape)}, per-level usage={usage}")
    print(f"Unique SID tuples: {len(set(map(tuple, sid.tolist())))}/{len(sid)}")