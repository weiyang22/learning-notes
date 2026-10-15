"""
RQ-VAE: 残差量化 VAE (用于 SID 构造)

基于 TRM (arXiv 2601.22694) §3.2 的 Hybrid Tokenization, 采用 VAE 风格
(encoder + decoder + 残差量化), 是用户请求的 RQ-VAE 变体。

与 RQ-Kmeans 的对比:
  - RQ-VAE: 有 encoder, 把高维 LLM embedding 投影到 latent_dim 子空间后量化
  - RQ-Kmeans: 直接在原始空间做 mini-batch K-means
  - 优势: codebook 更紧凑, 重建 MSE 更低, 是 Eugene Yan 推荐方案
  - 代价: 训练更慢 (需要 GPU + 反向传播)

Public API:
    - RQVAE: 端到端 Encoder + ResidualQuantizer + Decoder
    - UnifiedRQVAE: 两路 (query/item) + shared/individual 层
    - ResidualQuantizer: 残差量化栈 (内部用)
    - train_rqvae: 便捷训练循环 (Adam + EMA codebook)
    - encode_to_sid: 提取 (N, num_layers) 整数 SID
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RQVAEConfig:
    """RQ-VAE 配置。

    关键字段:
        use_ema: True 用 EMA 更新 codebook (推荐, 训练稳定)
                 False 用 Adam 优化 codebook (容易坍缩)
        ema_decay: EMA 衰减率, 默认 0.95 (保留 95% 历史)
        commitment_cost: commitment loss 系数, 默认 0.25
        restart_interval: 多久重启一次死 code (按 global step 计)
    """
    input_dim: int = 768          # LLM embedding 维度
    hidden_dim: int = 512         # encoder/decoder 中间层
    latent_dim: int = 256         # 量化空间的维度 (codebook 行向量长度)
    num_layers: int = 5           # RQ 层数 (= SID 长度)
    codebook_size: int = 256      # 每层码本大小 K
    commitment_cost: float = 0.25
    ema_decay: float = 0.95
    eps: float = 1e-5             # EMA 数值稳定项
    restart_interval: int = 200   # 死 code 重启间隔 (step)
    use_ema: bool = True          # True: EMA 更新 codebook; False: Adam


class ResidualQuantizer(nn.Module):
    """残差量化: 把 z (latent_dim) 编码成 [s1..sL] 整数, 解码回重建向量。

    encode: z → [s1, s2, ..., sL] 整数序列
    decode: [s1, ..., sL] → 量化向量 (各层 codeword 之和)
    """

    def __init__(self, num_layers: int, codebook_size: int, latent_dim: int,
                 ema_decay: float = 0.95, eps: float = 1e-5,
                 commitment_cost: float = 0.25, use_ema: bool = True,
                 restart_interval: int = 200):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        # 防 restart_interval==0 导致每步都重启
        self.restart_interval = max(1, restart_interval)

        # 注册 codebook 作为可学习参数 (非 EMA 模式时 Adam 优化)
        self.register_parameter(
            "codebooks",
            nn.Parameter(torch.randn(num_layers, codebook_size, latent_dim) * 0.05),
        )
        # EMA 状态: cluster size 和向量和 (与 codebooks 同形)
        self.register_buffer("ema_cluster_sizes",
                             torch.zeros(num_layers, codebook_size))
        self.register_buffer("ema_w",
                             torch.randn(num_layers, codebook_size, latent_dim) * 0.05)
        # 死 code 重启追踪: 每层每个 code 上次被用的 step
        self.register_buffer("code_last_used",
                             torch.zeros(num_layers, codebook_size, dtype=torch.long))
        # 全局 step 计数
        self.register_buffer("global_step",
                             torch.zeros(1, dtype=torch.long))
        # 是否做过数据相关初始化
        self.register_buffer("initialized",
                             torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _maybe_data_init(self, z: torch.Tensor):
        """数据相关初始化: 用首个 batch 的真实残差 seed 每层码本。

        纯随机初始化 (×0.05) 会让所有点在第一步都 argmin 到同一个 code,
        陷入单 code 坍缩且 EMA 无法逃逸 (只有那个 code 被更新)。
        从真实数据采样初始码字, 让码本一开始就铺开在数据分布上——
        这与 RQ-Kmeans 的 K-means++ 初始化同理, 是 VQ-VAE 防坍缩的标准做法。
        """
        if not self.use_ema or bool(self.initialized.item()):
            return
        residual = z
        k = self.codebook_size
        for layer in range(self.num_layers):
            n = residual.size(0)
            if n == 0:
                break
            if n >= k:
                # 数据足够多: 随机采 K 个真实残差作为初始 code
                idx = torch.randperm(n, device=z.device)[:k]
            else:
                # 数据不够 K: 有放回抽样
                idx = torch.randint(0, n, (k,), device=z.device)
            init = residual[idx].clone()
            self.codebooks[layer].data.copy_(init)
            self.ema_w[layer].copy_(init)
            # 初始化 cluster size 为 1 (避免除 0)
            self.ema_cluster_sizes[layer].fill_(1.0)
            # 用刚初始化好的码本推进残差, 供下一层初始化
            _, quantized, _, _ = self._quantize_one_layer(residual, layer)
            residual = residual - quantized
        self.initialized.fill_(1)

    def _quantize_one_layer(self, residual: torch.Tensor, layer: int
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单层量化核心: 算距离、找最近邻、构造量化向量、计算两个 loss。

        返回: (indices, quantized, commitment_loss, codebook_loss)
        """
        # 当前层码本 (K, D), 每行是一个 codeword 向量
        cb = self.codebooks[layer]
        # 计算每个 residual 向量与所有 K 个 codeword 的欧氏距离平方
        # 使用数学恒等式: ||r - c||² = ||r||² - 2r·c + ||c||² 避免显式广播
        dist = (residual.pow(2).sum(-1, keepdim=True)
                - 2 * residual @ cb.t()
                + cb.pow(2).sum(-1).unsqueeze(0))
        # 找最近邻 codeword 的索引作为该层量化结果 (硬分配)
        indices = dist.argmin(dim=-1)
        # 构造 one-hot 矩阵以便用矩阵乘法得到量化向量
        onehot = F.one_hot(indices, self.codebook_size).float()
        quantized = onehot @ cb
        # commitment loss: 让 encoder 的输出 z 靠近量化结果 (z 会被向前传)
        commitment_loss = F.mse_loss(residual, quantized.detach())
        # codebook loss: 让 codeword 靠近 encoder 的输出 (codebook 的梯度)
        codebook_loss = F.mse_loss(quantized, residual.detach())
        return indices, quantized, commitment_loss, codebook_loss

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """残差量化推理: 依次在每层找最近邻, 用其作为该层 code 并更新残差。"""
        residual = z.clone()
        all_indices = []
        for layer in range(self.num_layers):
            # 在第 layer 层做最近邻量化
            indices, _, _, _ = self._quantize_one_layer(residual, layer)
            # 减去本层贡献, 更新残差给下一层 (这就是"残差"量化的核心思想)
            residual = residual - self.codebooks[layer][indices]
            all_indices.append(indices)
        # 拼接成 (N, L) 的整数 SID 矩阵
        return torch.stack(all_indices, dim=-1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """SID → 重建向量: 每层查表并求和 (与 encode 的残差减法互逆)。"""
        out = torch.zeros(indices.size(0), self.latent_dim,
                          device=indices.device, dtype=self.codebooks.dtype)
        for layer in range(self.num_layers):
            out = out + self.codebooks[layer][indices[:, layer]]
        return out

    def forward(self, z: torch.Tensor):
        """训练阶段的前向传播: 返回量化向量、索引、VQ 损失、诊断信息。"""
        # 第一次 forward 时做数据相关初始化
        self._maybe_data_init(z.detach())
        residual = z.clone()
        all_indices = []
        all_commit = []
        all_codebook = []
        quantized_sum = torch.zeros_like(z)
        for layer in range(self.num_layers):
            # layer_input 是本层量化的输入 (即上一层的残差)。
            # 必须在做残差减法之前捕获它, 因为 EMA 要用它 (而不是量化后的码字) 来更新码本。
            layer_input = residual
            indices, quantized, commit, codebook = self._quantize_one_layer(layer_input, layer)
            quantized_sum = quantized_sum + quantized
            residual = layer_input - quantized
            all_indices.append(indices)
            all_commit.append(commit)
            all_codebook.append(codebook)
            self._ema_update(layer, indices, layer_input)

        indices = torch.stack(all_indices, dim=-1)
        # STE (straight-through estimator): 量化值在 forward 等于 quantized_sum,
        # 在 backward 时把梯度直接复制给 encoder 输出 z
        quantized_st = z + (quantized_sum - z).detach()
        vq_loss = (sum(all_commit) / self.num_layers) * self.commitment_cost
        if not self.use_ema:
            # 非 EMA 模式: 把 codebook loss 也加到 vq_loss (让 Adam 优化 codebook)
            vq_loss = vq_loss + sum(all_codebook) / self.num_layers

        self.global_step += 1
        # 定期重启长期没被用的死 code
        if self.use_ema and self.global_step.item() % self.restart_interval == 0:
            self._restart_dead_codes(z)

        info = {
            "indices": indices,
            "perplexity": self._codebook_perplexity(indices),
        }
        return quantized_st, indices, vq_loss, info

    def _ema_update(self, layer: int, indices: torch.Tensor,
                    flat_input: torch.Tensor):
        """EMA 更新 codebook。

        关键: 用于更新的必须是本层输入 flat_input (encoder 输出的残差),
        而不是量化后的码字 quantized。若误用 quantized, 则
          ema_w[k] = Σ_{i→k} cb[k] = count_k · cb[k]
          codebook[k] = ema_w[k] / count_k = cb[k]   (原地不动!)
        码本会被冻结在随机初始化, encoder 被迫把所有输入挤进那一小片区域,
        造成整层坍缩到极少数 code (实测 layer0 只剩 1 个 code)。
        """
        if not self.use_ema:
            return
        with torch.no_grad():
            onehot = F.one_hot(indices, self.codebook_size).float()
            # 统计本 batch 中每个 code 被分配的次数
            new_sizes = onehot.sum(dim=0)
            # EMA 更新 cluster size: 保留历史 95% + 本 batch 5%
            self.ema_cluster_sizes[layer].mul_(self.ema_decay).add_(
                new_sizes, alpha=1 - self.ema_decay)
            # EMA 更新 codebook 的"向量和": Σ 分配到 code k 的输入向量
            # (后面再除以 cluster size 得到该簇的均值, 即码字向数据靠拢)
            new_w = onehot.t() @ flat_input
            self.ema_w[layer].mul_(self.ema_decay).add_(
                new_w, alpha=1 - self.ema_decay)
            # 等价公式: c_k = ema_w_k / ema_cluster_size_k
            n = self.ema_cluster_sizes[layer].clamp(min=self.eps)
            self.codebooks[layer].data.copy_(self.ema_w[layer] / n.unsqueeze(-1))
            # 记录本 batch 用到的 code (用于死 code 重启)
            used = (new_sizes > 0).nonzero(as_tuple=False).flatten()
            self.code_last_used[layer, used] = self.global_step.item()

    def _codebook_perplexity(self, indices: torch.Tensor) -> float:
        """计算每层码本实际使用的 perplexity (= exp(entropy)), 越大越均匀。"""
        with torch.no_grad():
            per_layer = []
            for layer in range(self.num_layers):
                ids = indices[:, layer]
                counts = torch.bincount(ids, minlength=self.codebook_size).float()
                p = counts / counts.sum().clamp(min=1)
                ent = -(p[p > 0] * p[p > 0].log()).sum()
                per_layer.append(ent.exp().item())
            return sum(per_layer) / len(per_layer)

    @torch.no_grad()
    def _restart_dead_codes(self, z: torch.Tensor):
        """重启死 code: 把长期没被用的 codeword 重新初始化为 batch 内的样本。"""
        # stale = (当前 step - 上次使用 step) > restart_interval
        stale = (self.global_step.item() - self.code_last_used) > self.restart_interval
        for layer in range(self.num_layers):
            dead_mask = stale[layer]
            if dead_mask.any() and z.size(0) > 0:
                # 随机采样 batch 内的样本作为新 codeword
                rand_idx = torch.randint(0, z.size(0),
                                         (int(dead_mask.sum().item()),),
                                         device=z.device)
                self.codebooks[layer].data[dead_mask] = z[rand_idx]


class RQVAE(nn.Module):
    """Encoder + ResidualQuantizer + Decoder 的端到端 SID 训练模块。"""

    def __init__(self, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg
        # 标准 MLP encoder: input → hidden → latent
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        # 对称的 decoder
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )
        self.quantizer = ResidualQuantizer(
            num_layers=cfg.num_layers,
            codebook_size=cfg.codebook_size,
            latent_dim=cfg.latent_dim,
            ema_decay=cfg.ema_decay,
            eps=cfg.eps,
            commitment_cost=cfg.commitment_cost,
            use_ema=cfg.use_ema,
            restart_interval=cfg.restart_interval,
        )

    def encode_z(self, x: torch.Tensor) -> torch.Tensor:
        """x → 量化前 latent z。"""
        return self.encoder(x)

    def decode_z(self, z: torch.Tensor) -> torch.Tensor:
        """latent z → 重建 x。"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode_z(x)
        z_q, indices, vq_loss, info = self.quantizer(z)
        x_hat = self.decode_z(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss
        info["recon_loss"] = recon_loss.item()
        info["vq_loss"] = vq_loss.item()
        info["loss"] = loss.item()
        return x_hat, indices, loss, info

    @torch.no_grad()
    def encode_to_sid(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, input_dim) → sid: (N, num_layers) 整数 codes。"""
        z = self.encode_z(x)
        return self.quantizer.encode(z)


class UnifiedRQVAE(nn.Module):
    """两路 (query/item) RQ-VAE, 共享前几层码本 + 各自专属后几层。

    用于方案 2 的统一语义空间, 让 query 和 item SID 共享前缀,
    便于倒排索引召回 + Hamming 距离粗排。
    """

    def __init__(self, cfg: RQVAEConfig, num_shared: int = 2,
                 num_individual: int = 3):
        super().__init__()
        # 关键约束: shared + individual 必须等于总层数
        assert num_shared + num_individual == cfg.num_layers
        self.num_shared = num_shared
        self.num_individual = num_individual

        # query 和 item 各自一套 encoder/decoder
        self.encoder_q = self._make_encoder(cfg)
        self.encoder_i = self._make_encoder(cfg)
        self.decoder_q = self._make_decoder(cfg)
        self.decoder_i = self._make_decoder(cfg)

        # 共享量化器 (前 num_shared 层), query 和 item 都过同一套码本
        self.shared_quantizer = ResidualQuantizer(
            num_layers=num_shared,
            codebook_size=cfg.codebook_size,
            latent_dim=cfg.latent_dim,
            ema_decay=cfg.ema_decay,
            eps=cfg.eps,
            commitment_cost=cfg.commitment_cost,
            use_ema=cfg.use_ema,
            restart_interval=cfg.restart_interval,
        )
        # 个体量化器 (后 num_individual 层), query 和 item 各自一套
        self.individual_q = ResidualQuantizer(
            num_layers=num_individual,
            codebook_size=cfg.codebook_size,
            latent_dim=cfg.latent_dim,
            ema_decay=cfg.ema_decay,
            eps=cfg.eps,
            commitment_cost=cfg.commitment_cost,
            use_ema=cfg.use_ema,
            restart_interval=cfg.restart_interval,
        )
        self.individual_i = ResidualQuantizer(
            num_layers=num_individual,
            codebook_size=cfg.codebook_size,
            latent_dim=cfg.latent_dim,
            ema_decay=cfg.ema_decay,
            eps=cfg.eps,
            commitment_cost=cfg.commitment_cost,
            use_ema=cfg.use_ema,
            restart_interval=cfg.restart_interval,
        )

    @staticmethod
    def _make_encoder(cfg: RQVAEConfig) -> nn.Module:
        return nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    @staticmethod
    def _make_decoder(cfg: RQVAEConfig) -> nn.Module:
        return nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    def _two_stage_encode(self, x: torch.Tensor, encoder: nn.Module,
                          individual: ResidualQuantizer) -> torch.Tensor:
        """两阶段编码: shared → 个体, 拼接成完整 SID。"""
        z = encoder(x)
        s_shared = self.shared_quantizer.encode(z)
        z_q_shared = self.shared_quantizer.decode(s_shared)
        # 第二阶段编码残差: z - z_q_shared
        residual = z - z_q_shared
        s_indiv = individual.encode(residual)
        return torch.cat([s_shared, s_indiv], dim=-1)

    @torch.no_grad()
    def encode_query_to_sid(self, q_emb: torch.Tensor) -> torch.Tensor:
        return self._two_stage_encode(q_emb, self.encoder_q, self.individual_q)

    @torch.no_grad()
    def encode_item_to_sid(self, i_emb: torch.Tensor) -> torch.Tensor:
        return self._two_stage_encode(i_emb, self.encoder_i, self.individual_i)

    def forward(self, q_emb: torch.Tensor, i_emb: torch.Tensor):
        zq = self.encoder_q(q_emb)
        zi = self.encoder_i(i_emb)
        # query 和 item 共用 shared_quantizer
        zq_q, _, vq_q_shared, _ = self.shared_quantizer(zq)
        zi_q, _, vq_i_shared, _ = self.shared_quantizer(zi)
        zq_residual = zq - zq_q
        zi_residual = zi - zi_q
        # 个体部分各自量化
        zq_indiv_q, _, vq_q_indiv, _ = self.individual_q(zq_residual)
        zi_indiv_q, _, vq_i_indiv, _ = self.individual_i(zi_residual)

        # 完整量化向量 = shared + individual
        zq_full = zq_q + zq_indiv_q
        zi_full = zi_q + zi_indiv_q
        # STE
        zq_st = zq + (zq_full - zq).detach()
        zi_st = zi + (zi_full - zi).detach()

        q_hat = self.decoder_q(zq_st)
        i_hat = self.decoder_i(zi_st)
        recon_q = F.mse_loss(q_hat, q_emb)
        recon_i = F.mse_loss(i_hat, i_emb)
        vq_loss = vq_q_shared + vq_i_shared + vq_q_indiv + vq_i_indiv
        loss = recon_q + recon_i + vq_loss
        return loss, {
            "recon_q": recon_q.item(),
            "recon_i": recon_i.item(),
            "vq_loss": vq_loss.item(),
            "loss": loss.item(),
        }


def train_rqvae(model: RQVAE, embeddings: torch.Tensor,
                epochs: int = 50, batch_size: int = 256,
                lr: float = 1e-3, device: str = "cpu",
                log_every: int = 5, verbose: bool = True) -> RQVAE:
    """训练 RQ-VAE: 单循环 Adam 优化 encoder+decoder (codebook 用 EMA 单独更新)。

    Args:
        model: 待训练的 RQVAE 实例。
        embeddings: (N, input_dim) 训练语料, 例如 item embeddings。
        epochs: 训练轮数。
        batch_size: 批大小。
        lr: 学习率 (只用于 encoder/decoder)。
        device: 计算设备 cpu/cuda。
        log_every: 日志打印间隔 (轮数)。
        verbose: 是否打印日志。

    Returns:
        训练好的 model, 已设为 eval 模式。
    """
    model = model.to(device)
    embeddings = embeddings.to(device)
    n = embeddings.size(0)

    # 优化器只更新 encoder/decoder, 不动 codebook (codebook 走 EMA)
    # 非 EMA 模式: 把 codebook 也加进优化器
    enc_dec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    if not model.cfg.use_ema:
        enc_dec_params += list(model.quantizer.codebooks.parameters())
    opt = torch.optim.Adam(enc_dec_params, lr=lr)

    model.train()
    for epoch in range(epochs):
        # 每轮打散样本顺序
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = embeddings[idx]
            opt.zero_grad()
            _, _, loss, info = model(batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            steps += 1

        # 日志: 每 log_every 轮打印一次, 第一轮也打
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"[rqvae] epoch {epoch+1}/{epochs}  "
                  f"loss={epoch_loss/max(steps,1):.4f}  "
                  f"recon={info['recon_loss']:.4f}  "
                  f"vq={info['vq_loss']:.4f}  "
                  f"perplexity={info['perplexity']:.1f}")
    model.eval()
    return model


def encode_to_sid(model: RQVAE, embeddings: torch.Tensor,
                  batch_size: int = 1024,
                  device: str = "cpu") -> torch.Tensor:
    """跑训练好的模型并返回 (N, num_layers) 整数 SID 矩阵 (CPU 上)。"""
    model = model.to(device)
    model.eval()
    embeddings = embeddings.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, embeddings.size(0), batch_size):
            batch = embeddings[start:start + batch_size]
            sid = model.encode_to_sid(batch).cpu()
            out.append(sid)
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = RQVAEConfig(input_dim=64, latent_dim=32, num_layers=5,
                      codebook_size=256, hidden_dim=64)
    model = RQVAE(cfg)
    fake = torch.randn(200, 64)
    train_rqvae(model, fake, epochs=10, batch_size=32, lr=1e-3, log_every=2)
    sid = encode_to_sid(model, fake)
    assert sid.shape == (200, 5), f"expected (200, 5), got {sid.shape}"
    assert (sid >= 0).all() and (sid < cfg.codebook_size).all()
    print(f"OK: SID tensor shape={tuple(sid.shape)}, unique counts per layer = "
          f"{[int(torch.unique(sid[:, i]).numel()) for i in range(5)]}")