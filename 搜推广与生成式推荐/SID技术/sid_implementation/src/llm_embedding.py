"""
基于 LLM 的端到端 embedding 提取器 (SID 流水线入口)

支持的模型:
  - 文本 LLM: Qwen2.5 / LLaMA-3 / GLM-4 / InternLM2.5 等
  - 多模态 LLM: Qwen2-VL / InternVL2 / LLaVA-OV (text + image)
  - 4B~7B 参数量, 支持 bf16 / int8 / int4 量化

支持的三种 pooling 策略:
  - mean: 对所有非 pad token 取平均, BERT 系常用
  - cls: 取第一个 token (BERT [CLS])
  - last: 取最后一个非 pad token, Qwen / LLaMA 推荐 (Eugene Yan 实践)

Public API:
    - EmbedderConfig: 配置 dataclass (模型名、dtype、量化、pooling 等)
    - TextEmbedder: 文本 LLM 提取器
    - MultimodalEmbedder: 多模态 LLM 提取器
    - HashEmbedder: ponytail 回退 (无 transformers 时用)
    - get_embedder: 工厂函数
    - texts_to_sid / items_to_sid: 端到端文本/多模态 → SID

Ponytail 决策:
  - 不硬依赖 transformers: 延迟 import, 让非 LLM 用户不用安装
  - transformers 不可用时自动回退到 HashEmbedder (确定性, 跑通流程)
  - 单文件不搞配置层
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# 让脚本可以直接 python -m src.llm_embedding 运行
_sys_path = str(Path(__file__).resolve().parent.parent)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """对所有非 pad token 取平均, 忽略 padding。

    Args:
        hidden: (B, T, D) 模型最后一层 hidden states。
        attention_mask: (B, T) long, 1 表示真 token, 0 表示 pad。

    Returns:
        (B, D) 平均后的 embedding。
    """
    # 把 mask 扩成 (B, T, 1) 与 hidden 同形, 再做加权求和
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    # 分母至少 1 (避免全 pad 时除 0, 此时该样本 embedding 全 0)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """取第一个 token (BERT [CLS] 风格)。

    适用场景: BERT 系模型, [CLS] 位置被预训练用作句向量。
    对 Qwen / LLaMA 这种 decoder-only, [CLS] 没有特殊语义, 不推荐。
    """
    return hidden[:, 0, :]


def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """取最后一个非 pad token (decoder-only LLM 推荐)。

    参考:
      - Qwen3-Embedding 模型卡
      - Eugene Yan 的 semantic-ids-llm 教程

    为什么对 Qwen/LLaMA 正确?
      decoder-only 模型用 causal mask, 最后一个 token 看到了所有上文,
      适合作为整句的语义聚合。

    为什么需要判断左右 padding?
      - 左 padding: 把所有 pad 放在开头, 最后一个位置 (-1) 永远是真 token
      - 右 padding: 把 pad 放在末尾, 需要用 mask.sum() 算出真实长度
    """
    # 检查 batch 是否全用左 padding: 所有样本最后一列都是 1
    left_padding = bool((attention_mask[:, -1] == 1).all().item())
    if left_padding:
        return hidden[:, -1]
    # 右 padding: 用 sum 算出每个样本的真实长度, 再 gather
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return hidden[
        torch.arange(hidden.size(0), device=hidden.device),
        sequence_lengths,
    ]


# pooling 函数名字 → 实际函数的映射, 用字符串配置避免传函数引用
POOL_FNS = {"mean": _mean_pool, "cls": _cls_pool, "last": _last_token_pool}


def verify_single_vs_batch(embed_fn, texts: Sequence[str]) -> bool:
    """单条 vs 批次一致性检查: 同一个 text 在 batch 中的 embedding 应等于单独跑。

    Eugene Yan 推荐的预检查: 捕获 padding bug (比如 batch 内右 padding 但模型按左 padding 处理)。
    早期发现这种 bug 可以省掉后期几小时调试。

    Args:
        embed_fn: (texts) -> Tensor 的可调用对象 (例如 TextEmbedder 实例)。
        texts: 至少 1 条文本。

    Returns:
        True 表示一致, False 表示存在 padding 处理问题。
    """
    if not texts:
        return True
    single = embed_fn([texts[0]])[0]
    batch = embed_fn(list(texts))[0]
    diff = torch.norm(single - batch).item()
    consistent = diff < 1e-3
    if not consistent:
        # 警告而不是抛错, 留出继续运行的空间
        print(f"[verify] WARN: single-vs-batch diff={diff:.4f} (likely padding bug)")
    return consistent


@dataclass
class EmbedderConfig:
    """embedding 提取器的全部配置。

    默认模型是 Qwen2.5-1.5B-Instruct:
      - 1.5B 参数量, 单卡可跑
      - bf16 精度约 3GB 显存
      - 中文 / 英文双语
      - 支持 instruction prompting, 适合做句向量
    """
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    pooling: str = "mean"  # "mean" / "cls" / "last"
    dtype: str = "bf16"    # "fp32" / "bf16" / "fp16"
    quantization: Optional[str] = None  # None / "int8" / "int4" (bitsandbytes)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    trust_remote_code: bool = False  # 是否信任远程自定义代码 (Qwen / GLM 需要)
    cache_dir: Optional[str] = None  # 模型下载/缓存目录
    output_dim: Optional[int] = None  # 截取前 N 维, 降低下游维度


class HashEmbedder:
    """Ponytail 回退 embedder: 用特征哈希 (hashing trick) 生成确定性 embedding。

    适用场景:
      - transformers / 模型权重不可用
      - 单元测试需要可复现 embedding
      - 离线 demo

    特点:
      - 完全确定性 (固定 seed): 同输入永远同输出
      - 零依赖 (只用 numpy)
      - 信息量低: 仅基于 token 词袋, 没有语义
    """

    def __init__(self, output_dim: int = 256, seed: int = 42):
        self.output_dim = output_dim
        self.seed = seed

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """texts → (N, output_dim) float32 tensor, 每行已 L2 归一化。

        算法 (经典 hashing trick):
          对每个文本:
            对每个 token t:
              h = hash((t, seed)) % output_dim  → 落到一个桶
              sign = +1 或 -1 (由 hash(t) 的奇偶决定)
              out[h] += sign
          最后做 L2 归一化。
        """
        rng = np.random.default_rng(self.seed)
        out = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # 简单按空格分词, 小写化 (跟真实 tokenizer 差距大, 但够 demo 用)
            for tok in t.lower().split():
                # 用 (token, seed) 元组做 hash, 保证不同 seed 下不同桶分布
                h = hash((tok, self.seed)) % self.output_dim
                # 用 hash(tok) 的奇偶做符号 (让正负 token 区分)
                sign = 1.0 if (hash(tok) & 1) else -1.0
                out[i, h] += sign
        # L2 归一化, 让 cosine 距离有意义
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return torch.from_numpy(out / norms)


class TextEmbedder:
    """纯文本 LLM embedding 提取器 (Qwen / LLaMA / GLM 等)。

    依赖: pip install transformers
    模型权重: 首次运行时会自动从 HuggingFace 下载到 cache_dir
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        # 延迟加载模型, 构造时不下载, 第一次 embed() 时才加载
        self._model = None
        self._tokenizer = None

    def _load(self):
        """懒加载 tokenizer 和 model。

        为什么要分开 _model 和 _tokenizer?
            transformers 把 tokenizer 和 model 分两个类, 前者处理文本编码,
            后者做实际推理; 加载顺序可以 tokenizer 先于 model (节省失败时的下载量)。
        """
        if self._model is not None:
            return  # 已加载过, 直接返回
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            # 显式 raise 而非静默回退: 调用方应该知道模型不可用
            raise ImportError(
                "TextEmbedder needs `transformers`. "
                "Install with `pip install transformers`."
            ) from e

        # dtype 字符串 → torch dtype 映射
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16,
                     "fp16": torch.float16}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        # 量化配置 (可选, 用 bitsandbytes)
        quant_config = None
        if self.cfg.quantization in ("int8", "int4"):
            try:
                from transformers import BitsAndBytesConfig
                if self.cfg.quantization == "int8":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    # int4 需要指定计算 dtype (一般与 torch_dtype 一致)
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
            except ImportError:
                raise ImportError(
                    "int8/int4 quantization needs `bitsandbytes`."
                )

        # 加载参数 dict, 同时处理 trust_remote_code / cache_dir 等
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=self.cfg.trust_remote_code,
            cache_dir=self.cfg.cache_dir,
        )
        if quant_config is not None:
            # 量化场景下 device_map 必须由 bitsandbytes 处理
            load_kwargs["quantization_config"] = quant_config
        if self.cfg.device == "cpu":
            load_kwargs["device_map"] = "cpu"
        elif self.cfg.device == "cuda":
            # "auto" 让 accelerate 自动选 device (单卡/多卡都行)
            load_kwargs["device_map"] = "auto"

        # tokenizer 必须先加载, 后面 model 推理要用
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=self.cfg.trust_remote_code,
            cache_dir=self.cfg.cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self.cfg.model_name, **load_kwargs
        )
        # 非量化场景下手动搬到目标设备 (device_map="auto" 有时不会放到指定 device)
        if quant_config is None and self.cfg.device not in ("cpu", "auto"):
            self._model = self._model.to(self.cfg.device)
        self._model.eval()  # 切到推理模式 (关闭 dropout)

    @torch.no_grad()
    def embed(self, texts: Sequence[str],
              batch_size: int = 8) -> torch.Tensor:
        """texts → (N, hidden_dim) float32 embedding (在 CPU 上)。

        Args:
            texts: 输入文本列表。
            batch_size: 每批大小, 默认 8 (避免 OOM; CPU 上可更大)。

        Returns:
            (N, hidden_dim) 张量, 在 CPU 上 (避免显存爆炸)。
        """
        self._load()
        pool_fn = POOL_FNS.get(self.cfg.pooling, _mean_pool)
        all_emb = []
        # 按 batch 切分循环推理
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            # padding=True 自动把 batch 内对齐到最长; truncation 防止超 max_length
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self._model.device)
            # output_hidden_states=False: 只需要 last_hidden_state, 省内存
            outputs = self._model(**inputs, output_hidden_states=False)
            hidden = outputs.last_hidden_state  # (B, T, D)
            pooled = pool_fn(hidden, inputs["attention_mask"])
            # 可选: 截取前 N 维 (output_dim), 给下游统一维度
            if self.cfg.output_dim and pooled.size(-1) != self.cfg.output_dim:
                pooled = pooled[:, :self.cfg.output_dim]
            # 转 CPU + float32, 避免 fp16/bf16 在 numpy 中处理麻烦
            all_emb.append(pooled.cpu().float())
        return torch.cat(all_emb, dim=0)

    def __call__(self, texts: Sequence[str], batch_size: int = 8) -> torch.Tensor:
        """让 embedder 实例可以像函数一样调用: embedder(texts)"""
        return self.embed(texts, batch_size=batch_size)


class MultimodalEmbedder:
    """多模态 LLM embedding 提取器 (Qwen2-VL / InternVL2 / LLaVA-OV 等)。

    与 TextEmbedder 的区别:
      - 输入是 dict 列表, 每个 dict 含 text + (可选) image
      - 用 AutoProcessor 而不是 AutoTokenizer (要处理图像)
      - 推理时把图像和文本一起送进模型

    依赖: pip install transformers pillow
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "MultimodalEmbedder needs `transformers`."
            ) from e

        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16,
                     "fp16": torch.float16}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        # processor 同时处理文本和图像
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=self.cfg.trust_remote_code,
            cache_dir=self.cfg.cache_dir,
        )
        # device_map: "auto" 让 accelerate 自动分配; 也可显式传 cpu/cuda
        self._model = AutoModel.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch_dtype,
            device_map=self.cfg.device if self.cfg.device in ("cpu", "auto")
                         else "auto",
            trust_remote_code=self.cfg.trust_remote_code,
            cache_dir=self.cfg.cache_dir,
        )
        if self.cfg.device == "cuda" and not self.cfg.quantization:
            self._model = self._model.to("cuda")
        self._model.eval()

    @torch.no_grad()
    def embed(self, items: Sequence[dict],
              batch_size: int = 4) -> torch.Tensor:
        """items → (N, hidden_dim) embedding。

        Args:
            items: List[{"text": str, "image": Optional[PIL.Image]}], 每个元素一件多模态数据。
            batch_size: 多模态 batch 通常比纯文本小 (显存占用大)。

        Returns:
            (N, hidden_dim) float32 tensor in CPU。
        """
        self._load()
        pool_fn = POOL_FNS.get(self.cfg.pooling, _mean_pool)
        all_emb = []
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [it.get("text", "") for it in batch_items]
            images = [it.get("image") for it in batch_items]
            # 任一元素带 image 就要走多模态分支
            has_any_image = any(im is not None for im in images)
            if has_any_image:
                try:
                    inputs = self._processor(
                        text=texts, images=images,
                        padding=True, truncation=True,
                        max_length=self.cfg.max_length,
                        return_tensors="pt",
                    ).to(self._model.device)
                except Exception as e:
                    # 某些模型对 image 兼容性差, 回退到 text-only
                    print(f"[warn] processor failed ({e}); falling back to text-only")
                    inputs = self._processor(
                        text=texts, padding=True, truncation=True,
                        max_length=self.cfg.max_length,
                        return_tensors="pt",
                    ).to(self._model.device)
            else:
                # 全文本 batch, 直接走 text 路径 (更快)
                inputs = self._processor(
                    text=texts, padding=True, truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                ).to(self._model.device)
            outputs = self._model(**inputs, output_hidden_states=False)
            hidden = outputs.last_hidden_state
            pooled = pool_fn(hidden, inputs["attention_mask"])
            if self.cfg.output_dim and pooled.size(-1) != self.cfg.output_dim:
                pooled = pooled[:, :self.cfg.output_dim]
            all_emb.append(pooled.cpu().float())
        return torch.cat(all_emb, dim=0)

    def __call__(self, items: Sequence[dict], batch_size: int = 4) -> torch.Tensor:
        return self.embed(items, batch_size=batch_size)


def get_embedder(cfg: EmbedderConfig, *, multimodal: bool = False):
    """工厂: 根据 multimodal flag 返回 TextEmbedder 或 MultimodalEmbedder。

    Ponytail 简化: 不搞复杂的注册表, 一个 if 就够。
    """
    if multimodal:
        return MultimodalEmbedder(cfg)
    return TextEmbedder(cfg)


def texts_to_sid(texts: Sequence[str],
                 embedder_cfg: Optional[EmbedderConfig] = None,
                 tokenizer_type: str = "rq_kmeans",
                 num_layers: int = 5,
                 codebook_size: int = 256,
                 sid_kwargs: Optional[dict] = None,
                 use_fallback: bool = True) -> Tuple[torch.Tensor, object]:
    """端到端: texts → SID 矩阵 + 训练好的 tokenizer。

    完整流程: text → LLM embed → RQ-Kmeans (或 RQ-VAE) 训练 → SID
    LLM 失败时自动回退到 HashEmbedder (use_fallback=True)。

    Args:
        texts: 输入文本。
        embedder_cfg: EmbedderConfig; None 时用默认 Qwen2.5-1.5B。
        tokenizer_type: "rq_kmeans" 或 "rq_vae"。
        num_layers: SID 层数。
        codebook_size: 每层码本大小。
        sid_kwargs: 透传给 tokenizer 的额外参数。
        use_fallback: LLM 不可用时是否回退到 HashEmbedder。

    Returns:
        ((N, num_layers) int64 SID, 训练好的 tokenizer 对象)。
    """
    sid_kwargs = sid_kwargs or {}
    cfg = embedder_cfg or EmbedderConfig()

    # 第一步: 拿 embedding (LLM 或 fallback)
    try:
        embedder = get_embedder(cfg, multimodal=False)
        embs = embedder(texts)
    except (ImportError, OSError) as e:
        # ImportError: 没装 transformers; OSError: 模型下载失败 / 网络问题
        if not use_fallback:
            raise
        print(f"[warn] LLM unavailable ({e}); using HashEmbedder fallback")
        fb = HashEmbedder(output_dim=cfg.output_dim or 256)
        embs = fb.embed(texts)

    # 第二步: 训练 tokenizer + 编码 SID
    if tokenizer_type == "rq_kmeans":
        from src.rq_kmeans import fit_rq_kmeans, encode_with_rq_kmeans
        # verbose 关键字传给 fit_rq_kmeans 会和 sid_kwargs 冲突, 这里过滤掉
        sid_kwargs = {k: v for k, v in sid_kwargs.items() if k != "verbose"}
        tok = fit_rq_kmeans(embs, num_layers=num_layers,
                            codebook_size=codebook_size,
                            device="cpu", verbose=False, **sid_kwargs)
        sid = encode_with_rq_kmeans(tok, embs)
    elif tokenizer_type == "rq_vae":
        from src.rq_vae import RQVAE, RQVAEConfig, train_rqvae, encode_to_sid
        # RQ-VAE 配置自适应 embedding 维度
        vae_cfg = RQVAEConfig(
            input_dim=embs.size(1),
            hidden_dim=min(512, embs.size(1) * 4),
            latent_dim=min(256, embs.size(1)),
            num_layers=num_layers,
            codebook_size=codebook_size,
        )
        model = RQVAE(vae_cfg)
        train_rqvae(model, embs, epochs=sid_kwargs.get("epochs", 20),
                    batch_size=sid_kwargs.get("batch_size", 64),
                    lr=1e-3, verbose=False)
        sid = encode_to_sid(model, embs)
        tok = model
    else:
        raise ValueError(f"unknown tokenizer_type: {tokenizer_type}")

    return sid, tok


def items_to_sid(items: Sequence[dict],
                 embedder_cfg: Optional[EmbedderConfig] = None,
                 num_layers: int = 5,
                 codebook_size: int = 256,
                 sid_kwargs: Optional[dict] = None,
                 use_fallback: bool = True) -> Tuple[torch.Tensor, object]:
    """端到端多模态版: items[{text, image}] → SID + tokenizer。

    与 texts_to_sid 的区别: 用多模态 embedder; 多模态不可用时回退到文本 HashEmbedder。
    """
    sid_kwargs = sid_kwargs or {}
    cfg = embedder_cfg or EmbedderConfig()

    try:
        embedder = get_embedder(cfg, multimodal=True)
        embs = embedder(items)
    except (ImportError, OSError) as e:
        if not use_fallback:
            raise
        print(f"[warn] multimodal unavailable ({e}); using HashEmbedder on text only")
        fb = HashEmbedder(output_dim=cfg.output_dim or 256)
        embs = fb.embed([it.get("text", "") for it in items])

    # 多模态版本默认用 RQ-Kmeans (论文中也是这么配的)
    from src.rq_kmeans import fit_rq_kmeans, encode_with_rq_kmeans
    sid_kwargs = {k: v for k, v in sid_kwargs.items() if k != "verbose"}
    tok = fit_rq_kmeans(embs, num_layers=num_layers,
                        codebook_size=codebook_size,
                        device="cpu", verbose=False, **sid_kwargs)
    sid = encode_with_rq_kmeans(tok, embs)
    return sid, tok


# ---------------------------------------------------------------------------
# 自测 (无 LLM 环境也能跑通, 用 HashEmbedder fallback)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = EmbedderConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        pooling="mean",
        dtype="bf16",
        output_dim=256,
    )

    # 用 HashEmbedder 直接演示端到端, 避免下载 1.5B 模型
    fallback = HashEmbedder(output_dim=256)
    embs = fallback.embed([
        "iPhone 15 Pro Max 256GB Titanium",
        "Samsung Galaxy S24 Ultra",
        "Sony WH-1000XM5 wireless headphones",
        "Nike Air Force 1 sneakers white",
        "Kindle Paperwhite e-reader",
    ])
    print(f"Fallback embeddings: {tuple(embs.shape)}")

    from src.rq_kmeans import fit_rq_kmeans, encode_with_rq_kmeans
    tok = fit_rq_kmeans(embs, num_layers=5, codebook_size=64,
                        n_passes=2, batch_size=2, verbose=False)
    sid = encode_with_rq_kmeans(tok, embs)
    print(f"SID shape: {tuple(sid.shape)}")
    for i, t in enumerate([
        "iPhone 15 Pro Max", "Galaxy S24", "WH-1000XM5",
        "Air Force 1", "Kindle Paperwhite",
    ]):
        print(f"  {t:30s} -> {sid[i].tolist()}")
    print("OK (fallback path)")