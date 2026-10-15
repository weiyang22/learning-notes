"""
End-to-end SID: raw text/multimodal -> LLM embedding -> RQ-VAE/Kmeans -> SID.

【本文件做了什么】
================
把"完整真实链路"从头跑一遍:
    1. 准备 8 个商品的真实文本描述(iPhone / Galaxy / 索尼耳机 / Nike 鞋 / Kindle 等)
    2. 用 LLM(默认 Qwen2.5-1.5B / 可选 Qwen2-VL-7B)把文本 → dense embedding
       (没有 GPU 或没装 transformers 时,自动降级到确定性 HashEmbedder)
    3. 用 Mini-batch RQ-Kmeans 把 embedding → 5 层 SID
    4. 计算每物品的重建误差并打印
    5. 保存 embedding / SID / metadata / tokenizer

【核心论点】
================
这是 SID 链路最朴素、最可解释的形态:
    raw_text  →  LLM.encode  →  dense (N, D)  →  RQ-Kmeans.fit  →  (N, 5) SID
对比方案 1(run_candidate_sid.py):
    1. 多了"LLM 编码"这一步,因为这里 demo 用真实文本
    2. 没用 RQ-VAE,改用 RQ-Kmeans(更快、纯无监督、TRM 论文原算法)
    3. 没有用户行为参与 → 这是 cold-start 友好路径

【环境变量】
================
SID_USE_LLM=1    强制使用真实 LLM(否则用 HashEmbedder)
SID_MODEL_NAME   指定模型名,默认 Qwen2.5-1.5B-Instruct
SID_MULTIMODAL=1 用多模态模型(Qwen2-VL-7B),需要图片输入
SID_QUANT=int8   用 8-bit 量化(需要 bitsandbytes)

【运行】
================
    # 快速模式(无需 GPU,无 LLM weights,直接看输出)
    python3 examples/run_end_to_end_sid.py

    # 真实 LLM 模式(需要 GPU + 下载 weights)
    SID_USE_LLM=1 python3 examples/run_end_to_end_sid.py

    # 多模态模式
    SID_MULTIMODAL=1 SID_USE_LLM=1 python3 examples/run_end_to_end_sid.py

【推荐先读】
================
    - docs/00_sid_overview_and_principles.md § 2(LLM→embedding 原理)
    - docs/01_candidate_sid_with_behavior.md § 3(整体方案)
    - src/llm_embedding.py(LLM 包装实现)
    - src/rq_kmeans.py(RQ-Kmeans 实现)
"""
import json
import os
import sys
from pathlib import Path

# 同 run_candidate_sid.py,把仓库根加到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# EmbedderConfig      : LLM 配置(model_name/pooling/dtype/...)
# get_embedder        : 工厂,根据 multimodal 标志返回 Text 或 Multimodal 类
# HashEmbedder        : 兜底,确定性 hashing,不依赖 transformers
# texts_to_sid        : 文本 → SID 的一站式接口
# items_to_sid        : {text, image} 多模态 → SID 的一站式接口
from src.llm_embedding import (
    EmbedderConfig, get_embedder, HashEmbedder,
    texts_to_sid, items_to_sid,
)


# ============================================================
# 示例商品数据 (8 个真实物品,各自一段描述)
# 这里用纯文本(无 image),所以走 TextEmbedder 路径
# 如果在多模态模式下,可以把 image 字段填上 PIL.Image 对象
# ============================================================
SAMPLE_ITEMS = [
    {
        "text": "Apple iPhone 15 Pro Max 256GB Natural Titanium, A17 Pro chip, 48MP camera",
        "image": None,
    },
    {
        "text": "Samsung Galaxy S24 Ultra 512GB Titanium Black, 200MP camera, S Pen",
        "image": None,
    },
    {
        "text": "Sony WH-1000XM5 wireless noise cancelling headphones, 30hr battery",
        "image": None,
    },
    {
        "text": "Nike Air Force 1 '07 white sneakers, classic basketball shoes",
        "image": None,
    },
    {
        "text": "Amazon Kindle Paperwhite 16GB e-reader, 6.8 inch glare-free display",
        "image": None,
    },
    {
        "text": "Sony A7 IV full-frame mirrorless camera body, 33MP, 4K 60p video",
        "image": None,
    },
    {
        "text": "Dyson V15 Detect cordless vacuum cleaner, laser dust detection",
        "image": None,
    },
    {
        "text": "Logitech MX Master 3S wireless mouse, 8K DPI, quiet clicks",
        "image": None,
    },
]


def get_embedder_smart():
    """
    【智能选择 embedder】
    ================
    根据环境变量决定用真 LLM 还是 HashEmbedder:
        - SID_USE_LLM=1  强制真 LLM(失败也尝试降级)
        - 否则          直接用 HashEmbedder(不下载 weights)

    返回:
        embedder   : TextEmbedder / MultimodalEmbedder / HashEmbedder 的实例
        cfg        : 配置对象,后面 save 到 meta.json 用
        is_fallback: bool,True 表示用了 HashEmbedder
    """
    # SID_USE_LLM=0 是默认值,=1 才用真 LLM
    use_llm = os.environ.get("SID_USE_LLM", "0") == "1"

    # 选模型:多模态则用 Qwen2-VL-7B,否则用 Qwen2.5-1.5B(轻量)
    cfg = EmbedderConfig(
        model_name=os.environ.get(
            "SID_MODEL_NAME",
            "Qwen/Qwen2-VL-7B-Instruct" if os.environ.get("SID_MULTIMODAL", "0") == "1"
            else "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        pooling="mean",   # 用 mean pooling(decoder-only 也可以改成 last_token)
        dtype="bf16",     # bf16 是当前 LLM 推理的事实标准(数值稳定 + 显存节省)
        quantization=os.environ.get("SID_QUANT", None),  # None/int8/int4
        output_dim=256,   # 截断到 256 维,下游 RQ-Kmeans 默认吃这个维度
        max_length=256,   # 长文本会被截断(实际产品描述通常 < 100 token)
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 多模态标志
    multimodal = os.environ.get("SID_MULTIMODAL", "0") == "1"

    if use_llm:
        try:
            print(f"[e2e] Using real LLM: {cfg.model_name} (multimodal={multimodal})")
            return get_embedder(cfg, multimodal=multimodal), cfg, False
        except (ImportError, OSError) as e:
            # 真实场景里 ImportError=没装 transformers, OSError=weights 没下载或磁盘满
            print(f"[e2e] LLM unavailable ({e}); falling back to HashEmbedder")

    # 默认路径(快速、零依赖、可复现)
    print("[e2e] Using HashEmbedder (set SID_USE_LLM=1 to use real LLM)")
    return HashEmbedder(output_dim=cfg.output_dim), cfg, True


def main():
    # ============================================================
    # 0. Banner
    # ============================================================
    print("=" * 70)
    print("END-TO-END SID: Text/Multimodal -> LLM Embedding -> RQ-Kmeans -> SID")
    print("=" * 70)

    # ============================================================
    # 1. 拿 embedder 并编码所有商品 → (N, D)
    # ============================================================
    embedder, cfg, is_fallback = get_embedder_smart()

    # 这里有一个分支逻辑(因为 embedder 接口略有差异):
    #   - HashEmbedder          → 只有 .embed(texts)
    #   - TextEmbedder          → .embed(texts, batch_size) 或 __call__(texts, batch_size)
    #   - MultimodalEmbedder    → 接受 list[dict],传 {text, image}
    if is_fallback:
        embs = embedder.embed([it["text"] for it in SAMPLE_ITEMS])
    else:
        if cfg.pooling and hasattr(embedder, "embed"):
            # isinstance(embedder, type(None)) is False 是为了让 linter 闭嘴:embedder 必非 None
            if isinstance(embedder, type(None)) is False:
                try:
                    # 优先用 __call__ 接口(支持 batch_size)
                    embs = embedder([it["text"] for it in SAMPLE_ITEMS], batch_size=4)
                except Exception:
                    # 退化到 .embed()
                    embs = embedder.embed([it["text"] for it in SAMPLE_ITEMS])
            else:
                embs = embedder.embed([it["text"] for it in SAMPLE_ITEMS])
        else:
            embs = embedder.embed([it["text"] for it in SAMPLE_ITEMS])
    # embs 现在是 torch.Tensor,shape (N, output_dim)=(8, 256)
    print(f"Embedding shape: {tuple(embs.shape)}, dtype: {embs.dtype}")

    # ============================================================
    # 2. 训练 RQ-Kmeans 并产出 SID
    # ============================================================
    # 这里手动拆开跑(fit + encode)而不是用 texts_to_sid,
    # 是为了后面能算"重建误差"
    from src.rq_kmeans import fit_rq_kmeans, encode_with_rq_kmeans
    num_layers = 5         # SID 长度 L
    codebook_size = 64      # 每层 K(故意用 64 而不是 256,这样 8 个物品更好聚类)
    # fit_rq_kmeans 走的是 Sculley 2010 mini-batch K-means + K-means++ init
    # n_passes=5 表示扫整个数据集 5 遍
    # verbose=True 会打印每层初始化情况
    tok = fit_rq_kmeans(embs, num_layers=num_layers, codebook_size=codebook_size,
                        n_passes=5, batch_size=4, verbose=True)
    # 训练完后,对每个 embedding 编码出 SID
    sid = encode_with_rq_kmeans(tok, embs)
    # sid.shape = (N, L) = (8, 5),int64
    print(f"SID shape: {tuple(sid.shape)}")

    # ============================================================
    # 3. 打印每件商品的 SID + 重建误差
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for i, item in enumerate(SAMPLE_ITEMS):
        # 截断标题方便打印
        title = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
        sid_tuple = sid[i].tolist()
        print(f"  [{i}] {title}")
        print(f"      SID = {sid_tuple}")
        # 用 SID 反向 "解码" 回到 embedding,然后跟原 embedding 算欧氏距离
        # 重建误差小 = SID 保留了大部分语义信息
        recon = tok.decode_from_sid(sid[i:i+1])
        err = float(torch.norm(embs[i] - recon[0]).item())
        print(f"      Reconstruction error: {err:.4f}")

    # ============================================================
    # 4. 保存所有结果(给下游 / 调试用)
    # ============================================================
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    # 存 embedding 和 SID 为 npy(可被 numpy/PyTorch 加载)
    np.save(out_dir / "end_to_end_embs.npy", embs.numpy())
    np.save(out_dir / "end_to_end_sids.npy", sid.numpy())
    # 存元信息:用什么模型、SID 超参、每件商品对应的文本和 SID
    with open(out_dir / "end_to_end_meta.json", "w") as f:
        json.dump({
            "model": cfg.model_name if not is_fallback else "HashEmbedder(fallback)",
            "is_fallback": is_fallback,
            "num_layers": num_layers,
            "codebook_size": codebook_size,
            "pooling": cfg.pooling,
            "output_dim": embs.size(1),
            "items": [
                {"text": it["text"], "sid": sid[i].tolist()}
                for i, it in enumerate(SAMPLE_ITEMS)
            ],
            # ensure_ascii=False 让中文不被转义
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_dir}/end_to_end_{{embs,sids,meta}}")

    # 保存 tokenizer(codebook),这样下次可以"推理",不用重训
    tok.save(str(out_dir / "end_to_end_rq_kmeans.pt"))
    print(f"Saved tokenizer to {out_dir}/end_to_end_rq_kmeans.pt")


if __name__ == "__main__":
    main()
