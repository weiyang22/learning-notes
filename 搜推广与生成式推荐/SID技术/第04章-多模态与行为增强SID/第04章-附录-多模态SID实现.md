---
title: 附录-多模态SID实现
tags:
  - 多模态
  - SID
  - 实现
  - CLIP
  - 视觉-文本对齐
description: 多模态 SID 系统完整 PyTorch 实现. 文本/图像双塔融合 → 残差量化 → 跨模态语义空间. 附完整训练评估代码
last_updated: 2026-09-11
status: ✅ v7.2 - 多模态 SID 完整实现
chapter_id: SID技术-4
---

# 附录-多模态 SID 实现

> 本附录为第 04 章 多模态与行为增强 SID 提供完整 PyTorch 实现. 涵盖 文本/图像双塔 → 残差量化 → 跨模态语义空间 → 训练 + 评估. 整合 [happy-llm](https://github.com/datawhalechina/happy-llm) 与 [self-llm](https://github.com/datawhalechina/self-llm) 的实战经验.

---

## 1. 系统架构

```text
                      ┌─────────────────────────┐
                      │     文本编码器 (BERT)     │
                      └────────────┬────────────┘
                                   ↓
文本 "黑色连衣裙"  ─────────────→ t_emb (768d)
                                   ↓
                      ┌─────────────────────────┐
                      │     图像编码器 (CLIP)     │
                      └────────────┬────────────┘
                                   ↓
图片 [H, W, 3]     ─────────────→ i_emb (512d)
                                   ↓
                      ┌─────────────────────────┐
                      │    投影 + 融合 (Linear)   │
                      └────────────┬────────────┘
                                   ↓
                       融合向量 f_emb (640d)
                                   ↓
                ┌──────────────────────────────┐
                │   RQ-VAE 编码器 (残差量化)     │
                └──────────────────────────────┘
                                   ↓
                       SID: [c1, c2, c3, c4]
```

---

## 2. 文本编码器

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """基于 BERT 的文本编码器"""

    def __init__(self, model_name="bert-base-chinese", cache_dir=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # 输出维度
        self.output_dim = self.bert.config.hidden_size  # 768 for BERT-base

    def forward(self, texts, device="cuda"):
        """
        Args:
            texts: list[str] 文本列表
        Returns:
            embeddings: [B, 768] 文本嵌入
        """
        # Tokenize
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        ).to(device)

        # BERT 编码
        outputs = self.bert(**inputs)

        # 取 [CLS] token 的隐藏状态
        embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        return embeddings

    def freeze(self):
        """冻结 BERT 参数, 只训练下游模块"""
        for param in self.bert.parameters():
            param.requires_grad = False
```

---

## 3. 图像编码器

```python
import timm
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """基于 CLIP / ViT 的图像编码器"""

    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        self.output_dim = self.backbone.num_features  # 768 for ViT-base
        self.image_size = 224

    def forward(self, images):
        """
        Args:
            images: [B, 3, 224, 224] 预处理后的图片
        Returns:
            embeddings: [B, 768] 图像嵌入
        """
        return self.backbone(images)

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
```

---

## 4. 多模态融合投影

```python
class MultimodalFusion(nn.Module):
    """文本 + 图像 → 统一向量空间"""

    def __init__(self, text_dim=768, image_dim=768, fused_dim=640):
        super().__init__()
        # 文本投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim)
        )

        # 图像投影
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim)
        )

        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=8, batch_first=True
        )

    def forward(self, text_emb, image_emb=None):
        """
        Args:
            text_emb: [B, text_dim]
            image_emb: [B, image_dim] 或 None (纯文本场景)
        Returns:
            fused: [B, fused_dim]
        """
        t = self.text_proj(text_emb)  # [B, fused_dim]

        if image_emb is None:
            return t

        i = self.image_proj(image_emb)  # [B, fused_dim]

        # 跨模态融合
        t_seq = t.unsqueeze(1)  # [B, 1, D]
        i_seq = i.unsqueeze(1)  # [B, 1, D]
        combined = torch.cat([t_seq, i_seq], dim=1)  # [B, 2, D]
        attended, _ = self.cross_attn(combined, combined, combined)
        fused = attended.mean(dim=1)  # [B, D]

        return fused
```

---

## 5. 多模态 RQ-VAE 编码器

```python
class MultimodalRQVAE(nn.Module):
    """多模态残差量化 VAE"""

    def __init__(
        self,
        input_dim=640,    # 融合维度
        hidden_dim=1024,
        num_quantizers=4,
        codebook_size=256
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # Encoder: 融合向量 → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)  # 输出与输入同维
        )

        # Decoder: latent → 重建
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 多个码本 (残差量化)
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, input_dim) for _ in range(num_quantizers)
        ])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def encode(self, x):
        """编码为 SID tokens"""
        z = self.encoder(x)
        residual = z
        codes = []
        for cb in self.codebooks:
            # 找最近码
            distances = torch.cdist(
                residual.unsqueeze(1),
                cb.weight.unsqueeze(0)
            )  # [B, 1, C]
            code = distances.argmin(dim=-1).squeeze(1)  # [B]
            codes.append(code)

            # 减去量化值, 残差
            quantized = cb(code)  # [B, D]
            residual = residual - quantized

        return torch.stack(codes, dim=1)  # [B, num_quantizers]

    def decode(self, codes):
        """从 SID tokens 重建"""
        z = torch.zeros(codes.size(0), self.codebooks[0].weight.size(1),
                       device=codes.device)
        for i, cb in enumerate(self.codebooks):
            z = z + cb(codes[:, i])
        return self.decoder(z)

    def forward(self, x):
        codes = self.encode(x)
        x_recon = self.decode(codes)
        return x_recon, codes


# === STE 直通估计器 ===
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook):
        # 找最近码
        distances = torch.cdist(x.unsqueeze(1), codebook.weight.unsqueeze(0))
        codes = distances.argmin(dim=-1).squeeze(1)
        quantized = codebook(codes)
        return quantized, codes

    @staticmethod
    def backward(ctx, grad_output, grad_codes):
        # 直通估计
        return grad_output, None


def rq_vae_step(z_e, codebook):
    """单步 RQ-VAE 量化"""
    quantized, codes = StraightThroughEstimator.apply(z_e, codebook)
    return quantized, codes
```

---

## 6. 完整多模态 SID 模型

```python
class MultimodalSIDModel(nn.Module):
    """完整的多模态语义 ID 模型"""

    def __init__(self, num_quantizers=4, codebook_size=256):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.fusion = MultimodalFusion(text_dim=768, image_dim=768, fused_dim=640)

        self.rq_vae = MultimodalRQVAE(
            input_dim=640,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        )

        # 跨模态对齐头 (对比学习)
        self.contrastive_head = nn.Linear(640, 128)

    def freeze_encoders(self):
        """预训练后冻结文本/图像编码器"""
        self.text_encoder.freeze()
        self.image_encoder.freeze()

    def forward(self, texts=None, images=None):
        """
        Args:
            texts: list[str] 文本
            images: [B, 3, 224, 224] 图像
        Returns:
            codes: [B, num_quantizers] SID tokens
            contrastive_emb: 对比学习嵌入
        """
        # 编码
        text_emb = self.text_encoder(texts) if texts else None
        image_emb = self.image_encoder(images) if images is not None else None

        # 多模态融合
        fused = self.fusion(text_emb, image_emb)  # [B, 640]

        # 量化得到 SID
        codes = self.rq_vae.encode(fused)  # [B, num_quantizers]

        # 对比学习嵌入
        contrastive_emb = self.contrastive_head(fused)  # [B, 128]
        contrastive_emb = nn.functional.normalize(contrastive_emb, dim=-1)

        return codes, contrastive_emb, fused
```

---

## 7. 数据集与训练循环

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class MultimodalSEDataset(Dataset):
    """多模态 (文本 + 图像) SID 训练数据集"""

    def __init__(self, data_list, tokenizer, image_size=224):
        """
        data_list: [
            {"text": "...", "image_path": "..."},
            ...
        ]
        """
        self.data = data_list
        self.tokenizer = tokenizer

        # 图像预处理
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        image = Image.open(item['image_path']).convert('RGB')
        image = self.image_transform(image)
        return {'text': text, 'image': image}


def train_step(model, batch, optimizer, device='cuda'):
    """单步训练"""
    model.train()
    optimizer.zero_grad()

    texts = batch['text']
    images = batch['image'].to(device)

    # 前向
    codes, contrastive_emb, fused = model(texts, images)

    # === 损失 1: 重建损失 (RQ-VAE) ===
    x_recon, _ = model.rq_vae(fused.detach())
    recon_loss = torch.nn.functional.mse_loss(x_recon, fused.detach())

    # === 损失 2: 对比学习 (跨模态对齐) ===
    # 文本嵌入 (用同一个 text_encoder 但 no grad)
    with torch.no_grad():
        text_only_emb = model.fusion.text_proj(model.text_encoder(texts))

    text_proj = model.contrastive_head(text_only_emb.detach())
    text_proj = torch.nn.functional.normalize(text_proj, dim=-1)

    contrastive_loss = 1 - (text_proj * contrastive_emb).sum(dim=-1).mean()

    # === 损失 3: 承诺损失 (commitment loss) ===
    commit_loss = torch.nn.functional.mse_loss(fused, x_recon.detach())

    # 总损失
    loss = recon_loss + 0.1 * contrastive_loss + 0.05 * commit_loss
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'contrastive_loss': contrastive_loss.item()
    }


def evaluate_sid_quality(model, dataloader, device='cuda'):
    """评估 SID 质量: 重建误差 + 检索准确率"""
    model.eval()
    total_recon = 0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            images = batch['image'].to(device)

            codes, _, fused = model(texts, images)

            # 重建
            x_recon, _ = model.rq_vae(fused)
            total_recon += torch.nn.functional.mse_loss(
                x_recon, fused, reduction='sum'
            ).item()
            n += fused.size(0)

    return {
        'recon_mse': total_recon / n,
        'avg_codes_per_item': 4  # num_quantizers
    }
```

---

## 8. 端到端使用示例

```python
import torch
from torch.optim import AdamW

# 1. 准备数据 (示意)
data_list = [
    {"text": "黑色连衣裙", "image_path": "dress1.jpg"},
    {"text": "运动鞋", "image_path": "shoes1.jpg"},
    # ... 更多数据
]

# 2. 初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultimodalSIDModel(num_quantizers=4, codebook_size=256).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

dataset = MultimodalSEDataset(data_list, model.text_encoder.tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 训练
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        metrics = train_step(model, batch, optimizer, device)
        total_loss += metrics['loss']

    eval_metrics = evaluate_sid_quality(model, dataloader, device)
    print(f"Epoch {epoch+1}: train_loss={total_loss/len(dataloader):.4f}, "
          f"recon_mse={eval_metrics['recon_mse']:.4f}")

# 4. 推断: 文本 → SID
model.eval()
with torch.no_grad():
    codes, _, _ = model(texts=["黑色连衣裙"])
    print(f"黑色连衣裙 → SID: {codes[0].tolist()}")
    # e.g. [23, 145, 67, 234]

# 5. 推断: 图像 → SID
with torch.no_grad():
    from PIL import Image
    img = Image.open("query.jpg").convert("RGB")
    img_tensor = model.image_encoder.image_transform(img).unsqueeze(0).to(device)
    codes, _, _ = model(images=img_tensor)
    print(f"图片 → SID: {codes[0].tolist()}")

# 6. 跨模态检索 (用 SID 索引)
# 实际部署时, 把所有商品 SID 预计算存入 FAISS
# 检索时, 计算 query SID, 然后 ANN 找最近邻
```

---

## 9. 关键公式

### 9.1 跨模态融合

$$f_{\text{fused}} = \text{CrossAttn}(\text{Proj}_t(t_{\text{emb}}), \text{Proj}_i(i_{\text{emb}}))$$

### 9.2 残差量化 (RQ)

$$z_e = \text{Encoder}(f_{\text{fused}})$$
$$c_k = \arg\min_{c \in C_k} \| r_{k-1} - e_c \|^2$$
$$r_k = r_{k-1} - e_{c_k}$$
$$\text{SID} = [c_1, c_2, \ldots, c_K]$$

### 9.3 对比学习 (InfoNCE)

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(s(q, k^+)/\tau)}{\sum_{k \in \text{batch}} \exp(s(q, k)/\tau)}$$

### 9.4 重建损失

$$\mathcal{L}_{\text{recon}} = \| f_{\text{fused}} - \text{Decoder}(\sum_{k=1}^{K} e_{c_k}) \|^2$$

---

## 10. 常见问题与调试

### Q1: 训练时 loss 不下降

**诊断步骤**:
1. 检查 `recon_loss` — 如果 > 1.0, 说明 encoder/decoder 没学到东西
2. 检查 `contrastive_loss` — 如果接近 1.0, 说明文本/图像没对齐
3. 降低学习率到 1e-5, 观察 100 个 step

### Q2: SID 重建率低

**解决**:
- 增加 `num_quantizers` (4 → 8)
- 增大 `codebook_size` (256 → 1024)
- 减小重建目标 (先用 MSE, 后用 cosine)

### Q3: 跨模态检索差

**解决**:
- 训练时让图像文本对齐 (CLIP 风格)
- 推理时同时用文本和图像的 SID 加权融合
- 增加 batch size (更大的对比学习 batch)

---

## 11. 参考资源

- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) — 第 5 章 LLaMA2 实现
- [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) — 50+ 模型实战
- [datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook) — Prompt Engineering
- [CLIP 论文](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [RQ-VAE 论文](https://arxiv.org/abs/2203.01935) — Lee et al., 2022 (Google)

---

> 配套阅读: [第04章 多模态与行为增强SID](../第04章-多模态与行为增强SID/) — 本附录是该章节的代码落地.
