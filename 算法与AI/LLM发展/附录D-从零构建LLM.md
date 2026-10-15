---
title: 附录D 从零构建大语言模型 (LLaMA2 实战)
tags:
  - 从零构建LLM
  - LLaMA2
  - 预训练
  - Tokenizer
  - RMSNorm
  - SwiGLU
  - RoPE
  - 数据处理
description: 从零构建大语言模型. 整合 happy-llm 第 5-6 章内容. 实现 Tokenizer → 预训练 → SFT → LoRA 的完整流程
last_updated: 2026-09-11
status: ✅ v1.0 - 整合 happy-llm
chapter_id: appendix-d
---

# 附录D 从零构建大语言模型 (LLaMA2 实战)

> 本附录整合自 [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) 第 5-6 章. 我们将从零实现一个简化版 LLaMA2, 完整跑通预训练 → SFT → LoRA 微调.

---

## 1. LLaMA2 架构概览

LLaMA2 = Meta 2023 年开源的 7B-70B LLM. 核心架构特点:

| 组件 | LLaMA2 选择 | 与原始 Transformer 区别 |
| :--- | :--- | :--- |
| **Norm** | RMSNorm (Pre-Norm) | 不用 LayerNorm, 计算更快 |
| **激活** | SwiGLU | 不用 ReLU/GELU, 性能更好 |
| **位置编码** | RoPE | 不用绝对位置编码 |
| **注意力** | GQA (Grouped Query Attention) | MHA 的优化版 |
| **FFN 维度** | (2/3) × 4d (而非 4d) | 调整后的隐藏维度 |
| **上下文** | 4K (LLaMA2) / 32K (LLaMA2-Long) | 旋转位置编码外推 |

---

## 2. 整体架构图

```text
输入 Token IDs (B, L)
    ↓
Token Embedding (B, L, d)
    ↓
N × Transformer Block:
    ├─ RMSNorm
    ├─ Grouped Query Attention (GQA) + RoPE
    ├─ 残差连接
    ├─ RMSNorm
    ├─ SwiGLU FFN
    └─ 残差连接
    ↓
Final RMSNorm
    ↓
LM Head (d → vocab_size) → Logits (B, L, V)
    ↓
采样 / 生成
```

---

## 3. 完整代码实现 (PyTorch)

### 3.1 RMSNorm

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # x: (B, L, d)
        # RMS(x) = sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
```

**为什么用 RMSNorm**:
- 比 LayerNorm 少一次均值计算
- 性能相近, 速度更快
- 在 LLaMA / GPT 等大模型广泛使用

### 3.2 RoPE (Rotary Position Embedding)

```python
import torch

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 的 cos/sin 频率"""
    # freqs[i] = 1 / theta^(2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    
    # 位置 0, 1, 2, ..., max_seq_len-1
    t = torch.arange(max_seq_len)
    
    # 外积: (max_seq_len, dim//2)
    freqs = torch.outer(t, freqs)
    
    # 复数: cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """对 q, k 应用 RoPE
    
    Args:
        xq: (B, H, L, d) - query
        xk: (B, H, L, d) - key
        freqs_cis: (L, d//2) 复数
    """
    # 拆成 (B, H, L, d//2, 2) -> 复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 广播: (1, 1, L, d//2) -> (B, H, L, d//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = xq_ * freqs_cis  # 复数乘法 = 旋转
    xk_out = xk_ * freqs_cis
    
    # 转回实数
    xq_out = torch.view_as_real(xq_out).flatten(-2)
    xk_out = torch.view_as_real(xk_out).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**RoPE 的核心思想**: 把位置信息编码为旋转, 应用到 query/key 上.

$$q'_i = q_i \cdot e^{i \cdot m \cdot \theta_i}$$
$$k'_i = k_i \cdot e^{i \cdot n \cdot \theta_i}$$

相对位置差: $q'_i \cdot \bar{k'}_j = q_i \cdot k_j \cdot e^{i \cdot (m-n) \cdot \theta_i}$

### 3.3 SwiGLU FFN

```python
class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit"""
    def __init__(self, dim, hidden_dim, multiple_of=4):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        # LLaMA 风格: 三个投影 (gate, up, down)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
    
    def forward(self, x):
        # SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 3.4 Grouped Query Attention (GQA)

```python
class GQA(nn.Module):
    """Grouped Query Attention: MHA 的优化版"""
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads  # 通常是 n_heads 的 1/4 或 1/8
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头被重复多少次
        
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape
        
        # 投影
        xq = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
        
        # RoPE
        xq, xk = apply_rotary_emb(xq.transpose(1,2), xk.transpose(1,2), freqs_cis)
        
        # GQA: 重复 KV 头
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        
        # 注意力
        # xq: (B, n_heads, L, head_dim)
        scores = (xq @ xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask  # causal mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ xv
        
        # 合并头
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)
```

### 3.5 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, norm_eps=1e-6):
        super().__init__()
        self.attn = GQA(dim, n_heads, n_kv_heads)
        self.ffn = SwiGLU(dim, hidden_dim)
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
    
    def forward(self, x, freqs_cis, mask=None):
        # Pre-Norm + 残差
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

### 3.6 完整 LLaMA2 模型

```python
class LLaMA2(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, 
                 hidden_dim, max_seq_len=2048, norm_eps=1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # 嵌入层
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # RoPE 频率
        freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, norm_eps)
            for _ in range(n_layers)
        ])
        
        # Final Norm + LM Head
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 权重共享 (输入嵌入和输出层)
        self.tok_embeddings.weight = self.output.weight
    
    def forward(self, tokens, targets=None):
        B, L = tokens.shape
        
        # 嵌入
        h = self.tok_embeddings(tokens)  # (B, L, dim)
        
        # 获取当前序列长度的 RoPE
        freqs_cis = self.freqs_cis[:L]
        
        # Causal mask
        mask = torch.full((L, L), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        
        # Transformer 层
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # 输出
        h = self.norm(h)
        logits = self.output(h)  # (B, L, vocab_size)
        
        # 损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


# 创建 LLaMA2-7B 配置
def llama2_7b_config():
    return dict(
        vocab_size=32000,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,      # GQA
        hidden_dim=11008,
        max_seq_len=4096,
    )

# 创建小模型 (用于实验)
def llama2_small_config():
    return dict(
        vocab_size=32000,
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        hidden_dim=1408,
        max_seq_len=2048,
    )

model = LLaMA2(**llama2_small_config())
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
# 参数量: ~30M (小模型)
```

---

## 4. Tokenizer 训练

LLaMA 使用 SentencePiece BPE:

```python
import sentencepiece as spm

# 1. 准备训练数据 (一行一句)
with open('corpus.txt', 'w') as f:
    for line in corpus:
        f.write(line + '\n')

# 2. 训练 tokenizer
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.9995,
    num_threads=8,
)

# 3. 加载
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')

# 4. 使用
tokens = sp.encode("Hello, world!")
print(tokens)
# [15339, 11, 884, 4]

decoded = sp.decode(tokens)
print(decoded)
# Hello, world!

# 5. 测试特殊字符
print(sp.encode("中文测试 🚀"))
# [12968, 23173, 4, 3, 16, 190]
```

**LLaMA2 中文支持**: LLaMA2 原生对中文支持差. 需要:
1. 中文 tokenizer (vocab + 20K 中文 token)
2. 中文词表扩展 (continued training)

详见 [all-in-rag/datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 项目.

---

## 5. 数据处理 Pipeline

### 5.1 文本清洗

```python
import re
from datasets import load_dataset

def clean_text(text):
    """基础文本清洗"""
    # 移除 HTML
    text = re.sub(r'<[^>]+>', '', text)
    # 移除 URL
    text = re.sub(r'http\S+', '', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 移除特殊字符 (保留中英文标点)
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()"\'-]', '', text)
    return text

# 加载数据
dataset = load_dataset("wikipedia", "zh", split="train", streaming=True)

# 清洗
cleaned = []
for i, example in enumerate(dataset):
    if i >= 10000:
        break
    text = clean_text(example['text'])
    if len(text) > 100:  # 过滤太短
        cleaned.append({"text": text})

print(f"清洗后样本数: {len(cleaned)}")
```

### 5.2 Tokenize + 打包

```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# Tokenize
tokenized = []
for example in cleaned:
    ids = tokenizer.encode(example['text'])
    tokenized.extend(ids)

print(f"总 token 数: {len(tokenized):,}")

# 打包为固定长度序列
seq_len = 2048
num_seqs = len(tokenized) // seq_len

# 转为 numpy (便于保存)
arr = np.array(tokenized[:num_seqs * seq_len], dtype=np.int32).reshape(num_seqs, seq_len)
print(f"训练样本数: {arr.shape}")  # (n, 2048)

# 保存
np.save('pretrain_data.npy', arr)
```

### 5.3 数据去重

```python
# 使用 SimHash 去重 (大规模高效)
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf-8'))
    return m

# 建索引
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for i, text in enumerate(corpus):
    m = create_minhash(text)
    lsh.insert(f"doc_{i}", m)

# 查询重复
m = create_minhash(query_text)
results = lsh.query(m)
# results 是相似文档 ID 列表
```

---

## 6. 预训练 (Pretraining)

### 6.1 数据准备

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    """预训练数据集"""
    def __init__(self, data_path, seq_len=2048):
        self.data = np.load(data_path)  # (N, seq_len)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].astype(np.int64))
        # 自回归: 输入 = 前 L-1 个 token, 目标 = 后 L-1 个 token
        return {
            'input_ids': x[:-1],
            'labels': x[1:]
        }

dataset = PretrainDataset('pretrain_data.npy', seq_len=2048)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
```

### 6.2 训练循环 (单 GPU)

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model = LLaMA2(**llama2_small_config()).cuda()
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=10000)

# 训练
for epoch in range(3):
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        
        logits, loss = model(input_ids, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")
    
    # 保存 checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
```

### 6.3 多卡训练 (DDP)

```bash
# 单机多卡
torchrun --nproc_per_node=4 pretrain.py

# 多机多卡
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=... pretrain.py
```

```python
# pretrain.py 片段
import torch.distributed as dist

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = LLaMA2(**config).cuda(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

---

## 7. 有监督微调 (SFT)

### 7.1 数据格式

```json
[
  {
    "instruction": "解释 Transformer 中的 self-attention 机制",
    "input": "",
    "output": "Self-attention 是 Transformer 的核心机制, 它通过计算序列中每个位置与其他位置的相关性..."
  },
  {
    "instruction": "将以下文本翻译成英文",
    "input": "今天天气很好, 适合出门游玩.",
    "output": "The weather is nice today, perfect for going out."
  }
]
```

### 7.2 数据加载

```python
from datasets import Dataset
from transformers import AutoTokenizer

def load_sft_dataset(json_path, tokenizer, max_len=2048):
    data = json.load(open(json_path))
    
    def format_example(ex):
        if ex['input']:
            prompt = f"### 指令:\n{ex['instruction']}\n\n### 输入:\n{ex['input']}\n\n### 回答:\n"
        else:
            prompt = f"### 指令:\n{ex['instruction']}\n\n### 回答:\n"
        response = ex['output']
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        
        # 拼接, 长度截断
        input_ids = (prompt_ids + response_ids)[:max_len]
        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[:max_len]
        
        return {'input_ids': input_ids, 'labels': labels}
    
    return Dataset.from_list([format_example(ex) for ex in data])
```

### 7.3 SFT 训练

```python
from transformers import Trainer, TrainingArguments

dataset = load_sft_dataset('sft_data.json', tokenizer)
training_args = TrainingArguments(
    output_dir='./sft_output',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,  # 比预训练小
    bf16=True,
    save_steps=500,
    logging_steps=50,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### 7.4 完整 Chat Template

LLaMA2 的官方对话格式:

```python
def build_chat_prompt(messages):
    """构建 LLaMA2 Chat 对话 prompt"""
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST] ", " [/INST]"
    
    prompt = f"<s>{B_SYS}You are a helpful, respectful and honest assistant.{E_SYS}"
    
    for msg in messages:
        if msg['role'] == 'user':
            prompt += f"{B_INST}{msg['content']}{E_INST}"
        elif msg['role'] == 'assistant':
            prompt += f" {msg['content']} </s><s>"
    
    return prompt
```

---

## 8. LoRA 高效微调

详见 [算法与AI/深度学习手撕/第08章-微调与对齐/第08章-01-LoRA](../深度学习手撕/第08章-微调与对齐/第08章-01-LoRA.md).

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 配置
lora_config = LoraConfig(
    r=16,                              # rank
    lora_alpha=32,                      # scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 所有线性层
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 20M || all params: 7B || trainable%: 0.28%
```

---

## 9. 模型评估

### 9.1 困惑度 (Perplexity, PPL)

```python
import torch
from torch.nn.functional import cross_entropy

def compute_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            
            logits, _ = model(input_ids, labels)
            
            # 只计算非 -100 位置的损失
            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
            
            loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            n_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

ppl = compute_perplexity(model, eval_dataloader)
print(f"Perplexity: {ppl:.2f}")
```

### 9.2 下游任务评测

```python
# 使用 lm-evaluation-harness
import lm_eval

results = lm_eval.simple_evaluate(
    model=lm_eval.api.registry.get_model("hf")("/path/to/model"),
    tasks=["hellaswag", "mmlu", "arc_easy"],
    batch_size=8,
)
print(results)
```

---

## 10. 关键公式汇总

### 10.1 RMSNorm

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

### 10.2 RoPE

$$q'_i = q_i \cdot e^{i \cdot m \cdot \theta_i}, \quad k'_i = k_i \cdot e^{i \cdot n \cdot \theta_i}$$
$$\theta_i = 10000^{-2i/d}$$

### 10.3 SwiGLU

$$\text{SwiGLU}(x) = (\text{SiLU}(x W_1) \odot x W_3) W_2$$

### 10.4 GQA

$$\text{GQA}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$
其中 $K, V$ 在组内共享 (n_kv_heads < n_heads)

### 10.5 交叉熵损失

$$\mathcal{L} = -\sum_{t} \log P(x_{t+1} | x_1, \ldots, x_t)$$

---

## 11. 推荐学习路径

1. **理论** (1 周): 读 Attention Is All You Need + LLaMA2 paper
2. **实现** (2 周): 用本附录代码搭出小模型
3. **预训练** (1 周): 在小数据集上跑通 pretraining
4. **SFT** (1 周): 在指令数据上做 SFT
5. **LoRA** (1 周): 尝试 LoRA 高效微调
6. **评估** (1 周): 用 PPL 和下游任务评估

---

## 12. 参考资源

- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) — 第 5 章 动手搭建大模型
- [meta-llama/llama](https://github.com/meta-llama/llama) — LLaMA 官方实现
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — 简洁的 C 实现
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nA) — 经典教程

### 经典论文
1. **LLaMA**: Touvron et al., 2023. *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971.
2. **LLaMA2**: Touvron et al., 2023. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288.
3. **RoPE**: Su et al., 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
4. **RMSNorm**: Zhang & Sennrich, 2019. *Root Mean Square Layer Normalization*. NeurIPS.
5. **GQA**: Ainslie et al., 2023. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv:2305.13245.
6. **SwiGLU**: Shazeer, 2020. *GLU Variants Improve Transformer*. arXiv:2002.05202.

---

> 本附录展示了从零构建 LLM 的完整代码. 配合 [算法与AI/深度学习手撕/第10章-经典模型架构/第10章-02-LLaMA完整实现](../深度学习手撕/第10章-经典模型架构/第10章-02-LLaMA完整实现.md) 阅读, 可以深入理解 LLM 实现细节.
