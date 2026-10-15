---
title: SFT 等后训练(Post-Training)教程
tags:
  - SFT
  - RLHF
  - DPO
  - GRPO
  - 奖励模型
  - 对齐
  - 后训练
  - 偏好优化
description: 从 Pretrain 到部署的最后一公里. SFT → RM → PPO → DPO/GRPO/RLAIF 全链路详解;每步的 why、公式、代码、踩坑与工业实践
last_updated: 2026-08-31
status: ✅ v1.0 - 已扩充至 2000+ 行
chapter_id: 09
---

# 第09章 SFT 等后训练(Post-Training)教程

> 后训练是把"会续写"的 base model 变成"会听话、对齐人类意图"的 assistant 的关键环节。本章讲 SFT、奖励建模、PPO/DPO/GRPO 等偏好优化算法的数学动机、工程实现与踩坑经验。

---

## 1. 后训练全景图:从 Pretrain 到部署

### 1.1 为什么需要后训练

Pretrain 后的 base model 只会"续写"。给定 prompt "中国的首都是",它可能续写为"中国的首都是哪座城市?这是许多人关心的旅游问题……"——语法通顺,但答非所问。

后训练的目标是让模型:
1. **听懂指令**(Instruction Following): 识别 prompt 是问题/任务/请求,生成对应回答
2. **对齐人类偏好**(Alignment): 回答有用、无害、诚实(RLHF 三原则)
3. **提升推理能力**(Reasoning): 通过 RL 激励 chain-of-thought 等能力
4. **领域适配**(Domain Adaptation): 注入特定领域的知识与风格

> **关键**: 后训练不是"独立环节",而是 Pretrain 与部署之间的桥梁;Pretrain 学的是"世界知识",后训练学的是"如何运用知识"。

### 1.2 后训练技术栈总览

```text
Base Model (Pretrain 完)
   ↓
Stage 1: SFT (Supervised Fine-Tuning)
   ↓
Stage 2: Reward Modeling (RM)
   ↓
Stage 3: RL Alignment (PPO / DPO / GRPO / RLAIF)
   ↓
Stage 4 (可选): Constitutional AI / Self-Improve
   ↓
Final Model (部署上线)
```

**Stage 1 - SFT**: 用人工标注的 (prompt, response) 对训练模型"模仿人类答案"。本质是有监督的 instruction tuning。

**Stage 2 - RM**: 用人类偏好数据(同一个 prompt 的两个回答 A vs B,人标注哪个更好)训练奖励模型,给出任意回答的"分数"。

**Stage 3 - RL**: 用 RM 作为奖励信号,通过强化学习(PPO、GRPO 等)进一步优化模型,使其输出 RM 评分高的回答。

**Stage 4 - 自进化**: 用模型自身生成 + 验证(数学正确性、单元测试通过等)做 RL(RLVR),无需人类偏好标注。

### 1.3 后训练 vs Pretrain 的核心差异

| 维度 | Pretrain | 后训练 |
| :--- | :--- | :--- |
| 数据量 | TB 级 token | 10K-1M 条对话 |
| 训练目标 | Next-token prediction | SFT: CE Loss;RL: PPO/DPO/GRPO |
| 算力 | 千卡 × 月 | 数十卡 × 天 |
| 学习率 | 1e-4 ~ 3e-4 | 1e-5 ~ 5e-6(避免灾难性遗忘) |
| 数据格式 | 纯文本流 | 多轮对话 + system prompt |
| 评估指标 | Loss / Perplexity | 人类偏好 / Benchmark 分数 |

### 1.4 后训练的时间线

```text
2017  Transformer (Vaswani et al.)
2020  GPT-3 + 第一版指令微调
2021  InstructGPT (OpenAI)  - 提出 SFT + RM + PPO 三阶段
2022  FLAN-T5 / T0 / 自然指令
2022  ChatGPT (OpenAI)  - 工业落地,引爆 RLHF 浪潮
2023  LLaMA-2-Chat + RLHF 开源
2023  Self-Instruct / Alpaca (数据合成)
2023  Constitutional AI (Anthropic)
2023  DPO (Direct Preference Optimization, Stanford)
2024  ORPO / KTO / SimPO (DPO 改进)
2024  GRPO (DeepSeek, 用于 R1)
2024  RLVR (RL with Verifiable Rewards)
2025  o1 / R1 (Test-time compute scaling)
2025  Self-Play / Online Iterative RL
```

### 1.5 后训练失败的常见原因

| 现象 | 根因 | 解法 |
| :--- | :--- | :--- |
| 灾难性遗忘(忘了预训练知识) | SFT 学习率太高 / epoch 太多 | 低 lr(5e-6)+ 1-2 epoch + 数据配比 |
| Reward Hacking(RM 分数高但质量差) | RM 泛化不足 / 长度偏差 | 多样化偏好数据 + 长度惩罚 |
| 模式坍缩(所有回答一个模板) | SFT 数据风格单一 | 多风格、多领域数据 |
| 安全性下降(RLHF 后反而更毒) | RM 不懂安全 | RLAIF / Constitutional AI |
| 多轮对话混乱 | SFT 单轮训练 | 多轮对话数据 |
| 工具调用失败 | 没工具调用 SFT 数据 | Tool-use 专项 SFT |

---

## 2. SFT(监督微调)详解

### 2.1 SFT 的基本形式

给定数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$,其中 $x_i$ 是 prompt,$y_i$ 是 target response,SFT 最小化交叉熵:

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^N \sum_{t=1}^{|y_i|} \log P(y_{i,t} | y_{i,<t}, x_i; \theta)$$

**关键设计选择**:
1. **只对 response 计算 loss**(prompt 部分 mask 掉),避免模型学到"复制 prompt"
2. **多轮对话格式**: 拼接成 [user1, assistant1, user2, assistant2, ...],只对 assistant 部分算 loss
3. **Padding 与 packing**: 短序列 padding 到 batch 内最大长度,或多个短序列拼成一个长序列

### 2.2 SFT 数据格式

#### 2.2.1 原始格式(对话列表)

```json
[
  {"role": "system", "content": "你是一个有帮助的助手。"},
  {"role": "user", "content": "法国的首都是?"},
  {"role": "assistant", "content": "法国的首都是巴黎。"}
]
```

#### 2.2.2 Chat Template(以 LLaMA-3 为例)

```python
# LLaMA-3 chat template
LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>"""

# 例
text = LLAMA3_TEMPLATE.format(
    system="你是一个有帮助的助手。",
    user="法国的首都是?",
    assistant="法国的首都是巴黎。"
)
# 关键: 只有 assistant 部分算 loss,其他部分用 -100 mask
```

#### 2.2.3 主流 Chat Template 对比

| 模型 | System 分隔符 | User 分隔符 | Assistant 分隔符 | EOT |
| :--- | :--- | :--- | :--- | :--- |
| **LLaMA-2** | `[INST] <<SYS>>` | `[INST]` | `[/INST]` | `</s>` |
| **LLaMA-3** | `<\|start_header_id\|>system<\|end_header_id\|>` | `<\|start_header_id\|>user<\|end_header_id\|>` | `<\|start_header_id\|>assistant<\|end_header_id\|>` | `<\|eot_id\|>` |
| **ChatGLM** | 无 | `[Round N]` | `` ` `` | `` ` `` |
| **Qwen** | `<\|im_start\|>system\n` | `<\|im_start\|>user\n` | `<\|im_start\|>assistant\n` | `<\|im_end\|>` |
| **Mistral** | 无固定 | `[INST]` | `[/INST]` | `</s>` |

### 2.3 Loss Masking 的正确实现

**错误实现**(对所有 token 算 loss,包括 prompt):
```python
# 错: prompt 部分也算了 loss,模型学到"复制 prompt"
loss = F.cross_entropy(logits, labels)
```

**正确实现**(只对 response 部分算 loss):
```python
def mask_labels(input_ids, response_token_ranges, pad_token_id=-100):
    """
    response_token_ranges: List[List[int]], 每个对话中 response token 的起止位置
    """
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100  # padding 不算 loss

    # 默认: 所有 token 都 mask
    labels[:] = -100
    # 只对 response 区间算 loss
    for batch_idx, ranges in enumerate(response_token_ranges):
        for start, end in ranges:
            labels[batch_idx, start:end] = input_ids[batch_idx, start:end]
    return labels
```

**更通用的实现**(用 chat template 的 token mask):
```python
def tokenize_with_loss_mask(messages, tokenizer):
    """
    messages: List[Dict], e.g. [{"role":"user","content":"..."}, ...]
    返回: input_ids, labels (assistant 部分保留,其他 -100)
    """
    input_ids = []
    labels = []
    for msg in messages:
        if msg["role"] == "assistant":
            # 渲染 assistant header + content
            content = msg["content"] + tokenizer.eos_token
            tokens = tokenizer.encode(content, add_special_tokens=False)
            input_ids.extend(tokens)
            labels.extend(tokens)  # 保留 label
        else:
            # user / system: 渲染 header, mask 掉 content
            prefix_tokens = tokenizer.encode(
                f"<{msg['role']}>\n", add_special_tokens=False
            )
            input_ids.extend(prefix_tokens)
            labels.extend([-100] * len(prefix_tokens))  # mask

            content_tokens = tokenizer.encode(
                msg["content"] + tokenizer.eos_token, add_special_tokens=False
            )
            input_ids.extend(content_tokens)
            labels.extend([-100] * len(content_tokens))
    return input_ids, labels
```

### 2.4 数据配比(Data Mixing)

SFT 数据通常混合多个来源,配比决定模型能力倾向:

| 数据源 | 比例建议 | 作用 |
| :--- | :--- | :--- |
| **通用对话**(ShareGPT, UltraChat) | 50-70% | 通用助手能力 |
| **指令数据**(Alpaca, FLAN) | 10-20% | 多任务指令 |
| **代码数据**(CodeAlpaca, Magicoder) | 5-15% | 代码生成 |
| **数学数据**(MathInstruct, MetaMath) | 5-10% | 数学推理 |
| **中文数据**(COIG, Firefly) | 5-20%(中文 LLM) | 中文能力 |
| **领域数据**(法律/医疗/金融) | 0-100%(专用) | 垂直能力 |
| **安全数据**(Constitutional) | 2-5% | 安全性 |

**LIMA 论文**(2023)核心观点: 1000 条高质量数据 > 100K 条低质量数据。但实际工业落地仍需要 50K-100K+ 数据才能鲁棒。

### 2.5 数据质量过滤

#### 2.5.1 规则过滤

```python
def rule_based_filter(example):
    """基础质量过滤"""
    # 长度过滤
    if len(example["response"]) < 50:
        return False
    if len(example["response"]) > 4000:
        return False  # 过长可能质量差
    # 比例过滤
    if len(example["response"]) < len(example["prompt"]) * 0.3:
        return False  # 回答太短,可能是敷衍
    # 重复过滤
    words = example["response"].split()
    if len(set(words)) / max(len(words), 1) < 0.3:
        return False  # 词汇重复率过高
    # 关键词过滤
    blacklist = ["作为 AI", "I cannot", "I'm sorry"]
    for kw in blacklist:
        if example["response"].lower().startswith(kw.lower()):
            return False  # 模板化回答
    return True
```

#### 2.5.2 模型打分过滤(基于奖励模型或 perplexity)

```python
def score_based_filter(examples, reward_model, threshold=5.0):
    """用奖励模型打分,只保留分数高于阈值的数据"""
    scored = []
    for ex in examples:
        score = reward_model.score(ex["prompt"], ex["response"])
        if score > threshold:
            scored.append(ex)
    return scored
```

#### 2.5.3 去重(Semantic Deduplication)

用 embedding 模型(如 BGE、Snowflake)对 (prompt, response) 编码,用 FAISS 找近邻,保留相似度低于阈值的样本。

### 2.6 学习率与 Epoch

| 模型规模 | Peak LR | Warmup Steps | Epoch | 总 Token 量 |
| :--- | :--- | :--- | :--- | :--- |
| **7B** | 2e-5 | 50-100 | 2-3 | ~1B tokens |
| **13B** | 1e-5 | 50-100 | 2-3 | ~2B tokens |
| **70B** | 5e-6 | 50-100 | 2-3 | ~4B tokens |
| **>100B** | 2e-6 | 100-200 | 1-2 | ~5B tokens |

**关键**: SFT 学习率比 Pretrain 低 10-100 倍。常用 warmup 5-10% 总 step,cosine decay 到 10% peak。

```python
# LLaMA-Factory 风格的学习率调度
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.05 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
```

### 2.7 Sequence Packing(序列打包)

传统做法: padding 到 batch 内最长序列,大量算力浪费在 padding token 上。

**Packing**: 把多个短序列拼成一个长序列(通常 2048 / 4096),中间用 EOS 隔开。

```python
def pack_sequences(examples, max_length=4096, eos_token_id=2):
    """
    把多个短样本拼成长序列
    输入: examples = [{"input_ids": [...], "labels": [...]}, ...]
    输出: List[{"input_ids": [...], "labels": [...]}] 每个长度 ≤ max_length
    """
    packed = []
    current_input = []
    current_labels = []
    for ex in examples:
        # 加 EOS 分隔
        ids = ex["input_ids"] + [eos_token_id]
        labels = ex["labels"] + [-100]  # EOS 也不算 loss
        if len(current_input) + len(ids) > max_length:
            packed.append({"input_ids": current_input, "labels": current_labels})
            current_input = []
            current_labels = []
        current_input.extend(ids)
        current_labels.extend(labels)
    if current_input:
        packed.append({"input_ids": current_input, "labels": current_labels})
    return packed

# 收益: 训练吞吐提升 2-5x(取决于数据平均长度)
```

**FlashAttention 兼容**: packing 后需要用 `cu_seqlens` 标记各样本边界,避免 attention 跨样本"看到"其他样本。

### 2.8 LoRA SFT(参数高效微调)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # 通常 = 2 * r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.0623

# 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # LoRA lr 比 full FT 高 10x
        num_train_epochs=3,
        bf16=True,
    ),
)
trainer.train()
```

**QLoRA**(NF4 量化 + LoRA): 4-bit 量化 + LoRA,单卡 48GB 可微调 65B 模型。

### 2.9 SFT 训练完整代码(LLaMA-Factory 风格)

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 1. 加载模型与 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. 配置 LoRA
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# 3. 加载数据
dataset = load_dataset("json", data_files="sft_data.json", split="train")

def format_and_tokenize(example):
    """应用 chat template,生成 labels with mask"""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    # 找到 assistant 内容的起始位置
    prompt_messages = messages[:-1]
    prompt_len = len(tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    ))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "labels": labels}

dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

# 4. 配置训练参数
training_args = TrainingArguments(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    gradient_checkpointing=True,  # 显存不够时开启
    optim="adamw_torch",
    report_to="wandb",
)

# 5. 训练
model = get_peft_model(model, peft_config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```

---

## 3. RLHF 三阶段:SFT → RM → PPO

### 3.1 RLHF 的动机

SFT 后模型能模仿,但不能保证"有用、无害、诚实"。SFT 数据只展示了"好回答",没展示"差回答";模型不知道差回答"差在哪"。

RLHF 用人类偏好数据训练奖励模型(RM),再用 RM 作为奖励信号训练 LLM。

### 3.2 阶段一:Reward Model(RM)训练

#### 3.2.1 数据形式

偏好数据形式:
```json
{
  "prompt": "法国的首都是?",
  "chosen": "法国的首都是巴黎。",
  "rejected": "法国的首都是伦敦。"
}
```

#### 3.2.2 Bradley-Terry 模型

假设人类偏好服从 BT 模型:对 prompt $x$,两个回答 $y_w$(winner) 和 $y_l$(loser):

$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

其中 $r(\cdot, \cdot)$ 是奖励模型。RM 训练目标是最小化负对数似然:

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r(x, y_w) - r(x, y_l))$$

#### 3.2.3 RM 实现代码

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class RewardModel(nn.Module):
    """基于 LLM 的奖励模型
    用最后一个 token 的 hidden state 预测标量奖励
    """
    def __init__(self, base_model_path):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_path)
        hidden_size = self.backbone.config.hidden_size
        # 标量奖励 head
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 取最后一个 token 的 hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        # 用最后一个非 padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_token_hidden = last_hidden[batch_idx, sequence_lengths]  # [B, H]
        reward = self.reward_head(last_token_hidden).squeeze(-1)  # [B]
        return reward


def compute_rm_loss(rm_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    """RM 训练损失"""
    reward_chosen = rm_model(chosen_ids, chosen_mask)
    reward_rejected = rm_model(rejected_ids, rejected_mask)
    # Bradley-Terry 损失
    loss = -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()
    # 准确率(可选监控)
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return loss, accuracy

# 训练
optimizer = torch.optim.AdamW(rm_model.parameters(), lr=1e-5)
for step, batch in enumerate(train_loader):
    loss, acc = compute_rm_loss(rm_model, **batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, loss={loss.item():.4f}, acc={acc.item():.4f}")
```

#### 3.2.4 RM 训练数据规模

| 数据规模 | RM 效果 | 适用 |
| :--- | :--- | :--- |
| 1K-10K | 弱,容易过拟合 | 仅 demo |
| 10K-100K | 中等 | 小规模生产 |
| 100K-1M | 强 | 大模型生产 |
| >1M | 接近 SOTA | GPT-4 级别 |

数据来源: 人工标注(Anthropic HH-RLHF, OpenAI TL;DR)、合成数据(LMSYS Chatbot Arena)。

### 3.3 阶段三:PPO 训练

#### 3.3.1 PPO 在 RLHF 中的目标函数

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{(x, y) \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)) \right]$$

其中:
- $r_\phi(x, y)$ 是 RM 评分
- $\pi_\theta$ 是当前 LLM 策略
- $\pi_{\text{ref}}$ 是 SFT 后的 reference 模型(frozen)
- $\beta$ 是 KL 系数(平衡"压榨 RM"与"不偏离 SFT")
- KL 散度作为正则,防止模型为了高奖励而"胡说八道"(reward hacking)

#### 3.3.2 PPO 的具体实现细节

PPO 用 clipped surrogate objective:

$$\mathcal{L}_{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$,$\hat{A}_t$ 是 advantage。

在 RLHF 中:
- 状态 $s_t = (x, y_{<t})$
- 动作 $a_t = y_t$ (下一个 token)
- 奖励 $r_t$ 仅在序列末尾($t = T$)非零,等于 RM 分数减去 KL 惩罚
- Advantage 用 GAE 估计

#### 3.3.3 完整 PPO 训练循环

```python
import torch
from torch.optim import AdamW

class PPOConfig:
    learning_rate: float = 1e-6
    batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4  # 每批数据训 PPO 4 次
    clip_range: float = 0.2
    kl_coef: float = 0.05  # β
    gamma: float = 1.0  # 单回合,无折扣
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

def compute_advantages(rewards, values, next_values, gamma, gae_lambda):
    """GAE advantage 估计
    rewards: [B, T]
    values: [B, T] (critic 对每步的 value 估计)
    next_values: [B, T] (下一步的 value)
    """
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    T = rewards.size(1)
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = next_values[:, t]
        else:
            next_value = values[:, t+1]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
        advantages[:, t] = last_gae_lam
    returns = advantages + values
    return advantages, returns


def ppo_train_step(policy_model, ref_model, value_model, rm_model,
                   prompts, responses, config):
    """单次 PPO 训练步骤"""
    # 1. 计算 log_probs(用当前 policy)
    input_ids = torch.cat([prompts, responses], dim=1)
    prompt_len = prompts.size(1)

    with torch.no_grad():
        # Reference policy 的 log_probs(frozen)
        ref_logits = ref_model(input_ids).logits
        ref_logps = compute_log_probs(ref_logits, responses, prompt_len)

        # Critic 的 value 估计
        values = value_model(input_ids, attention_mask)  # [B, T]
        # 取 response 部分
        values = values[:, prompt_len-1:-1]  # [B, response_len]

        # RM 分数
        rewards = torch.zeros(input_ids.size(0), input_ids.size(1) - prompt_len)
        # 仅在最后一个 token 给 RM 分数
        rewards[:, -1] = rm_model.score(input_ids, attention_mask)
        # KL 惩罚(per token)
        kl_per_token = compute_kl(logits, ref_logits, responses, prompt_len)
        rewards = rewards - config.kl_coef * kl_per_token

    # 2. PPO epochs
    for ppo_epoch in range(config.ppo_epochs):
        # 当前 policy 的 log_probs
        logits = policy_model(input_ids).logits
        logps = compute_log_probs(logits, responses, prompt_len)
        ratio = torch.exp(logps - ref_logps)  # π_θ / π_ref (重要: PPO 用 ref 作 old policy 的近似)

        # 重新计算 value
        new_values = value_model(input_ids, attention_mask)
        new_values = new_values[:, prompt_len-1:-1]

        # Advantage
        advantages, returns = compute_advantages(
            rewards, values, new_values, config.gamma, config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (clipped)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = ((new_values - returns) ** 2).mean()

        # 总损失
        loss = policy_loss + config.value_coef * value_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
        optimizer.step()

    return loss.item()
```

#### 3.3.4 PPO 训练所需的"四模型"

PPO 训练需同时加载四个模型:

| 模型 | 大小 | 作用 |
| :--- | :--- | :--- |
| **Policy Model** ($\pi_\theta$) | 7B | 训练目标 |
| **Reference Model** ($\pi_{\text{ref}}$) | 7B | 计算 KL 锚点 |
| **Reward Model** ($r_\phi$) | 7B | 计算奖励 |
| **Value Model** (Critic) | 7B | 估计 advantage |

**显存挑战**: 7B 模型 4 个,FP16 占 ~56GB;还需 Adam 状态(~56GB),优化器(~28GB),总计 ~140GB。需多卡并行 + LoRA + gradient checkpointing。

### 3.4 PPO 的工业落地难题

#### 3.4.1 显存压力

```text
7B PPO 显存占用 (FP16):
  Policy:  14 GB (权重) + 28 GB (Adam) + 14 GB (梯度) = 56 GB
  Ref:     14 GB (推理)                                 = 14 GB
  RM:      14 GB                                         = 14 GB
  Critic:  14 GB (权重) + 28 GB (Adam) + 14 GB (梯度)   = 56 GB
  Activations + KV Cache: ~10 GB
  ─────────────────────────────────────────
  Total: ~150 GB (单卡装不下!)

解决方案:
  - LoRA: Policy + Critic 用 LoRA,显存降 90%
  - 多卡拆分: Policy 在 8 卡, RM 在 4 卡, Ref 在 4 卡
  - ZeRO-3 + CPU offload
  - 8-bit Adam
```

#### 3.4.2 Reward Hacking

模型学到"骗 RM"的策略:
- 长度偏差: 训练出"越长越好",模型生成冗长内容
- 风格偏差: 训练出"用某些关键词得分高"
- 重复偏差: 训练出"重复关键句得分高"

**缓解方案**:
```python
# 1. 长度归一化(避免长度偏差)
reward = rm_score / (response_length ** alpha)
# alpha = 0.5 是常用值

# 2. KL 系数动态调整
# 监控 KL 散度,KL 过大时增加 β
def adaptive_kl_coef(current_kl, target_kl=10.0, beta=0.05):
    if current_kl > 1.5 * target_kl:
        return beta * 1.5
    elif current_kl < 0.5 * target_kl:
        return beta * 0.5
    return beta

# 3. Ensemble RM: 用多个 RM 取平均,降低单 RM 偏置
reward = 0.4 * rm1.score(...) + 0.3 * rm2.score(...) + 0.3 * rm3.score(...)
```

#### 3.4.3 KL 散度的精确估计

$k_3$ 估计器(MC 低方差):

$$\text{KL}_3(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{1}{T} \sum_t \left[ \frac{\pi_\theta(y_t|y_{<t})}{\pi_{\text{ref}}(y_t|y_{<t})} - \log \frac{\pi_\theta(y_t|y_{<t})}{\pi_{\text{ref}}(y_t|y_{<t})} - 1 \right]$$

```python
def k3_estimator(log_ratio):
    """无偏 KL 估计"""
    return (log_ratio.exp() - log_ratio - 1).mean()
```

### 3.5 PPO 训练监控

关键指标:

| 指标 | 含义 | 期望趋势 |
| :--- | :--- | :--- |
| `reward_mean` | 平均 RM 分数 | 上升 |
| `reward_std` | 奖励方差 | 略下降 |
| `kl_mean` | 与 ref 的 KL | 小幅上升(目标 <10) |
| `response_length` | 平均生成长度 | 略上升(注意 hacking) |
| `policy_ratio` | π_θ / π_old | 接近 1 |
| `clip_fraction` | 被 clip 的 token 比例 | 5-15% |
| `value_loss` | Critic loss | 下降 |
| `entropy` | 策略熵 | 略下降 |

```python
# 训练时打印
print(f"step={step} reward={reward_mean:.3f}±{reward_std:.3f} "
      f"kl={kl:.3f} len={length:.1f} clip={clip_frac:.2%}")
```

### 3.6 PPO 的超参表(经验值)

| 超参 | 范围 | 典型值 | 影响 |
| :--- | :--- | :--- | :--- |
| **kl_coef** (β) | 0.01-0.2 | 0.05 | 越大越保守 |
| **clip_range** (ε) | 0.1-0.3 | 0.2 | 越大步长越大 |
| **gae_lambda** | 0.9-0.99 | 0.95 | 越大越接近 MC |
| **gamma** | 0.95-1.0 | 1.0 | RLHF 是单回合 |
| **ppo_epochs** | 1-4 | 4 | 越大越快但容易崩 |
| **mini_batch_size** | 4-32 | 8-16 | 显存决定 |
| **value_coef** | 0.4-0.6 | 0.5 | Critic loss 权重 |
| **max_grad_norm** | 0.5-2.0 | 1.0 | 梯度裁剪 |

---

## 4. DPO(直接偏好优化)

### 4.1 DPO 的核心思想

**跳过 RM 训练**,直接在偏好数据上优化 LLM。

DPO 的关键推导: PPO 的最优策略有闭式解!

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

反过来,给定策略 $\pi$,可以反解出隐含奖励:

$$r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

代入 BT 损失:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

### 4.2 DPO 损失代码

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """
    policy_chosen_logps: π_θ(y_w|x) 的对数概率, [B]
    policy_rejected_logps: π_θ(y_l|x) 的对数概率, [B]
    ref_chosen_logps: π_ref(y_w|x) 的对数概率, [B]
    ref_rejected_logps: π_ref(y_l|x) 的对数概率, [B]
    """
    # 隐含奖励差
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)

    # DPO 损失
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    # 准确率(监控)
    accuracy = (logits > 0).float().mean()
    return loss, accuracy


def get_logps(model, input_ids, attention_mask, labels):
    """计算 (prompt, response) 对中 response 的对数概率"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # Shift: 预测下一个 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # CE loss 自动对 label=-100 做 mask
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )  # [B * T]

    per_token_loss = per_token_loss.view(shift_labels.size(0), shift_labels.size(1))
    # 序列级 log-prob = 各 token loss 之和(取负)
    valid_mask = (shift_labels != -100).float()
    seq_logps = -(per_token_loss * valid_mask).sum(dim=-1)  # [B]
    return seq_logps
```

### 4.3 DPO 训练循环(trl 风格)

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,  # DPO lr 比 SFT 还低
    num_train_epochs=1,  # 通常 1 epoch
    beta=0.1,  # KL 系数
    max_length=2048,
    max_prompt_length=1024,
    bf16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    loss_type="sigmoid",  # 标准 DPO
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=config,
    train_dataset=dpo_dataset,  # 包含 "prompt", "chosen", "rejected"
    tokenizer=tokenizer,
)
trainer.train()
```

### 4.4 DPO 的优缺点

#### 优点
1. **无需 RM**: 省去 RM 训练 + PPO 复杂流程
2. **稳定**: 单阶段训练,无 RL 训练的不稳定
3. **显存友好**: 只需 2 个模型(policy + ref),不用 critic
4. **效果**: 在多个 benchmark 上接近 PPO

#### 缺点
1. **数据效率低**: 需要更多偏好数据才能达到 RM+PPO 的效果
2. **不灵活**: 无法注入规则化奖励(如数学正确性)
3. **隐式奖励偏差**: 隐式奖励有"长度偏差"等偏置
4. **不擅长推理**: 对"会做数学题"这类需要 verifier 的任务效果差

### 4.5 DPO 的衍生算法

| 算法 | 提出 | 核心改进 | 公式变化 |
| :--- | :--- | :--- | :--- |
| **DPO** | Rafailov 2023 | 基础 | 标准 sigmoid loss |
| **IPO** | Azar 2023 | 防止 overfitting | $\text{loss} = (\log \frac{\pi_w}{\pi_l} - \frac{1}{2\beta})^2$ |
| **KTO** | Ethayarajh 2024 | 去掉配对,单条数据 | 用 Kahneman-Tversky 效用理论 |
| **ORPO** | Hong 2024 | SFT + DPO 一体化 | $\text{loss} = \text{SFT} - \lambda \log \sigma(\text{odds ratio})$ |
| **SimPO** | Meng 2024 | 长度归一 + 无 reference | $\text{loss} = -\log \sigma(\frac{\beta}{|y_w|} \log \pi_w - \frac{\beta}{|y_l|} \log \pi_l)$ |
| **CPO** | Xu 2024 | 加 SFT 损失 | $\text{loss} = \text{DPO} + \text{SFT\_CE}$ |
| **RPO** | Pang 2024 | 加 NLL | 类似 CPO |

### 4.6 DPO 训练常见踩坑

| 现象 | 根因 | 解法 |
| :--- | :--- | :--- |
| 训练 loss 很低但效果差 | reward hacking(长度偏差) | 长度归一化 / SimPO |
| chosen 概率持续下降 | DPO 的"reference drift" | 减小 lr, 增大 β |
| chosen/rejected 概率都下降 | 模型整体退化了 | 检查数据质量 / 重启 |
| 训练不稳定(loss 抖动) | lr 太高 | 1e-7 ~ 5e-7 |
| 偏好反转(模型偏好变差) | chosen/rejected 区分度太低 | 数据清洗 |

---

## 5. GRPO(组内归一化策略优化)

### 5.1 GRPO 的动机

PPO 需要 critic(价值模型),显存开销大。GRPO(DeepSeek, 2024)去掉 critic,用"组内相对奖励"做基线。

### 5.2 GRPO 的核心思想

对每个 prompt $x$,采样一组回答 $\{y_1, ..., y_K\}$:

1. 用 RM 给每个回答打分 $r_1, ..., r_K$
2. 计算组内相对优势:

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_K)}{\text{std}(r_1, ..., r_K)}$$

3. 用 PPO-like 目标优化,但用 $A_i$ 而非 critic 估计的 advantage

### 5.3 GRPO 损失

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_i \left[ \min\left( \rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}$。

### 5.4 GRPO 实现代码

```python
class GRPOTrainer:
    """Group Relative Policy Optimization (DeepSeek 风格)
    对每个 prompt 采样 K 个回答,用组内归一化作为 advantage
    """
    def __init__(self, model, ref_model, rm_model, config):
        self.model = model
        self.ref_model = ref_model
        self.rm_model = rm_model
        self.config = config

    def compute_group_advantages(self, rewards):
        """rewards: [B, K] - 每个 prompt 的 K 个回答的奖励"""
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True) + 1e-8
        advantages = (rewards - mean) / std
        return advantages  # [B, K]

    def train_step(self, prompts, responses_list, old_logps_list):
        """
        prompts: [B, P]
        responses_list: List of K responses per prompt, [B, K, R]
        old_logps_list: 老策略的 log-probs, [B, K, R]
        """
        B, K = len(responses_list), len(responses_list[0])
        R = responses_list[0][0].size(-1)

        # 1. 用 RM 给每个回答打分
        rewards = []
        for i in range(B):
            for k in range(K):
                response = responses_list[i][k]
                full_seq = torch.cat([prompts[i], response])
                score = self.rm_model.score(full_seq)
                rewards.append(score.item())
        rewards = torch.tensor(rewards).view(B, K)  # [B, K]

        # 2. 计算组内 advantages
        advantages = self.compute_group_advantages(rewards)  # [B, K]
        # 展平用于训练
        advantages_flat = advantages.view(-1)  # [B*K]

        # 3. PPO-style 损失
        all_logps = []
        all_old_logps = []
        for i in range(B):
            for k in range(K):
                response = responses_list[i][k]
                full_seq = torch.cat([prompts[i], response])
                logp = compute_logps(self.model, full_seq, response)
                all_logps.append(logp)
                all_old_logps.append(old_logps_list[i][k])

        all_logps = torch.stack(all_logps)  # [B*K]
        all_old_logps = torch.stack(all_old_logps)  # [B*K]

        # 4. PPO clipped objective
        ratio = torch.exp(all_logps - all_old_logps)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range,
                            1 + self.config.clip_range) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. KL 惩罚
        with torch.no_grad():
            ref_logps = []
            for i in range(B):
                for k in range(K):
                    full_seq = torch.cat([prompts[i], responses_list[i][k]])
                    ref_logp = compute_logps(self.ref_model, full_seq, responses_list[i][k])
                    ref_logps.append(ref_logp)
            ref_logps = torch.stack(ref_logps)
        kl_penalty = (all_logps - ref_logps).mean()
        loss = policy_loss + self.config.kl_coef * kl_penalty

        # 6. 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_penalty.item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
        }
```

### 5.5 GRPO 在数学推理上的应用

DeepSeek-R1 用 GRPO 训练数学推理能力。奖励信号分两部分:

1. **正确性奖励**: 模型生成的最终答案是否正确(用数学验证器)
2. **格式奖励**: 输出是否在 `<think>...</think><answer>...</answer>` 格式内

```python
def math_reward_function(prompt, response, ground_truth):
    """数学题的双重奖励"""
    # 1. 提取最终答案
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        return -1.0  # 格式错误
    answer = answer_match.group(1).strip()

    # 2. 检查正确性
    correct = check_math_answer(answer, ground_truth)

    # 3. 检查格式(有 <think>)
    has_think = '<think>' in response and '</think>' in response

    # 4. 组合奖励
    if correct and has_think:
        return 1.0
    elif correct:
        return 0.5  # 对但格式不对
    elif has_think:
        return 0.1  # 格式对但错(鼓励思考过程)
    else:
        return -0.5
```

### 5.6 GRPO vs PPO 对比

| 维度 | PPO | GRPO |
| :--- | :--- | :--- |
| Critic | 需要 | 不需要 |
| 显存 | 高(4 模型) | 中(3 模型) |
| Advantage 估计 | GAE + Critic | 组内归一化 |
| 数据效率 | 高 | 中(需要采样 K 个回答) |
| 适用场景 | 通用 RLHF | 推理 / 可验证任务 |

---

## 6. RLAIF 与 Constitutional AI

### 6.1 RLAIF 的动机

RLHF 的人类偏好标注:
- 成本高(每条 $0.5-$5)
- 速度慢(标注员有限)
- 难规模化

RLAIF(RL from AI Feedback)用 LLM 替代人类标注偏好。

### 6.2 Constitutional AI(Anthropic 2022)

两阶段:

**Stage A - Supervised**:
1. LLM 生成回答
2. LLM 根据宪法(Constitution)原则自我批评
3. LLM 根据批评改写回答
4. 用改写后的回答训练 SFT

**Stage B - RL**:
1. LLM 生成两个回答
2. LLM 根据宪法判断哪个更符合原则
3. 用这些 AI 偏好训练 RM
4. 用 RM 训练 PPO

### 6.3 Constitution 示例

```text
原则 1: 请选择更安全、更无害的回答
原则 2: 请选择更诚实、不捏造事实的回答
原则 3: 请选择更尊重用户、不居高临下的回答
原则 4: 请选择不包含有害偏见、歧视的回答
原则 5: 请选择更具体、更有帮助的回答
```

### 6.4 RLAIF 实现

```python
def ai_judge(prompt, response_a, response_b, constitution):
    """用 LLM 做偏好判断"""
    judge_prompt = f"""你是一个 AI 偏好评估员。请根据以下原则判断两个回答哪个更好。

原则:
{constitution}

用户问题:
{prompt}

回答 A:
{response_a}

回答 B:
{response_b}

请先简要分析两个回答,然后输出 A 或 B 表示哪个更好。"""
    response = llm.generate(judge_prompt)
    if "回答 B" in response or "B 更好" in response:
        return "B"
    return "A"


def build_ai_preference_dataset(prompts, constitution, llm):
    """构建 AI 偏好数据集"""
    dataset = []
    for prompt in tqdm(prompts):
        # 采样两个回答
        response_a = llm.generate(prompt, temperature=0.7)
        response_b = llm.generate(prompt, temperature=0.9)
        # AI 判断偏好
        preferred = ai_judge(prompt, response_a, response_b, constitution)
        if preferred == "A":
            chosen, rejected = response_a, response_b
        else:
            chosen, rejected = response_b, response_a
        dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return dataset
```

### 6.5 RLAIF 的优劣

**优势**:
- 成本低(无需人类标注)
- 速度快(可批量生成)
- 可解释(LLM 给出判断理由)
- 可规模化(数据量无上限)

**劣势**:
- AI 偏见会被放大
- 难处理主观问题
- "对齐税"(Alignment Tax): RLHF 后通用能力下降

---

## 7. Reward Model 设计进阶

### 7.1 Reward Hacking 的常见形式

| Hacking 类型 | 现象 | 缓解 |
| :--- | :--- | :--- |
| **长度偏差** | 模型生成越长越好 | 长度归一 / 长度惩罚 |
| **格式偏差** | 模型偏爱 Markdown 表格 | 多样化训练数据 |
| **重复偏差** | 模型重复关键词 | 重复惩罚 |
| **风格偏差** | 模型偏好某种口吻 | 风格归一 |
| **幻觉奖励** | 模型学会"自信地胡说" | 加 factuality reward |

### 7.2 多维度 RM

用多个 RM 评估不同维度:

```python
class MultiRewardModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_path)
        # 多个 reward head
        self.helpfulness_head = nn.Linear(hidden_size, 1)
        self.harmlessness_head = nn.Linear(hidden_size, 1)
        self.honesty_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        last_hidden = hidden[:, -1, :]  # 最后一个 token
        return {
            "helpfulness": self.helpfulness_head(last_hidden).squeeze(-1),
            "harmlessness": self.harmlessness_head(last_hidden).squeeze(-1),
            "honesty": self.honesty_head(last_hidden).squeeze(-1),
        }

# 训练:每个 head 用对应维度的偏好数据
# 推理:综合奖励 = w1 * helpful + w2 * harmless + w3 * honest
```

### 7.3 Generative RM(LLM-as-Judge)

用 LLM 自身做奖励模型。代表:CLoud(Anthropic 2024),GPTRater。

```python
def generative_rm_score(prompt, response, judge_prompt_template):
    judge_prompt = judge_prompt_template.format(
        prompt=prompt, response=response
    )
    # 用 LLM 生成评分
    output = judge_llm.generate(judge_prompt)
    # 解析分数(0-10)
    score_match = re.search(r'\b([0-9]|10)\b', output)
    if score_match:
        return float(score_match.group(1)) / 10.0
    return 0.5
```

**优势**: 可解释(LLM 给出 reasoning)、少样本学习能力强。
**劣势**: 慢(自回归生成)、成本高(每条要 generate)。

### 7.4 Process Reward Model(PRM)

对推理的每一步打分,而非仅对最终答案。

```python
def prm_step_score(prompt, reasoning_steps, final_answer, ground_truth):
    """对推理链的每一步打分"""
    scores = []
    for i, step in enumerate(reasoning_steps):
        # 用 PRM 给该步骤打分
        step_score = prm_model.score_step(prompt, reasoning_steps[:i+1])
        scores.append(step_score)
    # 最终答案正确性奖励
    final_correct = check_answer(final_answer, ground_truth)
    if final_correct:
        scores.append(1.0)
    else:
        scores.append(-1.0)
    return scores
```

**应用**: math/MATH, o1 的 process reward。

---

## 8. RLVR(可验证奖励的强化学习)

### 8.1 RLVR 的动机

很多任务有"自动验证器":数学(答案对错)、代码(单测通过)、定理(证明可检查)。无需 RM,直接用 verifier 的 0/1 奖励做 RL。

### 8.2 RLVR 的典型应用

| 任务 | 验证器 | 奖励 |
| :--- | :--- | :--- |
| **数学** | sympy / Wolfram | 答案对错(0/1) |
| **代码** | 单元测试 | 通过用例比例 |
| **定理证明** | Lean / Coq | 证明完成 |
| **SQL** | 数据库执行 | 查询结果正确 |
| **化学** | RDKit | 分子式有效 |
| **指令遵循** | 规则匹配 | 满足约束 |

### 8.3 RLVR 训练循环

```python
class RLVRDataset:
    def __init__(self, problems, verifier):
        self.problems = problems
        self.verifier = verifier

    def get_reward(self, prompt, response):
        # 调用 verifier
        try:
            correct = self.verifier(response)
            return 1.0 if correct else 0.0
        except Exception:
            return -1.0  # 输出格式错误


def rlvr_train_step(model, ref_model, problems, config):
    """RLVR 单步训练"""
    # 1. 采样多个回答
    K = 8  # 每个问题采样 8 个
    prompts = [p["prompt"] for p in problems]
    responses_list = [model.sample(p, n=K, temperature=0.8) for p in prompts]

    # 2. 用 verifier 给奖励
    rewards = []
    for p, responses in zip(problems, responses_list):
        r = [RLVRDataset(p, p["verifier"]).get_reward(p["prompt"], resp)
             for resp in responses]
        rewards.append(r)
    rewards = torch.tensor(rewards)  # [B, K]

    # 3. GRPO 训练
    advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / \
                 (rewards.std(dim=-1, keepdim=True) + 1e-8)

    # ... (同 GRPO)
```

### 8.4 RLVR vs RLHF 对比

| 维度 | RLHF | RLVR |
| :--- | :--- | :--- |
| 奖励来源 | 人类偏好 / RM | 验证器(规则) |
| 数据标注 | 昂贵 | 免费(自动) |
| 适用任务 | 通用对话 | 可验证任务 |
| 奖励噪声 | 高 | 低(0/1 准确) |
| 训练稳定性 | 差 | 好 |

### 8.5 RLVR 的局限

- 仅适用于有 verifier 的任务(数学/代码/定理等)
- Verifier 可能被 hacking(找到 hack 答案)
- 仅 0/1 奖励可能信息不足,可加格式奖励

---

## 9. Online Iterative RL(在线迭代强化学习)

### 9.1 Online RL 的动机

传统 RLHF 是 offline:用静态数据集训。但模型在训练中会"用完"数据集的信号,无法继续提升。

**Online RL**: 模型在训练中持续生成新回答,RM 实时打分,持续学习。

### 9.2 Online DPO / Iterative DPO

```python
def iterative_dpo_step(model, ref_model, rm_model, prompts, n_iters=3):
    """Iterative DPO: 多个 DPO 轮次,每轮用最新模型生成 rejected"""
    for iter in range(n_iters):
        # 1. 用当前模型采样回答
        candidates = []
        for prompt in prompts:
            chosen = get_human_chosen_response(prompt)  # 或用 SFT 数据
            rejected = model.generate(prompt, temperature=0.9)
            candidates.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        # 2. 用 RM 过滤掉低质量 rejected
        filtered = [c for c in candidates if rm_model.score(c["prompt"], c["rejected"]) > threshold]
        # 3. DPO 训练
        dpo_train(model, ref_model, filtered)
    return model
```

代表:**Self-Rewarding LM**(Meta 2024)、**RLHF-V / Iterative DPO**。

### 9.3 Online PPO 的工程挑战

```python
class OnlinePPOLoop:
    def __init__(self, model, ref_model, rm_model, prompts):
        self.model = model
        self.ref_model = ref_model
        self.rm_model = rm_model
        self.prompts = prompts

    def step(self, batch_size=64):
        """单步 online PPO"""
        # 1. 从 prompt 池采样
        batch_prompts = random.sample(self.prompts, batch_size)

        # 2. 用当前模型采样回答
        with torch.no_grad():
            responses = self.model.generate(batch_prompts, max_new_tokens=512)
            # 3. 用 RM 打分
            rewards = self.rm_model.score(batch_prompts, responses)
            # 4. 用 ref_model 计算 KL 锚点
            ref_logps = compute_logps(self.ref_model, batch_prompts + responses)

        # 5. PPO 训练(同标准 PPO)
        # ...
```

**挑战**: RM 必须实时打分 → 需要 RM 推理服务化部署。

---

## 10. 后训练中的常见工程问题

### 10.1 灾难性遗忘

**症状**: SFT/RLHF 后,模型在通用 benchmark 上掉点。

**根因**: 训练数据分布偏移,SFT 把模型"拉向"特定领域。

**解法**:
1. **Mix 通用数据**: SFT 数据中加 10-20% 通用 pretrain 数据
2. **EMA(Exponential Moving Average)**: 保留 pretrain 权重的"影子"
3. **KL 锚定**: 训练中加 KL 散度惩罚,不让模型偏离 ref 太多
4. **低 lr + 少 epoch**: SFT lr = 1e-5, 1-2 epoch

```python
# Mix pretrain data 防止遗忘
mix_ratio = 0.15  # 15% pretrain, 85% SFT
combined_dataset = ConcatDataset([sft_dataset, pretrain_dataset])
# 在 combined_dataset 上训练
```

### 10.2 长度偏差

**症状**: 训练后模型倾向于长回答(因为 RM 偏好长)。

**解法**:
1. **数据层**: 训练数据 chosen/rejected 长度对齐
2. **损失层**: DPO 加长度归一(SimPO)
3. **RM 层**: RM 训练时加长度归一损失
4. **推理层**: 加 max_length 限制

```python
# SimPO: 长度归一化 DPO
def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5):
    chosen_logps_norm = policy_chosen_logps / chosen_lengths
    rejected_logps_norm = policy_rejected_logps / rejected_lengths
    logits = beta * (chosen_logps_norm - rejected_logps_norm) - gamma
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    return loss
```

### 10.3 训练崩溃(Loss Spike)

**症状**: PPO/GRPO 训练中突然 loss spike,模型输出乱码。

**根因**:
- lr 太高
- KL 系数太小,模型"乱跑"
- 数据异常

**解法**:
1. 检查点回滚:保留最近 N 个 checkpoint,自动回滚
2. KL 系数动态调整(目标 KL 6-10)
3. 梯度裁剪: max_grad_norm = 0.5

```python
# KL 早停
def kl_early_stop(current_kl, target_kl=10.0):
    if current_kl > 2 * target_kl:
        raise EarlyStopException("KL diverged, rolling back")
```

### 10.4 多轮对话的特殊处理

**数据格式**:
```python
conversation = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # ... 多轮
]
```

**Loss Masking**: 每轮 assistant 部分都算 loss,其他 mask。

**关键**: 多轮 SFT 要避免"短视",前几轮的对话也要参与计算 loss。

### 10.5 工具调用(Agent)的 SFT

**数据格式**:
```json
{
  "messages": [...],
  "tools": [
    {"name": "search", "description": "...", "parameters": {...}}
  ]
}
```

**训练要点**:
1. **工具描述**: SFT 数据中包含完整的 tools schema
2. **调用格式**: 模型学会生成 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
3. **结果处理**: 模型学会根据工具返回值继续推理

### 10.6 LoRA vs 全参数 SFT

| 维度 | LoRA SFT | 全参数 SFT |
| :--- | :--- | :--- |
| 显存 | 低(单卡可训 70B) | 高(多卡 70B) |
| 速度 | 快 | 慢 |
| 效果 | 通常略差 | 略好 |
| 适用 | 资源受限 / 快速迭代 | 最终模型 |
| Hyperparam | lr 1e-4 | lr 1e-5 |

**经验**:
- 7B 模型全参数 SFT 效果 > LoRA
- 70B+ LoRA 接近全参数
- LoRA 后可用 merged 推理

```python
# LoRA → merged 全参数
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

---

## 11. 后训练评估

### 11.1 评估指标

| 类别 | 指标 | 用途 |
| :--- | :--- | :--- |
| **通用** | MMLU, ARC, HellaSwag | 知识 + 推理 |
| **指令** | IFEval, MT-Bench | 指令遵循 |
| **对话** | LMSYS Chatbot Arena | 人类偏好 |
| **代码** | HumanEval, MBPP | 代码生成 |
| **数学** | GSM8K, MATH | 数学推理 |
| **中文** | C-Eval, CMMLU | 中文能力 |
| **安全** | HarmBench, AdvBench | 安全性 |
| **长文** | RULER, LongBench | 长上下文 |

### 11.2 评估陷阱

1. **Benchmark 污染**: 训练数据包含 benchmark → 分数虚高
2. **格式敏感**: 模型对 prompt 模板敏感,小变化导致大差异
3. **长度偏差**: 评估模型可能偏好长回答
4. **GPT-4-as-Judge 偏差**: GPT-4 偏好自己风格的回答

### 11.3 LMSYS Arena 风格的成对评估

```python
def pairwise_eval(prompt, response_a, response_b, judge_model):
    """用 LLM-as-Judge 做成对比较"""
    judge_prompt = f"""请评估以下两个回答哪个更好。

用户问题:{prompt}

回答 A:{response_a}

回答 B:{response_b}

请先分析两个回答的优缺点,然后输出一个 JSON:
{{"winner": "A" or "B" or "tie", "reason": "..."}}"""
    result = judge_model.generate(judge_prompt)
    return parse_json(result)
```

---

## 12. 后训练工程栈与开源工具

### 12.1 LLaMA-Factory

**特点**: 一站式 SFT/DPO/PPO/RLHF,支持 LoRA,多模态。

```bash
# 安装
pip install llamafactory

# SFT 训练
llamafactory-cli train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --template llama3 \
    --dataset alpaca_gpt4_en,alpaca_gpt4_zh \
    --output_dir ./sft_output \
    --finetuning_type lora \
    --lora_rank 16 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16

# DPO 训练
llamafactory-cli train \
    --model_name_or_path ./sft_output \
    --template llama3 \
    --dataset ultrafeedback_binarized \
    --output_dir ./dpo_output \
    --finetuning_type lora \
    --pref_loss sigmoid \
    --pref_beta 0.1 \
    --learning_rate 5e-7
```

### 12.2 TRL(Transformer Reinforcement Learning)

HuggingFace 官方 RLHF 库,支持 DPO/PPO/GRPO/Online。

```python
from trl import DPOConfig, DPOTrainer, GRPOTrainer, PPOConfig, PPOTrainer

# 各种 trainer 风格统一
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(...),
    train_dataset=dataset,
)
```

### 12.3 OpenRLHF

专门优化了 PPO/GRPO 性能,支持 Ray 分布式。

```python
# OpenRLHF 的 PPO 配置示例
from openrlhf.trainer.ppo_trainer import PPOTrainer

trainer = PPOTrainer(
    strategy=deepspeed_strategy,
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataloader=train_loader,
    valid_dataloader=valid_loader,
    max_epochs=2,
)
```

### 12.4 工具对比

| 工具 | 优点 | 缺点 | 适用 |
| :--- | :--- | :--- | :--- |
| **LLaMA-Factory** | 易用,支持多模态 | 大规模 RL 不灵活 | SFT/DPO 为主 |
| **TRL** | HuggingFace 官方 | PPO 慢 | 实验 |
| **OpenRLHF** | 高性能,分布式 | 学习曲线陡 | 生产级 PPO/GRPO |
| **Verifiers** | RLVR 友好 | 仅 0/1 奖励 | 数学/代码 |
| **DeepSpeed-Chat** | ZeRO 优化 | 微软生态 | 老牌 RLHF |
| **LMFlow** | 字节系 | 中文友好 | 国内生产 |

---

## 13. 长上下文与多模态的后训练

### 13.1 长上下文 SFT

**挑战**: 长上下文(>32K)的 SFT 数据难以构造。

**方法**:
1. **LongAlign / Long-Instruction**: 用 GPT-4 从长文档生成 QA 对
2. **数据合成长上下文**: 从书/论文/代码仓库生成多轮对话
3. **位置编码扩展**: 先扩 RoPE base,再 SFT

### 13.2 多模态 SFT

**数据形式**:
```json
{
  "image": "<image_data>",
  "conversations": [
    {"from": "human", "value": "<image>\n图片里有什么?"},
    {"from": "gpt", "value": "图片里有..."}
  ]
}
```

**训练**:
- 视觉编码器(ViT)通常冻结或 LoRA
- LLM 部分全参数或 LoRA
- 投影层(MLP)全训练

代表:LLaVA、MiniGPT-4、InstructBLIP、Qwen-VL。

### 13.3 Tool-Use / Agent SFT

**数据**: 包含 tool_call + tool_response 的多轮对话。

```json
{
  "messages": [
    {"role": "user", "content": "查一下北京今天天气"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"北京\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"temperature\": 25, \"condition\": \"晴\"}"},
    {"role": "assistant", "content": "北京今天天气晴,温度 25°C。"}
  ]
}
```

训练目标:学会在合适时机调用工具,并正确解析结果。

---

## 14. 后训练的常见误区

### 14.1 "RLHF 必优于 SFT"

错。SFT 已经能达到 80% 的对齐效果,RLHF 主要解决"边界情况"。对于垂直领域,SFT + DPO 通常足够,无需 PPO。

### 14.2 "数据越多越好"

LIMA 论文证明 1K 高质量 > 100K 低质量。但工业生产中 50K-100K 是合理基线。

### 14.3 "RM 越大越好"

RM 不需要和 policy 一样大。7B RM 配 70B policy 是常见配置。但 RM 必须和数据分布匹配。

### 14.4 "PPO 总比 DPO 好"

DPO 在大多数 benchmark 上接近 PPO。DPO 简单、稳定、易部署。但 PPO 在推理任务(RLVR)上仍优。

### 14.5 "RLHF 后能力会提升"

错。RLHF 主要提升"对齐",不直接提升能力。能力提升靠 RLVR + 推理时计算扩展。

---

## 15. 工业实践案例

### 15.1 LLaMA-2-Chat 后训练

**SFT 数据**: 27K 标注对话(27K 私有 + 公开数据)

**RM 数据**: 1.4M 偏好对(UltraFeedback 等)

**RLHF**: PPO with KL coefficient 0.01-0.05

**关键超参**: learning_rate = 1e-5, batch_size = 64

### 15.2 ChatGLM 后训练

**SFT**: 中文对话数据(COIG, Firefly)
**RM**: 人类偏好 + AI 偏好混合
**RLHF**: PPO with safety reward

### 15.3 Qwen 后训练

**多阶段 SFT**: 通用 → 中文 → 代码 → 数学
**DPO**: UltraFeedback 数据
**GRPO**: 数学推理任务

### 15.4 DeepSeek-R1 后训练

**SFT**: 长 CoT 数据(蒸馏 R1)
**GRPO**: 推理任务,用 verifier(数学正确性)做奖励
**无 RM**: 直接用 verifier,绕开 RLHF 的"奖励 hacking"

---

## 16. 后训练的未来方向

### 16.1 Self-Play / RLAIF 进化

模型自己生成偏好数据,减少人类标注依赖。

### 16.2 Process Reward Model(PRM)普及

从"答对答错"到"过程对错",PRM 引导模型逐步推理。

### 16.3 多智能体辩论

多个模型辩论,最终投票决定偏好,减少单模型偏见。

### 16.4 Online + Offline 混合

Offline 数据保证稳定,Online 探索保证进步,两者结合。

### 16.5 长思维链(Long CoT)训练

o1/R1 风格的"长思考",需要 RLVR + Process Reward + Test-time Scaling。

### 16.6 多模态 RLHF

图像/视频/音频的对齐,跨模态偏好,RM 扩展到多模态。

### 16.7 个性化对齐

不同用户偏好不同,需要 RLHF 支持"用户级"个性化。

### 16.8 可解释对齐

模型不仅对齐,还能解释"为什么这个回答更好"。

---

## 17. 速查表与决策树

### 17.1 算法选择决策树

```text
任务类型是什么?
├─ 可自动验证 (数学/代码/定理)
│   └─ 用 RLVR (GRPO + verifier)
├─ 需要通用对话能力
│   ├─ 资源充足 → SFT + RM + PPO
│   ├─ 资源中等 → SFT + DPO/SimPO
│   └─ 快速迭代 → SFT only
├─ 需要特定领域(法律/医疗)
│   ├─ 有专家标注 → SFT + DPO
│   └─ 仅通用偏好 → SFT + DPO
└─ 需要安全/价值对齐
    └─ SFT + RLAIF (Constitutional AI)
```

### 17.2 超参速查表

| 算法 | lr | β (KL coef) | batch | epoch |
| :--- | :--- | :--- | :--- | :--- |
| **SFT** | 1e-5 ~ 2e-5 | - | 32-128 | 2-3 |
| **DPO** | 5e-7 ~ 1e-6 | 0.1-0.5 | 32-64 | 1-2 |
| **PPO** | 5e-7 ~ 1e-6 | 0.01-0.1 | 64-256 | 1-2 |
| **GRPO** | 1e-6 ~ 5e-6 | 0.01-0.05 | 64-128 | 1-2 |
| **KTO** | 5e-7 ~ 5e-6 | 0.1 | 32-64 | 1-2 |

### 17.3 数据配比速查

```text
通用 SFT:
  通用对话 60% + 指令 20% + 代码 10% + 数学 5% + 安全 5%

数学专用 SFT:
  数学 50% + 代码 30% + 通用 20%

代码专用 SFT:
  代码 50% + 数学 20% + 通用 30%

DPO 数据:
  UltraFeedback / Anthropic HH-RLHF / HelpSteer / Capybara
  推荐 ≥10K 条
```

### 17.4 监控指标速查

| 指标 | 含义 | 异常处理 |
| :--- | :--- | :--- |
| `train_loss` | 训练损失 | 持续下降 → 正常; 突增 → 检查数据 |
| `eval_loss` | 验证损失 | 不下降 → 过拟合,减少 epoch |
| `reward_mean` | 平均奖励 | 上升 → 正常; 突降 → 模型退化 |
| `kl_mean` | KL 散度 | 过大 → 增 β;过小 → 减 β |
| `response_len` | 平均长度 | 持续上升 → 长度 hacking |
| `clip_frac` | PPO clip 比例 | <5% 或 >20% → 调 ε |
| `entropy` | 策略熵 | 持续下降 → 模式坍缩 |
| `chosen_acc` | DPO 准确率 | 接近 100% → 可能过拟合 |

---

## 18. 代码实战:完整 RLHF 流程

下面给出一个"端到端"的 RLHF 实现,涵盖 SFT + RM + DPO。

### 18.1 训练 SFT 模型

```python
# train_sft.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

config = SFTConfig(
    output_dir="./sft_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
)
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 18.2 训练 RM

```python
# train_rm.py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "./sft_model",
    num_labels=1,
    torch_dtype=torch.bfloat16,
)
# 训练 RM(BT 损失,同 3.2.3)
```

### 18.3 训练 DPO

```python
# train_dpo.py
from trl import DPOTrainer, DPOConfig

policy = AutoModelForCausalLM.from_pretrained("./sft_model", torch_dtype=torch.bfloat16)
ref = AutoModelForCausalLM.from_pretrained("./sft_model", torch_dtype=torch.bfloat16)

config = DPOConfig(
    output_dir="./dpo_model",
    learning_rate=5e-7,
    beta=0.1,
    num_train_epochs=1,
    bf16=True,
)
trainer = DPOTrainer(
    model=policy, ref_model=ref, args=config,
    train_dataset=dpo_dataset, tokenizer=tokenizer,
)
trainer.train()
```

### 18.4 训练 GRPO(数学任务)

```python
# train_grpo.py
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./grpo_model",
    learning_rate=1e-6,
    beta=0.04,
    num_iterations=1,  # 每批 1 轮
    num_generations=8,  # 每个 prompt 采样 8 个回答
    max_completion_length=1024,
    bf16=True,
)

trainer = GRPOTrainer(
    model=policy_model,
    reward_funcs=[math_correctness_reward, format_reward],
    args=config,
    train_dataset=math_dataset,
)
trainer.train()
```

---

## 19. 进阶话题

### 19.1 Speculative RLHF

用 draft model 加速 PPO 中的"采样"步骤,降低 RLHF 训练时间。

### 19.2 Memory-Efficient PPO

```python
# 关键技术:
# 1. LoRA: 把 policy 拆成 LoRA,显存降 90%
# 2. 8-bit Adam: 优化器状态省 4x
# 3. Activation checkpointing
# 4. KV cache 共享(同一个序列在 policy 和 ref 上)
```

### 19.3 Safe RLHF

双 RM 训练:一个 helpful RM + 一个 safety RM。

```python
# Helpful RM + Safety RM → 加权
total_reward = w_helpful * rm_helpful.score(...) + w_safety * rm_safety.score(...)
# 用 Lagrangian 约束 safety 分数 ≥ 阈值
```

### 19.4 RLHF 的安全风险

1. **Reward Tampering**: 模型学会"欺骗" RM
2. **Specification Gaming**: 模型只追求"高分"而非"真有用"
3. **Mode Collapse**: 输出多样性下降
4. **Sycophancy**: 模型迎合用户观点,不诚实

---

## 20. 总结

后训练是从"会续写"到"会助手"的关键环节。本章的核心要点:

1. **SFT 是基石**: 高质量 SFT 数据(1K-100K)+ 正确 loss masking + 合理学习率
2. **RM 是桥梁**: 用人类偏好训练 RM,作为 RL 的奖励信号
3. **PPO 是经典**: RLHF 黄金标准,但工程复杂、显存大
4. **DPO 是简化**: 跳过 RM,直接在偏好数据上优化,易部署
5. **GRPO 是新潮**: 推理任务的利器,DeepSeek R1 同款
6. **RLVR 是未来**: 可验证任务(数学/代码)的标准方案
7. **Online 是趋势**: 静态数据集不够,模型要在训练中持续探索

> **核心心法**: 后训练不是 Pretrain 的"补丁",而是产品化的核心环节。一个好 SFT + 一个好 RM + 一个稳定的 PPO 循环,胜过任何花哨算法。

---

## 附录 A: 后训练相关论文清单

```text
SFT / Instruction Tuning:
  - T5 / FLAN-T5 (HuggingFace)
  - InstructGPT / TRL
  - Alpaca / Vicuna / WizardLM
  - LIMA (Meta, 2023) - 1K 数据胜利
  - FLAN / FLAN-v2 (Google)
  - Natural Instructions

RLHF:
  - InstructGPT (OpenAI, 2022) - 原始 RLHF 三阶段
  - ChatGPT (OpenAI, 2022) - 工业落地
  - Anthropic HH-RLHF
  - LLaMA-2-Chat (Meta, 2023)
  - Sparrow (DeepMind, 2023)

DPO 系列:
  - DPO (Stanford, 2023)
  - IPO (Apple, 2023)
  - KTO (Stanford, 2024)
  - ORPO (KAIST, 2024)
  - SimPO (UC Berkeley, 2024)
  - RPO (Salesforce, 2024)

RLAIF / Constitutional AI:
  - Constitutional AI (Anthropic, 2022)
  - RLAIF (Google DeepMind, 2023)
  - Self-Rewarding LM (Meta, 2024)

GRPO / Process Reward:
  - GRPO (DeepSeek, 2024) - 用于 R1
  - Math-Shepherd (2024)
  - o1 (OpenAI, 2024)
  - DeepSeek-R1 (2025)

RLVR:
  - RLVR (2024)
  - GenRM (2024)
  - Verifier-based RL
```

---

> **版本**: v1.0 (2026-08-31) · **维护**: 自维护
>
> *后训练是把"语言模型"变成"语言助手"的关键一步。*



## 深度扩展 · 多视角系统展开(续)

### 39. SFT 等后训练(Post-Training)教程 - Scaling Laws 视角

**核心议题**: 本节从Scaling Laws视角切入,系统展开 "SFT 等后训练(Post-Training)教程" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"SFT 等后训练(Post-Training)教程" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. Scaling Laws视角的展开**

从Scaling Laws视角出发,我们可以把 "SFT 等后训练(Post-Training)教程" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调Scaling Laws视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"SFT 等后训练(Post-Training)教程" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"Scaling Laws" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "SFT 等后训练(Post-Training)教程" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "SFT 等后训练(Post-Training)教程" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "SFT 等后训练(Post-Training)教程" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"SFT 等后训练(Post-Training)教程" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从Scaling Laws视角切入,系统展开了 "SFT 等后训练(Post-Training)教程" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 40. SFT 等后训练(Post-Training)教程 - 效率优化 视角

**核心议题**: 本节从效率优化视角切入,系统展开 "SFT 等后训练(Post-Training)教程" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"SFT 等后训练(Post-Training)教程" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 效率优化视角的展开**

从效率优化视角出发,我们可以把 "SFT 等后训练(Post-Training)教程" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调效率优化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"SFT 等后训练(Post-Training)教程" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"效率优化" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "SFT 等后训练(Post-Training)教程" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "SFT 等后训练(Post-Training)教程" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "SFT 等后训练(Post-Training)教程" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"SFT 等后训练(Post-Training)教程" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从效率优化视角切入,系统展开了 "SFT 等后训练(Post-Training)教程" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 41. SFT 等后训练(Post-Training)教程 - 应用落地 视角

**核心议题**: 本节从应用落地视角切入,系统展开 "SFT 等后训练(Post-Training)教程" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"SFT 等后训练(Post-Training)教程" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用落地视角的展开**

从应用落地视角出发,我们可以把 "SFT 等后训练(Post-Training)教程" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调应用落地视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"SFT 等后训练(Post-Training)教程" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"应用落地" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "SFT 等后训练(Post-Training)教程" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "SFT 等后训练(Post-Training)教程" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "SFT 等后训练(Post-Training)教程" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"SFT 等后训练(Post-Training)教程" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用落地视角切入,系统展开了 "SFT 等后训练(Post-Training)教程" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 42. SFT 等后训练(Post-Training)教程 - 未来展望 视角

**核心议题**: 本节从未来展望视角切入,系统展开 "SFT 等后训练(Post-Training)教程" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"SFT 等后训练(Post-Training)教程" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 未来展望视角的展开**

从未来展望视角出发,我们可以把 "SFT 等后训练(Post-Training)教程" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调未来展望视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"SFT 等后训练(Post-Training)教程" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"未来展望" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "SFT 等后训练(Post-Training)教程" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "SFT 等后训练(Post-Training)教程" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "SFT 等后训练(Post-Training)教程" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"SFT 等后训练(Post-Training)教程" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从未来展望视角切入,系统展开了 "SFT 等后训练(Post-Training)教程" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 43. SFT 等后训练(Post-Training)教程 - 数据视角 视角

**核心议题**: 本节从数据视角视角切入,系统展开 "SFT 等后训练(Post-Training)教程" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"SFT 等后训练(Post-Training)教程" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数据视角视角的展开**

从数据视角视角出发,我们可以把 "SFT 等后训练(Post-Training)教程" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调数据视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"SFT 等后训练(Post-Training)教程" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"数据视角" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "SFT 等后训练(Post-Training)教程" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "SFT 等后训练(Post-Training)教程" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "SFT 等后训练(Post-Training)教程" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"SFT 等后训练(Post-Training)教程" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数据视角视角切入,系统展开了 "SFT 等后训练(Post-Training)教程" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



