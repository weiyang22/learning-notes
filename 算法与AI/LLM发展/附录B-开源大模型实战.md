---
title: 附录B 开源大模型实战指南
tags:
  - 开源LLM
  - Qwen
  - LLaMA
  - GLM
  - DeepSeek
  - InternLM
  - Kimi
  - MiniCPM
  - 环境配置
  - 部署
  - 微调
description: 主流开源大模型实战. 整合 self-llm/llm-universe 内容. 涵盖 50+ 模型的环境配置、部署、微调、AMD ROCm / 昇腾 NPU / Apple M 等多平台支持
last_updated: 2026-09-11
status: ✅ v1.0 - 整合 self-llm
chapter_id: appendix-b
---

# 附录B 开源大模型实战指南

> 本附录整合自 [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) (开源大模型食用指南) 的核心内容. 涵盖国内外 50+ 主流开源 LLM 的部署、使用、微调全流程.

---

## 1. 开源 LLM 生态概览

### 1.1 国内主流模型 (Chinese Open-source LLMs)

| 模型家族 | 厂商 | 代表模型 | 参数规模 | 特点 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen** | 阿里通义 | Qwen3, Qwen2.5, Qwen-VL | 0.5B-72B | 全模态, MoE, 长上下文 |
| **GLM** | 智谱 AI | GLM-4.5, GLM-4.7-Flash, GLM-5.2 | 6B-130B | 推理强, Thinking 模式 |
| **DeepSeek** | 深度求索 | DeepSeek-V3, R1, Coder-V2 | 7B-671B (MoE) | MLA + MoE, 极致性价比 |
| **InternLM** | 上海 AI Lab | InternLM2, InternLM3 | 1.8B-20B | 全栈工具链 (XTuner/LMDeploy) |
| **Kimi** | 月之暗面 | Kimi-K2.5 | 多种 | 长上下文 (200K+) |
| **MiniCPM** | 面壁智能 | MiniCPM-o-2.6, MiniCPM-5 | 1B-8B | 小而强, 多模态 |
| **Yi** | 零一万物 | Yi-1.5 | 6B-34B | 中英双语 |
| **Baichuan** | 百川智能 | Baichuan2 | 7B-13B | 中文优化 |
| **BlueLM** | vivo | BlueLM | 7B | 端侧优化 |

### 1.2 国外主流模型

| 模型家族 | 厂商 | 代表模型 | 特点 |
| :--- | :--- | :--- | :--- |
| **Llama** | Meta | Llama3.1, Llama4 | 开源标杆 |
| **Gemma** | Google | Gemma3, Gemma-2 | 轻量高效 |
| **Phi** | Microsoft | Phi4 | 小尺寸 |
| **Mistral** | Mistral AI | Mistral-Large | MoE 架构 |
| **OpenELM** | Apple | OpenELM | 端侧 |
| **gpt-oss** | OpenAI | gpt-oss-20B | 开源版 |

### 1.3 多模态模型

| 模型 | 类型 | 能力 |
| :--- | :--- | :--- |
| **Qwen2-VL / Qwen3-VL** | 视觉语言 | 图文理解, OCR, 文档 |
| **Qwen-Audio** | 语音 | ASR + TTS |
| **Hunyuan3D-2** | 3D | 文/图生 3D |
| **SpatialLM** | 空间 | 3D 场景理解 |
| **MiniCPM-o** | 全模态 | 图文音视 |
| **Index-1.9B** (哔哩哔哩) | 多模态 | 角色扮演, 音频 |

---

## 2. 环境配置基础

### 2.1 Linux 通用环境 (NVIDIA GPU)

```bash
# 1. CUDA 驱动
nvidia-smi  # 验证驱动

# 2. Conda 环境
conda create -n llm python=3.10 -y
conda activate llm

# 3. PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 4. Transformers + 加速库
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install sentencepiece  # Qwen tokenizer
pip install tiktoken      # GPT tokenizer

# 5. 推理引擎 (二选一)
# vLLM (生产推荐)
pip install vllm
# 或 TGI (HuggingFace)
docker run -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id Qwen/Qwen2.5-7B-Instruct
```

### 2.2 模型下载 (ModelScope)

国内推荐 ModelScope (速度快), 海外用 HuggingFace.

```python
from modelscope import snapshot_download

# 下载 Qwen2.5-7B
model_dir = snapshot_download(
    'qwen/Qwen2.5-7B-Instruct',
    cache_dir='/root/models',
    revision='master'
)
print(f"模型下载到: {model_dir}")
```

### 2.3 HuggingFace 下载

```python
from huggingface_hub import snapshot_download

# 下载 Llama-3.1-8B
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./models/Llama-3.1-8B-Instruct",
    token="hf_xxx"  # 需要 token
)
```

---

## 3. 部署实战

### 3.1 Transformers 原生部署 (适合开发调试)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 对话
messages = [
    {"role": "system", "content": "你是 Datawhale 的助手."},
    {"role": "user", "content": "什么是大模型?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### 3.2 vLLM 生产部署 (推荐) ⭐

```python
from vllm import LLM, SamplingParams

# 加载模型 (一次)
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,  # GPU 数
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

# 推理参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 单条推理
prompt = "用一句话解释 Transformer"
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)

# 批量推理
prompts = ["解释 Attention", "解释 MLP", "解释 LayerNorm"]
outputs = llm.generate(prompts, sampling_params)
for out in outputs:
    print(out.outputs[0].text)
```

### 3.3 OpenAI 兼容 API 服务

```python
# vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7
)
print(response.choices[0].message.content)
```

### 3.4 Ollama 本地部署 (Mac / Linux)

```bash
# Mac (Apple Silicon) / Linux / Windows
curl -fsSL https://ollama.ai/install.sh | sh

# 下载并运行
ollama run qwen2.5:7b
ollama run llama3.1:8b
ollama run deepseek-r1:7b
```

---

## 4. 量化技术

### 4.1 GPTQ 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import BaseQuantizeConfig

# 4-bit GPTQ 量化
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    quantize_config=quantize_config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# 保存量化模型
model.save_quantized('./Qwen2.5-7B-Instruct-GPTQ')
tokenizer.save_pretrained('./Qwen2.5-7B-Instruct-GPTQ')
```

### 4.2 AWQ 量化

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model_path = 'Qwen/Qwen2.5-7B-Instruct'
quant_path = './Qwen2.5-7B-Instruct-AWQ'
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### 4.3 BnB (BitsAndBytes) 4-bit

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## 5. 微调实战

### 5.1 LoRA 微调 (最常用)

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# 1. 加载基础模型
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 小模型演示
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,                      # LoRA rank
    lora_alpha=32,            # alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 1.5M || all params: 1.5B || trainable%: 0.10%

# 3. 准备数据
dataset = load_dataset("your_dataset", split="train")

def format_example(example):
    text = f"### 指令:\n{example['instruction']}\n\n### 回答:\n{example['output']}"
    return tokenizer(text, truncation=True, max_length=512)

tokenized = dataset.map(format_example)

# 4. 训练
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./qwen-lora-finetuned")
```

### 5.2 QLoRA (4-bit + LoRA) ⭐ 显存友好

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# 4-bit 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# 显存占用: 7B 模型 + QLoRA ≈ 6GB (vs 全参 14GB)
```

### 5.3 全参数微调 (DeepSpeed)

```python
# 适用于多卡 A100/H100
# 使用 DeepSpeed ZeRO-2/3
deepspeed_config = """
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "bf16": {"enabled": true}
}
"""

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    deepspeed=deepspeed_config_path
)
```

---

## 6. 多平台支持

### 6.1 AMD ROCm 平台

ROCm 7+ 已支持主流模型, 使用 LM Studio / vLLM / Ollama / llama.cpp.

```bash
# ROCm 环境验证
rocm-smi

# vLLM ROCm
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2

# 运行
vllm serve Qwen/Qwen2.5-7B-Instruct --device cuda
```

### 6.2 昇腾 Ascend NPU 平台

支持 Qwen3 / Qwen3.5 / Qwen3.6 / GLM 等.

```bash
# 使用 vLLM-Ascend
pip install vllm-ascend

# MindIE 服务化部署
# sglang-ascend
pip install sglang-ascend

# 性能测试 (AISBench)
pip install ais_bench
```

### 6.3 Apple M 系列

```bash
# 使用 MLX 框架
pip install mlx mlx-lm

# 或 Ollama (MLX 后端)
ollama serve  # Mac GPU 加速
```

---

## 7. 推理优化与部署

### 7.1 KV Cache 优化

```python
# PagedAttention (vLLM 核心)
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True,   # 前缀缓存
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=2048,
    max_num_seqs=256,
)
```

### 7.2 Continuous Batching

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=256,           # 最大并发
    scheduling_policy="fcfs",    # 调度策略
    enable_chunked_prefill=True,  # 分块 prefill
)
```

### 7.3 Speculative Decoding (推测解码)

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_model="Qwen/Qwen2.5-0.5B-Instruct",  # 小模型作为 draft
    num_speculative_tokens=5,
)
```

---

## 8. RAG 与 Agent 集成

### 8.1 LangChain 集成

```python
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. 加载 LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    device=0,
    model_kwargs={"temperature": 0.7, "max_length": 2048}
)

# 2. 加载向量库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = FAISS.load_local("./faiss_index", embeddings)

# 3. RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 4. 问答
answer = qa.run("Datawhale 是什么组织?")
print(answer)
```

### 8.2 完整 RAG 实战 (datawhalechina/all-in-rag 推荐)

详见 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程.

---

## 9. 常见问题与调试

### Q1: 模型下载慢 / 失败

**解决**:
```bash
# 国内: 使用 ModelScope
pip install modelscope
# 配置镜像
export MODELSCOPE_CACHE=/root/models

# 海外: 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 显存不足 (OOM)

**解决**:
1. **量化**: 4-bit 量化节省 ~75% 显存
2. **LoRA**: 只训练部分参数
3. **Gradient Checkpointing**: 牺牲时间换显存
4. **DeepSpeed ZeRO**: 多卡分片
5. **Flash Attention**: 显存节省 + 加速

```python
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

### Q3: 中文乱码

**解决**:
- 使用 Qwen / GLM / Baichuan 等原生中文模型
- 检查 tokenizer 是否正确加载 (trust_remote_code=True)
- 设置 `torch_dtype=torch.bfloat16`

### Q4: 推理慢 / 显存不够装不下

**解决**:
- 使用 vLLM (生产推荐, PagedAttention)
- 使用 GGUF 格式 + llama.cpp (CPU/Mac)
- 模型并行 (tensor_parallel_size=N)

---

## 10. 模型选择指南

| 场景 | 推荐模型 | 显存需求 |
| :--- | :--- | :--- |
| **本地对话** | Qwen2.5-7B-Instruct | 16GB |
| **代码生成** | DeepSeek-Coder-V2, Qwen2.5-Coder | 16-32GB |
| **长文档** | Kimi-K2.5 (200K) | 32GB+ |
| **多模态** | Qwen2-VL-7B, MiniCPM-o | 16GB |
| **手机端** | Qwen2.5-0.5B, MiniCPM-1B | <4GB |
| **企业生产** | Qwen2.5-72B, DeepSeek-V3 | 多卡 A100/H100 |
| **学术研究** | Llama-3.1-8B, Gemma-2-9B | 16GB |

---

## 11. 推荐学习路径

1. **入门**: Qwen2.5-0.5B-Instruct (Ollama 本地运行)
2. **进阶**: Qwen2.5-7B-Instruct + LoRA 微调
3. **生产**: Qwen2.5-72B + vLLM + RAG
4. **研究**: DeepSeek-V3 (MoE) + MLA + GRPO
5. **前沿**: 推理模型 (o1 / R1) + Agent

---

## 12. 参考资源

- [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) — 50+ 模型部署微调
- [datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe) — 应用开发
- [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) — RAG 全栈
- [Qwen 官方文档](https://qwen.readthedocs.io/)
- [vLLM 文档](https://docs.vllm.ai/)
- [HuggingFace 文档](https://huggingface.co/docs)

### 经典论文
1. **LoRA**: Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
2. **QLoRA**: Dettmers et al., 2023. *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
3. **vLLM (PagedAttention)**: Kwon et al., 2023. *Efficient Memory Management for LLM Serving with PagedAttention*. SOSP 2023.

---

> 本附录覆盖了 self-llm 的核心要点. 实际部署/微调时, 请根据具体硬件和需求选择合适的配置. 后续附录将整合 llm-cookbook 的 Prompt Engineering 与 all-in-rag 的 RAG 实践.
