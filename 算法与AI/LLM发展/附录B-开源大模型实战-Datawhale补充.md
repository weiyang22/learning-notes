---
title: 附录B-开源大模型实战 · Datawhalechina 补充
tags:
  - LLM
  - 开源模型
  - 微调部署
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/self-llm（开源大模型食用指南）、handy-ollama、unlock-hf，补充各开源模型的微调部署实操手册索引与环境配方
companion_to: 附录B-开源大模型实战.md
source: https://github.com/datawhalechina/self-llm ; https://github.com/datawhalechina/handy-ollama ; https://github.com/datawhalechina/unlock-hf
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 开源大模型实战 · Datawhalechina 对照补充

> 本文件是对同目录《附录B-开源大模型实战.md》的补充，内容整理自 [self-llm](https://github.com/datawhalechina/self-llm)（开源大模型食用指南，32k stars）、[handy-ollama](https://github.com/datawhalechina/handy-ollama)（CPU 部署）与 [unlock-hf](https://github.com/datawhalechina/unlock-hf)（HuggingFace 生态）。原文档讲开源模型的全景；self-llm 的差异化价值是**逐模型的环境配置 → 微调（全参/LoRA）→ 部署（vLLM/FastAPI）→ 评测的"食谱式"实操**——每个模型一份可直接照做的手册。

---

## 一、补充来源地图

| 来源 | 定位 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| self-llm | 各开源模型（Qwen/GLM/LLaMA/InternLM/DeepSeek 等系列）的 Linux 微调部署手册 | 附录B 正文 | **核心补充**：逐模型食谱 |
| handy-ollama | Ollama CPU 部署 | 附录B 部署节 | 轻量/边缘轨 |
| unlock-hf | HuggingFace 生态百般用法 | 附录B 工具节 | transformers/datasets/PEFT/TRL 工具链 |
| self-llm 的环境章节 | conda/驱动/CUDA 基线 | 附录B 环境节 | 可复现环境配方 |

---

## 二、核心补充内容

### 2.1 self-llm 的"食谱"结构（作为附录B 的导航模板）

self-llm 每个模型章节遵循同一模板，可直接用作本地附录B 的内容组织标准：

```text
环境配置（conda + 驱动/CUDA 版本矩阵）
→ 模型下载（ModelScope/HF 镜像）
→ Inference（原生 / transformers / vLLM 三种推理方式）
→ 全参微调（xtuner/LLaMA-Factory 路线，显存门槛表）
→ LoRA/QLoRA 微调（rank/alpha/target_modules 配方）
→ 部署（vLLM OpenAI 兼容服务 / FastAPI + UI）
→ 评测（基准数据集 + 量化质量抽查）
```

**选型要点**（self-llm 各章横评得出的口径）：推理吞吐优先 → vLLM 路线；微调显存受限 → QLoRA（4bit）+ paged_adamw_8bit；生产 API → vLLM 的 OpenAI 兼容层 + 前缀缓存开启。

### 2.2 显存门槛速查（self-llm 微调章节横向归纳）

以 7~8B 级模型为例的常见门槛（训练侧，batch=1 + 梯度累积）：

| 方式 | 大致显存 | 说明 |
| :--- | ---: | :--- |
| 全参微调（bf16 + DeepSpeed ZeRO-3） | 8×A100-80G 级 | 多机起步 |
| LoRA（bf16） | 单张 24~48G | rank=8~64 |
| QLoRA（4bit + paged optimizer） | 单张 12~24G | 消费级卡可跑 |
| 推理（vLLM，fp16） | ~16G | 7B 级 |
| 推理（GPTQ/AWQ 4bit） | ~6G | 与本地 K8s 补充文件的量化线衔接 |

> 门槛随框架版本变化，以 self-llm 对应模型章节的最新数字为准；此处给出的是数量级锚点。

### 2.3 HuggingFace 工具链地图（unlock-hf）

unlock-hf 把生态工具按"数据 → 训练 → 评测 → 部署"整理，正好补附录B 的工具索引层：

- **数据**：`datasets`（流式加载/Map-Filter 清洗）、`datatroque` 类清洗；
- **训练**：`transformers` Trainer、`PEFT`（LoRA/QLoRA/Prefix）、`TRL`（SFTTrainer/DPOTrainer——与本地第05章对齐线的代码入口）；
- **评测**：`lm-eval-harness`、OpenCompass（中文）；
- **部署**：`transformers` 的 `device_map="auto"`、vLLM、Ollama（GGUF 导出）。

### 2.4 CPU 轨（handy-ollama）

handy-ollama 覆盖 Ollama 的安装、Modelfile 定制、GGUF 量化档位、API 网关化，适用位：本地开发/CI 冒烟/离线环境。与附录B 的 GPU 生产轨构成双轨（详见《工程与职业/0-K8s与ML模型部署-Datawhale补充.md》的 2.3）。

---

## 三、可直接动手的实践

1. 按 self-llm 任一模型章节（建议 Qwen 系列）完整走一遍"下载→LoRA 微调→vLLM 部署→OpenAPI 调用"，记录每步实际耗时与显存；
2. 用 unlock-hf 的 TRL DPOTrainer 给第05章对齐线补一次最小 DPO 实验；
3. 把手上的 7B 模型导出 GGUF，在 Ollama 上跑 CPU 推理，与 vLLM GPU 推理做吞吐-成本对照表。

---

## 四、延伸链接

- self-llm：<https://github.com/datawhalechina/self-llm>
- handy-ollama：<https://datawhalechina.github.io/handy-ollama/>
- unlock-hf：<https://github.com/datawhalechina/unlock-hf>
- 本地相关文件：第07章-LLM训练全流程实务、第09章-SFT等后训练教程、第05章-对齐技术演进-Datawhale补充.md（TRL/DPO 衔接）、工程与职业/0-K8s与ML模型部署-Datawhale补充.md
