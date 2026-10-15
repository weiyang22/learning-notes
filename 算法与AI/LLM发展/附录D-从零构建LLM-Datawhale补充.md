---
title: 附录D-从零构建LLM · Datawhalechina 补充
tags:
  - LLM
  - 从零构建
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/happy-llm 第五章（动手搭建大模型）与 llms-from-scratch-cn，补充 LLaMA2 手搓实现、Tokenizer 训练与 215M 预训练全流程
companion_to: 附录D-从零构建LLM.md
source: https://github.com/datawhalechina/happy-llm ; https://github.com/datawhalechina/llms-from-scratch-cn
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 从零构建 LLM · Datawhalechina 对照补充

> 本文件是对同目录《附录D-从零构建LLM.md》的补充，内容整理自 [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)（从零开始构建大模型，34k stars）**第五章 动手搭建大模型 + 第六章 训练实践**，以及 [datawhalechina/llms-from-scratch-cn](https://github.com/datawhalechina/llms-from-scratch-cn)（从 0 构建 GLM4/Llama3/RWKV6）。原文档已有从零构建的框架性内容；happy-llm 的差异化价值是**一条完整可复现的产出链**：手搓 LLaMA2 → 训练 Tokenizer → 215M 预训练 → SFT → 模型开源在 ModelScope 上可直接对照。

---

## 一、补充来源地图

| datawhale 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| happy-llm ch2 | 手把手搭 Transformer | 附录D 正文 | 逐模块（MHA/FFN/LN）的 PyTorch 实现 |
| happy-llm ch5 | **LLaMA2 完整实现**、Tokenizer 训练、215M 预训练 | 附录D 正文 | **核心补充**：RMSNorm/RoPE/SwiGLU/GQA 的具体代码 |
| happy-llm ch6 | Transformers 框架的 Pretrain/SFT/LoRA | 附录D 末尾 | 从"手搓"切换到"框架"的桥 |
| happy-llm ch7/ch8 | 评测、RAG、Agent；GRPO/Agentic-RL | 附录C/其他章 | 构建之后的去向 |
| llms-from-scratch-cn | GLM4 / Llama3 / RWKV6 逐步构建 | 附录D 正文 | **多架构对照**：同任务不同架构的实现差异 |
| tiny-universe | 大模型白盒子（各组件手搓合集） | 附录D 正文 | BPE/LoRA/RLHF 等组件级白盒 |

---

## 二、核心补充内容

### 2.1 LLaMA2 关键组件的实现要点（happy-llm ch5）

happy-llm 给出的 LLaMA2 四件套实现，每件都有"与标准 Transformer 的差异点"，原文档可对照检查自己的实现：

**① RMSNorm（替代 LayerNorm）**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight        # 无均值中心化，减一次归约
```

**② RoPE 旋转位置编码**：把位置信息作为复数旋转作用在 Q/K 的相邻两维上，注意力内积自动获得相对位置依赖——实现上预计算 `cos/sin` 缓存，按 `position_ids` 索引（与本地《数学/第07章-15 RoPE 旋转编码》的数学推导正好衔接）。

**③ SwiGLU 前馈**：`FFN(x) = W_down( SiLU(W_gate x) ⊙ W_up x )`，注意 `hidden_dim ≈ 8/3 * d_model` 的取整约定。

**④ GQA 分组查询注意力**：KV head 数 < Q head 数（如 32:8），推理 KV cache 缩小 4 倍——这是"为什么 LLaMA2-70B 能单机推理"的工程关键。

### 2.2 Tokenizer 训练与数据配比（happy-llm ch5.3）

happy-llm 完整走了一遍 **BPE 训练**（`tokenizers` 库），并列出原文档常忽略的三个实践点：

1. **预分词规则**决定数字/中文的切分粒度（中文建议字级或词级混合，数字建议 3 位一组以稳定算术）；
2. **特殊 token 清单前置**：`<bos>/<eos>/<pad>/<unk>` 与对话模板 token 必须在训练时预留 id，训练后追加会破坏 embedding 分布；
3. **词表大小的算账**：词表 V 与 embedding 参数 `V × d` 成正比——215M 模型选 V=64k、d=576 的配比来源。

### 2.3 215M 预训练的可复现配置（happy-llm ch5/ch6）

happy-llm 把训练产物开源（ModelScope：`happy-llm-215M-base` / `-sft`），其公开配置可直接作为本地实验的锚点：

| 项 | 配置 | 说明 |
| :--- | :--- | :--- |
| 参数量 | 215M（d=576, 26 层, 12 head） | 单卡可训 |
| 词表 | 64k BPE | 中文语料 |
| 数据 | 中英文清洗语料（ch5 给出清洗管线） | 去重/低质过滤 |
| 框架 | 先纯 PyTorch（ch5 手写训练循环），后 Transformers+SwanLab（ch6） | 两阶段对照学习 |
| 监控 | loss 曲线 + 学习率调度 + 梯度范数 | ch6 的 SwanLab 面板可对照 |

**对照学习建议**：先按 ch5 手写训练循环跑通（理解每一步在做什么），再用 ch6 的框架版跑同样配置——两版 loss 曲线应基本重合，重合了才说明两版实现等价。这是检验"从零构建"正确性的最硬标准。

### 2.4 多架构对照（llms-from-scratch-cn）

llms-from-scratch-cn 的独特价值是**同一任务、三种架构**的平行实现，适合做架构对比笔记：

| 架构 | 核心差异 | 适合对照的原文档章节 |
| :--- | :--- | :--- |
| GLM4 | 自回归填空（autoregressive blank infilling） | LLM发展/第02章-架构演进 |
| Llama3 | 更大词表(128k)/GQA/长上下文 | 本附录 LLaMA2 实现的升级对照 |
| RWKV6 | 线性注意力 RNN 化，推理 O(1) 显存 | LLM发展/Mamba/0-Mamba与SSM详解 |

---

## 三、可直接动手的实践

1. **对照实现**：按 ch5 逐模块实现 LLaMA2，与本附录已有代码逐块 diff，重点核对 RoPE 的位置索引与 GQA 的 head 映射；
2. **Tokenizer 复刻**：用 `tokenizers` 训练一个 64k 中文 BPE，检查数字/空格的切分行为；
3. **两阶段训练**：手写循环与 Transformers 框架各跑 2000 step，对比 loss 曲线是否重合；
4. **进阶**：用 tiny-universe 的组件白盒（LoRA/RLHF 模块）给 215M SFT 模型加 LoRA 微调，体会 PEFT（衔接 LLM发展/第09章-SFT等后训练教程）。

---

## 四、延伸链接

- happy-llm 在线阅读：<https://datawhalechina.github.io/happy-llm/>
- ch5 源码：<https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter5>
- 215M 模型（ModelScope）：<https://www.modelscope.cn/models/kmno4zx/happy-llm-215M-base>
- llms-from-scratch-cn：<https://github.com/datawhalechina/llms-from-scratch-cn>
- 本地相关文件：LLM发展/第02章-架构演进、第07章-LLM训练全流程实务、第09章-SFT等后训练教程；数学/第07章-15-RoPE旋转编码
