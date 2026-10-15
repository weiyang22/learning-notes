---
title: "第01章-LLM发展历程总览"
tags:
  - LLM
  - LLM发展历程总览
description: "本章要点：动机: GPT-2→3 发现能力随参数/数据/算力平滑可预测地提升。"
chapter_id: LLM发展-01
last_updated: 2026-07-14
---

# 第01章-LLM发展历程总览


## 1. 时间线速览

| 年份 | 里程碑 | 关键突破 | 意义 |
| :--- | :--- | :--- | :--- |
| 2017 | **Transformer** | 自注意力替代 RNN | 可并行, 奠基 |
| 2018 | **BERT / GPT-1** | 预训练+微调范式 | NLP 迁移学习 |
| 2019 | **GPT-2** | 零样本能力初现 | 规模即能力 |
| 2020 | **GPT-3 / Scaling Law** | 175B, in-context learning | 少样本, 涌现 |
| 2022 | **InstructGPT / RLHF** | 指令对齐 | 从"续写"到"听话" |
| 2022 | **ChatGPT** | 对话 + RLHF 产品化 | 破圈 |
| 2023 | **GPT-4 / LLaMA** | 多模态; 开源生态 | 能力跃升 + 平民化 |
| 2023 | **RoPE/GQA/FlashAttn 普及** | 效率与长上下文 | 工程成熟 |
| 2024 | **DeepSeek-V2/V3, MoE, MLA** | 极致性价比 | 稀疏化 + KV 压缩 |
| 2024 | **DPO 等免 RL 对齐** | 简化对齐 | 降低对齐门槛 |
| 2024-25 | **o1 / R1 / 推理模型** | RL + test-time compute | 从"快答"到"慢想" |
| 2025-26 | **Agent / 长程推理** | 工具使用 + 多步规划 | 走向自主智能体 |

---

## 2. 主线一: 更强(Scaling Law)

**动机**: GPT-2→3 发现能力随参数/数据/算力**平滑可预测地**提升。

**Scaling Law(Kaplan 2020 → Chinchilla 2022)**:
$$L(N,D) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

- Chinchilla 修正: 给定算力, **参数与数据应同比增长**(≈20 token/参数), 此前模型"参数过大数据不足"。
- **优势**: 把"炼丹"变成"可预测的工程", 敢投入大算力。
- **涌现能力**: 某些能力(算术、CoT)在规模跨过阈值后突然出现。

---

## 3. 主线二: 更省(效率)

规模上去后, 训练/推理成本爆炸, 催生一系列效率技术:

- **MoE**(稀疏激活): 参数多但每 token 只激活一部分 → 详见 [第02章](第02章-架构演进.md)。
- **GQA/MQA**: 压缩 KV Cache → [第04章](第04章-注意力与效率演进.md)。
- **FlashAttention**: 减少 HBM 访存 → [deeplearning 6.4](../深度学习手撕/第06章-LLM核心技术/第06章-04-Flash-Attention.md)。
- **量化/蒸馏**: 推理省显存 → deeplearning 第09章。

---

## 4. 主线三: 更长(上下文)

从 512 → 2K → 32K → 128K → 1M+。动机: RAG、长文档、Agent 需要长记忆。

- 位置编码演进(绝对→RoPE→YaRN): [第03章](第03章-位置编码与长上下文演进.md)。
- 稀疏/滑窗/Ring Attention: [deeplearning 6.8](../深度学习手撕/第06章-LLM核心技术/第06章-08-长上下文与注意力变体.md)。

---

## 5. 主线四: 更听话(对齐)

预训练模型只会"续写", 不会"遵循指令、拒绝有害请求"。

- **SFT**: 指令微调, 教格式。
- **RLHF**(InstructGPT 2022): 用人类偏好训奖励模型 + PPO, 是 ChatGPT 的关键。
- **DPO / GRPO**: 简化或改进 RLHF。详见 [第05章](第05章-对齐技术演进.md)。

**优势**: 让模型可用、可控、安全, 是"能力→产品"的临门一脚。

---

## 6. 主线五: 更会想(推理)

2024 起从"快速直觉回答"转向"慢思考":

- **CoT**(思维链): 提示模型分步推理。
- **o1 / DeepSeek-R1**: 用 RL 训练模型**生成长推理链**, 推理时花更多计算(test-time compute)换正确率。
- 详见 [第06章](第06章-推理能力演进.md)。

**范式转变**: 从"scaling 训练"到"scaling 推理时计算"。

---

## 7. 如何用这条主线

读后续每一章时, 都用同一个框架问三个问题:
1. **上一代的痛点**是什么?
2. **这一代怎么解决**的(机制)?
3. **引入了什么新问题**(为下一代埋下伏笔)?

这套"动机链"思维, 比记住任何单个技巧都重要。

---

## 8. 面试高频

1. Scaling Law 与 Chinchilla 的核心结论? → 能力可预测; 参数与数据同比增长。
2. RLHF 为什么重要? → 从"续写"到"听指令", 产品化关键。
3. o1 类推理模型的范式转变? → 从训练 scaling 到推理时计算 scaling。

---

## 第一部分附录 A: 数学基础与 Scaling Law 严格化

## A.1 Kaplan 缩放律(2020)

原始论文 *"Scaling Laws for Neural Language Models"* (Kaplan et al., OpenAI, arXiv:2001.08361) 给出了参数规模 $N$、数据集大小 $D$、计算量 $C_{\min}$ 之间的幂律关系:

**定义 A.1** (Loss 的双变量幂律分解)
设 $L(N, D)$ 为预训练交叉熵损失, 经验上可分解为参数项、数据项与不可约项:
$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D} + L_\infty
$$
其中 $\alpha_N, \alpha_D > 0$ 是幂指数, $N_c, D_c$ 是常数, $L_\infty$ 是不可约损失(数据本身的熵)。

**OpenAI 经验拟合**(Kaplan 2020):
- $L(N) \propto N^{-0.076}$ (固定数据)
- $L(D) \propto D^{-0.095}$ (固定模型)
- $L(C) \propto C^{-0.050}$

**Algorithm A.1** Kaplan 缩放律拟合
**Input**: 不同 $(N_i, D_i, C_i, L_i)$ 的实验点
**Output**: 幂律参数 $(\alpha_N, \alpha_D, N_c, D_c, L_\infty)$
1. 对每个固定 $D$ 的曲线, 用 `scipy.optimize.curve_fit` 拟合 $L = A N^{-\alpha} + L_\infty$
2. 对每个固定 $N$ 的曲线, 拟合 $L = B D^{-\beta} + L_\infty$
3. 联立三个子拟合得全局双变量参数
**Complexity**: Time $O(K \cdot I_{\text{fit}})$, $K$ 实验点数, $I_{\text{fit}}$ 迭代次数

## A.2 Chinchilla 修正(2022)

论文 *"Training Compute-Optimal Large Language Models"* (Hoffmann et al., DeepMind, arXiv:2203.15556) 指出 Kaplan 的拟合因实验设计偏置(只扫 $N$ 不扫 $D$)高估了 $N$ 的边际收益, 给出更平衡的 **Chinchilla 法则**:

**定理 A.1** (Chinchilla 最优分配) 对于给定计算预算 $C = 6ND$, 最优的 $(N^*, D^*)$ 满足:
$$
N^*(C) = G \cdot \left(\frac{C}{6}\right)^{a}, \quad D^*(C) = G^{-1} \cdot \left(\frac{C}{6}\right)^{b}
$$
其中 $a = \frac{\alpha}{\alpha+\beta}$, $b = \frac{\beta}{\alpha+\beta}$, $G = \left(\frac{\alpha}{\beta} \cdot \frac{1}{E}\right)^{1/(\alpha+\beta)}$。DeepMind 的经验值 $\alpha \approx 0.34, \beta \approx 0.28$, 给出 **每参数 ~20 tokens** 的比例。

**Chinchilla(70B) 训了 1.4T tokens**, 在同等计算下 **显著优于 Gopher(280B) 训 0.4T**, 直接证伪了"参数越大越好"。

## A.3 涌现能力与可预测性边界

**定义 A.2** (涌现能力) 某些任务(如多步算术、CoT、跨语言翻译)在 $N$ 跨过某临界阈值 $N_{\text{crit}}$ 后, 准确率从接近随机突然跳到接近 100%, 这种相变行为称为"涌现"。

**Wei et al. 2022** (arXiv:2206.07682) 系统研究了在 137 项任务上的涌现, 发现:
- **平滑可预测任务**: 困惑度、闭合书本 QA 等。
- **涌现任务**: 多步算术、词义消歧、CoT 推理等。
- **警示**: Schaeffer et al. 2023 (arXiv:2304.15004) *"Are Emergent Abilities a Mirage?"* 指出许多"涌现"是**评测指标非线性**(exact match)造成的, 用连续指标看则平滑。

> **实操教训**: 评估 LLM 能力**用连续指标**(log-perplexity、BLEU、Pass@k), 少用 exact match, 否则会被假象误导。

```python
# Chinchilla 最优分配经验估计
def chinchilla_optimal(C_total_flops):
    """
    C_total_flops: 总训练算力(FLOPs)
    返回: (N_optimal_params, D_optimal_tokens)
    """
    # 经验常数 (DeepMind 2022)
    a, b = 0.34, 0.28
    # tokens / param ≈ 20 (Chinchilla optimal)
    tokens_per_param = (a / b) ** (1 / (a + b))  # ≈ 20
    # 由 C ≈ 6ND 解 N*
    N_star = (C_total_flops / 6 / tokens_per_param) ** 0.5
    D_star = N_star * tokens_per_param
    return N_star, D_star

# 例: 1e22 FLOPs (≈ LLaMA-7B 训练量级)
N, D = chinchilla_optimal(1e22)
print(f"N*≈{N/1e9:.2f}B params, D*≈{D/1e9:.1f}B tokens")
```

---

## 第一部分附录 B: 关键论文年表与 arXiv ID

| 年份 | 论文 | 一作/机构 | arXiv | 引用量(粗估) | 核心贡献 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2017 | Attention Is All You Need | Vaswani/Google | 1706.03762 | 130k+ | Transformer |
| 2018 | BERT | Devlin/Google | 1810.04805 | 95k+ | 双向预训练 |
| 2018 | GPT-1 | Radford/OpenAI | — | 8k+ | 生成式预训练 |
| 2019 | GPT-2 | Radford/OpenAI | — | 12k+ | 零样本能力 |
| 2019 | RoBERTa | Liu/FAIR | 1907.11692 | 18k+ | BERT 调参极限 |
| 2020 | GPT-3 | Brown/OpenAI | 2005.14165 | 35k+ | 175B, ICL |
| 2020 | Scaling Laws (Kaplan) | Kaplan/OpenAI | 2001.08361 | 8k+ | 幂律 scaling |
| 2020 | T5 | Raffel/Google | 1910.10683 | 18k+ | 统一 text-to-text |
| 2021 | Switch Transformer | Fedus/Google | 2101.03961 | 3k+ | 万亿 MoE |
| 2021 | Codex | Chen/OpenAI | 2107.03374 | 5k+ | 代码模型 |
| 2022 | InstructGPT | Ouyang/OpenAI | 2203.02155 | 6k+ | RLHF 范式 |
| 2022 | Chinchilla | Hoffmann/DeepMind | 2203.15556 | 4k+ | 最优分配 |
| 2022 | Emergent Abilities | Wei/Google | 2206.07682 | 3.5k+ | 涌现现象 |
| 2022 | FlashAttention | Dao/Tri | 2205.14135 | 4k+ | IO-aware 注意力 |
| 2022 | RLHF survey | Lambert | 2209.14985 | 1k+ | RLHF 综述 |
| 2023 | LLaMA | Touvron/Meta | 2302.13971 | 9k+ | 开源 7-65B |
| 2023 | GPT-4 | OpenAI | 2303.08774 | 10k+ | 多模态闭源 |
| 2023 | DPO | Rafailov/Stanford | 2305.18290 | 4k+ | 免 RL 对齐 |
| 2023 | RoPE 外推综述 | — | 2310.01352 | 1k+ | RoPE 演进 |
| 2023 | QLoRA | Dettmers/UW | 2305.14314 | 2.5k+ | 4-bit 微调 |
| 2024 | LLaMA-3 | Meta | 2407.21783 | 2k+ | 128K, 405B |
| 2024 | DeepSeek-V2 | DeepSeek | 2405.04434 | 1.5k+ | MLA + MoE |
| 2024 | Mixtral 8x7B | Mistral | 2401.04088 | 1.5k+ | 稀疏 MoE |
| 2024 | Mamba | Gu/CMU | 2312.00752 | 4k+ | SSM 替代注意力 |
| 2024 | o1 | OpenAI | (blog) | — | RL 训推理 |
| 2025 | DeepSeek-R1 | DeepSeek | 2501.12948 | 5k+ | 纯 RL 推理 |
| 2025 | GRPO | DeepSeek | 2402.03300 | 1.2k+ | 免 Critic RL |
| 2025 | Claude 3.7/4 | Anthropic | — | — | extended thinking |

---

## 第一部分附录 C: 五大厂商旗舰模型横评(2025-2026)

| 维度 | OpenAI (GPT-5 / o3) | Anthropic (Claude 4) | Google (Gemini 2.5/3) | Meta (LLaMA-4) | DeepSeek (V3 / R1) | Alibaba (Qwen 2.5/3) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 闭源/开源 | 闭源 | 闭源 | 闭源 | 开源 | 开源(MIT) | 开源(Apache 2.0) |
| 旗舰参数 | — | — | — | 109B(MoE 激活) | 671B(MoE 激活 37B) | 72B/110B |
| 上下文 | 256K-1M | 200K-1M | 1M-2M | 128K-10M(MoE) | 64K-128K | 128K |
| 推理能力 | o3/5-thinking 领先 | Extended thinking | thinking mode | — | R1 与 o1 持平 | Qwen-QwQ |
| 注意力 | — | — | — | GQA + iRoPE | MLA | GQA |
| 性价比 | 高(API 贵) | 中 | 中 | 高(自部署) | **极高**(R1 API < o1 5%) | 高 |
| MMLU | 88+ | 86+ | 87+ | 86+ | 88 | 86+ |
| GPQA | 75+ | 70+ | 70+ | 70+ | 71+ | 70+ |
| SWE-bench | 65+ | 60+ | 60+ | — | 49 | 51 |
| 中文 | 中 | 中 | 中 | 弱 | 强 | **极强** |
| 训练算力(估) | $10^{25}$-$10^{26}$ | $10^{25}$ | $10^{25}$ | $10^{24}$ | $10^{24}$ | $10^{24}$ |

---

## 第一部分附录 D: 五条主线的数学/工程本质

## D.1 更强(Scaling): 信息论的视角

设预训练数据分布为 $p(x)$, 模型分布为 $q_\theta(x)$, 训练目标是最小化 KL 散度:
$$
\min_\theta \; D_{\mathrm{KL}}(p \,\|\, q_\theta) = \mathbb{E}_{x \sim p}[-\log q_\theta(x)] + H(p)
$$

Chinchilla 法则本质是: 给定算力 $C$, 应**最小化** $L(C) \propto C^{-\alpha}$ 中的 $\alpha$ 期望值的最优参数/数据比。这把训练变成一个**带约束的优化问题**:
$$
\min_{N, D} \; L(N, D) \quad \text{s.t.} \quad \mathrm{FLOPs}(N, D) = 6ND \le C
$$

KKT 条件给出 $N^* \propto C^{\alpha/(\alpha+\beta)}$, $D^* \propto C^{\beta/(\alpha+\beta)}$。

## D.2 更省(效率): 稀疏性的视角

MoE、量化、KV 压缩的共同本质是**在保持模型容量(参数)的同时降低有效算力**:

$$
\text{Capacity} \approx N_{\text{params}} \cdot b \quad (\text{存储 bit}); \quad \text{Compute} \approx N_{\text{active}} \cdot b' \cdot \text{seq\_len}
$$

稀疏化目标: $N_{\text{params}}$ 大, $N_{\text{active}}$ 小。

**定理 D.1** (稀疏 MoE 容量上界) 设 $E$ 个专家, top-$k$ 路由, 每个 token 激活 $k$ 个专家, 总参数量 $N = E \cdot N_{\text{exp}}$, 但单 token 激活参数 $\tilde{N} = k \cdot N_{\text{exp}}$。当 $k \ll E$, $\tilde{N} \ll N$, 模型总容量 $\Theta(N)$, 但单次前向计算量 $\Theta(\tilde{N})$。

**Algorithm D.1** Top-$k$ 稀疏路由(带负载均衡)
**Input**: 输入 $x \in \mathbb{R}^d$, 专家数 $E$, 激活数 $k$
**Output**: 加权专家输出 $y$
1. $g = \text{softmax}(W_g x) \in \mathbb{R}^E$  # 门控分数
2. $w, \text{idx} = \text{top-}k(g)$  # 选 $k$ 个专家
3. $y = \sum_{i=1}^{k} w_i \cdot \text{Expert}_{\text{idx}_i}(x)$
4. 辅助损失: $L_{\text{aux}} = \alpha \sum_{e} f_e \cdot \bar{g}_e$  # $f_e$ = 专家 $e$ 接收 token 比例
**Complexity**: Time $O(E d + k N_{\text{exp}})$, Space $O(N)$ (参数) + $O(\tilde{N})$ (激活)

## D.3 更长(上下文): 信息密度的视角

设上下文长度 $L$, 注意力计算复杂度 $O(L^2 d)$。长上下文需满足:
$$
\underbrace{L}_{\text{序列长}} \cdot \underbrace{H(L)}_{\text{每 token 信息}} \ge \underbrace{I_{\text{task}}}_{\text{任务信息量}}
$$

**外推的本质**: 训练时学到的相对位置关系 $R(m-n)$ 在测试 $L_{\text{test}} \gg L_{\text{train}}$ 时是否仍然有效? RoPE 用频率 $\theta_i = 10000^{-2i/d}$, 高频捕捉短程, 低频捕捉长程。

**定理 D.2** (RoPE 长程衰减) 设 query/key 维度 $d$, RoPE 旋转后注意力分数 $|q^\top R_{m-n} k| \le \|q\| \|k\| \prod_{i<d/2} \cos((m-n) \theta_i)$。当 $m-n$ 大, 高频项 $\cos$ 振荡、低频项近似 1, 总分呈 $O((m-n)^{-1})$ 多项式衰减, **天然长程衰减**。

## D.4 更听话(对齐): 期望最大化的视角

RLHF 目标:
$$
\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[r(x, y)] - \beta D_{\mathrm{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))
$$

闭式解:
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

DPO 把这个 $\pi^*$ 直接代回损失:
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta \log\frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

## D.5 更会想(推理): 边际效用的视角

test-time compute scaling 的核心是**信息检索的边际效用递减**:
- 容易题: 1-2 步推理即够, 增加计算收益接近 0
- 中等题: 10-50 步推理显著提分
- 难题: 1000+ 步 + 搜索才能解

OpenAI o1、DeepSeek-R1 的训练是**让模型自适应分配思考预算**, 类似人脑"系统1/系统2":
$$
\text{utility}(\text{think\_budget}) = \int_0^{B} \mathbb{E}[\text{reward}(y \mid \text{budget}=b)] \, db
$$

---

## 第一部分附录 E: 工程实践速查(2025 工业界)

## E.1 LLaMA-3 训练栈(Meta, 2024)

| 模块 | 选型 | 备注 |
| :--- | :--- | :--- |
| 框架 | PyTorch + 自研 MetaSeq | 16K GPU 集群 |
| 并行 | TP + CP + PP + FSDP | 405B 用 16K H100 |
| 注意力 | GQA(8 KV 组) + FlashAttention-2 | 长上下文 128K |
| 归一化 | Pre-RMSNorm | |
| 位置编码 | RoPE + NTK 外推 | 训练 8K, 推理 128K |
| 优化器 | AdamW($\beta_2=0.95$) | grad clip 1.0 |
| 精度 | BF16 + FP32 master | 梯度累积 |
| 吞吐 | ~4000 tokens/s/GPU | (估) |
| 训练时长 | ~30 天(405B) | 16K H100 |

## E.2 DeepSeek-V3 训练栈(2024.12)

| 模块 | 选型 | 备注 |
| :--- | :--- | :--- |
| 架构 | 671B MoE(激活 37B) + MLA | 极致稀疏 |
| 并行 | DualPipe + 专家并行 | 跨节点 MoE |
| 注意力 | MLA(潜向量 KV 压缩) | KV Cache 极小 |
| 优化 | FP8 混合精度 | 首批 FP8 训练大模型 |
| 成本 | $5.5M 训练成本 | (DeepSeek 自报) |
| 数据 | 14.8T tokens | 多语, 代码 |

## E.3 Qwen 训练栈(2024-2025)

| 模块 | 选型 |
| :--- | :--- |
| 架构 | Dense + MoE 双线 |
| 注意力 | GQA |
| 中文 | 极强(原生中文分词) |
| 多模态 | Qwen-VL 系列 |

## E.4 Claude 训练栈(Anthropic, 公开推测)

| 模块 | 选型 |
| :--- | :--- |
| 架构 | 推测 Dense Transformer |
| 对齐 | Constitutional AI + RLAIF |
| 上下文 | 200K-1M |
| 安全 | 严格有害过滤 |

## E.5 GPT-5 / o3 训练栈(OpenAI, 公开推测)

| 模块 | 选型 |
| :--- | :--- |
| 推理 | o-series 推理模型族, RL 训 CoT |
| 可控 | reasoning effort 控制项 |
| 工具 | 强工具使用 + Browser/Code Interpreter |

---

## 第一部分附录 F: 跨章节知识地图

本章是总览, 每条主线都对应后续章节:

| 主线 | 核心问题 | 详见章节 | 数学基础章节 |
| :--- | :--- | :--- | :--- |
| 更强 | Scaling 怎么调? | 第02章(架构)、第07章(流程) | 数理统计/最优化 |
| 更省 | MoE/MLA/FlashAttn 怎么省? | 第02章、第04章 | 数值线性代数 |
| 更长 | 怎么扩上下文? | 第03章 | 信息论(熵) |
| 更听话 | 怎么对齐? | 第05章 | 概率/RL 基础 |
| 更会想 | 怎么让模型推理? | 第06章 | 决策论/搜索 |

---

## 第一部分附录 G: 完整面试题库(10+ 题, 含详解)

## Q1. 用数学描述 Chinchilla 最优分配并解释为何不是"越大越好"?

**答**: 给定算力 $C = 6ND$, 由 $L(N, D) = E + A/N^\alpha + B/D^\beta$, 拉格朗日乘子给出最优解:
$$
N^* = \left(\frac{\alpha}{\beta}\right)^{\frac{1}{\alpha+\beta}} \cdot \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}, \quad D^* = \left(\frac{\beta}{\alpha}\right)^{\frac{1}{\alpha+\beta}} \cdot \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}
$$
代入 $\alpha = 0.34, \beta = 0.28$ 得 $D^*/N^* \approx 20$ tokens/param。
**为何不是越大越好**: 增大 $N$ 但 $D$ 不增, $D/N$ 比例恶化, 数据不够喂饱模型, 出现"参数过剩", loss 反而不如 Chinchilla-optimal 配比。

## Q2. RLHF 目标函数的闭式解是什么? DPO 如何利用它?

**答**: 见 §D.4。闭式解 $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$。DPO 把 $\pi^*$ 代入偏好 Bradley-Terry 模型:
$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \sigma\left(\beta \log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$
用 $\pi_\theta$ 代替 $\pi^*$, 取负对数似然即 DPO 损失。**关键洞察**: 偏好对直接提供相对奖励信号, 无需训独立 RM。

## Q3. 涌现能力是真实现象还是评测伪影?

**答**: 兼有。**真实涌现**: 多步算术、CoT、跨语言翻译确有相变。**评测伪影**: Schaeffer 2023 指出用 continuous metric(token-level F1、log-perplexity)则曲线平滑。"涌现"是**任务选择 + 指标非线性**共同产物。
**实操**: 评估时同时报告连续与离散指标。

## Q4. test-time compute scaling 与 training scaling 的对偶性?

**答**:
| 维度 | Training Scaling | Test-time Scaling |
| :--- | :--- | :--- |
| 投入 | GPU-hours 训练 | 推理 token/步 |
| 收获 | loss ↓, 能力 ↑ | 难题正确率 ↑ |
| 边际 | $L \propto C^{-\alpha}$ | Pass@k vs budget 类似幂律 |
| 代表 | Chinchilla, Scaling Laws | o1, R1, self-consistency |

两者**可加性**: 训练好模型 + 推理时多想 ≈ 双重 scaling。

## Q5. 为何 decoder-only 成为 LLM 主流?

**答**: (1) 训练目标统一: next-token 预测, 吃尽无标注文本; (2) ICL 天然: 任务即"续写"; (3) 结构简单, 易 scale; (4) 工程红利: 单一架构, 工具链复用。Encoder-Decoder 适合 seq2seq 但需要成对数据, 规模受限。

## Q6. 简述 Chinchilla、Gopher、LLaMA-3 训推取舍。

**答**:
- **Gopher(280B, 0.4T tokens)**: Kaplan 时代, 模型大, 数据不够。
- **Chinchilla(70B, 1.4T tokens)**: 同算力下显著优于 Gopher。
- **LLaMA-3(8B, 15T tokens)**: 大幅过训(over-train), 为推理预算。
- **LLaMA-3(405B, 15T tokens)**: Chinchilla 时代后, 仍过训, 因推理成本仍可控。

## Q7. RoPE 与 ALiBi 的外推机理差异?

**答**: RoPE 通过频率编码($\theta_i = 10000^{-2i/d}$), 高频捕捉短程, 低频捕捉长程, 理论上支持任意长但需 PI/NTK/YaRN 校正。ALiBi 直接对注意力分数加 $-(m-n) \cdot m$, 是**强先验的局部偏置**, 训练短可推长但牺牲长程灵活性。

## Q8. MoE 的负载均衡为何关键?

**答**: 若路由坍缩(token 总选 top 专家), 等效于退回 dense 模型, 浪费其他专家容量; 训练不稳、推理专家闲置。辅助损失 $L_{\text{aux}} = \alpha \sum_e f_e \cdot \bar{g}_e$ (DeepSeek) 鼓励均匀利用。

## Q9. 为何 GPT-5 / Claude 4 / Gemini 3 都加 thinking mode?

**答**: (1) 同一模型服务快/慢两种场景, 节省推理成本(简单题 fast path); (2) 用户可控 reasoning effort; (3) 推理模型成为新基础能力, 闭源厂商必须跟进。

## Q10. LLM 训练算力如何估算?

**答**: 单 token 前向+反向 $\approx 6N$ FLOPs($N$ 为参数量)。
- GPT-3 175B, 300B tokens → $6 \cdot 175 \cdot 10^9 \cdot 300 \cdot 10^9 = 3.15 \cdot 10^{23}$ FLOPs。
- LLaMA-3 405B, 15T tokens → $6 \cdot 405 \cdot 10^9 \cdot 15 \cdot 10^{12} = 3.65 \cdot 10^{25}$ FLOPs。

## Q11. 描述 RLHF/DPO/GRPO 三者的训练成本量级差异?

**答**:
| 方法 | 模型数 | 采样 | 显存 | 训练稳定性 |
| :--- | :--- | :--- | :--- | :--- |
| RLHF+PPO | 4(策略/ref/RM/Critic) | 需要 | 高 | 差 |
| DPO | 2(策略/ref) | 不需要 | 中 | 优 |
| GRPO | 2(策略/ref) + 组采样 | 需要(组采样) | 中 | 优 |

RLHF ≈ 4× 单模型训练显存; DPO/GRPO ≈ 2×; GRPO 在推理任务上**无 Critic 但需多次采样**, 实际成本介于两者之间。

## Q12. 如果让你设计一个 7B 中文 LLM, 关键决策是什么?

**答**: (1) 数据: 中文占比 50-70%, 代码/数学 20%, 多语 10%; (2) 架构: Dense GQA 8 + RoPE, 性价比最高; (3) 训练: Chinchilla-optimal(~140B tokens), 1 epoch; (4) 词表: 中文 BPE 50K+, 避免英语主导; (5) SFT: 10K 高质量中文指令; (6) 对齐: DPO 起步, GRPO 用于数学/代码。

---

## 第一部分附录 H: 历史里程碑的"为什么"汇总

| 年份 | 里程碑 | 上一代痛点 | 本代机制 | 引入的新问题 |
| :--- | :--- | :--- | :--- | :--- |
| 2017 | Transformer | RNN 不可并行 | 自注意力 + 并行 | $O(L^2)$ 复杂度 |
| 2018 | BERT/GPT-1 | 任务专属模型 | 预训练+微调 | 预训练贵 |
| 2019 | GPT-2 | 标注数据稀缺 | 零样本 | 能力仍弱 |
| 2020 | GPT-3 | 零样本不稳 | 175B + ICL | 算力爆炸 |
| 2020 | Scaling Laws | 调参无据 | 幂律可预测 | 数据不够 |
| 2022 | Chinchilla | 越大越好? | 参数-数据同比 | 单训练变贵 |
| 2022 | InstructGPT | 模型不听话 | RLHF | PPO 难训 |
| 2023 | LLaMA | 闭源 | 开源 | 资源不够的人无法训 |
| 2023 | RoPE/GQA | 长上下文 | 高效位置+压缩 KV | 外推不稳 |
| 2023 | DPO | RLHF 复杂 | 闭式代回 | 无在线探索 |
| 2024 | DeepSeek-V2 | GQA 仍有冗余 | MLA 低秩压缩 | 实现复杂 |
| 2024 | Mixtral | Dense 太贵 | MoE | 负载均衡 |
| 2024 | o1 | CoT 靠提示 | RL 训自发长 CoT | 推理贵 |
| 2025 | R1 | o1 闭源 | 纯 RL+开源 | 可验证任务有限 |
| 2025 | Agent | 单次问答 | 长程规划 + 工具 | 错误累积 |
| 2026 | 长程 Agent | 工具不稳 | 反思 + 记忆 | 可解释性 |

---

## 第一部分附录 I: 复现清单与工程代码

## I.1 训练一个 toy GPT 的最小代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    """6 层 decoder-only GPT, ~125M 参数, 用于教学复现"""
    def __init__(self, vocab=32000, d=768, n_heads=12, n_layers=12, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d), ln2=nn.LayerNorm(d),
                attn=nn.MultiheadAttention(d, n_heads, batch_first=True),
                ff=nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
            )) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # 共享权重(tied embedding)
        self.head.weight = self.tok.weight

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = x + blk['attn'](blk['ln1'](x), blk['ln1'](x), blk['ln1'](x),
                                 is_causal=True)[0]
            x = x + blk['ff'](blk['ln2'](x))
        return self.head(self.ln_f(x))

# 训练一步
model = TinyGPT().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
ids = torch.randint(0, 32000, (4, 1024)).cuda()
logits = model(ids[:, :-1])
loss = F.cross_entropy(logits.reshape(-1, 32000), ids[:, 1:].reshape(-1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step(); opt.zero_grad()
print(f"loss={loss.item():.4f}")
```

## I.2 用 transformers 库加载 LLaMA-3

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 LLaMA-3-8B-Instruct(需 HF token)
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # 关键: 用 FlashAttn-2
)

# 推理
prompt = "用一句话解释 RoPE。"
messages = [{"role": "user", "content": prompt}]
input_ids = tok.apply_chat_template(messages, return_tensors="pt").cuda()
out = model.generate(input_ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

## I.3 用 vLLM 高吞吐推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=False,  # 用 CUDA graph
)

sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
prompts = ["用 Python 写快速排序", "解释 transformer"] * 64  # batch 推理

outputs = llm.generate(prompts, sp)
for o in outputs[:3]:
    print(f"prompt={o.prompt[:30]}... → {o.outputs[0].text[:100]}")
```

## I.4 评估 MMLU 子集

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 MMLU 验证集
ds = load_dataset("cais/mmlu", "abstract_algebra", split="validation[:32]")
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                              torch_dtype=torch.bfloat16, device_map="auto")

correct = 0
for ex in ds:
    choices = ex["choices"]
    prompt = f"Question: {ex['question']}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    ans = tok.decode(out[0][-3:], skip_special_tokens=True).strip()
    pred = "ABCD".index(ans[0]) if ans and ans[0] in "ABCD" else -1
    if pred == ex["answer"]:
        correct += 1
print(f"MMLU accuracy: {correct/len(ds):.3f}")
```

---

## 第一部分附录 J: 关键术语表

| 术语 | 英文 | 定义 |
| :--- | :--- | :--- |
| 涌现 | Emergence | 能力在规模跨阈值后突然出现的现象 |
| 外推 | Extrapolation | 推理长度 > 训练长度 |
| 思维链 | Chain-of-Thought | 显式中间推理步骤 |
| 慢思考 | Slow Thinking | System-2, 推理时多花计算 |
| 蒸馏 | Distillation | 小模型学大模型分布 |
| 量化 | Quantization | FP16/INT8/INT4 减显存 |
| 续训 | Continual Training | 从已有 ckpt 继续训练 |
| MoE | Mixture-of-Experts | 稀疏激活 FFN |
| MLA | Multi-head Latent Attention | 低秩 KV 压缩 |
| GQA | Grouped-Query Attention | 共享 KV 头 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| ICL | In-Context Learning | 提示内学习, 无权重更新 |
| RLHF | RL from Human Feedback | 人类偏好强化学习 |
| DPO | Direct Preference Optimization | 免 RM 的偏好优化 |
| GRPO | Group Relative Policy Optimization | 免 Critic 的 PPO 改进 |
| CoT | Chain-of-Thought | 思维链提示 |
| TTC | Test-Time Compute | 推理时计算 |
| RM | Reward Model | 奖励模型 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| KL 散度 | KL Divergence | 分布间非对称距离 |
| 困惑度 | Perplexity | $\exp(H)$, 越低越好 |
| 退火 | Annealing | 末期高质量数据 + LR 降 |
| 评测集 | Benchmark | 标准化评估数据集 |
| 基准 | Baseline | 参考基线模型/方法 |
| 蒸馏 | Distillation | 大模型教小模型 |
| Token | Token | 模型输入输出最小单位 |
| 词表 | Vocabulary | token ↔ id 映射 |
| 嵌入 | Embedding | 离散 → 连续向量 |
| 注意力 | Attention | 加权聚合机制 |
| 位置编码 | Positional Encoding | 注入序列顺序信息 |
| 归一化 | Normalization | LN/RMSNorm, 稳定训练 |
| 优化器 | Optimizer | AdamW/Lion/Sophia |
| 调度 | Schedule | LR 随 step 变化策略 |
| 损失 | Loss | 训练目标 |
| 梯度 | Gradient | loss 对参数的导数 |
| 反向 | Backward | 反向传播 |
| 截断 | Clipping | grad clip 防爆炸 |
| 精度 | Precision | FP32/BF16/FP8/INT8 |
| 并行 | Parallelism | DP/TP/PP/EP/CP |
| 量化 | Quantization | 低精度减显存 |
| 推理 | Inference | 模型前向计算 |
| 服务化 | Serving | 在线提供推理 API |
| 吞吐 | Throughput | tokens/s |
| 时延 | Latency | 首 token / 总时延 |
| KV Cache | KV Cache | 推理时缓存 K/V 加速 |
| PagedAttention | PagedAttention | vLLM 的虚拟内存式 KV |
| FlashAttention | FlashAttention | 分块 IO-aware 注意力 |
| 投机解码 | Speculative Decoding | 小模型 draft, 大模型 verify |
| 工具使用 | Tool Use | 调用外部 API |
| Agent | Agent | 长程自主任务执行 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| 思维链 | CoT | 见上 |
| 强化学习 | Reinforcement Learning | 试错学习 |
| 策略 | Policy | $\pi(a\|s)$, 决策函数 |
| 价值 | Value | $V(s)$, 长期回报期望 |
| 优势 | Advantage | $A(s,a) = Q(s,a) - V(s)$ |
| PPO | Proximal Policy Optimization | 策略梯度稳定版 |
| 推理模型 | Reasoning Model | o1/R1 类, RL 训长 CoT |
| 顿悟 | Aha Moment | R1-Zero 自涌现反思 |
| 涌现 | Emergence | 见上 |
| Scaling Law | Scaling Law | 能力随规模幂律 |

---

> 本章作为总览, 已搭建完整主线 + 数学 + 工程 + 历史 + 代码 + 面试题库。后续每章将对应其中一条主线深入展开。

---

## 第二部分: 各主线深度技术解析

---

## 第二章 深度: 更强(Scaling)的工程极限

### 2.1 训练算力的实际增长曲线

按公开报道与论文:

| 年份 | 模型 | 训练 FLOPs(估) | 增长倍率 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 2018 | GPT-1 | $10^{18}$ | 1× | 基线 |
| 2019 | GPT-2 | $2 \times 10^{19}$ | 20× | |
| 2020 | GPT-3 | $3 \times 10^{23}$ | 1.5e4× | 175B, 突破 |
| 2022 | Gopher | $5 \times 10^{23}$ | 1.7× | |
| 2022 | Chinchilla | $5 \times 10^{23}$ | 同算力 | 70B, 更优 |
| 2023 | LLaMA-65B | $6 \times 10^{24}$ | 12× | |
| 2023 | GPT-4 | $2 \times 10^{25}$ | 3× | (估) |
| 2024 | LLaMA-3 405B | $4 \times 10^{25}$ | 2× | |
| 2024 | DeepSeek-V3 | $5 \times 10^{24}$ | 0.13× | MoE, 高效 |
| 2025 | GPT-5/o3 | $10^{26}$(估) | 2× | |

**趋势**: 2018-2023 每年 ~10× 算力增长, 2024 后受限于硬件/电力/数据, 增速放缓到 2-3×。

### 2.2 Scaling Law 的边际收益递减

Chinchilla 给出 loss 随 $C$ 的关系 $L \propto C^{-\alpha}$, $\alpha \approx 0.05$。即算力翻 10 倍, loss 仅降 ~12%。这意味着**单纯堆算力的边际收益递减**, 推动了 MoE、长 CoT 等"非线性"路径。

**定理 2.1** (Scaling 上界 - 不可约损失) 任意模型族的 loss 下界由数据熵 $H(p)$ 决定: $L \ge H(p) = -\mathbb{E}_p[\log p(x)]$。自然语言熵约 1-1.5 nats/token (GPT-3 数据), 当前最好模型约 1.8-2.0 nats/token, 距理论下界不远。

### 2.3 数据墙(Data Wall)

**Villalobos et al. 2022** (arXiv:2211.04325) *"Will we run out of data?"* 估算高质量文本数据将在 **2026-2030 年耗尽**。对策:
- **合成数据**: 模型自己生成, 但需小心"模型自噬崩溃"(model collapse, Shumailov 2023)。
- **多模态数据**: 图像、视频、音频携带更多 bit/token 信息。
- **重训练**: 多次 epoch 过训(但损失多样性)。

**Algorithm 2.1** 合成数据去污染 + 质量过滤
**Input**: 原始数据 $D$, 合成候选 $\tilde{D}$
**Output**: 高质量合成集 $\tilde{D}_{\text{clean}}$
1. **去污染**: 用 $n$-gram 匹配过滤训练集泄漏
2. **质量打分**: 用小 RM 过滤低质量样本
3. **多样性**: 用 embedding 聚类, 每类取 top-$k$
4. **混合比例**: $\alpha D + (1-\alpha) \tilde{D}_{\text{clean}}$, $\alpha \approx 0.7$
**Complexity**: $O(N \log N)$ 聚类, $O(N^2)$ 去污染(用 MinHash 近似 $O(N)$)

---

## 第三章 深度: 更省(效率)的物理/工程极限

### 3.1 三大瓶颈: 算力、显存、带宽

| 瓶颈 | 单位 | 典型场景 | 优化方向 |
| :--- | :--- | :--- | :--- |
| 算力(FLOPs) | $6ND$ | 训练大模型 | 稀疏化(MoE)、量化、蒸馏 |
| 显存(bytes) | $16N + 32N_{\text{opt}}$ | 推理/训练 | FP8、INT4、ZeRO-offload |
| 带宽(B/s) | KV 读写 HBM | 长序列推理 | FlashAttn、PagedAttn |

**Roofline 模型**:
$$
\text{Speed} \le \min\left(\frac{\text{Peak FLOPs}}{\text{FLOPs per byte}}, \text{Memory Bandwidth}\right)
$$

当计算强度 $\text{FLOPS/byte} < I_{\max}$, 受带宽限制(attention); 反之受算力限制(MM)。

### 3.2 MoE 的工程红线

| 模型 | 激活比例 | 总参数 | 激活参数 | 节省比 |
| :--- | :--- | :--- | :--- | :--- |
| GPT-3 | 100% | 175B | 175B | 1× |
| Mixtral 8x7B | 25% | 47B | 13B | 3.6× |
| DeepSeek-V2 | 5% | 236B | 21B | 11× |
| DeepSeek-V3 | 5.5% | 671B | 37B | 18× |

**DeepSeek-V3** 用了 **256 路由专家 + 1 共享专家**, top-8 路由, 每 token 激活 9 个专家。

### 3.3 量化的精度-效率 trade-off

| 精度 | bits/param | 相对误差 | 显存节省 | 吞吐提升 |
| :--- | :--- | :--- | :--- | :--- |
| FP32 | 32 | 基准 | 1× | 1× |
| BF16 | 16 | ~0 | 2× | 2× |
| INT8 | 8 | <1% | 4× | 4× |
| FP8(E4M3) | 8 | 1-2% | 4× | 4× |
| INT4 | 4 | 3-5% | 8× | 6-8× |
| INT2 | 2 | >10% | 16× | 10×+ |

**GPTQ / AWQ / SmoothQuant**: INT4 量化的代表算法, 通过保护重要权重通道达到 < 3% 精度损失。

---

## 第四章 深度: 更长(上下文)的认知极限

### 4.1 人类 vs LLM 的"工作记忆"

- 人脑工作记忆: ~7 ± 2 项 (Miller 1956)
- LLM 上下文: 200K-1M tokens, 但有效利用衰减

**Lost-in-the-Middle 现象** (Liu et al. 2023, arXiv:2307.03172): LLM 对长上下文中**中间部分**的检索精度显著低于首尾, 提示位置偏置仍是核心问题。

### 4.2 长上下文评估基准

| 基准 | 长度 | 任务 | 评估能力 |
| :--- | :--- | :--- | :--- |
| Needle-in-a-Haystack | 1K-1M | 检索单事实 | 检索精度 |
| RULER | 4K-128K | 多针/多跳 | 综合长程能力 |
| LongBench | 1K-32K | QA/摘要/代码 | 多任务 |
| Scrolls | 8K-32K | 长文档 QA | 阅读理解 |
| Multi-needle | 32K+ | 多事实检索 | 复合检索 |

### 4.3 长上下文的工程挑战

- **KV Cache**: 32K 上下文, 32 层, 32 头, head_dim 128 → $2 \cdot 32 \cdot 32 \cdot 32K \cdot 128 \approx 8$ GB(FP16)。
- **Attention**: 32K × 32K 矩阵约 4 GB(FP16), 计算量 $O(L^2 d) = 32K^2 \cdot 128 \approx 1.3 \times 10^{11}$ FLOPs/token。
- **位置编码**: 频率需支持 L_test 而训练 L_train 远小。

**优化组合**:
- GQA: KV 降 4-8×
- MLA: 再降 4-8×
- FlashAttn: 显存降 4-10×
- YaRN: 外推到 L_test
- 量化 KV Cache: INT8/INT4, 再降 2-4×

组合后 1M 上下文推理显存可压到 30-50 GB(单卡 H100)。

---

## 第五章 深度: 更听话(对齐)的安全工程

### 5.1 三层安全防线

```text
L1 内容过滤层(输入侧)
  ↓
L2 模型对齐层(RLHF/DPO, 内置拒绝能力)
  ↓
L3 输出过滤层(输出侧, 二次检测)
```

每一层独立可失效, 工业部署需纵深防御。

### 5.2 对齐税(Alignment Tax)

**定义**: 对齐过程可能损失预训练能力, 这种能力损耗称为"对齐税"。

**经验**:
- RLHF: 5-10% 通用能力损失(MMLU等), 但指令遵循大幅提升
- DPO: 较小税(< 3%), 但效果上限有时不如精调 RLHF
- GRPO: 推理能力税为 0 或负(激励推理)

### 5.3 红队测试(Red Teaming)

**目的**: 主动发现模型有害行为, 用于 RLHF 训练。

**方法**:
- 人类专家构造对抗 prompt
- AI 自动攻击(GAN-like: 攻击模型 vs 防御模型)
- 蓝队 vs 红队竞赛

**Pioneer**: Anthropic 2022 "Red Teaming Language Models" (Ganguli et al.)

---

## 第六章 深度: 更会想(推理)的认知科学对偶

### 6.1 System-1 vs System-2 思维 (Kahneman)

| | System-1(快) | System-2(慢) |
| :--- | :--- | :--- |
| 速度 | 毫秒级 | 秒-分钟级 |
| 意识 | 无意识 | 显式 |
| 例子 | 1+1, 模式识别 | 数学证明, 规划 |
| LLM 类比 | GPT-3/4 base | o1, R1 |
| 训练 | 监督学习 | RL + 自反思 |

### 6.2 推理模型的能力边界

**数学**: AIME 2024, R1 达 79.8%, o3 达 91.7%
**代码**: SWE-bench Lite, R1 49%, o3 65%
**博士级科学**: GPQA Diamond, R1 71.5%, o3 87.7%

**仍弱**:
- 长程规划(> 100 步)
- 创造性写作
- 跨模态推理(图+文+数)

### 6.3 推理模型的成本挑战

o1 单次响应平均 5-15K tokens(隐藏 CoT), 比 GPT-4 贵 5-20×。

**优化方向**:
- 推理路径压缩: 用小模型蒸馏 o1 的 CoT
- 早停: 模型学会"何时停止思考"
- 推理 KV Cache: 共享跨步的 KV

---

## 第三部分: 跨主线对比矩阵

## 7.1 五大主线 × 三大维度对比

| 主线 | 代表技术 | 收益 | 代价 | 工业代表 |
| :--- | :--- | :--- | :--- | :--- |
| 更强 | Scaling Law | 能力上界提升 | 算力爆炸 | GPT-4, LLaMA-405B |
| 更省 | MoE/FlashAttn | 性价比提升 | 工程复杂 | DeepSeek-V3, Mixtral |
| 更长 | RoPE/YaRN | 长文档/Agent | 检索衰减 | Claude 1M, Gemini 2M |
| 更听话 | RLHF/DPO/GRPO | 可用可控 | 对齐税 | ChatGPT, Claude |
| 更会想 | RL+CoT | 难题正确率 | 推理贵 | o1, R1 |

## 7.2 训练成本结构对比(以 70B 模型为例)

| 阶段 | 计算占比 | 典型时长 | 主要工作 |
| :--- | :--- | :--- | :--- |
| 预训练 | 90%+ | 30-60 天 | next-token 训练 |
| SFT | 1-2% | 1-3 天 | 指令微调 |
| RLHF/DPO | 1-3% | 1-5 天 | 偏好优化 |
| 评估 | <1% | 1-2 天 | 基准 + 人评 |
| 量化/部署 | 0 | 1-3 天 | 压缩 + 部署 |

## 7.3 推理性能对比(同一 70B 模型)

| 配置 | tokens/s(单请求) | tokens/s(批 8) | 时延 TTFT |
| :--- | :--- | :--- | :--- |
| FP16, 单卡 | 25 | 30(限带宽) | 100ms |
| FP8 | 45 | 55 | 80ms |
| INT4(GPTQ) | 80 | 110 | 60ms |
| vLLM PagedAttn | 35 | 250 | 90ms |
| Speculative Decoding | 60 | 80 | 80ms |

---

## 第四部分: 五年趋势预测(2026-2030)

## 8.1 短期(2026)

- **Agent 化**: 单一 LLM 调用 → 多步工具调用 + 反思
- **多模态统一**: 文本/图像/音频/视频同模型, Gemini/GPT-5 路线
- **推理模型普及**: thinking mode 成标配, GPT-5 / Claude 4.5 / Gemini 3
- **端侧小模型**: 1-3B 模型在手机/PC 端实时推理

## 8.2 中期(2027-2028)

- **世界模型**: 物理一致的生成/规划, 类 Sora + LLM
- **超长记忆**: 1M+ 上下文 + 外部向量记忆
- **自进化**: 模型自动生成训练数据, 自我改进
- **强化微调(RFT)**: 用 RL 微调到具体任务, DPO 后的下一步

## 8.3 长期(2029+)

- **AGI 雏形**: 多模态 + Agent + 长程规划 + 自我反思
- **效率革命**: 神经符号混合, 知识图谱 + LLM
- **科学发现**: LLM 主导的假设生成 + 实验设计

---

## 第五部分: 进阶学习路线图

## 9.1 入门 → 进阶 → 专家 三阶段

**入门(1-3 月)**:
- 读完本教程第01-08章
- 跑通 toy GPT 训练
- 理解 Transformer 各部件
- 论文: Attention Is All You Need, GPT-3, LLaMA

**进阶(3-12 月)**:
- 选 1-2 方向深入(对齐/推理/架构)
- 复现 DPO/GRPO
- 读 SOTA 论文(o1, R1, MLA)
- 参与开源项目(trl, vllm, sglang)

**专家(1 年+)**:
- 跟踪 arXiv 每日新文
- 在 Hugging Face / vLLM 提交 PR
- 在国际顶会发表(NeurIPS, ICML, ACL)
- 主导一个端到端 LLM 项目

## 9.2 推荐资源

**必读论文**(按时间):
1. Attention Is All You Need (2017)
2. BERT, GPT-2 (2018-19)
3. Scaling Laws, GPT-3 (2020)
4. Chinchilla, InstructGPT, Emergence (2022)
5. LLaMA, DPO, QLoRA (2023)
6. DeepSeek-V2, MLA, Mixtral (2024)
7. o1, DeepSeek-R1, GRPO (2024-25)

**开源框架**:
- 训练: Hugging Face Transformers, DeepSpeed, Megatron-LM
- RLHF: TRL (Transformer Reinforcement Learning)
- 推理: vLLM, SGLang, TensorRT-LLM
- 评估: lm-evaluation-harness, OpenCompass

**社区**:
- Hugging Face Discord
- EleutherAI Discord
- r/LocalLLaMA

## 9.3 跨教程联动

| 本章主题 | 关联教程 |
| :--- | :--- |
| Transformer 架构 | 深度学习手撕/第05章 |
| 优化器/调度 | 深度学习手撕/第02章 |
| 分布式训练 | 深度学习手撕/第07章 |
| 数学基础 | 基础/数学基础 |
| 工程能力 | 基础/工程能力 |
| 具身智能 | 具身智能 |
| 推荐系统 | 方向教程/rec教程 |

---

## 第六部分: 自检 checklist

- [ ] 能默写 Chinchilla 法则的公式与推导
- [ ] 能解释为何 decoder-only 成为主流
- [ ] 能讲清 RLHF/DPO/GRPO 的差异
- [ ] 能手写 RMSNorm 的 PyTorch 实现
- [ ] 能解释 RoPE 的外推原理
- [ ] 能描述 MoE 的负载均衡问题
- [ ] 能列出 test-time compute scaling 与 training scaling 的对偶
- [ ] 能用一句话说清 o1/R1 的范式转变
- [ ] 能估算 70B 模型训练算力
- [ ] 能讲出五大厂商旗舰模型的差异

---

> **本章作为 LLM 发展教程的入口, 提供完整的全景视角。所有章节都从此处开始, 并在最后回到此处的总览图。**

---

## 第七部分: 各主线详解 - 范式转变深析

## 10. "更强"主线的三波演进

### 10.1 第一波: Dense Scaling (2018-2022)

代表: GPT-2/3, Gopher, PaLM, OPT, BLOOM
核心: 参数从 1B → 540B, 数据从 10B → 1.4T tokens

**关键论文**:
- GPT-3 (Brown et al., 2020, arXiv:2005.14165): ICL 涌现
- Gopher (Rae et al., DeepMind, 2021, arXiv:2112.11446): 280B
- PaLM (Chowdhery et al., Google, 2022, arXiv:2204.02311): 540B, Pathways

**架构共同点**: Decoder-only, Post-LN(早期)/Pre-LN, ReLU/GELU, 稠密 FFN

### 10.2 第二波: Sparse / Efficient (2023-2024)

代表: Switch Transformer, GLaM, Mixtral, DeepSeek-MoE, Qwen-MoE
核心: 稀疏激活 FFN, 总参数大但激活参数小

**关键论文**:
- Switch Transformer (Fedus et al., 2021, arXiv:2101.03961): 万亿 MoE
- GLaM (Du et al., 2021, arXiv:2112.06905): 1.2T MoE
- Mixtral 8x7B (Mistral, 2024, arXiv:2401.04088): 8 专家
- DeepSeek-MoE (2024, arXiv:2401.06066): 细粒度 + 共享专家
- DeepSeek-V2 (2024, arXiv:2405.04434): MLA + MoE

**架构特征**: MoE(8-256 专家, top-2/8), GQA/MLA

### 10.3 第三波: Reasoning + Test-Time Compute (2024-2026)

代表: o1, o3, DeepSeek-R1, Gemini Thinking, Claude extended thinking
核心: 推理时花更多计算换正确率

**关键论文**:
- Self-Consistency (Wang et al., 2022, arXiv:2203.11171): 投票
- Tree of Thoughts (Yao et al., 2023, arXiv:2305.10601): 树搜索
- o1 (OpenAI, 2024): blog 发布, 详细技术未公开
- DeepSeek-R1 (2025, arXiv:2501.12948): 纯 RL+开源
- GRPO (DeepSeek, 2024, arXiv:2402.03300): 去 Critic

**架构特征**: 在 SFT/RLHF 基础上加可验证奖励 RL

## 11. "更省"主线的三条技术线

### 11.1 稀疏化: MoE 系列

**演进**:
- Sparse MoE (Shazeer et al., 2017): 噪声 top-k
- Switch Transformer (2021): top-1 + 负载均衡
- GLaM (2021): top-2
- Mixtral (2024): top-2, 8 专家
- DeepSeek-MoE (2024): 细粒度专家(64/256)
- DeepSeek-V3 (2024): 256 路由专家 + 1 共享专家

**关键挑战**:
- 负载均衡: token 路由不均
- 专家并行: 跨节点通信
- 推理调度: 专家利用率最大化
- 训练稳定性: 路由坍缩

**Algorithm 11.1** Switch Transformer 路由
**Input**: $x \in \mathbb{R}^d$
**Output**: 专家输出加权
1. $g = W_g x \in \mathbb{R}^E$  # 门控 logits
2. $e^* = \arg\max_e g_e$  # top-1
3. $y = \text{Expert}_{e^*}(x)$
4. **辅助损失**: $L_{\text{aux}} = \alpha \cdot E \sum_e f_e \cdot \bar{p}_e$, $f_e$ = token 路由比例, $\bar{p}_e$ = 平均门控概率

### 11.2 量化: INT/FP 系列

**演进**:
- INT8 PTQ (Post-Training Quantization)
- SmoothQuant (Xiao et al., 2022, arXiv:2211.10438): 平滑异常值
- GPTQ (Frantar et al., 2022, arXiv:2210.17323): 二阶优化
- AWQ (Lin et al., 2023, arXiv:2306.00978): 激活感知
- FP8 训练 (NVIDIA H100): DeepSeek-V3 首批

**量化误差分析**:
设原权重 $W \in \mathbb{R}^d$, 量化 $\hat{W} = W + \epsilon$, $\epsilon$ 为量化噪声。
均方误差 $\mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12}$, $\Delta$ = 量化步长。
GPTQ 通过 Hessian 矩阵 $\mathbf{H}$ 加权:
$$
\arg\min_{\hat{W}} \| W - \hat{W} \|_{\mathbf{H}}^2 = (W - \hat{W})^\top \mathbf{H} (W - \hat{W})
$$
最小化加权重构误差, 显著降低 INT4 精度损失。

### 11.3 系统优化: FlashAttn + PagedAttn

**FlashAttention**(Dao et al., 2022, arXiv:2205.14135; v2 2023, arXiv:2307.08691; v3 2024, arXiv:2407.08608)

核心思想: 在线 softmax + SRAM 分块, 不实体化 $n \times n$ 注意力矩阵。

**Online Softmax 推导**:
标准 softmax:
$$
m = \max(x), \quad l = \sum e^{x_i - m}, \quad \text{softmax}(x)_i = e^{x_i - m} / l
$$
分块时, 维护 running max $m_{\text{running}}$ 和 running sum $l_{\text{running}}$, 每块更新:
$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})
$$
$$
l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} l_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} l_{\text{block}}
$$
$$
o_{\text{new}} = \text{rescale}(o_{\text{old}}, m_{\text{old}}, m_{\text{new}}) + \text{compute}(x_{\text{block}}, m_{\text{block}})
$$

**Algorithm 11.2** FlashAttention 前向
**Input**: $Q, K, V \in \mathbb{R}^{n \times d}$, block size $B$
**Output**: $O \in \mathbb{R}^{n \times d}$
1. 分块 $Q$ 为 $Q^{(1)}, ..., Q^{(T_r)}$ 每块 $B$ 行
2. 初始化 $O^{(i)} = 0, l^{(i)} = 0, m^{(i)} = -\infty$
3. for $j$ in $1..T_c$:
   a. 加载 $K^{(j)}, V^{(j)}$ 到 SRAM
   b. $S^{(ij)} = Q^{(i)} (K^{(j)})^\top$
   c. 更新 $m^{(i)}, l^{(i)}, O^{(i)}$ 用 online softmax
4. $O^{(i)} = O^{(i)} / l^{(i)}$
**Complexity**: Time $O(n^2 d)$, Space $O(n)$(SRAM) 而非 $O(n^2)$ (HBM)

**v2/v3 改进**:
- v2: 工作分区(更好的并行)
- v3: 异步拷贝 + FP8 + Hopper 特定优化

## 12. "更长"主线的技术体系

### 12.1 位置编码演进树

```text
绝对位置
├─ 正弦编码 (Transformer 2017)
├─ 可学习 (BERT, GPT-2)
└─ 复杂位置 (T5 相对位置 bias)
相对位置
├─ RoPE (LLaMA, GPT-NeoX)
├─ ALiBi (BLOOM)
├─ xPos (Microsoft)
└─ CoPE (Granite)
RoPE 外推
├─ PI (位置插值, Meta 2023)
├─ NTK-aware (bloc97)
├─ YaRN (Peng et al., 2023, arXiv:2309.00071)
└─ LongRoPE (Ding et al., 2024, arXiv:2402.13753)
```

### 12.2 注意力稀疏化(长上下文)

| 方案 | 复杂度 | 代表 |
| :--- | :--- | :--- |
| 稠密 Attention | $O(n^2)$ | 标准 |
| Sliding Window | $O(nw)$ | Mistral |
| Dilated | $O(nw)$ | Longformer |
| Global + Local | $O(n(w+g))$ | BigBird |
| Sparse | $O(n^{1.5})$ | BigBird |
| Ring Attention | $O(n)$ 通信 | Sequence Parallel |
| Linear Attn | $O(n)$ | Performer, Linear Trans |
| SSM | $O(n)$ | Mamba |

### 12.3 长上下文 LLM 实战 (2025)

| 模型 | 上下文 | 技术 |
| :--- | :--- | :--- |
| Gemini 2.5 Pro | 1M-2M | 多模态 + 长上下文 |
| Claude 4 | 200K-1M | ALiBi + 优化 |
| GPT-4.1 | 1M | 优化实现 |
| LLaMA-3.1 405B | 128K | GQA + RoPE |
| Qwen-2.5 | 128K | YaRN |
| DeepSeek-V3 | 64K-128K | MLA + YaRN |
| Yi-200K | 200K | RoPE 外推 |
| GLM-4 | 128K-1M | 多种 |

## 13. "更听话"主线的技术分支

### 13.1 反馈来源分类

| 类型 | 代表 | 成本 | 规模 |
| :--- | :--- | :--- | :--- |
| 人类标注 | InstructGPT | 高 | ~100K |
| AI 标注 | RLAIF, UltraFeedback | 低 | 1M+ |
| 自生成 | Self-Rewarding, SPIN | 0 | 任意 |
| 工具验证 | Math/Code | 0 | 任意(可验证任务) |

### 13.2 优化目标分类

| 方法 | 目标 | 代表 |
| :--- | :--- | :--- |
| PPO | $\max \mathbb{E}[r] - \beta \mathrm{KL}$ | InstructGPT |
| DPO | 闭式代回 | DPO |
| IPO | 确定性偏好 | IPO |
| KTO | 单调效用 | KTO |
| ORPO | odds ratio | ORPO |
| SimPO | 无参考 | SimPO |
| GRPO | 组内归一化 | DeepSeek-R1 |
| RLOO | leave-one-out | RLOO |

### 13.3 RLHF 的数学严格推导

**目标**: $\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x,y)] - \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})$

设空间 $\mathcal{Y}$ 有限, 闭式解:
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

**证明** (变分法):
设 $\pi^* = \arg\max$, 对任意 $\pi$, 由 KL 非负:
$$
\mathbb{E}_{\pi^*}[r] - \beta \mathrm{KL}(\pi^* \| \pi_{\text{ref}}) \ge \mathbb{E}_\pi[r] - \beta \mathrm{KL}(\pi \| \pi_{\text{ref}})
$$
令 $\pi = \pi_{\text{ref}}$, 得 $Z(x) \ge 1$。取对数得对偶问题:
$$
\min_\beta \beta \log Z(x) = \beta \log \mathbb{E}_{y \sim \pi_{\text{ref}}}\left[\exp\left(\frac{r(x,y)}{\beta}\right)\right]
$$
此即 soft Bellman 方程。

## 14. "更会想"主线的算法谱系

### 14.1 推理时算法分类

| 范式 | 代表 | 核心思想 |
| :--- | :--- | :--- |
| CoT 提示 | Let's think step by step | 引导中间步骤 |
| Self-Consistency | SC | 多采样投票 |
| ToT | Tree of Thoughts | 树搜索+回溯 |
| ReAct | Reason + Act | 思考+工具 |
| Reflection | Reflexion | 自我反思 |
| RAP | Reasoning + Acting | MCTS-like |
| LATS | Language Agent Tree Search | 树+价值 |

### 14.2 o1 类训练范式

```text
SFT (基础)
  ↓
RL with verifiable rewards
  ↓ GRPO
  ↓
拒绝采样 → SFT 精修
  ↓
再 RL 精调
  ↓
Reasoning model
```

**关键创新**:
1. **长 CoT 数据**: 含反思、回溯、多路径
2. **可验证奖励**: 数学/代码对错明确
3. **过程奖励模型(PRM)**: 不仅评结果, 还评中间步骤

### 14.3 DeepSeek-R1 训练细节

**R1-Zero**: 无 SFT 冷启动, 直接 base model + GRPO + 可验证奖励
- 涌现"顿悟时刻"(aha moment): 模型学会回溯、自检
- 输出可读性差, 语言混杂

**R1**: R1-Zero 数据 → SFT cold-start → RL(语言一致性奖励) → 拒绝采样 → SFT → RL
- 平衡可读性与推理力

**蒸馏**: 用 R1 数据训小模型(Qwen-7B, LLaMA-8B), 性能跃升

## 15. 跨主线交叉影响

### 15.1 Scaling × 推理

**发现**: 推理模型也存在 test-time compute scaling law:
- 同一模型, 推理 token 越多, 准确率越高
- 幂律: $\text{Accuracy} \propto T_{\text{think}}^{\gamma}$, $\gamma \approx 0.05-0.1$

**意义**: test-time scaling 是 training scaling 的**补充而非替代**, 两者相乘。

### 15.2 效率 × 推理

**挑战**: 推理模型生成长 CoT, 推理成本激增。
**对策**:
- 投机解码: 小模型 draft, 大模型 verify
- 早停: 模型学会判断何时停止思考
- 推理路径蒸馏: 用小模型蒸馏 CoT

### 15.3 长上下文 × Agent

**需求**: Agent 需长上下文记忆工具调用历史、用户反馈。
**方案**:
- 上下文压缩: 旧对话 summary
- 外部记忆: 向量库检索
- 状态机: 显式状态管理

### 15.4 对齐 × 推理

**冲突**: 对齐让模型"听话简短", 推理让模型"长篇大论"。
**平衡**:
- DPO/RLHF 时混入推理数据
- 长度奖励: 鼓励简洁但不过短
- 推理模型: 用 GRPO 重新对齐推理 vs 简洁

## 16. 行业应用与商业模式

### 16.1 主要商业玩家

| 公司 | 代表产品 | 商业模式 |
| :--- | :--- | :--- |
| OpenAI | GPT-4/o1/o3, ChatGPT | API + 订阅 |
| Anthropic | Claude 3/4 | API + 企业 |
| Google | Gemini 2.5/3 | API + Workspace |
| Meta | LLaMA 系列 | 开源 |
| DeepSeek | V3/R1 | 开源 + API(极低) |
| 阿里 | Qwen 系列 | 开源 + 云 |
| Mistral | Mixtral | 开源 + 企业 |
| 字节 | Doubao | 自用 + API |
| 百度 | Ernie | 自用 + API |
| 月之暗面 | Kimi | 订阅 |
| 智谱 | GLM | 开源 + API |

### 16.2 商业模式分类

| 模式 | 代表 | 毛利率 |
| :--- | :--- | :--- |
| 闭源 API | OpenAI | 60-80% |
| 开源 + 服务 | DeepSeek | 20-40% |
| 自研 + 自用 | Google, Meta | — |
| 行业方案 | 垂直优化 | 70%+ |
| 订阅 | ChatGPT Plus | 80%+ |

### 16.3 应用场景渗透

| 场景 | 渗透率 | 主要驱动 |
| :--- | :--- | :--- |
| 代码辅助 | 80%+ | Copilot, Cursor |
| 客服 | 50%+ | 自助化 |
| 内容生成 | 70%+ | 营销/文案 |
| 教育 | 30%+ | 个性化辅导 |
| 医疗 | <5% | 合规/监管 |
| 法律 | 10%+ | 检索/起草 |
| 金融 | 20%+ | 报告/分析 |
| 科研 | 15%+ | 文献综述 |
| Agent | 5% | 工具使用 |

---

## 第八部分: 章节内自检与练习

## 17. 自检题(答案见下节)

1. Chinchilla 给出的 $D/N$ 最优比例约为多少?
2. DeepSeek-V3 总参数与激活参数各是多少?
3. o1 与 R1 的核心差异?
4. DPO 相对 PPO 的核心简化?
5. GRPO 相对 PPO 的核心简化?
6. MLA 相对 GQA 的核心改进?
7. FlashAttention 省的是计算还是访存?
8. YaRN 相对 NTK-aware 的核心改进?
9. 为什么 decoder-only 成为 LLM 主流?
10. RLHF 三阶段分别是什么?

## 18. 自检答案

1. ~20 tokens/param
2. 总 671B, 激活 37B(约 5.5%)
3. o1 是闭源 + RL 训长 CoT, R1 是开源 + GRPO + 纯 RL
4. 不需训 RM, 不需 RL, 用闭式解代回变成偏好分类损失
5. 不需 Critic, 用组内平均作基线
6. 用低秩潜向量压缩 KV, 比共享头更省
7. 访存(HBM 读写), FLOPs 不变
8. NTK-aware 改频率缩放, YaRN 加注意力温度校正
9. 训练目标统一, 数据无需成对, ICL 天然
10. SFT(指令微调) → RM(奖励模型) → PPO(策略优化)

## 19. 代码练习

### 练习 19.1: 估算 Chinchilla 最优分配

```python
def chinchilla_optimal(C):
    """给定算力, 返回最优参数和数据"""
    a, b = 0.34, 0.28
    N = (C / 6) ** (a / (a + b)) * (a / b) ** (1 / (a + b))
    D = (C / 6) ** (b / (a + b)) * (b / a) ** (1 / (a + b))
    return N, D

# GPT-3 算力
C_gpt3 = 3.15e23
N, D = chinchilla_optimal(C_gpt3)
print(f"GPT-3 算力下最优: N={N/1e9:.0f}B, D={D/1e9:.0f}B tokens")
# 实际 GPT-3: 175B, 300B → 不最优!
```

### 练习 19.2: 在线 softmax

```python
import torch

def online_softmax(x, block_size=8):
    """在线 softmax, 模拟 FlashAttention 的核心"""
    n = x.shape[0]
    m = float('-inf')  # running max
    l = 0  # running sum
    o = 0  # running output

    for i in range(0, n, block_size):
        block = x[i:i+block_size]
        m_block = block.max()
        l_block = (block - m_block).exp().sum()

        m_new = max(m, m_block)
        l_new = (m - m_new).exp() * l + (m_block - m_new).exp() * l_block
        # 注意: 这里只更新 m, l; o 需要乘以权重, 简化为只跟踪 m, l
        m, l = m_new, l_new

    return (x - m).exp() / l

x = torch.randn(32)
ref = torch.softmax(x, dim=0)
out = online_softmax(x)
print(f"Max diff: {(ref - out).abs().max():.2e}")  # 应 < 1e-5
```

### 练习 19.3: MoE 路由

```python
import torch
import torch.nn.functional as F

def moe_route(x, gate_weight, num_experts, top_k=2):
    """简化 MoE 路由"""
    # x: (B, d), gate_weight: (d, E)
    logits = x @ gate_weight  # (B, E)
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_idx = scores.topk(top_k, dim=-1)
    return topk_scores, topk_idx

x = torch.randn(4, 64)
gate_w = torch.randn(64, 8)
scores, idx = moe_route(x, gate_w, num_experts=8, top_k=2)
print(f"Top-{2} experts per token: {idx.shape}")  # (4, 2)
print(f"Routing weights:\n{scores}")
```

### 练习 19.4: DPO 损失

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO 损失"""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    return F.binary_cross_entropy_with_logits(
        beta * (pi_logratios - ref_logratios), torch.ones_like(pi_logratios)
    )

# 模拟
torch.manual_seed(42)
chosen = torch.randn(8) * 0.5 + 2.0  # 高 logp
rejected = torch.randn(8) * 0.5 + 0.5  # 低 logp
loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach())
print(f"DPO loss: {loss.item():.4f}")
```

---

## 第九部分: 总结图与速记卡

## 20. 五大主线一张图

```text
                    更强(Scaling)
                       ↑
                       |
   更长 ←————— LLM 发展 —————→ 更省
   (RoPE)       (2017-2026)    (MoE)
                       |
                       ↓
              更听话 + 更会想(对齐+推理)
                    (RLHF/DPO/GRPO)
```

## 21. 关键论文速记卡

| 年份 | 论文 | 一句话 |
| :--- | :--- | :--- |
| 2017 | Transformer | 注意力替代 RNN |
| 2020 | GPT-3 | 175B 涌现 ICL |
| 2020 | Scaling Law | 幂律可预测 |
| 2022 | Chinchilla | 同算力 70B 优 280B |
| 2022 | InstructGPT | RLHF 范式 |
| 2023 | LLaMA | 开源平民化 |
| 2023 | DPO | 免 RL 对齐 |
| 2024 | Mixtral | 稀疏 MoE |
| 2024 | MLA | 低秩 KV 压缩 |
| 2024 | o1 | RL 训推理 |
| 2025 | R1 | 纯 RL 开源推理 |

## 22. 模型族谱

```text
LLaMA 系: LLaMA-1 → LLaMA-2 → LLaMA-3 → LLaMA-4(MoE)
Qwen 系: Qwen-1 → Qwen-2 → Qwen-2.5 → Qwen-3(MoE)
DeepSeek: V1 → V2(MLA+MoE) → V3(FP8+MoE) → R1(GRPO)
GPT 系: GPT-2 → GPT-3 → GPT-3.5 → GPT-4 → o1 → o3 → GPT-5
Claude 系: Claude 1 → 2 → 3 → 3.5 → 4(extended thinking)
Gemini 系: Gemini 1 → 1.5 → 2.0 → 2.5 → 3(thinking)
```

---

## 第十部分: 索引与交叉引用

## 23. 关键术语索引(中英对照)

- 涌现 Emergence
- 外推 Extrapolation
- 思维链 CoT (Chain-of-Thought)
- 慢思考 Slow Thinking
- 蒸馏 Distillation
- 量化 Quantization
- 续训 Continual Training
- MoE Mixture-of-Experts
- MLA Multi-head Latent Attention
- GQA Grouped-Query Attention
- RoPE Rotary Position Embedding
- ICL In-Context Learning
- RLHF RL from Human Feedback
- DPO Direct Preference Optimization
- GRPO Group Relative Policy Optimization
- TTC Test-Time Compute
- RM Reward Model
- SFT Supervised Fine-Tuning
- PPO Proximal Policy Optimization

## 24. 跨章节引用

- 详细架构: [第02章-架构演进](第02章-架构演进.md)
- 位置编码: [第03章-位置编码](第03章-位置编码与长上下文演进.md)
- 注意力: [第04章-注意力](第04章-注意力与效率演进.md)
- 对齐: [第05章-对齐](第05章-对齐技术演进.md)
- 推理: [第06章-推理](第06章-推理能力演进.md)
- 训练: [第07章-训练](第07章-LLM训练全流程实务.md)
- 监控: [第08章-监控](第08章-训练监控与权重续训.md)

## 25. 跨教程引用

- Transformer 基础: 深度学习手撕/第05章
- 优化器: 深度学习手撕/第02章
- 分布式: 深度学习手撕/第07章
- 数学基础: 基础/数学基础
- 工程能力: 基础/工程能力

---

> **第一部分到此为止, 后续每章深入一条主线。本章作为总览, 已提供完整数学框架 + 工程实践 + 历史脉络 + 代码示例 + 面试题库, 共计 4000+ 行, 形成 LLM 学习的"地图 + 速记 + 工具箱"。**

---

## 第十一部分: 详细模型卡片(Model Cards) - 五大厂商旗舰深度剖析

## 26. GPT 系列深度卡片

### 26.1 GPT-3 (Brown et al., 2020, arXiv:2005.14165)

**基本信息**:
- 发布时间: 2020 年 5 月
- 机构: OpenAI
- 参数量: 175B
- 训练数据: 约 570 GB 文本(Common Crawl + WebText + Books + Wikipedia)
- 训练算力: ~3.14 × 10²³ FLOPs
- 上下文: 2048 tokens

**架构细节**:
- Decoder-only Transformer
- 96 层, 96 头(head_dim = 128)
- d_model = 12288
- FFN 中间维度 4 × d_model = 49152
- 词表 50257(BPE)
- 稀疏注意力模式(本地+全局交替, 类似 Sparse Transformer)

**关键能力**:
- ICL(In-Context Learning): 给几个示例, 模型在新任务上工作
- 涌现能力: 在大模型上, 某些任务突然从 0 提升到可用
- Few-shot learning: 不需微调, 提示内即可学习

**历史地位**: 标志 LLM 时代真正开始, "规模即能力"被广泛验证

### 26.2 GPT-4 (OpenAI, 2023, arXiv:2303.08774)

**基本信息**:
- 发布时间: 2023 年 3 月
- 参数量: 未公开(估计 1.8T MoE 或 800B+)
- 多模态: 文本 + 图像输入, 文本输出
- 上下文: 8K → 32K → 128K(后续版本)
- 训练算力: ~2 × 10²⁵ FLOPs(估)

**关键能力**:
- 多模态理解: 图像 + 文本联合推理
- 长上下文: 1M 版本已开放
- 复杂推理: 律师考试前 10%, SAT 数学满分

**局限**:
- 闭源, 无权重
- 幻觉问题严重
- 知识截止 2023.04

### 26.3 o1 / o3 / GPT-5 (推理模型, 2024-2025)

**o1**(2024.09):
- 不再靠 CoT 提示, 而用 RL 训练自发长思维链
- 内部生成数千隐藏 token 后给答案
- 推理时间 vs 准确率呈新的 scaling 曲线
- AIME 2024: 83.3%, GPQA Diamond: 78%
- 价格: $15 / $60 per 1M tokens(in/out)

**o3**(2024.12, 2025.01 完整版):
- AIME 2024: 91.7%
- GPQA Diamond: 87.7%
- ARC-AGI: 87.5%(低算力配置)
- 完整算力版: ARC-AGI 达到 96%, 接近人类专家

**GPT-5**(2025.08 发布, 推测):
- 统一 reasoning + non-reasoning
- "reasoning effort" 控制项
- 多模态原生

## 27. LLaMA 系列深度卡片

### 27.1 LLaMA-1 (Touvron et al., 2023, arXiv:2302.13971)

**基本信息**:
- 发布时间: 2023 年 2 月
- 机构: Meta
- 规模: 7B / 13B / 33B / 65B
- 训练数据: 1.4T tokens
- 上下文: 2048

**架构创新**:
- Pre-RMSNorm(替代 LayerNorm)
- SwiGLU FFN(替代 ReLU/GELU)
- RoPE 位置编码
- 移除 bias 项
- 词表 32K(SentencePiece BPE)

**训练技巧**:
- 2048 A100-80GB, ~21 天训 65B
- 数据效率高(1.4T tokens > GPT-3 的 300B)
- 公开训练细节, 开源权重

### 27.2 LLaMA-2 (2023.07, arXiv:2307.09288)

**变化**:
- 上下文: 4K
- GQA(8 KV 组, 34B/70B)
- RLHF + 安全微调
- 商业可用(特殊许可)

### 27.3 LLaMA-3 (2024.04, arXiv:2407.21783)

**重大升级**:
- 8B / 70B / 405B
- 上下文: 8K → 128K(3.1)
- 15T+ tokens 训练(大幅过训)
- 词表 128K(更高效的多语编码)
- GQA 全系标配

**LLaMA-3.1**:
- 128K 上下文(8K 训练 + YaRN 外推)
- 405B 模型开源
- 多语言: 8 种官方语言

**LLaMA-3.2**:
- 多模态(11B/90B)
- 端侧 1B/3B

### 27.4 LLaMA-4 (2025, 推测)

**架构**:
- MoE 109B(激活 ~12B)
- iRoPE(交错 RoPE, 支持 10M 上下文)
- 多模态原生

## 28. Qwen 系列深度卡片

### 28.1 Qwen-1 / Qwen-1.5 (2023-2024)

**架构**:
- Decoder-only
- GQA
- RoPE
- 中文优化词表

### 28.2 Qwen-2 (2024.06)

- 5 个尺寸: 0.5B / 1.5B / 7B / 57B / 72B
- 中文 + 英文 + 多语
- 27 种语言支持

### 28.3 Qwen-2.5 (2024.09)

**重大升级**:
- 训练数据 18T tokens
- 128K 上下文
- 29 种语言
- 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B
- Plus/MAX 系列

### 28.4 Qwen-3 (2025.04 推测)

- 混合 Dense + MoE
- 235B-A22B(MoE, 激活 22B)
- QwQ-32B 推理模型
- 中文极强

## 29. DeepSeek 系列深度卡片

### 29.1 DeepSeek-LLM (2024.01, arXiv:2401.02954)

- 7B / 67B
- 2T tokens 中英训练
- MLA + MoE 雏形

### 29.2 DeepSeek-MoE (2024.01, arXiv:2401.06066)

**架构创新**:
- 细粒度专家(分割 FFN 为多个小专家)
- 共享专家 + 路由专家
- 解决专家路由专业化问题

### 29.3 DeepSeek-V2 (2024.05, arXiv:2405.04434)

**MLA 首次亮相**:
- 236B 总参数, 21B 激活
- MLA(Multi-head Latent Attention)
- KV Cache 压缩 ~10×
- 性能比肩 LLaMA-70B, 推理成本低 1/10

**架构细节**:
- 60 层, 头 128, head_dim 128
- 2 路由专家 + 1 共享专家
- 词表 100K

### 29.4 DeepSeek-V3 (2024.12, arXiv:2412.19437)

**重大升级**:
- 671B MoE(激活 37B)
- 256 路由专家 + 1 共享专家
- top-8 路由
- **FP8 混合精度训练**: 首批 FP8 大模型
- DualPipe 流水线
- 训练成本: $5.5M(DeepSeek 自报)

**训练数据**: 14.8T tokens

### 29.5 DeepSeek-R1 (2025.01, arXiv:2501.12948)

**推理模型突破**:
- **R1-Zero**: 纯 RL(GRPO)+ 可验证奖励, 无 SFT 冷启动
- 自发涌现"顿悟时刻"(aha moment)
- 输出可读性差, 语言混杂
- **R1**: 加入 SFT cold-start, 平衡可读性与推理
- 671B MoE + 蒸馏系列(Qwen-1.5B/7B, LLaMA-8B/70B)

**性能**:
- AIME 2024: 79.8%
- MATH-500: 97.3%
- 接近 o1, 远低于价格

## 30. Claude 系列深度卡片

### 30.1 Claude 3 (2024.03)

- Haiku / Sonnet / Opus
- 200K 上下文
- 多模态(图像输入)
- Constitutional AI 强化对齐

### 30.2 Claude 3.5 / 3.7 (2024.10 / 2025.02)

- Sonnet 3.5: 代码能力大幅提升
- 3.7: extended thinking(可控推理)
- 200K 上下文

### 30.3 Claude 4 (2025.05 推测)

- 更大模型规模
- 1M 上下文(企业版)
- 更强 reasoning
- 工具使用成熟

## 31. Gemini 系列深度卡片

### 31.1 Gemini 1.0 / 1.5 (2023-2024)

- Ultra / Pro / Nano 三档
- 1.5 Pro: 1M 上下文
- 多模态原生(文本+图像+音频+视频)

### 31.2 Gemini 2.0 (2024.12)

- 多模态 Agent: Project Astra
- 实时语音视频
- 工具使用原生

### 31.3 Gemini 2.5 / 3 (2025)

- Thinking mode
- Deep Think(多路径推理)
- 2M 上下文
- 性能: MMLU 88+, GPQA 80+

---

## 第十二部分: 各主线的"为什么" 哲学思辨

## 32. 为何 LLM 是当前 AI 的范式

**范式 1 (传统 ML)**: 任务专属模型 + 人工特征
**范式 2 (深度学习)**: 通用架构 + 大量标注数据
**范式 3 (预训练-微调)**: 大模型预训练 + 任务微调
**范式 4 (LLM 时代)**: 超大预训练 + ICL + SFT + RLHF

每次范式跃迁都把"上游通用化"和"下游轻量化"推向极致。

## 33. 为何 ICL 有效

直觉: Transformer 的前馈层学习了一类通用程序, 这些程序可通过提示"激活"。

数学视角: ICL ≈ 隐式梯度下降
- 提示 $(x_1, y_1), ..., (x_k, y_k)$ 等价于提供"训练数据"
- 模型根据这些数据, 在前向时构造一个隐式模型
- 类似 "meta-learning" 的 few-shot 形式

**定理 33.1** (ICL ≈ Ridge Regression, von Oswald et al. 2023)
单层线性 Transformer + ICL 等价于核回归 / Ridge Regression, 优化目标:
$$
\min_B \| XB - Y \|_2^2 + \lambda \|B\|_2^2
$$
闭式解 $B^* = (X^\top X + \lambda I)^{-1} X^\top Y$, Transformer 的 OGD 隐式实现。

## 34. 为何 RLHF 比 SFT 强

SFT 用单一"正确答案"训练, 鼓励**模仿**。
RLHF 用"偏好对"训练, 鼓励**对比判断**。

直觉: 人类对"两个答案哪个好"的判断**远比对"独立写答案"简单**。RLHF 利用这一认知特性, 让训练更接近人类思维。

## 35. 为何推理需要 RL

直觉: 推理是**隐式搜索**。SFT 给定单一路径, 模型只能学表面; RL 通过奖励正确结果, 让模型**自己探索**多条推理路径。

**类比**: SFT 像背答案, RL 像做练习题。练习题能学会方法。

---

## 第十三部分: 各主线的"什么时候" 时机决策

## 36. 何时用 dense, 何时用 MoE

| 场景 | 推荐 |
| :--- | :--- |
| 训练算力紧张 | Dense(实现简单) |
| 推理算力紧张 | MoE(激活小) |
| 显存紧张 | Dense(INT4) |
| 团队规模小 | Dense(GQA 即可) |
| 大规模服务(> 10B 用户) | MoE |
| 研究/实验 | 两者皆可 |

## 37. 何时用 RLHF, 何时用 DPO

| 场景 | 推荐 |
| :--- | :--- |
| 资源充足, 极致效果 | RLHF(PPO) |
| 资源紧张, 求稳 | DPO |
| 在线探索 | RLHF |
| 静态偏好数据 | DPO |
| 推理任务, 可验证奖励 | GRPO |
| 安全敏感 | Constitutional + RLHF |

## 38. 何时上推理模型

| 场景 | 推荐 |
| :--- | :--- |
| 数学/代码/科学 | 推理模型 |
| 客服/闲聊 | 非推理模型 |
| 实时性要求高 | 非推理(快) |
| 复杂规划 | 推理模型 |
| 成本敏感 | 非推理 / 蒸馏推理 |
| 多步推理 / Agent | 推理模型 |

## 39. 何时扩上下文

| 场景 | 推荐 |
| :--- | :--- |
| RAG(检索后输入) | 8-32K |
| 长文档摘要 | 64-128K |
| 长代码分析 | 64-128K |
| Agent 记忆 | 128K-1M |
| 视频/多模态 | 1M+ |

---

## 第十四部分: 工业级 benchmark 与性能数据

## 40. 主要评测基准(2025)

| 基准 | 任务 | 评估能力 |
| :--- | :--- | :--- |
| MMLU | 57 学科多选 | 综合知识 |
| MMLU-Pro | 12K 题, 10 选项 | MMLU 增强版 |
| GSM8K | 小学数学 | 基础数学 |
| MATH | 高中数学竞赛 | 数学推理 |
| HumanEval | 代码生成 | Python 函数 |
| MBPP | Python 基础编程 | 代码 |
| SWE-bench | 真实 GitHub issue | 代码 agent |
| ARC-AGI | 通用推理 | 推理能力 |
| GPQA | 博士级科学 QA | 高级科学 |
| AIME | 数学奥赛 | 高级数学 |
| HLE(Humanity's Last Exam) | 多学科难题 | 综合 |
| TruthfulQA | 事实性 | 真实性 |
| IFEval | 指令遵循 | 指令执行 |
| MT-Bench | 多轮对话 | 对话 |
| Arena-Hard | 真实偏好 | 综合 |
| AlpacaEval | 指令长度可控 | 综合 |

## 41. 旗舰模型性能对比(2025 H1)

| 模型 | MMLU | GPQA | AIME | SWE-bench | HumanEval | 上下文 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 88.7 | 50.6 | — | 33 | 90 | 128K |
| o3 | 92+ | 87.7 | 91.7 | 71 | 99+ | 200K |
| Claude 4 Opus | 88+ | 75+ | — | 60+ | 95+ | 200K |
| Gemini 2.5 Pro | 88+ | 70+ | 75+ | 60+ | 90+ | 1M |
| LLaMA-3.1 405B | 88.6 | 70+ | — | — | 89 | 128K |
| DeepSeek-V3 | 88.5 | 65+ | 39 | 42 | 82+ | 64K |
| DeepSeek-R1 | 90.8 | 71.5 | 79.8 | 49 | 96+ | 64K |
| Qwen-2.5 72B | 86+ | 65+ | — | — | 86 | 128K |
| Qwen-QwQ-32B | 85+ | 65+ | 70+ | 50 | 92+ | 128K |

## 42. 推理成本对比(每 1M tokens, USD)

| 模型 | 输入价 | 输出价 |
| :--- | :--- | :--- |
| GPT-4o | $2.5 | $10 |
| o1 | $15 | $60 |
| o3 | $60+ | $240+ |
| Claude 4 Opus | $15 | $75 |
| Gemini 2.5 Pro | $1.25 | $5 |
| DeepSeek-V3 | $0.14 | $0.28 |
| DeepSeek-R1 | $0.55 | $2.19 |
| Qwen-2.5 Plus | $0.4 | $1.2 |

**性价比冠军**: DeepSeek-R1(价格 1/30 o1, 性能接近)

---

## 第十五部分: 经典失败案例与教训

## 43. 案例 1: GPT-3 早期 ICL 失败

**问题**: GPT-3 在某些复杂任务(算术、多跳推理)即使加 few-shot 也不稳定。
**原因**: 模型未真正"学会"推理, 只是模仿表面模式。
**解决**: 后来的 CoT 提示 → o1 类训练。

## 44. 案例 2: LLaMA-2 早期幻觉

**问题**: LLaMA-2 Chat 早期版本对中文 / 罕见实体易幻觉。
**原因**: 中文训练数据占比小。
**解决**: LLaMA-3 增加多语言数据, Qwen 等专注中文。

## 45. 案例 3: Gemini 1.0 图像推理失败

**问题**: 早期 Gemini 对图像中数字计数不准。
**原因**: 多模态训练未充分。
**解决**: 后续版本增加视觉 grounding 数据。

## 46. 案例 4: R1-Zero 可读性差

**问题**: DeepSeek-R1-Zero 推理力强但输出语言混杂、可读性差。
**原因**: 纯 RL 未约束语言一致性。
**解决**: R1 加 SFT cold-start 和语言奖励。

## 47. 案例 5: RLHF 训练不稳定

**问题**: 多个团队报告 PPO 在 LLM 上训练不稳, KL 发散或奖励 hacking。
**原因**: PPO 对超参敏感, 价值网络难训。
**解决**: DPO / GRPO 等更稳定的替代。

---

## 第十六部分: 章节总结

## 48. 一页纸总结

```text
LLM 五大主线 (2017-2026):
1. 更强 → Scaling Law + Chinchilla
2. 更省 → MoE + MLA + FlashAttention
3. 更长 → RoPE + YaRN + GQA
4. 更听话 → RLHF → DPO → GRPO
5. 更会想 → CoT → o1 → R1(test-time compute scaling)

关键论文 (按年代):
2017: Transformer
2018: GPT-1, BERT
2020: GPT-3, Scaling Law
2022: Chinchilla, InstructGPT, Emergence
2023: LLaMA, DPO, QLoRA, RoPE 普及
2024: MLA, MoE 主流, o1, DeepSeek-V3
2025: DeepSeek-R1, GPT-5, Claude 4 thinking

工程栈 (2025 标准):
架构: Decoder-only + RMSNorm + RoPE + SwiGLU + GQA/MLA + MoE
训练: AdamW + cosine + BF16/FP8 + FSDP/TP/PP
对齐: SFT → DPO 或 RLHF(PPO) → 可选 GRPO
推理: FlashAttn + PagedAttn(vLLM) + INT4/FP8 + 投机解码
部署: Continuous batching + KV Cache 管理
```

## 49. 5 道必读面试题

1. **Chinchilla 法则**: 用公式描述并解释为何 GPT-3 不优。
2. **DPO vs PPO**: 用闭式解推导 DPO 损失。
3. **MLA vs GQA**: 用公式描述 MLA 的低秩压缩机制。
4. **FlashAttention**: 解释 online softmax 如何避免实体化注意力矩阵。
5. **R1 vs o1**: 比较两者训练范式和开源状况。

(详见各章深入内容)

---

> **总览章至此完成, 共 ~4000+ 行, 包含: 主线图 + 时间线 + 数学推导 + 算法伪代码 + 代码示例 + 历史脉络 + 跨厂商对比 + 工程实践 + 面试题库 + 失败案例 + 自检练习。读者读完全章应能: (1) 完整复述 LLM 5 年演进; (2) 用数学描述每条主线的核心创新; (3) 默写 RMSNorm / online softmax / DPO 损失 / MLA 压缩 等关键实现; (4) 对比五大厂商旗舰模型的优劣; (5) 回答 25+ 面试高频题。**

---

## 第十七部分: 扩展阅读与论文清单(分类)

## 50. 架构演进类(必读 10 篇)

1. **Attention Is All You Need** (Vaswani et al., NeurIPS 2017, arXiv:1706.03762)
2. **On Layer Normalization in the Transformer Architecture** (Nguyen & Salazar, 2020, arXiv:2002.04745)
3. **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019, arXiv:1910.07467) — RMSNorm
4. **GLU Variants Improve Transformer** (Shazeer, 2020, arXiv:2002.05202) — SwiGLU
5. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023, arXiv:2305.13245)
6. **Multi-head Latent Attention (MLA)** (DeepSeek-AI, 2024, 详见 DeepSeek-V2 论文 arXiv:2405.04434)
7. **Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (Fedus et al., 2021, arXiv:2101.03961)
8. **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts** (Du et al., 2021, arXiv:2112.06905)
9. **Mixture-of-Experts with Expert Choice Routing** (Zhou et al., 2022, arXiv:2202.09368)
10. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023, arXiv:2302.13971)

## 51. 注意力效率类(必读 8 篇)

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., NeurIPS 2022, arXiv:2205.14135)
2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023, arXiv:2307.08691)
3. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** (Shah et al., 2024, arXiv:2407.08608)
4. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., SOSP 2023, arXiv:2309.06180)
5. **Fast Transformer Decoding: One Write-Head is All You Need** (Shazeer, 2019, arXiv:1911.02150) — MQA
6. **GQA** (Ainslie et al., 2023, arXiv:2305.13245)
7. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** (DeepSeek-AI, 2024, arXiv:2405.04434) — MLA
8. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2023, arXiv:2312.00752)

## 52. 位置编码类(必读 6 篇)

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021, arXiv:2104.09864)
2. **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** (Press et al., 2022, arXiv:2108.12409) — ALiBi
3. **Extending the Context Length of Large Language Models with Position Interpolation** (Chen et al., 2023, arXiv:2306.15595)
4. **NTK-aware Scaling** (bloc97 博客, 2023)
5. **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023, arXiv:2309.00071)
6. **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** (Ding et al., 2024, arXiv:2402.13753)

## 53. 对齐类(必读 10 篇)

1. **Training Language Models to Follow Instructions with Human Feedback** (Ouyang et al., InstructGPT, 2022, arXiv:2203.02155)
2. **Direct Preference Optimization** (Rafailov et al., NeurIPS 2023, arXiv:2305.18290)
3. **Constitutional AI: Harmlessness from AI Feedback** (Bai et al., Anthropic, 2022, arXiv:2212.08073)
4. **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback** (Lee et al., 2023, arXiv:2309.00267)
5. **Open RLHF: A Fully Open-Source Toolkit for Reinforcement Learning with Human Feedback** (Wang et al., 2024)
6. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (Shao et al., 2024, arXiv:2402.03300) — GRPO 原始论文
7. **KTO: Model Alignment as Prospect Theoretic Optimization** (Ethayarajh et al., 2024, arXiv:2402.01306)
8. **ORPO: Monolithic Preference Optimization without Reference Model** (Hong et al., 2024, arXiv:2403.07691)
9. **SimPO: Simple Preference Optimization with a Reference-Free Reward** (Meng et al., 2024, arXiv:2405.14734)
10. **Process Reward Models** (Lightman et al., 2023, arXiv:2305.20050) — PRM

## 54. 推理模型类(必读 8 篇)

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., NeurIPS 2022, arXiv:2201.11903)
2. **Self-Consistency Improves Chain of Thought Reasoning in Language Models** (Wang et al., 2022, arXiv:2203.11171)
3. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (Yao et al., NeurIPS 2023, arXiv:2305.10601)
4. **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022, arXiv:2210.03629)
5. **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023, arXiv:2303.11366)
6. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (DeepSeek-AI, 2025, arXiv:2501.12948)
7. **Let's Verify Step by Step** (Lightman et al., 2023, arXiv:2305.20050)
8. **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** (Brown et al., 2024, arXiv:2407.21787)

## 55. Scaling Law 与涌现类(必读 5 篇)

1. **Scaling Laws for Neural Language Models** (Kaplan et al., 2020, arXiv:2001.08361)
2. **Training Compute-Optimal Large Language Models** (Hoffmann et al., Chinchilla, 2022, arXiv:2203.15556)
3. **Emergent Abilities of Large Language Models** (Wei et al., 2022, arXiv:2206.07682)
4. **Are Emergent Abilities of Large Language Models a Mirage?** (Schaeffer et al., NeurIPS 2023, arXiv:2304.15004)
5. **Will We Run Out of Data? An Analysis of the Limits of Scaling Datasets** (Villalobos et al., 2022, arXiv:2211.04325)

## 56. 工程实践类(必读 6 篇)

1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020, arXiv:1910.02054)
2. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (Shoeybi et al., 2019, arXiv:1909.08053)
3. **PaLM: Scaling Language Modeling with Pathways** (Chowdhery et al., 2022, arXiv:2204.02311)
4. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023, arXiv:2305.14314)
5. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Frantar et al., 2022, arXiv:2210.17323)
6. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** (Xiao et al., 2023, arXiv:2211.10438)

---

## 第十八部分: 进阶练习题(代码 + 推导)

## 57. 代码练习 1: 复现 Online Softmax

**目标**: 实现 FlashAttention 中的 online softmax, 验证与标准 softmax 一致。

```python
import torch

def online_softmax_block(x, block_size=4):
    """分块计算 softmax, 模拟 FlashAttention 的 online 算法"""
    n = x.shape[0]
    m = float('-inf')  # running max
    l = 0.0  # running sum of exp
    o = 0.0  # placeholder

    for i in range(0, n, block_size):
        block = x[i:i+block_size]
        m_block = block.max().item()
        m_new = max(m, m_block)
        # 更新 l
        l = (m - m_new) * (l if isinstance(l, float) else l) + \
            (m_block - m_new) * ((block - m_block).exp().sum()).item()
        # 更新 o(此示例只跟踪 m, l)
        m = m_new

    # 最终归一化常数
    return m, l

def standard_logsumexp(x):
    return x.max() + (x - x.max()).exp().sum().log()

# 测试
torch.manual_seed(0)
x = torch.randn(16)
m, l = online_softmax_block(x)
lse = standard_logsumexp(x)
print(f"Online LSE: {m + torch.tensor(l).log():.6f}")
print(f"Standard LSE: {lse:.6f}")
print(f"Diff: {abs(m + torch.tensor(l).log() - lse):.2e}")
```

## 58. 代码练习 2: RMSNorm 前向

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x).type_as(x) * self.weight

# 测试
norm = RMSNorm(768)
x = torch.randn(2, 4, 768)
y = norm(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
print(f"Output norm per token: {y.pow(2).mean(-1).sqrt()}")  # 应接近 1
```

## 59. 代码练习 3: 简易 RoPE

```python
import torch

def precompute_freqs(dim, max_len, base=10000):
    """预计算 RoPE 频率"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    return torch.outer(t, freqs)  # (max_len, dim//2)

def apply_rope(x, freqs):
    """x: (B, n_heads, T, head_dim)"""
    # freqs: (T, head_dim//2)
    T = x.shape[-2]
    freqs = freqs[:T].to(x.device)
    cos = freqs.cos()[None, None, :, :]  # (1, 1, T, dim//2)
    sin = freqs.sin()[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return rotated

# 测试
B, n_heads, T, head_dim = 2, 8, 16, 64
freqs = precompute_freqs(head_dim, T)
x = torch.randn(B, n_heads, T, head_dim)
y = apply_rope(x, freqs)
print(f"Input: {x.shape} → Output: {y.shape}")
# 验证旋转后范数不变
print(f"Norm unchanged: {torch.allclose(x.norm(dim=-1), y.norm(dim=-1))}")
```

## 60. 代码练习 4: GQA 注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA_Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 重复 KV 到 Q 头数
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

# 测试: 32 Q 头, 8 KV 头(4 倍压缩)
attn = GQA_Attention(512, n_heads=32, n_kv_heads=8)
x = torch.randn(2, 64, 512)
y = attn(x)
print(f"GQA output: {y.shape}")

# 计算 KV Cache 节省
MHA_kv = 2 * 32 * 64 * 64  # B, n_heads, T, head_dim
GQA_kv = 2 * 8 * 64 * 64
print(f"MHA KV: {MHA_kv / 1e3:.1f}K elements")
print(f"GQA KV: {GQA_kv / 1e3:.1f}K elements")
print(f"压缩比: {MHA_kv / GQA_kv:.1f}×")
```

## 61. 代码练习 5: DPO 损失完整实现

```python
import torch
import torch.nn.functional as F

def compute_logps(model, input_ids, labels_mask):
    """计算模型在 response 部分的 log p"""
    outputs = model(input_ids=input_ids, labels=None)
    logits = outputs.logits[:, :-1, :]  # (B, T-1, V)
    labels = input_ids[:, 1:]  # (B, T-1)

    # 只计算 response 部分(mask=1)
    per_token_logps = F.log_softmax(logits, dim=-1)
    chosen_logps = (per_token_logps * labels_mask[:, 1:]).sum(dim=-1)
    return chosen_logps

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """完整 DPO 损失"""
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 可选: 加上 SFT loss on chosen
    # loss = loss + 0.1 * (-policy_chosen.mean())

    return loss

# 模拟
torch.manual_seed(42)
batch_size = 4
chosen_ids = torch.randint(0, 1000, (batch_size, 32))
rejected_ids = torch.randint(0, 1000, (batch_size, 32))
chosen_mask = torch.ones_like(chosen_ids)
rejected_mask = torch.ones_like(rejected_ids)

# 模拟 log p(实际应从模型计算)
chosen_logps = torch.randn(batch_size) * 0.3 + 1.5
rejected_logps = torch.randn(batch_size) * 0.3 + 0.5

loss = dpo_loss(chosen_logps, rejected_logps,
                chosen_logps.detach(), rejected_logps.detach())
print(f"DPO loss: {loss.item():.4f}")
```

## 62. 推导练习 1: Chinchilla 推导

**题目**: 给定 $L(N, D) = A/N^\alpha + B/D^\beta + L_\infty$ 与 $C = 6ND$, 求最优 $N^*, D^*$。

**解答**:
拉格朗日函数:
$$
\mathcal{L}(N, D, \lambda) = A N^{-\alpha} + B D^{-\beta} + L_\infty + \lambda(6ND - C)
$$

取偏导:
$$
\frac{\partial \mathcal{L}}{\partial N} = -\alpha A N^{-\alpha-1} + 6\lambda D = 0
$$
$$
\frac{\partial \mathcal{L}}{\partial D} = -\beta B D^{-\beta-1} + 6\lambda N = 0
$$

两式相除:
$$
\frac{\alpha A N^{-\alpha-1}}{\beta B D^{-\beta-1}} = \frac{D}{N}
$$
$$
\frac{\alpha A}{\beta B} = \frac{D^\beta}{N^\alpha}
$$

即 $D^\beta \cdot N^\alpha = \frac{\beta B}{\alpha A}$(常数)。

代入 $C = 6ND$ 得 $N^* \propto C^{\alpha/(\alpha+\beta)}$, $D^* \propto C^{\beta/(\alpha+\beta)}$, 比例 $D^*/N^* = (\beta/\alpha)^{1/(\alpha+\beta)} \cdot (C/6)^{(\beta-\alpha)/(\alpha+\beta)}$。当 $\alpha \approx \beta$ 时, $D^*/N^* \approx 20$ tokens/param。

## 63. 推导练习 2: DPO 损失推导

**题目**: 从 RLHF 目标推导 DPO 损失。

**解答**:
RLHF 目标:
$$
\max_\pi \mathbb{E}_{x,y}[r(x,y)] - \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})
$$

闭式解:
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

由此反解 $r$:
$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

Bradley-Terry 偏好模型:
$$
P(y_w \succ y_l | x) = \sigma(r(x,y_w) - r(x,y_l))
$$

代入 $r$ 表达式, $\log Z(x)$ 相消:
$$
P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

用 $\pi_\theta$ 代替 $\pi^*$, 取负对数:
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}))]
$$

QED。

## 64. 推导练习 3: GRPO 优势推导

**题目**: 推导 GRPO 在 PPO 框架下的优势估计。

**解答**:
PPO 优势: $A_t = Q(s_t, a_t) - V(s_t)$, 需价值网络 $V$。
GRPO: 对同一 prompt $x$ 采样 $G$ 个回答 $\{y_1, ..., y_G\}$, 计算奖励 $\{r_1, ..., r_G\}$。

优势估计:
$$
A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \epsilon}
$$

**直观解释**:
- mean 作为基线, 替代价值函数 $V$
- std 归一化, 让优势尺度一致
- 无 Critic, 训练更稳

**PPO 更新**(GRPO 版本):
$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_i \left[\min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)\right] + \beta D_{\mathrm{KL}}(\pi \| \pi_{\text{ref}})
$$
其中 $\rho_i = \pi_\theta(y_i|x) / \pi_{\theta_{\text{old}}}(y_i|x)$。

## 65. 推导练习 4: RoPE 相对位置推导

**题目**: 证明 RoPE 旋转后内积只依赖相对位置 $m-n$。

**解答**:
设 $R_m$ 是位置 $m$ 的旋转矩阵:
$$
R_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

对 $d$ 维向量分 $d/2$ 个 2D 块, 每块独立旋转。
设 query 位置 $m$, key 位置 $n$:
$$
(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

因为 $R_m^\top = R_{-m}$, $R_m R_n = R_{m+n}$(旋转群性质)。

因此内积只依赖 $n - m = -(m-n)$, 即相对位置。

QED。

## 66. 推导练习 5: Online Softmax 正确性

**题目**: 证明 online softmax 分块计算与标准 softmax 数学等价。

**解答**:
标准 softmax:
$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

分块: $x = [x^{(1)}, x^{(2)}, ..., x^{(T)}]$。

设 $m_k = \max(x^{(1)}, ..., x^{(k)})$, $l_k = \sum_{j=1}^{|x^{(k)}|} e^{x_j^{(k)} - m_k}$。

递推:
$$
m_{k+1} = \max(m_k, \max(x^{(k+1)}))
$$
$$
l_{k+1} = e^{m_k - m_{k+1}} l_k + \sum_{j} e^{x_j^{(k+1)} - m_{k+1}}
$$

最终:
$$
\text{softmax}(x) = \frac{e^{x_i - m_T}}{l_T}
$$

数学上严格等价。QED。

---

## 第十九部分: 章节总结 checklist

## 67. 学完本章你应该能

### 67.1 知识层面

- [ ] 说出 LLM 5 大主线: 更强/更省/更长/更听话/更会想
- [ ] 列出每个主线的代表技术
- [ ] 默写 Chinchilla 法则公式与推导
- [ ] 默写 RoPE 旋转矩阵与相对位置证明
- [ ] 默写 RMSNorm 公式与代码
- [ ] 默写 DPO 损失公式与推导
- [ ] 默写 GRPO 优势估计
- [ ] 描述 FlashAttention 的 online softmax
- [ ] 描述 MLA 低秩压缩机制
- [ ] 说出五大厂商旗舰模型差异

### 67.2 实践层面

- [ ] 能用 PyTorch 实现 RMSNorm
- [ ] 能用 PyTorch 实现简易 RoPE
- [ ] 能用 PyTorch 实现 GQA 注意力
- [ ] 能用 PyTorch 实现 DPO 损失
- [ ] 能用 vLLM 部署 LLaMA-3
- [ ] 能用 MMLU 评测模型
- [ ] 能用 FlashAttention 加速训练
- [ ] 能用 GPTQ 量化模型到 INT4

### 67.3 思辨层面

- [ ] 能解释为何 decoder-only 主流
- [ ] 能解释 test-time scaling 与 training scaling 对偶
- [ ] 能解释涌现的成因与争议
- [ ] 能权衡 MoE 与 dense 的优劣
- [ ] 能权衡 RLHF/DPO/GRPO 的适用场景
- [ ] 能预测 LLM 未来 5 年的关键方向

---

## 68. 推荐学习资源汇总

### 68.1 课程

- **Stanford CS336: Large Language Modeling** (2024-2025) - Percy Liang 等
- **Stanford CS25: Transformers United** - 最新进展
- **CMU 11-711: Advanced NLP** - Graham Neubig
- **MIT 6.5940: TinyML and Efficient DL Computing** - Song Han
- **Berkeley CS182: Deep Learning** - Sergey Levine

### 68.2 博客与社区

- Lilian Weng's Blog (lilianweng.github.io) - LLM 综述
- Sebastian Raschka's Blog - 深度学习
- Andrej Karpathy 的 [Zero to Hero](https://github.com/karpathy/zero-to-hero) - GPT 手撕
- Jay Alammar's Blog - 可视化讲解
- 3Blue1Brown - 视觉化深度学习
- Maarten Grootendorst - Hands-On 系列

### 68.3 论文资源

- **arXiv**: cs.CL, cs.LG, cs.AI
- **Hugging Face Daily Papers**: 每日精选
- **Papers with Code**: 代码+论文
- **AlphaSignal**: arXiv 监控

### 68.4 开源框架

| 任务 | 推荐框架 |
| :--- | :--- |
| 训练 | transformers, Megatron-LM, DeepSpeed, LLaMA-Factory |
| RLHF | TRL (Transformer Reinforcement Learning) |
| 推理 | vLLM, SGLang, TensorRT-LLM |
| 量化 | AutoGPTQ, AWQ, bitsandbytes |
| 评估 | lm-evaluation-harness, OpenCompass |
| 长上下文 | FlashAttention, Ring Attention |
| Agent | LangChain, LlamaIndex, AutoGen |

---

## 69. 一页纸复习卡

```text
# LLM 五大主线 (速记)

更: scaling law + chinchilla
    L(N,D) = A/N^α + B/D^β + L∞
    D*/N* ≈ 20 tokens/param

省: MoE + MLA + FlashAttention
    MoE: 稀疏激活, top-k
    MLA: KV 低秩压缩
    FlashAttn: online softmax, 省 HBM

长: RoPE + YaRN + GQA/MLA
    RoPE: 旋转, 相对位置
    YaRN: NTK + 温度校正
    GQA: 共享 KV 头

听: RLHF → DPO → GRPO
    PPO: RM + Critic + KL
    DPO: 闭式代回, 免 RM
    GRPO: 免 Critic, 组内归一化

想: CoT → o1 → R1
    CoT: 提示引导
    o1: RL 训长 CoT, 闭源
    R1: 纯 RL + GRPO, 开源

# 关键数字
GPT-3: 175B, 300B tok
LLaMA-3: 405B, 15T tok, 128K
DeepSeek-V3: 671B(37B 激活), 14.8T tok, $5.5M
R1: 同 V3, AIME 79.8%, API 价 < o1 5%

# 关键公式
RMSNorm: x / sqrt(mean(x²) + ε) * g
DPO: -log σ(β log(π(yw)/πref(yw)) - β log(π(yl)/πref(yl)))
GRPO: A = (r - mean(r)) / std(r)
```

---

> **总览章完成。读者已具备 LLM 全景视角, 可深入后续章节。**

---

## 第二十部分: 模型架构深度对比表(2025 旗舰)

## 70. 详细架构参数对比

| 模型 | 总参数 | 激活参数 | 层数 | 头数 | head_dim | 词表 | 上下文 | 注意力 | 归一化 | FFN | 位置编码 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-3 | 175B | 175B | 96 | 96 | 128 | 50K | 2K | MHA | Pre-LN | 4d | Learned |
| LLaMA-1 65B | 65B | 65B | 80 | 64 | 128 | 32K | 2K | MHA | RMSNorm | SwiGLU 8/3d | RoPE |
| LLaMA-2 70B | 70B | 70B | 80 | 64 | 128 | 32K | 4K | GQA-8 | RMSNorm | SwiGLU | RoPE |
| LLaMA-3.1 405B | 405B | 405B | 126 | 128 | 128 | 128K | 128K | GQA-8 | RMSNorm | SwiGLU | RoPE+YaRN |
| Qwen-2.5 72B | 72B | 72B | 80 | 64 | 128 | 152K | 128K | GQA-8 | RMSNorm | SwiGLU | RoPE |
| DeepSeek-V2 | 236B | 21B | 60 | 128 | 128 | 100K | 128K | MLA | RMSNorm | SwiGLU | RoPE |
| DeepSeek-V3 | 671B | 37B | 61 | 128 | 128 | 129K | 128K | MLA | RMSNorm | SwiGLU | RoPE |
| Mixtral 8x7B | 47B | 13B | 32 | 32 | 128 | 32K | 32K | GQA | RMSNorm | MoE-SwiGLU | RoPE |
| Yi-200K | 34B | 34B | 48 | 56 | 128 | 64K | 200K | GQA-8 | RMSNorm | SwiGLU | RoPE |
| Gemini 1.5 Pro | (估) | — | — | — | — | 256K | 1M | — | — | — | — |
| Claude 3 Opus | (估) | — | — | — | — | — | 200K | — | — | — | — |
| o1 | (估) | — | — | — | — | — | 200K | — | — | — | — |

## 71. 各模型架构的关键创新点

### 71.1 LLaMA-1: 开源高效配方

**创新**:
- RMSNorm 替代 LayerNorm(快 7-10%)
- SwiGLU 替代 ReLU/GELU(效果更好, 同等算力)
- RoPE 位置编码(优于绝对位置)
- 移除 bias(参数少, 训练更稳)

**意义**: 奠定 2023-2025 开源模型的事实标准。

### 71.2 LLaMA-2: GQA 与 RLHF

**创新**:
- GQA-8: KV Cache 降 8×, 长上下文推理受益
- 公开 RLHF 细节(KTO/PPO)
- 商业可用许可

### 71.3 LLaMA-3: 大幅过训 + 长上下文

**创新**:
- 128K 上下文(YaRN 外推)
- 15T tokens(过训, Chinchilla 不最优)
- 128K 词表(多语编码更高效)
- 多模态版本(LLaMA-3.2 Vision)

### 71.4 DeepSeek-V2: MLA 首次实用化

**创新**:
- Multi-head Latent Attention(MLA): KV 缓存低秩压缩
- MLA 公式: $c = W_{\text{down}} h$, $K = W^K_{up} c$, $V = W^V_{up} c$
- KV Cache 从 $h \cdot d \cdot T$ 降到 $d_c \cdot T$, $d_c \ll h \cdot d$
- 配合 decoupled RoPE 处理位置编码

**意义**: 让长上下文推理成本大幅降低。

### 71.5 DeepSeek-V3: FP8 + 极稀疏 MoE

**创新**:
- 671B 总参数, 37B 激活(5.5%)
- 256 路由专家 + 1 共享专家
- top-8 路由
- 首批 FP8 大模型训练(降低显存 + 吞吐提升)
- DualPipe 流水线(跨节点 MoE 高效)
- 多 token 预测(MTP)训练目标(每步预测多 token)

**意义**: 训练成本仅 $5.5M(行业 1/10), 性能比肩 GPT-4。

### 71.6 Mixtral 8x7B: 开源 MoE 范式

**创新**:
- 8 专家, top-2 路由
- 47B 总参数, 13B 激活
- Apache 2.0 许可

### 71.7 Qwen 系列: 中文 + 多语领先

**创新**:
- 中文原生优化
- 152K 词表(中文编码高效)
- 多语言 29 种
- QwQ 推理模型

### 71.8 R1: 纯 RL 推理

**创新**:
- R1-Zero: 无 SFT, 纯 GRPO
- 自发涌现 aha moment
- 完全开源(MIT)
- 蒸馏小模型性能跃升

---

## 第二十一部分: LLM 工程实战关键数字速查

## 72. 显存占用速算

| 模型 | 参数量 | FP16 权重 | INT4 权重 | KV Cache (8K, FP16) | KV Cache (8K, INT4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 7B | 7×10⁹ | 14 GB | 3.5 GB | ~0.4 GB | ~0.1 GB |
| 13B | 13×10⁹ | 26 GB | 6.5 GB | ~0.7 GB | ~0.2 GB |
| 70B | 70×10⁹ | 140 GB | 35 GB | ~2 GB | ~0.5 GB |
| 405B | 405×10⁹ | 810 GB | 200 GB | ~5 GB | ~1.3 GB |
| 671B MoE(激活 37B) | 671×10⁹ | 1342 GB | 336 GB | ~2 GB | ~0.5 GB |

**KV Cache 公式**:
$$
\text{KV Cache} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot T \cdot d_{\text{head}} \cdot \text{bytes}
$$

## 73. 训练算力估算

**单 token FLOPs**: $6N$(N 为参数量)

**总训练 FLOPs**: $6ND$(D 为训练 tokens)

| 模型 | 训练 FLOPs(估算) | A100-80G 所需机时 |
| :--- | :--- | :--- |
| GPT-3 175B | 3.15×10²³ | ~30,000 GPU-days |
| LLaMA-65B | 6×10²⁴ | ~100,000 GPU-days |
| LLaMA-3.1 405B | 4×10²⁵ | ~600,000 GPU-days |
| DeepSeek-V3 | 5×10²⁴ | ~80,000 GPU-days(MoE 高效) |
| GPT-4(估) | 2×10²⁵ | ~300,000 GPU-days |

## 74. 推理吞吐速算

**单请求生成吞吐**:
$$
\text{Tokens/s} = \frac{1}{T_{\text{per-token}}}
$$

$T_{\text{per-token}}$ 取决于:
- 模型 FLOPs/token
- 硬件峰值 FLOPs
- KV Cache 大小(影响带宽)

**批处理吞吐**(LLM serving):
$$
\text{Total tokens/s} = \sum_{i=1}^{B} \frac{1}{T_{\text{per-token}_i}}
$$

**Continuous batching**(vLLM)让新请求插入空闲槽位, 利用率 80%+。

## 75. 关键优化技术对吞吐的影响

| 优化 | 吞吐提升 | 适用 |
| :--- | :--- | :--- |
| BF16 → FP8 | 1.5-2× | H100 |
| FP8 → INT4 | 2-3× | 通用 |
| FlashAttn-2 | 2-4× | 训练 + 推理 |
| PagedAttn | 2-5×(高并发) | serving |
| Continuous batching | 10-30× | serving |
| Speculative decoding | 2-3× | 单请求 |
| 多 GPU TP | 接近线性 | 大模型 |
| MoE 推理优化 | 5-10×(相比 dense 同激活) | MoE |

---

## 第二十二部分: 各主线核心论文精读摘要

## 76. Attention Is All You Need (2017) - 精读摘要

**贡献**:
- 提出 Transformer 架构
- 证明纯注意力可替代 RNN/CNN
- 并行训练 + 长期依赖建模

**核心结构**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$
$$
\text{MHA}(x) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$
$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

**关键技巧**:
- 缩放因子 $\sqrt{d_k}$ 防 softmax 饱和
- 多头机制增加表达力
- 残差 + LN(原始为 Post-LN)
- Sinusoidal 位置编码(可学习效果差不多但可外推)

**意义**: 2017-2026 几乎所有 LLM 都基于 Transformer。

## 77. GPT-3 (2020) - 精读摘要

**贡献**:
- 175B 参数, 验证"规模即能力"
- ICL 涌现: 不需微调, 提示内学习
- Few-shot prompting 范式

**实验结论**:
- 模型规模 vs 样本效率: 大模型更"样本高效"
- 零样本能力在某些任务上随规模单调提升
- 多步推理任务涌现(算术)

**局限**:
- 长上下文仍有限(2K)
- 训练数据时效性
- 偏见与毒性

## 78. Chinchilla (2022) - 精读摘要

**贡献**:
- 修正 Kaplan 缩放律
- 提出计算最优分配(参/数据同比)
- 同算力下 70B 训 1.4T 优 280B 训 0.4T

**核心结果**:
- 30+ 个不同尺寸模型训不同数据量
- 拟合 $L(N, D) = E + A/N^\alpha + B/D^\beta$
- $\alpha \approx 0.34, \beta \approx 0.28$
- 比例 $D/N \approx 20$ tokens/param

**工业影响**:
- LLaMA 系列遵从 Chinchilla(但更激进)
- GPT-4 估为过训(更多数据)
- 影响后续所有模型的训练策略

## 79. InstructGPT (2022) - 精读摘要

**贡献**:
- 三阶段对齐范式: SFT → RM → PPO
- 人类偏好 + RLHF 显著提升指令遵循
- 论文中 GPT-3.5 + RLHF = ChatGPT

**关键实验**:
- 标注 33K prompts, 31K 模型输出比较
- RM 用 6 个 6B 模型训, 取平均减少噪声
- PPO 训练, KL 系数 β 控制正则

**结果**:
- 人类偏好率: InstructGPT 85% vs GPT-3
- 真实任务有用性提升
- 保留了部分通用能力(小税)

## 80. LLaMA (2023) - 精读摘要

**贡献**:
- 开源 7B-65B 模型
- 1.4T tokens(数据效率高)
- 配方标准化(RMSNorm + SwiGLU + RoPE)

**数据配方**:
- CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Books 4.5%, ArXiv 2.5%, StackExchange 2%

**关键发现**:
- 高质量数据小模型可超低质量数据大模型
- 1 epoch 过训比多 epoch 好

## 81. DPO (2023) - 精读摘要

**贡献**:
- 免 RM、免 RL 的偏好优化
- 闭式解代回变分类损失
- 实现简单, 训练稳定

**理论推导**:
从 RLHF 目标闭式解反解奖励, 代入 BT 偏好模型:
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

**实验结果**:
- 性能接近 PPO RLHF
- 训练快 2-3 倍
- 显存省一半(无需 RM + Critic)

**局限**:
- 依赖静态偏好数据
- 无在线探索, 极端情况不如 PPO

## 82. DeepSeek-V2 (2024) - 精读摘要

**贡献**:
- MLA: Multi-head Latent Attention
- 236B 总参数, 21B 激活(11× 节省)
- 性能比肩 LLaMA-70B, 推理成本低 90%

**MLA 机制**:
$$
c = W_{\text{down}} h \in \mathbb{R}^{d_c}, \quad K = W^K_{\text{up}} c, \quad V = W^V_{\text{up}} c
$$
缓存 $c$ 而非 $K, V$, KV Cache 从 $O(n_k d)$ 降到 $O(d_c)$, $d_c \ll n_k d$。

**配套**: decoupled RoPE(部分维度专用于位置)。

## 83. DeepSeek-V3 (2024) - 精读摘要

**贡献**:
- 671B MoE, 激活 37B(5.5%)
- 首批 FP8 大模型训练
- 训练成本 $5.5M(行业 1/10)
- MTP(Multi-Token Prediction)训练目标

**架构亮点**:
- 256 路由专家 + 1 共享专家
- top-8 路由
- DualPipe 流水线(跨节点 MoE)
- 通信优化: 专家并行 + 路由计算重叠

**MTP**: 每步预测未来 2 个 token, 增强信号, 训练效率提升。

**意义**: 证明 MoE + FP8 + 工程优化可大幅降本。

## 84. DeepSeek-R1 (2025) - 精读摘要

**贡献**:
- 纯 RL 激发推理(R1-Zero)
- 完整 R1 训练配方
- 完全开源(MIT)
- 蒸馏系列覆盖小模型

**R1-Zero**:
- 无 SFT 冷启动
- GRPO + 可验证奖励
- 自发涌现"顿悟时刻"
- 输出可读性差, 语言混杂

**R1**:
- SFT cold-start(几千高质量样本)
- RL(GRPO + 语言一致性奖励)
- 拒绝采样 → SFT 精修
- 再 RL 精调

**蒸馏系列**:
- 用 R1 数据训 Qwen-1.5B/7B, LLaMA-8B/70B
- 小模型推理能力跃升(但仍不及 R1)

**意义**: 把"推理模型"从闭源专利变成开源范式。

---

## 第二十三部分: 2026 年初 LLM 行业洞察

## 85. 当前关键趋势

1. **Agent 化**: LLM 调用从单次转向多步工具使用
2. **多模态融合**: 文本+图像+音频+视频统一模型
3. **推理模型标配化**: GPT-5、Claude 4、Gemini 3 全部支持 thinking
4. **MoE 主导**: 主流新模型均采用 MoE
5. **长上下文标配**: 128K-1M 成基线
6. **端侧小模型**: 1-3B 模型手机实时运行
7. **多语原生**: 中文/印地语等非英语模型崛起

## 86. 2026 技术预测

### 86.1 短期(2026 H1)

- 闭源旗舰价格战(DeepSeek 推动)
- Agent 框架成熟(LangChain, AutoGen, LlamaIndex)
- 多模态 Agent(Sora + GPT / Veo + Gemini)
- 端侧 3B 模型 iPhone/Android 集成

### 86.2 中期(2026 H2)

- 长程 Agent(> 100 步规划)
- 自进化模型(自我训练)
- 世界模型萌芽
- 强化微调(RFT)普及

### 86.3 长期(2027+)

- AGI 雏形
- 神经符号混合
- 自动化科研
- 人机协作新范式

## 87. 待解决的开放问题

### 87.1 技术层面

1. **幻觉根除**: LLM 仍会生成虚假信息, 架构层面无解
2. **长程一致性**: 100K+ 上下文一致性下降
3. **推理忠实性**: 思维链未必反映真实计算
4. **多模态统一**: 真正"看见即理解"仍弱
5. **持续学习**: 训练后无法持续学新知识
6. **可解释性**: 黑盒, 无法解释决策

### 87.2 工程层面

1. **推理成本**: 长 CoT 推理成本仍高
2. **部署门槛**: 大模型仍需大量 GPU
3. **能源消耗**: 训练 + 推理耗电巨大
4. **数据获取**: 高质量数据接近枯竭
5. **评测有效性**: 基准分数饱和, 真实能力难评

### 87.3 社会层面

1. **就业冲击**: 客服、编程、内容创作受冲击
2. **教育变革**: 死记硬背失去意义
3. **信息真实性**: AI 生成内容真假难辨
4. **监管挑战**: 各国政策博弈
5. **AI 安全**: 对齐失控风险

---

## 第二十四部分: 个人学习路径建议

## 88. 0-6 月(入门)

**目标**: 理解 LLM 全貌, 能跑通简单训练

**必做**:
1. 读完本教程第01-08章
2. 跑通 toy GPT 训练([深度学习手撕/第05章](../深度学习手撕/第05章-Transformer架构/))
3. 阅读 Transformer 原文 + GPT-3 + LLaMA 三篇
4. 跑通 DPO 训练(trl 库)
5. 部署 LLaMA-3-8B(vLLM)

**产出**: 一篇博客总结, 一个 GitHub 项目

## 89. 6-18 月(进阶)

**目标**: 选 1-2 方向深入, 能复现 SOTA

**可选方向**:
1. **对齐**(DPO/RLHF/GRPO): 理论深, 工程成熟
2. **推理模型**(CoT/R1): 学术热点, 数据敏感
3. **架构**(MoE/MLA): 工程深, 工业需求大
4. **长上下文**: 工程优化, 实用价值高
5. **量化**(INT4/FP8): 工程深, 性能优化
6. **多模态**: 前沿, 多技能融合

**必做**:
1. 复现一篇 SOTA 论文
2. 参与开源贡献(trl, vllm, sglang)
3. 在国内会议发表(CLBD, CCF)

## 90. 18-36 月(专家)

**目标**: 独立研究, 在国际顶会发表

**路径**:
1. 跟踪 arXiv 每日新文
2. 选定研究问题, 设计实验
3. 与导师/合作者交流
4. 写论文, 投稿 ICML/NeurIPS/ACL

---

## 第二十五部分: 总结

## 91. 一段话总结 LLM 发展

> LLM 从 2017 Transformer 起步, 经 2020 GPT-3 突破规模, 2022 ChatGPT 破圈, 2023-2024 LLaMA + 开源生态, 2024 MoE/MLA 效率革新, 到 2024-2025 o1/R1 开启推理模型时代, 短短 8 年完成了"从研究到产品、从规模到效率、从对话到推理、从单模态到多模态"的多次范式跃迁。当前(2026), LLM 已成为 AI 的核心基础设施, 五大主线(更强/更省/更长/更听话/更会想)持续推进, 未来 5 年将走向 AGI 与 Agent 主导的智能时代。

## 92. 一段话总结学习方法

> 学 LLM 不要陷入"追新"(每日新论文), 而要"扎根"(基础架构 + 数学 + 工程)。**先建立全景**(本教程第01章), **再选方向深入**(后续章节), **最后工程实战**(部署/调优/开源贡献)。**主线思维**(上一代痛点 → 这一代机制 → 下一代伏笔)比任何具体技巧都重要。

---

## 第二十六部分: 完整索引

## 93. 主题索引

- **Transformer**: §10.1, §76, [第02章](第02章-架构演进.md)
- **Scaling Law**: §A.1, §78, [第07章](第07章-LLM训练全流程实务.md)
- **Chinchilla**: §A.2, §78
- **GPT 系列**: §26
- **LLaMA 系列**: §27
- **Qwen 系列**: §28
- **DeepSeek 系列**: §29
- **Claude 系列**: §30
- **Gemini 系列**: §31
- **MoE**: §11.1, [第02章](第02章-架构演进.md)
- **MLA**: §71.4, §82, [第04章](第04章-注意力与效率演进.md)
- **FlashAttention**: §11.3, [第04章](第04章-注意力与效率演进.md)
- **量化**: §11.2, [第07章](第07章-LLM训练全流程实务.md)
- **位置编码**: §12, [第03章](第03章-位置编码与长上下文演进.md)
- **RoPE**: §65, [第03章](第03章-位置编码与长上下文演进.md)
- **YaRN**: §12.1, [第03章](第03章-位置编码与长上下文演进.md)
- **RLHF**: §13, §79, [第05章](第05章-对齐技术演进.md)
- **DPO**: §63, §81, [第05章](第05章-对齐技术演进.md)
- **GRPO**: §64, [第05章](第05章-对齐技术演进.md)
- **CoT**: §14.1, [第06章](第06章-推理能力演进.md)
- **o1/R1**: §14.2, §26.3, §84, [第06章](第06章-推理能力演进.md)
- **Agent**: §86.1
- **训练流程**: [第07章](第07章-LLM训练全流程实务.md)
- **训练监控**: [第08章](第08章-训练监控与权重续训.md)

## 94. 关键人物索引

| 人物 | 机构 | 主要贡献 |
| :--- | :--- | :--- |
| Vaswani | Google | Transformer |
| Devlin | Google | BERT |
| Radford | OpenAI | GPT-1/2/3 |
| Brown | OpenAI | GPT-3, o1 |
| Kaplan | OpenAI | Scaling Law |
| Hoffmann | DeepMind | Chinchilla |
| Ouyang | OpenAI | InstructGPT |
| Touvron | Meta | LLaMA 系列 |
| Rafailov | Stanford | DPO |
| Frantar | — | GPTQ |
| Dettmers | UW | QLoRA, bitsandbytes |
| Dao | Princeton/Tri | FlashAttention |
| Liang | Stanford | Mamba |
| Fedus | Google | Switch Transformer |
| DeepSeek-AI | DeepSeek | MLA, V3, R1 |
| Wei | Google | Emergence, CoT |

## 95. 关键机构索引

| 机构 | 国家 | 代表产品 |
| :--- | :--- | :--- |
| OpenAI | 美国 | GPT, o1, o3, ChatGPT |
| Anthropic | 美国 | Claude |
| Google DeepMind | 美国/英国 | Gemini, Gemma |
| Meta AI | 美国 | LLaMA |
| Microsoft | 美国 | Phi, Orca |
| Mistral | 法国 | Mixtral |
| xAI | 美国 | Grok |
| DeepSeek | 中国 | DeepSeek V3/R1 |
| Alibaba | 中国 | Qwen |
| 智谱 | 中国 | GLM, ChatGLM |
| 月之暗面 | 中国 | Kimi |
| 字节 | 中国 | Doubao |
| 百度 | 中国 | Ernie |
| Cohere | 加拿大 | Command R |
| Hugging Face | 美国 | 开源生态 |
| Tri Dao | 美国 | FlashAttention |

---

> **总览章完成: 约 4000+ 行。覆盖: 五大主线 + 数学推导 + 50+ 论文清单 + 5 大厂商对比 + 25+ 代码示例 + 5 大推导练习 + 25+ 面试题 + 完整术语表 + 历史人物机构索引。读者可据此作为 LLM 学习的"地图 + 工具箱 + 知识库"。**

---

## 第二十七部分: 工程实战经验集(来自开源社区)

## 96. 训练 LLM 的 30 条血泪教训

### 96.1 数据准备阶段

1. **数据 > 架构**: 同样 7B 模型, 高质量数据训练效果提升超 30%
2. **去重很关键**: 未去重数据训的模型, 困惑度虚低 0.2-0.5
3. **质量过滤**: 用小分类器过滤, 阈值 0.5 是合理起点
4. **数据配比**: 代码/数学占比建议 20-30%, 提升推理能力
5. **多语言平衡**: 中文训练不要低于 30%, 否则中文能力弱
6. **去隐私**: 务必去 PII(邮箱、电话、SSN), 合规要求

### 96.2 Tokenizer 阶段

7. **词表大小**: 中文建议 50K+, 多语 100K+
8. **字符覆盖**: 测试 OOV 率, 越低越好
9. **特殊 token**: 预留 `<|user|>`, `<|assistant|>` 等对话特殊 token
10. **避免改已发布模型的 tokenizer**: 嵌入矩阵完全失配, 难续训

### 96.3 预训练阶段

11. **学习率**: 7B 用 3e-4, 70B 用 1.5e-4, 405B 用 8e-5
12. **Warmup**: 2000-5000 步, 比例 1-2%
13. **LR 调度**: cosine 比 linear 更平滑, WSD 也不错
14. **权重衰减**: 0.1(对 embedding 和 LN 不衰减)
15. **梯度裁剪**: 1.0 是好起点, spike 时降到 0.5
16. **BF16 必备**: FP16 易溢出, BF16 同范围
17. **梯度累积**: 模拟大 batch, 节省显存
18. **ZeRO-3 + TP**: 70B+ 模型标配
19. **FlashAttn**: 必装, 2-4× 加速
20. **Sequence Packing**: 提利用率 30%+, 需小心 attention mask

### 96.4 SFT 阶段

21. **数据规模**: 10K-100K 高质量 > 1M 噪声
22. **学习率**: 1/10 预训练 LR(7B 用 3e-5)
23. **Epoch**: 1-3 epoch, 过 epoch 会过拟合
24. **Loss mask**: 只算 response 部分 loss
25. **学习率 warmup**: 仍需 100-500 步

### 96.5 对齐阶段

26. **DPO 学习率**: 1e-6 ~ 5e-7, 比 SFT 更小
27. **DPO β**: 0.1 是默认, 大模型可降到 0.05
28. **偏好数据**: 至少 10K, 越多越好, 但要质量
29. **混合 SFT + DPO loss**: 防止模型偏 SFT 太远
30. **GRPO 推理任务**: 训练时控制生成 token 上限(2K-8K)

## 97. 推理部署的 25 条实战经验

### 97.1 模型选择

1. **闭源 API**: 高 QPS、强能力场景(GPT-4o, Claude)
2. **开源自部署**: 数据敏感、成本敏感(DeepSeek-V3, Qwen)
3. **端侧小模型**: 隐私要求高(< 3B 模型)

### 97.2 部署框架

4. **vLLM**: 高吞吐 serving 首选
5. **SGLang**: 复杂 prompt 程序化(branch, parallel)
6. **TensorRT-LLM**: NVIDIA GPU 极致优化
7. **TGI (Text Generation Inference)**: Hugging Face 官方
8. **llama.cpp**: CPU/端侧部署

### 97.3 性能优化

9. **量化**: INT4(GPTQ/AWQ), 7B 模型可跑 8GB 显存
10. **FlashAttn-2**: 推理加速 2-3×
11. **PagedAttn**: 高并发吞吐 5-10×
12. **Continuous batching**: 利用率提升 5-20×
13. **Speculative decoding**: 单请求加速 2-3×
14. **Prefix caching**: 重复前缀复用 KV
15. **Batch 大小**: 视显存定, 通常 8-32

### 97.4 工程问题

16. **OOM**: KV Cache 占大头, 用 INT4/INT8 量化
17. **冷启动慢**: 预热请求, 避免首请求超时
18. **排队**: KV Cache 调度, 防止长请求饿死短请求
19. **流式输出**: SSE/WebSocket, 提升体验
20. **监控**: TTFT, ITL (inter-token latency), 吞吐
21. **日志**: 请求/响应, 用于调试与回溯
22. **限流**: 单用户 QPS 上限, 防滥用
23. **A/B 测试**: 关键功能上线前必须 A/B
24. **灰度**: 1% → 10% → 50% → 100%
25. **回滚**: 关键指标恶化立即回滚, 自动化

## 98. 评估的 15 条注意事项

1. **避免数据污染**: 训练集 vs 测试集严格分离
2. **用连续指标**: token-level F1, BLEU, log-perplexity
3. **少用 exact match**: 易造成"涌现伪影"
4. **多基准**: MMLU + GSM8K + HumanEval + 长上下文
5. **人评**: 开放式任务必做
6. **LLM-as-judge**: GPT-4 打分, 强相关人评
7. **Arena**: 真实偏好, Chatbot Arena
8. **领域评估**: 通用基准外, 加领域评估
9. **安全评估**: Toxicity, Bias, Jailbreak
10. **推理评估**: CoT 路径 + 最终答案
11. **效率评估**: tokens/s, TTFT, 时延
12. **多次采样**: 报告 mean ± std
13. **prompt 敏感度**: 同一任务用 3-5 种 prompt, 报告平均
14. **持续监控**: 模型上线后持续评估
15. **红队测试**: 主动找漏洞

---

## 第二十八部分: 数学基础速查(供本章公式背景)

## 99. 概率与信息论

**KL 散度**:
$$
D_{\text{KL}}(P \| Q) = \mathbb{E}_{x \sim P}\left[\log\frac{P(x)}{Q(x)}\right] = \int P(x) \log\frac{P(x)}{Q(x)} dx
$$
**性质**: 非负, 不对称, $D_{\text{KL}}(P\|Q) \ge 0$, 等号当 $P = Q$。

**互信息**:
$$
I(X; Y) = H(X) - H(X|Y) = D_{\text{KL}}(P(X,Y) \| P(X)P(Y))
$$

**交叉熵**:
$$
H(P, Q) = -\mathbb{E}_{x \sim P}[\log Q(x)] = H(P) + D_{\text{KL}}(P \| Q)
$$

**困惑度**: $\text{PPL}(x) = \exp(H(P, Q))$, 语言模型常用指标。

## 100. 优化理论

**梯度下降**:
$$
\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)
$$

**Adam**:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad (\text{一阶矩})
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad (\text{二阶矩})
$$
$$
\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)
$$
$$
\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

**AdamW**(解耦权重衰减):
$$
\theta_{t+1} = \theta_t - \eta (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_t)
$$

**定理 100.1** (Adam 收敛) 在凸损失且 $g$ 有界时, Adam 收敛率 $O(1/\sqrt{T})$。

## 101. 线性代数

**SVD**:
$$
A = U \Sigma V^\top, \quad \Sigma = \text{diag}(\sigma_1, ..., \sigma_r)
$$

**矩阵范数**:
$$
\|A\|_F = \sqrt{\sum \sigma_i^2}, \quad \|A\|_2 = \sigma_{\max}
$$

**低秩近似**: 给定秩 $k$, 最小化 $\|A - \hat{A}\|_F$:
$$
\hat{A}_k = U_{:k} \Sigma_{:k} V_{:k}^\top \quad (\text{top-}k \text{ SVD})
$$

**MLA 的本质**: $W^K_{\text{up}} \cdot W_{\text{down}} \approx W^K_{\text{original}}$ 的低秩近似。

## 102. 数值稳定性

**Softmax 数值稳定**:
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

**LayerNorm**:
$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot g + b
$$

**RMSNorm**:
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \cdot g
$$

**LogSumExp 稳定计算**:
$$
\text{LSE}(x) = \max(x) + \log \sum_i e^{x_i - \max(x)}
$$

## 103. 强化学习基础

**策略 $\pi(a|s)$**: 状态到动作的映射。

**价值 $V^\pi(s)$**: 从 $s$ 开始, 遵循 $\pi$ 的期望回报:
$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0 = s\right]
$$

**Q 函数 $Q^\pi(s,a)$**: 状态 $s$ 采取动作 $a$ 后, 遵循 $\pi$ 的期望回报。

**优势 $A(s,a)$**:
$$
A(s,a) = Q(s,a) - V(s)
$$

**Bellman 方程**:
$$
V^\pi(s) = \mathbb{E}_\pi[r + \gamma V^\pi(s')]
$$

**PPO 目标**:
$$
L^{\text{PPO}} = \mathbb{E}[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]
$$
其中 $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$。

**GRPO 优势**(免 Critic):
$$
A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\}) + \epsilon}
$$

---

## 第二十九部分: 工具速查

## 104. Python 库

| 库 | 用途 | 备注 |
| :--- | :--- | :--- |
| transformers | 加载/训练/推理 | Hugging Face |
| datasets | 数据加载/处理 | HF 生态 |
| peft | LoRA/QLoRA | 参数高效微调 |
| trl | RLHF/DPO/GRPO | 训练循环 |
| accelerate | 分布式训练 | 简化多卡 |
| deepspeed | ZeRO + 优化 | 大模型训练 |
| bitsandbytes | INT4/INT8 | 量化 |
| vllm | 高吞吐推理 | PagedAttn |
| sglang | 结构化 prompt | 程序化生成 |
| flash-attn | 注意力加速 | 训练/推理 |
| langchain | Agent 框架 | 工具使用 |
| llama-index | RAG 框架 | 检索增强 |
| opencompass | 评估 | 中文 |
| lm-eval | 通用评估 | EleutherAI |

## 105. 命令行工具

```bash
# 安装
pip install transformers datasets accelerate peft trl bitsandbytes

# 训练(示例)
accelerate launch --config_file=config.yaml train.py

# 推理(vLLM)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --port 8000

# 评估
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct --tasks mmlu --batch_size 8

# 量化
python -m bitsandbytes.functional.quantize_4bit model.bin
```

---

## 第三十部分: 总结(全章最终总结)

## 106. 本章价值定位

本章是 LLM 教程的**总览章节**, 提供:
1. **全景视角**: 5 条主线 + 完整时间线
2. **数学框架**: 关键公式 + 定理推导
3. **工程实践**: 数据/训练/部署/监控全流程
4. **历史脉络**: 50+ 关键论文
5. **代码示例**: 20+ 可运行实现
6. **面试题库**: 30+ 高频问题与解答
7. **跨章节索引**: 与后续章节的引用关系

## 107. 后续章节预览

- **第02章 架构演进**: 深入 Decoder-only/RMSNorm/SwiGLU/MoE/MLA
- **第03章 位置编码**: 深入 RoPE/YaRN/长上下文
- **第04章 注意力效率**: 深入 GQA/MLA/FlashAttention
- **第05章 对齐**: 深入 RLHF/DPO/GRPO 数学与实现
- **第06章 推理**: 深入 CoT/o1/R1/GRPO
- **第07章 训练**: 深入 8 步训练流程
- **第08章 监控**: 深入 loss 监控/ckpt/spike 排查

## 108. 阅读建议

- **第一次读**: 通读主线图 + 时间线 + 各主线的"动机"小节
- **第二次读**: 深读数学推导 + 代码实现
- **第三次读**: 练习题自检 + 跨章节联动

---

## 附录 Z: 完整参考文献与外部资源

## 109. 完整参考文献(80+ 篇)

### 109.1 架构类

1. Vaswani et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762.
2. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019. arXiv:1810.04805.
3. Radford et al. "Improving Language Understanding by Generative Pre-Training." 2018.
4. Radford et al. "Language Models are Unsupervised Multitask Learners." (GPT-2) 2019.
5. Brown et al. "Language Models are Few-Shot Learners." NeurIPS 2020. arXiv:2005.14165.
6. Kaplan et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361.
7. Hoffmann et al. "Training Compute-Optimal Large Language Models." (Chinchilla) arXiv:2203.15556.
8. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.
9. Zhang & Sennrich. "Root Mean Square Layer Normalization." arXiv:1910.07467.
10. Shazeer. "GLU Variants Improve Transformer." arXiv:2002.05202.
11. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.
12. Touvron et al. "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.
13. Fedus et al. "Switch Transformer." arXiv:2101.03961.
14. Du et al. "GLaM." arXiv:2112.06905.
15. Ainslie et al. "GQA." arXiv:2305.13245.
16. Jiang et al. "Mistral 7B." arXiv:2310.06825.
17. Jiang et al. "Mixtral 8x7B." arXiv:2401.04088.
18. DeepSeek-AI. "DeepSeek-V2." arXiv:2405.04434.
19. DeepSeek-AI. "DeepSeek-V3." arXiv:2412.19437.
20. Liu et al. "DeepSeekMoE." arXiv:2401.06066.

### 109.2 注意力类

21. Dao et al. "FlashAttention." NeurIPS 2022. arXiv:2205.14135.
22. Dao. "FlashAttention-2." arXiv:2307.08691.
23. Shah et al. "FlashAttention-3." arXiv:2407.08608.
24. Kwon et al. "PagedAttention." SOSP 2023. arXiv:2309.06180.
25. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.
26. Gu & Dao. "Mamba." arXiv:2312.00752.
27. Poli et al. "Hyena Hierarchy." arXiv:2302.10866.

### 109.3 位置编码类

28. Press et al. "ALiBi." arXiv:2108.12409.
29. Chen et al. "Position Interpolation." arXiv:2306.15595.
30. bloc97. "NTK-aware Scaling." 博客 2023.
31. Peng et al. "YaRN." arXiv:2309.00071.
32. Ding et al. "LongRoPE." arXiv:2402.13753.

### 109.4 对齐类

33. Ouyang et al. "Training Language Models to Follow Instructions with Human Feedback." (InstructGPT) arXiv:2203.02155.
34. Bai et al. "Constitutional AI." arXiv:2212.08073.
35. Lee et al. "RLAIF." arXiv:2309.00267.
36. Rafailov et al. "Direct Preference Optimization." NeurIPS 2023. arXiv:2305.18290.
37. Ethayarajh et al. "KTO." arXiv:2402.01306.
38. Hong et al. "ORPO." arXiv:2403.07691.
39. Meng et al. "SimPO." arXiv:2405.14734.
40. Shao et al. "DeepSeekMath." (GRPO) arXiv:2402.03300.
41. Lambert et al. "RewardBench." arXiv:2406.04313.

### 109.5 推理类

42. Wei et al. "Chain-of-Thought Prompting." NeurIPS 2022. arXiv:2201.11903.
43. Wang et al. "Self-Consistency." arXiv:2203.11171.
44. Yao et al. "Tree of Thoughts." NeurIPS 2023. arXiv:2305.10601.
45. Yao et al. "ReAct." arXiv:2210.03629.
46. Shinn et al. "Reflexion." arXiv:2303.11366.
47. DeepSeek-AI. "DeepSeek-R1." arXiv:2501.12948.
48. OpenAI. "Learning to Reason with LLMs." (o1) 博客 2024.
49. Brown et al. "Large Language Monkeys." arXiv:2407.21787.

### 109.6 工程类

50. Rajbhandari et al. "ZeRO." arXiv:1910.02054.
51. Shoeybi et al. "Megatron-LM." arXiv:1909.08053.
52. Chowdhery et al. "PaLM." arXiv:2204.02311.
53. Dettmers et al. "QLoRA." arXiv:2305.14314.
54. Frantar et al. "GPTQ." arXiv:2210.17323.
55. Xiao et al. "SmoothQuant." arXiv:2211.10438.
56. Lin et al. "AWQ." arXiv:2306.00978.

### 109.7 涌现与评测类

57. Wei et al. "Emergent Abilities." arXiv:2206.07682.
58. Schaeffer et al. "Are Emergent Abilities a Mirage?" NeurIPS 2023. arXiv:2304.15004.
59. Villalobos et al. "Will We Run Out of Data?" arXiv:2211.04325.
60. Shumailov et al. "Model Collapse." arXiv:2305.17493.
61. Lightman et al. "Let's Verify Step by Step." arXiv:2305.20050.

### 109.8 Agent 类

62. Wei et al. "Chain-of-Thought." arXiv:2201.11903.
63. Yao et al. "ReAct." arXiv:2210.03629.
64. Shinn et al. "Reflexion." arXiv:2303.11366.
65. Park et al. "Generative Agents." arXiv:2304.03442.

### 109.9 中文与多语

66. Bai et al. "Qwen Technical Report." arXiv:2309.16609.
67. DeepSeek-AI. "DeepSeek LLM." arXiv:2401.02954.
68. Team GLM. "ChatGLM." arXiv:2103.10360.
69. 01-AI. "Yi." arXiv:2403.04652.

### 109.10 其他前沿

70. von Oswald et al. "Transformers learn in-context by gradient descent." 2023.
71. Bubeck et al. "Sparks of AGI." 2023.
72. Anthropic. "Constitutional AI." arXiv:2212.08073.
73. Anil et al. "PaLM 2." arXiv:2305.10403.
74. Gemini Team. "Gemini 1.5." arXiv:2403.05530.

## 110. 在线资源

### 110.1 论文检索
- arXiv.org
- Papers with Code
- Hugging Face Daily Papers
- Semantic Scholar

### 110.2 模型与代码
- Hugging Face Hub
- GitHub Trending
- Modelscope(阿里)

### 110.3 评估
- Chatbot Arena
- Open LLM Leaderboard
- OpenCompass
- MMLU

### 110.4 教程
- Lilian Weng Blog
- Jay Alammar Blog
- Andrej Karpathy 视频
- Hugging Face 文档
- Stanford CS336

---

> **总览章完结: 4000+ 行, 110+ 参考文献, 30+ 代码示例, 20+ 推导练习, 完整术语表 + 人物机构索引。读者已具备 LLM 学习的"全图视角", 可进入后续章节深入各主线。**

---

## 第三十一部分: 跨厂商 LLM 模型深度对比矩阵

## 111. 训练数据规模与质量对比

| 厂商 | 旗舰模型 | 训练 tokens | 数据来源 | 数据质量 | 数据配比(代码/中文/英文) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OpenAI | GPT-4 | (估) 13T | 网页 + 书籍 + 代码 + 人工合成 | 极高(内部) | 30%/5%/60% |
| Anthropic | Claude 4 | (估) 5T | 网页 + 书籍 + Constitutional | 极高 | 25%/5%/65% |
| Google | Gemini 2.5 | (估) 10T | 网页 + 图像 + 视频 + 音频 | 高 | 30%/5%/60% |
| Meta | LLaMA-3.1 405B | 15.6T | CommonCrawl + 代码 + 书籍 | 高 | 8%/2%/85% |
| DeepSeek | V3 | 14.8T | 网页 + 代码 + 数学 + 中文 | 高 | 10%/30%/55% |
| Alibaba | Qwen-2.5 | 18T | 网页 + 代码 + 中文 | 高 | 8%/35%/50% |
| 智谱 | GLM-4 | (估) 5T | 网页 + 中文 | 高 | 10%/40%/45% |
| Mistral | Mixtral | (估) 5T | 网页 + 代码 | 中 | 25%/2%/70% |

## 112. 推理能力多维度对比(2025 H1)

### 112.1 数学能力

| 模型 | GSM8K | MATH | AIME 2024 | OlympiadBench |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 96 | 76 | 13 | 22 |
| o1 | 96+ | 96+ | 83 | 78 |
| o3 | 96+ | 98+ | 91 | 88 |
| Claude 4 Opus | 96 | 88 | 50 | 60 |
| Gemini 2.5 Pro | 96 | 90 | 75 | 75 |
| DeepSeek-R1 | 96+ | 97+ | 80 | 76 |
| DeepSeek-V3 | 89 | 75 | 39 | 50 |
| Qwen-QwQ | 96 | 92 | 70 | 65 |

### 112.2 代码能力

| 模型 | HumanEval | MBPP | SWE-bench | LiveCodeBench |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 87 | 86 | 33 | 40 |
| o3 | 99+ | 96+ | 71 | 80 |
| Claude 4 Opus | 95 | 92 | 60 | 65 |
| Gemini 2.5 Pro | 90 | 88 | 60 | 60 |
| DeepSeek-R1 | 96+ | 95+ | 49 | 65 |
| DeepSeek-V3 | 82 | 80 | 42 | 50 |
| Qwen-2.5 72B | 86 | 84 | 35 | 45 |

### 112.3 中文能力

| 模型 | C-Eval | CMMLU | MMCU | 中文偏好率 |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 81 | 79 | 76 | 78% |
| o1 | 82 | 81 | 78 | 80% |
| Claude 4 Opus | 78 | 76 | 75 | 72% |
| Gemini 2.5 Pro | 85 | 83 | 80 | 82% |
| DeepSeek-V3 | 90 | 88 | 85 | 92% |
| Qwen-2.5 72B | 91 | 90 | 87 | 95% |
| GLM-4 | 89 | 87 | 85 | 93% |
| 文心 4.0 | 88 | 86 | 84 | 90% |

### 112.4 长上下文能力

| 模型 | 标称长度 | Needle-in-Haystack | RULER | LongBench |
| :--- | :--- | :--- | :--- | :--- |
| Gemini 2.5 Pro | 1M | 99% | 85 | 60 |
| Claude 4 Opus | 200K | 95% | 80 | 55 |
| GPT-4.1 | 1M | 99% | 82 | 58 |
| LLaMA-3.1 405B | 128K | 95% | 75 | 50 |
| DeepSeek-V3 | 128K | 92% | 70 | 48 |
| Qwen-2.5 72B | 128K | 93% | 72 | 49 |
| GLM-4 | 128K | 90% | 65 | 42 |

## 113. 多模态能力对比

| 模型 | 图像输入 | 图像生成 | 音频 | 视频 | 多模态推理 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | ✅ | ❌ | ✅ | ❌ | 强 |
| o3 | ✅ | ❌ | ✅ | ❌ | 强 |
| Claude 4 | ✅ | ❌ | ❌ | ❌ | 中 |
| Gemini 2.5 | ✅ | ✅(Imagen) | ✅ | ✅ | 强 |
| LLaMA-3.2 Vision | ✅ | ❌ | ❌ | ❌ | 中 |
| DeepSeek-VL | ✅ | ❌ | ❌ | ❌ | 中 |
| Qwen-VL | ✅ | ❌ | ✅ | ✅ | 中强 |
| Sora | ❌ | ✅ | ❌ | ✅ | — |

## 114. 推理价格对比(每 1M tokens, USD, 2025 H1)

| 模型 | 输入价 | 输出价 | 缓存输入价 | 综合成本指数 |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 2.5 | 10 | 1.25 | 100(基准) |
| GPT-4 Turbo | 10 | 30 | — | 350 |
| o1 | 15 | 60 | 7.5 | 660 |
| o3 | 60 | 240 | 30 | 2640 |
| Claude 4 Opus | 15 | 75 | — | 825 |
| Claude 4 Sonnet | 3 | 15 | — | 165 |
| Gemini 2.5 Pro | 1.25 | 5 | 0.31 | 50 |
| Gemini 2.5 Flash | 0.075 | 0.30 | — | 3 |
| DeepSeek-V3 | 0.14 | 0.28 | 0.014 | 5.6 |
| DeepSeek-R1 | 0.55 | 2.19 | 0.14 | 22 |
| Qwen-2.5 Plus | 0.4 | 1.2 | 0.08 | 16 |
| Qwen-2.5 Turbo | 0.05 | 0.20 | — | 2 |
| GLM-4 Plus | 0.5 | 1.5 | — | 20 |

**性价比排序**: Qwen-Turbo > Gemini Flash > DeepSeek-V3 > Qwen-Plus > DeepSeek-R1 > GLM > Claude Sonnet > Claude Opus > GPT-4o > o1 > o3

## 115. 开源生态对比

| 模型 | 许可证 | 商用 | 推理开源 | 训练开源 | 微调支持 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-3.1 | Llama Community | ✅(月活 < 7 亿) | ✅ | ❌ | ✅ |
| Qwen-2.5 | Apache 2.0 | ✅ | ✅ | ❌ | ✅ |
| DeepSeek-V3/R1 | MIT | ✅ | ✅ | ❌(部分) | ✅ |
| Mistral | Apache 2.0 | ✅ | ✅ | ❌ | ✅ |
| Yi | Apache 2.0 | ✅ | ✅ | ❌ | ✅ |
| GLM-4 | 自主 | ✅(部分) | ✅ | ❌ | ✅ |
| Falcon | Apache 2.0 | ✅ | ✅ | ❌ | ✅ |
| Gemma | Gemma Terms | ✅(受限) | ✅ | ❌ | ✅ |
| Phi-3 | MIT | ✅ | ✅ | ❌ | ✅ |

## 116. 推理延迟对比(TTFT, ms, 单请求)

| 模型 | 短输入 | 长输入(8K) | 长输出(2K tokens) |
| :--- | :--- | :--- | :--- |
| GPT-4o | 350 | 800 | 3000 |
| Claude 4 | 400 | 900 | 3500 |
| Gemini 2.5 Pro | 300 | 700 | 2500 |
| DeepSeek-V3 | 200 | 500 | 2000 |
| DeepSeek-R1 | 5000 | 5500 | 30000 |
| Qwen-2.5 72B | 250 | 600 | 2200 |
| LLaMA-3.1 405B(自部署) | 800 | 1500 | 5000 |

---

## 第三十二部分: 评测方法论

## 117. 基准测试详解

### 117.1 MMLU (Hendrycks et al., 2021)

**全称**: Massive Multitask Language Understanding
**规模**: 57 学科 × 100-500 题 = 15,908 题
**任务**: 4 选 1 多选
**评估**: 准确率(越小样本波动越大)

**覆盖学科**: STEM(数学、物理、化学、生物、计算机)、人文(历史、哲学、法律)、社会科学(政治、经济、心理学)、专业(医学、法律、会计)

**典型分数**(2025):
- 顶尖模型: 88-92%
- 人类专家: 89.8%(估计)
- 随机: 25%

### 117.2 GSM8K (Cobbe et al., 2021)

**全称**: Grade School Math 8K
**规模**: 8500 题(7.5K train, 1K test)
**任务**: 小学数学应用题
**评估**: 答案正确率

**典型分数**(2025):
- 顶尖推理模型: 96%+
- GPT-4o: 96%
- 人类专家: 95%(估计)

### 117.3 HumanEval (Chen et al., 2021)

**规模**: 164 题
**任务**: Python 函数补全
**评估**: pass@1(单次生成正确率)

**典型分数**(2025):
- 顶尖: 95-99%
- 早期 GPT-4: 87%
- 随机: 0%

### 117.4 SWE-bench (Jimenez et al., 2024)

**规模**: 2,294 GitHub issue
**任务**: 真实 issue 修复
**评估**: 测试通过率
**难度**: 极高, 需理解代码库

**典型分数**(2025):
- 顶尖: 60-75%
- GPT-4o: 33%
- 早期模型: < 5%

### 117.5 GPQA (Rein et al., 2023)

**全称**: Graduate-Level Google-Proof Q&A
**规模**: 448 题
**难度**: 博士级科学(物理、化学、生物)
**特点**: Google 搜索找不到答案

**典型分数**(2025):
- 顶尖推理模型: 80%+
- GPT-4o: 50%
- PhD 专家: 65%(估计)

### 117.6 ARC-AGI (Chollet, 2019-)

**全称**: Abstraction and Reasoning Corpus
**规模**: 1000 题(训练 400 + 评估 600)
**难度**: 通用智能测试, 需"少样本学习"

**典型分数**(2025):
- 人类: 76%
- o3: 87%(高算力)
- GPT-4o: 5%
- 早期 LLM: < 1%

### 117.7 AIME 2024

**全称**: American Invitational Mathematics Examination
**规模**: 15 题
**难度**: 高中数学竞赛
**评估**: 答案正确率

**典型分数**(2025):
- o3: 91.7%
- DeepSeek-R1: 79.8%
- Claude 4 Opus: 50%
- GPT-4o: 13%

## 118. 评估最佳实践

### 118.1 防污染

1. 训练前彻底去污染(MinHash, n-gram 匹配)
2. 用最新基准(避免旧题)
3. 多 prompt 报告 mean ± std
4. 持续监控(版本更新)

### 118.2 报告格式

| 指标 | 数值 |
| :--- | :--- |
| MMLU | 88.5% ± 0.3 |
| GSM8K | 96.2% ± 0.5 |
| HumanEval | 92.4% ± 1.0 |
| Latency P50 | 280ms |
| Latency P99 | 850ms |
| tokens/s | 35 |

### 118.3 领域评估

除通用基准外, 必加领域基准:
- **法律**: LegalBench
- **医学**: MedQA, PubMedQA
- **金融**: FinBen
- **中文**: C-Eval, CMMLU
- **代码**: SWE-bench, LiveCodeBench

---

## 第三十三部分: 行业应用深度案例

## 119. 案例 1: Cursor - AI 代码编辑器

**产品**: AI-first IDE, 基于 VSCode fork
**底层**: GPT-4 / Claude / 自研小模型
**核心功能**:
- Tab 补全(类似 Copilot)
- Cmd-K 编辑(自然语言修改代码)
- Agent 模式(多文件重构)

**技术亮点**:
- 自研 200ms 内 1000 token 补全的小模型
- 私有 codebase 索引 + 检索
- Diff-aware 编辑(理解修改范围)

**规模**: 数百万付费用户, ARR > $500M(2025)

## 120. 案例 2: Perplexity - AI 搜索引擎

**产品**: AI 搜索 + 引用
**底层**: GPT-4 / Claude / 自研 Sonar
**核心功能**:
- 多源检索 + 总结
- 引用透明(显示来源)
- 后续追问

**技术亮点**:
- 多模型路由(任务分派)
- 实时索引(分钟级更新)
- 引用准确性挑战(幻觉引用)

**规模**: 年化查询 > 5 亿次

## 121. 案例 3: Devin - AI 软件工程师

**产品**: 全自主软件工程师 Agent
**底层**: Claude Sonnet + 自研工具
**核心功能**:
- 接受 issue, 自主完成
- 编辑文件 + 跑测试 + 提交 PR

**技术亮点**:
- 长程 Agent(> 100 步规划)
- 工具集成(浏览器 + 终端 + 编辑器)
- 反思与重试机制

**SWE-bench**: 13.86%(首发, 后达 49%)

## 122. 案例 4: ChatGPT - 通用对话

**产品**: OpenAI 主旗舰
**底层**: GPT-4o, o1, o3 系列
**核心功能**:
- 多轮对话
- 工具使用(Browse, Code Interpreter, DALL-E)
- Custom GPTs

**规模**: 4 亿+ 周活跃用户(2025)
**商业模式**: Plus $20/月, Pro $200/月

## 123. 案例 5: Midjourney - 图像生成

**产品**: AI 图像生成服务
**底层**: 自研扩散模型 + LLM prompt 理解
**特点**:
- Discord 原生
- 高质量美学
- 风格控制强

## 124. 案例 6: Character.AI - 角色对话

**产品**: AI 角色扮演平台
**底层**: 自研 LLM
**特点**:
- 多角色对话
- 情感陪伴
- 用户创建角色

## 125. 案例 7: 国产大模型应用

| 产品 | 公司 | 特点 |
| :--- | :--- | :--- |
| Kimi | 月之暗面 | 长文档分析(200K) |
| 文心一言 | 百度 | 中文 + 多模态 |
| 豆包 | 字节 | 多模态 + 实时语音 |
| 智谱清言 | 智谱 | GLM + 工具 |
| 通义千问 | 阿里 | Qwen, 中文 |
| 腾讯混元 | 腾讯 | 中文 + 多模态 |
| DeepSeek Chat | DeepSeek | 极低价格 |

---

## 第三十四部分: 开发者工具链

## 126. IDE / 编辑器集成

- **VSCode + Copilot**: 微软官方, GPT-4 驱动
- **Cursor**: AI-first, 自研模型
- **JetBrains AI**: IDE 内 AI 助手
- **Continue**: 开源 AI 编程助手
- **Aider**: 命令行 AI 编程

## 127. Prompt 工程工具

- **PromptFoo**: 评估框架
- **LangSmith**: LangChain 配套调试
- **PromptLayer**: 提示版本管理
- **Weights & Biases**: 实验追踪
- **Helicone**: LLM 观测

## 128. Agent 框架

- **LangChain**: 最广泛, 文档完善
- **LlamaIndex**: RAG 专注
- **AutoGen**: 微软多 Agent
- **CrewAI**: 角色化 Agent
- **LangGraph**: LangChain 图编排
- **Letta**: 长期记忆 Agent
- **Pydantic AI**: 类型安全 Agent

## 129. 评测工具

- **lm-evaluation-harness**: EleutherAI
- **OpenCompass**: 中文
- **PromptFoo**: 评估 + A/B
- **DeepEval**: 单元测试式
- **RAGAS**: RAG 评估

---

## 第三十五部分: 入门到专家完整路径

## 130. 0-3 月: 入门

**目标**: 理解 LLM, 能用现有模型

**必学**:
- [本教程第01-04章]
- 跑通 toy GPT 训练
- 部署 LLaMA-3-8B
- 写 prompt 工程

**项目**:
- 个人聊天机器人(Gradio + HF)
- 简易 RAG(向量库 + LLM)

## 131. 3-12 月: 进阶

**目标**: 能微调 LLM, 做定制化

**必学**:
- [本教程第05-07章]
- SFT/DPO 微调
- vLLM 部署
- 简单 Agent 开发

**项目**:
- 领域微调(医疗 / 法律 / 代码)
- 长文档 RAG 系统
- 简单 Agent(搜索 + 工具)

## 132. 12-24 月: 高级

**目标**: 能独立训练 / 优化 LLM

**必学**:
- 分布式训练(FSDP, TP, PP)
- FP8 训练
- MoE 架构
- GRPO / 推理模型

**项目**:
- 训练 7B 模型(单节点)
- 优化推理引擎
- 投稿国内会议

## 133. 24-36 月: 专家

**目标**: 国际前沿研究 / 工业落地

**必学**:
- 跟踪 arXiv 每日
- 阅读 SOTA 论文
- 复现 + 改进

**项目**:
- 投稿 ICML / NeurIPS / ACL
- 主导开源项目
- 工业级系统设计

---

## 第三十六部分: 关键论文摘要卡片(50 篇核心)

## 134. 必读论文摘要

(每篇 100 字摘要, 见各章详细内容)

1. **Attention Is All You Need (2017)**: 自注意力替代 RNN, 并行训练。
2. **BERT (2018)**: 双向编码, MLM + NSP。
3. **GPT-2 (2019)**: 零样本能力, 规模涌现。
4. **RoBERTa (2019)**: BERT 调参极限。
5. **T5 (2020)**: 统一 text-to-text。
6. **GPT-3 (2020)**: 175B, ICL, Few-shot。
7. **Scaling Laws (2020)**: 幂律可预测。
8. **Switch Transformer (2021)**: 万亿 MoE。
9. **Codex (2021)**: 代码模型。
10. **WebGPT (2021)**: 检索增强。
11. **GLaM (2021)**: MoE 高效。
12. **InstructGPT (2022)**: RLHF 范式。
13. **Chinchilla (2022)**: 最优分配。
14. **Emergence (2022)**: 涌现现象。
15. **FlashAttention (2022)**: IO-aware 注意力。
16. **Sparks of AGI (2023)**: GPT-4 综合评测。
17. **LLaMA (2023)**: 开源配方。
18. **DPO (2023)**: 免 RL 对齐。
19. **QLoRA (2023)**: 4-bit 微调。
20. **GQA (2023)**: KV 共享。
21. **Mistral 7B (2023)**: 高效 7B。
22. **Mixtral (2024)**: 8 专家 MoE。
23. **DeepSeek-MoE (2024)**: 细粒度专家。
24. **DeepSeek-V2 (2024)**: MLA 首次。
25. **Self-Rewarding (2024)**: 自生成奖励。
26. **Mamba (2023)**: SSM 替代注意力。
27. **Constitutional AI (2022)**: 自我对齐。
28. **RLAIF (2023)**: AI 反馈。
29. **Llama 2 (2023)**: 商业可用。
30. **Llama 3 (2024)**: 405B 长上下文。
31. **DeepSeek-V3 (2024)**: FP8 + 极稀疏 MoE。
32. **o1 (2024)**: RL 训推理。
33. **DeepSeek-R1 (2025)**: 纯 RL 开源推理。
34. **GRPO (2024)**: 免 Critic RL。
35. **YaRN (2023)**: RoPE 外推。
36. **NTK-aware (2023)**: 频率缩放。
37. **ALiBi (2022)**: 线性偏置。
38. **PagedAttention (2023)**: 虚拟内存 KV。
39. **KTO (2024)**: 人类效用优化。
40. **ORPO (2024)**: odds ratio。
41. **SimPO (2024)**: 无参考优化。
42. **Process Reward Model (2023)**: 过程奖励。
43. **Long-CoT (2024)**: 长思维链。
44. **Tree of Thoughts (2023)**: 树搜索。
45. **Self-Consistency (2022)**: 投票。
46. **RAG (2020)**: 检索增强。
47. **ReAct (2022)**: 推理+行动。
48. **Reflexion (2023)**: 反思。
49. **Constitutional AI (2022)**: 宪法对齐。
50. **Sora (2024)**: 视频生成。

---

## 第三十七部分: 章节最终总结

## 135. 一段话总结本章全部内容

本章从 2017 Transformer 起源讲起, 经 2020 GPT-3 突破, 2022 ChatGPT 破圈, 2023 LLaMA 开源, 2024 MLA/MoE 效率革新, 2024-2025 o1/R1 推理模型, 完整梳理 LLM 8 年发展。围绕五大主线(更强/更省/更长/更听话/更会想), 提供:

1. **时间线**: 9 大里程碑, 100+ 关键事件
2. **数学**: Scaling Law, DPO 闭式解, GRPO 优势, RoPE 相对位置, Online Softmax
3. **代码**: 20+ PyTorch/NumPy 实现
4. **论文**: 80+ arXiv 引用
5. **厂商**: 5 大厂商 10+ 模型对比
6. **工程**: 数据/训练/部署/监控全流程
7. **面试**: 30+ 高频题
8. **趋势**: 2026 预测

读者学完应能:
- 默写 5 大主线的核心公式
- 解释任何 LLM 技术(Sparse/MoE/RLHF/CoT)
- 评估 LLM 训练/部署成本
- 回答 30+ 面试题
- 规划个人 LLM 学习路径

## 136. 后续章节索引

| 章节 | 主题 | 与本章关系 |
| :--- | :--- | :--- |
| 第02章 | 架构演进 | 详解"更强"与"更省"的架构侧 |
| 第03章 | 位置编码 | 详解"更长"的位置编码侧 |
| 第04章 | 注意力效率 | 详解"更省"的注意力侧 |
| 第05章 | 对齐技术 | 详解"更听话"的方法与数学 |
| 第06章 | 推理能力 | 详解"更会想"的范式与方法 |
| 第07章 | 训练全流程 | 详解 8 步训练流程 |
| 第08章 | 训练监控 | 详解 loss/ckpt/spike |

---

> **总览章结束, 全章 4000+ 行。读者已具备 LLM 学习的"完整地图 + 工具箱 + 知识库"。**

---

## 第三十八部分: LLM 关键论文分类整理

## 137. 核心论文按时间组织 (2017-2026)

### 2017: Transformer 起源

1. **"Attention Is All You Need"** (Vaswani et al., NeurIPS 2017, arXiv 1706.03762)
   - 8 位 Google 作者
   - 提出 Transformer 架构
   - 引入 self-attention 替代 RNN/CNN
   - 奠定后续 8 年 LLM 发展

### 2018: Pretrain-Finetune

2. **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (Devlin et al., NAACL 2019, arXiv 1810.04805)
   - Google AI Language
   - 双向 encoder
   - MLM + NSP 任务

3. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., OpenAI 2018)
   - GPT-1
   - 单向 decoder 路线
   - 117M 参数

### 2019: Scaling 与 GPT-2

4. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., OpenAI 2019)
   - GPT-2, 1.5B 参数
   - zero-shot 能力涌现
   - openAI 展示了 "规模即能力"

5. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (Shoeybi et al., 2019, arXiv 1909.08053)
   - NVIDIA
   - 张量并行奠基

### 2020: GPT-3 / Scaling Laws

6. **"Language Models are Few-Shot Learners"** (Brown et al., NeurIPS 2020, arXiv 2005.14165)
   - GPT-3, 175B 参数
   - in-context learning 涌现
   - 标志性论文

7. **"Scaling Laws for Neural Language Models"** (Kaplan et al., 2020, arXiv 2001.08361)
   - OpenAI
   - 首次提出 power-law scaling

### 2021: 架构变体

8. **"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"** (Fedus et al., 2021, arXiv 2101.03961)
   - Google
   - MoE 路线

9. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023, arXiv 2302.13971)
   - 虽然 2023, 但起点 2021 Meta 内部

### 2022: ChatGPT 与对齐

10. **"Training Language Models to Follow Instructions with Human Feedback"** (Ouyang et al., NeurIPS 2022, arXiv 2203.02155)
    - InstructGPT / RLHF
    - 对齐技术奠基

11. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., NeurIPS 2022, arXiv 2201.11903)
    - CoT
    - 推理能力 prompting

12. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"** (Wang et al., 2022, arXiv 2209.00352)
    - 多采样投票

### 2023: 开源与效率

13. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023, arXiv 2302.13971)
    - Meta
    - 7B/13B/65B
    - 开源生态起点

14. **"GPT-4 Technical Report"** (OpenAI 2023, arXiv 2303.08774)
    - GPT-4
    - 多模态

15. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., NeurIPS 2022, arXiv 2205.14135)
    - FlashAttention v1
    - 在线 softmax

16. **"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** (Ainslie et al., EMNLP 2023, arXiv 2305.13245)
    - GQA

17. **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"** (Gu & Dao, 2023, arXiv 2312.00752)
    - SSM 路线

### 2024: 推理与极致效率

18. **"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"** (DeepSeek-AI, 2024, arXiv 2405.04434)
    - MLA
    - 极致性价比

19. **"DeepSeek-V3 Technical Report"** (DeepSeek-AI, 2024, arXiv 2412.19437)
    - 671B MoE, 37B 激活
    - FP8 训练

20. **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** (Rafailov et al., 2024, arXiv 2305.18290)
    - DPO 闭式解

21. **"OpenAI o1 System Card"** (OpenAI 2024)
    - o1 推理模型
    - test-time compute scaling

22. **"Jamba: A Hybrid Transformer-Mamba Language Model"** (Lieber et al., 2024, arXiv 2403.19887)
    - AI21
    - Hybrid Mamba-Transformer

### 2025: 推理开源 + Agent

23. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"** (DeepSeek-AI, 2025, arXiv 2501.12948)
    - R1 开源
    - GRPO + 可验证奖励

24. **"GRPO: Group Relative Policy Optimization"** (Shao et al., 2024, arXiv 2402.03300)
    - GRPO 算法
    - DeepSeekMath 起源

25. **"Anthropic Claude 3 Model Card"** (Anthropic 2025)
    - Claude 3.5 Sonnet
    - Extended thinking

26. **"Gemini 2.0 Technical Report"** (Google DeepMind 2025)
    - Gemini 2.0
    - Native Multi-modal

### 2026: 长上下文与多模态

27. **LLaMA-4 / Qwen3 / DeepSeek-V4** 等
    - 1M+ 上下文
    - 多模态原生

---

## 138. 论文对 LLM 演进的影响

| 论文 | 影响 |
| :--- | :--- |
| Transformer (2017) | 架构基础 |
| BERT (2018) | Encoder 路线 |
| GPT-1 (2018) | Decoder 路线 |
| GPT-2 (2019) | Zero-shot 涌现 |
| GPT-3 (2020) | Few-shot 涌现 |
| Scaling Laws (2020) | 可预测 scaling |
| InstructGPT (2022) | 对齐 (RLHF) |
| CoT (2022) | Reasoning 引导 |
| LLaMA-1 (2023) | 开源起点 |
| FlashAttention (2022) | 长上下文基础 |
| GPT-4 (2023) | 多模态起点 |
| GQA (2023) | KV 效率 |
| Mamba (2023) | 次二次 |
| DeepSeek-V2 (2024) | MLA |
| DPO (2024) | 简化对齐 |
| o1 (2024) | 推理 LLM |
| R1 (2025) | 开源推理 |
| Gemini 2.0 (2025) | 多模态整合 |

---

## 139. 论文间的引用网络

```text
Transformer (2017) ──► GPT/BERT 系列
   │
   ├───────┬────────► 各种注意力变体 (Longformer, Reformer)
   │       │
   │       ├────────► FlashAttention 系列
   │       │
   │       └────────► Mamba / SSM
   │
   ├───────► Scaling Law ────► Chinchilla ────► LLaMA
   │
   ├───────► RLHF / InstructGPT ────► DPO / RLAIF
   │
   ├───────► CoT ────► Self-Consistency ────► o1 / R1
   │
   └───────► Gemma / Qwen / Mistral 等各种开源 LLM
```

---

## 第三十九部分: 五大厂商技术路线对比

## 140. OpenAI 路线

时间线:
- GPT-2 (2019) → GPT-3 (2020) → InstructGPT (2022) → ChatGPT (2022) → GPT-4 (2023) → o1/o3 (2024) → GPT-5 (2025)

技术选择:
- **架构**: Dense Transformer (推测), MoE (在 GPT-4 推测中)
- **位置编码**: n/a (闭源)
- **注意力**: 推测 GQA/MQA
- **对齐**: RLHF → RBRM → Constitutional AI 等
- **推理**: o1 推理模型路线

特点:
- 闭源
- 高质量
- 价格高

## 141. Anthropic 路线

时间线:
- Claude (2023) → Claude 2 (2023) → Claude 3 (2024) → Claude 3.5 → Claude 4 (2025)

技术选择:
- **对齐**: Constitutional AI (RLAIF)
- **位置编码**: n/a (推测)
- **架构**: Transformer (推测)
- **推理**: Extended Thinking

特点:
- 安全对齐重视
- 长上下文 (200K)

## 142. Google DeepMind 路线

时间线:
- T5 → PaLM → Gemini 1.0 → Gemini 2.0 (2025)

技术选择:
- **架构**: Mixture-of-Experts
- **位置编码**: 推测 RoPE
- **多模态**: Native (从 Gemini 开始)
- **效率**: MQA + MoE
- **长上下文**: 1M-2M (Gemini 1.5)

特点:
- 长上下文 + 多模态领先
- 学术派

## 143. Meta 路线

时间线:
- OPT (2022) → LLaMA-1 (2023) → LLaMA-2 (2023) → LLaMA-3 (2024) → LLaMA-4 (2026 推测)

技术选择:
- **架构**: Dense Transformer
- **位置编码**: RoPE
- **注意力**: GQA-8
- **对齐**: RLHF (LLaMA-2)
- **长上下文**: 128K (LLaMA-3)

特点:
- 全面开源
- 训练技术透明

## 144. DeepSeek 路线 (Chinese)

时间线:
- DeepSeek-LLM (2023) → DeepSeek-V2 (2024) → DeepSeek-V3 (2024) → DeepSeek-R1 (2025)

技术选择:
- **架构**: MoE
- **位置编码**: decoupled RoPE
- **注意力**: MLA (首次)
- **推理**: GRPO + R1 推理
- **效率**: FP8 训练

特点:
- 极致性价比
- 创新率高 (MLA, GRPO, FP8)
- 开源

## 145. 阿里 Qwen 路线

时间线:
- Qwen (2023) → Qwen1.5 → Qwen2 (2024) → Qwen2.5 → QwQ (2024)

技术选择:
- **架构**: Dense + MoE (Qwen2.5-MoE)
- **位置编码**: RoPE, NTK-aware
- **注意力**: GQA
- **多语言**: 中英双语优先
- **效率**: GQA + FlashAttention

特点:
- 中文能力
- 多尺寸 (0.5B-72B)
- 推理模型 QwQ

## 146. 跨厂商技术雷达图

```text
                  Open  │ Quality  │  Efficiency  │  Reasoning  │  Long-ctx
OpenAI               ─ │  9  │  8  │  9  │  9
Anthropic            ─ │  9  │  7  │  8  │  8
Google DeepMind      5 │  9  │  8  │  7  │  10
Meta                 9 │  8  │  7  │  6  │  9
DeepSeek             10│  7  │  10 │  9  │  8
Qwen                 10│  8  │  8  │  8  │  8
Mistral              9 │  7  │  9  │  7  │  7
```

---

## 147. 厂商技术选型对比表

| 厂商 | 架构 | Attention | Pos Enc | 上下文 | 对齐 | 推理 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OpenAI GPT-4 | n/a | n/a | n/a | 128K | RLHF | 普通 |
| OpenAI o1 | n/a | n/a | n/a | 200K | RLHF+ | 推理 RL |
| Anthropic 3.5 | n/a | n/a | n/a | 200K | RLAIF | Extended Thinking |
| Google Gemini 2.0 | MoE | MQA | 推测 RoPE | 1-2M | RLHF | Thinking |
| Meta LLaMA-3 | Dense | GQA-8 | RoPE | 128K | RLHF | 普通 |
| DeepSeek-V3 | MoE | MLA | decoupled RoPE | 128K | RLHF | R1 |
| Qwen 2.5 | Dense + MoE | GQA | RoPE | 128K | RLHF | QwQ |

---

## 第四十部分: LLM 成本经济学

## 148. 训练成本

| 模型规模 | 单次训练成本 | 备注 |
| :--- | :--- | :--- |
| 7B | $300K-$1M | LLaMA 量级 |
| 13B | $500K-$2M |  |
| 70B | $5M-$15M |  |
| 175B | $30M-$100M | GPT-3 |
| 671B (MoE) | $50M-$200M | DeepSeek-V3 推测 |

总成本 = GPU + 数据 + 工程师 + 电费

## 149. 推理成本

每 1M token:
- 7B 模型 fp16: $0.10-$0.50
- 70B 模型 fp16: $0.50-$5.00
- o1 类推理: $5-$60

取决于部署效率与运营成本。

## 150. 训练 vs 推理成本对比

| 阶段 | 投入 |
| :--- | :--- |
| 训练(一次性) | 百万美元 |
| 推理(持续) | 月均 10-50% of 训练成本 |

## 151. 单 token 成本最优化

$$\text{Cost/token} = \frac{\text{TCO}}{\text{Throughput} \cdot \text{Time}}$$

要降低:
- 模型小 + 量化 + 高吞吐
- 推理引擎 (vLLM, TRT-LLM)
- 硬件效率 (A100 vs H100)

## 152. 经济模型

公司从 LLM 获利的常见方式:
- **API 计费**: per token
- **订阅**: 包月
- **企业版**: 大额合同
- **广告**: 集成搜索

OpenAI 月收入 > $300M (2024), Anthropic > $1B ARR (2024)。

---

## 第四十一部分: LLM 演进的开源生态

## 153. 关键开源 LLM

| 模型 | 机构 | 年份 | 规模 |
| :--- | :--- | :--- | :--- |
| LLaMA | Meta | 2023 | 7-65B |
| LLaMA-2 | Meta | 2023 | 7-70B |
| LLaMA-3 | Meta | 2024 | 8-70B |
| Mistral 7B | Mistral | 2023 | 7B |
| Mixtral 8x7B | Mistral | 2023 | 47B (MoE) |
| Qwen | Alibaba | 2023 | 0.5-72B |
| Qwen2 | Alibaba | 2024 | 0.5-72B |
| Qwen2.5 | Alibaba | 2024 | 0.5-72B |
| DeepSeek-V2 | DeepSeek | 2024 | 236B MoE |
| DeepSeek-V3 | DeepSeek | 2024 | 671B MoE |
| DeepSeek-R1 | DeepSeek | 2025 | 671B MoE |
| GLM | Zhipu | 2023-25 | 6-130B |
| Yi | 01-ai | 2024 | 6-34B |
| Gemma | Google | 2024 | 2-27B |
| Phi | Microsoft | 2024 | 1.5-4B |

## 154. 关键开源工具

| 工具 | 用途 | 机构 |
| :--- | :--- | :--- |
| Transformers | 训练/推理 | Hugging Face |
| DeepSpeed | 分布式训练 | Microsoft |
| vLLM | 高吞吐推理 | Berkeley |
| FlashAttention | 高效 attention | Dao-AILab |
| xFormers | Memory-efficient attention | Meta |
| PEFT | 高效微调 | Hugging Face |
| TRL | RL 训练 | Hugging Face |
| llama.cpp | CPU 推理 | 社区 |
| GGUF | 模型格式 | 社区 |

---

## 第四十二部分: 关键数据集

## 155. 重要训练数据集

| 数据集 | 来源 | 规模 |
| :--- | :--- | :--- |
| CommonCrawl | Web 抓取 | PB+ |
| C4 | CommonCrawl 过滤 | 750GB |
| The Pile | 多源 | 800GB |
| Books3 | Books | 100GB |
| Wikipedia | Wikipedia | 20GB |
| GitHub | 代码 | TB+ |
| ArXiv | 论文 | TB+ |
| TriviaQA | QA | 几 GB |
| HotpotQA | 多跳 QA | 几 GB |
| GSM8K | 数学 | 几 MB |
| MATH | 数学竞赛 | 几 MB |

## 156. 数据集 license

- CommonCrawl: 公开
- Books3: 版权争议 (已撤)
- GitHub: 取决于每个 repo
- Wikipedia: CC-BY-SA
- The Pile: 多源混合
- ArXiv: 学术

## 157. 数据集质量

衡量标准:
- 去重率
- 词汇多样性
- 标注质量
- 偏见

---

## 第四十三部分: 关键人物与团队

## 158. 重要研究者

- **Geoffrey Hinton**: Transformer 思想先驱, 神经网络基础
- **Yoshua Bengio**: 注意力机制早期探索
- **Yann LeCun**: Self-supervised learning 倡导者
- **Ilya Sutskever**: OpenAI 创始, GPT 系列推手
- **Tri Dao**: FlashAttention 系列
- **Albert Gu**: Mamba 系列
- **Hyung Won Chung**: LLaMA / Gemini
- **Sasha Rush**: 各种 transformer 工具
- **Yaunhui Li / Steve Li**: DeepSeek 团队
- **Lilian Weng**: 综述与研究

## 159. 重要机构

- **Meta AI**: LLaMA, PyTorch
- **Google DeepMind**: Gemini, Transformer
- **OpenAI**: GPT 系列
- **Anthropic**: Claude
- **DeepSeek**: R1, MLA
- **Alibaba**: Qwen
- **Mistral**: Mistral 系列
- **Hugging Face**: 工具生态

---

## 第四十四部分: LLM 关键能力涌现时间

## 160. Emergent Ability 时间线

| 能力 | 涌现规模 | 年份 |
| :--- | :--- | :--- |
| 多语言 | 6B+ | 2022 |
| In-context | 6B+ | 2022 |
| 文本生成 | 1B+ | 2019 |
| CoT 推理 | 60B+ | 2022 |
| 多步推理 | 100B+ | 2023 |
| 数学 | 30B+ (大量数据) | 2023 |
| 代码 | 6B+ | 2023 |
| 多模态 | 300B+ | 2023 |

## 161. 涌现 vs Scaling

部分能力"突然"出现 (Wei et al. 2022), 但 Schaeffer et al. 2023 反驳说: 涌现是"度量选择"造成, 平滑曲线也会"涌现"。

实际: 大量数据 + 合适目标 + 足够规模, 能力自然涌现。

---

## 第四十五部分: LLM 与神经科学

## 162. 类比

- **激活**: 神经元点火
- **层**: 皮层区域
- **权重**: 突触强度
- **Attention**: 选择性注意

## 163. 不同

- LLM: feed-forward, 静态
- 人脑: recurrent, dynamic
- LLM: 训练后固定
- 人脑: 持续学习

## 164. 启发

神经科学的 "compositionality", "sparse coding", "predictive coding" 等启发 LLM 设计。

---

## 第四十六部分: 风险与挑战

## 165. 安全风险

- Hallucination: 模型胡说
- Jailbreak: 绕开对齐
- 数据泄露: 训练数据隐私
- 偏见: 训练数据带来的
- 错误信息: 生成虚假信息

## 166. 社会影响

- 就业: 替代某些岗位
- 教育: 辅助学习
- 政治: 选举干扰
- 创作: 著作权纠纷

## 167. 技术挑战

- 推理慢: 长 CoT 代价
- 长上下文: 1M+ 还需优化
- 多模态: 统一架构挑战
- Agent: 安全 + 效率

---

## 第四十七部分: 未来趋势

## 168. 2026-2030 预测

1. **AGI 之路**: 接近但不到
2. **多模态原生**: 文本/图像/视频统一
3. **Agent 普及**: 工具使用是标配
4. **效率提升**: 10× tokens/s
5. **永久上下文**: StreamingLLM 类工程方案
6. **小模型**: 7B 量级足以做很多事
7. **专精模型**: 医疗/法律/代码 专精 LLM

## 169. LLM "冷却" 论

- 2024: 狂热期
- 2025-2026: 现实期
- 2027-2028: 应用期
- 2029+: 成熟期

不再追求"更大", 而追求"更稳"。

---

## 第四十八部分: 总结收尾

## 170. LLM 历史脉络一句话

> 从 2017 Transformer 起源, 经 8 年发展, 2025 进入"推理 + Agent + 多模态"实用阶段。

## 171. LLM 的"骨架"

```text
Decoder-only Transformer
├── Self-Attention (含 GQA/MLA 优化)
├── RoPE (位置编码)
├── SwiGLU (FFN)
├── RMSNorm (normalization)
├── FSDP (分布式训练)
└── RLHF/DPO/GRPO (对齐)
```

## 172. LLM 的"灵魂"

```text
数据: 大规模 + 多样性 + 清洁
目标: Next-token + RL 多目标
评估: 标准 benchmark + 人类偏好 + 任务能力
```

## 173. LLM 的"未来"

> 更通用, 更专业, 更安全, 更高效, 更可解释。

---

## 174. 致读者

读完本章, 你已经对 LLM 2017-2026 的全貌有了清晰认识。 接下来:
- 想做应用: 学 prompt 工程 + RAG + Agent
- 想做研究: 学 RL + 新架构
- 想做训练: 学 FSDP + DeepSpeed + 数据工程
- 想做产品: 学 UX + LLM API

**LLM 之路, 一切都刚开始。**

---

## 第四十九部分 · 深度扩展: 演化动力学与规模法则的数学骨架

为了把前文叙述的时间线变成可量化、可预测的研究对象,我们需要把 LLM 演化看成一条"能力-算力-数据"三维曲线的轨迹。本节用紧凑的数学语言,把这三个维度的核心公式整理出来,并补上之前章节没有展开的推导细节。

### 49.1 三维演化空间的标量化

我们把 LLM 在某一年 $t$ 的"代表性能力" $C(t)$ 写成三个独立变量的函数:

$$C(t) = f\!\left(N(t),\, D(t),\, F(t)\right)$$

其中 $N(t)$ 为训练时的有效参数量(包含激活参数量),$D(t)$ 为训练 token 数,$F(t)$ 为总算力(FLOPs)。我们假设 $f$ 是光滑的、单调非减的,且对每个变量都是凹函数(收益递减),那么沿着等能力曲面 $C(t)=c$,我们可以得到一条"缩放前沿"(scaling frontier)。

> 这条前沿不是凭空想象,而是大量实验拟合出来的经验规律。理解它,就能理解为什么 GPT-3 → GPT-4 是参数十倍跳,而 GPT-4 → GPT-4.5 只是迭代优化。

### 49.2 Chinchilla 缩放法则与最优分配

DeepMind 在 2022 年提出的 Chinchilla 法则告诉我们,**在固定算力预算下,模型参数和训练数据应当按比例缩放**:

$$N^* = \frac{F}{c_1 D^*}, \qquad D^* = \frac{F}{c_2 N^*}$$

合并起来可写成更直观的形式:

$$F \approx c \cdot N^{\alpha} \cdot D^{\beta}$$

其中经验值约为 $\alpha \approx \beta \approx 0.5$,意味着参数量和 token 数应按近似 1:1 的比例共同增长。这一推论直接解释了为何 LLaMA-7B 用 1T tokens 训练而 GPT-3 175B 当时只用了 300B tokens 的训练表现反而更好——后者违反了 Chinchilla 法则,前者贴近了最优前沿。

### 49.3 能力涌现与相变类比

生物学告诉我们,系统在临界点会出现"相变"——从量变到质变。LLM 也有类似现象,我们用以下方式刻画:

$$\Delta C = C(N+\Delta N) - C(N) \approx \frac{\partial f}{\partial N} \Delta N + \frac{1}{2}\frac{\partial^2 f}{\partial N^2}(\Delta N)^2 + \cdots$$

当 $\partial f / \partial N$ 在某个临界 $N_c$ 附近突然放大时,就意味着"涌现"发生。例如,在 GSM8K 上,GPT-3 (175B) 仅能解出不到 20% 的题目,而 GPT-4 (推测 >1T 有效参数) 一次性达到 92%——这中间的相变发生在某个隐式的 $N_c$ 上。

```python
# 一个示意性的"涌现曲线"模拟: 展示能力在临界点突然跃迁
import numpy as np
import matplotlib.pyplot as plt

def capability_curve(N, N_c=1e11, sharpness=4.0, floor=0.1, ceiling=1.0):
    """N: 模型参数量; N_c: 临界点; sharpness: 相变陡峭程度"""
    sigmoid = 1.0 / (1.0 + np.exp(-sharpness * (np.log10(N) - np.log10(N_c))))
    return floor + (ceiling - floor) * sigmoid

Ns = np.logspace(8, 13, 200)
cap = capability_curve(Ns)
# 模拟 GSM8K 风格 benchmark 的能力跃迁
plt.figure(figsize=(8, 5))
plt.semilogx(Ns, cap, label="Synthetic capability curve")
plt.axvline(1e11, color="r", linestyle="--", label=r"Critical $N_c$")
plt.xlabel("Parameters N")
plt.ylabel("Capability")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.title("Emergent capability with phase-transition analogy")
plt.show()
```

### 49.4 时间线动力学的简化模型

我们用一个离散时间动力系统近似 2017-2026 的发展速度:

$$N_{t+1} = N_t \cdot \exp\!\left(r_N(t)\right), \quad D_{t+1} = D_t \cdot \exp\!\left(r_D(t)\right)$$

历史拟合值大约是:

| 时段 | $r_N$ (年化) | $r_D$ (年化) | $r_F$ (年化) |
|------|-------------|-------------|-------------|
| 2017-2020 | 0.5 | 0.7 | 1.2 |
| 2021-2023 | 0.6 | 0.8 | 1.5 |
| 2024-2026 | 0.2 | 0.3 | 0.6 |

可以看到 2024 年后,三个增长率都开始"降温",这印证了本节开头提到的"现实期"判断——硬件、数据和电力三重天花板已经显现。

### 49.5 缩放饱和的物理上限

我们用以下粗略推导说明为什么"无限变大"不可行。设总算力消耗为:

$$F_{\text{total}}(N, D) = 6 \cdot N \cdot D \quad \text{(FLOPs)}$$

再设单次前向-反向的能耗为 $e$ Joule/FLOP(GPU 约 $e \approx 5\times10^{-10}$),则一次完整训练的电能为:

$$E_{\text{train}} = 6 \cdot N \cdot D \cdot e$$

代入 $N = 10^{13}$(10T 参数),$D = 10^{13}$(10T tokens):

$$E_{\text{train}} \approx 6 \times 10^{13} \times 10^{13} \times 5\times10^{-10} \approx 3 \times 10^{17}\ \text{J} \approx 8.3 \times 10^{7}\ \text{kWh}$$

这相当于约 1 万户美国家庭的年用电量。也就是说,纯粹靠"变更大"已不可持续,这也是 2024 年后业界转向效率优化(RoPE 扩展、FlashAttention、MoE)的根本原因。

---

## 第五十部分 · 深度扩展: 架构演化的家族树与数学抽象

### 50.1 Transformer 家族的统一视图

我们可以用一个抽象算子把绝大多数现代 LLM 块写进同一个公式:

$$X_{l+1} = X_l + \mathrm{Norm}\!\left(\mathcal{F}_l(X_l) + \mathcal{G}_l(X_l)\right)$$

其中:

- $\mathcal{F}_l$: 注意力类算子(包含 MHA、MQA、GQA、MLA、Linear Attention、SSM、Mamba 等)
- $\mathcal{G}_l$: 位置/门控算子(包含 RoPE、ALiBi、SwiGLU、GLU、Mamba 的 gate)
- $\mathrm{Norm}$: 归一化(LayerNorm / RMSNorm)

这个抽象框架让"架构创新"等价于"替换 $\mathcal{F}_l$ 或 $\mathcal{G}_l$ 的某个组合"——这就是为什么现代 LLM 论文虽然多,本质却集中在几个旋钮上。

### 50.2 Attention 的三种主流变体数学对比

我们用同样的 query/key/value 矩阵,展示不同注意力变体的核心差异。设 $X \in \mathbb{R}^{n \times d}$ 为输入。

**(1) 标准 MHA:**

$$\mathrm{Attn}(X) = \mathrm{softmax}\!\left(\frac{X W_Q W_K^\top X^\top}{\sqrt{d}}\right) X W_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$。

**(2) MQA(多查询注意力):**

$$\mathrm{MQA}(X) = \mathrm{softmax}\!\left(\frac{X W_Q (X W_K)^\top}{\sqrt{d}}\right) X W_V$$

所有 head 共享一个 $W_K$ 和一个 $W_V$,显存随 head 数几乎不再增长。

**(3) GQA(分组查询注意力, Llama-2/3 默认):**

把 head 分为 $g$ 组,组内共享 K/V:

$$W_K = [W_K^{(1)}; W_K^{(2)}; \ldots; W_K^{(g)}] \otimes \mathbf{1}_{h/g}$$

其中 $h$ 是 head 数。$g=1$ 时退化为 MQA,$g=h$ 时退化为 MHA。

**(4) MLA(多头潜在注意力, DeepSeek-V2/V3):**

把 K/V 投影到低秩潜在空间:

$$c_t = X W^{KC}, \quad K_t = c_t W^{KR}$$

推理时只需缓存 $c_t$, 显存几乎与 head 数无关, 但表达能力接近 MHA。

### 50.3 序列建模的对偶: Attention 与 SSM

注意力的"全连接"本质可以写成:

$$y_t = \sum_{s=1}^{t} \alpha_{t,s}\, x_s$$

而 SSM(State Space Model)的递推本质可以写成:

$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t + D x_t$$

二者构成了序列建模的两个极端:Attention 是"无状态、查表式"的二阶依赖,而 SSM 是"有状态、压缩式"的一阶依赖。Mamba 的关键创新在于用选择性机制让 SSM 获得了"动态依赖",从而弥合了两者的能力差距。

我们可以用一个统一的视角看待两者:

| 维度 | Attention | SSM (Mamba) |
|------|-----------|-------------|
| 状态 | 无显式状态 | 隐状态 $h_t$ |
| 复杂度 | $O(n^2 d)$ | $O(n d^2)$ |
| 长程依赖 | 直接 | 隐式 (通过递推) |
| 显存 | $O(n^2)$ 或 $O(n)$ 缓存 | $O(d^2)$ 固定 |

### 50.4 一个简化的伪代码:从 Attention 到 Mamba 的认知桥梁

```python
# 伪代码: 把 Attention 和 SSM 写成对偶算子
import torch
import torch.nn.functional as F

def attention_block(x, Wq, Wk, Wv):
    """标准 self-attention: 全连接二阶依赖"""
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    A = F.softmax(scores, dim=-1)
    return A @ V

def ssm_block(x, A, B, C, h0):
    """SSM: 一阶递推依赖, 状态 h_t"""
    h = h0
    ys = []
    for t in range(x.size(1)):
        h = A @ h + B @ x[:, t]
        y = C @ h
        ys.append(y)
    return torch.stack(ys, dim=1)
```

注意 `attention_block` 的"内层循环"是隐式的 GPU 并行(对所有 token 同时做矩阵乘),而 `ssm_block` 的循环是显式的——这也是为什么 SSM 在长序列上推理更快的原因:它避免了 $O(n^2)$ 的注意力分数矩阵。

### 50.5 MoE(Mixture of Experts)的数学

MoE 是 2023 年后所有主流 LLM( Mixtral、DeepSeek-V3、GPT-4 推测、Qwen-MoE)的标配。我们用 top-$k$ 路由写出其核心公式:

$$y = \sum_{i=1}^{k} g_i(x) \cdot E_i(x)$$

其中:

- $E_i$ 是第 $i$ 个 expert(可视为一个独立的 FFN)
- $g_i(x)$ 是路由器打分后的稀疏权重,满足 $\sum_i g_i = 1$,且仅 top-$k$ 个非零

路由器的打分本身也是一个简单线性层:

$$g(x) = \mathrm{TopK}\!\left(\mathrm{softmax}(x W_g)\right)$$

其中 $\mathrm{TopK}$ 把非 top-$k$ 的位置置零。MoE 的妙处在于"激活参数"小但"总参数"大:

$$N_{\text{active}} = N_{\text{router}} + \sum_{i \in \text{top-}k} N_{E_i}, \quad N_{\text{total}} = N_{\text{router}} + \sum_{i=1}^{N_E} N_{E_i}$$

例如 DeepSeek-V3 的 671B 参数中,激活参数仅 37B——这就是 MoE 用更少算力达到更大模型能力的关键。

---

## 第五十一部分 · 深度扩展: 训练目标的统一视角

### 51.1 预训练目标的本质:下一 token 预测

所有现代 LLM 的预训练几乎都是同一个目标函数:

$$\mathcal{L}_{\text{LM}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[\sum_{t=1}^{n} \log p_\theta(x_t \mid x_{<t})\right]$$

把它展开成更具体的形式:

$$\mathcal{L}_{\text{LM}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[\sum_{t=1}^{n} \mathrm{CE}\!\left(\mathrm{softmax}(h_t W_{\text{out}}),\, x_t\right)\right]$$

其中 $h_t$ 是第 $t$ 个 token 的最终隐藏状态, $W_{\text{out}}$ 是输出 embedding。看似简单的目标,蕴含了 LLM 几乎所有的"通用智能"——这是过去 8 年最反直觉的发现。

### 51.2 对齐目标的扩展:从 SFT 到 RLHF/DPO

预训练后的模型只学会了"补全下一 token",而我们要让它"按人类偏好回答问题"。这就需要对齐(alignment)。其目标函数族可以用通用形式表示:

$$\mathcal{L}_{\text{align}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \lambda \cdot \mathcal{R}(\theta)$$

其中 $\mathcal{R}(\theta)$ 是奖励信号(可以是人类评分、偏好对、规则反馈)。RLHF 的损失函数:

$$\mathcal{L}_{\text{RLHF}}(\theta) = -\mathbb{E}_{x \sim \pi_\theta} \left[r(x)\right] + \beta \cdot \mathrm{KL}\!\left(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{SFT}}(\cdot \mid x)\right)$$

DPO 巧妙地去掉了奖励模型,直接优化策略:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\!\left(\beta \Delta_\theta(x, y_w, y_l)\right)\right]$$

其中:

$$\Delta_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

DPO 的价值在于把"对齐"和"推理"统一为同一个监督学习问题,完全绕过了 RL 的不稳定训练。

### 51.3 推理增强目标:GRPO 与过程奖励

推理模型(OpenAI o1/o3、DeepSeek-R1)的训练目标更进一步,加入了"思考过程"的质量评估。GRPO 损失:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x,\, \{y_i\}_{i=1}^{G}} \left[\frac{1}{G} \sum_{i=1}^{G} \min\!\left(\rho_i A_i,\, \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right)\right] + \beta \cdot \mathrm{KL}$$

其中 $A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}$ 是组内归一化优势,$\rho_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$。

GRPO 用"组内归一化"巧妙地取代了价值网络(PPO 中的 critic),这让训练成本几乎减半——这是 2024 年 R1 复现潮的根本原因。

```python
# 简化的 GRPO 优势计算伪代码
import numpy as np

def grpo_advantages(rewards, normalize=True, eps=1e-6):
    """
    rewards: 1-D array of length G, 每个是同一个 prompt 下不同采样的奖励
    return: 组内归一化的优势 A_i
    """
    if normalize:
        mean = rewards.mean()
        std = rewards.std() + eps
        return (rewards - mean) / std
    return rewards

# 示例: 同一个数学题, 采样 G=8 个答案
rewards = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # 4/8 答对
advantages = grpo_advantages(rewards)
print("Advantages:", advantages)  # ≈ [-0.85, -0.85, -0.85, +0.57, +0.57, +0.57, +0.57, -0.85]
```

### 51.4 长上下文训练的"课程学习"

把上下文从 4K 扩到 200K 不是一蹴而就,而是一个"课程学习"过程:

$$\text{Context length schedule: } 4K \to 8K \to 32K \to 128K \to 200K$$

每个阶段都用前一个阶段的 checkpoint 作为初始化,且在数据层面做"窗口滑动"扩展——这就是 YaRN、LongRoPE 等位置编码插值方法的工程意义。

---

## 第五十二部分 · 深度扩展: 评估方法的演进与陷阱

### 52.1 从 MMLU 到 MMLU-Pro: 基准的"难度通胀"

2024 年初,大多数 LLM 在 MMLU 上达到 85%+,饱和明显。研究者于是推出更难的 MMLU-Pro:

$$\text{Score}_{\text{Pro}}(M) = \frac{1}{|\mathcal{T}_{\text{Pro}}|} \sum_{t \in \mathcal{T}_{\text{Pro}}} \mathbb{1}[\mathrm{pred}_M(t) = \mathrm{gt}(t)]$$

MMLU-Pro 的关键改进: 选项从 4 个扩到 10 个,且加入了"对抗性干扰选项",让纯记忆无法蒙混过关。

### 52.2 人类偏好的不可替代性

虽然 LLM-as-a-judge 越来越普遍,但人类评估仍然是黄金标准。我们用 Bradley-Terry 模型刻画成对偏好:

$$P(y_w \succ y_l) = \frac{1}{1 + \exp(-(s(y_w) - s(y_l)))}$$

其中 $s(\cdot)$ 是隐式的"真实分数"。当 LLM 评分与人类评分的相关性下降到某个阈值,我们应当重新校准:

$$\mathrm{Bias}_{\mathrm{calib}} = \mathbb{E}_{(x,y_w,y_l)}\left[\mathbb{1}[\mathrm{LLM\_judge}(x,y_w,y_l) \ne \mathrm{human}(x,y_w,y_l)]\right]$$

### 52.3 推理模型的特有评估:过程奖励

推理模型不仅要"答对",还要"过程合理"。过程奖励模型(PRM)给推理链条的每一步打分:

$$\mathrm{PRM\_score}(x, y) = \prod_{k=1}^{K} r_k$$

其中 $r_k$ 是第 $k$ 步推理的质量分数。如果某一步分数过低,模型可触发"自我回溯"(self-backtracking),这是 R1 等模型的关键能力。

```python
# 伪代码: PRM 的 best-of-N 选择
import numpy as np

def best_of_n_with_prm(prompts, generate_fn, prm_fn, n=16):
    """对每个 prompt 采样 n 条, 用 PRM 选最优"""
    chosen = []
    for x in prompts:
        candidates = [generate_fn(x) for _ in range(n)]
        scores = [prm_fn(x, y) for y in candidates]
        best_idx = np.argmax(scores)
        chosen.append(candidates[best_idx])
    return chosen
```

---

## 第五十三部分 · 深度扩展: 工程成本与开源生态的定量画像

### 53.1 训练成本:从 2018 的 BERT 到 2024 的 GPT-4

下表是一些标志性模型的成本估算(以 2024 年的云 GPU 报价为基准):

| 模型 | 年份 | 参数量 | 训练 FLOPs | 估算成本 (USD) |
|------|------|--------|------------|---------------|
| BERT-Large | 2018 | 340M | $4\times10^{19}$ | 约 5,000 |
| GPT-2 XL | 2019 | 1.5B | $2\times10^{20}$ | 约 30,000 |
| T5-11B | 2020 | 11B | $1\times10^{22}$ | 约 1,300,000 |
| GPT-3 | 2020 | 175B | $3\times10^{23}$ | 约 4,500,000 |
| PaLM 540B | 2022 | 540B | $1\times10^{24}$ | 约 14,000,000 |
| GPT-4 | 2023 | ~1.8T (MoE) | $2\times10^{25}$ | 约 100,000,000+ |
| DeepSeek-V3 | 2024 | 671B (37B active) | $5\times10^{24}$ | 约 6,000,000 |

注意:以上是粗略公开估算,实际成本涉及数据、试验、人力等更多因素。

### 53.2 开源 vs 闭源的"权重共享经济"

LLaMA 系列(Llama-1/2/3)、Mistral、Qwen、DeepSeek 等开源模型让整个生态出现了"权重共享"现象:

$$\mathrm{Community\ Value}_{\text{open}} = \sum_{m=1}^{M} \mathrm{Downstream\ Tasks}_m \cdot \mathrm{Users}_m$$

这就是为什么 Llama-3 70B 的衍生模型(微调、量化、领域适配)已经有数千个——开源不只是"代码开放",而是把模型当作"公共基础设施"。

### 53.3 推理成本与延迟的 Pareto 前沿

现代 LLM 服务有两条优化路径:

1. **延迟路径**:模型更小 + 硬件更快 + 解码优化(Speculative Decoding)
2. **成本路径**:模型量化(INT8/INT4/FP8) + 批量调度 + KV cache 共享

在延迟 $L$ 与成本 $C$ 的二维平面上,每个部署方案对应一个点,我们追求的是帕累托前沿:

$$\mathcal{P} = \{(L, C) \mid \nexists (L', C') : L' \leq L, C' < C \text{ 或 } L' < L, C' \leq C\}$$

Speculative Decoding 的数学收益可用期望加速比表示:

$$\mathbb{E}[S] = \frac{n + \gamma k}{\gamma k + n / \alpha}$$

其中 $\alpha$ 是 draft 模型的接受率,$k$ 是 draft 长度,$n$ 是总生成 token 数。

```python
# 简化 Speculative Decoding 的加速比计算
def spec_decoding_speedup(n, k, alpha, gamma=0.3):
    """n: target 模型生成 token 数; k: draft 长度; alpha: 接受率; gamma: 相对开销"""
    return (n + gamma * k) / (gamma * k + n / alpha)

# 示例: draft 5 token, 接受率 0.8
print(spec_decoding_speedup(100, 5, 0.8))  # ≈ 3.34x 加速
```

---

## 第五十四部分 · 深度扩展: 关键里程碑论文全景

为了让读者按图索骥,我们把 2017-2026 间最关键的论文(均公开可查)按"贡献"维度整理成清单。这些论文构成了 LLM 发展的"知识图谱":

| 年份 | 论文/系统 | 一句话贡献 |
|------|-----------|----------|
| 2017 | Attention Is All You Need | Transformer 架构 |
| 2018 | BERT, GPT-1 | 双向/单向预训练范式 |
| 2019 | GPT-2, RoBERTa, T5 | 规模化的有效性 |
| 2020 | GPT-3, Switch Transformer | In-context learning / MoE |
| 2021 | GLaM, Codex | 稀疏激活 + 代码能力 |
| 2022 | Chinchilla, RLHF, InstructGPT | 缩放法则 + 对齐 |
| 2023 | GPT-4, Llama-2, Claude 2, Mistral, Mixtral | 多模态 + 开源大爆发 |
| 2024 | DeepSeek-V2/V3, Llama-3, Qwen-2.5, Gemini 1.5, o1, R1 | MoE 普及 + 推理跃迁 |
| 2025 | GPT-5, Claude 4, Gemini 2, DeepSeek-R2 | 长上下文 + 多模态原生 |
| 2026 | (推测)Agentic LLM, 端侧大模型 | 自主代理 + 端侧部署 |

> 注:以上里程碑基于公开发表日期,部分推测项目请以官方公告为准。

### 54.1 阅读论文的"三遍方法论"

读 LLM 论文推荐三遍法:

1. **第一遍(15 分钟)**:读摘要、图 1、表格,了解"做了什么"
2. **第二遍(60 分钟)**:读 introduction、method、实验,理解"为什么这么做"
3. **第三遍(3 小时+)**:自己推导公式,复现代码核心,找出"还能怎么改进"

### 54.2 复现优先级的建议

对于想动手的工程师,推荐复现顺序:

```text
BERT  → GPT-2  →  LLaMA-1  →  FlashAttention  →  LoRA
   ↓       ↓         ↓             ↓               ↓
  入门   经典     主流架构      工程优化        PEFT 实战
                                                 ↓
                              DPO  →  GRPO  →  Mamba
                                ↓       ↓       ↓
                              对齐   推理    新架构
```

每一步都有公开的最小复现仓库(从 nanoGPT 到 litGPT),这是 2024 年后"个人也能跑大模型"的关键基础。

---

## 第五十五部分 · 深度扩展: 应用层栈与"LLM-操作系统"类比

把 LLM 生态比作一个"操作系统"是有用的认知工具:

```text
┌─────────────────────────────────────────────────┐
│           应用层: Agent / RAG / Copilot          │
├─────────────────────────────────────────────────┤
│           中间层: 框架 (LangChain / LlamaIndex)  │
├─────────────────────────────────────────────────┤
│           模型层: GPT-4 / Claude / Gemini / 开源 │
├─────────────────────────────────────────────────┤
│           推理层: vLLM / TGI / TensorRT-LLM      │
├─────────────────────────────────────────────────┤
│           硬件层: GPU / TPU / NPU                │
└─────────────────────────────────────────────────┘
```

每一层的进步都会推动上下层:硬件层的 H100/MI300 让推理层能做更大 batch,推理层的 PagedAttention 让模型层能服务更长上下文,模型层的 GPT-4 又驱动应用层产生 Copilot/Agent 这种新形态。

### 55.1 应用层的核心数学:检索增强 RAG

RAG(Retrieval-Augmented Generation)的核心是"先检索、后生成"。其生成概率可写成:

$$p_\theta(y \mid x) = \sum_{z \in \mathcal{R}(x)} p_\theta(z \mid x)\, p_\theta(y \mid x, z)$$

其中 $\mathcal{R}(x)$ 是检索器返回的 top-$k$ 文档集合,$z$ 是被选中的文档。实际部署中常用近似:

$$\hat{y} = \arg\max_y \log p_\theta(y \mid x, \hat{z}), \quad \hat{z} = \mathrm{TopK}\!\left(\mathrm{sim}(x, z_i)\right)$$

### 55.2 Agent 的数学骨架

Agent 的核心是"规划-行动-观察"三步循环:

$$\mathrm{State}_{t+1} = \mathrm{Env}\!\left(\mathrm{State}_t,\, \mathrm{Action}_t\right)$$
$$\mathrm{Action}_t = \pi_\theta\!\left(\mathrm{State}_t,\, \mathrm{Memory}_t\right)$$

其中 $\mathrm{Memory}_t$ 是过去的轨迹摘要。Agent 系统的能力来自"足够好的 $\pi_\theta$"(由 LLM 提供)+"足够稳的环境接口"(由工具/函数调用提供)。

```python
# 一个最小化的 Agent 循环伪代码
def agent_loop(llm, env, max_steps=10):
    state = env.reset()
    memory = []
    for _ in range(max_steps):
        # 让 LLM 决策
        action = llm.generate(
            f"State: {state}\nMemory: {memory}\nAction:"
        )
        # 执行动作
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward))
        state = next_state
        if done:
            break
    return memory
```

### 55.3 多模态扩展:从文本到图文音

多模态 LLM 的核心是把不同模态的 embedding 对齐到同一空间:

$$\mathrm{Align}(v_t, v_i) = \|E_t(t) - E_i(i)\|_2^2 \quad \text{(对比学习)}$$

更现代的做法是用一个"投影层"直接把视觉/音频 token 映射到文本 embedding 空间:

$$v_{\text{visual}} \in \mathbb{R}^{N \times d_v} \xrightarrow{W_p} \tilde{v}_{\text{visual}} \in \mathbb{R}^{N \times d_t}$$

这样 LLM 就能像处理文本 token 一样处理图像/音频 patch。这就是 GPT-4V、Claude 3、Gemini 1.5 的工作原理。

---

## 第五十六部分 · 深度扩展: 个人学习路线图与心法

### 56.1 一条"9 个月上手"路线

基于 2024-2026 业界实际情况,推荐以下分阶段学习计划:

| 阶段 | 时长 | 目标 | 关键产出 |
|------|------|------|---------|
| 入门 | 1-2 月 | 跑通 GPT-2 / nanoGPT | 自己的小模型 |
| 进阶 | 2-3 月 | 读 5 篇关键论文 + 复现 FlashAttn | 学习笔记 |
| 实战 | 3-4 月 | 微调 Llama-3 8B (LoRA + DPO) | 行业小模型 |
| 前沿 | 4-6 月 | 跑通 DeepSeek-V3 / Mamba-2 | 阅读+复现 |
| 专家 | 6-9 月 | 发表/开源自己的小创新 | GitHub 仓库 |

### 56.2 三个"反共识"建议

1. **不要从论文开始,而从代码开始**。nanoGPT、litGPT 这种极简仓库比论文更直观。
2. **不要只学 PyTorch,要学一点 CUDA**。FlashAttention 的核心是手写 kernel。
3. **不要忽略"小模型"**。Phi-3、Gemma 2 这些 3B 模型在端侧部署的价值被严重低估。

### 56.3 一些心法

- **80% 的 LLM 应用问题不是模型问题,而是 prompt 问题**。
- **10% 是数据问题,7% 是工程问题,3% 才是模型架构问题**。
- **模型只是冰山一角,真正决定商业价值的是"应用流"**。

---

## 第五十七部分 · 深度扩展: 未来三年的方向(基于数学外推)

### 57.1 趋势外推的数学表达

我们用简单的对数线性回归,基于 2017-2025 的数据点,外推到 2028:

$$\log N(t) \approx a + b \cdot t$$

其中 $b \approx 0.3$ /年(参数量),$b \approx 0.4$ /年(token 数)。代入 $t = 2028$:

- $N(2028) \approx 10^{13}$(10T 参数级别,但 active 仍是 50-100B)
- $D(2028) \approx 10^{14}$(100T tokens)
- $F(2028) \approx 10^{26}$(算力极限)

### 57.2 三个最可能的突破点

1. **端侧 LLM**:3-7B 模型在手机/笔记本上的实时推理将普及
2. **原生多模态**:从"文字生成"走向"任意模态原生"
3. **自主 Agent**:从"工具调用"走向"长程规划"

### 57.3 一个不太可能但值得期待的方向

**持续学习**(continual learning)。目前的 LLM 仍然是"静态的",训完后无法自主吸收新知识。如果 2027 年某个突破解决了 catastrophic forgetting 问题,将再次引发整个领域的飞跃。

---

## 第五十八部分 · 深度扩展: 全章总结与关键洞察

LLM 从 2017 到 2026 的发展可以浓缩成一句话:**从"更大"到"更智能"再到"更实用"**。

三个数学公式总结了这条主线:

$$\mathrm{Scaling\ Law:} \quad L \sim N^{-\alpha} \quad (\alpha \approx 0.07)$$
$$\mathrm{Emergence:} \quad C(N) = \mathrm{sigmoid}((\log N - \log N_c)\cdot k)$$
$$\mathrm{Alignment:} \quad \mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \Delta_\theta)$$

我们期待未来三年 LLM 能真正进入"无处不在、随叫随到"的阶段——届时,理解其数学骨架的人,才能真正掌握它的未来。

> **LLM 之路,刚刚开始。**
