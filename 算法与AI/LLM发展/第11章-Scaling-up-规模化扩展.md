---
title: "第11章-Scaling-up-规模化扩展"
tags:
  - LLM
  - Scaling
  - up
  - 规模化扩展
description: "本章要点：2017 Transformer (Vaswani et al.) - 65M 参数,首次定义架构"
chapter_id: LLM发展-11
last_updated: 2026-07-14
---

# 第11章-Scaling-up-规模化扩展

## 第11章 Scaling up - 规模化扩展教程

> Scaling 是大模型过去十年最核心的"魔法"。本章从 Scaling Laws 的数学讲起,延伸到 Pre-train Scaling、Post-train Scaling、Test-time Scaling,以及工业界规模化扩展的工程实践。

---

## 1. Scaling 的故事:从"调参"到"调规模"

### 1.1 Scaling 的历史时间线

```text
2017  Transformer (Vaswani et al.) - 65M 参数,首次定义架构
2018  GPT-1 (117M) / BERT (340M) - 规模仍小
2019  GPT-2 (1.5B) / T5 (11B) - 首次系统化 scaling 研究
2020  GPT-3 (175B) - 证明 in-context learning
2020  Scaling Laws (Kaplan et al.) - 首个 scaling law 论文
2022  Chinchilla (DeepMind) - 数据 vs 参数量平衡
2022  Emergent Abilities (Wei et al.) - 涌现能力论文
2023  LLaMA-1 (65B, 1.4T tokens) - 开源 scaling 范式
2023  LLaMA-2 (70B) / GPT-4 / Claude 2
2024  LLaMA-3 (405B, 15.6T tokens) / Gemini 1.5
2024  DeepSeek-V3 (671B MoE, $5.5M)
2024  o1 (OpenAI) - Test-time scaling 开端
2025  DeepSeek-R1 / o3 - Reasoning scaling
2025  LLaMA-4 (Meta) - MoE 旗舰
```

### 1.2 Scaling 的两个维度

| 维度 | 含义 | 典型 Scaling |
| :--- | :--- | :--- |
| **Pre-train Scaling** | Pretrain 阶段扩展 | 模型大小、数据、算力 |
| **Post-train Scaling** | 后训练扩展 | RLHF 数据、RM 规模、RL 步数 |
| **Test-time Scaling** | 推理时扩展 | 推理时算力、CoT 长度、采样次数 |

> **关键洞察**: Scaling 不是单一维度,而是 pre-train + post-train + test-time 三轴。

### 1.3 Scaling 的"魔法"案例

**In-Context Learning(ICL)涌现**:
- 1B 参数模型:无 ICL 能力
- 10B+ 参数:ICL 突然出现
- 100B+ 参数:ICL 接近 fine-tune 质量

**Chain-of-Thought(CoT)涌现**:
- < 60B 参数:CoT 几乎无用
- ≥ 100B 参数:CoT 显著提升(GSM8K: 10% → 50%+)

**多步推理(Code/SQL)涌现**:
- 1B-10B:几乎不能做
- 100B+:开始能做多步规划

---

## 2. Kaplan Scaling Laws(2020)

### 2.1 核心发现

OpenAI 团队 2020 年发布,首次系统化提出 scaling law:

> 模型的损失 $L$ 与参数量 $N$、数据集大小 $D$、训练算力 $C$ 存在幂律关系:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

### 2.2 关键公式

**Compute-optimal 模型大小**:
$$N_{\text{optimal}} \propto C^{0.73}$$

**Compute-optimal 数据大小**:
$$D_{\text{optimal}} \propto C^{0.27}$$

也就是说,算力翻 10x,模型大小应该翻 5.4x,数据应该翻 1.86x。

### 2.3 Kaplan 的局限

- 训练算力范围较窄(1e9 ~ 1e12 FLOPs)
- 没考虑 inference cost
- 没考虑训练效率(epoch 数)
- 训练 token 远低于最优

**这就是为什么 GPT-3 用了 300B tokens 训 175B 模型,被 Chinchilla 指出是"训练不充分"**。

### 2.4 Kaplan 论文复现代码

```python
import numpy as np
import matplotlib.pyplot as plt

def kaplan_loss(N, alpha_N=0.076, N_c=8.8e13):
    """Kaplan scaling law"""
    return (N_c / N) ** alpha_N

def compute_optimal_N(C, A=1.3e8, B=1.5e4):
    """最优模型大小(算力 C, FLOPs)"""
    # N_opt = (C / A) ** 0.73
    return (C / A) ** 0.73

def compute_optimal_D(C, A=1.3e8, E=1.1e2):
    """最优数据大小"""
    return (C / E) ** 0.27

# 验证:GPT-3 用了 ~3.14e23 FLOPs 训 175B
C_gpt3 = 3.14e23
N_optimal = compute_optimal_N(C_gpt3)
print(f"GPT-3 算力下,最优模型大小: {N_optimal/1e9:.1f}B")
# 输出:GPT-3 算力下,最优模型大小: ~ 略大于 175B (基本对齐)
```

---

## 3. Chinchilla Scaling Laws(2022)

### 3.1 核心发现

DeepMind 2022 年的 Chinchilla 论文指出 Kaplan 的训练不充分:

**核心结论**: 给定算力 $C$,最优模型大小 $N$ 和数据量 $D$ 满足:

$$\boxed{D \approx 20 \cdot N}$$

即每个模型参数应配 ~20 个 tokens。

### 3.2 Chinchilla 的精确公式

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

其中:
- $A, B, E$ 是常数
- $\alpha \approx 0.34$, $\beta \approx 0.28$

**给定算力 $C = 6ND$(forward + backward)**:

$$N_{\text{optimal}} \approx 0.65 \cdot C^{0.45}$$

$$D_{\text{optimal}} \approx 13.0 \cdot C^{0.55}$$

### 3.3 Chinchilla 对工业界的影响

| 模型 | 参数 | 训练 tokens | Chinchilla 推荐 | 是否"过度训练" |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-3** | 175B | 300B | 3.5T | 是,数据不足 |
| **Gopher** | 280B | 400B | 5.6T | 是 |
| **Chinchilla** | 70B | 1.4T | 1.4T | **最优** |
| **LLaMA-1** | 65B | 1.4T | 1.3T | **接近最优** |
| **LLaMA-2** | 70B | 2T | 1.4T | 略过训 |
| **LLaMA-3** | 405B | 15.6T | 8.1T | **过度训练** |

**LLaMA-3 的"反 Chinchilla"**: 8x 过训,但效果反而更好。说明数据质量/多样性的提升,数据可以超过 20N 仍有效。

### 3.4 实际训练 token 的演变

```text
2018 GPT-1:    1B tokens
2019 GPT-2:    10B tokens
2020 GPT-3:    300B tokens
2023 LLaMA-1:  1.4T tokens
2023 LLaMA-2:  2T tokens
2024 LLaMA-3:  15.6T tokens (8x 过 Chinchilla)
2024 DeepSeek-V3: 14.8T tokens
2025 (推测):   50T+ tokens
```

### 3.5 Chinchilla 后的"反 Chinchilla"

**现象**: 2023-2024 年,主流模型训练 token 都远超 20N。

**原因**:
1. **数据质量**: 高质量数据可以"复用",多次 epoch 不损伤
2. **数据多样性**: 多领域、多语言、多模态,扩展性更好
3. **推理成本**: 小模型推理成本低,但任务能力不足,折中考虑
4. **训练算力约束**: 模型大小可继续扩,但 token 受限

**新公式**(LLaMA-3 经验):

$$\text{Effective Compute} \propto N^{0.5} \cdot D^{0.5}$$

---

## 4. 涌现能力与相变

### 4.1 什么是涌现能力

**Emergent Ability**: 在小模型中不存在,在某个规模阈值后突然出现的能力。

代表论文:Wei et al. 2022, "Emergent Abilities of Large Language Models"

### 4.2 涌现能力列表

| 能力 | 涌现规模 | 代表 Benchmark |
| :--- | :--- | :--- |
| **In-Context Learning** | ~10B | 多个 ICL 任务 |
| **Chain-of-Thought** | ~60B | GSM8K |
| **多步推理** | ~100B | MATH, MMLU |
| **指令遵循** | ~10B | IFEval |
| **代码生成** | ~10B | HumanEval |
| **多语言** | ~50B | XCOPA |
| **工具使用** | ~100B | API-Bank |

### 4.3 涌现的可解释性

**两种观点**:

1. **真涌现**: 能力由"质的飞跃"产生,不可平滑扩展
2. **平滑扩展**: 涌现是评估指标的"非线性"造成(如 exact match vs partial match)

**Schaeffer et al. 2023**("Are Emergent Abilities a Mirage?")指出: 用平滑指标(如 token-level log-likelihood),能力增长是平滑的;用严格指标(如 exact match),才出现"涌现"。

### 4.4 相变与 Scaling

```python
import numpy as np

def sigmoid_emergence(N, threshold=60, sharpness=0.5):
    """涌现能力的 sigmoid 模型"""
    return 1.0 / (1.0 + np.exp(-sharpness * (N - threshold)))

# 模型大小 vs 能力
N = np.logspace(1, 3, 100)  # 10B ~ 1000B
ability = sigmoid_emergence(N, threshold=60, sharpness=0.3)

import matplotlib.pyplot as plt
plt.semilogx(N, ability)
plt.xlabel('Model Size (B)')
plt.ylabel('Capability')
plt.title('Emergent Capability: Sigmoid Model')
plt.grid()
```

---

## 5. 模型规模扩展(Pre-train Scaling)

### 5.1 模型大小的选择

| 应用 | 推荐模型大小 | 理由 |
| :--- | :--- | :--- |
| **学术研究** | 1B-7B | 单卡可训 |
| **垂直领域** | 7B-13B | 性价比最佳 |
| **通用助手** | 70B-200B | 接近 GPT-4 |
| **前沿旗舰** | 400B+ | SOTA |
| **MoE 旗舰** | 1T+ 总参, 30-50B 激活 | 算力效率 |

### 5.2 Scaling 与训练成本

$$\text{Cost} = N \cdot D \cdot 6 \cdot P_{\text{FLOP/s}} \cdot \text{time}$$

| 模型 | N | D | 训练 FLOPs | GPU 数 | 训练时间 | 成本 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-1 7B | 7B | 1.4T | 6.6e19 | 80 A100 | 14 天 | $1M |
| LLaMA-2 70B | 70B | 2T | 1.7e22 | 1000 A100 | 30 天 | $20M |
| GPT-4 | ~1.8T | ~13T | ~2.1e25 | 25000 A100 | 90-180 天 | $100M+ |
| DeepSeek-V3 | 671B MoE | 14.8T | 2.5e24 | 2048 H800 | 57 天 | $5.5M |
| LLaMA-3 405B | 405B | 15.6T | 4.9e24 | 16K H100 | 54 天 | $300M+ |

### 5.3 训练成本优化技术

1. **混合精度(FP16/BF16)**: 算力需求减半
2. **FlashAttention**: 长序列训练算力节省 4x
3. **ZeRO / FSDP**: 多卡显存共享
4. **Activation Checkpointing**: 显存节省 30%
5. **FP8 训练(H100)**: 算力再减半
6. **MoE 架构**: 激活参数仅 5-10%,算力降 5-10x

---

## 6. 数据规模扩展

### 6.1 数据来源

| 来源 | 占比 (典型) | 特点 |
| :--- | :--- | :--- |
| **CommonCrawl** | 50-70% | 海量网页,质量参差 |
| **GitHub** | 5-10% | 代码,质量高 |
| **Wikipedia** | 3-5% | 事实性,多语言 |
| **Books** | 3-5% | 长文本,高质量 |
| **学术论文** | 2-5% | 严谨,专业 |
| **问答/对话** | 5-15% | 指令数据 |
| **多模态** | 5-20% | 图像/视频/音频 |

### 6.2 数据处理 Pipeline

```text
原始数据(数百 TB)
   ↓
1. URL 过滤 (黑名单/白名单)
   ↓
2. 启发式过滤 (长度/语言/格式)
   ↓
3. 去重 (exact + fuzzy + semantic)
   ↓
4. 质量打分 (fastText/ML/heuristic)
   ↓
5. 领域分类
   ↓
6. 污染去除 (benchmark contamination)
   ↓
高质量数据(数 TB ~ 数十 TB)
```

### 6.3 数据配比

LLaMA-3 的数据配比(经验):
```text
CommonCrawl:    50% (但其中 70% 通过质量过滤)
GitHub:         15%
Wikipedia:      8%
Books:          5%
Papers:         5%
Q&A / Forums:   12%
其他:           5%
```

### 6.4 数据质量与数量 Scaling

**关键论文**: "Scaling Data-Constrained Language Models"(Muennighoff 2023)

发现: 数据有限时,数据重复可以"凑"出更多训练量,但有上限(4 epoch 后收益递减)。

```python
# 训练 token = 数据 × epoch
# 例:10T 数据 × 1 epoch = 10T tokens
# 2T 数据 × 5 epoch = 10T tokens(可能略差)

# 经验法则:
# - 高质量数据:可训 4+ epoch
# - 低质量数据:1 epoch 就够了
# - 重复数据:2-3 epoch 收益递减
```

### 6.5 数据合成的角色

**Self-Instruct / Evol-Instruct**: LLM 生成 SFT 数据
**OSS-Instruct**: 从代码仓库生成指令
**MathInstruct / MetaMath**: LLM 重写数学题

代表:LIMA(1K 数据胜 100K)、Alpaca(52K)、WizardLM(200K)、UltraChat(1M)

---

## 7. 训练算力扩展

### 7.1 Compute-Optimal Scaling

$$\text{Compute} \approx 6 \cdot N \cdot D \quad (\text{FLOPs})$$

其中 6 = 2 (forward) + 4 (backward, 因 attention 等操作反向是 forward 的 2 倍)。

### 7.2 算力增长趋势

```text
2017 Transformer:    1e19 FLOPs  (几天,1卡)
2018 GPT-1:          1.4e19 FLOPs
2019 GPT-2 (1.5B):   2.4e20 FLOPs  (数十卡 × 数天)
2020 GPT-3 (175B):   3.14e23 FLOPs (千卡 × 月)
2022 Chinchilla:     5.76e23 FLOPs
2023 LLaMA-2 70B:    1.7e24 FLOPs
2024 GPT-4:          ~2.1e25 FLOPs
2025 推测:           ~1e26 FLOPs
```

10 年算力增长 ~10^7 倍。

### 7.3 GPU 集群规模

| 模型 | GPU 数 | GPU 类型 | 互联 |
| :--- | :--- | :--- | :--- |
| GPT-3 175B | 1,000 | V100 | InfiniBand |
| LLaMA-2 70B | 2,000 | A100 | NVLink+IB |
| GPT-4 | 25,000 | A100 | NVLink+IB |
| LLaMA-3 405B | 16,000 | H100 | NVLink+IB |
| DeepSeek-V3 | 2,048 | H800 | NVLink+IB |

---

## 8. 长上下文 Scaling

### 8.1 长上下文的"为什么"

- **多文档理解**: 长报告、论文
- **多轮对话**: 累积上下文
- **代码仓库**: 百万行级
- **多模态**: 视频 = 时序长序列
- **RAG 优化**: 更长的检索结果

### 8.2 长上下文 Scaling 的核心挑战

**核心**: Transformer 注意力复杂度 $O(T^2)$。

| 序列长度 | 标准 Attention 显存 | 标准 Attention 算力 |
| :--- | :--- | :--- |
| 4K | 128 MB | 1x |
| 16K | 2 GB | 16x |
| 64K | 32 GB | 256x |
| 256K | 512 GB | 4096x |

### 8.3 长上下文的技术方案

| 技术 | 复杂度 | 适用 |
| :--- | :--- | :--- |
| **FlashAttention** | $O(T)$ 访存 | 32K-256K |
| **Sliding Window** | $O(T \cdot W)$ | 任意长,但损失全局 |
| **Linear Attention** | $O(T)$ | 无限长,但质量略差 |
| **Ring Attention** | $O(T)$ 多卡 | 100K-10M |
| **RoPE 外推** | $O(T^2)$ 算力 | 4x 训练长度 |
| **YaRN / NTK-aware** | $O(T^2)$ | 16x 训练长度 |
| **InfLLM / StreamingLLM** | $O(\sqrt{T})$ | 任意长 |

### 8.4 长上下文评估:Needle-in-a-Haystack

```python
# 经典测试:把"针"插入长文本,看模型能否找到
needle = "The best thing to do in San Francisco is eat a sandwich at the park."
haystack = "long_text_with_needle_inserted_at_position"

prompt = f"""Find this specific information in the text below:
{needle}

Text:
{haystack[:10000]}"""

# 模型输出
response = model.generate(prompt)
assert "eat a sandwich" in response
```

### 8.5 长上下文 Scaling 的实际应用

| 模型 | 上下文 | 训练长度 | 外推方法 |
| :--- | :--- | :--- | :--- |
| LLaMA-2 | 4K | 4K | - |
| LLaMA-3 | 128K | 8K | RoPE base 500K + 训练长度 |
| Mistral | 32K | 32K | RoPE base 1M |
| Claude 3 | 200K | 200K | 多项优化 |
| Gemini 1.5 | 1M-2M | 1M | 多项优化 |
| Qwen-2.5 | 128K-1M | 32K | YaRN + DCFF |

---

## 9. 多模态 Scaling

### 9.1 多模态扩展的动机

- 人类智能 80% 来自视觉
- 文字数据有限,多模态扩展空间大
- 多模态对齐是 AGI 必备

### 9.2 多模态扩展的方向

| 模态 | 训练数据 | 代表模型 |
| :--- | :--- | :--- |
| **图像+文本** | LAION, CC3M | CLIP, LLaVA, Qwen-VL |
| **视频+文本** | YT, Kinetics | Video-LLaVA, InternVideo |
| **音频+文本** | AudioSet, LibriSpeech | AudioLM, Qwen-Audio |
| **3D+文本** | Objaverse, ShapeNet | PointLLM |
| **代码+文本** | GitHub | CodeLLaMA, DeepSeek-Coder |
| **多模态混合** | 混合 | GPT-4V, Gemini |

### 9.3 多模态 Scaling Laws

**关键论文**: "Multimodal Scaling Laws"(Aghajanyan 2023)

发现:
- 视觉 token 的 scaling 与文本类似
- 图文对齐需要 ~1e9 样本
- 多模态涌现阈值比纯文本高 10x

### 9.4 多模态训练 Pipeline

```text
Stage 1: 视觉编码器预训练 (CLIP-style 对比学习)
Stage 2: 视觉-语言对齐 (MLP projector + 冻结 LLM 训练)
Stage 3: 视觉指令微调 (VLM SFT)
Stage 4: 多模态 RLHF (偏好数据 + 视觉 RM)
```

代表:LLaVA-1.5, Qwen-VL, InternVL, GPT-4V

---

## 10. Post-train Scaling(RLHF 规模化)

### 10.1 RLHF 数据 Scaling

**LLaMA-2 RLHF 数据**:
- 1.4M 偏好对 (UltraFeedback, Anthropic HH 等)
- 多轮迭代训练
- 每次迭代加 100K-500K 新偏好

**发现**:
- RLHF 数据量与能力提升的对数关系
- 多样性比数量重要
- 高质量 RM 需要 ≥100K 数据

### 10.2 RLHF 计算 Scaling

**Iterative RLHF**:
- 用最新模型生成新偏好数据
- 训练新 RM
- 用新 RM 训练新模型
- 循环

代表:Self-Rewarding LM (Meta 2024), RLHF-V

### 10.3 后训练 Scaling 的实际收益

| 模型 | SFT-only | + RLHF | 提升 |
| :--- | :--- | :--- | :--- |
| Alpaca | 52% win | - | - |
| Vicuna | - | - | - |
| LLaMA-2-Chat | 70% win | 90% win | +20% |
| GPT-4 (推测) | - | - | - |

> **关键**: RLHF 主要提升"对齐"(用户感知),对 benchmark 分数提升有限(2-5%)。

---

## 11. Test-time Scaling(推理时扩展)

### 11.1 什么是 Test-time Scaling

> Test-time Scaling = 在推理时投入更多算力,获得更好输出。

三种形式:
1. **Self-consistency**: 多次采样,投票
2. **Chain-of-Thought**: 增加 CoT 长度
3. **Tree-of-Thought / Search**: 结构化搜索

### 11.2 Self-Consistency(2022)

```python
def self_consistency(prompt, model, n_samples=10, temperature=0.7):
    """多次采样,投票选最终答案"""
    answers = []
    for _ in range(n_samples):
        out = model.generate(prompt, temperature=temperature, max_tokens=512)
        # 提取最终答案
        answer = extract_final_answer(out)
        answers.append(answer)
    # 投票
    from collections import Counter
    final_answer = Counter(answers).most_common(1)[0][0]
    return final_answer

# GSM8K: 60% → 75% (+15%) with n=10 samples
```

### 11.3 Chain-of-Thought Scaling

```python
def cot_scaling(model, problem, n_cot_samples=10):
    """多 CoT 采样,验证答案"""
    candidates = []
    for _ in range(n_cot_samples):
        cot = model.generate(
            problem,
            temperature=0.7,
            max_tokens=2000,
            system="Let's think step by step.",
        )
        answer = extract_answer(cot)
        candidates.append((cot, answer))
    # 一致性过滤
    from collections import Counter
    answers = [a for _, a in candidates]
    majority = Counter(answers).most_common(1)[0][0]
    # 选择 majority 的 CoT
    for cot, ans in candidates:
        if ans == majority:
            return cot, ans
    return candidates[0]
```

### 11.4 o1 / R1 风格的 Test-time Scaling

**OpenAI o1**(2024 年 9 月):首次系统化"推理时扩展"
**DeepSeek-R1**(2025 年 1 月):开源复现 + 详细技术报告

**核心思想**:
- 训练时:RL 激励模型生成"长 CoT"
- 推理时:模型自动进行多步推理、反思、回溯

```python
# o1 / R1 风格的 prompt
prompt = """
Solve this problem by thinking very carefully.

Problem: What is the sum of the first 1000 positive integers?

Let me think step by step:
1. First, I recall the formula for sum of arithmetic series...
2. Or I can use the formula n(n+1)/2 where n=1000
3. Calculating: 1000 * 1001 / 2 = 500500
4. Verify: 1000+999+...+1 = ?

Actually let me verify with the pairing method:
1+1000=1001, 2+999=1001, ..., 500+501=1001
There are 500 pairs, each summing to 1001.
Total = 500 * 1001 = 500500

Answer: 500500
"""

# 模型自动产生这种长思考过程
```

### 11.5 Test-time Scaling 的 Scaling Law

**Snell et al. 2024**: "Scaling LLM Test-Time Compute"

发现:对于困难任务,test-time compute 的对数增长,与 pretraining compute 的对数增长,产生类似的能力提升。

$$L(\text{test-time compute}) \propto C_{\text{test}}^{-\alpha}$$

这意味着可以用"推理算力"换"训练算力"。

### 11.6 Test-time Scaling vs Pre-train Scaling

| 维度 | Pre-train Scaling | Test-time Scaling |
| :--- | :--- | :--- |
| 成本 | 一次性 | 每次推理 |
| 适用任务 | 通用 | 主要用于推理任务 |
| 收益 | 普遍提升 | 推理任务显著 |
| 代表 | GPT-4 (175B → 1.8T) | o1 (推理 100x 算力) |
| 折中 | 推理便宜,通用差 | 推理贵,任务特化 |

### 11.7 Test-time Scaling 的工程挑战

1. **延迟**: 单 query 耗时翻倍到 10x
2. **成本**: 用户付费 vs 服务商亏损
3. **KV Cache**: 长 CoT 需要大 cache
4. **模型设计**: 需要"内部推理"能力

代表方案:
- OpenAI o1: 推理时 ~100x 算力
- DeepSeek-R1: 开源复现
- Anthropic extended thinking

---

## 12. 数据质量 Scaling

### 12.1 质量 Scaling 的发现

**LIMA 论文**(2023): 1,000 条高质量数据 > 100,000 条低质量数据
**Phi 论文**(2023): 用 GPT-3.5 生成"教科书质量"数据,1.3B 模型达到 175B 效果

### 12.2 数据质量评估

| 维度 | 评估方法 |
| :--- | :--- |
| **Perplexity** | 低 PPL 数据通常更有意义 |
| **Reward Model** | 高 RM 分数据更优 |
| **教育价值** | GPT-4 评分"教学价值" |
| **多样性** | embedding 聚类评估 |
| **去重** | exact + semantic deduplication |

### 12.3 数据配比的 Scaling

**D-CPT**(2023): 用模型损失作为数据权重,自动发现最优配比

```python
def compute_data_weights(domain_losses, total_loss):
    """D-CPT 数据权重计算"""
    weights = {}
    for domain, loss in domain_losses.items():
        # Loss 越低,贡献越小,可以减少
        weights[domain] = loss / total_loss
    return weights

# 例:数学 loss 低 → 数学数据权重低
# 反而需要提升数学数据量
```

### 12.4 Phi-3 / 数据 Scaling 革命

**Phi-3 Mini**(3.8B): 用 3.3T 高质量"合成"训练数据
- 远超 Phi-2 的数据
- 但比 Llama-3 70B 还快

**启示**: 数据质量 + 数量,可以"以小胜大"。

---

## 13. Scaling 的极限与挑战

### 13.1 数据耗尽

**Villalobos et al. 2024**: "Will we run out of data?"

预测: 高质量文本数据将在 2026-2030 年耗尽。

**应对**:
- 合成数据(LLM 生成)
- 多模态扩展(图像/视频)
- 数据复用(多 epoch)
- 自我改进(self-improve)

### 13.2 计算 Scaling 极限

**GPU 算力增长**: 2-3 年翻倍(摩尔定律放缓)
**电力与冷却**: 数据中心建设受电力限制

### 13.3 评估困难

**现象**: 模型在 benchmark 上接近 SOTA,但实际能力未必同步提升。

**Benchmark 污染**: 训练数据可能包含 benchmark 答案。
**Eval 集局限**: MMLU 等不能完全代表能力。

### 13.4 推理成本爆炸

**训练成本 1x, 推理成本 1000x**:
- GPT-4 训练成本 ~$100M
- GPT-4 推理每天收入 $1M+ (ChatGPT 用户)
- o1 推理成本是 GPT-4 的 10-100x

**长期不可持续**: 需要更高效的推理方案。

---

## 14. 工业界 Scaling 案例

### 14.1 GPT 系列(OpenAI)

| 模型 | 时间 | 参数量 | 训练算力 |
| :--- | :--- | :--- | :--- |
| GPT-1 | 2018 | 117M | ~1e19 |
| GPT-2 | 2019 | 1.5B | ~2e20 |
| GPT-3 | 2020 | 175B | ~3e23 |
| GPT-3.5 | 2022 | ~175B+ | ~3e23 |
| GPT-4 | 2023 | ~1.8T (MoE 推测) | ~2e25 |
| GPT-4o | 2024 | 推测同 GPT-4 | + RLHF/SFT |
| o1 | 2024 | 推测 GPT-4 级别 | + test-time |
| o3 | 2024 | 更大 | + test-time 100x |

### 14.2 Claude 系列(Anthropic)

- Claude 1/2: 类似 GPT-4 早期
- Claude 3 Opus/Sonnet/Haiku: 多尺寸策略
- Claude 3.5 Sonnet: 编程 + 推理双优
- Claude 4: 引入"extended thinking"

### 14.3 Gemini 系列(Google)

- Gemini 1.0: 原生多模态
- Gemini 1.5: 1M 上下文 + MoE
- Gemini 2.0: Agent + 多模态推理

### 14.4 LLaMA 系列(Meta)

- LLaMA-1: 开源范式 (2023)
- LLaMA-2: RLHF 增强 (2023)
- LLaMA-3: 405B 旗舰 + 15.6T tokens (2024)
- LLaMA-4: MoE 原生 (2025, 推测)

### 14.5 DeepSeek 系列

- DeepSeek-V1: 67B 早期
- DeepSeek-V2: MLA + 236B MoE (2024)
- DeepSeek-V3: 671B MoE + $5.5M 训练成本 (2024)
- DeepSeek-R1: GRPO + Test-time (2025)

**DeepSeek 的成本革命**: 用算法 + 工程,把训练成本压到 1/30 同等规模模型。

---

## 15. 分布式训练架构

### 15.1 并行策略对比

| 策略 | 切分维度 | 通信需求 | 适用 |
| :--- | :--- | :--- | :--- |
| **DP(Data Parallel)** | 数据 | AllReduce 梯度 | 大 batch |
| **TP(Tensor Parallel)** | 矩阵切分 | AllGather + ReduceScatter | 大 FFN |
| **PP(Pipeline Parallel)** | 层切分 | Send/Recv | 超大模型 |
| **EP(Expert Parallel)** | MoE 专家 | All-to-All | MoE |
| **SP(Sequence Parallel)** | 序列切分 | AllGather | 长序列 |
| **CP(Context Parallel)** | 类似 SP | Ring Attention | 1M+ 上下文 |

### 15.2 3D Parallelism(DP + TP + PP)

```text
8 卡节点 × 8 节点 = 64 卡训练 70B 模型
- TP=8(节点内 NVLink 高速)
- PP=4(层切到 4 段)
- DP=2(2 路数据并行)

总并行度 = 8 × 4 × 2 = 64 卡
```

代表实现:Megatron-LM, DeepSpeed

### 15.3 ZeRO 系列再回顾

| 阶段 | 优化项 | 显存节省 | 通信开销 |
| :--- | :--- | :--- | :--- |
| **ZeRO-1** | 优化器状态分片 | 4x | 略增 |
| **ZeRO-2** | + 梯度分片 | 8x | 略增 |
| **ZeRO-3** | + 参数分片 | N 倍 (N=并行度) | 显著增 |
| **ZeRO-Infinity** | + CPU/NVMe Offload | 无限 | 显著增 |

### 15.4 大模型训练通信开销

$$\text{通信量} \propto \frac{N_{\text{model}}}{N_{\text{GPUs}}} + \text{gradient}$$

**经验**: TP 在 NVLink 上高效;PP 在 IB 上;DP 通信随 batch 增加。

---

## 16. 长思维链(Long CoT)训练

### 16.1 Long CoT 的本质

**Short CoT**: 直接给答案
**Long CoT**: 思考 → 反思 → 修正 → 验证 → 输出

### 16.2 Long CoT 数据合成

```python
def synthesize_long_cot(problem, model, verifier):
    """用强模型 + verifier 生成 Long CoT"""
    # 1. 模型生成 CoT
    cot = model.generate(
        problem,
        max_tokens=8000,  # 允许长思考
        temperature=0.7,
        system="Think very carefully. Reflect on your approach."
    )
    # 2. 提取答案
    answer = extract_answer(cot)
    # 3. Verifier 验证
    is_correct = verifier.check(problem, answer)
    if is_correct:
        return cot, answer
    else:
        # 失败:重试,或人工修正
        return None
```

### 16.3 RLVR + Long CoT

DeepSeek-R1 流程:

```text
1. SFT on long CoT 数据(蒸馏 R1 早期版本)
2. GRPO 训练,用 verifier 验证
3. 持续迭代,RL 激励长 CoT
4. 最终模型:自主产生高质量 Long CoT
```

### 16.4 Long CoT 评估

```python
# 关键指标
def evaluate_long_cot(model, problems):
    metrics = {
        "accuracy": 0.0,
        "avg_length": 0.0,  # CoT 平均长度
        "backtrack_count": 0,  # 反思次数
        "verification_rate": 0.0,  # 自验证比例
    }
    for problem in problems:
        cot = model.generate(problem, max_tokens=16000)
        answer = extract_answer(cot)
        metrics["accuracy"] += check_answer(answer, ground_truth)
        metrics["avg_length"] += len(cot.split())
        metrics["backtrack_count"] += cot.count("Wait")
        metrics["verification_rate"] += "Let me verify" in cot
    return {k: v / len(problems) for k, v in metrics.items()}
```

---

## 17. Scaling 与经济性

### 17.1 训练成本曲线

```text
2017 Transformer: ~$1K (个人)
2018 GPT-1: ~$10K (小团队)
2019 GPT-2: ~$100K
2020 GPT-3: ~$5M
2022 Gopher/Chinchilla: ~$10M
2023 LLaMA-2 70B: ~$20M
2024 GPT-4: ~$100M+
2024 LLaMA-3 405B: ~$300M+
2025 推测: $1B+
```

10 年成本增长 ~10^7 倍,接近 GPT-4 已经接近大公司能承受的极限。

### 17.2 推理经济性

```python
# 推理成本估算
def inference_cost(model_size_b, tokens_per_query, queries_per_day,
                   cost_per_flop=1e-18):  # H100 ~1e-18 $/FLOP
    flops_per_token = 2 * model_size_b * 1e9  # forward only
    cost_per_query = flops_per_token * tokens_per_query * cost_per_flop
    daily_cost = cost_per_query * queries_per_day
    return {
        "cost_per_query": cost_per_query,
        "daily_cost": daily_cost,
        "monthly_cost": daily_cost * 30,
    }

# LLaMA-70B 推理
cost = inference_cost(70, 1000, 1e6)
# cost_per_query: $0.14
# daily_cost: $140K
# monthly_cost: $4.2M
```

### 17.3 碳足迹

| 模型 | 训练能耗 | 碳排放 |
| :--- | :--- | :--- |
| GPT-3 | 1287 MWh | ~550 吨 CO2 |
| LLaMA-2 70B | ~2000 MWh | ~1500 吨 CO2 |
| GPT-4 | ~50000 MWh | ~25000 吨 CO2 |

**对比**: 一个普通美国人一年的碳足迹 ~16 吨。GPT-4 训练 ≈ 1500 人/年。

### 17.4 成本优化方向

1. **高效架构**: MoE、Linear Attention、State-Space
2. **高效训练**: FP8、FlashAttention、ZeRO
3. **数据效率**: 高质量 + 合成数据
4. **小模型 + 推理扩展**: DeepSeek-R1 范式
5. **边缘部署**: 端侧小模型

---

## 18. Scaling 的未来方向

### 18.1 数据合成(Data Synthesis)

**Self-Play / Self-Improve**:
- 模型生成 QA 对
- 模型验证答案
- 用合成数据再训练

代表:Self-Rewarding LM, Magpie, Self-Instruct 系列

### 18.2 多模态统一

**GPT-4o, Gemini**: 原生多模态,图像/音频/视频/文本统一表征

**优势**:
- 单一模型多任务
- 跨模态推理(看图 + 听音 + 写文)
- 数据利用效率提升

### 18.3 长 CoT + Test-time Scaling

o1, R1 引领的新范式:
- 用推理算力换训练算力
- 推理任务上接近"无限 scaling"
- 适合数学/代码/科学问题

### 18.4 持续学习(Continual Learning)

**当前问题**: 训练完就固定,知识冻结
**未来**: 持续从新数据学习,无灾难性遗忘

代表方案:
- 检索增强(RAG)
- 模型编辑(Model Editing)
- Online RL

### 18.5 小模型 + 蒸馏

**核心**: 大模型的能力可以通过"蒸馏"传给小模型

代表:DistilBERT, Phi-3 (3.8B vs 175B), Gemma-2 (2B vs 70B)

**蒸馏路径**:
1. SFT 蒸馏:用大模型回答训小模型
2. Logit 蒸馏:对齐概率分布
3. CoT 蒸馏:大模型 CoT 训小模型

### 18.6 可持续 Scaling

**挑战**: 算力、数据、能源约束
**方向**:
- 算法效率(更少算力训更好模型)
- 硬件进步(GPU 算力持续提升)
- 数据利用(更多合成/多模态)
- 部署效率(推理成本控制)

---

## 19. Scaling 实验设计

### 19.1 IsoFLOP Curves(等算力曲线)

```python
# 经典 scaling 实验设计
configs = [
    # (model_size, data_size, compute)
    (50e6, 1e9, 3e17),   # 50M params, 1B tokens
    (100e6, 2e9, 6e17),
    (250e6, 4e9, 6e18),
    (500e6, 8e9, 2.4e19),
    (1e9, 16e9, 9.6e19),
    (2e9, 32e9, 3.8e20),
    # ...
]

# 在每个 (N, D) 组合上训练到收敛
# 记录 loss → 得到 isoFLOP 曲线
# 找到该算力下最优的 (N, D)
```

### 19.2 IsoLoss Curves(等损失曲线)

```python
# 找达到目标 loss 的最小算力
target_loss = 2.5
# 训练多个 (N, D) 直到 loss <= target_loss
# 记录达到 target_loss 所需的 compute
# 拟合 compute vs target_loss 曲线
```

### 19.3 Scaling 拟合代码

```python
import numpy as np
from scipy.optimize import curve_fit

def scaling_law(N, a, b, c):
    """Loss = a * N^(-b) + c"""
    return a * N ** (-b) + c

# 实验数据
N_data = np.array([1e7, 3e7, 1e8, 3e8, 1e9, 3e9])  # 模型大小
L_data = np.array([3.5, 3.0, 2.7, 2.5, 2.35, 2.25])  # 验证损失

# 拟合
popt, _ = curve_fit(scaling_law, N_data, L_data, p0=[10, 0.1, 2])
print(f"拟合: a={popt[0]:.2f}, b={popt[1]:.3f}, c={popt[2]:.3f}")

# 预测:1e10 参数的损失
print(f"预测: {scaling_law(1e10, *popt):.3f}")
```

---

## 20. 实用 Scaling Checklist

### 20.1 模型大小决策

```text
可用算力: C (FLOPs)
可用数据: D (tokens)
Chinchilla 推荐:
  N_optimal ≈ C^0.45 / 6
  D_optimal ≈ C^0.55
反 Chinchilla (LLaMA-3 经验):
  N ≈ C^0.45 / 6
  D = 8 × 20N (数据"过度")
```

### 20.2 数据准备 Checklist

- [ ] 原始数据清洗(URL/language/format)
- [ ] 去重(exact + MinHash + semantic)
- [ ] 质量打分(ML 模型 + heuristic)
- [ ] 领域分类(数学/代码/通用/...)
- [ ] Benchmark 污染检测
- [ ] 配比实验(多 epoch 训练,看各 benchmark)

### 20.3 训练 Checklist

- [ ] 选 compute-optimal (N, D)
- [ ] 学习率 warmup + cosine decay
- [ ] ZeRO / FSDP 配置
- [ ] FlashAttention v2/v3
- [ ] 监控 loss / tokens/sec / 显存
- [ ] checkpoint 策略(每 N step)
- [ ] 数据配比 ablation

### 20.4 后训练 Checklist

- [ ] SFT 数据(50K+ 高质量)
- [ ] DPO/RM 数据(100K+ 偏好对)
- [ ] RM 训练 + 验证
- [ ] PPO/DPO/GRPO 训练 + 监控
- [ ] KL 散度动态调整
- [ ] Reward Hacking 检测

---

## 21. 总结

Scaling 是大模型过去十年最核心的"魔法"。本章核心要点:

1. **Kaplan → Chinchilla**: 模型与数据应同比例扩展 (D ≈ 20N)
2. **反 Chinchilla**: 数据质量提升,数据可以"过度"
3. **涌现能力**: 某些能力在阈值后突然出现,但可能源于评估指标
4. **多轴 Scaling**: Pre-train + Post-train + Test-time 三轴协同
5. **Test-time Scaling**: o1/R1 范式,推理时算力换能力
6. **分布式 Scaling**: 3D Parallelism + ZeRO + 通信优化
7. **成本挑战**: 数据耗尽、推理成本爆炸、能源约束
8. **未来**: 算法效率 + 多模态 + 持续学习

> **核心心法**: Scaling 是科学(数学公式),也是艺术(数据质量、配比、训练细节)。**没有万能公式,只有不断实验**。

---

## 附录 A: Scaling 相关论文清单

```text
Scaling Laws:
  - Kaplan 2020: Scaling Laws for Neural Language Models
  - Chinchilla 2022: Training Compute-Optimal Large Language Models
  - Hoffmann 2022: An Empirical Analysis of Compute-Optimal LLM Training
  - Muennighoff 2023: Scaling Data-Constrained Language Models

Emergent Abilities:
  - Wei 2022: Emergent Abilities of Large Language Models
  - Schaeffer 2023: Are Emergent Abilities a Mirage?

Long Context:
  - RoPE (Su 2021)
  - YaRN (Peng 2023)
  - LongRoPE (Ding 2024)
  - Ring Attention (Liu 2024)
  - InfLLM (Xiao 2024)

Test-time Scaling:
  - Self-Consistency (Wang 2022)
  - ToT (Yao 2023)
  - Snell 2024: Scaling LLM Test-Time Compute
  - o1 (OpenAI 2024)
  - DeepSeek-R1 (2025)

Multi-modal:
  - CLIP (Radford 2021)
  - LLaVA (Liu 2023)
  - Qwen-VL (Bai 2023)
  - InternVL (Chen 2024)

Efficient Scaling:
  - MoE (Shazeer 2017)
  - Switch Transformer (Fedus 2022)
  - DeepSeekMoE (2024)
  - Phi-3 (Microsoft 2024)

Distributed Training:
  - ZeRO (Rajbhandari 2020)
  - FSDP (Zhao 2023)
  - Megatron-LM (Shoeybi 2019)
  - 3D Parallelism (Narayanan 2021)

Post-training Scaling:
  - InstructGPT (Ouyang 2022)
  - LLaMA-2-Chat (Touvron 2023)
  - Self-Rewarding LM (Yuan 2024)
  - Constitutional AI (Bai 2022)
  - RLVR (2024)
```

---

## 22. 训练 Loss 与 Benchmark 的 Scaling 关系

### 22.1 Loss 与 Benchmark 关联

**经验**: 训练 Loss 下降 0.1,通常对应 benchmark 提升 1-3%。但某些"涌现"任务上 Loss 减少 0.05 可能带来 10%+ benchmark 提升。

### 22.2 Scaling 对不同任务的不均衡影响

| 任务类型 | Scaling 收益 | 代表 benchmark |
| :--- | :--- | :--- |
| **知识类** | 强 | MMLU, ARC |
| **推理类** | 极强(涌现) | GSM8K, MATH |
| **代码类** | 强 | HumanEval, MBPP |
| **翻译** | 中 | WMT |
| **摘要** | 中 | CNN/DM |
| **对话** | 中 | MT-Bench |
| **创作** | 中 | 主观评估 |

### 22.3 Negative Log-Likelihood(NLL)与下游任务

**关键发现**: NLL 与大多数 benchmark 高度相关,但某些任务(如推理)上,NLL 变化小但 benchmark 变化大。

### 22.4 验证 Loss vs 训练 Loss

训练 Loss 与验证 Loss 的差距反映过拟合:
- 差距小:模型可继续 scaling(欠拟合)
- 差距大:模型已饱和,需要正则化或更多数据

```python
# 监控 gap
def monitor_overfit(train_loss, val_loss, threshold=0.3):
    if val_loss - train_loss > threshold:
        return "Overfitting: 增加正则 / 减少 epoch"
    if val_loss - train_loss < 0.05:
        return "Underfitting: 可继续 scaling"
    return "Healthy"
```

---

## 23. Compute-Optimal Inference(推理最优算力)

### 23.1 推理算力的组成部分

```python
def total_inference_compute(model_size_b, num_queries, tokens_per_query):
    """推理总算力 (FLOPs)"""
    flops_per_token = 2 * model_size_b * 1e9

    # Prefill + Decode
    prompt_len = tokens_per_query // 4
    generated_len = 3 * prompt_len

    prefill_flops = num_queries * prompt_len * flops_per_token
    decode_flops = num_queries * generated_len * flops_per_token * 2  # KV Cache

    return prefill_flops + decode_flops

# LLaMA-70B 推理 100 万 query, 平均 1000 token
print(total_inference_compute(70, 1e6, 1000))  # ~4.2e22 FLOPs
```

### 23.2 推理成本对比

| 模型 | 单 query 成本 | 月度成本 (1000 req/s) |
| :--- | :--- | :--- |
| LLaMA-7B | $0.0002 | $5K |
| LLaMA-70B | $0.002 | $50K |
| GPT-4 级别 | $0.03 | $750K |
| o1 级别 | $0.10-0.50 | $2.5M-12M |

### 23.3 推理算力优化技术

1. **量化**: INT8/INT4 → 2-8x 算力节省
2. **Speculative Decoding**: 提速 2-3x
3. **PagedAttention**: 显存利用率 95%+
4. **Continuous Batching**: 吞吐 5-10x
5. **MoE 部署**: 激活参数小,推理便宜

---

## 24. Scaling 的稳定性

### 24.1 Loss Spike 问题

**现象**: 训练中 loss 突然飙升(NaN 或巨大值),需要回滚 checkpoint。

**根因**:
1. 数据异常(NaN/Inf)
2. 注意力异常(softmax 满分布)
3. 数值精度(BF16 下溢)
4. 优化器状态发散

**应对**:
- Loss spike 检测 + 自动回滚
- Gradient clipping (max_grad_norm=1.0)
- QK-norm(防止注意力 logit 爆炸)
- Z-loss(稳定 logits)

```python
# Z-loss 稳定 logits
def stable_loss(logits, labels, alpha=1e-4):
    ce_loss = F.cross_entropy(logits, labels)
    z_loss = alpha * torch.logsumexp(logits, dim=-1).square().mean()
    return ce_loss + z_loss
```

### 24.2 训练失败模式分类

| 失败模式 | 表现 | 解法 |
| :--- | :--- | :--- |
| **Divergence** | Loss 持续上升 | 降低 lr, 检查数据 |
| **Loss Spike** | 突然飙升 | 回滚 checkpoint |
| **NaN** | 模型输出 NaN | Gradient clipping, 数据清洗 |
| **Plateau** | Loss 不再下降 | 调整学习率, 数据检查 |
| **Catastrophic Forgetting** | 早期能力下降 | Mix pretrain data, 低 lr |
| **Mode Collapse** | 输出多样性下降 | 数据多样性, KL 锚定 |

### 24.3 训练监控指标体系

```python
# 关键监控指标
metrics = {
    "train_loss": ...,
    "eval_loss": ...,
    "gradient_norm": ...,
    "learning_rate": ...,
    "tokens_per_second": ...,
    "gpu_memory_used": ...,
    "gpu_utilization": ...,
    "throughput": ...,
}

# 异常检测
def check_anomalies(metrics, history):
    if metrics["gradient_norm"] > 10 * history["gradient_norm"]:
        alert("Gradient explosion")
    if metrics["eval_loss"] - metrics["train_loss"] > 0.5:
        alert("Overfitting")
    if metrics["tokens_per_second"] < 0.5 * history["tokens_per_second"]:
        alert("Throughput degradation")
```

---

## 25. 数据效率 Scaling(Phi 范式)

### 25.1 Phi 模型的数据革命

**Phi-1 / Phi-1.5 / Phi-2 / Phi-3 / Phi-4** 系列展示了"数据效率"的 Scaling 革命。

| 模型 | 参数量 | 训练 token | 训练策略 | MMLU |
| :--- | :--- | :--- | :--- | :--- |
| **Phi-1** | 1.3B | 7B(代码) | 高质量"教科书"代码 | - |
| **Phi-1.5** | 1.3B | 100B | 高质量合成 + 真实 | ~50 |
| **Phi-2** | 2.7B | 1.4T | 精挑数据 | ~56 |
| **Phi-3 Mini** | 3.8B | 3.3T | LLaMA-3 同等数据 | ~70 |
| **Phi-4** | 14B | 10T+ | 混合 | ~84 |

**启示**: 数据质量提升,小模型可胜大模型。

### 25.2 高质量数据来源

1. **教科书合成**: GPT-4 生成"教学式"文本
2. **Wikipedia + 衍生**: 事实性 + 多领域
3. **Stack Overflow + GitHub**: 代码数据
4. **学术论文(精选)**: 科学严谨
5. **GPT-4 生成 QA**: 多样性高
6. **人工精标**: 高质量但成本高

### 25.3 数据去重与污染去除

```python
def deduplication_pipeline(data):
    """数据去重三阶段"""
    data = exact_dedup(data)                    # 1. Exact
    data = minhash_dedup(data, threshold=0.8)   # 2. Fuzzy
    data = semantic_dedup(data, threshold=0.95) # 3. Semantic
    return data


def check_benchmark_contamination(text, benchmark_names):
    """检查 benchmark 污染"""
    for name in benchmark_names:
        if name in text:
            return True
    return False
```

---

## 26. Scaling 与评估的"鸿沟"

### 26.1 评估指标的局限

**问题**: 模型在 benchmark 上接近 SOTA,但实际能力未必同步。

**典型鸿沟**:
- LLaMA-2 70B vs GPT-4: benchmark 接近,但用户偏好差距大
- 涌现能力可能"幻觉"(Schaeffer 2023)

### 26.2 动态评估(Dynamic Eval)

**思路**: 评估集随模型更新而更新,防止"应试"。

代表:LiveBench, SEAL Leaderboard

### 26.3 能力维度评估矩阵

| 维度 | 评估方法 |
| :--- | :--- |
| **知识** | MMLU, ARC, HellaSwag |
| **推理** | GSM8K, MATH, BIG-Bench-Hard |
| **代码** | HumanEval, MBPP, LiveCodeBench |
| **指令** | IFEval, AlpacaEval |
| **对话** | MT-Bench, LMSYS Arena |
| **安全** | HarmBench, AdvBench |
| **长文** | RULER, LongBench, Needle-in-Haystack |
| **Agent** | AgentBench, ToolBench |

### 26.4 LLM-as-Judge 评估范式

```python
def gpt4_judge(prompt, response_a, response_b):
    """用 GPT-4 做偏好判断"""
    judge_prompt = f"""
    请判断以下两个回答哪个更好。

    用户问题: {prompt}

    回答 A: {response_a}

    回答 B: {response_b}

    评估维度:
    1. 准确性 (40%)
    2. 有用性 (30%)
    3. 流畅性 (20%)
    4. 安全性 (10%)

    输出格式:
    Winner: A or B
    Reason: ...
    """
    return openai_client.generate(judge_prompt)

# 风险: GPT-4 偏好自己风格的回答
# 解法: ensemble + 多模型 judge
```

---

## 27. Scaling 与硬件的耦合

### 27.1 GPU 演进与 Scaling 同步

| 时期 | GPU | FP16 TFLOPS | HBM |
| :--- | :--- | :--- | :--- |
| 2017 | V100 | 125 | 16 GB |
| 2020 | A100 | 312 | 80 GB |
| 2022 | H100 | 989 | 80 GB |
| 2024 | B100 | 1800 | 192 GB |
| 2026 (推) | B200 | ~3000 | ~300 GB |

GPU 算力 8 年翻 ~24x。

### 27.2 NVLink 与集群互联

```text
单机 8 卡 NVLink 带宽: 900 GB/s (H100)
集群 InfiniBand HDR: 200 Gb/s (25 GB/s)
集群 InfiniBand NDR: 400 Gb/s (50 GB/s)
```

**关键**: 训练 405B+ 模型需要高带宽互联,否则通信成瓶颈。

### 27.3 训练集群规模演变

| 时期 | 典型集群 | 代表训练 |
| :--- | :--- | :--- |
| 2017 | 8-32 GPU | Transformer 训练 |
| 2020 | 1000 GPU | GPT-3 |
| 2023 | 2000 GPU | LLaMA-2 70B |
| 2024 | 16K GPU | LLaMA-3 405B |
| 2025+ | 50K+ GPU | 推测下一代旗舰 |

### 27.4 数据中心挑战

- **电力**: 50MW+ 数据中心已成常态
- **冷却**: 液冷普及,提高功率密度
- **网络**: NVLink + IB 多层互联
- **存储**: 高速 NVMe + 分布式文件系统(Lustre, GPFS)

---

## 28. Scaling 与算法协同

### 28.1 算法效率提升(免费的算力 Scaling)

| 算法 | 算力效率提升 | 等价"免费"算力 |
| :--- | :--- | :--- |
| **FlashAttention** | 4x 长 seq | 4x |
| **GeLU → SwiGLU** | 1.5x 质量 | 1.5x |
| **Pre-LN** | 5x 训练稳定性 | 5x |
| **QK-Norm** | 稳定性 | 允许更深 |
| **MoE** | 5x 激活稀疏 | 5x |
| **FP8** | 2x | 2x |

**累积**: 过去 5 年算法效率提升 ~50-100x,等价于"免费"算力。

### 28.2 算法-硬件协同设计

代表案例:
- **TPU + Transformer**: Google 为 LLM 设计专用芯片
- **Trainium / Inferentia**: AWS 专用 AI 芯片
- **FP8 训练**: H100 引入 FP8,与模型协同
- **WGMMA**: Hopper 引入新的矩阵乘指令

### 28.3 软硬协同 Scaling 公式

$$\text{Throughput} \propto \text{Hardware FLOPS} \times \text{Algorithmic Efficiency} \times \text{Utilization}$$

三个因素都同等重要。

---

## 29. Scaling 实验的统计方法

### 29.1 多次训练取平均

```python
def train_with_seeds(config, n_seeds=3):
    """多次训练,减少随机性"""
    results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = train(config)
        results.append(evaluate(model))
    return np.mean(results), np.std(results)
```

### 29.2 Bootstrap 置信区间

```python
def bootstrap_ci(metrics, n_bootstrap=1000):
    """Bootstrap 估计置信区间"""
    bootstraps = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(metrics, size=len(metrics), replace=True)
        bootstraps.append(np.mean(sample))
    return np.percentile(bootstraps, [2.5, 97.5])
```

### 29.3 显著性检验

```python
from scipy.stats import ttest_ind

# 配对 t 检验: 模型 A vs 模型 B
t_stat, p_value = ttest_ind(scores_a, scores_b)
# p < 0.05 → 显著差异
```

---

## 30. Scaling 与人类智能

### 30.1 Scaling 的"极限"问题

**关键问题**: Scaling 是否能产生 AGI?

```text
乐观:
  - Loss 持续下降, 能力持续提升
  - 涌现能力证明"质变"可能
  - 数据 + 算力足够大, 可逼近人类

谨慎:
  - Loss 边际收益递减
  - 数据 / 算力终有上限
  - 某些能力可能根本不能 scaling
```

### 30.2 当前 Scaling 的"瓶颈"

1. **数据**: 高质量文本耗尽
2. **算力**: 训练成本超 $1B
3. **推理**: 推理成本爆炸
4. **评估**: 难以准确衡量能力
5. **理论**: Scaling Law 缺乏理论保证

### 30.3 后 Scaling 时代

**假设**: Scaling 边际收益递减后的策略

1. **算法革命**: 新架构(Mamba、Hyena 等)
2. **多模态融合**: 跨模态推理
3. **Agent 化**: 用工具 + 规划
4. **持续学习**: 在线学习 + 自适应
5. **人机协作**: LLM + 人类反馈

---

## 31. Scaling 的成本函数对比

### 31.1 训练成本函数

```python
def training_cost(model_size, data_tokens, gpu_hours_per_billion_tokens=10,
                  cost_per_gpu_hour=2.0):
    """训练成本估算 (USD)"""
    model_size_b = model_size / 1e9
    data_in_billion = data_tokens / 1e9
    total_gpu_hours = data_in_billion * gpu_hours_per_billion_tokens * (model_size_b ** 0.5)
    return total_gpu_hours * cost_per_gpu_hour

# LLaMA-2 70B
print(training_cost(70e9, 2e12))  # ~$20M

# GPT-4 (推测)
print(training_cost(1.8e12, 13e12))  # ~$200M+
```

### 31.2 推理成本函数

```python
def inference_cost_per_query(model_size, input_tokens, output_tokens,
                              cost_per_million_tokens=1.0):
    """每 query 推理成本 (USD)"""
    total_tokens = input_tokens + output_tokens
    return total_tokens / 1e6 * cost_per_million_tokens * (model_size / 70e9)
```

### 31.3 总拥有成本 (TCO)

$$\text{TCO} = \text{Train Cost} + \sum_{i=1}^{N_{\text{users}}} \text{Inference Cost}_i$$

一个 405B 模型训练 $300M,但 1 年推理可达 $1B+。

---

## 32. Scaling 与能源/环境

### 32.1 训练能源估算

```python
def training_energy_mwh(model_size, data_tokens, gpu_power_w=700):
    """训练能源 (MWh)"""
    gpu_hours = (model_size / 1e9) * (data_tokens / 1e12) * 1000
    energy_kwh = gpu_hours * gpu_power_w / 1000
    return energy_kwh / 1000  # MWh

# LLaMA-2 70B
print(training_energy_mwh(70e9, 2e12))  # ~2000 MWh
```

### 32.2 数据中心能源

| 数据中心 | 功率 | 冷却 |
| :--- | :--- | :--- |
| 单机柜 | 10-50 kW | 风冷 |
| 中型 DC | 10-50 MW | 风冷/液冷混合 |
| LLM 训练 DC | 50-300 MW | 液冷 |
| 超大型 DC | 300+ MW | 液冷/浸没 |

### 32.3 碳排放估算

```python
def carbon_footprint(energy_mwh, grid_carbon_intensity=0.5):
    """碳排放估算 (吨 CO2)"""
    energy_kwh = energy_mwh * 1000
    co2_kg = energy_kwh * grid_carbon_intensity
    return co2_kg / 1000

# LLaMA-2 70B
print(carbon_footprint(2000))  # ~1000 吨 CO2
```

### 32.4 可持续 Scaling 路径

1. **可再生能源**: 太阳能/风电数据中心
2. **高效架构**: 算法效率提升
3. **模型蒸馏**: 大模型 → 小模型
4. **算力共享**: 多租户复用
5. **碳追踪**: 训练时记录碳足迹

---

## 33. 实战案例库

### 33.1 案例 1:LLaMA-3 Scaling 全景

**目标**: 训练 405B 旗舰开源模型
**算力**: 16K H100 × 54 天
**数据**: 15.6T tokens(8x Chinchilla)
**关键决策**:
- 数据质量 > 数量 → 用 GPT-4 / Llama-2 过滤
- 128K 上下文 → RoPE base 调到 500K
- 训练稳定性 → QK-norm

### 33.2 案例 2:DeepSeek-V3 成本优化

**目标**: 671B MoE, 训练 $5.5M
**关键技术**:
- FP8 训练(无精度损失)
- MLA(KV Cache 压缩 1/16)
- DeepSeekMoE(细粒度 + 共享专家)
- DualPipe(通信计算 overlap)
- 多专家并行(EP)

### 33.3 案例 3:Phi-3 数据效率

**目标**: 3.8B 模型接近 70B 效果
**训练数据**: 3.3T 高质量 token
**关键决策**:
- 用 LLaMA-3 数据 + 合成数据
- 严苛质量过滤
- 多次 epoch 高质量数据

---

## 34. Scaling 实验 Checklist

### 34.1 起步阶段

```text
□ 明确目标 (训练 SOTA / 落地产品 / 研究)
□ 选定 base architecture (decoder-only / MoE)
□ 估算可用算力 (FLOPs)
□ 估算可用数据 (tokens)
□ 选定 Chinchilla / 反 Chinchilla 公式
□ 估算 (N, D) 配置
```

### 34.2 数据准备阶段

```text
□ 原始数据采集 (web, code, books, papers)
□ 数据清洗 (URL, format, language)
□ Exact + fuzzy + semantic 去重
□ 质量打分 (heuristic + RM)
□ Benchmark 污染检测
□ 配比实验 (ablation)
```

### 34.3 训练阶段

```text
□ 选定超参 (lr, batch, warmup)
□ 配 FlashAttention v2/v3
□ 配 ZeRO / FSDP
□ 配混合精度 (BF16 / FP8)
□ 监控: loss / grad_norm / tokens_per_sec
□ Checkpoint 策略 (每 N step)
□ Loss spike 自动回滚
□ Eval 跑 benchmark
```

### 34.4 后训练阶段

```text
□ SFT 数据准备 (50K+ 高质量)
□ RM 数据 + RM 训练
□ RLHF 算法选择 (DPO / PPO / GRPO)
□ 安全 + 价值对齐 (RLAIF)
□ 推理 benchmark + 用户评估
```

---

## 35. 关键 Scaling 公式汇总

### 35.1 核心公式速查

| 公式 | 含义 | 来源 |
| :--- | :--- | :--- |
| $L(N) = (N_c/N)^{\alpha_N}$ | 损失 vs 模型大小 | Kaplan |
| $D \approx 20N$ | 数据 vs 模型 | Chinchilla |
| $C \approx 6ND$ | 算力 vs 模型+数据 | 通用 |
| $N_{\text{opt}} \propto C^{0.45}$ | Compute-optimal 模型 | Chinchilla |
| $D_{\text{opt}} \propto C^{0.55}$ | Compute-optimal 数据 | Chinchilla |
| $\text{LR} \propto 1/\sqrt{D}$ | 最优学习率 | 经验 |
| $\text{Batch}_{\text{opt}} \propto C^{0.25}$ | 最优 batch | 经验 |

### 35.2 不同阶段的 Scaling 目标

| 阶段 | Scaling 目标 | 关键指标 |
| :--- | :--- | :--- |
| **小模型 (<1B)** | 单卡可训,快速迭代 | Loss, PPL |
| **中等模型 (1-10B)** | 多卡训练,通用能力 | MMLU, IFEval |
| **大模型 (10-100B)** | 千卡训练,前沿 | MT-Bench, Arena |
| **超大模型 (>100B)** | 万卡训练,AGI 探索 | 复杂推理, 长 context |

### 35.3 不同任务的 Scaling 策略

```text
通用对话:
  → 7-13B, 1-2T tokens, SFT + DPO
代码生成:
  → 7-34B, 1T+ tokens (代码重), 多 epoch
数学推理:
  → 7-70B, 1T+ tokens (数学重), RLVR
长文本:
  → 7-70B, 128K-1M context, RoPE 外推
多模态:
  → 7-34B, 图文对数据, VLM 训练
```

---

## 36. Scaling 与开源生态

### 36.1 主流开源训练框架

| 框架 | 特点 | 代表用户 |
| :--- | :--- | :--- |
| **HuggingFace Transformers** | 易用,生态全 | 通用 |
| **Megatron-LM** | 高效 TP/PP | NVIDIA |
| **DeepSpeed** | ZeRO + 优化 | Microsoft |
| **FSDP** | PyTorch 原生 | PyTorch |
| **ColossalAI** | 大模型优化 | HPC-AI Tech |
| **FastLM** | 快速训练 | (开源) |
| **LLaMA-Factory** | 一站式 SFT/DPO | 社区 |

### 36.2 主流推理引擎

| 引擎 | 特点 | 适用 |
| :--- | :--- | :--- |
| **vLLM** | PagedAttention, 高吞吐 | 生产级服务 |
| **TGI** | HuggingFace 官方 | HF 生态 |
| **TensorRT-LLM** | NVIDIA 优化,极致性能 | 低延迟 |
| **SGLang** | RadixAttention | 多轮对话 |
| **LMDeploy** | 中文优化 | 中文 LLM |
| **llama.cpp** | CPU/边缘 | 本地部署 |
| **MLC-LLM** | 移动端 | 端侧 |

### 36.3 主流数据集

| 数据集 | 规模 | 类型 |
| :--- | :--- | :--- |
| **CommonCrawl** | 数十 TB | Web |
| **RedPajama** | 1.2T tokens | 复制 LLaMA 训练集 |
| **The Pile** | 800 GB | 多领域 |
| **SlimPajama** | 600B tokens | 精炼版 |
| **FineWeb** | 15T tokens (1.5T 精) | 高质量 web |
| **Cosmopedia** | 25B tokens | 合成教科书 |
| **OpenWebMath** | 14.7B tokens | 数学 web |

---

## 37. Scaling 与商业化

### 37.1 Scaling 与产品迭代

```text
研究阶段 (POC):
  → 1-7B 模型, 1-10 GPU, 1-2 周训练
  → 验证 scaling 思路

MVP 阶段:
  → 7-13B 模型, 100 GPU, 1 月训练
  → 部署推理服务

规模阶段:
  → 70-200B 模型, 1000 GPU, 3 月训练
  → 商业化落地

旗舰阶段:
  → 400B+ 模型, 16K GPU, 6 月训练
  → 行业领先
```

### 37.2 Scaling 与成本/收益

```python
def roi_analysis(model_size, training_cost, expected_revenue, lifespan_years=3):
    """Scaling ROI 分析"""
    total_cost = training_cost + (expected_revenue * 0.3) * lifespan_years  # 推理成本
    roi = (expected_revenue * lifespan_years - total_cost) / total_cost
    return roi

# 例: 训练 $50M, 预期 3 年收入 $200M
print(roi_analysis(70e9, 50e6, 200e6))  # ~150% ROI
```

### 37.3 Scaling 与竞争壁垒

Scaling 形成的壁垒:
1. **数据壁垒**: 高质量数据私有化
2. **算力壁垒**: 大集群 + 长训练时间
3. **工程壁垒**: 分布式训练经验
4. **人才壁垒**: 顶级 ML 研究员

---

## 38. 总结

Scaling 是大模型过去十年最核心的"魔法"。本章核心要点:

1. **Kaplan → Chinchilla**: 模型与数据应同比例扩展 (D ≈ 20N)
2. **反 Chinchilla**: 数据质量提升,数据可以"过度"
3. **涌现能力**: 某些能力在阈值后突然出现,但可能源于评估指标
4. **多轴 Scaling**: Pre-train + Post-train + Test-time 三轴协同
5. **Test-time Scaling**: o1/R1 范式,推理时算力换能力
6. **分布式 Scaling**: 3D Parallelism + ZeRO + 通信优化
7. **成本挑战**: 数据耗尽、推理成本爆炸、能源约束
8. **未来**: 算法效率 + 多模态 + 持续学习

> **核心心法**: Scaling 是科学(数学公式),也是艺术(数据质量、配比、训练细节)。**没有万能公式,只有不断实验**。

---

> **版本**: v1.0 (2026-08-31) · **维护**: 自维护
>
> *Bigger models, more data, more compute. But also: better data, better algorithms, better tests.*

---

## 39. 关键 Scaling 数字速记

### 39.1 模型规模 vs 算力

| 模型规模 | 训练算力 (FLOPs) | GPU 数 | 训练时间 | 成本 (USD) |
| :--- | :--- | :--- | :--- | :--- |
| 1B | ~1e20 | 32 A100 | 1 天 | ~$1K |
| 7B | ~6e22 | 256 A100 | 7 天 | ~$1M |
| 13B | ~1e23 | 512 A100 | 14 天 | ~$3M |
| 70B | ~1e24 | 1000 A100 | 30 天 | ~$20M |
| 175B | ~3e24 | 2000 A100 | 60 天 | ~$50M |
| 405B | ~5e25 | 16K H100 | 54 天 | ~$300M |
| 1T+ | ~1e26 | 50K+ H100 | 100 天 | ~$1B+ |

### 39.2 推理吞吐量速记

| 模型 | 单卡吞吐 (tokens/s) | 8 卡吞吐 | 推理延迟 (ms/token) |
| :--- | :--- | :--- | :--- |
| 7B (BF16) | ~100 | ~700 | ~10 |
| 7B (INT4) | ~300 | ~2200 | ~3 |
| 70B (BF16, TP=4) | ~30 | ~200 | ~50 |
| 405B (BF16, TP=8) | ~10 | ~70 | ~150 |
| 405B (FP8, TP=8) | ~20 | ~140 | ~75 |

### 39.3 显存占用速记

| 模型 | FP32 | FP16/BF16 | INT8 | INT4 |
| :--- | :--- | :--- | :--- | :--- |
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |
| 405B | 1620 GB | 810 GB | 405 GB | 203 GB |

### 39.4 训练 token 速记

| 模型 | Chinchilla 推荐 | 实际训练 | 倍数 |
| :--- | :--- | :--- | :--- |
| GPT-3 | 3.5T | 0.3T | 0.09x (欠训) |
| LLaMA-1 65B | 1.3T | 1.4T | 1.1x (最优) |
| LLaMA-2 70B | 1.4T | 2T | 1.4x (略过) |
| LLaMA-3 405B | 8.1T | 15.6T | 1.9x (过训) |

---

## 40. 一句话心法

**Scaling 是经验科学,不是理论科学**。我们仍在摸索"多少数据 × 多少模型 × 多少算力 → 多少能力"的最佳组合,但核心规律已清晰:

1. **数据 > 模型 > 算力**: 同样的 FLOPs,花在数据上比花在模型上更划算
2. **质量 > 数量**: 高质量数据多次 epoch > 低质量单 epoch
3. **三轴协同**: Pre-train + Post-train + Test-time,缺一不可
4. **持续迭代**: Scaling 永远在进行中,模型 v1 出来后,v2 总能做得更好

> *Bigger isn't always better, but always bigger is a good start.*

---

## 41. Scaling 演进年表(完整版)

```text
2017 - Transformer (Vaswani et al., 65M)
      Scaling Law 思想萌芽

2018 - GPT-1 (117M) / BERT (340M)
      Pre-train + Fine-tune 范式确立

2019 - GPT-2 (1.5B) / T5 (11B)
      Scaling 首次系统化研究

2020 - GPT-3 (175B, 300B tokens)
      Kaplan Scaling Law 论文
      In-Context Learning 涌现

2021 - GLaM / Gopher / Switch Transformer
      MoE 大规模工程化

2022 - Chinchilla (70B, 1.4T tokens)
      Emergent Abilities 论文
      RLHF 工业化 (InstructGPT)

2023 - LLaMA-1 (65B, 1.4T tokens)
      ChatGPT 引爆 RLHF
      DPO 论文 / Constitutional AI
      LLaMA-2 (70B) + RLHF

2024 - LLaMA-3 (405B, 15.6T tokens)
      DeepSeek-V2 / V3 ($5.5M 训练成本)
      Gemini 1.5 (1M 上下文)
      o1 (Test-time Scaling 开端)
      Self-Rewarding / Online RL

2025 - DeepSeek-R1 (开源 o1 复现)
      LLaMA-4 (MoE 原生旗舰)
      Reasoning Scaling 全面铺开
      Multi-modal + Test-time + Post-train 三轴协同

2026+ - 推测:
      - 1T+ 总参 MoE 模型成为旗舰标配
      - Test-time Scaling 推理任务全面铺开
      - 数据合成占训练数据 30-50%
      - 长 CoT 训练成为推理模型标配
      - 多模态原生统一 (text/image/audio/video)
```

---

## 42. Scaling 的"反常识"洞察

### 42.1 反直觉发现 1:数据 > 模型

同等算力下,把算力花在"更多数据 + 较小模型"上,比"较少数据 + 较大模型"更划算(Chinchilla 反直觉)。

### 42.2 反直觉发现 2:小模型 + 强数据 > 大模型 + 弱数据

Phi-3 3.8B 用 3.3T 高质量数据 ≈ Llama-3 70B 用 15T 中等数据(MMLU ~70)。

### 42.3 反直觉发现 3:涌现是"幻觉"

用 strict 指标(exact match)看似涌现,用 loose 指标(token log-prob)是平滑。涌现是评估方式造成的。

### 42.4 反直觉发现 4:训练 > 推理

RLHF 后,模型 inference 比 base model 慢,但用户感知"更聪明"。**对齐 ≠ 智力提升**。

### 42.5 反直觉发现 5:Test-time Scaling 部分替代 Pre-train Scaling

推理时算力增加 ~100x,某些推理任务可接近训练算力增加 ~100x 的效果。

### 42.6 反直觉发现 6:MoE 比 Dense 算力效率高

DeepSeek-V3 671B MoE 激活 37B,效果 ≈ 200B+ Dense,训练成本 1/5。

---

## 43. 推荐的 Scaling 实验顺序

```text
Step 1: 小模型快速验证 (1B, 100M tokens, 1 GPU)
  - 验证数据 pipeline
  - 验证 loss 能下降
  - 验证训练稳定性

Step 2: 中等模型 (7B, 100B tokens, 32 GPU)
  - 验证 scaling 方向
  - 验证混合精度 / FlashAttention
  - 验证超参 (lr, batch, warmup)

Step 3: 数据 scaling (固定 7B, 1T+ tokens)
  - 验证数据质量的影响
  - 验证数据配比

Step 4: 中等模型全规模 (7B, 1-2T tokens)
  - 训练最终模型
  - Eval 各种 benchmark

Step 5: (可选) 大模型 (70B+, 千卡)
  - 验证大模型 scaling
  - 验证多模态 / 长 context
  - 最终发布
```

---

> **版本**: v1.0 (2026-08-31) · **维护**: 自维护
>
> *Scaling is the art of finding the right amount of everything — data, parameters, compute, and patience.*
