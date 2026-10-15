---
title: 7-LLM vs GenRec RLHF与RL对齐对比
tags:
  - LLM
  - GenRec
  - RLHF
  - GRPO
  - DPO
  - PPO
  - GPR
  - OxygenREC
  - OneRec
  - Reward
  - 对齐
  - 对比
description: 系统对比 LLM(RLHF / DPO / GRPO / PPO)与 GenRec(GPR / OxygenREC / OneRec-Think / Bandit)在 RL 对齐方法上的核心差异
last_updated: 2026-07-25
status: ✅ v1.0 - RLHF与RL对齐对比
---

# 7-LLM vs GenRec RLHF与RL对齐对比

> [!info] **本章定位**: 在 [[4-Token化与词表对比]]、`5-训练目标与损失对比` 和 [[6-评估指标对比]] 之后,本章进入 LLM 与 GenRec 的"对齐"环节。LLM 用 RLHF/DPO/GRPO 对齐人类偏好,GenRec 用 DPO/GBPO/GRPO 对齐业务指标。注:本章为系列核心章节之一,后续章节(8-19)继续展开推理、部署、Scaling、工业案例等主题。
>
> [!tip] **阅读建议**: 必须先读 `../RL/3-RL在LLM中的经典方法.md` 与 `../RL/4-RL在GenRec中的经典方法.md`。本章是它们的对比。
>
> [!info] **与本系列其他章节的关系**: 上一篇 [[6-评估指标对比]];后续 [[8-推理与部署对比]] / [[9-Scaling与扩展对比]] / [[10-工业案例对比]] / [[11-融合与未来方向]] / `12-19 扩展章节`。

---

## 目录

- [0. 一句话总结](#0-一句话总结)
- [1. LLM RLHF](#1-llm-rlhf)
  - [1.1 InstructGPT 三阶段](#11-instructgpt-三阶段)
  - [1.2 PPO for LLM](#12-ppo-for-llm)
  - [1.3 DPO 与变体](#13-dpo-与变体-ipo--kto--simpo--orpo)
  - [1.4 GRPO (DeepSeek)](#14-grpo-deepseek)
  - [1.5 RLAIF / Constitutional AI](#15-rlaif--constitutional-ai)
  - [1.6 Self-Rewarding](#16-self-rewarding)
  - [1.7 RLVR (Verifiable Rewards)](#17-rlvr-verifiable-rewards)
  - [1.8 推理增强 (R1, o1)](#18-推理增强-r1-o1)
- [2. GenRec RL](#2-genrec-rl)
  - [2.1 OneRec V1 (DPO)](#21-onerec-v1-dpo)
  - [2.2 OneRec V2 (GBPO)](#22-onerec-v2-gbpo)
  - [2.3 OneRec-Think (GRPO + Rollout-Beam)](#23-onerec-think-grpo--rollout-beam)
  - [2.4 OneRec-Foundation](#24-onerec-foundation-general-distillation--rec-rl)
  - [2.5 GPR HEPO](#25-gpr-hepo)
  - [2.6 OxygenREC SA-GCPO](#26-oxygenrec-sa-gcpo)
  - [2.7 OneMall GRPO / GSPO](#27-onemall-grpo--gspo)
- [3. Reward Model 对比](#3-reward-model-对比)
- [4. Critic Model 对比](#4-critic-model-对比)
- [5. KL 约束对比](#5-kl-约束对比)
- [6. 探索 vs 利用对比](#6-探索-vs-利用对比)
- [7. 在线 vs 离线 RL 对比](#7-在线-vs-离线-rl-对比)
- [8. 奖励黑客对比](#8-奖励黑客对比)
- [9. 计算成本对比](#9-计算成本对比)
- [10. 实际效果对比](#10-实际效果对比)
- [11. 未来融合方向](#11-未来融合方向)
- [12. 深度专题](#12-深度专题)
- [附录 A · 速查表](#附录-a--速查表)
- [附录 B · 参考与链接](#附录-b--参考与链接)

---

## 0. 一句话总结

> **LLM 用 RLHF/DPO/GRPO 对齐"人类偏好",GenRec 用 DPO/GBPO/GRPO 对齐"业务指标"**;两者在 Reward 来源、Critic 模型、KL 约束、在线 vs 离线 RL 上呈现"本质同构 + 应用差异"的格局,RL 已成两边对齐事实标准的统一接口。

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| Reward 来源 | 人类偏好 / RM / RLAIF | 业务指标(CTR/CVR/GMV) |
| Critic 模型 | Value Model(必) | Group Baseline(GRPO) |
| 主要算法 | PPO / DPO / GRPO | GBPO / GRPO / DPO |
| KL 约束 | 必(KL vs ref) | 必(KL vs ref) |
| 在线 RL | 几乎不用 | 部分使用 |
| 训练成本 | 高(4 模型) | 中(2 模型) |
| 业务对齐 | 弱 → 强(RLHF) | 直接 |

下面正式进入 11 节正文 + 1 节深度专题。

---

## 1. LLM RLHF

LLM 的 RLHF 从 InstructGPT(2022)开始,经过 ChatGPT、Claude、Llama 2、Qwen、DeepSeek 等多代演进,已形成完整技术栈。

### 1.1 InstructGPT 三阶段

#### 1.1.1 三阶段流程

```text
Stage 1: SFT (Supervised Fine-Tuning)
  输入: prompt → 人工撰写的高质量回答
  目标: 学习指令遵循

Stage 2: RM (Reward Model)
  输入: prompt + 两个回答 (y_w, y_l)
  目标: 训练一个分数模型 $r_\phi(prompt, y)$

Stage 3: PPO (Proximal Policy Optimization)
  输入: prompt, 输出: 回答
  目标: 最大化 $r_\phi$ 同时保持与 SFT 模型接近
```

#### 1.1.2 三阶段的核心组件

```python
# Stage 1: SFT
policy_sft = train_sft(base_model, sft_dataset)

# Stage 2: RM
reward_model = train_reward_model(
    base_model_or_sft,
    preference_dataset,  # {(prompt, y_w, y_l)}
)

# Stage 3: PPO
policy_ppo = ppo_train(
    policy=policy_sft,         # 当前策略
    ref_policy=policy_sft,     # 参考策略(冻结)
    reward_model=reward_model,  # 奖励模型(冻结)
    value_model=value_model,   # 价值模型
    prompts=prompts,
)
```

#### 1.1.3 三阶段的优劣

| 阶段 | 优点 | 缺点 |
| :--- | :--- | :--- |
| SFT | 简单稳定 | 仅模仿,无偏好 |
| RM | 捕获偏好 | 需要标注 |
| PPO | 直接对齐奖励 | 训练不稳定,4 模型 |

### 1.2 PPO for LLM

#### 1.2.1 PPO 的目标函数

PPO 的 clipped objective:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \left[ \min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

其中:

$$
\rho_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

$\hat{A}_t$ 是 advantage,通常用 GAE 估计。

#### 1.2.2 PPO 在 LLM 中的完整目标

$$
\mathcal{L}_{\text{PPO-LLM}} = \mathcal{L}_{\text{clip}} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) - \lambda \cdot r_\phi(\text{prompt}, y)
$$

> **三项**:
> - $\mathcal{L}_{\text{clip}}$: PPO clip
> - $\beta \cdot \text{KL}$: 防止策略偏离太远
> - $-\lambda \cdot r_\phi$: 最大化奖励

#### 1.2.3 4 个模型角色

| 模型 | 角色 | 训练状态 |
| :--- | :--- | :--- |
| Policy ($\pi_\theta$) | 演员 | 训练 |
| Ref ($\pi_{\text{ref}}$) | 参考 | 冻结 |
| Value ($V_\phi$) | 评论家 | 训练 |
| Reward ($r_\phi$) | 评分 | 冻结 |

#### 1.2.4 PPO 的工程挑战

- **显存**: 4 个模型 × policy 大小 = 极大
- **不稳定**: 训练 reward 可能崩
- **样本效率**: 需要大量 rollout
- **超参敏感**: KL 系数、clip 范围

#### 1.2.5 PyTorch 伪代码 (PPO for LLM)

```python
def ppo_step(policy, ref_policy, value_model, reward_model, prompts, gen_kwargs):
    # 1. 用当前 policy 生成回答
    responses = policy.generate(prompts, **gen_kwargs)

    # 2. 算 log probs
    with torch.no_grad():
        policy_logprobs = policy.compute_logprobs(prompts, responses)
        ref_logprobs = ref_policy.compute_logprobs(prompts, responses)

    # 3. 算 rewards
    rewards = reward_model.score(prompts, responses)

    # 4. 算 values
    values = value_model(prompts, responses)

    # 5. 算 GAE
    advantages = compute_gae(rewards, values, ...)
    returns = advantages + values

    # 6. PPO update
    for epoch in range(ppo_epochs):
        new_logprobs = policy.compute_logprobs(prompts, responses)
        ratio = (new_logprobs - policy_logprobs).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip, 1 + clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty
        kl = (new_logprobs - ref_logprobs).mean()
        kl_loss = beta * kl

        # Total loss
        loss = policy_loss + kl_loss - lambda_ * rewards.mean()
        loss.backward()
        optimizer.step()
```

### 1.3 DPO 与变体 (IPO / KTO / SimPO / ORPO)

#### 1.3.1 DPO (Direct Preference Optimization)

DPO 直接对 (y_w, y_l) 对优化,不需要显式 reward model:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
$$

> **核心**: 偏好对 → 隐式 reward → 直接优化。

#### 1.3.2 DPO 推导

DPO 起源于 Bradley-Terry 模型:

$$
P(y_w > y_l) = \sigma(r(y_w) - r(y_l))
$$

其中 $r(y) = \beta \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$ 是隐式奖励。

#### 1.3.3 DPO 的 PyTorch 实现

```python
def dpo_loss(policy_logprobs_w, policy_logprobs_l,
             ref_logprobs_w, ref_logprobs_l, beta=0.1):
    """
    policy_*: 当前策略对 (y_w, y_l) 的 log probs
    ref_*: 参考策略对 (y_w, y_l) 的 log probs
    """
    pi_logratios = policy_logprobs_w - policy_logprobs_l
    ref_logratios = ref_logprobs_w - ref_logprobs_l

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

#### 1.3.4 DPO 变体

| 变体 | 关键改进 | 提出者 |
| :--- | :--- | :--- |
| **DPO** | 原始 | Stanford 2023 |
| **IPO** | Identity Preference Optimization, 防止 overfit | Meta 2024 |
| **KTO** | Kahneman-Tversky, 用单项数据 | Meta 2024 |
| **SimPO** | 简单偏好, length normalized | Stanford 2024 |
| **ORPO** | Odds Ratio, 无需 ref model | 2024 |

#### 1.3.5 IPO (Identity Preference Optimization)

$$
\mathcal{L}_{\text{IPO}} = (\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2\beta} \tau)^2
$$

> **关键**:用 MSE 替代 log sigmoid,防止过拟合。

#### 1.3.6 KTO (Kahneman-Tversky Optimization)

$$
\mathcal{L}_{\text{KTO}} = -\lambda_w \sigma(\beta \Delta_w) + \lambda_l \sigma(\beta \Delta_l)
$$

> **关键**:用前景理论(损失厌恶),不需要成对数据。

#### 1.3.7 SimPO (Simple Preference Optimization)

$$
\mathcal{L}_{\text{SimPO}} = -\log \sigma(\beta \cdot \frac{\log \pi_\theta(y_w)}{|y_w|} - \beta \cdot \frac{\log \pi_\theta(y_l)}{|y_l|} - \gamma)
$$

> **关键**:长度归一化,简单高效。

#### 1.3.8 ORPO (Odds Ratio Preference Optimization)

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

其中 $\mathcal{L}_{\text{OR}} = -\log \sigma(\log \frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)})$。

> **关键**:无需 ref model,无需 SFT 阶段。

### 1.4 GRPO (DeepSeek)

#### 1.4.1 GRPO 的核心思想

GRPO (Group Relative Policy Optimization) 是 DeepSeek-R1 提出的方法,**不需要 critic 模型**。

#### 1.4.2 GRPO 算法

对每个 prompt $x$:

1. 采样 $G$ 个回答 $\{y_1, \dots, y_G\}$
2. 计算每个回答的 reward $r_i$
3. 计算 group baseline: $\bar{r} = \frac{1}{G} \sum r_i$
4. Advantage: $\hat{A}_i = (r_i - \bar{r}) / \sigma_r$
5. 优化:

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ \frac{1}{G} \sum_i \min(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i) + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

#### 1.4.3 GRPO 与 PPO 的对比

| 维度 | PPO | GRPO |
| :--- | :--- | :--- |
| Critic 模型 | 必(V 模型) | 不需要 |
| Advantage 估计 | GAE | Group baseline |
| 显存 | 大(4 模型) | 中(2 模型) |
| 训练稳定性 | 中 | 高 |
| 适用 | 任意偏好 | 可验证 reward |

#### 1.4.4 GRPO 的 PyTorch 实现

```python
def grpo_loss(policy, ref_policy, prompt, G=8, beta=0.04, clip=0.2):
    # 1. 采样 G 个回答
    responses = [policy.generate(prompt) for _ in range(G)]

    # 2. 算 rewards
    rewards = torch.tensor([score_fn(r) for r in responses])

    # 3. 算 advantage
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. 算 ratio
    policy_logprobs = torch.stack([policy.compute_logprobs(prompt, r) for r in responses])
    ref_logprobs = torch.stack([ref_policy.compute_logprobs(prompt, r) for r in responses])
    ratio = (policy_logprobs - ref_logprobs).exp()

    # 5. PPO loss
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip, 1 + clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 6. KL penalty
    kl = (policy_logprobs - ref_logprobs).mean()
    total_loss = policy_loss + beta * kl
    return total_loss
```

#### 1.4.5 GRPO 的优势

- **无需 critic**:节省 50% 显存
- **group baseline**:天然稳定
- **可验证 reward**:数学 / 代码任务非常合适

#### 1.4.6 GRPO 的局限

- **每 prompt 需 G 个样本**:计算成本 × G
- **同 prompt 反复采样**:训练较慢
- **优势方差大**:极端回答可能主导

### 1.5 RLAIF / Constitutional AI

#### 1.5.1 RLAIF (RL from AI Feedback)

**核心**:用 LLM 自己当 reward model,代替人类标注。

```text
Step 1: prompt → 多个回答
Step 2: LLM judge 选最好的
Step 3: 用选好的回答训练 PPO
```

#### 1.5.2 Constitutional AI (Anthropic)

**核心**:用一组"宪法原则"约束 LLM 行为。

```text
原则示例:
  - 请选择最有帮助、最诚实、最无害的回答
  - 请避免种族、性别歧视
  - 请提供事实信息,不编造

训练:
  Step 1: LLM 生成回答 A
  Step 2: 用宪法原则评估 A
  Step 3: 修改 A 以符合原则,得到 B
  Step 4: 用 (B, A) 对训练 DPO
```

#### 1.5.3 RLAIF 的优劣

| 优点 | 缺点 |
| :--- | :--- |
| 节省标注成本 | LLM judge 有偏 |
| 规模可扩展 | 错误累积 |
| 实时可用 | 难解释 |

### 1.6 Self-Rewarding

#### 1.6.1 核心思想

LLM 既生成回答,又评估自己的回答。

```text
Step 1: LLM 生成多个候选回答
Step 2: LLM 评分每个回答
Step 3: 选最高分
Step 4: 用最高分回答训练
```

#### 1.6.2 Self-Rewarding 的训练

```python
def self_rewarding_step(llm, prompt):
    # 1. 生成多个回答
    candidates = [llm.generate(prompt) for _ in range(K)]

    # 2. 自我评分
    scores = [llm.evaluate(prompt, c) for c in candidates]

    # 3. 选最好
    best_idx = scores.index(max(scores))
    best_response = candidates[best_idx]

    # 4. 用 best 训练
    loss = -llm.compute_logprobs(prompt, best_response).mean()
    return loss
```

### 1.7 RLVR (Verifiable Rewards)

#### 1.7.1 核心思想

对于**可验证**的任务(数学、代码),reward 不需要 RM,直接用规则:

- 数学:答对 = 1, 答错 = 0
- 代码:pass 测试 = 1, fail = 0

#### 1.7.2 RLVR vs RLHF

| 维度 | RLHF | RLVR |
| :--- | :--- | :--- |
| Reward 来源 | 人类偏好 | 规则验证 |
| 训练成本 | 高(标注) | 低(规则) |
| 适用 | 开放任务 | 可验证任务 |
| 信号噪声 | 中(主观) | 低(客观) |

#### 1.7.3 RLVR 在 DeepSeek-R1 中的应用

```text
数学任务:
  - 用规则验证最终答案
  - GRPO 训练
  - 涌现 "aha moment"(模型自发学会长推理)

代码任务:
  - 用测试用例验证
  - GRPO 训练
```

### 1.8 推理增强 (R1, o1)

#### 1.8.1 o1 系列

OpenAI o1(2024.09)首次把"CoT 推理"作为可独立优化目标。

#### 1.8.2 训练范式

```text
Stage 1: SFT (普通指令微调)
Stage 2: RLVR (用 GRPO + 规则 reward)
   - 训练模型"长思考"
   - 奖励"正确 + 推理链好"
Stage 3: Self-play / Iterative refinement
```

#### 1.8.3 推理增强的代价

| 模型 | GSM8K | 推理 token 数 | 延迟 |
| :--- | :---: | :---: | :---: |
| GPT-4o | 92 | ~200 | 1x |
| o1-mini | 93 | ~1000 | 5x |
| o1 | 96 | ~5000 | 30x |
| o3 | 97 | ~20000 | 100x |

> **关键**:推理增强 = 训练成本 + 推理成本双升。

---

## 2. GenRec RL

GenRec 的 RL 起步晚但进展快。阿里 OneRec V1/V2/Think/Foundation 形成完整技术栈。

### 2.1 OneRec V1 (DPO)

#### 2.1.1 核心思想

OneRec V1(2025.04)是阿里第一个端到端 RL 生成式推荐,用 DPO 训练。

#### 2.1.2 偏好对构造

```text
正样本: 用户实际点击 / 购买的物品
负样本: 曝光未点击 / 类似物品

偏好对: (history + 偏好上下文, positive_item, negative_item)
```

#### 2.1.3 DPO 训练

$$
\mathcal{L}_{\text{OneRec-V1}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(i_+ | h)}{\pi_{\text{ref}}(i_+ | h)} - \beta \log \frac{\pi_\theta(i_- | h)}{\pi_{\text{ref}}(i_- | h)} \right)
$$

#### 2.1.4 OneRec V1 DPO 的工程细节

- **正样本**:用户点击序列中真实点击的物品
- **负样本**:曝光未点击的物品(用户实际看到但没点)
- **reference**:SFT 后的模型
- **β**:0.1 – 1.0 之间

#### 2.1.5 OneRec V1 DPO 的效果

```text
离线指标:
  Recall@10:    +12%
  NDCG@10:      +15%
  Hit@10:       +10%

在线指标:
  CTR:          +5.2%
  GMV:          +3.8%
  长尾 Hit@10:  +18%
```

### 2.2 OneRec V2 (GBPO)

#### 2.2.1 GBPO (Group-Based Preference Optimization)

OneRec V2(2025.05)提出 **GBPO**:Group-Based Preference Optimization,核心改进是用"群体偏好"而非"个体偏好"。

#### 2.2.2 群体偏好构造

```text
对每个用户:
  生成 G 个候选物品列表
  按业务指标排序: best_1, ..., best_G
  构造偏好对: (best_1 > best_G), (best_1 > best_{G-1}), ...

GBPO 损失:
  L_GBPO = -log σ(β * Δ_θ(y_best, y_worst))
```

#### 2.2.3 GBPO 与 DPO 的区别

| 维度 | DPO | GBPO |
| :--- | :--- | :--- |
| 偏好构造 | 个体偏好对 | 群体偏好对 |
| 信号强度 | 中 | 高 |
| 偏好来源 | 用户行为 | 业务指标 |
| 适合 | 单步推荐 | 列表级推荐 |

#### 2.2.4 GBPO 的 PyTorch 实现

```python
def gbpo_loss(policy, ref_policy, history, candidate_items, business_rewards, beta=0.1):
    """
    candidate_items: (B, G) G 个候选
    business_rewards: (B, G) 每个候选的业务指标
    """
    # 1. 选最好和最差
    best_idx = business_rewards.argmax(dim=-1)
    worst_idx = business_rewards.argmin(dim=-1)

    best_items = candidate_items.gather(1, best_idx.unsqueeze(1))
    worst_items = candidate_items.gather(1, worst_idx.unsqueeze(1))

    # 2. 算 log probs
    best_logprobs = policy.compute_logprobs(history, best_items)
    worst_logprobs = policy.compute_logprobs(history, worst_items)

    with torch.no_grad():
        best_ref = ref_policy.compute_logprobs(history, best_items)
        worst_ref = ref_policy.compute_logprobs(history, worst_items)

    # 3. DPO loss
    pi_logratios = best_logprobs - worst_logprobs
    ref_logratios = best_ref - worst_ref
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

#### 2.2.5 OneRec V2 GBPO 的效果

```text
离线指标 (相比 V1):
  Recall@10:    +5%
  NDCG@10:      +6%

在线指标:
  CTR:          +1.8%
  GMV:          +1.5%
  RPM:          +2.2%
```

### 2.3 OneRec-Think (GRPO + Rollout-Beam)

#### 2.3.1 核心思想

OneRec-Think(2025.06)把 LLM 的 GRPO 思路搬到 GenRec,**让推荐模型学会"思考"**(类似 o1 思路)。

#### 2.3.2 Rollout-Beam

```text
传统 Beam Search:
  - 每步选 top-K
  - 确定性, 无探索

Rollout-Beam (OneRec-Think):
  - 每步选 top-K
  - 但 K 路并行 rollout(每个假设延展 G 步)
  - 选 rollout 后 NDCG 最高的
  - 类似 MCTS
```

#### 2.3.3 GRPO 在 OneRec-Think 中的应用

```text
Step 1: 用户历史 → K 个候选列表
Step 2: rollout 评估每个候选列表(用 NDCG proxy)
Step 3: 算 group advantage
Step 4: GRPO 优化
```

#### 2.3.4 OneRec-Think 的效果

```text
离线指标 (相比 V2):
  Recall@10:    +3%
  NDCG@10:      +4%

在线指标:
  长尾覆盖率:    +8%
  新颖性:        +5%
```

### 2.4 OneRec-Foundation (General Distillation + Rec-RL)

#### 2.4.1 核心思想

OneRec-Foundation(2025.07)是 OneRec 系列最新进展,核心是**统一多源知识** + **Rec-RL**。

#### 2.4.2 General Distillation

```text
Teacher 模型:
  - 文本理解 (Qwen)
  - 协同过滤 (DIN)
  - 多任务 (MMoE)

Distill 方式:
  - 软标签蒸馏
  - Embedding 蒸馏
  - ListNet 蒸馏

Student: 单一 OneRec-Foundation 模型
```

#### 2.4.3 Rec-RL

Rec-RL = 把业务指标(CTR/GMV)作为 reward,用 GRPO 训练。

```text
Step 1: OneRec-Foundation 生成候选列表
Step 2: 业务指标评估 (CTR / RPM / GMV)
Step 3: 构造 reward
Step 4: GRPO 优化
```

#### 2.4.4 OneRec-Foundation 的整体流程

```text
Pretrain (SFT):
  - 大规模用户行为
  - NIP loss

Distill (General Distillation):
  - Teacher 模型软标签
  - 多任务蒸馏

RL (Rec-RL):
  - 业务指标 reward
  - GRPO 训练

Inference:
  - SID 自回归生成
  - TRIE 约束解码
  - 业务规则重排
```

### 2.5 GPR HEPO

#### 2.5.1 GPR (Generative Preference Recommender)

GPR(2024)是早期 GenRec + RL 探索之一。

#### 2.5.2 HEPO (Hybrid Exploration Policy Optimization)

```text
核心: 在生成阶段加入探索
  - ε-greedy: 偶尔采样非 top-1
  - entropy bonus: 鼓励多样性

训练:
  - DPO loss + 探索 loss
```

#### 2.5.3 GPR 的优势

- 探索 vs 利用平衡
- 长尾覆盖好
- 用户兴趣更广

### 2.6 OxygenREC SA-GCPO

#### 2.6.1 OxygenREC 背景

OxygenREC(2025)是字节的 GenRec 工作,见 `../GR/第09章-规模化排序与新架构/deep-dives/20-OxygenREC-深度解析.md`。

#### 2.6.2 SA-GCPO (Self-Adaptive Group Constrained Policy Optimization)

```text
核心:
  - 用 group 偏好
  - 自适应 KL 约束
  - Self-adaptive
```

#### 2.6.3 与 OneRec GBPO 的对比

| 维度 | OneRec GBPO | OxygenREC SA-GCPO |
| :--- | :--- | :--- |
| KL 约束 | 固定 β | 自适应 |
| 群体构造 | 业务奖励排序 | 多策略融合 |
| 工程复杂度 | 中 | 高 |

### 2.7 OneMall GRPO / GSPO

#### 2.7.1 OneMall GRPO

字节电商的 GRPO 实现:

```text
Step 1: 每个 prompt 生成 G 个推荐列表
Step 2: 业务指标 (CTR + GMV) 评分
Step 3: 算 group advantage
Step 4: 训练 policy
```

#### 2.7.2 GSPO (Group Sequence Policy Optimization)

字节 2025 的改进版本:

```text
改进:
  - 序列级 reward (考虑列表内部序)
  - 更稳定的 group baseline
  - 与推荐场景深度适配
```

#### 2.7.3 GSPO vs GRPO

| 维度 | GRPO | GSPO |
| :--- | :--- | :--- |
| Reward 类型 | 单回答 | 序列级 |
| Group baseline | 标量 | 序列 |
| 适合 | 短生成 | 长列表生成 |

---

## 3. Reward Model 对比

### 3.1 LLM: Bradley-Terry 偏好对

LLM 的 RM 几乎都是 Bradley-Terry 形式:

$$
P(y_w > y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

训练损失:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

### 3.2 GenRec: 业务指标 (CTR/GMV)

GenRec 的"奖励"直接是业务指标:

| 业务指标 | 范围 | 噪声 |
| :--- | :--- | :---: |
| CTR | [0, 1] | 中 |
| CVR | [0, 1] | 大 |
| GMV | [0, ∞) | 大 |
| RPM | [0, ∞) | 大 |

### 3.3 RM 来源对比

| 来源 | LLM | GenRec |
| :--- | :--- | :--- |
| 人类标注 | ✅(主流) | ❌ |
| AI 标注 (RLAIF) | ✅ | ❌ |
| 业务指标 | ❌ | ✅ |
| 规则验证 (RLVR) | ✅(数学/代码) | ✅(可达性/库存) |

### 3.4 RM 训练的数学对比

LLM RM:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

GenRec "RM" (用 reward head):

$$
r_\phi(h, i) = \text{BCE}(\text{MLP}(\phi(h, i)), \text{click})
$$

> GenRec 的"RM"实际上**直接是一个 head**,不是独立模型。

### 3.5 RM 的"维度诅咒"

LLM RM 输出维度:1(标量分数)
GenRec RM 输出维度:1(CTR / CVR)
LLM GRPO Group reward: G 个
GenRec GRPO Group reward: G 个列表

---

## 4. Critic Model 对比

### 4.1 LLM: Value Model

PPO 需要 Value Model $V_\phi(s)$,估计**从状态 $s$ 开始的累积奖励**。

$$
V_\phi(s_t) \approx \mathbb{E}[R_t | s_t]
$$

Value Model 训练:

$$
\mathcal{L}_{\text{value}} = \mathbb{E}_t \left[ (V_\phi(s_t) - \hat{R}_t)^2 \right]
$$

### 4.2 GenRec: 自适应 Group Baseline

GRPO 不需要 Value Model,用 group baseline:

$$
\bar{r} = \frac{1}{G} \sum_{i=1}^G r_i
$$

$$
\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}
$$

### 4.3 Critic 对比

| 维度 | Value Model | Group Baseline |
| :--- | :--- | :--- |
| 模型数量 | 1 (额外) | 0 (用现有) |
| 显存 | × 1.5 | × 1.0 |
| 训练成本 | 高 | 低 |
| 稳定性 | 中 | 高 |
| 偏差 | 可能 bias | 低 |
| 信号强度 | 高(单值) | 中(相对) |

### 4.4 PPO vs GRPO 的工程对比

```text
PPO:
  Policy + Ref + Value + Reward = 4 模型
  显存: ~4 × policy size
  训练: 慢,但稳定

GRPO:
  Policy + Ref = 2 模型
  显存: ~2 × policy size
  训练: 快,但每 prompt 需 G 次采样
```

> **核心**:GRPO 是 PPO 的"极简版",用 group baseline 替代 value model。

---

## 5. KL 约束对比

### 5.1 KL 约束的数学

$$
\mathcal{L}_{\text{KL}} = \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

KL 防止策略偏离 reference 太远:

$$
\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} \right]
$$

### 5.2 KL 系数对比

| 场景 | β | 备注 |
| :--- | ---: | :--- |
| LLM PPO | 0.01 – 0.1 | 标准 |
| LLM DPO | 0.1 – 0.5 | 偏好强度 |
| LLM GRPO | 0.01 – 0.04 | DeepSeek-R1 |
| GenRec V1 DPO | 0.1 – 1.0 | 业务偏好 |
| GenRec V2 GBPO | 0.1 – 0.5 | 群体偏好 |
| GenRec Think GRPO | 0.01 – 0.04 | 类 R1 |

### 5.3 KL 约束的工程实现

#### 5.3.1 PPO 中的精确 KL

```python
kl = policy_logprobs - ref_logprobs  # per-token KL
loss_kl = beta * kl.mean()
```

#### 5.3.2 GRPO 中的近似 KL

```python
# k1 estimator (DeepSeek-Math)
k1 = policy_logprobs - ref_logprobs
# k2 estimator (更精确)
k2 = (policy_logprobs - ref_logprobs) ** 2 / 2
kl = (k1 - k2).mean()
loss_kl = beta * kl
```

#### 5.3.3 k3 estimator (无偏)

```python
ratio = (policy_logprobs - ref_logprobs).exp()
k3 = (ratio - 1) - ratio.log()
kl = k3.mean()
```

### 5.4 KL 过大 / 过小的后果

| 现象 | 原因 | 解决 |
| :--- | :--- | :--- |
| 训练不收敛 | β 太小 | 加大 β |
| 业务无提升 | β 太大 | 减小 β |
| 模型崩塌 | β 太小 + LR 大 | 减小 LR + 加大 β |
| 模型不变 | β 太大 + LR 小 | 加大 LR |

---

## 6. 探索 vs 利用对比

### 6.1 LLM: 推理多样性

LLM 的"探索"主要在**推理时**:

| 技巧 | 作用 |
| :--- | :--- |
| Temperature | 控制分布尖锐度 |
| Top-p / Top-k | 截断低概率 token |
| Beam Search | 多路径 |
| Diverse Beam | 强制多样性 |
| Repetition Penalty | 防止重复 |

### 6.2 GenRec: 召回覆盖率

GenRec 的"探索"既在**训练时**也在**推理时**:

| 阶段 | 技巧 | 作用 |
| :--- | :--- | :--- |
| 训练 | ε-greedy | 偶尔采样非 top-1 |
| 训练 | Entropy bonus | 鼓励分布多样性 |
| 训练 | GRPO group | 群体探索 |
| 推理 | Temperature | 候选多样性 |
| 推理 | ε-greedy | 长尾覆盖 |

### 6.3 探索 vs 利用的对比

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| 主要探索 | 推理时 | 训练 + 推理 |
| 探索维度 | 文本多样性 | 物品覆盖 |
| 探索强度 | 中 | 高(长尾需要) |
| 探索失败代价 | 中(回答差) | 高(用户流失) |
| 探索奖励 | 低 | 直接业务价值 |

### 6.4 RL 中的探索

| 算法 | 探索机制 |
| :--- | :--- |
| PPO | entropy bonus |
| GRPO | group sampling |
| OneRec-Think | Rollout-Beam |
| OxygenREC | SA-GCPO 多策略 |

---

## 7. 在线 vs 离线 RL 对比

### 7.1 LLM: 几乎都是离线

LLM 的 RL 几乎全部是**离线**的:

```text
Step 1: 离线训练 policy (PPO/DPO/GRPO)
Step 2: 上线 policy
Step 3: 用户交互
Step 4: 收集偏好数据
Step 5: 重新训练
```

原因:
- LLM 部署成本高,频繁更新不现实
- 离线 RL 已经足够强
- 在线 RL 风险大

### 7.2 GenRec: 在线 + 离线

GenRec 的 RL **同时支持在线和离线**:

```text
离线 RL:
  - 训练周期: 几小时 ~ 几天
  - 数据来源: 历史日志
  - 算法: PPO / DPO / GRPO

在线 RL:
  - 训练周期: 实时(秒级)
  - 数据来源: 实时行为
  - 算法: Contextual Bandit, 简化 RL
```

### 7.3 在线 RL 的设计

| 维度 | 离线 RL | 在线 RL |
| :--- | :--- | :--- |
| 数据 | 历史日志 | 实时反馈 |
| 训练频率 | 周/月 | 实时 |
| 风险 | 中 | 高 |
| 算法 | 复杂 | 简单(Bandit) |
| 评估 | 离线指标 | 在线指标 |

### 7.4 GenRec 在线 RL 案例

#### 7.4.1 Contextual Bandit

```python
class ContextualBandit:
    def __init__(self, model):
        self.model = model

    def recommend(self, context, candidates):
        # ε-greedy
        if random() < self.epsilon:
            return random.choice(candidates)
        scores = self.model(context, candidates)
        return scores.argmax()

    def update(self, context, item, reward):
        # 在线更新
        self.model.update(context, item, reward)
```

#### 7.4.2 实时 RL

```text
用户访问:
  - 收集 context
  - 实时推荐
  - 收集反馈
  - 立即更新

延迟要求: < 100ms
```

### 7.5 LLM 与 GenRec 在 RL 上的根本差异

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| 离线 RL | ✅ | ✅ |
| 在线 RL | ❌ (极少) | ✅ (常用) |
| Bandit | ❌ | ✅ |
| 部署成本 | 极高 | 中 |
| 风险 | 高 | 中 |

---

## 8. 奖励黑客对比

### 8.1 LLM 奖励黑客

#### 8.1.1 常见 Hack

| Hack | 表现 | 原因 |
| :--- | :--- | :--- |
| 长度膨胀 | 回答变长 | RM 偏长回答 |
| 重复生成 | 模型复读 | entropy bonus 不足 |
| 风格化 | 回答模板化 | RM 偏特定风格 |
| Sycophancy | 迎合用户 | RM 训练数据偏 |
| 过度谨慎 | 拒绝回答 | 安全约束过强 |

#### 8.1.2 防御

```text
- KL 约束 (限制偏离 ref)
- Length penalty (显式惩罚过长)
- Diversity bonus (奖励多样性)
- Multi-RM ensemble (多 RM 投票)
- Constitutional AI (规则约束)
```

### 8.2 GenRec 奖励黑客

#### 8.2.1 常见 Hack

| Hack | 表现 | 原因 |
| :--- | :--- | :--- |
| 推荐爆款 | 集中推头物品 | CTR reward 偏 |
| 推送低质 | 推擦边物品 | 点击率高但差 |
| 过度商业 | 推广告 | RPM 高 |
| 推送过时 | 推荐旧物品 | 历史数据偏 |
| 同质化 | 推荐都类似 | entropy 不足 |

#### 8.2.2 防御

```text
- 多 reward 加权 (CTR + GMV + 多样性)
- 长期价值 reward (LTGM)
- 多样性约束 (Coverage)
- 长尾物品 boost (IPS)
- 实时监控 (异常熔断)
```

### 8.3 奖励黑客对比

| 维度 | LLM | GenRec |
| :--- | :--- | :--- |
| Hack 模式 | 文本偏置 | 物品偏置 |
| 表现 | 复读 / 迎合 | 推头 / 同质化 |
| 危害 | 中(用户体验) | 大(生态) |
| 防御难度 | 中 | 大 |

---

## 9. 计算成本对比

### 9.1 LLM RL 成本

```text
4 模型训练:
  - Policy: 7B × 2 = 14GB (FP16)
  - Ref: 7B × 2 = 14GB (FP16)
  - Value: 7B × 2 = 14GB (FP16)
  - Reward: 7B × 2 = 14GB (FP16)
  - 总: 56GB

Rollout:
  - 256 prompts × 1024 tokens
  - 每 prompt 一次前向
  - 总: 256K tokens

Reward 计算:
  - 256 prompts × reward_model 前向

优化:
  - Adam: 7B × 8 = 56GB
```

### 9.2 GenRec RL 成本

```text
2 模型训练 (DPO/GRPO):
  - Policy: 7B × 2 = 14GB
  - Ref: 7B × 2 = 14GB
  - 总: 28GB (DPO)

DPO/GRPO:
  - 256 × 2 = 512 个 (pos, neg) 对
  - 或 256 × G = 256 × 8 = 2048 个生成
  - 总: 256K tokens (DPO) / 2M tokens (GRPO)

GRPO Rollout:
  - 256 prompts × G=8 × 1024 tokens
  - 总: 2M tokens (远大于 DPO)
```

### 9.3 成本对比

| 维度 | LLM PPO | LLM GRPO | GenRec DPO | GenRec GRPO |
| :--- | :---: | :---: | :---: | :---: |
| 模型数 | 4 | 2 | 2 | 2 |
| 单 token 成本 | 高 | 中 | 中 | 中 |
| Rollout 成本 | 中 | 中(G 倍) | 无 | 中(G 倍) |
| 总 GPU 时 | 高 | 中 | 低 | 中 |

### 9.4 GRPO 的"乘法诅咒"

```text
GRPO 的 Rollout:
  prompt 数 = N
  每 prompt G 个生成
  每生成 L 个 token

总生成 token = N × G × L

G=8, N=256, L=1024:
  = 256 × 8 × 1024 = 2M tokens (PPO 的 8 倍)
```

> **核心**:GRPO 用计算换显存,但代价是 G 倍 rollout。

---

## 10. 实际效果对比

### 10.1 LLM RLHF 效果

| 模型 | MT-Bench | 人类偏好 win rate |
| :--- | :---: | :---: |
| Llama 2 7B (SFT) | 6.5 | 50% |
| Llama 2 7B (SFT + RLHF) | 7.0 | 65% |
| Llama 2 70B (SFT) | 7.5 | 60% |
| Llama 2 70B (SFT + RLHF) | 8.5 | 85% |
| GPT-4 | 9.32 | 95% |

> RLHF 通常带来 5-10 分提升。

### 10.2 GenRec RL 效果

| 模型 | Recall@10 | NDCG@10 | 在线 CTR | 在线 GMV |
| :--- | :---: | :---: | :---: | :---: |
| 基础 SFT (OneRec V1 base) | 0.32 | 0.18 | — | — |
| + DPO | 0.36 | 0.21 | +5.2% | +3.8% |
| + GBPO | 0.38 | 0.22 | +7.0% | +5.3% |
| + GRPO + Rollout | 0.39 | 0.23 | +8.5% | +6.5% |
| + Distill + Rec-RL | 0.40 | 0.24 | +10.0% | +7.8% |

> GenRec 的 RL 收益比 LLM 更直接(对齐业务)。

### 10.3 关键发现对比

| 发现 | LLM | GenRec |
| :--- | :--- | :--- |
| RL 必加 | 是 | 是 |
| RL 提升 | 5-10 分 MT-Bench | 5-10% CTR |
| RL 风险 | 中(训练崩) | 低(信号强) |
| RL 算力 | 高(4 模型) | 中(2 模型) |
| 业务对齐 | 间接 | 直接 |

---

## 11. 未来融合方向

### 11.1 三大融合方向

#### 11.1.1 统一 RL 算法栈

```text
未来:
  PPO / DPO / GRPO / GBPO / GSPO
  都收敛到一套统一的 RL 算法
  - LLM 用 (偏好 reward)
  - GenRec 用 (业务 reward)
  - 接口统一,实现统一
```

#### 11.1.2 跨域 RL

```text
多任务 RL:
  - 一个 policy 同时对齐 LLM + GenRec
  - 共享 backbone
  - 任务专属 head
```

#### 11.1.3 在线 + 离线混合 RL

```text
未来:
  - 离线 GRPO 训练基础 policy
  - 在线 Bandit 实时调整
  - 二者协同
```

### 11.2 技术突破点

| 突破点 | 当前状态 | 期望 |
| :--- | :--- | :--- |
| 在线 RL | 部分应用 | 主流 |
| Critic-free | GRPO 已实现 | 标准化 |
| 多 reward 融合 | 简单加权 | 自动平衡 |
| RL 与 SFT 统一 | 无 | 一阶段训练 |
| 业务 reward 自适应 | 固定 | 动态调整 |

### 11.3 与本系列其他章节的衔接

```text
[[4-Token化与词表对比]]  (输入表征)
   ↓
`5-训练目标与损失对比` (训练目标)
   ↓
[[6-评估指标对比]]       (评估指标)
   ↓
[[7-RLHF与RL对齐对比]] (本章 - 对齐训练)
```

### 11.4 总结

> **LLM 用 RL 对齐"人类偏好",GenRec 用 RL 对齐"业务指标"。两者本质同构(都是 reward-driven optimization),差别仅在 reward 来源。RL 是 LLM 与 GenRec 共同的事实标准对齐接口,未来会进一步融合**。

---

## 12. 深度专题

### 12.1 RLHF 完整流程的细节

#### 12.1.1 SFT 阶段详解

```text
数据:
  - 人工撰写的高质量回答
  - 或 GPT-4 生成 + 人工筛选
  - 通常 10K - 100K 条

训练:
  - 全参数微调
  - LR: 1e-5 ~ 5e-5
  - Epoch: 2-5
  - 用 NEFTune (Noisy Embedding FT) 提升鲁棒性
```

#### 12.1.2 RM 训练详解

```text
数据:
  - prompt + y_w + y_l 三元组
  - 10K - 1M 条

训练:
  - 模型: base model (或 SFT 后)
  - 输出: 标量分数
  - Loss: -log σ(r_w - r_l)
  - LR: 1e-5 ~ 5e-5
  - Epoch: 2-3
```

#### 12.1.3 PPO 训练详解

```text
数据:
  - 大量 prompt(无需回答)
  - 或真实使用 query

训练:
  - Rollout: 生成回答
  - Reward: RM 评分
  - Update: PPO clip + KL
  - Epoch: 1 (per rollout)
  - LR: 5e-7 ~ 5e-6 (低)
```

### 12.2 DPO 的优势与局限

#### 12.2.1 DPO 优势

- **无需 RM**:节省训练成本
- **无需在线采样**:离线训练
- **理论保证**:与 PPO 等价的 policy 极限
- **实现简单**:标准 SFT 流程

#### 12.2.2 DPO 局限

- **过拟合**:偏好对分布外的回答可能崩
- **不稳定性**:多次训练结果差异大
- **长度偏置**:可能偏长回答
- **无 value**:无法评估中间状态

#### 12.2.3 DPO 失败的常见原因

| 现象 | 原因 |
| :--- | :--- |
| 训练崩 | β 太大, 偏好对噪声 |
| 无提升 | β 太小, 偏置不够 |
| 偏长回答 | length 偏置 |
| 模式坍塌 | 单一风格过度优化 |

### 12.3 GRPO 的"乘 G"计算

#### 12.3.1 详细流程

```python
# GRPO 单步
def grpo_step(model, ref_model, prompts, G=8):
    loss = 0
    for prompt in prompts:
        # 1. 采样 G 个回答
        responses = [model.generate(prompt) for _ in range(G)]

        # 2. 算 rewards
        rewards = [reward_fn(prompt, r) for r in responses]

        # 3. group advantage
        advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

        # 4. PPO loss
        for r, adv in zip(responses, advantages):
            policy_logprobs = model.compute_logprobs(prompt, r)
            ref_logprobs = ref_model.compute_logprobs(prompt, r)
            ratio = exp(policy_logprobs - ref_logprobs)
            surr = ratio * adv
            loss += min(surr, clip(surr, 1-eps, 1+eps))
            loss += beta * kl

    return loss / (len(prompts) * G)
```

#### 12.3.2 GRPO 的显存优化

```text
标准 GRPO:
  - 每个 response 独立前向
  - 显存: G × seq × dim

优化:
  - Batched G 个 response
  - 显存: G × seq × dim (一样,但 GPU 利用率高)
  - 进一步: Group-level batch
```

### 12.4 Reward Hacking 的诊断

#### 12.4.1 LLM 诊断

```python
def detect_length_hack(model_outputs):
    """检测长度膨胀"""
    lengths = [len(o) for o in model_outputs]
    baseline = np.mean(lengths_history)
    return lengths[-1000:] / baseline > 1.3  # 涨 30% 可能 hack
```

#### 12.4.2 GenRec 诊断

```python
def detect_popularity_hack(recommended_items, item_popularity):
    """检测流行度偏置"""
    avg_pop = np.mean([item_popularity[i] for i in recommended_items])
    baseline = np.mean(list(item_popularity.values()))
    return avg_pop / baseline > 2.0  # 偏 2 倍可能 hack
```

### 12.5 RL 中的样本效率

| 算法 | 样本效率 | 备注 |
| :--- | :---: | :--- |
| PPO | 中 | 多次 epoch |
| DPO | 高 | 直接优化偏好 |
| GRPO | 中 | G 倍 rollout |
| GBPO | 高 | 群体构造 |
| Rec-RL | 中 | 业务反馈 |

### 12.6 Multi-RM Ensemble

#### 12.6.1 多 RM 投票

```python
class MultiRM:
    def __init__(self, rms):
        self.rms = rms

    def score(self, prompt, response):
        scores = [rm.score(prompt, response) for rm in self.rms]
        return np.mean(scores)
```

#### 12.6.2 多 Reward 业务应用

```python
class GenRecMultiReward:
    def __init__(self):
        self.ctr_reward = CTRReward()
        self.diversity_reward = DiversityReward()
        self.long_tail_reward = LongTailReward()

    def score(self, recommended_list):
        ctr = self.ctr_reward(recommended_list)
        diversity = self.diversity_reward(recommended_list)
        long_tail = self.long_tail_reward(recommended_list)
        return (
            0.6 * ctr +
            0.3 * diversity +
            0.1 * long_tail
        )
```

### 12.7 Online Bandit vs Full RL

| 维度 | Bandit | Full RL |
| :--- | :--- | :--- |
| 状态 | 无 | 有 |
| 序列 | 单步 | 多步 |
| 算法 | UCB / Thompson | PPO / GRPO |
| 训练成本 | 极低 | 高 |
| 适用 | 简单场景 | 复杂场景 |

### 12.8 真实部署 RL 的注意事项

#### 12.8.1 检查点

```python
# 训练前 checklist:
checklist = {
    "Reward function 验证": "✓",
    "KL 系数已调": "✓",
    "梯度裁剪已设": "✓",
    "Reference 模型冻结": "✓",
    "Logits clip 已设": "✓",
    "NaN 检查已设": "✓",
    "Loss spike 监控已设": "✓",
    "Checkpoint 间隔已设": "✓",
}
```

#### 12.8.2 部署前测试

```text
- 离线回放 (用历史数据模拟)
- 小流量 A/B (1% → 10%)
- 实时监控 (reward 异常)
- 自动熔断 (回滚机制)
```

### 12.9 RL 与 SFT 的统一

#### 12.9.1 当前范式

```text
Step 1: Pretrain
Step 2: SFT
Step 3: RL (PPO/DPO/GRPO)
```

#### 12.9.2 未来范式

```text
Step 1: Pretrain
Step 2: One-stage (SFT + RL unified)
  - 同时用 (prompt, y) 和 (prompt, y_w, y_l)
  - 联合 CE + DPO loss
```

### 12.10 RLHF 的"全栈"成本

```text
LLM RLHF (7B 模型):
  - SFT 数据: 50K 条
  - RM 数据: 100K 偏好对
  - PPO 训练: 7 天 × 64 A100
  - 总成本: ~$50K USD

GenRec RL (7B 模型):
  - SFT 数据: 100B 样本(已有)
  - DPO 偏好对: 10M
  - DPO 训练: 3 天 × 32 A100
  - 总成本: ~$20K USD

GRPO 训练:
  - G=8 rollout
  - 总成本: PPO × G
```

### 12.11 RL 算法选择的决策树

```text
需求:
  ├─ 想要简单 → DPO / SimPO
  ├─ 想要 SOTA → GRPO / GSPO
  ├─ 想要显式 RM → PPO
  ├─ 想要业务对齐 → GBPO / Rec-RL
  ├─ 想要可验证 → RLVR
  ├─ 想要长尾 → GBPO + Long-tail Reward
  └─ 想要在线 → Bandit
```

### 12.12 RL 与 RAG / Tool Use 的协同

```text
未来 LLM:
  - RLHF / DPO 对齐偏好
  - RAG 提供事实
  - Tool Use 调用外部 API

未来 GenRec:
  - RL 对齐业务
  - Retrieval 提供候选
  - Tool Use 调整参数
```

### 12.13 RLHF 的安全性评估

| 评估维度 | 方法 | 工具 |
| :--- | :--- | :--- |
| 毒害 | 关键词 + 模型分类 | OpenAI Moderation |
| 偏见 | BBQ / WinoGender | 公开 benchmark |
| 越狱 | AdvBench | 学术数据集 |
| 隐私 | 个人信息检测 | 自研 |
| 错误信息 | TruthfulQA | 公开 benchmark |

### 12.14 GenRec RL 的业务约束

| 约束 | 说明 |
| :--- | :--- |
| 库存约束 | 推无货商品 = 0 |
| 公平性 | 不能歧视用户群 |
| 多样性 | 推荐不能太集中 |
| 新颖性 | 不能全是用户已知的 |
| 长期价值 | 不能透支长期 |

### 12.15 RL 算法与传统推荐对比

| 维度 | 传统推荐 | GenRec RL |
| :--- | :--- | :--- |
| 算法 | BPR / ListNet | DPO / GRPO |
| 目标 | 召回 / 排序准确 | 业务指标 |
| 训练 | 离线 | 离线 + 在线 |
| 信号 | 偏好对 | 业务反馈 |
| 灵活度 | 中 | 高 |

### 12.16 RL 训练的 GPU 优化

| 技巧 | 节省显存 |
| :--- | :---: |
| LoRA + RL | 80% |
| 8-bit Optimizer | 75% |
| Activation Checkpointing | 50% |
| Gradient Accumulation | — (时间换) |
| FSDP (大模型) | — (分布式) |
| Reference 模型量化 | 50% |

### 12.17 RLHF 的 Eval 闭环

```text
训练 → 离线 eval → 内部 test → 灰度 → 在线 eval
                                          ↓
                                          反馈数据回流
                                          ↓
                                        下一轮训练
```

> 核心:**Eval 不是终点,是新一轮训练的起点**。

### 12.18 RL 未来 5 年的趋势

1. **统一算法**:PPO / DPO / GRPO 收敛到统一接口
2. **自动化 reward 设计**:LLM 自动构造 reward
3. **多模态 RL**:文本 + 图像 + 行为联合 reward
4. **在线 RL**:实时反馈驱动
5. **跨域 RL**:LLM + GenRec 共享 RL 算法

### 12.19 KL 散度的 4 种实现

#### 12.19.1 k1 estimator (最简单)

```python
def kl_k1(policy_logprobs, ref_logprobs):
    """最简单的 KL 估计"""
    return (policy_logprobs - ref_logprobs).mean()
```

> 优点:简单。缺点:有偏。

#### 12.19.2 k2 estimator (无偏,小误差近似)

```python
def kl_k2(policy_logprobs, ref_logprobs):
    """r = π/π_ref, kl = (r-1) - log(r)"""
    logratio = policy_logprobs - ref_logprobs
    return ((logratio.exp() - 1) - logratio).mean()
```

#### 12.19.3 k3 estimator (DeepSeek)

```python
def kl_k3(policy_logprobs, ref_logprobs):
    """k3 = (r-1) - log(r), 与 k2 等价"""
    logratio = policy_logprobs - ref_logprobs
    return ((logratio.exp() - 1) - logratio).mean()
```

> GRPO 默认用 k3 estimator。

#### 12.19.4 js estimator (Jensen-Shannon)

```python
def kl_js(policy_logprobs, ref_logprobs):
    """JS 散度,有界,稳定"""
    p = policy_logprobs.exp()
    q = ref_logprobs.exp()
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p, m) + F.kl_div(q, m))
```

> 优势:有界,适合大 KL。

### 12.20 多 Reward 联合训练

#### 12.20.1 加权求和

```python
total_reward = (
    w1 * reward1 +
    w2 * reward2 +
    w3 * reward3
)
```

> 简单但需调权重。

#### 12.20.2 不确定性加权 (UW)

```python
class UWReward:
    def __init__(self, num_rewards):
        self.log_vars = nn.Parameter(torch.zeros(num_rewards))

    def compute(self, rewards):
        # rewards: list of per-reward tensors
        weighted = []
        for i, r in enumerate(rewards):
            precision = torch.exp(-self.log_vars[i])
            weighted.append(precision * r + self.log_vars[i])
        return sum(weighted)
```

#### 12.20.3 Pareto 多目标 RL

```python
class ParetoReward:
    """Pareto 多目标 RL"""
    def compute(self, rewards):
        # 多目标 Pareto 前沿
        # 沿 Pareto 优化,牺牲一个换另一个
        ...
```

### 12.21 KL Penalty vs KL Constraint

#### 12.21.1 KL Penalty (PPO / GRPO 用)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

> 软约束,自动平衡。

#### 12.21.2 KL Constraint (TRPO)

$$
\max \mathcal{L}_{\text{policy}} \quad \text{s.t.} \quad \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \le \delta
$$

> 硬约束,更严格。

#### 12.21.3 工业选择

- **LLM**:KL penalty 为主(PPO / GRPO)
- **GenRec**:KL penalty 为主(GBPO / Rec-RL)
- **GenRec 严苛场景**:KL constraint(广告、合规)

### 12.22 RLHF 与 RLAIF 的对比

| 维度 | RLHF | RLAIF |
| :--- | :--- | :--- |
| 标注者 | 人类 | LLM |
| 成本 | 高 | 低 |
| 规模 | 小(1万-10万) | 大(无限) |
| 一致性 | 中 | 高 |
| 偏见 | 人类偏差 | LLM 偏差 |
| 适用 | 关键决策 | 大规模 |

### 12.23 GenRec RL 的"业务奖励设计"

#### 12.23.1 多业务 reward

```python
class GenRecBusinessReward:
    def __init__(self):
        self.ctr_weight = 0.4
        self.cvr_weight = 0.3
        self.diversity_weight = 0.2
        self.long_tail_weight = 0.1

    def compute(self, recommended_list, user_history):
        ctr = self.ctr_model.predict(recommended_list)
        cvr = self.cvr_model.predict(recommended_list)
        diversity = self.compute_diversity(recommended_list)
        long_tail = self.compute_long_tail(recommended_list)

        return (
            self.ctr_weight * ctr +
            self.cvr_weight * cvr +
            self.diversity_weight * diversity +
            self.long_tail_weight * long_tail
        )
```

#### 12.23.2 长尾物品 boost

```python
def long_tail_boost(item):
    """长尾物品加权"""
    pop = item.popularity  # 0-1
    if pop < 0.01:  # 长尾
        return 1.5
    elif pop < 0.1:
        return 1.1
    else:
        return 1.0
```

### 12.24 RL 中的"权重共享"问题

#### 12.24.1 LLM 的权重共享

```text
LLM:
  - Policy, Ref, Value, Reward 通常共享 base model
  - 不同 head 各自独立
```

#### 12.24.2 GenRec 的权重共享

```text
GenRec:
  - Policy, Ref 共享 encoder
  - Reward head 独立训练
```

### 12.25 RL 训练的"早停" 准则

| 准则 | 阈值 | 含义 |
| :--- | :--- | :--- |
| Reward 曲线饱和 | 1000 步无提升 | 收敛 |
| KL 过大 | > 0.1 per token | 偏离 ref |
| Logits 熵降 | < 阈值 | 模式坍塌 |
| Val loss 反弹 | 5% 反弹 | 过拟合 |

### 12.26 RL 训练资源规划

```text
LLM RLHF (7B):
  - GPU: 64 × A100 (80G)
  - 时间: 7 天
  - 总成本: ~$50K USD

GenRec RL (7B):
  - GPU: 32 × A100 (80G)
  - 时间: 3 天
  - 总成本: ~$20K USD

RL 推理 (部署):
  - 单次推理: ~1s
  - QPS 限制: ~10 / GPU
```

### 12.27 RLHF 与 DPO 的选择矩阵

| 场景 | 推荐 | 原因 |
| :--- | :--- | :--- |
| 资源充足 | PPO | SOTA |
| 资源中等 | GRPO | 无 critic |
| 资源紧张 | DPO | 简单 |
| 偏好数据少 | KTO | 单项数据 |
| 推理增强 | RLVR + GRPO | 可验证 |
| 业务对齐 | GBPO / Rec-RL | 业务 reward |

### 12.28 RL 与 In-Context Learning

| 维度 | RL | ICL |
| :--- | :--- | :--- |
| 是否改参数 | 是 | 否 |
| 推理成本 | 中 | 低 |
| 适配新任务 | 慢 | 快 |
| 长期价值 | 高 | 低 |
| 适用 | 高频任务 | 一次性 |

### 12.29 RLHF 与 Continual Learning

```text
LLM Continual RL:
  - 持续收集偏好
  - 增量训练
  - 防止 catastrophic forgetting
  - 关键技术: EWC / Replay

GenRec Continual RL:
  - 实时反馈
  - 在线更新
  - 防止兴趣漂移
  - 关键技术: Replay Buffer
```

### 12.30 RL 在 LLM 与 GenRec 中的"镜像对称"

```text
LLM RL 范式:
  Pretrain → SFT → RLHF(DPO/GRPO)

GenRec RL 范式:
  Pretrain(NIP) → SFT(NIP fine-tune) → RL(GBPO/GRPO/Rec-RL)

关键镜像:
  - 训练数据: 文本 vs 行为
  - Reward: 偏好 vs 业务
  - Critic: Value Model vs Group Baseline
  - KL: 与 ref vs 与 ref
  - 输出: Token vs Item
```

> **核心**:LLM 与 GenRec 的 RL 范式"镜像对称",底层算法同构。

---

## 附录 A · 速查表

### A.1 LLM RL 算法速查

| 算法 | Critic | Reward 来源 | 显存 | 稳定性 |
| :--- | :--- | :--- | :---: | :---: |
| PPO | Value | RM / 人类 | 高 | 中 |
| DPO | 无 | 偏好对 | 中 | 中 |
| GRPO | 无 | 任意 | 中 | 高 |
| RLAIF | Value | LLM | 高 | 中 |
| Constitutional | Value | 规则 | 高 | 中 |
| Self-Rewarding | Value | 自我 | 高 | 低 |
| RLVR | 无 | 规则 | 中 | 高 |

### A.2 GenRec RL 算法速查

| 算法 | Critic | Reward 来源 | 显存 | 稳定性 |
| :--- | :--- | :--- | :---: | :---: |
| DPO (OneRec V1) | 无 | 用户偏好 | 中 | 中 |
| GBPO (OneRec V2) | 无 | 业务指标 | 中 | 高 |
| GRPO (Think) | 无 | 业务 + rollout | 中 | 高 |
| Rec-RL (Foundation) | 无 | 业务 | 中 | 高 |
| HEPO | 无 | 偏好 + 探索 | 中 | 中 |
| SA-GCPO | 无 | 业务自适应 | 中 | 高 |
| GSPO | 无 | 序列级业务 | 中 | 高 |

### A.3 RL 算法对比矩阵

| 算法 | LLM | GenRec | 共同点 |
| :--- | :--- | :--- | :--- |
| PPO | ✅ | (少) | Critic-based |
| DPO | ✅ | ✅ | Reference + 偏好对 |
| GRPO | ✅ | ✅ | Group baseline |
| Self-Rewarding | ✅ | (少) | 自评 |
| RLAIF | ✅ | ❌ | LLM judge |
| RLVR | ✅ | (少) | 规则 reward |
| GBPO | ❌ | ✅ | 群体偏好 |

### A.4 Reward 来源速查

| 来源 | LLM | GenRec |
| :--- | :--- | :--- |
| 人类偏好 | ✅ | ❌ |
| AI 偏好 | ✅ | ❌ |
| 业务指标 | ❌ | ✅ |
| 规则验证 | ✅ | ✅ |
| 长尾奖励 | ❌ | ✅ |

### A.5 关键 takeaway

1. **RL 是 LLM 与 GenRec 共同的对齐接口**
2. **PPO → DPO → GRPO 演进**:减少 critic, 提升稳定
3. **Reward 来源**:LLM 用偏好,GenRec 用业务
4. **GRPO 是当前主流**:无 critic, group baseline
5. **在线 RL**:LLM 几乎不用,GenRec 常用

### A.6 RL 训练反模式

| 反模式 | 表现 | 解决 |
| :--- | :--- | :--- |
| 长度膨胀 | 回答变长 | Length penalty |
| 流行度偏置 | 推头物品 | 多样性 reward |
| Reward 拟合 | 训练崩 | KL 加大 |
| GRPO 方差大 | 训练不稳 | G 加大 |
| 偏好对噪声 | 训练崩 | 数据清洗 |
| Ref 模型变化 | KL 异常 | 冻结 ref |

---

## 附录 B · 参考与链接

### B.1 内部链接(同模块)

- [[0-总览与核心区别]] - LLM-vs-GenRec 主目录
- [[4-Token化与词表对比]]
- `5-训练目标与损失对比`
- [[6-评估指标对比]]

### B.2 RL 模块(11 份)

- [[0-总览与核心区别]] - RL 主目录
- `../RL/1-RL基础与数学.md`
- `../RL/2-经典RL算法详解.md`
- `../RL/3-RL在LLM中的经典方法.md` - **必读**: LLM RL 经典方法
- `../RL/4-RL在GenRec中的经典方法.md` - **必读**: GenRec RL 经典方法
- `../RL/5-奖励设计与RL对齐.md` - Reward 设计
- `../RL/6-RL工程实践与部署.md`
- `../RL/7-论文案例-OneRec系列RL.md` - **OneRec V1/V2/Think**
- `../RL/8-论文案例-其他GenRec RL.md` - **GPR / OxygenREC / OneMall**
- `../RL/9-RL未来方向与开放问题.md`
- `../RL/补充RL论文清单.md`

### B.3 GR deep-dives (24 篇)

> **GR 第09章 deep-dives**:

- `../GR/第09章-规模化排序与新架构/deep-dives/01-TokenMixer-Large-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/02-OneTrans-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/03-STCA-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/04-MDL-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/05-MixFormer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/06-MSN-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/07-HyFormer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/08-STORE-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/09-FAT-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/10-HHFT-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/11-EST-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/12-SORT-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/13-UniScale-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/14-MGDIN-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/15-HeMix-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/16-CDNet-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/17-LFM4Ads-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/18-SMES-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/19-LASER-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/20-OxygenREC-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/21-InterFormer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/22-Kunlun-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/23-HoMer-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives/24-MTmixAtt-AutoToken-深度解析.md`

### B.4 GR deep-dives-extended (10 篇)

- `../GR/第09章-规模化排序与新架构/deep-dives-extended/25-PLUM-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/26-GRAB-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/27-PinRec-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/28-NetflixGenRec-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/29-COBRA-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/30-LiGR-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/31-STATIC-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/32-GRAM-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/33-DeepInterestGR-深度解析.md`
- `../GR/第09章-规模化排序与新架构/deep-dives-extended/34-SelfEvolvingRec-深度解析.md`

### B.5 外部参考

- Ouyang et al. (2022). Training Language Models to Follow Instructions with Human Feedback. (InstructGPT, RLHF)
- Schulman et al. (2017). Proximal Policy Optimization. (PPO)
- Rafailov et al. (2023). Direct Preference Optimization. (DPO)
- Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning. (GRPO)
- Azar et al. (2024). A General Theoretical Paradigm to Understand Preference Optimization. (IPO)
- Ethayarajh et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. (KTO)
- Meng et al. (2024). SimPO: Simple Preference Optimization. (SimPO)
- Hong et al. (2024). ORPO: Monolithic Preference Optimization. (ORPO)
- Bai et al. (2022). Constitutional AI. (RLAIF)
- OpenAI. (2024). Learning to Reason with LLMs. (o1)
- DeepSeek. (2025). DeepSeek-R1. (GRPO + RLVR)
- OneRec 团队 (2025). OneRec V1/V2/Think/Foundation. (DPO/GBPO/GRPO/Rec-RL)
- 陈亮等 (2025). GenRec RL Survey.
---

## 关键洞察

> [!important] **核心 insight**: LLM 用 RLHF/DPO/GRPO 对齐人类偏好,GenRec 用 DPO/GBPO/GRPO 对齐业务指标;两者目标函数几乎相同(KL 散度约束),但 reward 来源不同——LLM 是即时人类反馈,GenRec 是滞后业务信号。

---

## 关联阅读

1. `5-训练目标与损失对比` - 训练目标对比
2. [[6-评估指标对比]] - 评估对比
3. [[8-推理与部署对比]] - 推理对比
4. [[11-融合与未来方向]] - 融合未来

---

> [!summary] **关键 takeaway**:
> 1. **LLM RLHF**:从 PPO → DPO → GRPO,reward 来自人类偏好。
> 2. **GenRec RL**:从 DPO → GBPO → GRPO/Rec-RL,reward 来自业务指标。
> 3. **GRPO 是当前主流**:无需 critic, group baseline,稳定高效。
> 4. **Reward 来源差异**是 LLM 与 GenRec RL 的核心区别。
> 5. **未来融合**:统一 RL 算法栈,跨 LLM + GenRec。

> **核心章节回顾**: [[4-Token化与词表对比]] → `5-训练目标与损失对比` → [[6-评估指标对比]] → [[7-RLHF与RL对齐对比]] → [[8-推理与部署对比]] → [[9-Scaling与扩展对比]] → [[10-工业案例对比]] → [[11-融合与未来方向]]

> 通过这 4 章,我们系统对比了 LLM 与 GenRec 在 Tokenization、训练目标、评估指标、RL 对齐四个核心环节的根本差异与融合趋势。两者的差异源于"输入表征 + 优化目标 + 业务场景"的不同,但在算法层面正在快速收敛——RL 是当前最显著的共同语言。
