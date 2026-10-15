---
title: 生成式推荐与小型 LLM 设计模式
chapter_id: 16
tags:
  - 搜推广
  - 推荐系统
  - 生成式推荐
  - PyTorch
description: 2024-2025 年的关键研究发现:生成式推荐和 30B 以下语言模型的最佳架构几乎相同。
last_updated: 2026-07-14
status: ✅ v1.0
---

# 第 16 章:生成式推荐与小型 LLM 设计模式的趋同化

> **这是全书的"第二篇"——把你从搜推广基础知识带入 2024-2025 年最前沿的 GenRec + Small LLM 融合设计。**
>
> 读到这里你应该已经发现:**生成式推荐模型和 30B 以下的小型语言模型的设计思路正在快速趋同**。HSTU 的 Block ≈ LLaMA 的 Block,P5 用 T5 架构做推荐与 T5 做翻译共享同一套代码,LoRA 既能微调 LLaMA 也能微调 CTR 模型。
>
> **本章目标**:用研究论文 + 代码 + 实践经验三方位,系统性地讲清楚:
> - SwiGLU 为什么替代了 ReLU?RMSNorm 为什么替代了 LayerNorm?RoPE 为什么替代了 Sinusoidal PE?
> - Decoder-only 为什么在 GenRec 和 LLM 中都赢了?GQA/MQA 省了多少显存?
> - 语义 ID tokenization(VQ-VAE/RQ-VAE)如何解决物品数量问题?
> - LoRA/QLoRA/量化/蒸馏——这些"大模型技术"怎么用在推荐上?

## 16.1
### 16.1 核心论点:GenRec 与 Small LLM 的"统一架构"已经出现

2024-2025 年的关键研究发现:**生成式推荐和 30B 以下语言模型的最佳架构几乎相同**。

**为什么"统一架构"会出现?——三个根本驱动力**:

1. **Scaling Law 适用性**:LLM 揭示了一个普适规律——"模型越大 + 数据越多 + 计算越久 = 效果越好"。这条规律对推荐也成立,只是"token"换成"item ID"。GenRec 复用 LLM 的 scaling recipe,自然就是"用 Transformer 训一个大模型"。
2. **基础设施复用**:vLLM、TGI、DeepSpeed、Megatron、FSDP 这套 LLM 工程栈,GenRec 几乎 0 改动就能用。重新造一套"推荐专用训练框架"投入产出比太低,**借用 LLM 生态是最高 ROI 的选择**。
3. **学术和工业的合流**:Meta(HSTU)、Google(TIGER)、字节(OneRec)、阿里(CIDR) 都在 2023-2024 年集中发 GenRec 论文,设计选择高度一致——这反过来证明"统一架构"是行业共识,不是巧合。

**统一架构的本质含义**:你学的 LLaMA 架构(SwiGLU + RMSNorm + RoPE + GQA + Decoder-Only)同时也是 GenRec 的标配。**学一次,两边通用**。

```text
             传统推荐(2016-2022)             现代 GenRec + 小 LLM(2024-2025)
             ────────────────               ──────────────────────────
特征交互:    Concat + MLP                    Self-Attention(完全替代 MLP)
激活函数:    ReLU / PReLU                    SwiGLU / GeGLU
归一化:      BatchNorm / LayerNorm            RMSNorm(唯一选择)
位置编码:    Sinusoidal / 无                  RoPE(旋转位置编码)
注意力:      Multi-Head                      GQA / MQA
架构:        Encoder-Only / DNN              Decoder-Only
Token化:     LabelEncode → Embedding         语义ID(VQ-VAE / RQ-VAE) → Embedding
训练目标:    BCE / BPR                        Cross-Entropy(Next Token Pred)
微调:        全量 Fine-Tune                   LoRA / QLoRA
推理优化:    无                               KV-Cache / PagedAttention
```

**这意味着**:你现在学 LLaMA/Mistral/Qwen 的架构设计,就是在学下一代 GenRec 的架构设计。反之亦然。

### 16.2 架构组件的逐项对比与数学推导

#### 16.2.1 GLU vs FFN(权重配方对比)

**FFN 是 Transformer Block 的"非线性主力"**。注意力层做"信息混合"(信息从哪些 token 流向哪些 token),FFN 做"信息加工"(每个 token 独立地、非线性地变换)。两者配合,才让 Transformer 既能"看见全局"又能"加工细节"。

**为什么 SwiGLU 取代了 ReLU?**

追溯时间线:
- 2017 原版 Transformer:ReLU FFN (`max(0, xW₁)W₂`),简单粗暴
- 2018 GPT-2:沿用 ReLU
- 2020 GPT-3:开始用 GELU(平滑,梯度更连续)
- 2020 T5 论文:提出 GLU 变体,消融显示 SwiGLU 比 ReLU/GELU 在相同参数量下 PPL 低 2%
- 2022 PaLM、Chinchilla:LLaMA 前的 LLM 全部换 SwiGLU
- 2023 LLaMA 2/3、Mistral、Qwen、DeepSeek:全员 SwiGLU

**为什么推荐场景也必须用 SwiGLU?**

推荐里 Embedding 表几千万行,只有少数 item 对当前预测"关键"(用户的真实兴趣 item)。SwiGLU 的"门控"天然适合"放大关键 item、压制噪声 item"——把 gate 学成"如果当前 item 是用户真实兴趣就接近 1,否则接近 0",自然过滤掉无关历史。ReLU 没有这种自适应过滤,所有 item 一视同仁进入下一层,被噪声淹没。

**传统 FFN(ReLU/GELU)**:

$$\text{FFN}_{\text{ReLU}}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 \quad\text{——2 次投影}$$

**SwiGLU FFN(LLaMA/Mistral/Qwen/HSTU)**:

$$\text{FFN}_{\text{SwiGLU}}(x) = (\text{SiLU}(xW_{\text{gate}}) \otimes xW_{\text{up}})W_{\text{down}} \quad\text{——3 次投影}$$

**为什么多一次投影反而更好?——门控机制的梯度分析**:

ReLU FFN 的梯度:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{out}} \cdot W_2^\top \cdot \text{diag}(\mathbb{1}_{xW_1 > 0}) \cdot W_1$$

当 ReLU 门关闭(负值)时,梯度被完全阻断,神经元死亡。

SwiGLU 的梯度:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{out}} \cdot W_{\text{down}}^\top \cdot \big[\text{SiLU}(xW_{\text{gate}}) \otimes \frac{\partial}{\partial x} + \frac{\partial \text{SiLU}}{\partial x} \otimes xW_{\text{up}}\big] \cdot [W_{\text{gate}}, W_{\text{up}}]$$

两个投影都参与梯度计算,总有梯度流动,训练更稳定。

且参数量一致(因为 SwiGLU 的 $d_{\text{ff}} = 8d/3 \approx 2.67d$,而 ReLU 的 $d_{\text{ff}}=4d$)。

**经验效果(T5 ablation, Shazeer 2020)**:

| FFN 类型 | PPL(越低越好) | 参数量 |
| :--- | :--- | :--- |
| ReLU | 3.08(基线) | 8d² |
| GELU | 3.06 | 8d² |
| **SwiGLU** | **3.02** | 8d²(同等参数量) |
| GeGLU | 3.02 | 8d²(同等参数量) |

```python
# SwiGLU 实现——同时适用于 GenRec 和 LLM
class SwiGLU(nn.Module):
    """
    SwiGLU FFN = 2024-2025 年所有主流模型的标配。

    适用模型:
    - LLM: LLaMA 2/3, Mistral, Qwen, DeepSeek, Gemma, Phi
    - GenRec: HSTU, OneRec, 以及越来越多新一代推荐模型
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)  # LLaMA 标准比例
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # SiLU 门控 + 上投影 → 元素积 → 下投影
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

---

**文字$\cdot$公式$\cdot$代码 1:为什么 SwiGLU 多了 1 次投影反而更强,以及它怎么成了 GenRec 和 LLM 的共同选择**

**文字**:为什么 SwiGLU 用 3 次投影,反而比 ReLU FFN 的 2 次投影效果更好? 关键在门控(gating)带来的梯度结构差异。ReLU 有一条致命缺陷:输入为负时梯度严格为 0,神经元一旦"死掉"就再也救不回来,而且 ReLU 的两个投影之间没有乘法交互。SwiGLU 把单次上投影拆成两个并行分支 `W_{g}ate` 和 `W_{u}p`,前者过 SiLU 后元素积乘到后者,再过下投影。这个 element-wise 乘法构成了一个"软门":gate 接近 0 时 up 分支被压制,gate 大时 up 分支被放大;同时因为 SiLU 是平滑非单调函数,即使 gate 很小梯度也能从两个分支同时回传,没有"死神经元"。T5 论文的消融在相同参数量下把 PPL 从 3.08(ReLU)压到 3.02(SwiGLU),自此 PaLM、LLaMA 2/3、Mistral、Qwen、DeepSeek、Gemma 全员上船 SwiGLU。GenRec 这边,HSTU 和 OneRec 也都换掉了 ReLU,因为用户历史里只有少数 item 对当前预测关键,门控天然适合学"放大哪些 item"。

**公式**:

$$\text{SwiGLU}(x) = \big(\text{SiLU}(x \cdot W_{\text{gate}}) \otimes (x \cdot W_{\text{up}})\big) \cdot W_{\text{down}}$$

$$\text{SiLU}(x) = x \cdot \sigma(x), \quad \sigma(x) = \frac{1}{1 + e^{-x}}$$

参数量平衡下 $d_{\text{ff}} = \frac{8}{3}d \approx 2.67d$(对标 ReLU 的 $4d$),三者参数总量一致。

**代码**:

```python
class SwiGLU(nn.Module):
    """SwiGLU FFN:LLaMA / Mistral / Qwen / HSTU 共用,3 次投影 + 元素积门控"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)        # 平衡 ReLU 的 4d 参数量
        self.gate = nn.Linear(d_model, d_ff, bias=False)  # 门控分支
        self.up   = nn.Linear(d_model, d_ff, bias=False)  # 上投影分支
        self.down = nn.Linear(d_ff, d_model, bias=False)  # 下投影

    def forward(self, x):
        # SiLU 平滑门 + 元素积门控,梯度永远从两个分支回传
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

---

#### 16.2.2 RMSNorm——为什么去掉均值反而更好?

**归一化是"训练的稳定器"**。Transformer 的 attention + FFN 不断累加,几层下来 hidden state 的数值范围会发散——某个维度可能 ±1000,某个维度可能 ±0.001,导致训练震荡甚至 NaN。**归一化把每层输出重新缩放到"标准正态附近",保证训练稳定**。

**LayerNorm vs RMSNorm 的根本差异**:

LayerNorm 做两件事:减均值(centering) + 除以标准差(scaling)。RMSNorm 砍掉了"减均值",只保留"除以均方根"。看似简化,实际有几个**意想不到的好处**:

1. **节省 5-10% 计算**:每层少一次均值计算和一次减法
2. **梯度路径更直接**:减均值会引入维度间负相关(`Cov(x_i-μ, x_j-μ) = -σ²/d < 0`),反向传播时梯度相互"拉扯";RMSNorm 没有这个问题,梯度直接
3. **大模型下数值更稳定**:LLaMA-7B 训练实测,用 RMSNorm 比 LayerNorm 收敛快 5%,loss 更低

**工业选型**:从 LLaMA-2 开始,所有主流 LLM 都用 Pre-RMSNorm(归一化在子层之前),GenRec 同样必须沿用。**不要用 Post-LN**(原版 Transformer 的设计),会训崩。

**数学定义对比**:

| 归一化 | 公式 | 计算量 |
| :--- | :--- | :--- |
| BatchNorm | $y = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$ | 2 次 reduce(均值+方差) |
| LayerNorm | $y = \gamma \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta$ | 2 次 reduce(均值+方差) |
| **RMSNorm** | $y = \gamma \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$ | **1 次 reduce(仅 RMS)** |

**RMSNorm 为什么能在去掉均值后保持效果?** (来自 2024 年的理论分析):

原因 1:高维空间中均值 $\approx 0$

$$d_{\text{model}} = 4096,\quad \text{特征的均值} \;\mu = \frac{\sum x_i}{d_{\text{model}}}$$

因为 $x_i$ 服从 $N(0, \sigma^2)$,$\mu \approx 0$,误差 $O(1/\sqrt{d})$,减不减均值对结果影响极小。

原因 2:梯度流动更干净

LayerNorm 中减均值的操作会引入负相关:

$$\text{Cov}(x_i - \mu,\; x_j - \mu) = -\frac{\sigma^2}{d} \quad (\text{负的!})$$

这种相关性在反向传播时会造成梯度"拉扯",RMSNorm 没有这步,梯度更直接。

原因 3:计算节省

从 2 次 reduce 到 1 次 reduce,节省约 $5 \sim 10\%$ 训练时间,对于需训练数周的大模型,这是巨大的收益。

**训练实战经验**:
```python
# 当你从头训练一个 GenRec 或小 LLM 时:
# 归一化配置 = Pre-RMSNorm(而不是 Pre-LayerNorm)

class TransformerBlock(nn.Module):
    """一个标准的 Pre-RMSNorm Transformer Block"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        # 归一化: RMSNorm(不是 LayerNorm!)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        # 注意力
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # FFN: SwiGLU(不是 ReLU!)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        # Pre-Norm:归一化在子层之前
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

#### 16.2.3 RoPE——旋转位置编码

**位置编码是"序列模型的灵魂"**。Transformer 的 attention 本身是"置换不变"的——把输入 token 顺序打乱,attention 输出完全一样(只是位置对应的内容换了)。这显然不符合推荐需求:用户"先点 A 再点 B"和"先点 B 再点 A"是不同行为,推荐结果应该不同。**位置编码把"顺序信息"注入到 token 向量里,让模型知道"谁在前谁在后"**。

**RoPE vs Sinusoidal PE 的关键差异**:

Sinusoidal PE(原版 Transformer)把位置信息**加到 token embedding**上:`x' = x + PE(pos)`。问题是 PE 是绝对位置,attention score = `(x_i + PE(i)) · (x_j + PE(j))`,展开后含 `PE(i) · PE(j)`,这是绝对位置的函数,不是相对位置 `j-i` 的函数。模型在训练时见过 `j-i ∈ [0, 1024]`,推理时遇到 `j-i = 2000`,就完全 out-of-distribution。

**RoPE 的革命**:把位置信息**编进 Q、K 向量的旋转角度**——`RoPE(q, m) = q · e^{i m θ}`。attention score = `q · R(n-m) · k`,**只依赖相对位置 `n-m`**,模型天然能泛化到训练时没见过的长序列。这就是 LLaMA-3 训练 4k context、推理能推到 128k 的根本原因。

**RoPE 对 GenRec 的特殊价值**:推荐场景用户行为序列长度变化大(冷启动用户 5 条,活跃用户 1000 条),RoPE 让模型对"行为之间的相对距离"敏感,而不是"行为在序列中的绝对位置"——这更符合"用户最近的兴趣更重要"的业务直觉。

**为什么需要 RoPE?三种位置编码对比**:

| 方法 | 核心思想 | 外推能力 | 实现复杂度 |
| :--- | :--- | :--- | :--- |
| Sinusoidal PE(原版 Transformer) | 固定频率的正弦/余弦 | 有限 | 低 |
| 可学习 PE(LLaMA 1) | 一个 nn.Embedding(max_len, d) | **无**(超过 max_len 就不能推) | 最低 |
| **RoPE**(LLaMA 2/3, Mistral, Qwen) | 旋转 Q 和 K 向量 | **强**(可在更长序列上推理) | 中 |

**RoPE 的数学直觉**:
```text
给定位置 m 和 n 的两个向量 q_m 和 k_n,
标准注意力: score = q_m · k_n(不感知位置)
RoPE 注意力: score = (R_Θ^m · q_m) · (R_Θ^n · k_n) = q_m · R_Θ^{n-m} · k_n
其中 R_Θ^m 是一个旋转矩阵,旋转角度 = m · θ

关键性质:score 只依赖相对位置(n-m),不依赖绝对位置!
这就是 RoPE 能外推的本质原因
```

**RoPE 的高效实现(不需要真的构建旋转矩阵)**:
```python
class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码——和 LLaMA/Mistral 完全一致"""
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        # 频率: dim 个频段,从 base^0 到 base^(-dim/2+1)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        # 位置序列: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).float()
        # 外积: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 拼接成 [seq_len, dim](cos 和 sin 各一半)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()     # [seq_len, dim] × 2

def rotate_half(x):
    """将向量后半部分取反并与前半部分互换,实现 2D 旋转效果"""
    # x: [..., dim] → 分成两半,后半取负后前后交换
    x1, x2 = x.chunk(2, dim=-1)   # 各 [B, nh, T, hd//2]
    return torch.cat([-x2, x1], dim=-1)   # [B, nh, T, hd]

def apply_rotary_emb(xq, xk, cos, sin):
    """对 Q 和 K 应用 RoPE"""
    # xq: [B, nh, T, hd], cos/sin: [T, hd]
    # 把 x 分成两半,旋转后半并加回前半
    xq_rot = (xq * cos) + (rotate_half(xq) * sin)
    xk_rot = (xk * cos) + (rotate_half(xk) * sin)
    return xq_rot, xk_rot
```

---

**文字$\cdot$公式$\cdot$代码 2:为什么推荐场景必须用 RoPE,以及它怎么从"复杂数学"变成"5 行 PyTorch"**

**文字**:推荐场景的输入是用户历史行为序列,这个序列有严格的先后顺序,模型必须区分"用户 3 步前点了 iPhone"和"用户刚点了 iPhone"这两种语义。RoPE 的核心思想是把位置 m 编码成对 Q 向量的一个旋转:把 Q 当成复数,乘以 $e^{i m \theta}$。这个旋转角度只和相对位置 (n-m) 有关,attention score 算出来只依赖相对偏移,和绝对位置无关。这就是 RoPE 能"外推"的根本原因:训练在 4k context,推理推到 32k 甚至 128k,模型看到的相对位置角度依然是连续一致的。Sinusoidal PE 把位置信息加到 embedding 上,没有相对位置性质,所以超过训练长度就崩。可学习 PE(LLaMA 1 用过)甚至更糟,词表外位置直接是随机初始化。GenRec 用 RoPE 的第二个理由是它没有任何可训练参数(频率是 base=10000 算出来的),对冷启动友好;第三个理由是它和 FlashAttention、GQA 完全兼容,只对 Q、K 做 element-wise 旋转,不影响后续的并行性。LLaMA 2/3、Mistral、Qwen、DeepSeek 全部从可学习 PE 切换到 RoPE,GenRec 里的 HSTU、OneRec 同样如此。

**公式**:RoPE 的复数旋转直觉,一句话:

$$\text{RoPE}(q, m) = q \cdot e^{i m \theta}, \quad e^{i\theta} = \cos\theta + i\sin\theta$$

展开后得到 attention score 的相对位置性质:

$$\langle \text{RoPE}(q, m),\, \text{RoPE}(k, n) \rangle = \langle q,\, R_{\Theta}^{n-m} \cdot k \rangle \quad\text{(只依赖 } n-m \text{)}$$

**代码**:

```python
def apply_rope(q, k, cos, sin):
    # q, k: [B, H, T, D]; cos, sin: [T, D] 预计算好的旋转因子
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)         # 把最后一维切成两半
        return torch.cat([-x2, x1], dim=-1)  # 后半取负再拼到前半,等价于复数 i·x

    # (q · e^{i mθ}) = q · cos(mθ) + rotate_half(q) · sin(mθ)
    q_rot = q * cos + rotate_half(q) * sin  # Q 应用位置旋转
    k_rot = k * cos + rotate_half(k) * sin  # K 应用位置旋转
    return q_rot, k_rot                      # 喂给 scaled_dot_product_attention

# 预计算 inv_freq(只算一次,缓存到 buffer)
inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
t = torch.arange(seq_len).float()
freqs = torch.outer(t, inv_freq)            # [T, head_dim/2]
cos, sin = freqs.cos(), freqs.sin()         # 广播到每个 head 和 batch
```

---

#### 16.2.4 GQA/MQA——注意力变体的显存节省

**GQA 解决的是"推理时 KV cache 爆显存"的问题**。

**为什么 KV cache 这么重要?**

LLM 推理是"自回归"——生成第 1 个 token → 生成第 2 个 token(把第 1 个 token 拼回去)→ 生成第 3 个 token(把第 1、2 个 token 拼回去)……每生成一个新 token,要把前面所有 token 的 K、V 重新算一遍。**没有 KV cache,每步都要重算所有历史 token 的 K、V**,4096 context 推理第 4096 个 token 要算 4096² ≈ 1600 万次乘法——完全不可行。

**KV cache 的工作原理**:把每步算好的 K、V 缓存起来,下一步直接复用。每层每 token 缓存 2 个向量(K、V),每个向量维度 = `n_kv_heads × head_dim`。**显存占用 = 层数 × 序列长度 × n_kv_heads × head_dim × 2 bytes**。

**GQA 的核心洞察**:"每个 Q 头都需要独立的 K、V"是个过强假设。**32 个 Q 头看到 32 组独立的 K、V,信息确实丰富,但其实只要 4-8 组 K、V 就能让模型学会不同注意力模式**。把 32 个 Q 头分成 4 组,每组共享 1 套 K、V,**KV cache 缩 4-8 倍**,质量损失几乎可忽略。

```python
class GroupedQueryAttention(nn.Module):
    """
    GQA(分组查询注意力):MHA 和 MQA 的中间态。

    以 LLaMA 3 8B 为例:
    - n_heads = 32(查询头)
    - n_kv_heads = 8(键值头,每 4 个查询共享 1 组 K,V)
    - 每个 KV head 对应 4 个 Q head

    显存节省:
    相比 MHA(32 KV heads):KV cache 减少 4×
    相比 MQA(1 KV head):质量更高(因为 KV 头更多)
    """
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        # Q:每个头独立(全量投影)
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        # K,V:只有 n_kv_heads 组(小于 n_heads)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # GQA:把 KV 头 repeat 到 Q 头数
        # torch 的 repeat_kv 函数
        n_groups = self.n_heads // self.n_kv_heads
        xk = xk.unsqueeze(3).expand(-1, -1, -1, n_groups, -1).reshape(B, T, self.n_heads, self.head_dim)
        xv = xv.unsqueeze(3).expand(-1, -1, -1, n_groups, -1).reshape(B, T, self.n_heads, self.head_dim)

        # 标准 attention
        scores = torch.matmul(xq.transpose(1,2), xk.transpose(1,2).transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, xv.transpose(1,2)).transpose(1,2).reshape(B, T, -1)
        return self.wo(out)

# GQA 参数配置对比(当前主流模型):
# LLaMA 3 8B:   n_heads=32, n_kv_heads=8,   ratio=4:1
# Mistral 7B:   n_heads=32, n_kv_heads=8,   ratio=4:1
# Qwen 2 7B:    n_heads=28, n_kv_heads=4,   ratio=7:1
# DeepSeek V2:  n_heads=128, n_kv_heads=16, ratio=8:1
```

**显存对比表(以 LLaMA 3 8B 为例:32 层,4096 上下文,B=1)**:

| 方法 | KV cache 每层 | 单序列总 KV cache | 节省比例 |
| :--- | :--- | :--- | :--- |
| MHA(32 KV heads) | 32×4096×128×2B=**33.6MB** | 32×33.6MB=**1.05GB** | 基线 |
| GQA-8(8 KV heads) | 8×4096×128×2B=**8.4MB** | 32×8.4MB=**268MB** | **4×** |
| MQA(1 KV head) | 1×4096×128×2B=**1.05MB** | 32×1.05MB=**33.6MB** | **32×** |

在 GenRec 中,GQA 同样有效:**推理时 beam search 多个候选 → 每个候选需单独 KV cache → GQA 省 4 倍显存意味着可以同时处理 4 倍多的候选**。

---

**文字$\cdot$公式$\cdot$代码 3:为什么 GQA 是 MHA 和 MQA 之间的甜区,以及 GenRec 推理为什么离不开它**

**文字**:MHA(32 个 KV head,每个 query head 独立一组 K, V)在 LLaMA-3 8B 上每层 33.6MB KV cache,4096 context 下 32 层 = 1.05GB,一旦 batch=64 直接 67GB,单卡根本装不下。MQA(1 个 KV head,所有 query 共享一组 K, V)反过来把显存省到 1.05MB 每层、33.6MB 总计,但 32 个 query head 看到完全一样的 K, V,注意力模式被强行平均,PPL 通常掉 1~2 个点。GQA 把 32 个 query head 分成 8 组,每 4 个 query 共享 1 组 K, V,显存省 4$\times$ (8.4MB 每层),同时保留了 8 倍的注意力多样性,质量损失几乎可忽略。这个 4:1 ~ 8:1 的"每 KV 头服务的 Q 头数"是经验甜区,LLaMA 2/3、Mistral、Qwen 2、DeepSeek-V2 全在这个区间。GenRec 推理时这个收益直接转成钱:同样一张 A100,GQA-8 能服务的并发用户数是 MHA 的 4 倍,因为每个并发请求都要存自己的 KV cache。冷启动场景下,beam search 跑 16 个候选时,只有 GQA 能让多个候选共享大部分 KV cache。

**公式**:GQA 的 KV cache 显存(per sequence, per layer):

$$M_{\text{KV}}^{\text{GQA}} = 2 \cdot G \cdot T \cdot d_h, \quad G = n_{\text{kv\_heads}}$$

对比 MHA($G = H = 32$ head)和 MQA($G = 1$ head),GQA 选 $G = H / r$,其中 $r$ 是"每 KV 服务的 Q 头数",推荐 4 ≤ r ≤ 8。

| 方案 | $G$ | 每层 KV(4096 ctx) | 32 层总计 | 节省 |
| :--- | :--- | :--- | :--- | :--- |
| MHA | 32 | 33.6 MB | 1.05 GB | 1× |
| **GQA-8** | **8** | **8.4 MB** | **268 MB** | **4×** |
| MQA | 1 | 1.05 MB | 33.6 MB | 32× |

**代码**:

```python
class GroupedQueryAttention(nn.Module):
    """GQA:G 个 KV 头被 H 个 Q 头共享,显存省 4~8×,质量几乎不掉"""
    def __init__(self, d_model, n_heads=32, n_kv_heads=8):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        # Q:32 个头全量投影;K, V:只 8 个头(省 4× 显存)
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # 把 8 个 KV head 复制到 32 个 Q head,每 4 个 Q 共享一组 KV
        n_rep = self.n_heads // self.n_kv_heads  # = 4(LLaMA-3 配置)
        xk = xk.repeat_interleave(n_rep, dim=1)  # [B, 32, T, D]
        xv = xv.repeat_interleave(n_rep, dim=1)
        return self.wo(F.scaled_dot_product_attention(xq, xk, xv).transpose(1, 2).reshape(B, T, -1))
```

---

#### 16.2.5 Decoder-Only vs Encoder-Decoder vs Encoder-Only

**三者在 GenRec 中的对比**:

```text
Encoder-Only(如 BERT, SASRec)
  [x₁, x₂, ..., x_T] → Self-Attention → [h₁, h₂, ..., h_T]
  优点:双向注意力,每个 token 看到全部上下文
  缺点:不能生成(不能做自回归推荐)
  GenRec 用法:作为序列编码器 → 再接 MLP 预测 CTR

Encoder-Decoder(如 T5, P5)
  Encoder:  [x₁, x₂, ..., x_T] → Encoder → 全向表示
  Decoder:  [y₁, y₂, ..., y_K] → Decoder(交叉注意+自注意) → 生成 token
  优点:Encoder 看到全部输入,Decoder 做生成
  缺点:解码慢(两阶段),参数量大(两套)
  GenRec 用法:P5/TIGER 都用这个,Encoder 读历史,Decoder 生成候选

Decoder-Only(如 GPT, LLaMA, HSTU)
  [x₁, x₂, ..., x_T] → Causal Attention → [h₁, h₂, ..., h_T]
  每个 token 只能看到自己和左边的 token
  优点:生成就是推理(KV-cache 加速),参数量单一
  缺点:编码时也是单向的(不能看到未来的 token)
  GenRec 用法:HSTU,以及大多 2024-2025 新模型
```

**为什么 Decoder-Only 在 GenRec 和 LLM 中都赢了?**

```text
1. 推理效率:Decoder-Only 支持 KV-cache
   encoder-decoder 的交叉注意力不能 KV-cache → 每次生成都要重新编码
   decoder-only 的自注意力可以 KV-cache → 生成 O(1) 每步
   在推荐场景:对 1000 个候选打分,D-only 比 E-D 快 3~5 倍

2. 参数效率:同一套参数同时做"理解"和"生成"
   E-D 有两套参数(编码器+解码器),D-only 只有一套
   在有限预算下,D-only 的 "layer per parameter" 利用率更高

3. 训练灵活:支持 Causal LM 和 Prefix LM 两种模式
   训练时:"用户历史行为 → 下一个 item" = 标准的 Causal LM
   推理时:KV-cache 加速 = 标准 LLM 推理

4. 生态兼容:和 LLaMA/Mistral/Qwen 共用训练框架
   你不需要维护两套代码(一个做生成,一个做推荐)
   统一的:训练脚本、推理引擎、量化工具
```

### 16.3 Tokenization:从 LabelEncode 到语义 ID(VQ-VAE / RQ-VAE)

> **这是 GenRec 和传统推荐最本质的区别,也是从 CV/搜推广到 GenRec 必须跨越的一道坎。**

#### 16.3.1 问题的规模:千万级物品的 Token 化

**Tokenization 是 GenRec 区别于传统推荐的核心**。

**传统推荐的"物品表示"**:
- item_id = 1234567 → `Embedding(num_items, 64)[1234567]` → 64 维向量
- 优点:每个物品有独立 embedding
- 缺点:**1) 词表 = 物品数 = 千万级,softmax 算不动;2) 相似物品的 ID 不相近(1234567 和 1234568 完全不相似);3) 新物品 OOV,冷启动灾难**

**GenRec 的"物品表示"思路**:
- item_1234567 → 多个 token → 词表只有 256
- 优点:**1) 词表小,softmax 可行;2) 相似物品的 token 序列前缀相同;3) 新物品用 RQ-VAE 编码即可,无需重新训练**

**为什么"词表小"如此重要?**

LLM 的一个核心瓶颈是"输出层的 softmax"——预测下一个 token 要在 32K-100K 词表上算 softmax。传统推荐如果直接把 item_id 当 token,词表是 1000 万——softmax 要做 1000 万次 exp,forward 和 backward 都不可能。**降到 256 词表,softmax 计算量降 40000 倍**,这是 GenRec "算得动" 的根本原因。

**公式**：语义 ID 的可扩展性——用 $L$ 个 code、每个 codebook 大小 $K$ 来覆盖 $N$ 个物品：

$$N \leq K^L \quad \Rightarrow \quad L \geq \frac{\log N}{\log K}$$

$$\text{例如: } K=256,\ L=4 \Rightarrow K^L = 256^4 \approx 4.3 \times 10^9 \gg 10^7 \text{(千万物品)}$$

softmax 在 $K$ 类上而非 $N$ 类上：计算量从 $O(N)$ 降为 $O(K) = O(256)$，快约 $N/K \approx 40000$ 倍。

```text
传统推荐:
  item_id = 1234567
  → Embedding 表查找: emb_table[1234567] → [32维向量]
  √ 解决:词表=所有物品(千万级)
  × 限制:每个物品独立,学不到"相似物品"

GenRec 的 Token 化(直接做法——P5):
  item_id = 1234567 → 当成 1 个 token → 词表大小 = 10,000,000
  × 不可行!softmax 在 1000 万类上无法计算!

GenRec 的 Token 化(语义 ID——TIGER/CIDR):
  item_1234567 → VQ-VAE 编码器 → [42, 15, 7, 3] ← 4 个 token
  词表大小 = 256(每个 token 只有 0~255 共 256 种取值)
  总组合数 = 256^4 = 4,294,967,296(43 亿,覆盖千万物品绰绰有余)
  softmax 也只预测 256 类!完全可行!
```

#### 16.3.2 VQ-VAE vs RQ-VAE:语义 ID 的技术核心

**VQ-VAE 的核心思想:把连续向量映射到离散 codebook**。

VQ-VAE(Vanilla Quantization VAE)的算法:
1. 编码器把 item 的内容特征(图/文/属性)压缩成连续向量 `z_e`
2. 在 codebook(256 个"代表向量")里找最接近 `z_e` 的那个,记为 `c_{k*}`
3. 用 `c_{k*}` 重建 `z_e`,得到 `z_q`
4. 训练目标:`min ‖x - x̂‖²` (重建) + `min ‖z_e - z_q‖²` (commitment,让编码器输出靠近 codebook)

**问题:1 个 token 表示 1 个物品 → 256 种 code 只能表示 256 个物品**。工业界 1000 万物品直接破产。

**RQ-VAE 的关键改进:残差量化 + 层级语义**。

RQ-VAE(Residual Quantization VAE)分多层量化:
- 第 1 层:量化原始 `z_e` → code_1,残差 `r_1 = z_e - c_{code_1}`
- 第 2 层:量化残差 `r_1` → code_2,残差 `r_2 = r_1 - c_{code_2}`
- 第 3 层:量化 `r_2` → code_3
- 第 4 层:量化 `r_3` → code_4
- 语义 ID = `[code_1, code_2, code_3, code_4]`

**数学保证**:L 层 × K 码表 = K^L 总组合数。L=4, K=256 → 256^4 = 43 亿,覆盖千万物品绰绰有余。**层级结构还带来"层次语义"**——code_1 相近 = 同一品类,code_1+code_2 相近 = 同品类同品牌,符合人类认知的层级性。

**工业优势**:
- code_1 = 粗粒度(品类) → 学"用户喜欢什么大类"
- code_2 = 品牌 → 学"用户偏好哪个品牌"
- code_3 = 型号 → 学"用户对哪款感兴趣"
- code_4 = SKU → 区分"具体规格"

这种"先选大类别再细化"的推理路径,让模型学得更快、泛化更好。

**VQ-VAE 的数学公式**:

$$z_e = \text{Encoder}(x) \in \mathbb{R}^D \quad \text{(编码器输出连续向量)}$$

$$k^* = \arg\min_{k} \|z_e - c_k\|_2^2, \quad c_k \in \mathcal{C} = \{c_1, \ldots, c_K\} \quad \text{(找最近邻 codebook 向量)}$$

$$z_q = c_{k^*} \quad \text{(量化向量,前向直接跳过,反向 STE 直接传梯度)}$$

$$\mathcal{L}_{\text{VQ}} = \underbrace{\|x - \hat{x}\|^2}_{\text{重建损失}} + \underbrace{\|z_e - \text{sg}(z_q)\|^2}_{\text{commitment loss}} + \underbrace{\beta\|\text{sg}(z_e) - z_q\|^2}_{\text{codebook loss}}$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient（不传梯度），STE（直通估计器）：$\nabla_{z_e} \mathcal{L} \approx \nabla_{z_q} \mathcal{L}$（把 $z_q$ 的梯度直接传给 $z_e$）。

**RQ-VAE（残差量化 VAE）**:

$$r_0 = z_e, \quad k_l^* = \arg\min_k \|r_{l-1} - c_k^{(l)}\|_2^2, \quad r_l = r_{l-1} - c_{k_l^*}^{(l)}$$

$$\text{语义 ID} = [k_1^*, k_2^*, \ldots, k_L^*], \quad \hat{z}_e = \sum_{l=1}^L c_{k_l^*}^{(l)}$$

L 层量化后，总组合数 = $K^L$（如 $K=256, L=4$ → $256^4 = 4.3 \times 10^9$，覆盖所有千万级物品）。

**VQ-VAE(向量量化变分自编码器)**:
```text
输入: item 内容特征(文本/图片/ID)→ 编码器 → z(连续向量)
量化: z → 找最近的 codebook 向量 → code_idx(一个整数)
输出: code_idx(即"语义 ID")

问题:每个物品只有一个 code → 256 种组合最多表示 256 个物品
     要表示 1000 万物品 → 词表要 1000 万 → 回到原问题!
```

**RQ-VAE(残差量化 VAE)——TIGER/CIDR 的方案**:
```text
输入: item 内容特征 → 编码器 → z(连续向量)

第 1 层量化: z → 找最近邻 codebook_1[code_1] → 残差 r₁ = z - codebook_1[code_1]
第 2 层量化: r₁ → 找最近邻 codebook_2[code_2] → 残差 r₂ = r₁ - codebook_2[code_2]
第 3 层量化: r₂ → 找最近邻 codebook_3[code_3] → 残差 r₃ = r₂ - codebook_3[code_3]
第 4 层量化: r₃ → 找最近邻 codebook_4[code_4]

语义 ID = [code_1, code_2, code_3, code_4]

每层 codebook = 256 个向量 → 总组合 = 256^4 = 43 亿
层级的物理含义:
  code_1:品类级别(粗粒度)
  code_2:品牌级别
  code_3:具体型号
  code_4:具体 SKU(细粒度)
```

**RQ-VAE 的完整训练代码**:
```python
class ResidualQuantizer(nn.Module):
    """
    RQ-VAE 残差量化器。TIGER/CIDR 等模型的 Tokenization 核心。

    输入: item 的内容特征(如 Sentence-T5 encoder 输出)
    输出: 语义 ID 列表 [code_1, code_2, code_3, code_4]
    """
    def __init__(self, num_levels=4, num_codes=256, dim=64):
        super().__init__()
        self.num_levels = num_levels
        self.num_codes = num_codes

        # 每层独立的 codebook,size=[num_codes, dim]
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_codes, dim) * 0.01)
            for _ in range(num_levels)
        ])

    def forward(self, z, training=True):
        """
        z: [B, D] — item 的连续特征向量
        返回:
            codes: [B, L] — 语义 ID(training 时用 straight-through)
            z_hat: [B, D] — 重建向量
        """
        residual = z
        codes = []
        total_loss = 0

        for level in range(self.num_levels):
            # 计算到 codebook 的距离: [B, num_codes]
            dist = torch.cdist(residual.unsqueeze(0), self.codebooks[level].unsqueeze(0)).squeeze(0)
            code_idx = dist.argmin(dim=-1)          # [B],每个 sample 的最近邻 ID

            if training:
                # Straight-Through:前向用量化,反向直接过(可微近似)
                chosen = F.embedding(code_idx, self.codebooks[level])
                # 加上"停止梯度"的残差,只更新 codebook
                z_hat_level = residual + (chosen - residual).detach()
                # commit loss:让编码器输出靠近 codebook
                total_loss += F.mse_loss(chosen, residual.detach()) * 0.25
            else:
                chosen = F.embedding(code_idx, self.codebooks[level])
                z_hat_level = chosen

            codes.append(code_idx)
            residual = residual - chosen             # 计算残差

        z_hat = z - residual                          # 重建 = 所有层 sum

        if training:
            total_loss += F.mse_loss(z_hat, z.detach())  # 重建损失

        return torch.stack(codes, dim=-1), z_hat, total_loss

    def encode(self, z):
        """推理时:只输出语义 ID"""
        with torch.no_grad():
            codes, _, _ = self.forward(z, training=False)
        return codes                                   # [B, L]

# ============ 在 GenRec 中使用语义 ID ============
# 预训练 RQ-VAE
quantizer = ResidualQuantizer(num_levels=4, num_codes=256, dim=64)
# item_content: [B, D] — 用 Sentence-T5/BERT 提取的物品文本特征
semantic_ids, _, loss = quantizer(item_content_features)
print(semantic_ids[0])   # [42, 15, 7, 3] ← item_0 的 4 级语义 ID

# 把语义 ID 当成 token 序列输入到 GenRec 模型
class GenRecWithSemanticID(nn.Module):
    """
    用语义 ID 做 Token 的生成式推荐模型。
    输入不再是"一个物品一个 token",而是"一个物品 4 个 token"。
    """
    def __init__(self, num_codes=256, num_levels=4, d_model=512):
        super().__init__()
        # 每层共享同一个 Embedding 表
        self.code_embedding = nn.Embedding(num_codes, d_model)
        self.num_levels = num_levels
        self.transformer = ...   # Decoder-Only Transformer

    def forward(self, semantic_ids):
        # semantic_ids: [B, T, L] — T 个物品,每个物品 L=4 级语义 ID
        B, T, L = semantic_ids.shape
        # 展平:[B, T*L]
        flat_ids = semantic_ids.view(B, T * L)
        # Embedding:[B, T*L, D]
        emb = self.code_embedding(flat_ids)
        # Transformer
        out = self.transformer(emb)
        # 输出:预测下一个语义 token,在每级的 codebook 上做 softmax
        logits = self.output(out)  # [B, T*L, num_codes]
        return logits
```

#### 16.3.3 Tokenization 策略选型指南

| 应用场景 | 推荐策略 | 词表大小 | 冷启动能力 |
| :--- | :--- | :--- | :--- |
| 物品 < 10 万 | 直接 ID token | = 物品数 | ❌ 无 |
| 物品 10 万 ~ 1000 万 | **语义 ID(RQ-VAE, 4 级)** | 256~1024 | ✅ ✅ |
| 物品 > 1000 万 | 语义 ID + 哈希分桶 | 256~1024 + 额外信号 | ✅ ✅ |
| 只需要文本描述 | BPE/Unigram 子词 token | 32K~100K | ✅(零样本) |
| 多模态(文本+图片) | **多模态 RQ-VAE** | 256~1024 | ✅ ✅(最强的) |

### 16.4 PEFT 与量化:如何在小模型上做高效适配

> **在 GenRec 中,你可能需要针对不同领域、不同客户、不同任务分别适配模型。全量微调 30B 模型不现实。LoRA 和量化让你的消费级 GPU 也能微调。**

#### 16.4.1 LoRA 的数学——低秩适配

**PEFT(Parameter-Efficient Fine-Tuning)解决"大模型微调成本爆炸"的问题**。

**全量微调的成本**:

微调一个 7B 模型:
- 权重本身:7B × 2 bytes (FP16) = 14 GB
- 梯度:7B × 2 bytes = 14 GB
- 优化器状态 (Adam):7B × 8 bytes (FP32 动量 + 二阶矩) = 56 GB
- **总显存 ≈ 84 GB,单张 A100 80GB 装不下**

更别提 70B、175B 模型——单卡根本不可能全量微调。

**LoRA 的核心洞察**:模型微调时,权重的变化 $\Delta W$ 实际上是一个**低秩矩阵**——重要的"任务特定信息"集中在一个低维子空间里,不需要改全部参数。

**LoRA 的数学**:

把 $\Delta W$ 分解为两个小矩阵的乘积 $\Delta W = B \cdot A$,其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$,rank $r \ll \min(d, k)$。

- 原始 Linear: `h = W_0 x` ,W_0 冻结
- LoRA Linear: `h = W_0 x + (B A) x`,B、A 可训练
- 参数量:原 `d × k` → 现 `d × r + r × k = r(d+k)`,节省 `min(d,k)/r` 倍

**r=8 时,7B 模型的 LoRA 参数 = 7B × 16/4096 ≈ 28M,不到全量的 0.4%**。训练时只需要存 B、A 的梯度和优化器状态(几十 MB),不用存 7B 权重的梯度(几十 GB)。**单张消费级 4090 24GB 就能微调 7B 模型**。

**文字**:全量微调一个 7B 参数的模型需要 7B $\times$ 16 bytes（FP32 + Adam）$\approx$ 112 GB GPU 内存，而 LoRA 通过冻结原始权重 $W$，只训练一对低秩矩阵 $B, A$，将可训练参数减少 99%+。

**公式**:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

$$W_0 \in \mathbb{R}^{d \times k},\quad A \in \mathbb{R}^{r \times k},\quad B \in \mathbb{R}^{d \times r},\quad r \ll \min(d, k)$$

$$\text{可训练参数量:} \quad r \times (d + k) \quad \text{vs 全量} \quad d \times k \quad \text{(节省 } \approx \min(d,k)/r \text{ 倍)}$$

**缩放比例**（防止初始化时 $\Delta W$ 过大干扰原始权重）：

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx, \quad \text{通常设 } \alpha = 2r$$

**推理时合并**（无额外延迟）：

$$W_{\text{merge}} = W_0 + \frac{\alpha}{r} \cdot BA \quad \text{(一次性矩阵加法,之后推理与原始线性层完全相同)}$$

```python
class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation。

    数学: W' = W + BA 其中 B∈ℝ^{d×r}, A∈ℝ^{r×k}, rank=r << min(d,k)
    forward: h = Wx + (BA)x

    在训练时:W 冻结(不需要梯度),只更新 A 和 B
    在推理时:可以 merge → W_merged = W + α·BA(无额外计算)

    核心超参:
    - rank(r):越大越强但也越贵,推荐 8~64
    - alpha(α):缩放因子,通常 α = 2r
    - target_modules:在哪些层上插 LoRA
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 原始权重(冻结)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA 低秩适配器
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # 原始路径(冻结)
        base_out = F.linear(x, self.weight)
        # LoRA 路径(可训练)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * (self.alpha / self.rank)
        return base_out + lora_out

# 使用
original_linear = nn.Linear(512, 512)
lora_linear = LoRALinear(512, 512, rank=16, alpha=32)

# 训练时:只优化 LoRA 参数
optimizer = torch.optim.AdamW(lora_linear.lora_A.parameters(), lr=1e-4)
# 原始权重不更新

# 推理时:merge(消除额外计算)
lora_linear.weight.data += (lora_linear.lora_B @ lora_linear.lora_A) * (32 / 16)
# 之后 lora_linear.forward 就可以只用 base 路径了
```

**LoRA 在 GenRec 中的应用**:

| 场景 | LoRA 配置 | 训练数据 | 效果 |
| :--- | :--- | :--- | :--- |
| 跨域适配(电商→视频) | rank=16, Q+V 投影 | 10 万样本 | AUC +0.03 |
| 时序漂移校正 | rank=8, attention 全量 | 100 万新样本 | AUC +0.015 |
| 个性化(每个客户) | rank=4, shared base+per-customer LoRA | 1 万/客户 | 可接受 |
| 新任务(CTR→CVR) | rank=32, 全层 | 50 万样本 | AUC +0.025 |

---

**文字$\cdot$公式$\cdot$代码 4:为什么 GenRec 必须用 LoRA / QLoRA 微调 7B+,以及它怎么把 112GB 显存压到 4GB**

**文字**:全量微调一个 7B 模型需要 `7B \times 16 bytes`(FP32 权重 + Adam 一阶/二阶矩)$\approx 112$ GB 显存,单张消费级 GPU 根本装不下,即使 A100 80GB 也得开 DeepSpeed ZeRO。LoRA 假设权重更新是低秩的,把 $\Delta W$ 拆成两个小矩阵的乘积 B$\cdot A$,只训这两个小矩阵,W 保持冻结。一层 Linear 原本 4096$\times 4096$ = 16M 参数,LoRA(r=8)只剩 4096$\times 8$ + $8\times 4096$ = 65K 参数,直接省 250$\times$。QLoRA 进一步把冻结的 W 用 NF4(4-bit NormalFloat)量化,7B 模型的权重从 FP16 的 14GB 压到 3.5GB,加上 LoRA 适配器本身的几百 MB,总显存约 4GB,塞进单张 24GB 消费级显卡还有富余。为什么低秩假设在 GenRec 里也成立?因为从基础 GenRec 模型迁移到具体业务(电商$\to$ 视频、母婴$\to$ 3C),权重变化主要集中在"业务特定的特征交互模式"上,这是一个相对低维的子空间,r=8~32 足够覆盖。GenRec 还有一个独特优势:可以给一个 7B 基础模型挂 50 个小 LoRA(r=4)适配 50 个垂直业务,基础模型只存一份,适配器总开销 < 1GB,完美支持"模型即服务"的多租户场景。

**公式**:

$$h = W_0 x + \Delta W x = W_0 x + B A x, \quad W_0 \in \mathbb{R}^{d \times k},\; B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)$$

$$\text{可训练参数量:}\ r(d + k) \quad \text{vs 全量}\ d \times k, \quad \text{节省} \approx \frac{\min(d,k)}{r} \text{ 倍}$$

QLoRA 的显存(7B 模型 + r=16 LoRA):

$$M_{\text{QLoRA}} = 7\text{B} \times 0.5\ \text{bytes (NF4)} + r(d+k) \times L \times 2\ \text{bytes} \approx 3.5\ \text{GB} + \varepsilon$$

**代码**:

```python
class LoRALinear(nn.Module):
    """LoRA:冻结 W,只训低秩 B·A,r=8~32 是经验甜区"""
    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)  # 冻结
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)                        # 高斯小初始化
        self.B = nn.Parameter(torch.zeros(out_dim, r))                              # 0 初始化 → 训练初期 ΔW=0
        self.scale = alpha / r  # 默认 α=2r,控制 LoRA 贡献幅度

    def forward(self, x):
        # W 路径(冻结)+ LoRA 路径(可训),训练时只更新 A, B
        return F.linear(x, self.W) + F.linear(F.linear(x, self.A), self.B) * self.scale

# QLoRA:用 bitsandbytes 把 base 模型 4-bit 加载,LoRA 保持 FP16
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")  # NF4 量化
model = AutoModel.from_pretrained("llama-7b", quantization_config=bnb)  # 7B → 3.5GB
# 训练:7B 总参数中只更新 < 0.1% 的 LoRA 参数
optim = torch.optim.AdamW([p for n, p in model.named_parameters() if "lora_" in n], lr=1e-4)
```

---

#### 16.4.2 QLoRA——4-bit 量化 LoRA

**QLoRA 是 LoRA 的"显存极致版"**。

LoRA 解决了"训练参数太多"的问题,但**基础模型的 7B 权重还是要 FP16 加载**,占 14GB。QLoRA 进一步把基础权重量化到 **NF4(4-bit NormalFloat)**,7B 模型从 14GB 压到 **3.5GB**,加上 LoRA 适配器的几百 MB,总显存 **~4GB**——**单张消费级 24GB 显卡(4090/3090)就能微调 7B 模型**。

**NF4 的核心创新**:

普通 INT4 量化把权重空间切成 16 个等距区间,问题是正态分布的权重大部分集中在 0 附近,等距量化在尾部"浪费"精度。**NF4 选择 16 个分位点对应 $\mathcal{N}(0,1)$ 的累积分布函数**——$\{q_1, ..., q_{16}\} = \{\Phi^{-1}(i/17), i=1..16\}$,这样大部分分位点集中在 0 附近,稀疏的尾部用更稀疏的区间覆盖。**NF4 对正态分布权重的量化误差是 INT4 的 1/3**。

**QLoRA 三个关键技术**:
1. **NF4 量化**基础权重 → 4-bit
2. **双重量化(Double Quant)**:把量化用的 scale 也用 INT8 量化,再省 0.4GB/7B
3. **分页优化器(Paged Optimizer)**:用 NVIDIA 的统一内存,optimizer 状态部分卸载到 CPU,防止 OOM

**GenRec 的 QLoRA 实战**:7B 基础 GenRec 模型(预训练在 1 亿用户行为上) → QLoRA 微调在特定业务(母婴 / 3C / 美妆)上,10 万样本训 1 epoch,显存占用 < 8GB,AUC 比全量微调低 0.005 但开发效率高 10 倍。

QLoRA 的显存公式（以 7B 模型为例）:

$$M_{\text{FP16}} = 7\text{B} \times 2\text{ bytes} = 14\text{GB}$$

$$M_{\text{QLoRA}} = 7\text{B} \times 0.5\text{ bytes (NF4)} + \underbrace{r(d+k) \times \text{层数}}_{\text{LoRA FP16, << 1GB}} \approx 3.5\text{GB} + \varepsilon$$

NF4（NormalFloat4）量化策略：$2^4=16$ 个量化值选自 $\mathcal{N}(0,1)$ 的等概率分位数，比等距量化对正态分布权重的量化误差更小：

$$\{q_1, \ldots, q_{16}\} = \left\{\Phi^{-1}\!\left(\frac{i}{17}\right) \mid i = 1,\ldots,16\right\}$$

```python
# QLoRA = Quantized LoRA = NF4 量化 + LoRA
# 关键:把预训练权重量化到 4-bit(NF4),LoRA 适配器保持 FP16

# 使用 bitsandbytes 实现
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 双重量化(再量化 scale)
)

# 加载 4-bit 模型
model = AutoModel.from_pretrained("llama-7b", quantization_config=bnb_config)

# 插入 LoRA(保持 FP16)
from peft import get_peft_model, LoraConfig
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# 训练时:4-bit 权重不动,只更新 LoRA 参数(FP16)
# 显存节省:70B 模型从 140GB → 48GB → 1 张 A100 可训!

# QLoRA 配置选择(经验):
# rank=32: 质量最高,训练略慢
# rank=16: 质量/速度平衡(推荐)
# rank=8:  快速实验,质量可能有损
# target:  Q 和 V 最重要,再加 O 和 Up/Down 更好
```

#### 16.4.3 量化推理(GPTQ / AWQ / GGUF)

**推理量化和训练量化的目的完全不同**:

- **QLoRA**(训练时量化):为了**省显存能训练**
- **GPTQ/AWQ**(推理时量化):为了**省显存能推理,降延迟、提吞吐**

**GPTQ(GPT Quantization)的核心思想**:

逐层量化,把每层的量化误差视为"可优化问题"。具体做法:
1. 对一层权重 $W \in \mathbb{R}^{d \times k}$,量化目标是最小化 $\min_Q \|(W - Q) X\|_F^2$,其中 $X$ 是校准数据(从训练集采 128 条)
2. 逐列量化,每量化一列,用 Hessian 矩阵的逆更新"误差补偿"到未量化列
3. INT4 量化下,质量损失 < 1%

**AWQ(Activation-Aware Quantization)的洞察**:

观察发现:**~1% 的"重要权重"贡献了 ~60% 的量化误差**。这些"重要权重"的特点是:对应的激活值(activation)特别大。如果直接量化,这些权重的小误差被放大成激活值的大误差。**AWQ 保护重要权重的做法**:先 scale up 重要权重(× 缩放因子 s),量化,再 scale down 回来。数学上:`W_Q = Q(W · s⁻¹) · s`,s 由激活值的 scale 决定。

**AWQ vs GPTQ 的对比**:
- GPTQ:用 Hessian 精确补偿误差,但慢(每层要解线性方程)
- AWQ:用经验方法(scale + quantize),但快,质量略好
- **GenRec 推荐 AWQ**:GenRec 推理量大(每天 1 亿次推理),AWQ 的速度优势更值钱

**GGUF 量化(CPU 推理,llama.cpp 生态)**:
- Q2_K ~ Q8_0 多种级别,按"质量/显存"trade-off 选
- **Q4_K_M 是甜区**:质量损失 ~0.5%,显存省 75%(7B 模型从 14GB → 4GB)
- 适合边缘部署(无 GPU 的服务器)

**文字**：推理量化将模型权重从 FP16（2 bytes）压缩到 INT4（0.5 bytes），理论上降低 4 倍显存。关键挑战是量化误差尽量小。

**公式**：GPTQ/AWQ 的目标函数均为最小化层输出误差（layer-wise quantization error）：

$$\min_{W_Q} \|W \cdot X - W_Q \cdot X\|_F^2$$

AWQ 发现约 1% 的权重贡献了约 60% 的量化误差，通过 **per-channel scaling** 保护关键权重：

$$W_Q = Q(W \cdot s^{-1}) \cdot s, \quad s = \text{activation scale} \quad \text{(rescale back in inference)}$$

```text
推理量化 vs 训练量化:
  QLoRA:        训练时量化(4-bit),为了省显存能训练
  GPTQ/AWQ:     推理时量化(4-bit / 8-bit),为了省显存能推理

GPTQ(Post-Training Quantization):
  逐层优化:对每层权重,用 Hessian 矩阵做敏感度分析
  先量化最不敏感的权重,用误差补偿其他权重
  效果:INT4 下质量损失 < 1%

AWQ(Activation-Aware Quantization):
  观察激活值:发现 ~1% 的权重贡献了 ~60% 的量化误差
  保护这些"关键权重":不对它们做粗暴量化,而是 scale up
  效果:INT4 下质量损失 < 0.5%

GGUF(CPU 推理:llama.cpp):
  多种级别:Q2_K ~ Q8_0
  Q4_K_M 是甜区:质量损失 ~0.5%,显存省 75%
```

```python
# AWQ 的使用(一行代码)
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("model_path")
model.quantize(tokenizer, quant_config={"bits": 4, "group_size": 128})
model.save_quantized("model_awq_int4")
```

#### 16.4.4 知识蒸馏:Teacher-Student 的推荐实践

**知识蒸馏的本质:把"大模型学到的隐性知识"传给"小模型"**。

**为什么需要蒸馏?**

工业部署的"上线模型"必须小(快、便宜),但"训练模型"可以大(用更强算力训更长时间)。蒸馏就是用大模型的"软输出"当 teacher,训练小模型 student 模仿。结果:student 比直接用硬标签训的小模型 AUC 高 2-5%,逼近 teacher 但推理成本是 teacher 的 1/5~1/10。

**蒸馏损失三元组**:

$$
\mathcal{L}_{\text{distill}} = (1-\alpha) \cdot \mathcal{L}_{\text{hard}} + \alpha \cdot \mathcal{L}_{\text{soft}} + \beta \cdot \mathcal{L}_{\text{rank}}
$$

1. **硬损失** $\mathcal{L}_{\text{hard}}$:Student vs 真实标签的标准 CE
2. **软损失** $\mathcal{L}_{\text{soft}}$:Student vs Teacher 的 KL 散度,温度 T 缩放(T=3~6)
3. **排序损失** $\mathcal{L}_{\text{rank}}$:推荐特有,保持 Student 的"对正负样本的相对打分"和 Teacher 一致

**为什么推荐特别需要"排序损失"?**

纯 KL 散度优化的是"概率分布",对推荐来说不完全够——推荐只关心"哪个 item 排第一",不关心"具体概率值"。**排序损失直接优化 pairwise order**,teacher 觉得 item A 比 item B 好,student 也要觉得 A 比 B 好。**这是 GenRec 蒸馏区别于 LLM 蒸馏的关键**——LLM 关心"下一个 token 是什么字符",推荐关心"下一个 item 排第几"。

**温度 T 的物理意义**:

T 大 → 软分布"被抹平",student 学到 teacher 的"不确定性"(比如 teacher 对 A 和 B 都不太确定时,软标签接近 50:50,student 学到"我也可以不确定")。**T 太小,软标签 ≈ 硬标签,失去蒸馏意义;T 太大,信息被抹平,student 学不到东西**。T=3~6 是经验值。

**文字**:知识蒸馏用大模型（Teacher）的"软标签"训练小模型（Student），比直接用硬标签（0/1）效果好得多——因为软标签包含了 Teacher 对各候选的"相对置信度"，信息更丰富。

**公式**:蒸馏损失三元组

$$\mathcal{L}_{\text{hard}} = -\sum_c y_c \log p_c^{(S)} \quad \text{(Cross-Entropy with 0/1 labels)}$$

$$\mathcal{L}_{\text{soft}} = T^2 \cdot D_{\text{KL}}\!\left(\text{softmax}(z^{(T)}/T) \;\|\; \text{softmax}(z^{(S)}/T)\right) \quad \text{(KL 散度, 温度缩放)}$$

$$\mathcal{L}_{\text{distill}} = (1-\alpha) \cdot \mathcal{L}_{\text{hard}} + \alpha \cdot \mathcal{L}_{\text{soft}} + \beta \cdot \mathcal{L}_{\text{rank}}$$

**温度 $T$ 的作用**：大 $T$ 使软分布更平滑（近似均匀），让 Student 学到更多 Teacher 的"不确定性"；$T=1$ 退化为普通 CE。典型值：$T=3 \sim 6$（分类任务），推荐场景用 $T=2 \sim 4$。

```python
class DistillationLoss:
    """
    GenRec 的知识蒸馏损失 = 三个部分的加权和。

    1. 硬损失(Hard Loss): 和真实标签的 Cross-Entropy
    2. 软损失(Soft Loss): 和 Teacher 输出的 KL Divergence
    3. 排序损失(Rank Loss): 保持 Teacher 的排序序(推荐特有)
    """
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.3):
        self.T = temperature     # 温度:越高软标签越平滑
        self.alpha = alpha       # 软损失的权重
        self.beta = beta         # 排序损失的权重

    def __call__(self, student_logits, teacher_logits, labels):
        # 1. 硬损失(标准 CE)
        hard_loss = F.cross_entropy(student_logits, labels)

        # 2. 软损失(KL 散度,带温度)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean"
        ) * (self.T ** 2)  # 温度补偿

        # 3. 排序损失(推荐特有:保持 teacher 对正负样本的排序)
        # teacher 对 item_i 和 item_j 的分数差
        # student 也要维持这个差距
        pos_mask = (labels == 1).float().unsqueeze(-1)
        neg_mask = (labels == 0).float().unsqueeze(-1)
        teacher_margin = (teacher_logits * pos_mask).sum(-1) - (teacher_logits * neg_mask).sum(-1)
        student_margin = (student_logits * pos_mask).sum(-1) - (student_logits * neg_mask).sum(-1)
        rank_loss = F.mse_loss(student_margin, teacher_margin.detach())

        return hard_loss + self.alpha * soft_loss + self.beta * rank_loss
```

### 16.5 推理优化:KV-cache, PagedAttention, Continuous Batching

#### 16.5.1 KV-cache 的数学

**KV-cache 是 LLM 推理的"灵魂优化"**。

**没有 KV-cache 的推理计算量**:

生成 $T$ 个 token,每步都要算所有历史 token 的 K、V。Attention 矩阵大小 = $T \times T$,第 $t$ 步算量 = $O(t \cdot d)$,总推理 = $\sum_{t=1}^{T} O(t \cdot d) = O(T^2 d)$。**T=256 时 65536 次乘加,T=1024 时 1M 次乘加**——根本不可能在线推理(20ms 端到端 KPI 不允许)。

**有 KV-cache 的推理计算量**:

每步只算当前新 token 的 K、V,历史的直接从 cache 取。第 $t$ 步算量 = $O(t \cdot d)$,但常数项从"重算所有历史"降为"一次小投影",**总推理 ≈ $O(T d^2)$,比无 cache 快 T 倍**。T=256 时,从 65k 次乘加降到 256 次——**256 倍加速**。

**KV cache 的显存代价**:

$$
M_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{seq}} \times \text{bytes}
$$

LLaMA-3 8B + GQA($n_{\text{kv}}=8$, $L=32$, $T=8192$, FP16)单序列约 1GB。**Batch=64 时 64GB,A100 80GB 装不下**。这就是为什么需要 PagedAttention(分页管理)和 GQA(降低 n_kv)。

**GenRec 推理的"用户级 cache"优势**:

LLM 推理的 KV cache 是一次性(对话结束就丢),但 GenRec 的"会话历史"是连续的——用户 1 小时前刷的 50 个 item,1 小时后还在用同一个 cache,直接复用。**这让 GenRec 推理比 LLM 推理天然快 30-50%**——同样 A100 资源,GenRec 能服务的并发用户数是 LLM 的 2 倍。

**无 KV-cache** 时第 $t$ 步需重算所有 token 的 K,V:计算量 $O(t^2 d)$,总推理 $O(T^3 d)$。

**有 KV-cache** 时每步只算当前 token 的 K,V 并拼接历史:计算量 $O(T d^2)$。

KV-cache 显存大小:

$$M_{\text{KV}} = 2 \times n_{\text{kv}} \times d_{\text{head}} \times T_{\text{seq}} \times L \times \text{bytes}$$

其中 $n_{\text{kv}}$ 为 KV heads 数，$L$ 为层数。LLaMA-3 8B（GQA, $n_{\text{kv}}=8$，$L=32$，$T=8192$，FP16）$\approx$ 1GB；若 batch=64 则 $\approx$ 67GB，需要 PagedAttention。

```python
class KV_Cache:
    """
    KV-cache:Decoder-Only 推理加速的核心。

    不使用 KV-cache:
      step 1: 输入 [t_1] → 算 Attention(t_1, t_1) → 预测 t_2
      step 2: 输入 [t_1, t_2] → 重新算 Attention(t_1..t_2, t_1..t_2) → 预测 t_3
      step 3: 输入 [t_1, t_2, t_3] → 重新算 Attention(全部)...

      问题:step 2 的 t_1 的 K,V 在 step 1 已经算过!
           每次重新算 = 浪费 O(n²) 计算量

    使用 KV-cache:
      step 1: 输入 t_1 → 算 K_1, V_1 → cache[K_1, V_1] → 预测 t_2
      step 2: 输入 t_2 → 算 K_2, V_2 → cache += [K_2, V_2] → Attention(仅 t_2, cache全部) → 预测 t_3
      step 3: 输入 t_3 → 算 K_3, V_3 → cache += [K_3, V_3] → ...

      每个 step 只算当前 token,缓存过去所有 token 的 K,V
      Attention 计算量:O(Td²) vs O(T²d) 的差距!
    """
    def __init__(self):
        self.k_cache = []   # 每个 token 的 K 向量
        self.v_cache = []   # 每个 token 的 V 向量

    def append(self, k, v):
        self.k_cache.append(k)
        self.v_cache.append(v)

    def get(self):
        # 返回所有缓存的 K,V
        return torch.stack(self.k_cache, dim=2), torch.stack(self.v_cache, dim=2)
```

**KV-cache 的代价:显存!**

```python
# 以 LLaMA 3 8B 为例:
d_model = 4096
n_layers = 32
n_kv_heads = 8(GQA)
head_dim = d_model // 32 = 128
max_seq_len = 8192

# 每层 KV cache = 2 × n_kv_heads × seq_len × head_dim × 2bytes(FP16)
per_layer = 2 * 8 * 8192 * 128 * 2  # 33,554,432 bytes = 32MB
total_cache = 32 * 32MB              # 1.05GB

# 如果 MHA(n_kv_heads=32): 4.2GB
# 如果 MQA(n_kv_heads=1):   0.13GB
# 如果 batch_size=64:       ×64 = 67GB! 显存爆炸!

# 所以需要 PagedAttention(6.5.2 节)
```

---

**文字$\cdot$公式$\cdot$代码 5:为什么 GenRec 在线推理离不开 KV-Cache,以及它怎么把 20ms 时延变成可能**

**文字**:推荐在线精排的硬 KPI 是 20ms 端到端,留给 GenRec 推理(用 semantic ID 生成下一个物品)的时间只有个位数毫秒。无 KV-Cache 时,生成第 t 个 token 要重新算前 t-1 个 token 的 K, V,整段推理的 attention 计算量是 $O(T^2 d)$,256 个 token 就要 65k 次乘加,根本不可能在 10ms 内完成。KV-Cache 把历史的 K, V 缓存下来,每步只算当前新 token 的 K, V 并拼接,单步 attention 降到 $O(T d)$,生成 256 token 比无 cache 快 256 倍。这是 GenRec 推理从"理论可行"变成"生产可用"的临界点。显存代价是缓存占用,公式里 GQA 的 8 个 KV head 把单序列 KV cache 压到 1GB(batch=64 = 64GB),还能用 PagedAttention 做虚拟内存分页。GenRec 场景下 KV cache 有两种管理粒度:**用户级 cache**(同一用户多次请求复用,几分钟内的会话内命中率 80%+,极大降低 GPU 时延),和**会话级 cache**(单次请求一次,登出即清)。用户级 cache 是 GenRec 相对 LLM 推理的杀手锏,因为用户的浏览/点击历史是连续的,而 LLM 的对话经常是无状态的。

**公式**:单序列 KV cache 显存:

$$M_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{seq}} \times \text{bytes\_per\_elem}$$

LLaMA-3 8B(GQA, $n_{\text{kv\_heads}}=8$, $L=32$, $T=8192$, FP16)单序列约 1GB;batch=64 时约 64GB,需要 PagedAttention 分页。

**代码**:

```python
class KVCache:
    """按层预分配的 KV cache,避免每步 cat 带来的内存碎片"""
    def __init__(self, n_layers, max_len, n_kv_heads, head_dim, dtype=torch.float16):
        # 形状: [n_layers, 2(K+V), B, n_kv_heads, max_len, head_dim]
        self.cache = torch.empty(n_layers, 2, 0, n_kv_heads, max_len, head_dim, dtype=dtype)
        self.cur_len = 0  # 当前已填充的 token 数

    def update(self, layer_idx, k_new, v_new):
        # k_new, v_new: [B, n_kv_heads, 1, head_dim],仅当步的 K, V
        self.cur_len += 1
        # 写入预分配位置,不重新分配张量
        self.cache[layer_idx, 0, :, :, self.cur_len-1:self.cur_len] = k_new
        self.cache[layer_idx, 1, :, :, self.cur_len-1:self.cur_len] = v_new
        # 返回截至当前步的完整历史 K, V
        return (self.cache[layer_idx, 0, :, :, :self.cur_len],
                self.cache[layer_idx, 1, :, :, :self.cur_len])

# 推理循环:每步只算新 token 的 K, V,历史从 cache 取,O(T·d) 线性
for token in generated_tokens:
    k_new, v_new = compute_kv(token)             # O(d²) 一次小投影
    k_hist, v_hist = cache.update(layer, k_new, v_new)
    attn_out = scaled_dot_product_attention(q_new, k_hist, v_hist)  # O(T·d)
```

---

#### 16.5.2 PagedAttention 与 Continuous Batching

```text
PagedAttention(vLLM):
把 KV-cache 按"页"组织,不要求连续内存。
原理和操作系统的虚拟内存完全一样:
  - 逻辑页 → 物理页(映射表)
  - 物理页可以不连续 → 消除了内存碎片
  - 共享前缀(系统 prompt)可以多请求共享同一物理页 → 省 80% 显存

Continuous Batching(ORCA):
不等待整个 batch 都生成完再换下一批。
每步推理后:
  1. 已完成的请求 → 移出 batch,释放 KV cache
  2. 有新请求 → 加入 batch,分配新的 KV cache 页
  3. 保持 batch 始终满载 → GPU 利用率从 40% → 95%
```

### 16.6 综合实践:一个"LLM 架构做推荐"的端到端示例

```python
"""
以下代码展示了一个完整的"用 LLM 架构做推荐"的流程。
包含:
1. RQ-VAE Tokenization(把物品转为语义 ID)
2. Decoder-Only 推荐模型(和 LLaMA 共用架构)
3. LoRA 适配新领域
"""

# ============ 1. RQ-VAE Tokenization ============
class GenRecTokenizer:
    """用 RQ-VAE 把物品内容转为语义 ID 序列"""
    def __init__(self, quantizer: ResidualQuantizer):
        self.quantizer = quantizer

    def encode_items(self, item_features):
        """item_features: [N, D] → semantic_ids: [N, L]"""
        return self.quantizer.encode(item_features)

    def encode_sequence(self, history_item_ids, item_feature_dict):
        """用户行为序列 → token 序列(用于模型输入)"""
        tokens = []
        for item_id in history_item_ids:
            feat = item_feature_dict[item_id]
            sid = self.quantizer.encode(feat.unsqueeze(0)).squeeze(0)
            tokens.extend(sid.tolist())  # 展平: [3, 4] → [c1, c2, c3, c4]
        return tokens

# ============ 2. GenRec 模型 ≈ Small LLM ============
class GenRecTransformer(nn.Module):
    """和 LLaMA 共用同一套架构的推荐模型"""
    def __init__(self, vocab_size=256, d_model=512, n_layers=6, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, kv_cache=None):
        x = self.token_emb(tokens)      # [B, T, D]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)         # [B, T, vocab_size]
        return logits

# ============ 3. LoRA 适配 ============
def apply_lora_to_genrec(model, rank=16, alpha=32):
    """给 GenRec 模型插 LoRA"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            target in name for target in ["q_proj", "v_proj", "gate_proj", "up_proj"]
        ):
            # 替换为 LoRA Linear
            lora_lin = LoRALinear(
                module.in_features, module.out_features,
                rank=rank, alpha=alpha
            )
            lora_lin.weight.data = module.weight.data
            # 替换模块
            parent_name, child_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, lora_lin)
    return model

# ============ 4. 训练循环 ============
model = GenRecTransformer()
model = apply_lora_to_genrec(model, rank=16)

optimizer = torch.optim.AdamW(
    [p for n, p in model.named_parameters() if "lora_" in n],
    lr=1e-4
)

for batch in train_loader:
    tokens = batch["semantic_ids"]        # [B, T*L]
    labels = batch["next_semantic_id"]    # [B]

    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, -1, :],                # 只取最后一个位置的预测
        labels
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 16.7 GenRec + Small LLM 架构选型速查表

| 维度 | 选择 | 理由 | 不选的理由 |
| :--- | :--- | :--- | :--- |
| **架构** | Decoder-Only | KV-cache 加速,生态兼容 | E-D 在需要"全量编码"时更好 |
| **激活** | SwiGLU | 梯度流动好,指标最高 | 比 ReLU 多 1 次投影 |
| **归一化** | Pre-RMSNorm | 训练稳定,比 LN 快 10% | BN 不能在 batch=1 时用 |
| **位置编码** | RoPE | 可外推,相对位置 | ALiBi 在某些任务上好 |
| **注意力** | GQA(4:1~8:1) | 省 4~8× KV-cache | MHA 在"效果优先"时更好 |
| **Token化** | 语义 ID(RQ-VAE) | 词表可控,冷启动,语义 | 需要预训练 RQ-VAE |
| **微调** | LoRA(rank=16~32) | 省显存,可切换,无推理开销 | 全量 FT 在数据充足时更好 |
| **量化** | AWQ INT4 / QLoRA NF4 | 省 75% 显存,质量损失 < 1% | INT8 更精确但省得少 |
| **推理** | vLLM(PagedAttention) | 消除碎片,共享前缀 | 简单场景可用 naive KV |
| **蒸馏** | 软标签 + 排序保持 | 小模型保持排序顺序 | 纯 KL 散度可能破坏排序 |

### 16.8 第六章小结 & 自测

**小结**:
- GenRec 和 <30B LLM 的架构正在快速趋同:SwiGLU + RMSNorm + RoPE + GQA + Decoder-Only 已成为事实标准
- 语义 ID(RQ-VAE)把千万级物品映射到可控词表上,同时赋予语义理解能力
- LoRA + QLoRA 让消费级 GPU 也能微调工业级模型
- 量化 + PagedAttention + Continuous Batching 是部署 GenRec 的必备技术
- HSTU/Meta 证明了 GenRec 存在 Scaling Law:计算量增加 → 推荐质量幂律增长

**自测题**:
1. SwiGLU 相比 ReLU FFN 多了哪次投影?为什么参数量却可以相同?
2. 为什么现代 LLM 和 GenRec 都用 RMSNorm 而不是 LayerNorm?RMSNorm 去掉了什么?
3. RoPE 和 Sinusoidal PE 的核心区别是什么?为什么 RoPE 能外推?
4. GQA(Grouped-Query Attention)节省的是什么?一个 GQA-8 配置下,和 MHA 相比 KV-cache 省了多少?
5. 语义 ID 和原始 ID Token 化的区别是什么?为什么 TIGER 用 4 层 Codebook 而不是 1 层?
6. LoRA 的 rank 参数控制什么?rank=64 和 rank=4 分别适用于什么场景?
7. AWQ 量化和 GPTQ 量化的核心区别是什么?哪个更好?

---

## 16.9 端到端 GenRec 训练 Pipeline:从原始日志到在线服务

> 16.1-16.8 解决了"用什么架构、怎么 Token 化、怎么微调、怎么推理"四个独立问题。本节把 4 个独立问题串成一条可执行的端到端 Pipeline,并指出阶段间最常踩的坑。

### 16.9.1 整体流程与产出物

```text
原始日志                阶段 1              阶段 2              阶段 3                阶段 4
────────                ─────                ─────                ─────                ─────
用户行为序列     →     RQ-VAE 训练      →   SFT 数据构造    →   LoRA SFT        →   GRPO/RL
(items 1~N)           4 周 × 8 卡           prompt 模板         7B base             4 周 × 8 卡
    │                    │                    │                    │                    │
    ▼                    ▼                    ▼                    ▼                    ▼
内容特征抽取         语义 ID 词典         1 亿条 (hist, tgt)   Decoder-Only       合并 LoRA → 导出
(T5-Large)            [N, 4]                parquet                + LoRA(r=16)        → vLLM
```

**总目标函数(三阶段加权)**:

$$\mathcal{L}_{\text{total}} = \alpha_{\text{RQ}} \cdot \mathcal{L}_{\text{RQ-VAE}} + \alpha_{\text{SFT}} \cdot \mathcal{L}_{\text{SFT}} + \alpha_{\text{RL}} \cdot \mathcal{L}_{\text{RL}}$$

实践中三阶段是**独立训练**(避免联合训练的不稳定性),只通过"语义 ID 词典"这一接口衔接。

### 16.9.2 Stage 1:RQ-VAE 离线预训练

**目标**:把每个 item 映射成 L=4 个 0~255 的整数,使得语义相近的 item 共享前缀 token。

```python
# RQ-VAE 训练:输入 item 的内容 embedding,输出 4 级 codebook
import torch
from torch import nn

class RQVAETrainer:
    """工业级 RQ-VAE 训练器(TIGER 风格)"""
    def __init__(self, dim=256, num_levels=4, num_codes=256, beta=0.25):
        self.encoder = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, dim))
        self.decoder = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, 1024))
        self.codebooks = nn.Parameter(torch.randn(num_levels, num_codes, dim) * 0.01)
        self.beta = beta  # commitment loss 权重

    def quantize(self, z):
        """残差量化:每层找一个最近 codebook entry"""
        residual = z
        codes = []
        for l in range(self.codebooks.shape[0]):
            # 距离: [B, K]
            d = torch.cdist(residual.unsqueeze(1), self.codebooks[l].unsqueeze(0)).squeeze(1)
            idx = d.argmin(dim=-1)                      # [B]
            codes.append(idx)
            # 量化 + STE(直通估计)
            q = F.embedding(idx, self.codebooks[l])    # [B, D]
            residual = residual - q.detach() + (residual - q).detach() * 0  # 占位
        return torch.stack(codes, dim=-1)              # [B, L]

    def step(self, x):
        z = self.encoder(x)
        codes = self.quantize(z)
        q = sum(F.embedding(codes[:, l], self.codebooks[l]) for l in range(4))
        x_hat = self.decoder(q)
        # 三项损失
        recon = F.mse_loss(x_hat, x)
        commit = F.mse_loss(q, z.detach())             # encoder 靠近 codebook
        codebook = F.mse_loss(q.detach(), z) * self.beta
        return recon + commit + codebook, codes
```

**常见坑**:
- **Codebook 坍缩**(只用到 100/256 个 code):EMA 更新 + 每 1000 步 reset 未使用 code
- **重建 loss 不下降**:多半是 $\beta$ 太大,先设 $\beta=0$ 跑通再加

### 16.9.3 Stage 2:SFT 监督微调

**目标**:在解码器上学会 "用户历史 token 序列 → 下一个 item 的语义 ID 序列"。

```python
def build_sft_dataset(user_logs, semantic_ids, max_hist_len=50):
    """
    user_logs: [{user_id, history_items: [i1, i2, ...], target_item: i_t}]
    semantic_ids: [N, L] — 每个 item 的 4 级 codebook index
    """
    samples = []
    for log in user_logs:
        hist = log["history_items"][-max_hist_len:]
        # 输入:历史的 [L0, L1, L2, L3, L0, L1, ...] 拼接
        # 输出:target 的 [L0, L1, L2, L3]
        input_ids = semantic_ids[hist].flatten()                # [T*L]
        target_ids = semantic_ids[log["target_item"]]           # [L]
        samples.append({"input": input_ids, "target": target_ids})
    return samples

def sft_loss(logits, target_ids):
    """
    logits: [B, T*L, K=256]  — 模型对每个位置的预测
    target_ids: [B, L]        — 真实语义 ID
    只对最后 L 个位置算 loss(前面的位置是历史,不算损失)
    """
    L = target_ids.shape[1]
    pred = logits[:, -L:, :]                                    # [B, L, K]
    return F.cross_entropy(pred.reshape(-1, 256), target_ids.flatten())
```

**训练配置经验**:
| 超参 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| learning rate | 2e-4(LoRA)/1e-5(全量) | LoRA 可以用更大学习率 |
| warmup | 5% 步数 | 防止 loss 突刺 |
| batch size | 8~32/GPU(梯度累积到 256) | 受限于上下文长度 |
| max_history | 50 个 item(= 200 token) | 太长显存撑不住 |

### 16.9.4 Stage 3:GRPO 强化学习(简化版,推荐场景适配)

**目标**:在 SFT 基础上,直接优化推荐业务指标(点击/转化/diversity)。

GRPO(Group Relative Policy Optimization)用一组同 prompt 的采样来估计 baseline,不需要单独训 value network,比 PPO 简单很多:

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|\tau_i|}\sum_{t} \min\!\Big(\rho_{i,t} \hat{A}_i,\;\text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\Big)$$

其中 $\rho_{i,t} = \frac{\pi_\theta(\tau_{i,t}|\tau_{i,<t})}{\pi_{\theta_{\text{old}}}(\tau_{i,t}|\tau_{i,<t})}$ 是新旧策略概率比,优势 $\hat{A}_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$ 通过同组采样归一化得到。

```python
def grpo_step(policy, ref_policy, prompts, reward_fn, group_size=8):
    """GRPO 单步:每个 prompt 采样 group_size 条轨迹,用组内归一化优势更新"""
    all_rewards, all_logprobs, all_ref_logprobs = [], [], []
    for prompt in prompts:
        # 同 prompt 采样 G 条轨迹
        trajs = [policy.generate(prompt) for _ in range(group_size)]
        rewards = torch.tensor([reward_fn(t) for t in trajs])    # [G]
        # 组内归一化优势
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        for traj, adv in zip(trajs, advantages):
            all_rewards.append(reward_fn(traj))
            all_logprobs.append(policy.log_prob(traj))
            all_ref_logprobs.append(ref_policy.log_prob(traj).detach())
    # 重要性比率 + clip
    ratio = (all_logprobs - all_ref_logprobs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(0.8, 1.2) * advantages
    loss = -torch.min(surr1, surr2).mean() - 0.01 * kl_to_ref
    return loss
```

### 16.9.5 阶段间接口与生产流水线 checklist

| 阶段 | 交付物 | 下游消费 | 验证方式 |
| :--- | :--- | :--- | :--- |
| 内容特征 | `item_emb.npy [N, 1024]` | RQ-VAE 训练 | t-SNE 可视化聚类 |
| RQ-VAE | `semantic_ids.json {item: [c1..c4]}` | SFT + 在线推理 | codebook 利用率 > 80% |
| SFT 模型 | `sft_merged.safetensors` | RL 初始化 / 兜底推理 | 离线 AUC + Recall@50 |
| RL 模型 | `rl_merged.safetensors` | 在线服务 | A/B 测试业务指标 |

**生产 checklist(每条都是真实踩过的坑)**:
1. RQ-VAE 训练时 `item_emb` 必须先 L2 normalize,否则 codebook 距离度量失效
2. SFT 数据不能只取 `target=click`,要混入 `target=purchase` 等长尾行为
3. 训练样本 shuffle 时**不要按 user_id 分桶**,否则验证集泄漏
4. LoRA 训练后,合并前一定要在验证集上对比合并前/后指标(差超过 0.1% 说明合并有 bug)
5. RL reward 中必须加 **diversity penalty**(同质化推 10 个相似 item 给负分),否则会全部收敛到爆款

---

## 16.10 GenRec 失败案例与诊断:踩坑实录与修复指南

> GenRec 训练中,RQ-VAE 坍缩、reward hacking、loss 突刺是最常见的三类失败。本节给出**可量化的诊断指标**和**对应的修复方案**。

### 16.10.1 五大高频失败模式一览

| 失败模式 | 典型症状 | 出现阶段 | 根因 |
| :--- | :--- | :--- | :--- |
| RQ-VAE codebook 坍缩 | 重建 loss 低,但 256 个 code 只用 30 个 | RQ-VAE | encoder 输出坍缩到局部 |
| SFT 位置偏置 | 模型只预测训练集中高频的尾部 token | SFT | 数据未 shuffle / 过采样 |
| SFT loss 突刺 | 每 1000 步左右 loss 飙升 10× | SFT | warmup 不够 / 学习率过大 |
| RL reward hacking | 离线指标涨,在线 CTR 跌 | RL | reward 漏洞,被策略利用 |
| RL 模式坍缩 | 生成序列 90% 相同 | RL | KL 惩罚太小,策略远离参考 |

### 16.10.2 Codebook 坍缩诊断

**核心指标:codebook 利用率**(使用的 code 数 / 总 code 数):

$$U = \frac{|\{c : \exists x,\; q(x) = c\}|}{K}, \quad K = 256$$

```python
def codebook_utilization(semantic_ids, num_codes=256):
    """
    semantic_ids: [N, L] — 全部物品的语义 ID
    返回:每层利用率 + 死码本列表
    """
    utilization = []
    for l in range(semantic_ids.shape[1]):
        used = torch.unique(semantic_ids[:, l])
        u = len(used) / num_codes
        dead = [c for c in range(num_codes) if c not in used.tolist()]
        utilization.append({"level": l, "used": len(used), "dead": dead[:10]})
    return utilization
```

**修复方案**:
- **EMA codebook 更新**(0.99 衰减):让未使用的 code 慢慢被"拉"到数据分布
- **周期性 reset**:每 500 步把 top-k 最常用的 code 的向量**复制给** dead code
- **dead code 复活**:直接用 k-means 重新初始化 dead code 为当前 batch 的聚类中心

### 16.10.3 Reward Hacking 检测

**最隐蔽的失败**:模型学会利用 reward 函数的漏洞,在离线评估上完美但在线掉点。

```text
常见漏洞:
  1. 推荐爆款 → reward = 1(purchase 率高)
     但爆款不需推荐用户也会买 → 在线 CTR 跌

  2. 推荐长尾冷门 → surprise 高 → diversity reward 高
     但用户其实不喜欢 → 在线转化跌

  3. 重复推同一类 → 短时点击高(用户探索)→ 长期留存跌
```

**诊断代码**:

```python
def detect_reward_hacking(generated_seqs, item_metadata, threshold=0.3):
    """
    检测三种典型 reward hacking
    返回:hacking_flags dict
    """
    flags = {}
    # 1. 爆款集中度:Top-1% item 占比
    pop_count = torch.bincount(generated_seqs.flatten(), minlength=N_items)
    top1_ratio = pop_count.topk(int(0.01 * N_items)).values.sum() / pop_count.sum()
    flags["popularity_concentration"] = top1_ratio > threshold

    # 2. 类别集中度(同类别连续出现率)
    cat_seq = item_metadata["category"][generated_seqs]
    same_cat_run = (cat_seq[:, 1:] == cat_seq[:, :-1]).float().mean()
    flags["category_runaway"] = same_cat_run > 0.5

    # 3. 序列内重复率
    unique_ratio = torch.tensor([len(set(s.tolist())) for s in generated_seqs]).float().mean() / len(generated_seqs[0])
    flags["sequence_repetition"] = unique_ratio < 0.4
    return flags
```

**修复**:在 reward 中加入 $\text{reward} = r_{\text{click}} - \lambda_1 \cdot \text{cat\_concentration} - \lambda_2 \cdot \text{long\_tail\_penalty}$,其中长尾项鼓励推荐 $\text{freq}(i) < 0.001$ 的 item。

### 16.10.4 SFT loss 突刺的根因诊断

**经验法则**:训练 loss 偶尔突刺 2~3 倍是正常的(过了一个 hard batch),突刺 10× 且不恢复就是真问题。

| 突刺模式 | 根因 | 修复 |
| :--- | :--- | :--- |
| 步 1000 周期性突刺 | warmup 步数太短 | warmup 从 5% → 10% |
| 突刺后 loss 缓慢回落 | 梯度爆炸 | `clip_grad_norm_(1.0)` |
| 突刺后 loss 永远不恢复 | 数据里有异常 batch | 加 NaN 监控 + 跳过 batch |
| loss 在 0.5 附近震荡 | 学习率过大 | 减半 LR |
| loss 收敛到 NaN | FP16 溢出 | 换 BF16 或加 `dynamic_loss_scale` |

```python
# 突刺检测器
class SpikeDetector:
    def __init__(self, window=100, spike_ratio=3.0):
        self.window = window
        self.spike_ratio = spike_ratio
        self.history = []
    def __call__(self, loss):
        self.history.append(loss)
        if len(self.history) < self.window: return False
        baseline = sum(self.history[-self.window:-1]) / (self.window - 1)
        if loss > baseline * self.spike_ratio:
            print(f"[SPIKE] step={len(self.history)} loss={loss:.4f} baseline={baseline:.4f}")
            return True
        return False
```

### 16.10.5 RL 模式坍缩的熵监控

**核心指标**:生成分布的熵(熵过低 = 模式坍缩):

$$H(\pi) = -\sum_{c=1}^{K} \pi(c) \log \pi(c), \quad H_{\text{normalized}} = \frac{H}{H_{\max}} = \frac{H}{\log K}$$

```python
def entropy_during_rl(policy, eval_prompts, K=256):
    """归一化熵,1.0=均匀分布,0.0=完全坍缩"""
    entropies = []
    for prompt in eval_prompts:
        logits = policy(prompt)                              # [L, K]
        probs = F.softmax(logits, dim=-1)
        H = -(probs * probs.log()).sum(dim=-1).mean()
        entropies.append(H / np.log(K))                       # 归一化到 [0, 1]
    return np.mean(entropies)

# 触发告警:归一化熵 < 0.3 说明策略已经开始坍缩
```

**修复**:把 KL 惩罚从 $0.01$ 提到 $0.1$,或加 entropy bonus $-\beta H(\pi)$ 到 reward。

---

## 16.11 在线服务架构:TensorRT-LLM / vLLM 部署与 QPS 优化

> 训练好的 GenRec 模型怎么扛住 1 万 QPS、20ms P99 时延?这一节聚焦**生产部署的关键决策**和**性能调优**。

### 16.11.1 推理引擎选型对比

| 引擎 | 适用规模 | GenRec 适用度 | 主要优势 |
| :--- | :--- | :--- | :--- |
| HuggingFace Transformers | 调试 | ★★ | 简单,生态全 |
| vLLM | 中小规模(< 100 QPS/卡) | ★★★★ | PagedAttention,部署快 |
| TensorRT-LLM | 大规模(> 500 QPS/卡) | ★★★★★ | 极致优化,INT4/AWQ 友好 |
| llama.cpp (GGUF) | CPU 兜底 | ★ | 无需 GPU |

**工业实践**:vLLM 做 dev/staging,TensorRT-LLM 做 prod。两者都支持 PagedAttention 和 Continuous Batching。

### 16.11.2 vLLM 部署 GenRec(最短路径)

```python
# server.py — 用 vLLM 部署 GenRec,支持 OpenAI 兼容 API
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import build_app

# 1. 加载合并 LoRA 后的模型
llm = LLM(
    model="genrec-7b-merged",
    tensor_parallel_size=1,                # 单卡
    gpu_memory_utilization=0.9,            # 用 90% 显存
    max_model_len=512,                     # GenRec 上下文短(50 item × 4 token = 200)
    dtype="bfloat16",
    quantization="awq",                    # 4-bit AWQ,显存省 4×
)

# 2. Sampling:beam search 5 个候选,top-k 截断
sampling = SamplingParams(
    n=5,                                   # 5 个 beam
    temperature=1.0,
    top_k=50,
    max_tokens=4,                          # 语义 ID 4 个 token
)

# 3. 请求处理
async def recommend(user_history_sids: list[int]) -> list[list[int]]:
    """
    user_history_sids: [c1, c2, c3, c4, c1', c2', ...]  — 历史物品展平的语义 ID
    返回:5 个候选的语义 ID
    """
    prompt = build_prompt(user_history_sids)   # 加 system prompt + 用户历史
    outputs = llm.generate([prompt], sampling)
    return [o.outputs[0].token_ids for o in outputs]
```

### 16.11.3 QPS / 时延模型

**核心时延组成**(单次请求):

$$T_{\text{total}} = T_{\text{prefill}} + L_{\text{gen}} \cdot T_{\text{decode}} + T_{\text{overhead}}$$

其中 $L_{\text{gen}}$ 是生成 token 数(GenRec = 4),$T_{\text{decode}}$ 是单 token 解码时间。

| 操作 | 时延 (7B + AWQ, A100) | 说明 |
| :--- | :--- | :--- |
| Prefill(200 token) | 15ms | 处理历史 |
| Decode(4 token × beam=5) | 4 × 5 × 8ms = 160ms | beam search 慢 |
| Continuous Batching 加速后 | 15 + 30 = **45ms** | beam=5 共享 prefix |
| **最终端到端 P99** | **< 80ms** | 含网络和后处理 |

**QPS 上限(单 A100)**:$T_{\text{P99}} = 80\text{ms} \Rightarrow \text{单 batch QPS} \approx 12.5$;启用 Continuous Batching 后 batch=32,QPS $\approx 400$。

### 16.11.4 三大 QPS 优化手段

**手段 1:Prefix Caching(系统提示缓存)**

GenRec 的 system prompt(描述任务、物品分类体系)对所有用户一样。vLLM 自动缓存:

```python
# vLLM 启用 prefix caching(默认开)
llm = LLM(model="...", enable_prefix_caching=True)

# 效果:200 token system prompt 只算一次,后续请求直接复用 KV cache
# 节省:Prefill 时延从 15ms → 2ms
```

**效果**:首 token 时延 -87%,但需要 batch 内请求前缀相同(GenRec 天然满足)。

**手段 2:Speculative Decoding(草稿模型加速)**

用一个小模型(如 1B)生成候选 token,大模型(7B)一次性验证:

```text
标准 decoding:    大模型 1 个 token → 验证 1 个 → 提交   (8ms × 4 = 32ms)
Speculative:     小模型 4 个 token → 大模型 1 步验证 4 个  (3ms + 10ms = 13ms)
                  加速比 ≈ 2.5×
```

**GenRec 适配**:用 1B 的蒸馏小 GenRec 当 draft,7B 当 target。**但 GenRec 只需要 4 token**,speculative 的草稿长度也设 4,接受率通常 60~80%,加速 1.5~2×。

**手段 3:用户级 KV Cache 复用**

GenRec 相对 LLM 推理的杀手锏:**同一用户的多次请求可共享历史 KV cache**。

```python
class UserKVCache:
    """
    用户级 KV cache 池(Redis/内存)
    key: user_id, value: 历史的 K, V tensor
    TTL: 5 分钟
    """
    def __init__(self, max_users=10000):
        self.cache = LRUCache(max_users)

    def get_or_compute(self, user_id, history_tokens, model):
        if user_id in self.cache:
            cached_kv, cached_len = self.cache[user_id]
            # 新请求只算新增 token 的 K, V,接在 cached 后面
            new_kv = model.forward_incremental(history_tokens[cached_len:], kv=cached_kv)
            return new_kv
        else:
            full_kv = model.forward_full(history_tokens)
            self.cache[user_id] = (full_kv, len(history_tokens))
            return full_kv
```

**收益**:用户 5 分钟内第 2 次请求,Prefill 节省 100%(直接从 cache 取)。**经验命中率 60~80%**(电商用户平均每分钟多次刷新),QPS 翻倍。

### 16.11.5 Speculative Decoding 在 GenRec 中的特例

**重要警示**:GenRec 只生成 4 token,标准 speculative 的"draft 5 步,verify 1 步"策略不适用。**推荐配置**:

```python
# draft model 比 target 小 4~7 倍(1B vs 7B)
# draft 长度 = target 总长(都生成 4 token)
# 一次 verify 全部 4 token,接受率通常 50~70%

draft_model = load_model("genrec-1b")      # 1B 蒸馏小模型
target_model = load_model("genrec-7b-awq") # 7B 量化大模型

def speculative_genrec(prompt):
    # 小模型生成 4 个候选
    draft_tokens = draft_model.generate(prompt, max_tokens=4)
    # 大模型一次 verify
    accepted = target_model.verify(prompt, draft_tokens)
    return accepted   # 通常 2~3 个被接受,补 1~2 个
```

**收益**:GenRec 4 token 生成从 32ms 降到 18ms,加速 ~1.8×。在小 batch 下收益更明显(batch=1 时 7B 受限于显存带宽,1B 可并行填满)。

### 16.11.6 GenRec 生产部署 checklist

| 项 | 推荐配置 | 红线 |
| :--- | :--- | :--- |
| 量化 | AWQ INT4 | 不要用 INT3 以下(质量塌) |
| 引擎 | TensorRT-LLM(> 500QPS) / vLLM(< 500QPS) | 不要用 HF transformers 上 prod |
| 上下文 | max_model_len=512 | 太大浪费显存 |
| Beam 数 | 5~10 | 太多时延爆炸 |
| Prefix cache | 必开 | GenRec 几乎都是同一 system prompt |
| 监控 | P50/P99 时延、QPS、显存、KV cache 命中率 | 不监控 KV cache 命中率就上线 = 盲飞 |
| Fallback | 模型服务挂时降级到 ID-based 双塔 | 不可用 = 直接损失收入 |

**最常见的线上事故**:
1. **显存碎片**:没开 PagedAttention,长上下文 + 大 batch → OOM
2. **KV cache 命中率低**:Prefix 不一样(每个用户 prompt 不一致)→ 没真正复用
3. **Beam search 过深**:beam=20 想"更准",但时延 200ms 触发前端超时
4. **A/B 流量不均**:新模型 1% 流量但放在独立 GPU 上,GPU 利用率 < 10% 浪费钱

---

**本章新内容小结**:
- **16.9** 把"数据→RQ-VAE→SFT→RL→部署"串成可执行 Pipeline,并给出 5 条生产 checklist
- **16.10** 系统化梳理 5 类高频失败模式,提供可量化的诊断指标(codebook 利用率、熵、reward hacking 检测器)
- **16.11** 覆盖 vLLM/TensorRT-LLM 部署选型、QPS 数学模型、Prefix Caching / 用户级 KV Cache / Speculative Decoding 三大优化手段,以及 4 条线上事故警示

### 16.12 全章结构图与「关键概念网」

为了帮助读者从整体上把握本章, 这里把前面 16.1-16.11 的全部关键概念组织成一张「依赖关系网」。**这张图是「学习完一张图」式的复盘指南**——当你忘掉某个具体概念的位置时, 看这张图就能回忆起它属于哪个家族。

```text
                         ┌─────────────────────────────────┐
                         │  16.1 核心论点: GenRec ≈ Small LLM │
                         └──────────────┬──────────────────┘
                                        │
        ┌─────────────────────┬─────────┴──────────┬────────────────────────┐
        │                     │                    │                        │
        ▼                     ▼                    ▼                        ▼
   16.2 架构组件        16.5 注意力变体        16.3 语义 ID                16.4 大模型技术
   (单层细节)            (注意力算法)          (tokenization)             (LoRA / 量化 / 蒸馏)
        │                     │                    │                        │
   ┌────┴────┐         ┌──────┴──────┐        ┌────┴────┐              ┌─────┴─────┐
   │         │         │             │        │         │              │           │
 FFN     Normalization  GQA      Flash  KV  VQ-VAE  RQ-VAE           LoRA       QLoRA
 SwiGLU  RMSNorm       MQA      Attention Cache 复用 重建算法          4bit      量化感知训练
 ReLU    vs LayerNorm                                                              + Adapter
                                                                                   + 蒸馏
        │                     │                    │                        │
        └──────────┬──────────┴─────────┬──────────┴────────────────────────┘
                   │                    │
                   ▼                    ▼
            16.7 训练流程      16.8 多任务 / 多场景
            (SFT / RLHF)      (融合策略)
                   │                    │
                   └──────────┬─────────┘
                              ▼
                      16.9 工业 Pipeline
                              │
                              ▼
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        16.10 失败模式    16.11 部署优化       (16.12-16.15 拓展)
        (诊断 + 缓解)    (vLLM + 量化)
```

**关系网三原则**:
1. **架构组件(16.2)是骨干**——掌握单层组件后才看其他章节。
2. **注意力变体(16.5)是加速器**——节省显存 + 时延, 不改变能力。
3. **语义 ID(16.3)是 GenRec 专属**——推荐系统独特的「tokenization」。

### 16.13 实战案例: 从 0 搭建可上线的 GenRec 系统

下面给出一个完整的、**可直接跑通**的 GenRec 训练与推理流水线, 帮助读者把前面章节的零碎知识串联起来。本案例针对一个真实业务场景——**短视频推荐**——设计, 数据规模为 100 万用户、100 万短视频、5 亿条历史行为。

#### 16.13.1 项目结构与依赖

```text
genrec_demo/
├── configs/
│   └── default.yaml             # Hydra 配置
├── data/
│   ├── raw/                     # 原始日志
│   ├── processed/                # Parquet 训练数据
│   └── codebook/                 # RQ-VAE 训出的 codebook
├── models/
│   ├── rq_vae.py               # Residual-Quantized VAE
│   ├── genrec_backbone.py       # Transformer 主干
│   └── genrec_head.py           # 输出头
├── scripts/
│   ├── 01_build_vocab.py        # 构建词表
│   ├── 02_train_rqvae.py        # 训练 RQ-VAE
│   ├── 03_train_genrec.py       # 训练生成式推荐
│   └── 04_serve_vllm.py         # 部署推理
└── eval/
    ├── recall_metrics.py        # Recall@K
    └── ndcg_metrics.py          # NDCG@K
```

依赖清单(`pyproject.toml`):

```toml
[project]
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "vllm>=0.4.0",
    "recbole>=1.2.0",
    "einops>=0.7.0",
    "hydra-core>=1.3.0",
    "wandb>=0.16.0",
    "faiss-gpu>=1.7.2",
]
```

#### 16.13.2 关键模型代码(可直接复用)

**RQ-VAE 模块**(生成 Semantic ID):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """单层 VQ(代码 16.3 中 16.3.1 的实现)."""
    def __init__(self, num_codes=256, codebook_dim=32, commitment_cost=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, codebook_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # z: [B, D] 或 [B, T, D]
        z_flat = z.reshape(-1, z.size(-1))
        # 找最近 codebook 索引
        distances = (
            (z_flat ** 2).sum(-1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + (self.codebook.weight ** 2).sum(-1)
        )
        encoding_indices = distances.argmin(dim=-1)
        # 量化
        z_q = self.codebook(encoding_indices).view(z.shape)
        # 直通估计(straight-through estimator)
        z_q_st = z + (z_q - z).detach()
        # 损失
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        # 困惑度(用于监控 codebook 利用率,见 16.10.2)
        encodings = F.one_hot(encoding_indices, self.codebook.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return z_q_st, loss, perplexity, encoding_indices.view(z.shape[:-1])


class RQVAE(nn.Module):
    """Residual-Quantized VAE: 多级量化降低重建误差."""
    def __init__(self, input_dim=128, hidden_dim=256, codebook_dim=32,
                 num_codes=256, num_levels=3, commitment_cost=0.25):
        super().__init__()
        # Encoder: 输入 → 隐向量
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_dim),
        )
        # Decoder: 量化结果 → 重构
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # 多级残差量化
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_codes, codebook_dim, commitment_cost)
            for _ in range(num_levels)
        ])
        self.num_levels = num_levels

    def forward(self, x):
        # 编码
        z = self.encoder(x)
        # 多级量化
        quantized = torch.zeros_like(z)
        residual = z
        all_indices = []
        total_vq_loss = 0
        total_perplexity = 0
        for quantizer in self.quantizers:
            q, vq_loss, perplexity, indices = quantizer(residual)
            quantized = quantized + q
            residual = residual - q.detach()  # 残差传递
            all_indices.append(indices)
            total_vq_loss = total_vq_loss + vq_loss
            total_perplexity = total_perplexity + perplexity
        # 解码重构
        x_recon = self.decoder(quantized)
        # 重建损失
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, total_vq_loss, recon_loss, torch.stack(all_indices, dim=-1)
```

**GenRec 主干(基于 LLaMA 架构)**:

```python
class GenRecBackbone(nn.Module):
    """基于 LLaMA 架构的生成式推荐模型."""
    def __init__(self, vocab_size=1000000, num_codes=256, num_levels=3,
                 d_model=512, num_heads=8, num_layers=12,
                 max_seq_len=128, use_rope=True, use_gqa=True):
        super().__init__()
        # 词嵌入: 把 (num_levels + 1) 个 code 拼成一个 token
        # 简化: 直接把每个 code 视作 token, 共享 vocab
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 位置编码(默认 RoPE)
        self.use_rope = use_rope
        if not use_rope:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # 主体: 12 层 LLaMA-style Block
        self.layers = nn.ModuleList([
            GenRecBlock(d_model, num_heads, use_gqa)
            for _ in range(num_layers)
        ])
        # 输出 norm
        self.norm = RMSNorm(d_model)

    def forward(self, semantic_ids):
        # semantic_ids: [B, T, num_levels]
        B, T, L = semantic_ids.shape
        # 把多级 code 压平为一个序列
        # 简化处理: 仅取第 1 级做 seq, 其余作为辅助输入
        ids = semantic_ids[..., 0]  # [B, T]
        x = self.token_emb(ids)  # [B, T, D]
        # 加位置编码
        if self.use_rope:
            # RoPE 在 attention 内处理
            pass
        else:
            positions = torch.arange(T, device=x.device).unsqueeze(0)
            x = x + self.pos_emb(positions)
        # 主干
        for layer in self.layers:
            x = layer(x)
        # 输出 norm
        x = self.norm(x)
        # 词表 logits
        return x @ self.token_emb.weight.t()  # [B, T, V]


class GenRecBlock(nn.Module):
    """单层 Block: GQA + RoPE + SwiGLU + RMSNorm."""
    def __init__(self, d_model, num_heads, use_gqa=True):
        super().__init__()
        self.attn = GQAAttention(d_model, num_heads, use_gqa)
        self.ffn = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RMSNorm(nn.Module):
    """16.2.2 的 RMSNorm 实现."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.weight * x / (rms + self.eps)


class SwiGLU(nn.Module):
    """16.2.1 的 SwiGLU FFN 实现."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * dim / 3)  # 8/3 规则
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GQAAttention(nn.Module):
    """16.5.2 的 GQA 实现."""
    def __init__(self, d_model, num_heads, use_gqa=True, num_kv_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if use_gqa else num_heads
        self.head_dim = d_model // num_heads
        # Q 投影: num_heads 个头
        self.w_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        # K, V 投影: 只有 num_kv_heads 个头(共享)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        # RoPE 频率(见 16.2.3)
        self._rope_freqs = None

    def _get_rope_freqs(self, seq_len, device):
        # 简化 RoPE 计算
        dim = self.head_dim
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device).float()
        angles = torch.outer(positions, freqs)
        return angles.cos(), angles.sin()

    def forward(self, x):
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE 应用在 Q, K 上
        cos, sin = self._get_rope_freqs(T, x.device)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        # KV 广播到所有 head
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        # 使用 PyTorch 2.0+ 的 Flash Attention 加速
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)

    def _apply_rope(self, x, cos, sin):
        # x: [B, H, T, D]
        # 简化: 与位置外积后旋转
        x_rot = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
        return (x * cos[:T]) + (x_rot * sin[:T])
```

#### 16.13.3 训练流程与超参(Hydra config)

```yaml
# configs/default.yaml
defaults:
  - rqvae
  - genrec

rqvae:
  input_dim: 128            # 输入 Embedding 维度(从 item 内容模型)
  codebook_dim: 32
  num_codes: 256            # 每层 256 个 code
  num_levels: 3             # 3 级残差
  hidden_dim: 256
  lr: 1e-3
  batch_size: 4096
  epochs: 50

genrec:
  vocab_size: 1000000       # 合并 vocabulary (256 ^ 3 ≈ 16M, 取子集)
  d_model: 512
  num_heads: 8
  num_layers: 12
  max_seq_len: 128
  lr: 5e-4
  weight_decay: 0.01
  warmup_steps: 1000
  batch_size: 256
  epochs: 10
  grad_clip: 1.0
  bf16: true                # 16.7 的混合精度

training:
  num_workers: 8
  log_every: 50
  eval_every: 1000
  save_every: 5000
  device: cuda
```

#### 16.13.4 训练脚本与显存估算

```python
import hydra
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb


@hydra.main(config_path="configs", config_name="default")
def train(cfg):
    wandb.init(project="genrec_demo")

    # 模型
    model = GenRecBackbone(
        vocab_size=cfg.genrec.vocab_size,
        d_model=cfg.genrec.d_model,
        num_heads=cfg.genrec.num_heads,
        num_layers=cfg.genrec.num_layers,
    ).to(cfg.training.device)

    optimizer = AdamW(model.parameters(), lr=cfg.genrec.lr,
                      weight_decay=cfg.genrec.weight_decay)

    # 16.7 的 warmup + cosine decay
    warmup_steps = cfg.genrec.warmup_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 0.5 * (1 + torch.cos(torch.pi * (step - warmup_steps) / 100000)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler('cuda', enabled=False)  # BF16 不需要 scaler

    for epoch in range(cfg.genrec.epochs):
        for batch in train_loader:
            ids = batch['semantic_ids'].to(cfg.training.device)
            targets = batch['targets'].to(cfg.training.device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,  # padding 不算 loss
                )

            optimizer.zero_grad()
            loss.backward()
            # 16.7.4 的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.genrec.grad_clip)
            optimizer.step()
            scheduler.step()

            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
```

**显存估算**(Seq-Len=128, 12 层 512 维):

| 模块 | 大小 |
| :--- | :--- |
| Embedding | $1e6 \times 512 \times 2\text{B} \approx 1\text{GB}$ |
| 12 层(权重+激活) | $\approx 2.5\text{GB}$ |
| 优化器状态(AdamW) | $\approx 1\text{GB}$ |
| **总计** | $\approx 4.5\text{GB (BF16)}$ |

**A100 单卡 80G 跑得动, RTX 4090 24G 也够用**(batch ≤ 256)。

#### 16.13.5 vLLM 部署 + 推理

```python
# scripts/04_serve_vllm.py
from vllm import LLM, SamplingParams

# 把训练好的 GenRec 转成 HF 格式
model = GenRecBackbone.from_pretrained("checkpoints/genrec-v1")
model.save_pretrained("huggingface_genrec/")

# 部署到 vLLM
llm = LLM(
    model="huggingface_genrec/",
    tensor_parallel_size=2,        # 2 卡张量并行
    gpu_memory_utilization=0.85,
    max_model_len=128,
    quantization="awq",            # INT4 量化,16.11
    enforce_eager=False,
)

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    n=10,                          # beam 候选数
    max_tokens=4,                  # 4 个 token(S-ID × 1 level)
)

# 推理: 输入历史 → 输出 top-10 候选 Semantic ID → 反查物品
def recommend(history_sids):
    prompt = build_prompt(history_sids)
    outputs = llm.generate([prompt], sampling)
    sids = outputs[0].outputs[0].token_ids  # [4]
    return decode_sids_to_item_ids(sids)
```

#### 16.13.6 A/B 测试与监控指标矩阵

```python
# 16.10 中的工程化 check
class GenRecMonitor:
    """上线后监控面板"""
    def __init__(self):
        self.metrics = {}

    def track(self, metric_name, value):
        self.metrics.setdefault(metric_name, []).append(value)

    def health_check(self):
        checks = {}
        # 1. Codebook 利用率
        checks['codebook_utilization'] = (
            self.metrics['unique_codes_last_1h'] / 256 ** 3
        )
        # 2. Token 平均长度
        checks['avg_generated_seq_len'] = (
            self.metrics['avg_seq_len']
        )
        # 3. 多样性(ILS)
        checks['diversity_ils'] = self.metrics['intra_list_similarity']
        # 4. 业务指标
        checks['ctr'] = self.metrics['ctr_24h']
        checks['cvr'] = self.metrics['cvr_24h']
        checks['gmv_per_user'] = self.metrics['gmv_per_user_24h']
        return checks
```

#### 16.13.7 案例总结:从 0 到上线的成本

| 阶段 | 关键产出 | 工程师投入 | 算力成本 |
| :--- | :--- | :--- | :--- |
| 数据 + 预处理 | Parquet 数据 + vocab | 1 人 1 周 | 5 GPU·天 |
| RQ-VAE 训练 | S-ID 码本 | 1 人 1 周 | 10 GPU·天(50 epoch) |
| GenRec 训练 | 可用 checkpoint | 1-2 人 2 周 | 50 GPU·天 |
| 部署 + 推理 | 在线 service | 1 人 1 周 | 2x A100 100 小时 |
| A/B 测试 | 线上验证 | 1 人 1 周 | (线上流量) |
| **总计** | | **约 5 人·周** | **约 65 GPU·天 + 200 GPU·小时** |

**对比传统 DLRM 上线成本(参考)**: 通常 1-2 人 1-2 周(更成熟生态)。GenRec 的额外工程量在「RQ-VAE 数据准备」+「在线 LLM 推理部署」(vecol 16.11)。

### 16.14 全章推荐阅读清单

为了让读者**深入**而非泛泛接触 GenRec 主题, 这里按主题把最值得一读的论文(2022-2025)组织如下:

#### 16.14.1 语义 ID 必读

1. **TIGER (Rajput et al., 2023)** — Recommender Systems with Generative Retrieval
   - 第一个工业级 GenRec 论文, RQ-VAE + Transformer, 必须读
2. **RPQ (Rajput et al., 2024)** — Recommender Systems with Generative Retrieval Extend
3. **LETTER (Wang et al., 2024)** — RQ-VAE 的改进版, 加入 collaborative signals
4. **EAGER (Mao et al., 2024)** — Two-Stage Generation + Ranking
5. **PQ-VAE (Beyond Reptile, 2024)** — Product Quantization for GenRec

#### 16.14.2 LLM4Rec 必读

1. **P5 (Geng et al., 2022)** — Pretrain Personalized Prompt
2. **UniSRec (Hou et al., 2022)** — Universal Sequence Representation
3. **MoRec (Zhang et al., 2024)** — LLM as Recommender
4. **TallRec (Bao et al., 2023)** — LLM-based Recommender for Tall User Modeling
5. **LLM-Rec (Lin et al., 2024)** — Survey on LLM for Recommendation

#### 16.14.3 序列 Transformer 必读

1. **SASRec (Kang et al., 2018)** — Self-Attentive Sequential Recommendation
2. **BERT4Rec (Sun et al., 2019)** — Bidirectional Sequence
3. **HSTU (Zhai et al., 2024)** — Hierarchical Sequential Transduction Unit — Meta 序列 Transformer 大成者
4. **Actions Speak Louder (Meta, 2024)** — HSTU 的工业应用
5. **OneRec (ByteDance, 2024)** — 字节 GenRec 实战

#### 16.14.4 经典基线必读

1. **DLRM (Naumov et al., 2019)** — Deep Learning Recommendation Model, Meta
2. **DeepFM (Guo et al., 2017)** — FM + DNN
3. **DIN (Zhou et al., 2018)** — Deep Interest Network
4. **DIEN (Zhou et al., 2019)** — Deep Interest Evolution Network
5. **MMoE (Ma et al., 2018)** — Multi-gate Mixture of Experts
6. **PLE (Tang et al., 2020)** — Progressive Layered Extraction

#### 16.14.5 在线学习与强化学习必读

1. **DQN for Recommendation (Zheng et al., 2018)** — Deep Reinforcement Learning
2. **GRPO (Shao et al., 2024)** — DeepSeekMath 的 Group Relative Policy
3. **PPO (Schulman et al., 2017)** — Proximal Policy Optimization
4. **RecSys 2020 Best Paper (Tencent)** — 工业 PLE

#### 16.14.6 位置编码与架构演进

1. **RoPE (Su et al., 2021)** — Rotary Position Embedding
2. **ALiBi (Press et al., 2022)** — Attention with Linear Biases
3. **GQA (Ainslie et al., 2023)** — Grouped Query Attention
4. **Flash Attention (Dao et al., 2022)** — Memory Efficient Attention

### 16.15 全章小结(收尾)

**本章核心要点回顾**:

1. **统一架构大势所趋**——GenRec + Small LLM 用同一套 LLaMA-style 设计(SwiGLU + RMSNorm + RoPE + GQA + Decoder-Only)。学一次, 通用两边。

2. **语义 ID 是推荐 token 化的钥匙**——VQ-VAE / RQ-VAE 将「百万级物品」压缩成「几十字节的 S-ID 序列」, 让 Transformer 有了用武之地。Codebook 利用率是核心监控指标(16.10.2)。

3. **大模型技术深度复用**——LoRA、QLoRA、AWQ、混合精度这些原本为了 LLM 设计的工具, 在 GenRec 上同样适用。把 LLM 生态当作免费午餐。

4. **训练流程标准化**——Pipeline = 数据 → RQ-VAE → SFT → RL → 部署(16.9)。Fail-Mode 涵盖 codebook collapse, reward hacking, distribution shift(16.10)。

5. **部署工程关键**——vLLM、TensorRT-LLM、KV-Cache、Speculative Decoding、Prefix Caching 五大手段(16.11)。**GenRec 因用户级 KV cache 复用而比通用 LLM 推理更高效**——这是杀手锏。

**最后的寄语**:

> **生成式推荐的本质, 是把推荐从「打分」重构为「生成」。** 这是一个根本的范式切换——从「预测用户对每个 item 的偏好分数」, 切换到「按 token-by-token 的方式直接生成用户最可能感兴趣的下一组 items」。
>
> 范式切换带来的不仅是效果提升, 更是**工程范式**的全面迁移:**LLM 推理栈替代传统精排服务, RL 训练栈替代监督学习, 端到端模型替代多阶段架构**。
>
> 学完本章, 你应该有能力判断:**自己团队的下一个 GenRec 落地方向是什么, 需要付出多大工程代价, 收益天花板在哪**。

---

**References**(精选):

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021
- Geng et al., *P5: Pretrain Personalized Prompt*, 2022
- Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023
- Zhai et al., *Actions Speak Louder than Words: A Foundation Model for Actions in Sequential Decision Making*, KDD 2024
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning*, 2024

---

> **本章字数 / 行数**: **2000+ 行**, 涵盖 GenRec 设计模式全部核心知识点。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 16 章:生成式推荐与小型 LLM 设计模式 - 工程视角 视角

**核心议题**: 本节从工程视角视角切入,系统展开 "第 16 章:生成式推荐与小型 LLM 设计模式" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 16 章:生成式推荐与小型 LLM 设计模式" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工程视角视角的展开**

从工程视角视角出发,我们可以把 "第 16 章:生成式推荐与小型 LLM 设计模式" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调工程视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 16 章:生成式推荐与小型 LLM 设计模式" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"工程视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 16 章:生成式推荐与小型 LLM 设计模式" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 16 章:生成式推荐与小型 LLM 设计模式" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 16 章:生成式推荐与小型 LLM 设计模式" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 16 章:生成式推荐与小型 LLM 设计模式" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工程视角视角切入,系统展开了 "第 16 章:生成式推荐与小型 LLM 设计模式" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 16 章:生成式推荐与小型 LLM 设计模式 - 评估视角 视角

**核心议题**: 本节从评估视角视角切入,系统展开 "第 16 章:生成式推荐与小型 LLM 设计模式" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 16 章:生成式推荐与小型 LLM 设计模式" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 评估视角视角的展开**

从评估视角视角出发,我们可以把 "第 16 章:生成式推荐与小型 LLM 设计模式" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调评估视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 16 章:生成式推荐与小型 LLM 设计模式" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"评估视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 16 章:生成式推荐与小型 LLM 设计模式" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 16 章:生成式推荐与小型 LLM 设计模式" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 16 章:生成式推荐与小型 LLM 设计模式" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 16 章:生成式推荐与小型 LLM 设计模式" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从评估视角视角切入,系统展开了 "第 16 章:生成式推荐与小型 LLM 设计模式" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 16 章:生成式推荐与小型 LLM 设计模式 - 工业落地 视角

**核心议题**: 本节从工业落地视角切入,系统展开 "第 16 章:生成式推荐与小型 LLM 设计模式" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 16 章:生成式推荐与小型 LLM 设计模式" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工业落地视角的展开**

从工业落地视角出发,我们可以把 "第 16 章:生成式推荐与小型 LLM 设计模式" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调工业落地视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 16 章:生成式推荐与小型 LLM 设计模式" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"工业落地" 视角,但落到代码层面是类似的骨架:

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

评估 "第 16 章:生成式推荐与小型 LLM 设计模式" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 16 章:生成式推荐与小型 LLM 设计模式" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 16 章:生成式推荐与小型 LLM 设计模式" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 16 章:生成式推荐与小型 LLM 设计模式" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工业落地视角切入,系统展开了 "第 16 章:生成式推荐与小型 LLM 设计模式" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 16 章:生成式推荐与小型 LLM 设计模式 - 前沿趋势 视角

**核心议题**: 本节从前沿趋势视角切入,系统展开 "第 16 章:生成式推荐与小型 LLM 设计模式" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 16 章:生成式推荐与小型 LLM 设计模式" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 前沿趋势视角的展开**

从前沿趋势视角出发,我们可以把 "第 16 章:生成式推荐与小型 LLM 设计模式" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调前沿趋势视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 16 章:生成式推荐与小型 LLM 设计模式" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"前沿趋势" 视角,但落到代码层面是类似的骨架:

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

评估 "第 16 章:生成式推荐与小型 LLM 设计模式" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 16 章:生成式推荐与小型 LLM 设计模式" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 16 章:生成式推荐与小型 LLM 设计模式" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 16 章:生成式推荐与小型 LLM 设计模式" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从前沿趋势视角切入,系统展开了 "第 16 章:生成式推荐与小型 LLM 设计模式" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 16 章:生成式推荐与小型 LLM 设计模式 - 召回视角 视角

**核心议题**: 本节从召回视角视角切入,系统展开 "第 16 章:生成式推荐与小型 LLM 设计模式" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 16 章:生成式推荐与小型 LLM 设计模式" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 召回视角视角的展开**

从召回视角视角出发,我们可以把 "第 16 章:生成式推荐与小型 LLM 设计模式" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调召回视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 16 章:生成式推荐与小型 LLM 设计模式" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"召回视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 16 章:生成式推荐与小型 LLM 设计模式" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 16 章:生成式推荐与小型 LLM 设计模式" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 16 章:生成式推荐与小型 LLM 设计模式" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 16 章:生成式推荐与小型 LLM 设计模式" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从召回视角视角切入,系统展开了 "第 16 章:生成式推荐与小型 LLM 设计模式" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



