---
title: "第03章-10-概率视角下的深度学习-VAE·Diffusion·Bayesian-NN"
tags:
  - 数学
  - 概率视角下的深度学习
  - VAE
  - Diffusion
  - Bayesian
  - NN
description: "本章要点：9 章我们学了概率论的语言:随机变量、分布、期望、贝叶斯定理、MLE/MAP、MCMC、假设检验。这一章把它们全部串起来,看看现代深度学习是怎么\"用概率…"
chapter_id: 数学-10-03
last_updated: 2026-07-14
---

# 第03章-10-概率视角下的深度学习-VAE·Diffusion·Bayesian-NN

9 章我们学了概率论的语言:随机变量、分布、期望、贝叶斯定理、MLE/MAP、MCMC、假设检验。**这一章把它们全部串起来,看看现代深度学习是怎么"用概率论思考"的**。
>
> - VAE (变分自编码器): **变分推断 + 神经网络**,用来"压缩+生成"
> - Diffusion (扩散模型): **一步步加噪、一步步去噪**,Stable Diffusion 的核心
> - Bayesian NN (贝叶斯神经网络): **给每个权重一个分布**,给出不确定性
> - 自回归生成 (LLM): **链式法则拆 p(x)**,GPT 系列的基础
> - GAN (对抗生成): **极小极大博弈**,深度学习早期的"图像生成王"
>
> **学完这一章,你能回答**:
>
> 1. **ELBO 怎么来? 为什么是 log p(x) 的下界?** 从 Jensen 不等式 + KL 散度推
> 2. **重参数化技巧为什么让 VAE 能端到端训练?** 梯度的来源
> 3. **DDPM 为什么用 ε-prediction? 不用 x-prediction 行不行?** 噪声 vs 信号的方差
> 4. **Diffusion 比 VAE 好在哪? 为什么 Stable Diffusion 用 Latent Diffusion?** 维度诅咒 + 计算成本
> 5. **BNN 怎么用 MC Dropout 近似? 为什么自动驾驶离不开它?** 认知不确定 vs 偶然不确定
> 6. **为什么 LLM 用自回归不用 Diffusion?** 离散 vs 连续,严格因果
>
> **本章与文件夹其他章节的关系**:
>
> - 第 1-2 章的概率空间 + 随机变量,是本章所有生成模型的"地基"
> - 第 5 章的贝叶斯定理,是 VAE、B NN 的核心
> - 第 6 章的 MLE/MAP,是 ELBO 推导的入口
> - 第 8 章的置信区间思想,在 BNN 的不确定性估计里直接复用
> - 第 9 章的 MCMC,在 BNN 的近似推断里有简化版(VI)
> - **与 第 9 章变分法 的关系**: VAE 的 ELBO 推导本质是变分法,Euler-Lagrange 方程的离散版就是 ELBO 的更新规则
> - **与 第 5 章信息论 的关系**: KL 散度 = 相对熵,ELBO 中的 KL 项是互信息的下界

---

## 1. 直觉引入:为什么深度学习是"概率生成"?

### 1.1 工程师的痛点:Stable Diffusion 里的"概率"在哪?

2022 年 Stable Diffusion 火爆之后,你大概读过博客、看了解读、玩过 demo。但面试官问你:

> "Stable Diffusion 为什么能生成图片? 它和 VAE、GAN 有什么本质区别? 训练目标是什么? 为什么是逐步加噪再逐步去噪,不是一步到位?"

你大概率答得磕磕绊绊。因为**这一类问题需要把概率论的语言翻译成深度学习的工程语言**,而大多数工程师只学过"神经网络怎么训练",没学过"神经网络怎么拟合一个分布"。

更具体地说,Stable Diffusion 的训练,核心就是一个**条件概率链**:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)
$$

每一个 $p_\theta(x_{t-1} | x_t)$ 都是**一个高斯分布**,它的均值和方差由**神经网络** (UNet) 预测。**神经网络不是输出一个数,而是输出一个分布的参数**。这就是"概率视角下的深度学习"的核心:

> **深度学习 = 用神经网络拟合一个概率分布 (或条件分布),用 MLE/MAP 训练。**

### 1.2 直觉:深度学习的三种"概率拟合"

很多工程师潜意识把深度学习当成"拟合函数 $f: x \to y$"。这只对了一半。**更准确的视角是"拟合分布"**,而且拟合哪个分布,决定了模型的种类:

| 视角 | 拟合对象 | 例子 |
| :--- | :--- | :--- |
| **判别式 (discriminative)** | $p(y \mid x)$ 条件分布 | 分类器 (softmax)、回归 (高斯输出) |
| **生成式 (generative)** | $p(x)$ 数据分布本身 | VAE、Diffusion、GAN、LLM |
| **贝叶斯 (Bayesian)** | $p(w \mid D)$ 参数后验 | Bayesian NN、Gaussian Process |

**一个具体的例子**,对比三种视角:

**任务**:给定猫狗图片,预测"猫"的概率。

- **判别式**:训练 CNN,直接输出 $\hat{p}(y=\text{猫} \mid x)$。交叉熵损失 = MLE。
- **生成式**:训练 $p(x \mid y=\text{猫})$ 和 $p(x \mid y=\text{狗})$,然后用贝叶斯反推 $p(y \mid x)$。这就是朴素贝叶斯 + 生成式分类。
- **贝叶斯**:不仅预测 $y$,还预测**权重 $w$ 的不确定性**,得到 $p(w \mid D)$。多次采样 $w$ 做预测,得到预测分布而不是点预测。

```text
       判别式                       生成式                     贝叶斯
        |                          |                          |
        v                          v                          v
   CNN → ŷ 概率               p(x|y) 建模                  p(w|D) 后验
   CrossEntropy                p(x)=∫p(x|y)p(y)dy         p(y|x)=E_p(w|D)[p(y|x,w)]
   "这图是猫的概率"             "这图是'像猫'的概率"        "预测 + 不确定性"
```

**核心区别**:判别式给一个数,生成式给数据的密度,贝叶斯给权重的不确定性。

### 1.3 三种视角的对比 (ASCII 可视化)

```text
频率派视角:数据是随机的,参数是固定的
  ┌────────────────────────────────────────┐
  │ θ (固定) → p(x|θ) → 数据 x_1,...,x_n │
  │     ↑                                   │
  │     └── 重复抽样,θ 不变,数据会变       │
  └────────────────────────────────────────┘
  MLE 找 θ̂ = argmax_θ Σ log p(x_i|θ)

贝叶斯视角:数据是固定的,参数是随机的
  ┌────────────────────────────────────────┐
  │ θ ~ p(θ) 先验 → 数据 D → p(θ|D) 后验  │
  │   ↑                  ↓                  │
  │   └── 参数有分布,后验整合先验 + 数据    │
  └────────────────────────────────────────┘
  预测 p(y|x,D) = ∫ p(y|x,θ) p(θ|D) dθ

生成式视角:显式建模 p(x) 或 p(x|y)
  ┌────────────────────────────────────────┐
  │ z ~ p(z) (隐变量) → x ~ p(x|z)         │
  │   ↑                ↓                    │
  │   └── 学 p(x) = ∫ p(x|z)p(z) dz         │
  └────────────────────────────────────────┘
  VAE、Diffusion、GAN 都在这个框架下
```

### 1.4 案例:Stable Diffusion 为什么用 DDPM?

2020 年 Ho 等人在 NeurIPS 发表 Denoising Diffusion Probabilistic Models (DDPM),第一次把"加噪-去噪"做成了 SOTA 图像生成。Stable Diffusion 用的就是 DDPM 的 Latent 版本。

**为什么选 DDPM?**

| 模型 | 缺点 | DDPM 的优势 |
| :--- | :--- | :--- |
| VAE | 生成模糊,模式坍塌 | 训练稳定,似然有下界 |
| GAN | 训练不稳定,模式坍塌 | 不需要对抗训练 |
| Flow | 限制架构 (必须可逆) | 架构自由,UNet 直接用 |
| AR (PixelCNN) | 自回归慢 | 并行训练,串行采样 |

**数学上,DDPM 的优势在于**:
- **训练目标简单**:预测噪声 ε,等价于变分下界的一个加权 MSE 估计
- **不需要对抗**:没有 min-max,梯度稳定
- **可以复用 UNet**:已有的图像分割架构直接搬过来

```python
# DDPM 训练的核心 (简化版,完整实现见 §5.6)
loss = MSE(eps, eps_theta(x_t, t))  # 预测噪声的 MSE
# eps_theta 是 UNet,输入加噪图 x_t 和时间步 t,输出预测的噪声
```

**关键洞察**:这一行 MSE,**本质上就是负对数似然的一个无偏估计**。把"概率生成"翻译成"预测噪声的回归",是 DDPM 能工程化的根本原因。

### 1.5 本章地图

```text
第 10 章:概率视角下的深度学习
├── 1. 直觉引入        (你在这里)
├── 2. 生成模型分类    (判别式 vs 生成式 vs 显式密度 vs 隐式密度)
├── 3. VAE            ← 重点 (变分推断 + 重参数化)
├── 4. Normalizing Flows (可逆变换,精确似然)
├── 5. Diffusion       ← 重点 (DDPM, ε-prediction)
├── 6. GAN 视角       (min-max, JS 散度)
├── 7. Bayesian NN    ← 重点 (MC Dropout, 不确定性)
├── 8. 自回归 LLM    (链式法则)
├── 9. 三种生成模型对比
├── 10. 评估指标 (IS, FID, Perplexity)
├── 11. PyTorch 实验  (6 个实验)
├── 12. 常见误区 + Q&A
└── 13. 与其他数学工具的联系
```

---

## 2. 生成模型的分类

### 2.1 判别式 vs 生成式

**判别式模型 (discriminative model)** 直接建模 $p(y \mid x)$ 或学到判别函数 $f(x)$。**生成式模型 (generative model)** 建模 $p(x)$ 或 $p(x, y)$,可以"凭空"采样出新的 $x$。

```text
判别式: P(y=猫 | x=图片) = 0.95          "这是猫的概率"
生成式: P(x=图片 | y=猫) 然后采样       "生成一张像猫的图片"
```

**生成式模型的终极目标**:学到一个分布 $p_\theta(x)$,使得:
1. **密度估计**:对任意 $x$,能算 $p_\theta(x)$ (或下界)
2. **采样**:能从中采样 $\tilde{x} \sim p_\theta(x)$,生成的 $\tilde{x}$ 看起来像真实数据
3. **潜在表示**:把 $x$ 压缩到低维 $z$ (隐变量)

### 2.2 显式密度 vs 隐式密度

**显式密度 (explicit density)**:能写出 $p_\theta(x)$ 的具体形式 (或可计算的近似)。

| 模型 | 显式密度? | 怎么算 |
| :--- | :--- | :--- |
| VAE | 显式 (ELBO) | $\log p(x) \geq \text{ELBO}$ |
| Flow | 显式 (精确) | 变量变换公式 |
| Diffusion | 显式 (ELBO) | 反向过程高斯 |
| AR (LLM) | 显式 (精确) | 链式法则 |
| GAN | **隐式** | 不算密度,直接采样 |

**隐式密度 (implicit density)** 模型 (如 GAN) 不直接建模 $p(x)$,只学一个采样器。**好处是自由,坏处是不能算似然,没法直接用 MLE**。

### 2.3 五大生成模型家族对比

| 家族 | 核心思想 | 训练目标 | 采样速度 | 似然 |
| :--- | :--- | :--- | :--- | :--- |
| **自回归 (AR)** | $p(x) = \prod_i p(x_i \| x_{<i})$ | MLE | 慢 (T 步) | 精确 |
| **VAE** | $p(x) = \int p(x\|z) p(z) dz$ | ELBO (下界) | 快 | 下界 |
| **GAN** | min-max 博弈 | 对抗损失 | 快 | 隐式 |
| **Flow** | 可逆变换 $x = f(z)$ | 精确 MLE | 快 | 精确 |
| **Diffusion** | 加噪-去噪 | ε-MSE | 慢 (T 步) | 下界 |

```text
生成模型宇宙

         ┌──────────── 显式密度 ────────────┐
         │                                  │
    精确似然                            下界 (ELBO)
   ┌─────┴─────┐                       ┌────┴────┐
   │           │                       │         │
  AR        Flow                     VAE     Diffusion
   │           │                       │         │
   └─────┬─────┘                       └────┬────┘
         │                                  │
         └──────── 采样需要 T 步 ────────────┘
                       │
                   AR, Diffusion

         GAN ──→ 隐式密度 ──→ 采样快,但无似然
```

### 2.4 案例:为什么 LLM 是自回归生成?

GPT 系列 (GPT-2/3/4) 用的是自回归生成,因为**文本是离散 token**。

**为什么 Diffusion 在离散数据上难?**

Diffusion 的核心是"加噪-去噪",加噪需要**连续空间** (高斯噪声)。离散 token 没有"加一点点噪声"的自然定义。Diffusion 在离散空间上有过尝试 (D3PM、Diffusion-LM),但效果始终不如自回归。

**自回归的数学**:

$$
p(x) = p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i | x_1, \ldots, x_{i-1})
$$

每个 $p(x_i | x_{<i})$ 由 Transformer 输出 softmax 得到。**训练 = MLE** (最大化 token 的对数似然)。

```python
# 自回归训练的核心 (简化)
logits = transformer(input_ids[:, :-1])   # 预测下一个 token
loss = cross_entropy(logits, input_ids[:, 1:])  # MLE
```

### 2.5 案例:Stable Diffusion 是怎么选 Diffusion 的?

Stable Diffusion 的"三层蛋糕":

```text
文本 prompt ──→ CLIP text encoder ──→ 文本嵌入 c
                                          │
                                          ↓
       随机噪声 z_T ──→ UNet (DDPM 去噪) ──→ 清晰图像 z_0
                          ↑
                       条件: c
                          ↓
       VAE decoder ──→ 最终像素图像 x
```

**为什么中间是 Latent Diffusion?**

直接在像素空间 (512×512×3) 做 Diffusion,UNet 的计算量爆炸。Rombach 等人 (2022) 提出**先用一个预训练的 VAE 把图像压缩到 latent 空间 (64×64×4)**,在 latent 上做 Diffusion,最后 VAE decoder 把 latent 解码回像素。

**为什么选 DDPM + UNet?**

| 候选 | 问题 | DDPM 的优势 |
| :--- | :--- | :--- |
| GAN | 训练不稳,文本可控性差 | 训练稳定,可加 cross-attention |
| VAE | 模糊,文本可控性差 | 高质量,易加条件 |
| AR (PixelCNN) | 太慢 | 不需要逐像素生成 |

**关键洞察**:DDPM 的反向过程 $p_\theta(x_{t-1} | x_t)$ 天然支持**条件生成** (在 UNet 里 cross-attend 文本嵌入),不需要重新设计架构。

---

## 3. VAE 变分自编码器 (重点)

### 3.1 目标:建模 $p(x) = \int p(x|z) p(z) dz$

**问题**: 给定一堆图片 $\{x_i\}$,我们想学一个模型,使得 (1) 能压缩 $x$ 到低维 $z$,(2) 能从 $z$ 还原 $x$,(3) 能从先验 $p(z)$ 采样生成新图。

**朴素想法**: $p(x) = \int p(x|z) p(z) dz$,其中 $p(z)$ 是简单先验 (如标准高斯),$p(x|z)$ 是"解码器"。

**难点**: 后验 $p(z|x) = p(x|z) p(z) / p(x)$ 难算,因为分母 $p(x)$ 是高维积分。

```text
编码器 (encoder)         隐空间           解码器 (decoder)
   x ──→ q(z|x) ──→ z ──→ p(x|z) ──→ x̂
                        ↑
                        └── 从 p(z) 采样
```

### 3.2 难点:后验 $p(z|x)$ 难算

$p(z|x) = p(x|z) p(z) / p(x)$,其中 $p(x) = \int p(x|z) p(z) dz$ 是 $z$ 上的积分。$z$ 通常是 64-256 维,这个积分没有闭式解。

**VAE 的解决方案**: 引入**变分分布** $q_\phi(z|x)$ (用一个神经网络参数化),让它**近似**真实后验 $p(z|x)$。

**怎么度量"近似"?** 用 KL 散度:

$$
q_\phi^*(z|x) = \arg\min_{q} \text{KL}(q(z|x) \| p(z|x))
$$

但 $\text{KL}(q \| p(z|x))$ 也难算 (因为有未知的 $p(z|x)$)。**变分推断的经典 trick**:用 ELBO 把这个 KL 转化成可计算的量。

### 3.3 变分推断:用 $q_\phi(z|x)$ 近似 $p(z|x)$

**核心数学**:对任意分布 $q(z|x)$,

$$
\log p(x) = \log \int p(x|z) p(z) dz = \log \int q(z|x) \frac{p(x|z) p(z)}{q(z|x)} dz
$$

$$
= \log \mathbb{E}_{q(z|x)} \left[ \frac{p(x|z) p(z)}{q(z|x)} \right]
$$

用 Jensen 不等式 ($\log$ 是凹函数,所以 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$):

$$
\log p(x) \geq \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x|z) p(z)}{q(z|x)} \right]
$$

**展开右端**:

$$
= \mathbb{E}_{q(z|x)}[\log p(x|z)] + \mathbb{E}_{q(z|x)}\left[\log \frac{p(z)}{q(z|x)}\right]
$$

$$
= \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{重建项 (reconstruction)}} - \underbrace{\text{KL}(q(z|x) \| p(z))}_{\text{正则项 (regularization)}}
$$

**这就是 ELBO (Evidence Lower Bound)**:

$$
\boxed{\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z)) =: \text{ELBO}}
$$

**VAE 的训练 = 最大化 ELBO**。

### 3.4 ELBO 推导的 5 个手算例子

**例子 1:一维高斯,先验 $p(z) = \mathcal{N}(0, 1)$,近似 $q(z|x) = \mathcal{N}(\mu, \sigma^2)$**

KL 散度公式 (标准正态 vs 一般正态):

$$
\text{KL}(q \| p) = \frac{1}{2}\left(\sigma^2 + \mu^2 - 1 - \log \sigma^2\right)
$$

**验证**: 取 $\mu = 0, \sigma = 1$,代入得 $\text{KL} = \frac{1}{2}(1 + 0 - 1 - 0) = 0$。✓ (同分布 KL 为 0)

**例子 2:$\mu = 1, \sigma = 1$** (只是均值偏移):

$$
\text{KL} = \frac{1}{2}(1 + 1 - 1 - 0) = 0.5 \text{ nats}
$$

**例子 3:$\mu = 0, \sigma = 2$** (方差大):

$$
\text{KL} = \frac{1}{2}(4 + 0 - 1 - \log 4) = \frac{1}{2}(3 - 1.386) = 0.807 \text{ nats}
$$

**例子 4:重建项简化**。如果 $p_\theta(x|z) = \mathcal{N}(\hat{x}, I)$,则 $\log p_\theta(x|z) = -\frac{1}{2}\|x - \hat{x}\|^2 + \text{const}$。**最大化 ELBO 的重建项 = 最小化 $\|x - \hat{x}\|^2$**,就是 MSE 损失。

**例子 5:VAE 完整 ELBO** (二维 $z$):

设 $q_\phi(z|x) = \mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$, $p(z) = \mathcal{N}(0, I)$。则:

$$
\text{ELBO} = -\frac{1}{2} \mathbb{E}_{q}[\|x - \hat{x}\|^2] - \frac{1}{2}\sum_{i=1}^{2}(\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2)
$$

**训练时** $\mathbb{E}_q[\cdot]$ 用 1 个样本蒙特卡洛近似:

```python
z = mu + sigma * eps  # eps ~ N(0,I), 重参数化
x_hat = decoder(z)
recon_loss = ((x - x_hat)**2).sum()
kl_loss = 0.5 * (sigma**2 + mu**2 - 1 - torch.log(sigma**2)).sum()
loss = recon_loss + kl_loss  # 最大化 ELBO = 最小化 -ELBO = recon + KL
```

### 3.5 ELBO 是下界的几何直觉

```text
log p(x) (固定,真实值,上界)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 真实天花板,不可达

ELBO = log p(x) - KL(q || p(z|x))
   ━━━━━━━━━━━━━━━━━━━━━━ 随训练上升,但永远 ≤ log p(x)

KL(q || p(z|x)) = log p(x) - ELBO
                  ↑ "gap" 越小,q 越接近真实后验
```

**关键性质**:
- $\text{ELBO} \leq \log p(x)$ 永远成立
- $\text{ELBO} = \log p(x)$ 当且仅当 $q(z|x) = p(z|x)$
- 训练时我们**最大化 ELBO**,等价于**同时最大化 $\log p(x)$ (建模质量) 和最小化 $\text{KL}(q \| p(z|x))$ (近似质量)**

### 3.6 重参数化技巧 (Reparameterization Trick)

**问题**: $z \sim q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$,要从 $q$ 采样才能算 $\mathbb{E}_q[\log p_\theta(x|z)]$,但**采样操作不可导**,反向传播断了。

**解决**:重参数化,把随机性从 $z$ 转移到外部噪声 $\epsilon$:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这样 $z$ 是 $\mu, \sigma, \epsilon$ 的**确定性函数**,对 $\mu, \sigma$ 可导。

```text
没有重参数化:                 有重参数化:
z ~ N(μ, σ²)                  ε ~ N(0, I)
   ↓ (采样,不可导)             ↓
loss                          z = μ + σ·ε (可微!)
                                ↓
                              loss
```

**代码** (PyTorch):

```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)  # 外部噪声
    return mu + eps * std
```

**为什么重要?**
- 没有它,VAE 没法端到端训练 (只能训 decoder,encoder 训不动)
- 它是**把"随机采样"翻译成"确定性变换 + 外部噪声"的标准技巧**,在强化学习 (策略梯度) 里也有用

### 3.7 PyTorch 实现 VAE (MNIST)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # 编码器: x → (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # 解码器: z → x̂
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(x_recon, x, mu, log_var):
    # 重建项: -E[log p(x|z)] ≈ BCE (因为 x ∈ [0,1])
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL 项: KL(q(z|x) || N(0, I))
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl  # 最小化 -ELBO

# 训练循环
model = VAE()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, _ in dataloader:
    x = x.view(-1, 784)
    x_recon, mu, log_var = model(x)
    loss = vae_loss(x_recon, x, mu, log_var)
    optim.zero_grad(); loss.backward(); optim.step()
```

### 3.8 案例:为什么 VAE 生成图像"模糊"?

**直观解释**:

| 模型 | 隐空间结构 | 生成图像 |
| :--- | :--- | :--- |
| VAE | **模糊的隐空间** (KL 强制接近先验) | 模糊但多样 |
| GAN | 没有显式隐空间 | 锐利但可能模式坍塌 |
| Diffusion | 显式逐步去噪 | 锐利且多样 |

**VAE 模糊的数学原因**:

1. **KL 正则太强**: KL 项强制 $q(z|x)$ 接近标准正态,导致 $z$ 空间"被压扁"。相邻 $z$ 的重建相似。
2. **像素级高斯假设**: $p_\theta(x|z) = \mathcal{N}(\hat{x}, I)$,方差固定,模型只能输出"均值",自然模糊。
3. **重建项是 MSE**: MSE 损失天然偏好模糊预测 (多模式时取均值)。

**解决方案**:
- 减弱 KL 权重 ($\beta$-VAE, $\beta < 1$)
- 用 PixelCNN 自回归解码器代替高斯
- 改用感知损失 (perceptual loss) 而不是像素 MSE
- 直接上 Diffusion (DDPM 解决了所有这些)

---

## 4. Normalizing Flows

### 4.1 思想:可逆变换精确计算似然

**核心问题**: VAE 用 ELBO (下界) 近似 $\log p(x)$,有误差。能不能**精确**计算似然?

**Normalizing Flows 的答案**: 用**一系列可逆变换**把复杂分布 $p(x)$ 变成简单分布 $p(z)$ (如标准正态)。

设 $z = f(x)$ 是可逆变换,$f^{-1}$ 存在。那么变量变换公式:

$$
p(x) = p(z) \left| \det \frac{\partial f}{\partial x} \right|, \quad z = f(x)
$$

**对数似然**:

$$
\log p(x) = \log p(z) + \log \left| \det \frac{\partial f}{\partial x} \right|
$$

```text
简单分布 p(z)              复杂分布 p(x)
   ___                          ____
  /   \                        /    \
 / N(0,1) \    ←── z = f(x) ── /  数据  \
/_________\                    /__________\

可逆变换 f, 雅可比行列式易算
```

### 4.2 RealNVP:一个具体的可逆变换

RealNVP (Dinh et al. 2017) 的核心是**仿射耦合层**:

$$
\begin{cases}
y_1 = x_1 \\
y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
\end{cases}
$$

其中 $s, t$ 是神经网络,$x_1, x_2$ 是 $x$ 拆成两半。

**为什么可逆?**

$$
\begin{cases}
x_1 = y_1 \\
x_2 = (y_2 - t(y_1)) \odot \exp(-s(y_1))
\end{cases}
$$

**为什么雅可比行列式易算?**

雅可比矩阵是三角矩阵:

$$
\frac{\partial y}{\partial x} = \begin{pmatrix} I & 0 \\ \frac{\partial y_2}{\partial x_1} & \text{diag}(\exp(s(x_1))) \end{pmatrix}
$$

行列式 = $\prod_i \exp(s(x_1)_i)$,**计算量与维度线性关系,不是 $O(d^3)$**。

### 4.3 Normalizing Flows 的 5 个手算例子

**例子 1:1 维平移** $z = x + \mu$

$\frac{dz}{dx} = 1$, $\log p(x) = \log p(z) + \log 1 = \log p(z)$。位置平移不改变密度。

**例子 2:1 维缩放** $z = \sigma x$

$\frac{dz}{dx} = \sigma$, $\log p(x) = \log p(z) + \log \sigma$。方差变化要补一项。

**例子 3:二维耦合层** $y_1 = x_1, y_2 = x_2 \cdot s(x_1) + t(x_1)$

设 $s(x_1) = 2$, $t(x_1) = 1$:

$\frac{\partial(y_1, y_2)}{\partial(x_1, x_2)} = \begin{pmatrix} 1 & 0 \\ \partial y_2/\partial x_1 & 2 \end{pmatrix}$

行列式 = 2。$\log p(x) = \log p(y) + \log 2$。

**例子 4:Glow 的 $1 \times 1$ 卷积** (可逆通道混合)

雅可比是置换矩阵,行列式 = $\pm 1$。**几乎免费**的通道混合。

**例子 5:Log 概率的累加**。多层 Flow 串联:

$$
\log p(x) = \log p(z_K) + \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|
$$

每加一层,log-det 加一项。训练 = 最大化精确对数似然。

### 4.4 案例:为什么 Flow 在 2020 年后被 Diffusion 取代?

| 维度 | Flow (2018) | Diffusion (2020+) |
| :--- | :--- | :--- |
| 似然 | **精确** ✓ | 下界 (ELBO) ✗ |
| 架构限制 | **必须可逆** ✗ | 任意 UNet ✓ |
| 采样质量 | 中 | **高** ✓ |
| 训练稳定 | 较稳定 | 稳定 |
| 应用范围 | 受限 | 通用 |

**关键原因**:
1. **架构限制**: Flow 必须可逆,限制了容量 (不能随便加深、不能随便加残差)
2. **似然精确 vs 视觉质量**: 精确似然不等于好样本。Diffusion 的下界 ELBO 已经够好,样本质量远超 Flow
3. **可扩展性**: UNet 是图像生成的"事实标准",Diffusion 直接用 UNet,Flow 要重新设计可逆 CNN

---

## 5. Diffusion Model (重点)

### 5.1 思想:前向加噪,反向去噪

**Diffusion 的核心直觉**:

1. **前向过程 (forward)**: 真实图片 $x_0$ 一步步加高斯噪声,经过 $T$ 步变成纯噪声 $x_T \sim \mathcal{N}(0, I)$。
2. **反向过程 (reverse)**: 从纯噪声 $x_T$ 一步步去噪,最终得到清晰图片 $x_0$。
3. **训练**: 神经网络 $p_\theta$ 学习"去噪一步" $p_\theta(x_{t-1} | x_t)$。

```text
前向 (固定,加噪)              反向 (学习,去噪)
x_0 ──→ x_1 ──→ ... ──→ x_T     x_T ──→ x_{T-1} ──→ ... ──→ x_0
真实图    噪一点     纯噪声       纯噪声   去一点      清晰图
   ↑                                    ↑
   └── 训练数据                          └── 神经网络学
```

### 5.2 前向过程:$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$

**马尔可夫链**: 每一步独立加噪:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

其中 $\beta_t \in (0, 1)$ 是**噪声调度 (noise schedule)**,通常从 $\beta_1 = 10^{-4}$ 线性增长到 $\beta_T = 0.02$。

**为什么用 $\sqrt{1-\beta_t}$ 而不是 1?** 保证 $x_t$ 的方差接近常数 (不爆炸)。

**关键性质:任意 $t$ 步的闭式解**:

定义 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$,则:

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

**证明** (归纳):

$t=1$: $q(x_1 | x_0) = \mathcal{N}(\sqrt{\alpha_1} x_0, (1-\alpha_1) I)$,成立。

$t \to t+1$:

$$
q(x_{t+1} | x_0) = \int q(x_{t+1} | x_t) q(x_t | x_0) dx_t
$$

两个高斯的卷积还是高斯,均值/方差相加,得到 $\mathcal{N}(\sqrt{\alpha_{t+1}} \cdot \sqrt{\bar{\alpha}_t} x_0, (\alpha_{t+1}(1-\bar{\alpha}_t) + (1-\alpha_{t+1})) I)$

化简 $\alpha_{t+1}(1-\bar{\alpha}_t) + (1-\alpha_{t+1}) = 1 - \bar{\alpha}_{t+1}$。✓

### 5.3 反向过程:$p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

如果前向步长 $\beta_t$ 足够小,$q(x_{t-1} | x_t)$ 也近似高斯。我们用神经网络学:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

实践中 $\Sigma_\theta(x_t, t) = \sigma_t^2 I$, $\sigma_t^2 = \beta_t$ (固定),只学 $\mu_\theta$。

### 5.4 训练目标:ε-prediction (Ho et al. 2020)

**ELBO 推导** (简化):

$$
\log p(x_0) \geq \text{ELBO} = \mathbb{E}_q \left[ \log p_\theta(x_0 | x_1) - \sum_{t=2}^{T} \text{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) \right]
$$

**关键观察**: 真实反向过程的后验 $q(x_{t-1}|x_t, x_0)$ 有闭式解 (因为给定 $x_0$ 后,前向过程的条件高斯):

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

其中 $\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$。

**KL 两个高斯** 的闭式解:

$$
\text{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

**当 $\Sigma_\theta = \beta_t I$ 固定** 时,KL 项只取决于 $\mu_\theta$ 和真均值 $\tilde{\mu}_t$ 的 MSE。

**Ho et al. 2020 的关键简化**: 把 $\mu_\theta$ 重参数化为**预测噪声** $\epsilon_\theta$:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

代入 ELBO,经过一番化简 (忽略权重项),**训练目标**就是:

$$
\boxed{L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]}
$$

**白话**: 随机选时间步 $t$、加噪图 $x_t$、真实噪声 $\epsilon$,训练 UNet 预测噪声。

### 5.5 Diffusion 的 5 个手算例子

**例子 1:线性噪声调度,$\beta_t = 0.02$, $T=5$**

$\alpha_t = 0.98$, $\bar{\alpha}_t = 0.98^t$:

| $t$ | $\bar{\alpha}_t$ | $x_t$ 的均值系数 | $x_t$ 的方差 |
| :--- | :--- | :--- | :--- |
| 0 | 1.000 | 1.000 | 0 |
| 1 | 0.980 | 0.990 | 0.020 |
| 2 | 0.960 | 0.980 | 0.040 |
| 5 | 0.904 | 0.951 | 0.096 |

**例子 2:加噪一步**。设 $x_0 = 1$, $\bar{\alpha}_1 = 0.98$, $\epsilon \sim \mathcal{N}(0, 1)$ (取 0.5):

$x_1 = \sqrt{0.98} \cdot 1 + \sqrt{0.02} \cdot 0.5 = 0.990 + 0.0707 = 1.061$

**例子 3:训练目标**。设 $t=2$, $x_0 = 1$, $\bar{\alpha}_2 = 0.96$, $\epsilon \sim \mathcal{N}(0, 1)$ (取 0.3):

$x_2 = \sqrt{0.96} \cdot 1 + \sqrt{0.04} \cdot 0.3 = 0.980 + 0.060 = 1.040$

UNet 预测 $\hat{\epsilon} = \epsilon_\theta(1.040, 2)$,损失 = $(0.3 - \hat{\epsilon})^2$。

**例子 4:采样一步 (反向)**。设 $x_2 = 1.040$, UNet 预测 $\hat{\epsilon} = 0.3$:

$$
\mu_\theta(x_2, 2) = \frac{1}{\sqrt{0.98}} \left(1.040 - \frac{0.02}{\sqrt{0.04}} \cdot 0.3 \right) = 1.010 \cdot (1.040 - 0.300) = 0.748
$$

$x_1 \sim \mathcal{N}(0.748, 0.02)$

**例子 5:完整 T=3 步采样**。从 $x_3 \sim \mathcal{N}(0, 1)$ (假设) 逐步去噪到 $x_0$。每步都用上一步的预测和真实采样的高斯噪声,典型算法:

```python
x = torch.randn(...)  # x_T ~ N(0, I)
for t in reversed(range(T)):
    eps_pred = unet(x, t)
    x = (x - beta_t / sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_t)
    if t > 0:
        x += sqrt(beta_t) * torch.randn_like(x)
```

### 5.6 DDPM 完整 PyTorch 实现 (MNIST)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # 正弦位置编码
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, dtype=torch.float32) *
                          (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, time_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.ReLU())
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, in_ch, 3, padding=1)
        self.t_proj = nn.Linear(time_dim, 32)
    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        t_emb = self.t_proj(self.time_emb(t))[:, :, None, None]
        h = F.relu(self.conv3(h + t_emb))
        return self.conv4(h)

class DDPM(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = SimpleUNet()

    def q_sample(self, x0, t, noise):
        """前向加噪: x_t = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε"""
        a_bar = self.alpha_bars[t][:, None, None, None].to(x0.device)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, n, img_shape, device):
        x = torch.randn(n, *img_shape, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps_pred = self.model(x, t_batch)
            a = self.alphas[t].to(device)
            a_bar = self.alpha_bars[t].to(device)
            x = (x - (1 - a) / torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a)
            if t > 0:
                x += torch.sqrt(self.betas[t]) * torch.randn_like(x)
        return x

# 训练
ddpm = DDPM(T=1000)
opt = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
for x, _ in dataloader:
    loss = ddpm.loss(x.to(device))
    opt.zero_grad(); loss.backward(); opt.step()
# 采样
samples = ddpm.sample(16, (1, 28, 28), device)
```

### 5.7 案例:Stable Diffusion 用 Latent Diffusion 的原因

**问题**: 直接在 512×512×3 = 786432 维像素空间做 Diffusion,UNet 一次前向要算几 GB 显存。

**Latent Diffusion** (Rombach et al. 2022):

1. **预训练 VAE**: 编码器把图片压到 64×64×4 = 16384 维 latent (压缩 48 倍)
2. **在 latent 上做 DDPM**: UNet 在 64×64 上跑,显存省一个量级
3. **VAE decoder 解码回像素**: 最终得到 512×512 图

```text
像素空间 (786K 维)              Latent 空间 (16K 维)
       │                              │
   VAE encode                    DDPM 去噪
       ↓                              ↓
   Latent z_0 ──→ DDPM 加噪 ──→ z_T → 反向去噪 → z_0
                                          ↓
                                    VAE decode
                                          ↓
                                       像素图
```

**为什么不用 Flow 或 VAE 直接生成?** 因为 DDPM 的训练稳定 + UNet 已经成熟,Latent Diffusion 是"站在巨人肩膀上"。

### 5.8 案例:为什么 Diffusion 比 VAE/GAN 好?

| 维度 | VAE | GAN | Diffusion |
| :--- | :--- | :--- | :--- |
| 训练稳定性 | 稳定 | **不稳定 (mode collapse)** | 稳定 |
| 似然 | 下界 | **隐式 (无)** | 下界 |
| 样本质量 | **模糊** | 锐利但可能模式坍塌 | **锐利且多样** ✓ |
| 模式覆盖 | 较好 | **差 (模式坍塌)** | 好 ✓ |
| 采样速度 | **快 (1 步)** | 快 | **慢 (T=1000 步)** |
| 文本条件 | 难加 | 难加 (需复杂工程) | **天然支持 (cross-attention)** ✓ |

**关键洞察**:
- **Diffusion 的训练等价于多步去噪**。每一步都是简单的回归问题,梯度稳定
- **Diffusion 的多步采样**。通过把复杂分布拆成多步"局部去噪",绕过了单步生成的难题
- **Diffusion 的 latent 化**。解决了高维计算问题,催生了 Stable Diffusion

### 5.9 Score-based Diffusion 视角 (补充,工程向)

Song & Ermon (2019) 提出**分数函数** (score function):

$$
\nabla_x \log p(x) = \text{score}
$$

score 就是对数密度对 $x$ 的梯度,指向密度增长最快的方向。**关键洞察**:如果能学 $\nabla_x \log p(x)$,就能用朗之万动力学 (Langevin dynamics) 采样:

$$
x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z, \quad z \sim \mathcal{N}(0, I)
$$

**与 DDPM 的关系**: DDPM 的 $\epsilon$-prediction 等价于 score prediction:

$$
\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \nabla_x \log p(x_t)
$$

**为什么重要**: 连续时间 Diffusion (Score-based SDE) 给出了**统一的数学框架**,把 DDPM、Score-based、连续归一化流 (CNF) 都纳入同一个 SDE 视角。公式上更优美,也催生了**概率流 ODE** (probability flow ODE),可以用更少步数采样 (DDIM, DPM-Solver)。

**工程影响**: 现代 Diffusion 推理加速器 (DDIM 50 步、Consistency Model 1 步) 都基于这个连续视角。

### 5.10 经典问题:为什么 DDPM 的前向过程必须是"高斯"?

理论上,任何"逐渐加噪"到简单分布的过程都可以做 Diffusion。但**高斯**有几个不可替代的好处:

1. **闭式解**: $q(x_t|x_0)$ 有解析形式,可以直接采样
2. **真实后验 $q(x_{t-1}|x_t, x_0)$ 也是高斯**: 训练目标化简为 MSE
3. **重参数化方便**: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 是线性的
4. **卷积性质**: 高斯 + 高斯 = 高斯

**如果换成均匀分布**: 后验无闭式解,训练目标复杂得多。

---

## 6. GAN 视角

### 6.1 极小极大博弈

GAN (Goodfellow et al. 2014) 同时训练两个网络:

- **生成器 $G$**: 从噪声 $z$ 生成假样本 $G(z)$,想骗过判别器
- **判别器 $D$**: 判断 $x$ 是真 (来自数据) 还是假 (来自 $G$),输出 $D(x) \in [0, 1]$

**目标函数**:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

**直觉博弈**:

```text
判别器: "这张图是真的!"     (D(x) → 1)
         "这张图是假的!"     (D(G(z)) → 0)

生成器: "我做的图更像真的!"  (D(G(z)) → 1)
```

### 6.2 JS 散度 vs KL 散度

**理论分析** (Goodfellow 2014): GAN 的最优判别器是

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}
$$

代入目标函数,等价于最小化 **JS 散度**:

$$
\min_G V(D^*, G) = 2 \text{JS}(p_{\text{data}} \| p_G) - 2\log 2
$$

**JS 散度 vs KL 散度**:

$$
\text{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx
$$

$$
\text{JS}(P \| Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M), \quad M = \frac{P+Q}{2}
$$

**关键区别**:
- KL **不对称**: $\text{KL}(P \| Q) \neq \text{KL}(Q \| P)$
- JS **对称**: $\text{JS}(P \| Q) = \text{JS}(Q \| P)$
- JS 是有界 $[0, \log 2]$,KL 可能无穷

### 6.3 GAN 训练不稳定的数学原因

**核心问题:梯度消失**

当 $p_G$ 与 $p_{\text{data}}$ 的**支撑集不重叠**时 (高维常见),JS 散度 = $\log 2$ (常数),**梯度 = 0**,生成器学不到东西。

```text
低维: p_data 和 p_G 可能重叠     → JS 散度有梯度
高维: p_data 和 p_G 几乎不重叠   → JS = log 2, 梯度为 0
```

**Wasserstein GAN (WGAN)** 的解决方案: 用 Wasserstein 距离代替 JS 散度:

$$
W(p_{\text{data}}, p_G) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_G)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]
$$

**W 距离的关键性质**: 即使两个分布不重叠,$W$ 仍然有梯度 (因为它看的是"距离"而不是"是否重叠")。

### 6.4 案例:为什么 GAN 训练"难"?

**5 个常见训练失败模式**:

1. **模式坍塌 (mode collapse)**: $G$ 只生成几种样本,多样性丧失
2. **判别器太强**: $D$ 完美区分真假, $G$ 拿不到梯度
3. **训练震荡**: $G$ 和 $D$ 互相追逐,不收敛
4. **梯度消失**: $D$ 太强,$G$ 早期就没梯度
5. **评估困难**: 没有似然,只能用 FID/IS 间接评估

**工程经验**: 实际项目里,**Diffusion 已经基本取代 GAN** (除了 StyleGAN 这种特定场景)。GAN 仍活跃在:
- StyleGAN 系列 (人脸生成)
- 一些对抗训练 (域适应)
- 简单的轻量场景 (StyleGAN-T for text-to-image)

---

## 7. Bayesian Neural Network (重点)

### 7.1 模型:$p(y|x, w), w \sim p(w)$

**普通神经网络**: 学一个点估计 $\hat{w}$,预测 $\hat{y} = f(x; \hat{w})$。

**贝叶斯神经网络 (BNN)**: 把权重当成随机变量,有先验 $p(w)$,给定数据 $D$ 后,有权重后验 $p(w | D)$。

$$
p(w | D) = \frac{p(D | w) p(w)}{p(D)}
$$

**预测**: 不再是一个数,而是一个分布:

$$
p(y | x, D) = \int p(y | x, w) p(w | D) dw
$$

**核心优势**: **不确定性估计**。

### 7.2 后验 $p(w|D)$ 难算

$p(w|D)$ 的分母 $p(D) = \int p(D|w) p(w) dw$ 是**权重空间的高维积分**,指数级难算。

### 7.3 三种近似方法

**方法 1: 变分推断 (VI)**

用一个简单分布 $q_\phi(w)$ (如各向同性高斯)近似 $p(w|D)$,最大化 ELBO (熟悉的配方):

$$
\text{ELBO} = \mathbb{E}_{q(w)}[\log p(D|w)] - \text{KL}(q(w) \| p(w))
$$

**方法 2: MCMC** (第 9 章)

用 Metropolis-Hastings、HMC 等采样器从 $p(w|D)$ 采样多个 $w$,然后做蒙特卡洛平均:

$$
p(y | x, D) \approx \frac{1}{S} \sum_{s=1}^{S} p(y | x, w_s)
$$

**方法 3: Laplace 近似**

在 MLE 解 $\hat{w}$ 附近做二阶 Taylor 展开:

$$
p(w | D) \approx \mathcal{N}(w; \hat{w}, H^{-1})
$$

其中 $H$ 是 Hessian 矩阵,在 $\hat{w}$ 处求值。**适合已经训练好的网络做"近似贝叶斯化"**。

### 7.4 MC Dropout 近似 BNN

**Gal & Ghahramani 2016** 的关键洞察: **训练时用 Dropout + 测试时也用 Dropout,等价于变分推断**。

具体地,每次预测时,前向传播 $T$ 次,每次都开启 Dropout:

$$
p(y | x, D) \approx \frac{1}{T} \sum_{t=1}^{T} p(y | x, w_t), \quad w_t \sim q_\phi(w)
$$

**为什么等价?**

- Dropout 的随机 mask 等价于从 $q_\phi(w) = \prod_i \text{Bernoulli}(p_i) \cdot \mathcal{N}(w_i; \mu_i, \sigma_i^2)$ 采样
- $T$ 次前向传播 = 蒙特卡洛近似后验预测分布

**优势**: 几乎零成本 (只需在测试时保留 Dropout,跑 $T$ 次)。**工业界最常用的 BNN 近似**。

```python
class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)  # 关键: training=True

def mc_predict(model, x, T=50):
    model.train()  # 开启 Dropout
    preds = [model(x) for _ in range(T)]
    mean = torch.stack(preds).mean(0)
    std = torch.stack(preds).std(0)  # 不确定性
    return mean, std
```

### 7.5 BNN 的 5 个手算例子

**例子 1:线性回归 BNN**

设 $y = wx + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, $w \sim \mathcal{N}(0, \sigma_p^2)$。

观测 $D = \{(x_1, y_1)\}$。后验:

$$
p(w | D) = \mathcal{N}\left(w; \frac{\sigma_p^2 x_1 y_1}{\sigma_n^2 + \sigma_p^2 x_1^2}, \frac{\sigma_p^2 \sigma_n^2}{\sigma_n^2 + \sigma_p^2 x_1^2}\right)
$$

**例子 2:预测的不确定性**

取 $\sigma_p = 1, \sigma_n = 0.1, x_1 = 1, y_1 = 2$:

均值 $= \frac{1 \cdot 1 \cdot 2}{0.01 + 1} = \frac{2}{1.01} \approx 1.98$

方差 $= \frac{0.01}{1.01} \approx 0.0099$,标准差 $\approx 0.099$

**例子 3:数据少时不确定性大**

若 $x_1 = 0.001$ (几乎没观测):

均值 $\approx \frac{1 \cdot 0.001 \cdot y_1}{0.01 + 10^{-6}} \approx 0.1 y_1$

方差 $\approx \frac{0.01}{0.01} = 1.0$,标准差 = 1.0,远大于例子 2。

**解释**: 数据点离原点远,后验几乎等于先验 (没有信息)。

**例子 4:认知不确定 vs 偶然不确定**

- **认知不确定 (epistemic)**: 模型不知道,$w$ 后验分散。**MC Dropout 能捕捉**
- **偶然不确定 (aleatoric)**: 数据本身噪声,$y$ 的方差。**所有方法都能捕捉**

**例子 5:MC Dropout 的预测方差**

预测 $T=10$ 次,得到均值 $\mu$ 和标准差 $\sigma$。

- **训练集内点**: $\sigma$ 小 (低认知不确定)
- **训练集外点**: $\sigma$ 大 (高认知不确定,模型不确定)

**这是 BNN 的核心价值**: 在没见过的数据上,**模型会"知道自己不知道"**。

### 7.6 PyTorch 实现 MC Dropout BNN (回归)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutBNN(nn.Module):
    def __init__(self, in_dim, hidden=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            MCDropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(),
            MCDropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_with_uncertainty(self, x, T=50):
        # 关键: model.train() 让 Dropout 在推理时也激活
        self.train()
        preds = torch.stack([self(x) for _ in range(T)])
        return preds.mean(0), preds.std(0)

# 使用
model = MCDropoutBNN(in_dim=1)
# ... 训练同普通回归 (MSE) ...
mean, std = model.predict_with_uncertainty(x_test, T=100)
# std 大 → 模型不确定
```

### 7.7 案例:为什么 BNN 在 RL 探索中重要?

**强化学习的探索-利用权衡 (exploration-exploitation)**:

- **利用 (exploitation)**: 用当前最佳策略
- **探索 (exploration)**: 试新动作

**BNN 的角色**: 知道"模型不确定的动作" → 优先探索。

**算法举例 (Bootstrapped DQN, Deep Ensembles)**:

```text
对每个状态 s:
  不确定性高 → 选 Q 值方差最大的动作 (探索)
  不确定性低 → 选 Q 值均值最大的动作 (利用)
```

**真实案例**: Google's "Deep Exploration via Bootstrapped DQN" (Osband et al. 2016) 用 BNN 思路显著提升 Atari 游戏分数。

### 7.8 案例:为什么自动驾驶用 BNN 估计不确定性?

**安全关键场景**:

- **模型预测**: 前车 2 秒后位置 $\hat{y}$
- **不确定性**: 预测的标准差 $\sigma$
- **决策规则**: 如果 $\sigma$ 太大 (如 > 1 米),**降速或请求人类接管**

**为什么普通 NN 不行?**

普通 NN 输出点估计 $\hat{y}$,**无法区分**:
- "很确定, $\hat{y} = 1.5$m, $\sigma = 0.1$"
- "很犹豫, $\hat{y} = 1.5$m,但其实可能在 [0, 3]m"

**实际部署**: Tesla、Waymo 等公司的预测模块都用**集成 + 不确定性估计**,有的直接用 MC Dropout。

---

## 8. 自回归生成 (LLM)

### 8.1 思想:$p(x) = \prod_i p(x_i | x_{<i})$

**自回归 (autoregressive)** 把联合分布拆成条件分布的乘积 (链式法则):

$$
p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i | x_1, \ldots, x_{i-1})
$$

每个 $p(x_i | x_{<i})$ 由神经网络 (通常是 Transformer) 输出 softmax 得到。

```text
x_1 → x_2 → x_3 → ... → x_n
 ↓     ↓     ↓           ↓
p(x_1)  p(x_2|x_1)  p(x_3|x_1,x_2)  ...  p(x_n|x_<n)
```

### 8.2 与 Diffusion 的对比

| 维度 | 自回归 (LLM) | Diffusion |
| :--- | :--- | :--- |
| 数据 | 离散 token | 连续像素/latent |
| 训练目标 | 精确 MLE | ELBO (下界) |
| 采样 | **慢 (n 步,串行)** | 慢 (T 步,可并行) |
| 并行训练 | ✓ (Teacher Forcing) | ✓ (随机 t) |
| 条件生成 | 简单 (前缀) | 加 cross-attention |
| 文本质量 | **极强** ✓ | 较弱 |

### 8.3 案例:为什么 LLM 用自回归不用 Diffusion?

**三个原因**:

1. **离散 token 难加噪**: "加一点点噪声" 在 token 空间没有自然定义。D3PM (2021) 等尝试过,但效果差。

2. **并行训练友好**: Teacher Forcing 让训练可以并行 (整个序列同时算),推理虽然慢,但训练快。Diffusion 在并行训练上更友好 (整个序列同时算)。

3. **严格因果**: 自回归天然保持因果 (不能看到未来),LLM 需要这个性质。Diffusion 的反向过程虽然也"严格",但训练时是双向 attention。

### 8.4 案例:GPT 系列如何"概率生成"?

**GPT-3 的训练目标**:

$$
L = -\sum_{i=1}^{n} \log p_\theta(x_i | x_{<i})
$$

其中 $p_\theta(x_i | x_{<i}) = \text{softmax}(W h_i^{(L)})$, $h_i^{(L)}$ 是 Transformer 第 $L$ 层在位置 $i$ 的隐藏状态。

**采样**: 从 $p(x_1)$,采样 $x_1$;然后从 $p(x_2|x_1)$ 采样 $x_2$;... 直到 $x_n$ 或 <eos>。

**温度采样** (temperature sampling):

$$
p_T(x_i | x_{<i}) = \text{softmax}(h_i / T)
$$

- $T < 1$: 更确定 (更"安全",但可能重复)
- $T > 1$: 更随机 (更有"创造性")
- $T = 1$: 标准采样

**Top-p (nucleus) 采样**: 只从累积概率 ≥ $p$ 的最小 token 集合中采样,避免低概率 token。

```python
# 简化版 GPT 采样
def sample(model, prompt, max_len=100, temperature=1.0, top_p=0.9):
    tokens = tokenize(prompt)
    for _ in range(max_len):
        logits = model(tokens)  # [seq_len, vocab]
        next_logits = logits[-1] / temperature
        probs = F.softmax(next_logits, dim=-1)
        # Top-p filter
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
        tokens = torch.cat([tokens, next_token])
    return detokenize(tokens)
```

### 8.5 最近方向:Masked Diffusion (LLM 的 Diffusion 替代品)

2024 年起,出现了一批尝试把 Diffusion 思想应用到离散文本的工作 (e.g., MDLM, SEDD)。核心思想:

1. **掩码 = 加噪**: 一开始 token 全被掩码 (MASK),通过 Diffusion 逐步"揭示"真实 token
2. **保留因果 (可选)**: 部分方法保持自回归顺序 (e.g., AR-Diffusion)
3. **并行采样**: 不像自回归必须逐步,理论上可以"任意顺序揭示",并行度更高

**当前状态**: 学术前沿,工业尚未大规模采用。GPT 系列仍是 LLM 的事实标准。

**面试亮点**: 如果被问到"未来 LLM 会用 Diffusion 吗?",可以从 (1) 离散加噪的天然性、(2) 并行度、(3) 现有生态的迁移成本 三个角度回答。

---

## 9. 流模型 vs 自回归 vs Diffusion

### 9.1 似然精确度对比

| 模型 | 似然 |
| :--- | :--- |
| 自回归 | **精确** ✓ (链式法则) |
| Flow | **精确** ✓ (变量变换) |
| VAE | **下界** (ELBO) |
| Diffusion | **下界** (ELBO) |
| GAN | **隐式** (无) |

### 9.2 采样质量对比 (FID,越低越好)

| 模型 | MNIST FID | CIFAR-10 FID | ImageNet FID |
| :--- | :--- | :--- | :--- |
| GAN (BigGAN) | - | 14.7 | 7.3 |
| VAE | 30+ | 50+ | - |
| Diffusion (DDPM) | 3.2 | 4.0 | 7.3 |
| Latent Diffusion | - | **2.8** | **3.6** |

**结论**: 在大多数基准上,**Diffusion > GAN > VAE**。

### 9.3 训练稳定性对比

| 模型 | 稳定性 |
| :--- | :--- |
| AR | **最稳定** (MLE, 简单回归) |
| Flow | 稳定 |
| Diffusion | **稳定** (ε-MSE) |
| VAE | 稳定 (但 KL 可能崩塌) |
| GAN | **不稳定** (min-max, mode collapse) |

### 9.4 选择指南

```text
要生成离散文本?  ──→ 自回归 (LLM)
要生成高质量图像? ──→ Latent Diffusion (Stable Diffusion)
要精确似然?       ──→ Flow 或 AR
要轻量/快速?      ──→ VAE 或 GAN (小模型)
要不确定性?       ──→ 加 MC Dropout / Deep Ensemble
不确定?           ──→ 默认 Diffusion + UNet
```

---

## 10. 评估指标

### 10.1 Inception Score (IS)

**定义**:

$$
\text{IS} = \exp\left( \mathbb{E}_{x \sim p_G} \left[ \text{KL}(p(y|x) \| p(y)) \right] \right)
$$

其中 $p(y|x)$ 是 Inception 模型对生成图的类别预测,$p(y)$ 是边缘分布。

**直觉**:
- $p(y|x)$ **越尖锐** (生成图属于某类): 单图置信度高
- $p(y)$ **越均匀** (各类都有): 多样性好

**KL 散度越大 → IS 越高**。

**局限**: 不能检测**模式坍塌** (如果所有图都是"狗", $p(y)$ 集中在"狗", KL 可能仍然大)。**Inception 依赖 ImageNet 训练,不适合非自然图像**。

### 10.2 FID (Fréchet Inception Distance)

**核心思想**: 把生成图和真实图都通过 Inception 模型,得到特征,然后算两个高斯分布的 Fréchet 距离。

**算法**:

1. 用 Inception (倒数第二层) 提取特征: $\phi(x) \in \mathbb{R}^{2048}$
2. 计算 $\mu_g, \Sigma_g$ (生成图特征的均值/协方差),$\mu_r, \Sigma_r$ (真实图)
3. FID = $\|\mu_g - \mu_r\|^2 + \text{Tr}(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2})$

**直觉**: FID 同时考虑**生成质量** (均值接近) 和**多样性** (协方差接近)。

**优势**: 比 IS 鲁棒,能检测模式坍塌,业界标配。

### 10.3 Log-likelihood

对显式密度模型 (VAE、Flow、Diffusion、AR),可以直接算 (或下界) $\log p(x)$。

**优点**: 数学上严格,模型比较公平。

**缺点**:
- 不一定与样本质量正相关 (低似然但好样本)
- FID 0.5 但 Log-likelihood 差 ≠ 模型差

### 10.4 困惑度 (Perplexity, for LLMs)

**定义**:

$$
\text{PPL}(x) = \exp\left(-\frac{1}{n} \sum_{i=1}^{n} \log p(x_i | x_{<i})\right)
$$

**直觉**: "模型平均每个 token 多不惊讶"。PPL = 10 表示模型在每个位置平均等效于 10 个等概率选择。

**GPT-3 175B 的 PPL**: 在标准基准上 ~20,优于 GPT-2 (~35)。

### 10.5 评估指标的 5 个手算例子

**例子 1:二分类 IS (人造例子)**

设 $p(y|x) = \text{Bernoulli}(0.9)$, $p(y) = 0.5$:

$\text{KL}(\text{Bernoulli}(0.9) \| \text{Bernoulli}(0.5)) = 0.9 \log(1.8) + 0.1 \log(0.2) = 0.531 - 0.322 = 0.369$ nats

$\text{IS} = \exp(0.369) = 1.45$

**例子 2:好的 IS**

$p(y|x) = \text{Bernoulli}(0.99)$ (更确定), $p(y) = 0.5$:

$\text{KL} = 0.99 \log(1.98) + 0.01 \log(0.02) = 0.685 - 0.078 = 0.607$ nats

$\text{IS} = \exp(0.607) = 1.83$ (更高)

**例子 3:FID 简化为 1 维高斯**

$\mu_g = 0.5, \sigma_g^2 = 1$, $\mu_r = 0, \sigma_r^2 = 1$:

$\text{FID} = (\mu_g - \mu_r)^2 + (\sigma_g - \sigma_r)^2 = 0.25 + 0 = 0.25$

**例子 4:模式坍塌在 FID 上**

$\mu_g = 0, \sigma_g^2 = 0.01$ (坍塌到一点),$\mu_r = 0, \sigma_r^2 = 1$:

$\text{FID} = 0 + (0.1 - 1)^2 = 0.81$

**FID 大 → 模式坍塌检测成功**。

**例子 5:PPL 简化**

句长 $n = 10$,平均每 token $\log p = -2.3$:

$\text{PPL} = \exp(2.3) \approx 10$ (等价于 10 个均匀选择)

GPT-3 标准的 WikiText PPL ≈ 20.5,即"平均每 token 等效 ~20 个均匀选择"。

### 10.6 案例:为什么 FID 是生成模型的标配?

| 指标 | 是否检测模式坍塌 | 是否依赖外部模型 | 业界使用 |
| :--- | :--- | :--- | :--- |
| IS | 部分 | Inception | 早期主流 |
| FID | ✓ | Inception | **当前主流** ✓ |
| Log-likelihood | ✗ | 无 | 学术 |
| PPL (LLM) | - | 无 | LLM 标配 |

**FID 的成功**:
1. **同时衡量质量与多样性**: FID 高 = 生成差或坍塌
2. **鲁棒**: 不依赖分类 (虽然用 Inception 提取特征)
3. **可比**: 不同模型在相同数据集上 FID 可直接比较
4. **便宜**: 一次前向提取特征,O(n) 算 Fréchet 距离

### 10.7 其他指标 (补充)

**Precision-Recall Distribution (PRD) / Density-Coverage (Kynkäänniemi et al. 2019)**:

弥补 FID 的盲点: FID 是单一数值,无法区分"模式坍塌"和"模式遗漏"。PRD 用两个曲线 (precision vs recall) 评估:

- **Precision**: 生成样本中"真实"的占比
- **Recall**: 真实样本中"被生成覆盖"的占比

```python
# FID 完整计算
def calc_fid(real_features, gen_features):
    mu_r, sigma_r = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = gen_features.mean(0), np.cov(gen_features, rowvar=False)
    diff = mu_r - mu_g
    covmean = linalg.sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
```

**Negative Log-Likelihood (NLL)** for 显式密度模型:

$$
\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i)
$$

单位:**nats/维度** (标准化的 nats per dim)。NLL 越低,模型越好。

**Bits Per Dimension (BPD)** for 图像:

$$
\text{BPD} = \frac{\text{NLL (nats)}}{N \cdot H \cdot W \cdot C \cdot \log 2}
$$

**Score-based 模型的评估**: 除了 FID,还常用 NLL (变分下界)、FLOPS、采样时间。

### 10.8 指标选择的实操建议

```text
你的任务是什么?
│
├── 生成图像
│   ├── 通用评估     → FID (标配)
│   ├── 多样性敏感   → FID + Recall
│   ├── 类条件生成   → Conditional FID + Class-IS
│   └── 真实应用     → FID + 人类评估
│
├── 生成文本
│   ├── 通用语言模型 → PPL (perplexity)
│   ├── 翻译/摘要   → BLEU / ROUGE / BERTScore
│   ├── 对话质量    → 人类评估 + GPT-4 评分
│   └── 多样性      → Distinct-N / Self-BLEU
│
└── 需要不确定性
    ├── 回归       → MC Dropout σ + Calibration
    └── 分类       → ECE + Brier Score
```

**经验法则**: 单一指标有偏,工业项目至少用 **2 个指标 + 1 个人工抽检**。

---

## 11. PyTorch 实验

### 11.1 实验 1:VAE 在 MNIST

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, z=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, z)
        self.fc_lv = nn.Linear(400, z)
        self.dec = nn.Sequential(
            nn.Linear(z, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid())
    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.dec(z), mu, lv

def loss(xr, x, mu, lv):
    rec = F.binary_cross_entropy(xr, x, reduction='sum')
    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum()
    return rec + kl

# 数据
ds = datasets.MNIST('.', download=True,
    transform=transforms.ToTensor())
loader = DataLoader(ds, batch_size=128, shuffle=True)
vae = VAE(); opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

for ep in range(10):
    for x, _ in loader:
        x = x.view(-1, 784)
        xr, mu, lv = vae(x)
        l = loss(xr, x, mu, lv)
        opt.zero_grad(); l.backward(); opt.step()
    print(f"Epoch {ep}: loss = {l.item():.2f}")
```

### 11.2 实验 2:1D 玩具 Diffusion

```python
import torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt

# 目标分布: 双峰高斯混合
def sample_data(n):
    mix = torch.rand(n) < 0.5
    return torch.randn(n) + mix * 4 - 2

# 简单 MLP
net = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1))

T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, 0)

def q_sample(x0, t, eps):
    a = alpha_bars[t]
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps

# 训练
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for step in range(2000):
    x0 = sample_data(128).unsqueeze(1)  # [128, 1]
    t = torch.randint(0, T, (128,))
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, eps)
    t_emb = t.float().unsqueeze(1) / T
    inp = torch.cat([xt, t_emb], dim=1)
    pred = net(inp)
    loss = ((pred - eps)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# 采样
with torch.no_grad():
    x = torch.randn(1000, 1)
    for t in reversed(range(T)):
        t_emb = torch.full((1000, 1), t/T)
        eps_pred = net(torch.cat([x, t_emb], 1))
        a, ab = alphas[t], alpha_bars[t]
        x = (x - (1-a) / (1-ab).sqrt() * eps_pred) / a.sqrt()
        if t > 0:
            x += betas[t].sqrt() * torch.randn_like(x)
    plt.hist(x.numpy(), bins=50); plt.show()
```

### 11.3 实验 3:MC Dropout 回归

```python
import torch, torch.nn as nn
import numpy as np

# 训练数据: y = sin(x) + noise, 训练在 [-3, 3]
x_train = torch.linspace(-3, 3, 50).unsqueeze(1)
y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)

# 测试: 包括训练区间外
x_test = torch.linspace(-6, 6, 200).unsqueeze(1)

# MC Dropout 模型
model = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 1))

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for ep in range(500):
    pred = model(x_train)
    loss = ((pred - y_train)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# MC 预测
model.train()  # 关键:开启 Dropout
preds = torch.stack([model(x_test) for _ in range(100)])
mean, std = preds.mean(0), preds.std(0)

# 可视化:训练区间外 std 大(高不确定性)
import matplotlib.pyplot as plt
plt.fill_between(x_test.squeeze(), (mean-2*std).squeeze(),
                 (mean+2*std).squeeze(), alpha=0.3)
plt.plot(x_test, mean, 'b'); plt.scatter(x_train, y_train, c='r')
plt.show()
```

### 11.4 实验 4:VAE 生成图像可视化

```python
@torch.no_grad()
def visualize_vae(vae, n=10):
    # 从先验采样
    z = torch.randn(n, 20)
    samples = vae.dec(z).view(-1, 28, 28)
    # 重建
    x = next(iter(loader))[0][:n].view(-1, 784)
    recons, _, _ = vae(x)
    recons = recons.view(-1, 28, 28)

    fig, ax = plt.subplots(2, n, figsize=(n, 2))
    for i in range(n):
        ax[0, i].imshow(samples[i], cmap='gray'); ax[0, i].axis('off')
        ax[1, i].imshow(recons[i], cmap='gray'); ax[1, i].axis('off')
    plt.show()
```

### 11.5 实验 5:DDPM MNIST 完整训练 (核心循环)

详见 §5.6。这里给出训练主循环:

```python
ddpm = DDPM(T=1000).to(device)
opt = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
for ep in range(50):
    losses = []
    for x, _ in loader:
        x = x.to(device)
        loss = ddpm.loss(x)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"Epoch {ep}: avg loss = {np.mean(losses):.4f}")
    # 每 5 个 epoch 采样看效果
    if ep % 5 == 0:
        samples = ddpm.sample(16, (1, 28, 28), device)
        save_image(samples, f"ddpm_epoch_{ep}.png", nrow=4)
```

**预期**: 50 epoch 后,生成图像接近 MNIST 真实分布。

### 11.6 实验 6:BNN 分类不确定性

```python
class MCDropoutClassifier(nn.Module):
    def __init__(self, in_dim, n_class, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_class))
    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x, T=50):
        self.train()  # Dropout 开启
        probs = torch.stack([F.softmax(self(x), -1) for _ in range(T)])
        mean = probs.mean(0)
        # 不确定性: 平均预测熵 + 预测分歧
        entropy = -(mean * mean.log()).sum(-1)  # 预测熵
        disagreement = probs.var(0).sum(-1)     # 预测分歧
        return mean, entropy, disagreement

# 在 OOD 数据上 (训练集外),disagreement 应该更大
```

---

## 12. 常见误区 + Q&A

### 12.1 常见误区 (6 条)

**误区 1:"VAE 生成图像模糊" = VAE 不好**

❌ **错误**: VAE 模糊是**架构特性**,不是 bug。
- **真因**: KL 正则太强 + 像素高斯假设 + MSE 损失
- **解决**: β-VAE, 自回归解码器,感知损失
- **何时用 VAE**: 需要**快速采样 + 隐空间语义结构** (插值、解耦),Diffusion 不擅长

**误区 2:"Diffusion 一定比 GAN 好"**

❌ **错误**: 取决于任务。
- **Diffusion 优势**: 训练稳定,样本多样,文本条件好
- **GAN 优势**: 采样快 (1 步),StyleGAN 在人脸上仍 SOTA
- **何时选 GAN**: 极快推理、对质量不极致追求的场景

**误区 3:"ELBO 越大越好"**

❌ **错误**: ELBO 是 $\log p(x)$ 的下界,不是 $\log p(x)$ 本身。
- **真因**: ELBO 高 → $q$ 接近 $p(z|x)$ **且** $p_\theta$ 拟合好
- **但**: 两个模型 ELBO 相同,实际 $p(x)$ 可能不同 (下界松紧度不同)
- **正确做法**: 同时看 ELBO 和样本质量 (FID)

**误区 4:"BNN 一定比普通 NN 准"**

❌ **错误**: BNN 通常**不比普通 NN 准**,它的核心价值是**不确定性**。
- **真因**: 普通 NN 的 MLE 解可能更准;BNN 通过后验平均,有时反而略差
- **正确用法**: 当需要"知道自己不知道"时 (自动驾驶、医疗),用 BNN;纯准确率优先,普通 NN 更简单

**误区 5:"FID 越低越好"**

❌ **错误**: FID 有上限 (数据本身的 FID),且与人类判断不完全相关。
- **真因**: FID 假设高斯分布,实际分布可能多模态
- **盲点**: FID 对小的失真不敏感 (Inception 特征已抽象)
- **正确用法**: FID + 人工评估 + 任务指标 (如分类准确率)

**误区 6:"Diffusion 训练数据多就行"**

❌ **错误**: Diffusion 训练**算力密集**。
- **真因**: 每步都要算 UNet, T=1000 步训练 = 普通模型 1000 倍算力
- **工业现实**: Stable Diffusion 训练要 256 张 A100 跑几周,个人几乎不可能从零训练
- **替代**: 用预训练模型微调,或用 LoRA、ControlNet 等轻量方法

### 12.2 面试 Q&A (5 题 L7/L8 难度)

**Q1:为什么 ELBO 是 $\log p(x)$ 的下界? 完整推导。**

**A**:

$$
\log p(x) = \log \int p(x|z) p(z) dz
$$

引入 $q(z|x)$:

$$
= \log \int q(z|x) \frac{p(x|z) p(z)}{q(z|x)} dz = \log \mathbb{E}_q\left[\frac{p(x|z) p(z)}{q(z|x)}\right]
$$

Jensen 不等式 ($\log$ 凹函数):

$$
\geq \mathbb{E}_q\left[\log \frac{p(x|z) p(z)}{q(z|x)}\right] = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))
$$

即 $\log p(x) \geq \text{ELBO}$。✓

**Q2:为什么 Diffusion 优于 VAE?**

**A**:
1. **训练目标**: VAE 用 ELBO (重建项 + KL),重建项是像素 MSE → 模糊;Diffusion 用 ε-MSE,但**多步去噪**避开了单步"输出均值"问题
2. **架构**: Diffusion 用 UNet,可以深、可加 cross-attention (文本条件);VAE 解码器限制多
3. **似然 vs 质量**: Diffusion 的多步采样等于"分治",逐步精细化;VAE 一步出图
4. **模式覆盖**: Diffusion 的随机反向过程天然多样;VAE 的 KL 强制 $z$ 接近先验,多样性受限
5. **训练稳定**: 没有对抗,梯度直接来自 MSE

**Q3:重参数化技巧为什么重要?**

**A**:

没有重参数化:
- $z \sim q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ 需要采样
- $\mathbb{E}_q[\log p_\theta(x|z)]$ 的梯度 $\nabla_\phi$ 无法用反向传播 (采样操作不可导)
- 只能用 REINFORCE / score function 估计器,方差大

有重参数化:
- $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- $z$ 是 $\mu, \sigma$ 的确定性可微函数
- $\nabla_\phi \mathbb{E}_q = \mathbb{E}_\epsilon[\nabla_\phi \log p_\theta(x | \mu + \sigma\epsilon)]$
- 直接用自动微分,低方差

**Q4:BNN 提供的不确定性有什么用?**

**A**:

**核心价值**: 让模型**知道自己不知道**。

**应用场景**:

1. **自动驾驶**: 预测行人位置 + 不确定性;不确定时降速
2. **医疗诊断**: 预测疾病概率 + 不确定性;高不确定时请求人工复核
3. **强化学习**: 不确定性 → 探索;确定性 → 利用
4. **主动学习**: 不确定性高的样本优先标注
5. **异常检测**: 不确定性突增 → 数据分布漂移

**两类不确定**:
- **认知不确定 (epistemic)**: 模型不知道。**数据多了能消除**。MC Dropout 捕捉。
- **偶然不确定 (aleatoric)**: 数据本身噪声。**数据多了也消不了**。所有方法都能捕捉。

**Q5:为什么 LLM 用自回归不用 Diffusion?**

**A**:

1. **离散 token 难加噪**: 加噪需要连续空间。"把 token '猫' 加 0.1 的噪声" 没有自然定义
2. **并行训练友好**: Teacher Forcing 让训练可以并行 (整个序列同时算)
3. **严格因果**: 自回归天然保持因果,LLM 需要 (不能看到未来)
4. **采样质量**: 离散空间的 Diffusion (D3PM、Diffusion-LM) 效果始终不如自回归
5. **生态成熟**: GPT 系列已用自回归训练几十年,工程优化充分

**反例**: 图像/视频/音频是连续的,Diffusion 主导。**未来可能有统一的"离散+连续 Diffusion" (如 masked diffusion)**。

---

## 13. 与其他数学工具的联系

| 工具 | 在本章哪里用 | 关联章节 |
| :--- | :--- | :--- |
| **变分法** | VAE 的 ELBO 推导本质是变分法;Euler-Lagrange 的离散版是 ELBO 更新 | 1-高数 第 9 章变分法 |
| **贝叶斯定理** | VAE 引入 $q$ 近似后验;BNN 后验 $p(w\|D)$;Diffusion 的 $q(x_{t-1}\|x_t, x_0)$ | 本文件夹第 5 章 |
| **KL 散度** | VAE 的 KL 正则项;Diffusion 反向 KL;GAN 的 JS 散度;MC Dropout 的 KL 项 | 本文件夹第 11 章 + 5-信息论 |
| **MCMC** | BNN 的 MCMC 近似后验;HMC 在大模型上不实用但理论上重要 | 本文件夹第 9 章 |
| **信息瓶颈** | VAE 的互信息上界 $\text{ELBO} \leq I(x; z) + \text{const}$ | 5-信息论 第 7 章信息瓶颈 |
| **假设检验** | 生成模型的 FID 显著性检验;新模型 vs baseline 的 p 值 | 本文件夹第 8 章 |
| **MLE/MAP** | 所有生成模型的训练本质;LLM 的 MLE;Bayesian 视角的 MAP | 本文件夹第 6 章 |
| **EM 算法** | VAE 的训练与 EM 关系;GMM 是 VAE 的特例 (离散隐变量) | 1-高数 第 9 章 EM |
| **ODE / SDE** | 连续时间 Diffusion (Score-based) 用 SDE;DDIM 用 ODE 加速 | 1-高数 第 2 章 |

**关键洞察**:**这一章几乎用到了前面所有概率论和高数的工具**。VAE 用变分法,Diffusion 用贝叶斯 + KL,BNN 用贝叶斯 + MCMC,GAN 用 KL/JS 散度。学习本章时,**强烈建议回到对应章节复习**。

---

## 附录:本章速查表

### 核心公式速查

| 公式 | 名字 | 用途 |
| :--- | :--- | :--- |
| $\text{ELBO} = \mathbb{E}_q[\log p(x\|z)] - \text{KL}(q(z\|x) \| p(z))$ | ELBO | VAE 训练 |
| $z = \mu + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ | 重参数化 | VAE 可微采样 |
| $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | 前向加噪 | Diffusion |
| $L = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ | ε-prediction | DDPM 训练 |
| $p(y\|x, D) \approx \frac{1}{T} \sum_{t=1}^{T} p(y\|x, w_t)$, $w_t$ from Dropout | MC Dropout | BNN 近似 |
| $\text{FID} = \|\mu_g - \mu_r\|^2 + \text{Tr}(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2})$ | FID | 评估 |
| $\text{PPL} = \exp(-\frac{1}{n}\sum_i \log p(x_i\|x_{<i}))$ | 困惑度 | LLM 评估 |

### 5 个 PyTorch 一行调用

```python
# VAE 重参数化
z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
# Diffusion 前向加噪
x_t = torch.sqrt(alpha_bar[t]) * x_0 + torch.sqrt(1 - alpha_bar[t]) * eps
# MC Dropout 预测
mean = torch.stack([model(x) for _ in range(T)]).mean(0)
# FID 简化
fid = ((mu_g - mu_r)**2).sum() + ((sigma_g - sigma_r)**2).sum()
# DDPM 采样一步
x = (x - (1-a)/torch.sqrt(1-ab) * eps_pred) / torch.sqrt(a)
```

### 推荐阅读

1. **VAE 原始论文**: Kingma & Welling, "Auto-Encoding Variational Bayes", 2013
2. **DDPM 原始论文**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
3. **Latent Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
4. **MC Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
5. **GAN**: Goodfellow et al., "Generative Adversarial Networks", NeurIPS 2014
6. **Flow**: Dinh et al., "RealNVP", ICLR 2017
7. **Score-based Diffusion**: Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", NeurIPS 2019
8. **综述**: Yang et al., "Diffusion Models: A Comprehensive Survey of Methods and Applications", 2023

---

> **本章版本**: v1.0
> **最后更新**: 2026-06-23
> **总行数**: ~2100 行
> **配套代码**: 见 §11 PyTorch 实验 (VAE + DDPM + MC Dropout 完整实现)
---

## 附录:补充内容——三大生成模型的工程深挖

> **这一节是给"学完正文想再深一点"的工程师**。目标:把 VAE / Diffusion / Bayesian NN 拆到能直接改 PyTorch 源码的颗粒度,补上正文没展开的 7 个工程关键点。

### A.1 VAE 的 7 个工程坑

#### A.1.1 KL 消失 (KL Vanishing) 与退化解

**现象**:训练到后期,KL 项趋近于 0,$q(z|x) \approx p(z)$。**坏处**: 后验完全塌到先验,decoder 拿不到 $x$ 的信息 (因为 $z$ 不编码 $x$ 了),重建质量暴跌。

**数学上**:ELBO = 重建项 - KL。**当 KL → 0**,decoder 拿到的 $z$ 与 $x$ 无关,只能用 $p(z) = \mathcal{N}(0, I)$ 生成"平均图",重建项拉胯。

**5 个手算例子** (KL 退化的逐步演示):

**例子 1:理想状态** ($\mu = 0, \sigma = 1$)

$\text{KL} = 0.5(\sigma^2 + \mu^2 - 1 - \log \sigma^2) = 0.5(1 + 0 - 1 - 0) = 0$

KL = 0,但 $q = p$ = 标准正态,**理论上是可以的** (但实际上 encoder 会"懒"到直接输出 $\mu = 0, \sigma = 1$,不编码任何 $x$)。

**例子 2:正样本情况** ($\mu = 0.5, \sigma = 0.7$)

$\text{KL} = 0.5(0.49 + 0.25 - 1 - \log 0.49) = 0.5(-0.26 + 0.713) = 0.227$ nats

正常情况。$q$ 偏离先验,encoder 学到东西。

**例子 3:KL 退化的临界** ($\mu = 0, \sigma = 0.95$)

$\text{KL} = 0.5(0.9025 + 0 - 1 - \log 0.9025) = 0.5(-0.0975 + 0.1025) = 0.0025$ nats

KL 接近 0 但 $q \neq p$。**这是"刚要塌"的状态**,需要监控。

**例子 4:完全退化** ($\mu = 0, \sigma = 1$ 但 encoder 不读 $x$)

如果 encoder 的输出层权重全部趋近于 0,$\mu(x) \approx 0, \sigma(x) \approx 1$ 对所有 $x$。KL = 0,但**这是灾难**:$z$ 与 $x$ 无关,重建不出来。

**例子 5:Free Bits 解决**。设阈值 $\lambda$ (e.g., 0.5 nats/dim):

$$
\text{KL}_\text{free} = \max(\text{KL}, \lambda)
$$

KL 低于 $\lambda$ 时不计入损失,防止 encoder 偷懒。

```python
# VAE 训练中的 KL 监控
def kl_per_dim(mu, log_var):
    # 每个 latent 维度的平均 KL
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl.sum(1).mean()  # 标量

# 训练循环里
kl = kl_per_dim(mu, log_var)
if kl.item() < free_bits:  # e.g., free_bits=0.5
    kl = torch.tensor(0.0, device=kl.device)
loss = recon + kl  # 退化的 KL 被"免费放过"
```

#### A.1.2 β-VAE:在 ELBO 里调 KL 的权重

**原始 ELBO**:
$$
\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(x|z)]}_{\text{重建}} - \underbrace{\text{KL}(q(z|x) \| p(z))}_{\text{正则}}
$$

**β-VAE (Higgins et al. 2017)**:
$$
\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta \cdot \text{KL}(q(z|x) \| p(z))
$$

- $\beta < 1$ (e.g., 0.1): 弱正则,$z$ 空间大,生成多样但可能模糊
- $\beta = 1$: 标准 VAE
- $\beta > 1$ (e.g., 4): 强正则,disentanglement (解耦),$z$ 的每个维度学一个独立因子

**Disentanglement 的例子**: 训练在 CelebA (人脸) 上,$\beta = 4$ 时,可能学出 $z_1$ = 性别,$z_2$ = 头发颜色,$z_3$ = 笑容,**每个维度独立可调**。

```python
# β-VAE 损失 (β = 4 是经典)
beta = 4.0
loss = recon + beta * kl

# 监控
print(f"recon: {recon.item():.2f}, kl: {kl.item():.2f}, β·kl: {beta*kl.item():.2f}")
```

#### A.1.3 IWAE (Importance Weighted Autoencoder):更紧的 ELBO

**问题**: 标准 ELBO 是 $\log p(x)$ 的下界,但**很松**(对 $q$ 差时)。**IWAE (Burda et al. 2016)** 用多个样本估计:

$$
\mathcal{L}_{\text{IWAE}} = \mathbb{E}_{z_1, \ldots, z_K \sim q} \left[ \log \frac{1}{K} \sum_{k=1}^{K} \frac{p(x, z_k)}{q(z_k|x)} \right]
$$

**$K = 1$**: 退化为标准 ELBO
**$K \to \infty$**: 收敛到 $\log p(x)$ (紧)

```python
def iwae_loss(x, encoder, decoder, K=5):
    mu, log_var = encoder(x)
    std = (0.5 * log_var).exp()
    # K 个样本
    z = mu[:, None] + std[:, None] * torch.randn(K, *mu.shape)  # [K, B, latent]
    # 评估 log p(x|z) + log p(z) - log q(z|x) = log p(x,z) - log q(z|x)
    # 这里简化:用重建 MSE 代替 log p(x|z),KL 代替 p(z)/q(z|x) 的对数
    log_w = -F.mse_loss(decoder(z), x[:, None], reduction='none').sum([-1, -2]) \
            - 0.5 * (log_var - mu.pow(2) - log_var.exp())  # [K, B]
    return -(log_w.logsumexp(0) - math.log(K)).mean()
```

#### A.1.4 Conditional VAE (CVAE):加条件生成

**任务**: 生成指定标签的样本 (e.g., 生成数字"7")。

**数学**:
$$
p(x | y) = \int p(x | z, y) p(z | y) dz
$$

**ELBO**:
$$
\mathcal{L} = \mathbb{E}_{q(z|x,y)}[\log p(x | z, y)] - \text{KL}(q(z|x,y) \| p(z|y))
$$

```python
class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim):
        super().__init__()
        # 编码器输入: x + y 拼接
        self.enc = nn.Sequential(nn.Linear(x_dim + y_dim, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_lv = nn.Linear(256, latent_dim)
        # 解码器输入: z + y 拼接
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + y_dim, 256), nn.ReLU(),
            nn.Linear(256, x_dim), nn.Sigmoid()
        )
    def forward(self, x, y):
        h = self.enc(torch.cat([x, y], -1))
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.dec(torch.cat([z, y], -1)), mu, lv
```

**应用**: CVAE 是 Stable Diffusion 的"前身"思想 (条件生成 + 隐空间)。

#### A.1.5 VQ-VAE:用离散隐变量代替连续

**核心思想** (van den Oord et al. 2017): $z$ 不是连续向量,而是**离散 codebook 中的索引**。

$$
z_e = \text{Encoder}(x) \in \mathbb{R}^{d \times k}
$$

$$
z_q = \text{argmin}_j \| z_e - e_j \|, \quad e_j \in \mathbb{R}^{k \times d} \text{ (codebook)}
$$

**训练目标**:
$$
\mathcal{L} = \| x - \text{Dec}(z_q) \|^2 + \| \text{sg}[z_e] - e \|^2 + \beta \| z_e - \text{sg}[e] \|^2
$$

第三项是 "commitment loss" (承诺损失),让 encoder 输出靠近 codebook。

**应用**: VQ-VAE + 自回归 (PixelCNN) = **DALL-E 1**。VQ-VAE + Transformer = **DALL-E 2** 的前半段。

```python
# 简化 VQ-VAE 的量化步骤
def vq_quantize(z_e, codebook):
    # z_e: [B, latent_dim], codebook: [K, latent_dim]
    dist = (z_e.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
    idx = dist.argmin(-1)  # [B]
    z_q = codebook[idx]
    return z_q, idx
```

#### A.1.6 Hierarchical VAE:多层级隐空间

**动机**: 单层 $z$ 容量有限。多层级 (NVAE, Very Deep VAE):

$$
p(x) = \int p(x | z_1) p(z_1 | z_2) \cdots p(z_{L-1} | z_L) p(z_L) dz_1 \cdots dz_L
$$

**好处**:
- 每层 $z_l$ 学不同抽象特征 (底层 = 纹理,高层 = 语义)
- 缓解"模糊"问题 (像 NVAE 在 CelebA 上 FID 接近 GAN)

**坏处**: 训练复杂,需要精心设计每层先验。

#### A.1.7 7 个 VAE 工程坑的速查表

| 坑 | 现象 | 解法 |
| :--- | :--- | :--- |
| KL 消失 | KL → 0,重建崩塌 | Free Bits, β < 1, 监控 KL/dim |
| 重建模糊 | 像被"平均"了 | β-VAE, 感知损失, VQ-VAE |
| Posterior collapse | $z$ 完全不编码 $x$ | 弱化 KL, 改用 IWAE |
| 采样不锐利 | 生成的图"软" | 高斯方差可学, 不固定为 1 |
| 模式坍塌 | 只生成几个样本 | β-VAE, VQ-VAE + 强解码器 |
| 训练震荡 | loss 上下抖 | warmup, KL 退火 (见 A.1.8) |
| 评估无似然 | FID 看不出质量提升 | 加 log-likelihood 估计 |

#### A.1.8 KL 退火 (KL Annealing) 的 PyTorch 实现

**问题**: 一开始 KL 太强,encoder 学不动;后期 KL 应该收紧。

**解法**: KL 系数从 0 慢慢升到 1:

```python
def kl_anneal_weight(epoch, anneal_epochs=10, mode='linear'):
    if mode == 'linear':
        return min(1.0, epoch / anneal_epochs)
    elif mode == 'sigmoid':
        return float(1 / (1 + np.exp(-(epoch - anneal_epochs/2) * 0.5)))
    elif mode == 'cyclical':  # Bowman et al. 2016
        # 周期性增加 KL 权重
        cycle_len = anneal_epochs // 4
        return min(1.0, (epoch % cycle_len) / cycle_len)

# 训练
for epoch in range(100):
    beta = kl_anneal_weight(epoch, anneal_epochs=20, mode='sigmoid')
    for x in dataloader:
        recon, mu, lv = model(x)
        loss = recon_loss + beta * kl_loss
        loss.backward()
```

---

### A.2 Diffusion 的 7 个工程深挖

#### A.2.1 噪声调度的 5 种选择

**Linear** (DDPM 原版):
$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1), \quad \beta_1 = 10^{-4}, \beta_T = 0.02
$$

**Cosine** (Nichol & Dhariwal 2021):
$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

$s = 0.008$。**优势**: 在前几步保留更多信号,后几步的衰减更平滑。

**Quadratic / Sigmoid / Warmup-Stable-Decay (WSD)**:
工程上常用根据数据集调优。

**5 个手算对比例子**:

**例子 1:Linear at t=500 (T=1000)**

$\beta_{500} = 10^{-4} + (500/999) \cdot (0.02 - 10^{-4}) \approx 0.010$

$\bar{\alpha}_{500} \approx \prod_{i=1}^{500} (1 - \beta_i) \approx e^{-\sum \beta} \approx e^{-0.01 \cdot 500} = e^{-5} \approx 0.0067$

$x_{500}$ 的信噪比 $\bar{\alpha}/(1-\bar{\alpha}) \approx 0.0068$,几乎纯噪声。

**例子 2:Cosine at t=500**

$\bar{\alpha}_{500} = \cos^2\left(\frac{0.5 + 0.008}{1.008} \cdot \frac{\pi}{2}\right) / \cos^2(0.008 \cdot \pi/2)$

$\approx \cos^2(0.789) / \cos^2(0.0126) \approx 0.479 / 0.999 \approx 0.479$

**信噪比 = 0.479 / 0.521 ≈ 0.92**,**保留近一半信号**。

**例子 3:为什么 Cosine 在低 t 更好**

t=10 时:

- Linear: $\bar{\alpha}_{10} \approx 1 - 10 \cdot 0.0001 = 0.999$,$\beta \approx 0.0001$ (几乎不噪)
- Cosine: $\bar{\alpha}_{10} \approx \cos^2(\text{small angle}) \approx 1 - 0.0002$,**更多噪** (更好,模型能学到低噪去噪)

**例子 4:SNR 的物理意义**

$\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$

- SNR 大 → 信号主导,去噪简单
- SNR 小 → 噪声主导,去噪困难 (类似从纯噪声中恢复)

**例子 5:WSD 调度的工程优势**

Warmup-Stable-Decay:前 10% warmup,中间 70% 稳定,最后 20% 衰减。**FID 比 Linear 低 20-30%** (工程经验)。

```python
def get_schedule(T, schedule_type='cosine'):
    if schedule_type == 'linear':
        return torch.linspace(1e-4, 0.02, T)
    elif schedule_type == 'cosine':
        s = 0.008
        t = torch.arange(T) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        return 1 - f / f[0]  # 转 beta
```

#### A.2.2 采样加速:从 DDPM 1000 步到 DDIM 50 步再到 Consistency 1 步

**DDIM (Song et al. 2020)** 的关键洞察: 前向过程**不需要**是马尔可夫的。

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon
$$

- $\sigma_t = 0$:**确定性 DDIM** (同样噪声输入 → 同样输出,100 步能稳定生成)
- $\sigma_t > 0$: 还原到 DDPM

**DPM-Solver (Lu et al. 2022)**: 把 DDIM 看作 ODE,用更高级的数值方法 (e.g., Runge-Kutta),**15-20 步**就能达到 1000 步 DDPM 的质量。

**Consistency Model (Song et al. 2023)**: 训练一个**单步映射** $f: x_t \to x_0$,1 步采样。

**5 个手算对比**:

**例子 1:DDPM 1000 步, FID 4.0**

**例子 2:DDIM 50 步, FID 4.5** (略差但快 20 倍)

**例子 3:DDIM 10 步, FID 13** (退化)

**例子 4:DPM-Solver 20 步, FID 4.2** (近 DDPM)

**例子 5:Consistency 1 步, FID 6-10** (1 步但质量略降,工程可用)

```python
# DDIM 50 步采样 (简化)
@torch.no_grad()
def ddim_sample(model, x_T, alpha_bars, n_steps=50):
    # 把 1000 步映射到 50 步
    step_indices = torch.linspace(0, len(alpha_bars)-1, n_steps+1).long()
    x = x_T
    for i in reversed(range(n_steps)):
        t = step_indices[i]
        t_prev = step_indices[i-1] if i > 0 else -1
        a_t, a_prev = alpha_bars[t], (alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0))
        eps = model(x, t.expand(x.shape[0]))
        x0_pred = (x - (1-a_t).sqrt() * eps) / a_t.sqrt()
        x = a_prev.sqrt() * x0_pred + (1-a_prev).sqrt() * eps  # σ=0
    return x
```

#### A.2.3 Classifier-Free Guidance:无分类器的条件生成

**传统 (Classifier Guidance)**: 用一个分类器 $p(y|x_t)$ 引导反向过程。

**Classifier-Free Guidance (Ho & Salimans 2022)**: 不用分类器,让 UNet 同时学**有条件**和**无条件**两个预测:

$$
\tilde{\epsilon}_\theta(x_t, t, y) = (1 + w) \epsilon_\theta(x_t, t, y) - w \epsilon_\theta(x_t, t, \emptyset)
$$

- $w = 0$: 标准条件生成
- $w > 0$ (e.g., 7.5): 增强条件,质量提升但多样性下降
- $w$ 是 Stable Diffusion 里的 "Guidance Scale"

```python
# Classifier-Free Guidance 推理
def cfg_sample(unet, x_t, t, y, guidance_scale=7.5):
    eps_cond = unet(x_t, t, y)  # 条件
    eps_uncond = unet(x_t, t, None)  # 无条件
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

#### A.2.4 Latent Diffusion 的 VAE 细节

**Stable Diffusion 的 VAE** (来自 CompVis):
- 编码器: 256×256×3 → 32×32×4
- 解码器: 32×32×4 → 256×256×3
- 用 **KL-regularized** VAE 训练 (不是 VQ-VAE)

**为什么 latent 维度是 4?**

太小 (1-2): 信息损失大,重建模糊。
太大 (16+): 后续 UNet 计算量爆炸。
**4 = 工程甜点** (经验值)。

**5 个手算例子** (latent 维度选择):

| latent_dim | 压缩比 | 重建 FID | UNet 速度 |
| :--- | :--- | :--- | :--- |
| 4 | 48× | 0.5 | 1× |
| 8 | 24× | 0.3 | 4× |
| 16 | 12× | 0.2 | 16× |
| 32 | 6× | 0.1 | 64× |
| 1 | 192× | 10+ | 0.3× |

**例子 5 的工程结论**: latent_dim = 4 平衡质量与速度。

#### A.2.5 连续时间 Diffusion:Score-based SDE

**宋旸 (Yang Song) 2021** 统一了 DDPM 和 Score-based:

**前向 SDE**:
$$
dx = f(x, t) dt + g(t) dw
$$

**反向 SDE**:
$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) dw
$$

**关键洞察**: 只要学了 score $\nabla_x \log p_t(x)$,就能从 $p_T$ 反向生成 $p_0$。

**Probability Flow ODE**:
$$
dx = [f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)] dt
$$

ODE 的好处:可以用 ODE solver 自适应步数,工程上 DDIM / DPM-Solver 都基于此。

**5 个手算例子**:

**例子 1:VP-SDE (Variance Preserving)**

$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$

对应 DDPM。

**例子 2:VE-SDE (Variance Exploding)**

$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dw$

对应 NCSN (Noise Conditional Score Network)。

**例子 3:sub-VP-SDE**

方差增长比 VE 慢,介于 VP 和 VE 之间。

**例子 4:score 与 ε 的关系**

$\nabla_x \log p_t(x) = -\frac{\epsilon_\theta(x, t)}{\sigma_t}$,其中 $\sigma_t = \sqrt{1-\bar{\alpha}_t}$。

**例子 5:为什么 SDE 重要**

- **数学优美**: 把 DDPM / Score / CNF 统一
- **ODE 加速**: 概率流 ODE 让自适应步数成为可能
- **Likelihood 可算**: CNF 给出精确似然

#### A.2.6 DALL-E 2 / Imagen 的层级 Diffusion

**DALL-E 2 (Ramesh et al. 2022)** 的"两阶段":

1. **Prior**: 把 CLIP 文本嵌入 $c$ 映射到图像隐空间 $z$ (用 Diffusion)
2. **Decoder**: 从 $z$ 生成图像 (用 Latent Diffusion)

**Imagen (Saharia et al. 2022)**: 类似,但**直接在像素空间**用级联 Diffusion (64×64 → 256×256 → 1024×1024)。

**Stable Diffusion**: 与 DALL-E 2 类似,但 Prior 是**单步** (CLIP 嵌入 → latent 不用 Diffusion)。

**5 个对比例子**:

| 模型 | 文本编码 | 图像生成 | 分辨率 |
| :--- | :--- | :--- | :--- |
| DALL-E 2 | CLIP | Diffusion Prior + Decoder | 1024×1024 |
| Imagen | T5-XXL | 级联 Diffusion | 1024×1024 |
| Stable Diffusion | CLIP ViT-L/14 | Latent Diffusion | 512×512 |
| Midjourney | 自研 | Diffusion | 1024+ |
| Parti | 自研 | 自回归 ViT-VQGAN | 256×256 |

#### A.2.7 Diffusion 在 NLP / RL / Time Series 的扩展

**Diffusion-LM (Li et al. 2022)**: 把 token embedding 当作连续,加噪-去噪,生成离散文本 (效果不如自回归)。

**Diffusion RL (Diffusion Policy, Chi et al. 2023)**: 用 Diffusion 建模动作分布,在机器人领域 SOTA。

**TimeGrad (Rasul et al. 2021)**: 时间序列的 Diffusion,用于预测。

**5 个手算例子** (跨域对比):

**例子 1:Diffusion-LM 训练**

$x_0$ = token embedding (e.g., 词向量),$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。**问题**: token embedding 加噪后生成的 $x_0$ 可能是"无效 token"。需要 **rounding** 步骤。

**例子 2:Diffusion Policy**

在机器人领域,动作 $a \in \mathbb{R}^{\text{horizon} \times \text{action_dim}}$,用 Diffusion 建模 $p(a|s)$,**比 Behavior Cloning 更稳**。

**例子 3:TimeGrad**

时间序列 $x_{1:T}$,逐步加噪到 $x_T \sim \mathcal{N}(0, I)$,**反向过程**用 RNN + Attention 处理时序依赖。

**例子 4:Video Diffusion**

把视频当作 (T, H, W, C) 的 4D tensor,在 3D UNet 上做 Diffusion。**计算量爆炸**,需要时空分离 attention。

**例子 5:Audio Diffusion (AudioLDM)**

把 mel-spectrogram 当 latent,在其上做 Diffusion。**优势**: 不用 raw waveform 的 1D Diffusion,显存友好。

---

### A.3 Bayesian NN 的 7 个工程深挖

#### A.3.1 5 种 BNN 近似方法速查

| 方法 | 后验近似 | 训练 | 推理 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MC Dropout** | 各向异性高斯 | 标准 + Dropout | T 次前向 | 简单 | 理论不严 |
| **Deep Ensemble** | 离散混合 | 训 K 个模型 | K 次前向 | 鲁棒 | K 倍成本 |
| **VI (BBB)** | 因子化高斯 | 优化 ELBO | T 次采样 | 端到端 | 局部最优 |
| **Laplace** | 高斯 | 训 + Hessian | 闭式 | 简单后处理 | 局部 |
| **SGLD** | 经验分布 | MCMC 采样 | S 次平均 | 渐近精确 | 慢 |

**5 个手算对比**:

**例子 1:MC Dropout 训练成本**

$\text{训练} = 1 \times \text{普通 NN}$,$\text{推理} = T \times \text{普通 NN}$。

$T = 50$ 时,**推理慢 50 倍**。

**例子 2:Deep Ensemble 训练成本**

$\text{训练} = K \times \text{普通 NN}$,$\text{推理} = K \times \text{普通 NN}$。

$K = 5$ 时,**训练和推理都慢 5 倍**,但比 MC Dropout 准 (理论保证)。

**例子 3:BBB (Bayes by Backprop) ELBO**

$\mathcal{L} = \mathbb{E}_{q(w)}[\log p(D|w)] - \text{KL}(q(w) \| p(w))$

$q(w) = \prod_i \mathcal{N}(w_i; \mu_i, \sigma_i^2)$。**学 $\mu, \sigma$ 的参数量 = 普通 NN 的 2 倍**。

**例子 4:Laplace 精度**

$\text{精度} = \text{Hessian} \in \mathbb{R}^{P \times P}$,$P$ = 参数量。

ResNet-50 ($P \approx 25M$): Hessian 矩阵 $25M \times 25M$ = 5 PB。**实际用对角近似或 Kronecker 分解 (K-FAC)**。

**例子 5:SGLD (Stochastic Gradient Langevin Dynamics)**

更新规则:
$$
w_{t+1} = w_t - \frac{\eta}{2}(\nabla L(w_t) + \lambda w_t) + \sqrt{\eta} \epsilon_t
$$

后两项分别对应 Langevin 动力学的"势能梯度"和"随机扰动"。**理论上**渐近从后验采样。

#### A.3.2 Deep Ensemble 的工程实现

**最简单也最有效的 BNN 近似** (Lakshminarayanan et al. 2017)。

```python
class Ensemble(nn.Module):
    def __init__(self, base_model_cls, n_models=5, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([base_model_cls(**kwargs) for _ in range(n_models)])
    def forward(self, x):
        return torch.stack([m(x) for m in self.models])  # [K, B, ...]
    def predict_with_uncertainty(self, x):
        preds = self.forward(x)
        return preds.mean(0), preds.std(0)  # 均值 + 不确定性

# 训练: 训 K 个独立模型 (可以用不同随机种子 + 不同数据子集 bootstrap)
ensemble = Ensemble(MyModel, n_models=5, in_dim=1)
for model in ensemble.models:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 训...
```

**为什么有效?**
- 每个模型收敛到不同的局部最优
- 集成预测 = 用"投票"压低噪声
- 经验上**比 MC Dropout 更准** (Deep Ensemble 通常是 BNN 论文的 baseline)

**5 个手算例子**:

**例子 1:2 个模型集成,分歧大**

$\text{model}_1: \hat{y} = 0.5$, $\text{model}_2: \hat{y} = 1.5$, $\text{mean} = 1.0, \text{std} = 0.5$

不确定性高,可能 OOD。

**例子 2:2 个模型集成,分歧小**

$\text{model}_1: \hat{y} = 1.0, \text{model}_2: \hat{y} = 1.05$, $\text{mean} = 1.025, \text{std} = 0.025$

确定性高,模型"自信"。

**例子 3:5 个模型,3 个相同 + 2 个不同**

体现"投票效应"。

**例子 4:训练成本**

5 模型 × 1 小时 = 5 小时。**5 倍成本**是工程上最大的缺点。

**例子 5:多样性 vs 准确性**

训练时用**不同数据子集 (bagging)** 或**不同损失函数**可以增加多样性,但降低单模型精度。**sweet spot: 5-10 个模型**。

#### A.3.3 Temperature Scaling 与 Calibration

**问题**: 现代 NN 经常**过自信** (over-confident)。预测 $p(y=\text{猫}) = 0.99$ 但实际只有 80% 准确。

**Temperature Scaling (Guo et al. 2017)**: 在 softmax 前除以温度 $T$:

$$
p_T(y = k | x) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}
$$

- $T = 1$: 标准
- $T > 1$: 更"软" (降低置信度)
- $T < 1$: 更"硬" (提高置信度)

```python
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.T

# 训练: 最小化 NLL
scaler = TemperatureScaler()
opt = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=50)
def closure():
    opt.zero_grad()
    loss = F.cross_entropy(scaler(val_logits), val_labels)
    loss.backward()
    return loss
opt.step(closure)
```

**5 个手算例子**:

**例子 1:无 TS 的过自信**

logits = [2, 1, 0] → softmax = [0.66, 0.24, 0.09]
真实准确率: 0.5 → **calibration error = 0.16**

**例子 2:TS T=2**

logits/T = [1, 0.5, 0] → softmax = [0.41, 0.27, 0.19]
真实准确率: 0.5 → **calibration error = 0.09** (更接近)

**例子 3:TS T=0.5**

logits/T = [4, 2, 0] → softmax = [0.84, 0.11, 0.04]
更过自信。

**例子 4:Expected Calibration Error (ECE)**

$\text{ECE} = \sum_b \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$

分桶 (e.g., 10 桶),每桶算平均置信度 vs 准确率,加权和。

**例子 5:为什么 calibration 重要**

医疗诊断:模型说"99% 是癌症"但实际只有 60%。**calibrated 模型**让医生更信任概率。

#### A.3.4 Evidential Deep Learning:单次前向的不确定性

**核心思想 (Sensoy et al. 2018)**: 不用多次前向,直接让 NN 输出**Dirichlet 分布参数** $\alpha$。

**预测**:
$$
p(y = k | x) = \frac{\alpha_k}{\sum_j \alpha_j}
$$

**不确定性**:
$$
\underbrace{\frac{K}{\sum_j \alpha_j}}_{\text{Dempster-Shafer 不确定}} = \text{epistemic uncertainty}
$$

```python
class EvidentialNN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_classes)  # 输出 log α
    def forward(self, x):
        log_alpha = self.head(self.body(x))  # [B, K]
        alpha = log_alpha.exp() + 1  # 保证 α > 0
        return alpha  # Dirichlet 参数
    def loss(self, alpha, y):
        # Dirichlet-Multinomial NLL
        S = alpha.sum(-1, keepdim=True)
        loss = (torch.lgamma(S) - torch.lgamma(alpha.sum(-1)) -
                (torch.lgamma(torch.ones_like(alpha)) -
                 torch.lgamma(alpha.gather(-1, y.unsqueeze(-1))))).mean()
        return loss
    def predict(self, alpha):
        return alpha / alpha.sum(-1, keepdim=True)  # 期望概率
    def uncertainty(self, alpha):
        K = alpha.shape[-1]
        return K / alpha.sum(-1)  # 越高越不确定
```

**5 个手算例子**:

**例子 1:in-distribution 输入**

$\alpha = [10, 1, 1]$, 预测 = [0.83, 0.08, 0.08], 确信度 = 12/12 = 1.0

**例子 2:out-of-distribution 输入**

$\alpha = [0.5, 0.5, 0.5]$, 预测 = [0.33, 0.33, 0.33], 确信度 = 3/1.5 = 2.0 (高不确定)

**例子 3:训练损失**

$\mathcal{L} = \mathbb{E}[\text{Dir-Mult NLL}] + \lambda \cdot \text{KL}(\text{Dir}(\alpha) \| \text{Dir}(1))$

**例子 4:为什么 1 次前向就够**

不需要 MC Dropout 跑 T 次。**训练 1 次,推理 1 次**,速度 = 普通 NN。

**例子 5:工业部署**

Tesla, Bosch 等公司使用 Evidential NN 用于目标检测的不确定性。

#### A.3.5 主动学习与 BNN 的交集

**主动学习 (Active Learning)**: 标注预算有限,**选最有价值的样本标注**。

**BNN 的角色**: 不确定性高的样本 → 最该被标注。

**5 个手算例子** (基于 BNN 的主动学习策略):

**例子 1:Uncertainty Sampling**

选 BNN 预测不确定性最大的 $N$ 个样本标注。

$I(x) = \mathbb{H}(p(y|x, D)) = -\sum_k p(y=k|x,D) \log p(y=k|x,D)$

**例子 2:BALD (Bayesian Active Learning by Disagreement)**

$I(x) = \mathbb{H}(p(y|x, D)) - \mathbb{E}_{p(w|D)}[\mathbb{H}(p(y|x, w))]$

**意思**: 总不确定性 - 模型平均认知的不确定性 = **信息增益**。

**例子 3:Ensemble-based Query**

$I(x) = \text{Var}(\{f_w(x)\}_{w \sim p(w|D)})$

**例子 4:批量选择 (BatchBALD)**

选**互信息最大**的 $B$ 个样本 (避免都选相似的不确定样本)。

**例子 5:实际效果**

CIFAR-10 上,BNN 主动学习可以用 **20% 标注**达到 **90% 准确率** vs 随机采样需要 60% 标注。

#### A.3.6 BNN 在 OOD 检测的应用

**OOD (Out-of-Distribution) 检测**: 训练集是猫狗,测试时来了汽车 → 应该标"不知道"。

**5 个 BNN-OOD 方法**:

**方法 1:Mahalanobis Distance (Lee et al. 2018)**

用预训练特征的均值/协方差算 Mahalanobis 距离:

$I(x) = \min_k (f(x) - \mu_k)^T \Sigma^{-1} (f(x) - \mu_k)$

**方法 2:Ensemble Disagreement**

$5$ 个模型预测不一致 → OOD。

**方法 3:MC Dropout Variance**

$I(x) = \text{Var}(\{f_w(x)\}_{w \sim \text{Dropout}})$

**方法 4:Evidential Uncertainty**

$I(x) = K / \sum \alpha$ (见 A.3.4)。

**方法 5:Energy Score (Liu et al. 2020)**

$I(x) = -T \log \sum_k \exp(f_k(x) / T)$

```python
# Energy-based OOD detection
def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=-1)

# OOD = 高 energy (低的"证据")
ood_score = energy_score(model(x_test))
ood_mask = ood_score > threshold
```

**5 个手算例子**:

**例子 1:In-distribution 猫**

logits = [3, 0, -2] → softmax = [0.95, 0.04, 0.01]
energy = -1 · log(0.95 + 0.04 + 0.01) = 0.05 (低, ID)

**例子 2:Out-of-distribution 飞机**

logits = [0.5, 0.4, 0.3] → softmax = [0.39, 0.36, 0.30]
energy = -1 · log(1.05) ≈ -0.05 (略高,OOD 倾向)

**例子 3:AUROC 评估**

在 ID/OOD 数据集上跑 OOD detector,画 ROC 算 AUROC。**好的 OOD detector AUROC > 0.95**。

**例子 4:为什么 BNN 适合 OOD**

BNN 给"自己不知道"的能力 → 自然区分 ID vs OOD。普通 NN 给"过度自信"。

**例子 5:工业部署**

Waymo 的目标检测模块有 OOD 检测 (路上的异常物体,比如突然出现的动物)。

#### A.3.7 BNN 的 PyTorch 完整实现 (回归 + 分类)

**回归版**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutBNN(nn.Module):
    """MC Dropout BNN for regression"""
    def __init__(self, in_dim, hidden=128, p=0.2, T=50):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p),  # MC Dropout
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x, return_std=True):
        """MC 采样 T 次"""
        self.train()  # 关键: 保持 Dropout
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(self.T)])
        mean = preds.mean(0)
        std = preds.std(0) if return_std else None
        return mean, std


# 训练: 用 NLL (假设高斯似然)
def gaussian_nll_loss(pred_mean, pred_log_var, y):
    """-log p(y|x) = 0.5 log(2π σ²) + 0.5 (y-μ)²/σ²"""
    return 0.5 * pred_log_var + 0.5 * (y - pred_mean).pow(2) * (-pred_log_var).exp()

class HeteroscedasticBNN(nn.Module):
    """预测均值 + 对数方差 (异方差 BNN)"""
    def __init__(self, in_dim, hidden=128, p=0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.body(x)
        return self.mu_head(h), self.logvar_head(h)
    def predict(self, x, T=50):
        self.train()
        with torch.no_grad():
            mus, lvs = [], []
            for _ in range(T):
                mu, lv = self(x)
                mus.append(mu)
                lvs.append(lv)
            mus = torch.stack(mus)
            lvs = torch.stack(lvs)
        # 总不确定性 = epistemic + aleatoric
        epistemic = mus.var(0)
        aleatoric = lvs.exp().mean(0)
        total = epistemic + aleatoric
        return mus.mean(0), total, epistemic, aleatoric
```

**分类版**:

```python
class MCDropoutClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128, p=0.3, T=50):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)
    def predict_with_uncertainty(self, x):
        """返回预测概率 + 预测熵 + mutual information"""
        self.train()
        with torch.no_grad():
            probs = F.softmax(torch.stack([self(x) for _ in range(self.T)]), -1)
        # 预测期望
        mean_probs = probs.mean(0)  # [B, K]
        # 预测熵 (总不确定)
        pred_entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(-1)
        # 互信息 (认知不确定)
        entropy_per_sample = -(probs * (probs + 1e-10).log()).sum(-1).mean(0)
        mi = pred_entropy - entropy_per_sample
        return mean_probs, pred_entropy, mi

# 用法
model = MCDropoutClassifier(in_dim=10, n_classes=3)
probs, entropy, mi = model.predict_with_uncertainty(x_test)
# entropy 高 → 总不确定
# mi 高 → 认知不确定 (多采样能消除)
```

**5 个手算例子** (分类 BNN):

**例子 1:高 confidence ID 样本**

3 次采样: [0.9, 0.05, 0.05], [0.88, 0.06, 0.06], [0.91, 0.04, 0.05]
mean = [0.90, 0.05, 0.05], entropy ≈ 0.33 nats, MI ≈ 0.005 (低,ID)

**例子 2:认知不确定 (model 不熟)**

3 次采样: [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]
mean = [0.4, 0.3, 0.3], entropy ≈ 1.09, MI ≈ 0.04 (中,模型分歧)

**例子 3:偶然不确定 (数据噪声)**

3 次采样: [0.6, 0.2, 0.2], [0.62, 0.21, 0.17], [0.58, 0.22, 0.20]
mean ≈ [0.6, 0.21, 0.19], MI ≈ 0.001 (低), entropy ≈ 0.94 (中,数据本身难)

**例子 4:同时高**

3 次采样: [0.4, 0.4, 0.2], [0.3, 0.3, 0.4], [0.5, 0.2, 0.3]
MI 高 + entropy 高 = 严重 OOD

**例子 5:OOD 检测**

如果 MI > 阈值 → flag 为 OOD,不要预测。

---

### A.4 三大模型的 PyTorch 完整可跑示例

#### A.4.1 完整可跑:Conditional VAE 在 MNIST 上生成指定数字

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CVAE_MNIST(nn.Module):
    def __init__(self, latent_dim=20, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        # 编码器: x + one-hot(y) → (μ, log σ²)
        self.enc = nn.Sequential(
            nn.Linear(784 + n_classes, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_lv = nn.Linear(400, latent_dim)
        # 解码器: z + one-hot(y) → x
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 400), nn.ReLU(),
            nn.Linear(400, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid(),
        )

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], -1))
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, log_var):
        return mu + (0.5 * log_var).exp() * torch.randn_like(mu)

    def decode(self, z, y):
        return self.dec(torch.cat([z, y], -1))

    def forward(self, x, y):
        mu, lv = self.encode(x, y)
        z = self.reparameterize(mu, lv)
        return self.decode(z, y), mu, lv

def train_cvae():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    model = CVAE_MNIST().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for x, y in dl:
            x = x.view(-1, 784).to(device)
            y_oh = F.one_hot(y, 10).float().to(device)
            x_recon, mu, lv = model(x, y_oh)
            # 重建项: BCE (sigmoid 输出)
            recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
            # KL
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum()
            loss = recon + kl
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss / len(ds):.2f}")
    return model

def sample_digit(model, digit=7, n=8):
    """生成指定数字"""
    device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        y = torch.full((n,), digit, device=device)
        y_oh = F.one_hot(y, 10).float()
        x_recon = model.decode(z, y_oh)
    return x_recon.view(-1, 1, 28, 28).cpu()

# 训练
model = train_cvae()
# 生成数字 7
samples_7 = sample_digit(model, digit=7, n=8)
```

#### A.4.2 完整可跑:Classifier-Free Guided Diffusion on MNIST

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half) * (math.log(10000) / (half - 1)))
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class LabelEmb(nn.Module):
    def __init__(self, n_classes, dim):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, dim)  # +1 for "unconditional"
    def forward(self, y):
        return self.emb(y)

class CFG_UNet(nn.Module):
    """简化的 UNet,支持时间步 + 标签条件"""
    def __init__(self, n_classes=10, t_dim=128, y_dim=128, ch=64):
        super().__init__()
        self.t_emb = nn.Sequential(TimeEmb(t_dim), nn.Linear(t_dim, t_dim), nn.ReLU())
        self.y_emb = nn.Sequential(LabelEmb(n_classes, y_dim), nn.Linear(y_dim, y_dim), nn.ReLU())
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )
        self.t_proj = nn.Linear(t_dim, ch)
        self.y_proj = nn.Linear(y_dim, ch)
    def forward(self, x, t, y):
        # y = n_classes 表示无条件
        h = self.net[0](x)
        t_h = self.t_proj(self.t_emb(t))[:, :, None, None]
        y_h = self.y_proj(self.y_emb(y))[:, :, None, None]
        h = F.relu(self.net[1](h) + t_h + y_h)
        h = F.relu(self.net[2](h))
        h = F.relu(self.net[3](h))
        return self.net[4](h)

class CFG_DDPM(nn.Module):
    def __init__(self, T=500, n_classes=10):
        super().__init__()
        self.T = T
        self.n_classes = n_classes
        # 余弦调度
        s = 0.008
        t_arr = torch.arange(T) / T
        f = torch.cos((t_arr + s) / (1 + s) * math.pi / 2) ** 2
        self.alpha_bars = (f / f[0]).float()
        self.betas = (1 - self.alpha_bars / torch.cat([torch.ones(1), self.alpha_bars[:-1]])).clamp(0, 0.999)
        self.alphas = 1 - self.betas
        self.model = CFG_UNet(n_classes=n_classes)

    def q_sample(self, x0, t, noise):
        a = self.alpha_bars[t][:, None, None, None].to(x0.device)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, x0, y):
        """10% 用无条件,90% 用条件"""
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        # 10% 概率丢标签
        uncond_mask = torch.rand(b, device=x0.device) < 0.1
        y_in = y.clone()
        y_in[uncond_mask] = self.n_classes
        eps_pred = self.model(x_t, t, y_in)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample_cfg(self, n, y, img_shape, device, guidance_scale=3.0):
        """CFG 采样"""
        x = torch.randn(n, *img_shape, device=device)
        y_cond = y
        y_uncond = torch.full_like(y, self.n_classes)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps_cond = self.model(x, t_batch, y_cond)
            eps_uncond = self.model(x, t_batch, y_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            a = self.alphas[t].to(device)
            ab = self.alpha_bars[t].to(device)
            x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                x += self.betas[t].sqrt() * torch.randn_like(x)
        return x

def train_cfg_ddpm():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    model = CFG_DDPM(T=500).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    for epoch in range(5):
        for x, y in dl:
            x = x.to(device); y = y.to(device)
            loss = model.loss(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
    return model

# 训练
model = train_cfg_ddpm()
# 生成数字 3 with guidance
samples = model.sample_cfg(n=8, y=torch.tensor([3]*8), img_shape=(1,28,28),
                            device=next(model.parameters()).device, guidance_scale=3.0)
```

#### A.4.3 完整可跑:MC Dropout BNN 做回归 + 不确定性可视化

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def make_data(n=200, noise=0.1):
    """y = sin(2x) + noise * sin(5x)"""
    np.random.seed(42)
    x = np.random.uniform(-3, 3, n)
    y = np.sin(2 * x) + noise * np.random.randn(n) * np.abs(np.sin(x))
    return torch.tensor(x, dtype=torch.float32)[:, None], torch.tensor(y, dtype=torch.float32)[:, None]

class MC_Dropout_MLP(nn.Module):
    def __init__(self, hidden=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)
    def predict(self, x, T=100):
        self.train()  # 关键: 开启 Dropout
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(T)])
        return preds.mean(0), preds.std(0)

def train_bnn_regression():
    device = 'cpu'
    x_train, y_train = make_data(n=300)
    x_train, y_train = x_train.to(device), y_train.to(device)
    model = MC_Dropout_MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(2000):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    # 推理
    x_test = torch.linspace(-4, 4, 200)[:, None].to(device)
    mean, std = model.predict(x_test, T=100)
    return x_train.cpu().numpy(), y_train.cpu().numpy(), x_test.cpu().numpy(), mean.cpu().numpy(), std.cpu().numpy()

# 训练 + 可视化
x_tr, y_tr, x_te, mean, std = train_bnn_regression()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(x_tr, y_tr, alpha=0.3, label='train')
plt.plot(x_te, np.sin(2 * x_te), 'g--', label='true')
plt.plot(x_te, mean, 'b-', label='BNN mean')
plt.fill_between(x_te.ravel(), (mean - 2*std).ravel(), (mean + 2*std).ravel(),
                  alpha=0.3, color='blue', label='±2σ')
plt.legend()
plt.title('MC Dropout BNN:回归 + 不确定性')
plt.subplot(1, 2, 2)
plt.plot(x_te, std, 'r-')
plt.fill_between(x_te.ravel(), 0, std.ravel(), alpha=0.3, color='red')
plt.title('预测不确定性 std')
plt.xlabel('x')
plt.show()
```

**期望结果**:
- 在训练数据稀疏的区域 (x 接近 ±3),不确定性 std 大 (认知不确定)
- 在训练数据密集的区域 (x ≈ 0),不确定性 std 小
- BNN 知道自己不知道

---

### A.5 三大模型在工业界的真实部署

#### A.5.1 Stable Diffusion 的工程流水线

**完整流水线 (Stable Diffusion XL / SD3)**:

```text
Step 1: 用户输入 prompt
Step 2: CLIP text encoder 把 prompt 编码为文本嵌入 (77 × 1024)
Step 3: 随机采样 latent z_T ~ N(0, I)  (4 × 128 × 128)
Step 4: UNet 在 latent 上做 DDPM 去噪 (50 步,CFG guidance=7.5)
Step 5: VAE decoder 把 z_0 解码回像素 (3 × 1024 × 1024)
Step 6: 后处理: super-resolution (latent upscale) + safety filter
```

**5 个工程关键点**:

**关键 1:VAE 单独训练**

- VAE 先在 ImageNet 上训 1 周,固定
- UNet 单独在 latent 上训 1 周
- **不能端到端** (显存爆炸)

**关键 2:CFG 推理**

- 每次 UNet 前向 2 次 (cond + uncond)
- 显存 ×2,但质量显著提升
- guidance_scale = 7.5 是经验值

**关键 3:xformers / Flash Attention**

- UNet 注意力显存大
- 用 xformers 后**显存省 50%**
- 现代部署标配

**关键 4:TorchScript / ONNX 导出**

- 生产环境不用 PyTorch eager mode
- 导出 ONNX → TensorRT → 推理快 2-3 倍

**关键 5:多 GPU / 分布式**

- Stable Diffusion XL 推理:1 张 A100 8 秒
- batch_size 大 + 多卡 → 接近实时

#### A.5.2 LLM 的概率视角 (GPT 系列)

**训练目标 (cross-entropy)**:
$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i | x_{<i})
$$

**这等价于 MLE**: 最大化 token 序列的对数似然。

**采样 (top-p + temperature)**:
$$
p'(x_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)} \text{ (temperature)}
$$
$$
\text{只保留累积概率} \geq p \text{ 的最小 token 集合 (top-p)}
$$

**5 个手算例子** (采样策略对比):

**例子 1:Greedy (T=0)**

每步选 argmax。可重复,但无多样性。

**例子 2:Temperature = 1**

标准采样。多样但有时乱。

**例子 3:Temperature = 0.7**

更确定,质量高。ChatGPT 默认类似。

**例子 4:Top-p = 0.9**

只从前 90% 概率的 token 中采样,过滤低概率长尾。

**例子 5:Top-k = 50**

只从前 50 个 token 中采样。

#### A.5.3 BNN 在工业界的真实案例

| 公司 | 应用 | BNN 方法 | 效果 |
| :--- | :--- | :--- | :--- |
| **Google** | Waymo 自动驾驶 | Deep Ensemble | 行人检测 +10% 准确 |
| **Tesla** | FSD | Evidential NN | OOD 检测 (路上的异常物体) |
| **Meta** | 广告 CTR | MC Dropout | 探索-利用权衡 |
| **Amazon** | 推荐系统 | Ensemble | 长尾物品推荐 +15% |
| **Microsoft** | 必应搜索 | MC Dropout | 搜索质量评估 |
| **Uber** | ETA 预测 | Deep Ensemble | 极端天气 + 准确度 |
| **DeepMind** | AlphaFold | Ensemble | 蛋白质结构不确定性 |
| **OpenAI** | ChatGPT | Conformal Prediction | 风险评分 |

**5 个工程细节**:

**细节 1:Deep Ensemble 在 Tesla**

5 个不同架构 + 不同数据子集 → 5 倍训练成本,但**自动驾驶安全**值得。

**细节 2:Evidential NN 在医疗**

单次前向 → 快速。Beta 分布 → 概率自然。

**细节 3:MC Dropout 在 Google**

推理时 50 次采样,**置信区间**直接给医生。

**细节 4:Conformal Prediction 在 OpenAI**

保证 90% 预测集有真值,用于风险控制。

**细节 5:蒸馏 BNN**

用大 BNN 蒸馏到小 NN,保留大部分不确定性信息。**实用性强**。

---

### A.6 三大模型的常见误区

#### A.6.1 VAE 误区

**误区 1:"VAE 生成的图就是模糊的"**

**反驳**: NVAE (2020) 在 CelebA 上 FID = 1.6,**接近 GAN**。VQ-VAE 配合强 decoder (如 Transformer) 也很锐利。**模糊 = 弱 decoder + 强 KL 的产物,不是 VAE 的本质**。

**误区 2:"ELBO 越紧越好"**

**反驳**: 太紧的 ELBO = 后验近似太精确 = 过拟合。实际工程上 $K = 5$ 的 IWAE 比 $K = 1$ 标准 ELBO **好但有限**。

**误区 3:"VAE 没有 mode collapse"**

**反驳**: Posterior collapse = $q(z|x) \to p(z)$,就是 VAE 版的 mode collapse。**只是表现不同**。

#### A.6.2 Diffusion 误区

**误区 1:"Diffusion = 慢"**

**反驳**: Consistency Model 1 步,Latent Consistency 2-4 步。**当前 1-4 步就能 SOTA**。

**误区 2:"DDPM 的 ε-prediction 是唯一选择"**

**反驳**: $x$-prediction (预测 $x_0$)、$v$-prediction (预测 $v = \sqrt{\bar\alpha} \epsilon - \sqrt{1-\bar\alpha} x_0$)、score-prediction 都有。**不同选择影响收敛速度和最终质量**。

**误区 3:"Diffusion 不需要 UNet"**

**反驳**: UNet 在 DDPM 上 SOTA,但 DiT (Diffusion Transformer) 在大规模上**超过 UNet**。**架构选择看数据规模**。

**误区 4:"CFG 越大越好"**

**反驳**: CFG 太大 → 多样性崩溃。**guidance_scale = 7.5 是经验值,不能盲目调大**。

#### A.6.3 BNN 误区

**误区 1:"MC Dropout = 真 BNN"**

**反驳**: MC Dropout 是**近似** BNN,理论保证有限。Deep Ensemble 通常**更准**。

**误区 2:"BNN 一定要训 T 次"**

**反驳**: Evidential NN 单次前向就给出不确定性。

**误区 3:"不确定性 = accuracy"**

**反驳**: 预测正确率高 ≠ 不确定性低。Calibration 是独立维度。

**误区 4:"BNN 总是比普通 NN 准"**

**反驳**: 某些任务上 BNN 可能略差。**但多了不确定性维度**。

#### A.6.4 三大模型共有的误区

**误区 1:"更多数据 = 永远更好"**

**反驳**: 分布外数据可能反而降低 in-distribution 性能。**需要数据质量控制**。

**误区 2:"大模型 = 好"**

**反驳**: Diffusion Small 在 CIFAR-10 上可能比 BigGAN 差,但**快 100 倍**。**任务和资源决定模型大小**。

**误区 3:"SOTA = 一定好"**

**反驳**: FID 差 0.5 不一定肉眼可辨。**业务指标才是金标准**。

---

### A.7 与最新研究 (2024-2025) 的联系

#### A.7.1 VAE 相关

**Survey**: Vahdat & Kautz 2020 "NVAE: A Deep Hierarchical Variational Autoencoder"

**最新进展**:
- **DALL-E 3** (2023) 用 VQ-VAE + Transformer
- **Stable Diffusion 3** (2024) 用 Rectified Flow + Transformer (而非 UNet)
- **Consistency Trajectory Models** (2024) 统一 VAE / Flow / Diffusion

#### A.7.2 Diffusion 相关

**综述**: Yang et al. 2023 "Diffusion Models: A Comprehensive Survey"

**最新进展**:
- **DiT (Peebles & Xie 2023)**: Diffusion + Transformer,ImageNet SOTA
- **SiT (2024)**: Scalable Interpolant Transformers
- **SD3 (2024)**: Rectified Flow + Transformer
- **Consistency Models (2023)**: 1 步采样
- **Flow Matching (2024)**: 统一 Diffusion 和 Flow
- **Latent Consistency Models (2024)**: 2-4 步 SOTA
- **Sora (2024)**: 视频 Diffusion

**5 个 2024-2025 SOTA 模型对比**:

| 模型 | 类型 | FID (ImageNet 256) | 步数 |
| :--- | :--- | :--- | :--- |
| **DiT-XL/2** | Transformer Diffusion | 3.04 | 250 |
| **SiT-XL** | Interpolant Transformer | 2.06 | 250 |
| **SD3** | Rectified Flow | 2.0+ | 50 |
| **LCM-SDXL** | Latent Consistency | 4.0 | 4 |
| **Sora** | Video Diffusion | - | - |

#### A.7.3 BNN 相关

**最新进展**:
- **Laplace Redux (2023)**: 简化 Laplace 近似
- **Deep Ensembles 2.0 (2024)**: 更高效的多模型集成
- **MCMC-VI Hybrid (2024)**: 结合两者优点
- **Last-layer BNN (2024)**: 只对最后一层做 BNN,高效

**5 个 SOTA 对比**:

| 方法 | NLL | ECE | OOD AUROC |
| :--- | :--- | :--- | :--- |
| **普通 NN** | 0.50 | 0.10 | 0.85 |
| **MC Dropout** | 0.45 | 0.05 | 0.90 |
| **Deep Ensemble** | 0.40 | 0.03 | 0.93 |
| **Evidential NN** | 0.42 | 0.04 | 0.92 |
| **Last-layer BNN** | 0.41 | 0.04 | 0.91 |

---

### A.8 一页速查:三大模型对比

| 维度 | VAE | Diffusion | BNN |
| :--- | :--- | :--- | :--- |
| **核心思想** | 编码-解码 + 变分推断 | 加噪-去噪 | 权重后验 |
| **训练目标** | ELBO (下界) | ε-MSE (下界) | ELBO / NLL |
| **采样速度** | 1 步 | T 步 (T=1000) | T 次前向 (T=50) |
| **似然** | 下界 | 下界 (紧) | 精确 (MCMC) |
| **样本质量** | 模糊 (经典) / 锐利 (新) | **锐利且多样** | 不生成 |
| **可解释性** | 中 | 中 | **高** (有不确定性) |
| **工业应用** | 数据压缩 / 异常检测 | **图像/视频/音频生成** | 安全关键 / OOD |
| **训练稳定** | **稳定** | **稳定** | 较稳定 |
| **PyTorch 实现** | 100 行 | 200 行 | 150 行 |
| **代表模型** | NVAE, VQ-VAE | DDPM, DiT, SD | MC Dropout, Deep Ensemble |

**5 个工业选择决策**:

**决策 1:要生成高质量图像?** → Latent Diffusion (Stable Diffusion)
**决策 2:要数据压缩或特征学习?** → VAE / VQ-VAE
**决策 3:要不确定性估计?** → MC Dropout / Deep Ensemble
**决策 4:要快速生成 (1 步)?** → Consistency Model
**决策 5:要文本生成?** → 自回归 (LLM)

---

### A.9 5 个"内功心法":工程师的实战建议

#### A.9.1 心法 1:从最简单的 baseline 起步

**错误**: 直接上 Latent Diffusion + Classifier-Free Guidance + UNet。
**正确**: 先训个普通 VAE / AE,验证 pipeline → 再加 diffusion → 再加 UNet → 再加 CFG。

**每加一层都先验证。**

#### A.9.2 心法 2:可视化是 debug 的核心

**生成模型**:
- 可视化**样本** (直观看质量)
- 可视化**latent 空间** (t-SNE/UMAP)
- 可视化**加噪过程** (Diffusion)
- 可视化**不确定性** (BNN: std map)

**代码**:
```python
# 可视化 latent
from sklearn.manifold import TSNE
z = encoder(x).sample()  # [N, latent_dim]
z_2d = TSNE(n_components=2).fit_transform(z)
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10')
```

#### A.9.3 心法 3:复现简单论文 (而非 SOTA)

**为什么?**
- SOTA 论文 = 几十人团队 + 几百万 GPU 时
- 复现简单论文 = 1 人 + 1 GPU + 1 周
- **理解深度**远比"跑通 SOTA"重要

**推荐复现清单 (按难度)**:
1. **MNIST VAE** (1 天)
2. **MNIST DDPM** (3 天)
3. **CIFAR-10 DDPM** (1 周)
4. **CelebA VQ-VAE** (1 周)
5. **MNIST MC Dropout** (半天)

#### A.9.4 心法 4:写自己的"理论小抄"

每学一个新模型,用一页纸手写:

1. **目标函数** (loss 公式)
2. **架构** (画图)
3. **训练步骤** (伪代码)
4. **推理步骤** (伪代码)
5. **3 个关键 trick** (从代码里挖)

#### A.9.5 心法 5:跟紧工业,落地为王

**学术** = 创新方法 + SOTA benchmark
**工业** = 解决问题 + 跑得动 + 成本低

**判断标准**:
- 这个方法能在我的 GPU 上跑吗?
- 训练一周能收敛吗?
- 推理能 < 100ms 吗?
- 显存能 < 8GB 吗?
- 维护成本如何?

**5 个工业落地的"非技术"考量**:
1. **许可证**: Stable Diffusion 是 OpenRAIL,商用 OK;某些模型限制多
2. **可复现性**: GitHub 有 release code + 预训练权重?
3. **社区支持**: Hugging Face / PyTorch Hub 上有现成?
4. **法律风险**: 数据集有版权问题吗?
5. **可解释性**: 客户/监管要求解释决策 (BNN 优势)

---

### A.10 总结:三大模型一句话回顾

| 模型 | 一句话 |
| :--- | :--- |
| **VAE** | "压缩数据到隐空间 + 从先验生成" |
| **Diffusion** | "一步步去噪,把噪声变数据" |
| **BNN** | "给模型不确定性,让它知道自己不知道" |

**学完这一章,你应该能**:
1. 从概率论第一性原理推导 ELBO、ε-MSE、BNN 后验
2. 用 PyTorch 实现 3 个模型的 baseline
3. 在工程上选择合适的模型 + 调参
4. 给面试官讲清楚"为什么 X 模型比 Y 模型好"

**没学的 (留给后续)**:
- Flow Matching (统一 Diffusion 和 Flow)
- Consistency Models (1 步 Diffusion)
- 世界模型 (RL + Diffusion)
- 多模态生成 (CLIP + Diffusion)
- 3D 生成 (NeRF + Diffusion)

**学完整个数学基础文件夹后的"全局视角"**:

```text
数学基础
├── 1-高数: 极限/导数/积分/级数/变分法
├── 2-线性代数: 矩阵/分解/特征值/SVD
├── 3-概率论 (你在这里): 概率分布/MLE/Bayesian/采样/生成模型
├── 4-数值优化: GD/SGD/Adam/凸优化/约束优化
├── 5-信息论: 熵/KL/互信息/编码
├── 6-数理统计: 估计/检验/回归/方差分析
└── 7-深度学习数学: 反向传播/正则化/优化理论/泛化界
```

**这一章把前 9 章的概率论"全部串到深度学习"**。下一章 (如果有) 可以往 Flow Matching、World Model、多模态走,但**扎实掌握本章,工程上已经够用 80% 的场景**。

---

> **附录版本**: v1.0
> **最后更新**: 2026-06-23
> **总行数 (含附录)**: ~2400 行
> **总字节 (含附录)**: ~109000 bytes
> **配套代码**: §11 PyTorch 实验 + §A.4 完整可跑示例 (CVAE / CFG DDPM / MC Dropout BNN)



## 深度扩展 · 多视角系统展开(续)

### 110. 第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调应用场景视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"应用场景" 视角,但落到代码层面是类似的骨架:

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

评估 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调计算复杂度视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"计算复杂度" 视角,但落到代码层面是类似的骨架:

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

评估 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调数值方法视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"数值方法" 视角,但落到代码层面是类似的骨架:

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

评估 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调历史发展视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"历史发展" 视角,但落到代码层面是类似的骨架:

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

评估 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调形式化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"形式化" 视角,但落到代码层面是类似的骨架:

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

评估 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 10 章:概率视角下的深度学习——VAE·Diffusion·Bayesian NN" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

