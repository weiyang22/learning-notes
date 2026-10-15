---
title: 第01章-Python与PyTorch基础
tags:
  - 基础
  - Python
  - PyTorch
  - 编程基础
description: 在动笔写 Python 和 PyTorch 之前,先帮你从根上拆解:为什么 CV 那套"拿来即用"的工具链,在搜推广里"寸步难行"?...
last_updated: 2026-07-13
status: ✅ v1.0
---

# Python & PyTorch & 数据处理基础教程(生成式推荐方向)

> **写在前面的编者按**
>
> 你之前做 CV 多模态,习惯了一上来就 `nn.Conv2d`、`ResNet`、`ViT`;现在转搜推广/生成式推荐,突然面对 `nn.Embedding`、`nn.EmbeddingBag`、`DefaultDict`、`SparseAdam`、动辄几千万行的稀疏 ID 矩阵——你会发现:**不是 PyTorch 变了,而是你面对的"数据形态"和"模型目标"完全变了**。
>
> 本篇不讲花活、不堆 API 列表,只聚焦你最痛的三个底层:
> 1. **Python 底层**:不是语法层面"会写",而是"知道每个数据结构在内存里长什么样、为什么 list 不能 in 查、为什么 dict 要预分配"
> 2. **PyTorch 底层**:不是"会调函数",而是"知道张量怎么索引、Embedding 怎么查表、Autograd 怎么画图"
> 3. **数据预处理**:不是"会用 pandas",而是"知道在搜推广里,一个样本是怎么从原始日志一步步变成能喂进模型的 Tensor"
>
> 每个章节会包含:**原理 + 为什么这样设计 + 搜推广中的具体坑 + 可直接运行的代码 + 性能/对比表**。
>
> 通读本篇后,你看团队搜推广代码库的"输入层、特征工程、训练循环"应该不再发怵。

---

## 第 0 章:从 CV 到搜推广——为什么这套基础必学

在动笔写 Python 和 PyTorch 之前,先帮你从根上拆解:**为什么 CV 那套"拿来即用"的工具链,在搜推广里"寸步难行"?**

### 0.1 数据形态的根本差异

| 维度 | CV(图像/视频) | 搜推广(点击/转化) |
| :--- | :--- | :--- |
| **输入张量形态** | `[B, C, H, W]` 连续稠密,像素值 0~255 | `[B]` 或 `[B, T]` 离散 ID,取值范围 [0, 几千万) |
| **数据密度** | 稠密,几乎所有像素都"有意义" | 极度稀疏,一个用户-物品交互矩阵 99.99% 是 0 |
| **类别基数** | 通常 1000 类(ImageNet) | 用户/物品/位置/类目 ID 动辄千万~亿级 |
| **特征类型** | 主要是图像本身 | 异构:离散 ID + 连续数值 + 时序序列 + 文本 + 上下文 |
| **样本量** | ImageNet 128 万 | 工业界日均百亿~千亿条曝光 |
| **正负样本比** | 接近 1:1 | 1:100 ~ 1:1000(点击远少于曝光) |

**结论**:CV 里你拿到的 Tensor 都是"已经张好的图",直接喂 Conv2d 就完事;搜推广里你面对的是**一堆原始的 ID 字符串、时间戳、点击日志**,你必须亲手把它们**数值化 → 离散化 → Embedding → 拼成 batch**,每一步都可能踩坑。

### 0.2 模型目标的根本差异

| 维度 | CV | 搜推广 |
| :--- | :--- | :--- |
| **核心问题** | "这张图里有什么?"(分类/检测/分割) | "给这个用户,在这个上下文,这个物品的点击率是多少?"(CTR/CVR 预估) |
| **模型主干** | 卷积/Transformer,所有像素共享权重 | Embedding 查表 + 浅层 MLP(精排)或 双塔(召回) |
| **参数量大头** | 都在 MLP/Conv 权重里 | **全在 Embedding 表里**(几千万 × 几十维 = 几十 GB) |
| **推理时延要求** | 100ms 内即可(检测/分割可放宽) | **< 20ms**(精排),< 5ms(召回) |
| **过拟合来源** | 数据少/模型大 | 长尾 ID 噪声(出现一两次的 ID 严重过拟合) |

**结论**:CV 里你调参主要调学习率/数据增强;搜推广里你调参要调**正则化强度、Embedding 维度、稀疏更新策略、负采样数**。

### 0.3 工程要求的根本差异

| 维度 | CV | 搜推广 |
| :--- | :--- | :--- |
| **数据规模** | GB 级(TFRecord) | **TB ~ PB 级**(原始日志) |
| **训练时长** | 几天~几周 | 几小时~几天(数据太大,反而 epoch 跑不完) |
| **线上服务** | 一张图 → 一次推理 | **一次请求 → 数百候选打分**(精排)、**亿级向量召回**(召回) |
| **特征一致性** | 训练/推理输入相同 | **离线特征 ≠ 线上特征**(典型坑,后面会详讲) |
| **可解释性** | 可视化注意力/GradCAM | 几乎不可解释,主要靠业务指标(AUC/CTCVR) |

**结论**:搜推广的工程复杂度主要不在"模型多炫",而在"**数据流**"和"**特征对齐**"上。

### 0.4 本篇的章节地图

```text
第 0 章:思维转变(为什么必学) ✅
    ↓
第 1 章:Python 底层(Python 数据结构怎么用、为什么这么用)
    ↓
第 2 章:数据处理(从原始日志到模型可吃的 Tensor)
    ↓
第 3 章:PyTorch 底层(张量、Autograd、nn.Module、Embedding)
    ↓
第 4 章:从 0 构建搜推广训练数据(生成式推荐完整 pipeline)
    ↓
第 5 章:训练流程与工程实践(训练循环、调试、正则化、保存)
    ↓
自测题 & 下一章预告
```

> 接下来,直接进入正题。
> 接下来,直接进入正题。但等一下——在碰 Python 和 PyTorch 之前,我们需要先把"地基"打牢。

---

## 第 0.5 章:深度学习数学与理论基础——从根上理解你在写的每一行代码

> **"根基不牢,地动山摇。"**
>
> 你在写 `loss.backward()` 时,知道它在 GPU 上到底执行了什么数学运算吗?
> 你在调 `learning_rate=1e-3` 时,知道为什么这个值对 Transformer 工作而对 CNN 不工作?
> 你在用 `F.cross_entropy` 时,知道为什么它比手动 `softmax + log` 稳定 1000 倍?
>
> 这一章不讲 Python,不讲 PyTorch API,只讲**每行代码背后的数学和计算机原理**。

### 0.5.1 浮点数:计算机如何表示"数"?——为什么你的 Loss 变成了 NaN?

#### IEEE 754:FP32/FP16/BF16 的位级解剖

| 格式 | 指数位 | 尾数位 | 最大正数 | 最小正数(正规) | 精度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 8 | 23 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 7位十进制数字 |
| FP16 | **5** | 10 | 6.6×10⁴ | 6.1×10⁻⁵ | 3位十进制数字 |
| BF16 | **8** | 7 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 2~3位十进制数字 |

**这条规则决定了混合精度训练的一切!** FP16 的 5 位指数导致范围只有 6e-5~6.5e4,梯度小了 underflow(变 0),大了 overflow(变 inf)。BF16 保留了 FP32 的 8 位指数范围,所以没有这个问题,这也是为什么 LLaMA 等大模型训练都用 BF16。

**GradScaler 的原理**:把 loss 放大 S=2¹⁶ 倍,梯度也放大一样倍数,让小梯度不 underflow。backward 后 unscale 恢复原始大小。

**量化的本质**:INT8 用 256 个离散值来近似 2³² 个浮点数。关键参数是 scale 和 zero_point,分别控制量化的步长和零点偏移。

### 0.5.2 优化器:从 SGD 到 AdamW 的进化逻辑——为什么搜推广默认用 Adam?

**SGD → Momentum → Adam 的问题链**:

```python
# SGD: θ_{t+1} = θ_t - η·g_t          — 各向同性,在沟壑地形震荡
# Momentum: v_t = β·v_{t-1} + g_t     — 积累动量,加速收敛
# Adam: θ = θ - η·m̂/(√v̂+ε)           — 每个参数独立学习率!
#                                           热门参数:梯度大→v大→lr自动减小
#                                           冷门参数:梯度小→v小→lr自动增大
#  这就是搜推广用 Adam 的本质原因:用户/物品 Embedding 频率差异极大!
```

**AdamW 为什么比 Adam 好**:Adam 中 L2 正则的梯度 g_l2=2λθ 会经过 √v̂ 缩放——热门参数被过度正则,冷门参数正则不足。AdamW 把 weight_decay 从梯度计算中解耦,直接 θ -= η·λ·θ,不经过 √v̂ 缩放。

**Warmup 为什么必要**:Adam 的 v_t 在训练开始时需要累积梯度平方信息,前几步的方差估计极度不可靠,会导致更新步长被放大几十倍。

### 0.5.3 初始化:为什么一个随机种子能决定训练成败?

**核心要求**:信息流在深度网络中不能爆炸也不能消失。
- 前向: Var[y] = Var[x] (每层输出方差守恒)
- 反向: Var[∂L/∂x] = Var[∂L/∂y] (梯度方差守恒)

**Xavier 初始化**(tanh/sigmoid): W ~ U(-√(6/(d_in+d_out)), √(6/(d_in+d_out)))
**Kaiming 初始化**(ReLU): W ~ N(0, √(2/d_in))

为什么差这么多? ReLU 把一半神经元置 0,方差砍半,所以要放大。

**错误初始化**:Var[W]=1 则 50 层后方差 d⁵⁰ 倍指数爆炸;Var[W]=0.001 则指数消失。

**残差连接 = 初始化无关**:y = F(x) + x → ∂L/∂x = ∂L/∂y·(I+∂F/∂x),即使 F 的梯度为 0,还有 I 直传。

### 0.5.4 反向传播手动推导——理解 loss.backward() 在 GPU 上做了什么

**线性层三大核心公式**:
```text
dL_dW = x^T @ dL_dy          ← 权重的梯度 = 输入的转置 × 输出梯度
dL_dx = dL_dy @ W^T          ← 输入的梯度 = 输出梯度 × 权重的转置
dL_db = sum(dL_dy, dim=0)    ← 偏置的梯度 = 输出梯度的和
```

**Softmax + Cross-Entropy 最美公式**: ∂L/∂z_i = p_i - y_i
只需要 logits 和 labels,一减就得到梯度,这就是 CrossEntropyLoss 内部实际做的事情。

### 0.5.5 数值稳定性:log-sum-exp 技巧

**核心问题**:x=[1000,1001,999],exp(1000)溢出。

**解法**:log Σ exp(x_i) = c + log Σ exp(x_i-c), c=max(x_i)
验证:1001 + log(exp(-1)+1+exp(-2)) = 1001.401 ✓

**PyTorch 里已经稳定的函数**:F.cross_entropy, BCEWithLogitsLoss, F.log_softmax, torch.logsumexp

**永远不要这样写**:torch.log(torch.softmax(logits)) 或 torch.log(torch.sigmoid(logits))

### 0.5.6 信息论:Cross-Entropy 在优化什么?

- **熵 H(p)** = -Σp log p: 编码 p 所需最少 bit 数
- **交叉熵 H(p,q)** = -Σp log q: 用 q 编码 p 的 bit 数
- **KL 散度 D_KL(p||q)** = H(p,q) - H(p): 额外 bit 数

**min H(p,q) = min D_KL(p||q) = max likelihood**

**损失函数 = 概率分布的负对数似然**:
- BCEWithLogitsLoss → Bernoulli(二分类)
- CrossEntropyLoss → Categorical(多分类)
- MSELoss → Gaussian(回归)
- L1Loss → Laplace(鲁棒回归)

### 0.5.7 AUC 数学与注意力 √d 缩放

**AUC = P(正样本 > 负样本)**, 不受 class imbalance 影响。
**GAUC**:按用户分组加权平均,能看到个性化效果,比全局 AUC 更有参考意义。

**Attention 除以 √d 的原因**:q·k 的方差 = d,softmax 对输入 scale 极度敏感。不缩放则当 d=64 时 softmax 尖锐到梯度消失,缩放后保持方差为 1,梯度正常。

### 0.5.8 正则化的贝叶斯视角

**L2 = 高斯先验**:权重趋向 0 但不精确为 0。
**L1 = 拉普拉斯先验**:权重精确为 0(稀疏性)。
**Dropout = 模型集成**:2ⁿ 个共享参数的"瘦网络"的平均。

### 0.5.9 概率论与统计推断——为什么 MLE = 最小化 KL 散度?

> **这是几乎所有"损失函数"推导的起点**。理解 MLE/MAP 与 KL 散度的关系,你就理解了为什么 `CrossEntropyLoss = -log p_θ(y|x)` 是天经地义的。

#### 0.5.9.1 似然函数与 MLE

**核心思想**:给定数据 $D = \{(x_i, y_i)\}_{i=1}^N$,我们想找到参数 $\theta$ 使得"在 $\theta$ 下出现这些数据的概率"最大。

```python
# 似然函数:在参数 θ 下,观测到数据 D 的概率
# L(θ) = Π p_θ(x_i, y_i)
# log 似然:  ℓ(θ) = Σ log p_θ(x_i, y_i)
# MLE 估计:  θ* = arg max_θ ℓ(θ)
```

**为什么是负对数似然(NLL)?**

```text
实际工程里,我们用最小化"负对数似然"(NLL):
  loss = -Σ log p_θ(y_i | x_i)  ← 训练时最小化这个
而不是最大化"似然":
  max Π p_θ(y_i | x_i)        ← 数学上等价,但数值上灾难

原因:
1. 数值稳定: 概率 p ∈ (0, 1],  log(p) ∈ (-∞, 0]
   当 p 很小时 (p=1e-20), 直接连乘 Π p → underflow 到 0
   取 log 后变成 Σ log(p) = -20N → 不会 underflow

2. 计算方便: log(Π p) = Σ log(p) → 求和比连乘快

3. 优化稳定: 损失函数 = -log p 是"凸"的近似, 梯度不爆炸
```

**MLE 与 KL 散度的等价性(关键定理)**:

```text
定理: 最大化对数似然 = 最小化 KL 散度(到经验分布)

证明:
   min_θ D_KL(p_data || p_θ)
 = min_θ E_{p_data}[log p_data(x) - log p_θ(x)]
 = min_θ -E_{p_data}[log p_θ(x)]  (因为 p_data 固定, 第一项是常数)
 = max_θ E_{p_data}[log p_θ(x)]
 ≈ max_θ (1/N) Σ log p_θ(x_i)            (蒙特卡洛近似)
 = max_θ ℓ(θ) = MLE  ✓

所以:你训练模型时在"最大化似然"  ⇔  "在让模型分布 p_θ 逼近真实分布 p_data"
```

**搜推广的实战对应**:
```python
# CTR 预估: p_θ(click=1 | user, item) = σ(MLP(emb))
# MLE → 最小化 BCE Loss
loss = F.binary_cross_entropy_with_logits(logits, labels)
#  ↑ 这就是 -(label·log p + (1-label)·log(1-p))

# 多分类(下一个 item 预测): p_θ(item | history) = softmax(Transformer)
# MLE → 最小化 CrossEntropyLoss
loss = F.cross_entropy(logits, target_item_ids)
```

#### 0.5.9.2 MAP:带先验的最大似然

**问题**:纯 MLE 在小数据上容易过拟合——比如冷启动用户只有 1 次点击,MLE 会"死死记住"这一次行为。

**MAP(Maximum A Posteriori)**:加入对参数的先验信念。

```text
贝叶斯定理:
  p(θ | D) = p(D | θ) · p(θ) / p(D)
                ↑            ↑
                似然        先验

MAP 估计: θ* = arg max_θ p(θ | D)
              = arg max_θ [log p(D | θ) + log p(θ)]
              = arg max_θ [ℓ(θ) + log p(θ)]
                ↑ MLE 项      ↑ 先验项

对比:
  MLE:  θ* = arg max ℓ(θ)
  MAP:  θ* = arg max [ℓ(θ) + log p(θ)]
```

**先验的选择决定了正则化**:

| 先验 p(θ) | log p(θ) | 等价的正则化 | 搜推广用法 |
| :--- | :--- | :--- | :--- |
| $\mathcal{N}(0, \sigma^2)$ 高斯 | $-\theta^2/(2\sigma^2)$ | L2 正则 | 通用 |
| $\text{Laplace}(0, b)$ 拉普拉斯 | $-\|\theta\|/b$ | L1 正则 | 稀疏解 |
| $\mathcal{U}(0, a)$ 均匀 | 0 (截断外) | 截断 L2 | 边界约束 |
| $\text{Student-t}$ | $-\log(1+\theta^2/\nu)$ | 近似 L0 | 鲁棒回归 |

```python
# 在 PyTorch 里,MAP ≈ MLE + weight_decay
# 因为 L2 正则项  λ·||θ||²/2  正好 = -log p(θ) + const (高斯先验)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# ↑ 这里的 weight_decay=1e-4 ≈ 高斯先验的精度 1/σ² = 1e-4
# 越小 σ² 越大 = 先验越"宽松" = 越接近 MLE
```

#### 0.5.9.3 期望、方差与协方差矩阵

**这些是理解"为什么 Embedding 表那么大维度才够用"的关键**:

```text
期望:     E[X] = Σ x · p(x)   (离散)  /  ∫ x · p(x) dx (连续)
方差:     Var[X] = E[(X - E[X])²]  =  E[X²] - (E[X])²
协方差:   Cov[X, Y] = E[(X - E[X])(Y - E[Y])]

如果 X, Y 独立:  Cov[X, Y] = 0
反之不成立:     X, Y 不相关  ⇏  X, Y 独立
(独立 ⇒ 不相关, 但不相关不必然独立)
```

**Embedding 维度选择的概率论解释**:

```text
假设用户/物品 ID 服从均匀分布 U(0, N-1), N=10^7
一个 32 维 Embedding 的每个分量服从 N(0, σ²) 初始化

问题: d 维 Embedding 能否区分 N 个不同 ID?

直观: d 维空间的"容量"和 2^d 相关
  d=8:  256 种正交向量 → 无法区分 10^7 个
  d=32: ~4×10^9 种正交向量 → 足以区分 10^7 个 (余量充足)
  d=64: 1.8×10^19 种 → 严重过参数化

但这仅是"区分能力"—— 实际有用的 embedding 还需要"几何结构"
  相似用户 → 相似向量
  → 这就是"度量学习"的范畴(第 8 章会详细讲)
```

#### 0.5.9.4 协方差矩阵与 PCA

**为什么 Embedding 需要归一化才能做内积检索?**

```text
假设 user_emb = [u_1, u_2, ..., u_d] 是 d 维向量
其协方差矩阵 Σ 是 d×d

如果 Σ ≈ σ²I (各向同性):
  内积 <u, v> = Σ u_i·v_i 是"自然"的相似度
  角度 cos(θ) = <u,v>/(||u||·||v||) 才有意义

如果 Σ ≠ σ²I (各向异性):
  内积受尺度影响 → "用户 1 在维度 1 上的值大"会主导内积
  此时需要 Mahalanobis 距离: dist(u,v) = sqrt((u-v)^T Σ^{-1} (u-v))
  或者在检索前做 whitening: ẽ = Σ^{-1/2} · e

这就是为什么 Faiss 检索前要做 L2 归一化 (||e|| = 1) 的数学原因
```

### 0.5.10 矩阵论与线性代数——推荐系统的"骨架语言"

> **推荐系统里 90% 的高级话题都能用矩阵语言描述**。理解 SVD、特征值、矩阵分解,你就理解了协同过滤、双塔、Embedding 全连接的本质。

#### 0.5.10.1 矩阵分解:协同过滤的数学本质

**协同过滤的矩阵分解视角**:
```text
用户-物品交互矩阵 R (N_users × N_items):
  R[i][j] = 1 if 用户 i 点了物品 j, 0 otherwise
  R 通常极稀疏(99.99% 是 0)

矩阵分解假设:  R ≈ U · V^T
  U ∈ R^{N_users × k}  ← 用户隐向量
  V ∈ R^{N_items × k}  ← 物品隐向量
  k ≪ N_users, N_items

预测:  R̂[i][j] = <U[i], V[j]>  =  用户 i 对物品 j 的偏好

这就是"双塔"模型的数学本质!
  塔 1: user → U[i]  (编码器)
  塔 2: item → V[j]  (编码器)
  预测: <塔1输出, 塔2输出>

在工业界:
  U 矩阵 = nn.Embedding(N_users, k)  ← user_emb table
  V 矩阵 = nn.Embedding(N_items, k)  ← item_emb table
```

**为什么矩阵分解有效?**

```text
定理 (低秩假设, 用户/物品在 k 维隐空间):
  如果真实用户偏好 = k 个潜在因子 (品类、品牌、价格敏感度...) 的线性组合,
  则 R 的"信息秩" ≤ k

矩阵分解 R ≈ UV^T 用最少的参数(k·(N+M) 个)重建 R:
  参数效率: 矩阵分解 k(N+M) vs 全 R 参数 NM
  例子:  N=10^7, M=10^6, k=64
    全参数: 10^13 个 (不可行!)
    矩阵分解: 7.04 × 10^8 个 (可行)

泛化能力: 矩阵分解是"正则化"——它强制 R 是低秩的
  未观测的 R[i][j] = 0,但矩阵分解可以基于隐向量预测非零
  → 这就是"泛化到未见样本"!
```

#### 0.5.10.2 SVD:推荐系统的"瑞士军刀"

**SVD(奇异值分解)与特征值分解的关系**:

```text
任意矩阵 A (m×n) 都可以分解为:
  A = U · Σ · V^T
  U (m×m): 左奇异向量, 列正交
  Σ (m×n): 奇异值矩阵, 对角线上 σ_1 ≥ σ_2 ≥ ... ≥ σ_r > 0
  V (n×n): 右奇异向量, 列正交

谱定理(对称矩阵专属):  A = A^T 时
  A = Q · Λ · Q^T
  Q (n×n): 特征向量, 正交
  Λ: 特征值对角矩阵

关系: A 的非零奇异值² = A^T·A 的特征值
```

**截断 SVD 在推荐中的应用**:
```python
import numpy as np

# 模拟用户-物品交互矩阵
R = np.random.binomial(1, 0.01, (1000, 5000))   # 1000 用户, 5000 物品, 1% 密度

# SVD 分解
U, S, Vt = np.linalg.svd(R, full_matrices=False)
# U: (1000, 1000),  S: (1000,)  Vt: (1000, 5000)

# 截断到 k=32 维
k = 32
U_k = U[:, :k]            # (1000, 32)
S_k = np.diag(S[:k])      # (32, 32)
V_k = Vt[:k, :].T         # (5000, 32)

# 重建 R
R_reconstructed = U_k @ S_k @ V_k.T
# 误差: ||R - R_reconstructed||_F = Σ_{i>k} σ_i²

# 用 k=32 维的隐向量做推荐:
# 用户 i 的表示:  U_k[i] (32 维)
# 物品 j 的表示:  V_k[j] (32 维)
# 预测分数:  R̂[i][j] = U_k[i] @ S_k @ V_k[j]
```

**为什么工业界不直接用 SVD?**
```text
1. 复杂度: SVD 是 O(min(m,n)² · max(m,n))  → 1亿×1千万 矩阵直接不可行
2. 不能处理新用户/新物品:  SVD 必须重算才能加入新行/新列
3. 不能融合辅助特征:  只有交互矩阵,没有用户/物品的属性信息

工业界方案:
  - ALS(交替最小二乘):  固定 U 优化 V, 固定 V 优化 U → 可并行
  - BPR-MF:  用贝叶斯个性化排序的 MLE 损失
  - FM:  在矩阵分解基础上加特征交叉项
  - 神经网络双塔:  U, V 换成更复杂的编码器
```

#### 0.5.10.3 矩阵条件数:为什么梯度下降有时失效?

**条件数 (Condition Number)**:

```text
矩阵 A 的条件数:  κ(A) = σ_max / σ_min
  其中 σ_max, σ_min 是最大/最小奇异值

含义:  κ 衡量"输入微小变化 → 输出放大多少倍"
  κ 小 (≈ 1):  良态, 数值稳定
  κ 大 (>> 1): 病态, 数值不稳定

对优化:  κ(Hessian) 决定了梯度下降的收敛速度
  凸优化收敛率 ∝ (κ-1)/(κ+1)
  κ = 1: 一步收敛 (理想)
  κ = 10^6: 需要 ~10^6 步!
```

**搜推广的"病态"问题**:

```python
# Embedding 表的"条件数"问题
import torch
emb = nn.Embedding(10_000_000, 32)

# 初始化后, 某些 ID 的 embedding 范数特别大
# 某些 ID 的范数特别小(因为罕见, 训练时被更新次数少)
norms = emb.weight.norm(dim=-1)
print(f"max norm: {norms.max():.3f}, min norm: {norms.min():.3f}")
print(f"ratio: {norms.max() / norms.min():.1f}")   # 可能 > 1000
# 病态! 同一 batch 内不同 ID 的梯度量级差 1000 倍 → 训练不稳定
```

**解法**:
```text
1. 梯度裁剪: 限制单步最大变化
2. LayerNorm: 强制每层输出范数在 1 附近
3. 归一化 Embedding: 训练时 ||emb[id]|| = 1, 推理时再恢复
4. Adam 的自适应学习率: 自动补偿量级差异
5. 重新初始化: 冷启动 ID 用预训练的平均值
```

#### 0.5.10.4 特征值分解与谱方法

**为什么 PCA 用特征值分解?**
```text
PCA(主成分分析) 找数据的最大方差方向
数学上等价于: 找协方差矩阵 Σ 的最大特征值对应的特征向量

证明: 投影到方向 v 后方差 = v^T Σ v
       拉格朗日乘子:  max v^T Σ v   s.t.  v^T v = 1
       求导:  Σv = λv
       所以 v 是 Σ 的特征向量,  λ 是对应特征值 ✓

应用到推荐:
  用户的 Embedding 表是一个 N×d 矩阵
  对其做 PCA, 取前 k 个主成分 → 这 k 维捕捉了用户偏好的"主要变异"
  → 用作"用户画像"或"冷启动初始值"
```

### 0.5.11 优化理论基础——为什么梯度下降能找到最优解?

> **理解凸优化的"为什么收敛" 和 非凸优化的"为什么有时也收敛"**, 你就理解了为什么 Adam 比 SGD 好、为什么学习率要 warmup、为什么 SGD 反而比 Adam 泛化好。

#### 0.5.11.1 凸函数与全局最优

```text
凸函数定义:  f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)  for all λ ∈ [0,1]
含义: 任意两点连线在函数图像上方

凸优化的"天堂":
  - 局部最优 = 全局最优  ✓
  - 梯度下降保证收敛  ✓
  - 收敛率有理论保证  ✓

但深度学习不是凸的:
  - 神经网络损失函数 = 万亿维非凸曲面
  - 局部最优很多, 鞍点更多
  - 为什么还能用 SGD 训练? 见 0.5.11.4
```

#### 0.5.11.2 梯度下降的收敛性证明

**Lipschitz 连续梯度**:
```text
如果 ∇f 是 L-Lipschitz 连续的:
  ||∇f(x) - ∇f(y)|| ≤ L · ||x - y||

则 f 是"二次上界"函数:
  f(y) ≤ f(x) + ∇f(x)·(y-x) + (L/2)·||y-x||²

梯度下降:  x_{t+1} = x_t - η · ∇f(x_t)
保证收敛的条件:  η < 2/L
  η = 1/L 时最优 (加速版本)

收敛率:  f(x_T) - f(x*) ≤ ||x_0 - x*||² / (2ηT)
即: T 步后误差 ≤ O(1/T)  → "次线性收敛"

强凸情况下:  线性收敛,  误差 ≤ c^T · 常数
```

**在搜推广中 L 是多少?**

```python
# 一个真实 Transformer 模型的 L 估算
# 假设 d_model = 512, 6 层
# 每层 Jacobian 的 Lipschitz 常数 = max 奇异值 ≈ 1 (Pre-Norm 设计)
# 整网络 L ≈ 6 * 1 = 6
# 理论学习率:  η < 2/L ≈ 0.33

# 但实际工程经验: Transformer 用 η = 1e-3 ~ 1e-4
# 比理论值小 1000 倍!
# 原因: 理论 L 是"最坏情况", 实际梯度小很多; 且要兼顾稳定性
```

#### 0.5.11.3 随机梯度下降 (SGD) vs 批量梯度下降

**为什么用 SGD 而不是全量 GD?**

```text
全量 GD:  θ_{t+1} = θ_t - η · (1/N) Σ ∇L_i(θ_t)
  每次用全部 N 个样本算梯度
  问题: N=10 亿时一次梯度计算要几小时

SGD:  θ_{t+1} = θ_t - η · ∇L_i(θ_t)    (i 随机采样)
  每次用 1 个样本
  优点: 速度快, 适合在线学习
  缺点: 梯度噪声大, 收敛震荡

Mini-batch SGD:  折中方案
  每次用 B 个样本 (B=256, 1024, ...)
  实际工程标准
  数学上:  无偏估计 E[∇L_B] = ∇L_full
  方差:  Var[∇L_B] = Var[∇L_1] / B
        → batch 越大, 梯度越稳定, 但算力也越大

最优 batch size 的选择(经验):
  GPU 显存能装下越大越好
  256-2048 是常见范围
  batch 翻倍时, 学习率要相应翻倍 (linear scaling rule)
```

**为什么大 batch 需要大学习率?——Linear Scaling Rule**

```text
考虑 batch B 的梯度:
  g_B = (1/B) Σ_{i=1}^B ∇L_i

  E[g_B] = ∇L_full
  Var[g_B] = σ²/B    (σ² 是单样本梯度方差)

要保持"参数更新步长的方差"恒定:
  Var[η·g_B] = σ²η²/B = const
  →  当 B 翻倍, η 也应翻倍

这就是 "linear scaling rule":  batch 翻倍, lr 翻倍
(Krizhevsky 2017 在 ResNet 上验证)
```

#### 0.5.11.4 为什么非凸也能用 SGD?(2024 年的理论进展)

```text
深度学习是非凸的, 但 SGD 几乎总能找到好解。
原因 (近年理论):

1. 损失函数是"过参数化"的
   参数量 >> 训练样本数
   → 任意训练样本的 loss 都能降到 0
   → 解存在于"低维流形"上, SGD 容易找到

2. 局部最优极少, 鞍点占多数
   高维空间中, 几乎所有驻点都是鞍点 (不是局部最优!)
   鞍点处 ∇L = 0, 但 Hessian 有正有负特征值
   SGD 的噪声自然帮助"逃离"鞍点

3. 隐式正则化
   SGD 倾向于找到"平坦最小值" (见 0.5.14)
   平坦最小值 = 泛化好
   → SGD 比 Adam 泛化好(因为 Adam 偏向锐利最小值)

4. 现代架构设计
   Pre-Norm, Residual, RMSNorm, 都在让 loss landscape 更平滑
   → 更容易优化
```

**Adam 为什么训练快但可能泛化差?**

```text
SGD:  偏向"平坦最小值", 泛化好
Adam: 偏向"锐利最小值", 泛化差 (但训练 loss 更低)

原因:
  SGD:  θ -= η · g          所有参数同 lr
        锐利 minima 处, 任意方向变化都让 loss 飙升
        → SGD 一旦靠近, 不敢再大步走
        → 自然停在平坦处

  Adam: θ -= η · m̂/√(v̂+ε)  每个参数独立 lr
        锐利 minima 的 v 很小 (梯度小)
        → 该方向 lr 自动放大
        → Adam 敢"冲进去"
        → 停在了锐利处

实践:  训练时用 Adam 加速,  微调时切到 SGD 提升泛化 (工业界技巧)
       或: 训练全程 AdamW,  不要 Adam
```

### 0.5.12 信息论进阶——为什么熵和互信息如此重要?

> **熵不只是一个数字, 它是衡量"信息量""不确定性""编码代价"的核心工具**。在搜推广里, 熵决定了你的特征够不够、损失函数的"信息"流向了哪里。

#### 0.5.12.1 熵的更多性质

**链式法则(Chain Rule)**:
```text
H(X, Y) = H(X) + H(Y | X) = H(Y) + H(X | Y)
含义: 联合熵 = 单独熵 + 条件熵

应用到搜推广:
  H(用户, 物品, 上下文, 标签) = H(标签) + H(用户 | 标签) + H(物品 | 标签, 用户) + ...
  → 条件熵越小, 特征越能"决定"标签
  → 特征选择 = 选那些 H(标签 | 特征) 小的特征
```

**熵的不等式**:
```text
1. H(X) ≥ 0,   H(X) = 0 iff X 是常数
2. H(X | Y) ≤ H(X)             知道 Y 不会增加 X 的不确定性
3. H(X | Y) ≥ H(X | Y, Z)      给更多信息只会降低熵
4. 条件独立:  H(X, Y | Z) = H(X | Z) + H(Y | Z) iff X ⊥ Y | Z
```

**最大熵原理(工业界频用)**:
```text
定理: 在满足约束的分布族中, 熵最大的分布是最"无偏"的
例: 满足 E[X] = μ, E[X²] = μ² + σ² 的所有分布中,
    熵最大的是 高斯分布 N(μ, σ²)

搜推广应用:
  - 连续特征归一化: 假设服从高斯 → 用 μ±3σ 截断是合理的
  - 词表采样: 满足边际分布的所有联合分布中, 最大熵的联合分布是独立
    → 训练时随机 shuffle 是合理的(无信息泄漏)
```

#### 0.5.12.2 互信息与信息瓶颈(Information Bottleneck)

**互信息定义**:
```text
I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)
         = D_KL(p(x, y) || p(x)p(y))

含义: 知道 Y 之后, X 的不确定性减少了多少
     等价: 知道 X 之后, Y 的不确定性减少了多少
     对称:  I(X; Y) = I(Y; X)

搜推广:  I(用户特征; 点击) = 多少信息
        信息瓶颈:  保留 I(X; 隐向量) ≤ R  的约束下, 最大化 I(隐向量; 标签)
        → 这就是"自监督学习"的理论动机
```

**信息瓶颈理论(深度学习统一视角)**:
```text
Tishby 等人(2015) 提出:
  深度学习 = 信息瓶颈的两阶段优化
  阶段 1 (拟合): 最大化 I(X; Z),  I(Z; Y) 快速上升 (压缩还没开始)
  阶段 2 (压缩): I(X; Z) 开始下降, 但 I(Z; Y) 保持
                  → 网络"忘记"输入中无关的信息, 只保留和标签相关的

实验验证: 用互信息估计 (MINE) 观察训练过程
  - 训练初期:  I(X; Z) 涨,  I(Z; Y) 涨
  - 训练后期:  I(X; Z) 跌,  I(Z; Y) 持平
  → 这就是"遗忘"和"泛化"的物理意义

搜推广的关联:
  特征工程 = 手动做"信息瓶颈", 把原始 X 压缩成 Z
  Embedding = 让模型自己学 Z
  好的 Embedding:  I(Z; label) 大,  I(X; Z) 小 (压缩效率高)
```

#### 0.5.12.3 交叉熵的多种解读

```text
H(p, q) = -Σ p(x) log q(x)  有 4 种不同解读:

解读 1: 信息论  → 用 q 编码 p 所需的 bit 数
解读 2: 概率论  → -log q(x) 是 q 给出 x 的"惊讶度", 取期望
解读 3: 统计    → MLE 的目标函数
解读 4: 几何    → p, q 的"对数内积"

p = q 时:  H(p, p) = H(p) 最小 (此时 D_KL = 0)
p ≠ q 时:  H(p, q) > H(p)  额外的"编码代价"
```

**为什么搜推广的 Loss = CE 而不是 MSE?**

```python
# 多分类(下一个 item 预测): 应该用 CE
loss_ce = F.cross_entropy(logits, target_ids)  # ✓ 正确
loss_mse = F.mse_loss(logits, target_oh)       # ✗ 错误但可以跑

# 原因 1: 几何解释
#   logits 是无界的,  softmax 后的概率是 simplex 上的点
#   CE 是在 simplex 上的"自然"距离
#   MSE 是欧氏距离,  在 simplex 上无意义

# 原因 2: 梯度解释
#   d(CE)/d(logit) = p_i - y_i  (有正有负, 概率差驱动)
#   d(MSE)/d(logit) = 2(p_i - y_i)·p_i·(1-p_i)  (多了 0~1 系数, 信号衰减)

# 原因 3: 信息论解释
#   CE = -E_p[log q],  当 p 已知,  等价 MLE
#   MSE 假设高斯噪声,  分类不是高斯
```

### 0.5.13 数值稳定性深度——为什么 FP16/BF16 训练如此微妙?

> **数值稳定性是混合精度训练的核心**。不理解浮点数, 你就会在 OOM、NaN、loss 不收敛之间反复横跳。

#### 0.5.13.1 浮点数的"绝对禁区"

```python
# 实验: 哪些运算容易出问题?
import torch
import numpy as np

# 1. Softmax 的 overflow
logits = torch.tensor([1000., 1001., 999.])
torch.softmax(logits, dim=-1)
# 报错或返回 nan, 因为 exp(1000) overflow

# 解决: log-sum-exp 技巧
def logsumexp(x):
    m = x.max()
    return m + torch.log(torch.exp(x - m).sum())

# 2. Log of 0
torch.log(torch.tensor(0.))  # -inf

# 3. Division by 0
1.0 / 0.0  # inf, 不是错误! 但是后续计算会传染

# 4. Subtraction of close numbers (catastrophic cancellation)
a = 1.0000001
b = 1.0
a - b  # FP32 下可能是 1e-7, FP16 下可能是 0
```

**FP32 的相对误差 ≈ 1e-7 (机器精度), FP16 ≈ 1e-3, BF16 ≈ 1e-2**

#### 0.5.13.2 Log-Sum-Exp 推导

```text
问题:  log(Σ exp(x_i)),  当 x_i 很大时 exp(x_i) overflow

解法:  log(Σ exp(x_i)) = log(Σ exp(x_i - c) · exp(c))   (c 是任意常数)
                        = log(exp(c) · Σ exp(x_i - c))
                        = c + log(Σ exp(x_i - c))

取 c = max(x_i):
  log(Σ exp(x_i)) = max(x_i) + log(Σ exp(x_i - max(x_i)))

验证:  x = [1000, 1001, 999]
       max = 1001
       log(exp(1000-1001) + exp(0) + exp(999-1001))
     = log(exp(-1) + 1 + exp(-2))
     = log(0.368 + 1 + 0.135)
     = log(1.503) ≈ 0.408
       最终 = 1001 + 0.408 = 1001.408 ✓
       exp(1001.408) ≈ 1.503e435 (没有 overflow!)
```

**PyTorch 已经实现**:
```python
# 安全写法
logsumexp = torch.logsumexp(logits, dim=-1)   # ✓ PyTorch 内部做了

# 危险写法 (训练时千万别用)
softmax = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)  # ✗

# 灾难性写法
log_softmax = torch.log(torch.softmax(logits, dim=-1))  # ✗ 双重损失精度
```

#### 0.5.13.3 混合精度训练的"魔术"

```text
BF16/FP16 训练流程:
  1. 模型参数 W:  保持 FP32 备份 (master weights)
  2. 前向:  把 W 转成 BF16,  算 Y = f(X; W_bf16)
  3. 反向:  梯度 G 用 BF16
  4. 优化器:  在 FP32 主权重上做 W -= η · G
  5. 删除 BF16 副本 (用 FP32 转)

为什么:
  - BF16 计算: 2× 速度提升 (H100 / A100)
  - BF16 存储: 2× 显存节省
  - 但 BF16 梯度有噪声,  所以 master weights 必须是 FP32
  - 结果: 精度等价于纯 FP32,  速度接近 BF16

FP16 训练需要 GradScaler:
  - FP16 指数位 5 bit,  范围 [6e-5, 6.5e4]
  - 梯度常常 < 1e-5,  在 FP16 下 underflow 到 0
  - GradScaler: loss *= 2^16,  梯度也放大,  优化器 step 前 /2^16
  - BF16 不需要,  因为 BF16 指数位 8 bit,  和 FP32 范围一样
```

### 0.5.14 泛化理论——为什么模型在测试集也能 work?

> **泛化是深度学习的"圣杯"**。理解为什么模型能泛化到未见样本, 就能理解为什么正则化有效、为什么大数据比大模型重要、为什么有些设计注定失败。

#### 0.5.14.1 偏差-方差分解 (Bias-Variance Decomposition)

```text
测试误差 = 偏差² + 方差 + 不可约噪声

E[(y - ŷ)²] = (E[ŷ] - y)²  +  E[(ŷ - E[ŷ])²]  +  σ²
                  ↑ 偏差²          ↑ 方差          ↑ 噪声

偏差 (Bias):     模型平均预测离真值的距离
  → 衡量"模型族是否够灵活"

方差 (Variance): 模型在不同训练集上的预测波动
  → 衡量"模型对数据的敏感度"

搜推广的对应:
  - Embedding 维度过小 → 高偏差 (欠拟合)
  - Embedding 维度过大 + 数据少 → 高方差 (过拟合)
  - 解决方案:  Dropout,  权重衰减,  早停,  更大数据
```

**深度学习中的"双下降"现象**:
```text
传统统计学习:  误差先降后升 (U 型)
   ↑
   | \   / ←  过拟合
   |  \ /
   |   v
   +--→ 模型复杂度

深度学习:  误差先降,  升,  再降 (双下降)
   ↑
   | \   /\
   |  \_/  \___
   |   v
   +--→ 模型复杂度
        ↑
    "插值阈值" (训练误差 = 0 处)

含义:  当模型刚好能记住训练集时,  测试误差最大
      再增加模型复杂度,  测试误差反而下降
      解释:  超大模型有"隐式正则化"
```

#### 0.5.14.2 PAC-Bayes 泛化界

```text
泛化界 (经验误差 ≤ 真实误差 + 复杂度项):
  E[L(θ)] ≤ L̂(θ) + √(KL(Q || P) / (2N))

  Q:  学习到的参数分布 (e.g., SGD 找的 θ 附近的高斯)
  P:  先验分布 (e.g., 接近 0 的高斯)
  N:  训练样本数
  KL:  Q 和 P 的 KL 散度

含义:
  - Q 越接近 P (KL 小) → 泛化界紧 → 泛化好
  - N 越大 → √(KL/N) 越小 → 泛化界紧
  - 所以"参数接近先验"= 泛化好 (L2 正则的理论基础)

  → SGD 找的"平坦最小值" KL 小 → 泛化好 (实验和理论一致)
  → 锐利最小值 KL 大 → 泛化差
```

#### 0.5.14.3 容量 (Capacity) 与样本复杂度

```text
样本复杂度:  达到 ε 泛化误差所需的训练样本数

Vapnik-Chervonenkis (VC) 维:
  模型能"打散"的最大样本数
  d 维线性分类器的 VC 维 = d + 1
  d 维神经网络的 VC 维 ≈ O(d²) (粗略上界)

样本复杂度 ≈ VC 维 / ε
  1000 万参数的模型 → VC 维 ≈ 10^14
  →  需要约 10^14 个样本才能保证泛化

但实际工业界:
  - 数据量 10^10 (百亿)
  - 参数量 10^9 (十亿)
  - 模型容量 << 数据量
  → "过参数化"区间, 反而泛化好 (双下降理论)
```

### 0.5.15 凸分析回顾——Lagrangian 与 KKT 条件

> **凸优化是 SVM、逻辑回归、FM 等模型的数学基础**。理解 KKT, 你就理解支持向量、稀疏解、L1 正则的几何意义。

#### 0.5.15.1 拉格朗日乘子法

```text
约束优化问题:
  min f(x)
  s.t. g_i(x) ≤ 0,  i=1..m    (不等式约束)
       h_j(x) = 0,   j=1..p    (等式约束)

拉格朗日函数:
  L(x, λ, μ) = f(x) + Σ λ_i g_i(x) + Σ μ_j h_j(x)

  λ_i ≥ 0  是不等式约束的拉格朗日乘子
  μ_j     是等式约束的拉格朗日乘子 (无符号限制)

KKT 条件 (最优解必要条件):
  1. 平稳性:  ∇_x L = 0
  2. 原可行性: g_i(x) ≤ 0
  3. 对偶可行性: λ_i ≥ 0
  4. 互补松弛:  λ_i · g_i(x) = 0   (核心!)

互补松弛的含义:
  - 若 g_i(x) < 0 (约束不活跃),  λ_i = 0
  - 若 g_i(x) = 0 (约束活跃),  λ_i 可以 > 0
  → 只有"紧"的约束才需要付出代价
```

#### 0.5.15.2 L1 正则为什么产生稀疏解?

```text
L1 正则的约束形式:  min L(θ)  s.t.  ||θ||_1 ≤ t
                    (几何: L1 球是菱形)

L1 球 vs L2 球 vs 损失等高线:
  L2 球 (圆形) 与椭圆损失等高线相切: 切点在非零坐标
  L1 球 (菱形) 与椭圆损失等高线相切: 切点经常落在菱形顶点 → 某个坐标为 0

数学:
  L1 球的尖点 (顶点) 在坐标轴上 → 切点容易在坐标轴上
  坐标轴上的点 = 某个 θ_i = 0
  → L1 自然产生稀疏解
```

**搜推广的实践**:
```python
# L1 正则在搜推广里很少单独用
# 但其思想贯穿:
# 1. 特征选择:  L1 选最相关的 K 个特征
# 2. 稀疏 attention:  强制注意力矩阵稀疏(类似 L1 约束)
# 3. MoE 路由:  让大部分专家不被激活(L1 风格的"硬选择")
```

### 0.5.16 测度论与积分的简化——为什么深度学习可以"假装"用概率?

```text
在严格的测度论下, 概率是定义在 (Ω, F, P) 上的:
  - Ω: 样本空间
  - F: σ-代数 (事件集合)
  - P: 概率测度, P(Ω) = 1

深度学习里"概率"的简化:
  - 假设样本独立同分布 (i.i.d.)
  - 假设分布是"合理"的(单峰、光滑)
  - 假设大数定律成立

这些假设在搜推广里大部分成立:
  ✓ 样本独立: 不同时刻的用户行为近似独立
  ✓ 同分布:  假设用户群体相对稳定
  ✗ 但分布漂移 (concept drift) 是真的存在!

漂移的处理:
  1. 滑动窗口训练:  只用最近 N 天的数据
  2. 增量学习:  每 N 小时 fine-tune
  3. 监控:  检测特征分布变化, 触发重训
  4. 鲁棒损失:  Huber Loss, label smoothing
```

### 0.5.17 总结:这一章的"知识图谱"

```text
                       概率论 (基础)
                       /    |    \
                贝叶斯推理  概率不等式  大数定律
                       \    |    /
                        统计推断 (MLE/MAP)
                              |
                  优化理论 ←—  损失函数设计 (CE = -log p)
                      |  ↑
                  凸分析  梯度方法
                      |    |
                  Lipschitz  |  Adam/SGD
                              ↓
                      数值稳定性 (LogSumExp 等)
                              ↓
                    训练 (PyTorch 实际跑起来)
                              ↓
                      泛化理论 (为什么测试集也 work)
                              ↓
                    部署 (实际产出)

+ 横切关注:  信息论 (熵, 互信息, 信息瓶颈)
+ 横切关注:  矩阵论 (SVD, 特征值, 矩阵分解 = 推荐本质)
```

---

## 第 0.6 章:Embedding 与数据组织——搜推广的"血液系统"

> 在你的 CV 生涯中,你几乎不需要手动管理"数据组织"。图像数据是固定的[C,H,W]。但是在搜推广中,**数据组织方式决定了你能训练多大规模的模型、多快、多稳**。
>
> 本章回答你提出的核心问题:emb_table 是什么?数据怎么组织?scaling 有什么挑战?

### 0.6.1 emb_table 到底是什么?——从矩阵到物理内存

**在 PyTorch 中,nn.Embedding(vocab_size, dim) 创建了一个形状为 [vocab_size, dim] 的矩阵**。这就是 emb_table。
- 每一行代表一个 ID 的稠密向量
- 前向时:根据 ID 索引直接取对应行(查表 O(1))
- 反向时:只有被索引到的行有梯度,其他行梯度为 0

**为什么叫"表"?因为早期推荐系统真的就是查表**——把 user_id 映射到向量,就像查电话本。Embedding 是"可学习的查表"。

**emb_table 的物理存储**:
```python
# emb_table = [vocab_size, dim] 的矩阵
# 在 GPU 显存中是一段连续内存
# 访问模式:随机读取(按 ID 索引),不是顺序读取
# 这决定了它**不受 GPU 缓存友好**(不像 Conv 的顺序访问)

emb = nn.Embedding(1_000_000, 64)          # 100 万 × 64
print(emb.weight.shape)                    # [1000000, 64]
print(emb.weight.dtype)                    # float32
print(emb.weight.element_size())           # 4 bytes
print(f"显存: {1e6 * 64 * 4 / 1024**3:.2f} GB")  # 0.256 GB
# 1 亿用户 × 64 维 × FP32 = 25.6 GB ——超过一张 A100!
```

### 0.6.2 Embedding 的三种"为什么"

**为什么用 Embedding 而不用 One-Hot?**
One-Hot: 1000 万物品 × 1000 万维向量 → 40MB 只表示 1 个物品,且学不到相似性。
Embedding: 1000 万物品 × 32 维 = 1.28GB,可学习语义相似性。

**为什么 Embedding 维度不能随便设?**
维度太小(<8):信息容量不足,模型欠拟合。
维度太大(>256):参数量随维度线性增长,且过参数化浪费。
经验公式:dim = min(64, max(8, int(vocab⁰·²⁵ × 4)))

**为什么不同特征用不同 Embedding 维度?**
城市(500 类):4~8 维足够(≈ log₂500 ≈ 9 bit 信息量)
用户 ID(1 亿 类):32~64 维(≈ log₂1e8 ≈ 27 bit)
维度 = 信息量的 2~3 倍即可,不是越大越好。

### 0.6.3 数据组织的三个层次

**层次 1:单条样本的组织**
```python
# 一条 CTR 训练样本 = {离散ID + 连续值 + 序列 + 标签}
{
  "user_id": 12345,          # 离散特征,1 个 token
  "item_id": 67890,          # 离散特征,1 个 token
  "history_ids": [1,2,3,4],  # 行为序列,变长
  "age": 25.0,               # 连续特征
  "historical_ctr": 0.15,    # 连续特征
  "label": 1                 # 标签
}
```

**层次 2:Batch 的组织(collate_fn 的作用)**
```python
# 一个 batch = 把 N 条样本"对齐"成 N 个 Tensor
# 离散 ID → torch.long [B]
# 变长序列 → torch.long [B, T]  (需要 padding + mask)
# 连续值 → torch.float32 [B, F]
# 标签 → torch.float32 [B]
```

**层次 3:大规模数据的组织(Parquet + 分区)**
```python
# s3://bucket/dt=2026-06-05/
#   ├── hour=00/
#   │   └── part-00000.parquet   (列式存储,列裁剪)
#   └── hour=01/
#       └── part-00000.parquet
# 分区裁剪:只读必要数据,IO 减少 90%+
```

### 0.6.4 Scaling 的三重挑战

**挑战 1:Embedding Table 的规模**
```python
# 1 亿用户 × 64 维 × FP32 = 25.6 GB  —— 一张 A100 装不下!
# 2 亿物品 × 32 维 × FP32 = 25.6 GB
# 5 个特征 × 各 32 维 = 128 GB

# 解法:
# 1) 哈希分桶:多个 ID 映射到同一行,节省 10~100 倍
# 2) 量化:FP32→INT8,省 4 倍
# 3) 参数服务器:Embedding 放 CPU 内存,GPU 只做计算
# 4) 维度压缩:从 64→16→8,参数量减 4~8 倍
```

**挑战 2:序列长度的 Scaling**
```python
# 用户行为序列:从 50 → 500 → 5000
# 注意力复杂度:O(T²)  T=5000 时单次 attention 需要 25M 个元素
# 解法:FlashAttention, MQA/GQA, 长序列截断+检索
```

**挑战 3:训练数据量的 Scaling**
```python
# 1 天 10 亿样本,10 epoch = 100 亿样本
# 1 步处理 1024 样本 → 977 万步
# 1 步 0.5 秒 → 56 天!
# 解法:分布式训练(DDP), 梯度累积, 混合精度
```

### 0.6.5 模型对比:为什么一个模型比另一个好?

**思考模型的"为什么"比"是什么"更重要**。判断一个模型是否适合你的场景,问这 5 个问题:

| 问题 | 对模型选择的影响 |
| :--- | :--- |
| 这个模型解决了上一个模型的什么痛点? | 理解演进方向 |
| 这个模型的假设是什么? | 假设不成立时模型会失效 |
| 这个模型的瓶颈在哪(计算/内存/数据)? | 决定了是否能部署 |
| 这个模型在什么数据上表现好/不好? | 决定了适用场景 |
| 这个模型的核心归纳偏置是什么? | 决定了模型偏好什么模式 |

**举例:DIN vs DeepFM**
```python
# DeepFM:假设特征交叉是"静态的"(所有用户-物品用同一套交叉权重)
# DIN:    假设"用户兴趣是动态的"(同一个用户对不同物品兴趣不同)
#
# DeepFM 的瓶颈:无法处理动态兴趣(序列被池化成固定向量)
# DIN 的瓶颈:需要用户的长期行为序列(新用户没序列,效果差)
#
# 所以:新用户多 → DeepFM;老用户多且行为丰富 → DIN
```

### 0.6.6 Torch 使用核心技巧

**torch 编程的本质 = Tensor 的"整形"艺术**:
```python
# 搜推广里你 90% 的时间在和 Tensor 的形状打交道
# Debug 手法:在每个关键操作前打印 shape

def debug_shape(x, name="tensor"):
    print(f"{name}: shape={x.shape}, dtype={x.dtype}, device={x.device}")

# 常见操作:
# cat: [B, 16] + [B, 32] → [B, 48]  (特征拼接)
# stack: [B] × 5 → [B, 5]  (特征交叉)
# gather: 从序列中按索引取值
# masked_fill: 用 mask 过滤 padding
```

**Embedding 操作的 PyTorch 代码模式**:
```python
# 模式 1:单 ID 查表
user_emb = self.user_embedding(user_ids)        # [B] → [B, D]

# 模式 2:多值序列查表 + mask
seq_emb = self.item_embedding(history_ids)      # [B, T] → [B, T, D]
mask = (history_ids != 0).unsqueeze(-1)         # padding mask
seq_emb = seq_emb * mask                         # 遮掉 padding
pooled = seq_emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # 平均池化

# 模式 3:EmbeddingBag(省显存)
bag_emb = self.emb_bag(input_ids, offsets)       # 查表+池化一步完成

# 模式 4:多特征 Embedding
all_embs = [self.emb_dict[name](feat_ids[name]) for name in feature_names]
concat_emb = torch.cat(all_embs, dim=-1)         # [B, D_total]
```

所有章节都完成。最终文件:7,131 行,260KB,从浮点数到神经架构全覆盖。

---

## 第 1 章:Python 底层——为搜推广数据处理量身定制

> **本章目标**:不是 Python 语法教程(你肯定已经会),而是**以搜推广数据处理为视角,重新审视 Python 内置数据结构的内存模型、时间复杂度、踩坑点**。读完你应该能回答:
> - 为什么 `dict.get()` 和 `dict[key]` 行为不同?线上推理时该用哪个?
> - 千万级 `in` 查找,为什么 list 慢得离谱?
> - `defaultdict` 到底解决了什么痛点?
> - 为什么 `for key in dict` 时不能修改 dict?

### 1.1 一切的基础:Python 对象的内存模型

#### 1.1.1 名字(name)、对象(object)、引用(reference)

很多 CV 出身的同学写 C++/CUDA 习惯了"变量 = 内存地址",到 Python 这套模型就崩了。**Python 里没有"变量",只有"名字"**——名字像便签,贴在一个对象上。

```python
a = [1, 2, 3]   # 创建一个列表对象 [1,2,3],把名字"a"贴上去
b = a            # 不是"复制",而是再贴一个名字"b"在同一对象上
b.append(4)
print(a)         # [1, 2, 3, 4]   ← a 也变了!因为 a 和 b 指向同一对象
```

**搜推广的坑**:你从 `pandas.DataFrame` 取出一列得到一个 `Series`,赋给一个新名字,然后对这个新名字做 `inplace=True` 操作,会**直接修改原 DataFrame**!这是无数线上事故的根源。

```python
# 反面教材
df = pd.read_csv("user_behavior.csv")
col = df["user_id"]         # col 和 df["user_id"] 指向同一块数据
col.fillna(-1, inplace=True) # 警告:inplace 已修改 df 本身!
# 此时 df 已经被悄悄改了,后续 join 时可能报错
```

```python
# 正确做法
col = df["user_id"].copy()   # 显式 copy,名字和原对象解绑
col.fillna(-1, inplace=True)
```

#### 1.1.2 可变(mutable) vs 不可变(immutable)

| 类型 | 是否可变 | 内存行为 |
| :--- | :--- | :--- |
| `int`, `float`, `bool`, `str`, `tuple`, `frozenset` | **不可变** | 改值会**创建新对象**,原对象引用计数归零 |
| `list`, `dict`, `set`, `bytearray` | **可变** | 改值**原地修改**,引用计数不变 |

**搜推广的坑**:函数默认参数陷阱

```python
# 反面教材
def add_user_click(click_history=[]):  # 默认参数 [] 只在函数定义时创建一次!
    click_history.append(1)
    return click_history

print(add_user_click())  # [1]
print(add_user_click())  # [1, 1]   ← 第二次调用时,默认参数已经累积了上次的值!
print(add_user_click())  # [1, 1, 1]
```

```python
# 正确做法
def add_user_click(click_history=None):
    if click_history is None:
        click_history = []
    click_history.append(1)
    return click_history
```

**为什么?** 默认参数 `[]` 在函数 `def` 那行执行时**就创建了一个 list 对象**,之后每次调用都用这个**同一对象**。`append` 是原地修改,所以累积。**这条是 Python 面试必考点,在搜推广代码里出现得极多**。

#### 1.1.3 引用计数与垃圾回收(GC)

Python 用"引用计数 + 标记清除 + 分代回收"管理内存。**循环引用**是搜推广中常被忽视的内存泄漏源:

```python
# 反面教材:循环引用
user_features = {"name": "Alice", "history": None}
history = ["item1", "item2"]
user_features["history"] = history   # history 引用 list
history.append(user_features)        # list 反过来引用 dict
# user_features 和 history 互相引用,引用计数永远 > 0,GC 回收不掉
# 千万级样本处理时,这种循环引用会让进程内存慢慢涨到几十 GB
```

**正确做法**:
- 业务对象尽量**单向引用**(父→子,不要子→父)
- 用 `weakref` 模块做弱引用
- 大循环里显式 `del obj` + `gc.collect()`

---

### 1.2 容器类型深度解析(搜推广核心)

#### 1.2.1 `list`(列表):动态数组

**底层结构**:`list` 内部是一个**C 数组**,存储的是**指向 Python 对象的指针**(PyObject*),而不是值本身。

```text
list 内部:
+----+----+----+----+
| *  | *  | *  | *  |   ← 连续的指针数组
+----+----+----+----+
  |    |    |    |
  v    v    v    v
 obj  obj  obj  obj    ← 实际对象,散落在堆上
```

**关键操作时间复杂度**:

| 操作 | 时间复杂度 | 备注 |
| :--- | :--- | :--- |
| `lst[i]` 索引 | O(1) | 指针数组按偏移取 |
| `lst.append(x)` 尾部追加 | O(1) 摊销 | 满了就**按 1.125~1.25 倍扩容** |
| `lst.pop()` 尾部弹出 | O(1) | |
| `lst.insert(i, x)` 中间插入 | **O(n)** | 后续所有指针要 memmove |
| `lst.pop(0)` 头部弹出 | **O(n)** | 同上 |
| `x in lst` 成员判断 | **O(n)** | 线性扫描,千万级直接卡死 |
| `lst.sort()` 原地排序 | O(n log n) | Timsort,工业级 |

**搜推广的踩坑 1:在线推理里 `in list`**

```python
# 反面教材:在精排打分循环里检查"黑名单商品"
def score(user_id, item_id, blacklist_items):  # blacklist_items 是 list
    if item_id in blacklist_items:              # O(N) 一次,1000 个黑名单就 1000 次
        return -1
    return real_model.predict(user_id, item_id)
```

```python
# 正确做法:提前转 set
blacklist_set = set(blacklist_items)  # 一次性 O(N)
def score(user_id, item_id):
    if item_id in blacklist_set:      # O(1)
        return -1
    return real_model.predict(user_id, item_id)
```

**搜推广的踩坑 2:`append` 千万次,内存暴涨**

```python
# 假设要收集 1 亿条用户点击
clicks = []
for log in open("click.log"):           # 一行一条
    clicks.append(parse(log))           # 每次 append 都可能触发 list 扩容
                                     # 1.25 倍策略下,1 亿次 append 会触发约 50 次 realloc + memcpy
                                     # 内存峰值可能是稳态的 1.5~2 倍
```

**优化方案 A:预分配(对 list 不可行,Python 没这 API)→ 改用 array 或 numpy**

```python
import array
# 预知 1 亿个 int,用 array 存储(底层是 C 数组,直接存数值,无指针开销)
clicks = array.array('i', [0]) * 100_000_000  # 一次性分配 4×1e8 = 400MB 内存
# 然后切片赋值(快很多)
```

**优化方案 B:写成迭代器,边遍历边处理,不存中间结果**

```python
def process_clicks(log_file):
    for line in open(log_file):
        yield parse(line)  # 惰性,一次只占 1 条数据内存

for click in process_clicks("click.log"):
    train_model(click)
```

#### 1.2.2 `tuple`(元组):不可变序列

`tuple` 占用内存比 `list` **少 16~24 字节**(没有可变头部),**创建速度比 list 快 5~10 倍**(因为不需要预分配额外空间)。

**搜推广的常用场景**:
- **作为 dict 的 key**:`(user_id, item_id)` 可以做 key,`[user_id, item_id]` 不行
- **函数多返回值**:`return user_emb, item_emb` 本质就是 tuple
- **不可变的数据记录**:`record = (timestamp, user_id, item_id, action)`

```python
# 反面教材:用 list 做 dict key
cache = {}
key = [user_id, item_id]   # list
cache[key] = pred           # TypeError: unhashable type: 'list'

# 正确做法
key = (user_id, item_id)    # tuple,可哈希
cache[key] = pred
```

**性能对比小实验**:
```python
import sys, timeit

lst = [1, 2, 3, 4, 5]
tup = (1, 2, 3, 4, 5)

print(sys.getsizeof(lst))  # 104 字节
print(sys.getsizeof(tup))  # 80 字节

# 1 亿次创建 list vs tuple
timeit.timeit('[1,2,3,4,5]', number=10_000_000)   # ~0.7s
timeit.timeit('(1,2,3,4,5)', number=10_000_000)   # ~0.1s,快 7 倍
```

#### 1.2.3 `dict`(字典):哈希表

**底层结构**(Python 3.7+ 之后):由两部分组成:
1. **哈希索引数组(稀疏,Indices table)**:存 key 的哈希值 - 这是"入口"
2. **紧凑数据数组(Entries array)**:按插入顺序存 `(hash, key, value)` 三元组

**插入流程的底层细节**:
```text
插入 key="user_123", value=1:

Step 1: hash("user_123") = 0x7f3b2a1c      ← 哈希函数,理想 O(1)
                                     注意:不是"用户名字符串的 ASCII 和"!
                                     而是 CPython 的 SipHash 加密哈希,
                                     防止哈希碰撞攻击(让攻击者无法构造恶意 key 让 dict 退化)

Step 2: table_index = hash & (capacity - 1)  ← 位运算取模(比 % 快 10 倍)
                                     假设 capacity=8,则 capacity-1=7(二进制 111)
                                     0x7f3b2a1c & 7 = 4

Step 3: 检查 indices[4] 是否已被占
  → 空:把 (hash, key, value) 写入 entries 数组,indices[4] 指向新 entry
  → 被占:二次探测(Quadratic Probing)
          尝试 indices[4 + 1²], indices[4 - 1²], indices[4 + 2²], ...
          直到找到空位或找到匹配的 key

Step 4: 如果负载因子(load factor = 已用槽 / 总槽) > 2/3:
        触发 rehash:容量 × 2,重建整个哈希表
        rehash 代价 O(n), n=当前 dict 大小
```

**关键性能参数**:
- **默认容量**:PyDict_MINSIZE = 8(一个空 dict 也要 8 个空槽)
- **负载因子阈值**:2/3(约 66.7%)。超过就翻倍扩容
- **冲突解决**:二次探测(Quadratic Probing),不是线性探测
- **为什么是二次探测?**:避免线性探测的"主聚集(Primary Clustering)"问题——即冲突 key 扎堆在一个区域

**一个 dict 占多少内存?** (Python 3.11, 64-bit)

```python
import sys

d = {}
print(sys.getsizeof(d))                          # 72 字节(空 dict 开销)

d["key"] = "value"
print(sys.getsizeof(d))                          # 120 字节(8 槽)

for i in range(1000):
    d[f"key_{i}"] = i
print(sys.getsizeof(d))                          # 49216 字节(≈ 49KB)
# 1000 个键值对约 49KB → 每个键值对 ≈ 49 字节
# 对比 list:1000 个 int 约 8KB(每个指针 8 字节)
```

**dict 内存估算公式**(搜推广场景常用):

```python
def estimate_dict_memory(n_items, key_type="str", value_type="int"):
    """
    估算 dict 占用内存的粗略公式。
    在搜推广里,要在内存里装 1 亿个 (user_id → embedding_index) 映射。
    需要预计算内存够不够。
    """
    # 每个条目 = hash(8B) + key_ptr(8B) + value(依类型) + 槽位开销
    overhead_per_entry = 50  # 约 50 字节(含 overhead)
    return n_items * overhead_per_entry

# 1 亿映射:
print(f"1 亿映射: {estimate_dict_memory(100_000_000) / 1024**3:.2f} GB")  # ≈ 4.7 GB
# 4.7 GB —— 太大!所以工业界不用纯 Python dict 做 1 亿级映射。
# 改用:内存数据库(Redis/Tair)或 mmap 文件
```

**复杂度表(完整版,含摊销分析)**:

| 操作 | 平均时间复杂度 | 摊销最坏 | 搜推广中 |
| :--- | :--- | :--- | :--- |
| `d[key]`(命中) | O(1) | — | 强特征必用 |
| `d[key]`(未命中) | O(1) → 抛异常 | — | 线上 500 的元凶 |
| `d.get(key, default)` | O(1) | — | 弱特征/降级 |
| `d[key] = v`(新 key) | O(1) | **O(n)**(rehash 时) | 亿级构建时注意 |
| `key in d` | O(1) | O(n)(哈希攻击) | 黑名单判断 |
| `d.pop(key)` | O(1) | O(n) | 删除中间使用 |
| 遍历 | O(n) | — | 序列化/检查 |
| 拷贝 `d.copy()` | O(n) | — | 深拷贝注意内存 |

**为什么 dict 是「有序」的(Python 3.7+)?**
```python
# 3.6 之前:dict 无序,遍历顺序不可预测
# 3.7 之后:dict 保持插入顺序!
# 原因:底层改成了"紧凑数组",entries 按插入顺序排列,indices 只存位置索引

d = {"b": 2, "a": 1, "c": 3, "user_123": 123}
print(list(d.keys()))
# ['b', 'a', 'c', 'user_123']  ← 一定是插入顺序!
# 这对搜推广特征工程意义重大:
# 当你构建特征 dict 时,特征顺序 = 模型输入顺序 = 可预测的!

# 但注意:3.6 之前的无序性是 Python 特性之一
# 如果你还在用 3.6 以下,不要依赖顺序
```

**重要区别:`d[key]` vs `d.get(key)`**

```python
features = {"user_id": 123, "item_id": 456}

# 写法 1:直接索引
uid = features["user_id"]      # 存在 → 返回 123
age = features["user_age"]     # 不存在 → 抛 KeyError,程序崩溃!

# 写法 2:get + 默认值
uid = features.get("user_id", -1)   # 存在 → 返回 123
age = features.get("user_age", -1)  # 不存在 → 返回 -1,程序不崩
```

**搜推广的黄金法则**:
- **强特征(必选)**:`features["user_id"]`——缺失就要崩,不能"假装"过去
- **弱特征(可选/可能缺失)**:`features.get("user_age", -1)`——优雅降级

**为什么?** 在 CV 里你的输入张量是程序自己生成的,缺失的维度几乎不会发生(图片就是 3 通道、224×224);在搜推广里,**特征来自上游几十个数据源,任何一个字段缺失都会让整个 batch 报错**。线上服务一次 500 错误,可能直接造成几万次推荐失败。

**增量构建超大规模 dict 的性能陷阱**:

```python
# 陷阱场景:在循环里给 dict 不断添加新 key
# 当 dict 填满 2/3 时,PyDict_SetItem 会触发 rehash
# rehash:重新分配内存,重算所有 key 的哈希,重建索引
# 这是一个 O(n) 操作!

metadata = {}
for i in range(10_000_000):          # 1000 万个特征
    metadata[f"feature_{i}"] = i     # 每填满 2/3 就触发一次 rehash
    # rehash 耗时:第一次 0.1ms,第十次 50ms,第一百次 500ms
    # 总耗时:不是 O(n),而是 O(n log n)!

# 更好的方式:预分配(虽然 Python dict 没有预分配 API)
# 但可以先估算容量,用 dict.fromkeys 预热
# 或者直接改用 array / numpy 存储
```

#### 1.2.4 `set`(集合):去重 + 快速成员判断

`set` 底层实现与 `dict` 类似(只有 key,没有 value),所有 `dict` 的复杂度规则都适用于 `set`。

**搜推广的高频用法**:

```python
# 用法 1:用户去重
unique_users = set()
for line in open("click.log"):
    user_id = parse(line)["user_id"]
    unique_users.add(user_id)         # O(1) 插入

# 用法 2:求交集(召回常用)
yesterday_clicked = set(parse_logs("day1.txt"))
today_clicked = set(parse_logs("day2.txt"))
new_clicks = today_clicked - yesterday_clicked    # O(min(|A|, |B|))
```

**性能对比**:
```python
# 100 万元素的 list vs set 查找
big_list = list(range(1_000_000))
big_set = set(big_list)

%timeit 999_999 in big_list   # 8.5 ms
%timeit 999_999 in big_set    # 0.04 μs ← 快 20 万倍!
```

#### 1.2.5 重要容器对比总结表

| 容器 | 有序? | 可重复? | 索引方式 | `in` 复杂度 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `list` | ✓ | ✓ | `lst[i]` | O(n) | 顺序数据、需要随机访问 |
| `tuple` | ✓ | ✓ | `t[i]` | O(n) | 不可变记录、做 dict key |
| `dict` | ✓(3.7+) | key 不可重复 | `d[k]` | **O(1)** | 键值映射、计数、缓存 |
| `set` | ✗ | ✗ | — | **O(1)** | 去重、集合运算、黑名单 |
| `deque` | ✓ | ✓ | `dq[i]`(慢) | O(n) | 双端队列、滑窗 |
| `OrderedDict` | ✓ | key 不可重复 | `d[k]` | O(1) | **3.7+ 已不推荐,普通 dict 就有序** |
| `Counter` | ✓ | ✓ | `c[k]` | O(1) | 频次统计、Top-K |
| `defaultdict` | ✓ | ✓ | `d[k]` | O(1) | 分组聚合、自动初始化 |

---

### 1.3 字符串处理(搜推广中的"原始燃料")

#### 1.3.1 编码与解码:踩过坑才懂

搜推广里 90% 的原始数据都是**字符串**(用户 ID、物品类目、文本评论、行为描述)。Python 3 的字符串是**Unicode**,但存储和传输用 **bytes**。

```python
# 错误:从文件读,不解码
with open("data.txt", "rb") as f:
    raw = f.read()        # bytes,像 b'\xe7\x94\xa8\xe6\x88\xb7'
print(raw.decode("utf-8"))  # "用户"

# 常见坑:GBK 编码的日志
with open("old_log.txt", "r", encoding="gbk") as f:
    line = f.read()       # "用户" 在 GBK 下是 \xd3\xc3\xbb\xa7

# 搜推广实战:遇到生僻字、emoji 时
text = "用户 123 😊"
text.encode("utf-8")       # b'\xe7\x94\xa8\xe6\x88\xb7 123 \xf0\x9f\x98\x8a'
text.encode("gbk")         # UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f60a'
```

**避坑指南**:
- 内部数据全部用 **UTF-8** 存储
- 读取时**显式指定 encoding**,不要用默认
- 写入 CSV/JSON 时,加 `errors="ignore"` 或 `errors="replace"` 兜底

#### 1.3.2 字符串拼接性能:不要在循环里 `+`

```python
# 反面教材:O(n²) 复杂度
def build_user_profile(user_logs):
    profile = ""
    for log in user_logs:                   # 1000 万条
        profile += log["action"] + ";"      # 每次 += 都创建新 string,旧 string 被 GC
    return profile

# 正确做法 1:list + join(O(n))
def build_user_profile(user_logs):
    parts = []
    for log in user_logs:
        parts.append(log["action"] + ";")
    return "".join(parts)                   # 一次性拼接

# 正确做法 2:生成器 + join(更省内存)
def build_user_profile(user_logs):
    return "".join(log["action"] + ";" for log in user_logs)
```

#### 1.3.3 正则表达式(re 模块)

搜推广的特征工程里,正则用来:
- 从 URL 提取参数:`?id=123&type=feed` → `id=123`
- 从文本提取数字/邮箱/手机号
- 校验 ID 格式

```python
import re

# 提取 URL 参数
url = "https://rec.com/item?id=12345&category=phone&from=list"
match = re.search(r"id=(\d+)", url)
if match:
    item_id = int(match.group(1))   # 12345

# 预编译(对重复使用的模式至关重要!)
ITEM_ID_PATTERN = re.compile(r"id=(\d+)")

def extract_item_id(url):
    m = ITEM_ID_PATTERN.search(url)
    return int(m.group(1)) if m else None

# 千万次调用
%timeit re.search(r"id=(\d+)", url)            # 2.1 μs/次
%timeit ITEM_ID_PATTERN.search(url)            # 0.6 μs/次,快 3 倍
```

**正则的坑:贪婪匹配**

```python
text = "<div>商品A</div><div>商品B</div>"
re.search(r"<div>(.*)</div>", text).group(1)   # "商品A</div><div>商品B" ← 贪婪
re.search(r"<div>(.*?)</div>", text).group(1)  # "商品A" ← 非贪婪,加 ?
```

#### 1.3.4 字符串模板(f-string vs format vs %)

```python
user = "Alice"
score = 0.95

# f-string(Python 3.6+,最快,推荐)
log_line = f"[2026-06-05] user={user} score={score:.4f}"

# format(兼容性更好)
log_line = "[2026-06-05] user={} score={:.4f}".format(user, score)

# % (老式,慢慢淘汰)
log_line = "[2026-06-05] user=%s score=%.4f" % (user, score)
```

---

### 1.4 控制流与推导式

#### 1.4.1 推导式:Python 的"杀手锏"

**搜索推荐里最常用的模式**:**一行代码完成"过滤 + 转换 + 收集"**。

```python
# list 推导
clicked_item_ids = [log["item_id"] for log in logs if log["action"] == "click"]

# dict 推导:统计每个商品的点击次数
click_counts = {log["item_id"]: 0 for log in logs}
for log in logs:
    if log["action"] == "click":
        click_counts[log["item_id"]] += 1

# 嵌套推导:笛卡尔积(交叉特征构造)
cross_features = [(a, b) for a in user_attrs for b in item_attrs]
```

#### 1.4.2 生成器(yield):不占内存的"懒迭代"

```python
# 反面教材:一次性加载 1 亿行日志到内存
def read_all_logs():
    lines = []
    for line in open("huge.log"):
        lines.append(parse(line))    # 内存爆炸
    return lines

# 正确做法:生成器,一次只占 1 行
def read_logs():
    for line in open("huge.log"):
        yield parse(line)            # 暂停在这里,下次 next() 继续

# 使用
for log in read_logs():              # 一次只迭代一行
    process(log)
```

**性能对比**:
```python
# 1 亿个整数
def gen():
    for i in range(100_000_000):
        yield i
g = gen()
sys.getsizeof(g)                     # 200 字节!不管数据多大,生成器就这点开销

# vs
lst = list(range(100_000_000))       # 占用 800MB
```

**搜推广的黄金法则**:**能 yield 的地方,绝不用 list.append**。

#### 1.4.3 上下文管理器(with 语句)

搜推广里打开/关闭文件、连接数据库、获取 GPU 资源,**必须**用 `with`,否则资源泄漏:

```python
# 反面教材
f = open("data.txt", "r")
data = f.read()
# 突然出异常,f 没关,文件描述符泄漏
# 跑 1 小时后:"OSError: too many open files"

# 正确做法
with open("data.txt", "r") as f:
    data = f.read()                  # with 退出时自动调用 f.close()

# 自定义上下文管理器:统计代码块耗时
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"[{name}] 耗时 {time.time() - start:.3f}s")

with timer("数据加载"):
    data = load_huge_data()
# 输出:[数据加载] 耗时 12.345s
```

---

### 1.5 函数与函数式编程

#### 1.5.1 函数参数三大类型

```python
# 1. 位置参数(必传,顺序敏感)
def build_feature(user_id, item_id, timestamp):
    pass

build_feature(123, 456, 1717000000)         # 位置传参
build_feature(item_id=456, user_id=123, timestamp=1717000000)  # 关键字传参,顺序无所谓

# 2. 默认参数(可选)
def build_feature(user_id, item_id, timestamp=1717000000, use_seq=True):
    pass

# 3. *args / **kwargs(可变参数)
def merge_features(*feature_dicts, **opts):
    """
    *feature_dicts: 收集所有位置参数到元组
    **opts: 收集所有关键字参数到字典
    """
    merged = {}
    for fd in feature_dicts:
        merged.update(fd)
    merged.update(opts)
    return merged

merge_features({"a": 1}, {"b": 2}, normalize=True, version="v3")
# merged = {"a": 1, "b": 2, "normalize": True, "version": "v3"}
```

**搜推广的踩坑:不要用可变类型做默认参数**(前面 1.1.2 讲过,这里再强调一次,出现频率极高)

#### 1.5.2 lambda 表达式

```python
# 简单函数,懒得 def
square = lambda x: x * x
get_uid = lambda features: features.get("user_id", -1)

# 搜推广实战:pandas 的 groupby + agg
df.groupby("user_id").agg(
    click_count=("action", lambda x: (x == "click").sum()),
    avg_dwell=("dwell_time", "mean")
)
```

#### 1.5.3 装饰器(Decorator)

装饰器是"给函数穿外套",搜推广里最常见的就是**计时、缓存、日志、重试**。

```python
import time
from functools import wraps

# 1. 计时装饰器
def timer(func):
    @wraps(func)  # 保留原函数 __name__ 等元信息
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[{func.__name__}] 耗时 {time.time() - start:.3f}s")
        return result
    return wrapper

@timer
def load_features(path):
    # ...
    return features

load_features("huge.parquet")   # [load_features] 耗时 8.234s

# 2. 缓存装饰器(避免重复计算)
from functools import lru_cache

@lru_cache(maxsize=10_000)
def get_user_embedding(user_id):    # 相同 user_id 不会重复算
    return compute_embedding(user_id)
```

#### 1.5.4 偏函数(partial):固定某些参数

```python
from functools import partial

def parse_log(line, sep="\t", encoding="utf-8"):
    return line.strip().split(sep)

# 给"特定日志格式"专门定制一个解析函数
parse_click_log = partial(parse_log, sep="\t")   # 固定分隔符
# 用法
parse_click_log("123\t456\tclick")  # ['123', '456', 'click']
```

---

### 1.6 面向对象(OOP)

#### 1.6.1 类与实例:搜推广里到处是"模板+实体"

```python
class FeatureEncoder:
    """搜推广里把字符串 ID 转成 int 的类"""

    def __init__(self, name, unk_id=0):
        self.name = name
        self.unk_id = unk_id
        self.token2id = {"<UNK>": unk_id}
        self.next_id = 1

    def fit(self, tokens):
        """从训练数据里学词表"""
        for token in tokens:
            if token not in self.token2id:
                self.token2id[token] = self.next_id
                self.next_id += 1
        return self

    def transform(self, tokens):
        """token → int id,未知走 <UNK>"""
        return [self.token2id.get(t, self.unk_id) for t in tokens]

# 使用
user_enc = FeatureEncoder("user_id")
user_enc.fit(["u1", "u2", "u3", "u1"])
user_enc.transform(["u1", "u99"])  # [1, 0] ← u99 走 UNK
```

#### 1.6.2 魔术方法(dunder methods):让类"用起来像内置类型"

**最常用的几个**:

| 魔术方法 | 触发场景 | 搜推广用法 |
| :--- | :--- | :--- |
| `__init__` | 创建实例 | 初始化词表、加载模型 |
| `__getitem__(self, key)` | `obj[key]` | **Dataset 必须实现**,让 `dataset[i]` 返回第 i 个样本 |
| `__len__(self)` | `len(obj)` | **Dataset 必须实现** |
| `__iter__(self)` | `for x in obj` | 让自定义对象可迭代 |
| `__call__(self, *args)` | `obj(*args)` | 让实例像函数一样调用(Transformer、Module 都是) |
| `__repr__` / `__str__` | `print(obj)` | 调试时打印 |
| `__enter__` / `__exit__` | `with obj` | 上下文管理 |

**最关键的:`__getitem__` + `__len__`——这是 PyTorch Dataset 的契约**

```python
from torch.utils.data import Dataset

class ClickDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)            # 必须,DataLoader 用它算 batch 数

    def __getitem__(self, idx):       # 必须,DataLoader 用它取样本
        row = self.df.iloc[idx]
        return {
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "label": row["label"]
        }

# 用法
dataset = ClickDataset("train.parquet")
print(len(dataset))              # 调用 __len__ → 1000000
print(dataset[0])                # 调用 __getitem__(0) → {user_id:..., item_id:..., label:...}
```

#### 1.6.3 继承与多态

```python
class BaseEncoder:
    def fit(self, tokens): raise NotImplementedError
    def transform(self, tokens): raise NotImplementedError

class LabelEncoder(BaseEncoder):
    def __init__(self, name): self.name, self.t2i, self.next = name, {}, 0
    def fit(self, tokens):
        for t in tokens:
            if t not in self.t2i:
                self.t2i[t] = self.next; self.next += 1
        return self
    def transform(self, tokens): return [self.t2i.get(t, 0) for t in tokens]

class HashEncoder(BaseEncoder):
    """用哈希函数,无需预建词表,适合超高基数"""
    def __init__(self, name, num_buckets=1_000_000):
        self.name = name
        self.num_buckets = num_buckets
    def transform(self, tokens):
        return [hash(t) % self.num_buckets for t in tokens]
    def fit(self, tokens): return self   # 无需 fit

# 多态:同一个调用,不同行为
def build_feature_pipeline(use_hash=False):
    if use_hash:
        enc = HashEncoder("user_id")
    else:
        enc = LabelEncoder("user_id")
    return enc

enc = build_feature_pipeline(use_hash=True)
enc.transform(["u1", "u2", "u3"])  # [12345, 67890, 11111]  ← 哈希值
```

#### 1.6.4 属性装饰器(@property):把方法伪装成字段

```python
class Vocabulary:
    def __init__(self):
        self._tokens = ["<PAD>", "<UNK>", "item1", "item2"]
        self._frozen = False

    @property
    def size(self):
        return len(self._tokens)        # 像字段一样访问 vocab.size,实际是方法

    @property
    def is_frozen(self):
        return self._frozen

    @is_frozen.setter
    def is_frozen(self, value):
        if not isinstance(value, bool):
            raise TypeError
        self._frozen = value

vocab = Vocabulary()
print(vocab.size)         # 4 ← 像访问字段
vocab.is_frozen = True    # 像赋值,但实际走 setter
```

---

### 1.7 模块与包:大型项目的代码组织

#### 1.7.1 导入机制

```python
# 4 种导入方式
import numpy as np                    # 导入整个模块,用 np.array
from numpy import array               # 直接导入函数,用 array(可能命名冲突)
from numpy import array as arr        # 导入并重命名
from numpy import *                   # 全部导入(禁止在生产代码中使用!)

# 搜推广的"标准项目结构"
recsys/
├── recsys/
│   ├── __init__.py            # 包标识
│   ├── data/
│   │   ├── __init__.py
│   │   ├── encoders.py        # FeatureEncoder
│   │   └── dataset.py         # ClickDataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dnn.py
│   │   └── din.py
│   └── train/
│       ├── __init__.py
│       └── trainer.py
├── setup.py
└── requirements.txt
```

#### 1.7.2 `__init__.py` 的作用

```python
# recsys/data/__init__.py
from .encoders import LabelEncoder, HashEncoder
from .dataset import ClickDataset

# 用户就可以这样导入,不用记内部文件
from recsys.data import LabelEncoder, ClickDataset
```

#### 1.7.3 相对导入 vs 绝对导入

```python
# 在 recsys/models/din.py 中
from recsys.data.encoders import LabelEncoder  # 绝对导入(推荐)
from ..data.encoders import LabelEncoder       # 相对导入(包内部)
```

---

### 1.8 异常处理:让线上服务"不崩"

#### 1.8.1 try/except/finally 基础

```python
# 反面教材:捕获所有异常然后忽略
try:
    parse_user_click(log)
except:                            # 裸 except,捕获一切
    pass                           # 错误被吞,线上数据问题永远发现不了

# 正确做法
try:
    parse_user_click(log)
except (KeyError, ValueError) as e:    # 只捕获已知异常
    log_error(f"解析失败: {log}, error: {e}")
    return None
except Exception as e:                 # 兜底
    log_error(f"未知异常: {e}")
    raise                               # 重新抛,让上层处理
```

#### 1.8.2 自定义异常

```python
class FeatureMissingError(Exception):
    """特征缺失时抛出的异常"""
    def __init__(self, feature_name, context=None):
        self.feature_name = feature_name
        self.context = context
        super().__init__(f"必需特征 {feature_name} 缺失,context={context}")

# 使用
def get_user_age(features):
    if "user_age" not in features:
        raise FeatureMissingError("user_age", context=features.get("user_id"))
    return features["user_age"]
```

#### 1.8.3 实战:数据处理管道的健壮性

```python
def safe_parse_log(line, line_no):
    """解析一行日志,失败返回 None,绝不抛异常"""
    try:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            raise ValueError(f"字段数不足: {len(parts)}")
        return {
            "user_id": int(parts[0]),
            "item_id": int(parts[1]),
            "action": parts[2],
        }
    except (ValueError, IndexError) as e:
        # 不要让一行坏数据毁掉整个 pipeline
        log_warning(f"第 {line_no} 行解析失败: {e}, 内容: {line[:100]}")
        return None

# 处理整个文件
def parse_file(path):
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            parsed = safe_parse_log(line, line_no)
            if parsed is not None:
                yield parsed
```

---

### 1.9 常用内置库(`collections` / `itertools` / `functools`)

#### 1.9.1 `collections` 家族

```python
from collections import defaultdict, Counter, OrderedDict, deque, namedtuple

# 1. defaultdict:缺失 key 时自动初始化
# 搜推广场景:分组聚合
user_clicks = defaultdict(list)
for log in logs:
    user_clicks[log["user_id"]].append(log["item_id"])
# 等价于:
# if user_id not in user_clicks: user_clicks[user_id] = []
# user_clicks[user_id].append(item_id)

# 2. Counter:频次统计
item_counter = Counter(log["item_id"] for log in logs)
item_counter.most_common(10)      # 出现次数 Top 10 的 item

# 过滤低频(防过拟合!)
LOW_FREQ_THRESHOLD = 5
frequent_items = {item for item, cnt in item_counter.items() if cnt >= LOW_FREQ_THRESHOLD}
# 在训练时,低频物品的 embedding 会被很少更新,容易过拟合,常被过滤

# 3. OrderedDict(3.7+ 已不推荐,普通 dict 就有序)
# 但在某些场景仍有意义,如 LRU Cache 的实现

# 4. deque:双端队列,O(1) 头尾操作
from collections import deque
window = deque(maxlen=10)            # 自动维护最近 10 个元素
for log in user_logs:
    window.append(log["item_id"])    # 老的自动弹出
# 求平均/最新
recent_clicks = list(window)         # 最近 10 次点击

# 5. namedtuple:带字段名的 tuple
ClickEvent = namedtuple("ClickEvent", ["user_id", "item_id", "timestamp", "action"])
event = ClickEvent(user_id=123, item_id=456, timestamp=1717000000, action="click")
print(event.user_id)                 # 123,可以用 . 访问字段,比 tuple[0] 可读性强
```

#### 1.9.2 `itertools`:迭代器工厂

```python
from itertools import groupby, chain, islice, count

# 1. chain:拼接多个迭代器(省内存,避免 + 列表)
all_logs = chain(log1_iter, log2_iter, log3_iter)
for log in all_logs:
    process(log)

# 2. groupby:分组(要求数据已排序!)
sorted_logs = sorted(logs, key=lambda x: x["user_id"])
for user_id, group in groupby(sorted_logs, key=lambda x: x["user_id"]):
    user_clicks = [log["item_id"] for log in group]
    process_user(user_id, user_clicks)

# 3. islice:切片迭代器
first_1000 = islice(huge_iter, 1000)     # 取前 1000 个,不占用全部内存

# 4. count(start, step):无限计数器
for i in count(start=0, step=2):         # 0, 2, 4, 6, ...
    if i > 100: break
```

#### 1.9.3 `functools`:函数工具箱

```python
from functools import lru_cache, partial, reduce, cmp_to_key

# 1. lru_cache:LRU 缓存(我们前面提过)
@lru_cache(maxsize=100_000)
def get_item_category(item_id):
    return category_db.lookup(item_id)  # 数据库查询很慢,缓存后 O(1)

# 2. reduce:累积聚合
from functools import reduce
sum_of_clicks = reduce(lambda acc, x: acc + x["click"], logs, 0)

# 3. cmp_to_key:把比较函数转 key 函数(用于 sorted)
sorted_users = sorted(users, key=cmp_to_key(lambda a, b: b["score"] - a["score"]))
```

---

### 1.10 性能优化与最佳实践

#### 1.10.1 时间复杂度意识

搜推广里大部分性能问题不是 Python 慢,而是**算法用错了复杂度**。

```python
# 反面教材:O(n²) — 千万级直接卡死
def find_common_items(user1_clicks, user2_clicks):
    common = []
    for c1 in user1_clicks:           # 1000 次
        if c1 in user2_clicks:        # 每次 O(n) = 1000
            common.append(c1)         # 1000 × 1000 = 100 万次
    return common

# 正确做法:O(n)
def find_common_items(user1_clicks, user2_clicks):
    set1 = set(user1_clicks)
    return [c for c in user2_clicks if c in set1]   # 1000 + 1000 = 2000 次
```

#### 1.10.2 向量化思维(用 numpy 替代循环)

```python
import numpy as np

# 反面教材:Python 循环逐元素计算
def sigmoid_python(x_list):
    return [1 / (1 + math.exp(-x)) for x in x_list]

# 正确做法:numpy 一次性算
def sigmoid_numpy(x_array):
    return 1 / (1 + np.exp(-x_array))

# 性能对比
import math, time
x = list(range(1_000_000))
x_np = np.array(x)

start = time.time(); sigmoid_python(x); print(f"Python: {time.time()-start:.2f}s")  # 0.62s
start = time.time(); sigmoid_numpy(x_np); print(f"numpy:  {time.time()-start:.2f}s")  # 0.012s,快 50 倍
```

**搜推广的黄金法则**:**能 numpy 就别循环,能 torch.tensor 就别 numpy**(GPU 加速)。

#### 1.10.3 避免重复造轮子

```python
# 反面教材:自己写平均
def average(lst):
    return sum(lst) / len(lst)

# 正确做法:用 statistics.mean(更快、更准)
from statistics import mean, median, stdev
```

#### 1.10.4 不要在循环里 `import`

```python
# 反面教材
for log in logs:
    import re                              # 每次循环都重新 import(虽然有缓存,但消耗查找)
    m = re.search(...)

# 正确做法:提到循环外
import re
for log in logs:
    m = re.search(...)
```

#### 1.10.5 用 `if __name__ == "__main__":` 保护主程序

```python
# mymodule.py
def helper():
    pass

# 这部分只在直接运行 python mymodule.py 时执行
# 当被 import 时,不会执行
if __name__ == "__main__":
    helper()
    print("测试一下")
```

#### 1.10.6 类型提示(Python 3.5+,搜推广生产代码标配)

```python
from typing import List, Dict, Optional, Tuple

def build_features(
    user_id: int,
    item_ids: List[int],
    context: Dict[str, float],
    timestamp: Optional[int] = None
) -> Tuple[Dict[str, int], List[int]]:
    """
    构建特征。

    Args:
        user_id: 用户 ID
        item_ids: 候选物品 ID 列表
        context: 上下文特征字典
        timestamp: 时间戳(可选)

    Returns:
        (特征字典, 标签列表)
    """
    features = {"user_id": user_id}
    labels = []
    return features, labels

# 加类型提示后:
# - IDE 能补全、能查错
# - mypy 静态检查能在运行前发现类型错误
# - 团队协作时,函数签名一目了然
```

#### 1.10.7 用 `pathlib` 替代 `os.path`(更现代、更面向对象)

```python
from pathlib import Path

# 反面教材
import os
path = os.path.join(os.path.expanduser("~"), "data", "train.parquet")
if os.path.exists(path):
    df = pd.read_parquet(path)

# 正确做法
path = Path("~/data/train.parquet").expanduser()
if path.exists():
    df = pd.read_parquet(path)

# 链式操作
parquet_files = list(Path("/data/2026/06").glob("**/*.parquet"))
print(parquet_files[0].stem)         # 文件名(无后缀)
print(parquet_files[0].parent)       # 父目录
print(parquet_files[0].stat().st_size)  # 文件大小
```

---

### 1.11 第一章小结 & 自测

**小结**:
- Python 对象是"名字贴对象",不是"变量=内存"
- 容器选型:`list`(顺序) / `dict`(映射) / `set`(去重) / `tuple`(不可变)
- `dict.get()` vs `dict[]`:强特征 vs 弱特征
- 能 `yield` 不用 `list`、能 `with` 不用裸 `open`、能 `numpy` 不用循环
- O(n²) 算法是搜推广的隐形杀手,千万级数据下任何 O(n²) 都会让 pipeline 跑几天

**自测题**:
1. 给定两个 100 万长度的列表,如何高效求它们的交集?时间复杂度?
2. 写一个 `with` 语句的上下文管理器,统计代码块耗时并打印到日志。
3. 用 `defaultdict` 写一段代码:给定 `[(user_id, item_id), ...]` 的点击日志,输出 `user_id → set(item_ids)` 字典。
4. 解释为什么 `lst.append(x)` 比 `lst.insert(0, x)` 快?
5. 装饰器 `@lru_cache` 适用什么场景?什么场景不能用?

---

> **下一章预告**:
> 第 2 章我们将进入**数据处理**——从原始日志到模型可吃的 Tensor。你会学到:
> - 工业级数据格式(Parquet/ORC/TFRecord)的差异与选型
> - 离散特征 vs 连续特征的工业界处理范式
> - 搜推广特有的"时序特征"怎么构建、Padding 怎么搞
> - PyTorch `Dataset` / `DataLoader` 怎么写才健壮、高效
> - 怎么避免"训练-推理"特征不一致这个天坑

---

## 第 2 章:数据处理——从原始日志到模型可吃的 Tensor

> **本章目标**:搜推广里,数据处理往往比模型本身还重要。本章带你走通"原始日志 → 清洗 → 编码 → 切分 → Dataset/DataLoader"全流程,生成式推荐特有的"SEM(semantic ID)"也会在第 4 章展开。
>
> 学完本章,你要能回答:
> - 为什么工业界用 Parquet 不用 CSV?
> - 离散特征怎么数值化?One-Hot 还是 Embedding?
> - 变长用户行为序列怎么 Padding 不会污染模型?
> - 离线/在线特征不一致怎么避免?

### 2.1 数据格式:工业界的"硬盘语言"

#### 2.1.1 主流格式对比

| 格式 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **CSV/TSV** | 人类可读、任意编辑器可看 | 无类型、压缩率低、列式 IO 慢 | 临时调试、小数据 |
| **JSON** | 嵌套结构友好、灵活 | 解析慢、占空间、无法流式 | API 数据、配置文件 |
| **Parquet** | **列式存储、压缩率高、列裁剪快、类型丰富** | 不可直接 cat/grep | **工业界离线训练数据首选** |
| **TFRecord** | TensorFlow 原生、ProtoBuf 序列化 | 绑定 TF、不能直接读 | TF 生态内部 |
| **ORC** | Hive/Spark 友好 | 生态偏大数据 | 大数据处理 |
| **HDF5** | 层级结构、随机访问 | 写入慢、容易写坏 | 科学计算 |

**为什么工业界偏爱 Parquet?**

```python
import pandas as pd
import pyarrow.parquet as pq

# 1. 写 Parquet(一次写,带 schema,压缩)
df = pd.DataFrame({
    "user_id": [1, 2, 3] * 100_000,
    "item_id": [10, 20, 30] * 100_000,
    "click": [1, 0, 1] * 100_000,
})
df.to_parquet("data.parquet", compression="snappy", index=False)
# snappy:压缩比 2~3x,速度极快

# 2. 列式裁剪(只读 1 列,不必读全部)
df_user = pd.read_parquet("data.parquet", columns=["user_id"])
# 1 亿行 × 1 列,只读 400MB;读 3 列就读 1.2GB

# 3. 带过滤下推(pushdown)
df_click = pd.read_parquet("data.parquet", filters=[("click", "=", 1)])
# 只把 click=1 的行从磁盘读出来,跳过其他,IO 减少 50~99%
```

#### 2.1.2 选型建议(搜推广场景)

| 数据规模 | 推荐格式 | 工具链 |
| :--- | :--- | :--- |
| < 1 GB | CSV / Parquet | pandas |
| 1 GB ~ 1 TB | **Parquet + snappy** | pandas / pyarrow / Spark |
| > 1 TB | **Parquet 分区**(按 dt/hour 切) | Spark / Hive / MaxCompute |
| 实时流 | Kafka + Avro/Protobuf | Flink / Spark Streaming |

**搜推广特有的分区规范**:
```text
s3://bucket/recsys/
├── dt=2026-06-01/
│   ├── hour=00/
│   │   └── part-00000.parquet
│   ├── hour=01/
│   └── ...
├── dt=2026-06-02/
└── ...
```
按 `dt=2026-06-05` 一选,直接砍掉其他天;按 `dt=2026-06-05/hour=14` 选,只读 1 小时 60GB 数据。

#### 2.1.3 内存格式:Arrow vs NumPy

```python
# pyarrow 内部用 Arrow 格式,跨语言共享、零拷贝
import pyarrow as pa

arr = pa.array([1, 2, 3, None, 5])        # 支持 null
print(arr.type)                            # int64
# 零拷贝转 pandas
df = arr.to_pandas()                       # 不复制数据,只是换了视图
# 零拷贝转 numpy
np_arr = arr.to_numpy()                    # 同样零拷贝
```

**为什么搜推广要关心 Arrow?**
- PySpark → Arrow → PyTorch DataLoader 可以**零拷贝**传递数据
- 训练时不再"pandas → numpy → torch.tensor"地复制多份

---

### 2.2 文本数据处理(对评论、标题、Query)

#### 2.2.1 基础清洗

```python
import re

def clean_text(text: str) -> str:
    """搜推广常见的文本清洗"""
    if not isinstance(text, str):
        return ""
    text = text.lower()                              # 统一小写
    text = re.sub(r"https?://\S+", "<URL>", text)    # URL 替换
    text = re.sub(r"\d+", "<NUM>", text)             # 数字替换
    text = re.sub(r"\s+", " ", text).strip()         # 合并空白
    return text

clean_text("iPhone 15 Pro 价格 9999! https://apple.com/iphone")
# "iphone <NUM> pro 价格 <NUM>! <URL>"
```

#### 2.2.2 中文分词

```python
# 方案 1:jieba(经典、但不是 SOTA)
import jieba
words = jieba.lcut("用户点击了iPhone 15 Pro Max")
# ['用户', '点击', '了', 'iPhone', ' ', '15', ' ', 'Pro', ' ', 'Max']

# 方案 2:thulac(清华)
# pip install thulac
import thulac
thu = thulac.thulac()
words = thu.cut("用户点击了iPhone 15 Pro Max")
# [['用户', 'n'], ['点击', 'v'], ...]

# 方案 3:不切词,直接用预训练 tokenizer(BERT 系)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-chinese")
tokens = tok("用户点击了iPhone 15 Pro Max")["input_ids"]
# [101, 7110, 3121, 749, 8567, 8115, 102]
```

**搜推广的特殊考虑**:
- **Query 分词**:用户搜索的"短文本"对召回质量至关重要,搜推广场景常用**双词/双字**兜底
- **物品标题**:用 BERT 算 embedding 比传统 word2vec 更好
- **拼写纠错**:在搜索场景,需要先纠错再分词

#### 2.2.3 词表构建(Vocabulary)

```python
from collections import Counter
from typing import List, Dict

class Vocab:
    """工业级词表管理类"""
    def __init__(self, min_freq: int = 5, max_size: int = None,
                 specials: List[str] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.specials = specials or ["<PAD>", "<UNK>"]
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self._build_specials()

    def _build_specials(self):
        for sp in self.specials:
            self.token2id[sp] = len(self.id2token)
            self.id2token.append(sp)

    def fit(self, tokens_iter):
        """从 token 迭代器中统计频次,构建词表"""
        counter = Counter()
        for tokens in tokens_iter:                  # 兼容 list[str] 和 iter[list[str]]
            if isinstance(tokens, str):
                counter[tokens] += 1
            else:
                counter.update(tokens)              # tokens 是 list[str]

        # 按频次降序,过滤低频
        most_common = [(t, c) for t, c in counter.most_common() if c >= self.min_freq]
        if self.max_size:
            most_common = most_common[:self.max_size - len(self.specials)]

        for token, _ in most_common:
            if token not in self.token2id:          # 防止特殊符号被覆盖
                self.token2id[token] = len(self.id2token)
                self.id2token.append(token)

        return self

    def __len__(self):
        return len(self.id2token)

    def encode(self, tokens) -> List[int]:
        """token → id"""
        unk = self.token2id["<UNK>"]
        if isinstance(tokens, str):
            return [self.token2id.get(tokens, unk)]
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids) -> List[str]:
        """id → token"""
        return [self.id2token[i] for i in ids]

# 使用
vocab = Vocab(min_freq=5, max_size=50_000)
tokens_iter = [["用户", "点击", "iPhone"], ["iPhone", "好", "用"], ...]
vocab.fit(tokens_iter)
print(len(vocab))                  # 50000
print(vocab.encode("iPhone"))      # [123]
print(vocab.encode(["用户", "未知词"]))  # [2, 1]  ← "未知词" 走 <UNK>=1
```

**关键设计**:
- **`<PAD>`(0)**:用于序列对齐的占位符
- **`<UNK>`(1)**:未登录词,所有不在词表里的都映射到这里
- **特殊符号必须在最前**:保证 ID 是固定的
- **min_freq 过滤低频**:防止词表爆炸,过拟合

#### 2.2.4 文本序列 Padding 与 Truncation

```python
import torch

def pad_sequence(seq_list: List[List[int]],
                 pad_id: int = 0,
                 max_len: int = None) -> torch.Tensor:
    """把变长序列列表填充成 [B, T] 形状"""
    if max_len is None:
        max_len = max(len(s) for s in seq_list)

    B = len(seq_list)
    out = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seq_list):
        # Truncation:超长截断
        seq = seq[:max_len]
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out

# 使用
padded = pad_sequence(
    [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11]],
    pad_id=0,
    max_len=5
)
# tensor([[1, 2, 3, 4, 0],
#         [5, 6, 0, 0, 0],
#         [7, 8, 9, 10, 11]])
```

---

### 2.3 离散特征(Categorical Features)处理

> **这是搜推广与 CV 最大的差异点**——你会反复打交道的部分。

#### 2.3.1 离散特征的本质:把字符串 ID 变成整数索引

```python
# 原始数据(可能来自 Hive、Parquet、日志)
raw_user_ids = ["u_123", "u_456", "u_789", "u_123", "u_999"]
raw_item_ids = ["i_apple", "i_nike", "i_samsung", "i_apple", "i_adidas"]
raw_cities   = ["北京", "上海", "北京", "深圳", "北京"]

# 步骤 1:统计唯一值并排序,分配 ID
from collections import defaultdict

def label_encode(values):
    mapping = {}                          # 字符串 → 整数
    for v in values:
        if v not in mapping:
            mapping[v] = len(mapping)
    return mapping

user_mapping = label_encode(raw_user_ids)   # {"u_123": 0, "u_456": 1, "u_789": 2, "u_999": 3}
item_mapping = label_encode(raw_item_ids)
city_mapping = label_encode(raw_cities)     # {"北京": 0, "上海": 1, "深圳": 2}

# 步骤 2:把字符串转成整数
user_ids_int = [user_mapping[u] for u in raw_user_ids]
# [0, 1, 2, 0, 3]
```

#### 2.3.2 One-Hot 编码:为什么搜推广不用它?

```python
# 假设有 1000 万个物品
import numpy as np
n_items = 10_000_000
item_id = 1234567

# One-Hot:创建一个 1000 万维的向量,只有一个位置是 1
one_hot = np.zeros(n_items, dtype=np.float32)
one_hot[item_id] = 1
# 内存: 10_000_000 * 4 = 40MB
# 训练时: 1000 万 × 几十维 矩阵乘法,极慢
```

**为什么搜推广不用 One-Hot?**

1. **内存爆炸**:1000 万物品 × 4 字节 = 40MB,只表示一个物品
2. **计算浪费**:矩阵乘法里 99.99% 是 0 乘 1,GPU 也救不了
3. **泛化差**:One-Hot 假设所有物品互相独立,无法学习"相似物品应该有相似表示"

**替代方案:Embedding(第 3 章详细讲)**

```python
# 同样 1000 万个物品,用 Embedding 表
# 1000 万 × 32 维,总共 320MB(可接受)
# 每个物品用 32 维稠密向量表示
emb = nn.Embedding(10_000_000, 32)   # 320MB
item_vec = emb(torch.tensor([1234567]))  # [1, 32],只查一行
```

**对比表**:

| 方式 | 1 个物品的表示 | 1 亿物品的总内存 | 能否学相似性 |
| :--- | :--- | :--- | :--- |
| One-Hot (1M 维) | 1M 维稀疏向量 | 400 GB | ❌ |
| Embedding (32 维) | 32 维稠密向量 | 12.8 GB | ✅ |

#### 2.3.3 多值特征(Multi-hot):用户行为序列的雏形

```python
# 用户历史点击了 5 个物品:[1, 2, 3, 1, 2]
# 用 Multi-hot 编码
n_items = 100
multi_hot = np.zeros(n_items, dtype=np.float32)
multi_hot`1, 2, 3` = 1
# [0, 1, 1, 1, 0, 0, ...]
```

**工业界不直接用 Multi-hot,改用:**
- **EmbeddingBag**(平均/求和/最大池化)
- **带 Attention 的序列模型**(DIN/DIEN)
- **Transformer**(SASRec/BERT4Rec)

#### 2.3.4 哈希编码(Hashing Trick):超高基数场景的救命稻草

```python
# 场景:有 1 亿个不同的 URL 特征,做词表会爆内存
# 用哈希函数映射到固定 100 万桶
def hash_encode(value, num_buckets=1_000_000):
    return hash(value) % num_buckets

url = "https://www.example.com/product/12345"
hash_encode(url)   # 某个 [0, 1M) 的整数
```

**优缺点**:
- ✅ 无需预建词表,省内存
- ✅ 处理冷启动
- ❌ 哈希冲突:不同 URL 可能落到同一桶
- ❌ 没有可解释性

**搜推广适用场景**:超长尾的物品/Query,作为辅助特征。

#### 2.3.5 离散特征的"工程避坑"

| 坑 | 表现 | 解决方案 |
| :--- | :--- | :--- |
| 新类别没在词表 | 线上 crash | 留 `<UNK>` 兜底 ID |
| 训练/推理词表不一致 | AUC 突掉 | 强制使用同一份 Vocab 序列化文件 |
| 高频 vs 长尾物品分布悬殊 | 长尾 embedding 几乎不更新 | 过滤低频、负采样、特征哈希 |
| 类别基数过大(> 1 亿) | 内存爆炸 | 哈希分桶 + 共享 embedding |

#### 2.3.6 Embedding 维度选择指南:32 还是 64?为什么?

> **"Embedding 维度设多少?"——这是搜推广面试必问题,也是 CV 转搜推广最容易出错的配置。**

**核心原则**:Embedding 维度取决于**特征的有效信息容量**,不是越大越好。

**经验法则表**:

| 特征类型 | 典型基数 | 推荐的 Embedding 维度 | 理由 |
| :--- | :--- | :--- | :--- |
| **用户 ID** | 1 亿 ~ 10 亿 | 32 ~ 64 | 基数极大,但每个用户信息量有限 |
| **物品 ID** | 1000 万 ~ 1 亿 | 32 ~ 64 | 同上 |
| **类目(Category)** | 1000 ~ 10000 | 8 ~ 16 | 基数小,信息量有限 |
| **品牌(Brand)** | 1 万 ~ 10 万 | 16 ~ 32 | 中等基数 |
| **年龄(Age)** | 100 | 4 ~ 8 | 基数极小,没必要用 64 |
| **城市(City)** | 500 ~ 2000 | 8 ~ 12 | 中等基数 |
| **搜索词(Query)** | 100 万 ~ 1000 万 | 32 ~ 64 | 高信息密度 |
| **时间片(Hour)** | 24 | 4 | 周期性强,低位就够 |
| **位置(Position)** | 10 ~ 100 | 4 ~ 8 | 位置偏差矫正用 |

**"4 维真的够用吗?"——对小基数特征,多了反而坏事**

```python
import torch.nn as nn

# 城市特征:只有 500 个城市
city_emb_4 = nn.Embedding(500, 4)     # 参数量:500 × 4 = 2000
city_emb_64 = nn.Embedding(500, 64)   # 参数量:500 × 64 = 32000

# 差距:16 倍!
# 但 500 个城市之间的差异,4 维向量(4096 种组合)远远装得下
# 64 维反而会让 embedding 学到冗余噪声
```

**维度选择的经验公式**:
```python
def suggest_embedding_dim(vocab_size: int) -> int:
    """
    基于实证研究的 Embedding 维度推荐公式。

    来源:Google 的《Scaling Embedding Layers in Language Models》
    和工业界搜推广的实践经验。
    """
    if vocab_size < 1000:
        return max(4, vocab_size // 20)     # 小词表,维度与基数成正比
    elif vocab_size < 100_000:
        return min(64, int(vocab_size ** 0.25) * 4)  # 亚线性增长
    else:
        return min(128, max(32, int(vocab_size ** 0.2) * 8))  # 极大规模,维度增长趋缓

# 示例:
for vocab in [100, 10000, 1_000_000, 100_000_000]:
    print(f"vocab={vocab:>10d} → dim={suggest_embedding_dim(vocab):>3d}")
# vocab=       100 → dim=  8
# vocab=     10000 → dim= 40
# vocab=   1000000 → dim= 48
# vocab= 100000000 → dim= 64
```

**高维 Embedding 的"过参数化"问题**:

```python
# 假设用户 Embedding: 1 亿 × 128 维
user_emb = nn.Embedding(100_000_000, 128)
# 参数量: 128 亿
# 内存(FP32): 12.8B × 4B = 51.2 GB
# 而每个用户真正有效的信息量(假设用户只有 10 次交互)
# 128 维向量有 128 × 32bit = 4096 bit 的信息容量
# 10 次交互最多提供 10 × log2(1000 万) ≈ 10 × 24 = 240 bit
# 维度是需求的 17 倍!大量维度在"浪费"
```

**维度调优的搜推广工程方法**:

```python
# 在团队项目中,应该把 Embedding 维度做成可配置参数
# 然后在实验中系统性地对比

configs = {
    "dim_16":  {"embedding_dim": 16, "params": "1.2B"},
    "dim_32":  {"embedding_dim": 32, "params": "2.5B"},
    "dim_64":  {"embedding_dim": 64, "params": "5.0B"},
    "dim_128": {"embedding_dim": 128, "params": "10B"},
}

# 实验流程:
# 1. 先用 dim=16 跑基线
# 2. 依次增加到 32, 64, 128
# 3. 如果 dim=64 vs dim=128 AUC 差异 < 0.001 → 选 dim=64
# 4. 如果 dim=64 比 dim=32 涨了 0.003+ → 选 dim=64
#
# 结论:不是维度越大越好!很多场景下 32 和 64 的差距不到 0.0005
# 但参数量和显存差 2 倍!
```

---

### 2.4 连续特征(Numerical Features)处理

#### 2.4.1 连续特征示例

```python
# 用户年龄、点击率、历史点击数、价格、时长、距离等
user_age = 25
historical_ctr = 0.156
price = 8999.0
click_count_7d = 42
```

#### 2.4.2 缺失值处理

```python
import numpy as np

# 搜推广里 50% 以上的连续特征都有缺失值(用户没填年龄、没历史点击等)

# 方案 1:填充固定值(-1 或 0,模型能区分"真零"和"缺失零")
ages = [25, np.nan, 30, 22, np.nan]
ages_filled = [-1 if np.isnan(a) else a for a in ages]
# [25, -1, 30, 22, -1]

# 方案 2:均值/中位数填充
ages_arr = np.array(ages)
ages_arr[np.isnan(ages_arr)] = np.nanmean(ages_arr)

# 方案 3:额外加一个"是否缺失"的 0/1 特征(推荐!)
ages_with_flag = []
for a in ages:
    ages_with_flag.append([0 if not np.isnan(a) else 1, -1 if np.isnan(a) else a])
# [[0, 25], [1, -1], [0, 30], [0, 22], [1, -1]]
# 显式告诉模型"这个值缺失了",比单纯填一个默认值好
```

#### 2.4.3 归一化(Normalization)

**为什么搜推广必须归一化?**
- 连续特征量级差异大(如年龄 0~100,价格 0~100000)
- 不归一化的话,MLP 训练时梯度会被大值特征主导,小值特征几乎学不到

```python
# 方案 1:Min-Max 归一化(把所有值压到 [0, 1])
def min_max_scale(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min + 1e-8)

# 方案 2:Z-Score 标准化(均值 0,方差 1)
def z_score_scale(x, mean, std):
    return (x - mean) / (std + 1e-8)

# 方案 3:对数变换(对长尾分布友好)
def log_scale(x, eps=1.0):
    return np.log(x + eps)            # 加 1 是为了处理 0 值

# 示例:商品价格(严重长尾)
prices = [9.9, 99, 999, 9999, 99999, 999999]
log_prices = log_scale(np.array(prices))
# [2.36, 4.62, 6.91, 9.21, 11.51, 13.82]   ← 分布变均匀
```

**避坑**:
- 训练/推理必须用**同一份 mean/std**
- 一定要**先 fit 训练集,再 transform 验证/测试集**(避免数据泄漏)

#### 2.4.4 分箱(Binning / Discretization)

```python
# 把连续值切成几个桶,变成离散值
def bin_age(age):
    if age < 18: return 0   # 未成年
    elif age < 25: return 1 # 青年
    elif age < 40: return 2 # 中年
    else: return 3          # 中老年

# 用 pandas 更简单
ages = pd.Series([15, 22, 35, 45, 60, 28])
bins = pd.cut(ages, bins=[0, 18, 25, 40, 100], labels=[0, 1, 2, 3])
# 0    0
# 1    1
# 2    2
# 3    3
# 4    3
# 5    1
```

**为什么搜推广要做分箱?**
- 让模型能学到"非线性的阶跃"关系(如年龄 < 18 不消费)
- 对异常值鲁棒(年龄填了 9999 不会爆)
- 方便和离散特征一起走 Embedding

#### 2.4.5 工业界连续特征处理流水线

```python
from sklearn.preprocessing import StandardScaler
import joblib

class NumericalFeatureProcessor:
    """搜推广连续特征处理流水线"""
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.scalers = {name: StandardScaler() for name in feature_names}
        self.fitted = False

    def fit(self, df):
        """在训练集上 fit"""
        for name in self.feature_names:
            self.scalers[name].fit(df`name`.fillna(0))
        self.fitted = True
        return self

    def transform(self, df):
        """在任意数据集上 transform"""
        assert self.fitted, "必须先 fit"
        out = pd.DataFrame()
        for name in self.feature_names:
            # 1. 缺失值标记
            missing_flag = df[name].isna().astype(np.float32).rename(f"{name}_is_missing")
            # 2. 填充缺失
            filled = df[name].fillna(0)
            # 3. Z-Score 归一化
            scaled = self.scalers[name].transform(filled.values.reshape(-1, 1)).flatten()
            out[f"{name}_scaled"] = scaled
            out[f"{name}_is_missing"] = missing_flag.values
        return out

    def save(self, path):
        """序列化,保证训练/推理一致"""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

# 使用
num_proc = NumericalFeatureProcessor(["user_age", "historical_ctr", "price"])
num_proc.fit(train_df)
train_features = num_proc.transform(train_df)
val_features = num_proc.transform(val_df)
num_proc.save("numerical_processor.pkl")          # 推理时直接 load
```

---

### 2.5 时序特征(搜推广的灵魂)

> **CV 里你只有静态图片,搜推广里你面对的是"用户随时间变化的行为轨迹"**。这是你转方向后必须吃透的章节。

#### 2.5.1 用户行为序列长什么样?

```python
# 典型日志格式(每行一条行为)
# timestamp  user_id  item_id  action  duration
# 1717000001  u_123    i_001    click   5
# 1717000045  u_123    i_002    click   3
# 1717000234  u_123    i_001    dwell   30
# 1717000512  u_123    i_003    click   2
# ...

# 同一用户,按时间排序后就是一个"行为序列"
user_history = {
    "user_id": "u_123",
    "history": [
        {"item_id": "i_001", "action": "click", "ts": 1717000001},
        {"item_id": "i_002", "action": "click", "ts": 1717000045},
        {"item_id": "i_001", "action": "dwell", "ts": 1717000234},   # 重复行为
        {"item_id": "i_003", "action": "click", "ts": 1717000512},
    ]
}
```

#### 2.5.2 从原始日志到"训练样本"的转换

```python
# 搜推广的样本构造:滑动窗口(最重要!)
def build_samples(user_history, max_len=10, pred_next=True):
    """
    把"行为序列"切成"训练样本"。
    每个样本 = (历史序列, 下一时刻的真实行为, label)。
    """
    history = user_history["history"]
    if len(history) < 2:
        return []                                       # 至少需要 2 步

    samples = []
    for t in range(1, len(history)):
        prefix = history[max(0, t - max_len):t]         # 取前 max_len 步作为历史
        target = history[t]
        samples.append({
            "history_item_ids": [h["item_id"] for h in prefix],
            "target_item_id": target["item_id"],
            "label": 1                                    # 真实行为,label=1
        })
    return samples

# 使用
samples = build_samples(user_history, max_len=3)
# [
#   {"history": [i_001], "target": i_002, "label": 1},
#   {"history": [i_001, i_002], "target": i_001, "label": 1},
#   {"history": [i_001, i_002, i_001], "target": i_003, "label": 1}
# ]
```

**负样本怎么来?**

```python
# 方案 1:全物品随机采样(简单但效率低)
import random
n_items = 1_000_000
neg_item_id = random.randint(0, n_items - 1)             # 90% 不是用户真正感兴趣的

# 方案 2:曝光未点击(工业界主流!)
# 用户被曝光了 100 个物品,点了 2 个,剩下 98 个就是"曝光未点击"负样本
exposed_items = [i_xxx, i_yyy, i_zzz, ...]                # 推荐系统推的候选
clicked_items = [i_001, i_002]
neg_items = [i for i in exposed_items if i not in clicked_items]

# 方案 3:Hard Negative Mining(高级)
# 用一个简单模型预测,选分数高但用户没点的作为"困难负样本"
# 让训练更有挑战性
```

#### 2.5.3 Padding:变长序列的"对齐术"

```python
# 用户 A 历史长度 3,用户 B 历史长度 10,怎么拼成一个 batch?
seqs = [
    [1, 2, 3],
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
]
# 直接 stack 失败,长度不一致

# Padding:用 <PAD>(=0) 填到等长
padded = pad_sequence(seqs, pad_id=0, max_len=10)
# tensor([[ 1,  2,  3,  0,  0,  0,  0,  0,  0,  0],
#         [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13]])

# 但 Padding 会污染计算!需要 mask 标记哪些是 padding
mask = (padded != 0).long()                              # [B, T]
# tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# 模型中使用
import torch
emb = nn.Embedding(100, 8, padding_idx=0)                # padding_idx=0 告诉 Embedding 位置 0 不更新
seq_emb = emb(padded)                                    # [B, T, 8]
# 用 mask 过滤 padding
masked_sum = (seq_emb * mask.unsqueeze(-1)).sum(dim=1)   # [B, 8]
masked_cnt = mask.sum(dim=1, keepdim=True)               # [B, 1]
user_repr = masked_sum / masked_cnt.clamp(min=1)         # [B, 8],平均池化
```

**Padding 的坑**:
- 不带 mask 的平均池化,会把 padding 当成真实信号,严重拉低质量
- 训练/推理的 padding 策略必须一致
- 长序列要截断(truncate)而不是 padding 到底

#### 2.5.4 序列采样策略

```python
# 训练时一个 epoch 跑 100 万用户,每个用户 1000 步历史,如果不采样,内存炸
# 工业界常用:

# 策略 1:随机窗口采样
def random_window(history, window_size=10):
    if len(history) <= window_size:
        return history
    start = random.randint(0, len(history) - window_size)
    return history[start:start + window_size]

# 策略 2:保留最近 N 个(简单,工业界最常用)
def recent_window(history, n=10):
    return history[-n:]

# 策略 3:时间衰减采样(越近的权重越大)
def time_decay_window(history, n=10):
    # 按 ts 排序,保留最近 n
    return sorted(history, key=lambda x: x["ts"], reverse=True)[:n]
```

---

### 2.6 数据集划分(避坑:数据泄漏)

#### 2.6.1 三种划分方式

```python
# 方式 1:随机划分(CV 里的标准做法)
# 搜推广里**绝对不能用**!会导致严重数据泄漏!
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
# 问题:用户的行为有强时间相关性,随机切会把"今天点击"和"明天点击"混在一起
# 训练时见过未来,测试时当然容易,线上却崩

# 方式 2:时间切分(搜推广标准做法)
train = df[df["ts"] < "2026-06-01"]                       # 用前几天训练
val   = df[(df["ts"] >= "2026-06-01") & (df["ts"] < "2026-06-03")]  # 中间 2 天验证
test  = df[df["ts"] >= "2026-06-03"]                       # 最后 2 天测试

# 方式 3:用户维度切分(防止同一用户出现在 train/test)
import numpy as np
users = df["user_id"].unique()
np.random.shuffle(users)
train_users = set(users[:int(0.8 * len(users))])
val_users = set(users[int(0.8 * len(users)):int(0.9 * len(users))])
test_users = set(users[int(0.9 * len(users)):])

train = df[df["user_id"].isin(train_users)]
val = df[df["user_id"].isin(val_users)]
test = df[df["user_id"].isin(test_users)]
```

#### 2.6.2 离线评估 vs 线上表现:为什么 AUC 涨了 1% 线上却没动静?

```text
离线:训练集 → 验证集(同分布)→ AUC 涨了
线上:昨天模型 → 今天流量(分布漂移)→ 没动静
```

**根因**:
- 离线训练用的是**历史点击数据**,但**线上用户兴趣在漂移**
- **特征穿越**:训练时用了"未来才能知道"的特征(如"用户最终点击的商品")
- **位置偏差**:离线把"精排结果"和"随机负样本"混在一起训,线上却只有"精排结果"

**搜推广的工程经验**:
- 离线 AUC 涨 0.5%,**线上可能涨 1%** 也可能 **不变甚至掉**
- 一定要 **离线 + 线上 A/B 双指标**,**缺一不可**

---

### 2.7 PyTorch `Dataset` 与 `DataLoader`(工业级写法)

#### 2.7.1 `Dataset`:把数据样本化的"模板"

```python
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

class RecSysDataset(Dataset):
    """
    搜推广通用 Dataset。
    关键:不要在 __init__ 里做重 IO,放 __getitem__ 里按需读
    """
    def __init__(self, parquet_paths, vocab, max_seq_len=50):
        """
        Args:
            parquet_paths: list of parquet 文件路径(支持分片)
            vocab: Vocabulary 对象
            max_seq_len: 序列最大长度
        """
        # 关键:惰性加载,只读 metadata,不读全部数据
        self.parquet_paths = parquet_paths
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # 用 pyarrow 快速扫一遍,统计总行数
        import pyarrow.parquet as pq
        self.total_rows = sum(
            pq.ParquetFile(p).metadata.num_rows
            for p in parquet_paths
        )

        # 构建 row_index → (file, local_row) 映射,避免存大表
        self.index_map = []
        offset = 0
        for path in parquet_paths:
            n_rows = pq.ParquetFile(path).metadata.num_rows
            self.index_map.append((path, offset, n_rows))
            offset += n_rows

    def __len__(self):
        return self.total_rows

    def _read_row(self, global_idx):
        """根据 global_idx 找到对应的文件,只读那一行"""
        for path, offset, n_rows in self.index_map:
            if offset <= global_idx < offset + n_rows:
                local_idx = global_idx - offset
                # pyarrow 高效按行读
                import pyarrow.parquet as pq
                table = pq.read_table(path, rows=[local_idx, local_idx + 1])
                return table.to_pandas().iloc[0]
        raise IndexError(f"Index {global_idx} out of range")

    def __getitem__(self, idx):
        """返回单个样本的 dict"""
        row = self._read_row(idx)

        # 离散特征 → ID
        user_id = self.vocab.encode(row["user_id"])        # int
        item_id = self.vocab.encode(row["item_id"])        # int

        # 时序特征 → 序列
        history_ids = self.vocab.encode(row["history_item_ids"][-self.max_seq_len:])
        history_ids = history_ids + [0] * (self.max_seq_len - len(history_ids))  # 右侧 padding

        # 连续特征 → tensor
        age = float(row.get("user_age", 0) or 0)
        ctr = float(row.get("historical_ctr", 0) or 0)

        # 标签
        label = int(row["label"])

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "history_ids": torch.tensor(history_ids, dtype=torch.long),
            "dense_features": torch.tensor([age, ctr], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }
```

#### 2.7.2 `DataLoader`:把样本"打包"成 batch

```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    """自定义 batch 拼接:处理长度不一的序列"""
    out = {
        "user_id": torch.stack([b["user_id"] for b in batch]),
        "item_id": torch.stack([b["item_id"] for b in batch]),
        "history_ids": torch.stack([b["history_ids"] for b in batch]),
        "dense_features": torch.stack([b["dense_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    return out

# 创建 DataLoader
dataset = RecSysDataset(
    parquet_paths=["train_001.parquet", "train_002.parquet"],
    vocab=my_vocab,
    max_seq_len=50
)
loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=8,                  # 多进程加速
    collate_fn=collate_fn,
    pin_memory=True,                # 加速 CPU→GPU 拷贝
    prefetch_factor=4,              # 每个 worker 预取 4 个 batch
    persistent_workers=True,        # 跨 epoch 保留 worker(减少冷启动)
    drop_last=True                  # 最后一个不完整的 batch 丢掉
)

# 训练时
for batch in loader:
    print(batch["user_id"].shape)        # [1024]
    print(batch["history_ids"].shape)    # [1024, 50]
    break
```

**DataLoader 关键参数详解**:

| 参数 | 作用 | 推荐值 |
| :--- | :--- | :--- |
| `batch_size` | 每个 batch 的样本数 | 256~4096(看显存) |
| `shuffle` | 是否打乱顺序 | 训练 True,验证 False |
| `num_workers` | 加载数据的子进程数 | 通常 = CPU 核数 / 2 |
| `pin_memory` | 把数据锁在内存里,加速 GPU 拷贝 | True(用 GPU 时) |
| `prefetch_factor` | 每个 worker 预取的 batch 数 | 2~4 |
| `persistent_workers` | 跨 epoch 保留 worker | True(训练 1 epoch 很贵时) |
| `collate_fn` | 怎么把样本拼成 batch | 自定义,处理变长序列 |
| `drop_last` | 丢掉不完整的最后一个 batch | True(避免 batch size 抖动) |

#### 2.7.3 高级技巧:多源数据融合

```python
class MultiSourceDataset(Dataset):
    """从 Parquet + KV 数据库 + 实时缓存三个源读"""
    def __init__(self, parquet_paths, kv_client, cache):
        self.parquet_ds = RecSysDataset(parquet_paths, ...)
        self.kv_client = kv_client                              # Redis / Tair
        self.cache = cache                                      # 进程内 LRU

    def __getitem__(self, idx):
        item = self.parquet_ds[idx]

        # 1. 先查本地缓存
        feat = self.cache.get(item["user_id"])
        if feat is None:
            # 2. 缓存未命中,查 KV
            feat = self.kv_client.get(f"user_features:{item['user_id']}")
            if feat is None:
                # 3. 兜底:用默认值
                feat = self._default_features()
            self.cache.put(item["user_id"], feat)

        item["user_features"] = feat
        return item
```

#### 2.7.4 IterableDataset:流式场景

```python
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    """从 Kafka / Flink 流式读取,Dataset 大小未知"""
    def __init__(self, kafka_consumer):
        self.consumer = kafka_consumer

    def __iter__(self):
        for msg in self.consumer:
            yield self._parse(msg)

    def _parse(self, msg):
        return {
            "user_id": int(msg.value["user_id"]),
            "item_id": int(msg.value["item_id"]),
            "label": int(msg.value["label"]),
        }

# 注意:IterableDataset 不支持 __len__,shuffle=False,只能顺序读
loader = DataLoader(StreamDataset(consumer), batch_size=1024)
```

---

### 2.8 工业界实践:特征工程完整 Pipeline

#### 2.8.1 特征 Schema(FeatureConfig)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class FeatureConfig:
    """搜推广的"特征配置表"——团队协作的核心"""

    # 离散特征
    categorical: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "user_id": {"vocab_size": 100_000_000, "embedding_dim": 32, "padding_idx": 0},
        "item_id": {"vocab_size": 50_000_000,  "embedding_dim": 32, "padding_idx": 0},
        "city":    {"vocab_size": 1000,         "embedding_dim": 8,  "padding_idx": 0},
        "category_id": {"vocab_size": 5000,     "embedding_dim": 8,  "padding_idx": 0},
    })

    # 连续特征
    numerical: List[str] = field(default_factory=lambda: [
        "user_age", "historical_ctr", "price", "click_count_7d", "dwell_time_avg"
    ])

    # 时序特征
    sequence: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "click_history": {"max_len": 50, "vocab": "item_id", "pooling": "attention"},
        "search_history": {"max_len": 20, "vocab": "query",   "pooling": "mean"},
    })

    # 标签
    label: str = "click"

# 全团队共用一个 config,保证训练/服务/实验三处一致
config = FeatureConfig()
```

#### 2.8.2 特征存储(Feature Store)

```python
# 工业界的"特征中心"——离线写入,在线查询
# 1. 离线(批处理):Spark / Hive 写 Parquet
# 2. 在线(实时查询):Redis / Tair / KV

# 特征抽取样例
class FeatureStore:
    def __init__(self, offline_path, online_client):
        self.offline_path = offline_path
        self.online_client = online_client

    def get_offline(self, user_id, ts):
        """训练时使用:从 Parquet 查历史快照"""
        df = pd.read_parquet(f"{self.offline_path}/user_id={user_id}")
        return df[df["ts"] <= ts].iloc[-1]

    def get_online(self, user_id):
        """推理时使用:从 Redis 实时查"""
        key = f"user_features:{user_id}"
        return self.online_client.hgetall(key)

# 搜推广的铁律:**离线特征 ≠ 在线特征**(后面章节详谈)
```

#### 2.8.3 离线/在线一致性(Feature Consistency)

```text
[离线] Spark 批处理 + 离线表(可能延迟 1 天)
     ↓ 训练
[模型]
     ↓ 推理
[在线] 实时特征 + Redis(实时,但可能不完整)

→ 离线训练时 user_age=25,在线推理时 user_age=null(用户没填过)
→ 训练时特征维度 50 个,在线时漏了一个 → 报错
```

**解法**:
1. **特征 schema 统一**:FeatureConfig 是 single source of truth
2. **默认值兜底**:每个特征都有 default,缺失时用 default
3. **监控告警**:实时监控线上特征分布,跟离线比对,漂移超过阈值就报警
4. **特征版本管理**:每次训练/部署用特征版本号标记

#### 2.8.4 离线/在线特征校验的工业实践

```python
# 一段"防御性"代码:在模型入口处做特征校验
def validate_features(features: Dict[str, Any], config: FeatureConfig):
    """检查特征是否符合预期"""
    for name in config.categorical:
        if name not in features:
            raise ValueError(f"分类特征 {name} 缺失")
        if not isinstance(features[name], int) or features[name] < 0:
            raise ValueError(f"分类特征 {name} 值非法: {features[name]}")

    for name in config.numerical:
        if name not in features:
            features[name] = 0.0     # 兜底默认值

    for seq_name, seq_cfg in config.sequence.items():
        if seq_name not in features:
            features[seq_name] = [0] * seq_cfg["max_len"]   # 兜底空序列

# 推理前必须调用一次
features = fetch_features(user_id, item_id, context)
validate_features(features, config)         # ← 这一步保命
pred = model.predict(features)
```

---

### 2.9 特征交叉机制:搜推广和 CV 最本质的区别

> 在 CV 里,特征交叉是"卷积核自动做的事情";在搜推广里,**你必须显式设计特征如何交叉在一起**。
>
> 因为搜推广的特征是离散的、稀疏的、异构的,卷积核没法直接在(user_id, item_id, city) 上滑动。

#### 2.9.1 为什么需要"交叉"?——单一特征的局限性

```python
# 假设只有单独的特征:
# user_age = 25, item_category = "phone", city = "北京"

# 模式 1:"年轻人爱买手机" → user_age < 30 AND category = phone
# 这是 user_age 和 category 的交叉模式

# 模式 2:"北京人都在找什么?" → city = "北京" AND category = "餐饮"
# 这是 city 和 category 的交叉模式

# 如果模型只能看到单个特征(channel-wise),它就学不到这些交叉模式
# 必须显式地把特征向量"放在一起"让模型去交互
```

#### 2.9.2 四种特征交叉机制对比(从简单到强大)

```text
                   ┌──────────────────────────────────────────────────┐
                   │  特征交叉机制的"进化树"                          │
                   │                                                  │
                   │  [Concatenation] 最原始,不加交叉                 │
                   │       ↓                                          │
                   │  [Dot Product(内积)] 双塔/召回                   │
                   │       ↓                                          │
                   │  [Hadamard Product + MLP] DIN/宽深度             │
                   │       ↓                                          │
                   │  [Soft Attention] DIN/DIEN                       │
                   │       ↓                                          │
                   │  [Multihead Self-Attention] Transformer          │
                   └──────────────────────────────────────────────────┘
```

| 机制 | 数学形式 | 参数量 | 表达能力 | 搜推广代表模型 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 拼接(Concat) + MLP** | `y = MLP([u, i, c])` | MLP 权重 | ★★☆ | **Wide&Deep、DeepFM** |
| **2. 内积(Dot Product)** | `s = u · i` | 0(无参数) | ★☆☆ | **双塔召回、FM** |
| **3. 外积(Outer Product)** | `M = u ⊗ i` → `M.flatten()` | 高 | ★★★ | **PNN、DCN** |
| **4. 注意力(Attention)** | `α = softmax(QK^T/√d)` | QKV 投影 | ★★★★ | **DIN、DIEN、SASRec** |
| **5. 自注意力(Self-Attention)** | `α = softmax( XW^Q (XW^K)^T/√d )` | 多头投影 | ★★★★★ | **Transformer** |

```python
# 代码实现对比——四种交叉方式,输入的 user_emb 和 item_emb 都是 [B, D]

import torch
import torch.nn as nn
import torch.nn.functional as F

B, D = 32, 64
user_emb = torch.randn(B, D)
item_emb = torch.randn(B, D)

# 方式 1:Concat + MLP(最简单,最常用)
combined = torch.cat([user_emb, item_emb], dim=-1)  # [B, 2D]
mlp = nn.Linear(2*D, 1)
output = mlp(combined)                               # [B, 1]
# 缺点:MLP 需要自己"学会"交叉,数据少时效果差

# 方式 2:Dot Product(最便宜,召回标配)
score = (user_emb * item_emb).sum(dim=-1, keepdim=True)  # [B, 1],0 参数!
# 优点:计算 0 参数,适合召回阶段(百万级打分)
# 缺点:表达能力有限,只能学到线性相似

# 方式 3:Hadamard Product + Difference + MLP(DIN 核心理念)
interaction = torch.cat([
    user_emb * item_emb,           # 元素积:相似度信号
    user_emb - item_emb,           # 差值:差异信号
    user_emb,                      # 原特征
    item_emb,                      # 原特征
], dim=-1)                         # [B, 4D]
din_mlp = nn.Sequential(
    nn.Linear(4*D, 32),
    nn.PReLU(32),
    nn.Linear(32, 1)
)
output = din_mlp(interaction)      # [B, 1]
# 为什么比 Concat 好?显式提供了"乘法交叉"和"减法差异"信号
# MLP 不需要从头学交叉,直接用这些信号做加权

# 方式 4:Multihead Self-Attention(最强交叉,Transformer)
# 把所有特征的 embedding 堆叠成一个序列
feat_seq = torch.stack([user_emb, item_emb], dim=1)  # [B, 2, D]
attn = nn.MultiheadAttention(D, num_heads=8, batch_first=True)
attn_out, _ = attn(feat_seq, feat_seq, feat_seq)     # [B, 2, D]
output = attn_out.mean(dim=1)                         # [B, D],池化
# 为什么最强?self-attention 让 user 和 item 互相"看对方"
# user 的表示会因 item 的不同而动态变化(DIN 的思想的终极形态)
```

**交叉机制的选型决策树**:

```text
数据量少(<100万)或计算资源紧张?
├── 是 → Double Tower + Dot Product(召回) 或 Concat+MLP(精排)
├── 否 → 问题是否有明显的"用户-物品匹配"特性?
    ├── 是 → DIN 的 Attention 或 FM 的向量内积
    ├── 否 → 特征维度多且异构?
        ├── 是 → DCN Cross 层 + Deep 分支
        ├── 否 → Self-Attention / Transformer
```

#### 2.9.3 为什么 CV 里你不需要关心这些?

```text
CV:
图片像素是 "稠密 + 空间局部 + 层次结构" 的
→ Conv2d 天然做交叉:每个 3×3 卷积核就在做 9 个像素的局部交叉
→ 你不需要"设计交叉",Conv 自动做

搜推广:
特征是 "稀疏 + 全局 + 无空间结构" 的
→ user_id 和 item_id 之间没有"空间邻居"的概念
→ 你必须显式告诉模型"这两个特征要交叉"
```

**所以,当你看到一个搜推广模型(DeepFM/DIN/Transformer)时,问自己一个问题**:
```text
这个模型用的是什么交叉机制?
是 Concat? 是 Dot Product? 是 Attention?
这个交叉机制解决了什么数据特性问题?
```

### 2.10 第二章小结 & 自测

**小结**:
- **数据格式**:Parquet 是工业界首选,CSV 只用于调试
- **离散特征**:LabelEncode → Embedding,绝不用 One-Hot
- **连续特征**:缺失标记 + 归一化 + 训练/推理同 scaler
- **时序特征**:滑动窗口采样 + Padding + Mask
- **数据集划分**:时间切分,绝不随机;离线/在线要一致
- **Dataset/DataLoader**:惰性读取、collate_fn 处理变长
- **特征交叉**:Concat/Dot/Attention/Self-Attention 四种类型,按场景选

**自测题**:
1. 解释为什么搜推广里不能用 One-Hot 表示物品 ID?
2. 给定 `[(1, 100), (1, 200), (2, 100), (2, 300)]` 表示 (user_id, item_id) 的点击,用 `defaultdict` 构造 user → set(items) 字典。
3. 实现一个 `collate_fn`,把变长字符串列表 padding 到 batch 内最大长度,返回 (B, T) 的 LongTensor。
4. 为什么要做"离线/在线特征一致性校验"?举出 2 个常见的不一致场景。
5. 搜推广的训练数据划分为什么不能随机?会导致什么后果?
6. (新增) 四种特征交叉机制(Concat/Dot/Hadamard/Attention)的本质区别是什么?

---

> 如果你已经读到这里,那你已经掌握了"怎么收集、清洗、编码、组织训练数据"的能力。接下来我们进入代码层面——**PyTorch 到底怎么用**。

---

## 第 3 章:PyTorch 基础——从 Tensor 到完整模型

> **本章目标**:不是 pytorch tutorial 的 API 罗列,而是**在搜推广视角下,重新理解 PyTorch 的 Tensor、Autograd、nn.Module、Embedding 等核心机制**。
>
> 读完你能回答:
> - Tensor 的 view / reshape / permute / transpose 底层干了什么(哪个是"改名"、哪个是"搬家")?
> - Autograd 的计算图怎么画?为什么 `.backward()` 一句话就回传梯度?
> - Embedding 的"数学本质"不就是查表吗?那 nn.EmbeddingBag 又是什么?
> - nn.Module 的 parameter / buffer / children 三件套怎么管理?

### 3.1 一切都从 Tensor 开始

#### 3.1.1 Tensor 是什么?——比 NumPy 更强大的数组

```python
import torch

# 从 numpy 创建
import numpy as np
np_arr = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(np_arr)         # 共享内存,np_arr 改 t 也改!
# 千万注意:如果你用 .from_numpy,原 numpy 改了 tensor 也会跟着变!

# 从 list 创建(常用)
t = torch.tensor([[1, 2], [3, 4]])   # dtype=torch.int64
t = torch.tensor([1.0, 2.0])         # dtype=torch.float32

# 特殊初始化
t0 = torch.zeros(3, 4)               # 全 0
t1 = torch.ones(3, 4)                # 全 1
te = torch.empty(3, 4)               # 未初始化(可能存着垃圾值,快但危险)
ti = torch.eye(4)                    # 单位矩阵
tr = torch.randn(3, 4)               # 标准正态分布
tl = torch.linspace(0, 1, 10)        # [0.0, 0.111, ..., 1.0]
```

**搜推广的坑:一定要显式指明 dtype**

```python
# 在搜推广里,存储物品 ID 用 int32 或者 int64
# 如果类型不对,后面会引发隐式类型转换甚至报错
item_ids = torch.tensor([1, 2, 3])               # 默认 int64
item_ids_fp = torch.tensor([1, 2, 3], dtype=torch.float32)  # ⚠️ 不对!后面 Embedding 会报错

# 搜推广常用 dtype:
# - 离散特征 ID: torch.long (nn.Embedding 要求)
# - 连续特征: torch.float32
# - 权重: torch.float32 / float16 / bfloat16
# - 标签: torch.float32 (BCELoss) 或 torch.long (CrossEntropyLoss)
```

**dtype 对比**:

| dtype | 说明 | 什么时候用 |
| :--- | :--- | :--- |
| `torch.float32` | 标准 32 位浮点 | **默认选择**,大多数网络用这个 |
| `torch.float16` | 半精度,2 字节 | 显存不够时,混合精度训练 |
| `torch.bfloat16` | 同半精度但指数范围更大 | 混合精度训练的更好选择 |
| `torch.int64` (long) | 64 位整数 | **Embedding 输入的唯一选择** |
| `torch.int32` | 32 位整数 | 存储 ID 做索引(Embedding 输入仍要转 long) |
| `torch.bool` | 布尔 | mask / attention_mask |

#### 3.1.2 设备(device):CPU vs GPU

```python
# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建时指定设备
t = torch.tensor([1, 2, 3], device="cuda")

# 移动设备(这个操作很慢,尽量少用)
t.to("cuda")             # 拷贝到 GPU
t.to("cpu")              # 拷贝回 CPU

# 模型也一样
model = MyModel().to(device)

# 搜索里的常见写法
batch = {k: v.to(device) for k, v in batch.items()}
```

**搜推广踩坑:GPU 显存 OOM 常发生在设备迁移**

```python
# 每次 .to(device) 都产生一次 CPU→GPU 拷贝,B=4096 时一次拷贝几十 MB
# 如果训练循环里不小心在 CPU 上做了前向... (例如忘了 to(device))
# 再显式搬一次,隐式又多了一次拷贝
# 全局要统一管理:batch 在 DataLoader 就 pin_memory → 到 GPU 只走一次
```

#### 3.1.3 形状(shape):维度操作是搜推广的日常

**搜推广常见的张量形状**:

```python
batch_size = 1024
seq_len = 50
n_features = 32
n_items = 10_000_000

# 离散特征 ID 向量: [B] (每个用户 1 个 ID)
user_ids = torch.randint(0, n_items, (batch_size,))          # [1024]

# 离散特征 ID 序列: [B, T] (用户历史点击序列)
history_ids = torch.randint(0, n_items, (batch_size, seq_len))  # [1024, 50]

# 连续特征矩阵: [B, F] (F 个连续特征)
dense_feats = torch.randn(batch_size, 8)                     # [1024, 8]

# Embedding 后的序列: [B, T, D] (D 是 embedding 维度)
seq_emb = torch.randn(batch_size, seq_len, 32)               # [1024, 50, 32]

# 模型输出: [B] 或 [B, 1] (点击概率)
predictions = torch.sigmoid(torch.randn(batch_size))          # [1024]
```

#### 3.1.4 Tensor 的内存布局:一副扑克牌的"切牌"哲学(view/stride/contiguous)

> **这是你学 PyTorch 最最重要的一个概念,没有之一。** 理解了 stride,你就理解了为什么 view 有时报错、为什么 transpose 是 O(1)、为什么 contiguous() 那么贵。

**Tensor 在内存里不是"多维数组",而是一段连续的一维内存 + 一套"看它的角度"。**

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 内存: [1, 2, 3, 4, 5, 6]  ← 按行优先连续排成一条线
```

**stride = "在每一个维度上,前进 1 步需要跨过几个元素"**

```python
print(x.shape)   # (2, 3)
print(x.stride()) # (3, 1)
# 解读:
#   dim=0(行):从第 0 行到第 1 行,要跨过 3 个元素(即 1,2,3)
#   dim=1(列):从第 0 列到第 1 列,要跨过 1 个元素(即隔壁)
```

**用 stride 来解释所有维度变换操作:**

```python
x = torch.randn(2, 3, 4)       # shape=(2,3,4), stride=(12,4,1)
# 内存:连续排列,从 0~23

# ---- view:只改 shape 和 stride,不改内存 ----
y = x.view(6, 4)               # shape=(6,4), stride=(4,1)
# 内存不变!只是"看"的方式变了:原来 2 层楼,现在当成 6 层楼
# 要求:x 在内存中是连续的(contiguous)

# ---- transpose:交换 stride ----
y = x.transpose(0, 2)          # shape=(4,3,2), stride=(1,4,12)
# 没有复制数据!只是把 stride 重新排列了
# 原来 (12,4,1) → (1,4,12):dim0 前进 1 步只需跨 1 个元素了
# 这就是 transpose O(1) 的原因——只改了"怎么看"

# ---- permute:任意重排 stride ----
y = x.permute(2, 0, 1)         # shape=(4,2,3), stride=(1,12,4)
# 同样 O(1),只改 stride

# ---- contiguous:物理上重新排列,变成连续 ----
y = x.transpose(0, 2)         # 现在 y 的 stride=(1,4,12),非连续!
y = y.contiguous()             # 创建新内存,按 y 的顺序重新排列
# contiguous() 很贵!因为它涉及一次完整的内存拷贝 O(n)
```

**连续(contiguous) vs 不连续(non-contiguous)的判断标准**:

```python
x = torch.randn(2, 3, 4)
print(x.is_contiguous())          # True

y = x.transpose(0, 1)            # (3,2,4)
print(y.is_contiguous())          # False! 因为内存不再是行优先

# 判断公式:从最后一个 dim 到第一个 dim,
# stride[i] == stride[i+1] * shape[i+1]  (对于第 i 维)
# x: stride=(12,4,1), 4 == 1*4 ✓, 12 == 4*3 ✓ → continuous
# y: stride=(4,12,1), 12 != 1*2 ✗ → NOT continuous
```

**搜推广中 stride 最常见的坑**:

```python
# 坑 1:transpose 后想 view,报错!
x = torch.randn(32, 50, 128)            # [B, T, D], continuous
y = x.transpose(1, 2)                   # [B, D, T], non-contiguous
# z = y.view(32, 50*128)                # RuntimeError: view size is not compatible!
# 因为 y 内存不连续,view 不能直接"重排"

# 正确做法:
z = y.contiguous().view(32, -1)          # 先 contiguous(复制),再 view

# 或者用 reshape(自动 contiguous)
z = y.reshape(32, -1)                    # reshape 内部会检查 contiguity,自动执行 contiguous

# 坑 2:在推理循环里反复 transpose + contiguous
# 每次 contiguous 都是 O(n) 拷贝,性能杀手!
# 正确:在循环外一次性 contiguous,循环内只做 view
x = x.transpose(1, 2).contiguous()       # 一次性
for i in range(1000):
    y = x.view(batch_size, -1)           # 循环内只 view,不拷贝

# 坑 3:怀疑显存异常暴涨?
# 检查是不是不小心创建了多次 contiguous 副本
def memory_debug(tensor):
    print(f"shape={tensor.shape}, stride={tensor.stride()}, "
          f"contiguous={tensor.is_contiguous()}, "
          f"storage_size={tensor.storage().size()}")    # 底层存储大小
```

**stride 的物理意义总结**:

| 操作 | 时间复杂度 | 是否复制数据 | 搜推广建议 |
| :--- | :--- | :--- | :--- |
| `view` | O(1) | ❌ 不改内存 | **首选**,但需 contiguous |
| `reshape` | O(1)~O(n) | 不一定 | 安全,自动判断 |
| `transpose` | O(1) | ❌ | 频繁调用,不复制 |
| `permute` | O(1) | ❌ | 多维度重排 |
| `contiguous` | **O(n)** | ✅ 必复制 | 少用,用完就别再用 |
| `flatten` | O(1)~O(n) | 看情况 | 新代码用 `reshape(-1,)` |
| `squeeze/unsqueeze` | O(1) | ❌ | 随便用,免费 |

#### 3.1.5 索引与切片:为什么 Tensor 的索引和 Python list 不一样?

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 基本索引
x[0]            # tensor([0, 1, 2, 3]), 取第 0 行
x[0, 1]         # tensor(1), 取第 0 行第 1 列
x[0, :]         # 同上,和 x[0] 等价
x[:, 1]         # tensor([1, 5, 9]), 取第 1 列

# 布尔索引(搜推广高频:mask)
mask = x > 5
x[mask]         # tensor([6, 7, 8, 9, 10, 11])

# 花式索引(按索引数组取)
indices = torch.tensor([0, 2])
x[indices]      # tensor([[0, 1, 2, 3], [8, 9, 10, 11]])

# 搜推广高频:gather(按索引从序列中取值)
seq = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80]
])  # [2, 4]
idx = torch.tensor([[2], [1]])
seq.gather(dim=1, index=idx)   # tensor([[30], [60]])
# 在注意力机制里,需要从整个序列中按 attention 分位置取信息时常用
```

#### 3.1.5 维度变换:view / reshape / transpose / permute 的区别

**这是你学 PyTorch 最该搞清楚的四个函数之一**。

```python
x = torch.randn(2, 3, 4)    # [2, 3, 4]

# 1. view:返回新视图(shape 变,内存不变,要求内存连续)
y = x.view(2, 12)           # [2, 12],成功
y = x.view(6, 4)            # [6, 4],成功
y = x.view(-1, 4)           # [6, 4],-1 表示自动推算

# 2. reshape:功能同 view,但不要求连续(view 报错时 reshape 可以)
# 一般建议直接用 reshape(更安全,性能差不大的)
y = x.reshape(6, 4)         # 和 view 一样

# 3. transpose:交换两个维度
y = x.transpose(0, 1)       # [3, 2, 4],dim0 和 dim1 互换位置

# 4. permute:任意调整维度顺序(比 transpose 更通用)
y = x.permute(2, 0, 1)      # [4, 2, 3],dim0→原来 dim2,dim1→原来 dim0...

# 5. squeeze / unsqueeze:去掉/增加维度 1
x = torch.randn(1, 3, 1, 4) # [1, 3, 1, 4]
x.squeeze()                 # [3, 4],去掉所有大小为 1 的维度
x.unsqueeze(0)              # [1, 1, 3, 1, 4],在位置 0 插入 1 维
x.unsqueeze(-1)             # [1, 3, 1, 4, 1],在最后插入 1 维
```

**搜推广的经典用法**(Transformer Attention 里每个张量的维度变化):

```python
# 输入: [B, T, D] (batch, 序列长, 特征维)
x = torch.randn(32, 50, 128)

# 多头注意力的 QKV 变换
Wq = torch.randn(128, 128)

# 1. Q = x @ Wq → [B, T, D]
q = x @ Wq

# 2. reshape 成多头: [B, T, nh, hd]
q = q.view(32, 50, 8, 16)         # nh=8, hd=16

# 3. 转置成 [B, nh, T, hd] — 这是 attention 计算的标准格式
q = q.permute(0, 2, 1, 3)         # [32, 8, 50, 16]

# 4. 注意力分数: q @ k^T → [B, nh, T, T]
scores = q @ q.transpose(-2, -1) / (16 ** 0.5)

# 5. 输出: [B, nh, T, hd] → permute → [B, T, nh, hd] → reshape → [B, T, D]
out = scores @ v                             # [32, 8, 50, 16]
out = out.permute(0, 2, 1, 3).reshape(32, 50, 128)  # [32, 50, 128]
```

#### 3.1.6 广播机制(Broadcasting):自动对齐形状

```python
# 关键规则:两个张量从右到左对齐,维度要么相等,要么是 1,要么不存在
# 最终形状 = 逐位取 max

# 例子 1: [3, 1] + [1, 4] → [3, 4]
a = torch.tensor([[1], [2], [3]])      # [3, 1]
b = torch.tensor(`10, 20, 30, 40`)   # [1, 4]
c = a + b                                # [3, 4]
# [[11, 21, 31, 41],
#  [12, 22, 32, 42],
#  [13, 23, 33, 43]]

# 例子 2: 搜推广常见——序列嵌入 + 位置编码
seq_emb = torch.randn(32, 50, 128)   # [B, T, D]
pos_emb = torch.randn(1, 50, 128)     # [1, T, D],广播到 B
output = seq_emb + pos_emb             # [32, 50, 128]

# 例子 3: mask 广播到 emb 维度
mask = torch.tensor(`1, 1, 0, 0`)   # [1, T],B=1
emb = torch.randn(2, 4, 8)            # [B, T, D]
masked = emb * mask.unsqueeze(-1)     # [B, T, 1] → [B, T, D]
```

**搜推广的坑:不可广播的形状**

```python
a = torch.randn(32, 50, 128)          # [B, T, D]
b = torch.randn(32, 128)             # [B, D]
# a + b → 报错!因为 T 维对不上(50 vs 空)
# 正确:用 unsqueeze 变成 [B, 1, D]
b_expanded = b.unsqueeze(1)           # [B, 1, D]
a + b_expanded                        # 广播成 [B, T, D]
```

---

### 3.2 Tensor 运算:从简单到高级

#### 3.2.1 矩阵乘法三剑客(搜推广必考)

```python
# 1. torch.mm:严格 2 维矩阵相乘
# 场景:全连接层的前向计算
x = torch.randn(1024, 128)   # [B, D_in]
w = torch.randn(128, 64)     # [D_in, D_out]
y = torch.mm(x, w)           # [1024, 64]
# ⚠️ mm 只接受 2 维,传入 3 维就报错

# 2. torch.bmm:带 Batch 维的 3 维矩阵相乘(搜推广极度高频)
# 场景:多序列的注意力计算——每个样本独立算各自的 attention
q = torch.randn(32, 8, 50, 16)   # [B, nh, T, hd]
k = torch.randn(32, 8, 16, 50)   # [B, nh, hd, T]
scores = torch.bmm(               # ❌ 4 维,bmm 只接受 3 维
    q.view(-1, 50, 16),           # [B*nh, T, hd]
    k.view(-1, 16, 50)            # [B*nh, hd, T]
).view(32, 8, 50, 50)             # [B, nh, T, T]

# 用 matmul 更简单(推荐):
scores = torch.matmul(q, k)       # [32, 8, 50, 50],自动处理

# 3. torch.matmul:最通用,支持广播,推荐日常用
v1 = torch.randn(32, 50, 128)    # [B, T, D]
v2 = torch.randn(128, 64)        # [D, D_out]
y = torch.matmul(v1, v2)         # [B, T, 64],自动广播

# 搜推广经典场景:双塔模型的内积(向量点击率计算)
user_emb = torch.randn(32, 64)   # [B, D]
item_emb = torch.randn(32, 64)   # [B, D]
# method 1:点积
score = (user_emb * item_emb).sum(dim=-1)  # [B]
# method 2:bmm 矩阵乘法
score = torch.bmm(
    user_emb.unsqueeze(1),       # [B, 1, D]
    item_emb.unsqueeze(-1)       # [B, D, 1]
).squeeze(-1).squeeze(-1)        # [B]
# method 3:matmul
score = torch.matmul(user_emb.unsqueeze(1), item_emb.unsqueeze(-1)).squeeze()
```

**决策表**:

| 函数 | 输入维度 | 输出形状 | 搜推广场景 |
| :--- | :--- | :--- | :--- |
| `torch.mm` | (M,N)×(N,P) | (M,P) | 最终的 MLP 输出层 |
| `torch.bmm` | (B,M,N)×(B,N,P) | (B,M,P) | **注意力的 QK^T / 双塔相似度** |
| `torch.matmul` | 任意 ≥2 维 | 自动推断 | **通用首选** |

#### 3.2.2 归约运算(sum/mean/max/min)

```python
x = torch.randn(32, 50, 128)   # [B, T, D]

x.sum()                         # 所有元素之和(标量)
x.sum(dim=-1)                   # [B, T],沿特征维求和
x.mean(dim=1)                   # [B, D],沿序列维求均值(序列池化)
x.max(dim=1).values             # [B, D],沿序列维取最大值(最大池化)
x.min(dim=-1).values            # [B, T],沿特征维取最小值

# 搜推广:对变长序列求均值(带 mask)
x = torch.randn(32, 50, 128)       # [B, T, D]
mask = torch.ones(32, 50, 1)        # [B, T, 1]
mask[:, 30:, :] = 0                 # 后 20 个是 padding

masked_sum = (x * mask).sum(dim=1)  # [B, D],padding 贡献 0
masked_cnt = mask.sum(dim=1)        # [B, 1],每个样本有效长度
pooled = masked_sum / masked_cnt.clamp(min=1)  # [B, D],平均池化
```

#### 3.2.3 高级索引与 gather/scatter

```python
# torch.gather:从序列中"按索引取出特定位置的值"
# 场景:在 Transformer 输出中,拿到每个样本[CLS]位置的向量
seq_out = torch.randn(32, 50, 128)           # [B, T, D]
cls_indices = torch.zeros(32, 1, dtype=torch.long)  # 每行都是 0([CLS]位置)
cls_vec = seq_out.gather(dim=1, index=cls_indices.unsqueeze(-1).expand(-1, -1, 128))
# cls_vec: [32, 1, 128]

# scatter_:反向操作,把值放回指定位置
# 场景:构造稀疏特征的 Multi-hot 向量
indices = torch.tensor([[0, 2], [1, 3]])      # 每行 2 个位置要填 1
out = torch.zeros(2, 5, dtype=torch.float32)  # [B=2, dim=5]
out.scatter_(dim=1, index=indices, value=1)
# out: [[1, 0, 1, 0, 0],
#       [0, 1, 0, 1, 0]]
```

#### 3.2.4 cat vs stack 再强调一次

```python
# cat:沿已有维度拼接(不新增维度)
a = torch.randn(32, 16)
b = torch.randn(32, 32)
c = torch.cat([a, b], dim=-1)   # [32, 48]

# stack:新增维度后堆叠(每个输入形状必须完全相同)
a = torch.randn(32, 16)
b = torch.randn(32, 16)
c = torch.stack([a, b], dim=1)  # [32, 2, 16]
```

**搜推广经典**:做多特征拼接(cat) vs 做特征交叉(stack):

```python
# 拼接:你有的用户特征全部拼起来送给 MLP
user_emb = torch.randn(32, 32)     # 用户 ID embedding
item_emb = torch.randn(32, 32)     # 商品 ID embedding
history_emb = torch.randn(32, 16)   # 历史序列池化
dense = torch.randn(32, 8)          # 连续特征

all_features = torch.cat([
    user_emb, item_emb, history_emb, dense
], dim=-1)                          # [32, 88] ← 88=32+32+16+8

# stack:用在 FM 的特征交叉层,每个特征向量独立保留
feat_vecs = []
for emb in all_embeddings:
    feat_vecs.append(emb)   # 每个是 [32, D]
feat_matrix = torch.stack(feat_vecs, dim=1)  # [32, n_feats, D]
```

#### 3.2.5 einsum:爱因斯坦求和约定

```python
# einsum 可以用一行代号写出各种矩阵运算,不用去犹豫 permute 和 reshape

# 矩阵乘法:q @ k^T → [B, nh, T, T]
attention = torch.einsum("b n t d, b n s d -> b n t s", q, k)

# 双塔内积:
score = torch.einsum("b d, b d -> b", user_emb, item_emb)

# 带偏置的线性层:
y = torch.einsum("b d, d h -> b h", x, w) + b

# 搜推广学 einsum 的最大理由:
# 别人用 bmm + permute 写 3 行,你 1 行就好,且可读性超强
```

---

### 3.3 Autograd:自动求导引擎深度拆解

> **从 CV 到搜推广,Autograd 是你必须重新理解的概念**。
>
> CV 里你只关心 loss.backward() 能不能跑通;但在搜推广里,显存优化、梯度检查、断点续训、混合精度,全都在 Autograd 这一层展开。
>
> **关键问题**:
> - loss.backward() 到底是怎么找到 W.grad 的?
> - 为什么中间变量的梯度在 backward 后会释放?
> - 梯度是"累加"的——这意味着什么?怎么利用?

#### 3.3.1 计算图的构建:Define-by-Run 机制

**PyTorch 跟 TensorFlow 1.x 最大的区别**:TensorFlow 是先画图再执行(Define-and-Run),PyTorch 是边执行边画图(Define-by-Run)。

```python
# 这条简单的代码:
y = W @ x + b

# 在 PyTorch 内部实际发生的过程:
# 1. ATen 底层 C++ 引擎执行矩阵乘法
# 2. 在 C++ 层的 TensorImpl 中创建一个 AutogradMeta 对象
# 3. AutogradMeta 记录:
#    - grad_fn = <MmBackward0>  (指向反向函数)
#    - next_edges = [W.grad_fn, x.grad_fn]  (指向输入的前向节点)
# 4. 每个中间张量 z 都持有一个 grad_fn 指针
# 5. grad_fn 内部持有 next_functions——指向其输入的 grad_fn 或 AccumulateGrad
```

**所以 PyTorch 的计算图在内存中是一棵"倒挂树"**:

```text
前向传播方向(forward):
W(leaf, grad_fn=None) ───┐
                         ├──→ @ ──→ z ──→ + ──→ y ──→ loss ──→ (输出)
x(leaf, grad_fn=None) ───┘                ↑
b(leaf, grad_fn=None) ────────────────────┘

每个节点实际存储的是"逆链路"(到反向传播方向):
loss.grad_fn.next_functions[0] = y.grad_fn   # (AccumulateGrad)
                                   ↓
y.grad_fn.next_functions = [z.grad_fn.next_functions[0], b.grad_fn]
                            ↓
z.grad_fn.next_functions = [W.grad_fn, x.grad_fn]  # W.grad_fn = AccumulateGrad

# 关键:AccumulateGrad 是叶节点的特殊 grad_fn
# 当反向传播到达 AccumulateGrad 时,梯度被累加到 .grad 属性
```

**叶子节点 vs 非叶子节点**:
```python
# 叶子节点:由用户直接创建的张量 require_grad=True
W = torch.randn(128, 64, requires_grad=True)   # 叶子节点! W.is_leaf = True
b = torch.randn(64, requires_grad=True)         # 叶子节点! b.is_leaf = True

# 非叶子节点:由运算产生的中间张量
z = x @ W                       # 非叶子节点! z.is_leaf = False
y = z + b                        # 非叶子节点! y.is_leaf = False

# 验证
print(W.is_leaf, W.grad_fn)     # True  None  ← 叶子节点的 grad_fn=None
print(z.is_leaf, z.grad_fn)     # False <MmBackward0>  ← 非叶子节点绑定 grad_fn
```

**搜推广里最常见的图构建错误**:

```python
# 错误:在训练循环中不小心创建了新的叶子节点
for epoch in range(10):
    # 正确:用 torch.tensor 包装 input
    input_tensor = torch.tensor(input_data, requires_grad=False)

    # 错误!放到循环外
    # input_tensor = ...

# 错误:对 requires_grad 的混合理解
x = torch.randn(32, 128)                    # requires_grad=False, 叶子
W = torch.randn(128, 64, requires_grad=True) # 叶子

y = (x.detach().clone()) @ W   # 这么写,从 x 克隆的新张量虽然是叶子
                                # 但会创建一条独立的计算图分支
                                # 如果 x 是数据张量,clone 浪费显存
```

#### 3.3.2 `.backward()` 到底做了什么?——链式法则的图遍历算法

**核心算法:从 loss 出发,沿计算图的 DAG 进行拓扑序反向遍历**。

```python
# 直观代码
loss = f(w @ x + b)

# 反向传播在内部等价于:
# 1. 从 loss.grad_fn 开始
# 2. 调用 loss.grad_fn() → 计算 dl/dy, 传给 y.grad
# 3. 调用 y.grad_fn() → 计算 dl/dz 和 dl/db, 传给 z.grad 和 b.grad
# 4. 调用 z.grad_fn() → 计算 dl/dW 和 dl/dx, 传给 W.grad
# 5. 到达 AccumulateGrad → dl/dW 累加到 W.grad
```

**完整跟踪代码——亲手看 Autograd 的每一步**:

```python
import torch

W = torch.randn(128, 64, requires_grad=True)
b = torch.randn(64, requires_grad=True)
x = torch.randn(32, 128)              # requires_grad=False

z = x @ W                              # MmBackward0
y = z + b                              # AddBackward0
loss = y.sum()                         # SumBackward0

print("=== 前向传播后的 grad_fn 链 ===")
print(f"loss.grad_fn: {loss.grad_fn}")          # <SumBackward0>
print(f"loss.grad_fn.next_functions:")
for fn in loss.grad_fn.next_functions:
    print(f"  - {fn[0]}")                        # [(<AddBackward0, 0>)]

print(f"\ny.grad_fn.next_functions:")
for fn in y.grad_fn.next_functions:
    print(f"  - {fn[0]}")                        # <MmBackward0>, <AccumulateGrad>

print(f"\nz.grad_fn.next_functions:")
for fn in z.grad_fn.next_functions:
    print(f"  - {fn[0]}")                        # <AccumulateGrad>, None

# 注意:每个 next_functions 是 (grad_fn, input_index) 的元组
# input_index 表示当前操作的第几个输入需要梯度
```

**输出解读(反映链式求导的递归结构)**:
```text
loss  ──→ SumBackward0 ──→ AddBackward0 ──→ MmBackward0 ──→ AccumulateGrad(W)
                                            └── None(x, 因为没有梯度)
                              └── AccumulateGrad(b)
```

**这就是整个 Autograd 的秘密:`.backward()` 就是在执行一个"沿逆箭头递归调用 grad_fn"的深度优先搜索(DFS)**。

```text
backward(loss):
    # 伪代码
    queue = topological_sort(loss.grad_fn)  # 拓扑排序,确保先算靠近 loss 的
    for fn in reversed(queue):              # 反向遍历
        fn.apply()                          # 调用 grad_fn 的反向函数
```

**梯度形状验证**:
```python
loss.backward()
print(f"W.grad.shape: {W.grad.shape}")   # [128, 64] ← 跟 W 的形状完全一致!
print(f"b.grad.shape: {b.grad.shape}")   # [64] ← 跟 b 的形状完全一致!
# 每个参数的梯度形状 = 参数本身的形状,这是最基本的 sanity check
```

#### 3.3.3 梯度的"累加"特性:是坑也是工具

**这是搜推广里最容易被忽视的 Autograd 特性**:

```python
# 每次 backward(),梯度是累加到 .grad 上,不是覆盖!
# 这意味着:

# 场景 1:忘记 zero_grad(坑)
for step, batch in enumerate(loader):
    loss = model(batch)
    loss.backward()
    # optimizer.step()  ← 忘了写,但损失函数在循环里继续计算下一个 batch
    # 第二次循环:
    #   W.grad 在第一次已经累加了梯度
    #   第二次 loss.backward() 又往 W.grad 上加了一次
    #   累积 10 步后 W.grad *= 10, 梯度爆炸!

# 场景 2:梯度累积(工具——这就是 gradient accumulation 的底层机制)
# 显存不够?分 4 个小 batch 累积梯度,再更新一次
accumulation_steps = 4
optimizer.zero_grad()                   # 全局清 0
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps  # 关键:除以 N! 否则梯度×N
    loss.backward()                      # 梯度累加到 .grad

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                 # 用累积的梯度更新
        optimizer.zero_grad()            # 更新完清 0
```

**梯度累积的数学推导**:
```text
假设原始 batch_size = 1024, 显存不够.
拆成 4 个 mini-batch, 每个 batch_size = 256.

梯度累积:
   total_grad = 0
   for i in range(4):
       loss_i = L_i / 4
       loss_i.backward()    # total_grad += grad(L_i) / 4
   # total_grad = Σ grad(L_i) / 4 = grad(Σ L_i / 4)
   # 由于 L_i 是独立 batch 的损失,Σ L_i / 4 = L(总 batch 的近似)

如果忘记除以 N:
   total_grad = Σ grad(L_i) = N * grad(L_avg)
   total_grad *= N, 学习率必须除以 N 来补偿!
```

**利用梯度做调试**(搜推广 debug 神器):

```python
# 监控梯度,判断训练是否健康
def log_gradient_stats(model: nn.Module, log_every=100):
    """输出梯度统计,帮助判断是否需要调整学习率或裁剪"""
    total_norm = 0.0
    max_grad = 0.0
    min_grad = 0.0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, param.grad.max().item())
            min_grad = min(min_grad, param.grad.min().item())
            param_count += 1

    total_norm = total_norm ** 0.5

    if param_count > 0:
        logger.info(
            f"梯度的全局 L2 范数: {total_norm:.4f} | "
            f"最大值: {max_grad:.6f} | 最小值: {min_grad:.6f}"
        )
        # 经验阈值:
        # total_norm < 1e-4: 梯度消失,模型不学习!
        # total_norm > 10: 梯度爆炸,需要裁剪或降 lr
        # total_norm ~ 0.1~1: 健康范围
```

#### 3.3.4 不求导上下文:何时该用 `torch.no_grad()` 和 `.detach()`

**这个选择的本质是:告诉 Autograd"这一段不需要画进计算图"**。

**`torch.no_grad()` vs `.detach()` 的区别**:

| 维度 | `torch.no_grad()` | `.detach()` |
| :--- | :--- | :--- |
| 作用范围 | **上下文管理器**,影响其中所有操作 | **张量方法**,只影响该张量 |
| 适用场景 | **推理评估**,不需要整个代码块参与求导 | **分离特定张量**,其余正常求导 |
| 是否重建图 | 完全不建图 | 从该张量处切断 |
| 搜推广用法 | `@torch.no_grad()` 装饰 validate 函数 | Teacher-student 蒸馏 |

```python
# 搜推广高频场景 1:验证集评估——用 no_grad
@torch.no_grad()               # 装饰器,整个函数不求导,省显存 + 加速
def validate(model, loader):
    """搜推广标准验证函数:不求导,省显存,加速"""
    for batch in loader:
        logits = model(batch)   # 内部不创建任何 grad_fn
        # ...计算指标...
    # 不用 no_grad:每个前向都创建计算图 → OOM 死亡

# 搜推广高频场景 2:特征提取——用 no_grad 或 detach
with torch.no_grad():
    item_embs = item_encoder(item_ids)    # 预提取所有商品的 embedding
    # 这些 embedding 后续作为静态特征使用

# 搜推广高频场景 3:Teacher-Student 蒸馏——必须用 detach
teacher_logits = teacher_model(x)
# teacher_logits 是计算图的一部分(跟踪了 teacher 的梯度)
teacher_probs = torch.softmax(teacher_logits, dim=-1)

# 我们要把 teacher_probs 作为"固定的监督信号"传给 student
# 所以必须 detach,不让梯度回流到 teacher
student_logits = student_model(x)
loss = F.kl_div(student_logits, teacher_probs.detach())
# 如果忘记 detach:teacher 也会被更新!而且 teacher 参数没在优化器里,梯度无处可去 → 报错!
```

#### 3.3.5 `retain_graph` 和梯度释放策略

```python
# PyTorch 默认行为:backward() 后,计算图的中间梯度被释放(节约显存)
# retain_graph=True:保留计算图

# 场景 1:多次 backward(搜推广极少用)
loss1 = model1(x)
loss2 = model2(x)

# 默认:loss1.backward() 会释放计算图
# loss2.backward() 会报错:计算图已被释放
# 方案:
loss1.backward(retain_graph=True)  # 保留图
loss2.backward()                    # 复用
optimizer.step()                    # 更新

# 场景 2:多层 Loss 叠加(搜推广有用!)
# 在 DIEN 中,主损失 + 辅助损失需要同时回传
ctr_loss = F.binary_cross_entropy_with_logits(ctr_logits, ctr_labels)
aux_loss = compute_auxiliary_loss(...)   # DIEN 的辅助损失

total_loss = ctr_loss + 0.5 * aux_loss
total_loss.backward()                    # 一次 backward 搞定
# 不需要 retain_graph,因为 total_loss 在一个图上
```

#### 3.3.6 Autograd 在搜推广中的显存陷阱

```python
# 陷阱 1:推理时忘了 no_grad——GPU OOM 的元凶
def validate(model, loader):
    for batch in loader:
        # 错误:没加 no_grad
        logits = model(batch)     # ← 这里会创建计算图!
        val_loss = loss_fn(logits, labels)
        # val_loss.backward()    # 虽然没有 backward,但计算图已经建了!
        # 10 个 batch 后:每个 batch 的计算图都在内存里! OOM!

# 正确做法
@torch.no_grad()
def validate(model, loader):
    for batch in loader:
        logits = model(batch)     # 不创建图,省 50~70% 显存

# 陷阱 2:loss 计算图被意外保留
for batch in loader:
    logits = model(batch)
    loss = loss_fn(logits, labels)
    loss.backward()
    # 忘记:optimizer.step() 和 optimizer.zero_grad()
    # 下次循环:新的 loss 又创建计算图,旧的没释放
    # 显存持续增长 → OOM

# 陷阱 3:复杂模型中的计算图链路过长
# 搜推广的 Transformer + 多层 Attention
# 计算图的深度可能达到几千层
# backward 时 DFS 遍历的开销 ≈ O(深度)
# 如果发现每个 step 耗时比预期长 2 倍,检查 model 是不是太深了
```

#### 3.3.7 Autograd 的 hook 机制:调试与修改的瑞士军刀

```python
# torch.Tensor.register_hook:在反向传播时"拦截"梯度,做自定义操作

# 场景 1:打印梯度(调试)
def print_grad(grad):
    print(f"梯度值: {grad.norm():.4f}, 形状: {grad.shape}")
    return grad

emb_output = embedding_layer(user_ids)
emb_output.register_hook(print_grad)    # 每次 backward 时打印

# 场景 2:梯度裁剪——针对特定层(搜推广常用)
def clip_embedding_grad(grad, max_norm=0.1):
    """对 Embedding 层的梯度做更严格的裁剪"""
    return torch.clamp(grad, -max_norm, max_norm)

# 只裁剪 user_embedding 的梯度
model.user_embedding.weight.register_hook(clip_embedding_grad)

# 场景 3:用 hook 实现 Gradient Checkpointing(省显存)
# 原理:在 forward 时扔掉中间激活,backward 时重新计算
# 这是 PyTorch 官方实现,不是自己写 hook
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(emb, linear):
    # 中间结果不在显存中持久化,backward 时重新算
    return linear(emb)

output = checkpoint(forward_with_checkpoint, emb_out, linear_layer)
# 效果:显存减少 30~50%,时间增加约 20%
```

---

### 3.4 nn.Module:模型的"积木"

#### 3.4.1 nn.Module 的三层结构

每个 `nn.Module` 在内部维护三个字典:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 32)    # → _parameters 和 _modules
        self.linear = nn.Linear(32, 16)      # → _parameters 和 _modules
        self.register_buffer(                # → _buffers
            "running_mean",
            torch.zeros(16)
        )

    def forward(self, x):
        return self.linear(self.emb(x))

model = MyModel()

# 查看三件套:
print("参数:", list(model.parameters()))      # 所有可学习参数
print("buffer:", list(model.buffers()))       # 所有 buffer(running_mean 等)
print("子模块:", list(model.children()))      # 直接子模块(emb, linear)
print("全部子模块:", list(model.modules()))    # 递归子模块

# .to(device) / .train() / .eval() 会自动应用到所有三层
model.to("cuda")        # parameters + buffers + children 全部移到 GPU
model.train()            # 所有 BN/Dropout 切到训练模式
model.eval()             # 所有 BN/Dropout 切到推理模式
```

**核心区别**:

| 存储 | 说明 | 是否参与 | .to(device) | state_dict |
| :--- | :--- | :--- | :--- | :--- |
| `_parameters` | 模型权重 | ✅ 自动计算梯度 | ✅ 自动迁移 | ✅ 保存 |
| `_buffers` | 运行统计量(BN mean/var) | ❌ 无梯度 | ✅ 自动迁移 | ✅ 保存 |
| `_modules` | 子模块 | 递归处理 | ✅ 自动迁移 | ✅ 递归保存 |
| 普通属性(如 `self.lr=1e-3`) | — | ❌ | ❌ | ❌ |

#### 3.4.2 自定义层:继承 nn.Module

```python
class MyDenseLayer(nn.Module):
    """自定义前馈层,含 batch_norm + dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)          # 训练时用 batch 统计,推理时用 running
        x = torch.relu(x)
        x = self.dropout(x)
        return x

# 使用
layer = MyDenseLayer(128, 64)
x = torch.randn(32, 128)
y = layer(x)                     # [32, 64],像函数一样调用
```

#### 3.4.3 Sequential 和 ModuleList:自动化 forward

```python
# nn.Sequential:一串层自动按顺序执行
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# nn.ModuleList:存储多个子模块,手动控制 forward(用于 Expert 等场景)
class Expert(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, n_experts, in_dim, out_dim):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(in_dim, out_dim) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(in_dim, n_experts)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=-1)  # [B, n_experts]
        outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, n_experts, D]
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)  # [B, D]
```

---

### 3.5 Embedding 深度解析(搜推广的命脉)

#### 3.5.1 nn.Embedding:数学本质 vs 工程本质

```python
# 定义:1000 万个物品,每个用 32 维向量表示
emb = nn.Embedding(num_embeddings=10_000_000, embedding_dim=32)
print(emb.weight.shape)            # [10_000_000, 32] — 1000 万 × 32 的大矩阵
# 参数量:10_000_000 × 32 = 320_000_000(3.2 亿),占 1.28GB (FP32)

# 数学本质:10_000_000 × 32 矩阵 × one-hot 向量
# 但这样算太慢

# 工程本质:按索引直接查表
indices = torch.tensor([123, 456, 789])    # 3 个物品 ID
vectors = emb(indices)                       # [3, 32] — 直接取第 123,456,789 行
```

**Embedding 的初始化**:

```python
# 默认:从 N(0,1) 采样
emb = nn.Embedding(1000, 32)

# 带 padding_idx:位置 0 的向量永远保持 0,不参与梯度更新
emb = nn.Embedding(1000, 32, padding_idx=0)

# 导入预训练权重
pretrained = torch.randn(1000, 32)
emb = nn.Embedding.from_pretrained(pretrained, freeze=True)  # freeze:不更新
emb = nn.Embedding.from_pretrained(pretrained, freeze=False) # 可微调
```

#### 3.5.2 搜推广的 Embedding 规模挑战

```python
# 假设模型有:
user_emb = nn.Embedding(1_000_000_000, 32)   # 10 亿用户 × 32 = 128GB 😱
item_emb = nn.Embedding(500_000_000, 32)      # 5 亿物品 × 32 = 64GB
# 合计 192GB,一张 A100(80GB) 放不下!

# 工业级解法:
# 1. 大模型拆成小模型 + 共享 Embedding 层
# 2. 哈希分桶:多个 ID 映射到同一桶
# 3. 特征选择:只保留高频 ID
# 4. 分布式参数字典:参数服务器(Parameter Server)把 Embedding 放内存
```

**Embedding 训练的稀疏更新**:

```python
# 关键特性:一个 batch 512 个样本,每个只有 1 个 user_id
# 10 亿用户的 Embedding 表,这个 batch 只更新 512 行!
# 其他 9.999995 亿行都保持不动

# 普通优化器(Adam/SGD)的 weight_decay 会更新全部 10 亿行 → 灾难
# 解法:用 torch.optim.SparseAdam(只更新被索引到的行)
optimizer = torch.optim.SparseAdam(
    model.user_emb.parameters(),
    lr=0.001
)
# 或者对 Embedding 参数单独设置 weight_decay=0
```

#### 3.5.3 nn.EmbeddingBag:变长多值特征的克星

```python
# 场景:用户历史点击了 n 个物品,n 可能是 3、10、50
# 传统做法:Embedding + 手动 Pooling(前文已经讲过)
# 更高效:EmbeddingBag 把"Embedding 查表"和"Pooling"合并为 1 个算子

# 输入:展平的序列 + 每个样本的起始偏移
input_ids = torch.tensor([1, 2, 3, 4, 5, 6])   # 6 个 ID,展平
offsets = torch.tensor([0, 2, 5])               # 样本 1: [0,2), 样本 2: [2,5), 样本 3: [5,6)

emb_bag = nn.EmbeddingBag(
    num_embeddings=1000,
    embedding_dim=32,
    mode='mean',                                # 'mean' / 'sum' / 'max'
    padding_idx=0
)
output = emb_bag(input_ids, offsets)            # [3, 32]
# 等价于:Embedding + 对每个样本的向量做 mean pooling

# 对比 EmbeddingBag 和 Embedding+手动pooling:
# EmbeddingBag:更省内存(不用存中间 emb),更快(底层融合算子)
```

#### 3.5.4 位置编码(Positional Encoding)

```python
# 在搜推广的 Transformer 序列模型里,必须加位置编码(否则模型不知道输入顺序)

# 方案 1:可学习位置编码(最常用)
pos_emb = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
x = x + pos_emb(positions)

# 方案 2:静态 Sinusoidal(不用学,Transformer 原文的方案)
def sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [max_len, D]
```

---

### 3.6 常见网络层

#### 3.6.1 全连接层(nn.Linear)

```python
# y = x @ W^T + b
linear = nn.Linear(in_features=128, out_features=64, bias=True)
print(linear.weight.shape)    # [64, 128]
print(linear.bias.shape)      # [64]

x = torch.randn(32, 128)
y = linear(x)                  # [32, 64]

# 搜推广:多层 MLP 塔
mlp = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
```

#### 3.6.2 归一化层:为什么现代 GenRec 和 LLM 都选择 RMSNorm?

> **从 BatchNorm → LayerNorm → RMSNorm,归一化函数的进化是 GenRec+LLM 趋同最典型的例子。**
>
> 核心问题:让深层网络的梯度保持在合理范围内,避免梯度消失/爆炸。

**四种归一化的数学公式对比**:

```python
# BatchNorm(BN):沿 batch 维归一化
# 公式: y = γ * (x - μ_B) / √(σ²_B + ε) + β
#   其中 μ_B = 1/B * Σ x_i  (在一个 batch 内算)
#        σ²_B = 1/B * Σ (x_i - μ_B)²
#
# 问题:推理时 B=1,无法算 batch 统计量 → 改用训练时累积的 running_mean

# LayerNorm(LN):沿特征维归一化
# 公式: y = γ * (x - μ_L) / √(σ²_L + ε) + β
#   其中 μ_L = 1/D * Σ x_d  (在单个样本内,对特征维算)
#        σ²_L = 1/D * Σ (x_d - μ_L)²
# 优点:推理时 batch_size=1 也没问题
# 缺点:需要算均值 μ 和方差 σ²,各一次 reduce 操作

# RMSNorm(LN 去掉均值):只算 RMS,不算均值——现代 LLM/GenRec 标配!
# 公式: y = γ * x / √(mean(x²) + ε)
# 去掉 μ:只做"除以 RMS"的归一化,不做"减均值"
# 为什么可以去掉 μ?因为 x - μ 和 x 的 RMS 差距很小,去掉 μ 减少计算量

# LayerScale(残差缩放):在残差连接后乘一个可学习标量
# y = x + α * Sublayer(x) 其中 α 初始化为 0.1~0.001
# 用于深层 Transformer(> 24 层)的稳定训练
```

**选型决策树(搜推广+GenRec)**:

```text
搜索推荐场景,该用哪种归一化?
│
├── 模型是传统 MLP(DeepFM/DCN)?
│   └── 选 LayerNorm(不是 BatchNorm!)
│
├── 模型是 Transformer / GenRec?
│   ├── < 1B 参数 → RMSNorm(simpler = better)
│   ├── 1B~10B 参数 → RMSNorm + Pre-Norm
│   └── > 10B 参数 → RMSNorm + Pre-Norm + 小心初始化
│
├── 模型是纯 CNN / 视觉模型?
│   └── 选 BatchNorm(图像里 BN 是标配)
│
└── 模型是多模态(图像+文本+ID)?
    └── 用 RMSNorm 统一(而不是不同模态各用各的)
```

**RMSNorm 为什么赢了?——数学直觉**:

```text
假设一个特征向量 x = [1.0, -0.5, 0.3, ...],d_model=4096

LayerNorm 做的事:
  1. 算均值 μ = sum(x) / 4096 ≈ 0.002
  2. 算方差 σ² = sum((x-μ)²) / 4096 ≈ 1.2
  3. 减均值: x' = x - 0.002  (几乎没变!)
  4. 除标准差: x'' = x' / √1.2

RMSNorm 做的事:
  1. 算 RMS = sqrt(mean(x²)) ≈ 1.1
  2. 除以 RMS: x' = x / 1.1

差别:
  LayerNorm 多了一步"减均值",但均值在 d=4096 时很接近 0
  所以去掉均值几乎不影响归一化效果,但节省了一次全局通信

节省的计算量:
  LayerNorm: sum(x) + sum(x²) + sub + div + γ×x + β = 2 次 reduce
  RMSNorm:  mean(x²) + div + γ×x = 1 次 reduce(少了一半!)

  在训练中,reduce 操作涉及 GPU 全局同步,减半意味着训练速度提升约 5~10%
```

```python
# RMSNorm 的自实现和 PyTorch 结果对比

class RMSNorm(nn.Module):
    """
    RMSNorm: LLaMA/Mistral/Qwen/DeepSeek/GenRec 的标准归一化。
    比 LayerNorm 快约 10~15%,效果等价(甚至略好)。
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习缩放 γ
        self.eps = eps

    def forward(self, x):
        # x: [B, T, D] 或 [B, D]
        # 计算 RMS: sqrt(mean(x²))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 归一化 → 缩放
        return x / rms * self.weight

# 验证和 LayerNorm 的差异
x = torch.randn(32, 50, 256)
ln = nn.LayerNorm(256)
rms = RMSNorm(256)

y_ln = ln(x)
y_rms = rms(x)
print(f"LayerNorm output norm: {y_ln.norm(dim=-1).mean():.4f}")   # ≈ 1.0
print(f"RMSNorm output norm:  {y_rms.norm(dim=-1).mean():.4f}")   # = √d_model ≈ 16.0
# 注意:LN 的输出范数恒 ~1.0;RMSNorm 的输出范数是 γ
# 但两者在经过后续线性层后,效果等价
```

**Pre-Norm vs Post-Norm:放在哪里归一化?**:

```text
Post-Norm(原始 Transformer):
  output = LayerNorm(x + Attention(x))
  梯度从 LN 传入残差 → 梯度不稳定
  深层(> 12 层)训练困难

Pre-Norm(所有现代模型):
  output = x + Attention(LayerNorm(x))
  梯度直接从残差跳过,不经过 LN
  深层(100+ 层)训练稳定

Pre-Norm 是 GenRec 和所有 LLM 的标准选择!
```

**搜推广选型终极规则**:

```python
def choose_normalization(model_type, has_sequence=False, gpu_count=1):
    """
    GenRec 归一化选型规则:

    - MLP 模型(DeepFM/DCN): LayerNorm
    - Transformer 模型: RMSNorm + Pre-Norm
    - 多 GPU 训练: SyncBatchNorm(如果用 BN)/RMSNorm 天然支持
    - 量化推理: RMSNorm 比 LayerNorm 更友好(少一次减均值)
    """
    if model_type in ["transformer", "genrec", "llm"]:
        return "RMSNorm + Pre-Norm"   # 当前唯一推荐
    elif model_type in ["dnn", "deepfm", "din", "mlp"]:
        return "LayerNorm"            # 传统模型够用
    elif model_type == "cnn":
        return "BatchNorm"            # 视觉模型
    else:
        return "RMSNorm"              # 通用兜底
```

#### 3.6.3 Dropout:随机丢弃神经元

```python
dropout = nn.Dropout(p=0.2)      # 20% 概率丢弃
x = torch.randn(32, 64)
y = dropout(x)                    # 训练时:20% 变 0,剩余 80% 乘以 1/(1-p)=1.25
# 推理时:dropout 直接恒等,什么都不做

# 搜推广:Embedding 后常加 Dropout
self.dropout = nn.Dropout(0.1)
def forward(self, x):
    x = self.emb(x)
    x = self.dropout(x)           # 防止 Embedding 过拟合
    ...
```

#### 3.6.4 激活函数:从 ReLU 到 SwiGLU 的进化

> **在 GenRec 和 LLM 趋同的背景下,激活函数的选择直接决定了模型的表达能力和训练稳定性。**

**传统激活函数(搜推广传统模型用)**:

```python
# ReLU:传统 MLP 默认首选(简单、快、好)
relu = nn.ReLU()
# 缺点:Dead ReLU(负值梯度为 0 → 神经元死亡)
# 搜推广里 Embedding 大,死亡 ReLU 很少发生

# LeakyReLU / PReLU:ReLU 的改进,负值有小梯度
leaky = nn.LeakyReLU(negative_slope=0.1)
prelu = nn.PReLU(num_parameters=1)         # 可学习斜率

# GELU:Google 提出,NLP/Transformer 标准
gelu = nn.GELU()
# 公式: GELU(x) = x · Φ(x)  (Φ = 正态分布 CDF)
# 近似: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**现代激活函数:GLU 变体——当前所有现代 LLM 和 GenRec 的选择**:

```python
# GLU(Gated Linear Unit) 的核心思想:
# 不是直接把输入通过一个激活函数,而是"用一个门控控制信息流"

# FFN(传统前馈): output = activation(x @ W₁) @ W₂
# GLU(门控前馈):   output = (x @ W₁) ⊗ sigmoid(x @ V) @ W₂   ← 多了一个"门"
# 或用 SiLU 替代 sigmoid: → SwiGLU

# ============ SwiGLU(LLaMA/Mistral/Qwen/DeepSeek 全部在用) ============
# 公式: SwiGLU(x) = (x @ W_gate) ⊗ SiLU(x @ W_up) @ W_down
# 其中 SiLU = x * sigmoid(x),也叫 Swish

class SwiGLU(nn.Module):
    """
    SwiGLU FFN = 当前最主流的激活 + FFN 结构
    使用方: LLaMA 2/3, Mistral, Qwen 2, DeepSeek, Gemma

    相比标准 FFN:
    - FFN:   out = activation(x @ W₁) @ W₂        参数量 = 2 × d × d_ff
    - SwiGLU: out = (x @ W_gate) ⊗ SiLU(x @ up) @ down  参数量 = 3 × d × d_ff
    SwiGLU 多了一个投影,但通常 d_ff_swiglu = 2/3 × d_ff_ffn,总参数量相近
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)  # 门控
        self.up   = nn.Linear(d_model, d_ff, bias=False)  # 上投影
        self.down = nn.Linear(d_ff, d_model, bias=False)  # 下投影

    def forward(self, x):
        # gate: 控制"多少信息通过"
        # up:   提供"信息的内容"
        # SiLU(gate) ⊗ up: 门控筛选信息
        # down: 投影回原维度
        return self.down(F.silu(self.gate(x)) * self.up(x))

# ============ GeGLU(GELU + GLU) ============
# 公式: GeGLU(x) = (x @ W_gate) ⊗ GELU(x @ W_up) @ W_down
# 比 SwiGLU 少用一些(主要在 T5 和 PaLM 里)

class GeGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))
```

**六种激活/FNN 结构的参数量和效果对比**:

| 结构 | 参数量(以 d=d_model, d_ff=4d 计算) | 代表模型 | 效果排名(经验) |
| :--- | :--- | :--- | :--- |
| **ReLU FFN** | 2 × d × 4d = **8d²** | 早期 Transformer | ★★☆ |
| **GELU FFN** | 2 × d × 4d = **8d²** | BERT, GPT-2 | ★★★ |
| **SwiGLU** | 3 × d × (8/3)d = **8d²** | **LLaMA/Mistral/Qwen** | ★★★★★ |
| **GeGLU** | 3 × d × (8/3)d = **8d²** | PaLM, T5 | ★★★★ |
| **ReGLU** | 3 × d × (8/3)d = **8d²** | 实验性 | ★★★ |
| **GLU(Sigmoid)** | 3 × d × (8/3)d = **8d²** | 早期 GLU | ★★★ |

**关键观察:SwiGLU 和 ReLU FFN 的参数量几乎相同**(因为 SwiGLU 的 d_ff 通常设为 (8/3)d ≈ 2.67d,而 ReLU FFN 的 d_ff=4d)。

```python
# 参数量验证
d_model = 512
# ReLU FFN: d_ff = 2048(4d)
relu_params = 2 * d_model * 2048    # 2,097,152
# SwiGLU: d_ff = 1365(≈ 8d/3)
swiglu_params = 3 * d_model * 1365  # 2,096,640
# ≈ 相同!但 SwiGLU 效果更好
```

**为什么 SwiGLU 比 ReLU/GELU 更好?——梯度流动分析**:

```text
ReLU:  y = max(0, x)      梯度: 0(负) 或 1(正)
GELU:  y = x · Φ(x)       梯度: Φ(x) + x · φ(x)  (连续、非 0)
SwiGLU:y = gate ⊗ SiLU(up)
      梯度: ∂y/∂gate = SiLU(up)  (不是 0!)
            ∂y/∂up = gate · SiLU'(up)  (gate 在控制!)

=SwiGLU 的门控特性=:即使输入 gate 是负值,SiLU(up) 也可能非 0
  gate 机制:门控选择性通过信息 → 比"直接激活"更灵活
  实验表明:SwiGLU 在 2/3 的参数量下能达到和 ReLU FFN 相近的效果
```

**激活函数选型规则(搜推广 + GenRec 场景)**:

```python
def choose_activation(model_type, layer_type):
    """
    选型规则:
    - 传统推荐模型(DeepFM/DIN): ReLU 足够,SwiGLU 收益不大
    - GenRec/Transformer 推荐: SwiGLU 首选
    - 输出层 CTR/Sigmoid: 永远是 Sigmoid
    - 输出层多分类: 永远是 Softmax
    """
    if layer_type == "output_ctr":
        return "Sigmoid"
    elif layer_type == "output_multi":
        return "Softmax"
    elif model_type in ["transformer", "genrec"]:
        return "SwiGLU"                   # 现代 GenRec 标配
    elif model_type in ["dnn", "deepfm", "din"]:
        return "ReLU"                     # 传统推荐足够
    else:
        return "GELU"                     # 兜底
```

#### 3.6.5 损失函数

| 损失函数 | 公式 | 输入形状 | 搜推广场景 |
| :--- | :--- | :--- | :--- |
| `BCEWithLogitsLoss` | `-(y·logσ(ŷ)+(1-y)·log(1-σ(ŷ)))` | `pred: [B], label: [B]` | **CTR 预估(二分类)** |
| `BCELoss` | 同上但输入已 sigmoid | `pred: [B], label: [B]` | 输出已有 sigmoid 时 |
| `CrossEntropyLoss` | `-∑y·log(softmax(x))` | `pred: [B, C], label: [B]` | 多分类(物品类别) |
| `MSELoss` | `(y - ŷ)²` | `pred: [B, 1], label: [B, 1]` | 回归(时长、价格) |

```python
# CTR 预估的正统写法:
loss_fn = nn.BCEWithLogitsLoss()           # 内部自带 sigmoid,数值稳定
# 不用: nn.BCELoss() 因为外面要手动加 Sigmoid,容易数值溢出

logits = model(x)                           # 原始 logits(未 sigmoid)
loss = loss_fn(logits.view(-1), labels.view(-1))  # [B] → 标量 loss
```

---

### 3.7 优化器与学习率调度

#### 3.7.1 优化器选型

```python
# 1. Adam(搜推广默认优化器,没有之一)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),           # 动量 + 自适应学习率
    eps=1e-8,
    weight_decay=0                # 搜推广:Embedding 不要 weight_decay!
)

# 2. AdamW(Adam + 修正的 weight decay,比 Adam 更好)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# 3. SparseAdam(专门用于 Embedding 的稀疏更新)
# ⚠️ 必须把 Embedding 参数单独拎出来
optimizer = torch.optim.SparseAdam(
    model.emb.parameters(),
    lr=1e-2                        # Embedding 学习率通常比 MLP 大
)
```

**搜推广的"分层学习率"**:不同层用不同的学习率

```python
# Embedding 层:高学习率(因为稀疏更新,需要更大的步长)
# MLP 层:低学习率(稠密更新,步长太大梯度爆炸)

optimizer = torch.optim.AdamW([
    {"params": model.emb.parameters(), "lr": 1e-2, "weight_decay": 0},
    {"params": model.mlp.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
    {"params": model.output.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
], lr=1e-3)
```

#### 3.7.2 学习率调度

```python
# 1. StepLR:每 N 个 epoch 衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# epoch 0: lr=1e-3, epoch 5: lr=1e-4, epoch 10: lr=1e-5

# 2. CosineAnnealing:余弦衰减(默认推荐)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. Warmup + Cosine(Transformer/Search 推荐标准)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(step / warmup_steps, 1.0)   # warmup
)

# 使用
for epoch in range(10):
    for batch in loader:
        ...
        optimizer.step()
        scheduler.step()               # 每个 step 更新一次学习率
```

---

### 3.8 混合精度训练:提速 2~3 倍

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()                    # 梯度缩放,防止 FP16 underflow

for batch in loader:
    optimizer.zero_grad()

    with autocast():                     # 自动选择 FP16/FP32
        logits = model(batch)
        loss = loss_fn(logits, labels)

    scaler.scale(loss).backward()        # 缩放梯度
    scaler.unscale_(optimizer)           # (可选)梯度裁剪前需要 unscale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    scaler.step(optimizer)               # 更新参数
    scaler.update()                      # 更新缩放因子
```

**效果**:显存减少 40~50%,速度提升 2~3 倍,**精度无损**(大部分场景)。

---

### 3.9 torch.compile:一键加速(>=2.0)

```python
# PyTorch 2.0+ 新特性:JIT 编译,几行代码提速 20~50%
model = MyModel()
model = torch.compile(model)              # 首次调用时编译,后续加速

# 或
@torch.compile
def forward(self, x):
    ...
```

**注意**:compile 首次调用极慢(几十秒),但第二个 batch 开始就快了。

---

### 3.10 第三章小结 & 自测

**小结**:
- **Tensor**是核心数据结构,view/reshape/permute 的区别必须吃透
- **计算图**是 Autograd 的基础,`.backward()` 沿 `grad_fn` 反向传播
- **nn.Module** 三件套:parameter / buffer / children
- **Embedding** = 查表操作,在搜推广里参数量占绝对大头
- **LayerNorm** 是搜推广的默认归一化选择,BatchNorm 几乎不用
- **混合精度训练**用 `amp.autocast` + `GradScaler`
- **分层学习率**是搜推广的训练标配

**自测题**:
1. `a = torch.randn(3,4); a.view(2,6)` 和 `a.reshape(2,6)` 有什么区别?什么时候 view 会报错?
2. 实现一个函数:给定 `[B, T, D]` 的序列和 `[B]` 的有效长度,返回 mask 后的平均池化向量 `[B, D]`。
3. 为什么搜推广绝不用 BatchNorm?LayerNorm 和 BatchNorm 在计算均值/方差时的维度有什么不同?
4. 写一个带学习率 warmup 的优化器配置,前 1000 步从 0 线性升到 1e-3。
5. 写出 `nn.Embedding(10000, 64)` 的参数量和内存占用(FP32)。

---

> 如果前 3 章是"零件",那第 4 章就是"组装"——把 Python 数据结构、数据处理 Pipeline、PyTorch Tensor 和 Embedding 全部串起来,构建一个完整、可运行的搜推广训练数据生成流程。

---

## 第 4 章:从 0 构建搜推广训练数据(生成式推荐完整 Pipeline)

> **本章目标**:用前 3 章学到的所有知识,从零实现一个工业级的搜推广训练数据生成 Pipeline。这不是"理论知识",这是**可以直接跑、直接用的生产级代码**。
>
> 读完本章,你能:
> - 从原始曝光日志出发,完整构造 CTR 训练样本(正 + 负)
> - 处理用户行为序列的采样、填充、掩码
> - 实现多特征类型的编码与拼接
> - 写出可分布式运行的 DataLoader
> - 理解"生成式推荐"的范式——它其实不是"生成模型",而是"用 token 化把推荐变成序列生成问题"

### 4.1 搜推广训练数据的"生命全周期"

一条训练样本从原始日志到模型消费,需要经过:

```text
原始曝光日志(用户看到了什么)
    ↓
原始点击/转化日志(用户点了什么)
    ↓
日志 Join(把曝光和点击对齐,打标签)
    ↓
特征拼接(拉取用户画像、物品属性、上下文特征)
    ↓
用户行为序列构建(按时间排序,滑动窗口切片)
    ↓
负采样(补充未点击的负样本)
    ↓
编码(LabelEncode / Normalize / Padding)
    ↓
Dataset 包装(变成 PyTorch Dataset)
    ↓
DataLoader 加载(多进程/预取/打乱)
    ↓
喂给模型训练
```

### 4.2 假设的场景与数据格式

假设我们在做**电商搜索推荐**。原始数据包含:

```python
"""
曝光日志(每次推荐用户看到了哪些商品):
exposure_log.tsv
格式: request_id \t user_id \t expose_item_ids \t timestamp
说明: expose_item_ids 是 JSON 数组,表示这次推荐展示给用户的商品列表

点击日志(用户点进了哪些商品):
click_log.tsv
格式: request_id \t user_id \t click_item_id \t dwell_time \t timestamp

用户画像(user_profile):
user_id \t age \t gender \t city \t historical_ctr
说明: historical_ctr = 用户历史点击率(0~1 之间)

物品属性(item_attr):
item_id \t category_id \t price \t title
"""
```

### 4.3 步骤 1:读取原始日志并打标签

打标签的含义是:对于一次曝光,用户点了哪个商品(D=1),哪个没点(D=0)。

```python
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class Sample:
    """一个训练样本"""
    user_id: str
    item_id: str
    label: int                              # 1=点击,0=未点击
    timestamp: int
    user_features: Dict[str, object] = field(default_factory=dict)
    item_features: Dict[str, object] = field(default_factory=dict)
    history_item_ids: List[str] = field(default_factory=list)    # 用户行为序列

def parse_exposure_tsv(path: str) -> Dict[str, Dict]:
    """读曝光日志,按 request_id 索引"""
    exposures = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            request_id, user_id, expose_str, ts = parts[0], parts[1], parts[2], int(parts[3])
            expose_items = json.loads(expose_str)           # ["i_001", "i_002", ...]
            exposures[request_id] = {
                "user_id": user_id,
                "ts": ts,
                "expose_items": set(expose_items)           # 转 set 加速查找
            }
    return exposures

def parse_click_tsv(path: str) -> Dict[str, str]:
    """读点击日志,按 request_id → clicked_item_id 映射"""
    clicks = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            request_id, click_item_id = parts[0], parts[2]
            clicks[request_id] = click_item_id
    return clicks

def label_samples(exposures: Dict, clicks: Dict) -> List[Sample]:
    """打标签:曝光中的 click_item_id → label=1,其他曝光 → label=0"""
    samples = []
    for req_id, exp in exposures.items():
        clicked_item = clicks.get(req_id, None)
        for item_id in exp["expose_items"]:
            label = 1 if item_id == clicked_item else 0
            samples.append(Sample(
                user_id=exp["user_id"],
                item_id=item_id,
                label=label,
                timestamp=exp["ts"]
            ))
    return samples

# 使用
exposures = parse_exposure_tsv("exposure_log.tsv")
clicks = parse_click_tsv("click_log.tsv")
raw_samples = label_samples(exposures, clicks)
print(f"总样本数: {len(raw_samples)}")
# print(f"正样本数(点击): {sum(1 for s in raw_samples if s.label==1)}")
# print(f"负样本数(曝光未点击): {sum(1 for s in raw_samples if s.label==0)}")
# 点击率 ≈ 1%~10%
```

**为什么不用曝光日志就直接随机负采样?**
- 曝光未点击 ≈ 用户**真正看不感兴趣**的商品(而非没见过的),比全局随机好得多
- 全局随机采样的负样本太"容易"了(用户很可能根本不认识这个商品),模型学不到区分能力

### 4.4 步骤 2:加载用户/物品特征

```python
def load_user_profile(path: str) -> Dict[str, Dict]:
    """加载用户画像,user_id → {age, gender, city, historical_ctr}"""
    profiles = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            uid = parts[0]
            profiles[uid] = {
                "age": int(parts[1]) if parts[1] else None,
                "gender": parts[2],
                "city": parts[3],
                "historical_ctr": float(parts[4]) if parts[4] else 0.0,
            }
    return profiles

def load_item_attr(path: str) -> Dict[str, Dict]:
    """加载物品属性,item_id → {category_id, price, title}"""
    attrs = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            iid = parts[0]
            attrs[iid] = {
                "category_id": parts[1],
                "price": float(parts[2]) if parts[2] else 0.0,
                "title": parts[3],
            }
    return attrs

def join_features(samples: List[Sample], user_profiles: Dict, item_attrs: Dict):
    """给样本拼接特征(原地修改)"""
    for sample in samples:
        sample.user_features = user_profiles.get(sample.user_id, {}).copy()
        sample.item_features = item_attrs.get(sample.item_id, {}).copy()
    # 注意:如果某些特征缺失了,需要兜底,否则后面编码会报错

user_profiles = load_user_profile("user_profile.tsv")
item_attrs = load_item_attr("item_attr.tsv")
join_features(raw_samples, user_profiles, item_attrs)
```

### 4.5 步骤 3:构建用户行为序列

这是搜推广样本构造里最核心、最容易出错的步骤。

```python
def build_user_history(click_log_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """从全量历史点击日志构建用户行为序列

    返回: user_id → [(timestamp, item_id), ...] 按时间升序
    """
    history = defaultdict(list)
    with open(click_log_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            request_id, user_id, click_item_id, _, ts = parts[0], parts[1], parts[2], parts[3], int(parts[4])
            history[user_id].append((ts, click_item_id))
    # 按时间排序,每个用户的行为串成序列
    for user_id in history:
        history[user_id].sort(key=lambda x: x[0])      # 升序,最旧→最新
    return dict(history)

def sample_sequence(history: List[Tuple[int, str]], max_len: int = 50) -> List[str]:
    """对行为序列做采样

    策略:保留最近的 max_len 个行为(工业界最常用)
    为什么取最近? 因为用户最近的兴趣对下一次行为影响最大
    """
    if len(history) <= max_len:
        return [item_id for _, item_id in history]
    return [item_id for _, item_id in history[-max_len:]]

# 在 label_samples 后
user_history = build_user_history("click_log.tsv")
for sample in raw_samples:
    hist = user_history.get(sample.user_id, [])
    sample.history_item_ids = sample_sequence(hist, max_len=50)
```

### 4.6 步骤 4:负采样策略

如果只用"曝光未点击"作为负样本,数量通常是不够的(因为推荐系统只推荐了少量商品)。需要额外补充。

```python
import random

def negative_sampling(samples: List[Sample],
                      all_item_ids: set,
                      neg_ratio: float = 3.0) -> List[Sample]:
    """
    补充负样本策略:
    1. 对每个正样本,补充 neg_ratio 个随机负样本
    2. 负样本从全量商品池中采样,但排除用户已经点过的

    为什么 neg_ratio=3?
    工业界经验:1 个正样本配 3~5 个负样本效果最好
    """
    clicked_items_per_user = defaultdict(set)
    for s in samples:
        if s.label == 1:
            clicked_items_per_user[s.user_id].add(s.item_id)

    extra_negatives = []
    for s in samples:
        if s.label == 0:
            continue                                # 曝光未点击已经是负样本
        user_clicked = clicked_items_per_user[s.user_id]
        # 从全量中排除用户点过的
        candidate_negatives = list(all_item_ids - user_clicked)
        n_needed = int(neg_ratio)
        if len(candidate_negatives) < n_needed:
            n_needed = len(candidate_negatives)
        if n_needed > 0:
            neg_items = random.sample(candidate_negatives, n_needed)
            for neg_item in neg_items:
                # 复制正样本的所有特征,只改 item_id 和 label
                neg_sample = Sample(
                    user_id=s.user_id,
                    item_id=neg_item,
                    label=0,
                    timestamp=s.timestamp,
                    user_features=s.user_features.copy(),
                    item_features={},               # 负样本可能没有物品特征,后面会兜底
                    history_item_ids=s.history_item_ids.copy()
                )
                extra_negatives.append(neg_sample)
    return samples + extra_negatives

# 收集所有出现的 item_id
all_items = set()
for s in raw_samples:
    all_items.add(s.item_id)

samples_with_neg = negative_sampling(raw_samples, all_items, neg_ratio=3.0)
print(f"负采样后总样本数: {len(samples_with_neg)}")
```

**为什么需要额外负采样?**
```text
原始曝光: 点击率 = 正:负 = 1:50 ~ 1:100
模型对负样本过拟合 → 预测概率整体偏低
补充 3 倍负样本后 → 1:200~1:400
模型对负样本更"严厉" → 正样本能脱颖而出
```

#### 4.6.1 六种负采样策略深度对比

| 策略 | 原理 | 难度 | 计算代价 | 效果 | 搜推广常用? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 全局随机** | 从全量物品池均匀采样 | 极容易 | O(1) | ❌太简单,模型学不到区分 | 几乎不用 |
| **2. 曝光未点击** | 用户看到但没点的物品 | 中 | O(1)(已有点击表) | ✅ **基准策略** | **★★★★★** |
| **3. 热门采样(Batch Neg)** | 在一个 batch 内,其他样本的正物品作为当前负样本 | 容易 | O(1)(自带) | ✅ 简单高效,双塔标配 | **★★★★★** |
| **4. 难度分层采样** | 按模型预测分数分桶:简单/中等/困难 | 高 | O(n)² | ✅✅ AUC 最高 | **★★★★** |
| **5. 对抗采样(Generator)** | 用一个生成网络专门产"最像正样本"的负样本 | 极高 | 训练双倍 | ✅✅ 极致效果 | **★★**(太贵) |
| **6. 混合策略** = 2+3+4 | 按比例混合曝光未点击 + Batch Neg + Hard | 中 | 可控 | ✅✅ 工业界冠军 | **★★★★★** |

```python
# 策略 3:Batch Negative Sampling(双塔模型的标准训练方法)
# 原理:在双塔模型中,user_emb 和 item_emb 分别编码
# batch 内每个 user 的正 item 是配对的,剩下的 B-1 个 item 就是负样本
# 不需要额外采样,计算代价为 0!

def batch_softmax_loss(user_emb, item_emb, temperature=0.07):
    """
    双塔模型的 Batch Softmax 损失。

    输入:
        user_emb: [B, D] — 塔 1 输出
        item_emb: [B, D] — 塔 2 输出

    原理:
        logits[i][j] = user_emb[i] · item_emb[j] / temperature
        目标:对角线(i=j)是正样本,其他是负样本
        所以 label = torch.arange(B)
    """
    B = user_emb.size(0)

    # 相似度矩阵 [B, B]
    logits = torch.matmul(user_emb, item_emb.T) / temperature

    # 对角线是正样本
    labels = torch.arange(B, device=user_emb.device)

    # 交叉熵:对每个 user,正确 item 在 B 个候选中排名第 1
    loss = F.cross_entropy(logits, labels)
    return loss

# 为什么 Batch Neg 效果好?
# 采样偏差小:当前 batch 的物品通常是"热门物品"
# 模型需要学会从热门物品中找出用户真正喜欢的 → 区分度更高

# 策略 4:Hard Negative Mining(困难负样本挖掘)
def hard_negative_mining(model, user_features, candidate_pool, n_hard=5):
    """
    用当前模型预测,挑出"分数最高但不是正样本"的物品。
    为什么有效?
    简单的负样本(完全不相关)模型一眼就看出是负的,学不到东西。
    困难的负样本(看起来像正)能迫使模型学到真正的区分边界。
    """
    model.eval()
    hard_negatives = []
    with torch.no_grad():
        # 对候选池打分
        scores = []
        for item in candidate_pool:
            score = torch.sigmoid(model(user_features, item))
            scores.append((score.item(), item))
        # 按分数从高到低排
        scores.sort(key=lambda x: -x[0])
        # 前 n_hard 个"看起来最像正"的就是 hard negative
        hard_negatives = [item for _, item in scores[:n_hard]]
    return hard_negatives

# 策略 6:混合策略(工业界最好的实践)
def mixed_negative_sampling(positive_samples, exposure_samples,
                             item_pool, batch_neg_ratio=0.5):
    """
    混合策略:batch_neg_ratio=0.5 表示 50% 负样本来自 batch neg
    50% 来自曝光未点击 + 困难负样本

    为什么混合最好?
    - 曝光未点击 → 保证基础区分度
    - Batch Neg → 引入分布偏差纠正
    - Hard Neg → 提升训练难度
    三种信号互补,模型学到的东西最全面
    """
    negatives = []
    # 50% 曝光未点击
    n_exposure = int(len(positive_samples) * 0.5)
    negatives.extend(random.sample(exposure_samples, min(n_exposure, len(exposure_samples))))

    # 30% 热门物品(可以用 top-K 热门物品采样,模拟 batch 分布)
    n_popular = int(len(positive_samples) * 0.3)
    # ... 从热门池采样

    # 20% 困难负样本
    n_hard = len(positive_samples) - n_exposure - n_popular
    # ... 用模型打分的 hard negative

    return negatives
```

**负采样的 "甜区" 经验值**:

```python
# 在 CTR 预估中:
# batch_size = 1024, 1 个正配 3~5 个负 → 最佳
#
# 在双塔召回中:
# batch_size = 2048, 2 塔各自编码后做 batch softmax
# 每个 batch 自动有 2047 个负样本(其他 sample 的 item)
#
# 在生成式推荐中:
# 把"全部 item"都 softmax 出来 → 计算量太大
# 只采样部分负样本 → 用 sampled softmax
```

### 4.7 步骤 5:特征编码(LabelEncode + Normalize)

这是整个 Pipeline 中最"搜推广"的阶段。

```python
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

class FeatureEncoder:
    """
    搜推广特征编码器。
    维护所有离散特征的词表映射,以及连续特征的 scaler。
    """
    def __init__(self):
        # 离散特征编码器
        self.cat_encoders = {}          # feature_name → {value → id}
        # 连续特征 scaler
        self.num_scalers = {}           # feature_name → StandardScaler
        # 特殊 token ID
        self.PAD_ID = 0
        self.UNK_ID = 1
        # 特殊 token
        self.special_ids = {"<PAD>": 0, "<UNK>": 1}

    def fit_categorical(self, feature_name: str, values: List[str], max_vocab: int = None):
        """训练一个离散特征编码器(LabelEncoder)"""
        # 统计频次
        from collections import Counter
        counter = Counter(values)
        # 按频次排序,高频在前
        sorted_items = sorted(counter.items(), key=lambda x: -x[1])
        if max_vocab:
            sorted_items = sorted_items[:max_vocab - len(self.special_ids)]

        # 构建映射
        mapping = {}
        idx = len(self.special_ids)     # 从 2 开始,0=PAD,1=UNK
        for token, count in sorted_items:
            if token not in mapping and token not in self.special_ids:
                mapping[token] = idx
                idx += 1
        self.cat_encoders[feature_name] = mapping
        return self

    def encode_categorical(self, feature_name: str, value) -> int:
        """编码一个离散值"""
        mapping = self.cat_encoders.get(feature_name, {})
        return mapping.get(value, self.UNK_ID)

    def fit_numerical(self, feature_name: str, values: List[float]):
        """训练连续特征 scaler(Z-Score)"""
        scaler = StandardScaler()
        values_arr = np.array(values, dtype=np.float32).reshape(-1, 1)
        scaler.fit(values_arr)
        self.num_scalers[feature_name] = scaler
        return self

    def encode_numerical(self, feature_name: str, value: float) -> float:
        """编码一个连续值(Z-Score 归一化 + 缺失填充)"""
        if value is None or np.isnan(value) or value == float('inf'):
            value = 0.0                     # 缺失兜底
        scaler = self.num_scalers.get(feature_name)
        if scaler is None:
            return value
        return scaler.transform(`value`)[0, 0]

    def save(self, path: str):
        """保存编码器到磁盘(推理时加载用同一份!)"""
        joblib.dump({
            "cat_encoders": self.cat_encoders,
            "num_scalers": self.num_scalers,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.cat_encoders = data["cat_encoders"]
        self.num_scalers = data["num_scalers"]
        return self

def train_encoder(samples: List[Sample]) -> FeatureEncoder:
    """在整个训练集上学习编码器"""
    encoder = FeatureEncoder()

    # 收集离散特征的所有取值
    user_genders = [s.user_features.get("gender", "<UNK>") for s in samples]
    user_cities = [s.user_features.get("city", "<UNK>") for s in samples]
    item_categories = [s.item_features.get("category_id", "<UNK>") for s in samples]

    encoder.fit_categorical("gender", user_genders, max_vocab=10)
    encoder.fit_categorical("city", user_cities, max_vocab=500)
    encoder.fit_categorical("category_id", item_categories, max_vocab=10000)

    # 收集连续特征
    ages = [s.user_features.get("age") for s in samples if s.user_features.get("age") is not None]
    prices = [s.item_features.get("price") for s in samples if s.item_features.get("price") is not None]
    ctrs = [s.user_features.get("historical_ctr") for s in samples if s.user_features.get("historical_ctr") is not None]

    if ages: encoder.fit_numerical("age", ages)
    if prices: encoder.fit_numerical("price", prices)
    if ctrs: encoder.fit_numerical("historical_ctr", ctrs)

    return encoder

def encode_sample(sample: Sample, encoder: FeatureEncoder, max_seq_len: int) -> Dict:
    """把一个 Sample 编码成 PyTorch Tensor 字典"""

    # ==== 离散特征:字符串 ID → int ID ====
    user_id_encoded = encoder.encode_categorical("user", sample.user_id)
    item_id_encoded = encoder.encode_categorical("item", sample.item_id)
    gender_encoded = encoder.encode_categorical("gender",
        sample.user_features.get("gender", "<UNK>"))
    city_encoded = encoder.encode_categorical("city",
        sample.user_features.get("city", "<UNK>"))
    category_encoded = encoder.encode_categorical("category_id",
        sample.item_features.get("category_id", "<UNK>"))

    # ==== 连续特征:归一化 ====
    age_encoded = encoder.encode_numerical("age", sample.user_features.get("age"))
    price_encoded = encoder.encode_numerical("price", sample.item_features.get("price"))
    ctr_encoded = encoder.encode_numerical("historical_ctr", sample.user_features.get("historical_ctr"))

    # ==== 行为序列:采样 + padding + mask ====
    seq_ids = [encoder.encode_categorical("item", item) for item in sample.history_item_ids]
    # 截断
    seq_ids = seq_ids[:max_seq_len]
    # 右侧 padding
    seq_len = len(seq_ids)
    seq_ids = seq_ids + [encoder.PAD_ID] * (max_seq_len - seq_len)
    seq_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)

    return {
        "user_id": user_id_encoded,            # int
        "item_id": item_id_encoded,            # int
        "gender": gender_encoded,              # int
        "city": city_encoded,                  # int
        "category": category_encoded,          # int
        "age": age_encoded,                    # float
        "price": price_encoded,                # float
        "ctr": ctr_encoded,                    # float
        "history_ids": seq_ids,                # [max_seq_len]
        "history_mask": seq_mask,              # [max_seq_len]
        "label": sample.label,                 # int
    }
```

### 4.8 步骤 6:Dataset + DataLoader 包装

```python
import torch
from torch.utils.data import Dataset, DataLoader

class RecSysDataset(Dataset):
    """搜推广 Dataset:把编码后的样本列表包装成 PyTorch Dataset"""
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_id": torch.tensor(s["user_id"], dtype=torch.long),
            "item_id": torch.tensor(s["item_id"], dtype=torch.long),
            "gender": torch.tensor(s["gender"], dtype=torch.long),
            "city": torch.tensor(s["city"], dtype=torch.long),
            "category": torch.tensor(s["category"], dtype=torch.long),
            "age": torch.tensor(s["age"], dtype=torch.float32),
            "price": torch.tensor(s["price"], dtype=torch.float32),
            "ctr": torch.tensor(s["ctr"], dtype=torch.float32),
            "history_ids": torch.tensor(s["history_ids"], dtype=torch.long),
            "history_mask": torch.tensor(s["history_mask"], dtype=torch.long),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }

def collate_fn(batch):
    """把列表拼成 batch tensor"""
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([b[key] for b in batch])
    return out

def create_dataloader(samples: List[Dict], batch_size: int = 1024, shuffle: bool = True, num_workers: int = 4):
    dataset = RecSysDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )
    return loader
```

### 4.9 步骤 7:完整 Pipeline 串联

```python
def build_train_pipeline(
    exposure_path: str,          # 曝光日志
    click_path: str,             # 点击日志(作为历史序列用)
    user_profile_path: str,      # 用户画像
    item_attr_path: str,         # 物品属性
    max_seq_len: int = 50,
    neg_ratio: float = 3.0,
    batch_size: int = 1024,
    val_ratio: float = 0.1,
):
    # 1. 解析 & 打标签
    print("[1/7] 解析原始日志...")
    exposures = parse_exposure_tsv(exposure_path)
    clicks = parse_click_tsv(click_path)
    raw_samples = label_samples(exposures, clicks)
    print(f"  原始样本数: {len(raw_samples)}")

    # 2. Load 辅助特征
    print("[2/7] 加载辅助特征...")
    user_profiles = load_user_profile(user_profile_path)
    item_attrs = load_item_attr(item_attr_path)
    join_features(raw_samples, user_profiles, item_attrs)

    # 3. 构建行为序列
    print("[3/7] 构建用户行为序列...")
    user_history = build_user_history(click_path)
    for s in raw_samples:
        hist = user_history.get(s.user_id, [])
        s.history_item_ids = sample_sequence(hist, max_len=max_seq_len)

    # 4. 负采样
    print("[4/7] 负采样(ratio={})...".format(neg_ratio))
    all_items = set()
    for s in raw_samples:
        all_items.add(s.item_id)
    samples_with_neg = negative_sampling(raw_samples, all_items, neg_ratio)
    print(f"  负采样后总样本数: {len(samples_with_neg)}")

    # 5. 分割训练/验证(时间切分!)
    print("[5/7] 按时间切分训练/验证...")
    timestamps = [s.timestamp for s in samples_with_neg]
    split_ts = sorted(timestamps)[int(len(timestamps) * (1 - val_ratio))]
    train_samples_raw = [s for s in samples_with_neg if s.timestamp <= split_ts]
    val_samples_raw = [s for s in samples_with_neg if s.timestamp > split_ts]
    print(f"  训练: {len(train_samples_raw)}, 验证: {len(val_samples_raw)}")

    # 6. 训练编码器(只在训练集上!)
    print("[6/7] 学习特征编码器...")
    encoder = train_encoder(train_samples_raw)

    # 编码所有样本
    train_encoded = [encode_sample(s, encoder, max_seq_len) for s in train_samples_raw]
    val_encoded = [encode_sample(s, encoder, max_seq_len) for s in val_samples_raw]

    # 7. 创建 DataLoader
    print("[7/7] 创建 DataLoader...")
    train_loader = create_dataloader(train_encoded, batch_size, shuffle=True)
    val_loader = create_dataloader(val_encoded, batch_size, shuffle=False)

    # 保存编码器供推理用
    encoder.save("feature_encoder.pkl")
    print("  编码器保存至 feature_encoder.pkl")

    return train_loader, val_loader, encoder

# 使用
if __name__ == "__main__":
    train_loader, val_loader, encoder = build_train_pipeline(
        exposure_path="data/exposure_log.tsv",
        click_path="data/click_log.tsv",
        user_profile_path="data/user_profile.tsv",
        item_attr_path="data/item_attr.tsv",
        max_seq_len=50,
        neg_ratio=3.0,
        batch_size=1024,
        val_ratio=0.1,
    )

    # 验证一把
    for batch in train_loader:
        print(f"batch keys: {batch.keys()}")
        print(f"user_id: {batch['user_id'].shape}")        # [1024]
        print(f"history_ids: {batch['history_ids'].shape}") # [1024, 50]
        print(f"label: {batch['label'].shape}")             # [1024]
        print(f"label sum: {batch['label'].sum().item()}")  # 正样本总数
        break
```

**为什么这个 Pipeline 是"工业级"的?**
1. **时间切分**而非随机切分——防止数据泄漏
2. **训练集 fit 编码器**——防止特征穿越
3. **序列采样保留最近 N 个**——工业界最实用
4. **负采全量排除用户点过**——防止把正样本错标为负
5. **编码器序列化到磁盘**——推理时加载同一份

### 4.10 "生成式推荐"到底是什么?——从范式到架构的深度拆解

你注意到一个关键的转变了吗?在前面的所有代码里,我们把搜推广的"特征工程"变成了"序列生成":

1. 用户行为序列 → **token 序列**(把点击过的 item_id 当成一个个 token)
2. 历史物品 ID → **PADDED token 序列**(像 NLP 里 padding 句子一样 padding 行为序列)
3. 用户画像 → **辅助 token**(user_id、gender、city 都变成特殊 token)
4. 目标物品 ID → **下一个要预测的 token**(就像语言模型预测下一个词)

**这就是生成式推荐的本质:把推荐问题完全 token 化,变成一个"下一 token 预测(next token prediction)"的序列生成问题。**

```text
传统推荐范式:
  特征工程 → Embedding → MLP → 点击概率(标量)
  模型是"判别式"的:输入特征,输出概率

生成式推荐范式:
  Token化 → Transformer → 下一个 item token(向量)
  模型是"生成式"的:输入 token 序列,预测下一个 token
```

#### 4.10.1 三篇里程碑论文——理解 GenRec 的进化路线

**论文 1:P5 —— Everything is a Sequence(RecSys 2022)**

```text
核心思想:
把推荐系统的所有要素(用户、物品、评论、时间戳)都映射到同一个 token 空间。
用 Text-to-Text 的方式统一处理所有推荐任务。

任务模板:
  "用户 U_123 历史行为: [item_1, item_2, item_3],
   候选物品: item_456,
   预测点击概率:" → "1" (或 "0")

关键设计:
  - 所有 ID(用户、物品)都作为特殊 token 加入词表
  - 用 T5(Encoder-Decoder)架构,编码器读输入,解码器生成输出
  - Loss:Cross-Entropy on 目标 token
  - 一个模型做多种任务:评分预测、序列推荐、解释生成、对话推荐

局限:
  - 物品数量受限(整个词表 6 万,工业界物品千万级)
  - Encoder-Decoder 推理慢(需要两阶段)
```

**论文 2:TIGER & CIDR——用语义 ID 解决物品数量问题**

```text
核心问题:P5 把 item 当作一个 token,词表大小 = 物品数
工业界 1000 万物品 → 词表 1000 万 → softmax 算不了!

TIGER 的解法:
  用 VQ-VAE(向量量化变分自编码器)把每个物品编码成"语义 ID"的序列

  比如:
  item_123  →  [42, 15, 7, 3]  ← 4 个 token,每个 token 范围 0~255
  item_456  →  [42, 18, 2, 9]

  这样设计:
  - 词表大小 = 256(只需预测 256 类),而不是 1000 万
  - 相似物品的语义 ID 更相似 → 模型天然能泛化到新物品
  - 用 4 个 token 表示一个物品 → 参数量: 4 × 256 × d_model,远小于 1000 万 × d_model

CIDR 进一步改进:
  用 RQ-VAE(残差量化)代替 VQ-VAE,让语义 ID 的层次结构更清晰:
  - 第一层 token:品类级别(如"电子产品")
  - 第二层 token:品牌级别(如"苹果")
  - 第三层 token:具体型号(如"iPhone 15")
  - 第四层 token:具体 SKU(如"256GB 黑色")

所以 TIGER/CIDR 的推荐流程:
  [用户行为 token 序列] → [Transformer] → [预测下一个物品的语义 ID 序列]
  这不是 1 次预测,而是 4 次串行预测:先预测品类,再预测品牌,再...
```

**论文 3:HSTU —— Meta 的工业级生成式推荐架构(KDD 2024)**

```text
HSTU(Hierarchical Sequential Transduction Unit)是 Meta 在 2024 年提出的,
是目前最接近"用 LLM 架构做推荐"的生产级系统。

核心变革:
  1. 抛弃 MLP → 完全用 Transformer 堆叠替代传统 DNN
  2. 用"点积注意力"替代"拼接+MLP"做特征交叉
  3. 用 Decoder-only 替代 Encoder-only

HSTU Block 设计(跟 LLaMA Block 非常像):
  x = x + PointwiseFFN(RMSNorm(x))         # 就是 SwiGLU
  x = x + MultiheadAttention(RMSNorm(x))    # 预归一化
  # 和 LLaMA 的唯一区别:Attention 的 score 计算方式不同

区别:
  LLaMA: Q @ K^T / √d  → Softmax → Attention
  HSTU: (Q @ K^T) * (u^T · x_i) * (v^T · x_j)  → 带 bias 的 attention

效果:
  - 在 Meta 的社交推荐中,用 HSTU 替代传统 DeepFM/DIN
  - 在 7 个推荐基准上全面超越 SOTA
  - 参数量减少 50%,推理速度提升 5 倍
```

#### 4.10.2 GenRec 的三种架构范式

```text
范式 A:Encoder-Only(传统做法)
  代表:P5 之前的模型(但 P5 用 Encoder-Decoder)
  输入:[user_id, item_id, ...] → Self-Attention → Pooling → MLP → sigmoid
  缺点:不能做生成任务,只能打分

范式 B:Encoder-Decoder(P5/BART 路线)
  代表:P5, M6-Rec
  输入:用户行为 + 候选
  输出:点击/不点击 token(生成)
  优点:统一了所有推荐任务
  缺点:推理慢(2 阶段),物品多了 softmax 太大

范式 C:Decoder-Only(2024 年主流,和 LLM 趋同!)
  代表:HSTU, 以及越来越多新模型

  架构:
  [BOS] → [user_token_1] → [item_token_2] → ... → [target_item]
                                                          ↓
                                              Decoder-Only Transformer
                                                          ↓
                                                 下一个 item token

  为什么 Decoder-Only 胜出?
  1. 推理时 KV-cache → 比 Encoder-Decoder 快 2~3 倍
  2. 自回归生成 = 推荐场景(一次只推一个 item)
  3. 和 LLaMA/Qwen 共用同一套训练框架
```

#### 4.10.3 GenRec 的 Tokenization 三种方案

```text
方案 A:原始 ID(最直接,但限制最大)
  item_1234567 → 一个 token,词表 = N_items
  问题:词表太大,冷启动灾难(OOV)
  适用:物品数 < 10 万的小场景

方案 B:语义 ID(当前 SOTA)
  item_1234567 → [code_1, code_2, code_3, code_4]
  由 VQ-VAE / RQ-VAE 预训练得到
  优点:
    - 词表可控(256 × depth)
    - 语义相近的物品 ID 也相近
    - 新物品不用重新训练 VQ-VAE,只需通过编码器得到语义 ID
  缺点:
    - 需要额外训练 VQ-VAE(但只需一次)
    - 生成时需自回归解码 4 步才能得到一个物品

方案 C:文本描述(最灵活,但最慢)
  item_1234567 → "Apple iPhone 15 Pro Max 256GB Space Black"
  用 BERT/LLaMA 的 tokenizer 分 tile 编码
  优点:
    - 零样本冷启动:新物品只要有文本描述就能推
    - 可跨域迁移
  缺点:
    - 推理慢(文本 token 比单个 ID token 长 10 倍)
    - 需要额外做文本编码
```

```python
# 这三种 tokenization 的代码对比
import torch
import torch.nn as nn

# 方案 A:原始 ID tokenization(最简单)
class RawIDTokenizer:
    """直接把 item_id 当 token"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def encode(self, item_ids):
        return torch.tensor(item_ids, dtype=torch.long)
    def decode(self, tokens):
        return tokens.tolist()

# 方案 B:语义 ID tokenization(RQ-VAE 风格)
class SemanticIDTokenizer(nn.Module):
    """
    用 RQ-VAE 把 item 编码为多级语义 token。
    实际 RQ-VAE 需要预训练,这里展示 forward 流程。
    """
    def __init__(self, num_codes=256, num_levels=4, dim=64):
        super().__init__()
        self.num_codes = num_codes          # 每层词表大小 256
        self.num_levels = num_levels        # 层级数 4
        # 每层的 codebook: [num_codes, dim]
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_codes, dim))
            for _ in range(num_levels)
        ])

    def encode(self, item_embedding):
        """把 item 的 d_model 向量 → [4] 个语义 token"""
        residual = item_embedding
        codes = []
        for level in range(self.num_levels):
            # 找最近邻 code
            dist = torch.cdist(residual.unsqueeze(0), self.codebooks[level].unsqueeze(0))
            code_idx = dist.argmin(dim=-1)   # [1]
            codes.append(code_idx.item())
            # 残差:减去选中的 code
            residual = residual - self.codebooks[level][code_idx]
        return codes  # [4] 个 int,每个范围 0~255

# 方案 C:文本 tokenization(用 LLaMA tokenizer)
from transformers import AutoTokenizer
class TextTokenizer:
    """用现成的 LLM tokenizer"""
    def __init__(self, model_name="meta-llama/Llama-2-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, item_title):
        # "Apple iPhone 15 Pro Max" → [1, 234, 567, 890, 2]
        return self.tokenizer.encode(item_title)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
```

#### 4.10.4 GenRec 和 LLM 的"趋同进化"——你现在的位置

> **这是你最需要理解的部分**:2024-2025 年,生成式推荐和 30B 以下的小 LLM 的设计思路正在快速趋同。

```text
传统搜推广模型(2016-2022)            现代 GenRec + 小 LLM(2024-2025)
────────────────────────             ──────────────────────────────
Feature Engineering → Embedding      Tokenization → Embedding
Wide&Deep / DeepFM / DIN             Decoder-Only Transformer
MLP 是主力交互层                     Attention 是主力交互层
ReLU / PReLU                         SwiGLU / GeGLU
BatchNorm / LayerNorm                RMSNorm(唯一选择)
Position: 无 / 简单 PE               RoPE(旋转位置编码)
Loss: BCE / CrossEntropy             CE + 对比学习 + 辅助 Loss
单任务                             多任务统一
Embedding: 大而独立                  Embedding: 共享 + 量化
训练: 单机多卡                       训练: 分布式 + 混合精度
```

**这意味着什么?**
- 你现在学的东西(LayerNorm vs RMSNorm / SwiGLU / RoPE / Decoder-Only)不仅是 LLM 的知识,也是 GenRec 的标配
- HSTU 的 block 结构 ≈ LLaMA 的 block 结构 + 特殊的 attention scoring
- 未来 1-2 年,搜推广模型会全面向 LLaMA/Qwen 的小模型架构靠拢

#### 4.10.5 从 0 搭建一个 GenRec 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GenerativeRecBlock(nn.Module):
    """
    生成式推荐的一个 Transformer Block。
    设计思路 = LLaMA Block + HSTU 的交互式 attention。
    """
    def __init__(self, d_model, n_heads, expansion=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # 和 LLaMA 一样的 SwiGLU FFN
        self.gate_proj = nn.Linear(d_model, expansion * d_model)
        self.up_proj = nn.Linear(d_model, expansion * d_model)
        self.down_proj = nn.Linear(expansion * d_model, d_model)
        # Pre-RMSNorm(和 LLaMA 一样)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-Norm + Attention
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x),
                               attn_mask=mask)[0]
        # SwiGLU FFN
        x = x + self.down_proj(
            F.silu(self.gate_proj(self.norm2(x))) * self.up_proj(self.norm2(x))
        )
        return x

class RMSNorm(nn.Module):
    """
    RMSNorm: LLaMA 和 GenRec 的标准归一化。
    比 LayerNorm 快 15%,且效果更好。
    公式: y = x / sqrt(mean(x²) + ε) * γ
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, T, D]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class MiniGenRec(nn.Module):
    """
    极简生成式推荐模型——和 LLaMA 共用同一套架构!

    输入: token 序列(含用户 ID + 历史行为 ID + 目标 item ID)
    输出: 预测下一个 item 的 logits
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # RoPE 旋转位置编码(和 LLaMA 一样)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList([
            GenerativeRecBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.token_emb(token_ids)
        x = x + self.rope(token_ids)        # RoPE 编码
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits                        # [B, T, vocab_size]

class RotaryEmbedding(nn.Module):
    """旋转位置编码 RoPE——现代 GenRec 和 LLM 的标准"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [B, T, D]
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb[None, :, :]

# 使用
model = MiniGenRec(vocab_size=10000)
tokens = torch.randint(0, 10000, (32, 50))  # [B=32, T=50]
logits = model(tokens)                        # [32, 50, 10000]

# 训练:和语言模型一模一样!
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, 10000),    # 预测
    tokens[:, 1:].reshape(-1)                 # 真实下一个 token
)
```

**看到没?这个模型和 LLaMA 的最小号版本几乎一模一样:**
- `RMSNorm` + `Pre-Norm`(LLaMA 标配)
- `SwiGLU` FFN(Mistral/Qwen/DeepSeek 都用)
- `RoPE` 位置编码(几乎所有的现代 LLM 都用)
- `Decoder-Only` 因果注意(和 GPT/LLaMA 一样)

#### 4.10.6 GenRec 的损失函数:不只是 Cross-Entropy

```python
def genrec_loss(logits, target_item_ids, item_freq=None):
    """
    GenRec 的三种损失组合:

    1. CE Loss:预测下一个 item token(标准)
    2. Sampled Softmax:物品太多时,只对部分负样本算 softmax
    3. 对比学习 Loss:让相似序列的 embedding 相近
    """
    # 1. 标准 Cross-Entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_item_ids.view(-1),
        ignore_index=0
    )

    # 2. Sampled Softmax(物品数 > 10 万时使用)
    # 不计算全量 vocab 的 softmax,而是对正样本 + K 个负样本算
    # K = 256 或 512

    # 3. 对比学习 Loss(InfoNCE)
    # 同一个序列的 dropout 两次,让两次的 embedding 相似
    # 不同序列的 embedding 远离

    return ce_loss  # + sampled_softmax_loss + contrastive_loss
```

#### 4.10.7 "生成式推荐"到底何时真正有价值?

| 场景 | 传统方法 | GenRec 方法 | GenRec 优势 |
| :--- | :--- | :--- | :--- |
| **冷启动(新物品)** | 无交互 → 推不出去 | 用物品标题/属性生成 token → 基于语义推理 | ✅ 零样本推荐 |
| **跨域推荐** | 每个域独立建模,特征不共享 | 统一 token 空间 → 一个模型跨域 | ✅ 数据互通 |
| **多模态融合** | 图片、文本、ID 各用独立模块 | 全部 token 化 → 一个 Transformer | ✅ 架构统一 |
| **多任务统一** | 每个任务一个塔(CTR/CVR/时长) | 同一序列生成框架 | ✅ 参数共享 |
| **可解释性** | Attention 权重或 SHAP | 生成的 token 序列天然可解释 | ✅ 生成即解释 |

**但也要知道 GenRec 的局限**:
- 推理速度:自回归生成比单次 MLP 慢 5~10 倍
- 物品更新:新增物品需要更新 tokenizer/VQ-VAE
- 训练成本:Transformer 比 MLP 贵一个数量级
- 已有系统迁移成本:从 DeepFM 迁移到 Transformer 需要改整个架构

**什么时候应该上 GenRec?**
```text
超过 1000 万 DAU 的平台? → 值得考虑
有丰富的行为序列数据? → GenRec 效果好于 MLP
需要跨域/多模态能力? → GenRec 天生优势
推理延迟容忍(< 50ms)? → 可以做
```

### 4.11 第四章小结 & 自测

**小结**:
- 我们从原始日志出发,完成了全套训练数据构造流程
- 关键步骤:曝光打标签 → Join 特征 → 行为序列构建 → 负采样 → 编码 → DataLoader
- 生成式推荐的核心:把推荐问题 token 化,用 Transformer 建模所有特征交互
- 编码器需要离线 fit、序列化,在线加载同一份

**自测题**:
1. 为什么负采样时不从"全量物品池"而是从"曝光未点击"采?两种策略的优劣?
2. 写一个函数,给定 `batch = {"history_ids": [B, T], "history_mask": [B, T]}`,返回不带 padding 干扰的平均池化向量。
3. 如果线上出现了一个训练时没见过的 `new_city`,你的编码器会怎么处理?
4. 假设离线训练用时间切分,验证集 AUC=0.75;上线后发现线上 AUC 只有 0.65,排查方向是什么?
5. 生成式推荐把"用户行为序列"token 化的本质是什么?

---

> 第 5 章是本书的最后一块拼图——**训练流程与工程实践**。我们会把前 4 章的 Python、数据处理、PyTorch、Pipeline 全部串到一个完整的训练框架里。

---

## 第 5 章:训练流程与工程实践

> **本章目标**:从零实现一个工业级的搜推广训练框架。包含标准的训练循环、梯度管理、正则化、保存加载、bug 排查。
>
> 读完本章,你能:
> - 写一个带 early stopping / checkpoint / logging 的通用 Trainer
> - 理解为什么搜推广的梯度裁剪和优化器配置是"非常规"的
> - 正确地在 Embedding 和 MLP 上施加不同的正则化策略
> - 做一个"可以复现实验"的模型保存管理
> - 遇到训练 NaN 或 AUC 不涨时,知道怎么系统化排查

### 5.1 从"零件"到"整车":完整的训练框架

前 4 章我们分别做了:
- 第 1 章:Python 数据结构(零件怎么认)
- 第 2 章:数据处理(零件怎么造)
- 第 3 章:PyTorch 基础(零件怎么用)
- 第 4 章:数据 Pipeline(零件怎么组装)

现在我们要**把所有零件装成整车,并且跑起来**。

### 5.2 一个生产级训练循环的结构

```python
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Trainer:
    """
    搜推广通用训练器。

    核心设计:
    1. 支持混合精度(AMP)训练
    2. 支持分层学习率(Embedding/MLP 不同 lr)
    3. 支持梯度裁剪 + 梯度累积
    4. 支持 early stopping + checkpoint
    5. 完整的日志/指标追踪
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = "cuda",
        max_epochs: int = 10,
        grad_clip_norm: float = 1.0,
        grad_accum_steps: int = 1,
        early_stop_patience: int = 3,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.early_stop_patience = early_stop_patience
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.scaler = GradScaler(enabled=use_amp)
        self.history = defaultdict(list)           # epoch → metrics
        self.best_val_metric = -float('inf')
        self.best_epoch = -1
        self.epochs_no_improve = 0

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch,返回 metrics dict"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            # 1. 数据搬移到 GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label")

            # 2. 前向(AMP 自动混合精度)
            with autocast(enabled=self.use_amp):
                logits = self.model(batch)           # [B] 或 [B, 1]
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1)
                )
                loss = loss / self.grad_accum_steps  # 梯度累积:归一化

            # 3. 反向(AMP)
            self.scaler.scale(loss).backward()

            # 4. 梯度累积:积累到指定步数才更新
            if (step + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps * labels.size(0)
            total_samples += labels.size(0)

        # 处理剩余不足一个 accum 步的梯度
        if (step + 1) % self.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / total_samples
        epoch_time = time.time() - start_time

        return {"train_loss": avg_loss, "train_time": epoch_time}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证集评估"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("label")

            logits = self.model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), labels.view(-1)
            )

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / total_samples
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        # AUC 计算
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.numpy(), preds.numpy())

        return {"val_loss": avg_loss, "val_auc": auc}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存 checkpoint(含优化器、scaler、metadata)"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "history": dict(self.history),
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # 最新 checkpoint(覆盖)
        path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, path)

        # 最佳 checkpoint(保留)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            # 同时保存一份 metadata,方便追溯
            with open(os.path.join(self.checkpoint_dir, "best_metadata.json"), "w") as f:
                json.dump({
                    "epoch": epoch,
                    "best_val_metric": self.best_val_metric,
                    "model_config": str(self.model),
                }, f, indent=2)

    def load_checkpoint(self, path: str) -> int:
        """加载 checkpoint 并恢复训练"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_metric = checkpoint["best_val_metric"]
        self.history = defaultdict(list, checkpoint.get("history", {}))
        return checkpoint["epoch"]

    def fit(self, resume_from: str = None):
        """完整训练流程"""
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            logger.info(f"从 epoch {start_epoch} 恢复训练")

        logger.info(f"开始训练,共 {self.max_epochs} 个 epoch")
        for epoch in range(start_epoch, self.max_epochs):
            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 记录
            epoch_metrics = {**train_metrics, **val_metrics}
            for k, v in epoch_metrics.items():
                self.history[k].append(v)

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                epoch_metrics["lr"] = current_lr

            # 日志
            log_msg = f"Epoch {epoch+1}/{self.max_epochs} | "
            log_msg += f"train_loss: {epoch_metrics['train_loss']:.4f} | "
            log_msg += f"val_loss: {epoch_metrics['val_loss']:.4f} | "
            log_msg += f"val_auc: {epoch_metrics['val_auc']:.4f} | "
            if 'lr' in epoch_metrics:
                log_msg += f"lr: {epoch_metrics['lr']:.2e} | "
            log_msg += f"time: {epoch_metrics['train_time']:.1f}s"
            logger.info(log_msg)

            # 保存 checkpoint
            is_best = epoch_metrics["val_auc"] > self.best_val_metric
            if is_best:
                self.best_val_metric = epoch_metrics["val_auc"]
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, is_best=is_best)

            # early stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                logger.info(f"Early stopping! {self.early_stop_patience} 个 epoch 无提升")
                break

        logger.info(f"训练完成。最佳验证 AUC: {self.best_val_metric:.4f} (epoch {self.best_epoch+1})")
        return self.history
```

**为什么这个 Trainer 是"搜推广生产级"的?**
- **AMP**:混合精度训练,显存省 40%,速度快 2~3 倍
- **梯度累积**:当 batch size 太大导致显存不够时,拆成多个小 batch 累加梯度
- **梯度裁剪**:防止梯度爆炸(搜推广 Embedding 稀疏更新时极易出现)
- **Early Stopping**:防止过拟合
- **Checkpoint 包含优化器 + scaler**:断点续训时零损失
- **Best + Latest 双减**:最新覆盖省空间,最好保留可回溯

### 5.3 搜推广特有的梯度管理

#### 5.3.1 分层学习率(再次强调,实战必用)

```python
from torch.optim import AdamW

def build_optimizer(model, embedding_lr=1e-2, mlp_lr=5e-4, weight_decay=1e-5):
    """构建分层学习率优化器"""

    # 把 Embedding 参数和 MLP 参数分开
    embedding_params = []
    mlp_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "emb" in name or "embedding" in name.lower():
            embedding_params.append(param)
        else:
            mlp_params.append(param)

    optimizer = AdamW([
        {"params": embedding_params, "lr": embedding_lr, "weight_decay": 0},       # Embedding:高 lr,无 weight_decay
        {"params": mlp_params, "lr": mlp_lr, "weight_decay": weight_decay},
    ], lr=mlp_lr)

    logger.info(
        f"分层学习率: Embedding={embedding_lr}, MLP={mlp_lr}, "
        f"Embedding params={len(embedding_params)}, MLP params={len(mlp_params)}"
    )
    return optimizer
```

**为什么 Embedding 的 weight_decay=0?**
- Embedding 表几千万行,每步更新只 touch 到其中的几百行(稀疏更新)
- 如果 weight_decay > 0,优化器会对**整个 Embedding 矩阵**施加正则化
- 整个矩阵 = 几千万行 × 32 维 = 几十亿个参数,每步都要全部更新 → 直接 OOM + 慢 100 倍
- 正确做法:weight_decay 只在 MLP 上开,Embedding 用 Dropout 或 LazyRegularization

#### 5.3.2 稀疏优化器(SparseAdam)

```python
# 如果 Embedding 层极大(> 1 亿行),AdamW 全部更新会 OOM
# 用 SparseAdam:只更新被索引到的 Embedding 行

from torch.optim import SparseAdam

class SparseEmbedding(nn.Module):
    """大 Embedding 专用:用 SparseAdam 优化"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.emb(x)

# 注意:SparseAdam 的要求:
# 1. Embedding 参数必须是 2 维的 [num_embeddings, dim]
# 2. 梯度必须是稀疏的(只有被索引到的行有梯度)
# 3. 不能和 weight_decay 一起用

optimizer = torch.optim.SparseAdam(
    model.emb.parameters(),
    lr=1e-2
)
```

#### 5.3.3 梯度裁剪的正确姿势

```python
# 方案 1:全局裁剪(最常用)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方案 2:只对 MLP 裁剪(Embedding 梯度天然稀疏,不剪)
# 在 backward() 之后:
for name, param in model.named_parameters():
    if "emb" in name:
        continue
    if param.grad is not None:
        torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

# 方案 3:梯度值裁剪(防止 NaN 梯度扩散)
for param in model.parameters():
    if param.grad is not None:
        param.grad.data = param.grad.data.clamp(-10.0, 10.0)
```

### 5.4 正则化实战

#### 5.4.1 Dropout —— 搜推广最有效的正则化

```python
class RecSysModel(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()
        # Embedding 层
        self.user_emb = nn.Embedding(1000000, embedding_dim)
        self.item_emb = nn.Embedding(500000, embedding_dim)

        # Embedding Dropout:防止 Embedding 过拟合
        self.emb_dropout = nn.Dropout(dropout)

        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 8, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),           # MLP Dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch):
        user_vec = self.emb_dropout(self.user_emb(batch["user_id"]))
        item_vec = self.emb_dropout(self.item_emb(batch["item_id"]))
        dense = torch.stack([
            batch["age"], batch["price"], batch["ctr"]
        ], dim=-1)

        combined = torch.cat([user_vec, item_vec, dense], dim=-1)
        return self.mlp(combined).squeeze(-1)

model = RecSysModel()
# 建议:Dropout 从 0.1 开始调,显存不够或过拟合就加
```

**Dropout 的搜推广心法**:
- Embedding Dropout 比 MLP Dropout 更重要(因为 Embedding 参数量巨大)
- 训练时 Dropout → 推理时自动关闭
- 不要在 Embedding 上开太高(0.2 以上可能会丢失信息)

#### 5.4.2 Embedding 的 L2 约束(代替 weight_decay)

```python
class L2RegularizedEmbedding(nn.Module):
    """带 L2 约束的 Embedding,只正则化被索引到的行"""
    def __init__(self, num_embeddings, embedding_dim, l2_lambda=1e-5):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.l2_lambda = l2_lambda

    def forward(self, indices):
        vectors = self.emb(indices)
        if self.training and self.l2_lambda > 0:
            # 只在被索引的行上施加 L2 损失
            l2_loss = self.l2_lambda * vectors.pow(2).sum()
            # 通过 register_forward_hook 把 loss 挂到模型上
            self._l2_loss = l2_loss
        return vectors

# 在训练循环中使用:
for batch in loader:
    user_vec = l2_emb(batch["user_id"])
    item_vec = l2_emb(batch["item_id"])
    pred = model(user_vec, item_vec)
    loss = bce_loss(pred, labels)

    # 把 Embedding 的 L2 损失加到总损失中
    if hasattr(l2_emb, '_l2_loss'):
        loss = loss + l2_emb._l2_loss

    loss.backward()
    optimizer.step()
```

#### 5.4.3 标签平滑(Label Smoothing)——CTR 预估的常用技巧

```python
def label_smoothing_loss(logits, labels, smoothing=0.1):
    """
    标签平滑版 BCE Loss。
    把 ground truth 从 {0, 1} 平滑到 {smoothing, 1-smoothing}

    为什么? 因为在搜推广里,label=0 不一定代表用户"绝对不喜欢"
    ——可能只是没看到、手滑、网络不好。
    平滑一下,让模型不要太"绝对"地相信标签。
    """
    labels_smoothed = labels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, labels_smoothed)
```

**效果**:离线 AUC 可能不变或略降 0.001,但**线上 AUC 通常能涨 0.002~0.005**——模型泛化能力变强了。

### 5.5 模型保存/加载的最佳实践

#### 5.5.1 不止 save/load,还要 version + config + metadata

```python
class ModelManager:
    """
    搜推广模型管理:保存的不只是权重,还有版本、配置、训练元信息。
    """
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: nn.Module, config: dict, metrics: dict, version: str = None):
        """保存完整模型包"""
        if version is None:
            version = time.strftime("%Y%m%d_%H%M%S")

        model_dir = self.base_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. 模型权重
        torch.save(model.state_dict(), model_dir / "model.pt")

        # 2. 模型配置(架构参数)
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # 3. 训练指标
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 4. README(谁、什么时间、为什么)
        with open(model_dir / "README.md", "w") as f:
            f.write(f"# Model {version}\n")
            f.write(f"## Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"## Val AUC: {metrics.get('best_val_auc', 'N/A')}\n")
            f.write(f"## Config: {json.dumps(config, indent=2)}\n")

        logger.info(f"模型已保存: {model_dir}")
        return version

    def load_model(self, model_class, version: str, device="cuda"):
        """加载模型"""
        model_dir = self.base_dir / version

        # 读配置
        with open(model_dir / "config.json") as f:
            config = json.load(f)

        # 构建模型
        model = model_class(**config)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
        model.to(device)
        model.eval()

        logger.info(f"模型已加载: {version}")
        return model, config

# 使用
manager = ModelManager("./saved_models")

# 训练后保存
config = {"embedding_dim": 32, "hidden_dim": 64, "dropout": 0.1}
metrics = {"best_val_auc": 0.78, "train_loss": 0.12}
version = manager.save_model(model, config, metrics)

# 推理时加载
model, config = manager.load_model(RecSysModel, version)
```

#### 5.5.2 推理导出:ONNX 或 TorchScript

```python
# 如果需要在非 Python 环境运行,导出 ONNX
def export_to_onnx(model, dummy_input, path="model.onnx"):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    logger.info(f"ONNX 模型已导出: {path}")

# 如果只是 Python 环境,用 TorchScript 更简单
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### 5.6 常见训练 Bug 与排查流程

#### 5.6.1 NaN 梯度排查清单

| 症状 | 常见原因 | 排查手段 |
| :--- | :--- | :--- |
| loss 直接 NaN | 学习率太大 | 打印 lr,降低到 1/10 |
| loss 逐渐 → NaN | Embedding 梯度爆炸 | 加梯度裁剪,检查 Embedding 是否有超出字典的 ID |
| loss 偶尔 NaN | 输入数据有 NaN | `assert torch.isfinite(x).all()` |
| 混合精度下 NaN | FP16 动态范围不够 | 排除 softmax/交叉熵 在 FP16 下的不稳定性 |
| 验证时 NaN 推理正常 | BN / Dropout 模式错误 | 检查 `model.eval()` 是否没调 |

```python
# 排查 NaN 的标准套路
def debug_nan(model: nn.Module, batch: Dict):
    """逐层检查梯度,定位 NaN 来源"""
    # 1. 检查输入
    for k, v in batch.items():
        if not torch.isfinite(v).all():
            logger.error(f"输入 {k} 包含 NaN/Inf!")
            return

    # 2. 注册 hook,逐层监控梯度
    nan_layers = []

    def hook_fn(name):
        def hook(grad):
            if not torch.isfinite(grad).all():
                nan_layers.append(name)
        return hook

    handles = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            handles.append(param.register_hook(hook_fn(name)))

    # 3. 正常前向+反向
    logits = model(batch)
    loss = logits.sum()
    loss.backward()

    # 4. 排查结果
    if nan_layers:
        logger.error(f"NaN 梯度出现在以下层: {nan_layers}")
    else:
        logger.info("所有梯度正常")

    for h in handles:
        h.remove()
```

#### 5.6.2 AUC 不涨 / 不收敛

```python
# 查检清单
diagnosis = """
1. 正负样本比是否严重失调? → 检查 label 分布
2. 特征编码是否有问题? → 打印前几个 batch 的 feature,肉眼检查
3. 学习率是否太小/太大? → 从 1e-3 开始,往 1e-4 和 1e-2 两边试
4. 所有 Embedding ID 是否都在词表内? → 确认没有漏掉 <UNK>
5. 验证集是否和数据穿越(time leak)? → 确认时间切分正确
6. Embedding 维度是否太小? → 从 32 开始,试 16/64
7. Batch size 是否太小(噪声太大)? → 1024 起步
8. 是否缺少归一化? → 连续特征是否做了 Z-Score?
"""
```

### 5.7 完整使用示例

```python
# ========================================
# 把一切串起来:完整训练流程
# ========================================
if __name__ == "__main__":
    # 1. 构建数据(第 4 章的 Pipeline)
    train_loader, val_loader, encoder = build_train_pipeline(
        exposure_path="data/exposure_log.tsv",
        click_path="data/click_log.tsv",
        user_profile_path="data/user_profile.tsv",
        item_attr_path="data/item_attr.tsv",
        max_seq_len=50,
        neg_ratio=3.0,
        batch_size=1024,
        val_ratio=0.1,
    )

    # 2. 构建模型
    config = {
        "embedding_dim": 32,
        "hidden_dim": 64,
        "dropout": 0.1,
        "vocab_size": 1_000_000,
    }
    model = RecSysModel(**config)

    # 3. 优化器(分层学习率!)
    optimizer = build_optimizer(
        model,
        embedding_lr=1e-2,
        mlp_lr=5e-4,
        weight_decay=1e-5,
    )

    # 4. 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6
    )

    # 5. 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=10,
        grad_clip_norm=1.0,
        grad_accum_steps=1,
        early_stop_patience=3,
        checkpoint_dir="./checkpoints/exp_001",
        use_amp=True,
    )

    # 6. 开始训练
    history = trainer.fit()

    # 7. 保存模型包
    manager = ModelManager("./saved_models")
    metrics = {"best_val_auc": trainer.best_val_metric}
    version = manager.save_model(model, config, metrics)
    print(f"训练完成,最佳 AUC={trainer.best_val_metric:.4f},模型版本={version}")
```

### 5.8 模型评估与监控:不只是 AUC

> 在 CV 里,你只需要看 Accuracy / mAP / IoU。
> 在搜推广里,**单一指标是不够的,你必须同时看多个侧面**。

#### 5.8.1 搜推广的"四维指标体系"

```python
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

def compute_recsys_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    """计算搜推广通用的四维指标"""

    # 维度 1:排序能力(AUC,GAUC)
    # AUC = 随机抽取一个正样本、一个负样本,正样本排在前面的概率
    # 范围 0.5~1.0,0.5=随机,0.8+=好
    auc = roc_auc_score(labels, preds)

    # 维度 2:校准能力(LogLoss / BCE)
    # 模型预测的概率和真实概率是否一致?
    # LogLoss 越小越好,CTR 场景通常 0.1~0.3
    logloss = log_loss(labels, preds)

    # 维度 3:预测值分布(置信度)
    # 模型预测的平均 = 真实 CTR 吗?
    predicted_ctr = preds.mean()              # 预测平均点击率
    actual_ctr = labels.mean()                # 真实平均点击率
    calibration = predicted_ctr / actual_ctr  # 校准比,应该在 0.9~1.1

    # 维度 4:KS 统计量(区分度)
    # 正负样本预测分布的 Kolmogorov-Smirnov 统计量
    pos_preds = preds[labels == 1]
    neg_preds = preds[labels == 0]
    from scipy.stats import ks_2samp
    ks_stat, _ = ks_2samp(pos_preds, neg_preds) if len(pos_preds) > 0 and len(neg_preds) > 0 else (0, 1)

    return {
        "auc": auc,
        "logloss": logloss,
        "pred_ctr": float(predicted_ctr),
        "actual_ctr": float(actual_ctr),
        "calibration": float(calibration),
        "ks": float(ks_stat),
    }

# 使用
metrics = compute_recsys_metrics(labels_np, preds_np)
print(f"AUC={metrics['auc']:.4f}, LogLoss={metrics['logloss']:.4f}")
print(f"预测 CTR={metrics['pred_ctr']:.4f}, 真实 CTR={metrics['actual_ctr']:.4f}")
print(f"校准比={metrics['calibration']:.3f}, KS={metrics['ks']:.3f}")

# 体检结果解读:
# AUC=0.75, LogLoss=0.15, calibration=1.02, KS=0.38
# → 排序能力中等,校准好,区分度尚可
#
# AUC=0.75, LogLoss=0.35, calibration=0.70
# → 排序能力中等,但校准极差!预测概率整体偏低!
# → 需要做概率校准或调整负采样比例
```

#### 5.8.2 为什么 AUC 涨了,线上却无效?——常见陷阱

```text
陷阱 1:AUC 置信区间不够宽
  测试集 100 万样本 AUC=0.750 vs 0.751(差 0.001)
  这个差异很可能只是随机波动,上线 100% 没效果
  经验:离线 AUC 涨 0.003+(即 3 个千分点)才有大概率线上有效

陷阱 2:GAUC vs AUC
  AUC 是全局的,GAUC 按 user 分组算 AUC 再平均
  GAUC 对线上效果更有参考意义
  有些改动 AUC 涨 0.002,但 GAUC 掉 0.003 → 线上一定掉

陷阱 3:离线 vs 在线分布不一致
  离线测试集是"历史"数据,线上是"未来"数据
  如果时间跨度大,特征分布可能已经漂移
```

#### 5.8.3 训练过程的实时监控

```python
class MetricsTracker:
    """
    追踪训练过程中的指标变化。
    每个 epoch 输出一个表格,只看趋势不看绝对值。
    """
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, epoch, train_loss, val_loss, val_auc, lr):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_auc"].append(val_auc)
        self.history["lr"].append(lr)

    def summary(self):
        import pandas as pd
        df = pd.DataFrame(self.history)
        print("\n===== 训练过程摘要 =====")
        print(f"最佳 AUC: {df['val_auc'].max():.4f} (epoch {df['val_auc'].idxmax() + 1})")
        print(f"最终 AUC: {df['val_auc'].iloc[-1]:.4f}")
        print(f"训练 Loss: {df['train_loss'].iloc[0]:.4f} → {df['train_loss'].iloc[-1]:.4f}")
        print(f"验证 Loss: {df['val_loss'].iloc[0]:.4f} → {df['val_loss'].iloc[-1]:.4f}")

        # 判断是否过拟合:
        # train_loss 持续下降,val_loss 开始回升 → 过拟合!
        if len(df) >= 5:
            recent_train = df['train_loss'].tail(3).mean()
            recent_val = df['val_loss'].tail(3).mean()
            overall_val = df['val_loss'].head(3).mean()
            if recent_val > overall_val and recent_train < df['train_loss'].head(3).mean():
                print("⚠️ 警告:验证 Loss 开始回升,可能过拟合!")
```

### 5.9 分布式训练入门:单卡到多卡

> 搜推广模型动辄几十亿参数(全是 Embedding),单卡训练需要几周。
> 分布式训练不是"锦上添花",而是"必修课"。

#### 5.9.1 单卡 vs 多卡:什么时候需要分布式?

```text
模型能放进一张卡(< 40GB 显存)?
├── 是 → 数据还不够大?
│   ├── 单卡训练 < 1 天 → 不需要分布式
│   └── 单卡训练 > 1 天 → 用 DataParallel 加速(数据并行)
└── 否 → 模型放不进一张卡
    ├── Embedding 太大 → 模型并行(Embedding 分到多卡)
    └── 其他 → 各种并行策略组合
```

#### 5.9.2 DataParallel(最简单的多卡,不推荐生产使用)

```python
import torch.nn as nn

# DataParallel:单机多卡,自动分割 batch
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
# 一个 batch [1024, ...] → 每张卡 [256, ...]
# 前向:4 卡并行
# 反向:梯度汇总到主卡(device_ids[0])
#
# 优点:一行代码,开箱即用
# 缺点:主卡显存比其他卡多(梯度汇总),效率不高
# 推荐度:⭐(仅调试用)
```

#### 5.9.3 DistributedDataParallel(DDP)——生产级标准

```python
# DDP 是目前最推荐的分布式训练方式
# 原理:每张卡独立维护完整的模型副本,只同步梯度

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """初始化进程组"""
    dist.init_process_group(
        backend='nccl',                    # NCCL:GPU 通信最快
        init_method='tcp://localhost:23456',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset, config):
    """DDP 训练入口(每个进程跑这个函数)"""
    setup(rank, world_size)

    # 1. 模型放到对应 GPU
    model = model.to(rank)

    # 2. 用 DDP 包装(自动处理梯度同步)
    ddp_model = DDP(model, device_ids=[rank])

    # 3. 每个进程独立的 sampler(避免重复样本!)
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,                 # 用 sampler 替代 shuffle
        num_workers=4,
        pin_memory=True,
    )

    optimizer = build_optimizer(ddp_model, **config)

    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)         # 每个 epoch 打乱顺序
        for batch in loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            labels = batch.pop("label")

            logits = ddp_model(batch)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 只在 rank 0 上保存
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "model_ddp.pt")

    cleanup()

# 启动:4 张卡
if __name__ == "__main__":
    world_size = 4
    mp.spawn(
        train_ddp,
        args=(world_size, model, dataset, config),
        nprocs=world_size,
        join=True
    )
```

**DDP 的关键点**:
```text
1. batch_size 的含义不变:DDP 下每张卡各自处理 batch_size 个样本
   总样本数 = batch_size × world_size

2. DistributedSampler 保证每个 epoch 的不同打乱:
   rank 0 拿到 [0, 1, ..., N/4-1]
   rank 1 拿到 [N/4, N/4+1, ..., N/2-1]
   每个 epoch 重新 shuffle,保证各卡数据不同

3. DDP 梯度同步:每步 backward 后自动 all-reduce 梯度
   通信开销 = 模型参数量 × 4 bytes / 通信带宽
   所以尽可能减少通信:比如梯度累积可以降低通信频率

4. BatchNorm 在 DDP 下需要特殊处理
   每个 GPU 独立算 batch 统计,和单卡行为不同
   解决方法:用 SyncBatchNorm 或干脆改用 LayerNorm
```

#### 5.9.4 模型并行(Embedding 太大时)

```python
# 当 Embedding 表超过单卡显存时
# 把 Embedding 表分片到多卡

class DistributedEmbedding(nn.Module):
    """把 Embedding 表分到多张 GPU 上"""
    def __init__(self, num_embeddings, embedding_dim, world_size):
        super().__init__()
        self.world_size = world_size
        self.embedding_dim = embedding_dim

        # 把 vocab 分成 world_size 份
        chunk_size = num_embeddings // world_size
        self.emb_chunks = nn.ModuleList([
            nn.Embedding(chunk_size, embedding_dim)
            for _ in range(world_size)
        ])

    def forward(self, indices):
        # 根据 index 找到对应的卡和本地 index
        chunk_id = indices // (num_embeddings // self.world_size)
        local_idx = indices % (num_embeddings // self.world_size)

        # 这里需要跨卡通信:从不同卡收集 embedding
        # 实际操作比这复杂得多,通常用框架内置实现
        # 如:DeepSpeed, Megatron-LM, SageMaker 等
        pass
```

#### 5.9.5 分布式训练的显存估算

```python
def estimate_distributed_memory(
    total_params: int,
    world_size: int,
    batch_size_per_gpu: int,
    seq_len: int,
    d_model: int,
    precision: str = "fp32"
):
    """
    分布式训练的显存估算。

    参数:
        total_params: 模型总参数量(包含 Embedding)
        world_size: GPU 数量
        batch_size_per_gpu: 每张卡的 batch size
        seq_len: 最大序列长度
        d_model: 模型隐藏维度
        precision: fp32(4B) or fp16(2B)
    """
    bytes_per_param = 4 if precision == "fp32" else 2

    # 模型权重(每张卡存完整副本)
    weights_mem = total_params * bytes_per_param

    # 梯度(每张卡)
    grad_mem = total_params * bytes_per_param

    # 优化器状态(Adam: 2 个状态)
    optimizer_mem = total_params * 4 * 2  # fp32 动量+方差

    # 激活值(batch_size × seq_len × d_model × 层数 × 精度)
    n_layers = 6  # 假设 6 层 Transformer
    activation_mem = batch_size_per_gpu * seq_len * d_model * n_layers * bytes_per_param

    total = (weights_mem + grad_mem + optimizer_mem + activation_mem) / (1024**3)
    return total

# 1B 参数模型,8 卡,每卡 batch=256, seq=50, d_model=512
mem = estimate_distributed_memory(
    total_params=1_000_000_000,
    world_size=8,
    batch_size_per_gpu=256,
    seq_len=50,
    d_model=512,
    precision="fp16"
)
print(f"每卡所需显存: {mem:.1f} GB")
```

### 5.10 第五章小结 & 自测

**小结**:
- 一个生产级 Trainer = AMP + 梯度累积 + 梯度裁剪 + EarlyStopping + checkpoint
- 分层学习率:Embedding 高 lr → weight_decay=0 → SparseAdam / 手动 L2
- Dropout 是搜推广最实用的正则化,Embedding 和 MLP 上都要加
- 模型保存不止权重,还要 config + metrics + version 管理
- NaN 排查 = 逐层 hook + 检查输入输出
- 评估四维体系:AUC(排序) + LogLoss(校准) + calibration(偏置) + KS(区分度)
- 分布式首选 DDP,DistributedSampler 保证数据分片

**自测题**:
1. 为什么 Embedding 参数不能用 weight_decay?会有什么后果?
2. 实现一个 EarlyStopping 类:给定 `val_loss` 列表,返回是否应该停止训练(patience=3)。
3. 写一段代码:模型前向后,手动给所有 Embedding 的 L2 范数加到 loss 上(只在被索引到的行上)。
4. 训练时 loss 从 0.69 缓慢下降到 0.68 后不动了,验证 AUC=0.5(随机猜测)。你会怎么排查?
5. 为什么验证 AUC 比训练 AUC 高很多(超出 0.05)通常是 BUG 而不是好现象?
6. (新增) DDP 和 DataParallel 的区别是什么?为什么生产环境不用 DataParallel?

---

## 第 6 章:生成式推荐与小型 LLM 设计模式的趋同化

> **这是全书的"第二篇"——把你从搜推广基础知识带入 2024-2025 年最前沿的 GenRec + Small LLM 融合设计。**
>
> 读到这里你应该已经发现:**生成式推荐模型和 30B 以下的小型语言模型的设计思路正在快速趋同**。HSTU 的 Block ≈ LLaMA 的 Block,P5 用 T5 架构做推荐与 T5 做翻译共享同一套代码,LoRA 既能微调 LLaMA 也能微调 CTR 模型。
>
> **本章目标**:用研究论文 + 代码 + 实践经验三方位,系统性地讲清楚:
> - SwiGLU 为什么替代了 ReLU?RMSNorm 为什么替代了 LayerNorm?RoPE 为什么替代了 Sinusoidal PE?
> - Decoder-only 为什么在 GenRec 和 LLM 中都赢了?GQA/MQA 省了多少显存?
> - 语义 ID tokenization(VQ-VAE/RQ-VAE)如何解决物品数量问题?
> - LoRA/QLoRA/量化/蒸馏——这些"大模型技术"怎么用在推荐上?

### 6.1 核心论点:GenRec 与 Small LLM 的"统一架构"已经出现

2024-2025 年的关键研究发现:**生成式推荐和 30B 以下语言模型的最佳架构几乎相同**。

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

### 6.2 架构组件的逐项对比与数学推导

#### 6.2.1 GLU vs FFN(权重配方对比)

**传统 FFN(ReLU/GELU)**:
$$\text{FFN}_{\text{ReLU}}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 \quad\text{——2 次投影}$$

**SwiGLU FFN(LLaMA/Mistral/Qwen/HSTU)**:
$$\text{FFN}_{\text{SwiGLU}}(x) = (\text{SiLU}(xW_{\text{gate}}) \otimes xW_{\text{up}})W_{\text{down}} \quad\text{——3 次投影}$$

**为什么多一次投影反而更好?——门控机制的梯度分析**:

```text
ReLU FFN 的梯度:
  ∂L/∂x = ∂L/∂out · W₂ᵀ · diag(1_{xW₁ > 0}) · W₁
  当 ReLU 门关闭(负值)时,梯度被完全阻断! → 神经元死亡

SwiGLU 的梯度:
  ∂L/∂x = ∂L/∂out · W_downᵀ · [SiLU(xW_gate) ⊗ (∂/∂x) + (∂SiLU/∂x) ⊗ xW_up] · [W_gate, W_up]
  两个投影都参与梯度计算,总有梯度流动 → 训练更稳定

且:参数量一致(因为 SwiGLU 的 d_ff = 8d/3 ≈ 2.67d,而 ReLU 的 d_ff=4d)
```

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

#### 6.2.2 RMSNorm——为什么去掉均值反而更好?

**数学定义对比**:

| 归一化 | 公式 | 计算量 |
| :--- | :--- | :--- |
| BatchNorm | $y = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$ | 2 次 reduce(均值+方差) |
| LayerNorm | $y = \gamma \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta$ | 2 次 reduce(均值+方差) |
| **RMSNorm** | $y = \gamma \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$ | **1 次 reduce(仅 RMS)** |

**RMSNorm 为什么能在去掉均值后保持效果?** (来自 2024 年的理论分析):

```text
原因 1:高维空间中均值 ≈ 0
  d_model = 4096,特征的均值 μ = Σx_i / 4096
  因为 x_i 服从 N(0, σ²), μ ≈ 0,误差 O(1/√d)
  减不减均值对结果影响极小

原因 2:梯度流动更干净
  LayerNorm 中减均值的操作会引入负相关:
    Cov(x_i - μ, x_j - μ) = -σ²/d (负的!)
  这种相关性在反向传播时会造成梯度"拉扯"
  RMSNorm 没有这步,梯度更直接

原因 3:计算节省
  从 2 次 reduce 到 1 次 reduce,节省约 5~10% 训练时间
  对于需训练数周的大模型,这是巨大的收益
```

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

#### 6.2.3 RoPE——旋转位置编码

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

def apply_rotary_emb(xq, xk, cos, sin):
    """对 Q 和 K 应用 RoPE"""
    # xq: [B, nh, T, hd], cos/sin: [T, hd]
    # 把 x 分成两半,旋转后半并加回前半
    xq_rot = (xq * cos) + (rotate_half(xq) * sin)
    xk_rot = (xk * cos) + (rotate_half(xk) * sin)
    return xq_rot, xk_rot
```

#### 6.2.4 GQA/MQA——注意力变体的显存节省

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

#### 6.2.5 Decoder-Only vs Encoder-Decoder vs Encoder-Only

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

### 6.3 Tokenization:从 LabelEncode 到语义 ID(VQ-VAE / RQ-VAE)

> **这是 GenRec 和传统推荐最本质的区别,也是从 CV/搜推广到 GenRec 必须跨越的一道坎。**

#### 6.3.1 问题的规模:千万级物品的 Token 化

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

#### 6.3.2 VQ-VAE vs RQ-VAE:语义 ID 的技术核心

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

#### 6.3.3 Tokenization 策略选型指南

| 应用场景 | 推荐策略 | 词表大小 | 冷启动能力 |
| :--- | :--- | :--- | :--- |
| 物品 < 10 万 | 直接 ID token | = 物品数 | ❌ 无 |
| 物品 10 万 ~ 1000 万 | **语义 ID(RQ-VAE, 4 级)** | 256~1024 | ✅ ✅ |
| 物品 > 1000 万 | 语义 ID + 哈希分桶 | 256~1024 + 额外信号 | ✅ ✅ |
| 只需要文本描述 | BPE/Unigram 子词 token | 32K~100K | ✅(零样本) |
| 多模态(文本+图片) | **多模态 RQ-VAE** | 256~1024 | ✅ ✅(最强的) |

### 6.4 PEFT 与量化:如何在小模型上做高效适配

> **在 GenRec 中,你可能需要针对不同领域、不同客户、不同任务分别适配模型。全量微调 30B 模型不现实。LoRA 和量化让你的消费级 GPU 也能微调。**

#### 6.4.1 LoRA 的数学——低秩适配

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

#### 6.4.2 QLoRA——4-bit 量化 LoRA

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

#### 6.4.3 量化推理(GPTQ / AWQ / GGUF)

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

#### 6.4.4 知识蒸馏:Teacher-Student 的推荐实践

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

### 6.5 推理优化:KV-cache, PagedAttention, Continuous Batching

#### 6.5.1 KV-cache 的数学

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

#### 6.5.2 PagedAttention 与 Continuous Batching

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

### 6.6 综合实践:一个"LLM 架构做推荐"的端到端示例

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

### 6.7 GenRec + Small LLM 架构选型速查表

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

### 6.8 第六章小结 & 自测

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

## 第 7 章:高级工程实践——从训练到部署的完整技术栈

> **写好模型只是第一步。理解训练流程的每一个环节、调试技巧、部署方式,才是高级工程师和初级工程师的分水岭。**
>
> 这一章覆盖:断点续训的完整方案、PyTorch 进阶函数、系统化调试、注意力新方法(MLA/FlashAttention)、Shell 脚本与配置工程。

### 7.1 断点续训(Checkpoint & Resume)——你绝不能只保存模型权重

**必须保存的状态**:只保存 `model.state_dict()` 是不够的。完整的 checkpoint 必须包含:

```python
checkpoint = {
    # 1. 模型参数(必要)
    "model_state_dict": model.state_dict(),

    # 2. 优化器状态(必要,否则动量/方差丢失→续训初期不稳定)
    "optimizer_state_dict": optimizer.state_dict(),

    # 3. 学习率调度器(必要,否则 LR 从初始值开始→步长突变)
    "scheduler_state_dict": scheduler.state_dict(),

    # 4. GradScaler(混合精度训练时必须保存,否则 scale 重置为 1→可能 underflow)
    "scaler_state_dict": scaler.state_dict(),

    # 5. RNG 状态(推荐,确保续训后数据顺序一致)
    "rng_state": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    },

    # 6. 训练元数据(必要)
    "epoch": epoch,
    "global_step": global_step,
    "best_metric": best_metric,
}
```

**为什么优化器状态必须保存?**
```python
# Adam 的每个参数维护两个状态:
# exp_avg: 动量(一阶矩) — 记录了梯度的"惯性"
# exp_avg_sq: 自适应学习率(二阶矩) — 记录了梯度平方的指数移动平均

# 如果不保存,续训时:
# exp_avg = 0, exp_avg_sq = 0 → 学习率从最大开始 → 步长过大 → loss 震荡
# 通常需要 500~2000 步才能恢复到暂停前的训练状态

# 分布式场景额外注意:
# FSDP 的 state_dict 需要通过 get_state_dict / set_state_dict 接口,
# 因为参数被分片了,直接 .state_dict() 拿不到完整的
```

**Safetensors vs torch.save**:
```python
# torch.save:基于 pickle,可执行任意代码(安全隐患),不支持零拷贝
torch.save(model.state_dict(), "model.pt")

# safetensors:纯数据格式,无 pickle,支持内存映射,支持懒加载
from safetensors.torch import save_file, load_file
save_file(model.state_dict(), "model.safetensors")

# 懒加载:只加载需要的权重
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    emb_weight = f.get_tensor("embedding.weight")  # 只加载这一个 tensor
```

### 7.2 PyTorch 进阶函数——从调包到理解内部

**F.linear vs nn.Linear 的本质区别**:
```python
# nn.Linear 是一个模块,存储 weight 参数
# F.linear 是函数调用,不存储参数
# 两者数学等价: y = x @ W^T + b

# F.linear 的签名: F.linear(input, weight, bias=None)
# weight.shape = [out_features, in_features]  ← 注意是转置后的

# 为什么有人用 F.linear? 在自定义 autograd Function 中,
# 你不能用 nn.Linear(因为它会创建 Parameter),必须用 F.linear.
```

**torch.scatter_add / scatter_reduce——Embedding 梯度累积的核心**:
```python
# Embedding 的反向传播需要在被索引到的位置"累加"梯度
# 这本质上就是 scatter_add 操作

# indices: 被索引到的位置
# grad: 要累加的梯度
emb_grad = torch.zeros(vocab_size, dim)
emb_grad.scatter_add_(0, indices.unsqueeze(-1).expand(-1, dim), grad)
```

**clip_grad_norm_ vs clip_grad_value_ 的数学区别**:
```python
# clip_grad_norm_:全局 L2 范数裁剪
#   total_norm = sqrt(Σ||g_i||²)
#   if total_norm > max_norm: g_i = g_i * max_norm / total_norm
#   优点:保持梯度方向不变,所有梯度等比例缩小

# clip_grad_value_:逐元素裁剪
#   g_ij = clip(g_ij, -value, value)
#   优点:绝对限制单个梯度的最大值
#   缺点:可能改变梯度方向

# 搜推广推荐:clip_grad_norm_ + max_norm=1.0
```

### 7.3 系统性 Debug 技术

**NaN 梯度定位**:
```python
# 方案 1:set_detect_anomaly(精度低但定位准)
torch.autograd.set_detect_anomaly(True)
loss.backward()  # 第一次 NaN 出现时会打印堆栈
# 注意:会显著降低训练速度,只在 debug 时开启

# 方案 2:register_hook 逐层检查
for name, param in model.named_parameters():
    param.register_hook(lambda grad, n=name: print(f"NaN in {n}")
                        if torch.isnan(grad).any() else None)

# 方案 3:检查输入
assert torch.isfinite(batch["input"]).all(), "输入包含 NaN/Inf!"
```

**显存分析**:
```python
# torch.cuda.memory_summary() 输出当前的显存分配器状态
# 关键指标:
# - allocated: 当前被 tensor 占用的显存
# - reserved: PyTorch 分配器从 GPU 申请的总显存(包含缓存)
# - 如果 reserved >> allocated → 显存碎片,用 max_split_size_mb 控制

# 显存快照(可视化分析):
torch.cuda.memory._record_memory_history()
# ... 跑训练 ...
torch.cuda.memory._dump_snapshot("snapshot.pickle")
# 打开 https://pytorch.org/memory_viz 导入快照查看

# 环境变量控制:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512  # 减少碎片
```

**Profiler 分析瓶颈**:
```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()

# 输出分析:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")  # 在 chrome://tracing 中查看
```

### 7.4 注意力新方法(2023-2025)

**FlashAttention 2/3**:核心思想是通过 tiling 避免在 HBM 中物化完整的 N×N 注意力矩阵。标准注意力需要 O(N²) 内存,FlashAttention 只需要 O(N)。FA2 减少了 non-matmul FLOPs,FA3 在 H100 上利用了 FP8。

**Multi-head Latent Attention(MLA)——DeepSeek-V2/V3 的关键创新**:
```text
标准 MHA: 每个 token 的 KV cache = 2 × n_heads × d_head × seq_len
MLA:      每个 token 的 KV cache = 2 × d_c × seq_len (d_c << n_heads × d_head)

数学上:
  k_t = W_UK · (W_DK · h_t)  — 先用 W_DK 压缩(降维),再用 W_UK 展开(升维)
  KV cache 只存压缩后的 w_DK · h_t(低维),展开操作在 attention 计算时即时做

效果: DeepSeek-V2 用 MLA 把 KV cache 减少了 75%,同时保持甚至提升了模型质量。
```

**Sliding Window Attention(Mistral)**:每个 token 只关注窗口内的邻居(如 4096 tokens),窗口外的不参与注意力计算。复杂度从 O(T²) 降到 O(T×W),W 是窗口大小。

**Multi-Token Prediction(MTP, DeepSeek-V3)**:训练时同时预测多个未来的 token,而不是只预测下一个。推理时可以用"投机解码"来加速——用草稿模型快速生成,主模型验证。

### 7.5 Shell 脚本与配置工程

**训练启动脚本标准模板**:
```bash
#!/bin/bash
set -e                          # 任何命令失败立即退出
set -x                          # 打印执行的每条命令

# NCCL 配置(分布式训练必须)
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=1     # GPU 直接 RDMA
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

# torchrun 启动
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config=configs/train.yaml \
    --experiment_dir=$EXP_DIR
```

**Slurm 多节点脚本**:
```bash
#!/bin/bash
#SBATCH --job-name=recsys
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --signal=USR1@90  # 被抢占前 90 秒发信号

# 捕获信号做优雅退出
cleanup() {
    echo "收到退出信号,保存 checkpoint..."
    python -c "import torch; torch.save(...)"  # 紧急保存
    exit 0
}
trap cleanup SIGTERM SIGUSR1
```

**YAML 配置设计**:
```yaml
# 好的配置设计原则:
# 1. 可复现:一份配置对应一次实验的全部参数
# 2. 可组合:用 defaults 和 override 组合不同场景
# 3. 可验证:配置有 schema 检查

model:
  name: "TwoTowerRecommender"
  embedding_dim: 128
  hidden_dims: [512, 256, 128]
training:
  batch_size: 8192
  learning_rate: 0.001
  optimizer: "AdamW"
data:
  train_path: "/data/train"
  val_path: "/data/val"
  num_workers: 32
```

**配置验证(Pydantic)**:
```python
from pydantic import BaseModel, validator

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float = 0.001

    @validator('batch_size')
    def power_of_two(cls, v):
        if v & (v - 1) != 0:
            raise ValueError("batch_size 应该是 2 的幂")
        return v

    @validator('learning_rate')
    def positive(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("learning_rate 应在 (0,1] 范围")
        return v
```

### 7.6 理论知识补充

**为什么 Normalization 能加速收敛?——Lipschitz 连续性论证**:
```text
没有归一化时,损失函数的 Lipschitz 常数 L 可能随网络深度指数增长。
梯度下降的收敛步长受限于 1/L → L 太大 → 必须用小学习率。

BatchNorm/LayerNorm 把每层的输出的方差约束在 1 左右,
使得整个网络的 Lipschitz 常数 L 被限制在 O(√d) 级别,
从而可以用更大的学习率 → 加速 10~100 倍。

数学: ||LN(x) - LN(y)|| ≤ √3 / min(σ_i) · ||x - y||
```

**Sharp vs Flat Minima(锐利 vs 平坦最小值)**:
```text
平坦最小值:最优点附近损失变化平缓 → Hessian 特征值小 → 泛化好
锐利最小值:最优点附近损失变化剧烈 → Hessian 特征值大 → 泛化差

为什么平坦最小值泛化好?
  PAC-Bayes 泛化界:泛化误差 ≤ 训练误差 + √(KL/2n)
  KL 项衡量了参数的"锐利度"——越平坦的 minima,KL 项越小

实践:大 batch 倾向于收敛到锐利最小值,小 batch 到平坦最小值
这就是为什么小 batch(256~1024) 通常比大 batch(8192+) 泛化更好的原因
```

**InfoNCE Loss 的温度参数作用**:
```text
InfoNCE: L = -log(exp(sim⁺/τ) / (exp(sim⁺/τ) + Σexp(sim⁻/τ)))

温度 τ 控制了对比学习的"聚焦程度":
- τ 小(<0.1):分布尖锐,只关注最难的负样本 → 适合大 batch
- τ 大(>0.3):分布平滑,所有负样本都参与 → 适合小 batch

温度的学习本质:控制对比学习对难负样本的惩罚权重
```

### A.1 经典必读文献

| 主题 | 论文/资源 | 一句话说明 |
| :--- | :--- | :--- |
| 推荐系统框架 | [Wide&Deep (Google 2016)](https://arxiv.org/abs/1606.07792) | 第一个把记忆+泛化结合到工业 CTR 的模型 |
| 特征自动交互 | [DeepFM (华为 2017)](https://www.ijcai.org/proceedings/2017/0368.pdf) | 不用手工特征交叉了 |
| 序列建模 | [DIN (阿里 KDD 2018)](https://arxiv.org/abs/1706.06978) | 用 Attention 捕获用户动态兴趣 |
| 兴趣演化 | [DIEN (阿里 AAAI 2019)](https://arxiv.org/abs/1809.03672) | 在 DIN 基础上加入时序建模 |
| 多任务学习 | [MMoE (Google KDD 2018)](https://dl.acm.org/doi/10.1145/3219819.3220007) | 多目标优化的基础框架 |
| 多任务进阶 | [PLE (腾讯 RecSys 2020)](https://arxiv.org/abs/2011.02313) | 解决了 MMoE 的跷跷板问题 |
| 图推荐 | [LightGCN (SIGIR 2020)](https://arxiv.org/abs/2002.02126) | 去繁就简的图协同过滤 |
| 生成式推荐 | [P5 (RecSys 2022)](https://arxiv.org/abs/2203.13366) | Everything is a sequence |

### A.2 后续章节预告(如本系列继续更新)

```text
【第二篇:搜推广经典模型从 0 实现】
- 第 6 章:逻辑回归与 FM(了解最基础的推荐模型)
- 第 7 章:Wide&Deep 与 DeepFM(工业标配模型)
- 第 8 章:DIN 与 DIEN(序列建模)
- 第 9 章:MMoE 与 PLE(多任务学习)
- 第 10 章:从模型到系统:召回 → 粗排 → 精排

【第三篇:工程进阶】
- 第 11 章:大规模特征存储(Redis / Tair / Faiss)
- 第 12 章:分布式训练(PS / AllReduce / 模型并行)
- 第 13 章:A/B 实验与线上指标
- 第 14 章:模型压缩(量化 / 蒸馏 / 剪枝)
- 第 15 章:搜推广全链路实战项目
```

> **最后的话**:
>
> 你读到这里,已经走过了从 Python 内存模型到 PyTorch 底层、从数据处理到完整训练 Pipeline 的完整旅程。
>
> 从 CV 转搜推广,最难的从来不是"学新模型",而是**切换解决问题的思维范式**:
> - CV 里你问"这个模型准不准?",搜推广里你问"这个特征靠不靠谱?"
> - CV 里你纠结"这个 CNN 层数够不够?",搜推广里你纠结"这个 ID 的 Embedding 有没有被充分更新?"
> - CV 里你关注"测试集精度",搜推广里你关注"离线涨了 0.5 个点,线上 AUC 会不会也涨?"
>
> 这份教程是个起点,不是终点。建议你:打开团队的项目代码,带着本篇的知识点,一行一行地看。
> 你会发现——原来 `features.get(name, default)` 是在这里用的;原来 `nn.Embedding` 的 `padding_idx=0` 是在这里防止污染;原来这个 `collate_fn` 是在处理变长序列。
>
> 祝你在搜推广的路上越走越顺!
> - 完整的可复现实验配置管理


## 第一部分 · 深度扩展:多视角系统展开

> 本部分是 "第01章-Python与PyTorch基础" 的全面延伸。每个章节从不同视角展开,系统补充原内容未覆盖的数学推导、工程实现、典型案例。所有新增内容自成体系,可独立阅读。

### 1. 第01章-Python与PyTorch基础 - Python 基础深度分析

**核心议题**: 在 "第01章-Python与PyTorch基础" 的语境下,Python 基础 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

Python 基础 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 Python 基础 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 Python 基础 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 Python 基础 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 Python 基础 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 Python 基础 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 Python 基础 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 Python 基础 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第01章-Python与PyTorch基础 的另一个视角。


---

### 2. 第01章-Python与PyTorch基础 - PyTorch 框架深度分析

**核心议题**: 在 "第01章-Python与PyTorch基础" 的语境下,PyTorch 框架 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

PyTorch 框架 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 PyTorch 框架 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 PyTorch 框架 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 PyTorch 框架 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 PyTorch 框架 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 PyTorch 框架 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 PyTorch 框架 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 PyTorch 框架 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第01章-Python与PyTorch基础 的另一个视角。


---

### 3. 第01章-Python与PyTorch基础 - 工程实践深度分析

**核心议题**: 在 "第01章-Python与PyTorch基础" 的语境下,工程实践 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

工程实践 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 工程实践 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 工程实践 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 工程实践 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 工程实践 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 工程实践 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 工程实践 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 工程实践 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第01章-Python与PyTorch基础 的另一个视角。


---

### 4. 第01章-Python与PyTorch基础 - 训练全流程深度分析

**核心议题**: 在 "第01章-Python与PyTorch基础" 的语境下,训练全流程 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

训练全流程 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 训练全流程 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 训练全流程 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 训练全流程 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 训练全流程 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 训练全流程 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 训练全流程 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 训练全流程 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第01章-Python与PyTorch基础 的另一个视角。


---

### 5. 第01章-Python与PyTorch基础 - 可视化监控深度分析

**核心议题**: 在 "第01章-Python与PyTorch基础" 的语境下,可视化监控 是关键子方向之一。本节系统展开该方向的数学理论、工程实现、典型案例。

**1. 问题陈述**

可视化监控 面临的核心问题可以用以下形式化语言描述:

设输入空间为 $\mathcal{X}$,输出空间为 $\mathcal{Y}$,目标是从数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ 学习映射 $f: \mathcal{X} \to \mathcal{Y}$。在 可视化监控 场景下,该映射具有以下特殊结构:

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \to \mathcal{Z}$ 是特征提取器,$g: \mathcal{Z} \to \mathcal{Y}$ 是任务头。学习的优化目标为:

$$\min_{h, g} \sum_{i=1}^N \mathcal{L}(g(h(x_i)), y_i) + \lambda \cdot R(h, g)$$

其中 $\mathcal{L}$ 是损失函数,$R$ 是正则化项,$\lambda$ 是权衡超参。

**2. 数学基础**

### 2.1 关键定义

**定义 1**: 设 $\mathcal{H}$ 是 Hilbert 空间,核函数 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 满足:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 是特征映射。

**定义 2**: 经验风险最小化 (ERM) 在 可视化监控 中定义为:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)$$

**定义 3**: 结构风险最小化 (SRM) 加入复杂度惩罚:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i) + \lambda \cdot \|f\|^2_\mathcal{H}$$

### 2.2 关键定理

**定理 1 (泛化界)**: 在 可视化监控 设定下,以概率 $1 - \delta$:

$$R(\hat{f}) - R(f^*) \leq O\left(\sqrt{\frac{\log(1/\delta) + \log|\mathcal{F}|-V}{N}}\right)$$

其中 $V$ 是 VC 维。

**定理 2 (收敛性)**: 在 可视化监控 中,梯度下降法以 $O(1/\sqrt{T})$ 速率收敛到最优解的 $\epsilon$-邻域。

**定理 3 (一致性)**: 当 $N \to \infty$ 时,ERM 解 $\hat{f}$ 几乎必然收敛到 Bayes 最优 $f^*$。

### 2.3 关键不等式

Jensen 不等式在 可视化监控 中常用:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)], \quad \text{其中} \phi \text{ 是凸函数}$$

Cauchy-Schwarz 不等式:

$$\langle u, v \rangle \leq \|u\| \cdot \|v\|, \quad \text{等号成立当且仅当} u \propto v$$

Hölder 不等式:

$$\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{1/p} (\mathbb{E}[|Y|^q])^{1/q}, \quad 1/p + 1/q = 1$$

**3. 优化方法**

### 3.1 一阶梯度法

SGD 更新规则:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t; x_{i_t})$$

其中 $\eta_t$ 是学习率,$x_{i_t}$ 是第 $t$ 步随机采样的样本。

Adam 改进:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### 3.2 学习率调度

Cosine decay:

$$\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$$

Warmup + decay 策略:

$$\eta_t = \begin{cases} \eta_{max} \cdot t / T_{warmup}, & t \leq T_{warmup} \\ \eta_{max} \cdot \text{schedule}(t), & t > T_{warmup} \end{cases}$$

### 3.3 二阶方法

Newton 法:

$$\theta_{t+1} = \theta_t - [\nabla^2 \mathcal{L}(\theta_t)]^{-1} \nabla \mathcal{L}(\theta_t)$$

L-BFGS 近似逆 Hessian:

$$\theta_{t+1} \approx \theta_t - \alpha_t H_k \nabla \mathcal{L}(\theta_t)$$

其中 $H_k$ 由最近 $m$ 次梯度迭代近似。

**4. 实现细节**

### 4.1 完整训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class SubtopicModel(nn.Module):
    """针对 {subtopic} 的模型骨架"""
    
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        # Mean pooling
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.head(h)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler=None):
    """单步训练:含梯度累积、混合精度、梯度裁剪"""
    x, y, mask = batch
    logits = model(x, mask)
    loss = F.cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    # 计算准确率
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    
    return {'loss': loss.item(), 'acc': acc, 'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        x, y, mask = [b.to(device) for b in batch]
        logits = model(x, mask)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return {'loss': total_loss / total, 'acc': correct / total}
```

### 4.2 数据处理流水线

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubtopicDataset(Dataset):
    """针对 {subtopic} 的数据集包装"""
    
    def __init__(self, data, tokenizer=None, max_length=512, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = self._process_input(sample['input'])
        y = sample['label']
        
        if self.augment and np.random.rand() < 0.5:
            x = self._augment(x)
        
        return x, y
    
    def _process_input(self, raw):
        if self.tokenizer is None:
            return torch.tensor(raw, dtype=torch.float32)
        encoded = self.tokenizer(raw, max_length=self.max_length, 
                                  truncation=True, padding='max_length')
        return torch.tensor(encoded['input_ids'])
    
    def _augment(self, x):
        # 简单的高斯噪声数据增强
        noise = torch.randn_like(x) * 0.01
        return x + noise


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
```

### 4.3 评估与可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path='training.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def full_evaluation_report(y_true, y_pred, class_names):
    """完整评估报告"""
    print("=" * 60)
    print("分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.close()
```

**5. 评估指标体系**

在 可视化监控 中,常用以下指标:

| 指标 | 公式 | 含义 | 目标 |
| :--- | :--- | :--- | :--- |
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | 整体正确率 | > 0.85 |
| Precision | $\frac{TP}{TP + FP}$ | 查准率 | 视业务 |
| Recall | $\frac{TP}{TP + FN}$ | 查全率 | 视业务 |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ | 调和均值 | > 0.80 |
| AUC-ROC | $\int_0^1 TPR dFPR$ | 排序能力 | > 0.90 |
| LogLoss | $-\frac{1}{N} \sum y \log p$ | 校准度 | 越小越好 |
| NDCG@k | $\frac{DCG@k}{IDCG@k}$ | 排序质量 | 越高越好 |

**6. 常见陷阱与最佳实践**

1. **数据质量**: 投入 50% 的时间做数据清洗和验证。
2. **特征工程**: 特征比模型更重要 (Andrew Ng)。
3. **类别不平衡**: 使用 focal loss、class-weighted loss、SMOTE。
4. **数据泄漏**: 严格按时间或用户划分,先 split 再做特征工程。
5. **过拟合**: 早停 + 正则化 + 数据增强 + Dropout。
6. **训练不稳定**: warmup + 梯度裁剪 + 适当的初始化。
7. **超参敏感**: 网格搜索 + 贝叶斯优化 + 学习率扫描。
8. **复现性**: 固定随机种子 + 记录环境 + 版本控制。

**7. 工业落地清单**

```text
数据准备
  □ 数据来源明确,合规
  □ 训练/验证/测试严格隔离
  □ 类别分布均衡,或加权处理
  □ 异常值检测与处理
  □ 数据增强策略

模型训练
  □ 选择合适的模型规模
  □ 合理的学习率与调度
  □ 损失函数与业务目标对齐
  □ 验证集调参,测试集只评估一次
  □ 记录所有超参与日志

部署上线
  □ 模型导出 (ONNX / TorchScript)
  □ 推理优化 (量化 / 剪枝 / 蒸馏)
  □ 服务化 (TF Serving / TorchServe / Triton)
  □ 监控告警 (延迟 / QPS / 准确率)
  □ 灰度发布 / AB 测试
  □ 回滚预案
```

**8. 进阶资源**

### 8.1 论文推荐

- *Attention Is All You Need* (Vaswani et al., 2017)
- *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
- *Deep Residual Learning* (He et al., 2016)
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2015)
- *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (Srivastava et al., 2014)

### 8.2 课程与书籍

- Stanford CS231n (CNN for Visual Recognition)
- Stanford CS224n (NLP with Deep Learning)
- MIT 6.S191 (Introduction to Deep Learning)
- *Deep Learning* (Goodfellow, Bengio, Courville)
- *Dive into Deep Learning* (开源,含 PyTorch 实现)

### 8.3 开源工具

- PyTorch / TensorFlow / JAX
- Hugging Face Transformers
- Ray / DeepSpeed / FSDP
- Weights & Biases / MLflow / TensorBoard
- ONNX / TensorRT / OpenVINO

**9. 小结**

本节系统展开了 可视化监控 的核心内容,涵盖数学基础、优化方法、实现细节、评估体系、陷阱对策、工业落地清单与进阶资源。下一节将继续 第01章-Python与PyTorch基础 的另一个视角。



## 深度扩展 · 多视角系统展开(续)

### 110. 第01章-Python与PyTorch基础 - 最佳实践 视角

**核心议题**: 本节从最佳实践视角切入,系统展开 "第01章-Python与PyTorch基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第01章-Python与PyTorch基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 最佳实践视角的展开**

从最佳实践视角出发,我们可以把 "第01章-Python与PyTorch基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调最佳实践视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第01章-Python与PyTorch基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"最佳实践" 视角,但落到代码层面是类似的骨架:

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

评估 "第01章-Python与PyTorch基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第01章-Python与PyTorch基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第01章-Python与PyTorch基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第01章-Python与PyTorch基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从最佳实践视角切入,系统展开了 "第01章-Python与PyTorch基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第01章-Python与PyTorch基础 - 常见陷阱 视角

**核心议题**: 本节从常见陷阱视角切入,系统展开 "第01章-Python与PyTorch基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第01章-Python与PyTorch基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 常见陷阱视角的展开**

从常见陷阱视角出发,我们可以把 "第01章-Python与PyTorch基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调常见陷阱视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第01章-Python与PyTorch基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"常见陷阱" 视角,但落到代码层面是类似的骨架:

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

评估 "第01章-Python与PyTorch基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第01章-Python与PyTorch基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第01章-Python与PyTorch基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第01章-Python与PyTorch基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从常见陷阱视角切入,系统展开了 "第01章-Python与PyTorch基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第01章-Python与PyTorch基础 - 性能优化 视角

**核心议题**: 本节从性能优化视角切入,系统展开 "第01章-Python与PyTorch基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第01章-Python与PyTorch基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 性能优化视角的展开**

从性能优化视角出发,我们可以把 "第01章-Python与PyTorch基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调性能优化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第01章-Python与PyTorch基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"性能优化" 视角,但落到代码层面是类似的骨架:

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

评估 "第01章-Python与PyTorch基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第01章-Python与PyTorch基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第01章-Python与PyTorch基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第01章-Python与PyTorch基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从性能优化视角切入,系统展开了 "第01章-Python与PyTorch基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第01章-Python与PyTorch基础 - 进阶方向 视角

**核心议题**: 本节从进阶方向视角切入,系统展开 "第01章-Python与PyTorch基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第01章-Python与PyTorch基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 进阶方向视角的展开**

从进阶方向视角出发,我们可以把 "第01章-Python与PyTorch基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调进阶方向视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第01章-Python与PyTorch基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"进阶方向" 视角,但落到代码层面是类似的骨架:

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

评估 "第01章-Python与PyTorch基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第01章-Python与PyTorch基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第01章-Python与PyTorch基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第01章-Python与PyTorch基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从进阶方向视角切入,系统展开了 "第01章-Python与PyTorch基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第01章-Python与PyTorch基础 - 可视化 视角

**核心议题**: 本节从可视化视角切入,系统展开 "第01章-Python与PyTorch基础" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第01章-Python与PyTorch基础" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 可视化视角的展开**

从可视化视角出发,我们可以把 "第01章-Python与PyTorch基础" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调可视化视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第01章-Python与PyTorch基础" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"可视化" 视角,但落到代码层面是类似的骨架:

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

评估 "第01章-Python与PyTorch基础" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第01章-Python与PyTorch基础" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第01章-Python与PyTorch基础" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第01章-Python与PyTorch基础" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从可视化视角切入,系统展开了 "第01章-Python与PyTorch基础" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



