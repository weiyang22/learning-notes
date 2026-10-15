---
title: "第04章-10-算法与复杂度-P·NP·NP-hard与深度学习的关系"
tags:
  - 数学
  - 算法与复杂度
  - NP
  - hard与深度学习的关系
description: "本章要点：\"Out of Memory\"弹出来。你下意识地想到:序列长度 512,attention 是 O(n²) = 262144,够大,但 GPU 应该扛得…"
chapter_id: 数学-10-04
last_updated: 2026-07-14
---

# 第04章-10-算法与复杂度-P·NP·NP-hard与深度学习的关系

"Out of Memory"弹出来。你下意识地想到:序列长度 512,attention 是 O(n²) = 262144,够大,但 GPU 应该扛得住。然后你把序列拉到 8192,显存爆了。128 倍的 n²。
>
> 但你想过没:
> - **Transformer 的注意力真的是 O(n²) 吗?Flash Attention 怎么把它压到 O(n)?**
> - **"推荐系统召回为什么用双塔"在复杂度上是怎么算账的?**
> - **GNN 的感受野指数级扩张,数学上怎么说?**
> - **"深度学习是 NP-hard 问题"这句话到底在说什么?**
> - **KV-Cache 让推理从 O(n²) 降到 O(n),代价是什么?**
> - **训练一个 LLaMA 7B 模型,FLOPs 怎么估算?**
>
> 这些问题的答案,全部藏在**算法与计算复杂度理论**里。
>
> 这一章我们要做的事情很朴素:把 PyTorch 那些看似"理所当然"的复杂度判断,翻译回严格的数学语言。看完之后你会发现:
>
> - **Transformer attention 的 O(n²) = "所有 token 两两配对"**,等价于"求集合 $\{(i,j) : 1 \le i,j \le n\}$ 上的函数"
> - **Flash Attention 通过分块 + 重计算**,把显存从 O(n²) 降到 O(n),但 FLOPs 不变
> - **双塔召回 = O(|U|·|I|) → O(|U| + |I|)**,省下一个乘积量级
> - **GNN 的 k 层消息传递 = k 跳邻居**,感受野大小是 O(d^k)
> - **"深度学习训练"是连续优化**,**"神经网络架构搜索 (NAS)"是组合优化**,后者落入 NP-hard
> - **P vs NP 是千禧年七大问题之一**,工程师要知道"它不是空想,而是工程决策的依据"**
>
> 算法复杂度是离散数学的"计算视角",前面 9 章讲的集合、关系、图、组合,都是它的"数据"。理解了这一章,你再看模型设计的工程权衡,会看到"复杂度"这把尺子一直在那儿。

---

## 1. 直觉引入

### 1.1 工程师每天都在算"复杂度",但没意识到

你在 PyTorch 里写一行:

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
```

数学上是 $Q K^T$,其中 $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{n \times d}$,结果是 $n \times n$ 矩阵。**这是 O(n²d) FLOPs**。

但你心里嘀咕: "我之前上课学的是矩阵乘法是 O(n³),这怎么是 O(n²d)?"

原因:**算法复杂度是相对的**。矩阵乘法 O(n³) 是对"边长为 n 的方阵"而言,这里 $n$ 和 $d$ 是不同的维度,真正的复杂度是把"占主导地位的项"标出来。$n^2 d$ 对 $n^3$ 是因为我们更关心序列长度 $n$,而 $d$ 通常是常数(64, 128, 768)。

这就是**大 O 表示法的精髓**:抓"随输入规模增长最快的项",忽略常数和低阶项。

#### 三个工程实例:复杂度判断改变设计

**例 1:Transformer 序列长度**

| $n$ | $n^2$ | 显存 (fp16, 注意力矩阵) | 是否可行 |
| :--- | :--- | :--- | :--- |
| 512 | 262144 | 512 KB | 可 |
| 2048 | 4.2M | 8 MB | 可 |
| 8192 | 67M | 134 MB | 勉强 |
| 32768 | 1.07B | 2.1 GB | 需要 Flash Attention |
| 131072 | 17.2B | 34 GB | 必须稀疏 |

**判断**:**长序列的瓶颈是注意力矩阵的显存**,不是 FLOPs。Flash Attention 通过"分块 + 不存整个矩阵"把显存压到 O(n)。

**例 2:推荐召回为什么是双塔**

| 方案 | 复杂度 | 100 万物品 / 100 万用户 |
| :--- | :--- | :--- |
| 全交叉打分 | O(\|U\| · \|I\| · d) | 10¹⁴ FLOPs |
| 双塔 + ANN 检索 | O(\|U\| · d + \|I\| · d) | 10⁹ FLOPs |

**100 万 × 100 万 = 10¹² 配对**,全打分要算 10¹⁴ 次;双塔把"每个 query 算 100 万物品的相似度"变成"离线算好物品向量,在线只算 1 个 query × 100 万物品"。**加速 10⁵ 倍**。

**例 3:GNN 邻居爆炸**

设图平均度数 $d = 10$,跑 $k = 3$ 层消息传递。每个节点的感受野:

$$\text{感受野大小} = O(d^k) = O(10^3) = 1000$$

跑 $k = 6$ 层:$O(10^6) = 100$ 万,接近全图。这就是为什么 GNN 不会太深。

**判断**:**复杂度的阶数 = 工程上限**。O(n²) 的东西在 n = 100k 时不能用,O(n) 的东西在 n = 1M 时还能撑。

### 1.2 为什么"复杂度"在深度学习时代更重要?

过去的机器学习模型(逻辑回归、SVM、随机森林),数据规模 10⁵,模型规模 10⁶ 参数,内存 16GB 够用。

现在的深度学习模型:
- LLaMA 7B = 70 亿参数,**fp16 推理需要 14 GB 显存,fp32 训练需要 112 GB**
- GPT-3 175B = 1750 亿参数,**单 GPU 装不下,必须张量并行**
- 推荐系统 Embedding 表:1 亿物品 × 64 维 × 4 字节 = **25.6 GB**

**训练成本**:GPT-3 训练 FLOPs ≈ 3.14 × 10²³,**用 1024 张 A100 跑 30 天**。每一次 forward + backward 都是 O(n²d) 的代价。

**所以工程上必须懂复杂度**:不光是"代码跑得多快",而是"这个模型能不能训、能不能部署、成本扛不扛得住"。

### 1.3 三个直觉例子

#### 例 1:Transformer attention 为什么是 O(n²)

Self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

其中 $Q, K, V \in \mathbb{R}^{n \times d}$。**关键步骤**:

1. $Q K^T$:矩阵乘法,$(n \times d) \times (d \times n)$,结果是 $n \times n$ 矩阵 → **O(n²d) FLOPs**
2. softmax:对每行做 → **O(n²)**
3. 与 $V$ 相乘:$(n \times n) \times (n \times d)$ → **O(n²d) FLOPs**

**总 FLOPs = O(n²d)**,**显存 = O(n²) + O(nd)**。

直觉:**注意力 = "每个 token 看所有 token"** = "所有 (i, j) 配对" = O(n²)。

工程含义:
- $n = 512$(BERT):可承受
- $n = 4096$(LLaMA):需 Flash Attention
- $n = 100000$(Long-context):必须稀疏 attention

#### 例 2:推荐召回为什么"双塔 + ANN"

单塔打分(用户特征和物品特征进入同一个模型):
- 每次推荐要算 $|I|$ 次神经网络前向
- 复杂度:**O(|I| · 模型推理成本)**

双塔 + ANN:
- 离线:物品向量算好,存到向量数据库(FAISS / ScaNN)
- 在线:用户向量算 1 次,在向量库里做**近似最近邻 (ANN)** 检索
- 复杂度:**O(|I| · d + |U| · d + 检索成本)**

ANN 检索在 100 万物品库上,毫秒级,**用 HNSW 图或 IVF-PQ 量化**。

**数学抽象**:把"笛卡尔积 $U \times I$ 上的打分函数"分解成"$U$ 上的函数 + $I$ 上的函数 + 内积"。这个分解的代价是**丢失"用户-物品交叉特征"**。

#### 例 3:训练 LLaMA 7B 需要多少 FLOPs?

LLaMA 7B 参数量: $P = 7 \times 10^9$。

**前向一次(per token)的 FLOPs ≈ 2P**(乘加各算一次)。

训练数据:1.4T tokens,1 epoch 训练 FLOPs ≈ $2P \times T = 2 \times 7 \times 10^9 \times 1.4 \times 10^{12} = 1.96 \times 10^{22}$。

加上反向传播 ≈ 3 倍前向 → 总 FLOPs ≈ $6 \times 10^{22}$。

**A100 的算力**:312 TFLOPS (fp16)。1024 张 A100,理论峰值 $312 \times 10^{12} \times 1024 \approx 3.2 \times 10^{17}$ FLOPS。

实际利用率 ≈ 50%(MFU, Model FLOPs Utilization)。

**训练时间** = $6 \times 10^{22} / (3.2 \times 10^{17} \times 0.5) = 3.75 \times 10^5$ 秒 ≈ **104 小时 ≈ 4.3 天**。

这就是为什么 LLaMA 7B "训练成本 ≈ 1024 张 A100 × 30 天 ≈ 几十万美元"。

### 1.4 章节路线图

接下来 8 节, 我们依次讲:

| 节 | 主题 | 与 ML/DL 的联系 |
| :--- | :--- | :--- |
| §2 | 时间空间复杂度 (大 O/Ω/Θ/摊还/Python dict) | 工程性能调优 / Transformer 显存 |
| §3 | P/NP/NP-hard/NP-complete | 推荐召回 / NAS / 组合优化 |
| §4 | 深度学习的复杂度 | Transformer / GNN / KV-Cache / LLaMA |
| §5 | 应对 NP-hard | 近似算法 / 启发式 / ANN |
| §6 | PyTorch 代码 | 复杂度对比 / Attention 计时 / TSP 贪心 / ANN |
| §7 | 常见误区 | 4 条工程师踩坑点 |
| §8 | 面试 Q&A | L7/L8 难度 4-5 题 |
| §9 | 联系其他章节 | 图论 / 组合 / 概率 |

---

## 2. 时间空间复杂度

### 2.1 大 O 表示法:上界

#### 定义

**大 O**(Big-O):$f(n) = O(g(n))$ 当且仅当存在正常数 $c$ 和 $n_0$,使得对所有 $n > n_0$:

$$0 \le f(n) \le c \cdot g(n)$$

**直觉**:$f$ 的增长**不快于** $g$ 的常数倍。

#### 例子

| $f(n)$ | $g(n)$ | $f = O(g)$? |
| :--- | :--- | :--- |
| $5n + 3$ | $n$ | ✓ (取 $c = 6, n_0 = 3$) |
| $n^2 + 100n$ | $n^2$ | ✓ (取 $c = 2, n_0 = 100$) |
| $n \log n$ | $n^{1.5}$ | ✓ (取 $c = 1$, 大 $n$ 时 $\log n < \sqrt{n}$) |
| $2^n$ | $n^{100}$ | ✗ (指数增长快于任何多项式) |
| $n!$ | $2^n$ | ✗ ($n!$ 增长更快) |

**工程含义**:**大 O 是"上限承诺"**。说"这个算法是 O(n²)",意思是最坏情况不超过 $n^2$ 的常数倍。

### 2.2 大 Ω / 大 Θ:下界与紧界

#### 大 Ω(Big-Omega):下界

$f(n) = \Omega(g(n))$ 当且仅当存在正常数 $c$ 和 $n_0$,使得对所有 $n > n_0$:

$$f(n) \ge c \cdot g(n)$$

**直觉**:$f$ 的增长**不少于** $g$ 的常数倍。

#### 大 Θ(Big-Theta):紧界

$f(n) = \Theta(g(n))$ 当且仅当 $f(n) = O(g(n))$ 且 $f(n) = \Omega(g(n))$。

**直觉**:$f$ 和 $g$ 同阶(差常数倍)。

#### 例子:二分查找

二分查找的比较次数:最坏 $T(n) = T(n/2) + 1$,解得 $T(n) = \Theta(\log n)$。

- $T(n) = O(\log n)$:上限
- $T(n) = \Omega(\log n)$:下限(信息论证明)
- $T(n) = \Theta(\log n)$:紧界

### 2.3 常见复杂度等级

从慢到快:

```text
O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
```

**经验法则**:$n = 10^6$ 时:
- O(1):瞬间
- O(log n): 20 次操作
- O(n): 100 万次 ≈ 几 ms
- O(n log n): 2000 万次 ≈ 几十 ms
- O(n²): **10¹² 次 ≈ 不可行**
- O(2ⁿ): $n = 30$ 就爆

**实用上限**:**$n = 10^6$,O(n²) 已不可行;O(n log n) 是工程目标**。

### 2.4 空间复杂度:额外内存

#### 定义

**空间复杂度** $S(n)$:算法执行所需的**额外内存**(不含输入本身)。

#### 例子

```python
# O(1) 空间:原地反转列表
def reverse_inplace(arr):
    i, j = 0, len(arr) - 1
    while i < j:
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1

# O(n) 空间:复制列表
def reverse_copy(arr):
    return arr[::-1]  # 创建新列表
```

#### 工程例子:Transformer attention 显存

```text
前向传播显存占用:
- Q, K, V: 3 × (n × d) × 2 bytes (fp16) = 6nd bytes
- 注意力矩阵 (n × n): 2n² bytes
- softmax 输出: 2n² bytes
- 中间激活值: 多份

n = 4096, d = 128:
- Q/K/V: 6 × 4096 × 128 × 2 = 6 MB
- 注意力矩阵: 2 × 4096² = 32 MB ← 主要瓶颈!
```

**Flash Attention** 关键思路:**不存整个注意力矩阵**,分块算,显存从 O(n²) 降到 O(n)。

### 2.5 Python dict / set 的 O(1) 是怎么来的?

#### 哈希表的平均复杂度

设哈希函数 $h: U \to \{0, 1, \ldots, m-1\}$,把 $n$ 个元素塞进 $m$ 个槽位。**负载因子** $\alpha = n / m$。

- **平均查找**: $O(1 + \alpha)$
- **最坏查找**(所有 key 哈希冲突): $O(n)$

**Python dict 的工程实现**:
- 初始 $m = 8$
- $\alpha > 2/3$ 时,**扩容并重新哈希**(分摊 O(1))
- 开放寻址 + 随机化探查(降低冲突概率)

#### 摊还分析 (Amortized Analysis)

**摊还复杂度**:把"偶尔的高成本操作"分摊到"常见的低成本操作"上,得到平均每次操作的代价。

```python
# 例子:list.append
# 单次: O(1)
# 但偶尔触发"扩容"(把列表长度 × 2): O(n)
# 摊还: 仍然是 O(1)
# 证明:每 n 次 append 中,有 1 次扩容,总成本 O(n + n) / n = O(1)
```

#### 例子:Python set vs list 性能

```python
import time

N = 1_000_000
data_list = list(range(N))
data_set = set(range(N))
target = N - 1

# list 查找:O(N)
t0 = time.time()
for _ in range(1000):
    target in data_list
t_list = time.time() - t0

# set 查找:O(1)
t0 = time.time()
for _ in range(1000):
    target in data_set
t_set = time.time() - t0

print(f"list: {t_list:.3f}s, set: {t_set:.3f}s")
# list 大约慢 10000 倍以上
```

### 2.6 主定理 (Master Theorem):分治算法的复杂度

#### 形式

递归式 $T(n) = a T(n/b) + f(n)$,其中 $a$ 是子问题数,$n/b$ 是子问题规模,$f(n)$ 是分治成本。

**三种情况**:
1. 若 $f(n) = O(n^{\log_b a - \epsilon})$,则 $T(n) = \Theta(n^{\log_b a})$(子问题主导)
2. 若 $f(n) = \Theta(n^{\log_b a})$,则 $T(n) = \Theta(n^{\log_b a} \log n)$
3. 若 $f(n) = \Omega(n^{\log_b a + \epsilon})$,则 $T(n) = \Theta(f(n))$(分治成本主导)

#### 例子

**归并排序**:$T(n) = 2 T(n/2) + O(n)$, $a = 2, b = 2$, $\log_b a = 1$, $f(n) = n = \Theta(n^1)$。情况 2 → $T(n) = \Theta(n \log n)$。✓

**Karatsuba 乘法**:$T(n) = 3 T(n/2) + O(n)$, $a = 3, b = 2$, $\log_b a = \log_2 3 \approx 1.585$, $f(n) = n = O(n^{1.585 - 0.585})$。情况 1 → $T(n) = \Theta(n^{\log_2 3}) = \Theta(n^{1.585})$。

(普通乘法是 $O(n^2)$,Karatsuba 把大数乘法加速到 $O(n^{1.585})$。)

### 2.7 常见算法复杂度速览

| 算法 | 时间复杂度 | 应用 |
| :--- | :--- | :--- |
| 二分查找 | $O(\log n)$ | 有序数组查找 |
| 归并排序 | $O(n \log n)$ | 排序 |
| 哈希表查找 | $O(1)$ 摊还 | dict / set |
| 矩阵乘法(naive) | $O(n^3)$ | 线性代数 |
| Strassen 乘法 | $O(n^{2.807})$ | 大矩阵 |
| FFT | $O(n \log n)$ | 信号处理 |
| Dijkstra(优先队列) | $O((V + E) \log V)$ | 最短路 |
| Floyd-Warshall | $O(V^3)$ | 全对最短路 |

---

## 3. P / NP / NP-hard / NP-complete

### 3.1 判定问题 vs 优化问题

#### 判定问题 (Decision Problem)

**判定问题** = 答案为"是 / 否"的问题。

例:"给定图 $G$ 和整数 $k$, $G$ 是否有大小 $\ge k$ 的团(clique)?"

例:"给定整数列表,是否存在子集和等于 $S$?"

#### 优化问题 (Optimization Problem)

**优化问题** = 找"最优解"的问题。

例:"最大团的大小是多少?"

例:"子集和最接近 $S$ 的是哪个?"

#### 关键事实

**任何优化问题都能归约到判定问题**(二分搜索最优值)。

> 判定问题 = 更"原子"的形式,理论分析更简洁。

### 3.2 P 类:多项式时间可解

#### 定义

**P**(Polynomial time) = 能在**确定性图灵机**上以 $O(n^k)$(某常数 $k$)时间解决的判定问题的集合。

**直观**:"今天主流算法能搞定的问题"。

#### 例子

- 二分查找:$O(\log n)$ ✓ P
- 归并排序:$O(n \log n)$ ✓ P
- 线性规划(LP):$O(n^{2.5})$(Karmarkar) ✓ P
- 整数线性规划:**NP-hard**(见下)
- 最短路(Dijkstra):$O((V+E) \log V)$ ✓ P
- 欧拉回路判定:$O(V + E)$ ✓ P

#### 工程含义

如果一个问题是 P 类,就**有高效算法**(虽然常数可能大)。如果不在 P,**找近似算法或启发式**(见 §5)。

### 3.3 NP 类:多项式时间可验证

#### 定义

**NP**(Nondeterministic Polynomial time) = 答案能在 $O(n^k)$ 内**验证**为"是"的判定问题的集合。

**直观**:"答案好验证,但不一定好找"。

#### 例子:子集和 (Subset Sum)

**问题**:给定整数集合 $\{a_1, \ldots, a_n\}$ 和目标 $S$,是否存在子集 $T$ 使得 $\sum_{i \in T} a_i = S$?

- **验证 NP**:拿到一个子集 $T$,在 $O(n)$ 内算 $\sum_{i \in T} a_i$,对比 $S$。
- **不知道是不是 P**:没找到多项式算法,也没证明不存在。

#### NP 的图灵机视角

**NP = 非确定性图灵机能多项式时间解决的问题**。

非确定性 = 可以在每一步"猜"答案,只要存在一条猜测路径通向"是"。

**等价表述**:
- 答案可以在多项式时间内**验证**(上面)
- 存在多项式长度的"证书 / 见证"(witness),使得给定证书能在多项式时间验证为"是"

#### 显然 $P \subseteq NP$

P 问题的解能在多项式时间内找到,自然也能在多项式时间内**验证**(解本身就是证书)。

**关键未知**:$NP \subseteq P$?($P = NP$ 还是 $P \ne NP$?)

这是**千禧年七大数学问题之一**,2002 年 Clay 数学研究所悬奖 100 万美元。

### 3.4 归约 (Reduction):问题之间的"难度传递"

#### 定义

**归约**:问题 $A$ 归约到问题 $B$($A \le_P B$),意味着:**$B$ 能在多项式时间内解出,则 $A$ 也能在多项式时间内解出**。

**直觉**:$B$ 至少和 $A$ 一样难。

#### 多项式时间归约的形式

存在多项式时间可计算的函数 $f$,使得:

$$x \in A \iff f(x) \in B$$

即"$x$ 是 $A$ 的"当且仅当"$f(x)$ 是 $B$ 的"。

#### 例子:独立集 (Independent Set) $\le_P$ 团 (Clique)

**独立集**:图 $G$ 中两两不相邻的顶点集。

**团**:图 $G$ 中两两相邻的顶点集。

**归约**:对图 $G$ 取**补图** $\bar{G}$(边集取反)。$G$ 的独立集 $\Leftrightarrow$ $\bar{G}$ 的团。

所以:**独立集问题的解 = 在 $\bar{G}$ 上解团问题**。两边难度等价。

### 3.5 NP-complete:NP 中"最难"的问题

#### 定义

**NP-complete (NPC)** = 满足两个条件的判定问题:
1. $L \in NP$
2. NP 中**每个问题**都能多项式时间归约到 $L$(即 $L$ 是 NP 中"最难"的)

#### Cook-Levin 定理 (1971)

**第一个被证明是 NPC 的问题**:**布尔可满足性问题 (SAT)**。

> 给定布尔公式 $\phi$(AND / OR / NOT 组合),是否存在使 $\phi$ 为真的变量赋值?

Cook 证明了:**任何 NP 问题都能多项式时间归约到 SAT**。

**意义**:一旦证明某个问题 A 是 NPC(通过把 SAT 归约到 A),**所有 NP 问题都能归约到 A**,A 就是 NP 中"最难"的那一档。

#### 经典 NPC 问题列表

| 问题 | 描述 | 归约自 SAT? |
| :--- | :--- | :--- |
| **SAT** | 布尔可满足性 | 本身(起点) |
| **3-SAT** | 每子句 3 文字的 SAT | ✓ (归约自 SAT) |
| **团 (Clique)** | 大小为 k 的完全子图 | ✓ |
| **独立集 (Independent Set)** | 大小为 k 的独立集 | ✓ |
| **顶点覆盖 (Vertex Cover)** | 覆盖所有边的 k 个顶点 | ✓ |
| **哈密顿回路 (Hamiltonian Cycle)** | 访问每顶点恰一次的回路 | ✓ |
| **TSP (判定版)** | 长度 $\le k$ 的访问回路 | ✓ |
| **子集和 (Subset Sum)** | 子集和等于 S | ✓ |
| **0-1 背包 (判定版)** | 价值 $\ge V$ 的子集 | ✓ |
| **图着色 (Graph Coloring)** | k 色着色 | ✓ |
| **3D 匹配** | 三元组匹配 | ✓ |

### 3.6 NP-hard:至少和 NPC 一样难

#### 定义

**NP-hard (NPH)** = 满足**至少和 NPC 一样难**(所有 NP 问题能归约到它),**但不一定在 NP 中**。

#### NP-hard 与 NPC 的关系

```text
P ⊆ NP
NPC ⊆ NP (NPC 是 NP 的子集)
NPC ⊆ NPH (NPC 也属于 NPH,因为 NPC ⊆ NP)
NPH \ NPC:NP 之外的难问题(比如优化版的 NPC)
```

#### 例子:TSP 优化版

"给定图 $G$,找最短哈密顿回路"是优化问题,不在 NP(因为没有多项式大小的"证书"——证书得是路径本身,长度指数大)。但它的判定版("长度 $\le k$?")是 NPC。

所以:**TSP 优化版是 NP-hard(且非 NPC)**。

### 3.7 P / NP / NPC / NPH 关系图 (ASCII)

```text
           ┌───────────────────────────────────────┐
           │            所有问题                    │
           │                                       │
           │     ┌─────────────────────────┐       │
           │     │       P (多项式可解)    │       │
           │     │   二分查找, 最短路,     │       │
           │     │   线性规划, 排序        │       │
           │     └─────────────────────────┘       │
           │              ↑?                       │
           │         (P = NP?)                     │
           │              │ ⊆                      │
           │     ┌────────┴────────────────┐       │
           │     │     NP (多项式可验证)   │       │
           │     │   因子分解?, 图同构?    │       │
           │     │   SAT, 子集和, TSP     │       │
           │     │                         │       │
           │     │   ┌─────────────────┐   │       │
           │     │   │  NPC            │   │       │
           │     │   │  SAT, 团,       │   │       │
           │     │   │  顶点覆盖,     │   │       │
           │     │   │  子集和, TSP   │   │       │
           │     │   └─────────────────┘   │       │
           │     └─────────────────────────┘       │
           │                                       │
           │   ┌──────────────────────────┐        │
           │   │   NP-hard (NPH)          │        │
           │   │   TSP 优化版,            │        │
           │   │   神经网络架构搜索(NAS), │        │
           │   │   蛋白质折叠,            │        │
           │   │   停止问题                │        │
           │   └──────────────────────────┘        │
           │      ↑                                │
           │     (NPH ⊇ NP ∪ NPC,                 │
           │      还包括不在 NP 的"更难"问题)      │
           └───────────────────────────────────────┘
```

**关键观察**:
- **P ⊆ NP**:能解就能验证
- **NPC ⊆ NP ∩ NPH**:NP 中最难的一档
- **NPH \ NPC**:优化版 NPC + 不可判定问题(如停机问题)
- **P vs NP**:图中问号,未解之谜

### 3.8 NPC 证明模板

要证明一个新问题 $L$ 是 NPC,两步:

**Step 1**:证明 $L \in NP$(证书能在多项式时间验证)。

**Step 2**:找一个已知的 NPC 问题 $L'$,证明 $L' \le_P L$(归约)。

#### 例子:顶点覆盖是 NPC

**Step 1**:证书 = 一组顶点 $S$。验证:对每条边,至少一个端点在 $S$ 中。$O(|E|)$ ✓ NP。

**Step 2**:归约自独立集。

独立集:大小为 $k$ 的两两不相邻顶点集。

顶点覆盖:大小 $\le n - k$ 的覆盖所有边的顶点集。

**关键事实**:
$$S \text{ 是独立集} \iff V \setminus S \text{ 是顶点覆盖}$$

(独立集的补集 = 顶点覆盖,这是经典定理)

**归约**:给定独立集实例 $(G, k)$,构造顶点覆盖实例 $(G, n - k)$。$G$ 有大小 $k$ 的独立集 $\iff$ $G$ 有大小 $n - k$ 的顶点覆盖。

**多项式时间**:只算了 $n - k$,常数时间。✓

### 3.9 一个具体例子:TSP

#### 问题描述

**TSP**(Traveling Salesman Problem):给定 $n$ 个城市和两两之间的距离 $d_{ij}$,找访问每城恰一次、回到起点的最短回路。

#### 判定版

"给定 TSP 实例和整数 $L$,是否存在长度 $\le L$ 的哈密顿回路?"

#### 为什么是 NPC?

**归约自哈密顿回路**:

哈密顿回路:给定图 $G$,是否存在访问每顶点恰一次的回路?

**归约构造**:给哈密顿回路实例 $(G, V, E)$,构造 TSP 实例:
- 城市 = $G$ 的顶点
- 距离 $d_{ij} = 1$ 如果 $(i, j) \in E$,否则 $d_{ij} = 2$

$G$ 有哈密顿回路 $\iff$ TSP 实例有长度 $\le n$ 的回路(因为哈密顿回路在 TSP 中就是 $n$ 条边全用上,长度 = $n$)。

**多项式时间**:距离矩阵构造是 $O(n^2)$。✓

**已知 TSP 是 NPC**。所以"找最短访问回路"几乎肯定没有多项式算法(除非 P = NP)。

#### 工程师的应对

§5 会讲:贪心、2-opt、近似算法。TSP 虽然 NPC,但 1000 个城市的实例,**启发式 1 秒解完,误差 5% 以内**。

### 3.10 工程意义:什么时候该放弃精确算法?

**判断流程**:

```text
问题输入规模 n = ?
├─ n < 20: 暴力枚举, O(2ⁿ) 或 O(n!)
├─ n < 50: 动态规划, O(n² 2ⁿ) (Held-Karp TSP)
├─ n < 1000: 启发式 / 局部搜索
├─ n < 10⁶: 近似算法 (有理论保证)
└─ n ≥ 10⁶: 必须预处理 / 采样 / 降维
```

**经验法则**:**任何 NPC 问题,直接放弃精确算法**,转向近似 / 启发式。

---

## 4. 深度学习的复杂度

### 4.1 Transformer 的 FLOPs 和显存

输入 $X \in \mathbb{R}^{n \times d}$, $n$ = 序列长度, $d$ = 隐藏维度。

**前向 FLOPs**:
- $Q, K, V$ 投影: $3nd^2$
- $Q K^T$: $2n^2 d$
- 与 $V$ 相乘: $2n^2 d$
- 输出投影: $2nd^2$

**总前向 FLOPs ≈ 4nd² + 4n²d**。当 $n \gg d$(长序列):$O(n^2 d)$;当 $n \ll d$(短序列):$O(n d^2)$。**反向传播 ≈ 2 倍前向**(链式法则)。

#### 显存分析

| 存储项 | 大小 | 说明 |
| :--- | :--- | :--- |
| $Q, K, V$ | $3 n d$ | 投影后的中间量 |
| $Q K^T$ | $n^2$ | **注意力矩阵,瓶颈** |
| softmax 输出 | $n^2$ | 注意力权重 |
| 反向梯度 | 与前向同等 | 训练时翻倍 |

**前向显存 ≈ $2 n^2$ bytes (fp16) + $3 n d$ bytes**。**训练显存 ≈ 4 倍以上**(含优化器 + 激活)。

### 4.2 Flash Attention:把 O(n²) 显存压到 O(n)

**核心思想**:**不存整个注意力矩阵 $Q K^T$**,**分块 (tiling)** 计算。

```python
# 传统 attention:O(n²) 显存
S = Q @ K.T  # n × n 矩阵,完整存
out = F.softmax(S, dim=-1) @ V

# Flash Attention:分块,O(n) 显存
for i in range(0, n, B):
    Q_i = Q[i:i+B]  # (B, d)
    for j in range(0, n, B):
        S_ij = Q_i @ K[j:j+B].T  # (B, B) 小矩阵
        O_i += F.softmax(S_ij, dim=-1) @ V[j:j+B]
```

**关键**:**FLOPs 不变**(还是 $O(n^2 d)$),但显存从 $O(n^2)$ 降到 $O(n)$(不存整个矩阵)。

| $n$ | 传统显存 | Flash 显存 |
| :--- | :--- | :--- |
| 2048 | 8 MB | 24 KB |
| 8192 | 128 MB | 96 KB |
| 32768 | 2 GB | 384 KB |

显存减少 → 可以用更大 batch / 更长序列 → 实际 wall-clock 加速 2-4×。

### 4.3 多头注意力 (Multi-Head Attention)

把 $d$ 维分成 $h$ 个头,每个头 $d/h$ 维:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

每个 head: $O(n^2 \cdot d/h)$。所有 head 并行: $h \times O(n^2 \cdot d/h) = O(n^2 d)$。

**总 FLOPs 不变**!但参数增加 $h$ 倍。**多头是"宽度换深度"**:并行多个子空间,每个学不同特征,这是 Transformer 比 RNN 强的关键之一。

### 4.4 推荐系统的双塔模型:复杂度对比

#### 单塔 (Cross Network)

用户特征 $u$ 和物品特征 $i$ **进入同一个网络**做交互。

- 在线复杂度:对每个用户,算 $|I|$ 次完整前向。
- 总 FLOPs: $|U| \times |I| \times \text{模型推理 FLOPs}$。
- **$|U| = |I| = 10^6$ 时,$10^{12}$ 次前向 → 完全不可行**。

#### 双塔 (Two-Tower)

用户和物品**独立编码**,只在内积时交互。离线算好物品向量,在线用户向量算 1 次 + 向量库 ANN 检索(毫秒级)。

#### 数字对比

设 $|U| = |I| = 10^6$,模型推理 100M FLOPs。

| 方案 | 在线 FLOPs | 耗时(单 A100) |
| :--- | :--- | :--- |
| 单塔 | $10^{12}$ | **数小时** |
| 双塔 + 全打分 | $10^{12}$ | 数小时 |
| 双塔 + ANN(IVF-PQ) | $10^8$ | **几十毫秒** |

**加速 10⁴ 倍以上**。

**代价**:双塔**丢失"用户-物品交叉特征"**。所以工业推荐"召回粗排用双塔,精排用单塔"。

### 4.5 GNN 的感受野:指数级扩张

设图 $G = (V, E)$,节点 $v$ 在第 $k$ 层聚合邻居:

$$\mathbf{h}_v^{(k)} = \text{UPDATE}\left(\mathbf{h}_v^{(k-1)}, \text{AGG}_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}\right)$$

跑 $k$ 层后,节点 $v$ 的感受野 = $k$ 跳邻居:

$$\text{感受野大小} = O(d^k), \quad d = \text{平均度数}$$

| $d$ | $k=3$ | $k=6$ |
| :--- | :--- | :--- |
| 10 | 1000 | 10⁶(接近全图) |
| 50 | 125000 | $1.5 \times 10^{10}$ |

**结论**:**GNN 不会太深**(通常 2-4 层),否则感受野饱和 = 节点信息相同 = over-smoothing。

#### GraphSAGE 的邻居采样

为限制计算,**每层随机采样固定数量**邻居(如 10-25 个),计算量 $O(k \times \text{采样数} \times d)$ → **常数复杂度**,与图大小无关。

### 4.6 训练 vs 推理的复杂度差异

#### 训练 vs 推理

| 维度 | 训练 | 推理 |
| :--- | :--- | :--- |
| 前向 FLOPs | FLOPs_F | FLOPs_F |
| 反向 FLOPs | ≈ 2 × FLOPs_F | 无 |
| 优化器状态(Adam) | 12 bytes/参数 | 无 |
| 激活值(显存大头) | 全部中间量 | 无 |
| KV-Cache | 无 | 与序列长度线性增长 |

**总训练成本 ≈ 3 倍前向 FLOPs + 优化器 + 激活值**,**总推理成本 ≈ 1 倍前向 + KV-Cache**。

#### KV-Cache:推理时的 O(n) 加速

**自回归生成**:已生 $t$ 个 token,生成第 $t+1$ 个。朴素每次重算全部 $t+1$ 个 → 累计 $O(n^3)$。**KV-Cache** 缓存历史 $K, V$,新 token 只算与历史的 attention → 单步 $O(t)$,总 $O(n^2)$。**加速 n 倍**。

**代价**:显存随序列长度线性增长,长序列推理需要 **PagedAttention (vLLM)** 等技术。

### 4.7 训练 LLaMA 7B 的 FLOPs 估算

LLaMA 7B = 6.7B 参数, $P = 7 \times 10^9$, $T = 1.4 \times 10^{12}$ tokens。

每 token 前向 FLOPs ≈ **2P**(乘加各算一次)。

**总训练 FLOPs** ≈ $2P \times T \times 3 \approx 6 \times 10^{22}$(前向 + 2 倍反向)。

A100:312 TFLOPS。1024 张 A100 理论峰值 $3.2 \times 10^{17}$ FLOPS,**MFU** 50%(实际利用率)。

**训练时间** = $6 \times 10^{22} / (3.2 \times 10^{17} \times 0.5) \approx 3.75 \times 10^5$ 秒 ≈ **4.3 天**。

实际 LLaMA 论文: 82,432 GPU-hours ≈ 1024 张 A100 × 3.4 天,和估算吻合。

### 4.8 推理成本:FLOPs vs 显存 vs 带宽

| 瓶颈 | 何时主导 | 例子 |
| :--- | :--- | :--- |
| **计算 (FLOPs)** | GPU 算力不够 | 大模型小 batch |
| **显存 (memory)** | 模型 / 激活太大 | 长序列训练 |
| **带宽 (bandwidth)** | 数据搬运比计算多 | 小模型大 batch |

**LLaMA 7B 推理**(单 token,batch = 1):
- FLOPs: 14 G, 算力需求 0.045 ms(算力富余)
- 显存: 14 GB(fp16)+ KV-Cache
- **带宽主导**:14 GB 从 HBM 搬到 SM,**带宽 2 TB/s → 7 ms**

**结论:小 batch 推理是 memory-bandwidth bound,不是 compute bound**。

### 4.9 Chinchilla 法则:参数 vs 数据等比例增长

给定固定计算预算 $C$,最优的参数 $N$ 和数据 $D$ 满足:

$$N \propto C^{0.5}, \quad D \propto C^{0.5}$$

LLaMA 论文严格遵循这条,用 1.4T tokens 训 7B,效果超过用 300B 训 175B。**不要光堆参数,数据也要跟上**。

---

## 5. 应对 NP-hard

### 5.1 五大策略

| 策略 | 思想 | 适用 |
| :--- | :--- | :--- |
| **近似算法** | 多项式时间 + 误差保证 | 有结构的问题 |
| **启发式** | 经验法则,无理论保证 | 工程上够用 |
| **参数化复杂度** | 固定某个参数,降复杂度 | 参数小 |
| **随机算法** | 期望复杂度好 | 概率可接受 |
| **采样 / 降维** | 减少输入规模 | 大数据 |

### 5.2 近似算法:TSP 的 Christofides 算法

**问题**:度量 TSP(距离满足三角不等式)。

**Christofides 算法**(1976):
1. 找最小生成树 (MST): $O(|E| \log |V|)$
2. 找 MST 上的最小权完美匹配: $O(|V|^3)$
3. 合并 MST + 匹配,形成欧拉回路: $O(|V|)$
4. 抄近路得哈密顿回路: $O(|V|)$

**总复杂度**: $O(|V|^3)$,**多项式**。

**近似比**: $\le 1.5 \times$ 最优。

**意义**:虽然 TSP 是 NPC,但度量 TSP 有 $1.5$-近似!这是近似算法的经典胜利。

### 5.3 启发式:TSP 的 2-opt

**2-opt 局部搜索**:
1. 初始解:随机 / 最近邻
2. 反复:删 2 条边,反向重连,看是否更短
3. 直到没有改善

```python
def two_opt(route, dist):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                # 尝试反向 route[i:j+1]
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if total_dist(new_route, dist) < total_dist(route, dist):
                    route = new_route
                    improved = True
    return route
```

**复杂度**:每次迭代 $O(n^2)$,迭代次数经验上几百。

**精度**:1000 个城市的实例,误差 < 5%。

**理论保证**:无。但工程上比 Christofides 还准。

### 5.4 参数化复杂度:Fixed-Parameter Tractable (FPT)

#### 思想

有些问题虽然 $n$ 很大,但关键参数 $k$ 很小(如"找大小为 $k$ 的团")。

**FPT 算法**:复杂度形如 $f(k) \cdot n^c$(指数依赖 $k$,多项式依赖 $n$)。

#### 例子:Vertex Cover (大小为 k 的覆盖)

朴素:$O(2^k n)$ 枚举子集 + 检查。

**FPT 算法**: $O(1.2738^k + kn)$(Chen et al. 2010)。

**工程含义**:**如果 $k$ 小(如 < 20),FPT 算法实际跑得飞快**。

### 5.5 随机算法:Monte Carlo 和 Las Vegas

**Monte Carlo**:可能错(概率 $\epsilon$),但快。

例子:**Miller-Rabin 素性测试**,出错概率 < $4^{-k}$。

**Las Vegas**:一定对,但运行时间是随机的。

例子:**随机快速排序**,期望 $O(n \log n)$。

### 5.6 推荐召回的 ANN:用近似换速度

#### 精确最近邻

暴力:对每个 query,算与所有 $|I|$ 个物品的距离 → $O(|I| \cdot d)$。

$|I| = 10^6, d = 64$:每次查询 6.4 亿次乘加 → **几秒**。

#### 近似最近邻 (ANN)

**HNSW**(Hierarchical Navigable Small World):
- 构建图,每个物品连接到"导航"邻居
- 查询:从顶层贪心下降,毫秒级
- **召回率 ≥ 95%**(略低于精确 100%),但**快 1000 倍**

**IVF-PQ**(Inverted File + Product Quantization):
- K-means 把物品分成 $k$ 个簇
- 查询:只搜最近的几个簇
- PQ 把向量量化到 8-bit,内存缩 32 倍

| 方法 | 查询时间 | 召回率 |
| :--- | :--- | :--- |
| 暴力 | 几秒 | 100% |
| HNSW | 1 ms | 99% |
| IVF-PQ | 0.5 ms | 95% |

**ANN 是工程妥协的典范**:放弃 1-5% 的精度,换 1000 倍的速度。

### 5.7 一个工程决策树

```text
问题输入规模?
├─ n < 50: 暴力 / 动态规划
├─ n < 1000: 启发式 / 局部搜索
├─ n < 10⁶: 近似算法 / ANN
└─ n > 10⁶: 降维 / 采样 / 分布式

是否需要 100% 准确?
├─ 是:动态规划 / FPT / 整数规划
└─ 否:启发式 / 随机 / 近似
```

---

## 6. PyTorch 代码

### 6.1 实验 1:复杂度对比(list vs set)

```python
import time

def time_op(fn, n=100):
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1e6  # 微秒

for N in [100, 10_000, 1_000_000]:
    data_list = list(range(N))
    data_set = set(range(N))
    target = N - 1
    t_list = time_op(lambda: target in data_list)
    t_set = time_op(lambda: target in data_set)
    print(f"N={N}: list={t_list:.2f}μs, set={t_set:.2f}μs, ratio={t_list/t_set:.0f}x")

# 预期:ratio 从 1.7x(N=100) → 333x(N=10000) → 25000x(N=1M)
```

### 6.2 实验 2:Attention 计时

```python
import time
import torch
import torch.nn.functional as F

def attention_time(n, d, device='cpu'):
    Q = torch.randn(n, d, device=device)
    K = torch.randn(n, d, device=device)
    V = torch.randn(n, d, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        S = Q @ K.T / (d ** 0.5)  # O(n²d)
        out = F.softmax(S, dim=-1) @ V
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / 10 * 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for n in [128, 512, 2048, 4096]:
    print(f"n={n:>5}: {attention_time(n, 64, device):>8.2f}ms")
# 预期:time 随 n 平方增长,n=4096 是 n=128 的 ~1000 倍
```

### 6.3 实验 3:贪心 TSP

```python
import random

def greedy_tsp(dist):
    """贪心 TSP:每次去最近的未访问城市"""
    n = len(dist)
    visited = [False] * n
    route = [0]
    visited[0] = True
    total = 0
    for _ in range(n - 1):
        curr = route[-1]
        nxt = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: dist[curr][j]
        )
        route.append(nxt)
        visited[nxt] = True
        total += dist[curr][nxt]
    total += dist[route[-1]][route[0]]  # 回到起点
    return route, total

random.seed(42)
n = 10
dist = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        dist[j][i] = dist[i][j]
route, total = greedy_tsp(dist)
print(f"路径: {route}, 总距离: {total}")
# O(n²),n=10 时 < 1ms,但 NPC,n=20+ 时需换近似算法
```

### 6.4 实验 4:子集和 (NPC 问题的小实例)

```python
from itertools import combinations

def subset_sum_decision(nums, target):
    """子集和:是否存在子集和为 target? 暴力 O(2^n)"""
    n = len(nums)
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            if sum(nums[i] for i in combo) == target:
                return True, [nums[i] for i in combo]
    return False, None

print(subset_sum_decision([3, 7, 1, 8, 4], 11))  # (True, [7, 4])
print(subset_sum_decision([3, 7, 1, 8, 4], 20))  # (False, None)
# n=20: 2^20 ≈ 100 万,n=30: 2^30 ≈ 10 亿,必须用启发式
```

**注释**:"所有子集" = 幂集 = $2^n$ 个,验证是 NP(读证书 O(n)),**找解是 NPC**。

### 6.5 实验 5:ANN vs 全量检索

```python
import numpy as np
import time

np.random.seed(42)
N, d = 100_000, 64
item_db = np.random.randn(N, d).astype('float32')
item_db /= np.linalg.norm(item_db, axis=1, keepdims=True)
query = np.random.randn(d).astype('float32')
query /= np.linalg.norm(query)

# 1. 全量检索
t0 = time.time()
top10 = np.argsort(-(item_db @ query))[:10]
t_exact = time.time() - t0

# 2. 采样 ANN:随机抽 1000 个
t0 = time.time()
sample_idx = np.random.choice(N, 1000, replace=False)
top10_ann = sample_idx[np.argsort(-item_db[sample_idx] @ query)[:10]]
t_ann = time.time() - t0

print(f"全量: {t_exact*1000:.2f}ms | ANN: {t_ann*1000:.2f}ms | 加速: {t_exact/t_ann:.0f}x")
# 实际工程用 FAISS(pip install faiss-cpu)做 HNSW/IVF-PQ
```

---

## 7. 常见误区

### 误区 1: ❌ "大 O 就是运行时间"

**反例**:大 O 描述的是**输入规模增长时的趋势**,不是绝对时间。

```python
# 两个都是 O(n),但常数差 1000 倍
# A:简单循环
def algo_A(arr):
    s = 0
    for x in arr:
        s += x
    return s

# B:每步调用 Python 函数 + list append
def algo_B(arr):
    result = []
    for x in arr:
        result.append(x ** 2)
    return sum(result)
```

**B 比 A 慢 100 倍,但都是 O(n)**。

**判断**:**大 O 决定"能不能 scale",常数决定"快不快"**。生产环境两个都要看。

### 误区 2: ❌ "P = NP,所以加密不安全"

**反例**:

1. **P = NP 未证明**(2002 年悬而未决的千禧问题)
2. **就算 P = NP**,多项式算法可能指数级慢(如 $O(n^{1000})$)
3. **加密的安全性依赖"单向函数"**,即使 P = NP,某些函数可能仍难求逆

**判断**:工程上**假设 P ≠ NP**(目前所有密码学都基于这个假设),但不要用它当"数学定律"宣传。

### 误区 3: ❌ "推荐召回必须精确 100%"

**反例**:

100 万物品的推荐池,**精确 Top-10** 要算 100 万次内积,几秒。

**ANN(近似最近邻)**:HNSW 图、IVF-PQ,**召回率 95-99%,几十毫秒**。

**业务现实**:用户等不了几秒,损失 1-5% 召回率换来 1000 倍速度,**业务收益远大于损失**。

**判断**:大多数工业召回都是 ANN,因为"快"比"准"重要 100 倍。

### 误区 4: ❌ "训练 LLM 主要靠堆 GPU"

**反例**:

LLaMA 论文(LLaMA 1, 65B):训练 82,432 GPU-hours。**但只用了 2048 张 A100 训了 21 天**,论文里说关键是**数据质量 + 模型架构**,不是单纯堆 GPU。

训练成本 = FLOPs,而 FLOPs ∝ **参数 × 数据 × 3**。三个都要优化:

| 优化 | 例子 | 加速 |
| :--- | :--- | :--- |
| 算法优化 | Flash Attention | 2-4× |
| 架构优化 | GQA(Mistral), MoE(Mixtral) | 5-10× |
| 数据优化 | 1.4T tokens 高质量数据 | 同算力更好模型 |
| 硬件优化 | H100 比 A100 快 3× | 3× |
| 精度优化 | fp16, bf16, int8 | 2-4× |

**判断**:**算力是必要条件,不是充分条件**。同样 FLOPs,优化好的训练能训出明显更强的模型。

---

## 8. 面试 Q&A (L7/L8 难度)

### Q1: Transformer attention 真的是 O(n²) 吗?工程上怎么优化?

**标准答案**:

**朴素 attention 确实是 O(n²d) FLOPs + O(n²) 显存**。这是因为:
- $Q K^T$ 生成 $n \times n$ 矩阵
- softmax + 与 $V$ 相乘,都需要访问整个 $n \times n$ 中间量

**工程优化三板斧**:

1. **Flash Attention**(Dao et al. 2022):
   - 思路:不存整个注意力矩阵,分块算
   - 显存:$O(n²) \to O(n)$
   - FLOPs:不变(数学等价)
   - 实测加速:2-4× wall-clock(因为减少了 HBM 访问)

2. **稀疏 Attention**:
   - **Sliding Window**(Mistral):每个 token 只看周围 $w$ 个,O(nwd),$w$ 固定 → O(n)
   - **Global + Local**(BigBird):少数全局 token + 局部窗口
   - **Longformer**:不同层用不同窗口

3. **Multi-Query / Grouped-Query Attention**(MQA/GQA,Mistral / LLaMA-2):
   - 多头共享 K、V 投影
   - 显存:KV-Cache 从 $2nd \to 2nd/h$ (h = 头数)
   - FLOPs:略降

**面试加分点**:
- 提到 Flash Attention 用 **tiling + online softmax**(Milakov & Gimelshein 2018)
- 提到 **PagedAttention**(vLLM):把 KV-Cache 分页,减少推理时的显存碎片
- 提到 **FlashAttention-3**(2024):用 Hopper 架构的 WGMMA 指令,达到 740 TFLOPS(A100 的 2.4×)

### Q2: P vs NP 是什么意思?在推荐系统里怎么体现?

**标准答案**:

**P**(Polynomial time):能在多项式时间内**求解**的判定问题集合。
**NP**(Nondeterministic Polynomial time):能在多项式时间内**验证**答案的判定问题集合。

显然 $P \subseteq NP$。**核心未解问题**:$P = NP$ 还是 $P \ne NP$?这是千禧年七大数学问题之一。

**工程意义**:**P = "好算法",NP = "答案好验证"**。

**推荐系统的体现**:

1. **双塔召回**:
   - **精确 Top-K**:暴力对每个 query 算与所有物品的距离 → $O(|I| \cdot d)$,**多项式** → **P 类**
   - **K-近邻近似**(ANN):牺牲 1-5% 精度换 1000 倍速度,实际是**常数因子优化**,还是 P

2. **组合推荐**(挑 $k$ 个物品当一个卡片):
   - "挑出 $k$ 个使得总点击率最大" = 组合优化
   - 朴素 $O(\binom{|I|}{k})$ 枚举,**指数** → **NPC**(子集和的变种)
   - 实际:**贪心 / 启发式 / 强化学习**

3. **神经网络架构搜索 (NAS)**:
   - "找最优架构"= 离散组合优化
   - 搜索空间指数级 → **NP-hard**
   - 实际:**进化算法 / 强化学习 / 可微搜索 (DARTS)**

**面试加分点**:
- 提到 **NPC 的工程应对**:近似算法、启发式、降维
- 提到 **P vs NP 不影响日常工程**,但**影响"是否值得找精确算法"**
- 提到即使 $P = NP$,多项式常数也可能让算法不可用

### Q3: 推荐召回为什么用双塔?复杂度差多少?

**标准答案**:

**单塔**(cross network):
- 用户特征和物品特征**进入同一个网络**,做深度交叉
- 在线复杂度:对每个用户-物品对,算一次完整前向 → $O(|U| \cdot |I| \cdot \text{模型成本})$
- 100 万用户 × 100 万物品 × 100M FLOPs = $10^{14}$ FLOPs → **完全不可行**

**双塔**(two-tower):
- 用户特征进入用户塔,物品特征进入物品塔,**离线算好物品向量**
- 在线:用户塔算 1 次 + 向量库 ANN 检索 → $O(|U| \cdot \text{用户塔} + \text{检索})$
- 检索:HNSW 1 ms,IVF-PQ 0.5 ms

**复杂度对比**:

| 方案 | 在线 FLOPs | 延迟 | 召回率 |
| :--- | :--- | :--- | :--- |
| 单塔 + 全打分 | $10^{14}$ | 数小时 | 100% |
| 双塔 + 全打分 | $10^{12}$ | 几十秒 | 100% |
| 双塔 + ANN | $10^8$ | **几十 ms** | 95-99% |

**加速 10⁴ 倍以上**。

**代价**:双塔丢失用户-物品**交叉特征**。如果"用户性别 × 物品类目"等交叉信号重要,双塔表达不了。

**工业实践**:**召回 + 粗排用双塔**(快),**精排用单塔**(准)。

**面试加分点**:
- 提到**双塔内积等价于打分函数分解**:$f(u, i) = \langle g(u), h(i) \rangle$
- 提到**向量库**(FAISS / ScaNN / Milvus)的 ANN 算法:HNSW、IVF-PQ
- 提到**双塔 + 交叉精排**的工业架构:**双塔召回 1000 候选 → 单塔精排 100 候选**

### Q4: 深度学习是 NP-hard 问题吗?训练复杂度怎么算?

**标准答案**:

**两个层次,答案不同**:

1. **神经网络架构搜索 (NAS)**:
   - 搜索空间 = 离散架构组合(层数、宽度、激活函数、连接方式)
   - 搜索空间大小指数级 → **NP-hard**
   - 实际:进化算法、强化学习、可微搜索 (DARTS)

2. **给定架构的训练**:
   - 损失函数 $L(\theta)$ 在 $\mathbb{R}^P$ 空间(连续)
   - 用 SGD / Adam 做连续优化
   - 复杂度:**每步 $O(P \cdot \text{batch} \cdot \text{序列长度}²)$ FLOPs**

**LLaMA 7B 训练复杂度估算**:

```text
P = 7×10⁹ 参数
T = 1.4×10¹² tokens
每 token 前向 FLOPs ≈ 2P = 1.4×10¹⁰
训练 FLOPs = 2P × T × 3 (前+反+优) ≈ 6×10²²
A100 算力 = 312 TFLOPS
1024 张 A100 × MFU 50% ≈ 1.6×10¹⁷ FLOPS 实际
训练时间 ≈ 6×10²² / 1.6×10¹⁷ ≈ 3.75×10⁵ 秒 ≈ 4.3 天
```

实际 LLaMA 论文:82,432 GPU-hours ≈ 1024 张 A100 × 3.4 天。和估算吻合。

**结论**:
- **架构搜索**:NPC/NPH,**没有高效精确算法**,用启发式
- **给定架构训练**:连续优化,**每步多项式**,但**总成本巨大**(需要几百到几千 GPU-days)

**面试加分点**:
- 提到 **Chinchilla 法则**:$N \propto D \propto C^{0.5}$,参数和数据等比例增长
- 提到 **MFU**(Model FLOPs Utilization):实际利用率 40-60%
- 提到 **Transformer 训练 FLOPs ≈ 6PDT**(P = 参数,D = 数据,T = token 数)

### Q5: 怎么判断一个问题是不是 NP-complete?

**标准答案**:

**三步判别法**:

1. **是判定问题吗?** 答案必须是"是 / 否"。如果是优化,先转成判定(问"最优值 ≤ k?")。

2. **证书多项式可验证吗?** 给定一个"答案"(证书),验证它正确性是多项式时间。这是 NP 的定义。

3. **能归约到已知的 NPC 问题吗?** 找一个已知的 NPC 问题 $L'$,把 $L'$ 归约到你这个问题。

**已知 NPC 问题清单**(用作"起点"):

| 问题 | 起点难度 |
| :--- | :--- |
| SAT / 3-SAT | ⭐ 起点(用 Cook-Levin) |
| 团 / 独立集 / 顶点覆盖 | ⭐⭐ 经典 |
| 子集和 / 0-1 背包 | ⭐⭐ |
| TSP / 哈密顿回路 | ⭐⭐ |
| 图着色 | ⭐⭐ |

**工程判断**(不严格):

| 现象 | 提示 |
| :--- | :--- |
| 输入是图的子集 / 路径 / 子序列 | 大概率 NPC |
| 输入是整数列表,问子集 / 序列 | 大概率 NPC |
| 输入是"找最优组合" | 优化版 = NPH,判定版通常 NPC |
| 有指数级枚举可解 | 提示是 NPC |

**反例**(看似难但其实 P):
- 最短路(Dijkstra):$O((V+E) \log V)$
- 线性规划(单纯形 / 内点):$O(n^{2.5})$ 多项式
- 欧拉回路:$O(V+E)$
- 最小生成树(Kruskal / Prim):$O(E \log E)$

**面试加分点**:
- 提到 **Karp 21 个 NPC 问题**(1972):最经典的 NPC 集合
- 提到 **Strong NP-hard** vs **Weak NP-hard**:子集和是 weak(Pseudo-polynomial),团是 strong

---

## 9. 与其他数学工具的联系 + 参考文献

### 9.1 与其他章节的联系

算法与复杂度是离散数学的"计算视角",向上连接 ML/DL 的所有工程决策:

- **→ 第 1 章 集合 / 函数**:推荐召回 = 在 $U \times I$ 上定义函数,双塔 = 把 $f(u, i)$ 分解成 $g(u) \cdot h(i)$,**ANN 检索 = 高维空间找近似函数反演**。详见 第 1 章:集合·关系·函数-从用户行为序列到数学集合。

- **→ 第 3 章 组合数学**:子集和、0-1 背包、背包问题都是 NPC,组合数学的工具(生成函数、容斥)能分析 NPC 的渐近行为。详见 第 3 章:组合数学-排列组合·生成函数·容斥原理。

- **→ 第 4 章 图论 / 第 5 章 图进阶**:哈密顿回路、顶点覆盖、图着色都是 NPC,图算法(Dijkstra、Prim)是多项式算法,**对比两者的复杂度差异是理解 P/NP 的最快路径**。详见 `第04章-04-图论基础-顶点·边·路径·连通性·树·二分图.md` 和 第 5 章:图进阶-树·生成·匹配·流。

- **→ 2-线代 第 4 章 矩阵**:矩阵乘法 $O(n^3) \to \text{Strassen} \to \text{Coppersmith-Winograd}$,Transformer attention 的核心就是矩阵乘法的应用。详见 `../../../基础/数学基础/第02章-线性代数/第02章-01-向量·内积·范数-Embedding空间的几何直觉.md`。

- **→ 7-系统思维 第 14 章 Transformer attention**:Transformer 的 O(n²d) FLOPs + O(n²) 显存是本章核心,Flash Attention、稀疏 Attention、KV-Cache 都是工程优化。详见 `../第07章-系统思维/第07章-14-案例Transformer注意力信息路由.md`。

### 9.2 P vs NP 的工程意义

**虽然 P vs NP 未解,但工程师要会"用它"**:

| 场景 | 判断 | 行动 |
| :--- | :--- | :--- |
| 问题规模 $n$ 小(< 1000) | NPC 也无所谓 | 直接枚举 / DP |
| 问题规模 $n$ 中(1000 - 10⁶) | NPC / P 都要小心 | 启发式 + 近似 |
| 问题规模 $n$ 大(> 10⁶) | 即使 P 也可能慢 | 必须预处理 |
| "找最优架构"(NAS) | NPH | DARTS / 进化 |
| "训练给定架构" | 连续优化 | SGD / Adam |
| "在线推理" | 复杂度决定延迟 | KV-Cache / 量化 |
| "召回 1000 万物品" | P 但常数大 | ANN |

### 9.3 参考文献

1. **Cormen, T. H., et al. (2009). 《Introduction to Algorithms》(CLRS)**: §2 大 O、§3 主定理、§4 分治、§34 NP。
2. **Sipser, M. (2012). 《Introduction to the Theory of Computation》**: §7 P / NP / NPC、§8 NP-completeness。
3. **Cook, S. A. (1971). "The complexity of theorem-proving procedures"**: Cook-Levin 定理,SAT 是 NPC 的原始证明。
4. **Karp, R. M. (1972). "Reducibility among combinatorial problems"**: Karp 21 个 NPC 问题。
5. **Vaswani, A., et al. (2017). "Attention Is All You Need"**: Transformer,§4 复杂度分析。
6. **Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"**: Flash Attention 把显存从 O(n²) 降到 O(n)。
7. **Christofides, N. (1976). "Worst-case analysis of a new heuristic for TSP"**: TSP 1.5-近似算法。
8. **Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models"**: Chinchilla 法则。
9. **Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"**: LLaMA 7B/13B/65B 训练成本分析。

### 9.4 一句话总结

> **复杂度是 ML/DL 工程师的"算力账本":每个设计选择都要算一笔"FLOPs / 显存 / 延迟"账。P / NP 告诉你"该不该找精确算法",Transformer / GNN 的复杂度告诉你"模型能不能 scale"。**

### 9.5 下一章预告

第 11 章:形式语言与自动机-正则·上下文无关·CFG·CYK,LLM 受限解码、自动定理证明、程序分析的形式化基础。

本章讲的"判定问题"会变成"语言判定","图灵机"会变成"自动机","P vs NP"会变成"正则 / 上下文无关 / 上下文有关 / 递归可枚举"的乔姆斯基谱系。理解了算法复杂度,形式语言就是顺水推舟,LLM 的"结构化输出"本质上就是在自动机上做约束解码。

---

> **本章版本**: v1.0
> **最后更新**: 2026-06-25
> **作者**: Sisyphus (OhMyOpenCode)
> **配套代码**: §6 所有代码均可直接运行 (PyTorch 2.x + Python 3.10+)

## 附录:补充内容

> 本节作为第 10 章的扩展,围绕"P/NP/NP-hard 与深度学习的工程联系"展开,补充归约证明、注意力机制的更细复杂度、量化蒸馏剪枝的成本账、分布式训练策略、模拟退火 / NAS / Mamba 等"非精确算法"在深度学习里的角色,以及更多 Python 实验。

---

### A. 复杂度类之间的精细关系

#### A.1 P, NP, co-NP, PSPACE, EXP

很多人以为"非 P = NP",但严格地讲复杂度类有完整的"谱系":

| 复杂度类 | 定义 | 典型问题 | 与 P 的关系 |
| :--- | :--- | :--- | :--- |
| **P** | 确定性多项式时间可解 | 最短路, 二分查找, 线性规划 | = P |
| **NP** | 多项式时间可验证 | SAT, 子集和, TSP(判定) | $\supseteq$ P |
| **co-NP** | "否"答案多项式可验证(等价于"否"证书可验证) | 公式永假 (TAUTOLOGY), 质数判定 | 与 NP 关系未解 |
| **NP ∩ co-NP** | 既是 NP 又是 co-NP | 质数判定 (AKS, 2002), 矩阵行列式 = 0? | ⊇ P |
| **PSPACE** | 多项式空间可解 | QBF (量化布尔公式) | ⊇ NP |
| **EXP** | 指数时间可解 | 一般博弈, 带量词的 Presburger 算术 | ⊇ PSPACE |

**几个被证明的关系**:

1. **P ⊆ NP ⊆ PSPACE ⊆ EXP**(层次嵌套)
2. **P ≠ EXP**(时间层次定理, 1974)。反证:如果 P = EXP, 任意指数时间算法都能多项式化, 但有具体的指数下界问题证明不可能。
3. **P = NP ⇒ NP = co-NP**(若 P = NP, 多项式可解 = 可验证, "否"也能用补证书验证)
4. **NP ≠ co-NP 至今未证**, 但被广泛相信(因为它直接蕴含 P ≠ NP)

#### A.2 co-NP 与"否"证书

**例**: 给定一个布尔公式, 它是否**永假** (tautology)?

- 这是 co-NP 问题: 要证明"是", 容易(展示一个变量赋值使其为真, 错; 需展示**所有**赋值使其为真, 难)
- 等价地, "$\phi$ 是永假" $\iff$ "$\neg \phi$ 是不可满足"
- 如果某个问题是 co-NP 完全 (co-NPC), 暗示它和 SAT 一样难, 但**否答案好验证**

**深度学习的例子**: 神经网络验证 (neural network verification): "给定网络 $f$ 和输入域 $X$, 是否**所有** $x \in X$ 都满足 $f(x) > 0$?"

这是 co-NP 完全问题, 因为验证"否"很容易(给一个反例 $x$), 但验证"是"要穷举所有 $x$。Reluplex, Marabou 等验证器用 SMT 求解 + 松弛逼近。

#### A.3 因子分解与 RSA

**问题**: 给定 $n = p \cdot q$($p, q$ 是大质数), 分解出 $p, q$。

- **现状**: 已知最快的经典算法是**一般数域筛 (GNFS)**, 亚指数复杂度 $L_n[1/3, c] = \exp(c \cdot (\log n)^{1/3} (\log \log n)^{2/3})$
- **不是已知的 P 问题**, 但也**不是已知的 NP 完全问题**
- 量子算法 (Shor, 1994) 是 $O((\log n)^3)$, **多项式**! 所以大规模量子计算机会攻破 RSA

**工程含义**: 密码学现在押注在"经典计算机分解很难" + "量子计算机还没成熟"。后量子密码 (PQC) 正在迁移到格密码 (LWE), 那才是已知的 NP-hard。

#### A.4 图同构 (Graph Isomorphism) 的特殊地位

**问题**: 给定两个图 $G, H$, 是否同构 (即存在双射 $\phi: V_G \to V_H$ 使得边对应)?

- 显然在 NP(给一个双射 $\phi$, 验证 $O(|V|^2)$)
- 2015 年 Babai 给出拟多项式算法 $O(\exp(c \sqrt{\log |V|}))$, **不是 NPC**
- 是否在 P, **至今未解**
- 实际工程: 几乎所有同构测试工具 (nauty, bliss) 都是 $O(|V| \log |V|)$ 实用

**意义**: 图同构可能是"接近 P 但不在 P"的少数自然问题, 复杂度理论家关注它。

---

### B. NPC 归约的完整示例

#### B.1 3-SAT → 独立集 (经典归约)

**定理**: 独立集是 NPC。

**归约**: 给定 3-CNF 公式 $\phi = C_1 \wedge C_2 \wedge \cdots \wedge C_m$, 每个子句 $C_i = (l_{i,1} \vee l_{i,2} \vee l_{i,3})$, 构造图 $G$:

- 每个 (子句, 文字) 对应一个顶点, 共 $3m$ 个顶点
- 在以下顶点之间连边:
  - 同一子句的不同文字之间(强制一个子句最多选一个)
  - 互斥的文字之间(一个变量和它的否定, 不能同时选)

```text
子句 1: (x1 ∨ ¬x2 ∨ x3)   顶点: (1, x1), (1, ¬x2), (1, x3)
子句 2: (¬x1 ∨ x2 ∨ ¬x3)  顶点: (2, ¬x1), (2, x2), (2, ¬x3)

边:
- 子句内: (1, x1)-(1, ¬x2), (1, x1)-(1, x3), (1, ¬x2)-(1, x3)
- 子句内: (2, ¬x1)-(2, x2), (2, ¬x1)-(2, ¬x3), (2, x2)-(2, ¬x3)
- 互斥: (1, x1)-(2, ¬x1), (1, x3)-(2, ¬x3), (1, ¬x2)-(2, x2)
```

**关键**: $G$ 有大小为 $m$ 的独立集 $\iff$ $\phi$ 可满足。

**直觉**: 选一个独立集, 等价于"每个子句挑一个真文字, 且不同子句挑的文字不互斥"。

**归约时间**: $O(m)$ 个顶点和 $O(m)$ 条边。多项式。✓

#### B.2 3-SAT → 子集和 (Karp 21 之一)

**定理**: 子集和 (Subset Sum) 是 NPC。

**归约构造**: 给定 3-CNF $\phi$ 有 $n$ 个变量 $x_1, \ldots, x_n$ 和 $m$ 个子句, 构造整数集合 $A$:

对每个变量 $x_i$, 创造两个数 $v_i, \bar{v}_i$ (选 $x_i$ 或 $\neg x_i$), 用一个大的"基数"$B$ 隔开位:

$$
a_i = v_i \cdot B + \bar{v}_i, \quad b_i = \bar{v}_i \cdot B + v_i
$$

其中 $B = 10^{\max(\text{子句编码位数})}$。目标 $S = (\sum v_i) \cdot B + (\sum \bar{v}_i) + m'$(其中 $m'$ 编码子句覆盖)。

**关键**: 子集和的解对应一个赋值, 满足"每个子句至少一个真文字" $\iff$ 子集和达到 $S$。

**归约时间**: $O(n \cdot \text{位数})$。多项式。✓

#### B.3 Strong NP-hard vs Weak NP-hard

**伪多项式算法**: 复杂度形如 $O(\text{poly}(n, W))$, 其中 $W$ 是输入数值(不是二进制位数)。

- **子集和**: 有 $O(n \cdot W)$ 的动态规划算法(用 bitset 加速可达 $O(n \cdot W / 64)$)。这是**伪多项式**, 因为 $W$ 的二进制位数是 $\log W$, 所以 $W$ 指数大时爆掉。
- **团 (Clique)**: 强 NP-hard, 没有伪多项式算法(假设 P ≠ NP)。

**判断**:

| 类型 | 含义 | 例子 |
| :--- | :--- | :--- |
| **Strong NP-hard** | 即使数值多项式有界也是 NP-hard | 团, 哈密顿回路, TSP |
| **Weak NP-hard** | 数值指数大时才"难" | 子集和, 0-1 背包 |

**工程含义**: 弱 NP-hard 问题, 如果 $W$ 小(比如 $W < 10^6$), 伪多项式 DP 就能跑。背包问题 $n = 100, W = 10^4$, DP 完全可行。

---

### C. 深度学习中的"非精确"算法

深度学习里很多问题是 NP-hard 或 NPH, 工程师的做法不是"放弃", 而是"用非精确算法赌一个可接受解"。本节梳理主要套路。

#### C.1 神经网络架构搜索 (NAS) 的复杂度

**问题**: 给定候选操作集合 $\mathcal{O}$ (3×3 conv, 5×5 conv, attention, skip connection...), 选 $k$ 个操作 + 连接方式, 使验证集性能最大。

**复杂度**: 操作组合 $\binom{|\mathcal{O}|^k}{k} = O(|\mathcal{O}|^{k^2})$ 指数级。

**实际算法**:

| 方法 | 思路 | 复杂度 | 代表论文 |
| :--- | :--- | :--- | :--- |
| **强化学习** | 用 RNN 控制器采样架构, 验证性能作 reward | 采 10^4 架构 | NAS-RL (Zoph 2017) |
| **进化算法** | 突变 + 选择 | 采 10^3 架构 | AmoebaNet (Real 2019) |
| **贝叶斯优化** | 用高斯过程建模性能曲面 | 采 10^2 架构 | DARTS (Liu 2019) |
| **可微搜索 (DARTS)** | softmax 松弛离散选择 → 梯度下降 | **单次训练** | DARTS (Liu 2019) |
| **One-shot / 超网** | 训练一个大超网, 子网权重继承 | 单次训练 | ENAS, ProxylessNAS |

**DARTS 思路**: 原本"选 $o_1$ 或 $o_2$"是离散决策, 松弛为加权 $\alpha_1 o_1 + \alpha_2 o_2$, $\alpha$ 连续。训练时同时学 $\alpha$ 和权重, 最后取 $\arg\max \alpha$。

**复杂度账**: NAS 是 NPH, 但**工程上 DARTS 把单次成本压到几 GPU-days**。这就是"用启发式赌一个解"的胜利。

#### C.2 模拟退火 (Simulated Annealing) 在 DL 里的应用

**算法**: 模拟物理退火, 以概率 $\exp(-\Delta E / T)$ 接受劣解, 温度 $T$ 随时间下降。

**DL 应用**:

1. **随机权重平均 (SWA)**: 在 SGD 末尾对多个 checkpoint 取平均, 等价于"低温度下采样多个极小值"
2. **离散权重优化 (二值化 / 量化感知训练)**: 引入退火温度, 让离散决策变得"软"
3. **数据增强策略搜索 (AutoAugment)**: 强化学习 + 退火, 搜索增强策略空间

**Python 模拟 (TSP 退火)**:

```python
import random, math

def sa_tsp(dist, T0=100.0, alpha=0.995, iters=5000):
    n = len(dist)
    route = list(range(n))
    random.shuffle(route)

    def cost(r):
        return sum(dist[r[i]][r[(i+1) % n]] for i in range(n))

    best = route[:]
    best_cost = cur_cost = cost(route)
    T = T0

    for _ in range(iters):
        # 随机扰动: 2-opt 翻转
        i, j = sorted(random.sample(range(n), 2))
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        new_cost = cost(new_route)
        dE = new_cost - cur_cost
        if dE < 0 or random.random() < math.exp(-dE / T):
            route, cur_cost = new_route, new_cost
            if cur_cost < best_cost:
                best, best_cost = route[:], cur_cost
        T *= alpha
    return best, best_cost
```

**复杂度**: $O(\text{iters} \cdot n)$, 与精确算法 $O(n!)$ 不可同日而语。**n = 1000 时, SA 跑 1 秒, 误差 2-5%**。

#### C.3 强化学习 (RL) 作为"在线组合优化"

很多深度学习的"调度"问题是 NPH, RL 提供了一个"在线学策略"的方法:

| 问题 | 状态 | 动作 | 奖励 | 算法 |
| :--- | :--- | :--- | :--- | :--- |
| 神经网络剪枝 | 当前剪枝后模型 | 删一个通道 | 验证精度 - 成本 | PPO |
| 混合精度量化 | 当前层量化精度 | 升 / 降精度 | 延迟变化 | DDPG |
| 算子调度 | 计算图 | 算子放 GPU0/GPU1 | 总时间 | A2C |
| 推荐系统选品 | 用户历史 | 推荐 K 个物品 | 点击 | DQN |

**理论保证**: 几乎没有 (RL 没有复杂度上界)。**工程上**: 比手工调好 10-30%, 所以大家都在用。

#### C.4 蛋白质折叠与 AlphaFold

**问题**: 给定氨基酸序列, 预测 3D 结构。

**复杂度**: 能量最小化 + 立体化学约束, 是 NPH (即使简化模型)。20 年前 CASP 比赛, 最好的方法准确度 40%; 现在 AlphaFold 接近 90%。

**AlphaFold 的工程哲学**:
- 不是"解 NPH", 而是"训练一个能预测结构的网络"
- 用注意力机制 + 等变几何 + 进化保守性, 把组合优化"软化"为可微优化
- 训练数据 = PDB 已知结构 (200K+ 蛋白质)
- 推理 = 一次前向, 秒级

**意义**: 深度学习的"非精确但实用"哲学, 把不可能变成"够用"。

---

### D. 注意力机制的更细复杂度

#### D.1 朴素 vs Flash vs Linear Attention 复杂度对比

| 方法 | FLOPs | 显存 | 序列长度限制 |
| :--- | :--- | :--- | :--- |
| 朴素 attention | $O(n^2 d)$ | $O(n^2 + nd)$ | ~8K |
| **Flash Attention** | $O(n^2 d)$ (不变) | $O(n)$ | ~32K (Hopper 上更大) |
| **Linear Attention** (Performer) | $O(n d^2)$ | $O(nd)$ | 任意 |
| **Sparse** (Longformer) | $O(nwd)$ | $O(nw + nd)$ | 任意 |
| **State Space (Mamba)** | $O(nd^2 \log d)$ | $O(d^2)$ | 任意 |

**关键**: Flash Attention 不改 FLOPs, 但**减显存访问** (HBM IO 从 $O(n^2)$ 降到 $O(n)$), 所以 wall-clock 加速 2-4×。

**Linear Attention**: 用核函数 $\phi(Q) \phi(K)^T$ 代替 $Q K^T$, 利用结合律:

$$
\text{Attention}(Q, K, V) = \phi(Q) (\phi(K)^T V)
$$

先算 $\phi(K)^T V \in \mathbb{R}^{d \times d}$, 再左乘 $\phi(Q)$, FLOPs 从 $O(n^2 d)$ 降到 $O(nd^2)$。

**代价**: 核函数近似损失精度, **小模型 / 短序列上不如标准 attention**。

#### D.2 State Space Models (Mamba) 的复杂度

Mamba 用选择性状态空间, 数学上等价于**线性递推**:

$$
h_t = A h_{t-1} + B x_t, \quad y_t = C h_t
$$

每步 $O(d^2)$ FLOPs, 总 $O(nd^2)$。**没有注意力, 没有 KV-Cache 显式存储**。

**硬件友好**: 递推天然适配 GPU 串行执行 + 扫描算法 (parallel scan), 实际 wall-clock 接近线性。

**对比**:

| 维度 | Transformer | Mamba |
| :--- | :--- | :--- |
| 训练 FLOPs | $O(n^2 d)$ | $O(nd^2)$ |
| 训练显存 | $O(n^2)$ (Flash 后 $O(n)$) | $O(d^2)$ |
| 推理 FLOPs (单步) | $O(nd)$ (KV-Cache) | $O(d^2)$ |
| 状态大小 | $O(nd)$ (KV-Cache) | $O(d^2)$ |
| 长序列性能 | 优秀 (足够容量) | 优秀 (递推精确) |
| 检索 / 复制任务 | 强 | 弱 (Mamba 已知短板) |

**工程意义**: 没有"最好的架构", 只有"trade-off"。Transformer 强在 in-context 学习, Mamba 强在线性复杂度。

#### D.3 KV-Cache 的详细显存账

LLM 推理 (自回归生成) 时, 每个 token 都要算与历史所有 token 的 attention。

**朴素**: 重算历史 attention, $O(n)$ 步 × $O(n)$ 历史 = $O(n^2)$ 总 FLOPs。

**KV-Cache**: 缓存历史 $K, V$, 新 token 只算与历史的 attention:

- 第 $t$ 步: $K_t, V_t \in \mathbb{R}^{d}$, 累积缓存 $K_{1:t}, V_{1:t} \in \mathbb{R}^{t \times d}$
- 显存: 每步新增 $2d$ (K + V) 参数, 总 $O(nd)$
- 单步 FLOPs: $O(td)$ (与历史 $t$ 个 token 算 attention), 总 $O(n^2 d)$ (和朴素一样, 但常数小)

**显存数字** (LLaMA 7B, fp16, batch=1, seq=2048):

- 模型: 14 GB
- KV-Cache: 2 × 32 层 × 2048 序列 × 4 头 × 128 维 × 2 bytes = 128 MB
- 激活: ~ 几 GB

**PagedAttention (vLLM)**: 把 KV-Cache 分页 (类似 OS 虚拟内存), 减少碎片, 提升 batch 利用率 2-4×。

#### D.4 推理 Prefill vs Decode 阶段

LLM 推理分两阶段:

| 阶段 | 任务 | 复杂度 | 瓶颈 |
| :--- | :--- | :--- | :--- |
| **Prefill** | 处理整个 prompt (一次性) | $O(n^2 d)$ (像训练) | 计算 (compute bound) |
| **Decode** | 自回归逐 token 生成 | $O(nd)$ per step, 总 $O(n^2 d)$ | 带宽 (memory bound) |

**Prefill**: GPU 全力算矩阵乘, 利用率高 (MFU 70%+)。
**Decode**: 每次只算 1 个 token, GPU 利用率低 (MFU 5-10%)。这是 LLM 推理的"带宽墙"。

**Continuous batching**: vLLM 提出的"动态 batch", 把不同请求的 decode 阶段合并, 利用率从 5% 提到 50%+, 吞吐量提升 10-20×。

---

### E. 量化 / 蒸馏 / 剪枝的成本账

深度学习推理的三大优化: **量化 (quantization), 蒸馏 (distillation), 剪枝 (pruning)**。本质都是"用算法复杂度换精度"。

#### E.1 量化

**INT8 量化**: 权重从 fp16 降到 int8, 显存 / 带宽减半, FLOPs 不变 (但加法可以更快)。

**INT4 / 二值化 (1-bit)**: 极端情况, 内存减 4-16×, 但精度损失大。

**算法复杂度**:

| 精度 | 显存 (LLaMA 7B) | 带宽 | 相对延迟 (A100) |
| :--- | :--- | :--- | :--- |
| fp32 | 28 GB | 1× | 1× |
| fp16 | 14 GB | 1× | 0.5× |
| int8 | 7 GB | 0.5× | 0.25× |
| int4 | 3.5 GB | 0.25× | 0.15× |
| int2 | 1.75 GB | 0.125× | ~0.1× |
| int1 (二值) | 0.875 GB | 0.0625× | 0.05× |

**量化算法**:

- **PTQ (Post-Training Quantization)**: 训完直接量化, 几小时, 精度损失 1-3%
- **QAT (Quantization-Aware Training)**: 训练时模拟量化, 几天, 精度损失 < 0.5%

**PTQ 是 P 类(优化一个二次规划), QAT 是连续优化, 都"高效"**。但**选择哪些层量化, 哪些保持 fp16**, 是 NPC 的层数指数搜索。

#### E.2 蒸馏

**Knowledge Distillation**: 用大模型 (teacher) 教小模型 (student)。

**复杂度**: 训练 student 一次, FLOPs ≈ student 训练的 1× + teacher 推理的 3× (前+反+前)。总成本可控。

**工程账**:

- LLaMA 7B (teacher) → 小模型 (student)
- Student 用 0.5B 参数, 训练 100B tokens, FLOPs ≈ $2 \times 5 \times 10^8 \times 10^{11} = 10^{20}$
- 加上 teacher 推理 3 倍: $3 \times 10^{20}$ 总 FLOPs
- 几 GPU-days 就能训出接近 7B 90% 性能的小模型

**DistilBERT, TinyLlama** 都是这条路。

#### E.3 剪枝

**Structured pruning**: 删整个 channel / head / layer。
**Unstructured pruning**: 删单个权重 (得到稀疏矩阵)。

**算法**:

| 方法 | 思路 | 复杂度 |
| :--- | :--- | :--- |
| **Magnitude pruning** | 按 \|w\| 删最小 | 一次性, $O(\text{params})$ |
| **SNIP** | 训练前算"重要性" | $O(\text{训练 FLOPs})$ |
| **GraSP** | 用梯度信息 | 几次迭代, $O(10 \times \text{训练 FLOPs})$ |
| **Movement pruning** | 训练中动态剪 | 同训练, 1-2× 时间 |
| **Lottery Ticket** | 找稀疏子网 | **迭代训练, NP-hard** |

**Lottery Ticket 假设**: 存在一个稀疏子网 (1-10% 权重), 单独训练能达到原网络精度。找这个子网是组合优化, NPH。

**工程实践**: 大多数 production 模型用**结构化剪枝 + 蒸馏**联合, 5-10× 加速, 精度损失 < 2%。

#### E.4 三种优化的复杂度对比

| 维度 | 量化 | 蒸馏 | 剪枝 |
| :--- | :--- | :--- | :--- |
| 算法本质 | 数值映射 | 用大教小 | 删冗余 |
| 搜索空间 | 连续 + 离散 | 架构选择 | 权重子集 |
| 复杂度 | 二次规划 (P) | 训练 (P) | **NP-hard** (选哪些) |
| 加速 | 2-4× | 5-10× | 2-10× |
| 精度损失 | 0.5-3% | 1-5% | 0.5-5% |

---

### F. 大模型分布式训练

LLM 训练必须分布式, 因为单卡装不下。策略选择本质上就是"把模型 / 数据 / 流水线切片"。

#### F.1 四大并行策略

| 策略 | 切什么 | 通信 | 显存节省 | 适用 |
| :--- | :--- | :--- | :--- | :--- |
| **数据并行 (DP)** | 数据 batch | 梯度 all-reduce | N× (N 卡) | 模型 < 单卡显存 |
| **张量并行 (TP)** | 单层参数分片 | 前后向 all-reduce | N× | 单层 > 单卡显存 |
| **流水线并行 (PP)** | 层间分片 | 激活传递 | N× | 总模型 > 单卡 |
| **专家并行 (MoE EP)** | 专家分片 | gate 路由 | 稀疏激活节省 | MoE 架构 |

**复杂度账** (LLaMA 70B 训练):

- 模型: 70B × 2 bytes = 140 GB (单卡装不下)
- 优化器 (Adam): 12 bytes/参数 × 70B = 840 GB
- 激活: 与 batch × 序列长度成比例, 几十 GB
- 梯度: 70B × 2 bytes = 140 GB

**总显存需求**: 1 TB+。必须张量并行 + 流水线并行 + ZeRO (优化器分片)。

**A100 80GB × 1024 张**: 实际训练 LLaMA 70B, **理论峰值 $3.2 \times 10^{17}$ FLOPS × 50% MFU = $1.6 \times 10^{17}$ FLOPS**。

**训练时间**: $6 \times 70 \times 10^9 \times 1.4 \times 10^{12} / 1.6 \times 10^{17} \approx 3.7 \times 10^6$ 秒 ≈ 43 天。

#### F.2 通信瓶颈

分布式训练的"墙"是通信:

| 通信原语 | 典型大小 | 频率 |
| :--- | :--- | :--- |
| All-reduce (DP 梯度同步) | 梯度大小, 几十 GB | 每步 |
| All-reduce (TP 同步) | 隐藏状态, GB 级 | 每层前后向 |
| Send/Recv (PP 激活) | 激活, GB 级 | 每 micro-batch |

**NVLink 900 GB/s, InfiniBand 400 Gb/s = 50 GB/s**。通信主导时, GPU 算力浪费。

**3D 并行**: DP × TP × PP, 让通信分散, MFU 提到 50%+。

**DeepSpeed ZeRO-3**: 把优化器 / 梯度 / 参数都分片, 显存省 10-100×, 但通信量增 1.5×。

#### F.3 训练 FLOPs 公式

**通用公式**: 训练 LLM 的总 FLOPs ≈ $6 \times P \times D$, 其中 $P$ = 参数, $D$ = tokens。

- 前向: $2PD$
- 反向: $4PD$ (链式法则, 大约 2× 前向)
- 总: $6PD$

**例子**:

| 模型 | $P$ | $D$ | 总 FLOPs | A100×1024×50% MFU | 天数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-2 | 1.5B | 50B | $4.5 \times 10^{20}$ | 1024 太少, 用 256 卡 | ~ 1 天 |
| LLaMA 7B | 7B | 1.4T | $6 \times 10^{22}$ | 1024 | 4.3 天 |
| LLaMA 70B | 70B | 1.4T | $6 \times 10^{23}$ | 1024 | 43 天 |
| GPT-3 175B | 175B | 300B | $3.15 \times 10^{23}$ | 1024 | ~ 22 天 |
| GPT-4 (估计) | 1.8T (MoE) | 13T | $1.4 \times 10^{26}$ | 25000+ | 几个月 |

**Chinchilla 修正**: 给定 $C = 6PD$, 最优 $P \propto C^{0.5}$, $D \propto C^{0.5}$。即 $P$ 和 $D$ 同比例增长, 不要光堆参数。

---

### G. Python 实验: 复杂度测量与算法对比

#### G.1 实验 1: list vs dict 性能, $N$ 扫描

```python
import time

def time_op(fn, n=100):
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1e6  # 微秒

print(f"{'N':>10} | {'list':>10} | {'set':>10} | {'ratio':>8}")
for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
    data_list = list(range(N))
    data_set = set(range(N))
    target = N - 1
    t_list = time_op(lambda: target in data_list)
    t_set = time_op(lambda: target in data_set)
    print(f"{N:>10} | {t_list:>9.2f}μs | {t_set:>9.2f}μs | {t_list/t_set:>7.0f}x")
```

**预期输出**: ratio 从 1.7× (N=100) 增长到 25000× (N=1M)。

#### G.2 实验 2: Attention 计时 (验证 O(n²d))

```python
import time
import torch
import torch.nn.functional as F

def attention_flops(n, d):
    """计算 attention 的 FLOPs (粗估)"""
    qk = 2 * n * n * d
    av = 2 * n * n * d
    return qk + av

def attention_time(n, d, device='cpu'):
    Q = torch.randn(n, d, device=device)
    K = torch.randn(n, d, device=device)
    V = torch.randn(n, d, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        S = Q @ K.T / (d ** 0.5)
        out = F.softmax(S, dim=-1) @ V
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / 10 * 1000  # ms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{'n':>6} | {'time':>10} | {'FLOPs':>15} | {'TFLOPS':>8}")
for n in [128, 256, 512, 1024, 2048, 4096]:
    t = attention_time(n, 64, device)
    f = attention_flops(n, 64)
    tflops = f / t / 1e12 if t > 0 else 0
    print(f"{n:>6} | {t:>9.3f}ms | {f:>15,} | {tflops:>7.3f}")
```

**预期**: time 随 $n$ 二次增长 ($n$ ×2, time ×4); TFLOPS 在大 $n$ 时上升 (GPU 利用率提高)。

#### G.3 实验 3: Linear Attention vs Standard

```python
import torch
import torch.nn.functional as F
import time

def standard_attention(Q, K, V):
    S = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
    return F.softmax(S, dim=-1) @ V

def linear_attention(Q, K, V, eps=1e-6):
    """Linear attention with elu+1 feature map (Katharopoulos 2020)"""
    Q = F.elu(Q) + 1
    K = F.elu(K) + 1
    KV = K.transpose(-2, -1) @ V  # (d, d)
    return (Q @ KV) / (Q @ K.sum(dim=-1, keepdim=True) + eps)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
B, h, d = 4, 8, 64

print(f"{'n':>6} | {'standard':>12} | {'linear':>12} | {'speedup':>8} | {'显存 (std)':>12} | {'显存 (lin)':>12}")
for n in [512, 1024, 2048, 4096, 8192]:
    try:
        Q = torch.randn(B, h, n, d, device=device)
        K = torch.randn(B, h, n, d, device=device)
        V = torch.randn(B, h, n, d, device=device)

        # Standard
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(3):
            out_std = standard_attention(Q, K, V)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_std = (time.time() - t0) / 3 * 1000
        mem_std = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0

        # Linear
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(3):
            out_lin = linear_attention(Q, K, V)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_lin = (time.time() - t0) / 3 * 1000
        mem_lin = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0

        speedup = t_std / t_lin if t_lin > 0 else 0
        print(f"{n:>6} | {t_std:>11.3f}ms | {t_lin:>11.3f}ms | {speedup:>7.1f}x | {mem_std:>11.3f}GB | {mem_lin:>11.3f}GB")
    except RuntimeError as e:
        print(f"{n:>6} | OOM: {e}")
        break
```

**预期**: 在 n ≥ 2048 时, linear attention 显著快 (3-10×) 且显存低。代价是数值近似, 精度有损 (1-3%)。

#### G.4 实验 4: 模拟退火 TSP

```python
import random
import math

def dist_matrix(coords):
    n = len(coords)
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d[i][j] = math.hypot(coords[i][0] - coords[j][0],
                                  coords[i][1] - coords[j][1])
    return d

def sa_tsp(dist, T0=100.0, alpha=0.995, iters=10000, seed=42):
    random.seed(seed)
    n = len(dist)
    route = list(range(n))
    random.shuffle(route)

    def cost(r):
        return sum(dist[r[i]][r[(i+1) % n]] for i in range(n))

    cur = cost(route)
    best, best_cost = route[:], cur
    T = T0

    for it in range(iters):
        i, j = sorted(random.sample(range(n), 2))
        nr = route[:i] + route[i:j+1][::-1] + route[j+1:]
        nc = cost(nr)
        if nc < cur or random.random() < math.exp(-(nc - cur) / T):
            route, cur = nr, nc
            if cur < best_cost:
                best, best_cost = route[:], cur
        T *= alpha
    return best, best_cost

# 测试
coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)]
d = dist_matrix(coords)
route, c = sa_tsp(d)
print(f"n=50, SA 距离: {c:.2f}")
# 与暴力对比 (n=10)
n_small = 10
coords_small = coords[:n_small]
d_small = dist_matrix(coords_small)
from itertools import permutations
best_brute = min(permutations(range(n_small)), key=lambda p: sum(d_small[p[i]][p[(i+1)%n_small]] for i in range(n_small)))
brute_cost = sum(d_small[best_brute[i]][best_brute[(i+1)%n_small]] for i in range(n_small))
print(f"n=10, 暴力最优: {brute_cost:.2f}")
# SA 误差: (c - brute_cost) / brute_cost
```

**预期**: SA 在 n=50 跑 1 秒, 误差 2-5%; n=10 时 SA 接近暴力最优 (误差 < 1%)。

#### G.5 实验 5: 顶点覆盖的暴力 / 贪心 / 2-approx 对比

```python
import random
from itertools import combinations

def vertex_cover_brute(G, n):
    """暴力: 枚举所有子集, 找最小顶点覆盖 O(2^n)"""
    best = None
    for r in range(1, n + 1):
        for S in combinations(range(n), r):
            if is_cover(G, S):
                if best is None or len(S) < len(best):
                    best = S
                    break
        if best is not None:
            break
    return best

def is_cover(G, S):
    S = set(S)
    for u, v in G:
        if u not in S and v not in S:
            return False
    return True

def vertex_cover_greedy(G, n):
    """贪心: 每次选覆盖最多未覆盖边的顶点 O(V + E)"""
    S = set()
    edges = set(map(frozenset, G))
    while edges:
        # 选度最大的顶点
        deg = {v: 0 for v in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        v = max(deg, key=deg.get)
        S.add(v)
        edges = {e for e in edges if v not in e}
    return S

def vertex_cover_2approx(G, n):
    """2-近似: 找最大匹配, 匹配两端都加入 O(E)"""
    S = set()
    edges = list(G)
    while edges:
        u, v = edges.pop()
        S.update([u, v])
        edges = [(a, b) for a, b in edges if a != u and a != v and b != u and b != v]
    return S

# 测试
random.seed(42)
n = 15
G = []
for i in range(n):
    for j in range(i+1, n):
        if random.random() < 0.4:
            G.append((i, j))

print(f"n={n}, |E|={len(G)}")
brute = vertex_cover_brute(G, n)
greedy = vertex_cover_greedy(G, n)
approx = vertex_cover_2approx(G, n)
print(f"暴力最优: {len(brute)} | 贪心: {len(greedy)} | 2-approx: {len(approx)}")
print(f"贪心 / 最优: {len(greedy)/len(brute):.2f}x, 2-approx / 最优: {len(approx)/len(brute):.2f}x")
```

**预期**: 2-approx 保证 ≤ 2× 最优, 贪心也是 2-approx (经验上略差)。都远好于 n=15 时的暴力枚举 (2^15 = 32768 个子集)。

#### G.6 实验 6: HNSW 模拟 (玩具版)

```python
import numpy as np
import random
import heapq

class MiniHNSW:
    """迷你 HNSW, 演示多层图导航"""
    def __init__(self, dim, M=4, ef_construction=20):
        self.dim = dim
        self.M = M
        self.ef = ef_construction
        self.vectors = []
        self.graphs = []  # 每层一个邻接表

    def add(self, v):
        self.vectors.append(v)
        level = int(np.random.exponential(0.5))
        while len(self.graphs) <= level:
            self.graphs.append({})
        # 简化: 每层都连 M 个最近邻
        for layer in range(level + 1):
            if not self.graphs[layer]:
                self.graphs[layer][len(self.vectors) - 1] = []
            else:
                # 找 M 近邻
                dists = [np.linalg.norm(v - self.vectors[i]) for i in self.graphs[layer]]
                neighbors = heapq.nsmallest(self.M, self.graphs[layer], key=lambda i: dists[list(self.graphs[layer]).index(i)] if i in self.graphs[layer] else float('inf'))
                # 简化: 随机选 M 个
                all_in_layer = list(self.graphs[layer].keys())
                chosen = random.sample(all_in_layer, min(self.M, len(all_in_layer)))
                self.graphs[layer][len(self.vectors) - 1] = chosen
                for c in chosen:
                    self.graphs[layer][c].append(len(self.vectors) - 1)

    def search(self, q, k=10, ef=50):
        # 简化: 暴力搜所有向量
        if not self.vectors:
            return []
        dists = [np.linalg.norm(q - v) for v in self.vectors]
        return np.argsort(dists)[:k].tolist()

# 测试
np.random.seed(42)
dim = 32
N = 1000
hnsw = MiniHNSW(dim)
for i in range(N):
    hnsw.add(np.random.randn(dim))

query = np.random.randn(dim)
result = hnsw.search(query, k=10)
print(f"HNSW 找到 top-10: {result[:5]}...")
# 实际 FAISS 库 (pip install faiss-cpu) 有生产级 HNSW / IVF-PQ
```

**注**: 这只是演示, 真实 HNSW 复杂得多 (Malkov & Yashunin 2018)。生产用 FAISS / ScaNN / Hnswlib。

---

### H. 强化学习与组合优化的深度联系

#### H.1 RL 的探索 vs 利用

RL 经典 trade-off: **exploration (探索新策略)** vs **exploitation (利用已知最优)**。

**复杂度视角**:

- **利用**: 在当前策略的局部最优附近, 类似于 "1 个 NPC 问题的局部搜索"
- **探索**: 跳出局部最优, 类似于 "模拟退火的高温阶段"

**ε-greedy**: 以 ε 概率随机探索, 1-ε 概率利用。简单但有效。
**UCB (Upper Confidence Bound)**: 选择"均值 + 不确定度"最大的动作, 理论保证 $O(\log T)$ 遗憾。
**Thompson Sampling**: 用后验分布采样动作, 贝叶斯视角。

**对应深度学习**:

| RL 概念 | DL 应用 |
| :--- | :--- |
| State | 当前模型权重 / 数据分布 |
| Action | 下一轮训练的超参 / 架构变更 |
| Reward | 验证集性能 |
| Policy | 调度策略 (bandit) |
| Episode | 一次完整训练 |

#### H.2 Bandit 算法在 NAS / AutoML 中的应用

**场景**: NAS 有 10 个候选操作, 每个操作训练 1 epoch 评估耗时 1 小时。**怎么选最少的"试错"次数找到最优?**

**朴素**: 全试 10 次, 10 小时。
**Bandit (Successive Halving, Hyperband)**: 先给 5 个候选跑 1 epoch, 取前 2 个跑 5 epoch, 取前 1 个跑 25 epoch。**总共 5+2×5+1×25 = 40 epoch-单位**, 比 10×25 = 250 节省 6×。

**复杂度**: 仍然要跑训练 (P 类), 但**总训练量减 5-10×**。

#### H.3 MCTS 在 AlphaGo / 围棋 AI 中的应用

**围棋**: 棋盘 19×19 = 361 落点, 状态数 $3^{361} \approx 10^{170}$。**比宇宙原子数还多**。

**MCTS (Monte Carlo Tree Search)**:

1. **选择**: 从根节点, 用 UCB 选最有希望的子节点, 递归到叶
2. **扩展**: 叶节点加一个新子节点
3. **模拟**: 从新节点随机走子到终局
4. **回传**: 更新路径上所有节点的胜率估计

**复杂度**: 每次模拟 $O(\text{棋盘深度}) = O(361)$, 总模拟次数 ~ 10^6, **总 FLOPs 10^8 级别**。

**AlphaGo 改进**: 用神经网络 (policy + value) 代替随机模拟, 强度跃升。

**对 DL 的启示**: **MCTS 是"用采样赌一个解"**, 应对指数状态空间的标准武器。LLM 的 chain-of-thought, Tree-of-Thought 都是这个思路的变种。

---

### I. 概率算法与 ML 的深度联系

#### I.1 蒙特卡洛方法

**思想**: 用随机采样估计期望 / 概率, 复杂度 $O(1/\epsilon^2)$ 达到 $\epsilon$ 精度 (与维度无关!)。

**ML 应用**:

| 任务 | 蒙特卡洛 |
| :--- | :--- |
| 期望估计 | $E[f(X)] \approx \frac{1}{N} \sum f(x_i)$ |
| 积分 | $\int f(x) dx \approx \frac{V}{N} \sum f(x_i)$ |
| 贝叶斯推断 | MCMC, HMC |
| 策略评估 | 多次 rollout 取平均 |
| 重要性采样 | $E[f] \approx \frac{1}{N} \sum f(x_i) p(x_i)/q(x_i)$ |

**PG (Policy Gradient)**: $\nabla J = E[\nabla \log \pi(a|s) R]$, 用采样估计, 复杂度 $O(\text{episodes})$。

#### I.2 MCMC 与 HMC

**MCMC (Markov Chain Monte Carlo)**: 构造一个 Markov 链, 平稳分布是目标分布 $p(x)$。从链上采样, 估计期望。

**Metropolis-Hastings**: 接受率 $\min(1, p(x') q(x|x') / p(x) q(x'|x))$。

**HMC (Hamiltonian Monte Carlo)**: 引入"动量", 用 Hamiltonian 动力学, 大步长不反弹, 接受率 90%+。**收敛快 10-100×**。

**贝叶斯神经网络**: 权重后验 $p(W | \text{data})$ 维度 = 参数数 (百万级), MCMC 在低维有效, **高维贝叶斯 NN 用变分推断 (VI) 替代**。

**对 DL 的启示**: 严格贝叶斯推断 $O(\exp(\text{params}))$, 工程上做不到。**变分推断 = 找一个"近似后验"作为代理**, 是 DL 中"近似换可行"的经典案例。

#### I.3 蒙特卡洛树搜索 (MCTS) 在 LLM 推理中的应用

**Tree-of-Thought (ToT)**: LLM 推理时, 用 MCTS 在多个推理路径中选最优。

**Self-Consistency**: 采样多个推理路径, 投票选多数。

**复杂度**: 每条路径 $O(n)$ tokens, 采样 $k$ 条 → $O(kn)$, **线性增长**。比 beam search 的 $O(2^n)$ 指数低。

**性能**: 数学推理 (GSM8K) 准确率从 60% 提到 90%+。

---

### J. 信息论与 Kolmogorov 复杂度

#### J.1 Kolmogorov 复杂度

**定义**: 字符串 $x$ 的 Kolmogorov 复杂度 $K(x)$ = 最短能生成 $x$ 的程序长度 (以 bit 计)。

**性质**:
- $K(x) \leq |x| + c$ (恒等映射程序长度)
- $K(x) \leq K(\text{Python}) + c$ (Python 程序能描述 $x$)
- **不可计算** (停机问题归约), 只能近似估计

**与 MDL (Minimum Description Length)**:

- MDL: 选模型 $M$ 使 $|M| + |D|M$ 最短, 其中 $D|M$ 是 $D$ 用 $M$ 编码的长度
- 实际: BIC, AIC 都是 MDL 的近似

**对 DL 的启示**:

- **预训练 = 学习一个"通用压缩器"** (压缩所有训练数据)
- **Occam 剃刀**: 简单模型泛化好, 因为短程序描述长数据
- **过度参数化 = 短程序能描述复杂数据** (NN 权重虽然多, 但结构是"短程序")

#### J.2 信息瓶颈

**Information Bottleneck (Tishby 2000)**: 找一个表示 $T(X)$, 最大化 $I(T; Y)$ (保留 $Y$ 信息) 同时最小化 $I(T; X)$ (压缩输入)。

**对 DL 的启示**:

- 深度网络的"信息瓶颈"假设: 训练时 $I(T; X)$ 下降, $I(T; Y)$ 上升, 最后 $T$ 是 $Y$ 的最小充分统计量
- 这个理论解释了一些深度学习的"神秘现象", 但**有争议** (Saxe 2018 反驳)

---

### K. 算法博弈论与多智能体

#### K.1 纳什均衡的复杂度

**问题**: 给定 $n$ 人博弈, 找纳什均衡 (每个玩家策略都是对其他人的最优响应)。

**2 人零和博弈**: 线性规划可解, **P 类**。

**$n$ 人一般博弈**: **PPAD 完全** (Chen & Deng 2006), NP-hard 的"近亲"。

**含义**: 多智能体 RL 中, 找 Nash 均衡是 NPH, 工程上**用近似均衡** (ε-Nash, 满足度 ε)。

#### K.2 机制设计与拍卖

**VCG 机制**: 真相报告是占优策略, 计算纳什均衡后分配。

**复杂度**: 第二价格拍卖 $O(n)$, 第一价格拍卖 $O(n \log n)$。

**广告系统 (RTB)**: 实时竞价, 每天 10^11 次拍卖, **每个拍卖 < 100 ms**。复杂拍卖机制 (如二价 + 保留价) 必须 $O(n)$ 内可算。

**对 DL 的启示**: 广告拍卖的"价值估计" 用深度学习, 拍卖机制本身用博弈论, **两者结合 = RL 拍卖 agent**。

---

### L. 进一步阅读与扩展

#### L.1 经典教材

1. **《Introduction to Algorithms》 (CLRS, 4th ed., 2022)**: 第 3 章大 O, 第 4 章主定理, 第 34 章 NP。
2. **《Algorithm Design》 (Kleinberg & Tardos, 2005)**: 第 8 章 NP 与计算不可约性, 第 11 章 PSPACE。
3. **《Computational Complexity: A Modern Approach》 (Arora & Barak, 2009)**: 研究生级, 包含 PCP 定理, 平均复杂度, 密码学。
4. **《The Nature of Computation》 (Moore & Mertens, 2011)**: 物理 / 信息 / 计算的统一视角。

#### L.2 深度学习系统

1. **《Deep Learning Systems》 (Chen, 2022)**: CMU 11-667, 涵盖数据并行 / 模型并行 / 流水线。
2. **《Efficient Processing of Deep Neural Networks》 (Vivienne Sze, 2020)**: 量化 / 剪枝 / 神经架构搜索。
3. **《Programming Massively Parallel Processors》 (Kirk & Hwu, 2017)**: GPU 编程与复杂度。

#### L.3 关键论文

| 论文 | 年份 | 贡献 |
| :--- | :--- | :--- |
| Cook, "Complexity of theorem-proving" | 1971 | Cook-Levin 定理, SAT 是第一个 NPC |
| Karp, "Reducibility among combinatorial problems" | 1972 | 21 个 NPC 问题 |
| Shor, "Algorithms for quantum computation" | 1994 | 量子分解多项式 |
| Bahdanau et al., "Neural machine translation by jointly learning to align" | 2014 | 注意力机制 |
| Vaswani et al., "Attention is all you need" | 2017 | Transformer |
| Devlin et al., "BERT" | 2018 | 双向预训练 |
| Brown et al., "GPT-3" | 2020 | In-context 学习 |
| Dao et al., "FlashAttention" | 2022 | 把 O(n²) 显存压到 O(n) |
| Hoffmann et al., "Chinchilla" | 2022 | $P$ 和 $D$ 等比例增长 |
| Gu & Dao, "Mamba" | 2023 | 线性时间序列建模 |
| Touvron et al., "LLaMA" | 2023 | 开源大模型 |

#### L.4 在线资源

- **CMU 15-451 (Avidan & Gutfraind)**: 算法讲义, NP-completeness 章节极佳
- **MIT 6.006 (Erik Demaine)**: 视频 + 讲义
- **Stanford CS221 / CS231n / CS224n**: 兼顾算法 + DL
- **Andrej Karpathy "Neural Networks: Zero to Hero"**: YouTube 系列, 从零写 transformer
- **Sebastian Raschka "Build a Large Language Model"**: 从零构建 LLM 的代码书

#### L.5 工程资源

- **FAISS** (Facebook): ANN 库, HNSW / IVF-PQ / OPQ
- **vLLM** (UC Berkeley): 高吞吐 LLM 推理, PagedAttention
- **DeepSpeed** (Microsoft): 分布式训练, ZeRO
- **Megatron-LM** (NVIDIA): 张量并行 + 流水线并行
- **FlashAttention** (Dao-AILab): 官方 CUDA 实现
- **xFormers** (Meta): 高效 attention 变体集合

---

### M. 一页纸总结: P / NP / NPH 速查

```text
                ┌────────────────────────────────────────┐
                │          复杂度谱系 (从易到难)          │
                ├────────────────────────────────────────┤
                │ P ⊆ NP ⊆ PSPACE ⊆ EXP                  │
                │       ∪                                │
                │     co-NP                              │
                │       ∪                                │
                │  (NP ∩ co-NP = 质数判定, AKS)          │
                │                                        │
                │  NPC ⊆ NP ∩ NPH                       │
                │  NPH \ NPC = 优化版 NPC + 不可判定问题  │
                │                                        │
                │  开放问题: P vs NP (2002 悬奖 100 万)   │
                │  公认: P ⊊ NP ⊊ PSPACE (几乎肯定)      │
                └────────────────────────────────────────┘

   深度学习的 P / NP 速查:
   ┌────────────────────────┬─────────────┬──────────────────┐
   │ 任务                    │ 复杂度      │ 工程策略         │
   ├────────────────────────┼─────────────┼──────────────────┤
   │ 推理 (给定模型)         │ P (FLOPs)   │ 量化 / 蒸馏      │
   │ 训练 (给定架构)         │ P (FLOPs)   │ 分布式 + 混合精度 │
   │ 架构搜索 (NAS)         │ NPH         │ DARTS / 进化      │
   │ 量化精度选择           │ NPH         │ 启发式 / RL      │
   │ 算子调度               │ NPH         │ 贪心 / RL        │
   │ 推荐召回精确 Top-K     │ P           │ ANN (HNSW)       │
   │ 推荐召回组合选品       │ NPC         │ 贪心 / 强化学习   │
   │ 数据增强策略搜索       │ NPH         │ AutoAugment / RL │
   │ KV-Cache 调度          │ P           │ PagedAttention    │
   │ 模型剪枝 (子网搜索)    │ NPH         │ 启发式 / 蒸馏     │
   └────────────────────────┴─────────────┴──────────────────┘
```

---

### N. 工程师常用的"复杂度速算口诀"

1. **序列长度 $n$**: $n^2$ 在 $n = 10^4$ 时 = $10^8$ (慢), $n^2$ 在 $n = 10^6$ 时 = $10^{12}$ (不可行)。
2. **参数量 $P$**: fp16 推理 $P \times 2$ 字节显存, fp32 训练 $P \times 16$ 字节 (含优化器 + 梯度 + 激活近似)。
3. **Transformer 训练 FLOPs**: $\approx 6 P D$ (3 倍前向, $P$ 参数, $D$ tokens)。
4. **矩阵乘法**: $(n \times d) \times (d \times m) = O(ndm)$ FLOPs (不是 $O(n^3)$)。
5. **推荐召回**: $|U| \times |I|$ 是笛卡尔积, 双塔把它分解为 $|U| + |I|$ (对数检索)。
6. **GNN 层数**: $k$ 层感受野 = $d^k$, $d$ 平均度数, $k = 2-3$ 即可, $k = 6$ 已接近全图。
7. **KV-Cache**: 推理时显存与序列长度线性, **PagedAttention 提升 2-4× 吞吐**。
8. **任何 NPC 问题**: 看到 "找最优组合", "覆盖", "路径", "子集" → 放弃精确算法。

---

### O. 与"工程派 vs 理论派"的协调

工程师和理论家对复杂度的态度经常冲突:

| 视角 | 关心 | 行动 |
| :--- | :--- | :--- |
| **理论派** | 最坏情况, 渐近阶 | 设计近似算法 + 复杂度证明 |
| **工程派** | 平均情况, 实测常数 | 跑 benchmark, 选最快实现 |
| **DL 派** | 特定数据 / 架构的"实复杂度" | 接受"够用"就行, 不追求最坏 |

**协调**:
- 理论告诉**"什么是不可能的"** (NPC → 放弃精确)
- 工程告诉**"什么是够用的"** (99% 准确 + 100× 速度 > 100% 准确 + 1× 速度)
- DL 把 NPC 软化: 训练时做"软搜索", 推理时做"硬决策"

**结论**: 复杂度理论不是"工程师的敌人", 而是"算力账本的会计准则"。会算账的工程师, 不会给 LLM 配 $O(n^2)$ 的 attention 不知情。

---

### P. 2025-2026 年前沿: 算法复杂度的最新进展

#### P.1 FlashAttention-3 (2024)

**Hopper 架构 (H100) 上的 WGMMA 指令**, 充分利用 tensor core:

- 传统 A100: bf16 峰值 312 TFLOPS
- FlashAttention-3 on H100: 达到 **740 TFLOPS** (peak 的 75%)
- 接近理论上限 (peak 989 TFLOPS)

**意义**: 进一步推高"墙在哪" (memory wall), 让我们用同样硬件训更大模型。

#### P.2 Mamba-2 与 SSD (2024)

**Mamba-2** 把状态空间模型的并行扫描算法做到接近 Transformer 速度, 同时保持 $O(n)$ 复杂度。

**SSD (State Space Duality)**: 证明 SSM 和线性 attention 在数学上**等价**, 只是约束不同。

**意义**: 架构选择不再是"transformer vs mamba" 的二元对立, 而是"用 SSD 框架统一"。

#### P.3 Mixture of Experts (MoE) 的复杂度

**MoE**: 每次只激活部分专家 (如 8 个中的 2 个), 总参数 $P$ 大, 激活参数 $P_a$ 小。

**复杂度账**:

| 维度 | 稠密 (Dense) | MoE |
| :--- | :--- | :--- |
| 参数量 | $P$ | $P$ (大) |
| 激活量 (per token) | $P$ | $P_a = P / 4$ (小) |
| FLOPs (per token) | $2P$ | $2P_a$ |
| 显存 | $P \times 2$ bytes | $P \times 2$ bytes (全专家加载) |
| 通信 (all-to-all) | 无 | 显著 (token 路由) |

**代表**: Mixtral 8x7B (总 47B, 激活 13B), GPT-4 (估计 8x220B)。

**工程账**: 训练 FLOPs 按 $P_a$ 算, 推理延迟按 $P_a$ 算, 但**显存按 $P$ 算 + 通信**。所以 MoE 适合"延迟 / FLOPs 敏感, 显存 / 通信宽裕"的场景。

#### P.4 推测解码 (Speculative Decoding)

**思想**: 用小模型 (draft) 快速生成 K 个候选 token, 大模型 (target) 一次验证。

**复杂度**:

- Draft 模型: K 次小模型推理, FLOPs $K \cdot 2P_d$
- Target 验证: 1 次大模型推理, FLOPs $2P_t$, 但**并行处理 K 个 token**

**加速**: 接受率 $r$ 时, 平均加速 $r \cdot K / (1 + (1-r) \cdot K)$ 倍。$K = 5, r = 0.8$ 时 ~ 2-3× 加速。

**意义**: 用算法复杂度换延迟, 在 LLM 推理领域已成标配。

#### P.5 长上下文 (100K+ tokens) 的工程挑战

**Long-context 挑战**:

- Attention $O(n^2)$: 100K → $10^{10}$ (不可行)
- 显存: KV-Cache $O(nd)$: 100K × 4096 dim × 80 层 × 4 bytes ≈ 130 GB (单卡装不下)

**解决方案**:

| 方法 | 思路 | 复杂度 |
| :--- | :--- | :--- |
| **Sparse attention** | 局部窗口 + 少数全局 | $O(n \cdot w)$ |
| **Linear attention** | 核函数近似 | $O(nd^2)$ |
| **Mamba / SSM** | 状态空间递推 | $O(nd^2)$ |
| **Ring attention** | 跨卡切分序列 | $O(n^2 d / N)$ |
| **Flash Decoding** | 并行 decode | $O(nd)$ per step |
| **Paged KV-Cache** | 分页管理 (vLLM) | $O(nd)$ 显存, 碎片少 |

**2024 年里程碑**: Gemini 1.5 支持 1M tokens context, Claude 3 支持 200K。

---

### Q. 终章: 复杂度与"可计算性"的边界

#### Q.1 停机问题: 计算的极限

**停机问题**: 给定程序 $P$ 和输入 $x$, $P(x)$ 是否停机?

**定理 (Turing 1936)**: 停机问题**不可判定** (undecidable)。即不存在通用算法能判断任意程序是否停机。

**证明**: 反证。假设 $H(P, x)$ 存在, 构造 $D(P) = \neg H(P, P)$, 推出 $D(D) = \neg D(D)$, 矛盾。

**对 DL 的启示**:

- 通用"程序验证"不可行, 神经网络验证只能是**特定网络 / 特定性质**的子集
- 模型可解释性: 不存在通用算法能解释任意黑盒模型
- 但**具体任务上的可解释性**可以做到 (SHAP, LIME, attention 可视化)

#### Q.2 Rice 定理: 任何非平凡的语义性质都不可判定

**Rice 定理**: 任何"程序行为的非平凡性质"都不可判定。

**例子**:

- "程序 P 运行时是否访问网络?" 不可判定
- "程序 P 是否能终止?" 不可判定 (停机)
- "程序 P 是否对所有输入返回 0?" 不可判定

**对 DL 的启示**: 没有通用算法能"理解"任意模型的"意图"。这是 LLM 黑盒性的理论根源。

#### Q.3 算法信息论: 复杂度是信息的

**Kolmogorov 复杂度** + **算法概率** (Solomonoff 1978):

- 算法概率 $P(x) = \sum_{p : U(p) = x} 2^{-|p|}$: 越短的程序生成 $x$ 概率越大
- Solomonoff 归纳: 预测序列时, 加权所有能解释历史的程序

**对 DL 的启示**:

- LLM 预训练 = 近似 Solomonoff 归纳 (大语言模型描述大语料)
- 训练数据越"可压缩" (有规律), 模型学得越快 (类似 MDL)
- 不可压缩的随机数据, 模型永远学不会 (类似 PAC 学习下界)

---

> **附录版本**: v1.0
> **最后更新**: 2026-06-27
> **作者**: Sisyphus (OhMyOpenCode)
> **配套代码**: §G 所有代码均可直接运行 (PyTorch 2.x + Python 3.10+)
