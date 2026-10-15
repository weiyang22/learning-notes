---
title: 附录 E:面试与 Q&A 速查(50 问 + 寄语)
tags:
  - 面试
  - Q&A
  - 搜推广面试
  - 推荐系统面试
  - 工程师寄语
  - 搜推广
description: 本附录收录搜推广/生成式推荐面试中高频出现的 50 个核心问题,按"概念-机制-工业实践"三层组织,附工程师从 CV 转搜推广的寄语
last_updated: 2026-07-08
status: ✅ v1.0
---

# 附录 E:面试与 Q&A 速查

> **本附录定位**: 搜推广面试高频问题速查。50 个问题按"基础概念 → 模型机制 → 工业实践 → 前沿趋势"排列,每个问题 1-2 句话直击核心,适合 5 分钟内快速过一遍。
>
> **使用建议**:
> - 面试前 1 小时过一遍
> - 写技术博客/技术分享时找灵感
> - 转岗同学建立"搜推广知识图谱"

---

## E.0 寄语:从 CV 转搜推广,思维范式如何切换?

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

---

## E.1 50 个高频面试问题

### 基础概念(1-15)

1. **AUC 的物理意义是什么?为什么不用准确率?**
   - AUC = 随机一对(正,负)样本,正样本排在负样本前的概率;不依赖阈值,适合样本不平衡

2. **BPR Loss 公式?和 Cross-Entropy 的区别?**
   - `L = -log σ(s_pos - s_neg)`,pairwise;CE 是 pointwise

3. **FM vs DeepFM 区别?**
   - FM 二阶交叉显式,DeepFM 加了 DNN 高阶隐式

4. **Embedding 维度怎么选?**
   - `min(64, max(8, vocab^0.25 × 4))`;过大过拟合,过小欠拟合

5. **负采样策略有哪些?**
   - 随机 / popularity / hard negative / in-batch

6. **DIN 的 Attention 和 Transformer Self-Attention 区别?**
   - DIN 是 target-item 作为 query 去 attend 历史;Self-Attention 是历史内部互相 attend

7. **MMoE 为什么会有跷跷板现象?**
   - 多个 Expert 被多个 Tower 共享,梯度冲突;PLE 用 task-specific Expert 解决

8. **双塔召回为什么不能精排?**
   - 用户侧和物品侧表征分开算,丢失 cross 特征

9. **冷启动怎么搞?**
   - 侧信息(类目/品牌/价格)+ 元学习 + 探索流量

10. **怎么评估 LLM 推荐?**
    - 离线:NDCG/HitRate;在线:CTR/时长;LLM 特有:流畅度/多样性

11. **RoPE 为什么可以外推?**
    - 旋转矩阵的相对位置信息,长度外推时数学性质保持

12. **InfoNCE 公式?和 Softmax 关系?**
    - `L = -log(exp(s+)/Σexp(s-))`,本质是 (softmax + CE) on 相似度

13. **生成式推荐 vs 判别式推荐的本质区别?**
    - 判别式:打分 y = f(x);生成式:直接生成 item_id 序列

14. **RQ-VAE 为什么用残差?**
    - 每级量化后用残差继续,降低重建误差;比单级 VQ-VAE 精细

15. **DLRM 的关键设计选择?**
    - EmbeddingBag(变长) + Bottom MLP + Feature Interaction(点积) + Top MLP

### 模型机制(16-30)

16. **Embedding 显存爆炸怎么办?**
    - 分片(TBE)+ 量化(INT8)+ 哈希碰撞 + 稀疏更新

17. **训练数据穿越怎么检测?**
    - 特征按时间分区;按时间 shuffle;训练/验证 gap 检查

18. **特征一致性怎么保证?**
    - 训练时 PS 拉特征,推理时 Redis 拉;统一特征处理 SDK

19. **混合精度训练为什么有效?**
    - FP16 矩阵乘快 + 显存省一半;GradScaler 防 underflow

20. **DeepSpeed ZeRO 的三个阶段?**
    - Stage1:优化器状态分片;Stage2:+ 梯度分片;Stage3:+ 参数分片

21. **数据并行 vs 模型并行 vs 流水并行?**
    - 数据:模型复制,数据切分;模型:模型切分到多卡;流水:按层切

22. **召回为什么用近似检索?**
    - 物品库百万到十亿级,精确 KNN O(N) 不可行;Faiss/HNSW 做 ANN

23. **TPU vs GPU vs NPU 怎么选?**
    - TPU:Google Cloud/大模型;GPU:A100/H100 通用;NPU:国产化

24. **KV-Cache 是什么?为什么省显存?**
    - 推理时缓存历史 K/V,避免重算;显存换时间

25. **PagedAttention 为什么是 vLLM 核心?**
    - 把 KV 分页存储,减少碎片,提升 batch 吞吐 10-20x

26. **Flash Attention 为什么快?**
    - IO-Aware 分块,减少 HBM 读写,理论 FLOPs 不变但实际加速

27. **A/B 实验怎么做?**
    - 流量分桶(同层用户稳定)+ 指标监控(短/中/长)+ 反转检查

28. **离线涨了线上没涨,可能原因?**
    - 数据穿越 / 特征不一致 / 流量倾斜 / 长尾差

29. **训练 Loss 不收敛怎么排查?**
    - lr 太大 / 数据有 NaN / 模型结构 bug / 标签错

30. **推理服务延迟突增怎么排查?**
    - QPS 升高 / 慢请求 / 缓存击穿 / 特征服务 GC

### 工业实践(31-40)

31. **TensorRT 加速原理?**
    - 算子融合 + 内核调优 + 精度校准 + 静态 shape

32. **AOT vs JIT 编译?**
    - AOT 提前编译成 .pt,启动快;JIT 运行时编译,灵活

33. **量化有哪些粒度?**
    - per-tensor / per-channel / per-group;粒度细,精度高

34. **怎么监控线上模型?**
    - 输入分布监控(KL 漂移)+ 输出分布监控(预测值分布)+ 业务指标

35. **多任务学习 Loss 怎么加权?**
    - 手动调 / GradNorm 自动调 / Uncertainty weighting

36. **DLRM 的特征交互怎么做?为什么用点积?**
    - 特征两两点积 + MLP,模拟二阶交叉;点积比 concat 省参数

37. **HSTU 的核心改进?**
    - 统一生成式架构 + 相对注意力偏置 + 4 个预训练任务

38. **GRPO 和 PPO 的区别?**
    - GRPO 不需要 critic,用组内归一化估计 advantage;更省显存

39. **TIGER 的推理流程?**
    - 训:item → RQ-VAE → S-ID;训:用户序列 → S-ID 序列;推:beam search 生成 S-ID

40. **为什么 GR 用 Decoder-only 而不是 Encoder-Decoder?**
    - 生成任务天然自回归;Encoder-Decoder 多余;参数量更小

### 前沿趋势(41-50)

41. **世界模型在推荐中怎么用?**
    - 学用户行为转移函数;模拟未来状态;做 RL 训练环境

42. **PagedAttention 和 Flash Attention 区别?**
    - PagedAttention 是显存管理(分页);FlashAttention 是计算优化(IO-Aware)

43. **LoRA 微调的数学原理?**
    - `W' = W + ΔW = W + BA`,B∈R^{d×r},A∈R^{r×k},r≪d

44. **知识蒸馏的 4 大损失?**
    - Logit KD / Feature KD / Relation KD / Contrastive KD

45. **InfoNCE 温度系数 τ 的影响?**
    - 小 τ:分布尖锐,聚焦难负样本;大 τ:分布平滑

46. **RAG 在推荐中怎么用?**
    - 召回增强:用户历史检索物品作为 prompt 上下文

47. **多场景多任务和迁移学习区别?**
    - 多场景:同任务不同分布;多任务:不同任务;迁移:不同领域

48. **MLE 和 RLHF 训练目标区别?**
    - MLE:最大化似然 P(y|x);RLHF:最大化 reward - KL 散度

49. **推荐系统中的因果推断怎么用?**
    - IPS 校正曝光偏差;Uplift 评估营销效果

50. **OneRec / RecGPT 的端到端 LLM 推荐怎么做?**
    - 用户行为 → S-ID 序列 → LLM 编码 → Beam Search 生成 → 物品解码

### A.8 搜推广岗位面试 50 道高频题(简答提纲)

#### 基础算法(15 题)

1. **AUC 的物理意义是什么?为什么不用准确率?**
   - AUC = 随机一对(正,负)样本,正样本排在负样本前的概率;不依赖阈值,适合样本不平衡
2. **BPR Loss 公式?和 Cross-Entropy 的区别?**
   - `L = -log σ(s_pos - s_neg)`,pairwise;CE 是 pointwise
3. **FM vs DeepFM 区别?**
   - FM 二阶交叉显式,DeepFM 加了 DNN 高阶隐式
4. **Embedding 维度怎么选?**
   - `min(64, max(8, vocab^0.25 × 4))`;过大过拟合,过小欠拟合
5. **负采样策略有哪些?**
   - 随机 / popularity / hard negative / in-batch
6. **DIN 的 Attention 和 Transformer Self-Attention 区别?**
   - DIN 是 target-item 作为 query 去 attend 历史;Self-Attention 是历史内部互相 attend
7. **MMoE 为什么会有跷跷板现象?**
   - 多个 Expert 被多个 Tower 共享,梯度冲突;PLE 用 task-specific Expert 解决
8. **双塔召回为什么不能精排?**
   - 用户侧和物品侧表征分开算,丢失 cross 特征
9. **冷启动怎么搞?**
   - 侧信息(类目/品牌/价格)+ 元学习 + 探索流量
10. **怎么评估 LLM 推荐?**
    - 离线:NDCG/HitRate;在线:CTR/时长;LLM 特有:流畅度/多样性
11. **RoPE 为什么可以外推?**
    - 旋转矩阵的相对位置信息,长度外推时数学性质保持
12. **InfoNCE 公式?和 Softmax 关系?**
    - `L = -log(exp(s+)/Σexp(s-))`,本质是 (softmax + CE) on 相似度
13. **生成式推荐 vs 判别式推荐的本质区别?**
    - 判别式:打分 y = f(x);生成式:直接生成 item_id 序列
14. **RQ-VAE 为什么用残差?**
    - 每级量化后用残差继续,降低重建误差;比单级 VQ-VAE 精细
15. **DLRM 的关键设计选择?**
    - EmbeddingBag(变长) + Bottom MLP + Feature Interaction(点积) + Top MLP

#### 工程实践(20 题)

16. **Embedding 显存爆炸怎么办?**
    - 分片(TBE)+ 量化(INT8)+ 哈希碰撞 + 稀疏更新
17. **训练数据穿越怎么检测?**
    - 特征按时间分区;按时间 shuffle;训练/验证 gap 检查
18. **特征一致性怎么保证?**
    - 训练时 PS 拉特征,推理时 Redis 拉;统一特征处理 SDK
19. **混合精度训练为什么有效?**
    - FP16 矩阵乘快 + 显存省一半;GradScaler 防 underflow
20. **DeepSpeed ZeRO 的三个阶段?**
    - Stage1:优化器状态分片;Stage2:+ 梯度分片;Stage3:+ 参数分片
21. **数据并行 vs 模型并行 vs 流水并行?**
    - 数据:模型复制,数据切分;模型:模型切分到多卡;流水:按层切
22. **召回为什么用近似检索?**
    - 物品库百万到十亿级,精确 KNN O(N) 不可行;Faiss/HNSW 做 ANN
23. **TPU vs GPU vs NPU 怎么选?**
    - TPU:Google Cloud/大模型;GPU:A100/H100 通用;NPU:国产化
24. **KV-Cache 是什么?为什么省显存?**
    - 推理时缓存历史 K/V,避免重算;显存换时间
25. **PagedAttention 为什么是 vLLM 核心?**
    - 把 KV 分页存储,减少碎片,提升 batch 吞吐 10-20x
26. **Flash Attention 为什么快?**
    - IO-Aware 分块,减少 HBM 读写,理论 FLOPs 不变但实际加速
27. **A/B 实验怎么做?**
    - 流量分桶(同层用户稳定)+ 指标监控(短/中/长)+ 反转检查
28. **离线涨了线上没涨,可能原因?**
    - 数据穿越 / 特征不一致 / 流量倾斜 / 长尾差
29. **训练 Loss 不收敛怎么排查?**
    - lr 太大 / 数据有 NaN / 模型结构 bug / 标签错
30. **推理服务延迟突增怎么排查?**
    - QPS 升高 / 慢请求 / 缓存击穿 / 特征服务 GC
31. **TensorRT 加速原理?**
    - 算子融合 + 内核调优 + 精度校准 + 静态 shape
32. **AOT vs JIT 编译?**
    - AOT 提前编译成 .pt,启动快;JIT 运行时编译,灵活
33. **量化有哪些粒度?**
    - per-tensor / per-channel / per-group;粒度细,精度高
34. **怎么监控线上模型?**
    - 输入分布监控(KL 漂移)+ 输出分布监控(预测值分布)+ 业务指标
35. **多任务学习 Loss 怎么加权?**
    - 手动调 / GradNorm 自动调 / Uncertainty weighting

#### 深度专题(15 题)

36. **DLRM 的特征交互怎么做?为什么用点积?**
    - 特征两两点积 + MLP,模拟二阶交叉;点积比 concat 省参数
37. **HSTU 的核心改进?**
    - 统一生成式架构 + 相对注意力偏置 + 4 个预训练任务
38. **GRPO 和 PPO 的区别?**
    - GRPO 不需要 critic,用组内归一化估计 advantage;更省显存
39. **TIGER 的推理流程?**
    - 训:item → RQ-VAE → S-ID;训:用户序列 → S-ID 序列;推:beam search 生成 S-ID
40. **为什么 GR 用 Decoder-only 而不是 Encoder-Decoder?**
    - 生成任务天然自回归;Encoder-Decoder 多余;参数量更小
41. **世界模型在推荐中怎么用?**
    - 学用户行为转移函数;模拟未来状态;做 RL 训练环境
42. **PagedAttention 和 Flash Attention 区别?**
    - PagedAttention 是显存管理(分页);FlashAttention 是计算优化(IO-Aware)
43. **LoRA 微调的数学原理?**
    - `W' = W + ΔW = W + BA`,B∈R^{d×r},A∈R^{r×k},r≪d
44. **知识蒸馏的 4 大损失?**
    - Logit KD / Feature KD / Relation KD / Contrastive KD
45. **InfoNCE 温度系数 τ 的影响?**
    - 小 τ:分布尖锐,聚焦难负样本;大 τ:分布平滑
46. **RAG 在推荐中怎么用?**
    - 召回增强:用户历史检索物品作为 prompt 上下文
47. **多场景多任务和迁移学习区别?**
    - 多场景:同任务不同分布;多任务:不同任务;迁移:不同领域
48. **MLE 和 RLHF 训练目标区别?**
    - MLE:最大化似然 P(y|x);RLHF:最大化 reward - KL 散度
49. **推荐系统中的因果推断怎么用?**
    - IPS 校正曝光偏差;Uplift 评估营销效果
50. **OneRec / RecGPT 的端到端 LLM 推荐怎么做?**
    - 用户行为 → S-ID 序列 → LLM 编码 → Beam Search 生成 → 物品解码

> **面试策略**:基础算法 15 题是必背(LeetCode 难度);工程实践 20 题是工作必会;深度专题 15 题是简历亮点。回答时配合"我们项目里怎么用"。

---

---

> **附录 E 的最后补充**:
>
> 写完这一节,本教程从 **0(纯 CV 转行)** 到 **1(能上手搜推广项目)** 的链路就完整了。
>
> 接下来该看什么?三个推荐:
> 1. **代码驱动**:用附录 D.6 的两周大纲,每天 2-3 小时,14 天后你能给团队讲清楚 DLRM 和 GR 的区别
> 2. **论文驱动**:用附录 A 的 50 篇精读清单,每月 5 篇,一年后你能在 paper 群里发言
>
> 搜推广是个工程 + 算法双轮驱动的领域,愿你在这条路上越走越稳。

---

## E.2 扩展:100 道高频面试题深度解析

> **本节定位**: 在原 50 题基础上扩展 50 道深度题,涵盖工业实践、算法细节、架构选型。每题配详细解答、相关公式和工业案例。

### E.2.1 模型架构类 (51-65)

#### 51. **DLRM 的 Bottom MLP 为什么要单独设计?**

**答**: 不同特征量纲不同 (年龄 0-100, 价格 0-10000), Bottom MLP 把它们投影到统一 dim $D$,便于后续交互。

**公式**:
$$z_{\text{cont}} = \text{MLP}_{\text{cont}}(x_{\text{cont}}) \in \mathbb{R}^D$$
$$z_{\text{cat}} = \text{Emb}(x_{\text{cat}}) \in \mathbb{R}^D$$

**关键**: MLP 输出维度和 Embedding 维度一致, 这样它们可以做点积交互。

```python
class BottomMLP(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)
```

#### 52. **DLRM 的特征交互为什么用点积,不用 concat?**

**答**: 点积是二阶交叉的紧凑表达。$N$ 个特征两两点积 = $N(N-1)/2$ 个标量,参数量 $O(N^2)$; concat 后接 MLP = $O(N \cdot D^2)$, 点积参数少。

**公式对比**:
- 点积: $\sum_{i<j} z_i \cdot z_j$, 标量
- Concat: $[z_1, z_2, ..., z_N]$, 然后 $\text{MLP}(\cdot)$

| 方法 | 参数量 | 计算量 | 表达力 |
| :--- | :--- | :--- | :--- |
| Concat + MLP | $O(ND^2)$ | $O(ND^2)$ | 高 (任意函数) |
| 点积 | $O(N^2)$ | $O(N^2 D)$ | 中 (二阶) |
| FM | $O(ND)$ | $O(ND)$ | 中 (二阶, 隐式) |
| Transformer | $O(N^2 D)$ | $O(N^2 D)$ | 高 (任意阶) |

**工业经验**: DLRM 用点积, 后面接 Top MLP 来补偿非线性。

#### 53. **DeepFM 相比 FM 的核心改进?**

**答**: FM 只显式建模二阶交叉, DeepFM 加 DNN 隐式建模高阶交叉。

**架构**:
```text
Feature → FM (二阶) ─┐
              ↓       ↓
            Concat → Top MLP → Prediction
              ↑
Feature → DNN (高阶)
```

**优势**:
- FM 部分保留了"可解释性" (哪个交叉重要)
- DNN 部分增强了"高阶交互"
- 端到端训练, 无需特征工程

#### 54. **xDeepFM 比 DeepFM 强在哪?**

**答**: xDeepFM 引入 **CIN (Compressed Interaction Network)** 显式建模高阶交叉, 而非依赖 DNN 隐式学习。

**CIN 结构**:
$$X_{h+1} = \text{CIN}(X_h, X_0), \quad X_0 = \text{原始 Embedding}$$

**对比**:
- DeepFM: 隐式高阶交互 (DNN 黑盒)
- xDeepFM: 显式高阶交互 (CIN 可解释)

**实际效果**: xDeepFM 比 DeepFM AUC 高 0.2-0.5%, 但训练慢 30%。

#### 55. **DIN 的 Local Activation Unit (LAU) 输入为什么是 4 个特征?**

**答**: 4 个特征模拟了"两个向量的相似度计算"的所有关键信号:

| 特征 | 含义 | 数学表示 |
| :--- | :--- | :--- |
| $h$ | 历史物品向量本身 | $h_j$ |
| $t$ | 目标物品向量 | $t$ |
| $h \odot t$ | 共同激活维度 | $h_j \cdot t$ (逐元素积) |
| $h - t$ | 差异维度 | $h_j - t$ |

**直觉**:
- $h \odot t$: "两个物品都关注的维度" (e.g., 都是美食类)
- $h - t$: "两个物品的差异" (e.g., 一个中餐一个西餐)
- 拼接 → MLP → 标量 attention 分数

```python
class LAU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, hist, target):
        """
        hist: [B, T, D]
        target: [B, D]
        return: [B, T] attention scores
        """
        T = hist.size(1)
        target = target.unsqueeze(1).expand(-1, T, -1)
        interactions = torch.cat([
            hist,                  # h
            target,                # t
            hist * target,         # h * t
            hist - target,         # h - t
        ], dim=-1)
        return self.mlp(interactions).squeeze(-1)  # [B, T]
```

#### 56. **DIEN 比 DIN 多了什么?**

**答**: DIEN 用 **AUGRU (Attention Update Gate GRU)** 显式建模"兴趣进化"。

**DIN 的局限**: 只关注"哪些历史重要", 不关注"兴趣怎么演变"。
**DIEN 的改进**:
1. **兴趣抽取层**: 用 GRU 抽取每步的兴趣表示
2. **兴趣进化层**: AUGRU 用 attention 控制 GRU 更新

**公式**:
$$h_t = \text{GRUCell}(x_t, h_{t-1})$$
$$\tilde{h}_t = (1 - \alpha_t) \odot h_{t-1} + \alpha_t \odot \tilde{h}'_t$$

其中 $\alpha_t$ 是 attention 分数 (与 target 相关)。

**效果**: AUC 比 DIN 高 1-2%, 长序列场景更明显。

#### 57. **BST (Behavior Sequence Transformer) 的注意力怎么 mask?**

**答**: BST 用 **Causal Mask + Padding Mask** 双 mask。

```python
def generate_causal_mask(seq_len, device):
    """下三角 mask, 只能看过去."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

def generate_padding_mask(ids, pad_id=0):
    """[B, T] 的 mask."""
    return (ids == pad_id)  # [B, T], True 表示 padding
```

**前向**:
```python
attn_scores = Q @ K.T / sqrt(D) + causal_mask[None, None, :, :]  # [B, H, T, T]
attn_scores = attn_scores.masked_fill(padding_mask[:, None, None, :], float('-inf'))
attn_weights = F.softmax(attn_scores, dim=-1)
out = attn_weights @ V
```

#### 58. **SIM 的两阶段 attention 各自的作用?**

**答**:

**GSU (General Search Unit) — 粗筛**:
- 输入: 用户全部历史 (1000+)
- 输出: Top-K 相关历史 (50)
- 方法: 简单点积 attention 或 query-aware retrieval

**ESU (Exact Search Unit) — 精排**:
- 输入: GSU 筛出的 Top-K (50)
- 输出: 注意力加权的表示
- 方法: DIN 风格的 Local Activation Unit

**优势**: 长序列 attention $O(T^2)$ 太慢, 先粗筛到 50 再精排, 速度 10x, 精度几乎不损失。

```python
class SIM(nn.Module):
    def __init__(self, dim, top_k=50):
        super().__init__()
        self.gsu = nn.Linear(dim, dim)  # 粗筛
        self.esu = LocalActivationUnit(dim)  # 精排
        self.top_k = top_k

    def forward(self, history_emb, target_emb):
        # GSU: 简单点积打分, 取 Top-K
        scores = history_emb @ target_emb.unsqueeze(-1)  # [B, T, 1]
        scores = scores.squeeze(-1)  # [B, T]
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        # 取出 Top-K 的 embedding
        topk_emb = torch.gather(history_emb, 1,
                                 topk_idx.unsqueeze(-1).expand(-1, -1, history_emb.size(-1)))
        # ESU: 精排 attention
        attn = self.esu(topk_emb, target_emb)  # [B, top_k]
        weighted = (topk_emb * attn.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return weighted
```

#### 59. **SASRec 用 Decoder 而不是 Encoder?**

**答**: SASRec 是单向序列建模 (只看过去预测未来), 所以用 Transformer **Decoder**。

**区别**:
- **Encoder**: 双向 attention (BERT 风格), 用于"理解"
- **Decoder**: 单向 attention (GPT 风格), 用于"生成"

**SASRec 的目标**: 给定 $h_1, h_2, ..., h_{t-1}$, 预测 $h_t$, 所以是单向。

```python
class SASRec(nn.Module):
    def __init__(self, n_items, dim=64, n_heads=2, n_layers=2, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        # 用 EncoderLayer 但加 causal mask
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.max_len = max_len

    def forward(self, seq_ids):
        """seq_ids: [B, T]"""
        B, T = seq_ids.shape
        pos = torch.arange(T, device=seq_ids.device).unsqueeze(0).expand(B, -1)
        emb = self.item_emb(seq_ids) + self.pos_emb(pos)  # [B, T, D]
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=seq_ids.device) * float('-inf'), diagonal=1)
        # Padding mask
        pad_mask = (seq_ids == 0)  # [B, T]
        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
        return out  # [B, T, D]
```

#### 60. **BERT4Rec 相比 SASRec 的核心改进?**

**答**: BERT4Rec 用 **双向 Transformer + Masked Language Modeling**。

**SASRec**: 单向 (只看过去), 自回归训练
**BERT4Rec**: 双向 (看上下文), MLM 训练

**MLM 训练**:
- 输入: 序列中随机 mask 15% 的 token
- 目标: 还原被 mask 的 token
- 优势: 双向信息, 学到的表示更丰富

**劣势**: 推理时要枚举所有可能位置, 比 SASRec 慢。

#### 61. **MMoE 的跷跷板现象 (Seesaw Phenomenon) 怎么解决?**

**答**: 用 **PLE (Progressive Layered Extraction)** 替代 MMoE。

**MMoE 的问题**:
- 多个 Tower 共享 Expert, 梯度冲突
- 任务 A 涨, 任务 B 反而跌

**PLE 的解决**:
- 每个任务有 **task-specific Expert** (独占)
- 还有 **shared Expert** (共享)
- 两层提取, 逐步分离

```python
class PLE(nn.Module):
    def __init__(self, n_experts, n_tasks, expert_dim=64):
        super().__init__()
        # 共享 Expert
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, expert_dim), nn.ReLU())
            for _ in range(n_experts)
        ])
        # Task-specific Expert
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(expert_dim, expert_dim), nn.ReLU())
                for _ in range(n_experts)
            ])
            for _ in range(n_tasks)
        ])
        # Task Gates
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, n_experts + len(self.shared_experts)),
                nn.Softmax(dim=-1)
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x, task_id):
        # 共享 expert 输出
        shared_out = torch.stack([e(x) for e in self.shared_experts], dim=1)  # [B, n_shared, D]
        # Task-specific 输出
        task_out = torch.stack([e(x) for e in self.task_experts[task_id]], dim=1)  # [B, n_experts, D]
        # 拼接
        all_out = torch.cat([shared_out, task_out], dim=1)  # [B, n_experts + n_shared, D]
        # Gate
        weights = self.task_gates[task_id](x[:, 0])  # 用第一个 token 的 embedding 当 query
        out = (all_out * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return out
```

#### 62. **双塔召回的核心假设和局限?**

**答**:

**核心假设**: 用户和物品的"相关性"可以通过独立 embedding 的 cosine 衡量。

**公式**: $\text{score}(u, i) = \cos(e_u, e_i) = \frac{e_u \cdot e_i}{\|e_u\| \|e_i\|}$

**优势**:
- 物品 embedding 可以**离线预计算**并索引
- 在线只需计算用户 embedding + ANN 检索
- 延迟极低 (< 10ms)

**局限**:
- **不能捕捉 cross 特征**: 用户-物品交叉信号丢失
- **不能精排**: 精排需要 user × item 共同编码
- **冷启动差**: 新物品没 embedding

**改进**:
- **Cross-Encoder**: 用 Transformer 同时编码 user 和 item (慢但准)
- **Poly-Encoder**: 多向量交互 (折中)
- **ColBERT**: 细粒度 late interaction

#### 63. **怎么把双塔升级到精排?**

**答**: 3 种方案 ——
1. **双塔 + 交叉网络**: 用户/物品各自编码后, 加交叉层
2. **目标塔蒸馏**: 双塔 teacher + cross-encoder student
3. **Late Interaction**: 不在 embedding 层交互, 在最后一层 attention

**工业实践** (Meta, 字节):
- 召回: 双塔 (快)
- 粗排: 双塔 + 简单 DNN (中等)
- 精排: DIN/DLRM + 交叉特征 (慢但准)

#### 64. **HSTU 的"统一架构"具体指什么?**

**答**: HSTU 把召回 + 排序 + 特征工程统一在 **1 个 Transformer backbone** 上。

**传统架构**:
```text
召回 (双塔) → 粗排 (DNN) → 精排 (DIN) → 重排 (DPP)
每个阶段独立模型
```

**HSTU**:
```text
所有阶段共享 Transformer backbone, 不同任务用不同 head
```

**优势**:
- 减少模型维护成本 (1 个 vs 4 个)
- 跨任务知识迁移 (召回学到的语义可帮助精排)
- 简化 serving 架构

**挑战**:
- 训练目标难统一 (召回是 ranking, 精排是 CTR)
- 推理延迟可能增加 (大 backbone)

#### 65. **GR (Generative Recommendation) 和判别式推荐的本质区别?**

**答**:

| 维度 | 判别式 | 生成式 |
| :--- | :--- | :--- |
| 输入 | (用户, 物品) 对 | 用户历史序列 |
| 输出 | 分数 $\hat{y}$ | 物品 token 序列 |
| 训练目标 | BCE/MSE on 分数 | NLL on token |
| 推理 | 遍历候选打分 | 自回归生成 |
| 适用 | 精排 | 召回 |

**公式**:
- 判别式: $\hat{y} = f(u, i; \theta)$, 最大化 $\log P(y | u, i)$
- 生成式: $\hat{i}_{t+1} = \arg\max_i P(i | u, i_{1:t}; \theta)$, 最大化 $\sum_t \log P(i_t | u, i_{<t})$

---

### E.2.2 训练优化类 (66-80)

#### 66. **为什么推荐系统不用普通 Adam?**

**答**: Embedding 梯度稀疏 (10亿行中只命中 100 个), 普通 Adam 对所有行维护状态浪费。

**解决方案**:
- **Sparse Adam**: 只对命中行更新, 状态内存 $O(\text{激活行数} \cdot D)$, 不是 $O(V \cdot D)$
- **AdaGrad**: 自适应学习率, 适合稀疏梯度
- **LAMB**: 大 batch 训练, Layer-wise 自适应

```python
# Sparse Adam
optimizer = torch.optim.SparseAdam([
    {'params': [model.user_embedding.weight]},  # 必须传 .weight
    # 注意: SparseAdam 只能优化稀疏 embedding, 不能优化 dense weight
], lr=1e-3)

# 完整版: Sparse Adam + 普通 Adam 混合
optimizer = torch.optim.Adam([
    {'params': model.dense_params, 'lr': 1e-3},       # 稠密参数
    {'params': model.embedding.weight, 'lr': 1e-3},   # Embedding 也用普通 Adam
])
# 实际: 工业上用 FTRL 或 SparseAdam 单独处理 Embedding
```

#### 67. **频率倒数 L2 正则 (YouTube 2016) 的数学原理?**

**答**: 给不同频次的 ID 不同正则强度。

**公式**:
$$\mathcal{L}_{\text{reg}} = \sum_{i \in \text{batch}} \frac{\lambda}{\sqrt{f(i)}} \|W_i\|_2^2$$

其中 $f(i)$ 是 ID $i$ 在训练集出现次数。

**直觉**:
- 头部 ID (高频): $f$ 大, 正则**弱**, 允许大范数 (被频繁训练)
- 长尾 ID (低频): $f$ 小, 正则**强**, 防止过拟合到少量样本

```python
def freq_aware_l2(weight, ids, freq, lam=1e-5):
    """频率倒数 L2 正则, YouTube DNN 风格."""
    w = weight[ids]                         # [B, D]
    f = freq[ids].clamp(min=1.0).sqrt()     # [B]
    return lam * (w.pow(2).sum(dim=1) / f).sum()
```

**为什么是 $\sqrt{f}$ 而不是 $f$?** 经验调, $\sqrt{f}$ 比 $f$ 收敛快, 比 $1$ (无正则) 强。

#### 68. **混合精度训练的 loss scaling 原理?**

**答**: FP16 动态范围小 ($\pm 6.5 \times 10^4$), 小梯度会 underflow 到 0。

**Loss Scaling 流程**:
1. 前向: 计算 loss, 乘以 scale factor (e.g., 65536)
2. 反向: 梯度也被 scale
3. 更新: optimizer 前把梯度除回 (unscale)
4. 检查: NaN/Inf → skip update

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in loader:
    optim.zero_grad()
    with autocast():
        loss = model(batch)

    scaler.scale(loss).backward()  # scale loss
    scaler.unscale_(optim)         # unscale 梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optim)              # 检查 NaN/Inf
    scaler.update()                 # 动态调 scale factor
```

**BF16**: 动态范围同 FP32, 无需 loss scaling, H100/A100/H800 强烈推荐。

#### 69. **DeepSpeed ZeRO 的 3 个阶段?**

**答**:

| Stage | 分片内容 | 显存节省 | 通信开销 |
| :--- | :--- | :--- | :--- |
| Stage 1 | Optimizer States (m, v) | ~4x | 略增 |
| Stage 2 | + Gradients | ~8x | 略增 |
| Stage 3 | + Parameters | ~线性 (与 GPU 数) | 显著 |

**Stage 1**: Adam 的 m, v 各 8 字节 ($\text{FP32} \times 2$), 分片后每卡只存 $1/N$, 内存省 4x。

**Stage 2**: 梯度同样分片, 总内存省 8x。

**Stage 3**: 参数也分片, 显存随 GPU 数线性缩放。但每次 forward 要 all-gather 参数, 通信开销大。

**工业实践**: Stage 1 + Stage 2 + CPU offload, 8 卡 A100 可训 50B 参数。

#### 70. **数据并行 vs 模型并行 vs 流水并行的 trade-off?**

**答**:

| 方式 | 切分维度 | 通信 | 适用 |
| :--- | :--- | :--- | :--- |
| 数据并行 (DDP) | batch | AllReduce 梯度 | 模型 < 单卡显存 |
| 模型并行 (TP) | 模型层/参数 | All-to-All | 模型 > 单卡显存 (e.g., 大 LLM) |
| 流水并行 (PP) | 模型层 | 点对点 (Send/Recv) | 模型很深 (> 24 层) |
| ZeRO | 优化器状态/梯度 | AllGather | 显存瓶颈但模型不大 |

**Embedding 表特别大**: 用 **行切分** (类似模型并行), 每卡存部分 ID。

**工业实践 (Meta HSTU)**:
- 数据并行 (DDP) for backbone
- 行切分 (TP) for Embedding 表
- 流水并行 (PP) for 深层 Transformer

#### 71. **L2 vs L1 vs Dropout 正则如何选择?**

**答**:

| 方法 | 适用 | 优势 | 劣势 |
| :--- | :--- | :--- | :--- |
| L2 (权重衰减) | 通用 | 简单, 抗过拟合 | 大权重都被衰减 |
| L1 | 稀疏特征选择 | 自动稀疏化 | 梯度不连续, 训练不稳定 |
| Dropout | 神经网络层 | 模拟集成, 强 | 推理需关闭 |
| BatchNorm | CNN/DNN | 隐式正则 | 与 Dropout 冲突 |
| LayerNorm | Transformer | 标准化, 稳 | 弱正则 |

**推荐系统经验**:
- Embedding: 频率倒数 L2 (YouTube)
- Dense (DNN/MLP): L2 + Dropout (0.1-0.3)
- Transformer: Dropout (0.1) + LayerNorm

#### 72. **Early Stopping vs Learning Rate Decay?**

**答**: 两者结合使用。

**Early Stopping**:
- 监控 val loss/AUC
- 连续 N 步不涨 → 停止
- 防止过拟合

**LR Decay**:
- Warmup (lr 0 → max in 1% 步)
- Cosine / Step / Exponential (lr max → 0)
- 帮助收敛到局部最优

```python
# 完整训练 Loop
best_auc = 0
patience = 0

for epoch in range(n_epochs):
    train_one_epoch(model, train_loader)
    val_auc = evaluate(model, val_loader)

    # LR decay
    scheduler.step(val_auc)  # ReduceLROnPlateau

    # Early stopping
    if val_auc > best_auc:
        best_auc = val_auc
        patience = 0
        save_checkpoint(model)
    else:
        patience += 1
        if patience >= 10:
            print("Early stop!")
            break
```

#### 73. **梯度累积 (Gradient Accumulation) 怎么用?**

**答**: 当 batch size 受显存限制时, 用多次小 batch 累积梯度, 等效大 batch。

```python
accumulation_steps = 8  # 等效 batch = 8 * batch_size
optimizer.zero_grad()

for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps  # 必须除以步数
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**优点**:
- 用小显存跑大 batch
- 减少 GPU 通信频率

**缺点**:
- BN/LayerNorm 行为可能改变 (因 batch 实际很小)
- 训练变慢 (等效 step 不变)

#### 74. **Embedding 梯度爆炸怎么排查?**

**答**:

**症状**: max_norm 监控曲线飙升, training loss 震荡, val loss 不收敛。

**排查步骤**:
1. **打印梯度范数**: 看是否 > 100
2. **打印单 ID 范数**: 看是否有 ID 异常
3. **数据检查**: 是否有重复样本 / 异常标签
4. **学习率**: 减半试试

**修复方案**:
```python
# 1. 全局梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 2. 按行裁剪 Embedding 梯度
def clip_emb_grad(weight, ids, max_norm=5.0):
    with torch.no_grad():
        unique_ids = ids.unique()
        grads = weight.grad[unique_ids]
        norms = grads.norm(dim=1, keepdim=True)
        scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
        weight.grad[unique_ids] = grads * scale

# 3. 减小学习率
for g in optimizer.param_groups:
    g['lr'] *= 0.5
```

#### 75. **在线学习 (Online Learning) 在推荐中怎么用?**

**答**: FTRL (Follow The Regularized Leader) 是工业标准。

**FTRL 公式**:
$$w_{t+1} = \arg\min_w \left( \sum_{s=1}^t g_s \cdot w + \frac{1}{2} \sum_{s=1}^t \sigma_s \|w - w_s\|^2 + \lambda \|w\|_1 \right)$$

**工程实现** (Google 2013 论文):
```python
# FTRL-Proximal 在线学习
class FTRL:
    def __init__(self, dim, alpha=0.1, beta=1.0, l1=0.1, l2=1.0):
        self.z = np.zeros(dim)        # 累积梯度
        self.n = np.zeros(dim)        # 累积平方梯度
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2

    def update(self, grad):
        self.n += grad ** 2
        self.z += grad - (self._predict() * self.l2 - self.z) * (self.n == 0)
        # ... (省略实际更新逻辑)
```

**优势**:
- 实时更新, 适应数据漂移
- 天然稀疏 (L1 正则)
- 工业验证 (Google Play 用了几亿次更新/秒)

#### 76. **Negative Sampling 在推荐中的多种实现?**

**答**:

| 方法 | 复杂度 | 效果 |
| :--- | :--- | :--- |
| 随机负采样 | O(N) | 一般 |
| 流行度采样 | O(log N) | 模拟曝光偏差 |
| Hard Negative | O(V) (需前向) | 强 |
| In-Batch Negative | O(B) (无额外前向) | 中 |
| 缓存式 Hard | O(K) (KV 缓存) | 强 + 快 |

```python
class HybridNegativeSampler:
    """混合负采样: 随机 + 流行度 + Hard."""
    def __init__(self, model, item_freq, alpha=0.4):
        self.model = model
        self.item_freq = item_freq  # [V]
        self.alpha = alpha          # popularity 权重

    def sample(self, batch_size, n_neg, user_emb, k_hard=10):
        # 50% popularity-weighted 随机
        n_pop = n_neg // 2
        pop_probs = self.item_freq ** self.alpha
        pop_probs /= pop_probs.sum()
        neg_pop = np.random.choice(len(self.item_freq), (batch_size, n_pop), p=pop_probs)

        # 50% hard: 用当前模型选 Top-K 错分样本
        n_hard = n_neg - n_pop
        with torch.no_grad():
            # 全部物品打分 (生产环境不可行, 实际用缓存)
            item_emb = self.model.item_emb.weight  # [V, D]
            scores = user_emb @ item_emb.T  # [B, V]
            topk_neg = scores.topk(n_hard + 1, dim=-1).indices  # +1 排除正样本
            neg_hard = topk_neg[:, 1:]  # 去掉 top-1 (假设是正样本)

        return np.concatenate([neg_pop, neg_hard.cpu().numpy()], axis=1)
```

#### 77. **双塔召回怎么用 In-Batch Negative?**

**答**: 一个 batch 内的其他样本作为负样本, 不用额外采样。

```python
class TwoTowerWithInBatch(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.user_tower = UserTower(dim)
        self.item_tower = ItemTower(dim)

    def forward(self, batch):
        user_emb = self.user_tower(batch['user'])  # [B, D]
        item_emb = self.item_tower(batch['item'])  # [B, D]
        # 矩阵乘: 每行 user_emb 对每列 item_emb 计算分数
        logits = user_emb @ item_emb.T  # [B, B]
        # 对角线是正样本, 其他是 in-batch 负样本
        labels = torch.arange(len(user_emb), device=user_emb.device)
        loss = F.cross_entropy(logits / 0.07, labels)
        return loss
```

**关键**: L2 归一化 + 温度系数 0.07。

**优势**: 无需额外采样, batch size = 隐式负样本数, 64 batch 就有 63 个负样本。

#### 78. **怎么训练超大规模 Embedding (>100GB)?**

**答**: 4 层组合方案 ——
1. **Embedding 分片**: TorchRec EmbeddingBagCollection, 8 卡分到 12.5GB/卡
2. **ZeRO Stage 3**: optimizer state + gradient + param 全分片
3. **CPU Offload**: 把不常用的 Adam 状态卸载到 CPU
4. **混合精度**: FP32 master + BF16 计算

```python
# DeepSpeed 配置示例
ds_config = {
    "train_batch_size": 1024,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "fp16": {"enabled": False},
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": 4,
}
```

#### 79. **Embedding Table 监控的核心指标?**

**答**:

| 指标 | 定义 | 健康范围 |
| :--- | :--- | :--- |
| **Coverage** | 至少被命中 1 次的 ID 占比 | > 80% |
| **Mean Norm** | Embedding 平均 L2 范数 | 0.5 - 2.0 |
| **Max Norm** | Embedding 最大 L2 范数 | < 10 |
| **Std Norm** | 范数标准差 | 0.3 - 1.0 |
| **P99 Norm** | 99 分位范数 | < 5 |
| **Grad Norm** | Embedding 梯度平均 L2 | 1e-4 - 1e-2 |

**异常诊断**:
- Mean Norm 持续上涨 → 学习率过大
- Max Norm 突增 → 某个 ID 异常
- Std Norm 急剧变小 → 模型坍缩
- Coverage 急剧下降 → 新 ID 没被学到

#### 80. **A/B 测试的统计显著性怎么算?**

**答**: 用 **Z 检验** 判断两组指标差异是否显著。

**公式**:
$$Z = \frac{p_A - p_B}{\sqrt{\frac{p_A(1-p_A)}{N_A} + \frac{p_B(1-p_B)}{N_B}}}$$

其中 $p_A, p_B$ 是两组的转化率, $N_A, N_B$ 是样本量。

**判定**:
- $|Z| > 1.96$: 95% 置信度显著
- $|Z| > 2.58$: 99% 置信度显著
- $|Z| < 1.96$: 不显著, 需更多流量或更长周期

**Python 实现**:
```python
def ab_test_significance(p_a, n_a, p_b, n_b):
    """A/B 测试 Z 检验."""
    from scipy.stats import norm
    pooled_p = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se = (pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b)) ** 0.5
    z = (p_a - p_b) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # 双侧检验
    return {'z': z, 'p_value': p_value, 'significant': p_value < 0.05}
```

**工业经验**:
- 流量 1%, 跑 7 天: 可检测 1% 相对提升
- 流量 5%, 跑 14 天: 可检测 0.5% 相对提升
- 转化稀疏 (CTR < 5%): 需更长周期

---

### E.2.3 在线服务类 (81-90)

#### 81. **TensorRT 加速原理?**

**答**: 4 个层次优化 ——
1. **算子融合 (Kernel Fusion)**: Conv+BN+ReLU 合并成一个 kernel, 减少 IO
2. **内核调优**: 选择最快的 CUDA 实现 (e.g., 不同的 GEMM 算法)
3. **精度校准**: INT8 量化 (但保持精度)
4. **静态 Shape**: 不用动态 shape, 用 padding 对齐

```python
import tensorrt as trt

# 简化流程: ONNX → TRT Engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16
config.max_workspace_size = 1 << 30  # 1GB

engine = builder.build_engine(network, config)
# 保存 engine 用于 serving
```

**效果**: 推理加速 2-10 倍, 显存减少 50%。

#### 82. **AOT vs JIT 编译?**

**答**:

| 维度 | AOT (Ahead-of-Time) | JIT (Just-in-Time) |
| :--- | :--- | :--- |
| 时机 | 训练后立即编译 | 推理时编译 |
| 文件 | `.pt` (TorchScript) | 内存中编译 |
| 启动 | 快 (无需编译) | 慢 (首次推理编译) |
| 灵活 | 弱 (结构固定) | 强 (动态图) |
| 适用 | 线上 serving | 调参/Demo |

```python
# AOT 示例
model = MyModel()
scripted = torch.jit.script(model)  # 或 torch.jit.trace(model, example_input)
scripted.save("model.pt")

# 加载
loaded = torch.jit.load("model.pt")
```

#### 83. **KV-Cache 是什么? 为什么省显存?**

**答**: 推理时缓存历史 K/V, 避免重算。

**无 KV-Cache** (每次都重算所有历史):
- 生成 N 个 token, 每次 O(N^2) attention
- 总 O(N^3) FLOPs

**有 KV-Cache** (缓存历史 K/V):
- 第 t 步只需算当前 token 的 Q 与历史的 K, V
- 总 O(N^2) FLOPs

```python
class KVCacheAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            # 拼接历史 KV
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # [B, H, T_total, D_h]
            v = torch.cat([v_cache, v], dim=2)

        # Attention
        attn = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.dim)

        return out, (k, v)  # 返回新的 KV 缓存
```

**节省**: 显存换时间, 7B 模型从 OOM 到能跑。

#### 84. **PagedAttention 为什么是 vLLM 的核心?**

**答**: 把 KV 分页存储, 减少碎片, 提升 batch 吞吐 10-20x。

**传统 KV Cache 问题**:
- 每个序列分配连续显存, 不同长度浪费大
- 长序列占满显存, 短序列浪费

**PagedAttention**:
- 把 KV 分成固定大小的"页" (类似虚拟内存)
- 不需要连续物理显存
- 显存利用率从 ~30% 提升到 ~95%

**效果**: vLLM 推理吞吐 10-20x 提升。

#### 85. **Flash Attention 为什么快?**

**答**: IO-Aware 分块, 减少 HBM 读写。

**传统 Attention**:
1. 把 Q, K, V 加载到 HBM
2. 计算 $S = QK^T$ (中间结果存 HBM, $O(T^2)$)
3. 计算 $P = \text{softmax}(S)$ (再存 HBM)
4. 计算 $O = PV$ (再存 HBM)

**Flash Attention**:
1. 把 Q, K, V 分块 (e.g., 128 一块)
2. 在 SRAM (快存) 里完成所有计算
3. 只把最终结果写回 HBM
4. 理论 FLOPs 不变, 但 IO 减少 10x

**内存复杂度**: $O(T^2) \to O(T)$ (用 tiling + recomputation)

#### 86. **Serving 服务的常见架构?**

**答**:

```text
Load Balancer (Nginx/Envoy)
    ↓
[Service A] [Service B] [Service C]
(召回)      (粗排)      (精排)
   ↓          ↓          ↓
Redis     Faiss      Feature Store
   ↓
模型仓库 (Porsche / HuggingFace)
```

**关键设计**:
- 无状态服务 (Stateless)
- 水平扩展 (多副本)
- 异步特征拉取
- 熔断降级 (Hystrix / Sentinel)

#### 87. **怎么监控线上模型?**

**答**:

| 维度 | 指标 | 工具 |
| :--- | :--- | :--- |
| **输入分布** | PSI / KS 检验 / 特征值分布 | EvidentlyAI / 自研 |
| **输出分布** | 预测值分布 / 类别占比 | Grafana |
| **业务指标** | CTR / 转化率 / 时长 | 自研 + 数据仓库 |
| **服务健康** | P50/P99 延迟 / QPS / 错误率 | Prometheus |
| **Embedding 漂移** | Embedding 范数 / PSI | 自研 |
| **A/B 效果** | 实验组 vs 对照组 AUC/业务指标 | 自研 |

**告警**: 关键指标超阈值 → PagerDuty → on-call 工程师。

#### 88. **A/B 测试的流量分桶怎么做?**

**答**:

**方法 1: 哈希分桶**:
```python
def get_bucket(user_id, n_buckets=100):
    """基于 user_id 哈希分桶, 保证同一用户每次分到同一桶."""
    return hash(user_id) % n_buckets

# 实验组 50%, 对照组 50%
if get_bucket(user_id) < 50:
    return 'experiment'
else:
    return 'control'
```

**方法 2: 用户 ID 末位**:
- 末位 0-49: 对照
- 末位 50-99: 实验

**注意**:
- **用户级分桶**: 同一用户始终在同一组 (防止"组污染")
- **流量均匀**: 每组样本量大致相当
- **稳定分桶**: 不因系统重启而变化

#### 89. **怎么设计离线评估集?**

**答**:

**时间切分**: 训练集 (前 7 天) / 验证集 (第 8 天前半天) / 测试集 (第 8 天下午半天)

**注意**:
- **不能随机 shuffle**: 否则数据穿越
- **覆盖完整周期**: 包括工作日 + 周末
- **分布一致**: 测试集与线上分布对齐

```python
def time_split(df, train_end, val_end):
    train = df[df['timestamp'] < train_end]
    val = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
    test = df[df['timestamp'] >= val_end]
    return train, val, test

# 例: 训练 6.1-6.7, 验证 6.8 上午, 测试 6.8 下午
train, val, test = time_split(df, '2026-06-08 00:00', '2026-06-08 12:00')
```

#### 90. **在线推理延迟突增怎么排查?**

**答**:

**排查清单**:
1. **QPS 升高**: 流量突增 → GPU 排队
2. **慢请求**: 某个请求处理时间异常
3. **缓存击穿**: Redis 大量 miss → PS 过载
4. **特征服务 GC**: Java 服务 STW
5. **模型加载**: 新模型首次推理慢
6. **下游依赖**: 数据库/上游服务延迟

**排查工具**:
- 火焰图 (Flame Graph) 看 CPU 瓶颈
- Py-Spy / nsys 看 Python/CUDA 耗时
- Trace (OpenTelemetry) 看端到端

**修复**:
- 限流 (Rate Limiting)
- 缓存预热
- 异步处理
- 模型量化

---

### E.2.4 前沿与拓展 (91-100)

#### 91. **World Model 在推荐中怎么用?**

**答**: 学用户行为的"环境动力学", 模拟未来状态。

**核心思想**:
- 推荐 = 序列决策 (用户状态 → 物品 → 反馈)
- 学一个"世界模型" $P(s_{t+1} | s_t, a_t)$ 模拟状态转移
- 用世界模型做 RL 训练, 比真实环境快 1000x

**架构 (DreamerV3 风格)**:
```text
真实轨迹 → Encoder → 隐状态 $z$
                            ↓
                    RSSM (Recurrent State-Space Model)
                            ↓
              World Model 输出: $z_{t+1}$, $r_t$
                            ↓
                  Policy 用世界模型训练
```

**工业案例**:
- 字节跳动: 用世界模型做"长视野规划" (e.g., 用户未来 7 天的兴趣)
- Meta: 探索用世界模型做"探索 vs 利用"决策

#### 92. **RLHF 在推荐中能直接用吗?**

**答**: 不能直接用,需要改造。

**原版 RLHF**:
- 人类对模型输出排序 (preference)
- 训练 Reward Model $r_\phi$
- 用 PPO 优化 policy

**推荐中的改造**:
- "人类偏好" → "用户反馈" (点击/时长/转化)
- 用这些反馈训练 Reward Model
- 但有"反馈偏差" (用户只评价看到的)

**实际方案**:
- **Implicit RLHF**: 用点击/时长作为 reward
- **DPO**: 直接优化偏好, 不需要 reward model
- **GRPO**: 组内归一化估计 advantage (DeepSeek 2024)

#### 93. **DPO 相比 PPO 的优势?**

**答**:

**PPO**: 4 个模型 (policy, reference, value, reward), 复杂, 不稳定
**DPO**: 2 个模型 (policy, reference), 简单, 稳定

**DPO 公式**:
$$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

其中 $y_w$ 是优选输出, $y_l$ 是劣选输出。

**优势**:
- 无需 reward model
- 训练更稳定
- 显存省 4x

**劣势**:
- 依赖成对偏好数据
- 不适合"长决策序列" (但推荐是短决策, 适合)

#### 94. **GRPO 和 PPO 的区别?**

**答**: GRPO (Group Relative Policy Optimization, DeepSeek 2024) 不需要 critic, 用组内归一化估计 advantage。

**PPO**:
$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
需要 value network $V$ 估计 baseline

**GRPO**:
- 对每个 prompt 采样一组输出 $\{y_1, ..., y_G\}$
- 用组内 reward 的均值/标准差归一化估计 advantage
$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

**优势**:
- 无 critic, 显存省一半
- 训练稳定 (组内比较, 减少方差)
- 适合"可采样多输出"的任务 (LLM 推荐正适合)

**工业实践**: 字节/阿里在 GR 推荐中已用 GRPO 替代 PPO。

#### 95. **推荐系统的多任务学习目标怎么设计?**

**答**: 4 种策略 ——
1. **硬参数共享**: 多任务共享 backbone, 不同 head
2. **软参数共享**: 每个任务独立 backbone, 加正则让参数接近
3. **MMoE/PLE**: 多专家共享 + 任务门控
4. **ESMM**: 任务间有因果关系 (e.g., CTR → CTCVR)

```python
class ESMM(nn.Module):
    """Entire Space Multi-task Model (阿里 2018)."""
    def __init__(self, n_features, dim=64):
        super().__init__()
        # 共享底层
        self.shared = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # CTR tower
        self.ctr_tower = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # CVR tower (输入是 CTR × hidden)
        self.cvr_tower = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        shared = self.shared(x)
        pctr = self.ctr_tower(shared)
        pcvr = self.cvr_tower(shared)
        pctcvr = pctr * pcvr  # CTCVR = P(click) × P(convert|click)
        return pctr, pctcvr
```

**优势**: 解决 CVR 样本稀疏问题 (用 CTR 任务的丰富样本辅助)。

#### 96. **怎么评估 LLM 推荐系统?**

**答**: 5 类指标 ——
1. **召回指标**: Recall@K, HitRate@K
2. **排序指标**: NDCG@K, MAP@K
3. **多样性**: Intra-List Diversity (ILD), Coverage
4. **新颖性**: Novelty (推荐物品平均流行度的倒数)
5. **LLM 特有**: 流畅度 (perplexity), 多样性 (Self-BLEU)

**离线评估陷阱**:
- LLM 推荐生成的是 token 序列, 不直接是 item
- 需要把 Semantic ID 解码回 item
- 解码错误会被错误计入 recall

```python
def evaluate_gr(generated_sids, ground_truth_items, sid_to_item):
    """
    generated_sids: [B, T] 生成的 Semantic ID 序列
    ground_truth_items: [B] 真实物品 ID
    sid_to_item: dict, Semantic ID → item ID
    """
    hit_count = 0
    for gen_sids, gt_item in zip(generated_sids, ground_truth_items):
        # 解码生成的 Semantic ID → 物品 ID
        for sid in gen_sids:
            sid_tuple = tuple(sid.cpu().tolist())
            if sid_tuple in sid_to_item:
                pred_item = sid_to_item[sid_tuple]
                if pred_item == gt_item:
                    hit_count += 1
                    break
    return hit_count / len(ground_truth_items)
```

#### 97. **RAG 在推荐中怎么用?**

**答**: 把"召回增强"和"上下文注入"结合。

**场景**: 用户问"推荐适合我的咖啡", LLM 需要先检索用户历史。

**RAG 流程**:
1. 用户输入 → embedding → 检索 Top-K 相关物品
2. 物品描述作为 context 喂给 LLM
3. LLM 生成推荐理由 + Top-N 推荐

```python
class RAGRecSystem:
    def __init__(self, retriever, llm):
        self.retriever = retriever  # Faiss / Embedding Search
        self.llm = llm              # Llama / Qwen

    def recommend(self, user_query, user_id, top_k=5):
        # 1. 检索相关物品
        user_emb = self.get_user_embedding(user_id)
        retrieved_items, scores = self.retriever.search(user_emb, k=20)

        # 2. 构造 prompt
        context = "\n".join([
            f"物品 {i+1}: {item['title']}, 类别: {item['category']}"
            for i, item in enumerate(retrieved_items[:5])
        ])
        prompt = f"""
        用户历史兴趣: {context}
        用户当前问题: {user_query}
        请基于用户历史, 推荐最相关的 5 个物品, 并解释理由。
        """

        # 3. LLM 生成
        response = self.llm.generate(prompt)
        return response, retrieved_items[:top_k]
```

#### 98. **推荐系统的因果推断怎么用?**

**答**: 解决"曝光偏差"和"评估偏差"。

**核心方法**:
- **IPS (Inverse Propensity Scoring)**: 用曝光概率加权无偏估计
- **Uplift Modeling**: 估计"干预效果", 用于营销
- **Counterfactual Reasoning**: 反事实推断"如果给用户看了 X, 会怎样"

```python
class IPSEstimator:
    """IPS 校正曝光偏差."""
    def __init__(self, propensity_model):
        self.propensity = propensity_model  # P(exposure | user, item)

    def unbiased_auc(self, scores, labels, propensities):
        """无偏 AUC 估计."""
        n = len(scores)
        weighted_pos = 0
        weighted_neg = 0
        correct_pairs = 0

        for i in range(n):
            if labels[i] == 1:
                w = 1.0 / propensities[i]  # IPS 权重
                weighted_pos += w
                # 与所有负样本配对
                for j in range(n):
                    if labels[j] == 0:
                        if scores[i] > scores[j]:
                            wj = 1.0 / propensities[j]
                            correct_pairs += w * wj
            else:
                w = 1.0 / propensities[i]
                weighted_neg += w

        return correct_pairs / (weighted_pos * weighted_neg)
```

**工业实践**: 阿里、字节在"营销活动"评估中用 Uplift 建模。

#### 99. **OneRec / RecGPT 端到端 LLM 推荐怎么做?**

**答**: 把推荐系统统一到 LLM 框架。

**OneRec 架构** (字节 2025):
```text
用户行为 → 物品 Embedding → RQ-VAE → Semantic ID
                                       ↓
                              用户历史 → S-ID 序列
                                       ↓
                              LLM (Encoder-Decoder)
                                       ↓
                              Beam Search 生成 S-ID
                                       ↓
                              S-ID → 物品 ID → 推荐结果
```

**核心创新**:
1. **统一召回 + 粗排 + 精排**: 1 个 LLM 干所有事
2. **课程学习**: 从短序列开始, 逐步加长
3. **Beam Search 多样化**: 不只取 top-1, 强制覆盖多兴趣

**线上效果**:
- 主 feed CTR +5.7%
- 长尾内容曝光 +28%
- 但推理延迟 +3x (需优化)

#### 100. **未来 3-5 年推荐系统的趋势?**

**答**: 5 大方向 ——
1. **Foundation Model for Rec**: 推荐领域的"GPT", 预训练 + 微调
2. **LLM Agent**: 用 Agent 做多步推理推荐 (e.g., "帮我找适合生日聚会的餐厅")
3. **多模态深度融合**: 文本/图像/视频/音频统一表征
4. **世界模型 + RL**: 用世界模型做长视野规划, RL 优化长期 reward
5. **端侧 + 隐私**: Federated Learning, On-Device 推荐

**关键挑战**:
- **推理成本**: LLM 推荐是 DLRM 的 10-100x
- **冷启动**: 新用户/新物品仍难处理
- **评估**: 缺乏统一的 benchmark
- **工程**: 大模型 serving 的复杂度

---

## E.3 面试实战: STAR 法则答题模板

**STAR = Situation + Task + Action + Result**

### 模板示例: "你做过最有挑战的推荐项目?"

**S (Situation)**:
> 我们团队负责淘宝首页推荐, 日均 10 亿次请求, 模型是 DLRM, 业务指标停滞 6 个月。

**T (Task)**:
> 主管让我主导一个升级项目, 目标 CTR +5%, 延迟 < 50ms。

**A (Action)**:
> 1. 分析: 数据分析发现长尾物品曝光不足, 推荐多样性差
> 2. 实验: 尝试 DIN, 多任务 MMoE, MAB 多样性
> 3. 上线: DIN + 探索流量 + 重排多样性
> 4. 工程: TorchRec 分布式训练, TensorRT 推理优化

**R (Result)**:
> - CTR +6.2% (达成目标)
> - 长尾物品曝光 +35%
> - P99 延迟 42ms (达标)
> - 项目获得 Q3 优秀项目奖

### 模板示例: "训练 Loss 不收敛怎么排查?"

**S**: 训练 3 亿样本的 DLRM 模型, loss 在 0.5 附近不下降
**T**: 1 天内定位问题
**A**:
1. 检查数据: 发现 5% 样本 label 错误
2. 检查梯度: Embedding 梯度范数 = 0 (稀疏 ID 没命中)
3. 检查学习率: 减半后正常收敛
**R**: 4 小时定位, 6 小时修复, 模型上线 AUC 涨 1.2%

---

## E.4 反问环节 (候选人问面试官)

### 推荐系统工程师面试必问 10 题

1. **业务规模**: 日活 DAU / QPS / 团队规模?
2. **技术栈**: 主要用什么框架? PyTorch / TensorFlow / 自研?
3. **模型架构**: 主力模型是? DLRM / DIN / Transformer / GR?
4. **数据规模**: 训练数据多大? 每天更新吗?
5. **基础设施**: GPU 多少? 推理用什么? TensorRT / vLLM?
6. **团队方向**: 召回 / 排序 / 特征 / 工程?
7. **业务挑战**: 目前最大的技术挑战是?
8. **晋升路径**: P5 → P6 → P7 的标准是?
9. **学习资源**: 团队有内部培训吗? 推荐什么书/论文?
10. **工作节奏**: 996? 加班多吗?

---

## E.5 简历亮点与项目包装

### 5 个推荐系统简历项目模板

#### 项目 1: 召回系统从 DLRM 升级到 GR

**描述**:
> 主导将召回系统从 DLRM 升级到基于 Semantic ID 的生成式推荐, 召回 Recall@100 从 24% 提升到 32%, 长尾物品曝光占比从 8% 提升到 13%。

**技术栈**:
- RQ-VAE 训练 Semantic ID (PyTorch)
- Transformer Decoder 生成 (Hugging Face Transformers)
- 分布式训练 (TorchRec + DDP, 8 卡 A100)
- 在线 serving (vLLM + TensorRT)

**贡献**:
- 设计并实现 RQ-VAE 训练 pipeline
- 实现基于 Beam Search 的推理
- 优化推理延迟从 80ms 到 25ms (TensorRT)

#### 项目 2: 多任务精排模型优化

**描述**:
> 在阿里妈妈 CTR/CVR/CTCVR 多任务场景, 用 ESMM + PLE 替代 MMoE, AUC 涨 1.8%, CTCVR 涨 4.2%。

**贡献**:
- 解决 MMoE 跷跷板现象
- 设计 ESMM 链路 (CTR → CTCVR)
- 在线 A/B 测试, 流量 10%, 14 天显著

#### 项目 3: Embedding 系统从 50GB 扩到 500GB

**描述**:
> 业务增长导致 Embedding 表从 50GB 涨到 500GB, 单机训练不可行。设计分布式 Embedding 训练系统, 支持 8 卡 A100, 训练时长从 7 天缩到 18 小时。

**贡献**:
- 引入 TorchRec EmbeddingBagCollection
- 实现 ZeRO Stage 2 + 混合精度
- 设计多级 Embedding 缓存 (GPU HBM / CPU / SSD)

#### 项目 4: 冷启动新物品 CTR 提升

**描述**:
> 新物品 CTR 0.5%, 老物品 5%。设计 Prior Matching 冷启动方案, 新物品 CTR 提升到 2.3%, 1h 留存 +20pp。

**贡献**:
- 用侧信息 Embedding 平均初始化
- 设计专门冷启动分支
- 在线 A/B 验证

#### 项目 5: 在线推理延迟优化

**描述**:
> 推荐服务 P99 延迟从 120ms 优化到 30ms, 满足线上要求。

**贡献**:
- 量化 FP32 → INT8 (延迟减半)
- KV-Cache (GR 模型)
- 异步特征拉取 (Redis Pipeline)
- 模型蒸馏 (BERT-12 → BERT-4)

---

## E.6 算法岗 Offer 谈判指南

### 薪资构成

| 部分 | 占比 | 议价空间 |
| :--- | :--- | :--- |
| 基本工资 | 60-70% | 10-20% |
| 签字费 | 5-10% | 50-100% |
| 股票 | 20-30% | 30-50% |
| 绩效奖金 | 5-10% | 固定 |

### 议价技巧

1. **不要先报数字**: 让 HR 先说范围
2. **强调稀缺技能**: GR, VLA, 世界模型 (2026 仍稀缺)
3. **多家 offer 竞争**: 同时拿 2-3 个 offer
4. **关注总包**: 不要只看月薪

### 反向 offer 拒绝示例

> 感谢贵司的 offer, 我非常认可团队的技术方向。但经过综合考虑, 我决定接受另一家公司的 offer (总包高 30%, 业务更匹配我的兴趣)。希望未来有机会合作。

---

## E.7 推荐系统工程师成长路径

### 5 年成长时间线

```text
P5 (0-2 年): 工程师
  - 熟悉 PyTorch / Spark / Kafka
  - 能在指导下完成模块
  - 掌握 DLRM / DIN 主流架构

P6 (2-5 年): 高级工程师
  - 独立负责 1 个子方向
  - 理解业务 + 技术深度
  - 能跨团队协作

P7 (5-8 年): 资深工程师
  - 主导 1 个产品方向
  - 有 1-2 篇顶会论文
  - 团队 leader / mentor

P8 (8+ 年): 专家 / 总监
  - 行业影响力
  - 战略眼光
  - 培养下一代
```

### 学习资源清单 (按阶段)

**入门 (P5)**:
- 《深度学习推荐系统》(王喆)
- RecBole 库跑通主流模型
- 1-2 个 Kaggle 比赛

**进阶 (P6)**:
- 《推荐系统: 原理与实践》
- 精读 20 篇顶会论文 (RecSys, SIGIR, KDD)
- 1 个完整业务项目

**资深 (P7)**:
- 《Deep Learning for Matching / Ranking》(阿里)
- 1-2 篇顶会论文发表
- 主导 1 个新方向

**专家 (P8)**:
- 行业视野 + 战略眼光
- 公开博客 / 演讲 / 开源项目

---

## E.8 总结: 100 道面试题的"知识地图"

```text
                    推荐系统面试 100 题
                           |
       ┌────────────┬──────┴──────┬────────────┐
       |            |             |            |
    基础(15)    模型(15)    工程(20)    前沿(15)    深度专题(35)
       |            |             |            |
    AUC/BPR    DLRM/DIN    监控/部署   LLM/GR     细节深化
    Embedding  SIM/SASRec  量化/压缩   因果/RLHF  量化蒸馏
    负采样     多任务       Serving     RAG       KV-Cache
    冷启动     双塔/单塔   A/B测试     World Model   PagedAttn
    Transformer HSTU/GR    性能优化    Foundation
```

**学习建议**:
1. **先广后深**: 把 100 题都过一遍, 找到薄弱点
2. **结合项目**: 每个问题想想"我项目里怎么用"
3. **写代码**: 重点问题都用 PyTorch 实现一遍
4. **讲给别人**: 教别人是最高效的学习

---

> **附录 E 最终版** | 100 道高频面试题 + STAR 答题模板 + 反问环节 + 简历包装 + Offer 谈判 + 成长路径
> **实际大小**: ~85KB | 总 8 节 (E.0 - E.8)
> **最后更新**: 2026-07-15
> **风格**: 实战 + 深度 + 可直接面试使用

## E.9 模拟面试: 真实对话演练

### E.9.1 模拟面试 1: 算法基础 (中级)

**面试官**: 你能讲一下 AUC 的物理意义吗? 为什么推荐系统喜欢用 AUC 而不是准确率?

**候选人**:
> AUC (Area Under ROC Curve) 的物理意义是: 随机从数据集中抽一对正负样本, 模型把正样本排在负样本前面的概率。
>
> 公式上, AUC 等于所有正负样本对中, "正样本分数 > 负样本分数" 的比例:
>
> $$\text{AUC} = \frac{1}{|P| \cdot |N|} \sum_{i \in P} \sum_{j \in N} \mathbb{1}[s_i > s_j]$$
>
> 推荐系统喜欢 AUC 的原因有 3 个:
>
> 1. **不依赖阈值**: AUC 衡量的是"排序能力", 与具体阈值无关, 适合不同阈值策略
> 2. **适合样本不平衡**: 推荐场景正样本 (点击) 通常 < 5%, 准确率会被负样本主导, AUC 不受影响
> 3. **物理可解释**: AUC = 0.7 表示模型 70% 的概率能把正样本排到负样本前面, 直观
>
> **什么时候不用 AUC**:
> - 关注 top-K 召回: 用 Recall@K, NDCG@K
> - 关注绝对预测值: 用 LogLoss, Calibration
> - 关注排序质量: 用 NDCG/MAP

**面试官**: BPR Loss 公式是什么? 为什么比 Cross-Entropy 更适合推荐?

**候选人**:
> BPR (Bayesian Personalized Ranking) Loss 假设用户对正样本的偏好大于负样本:
>
> $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j) \in D} \log \sigma(\hat{x}_{ui} - \hat{x}_{uj})$$
>
> 其中 $\hat{x}_{ui}$ 是用户 $u$ 对物品 $i$ 的预测分数, $\sigma$ 是 sigmoid 函数。
>
> BPR 比 Cross-Entropy 更适合推荐的原因:
>
> 1. **Pairwise vs Pointwise**:
>    - BPR 学习"排序" (正样本 > 负样本)
>    - CE 学习"分类" (正样本 = 1, 负样本 = 0)
>    - 推荐场景更关注排序 (top-K 推荐), 所以 BPR 更合适
>
> 2. **自动 focus 难样本**:
>    - 当 $\hat{x}_{ui} - \hat{x}_{uj}$ 小时 (模型分不清), $\sigma(-\Delta) \approx 0.5$, 梯度大
>    - 当差值大时 (模型分得清), 梯度小
>    - CE 对所有样本一视同仁
>
> 3. **负样本利用**: BPR 显式使用负样本, CE 隐式 (0/1 label)
>
> **劣势**: BPR 只适合隐式反馈 (点击/购买), 不适合评分预测 (1-5 星), 此时用 MSE。

**面试官**: Embedding 维度怎么选? 为什么不是越大越好?

**候选人**:
> 经验公式是 Google 提出的 $D_{\text{suggest}} = \min(D_{\max}, 4 \cdot \lfloor V^{1/4} \rfloor)$, 其中 $V$ 是词表大小。
>
> 理论上, 维度太小 (< 8) 会欠拟合, 维度太大 (> 256) 会:
>
> 1. **过拟合**: 推荐数据噪声大于信号, 大维度学不到有用信息
> 2. **训练慢**: 矩阵乘 FLOPs 与 $D$ 线性增长
> 3. **显存贵**: Embedding 表 $V \times D \times 4$ bytes 随 $D$ 线性增长
>
> 我自己在 MovieLens 上做过实验, 维度从 32 涨到 256, AUC 几乎不变 (0.8267 → 0.8268), 但训练时长从 7 分钟涨到 38 分钟。
>
> 工业实践:
> - $V < 10^5$: 16-32 维
> - $V$ 在 $10^5 \sim 10^8$: 32-64 维
> - $V > 10^8$: 64-128 维, 配合 Hash Embedding

**面试官**: DIN 的 Attention 和 Transformer Self-Attention 有什么区别?

**候选人**:
> 关键区别在于 **Query 的来源**:
>
> - **DIN (Target Attention)**: Query 是 target item (要预测的物品), Key/Value 是用户历史
>   - 输入序列是用户历史 $h_1, h_2, ..., h_T$, target 是 $t$
>   - 计算的是 $P(h_j | t)$, 即"对当前 target, 每个历史的相关度"
>   - 这是 **cross attention** 的一种 (query 和 key 来自不同来源)
>
> - **Transformer Self-Attention**: Query/Key/Value 都来自同一序列
>   - 计算的是 $P(h_j | h_1, ..., h_T)$, 即"序列内部的相互关系"
>   - 这是 **self attention**
>
> DIN 用 Local Activation Unit (LAU) 计算 attention:
>
> $$e(h, t) = \text{MLP}([h; t; h \odot t; h-t])$$
>
> 输入 4 个特征 (历史/目标/元素积/差) 来捕捉相似度。
>
> Self-Attention 用 $QK^T / \sqrt{D}$ 计算, 更通用但可能不够"target-aware"。
>
> 实际选择:
> - 单一序列内部关系 (BERT, SASRec): Self-Attention
> - 用户历史 vs target item (DIN): Target Attention

---

### E.9.2 模拟面试 2: 工程实践 (高级)

**面试官**: 你做的最大的 Embedding 优化是什么? 效果怎样?

**候选人**: (使用 STAR 法则)
> **S (Situation)**: 在字节跳动做推荐召回, Embedding 表 50GB, 单机 A100 装不下, 训练时长 7 天。
>
> **T (Task)**: 主导 Embedding 系统从单机升级到分布式, 训练时长缩到 18 小时。
>
> **A (Action)**:
> 1. **分析瓶颈**: 用 NVIDIA Nsight 看 GPU 利用率, 发现 Embedding 加载占 60% 时间
> 2. **架构升级**:
>    - 引入 TorchRec EmbeddingBagCollection, Embedding 按行分片到 8 卡
>    - 每卡只存 6.25GB, 显存宽裕
> 3. **优化训练**:
>    - ZeRO Stage 2 分片 optimizer state
>    - 混合精度 (BF16 计算, FP32 master)
>    - Gradient Accumulation 4 步
> 4. **优化数据加载**:
>    - 序列特征 Dynamic Padding, 显存省 30%
>    - Bucket Sampler 按长度分桶
> 5. **监控**:
>    - 加 WandB 监控 Embedding 范数分布
>    - 单 ID 异常检测
>
> **R (Result)**:
> - 训练时长: 7 天 → 18 小时 (**4x 加速**)
> - GPU 利用率: 35% → 75%
> - 模型 AUC: 涨 0.4% (因为有空间用更大 batch)
> - 这个项目后来被复制到 3 个业务线

**面试官**: Embedding 训练时, 怎么保证训推一致?

**候选人**:
> 这是工业推荐的核心难题: 离线训练看到的特征分布, 必须与在线 serving 一致, 否则 AUC 涨、线上反而掉。
>
> 我的方案分 4 层:
>
> 1. **特征定义统一**:
>    - 用统一特征 SDK (Feast/Tecton), 一次定义训练/在线都用
>    - 字段名、类型、缺失值填充策略都集中管理
>
> 2. **特征计算统一**:
>    - 离线用 Spark SQL, 在线用 Flink SQL
>    - 同一份 SQL 脚本, 不同 runtime
>
> 3. **Embedding 一致**:
>    - 同一份 ID → row 映射 (MurmurHash 固定种子)
>    - 同一份 Embedding 表 (从 OSS 加载, 不在线训练)
>
> 4. **监控漂移**:
>    - PSI (Population Stability Index) > 0.25 报警
>    - KS 检验 p < 0.01 报警
>    - 在线 A/B 测试是终极验证
>
> **具体工具**:
> - 训练时记录每条样本的特征值
> - 在线服务把请求的特征值也记录 (sampling 1%)
> - 每天跑一次 PSI / KS 对比, 漂移超阈值触发 retrain

**面试官**: 你遇到过什么 OOM 问题? 怎么排查?

**候选人**:
> 训练一个 7B 参数的 GR 模型时, 8 卡 A100 80GB 全部 OOM。
>
> **排查步骤**:
> 1. `nvidia-smi` 看每卡显存占用: 80GB/80GB, 满
> 2. PyTorch Profiler 看具体在哪个 forward 占用最大
> 3. 发现 Embedding 表 (50GB) + Adam state (100GB, FP32) + activation (40GB) = 190GB / 卡
>
> **优化方案** (分层):
> 1. **Adam state 减半**:
>    - 改用 BF16 + FP32 master, Adam state 从 FP32 改 BF16 → 省 50%
> 2. **Adam state 分片**:
>    - DeepSpeed ZeRO Stage 2, 8 卡分片 → 省 87.5%
> 3. **Activation Checkpointing**:
>    - 不保存中间 activation, 反向时重算 → 省 30%
> 4. **CPU Offload**:
>    - 不常用的参数卸载到 CPU 内存 → 省 40%
>
> **结果**: 8 卡 A100 从 OOM → 用了 50GB/卡, 还有空间调大 batch。

---

### E.9.3 模拟面试 3: 前沿探索 (资深)

**面试官**: 你怎么看待生成式推荐 (GR) 的未来? 3-5 年会替代判别式吗?

**候选人**:
> 我的判断: **不是替代, 是分层共存**。
>
> **生成式推荐的优势**:
> 1. **统一架构**: 召回 + 粗排 + 精排一个 LLM, 简化工程
> 2. **知识迁移**: NLP 的进展 (RLHF, CoT, Agent) 可以平移到推荐
> 3. **冷启动友好**: 新物品生成 Semantic ID 即可
> 4. **多模态统一**: 文本/图像统一表征
>
> **生成式推荐的劣势**:
> 1. **推理成本**: LLM 推理是 DLRM 的 10-100 倍, 工业部署难
> 2. **训练数据**: 需要 PB 级数据, 中小厂做不了
> 3. **延迟**: 自回归生成几十个 token, 延迟 50-100ms, 难满足 < 30ms 要求
> 4. **冷启动**: 仍需要 RQ-VAE 训练, 工程复杂
>
> **我的预测** (3-5 年):
> - 大厂 (Meta, 字节, Google): 召回用 GR, 精排仍用判别式
> - 中厂: 召回仍是 DLRM + DIN, GR 是实验方向
> - 小厂: 全部判别式, GR 成本太高
>
> **关键转折点**: 当 GR 推理延迟能压到 10ms 以内, 会真正替代 DLRM。
>
> 可能的突破:
> - 专用芯片 (Groq, Cerebras)
> - 模型蒸馏 (大 GR 训小 GR)
> - 投机式解码 (Speculative Decoding)

**面试官**: 世界模型在推荐中能用吗? 你有什么具体方案?

**候选人**:
> 世界模型在推荐中的核心价值: **低成本仿真**。
>
> 推荐系统本质是序列决策:
> - 状态 $s_t$: 用户当前兴趣
> - 动作 $a_t$: 推荐物品
> - 奖励 $r_t$: 点击/时长/转化
> - 转移 $P(s_{t+1} | s_t, a_t)$: 用户兴趣演化
>
> 真实环境跑 1 次推荐要 100ms (含用户反馈), 世界模型跑 1 次只要 1ms。
>
> **具体方案** (借鉴 DreamerV3):
>
> 1. **训练数据**: 用户历史轨迹 (state, action, reward, next_state)
> 2. **世界模型架构**:
>    - Encoder: 把 user/item embedding 编码到隐空间 $z$
>    - RSSM: $z_{t+1} = f(z_t, a_t, h_{t-1})$
>    - Reward Predictor: $\hat{r}_t = g(z_t, a_t)$
> 3. **训练目标** (ELBO):
>    $$\mathcal{L} = \mathbb{E}_{q}[\log p(x_t|z_t) + \log p(r_t|z_t) - \text{KL}(q(z_t|h_t) \| p(z_t|z_{t-1}, a_{t-1}))]$$
> 4. **应用**:
>    - **长视野规划**: 在世界模型里 roll out 100 步, 选最优策略
>    - **探索 vs 利用**: 用世界模型模拟"探索新物品"的结果
>    - **RL 训练环境**: 替代真实环境, 训练 RL agent
>
> **挑战**:
> 1. **世界模型的精度**: 用户行为高度随机, 世界模型学不到精确转移
> 2. **数据需求**: 需要长期轨迹数据
> 3. **冷启动**: 新用户没历史, 世界模型预测不准
>
> **我的判断**: 世界模型在推荐中**目前仍偏研究**, 但 5 年内可能成为"长视野规划"的关键技术。
>
> 适合场景: 短视频 (高探索性), 电商促销 (长决策周期)

---

## E.10 总结与寄语

### 给求职者的话

> **简历**: 突出"业务结果 + 技术深度", 不要只列技术名词
> **面试**: 80% 项目 + 20% 八股, 用 STAR 法则讲项目
> **反问**: 问"业务挑战"和"团队方向", 显示深度思考
> **谈判**: 拿多个 offer, 强调稀缺技能 (GR, VLA, 世界模型)

### 给在职工程师的话

> **深度优先**: 选 1-2 个方向深耕, 不要追所有新东西
> **工程能力**: 80% 工业级 ML 是工程问题 (数据, 部署, 监控)
> **业务理解**: 不懂业务的算法工程师是"调参侠"
> **持续输出**: 写博客 / 做开源 / 分享, 强化自己的理解

### 给自学者的话

> **不追新**: Transformer 是 2017 提出的, 现在仍是主流
> **动手写**: 看完教程一定要自己实现一遍
> **看代码**: RecBole, TorchRec 都是工业级实现, 比博客准
> **找同伴**: 学习群 / 朋友互相讲, 教别人是最好的学习

---

> **附录 E 完整版** | 100 道高频面试题 + 6 类面试场景 + STAR 答题模板 + 反问环节 + 简历包装 + Offer 谈判 + 成长路径 + 模拟面试演练
> **总字数**: ~80,000 字
> **代码示例**: ~50 个
> **公式**: ~80 个 LaTeX
> **面试题**: 100 道 (含原 50 题 + 新增 50 题)
> **最后更新**: 2026-07-15
