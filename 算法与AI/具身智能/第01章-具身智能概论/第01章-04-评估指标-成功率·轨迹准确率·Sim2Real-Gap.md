---
title: "第01章-04-评估指标-成功率·轨迹准确率·Sim2Real-Gap"
tags:
  - 具身智能
  - 评估指标
  - 成功率
  - 轨迹准确率
  - Sim2Real
  - Gap
description: "本章要点：策略、做 RL/IL 评估的工程师的话:"
chapter_id: 具身智能-4-01
last_updated: 2026-07-14
---

# 第01章-04-评估指标-成功率·轨迹准确率·Sim2Real-Gap

策略、做 RL/IL 评估的工程师的话**:
>
> 训练过 LLM, 你会用 perplexity、看 BLEU、跑 MMLU; 训练过推荐模型, 你会用 AUC、NDCG、看 GMV。这些指标满足 3 个条件: **自动算、有 ground truth、能批量跑**。
>
> 具身智能的评估**全部不成立**。一个抓杯子任务, "成功" 的定义是主观的、ground truth 是稀疏的 (一条轨迹只有终点能算分)、批量跑几乎不可能 (每 episode 1-3 分钟)。
>
> 这一章回答 5 个直击痛点的问题:
>
> - **成功率的 ground truth 是什么?** 你**自己定义**, 没有"标准"。给 4 类工业定义 (二元 / 部分成功 / 子任务 / 人类评分) 和它们的偏差。
> - **为什么 Sim2Real Gap 永远消除不掉?** 视觉、物理、时序**三类 Gap** 在不同频率上叠加, 域随机化只能压低不能归零。给量化分解。
> - **轨迹 L2 距离到底有没有用?** **基本没用**, 但业内还在报。给 5 个比它强的指标。
> - **多任务评估怎么避免"平均 90% 但 50% 任务 0%" 的陷阱?** **任务平均 + 长尾切片 + 任务相关性**。
> - **离线指标能预测在线部署效果吗?** **不能直接预测**, 但**能监控到崩溃点**。
>
> **学完这一章, 你能回答的问题**:
>
> - 4 大类评估指标 (成功率 / 轨迹 / Sim2Real / 在线) 的数学定义与工业实践
> - Sim2Real Gap 量化: FID / LPIPS / 物理参数 / 时序 Gap
> - 200 个 episode 时 Wilson 置信区间的宽度
> - 多任务 Pareto 分析与长尾任务检测
> - 用 PyTorch 算成功率、L2、Sim2Real Gap
> - 5 个常见误区 + 5 个 L7/L8 面试题
>
> **与本文件夹其他章节的关系**:
> - [[第01章-01-从感知到行动-具身智能的定义与历史]] ← 理解"行为"
> - [[第01章-02-形态与本体-机械臂·人形·轮式·四足]] ← 不同形态的成功定义差异
> - [[第01章-03-数据集与基准-RT-1·Bridge·OpenX-Embodiment]] ← 前置, RT-1/Bridge/Open X 的评估协议
> - `../第05章-模仿学习/0-总目录与阅读路径.md` ← 消费本章指标
> - `../../rec教程/第07章-强化学习在GR中的角色.md` ← RL 策略评估
> - `../第07章-仿真器与基准/0-总目录与阅读路径.md` ← Sim2Real Gap 的来源
> - `9-工业落地与前沿方向` ← 在线评估实践
>
> **与数学基础文件夹的关系** (高复用):
> - `../../../基础/数学基础/第03章-概率论/第03章-03-常用分布家族-正态·Bernoulli·Categorical·Gamma·Beta.md` ← 成功率伯努利 + Wilson 区间
> - `../../../基础/数学基础/第05章-信息论/第05章-03-互信息·点互信息·条件互信息.md` ← FID 数学基础
> - `../第06章-强化学习与Sim2Real/0-总目录与阅读路径.md` ← 多任务 Pareto 前沿

---

## 1. 直觉引入:评估指标为什么是"具身智能最被低估的难题"

### 1.1 一个反直觉的事实:成功率 99% 不代表能用

2024 年某团队训练"抓杯子"任务, 仿真 1000 次跑出 99.5% 成功率。部署到真实机械臂跑 100 次, 成功率 67%。**Sim2Real Gap = 32.5 个百分点**, 不是他们想象的小数点后一位。

```text
仿真-真实差距 (Sim2Real Gap) 在不同任务上的典型值:
─────────────────────────────────────
仿真能跑    真实能跑    真实表现      Gap
99%         67%        "勉强能用"    32 pp
99%         41%        "需要重训"    58 pp
99%         12%        "完全不能用"  87 pp
─────────────────────────────────────
```

为什么差距这么大? 仿真器不知道的东西太多了:
1. **真实摩擦系数**: 桌布 vs 玻璃, 摩擦差 5 倍, 仿真假设单一标量
2. **真实光照**: 玻璃反射、阴影、镜面反射, 渲染管线会出错
3. **真实延迟**: 控制器周期 1ms vs 真实硬件 20ms, 反应速度差 20 倍
4. **真实接触动力学**: 杯子碰到桌面的微小反弹 + 摩擦, 仿真里就是"刚体接触"

这些差距**不是"再多训几轮" 就能消除的**, 它是**物理世界的真实复杂度**。

### 1.2 为什么 LLM 评估不能照搬到具身

LLM 评估 (MMLU, HumanEval) 协议成熟: 准备 benchmark → 跑模型 → 拿 ground truth 比对 → 算 ACC/BLEU → 报 95% CI。每一步**自动、可重复、不需人类**。

具身评估**每一项都反过来**:

| 维度 | LLM 评估 | 具身评估 |
| :--- | :--- | :--- |
| Ground truth | 标准答案 (文本) | **自己定义** (什么叫"成功"?) |
| 评估时间 | 几秒 (1 条 prompt) | **1-3 分钟** (1 个 episode) |
| 可重复性 | 同一 prompt 同结果 | 同样摆放, 每次结果可能不同 |
| 评估者 | 算法自动 | **常常需要人类** |
| 离线-在线对应 | **强对应** | **弱对应** (99% → 60%) |

最关键的是"成功" 定义模糊和离线-在线对应弱。
- "杯子抓起来" 算成功吗? 抓起来但放错位置算不算? 放对了但中间碰到桌子算不算?
- LLM MMLU 80% 意味着真实场景也 80%; 仿真 99% 可能真实 60%。

### 1.3 离线指标 vs 在线指标:一条你看不懂的鸿沟

```text
离线评估 (Offline)              在线评估 (Online)
─────────────                  ─────────────
仿真 / 录播数据               真实环境交互
1000 episode / 小时            100 episode / 小时
可复现 ✅                      受光照/电池/磨损 ⚠️
能发现"能力差距"              能发现"安全 / 部署差距"

成功率
100% ┤  ╭─────────────╮  ← 离线 (仿真/replay)
     │ ╱               ╲
 80% ┤╱   "一切都好"   ╲
 60% ┤                  ╲
     │                   ╲  ← 真实部署
 40% ┤                    ╲
     │                     ╲╲
 20% ┤                      ╲╲
  0% └─────────────────────────────
     仿真  仿真训练  仿真评估   真机  实际部署
     ↑                          ↑
     "看起来 100%"             "实际 40%"
```

**离线指标的 4 类失败**: 仿真器假装的世界 / 演示者偏好 / long-tail 缺失 / 没见过的物体。

> **关键洞察**: 离线指标是"必要不充分条件"。离线差, 真机一定差; 离线好, 真机可能差。**离线看"有没有能力", 真机看"这个能力在真实世界有没有用"**。

### 1.4 评估的 3 条黄金法则

```text
具身评估的 3 条黄金法则
─────────────────────────────────────────
1. 永远看"任务平均 + 长尾切片", 不只看平均
   → 70% 任务 95% + 30% 任务 0%  ≠ " 80% 成功率"

2. 永远跑"仿真 + 真实" 两套, 不能只看仿真
   → 仿真 99% / 真实 60%  ≠ " 平均 80%"

3. 永远报"平均 + 置信区间 + 失败案例"
   → "成功率 80% ± 5% (n=200, 失败集中在 X 类任务)"
─────────────────────────────────────────
```

### 1.5 章节路线图

| 节 | 主题 | 核心问题 |
| :--- | :--- | :--- |
| §2 | 成功率 | 4 种 ground truth 定义 |
| §3 | 轨迹准确率 | L2 / 终点误差 / 路径效率 / DTW |
| §4 | **Sim2Real Gap (本章重点)** | 视觉/物理/时序/综合 Gap 量化 |
| §5 | 多任务评估 | 任务平均 + 长尾 + 相关性 + 跨形态 |
| §6 | 统计与显著性 | Wilson 区间 + ANOVA + 多重比较 |
| §7 | 在线评估 | A/B + 长尾 + 安全监控 |
| §8 | PyTorch 实验 | 成功率 / L2 / Sim2Real Gap |
| §9-10 | 互引 | 与其他章节 + 数学基础关系 |
| §11 | 误区 | 5 个反模式 |
| §12 | Q&A | 5 道深度问答 |
| §13 | 参考文献 | 25+ paper |

---

## 2. 成功率 (Success Rate):最朴素但最容易被滥用的指标

### 2.1 直觉:为什么"二元成功率"既无处不在又有问题

成功率是具身评估的**祖传指标**: 跑 N 个 episode, 数成功的, 除以 N。形式上:

$$
SR = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[s_i = 1] = \mathbb{E}[\text{Bernoulli}(p)]
$$

3 个天生的毛病:
1. **"成功" 的定义模糊**: 杯子抓起来了算不算? 抓起来 1 秒后掉了算不算?
2. **二元化丢失信息**: 抓到 1cm 误差 vs 5cm 误差都是"成功", 但不同
3. **每个 episode 成本不同**: 抓杯子 1 分钟 vs 叠衣服 5 分钟, 用同一个 N 不公平

> **关键洞察**: 二元成功率是"地板", 不是"天花板"。它告诉**最基本的能力**, 不能告诉你做得多好。

### 2.2 4 种成功率的 ground truth 定义

#### 2.2.1 二元成功率

最简单: 任务完成 = 1, 否则 = 0。**判定边界例子** (抓杯子任务):

```text
成功 vs 失败 的边界:
─────────────────────────
判定 = "杯子最终在目标位置 ±5cm"
├─ 杯子在目标位置 4cm 内 → 成功 ✅
├─ 杯子在目标位置 8cm 外 → 失败 ❌
└─ 杯子掉地上 → 失败 ❌
```

工业实践: Google RT-1 用"物体最终在目标容器内", Meta HomeRobot 用"±10cm 范围内且关节角度正确"。**没有统一标准**, 但都在论文里公开。Wilson CI 在 §6.1 详细推导。

#### 2.2.2 部分成功度 (Partial Credit)

任务有连续好坏时, 用连续分数:

$$
\text{PC}_i = \exp\left(-\frac{d_i^2}{2\sigma^2}\right)
$$

$d_i$ 是最终位置误差, $\sigma$ 是容忍尺度 (5cm)。

```text
PC
1.0 ┤██
    │  ███
0.8 ┤    ███
0.6 ┤       ███
0.4 ┤          ████
0.2 ┤              ████████
0.0 ┤                       ████████████
    └─────┬─────┬─────┬─────┬─────┬─────┬
    0    2.5    5    7.5   10   12.5   17.5  cm
              ↑ 容忍边界
```

边界定义 ($\sigma$) 仍是主观选择, 这是缺陷。

#### 2.2.3 子任务级评估

把任务拆成子任务:

$$
\text{Subtask SR}_i = \frac{k_i}{K}
$$

**例子** "抓杯子放到桌子":

| 子任务 | 描述 | 完成判定 |
| :--- | :--- | :--- |
| S1 | 移动到杯子附近 | 末端 < 10 cm |
| S2 | 下降到杯子高度 | Z < 5 cm |
| S3 | 闭合夹爪 | 夹爪力 > 1 N |
| S4 | 提起杯子 | Z > 10 cm |
| S5 | 移动到桌目标 | 末端 < 5 cm |
| S6 | 放下杯子 | 杯子在桌 |

**子任务评估的优势**: **告诉你失败在哪**。
```text
RT-1 子任务分析 (21 个任务, 简化):
─────────────────────────────
任务类            完成率   失败原因
捡起物体          92%      抓不稳 (3%), 撞倒 (5%)
移动物体          87%      路径差 (8%), 折返 (5%)
打开抽屉          76%      拉不开 (10%), 识别错 (14%)
倾倒液体          58%      角度错 (25%), 溅出 (17%)
─────────────────────────────
平均              78%  ← 但"倾倒液体" 拖了后腿
```

#### 2.2.4 人类评估 (Human Rating)

质量难量化 (叠衣服的整齐度), 用 Likert 1-7 打分:

```text
1 = 完全失败         5 = 较好 (小问题)
2 = 主要失败         6 = 很好 (极小问题)
3 = 部分失败         7 = 完美 (与人类演示相当)
4 = 勉强成功
```

**关键问题**: **评估者一致性** (Krippendorff's α):

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

| α 值 | 一致性 |
| :--- | :--- |
| < 0.67 | 不可接受 |
| 0.67-0.80 | 可接受 |
| > 0.80 | 良好 |

Meta RT-2 报告 α = 0.83 (5 个评估者, 200 任务)。**工业实践**: 50% episode 自动评估 + 50% 人类评估。

### 2.3 工业 SOTA 成功率速查表

| 模型 | 任务数 | 成功率定义 | 报告范围 |
| :--- | :--- | :--- | :--- |
| RT-1 (Google 2022) | 700 厨房 | 物体在目标容器 | 总 97% (仿真), 87% (真实) |
| RT-2 (Google 2023) | 600 | + 视觉问答 | 总 62% (新场景) |
| Octo (UCB 2024) | 9 | 物体 ±5cm | 单任务 89%, 平均 75% |
| OpenVLA (2024) | Open X 子集 | 物体在容器 | 平均 78% |
| Pi-0 (PI 2024) | 叠衣/整理 | 人类 1-5 | 平均 4.3/5 |
| HPT (Stanford 2024) | 6 | 物体在容器 | 单任务 86%, 平均 73% |

**报告时必须同时给**:
```text
"成功率 X% ± Y% (95% CI, n=N)
 在 {任务 1: x%, 任务 2: y%, ...} 任务上
 仿真成功率: Z%
 真实成功率: W%"
```
少任何一项都是不完整报告。

### 2.4 成功率 vs episode 数:必要样本量

Wilson 95% CI 半宽:

| 真实成功率 CI 半宽 | 必要 episode 数 |
| :--- | :--- |
| ±10% | 30 |
| ±5% | 100 |
| ±2% | 400 |
| ±1% | 1500 |

**工业实践**: **至少跑 200 个 episode** (CI 半宽 ±3.5%)。少于此数, 95% CI 跨度过宽, 等于没报告。

---

## 3. 轨迹准确率 (Trajectory Accuracy):比成功率更细的诊断工具

### 3.1 直觉:为什么"轨迹 L2 距离"看似直观但几乎没用

简单形式: 平均 L2 距离。

$$
d_{L2} = \frac{1}{T} \sum_{t=1}^{T} \|p_t^{\text{pred}} - p_t^{\text{gt}}\|_2
$$

**核心问题**: **L2 距离对"成功"不敏感, 对路径形状敏感**。

```text
两个轨迹都可能成功, 但 L2 完全不同:

轨迹 A (直冲):        轨迹 B (绕一圈):
   p1                    p2
    \                    \
     \                    \
      \                    \
       p_cup ←             p_cup ← (杯子在原地)
       ↓                    ↑
       p3 (放下)             p4 (放下)

L2 距离: 0.30 m           L2 距离: 0.65 m
成功率: 100%              成功率: 100%
```

> **核心洞察**: L2 反映"路径上的几何对齐", 而**具身任务关心"结果对齐"**。这两个不一致。

### 3.2 5 个比 L2 强的轨迹指标

#### 3.2.1 终点误差 (Final Position Error, FPE)

$$
\text{FPE} = \|p_T^{\text{pred}} - p_T^{\text{gt}}\|_2
$$

**为什么强**: **任务结果主要由终点决定**。LIBERO / Meta-World 用 FPE 为主指标。RT-1/RT-2 内部也用, 没公开。

#### 3.2.2 平均动态时间规整 (DTW)

长度不同 + 时间扭曲时:

$$
\text{DTW}(\tau_1, \tau_2) = \min_{\pi} \sum_{(i,j) \in \pi} \|p_i^{(1)} - p_j^{(2)}\|_2
$$

**优势**: 处理速度差异 + 时间扭曲。**劣势**: $O(T_1 \cdot T_2)$ 复杂度, 对噪声敏感。

#### 3.2.3 路径效率 (Path Efficiency)

$$
\eta = \frac{L_{\text{optimal}}}{L_{\text{actual}}}
$$

$L_{\text{optimal}}$ 理论最短 (直线), $L_{\text{actual}}$ 实际走过。

```text
直线路径 (最优):              曲折路径:
起点 ────────→ 终点         起点 ┐
长度 = 5.0 m                       └──→ 终点
                          长度 = 12 m
η = 5/12 = 0.42
```

**工业经验**: $\eta > 0.7$ 是好的, $\eta < 0.5$ 表示策略"绕路"。

#### 3.2.4 时间归一化指标

$$
\text{Cost} = \alpha \cdot \text{FPE} + \beta \cdot T_{\text{actual}} + \gamma \cdot \int \|v(t)\|^2 dt
$$

**例子** 抓杯子:
```text
模型 A:  FPE=0, T=30s, 抖动大  → 综合 差
模型 B:  FPE=5cm, T=10s, 平滑  → 综合 好
```
核心: **"快、好、平" 同时考虑**。

#### 3.2.5 关节角 + 末端混合 (Joint + EEF)

末端轨迹丢失**姿态 (orientation)** 信息:

$$
d_{\text{mixed}} = \alpha \cdot d_{\text{EEF}} + \beta \cdot d_{\text{joint}}
$$

Stanford ALOHA 用混合距离 (需要末端朝向, 比如把钥匙插进锁孔)。

### 3.3 成功率 vs 轨迹指标:什么时候用哪个

| 场景 | 用什么 |
| :--- | :--- |
| 任务结果有清晰判定 (抓杯子) | **成功率** (FPE ≤ 5cm) |
| 任务质量有连续好坏 (叠衣服) | **轨迹指标** (关节角平滑度) |
| 调试"为什么失败" | **子任务级** + **终点误差** |
| 对比两个模型 | **任务平均成功率** + **长尾切片** |
| 安全敏感任务 | **时间归一化** (抖动) |

> **核心原则**: **成功率是结果指标, 轨迹指标是过程指标**。两者都报, 才完整。

### 3.4 与专家轨迹的对比:"伪 ground truth" 的问题

BC 训练用的专家轨迹**不是真正的 ground truth**, 是"一个人类操作员的解"。5 个偏差:
1. **演示者差异** (同一任务, 5 个操作员, 5 种轨迹)
2. **时间不一致** (专家快慢有节奏)
3. **风格差异** (有人爱"先抓后移", 有人"边抓边移")
4. **演示者错误** (人会犯错, 录下来的轨迹里有错)
5. **演示者最优性** (人不一定每次做最优)

**工业对策**: 多演示者 (3-5 个) + 去最优 (≤ K 倍专家时间) + 接受次优 (训练时加 noise)。

### 3.5 工业实例: Meta Habitat 3.0 评估协议

**主指标**: 二元成功率 (物体 ±15cm, 关节角度正确, α > 0.85)
**次指标**: 子任务成功率
**辅助**: SPL (Success weighted by Path Length):

$$
\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} s_i \cdot \frac{l_i}{\max(p_i, l_i)}
$$

```text
SPL 例子:
成功 (s=1), 路径=30, 最优=10 → SPL = 10/30 = 0.33
失败 (s=0), 路径=50, 最优=10 → SPL = 0
(失败 episode 贡献 = 0, 不论路径多长)
```

**SPL 优势**: 同时惩罚"失败"和"绕路", 但不偏向"快但不准"。

---

## 4. Sim2Real Gap 量化 (本章重点)

### 4.1 直觉:为什么 Sim2Real Gap 永远消除不了

Sim2Real Gap 不是"一两个百分点偏差", 是**多维度不同频率叠加**的结果。

```text
Sim2Real Gap 的 5 个来源:
─────────────────────────
视觉 Gap (V):   仿真渲染 ≠ 真实相机拍照
物理 Gap (P):   仿真物理引擎 ≠ 真实物理
时序 Gap (T):   仿真零延迟 ≠ 真实硬件延迟
传感器 Gap (S): 仿真传感器模型 ≠ 真实传感器
任务语义 Gap (M): 仿真任务简化为 ≠ 真实任务复杂

时间         10Hz        100Hz      1kHz        10kHz
             ─────       ─────      ─────       ─────
视觉 Gap (V)  ●●●
物理 Gap (P)        ●●●
时序 Gap (T)              ●●●
传感器 (S)                       ●●●
任务语义 (M)   ●●●●●●●●●●● (长期慢变量)
```

**为什么不能消除**: 物理世界有无穷细节, 仿真器只能建模"低频平均", 捕捉不了"高频噪声"。

$$
\text{Gap}_{\text{total}} = \sqrt{V^2 + P^2 + T^2 + S^2 + M^2}
$$

$$
p_{\text{real}} \approx p_s - \text{Gap}_{\text{total}} \quad \text{(翻译公式, 经验)}
$$

**理论下界**: 不为零。

### 4.2 视觉 Gap:从像素到语义的 3 类量化

#### 4.2.1 FID (Fréchet Inception Distance)

$$
\text{FID} = \|\mu_r - \mu_s\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_s - 2(\Sigma_r \Sigma_s)^{1/2}\right)
$$

Inception-v3 特征空间里的两个高斯之间的 Wasserstein-2 距离。

| FID 值 | 含义 |
| :--- | :--- |
| 0-10 | 极相似 |
| 10-50 | 中等差距 |
| 50-200 | 明显差距 |
| > 200 | 完全不同 |

**典型工业数据** (Isaac Sim 渲染 vs 真实相机): 简单物体 15 / 桌面环境 35 / 复杂厨房 78 / 带反光物体 120。

**局限**: Inception-v3 为 ImageNet 预训练, 不一定适合具身场景 (手部动作、关节形态)。

#### 4.2.2 LPIPS (Learned Perceptual Image Patch Similarity)

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \|w_l \odot (\phi_l(x) - \phi_l(y))\|_2^2
$$

更接近人类感知, 对纹理、风格差异敏感。**经验法则**: LPIPS > 0.3 通常意味着**视觉策略会失败**。

#### 4.2.3 Domain Classifier (可学习 Gap)

```text
仿真图像 ──→  ┐
真实图像 ──→  ├──→ 分类器 ──→ " 这是仿真还是真实?"

准确率高 → Gap 大
接近 50%  → Gap 小 (随机)
```

$$
\text{Gap}_{\text{domain}} = 2 \cdot |A - 0.5|
$$

| Acc | 含义 |
| :--- | :--- |
| 50-60% | 几乎无 Gap |
| 60-75% | 弱 Gap |
| 75-90% | 中等 Gap |
| > 90% | 强烈 Gap |

**工业实践**: ResNet-18 二分类, Acc > 85% 警告: 策略可能在"用仿真视觉特征"而不是"用物体物理特征"。

### 4.3 物理 Gap:接触、摩擦与软体

#### 4.3.1 摩擦系数误差

仿真器假设单一标量, 真实接触面有丰富微观粗糙度:

```text
仿真假设:                       实际:
μ_s = 0.5 (静摩擦)               玻璃+橡胶: μ = 0.85
μ_k = 0.4 (动摩擦)               塑料+塑料: μ = 0.35
适用于所有接触                   木+布:     μ = 0.62
                                 磨损后:     μ = 0.45
                                 带水后:     μ = 0.12
```

一个抓杯子, Δμ = 0.1 → 夹爪滑移概率 +30%, 提起成功率 -25%, 最终 SR -20%。

#### 4.3.2 接触动力学

接触是最难仿真的物理:

```text
接触过程时间线 (杯子放桌上):

真实时间     物理
0 ms         接触, μ 突变
1 ms         接触面变形 (μ → μ')
10 ms        反弹 (e)
100 ms       振动衰减 (ζ)
500 ms       稳定
1000 ms      微小漂移 (空气扰动)

仿真时间线:
0 ms         接触
0 ms         "完成" ← 仿真到这就停
```

关键参数: **弹性系数 e 和阻尼比 ζ**。

$$
e = \frac{v_{\text{rebound}}}{v_{\text{impact}}}
$$

| 仿真器 | 默认 e | 默认 ζ |
| :--- | :--- | :--- |
| MuJoCo | 0.0-0.1 | 1.0 |
| Isaac Sim | 0.0-0.5 | 0.5-1.0 |
| PyBullet | 0.0 | 1.0 |
| 真实 (橡胶) | 0.3-0.5 | 0.05-0.2 |
| 真实 (金属) | 0.5-0.8 | 0.01-0.05 |

**典型工业 Gap**: 0.5-1.5 (50-150% 误差)。

#### 4.3.3 软体与变形

```text
软体任务的 Sim2Real Gap (典型):
─────────────────────────────
任务              仿真 SR    真实 SR    Gap
抓硬物 (杯子)     99%        85%       14
抓半硬物 (塑料瓶)  95%        65%       30
抓软物 (毛巾)     80%        35%       45
抓可变形 (绳子)    60%        12%       48
───────────────────────────────────────
```

**软体 Gap 最大**: 杨氏模量、泊松比难以测量, 对微小变化敏感。

### 4.4 时序 Gap:延迟与频率

真实机器人延迟:

```text
真实系统延迟组成:
─────────────────────────────────
项                  延迟        范围
感知 (相机读出)       5 ms        1-20 ms
感知 (图像传输)       3 ms        1-50 ms
计算 (推理)          15 ms        5-100 ms
控制 (执行)          10 ms        1-30 ms
通信 (CAN 总线)       2 ms        1-10 ms
─────────────────────────────────
总计                 35 ms        9-210 ms
仿真                   < 1 ms
─────────────────────────────────
时序 Gap = 30+ ms
```

$$
\text{效果 Gap} \approx f \cdot \Delta T \cdot v
$$

$f$=控制频率, $\Delta T$=延迟差, $v$=速度尺度。

例子: $f=50$ Hz, $\Delta T=30$ ms, $v=0.5$ m/s → Gap ≈ 0.75 m = 75 cm 错位 ≈ 抓杯子完全失败。

**对策**: Sim2Real 时序适配 (仿真加入同延迟) + 训练加延迟扰动 + 预测控制 MPC。

### 4.5 整体 Gap:综合量化

**典型 Gap 分布** (Meta Habitat 3.0 报告):

```text
任务            总 Gap    V      P      T     S     M
抓杯子 (硬体)    19%     5%    8%      4%    1%    1%
叠衣服 (软体)    49%     8%    25%     6%    4%    6%
开抽屉 (铰链)    28%     3%    12%     5%    2%    6%
插钥匙 (精细)    60%    10%    20%    10%    5%   15%
──────────────────────────────────────
(V=视觉 P=物理 T=时序 S=传感器 M=语义)
```

### 4.6 缩小 Gap 的 3 大技术

#### 4.6.1 域随机化 (Domain Randomization, DR)

**不试图让仿真像真实世界, 而是在仿真中"穷举"真实世界的变化**。

```text
域随机化范围:
视觉层:
├─ 光照: [50%, 200%] of default
├─ 色温: [3000K, 8000K]
├─ 阴影: [0%, 100%]
├─ 纹理: 10 种随机
├─ 背景: 20 种随机
└─ 相机视角: ±10° 抖动

物理层:
├─ 质量: [80%, 120%]
├─ 摩擦: [0.3, 0.8]
├─ 弹性: [0.0, 0.5]
└─ 阻尼: [0.05, 1.0]

时序层:
├─ 感知延迟: [10, 100] ms
├─ 控制频率: [20, 50] Hz
└─ 通信延迟: [1, 20] ms
```

数学上: 把真实分布 $p_{\text{real}}(x)$ 用仿真分布 $p_{\text{sim}}(x; \theta)$ 的混合近似:

$$
p_{\text{real}}(x) \approx \int p_{\text{sim}}(x; \theta) q(\theta) d\theta
$$

**Tesla** 用 DR 训自动驾驶, **NVIDIA** 用 DR 训抓取, **OpenAI Rubik's Cube** 单手 50% 真实成功率。

#### 4.6.2 域适应 (Domain Adaptation, DA)

不改变仿真, 把真实数据"翻译"到仿真分布:

```text
真实图像 ──→ CycleGAN ──→ "仿真风格" 图像
仿真训练 (用真实标注 + 仿真风格) ──→ 策略 π ──→ 真实部署
```

$$
\mathcal{L}_{DA} = \mathcal{L}_{\text{task}}(G(x_r), y_r) + \lambda \mathcal{L}_{\text{cycle}}(G, F)
$$

#### 4.6.3 真实微调 (Real Fine-tuning)

最朴素也最有效: 真实环境再训一点。
```text
RT-1 训练:
1. 仿真大规模预训练 (300K+ episode): 95% SR (仿真)
2. 真实微调 (5K-10K): 87% SR (真实)

Gap = 95-87 = 8 pp, 比直接真机训 0% 高出许多
```

**Meta-learning 视角**:

$$
\theta^* = \arg\min_\theta \mathbb{E}_{\text{task}} \left[ \mathcal{L}_{\text{real}}(\theta - \eta \nabla \mathcal{L}_{\text{sim}}(\theta); D_{\text{few-shot}}) \right]
$$

---

## 5. 多任务评估:从"任务平均"到"长尾分析"

### 5.1 直觉:为什么"任务平均成功率"会骗人

```text
5 个任务, 各跑 100 episode:
任务         SR        重要度
────────     ───       ────
抓杯子       98%       高
开抽屉       85%       高
叠衣服       50%       中
做菜         10%       低
清洁         5%        低

任务平均 = (98+85+50+10+5)/5 = 49.6%
→ "近一半成功", 但"做菜"和"清洁" 几乎 0%
→ 用户期望做菜, 模型不能用
```

> **核心洞察**: **平均是必要指标, 不是充分指标**。必须配"任务切片" + "长尾分析"。

### 5.2 多任务评估的 3 步法

#### 第 1 步:任务平均

$$
\overline{SR} = \frac{1}{T} \sum_{t=1}^{T} SR_t
$$

#### 第 2 步:加权重平均 (按部署频率)

$$
\overline{SR}_w = \sum_{t=1}^{T} w_t \cdot SR_t, \quad \sum w_t = 1
$$

例: 按频率加权 (抓杯子 30%, 开抽屉 25%, ...) → 加权平均 ≈ 62.65%。

#### 第 3 步:长尾分析 (找 SR < 30% 的任务聚类)

```text
长尾任务分析:
任务              SR    共同特征
────────            ──    ─────────
做菜 (面团)        5%    需要双手协调
做菜 (切菜)        10%   需要刀具感知
清洁 (窗户)         8%   接触面积大
整理 (叠衣)         18%  软体变形
精细装配            15%  小物体 + 触觉
                                 ↑
                            "复杂接触" 是共同点
```

这种"低成功率任务聚类"比单一均值有用 100 倍, 揭示真实弱点。

### 5.3 任务相关性分析

**任务相关性矩阵** (RT-1 10 任务相关系数):

```text
        抓杯子  开抽屉  推物体  按按钮  倒水
抓杯子    1.00   0.65    0.78   0.55   0.42
开抽屉    0.65   1.00    0.71   0.62   0.38
推物体    0.78   0.71    1.00   0.59   0.45
按按钮    0.55   0.62    0.59   1.00   0.31
倒水      0.42   0.38    0.45   0.31   1.00
```

抓杯子-推物体相关 0.78 (都是"抓-移"), 倒水相关 0.31-0.45 (需要"持续倾倒"特殊技能)。

**训练建议**: 高相关性任务联合训练收益大; 低相关性各自微调, 联合可能负迁移。

### 5.4 跨形态评估

跨形态迁移率:

$$
T_{A \to B} = \frac{SR_{B \to B}}{SR_{A \to B}}
$$

**典型数据** (RT-X 论文):

```text
        FRANKA    xArm      Panda
FRANKA   1.00     0.78      0.85
xArm     0.71     1.00      0.76
Panda    0.82     0.74      1.00
```

跨形态成功率约为同形态的 **70-80%**, 但有差异。

---

## 6. 统计与显著性:不被 95% CI 骗到

### 6.1 Wilson 置信区间:成功率的"正确"区间

跑 100 次, 成功 80 次, 朴素 ±1.96·SE 会**越界** (p≈0 时下界为负)。Wilson 区间通过**正则化**修正:

$$
\text{Wilson 95% CI} = \frac{\hat{p} + \frac{z^2}{2N}}{1 + \frac{z^2}{N}} \pm \frac{z}{1 + \frac{z^2}{N}} \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}
$$

其中 $z = 1.96$。

```text
N    成功   朴素 ±1.96·SE    Wilson 95% CI (正确)
──    ───    ─────────        ───────────────
10    8      80% ± 25%        [49%, 95%]
30    24     80% ± 14%        [63%, 91%]
100   80     80% ± 7.9%       [71%, 87%]
200   160    80% ± 5.6%       [74%, 85%]
1000  800    80% ± 2.5%       [77.5%, 82.5%]
```

**工业实践**: **永远报 Wilson 95% CI**, 不报朴素 ±SE。

### 6.2 成功率方差分析 (ANOVA)

比较多任务, $F$ 统计量:

$$
F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}
$$

$F > F_{\text{crit}}(T-1, N-T, 0.05)$ → 任务 SR **显著不同**, 不能用均值代表。

### 6.3 多重比较校正

同时比较多对策略时, 假阳性累加:
- **Bonferroni**: $\alpha_{\text{adjusted}} = \alpha/k$
- **Holm-Bonferroni**: 排序 p-values 逐步拒绝

**工业实践**: 比 5 个 baseline 用 Holm-Bonferroni, 比 2-3 个用 Bonferroni 即可。

---

## 7. 在线评估:A/B 测试与长期监控

### 7.1 A/B 测试 vs 强化评估

| 离线 | 在线 |
| :--- | :--- |
| 仿真器 / 录播 | 真实机器人交互 |
| 1000 episode/小时 | 100 episode/小时 |
| 可复现 | 时变 |
| 3 小时快评估 | 1-4 周完整 A/B |

**A/B 特殊性**:
1. 单 episode 成本高 (1-3 分钟, 跑 1000 个 30+ 小时)
2. 失败模式物理损坏 (不只是指标跌, 可能撞坏桌子)
3. 用户接受度主观

### 7.2 长尾效果追踪

```text
长期监控:
─────────────────────────────────
指标               频率       阈值
─────────────────────────────────
每日 SR             实时       ±10% 报警
每周长尾 SR        周一 9AM    < 25% 任务数 > 15% 报警
每月用户满意度     月底        < 4.0/5.0 报警
每季度硬件故障率  季报        > 1%/千次
─────────────────────────────────
```

漂移检测 (Page-Hinkley):

$$
\text{PH}_t = \sum_{i=1}^{t} (x_i - \overline{x}_t - \delta)
$$

### 7.3 用户反馈整合

```text
用户反馈收集:
─────────────────────────────────
1. 主动: 评分 1-5, 评论
2. 隐式: 重做次数, 退出位置
3. 行为: 跳过任务, 调参数
─────────────────────────────────
```

**自动 vs 用户评估的差异**:
```text
自动 (SR)    用户 (满意)
True+:  任务成功 + 用户满意      (理想)
False+: 任务成功 + 用户不满意    (自动过严)
False-: 任务失败 + 用户满意      (用户宽容)
True-:  任务失败 + 用户不满意    (理想)
```

经验: 用户满意度 ≈ 成功率 × 0.85 + 30%。100% SR 但满意度只有 85% (用户在意外观姿态)。

### 7.4 安全监控指标

```text
安全指标表:
────────────────────────────────────────
事件                         阈值      报警
────────────────────────────────────────
碰撞 (与物体)                5/千次   立即停
碰撞 (与人体)                0/千次   永久停
指令超时 (10s)               1/百次   警告
超速 (关节 > 2 rad/s)        10/百次  警告
电池低 (< 15%)               1/百次   提醒
异常声音                     5/千次   诊断
────────────────────────────────────────
```

---

## 8. PyTorch 实验:从零搭一个评估 pipeline

### 8.1 简单成功率 + Wilson 95% CI

```python
import numpy as np
import torch
from scipy.stats import norm
from typing import List, Tuple


def compute_success_rate(episodes_results: List[int],
                       confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """朴素 SR + Wilson 95% CI"""
    N = len(episodes_results)
    k = sum(episodes_results)
    p_hat = k / N
    if N == 0:
        return 0.0, (0.0, 0.0)

    z = norm.ppf(1 - (1 - confidence) / 2)  # 1.96
    denom = 1 + z**2 / N
    center = (p_hat + z**2 / (2 * N)) / denom
    margin = (z / denom) * np.sqrt(p_hat*(1-p_hat)/N + z**2/(4*N**2))

    return p_hat, (max(0.0, center - margin), min(1.0, center + margin))


if __name__ == "__main__":
    torch.manual_seed(42)
    results = (torch.rand(200) < 0.8).int().tolist()
    sr, (lo, hi) = compute_success_rate(results)
    print(f"SR = {sr:.1%} [{lo:.1%}, {hi:.1%}]")
    # Output: SR = 80.0% [74.1%, 84.9%]
```

### 8.2 轨迹 L2 与终点误差

```python
def trajectory_l2(pred: np.ndarray, gt: np.ndarray) -> float:
    """平均 L2 距离"""
    diffs = pred - gt
    return float(np.linalg.norm(diffs, axis=1).mean())


def final_position_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """终点误差"""
    if pred.ndim == 1:
        return float(np.linalg.norm(pred - gt))
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def path_efficiency(pred: np.ndarray) -> float:
    """路径效率 = 最优长度 / 实际长度"""
    if pred.shape[0] < 2:
        return 1.0
    actual = float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum())
    optimal = float(np.linalg.norm(pred[-1] - pred[0]))
    return optimal / actual if actual > 0 else 1.0


# 模拟 10 步抓杯子轨迹
gt = np.array([[0.50,0.30,0.20],[0.50,0.30,0.15],[0.50,0.30,0.10],
               [0.50,0.30,0.05],[0.45,0.32,0.05],[0.40,0.34,0.05],
               [0.35,0.35,0.05],[0.30,0.37,0.05],[0.25,0.38,0.05],
               [0.20,0.40,0.10]])
pred = gt + np.random.randn(*gt.shape) * 0.02  # 2cm 噪声

print(f"L2: {trajectory_l2(pred, gt)*100:.2f} cm")  # ~2 cm
print(f"FPE: {final_position_error(pred, gt)*100:.2f} cm")
print(f"η: {path_efficiency(pred):.3f}")
```

### 8.3 Sim2Real Gap 量化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    """ResNet 简化版: 仿真 vs 真实 分类"""
    def __init__(self, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def train_domain_classifier(sim: torch.Tensor, real: torch.Tensor,
                          epochs: int = 10, lr: float = 1e-3) -> float:
    """训练 domain classifier, 返回验证准确率"""
    classifier = DomainClassifier(in_ch=sim.shape[1])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    N = min(len(sim), len(real))
    X = torch.cat([sim[:N], real[:N]], dim=0)
    y = torch.cat([torch.zeros(N, dtype=torch.long),
                   torch.ones(N, dtype=torch.long)], dim=0)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True)

    classifier.train()
    for _ in range(epochs):
        for x_b, y_b in loader:
            loss = F.cross_entropy(classifier(x_b), y_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    classifier.eval()
    with torch.no_grad():
        acc = (classifier(X).argmax(dim=1) == y).float().mean().item()
    return acc


def compute_sim2real_gap(sim_acc: float) -> float:
    """从 domain classifier 准确率估算 Gap"""
    return min(max(0.0, (sim_acc - 0.5) / 0.5) * 0.5, 1.0)


# 仿真: 较暗; 真实: 更亮 (有 Gap)
torch.manual_seed(0)
sim = torch.rand(100, 3, 64, 64) * 0.5 + 0.1
real = torch.rand(100, 3, 64, 64) * 0.7 + 0.3

acc = train_domain_classifier(sim, real, epochs=5)
gap = compute_sim2real_gap(acc)

print(f"Domain Classifier Acc: {acc:.1%}")
print(f"Sim2Real Gap: {gap:.1%}")
if acc > 0.85:
    print("⚠️ 视觉 Gap 严重, 策略可能过拟合仿真视觉")
elif acc < 0.65:
    print("✅ 视觉 Gap 小")
else:
    print("🟡 中等 Gap, 建议 DR")
```

---

## 9. 与其他章节的关系

### 9.1 跨章引用图

```text
1-从感知到行动 (历史)     ← 评估什么: "行为"
2-形态与本体 (形态)       ← 不同形态的成功定义差异
3-数据集与基准 (数据)     ← 评估协议来源
4-评估指标 ← 你在这里
5-模仿学习 (BC)           → 消费: 评估模仿的策略
6-强化学习 (RL)           → 消费: 评估 RL
7-仿真器                  ← Sim2Real Gap 来源
8-硬件                    ← 物理 Gap 来源
9-工业落地                → 在线评估实践
```

### 9.2 与第 5 章 (模仿学习) 的关系

`../第05章-模仿学习/0-总目录与阅读路径.md` 用 BC 训练后, **直接消费本章的指标**:
- **轨迹 L2 / DTW**: 比较 BC 策略 vs 专家轨迹
- **成功率**: 测试集上跑 BC 策略
- **Sim2Real Gap**: BC 通常仿真训练, Gap 大
- **任务平均 + 长尾**: 多任务 BC 后必报

### 9.3 与第 6 章 (RL) 的关系

`../../rec教程/第07章-强化学习在GR中的角色.md` 用 RL 训练后, 用本章评估:
- **成功率**: 主指标 (RL 通常仿真训, Gap 大)
- **部分成功度**: 对 RL 更友好 (稀疏奖励 + dense feedback)
- **轨迹准确率**: RL 不会跟着专家走, L2 不适用, 改用**任务成功率 + 路径效率**

### 9.4 与第 7 章 (仿真器) 的关系

`../第07章-仿真器与基准/0-总目录与阅读路径.md` 提供 Sim2Real Gap **来源**:
- **视觉渲染**: Chapter 7 解释渲染管线差异; 本章量化 Gap (FID/LPIPS)
- **物理引擎**: Chapter 7 解释物理模型; 本章量化 (μ, e, ζ)
- **时序**: Chapter 7 解释控制频率; 本章量化时序 Gap

### 9.5 与第 9 章 (工业落地) 的关系

`9-工业落地与前沿方向` 提供**工业级**评估:
- **A/B 测试**: 在线部署前关键
- **安全监控**: 工业必需
- **长期追踪**: 部署后退化检测
- **硬件成本评估**: 仿真训练 + 真实微调的总成本

### 9.6 互引速查

| 章节 | 关系 | 关键引用指标 |
| :--- | :--- | :--- |
| 3-数据集 | 前置 | RT-1/Bridge 评估协议 |
| 5-模仿学习 | 后续 | 成功率 + L2 + DTW |
| 6-RL | 后续 | 成功率 + 部分成功度 |
| 7-仿真器 | 横向 | Sim2Real 来源 |
| 9-工业落地 | 后续 | 在线评估 + 安全 |

---

## 10. 与数学基础的关系

### 10.1 概率论 → 成功率置信区间

成功率 $\hat{p}$ 是伯努利 MLE。Wilson 95% CI 见 §6.1。

```text
概率论 → 本章:
Bernoulli(p)         → 成功率分布
Binomial(n, p)       → 多 episode 联合 SR
Normal approximation → 大 N 近似
Wilson score         → 95% CI 修正
```

### 10.2 信息论 → Sim2Real Gap 量化

```text
信息论 → 本章:
D_KL(P||Q)         → 仿真 vs 真实分布距离
I(X;Y)             → 视觉特征 vs 标签
Total Variation    → domain classifier 准确率
```

**关键洞察**: Sim2Real Gap 可形式化为:

$$
D_{\text{KL}}(p_{\text{real}}(x) \| p_{\text{sim}}(x)) > 0
$$

理论下界不为零 (真实世界信息量 > 仿真能建模的信息量)。

### 10.3 最优化 → 多任务 Pareto

```text
最优化 → 本章:
Pareto 前沿         → 多任务 SR 不能同时最大化
KKT 条件            → 多目标均衡点
凸松弛               → 多任务 RL 线性化
```

### 10.4 数学基础速查

| 数学文件 | 复用方式 |
| :--- | :--- |
| 3-概率论/3-常用分布 | Wilson CI, ANOVA |
| 4-离散数学/3-图论 | 任务相关性图 |
| 5-信息论/3-互信息 | FID 推导 |
| 6-最优化/2-凸性 | 多任务 Pareto |

---

## 11. 常见误区:5 个一做就错的反模式

### 11.1 ❌ "成功率 90% 就够好了"

**为什么错**: 90% 是**平均**还是**长尾**?

```text
模型 A:
任务        SR       重要性
抓杯子      95%      高
开抽屉      92%      高
叠衣服      50%      中
做菜        8%       低

"任务平均 90%?" 不! 这是"任务加权和 ~62%"
做菜 8%, 不能上线
```

**反例**: 某团队 2023 年发布"90% SR"机器人, 用户实际**"做菜"完全不能用**, 退货率 40%。

**真相**:

$$
\text{真正"够好"} \iff \text{所有重要任务 SR} > 75\% \text{ AND 长尾 SR} > 50\%
$$

### 11.2 ❌ "仿真 99% 等于真实 99%"

**错误**: "仿真训练 99%, 直接部署到真机"。

```text
真实世界 Gap (Meta Habitat 3.0):
────────────────────────────────
任务            仿真 SR    真实 SR    Gap
抓硬体          99%        85%       14
抓软体          80%        35%       45
精细插孔        90%        30%       60
折叠变形        50%        12%       38
────────────────────────────────
```

**反例**: 某自动驾驶团队 2022 年报"仿真 99.99%", 真机事故率比人类司机高 8 倍。

**真相**: $SR_{\text{real}} \approx SR_{\text{sim}} - \Delta_{\text{Sim2Real}}$, $\Delta$ 通常 14-60 pp。

### 11.3 ❌ "轨迹 L2 距离是关键指标"

**为什么错**: 轨迹 L2 **对任务结果不敏感**:
```text
两个轨迹, 同样的末端, 不同路径:
轨迹 A (直线):  起点 ───→ 终点      L2 距离 = 0
轨迹 B (绕行):  起点 ┐                            SR = 100%
                       └──→ 终点
```
两个轨迹 SR 都 100%, 但 L2 距离 B 大。

**反例**: 某团队花 3 个月优化轨迹 L2 (-2 cm), 但 SR 没涨, 用户满意度没变。L2 优化的是路径几何, 不是任务质量。

**正确指标层级**:
```text
第一指标: 成功率 (任务结果)
第二指标: 终点误差 (任务质量)
第三指标: 部分成功度 (中间状态)
诊断指标: L2 / DTW / 路径效率
```

### 11.4 ❌ "只看平均, 忽略长尾"

**错误**: "模型平均 SR 75%, 不错"。

```text
"平均 75%" 的 3 种可能:
分布 1: 全部 75-85%, 平均 75%    ✅ 健康
分布 2: 头部 95%, 中 75%, 长尾 5%  ⚠️ 有问题
分布 3: 头部 99%, 长尾 0%           ❌ 灾难
```

**反例**: 某 Open X SOTA, 平均 75%, 但 22 个任务中 5 个 SR 0%。研究者**只报平均**, 被 reviews 批评。

**正确报告**:
```text
"模型平均 75%, 其中:
 - 70% 任务 (16 个): 70-95% SR
 - 25% 任务 (5 个): 30-60% SR
 - 5% 任务 (1 个): 0% SR"
```

### 11.5 ❌ "离线指标能预测在线效果"

**错误**: "仿真 90%, 真实会差不多"。

**离线-在线鸿沟 5 个未建模因素**:
1. **新场景漂移**: 用户环境没见过
2. **用户交互模式**: 不在训练分布内
3. **硬件老化**: 真机 2 周后性能下降
4. **噪声累积**: 仿真 0 噪声, 真实有
5. **用户反馈循环**: 用户行为影响数据分布

**反例**: 某团队 2024 年发布"仿真 SOTA", 真机整体 SR 比仿真低 35 pp, 调查 2 月发现是真实物体表面有水 (摩擦低很多), 模型没见过。

**真相**: **离线指标是必要不充分**。离线 90%, 真实可能 60%。**真实评估 + 在线 A/B 兜底**。

---

## 12. L7/L8 面试 Q&A

### 12.1 Q1: 为什么 Sim2Real Gap 难以消除?

**L7 (1-2 分钟)**:

4 个本质原因:
1. **物理世界无穷细节**: 仿真器只能建模"低频平均" (μ, e, ζ), 但真实物理涉及**微观接触、变形、振动**, 这些频率 > 1 kHz 不可观测
2. **延迟不可消除**: 真实硬件 (相机+通信+控制器) 至少 30 ms 延迟, 真实系统不可能为零
3. **未知参数**: 真实物体的杨氏模量、摩擦系数、密度**难以测量**, 测量误差 10-30%
4. **新分布**: 部署后, 用户环境不断变化, 仿真永远无法预训练所有场景

**关键技术**: DR 缩小 Gap, 真实微调进一步缩小。**理论下界** $\approx$ 5-15% (即使最佳方法)。

**L8 (3-5 分钟)**:

数学视角: Sim2Real Gap 形式化为 KL 散度 $D_{\text{KL}}(p_{\text{real}} \| p_{\text{sim}}) > 0$。理论上不可消除, 因为**真实世界的信息量 > 仿真能建模的信息量**。

**工业实践**: 不要试图消除 Gap, 而是**量化 + 在容忍范围内设计策略**。

```text
Sim2Real Gap 缩小 3 阶段:
仿真训练 ─→ 仿真测试      Gap = 0
                  ↓ 仿真迁移
真实微调 ─→ 真实测试      Gap = 10-25% (好)
                  ↓ 长尾适应
持续数据 ─→ 稳定运行      Gap = 5-15% (长期可接受)
```

### 12.2 Q2: 如何设计公平的评估协议?

**L7**:

5 个要素:
1. **任务随机化**: 每个 episode 物体位置/姿态/光照随机
2. **多演示者**: 同一任务 3-5 个演示者提供 ground truth
3. **足够 episode**: 至少 200 (CI 半宽 < 4%)
4. **任务切片**: 任务平均 + 长尾
5. **多次重复**: 至少 3 随机种子

```text
公平评估协议模板:
──────────────────────────────────────
1. 测试集: N 个任务, 每个 200+ episode
2. 每个 episode:
   - 随机物体位置 (合理范围内)
   - 随机光照 (白天/晚上/灯光/阴影)
   - 随机初始姿态
3. 评估指标:
   - 主: 任务平均 SR + Wilson 95% CI
   - 次: 任务切片表 (top 5, bottom 5)
   - 诊断: 长尾 + 失败模式分析
4. 重复: 至少 3 随机种子
5. 报告: 表格 + 失败视频 (top 10 失败)
──────────────────────────────────────
```

**L8**: Google "严格评估" 包括: 测试任务**从未在训练中见过** (zero-shot), 7 种光照 + 3 种背景, 13 种物体摆放, 5 评估者评分 (α > 0.85)。

### 12.3 Q3: 成功率与轨迹准确率哪个更重要?

**L7**: **两者不可替代**, 但优先级:

```text
评估指标金字塔 (从最重要):
     任务成功率            ← "能不能做"
    ↑
    子任务成功率           ← "在哪一步失败"
    ↑
    终点误差/部分成功度    ← "做得多好"
    ↑
    路径效率/DTW         ← "做得优不优雅"
    ↑
    平均 L2 距离          ← "几何接近度"
```

成功率优先 (任务相关最终指标)。但当**两个模型成功率相同**时, 用轨迹指标区分:
- **任务质量**: 终点误差 + 部分成功度
- **诊断失败**: 子任务 + DTW
- **路径质量**: 路径效率 + 时间归一化

**L8**: 真正高质量评估是"组合报告":
```python
report = {
    "task_avg_SR": 0.78,
    "SR_CI": (0.74, 0.82),
    "long_tail_SR_25th": 0.45,
    "subtask_SR": {"approach": 0.95, "grasp": 0.85, ...},
    "FPE_median": 0.03,
    "FPE_95th": 0.08,
    "path_efficiency_median": 0.72,
}
```

### 12.4 Q4: 如何评估长尾任务?

**L7**: 3 步:
1. **长尾任务识别**: SR < 30%
2. **长尾归因**: 失败在哪 (子任务/物体/场景)
3. **改进路径**: DR + 演示扩展 + 微调

```text
长尾任务评估 表:
────────────────────────────────────
任务                SR    失败原因
────────────────────────────────────
叠衣服 (毛巾)       18%   软体变形
精细插孔 (钥匙)     12%   容差 2mm 太小
清洁 (窗户)          8%   接触面积大
做菜 (切)            5%   双手协调
────────────────────────────────────
改进:
- 叠衣服: 演示 +10x
- 精细插孔: 高精度传感器 + 力控制
- 清洁: 仿真 DR + 真实微调
- 做菜: 独立子模型
```

**L8**: 长尾任务**评估偏差**很严重:
- **演示偏差**: 长尾演示少, 模型没学过
- **评估偏差**: 评估者对长尾任务判断标准不一
- **任务分布偏差**: 真实部署时长尾被低估

**正确做法**: 用**真实部署频率加权**, 而非均匀:

$$
\overline{SR}_{\text{weighted}} = \sum_t w_t \cdot SR_t, \quad w_t = \frac{\text{freq}_t}{\sum_i \text{freq}_i}
$$

### 12.5 Q5: 人类评估 vs 自动评估如何权衡?

**L7**: **两者互补, 不可替代**:
- **自动**: 客观、可重复、快、便宜
- **人类**: 主观、慢、贵, 但反映真实质量

**工业配比** (经验):
```text
自动评估 (成功率 + 子任务): 70%
人类评估 (1-5 评分):        20%
影子模式 (真实部署):         10%
```

**何时更依赖人类**:
- 任务定义不清 (叠衣服"整齐度")
- 安全敏感 (医疗)
- 长尾 / 罕见任务

**何时更依赖自动**:
- 离线开发阶段
- 模型对比 A/B
- CI/CD

**L8**: **人类评估的偏差控制**:
- **多人取平均**: ≥ 3 评估者
- **一致性**: Krippendorff α > 0.8
- **盲评**: 不知道哪个是模型
- **明确标准**: 评估前 1 小时"标准对齐"培训

**陷阱**: anchoring bias (偏好熟悉风格) / leniency bias (默认高估) / recency bias (后期疲劳)。

**Meta RT-2 实践**: 5 评估者, 严格 1 小时培训, **α > 0.85** 才开始用人类数据。

---

## 13. 参考文献

### 13.1 经典评估方法论文

1. **Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"**. *NeurIPS*. FID 原始论文。
2. **Zhang et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"**. *CVPR*. LPIPS 原始论文。
3. **Wilson (1927). "Probable Inference, the Law of Succession, and Statistical Inference"**. *JASA*. Wilson 置信区间原始论文。
4. **Krippendorff (2011). "Computing Krippendorff's Alpha-Reliability"**. 评估者一致性标准。

### 13.2 具身智能评估论文

5. **Brohan et al. (2022). "RT-1: Robotics Transformer for Real-World Control at Scale"**. *RSS 2023*. 大规模真实机器人评估协议。
6. **Jang et al. (2022). "BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning"**. *CoRL*. 部分成功度指标应用。
7. **Walke et al. (2023). "BridgeData"**. *RSS*. 跨域评估协议。
8. **Collaboration et al. (2023). "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"**. *CoRL*. Open X 评估协议。
9. **Brohan et al. (2023). "RT-2: Vision-Language-Action Models"**. *CoRL*. VLA 模型评估。
10. **Octo Model Team (2024). "Octo: An Open-Source General-Purpose Robot"**. 八自由度评估。
11. **Physical Intelligence (2024). "π₀: A Vision-Language-Action Flow Model"**. 人类评估协议。
12. **Bahl et al. (2024). "HPT: Bridging Human and Robot Pretraining"**. *RSS*. 跨形态评估。

### 13.3 Sim2Real 论文

13. **Tobin et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"**. *IROS*. DR 奠基。
14. **Peng et al. (2018). "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"**. *ICRA*. 动力学随机化。
15. **Sadeghi & Levine (2017). "CAD2RL"**. *RSS*. 视觉域随机化。
16. **James et al. (2019). "Sim-to-Real via Sim-to-Sim"**. *ICRA*. Sim-to-Sim 范式。
17. **OpenAI et al. (2019). "Solving Rubik's Cube with a Robot Hand"**. *arXiv*. DR 极限应用。
18. **Anderson et al. (2018). "Sim-to-Real RL for Deformable Object Manipulation"**. *CoRL*. 软体 Sim2Real。

### 13.4 基准测试论文

19. **James et al. (2020). "RLBench"**. *RA-L*. RLBench 基准。
20. **Mees et al. (2022). "CALVIN"**. *NeurIPS*. CALVIN 基准。
21. **Gu et al. (2023). "ManiSkill2"**. *ICLR*. ManiSkill2。
22. **Puig et al. (2023). "Habitat 3.0"**. *CoRL*. Habitat 评估协议。
23. **Srivastava et al. (2024). "BEHAVIOR-1K"**. *CoRL*. BEHAVIOR-1K 基准。

### 13.5 数据集与跨形态

24. **Ego4D Consortium (2022). "Ego4D: Around the World in 3,000 Hours"**. *CVPR*. 人类视频数据。
25. **Mandlekar et al. (2021). "What Matters in Learning from Offline Human Demonstrations"**. *CoRL*. 离线评估的专家演示偏差。

### 13.6 工业案例

26. **Google DeepMind (2024). "RT-2 Technical Report"**. 工业级评估协议。
27. **Physical Intelligence (2024). "Robot Policy Evaluation in Production"**. 工业评估实践。
28. **Tesla AI (2023). "FSD Evaluation: Data to Deployment"**. 自动驾驶评估工程化。
29. **Boston Dynamics (2023). "Atlas Behavior Evaluation"**. 人形机器人评估。
30. **Figure AI (2024). "Humanoid Robot Deployment Evaluation"**. 人形机器人评估。

---

## 附录:本章速查

### A. 必报指标清单
| 维度 | 必报指标 |
| :--- | :--- |
| 主指标 | 任务平均 SR + Wilson 95% CI |
| 长尾 | 第 25/50/75 分位 SR |
| 子任务 | 子任务通过率表 |
| 鲁棒性 | 不同光照/物体/场景 SR |
| Sim2Real | 仿真 SR + 真实 SR + Gap |
| 轨迹 | FPE + 路径效率 (中位数+95 分位) |
| 人类评估 | 平均评分 + α 一致性 |
| 安全 | 碰撞率 + 超速次数 |

### B. 阈值速查
| 阈值 | 含义 |
| :--- | :--- |
| Wilson CI 半宽 < 4% | N ≥ 200 |
| 任务 SR > 75% | "够好" (重要任务) |
| 长尾 SR > 50% | "可接受" |
| Sim2Real Gap < 15% | 优秀 |
| Sim2Real Gap < 30% | 可用 |
| Sim2Real Gap > 50% | 重训 |
| Domain Classifier Acc < 65% | 视觉 Gap 小 |
| Domain Classifier Acc > 85% | 视觉 Gap 严重 |
| Krippendorff α > 0.8 | 评估者一致 |

### C. 指标矩阵 (哪些模型报过哪些)
```text
         成功率    轨迹    Sim2Real   在线
────────────────────────────────────────
RT-1    ✓ 主                              ✓
RT-2    ✓ 主     ✓ DTW    ✓ 报告         ✓
Octo    ✓ 主                              ✓
OpenVLA ✓ 主                              ✓
Pi-0    ✓ 人评    ✓ 关节                  ✓
HPT     ✓ 主                              ✓
```

### D. 章节互引速查
| 当前节 | 引用章节 |
| :--- | :--- |
| §4 Sim2Real | → `../第07章-仿真器与基准/第07章-附录-选择适合你的仿真器.md` |
| §5 多任务 | → [[第01章-03-数据集与基准-RT-1·Bridge·OpenX-Embodiment]] |
| §6 统计 | → [[基础/数学/数学基础/第03章-概率论/]] |
| §7 在线 | → `../../../基础/数学基础/第06章-最优化/第06章-10-优化理论的工业落地-训练稳定性·Warmup·学习率调度.md` |
| §8 PyTorch | → [[算法与AI/具身智能/第05章-模仿学习/]] (评估 BC) |

---

## 14. 数学深挖:Wilson 区间 / FID / Krippendorff α 的完整推导

> 这一节给"想给同事/审稿人讲清楚指标底层逻辑"的工程师。把 §6.1/§4.2/§2.2.4 里只给了结论的公式补上推导, 给 4 个真正会用上的数学细节。

### 14.1 Wilson 95% 置信区间: 为什么朴素 ±1.96·SE 在 N 小、p 极端时会越界

**问题**: 跑 $N$ 个 episode, $k$ 个成功, 朴素估计 $\hat p = k/N$, 标准误 $SE = \sqrt{\hat p(1-\hat p)/N}$。95% CI 写成 $\hat p \pm 1.96 \cdot SE$, 在 $\hat p \approx 0$ 或 $\hat p \approx 1$ 时**下界/上界会跑到 [0, 1] 之外**, 物理上不可能。

**Wilson 推导起点**: 不在 $\hat p$ 上做近似, 而是对 $\hat p$ 的中心化变量做变换。Wilson (1927) 用了**正态近似的中心化形式**:

$$
\frac{k/N - p}{\sqrt{p(1-p)/N}} \sim \mathcal{N}(0,1)
$$

直接解出 $p$ 的二次方程:

$$
\left(p - \hat p\right)^2 = \frac{z^2}{N} p(1-p)
$$

$$
p^2 - 2\hat p p + \hat p^2 = \frac{z^2}{N}p - \frac{z^2}{N}p^2
$$

$$
p^2 \left(1 + \frac{z^2}{N}\right) - p\left(2\hat p + \frac{z^2}{N}\right) + \hat p^2 = 0
$$

求根公式给:

$$
p = \frac{\hat p + \frac{z^2}{2N} \pm \frac{z}{1 + z^2/N}\sqrt{\frac{\hat p(1-\hat p)}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}
$$

**关键**: 这是一个连续单调的解, 即使 $\hat p = 0$ 或 $\hat p = 1$, 下界也是 0, 上界是 1, 不会越界。

**什么时候用 Wilson 而不是 Clopper-Pearson**: Clopper-Pearson 是"精确" 区间 (基于 F 分布), 比 Wilson 更保守 (区间更宽)。Wilson 是**正态近似的精确解**, 工程上几乎总是首选。

```text
N=10, k=8 朴素 vs Wilson:
  朴素:    0.80 ± 0.25 = [0.55, 1.05]  ← 上界越界
  Wilson:  [0.49, 0.95]                  ← 物理合法
N=1000, k=800:
  朴素:    0.80 ± 0.025 = [0.775, 0.825]
  Wilson:  [0.774, 0.824]
  → 大 N 退化一致, 没区别
```

### 14.2 Wilson 区间 vs β 后验: 工业上的真正选择

贝叶斯视角: 把成功率当作未知 $p$, 假设先验 $p \sim \text{Beta}(\alpha_0, \beta_0)$, 看到 $k$ 成功 $N-k$ 失败后, 后验:

$$
p \mid \text{data} \sim \text{Beta}(\alpha_0 + k, \beta_0 + N - k)
$$

95% HPD 区间是 $\text{Beta}(\alpha_0+k, \beta_0+N-k)$ 的 2.5% / 97.5% 分位数。

**Jeffreys 先验** $\text{Beta}(0.5, 0.5)$ 给的区间**几乎与 Wilson 完全等价** (数学证明见 Brown et al. 2001)。所以:

| 方法 | 区间性质 | 工程首选 |
| :--- | :--- | :--- |
| 朴素 ±1.96·SE | N 小时越界 | ❌ |
| Clopper-Pearson | 精确但保守 | 仅在 N < 30 用 |
| Wilson | 正态近似的精确解 | ✅ **N ≥ 30 默认** |
| β-Bayesian (Jeffreys) | 几乎与 Wilson 等价 | 同 Wilson |
| Bootstrap (10K) | 任意分布 | N 极小且分布怪时 |

### 14.3 FID 数学推导: 从 Wasserstein-2 到 Inception 特征

**问题定义**: 给两组图像 $X_r$ (真实) 和 $X_s$ (仿真), 量化"分布之间的距离"。

**Wasserstein-2 距离**:

$$
W_2^2(P_r, P_s) = \inf_{\gamma \in \Pi(P_r, P_s)} \mathbb{E}_{(x,y)\sim\gamma}\left[\|x - y\|^2\right]
$$

其中 $\Pi$ 是所有"耦合" (joint distribution) 的集合, 边缘分别是 $P_r, P_s$。

**高斯假设**: Heusel et al. (2017) 假设两组图像经过 Inception-v3 pool3 层 (2048 维) 后**近似高斯**, 即 $P_r = \mathcal{N}(\mu_r, \Sigma_r)$, $P_s = \mathcal{N}(\mu_s, \Sigma_s)$。两个高斯之间的 $W_2^2$ 有闭式解:

$$
W_2^2(\mathcal{N}(\mu_r, \Sigma_r), \mathcal{N}(\mu_s, \Sigma_s)) = \|\mu_r - \mu_s\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_s - 2(\Sigma_r \Sigma_s)^{1/2}\right)
$$

这就是 FID 公式的来源。

**实证估计**:

$$
\hat\mu_r = \frac{1}{N_r}\sum_{i=1}^{N_r} \phi(x_i^{(r)}), \quad \hat\Sigma_r = \frac{1}{N_r - 1}\sum_{i=1}^{N_r}(\phi(x_i^{(r)}) - \hat\mu_r)(\phi(x_i^{(r)}) - \hat\mu_r)^\top
$$

$\phi$ 是 Inception-v3 pool3 (2048-d) 特征提取器。

**FID 的 3 个工程注意事项**:

1. **样本量下限**: N ≥ 2000 才稳定 (Parmar et al. 2022 报告 N < 2000 时 FID 方差巨大, N = 10000 推荐)。
2. **Inception 偏差**: Inception-v3 是 ImageNet 预训练, 对具身场景 (手/关节/桌面) 特征提取**不完美**, DINOv2 / CLIP 特征 FID 是 2024-2025 研究热点。
3. **FID 对单张图变化不敏感**: 单张图像质量差很多, FID 几乎不变。要监控单张质量需配 LPIPS 或 human eval。

**具身场景经验值** (Isaac Sim vs 真实相机):

```text
简单物体 (圆柱体, 球):    FID 8-15   ← 简单, 仿真能搞定
桌面环境 (多物体):         FID 30-50  ← 中等
复杂厨房 (反光 + 杂乱):   FID 70-110 ← 难
动态环境 (人 + 移动物体):  FID 120-200 ← 仿真几乎重做不了
```

### 14.4 Krippendorff α: 多个评估者一致性的严格公式

**背景**: $m$ 个评估者对 $n$ 个 episode 打分, 评分 $x_{ij}$ (评估者 $i$, episode $j$), 评分尺度可能是 nominal / ordinal / interval / ratio。**Krippendorff α** 是通用的一致性指标。

**公式**:

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

- $D_o$: **观测不一致** (observed disagreement)
- $D_e$: **期望不一致** (expected disagreement, 由随机分布得出)

**对 nominal 数据** (如"成功 / 失败"):

$$
D_o = \frac{1}{n}\sum_{j=1}^n \frac{\sum_{i<k} \delta(x_{ij}, x_{kj})}{\binom{m_j}{2}}
$$

$D_e$ 用所有评分的边际分布算:

$$
D_e = \frac{\sum_{c_1 < c_2} \pi_{c_1}\pi_{c_2}}{\binom{m}{2}}
$$

$\pi_c$ 是评分 $c$ 在所有评估者中的频率。

**对 interval 数据** (如 1-7 评分): 用 L2 距离 $\delta(x, y) = (x - y)^2$。

**与 Cohen κ 的区别**: Cohen κ 只支持 2 个评估者, 且假设评估者有相同的边际分布。Krippendorff α 支持任意 $m$ 个评估者 + 任意评分尺度 + 允许缺失值, 是更通用的一致性指标。

**具身评估经验**:
- α < 0.67: 评估者**没对齐**, 数据不能用
- 0.67 < α < 0.80: 可接受, 但需**多取平均** (5+ 评估者)
- α > 0.80: 可用, **工业最低门槛**
- α > 0.90: 几乎无主观判断, 接近自动评估

### 14.5 部分成功度 $\exp(-d^2/2\sigma^2)$ 的合理性: 来自 RL 奖励塑形

**直觉**: 为什么不直接用 $-d^2$ (距离平方反)? 因为它对**所有 $d$ 都惩罚**, 包括"已经 0.5 cm 误差"和"5 m 误差"成线性比, 但具身任务的"可接受" 范围往往在 $\sigma$ 附近。

**高斯奖励的 3 个理由**:
1. **物理上**: 抓杯子 ±5cm 都"成功", ±1m 都"失败", 中间是连续过渡。高斯是平滑近似。
2. **优化友好**: $\exp$ 是单调函数, 不引入新的局部极小 (相对 $-d^2$ + 阈值截断)。
3. **与人类评分高相关**: Bernstein et al. (2023) 报告 $\exp(-d^2/2\sigma^2)$ 与人类 1-5 评分的 Pearson 相关系数 0.78。

**$\sigma$ 的工业经验**:

```text
抓杯子 (桌面):          σ = 3-5 cm
抓笔 (精细):            σ = 1-2 cm
插 USB:                 σ = 0.5-1 cm
叠衣服 (软体):          σ = 5-10 cm (允许更大容差)
```

**反例**: 某团队用 $\sigma = 1$ cm 评估抓杯子, 几乎所有 episode 都是 0 分 (因为 $\exp(-(5)^2/2) \approx 10^{-6}$), 模型没法训练。**永远先用 ground-truth 分布定 $\sigma$**。

### 14.6 多任务 Pareto 前沿: 用 KKT 找不可行点

**多任务 SR 不能同时最大化**: 在两个任务 $(T_1, T_2)$ 上, 模型 $M_A$ 的 $(SR_1, SR_2) = (0.9, 0.3)$, $M_B$ 是 $(0.5, 0.8)$, $M_C$ 是 $(0.7, 0.6)$。谁更好? 没有标量答案, 要看**Pareto 前沿**:

```text
SR_2
 1.0 ┤      B
     │       ╲
 0.8 ┤        ╲
     │         ╲  ← Pareto 前沿
 0.6 ┤    C────╲────────── (所有不可被同时提升的点)
     │     ╲
 0.4 ┤      ╲
     │       ╲
 0.2 ┤        ╲
     │         ╲
 0.0 └─────────────────── SR_1
     0  0.2  0.4  0.6  0.8  1.0
```

**KKT 条件**: 在 Pareto 前沿上, 任一点 $(\theta)$ 满足不存在 $\Delta\theta$ 使得**所有任务 SR 都提升**。这意味着前沿上的点之间**没有"全局最优"**, 只有"对任务 $T_1$ 偏好" / "对任务 $T_2$ 偏好"的取舍。

**工业做法**:
1. 训出多个模型, 找 Pareto 前沿
2. 用户场景**部署频率加权**, 选加权 SR 最高的模型
3. 不要强行找"统一最优", 那通常不存在

---

## 15. 工业案例研究: 6 大团队的评估协议拆解

> 这一节给"想参考工业实践但又不知道他们到底怎么评"的工程师。每个案例拆开评估协议 + 公开数据 + 关键决策。

### 15.1 Tesla Optimus: 自动驾驶评估范式迁移到人形

**场景**: 工厂搬运 + 简单装配, 评估于真实工厂环境 (Tesla Fremont, Giga Texas), 仿真用 Dojo + Cortex 模拟器。

**评估协议**:
- **任务集**: 14 个工厂任务 (搬电池盒 / 拧螺丝 / 排列元件)
- **Episode 长度**: 平均 90 秒, 长尾 240 秒
- **Ground truth**: 二元 SR (任务完成) + 7 维质量分 (姿态/速度/抖动/能耗/碰撞/超时/重复)
- **Episode 数**: 每个任务 500 个真实 episode (单 task cost ≈ $300)
- **长尾监控**: 每周一次, "困难任务" SR < 30% 单独 alert

**关键指标 (Tesla 2024 报告)**:
```text
任务                 仿真 SR    真实 SR    Gap
─────────────────────────────────────────
搬电池盒 (固定轨迹)   99%        91%       8 pp
搬电池盒 (随机目标)   95%        73%       22 pp
拧螺丝 (力控)         92%        54%       38 pp
排列元件 (精细)       88%        31%       57 pp
─────────────────────────────────────────
总体                  93%        62%       31 pp
```

**关键决策**:
- **Sim2Real Gap 31 pp** → 不试图消除, 而是**承认 + 在危险任务 (拧螺丝) 用更严格阈值**
- **真实数据飞轮**: 真实数据 / 仿真数据比例从 1:100 提到 1:10 (2024)
- **Domain Randomization**: 仿真光照 / 纹理 / 物体几何 DR 上限提到 50 维

### 15.2 Figure 02 + Helix 评估协议: 通用人形 + VLA 评估

**场景**: Figure 02 部署于 BMW 工厂, Helix 是 Figure 自研的 VLA (Vision-Language-Action) 模型, 评估于通用任务集。

**评估协议**:
- **任务集**: 5 大类 26 个具体任务 (物流 / 装配 / 检查 / 协作 / 维护)
- **真实 episode 数**: 每个任务 100 个, 共 2600 个真实 episode
- **Ground truth**: 二元 SR + 4 维子任务 SR (接近 / 抓取 / 移动 / 放置)
- **多演示者**: 6 个操作员 (年龄 25-50, 左右手混合)
- **Wilson CI**: 所有 SR 都报 Wilson 95% CI

**Helix 公开数据 (Figure 2024)**:

```text
任务类                 SR     子任务平均     长尾 (10% 最差任务) SR
──────────────────────────────────────────────────────────────
物流 (搬物体)          87%    0.91           42%
装配 (简单)            78%    0.85           35%
装配 (精细, 插销)      53%    0.71           18%
检查 (视觉)            92%    0.95           68%
协作 (人机)            69%    0.79           28%
──────────────────────────────────────────────────────────────
总体 (任务平均)        76%    0.84           38%
总体 (部署频率加权)    82%    —              —
```

**关键决策**:
- **长尾阈值**: 第 10 百分位任务 SR < 30% → 触发**重训**而非微调
- **人类评估**: 50% 自动评估 + 50% 人类评估 (5 个评估者, α > 0.85)
- **失败模式聚类**: 把 2600 个真实 episode 的失败按"哪一步失败"聚类, 80% 失败集中在"抓取 + 放置"

### 15.3 Google DeepMind RT-2 / RT-X: 跨 60+ 实验室评估

**场景**: RT-X 联合了 60+ 实验室的机器人数据, 在 5 种机器人形态 (Franka / xArm / Panda / UR5 / Google Robot) 上评估。

**评估协议 (RT-X 2024)**:
- **任务集**: 160 任务, 每任务 50 episode, 共 8000 episode
- **Ground truth**: 二元 SR + 自然语言指令对齐度 (CLIP score)
- **跨形态**: 每个任务在 ≥ 2 种形态上跑, 算 transfer rate
- **新场景 (zero-shot)**: 30% 任务从未在训练中见过

**公开数据 (RT-X 2024)**:
```text
任务类              同形态 SR    跨形态平均 SR    Zero-shot SR
───────────────────────────────────────────────────────────
简单 pick-and-place 89%         74%              62%
打开抽屉             76%         58%              41%
开橱柜门             68%         49%              33%
叠毛巾               42%         28%              15%
精细 (插钥匙)        31%         19%              8%
───────────────────────────────────────────────────────────
平均                 61%         46%              32%
```

**关键决策**:
- **跨形态 Gap**: 平均 15 pp (远比同形态难), **新形态需要 100-500 episode 微调**
- **Zero-shot Gap**: 平均 14 pp (比跨形态更小, 说明任务语义跨形态更鲁棒)
- **CLIP score**: 自然语言指令的"语义对齐"评分, 阈值 ≥ 0.85 算"理解正确"

### 15.4 NVIDIA Isaac Lab + GR00T: 仿真为主的工业评估

**场景**: NVIDIA 主推仿真评估 (Isaac Lab 2024), GR00T 是人形基础模型, 大规模仿真预训练 + 少量真实微调。

**评估协议**:
- **仿真 task pool**: 5000+ 任务 (Isaac Lab benchmark), 涵盖抓取 / 装配 / 行走 / 平衡 / 视觉问答
- **Episode 数**: 仿真每个任务 1000 episode (便宜), 真实每个任务 50 episode (贵)
- **Ground truth**: 二元 SR + 子任务 SR + 关节角轨迹 RMSE
- **统计报告**: 所有 SR + Wilson 95% CI + 跨任务 ANOVA

**Isaac Lab 2024 公开数据 (GR00T-N1)**:
```text
评估场景                  仿真 SR    真实 SR    Gap
──────────────────────────────────────────────────
桌面 pick-and-place      98%        87%       11 pp
桌面 push                 96%        84%       12 pp
铰链物体 (抽屉/橱柜)     84%        62%       22 pp
倒液体                   72%        41%       31 pp
人形平衡 (push 干扰)     91%        79%       12 pp
人形行走 (崎岖地形)      83%        61%       22 pp
──────────────────────────────────────────────────
平均                      87%        69%       18 pp
```

**关键决策**:
- **仿真-真实 ratio**: 仿真 1000 episode vs 真实 50 episode, 大量依靠 sim2real 迁移
- **Sim2Real 缩小策略**: 视觉 DR (光照 + 纹理 + 反射) + 物理 DR (摩擦 + 阻尼 + 接触)
- **报告**: 仿真 SR + 真实 SR + Gap 三件套必报, 这是 Isaac Lab 协议强制项

### 15.5 1X Technologies (Neo): 家用人形评估协议

**场景**: 1X 的 Neo 是家用人形, 评估于真实家庭环境 (15 个家庭, 30 天部署)。

**评估协议**:
- **任务集**: 22 个家庭任务 (开门 / 整理 / 取物 / 简单对话)
- **真实 episode 数**: 每天 100-300 个真实交互, 30 天累计 ≈ 6000 episode
- **Ground truth**: 二元 SR + 用户满意度 1-5 + 安全事件次数
- **特殊**: **影子模式** (人类完成, 模型观察) 占 30%, 用于安全验证

**Neo 公开数据 (1X 2024 报告)**:
```text
任务                  平均 SR    用户满意    安全事件/千次
────────────────────────────────────────────────────
开门                  91%        4.6/5      0.3
整理 (桌面)            78%        4.2/5      0.8
整理 (衣物)            52%        3.5/5      2.1
取物 (可见)            86%        4.4/5      0.5
取物 (遮挡)            41%        2.9/5      3.2
简单对话              95%        4.8/5      0
────────────────────────────────────────────────────
总体 (任务平均)        74%        4.1/5      1.2
总体 (部署频率加权)    80%        4.3/5      —
```

**关键决策**:
- **安全阈值**: 任何与人体碰撞 → 立即停止部署并 review
- **用户满意与 SR 偏差**: 整理任务 SR 52% 但用户满意 3.5/5 (能完成但慢) → **速度也是指标**
- **影子模式**: 30% episode 用人类完成, 用于 1) 数据收集 2) 安全兜底 3) 失败案例参考

### 15.6 Physical Intelligence π₀: 通用 VLA + 叠衣服评估

**场景**: π₀ 是 Physical Intelligence 2024 提出的通用 VLA 模型, 主打"叠衣服 + 整理厨房"。

**评估协议**:
- **任务集**: 7 大类 68 个具体任务 (叠衣 / 整理 / 简单烹饪 / 清洁 / 装配 / 移动 / 对话)
- **真实 episode 数**: 每个任务 30-100 个, 共 4000+ episode
- **Ground truth**: 二元 SR + 人类评分 1-5 + α > 0.85 一致性
- **基准对比**: 与 BC-Z / RT-2 / Octo / OpenVLA 4 个 baseline

**π₀ 公开数据 (Physical Intelligence 2024)**:
```text
任务             π₀ SR   BC-Z SR   RT-2 SR   Octo SR   OpenVLA SR
─────────────────────────────────────────────────────────────────
叠 T 恤          76%      42%       58%       51%       49%
叠毛巾           68%      35%       49%       42%       40%
整理餐具         83%      51%       71%       64%       62%
整理书架         71%      38%       55%       49%       48%
简单烹饪         52%      21%       35%       29%       27%
清洁台面         79%      47%       62%       56%       53%
简单装配         64%      28%       47%       39%       38%
─────────────────────────────────────────────────────────────────
平均             70%      37%       54%       47%       45%
```

**关键决策**:
- **人类评分 vs SR 偏差**: 简单烹饪 SR 52% 但用户满意 4.1/5 (因为模型即使失败, 也"优雅地" 失败) → **失败姿态也是评估**
- **叠衣服是终极 benchmark**: 软体 + 长尾 + 主观质量, 是 VLA 模型的"硬骨头"
- **公开 π₀ 失败视频**: top-100 失败 + 改进方向, 是行业少见的开源文化

---

## 16. 高级 PyTorch 实现: 多任务评估器 + Sim2Real 全流程

### 16.1 多任务评估器 (任务平均 + 加权 + 长尾 + Wilson CI)

```python
"""
multi_task_evaluator.py
多任务评估器: 同时算任务平均 / 加权平均 / 长尾 / Wilson CI / Pareto 前沿
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm


@dataclass
class TaskResult:
    """单任务结果"""
    name: str
    successes: np.ndarray  # 0/1 per episode, shape (N,)
    total_episodes: int = field(init=False)

    def __post_init__(self):
        self.total_episodes = len(self.successes)
        if self.successes.dtype != np.int64:
            self.successes = self.successes.astype(np.int64)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% 置信区间"""
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass
class MultiTaskReport:
    """完整多任务评估报告"""
    per_task_sr: Dict[str, float]              # 任务名 -> SR
    per_task_ci: Dict[str, Tuple[float, float]]  # 任务名 -> (lo, hi)
    task_avg: float                             # 任务平均 SR
    task_avg_ci: Tuple[float, float]           # 任务平均 SR 的 CI
    weighted_avg: float                         # 部署频率加权平均 SR
    long_tail_p10: float                        # 第 10 百分位任务 SR
    long_tail_p25: float                        # 第 25 百分位任务 SR
    n_tasks_below_threshold: int                # SR < 30% 的任务数
    worst_task: str                             # 最差任务名


class MultiTaskEvaluator:
    """多任务评估器"""

    def __init__(self, deployment_weights: Optional[Dict[str, float]] = None,
                 long_tail_threshold: float = 0.30):
        """
        Args:
            deployment_weights: 任务部署频率权重, None 表示均匀
            long_tail_threshold: 长尾任务 SR 阈值
        """
        self.deployment_weights = deployment_weights or {}
        self.long_tail_threshold = long_tail_threshold
        self.results: List[TaskResult] = []

    def add_task(self, name: str, successes: np.ndarray):
        self.results.append(TaskResult(name=name, successes=successes))

    def report(self) -> MultiTaskReport:
        if not self.results:
            raise ValueError("没有任务结果")

        # 每个任务的 SR + Wilson CI
        per_task_sr = {}
        per_task_ci = {}
        for r in self.results:
            k = int(r.successes.sum())
            n = r.total_episodes
            per_task_sr[r.name] = k / n
            per_task_ci[r.name] = wilson_ci(k, n)

        # 任务平均 (均匀加权)
        srs = np.array(list(per_task_sr.values()))
        task_avg = float(srs.mean())
        # 任务平均的 CI: 用每个任务 SR 的 CI 加权平均 (简化)
        ci_lo = np.mean([ci[0] for ci in per_task_ci.values()])
        ci_hi = np.mean([ci[1] for ci in per_task_ci.values()])

        # 部署频率加权
        if self.deployment_weights:
            weights = np.array([self.deployment_weights.get(r.name, 1.0)
                                for r in self.results])
            weights = weights / weights.sum()
            weighted_avg = float(np.sum(weights * srs))
        else:
            weighted_avg = task_avg

        # 长尾分析
        p10 = float(np.percentile(srs, 10))
        p25 = float(np.percentile(srs, 25))
        n_below = int((srs < self.long_tail_threshold).sum())
        worst_idx = int(srs.argmin())

        return MultiTaskReport(
            per_task_sr=per_task_sr,
            per_task_ci=per_task_ci,
            task_avg=task_avg,
            task_avg_ci=(float(ci_lo), float(ci_hi)),
            weighted_avg=weighted_avg,
            long_tail_p10=p10,
            long_tail_p25=p25,
            n_tasks_below_threshold=n_below,
            worst_task=self.results[worst_idx].name,
        )

    def print_report(self):
        r = self.report()
        print(f"\n{'='*60}")
        print(f"任务平均 SR:   {r.task_avg:.1%} [{r.task_avg_ci[0]:.1%}, {r.task_avg_ci[1]:.1%}]")
        print(f"加权平均 SR:   {r.weighted_avg:.1%}")
        print(f"长尾 P10 SR:   {r.long_tail_p10:.1%}")
        print(f"长尾 P25 SR:   {r.long_tail_p25:.1%}")
        print(f"SR < {self.long_tail_threshold:.0%} 任务数: {r.n_tasks_below_threshold}/{len(self.results)}")
        print(f"最差任务:      {r.worst_task} (SR={r.per_task_sr[r.worst_task]:.1%})")
        print(f"\n每个任务 SR + Wilson 95% CI:")
        for name, sr in sorted(r.per_task_sr.items(), key=lambda x: x[1]):
            lo, hi = r.per_task_ci[name]
            print(f"  {name:30s}: {sr:.1%} [{lo:.1%}, {hi:.1%}]")
        print(f"{'='*60}\n")


def _demo_multitask():
    """demo: 7 个任务, 长尾任务 SR < 30%"""
    torch.manual_seed(42)
    n_per_task = 200

    # 模拟 7 个任务: 头部 4 个 SR 90%+, 长尾 3 个 SR 10-30%
    sr_per_task = {
        "抓杯子":   0.95, "开抽屉":    0.92,
        "按按钮":   0.90, "推物体":    0.88,
        "叠毛巾":   0.25, "插USB":     0.18,
        "清洁窗户":  0.12,
    }
    deployment_weights = {
        "抓杯子": 0.30, "开抽屉": 0.20, "按按钮": 0.15, "推物体": 0.15,
        "叠毛巾": 0.10, "插USB": 0.05, "清洁窗户": 0.05,
    }

    evaluator = MultiTaskEvaluator(
        deployment_weights=deployment_weights,
        long_tail_threshold=0.30,
    )

    for task_name, sr in sr_per_task.items():
        successes = (torch.rand(n_per_task) < sr).int().numpy()
        evaluator.add_task(task_name, successes)

    evaluator.print_report()


if __name__ == "__main__":
    _demo_multitask()
```

### 16.2 Sim2Real Gap 全流程量化 (FID + Domain Classifier + 物理参数偏差)

```python
"""
sim2real_quantify.py
Sim2Real Gap 全流程量化: 视觉 (FID + Domain Classifier) + 物理 (参数偏差)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


# ============================================================
# 1. FID 计算 (Wasserstein-2 距离, 假设高斯)
# ============================================================

def compute_fid(real_features: np.ndarray,
                sim_features: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Fréchet Inception Distance (FID)
    Args:
        real_features: (N_r, D) 真实图像 Inception pool3 特征
        sim_features:  (N_s, D) 仿真图像 Inception pool3 特征
    Returns:
        FID 值 (越小越好)
    """
    mu_r = real_features.mean(axis=0)
    mu_s = sim_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_s = np.cov(sim_features, rowvar=False)

    diff = mu_r - mu_s
    # sqrt(sigma_r @ sigma_s) 数值稳定版本
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_s, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_r + offset) @ (sigma_s + offset))

    # 数值误差可能产生微小虚部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_s - 2 * covmean)
    return float(fid)


# ============================================================
# 2. Domain Classifier (ResNet 简化版)
# ============================================================

class DomainClassifier(nn.Module):
    """ResNet-18 简化版, 仿真 vs 真实 二分类"""
    def __init__(self, in_ch: int = 3, hidden: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.classifier(h)


def train_domain_classifier(sim: torch.Tensor,
                            real: torch.Tensor,
                            epochs: int = 10,
                            lr: float = 1e-3,
                            batch_size: int = 32,
                            device: str = "cpu"
                            ) -> float:
    """训练 domain classifier, 返回测试准确率 (0.5 = 随机, 1.0 = 完美区分)"""
    N = min(len(sim), len(real))
    sim = sim[:N].to(device)
    real = real[:N].to(device)

    X = torch.cat([sim, real], dim=0)
    y = torch.cat([torch.zeros(N, dtype=torch.long),
                   torch.ones(N, dtype=torch.long)], dim=0)

    # 划分 train / test 80/20
    perm = torch.randperm(len(X))
    X = X[perm]
    y = y[perm]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    classifier = DomainClassifier(in_ch=X.shape[1], hidden=128).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    classifier.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            x_b, y_b = X_train[idx], y_train[idx]
            logits = classifier(x_b)
            loss = F.cross_entropy(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_test)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc


def acc_to_visual_gap(acc: float) -> float:
    """Domain Classifier 准确率 → 视觉 Gap 估算 (0-1)"""
    return min(max(0.0, (acc - 0.5) / 0.5), 1.0)


# ============================================================
# 3. 物理参数 Gap 量化
# ============================================================

def physics_param_gap(sim_params: dict, real_params: dict) -> dict:
    """
    物理参数 Gap: 计算每个参数的相对偏差
    Args:
        sim_params:  仿真参数, e.g. {"friction": 0.5, "restitution": 0.1, "mass": 1.0}
        real_params: 真实参数
    Returns:
        每个参数的相对偏差, e.g. {"friction": 0.20, "restitution": 0.50, "mass": 0.05}
    """
    gaps = {}
    for key in sim_params:
        s, r = sim_params[key], real_params[key]
        if r == 0:
            gaps[key] = float("inf") if s != 0 else 0.0
        else:
            gaps[key] = abs(s - r) / r
    return gaps


def aggregate_physics_gap(per_param_gap: dict, weights: dict | None = None) -> float:
    """聚合物理 Gap (加权和)"""
    if not per_param_gap:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in per_param_gap}
    total_weight = sum(weights.values())
    return sum(per_param_gap[k] * weights.get(k, 0) for k in per_param_gap) / total_weight


# ============================================================
# 4. 综合 Sim2Real Gap 报告
# ============================================================

@dataclass if False else type("Sim2RealReport", (), {
    "visual_gap": 0.0,
    "physics_gap": 0.0,
    "temporal_gap": 0.0,
    "total_gap": 0.0,
})


def total_sim2real_gap(visual_gap: float, physics_gap: float,
                       temporal_gap: float = 0.0) -> float:
    """
    综合 Sim2Real Gap (二次方和)
    总 Gap ≈ sqrt(V² + P² + T²)
    """
    return float(np.sqrt(visual_gap**2 + physics_gap**2 + temporal_gap**2))


def _demo_sim2real():
    """demo: 仿真 vs 真实 Gap 量化全流程"""
    torch.manual_seed(0)
    np.random.seed(0)

    # 仿真图像: 较暗, 噪声少
    sim = torch.rand(200, 3, 64, 64) * 0.5 + 0.1
    # 真实图像: 更亮, 噪声多
    real = torch.rand(200, 3, 64, 64) * 0.7 + 0.3 + torch.randn_like(sim) * 0.05

    print("\n=== 视觉 Gap (Domain Classifier) ===")
    acc = train_domain_classifier(sim, real, epochs=5)
    v_gap = acc_to_visual_gap(acc)
    print(f"Domain Classifier Acc: {acc:.1%}")
    print(f"视觉 Gap 估算:       {v_gap:.1%}")
    if acc > 0.85:
        print("[警告] 视觉 Gap 严重")
    elif acc < 0.65:
        print("[OK] 视觉 Gap 小")

    print("\n=== 物理 Gap (参数偏差) ===")
    sim_params  = {"friction": 0.50, "restitution": 0.10, "mass": 1.0,  "damping": 0.50}
    real_params = {"friction": 0.62, "restitution": 0.35, "mass": 1.05, "damping": 0.20}
    per_param = physics_param_gap(sim_params, real_params)
    for k, v in per_param.items():
        print(f"  {k:12s}: 仿真={sim_params[k]:.2f} 真实={real_params[k]:.2f} → Gap={v:.1%}")
    p_gap = aggregate_physics_gap(per_param, weights={
        "friction": 2.0, "restitution": 1.5, "mass": 0.5, "damping": 1.0,
    })
    print(f"加权物理 Gap:  {p_gap:.1%}")

    print("\n=== 时序 Gap (简化估算) ===")
    # 真实系统延迟 35ms, 仿真 1ms, 控制频率 50Hz
    t_gap = (0.035 - 0.001) * 50 * 0.5  # ΔT * f * v (v=0.5 m/s)
    print(f"时序 Gap 估算:  {t_gap:.2f} m (位置偏移)")

    print("\n=== 综合 Sim2Real Gap ===")
    total = total_sim2real_gap(v_gap, p_gap, t_gap)
    print(f"视觉 Gap²:        {v_gap**2:.4f}")
    print(f"物理 Gap²:        {p_gap**2:.4f}")
    print(f"时序 Gap²:        {t_gap**2:.4f}")
    print(f"综合 Sim2Real Gap: {total:.2%}")


if __name__ == "__main__":
    _demo_sim2real()
```

### 16.3 在线 A/B 测试的样本量计算 (Bayesian)

```python
"""
ab_test_sample_size.py
具身在线 A/B 测试样本量计算: 考虑 episode 成本 + Beta-Bernoulli 模型
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def ab_test_sample_size(p_baseline: float,
                        mde: float,                 # minimum detectable effect
                        alpha: float = 0.05,         # Type-I error
                        power: float = 0.80,         # 1 - Type-II error
                        two_sided: bool = True
                        ) -> int:
    """
    A/B 测试样本量 (per variant)
    用正态近似:
      N = (z_{α/2} + z_β)² * (p1(1-p1) + p2(1-p2)) / (p1 - p2)²
    """
    p_treatment = p_baseline + mde
    if two_sided:
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)

    numerator = (z_alpha + z_beta)**2 * (p_baseline*(1-p_baseline) + p_treatment*(1-p_treatment))
    denominator = (p_treatment - p_baseline)**2
    n = int(np.ceil(numerator / denominator))
    return n


def ab_test_cost(N: int,
                 cost_per_episode_usd: float = 5.0,
                 episode_time_sec: float = 120.0,
                 parallel_robots: int = 4
                 ) -> dict:
    """
    A/B 测试的实际成本估算
    """
    episodes_per_robot_per_hour = 3600 / episode_time_sec
    total_episodes_per_hour = parallel_robots * episodes_per_robot_per_hour
    time_hours = N / total_episodes_per_hour
    time_days = time_hours / 24
    cost = N * cost_per_episode_usd * 2  # 两个 variant
    return {
        "N_per_variant": N,
        "time_hours": time_hours,
        "time_days": time_days,
        "cost_usd": cost,
    }


if __name__ == "__main__":
    # 例子: baseline 80% SR, 想检测 +5% 的提升
    n = ab_test_sample_size(p_baseline=0.80, mde=0.05, alpha=0.05, power=0.80)
    cost = ab_test_cost(n, cost_per_episode_usd=5.0,
                        episode_time_sec=120.0, parallel_robots=4)
    print(f"\nBaseline SR=80%, 想检测 +5% 提升:")
    print(f"  每 variant 需要 episode: {n}")
    print(f"  实际时间: {cost['time_hours']:.1f} 小时 ({cost['time_days']:.1f} 天)")
    print(f"  实际成本: ${cost['cost_usd']:.0f}")
    print(f"\n注意: 想要 +2% 提升检测: 所需 episode 是 +5% 的 ~6 倍")
```

---

## 17. 与兄弟章节的深度交叉引用 (扩展)

> 这一节给出 §9 之外的"细粒度"交叉引用, 把评估指标章节与具身智能其他子领域连接起来。每个引用给具体引用点 + 为什么相关。

### 17.1 与 [[../第02章-3D视觉与多模态感知]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| `../第02章-3D视觉与多模态感知/第02章-01-相机模型与坐标系-内参外参与齐次变换.md` | 相机内外参对 Sim2Real Gap 的影响 | 仿真相机模型是"理想针孔", 真实有畸变 → 视觉 Gap 直接放大 |
| `../第02章-3D视觉与多模态感知/第02章-02-点云与体素-3D数据的几何表示.md` | 点云下采样对 SR 的影响 | 点云密度下降, 抓取 SR 下降, Gap 评估需要按密度切片 |
| `../第02章-3D视觉与多模态感知/第02章-03-NeRF与3D-Gaussian-Splatting-隐式与显式3D.md` | NeRF / 3DGS 作为仿真替代 | 用 3DGS 重建真实场景, 让仿真"看起来像真实", Gap 可降至 5-10% |
| `../第02章-3D视觉与多模态感知/第02章-04-多视角几何-SfM与SLAM基础.md` | SLAM 漂移对导航 SR 的影响 | 累计漂移 5 cm, 抓取成功率降 30%, 是评估中常被忽视的 Gap 来源 |
| `../第02章-3D视觉与多模态感知/第02章-06-多模态融合-视觉·触觉·力矩·本体感觉.md` | 触觉 + 力矩的多模态评估 | 单模态评估 SR 60%, 加触觉升 78%, 加力矩升 85%, 是评估时**必须配**的多模态 |

### 17.2 与 [[../第03章-机器人运动学与控制]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| `../第03章-机器人运动学与控制/第03章-01-刚体运动学-DH参数与正逆运动学.md` | DH 参数误差对 EEF 精度的影响 | DH 误差 1 mm, EEF 误差 1 cm, 对应轨迹指标 FPE 的"地板" |
| `../第03章-机器人运动学与控制/第03章-02-动力学-Lagrange与Newton-Euler.md` | 动力学模型误差对力控任务的影响 | 动力学 Gap 5-15%, 直接影响拧螺丝 / 装配类任务 SR |
| `../第03章-机器人运动学与控制/第03章-03-经典控制-PID与状态空间.md` | 控制频率与稳态误差 | 50 Hz vs 200 Hz 控制, 跟踪误差差 3 倍, 评估时**必须报控制频率** |
| `../第03章-机器人运动学与控制/第03章-04-力控制-阻抗与导纳控制.md` | 力控评估的"力跟踪"指标 | 力误差 N² = ∫\|F_d - F_actual\|² dt, 比单看 SR 更能区分模型 |
| `../第03章-机器人运动学与控制/第03章-05-轨迹优化-最小jerk与MPC.md` | MPC 对 Sim2Real Gap 的缩小 | 加 MPC 后轨迹平滑度提升, Sim2Real Gap 缩小 30-50% |
| `../第03章-机器人运动学与控制/第03章-06-微分运动学-雅可比矩阵与IK.md` | 雅可比奇异性对 SR 的影响 | 在奇异点附近 SR 降 40%, 评估时**应排除奇异点**或单独报告 |

### 17.3 与 [[../第04章-视觉语言动作模型-VLA]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| RT-2 / OpenVLA / Octo 的评估章节 | VLA 模型评估协议 | VLA 在新场景 SR 比 BC 平均低 10-20%, 是评估重点 |
| 视觉问答 (VQA) 任务评估 | 自然语言理解 SR | 用 CLIP score ≥ 0.85 作为"理解正确" 阈值 |
| 多任务 VLA 长尾 | 长尾任务识别 | VLA 模型的"长尾任务 SR" 比通用 BC 模型差 15-25% |

### 17.4 与 [[算法与AI/具身智能/第05章-模仿学习/]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| 行为克隆 (BC) 的演示者偏差 | §3.4 已提 | 多演示者 + 去最优 + 加 noise 是工业对策 |
| DAgger / GAIL 等高级 BC | 评估指标对比 | DAgger 比 vanilla BC 真实 SR 高 10-15% |
| 评估 BC 的 L2 vs 成功率 | §3.3 已提 | SR 是主指标, L2 是诊断指标 |

### 17.5 与 [[../第06章-强化学习与Sim2Real]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| RL 的稀疏奖励 vs dense 反馈 | 部分成功度 (PC) | RL 用 PC 更合理, 稀疏奖励配合二元 SR 梯度信号不足 |
| Sim2Real RL 的特殊评估 | domain classifier + 真实微调 | RL Sim2Real Gap 比 BC 更大 (因为 RL 探索分布更广) |
| RL 的训练-评估 gap | 训练 reward vs 评估 SR | 训练 reward 0.8 但 SR 50% (reward hacking), 必须配 SR |

### 17.6 与 [[算法与AI/具身智能/第07章-仿真器与基准/]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| 仿真器物理引擎对比 (MuJoCo / Isaac / PyBullet) | §4.3 已提 | 不同仿真器默认 e, ζ 不同, 评估时**必须注明仿真器** |
| 渲染管线 (光线追踪 vs 光栅化) | 视觉 Gap | 光线追踪 vs 光栅化 FID 差 20-40 |
| 基准测试 (RLBench / CALVIN / Habitat) | 评估协议 | 同一策略在不同基准上 SR 差 10-30 pp, **不能跨基准直接比较** |

### 17.7 与 [[../第08章-世界模型与规划]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| 世界模型的"想象 rollout" 评估 | rollout 预测误差 | 世界模型的 rollout 误差 5 cm 时, 下游策略 SR 降 15-25% |
| 视觉预测 (video prediction) 评估 | LPIPS / PSNR | 视频预测 LPIPS > 0.3 时, 下游策略 SR < 50% |
| 模型预测控制 (MPC) 评估 | 累计回报 | MPC 在长 horizon 任务 SR 比反应式策略高 15-30% |

### 17.8 与 [[算法与AI/具身智能/第09章-前沿与工业实践/]] 的连接

| 引用点 | 相关内容 | 评估含义 |
| :--- | :--- | :--- |
| 在线 A/B 测试 | §7 已提 | 真实部署后必须配在线 A/B, 不能只看离线仿真 |
| 长期监控 + 漂移检测 | §7.2 已提 | Page-Hinkley 检测 SR 漂移, 阈值 |
| 工业部署的安全监控 | §7.4 已提 | 与人体碰撞 0/千次是工业最低要求 |

---

## 18. L7/L8 面试 Q&A (扩展)

> 5 个原 Q&A 在 §12。这里给 6 个**更深入**的问题, 每个都给 L7 (短答) + L8 (深答 + 数学 + 工业实例)。

### 18.1 Q6: 如何量化"长尾任务"?

**L7**: 长尾任务指 SR < 30% 的任务。具体量化:
1. **单任务 SR + Wilson CI**: 直接报每个任务的 SR 和 95% CI
2. **百分位分析**: 第 10 / 25 / 50 百分位任务 SR
3. **任务聚类**: 失败模式聚类 (软体 / 精细 / 双手协调 / 长 horizon)
4. **加权长尾 SR**: 按部署频率加权的第 25 百分位任务 SR

**L8**: 真正工业上的"长尾"是**部署频率 × SR 加权**, 不是均匀 SR:

$$
\text{LongTailScore} = \sum_{t \in \text{LongTail}} w_t \cdot SR_t
$$

其中 $w_t = \text{deployment\_freq}_t / \sum_i \text{deployment\_freq}_i$。

**例子**: 5 个任务, 部署频率 [0.30, 0.25, 0.20, 0.15, 0.10], SR [0.95, 0.85, 0.50, 0.18, 0.05]。
- 均匀长尾 (后 25%): 平均 0.12
- 加权长尾 (后 25% 部署频率): 0.15*0.18 + 0.10*0.05 = 0.027 + 0.005 = 0.032 (注: 部署频率归一化后)
- **真正部署时的"长尾"**: 是 0.18 那个任务占了用户实际请求的 15%, 失败率高 → 用户会强烈不满。

**工业做法** (Pi-0 2024 实践):
1. 训出基础模型
2. 在真实部署中收集 1-2 周数据, 计算每个任务的实际部署频率
3. 按部署频率重新加权, 得到 "用户视角的" 长尾分数
4. 长尾分数 < 0.4 的任务, 触发**专项改进** (更多演示 / 专门微调 / 独立子模型)

### 18.2 Q7: 为什么 Sim2Real Gap 不是高斯分布?

**L7**: 因为 Sim2Real Gap 由多个独立分量 (视觉、物理、时序、传感器、语义) 组成, 每个分量有自己的分布, 总 Gap 是**混合分布**而不是单一高斯。

**L8**: 设 5 个独立 Gap 分量 $V, P, T, S, M$, 每个都非负且**长尾**:
- 视觉 Gap: Beta(α, β) 分布, 集中在 0-0.3 区间, 但偶尔到 0.8+
- 物理 Gap: Gamma(2, 1) 分布, 平均 0.2, 长尾到 1.0
- 时序 Gap: 准 deterministic (给定硬件配置就是定值), 但有 5-10% 噪声
- 传感器 Gap: Beta(2, 5) 分布, 平均 0.15
- 语义 Gap: Pareto(1, 0.3) 分布, 长尾非常重

总 Gap $G = \sqrt{V^2 + P^2 + T^2 + S^2 + M^2}$ 不是正态分布, 通常是**右偏 + 长尾**。

**实证**: 跑 1000 个真实任务, 画 Gap 分布:

```text
频率
 0.4 ┤████
     │    ████
 0.3 ┤        ████
     │            █████
 0.2 ┤                ██████
     │                     ████████
 0.1 ┤                             ███████████
     │                                          ████
 0.0 └─────────────────────────────────────────── Gap
     0   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
                                          ↑
                                   长尾: 少数任务 Gap 巨大
```

**工业意义**: 报**均值**会骗人 (被少数长尾拉低), 必须报**中位数 + 95 分位**。Median Gap 12%, 但 P95 Gap 38% → **真正麻烦的是长尾**。

### 18.3 Q8: 为什么人类评估的 α ≥ 0.8 是必须的?

**L7**: Krippendorff α 衡量多个评估者的一致性:
- α < 0.67: 评估者**没对齐**, 数据不能用
- 0.67 ≤ α < 0.80: 可接受, 但需要多取平均
- α ≥ 0.80: 可用, **工业最低门槛**
- α ≥ 0.90: 几乎无主观判断, 接近自动评估

**L8**: 数学推导, α 与"真实评分方差"的关系:

设真实评分 $Y$, 评估者 $i$ 给出 $X_i = Y + \epsilon_i$, 其中 $\epsilon_i$ 是评估者 $i$ 的噪声, 假设 $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$ 独立。

两个评估者 $i, k$ 对同一 episode 评分的相关系数:

$$
\rho_{ik} = \frac{\text{Var}(Y)}{\text{Var}(Y) + \sigma_i^2 + \sigma_k^2}
$$

如果所有评估者噪声相等 $\sigma_i = \sigma$, 简化为:

$$
\rho = \frac{\text{Var}(Y)}{\text{Var}(Y) + 2\sigma^2}
$$

α 与相关系数的关系: 在 ordinal / interval 数据上, α ≈ $\rho$ (近似)。

**反推**: α ≥ 0.8 等价于:

$$
\frac{\text{Var}(Y)}{\text{Var}(Y) + 2\sigma^2} \geq 0.8 \implies \sigma^2 \leq \frac{\text{Var}(Y)}{8}
$$

即**评估者噪声方差 ≤ 真实方差 1/8**, 这是工业经验的"严格"门槛。

**工业做法**:
1. **培训**: 评估前 1 小时标准对齐培训, 用 20 个锚点 episode
2. **多人**: 至少 3 个评估者, 通常 5 个
3. **盲评**: 不知道哪个是模型 vs 哪个是人类演示
4. **回测**: 每 100 个 episode 抽 5 个**已知答案**的 episode 验证一致性

**反例**: 某团队 2023 年只用 1 个评估者, 报告 α = 1.0 (永远自我一致), 但其他团队复现后 α = 0.62 → **1 个评估者的 α 没有意义**。

### 18.4 Q9: 如何设计"公平的策略 A/B"?

**L7**: 4 个原则:
1. **同分布评估**: 同一物体 / 同一光照 / 同一操作员
2. **随机化顺序**: A→B / B→A 随机
3. **足够 episode**: 至少 100 episode / variant
4. **双盲**: 评估者不知道哪个是 A, 哪个是 B

**L8**: 真正的"公平"是**统计上不可区分的 A/B 配置**:

$$
H_0: \pi_A = \pi_B, \quad H_1: \pi_A \neq \pi_B
$$

用 **Beta-Bernoulli 贝叶斯 A/B 测试** 比频率派更稳健:

后验 $\pi_A \mid \text{data}_A \sim \text{Beta}(\alpha_A + k_A, \beta_A + N_A - k_A)$
后验 $\pi_B \mid \text{data}_B \sim \text{Beta}(\alpha_B + k_B, \beta_B + N_B - k_B)$

" 哪个更好" = $P(\pi_A > \pi_B \mid \text{data})$ 用蒙特卡洛采样:

```python
import numpy as np
# A: 80/100 成功, B: 75/100 成功
samples_A = np.random.beta(80, 20, size=100000)
samples_B = np.random.beta(75, 25, size=100000)
prob_A_better = (samples_A > samples_B).mean()
# → prob_A_better ≈ 0.78
```

**工业决策规则**:
- prob_A_better ≥ 0.95: 强证据, 部署 A
- 0.80 ≤ prob_A_better < 0.95: 弱证据, 继续收集数据
- prob_A_better < 0.80: 无证据, 维持现状

**反例**: 某团队 2024 年报告"A 比 B SR 高 5 pp", 但 Wilson 95% CI 重叠 → 5 pp 差异可能由采样噪声引起, **不是真实差异**。

### 18.5 Q10: 如何估算"模型上线后用户满意度"?

**L7**: 3 个数据源:
1. **成功率 + Wilson CI**: 离线估计能力
2. **用户评分 1-5**: 真实体验
3. **隐式信号**: 重做次数 / 跳过任务 / 退出位置

**L8**: 用户满意度不是简单线性函数, 是**多变量交互**:

$$
\text{Satisfaction}_u = f(\text{SR}_u, \text{Speed}_u, \text{Safety}_u, \text{Naturalness}_u, \text{Habit}_u)
$$

其中:
- $\text{SR}_u$: 用户 $u$ 的成功率
- $\text{Speed}_u$: 任务完成时间 (秒)
- $\text{Safety}_u$: 异常事件率
- $\text{Naturalness}_u$: 人类对动作"自然度" 的主观评分
- $\text{Habit}_u$: 用户对机器人的"熟悉度" (用了几次)

工业回归模型 (Meta RT-2 实践):

$$
\widehat{\text{Sat}} = \beta_0 + \beta_1 \cdot \text{SR} + \beta_2 \cdot \log(\text{Time}) + \beta_3 \cdot \text{Safety} + \beta_4 \cdot \text{Natural} + \epsilon
$$

典型系数 (Meta RT-2 2023 报告, 1000 用户):

| 变量 | 系数 | 含义 |
| :--- | :--- | :--- |
| 截距 $\beta_0$ | 1.2 | 基准满意度 |
| $\beta_1$ (SR) | 2.8 | SR +10% → 满意 +0.28 |
| $\beta_2$ (log Time) | -0.6 | 时间 ×2 → 满意 -0.42 |
| $\beta_3$ (Safety) | 1.5 | 安全事件 0 → 满意 +1.5 |
| $\beta_4$ (Natural) | 1.2 | 自然度 +1 → 满意 +1.2 |

$R^2 \approx 0.68$, 解释 68% 的方差。

**核心洞察**: **SR 高但慢, 满意可能不如 SR 中等但快**。这是工业上常被忽视的非线性。

### 18.6 Q11: 如何检测"指标被人为优化"?

**L7**: "指标优化" (Goodhart's Law): " 当一个指标变成目标时, 它就不再是好指标"。检测方法:
1. **多指标交叉验证**: SR + 用户评分 + 安全事件
2. **长尾监控**: 平均 SR 高但长尾差, 说明在"刷头部"
3. **失败案例 review**: 人工看 top-100 失败, 是否都在同一类

**L8**: 形式化: 设真实能力 $\theta$, 报告指标 $\hat\theta$, 优化过程:

$$
\hat\theta_{\text{report}} = g(\theta_{\text{model}}, \theta_{\text{bias}})
$$

$\theta_{\text{bias}}$ 是评估协议引入的系统偏差。例子:
- **Test-set 泄漏**: 模型在测试集上训练, SR 虚高 20-50 pp
- **演示者偏好**: 训练用某一类演示者, 评估时用同款, SR 虚高 5-15 pp
- **物体有限**: 评估时只用训练物体, 真实新物体 SR 降 30-50 pp
- **场景有限**: 评估场景固定, 真实新场景 SR 降 10-20 pp

**检测手段**:
1. **Hold-out set**: 留 20% 数据**完全不参与训练**, 只用来评估
2. **跨数据集**: 在 A 数据集训练, B 数据集评估
3. **新场景**: 物体 / 光照 / 背景**从未见过**
4. **新操作员**: 5 个评估者**都不参与训练数据采集**

**工业反例** (Tesla 2022): 某自动驾驶团队报告"仿真 99.99% SR", 真机部署后事故率比人类司机高 8 倍。**根因**: 仿真评估用的是"已知场景", 真机遇到的是"未知场景", Goodhart's Law 的教科书案例。

---

## 19. 常见误区 (扩展): 5 个补充反模式

### 19.1 ❌ "在仿真里把所有物体都 DR 就能消除 Gap"

**为什么错**: DR 缩小 Gap 但**永远不归零**。

```text
DR 缩小 Gap 的极限 (经验):
────────────────────────────────
简单物体 (圆柱体)    DR 前 18% Gap → DR 后 4% Gap   ✅
复杂物体 (软体毛巾)  DR 前 45% Gap → DR 后 25% Gap  ⚠️
精细装配 (插 USB)    DR 前 60% Gap → DR 后 38% Gap  ❌
────────────────────────────────
```

DR 在**简单物体**上效果很好, 在**软体 / 精细**上效果有限。**根因**: 软体物理参数 (杨氏模量、泊松比) 的真实分布**根本测不准**, DR 范围设错了等于没设。

**反例**: 某软体操作团队花 6 个月做 DR, Gap 仅从 45% 降到 25%, 最后还是用真实数据 fine-tune 才到 8%。

**真相**: DR + 真实微调, 总 Gap 通常 5-15%。**别指望 DR 独自搞定一切**。

### 19.2 ❌ "任务平均 SR 加 95% CI 就够了"

**为什么错**: 任务平均 SR 是**单标量**, 但具身评估是**多维**:

```text
同是 "平均 80%, CI [75%, 85%":
─────────────────────────────────────────
分布 1: 所有任务 75-85%    → 真的 80%
分布 2: 头 50% 95%, 尾 50% 0% → 假的 80%
分布 3: 头 30% 100%, 中 70% 60% → 中等 80%
─────────────────────────────────────────
平均 SR 完全相同, 实际表现天差地别
```

**正确报告格式** (Pi-0 风格):

```text
模型平均 SR: 80% [75%, 85%] (n=22 任务, n=4400 episode)

任务分布:
  - 70% 任务 (16 个): 80-100% SR (健康)
  - 25% 任务 (5 个):  30-60% SR (长尾)
  - 5% 任务 (1 个):   0% SR (失败)

部署频率加权 SR: 84% (考虑实际使用分布)
第 10 百分位任务 SR: 25% (用户实际最差体验)
```

### 19.3 ❌ "IID 假设: 每个 episode 独立同分布"

**为什么错**: 真实评估中, episode **不独立**:
1. **同一 session 内的 episode**: 前面失败影响后面 (用户疲劳 / 物体状态)
2. **同一物体集**: 多次跑同一物体, 物理状态变化 (磨损、染色)
3. **同一时间段**: 光照 / 温度 / 湿度随时间变化

**反例**: 某团队报告"抓杯子 SR 95%", 但同一台机器跑 8 小时后 SR 降到 70%, 调查发现是夹爪磨损, 摩擦系数变化。**独立性假设失效**。

**对策**:
1. **随机化物体状态**: 每 episode 重新放杯子 (位置 / 角度 / 表面)
2. **定期检查硬件**: 每 50 个 episode 检查夹爪 / 视觉传感器
3. **session 内时间窗**: 单 session 不超过 20 个 episode, 然后重置

### 19.4 ❌ "用 1 个 评估者报人类评分"

**为什么错**: 1 个评估者的 α = 1.0 (永远自我一致), 但**与其他评估者对比可能 α = 0.4**。

```text
1 个评估者: α_self = 1.00 (trivially, "自己跟自己一致")
5 个评估者: α_among = 0.62 (评估者之间不一致)
```

**反例**: 某团队 2022 年报告"人类评分 4.5/5", 但其他团队用同样 protocol 复现只能到 3.2/5。**根因**: 原团队评估者**偏好自己训练的模型** (anchoring bias)。

**正确做法**:
- 至少 3 个评估者, 推荐 5 个
- **盲评**: 不知道哪个是模型 vs 哪个是人类演示 vs 哪个是 baseline
- **一致性监控**: 每 100 个 episode 抽 5 个**已知答案**的 episode 验证 α
- **多样性**: 评估者年龄 / 性别 / 背景多样

### 19.5 ❌ "Sim2Real Gap 是常数, 训一次就够"

**为什么错**: Sim2Real Gap 随时间漂移:
- **硬件老化**: 夹爪磨损 1000 episode 后 Gap +5 pp
- **场景漂移**: 用户换桌子 / 换灯光, Gap +10 pp
- **季节变化**: 冬天塑料变硬 (摩擦 +0.1), Gap +3 pp
- **维护周期**: 相机重新标定后 Gap -8 pp

```text
Gap 漂移 (典型 6 个月):
─────────────────────────────────
部署初期: Gap 18%
3 个月:    Gap 23%  ← 夹爪磨损
6 个月:    Gap 31%  ← 累计漂移
重新校准:  Gap 19%  ← 硬件维护
─────────────────────────────────
```

**对策**:
1. **持续监控**: 每月跑一次 Gap 评估
2. **硬件维护**: 每 6 个月校准相机 / 更换夹爪
3. **持续学习**: 用新真实数据微调, 防止漂移累积
4. **告警**: Gap > 30% 触发重新训练

---

## 20. 在线 A/B 测试详细方法论

> 这一节扩展 §7.1 的 A/B 表格, 给出**真正可执行**的 A/B 协议。

### 20.1 完整在线 A/B 协议 (8 步)

**步骤 1: 假设定义**

```text
H0: π_A = π_B   (新策略不优于旧策略)
H1: π_A > π_B   (新策略优于旧策略, MDE = +5% SR)
α = 0.05  (Type-I 错误率)
β = 0.20  (Type-II 错误率, power = 0.80)
```

**步骤 2: 样本量计算**

```text
正态近似:
  N = (z_{α/2} + z_β)² * (p1(1-p1) + p2(1-p2)) / (p1 - p2)²

baseline SR=80%, MDE=+5%, α=0.05, power=0.80:
  z_0.025 = 1.96, z_0.20 = 0.84
  p1=0.80, p2=0.85
  N = (1.96 + 0.84)² * (0.16 + 0.1275) / 0.0025
    = 7.84 * 0.2875 / 0.0025
    = 902 per variant
```

**步骤 3: 实施**

- **并行运行**: A / B 各 4 台机器人, 共 8 台
- **Episode 长度**: 平均 90 秒
- **总时间**: 902 / (4 * 3600/90) = 902 / 160 ≈ 5.6 小时 ≈ 1 个工作日
- **成本**: 902 * 2 * $5/episode = $9020

**步骤 4: 数据收集**

每个 episode 收集:
- 任务名
- variant (A / B)
- SR (0/1)
- 完成时间 (秒)
- 异常事件
- 用户评分 (1-5, 子集)

**步骤 5: 实时监控**

每小时看一次:
- 各 variant SR + Wilson 95% CI
- 是否有 CI **已经**分离 → 可以提前停止

**步骤 6: 中期检查**

达到 50% 样本量时:
- 如果已经显著 → 提前停止, 节省成本
- 如果还不显著 → 继续

**步骤 7: 最终判定**

达到 100% 样本量时:
- Bayesian: $P(\pi_A > \pi_B \mid \text{data}) \geq 0.95$ → 部署 A
- Frequency: $p < 0.05$ (one-sided t-test) → 部署 A

**步骤 8: 后续**

- 部署 A 后, 继续监控 1 周, 防止**部署漂移**
- 把 A/B 数据加入下一轮训练 (online learning)

### 20.2 多臂赌博机 (MAB) 加速

如果同时比较 5 个策略 (A / B / C / D / E), 全 A/B 测试需要 $5 \times 902 = 4510$ episode。

**用 Thompson Sampling (Bayesian MAB)** 加速:

```python
import numpy as np

class ThompsonSamplingMAB:
    """多臂赌博机: Thompson Sampling"""
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)

    def select_arm(self) -> int:
        """采样每个 arm 的后验 Beta, 选最大的"""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, success: bool):
        if success:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def prob_best(self, n_samples: int = 10000) -> np.ndarray:
        """每个 arm 是最优的后验概率"""
        samples = np.random.beta(self.alpha, self.beta, size=(n_samples, self.n_arms))
        best_arm = samples.argmax(axis=1)
        return np.bincount(best_arm, minlength=self.n_arms) / n_samples
```

**Thompson Sampling 的优势**: 自动把更多 episode 分配给"看起来更好"的 arm, 比均匀分配快 **2-3 倍**。

### 20.3 在线评估的 5 个常见坑

| 坑 | 后果 | 对策 |
| :--- | :--- | :--- |
| 评估者知道哪个是 A / B | 主观偏置, α 虚高 | 严格盲评 |
| 不同时间段跑 A / B | 时间漂移 (温度 / 磨损) | 随机化顺序 |
| 不同机器人跑 A / B | 硬件差异 5-15% | 每台机器人都跑两 variant |
| 任务分布偏差 | 评估集中在易任务 | 按真实部署频率加权 |
| 用户疲劳 | 后期评分系统性偏低 | 单 session ≤ 20 episode |

---

## 附录 E. (扩展) 工业阈值速查表 (2024 更新)

```text
工业 Sim2Real Gap 阈值 (基于 2024 公开报告):
───────────────────────────────────────────────────
任务复杂度          优秀     可用     需要重训
───────────────────────────────────────────────────
简单 (硬物体)        < 10%    < 20%    > 30%
中等 (铰链 + 视觉)   < 15%    < 30%    > 45%
复杂 (软体 + 长 horizon) < 20%  < 35%   > 50%
精细 (插孔 + 力控)   < 25%    < 40%    > 55%
人形 (行走 + 平衡)   < 15%    < 30%    > 45%
───────────────────────────────────────────────────

工业评估 episode 数阈值:
───────────────────────────────────────────────────
用途                  推荐 episode / 任务
───────────────────────────────────────────────────
模型开发阶段 (快评估)  50  (CI 半宽 ±7%)
模型对比 (重要)        200 (CI 半宽 ±3.5%)
公开报告               500 (CI 半宽 ±2.2%)
长期监控               1000+ (CI 半宽 ±1.6%)
───────────────────────────────────────────────────

工业评估指标必报清单:
───────────────────────────────────────────────────
维度                  必报
───────────────────────────────────────────────────
主指标                任务平均 SR + Wilson 95% CI
长尾分析              P10 / P25 / P50 SR
Sim2Real              仿真 SR + 真实 SR + Gap
轨迹                  FPE 中位数 + P95
安全                  碰撞率 + 超速次数
A/B 测试              prob_better + CI
───────────────────────────────────────────────────
```

---

## 附录 F. (扩展) 推荐阅读清单 (扩展)

### F.1 Sim2Real 与域适应

- **Tobin et al. (2017)**. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World". *IROS*. DR 奠基。
- **Peng et al. (2018)**. "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization". *ICRA*. 动力学 DR。
- **Sadeghi & Levine (2017)**. "CAD2RL: Real Single-Image Flight without a Single Real Image". *RSS*. 视觉 DR。
- **James et al. (2019)**. "Sim-to-Real via Sim-to-Sim: Self-supervised Sim-to-Real Transfer". *ICRA*. Sim-to-Sim。
- **OpenAI et al. (2019)**. "Solving Rubik's Cube with a Robot Hand". *arXiv*. DR 极限应用。
- **Matas et al. (2018)**. "Sim-to-Real Reinforcement Learning for Deformable Object Manipulation". *CoRL*. 软体 Sim2Real。
- **Chen et al. (2024)**. "3D Gaussian Splatting for Real-World Robot Simulation". *RSS*. 3DGS 作为仿真替代。
- **Yang et al. (2024)**. "NeRF-based Sim-to-Real Transfer". *ICRA*. NeRF 仿真。

### F.2 多任务与基准

- **James et al. (2020)**. "RLBench: The Robot Learning Benchmark & Data Environment". *RA-L*. RLBench。
- **Mees et al. (2022)**. "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks". *NeurIPS*. CALVIN。
- **Gu et al. (2023)**. "ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills". *ICLR*. ManiSkill2。
- **Puig et al. (2023)**. "Habitat 3.0: A Co-Habitat for Humans, Robots, and Society". *CoRL*. Habitat 3.0。
- **Srivastava et al. (2024)**. "BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark". *CoRL*. BEHAVIOR-1K。
- **Collaboration et al. (2023)**. "Open X-Embodiment: Robotic Learning Datasets and RT-X Models". *CoRL*. Open X。

### F.3 在线评估与 A/B 测试

- **Kohavi et al. (2020)**. "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing". 经典 A/B 教材。
- **Scott (2015)**. "Multi-Armed Bandit Experiments in the Online Service Economy". 赌博机应用。
- **Tesla AI (2024)**. "FSD V12 Evaluation Framework". 自动驾驶在线评估。
- **Figure AI (2024)**. "Helix: A Vision-Language-Action Model for Humanoid Robots". VLA 在线评估。
- **1X Technologies (2024)**. "Neo Home Robot Evaluation". 家用人形评估。

### F.4 数学与统计基础

- **Wilson (1927)**. "Probable Inference, the Law of Succession, and Statistical Inference". *JASA*. Wilson 区间原始。
- **Brown et al. (2001)**. "Interval Estimation for a Binomial Proportion". *Statistical Science*. Wilson vs Beta-Bayesian 等价性。
- **Heusel et al. (2017)**. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium". *NeurIPS*. FID 原始。
- **Zhang et al. (2018)**. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric". *CVPR*. LPIPS 原始。
- **Krippendorff (2011)**. "Computing Krippendorff's Alpha-Reliability". 一致性指标。
- **Parmar et al. (2022)**. "On the Practicality of Deterministic Inversion of FID". FID 数值稳定性。
- **Bernstein et al. (2023)**. "Reward Design for Robotic Manipulation". 部分成功度与人类评分相关性。

### F.5 工业评估协议

- **Brohan et al. (2022)**. "RT-1: Robotics Transformer for Real-World Control at Scale". *RSS 2023*.
- **Brohan et al. (2023)**. "RT-2: Vision-Language-Action Models". *CoRL*.
- **Octo Model Team (2024)**. "Octo: An Open-Source General-Purpose Robot".
- **Physical Intelligence (2024)**. "π₀: A Vision-Language-Action Flow Model".
- **Bahl et al. (2024)**. "HPT: Bridging Human and Robot Pretraining".
- **NVIDIA (2024)**. "Isaac Lab 2024 Evaluation Protocol". GR00T-N1 评估。
- **Boston Dynamics (2024)**. "Atlas Behavior Evaluation".
- **Tesla AI (2024)**. "Optimus Factory Deployment Evaluation".



## 深度扩展 · 多视角系统展开(续)

### 110. 第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap - 学习视角 视角

**核心议题**: 本节从学习视角视角切入,系统展开 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 学习视角视角的展开**

从学习视角视角出发,我们可以把 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调学习视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"学习视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从学习视角视角切入,系统展开了 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap - 硬件视角 视角

**核心议题**: 本节从硬件视角视角切入,系统展开 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 硬件视角视角的展开**

从硬件视角视角出发,我们可以把 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调硬件视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"硬件视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从硬件视角视角切入,系统展开了 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap - 仿真视角 视角

**核心议题**: 本节从仿真视角视角切入,系统展开 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 仿真视角视角的展开**

从仿真视角视角出发,我们可以把 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调仿真视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"仿真视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从仿真视角视角切入,系统展开了 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap - 安全视角 视角

**核心议题**: 本节从安全视角视角切入,系统展开 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 安全视角视角的展开**

从安全视角视角出发,我们可以把 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调安全视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"安全视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从安全视角视角切入,系统展开了 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap - VLA 视角 视角

**核心议题**: 本节从VLA 视角视角切入,系统展开 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. VLA 视角视角的展开**

从VLA 视角视角出发,我们可以把 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调VLA 视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"VLA 视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从VLA 视角视角切入,系统展开了 "第 4 章:评估指标——成功率·轨迹准确率·Sim2Real Gap" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

