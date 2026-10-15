---
title: AB 测试与因果推断
tags:
  - A/B 测试
  - 因果推断
  - 假设检验
  - 显著性
  - 因果图
  - 工具与平台
description: 3. AB 测试与因果推断
last_updated: 2026-07-16
status: ✅ v1.0
---

# 3. AB 测试与因果推断

> 推荐系统工程师的"决策科学":从"拍脑袋"到"看数据"。

## 0. 一句话总结

**A/B 测试 = 用统计学方法在受控实验下量化改动效果;因果推断 = 回答"为什么";两者结合 = "知道发生了什么 + 知道为什么 + 知道下一步该做什么"。**

---

## 1. 为什么需要 A/B 测试

### 1.1 离线指标的局限

**离线指标 = 在历史数据上算的指标**。

| 离线指标 | 局限 |
|----------|------|
| AUC / LogLoss | 衡量排序能力,不直接等于业务效果 |
| Recall@K | 召回质量,不等同于用户满意度 |
| NDCG | 假设用户按相关性排序,实际未必 |
| 离线 CTR | 历史点击≠未来点击(数据有偏) |

**经典案例:** Netflix 在离线指标上大幅提升,但在线 AB 反而掉量——因为离线数据忽略了"封面图"对新内容的影响。

### 1.2 在线指标的必要性

**在线指标 = 用户真实反馈**。

| 指标类型 | 例子 | 特点 |
|----------|------|------|
| 用户行为 | CTR、CVR、停留时长 | 即时,但有新奇效应 |
| 用户满意度 | 留存率、次日回访 | 滞后 1-7 天 |
| 商业指标 | GMV、广告收入 | 滞后但权威 |
| 长期价值 | LTV、复购率 | 滞后 1-3 个月 |

**为什么离线指标不能替代在线指标?**

1. **分布偏移**:训练数据≠线上数据分布
2. **位置偏差**:用户只看到曝光位置的内容
3. **新奇效应**:新策略初期用户有好奇心,效果虚高
4. **生态效应**:策略改动影响整个生态(广告主、内容创作者)
5. **正反馈循环**:点击多的物品被推得更多,推荐越来越窄

### 1.3 决策需要数据

**决策金字塔:**

```text
           [业务目标]
              ↑
       [A/B 测试验证]
              ↑
       [离线指标初筛]
              ↑
       [直觉 + 经验]
```

**A/B 测试的三大价值:**

1. **效果验证**:新模型/新策略是否真的有效?
2. **效果量化**:提升多少?可持续吗?
3. **风险控制**:有问题立即回滚,损失可控

---

## 2. A/B 测试基础

### 2.1 实验单位 (用户/请求/物品)

**实验单位 = 流量分配的最小单元**。

| 单位 | 含义 | 适用场景 | 优缺点 |
|------|------|----------|--------|
| **用户级** | 按 user_id 分桶 | 大多数场景(推荐、广告) | 稳定,但样本量大 |
| **会话级** | 按 session_id 分桶 | 短期场景 | 用户跨会话会冲突 |
| **请求级** | 每次请求独立分桶 | 单次曝光实验 | 易污染,慎用 |
| **物品级** | 按 item_id 分桶 | 内容侧实验(物品质量) | 用户会跨桶看 |
| **地域级** | 按城市分桶 | 跨地域政策 | 样本少 |

**用户级是最常用的实验单位**,因为:

- 用户行为相对稳定
- 避免同一用户在不同组看到不同内容,造成体验割裂
- 易于分析和归因

```python
# 用户级分桶:stable hash
def user_bucket(user_id, exp_id, num_buckets=100):
    """稳定的用户分桶"""
    raw = hashlib.md5(f"{user_id}:{exp_id}".encode()).hexdigest()
    return int(raw[:8], 16) % num_buckets

# 使用示例
bucket = user_bucket(uid, "exp_recall_v2")
if bucket < 10:  # 10% 流量
    return new_strategy
else:
    return old_strategy
```

### 2.2 流量分配 (1%/5%/10%/50%)

**流量分配的决策框架:**

| 流量比例 | 适用场景 | 持续时间 |
|----------|----------|----------|
| 1% | 全新策略上线验证 | 1-3 天 |
| 5% | 验证稳定性 | 3-7 天 |
| 10% | 收集足够样本 | 7-14 天 |
| 50% | 重要改动 | 14-30 天 |
| 100% | 全量 | 持续 |

**流量分配的工程考量:**

1. **样本量充足**:每个组必须有足够样本达成统计显著性
2. **互斥与正交**:
   - 互斥:用户不能同时在两个互斥实验里
   - 正交:不同实验的流量独立,互不干扰
3. **Holdout 保留**:5-10% 流量永远不参与实验,作为 baseline

```python
# 互斥实验 vs 正交实验
class ExperimentAllocation:
    """流量分配系统"""
    def __init__(self):
        self.exclusive_layers = []  # 互斥层(同一用户只能落一个)
        self.orthogonal_layers = [] # 正交层(用户独立分流)

    def assign(self, user_id):
        result = {}

        # 互斥层:用户只能落一个
        for layer in self.exclusive_layers:
            bucket = user_bucket(user_id, layer.name)
            cumulative = 0
            for exp in layer.experiments:
                cumulative += exp.traffic_ratio
                if bucket < cumulative * 100:
                    result[layer.name] = exp.name
                    break

        # 正交层:用户独立分流(可以同时落多个)
        for layer in self.orthogonal_layers:
            bucket = user_bucket(user_id, layer.name)
            cumulative = 0
            for exp in layer.experiments:
                cumulative += exp.traffic_ratio
                if bucket < cumulative * 100:
                    result[layer.name] = exp.name
                    break

        return result
```

### 2.3 实验时长

**实验时长决策:**

| 改动类型 | 推荐时长 | 原因 |
|----------|----------|------|
| UI 调整 | 1-3 天 | 新奇效应消退快 |
| 模型升级 | 7-14 天 | 覆盖完整业务周期 |
| 召回通道 | 14-21 天 | 验证长尾效果 |
| 算法大改 | 21-30 天 | 验证稳定性和生态效应 |

**为什么不能太短?**

1. **新奇效应**:新策略初期效果虚高
2. **学习效应**:用户需要时间适应
3. **业务周期**:电商有周末高峰,需要覆盖完整周期
4. **样本量**:短时间样本不足,统计功效低

**为什么不能太长?**

1. **机会成本**:实验占用流量,无法做其他实验
2. **业务风险**:长期挂着一个有问题的实验
3. **生态污染**:实验组与对照组长期差异可能导致生态变化

### 2.4 核心指标 vs 护栏指标

**核心指标 = 实验成功与否的关键指标**。

**护栏指标 = 不能因为做实验而恶化的指标**。

| 类型 | 推荐指标 |
|------|----------|
| **核心指标** | CTR、CVR、留存率、时长、GMV |
| **护栏指标** | 服务器延迟、错误率、用户投诉率、退款率 |
| **反向指标** | 用户负反馈、广告收入(若有冲击) |

```yaml
# 实验配置示例
experiment:
  name: "exp_ranker_v3"
  layer: "ranking"
  traffic_ratio: 0.10

  primary_metrics:
    - name: ctr
      direction: increase      # 期望上升
      mde: 0.02                # 最小可检测效应 2%
    - name: avg_watch_time
      direction: increase
      mde: 0.05

  guardrail_metrics:
    - name: p99_latency
      direction: not_increase
      threshold: 100ms
    - name: error_rate
      direction: not_increase
      threshold: 0.001
    - name: user_complaint_rate
      direction: not_increase
      threshold: 0.0001
    - name: refund_rate
      direction: not_increase
      threshold: 0.05

  duration_days: 14
  daily_sample: 100000
```

---

## 3. 统计学基础

### 3.1 假设检验 (t 检验 / 卡方)

**假设检验 = 用样本数据判断"两组是否有显著差异"**。

**核心逻辑:**

1. 提出原假设 H₀:两组无差异(对照组 = 实验组)
2. 提出备择假设 H₁:两组有差异
3. 计算 p-value:p-value < α 则拒绝 H₀
4. α 一般取 0.05

**两类检验:**

| 检验 | 适用场景 | 公式 |
|------|----------|------|
| **t 检验** | 连续变量(均值) | t = (X̄₁ - X̄₂) / SE |
| **z 检验** | 大样本比例 | z = (p₁ - p₂) / SE |
| **卡方检验** | 分类变量 | χ² = Σ(O - E)² / E |
| **Mann-Whitney** | 非正态 | 比较中位数 |

**Python 实现:**

```python
import numpy as np
from scipy import stats

def t_test(group_a, group_b, alpha=0.05):
    """独立样本 t 检验"""
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    significant = p_value < alpha

    mean_diff = np.mean(group_a) - np.mean(group_b)
    pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
    cohens_d = mean_diff / pooled_std  # 效应量

    return {
        'mean_a': np.mean(group_a),
        'mean_b': np.mean(group_b),
        'mean_diff': mean_diff,
        'p_value': p_value,
        'significant': significant,
        'cohens_d': cohens_d
    }

def proportion_test(success_a, n_a, success_b, n_b, alpha=0.05):
    """比例检验(CTR/CVR)"""
    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        'p_a': p_a,
        'p_b': p_b,
        'lift': (p_b - p_a) / p_a,
        'z_score': z,
        'p_value': p_value,
        'significant': p_value < alpha
    }

def chi_squared_test(observed_a, observed_b, alpha=0.05):
    """卡方检验"""
    contingency = np.array([observed_a, observed_b])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < alpha
    }
```

### 3.2 显著性水平 α

**α = 第一类错误的概率(弃真概率)**,即"实际上没有差异,但实验判断有差异"的概率。

| α | 含义 | 适用场景 |
|----|------|----------|
| 0.01 | 严格 | 大额决策(并购) |
| 0.05 | 标准 | 大多数业务实验 |
| 0.10 | 宽松 | 探索性实验 |

**注意:** α 不是越小越好——严格意味着需要更多样本,实验更慢。

### 3.3 统计功效 1-β

**β = 第二类错误的概率(取伪概率)**,即"实际上有差异,但实验判断没有"的概率。
**功效 1-β = 1 - β = 正确检测到差异的概率**。

**工业界常用功效:0.8**(即 β=0.2)

**影响功效的因素:**

1. **样本量**:样本越大,功效越高
2. **效应大小**:差异越大,功效越高
3. **方差大小**:方差越小,功效越高
4. **α 水平**:α 越大,功效越高(但误报率高)

### 3.4 样本量计算

**样本量公式(比例指标):**

```text
n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²
```

**示例:**

```python
from scipy.stats import norm

def sample_size_proportion(p1, mde, alpha=0.05, power=0.8):
    """
    计算比例指标的样本量(每组)

    p1: 对照组比例
    mde: 最小可检测效应(相对值,如 0.02 表示 2% 提升)
    """
    p2 = p1 * (1 + mde)
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1-p1) + p2 * (1-p2))
    denominator = (p1 - p2) ** 2

    return int(np.ceil(numerator / denominator))

# 示例:对照 CTR=0.05,期望提升 2%,alpha=0.05, power=0.8
n = sample_size_proportion(0.05, 0.02)
print(f"每组需要 {n} 个样本")  # 约 380,000
```

**常用场景的样本量参考:**

| 场景 | 基线 CTR | 期望提升 | 每组样本量 | 实验时长(假设日 100 万 UV) |
|------|----------|----------|-----------|------------------------------|
| 大改动 | 5% | 5% | 60,000 | 1 天 |
| 中改动 | 5% | 2% | 380,000 | 4 天 |
| 小改动 | 5% | 1% | 1,500,000 | 15 天 |
| 微调 | 5% | 0.5% | 6,000,000 | 60 天 |

### 3.5 置信区间

**置信区间 = 真实值的合理范围**。

```python
def confidence_interval_proportion(p, n, confidence=0.95):
    """比例的置信区间(Wilson 区间)"""
    z = norm.ppf(1 - (1 - confidence) / 2)

    # Wilson 区间(更准确)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    return (center - margin, center + margin)

# 示例:CTR = 0.05, n = 1000000
ci = confidence_interval_proportion(0.05, 1000000)
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
# 输出: 95% CI: [0.0499, 0.0501]
```

**置信区间的解读:**

"我们有 95% 的信心,真实 CTR 在 4.99% 到 5.01% 之间"。注意:**不是"CTR 有 95% 概率落在这个区间"**(常见误解)。

---

## 4. 实验设计模式

### 4.1 简单 A/B

最基础的实验设计:把用户随机分成 2 组,对照组 vs 实验组。

```text
[流量 100%]
  ├─ 对照组 50%
  └─ 实验组 50%
```

**优点:** 简单
**缺点:** 只能测一个改动

### 4.2 多变量测试 (MVT)

**MVT = 同时测试多个变量的多个水平**。

```text
[流量 100%]
  ├─ A1B1 (颜色红 + 文案 1)
  ├─ A1B2 (颜色红 + 文案 2)
  ├─ A2B1 (颜色蓝 + 文案 1)
  └─ A2B2 (颜色蓝 + 文案 2)
```

**优点:** 一次实验多个变量
**缺点:** 需要的样本量 = n × 变量数(组合爆炸)

**适用:** UI 改版、广告创意、推送文案

```python
# MVT 样本量计算
def sample_size_mvt(num_groups, p1, mde, alpha=0.05, power=0.8):
    """多组比较需要更多样本(用 Bonferroni 校正)"""
    adjusted_alpha = alpha / (num_groups - 1)  # Bonferroni
    return sample_size_proportion(p1, mde, alpha=adjusted_alpha, power=power)
```

### 4.3 多臂老虎机 (MAB)

**MAB = 把流量动态分配给表现好的实验组**。

```text
[流量]
  ├─ 实验组 A: 30% (CTR=5%)
  ├─ 实验组 B: 20% (CTR=4.5%)
  ├─ 实验组 C: 50% (CTR=5.5%) ← 表现好,分配更多流量
```

**MAB 的核心算法:**

| 算法 | 描述 | 适用 |
|------|------|------|
| **ε-greedy** | 90% 选最优,10% 探索 | 简单场景 |
| **UCB** | 基于置信上界选择 | 探索-利用权衡好 |
| **Thompson Sampling** | 贝叶斯采样 | 工业界主流 |

**Thompson Sampling 实现:**

```python
class ThompsonSamplingMAB:
    """Thompson Sampling MAB"""
    def __init__(self, num_arms, alpha=1, beta=1):
        self.alpha = [alpha] * num_arms  # Beta 分布的 α
        self.beta = [beta] * num_arms    # Beta 分布的 β

    def select_arm(self):
        """选择臂"""
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                   for i in range(len(self.alpha))]
        return np.argmax(samples)

    def update(self, arm, reward):
        """更新参数"""
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# 使用
mab = ThompsonSamplingMAB(num_arms=4)  # 4 个实验组
for user in users:
    arm = mab.select_arm()  # 选哪组
    show_variant(user, arm)  # 展示对应版本
    # ... 用户行为后
    reward = 1 if user_clicked else 0
    mab.update(arm, reward)
```

**MAB 的适用与局限:**

| 适用 | 不适用 |
|------|--------|
| 个性化场景(每个用户单独决策) | 全局效果评估 |
| 短期收益最大化 | 长期价值评估 |
| 探索成本高(不能并行) | 简单的算法对比 |

### 4.4 Switchback 实验

**Switchback = 按时间段切换对照组/实验组,而不是按用户分桶**。

```text
14:00-14:30: 对照组
14:30-15:00: 实验组
15:00-15:30: 对照组
...
```

**适用场景:**

- 网约车(供需匹配,用户级分桶不适用)
- 外卖配送
- 搜索引擎(查询独立)

**优点:** 避免用户级分桶的污染
**缺点:** 时间序列自相关,需要更复杂的统计方法

```python
# Switchback 实验代码
import schedule
import time

def switchback_experiment():
    """每 30 分钟切换一次"""
    state = {"variant": "control"}

    def switch():
        state["variant"] = "treatment" if state["variant"] == "control" else "control"
        print(f"Switched to {state['variant']}")

    schedule.every(30).minutes.do(switch)
    while True:
        schedule.run_pending()
        # 用 state["variant"] 决定当前请求走哪组
        time.sleep(1)
```

### 4.5 Holdout 实验

**Holdout = 长期保留一个"对照组",作为所有实验的基线**。

```text
[流量]
  ├─ 长期 Holdout 5%  (永不变,作为所有实验的基线)
  └─ 实验流量 95% (各种实验共用)
```

**价值:** 当业务整体效果变化时,可以用 Holdout 评估"如果没做实验,业务会怎样"。

### 4.6 交错实验 (Interleaving)

**Interleaving = 不分用户分桶,而是把多个模型的输出"混合"展示给同一用户**。

```text
模型 A 输出: [v1, v2, v3, v4, v5]
模型 B 输出: [w1, w2, w3, w4, w5]
交错输出:   [v1, w1, v2, w2, v3, w3, v4, w4, v5, w5]
```

**优点:**

- 同样的样本量,统计功效比 A/B 高 5-100 倍
- 减少样本量需求

**缺点:**

- 实现复杂
- 只能用于排序类实验(召回/精排/重排)
- 不能用于 UI 改版

```python
def interleaved_rank(model_a_scores, model_b_scores, k=10):
    """Team-Draft Interleaving"""
    a_remaining = list(zip(range(len(model_a_scores)), model_a_scores))
    b_remaining = list(zip(range(len(model_b_scores)), model_b_scores))

    a_remaining.sort(key=lambda x: -x[1])
    b_remaining.sort(key=lambda x: -x[1])

    interleaved = []
    attribution = []  # 每个 item 来自哪个模型

    while len(interleaved) < k and (a_remaining or b_remaining):
        # 抛硬币决定哪个模型先选
        if not a_remaining:
            winner = 'b'
        elif not b_remaining:
            winner = 'a'
        else:
            winner = 'a' if random.random() < 0.5 else 'b'

        if winner == 'a' and a_remaining:
            item_id, _ = a_remaining.pop(0)
            interleaved.append(item_id)
            attribution.append('a')
        elif winner == 'b' and b_remaining:
            item_id, _ = b_remaining.pop(0)
            interleaved.append(item_id)
            attribution.append('b')

    return interleaved, attribution

# 计算偏好
def interleaving_preference(attribution, user_clicks):
    """根据用户点击计算用户偏好哪个模型"""
    a_clicks = sum(1 for i, c in enumerate(user_clicks) if c and attribution[i] == 'a')
    b_clicks = sum(1 for i, c in enumerate(user_clicks) if c and attribution[i] == 'b')
    return 'a' if a_clicks > b_clicks else ('b' if b_clicks > a_clicks else 'tie')
```

---

## 5. 高级话题

### 5.1 多重检验校正 (Bonferroni / FDR)

**多重检验 = 一次实验测多个指标**,问题:"做 20 个检验,即使 H₀ 全为真,也会有 1 个假阳性(p < 0.05)"。

**Bonferroni 校正:**

```python
def bonferroni_correct(p_values, alpha=0.05):
    """Bonferroni 校正"""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]

# 例:5 个指标
p_values = [0.03, 0.04, 0.045, 0.06, 0.5]
results = bonferroni_correct(p_values)
# 输出:[False, False, False, False, False](因为 0.05/5 = 0.01)
```

**FDR (False Discovery Rate) 校正:**

更宽松的校正,允许少量假阳性。

```python
from statsmodels.stats.multitest import multipletests

def fdr_correct(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR 校正"""
    rejected, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return rejected, corrected_p

# 例
p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
rejected, corrected = fdr_correct(p_values)
```

**工业界实践:**

- 多个核心指标用 Bonferroni
- 探索性分析用 FDR

### 5.2 新奇效应 (Novelty Effect)

**新奇效应 = 用户对新策略有好奇心,初期效果虚高**。

```text
CTR 趋势:
  Day 1: +10% (新奇效应)
  Day 3: +6%
  Day 7: +3%
  Day 14: +2%  ← 真实效应
```

**应对:**

1. 延长实验时间(14 天+)
2. 分析用户分层:新用户 vs 老用户(老用户新奇效应更强)
3. 用 Day 7+ 的数据做主判断

### 5.3 学习效应

**学习效应 = 用户慢慢适应新策略,效果逐渐提升**。

```text
CTR 趋势:
  Day 1: +1%
  Day 3: +3%
  Day 7: +5%
  Day 14: +6%  ← 真实效应
```

**应对:**

1. 实验时长足够
2. 关注末期数据
3. 用混合效应模型分析趋势

### 5.4 季节性影响

**季节性 = 业务有日/周/月/年周期性**。

| 周期 | 现象 | 实验影响 |
|------|------|----------|
| 日 | 早晚高峰 | 1 天实验可能不均 |
| 周 | 周末高 | 1 周实验覆盖完整周期 |
| 月 | 月初/月末效应 | 大改需要覆盖完整月 |
| 年 | 大促/节日 | 大促前后不能做实验 |

**应对:**

1. 至少覆盖 1 个完整业务周期
2. 节假日/大促前后不做实验
3. 在分析时分周期查看数据

### 5.5 长期效应 vs 短期效应

**短期效应 = 用户即时反馈**(CTR、停留时长)
**长期效应 = 用户长期行为**(留存、LTV)

**核心矛盾:** 短期最优 ≠ 长期最优

**典型场景:**

1. **标题党**:短期 CTR 高,但长期留存低
2. **推送骚扰**:短期点击高,但用户长期流失
3. **强插广告**:短期 GMV 高,但用户体验恶化

**应对:**

| 策略 | 做法 |
|------|------|
| 多窗口实验 | 同时看 1 天/7 天/30 天效果 |
| Holdout 长期跟踪 | 5% 用户永不实验,看长期趋势 |
| 长期指标做护栏 | 留存、LTV 作为反向指标 |
| LTV 建模 | 直接建模用户生命周期价值 |

```python
# 多窗口效果分析
def multi_window_analysis(exp_id, windows=[1, 7, 30]):
    """多窗口效果分析"""
    results = {}
    for window in windows:
        results[f'day_{window}'] = analyze_experiment(exp_id, window)
    return results

# 输出示例:
# {
#   'day_1':  {'ctr_lift': '+5%', 'significant': True},
#   'day_7':  {'ctr_lift': '+2%', 'significant': True},
#   'day_30': {'ctr_lift': '+1%', 'significant': False}
# }
```

---

## 6. 因果推断

### 6.1 因果图 (Causal Graph)

**因果图 = 用有向无环图 (DAG) 表示变量间的因果关系**。

```text
[Treatment X] ──→ [Outcome Y]
       ↓                ↑
       └→ [Confounder Z] ─┘  (混杂因子)
```

**关键概念:**

| 概念 | 含义 |
|------|------|
| **Treatment (X)** | 干预变量(实验组) |
| **Outcome (Y)** | 结果变量(CTR、留存) |
| **Confounder (Z)** | 同时影响 X 和 Y 的混杂因子 |
| **Mediator (M)** | 中介变量(X → M → Y) |
| **Collider (C)** | X → C ← Y |

**3 条核心规则:**

1. **链式**:X → Z → Y,控制 Z 阻断路径
2. **分叉**:X ← Z → Y,控制 Z 阻断路径
3. **对撞**:X → Z ← Y,**不要**控制 Z,否则打开路径

### 6.2 反事实推理

**反事实 = "如果当时做了不同选择,会怎样?"**

反事实推理的核心:估计**个体处理效应 (ITE)**。

```text
ITE(u) = Y_i(u, T=1) - Y_i(u, T=0)
```

但 Y_i(u, T=0) 是观察不到的(用户要么在对照组,要么在实验组)。

**平均处理效应 (ATE):**

```text
ATE = E[Y(T=1) - Y(T=0)]
```

**条件平均处理效应 (CATE):**

```text
CATE(u) = E[Y(T=1) - T=0) | U=u]
```

### 6.3 DAG 与 do-calculus

**Pearl 的 do-calculus** = 因果推断的数学框架。

```text
P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)
```

**核心操作:**

1. `do(X=x)`:强制把 X 设为 x
2. `P(Y | do(X=x))`:X=x 时 Y 的分布
3. 与条件概率 P(Y | X=x) 不同——后者可能有混杂

**后门调整 (Backdoor Adjustment):**

```python
# 后门调整:控制所有后门路径上的混杂因子
def backdoor_adjustment(X, Y, Z, data):
    """
    X: treatment
    Y: outcome
    Z: 满足后门准则的变量集(阻断所有 X 到 Y 的后门路径)
    """
    # P(Y | do(X)) = Σ_z P(Y | X, Z=z) P(Z=z)
    effect = 0
    for z_val in data[Z].unique():
        subset = data[data[Z] == z_val]
        p_z = len(subset) / len(data)
        # P(Y | X=1, Z=z) - P(Y | X=0, Z=z)
        cond_effect = (
            subset[subset[X] == 1][Y].mean() -
            subset[subset[X] == 0][Y].mean()
        )
        effect += cond_effect * p_z
    return effect
```

### 6.4 工具变量

**工具变量 (IV) = 一个影响 X 但只通过 X 影响 Y 的变量**。

```text
       [IV Z]
          ↓
[Confounder U] ──→ [X] ──→ [Y]
```

**工具变量要满足:**

1. **相关性**:Z 与 X 强相关
2. **排他性**:Z 只通过 X 影响 Y
3. **独立性**:Z 与混杂因子独立

**经典示例:**

- 研究"教育对收入的影响":用"出生季度"作为 IV(影响是否按时入学,但不直接影响收入)

**在推荐系统中:**

- 研究"推荐算法对点击的影响":用"随机化排序扰动"作为 IV

### 6.5 倾向得分

**倾向得分 (PS) = 给定混杂变量时,接受处理的概率**。

```text
PS(u) = P(T=1 | X=u)
```

**用法:**

1. 计算每个用户的 PS
2. 匹配相似 PS 的用户(对照组 vs 实验组)
3. 比较匹配后两组的 Y

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(treatment, outcome, confounders, data):
    """倾向得分匹配"""
    # 1. 估计倾向得分
    model = LogisticRegression()
    model.fit(data[confounders], data[treatment])
    ps = model.predict_proba(data[confounders])[:, 1]
    data['ps'] = ps

    # 2. 匹配
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control`'ps'`)
    distances, indices = nn.kneighbors(treated`'ps'`)

    matched_control = control.iloc[indices.flatten()]

    # 3. 比较
    ate = treated[outcome].mean() - matched_control[outcome].mean()
    return ate

# 使用
ate = propensity_score_matching(
    treatment='got_treatment',
    outcome='retained',
    confounders=['age', 'gender', 'prior_clicks', 'session_count'],
    data=user_data
)
```

### 6.6 双重差分 (DiD)

**双重差分 (Difference-in-Differences) = 比较处理组与对照组在干预前后的差异**。

```text
                干预前     干预后
处理组:        Y_t_pre    Y_t_post
对照组:        Y_c_pre    Y_c_post

DiD = (Y_t_post - Y_t_pre) - (Y_c_post - Y_c_pre)
```

**适用场景:**

- 不能完全随机化(如政策评估)
- 自然实验(如某地区上线新功能)
- 历史对比(对照当前 vs 去年同时段)

```python
def did_estimate(y_treat_post, y_treat_pre, y_ctrl_post, y_ctrl_pre):
    """双重差分估计"""
    treat_diff = y_treat_post - y_treat_pre
    ctrl_diff = y_ctrl_post - y_ctrl_pre
    return treat_diff - ctrl_diff

# 推荐系统示例:某地区上线新召回通道
# 处理组(新召回): 干预前 CTR=0.05, 干预后 CTR=0.055
# 对照组(老召回): 干预前 CTR=0.052, 干预后 CTR=0.053
treat_diff = 0.055 - 0.05  # 0.005
ctrl_diff = 0.053 - 0.052  # 0.001
did = treat_diff - ctrl_diff  # 0.004
# 真实因果效应 ≈ 0.004
```

**DiD 的关键假设:平行趋势(Parallel Trends)**——干预前,处理组和对照组应有相似的趋势。

---

## 7. 推荐系统特有的 A/B 测试

### 7.1 多目标 A/B

**多目标 A/B = 同时测多个目标的提升**。

```python
# 推荐系统的典型多目标
metrics = {
    'ctr': {'direction': 'increase', 'weight': 0.4},
    'watch_time': {'direction': 'increase', 'weight': 0.3},
    'like_rate': {'direction': 'increase', 'weight': 0.2},
    'skip_rate': {'direction': 'decrease', 'weight': 0.1},
}

# 综合得分
def composite_score(metrics_results):
    score = 0
    for metric, result in metrics_results.items():
        if result['significant']:
            score += metrics[metric]['weight'] * result['lift']
    return score
```

**加权融合的局限:** 不同指标的最优模型可能不同。

**替代方案:**

1. **Pareto 优化**:找 Pareto 前沿上的最优解
2. **多目标帕累托 A/B**:同时报告多个指标
3. **约束优化**:固定一个指标,优化另一个

### 7.2 长期价值评估

**LTV (Lifetime Value) = 用户生命周期价值**。

**短期指标的局限:**

| 短期 | 长期 |
|------|------|
| CTR | 留存 |
| CVR | 复购 |
| 时长 | LTV |
| 分享 | 推荐带来新用户 |

**评估 LTV 的实验设计:**

```text
1. 短期实验(14 天):验证短期指标
2. 中期实验(3 个月):验证留存
3. 长期跟踪(1 年+):验证 LTV
```

**Holdout 群组:**

保留 5% 用户长期跟踪,3-5 年后再看 LTV。

### 7.3 反事实评估

**反事实评估 = 用历史数据估计"如果用新模型,会怎样"**。

```python
class InversePropensityScore:
    """IPS 估计器"""
    def __init__(self, logging_policy):
        self.logging_policy = logging_policy  # 记录策略

    def estimate(self, new_policy, logged_data):
        """
        logged_data: (context, action, reward, prop_score)
        """
        ips_estimate = 0
        total_weight = 0

        for ctx, action, reward, prop in logged_data:
            # 新策略选择这个 action 的概率
            new_prop = new_policy(ctx, action)
            # IPS 权重
            w = new_prop / max(prop, 1e-6)
            ips_estimate += w * reward
            total_weight += w

        return ips_estimate / total_weight
```

**反事实评估的局限:**

1. 只能在已记录的 action 上评估(支持集有限)
2. 权重方差大(IPS 估计)
3. 不能用于全新策略

### 7.4 Interleaving 评估

**Interleaving 在推荐系统中的优势:**

| 对比 | A/B | Interleaving |
|------|-----|--------------|
| 样本量需求 | 大 | 小 5-100 倍 |
| 灵敏度 | 低 | 高 |
| 实现复杂度 | 低 | 中 |
| 适用范围 | 所有 | 仅排序 |

**典型流程:**

```text
1. 准备两套候选排序结果
2. 交错合并,展示给用户
3. 收集用户反馈(点击、时长)
4. 根据反馈计算两套模型的"胜率"
5. 胜率 > 95% 显著才算胜出
```

---

## 8. 真实案例

### 8.1 字节抖音 A/B 测试体系

字节的 A/B 平台是**业界最成熟**之一,核心特点:

**平台能力:**

- **流量管理**:支持任意流量比例,精准分流
- **实验配置**:可视化配置,支持 MVT / MAB / Switchback
- **指标体系**:统一指标中心,200+ 业务指标
- **实时分析**:T+0 数据回流,小时级出报告
- **自动报警**:异常自动暂停实验

**实验规模:**

- 每天同时在线实验:2000+
- 流量利用率:70%+
- 实验人员覆盖:全员(产品、算法、研发)

**关键设计:**

```yaml
# 字节的实验配置示例
experiment:
  name: "feed_ranker_v4_online"
  owner: "rank_team"

  layer:
    name: "ranker_online"
    type: "exclusive"
    traffic: 100%

  experiments:
    - name: "ranker_v4"
      traffic_ratio: 0.10
      variants:
        - {name: "control", weight: 0.5}
        - {name: "treatment_v4", weight: 0.5}

  metrics:
    primary: ["ctr", "watch_time_per_session"]
    guardrail: ["p99_latency", "error_rate"]

  duration:
    min_days: 7
    max_days: 30
    auto_stop: false
```

### 8.2 Meta 平台实验

Meta (Facebook) 的实验平台核心:

1. **PlanOut**:开源的实验配置语言
2. **Deltoid**:实验分流引擎
3. **AutoLift**:自动检测实验异常

**PlanOut 示例:**

```python
from planout.ops.random import *
import planout

class FeedExperiment(planout.Experiment):
    def assign(self, params, user_id):
        params.exposure = BernoulliTrial(p=0.5, salt='exposure')
        if params.exposure:
            params.rank_variant = UniformChoice(
                choices=['control', 'treatment'],
                unit=user_id,
                salt='rank_variant'
            )
            if params.rank_variant == 'treatment':
                params.boost_score = RandomFloat(
                    min=0.0, max=0.2, unit=user_id, salt='boost'
                )
```

### 8.3 阿里妈妈

阿里妈妈(阿里广告)的实验体系特点:

1. **大规模并发**:万级实验同时跑
2. **智能流量调度**:基于实时效果的自动调权
3. **广告主权益保护**:新策略不损害广告主 ROI
4. **跨场景实验**:电商、直播、本地生活统一平台

**广告特有的实验设计:**

```text
1. 创意层实验(标题/图片/视频)
2. 定向层实验(人群定向)
3. 出价层实验(ocpc/ocpm)
4. 排序层实验(粗排/精排)
5. 落地页实验
```

---

## 9. 工具与平台

### 9.1 自研平台

**自研 vs 商用平台:**

| 维度 | 自研 | 商用 |
|------|------|------|
| 灵活性 | 高 | 中 |
| 成本 | 高 | 中 |
| 维护 | 重 | 轻 |
| 与内部系统集成 | 易 | 难 |

**字节 Libra:**

字节内部统一的 A/B 实验平台,支持 2000+ 并发实验。

**阿里 PEX:**

阿里妈妈的实验平台,专门为广告设计。

**美团 Xavier:**

美团的 A/B 平台,支持多业务。

### 9.2 Optimizely / LaunchDarkly

**Optimizely** = 商用 A/B 测试平台,主打 Web/移动端 UI 测试。

```python
# Optimizely SDK 用法
from optimizely import optimizely

optimizely_client = optimizely.Optimizely(sdk_key='your-sdk-key')

# 激活实验
if optimizely_client.is_feature_enabled('new_ranker', user_id):
    variant = optimizely_client.get_feature_variable_string('new_ranker', 'model', user_id)
    if variant == 'v2':
        return rank_v2(user_ctx)
```

**LaunchDarkly** = Feature Flag 平台,功能类似,但更偏工程化。

### 9.3 统计计算 (R / Python)

**R 统计生态:**

```r
# t 检验
t.test(group_a, group_b)

# 比例检验
prop.test(c(success_a, success_b), c(n_a, n_b))

# 样本量计算
power.prop.test(p1=0.05, p2=0.055, power=0.8)

# ANOVA(多组)
aov(metric ~ variant, data=df)

# 回归分析(协变量调整)
model <- lm(ctr ~ treatment + age + gender + prior_engagement, data=df)
```

**Python 统计栈:**

```python
# 常用库
import statsmodels.api as sm
from scipy import stats
from causalimpact import CausalImpact  # 因果推断
from dowhy import DoWhy  # 因果图

# 比例检验
statsmodels.stats.proportion.proportions_ztest(
    count=[success_a, success_b],
    nobs=[n_a, n_b]
)

# 回归分析
import statsmodels.formula.api as smf
model = smf.ols('ctr ~ treatment + age + gender', data=df).fit()
print(model.summary())

# 因果推断 - DoWhy
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment='got_treatment',
    outcome='retained',
    common_causes=['age', 'gender', 'prior_engagement']
)

identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression')
print(estimate)
```

---

## 10. 常见陷阱

### 10.1 用户分桶错误

**陷阱 1:哈希函数不一致**

```python
# 错误:不同语言/版本的 hash 结果不同
bucket = hash(user_id) % 100

# 正确:使用一致的 MD5/SHA
import hashlib
bucket = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16) % 100
```

**陷阱 2:hash 输入变了**

```python
# 错误:hash 输入加了时间戳,每次分桶不同
bucket = hash(f"{user_id}:{time.time()}") % 100

# 正确:hash 输入稳定
bucket = hash(f"{user_id}:ranker_v3") % 100  # 实验 ID 必须稳定
```

**陷阱 3:跨实验相关性**

```python
# 错误:两个实验用了相同的 hash
bucket = hash(f"{user_id}:experiment") % 100

# 正确:每个实验独立的 salt
bucket = hash(f"{user_id}:experiment_v1") % 100
bucket2 = hash(f"{user_id}:experiment_v2") % 100
```

### 10.2 流量倾斜

**陷阱 1:实验初期流量突增**

```text
新功能上线 → 用户新鲜感 → 实验组点击暴涨
```

**应对:** 实验初期 1-2 天数据不参与分析。

**陷阱 2:外部事件干扰**

```text
春节 → 全站流量下降 → 所有指标下降
```

**应对:** 同时段对照组对比,而不是历史对比。

**陷阱 3:实验组用户画像偏移**

```text
实验组配置:50% 北京 + 50% 上海
对照组配置:100% 成都
```

**应对:** 用 stratified sampling 保证用户分布一致。

### 10.3 辛普森悖论

**辛普森悖论 = 整体趋势与子组趋势相反**。

```text
              总样本
对照组转化率: 1000/10000 = 10%
实验组转化率: 1100/10000 = 11%
→ 实验组更好

但分用户等级看:
新用户:    对照 200/2000=10%  实验 100/1000=10%   一样
老用户:    对照 800/8000=10%  实验 1000/9000=11% 实验稍好

→ 看似实验组更好,但只是因为实验组老用户占比更高
```

**应对:**

1. 分层分析(按用户画像分桶)
2. 协变量调整(回归)
3. CUPED 方差缩减技术

**CUPED (Controlled-experiment Using Pre-Experiment Data):**

```python
def cuped_adjustment(y, y_pre, theta):
    """用实验前数据协变量调整"""
    y_adjusted = y - theta * (y_pre - y_pre.mean())
    return y_adjusted

# theta 通过回归估计:y ~ y_pre
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(y_pre.reshape(-1, 1), y)
theta = reg.coef_[0]

y_adjusted = cuped_adjustment(y, y_pre, theta)
# 调整后方差更小,功效更高
```

---

## 11. 与 RL/GR 的连接

### 11.1 RL 微调后如何 A/B 测试

**RL 训练的 A/B 测试难点:**

1. **状态分布偏移**:RL 策略会改变状态分布
2. **长期价值**:RL 优化长期回报,但短期指标可能下降
3. **安全风险**:RL 探索可能产生有害动作

**RL A/B 的特殊设计:**

```yaml
rl_experiment:
  # 1. 影子模式阶段:RL 模型与生产模型并行,只观察不干预
  shadow_mode:
    duration_days: 7
    traffic_ratio: 1.0  # 100% 流量都走,但 RL 不实际生效

  # 2. 小流量试投
  small_traffic:
    duration_days: 7
    traffic_ratio: 0.01

  # 3. 长期效果跟踪
  long_term:
    metrics:
      - 短期:ctr/cvr
      - 中期:7天留存
      - 长期:lTV
    duration_days: 90
```

**RL 的特殊指标:**

- **Reward 实测值**:实际获得的 reward 与预期是否一致
- **状态分布**:RL 是否把用户引导到预期状态
- **长期累积 reward**:用户整个会话/生命周期的总 reward

### 11.2 GR 系统的 A/B 设计

**GR (Generative Recommendation) 的 A/B 难点:**

1. **生成质量难评估**:GR 输出的序列质量难以量化
2. **新颖性偏差**:GR 可能生成全新的物品,没有历史数据
3. **生成多样性**:GR 可能倾向于生成相似物品

**GR 的 A/B 指标设计:**

```yaml
gr_metrics:
  # 基础指标
  base:
    - ctr
    - cvr
    - watch_time

  # 生成质量指标
  generation_quality:
    - intra_list_diversity  # 列表内多样性
    - novelty              # 新颖性
    - serendipity          # 惊喜度
    - coverage             # 覆盖率

  # 长期指标
  long_term:
    - 7d_retention
    - 30d_ltv
```

**GR 评估的特殊方法:**

- **Interleaving 必用**:GR 输出是序列,Interleaving 比 A/B 灵敏度高
- **反事实评估**:用 IPS/DM 估计器
- **人类评估**:关键决策点需要人工评估

```python
def evaluate_gr_quality(generated_items, user_history):
    """GR 输出质量评估"""
    return {
        'intra_diversity': 1 - mean_pairwise_similarity(generated_items),
        'novelty': mean(item_novelty_score(item, user_history) for item in generated_items),
        'coverage': len(set(generated_items)) / len(generated_items),
        'serendipity': mean(serendipity_score(item, user_history) for item in generated_items)
    }
```

---

## 12. 资源与延伸

### 工业实践
- [字节跳动: A/B 实验平台演进](https://mp.weixin.qq.com/)
- [Meta: PlanOut 开源](https://github.com/facebookarchive/planout)
- [Uber: Michelangelo 平台](https://www.uber.com/blog/)
- [Microsoft: ExP Platform](https://www.microsoft.com/en-us/research/project/experimentation-platform/)

### 学术论文
- [Controlled Experiments at Scale(LinkedIn)](https://www.kdd.org/exploration_files/p083-lin.pdf)
- [Trustworthy Online Controlled Experiments(Kohavi 等)](https://www.cambridge.org/press/)
- [Improving the Sensitivity of Online Controlled Experiments(微软 CUPED)](https://www.microsoft.com/en-us/research/publication/improving-the-sensitivity-of-online-controlled-experiments-by-utilizing-pre-experiment-data/)

### 书籍
- 《Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing》(Ron Kohavi 等, 2020)
- 《Causal Inference for Statistics, Social, and Biomedical Sciences》(Imbens & Rubin, 2015)
- 《The Book of Why》(Judea Pearl, 2018)

### 工具
- [Optimizely](https://www.optimizely.com/)
- [LaunchDarkly](https://launchdarkly.com/)
- [Eppo](https://www.geteppo.com/)
- [Statsig](https://statsig.com/)

### Python 库
- [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels](https://www.statsmodels.org/)
- [DoWhy](https://github.com/py-why/dowhy)
- [CausalImpact](https://github.com/google/CausalImpact)
- [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs)

---

## 附录 A: A/B 实验 Checklist

**实验前:**
- [ ] 明确假设(预期效果方向)
- [ ] 计算样本量
- [ ] 设计实验单位、流量比例、时长
- [ ] 定义核心指标 + 护栏指标
- [ ] 写好实验配置(实验名、描述、负责人)
- [ ] 配置监控和告警
- [ ] 准备回滚预案

**实验中:**
- [ ] 每日检查数据健康度
- [ ] 关注护栏指标
- [ ] 关注流量分布(对照组 vs 实验组)
- [ ] 注意外部事件(节假日、热点)
- [ ] 定期同步进展

**实验后:**
- [ ] 等统计功效达标
- [ ] 多窗口分析(1d/7d/30d)
- [ ] 分用户群分析(新老用户、画像)
- [ ] 显著性检验 + 置信区间
- [ ] 写实验报告
- [ ] 决策:全量、回滚、保留
- [ ] 归档数据

---

## 附录 B: 常用统计检验速查

| 场景 | 检验 | Python |
|------|------|--------|
| 两组连续均值 | t 检验 | `scipy.stats.ttest_ind` |
| 两组比例 | z 检验 | `statsmodels.proportion.proportions_ztest` |
| 多组均值 | ANOVA | `scipy.stats.f_oneway` |
| 多组比较 | Tukey HSD | `statsmodels.stats.multicomp.pairwise_tukeyhsd` |
| 分类变量 | 卡方 | `scipy.stats.chi2_contingency` |
| 非正态 | Mann-Whitney | `scipy.stats.mannwhitneyu` |
| 相关性 | Pearson / Spearman | `scipy.stats.pearsonr/spearmanr` |
| 协变量调整 | 回归 | `statsmodels.formula.api.ols` |
| 方差缩减 | CUPED | 自实现 |

---

## 附录 C: 因果推断方法速查

| 方法 | 适用 | 关键假设 |
|------|------|----------|
| 随机实验 | 黄金标准 | 随机化成功 |
| 倾向得分匹配 | 观察数据 | 无未观测混杂 |
| 工具变量 | 自然实验 | IV 满足排他性 |
| 双重差分 | 面板数据 | 平行趋势 |
| 合成控制 | 单处理单位 | 多个对照单位 |
| 回归不连续 | 阈值附近 | 连续性 |

---

> 最后更新: 2026-07-23 · 作者: 算法工程师

---

## 扩展 A · AB测试与因果推断 的可执行手册（2026-07-24）

> 本扩展围绕“区分相关性与因果性，在分流、指标、统计功效和长期影响之间建立可靠实验”补充决策表、练习卡和复盘证据。它不替代正文，而是把正文转化为可以执行、检查和复用的工作单元。

### A.1 使用规则

1. 先选一个对象，不要同时打开整章；
2. 先写假设和约束，再选择方法或工具；
3. 每次只改变一个主要变量，保留原始证据；
4. 结论必须连接到一个指标、一个链接或一个可运行的最小例子；
5. 没有通过验证的内容，标记为“待证据”，不要写成确定事实。

### A.2 核心对象与证据门槛

| 编号 | 对象 | 首要动作 | 最小产出 | 状态 |
|---|---|---|---|---|
| 01 | 实验假设 | 写出反事实 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 02 | 分流单元 | 确定分析单元 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 03 | 随机化 | 检查SRM | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 04 | 样本量 | 计算最小样本 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 05 | 主指标 | 锁定分析口径 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 06 | 护栏指标 | 预注册停止规则 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 07 | CUPED | 切分用户分层 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 08 | 异质性 | 解释不显著结果 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 09 | 新奇效应 | 写出反事实 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 10 | 长期留存 | 确定分析单元 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 11 | 干扰效应 | 检查SRM | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |
| 12 | 实验复盘 | 计算最小样本 | 一页纸记录 + 1 个可复核证据 | 未完成 / 进行中 / 已验证 |

### A.3 场景决策卡

| # | 触发场景 | 关注对象 | 建议动作 | 停止条件 |
|---|---|---|---|---|
| 1 | 数据规模小 | 实验假设 | 计算最小样本 | 先做最小验证；若证据不足则回到问题定义 |
| 2 | 数据规模小 | 分流单元 | 锁定分析口径 | 先做最小验证；若证据不足则回到问题定义 |
| 3 | 数据规模小 | 随机化 | 预注册停止规则 | 先做最小验证；若证据不足则回到问题定义 |
| 4 | 数据规模小 | 样本量 | 切分用户分层 | 先做最小验证；若证据不足则回到问题定义 |
| 5 | 数据规模小 | 主指标 | 解释不显著结果 | 先做最小验证；若证据不足则回到问题定义 |
| 6 | 数据规模小 | 护栏指标 | 写出反事实 | 先做最小验证；若证据不足则回到问题定义 |
| 7 | 数据规模小 | CUPED | 确定分析单元 | 先做最小验证；若证据不足则回到问题定义 |
| 8 | 数据规模小 | 异质性 | 检查SRM | 先做最小验证；若证据不足则回到问题定义 |
| 9 | 数据规模小 | 新奇效应 | 计算最小样本 | 先做最小验证；若证据不足则回到问题定义 |
| 10 | 数据规模小 | 长期留存 | 锁定分析口径 | 先做最小验证；若证据不足则回到问题定义 |
| 11 | 数据规模小 | 干扰效应 | 预注册停止规则 | 先做最小验证；若证据不足则回到问题定义 |
| 12 | 数据规模小 | 实验复盘 | 切分用户分层 | 先做最小验证；若证据不足则回到问题定义 |
| 13 | 流量突增 | 实验假设 | 解释不显著结果 | 先做最小验证；若证据不足则回到问题定义 |
| 14 | 流量突增 | 分流单元 | 写出反事实 | 先做最小验证；若证据不足则回到问题定义 |
| 15 | 流量突增 | 随机化 | 确定分析单元 | 先做最小验证；若证据不足则回到问题定义 |
| 16 | 流量突增 | 样本量 | 检查SRM | 先做最小验证；若证据不足则回到问题定义 |
| 17 | 流量突增 | 主指标 | 计算最小样本 | 先做最小验证；若证据不足则回到问题定义 |
| 18 | 流量突增 | 护栏指标 | 锁定分析口径 | 先做最小验证；若证据不足则回到问题定义 |
| 19 | 流量突增 | CUPED | 预注册停止规则 | 先做最小验证；若证据不足则回到问题定义 |
| 20 | 流量突增 | 异质性 | 切分用户分层 | 先做最小验证；若证据不足则回到问题定义 |
| 21 | 流量突增 | 新奇效应 | 解释不显著结果 | 先做最小验证；若证据不足则回到问题定义 |
| 22 | 流量突增 | 长期留存 | 写出反事实 | 先做最小验证；若证据不足则回到问题定义 |
| 23 | 流量突增 | 干扰效应 | 确定分析单元 | 先做最小验证；若证据不足则回到问题定义 |
| 24 | 流量突增 | 实验复盘 | 检查SRM | 先做最小验证；若证据不足则回到问题定义 |
| 25 | 指标冲突 | 实验假设 | 计算最小样本 | 先做最小验证；若证据不足则回到问题定义 |
| 26 | 指标冲突 | 分流单元 | 锁定分析口径 | 先做最小验证；若证据不足则回到问题定义 |
| 27 | 指标冲突 | 随机化 | 预注册停止规则 | 先做最小验证；若证据不足则回到问题定义 |
| 28 | 指标冲突 | 样本量 | 切分用户分层 | 先做最小验证；若证据不足则回到问题定义 |
| 29 | 指标冲突 | 主指标 | 解释不显著结果 | 先做最小验证；若证据不足则回到问题定义 |
| 30 | 指标冲突 | 护栏指标 | 写出反事实 | 先做最小验证；若证据不足则回到问题定义 |
| 31 | 指标冲突 | CUPED | 确定分析单元 | 先做最小验证；若证据不足则回到问题定义 |
| 32 | 指标冲突 | 异质性 | 检查SRM | 先做最小验证；若证据不足则回到问题定义 |
| 33 | 指标冲突 | 新奇效应 | 计算最小样本 | 先做最小验证；若证据不足则回到问题定义 |
| 34 | 指标冲突 | 长期留存 | 锁定分析口径 | 先做最小验证；若证据不足则回到问题定义 |
| 35 | 指标冲突 | 干扰效应 | 预注册停止规则 | 先做最小验证；若证据不足则回到问题定义 |
| 36 | 指标冲突 | 实验复盘 | 切分用户分层 | 先做最小验证；若证据不足则回到问题定义 |
| 37 | 资源受限 | 实验假设 | 解释不显著结果 | 先做最小验证；若证据不足则回到问题定义 |
| 38 | 资源受限 | 分流单元 | 写出反事实 | 先做最小验证；若证据不足则回到问题定义 |
| 39 | 资源受限 | 随机化 | 确定分析单元 | 先做最小验证；若证据不足则回到问题定义 |
| 40 | 资源受限 | 样本量 | 检查SRM | 先做最小验证；若证据不足则回到问题定义 |
| 41 | 资源受限 | 主指标 | 计算最小样本 | 先做最小验证；若证据不足则回到问题定义 |
| 42 | 资源受限 | 护栏指标 | 锁定分析口径 | 先做最小验证；若证据不足则回到问题定义 |
| 43 | 资源受限 | CUPED | 预注册停止规则 | 先做最小验证；若证据不足则回到问题定义 |
| 44 | 资源受限 | 异质性 | 切分用户分层 | 先做最小验证；若证据不足则回到问题定义 |
| 45 | 资源受限 | 新奇效应 | 解释不显著结果 | 先做最小验证；若证据不足则回到问题定义 |
| 46 | 资源受限 | 长期留存 | 写出反事实 | 先做最小验证；若证据不足则回到问题定义 |
| 47 | 资源受限 | 干扰效应 | 确定分析单元 | 先做最小验证；若证据不足则回到问题定义 |
| 48 | 资源受限 | 实验复盘 | 检查SRM | 先做最小验证；若证据不足则回到问题定义 |

### A.4 复盘记录模板

```markdown
# 任务标题

## 问题与约束
- 谁被什么阻塞？
- 约束来自数据、延迟、成本还是组织？

## 选择与证据
- 选择了什么？为什么不是另外两个选项？
- 证据链接：`相关笔记`
- 关键指标及口径：

## 结果与下一步
- 哪个假设被支持？
- 哪个假设被推翻？
- 下一次只改变什么？
```

### A.5 质量门禁

| # | 检查项 | 结果 | 证据 | 跟进 |
|---|---|---|---|---|
| 1 | 输入是否有时间边界 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 2 | 指标是否定义分母 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 3 | baseline 是否足够强 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 4 | 异常是否可重现 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 5 | 失败是否记录 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 6 | 资源成本是否可解释 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 7 | 结论是否超出证据 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 8 | 链接是否指向原文 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 9 | 代码是否能最小运行 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 10 | 是否有回滚或停止条件 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 11 | 是否区分离线与在线 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |
| 12 | 是否检查长尾子群 | 通过 / 不通过 / 不适用 | 写出一个证据位置 | 负责人 + 截止时间 |

### A.6 主动回忆题

| # | 对象 | 问题 | 回答要求 |
|---|---|---|---|
| 1 | 实验假设 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 2 | 分流单元 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 3 | 随机化 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 4 | 样本量 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 5 | 主指标 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 6 | 护栏指标 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 7 | CUPED | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 8 | 异质性 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 9 | 新奇效应 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 10 | 长期留存 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 11 | 干扰效应 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 12 | 实验复盘 | 如果把它删掉会怎样？ | 用三句话回答，并附一条内部链接 |
| 13 | 实验假设 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 14 | 分流单元 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 15 | 随机化 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 16 | 样本量 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 17 | 主指标 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 18 | 护栏指标 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 19 | CUPED | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 20 | 异质性 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 21 | 新奇效应 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 22 | 长期留存 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 23 | 干扰效应 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 24 | 实验复盘 | 最小替代方案是什么？ | 用三句话回答，并附一条内部链接 |
| 25 | 实验假设 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 26 | 分流单元 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 27 | 随机化 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 28 | 样本量 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 29 | 主指标 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 30 | 护栏指标 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 31 | CUPED | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 32 | 异质性 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 33 | 新奇效应 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 34 | 长期留存 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 35 | 干扰效应 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 36 | 实验复盘 | 什么数据会让结论失效？ | 用三句话回答，并附一条内部链接 |
| 37 | 实验假设 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 38 | 分流单元 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 39 | 随机化 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 40 | 样本量 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 41 | 主指标 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 42 | 护栏指标 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 43 | CUPED | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 44 | 异质性 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 45 | 新奇效应 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 46 | 长期留存 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 47 | 干扰效应 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |
| 48 | 实验复盘 | 谁会受到副作用影响？ | 用三句话回答，并附一条内部链接 |

### A.7 内部联动

- 本扩展的主链接：[[../工程与职业/0-工程师核心能力图谱|工程能力图谱]] · [[../工程与职业/6-代码质量与工程实践|代码质量]]
- 建议把完成后的任务卡回链到对应章节，不要只在日志里留下孤立结论。
- 如果一个结论同时影响工程、业务和论文阅读，优先补充到 知识库关联矩阵。

### 示例代码：python 自检

```python
from dataclasses import dataclass

@dataclass
class Evidence:
    claim: str
    metric: str
    source: str
    status: str = "待验证"

def close_loop(evidence):
    assert evidence.claim and evidence.metric and evidence.source
    return evidence.status in {"已验证", "部分验证"}

# 先把“观点”转换成带来源的证据对象，再决定是否推广结论。
```

### A.8 分阶段任务卡

| 阶段 | 对象 | 动作 | 证据 | 主要风险 |
|---|---|---|---|---|
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 写出反事实 | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 确定分析单元 | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 检查SRM | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 计算最小样本 | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 锁定分析口径 | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 预注册停止规则 | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 切分用户分层 | 论文段落 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 指标快照 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 代码片段 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 架构图 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 日志链接 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 实验表 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 会议纪要 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 失败记录 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 回滚记录 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 用户反馈 | 维护成本过高 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 口径漂移 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 时间穿越 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 样本偏置 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 资源不足 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 接口变化 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 异常未捕获 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 结论过度外推 |
| 定义 | 实验假设 | 解释不显著结果 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 写出反事实 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 确定分析单元 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 检查SRM | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 计算最小样本 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 锁定分析口径 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 预注册停止规则 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 切分用户分层 | 论文段落 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 指标快照 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 代码片段 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 架构图 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 日志链接 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 实验表 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 会议纪要 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 失败记录 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 回滚记录 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 用户反馈 | 维护成本过高 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 口径漂移 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 时间穿越 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 样本偏置 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 资源不足 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 接口变化 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 异常未捕获 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 结论过度外推 |
| 定义 | 分流单元 | 解释不显著结果 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 资源不足 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 接口变化 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 写出反事实 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 资源不足 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 接口变化 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 确定分析单元 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 架构图 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 架构图 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 架构图 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 架构图 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 架构图 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 实验表 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 实验表 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 实验表 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 实验表 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 实验表 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 资源不足 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 接口变化 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 检查SRM | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 资源不足 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 接口变化 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 计算最小样本 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 资源不足 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 接口变化 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 锁定分析口径 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 资源不足 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 接口变化 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 预注册停止规则 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 代码片段 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 架构图 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 日志链接 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 实验表 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 会议纪要 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 失败记录 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 回滚记录 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 用户反馈 | 维护成本过高 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 口径漂移 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 时间穿越 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 样本偏置 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 资源不足 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 接口变化 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 异常未捕获 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 结论过度外推 |
| 定义 | 随机化 | 切分用户分层 | 论文段落 | 维护成本过高 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 口径漂移 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 时间穿越 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 样本偏置 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 资源不足 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 接口变化 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 异常未捕获 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 结论过度外推 |
| 定义 | 随机化 | 解释不显著结果 | 指标快照 | 维护成本过高 |
| 定义 | 随机化 | 解释不显著结果 | 代码片段 | 口径漂移 |

### A.10 本扩展的完成标准

- [ ] 至少完成 3 张对象卡；
- [ ] 至少保留 1 个失败案例；
- [ ] 至少补充 2 条内部 `` `` 链接；
- [ ] 至少有 1 个可以运行或复核的证据；
- [ ] 结论写出适用边界和下一次动作；

> **维护提示**：本扩展服务于《AB测试与因果推断》，后续新增内容应优先更新表格中的证据列，而不是重复增加口号。
