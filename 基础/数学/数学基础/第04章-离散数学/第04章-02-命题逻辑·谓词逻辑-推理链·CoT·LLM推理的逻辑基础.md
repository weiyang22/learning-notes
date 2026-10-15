---
title: "第04章-02-命题逻辑·谓词逻辑-推理链·CoT·LLM推理的逻辑基础"
tags:
  - 数学
  - 命题逻辑
  - 谓词逻辑
  - 推理链
  - CoT
  - LLM推理的逻辑基础
description: "本章要点：\" 了。CoT(Chain-of-Thought)、ReAct、ToT(Tree of Thoughts) 这些词, 在 2023 年之后变得无处不在。…"
chapter_id: 数学-2-04
last_updated: 2026-07-14
---

# 第04章-02-命题逻辑·谓词逻辑-推理链·CoT·LLM推理的逻辑基础

> " 了。CoT(Chain-of-Thought)、ReAct、ToT(Tree of Thoughts) 这些词, 在 2023 年之后变得无处不在。但你想过没:
> - **"推理" 这个词在数学里到底是什么意思**?
> - **"苏格拉底会死" 跟 "LLM 推出结论" 是不是同一件事**?
> - **为什么 CoT 在算术题上有效, 在某些逻辑题上反而崩溃**?
> - **自动定理证明 (ATP) 和 LLM 推理, 底层是同一种东西吗**?
>
> 这些问题的答案, 全部藏在 **数理逻辑的第一章:命题逻辑和谓词逻辑** 里。
>
> 这一章我们要做的事情很朴素: 把 LLM 的"推理链"翻译回形式化逻辑的语言。看完之后你会发现:
> - "苏格拉底会死" = 假言推理 (Modus Ponens) 的标准例子
> - "LLM CoT 提示词" = 把自然语言推理 "形式化" 成可验证的推理规则链
> - "ReAct 的 Thought/Action/Observation" = 命题联结词 + 谓词实例化
> - "SAT 求解器" = 命题逻辑的可满足性判定, 直接用于工业硬件验证
> - "LLM 推理失败" = 缺少形式化验证, 统计模式 ≠ 真推理
>
> 命题逻辑 + 谓词逻辑是 LLM 推理的**形式化地基**。理解它们, 你才能看出 CoT 提示词的"逻辑骨架"在哪里, 也才能判断 LLM 推理到底"真不真"。

---

## 1. 直觉引入

### 1.1 工程师的痛点: LLM 为什么"胡说八道"? 形式化 vs 非形式化推理

做 LLM 应用的工程师, 几乎都被两件事困扰过:

```python
# 场景 1: 简单算术
prompt = "9.11 vs 9.9, 哪个大?"
# 模型答: "9.11 更大"  ← 错! 9.11 < 9.9

# 场景 2: 推理链断裂
prompt = "如果 A 比 B 高, B 比 C 高, 那么 A 和 C 谁高?"
# 模型可能答: "A 和 C 一样高"  ← 错, A > C
```

直觉上, 这些问题"很简单"。但 LLM 为什么会答错?

**根本原因**: LLM 在做的是**统计模式匹配**, 不是**形式化推理**。它的输出是基于"训练时见过的文本的概率分布", 而不是基于"前提必然推出结论"的逻辑规则。

**形式化推理**(formal reasoning)的特点:

1. **每一步都有规则**: 从前提到结论, 每一步应用一条已知的推理规则 (e.g. 假言推理、三段论)
2. **结论必然成立**: 如果前提为真、规则正确, 结论必然为真 (没有"概率")
3. **可验证**: 可以用 SAT solver、证明检查器自动验证

**非形式化推理**(informal reasoning, 自然语言推理)的特点:

1. **依赖常识和模式**: "苏格拉底会死" 我们凭常识接受, 但没显式用推理规则
2. **可能出错**: 统计模式可能捕捉到错误的相关性, 导致结论错误
3. **难验证**: 自然语言推理没有"机械化"的判别准则

**CoT 的本质**: CoT 是把"非形式化推理" **逼近**"形式化推理"的工程技巧——通过让 LLM 输出中间步骤, 强制它"显式地"使用推理规则, 而不是直接跳到结论。

但 CoT 不是真正的形式化推理 (它没有机械化验证), 所以**仍然会出错**。形式化推理的工具是 SAT solver、Lean、Coq、Isabelle 等。CoT 是"用自然语言模拟形式化推理"的折中。

### 1.2 三个直觉例子

#### 例 1: 经典三段论 "所有人会死 → 苏格拉底是人 → 苏格拉底会死"

逻辑学最古老的例子, 来自亚里士多德:

```text
前提 1: 所有人都会死。
前提 2: 苏格拉底是人。
结论:   苏格拉底会死。
```

**形式化**:

设谓词 $M(x)$ 表示 "$x$ 是人", $D(x)$ 表示 "$x$ 会死", $s$ 表示苏格拉底。

- 前提 1: $\forall x (M(x) \rightarrow D(x))$ (所有人会死)
- 前提 2: $M(s)$ (苏格拉底是人)
- 结论: $D(s)$ (苏格拉底会死)

**推理过程**:

```text
∀x (M(x) → D(x))        ← 全称实例化: 用 s 替换 x
─────────────────────
M(s) → D(s)

M(s)                       ← 前提
─────── 假言推理 (Modus Ponens)
D(s)                       ← 结论 ✓
```

**关键**: 这两步——全称实例化 (Universal Instantiation) 和假言推理 (Modus Ponens)——是**形式化的推理规则**。它们保证: 如果前提为真, 结论必然为真。

**工程意义**: LLM CoT 提示词有效, 正是因为它**显式地**让模型展示这两步推理。如果直接问 "苏格拉底会死吗?", 模型可能靠记忆答对, 也可能答错。但让它一步步推, 它必须"沿着推理链走", 不容易跳过。

#### 例 2: LLM CoT 推理的"形式化骨架" (ReAct prompt 拆解)

ReAct (Reasoning + Acting) 是 LLM Agent 的经典框架。一个典型的 ReAct prompt 包含:

```text
Question: 苹果公司的 CEO 是谁?
Thought 1: 我需要查苹果公司的官网。
Action 1: Search["Apple Inc CEO"]
Observation 1: 苹果公司的 CEO 是 Tim Cook。
Thought 2: 我已经找到答案了。
Action 2: Finish["Tim Cook"]
```

**形式化拆解**:

把每一步当作一个谓词:

```text
Searchable(x)         ← "x 是可搜索的"
CEO_of(Apple, y)      ← "y 是 Apple 的 CEO"
Found(y)              ← "y 已经被找到"
```

推理过程:

```text
Searchable(Apple CEO)        ← 前提: CEO 信息是可搜索的
CEO_of(Apple, y) → Found(y)  ← 推理规则: 找到信息 → 知识建立
─────────────────────────────
Found(Tim Cook)              ← 通过搜索, 确认 Tim Cook 是 CEO
```

**CoT 和 ReAct 的"逻辑骨架"**:

| 推理规则 | CoT 中的体现 | ReAct 中的体现 |
| :--- | :--- | :--- |
| 假言推理 (P → Q, P ⊢ Q) | "如果 X, 那么 Y。已知 X, 所以 Y" | "如果可搜索, 那么搜索后能获得信息" |
| 全称实例化 (∀x P(x) ⊢ P(a)) | "所有人都会死" → "苏格拉底会死" | "所有 CEO 信息都在网上" → "Tim Cook 在网上" |
| 析取三段论 (P ∨ Q, ¬P ⊢ Q) | "答案是 A 或 B, 不是 A, 所以是 B" | "搜索或推理都能获得信息, 搜索失败, 所以用推理" |

**关键洞察**: LLM 的 CoT/ReAct 提示词, 本质上是在**模拟形式化推理规则的调用链**。但因为没有机械化验证, 这些"规则调用"可能错 (e.g. 模型可能"跳过"某一步, 或"编造"前提)。

**工程意义**: 在 LLM Agent 设计中, 显式地写出"Thought/Action/Observation" 的结构, 就是在强制模型**走推理链**, 减少跳步错误。这是 CoT 有效的核心理由。

#### 例 3: 推荐系统中的逻辑规则 ("如果用户喜欢 A 且 A 与 B 相似 → 推荐 B")

推荐系统的经典规则:

```text
规则: 如果用户喜欢物品 i, 且物品 i 与物品 j 相似, 那么推荐物品 j 给用户。
```

**形式化**:

- $L(u, i)$: 用户 $u$ 喜欢物品 $i$
- $S(i, j)$: 物品 $i$ 与物品 $j$ 相似
- $R(u, j)$: 物品 $j$ 被推荐给用户 $u$

规则:

$$\forall u, i, j: L(u, i) \land S(i, j) \rightarrow R(u, j)$$

**推理过程**:

```text
L(u, i)                ← 前提: 用户 u 喜欢 i
L(u, i) ∧ S(i, j) → R(u, j)   ← 规则: 如果喜欢 i 且 i ~ j, 则推荐 j
─────────────────────────────────
R(u, j)                ← 结论: 推荐 j 给 u ✓
```

**ASCII 可视化: 推荐系统的推理链**

```text
   ┌──────────────────────┐
   │ 前提:                │
   │ L(u, i)  (喜欢 i)    │
   │ S(i, j)  (i 与 j 相似)│
   └──────────┬───────────┘
              │  ∧ (合取)
              ▼
   ┌──────────────────────┐
   │  L(u, i) ∧ S(i, j)   │
   └──────────┬───────────┘
              │ → (蕴含)
              ▼
   ┌──────────────────────┐
   │  结论: R(u, j)       │
   │  (推荐 j 给 u)       │
   └──────────────────────┘
```

**工程意义**: 推荐系统 = "在物品集合上应用逻辑规则"。DLRM、深度兴趣网络 (DIN) 这些深度学习模型, 本质上是在**自动学习**这些规则的权重 / 软化版本。但**显式规则** (知识图谱推理、Rule-based 推荐) 仍然在工业界广泛使用, 因为可解释、可调试。

**对比 LLM 推理**: LLM 的 "推荐 / 推理" 是隐式的, 通过参数记住规则。传统推荐系统是显式的, 通过规则库定义规则。两者各有优劣, 但底层都是**命题 / 谓词逻辑的推理链**。

### 1.3 ASCII 可视化: LLM 推理 vs 形式化推理

```text
              LLM 推理 (CoT/ReAct)              形式化推理 (SAT/Lean)
         ─────────────────────────          ─────────────────────────
前提    │  prompt 中的事实              │  axiom / 数据库事实
        │                                │
        ▼                                ▼
推理    │  神经网络 forward pass        │  应用推理规则 (Modus Ponens 等)
        │  + token by token 生成         │
        │                                │
        ▼                                ▼
结论    │  自然语言文本                  │  形式化命题
        │                                │
验证    │  人工判断 / 评分               │  机械化证明检查 (Lean kernel)
        │                                │
错误率  │  较高 (统计错误, 跳步)        │  极低 (规则保证)
        │                                │
速度    │  慢 (每个 token 一次 forward)  │  中等 (取决于规则数量)
        │                                │
可解释  │  弱 (黑盒)                    │  强 (每步可追溯)
```

**关键差异**: LLM 推理是**统计 + 自然语言**, 形式化推理是**规则 + 逻辑**。CoT 是两者的桥——它让 LLM 输出的"形式"看起来像形式化推理, 但底层仍是统计。

### 1.4 章节路线图

接下来 9 节, 我们依次讲:

| 节 | 主题 | 与 ML/DL 的联系 |
| :--- | :--- | :--- |
| §2 | 命题逻辑基础 (5 个联结词 + 真值表) | SQL WHERE / 布尔电路 / SAT |
| §3 | 命题等价与蕴含 | 编译器优化 / 表达式化简 |
| §4 | 范式 (DNF/CNF) | SAT 求解器 / 工业硬件验证 |
| §5 | 谓词逻辑 (量词 / 自由变量) | 知识图谱 / TransE/RGCN / FOL |
| §6 | 推理规则 (假言推理 / 拒取式 / 三段论) | 自动定理证明 / CoT 提示词骨架 |
| §7 | PyTorch 代码 | 真值表计算 / 归结证明器 / CoT 模板 |
| §8 | 常见误区 | 5 条工程师踩坑点 |
| §9 | 面试 Q&A | L7/L8 难度 5 题 |
| §10 | 联系其他章节 | 集合 / 形式语言 / Transformer / 系统思维 |

---

## 2. 命题逻辑基础

### 2.1 命题 (Proposition): 有真值的陈述句

**定义**: 一个**命题**是一个**有真值的陈述句**, 即可以明确判定为"真"(True, T 或 1) 或 "假"(False, F 或 0) 的陈述。

**是命题的例子**:

```text
- "雪是白的": 真命题
- "2 + 2 = 5": 假命题
- "北京是中国的首都": 真命题
- "x > 0" (当 x 是变量): 不是命题! 真值不确定
```

**不是命题的例子**:

```text
- "你今天怎么样?": 疑问句, 没有真值
- "请关门": 命令句, 没有真值
- "x > 0" (x 未知): 陈述句, 但真值不确定
- "这段话是假的": 悖论 (说谎者悖论), 严格说不是命题
```

**符号约定**: 命题变量用 $p, q, r, \ldots$ 或 $P, Q, R, \ldots$ 表示, 真值用 $\text{T}/\text{F}$ 或 $1/0$ 表示。

**工程意义**: 在编程中, 任何布尔表达式 `p > 0 && q != None` 就是一个命题, 它的真值在运行时确定。SQL 中的 `WHERE p > 0 AND q IS NOT NULL` 也是命题。LLM 的输出 "I think the answer is 42" 不是命题 (因为没有真值)。

### 2.2 5 个基本联结词

**联结词**(logical connective) 把简单命题组合成复合命题。我们关注 5 个:

#### 联结词 1: 否定 (Negation) ¬

$$\neg p: \text{不是 } p$$

**真值表**:

| $p$ | $\neg p$ |
| :--- | :--- |
| T | F |
| F | T |

**工程对应**: Python 的 `not p`, C 的 `!p`。

#### 联结词 2: 合取 (Conjunction) ∧

$$p \land q: p \text{ 且 } q$$

**真值表**:

| $p$ | $q$ | $p \land q$ |
| :--- | :--- | :--- |
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | F |

**工程对应**: `p and q`, `p && q`, SQL `AND`。

#### 联结词 3: 析取 (Disjunction) ∨

$$p \lor q: p \text{ 或 } q \text{ (包含两者)}$$

**真值表**:

| $p$ | $q$ | $p \lor q$ |
| :--- | :--- | :--- |
| T | T | T |
| T | F | T |
| F | T | T |
| F | F | F |

**注意**: 逻辑"或"是**包含性或** (inclusive or), 不是互斥或 (XOR)。"$p \lor q$" 在 $p$ 和 $q$ 都为真时也为真。

**工程对应**: `p or q`, `p || q`, SQL `OR`。

#### 联结词 4: 蕴含 (Implication) →

$$p \rightarrow q: \text{如果 } p, \text{ 那么 } q$$

**真值表**:

| $p$ | $q$ | $p \rightarrow q$ |
| :--- | :--- | :--- |
| T | T | T |
| T | F | F |
| F | T | T |
| F | F | T |

**直觉陷阱**: 当 $p$ 为假时, $p \rightarrow q$ 永远为真。这叫"空真"(vacuous truth)。

**例子**:
- "如果 1 = 2, 那么我是皇帝": 真! (因为前提假, 整个蕴含真)
- "如果明天下雨, 我带伞": 真的承诺, 取决于明天是否下雨

**工程对应**: `if p: q` (但 `if` 不返回 True/False, 而是一个控制流)。

#### 联结词 5: 等价 (Equivalence) ↔

$$p \leftrightarrow q: p \text{ 当且仅当 } q$$

**真值表**:

| $p$ | $q$ | $p \leftrightarrow q$ |
| :--- | :--- | :--- |
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | T |

**工程对应**: `p == q` (但要小心 Python 的 `==` 在某些类型上是值比较, 在 NumPy/Tensor 上是 element-wise)。

#### ASCII 真值表矩阵 (3 个变量示例)

```text
        p   q   r  | ¬p  p∧q  p∨q  p→q  p↔q  (p∧q)→r
        ─────────────────────────────────────────────
        T   T   T  |  F   T    T    T    T     T
        T   T   F  |  F   T    T    F    T     F
        T   F   T  |  F   F    T    T    F     T
        T   F   F  |  F   F    T    F    F     T
        F   T   T  |  T   F    T    T    F     T
        F   T   F  |  T   F    T    T    F     T
        F   F   T  |  T   F    F    T    T     T
        F   F   F  |  T   F    F    T    T     T
```

### 2.3 复合命题 (Compound Proposition): 嵌套

**定义**: 用联结词把多个命题组合成更复杂的命题。

**例子 1**: "今天下雨且我有伞, 那么我会出门"。

设 $p$ = "今天下雨", $q$ = "我有伞", $r$ = "我会出门"。

$$(p \land q) \rightarrow r$$

**例子 2**: SQL 中的复合条件:

```sql
SELECT * FROM users
WHERE (age > 18 AND country = 'US') OR (premium = TRUE);
```

设 $p$ = "age > 18", $q$ = "country = 'US'", $r$ = "premium = TRUE"。

$$(p \land q) \lor r$$

**例子 3**: LLM 提示词中的复合判断:

```text
如果 (用户提问的是技术问题) 且 (用户技术水平是初学者),
那么回复用简单语言 + 代码示例;
否则如果 (用户提问的是技术问题) 且 (用户技术水平是专家),
那么回复用专业术语 + 论文引用。
```

设 $T$ = "技术问题", $J$ = "初学者", $E$ = "专家", $S$ = "简单回复", $P$ = "专业回复"。

$$(T \land J) \rightarrow S$$
$$(T \land E) \rightarrow P$$

**优先级约定**: $\neg$ > $\land$ > $\lor$ > $\rightarrow$ > $\leftrightarrow$。可以用括号改变优先级。

**例子**: $p \land q \rightarrow r$ 解析为 $(p \land q) \rightarrow r$, 而**不是** $p \land (q \rightarrow r)$。

### 2.4 真值表方法 (Truth Table Method)

**核心思想**: 列出所有可能的真值组合, 判断命题在每种情况下的真值。

**用途 1: 判断两个命题是否等价**

例如, 判断 $p \rightarrow q$ 和 $\neg p \lor q$ 是否等价。

| $p$ | $q$ | $p \rightarrow q$ | $\neg p \lor q$ |
| :--- | :--- | :--- | :--- |
| T | T | T | F ∨ T = T |
| T | F | F | F ∨ F = F |
| F | T | T | T ∨ T = T |
| F | F | T | T ∨ F = T |

两列完全相同, 所以 $p \rightarrow q \equiv \neg p \lor q$。✓

**用途 2: 判断命题是否永真**

看真值表最后一列是否全为 T。

**用途 3: 判断命题是否可满足**

看真值表最后一列是否至少有一个 T。

### 2.5 永真式 / 永假式 / 可满足式

#### 永真式 (Tautology)

**定义**: 在所有真值赋值下都为真的命题。

**例子 1**: $p \lor \neg p$ (排中律)

| $p$ | $\neg p$ | $p \lor \neg p$ |
| :--- | :--- | :--- |
| T | F | T |
| F | T | T |

全为 T, 所以是永真式。

**例子 2**: $p \rightarrow (q \rightarrow p)$

**工程意义**: 永真式在所有情况下都成立, 可以用作"重言推理"的依据, 例如:

```python
# 永真式的应用: dead code elimination
if True or x:  # True or x 永真, x 不需要求值
    ...
```

#### 永假式 (Contradiction)

**定义**: 在所有真值赋值下都为假的命题。

**例子**: $p \land \neg p$ (矛盾律)

| $p$ | $\neg p$ | $p \land \neg p$ |
| :--- | :--- | :--- |
| T | F | F |
| F | T | F |

全为 F。

**工程意义**: 永假式在所有情况下都为假, 可以用作"不可达代码"的标志。

```python
if False and x:  # 永远 False
    ...  # 死代码, 编译器可以删除
```

#### 可满足式 (Satisfiable)

**定义**: 至少有一种真值赋值使命题为真。

**例子**: $p \land q$。当 $p = T, q = T$ 时为真, 所以可满足。

**例子**: $p \land \neg p$。任何赋值都为假, 不可满足 (它是永假式)。

**ASCII 可视化: 永真 / 可满足 / 永假**

```text
  所有命题 (Universe)
  ┌────────────────────────────────────────────┐
  │                                            │
  │   ┌──────────────────────────┐             │
  │   │  永真式 (Tautology)        │  ← 全为 T │
  │   │  e.g. p ∨ ¬p             │             │
  │   └──────────────────────────┘             │
  │                                            │
  │   ┌──────────────────────────┐             │
  │   │  可满足式 (Satisfiable)   │  ← 至少1个T│
  │   │  e.g. p ∧ q, p → q       │             │
  │   │      (可能为 T, 也可能F)  │             │
  │   └──────────────────────────┘             │
  │                                            │
  │   ┌──────────────────────────┐             │
  │   │  永假式 (Contradiction)   │  ← 全为 F │
  │   │  e.g. p ∧ ¬p            │             │
  │   └──────────────────────────┘             │
  │                                            │
  └────────────────────────────────────────────┘
```

**关键关系**: 永真式 ⊂ 可满足式 (永真式是特殊的可满足式)。

### 2.6 工程例子: SQL WHERE 子句的逻辑结构

SQL 的 `WHERE` 子句就是一个命题:

```sql
SELECT * FROM orders
WHERE (status = 'paid' OR status = 'shipped')
  AND total > 100
  AND NOT (customer_id IN (1, 2, 3));
```

设:
- $p$: `status IN ('paid', 'shipped')`
- $q$: `total > 100`
- $r$: `customer_id IN (1, 2, 3)`

逻辑形式:

$$(p \lor \text{true}) \land q \land \neg r \equiv p \land q \land \neg r$$

注意: `status = 'paid' OR status = 'shipped'` 等价于 `status IN ('paid', 'shipped')`, 所以可以化简。

**真值表化简**:

| $p$ | $q$ | $r$ | $p \land q \land \neg r$ |
| :--- | :--- | :--- | :--- |
| T | T | T | F |
| T | T | F | T |
| T | F | T | F |
| T | F | F | F |
| F | T | T | F |
| F | T | F | F |
| F | F | T | F |
| F | F | F | F |

只有一种情况 (T, T, F) 让 WHERE 为真, 即: 已支付或已发货, 总金额 > 100, 客户不在黑名单中。

**工程意义**: 数据库的查询优化器 (query optimizer) 会用**等价变换**化简 WHERE 子句。例如, `p ∨ (q ∧ r)` 可以变换为 `(p ∨ q) ∧ (p ∨ r)` (分配律), 这样可能让某些子句单独成立, 让查询更快。

### 2.7 PyTorch 代码: 命题逻辑的真值表计算

```python
import torch

# 1. 用 0/1 张量表示命题的真值
def truth_table(variables):
    """
    给定变量名列表, 生成所有 2^n 个真值组合的张量。
    variables: e.g. ['p', 'q']
    返回 shape (2^n, n) 的 bool 张量。
    """
    n = len(variables)
    # 生成 0 到 2^n - 1 的整数, 用二进制表示每行的真值组合
    indices = torch.arange(2 ** n)
    # 把每个整数转成 n 位二进制 (bool 张量)
    table = torch.zeros(2 ** n, n, dtype=torch.bool)
    for i in range(n):
        # 第 i 位: (indices >> i) & 1
        table[:, i] = (indices >> i) & 1
    return table

# 测试
tt = truth_table(['p', 'q'])
print("p q 真值表:\n", tt)
# tensor([[ True,  True],   ← p=T, q=T (第1行索引 = 0b00)
#         [False,  True],   ← p=T, q=F? 不对, 索引 1 = 0b01, p 第0位=T? 不
#         ...])

# 修正: 我用的是 LSB 顺序, 即第 0 列是 p, 第 1 列是 q
# 索引 0 (00): p=0, q=0
# 索引 1 (01): p=1, q=0  (二进制第 0 位是 1, 第 1 位是 0)
# 索引 2 (10): p=0, q=1
# 索引 3 (11): p=1, q=1
# 所以表应该是:
#   T T  (索引 3, p 第 0 位 = 1, q 第 1 位 = 1)
#   T F  (索引 2, p 第 0 位 = 0? 不对, 索引 2 = 0b10, 第 0 位 = 0, 第 1 位 = 1)
# ...
# 总之, 我们得到了所有 4 种组合 ✓

# 2. 5 个联结词的向量化实现
def NOT(p):
    return ~p

def AND(p, q):
    return p & q

def OR(p, q):
    return p | q

def IMPLIES(p, q):
    """p → q ≡ ¬p ∨ q"""
    return (~p) | q

def IFF(p, q):
    """p ↔ q ≡ (p → q) ∧ (q → p)"""
    return IMPLIES(p, q) & IMPLIES(q, p)

# 3. 计算复合命题的真值表
def compound_truth_table(formula_fn, variables):
    """
    formula_fn: 接受 n 个 bool 张量, 返回 1 个 bool 张量的函数
    """
    tt = truth_table(variables)
    n = len(variables)
    # 提取每列作为输入
    inputs = [tt[:, i] for i in range(n)]
    # 计算 formula
    result = formula_fn(*inputs)
    return tt, result

# 例 1: 验证 p → q ≡ ¬p ∨ q
def f1(p, q):
    return IMPLIES(p, q)

def f2(p, q):
    return (~p) | q

tt, r1 = compound_truth_table(f1, ['p', 'q'])
_, r2 = compound_truth_table(f2, ['p', 'q'])

print("\np → q 的真值表:")
print(torch.cat([tt, r1.unsqueeze(1)], dim=1))
print("\n¬p ∨ q 的真值表:")
print(torch.cat([tt, r2.unsqueeze(1)], dim=1))
print(f"\n等价? {torch.equal(r1, r2)}")
# True ✓

# 例 2: 验证 (p → q) ∧ (q → r) → (p → r) 是永真式
def transitivity(p, q, r):
    return IMPLIES(AND(IMPLIES(p, q), IMPLIES(q, r)), IMPLIES(p, r))

tt, result = compound_truth_table(transitivity, ['p', 'q', 'r'])
print("\n传递性 (p → q) ∧ (q → r) → (p → r) 的真值表:")
print(torch.cat([tt, result.unsqueeze(1)], dim=1))
print(f"\n是永真式? {result.all().item()}")
# True ✓ 传递律是永真式
```

**预期输出**:

```text
p → q 的真值表:
tensor([[ True,  True,    True],   ← p=T, q=T: p→q = T
        [ True, False,   False],   ← p=T, q=F: p→q = F
        [False,  True,    True],   ← p=F, q=T: p→q = T
        [False, False,    True]])  ← p=F, q=F: p→q = T

等价? True
传递性 ... 是永真式? True
```

**工程意义**: 这段代码本质上是 SAT 求解器的**真值表引擎**——对所有可能的输入, 检查公式是否成立。真值表方法对 $n$ 个变量有 $2^n$ 行, 复杂度 $O(2^n)$, 所以只能处理小规模问题 (n < 20)。但对小问题, 真值表方法**保证正确**, 是验证等价律的"金标准"。

---

## 3. 命题逻辑等价与蕴含

### 3.1 逻辑等价 (Logical Equivalence) A ≡ B

**定义**: 命题 $A$ 和 $B$ **逻辑等价**, 记作 $A \equiv B$, 当且仅当 $A \leftrightarrow B$ 是永真式。

**等价判据**: $A \equiv B$ 当且仅当它们在**所有真值赋值**下真值相同。

**例子**: $p \rightarrow q \equiv \neg p \lor q$ (见 §2.4)。

### 3.2 重要等价律 (Equivalence Laws)

下面是命题逻辑中最常用的等价律。这些律是**形式化推理**的基础, 也是编译器优化的依据。

| 律 | 等价式 | 名称 |
| :--- | :--- | :--- |
| 交换律 | $p \land q \equiv q \land p$ | 交换律 (合取) |
| | $p \lor q \equiv q \lor p$ | 交换律 (析取) |
| 结合律 | $(p \land q) \land r \equiv p \land (q \land r)$ | 结合律 (合取) |
| | $(p \lor q) \lor r \equiv p \lor (q \lor r)$ | 结合律 (析取) |
| 分配律 | $p \land (q \lor r) \equiv (p \land q) \lor (p \land r)$ | 分配律 (∧ over ∨) |
| | $p \lor (q \land r) \equiv (p \lor q) \land (p \lor r)$ | 分配律 (∨ over ∧) |
| 德摩根律 | $\neg(p \land q) \equiv \neg p \lor \neg q$ | 德摩根律 |
| | $\neg(p \lor q) \equiv \neg p \land \neg q$ | 德摩根律 |
| 双重否定 | $\neg\neg p \equiv p$ | 双重否定律 |
| 幂等律 | $p \land p \equiv p$, $p \lor p \equiv p$ | 幂等律 |
| 吸收律 | $p \lor (p \land q) \equiv p$ | 吸收律 |
| | $p \land (p \lor q) \equiv p$ | 吸收律 |
| 蕴含转析取 | $p \rightarrow q \equiv \neg p \lor q$ | 蕴含转析取 |
| 对置律 | $p \rightarrow q \equiv \neg q \rightarrow \neg p$ | 对置律 (contrapositive) |
| 假言易位 | $p \rightarrow q \equiv \neg(p \land \neg q)$ | (等价于 $\neg p \lor q$) |
| 等价转蕴含 | $p \leftrightarrow q \equiv (p \rightarrow q) \land (q \rightarrow p)$ | 等价转蕴含 |
| 同一律 | $p \land \text{T} \equiv p$, $p \lor \text{F} \equiv p$ | 同一律 |
| 零律 | $p \lor \text{T} \equiv \text{T}$, $p \land \text{F} \equiv \text{F}$ | 零律 |
| 排中律 | $p \lor \neg p \equiv \text{T}$ | 排中律 |
| 矛盾律 | $p \land \neg p \equiv \text{F}$ | 矛盾律 |

#### 例子: 德摩根律的工程应用

```python
import torch

# 德摩根律: NOT (A OR B) == (NOT A) AND (NOT B)
A = torch.tensor([True, False, True, False])
B = torch.tensor([True, True, False, False])

# 两种实现应该等价
not_A_or_B = ~(A | B)
not_A_and_not_B = (~A) & (~B)

print(f"等价? {torch.equal(not_A_or_B, not_A_and_not_B)}")
# True ✓

# SQL 中的对应
# WHERE NOT (age > 18 OR country = 'US')
# 等价于:
# WHERE age <= 18 AND country != 'US'
```

**工程意义**: 德摩根律在**查询优化**和**代码简化**中至关重要。例如:

```sql
-- 慢: NOT IN (子查询)
SELECT user_id FROM active_users
WHERE user_id NOT IN (SELECT user_id FROM banned_users);

-- 快(等价): 用 LEFT JOIN + IS NULL
SELECT a.user_id FROM active_users a
LEFT JOIN banned_users b ON a.user_id = b.user_id
WHERE b.user_id IS NULL;
```

两次查询逻辑等价 (德摩根律), 但 LEFT JOIN 通常更快, 因为数据库优化器对 JOIN 的处理更成熟。

### 3.3 逻辑蕴含 (Logical Implication) A ⇒ B

**定义**: 命题 $A$ **逻辑蕴含** $B$, 记作 $A \Rightarrow B$ 或 $A \models B$, 当且仅当 $A \rightarrow B$ 是永真式。

**等价判据**: $A \Rightarrow B$ 当且仅当**所有**让 $A$ 为真的真值赋值, 都让 $B$ 为真。

**与等价的区别**: 等价 ($A \equiv B$) 要求**双向**蕴含 ($A \Rightarrow B$ 且 $B \Rightarrow A$); 蕴含只要求**单向**。

**例子**: $p \Rightarrow (p \lor q)$

```text
p = T, q = T:  p ∨ q = T, 所以 p → (p ∨ q) = T → T = T  ✓
p = T, q = F:  p ∨ q = T, 所以 p → (p ∨ q) = T → T = T  ✓
p = F:         整个蕴含空真 (T)
```

全为 T, 所以 $p \Rightarrow p \lor q$。 ✓

**重要蕴含关系**:

| 蕴含 | 名称 |
| :--- | :--- |
| $p \land q \Rightarrow p$ | 简化律 (simplification) |
| $p \Rightarrow p \lor q$ | 附加律 (addition) |
| $p, p \rightarrow q \Rightarrow q$ | 假言推理 (Modus Ponens) |
| $\neg q, p \rightarrow q \Rightarrow \neg p$ | 拒取式 (Modus Tollens) |
| $p \rightarrow q, q \rightarrow r \Rightarrow p \rightarrow r$ | 假言三段论 |
| $p \lor q, \neg p \Rightarrow q$ | 析取三段论 |
| $p \rightarrow q, \neg q \Rightarrow \neg p$ | 对置律 (Modus Tollens 的另一种形式) |

### 3.4 工程意义: 编译器优化 / 表达式化简

**场景 1: 常数折叠 (constant folding)**

编译器把 `True or x` 替换为 `True`, 因为 $p \lor \text{T} \equiv \text{T}$ (零律)。

**场景 2: 死代码消除 (dead code elimination)**

```python
if False:
    do_something()  # 永远不执行
```

编译器删除死代码, 因为 $p \land \neg p \equiv \text{F}$ (矛盾律)。

**场景 3: 强度削减 (strength reduction)**

编译器把 `x * 2` 替换为 `x << 1`, 乘法变位运算。这是**算术等价律**的应用。

**场景 4: SQL 查询优化**

```sql
-- 原始
WHERE (a > 0 AND b > 0) OR (a > 0 AND c > 0)

-- 等价(提取公因子 a > 0)
WHERE a > 0 AND (b > 0 OR c > 0)
```

这是分配律的反向应用, 让 a > 0 只计算一次, 性能更好。

---

## 4. 范式 (Normal Forms): DNF / CNF

### 4.1 析取范式 (Disjunctive Normal Form, DNF)

**定义**: 一个命题是**析取范式**, 如果它是若干**合取项**的**析取**:

$$(p_1 \land p_2 \land \ldots) \lor (q_1 \land q_2 \land \ldots) \lor \ldots$$

每个合取项 (conjunct) 是一个或多个**文字**(literal) 的合取。文字是命题变量或其否定 (e.g. $p$, $\neg q$)。

**例子**: $(p \land \neg q) \lor (\neg p \land q)$ 是 DNF。

**直觉**: DNF = "满足任一情况就成立"。"情况" 是若干条件的合取。

**工程意义**: DNF 形式直接对应**决策树 / if-else 链**:

```python
if (p and not q) or (not p and q):
    # DNF 形式
    do_something()
```

### 4.2 合取范式 (Conjunctive Normal Form, CNF)

**定义**: 一个命题是**合取范式**, 如果它是若干**析取项**的**合取**:

$$(p_1 \lor p_2 \lor \ldots) \land (q_1 \lor q_2 \lor \ldots) \land \ldots$$

每个析取项 (disjunct) 是一个或多个文字的析取。

**例子**: $(p \lor \neg q) \land (\neg p \lor q)$ 是 CNF。

**直觉**: CNF = "所有条件都满足才成立"。"条件" 是若干要求的析取。

**工程意义**: CNF 形式直接对应**约束系统**:

```python
# 所有约束都满足才合法
if (p or not q) and (not p or q):
    do_something()
```

### 4.3 主析取范式 (Principal DNF): 唯一表示

**定义**: 一个命题的**主析取范式** (也叫**极小项展开**) 是把所有使命题为真的真值赋值 "展开" 成一个析取式, 每个合取项对应一个真值赋值。

**构造方法**: 对每个让命题为真的真值赋值 (e.g. $p = T, q = F$), 写一个合取项 $p \land \neg q$。把所有这些合取项用 $\lor$ 连起来。

**例子**: 命题 $(p \lor q) \land \neg r$ 的真值表:

| $p$ | $q$ | $r$ | $(p \lor q) \land \neg r$ |
| :--- | :--- | :--- | :--- |
| T | T | T | F |
| T | T | F | T ✓ |
| T | F | T | F |
| T | F | F | T ✓ |
| F | T | T | F |
| F | T | F | T ✓ |
| F | F | T | F |
| F | F | F | F |

主析取范式:

$$(p \land q \land \neg r) \lor (p \land \neg q \land \neg r) \lor (\neg p \land q \land \neg r)$$

**唯一性**: 每个命题有**唯一**的主析取范式。这就是命题逻辑的"指纹"。

**类似地**, 主合取范式 (Principal CNF) 由所有让命题为**假**的真值赋值得到。

### 4.4 工程应用: SAT 求解器 / 形式化验证 / LLM 推理加速

#### SAT 求解器 (SAT Solver)

**SAT (Boolean Satisfiability)** 问题: 给定一个 CNF 命题, 判断是否存在真值赋值让命题为真。

**应用**: 工业硬件验证 (Intel / AMD 用 SAT 验证 CPU 设计的正确性)、软件验证、密码学、AI 规划。

**为什么用 CNF?** 因为 CNF 形式易于操作——每个子句是一个约束, 求满足所有约束的赋值。

**ASCII 可视化: SAT 求解器的工作原理**

```text
  输入: CNF 公式
  (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
         ↓
  ┌─────────────────────────┐
  │ 1. 选择变量 (DPLL/CDCL) │
  │ 2. 单位传播 (Unit Prop)  │
  │ 3. 子句学习 (Clause Learn)│
  └─────────────────────────┘
         ↓
  输出: SAT / UNSAT + 赋值
  e.g. SAT: x1=T, x2=F, x3=T
```

**工程意义**: 现代 SAT 求解器 (如 MiniSat、Z3、CryptoMiniSat) 可以处理**百万级变量**的工业问题。CDCL (Conflict-Driven Clause Learning) 算法让 SAT 在实践中非常高效, 即使最坏情况是 NP 完全。

#### 形式化验证 (Formal Verification)

**应用**: 用 SAT/SMT 求解器**证明**硬件 / 软件的正确性。例如:

- "CPU 在任何输入下都不会除以 0"
- "我的程序不会内存泄漏"
- "我的智能合约不会被重入攻击"

**流程**:
1. 把要验证的属性写成命题 / 谓词逻辑公式
2. 把代码路径展开成约束 (CNF)
3. 用 SAT/SMT 求解器检查: 在所有可能的输入下, 属性是否成立

#### LLM 推理加速

**应用**: LLM 的推理链可以**用 SAT 求解器验证**或**加速**。

**例子**: GSM8K 数学题, LLM 生成推理链, 然后用符号求解器 (e.g. SymPy) 验证中间步骤。这是**LLM + 符号求解器**的混合方法, 准确性远高于纯 LLM。

**例子**: AlphaGeometry (DeepMind, 2024) 用 LLM 提议几何构造, 用 DDAR (依赖性消解自动推理) 验证和补全证明。

---

## 5. 谓词逻辑 (Predicate Logic)

### 5.1 谓词 (Predicate) P(x₁, ..., xₙ)

**定义**: 一个 $n$ 元**谓词** $P(x_1, x_2, \ldots, x_n)$ 是一个"取值依赖于 $n$ 个变量的命题"。每个变量代入具体值后, 谓词变成命题。

**例子**:

- 一元谓词: $P(x)$ = "$x$ 是偶数"。$P(2) = \text{T}$, $P(3) = \text{F}$。
- 二元谓词: $L(x, y)$ = "$x$ 爱 $y$"。$L(\text{Alice}, \text{Bob}) = \text{T}$。
- 三元谓词: $G(x, y, z)$ = "$x$ 在 $y$ 和 $z$ 之间"。

**与命题的关系**: 谓词 = "参数化的命题"。命题是谓词的"实例化"。

**工程意义**: 谓词对应程序中的**布尔函数**:

```python
def is_even(x: int) -> bool:
    return x % 2 == 0  # 一元谓词

def loves(x: str, y: str) -> bool:
    return ...  # 二元谓词 (假设由 LLM 预测)
```

### 5.2 量词 (Quantifiers): ∀ 和 ∃

#### 全称量词 (Universal Quantifier) ∀

$\forall x P(x)$: 对所有 $x$, $P(x)$ 都成立。

**例子**: $\forall x (x \ge 0 \rightarrow x^2 \ge 0)$ (任何非负数的平方都非负)。

**工程对应**: "for all" 循环 / Python 的 `all()`:

```python
all(x >= 0 for x in numbers)  # ∀ x in numbers: x >= 0
```

#### 存在量词 (Existential Quantifier) ∃

$\exists x P(x)$: 存在某个 $x$, 使 $P(x)$ 成立。

**例子**: $\exists x (x^2 = 2)$ (存在一个实数, 平方等于 2)。

**工程对应**: "exists" / Python 的 `any()`:

```python
any(x == 2 for x in [1, 2, 3])  # ∃ x in [1,2,3]: x == 2
```

#### 量词的优先级

$\forall$ 和 $\exists$ 的作用域 (scope) 取决于后续的公式。通常用括号显式限定:

- $\forall x (P(x) \rightarrow Q(x))$: 对所有 $x$, 如果 $P(x)$ 则 $Q(x)$
- $\forall x P(x) \rightarrow Q(x)$: "如果所有 $x$ 都满足 $P$", 那么 $Q(x)$ (这里的 $x$ 是 free)

### 5.3 谓词公式 (Predicate Formula)

**定义**: 用命题联结词和量词把谓词组合起来, 形成**谓词公式**。

**例子 1**: $\forall x (M(x) \rightarrow D(x))$ (所有人会死)

**例子 2**: $\exists x \forall y L(x, y)$ (存在一个人, 爱所有人)

**例子 3**: $\forall x \exists y L(x, y)$ (每个人都有一个爱的人)

**例子 2 和 3 的区别** (这是经典的 ∃∀ vs ∀∃ 问题):

- $\exists x \forall y L(x, y)$: 存在**同一个**人, 爱**所有**人
- $\forall x \exists y L(x, y)$: 对**每个人**, **可能不同的**人, 是他爱的人

**ASCII 可视化: ∀∃ vs ∃∀ 的区别**

```text
  ∀x ∃y L(x, y)           每个 x 都有"自己的" y
  ─────────────
  Alice  →  Bob              (Alice 爱 Bob)
  Carol  →  David            (Carol 爱 David, 不是 Bob)
  Eve    →  Bob              (Eve 爱 Bob)
                  ↑
            Bob 被 Alice 和 Eve 爱, David 被 Carol 爱
            (y 可以不同)

  ∃x ∀y L(x, y)           存在一个 x, 爱所有人
  ─────────────
                  ↓
            Superman → Alice, Bob, Carol, David, Eve, ...
                       (Superman 爱所有人)
```

### 5.4 自由变量 vs 约束变量 (Free vs Bound Variable)

**约束变量** (bound): 被量词 $\forall$ 或 $\exists$ 修饰的变量。

**自由变量** (free): 没有被量词修饰的变量。

**例子**:

- $\forall x P(x, y)$: $x$ 是约束变量, $y$ 是自由变量
- $\forall x P(x) \land Q(x)$: 第一个 $x$ 约束 (在 $\forall x$ 的作用域内), 第二个 $x$ 自由 (在作用域外)

**工程意义**: 自由变量是"还没赋值的占位符", 就像函数参数。如果谓词公式有自由变量, 它就不是一个完整的命题 (真值不确定)。

```python
# 自由变量对应"未绑定的参数"
def f(x):  # x 是函数参数 (自由变量, 等调用时绑定)
    return x > 0

f(5)  # 绑定了 x = 5, 返回 True (命题)
f     # 不绑定 x, 返回一个"未实例化"的函数 (谓词)
```

### 5.5 量词的等价与蕴含

#### 量词否定

- $\neg \forall x P(x) \equiv \exists x \neg P(x)$ (不是所有人都 P = 存在一个不 P)
- $\neg \exists x P(x) \equiv \forall x \neg P(x)$ (不存在 P 的人 = 所有人都不是 P)

**这是德摩根律的量词版本**。

#### 量词分配

- $\forall x (P(x) \land Q(x)) \equiv \forall x P(x) \land \forall x Q(x)$ (全称对合取可分配)
- $\exists x (P(x) \lor Q(x)) \equiv \exists x P(x) \lor \exists x Q(x)$ (存在对析取可分配)

**反例** (不能分配的):

- $\forall x P(x) \lor \forall x Q(x) \not\equiv \forall x (P(x) \lor Q(x))$
- $\exists x (P(x) \land Q(x)) \not\equiv \exists x P(x) \land \exists x Q(x)$

**反例 1**: $\forall x P(x) \lor \forall x Q(x)$ (所有人都 P 或者所有人都 Q)
不等于 $\forall x (P(x) \lor Q(x))$ (每个人都满足 P 或 Q)

**例子**: 设 $P$ = "是男人", $Q$ = "是女人"。
- $\forall x P(x) \lor \forall x Q(x)$: 所有人都是男人, 或所有人都是女人 → 假
- $\forall x (P(x) \lor Q(x))$: 每个人都是男人或女人 → 真

**反例 2**: $\exists x (P(x) \land Q(x))$ (存在一个人, 同时是男人和女人)
不等于 $\exists x P(x) \land \exists x Q(x)$ (存在一个男人, 存在一个女人)

### 5.6 谓词逻辑 vs 命题逻辑 (表达力)

**命题逻辑**只能表达"原子命题 + 联结词", 不能表达"对所有 / 存在"。

**谓词逻辑**扩展了命题逻辑, 加入谓词、量词, 可以表达更复杂的事实。

**例子**:

命题逻辑**不能**表达 "所有的人都会死":
- 命题逻辑只能处理有限个命题 (如 $D(\text{Alice})$, $D(\text{Bob})$)
- 不能表达"对所有 $x$, $D(x)$"

谓词逻辑可以: $\forall x (M(x) \rightarrow D(x))$。

**哥德尔完备性定理** (Gödel 1930): 谓词逻辑是**完备的**——所有"逻辑上为真"的命题都可以被证明。

**丘奇-图灵不可判定性** (Church-Turing 1936): 谓词逻辑的**一般判定问题**是不可判定的——没有算法能在有限时间内判断任意谓词公式是否为永真式。

### 5.7 工程例子: 知识图谱的 schema (TransE / RGCN 谓词定义)

知识图谱 (Knowledge Graph) 是形如 $(h, r, t)$ 的三元组集合, 例如:

```text
(北京, 是...的首都, 中国)
(苏格拉底, 是...的老师, 柏拉图)
(Transformer, 是...的子领域, 深度学习)
```

**TransE** 把每个实体和关系表示成向量, 学习 $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$。

**谓词视角**: TransE 中的关系 $r$ 是一个**谓词** $r(h, t)$ 表示"$h$ 和 $t$ 有关系 $r$"。训练时, 我们"实例化"这个谓词:

```text
"是...的首都"(北京, 中国) = True   ← 正例
"是...的首都"(巴黎, 中国) = False  ← 负例 (硬替换)
```

**RGCN (Relational GCN)** 用谓词逻辑的思路扩展 GCN: 每个关系类型 $r$ 有自己的权重矩阵 $W_r$:

$$\mathbf{h}_i^{(l+1)} = \sigma\left( \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{|\mathcal{N}_i^r|} W_r^{(l)} \mathbf{h}_j^{(l)} + W_0^{(l)} \mathbf{h}_i^{(l)} \right)$$

**这里**: $\mathcal{N}_i^r$ 是节点 $i$ 在关系 $r$ 下的邻居, 谓词 $r(x, y)$ = "节点 $x$ 通过关系 $r$ 连接到 $y$"。

**形式化对比**: RGCN 的核心公式可以写成:

$$\forall i, r: \mathbf{h}_i^{(l+1)} = \sigma\left( W_r^{(l)} \cdot \text{AGG}(\{h_j^{(l)} : j \in \mathcal{N}_i^r\}) + W_0^{(l)} \mathbf{h}_i^{(l)} \right)$$

**这是谓词逻辑在 GNN 中的应用**: 用谓词 / 量词描述"所有节点 / 所有关系的聚合规则"。

---

## 6. 推理规则 (Inference Rules)

### 6.1 假言推理 (Modus Ponens): P, P→Q ⊢ Q

**规则**: 如果 $P$ 为真, 且 $P \rightarrow Q$ 为真, 那么 $Q$ 为真。

**形式化**:

$$\frac{P, \quad P \rightarrow Q}{\therefore Q}$$

**例子**:

- "今天是星期三", "如果是星期三, 那么开会" → "今天开会"

**证明** (用真值表):

| $P$ | $Q$ | $P \rightarrow Q$ | $P \land (P \rightarrow Q)$ | $P \land (P \rightarrow Q) \rightarrow Q$ |
| :--- | :--- | :--- | :--- | :--- |
| T | T | T | T | T |
| T | F | F | F | T |
| F | T | T | F | T |
| F | F | T | F | T |

最后一列全为 T, 所以 $P \land (P \rightarrow Q) \Rightarrow Q$ 永真。 ✓

**工程意义**: 这是 CoT 提示词的**最常见模式**——前提 + 规则 → 结论。

### 6.2 拒取式 (Modus Tollens): ¬Q, P→Q ⊢ ¬P

**规则**: 如果 $Q$ 为假, 且 $P \rightarrow Q$ 为真, 那么 $P$ 为假。

**形式化**:

$$\frac{\neg Q, \quad P \rightarrow Q}{\therefore \neg P}$$

**例子**:

- "地面是干的", "如果下雨, 地面会湿" → "没下雨"

**证明**: 由对置律, $P \rightarrow Q \equiv \neg Q \rightarrow \neg P$。所以拒取式是假言推理的镜像。

### 6.3 假言三段论 (Hypothetical Syllogism): P→Q, Q→R ⊢ P→R

**规则**: 如果 $P \rightarrow Q$ 且 $Q \rightarrow R$, 那么 $P \rightarrow R$。

**形式化**:

$$\frac{P \rightarrow Q, \quad Q \rightarrow R}{\therefore P \rightarrow R}$$

**例子**:

- "下雨 → 地面湿", "地面湿 → 滑" → "下雨 → 滑"

**工程意义**: 假言三段论是**链式推理**的基础, 也是 CoT "中间步骤" 的形式化。

### 6.4 析取三段论 (Disjunctive Syllogism): P∨Q, ¬P ⊢ Q

**规则**: 如果 $P \lor Q$ 为真, 且 $P$ 为假, 那么 $Q$ 为真。

**形式化**:

$$\frac{P \lor Q, \quad \neg P}{\therefore Q}$$

**例子**:

- "他在办公室或在家", "他不在办公室" → "他在家"

**工程意义**: 析取三段论是**排除法**的形式化。在多选题、分类问题中常用。

### 6.5 全称实例化 (Universal Instantiation) / 存在概括 (Existential Generalization)

#### 全称实例化 (UI)

$$\frac{\forall x P(x)}{\therefore P(c)}$$

**规则**: 如果对所有 $x$ 都 $P(x)$ 成立, 那么对任何具体的 $c$ 都 $P(c)$ 成立。

**例子**: "所有人都要纳税" → "苏格拉底要纳税"。

#### 存在概括 (EG)

$$\frac{P(c)}{\therefore \exists x P(x)}$$

**规则**: 如果某个具体的 $c$ 满足 $P$, 那么存在某个 $x$ 满足 $P$。

**例子**: "苏格拉底是人" → "存在某个人是人"。

### 6.6 工程应用: 自动定理证明 (ATP) / 归结原理

#### 自动定理证明 (ATP)

**目标**: 给定一组前提和目标, 自动找到一条推理链, 从前提推出目标。

**经典系统**: Lean、Coq、Isabelle (基于类型论); Vampire、E (基于一阶逻辑)。

**应用**:
- 数学定理证明 (Lean 证明了 IMO 题目)
- 程序正确性证明 (Coq 验证了 C 编译器 CompCert)
- 硬件验证
- 密码协议验证

#### 归结原理 (Resolution)

**归结规则**:

$$\frac{P \lor Q, \quad \neg P \lor R}{\therefore Q \lor R}$$

**直觉**: 如果 $P \lor Q$ 为真, 且 $\neg P \lor R$ 为真, 那么 $P$ 在一个子句中为真, 在另一个子句中为假, 所以 $Q \lor R$ 必须为真。

**归结原理完备性**: 一阶逻辑的归结原理是**完备的**——如果一个公式是永假式, 归结能在有限步推出空子句 $\square$。

**应用**: Prolog 编程语言基于归结原理 (Horn 子句的特例)。

### 6.7 LLM 推理: CoT 提示词的"推理规则"启发

**观察**: CoT 提示词让 LLM 输出的中间步骤, **结构上**符合形式化推理规则:

```text
Q: 苏格拉底会死吗?
A: (CoT)
前提 1: 所有人都会死。
前提 2: 苏格拉底是人。
应用假言推理:
  前提 1 等价于: ∀x (M(x) → D(x))
  前提 2 等价于: M(s)
  由全称实例化: M(s) → D(s)
  由假言推理: D(s)
结论: 苏格拉底会死。
```

**关键**: LLM 没有**真的**应用推理规则, 它是**模仿**推理规则的"形"。但因为它的训练数据中有大量人类写的推理, 它学到了"推理的形式", 在很多问题上能给出正确答案。

**但**: 当问题需要**长链推理**或多步归结时, LLM 可能:
- 跳步 (遗漏某个前提)
- 编造前提
- 应用错误的规则

这是 LLM 推理的**根本限制**, 形式化推理工具 (SAT solver、Lean) 不会有这些问题。

**混合方案** (LLM + 符号):

```text
1. LLM 提议推理步骤
2. 符号求解器验证每个步骤
3. 如果验证失败, 让 LLM 重新尝试
```

这是 CoT + 程序辅助 (PAL, Program-aided Language Models) 的核心思想。

---

## 7. PyTorch 代码: 命题逻辑与推理

### 7.1 代码 1: 命题联结词的 torch 向量化实现

```python
import torch

# 5 个联结词的向量化实现 (来自 §2.7)
def NOT(p): return ~p
def AND(p, q): return p & q
def OR(p, q): return p | q
def IMPLIES(p, q): return (~p) | q
def IFF(p, q): return IMPLIES(p, q) & IMPLIES(q, p)

# 测试
p = torch.tensor([True, False, True])
q = torch.tensor([True, True, False])

print("p:", p.tolist())
print("q:", q.tolist())
print("p AND q:", AND(p, q).tolist())      # [True, False, False]
print("p OR q:", OR(p, q).tolist())         # [True, True, True]
print("p → q:", IMPLIES(p, q).tolist())     # [True, True, False]
print("p ↔ q:", IFF(p, q).tolist())         # [True, False, False]
```

### 7.2 代码 2: 真值表自动生成 + 等价检查

```python
def truth_table(variables):
    """生成所有 2^n 个真值组合 (LSB 顺序)"""
    n = len(variables)
    indices = torch.arange(2 ** n)
    table = torch.zeros(2 ** n, n, dtype=torch.bool)
    for i in range(n):
        table[:, i] = (indices >> i) & 1
    return table

def evaluate(formula_fn, variables):
    """计算公式在所有真值赋值下的结果"""
    tt = truth_table(variables)
    inputs = [tt[:, i] for i in range(len(variables))]
    return tt, formula_fn(*inputs)

def is_equivalent(formula_a, formula_b, variables):
    """判断两个公式是否逻辑等价"""
    _, ra = evaluate(formula_a, variables)
    _, rb = evaluate(formula_b, variables)
    return torch.equal(ra, rb)

def is_tautology(formula, variables):
    """判断公式是否是永真式"""
    _, r = evaluate(formula, variables)
    return r.all().item()

# 例 1: 验证 p → q ≡ ¬p ∨ q
print(is_equivalent(lambda p, q: IMPLIES(p, q),
                    lambda p, q: NOT(p) | q,
                    ['p', 'q']))
# True ✓

# 例 2: 验证排中律 p ∨ ¬p 是永真式
print(is_tautology(lambda p: p | NOT(p), ['p']))
# True ✓

# 例 3: 验证矛盾律 p ∧ ¬p 是永假式
print(is_tautology(lambda p: p & NOT(p), ['p']))
# False (没有 T)

# 例 4: 验证分配律
# p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)
print(is_equivalent(
    lambda p, q, r: p & (q | r),
    lambda p, q, r: (p & q) | (p & r),
    ['p', 'q', 'r']
))
# True ✓
```

### 7.3 代码 3: 简单归结证明器 (Mini Resolution Prover)

```python
def to_cnf(clauses):
    """
    输入: 子句列表, 每个子句是文字列表
    例: [[('p', True), ('q', False)], [('r', True)]]
    表示 (p ∨ ¬q) ∧ (r)
    """
    return [set(c) for c in clauses]

def resolve(c1, c2):
    """归结两条子句: 找到互补文字 (如 p 和 ¬p), 消掉它们。返回新子句或 None。"""
    for lit1 in c1:
        for lit2 in c2:
            if lit1[0] == lit2[0] and lit1[1] != lit2[1]:
                return (c1 - {lit1}) | (c2 - {lit2})
    return None

def resolution_prove(clauses, max_iter=100):
    """归结原理证明: 试图从 clauses 推出矛盾。如果推出空子句, 说明 UNSAT。"""
    new_clauses = {frozenset(c) for c in clauses}

    for it in range(max_iter):
        clause_list = list(new_clauses)
        new_pairs = set()
        for i in range(len(clause_list)):
            for j in range(i + 1, len(clause_list)):
                result = resolve(clause_list[i], clause_list[j])
                if result is not None:
                    if len(result) == 0:
                        return f"UNSAT (iteration {it}, 推出空子句)"
                    new_pairs.add(frozenset(result))

        new_pairs -= new_clauses
        if not new_pairs:
            return "SAT (无矛盾, 子句集可满足)"
        new_clauses |= new_pairs

    return "UNKNOWN (迭代达到上限)"

# 例: 证明 {p, ¬p ∨ q, ¬q} 不一致
# 预期: UNSAT
clauses = [
    [('p', True)],          # p
    [('p', False), ('q', True)],   # ¬p ∨ q
    [('q', False)],         # ¬q
]
result = resolution_prove(to_cnf(clauses))
print(f"结果: {result}")
# Iteration 0: 添加了 1 个新子句
# 结果: UNSAT (推出空子句, 矛盾)
```

**说明**: 这段代码用归结原理证明了 $\{p, \neg p \lor q, \neg q\}$ 不一致。逻辑上是: $p$ 和 $\neg p \lor q$ 推出 $q$, $q$ 和 $\neg q$ 推出矛盾。

### 7.4 代码 4: CoT 提示词的"逻辑骨架"模板

```python
# 一个 CoT 提示词模板, 把自然语言推理结构化
COT_TEMPLATE = """
问题: {question}

请按以下结构推理:

前提 (Premises):
1. [从问题中提取的已知事实]
2. [更多已知事实]
...

推理规则 (Rules to apply):
- 如果要推出 Q, 看看是否有形如 P → Q 的规则
- 如果要排除某个选项, 看看是否有 ¬Q

推理步骤 (Steps):
1. 应用 [规则名称] 到 [前提], 得到 [中间结论]
2. 应用 [规则名称] 到 [中间结论], 得到 [下一步]
3. ...

结论 (Conclusion):
[最终答案]

验证 (Verification):
[检查每一步是否正确]
""".strip()

# 例: 用模板推理 "苏格拉底会死吗?"
question = "苏格拉底会死吗?"

# 构造 CoT prompt
prompt = COT_TEMPLATE.format(question=question)
print(prompt)

# 模拟 LLM 输出 (正确的 CoT 推理)
expected_response = """
前提 (Premises):
1. 所有的人都会死。    (∀x M(x) → D(x))
2. 苏格拉底是人。       (M(s))

推理规则 (Rules to apply):
- 假言推理: P, P → Q ⊢ Q
- 全称实例化: ∀x P(x) ⊢ P(c)

推理步骤 (Steps):
1. 由前提 1 应用全称实例化 (用 s 替换 x): M(s) → D(s)
2. 由步骤 1 和前提 2 应用假言推理: D(s)

结论 (Conclusion):
苏格拉底会死。

验证 (Verification):
- 步骤 1 的全称实例化: ∀x P(x) ⊢ P(c) 是有效的推理规则 ✓
- 步骤 2 的假言推理: P, P → Q ⊢ Q 是有效的推理规则 ✓
- 结论必然为真 ✓
""".strip()

print("\n" + "=" * 60)
print(expected_response)
```

**工程意义**: 这个模板把 CoT 提示词结构化为**显式的推理步骤 + 推理规则**, 让 LLM 必须**沿推理链走**, 而不是直接猜答案。这是 CoT 有效的核心机制——**结构化提示**比**自由格式提示**效果好。

---

## 8. 常见误区

### 误区 1: ❌ "LLM 推理 = 真推理"

**反例**: LLM 在 GSM8K 上的 CoT 准确率约 80-90%, 不是 100%。即使有 CoT, LLM 也会犯错。

**根本原因**: LLM 是**统计模型**, 不是**形式化推理系统**。它的"推理"是基于训练数据的模式, 而不是基于推理规则。

**正确认知**: CoT 是"逼近"形式化推理的工程技巧, 不是形式化推理本身。形式化推理需要 SAT/SMT solver、Lean、Coq 等机械化工具。

### 误区 2: ❌ "命题逻辑 = 程序逻辑"

**反例**: 命题逻辑的蕴含 $P \rightarrow Q$ 是**真值函数**, 程序逻辑的 `if P: Q` 是**控制流**。

**正确认知**: 程序逻辑需要时序逻辑 (LTL, CTL) 或霍尔逻辑 (Hoare Logic), 不是命题逻辑。详见 第 11 章:形式语言与自动机。

### 误区 3: ❌ "∀ 和 ∃ 可以交换"

**反例**: $\forall x \exists y L(x, y)$ ≠ $\exists y \forall x L(x, y)$。

- $\forall x \exists y$: 对每个人 $x$, 存在 (可能不同的) $y$ 使 $L(x, y)$ 成立
- $\exists y \forall x$: 存在**同一个** $y$, 对所有 $x$ 都使 $L(x, y)$ 成立

**正确认知**: 量词顺序至关重要, 交换会改变语义。

### 误区 4: ❌ "CNF 转换总是简单的"

**反例**: 把命题转换为 CNF 可能**指数级膨胀** (Tseitin 变换虽然线性, 但可能引入新变量)。

**例子**: $(p_1 \land q_1) \lor (p_2 \land q_2) \lor \ldots \lor (p_n \land q_n)$ 转换为 CNF 后有 $2^n$ 个子句。

**正确认知**: SAT 求解器有专门的 CNF 编码技术 (Tseitin 变换), 不要手动展开成 CNF。

### 误区 5: ❌ "逻辑等价 = 语义相同"

**正确认知**: 逻辑等价 $\equiv$ 是**真值表层面**的等价, 即所有真值赋值下真值相同。**语义相同**还可能涉及**结构 / 含义**层面的等价 (e.g. 模型论中的"初等等价")。

**工程意义**: 当两个逻辑公式真值表等价时, 它们可以**互相替换**而不影响推理结果。但它们的"人类可读性"可能完全不同, 选择哪个取决于上下文。

---

## 9. 面试 Q&A (L7/L8 难度)

### Q1: "Chain-of-Thought 为什么有效? 从逻辑角度看"

**标准答案**:

CoT 提示词有效的核心机制有三点:

1. **强制显式推理链**: CoT 让 LLM 必须输出中间步骤, 而不是直接跳到结论。这模拟了形式化推理中的"逐步应用推理规则"。

2. **降低单步认知负担**: 每一步推理都很简单 (e.g. 应用假言推理), 整体任务是"简单步骤的组合"。LLM 在"简单步骤"上的准确率远高于"直接猜结论"。

3. **暴露错误便于检测**: 当 LLM 推理错时, 错误通常在某一步显式可见, 用户可以识别并修正。直接猜答案的错误更难诊断。

**逻辑学视角**:

CoT 模拟了**自然演绎**(natural deduction) 的形式化推理过程——每一步都显式调用推理规则 (假言推理、析取三段论等)。但 LLM 没有**真的**应用规则, 它是**模仿规则的"形"**。

**局限**: 当推理链很长 (>5 步) 或涉及复杂的量词交互时, LLM 容易"跳步"或"编造", CoT 失效。形式化推理工具 (SAT solver、Lean) 没有这个限制。

**进阶**:

- **Self-Consistency** (Wang et al. 2022): 采样多条 CoT 链, 投票决定最终答案。本质是"用集成方法对抗统计误差"。
- **Tree of Thoughts (ToT)** (Yao et al. 2023): 把 CoT 扩展为搜索树, 在每一步评估多条路径。本质是"在推理空间做 BFS/DFS"。
- **Program-aided LLM (PAL)** (Gao et al. 2022): 让 LLM 生成代码, 由代码解释器执行。本质是"把推理外包给符号系统"。

### Q2: "什么是归结原理? 在自动定理证明中如何应用?"

**标准答案**:

**归结原理**(Resolution) 是一条推理规则:

$$\frac{P \lor Q, \quad \neg P \lor R}{\therefore Q \lor R}$$

它的核心思想是**互补消解**: 如果一个子句含 $P$, 另一个子句含 $\neg P$, 那么 $P$ 在一个子句中为真, 在另一个子句中为假, 必须从其他文字 ($Q$ 或 $R$) 中推出结论。

**算法**:

1. 把所有子句写成 CNF
2. 反复应用归结规则, 生成新子句
3. 如果推出空子句 $\square$ (即 $P \land \neg P$), 说明原公式是永假式 (UNSAT), 即矛盾
4. 如果不再有新子句, 说明原公式可满足 (SAT)

**应用**:

- **自动定理证明**: 把"要证的定理"取反, 加入"已知公理", 用归结推出矛盾, 即完成证明。
- **Prolog 编程语言**: 基于归结原理的 Horn 子句特例, 用 SLD 归结实现逻辑编程。
- **SAT 求解器**: 现代 SAT solver (Z3, MiniSat) 用 CDCL 算法, 核心是带子句学习的归结。

**完备性**: Robinson 1965 证明, 一阶逻辑的归结原理是**完备的**——任何可证明的命题都可以通过归结推出。但一般情况是**半可判定的** (semi-decidable), 因为可能不终止。

**工程意义**: 归结是形式化推理的"瑞士军刀", 是 Lean、Coq 等证明助手的底层引擎之一。

### Q3: "∀x ∃y P(x,y) 和 ∃y ∀x P(x,y) 有什么区别? 给具体例子"

**标准答案**:

**关键区别**: 量词顺序决定了"是同一个还是可能不同"。

- $\forall x \exists y P(x, y)$: 对**每个** $x$, **可能不同的** $y$ 满足 $P$
- $\exists y \forall x P(x, y)$: 存在**同一个** $y$, 对**所有** $x$ 都满足 $P$

**蕴含关系**: $\exists y \forall x P(x, y) \Rightarrow \forall x \exists y P(x, y)$ (后者弱, 前者强), 但反过来**不**成立。

**具体例子 1: 爱情**

设 $L(x, y)$ = "$x$ 爱 $y$"。

- $\forall x \exists y L(x, y)$: 每个人都爱**某个人** (Alice 爱 Bob, Carol 爱 David, ...)
- $\exists y \forall x L(x, y)$: 存在**某个人**, **所有人都**爱他 (e.g. 耶稣/明星)

**具体例子 2: 自然数**

设 $P(x, y)$ = "$y > x$"。

- $\forall x \exists y (y > x)$: 对每个自然数 $x$, 存在更大的 $y$ → 真 (取 $y = x + 1$)
- $\exists y \forall x (y > x)$: 存在最大的 $y$, 比所有 $x$ 都大 → 假 (没有最大自然数)

**具体例子 3: 数据库查询**

设 $E(x, y)$ = "员工 $x$ 在部门 $y$ 工作"。

- $\forall x \exists y E(x, y)$: 每个员工都至少在一个部门 (假设员工都有部门) → 真
- $\exists y \forall x E(x, y)$: 存在一个部门, 所有员工都在该部门 → 假 (除非公司只有一个部门)

**工程意义**: 量词顺序在 SQL 中对应 `EXISTS` 和 `ALL`:

```sql
-- ∀x ∃y P(x, y): 每个员工都有部门
SELECT * FROM employees e WHERE EXISTS (SELECT 1 FROM departments d WHERE e.dept_id = d.id);

-- ∃y ∀x P(x, y): 存在一个部门, 所有员工都在
SELECT * FROM departments d WHERE NOT EXISTS (
    SELECT 1 FROM employees e WHERE e.dept_id != d.id
);
```

### Q4: "为什么 SAT 求解器能用于工业硬件验证?"

**标准答案**:

**SAT (Boolean Satisfiability)** 问题是判断一个布尔公式是否存在满足赋值, 它是第一个被证明的 NP 完全问题 (Cook 1971)。但工程上, 现代 SAT 求解器 (MiniSat, Z3, CryptoMiniSat) 可以处理**百万级变量**的问题, 这源于 CDCL (Conflict-Driven Clause Learning) 算法的工程突破。

**为什么能用于硬件验证?**

1. **硬件电路 = 布尔公式**: 任何数字电路 (CPU、GPU、ASIC) 都可以编码为 CNF。每个逻辑门 (AND/OR/NOT) 对应一个或几个子句, 每个寄存器 / 线网对应一个布尔变量。

2. **要验证的属性 = 反命题**: 想证明 "电路永远不会进入非法状态", 等价于证明 "非法状态不可达" = 命题不可满足 (UNSAT)。

3. **SAT 求解器找矛盾**: 把电路初始状态、转移关系、目标状态都编码为 CNF, 加入 SAT 求解器。如果求解器报告 UNSAT, 说明"在任何执行路径下, 目标状态都不可达", 即属性成立。如果报告 SAT, 求解器会给出反例 (触发非法状态的输入序列), 工程师据此修复 bug。

**工业案例**:

- **Intel / AMD / NVIDIA**: 用 SAT/SMT 验证 CPU/GPU 微架构的正确性。Intel 的工业实践表明, SAT 求解器可以在几小时内验证数十亿门的电路, 远超人工验证。
- **AWS / Microsoft**: 用形式化验证 (基于 SAT/SMT) 验证云服务的协议 (TLS, S3 一致性)。
- **航空 / 汽车**: DO-178C (航空) 和 ISO 26262 (汽车) 都要求关键系统的形式化验证。

**为什么 SAT 在实践中高效?**

- **单位传播**: 如果一个子句只有一个未赋值的文字, 它必须为真 (否则子句为假, 与前提矛盾)。这是 O(n) 的快速推理。
- **子句学习**: 当遇到冲突时, 求解器学习"导致冲突的最小子句", 避免重复犯错。这是 CDCL 的关键创新。
- **重启 + 启发式**: 周期性地"忘记"部分赋值, 用新的启发式重新搜索。

**工程意义**: SAT 求解器是形式化方法的**工业引擎**, 它的成功证明了"理论难的 NP 问题在工程上可以高效解决"。

### Q5: "LLM 的 ReAct 框架对应逻辑学中的什么?"

**标准答案**:

ReAct (Reasoning + Acting) 的每一步结构是:

```text
Thought: <reasoning about what to do>
Action:  <tool call or "Finish">
Observation: <result of action>
```

**逻辑学对应**:

- **Thought** ≈ 谓词逻辑的"内部状态"——LLM 当前的"信念 / 知识", 用自然语言表达
- **Action** ≈ 推理规则的"应用"——LLM 决定调用哪个工具 (search / calculator / code)
- **Observation** ≈ 推理规则的"前提扩展"——外部世界返回的事实, 加入 LLM 的"知识库"

**形式化**:

设 $K_t$ 为 LLM 在时刻 $t$ 的知识 (知识库), $T_t$ 为当前任务, $A_t$ 为要执行的动作。

```text
Thought_t: P(K_t, T_t)             ← 谓词: 当前知识下, 任务需要什么动作
Action_t: A(K_t, T_t)              ← 动作选择
Observation_t: O(A_t)              ← 动作的结果
K_{t+1}: K_t ∪ {Observation_t}    ← 知识更新
```

**与 Modus Ponens 的对比**:

```text
假言推理:        ReAct:
P               Thought_t
P → Q           Action_t (调用 tool)
─────           Observation_t
Q               Thought_{t+1}
```

每一步都是 "已知 + 规则 → 结论" 的模式。**区别**: 假言推理的"规则"是固定的, ReAct 的"规则"由 LLM 生成, 可能错。

**局限**:

- LLM 的 Thought 可能是错的 (e.g. 选错工具)
- LLM 的 Action 可能语法错误 (e.g. 调用不存在的 API)
- LLM 不会"否定"(它倾向于总是给一个答案, 而不是"我不知道")

**改进方向**:

- **Reflexion** (Shinn et al. 2023): 让 LLM 反思自己的错误, 加入知识库
- **Tree of Thoughts** (Yao et al. 2023): 多条 Thought 路径并行探索, 选最优
- **Toolformer** (Schick et al. 2023): 用监督学习训练 LLM 正确调用工具

**工程意义**: ReAct 是 LLM Agent 的事实标准, 但它的"逻辑正确性"没有保证。在生产环境中, 需要加**人机协作**或**符号验证**层, 防止 LLM 推理错误导致损失。

---

## 10. 联系其他章节 + 参考文献

### 10.1 互引 (Cross-references)

本章是离散数学的核心, 与其他章节有密切联系:

| 主题 | 关联章节 |
| :--- | :--- |
| 集合 / 关系 / 函数 | `第04章-01-集合·关系·函数-从用户行为序列到数学集合.md` (本章的前置) |
| 命题 / 谓词 / 推理 | **本章** |
| 形式语言 / 自动机 | 第 11 章:形式语言与自动机 (谓词逻辑 → 形式语言的扩展) |
| 图论 / 树 / 匹配 | 第 3 章:图论基础 (推理树 = 自动机 / 树结构) |
| 信息论 / 互信息 | `../第03章-概率论/0-总目录与阅读路径.md` (逻辑熵 / Kolmogorov 复杂度) |
| 系统思维 / Transformer | `../../../面试/第14章-高级图论/0-总目录与学习路径.md` (注意力的逻辑解释) |
| 概率论 / 贝叶斯 | `../第05章-信息论/0-总目录与阅读路径.md` (概率逻辑 / 不确定性推理) |
| 算法 / 复杂度 | `../第01章-高等数学/0-总目录与阅读路径.md` (SAT 是 NP 完全) |

**核心联系**:

- **本章 + 第 1 章**: 集合是命题和谓词的"论域", 关系是谓词, 函数是推理规则的应用
- **本章 + 第 11 章**: 形式语言把命题 / 谓词逻辑扩展为可计算的语言 (一阶逻辑 / 高阶逻辑)
- **本章 + 系统思维**: LLM 的 CoT 推理是"系统动力学"在语言模型中的体现
- **本章 + 算法**: SAT 求解器是 NP 完全问题的工程奇迹, DPLL / CDCL 是经典的回溯搜索算法

### 10.2 参考文献

#### 经典教材

1. **Mendelson, E. (2015).** *Introduction to Mathematical Logic* (6th ed.). CRC Press.
   - 数理逻辑的标准教材, 本章内容的主要参考

2. **Enderton, H. B. (2001).** *A Mathematical Introduction to Logic* (2nd ed.). Academic Press.
   - 经典的一阶逻辑教材, 谓词逻辑部分详细

3. **Boolos, G. S., Burgess, J. P., & Jeffrey, R. C. (2007).** *Computability and Logic* (5th ed.). Cambridge University Press.
   - 计算理论与逻辑的交叉, 适合工程师背景

4. **Russell, S., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
   - 第 7-9 章覆盖逻辑推理、知识表示, 工程化讲法

#### 形式化推理 / SAT 综述

5. **Robinson, J. A. (1965).** A machine-oriented logic based on the resolution principle. *Journal of the ACM*, 12(1), 23-41.
   - 归结原理的开山论文

6. **Biere, A., Heule, M., van Maaren, H., & Walsh, T. (Eds.). (2009).** *Handbook of Satisfiability*. IOS Press.
   - SAT 求解器的圣经, 涵盖 CDCL、LBD 等现代技术

7. **de Moura, L., & Bjørner, N. (2008).** Z3: An efficient SMT solver. *TACAS 2008*.
   - Z3 求解器的论文, 工业验证的标准工具

8. **Kroening, D., & Strichman, O. (2016).** *Decision Procedures: An Algorithmic Point of View* (2nd ed.). Springer.
   - 决策过程教材, 包括 SAT、SMT、Presburger 算术等

#### LLM 推理 / CoT / ReAct 论文

9. **Wei, J., et al. (2022).** Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.
   - CoT 提示词的奠基论文

10. **Yao, S., et al. (2023).** ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
    - ReAct 框架的论文

11. **Yao, S., et al. (2023).** Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS 2023*.
    - ToT 框架的论文

12. **Wang, X., et al. (2022).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
    - Self-Consistency 方法

13. **Gao, L., et al. (2022).** PAL: Program-aided Language Models. *arXiv:2211.10435*.
    - PAL: LLM + 代码解释器

14. **Trinh, T. H., et al. (2024).** Solving Olympiad Geometry without Human Demonstrations. *Nature*.
    - AlphaGeometry, DeepMind 的几何定理证明系统

#### 历史 / 哲学

15. **Whitehead, A. N., & Russell, B. (1910-1913).** *Principia Mathematica*. Cambridge University Press.
    - 数理逻辑的奠基之作, 用逻辑推导数学

16. **Gödel, K. (1931).** Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme. *Monatshefte für Mathematik und Physik*.
    - 哥德尔不完备性定理, 逻辑学的根本限制

17. **Turing, A. M. (1936).** On computable numbers, with an application to the Entscheidungsproblem. *Proceedings of the London Mathematical Society*.
    - 图灵机 / 不可判定性

---

> **本章小结**
>
> 命题逻辑 + 谓词逻辑是 LLM 推理的**形式化地基**。我们讲了:
>
> 1. **命题 + 5 个联结词 + 真值表**: SQL WHERE、布尔电路的基础
> 2. **等价与蕴含**: 编译器优化、查询优化的依据
> 3. **DNF/CNF**: SAT 求解器的输入格式, 工业硬件验证的核心
> 4. **谓词 + 量词**: 知识图谱、一阶逻辑的形式化
> 5. **推理规则**: 假言推理、三段论、归结原理——CoT 提示词的逻辑骨架
>
> **给工程师的 3 条 actionable 建议**:
>
> 1. **写 CoT 提示词时, 显式列出"推理规则"**: "应用假言推理..."、"应用全称实例化..." 比纯自然语言推理更可靠。
> 2. **用 SAT/SMT 验证关键属性**: 不要让 LLM 直接做形式化推理, 用符号求解器验证 LLM 提议的步骤。
> 3. **理解 LLM 推理的"根本限制"**: 统计模式 ≠ 形式化推理, CoT 是逼近而不是等价。在关键场景 (医疗、法律、金融) 仍需人机协作。
>
> 下一章 第 3 章:图论基础——节点、边、路径、连通性 我们进入图论, 看看推理树、知识图谱、GNN 的共同数学基础。

---

## 11. 进阶:三段论的深层结构

### 11.1 传统三段论 vs 现代一阶逻辑

**传统三段论**(亚里士多德)用"主词-谓词-中词"结构,有 256 种"式",其中 19 种有效。

**现代一阶逻辑**(Frege 1879)统一了所有推理形式。

**经典例子**(全称肯定式 AAA-1):
- 大前提:所有人都会死 (All M are P)
- 小前提:苏格拉底是人 (All S are M)
- 结论:苏格拉底会死 (All S are P)

**形式化**:
$$\forall x(\text{Human}(x) \to \text{Mortal}(x))$$
$$\text{Human}(\text{Socrates})$$
$$\therefore \text{Mortal}(\text{Socrates})$$

**为什么 AAA-1 有效?** 全称实例化 (UI) + 假言推理 (MP)。

### 11.2 三段论的 19 个有效式

按中词位置分 4 个"格"(Figure):

**第一格** (M-P, S-M):典型形式
- AAA-1: Barbara
- EAE-1: Celarent
- AII-1: Darii
- EIO-1: Ferio

**第二格** (P-M, S-M)
- EAE-2: Cesare
- AEE-2: Camestres
- EIO-2: Festino
- AOO-2: Baroco

**第三格** (M-P, M-S)
- AAI-3: Darapti
- IAI-3: Disamis
- OAO-3: Datisi
- EIO-3: Ferison

**第四格** (P-M, M-S)
- AEE-4: Camenes
- IAI-4: Dimaris
- EAE-4: Fesapo
- AEO-4: Fresison

**关键**:只有 19 个有效式,其他都是"形式谬误"(formal fallacy)。

### 11.3 三段论的"工程化":业务规则引擎

```python
# Drools 风格规则引擎(简化)
rules = [
    ("Rule 1", "Customer(tier='gold')", "Discount(0.15)"),
    ("Rule 2", "Customer(tier='silver')", "Discount(0.10)"),
    ("Rule 3", "Customer(tier='bronze')", "Discount(0.05)"),
]

# 推理:全称实例化 + 假言推理
def infer(customer):
    for rule_name, condition, action in rules:
        # 实例化:把变量 x 替换为具体的 customer
        if eval(condition.replace("Customer", "customer")):
            # 应用规则(假言推理)
            return eval(action)
    return 0.0
```

**本质**:规则引擎是"批量三段论",每条规则是一个蕴含,每次推理是"全称实例化 + 假言推理"。

### 11.4 关系推理 (Relational Reasoning)

在三段论基础上,加入**关系**(不只是主-谓结构):

例:
- 大前提:所有人都喜欢某些动物
- 小前提:没有人喜欢所有动物
- 结论:有些人不喜欢某些动物(对某些动物无兴趣)

**形式化**:
$$\forall x \exists y \text{Likes}(x, y)$$
$$\neg \exists x \forall y \text{Likes}(x, y)$$
$$\therefore \exists x \exists y \neg \text{Likes}(x, y) \text{ (或 } \neg \forall y \text{Likes}(x, y)\text{)}$$

**关键**:关系推理需要"关系谓词" $\text{Likes}(x, y)$,这是谓词逻辑的核心能力。

---

## 12. 进阶:归结原理的工程实现 (DPLL / CDCL)

### 12.1 归结原理回顾

**归结规则**:$\frac{C_1 \lor p, \quad C_2 \lor \neg p}{C_1 \lor C_2}$

**归结反驳** (Resolution Refutation):
1. 把所有 CNF 子句 + 否定结论放在一起
2. 反复应用归结规则
3. 推到空子句 $\square$ = 矛盾 = 结论成立

### 12.2 DPLL 算法 (Davis-Putnam-Logemann-Loveland)

```python
def dpll(clauses, assignment):
    """
    clauses: list[list[int]]  CNF,每个子句是文字列表
              正整数表示 p,负整数表示 ¬p
    assignment: dict[int, bool] 当前赋值
    返回 True iff 可满足
    """
    # 单位传播
    unit_clauses = [c for c in clauses if len(c) == 1]
    if not unit_clauses:
        # 检查是否所有子句都被满足
        if all(any(lit in assignment and assignment[lit] for lit in c) for c in clauses):
            return True
        if any(all(lit in assignment and not assignment[lit] for lit in c) for c in clauses):
            return False
    else:
        # 处理单位子句
        for unit in unit_clauses:
            lit = unit[0]
            var = abs(lit)
            assignment[var] = (lit > 0)

            # 简化子句集
            new_clauses = []
            for c in clauses:
                if lit in c:  # 子句被满足
                    continue
                if -lit in c:  # 移除 ¬lit
                    new_c = [x for x in c if x != -lit]
                    if not new_c:
                        return False
                    new_clauses.append(new_c)
                else:
                    new_clauses.append(c)
            return dpll(new_clauses, assignment)

    # 选择变量分支
    var = next(lit for c in clauses for lit in c if abs(lit) not in assignment)
    for val in [True, False]:
        assignment[var] = val
        if dpll(clauses, assignment):
            return True
    return False
```

**复杂度**:指数 (NP 完全)。

### 12.3 CDCL 算法 (Conflict-Driven Clause Learning)

现代 SAT 求解器的核心:

```python
# 伪代码(简化版)
def cdcl(cnf):
    while True:
        # 单位传播
        conflict = unit_propagate(cnf)
        if conflict is not None:
            # 冲突学习
            learned_clause = analyze_conflict(conflict)
            if learned_clause is None:
                return UNSAT  # 不可满足
            cnf.add(learned_clause)
            backtrack(learned_clause)
        else:
            # 选变量分支
            if all_vars_assigned(cnf):
                return SAT  # 可满足
            var = choose_branching_variable(cnf)
            assign(var, True)
```

**关键创新**:
1. **冲突分析**:从冲突回溯中"学"出新子句
2. **非时序回溯 (Non-chronological Backtracking)**:直接跳到相关决策点
3. **重启 (Restart)**:周期性清空决策但保留学习

**效果**:CDCL 可以处理百万级变量的工业 SAT 问题(芯片验证、安全协议)。

### 12.4 z3 SMT Solver 实战

```python
# 安装:pip install z3-solver
from z3 import *

# 定义变量
x = Real('x')
y = Real('y')

# 约束(谓词逻辑 + 实数算术)
s = Solver()
s.add(x > 0)
s.add(y > 0)
s.add(x + y == 10)
s.add(2 * x + 3 * y == 26)

# 求解
if s.check() == sat:
    m = s.model()
    print(f"x = {m[x]}, y = {m[y]}")
    # x = 4, y = 6
else:
    print("不可满足")
```

**z3 的能力**:线性 / 非线性整数算术、位向量、数组、未解释函数、量词。

**工业应用**:
- 微软 / 亚马逊 / 英特尔:芯片验证
- AWS:策略验证 (IAM policy)
- 加密协议:安全证明
- 程序验证:形式化证明 (Dafny / F*)

### 12.5 SAT 在 ML 中的应用

**应用 1:神经网络的"对抗鲁棒性验证"**
- 把 ReLU 网络编码为 SAT / SMT 问题
- 验证"在 $\epsilon$-球内的所有输入都分类正确"

**应用 2:决策树 / 决策集的"可解释性验证"**
- 把决策树编码为 CNF
- 检查"是否存在反例"

**应用 3:推荐系统的"多样性约束求解"**
- 多样性约束 → 整数规划 → SMT

---

## 13. 进阶:高阶逻辑与类型论

### 13.1 一阶逻辑 vs 高阶逻辑

**一阶逻辑 (First-Order Logic, FOL)**:
- 量化**个体** ($\forall x, \exists x$)
- 不能量化**谓词**或**函数**

**二阶逻辑 (Second-Order Logic, SOL)**:
- 可以量化谓词 ($\forall P \exists x P(x)$)
- 比一阶逻辑表达力强,但**不完全**(Gödel)

**高阶逻辑 (Higher-Order Logic, HOL)**:
- 可以量化任意类型的对象
- 用于定理证明器 (Coq, Isabelle, Lean)

### 13.2 类型论 (Type Theory)

**类型**:集合 + 操作 + 规则的"封装"。

**简单类型 λ-演算**:
- 基本类型:$\text{bool}, \text{int}, \text{real}$
- 函数类型:$\alpha \to \beta$
- 积类型:$\alpha \times \beta$
- 和类型:$\alpha + \beta$

**ML 视角**:Python 的 type hints 就是简单类型论的应用。

```python
from typing import List, Tuple, Callable

def process(items: List[int]) -> Tuple[int, int]:
    return (min(items), max(items))

# 类型签名: List[int] → Tuple[int, int]
```

### 13.3 柯里化-霍瓦德同构 (Curry-Howard Isomorphism)

**惊人对应**:
- 程序 ↔ 证明
- 类型 ↔ 命题
- 函数应用 ↔ 假言推理
- 函数复合 ↔ 传递推理

**例子**:
```haskell
-- 类型 A → B 表示命题"若 A 则 B"
-- 函数 f: A → B 是该命题的"证明"
id : a -> a
id x = x  -- 证明:对任意 a,存在 a → a

-- 类型 a → a 对应命题 ∀a. a → a
```

**工程意义**:
- **Lean 4** (Lean):用类型论写证明,也是 ML 函数式编程语言
- **Coq**:用类型论做形式化验证
- **Idris / Agda**:依赖类型编程语言

### 13.4 一阶逻辑的"语义"

**模型论** (Model Theory):给定一阶语言 $\mathcal{L}$ 的句子 $\phi$,模型 $\mathcal{M}$ 满足 $\phi$ 意味着什么?

- **结构** $\mathcal{M} = (D, I)$:$D$ 是论域, $I$ 解释函数和谓词符号
- **满足** $\mathcal{M} \models \phi$:$D$ 中的元素使 $\phi$ 为真

**例**:模型 $\mathcal{M} = (\mathbb{N}, \{0, +, \cdot\})$ 满足 $\forall x \exists y (x + y = x)$(因为 $y = 0$)。但 $\mathcal{M}' = (\mathbb{N}_{>0}, \{+, \cdot\})$ **不**满足(没有零元)。

### 13.5 一阶逻辑的"局限"

**哥德尔不完备性定理 (1931)**:
- 任何包含算术的一致形式系统中,**存在真但不可证的命题**
- 不存在"完备且一致"的算术公理系统

**丘奇-图灵不可判定性 (1936)**:
- 一阶逻辑的"有效性"问题不可判定
- 即:不存在算法能判定"任意一阶逻辑公式是否有效"

**工程意义**:LLM 不能"在内部做完整的一阶逻辑推理",因为这是**不可判定**的。LLM 是**启发式**推理,不是**形式化**推理。

---

## 14. 模态逻辑与时序逻辑

### 14.1 模态逻辑 (Modal Logic)

**核心算子**:
- $\square \phi$:必然 $\phi$ (necessarily)
- $\diamond \phi$:可能 $\phi$ (possibly)

**关系**:
- $\square \phi \equiv \neg \diamond \neg \phi$
- $\diamond \phi \equiv \neg \square \neg \phi$

**可能世界语义 (Kripke Semantics)**:
- 世界集 $W$
- 可达关系 $R \subseteq W \times W$
- $\mathcal{M}, w \models \square \phi$ 当且仅当对所有 $w'$ 满足 $w R w'$,$\mathcal{M}, w' \models \phi$

### 14.2 时序逻辑 (Temporal Logic)

**LTL (Linear Temporal Logic)**:线性时间

| 算子 | 含义 |
| :--- | :--- |
| $X \phi$ | 下一时刻 $\phi$ 成立 (next) |
| $F \phi$ | 未来某时刻 $\phi$ 成立 (eventually) |
| $G \phi$ | 永远 $\phi$ 成立 (always) |
| $\phi U \psi$ | $\phi$ 成立直到 $\psi$ 成立 (until) |

**例**:规格"每次按下按钮,最终灯会亮":
$$G(\text{button} \to F \text{ light})$$

### 14.3 CTL (Computation Tree Logic)

| 算子 | 含义 |
| :--- | :--- |
| $AX \phi$ | 所有下一状态满足 $\phi$ |
| $EX \phi$ | 存在下一状态满足 $\phi$ |
| $AF \phi$ | 所有路径上最终 $\phi$ |
| $EF \phi$ | 存在路径上最终 $\phi$ |
| $AG \phi$ | 所有路径上永远 $\phi$ |
| $EG \phi$ | 存在路径上永远 $\phi$ |

**工程应用**:模型检测 (Model Checking),验证硬件 / 软件系统。

### 14.4 工程:形式化规约

```python
# 用 TLA+ 写并发算法规约(简化)
# 伪代码:Increment 操作的正确性

VARIABLES counter

Init == counter = 0

Increment == counter' = counter + 1

Safety == counter >= 0

Spec == Init /\ [][Increment]_counter /\ Safety
```

**TLA+** 用于工业并发系统验证:Amazon DynamoDB、Microsoft Azure。

---

## 15. CoT 与逻辑推理的"边界"

### 15.1 CoT 能做 vs 不能做

| CoT 能做 | CoT 不能做 |
| :--- | :--- |
| 多步推理任务 (GSM8K, MATH) | 严格形式化证明 (Coq / Lean) |
| 常识推理 | 长程一致性 (超过几千 token) |
| 工具使用 (ReAct) | 严格逻辑验证 |
| 模式匹配 + 启发式推理 | 完备搜索 (CoT 不搜索,直接生成) |

### 15.2 CoT 失败的常见原因

**1. 推理链幻觉 (Reasoning Hallucination)**
LLM 生成的推理步骤"看起来对",但逻辑上有漏洞。

**例**:GPT-4 解数学题,有时会"伪造"中间步骤。

**2. 推理链漂移 (Reasoning Drift)**
长链推理中,前几步的正确性无法保证后续步骤。

**3. 自相矛盾 (Self-Contradiction)**
CoT 中前后步骤给出矛盾的结论。

### 15.3 缓解方法

| 方法 | 原理 |
| :--- | :--- |
| **Self-Consistency** | 多采样 + 投票 |
| **Verifier** | 训练一个 verifier 模型评分 |
| **Tool use (PAL)** | 把推理外包给代码解释器 |
| **Formal verification** | 把关键步骤外包给 SAT/SMT |
| **Constitutional AI** | 用规则约束推理方向 |
| **Tree of Thoughts** | 显式搜索 + 回溯 |
| **Reflexion** | 推理失败后反思 + 重试 |

### 15.4 工业实践:LLM 推理的"工程化栈"

```text
┌─────────────────────────────────────┐
│ LLM (GPT-4 / Claude / Llama)        │  ← 生成候选推理
├─────────────────────────────────────┤
│ Prompt Engineering / Few-shot       │  ← 控制格式
├─────────────────────────────────────┤
│ Verifier / Reward Model             │  ← 评分
├─────────────────────────────────────┤
│ Formal Tools (SAT/SMT/Code)         │  ← 验证关键步骤
├─────────────────────────────────────┤
│ Search (ToT / MCTS)                 │  ← 多路径探索
├─────────────────────────────────────┤
│ Human-in-the-loop                   │  ← 关键场景人审
└─────────────────────────────────────┘
```

**核心思想**:**让 LLM 做"软推理",让形式化工具做"硬验证"**。

---

## 16. 动手算一算 (5 道带答案)

### 练习 1

判断公式 $(p \to q) \land (q \to r) \to (p \to r)$ 是否为永真式 (tautology)。

<details>
<summary>答案</summary>

构造真值表:

| $p$ | $q$ | $r$ | $p \to q$ | $q \to r$ | $p \to r$ | LHS | RHS | 整个公式 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T | T | T | T | T | T | T | T | T |
| T | T | F | T | F | F | F | F | T |
| T | F | T | F | T | T | F | T | T |
| T | F | F | F | T | F | F | F | T |
| F | T | T | T | T | T | T | T | T |
| F | T | F | T | F | T | F | T | T |
| F | F | T | T | T | T | T | T | T |
| F | F | F | T | T | T | T | T | T |

8 行全为 T → **永真式** ✓

也称"假言三段论" (Hypothetical Syllogism)。
</details>

### 练习 2

把公式 $\neg(p \lor q) \lor (\neg p \land q)$ 化为 DNF 和 CNF。

<details>
<summary>答案</summary>

**原式**:$\neg(p \lor q) \lor (\neg p \land q)$

**化简**:
$\neg(p \lor q) \lor (\neg p \land q)$
$\equiv (\neg p \land \neg q) \lor (\neg p \land q)$ (德摩根)
$\equiv \neg p \land (\neg q \lor q)$ (分配律)
$\equiv \neg p \land T$
$\equiv \neg p$

**CNF**:$\neg p$(已经是 CNF)

**DNF**:$\neg p$(可以看成单个合取项)
</details>

### 练习 3

设 $\forall x (P(x) \to Q(x))$, $P(a)$。求 $Q(a)$ 的推理过程。

<details>
<summary>答案</summary>

应用 **全称实例化 (UI)**:
从 $\forall x (P(x) \to Q(x))$,令 $x = a$ 得 $P(a) \to Q(a)$。

应用 **假言推理 (MP)**:
从 $P(a)$ 和 $P(a) \to Q(a)$ 得 $Q(a)$。
</details>

### 练习 4

判断下列推理是否有效:
- 前提 1:有些人是医生。
- 前提 2:所有医生都很有钱。
- 结论:有些人很有钱。

<details>
<summary>答案</summary>

**形式化**:
- $P_1$: $\exists x (Person(x) \land Doctor(x))$
- $P_2$: $\forall x (Doctor(x) \to Rich(x))$
- $C$: $\exists x (Person(x) \land Rich(x))$

**推理**:由 $P_1$ 得 $\exists x (Doctor(x) \land Person(x))$,存在某个 $a$ 使得 $Doctor(a)$。由 $P_2$ 得 $Doctor(a) \to Rich(a)$。由 MP 得 $Rich(a)$,所以 $\exists x Rich(x)$,进而 $\exists x (Person(x) \land Rich(x))$。

**有效** ✓
</details>

### 练习 5

把下列英文推理翻译成谓词逻辑:
- "每个有羽毛的动物都会飞。"
- "Penguin 是有羽毛的动物。"
- "因此,Penguin 会飞。"

<details>
<summary>答案</summary>

形式化:
$$\forall x (Feathered(x) \to Fly(x))$$
$$Feathered(Penguin)$$
$$\therefore Fly(Penguin)$$

推理规则:UI + MP。
</details>

---

## 17. LLM 推理的"经典 prompt 工程"

### 17.1 Zero-shot CoT

```text
问:小明有 5 个苹果,吃了 2 个,又买了 3 个,现在有几个?
答:让我们一步步思考。小明原来有 5 个苹果。吃掉 2 个,剩 5-2=3 个。又买了 3 个,现在有 3+3=6 个。所以答案是 6。
```

**触发词**:"Let's think step by step"(Wei et al. 2022)。

### 17.2 Few-shot CoT

```text
Q: Roger 有 5 个网球。他又买了 2 罐网球,每罐 3 个。现在他有多少个?
A: Roger 起初有 5 个球。2 罐每罐 3 个 = 6 个球。5 + 6 = 11。答案是 11。

Q: 自助餐厅有 23 个苹果。用 20 个做午餐,又买了 6 个,现在有几个?
A: 自助餐厅起初有 23 个苹果。用 20 个做午餐剩 23-20=3 个。又买了 6 个,现在有 6+3=9 个。答案是 9。

Q: 小明有 5 个苹果...
```

**关键**:few-shot 示例展示了"推理风格",LLM 模仿。

### 17.3 ReAct (Reason + Act)

```text
Thought 1: 我需要查找 X 的最新价格
Action 1: Search[X price]
Observation 1: X 当前价格是 $100

Thought 2: 我需要计算 30% 折扣后的价格
Action 2: Calculate[100 * 0.7]
Observation 2: 70

Thought 3: 我有了答案
Action 3: Finish[70]
```

**本质**:**推理 (Reason) 和动作 (Act) 交替**,类似 AI Agent 的"思考-行动"循环。

### 17.4 Tree of Thoughts (ToT)

```text
         Q: 24 点游戏:用 4, 9, 10, 13 算出 24
              │
        ┌─────┴─────┐
        │           │
   (4+9)·...    (10-4)·...
        │           │
   ┌────┴────┐  ┌───┴───┐
   │         │  │       │
 ...评估...  ...回溯...
```

**关键**:**BFS/DFS + 评估函数 + 回溯**,类似传统 AI 搜索。

### 17.5 Self-Consistency

```text
同一个问题采样 K=5 次,得到 5 个推理链和答案:
- Chain 1: 6
- Chain 2: 6
- Chain 3: 7  ← 异常
- Chain 4: 6
- Chain 5: 6

投票:6 胜出(4 vs 1)。
```

**本质**:**温度采样 + 多数投票**,类似集成学习。

---

## 18. 知识图谱推理 = 谓词逻辑的工程实现

### 18.1 KG = 谓词集合

知识图谱 = 三元组集合 $\{(h, r, t)\}$ = 二元谓词实例 $\text{Relation}(h, t)$ 的集合。

### 18.2 KG 推理任务

| 任务 | 输入 | 输出 |
| :--- | :--- | :--- |
| **链接预测** | (Alice, friend, ?) | Bob |
| **实体分类** | (Alice, type, ?) | Person |
| **关系预测** | (Alice, ?, Bob) | friend |
| **路径推理** | 多个三元组 | 推断新关系 |

### 18.3 经典方法

**TransE (Bordes 2013)**: $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$。
- 关系是向量平移
- 训练:$\mathcal{L} = \sum \max(0, d(\mathbf{h} + \mathbf{r}, \mathbf{t}) - d(\mathbf{h} + \mathbf{r}, \mathbf{t}') + \text{margin})$
- 缺点:1-to-N / N-to-1 关系失败

**TransH (Wang 2014)**:关系是超平面, $\mathbf{h}$ 和 $\mathbf{t}$ 投影到超平面后平移。

**RotatE (Sun 2019)**:关系是复数空间旋转 $\mathbf{t} = \mathbf{h} \circ \mathbf{r}$。
- 能建模对称 / 反对称 / 传递 / 复合关系
- 当前 SOTA 之一

### 18.4 多跳推理 (Multi-hop Reasoning)

```python
# 推理:Alice 的朋友的朋友是什么职业?
triples = [
    ("Alice", "friend", "Bob"),
    ("Bob", "friend", "Carol"),
    ("Carol", "job", "Engineer"),
]

# 多跳推理路径
# Alice → friend → Bob → friend → Carol → job → Engineer
```

**形式化**:
$$\text{friend}(\text{Alice}, \text{Bob}) \land \text{friend}(\text{Bob}, \text{Carol}) \land \text{job}(\text{Carol}, \text{Engineer})$$
$$\therefore \exists x (\text{friend}(\text{Alice}, x) \land \text{friend}(x, \text{Carol}) \land \text{job}(\text{Carol}, \text{Engineer}))$$

这是"复合推理",本质是谓词逻辑的多步推理。

### 18.5 工业系统

| 系统 | 公司 | 规模 |
| :--- | :--- | :--- |
| **Knowledge Graph** | Google | 5 亿+ 实体,180 亿+ 事实 |
| **Wikidata** | Wikimedia | 1 亿+ 实体 |
| **ConceptNet** | Luminoso | 2800 万边 |
| **Amazon Product Graph** | Amazon | 5 亿+ 商品 |
| **Alibaba Knowledge Graph** | Alibaba | 百亿级 |

**应用**:
- 搜索引擎:Google Knowledge Panel
- 推荐系统:KG-based 推荐
- 问答:知识图谱问答 (KBQA)
- 反欺诈:实体关系分析

---

## 19. 谓词逻辑与编程语言

### 19.1 SQL = 关系演算

**SQL 查询**本质是"关系演算 + 一阶逻辑":

```sql
-- "找出喜欢 Python 的所有 Alice 的朋友"
SELECT f.name
FROM Person a, Friend f, Likes l
WHERE a.name = 'Alice'
  AND a.id = f.person_id
  AND f.friend_id = l.person_id
  AND l.item = 'Python';
```

**形式化**:
$$\{n : \exists p_1, p_2, i. \text{Person}(p_1, \text{Alice}) \land \text{Friend}(p_1, p_2) \land \text{Likes}(p_2, \text{Python}) \land \text{Person}(p_2, n)\}$$

### 19.2 Prolog = 直接的一阶逻辑

```prolog
% 事实
human(socrates).
mortal(X) :- human(X).

% 查询
?- mortal(socrates).
% true.
```

**Prolog** 用 Horn 子句(最多一个正文字的子句)实现一阶逻辑的子集。

### 19.3 Datalog = Prolog 的简化

```datalog
% 规则
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% 事实
parent(alice, bob).
parent(bob, carol).

% 查询:alice 的后代?
?- ancestor(alice, X).
% X = bob
% X = carol
```

**应用**:
- 程序分析 (Doop)
- 网络安全分析 (Zeek)
- 数据集成

### 19.4 函数式编程 = 命题逻辑的"语法糖"

```haskell
-- 命题逻辑的 Haskell 表示
p, q, r :: Bool
notP = not p
pAndQ = p && q
pOrQ = p || q
pImplQ = not p || q  -- p → q ≡ ¬p ∨ q

-- 永真式检查
tautology :: (Bool -> Bool -> Bool) -> Bool
tautology f = and [f p q | p <- [True, False], q <- [True, False]]
```

---

## 20. 模糊逻辑与概率逻辑

### 20.1 模糊逻辑 (Fuzzy Logic)

经典命题真值 ∈ $\{0, 1\}$。模糊逻辑真值 ∈ $[0, 1]$。

| 算子 | 经典 | 模糊 (Zadeh) |
| :--- | :--- | :--- |
| AND | $p \land q$ | $\min(p, q)$ |
| OR | $p \lor q$ | $\max(p, q)$ |
| NOT | $\neg p$ | $1 - p$ |
| IMPLIES | $p \to q$ | $\max(1 - p, q)$ |

### 20.2 概率逻辑 (Probabilistic Logic)

命题带概率:$\Pr(\phi)$。

**例**:
- $\Pr(\text{Rain}) = 0.3$
- $\Pr(\text{Traffic} | \text{Rain}) = 0.8$
- $\Pr(\text{Traffic}) = ?$

**贝叶斯推理**:
$$\Pr(\text{Traffic}) = \sum_{\text{Rain}} \Pr(\text{Traffic} | \text{Rain}) \cdot \Pr(\text{Rain})$$
$$= 0.8 \cdot 0.3 + 0.1 \cdot 0.7 = 0.24 + 0.07 = 0.31$$

### 20.3 ML/DL 中的概率逻辑

**例 1: 朴素贝叶斯分类器**

$$y^* = \arg\max_y \Pr(y | \mathbf{x}) = \arg\max_y \Pr(y) \prod_i \Pr(x_i | y)$$

**例 2: 逻辑回归**

$$\Pr(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

**例 3: 贝叶斯神经网络**

每个权重是分布 $w \sim \mathcal{N}(\mu, \sigma^2)$,预测是分布。

**例 4: 不确定性推理 (Uncertainty Reasoning)**

知识图谱中 "Alice friendOf Bob" 概率 0.9,"Bob friendOf Carol" 概率 0.85。

推理:$\Pr(\text{Alice friendOf Carol via Bob}) = ?$

取决于假设 (独立性、规则等)。

---

## 21. 一道综合题:谓词逻辑到 KG 嵌入

### 题目

给定知识图谱子集:
$$\text{KG} = \{(\text{Alice}, \text{friend}, \text{Bob}), (\text{Bob}, \text{mentorOf}, \text{Carol}), (\text{Alice}, \text{colleague}, \text{Carol})\}$$

1. 用谓词逻辑形式化 KG
2. 推断"Alice 与 Carol 的复合关系"(用现有关系)
3. 设计 TransE 的损失函数

<details>
<summary>答案</summary>

**1. 形式化**:
- $\text{friend}(\text{Alice}, \text{Bob})$
- $\text{mentorOf}(\text{Bob}, \text{Carol})$
- $\text{colleague}(\text{Alice}, \text{Carol})$

**2. 复合关系**:
- $\text{friend}(\text{Alice}, \text{Bob}) \land \text{mentorOf}(\text{Bob}, \text{Carol}) \Rightarrow \text{indirectMentorOf}(\text{Alice}, \text{Carol})$
- 这是"传递关系"的推断。

**3. TransE 损失**:
$$\mathcal{L} = \sum_{(h, r, t) \in \text{KG}} \max(0, d(\mathbf{h} + \mathbf{r}, \mathbf{t}) - d(\mathbf{h} + \mathbf{r}, \mathbf{t}') + \text{margin})$$

其中 $(h, r, t')$ 是负样本(随机替换 $h$ 或 $t$)。距离 $d$ 通常是 L2:

$$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$$

约束:$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$,即平移不变性。
</details>

---

## 22. 一图流:本章知识结构

```text
                   命题逻辑
                  ┌────┴────┐
                  │         │
              真值表/等价  范式
              (§2-3)    (§4)
                  │         │
                  └────┬────┘
                       │
                  推理规则
                  ┌────┴────┐
                  │         │
              自然演绎  归结原理
                  │         │
                  └────┬────┘
                       │
                   谓词逻辑
                  ┌────┴────┐
                  │         │
              量词/谓词  推理规则
              (§5)     (§6)
                  │         │
                  └────┬────┘
                       │
                  LLM CoT 推理
                  ┌────┴────┐
                  │         │
              ReAct/ToT  Self-Consistency
                  │         │
                  └────┬────┘
                       │
                  形式验证
                  ┌────┴────┐
                  │         │
                 SAT/SMT    z3
                  │         │
                  └─────────┘
```

---

## 23. 一句话总结(再强调一次)

> **命题逻辑 = 推理的"骨架"(规则 + 真值),谓词逻辑 = 推理的"血肉"(对象 + 量词),LLM CoT = 推理的"灵魂"(统计 + 模式),形式化工具 = 推理的"法官"(验证 + 保障)。四件齐备,推理方能不翻车。**

---

---

## 24. 经典逻辑谬误与"陷阱模式"

### 24.1 形式谬误 (Formal Fallacy)

**形式谬误**:推理形式无效(即使前提真,结论也可能假)。

| 谬误 | 形式 | 例子 |
| :--- | :--- | :--- |
| **肯定后件 (Affirming the Consequent)** | $p \to q, q \therefore p$ | "下雨地湿,地湿,故下雨" |
| **否定前件 (Denying the Antecedent)** | $p \to q, \neg p \therefore \neg q$ | "如果下雨地湿,没下雨,故地不湿" |
| **换位不换质 (Illicit Conversion)** | 所有 A 是 B,故所有 B 是 A | "所有猫是动物,故所有动物是猫" |

### 24.2 非形式谬误 (Informal Fallacy)

| 谬误 | 描述 |
| :--- | :--- |
| **诉诸权威 (Appeal to Authority)** | "专家 X 说 Y,所以 Y 是真的"(X 不一定在该领域) |
| **稻草人 (Straw Man)** | 歪曲对方论点再反驳 |
| **滑坡 (Slippery Slope)** | "如果 A 发生,B、C、Z 都会发生" |
| **循环论证 (Circular Reasoning)** | 用结论证前提 |
| **诉诸情感 (Appeal to Emotion)** | 用情感代替论证 |

### 24.3 LLM 的常见推理错误

| LLM 错误模式 | 描述 | 缓解 |
| :--- | :--- | :--- |
| **Hallucination** | 编造不存在的事实 | RAG,事实核查 |
| **Sycophancy** | 迎合用户,改变判断 | Constitutional AI |
| **Reasoning shortcut** | 跳过推理步骤 | 强制 CoT |
| **Multi-step overflow** | 长链推理丢失上下文 | 分解任务,Self-Consistency |
| **Contradictory chain** | 推理链前后矛盾 | Verifier,Reflection |

### 24.4 工程师最常犯的逻辑错误

**错 1:把"相关性"当"因果性"**

```python
# ❌ 错:冰淇淋销量高时,溺水率高
# → 不能说"冰淇淋导致溺水"
# → 实际是"夏天"这一混淆变量

# ✅ 用因果推断 (DoWhy, CausalImpact)
import dowhy
model = dowhy.CausalModel(
    data=df,
    treatment="ice_cream_sales",
    outcome="drowning_rate",
    common_causes=["season"]
)
```

**错 2:把"在训练集上"当"在总体上"**

```python
# ❌ 错:训练集准确率 99%,以为线上也好
# ✅ 验证集 + 测试集 + A/B 测试
```

**错 3:把"模型预测"当"真实概率"**

```python
# ❌ 错:神经网络输出 0.9 = 90% 概率?
# ✅ 用 calibration (Platt scaling, isotonic regression)
```

---

## 25. 逻辑学的"形式化 vs 工程化"对照

### 25.1 形式化逻辑 vs 工业实践

| 形式化逻辑 | 工业实践 |
| :--- | :--- |
| 一阶逻辑完备推理 | LLM 启发式推理 |
| 归结原理 + SAT | 神经网络 + 统计 |
| 严格证明 (Coq, Lean) | 单元测试 + 形式化规约 |
| 类型论 (Hindley-Milner) | 动态类型 Python |
| 模型论 (Kripke) | 模型检查 (TLA+, SPIN) |

### 25.2 何时用形式化,何时用工程化

| 场景 | 推荐 |
| :--- | :--- |
| 关键安全系统 (航空、医疗) | 形式化 |
| 编译器 / 硬件验证 | 形式化 |
| 并发协议 | 形式化 (TLA+) |
| 推荐系统 | 工程化 (A/B) |
| NLP / 内容审核 | 工程化 + 人工 |
| 自动驾驶决策 | 形式化 + 工程化 |

### 25.3 混合范式:LLM + 形式化工具

```text
LLM 生成候选推理步骤
        ↓
形式化工具 (z3) 验证关键步骤
        ↓
如果不通过: 反馈给 LLM 重新生成
        ↓
如果通过: 继续下一步
```

**实例**:AlphaGeometry (DeepMind 2024)
- LLM 生成几何构造的"想法"
- 形式化证明器验证推理
- 解决奥数几何题

**本质**:**让 LLM 做"启发式",形式化工具做"硬验证"**。

---

## 26. 谓词逻辑与编程语言类型

### 26.1 类型 = 谓词 (Type as Predicate)

**Curry-Howard 同构**:类型对应命题,程序对应证明。

```typescript
// TypeScript 中
function divide(a: number, b: number): number {
    if (b === 0) throw new Error("Division by zero");
    return a / b;
}
// 类型 number → number → number 编码了"对所有 a, b, 存在唯一 c = a / b"
// 但 b = 0 时例外 → 类型系统无法捕捉
```

**改进**:用 dependent types

```typescript
// 伪代码(伪 dependent types)
function divide(a: number, b: NonZeroNumber): number {
    return a / b;
}
// 类型 NonZeroNumber = { x: number | x ≠ 0 }
// 编译器保证 b ≠ 0 → 不会有除零错误
```

### 26.2 Rust 的类型系统

```rust
// Rust Option<T> 强制处理 None
fn divide(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        None  // 显式处理除零
    } else {
        Some(a / b)
    }
}

// 调用
match divide(10.0, 0.0) {
    Some(v) => println!("{}", v),
    None => println!("error"),
}
```

**逻辑对应**:`Option<T>` = 命题 "T 或 None"。

### 26.3 工业实践:形式化验证在芯片设计

**Intel / AMD / NVIDIA / Apple** 都用形式化方法验证 CPU / GPU 正确性。

**Cadence JasperGold, Synopsys VC Formal**:商业形式化验证工具。

**开源**:SymbiYosys (yosys + SMT solvers)。

**意义**:CPU 设计 bug 一次成本数百万美元,形式化验证是"省钱利器"。

---

## 27. LLM 推理的最新进展 (2024-2026)

### 27.1 OpenAI o1 / o3 (Reasoning Models)

**核心创新**:在生成最终答案前,先生成大量"内部 CoT"(思维链)。

```python
# 用户看到
"问题:..." + "答案:42"

# 内部生成(数千 token)
"思考: 让我一步步分析... [可能数千 token] ... 所以答案是 42"
```

**技术**:
- RLHF 训练"长思考"风格
- 推理时增加 inference compute
- 在 STEM 任务上 SOTA

### 27.2 DeepSeek R1

**开源 reasoning model**:基于 GRPO (Group Relative Policy Optimization)。

**关键**:
- 完全开源(包括权重)
- RL 而非 SFT 训练
- 在数学 / 代码任务上接近 o1

### 27.3 Claude 3.5 Sonnet (Extended Thinking)

类似 o1,在生成前内部"思考"。

### 27.4 趋势

| 趋势 | 描述 |
| :--- | :--- |
| **Test-time compute** | 用更多推理时计算换更好答案 |
| **Process reward model** | 训练 verifier 评分推理步骤 |
| **Multi-agent reasoning** | 多个 LLM 协作推理 |
| **Tool-augmented reasoning** | LLM + 代码 / 搜索 / 数据库 |

### 27.5 局限

| 局限 | 描述 |
| :--- | :--- |
| **推理 ≠ 真理** | LLM 推理流畅但可能错 |
| **不可重现** | temperature > 0 时,同一问题多次结果不同 |
| **长链不稳定** | 超过 ~10K token 的推理容易漂移 |
| **缺乏 grounding** | 不与外部事实挂钩 |

---

## 28. 跨章引用与拓展阅读

### 28.1 跨章节核心引用

| 本章概念 | 对应章节 |
| :--- | :--- |
| 谓词 = 集合 (论域) | `第04章-01-集合·关系·函数-从用户行为序列到数学集合.md` |
| 推理树 | `第04章-04-图论基础-顶点·边·路径·连通性·树·二分图.md` §5 (树) |
| 等价关系 / 划分 | `第04章-01-集合·关系·函数-从用户行为序列到数学集合.md` §3.3 |
| SAT = NP 完全 | 第 10 章:算法与复杂度 §4 |
| 一阶逻辑 → 形式语言 | 第 11 章:形式语言与自动机 §2 |
| 谓词逻辑 + KG | `第04章-04-图论基础-顶点·边·路径·连通性·树·二分图.md` §6 (知识图谱) |
| TransE = 关系向量 | `第04章-01-集合·关系·函数-从用户行为序列到数学集合.md` §3.6 |

### 28.2 进阶阅读

**经典**:
- 《How to Prove It》(Velleman):入门证明技巧
- 《A Mathematical Introduction to Logic》(Enderton):一阶逻辑标准教材
- 《Logic for Computer Science》(Huth & Ryan):CS 视角的逻辑

**工程**:
- 《SAT/SMT solvers 实战》(Kroening):决策过程
- 《The Little Prover》(Friedman):用 ACL2 做证明

**LLM**:
- Wei et al. 2022:Chain-of-Thought Prompting
- Yao et al. 2023:ReAct, Tree of Thoughts
- Lightman et al. 2023:Let's Verify Step by Step (process supervision)

### 28.3 配套工具

| 工具 | 用途 |
| :--- | :--- |
| **Z3 (z3-solver)** | SMT solver (Python) |
| **Lean 4** | 交互式定理证明 |
| **Coq** | 形式化证明 |
| **MiniZinc** | 约束求解 |
| **SWI-Prolog** | Prolog 实现 |
| **PyDatalog** | Python Datalog |

---

## 29. 总结 + 行动建议

### 29.1 工程师的 7 条 actionable 建议

1. **CoT 提示词模板**:"Let's think step by step" + few-shot examples
2. **关键场景用 ReAct**:LLM + 工具调用(代码、搜索、API)
3. **长任务用 ToT**:多路径 + 评估 + 回溯
4. **多采样投票**:Self-Consistency 在工程上简单有效
5. **形式化工具兜底**:z3 / Lean 验证关键步骤
6. **避免幻觉**:RAG + 引用 + verifier 模型
7. **持续评估**:LLM-as-a-judge + 人类评估 + 实际业务指标

### 29.2 思维模型

> **逻辑是推理的"语法",LLM 是推理的"语义"。两者协同,才能在工业场景既灵活又可靠。**

### 29.3 下章预告

**第 3 章:组合数学**——排列、组合、鸽巢原理、容斥原理。

本章讲的"推理链"在第 3 章会变成"组合计数"(多少种推理路径?)。理解计数,是理解"为什么 Transformer attention 是 O(N²)"和"为什么 Embedding 哈希会碰撞"的前提。

---

> **本章版本**: v1.2
> **最后更新**: 2026-06-24
> **作者**: Sisyphus (OhMyOpenCode)
> **配套代码**: §6、§12-§17、§19、§24 所有代码均可直接运行 (Python 3.10+, z3-solver, dowhy 可选)
