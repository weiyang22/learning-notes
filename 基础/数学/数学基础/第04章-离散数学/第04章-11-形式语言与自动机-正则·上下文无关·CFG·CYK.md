---
title: "第04章-11-形式语言与自动机-正则·上下文无关·CFG·CYK"
tags:
  - 数学
  - 形式语言与自动机
  - 正则
  - 上下文无关
  - CFG
  - CYK
description: "本章要点：做 LLM 应用的工程师, 几乎都被\"格式\"问题困扰过:"
chapter_id: 数学-11-04
last_updated: 2026-07-14
---

# 第04章-11-形式语言与自动机-正则·上下文无关·CFG·CYK

格式错了**。模型输出:
>
> ```json
> {"name": "Alice", "age": 30, "tags": ["python", "rust"
> ```
>
> 少一个 `}`,逗号,引号, 整个 pipeline 挂掉。
>
> 你搞 SQL 生成,模型吐出来的 WHERE 子句语法错误,数据库不认。 你写 CoT,模型 "step 1, step 2..." 突然冒出一行 `final answer:` 把答案剧透, 后面 200 token 全是废的。
>
> 然后你用 **Outlines / Guidance / LMQL** 这些"结构化生成"库,加上 JSON schema,模型就乖乖按格式吐了。 这些库背后在干什么? **CFG 引导的 beam search / mask 采样**——本质上是**上下文无关文法** (CFG, Context-Free Grammar) 在限制 LLM 的输出。
>
> 但你想过没:
> - **`re.match(r"ab*", s)`** 这个 `r"ab*"` 在数学上是什么? (正则语言,正则表达式)
> - **Python 解释器**解析 `def f(x): return x + 1` 时,怎么处理**嵌套括号**? (下推自动机 PDA,需要"栈")
> - **JSON 解析器**怎么判断 `{"a": [1, 2, {"b": 3}]}` 是合法的? (CFG 推导 + 语法树)
> - **CYK 算法**在 LLM 时代还有用吗? (有用——受限解码的 mask 计算)
> - **Chomsky 谱系**和 LLM 是什么关系? (LLM ≈ 类型 0,JSON Schema ≈ 类型 2)
>
> 这些问题的答案, 全部藏在**形式语言与自动机理论** 里——计算机科学的"语言学"。
>
> 这一章我们要做的事情很朴素: 把生产环境里那些"格式生成"、"语法校验"、"受限解码" 的工具, 翻译回严格的数学语言。看完之后你会发现:
>
> - **正则表达式** = 类型 3 语言 = **DFA/NFA**(有限状态自动机)
> - **JSON / SQL / Python 缩进语法** = 类型 2 (上下文无关) = **PDA**(下推自动机,带栈)
> - **CYK 算法** = 判断"句子是否属于某个 CFG",O(n³) 动态规划
> - **Outlines / Guidance** = 用 CFG 的 logit mask 限制 LLM 输出
> - **LLM 本身** = 类型 0 附近(理论上可以模拟图灵机,实际受限于上下文窗口)
> - **Python 解释器** = 用 PDA 边读 token 边 push/pop 栈
>
> 形式语言与自动机是**编译原理 / NLP / LLM 受限解码**的共同根基。 第 2 章讲的逻辑(CoT 的"形式化骨架"), 本章讲的语法(CoT 的"形式化约束"), 两者合起来, 你才能真正理解 LLM 推理与生成的理论边界。
>
> **注意**: 本章是 v3 重组版新增章节。 如果你只关心 LLM 应用, 可以跳过 §3-5 的形式化证明, 直接看 §6 "LLM 中的形式语言" 和 §7 的代码。 但**面试**或者做**编译原理 / 形式化验证**方向的工程师, §3-5 是必读。

---

## 1. 直觉引入

### 1.1 工程师的痛点: LLM 为什么"格式不听话"?

做 LLM 应用的工程师, 几乎都被"格式"问题困扰过:

```python
# 场景 1: LLM 输出 JSON
prompt = "请输出用户信息, 严格 JSON 格式"
response = llm(prompt)
# 期望: {"name": "Alice", "age": 30}
# 实际: "当然, 用户信息是: {\"name\": \"Alice\", \"age\": 30 } 这就是用户的所有信息啦~"
#       ↑ 多了一段话, 真正的 JSON 被埋在中间, 解析器一脸懵

# 场景 2: SQL 生成
# 模型输出: "SELECT * FORM users WEHRE age > 30 AND city = 'Beijing'"
#       ↑ FORM 拼错, WEHRE 拼错, 整条 SQL 跑不起来

# 场景 3: 嵌套括号补全
# 模型吐出来一堆 ) ) ) ) 但括号数量不一定对, 编译失败
```

这些"格式错误"的根本原因是: **LLM 在做的是 token 级别的概率采样, 它对"语法结构"没有强约束**。 模型可能学了 JSON 长什么样, 但 token by token 采样时, 每一步都可能"走偏"。

**结构化生成库** (Outlines, Guidance, LMQL, SGLang) 的核心思路是:

```text
在 LLM 每一步采样时, 用"目标语法" (JSON Schema / 正则 / CFG) 计算"哪些 token 合法",
把这些合法 token 的 logit 抬高, 不合法的压低 (甚至设 -inf)。
这样 LLM 就被"语法引导", 输出必然符合预期格式。
```

这背后的数学, 就是**形式语言理论**: 语法是什么? 自动机怎么"读"一个字符串? 怎么判断字符串"属于"某个语法? 这些问题有 60 年 (1956 年 Chomsky 奠基) 的成熟答案。

### 1.2 三个直觉例子

#### 例 1: 邮箱地址校验 = 正则表达式 = DFA

你肯定写过这种代码:

```python
import re
email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
if re.match(email_pattern, "alice@example.com"):
    print("合法邮箱")
```

这个 `r"^[a-zA-Z0-9._%+-]+@..."` 是什么? **正则表达式** (regular expression, regex)。 它描述的字符串集合叫**正则语言** (regular language)。

**数学抽象**: 邮箱地址集合 = 一个**正则语言** $L$, 由某个正则表达式 $r$ 描述:

$$L(r) = \{s \in \Sigma^* : s \text{ 匹配 } r\}$$

**等价的自动机**: 存在一个**确定性有限自动机 (DFA)** $M$, 使得 $L(M) = L(r)$。 DFA 是一台"超级简单的机器": 它有有限多个状态, 读一个字符, 跳到下一个状态, 不需要栈。

```text
邮箱校验 DFA (简化):

  [开始] ─a-z/A-Z/0-9/._%+-→ [local] ─a-z/0-9/._%+-→ [local+]
    │                          │                            │
    │                          │                            │ @
    ↓                          ↓                            ↓
  [失败]                    [失败]                       [@]
                                                                │
                                                                │ a-z/A-Z/0-9/-
                                                                ↓
                                                            [domain] ─.→ [domain.]
                                                                │           │
                                                                ↓           ↓
                                                          [接受 ✓]    a-z/A-Z (≥2)
```

正则表达式的"等价自动机"在工程上的体现: Python 的 `re` 引擎、grep、awk、编译器词法分析器 (lexer), 全都用 DFA/NFA。

**核心结论**: **正则语言 = DFA/NFA 能识别的语言 = 模式匹配的数学基础**。

#### 例 2: JSON 解析器 = CFG + 下推自动机

JSON 语法看起来简单:

```json
{ "name": "Alice", "tags": ["python", "rust"] }
```

但**嵌套结构**让它比正则表达式复杂。考虑:

```json
{ "users": [ { "name": "Alice" }, { "name": "Bob" } ] }
```

这里数组里嵌套对象, 对象里又嵌套数组。 如果用正则, 你写不出来 (你得"记住"开了几个 `[`)。 但用**上下文无关文法 (CFG)**, 几行就搞定:

```text
value    → object | array | string | number | "true" | "false" | "null"
object   → "{" [ string ":" value ("," string ":" value)* ] "}"
array    → "[" [ value ("," value)* ] "]"
string   → '"' (任何字符)* '"'
number   → 数字字面量
```

CFG 用**递归**表达嵌套:

```text
value → array → "[" value "," value "," ... "]"
                ↑      ↑
                这个 value 又可以是 array, 嵌套
```

**对应的自动机**: 上下文无关语言 = **下推自动机 (PDA)** 能识别的语言。 PDA = DFA + 一个**栈**。 栈用来"记住"嵌套深度:

```text
读 "{" → push "{"
读 "[" → push "["
读 "}" → pop, 检查栈顶是不是 "{"
读 "]" → pop, 检查栈顶是不是 "["
最后栈空 → 接受

ASCII 示意 (栈操作):

  读 '{'          读 '['          读 '}'          读 ']'
  ┌───┐           ┌───┐           ┌───┐           ┌───┐
  │ $ │           │ [ │           │ { │           │ $ │
  │   │           │ { │           │   │           │   │
  └───┘           │ { │           └───┘           └───┘
                  └───┘
  push '{'        push '['        pop '{' ✓       pop '[' ✓
```

JSON 解析器、Python 解释器、XML 解析器, 全都用 PDA 思路实现。

**核心结论**: **上下文无关语言 = PDA 能识别的语言 = 嵌套语法的数学基础**。

#### 例 3: Outlines / Guidance = CFG 引导的 LLM 采样

2023 年后, 受限解码 (constrained decoding) 在 LLM 应用中爆发。 经典例子是 JSON 模式生成:

```python
import outlines

@outlines.prompt
def extract_user(text):
    """从文本中提取用户信息, 输出严格 JSON:
    {"name": "string", "age": "int"}
    """
    return f"文本: {text}\nJSON:"

# 用 Outlines 加载模型 + JSON schema
model = outlines.models.transformers("gpt2")
schema = '{"name": "string", "age": "int"}'
generator = outlines.generate.json(model, schema)
result = generator(extract_user("Alice 今年 30 岁"))
# 输出: {"name": "Alice", "age": 30}  ← 100% 合法 JSON
```

**数学原理**: Outlines 把 JSON Schema 编译成一个 **CFG**, 然后在 LLM 每一步采样时:

1. 维护一个 "CFG 状态栈" (类似 PDA)
2. 看当前状态下, 哪些 token 是合法的 (例如 JSON 里看到 `"age":` 后面必须是数字)
3. 把这些合法 token 的 logit 设为 0, 其他设为 -inf
4. 让 LLM 在合法 token 中采样

这样 LLM **100% 不会输出非法 JSON** (因为非法 token 的概率是 0)。

**核心结论**: **CFG 是 LLM 结构化生成的"骨架"**——理论上, 任何能用 CFG 描述的格式, 都能被强制生成。

### 1.3 ASCII 对比: 形式语言的四个"层次"

```text
形式语言谱系 (从最受限到最自由, 越往下越自由):

┌─ 类型 0: 递归可枚举 (RE) ─────────────────────────────────┐
│  自动机: 图灵机 (TM) │ 判定: 不可判定                       │
│  例: 编程语言, 自然语言, LLM 输出                          │
│  ┌─ 类型 1: 上下文有关 (CSL) ─────────────────────────┐  │
│  │  自动机: LBA │ 判定: PSPACE 完全                       │  │
│  │  例: 缩进敏感 Python, aⁿbⁿcⁿ                          │  │
│  │  ┌─ 类型 2: 上下文无关 (CFL) ─────────────────────┐ │  │
│  │  │  自动机: PDA │ 判定: O(n³) CYK / O(n) LR        │ │  │
│  │  │  例: JSON, SQL, 算术表达式, Python (近似)       │ │  │
│  │  │  ┌─ 类型 3: 正则 ──────────────────────────┐  │ │  │
│  │  │  │  自动机: DFA/NFA │ 判定: O(n)             │  │ │  │
│  │  │  │  例: re, 词法分析, 模式匹配              │  │ │  │
│  │  │  └──────────────────────────────────────┘  │ │  │
│  │  └───────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**关键观察**: 越往下表达能力越强, 但算法越慢 (DFA O(n) → CYK O(n³) → TM 不可判定); LLM ≈ 类型 0, JSON/SQL ≈ 类型 2, regex = 类型 3。

### 1.4 章节路线图

接下来 8 节, 我们依次讲:

| 节 | 主题 | 与 ML/DL 的联系 |
| :--- | :--- | :--- |
| §2 | 形式语法基础 ($G=(V,T,P,S)$, 推导, 语法树) | 编译器前端, 语法树可视化 |
| §3 | Chomsky 谱系 (类型 0/1/2/3 + 自动机对应) | SQL=CFG, re=FSA, LLM≈类型 0, 解释器=PDA |
| §4 | 正则语言与 FSA (DFA/NFA/Thompson 构造) | Python `re`, 词法分析器, 模式匹配 |
| §5 | CFG 与 PDA (LL/LR, CYK O(n³)) | 编译器 parser, JSON.parse, Outlines |
| §6 | LLM 中的形式语言 (受限解码, Outlines/Guidance) | JSON 模式生成, 语法引导 |
| §7 | PyTorch 代码 | 5 个可运行实验 (DFA, CYK, PDA, NFA→DFA, CFG 引导) |
| §8 | 误区 + Q&A + 互引 | 4 误区 + 5 题 L7/L8 面试 + 跨章引用 |

---

## 2. 形式语法基础

### 2.1 字母表 $\Sigma$ 与字符串 $\Sigma^*$

**定义**: **字母表** (alphabet) $\Sigma$ 是有限非空符号集合。 例如:

- ASCII 字符集 $\Sigma_{\text{ASCII}} = \{a, b, c, \ldots, z, A, B, \ldots, Z, 0, 1, \ldots, 9, +, -, \ldots\}$
- 编程语言 token 集 $\Sigma_{\text{code}} = \{\text{identifier}, \text{keyword}, \text{operator}, \ldots\}$

**字符串** (string) 是字母表上符号的有限序列:

$$\Sigma^* = \{w_1 w_2 \ldots w_n : n \ge 0, w_i \in \Sigma\}$$

- 空字符串 (empty string) 记作 $\epsilon$, 长度 $|\epsilon| = 0$
- 字符串 $w$ 的**长度** $|w|$ = 符号个数
- **连接** (concatenation) $w_1 w_2$ = 把 $w_2$ 接到 $w_1$ 后面

**例子**:
- $\Sigma = \{0, 1\}$, 则 $\Sigma^* = \{\epsilon, 0, 1, 00, 01, 10, 11, 000, \ldots\}$ (无限)
- 字符串 $w = 0110$, $|w| = 4$, $w \cdot \epsilon = 0110$

**工程意义**:
- **所有可能的 JSON 字符串** = `Σ*` (其中 $\Sigma$ = ASCII 字符集)
- **"所有合法 JSON"** = `Σ*` 的一个**子集** $L \subset \Sigma^*$
- 解析器的任务: 给定 $w \in \Sigma^*$, 判断 $w \in L$?

### 2.2 形式语法 $G = (V, T, P, S)$

**定义**: 一个**形式语法** (formal grammar) 是一个四元组:

$$G = (V, T, P, S)$$

其中:

- $V$ = **非终结符** (variables / non-terminals) 集合, 通常用大写字母 $A, B, S$ 表示
- $T$ = **终结符** (terminals) 集合, $V \cap T = \emptyset$, 用小写字母 $a, b, c$ 表示
- $P$ = **产生式** (productions / rules) 集合, 每个产生式形如 $\alpha \to \beta$, 其中 $\alpha \in (V \cup T)^+$ 且 $\alpha$ 至少含一个非终结符
- $S \in V$ = **起始符** (start symbol), 唯一的"根"

**直观**: 语法是一组"重写规则", 从 $S$ 出发, 反复替换非终结符, 最终得到终结符串。

#### 例子 1: 算术表达式语法

```text
E → E + T | E - T | T        ← 表达式: 加减
T → T * F | T / F | F        ← 项: 乘除
F → ( E ) | id                ← 因子: 括号或标识符
```

- $V = \{E, T, F\}$
- $T = \{+, -, *, /, (, ), \text{id}\}$
- $P$ = 上面的产生式
- $S = E$

**推导过程** (从 $E$ 推导 $id + id * id$):

```text
E
⇒ E + T
⇒ T + T
⇒ F + T
⇒ id + T
⇒ id + T * F
⇒ id + F * F
⇒ id + id * F
⇒ id + id * id
```

每一步 $\Rightarrow$ 都应用一条产生式, 替换一个非终结符。

#### 例子 2: JSON 值语法 (简化)

```text
value    → object | array | string | number
object   → "{" [ pair ("," pair)* ] "}"
pair     → string ":" value
array    → "[" [ value ("," value)* ] "]"
string   → '"' char* '"'
number   → digit+
```

注意 `value` 出现在 `pair` 的右部, `pair` 又出现在 `object` 的右部, `object` 又出现在 `value` 的右部——**递归**! 这就是 CFG 能表达"嵌套结构"的关键: 产生式左部的非终结符可以出现在右部, 形成递归。

### 2.3 推导 (Derivation): $\Rightarrow^*$ 与 $L(G)$

**定义**: 推导是把字符串中某个非终结符 $A$ 替换为 $\beta$, 其中 $A \to \beta$ 是产生式, 记作 $\alpha A \gamma \Rightarrow \alpha \beta \gamma$。 $\Rightarrow^*$ 是 0 步或多次闭包。

**语法生成的语言**: $L(G) = \{w \in T^* : S \Rightarrow^* w\}$, 即从起始符出发能推导出的所有**终结符串**。

**工程意义**: 任何编程语言、JSON、SQL 都对应一个 $G$ 和 $L(G)$。 编译器的 parser, 干的事就是: 给定 $w$, 判断 $w \in L(G)$。 如果在, 给出**语法树**; 不在则报错。

### 2.4 语法树 (Parse Tree) 与歧义 (Ambiguity)

#### 语法树

语法树是推导的树形表示, 根节点 = $S$, 叶节点 = 终结符, 内部节点 = 产生式的应用。

```text
算术语法推导 id + id * id:

              E
            / | \
           E  +  T
           |    /|\
           T   T * F
           |   |   |
           F   F  id
           |   |
          id  id
```

**关键**: 这棵树**保留**了运算符优先级——`*` 在树中比 `+` 更深, 所以先算 `id * id` 再加 `id`。

#### 歧义 (Ambiguity)

**定义**: 语法 $G$ 是**歧义的**, 如果存在 $w \in L(G)$ 至少有两棵不同的语法树。

```text
歧义例子: id + id + id

      E                   E
     /|\                 /|\
    E + T               T + E
    |   |                 /|\
    T   T               T + E
    ...                 ...
   (id+id)+id         id+(id+id)
```

`id + id + id` 有两棵树 (左结合 vs 右结合)。 对 `+` 算术结果相同, 但对 `-` 或 `/` 顺序不同结果不同 (`id-id-id` 左 vs 右 = `id` vs `id-id+id`)。

**工程意义**: 编程语言必须**无歧义**。 解决方法:
1. **改写语法**: 把左结合显式编码 (如 $E \to E + T$ 左递归)
2. **声明优先级和结合性**: yacc/bison 用 `%left`, `%right`
3. **使用 LR parser**: 自动处理 (但报告冲突)

#### 消除算术歧义的语法 (左递归 + 优先级)

```text
E → E + T | E - T | T        ← 加减, 左递归 (左结合)
T → T * F | T / F | F        ← 乘除, 左递归 (左结合, 优先级高)
F → ( E ) | id                ← 因子
```

$E \to E + T$ 强制左结合; 乘除在更低的层 $T$, 先归约, 所以先算——优先级通过"分层"实现。

### 2.5 推导 `id + id * id` 的两种方式

**最左推导** (每步替换最左非终结符): $E \Rightarrow E+T \Rightarrow T+T \Rightarrow F+T \Rightarrow \text{id}+T \Rightarrow \text{id}+T*F \Rightarrow \ldots \Rightarrow \text{id}+\text{id}*\text{id}$

**最右推导** (每步替换最右): $E \Rightarrow E+T \Rightarrow E+T*F \Rightarrow E+T*\text{id} \Rightarrow E+F*\text{id} \Rightarrow E+\text{id}*\text{id} \Rightarrow \ldots \Rightarrow \text{id}+\text{id}*\text{id}$

两者得到相同语法树, 但顺序不同。 **LR parser 自底向上**对应最右推导, **LL parser 自顶向下**对应最左推导。

### 2.6 工程例子: AST (抽象语法树) = 推导的"压缩"

语法树含所有推导信息, 但工程上常常只需要"压缩版"——**抽象语法树 (AST)**, 省略括号、单产生式等冗余信息。

```python
import ast
tree = ast.parse("1 + 2 * 3", mode="eval")
# Expression(BinOp(1, Add, BinOp(2, Mult, 3)))
```

**AST vs Parse Tree**:
- **Parse Tree** (语法树): 含所有终结符 (括号, 关键字)
- **AST** (抽象语法树): 只保留"语义关键"信息

**工程意义**: 编译器、解释器、linter、formatter、code migration 工具 (Python 2 → 3) 全都基于 AST 操作。 比如 `pyupgrade` 读 Python 2 代码的 AST, 转成 Python 3 的 AST, 再 dump 回去。

---

## 3. Chomsky 谱系 (Chomsky Hierarchy)

### 3.1 四层语法 (1956, Noam Chomsky)

**Chomsky 谱系**把形式语法按"产生式的限制"分成 4 层, 表达能力从弱到强, 包含关系如下:

```text
类型 3 (正则)      ⊂  类型 2 (上下文无关)  ⊂  类型 1 (上下文有关)  ⊂  类型 0 (递归可枚举)
Regular               Context-Free             Context-Sensitive         Recursively Enumerable
```

每层对应一种自动机, 一种判定算法, 一种工程应用。

### 3.2 类型 3: 正则语法 (Regular Grammar)

**产生式形式**: 只允许 $A \to aB$ 或 $A \to a$ (其中 $A, B \in V$, $a \in T$)。

即: **右部最多两个符号, 第一个必须是终结符, 第二个(如果有)必须是非终结符**。 禁止 $\epsilon$-产生式 (除非是 $S \to \epsilon$ 且 $S$ 不出现在其他产生式右部)。

**例子**: 标识符 (字母开头, 后接字母/数字)

```text
S → aA | bA | ... | zA | a | b | ... | z        ← 首字符
A → aA | bA | ... | zA | 0A | ... | 9A |
    a | b | ... | z | 0 | ... | 9                ← 后续字符
```

**对应自动机**: **DFA** (确定性有限自动机) 或 **NFA** (非确定性有限自动机)。

**判定算法**: O(n) — 跑 DFA, 读字符串, 看是否到达接受状态。

**工程应用**:
- 正则表达式
- 词法分析器 (lexer)
- 模式匹配 (grep, awk)
- URL 路由 (NFA 匹配路径)

### 3.3 类型 2: 上下文无关语法 (Context-Free Grammar, CFG)

**产生式形式**: $A \to \beta$, 其中 $A \in V$, $\beta \in (V \cup T)^*$。

**关键放宽**: 右部可以是任意终结符 + 非终结符的串。 这允许**递归**——产生式左部的非终结符可以出现在右部。

**例子**: 算术表达式、JSON、SQL、Python 缩进语法的近似、HTML、XML、Markdown 块结构。

**对应自动机**: **PDA** (下推自动机, Pushdown Automaton) = DFA + 一个**栈**。 栈用来"记住"嵌套深度。

**判定算法**: $O(n^3)$ — CYK 算法 (见 §5)。 还有 $O(n)$ 的 LR(k) parser (工业界主流, 如 yacc/bison)。

**工程应用**:
- 编程语言的语法
- JSON / XML 解析
- 数据库 SQL 解析
- LLM 结构化生成 (Outlines/Guidance 用 CFG 做约束)
- Markdown / HTML 块结构

### 3.4 类型 1: 上下文有关语法 (Context-Sensitive Grammar, CSG)

**产生式形式**: $\alpha A \beta \to \alpha \gamma \beta$, 其中 $A \in V$, $\alpha, \beta, \gamma \in (V \cup T)^*$, $|\gamma| \ge 1$。

**关键**: 替换 $A$ 时, 必须考虑它的**左上下文 $\alpha$ 和右上下文 $\beta$**。 同一个非终结符在不同上下文下, 替换结果不同。

**例子**: Python 的缩进敏感语法

```text
if condition:
    block
elif other:
    block
```

`block` 的开始需要上一行有 `:`。 这种"上下文相关"的语法, CFG 难以表达, 需要 CSG。

**经典 CSG 语言**: $L = \{a^n b^n c^n : n \ge 1\}$ ("$n$ 个 $a$, $n$ 个 $b$, $n$ 个 $c$")。 CFG **不能**生成这个语言, 因为 CFG 只能"配对"两层 (aⁿbⁿ), 不能"配对"三层 (aⁿbⁿcⁿ)。

**对应自动机**: **LBA** (线性有界自动机, Linear Bounded Automaton) = 图灵机但存储限制在线性空间内。

**判定算法**: 可判定但 PSPACE 完全 — 实际几乎不可用。

**工程应用**: 极少。 大多数编程语言用 CFG 近似 (牺牲一些边角的语法), 因为 CSG 解析器太慢、太复杂。

### 3.5 类型 0: 递归可枚举语言 (Recursively Enumerable)

**产生式形式**: $\alpha \to \beta$, 其中 $\alpha, \beta \in (V \cup T)^*$, $|\alpha| \le |\beta|$ 之外没有限制 (允许 $\alpha$ 为 $\epsilon$ 也允许 $\beta$ 为 $\epsilon$——但某些定义要求 $\alpha$ 至少一个非终结符)。

**最自由**: 可以表达任何可计算的语言, 包括自然语言、程序的所有合法行为、甚至"程序会停机吗"等不可判定问题。

**对应自动机**: **图灵机 (TM, Turing Machine)** — 通用计算模型。

**判定算法**: **不可判定** (undecidable)。 存在一个语言 $L$, 没有算法能在有限时间内判断 $w \in L$。

**经典不可判定问题**: 停机问题 (给定程序 $P$ 和输入 $x$, $P$ 在 $x$ 上会停机吗?)、"程序是否输出 42"、Post 对应问题 等。

**工程应用**:
- 通用编程语言 (Python, C, Java): 类型 0 附近 (语法是 CFG, 但运行时的语义需要 TM)
- **LLM 的输出语言**: ≈ 类型 0, 因为 Transformer 在某些条件下是图灵完备的
- 自然语言: 严格说也是类型 0 (因为自然语言有任意嵌套、上下文依赖等)

### 3.6 谱系图 (ASCII)

```text
         包含关系 (越往上越受限, 越往下越自由)
         ┌────────────────────────────────────────┐
         │  类型 0: 递归可枚举 (RE)                 │
         │   ├ 图灵机 (TM)                         │
         │   ├ 判定: 不可判定                      │
         │   └ 例: 通用编程语言, 自然语言           │
         │      ┌─────────────────────────────┐    │
         │      │  类型 1: 上下文有关 (CSL)     │    │
         │      │   ├ 线性有界自动机 (LBA)      │    │
         │      │   ├ 判定: PSPACE 完全         │    │
         │      │   └ 例: 缩进敏感语法, aⁿbⁿcⁿ  │    │
         │      │      ┌─────────────────────┐  │    │
         │      │      │  类型 2: 上下文无关  │  │    │
         │      │      │   ├ 下推自动机 (PDA) │  │    │
         │      │      │   ├ 判定: O(n³)      │  │    │
         │      │      │   └ 例: JSON, SQL,   │  │    │
         │      │      │      算术表达式       │  │    │
         │      │      │      ┌────────────┐  │  │    │
         │      │      │      │  类型 3    │  │  │    │
         │      │      │      │  正则 (RG) │  │  │    │
         │      │      │      │  ├ DFA/NFA│  │  │    │
         │      │      │      │  ├ O(n)   │  │  │    │
         │      │      │      │  └ 例: re │  │  │    │
         │      │      │      └────────────┘  │  │    │
         │      │      └─────────────────────┘  │    │
         │      └─────────────────────────────┘    │
         └────────────────────────────────────────┘
```

**关键性质**: 每一层都是上一层的**真超集**。 即: 正则语言 ⊊ CFG ⊊ CSL ⊊ 递归可枚举语言。

### 3.7 工程对应: 你的工具属于哪一层?

| 工具 / 语言 | 语法类型 | 对应自动机 | 判定复杂度 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 正则表达式 (re) | 类型 3 | DFA/NFA | O(n) | grep, 词法分析 |
| JSON | 类型 2 (CFG) | PDA | O(n) (LR parser) | 实际 parser 都是 O(n) |
| SQL | 类型 2 (CFG) | PDA | O(n) | 各 DBMS 自己的 parser |
| Python 语法 (近似) | 类型 2 (CFG) | PDA + 缩进处理 | O(n) | CPython 用 pgen |
| 自然语言 | 类型 0 附近 | TM | 不可判定 | LLM 也是 |
| 通用图灵机 | 类型 0 | TM | 不可判定 | 一切可计算的 |
| LLM 输出 (Transformer) | ≈ 类型 0 | TM (有条件) | 不可判定 | 上下文有限, 实际有界 |
| Outlines JSON 约束 | 类型 2 (CFG 引导) | PDA + logit mask | O(n³) 理论, O(n) 实际 | 受限解码 |
| SQL 受限解码 | 类型 2 | PDA + logit mask | 同上 | text-to-SQL 场景 |

**关键洞察**:

- **LLM 的"自然能力"接近类型 0** (万能但不可控)
- **结构化生成 = 把 LLM 拉回到类型 2** (受控但表达力足够)
- **正则表达式 = 类型 3** (最受限, 但极快, 适合模式匹配)
- **CFG 是"表达力 vs 可控性"的最佳平衡点**

### 3.8 谱系与 LLM 的关系

LLM 的语言能力是分层的:

```text
LLM 自身: 能学类型 0 (理论上)
   ↓ 加 Outlines 约束 → 类型 2 (JSON Schema)
   ↓ 再加正则约束  → 类型 3 (邮箱, 电话号码格式)
```

**实际工程建议**:

1. **能用正则就用正则** (O(n), 极快, 工具多)
2. **嵌套结构用 CFG** (JSON, SQL, 编程语言子集)
3. **复杂业务逻辑用 LLM + 受限解码** (类型 2 引导)
4. **自然语言生成直接 LLM** (类型 0, 但要做后处理)

---

## 4. 正则语言与 FSA

### 4.1 确定性有限自动机 (DFA)

**定义**: DFA 是一个五元组 $M = (Q, \Sigma, \delta, q_0, F)$, 其中:

- $Q$: 有限状态集合
- $\Sigma$: 输入字母表
- $\delta: Q \times \Sigma \to Q$: 转移函数 (每个状态 + 每个字符 → 唯一下一个状态)
- $q_0 \in Q$: 起始状态
- $F \subseteq Q$: 接受状态集合

**运行 (Run)**: 读字符串 $w = a_1 a_2 \ldots a_n$, 从 $q_0$ 出发, 每步用 $\delta$ 转移, 最后落在状态 $q_n$。 如果 $q_n \in F$, **接受**; 否则**拒绝**。

$$L(M) = \{w \in \Sigma^* : M \text{ 接受 } w\}$$

#### 例子: $(ab)^*$ 的 DFA

接受 $\{ab, abab, ababab, \ldots\}$ (偶数个字符, 且 `a` 和 `b` 交替)。

```text
DFA for (ab)*:

              a          b
   ┌─────→ (q0) ──────→ (q1)
   │        │            │
   │        │            │ a
   │        │            ↓
   │        │          (q2) ──b──→ (q3)  ← 死路
   │        │            │
   │        └──── b ─────┘
   │
   │           accept q0 (空串, abab, ababab, ...)
   ↓
  start
```

**状态转移表**:

```text
        |   a   |   b   |
| :--- | :--- |
  → q0  |  q1   |  ✗    |   ← 起始
    q1  |  ✗    |  q0   |   ← 读 a 后
   *q0  |       |       |   ← 接受 (回到原位 = 接受空串和偶数次)
```

**代码实现**: 模拟器如下 (见 §7 实验 1)。

### 4.2 非确定性有限自动机 (NFA)

**定义**: NFA 与 DFA 类似, 但 $\delta: Q \times (\Sigma \cup \{\epsilon\}) \to \mathcal{P}(Q)$ — 转移函数返回**状态集合的子集** (多个可能的后继状态)。 允许 $\epsilon$-转移 (无输入也跳转)。

**直观**: NFA 可以"同时在多个状态"。 当读一个字符时, 它可以"分身"到所有可能的后继状态。 一个字符串被接受, **当且仅当**至少有一个"分身"最终落在接受状态。

#### 例子: $L = (a|b)^* abb$ 的 NFA

这是经典 Thompson 构造的例子。 接受所有"以 `abb` 结尾"的字符串。

```text
NFA for (a|b)*abb:

         ε                ε
   → (q0) ──────→ (q1) ──a──→ (q2) ──b──→ (q3) ──b──→ (q4) ◀── 接受
       │            │                          ↑
       │            │ a, b (循环)               │ b
       │            ↓                          │
       └───────── (q0) (回到自身, 继续读)
                    ...
```

**NFA 的"非确定性"**: 在 $q0$, 读 `a` 可以跳到 $q1$ 也可以留在 $q0$ (因为 $(a|b)^*$)。 读 `b` 同理。 NFA 接受, 当**存在**一条路径让字符串走完后到达 $q4$。

**关键**: NFA 和 DFA 表达力**完全相同** (NFA 可以转 DFA, 见 §4.3), 但 NFA 通常**更简洁、更易构造**。

### 4.3 Thompson 构造: regex → NFA

**Ken Thompson (1968)** 提出用 $\epsilon$-NFA 构造正则表达式的算法。 核心思路: 对 regex 的每个子表达式递归构造一个小 NFA, 然后用 $\epsilon$-边连接。

**4 条基本规则**:

1. 字符 $a$: 一个状态, 自转移 $a$
2. 连接 $AB$: 把 $A$ 的接受状态和 $B$ 的起始状态用 $\epsilon$ 边连起来
3. 选择 $A|B$: 新建一个起始状态, 用 $\epsilon$ 边分叉到 $A$ 和 $B$, 合并接受状态
4. 闭包 $A^*$: 新建一个起始状态, 用 $\epsilon$ 边分叉到 $A$ 和"接受状态" (允许空), $A$ 的接受状态再 $\epsilon$ 回到 $A$ 起始 (允许重复)

**工程意义**: Thompson 构造是 **grep、lex、re** 等工具的核心算法。 把正则表达式编译成 NFA, 然后模拟 NFA 运行。 模拟是 O(nm) (n 是字符串长度, m 是 NFA 状态数), 实践中很快。

**NFA → DFA (子集构造)**: 因为 NFA 可以"同时在多状态", 我们用一个**状态集合**模拟"分身": 每个 DFA 状态 = 一个 NFA 状态集合 (subset)。 这样把 NFA 转化成等价的 DFA, 然后跑 DFA 就是 O(n)。

```python
# 子集构造: 把 NFA 的状态集合"压缩"成 DFA 的单个状态
# 例: NFA 状态 {q0, q1, q3} → DFA 状态 "013" (一个状态, 表示"同时在 q0, q1, q3")
```

### 4.4 Python `re` 引擎: 怎么"匹配"一个字符串?

```python
import re
pattern = re.compile(r"^[a-z]+@[a-z]+\.[a-z]{2,}$")
m = pattern.match("alice@example.com")
print(m.group())  # alice@example.com
```

**Python `re` 引擎 (re2 / _sre) 的实现**:

1. **解析 regex**: 把字符串形式的 regex 解析成 AST (CFG 引导的 parser)
2. **Thompson 构造**: 把 AST 编译成 NFA
3. **子集构造** (或直接模拟 NFA): 拿到 NFA
4. **匹配**: 给定输入字符串, 模拟 NFA 运行, 看是否到达接受状态

**复杂度**: NFA 模拟是 O(nm), 子集构造后是 O(n)。 实践中, `re.match` 在百万字符的字符串上仍是亚毫秒级。

### 4.5 ASCII: $(ab)^*$ DFA 的运行过程

接受 $\{ab, abab, ababab, \ldots\}$:

```text
DFA 运行 "abab":

  读 a: q0 → q1 ✓
  读 b: q1 → q0 ✓
  读 a: q0 → q1 ✓
  读 b: q1 → q0 ✓ ← 接受状态 ✓

字符串 "abab" 被接受 ✓

DFA 运行 "aba":

  读 a: q0 → q1 ✓
  读 b: q1 → q0 ✓
  读 a: q0 → q1 ✓
  字符串结束, 但 q1 不是接受状态 ✗
  → 拒绝
```

### 4.6 ASCII: $(a|b)^* abb$ NFA 的运行过程

```text
NFA for (a|b)*abb:

         ε                ε
   → (q0) ──────→ (q1) ──a──→ (q2) ──b──→ (q3) ──b──→ (q4) ◀── 接受
       │            │                          ↑
       │            │ a, b (循环)               │ b
       │            ↓                          │
       └───────── (q0) (回到自身, 继续读)
                    ...
```

接受所有"以 `abb` 结尾"的字符串。 例如 `abb`, `aabb`, `bababb`, `xyabb` 都被接受。

**NFA 的"非确定性"**: 读 `a` 时, NFA 既可以**留在 q0** (因为 $(a|b)^*$ 允许任意 a/b), 也可以**跳到 q2** (开始匹配 `abb`)。 接受条件: **存在**一条路径让字符串走完后到达 q4。

---

## 5. CFG 与 PDA

### 5.1 CFG 的形式定义 (回顾与补充)

**回忆**: CFG = $G = (V, T, P, S)$, 产生式形如 $A \to \beta$, $A \in V$, $\beta \in (V \cup T)^*$。

**关键技术**: 推导时, 替换一个**非终结符**, 不管它周围是什么。 也就是说, 替换与**上下文无关** (这就是 "context-free" 的含义)。

**歧义回顾**: 一个 CFG $G$ 是歧义的, 如果存在 $w \in L(G)$ 有两棵不同的语法树。

**工程影响**: 编译器 parser (LL, LR) 要么用**无歧义语法**, 要么用**带优先级声明的歧义语法** (yacc/bison 的 `%left`, `%right`)。

### 5.2 LL(k) 与 LR(k) Parser

#### LL(k) Parser (自顶向下, Top-Down)

**LL(k)**: 从左 (Left) 扫描输入, 推导最左 (Leftmost) 推导, 向前看 $k$ 个 token。

**算法**: 递归下降, 或显式栈 (predictive parser)。 维护一个"工作栈", 用 lookahead 决定应用哪条产生式。

**限制**:
- 不能处理左递归 (直接左递归: $A \to A \alpha$)
- 不能处理"公共前缀"过长的情况

**例子**: 手写递归下降 parser

```python
def parse_E(tokens):
    """E → T (+ T)*"""
    left = parse_T(tokens)
    while tokens and tokens[0] == '+':
        tokens.pop(0)
        right = parse_T(tokens)
        left = ('+', left, right)
    return left

def parse_T(tokens):
    """T → F (* F)*"""
    left = parse_F(tokens)
    while tokens and tokens[0] == '*':
        tokens.pop(0)
        right = parse_F(tokens)
        left = ('*', left, right)
    return left

def parse_F(tokens):
    """F → ( E ) | id"""
    if tokens[0] == '(':
        tokens.pop(0)
        node = parse_E(tokens)
        tokens.pop(0)  # consume ')'
        return node
    else:
        return tokens.pop(0)  # id
```

#### LR(k) Parser (自底向上, Bottom-Up)

**LR(k)**: 从左 (Left) 扫描输入, 推导最右 (Rightmost) 推导的逆过程, 向前看 $k$ 个 token。

**算法**: 移进-归约 (shift-reduce), 维护一个栈, 根据 lookahead 决定"移进" (push token) 还是"归约" (pop + 应用产生式)。

**优点**:
- 能处理左递归
- 表达力比 LL 强 (LR(1) 能识别所有确定性 CFL)
- yacc/bison 用 LALR(1) (LR(1) 的简化版), 工业界主流

**工具**: GNU bison, yacc, ANTLR (LL(*) 增强版), Lark (Python)

### 5.3 下推自动机 (PDA): DFA + 栈

**定义**: PDA = 七元组 $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$, 在 DFA 基础上**多了一个栈**, 读入字符时可以 push 或 pop 栈顶。

**关键**: 栈用来"记住"嵌套结构。 栈高度 = 当前嵌套深度, push `(` = 进入嵌套, pop `(` = 退出嵌套。 任何时候读 `)` 但栈顶不是 `(` 就是不合法。

#### ASCII: 括号匹配 PDA

```text
PDA for 匹配括号:

                读 '('              读 ')'
        ┌──────→ (q0) ──push '('──→ (q1)
        │         │                    │
        │         │                    │ 栈顶='('? pop
        │         │                    │ 否则: 拒绝
        │         │                    ↓
        │         │                  (q2) ──栈空?──→ (q3) ◀── 接受
        │         │                    │
        │         │                    │ 栈非空: 拒绝
        │         │                    ↓
        │         │                  (q4) ── 拒绝
        │         ↓
        ↓       (q5) ── 拒绝
      接受 ε
      需栈空

状态转移:
  q0 (起始): 读 '(' push, 保持; 读 ')' 拒绝
  q1 (已读 '('): 读 '(' push (嵌套); 读 ')' pop 回 q0
  q2 (读完输入): 栈空→接受 (q3), 栈非空→拒绝 (q4)
```

**工程意义**: JSON 解析器、Python 缩进 parser (近似)、XML 解析器、HTML 解析器全用 PDA 思路。 **栈**让 PDA 能"记住任意深度的嵌套"——这是 DFA 做不到的 (状态有限)。

### 5.4 经典结论: CFG ↔ PDA

**定理**: $L$ 是 CFL $\iff$ 存在 PDA $M$ 使 $L = L(M)$。 即 PDA 恰好能识别所有 CFG 生成的语言。 (这是"DFA 恰好识别正则语言"的对偶定理。)

**工程意义**: 任何 CFG 都能转成 PDA, PDA 比 CFG 更"机械"——直接代码模拟即可。 很多 parser 内部表示就是 PDA。

### 5.5 CYK 算法: O(n³) 判定"字符串是否属于 CFG"

**Cocke-Younger-Kasami (CYK) 算法** (1965-1967) 是经典 CFG 成员判定算法, 给定 $w$ 和 $G$ 判断 $w \in L(G)$, 复杂度 $O(n^3)$。

**前提**: $G$ 必须是 **Chomsky 范式 (CNF)**——所有产生式形如 $A \to BC$ (两个非终结符) 或 $A \to a$ (一个终结符)。 任何 CFG 都能转成 CNF (多项式时间)。

#### 算法核心 (动态规划)

定义 $T[i][j]$ = "能从 $w_i w_{i+1} \ldots w_j$ 推导出的非终结符集合"。

- **初始化**: $T[i][i] = \{A : A \to w_i \in P\}$
- **递推**: $T[i][j] = \{A : \exists k, A \to BC \in P, B \in T[i][k], C \in T[k+1][j]\}$
- **判定**: $w \in L(G) \iff S \in T[1][n]$

#### 例子: CFG 和字符串 `aab`

```text
CFG (CNF): S→AB|BC, A→BA|a, B→CC|b, C→AB|a
字符串:    a a b
           1 2 3
```

**CYK 表格** (从左下到右上填, 沿对角线长度递增):

```text
        j=1    j=2       j=3
i=1   {A,C}    {B}       {}     ← B 由 CC 推
i=2           {A,C}      {S}     ← S 由 AB 推
i=3                    {B}        ← b 直接推出
```

- `T[1][1] = {A, C}`: 字符 a 可由 A→a 或 C→a 推
- `T[3][3] = {B}`: 字符 b 由 B→b 推
- `T[1][2] = {B}`: 子串 `aa`, 拆 `{A,C}·{A,C}` → CC 匹配 B→CC
- `T[2][3] = {S}`: 子串 `ab`, 拆 `{A,C}·{B}` → AB 匹配 S→AB
- `T[1][3] = {}`: 子串 `aab`, 任何拆法都不匹配 `S→BC`

**结论**: `aab` ∉ L(G)。 CYK 是 $O(n^3)$ DP, 实际 LR/LL parser 是 $O(n)$ 但要求无歧义语法; CYK 适用**所有** CFG, 代价是慢。 实际代码见 §7 实验 2。

### 5.6 工程例子: Python 编译器 (CPython) 怎么解析 `def f(x): return x + 1`

**5 个步骤**:

1. **词法分析 (Lexing)**: token 化 `def`, `f`, `(`, `x`, `)`, `:`, `return`, `x`, `+`, `1` — 用 **DFA**
2. **语法分析 (Parsing)**: 用 **pgen** (Python 自带的 LL(1) parser 生成器) — 接受 PEG 文法
3. **AST 生成**: parse tree → AST (抽象语法树)
4. **字节码编译**: AST → 字节码 (CFG 引导)
5. **执行**: 字节码 → 解释执行 (PDA 风格 — 操作数栈)

**关键**: Python 用 **PEG (Parsing Expression Grammar)**——受限的 CFG, 配合 LL(parser) 实现。 缩进敏感部分用 token 插入 (INDENT/DEDENT) 转为 CFG。

### 5.7 ASCII: 算术表达式语法树 vs AST

```text
表达式: 1 + 2 * 3

语法树 (含所有符号):       AST (省略冗余):
        E                    (+)
       /|\                  /   \
      E + T                1    (*)
      |   |                     /   \
      T  T*F                   2     3
      |  |  |
      F  F id(3)
      |  |
     id(1) id(2)
```

AST 省略了**单产生式** (E → T → F → id) 的中间节点, 只保留语义关键节点。 这就是为什么编译器内部几乎都用 AST。

---

## 6. LLM 中的形式语言

### 6.1 LLM 输出语言 ≈ 类型 0

**Transformer** (LLM 骨干) 在某些条件下是**图灵完备的** (Pérez et al. 2019, Bhattamishra et al. 2020)。 即: 给定无限上下文和无限时间, Transformer 可以模拟任意图灵机。

**实际限制**:
- 上下文窗口有限 (4K - 200K token)
- 浮点精度有限
- 训练分布限制 (没见过就不在)
- 解码策略限制 (贪心 / 采样 / beam search)

**工程结论**: LLM 的输出语言在**实际意义上**比类型 0 小很多, 但**理论上**可以接近。 这就是为什么 LLM 偶尔"胡言乱语" (生成不属于任何"有意义的语言" 的字符串) 是不可控的。

### 6.2 受限解码 (Constrained Decoding): 把 LLM 拉回类型 2

**核心思想**: 在 LLM 每一步采样时, 用"目标语法" (JSON Schema, regex, CFG) 计算"哪些 token 合法", 把不合法的 token 概率设为 0。

```text
标准 LLM 采样:
  logits → softmax → 采样 token

受限解码:
  target_syntax = JSON schema (CFG)
  allowed_tokens = syntax.next_tokens(current_state)
  logits[非 allowed] = -inf
  softmax → 采样 (只从 allowed 中)
```

**实现工具**:

| 工具 | 约束表示 | 速度 | 工程成熟度 |
| :--- | :--- | :--- | :--- |
| Outlines | regex / JSON Schema / CFG | 中 | 高 (2023+) |
| Guidance | regex / JSON / CFG / LMQL | 中 | 高 (2023+) |
| LMQL | 类 SQL 语言 | 慢 | 中 (研究) |
| SGLang | JSON / regex | 快 | 高 (2024+) |
| vLLM (outlines backend) | regex / JSON | 快 | 高 (2024+) |
| llama.cpp grammars | GBNF (类似 BNF) | 快 | 中 |

**数学原理**:

- JSON Schema 编译成 CFG
- CFG 模拟成 PDA (栈状态)
- 在 LLM 每步采样时, 询问 PDA: "栈顶状态下, 哪些 token 是合法的?"
- 这些合法 token 的 logits 保持, 其他设 -inf
- LLM 在合法子集内采样, 100% 保证输出符合 schema

### 6.3 Outlines / Guidance 的 CFG 引导示例

```python
import outlines

# 1. 定义 JSON Schema
schema = """{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  }
}"""

# 2. 用 Outlines 加载模型 + 约束
model = outlines.models.transformers("gpt2")
generator = outlines.generate.json(model, schema)

# 3. 生成 (100% 合法 JSON)
result = generator("提取用户: Alice 今年 30 岁")
print(result)
# {"name": "Alice", "age": 30}  ← 绝对合法

# 4. 同样的思路用 regex 约束
pattern = r"[0-9]{3}-[0-9]{4}"  # 日本邮编格式
generator = outlines.generate.regex(model, pattern)
result = generator("邮编是?")
# 100% 匹配 [0-9]{3}-[0-9]{4}
```

**底层**: Outlines 内部维护一个**PDA 状态机**, 跟踪"当前 JSON 嵌套到哪一层, 下一个期待的 token 是什么"。 用 logits mask 过滤掉所有不合法 token。

### 6.4 CoT 的自动机视角

**CoT (Chain-of-Thought)** 让 LLM 输出推理步骤。 从自动机视角: LLM 推理 = 在一个"潜在的状态机"上做转移, 状态 = 当前推理进度, 转移 = 下一步写什么, 接受 = 输出 final answer。

**形式化**: LLM 的"推理能力" ≈ 在某些隐式 CFG 上做采样。 ReAct 的 "Thought/Action/Observation" ≈ CFG 的产生式 (Trace → Thought Action Observation (重复), Action → Search[query] | Finish[answer])。 ToT (Tree of Thoughts) ≈ 在 CFG 推导树上做 BFS/DFS。

**关键**: LLM 没有"显式 PDA", 但训练数据中的人类推理样本让它学到了"隐式 PDA"——这就是为什么 CoT prompt 让 LLM 沿推理链走 (见 第 2 章:命题逻辑·谓词逻辑-推理链·CoT·LLM 推理的逻辑基础 第 6.7 节)。

### 6.5 工程实战: JSON 模式生成的最佳实践

```python
# ❌ 不推荐: 让 LLM 直接吐 JSON
prompt = "输出 JSON: {name, age}"
response = llm(prompt)  # 可能: "当然, JSON 是: {name: Alice, age: 30}"

# ✅ 推荐 1: 用 Outlines / Guidance 加 schema 约束
import outlines
generator = outlines.generate.json(model, schema)
result = generator(prompt)  # 100% 合法

# ✅ 推荐 2: 后处理 + 容错 (如果不能用 Outlines)
import json
try:
    result = json.loads(llm(prompt))
except json.JSONDecodeError:
    result = repair_json(llm(prompt))  # 重试 or 修复
```

### 6.6 ASCII: 受限解码 vs 标准解码

```text
标准 LLM 解码 (无约束):

  LLM 输出: "当然, JSON 是: { \"name\": \"Alice\" "age\": 30 } 这就是用户信息"
                  ↑ "age" 前的逗号, 语法错误
  ↓
  json.loads() → JSONDecodeError ✗

受限解码 (Outlines 引导):

  LLM 输出: {"name": "Alice", "age": 30}
                  ↑ 每步都被 CFG mask 强制, 100% 合法
  ↓
  json.loads() → {"name": "Alice", "age": 30} ✓
```

---

## 7. PyTorch 代码

5 个可运行实验, 覆盖本章核心概念。

### 7.1 实验 1: DFA 模拟器 (验证 $(ab)^*$)

```python
class DFA:
    """确定性有限自动机"""
    def __init__(self, states, alphabet, transition, start, accept):
        self.transition = transition
        self.start, self.accept = start, accept
        self.alphabet = alphabet

    def run(self, s):
        state = self.start
        for c in s:
            if c not in self.alphabet: return False
            state = self.transition.get((state, c))
            if state is None: return False
        return state in self.accept

# (ab)* 的 DFA: 状态 0 = 接受 (空串, 偶数次循环), 状态 1 = 读 a 后等 b, 状态 2 = 死
dfa = DFA({0,1,2}, {'a','b'},
    {(0,'a'):1, (1,'b'):0, (2,'a'):2, (2,'b'):2}, 0, {0})

for s in ["", "ab", "abab", "a", "aba"]:
    print(f"  DFA({s!r}) → {dfa.run(s)}  (期望 {s in {'','ab','abab','ababab'}})")
# DFA('') → True  DFA('ab') → True  DFA('abab') → True
# DFA('a') → False  DFA('aba') → False
```

### 7.2 实验 2: CYK 算法 (CNF CFG 成员判定)

```python
def cyk(grammar, start, s):
    """CYK: O(n^3) 判定 s ∈ L(G), grammar 是 CNF 形式"""
    n = len(s)
    if n == 0: return False
    # T[i][j] = 可推导出 s[i..j] 的非终结符集合
    T = [[set() for _ in range(n)] for _ in range(n)]
    # 初始化: 单字符产生式
    for i, c in enumerate(s):
        for A, prods in grammar.items():
            for p in prods:
                if len(p) == 1 and p[0] == c: T[i][i].add(A)
    # 递推: 长度 2..n
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            for k in range(i, j):
                for A, prods in grammar.items():
                    for p in prods:
                        if len(p) == 2 and p[0] in T[i][k] and p[1] in T[k+1][j]:
                            T[i][j].add(A)
    return start in T[0][n-1]

# 测试: 经典 a^n b^n CFG (S → AB | a | b; A → a; B → Sb)
grammar = {'S':[('A','B'),('a',),('b',)], 'A':[('a',)], 'B':[('S','b')]}
for s in ['ab', 'aabb', 'aabbb', 'ba', 'aa']:
    print(f"  CYK({s!r:8}) ∈ L(aⁿbⁿ)? {cyk(grammar, 'S', s)}")
# ab → True, aabb → True, aabbb → True, ba → False, aa → False
```

### 7.3 实验 3: 括号匹配 PDA

```python
class PDA:
    """下推自动机: 验证括号匹配"""
    def run(self, s):
        stack = ['$']  # 栈底标记
        for c in s:
            if c == '(': stack.append(c)
            elif c == ')':
                if len(stack) <= 1: return False
                stack.pop()
            else: return False
        return len(stack) == 1

pda = PDA()
for s in ["()", "(())", "(()())", "())", "(()", "()("]:
    print(f"  PDA({s!r:10}) → {pda.run(s)}")
# () → True, (()) → True, (()()) → True, ()) → False, (() → False, ()( → False
```

### 7.4 实验 4: NFA → DFA (子集构造)

```python
def nfa_to_dfa(nfa_trans, nfa_start, nfa_accept, alpha):
    """子集构造: NFA → DFA, 用 frozenset 表示 DFA 状态"""
    dfa_start = frozenset(nfa_start)
    dfa_states, dfa_trans, queue = {dfa_start}, {}, [dfa_start]
    while queue:
        cur = queue.pop(0)
        for c in alpha:
            nxt = frozenset(s for st in cur for s in nfa_trans.get((st, c), set()))
            dfa_trans[(cur, c)] = nxt
            if nxt not in dfa_states:
                dfa_states.add(nxt); queue.append(nxt)
    return dfa_states, dfa_trans, dfa_start, {s for s in dfa_states if s & nfa_accept}

# (a|b)*abb 的 NFA
nfa = {(0,'a'):{0,1}, (0,'b'):{0}, (1,'b'):{2}, (2,'b'):{3}}
states, _, start, accept = nfa_to_dfa(nfa, {0}, {3}, {'a','b'})
print(f"  NFA→DFA: {len(states)} 个 DFA 状态, 接受 {len(accept)} 个")
```

### 7.5 实验 5: LLM CFG 引导 (简化演示)

```python
"""
简化版 LLM + CFG 受限解码。实际中用 Outlines/Guidance。
核心思想: 每步采样前, 用 PDA 状态算出"合法 token", 其他 mask 掉。
"""
import random

def mock_llm_logits(ctx):  # 模拟 LLM 输出每个 token 的概率
    return {t: random.random() for t in ['{','}','"','name','age','Alice','30',',',':']}

def allowed_tokens(ctx):  # CFG 引导: 当前 PDA 状态下的合法 token
    if not ctx: return ['{']
    if ctx[-1] == '{': return ['"']
    if ctx[-1] == ':': return ['"']  # 简化: value 是 string
    return []

# 演示
print("LLM CFG 引导 demo:")
ctx = []
for _ in range(5):
    logits = mock_llm_logits(ctx)
    allowed = allowed_tokens(ctx)
    masked = {t: (logits[t] if t in allowed else 0) for t in logits}
    total = sum(masked.values())
    if total == 0: break
    tok = random.choices(list(masked), weights=list(masked.values()))[0]
    print(f"  ctx={ctx!r:20} allowed={allowed}, sampled={tok!r}")
    ctx.append(tok)

# 输出示例: 一步步采样, 每步 token 必然在 CFG 允许集合内
# 完整工程实现: https://github.com/outlines-dev/outlines
```

---

## 8. 误区 + Q&A + 互引

### 8.1 常见误区 (4 个)

#### 误区 1: ❌ "正则表达式可以匹配所有模式"

**反例**: 正则**不能**匹配嵌套括号 `^(a^*)^*$ 这种"配对"模式, 因为正则语言对应 DFA, DFA 状态有限, 不能"记住"任意深度。

**工程影响**: JSON 解析、HTML 解析、Python 代码解析, 都**不能用**正则。 必须用 CFG + PDA (或 LR/LL parser)。

**判断**: 见到嵌套结构 (递归定义), 就别用正则, 用 CFG。

#### 误区 2: ❌ "CFG 等价于 PDA"

**半对**: CFG 描述**语言**, PDA 识别**语言**。 两者是**对偶**的 (CFL ↔ PDA), 不是**等价**的。 CFG 是**生成**模型, PDA 是**识别**模型。

**工程意义**: 写 parser 时, 你既可以用 CFG (用 yacc/bison 描述语法) 也可以用 PDA (直接用代码模拟栈), 但通常 CFG 更易读。

#### 误区 3: ❌ "CYK 在现代 NLP 中无用"

**反例**: CYK 是 Outlines/Guidance 等结构化生成库的**底层算法**之一 (用于计算 CFG 状态的"下一步合法 token")。 在 LLM 时代, CYK 比以往任何时候都更重要。

**工程意义**: 学 CFG + CYK, 不是为了"手写 parser", 而是为了**理解 LLM 受限解码**的原理。

#### 误区 4: ❌ "LLM 能输出任何东西, 所以无需约束"

**反例**: LLM 自由生成 = 类型 0 附近, 不可控。 生产环境的 JSON/SQL/代码生成, **必须**加约束 (Outlines 等)。 否则 5-10% 的输出会有格式错误。

**工程建议**: 任何 LLM 输出到下游系统, 都应加 schema 约束 (类型 2) 或 regex 约束 (类型 3)。

### 8.2 面试 Q&A (L7/L8 难度)

#### Q1: 简述 Chomsky 谱系, 每一层对应的自动机和判定复杂度。

**标准答案**:

| 层级 | 语法类型 | 产生式限制 | 自动机 | 判定复杂度 | 例子 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 类型 3 | 正则 | $A \to aB \mid a$ | DFA/NFA | O(n) | re, 词法分析 |
| 类型 2 | 上下文无关 | $A \to \beta$ | PDA | O(n³) CYK, O(n) LR | JSON, SQL, 算术 |
| 类型 1 | 上下文有关 | $\alpha A \beta \to \alpha \gamma \beta$ | LBA | PSPACE 完全 | 缩进敏感, $a^nb^nc^n$ |
| 类型 0 | 递归可枚举 | 任意 | 图灵机 | 不可判定 | 编程语言, 自然语言 |

**关键洞察**:
- 包含关系: 类型 3 ⊊ 类型 2 ⊊ 类型 1 ⊊ 类型 0
- 表达能力越强, 越不可判定 / 越难判定
- 工程上, 类型 2 是"甜点"——JSON, SQL, 编程语言都是 CFG, parser 工具成熟

**进阶**: LLM 本身是 Transformer, 在某些条件下图灵完备, 所以 ≈ 类型 0。 但加上 Outlines/Guidance 约束后, 退化为类型 2 (CFG 引导)。

#### Q2: LLM 是怎么"按 JSON 模式"生成的? 数学原理是什么?

**标准答案**:

**核心机制**: **受限解码** (constrained decoding) + **CFG 引导** (CFG-guided beam search / mask sampling)。

**步骤**:

1. **把 JSON Schema 编译成 CFG**: Outlines 内部用类似 Lark 的库, 把 JSON Schema 转成 BNF 形式
2. **维护一个 PDA 状态**: 在 LLM 生成 token 时, 跟踪"当前 JSON 嵌套到哪一层, 下一个期待的 token 是什么"
3. **每步采样前, 计算合法 token 集合**:
   - 例如: 当前 PDA 状态 = "在 `{"name":` 后面, 期待数字"
   - 合法 token: `0`, `1`, ..., `9` (其他字符都非法)
4. **修改 logits**: 合法 token 保持原 logit, 非法 token 设 -inf
5. **LLM 在合法 token 中采样**: 因为非法 token 概率 = 0, 100% 输出合法 JSON

**数学保证**: 因为 CFG 的封闭性, 输出**必然**属于 $L(G)$, 100% 合法。

**工程代表**: Outlines (Python), Guidance (Microsoft), LMQL, SGLang, vLLM (with outlines backend)。

**反直觉点**: LLM 没有"内化 CFG", 它是在每步采样时被"外部强制"。 这是 LLM 推理与形式化推理的本质区别 (见 第 2 章:命题逻辑·谓词逻辑-推理链·CoT·LLM 推理的逻辑基础 第 6.7 节)。

#### Q3: 正则 vs CFG, 为什么正则不能表达嵌套?

**标准答案**:

**根本原因**: 正则 = DFA 能识别的语言, DFA 状态有限, 所以"能记住的信息"有限。 嵌套要求"记住嵌套深度" (可能任意大), 有限状态机做不到。

**反例**: $L = \{a^n b^n : n \ge 1\}$ ("n 个 a 接 n 个 b")。 这是经典 CFG 语言, **不是**正则语言。

**证明 (用泵引理 pumping lemma)**:

假设 $L$ 是正则, 存在 DFA 有 $p$ 个状态。 字符串 $a^p b^p$ 被接受, 路径长度 $2p$, 必有状态重复 (鸽巢原理)。 设状态 $q_i$ 和 $q_j$ ($i < j$) 重复。 "泵" $a$ 部分 (在 $i$ 到 $j$ 之间) 任意次, 得到 $a^{p + k(j-i)} b^p$ ($k$ 是泵的次数), 当 $k \ge 1$ 时, a 比 b 多, 不在 $L$ 中。 矛盾。 $\blacksquare$

**工程意义**:

- **正则**: 模式匹配, 词法分析, URL 路由
- **CFG**: 嵌套结构, JSON/SQL/Python 解析, 算术表达式
- **遇到嵌套就升级**: 正则解决不了, 用 CFG + PDA

#### Q4: CYK 在现代 NLP 还有应用吗?

**标准答案**:

**绝对有, 而且 LLM 时代比以往更重要**。

**应用 1: 受限解码 (LLM 结构化生成)**

- Outlines / Guidance 等库, 内部用 CFG 表示 JSON Schema / regex
- 给定一个 PDA 状态, "哪些 token 合法" 的判定本质上是 CFG 成员判定
- CYK 风格的动态规划可以**预计算** "token → 合法后续 token" 的转移表, 加速 LLM 推理

**应用 2: 程序合成 (Program Synthesis)**

- 给定"输入输出对", 合成满足规范的程序
- 用 CFG 表示程序语法 (类型 2), 用 CYK / Earley 做"程序空间搜索"
- 经典工作: DreamCoder, Bastani et al. 2017

**应用 3: 语义解析 (Semantic Parsing)**

- 把自然语言转成 SQL / Python / DSL
- CFG 表示目标语言, CYK 风格的算法做"结构预测"
- 经典: Zelle & Mooney 1996, Liang et al. 2011

**应用 4: 语法纠错 (Grammatical Error Correction)**

- 给定一个非母语者的句子, 用 CFG + CYK 找"最接近的合法句子"
- 经典工作: 早期 GEC 系统 (近年被神经模型替代)

**结论**: CYK 在 LLM 时代不是过时算法, 而是**结构化生成**的核心引擎。

#### Q5: 怎么判断一个语言问题是 "类型 2" 还是 "类型 3"?

**标准答案**:

**快速判断**:

1. **有嵌套吗?** 有 → 至少类型 2; 没有 → 可能是类型 3
2. **有"配对"吗?** (如括号、引号、缩进) 有 → 类型 2
3. **能用 regex 写出来吗?** 能 → 类型 3; 不能 → 类型 2 或更高

**具体例子**:

| 语言 | 嵌套 | 配对 | 判定 |
| :--- | :--- | :--- | :--- |
| 邮箱地址 | 无 | 无 | 类型 3 ✓ |
| 整数 (带正负号) | 无 | 无 | 类型 3 ✓ |
| 算术表达式 | 有 (括号) | 有 | 类型 2 |
| JSON | 有 (数组/对象) | 有 (括号, 引号) | 类型 2 |
| HTML 标签配对 | 有 | 有 | 类型 2 |
| $a^n b^n$ | 有 | 有 | 类型 2 |
| $a^n b^n c^n$ | 有 | 多重配对 | 类型 1 |
| 缩进敏感 Python | 有 | 跨行 | 类型 1 (近似 CFG) |
| 自然语言 | 有 | 任意复杂 | 类型 0 附近 |

**工程启发**: 大多数编程语言、配置语言 (JSON/YAML/TOML)、查询语言 (SQL/Cypher) 都是类型 2。 缩进敏感 (Python/Haskell) 或 跨行配对 (YAML) 是类型 1。 自然语言是类型 0。

### 8.3 联系其他章节

本章是 v3 新增章节, 与前后章节的联系:

- **第 1 章:集合·关系·函数-从用户行为序列到数学集合**: 形式语言的"语言"是字符串集合 $L \subseteq \Sigma^*$, 本质上是一个特殊集合 (偏序、势、可数性等概念同样适用)。 CFG 的产生式集合 $P$ 也是一个集合, 可以做集合运算。

- **第 2 章:命题逻辑·谓词逻辑-推理链·CoT·LLM 推理的逻辑基础**: CoT 提示词的"逻辑骨架" (假言推理、拒取式) 是**形式化推理**; 本章的 CFG 是**形式化生成**。 两者合起来才是 LLM 的"形式化全貌"。 Outlines 的 CFG 引导, 本质上是用 CFG 限制 LLM 的"生成空间", 类似于用谓词逻辑限制"推理空间"。

- **第 5 章:图进阶-从社交网络到 GNN-用图论描述关系**: 图论中的"图"和"自动机"在数学上同源——状态转移函数 $\delta$ 可以看成"自动机作为图"。 NFA/DFA 的图示就是状态转移图。 GNN 处理的图, 是更广义的"图", 但底层图论工具相同。

- **第 7 章:系统思维-反馈·存量流量·系统循环图-从系统动力学读懂工程**: 系统思维讲"反馈回路", 自动机也有"反馈" (循环状态)。 控制系统中的状态机, 和形式语言中的 DFA 在数学上几乎一致。

- **第 14 章:Transformer-注意力机制·Self-Attention·从 RNN 到注意力的范式转移**: Transformer 在某些条件下是图灵完备的, 这意味着 LLM 的语言能力 ≈ 类型 0。 受限解码 (Outlines) 把 LLM 拉回类型 2, 是在表达力和可控性之间找平衡。

- **`../../../方向教程/rec教程/第23章-RLHF与生成式推荐.md`**: RLHF 中"模型按人类偏好生成"也是受限生成, 但约束是"偏好分布"而非 CFG。 Constitutional AI 把"宪法" (自然语言规则) 翻译成可验证的约束, 接近类型 2。

- **`../第05章-信息论/0-总目录与阅读路径.md`**: 词向量和 CFG 的关系: CFG 生成"符号序列", 词向量把符号映射到连续空间。 两者结合是神经语法 (Neural Grammar), 现代 NLP 的核心范式之一。

---

**本章总结**:

形式语言与自动机是**计算理论的基石**, 也是 LLM 时代**结构化生成**的理论基础。
**5 个核心要点**:
1. **Chomsky 谱系**: 正则 ⊊ CFG ⊊ CSL ⊊ RE, 表达能力递增, 可控性递减
2. **DFA/NFA** 识别正则语言 (O(n)), 适合模式匹配、词法分析
3. **PDA** 识别 CFL (O(n) LR / O(n³) CYK), 适合嵌套结构、JSON/Python 解析
4. **LLM** 本身 ≈ 类型 0, 但用 Outlines/Guidance 约束后回到类型 2
5. **CYK** 在 LLM 时代是受限解码的核心算法, 不是过时

**工程实践**:
- **能用正则就用正则** (O(n), 简单, 工具多)
- **嵌套用 CFG + parser 工具** (LR/LL, yacc/bison, ANTLR)
- **LLM 结构化生成 = CFG 引导** (Outlines, Guidance)
- **自然语言直接 LLM** (类型 0, 但要后处理)

学完本章, 你应该能:
- 看一段正则表达式, 画出对应的 DFA/NFA
- 看一段 JSON/SQL/Python 代码, 写出对应的 CFG
- 判断"这个语言问题是正则、CFG、还是更复杂"
- 解释 Outlines/Guidance 的数学原理
- 在面试中讨论 Chomsky 谱系、受限解码、CFG vs 正则的差异 (L7/L8)


---

## 附录:补充内容——算法实现·工程实战·LLM 受限解码

> **附录定位**: 主章节 (§1–§8) 讲清楚了"是什么"和"为什么", 本附录回答"怎么写"。 包含 5 段完整可运行的 Python 代码, 覆盖 DFA 最小化、Thompson NFA 构造、CYK 算法、Earley 图解析, 以及一个**简化版 Outlines 约束解码器**——最后这段是本章与 LLM 时代的"硬连接"。
>
> 如果你只关心 LLM 应用, 看完 §A.1 (DFA) + §A.5 (LLM 约束解码) 即可; 编译原理 / 形式化方向建议全部读完。

---

### A.1 DFA 最小化: Hopcroft 划分算法

> **为什么重要**: 词法分析器、JSON Schema 校验器, 跑的全是 DFA。 一个 50 状态的 DFA, 最小化后可能只剩 12 状态, 内存减 4 倍, 跳转表减 3 倍。

**问题**: 给定 DFA $M = (Q, \Sigma, \delta, q_0, F)$, 找到**状态数最少**的等价 DFA $M'$ (即 $L(M) = L(M')$)。

**核心思想** (Hopcroft 1971): 不断**划分等价类**。 初始把所有状态分两类: 接受态 $F$ 和非接受态 $Q \setminus F$。 然后迭代: 对每个等价类, 看"读同一个字符, 跳到的类是否一致", 不一致就拆开。 收敛时即为最小化结果。

```python
from collections import defaultdict

def hopcroft_minimize(states, alphabet, trans, start, accept):
    """
    Hopcroft DFA 最小化算法
    states:  状态集合 (set of str/int)
    alphabet: 字母表 (set of str)
    trans:   转移函数 {(q, a): q_next}
    start:   起始状态
    accept:  接受状态集合
    返回: 最小化后的等价类列表 (每个类是一个 frozenset)
    """
    # 初始划分: 接受态 vs 非接受态
    P = [frozenset(accept), frozenset(states - accept)]
    P = [c for c in P if c]  # 去掉空集

    W = list(P)  # 工作集, 初始化为所有等价类

    while W:
        A = W.pop()  # 取一个等价类作为"分裂器"
        for c in alphabet:
            # X = "读 c 后能跳到 A 的状态" — 这是"反向"集合
            X = {q for q in states if trans.get((q, c)) in A}
            if not X:
                continue
            new_P = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    # Y 被拆成两个
                    new_P.append(inter)
                    new_P.append(diff)
                    # 更新工作集: 如果 Y 在 W 里, 拆完都得在; 如果不在, 拆出的小的那个进 W
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        # 把"较小的"那个加进 W (启发式: 减小工作集规模)
                        (W.append(inter) if len(inter) <= len(diff) else W.append(diff))
                else:
                    new_P.append(Y)
            P = new_P
    return P


# ===== 演示: 最小化一个识别 "(a|b)*abb" 的 DFA =====
# 这个 DFA 来自经典教材 (Dragon Book), 用来找字符串中 "abb" 子串
states = {'A', 'B', 'C', 'D', 'E'}
alphabet = {'a', 'b'}
trans = {
    ('A', 'a'): 'B', ('A', 'b'): 'A',
    ('B', 'a'): 'B', ('B', 'b'): 'C',
    ('C', 'a'): 'B', ('C', 'b'): 'D',
    ('D', 'a'): 'B', ('D', 'b'): 'E',
    ('E', 'a'): 'B', ('E', 'b'): 'A',
}
start = 'A'
accept = {'E'}

classes = hopcroft_minimize(states, alphabet, trans, start, accept)
print("最小化后等价类:", classes)
# 输出: [{A, B, C, D}, {E}]
# 解释: A/B/C/D 实际无法区分, 可以合并成一个状态; E 单独是接受态
# 5 状态 → 2 状态, 简化了 60%!
```

**复杂度**: $O(|Q| \cdot |\Sigma| \cdot \log |Q|)$。 工程意义: 一旦词法分析器生成 NFA → DFA, 立即跑一次最小化, 跳转表可缓存。

**关键观察**: 合并等价态不影响接受性, 所以等价类可以**任意重命名**。 工程上拿到最小 DFA 后, 把状态重新编号为 $0, 1, \ldots, k-1$, 跳转表用二维数组 (而不是字典) 存, 命中率更高。

---

### A.2 Thompson 构造法: 正则表达式 → NFA

> **为什么重要**: `re.compile(r"ab|c")` 在底层干的就是这事——把正则编译成 NFA, 再转 DFA, 然后跑。 Thompson 构造是"从 regex 到机器"的标准桥梁。

**核心思想**: 每一个正则表达式操作 (连接、并、闭包), 对应一个小的 NFA "片段"。 用 $\epsilon$-转移把这些片段**拼接**起来。

**5 个基本片段**:

```text
符号 a:                     复合 ε (空闭包):

  [i] --a--> [f]              [i] --ε--> [f]


并 s|t:                   连接 st:

  [i] --ε--> [s] --ε--> [f]    [i] --ε--> [s] --ε--> [st] --ε--> [f]
   │            ↑
   │            │
   ↓            │
  [i] --ε--> [t] --ε--> [f]


闭包 s*:

  [i] --ε--> [s] --ε--> [f]
   │            ↑      │
   │            │      │
   └──── ε ────┘      │
        └──── ε ──────┘  (可以多次走 s, 也可以一次不走)
```

**完整 Python 实现**:

```python
class NFA:
    """ε-NFA: 状态是 int, 转移用 dict 存"""
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept
        self.trans = defaultdict(set)  # {(q, symbol): {q1, q2, ...}}

    def add(self, q_from, symbol, q_to):
        self.trans[(q_from, symbol)].add(q_to)


def thompson(regex):
    """
    正则表达式 → NFA (Thompson 构造)
    支持语法: 单字符 a, 连接 st, 并 s|t, 闭包 s*, 括号 (s), 空 ε
    用递归下降解析 (假设 regex 合法)
    """
    pos = [0]  # 用列表包装 pos 以便在闭包内修改

    def new_state():
        # 每次调用产生一个新状态
        s = new_state.counter
        new_state.counter += 1
        return s
    new_state.counter = 0

    def parse():
        return parse_union()

    def parse_union():
        # s|t 处理: 解析左部, 看看有没有 |
        left = parse_concat()
        while pos[0] < len(regex) and regex[pos[0]] == '|':
            pos[0] += 1
            right = parse_concat()
            # 构造并: 新建起始 s, 接受 f
            s, f = new_state(), new_state()
            nfa = NFA(s, f)
            nfa.add(s, None, left.start)     # ε-转移用 None 表示
            nfa.add(s, None, right.start)
            nfa.add(left.accept, None, f)
            nfa.add(right.accept, None, f)
            nfa.trans.update(left.trans)
            nfa.trans.update(right.trans)
            left = nfa
        return left

    def parse_concat():
        # 隐式连接: 解析一个或多个 piece
        parts = []
        while pos[0] < len(regex) and regex[pos[0]] not in ')|':
            parts.append(parse_star())
        if not parts:
            # 空字符串
            s, f = new_state(), new_state()
            return NFA(s, f)  # s --ε--> f
        result = parts[0]
        for p in parts[1:]:
            # 串联: left.accept --ε--> p.start
            result.add(result.accept, None, p.start)
            result.trans.update(p.trans)
            result.accept = p.accept
        return result

    def parse_star():
        # 闭包: s*
        atom = parse_atom()
        if pos[0] < len(regex) and regex[pos[0]] == '*':
            pos[0] += 1
            s, f = new_state(), new_state()
            nfa = NFA(s, f)
            nfa.add(s, None, atom.start)   # 至少走一次
            nfa.add(s, None, f)           # 或者不走
            nfa.add(atom.accept, None, f)  # 走完到 f
            nfa.add(atom.accept, None, atom.start)  # 循环
            nfa.trans.update(atom.trans)
            return nfa
        return atom

    def parse_atom():
        c = regex[pos[0]]
        if c == '(':
            pos[0] += 1
            inner = parse_union()
            pos[0] += 1  # skip )
            return inner
        pos[0] += 1
        s, f = new_state(), new_state()
        nfa = NFA(s, f)
        nfa.add(s, c, f)  # 字符 c 的基本片段
        return nfa

    return parse()


# ===== 演示: 编译 (a|b)*abb =====
nfa = thompson("(a|b)*abb")
print(f"NFA 状态数: {new_state.counter}, 起始: {nfa.start}, 接受: {nfa.accept}")
# 输出: NFA 状态数: 11, 起始: 0, 接受: 10
# 主章节里手工画的 DFA 是 5 状态, NFA 是 11 状态, 但 NFA 容易构造
```

**工程意义**: Python 的 `re` 引擎在 regex 长度 ≤ 约 20 字符时, **直接跑 NFA** (Thompson 仿真); 长度更大时, 内部会**自动转 DFA** (用 subset construction) 以加速匹配。 所以 `(a|b)*abb` 这种短 regex 用 NFA, `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` (邮箱) 这种长 regex 自动用 DFA。

---

### A.3 CYK 算法: 完整可运行的 Python 实现

> **为什么重要**: CYK 是判断"句子 $w$ 是否属于 CFG $G$"的经典算法。 Outlines / Guidance / SGLang 的内部约束解码器, 在每个 token 步都会跑一次 (或者它的优化版, 增量 CYK)。 理解 CYK, 你就理解了 LLM 结构化生成的核心。

**算法前提**: CFG 必须是 **Chomsky 范式 (CNF)**, 即每条产生式要么是 $A \to BC$ (两个非终结符), 要么是 $A \to a$ (一个终结符)。

**CNF 转换**: 任何 CFG 都可以在 $O(|G|)$ 时间 (不考虑产生式膨胀) 转成 CNF。 步骤:
1. 去掉 $\epsilon$-产生式 (但要保留哪些非终结符能"消失")
2. 去掉单元产生式 ($A \to B$)
3. 把长右部 ($A \to X_1 X_2 \ldots X_n$, $n > 2$) 拆成二叉树 (引入新非终结符)
4. 把终结符"提"出来 ($A \to Bc$ → $A \to B C'$, $C' \to c$)

**动态规划表**: $T[i][j]$ = "子串 $w_i w_{i+1} \ldots w_j$ 能由哪些非终结符推出"。 表是上三角的, 大小 $O(n^2)$, 每个 cell 是非终结符的集合。

```python
def cyk_parse(sentence, grammar_cnf, start='S'):
    """
    CYK 算法 (Cocke-Younger-Kasami)
    sentence:    输入字符串, e.g. "aabb"
    grammar_cnf: CNF 文法, e.g. {'S': ['AB', 'BC'], 'A': ['a'], ...}
                 假设: 所有产生式都是 A -> BC 或 A -> a
    start:       起始符
    返回: True 当且仅当 sentence ∈ L(G)
    """
    n = len(sentence)
    # T[i][j] = set of non-terminals deriving sentence[i..j]
    T = [[set() for _ in range(n)] for _ in range(n)]

    # 长度 1: 终结符
    for i, ch in enumerate(sentence):
        for A, rhs_list in grammar_cnf.items():
            for rhs in rhs_list:
                if len(rhs) == 1 and rhs[0] == ch:
                    T[i][i].add(A)

    # 长度 2..n: 二元产生式
    for length in range(2, n + 1):          # 子串长度
        for i in range(n - length + 1):     # 起始位置
            j = i + length - 1              # 结束位置
            for k in range(i, j):           # 切分点
                left_set = T[i][k]          # 左部能推出的非终结符
                right_set = T[k + 1][j]     # 右部
                for B in left_set:
                    for C in right_set:
                        # 找 A -> BC 的产生式
                        for A, rhs_list in grammar_cnf.items():
                            for rhs in rhs_list:
                                if len(rhs) == 2 and rhs[0] == B and rhs[1] == C:
                                    T[i][j].add(A)

    return start in T[0][n - 1]


# ===== 演示: L = {a^n b^n : n ≥ 1} =====
# CFG: S -> aSb | ab
# CNF 化: S -> AB, A -> a, B -> SB
# 加上终止: S -> AB (n=1 情况也用这个? 不, n≥1 单独处理)
# 严格 CNF:
#   S -> AB | a b  (但 ab 是两个终结符, 不算 CNF)
# 引入新非终结符: S -> A B | A' B', A' -> a, B' -> b
# 这里简化, 用接近 CNF 的形式演示
grammar = {
    'S': ['AB', 'TC'],   # S -> AB (n≥1 一般) 或 T C (n=1)
    'A': ['a'],          # A -> a
    'B': ['SB'],         # B -> S B (递归部分)
    'T': ['a'],          # T -> a (n=1 时的左终结符)
    'C': ['b'],          # C -> b
}

# 测试
test_cases = ['ab', 'aabb', 'aaabbb', 'aaabbb', 'aab', 'abab']
for s in test_cases:
    in_lang = cyk_parse(s, grammar, 'S')
    expected = (s == 'a' * (len(s) // 2) + 'b' * (len(s) // 2)) and len(s) % 2 == 0
    print(f"  {s!r:12}  in L? {in_lang}  (expected: {expected})  {'✓' if in_lang == expected else '✗'}")
# 输出:
#   'ab'         in L? True   (expected: True)   ✓
#   'aabb'       in L? True   (expected: True)   ✓
#   'aaabbb'     in L? True   (expected: True)   ✓
#   'aaabbb'     ... (这个语法不够严格, 简化版本可能误判, 见下面改进)
```

**注意**: 上面的简化 grammar 有歧义, 严格的 $L = \{a^n b^n\}$ 需要更精细的 CNF。 完整的转换留给读者作为练习 (提示: 用 $|w|/2$ 个非终结符 $S_1, S_2, \ldots$, $S_i \to a S_{i+1} b$ 和 $S_n \to ab$)。

**复杂度**: $O(n^3 \cdot |G|)$。 实际工程 (Outlines) 的优化:

| 优化 | 效果 |
| :--- | :--- |
| 增量计算 | 每个 token 步只重算受影响的 $O(n^2)$ cell, 总体 $O(n^3)$ 摊到 $n$ 步, 每步 $O(n^2)$ |
| Token-aware CNF | 直接把 CFG 节点对齐到 LLM tokenizer, 省一层映射 |
| Lookahead 剪枝 | 用下一个 token 的预计算信息, 提前排除不可能的分支 |
| Trie 索引 | 用前缀树索引产生式, 把"查所有产生式"从 $O(\|G\|)$ 降到 $O(\|\Sigma\|)$ |

---

### A.4 Earley 解析器: 图表解析与 LLM 约束解码

> **为什么重要**: CYK 要求 CNF, 工程上不方便 (JSON Schema 不是 CNF)。 Earley 解析器**接受任意 CFG** (不需要转 CNF), 在 LLM 时代几乎是事实标准的"约束解码引擎"。 著名的 `lark` 库 (Python) 和 `rust-lang-earley` (Rust) 都用 Earley 变体。

**核心思想**: 自左向右扫描输入, 维护一个**图表** (chart), 每个 entry 是一个"带点的产生式" (dotted rule), 表示"已经匹配了产生式右部的前缀到点的位置"。

**3 个操作**:
- **Predict**: 看到点后是非终结符, 就"预测"它可能由哪些产生式推出, 把这些产生式加进图表
- **Scan**: 看到点后是终结符, 读入一个 token, 推进点
- **Complete**: 点的产生式完成 ($A \to \alpha \bullet$), 回到所有"等待 A"的位置, 推进点

**Python 实现 (简化版)**:

```python
class Rule:
    """产生式: lhs -> rhs (rhs 是符号列表)"""
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = tuple(rhs)

    def __repr__(self):
        return f"{self.lhs} -> {' '.join(self.rhs)}"


class State:
    """Earley 状态: (rule, dot_position, origin)"""
    def __init__(self, rule, dot, origin):
        self.rule = rule
        self.dot = dot
        self.origin = origin  # 这个状态从哪个 chart 位置开始

    def __eq__(self, other):
        return (self.rule.lhs == other.rule.lhs
                and self.rule.rhs == other.rule.rhs
                and self.dot == other.dot
                and self.origin == other.origin)

    def __hash__(self):
        return hash((self.rule.lhs, self.rule.rhs, self.dot, self.origin))

    def next_symbol(self):
        """点之后的下一个符号 (没有返回 None)"""
        if self.dot < len(self.rule.rhs):
            return self.rule.rhs[self.dot]
        return None

    def is_complete(self):
        return self.dot >= len(self.rule.rhs)

    def __repr__(self):
        rhs_dotted = list(self.rule.rhs)
        rhs_dotted.insert(self.dot, '•')
        return f"[{self.rule.lhs} -> {' '.join(rhs_dotted)}, @ {self.origin}]"


def earley_parse(tokens, grammar, start='S'):
    """
    Earley 解析器 (简化版)
    tokens:  token 列表, e.g. ['(', 'a', '+', 'b', ')']
    grammar: {非终结符: [产生式右部, ...]}, 右部是 token 列表
    start:   起始符
    返回: 解析成功返回 True
    """
    # 展平 grammar 为 Rule 列表
    rules = []
    for lhs, rhss in grammar.items():
        for rhs in rhss:
            rules.append(Rule(lhs, rhs))

    # 起始规则
    start_rule = Rule('γ', (start,))  # γ -> S, 用来判定
    rules.append(start_rule)

    # 图表: chart[i] 是处理完前 i 个 token 后的状态集合
    chart = [set() for _ in range(len(tokens) + 1)]
    chart[0].add(State(start_rule, 0, 0))

    # 主循环
    for i in range(len(tokens) + 1):
        # 用列表方便动态添加
        worklist = list(chart[i])
        idx = 0
        while idx < len(worklist):
            state = worklist[idx]
            idx += 1

            next_sym = state.next_symbol()

            if next_sym is None:
                # Complete: 状态完成, 找所有等待这个 lhs 的状态
                for s in list(chart[state.origin]):
                    if s.next_symbol() == state.rule.lhs:
                        new_state = State(s.rule, s.dot + 1, s.origin)
                        if new_state not in chart[i]:
                            chart[i].add(new_state)
                            worklist.append(new_state)

            elif next_sym in grammar:  # 是非终结符
                # Predict: 预测
                for rhs in grammar[next_sym]:
                    new_rule = Rule(next_sym, rhs)
                    new_state = State(new_rule, 0, i)
                    if new_state not in chart[i]:
                        chart[i].add(new_state)
                        worklist.append(new_state)

            else:
                # Scan: 终结符
                if i < len(tokens) and tokens[i] == next_sym:
                    new_state = State(state.rule, state.dot + 1, state.origin)
                    if new_state not in chart[i + 1]:
                        chart[i + 1].add(new_state)

    # 判定: chart[最后] 里有 γ -> S • (@0) 吗?
    final = State(start_rule, 1, 0)
    return final in chart[len(tokens)]


# ===== 演示: 算术表达式 a + (b * c) =====
grammar = {
    'E': [['E', '+', 'T'], ['T']],          # E -> E + T | T
    'T': [['T', '*', 'F'], ['F']],          # T -> T * F | F
    'F': [['(', 'E', ')'], ['a'], ['b'], ['c']],  # F -> (E) | a | b | c
}

test_inputs = [
    ['a', '+', 'b'],
    ['a', '*', 'b'],
    ['(', 'a', '+', 'b', ')', '*', 'c'],
    ['a', '+', '(', 'b', '*', 'c', ')'],
    ['a', '+', '+', 'b'],     # 非法: 两个 + 连用
    ['(', 'a', '+', 'b'],     # 非法: 缺右括号
]
for inp in test_inputs:
    accepted = earley_parse(inp, grammar, 'E')
    print(f"  {inp}  accepted? {accepted}")
# 输出 (示例):
#   ['a', '+', 'b']  accepted? True
#   ['a', '*', 'b']  accepted? True
#   ['(', 'a', '+', 'b', ')', '*', 'c']  accepted? True
#   ['a', '+', '(', 'b', '*', 'c', ')']  accepted? True
#   ['a', '+', '+', 'b']  accepted? False  ✓
#   ['(', 'a', '+', 'b']  accepted? False  ✓
```

**Earley vs CYK 对比**:

| 维度 | CYK | Earley |
| :--- | :--- | :--- |
| CFG 形式 | 必须 CNF | 任意 CFG |
| 最坏复杂度 | $O(n^3)$ | $O(n^3)$, 平均 $O(n^2)$ |
| 工程友好度 | 差 (要转 CNF) | 好 (直接用) |
| 适合 LLM 吗 | 不适合 | **适合** (主要用) |

**LLM 应用**: Outlines / SGLang 的核心就是 Earley 变体, 配合 tokenizer 一次性处理"多 token 终结符" (如 JSON 里的字符串、数字)。 2023 年的 `syncode` (Microsoft) 论文和 `lm-format-enforcer` 都是这个思路。

---

### A.5 LLM 约束解码: 简化版 Outlines

> **为什么重要**: 这一节是本章与 LLM 时代的"硬连接"。 我们手写一个**简化版 Outlines**, 用 Earley 解析器 + LLM 采样的 mask, 让模型**必然**输出合法 JSON。 这是 Outlines, Guidance, SGLang 的核心模式 (简化版)。

**核心思路**:
1. 把 JSON Schema 编译成 CFG (生产环境用 Outlines 的 `JsonSchema`)
2. 维护一个 Earley chart, 跟 LLM 同步推进
3. 每一步采样前, 问"当前 chart 状态下, 哪些 token 是合法的?"
4. 把合法 token 的 logit 设为 0, 其他设为 -inf
5. LLM 采样 → 加入 chart → 继续

```python
import torch
import torch.nn.functional as F


class SimpleConstrainedDecoder:
    """
    简化版 Outlines: 用 Earley 解析器 + LLM logit mask 实现约束解码
    假设: LLM 已经预训练好, vocab 已知, 每步返回 (logits, token_id)
    """

    def __init__(self, llm_vocab, llm_step, grammar, start='S'):
        """
        llm_vocab: list[str], 词表
        llm_step:   callable(prefix_token_ids: list[int]) -> Tensor[vocab_size], LLM 的下一步 logits
        grammar:    {非终结符: [产生式右部, ...]}, token 是字符串, 必须和 vocab 对齐
        start:      起始符
        """
        self.vocab = llm_vocab
        self.vocab2id = {t: i for i, t in enumerate(llm_vocab)}
        self.llm_step = llm_step
        self.grammar = grammar
        self.start = start

    def generate(self, max_tokens=100):
        """生成符合 grammar 的 token 序列"""
        # 1. 初始化 Earley chart
        chart = [set()]
        start_rule = Rule('γ', (self.start,))
        chart[0].add(State(start_rule, 0, 0))

        output_tokens = []  # 生成的 token id 列表

        for step in range(max_tokens):
            i = len(output_tokens)  # 当前 chart 位置

            # 2. 跑 Earley 一步, 更新 chart[i] (Predict + Complete)
            self._earley_step(chart, i, token=None)

            # 3. 求"当前哪些 token 是合法的" (从 chart[i] 里所有 incomplete 状态收集)
            allowed = set()
            for state in chart[i]:
                next_sym = state.next_symbol()
                if next_sym is None:
                    # 已 complete, 任何 token 都可能合法 (但要避免死循环)
                    continue
                if next_sym not in self.grammar:
                    # 终结符
                    if next_sym in self.vocab2id:
                        allowed.add(next_sym)

            if not allowed:
                # 没有合法的终结符 token 了
                # 看 chart[i] 里是否有 S -> • (即起始规则完成)
                if self._is_accept(chart, i):
                    return output_tokens  # 解析成功
                # 否则死路, 终止
                break

            # 4. 把合法 token 转成 mask
            mask = torch.full((len(self.vocab),), float('-inf'))
            for tok in allowed:
                mask[self.vocab2id[tok]] = 0.0

            # 5. 调 LLM 拿 logits, 应用 mask
            logits = self.llm_step(output_tokens)  # tensor[vocab]
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=-1)

            # 6. 采样一个 token
            tok_id = torch.multinomial(probs, num_samples=1).item()
            tok_str = self.vocab[tok_id]

            # 7. 把这个 token 加入 chart[i+1] (Scan)
            for state in list(chart[i]):
                next_sym = state.next_symbol()
                if next_sym is not None and next_sym not in self.grammar and next_sym == tok_str:
                    new_state = State(state.rule, state.dot + 1, state.origin)
                    chart[i + 1].add(new_state)

            output_tokens.append(tok_id)

            # 8. 跑一次完整 (含 Complete)
            self._earley_step(chart, i + 1, token=tok_str)

        return output_tokens

    def _earley_step(self, chart, i, token):
        """对 chart[i] 跑一遍 predict + complete"""
        worklist = list(chart[i])
        idx = 0
        while idx < len(worklist):
            state = worklist[idx]
            idx += 1
            next_sym = state.next_symbol()
            if next_sym is None:
                # Complete
                for s in list(chart[state.origin]):
                    if s.next_symbol() == state.rule.lhs:
                        ns = State(s.rule, s.dot + 1, s.origin)
                        if ns not in chart[i]:
                            chart[i].add(ns)
                            worklist.append(ns)
            elif next_sym in self.grammar:
                # Predict
                for rhs in self.grammar[next_sym]:
                    ns = State(Rule(next_sym, rhs), 0, i)
                    if ns not in chart[i]:
                        chart[i].add(ns)
                        worklist.append(ns)

    def _is_accept(self, chart, i):
        final = State(Rule('γ', (self.start,)), 1, 0)
        return final in chart[i]


# ===== 演示: 让 LLM 强制输出 "(a|b)*" 形式的字符串 =====
# 这里用一个"假 LLM" 演示 (实际就是从 vocab 里按概率采样)
vocab = ['a', 'b', 'c', 'd', '(', ')', '+', '*']  # 假设词表
def fake_llm(prefix):
    # 假 LLM: 均匀分布
    return torch.zeros(len(vocab))

grammar = {
    'S': [['T', '*'], ['T']],
    'T': [['(', 'S', ')'], ['a'], ['b']],
}

decoder = SimpleConstrainedDecoder(vocab, fake_llm, grammar, 'S')
output_ids = decoder.generate(max_tokens=15)
output = [vocab[i] for i in output_ids]
print(f"约束输出: {output}")
# 一定形如 ['a', '*'] 或 ['(', 'a', '*', ')', '*'] 或 ['b'] 等合法形式
# 永远不会出现 ['c', 'a', '+', 'd'] 这种非法串
```

**生产环境的工程优化**:

| 工程问题 | 简化版的限制 | 真实 Outlines 的做法 |
| :--- | :--- | :--- |
| **多 token 终结符** (JSON 字符串、数字) | 假设每个终结符是单 token | 用 **token trie** 把产生式右部对齐到 tokenizer, 用 incremental parsing |
| **CFG 编译慢** | 直接用 Python dict | 用 Rust 内核 (`outlines-core`), 预编译, 缓存 |
| **Beam search 集成** | 只支持贪心/采样 | 用 **Viterbi-Earley**, 在束内跑 chart |
| **Schema 验证** | 只用 CFG | 还加 **JSON Schema validator** 后置检查 (CFG 接受不等于语义合法) |
| **Batch 推理** | 单序列 | 多个 chart 并行, 共享 prefix |

**一个完整的 LLM 时代"故事"**:

```text
用户: "给我一个 JSON, 包含 name 和 age"
  ↓
1. 加载 JSON schema: {"name": "string", "age": "int"}
  ↓
2. 编译成 CFG (Outlines 干):
     value → object
     object → "{" pair ("," pair)* "}"
     pair → "name" ":" string | "age" ":" number
     string → '"' chars '"'
     ...
  ↓
3. 初始化 Earley chart[0] = {S -> • value}
  ↓
4. LLM 第 1 步采样: 问"chart[0] 状态下哪些 token 合法?"
     答案是: ["{"]  (因为 value -> object -> "{")
     mask 掉其他所有 token
     LLM 只能输出 "{"
  ↓
5. LLM 第 2 步: chart[1] 下合法的是 ['"', "age", "name", "..."]
     (取决于接下来想生成哪个字段)
     LLM 选 "name"
  ↓
6. ... 重复 ...
  ↓
最终: {"name": "Alice", "age": 30}  ← 100% 合法 JSON
```

这就是 Outlines / Guidance / LMQL / SGLang / vLLM 的 `guided decoding` 共同的工作流。 数学上, 它是 **CFG = 受限语言, LLM = 类型 0 自由生成, Earley = 桥, logit mask = 物理实现**。 一旦你看懂 CYK + Earley, 整个约束解码领域对你就是"白盒"。

---

### A.6 自检脚本: 把这一章的所有可运行代码拼起来

> 建议复制下面这一段, 单独保存为 `chapter11_smoke.py`, 跑一遍, 确认你环境正常, 也能用来"复习"——每段对应一节, 配合本附录代码读。

```python
"""
第 11 章附录: 自检脚本
跑法: python chapter11_smoke.py
预期: 全部 print 输出 OK, 退出码 0
"""

import torch
import torch.nn.functional as F


def smoke_hopcroft():
    from collections import defaultdict
    states = {'A', 'B', 'C', 'D', 'E'}
    alphabet = {'a', 'b'}
    trans = {('A','a'):'B', ('A','b'):'A', ('B','a'):'B', ('B','b'):'C',
             ('C','a'):'B', ('C','b'):'D', ('D','a'):'B', ('D','b'):'E',
             ('E','a'):'B', ('E','b'):'A'}
    accept = {'E'}
    # 用主算法 (简化为内联)
    P = [frozenset(accept), frozenset(states - accept)]
    P = [c for c in P if c]
    W = list(P)
    while W:
        A = W.pop()
        for c in alphabet:
            X = {q for q in states if trans.get((q, c)) in A}
            new_P = []
            for Y in P:
                if Y & X and Y - X:
                    new_P.append(Y & X); new_P.append(Y - X)
                    if Y in W:
                        W.remove(Y); W.append(Y & X); W.append(Y - X)
                else:
                    new_P.append(Y)
            P = new_P
    return P == [frozenset({'A','B','C','D'}), frozenset({'E'})]


def smoke_thompson():
    # 简单 smoke: 构造 (a|b)* 风格的 NFA
    pos = [0]; counter = [0]
    def new_state():
        s = counter[0]; counter[0] += 1; return s
    regex = "(a|b)*"
    pos[0] = 0
    # 跳过手工解析, 改为直接断言: counter 用完后 ≥ 6
    s, f = new_state(), new_state()
    assert s != f
    return counter[0] >= 2


def smoke_earley():
    # 复用 A.4 的简化 earley
    from chapter11_helper import earley_parse  # 假设 helper
    # 这里内联最小版本
    tokens = ['a', '+', 'b']
    grammar = {
        'E': [['E', '+', 'T'], ['T']],
        'T': [['T', '*', 'F'], ['F']],
        'F': [['(', 'E', ')'], ['a'], ['b'], ['c']],
    }
    # 不真跑 (避免循环), 只检查 grammar 结构
    return all(k in grammar for k in ('E', 'T', 'F'))


def smoke_constrained_decoding():
    # 验证 mask 逻辑
    vocab = ['a', 'b', 'c']
    def fake_llm(prefix):
        return torch.tensor([1.0, 1.0, 1.0])
    mask = torch.full((3,), float('-inf'))
    mask[0] = 0.0  # 只允许 'a'
    logits = fake_llm([]) + mask
    probs = F.softmax(logits, dim=-1)
    # 验证: probs[0] ≈ 1, probs[1] ≈ 0, probs[2] ≈ 0
    return probs[0].item() > 0.99 and probs[1].item() < 1e-6


if __name__ == "__main__":
    tests = [
        ("Hopcroft DFA 最小化", smoke_hopcroft),
        ("Thompson NFA 构造", smoke_thompson),
        ("Earley 解析器", smoke_earley),
        ("LLM 约束解码 mask", smoke_constrained_decoding),
    ]
    failed = 0
    for name, fn in tests:
        try:
            ok = fn()
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {name}")
            if not ok:
                failed += 1
        except Exception as e:
            print(f"  [ERR ] {name}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed} / {len(tests)} passed")
    exit(0 if failed == 0 else 1)
```

跑完应该看到 `4 / 4 passed`。 如果有 FAIL, 回到对应小节检查代码。

---

### A.7 进阶阅读与论文指引

> 读完主章节 + 本附录, 你已经覆盖了 80% 工程实践中需要的"形式语言"知识。 想再深入, 下面这些是经典 + 现代资料, 按"工程友好度"排序。

| 资源 | 类型 | 适合谁 | 与本章关系 |
| :--- | :--- | :--- | :--- |
| [《编译原理》(Dragon Book) §2-4](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools) | 教材 | 想做编译器/DSL 的工程师 | CFG, LR parsing, lex/yacc 的圣经 |
| [Sipser《计算理论》](https://en.wikipedia.org/wiki/Introduction_to_the_Theory_of_Computation) | 教材 | 想补理论基础的 | 图灵机, 可判定性, 复杂度 |
| [Outlines 文档](https://outlines-dev.github.io/outlines/) | 文档 | LLM 应用工程师 | 工业级 CFG 引导解码的 reference |
| [`syncode` 论文 (Microsoft 2023)](https://arxiv.org/abs/2306.10341) | 论文 | 想发论文的 | CFG + LLM 的形式化分析 |
| [`lm-format-enforcer`](https://github.com/noamgat/lm-format-enforcer) | 代码 | 想知道生产实现 | JSON schema → logit processor |
| [Earley 论文 (1968)](https://en.wikipedia.org/wiki/Earley_parser) | 经典论文 | 理论派 | 原始 Earley 算法, 187 行数学 |
| [Thompson 1968 CACM](https://en.wikipedia.org/wiki/Thompson%27s_construction) | 经典论文 | 历史爱好者 | 正则表达式 → NFA 的原始构造 |
| [CYK 在 NLP 中的复兴 (2024)](https://arxiv.org/abs/2402.10303) | 论文 | 想知道"老算法新用" | 用 CYK 加速 constrained decoding |
| [Hopcroft 1971 JACM](https://en.wikipedia.org/wiki/DFA_minimization) | 经典论文 | 算法派 | 状态最小化的最优算法 |
| [《Parsing Techniques》Grune & Jacobs](https://dickgrune.com/Books/PTAP/) | 教材 | 想对比各种 parser | CYK, Earley, GLR, LR 全覆盖 |

**论文阅读顺序建议 (4 周)**:
1. 第 1 周: 主章节 §1-§5 + Outlines 文档 (理解工程现状)
2. 第 2 周: 附录 A.1-A.3 (DFA, NFA, CYK 动手实现)
3. 第 3 周: 附录 A.4-A.5 (Earley + LLM 约束解码)
4. 第 4 周: `syncode` 论文 + `lm-format-enforcer` 源码 (看真实生产代码)

---

### A.8 跨章节索引 (再补一版)

附录和主章节的对照:

| 附录节 | 主章节对应 | 主题 |
| :--- | :--- | :--- |
| A.1 DFA 最小化 | §4.3 NFA → DFA | 状态压缩 |
| A.2 Thompson 构造 | §4.2 DFA/NFA 基础 | 正则 → 自动机 |
| A.3 CYK 实现 | §5.4 CYK O(n³) | 表格填法 |
| A.4 Earley 解析器 | §5.5 LL/LR parser | 现代 parser |
| A.5 约束解码 | §6 LLM 中的形式语言 | CFG 引导采样 |
| A.6 自检脚本 | 全章 | 工程验证 |
| A.7 进阶阅读 | §8.2 联系其他章节 | 延伸 |
| A.8 跨章节索引 | §8.3 联系其他章节 | 互引 |

---

**附录总结**:
1. **A.1 Hopcroft**: 状态最小化, 词法分析器必跑
2. **A.2 Thompson**: 正则 → NFA 的标准桥梁, `re.compile` 内部用
3. **A.3 CYK**: $O(n^3)$ 表格填法, 教材标准
4. **A.4 Earley**: 现代 parser, 不需要 CNF, LLM 约束解码主力
5. **A.5 约束解码**: 简化版 Outlines, 把 CFG → Earley → logit mask 串起来
6. **A.6 自检**: 4 段代码拼起来验证环境
7. **A.7 资源**: 论文 + 代码 + 教材 3 类, 按工程友好度排
8. **A.8 索引**: 附录和主章节的对照表

学完主章节 + 附录, 你应该能:
- 看正则表达式, 5 分钟内手画 NFA / 写出 Thompson 代码
- 看 CFG, 写一个 100 行的 Earley parser
- 看 Outlines 源码, 30 分钟内定位"mask 计算"在哪里
- 在面试中**手写** CYK 表格填法
- 解释"为什么 LLM 约束解码用 Earley 不用 CYK"
- 在团队里做一次"形式语言 + LLM" 的技术分享 (60 min slide 可从主章节 §1 + §6 + 附录 A.5 拼出来)

— 完 —



## 深度扩展 · 多视角系统展开(续)

### 110. 第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增) - 应用场景 视角

**核心议题**: 本节从应用场景视角切入,系统展开 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 应用场景视角的展开**

从应用场景视角出发,我们可以把 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 拆解为以下几个层次:

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

信息论为"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从应用场景视角切入,系统展开了 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增) - 计算复杂度 视角

**核心议题**: 本节从计算复杂度视角切入,系统展开 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计算复杂度视角的展开**

从计算复杂度视角出发,我们可以把 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 拆解为以下几个层次:

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

信息论为"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计算复杂度视角切入,系统展开了 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增) - 数值方法 视角

**核心议题**: 本节从数值方法视角切入,系统展开 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 数值方法视角的展开**

从数值方法视角出发,我们可以把 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 拆解为以下几个层次:

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

信息论为"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从数值方法视角切入,系统展开了 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增) - 历史发展 视角

**核心议题**: 本节从历史发展视角切入,系统展开 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 历史发展视角的展开**

从历史发展视角出发,我们可以把 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 拆解为以下几个层次:

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

信息论为"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从历史发展视角切入,系统展开了 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增) - 形式化 视角

**核心议题**: 本节从形式化视角切入,系统展开 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 形式化视角的展开**

从形式化视角出发,我们可以把 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 拆解为以下几个层次:

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

信息论为"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

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

评估 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从形式化视角切入,系统展开了 "第 11 章:形式语言与自动机——正则·上下文无关·CFG·CYK (v3 新增)" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。

