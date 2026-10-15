---
title: 第24章-特征工程与特征交叉 · Datawhalechina 补充
tags:
  - 经典推荐
  - 特征交叉
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/fun-rec（小麦书）上篇 ch3"精准偏好预测"与 torch-rechub，补充记忆与泛化、二阶/高阶特征交叉的谱系推导与可运行实现
companion_to: 第24章-特征工程与特征交叉.md
source: https://github.com/datawhalechina/fun-rec ; https://github.com/datawhalechina/torch-rechub
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 特征工程与特征交叉 · Datawhalechina 对照补充

> 本文件是对同目录《第24章-特征工程与特征交叉.md》的补充，内容整理自 [datawhalechina/fun-rec](https://github.com/datawhalechina/fun-rec)（小麦书）**上篇 ch3 精准偏好预测**（记忆与泛化 / 二阶交叉 / 高阶交叉小节）与 [torch-rechub](https://github.com/datawhalechina/torch-rechub) 的模型实现。fun-rec ch3 的差异化价值是给特征交叉一个**"记忆 vs 泛化"的第一性框架**，再让 FM→FFM→FwFM→DeepFM→DCN→DCN-v2→xDeepFM 全谱系各就各位。

---

## 一、补充来源地图

| fun-rec 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch3.1 记忆与泛化 | 权重模型（记忆）vs Embedding 模型（泛化）的张力 | 第24章正文 | **核心补充**：谱系的组织原则 |
| ch3.2 二阶交叉 | FM/FFM/FwFM 推导 | 第24章正文 | 逐模型的改进动因链 |
| ch3.3 高阶交叉 | CrossNet（DCN/DCN-v2）、CIN（xDeepFM） | 第24章正文 | 显式高阶 vs 隐式 MLP 的对照 |
| torch-rechub 模型库 | DeepFM/DCN/DCN-v2 等 13+ 实现 | 第22章（关联） | "论文 → 代码"映射 |

---

## 二、核心补充内容

### 2.1 组织谱系的第一性框架：记忆 vs 泛化（fun-rec ch3.1）

fun-rec 把所有 CTR 模型的动机归结为一对张力，建议作为第24章的开篇视角：

| 路线 | 机制 | 代表 | 强项 | 弱项 |
| :--- | :--- | :--- | :--- | :--- |
| 记忆（Memorization） | 直接学"特征对 → 点击"的权重 | LR、FM 显式交叉 | 精确匹配、可解释 | 组合爆炸、零泛化 |
| 泛化（Generalization） | 学低维 Embedding 间接交叉 | 深度 MLP、双塔 | 长尾泛化 | 精确匹配弱、不可解释 |

Wide&Deep、DeepFM 及后续所有模型 = 两条路线的**混合配方**；配方差异就是各模型的身份。读任何新论文先问：它在两条路线上各加了什么。

### 2.2 二阶交叉的演进链（fun-rec ch3.2）

fun-rec 用"解决上一个模型的什么问题"串起二阶交叉谱系：

```text
LR 手工交叉（组合爆炸）
 → FM：所有特征共享一个 k 维隐向量，交叉 = 内积（参数 O(nk)）
    弱点：每个特征对所有领域共用一个向量
 → FFM：每个特征在每个领域各有一个向量（field-aware）
    弱点：参数 O(nfk)，只适合小规模特征
 → FwFM：为每对 field 学一个全局权重 w(f_i,f_j)，缓解噪声交叉
 → DeepFM：FM 侧 + Deep 侧共享同一 Embedding，免手工交叉
```

**常考细节**（fun-rec 强调、本地可补的）：FM 的 O(n²) 内积可以利用恒等式 $\sum_{i<j}\langle v_i,v_j\rangle x_ix_j = \frac{1}{2}\| \sum_i x_iv_i \|^2 - \frac{1}{2}\sum_i x_i^2\|v_i\|^2$ 化简为 O(nk)——这是面试高频推导点。

### 2.3 高阶交叉：DCN → DCN-v2 → xDeepFM（fun-rec ch3.3）

- **CrossNet（DCN）**：每层做 $x_{l+1} = x_0 \odot (W x_l + b) + x_l$，特征交叉阶数随层数线性增长、参数只有 $O(L \cdot d)$；
- **DCN-v2**：把 $W$ 从对角阵放开为满秩/低秩矩阵，表达力对齐 MLP 但保持显式交叉结构；低秩版 `rank` 参数是精度-成本旋钮；
- **xDeepFM 的 CIN**：显式学习"向量级"交叉（vector-wise）而非"比特级"，第 k 层是前 k-1 层与原始 embedding 的外积卷积——显式高阶与泛化的折中。

**选型经验**（fun-rec ch3.3 + torch-rechub 实践）：特征维度高、交叉稀疏 → DCN-v2（低秩）性价比最高；要可解释的交叉强度 → FM/DCN 显式分支单独输出权重；算力富裕 → xDeepFM。

### 2.4 torch-rechub 对照实践

torch-rechub 的 `DeepFM / Wide&Deep / DCN / DCN-v2` 实现与 fun-rec ch3 逐模型对应，建议按"读一节推导 → 跑一个模型（Criteo 小样本）→ 对比 AUC/参数量"的节奏走一遍，产出自己的谱系对照表（模型 / 交叉方式 / 阶数 / 参数量 / Criteo AUC）。

---

## 三、可直接动手的实践

1. 手推 FM 的 O(nk) 化简恒等式（2.2 的公式），并用 torch-rechub 的 FM 实现验证两式数值一致；
2. 在 Criteo 小数据集上跑 torch-rechub 的 DeepFM vs DCN-v2(rank=32)，记录参数量与 AUC，填入自己的谱系表；
3. 用"记忆 vs 泛化"框架给本地第24章已覆盖的模型逐个标注，找出只讲结构未讲动机的部分回填。

---

## 四、延伸链接

- fun-rec 在线阅读：<https://datawhalechina.github.io/fun-rec/>
- torch-rechub：<https://github.com/datawhalechina/torch-rechub>
- 本地相关文件：第22章-DLRM经典推荐算法详解（模型详解）、第03章-Embedding与数据组织（Embedding 侧）、第26章-粗排精排与重排-Datawhale补充.md（下游衔接）
