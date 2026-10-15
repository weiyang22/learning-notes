---
title: 第04章-多任务学习 · Datawhalechina 补充
tags:
  - 经典推荐
  - 多任务学习
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/fun-rec（小麦书）上篇 ch3 多目标建模三小节与 torch-rechub MMoE 实现，补充任务依赖建模与损失融合的系统讲义
companion_to: 第04章-多任务学习.md
source: https://github.com/datawhalechina/fun-rec ; https://github.com/datawhalechina/torch-rechub
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 多任务学习 · Datawhalechina 对照补充

> 本文件是对同目录《第04章-多任务学习.md》的补充，内容整理自 [datawhalechina/fun-rec](https://github.com/datawhalechina/fun-rec)（小麦书）**上篇 ch3 多目标建模**三小节：基础结构演进 / 任务依赖建模 / 多目标损失融合，并衔接 [torch-rechub](https://github.com/datawhalechina/torch-rechub) 的 MMoE 实现。本地与《数学/第07章-02 案例 MMoE 多任务跷跷板》有数学案例；fun-rec 的增量是**结构谱系的演进逻辑与损失融合的工程配方**。

---

## 一、补充来源地图

| fun-rec 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch3 多目标·基础结构演进 | Shared-Bottom → MMoE → 谱系梳理 | 第04章正文 | 演进逻辑串讲 |
| ch3 多目标·任务依赖建模 | ESMM / PLE（逐渐分离的专家） | 第04章正文 | **核心补充**：任务间关系的三种假设 |
| ch3 多目标·损失融合 | 不确定性加权 / GradNorm / 帕累托 | 第04章正文 | **核心补充**：本地损失融合弱 |
| torch-rechub | MMoE 等实现 | 第07章-02（关联） | 代码对照 |

---

## 二、核心补充内容

### 2.1 任务间关系的三种假设与对应结构（fun-rec 多目标·任务依赖）

fun-rec 把多目标结构选择归结为对"任务间关系"的假设，这是比记结构更通用的选型逻辑：

| 任务关系假设 | 合适结构 | 代表 |
| :--- | :--- | :--- |
| 独立平行 | 共享底层 + 独立塔 | Shared-Bottom / MMoE |
| 序列依赖（曝光→点击→转化） | 级联建模目标空间 | ESMM（whole-space 建模解 CVR 样本选择偏差） |
| 复杂相关+冲突 | 专家分组 + 门控分离 | PLE（CGC：任务独享专家 + 共享专家，Progressive 分层提取） |

**跷跷板现象的定位**：MMoE 的负迁移（一任务涨另一任务跌）源于"共享专家被迫同时服务冲突梯度"——PLE 的独享专家就是给冲突梯度各开一条通路。本地《数学/第07章-02》的跷跷板数学案例可与本框架互引。

### 2.2 多目标损失融合的工程配方（fun-rec 多目标·损失融合）

本地章节偏结构；损失融合的三个层级的配方值得补全：

1. **静态加权**：$L=\sum_i w_i L_i$，权重靠网格搜；问题：不同任务 loss 量纲与收敛速度不同；
2. **不确定性加权**：用可学习的同方差不确定性 $\sigma_i$ 自动配权，$L=\sum_i \frac{1}{2\sigma_i^2}L_i + \log\sigma_i$——注意 $\log\sigma_i$ 项防 σ 无限增大，实现时对 σ 用 softplus 参数化保证正值；
3. **梯度级方法**：GradNorm（拉平各任务梯度范数的收敛速度）与 PCGrad（冲突梯度投影消除）——工业上 PCGrad 简单有效，代价是每步多算一次梯度内积；
4. **业务级**：多目标分数融合（如 $p\_cvr^\alpha \times p\_ctr^\beta$）放进重排/融合排序阶段调参，与训练侧加权解耦——fun-rec 提醒这是线上最常用的"最后一道旋钮"。

### 2.3 torch-rechub 对照实践

torch-rechub 提供 MMoE 等多任务模型实现，建议实验：同一数据集上跑 Shared-Bottom vs MMoE（专家数 2/4/8），各任务分别看 AUC，亲手复现一次"跷跷板"，再对照 2.1 的假设框架写结论。

---

## 三、可直接动手的实践

1. 复现不确定性加权：在双任务（CTR+CVR）小实验中把静态权重换成 σ 加权，观察两任务 loss 曲线的收敛同步性；
2. 用 PCGrad 处理一次梯度冲突（实现约 20 行：逐任务梯度、两两冲突投影），对比跷跷板指标；
3. 把 2.1 的"三种假设"表并入本地选型章节，为自己业务的多任务组合写一行假设判断。

---

## 四、延伸链接

- fun-rec 在线阅读：<https://datawhalechina.github.io/fun-rec/>
- torch-rechub：<https://github.com/datawhalechina/torch-rechub>
- 本地相关文件：数学/数学基础/第07章-02-案例MMoE多任务跷跷板（数学侧）、第18章-多场景推荐与隐式推理（多场景=多任务的姊妹问题）、GR研究/GR/第06章-多场景与多任务
