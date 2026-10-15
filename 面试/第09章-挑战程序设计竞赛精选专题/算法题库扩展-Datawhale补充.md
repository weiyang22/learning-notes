---
title: 算法题库扩展 · Datawhalechina 补充
tags:
  - 面试
  - 算法
  - 习题
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/huawei-od-python（400+ 场景题）与 team-learning-program（树/图讲义），为第09~15章竞赛专题补充习题源与理论地基
companion_to: 第09章-挑战程序设计竞赛精选专题/（整章）
source: https://github.com/datawhalechina/huawei-od-python ; https://github.com/datawhalechina/team-learning-program
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 算法题库扩展 · Datawhalechina 对照补充

> 本文件是对同目录《第09章-挑战程序设计竞赛精选专题》整章（并辐射第11~15章高级专题）的补充。本地第09~15章的共同薄弱点是**讲解充分、习题量不足**；两份 datawhale 资源恰好互补：[huawei-od-python](https://github.com/datawhalechina/huawei-od-python) 提供 400+ 道带完整 Python 题解的场景化题，[team-learning-program](https://github.com/datawhalechina/team-learning-program) 的数据结构与算法组队学习提供树/图高级专题的系统讲义。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地章节 | 补充价值 |
| :--- | :--- | :--- | :--- |
| huawei-od-python `choice100` | 100 道华为 OD 机试题（逻辑/模拟/数据结构） | 第09章各专题、第10章 | **核心补充**：场景化习题源 |
| huawei-od-python `others100` | 100 道综合题（字符串/DP/图/数学） | 第12、13章 | 习题补强 |
| huawei-od-python `questions200` | 200 道基础到进阶题 | 第10章实战刷题集 | 题量扩充 |
| team-learning-program `07 树.md` | 线段树/树状数组/Trie 系统讲义 | 第09章-04/05 | 讲义级理论地基 |
| team-learning-program `08 图.md` | 最短路/MST/Tarjan/网络流/二分图 | 第14章全部 | **核心补充**：高级图论讲义 |
| team-learning-program `05 数组与稀疏矩阵.md` | 数组专题 | 第09章-02 尺取法、第15章（弱关联） | 打底 |

---

## 二、核心补充内容

### 2.1 题库使用方案：把"讲义型"章节变成"讲义+题单"

第09章每个专题配一个 OD 题单（按考察点匹配），建议的练习闭环：

```text
读本地专题讲义 → 2 道热身题（questions200 对应考点）
→ 2 道场景题（choice100/others100 对应考点）
→ 写题解入本地第10章-11 错题本 → 周末限时重做
```

按本地章节分配的题单起点（题号以 huawei-od-python 仓库目录为准）：

| 本地专题 | 考察点 | OD 题库对应区 |
| :--- | :--- | :--- |
| 第09章-01 二分搜索的4种应用 | 二分答案/边界处理 | choice100 二分区 |
| 第09章-03 数学问题 | 模拟/进制/贪心数学 | others100 数学区 |
| 第09章-04/05 线段树/BIT | 区间统计 | others100 区间题 + `07 树.md` 讲义 |
| 第09章-06 博弈论 | 博弈/状态分析 | others100 博弈区 |
| 第09章-07 状压DP与矩阵快速幂 | 状压/快速幂 | questions200 进阶区 |
| 第09章-08 网络流基础 | 建模能力 | `08 图.md` 网络流节 + others100 图论区 |
| 第12章 字符串高级 | KMP/哈希/自动机 | others100 字符串区 |
| 第13章 数学专题 | 数论/组合/概率DP | others100 数学区 |
| 第14章 高级图论 | 最短路/Tarjan/二分图 | `08 图.md` 全章 + 图论题区 |

### 2.2 高级图论的理论地基（team-learning-program `08 图.md`）

第14章的四个文件（最短路/MST/Tarjan/网络流）是压缩讲义；team-learning-program 的图章是**组队学习粒度**的展开版，直接可用的增量：

- 最短路：Dijkstra 堆优化正确性证明、SPFA 判负环、分层图建图技巧；
- Tarjan：割点/桥/强连通分量的统一栈框架（与本地第14章-03 互为对照实现）；
- 网络流：Dinic 的当前弧优化、最小割建模套路（割点割边模型、最大权闭合子图）——**建模套路清单**是本地缺的部分，竞赛与 OD 题里"最难的一步是建图"。

### 2.3 场景题的独特价值

与 LeetCode 的"纯算法"题相比，OD 场景题的输入输出是**完整工程题面**（多行解析、边界脏数据、模拟流程），与算法岗笔试的真实形态更接近。补充建议：第10章-12 模拟面试脚本中加入 2~3 道 OD 场景题的限时环节（40 分钟读题+编码+自测），训练题面解析能力。

---

## 三、可直接动手的实践

1. 建立题单跟踪：在本目录新增个人题单表（题号/考点/首刷日期/重做日期），按 2.1 闭环执行两周；
2. 用 `08 图.md` 的网络流建模清单重做第09章-08 的例题，把每题的"建图三要素（节点含义/边含义/容量）"写成模板卡片，并入《附录B-代码模板大全》；
3. 把第09章-04/05 的线段树/BIT 实现与 `07 树.md` 讲义对照，补齐惰性标记的正确性论证。

---

## 四、延伸链接

- huawei-od-python：<https://github.com/datawhalechina/huawei-od-python>
- team-learning-program（DataStructureAndAlgorithm）：<https://github.com/datawhalechina/team-learning-program>
- 本地相关文件：第10章-LeetCode实战刷题集（题单落点）、第10章-11-错题本、附录B-代码模板大全（模板落点）、面试笔记/2-算法工程师通用面试-2026（笔试环节衔接）
