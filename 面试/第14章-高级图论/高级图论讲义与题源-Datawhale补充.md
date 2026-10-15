---
title: 高级图论讲义与题源 · Datawhalechina 补充
tags:
  - 面试
  - 图论
  - 算法
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/team-learning-program《08 图》讲义与 huawei-od-python 图论题区，补充最短路/MST/Tarjan/网络流的展开讲义与建模套路
companion_to: 第14章-高级图论/（整章 4 节）
source: https://github.com/datawhalechina/team-learning-program ; https://github.com/datawhalechina/huawei-od-python
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 高级图论讲义与题源 · Datawhalechina 对照补充

> 本文件是对同目录《第14章-高级图论》整章（最短路 / 最小生成树 / 强连通分量与 Tarjan / 网络流与二分图匹配共 4 节）的补充，内容整理自 [team-learning-program](https://github.com/datawhalechina/team-learning-program) 组队学习《DataStructureAndAlgorithm/**08 图**》讲义与 [huawei-od-python](https://github.com/datawhalechina/huawei-od-python) 图论题区。本地四节是压缩版讲义；team-learning-program 的展开版补**正确性证明与建模套路清单**——竞赛与笔试中"最难的一步是建图"。

---

## 一、补充来源地图

| team-learning-program 08 图 内容 | 对应本地节 | 补充价值 |
| :--- | :--- | :--- |
| 图的存储（邻接矩阵/表/链式前向星） | 全章前置 | 实现选型 |
| Dijkstra（堆优化）/ Bellman-Ford / SPFA / Floyd | 第14章-01 | **核心补充**：正确性证明、判负环、分层图 |
| Prim / Kruskal（并查集） | 第14章-02 | 贪心选择性质的证明 |
| Tarjan（割点/桥/SCC 统一栈框架） | 第14章-03 | **核心补充**：low 数组统一推导 |
| 二分图（染色/匈牙利） | 第14章-04 | 匹配与最小点覆盖的对偶（König 定理） |
| 网络流（FF/EK/Dinic + 建模套路） | 第14章-04 | **核心补充**：建模三要素模板 |

---

## 二、核心补充内容

### 2.1 最短路的证明与进阶技巧（08 图·最短路节）

- **Dijkstra 贪心正确性**（归纳证明）：取当前未确定集中 dist 最小的 u，任何其他路径进入 u 的方案都已被松弛覆盖——前提**边权非负**；负权下该性质失效，改用 Bellman-Ford/SPFA；
- **SPFA 判负环**：记录每点入队次数，>n 即有负环（或用最长路径上限）；
- **分层图建模**：状态含"剩余资源"维（如免费次数/油量），把 (节点， 状态) 拆成多层图跑最短路——笔试高频套路，本地第14章-01 未覆盖；
- **Floyd 的正确打开方式**：全源、可处理负权（无负环）、k 循环在外层——顺序错了结果错，但编译不报错。

### 2.2 Tarjan 的统一 low 框架（08 图·连通性节）

割点、桥、强连通分量共用同一 DFS 框架，只是判定条件不同——统一记忆比三个算法分开记可靠：

```text
dfn[u] = 首次访问时间戳;  low[u] = u 能回溯到的最早时间戳
low[u] = min(dfn[u], min(low[v] for v in 子树), dfn[w] for w in 回边)
```

| 目标 | 根节点判定 | 非根节点判定 |
| :--- | :--- | :--- |
| 割点 | 子树数 ≥ 2 | 存在子 v 使 low[v] ≥ dfn[u] |
| 桥 | 子树数 ≥ 2 | low[v] > dfn[u]（严格大于） |
| SCC | — | low[u] == dfn[u] 时弹出栈 |

**易错点**：桥的回边不能用父边（要排除父节点那次访问）；有重边时父边排除需按边编号。

### 2.3 网络流建模三要素模板（08 图·网络流节）

Dinic（当前弧 + 分层图，O(V²E)）是默认实现；真正区分水平的是**建图**，模板卡片建议并入本地《附录B-代码模板大全》：

| 建模对象 | 节点含义 | 边与容量 | 常见问题 |
| :--- | :--- | :--- | :--- |
| 二分图匹配 | 左右部点 | 源→左 cap 1，右→汇 cap 1，可达边 cap 1 | 最大匹配 |
| 最小点覆盖/最大独立集 | 同上 | König：最大匹配 = 最小点覆盖 | 选择问题 |
| 最大权闭合子图 | 项目/依赖 | 正权连源、负权连汇、依赖边 ∞ | 收益最大化（总正权 − 最小割） |
| 上下界/费用流 | 带供需 | 下界改造 + SPFA 找增广 | 区间覆盖、带费用调度 |

### 2.4 题源联动（huawei-od-python 图论题区）

与《第09章 算法题库扩展-Datawhale补充.md》的题单机制一致：本章配图论题单——Dijkstra/拓扑序类（choice100）、连通与网络流建模类（others100），按"讲义 → 2 热身 + 2 场景题 → 错题本"闭环执行。

---

## 三、可直接动手的实践

1. 用统一 Tarjan 框架（2.2）一次实现三个判定，各配一个手造样例（含重边陷阱样例）；
2. 实现带当前弧优化的 Dinic，跑 2.3 的"最大权闭合子图"模板题（如"项目-设备租赁"模型）；
3. 做 2 道分层图最短路题（如"免费 k 次"），把建图层高与边数公式写进模板卡片。

---

## 四、延伸链接

- team-learning-program（DataStructureAndAlgorithm/08 图）：<https://github.com/datawhalechina/team-learning-program>
- huawei-od-python：<https://github.com/datawhalechina/huawei-od-python>
- 本地相关文件：第14章四节正文、第09章-08-网络流基础（Dinic 基础）、第09章-04/05（数据结构配套）、附录B-代码模板大全（模板落点）、面试/第09章…/算法题库扩展-Datawhale补充.md（题单机制）
