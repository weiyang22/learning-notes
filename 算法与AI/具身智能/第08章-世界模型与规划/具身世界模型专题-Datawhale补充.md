---
title: 具身世界模型专题 · Datawhalechina 补充
tags:
  - 具身智能
  - 世界模型
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/every-embodied 第17章具身世界模型（9 篇导读）与 learn-world-model 课程，补充本地第08章世界模型的前沿跟踪与基准实操
companion_to: 第08章-世界模型与规划/（整章）
source: https://github.com/datawhalechina/every-embodied ; https://github.com/datawhalechina/learn-world-model
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 具身世界模型专题 · Datawhalechina 对照补充

> 本文件是对同目录《第08章-世界模型与规划》整章的补充。本地第08章以经典体系为主（世界模型概论 / DreamerV3 / IRIS·DayDreamer / MPC / Plan2Explore / 世界模型+VLA）；两份 datawhale 资源补足前沿：[every-embodied 第 17 章具身世界模型](https://github.com/datawhalechina/every-embodied)（9 篇 2025 年模型导读）与 [learn-world-model](https://github.com/datawhalechina/learn-world-model)（Datawhale 世界模型课程，含架构深拆）。

---

## 一、补充来源地图

| 来源 | 内容 | 与本地第08章的关系 |
| :--- | :--- | :--- |
| learn-world-model 课程 | 世界模型课程主页 + 架构深拆（Architecture Deep Dive） | 对应第08章-01/02，补系统课程视角 |
| every-embodied 17 章：LeWM 分析解读与实验复现 | Datawhale 自维护的世界模型实验 | 对应第08章-02，补可复现实验 |
| RAW-Dream 任务无关世界模型强化 VLA | 世界模型 × VLA | 对应第08章-06，前沿版 |
| WoG 条件空间世界模型 | 条件生成视角 | 补充 |
| WALL-WM 事件级世界动作模型 | 事件级建模 | 补充 |
| RoboDream 可组合世界模型数据合成 | 数据合成 | 补充：世界模型当"数据引擎" |
| Gamma-World 多智能体世界模型 | 多智能体 | 补充 |
| BWM 动作条件世界模型 | 动作条件化 | 补充 |
| τ0-WM 统一视频-动作世界模型 | 视频+动作统一 | 补充：与 GenRec 视频生成同构 |
| GE-Sim 2.0 闭环视频世界模拟器 | 闭环模拟 | 补充 |
| 本地第07章-仿真器与基准（关联） | MuJoCo/Isaac/Habitat 等 | every-embodied 09 章补 LIBERO/SimplerENV 实操 |

---

## 二、核心补充内容

### 2.1 世界模型在具身智能中的三种角色（基于两份来源归纳）

本地第08章主要讲"世界模型帮 agent 想象"；every-embodied 17 章的 9 篇导读揭示前沿已分化出三种角色，建议原文档补充这个分类：

1. **想象器（Imaginator）**——本地第08章主线：在隐空间 rollout 评估动作后果（DreamerV3/RSSM 谱系）；
2. **数据引擎（Data Engine）**——RoboDream 代表：用世界模型合成训练数据，缓解机器人数据稀缺；这是 2025 年工业界最活跃的用法；
3. **闭环模拟器（Closed-loop Simulator）**——GE-Sim 2.0 代表：替代昂贵的物理仿真做大规模评估（Sim2Real Gap 的另一种解法，与本地第06章-04 Sim2Real 衔接）。

### 2.2 2025 具身世界模型速览表（every-embodied 17 章）

| 模型 | 一句话定位 | 值得读的点 |
| :--- | :--- | :--- |
| LeWM | Datawhale 自维护实验复现 | 从零跑通世界模型训练的完整实验 |
| RAW-Dream | 任务无关世界模型强化 VLA | 世界模型作为 VLA 的"预训练器" |
| WoG | 条件空间世界模型 | 条件化隐空间的构造方式 |
| WALL-WM | 事件级世界动作模型 | 时间粒度从帧级到事件级 |
| RoboDream | 可组合世界模型数据合成 | 数据引擎角色的代表作 |
| Gamma-World | 多智能体世界模型 | 多主体动力学建模 |
| BWM | 动作条件世界模型 | 动作作为生成条件的最小设计 |
| τ0-WM | 统一视频-动作世界模型 | 视频生成与动作生成的统一 token 空间 |
| GE-Sim 2.0 | 闭环视频世界模拟器 | 用视频模型做评估器 |

### 2.3 基准与数据：LIBERO / SimplerENV（every-embodied 09 章联动）

every-embodied 的第 9 章（具身智能数据及评估基准）是本地第07章"仿真器与基准"的直接补充：

- **LIBERO**： lifelong learning 操作基准（4 个任务族：Spatial/Object/Goal/100），VLA 论文的标配评测场；every-embodied 给出环境安装与评测脚本实操；
- **SimplerENV**：Google 原真机环境（RT-1/RT-2/Octo 系）的 Simpler 轻量复刻，**用渲染近似替代真机评测**——与"世界模型当评估器"的思想同源，是理解 Sim2Real 评测偏差的低成本入口。

### 2.4 learn-world-model 课程的使用方式

learn-world-model（Alpha Preview）按"课程主页 → Lecture 页 → 架构深拆"组织，适合作为第08章的**配套课程**而非替代：先读本章正文建立 DreamerV3 数学框架，再用课程的 Architecture Deep Dive 对照现代实现（RSSM 之外的架构变体），最后落到 every-embodied 的 LeWM 复现实验。

---

## 三、可直接动手的实践

1. 跑通 every-embodied 的 LeWM 复现实验，与本地第08章附录的 DreamerV3 实现做 loss/rollout 质量对比；
2. 在 SimplerENV 上评测一个开源 VLA（衔接 VLA模型清单补充文件的复现顺序），记录其与论文报告值的差距，分析 Sim2Real 评测偏差来源；
3. 用 RoboDream 思路做一个最小实验：训练小型世界模型为本地第05章模仿学习的数据集做增强，对比 BC 成功率变化。

---

## 四、延伸链接

- every-embodied 17 章：<https://github.com/datawhalechina/every-embodied/tree/main/17-具身世界模型>
- learn-world-model 在线课程：<https://datawhalechina.github.io/learn-world-model>
- 本地相关文件：第08章-02-DreamerV3（数学框架）、第08章-06-世界模型+VLA（角色 1 的 VLA 衔接）、第06章-04-Sim2Real（角色 3 的评估衔接）、第07章仿真器目录
