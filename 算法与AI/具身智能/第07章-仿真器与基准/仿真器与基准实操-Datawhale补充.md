---
title: 仿真器与基准实操 · Datawhalechina 补充
tags:
  - 具身智能
  - 仿真
  - 基准
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/every-embodied 第09章（LIBERO/SimplerENV 基准实操）与第03章硬件线，补充基准选型、安装与评测流程
companion_to: 第07章-仿真器与基准/（整章）
source: https://github.com/datawhalechina/every-embodied
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 仿真器与基准实操 · Datawhalechina 对照补充

> 本文件是对同目录《第07章-仿真器与基准》整章（MuJoCo / Isaac Sim / Habitat / ManiSkill / RoboCasa / SAPIEN 六节）的补充，内容整理自 [every-embodied](https://github.com/datawhalechina/every-embodied) **第 09 章 具身智能数据及评估基准 benchmark**（LIBERO / SimplerENV 实操）与第 03 章（硬件/LeRobot 线）。本地第07章按"仿真器"组织；every-embodied 的增量是按"**评测基准**"组织的实操——论文里的成功率数字怎么复现出来。

---

## 一、补充来源地图

| every-embodied 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| 09 章 LIBERO | 永续操作基准：环境安装、任务族、评测脚本 | 第07章各节 + 第08章 | **核心补充**：VLA 标配评测场实操 |
| 09 章 SimplerENV | Google 原真机任务的 Simpler 复刻评测 | 第07章 | **核心补充**：真机任务的低成本评测 |
| 03 章 LeRobot 遥操作 | 真机数据采集 | 第05章模仿学习（关联） | 数据采集 → 评测的闭环 |
| 19 章 组队学习路径 | 零基础入门路径 v6 | 第01章（关联） | 学习顺序参考 |

---

## 二、核心补充内容

### 2.1 仿真器（训练场）vs 基准（考场）的分层

本地第07章按仿真器组织；every-embodied 视角补一层分类，两者关系建议并入原文档开篇：

| 层 | 角色 | 代表 | 回答的问题 |
| :--- | :--- | :--- | :--- |
| 物理仿真器 | 提供物理与渲染 | MuJoCo / Isaac / SAPIEN | 场景怎么建、怎么交互 |
| 操作基准 | 标准任务+指标+协议 | **LIBERO / ManiSkill** | 模型能力如何横向比较 |
| 真机替代评测 | 用渲染复刻真机任务 | **SimplerENV** | 不上真机能否估计真机表现 |
| 导航基准 | 室内导航任务 | Habitat / VLN-CE | 长程空间智能 |

### 2.2 LIBERO：VLA 论文的标配考场（every-embodied 09 章）

- **四个任务族**：Spatial（空间关系）/ Object（物体泛化）/ Goal（目标泛化）/ 100（长程综合，100 个任务）——分别测一种泛化轴，读论文时先看它报的是哪个族；
- **实操要点**：安装（conda 环境 + MuJoCo 渲染依赖）、任务加载 API、把策略接进统一的 rollout 评测循环（与环境 reset/step 协议）、成功判据（物体目标位姿阈值）；
- **与 VLA 补充文件联动**：SmolVLA / OpenVLA / DiT4DiT 的复现都在 LIBERO 上出数——本库《第04章 VLA 模型清单补充》的复现顺序以此为统一评测场。

### 2.3 SimplerENV：真机评测的低成本替代（every-embodied 09 章）

- **动机**：RT-1/RT-2/Octo 等论文的数字出自 Google 真机环境，外部无法复现——SimplerENV 用渲染复刻这些任务的**近似评测**，让开源社区能对齐；
- **用法**：载入原研究公布的 checkpoint 或本地模型 → 跑指定任务集 → 报告 visual matching / variant matching 两档成功率；
- **解读纪律**：SimplerENV 分数 ≠ 真机分数，它是**相对排序工具**——用于模型间横向比较，绝对值与论文真机数字存在系统性 gap（这本身是理解 Sim2Real 评测偏差的活教材，与本地第06章-04 衔接）。

### 2.4 选型决策表（仿真器 × 基准的组合建议）

| 你的目标 | 推荐组合 |
| :--- | :--- |
| 复现 VLA 论文数字 | LIBERO（操作）或 SimplerENV（对齐 RT 系） |
| 训练自己的操作策略 | MuJoCo/ManiSkill 训练 → LIBERO 评测 |
| 长程导航 | Habitat / VLN-CE |
| 大规模并行训练 | Isaac Sim/Lab（GPU 并行） |
| 真机落地前验证 | 仿真评测（LIBERO/Simpler）→ LeRobot 真机小流量 |

---

## 三、可直接动手的实践

1. 跑通 LIBERO 安装与一个任务族的 rollout，用随机策略出基线成功率，记录评测循环的接口（reset/step/success 判定）；
2. 在 SimplerENV 上评测任一开源 VLA（衔接 VLA 清单补充文件的复现顺序），记录与论文数字的 gap 并归因（渲染差异/分辨率/动作空间）；
3. 画一张自己的"训练场 → 考场"路线图（从 2.4 选型表出发），标注每段的产出物。

---

## 四、延伸链接

- every-embodied 09 章：<https://github.com/datawhalechina/every-embodied/tree/main/09-具身智能数据及评估基准benchmark>
- LIBERO 官方：<https://libero-project.github.io/>；SimplerENV：<https://simpler-env.github.io/>
- 本地相关文件：第07章六节正文（仿真器本体）、第04章 VLA 清单补充（评测消费者）、第06章-04-Sim2Real（gap 归因）、第08章具身世界模型补充（GE-Sim 当评估器的思想源头）
