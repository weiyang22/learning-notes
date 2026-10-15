---
title: RL 与 Sim2Real · Datawhalechina 补充
tags:
  - 具身智能
  - 强化学习
  - Sim2Real
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/easy-rl（PPO/SAC 运动控制）与 every-embodied（大规模并行训练与域随机化），补充 Isaac 并行 RL 范式与 Sim2Real 工具箱
companion_to: 第06章-强化学习与Sim2Real/
source: https://github.com/datawhalechina/easy-rl ; https://github.com/datawhalechina/every-embodied
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# RL 与 Sim2Real · Datawhalechina 对照补充

> 本文件是对同目录《第06章-强化学习与Sim2Real》的补充，内容整理自 [easy-rl](https://github.com/datawhalechina/easy-rl)（蘑菇书 PPO/SAC 章，运动控制的算法地基）与 [every-embodied](https://github.com/datawhalechina/every-embodied)（大规模并行训练、Sim2Real 话题）。本地章讲 Sim2Real 方法本身；两份资料补**训练范式（GPU 并行）与算法地基**。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| easy-rl ch6/ch8/ch9 | PPO/连续控制/SAC | RL 算法地基 | **核心补充**：运动控制标配算法推导 |
| every-embodied | Isaac Gym/Lab 大规模并行 | 训练范式节 | **核心补充**：GPU 并行 RL 范式 |
| every-embodied | Sim2Real gap 来源与域随机化 | 04-Sim2Real 正文（关联） | 工具箱化 |

---

## 二、核心补充内容

### 2.1 运动控制的算法标配：PPO + SAC（easy-rl）

机器人 RL 的两大标配与其分工（把《RL 教程对照》的谱系落到运动控制）：

| 算法 | 在运动控制中的角色 | 关键配置 |
| :--- | :--- | :--- |
| PPO（on-policy） | 快速迭代、仿真里海量并行样本下首选 | clip ε=0.2、GAE(λ≈0.95)、奖励归一化 |
| SAC（off-policy + 最大熵） | 真机/低样本场景、探索需求高 | 自动温度 α、replay buffer 常态化 |

**奖励设计的运动控制惯例**：任务奖励（到达/抓取）+ 形状奖励（势能引导）+ 惩罚项（能耗/关节加速度/摔倒）——惩罚系数是"行为塑形"的主旋钮，任意一项过大会压过任务奖励。

### 2.2 GPU 大规模并行训练范式（every-embodied）

传统 CPU 并行（MuJoCo 多进程）到 GPU 并行（Isaac Gym/Lab）的范式变化：

1. **物理仿真进 GPU**：数千环境同时步进，采样吞吐提升 2~3 个数量级——on-policy 算法（PPO）因此成为可选项（样本效率让位于吞吐）；
2. **观测/奖励向量化**：奖励函数与终止条件全部写成张量运算（无 Python 循环），这是 Isaac 版 reward 的写法纪律；
3. **课程学习**：任务难度随成功率自动升级（Terrain 课程：地形难度逐级解锁）——大规模并行下课程学习从技巧变为标配。

### 2.3 Sim2Real 的 gap 来源清单与缩小工具箱

本地第06章-04 讲方法；every-embodied 补一张"来源→工具"对照：

| Gap 来源 | 缩小手段 |
| :--- | :--- |
| 物理参数漂移（质量/摩擦/刚度） | 域随机化 DR（训练时随机化参数区间） |
| 传感器噪声/延迟 | 观测噪声注入 + 延迟建模 |
| 执行器建模误差 | 执行器网络/系统辨识 + 力矩限幅 |
| 视觉差异 | 渲染随机化 / 走状态策略 + 视觉模块分训 |
| 接触动力学 | 提高仿真接触求解精度 + 真机微调 |

**实践顺序**：先量化 gap（哪个环节误差最大）→ 对症选工具 → 真机小流量验证——盲开 DR 全家桶会牺牲策略精度。

---

## 三、可直接动手的实践

1. 用 Isaac Gym/Lab（或其仿真回放）跑官方 anymal/搬运示例，观察数千环境的并行曲线与 PPO 收敛速度；
2. 做一次 DR 消融：摩擦系数随机化开/关两组训练，比较仿真内性能与（如有条件）迁移鲁棒性代理指标；
3. 把本地第06章-04 的方法逐个挂到 2.3 表格的对应行，查漏补缺。

---

## 四、延伸链接

- easy-rl（PPO/SAC 章）：<https://datawhalechina.github.io/easy-rl/>
- every-embodied：<https://github.com/datawhalechina/every-embodied>
- 本地相关文件：第06章正文、第06章-04-Sim2Real、搜推广与生成式推荐/RL/RL/RL教程对照-Datawhale补充.md（算法谱系）、第07章仿真器补充（训练场）、第05章模仿学习补充（数据线）
