---
title: RL教程对照 · Datawhalechina 补充
tags:
  - RL
  - 强化学习
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/easy-rl（蘑菇书）与 joyrl-book，为 RL 教程 9 章补充经典 RL 教材的系统化对照与代码实现入口
companion_to: RL/RL/（整个子目录 9 章）
source: https://github.com/datawhalechina/easy-rl ; https://github.com/datawhalechina/joyrl-book
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# RL 教程对照 · Datawhalechina 对照补充

> 本文件是对同目录 RL 教程 9 章（1-RL基础与数学 → 9-RL未来方向）的补充，内容整理自 [datawhalechina/easy-rl](https://github.com/datawhalechina/easy-rl)（蘑菇书，15k+ stars，RL 中文经典教材）与 [joyrl-book](https://github.com/datawhalechina/joyrl-book)（JoyRL 图书配套）。本地 RL 教程强在"RL → LLM/GenRec 的应用线"；蘑菇书的差异化价值是**基础与经典算法的教材级推导**（第 1~9 章），正好补本地"1-RL基础与数学、2-经典RL算法详解"两章的推导深度。

---

## 一、补充来源地图

| 蘑菇书章节 | 内容 | 对应本地文件 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch1~ch2 强化学习基础 / 马尔可夫决策过程 | 状态/动作/奖励/贝尔曼方程 | 1-RL基础与数学.md | **核心补充**：贝尔曼方程组推导与收敛性 |
| ch3 表格型方法 | Q-learning / Sarsa 探索策略 | 2-经典RL算法详解.md | 值方法的地基 |
| ch4~ch5 DQN 及其变种 | Double/Dueling/Prioritized | 2-经典RL算法详解.md | DQN 谱系系统性 |
| ch6 策略梯度 | REINFORCE、基线 | 2-经典RL算法详解.md | PG 推导（LLM RLHF 的数学前置） |
| ch7 DDPG | 连续控制 | 2-经典RL算法详解.md | 机器人线衔接 |
| ch8 PPO | 重要性采样、裁剪目标 | 3-RL在LLM中的经典方法.md | **与本地 PPO 裁剪数学案例互引** |
| ch9 SAC | 最大熵 RL | 6-RL工程实践与部署.md（关联） | 稳定训练配方 |
| joyrl-book | JoyRL 代码库配套讲义 | 全部 | 可运行实现 |

---

## 二、核心补充内容

### 2.1 贝尔曼方程的推导链（蘑菇书 ch2）——本地第1章的地基补强

蘑菇书对 RL 数学地基的标准推导链，建议并入 1-RL基础与数学.md：

1. **回报定义**：$G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$，γ 的两级作用（ discount + 收敛保证 ）；
2. **值函数分解**：$v^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t=s]$，对 $G_t$ 按"第一步奖励 + 后续回报"拆分得贝尔曼期望方程：
   $$v^\pi(s) = \sum_a \pi(a\mid s)\left[ r(s,a) + \gamma \sum_{s'} p(s'\mid s,a) v^\pi(s') \right]$$
3. **最优性原理（Bellman optimality）**：最优策略的子策略仍最优 → 贝尔曼最优方程 $v^*(s)=\max_a [r+\gamma \sum_{s'}p v^*(s')]$，这是 Q-learning 收敛性的根；
4. **与 LLM RL 的衔接**：RLHF 中 reward = 过程奖励 + 结果奖励折扣的设计直接对应 2 的拆分；本地《数学/第07章-05 RLHF 在搜推》是这条链的应用案例。

### 2.2 从 Q-learning 到 PPO 的"两条主线"叙事（蘑菇书 ch3~ch8）

蘑菇书把经典算法组织成**值方法与策略方法**两条主线，比按时间罗列更清晰：

- **值线**：Q-learning（off-policy，TD 误差）→ DQN（函数近似 + 目标网络 + 经验回放）→ Double（过估计）/ Dueling（V-A 分解）/ Prioritized（采样优先级）；
- **策略线**：REINFORCE（MC 梯度，方差大）→ 加基线 → Actor-Critic → A2C/A3C → TRPO（信任域）→ **PPO（裁剪的重要性采样目标）**；
- 两线在 **SAC** 汇合（最大熵 + off-policy actor-critic）。

**PPO 的关键一步**（蘑菇书 ch8，与本地《数学/第07章-06 PPO 裁剪目标》《3-RL在LLM中的经典方法》互引）：重要性采样比值 $r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ 的目标是无偏的，但比值偏离 1 时方差爆炸——裁剪 $L=\mathbb{E}[\min(r_t A_t, \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)]$ 用有偏换稳定。GRPO 的"组内归一化优势"（本地第07章-07）就是在这个骨架上替换优势估计。

### 2.3 joyrl-book 的代码入口

joyrl-book 提供与教材配套的可运行实现（DQN/PPO/SAC 等），与本地《6-RL工程实践与部署》的部署视角互补：先在 joyrl 上跑通算法（CartPole/Pendulum 级），再读本地的工程化章（环境并行、模型服务化）。

---

## 三、可直接动手的实践

1. 手推贝尔曼期望/最优方程（2.1），并写出 4 状态网格世界的值迭代解；
2. 用 joyrl 跑 PPO（CartPole），把训练曲线的 clip 比例与 KL 曲线记录下来，对照本地 PPO 数学案例理解"裁剪何时生效"；
3. 画出值线/策略线的算法谱系图（2.2），把本地 3~9 章的 LLM/GenRec 方法挂到对应节点上，检查应用线是否有"地基缺失"的节点。

---

## 四、延伸链接

- 蘑菇书在线阅读：<https://datawhalechina.github.io/easy-rl/>
- joyrl-book：<https://github.com/datawhalechina/joyrl-book>
- 本地相关文件：3-RL在LLM中的经典方法.md、7-论文案例-OneRec系列RL.md（RecZero/RecOne 衔接 GR 第08章补充文件）、数学/数学基础/第07章-05/06/07（RLHF/PPO/GRPO 数学案例）、算法与AI/具身智能/第06章（机器人 RL）
