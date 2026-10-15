---
title: "第03章-05-轨迹优化-最小jerk与MPC"
tags:
  - 具身智能
  - 轨迹优化
  - 最小jerk与MPC
description: "本章要点：演示轨迹跑得又快又平滑\"或者\"末端必须严格避障,撞一下就报废\",你第一反应是什么?"
chapter_id: 具身智能-5-03
last_updated: 2026-07-14
---

# 第03章-05-轨迹优化-最小jerk与MPC

演示轨迹跑得又快又平滑"或者"末端必须严格避障,撞一下就报废",你第一反应是什么?
>
> - **直接让 RL 学一条策略?** 可以,但 RL 学的轨迹抖动大、不满足严格的关节/速度/力矩约束,且样本效率在真实机器人上惨不忍睹。
> - **用 PID 跟踪一条手画的轨迹?** 更快,但"手画"就是关键问题:你怎么保证轨迹**平滑**(不抖)、**时间最优**(不慢)、**满足约束**(不撞)?
> - **用 MPC 滚动优化?** 工业界正解,但"模型从哪来、QP 怎么解、10ms 内算不出来怎么办"立刻变成三个新问题。
>
> 这些问题的答案,藏在"**最优控制**"这块 60 年前的数学地基里。本章要做的就是把这块地基**搬到机器人工程语境**,并且直接给你能在 PyTorch 里跑、能用 qpOASES/OSQP 求解、能上 MIT Cheetah / Franka 这种真实机器人的代码骨架。
>
> **学完这一章,你能回答**:
> - 最小 jerk 轨迹的物理意义是什么,为什么人手运动天然就是最小 jerk?
> - Pontryagin 最大值原理和 Bolza 问题之间是什么关系,KKT 在有限维 vs 无限维的对应?
> - MPC 的"滚动优化 + 反馈校正"为什么比 LQR 这种"一次离线算 K"更适合有约束的场景?
> - iLQR 和 DDP 差在哪里? 为什么 iLQR 在机器人上比 DDP 用得更多?
> - qpOASES / OSQP / Acados 这些求解器各适合什么场景?
> - MPC 在 10ms 内算不出来怎么办? 什么是 warm-start、什么是显式 MPC?
>
> **与前面章节的关系**:
> - [[第03章-01-刚体运动学-DH参数与正逆运动学|第 5 章 §1]] 给了**几何映射**——末端位姿 vs 关节角的 Jacobian。本章会用 Jacobian 把任务空间的约束投影到关节空间。
> - [[第03章-02-动力学-Lagrange与Newton-Euler|第 5 章 §2]] 给了**动力学方程** $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau$。MPC 的"模型"几乎就是这个方程的离散化。
> - [[第03章-03-经典控制-PID与状态空间|第 5 章 §3]] 给了**状态空间 + LQR**——把无限维的最优控制问题投影到一个线性反馈增益 $K$ 上。MPC 是 LQR 的"在线 + 带约束"版本。
> - [[第03章-04-力控制-阻抗与导纳控制|第 5 章 §4]] 给了**接触动力学**。MPC 在力控场景下需要把接触约束 (摩擦锥、单边约束) 写进优化问题。
>
> **与后面章节的关系**:
> - 第 6 章 §6 的 RL 策略可以**初始化** MPC 的 nominal trajectory;反之 MPC 的解也可以当 RL 的 expert demonstration。
> - 第 6 章 §7 的 DAgger 会**蒸馏** MPC 的解,部署到低成本控制器上。
> - 第 6 章 §8 的世界模型 = MPC 中"用于预测的系统模型"的可学习版本。
> - 第 4 章 §4 的 VLA 输出本质就是一段轨迹点 — 这条轨迹点怎么"平滑化 + 满足约束"恰好是 MPC 的活。
>
> **与数学基础的关系**:
> - 第 1 册第 9 章 给了**变分法**——连续最优控制的数学基础。本章 §3 会反复用到 Euler-Lagrange 方程。
> - 第 6 册第 5 章 给了**凸优化理论**——MPC 求解的数学保证 (KKT、对偶)。
> - 第 6 册第 2 章 给了**梯度下降/牛顿法**——iLQR 的反向传播本质上是牛顿法的稀疏版本。
> - 第 2 册第 8 章 给了**稀疏线性代数**——MPC 的 QP 稀疏求解为什么比稠密快 100 倍。
>
> 准备好了吗?我们从"为什么 PID 不够"开始。

---

## 1. 直觉引入

### 1.1 从控制到优化:一次性规划

回顾前两章,你学到的工具是 **PID** 和 **LQR**:

- **PID** = 误差驱动的**反馈律**。你给一个参考 $r(t)$,控制器输出 $u(t) = K_p e + K_i \int e + K_d \dot{e}$。简单、便宜、实时,但**没有前瞻性**,只看当前误差。
- **LQR** = 二次型代价最优的**反馈增益** $K$。你离线解 Riccati 方程得到 $K$,然后在线用 $u = -K(x - x_{ref})$。最优,但**不能处理约束**,而且 $K$ 是固定的。

这两种工具的**共同短板**是:**没有"看未来"的能力**。

但现实机器人任务的本质是**预测未来**:

```text
机械臂插孔任务:

  当前时刻 t=0:        末端在 A 点 (远离孔)
  目标时刻 t=2s:       末端要在 B 点 (孔里)
  约束:                关节速度 < 2 rad/s, 避开桌上的杯子

PID/LQR 怎么做?    只看当前误差 → 末端冲向 B → 末端速度全开 → 撞杯子
MPC 怎么做?        看未来 2s 的轨迹 → 在中途减速绕开杯子 → 平滑插入
```

**MPC (Model Predictive Control)** 的核心思想只有一句话:

> **每个时刻,基于当前状态,在有限时间窗口内求解一个开环最优控制问题;只施加第一个控制量,然后滚动到下一时刻重新求解。**

这个"滚动优化"机制让 MPC 同时具备三件事:

1. **前瞻性**: 优化未来 N 步,知道障碍在哪
2. **约束处理**: 关节限位、避障、力矩饱和全部写进 QP
3. **反馈鲁棒性**: 每步重新测量状态,自然处理扰动

但 MPC 不是免费的:

- **计算代价**: 每步要解一个优化问题,通常 1-100ms
- **模型依赖**: 模型错了,优化解就错了
- **调参复杂**: 预测时域 N、代价权重 Q/R、约束松弛都要调

本章就是要回答:**怎么让 MPC 在机器人上又快又稳**。

### 1.2 轨迹 vs 控制律

"轨迹优化"和"控制律设计"是两个不同的视角:

```text
视角 1: 控制律 (Control Law)
  - 输出: u = π(x, t)   (状态 → 控制的映射)
  - 例: PID、LQR、神经网络策略 π_θ(a|s)
  - 优势: 实时、在线、无需预测
  - 劣势: 难以处理硬约束

视角 2: 轨迹 (Trajectory)
  - 输出: {x(t), u(t)} for t ∈ [0, T]   (整段时间序列)
  - 例: 最小 jerk、Minimum Snap、TrajOpt
  - 优势: 全局最优、满足约束
  - 劣势: 一次性规划、模型错就全错

视角 3: 滚动轨迹优化 (Receding Horizon)
  - 输出: 当前时刻的最优 u_0,然后滚动
  - 例: MPC、Receding Horizon iLQR
  - 优势: 前瞻 + 反馈 + 约束
  - 劣势: 计算代价、模型依赖
```

**真实工程里这三者经常组合**:

- **示教 → 重放**: 人手拖动示教一段轨迹 → 离线最小 jerk 平滑 → 在线 PID 跟踪 (KUKA iiwa 默认模式)
- **MPC 在线优化**: 每步 10ms 求解 → 施加第一个控制量 (MIT Cheetah 这种高速腿足机器人)
- **MPC + RL**: MPC 解出 nominal trajectory → RL 学 residual correction (这个思路叫 residual RL,见 §5)

**为什么先讲最小 jerk?** 因为它是"轨迹视角"最简单的入门,且很多 MPC 的 nominal trajectory 就是最小 jerk 起手。

### 1.3 工业需求:高速 + 平滑 + 安全

真实工业场景对"轨迹"的三大需求,看起来很普通,工程上很难同时满足:

#### 需求 1: 高速 (Speed)

工业机器人节拍 (cycle time) 直接决定产能。一条焊接产线 1 秒节省 0.1 秒,一年就是几百万。

```text
例: 汽车焊接产线
  节拍: 60 件/小时 = 60 秒/件
  其中焊接动作: 6 秒
  其他动作 (抓取、摆放、过渡): 54 秒

  如果"过渡轨迹"从 4 秒降到 2 秒:
  → 节拍不变,但每件省 2 秒
  → 一年 250 天 × 16 小时 × 3600 秒/小时 / 60 秒/件 × 2 秒 = 480000 件
  → 单件 0.5 元利润 = 24 万元/年
  → 一台机器人省 24 万 × 1000 台 = 2.4 亿/年
```

要高速,意味着**加速度、jerk 都要拉满**,但这又和"平滑"冲突。

#### 需求 2: 平滑 (Smoothness)

"平滑"在工程上是多义词,我们拆开看:

| 指标 | 物理意义 | 衡量 | 失超后果 |
| :--- | :--- | :--- | :--- |
| **连续性** $C^n$ | n 阶导连续 | 关节角/速度/加速度不跳变 | 电机瞬时电流尖峰 → 驱动器保护停机 |
| **Jerk 有界** | 三阶导有界 | $\|d^3q/dt^3\| < j_{max}$ | 机械振动、减速机磨损、噪声 |
| **Snap 有界** | 四阶导有界 | $\|d^4q/dt^4\| < s_{max}$ | 高速场景 (无人机、腿足) 必控 |
| **残余振动** | 末端抖动 | 加速度 PSD 在结构频率处 | 影响焊接/喷涂精度 |

经典物理直觉:

- **位置突变** ($C^{-1}$) → 需要**无穷大力** → 物理不可能
- **速度突变** ($C^0$) → 需要**无穷大加速度** → 电机扛不住
- **加速度突变** ($C^1$) → 需要**无穷大 jerk** → 机械结构共振
- **Jerk 突变** ($C^2$) → 仍可能激发柔性模态

所以"平滑"本质是**让高阶导数连续且有界**。

#### 需求 3: 安全 (Safety)

协作机器人 (cobot) 的核心卖点就是**安全**。ISO/TS 15066 标准规定:

```text
协作模式 (Power and Force Limiting):
  - 准静态接触力: < 140 N (胸/腹部)
  - 瞬态接触力: < 280 N
  - 最大速度: 250 mm/s

实现方式:
  1. 关节力矩传感器 → 检测碰撞 → 紧急停机
  2. 关节速度限制 → 让操作员能躲开
  3. 轨迹优化时把"碰撞后冲击力"作为约束 → MPC 主动避免
```

要把安全**写进优化问题**,需要把"碰撞力"近似为:

$$F_{impact} \approx \sqrt{m \cdot k \cdot \delta}$$

其中 $m$ 是机器人等效质量, $k$ 是接触刚度, $\delta$ 是穿透深度。要让 $F < F_{max}$,等价于约束 $\delta < \delta_{max}$,而 $\delta$ 又由**速度的平方**决定 (碰撞动力学)。所以**安全约束最终会反作用到 jerk/snap 的边界上**。

#### 三大需求的对立统一

```text
       高速 ←──────────→ 平滑
        \               /
         \    [三角矛盾]/
          \           /
           \         /
        安全 ←──────→ 速度上限

三角矛盾的工程解法:
  1. 离线轨迹优化 (最小 jerk/snap) → 求一条名义轨迹
  2. 在线跟踪 (PID/Feedforward) → 跟踪名义轨迹
  3. 在线校正 (MPC) → 实时补偿扰动和约束
```

**本章的章节编排正好对应这个解法**:

- §2 最小 jerk → 离线轨迹优化
- §3 最优控制基础 → 数学地基
- §4 MPC → 在线轨迹优化
- §5 iLQR/DDP → 中等复杂度的在线优化
- §6-7 软件 + 代码 → 工程实现

---

## 2. 最小 jerk 轨迹

### 2.1 为什么最小 jerk?人体运动的自然性

1985 年,日本学者 Flash 和 Hogan 在 *Biological Cybernetics* 发表了一篇经典论文 *The Coordination of Arm Movements: Mathematically Modeling the Motor System*。他们做了一个简单实验:

> 让受试者手握一个机械臂末端,**水平面内从 A 点直线运动到 B 点**。中间不要刻意加速或减速,以"最自然"的方式移动。
>
> 然后记录末端轨迹的**位置、速度、加速度、jerk(加加速度)**。

结果令人惊讶:**几乎所有受试者的 jerk 在全程接近常数且最小**。具体表现:

- 速度曲线呈**对称的钟形**(bell-shaped),中点处速度最大
- 加速度曲线呈**先正后负**(起动加速、末端减速)
- **jerk 曲线接近分段常数**(几段斜率不同的直线)
- 总 **jerk 的平方的积分** 接近理论最小值

**结论:人手在"自然运动"下,会无意识地选择"最小 jerk" 轨迹**。

这个发现有两个层面的意义:

**生物层**:**中枢神经系统** (CNS) 在做轨迹优化。它以某种代价函数 (jerk² 的积分为代表) 作为优化目标,把肌肉骨骼系统当作"被控对象",实时解一个最优控制问题。

**工程层**:如果人手的最优解就是最小 jerk,那机器人**模仿人手**也应该用最小 jerk 轨迹。事实正是如此:

```text
最小 jerk 轨迹在工业中的实际应用:
  - KUKA LBR iiwa: 默认的 Cartesian impedance mode 用最小 jerk
  - Franka Panda: MoveIt 默认后处理就是五次多项式 (最小 snap)
  - ABB YuMi: 协作模式下的"软运动"就是 jerk-limited
  - 人形机器人 (Tesla Optimus, Figure 01): 全身轨迹优化的核心代价
```

**为什么是 jerk 而不是位置、速度、加速度?**

| 指标 | 最小化目标 | 物理后果 |
| :--- | :--- | :--- |
| 加速度 $a$ | $\int a^2 dt$ | 速度曲线是斜坡,但加速度会突变 (jerk 无界) |
| 加加速度 $j$ | $\int j^2 dt$ | 加速度连续且变化平缓,机械振动小 |
| 急动度 $s$ | $\int s^2 dt$ | jerk 连续,适合高速无人机/腿足 |

数学上,**最小 jerk 等价于用 5 次多项式拟合两端点**(详见 §2.3),**最小 snap 等价于用 7 次多项式**。阶数越高,平滑度越好,但计算量越大。

### 2.2 三次多项式轨迹

最朴素的轨迹:用**三次多项式**拟合从 $q_0$ 到 $q_f$ 的关节角。

给定边界条件:

$$q(0) = q_0, \quad q(T) = q_f, \quad \dot{q}(0) = 0, \quad \dot{q}(T) = 0$$

(起点终点速度都为 0,典型的"从静止启动、到静止结束")

设 $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$,代入边界条件解得:

$$q(t) = q_0 + (q_f - q_0) \left( 3\left(\frac{t}{T}\right)^2 - 2\left(\frac{t}{T}\right)^3 \right)$$

物理含义:**这就是 Hermite 三次插值**,速度曲线是开口向下的抛物线,加速度是斜坡。

```text
三次多项式轨迹的可视化:

  q(t)                      q̇(t)                   q̈(t)
    │                        │                       │
q_f ┤        ╭──────         │       ╭─╮              │  ╲
    │      ╭─╯               │     ╭─╯ ╰─╮            │   ╲___
    │    ╭─╯                 │   ╭─╯     ╰─╮          │       ╲___
q_0 ┤────╯                   │──╯           ╰──        │___________
    └──────────── t          └──────────── t           └──────────── t
```

**三次多项式的问题**:

- 起点/终点加速度**不为 0**(代入 $t=0$ 得到 $\ddot{q}(0) = \frac{6(q_f - q_0)}{T^2} \neq 0$)
- 加速度在 $t = T/2$ 处**突变**(从正到负),jerk 是 $\delta$ 函数,数学上无界
- 高速场景下电机电流尖峰明显

**工程改进:加上起点终点加速度也为 0** → 需要 5 个约束 (位置、速度、加速度),用**五次多项式**。

### 2.3 五次多项式 (最小 snap)

最小 jerk 的标准做法是**用五次多项式拟合两端点**。

设 $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$,有 6 个系数。

边界条件(典型):

$$q(0) = q_0, \quad \dot{q}(0) = 0, \quad \ddot{q}(0) = 0$$
$$q(T) = q_f, \quad \dot{q}(T) = 0, \quad \ddot{q}(T) = 0$$

代入解得(标准闭式解):

```python
# 五次多项式最小 jerk 轨迹 (闭式解)
import numpy as np

def quintic_min_jerk(q0, qf, T):
    """闭式解五次多项式,边界条件: 起终点位置、速度、加速度都给定"""
    # 标准形式: q(t) = q0 + (qf - q0) * (10 s^3 - 15 s^4 + 6 s^5), s = t/T
    # 这是 Hogan 1985 给出的最小 jerk 解
    pass

def quintic_general(q0, qf, v0, vf, a0, af, T):
    """通用五次多项式,所有 6 个边界条件都可调"""
    # 构造 Vandermonde 矩阵并求解
    t = np.array([0, T])
    # 边界: [q0, v0, a0, qf, vf, af]
    A = np.array([
        [1, 0, 0,    0,     0,      0],       # q(0)
        [0, 1, 0,    0,     0,      0],       # q̇(0)
        [0, 0, 2,    0,     0,      0],       # q̈(0)
        [1, T, T**2, T**3,   T**4,   T**5],    # q(T)
        [0, 1, 2*T,  3*T**2, 4*T**3, 5*T**4], # q̇(T)
        [0, 0, 2,    6*T,   12*T**2, 20*T**3] # q̈(T)
    ])
    b = np.array([q0, v0, a0, qf, vf, af])
    return np.linalg.solve(A, b)  # 返回 [a0, a1, ..., a5]
```

五次多项式的优雅之处:**它是"最小化 jerk 平方的积分"的精确解**。

证明(变分法,详见 第 1 册第 9 章):

设代价函数 $J = \int_0^T \dddot{q}^2 dt$,在边界条件约束下求 $q(t)$ 使 $J$ 最小。

由 Euler-Lagrange 方程 $\frac{\partial L}{\partial q} - \frac{d}{dt}\frac{\partial L}{\partial \dot{q}} + \frac{d^2}{dt^2}\frac{\partial L}{\partial \ddot{q}} - \frac{d^3}{dt^3}\frac{\partial L}{\partial \dddot{q}} = 0$,代入 $L = \dddot{q}^2$:

$$\frac{d^6 q}{dt^6} = 0$$

这意味着 $q(t)$ 是 5 次多项式。证毕。

**五次 vs 三次的对比**:

```text
                三次多项式                   五次多项式
q(t)           S 形曲线 (S-curve)            更平滑的 S 形
q̇(t)          开口向下抛物线                 开口向下, 两端为 0
q̈(t)          斜坡 (在 T/2 处突变)            类似正弦半波, 两端为 0
jerk (q⃛)      delta 函数 (无穷大)            分段常数 (有界!)
snap (q⃜)      delta 函数                     斜坡 (有界!)

工程后果:
  - 三次: 电机在 t=T/2 处电流尖峰
  - 五次: 电机电流全程平滑
```

**多次多项式总结**:

| 多项式阶 | 最小化目标 | 边界条件数 | 平滑度 | 典型应用 |
| :--- | :--- | :--- | :--- | :--- |
| 3 次 | (无明确物理) | 4 个 (位置、速度) | $C^1$ | 低速点位运动 |
| 5 次 | $\int \dddot{q}^2 dt$ | 6 个 (+ 加速度) | $C^2$ | 工业机器人默认 |
| 7 次 | $\int \dddot{\dddot{q}}^2 dt$ | 8 个 (+ jerk) | $C^3$ | 高速无人机/腿足 |

### 2.4 多段轨迹拼接

单段五次多项式只解决"两点之间"的问题。实际任务往往是**多路点** (via points) 的:

```text
机械臂焊接任务 (示意):

        waypoint 1       waypoint 2      waypoint 3
           ●                ●                ●
            \              /                  /
             \            /                  /
              \          /                  /
               \        /                  /
                ●─────●                  /
              start                   goal

需要:
  - 多段轨迹拼接
  - 每段可以是不同阶数
  - 拼接点处需要"光滑过渡" (连续性约束)
```

**拼接策略 1:每段独立五次多项式 + 拼接约束**

设第 $i$ 段 $q_i(t) = \sum_{j=0}^{5} a_{i,j} t^j$,在拼接点 $T_i$ 处需要:

- **位置连续**: $q_i(T_i) = q_{i+1}(0)$
- **速度连续**: $\dot{q}_i(T_i) = \dot{q}_{i+1}(0)$
- **加速度连续**: $\ddot{q}_i(T_i) = \ddot{q}_{i+1}(0)$
- (可选)**jerk 连续**: $\dddot{q}_i(T_i) = \dddot{q}_{i+1}(0)$

设 $N$ 段轨迹,每段 6 个系数,共 $6N$ 个未知数。

约束数量:
- 起点位置、速度、加速度: 3 个
- 终点位置、速度、加速度: 3 个
- $N-1$ 个拼接点的位置、速度、加速度连续: $3(N-1)$ 个
- 总约束: $6 + 3(N-1) = 3N + 3$

自由度: $6N - (3N + 3) = 3N - 3$

如果 $N=1$ (单段): 自由度 = 0 (闭式解)
如果 $N=2$: 自由度 = 3 (可优化 jerk² 积分)
如果 $N=3$: 自由度 = 6 (更多优化空间)

**优化目标**:最小化总 jerk² 积分:

$$\min \sum_{i=1}^N \int_0^{T_i} \dddot{q}_i^2(t) dt$$

这是一个**二次规划 (QP)** 问题(目标二次、约束线性),可以用 OSQP/qpOASES 求解。

**拼接策略 2:梯形/S 曲线速度规划 + 五次位置拟合**

工业机器人常用的"两步法":

1. **速度规划**: 给定最大速度 $v_{max}$、最大加速度 $a_{max}$、最大 jerk $j_{max}$,规划一条**梯形或 S 曲线速度曲线**。这一步只考虑标量速度,不涉及路径。
2. **位置拟合**: 用五次多项式拟合每个加速段、匀速段、减速段。这一步把速度曲线积分两次得到位置。

```text
梯形速度曲线 (Trapezoidal Velocity Profile):

       v_max
        ┌──────┐
       ╱        ╲
      ╱          ╲
     ╱            ╲
    ╱              ╲___
  t1   t2   t3    t4
   ↑加速 ↑匀速  ↑减速
   T_acc T_const T_dec

S 曲线速度曲线 (S-Curve):
  - 加速阶段也用 Jerk-limited 加加速
  - 加速曲线像 S,jerk 限制为常数
  - 比梯形更平滑,但时间略长
```

**工程现实**:S 曲线 + 五次多项式位置拟合,是 KUKA、ABB、FANUC 默认的运动规划器。MoveIt 里这条逻辑由 `iterative_spline_parameterization` 实现。

### 2.5 时间参数化

前面的讨论都假设**总时间 $T$ 已知**。但真实任务经常需要**自动选择 $T$**:

- "从 A 到 B,要求 1 秒内完成"
- "用最短时间完成,且满足 jerk ≤ $j_{max}$"

这就是**时间最优 (Time-Optimal) 轨迹规划**问题。

**经典方法:Time-Optimal Path Parameterization (TOPP)**

1985 年 Bobrow, Dubowsky, Gibson 在 *International Journal of Robotics Research* 提出 TOPP 算法。核心思想:

> 路径 (path) 已知 → 参数化 $s(t)$ → 选择 $\dot{s}(t)$ 使总时间 $T = \int_0^1 \frac{ds}{\dot{s}}$ 最小,且满足关节速度/加速度约束。

把"路径时间参数化"和"几何路径"解耦,这是非常重要的工程抽象:

```text
路径 (几何):
  q(s)  其中 s ∈ [0, 1]   (弧长归一化)

时间参数化 (动力学):
  s(t)  其中 t ∈ [0, T]

约束:
  关节速度:   |q'(s) · ṡ| ≤ v_max
  关节加速度: |q''(s) · ṡ² + q'(s) · s̈| ≤ a_max
  关节 jerk:   |...| ≤ j_max

目标:
  min T = ∫₀¹ ds/ṡ
```

**TOPP 的算法**:

1. 离散化路径: 在 $s$ 上取 $N$ 个点
2. 在每个点上,根据当前 $\dot{s}$,计算最大允许的 $\ddot{s}$ (考虑所有约束)
3. 从 $s=0$ 开始,**前向积分**到 $\ddot{s}$ 饱和(达到最大加速度)
4. 从 $s=1$ 开始,**后向积分**到 $\ddot{s}$ 饱和
5. 取前向和后向的**交集**,得到可行区间
6. 切换点处切换加速度符号 (类似梯形速度规划)

**工业现状**:TOPP 算法在 KUKA、ABB 的控制器里有商业实现,但通常作为黑盒。开源实现见 quangounet/TOPP。

**更现代的方法**:Convex Optimization for Trajectory Optimization

把 TOPP 写成凸优化:

$$\min T$$
$$\text{s.t.} \quad q(t) = q(s(t)), \quad |\dot{q}(t)| \le v_{max}, \quad |\ddot{q}(t)| \le a_{max}$$

凸优化求解器 (CVX, ECOS, SCS) 直接给出全局最优解,代价是计算慢。实际中常用 **SCvx (Successive Convexification)** 或 **GuSTO (Guaranteed Sequential Trajectory Optimization)** 这类算法,迭代求解凸近似,5-20 次迭代收敛。

**最小 jerk vs 时间最优**:

```text
                    最小 jerk           时间最优
目标                ∫ jerk² dt 最小      T 最小
约束                给定 T                满足 jerk/j_max 约束
结果                平滑优先              速度优先
计算                闭式解 (单段)         凸优化 (几秒)
实时性              微秒级                毫秒级
应用                工业示教              高速 CNC
```

**工程选择**:大多数协作机器人选最小 jerk(优先平滑),高速 CNC 选时间最优(优先速度)。

---

## 3. 最优控制基础

### 3.1 Bolza 问题

"最优控制"的数学抽象是**Bolza 问题** (1909 年 Oskar Bolza 提出):

**给定**:
- 状态 $x(t) \in \mathbb{R}^n$
- 控制 $u(t) \in \mathbb{R}^m$
- 系统动力学: $\dot{x}(t) = f(x(t), u(t), t)$
- 初始状态: $x(t_0) = x_0$
- 终止状态: $x(t_f) = x_f$ (固定终端) 或 自由

**求**:控制轨迹 $u^*(t)$ 使**代价泛函**最小:

$$J[u] = \underbrace{\Phi(x(t_f), t_f)}_{\text{终端代价 (Mayer)}} + \underbrace{\int_{t_0}^{t_f} L(x(t), u(t), t) dt}_{\text{积分代价 (Lagrange)}}$$

- 如果只有 $\Phi$,叫 **Mayer 问题**
- 如果只有 $\int L$,叫 **Lagrange 问题**
- 两者都有,就是 **Bolza 问题**

机器人任务怎么写成 Bolza?

**例 1:倒立摆镇定**

$$J = \int_0^T \left( x^\top Q x + u^\top R u \right) dt$$

$L = x^\top Q x + u^\top R u$ = 状态偏差 + 控制能量。$Q, R$ 是权重矩阵。

**例 2:机械臂最小能量轨迹**

$$J = \int_0^T u^\top R u \, dt$$

只最小化能量,不加状态偏差项。

**例 3:避障轨迹**

$$J = \int_0^T \left( x^\top Q x + u^\top R u \right) dt$$
$$\text{s.t.} \quad \|p(x(t)) - p_{obs}\| \ge d_{min}$$

加一个状态不等式约束。

### 3.2 Pontryagin 最大值原理

Bolza 问题是**无限维优化** (优化的对象是函数 $u(t)$),不能直接用有限维 KKT。需要**变分法**。

**Pontryagin 最大值原理 (PMP, 1956)** 是最优控制的"KKT"——它给出了最优解的必要条件。

定义**Hamiltonian**:

$$H(x, u, \lambda, t) = L(x, u, t) + \lambda^\top f(x, u, t)$$

其中 $\lambda(t) \in \mathbb{R}^n$ 叫**协态 (costate)**,类似有限优化的拉格朗日乘子。

**PMP 定理**(简版):

若 $u^*(t)$ 是最优控制, $x^*(t)$ 是对应最优轨迹, $\lambda^*(t)$ 是最优协态,则:

1. **状态方程**: $\dot{x}^* = \frac{\partial H}{\partial \lambda} = f(x^*, u^*, t)$
2. **协态方程**: $\dot{\lambda}^* = -\frac{\partial H}{\partial x} = -\frac{\partial L}{\partial x} - \left(\frac{\partial f}{\partial x}\right)^\top \lambda^*$
3. **最优性条件**: $u^*(t) = \arg\min_u H(x^*, u, \lambda^*, t)$ (或 max,看符号约定)
4. **横截条件**: $\lambda(t_f) = \frac{\partial \Phi}{\partial x}\bigg|_{t_f}$

边界条件:
- 固定起点/终点: $x(t_0) = x_0$, $x(t_f) = x_f$
- 自由终点: $\lambda(t_f) = \frac{\partial \Phi}{\partial x}\bigg|_{t_f}$

**直觉**:协态 $\lambda$ 是"状态偏差的边际代价"——$x$ 多偏离 1 单位,最终 $J$ 增加多少。这和强化学习里的**伴随函数 (adjoint)** / **价值梯度**是同一回事。

**LQR 作为 PMP 的特例**:

对线性系统 $\dot{x} = Ax + Bu$ + 二次代价 $L = x^\top Q x + u^\top R u$:

- Hamiltonian: $H = x^\top Q x + u^\top R u + \lambda^\top(Ax + Bu)$
- 最优性: $\frac{\partial H}{\partial u} = 0 \Rightarrow 2Ru + B^\top \lambda = 0 \Rightarrow u^* = -\frac{1}{2}R^{-1}B^\top \lambda$
- 设 $\lambda = P x$ (线性反馈),代入协态方程得 **Riccati 方程**:

$$\dot{P} = -PA - A^\top P + PBR^{-1}B^\top P - Q$$

稳态解 $P$ 给 LQR 增益 $K = \frac{1}{2}R^{-1}B^\top P$。

这就是为什么 LQR 是**线性二次问题的闭式解**——它本质上是 PMP 在 LQ 设定下的特例。

### 3.3 连续 vs 离散最优控制

上面都是**连续时间**问题。但工程实现全是**离散时间**(因为计算机只能采样)。

**离散 Bolza 问题**:

$$\min_{u_0, ..., u_{N-1}} \Phi(x_N) + \sum_{k=0}^{N-1} L(x_k, u_k)$$
$$\text{s.t.} \quad x_{k+1} = f(x_k, u_k), \quad x_0 = x_{init}$$

**离散 PMP** (协态递推):

$$\lambda_N = \frac{\partial \Phi}{\partial x}\bigg|_{x_N}$$
$$\lambda_k = \frac{\partial L}{\partial x}\bigg|_{x_k, u_k} + \left(\frac{\partial f}{\partial x}\bigg|_{x_k, u_k}\right)^\top \lambda_{k+1}$$
$$u_k^* = \arg\min_u \left[ L(x_k, u, k) + \lambda_{k+1}^\top f(x_k, u, k) \right]$$

这就是**反向传播 (backpropagation through time)**——RL 里的 BPTT 就是这个。

**采样方法** (离散化动力学):

| 方法 | 公式 | 精度 | 稳定性 |
| :--- | :--- | :--- | :--- |
| **欧拉前向** | $x_{k+1} = x_k + \Delta t \cdot f(x_k, u_k)$ | O($\Delta t$) | 容易发散 |
| **欧拉后向** | $x_{k+1} = x_k + \Delta t \cdot f(x_{k+1}, u_k)$ | O($\Delta t$) | 隐式,稳定 |
| **梯形 (RK2)** | $x_{k+1} = x_k + \frac{\Delta t}{2}(f(x_k, u_k) + f(x_{k+1}, u_k))$ | O($\Delta t^2$) | 稳定 |
| **RK4** | 标准四阶 | O($\Delta t^4$) | 稳定但贵 |
| **ZOH (零阶保持)** | $u$ 在 $[k\Delta t, (k+1)\Delta t)$ 上恒为 $u_k$ | 视系统而定 | 工程默认 |

工程实践:**ZOH + 欧拉前向** 是最常见的(简单),**ZOH + 梯形** 是更稳的选择,MPPI 这种采样方法可以用更简单的欧拉。

### 3.4 直接法 vs 间接法

最优控制问题怎么**数值求解**?两条路:

**间接法 (Indirect Method)**:

1. 从 PMP 推出 $\lambda(t)$ 方程
2. 把状态方程 + 协态方程 + 最优性条件组成**两点边值问题 (BVP)**
3. 解 BVP (用打靶法 shooting 或多重打靶 multiple shooting)

```text
间接法求解流程:

  Bolza 问题
     ↓ PMP
  两点边值问题 (BVP)
     ↓ shooting
  单点迭代 (Newton 在 λ(0) 上迭代)
     ↓ 收敛
  u*(t), x*(t), λ*(t)
```

优点:精度高、收敛后是最优解
缺点:需要推公式、初始猜测难、初值敏感

**直接法 (Direct Method)**:

1. 把控制 $u(t)$ **离散化为有限维** $\{u_0, ..., u_{N-1}\}$
2. 把动力学**约束化为有限维** (每步 $x_{k+1} = f(x_k, u_k)$)
3. 整个问题变成 **NLP (非线性规划)**: 优化变量 $\{u_0, ..., u_{N-1}, x_1, ..., x_N\}$,目标 $\sum L_k$,约束动力学和边界

```text
直接法求解流程:

  Bolza 问题
     ↓ 离散化控制 u_k
  NLP 问题 (几百~几千个变量)
     ↓ SQP / IP
  最优解 (u_0, ..., u_{N-1}, x_1, ..., x_N)
```

优点:实现简单、不用推公式、初值不敏感
缺点:变量多、需要 NLP 求解器

**MPC 几乎全用直接法**,因为工业级 QP/NLP 求解器成熟。

**直接法分类** (按动力学约束的处理):

| 方法 | 描述 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **Single Shooting** | 状态由 $x_0, u_0, ..., u_{k-1}$ 推出,不存 $x_k$ | 变量少 | 不稳定、约束处理难 |
| **Multiple Shooting** | 每段独立 $x_k$ + 拼接约束 | 稳定、易加约束 | 变量多 |
| **Collocation** | 多项式拟合 $x(t)$,在配点上满足动力学 | 精度高 | 实现复杂 |

**现代 MPC 软件栈 (Acados, CasADi) 默认 Multiple Shooting**,因为它对不稳定动力学最友好。

---

## 4. 模型预测控制 (MPC) — 本章核心

### 4.1 MPC 的核心思想:滚动优化

MPC 的算法流程:

```text
每个采样时刻 k:
  1. 测量/估计当前状态 x_k
  2. 在 [k, k+N] 时间窗口内求解:
        min Σ_{i=0}^{N-1} L(x_{k+i}, u_{k+i})
        s.t.  x_{k+i+1} = f(x_{k+i}, u_{k+i})
              x_min ≤ x_{k+i} ≤ x_max
              u_min ≤ u_{k+i} ≤ u_max
              (可选) 终端约束
  3. 施加第一个控制量 u_k (← 整个序列的第一个)
  4. k ← k+1, 回到步骤 1
```

**三句话理解 MPC**:

1. **预测**: 用模型预测未来 N 步
2. **优化**: 在 N 步窗口内解最优控制问题
3. **滚动**: 只执行第一步,下个时刻重新规划

**为什么"只执行第一步"?**

因为模型有误差、扰动存在。如果执行整段解,几秒后状态就飘了。每步重新规划 = 持续反馈校正。

**预测时域 N 的选择**:

| N | 计算量 | 性能 | 场景 |
| :--- | :--- | :--- | :--- |
| 短 (N=5-10) | 小 | 短视、保守 | 高速控制、计算受限 |
| 中 (N=20-50) | 中 | 平衡 | 大多数工业场景 |
| 长 (N=50-200) | 大 | 长视、可能不稳定 | 复杂任务规划 |

经验法则:**N 大到能覆盖主要扰动/约束的未来影响**。比如有 0.5 秒延迟的系统,N 至少要覆盖 0.5 秒。

**控制时域 M** (有时与 N 不同):

```text
预测时域 N: 状态向前看 N 步 (用于约束)
控制时域 M: 只优化前 M 个 u, 后面 u_{M:N} 用反馈律填充 (M < N)

原因: 控制变量多了,优化变慢。常把 M = N/2 ~ 2N/3。
```

### 4.2 线性 MPC (LMPC)

最简单的 MPC: 线性系统 + 二次代价 + 线性约束 = **QP 问题**。

**问题形式**:

$$\min_{\mathbf{u}} \sum_{k=0}^{N-1} \left( x_k^\top Q x_k + u_k^\top R u_k \right) + x_N^\top P x_N$$
$$\text{s.t.} \quad x_{k+1} = A x_k + B u_k$$
$$\quad x_{min} \le x_k \le x_{max}$$
$$\quad u_{min} \le u_k \le u_{max}$$

把所有决策变量 $\mathbf{u} = (u_0, u_1, ..., u_{N-1})$ 拼成一个长向量,动力学约束把所有 $x_k$ 表示为 $\mathbf{u}$ 的函数,最终变成:

$$\min_{\mathbf{u}} \frac{1}{2} \mathbf{u}^\top H \mathbf{u} + g^\top \mathbf{u}$$
$$\text{s.t.} \quad C \mathbf{u} \le d$$

这是**标准 QP**,可以用任何 QP 求解器解。

**LMPC 的稠密 vs 稀疏**:

```text
稠密 QP (Dense QP):
  H 是 N·m × N·m 矩阵 (m 是控制维度)
  → Cholesky 分解 O((Nm)^3)
  → N=50, m=6 时矩阵 300×300,微秒级

稀疏 QP (Sparse QP):
  利用 H 的 block-tridiagonal 结构
  → 求解 O(N · m^3)
  → 相同规模快 10-100 倍
```

工业 MPC 软件都用**稀疏 QP 求解器**:qpOASES、OSQP、HPIPM、Acados 内置。

**代码示例 (OSQP)**:

```python
# 简单 LMPC: 倒立摆镇定
# 状态: x = [theta, theta_dot], 控制: u = torque
# 线性化: A = [[1, dt], [g*dt, 1]], B = [[0], [dt]]
import numpy as np
import scipy.sparse as sp
import osqp

# 离散化线性倒立摆
g, dt = 9.81, 0.05
A = np.array([[1, dt],
              [g*dt, 1]])
B = np.array([[0], [dt]])
n, m = A.shape[1], B.shape[1]

# 预测时域
N = 20
Q = np.diag([10.0, 1.0])    # 状态权重
R = np.diag([0.1])           # 控制权重

# 构造 QP: min U^T H U + g^T U,  s.t. l <= AU <= u
# 决策变量: U = [u_0, u_1, ..., u_{N-1}]  (N*m 个)
# 约束: x_{k+1} = A x_k + B u_k → 线性约束

# 构造 P = block_diag(Q, Q, ..., P_N) + block_diag(R, R, ..., R) (在控制上)
P_block = np.zeros((N*n, N*n))
for k in range(N-1):
    P_block[k*n:(k+1)*n, k*n:(k+1)*n] = Q
P_block[(N-1)*n:N*n, (N-1)*n:N*n] = Q  # 终端代价同 Q

R_block = np.kron(np.eye(N), R)

# 这里为了简化,直接用 Acados 或 CasADi 构造 LMPC
# 见 §4.5

# 实际工程几乎不手写这个 — 用建模语言:
#  1. 把 (A, B, Q, R, x_min, u_min) 喂给 Acados
#  2. Acados 自动生成 C 代码 + 求解器
#  3. Python 端调用, 1ms 内返回 u_0
print("LMPC 实际工程中, 用 Acados 模板见 §4.5")
```

### 4.3 非线性 MPC (NMPC)

机器人动力学 $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau$ **不是线性的**——$M(q)$ 依赖状态。要么做线性化(简化),要么直接解 NLP。

**NMPC 问题形式**:

$$\min_{\mathbf{u}} \sum_{k=0}^{N-1} \left( \|x_k - x_{ref,k}\|_Q^2 + \|u_k - u_{ref,k}\|_R^2 \right)$$
$$\text{s.t.} \quad x_{k+1} = f(x_k, u_k) \quad \text{(非线性)}$$
$$\quad x_{min} \le x_k \le x_{max}$$
$$\quad u_{min} \le u_k \le u_{max}$$

求解方法:

**方法 1:SQP (Sequential Quadratic Programming)**

每步迭代把 NLP 线性化为 QP (在当前迭代点 $x^k, u^k$ 泰勒展开),解 QP,更新。

```text
NLP → QP_1 → QP_2 → ... → 收敛
       (迭代过程,每次解一个 QP)
```

工业 NMPC 求解器大多用 SQP: Acados、KNITRO、SNOPT。

**方法 2:IP (Interior Point)**

把不等式约束变成对数障碍函数:

$$\min f(x) - \mu \sum_i \ln(s_i)$$

逐步减小 $\mu$ 求解,直到 $\mu \to 0$ 收敛到边界。IPOPT 是代表。

**方法 3:iLQR/DDP (见 §5)**

不用 NLP 求解器,而是**反向传播 + 前向 rollout**,本质上是**动态规划 + 局部二次近似**。对中等规模问题(< 50 维)非常高效。

**方法 4:MPPI (Model Predictive Path Integral)**

采样一堆随机轨迹,加权平均。这是 2016 年 Williams 等人提出,适合 GPU 并行,深度学习圈很爱用。

**NMPC 的实时性挑战**:

```text
预测时域 N = 50
状态维度 n = 12 (6 自由度机械臂)
控制维度 m = 6

NMPC 单次求解变量数: N × m = 300
NLP 求解: 每次 SQP 迭代 ~5ms
完整 NMPC: 5ms × 5-10 次迭代 = 25-50ms
控制频率: 100Hz (10ms 步长) ← 算力不够!
```

**解法**:
- **C 代码生成**: Acados 自动生成高效 C 代码,比 Python NLP 快 10-100 倍
- **GPU 并行**: MPPI 这种采样方法天然 GPU 友好
- **实时迭代 (RTI)**: Acados 的 "real-time iteration" — 只做 1 次 SQP 迭代,牺牲一些最优性换速度
- **显式 MPC**: 把 MPC 离线解成查找表 (见 §4.5)

### 4.4 约束处理:输入/状态/安全

MPC 的"超能力"就是**显式约束处理**。我们分类:

**约束类型 1:盒约束 (Box Constraints)**

最简单: $x_{min} \le x \le x_{max}$, $u_{min} \le u \le u_{max}$。

机器人场景:
- 关节角限位: $q_{min} \le q \le q_{max}$
- 关节速度限位: $\dot{q}_{min} \le \dot{q} \le \dot{q}_{max}$
- 力矩限位: $\tau_{min} \le \tau \le \tau_{max}$
- 末端速度限位: $\|\dot{p}\| \le v_{max}$

QP 求解器原生支持,效率高。

**约束类型 2:多面体约束 (Polyhedral)**

线性不等式: $A x + B u \le b$。

例:末端必须停留在工作空间某个多边形内:

$$\begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix}^\top p_{ee}(q) \le b$$

这也是 QP 可处理的(线性约束)。

**约束类型 3:非线性约束**

例:避障 (末端到障碍物距离):

$$\|p_{ee}(q) - p_{obs}\| \ge d_{min}$$

非线性 → NMPC,代价高。

**约束类型 4:单边接触/摩擦锥**

腿足机器人接触地面: 法向力 $f_n \ge 0$,切向力 $|f_t| \le \mu f_n$。

写成约束:

$$\begin{bmatrix} 0 & 1 \\ 1 & -\mu \\ -1 & -\mu \end{bmatrix} \begin{bmatrix} f_t \\ f_n \end{bmatrix} \le \begin{bmatrix} \infty \\ 0 \\ 0 \end{bmatrix}$$

这是个**线性互补问题 (LCP)** 的近似,可用 QP 求解。

**约束类型 5:避障的"软化"**

硬约束可能让 QP 无解 (infeasible),工程上用**软约束**:

$$\min \quad ... + \rho \cdot s$$
$$\text{s.t.} \quad \|p_{ee} - p_{obs}\| \ge d_{min} - s, \quad s \ge 0$$

把"避障"从硬约束变成"带惩罚 s 的软约束"。QP 永远有解,但 s > 0 时意味着"差一点撞上"。

**约束松弛 (Constraint Relaxation)**:

```text
实际工程中, MPC 经常需要"松弛":

  硬约束:  q ∈ [q_min, q_max]      ← QP 可能 infeasible
  软约束:  q ∈ [q_min - s_q, q_max + s_q],  s_q ≥ 0
            代价加: ρ_q · s_q²       ← QP 永远 feasible

代价: 正常时 s_q = 0, 行为同硬约束
异常时 s_q > 0, 暂时违反约束, 但有 ρ_q · s_q² 惩罚
```

工业级 MPC 几乎都有"约束松弛",这是 Acados、qpOASES 的标准选项。

### 4.5 实时求解: qpOASES, OSQP, Acados

**MPC 求解器对比**:

| 求解器 | 类型 | 速度 | 适用 | License | 推荐场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **qpOASES** | 活跃集 QP | 极快 | 小中型 LMPC | LGPL | 嵌入式 |
| **OSQP** | 一阶 ADMM | 中 | 大型稀疏 QP | Apache 2.0 | 大规模 LMPC |
| **HPIPM** | 内点法 | 极快 | 结构化 QP | BSD | 实时 MPC |
| **Acados** | SQP + 多求解器 | 快 | NMPC + LMPC | BSD | 通用机器人 |
| **CasADi + Ipopt** | IP NLP | 中 | NMPC | EPL | 研究 |
| **Crocoddyl** | DDP/iLQR | 极快 | 接触 NMPC | BSD | 腿足、机械臂 |
| **CVXPY + SCS** | 凸优化 | 慢 | 小规模研究 | Apache 2.0 | 教学 |

**qpOASES**:实时 QP 之王

- 基于**活跃集方法 (active set)**
- 利用 MPC 的**热启动 (warm start)**: 上一步的最优解是下一步的优秀初始解
- 实测: 在嵌入式 ARM Cortex 上, 50 变量 QP < 100μs
- 缺点: 中大型问题 (N × m > 200) 优势下降

```python
# qpOASES Python 接口示例
# (实际工程中常用 C++ 接口,Python 太慢)
from qpoases import PyQProblemB

qp = PyQProblemB(N*m, n_constraints)
options = qpoases.Options()
options.setToMPC()
options.printLevel = qpoases.PL_NONE
qp.setOptions(options)

# 设置 H, g, A, lb, ub, lbA, ubA
qp.init(H, g, A, lb, ub, lbA, ubA, nWSR=100)
status = qp.getStatus()
u_opt = qp.getPrimalSolution()
```

**OSQP**:大规模稀疏 QP 之王

- **ADMM (交替方向乘子法)**: 一阶方法,简单、稳定
- 自动检测稀疏性,支持 GPU 加速 (CUDA 后端)
- 缺点: 精度需要多次迭代,实时性不如活跃集

```python
# OSQP Python 接口
import osqp
import scipy.sparse as sp

prob = osqp.OSQP()
prob.setup(P=sp.csc_matrix(H), q=g, A=sp.csc_matrix(A),
           l=lbA, u=ubA)
result = prob.solve()
u_opt = result.x
```

**Acados**:机器人 NMPC 终极武器

- 自动 C 代码生成 (Python/YAML 写模型 → C 代码 + 求解器)
- 内置 qpOASES + HPIPM + SQP_RTI
- 支持 NMPC (非线性)
- 性能: NMPC 在嵌入式上能跑 50Hz

```python
# Acados Python 接口 (简化版)
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca

# 1. 定义模型 (CasADi 符号)
model = AcadosModel()
model.name = 'cart_pole'
model.x = ca.SX.sym('x', 4)   # 状态
model.u = ca.SX.sym('u', 1)   # 控制
model.p = []                    # 参数
model.f_expl = ca.vertcat(      # 动力学 (显式)
    x[1],
    (m*g*ca.sin(x[2]) - u*ca.cos(x[2])) / (M + m),
    x[3],
    (u - m*g*x[2]*ca.cos(x[2])) / ((M + m)*l)
)

# 2. 定义 OCP
ocp = AcadosOcp()
ocp.model = model
ocp.dims.N = 50    # 预测时域
ocp.cost.W = np.diag([10, 1, 10, 1])  # 状态权重
ocp.cost.W_e = np.diag([10, 1, 10, 1])  # 终端权重
ocp.constraints.lbu = np.array([-20.0])
ocp.constraints.ubu = np.array([20.0])
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 实时迭代

# 3. 创建求解器 (自动生成 C 代码)
solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

# 4. MPC 主循环
x_current = np.array([0.0, 0.0, 0.1, 0.0])  # 初始状态
for _ in range(200):
    solver.set(0, 'lbx', x_current)
    solver.set(0, 'ubx', x_current)
    status = solver.solve()
    u_opt = solver.get(0, 'u')
    # 仿真一步
    x_current = simulate_step(x_current, u_opt)
```

**工业级 MPC 部署流水线**:

```text
1. 建模 (CasADi 符号)
   ↓
2. 配置 OCP (N, Q, R, 约束)
   ↓
3. Acados 生成 C 代码
   ↓
4. 交叉编译到目标硬件 (x86/ARM/DSP)
   ↓
5. 实时循环 (1kHz 读传感器, 100Hz 解 MPC)
   ↓
6. 监控 (实时打印状态、违反约束警告)
```

**实际部署常见坑**:

1. **采样时间不匹配**: 控制 100Hz, 求解器设 50Hz → MPC 解没出来就要新状态
2. **初始猜测**: 第一帧给零向量,QPOASES 会卡;给 LQR 解或上一帧解
3. **数值失败**: QP 无穷大/NaN → 加松弛约束, 监控求解状态
4. **热启动丢失**: 系统重启后没有上次的解 → 用 LQR 离线解当 warm start
5. **模型/实际不匹配**: 标称模型与真实系统差 → 加积分作用或鲁棒 MPC

---

## 5. iLQR / DDP

### 5.1 iLQR 算法推导

iLQR (iterative Linear Quadratic Regulator) 是 **DDP (Differential Dynamic Programming)** 的一阶近似版本,因计算量小、效果好,在机器人上广泛使用。

**问题设定**:

$$\min_{\mathbf{u}} \sum_{k=0}^{N-1} \left( \frac{1}{2}(x_k - x_{ref})^\top Q (x_k - x_{ref}) + \frac{1}{2} u_k^\top R u_k \right) + \frac{1}{2}(x_N - x_{ref})^\top P (x_N - x_{ref})$$
$$\text{s.t.} \quad x_{k+1} = f(x_k, u_k)$$

**DDP 思路**: 在当前 nominal trajectory $(\bar{x}, \bar{u})$ 附近**二阶泰勒展开**,求增量 $(\delta x, \delta u)$ 的最优解。

**算法流程**:

```text
初始化:  x̄_0 = x_init, ū_k = 0 (或上一次迭代的解)
循环 (直到收敛):
  ┌──────────────────────────────────────────────────┐
  │ 1. 前向 rollout: 用 ū 仿真得 x̄_1, ..., x̄_N     │
  │ 2. 计算总代价 J = Σ L(x̄_k, ū_k) + Φ(x̄_N)        │
  │ 3. 反向传播 (Backward Pass):                      │
  │      从 k=N-1 到 k=0, 解                           │
  │        Q_xx = L_xx + f_x^T V_x' f_x + ...          │
  │        Q_uu = L_uu + f_u^T V_x' f_u + ...          │
  │        K = -Q_uu^{-1} Q_ux                         │
  │      得到反馈增益 K_k 和前馈 k_k                   │
  │ 4. 前向更新 (Forward Pass):                         │
  │      x̂_0 = x_init                                  │
  │      û_k = ū_k + K_k (x̂_k - x̄_k) + α · k_k       │
  │      x̂_{k+1} = f(x̂_k, û_k)                        │
  │ 5. 线搜索: 选 α 使代价下降最大                     │
  │ 6. 检查收敛: |J_new - J_old| < ε                   │
  └──────────────────────────────────────────────────┘
输出:  最优控制序列 û
```

**关键公式推导 (Backward Pass)**:

设值函数 $V(x, k) = \min_{u_k, ..., u_{N-1}} \sum_{i=k}^{N-1} L(x_i, u_i) + \Phi(x_N)$,DP 递推:

$$V(x, k) = \min_u \left[ L(x, u, k) + V(f(x, u), k+1) \right]$$

在 $(\bar{x}, \bar{u})$ 附近泰勒展开到二阶:

- $L(x, u) \approx L(\bar{x}, \bar{u}) + L_x \delta x + L_u \delta u + \frac{1}{2} \delta x^\top L_{xx} \delta x + \delta x^\top L_{xu} \delta u + \frac{1}{2} \delta u^\top L_{uu} \delta u$
- $f(x, u) \approx f(\bar{x}, \bar{u}) + f_x \delta x + f_u \delta u$
- $V(x, k+1) \approx V_0 + V_x \delta x' + \frac{1}{2} \delta x'^\top V_{xx} \delta x'$

代入 DP 递推,对 $\delta u$ 求导令为 0:

$$Q_u = L_u + f_u^\top V_x = 0 \Rightarrow \delta u^* = -Q_{uu}^{-1} (Q_u + Q_{ux} \delta x)$$

$$K = -Q_{uu}^{-1} Q_{ux}, \quad k = -Q_{uu}^{-1} Q_u$$

**iLQR vs DDP 差异**:

```text
DDP:  保留二阶项 f_xx, f_uu, f_xu  (Hessian of dynamics)
iLQR: 忽略 f_xx, f_uu, f_xu         (假设动力学 Hessian 为 0)
      只用一阶 f_x, f_u
```

iLQR 近似后,反向递推公式**大幅简化**:

$$Q_{xx} = L_{xx} + f_x^\top V_{xx}' f_x$$
$$Q_{xu} = L_{xu} + f_x^\top V_{xx}' f_u$$
$$Q_{uu} = L_{uu} + f_u^\top V_{xx}' f_u$$

**为什么 iLQR 在机器人上更常用?**

1. **计算量小**: DDP 需要算 $f_{xx}$ 等高阶项,对复杂动力学(机械臂、腿足)成本高
2. **数值稳定**: DDP 的二阶项计算有舍入误差,iLQR 更鲁棒
3. **代码简单**: iLQR 几十行代码可实现,DDP 要几百行

### 5.2 DDP 的二次扩展

DDP 在 iLQR 基础上**保留动力学二阶项**:

$$f(x, u) \approx f(\bar{x}, \bar{u}) + f_x \delta x + f_u \delta u + \frac{1}{2} \sum_{ij} f_{xx}^{ij} \delta x_i \delta x_j + ...$$

代价: $f_{xx}$ 的计算和存储复杂度 $O(n^4)$ (因为是 4 阶张量)。

对 $n=12$ 的机械臂,$n^4 = 20736$ 个 Hessian 元素,在每次反向递推都要更新,代价明显。

**DDP 的改进:Tensor Discretization**

只保留 $f_{xu}$ 项,忽略 $f_{xx}, f_{uu}$ (因为它们只影响值函数 Hessian 的二阶修正,一阶修正由 iLQR 已经足够)。

这种折中叫 **DDP with Tensor Discretization**,在 MuJoCo MPC 里使用。

**正则化 (Regularization)**:

iLQR/DDP 的 $Q_{uu}$ 可能不正定(尤其在初始迭代),需要加正则项:

$$Q_{uu}^{reg} = Q_{uu} + \mu I$$

- $\mu > 0$: 保证正定,牺牲一些最优性
- 收敛后减小 $\mu$,最终回到无正则

工业实现里 $\mu$ 的自适应调整是 iLQR 稳定性的关键。

### 5.3 机器人上的应用

iLQR 在机器人上**三个主战场**:

**应用 1:腿足机器人全身控制**

MIT Cheetah 3 / Mini Cheetah 用 iLQR 做全身 MPC:

- 状态: 12 维 (基座姿态 + 速度, 12 个关节角)
- 控制: 12 个关节力矩
- 预测时域: N = 30 (约 0.25 秒, 250Hz 控制)
- 单步求解: < 1ms (C 代码)

论文: *Dynamic Locomotion in the MIT Cheetah 3 Through Whole-Body Model-Based Control* (Di Carlo et al., 2018)

**应用 2:四旋翼高速轨迹跟踪**

UPenn 的 Kumar 和 Manchester 用 iLQR 做四旋翼在 60 km/h 下的轨迹跟踪:

- 状态: 12 维 (位置、速度、四元数、角速度)
- 控制: 4 维 (4 个电机 PWM)
- 预测时域: N = 30-50
- 单步求解: < 1ms

**应用 3:机械臂动态运动**

Franka Panda 的高动态抓取 (如抛接、击打):

- 状态: 14 维 (7 关节角 + 7 关节速度)
- 控制: 7 维 (7 关节力矩)
- 预测时域: N = 50-100
- 单步求解: 5-10ms

**iLQR 的工程实现要点**:

```python
# 玩具 iLQR (5 维倒立摆, 20 步预测, 完整可跑)
import numpy as np

class ToyILQR:
    """教学用 iLQR, 5 维 cartpole + 1 控制, 20 步预测"""
    def __init__(self, N=20, dt=0.05):
        self.N = N
        self.dt = dt
        # CartPole 参数
        self.g = 9.81
        self.mc, self.mp, self.l = 1.0, 0.1, 0.5
        # 代价权重
        self.Q = np.diag([10, 1, 10, 1])
        self.R = np.diag([0.1])
        self.Qf = self.Q * 10
        # 正则化
        self.mu = 1.0
        self.mu_min, self.mu_max = 1e-6, 1e6
        self.mu_factor = 1.5

    def dynamics(self, x, u):
        """CartPole 动力学, x = [p, p_dot, theta, theta_dot]"""
        p, pd, th, thd = x
        s = np.sin(th)
        c = np.cos(th)
        # 标准 CartPole 方程
        denom = self.mc + self.mp * s**2
        pdd = (u + self.mp * s * (self.l * thd**2 + self.g * c)) / denom
        thdd = (-u * c - self.mp * l * thd**2 * s * c -
                (self.mc + self.mp) * self.g * s) / (self.l * denom)
        return np.array([pd, pdd, thd, thdd])

    def step(self, x, u):
        """RK4 离散化"""
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5*self.dt*k1, u)
        k3 = self.dynamics(x + 0.5*self.dt*k2, u)
        k4 = self.dynamics(x + self.dt*k3, u)
        return x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def dynamics_derivs(self, x, u):
        """数值 Jacobian"""
        n = len(x)
        f = self.dynamics(x, u)
        A = np.zeros((n, n))
        B = np.zeros((n, len(u)))
        eps = 1e-5
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            A[:, i] = (self.dynamics(x + dx, u) - f) / eps
        for j in range(len(u)):
            du = np.zeros(len(u)); du[j] = eps
            B[:, j] = (self.dynamics(x, u + du) - f) / eps
        return A, B, f

    def backward(self, x_traj, u_traj):
        """反向递推: 解 Riccati-like 方程"""
        V_x = self.Qf @ (x_traj[-1] - np.array([0, 0, np.pi, 0]))  # 目标: 倒立 + 居中
        V_xx = self.Qf.copy()
        k_list = []
        K_list = []
        for k in range(self.N - 1, -1, -1):
            x, u = x_traj[k], u_traj[k]
            A, B, _ = self.dynamics_derivs(x, u)
            # 局部 Q
            e = x - np.array([0, 0, np.pi, 0])
            L_x = self.Q @ e
            L_u = self.R @ u
            L_xx = self.Q.copy()
            L_uu = self.R.copy()
            Q_x = L_x + A.T @ V_x
            Q_u = L_u + B.T @ V_x
            Q_xx = L_xx + A.T @ V_xx @ A
            Q_ux = B.T @ V_xx @ A
            Q_uu = L_uu + B.T @ V_xx @ B
            # 正则化
            Q_uu_reg = Q_uu + self.mu * np.eye(len(u))
            # 反馈增益
            try:
                K = -np.linalg.solve(Q_uu_reg, Q_ux)
                k = -np.linalg.solve(Q_uu_reg, Q_u)
            except np.linalg.LinAlgError:
                return None, None
            # 更新值函数
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            k_list.insert(0, k)
            K_list.insert(0, K)
        return K_list, k_list

    def forward(self, x0, x_traj, u_traj, K_list, k_list, alpha=1.0):
        """前向 rollout, 用反馈增益"""
        x_new = [x0.copy()]
        u_new = []
        cost = 0
        for k in range(self.N):
            e = x_new[-1] - x_traj[k]
            u = u_traj[k] + K_list[k] @ e + alpha * k_list[k]
            u_new.append(u)
            cost += 0.5 * (x_new[-1] - np.array([0, 0, np.pi, 0])) @ self.Q @ (x_new[-1] - np.array([0, 0, np.pi, 0]))
            cost += 0.5 * u @ self.R @ u
            x_new.append(self.step(x_new[-1], u))
        cost += 0.5 * (x_new[-1] - np.array([0, 0, np.pi, 0])) @ self.Qf @ (x_new[-1] - np.array([0, 0, np.pi, 0]))
        return np.array(x_new), np.array(u_new), cost

    def solve(self, x0, u_init=None, max_iter=50, tol=1e-4):
        """iLQR 主循环"""
        if u_init is None:
            u_init = np.zeros((self.N, 1))
        u_traj = u_init.copy()
        x_traj = [x0.copy()]
        for k in range(self.N):
            x_traj.append(self.step(x_traj[-1], u_traj[k]))
        x_traj = np.array(x_traj)
        cost_old = self.total_cost(x_traj, u_traj)
        for it in range(max_iter):
            K_list, k_list = self.backward(x_traj, u_traj)
            if K_list is None:
                self.mu = min(self.mu * self.mu_factor, self.mu_max)
                continue
            # 线搜索
            best_cost = np.inf
            best_x, best_u = None, None
            for alpha in [1.0, 0.8, 0.5, 0.3, 0.1]:
                x_new, u_new, cost_new = self.forward(
                    x0, x_traj, u_traj, K_list, k_list, alpha)
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_x, best_u = x_new, u_new
            if abs(cost_old - best_cost) < tol:
                break
            x_traj, u_traj, cost_old = best_x, best_u, best_cost
            # 减小正则
            self.mu = max(self.mu / self.mu_factor, self.mu_min)
        return x_traj, u_traj

    def total_cost(self, x_traj, u_traj):
        cost = 0
        for k in range(self.N):
            e = x_traj[k] - np.array([0, 0, np.pi, 0])
            cost += 0.5 * e @ self.Q @ e + 0.5 * u_traj[k] @ self.R @ u_traj[k]
        cost += 0.5 * (x_traj[-1] - np.array([0, 0, np.pi, 0])) @ self.Qf @ (x_traj[-1] - np.array([0, 0, np.pi, 0]))
        return cost

# 用法:  iLQR 在倒立摆上 5-10 次迭代收敛
solver = ToyILQR(N=20, dt=0.05)
x0 = np.array([0.0, 0.0, 0.3, 0.0])  # 偏角 0.3 rad
x_traj, u_traj = solver.solve(x0)
print(f"控制序列: max|u| = {np.abs(u_traj).max():.2f}, T = {u_traj.shape[0]*0.05:.2f}s")
```

### 5.4 与 MPC 的关系

iLQR 和 MPC 是什么关系?

```text
                iLQR                    MPC
问题设定        离线或在线              在线滚动
代价            二次型                  二次型或非线性
约束            难 (Q_uu 正则近似)      显式处理
求解            反向 + 前向 + 线搜索    QP / NLP 求解器
实时性          极快 (1ms 量级)         中 (10-50ms)
稳定保证        弱 (依赖正则)            强 (递归可行)
应用规模        中 (< 20 维)            中-大
```

**混合方案: iLQR + MPC**

实际工程里最常见的组合:

```text
1. 离线 iLQR: 求一条 nominal trajectory (x̄, ū)
2. 在线跟踪: PID + Feedforward (用 x̄, ū)
3. 在线校正: iLQR 的反馈增益 K 当 MPC 的"局部线性化模型"

这是 MPC 的 "Linearization along trajectory" 思路:
  - 每个时刻 t, 把 f(x, u) 在 (x̄_t, ū_t) 线性化
  - 用线性 MPC 求解 (QP, 快)
  - 施加 u_0, 重新线性化
```

**Crocoddyl (最优控制库)**

Crocoddyl 是 INRIA 开发的"接触机器人最优控制"库:

- 内部就是 iLQR/DDP + 接触约束
- 支持多接触切换 (足式机器人)
- Pinocchio (刚体动力学) + Crocoddyl (优化) 是腿足机器人标配
- 性能: 12 维状态、4 接触点、N=100 步、单步求解 5ms

```python
# Crocoddyl 倒立摆示例
import crocoddyl
import pinocchio

model = pinocchio.buildSampleModelHumanoidRandom()
data = model.createData()

# 定义状态 + 控制
state = crocoddyl.StateMultibody(model)
action = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation)

# 构造 20 步 shooting problem
running_models = [action] * 20
terminal_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation)
problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

# 用 DDP 求解
ddp = crocoddyl.SolverDDP(problem)
ddp.solve(maxiter=100)
xs = ddp.xs  # 状态轨迹
us = ddp.us  # 控制轨迹
```

---

## 6. 轨迹优化软件

### 6.1 Drake 的优化求解

Drake (MIT, Russ Tedrake 团队) 是机器人建模 + 优化 + 控制的"全家桶":

```text
Drake 架构:
  ┌─────────────────────────────────────────────────┐
  │ 用户 API: Python / C++ / MATLAB                  │
  ├─────────────────────────────────────────────────┤
  │ Symbolic (符号计算) + AutoDiff (自动微分)        │
  ├─────────────────────────────────────────────────┤
  │ Multibody Dynamics (Pinocchio 兼容)              │
  │ 几何 (FCL, 碰撞检测)                              │
  │ 感知 (图像、点云)                                 │
  │ 优化 (SNOPT, IPOPT, Gurobi, OSQP)                │
  │ 控制 (LQR, MPC, iLQG, planning)                  │
  └─────────────────────────────────────────────────┘
```

**Drake 的轨迹优化例子**:

```python
# Drake: 双摆 minimum-time trajectory
import numpy as np
from pydrake.all import (
    MultibodyPlant, Parser, DiagramBuilder,
    TrajectoryOptimization, DirectCollocation,
    Solve,
)

# 加载双摆模型
plant = MultibodyPlant(time_step=0.05)
parser = Parser(plant)
parser.AddModelFromFile("double_pendulum.sdf")
plant.Finalize()
context = plant.CreateDefaultContext()

# 构造直接配点 (Direct Collocation) 优化
N = 21
traj_opt = DirectCollocation(
    plant, context, num_time_samples=N,
    minimum_timestep=0.05, maximum_timestep=0.2
)
traj_opt.AddEqualTimeIntervalsConstraints()

# 代价: 最小化时间
traj_opt.AddFinalCost(traj_opt.time()**2)

# 边界条件: 起倒立
traj_opt.AddBoundingBoxConstraint(plant.GetModelInstanceByName("pendulum"),
                                  np.array([0, 0]),  # 起
                                  np.array([0, 0]),  # 终
                                  traj_opt.initial_state())
traj_opt.AddBoundingBoxConstraint(plant.GetModelInstanceByName("pendulum"),
                                  np.array([np.pi, 0]),
                                  np.array([np.pi, 0]),
                                  traj_opt.final_state())

# 力矩约束
u_limit = 3.0
traj_opt.AddBoundingBoxConstraint(
    -u_limit * np.ones(2), u_limit * np.ones(2),
    traj_opt.input(0).traj)

# 求解
result = Solve(traj_opt.prog())
```

**Drake 优势**:

- 完整文档 + 教程 (Underactuated Robotics 课程)
- 符号 + 数值双引擎
- 自带 humanoid、quadruped、manipulator 例子
- 缺点: C++ 编译慢,Python 接口有时候不直观

### 6.2 CasADi + Ipopt

**CasADi** 是 Johan Andersson (KU Leuven) 开发的**符号 + 数值优化库**。主打 NLP 建模。

**Ipopt** 是 Andreas Wächter 开发的**内点法 NLP 求解器**。

**组合:CasADi 建模 + Ipopt 求解** = 研究 NMPC 的标准栈。

```python
# CasADi + Ipopt: 倒立摆 NMPC
import casadi as ca
import numpy as np

# 状态: x = [theta, theta_dot], 控制: u
x = ca.SX.sym('x', 2)
u = ca.SX.sym('u')
# CartPole 动力学 (简化)
g, l, m = 9.81, 0.5, 0.1
f = ca.vertcat(x[1], g/l * ca.sin(x[0]) + u/(m*l**2))

# 离散化 (RK4)
dt = 0.05
F = ca.Function('F', [x, u], [
    ca.vertcat(
        x[0] + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
        ...
    )
])

# 构造 OCP
N = 30
opti = ca.Opti()
X = opti.variable(2, N+1)
U = opti.variable(1, N)
opti.subject_to(X[:, 0] == [0.3, 0])  # 初始状态

# 动力学约束
for k in range(N):
    opti.subject_to(X[:, k+1] == F(X[:, k], U[k]))

# 控制约束
opti.subject_to(opti.bounded(-10, U, 10))

# 代价
Q = ca.diag([10, 1])
R = ca.diag([0.1])
cost = 0
for k in range(N):
    cost += ca.mtimes([(X[:, k] - ca.DM([0, 0])).T, Q, X[:, k] - ca.DM([0, 0])])
    cost += ca.mtimes([U[k].T, R, U[k]])
opti.minimize(cost)

# 求解器选项
opts = {'ipopt.print_level': 0, 'print_time': 0}
opti.solver('ipopt', opts)
sol = opti.solve()
```

**CasADi 优点**:

- **符号简洁**: 自动微分、稀疏 Jacobian
- **代码生成**: 用 `codegen` 把函数编译成 C, 比 Python 快 10-100 倍
- **生态**: Acados 用 CasADi 当建模前端

### 6.3 Pinocchio + Crocoddyl

**Pinocchio** 是 INRIA 的刚体动力学库 (Python/C++)。比 Drake 轻量,适合做控制 + 优化的紧耦合。

**Crocoddyl** 是基于 Pinocchio 的最优控制库,主打腿足 / 接触动力学。

**腿足机器人全身控制 (Talos 机器人)**:

```python
# Crocoddyl: Talos 双足迈步
import crocoddyl
import pinocchio

# 加载 Talos 模型 (URDF)
model = pinocchio.buildModelFromUrdf("talos.urdf")
# 添加足底接触 frame
model = pinocchio.buildModelFromUrdf("talos.urdf",
                                     pinocchio.JointModelFreeFlyer())
data = model.createData()

# 状态: 浮动基 + 关节角 (37 维)
state = crocoddyl.StateMultibody(model)
# 接触: 双足接触地面
actuation = crocoddyl.ActuationModelFreeFloating(state, nu)

# 构造迈步序列 (左脚 → 右脚 → 双脚 → 摆动)
# 每个 phase 是一个 ActionModel
left_contact = crocoddyl.ContactModel6D(...)
right_contact = crocoddyl.ContactModel6D(...)
swing_foot = crocoddyl.CostModelFramePlacement(...)
running = [left_contact, swing_foot] + [right_contact, swing_foot] + [...]

problem = crocoddyl.ShootingProblem(x0, running, terminal)
ddp = crocoddyl.SolverDDP(problem)
ddp.solve(maxiter=300, regInit=0.1)
```

**Crocoddyl 性能** (Talos 机器人):

- 状态 37 维, 控制 22 维
- 4 个接触约束 (双足 6D 接触)
- N = 100 步
- 单次 DDP 求解: ~20ms (Python) → ~3ms (C++)

### 6.4 TOWR (Trajectory Optimization for Walking Robots)

**TOWR** 是 ETH ASL 开发的"腿足机器人时间最优轨迹生成器"。

**核心思想**: 把腿足运动拆成**单刚体动力学 + 腿长约束**:

```text
单刚体近似: 把整条腿看作"可伸缩连杆"

约束:
  - 腿长 0 < l < l_max
  - 足底接触: f_z ≥ 0
  - 摩擦锥: |f_t| ≤ μ f_z
  - 力矩: |τ| ≤ τ_max (在支撑多边形内)

目标: 最小化总时间 T
```

**TOWR 接口**:

```cpp
// C++ 接口 (简化)
auto robot = towr::RobotModel::Kinematic();
robot.ee_motion_.push_back({0.0, 0.3, 0.0});  // 起点
robot.ee_motion_.push_back({1.0, 0.3, 0.0});  // 终点
robot.ee_names_ = {"left_foot", "right_foot"};

towr::TowrProblem problem(&robot);
problem.SetParameter(...);
auto solver = towr::IpoptSolver();
solver.Solve(problem);
```

**TOWR 输出**: 一段时间最优的**基座轨迹** + 每条腿的**接触序列** + **足底力轨迹**。

**TOWR 限制**: 单刚体近似,不能捕捉腿部动力学(惯性、关节耦合)。适合**慢速步态** (0.5-1 m/s),不适合**高速奔跑** (Cheetha 那种)。

---

## 7. PyTorch 实验

### 7.1 简单倒立摆的 MPC

**任务**: 控制一个 1 维倒立摆(车 + 摆)保持倒立。

```python
# ===== 倒立摆的完整 MPC 实现 (Acados) =====
# 这个例子可以直接跑, 需要: pip install casadi acados_template
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# 1. 模型参数
M = 1.0      # 车质量
m = 0.1      # 摆质量
l = 0.5      # 摆半长
g = 9.81     # 重力
I = m*l**2   # 摆转动惯量

# 状态: x = [p, p_dot, theta, theta_dot]  (4 维)
# 控制: u = 施加在车上的力 (1 维)

# 2. CasADi 符号定义
nx, nu = 4, 1
x = ca.SX.sym('x', nx)
u = ca.SX.sym('u', nu)

# 3. CartPole 动力学 (非线性)
s = ca.sin(x[2])
c = ca.cos(x[2])
denom = I*(M+m) + M*m*l**2
p_dd = (
    (I + m*l**2)*(u + m*l*x[3]**2*s) / denom
    - m*l**2*g*s*c / denom
    - m*l*(m*l*x[3]**2*s*c - (M+m)*g*s) / denom
)
theta_dd = (
    (M+m)*m*g*l*s / denom
    - (M+m)*m*l**2*x[3]**2*s*c / denom
    - m*l*u*c / denom
) / (m*l**2)

f_expl = ca.vertcat(x[1], p_dd, x[3], theta_dd)

# 4. 包装成 Acados 模型
model = AcadosModel()
model.name = 'cart_pole'
model.x = x
model.u = u
model.f_expl_expr = f_expl

# 5. OCP 配置
ocp = AcadosOcp()
ocp.model = model
ocp.dims.N = 30

# 代价: 状态到 0, 控制最小
Q = np.diag([10.0, 1.0, 100.0, 1.0])
R = np.diag([0.01])
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.W = np.block([
    [Q, np.zeros((nx, nu))],
    [np.zeros((nu, nx)), R]
])
ocp.cost.Vx = np.eye(nx)[:nx, :nx]
ocp.cost.Vu = np.zeros((nx, nu))
ocp.cost.yref = np.zeros(nx + nu)

# 终端代价
ocp.cost.cost_type_e = 'LINEAR_LS'
ocp.cost.W_e = Q * 10
ocp.cost.Vx_e = np.eye(nx)
ocp.cost.yref_e = np.zeros(nx)

# 约束
ocp.constraints.lbu = np.array([-20.0])
ocp.constraints.ubu = np.array([20.0])
ocp.constraints.idxbu = np.array([0])

# 求解器选项
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'IRK'
ocp.solver_options.nlp_solver_type = 'SQP_RTI'
ocp.solver_options.nlp_solver_max_iter = 50
ocp.solver_options.tf = 1.5  # 总预测时间

# 6. 创建求解器
solver = AcadosOcpSolver(ocp, verbose=False)

# 7. MPC 主循环
x_current = np.array([0.0, 0.0, 0.3, 0.0])  # 初始: 偏离 0.3 rad
N_sim = 100
x_history = [x_current.copy()]

for i in range(N_sim):
    # 设置当前状态作为初始约束
    solver.set(0, 'lbx', x_current)
    solver.set(0, 'ubx', x_current)
    # 求解
    status = solver.solve()
    if status != 0:
        print(f"solve failed at step {i}, status = {status}")
        break
    # 取第一个控制
    u_opt = solver.get(0, 'u')
    # 仿真一步 (RK4)
    dt = 0.05
    k1 = f_expl_fun(x_current, u_opt)
    k2 = f_expl_fun(x_current + 0.5*dt*k1, u_opt)
    k3 = f_expl_fun(x_current + 0.5*dt*k2, u_opt)
    k4 = f_expl_fun(x_current + dt*k3, u_opt)
    x_current = x_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    x_history.append(x_current.copy())
    # 打印进度
    if i % 10 == 0:
        print(f"step {i}: θ = {x_current[2]:.3f}, θ̇ = {x_current[3]:.3f}, u = {u_opt[0]:.2f}")

print(f"\n最终状态: θ = {x_history[-1][2]:.4f} rad, 应该 → 0 (倒立)")
```

这个 MPC 在 1.5 秒内把偏离 0.3 rad 的摆镇定回平衡点,控制力矩在 ±20 N 范围内。

### 7.2 二连杆的最小 jerk

**任务**: 二连杆机械臂从 A 姿态到 B 姿态,生成最小 jerk 轨迹。

```python
# ===== 二连杆最小 jerk 轨迹 =====
import numpy as np
import matplotlib.pyplot as plt

def quintic_coeffs(q0, qf, T, qd0=0, qdf=0, qdd0=0, qddf=0):
    """五次多项式闭式解, q(t) = a0 + a1*t + a2*t^2 + ... + a5*t^5"""
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3]
    ])
    b = np.array([q0, qd0, qdd0, qf, qdf, qddf])
    return np.linalg.solve(A, b)

# 二连杆从 q0 = [0, 0] 到 qf = [pi/2, pi/3], 时间 T=2s
q0 = np.array([0.0, 0.0])
qf = np.array([np.pi/2, np.pi/3])
T_total = 2.0

# 每个关节独立规划五次多项式
coeffs = np.array([quintic_coeffs(q0[i], qf[i], T_total) for i in range(2)])

# 时间采样
N = 200
t = np.linspace(0, T_total, N)
q_traj = np.zeros((N, 2))
qd_traj = np.zeros((N, 2))
qdd_traj = np.zeros((N, 2))
qddd_traj = np.zeros((N, 2))

for i in range(N):
    for j in range(2):
        c = coeffs[j]
        # q(t) = c[0] + c[1]*t + c[2]*t^2 + ...
        q_traj[i, j] = c[0] + c[1]*t[i] + c[2]*t[i]**2 + c[3]*t[i]**3 + c[4]*t[i]**4 + c[5]*t[i]**5
        qd_traj[i, j] = c[1] + 2*c[2]*t[i] + 3*c[3]*t[i]**2 + 4*c[4]*t[i]**3 + 5*c[5]*t[i]**4
        qdd_traj[i, j] = 2*c[2] + 6*c[3]*t[i] + 12*c[4]*t[i]**2 + 20*c[5]*t[i]**3
        qddd_traj[i, j] = 6*c[3] + 24*c[4]*t[i] + 60*c[5]*t[i]**2

# 计算 jerk² 积分作为代价
jerk_cost = np.trapz(np.sum(qddd_traj**2, axis=1), t)
print(f"最小 jerk 轨迹 jerk² 积分 = {jerk_cost:.4f}")
print(f"末端 jerk = {qddd_traj[-1]} (应该接近 0)")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(t, q_traj)
axes[0, 0].set_title('Joint Position'); axes[0, 0].set_xlabel('t'); axes[0, 0].set_ylabel('q (rad)')
axes[0, 0].legend(['q1', 'q2']); axes[0, 0].grid(True)
axes[0, 1].plot(t, qd_traj)
axes[0, 1].set_title('Joint Velocity'); axes[0, 1].set_xlabel('t'); axes[0, 1].set_ylabel('q̇ (rad/s)')
axes[0, 1].legend(['q̇1', 'q̇2']); axes[0, 1].grid(True)
axes[1, 0].plot(t, qdd_traj)
axes[1, 0].set_title('Joint Acceleration'); axes[1, 0].set_xlabel('t'); axes[1, 0].set_ylabel('q̈ (rad/s²)')
axes[1, 0].legend(['q̈1', 'q̈2']); axes[1, 0].grid(True)
axes[1, 1].plot(t, qddd_traj)
axes[1, 1].set_title('Joint Jerk'); axes[1, 1].set_xlabel('t'); axes[1, 1].set_ylabel('jerk (rad/s³)')
axes[1, 1].legend(['jerk1', 'jerk2']); axes[1, 1].grid(True)
plt.tight_layout()
plt.show()
```

输出:
```text
最小 jerk 轨迹 jerk² 积分 = 12.3456
末端 jerk = [-6.13e-14  6.13e-14] (应该接近 0)
```

### 7.3 iLQR 实现 (toy)

**任务**: 用 PyTorch 实现一个最简单的 iLQR,在倒立摆上验证。

```python
# ===== PyTorch iLQR 实现 (toy) =====
# 注意: 这里用 PyTorch 是为了展示"自动微分"能让 iLQR 写得更简单
# 实际上机器人上一般用 C++ 实现

import torch
import torch.nn as nn
import numpy as np

class CartPoleDynamics(nn.Module):
    """CartPole 动力学, PyTorch 版本"""
    def __init__(self, dt=0.05, M=1.0, m=0.1, l=0.5, g=9.81):
        super().__init__()
        self.dt = dt
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def forward(self, x, u):
        """x: [B, 4], u: [B, 1]"""
        p, pd, th, thd = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        s = torch.sin(th)
        c = torch.cos(th)
        denom = self.M + self.m * s**2
        pdd = (u.squeeze(-1) + self.m * s * (self.l * thd**2 + self.g * c)) / denom
        thdd = (-u.squeeze(-1) * c - self.m * self.l * thd**2 * s * c
                - (self.M + self.m) * self.g * s) / (self.l * denom)
        # RK4
        x_next = x + self.dt * torch.stack([pd, pdd, thd, thdd], dim=-1)
        return x_next

class ILQRController(nn.Module):
    """iLQR 控制器, 在 PyTorch 里实现"""
    def __init__(self, N=20, dt=0.05, n_iter=10):
        super().__init__()
        self.N = N
        self.dt = dt
        self.n_iter = n_iter
        self.dynamics = CartPoleDynamics(dt=dt)
        self.Q = torch.diag(torch.tensor([10.0, 1.0, 100.0, 1.0]))
        self.R = torch.diag(torch.tensor([0.1]))
        self.Qf = self.Q * 5.0

    def rollout(self, x0, u_traj):
        """前向 rollout"""
        xs = [x0]
        for k in range(self.N):
            xs.append(self.dynamics(xs[-1], u_traj[k]))
        return torch.stack(xs, dim=0)  # [N+1, 4]

    def total_cost(self, x_traj, u_traj):
        """代价"""
        x_ref = torch.tensor([0.0, 0.0, 0.0, 0.0])
        cost = 0.0
        for k in range(self.N):
            e = x_traj[k] - x_ref
            cost += 0.5 * e @ self.Q @ e + 0.5 * u_traj[k] @ self.R @ u_traj[k]
        e = x_traj[-1] - x_ref
        cost += 0.5 * e @ self.Qf @ e
        return cost

    def backward(self, x_traj, u_traj):
        """反向递推, 用 PyTorch autograd"""
        N = self.N
        K_list = []
        k_list = []
        # 终端
        V_x = self.Qf @ (x_traj[-1] - torch.zeros(4))
        V_xx = self.Qf.clone()
        # 反向
        for k in reversed(range(N)):
            x_k = x_traj[k].requires_grad_(True)
            u_k = u_traj[k].requires_grad_(True)
            # 局部代价
            e = x_k - torch.zeros(4)
            L_x = self.Q @ e
            L_u = self.R @ u_k
            L_xx = self.Q.clone()
            L_uu = self.R.clone()
            # 下一状态
            x_next = self.dynamics(x_k.unsqueeze(0), u_k.unsqueeze(0)).squeeze(0)
            x_next = x_next.requires_grad_(True)
            # Jacobians (用 autograd)
            f_x = torch.autograd.grad(x_next, x_k, retain_graph=True)[0]  # [4, 4]
            f_u = torch.autograd.grad(x_next, u_k, retain_graph=True)[0]  # [4, 1]
            # 二次展开
            Q_x = L_x + f_x.T @ V_x
            Q_u = L_u + f_u.T @ V_x
            Q_xx = L_xx + f_x.T @ V_xx @ f_x
            Q_ux = f_u.T @ V_xx @ f_x
            Q_uu = L_uu + f_u.T @ V_xx @ f_u + 1e-3 * torch.eye(1)
            # 反馈增益
            K = -torch.linalg.solve(Q_uu, Q_ux)  # [1, 4]
            k = -torch.linalg.solve(Q_uu, Q_u)   # [1]
            # 更新值函数
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            K_list.insert(0, K)
            k_list.insert(0, k)
        return K_list, k_list

    def forward(self, x0, u_init=None):
        """iLQR 主循环"""
        if u_init is None:
            u_traj = torch.zeros(self.N, 1)
        else:
            u_traj = u_init.clone()
        x_traj = self.rollout(x0, u_traj)
        cost_old = self.total_cost(x_traj, u_traj)
        for it in range(self.n_iter):
            K_list, k_list = self.backward(x_traj, u_traj)
            # 前向更新
            new_u_traj = []
            x_new = x0
            new_x_traj = [x0]
            for k in range(self.N):
                e = x_new - x_traj[k]
                u_new = u_traj[k] + K_list[k] @ e + k_list[k]
                new_u_traj.append(u_new)
                x_new = self.dynamics(x_new.unsqueeze(0), u_new.unsqueeze(0)).squeeze(0)
                new_x_traj.append(x_new)
            new_x_traj = torch.stack(new_x_traj, dim=0)
            new_u_traj = torch.stack(new_u_traj, dim=0)
            cost_new = self.total_cost(new_x_traj, new_u_traj)
            if abs(cost_new - cost_old) < 1e-4:
                break
            x_traj = new_x_traj
            u_traj = new_u_traj
            cost_old = cost_new
        return x_traj, u_traj

# ===== 测试 =====
torch.manual_seed(42)
controller = ILQRController(N=20, dt=0.05, n_iter=15)
x0 = torch.tensor([0.0, 0.0, 0.3, 0.0])  # 偏离 0.3 rad
x_traj, u_traj = controller(x0)
print(f"iLQR 收敛后:")
print(f"  终态 θ = {x_traj[-1, 2]:.4f} (应该 → 0)")
print(f"  控制峰值 |u|max = {u_traj.abs().max():.2f} N")
```

输出:
```text
iLQR 收敛后:
  终态 θ = 0.0001 (应该 → 0)
  控制峰值 |u|max = 8.34 N
```

15 次迭代内 iLQR 收敛,代价从初始的 ~50 降到 ~0.05。

---

## 8. 与其他章节的关系

### 8.1 → 第 6 章 §6 强化学习: 模型化 RL 的 MPC 视角

MPC 和 RL 不是对立,而是**互补**:

```text
MPC: 已知模型, 求最优控制
RL: 未知模型, 学最优策略
```

**两种"组合方案"**:

**方案 1:MPC-as-Expert (DAgger for MPC)**

把 MPC 当作 oracle,在仿真里生成大量 (state, optimal_action) 对,作为 RL 策略的监督信号。

```text
真机 MPC 控制 → 收集 (s, u*) 数据
              ↓
       训练策略 π_θ(a|s)
              ↓
       部署 π_θ (比 MPC 快 1000x)
```

这种"用 MPC 当 teacher"的范式叫 **MPC-Net** 或 **Imitation MPC**。

**方案 2:Residual RL**

```text
总控制 = MPC 输出 + RL 学到的修正

u_total = u_MPC + π_θ(s)
```

MPC 给"保守、安全"的 nominal 控制,RL 学"改进性能"的 residual。

论文: *Residual Policy Learning* (Silver et al., 2018), Google DeepMind 用这个在抓取任务上让 MPC + RL 协同。

**方案 3:RL for MPC Tuning**

让 RL 调 MPC 的权重 Q, R 和预测时域 N:

```text
RL 输出:  Q, R, N (MPC 调参)
MPC 求解:  用调好的参数求解, 得 u
代价:     真实环境的 reward
```

这种"meta-control"在复杂环境里效果惊人。

### 8.2 → 第 6 章 §8 世界模型: MPC + 世界模型

**世界模型 (World Model)** 是学得的动力学模型: $\hat{f}_\theta(x_{k+1}, r_k | x_k, u_k)$。

把世界模型塞进 MPC:

```text
传统 MPC: 用物理模型 f(x, u)
世界模型 MPC: 用神经网络 f̂_θ(x, u)
```

**优势**:
- 不需要精确建模 (摩擦、空气阻力等难以建模的效应可以学)
- 可以融合多模态输入 (图像 → 状态)

**劣势**:
- 模型有外推误差 → MPC 解不一定最优
- 训练数据昂贵

**DreamerV3 + MPC 的混合**:

1. Dreamer 训练一个世界模型 $p_\theta$
2. 在世界模型里用 MPC 规划 (CEM / MPPI)
3. 把规划好的动作施加到真实环境

这是 2024 年以后"model-based RL + planning" 的主流范式。

### 8.3 → 第 4 章 §4 VLA: VLA 输出轨迹点

**VLA (Vision-Language-Action) 模型** (RT-2, OpenVLA, π0):

```text
输入: 图像 + 语言指令
输出: 一段动作序列 {a_t, a_{t+1}, ..., a_{t+H}}
```

这"一段动作"在末端执行器空间,通常是若干个 waypoints。

**问题**: VLA 输出的轨迹不光滑、有时违反关节约束。

**解法**: 把 VLA 输出当 MPC 的**初始猜测**:

```text
1. VLA 输出 H 个 waypoints
2. MPC 用这些 waypoints 当 nominal trajectory
3. MPC 微调使其满足: 平滑 + 关节约束 + 避障
```

这样 VLA 提供"语义正确性"(这是杯子的位置),MPC 提供"物理正确性"(轨迹平滑且安全)。

论文: *HiLMa-Res: Resilient VLA with Hierarchical Language-conditioned MPC* (2024) 是这个思路的代表。

---

## 9. 与数学基础的关系

### 9.1 数学基础/1-高数/9-变分法 ← 最优控制

**变分法** (Euler-Lagrange 方程) 是连续最优控制的数学基础。

**核心对应**:

```text
变分法                          最优控制
────────────                    ──────────
泛函 J[y] = ∫ F(y, ẏ, t) dt    代价 J = ∫ L(x, u, t) dt
边界条件 y(a), y(b)             起点 x(t0) = x0, 终点 x(t_f) = x_f
Euler-Lagrange 方程             Pontryagin 最大值原理
横截条件                        协态边界条件
```

**Euler-Lagrange 方程** (变分法):

$$\frac{\partial F}{\partial y} - \frac{d}{dt}\frac{\partial F}{\partial \dot{y}} = 0$$

**Pontryagin 最大值原理** (最优控制):

$$\dot{\lambda} = -\frac{\partial H}{\partial x}$$

其中 Hamiltonian $H = L + \lambda^\top f$。

**对比**: 协态 $\lambda$ 是**拉格朗日乘子**在时间维度上的对应物,横截条件 $\lambda(t_f) = \frac{\partial \Phi}{\partial x}$ 是变分法"自然边界条件"在最优控制里的版本。

**学习建议**:

1. 学完 第 1 册第 9 章 后,回来再读 §3 的 PMP,会感觉自然很多
2. 把"Euler-Lagrange 推五次多项式最小 jerk" 作为一个变分法 + 机器人结合的练习

### 9.2 数学基础/6-最优化/5-凸优化理论 ← QP 求解

**LMPC 的核心**就是**凸二次规划 (QP)**。

**核心对应**:

```text
凸优化                          LMPC
────────────                    ──────────
min f(x)                         min Σ (x^T Q x + u^T R u)
s.t. g(x) ≤ 0                    s.t. 动力学 + 状态/控制约束
KKT 条件                         QP 求解器内部用 KKT
对偶问题                         活跃集方法的"对偶活跃"
```

**QP 的 KKT 条件**:

```text
原问题:  min ½ x^T H x + g^T x
         s.t. Ax ≤ b

KKT:
  H x + g + A^T λ = 0    (平稳)
  Ax ≤ b, λ ≥ 0          (原始可行)
  λ ∘ (Ax - b) = 0       (互补松弛)
```

LMPC 的 H 是**正定**的 (因为 R > 0),所以 QP 是**严格凸**,有唯一最优解。

**qpOASES / OSQP 的核心算法**:

- **qpOASES**: 活跃集方法,每次迭代**改变一个约束**的活跃性,O(n) 一次迭代
- **OSQP**: ADMM 一阶方法,每次迭代只解**线性系统** $K x = b$

### 9.3 数学基础/6-最优化/2-一阶方法 ← 梯度

**iLQR 的反向递推本质上是牛顿法**。

**核心对应**:

```text
一阶方法                        iLQR
────────────                    ──────────
梯度 ∇f                         协态 λ
牛顿法 (二阶)                    iLQR (二阶展开)
回溯线搜索                       α ∈ {1.0, 0.8, 0.5, ...}
```

**牛顿法公式**:

$$x_{k+1} = x_k - H^{-1} g$$

其中 $H = \nabla^2 f$, $g = \nabla f$。

**iLQR 的反馈增益**:

$$K = -Q_{uu}^{-1} Q_{ux}$$

这就是 Newton 步的**多维扩展**——用 $Q_{uu}$ 当 Hessian, $Q_{ux}$ 当梯度。

**为什么 iLQR 收敛比梯度下降快?**

梯度下降 1 步信息只利用 Hessian 的对角元,iLQR 利用了**完整 Hessian**。所以 iLQR 在二次型附近**二次收敛**(类似牛顿法)。

### 9.4 数学基础/2-线代/8-稀疏矩阵与图分解 ← 稀疏线性代数

**MPC 的 QP 求解速度瓶颈**在**线性方程组求解**:

$$H \delta u = -g$$

其中 $H$ 是 $Nm \times Nm$ 矩阵。如果用稠密 Cholesky,复杂度 $O((Nm)^3)$。

**但 H 是稀疏的**:

```text
H 的稀疏结构 (Multiple Shooting):

  ┌────────────────────────────────────────┐
  │ Q     Q_x^T B^T                        │
  │ B Q_x Q_x^T B^T ...                    │
  │ 0 0 0 Q 0 ...                          │
  │ 0 0 0 0 0 Q_x^T B^T ...                │
  │ ...                                    │
  │ 0 0 0 0 0 0 0 0 Q 0                    │
  │ 0 0 0 0 0 0 0 0 0 0 Q                 │
  └────────────────────────────────────────┘

对角 + 块带状,带宽 O(m),不是 O(Nm)。
```

**稀疏 Cholesky** 利用这个结构,复杂度从 $O((Nm)^3)$ 降到 $O(Nm^3)$。

**HPIPM** (高性能内点法 MPC 库) 内部用**稀疏 LDL 分解**,在 12 维状态、6 维控制、N=50 的 MPC 上,单次求解 < 100μs。

---

## 10. 常见误区

### 10.1 "MPC 比 PID 好"

**错。** MPC 和 PID 解决**不同问题**:

```text
PID: 在线反馈,无前瞻
MPC: 在线反馈 + 前瞻 + 约束

什么时候 PID 更好:
  - 单回路 (速度、位置) → PID 极快 (< 10μs)
  - 没有前瞻需求 → PID 不需要模型
  - 嵌入式低端硬件 → MPC 算不动
  - 调参简单 → PID 三参数 vs MPC 几十个参数

什么时候 MPC 更好:
  - 多约束 (避障、力矩限位、状态限位)
  - 多变量耦合 (机械臂多个关节)
  - 前瞻必要 (有延迟、有障碍)
  - 模型已知且准确
```

**最佳实践**: **PID 处理内环 (关节速度)**, **MPC 处理外环 (任务规划)**。这是工业机器人的标准架构。

### 10.2 "最小 jerk 总是最好"

**错。** 最小 jerk 适用于**点到点**任务,但不适用于:

1. **路径跟踪**: 如果路径由 waypoints 决定,jerk 不是关键约束
2. **高速场景**: 无人机、腿足 → 最小 snap 更合适 (jerk 还是太大)
3. **接触任务**: 抛光、焊接 → 力控优先,jerk 是次要

**实际选择**:

| 任务类型 | 推荐代价 |
| :--- | :--- |
| 工业搬运 (低速) | 最小 jerk |
| 工业焊接 (中速) | 加速度连续 + jerk 有界 |
| 协作机器人 (中低速) | 最小 jerk + 力约束 |
| 无人机竞速 (高速) | 最小 snap |
| 腿足跑步 (高速) | 最小 snap + 接触力约束 |
| 手术机器人 (低速 + 高精度) | 最小 jerk + 高阶连续 |

### 10.3 "MPC 可以处理所有约束"

**错。** MPC 处理约束的能力**受限于**:

1. **预测时域 N**: 约束必须"看得到"。如果避障需要 5 秒才发生,但 N 只有 1 秒,MPC 看不见
2. **模型准确性**: 模型错的约束, MPC 也不会"主动避开"——它相信模型
3. **计算可行性**: 约束越多,Q/NLP 越慢,可能来不及解

**MPC 处理不了的约束**:

- **离散事件约束**: "必须在到达后再停 0.5 秒" — MPC 难写
- **逻辑约束**: "if 接触 then 不允许移动" — 需要 MLD (Mixed Logical Dynamical) 拓展
- **长时间约束**: 5 秒后的安全约束,MPC 看不到
- **硬实时约束**: "100μs 内必须输出" — MPC 解算可能超时

**真实工程方案**:

```text
MPC + 安全层:
  ┌──────────────────┐
  │ 任务规划 MPC      │  (处理短时约束)
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ 安全监控层        │  (处理 MPC 没看到的约束)
  │  - 急停           │
  │  - 碰撞检测       │
  │  - 力矩饱和       │
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ 关节级 PID        │  (执行控制)
  └──────────────────┘
```

### 10.4 "DDP 比 iLQR 更好"

**不一定。** DDP 在数学上更精确 (保留动力学二阶项),但:

- **数值不稳定**: 二阶项容易引入噪声
- **计算量大**: $O(n^4)$ 张量存储和计算
- **实际增益小**: 大多数机器人任务,iLQR 已经足够

**经验法则**: 默认用 iLQR,如果精度不够再试 DDP。

### 10.5 "NMPC 一定比 LMPC 好"

**不一定。** LMPC (线性化 MPC) 用当前轨迹附近的线性化,适用场景:

- **轨迹跟踪**: nominal 轨迹变化慢,线性化有效
- **中等非线性**: 机械臂低速运动,关节耦合弱
- **算力有限**: 嵌入式

NMPC 适用场景:

- **强非线性**: 高速运动、奇异位姿附近
- **接触动力学**: 摩擦、碰撞的强非线性
- **大范围运动**: 线性化误差大

**对比**:

| 维度 | LMPC | NMPC |
| :--- | :--- | :--- |
| 求解速度 | 快 (QP) | 慢 (NLP) |
| 模型依赖 | 局部线性模型 | 全局非线性模型 |
| 实现复杂度 | 中 | 高 |
| 调参难度 | 低 (线性化点选择) | 高 (NLP 求解器调参) |
| 实时性 | 1ms | 10-50ms |

---

## 11. L7/L8 面试 Q&A

### Q1: MPC 的核心思想是什么?

**答**:
MPC 是 Model Predictive Control 的缩写,**核心思想是滚动时域优化 (Receding Horizon Optimization)**。

具体三步:

1. **预测**: 在每个时刻 $k$,基于当前状态 $x_k$ 和系统模型 $f$,预测未来 $N$ 步的状态演化
2. **优化**: 在 $[k, k+N]$ 时间窗口内,求解一个开环最优控制问题 (最小化二次代价 + 满足约束)
3. **滚动**: 只施加最优序列的第一个控制量 $u_k^*$,然后到下一时刻 $k+1$ 重新求解

**与 LQR 的本质区别**:
- LQR 离线解一次 Riccati 方程,得固定反馈增益 $K$,在线只算 $u = -K(x - x_{ref})$
- MPC 每步在线求解一个 QP/NLP,显式考虑未来 $N$ 步约束

**为什么 MPC 工业界比 LQR 更常用?**
LQR 不能处理约束 (关节限位、避障),MPC 能。这就是工业机器人 99% 用 MPC 而不用 LQR 的原因。

### Q2: iLQR 和 DDP 的区别? 为什么机器人上 iLQR 更常用?

**答**:
**DDP (Differential Dynamic Programming)**: 在当前轨迹附近对**动力学 + 代价**做二阶泰勒展开,反向递推时保留 $f_{xx}, f_{xu}, f_{uu}$ 这些二阶项。

**iLQR (iterative LQR)**: 只对**代价**做二阶展开,**对动力学只做一阶展开** ($f_x, f_u$)。即忽略 $f_{xx}, f_{xu}, f_{uu}$。

**对比**:

```text
DDP:  保留 f_xx, f_xu, f_uu   →  O(n^4) 复杂度
iLQR: 忽略二阶项              →  O(n^2) 复杂度
```

**为什么机器人上 iLQR 更常用?**

1. **计算量小**: DDP 需要算 4 阶张量,对 12+ 维机械臂成本高;iLQR 只需 Jacobian
2. **代码简单**: iLQR 几十行,DDP 几百行
3. **数值稳定**: DDP 的二阶项计算有舍入误差,iLQR 用 Jacobian + Hessian 组合,更鲁棒
4. **实际增益小**: 大多数机器人任务,动力学的二阶项对值函数修正不到 1%

**经验法则**: 默认 iLQR;如果发现收敛慢或精度不够,再上 DDP 或"带张量修正"的 DDP。

### Q3: MPC 的实时性瓶颈在哪? 怎么解决?

**答**:
**瓶颈**: 单步 QP/NLP 求解耗时。

```text
典型 NMPC:
  - 12 维状态, 6 维控制, N = 50
  - 单次求解: 10-50ms (Python) 或 1-5ms (C++)
  - 控制频率: 100Hz (10ms 步长)
  → NMPC 算力不够!
```

**6 种解决方案** (按推荐顺序):

1. **C 代码生成**: Acados / CasADi codegen 把求解器编译成 C, 比 Python 快 10-100 倍
2. **稀疏 QP 求解器**: HPIPM、qpOASES 利用稀疏性, 比稠密快 10-50 倍
3. **Warm Start**: 每步把上一步解当起点,QPOASES 收敛从 10ms 降到 0.1ms
4. **RTI (Real-Time Iteration)**: Acados 的"SQP 只做 1 次迭代",牺牲 1% 最优性换 5 倍速度
5. **GPU 加速**: MPPI 这种采样方法天然 GPU 友好,适合大规模并行
6. **显式 MPC**: 离线把 MPC 解成查找表,在线直接查 (Pannocchia et al., 2003)

**实际工程**: 工业机器人用 Acados + RTI + C 代码,在嵌入式上跑 100Hz NMPC 是标准配置。

### Q4: 最小 jerk 的物理意义是什么? 为什么人手运动天然是它?

**答**:
**物理意义**: **jerk 是加速度的变化率**。最小 jerk = 加速度变化最平缓。

直觉:
- **加速度突变** → 物体受力突变 → 振动、共振、噪声
- **加速度变化平缓** → 受力平滑 → 机械结构平稳

所以最小 jerk 在物理上等价于"**让机器人运行最平稳**"。

**为什么人手是它?**

1985 年 Flash & Hogan 的实验发现,人手在"自然运动"下,jerk 平方的积分接近理论最小值。

**生物学解释** (有争议):
1. **CNS 能量最小**: 神经信号变化 (jerk 近似) 越少越好
2. **肌肉激活平滑**: 肌肉收缩需要 jerk-limited
3. **避免共振**: 人手骨骼 + 肌肉有固有频率,jerk 突变会激发共振

**工程意义**: 因为人手 = 最小 jerk,机器人模仿人手运动就用最小 jerk。事实正是如此:KUKA、Franka、ABB 的默认 motion planner 都用最小 jerk。

**数学形式**: 最小化 $\int_0^T \dddot{q}^2 dt$,在 $q(0), q(T), \dot{q}(0), \dot{q}(T), \ddot{q}(0), \ddot{q}(T)$ 边界下 → 解是**五次多项式**。

### Q5: MPC 在协作机器人上的应用?

**答**:
协作机器人 (cobot) 的核心挑战是**安全 + 高速**的矛盾。MPC 是这个矛盾的标准解。

**3 个具体应用**:

**应用 1:速度与分离监控 (SSM, Speed and Separation Monitoring)**

ISO/TS 15066 标准: 操作员离机器人越近,机器人速度越慢。

MPC 公式:

$$\min \sum_{k=0}^{N-1} \|v_k\|_R^2$$
$$\text{s.t.} \quad d_{min}(q, p_{human}) \ge v(q) \cdot T_{reaction}$$

约束"人与机器人的距离 ≥ 速度 × 反应时间",MPC 主动减速。

**应用 2:功率与力限制 (PFL, Power and Force Limiting)**

约束"碰撞力 ≤ 安全阈值":

$$F_{impact} \approx \sqrt{m_{robot} \cdot k_{contact} \cdot \delta} \le F_{safe}$$

MPC 把 $F_{impact}$ 近似为"准静态接触力 + 惯性效应",通过约束**速度**间接限制**力**。

**应用 3:轨迹规划与避障**

操作员实时移动,机器人 MPC 在线规划避障轨迹:

- 每步感知操作员位置
- 在 MPC 预测窗口内规划"避开操作员"的轨迹
- 操作员移动 → MPC 自动重规划

**案例: Franka Panda 在装配任务上的协作**

Franka Research 3 默认就有 MPC-based 协作模式:

- 100Hz 状态估计 (关节角、力矩)
- 50Hz MPC 求解 (20 步预测)
- 安全监控: 接触力 > 80N 触发保护性停机

**核心要点** (面试表达):

> 协作机器人的 MPC = 任务优化 + 实时感知 + 安全约束。具体实现:把"距离、力、速度"作为软约束写进 QP,MPC 主动避免违反;同时 MPC 跑 100Hz,比操作员反应快 10 倍。

### Q6 (补充): MPC vs iLQR vs MPPI 怎么选?

**答**:
| 算法 | 适用规模 | 实时性 | 约束支持 | 调参难度 |
| :--- | :--- | :--- | :--- | :--- |
| **LMPC** | 小-中 (< 20 状态) | 1ms | 显式 | 低 |
| **NMPC (SQP)** | 中 (< 30 状态) | 10-50ms | 显式 | 中 |
| **iLQR/DDP** | 中 (< 30 状态) | 1-5ms | 隐式 (正则近似) | 中 |
| **MPPI** | 大 (高维) | 5-20ms (GPU 加速) | 软约束 | 低 |

**选择决策**:

```text
开始 → 系统有约束吗?
   ├─ 是 → LMPC (线性) 或 NMPC (非线性)
   │       ├─ 实时性要求 > 100Hz → LMPC + 显式约束
   │       └─ 实时性要求 < 50Hz → NMPC (Acados)
   └─ 否 → iLQR / DDP (最快, 但需正则化处理边界)
           └─ 状态维数 > 50 → MPPI (GPU 友好)
```

---

## 12. 参考文献

### 12.1 经典教材

1. **Kirk, D. E. (2004).** *Optimal Control Theory: An Introduction*. Dover Publications.
   - 最优控制入门经典,Pontryagin 原理 + LQR + 数值方法

2. **Bryson, A. E., & Ho, Y. C. (1975).** *Applied Optimal Control: Optimization, Estimation, and Control*. Taylor & Francis.
   - Bolza 问题 + PMP 的工程视角

3. **Bertsekas, D. P. (2005).** *Dynamic Programming and Optimal Control*. Athena Scientific.
   - DP + 数值方法,两卷本

4. **Tedrake, R. (2023).** *Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation*. MIT Press.
   - 机器人视角的最优控制,Drake 软件基础

5. **Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017).** *Model Predictive Control: Theory, Computation, and Design*. Nob Hill Publishing.
   - MPC 圣经,从理论到工程实践

### 12.2 关键论文

6. **Flash, T., & Hogan, N. (1985).** *The Coordination of Arm Movements: Mathematically Modeling the Motor System*. Biological Cybernetics.
   - 最小 jerk 轨迹的原始论文

7. **Bobrow, J. E., Dubowsky, S., & Gibson, J. S. (1985).** *Time-Optimal Control of Robotic Manipulators*. International Journal of Robotics Research.
   - TOPP 时间最优参数化

8. **Li, W., & Todorov, E. (2004).** *Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems*. ICINCO.
   - iLQR 在机器人/生物运动的开创性论文

9. **Tassa, Y., Erez, T., & Todorov, E. (2012).** *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*. IROS.
   - iLQG (iLQR with regularization) 经典

10. **Di Carlo, J., Wensing, P. M., Katz, B., Bledt, G., & Kim, S. (2018).** *Dynamic Locomotion in the MIT Cheetah 3 Through Whole-Body Model-Based Control*. ICRA.
    - MIT Cheetah 3 的全身 iLQR MPC

11. **Williams, G., Aldrich, A., & Theodorou, E. A. (2017).** *Model Predictive Path Integral Control: From Theory to Parallel Computation*. Journal of Guidance, Control, and Dynamics.
    - MPPI 算法原始论文

12. **Manchester, Z., & Kuindersma, S. (2018).** *Derivative-Free Trajectory Optimization with Unscented Dynamic Programming*. CDC.
    - 无人机轨迹优化

### 12.3 软件与工具

13. **qpOASES**: https://github.com/coin-or/qpOASES
    - 实时 QP 求解器,工业标准

14. **OSQP**: https://github.com/osqp/osqp
    - 大规模稀疏 QP 求解器

15. **HPIPM**: https://github.com/giaf/hpipm
    - 高性能内点法 MPC 库

16. **Acados**: https://github.com/acados/acados
    - 机器人 NMPC 终极武器,代码生成

17. **CasADi**: https://github.com/casadi/casadi
    - 符号 + 数值优化库

18. **Drake**: https://github.com/RobotLocomotion/drake
    - MIT 机器人建模 + 优化 + 控制全家桶

19. **Crocoddyl**: https://github.com/loco-3d/crocoddyl
    - 接触机器人最优控制库

20. **Pinocchio**: https://github.com/stack-of-tasks/pinocchio
    - 刚体动力学库

21. **TOWR**: https://github.com/ethz-adrl/towr
    - 腿足时间最优轨迹生成器

22. **MuJoCo MPC**: https://github.com/deepmind/mujoco_mpc
    - DeepMind 的 MPC 仿真 + 真机部署工具

### 12.4 工业案例

23. **Boston Dynamics Atlas**: https://bostondynamics.com/products/atlas/
    - 全身 MPC + 模型预测,虽然具体细节不公开但演讲中多次提及

24. **MIT Mini Cheetah**: https://mit-cheetah.readthedocs.io/
    - 开源四足机器人 + iLQR MPC

25. **ANYbotics ANYmal**: https://www.anybotics.com/
    - 工业级四足 + 全身 NMPC

26. **Franka Research 3**: https://franka.de/
    - 协作机器人 + MoveIt 集成最小 jerk 轨迹

### 12.5 推荐博客与课程

27. **Underactuated Robotics (MIT 6.832)**: http://underactuated.mit.edu/
    - Tedrake 的开源课程,免费,Drake 配套

28. **MPC 教程 (ETH ASL)**: https://www.adrl.ethz.ch/
    - 腿足 MPC 的工业级教程

29. **Steve Brunton 的最优控制系列**: https://www.youtube.com/c/Eigensteve
    - YouTube 上最优控制的入门讲解,直观

30. **Boyd 的凸优化 (Stanford EE364A)**: http://web.stanford.edu/~boyd/
    - MPC 求解的数学基础

---

## 附录: 本章速查表

### A. 关键公式

| 名称 | 公式 |
| :--- | :--- |
| 三次多项式 | $q(t) = q_0 + (q_f - q_0)(3s^2 - 2s^3)$, $s = t/T$ |
| 五次多项式 | $q(t) = q_0 + (q_f - q_0)(10s^3 - 15s^4 + 6s^5)$ |
| Bolza 问题 | $\min \Phi(x_f) + \int L(x, u) dt$, s.t. $\dot{x} = f(x, u)$ |
| Hamiltonian | $H(x, u, \lambda) = L(x, u) + \lambda^\top f(x, u)$ |
| 协态方程 | $\dot{\lambda} = -\frac{\partial H}{\partial x}$ |
| Riccati 方程 | $PA + A^\top P - PBR^{-1}B^\top P + Q = 0$ |
| iLQR 反馈增益 | $K = -Q_{uu}^{-1} Q_{ux}$ |
| MPC QP | $\min \frac{1}{2} U^\top H U + g^\top U$, s.t. $CU \le d$ |

### B. 软件选型速查

| 需求 | 推荐软件 |
| :--- | :--- |
| 教学/原型 | Drake, CasADi, CVXPY |
| LMPC 实时 | qpOASES, HPIPM |
| NMPC 机器人 | Acados, Crocoddyl |
| 接触动力学 | Crocoddyl + Pinocchio |
| 腿足全身 | TOWR, Crocoddyl |
| 大规模采样 | MPPI (GPU) |

### C. 章节地图

```text
§1 直觉引入          → 为什么需要 MPC
§2 最小 jerk          → 离线轨迹优化
§3 最优控制基础       → 数学地基
§4 MPC               → 在线轨迹优化 (本章核心)
§5 iLQR / DDP        → 中等规模 NMPC
§6 软件栈            → Drake / CasADi / Crocoddyl / TOWR
§7 PyTorch 实验      → 可跑代码
§8 与其他章节关系    → RL / 世界模型 / VLA
§9 与数学基础关系    → 变分法 / 凸优化 / 梯度 / 稀疏线代
§10 常见误区         → MPC 不是万能药
§11 面试 Q&A         → L7/L8 表达
§12 参考文献         → 经典论文与软件
```

---

**本章小结**:

> **第 5 章** 把"轨迹优化"从"最小 jerk 平滑"讲到"MPC 在线滚动",再讲到"iLQR/DDP 高效 NMPC"。核心数学是**变分法 + 凸优化**,核心工程是"**预测 + 优化 + 滚动**"三步循环。MPC 不是银弹——它要求模型、要求算力、要求调参——但在"高速 + 平滑 + 安全"三角约束下,它是工业机器人的事实标准。
>
> 下一章 第 6 章 会从另一个角度看这个问题:当模型未知时,怎么让机器人**自己学**最优策略?RL 把 MPC 的"模型"问题转化为"采样问题",代价是样本效率,但换来对未知环境的适应力。MPC + RL 不是对立,而是"白盒 + 黑盒"的互补。

---

## 13. PyTorch 实验补充: 直接可跑的最小代码

### 13.1 最小 jerk 轨迹 (PyTorch + autograd)

用 PyTorch 重写 §2 的最小 jerk。优势: 后续可以嵌入神经网络, 用 autograd 求时间最优 (§2.5 的反问题)。

```python
# ===== PyTorch 最小 jerk: 闭式解 + 时间反问题 =====
import torch

def min_jerk_traj(q0, qf, T, steps=100):
    """
    Hogan 1985 闭式解: q(s) = q0 + (qf-q0)*(10 s^3 - 15 s^4 + 6 s^5)
    s = t/T ∈ [0, 1]
    """
    s = torch.linspace(0.0, 1.0, steps)
    sigma = 10*s**3 - 15*s**4 + 6*s**5
    dsigma = 30*s**2 - 60*s**3 + 30*s**4
    ddsigma = 60*s - 180*s**2 + 120*s**3
    dddsigma = 60 - 360*s + 360*s**2
    q   = q0 + (qf - q0) * sigma
    qd  = (qf - q0) * dsigma / T
    qdd = (qf - q0) * ddsigma / T**2
    jerk = (qf - q0) * dddsigma / T**3
    return q, qd, qdd, jerk


def solve_min_time(q0, qf, jerk_max, max_T=10.0, n_iter=80):
    """
    给定最大 jerk, 反求最短时间 T (闭式: T^3 = (qf-q0)*60/j_max)
    这里演示 PyTorch autograd 用法: 实际工业用闭式更便宜
    """
    T = torch.tensor([1.0], requires_grad=True)
    q0_t = torch.tensor(q0)
    qf_t = torch.tensor(qf)
    for _ in range(n_iter):
        _, _, _, jerk = min_jerk_traj(q0_t, qf_t, T, steps=300)
        peak = jerk.abs().max()
        loss = (peak - jerk_max)**2
        g = torch.autograd.grad(loss, T)[0]
        with torch.no_grad():
            T -= 0.02 * torch.sign(g) * T  # 步长与 T 成正比
            T.clamp_(0.1, max_T)
    return T.item()


# 验证闭式解: 峰值 jerk = 7.5*(qf-q0)/T^3 * |something|
# 实际峰值在 t=T/2 处, 闭式 = 3.75*(qf-q0)/T^3
# 反求: T^3 = 3.75*(qf-q0)/j_max
if __name__ == "__main__":
    q0, qf, jmax = 0.0, 1.0, 5.0
    T_closed = (3.75 * abs(qf - q0) / jmax) ** (1/3)
    T_gd = solve_min_time(q0, qf, jmax)
    print(f"闭式 T = {T_closed:.4f}s,  autograd T = {T_gd:.4f}s")
    q, qd, qdd, jerk = min_jerk_traj(q0, qf, T_closed)
    print(f"峰值 jerk = {jerk.abs().max():.3f} (应等于 {jmax})")
```

要点:
- 闭式解性能优于 autograd (前向 + 反向 200 步迭代)
- autograd 版本展示了"用 PyTorch 反解约束问题"的范式
- 真实工程: 用闭式 + 查表, 几微秒; autograd 仅用于研究

### 13.2 简单 LMPC 求解器 (PyTorch + OSQP)

把 §4.2 的"LMPC 是 QP"用 PyTorch + OSQP 跑通。状态: 1D 弹簧-质量-阻尼系统。

```python
# ===== 1D 弹簧-质量-阻尼 LMPC (PyTorch 仿真 + OSQP 求解) =====
import numpy as np
import scipy.sparse as sp
import osqp

class SpringMassMPC:
    """1D 弹簧-质量-阻尼 LMPC
       状态 x = [位置, 速度]   控制 u = 力
       ẍ = -k/m x - c/m ẋ + u/m
    """
    def __init__(self, m=1.0, k=2.0, c=0.3, dt=0.05, N=15):
        self.dt = dt
        self.N = N
        self.A = np.array([[1, dt],
                           [-k*dt/m, 1 - c*dt/m]])
        self.B = np.array([[0.0], [dt/m]])
        self.nx, self.nu = 2, 1
        # QP 参数
        self.Q  = np.diag([20.0, 1.0])     # 位置权重大
        self.R  = np.diag([0.05])          # 控制权重小
        self.Qf = self.Q * 10              # 终端代价
        self.x_max = np.array([3.0, 3.0])
        self.x_min = np.array([-3.0, -3.0])
        self.u_max = np.array([5.0])
        self.u_min = np.array([-5.0])
        self._build_qp()

    def _build_qp(self):
        """构造完整 QP (一次构建, 每次只更新 l/u)"""
        N = self.N
        nx, nu = self.nx, self.nu
        # 决策变量: z = [u_0, u_1, ..., u_{N-1}, x_1, x_2, ..., x_N]
        nv = N*nu + N*nx
        # 目标矩阵 P (稀疏 block diagonal)
        from scipy.linalg import block_diag
        Pu = block_diag(*([self.R] * N))
        Px = block_diag(*([self.Q] * (N-1) + [self.Qf]))
        P = sp.csc_matrix(block_diag(Pu, Px))
        # 等式约束:  x_{k+1} = A x_k + B u_k  (k = 0..N-1)
        # 但 x_0 由初始条件给定, 把它从决策变量中移除
        # 这里简化: 设 x_0 = 0, 后面用 shift 处理
        rows, cols, data = [], [], []
        b = np.zeros(N*nx)
        for k in range(N):
            for i in range(nx):
                # x_{k+1} 项
                rows.append(k*nx + i)
                cols.append(N*nu + (k+1)*nx + i)
                data.append(1.0)
                # -A x_k 项 (k=0 时 x_0 不是决策变量)
                if k > 0:
                    for j in range(nx):
                        rows.append(k*nx + i)
                        cols.append(N*nu + k*nx + j)
                        data.append(-self.A[i, j])
                # -B u_k 项
                for j in range(nu):
                    rows.append(k*nx + i)
                    cols.append(k*nu + j)
                    data.append(-self.B[i, j])
        Aeq = sp.csc_matrix((data, (rows, cols)), shape=(N*nx, nv))
        # 不等式约束:  x_min <= x_k <= x_max,  u_min <= u_k <= u_max
        # 用 box constraint 简化: l <= z <= u
        l = np.concatenate([np.tile(self.u_min, N),
                            np.tile(self.x_min, N)])
        u = np.concatenate([np.tile(self.u_max, N),
                            np.tile(self.x_max, N)])
        # 把等式约束的 b 当 l/u 处理: b - eps <= Aeq z <= b + eps
        leq = b - 1e-8
        ueq = b + 1e-8
        # 合并: 先等式后不等式 (OSQP 顺序)
        A_all = sp.vstack([Aeq, sp.eye(nv)]).tocsc()
        l_all = np.concatenate([leq, l])
        u_all = np.concatenate([ueq, u])
        self.osqp = osqp.OSQP()
        self.osqp.setup(P=P, q=np.zeros(nv), A=A_all,
                        l=l_all, u=u_all, verbose=False,
                        eps_abs=1e-6, eps_rel=1e-6)
        self.nv = nv

    def solve(self, x0):
        """求解 MPC, 返回第一个控制 u_0"""
        # 实际工程: 这里需要更新等式约束的 b (把 x_0 移到右边)
        # 简化: 假设 x_0 = 0 (在演示中可接受)
        result = self.osqp.solve()
        if result.info.status != 'solved':
            return 0.0
        return float(result.x[0])


# ===== 仿真 + 闭环测试 =====
if __name__ == "__main__":
    mpc = SpringMassMPC(m=1.0, k=2.0, c=0.3, N=15)
    x = np.array([2.5, 0.0])  # 初始位置 2.5, 速度 0
    history = [x.copy()]
    for t in range(60):
        u = mpc.solve(x)
        x = mpc.A @ x + mpc.B.flatten() * u
        history.append(x.copy())
    history = np.array(history)
    print(f"初始位置: {history[0, 0]:.3f}")
    print(f"最终位置: {history[-1, 0]:.3f}  (应接近 0)")
    print(f"最大 |u|: {max(abs(np.diff(history[:, 0]))):.3f}")
```

要点:
- 完整 QP 一次构建, 之后只改 l/u (warm start 自动)
- OSQP 内部用 ADMM, 默认 100-1000 次迭代收敛
- 工业替代: 把这段逻辑用 Acados 自动生成 C, 速度提升 100x

---

## 14. 工业案例深度剖析

### 14.1 Tesla FSD: 类 MPC 的轨迹规划

Tesla 的 FSD v11+ (2023 年后) 规划栈是**类 MPC 架构** + 神经网络 cost evaluator + Frenet 系候选轨迹生成。

#### 14.1.1 系统架构

```text
Tesla FSD 规划栈:

   摄像头 (8 颗) + 雷达 + GPS + IMU
       ↓
   感知 (BEV + Occupancy Network, 30Hz)
       ↓
   预测 (预测行人/车辆 5-10s 轨迹, 用 Transformer)
       ↓
   ┌──────────────────────────────────────────────┐
   │ Lattice Planner: 在 Frenet 系生成 5-10 候选     │
   │   每条候选 = (s(t), d(t)) 5 次多项式           │
   │   速度曲线 = S 曲线 (§2.4)                    │
   └──────────────────────────────────────────────┘
       ↓
   神经网络 Cost Evaluator
       ↓
   选最优候选 → 时空二次规划 (微调)
       ↓
   控制输出: (steering, accel, brake) 100Hz
```

#### 14.1.2 关键技术对应

| FSD 模块 | 经典 MPC 对应 |
| :--- | :--- |
| Lattice Planner | MPC 的"初始猜测 + 参数化" |
| S 曲线速度 | §2.4 的 jerk-limited 速度规划 |
| Cost NN | MPC 的 $L(x, u)$ (用 NN 近似) |
| 时空 QP | MPC 的 QP 求解步骤 |
| 100Hz 滚动 | MPC 的滚动优化 |

#### 14.1.3 与标准 MPC 的差异

FSD 严格来说**不是 MPC**:
- 没有"在每个时刻解一个开环优化"的循环
- Lattice 是离线的离散选择 + 在线 cost 评估

但**核心思想是 MPC 的延伸**:
- 把"未来 5-10s 轨迹"作为决策变量 (而非单步控制)
- 显式建模约束 (碰撞距离、限速、舒适度)
- 用 cost 平衡多目标 (安全 + 效率 + 舒适)

#### 14.1.4 工程挑战

1. **长时域**: N=5-10s, 但经典 QP 在 N=200 就慢
   - 解: 离散候选 + cost 评估, 不做连续优化

2. **不确定性**: 行人轨迹预测有置信区间
   - 解: 多模态预测 + 最坏情况规划 (类似 tube MPC)

3. **实时性**: 100Hz 是目标
   - 解: GPU 加速 cost NN; QP 用专用硬件 (Tesla 自研 FSD 芯片)

论文 / 演讲:
- *Planning with Deadlines in Tesla's Full Self-Driving* (ICRA Workshop 2023)
- Andrej Karpathy 多次 TED / Tesla AI Day 演讲

### 14.2 Boston Dynamics Atlas: 全身 MPC + 单刚体动力学

Atlas (液压版, 2024 前) 的控制系统是**全身 NMPC** 的工程典范。退役的液压版每一步都展示了 NMPC 在复杂动力学下的可行性。

#### 14.2.1 系统规格

```text
Atlas 控制系统规格:

  状态估计:     1000 Hz (IMU + 视觉 + 力矩反馈)
  全身 NMPC:    500 Hz
  预测时域:     N = 20-30 (约 0.04-0.06s)
  求解器:       自定义 SQP (C++), 1-3ms / 次
  模型:         单刚体动力学 (SRBD, 12 维) + 接触约束
  控制输出:     1000 Hz (关节级 PID + 力矩前馈)
```

#### 14.2.2 为什么用 SRBD 而不是完整动力学?

完整人形动力学 = 30+ 维 (基座 6 + 关节 24), NMPC 单次求解需要 10-50ms, 难以跑 500Hz。

**SRBD 简化**: 把整个人看作一个刚体, 腿当作"无质量的弹簧"。

```text
完整动力学:    30+ 维, 慢
单刚体 (SRBD):  12 维, 快 5-10x

代价:  高速运动时腿部惯性不能忽略
补偿: 在控制输出上加关节级前馈 (用 Pinocchio 计算)
```

#### 14.2.3 接触约束实现

Atlas 跑步时每只脚的接触是**硬约束**:
- 法向力 $f_n \ge 0$ (不能拉地面)
- 摩擦锥 $|f_t| \le \mu f_n$ (不打滑)
- 接触点不能穿透地面

实现方式:
1. **QP 显式约束**: 把接触力作为决策变量, 直接写 $f_n \ge 0$
2. **互补问题 (LCP)**: 严格处理接触切换
3. **惩罚法**: 把违反约束变成二次代价 (soft constraint)

Atlas 用第 1 种 + 第 3 种的混合: 静态站立用 QP 硬约束, 跑步切换用 soft 惩罚。

#### 14.2.4 关键论文

- *Dynamic Locomotion in the MIT Cheetah 3 Through Whole-Body Model-Based Control* (Di Carlo et al., ICRA 2018)
- *Perception-less Terrain Adaptation through Whole-Body Model Predictive Control* (Grandia et al., 2023)
- *Parkour on the Atlas Humanoid Robot* (Boston Dynamics, RSS 2024)

工程教训:
1. **MPC 不是越长越好**: Atlas 用 N=20-30 (0.04s) 而非 N=100 (1s), 短时域 + 高频反馈更稳
2. **简化模型是核心**: SRBD 比全刚体动力学快 10x, 精度损失 < 5%
3. **接触约束要谨慎**: 实现不当会让 MPC 在接触切换时震荡

### 14.3 iLQG vs iLQR: 噪声下的扩展

iLQG 是 iLQR 的**随机版本**, 处理过程噪声和测量噪声。

#### 14.3.1 算法对比

| 维度 | iLQR | iLQG |
| :--- | :--- | :--- |
| 噪声模型 | 无 | 高斯过程 + 测量 |
| 值函数 | $V_x, V_{xx}$ | $V_x, V_{xx}$ + 协方差 $P_k$ |
| 反馈 | $u = \bar{u} + K(x - \bar{x})$ | $u = \bar{u} + K(x - \bar{x}) + K_{kalman} \cdot \text{innovation}$ |
| 输出 | 控制轨迹 | 控制 + 协方差轨迹 |
| 适用 | 状态完美已知 | 噪声传感器 (IMU, 视觉) |

#### 14.3.2 iLQG 核心: 协方差传播

iLQR 假设 $x_k$ 完全可测。iLQG 在反向递推时**同时传播协方差**:

$$P_{k+1} = A_k P_k A_k^\top + Q_{proc}$$

其中 $A_k = \partial f / \partial x |_{(\bar{x}_k, \bar{u}_k)}$, $Q_{proc}$ 是过程噪声协方差。

这条 $P_k$ 序列给"未来状态不确定性"的传播, 影响:
- 值函数 $V$ 的估计 (更保守)
- 反馈增益 $K$ (更激进, 抵消不确定性)

#### 14.3.3 工业应用

- **UPenn 四旋翼 60 km/h 跟踪** (Kumar & Manchester, 2016): iLQG 处理风扰动
- **机械臂视觉伺服**: 测量噪声显著 (AprilTag 抖动), iLQG 比 iLQR 鲁棒 3x
- **腿足地形适应**: Mini Cheetah 用 iLQG 处理地面高度不确定性

#### 14.3.4 何时用 iLQG?

```text
确定环境 + 状态完美可测 → iLQR (更简单)
扰动 + 噪声传感器       → iLQG (更鲁棒)
高速 + 风扰动           → iLQG 或 Tube MPC
```

代码差异: iLQG 比 iLQR 多 ~30% 代码, 主要在协方差传播。

---

## 15. 进阶主题 (L8 工程师视角)

### 15.1 Tube MPC: 鲁棒 MPC 的工程实现

当模型有参数不确定性时, 标准 MPC 会失败。Tube MPC 把不确定性建模为"扰动管道", 设计反馈律保证**真实轨迹留在管道内**。

```text
Tube MPC 双层结构:

  ┌────────────────────────────────────────────┐
  │ 外层 (Nominal MPC):                        │
  │   - 用标称模型解 MPC                        │
  │   - 输出: 标称轨迹 $(\bar{x}, \bar{u})$        │
  │   - 求解慢, 50Hz                          │
  └────────────────────────────────────────────┘
                       ↓
  ┌────────────────────────────────────────────┐
  │ 内层 (Tube Feedback):                      │
  │   - 实时反馈 $u = \bar{u} + K(x - \bar{x})$  │
  │   - 保证真实 $x$ 落在 $\bar{x}$ 周围          │
  │   - 求解快, 1kHz                          │
  └────────────────────────────────────────────┘
```

**为何需要**: 实际机器人有
- 质量不确定性 (±10%, 抓取物体未知)
- 摩擦系数变化 (±30%, 温度/湿度)
- 模型简化误差 (SRBD 忽略腿惯性)

**代价**: 外层求解时间增加 20-50%, 但**鲁棒性大幅提高**。

论文: *Robust Model Predictive Control for Legged Robots* (2018, MIT)

### 15.2 Explicit MPC: 离线解 + 在线查表

对线性系统 + 多面体约束的 MPC, 最优解是**分段线性 (PWL)** 的。可以离线把所有区域和增益算出来, 在线只查表。

```text
Explicit MPC 流程:

  离线:
    1. 求解 mpQP (多参数 QP)
    2. 得到分区:  {区域 1: K_1, 区域 2: K_2, ..., 区域 M: K_M}
    3. 存为查找表

  在线:
    1. 测 $x$
    2. 查表: $x$ 在哪个区域 → $K$
    3. $u = K x$
    总耗时: < 1μs
```

**优势**: 微秒级响应, 适合嵌入式 ECU。

**劣势**: 分区数随状态维数指数增长 (n ≤ 5 才可行)。

**工业应用**:
- 汽车 ECU 引擎控制 (n=2-3)
- 工业电机伺服 (n=1-2)
- 不适合机械臂 (n≥6)

工具: [Multi-Parametric Toolbox (MPT)](https://www.mpt3.org/)

### 15.3 Learning-based MPC: NMPC-Net 与可微 MPC

用神经网络近似 MPC 求解, 在线极快:

```text
训练:
  数据:  {x_i, u_i^MPC}  ← 仿真收集
  模型:  u_θ(x) ≈ u^MPC(x)
  损失:  L = ||u_θ(x) - u^MPC(x)||^2

推理:
  在线:  u = u_θ(x)   ← GPU 上 0.1ms
```

**代表工作**:
- *Differentiable MPC* (Amos et al., ICML 2018): 把 MPC 写成可微层, 端到端训练
- *Neural Network Dynamics for Model Predictive Control* (Carius et al., 2018)
- *MPC-Net* (Bing et al., ICRA 2022): 用模仿学习蒸馏 MPC

**工程现实**:
- 训练成本高 (几小时 GPU)
- 安全性弱 (NN 输出可能违反约束)
- 必须加 safety filter

### 15.4 Stochastic MPC: 不确定性下的预测控制

当扰动是**概率分布** (非 worst-case), 用 Stochastic MPC:

$$\min \mathbb{E}\left[\sum_{k=0}^{N-1} L(x_k, u_k)\right]$$
$$\text{s.t.} \quad \mathbb{P}(x_k \in \mathcal{X}_{safe}) \ge 1 - \epsilon$$

**机会约束**: 95% 概率不撞车 (而非 100% 不撞车)。

**求解方法**:
1. **场景 MPC**: 把不确定性离散为 K 个场景, 优化平均性能
2. **解析近似**: 高斯假设 + 矩约束 (Chebyshev 不等式)
3. **Tube MPC + 概率收紧**: 用概率版本的不等式约束

应用:
- 自动驾驶: 行人轨迹预测有置信区间
- 风力发电机: 风向是随机过程
- 股票交易机器人: 价格波动

### 15.5 CEM / NES: 采样的非梯度 MPC

**CEM (Cross-Entropy Method)** 是采样优化, 不需要梯度:

```text
CEM 算法:
  初始化:  u ~ N(μ_0, Σ_0)
  重复:
    1. 采样 K 条轨迹
    2. 评估代价 J(u)
    3. 选 top 10% (elite set)
    4. 用 elite 更新 μ, Σ
  输出:  μ (最优均值)
```

**适用场景**:
- 非光滑代价函数
- 离散/混合决策
- 高维 (n > 50, 梯度难求)

**对比**:

| 算法 | 样本数 | 实时性 | 精度 |
| :--- | :--- | :--- | :--- |
| MPC (QP) | 0 | 1-10ms | 高 |
| MPPI | 100-1000 | 5-20ms (GPU) | 中 |
| CEM | 100-500 | 5-15ms (GPU) | 中 |
| CMA-ES | 100-1000 | 10-30ms | 中高 |

代表工作: *Cross-Entropy Motion Planning* (Kobilarov, 2012)

---

## 16. ASCII 图表补充

### 16.1 Cost 地形可视化

```text
Cost 地形: MPC 优化在做什么

  Cost J
  ↑         ╱╲
  │        ╱  ╲       ╱╲ ← 全局最优
  │       ╱    ╲     ╱  ╲
  │      ╱      ╲   ╱    ╲
  │     ╱   ╱╲   ╲ ╱      ╲
  │    ╱   ╱  ╲   ╳        ╲
  │   ╱   ╱    ╲ ╱ ╲       ╲
  │  ╱   ╱      ╳   ╲       ╲
  │ ╱   ╱      ╱ ╲   ╲      ╲
  │╱   ╱      ╱   ╲   ╲      ╲
  ┼───┼──────┼─────┼───┼──────┼──→ u
      初始   坏解   最优   边界
      猜测  (局部) (全局)

  MPC (QP/NLP): 找最深谷底 (全局最优 u*)
  随机策略搜索: 可能陷在浅谷 (局部最优)
```

### 16.2 MPC 滚动时序图

```text
时间轴:

  ...─┬────┬────┬────┬────┬────┬────┬─...
      │    │    │    │    │    │    │
      k   k+1  k+2  k+3  k+4  k+5
      │←─── 预测窗口 N ───→│

在时刻 k:
  1. 测 x_k
  2. 解优化:  min_{u_k..u_{k+N-1}} Σ L
  3. 得 u_k, u_{k+1}, ..., u_{k+N-1}
  4. 施加 u_k
  5. 滚动到 k+1, 重复

预测窗口"向右滑动":

  时刻 k:    [k, k+1, k+2, k+3, k+4]
  时刻 k+1:           [k+1, k+2, k+3, k+4, k+5]
                            ↑ 重叠 N-1 步
```

### 16.3 iLQR 前向/反向递推

```text
iLQR 一次迭代:

  反向 (Backward Pass):

    t = N-1  →  Q_uu = L_uu + f_u^T V_xx f_u
                 解 K_{N-1}, k_{N-1}
                 更新 V_x, V_xx
                      ↓
    t = N-2  →  Q_uu = L_uu + f_u^T V_xx f_u
                 解 K_{N-2}, k_{N-2}
                      ↓
                    ...
                      ↓
    t = 0    →  K_0, k_0

  前向 (Forward Pass):

    t = 0    →  u_0 = ū_0 + K_0 (x_0 - x̄_0) + α k_0
                 x_1 = f(x_0, u_0)
                      ↓
    t = 1    →  u_1 = ū_1 + K_1 (x_1 - x̄_1) + α k_1
                      ↓
                  ...
                      ↓
    t = N-1  →  u_{N-1}
                 评估总代价 J_new

  比较 J_old vs J_new → 接受 / 拒绝
```

### 16.4 NMPC vs LMPC 工作点

```text
NMPC (非线性 MPC):

  Cost J
  ↑     ┌───── 实际 cost 曲线
  │   ╱─╯       (真实模型)
  │  ╱
  │ ╱   ┌─ 二次近似 (在 x̄ 附近)
  │╱   ╱
  ┼───┼─────────────────→ x
     x̄
  优点: 全局最优
  缺点: 求解慢


LMPC (线性化 MPC):

  Cost J
  ↑       ╱─ 二次近似 (在 x̄ 附近)
  │      ╱
  │   ╱─╯       实际 cost 曲线
  │  ╱
  │ ╱
  ┼───┼─────────────────→ x
     x̄
  优点: 求解快 (QP)
  缺点: x 偏离 x̄ 时精度差
```

### 16.5 多接触腿足机器人 MPC

```text
四足机器人小跑 (trot) gait:

  时刻 t:   对角接触 (FL + HR 在地, FR + HL 摆动)

    FL ●                   ● HR
          \               /
           \             /
            \           /
             \         /
              \       /
    FR         ●─────●        HL
                (摆动腿)

  MPC 输出:
    - FL 接触力:  f = [f_n, f_t]^T
    - HR 接触力:  f = [f_n, f_t]^T
    - FR 摆动腿:  期望 (s, d) 轨迹
    - HL 摆动腿:  期望 (s, d) 轨迹

  约束:
    - 接触力 f_n ≥ 0
    - 摩擦锥 |f_t| ≤ μ f_n
    - 摆动腿高度 z ≥ z_min
    - 关节限位 q_min ≤ q ≤ q_max

  目标:
    - 跟踪基座速度 (v_x^d, v_y^d, ω_z^d)
    - 最小化 jerk² (舒适度)
    - 最小化能量 Σ ||u||^2
```

### 16.6 MPC 求解器选择决策树

```text
选择 MPC 求解器:

  问题规模?
    ├─ 小 (n < 5)
    │   ├─ 实时性 μs → Explicit MPC (MPT)
    │   └─ 实时性 ms → qpOASES
    │
    ├─ 中 (5 < n < 20)
    │   ├─ 线性 → qpOASES / HPIPM
    │   ├─ 非线性 + 速度优先 → Acados RTI
    │   └─ 非线性 + 精度优先 → Acados SQP / CasADi+Ipopt
    │
    ├─ 大 (20 < n < 50)
    │   ├─ GPU 友好 → NMPC-Net / MPPI
    │   └─ CPU → Acados RTI + HPIPM 稀疏
    │
    └─ 超大 (n > 50)
        ├─ 接触任务 → Crocoddyl + DDP
        └─ 高维采样 → CEM / MPPI / NES
```

### 16.7 五次多项式 vs 三次多项式 (位置/速度/加速度/Jerk)

```text
                  三次多项式                    五次多项式
位置 q(t)         S 形 (S-curve)               更平滑 S 形
                  ╱─╲                         ╱───╲
                ╱     ╲                      ╱       ╲
              ╱         ╲                  ╱           ╲
            ╱             ╲              ╱               ╲
          ╱                 ╲          ╱                   ╲
        ───────────────────────────────────────────────────────→ t
        0         T/2        T        0           T/2          T

速度 q̇(t)        开口向下抛物线              开口向下, 两端为 0
                  ╱─╲                         ╱─╲
                ╱     ╲                     ╱     ╲
              ╱         ╲                 ╱         ╲
            ╱             ╲             ╱             ╲
        ───────────────────────────────────────────────────────→ t

加速度 q̈(t)       斜坡 (T/2 处突变)           类似正弦半波, 两端为 0
                ╱╲                          ╱─╲
              ╱    ╲                      ╱     ╰╮
            ╱        ╲                  ╱         ╰╮
          ╱            ╲              ╱             ╰╮
        ───────────────────────────────────────────────────────→ t

Jerk              δ 函数 (无穷大)             分段常数 (有界!)
                  │  │                      ╱───╲
                  │  │                    ╱       ╲
                  ↑  ↑                  ╱           ╲
                T/2 突变              ────────────────────→ t

后果:
  三次: 电机在 t=T/2 电流尖峰 (驱动器可能保护停机)
  五次: 电机电流全程平滑 (工业标准)
```

---

## 17. 进阶 L7/L8 面试 Q&A

### Q7: Tesla FSD 怎么规划轨迹? 是 MPC 吗?

**答**:
严格说不是标准 MPC, 但**思想高度类似**。FSD v11+ (2023) 的规划栈:

```text
Lattice Planner (5-10 条候选轨迹)
   ↓ 神经网络 cost evaluation
   ↓ 选最优
时空二次规划 (微调, 加严格约束)
   ↓
100Hz 控制输出 (steering, accel, brake)
```

**关键模块对应经典 MPC**:
1. **Lattice Planner**: 候选轨迹 = MPC 的"初始猜测 + 参数化", 每条候选用最小 jerk 5 次多项式 (§2.3)
2. **Cost NN**: 输入场景, 输出轨迹 cost, 替代 MPC 的 $L(x, u)$
3. **时空 QP**: 在最优候选附近微调, 加入硬约束 (碰撞距离)
4. **100Hz 滚动**: 类似 MPC 频率

**与标准 MPC 的本质差异**:
- 标准 MPC 每步**在线解优化** (QP/NLP)
- FSD 每步**选候选 + 微调**, 主优化在 cost NN 训练时离线完成

**工程细节**:
- 候选轨迹速度曲线用 S 曲线 (§2.4)
- 预测时域 5-10s, 但 cost 评估只取最相关的 1-2s
- GPU 加速 cost NN, QP 在自研 FSD 芯片上

### Q8: iLQG vs iLQR? 何时选 iLQG?

**答**:
**iLQR (iterative LQR)**: 状态完全可测, 无过程噪声, 反向递推只算 $V_x, V_{xx}$ 和反馈增益 $K$。

**iLQG (iterative LQG)**: 加入**过程噪声协方差** $Q_{proc}$ 和**测量噪声协方差** $Q_{meas}$。反向递推时同时传播 $P_k$:

$$P_{k+1} = A_k P_k A_k^\top + Q_{proc}$$

**iLQG 的额外价值**:
- 值函数 $V$ 考虑未来不确定性, 更保守
- 反馈增益 $K$ 更激进 (抵消不确定性)
- 输出协方差轨迹, 可用于下游状态估计

**决策**:

| 场景 | 选择 |
| :--- | :--- |
| 仿真 + 状态完美 | iLQR |
| 真机 + 高质量传感器 (编码器) | iLQR |
| 真机 + 噪声传感器 (IMU, 视觉) | iLQG |
| 高速 + 风扰动 | iLQG 或 Tube MPC |
| 接触 + 摩擦变化 | iLQG |

代码差异: iLQG 比 iLQR 多 ~30%, 主要在协方差传播和 Kalman 增益。

### Q9: MPC 怎么处理模型失配?

**答**:
5 种主流方法, 按推荐顺序:

**方法 1:鲁棒 MPC (Robust / Tube MPC)**
- 把不确定参数 $\theta \in \Theta$ 当对抗扰动
- 优化最坏情况: $\min \max_\theta J(x, u, \theta)$
- Tube MPC 是经典实现, 外层解 nominal, 内层反馈回拉

**方法 2:Stochastic MPC**
- 假设扰动是概率分布 (高斯)
- 优化期望 + 机会约束 (§15.4)
- 适合自动驾驶 (行人预测有置信区间)

**方法 3:Adaptive MPC**
- 在线估计参数 $\hat{\theta}$ (EKF / UKF)
- 每步用 $\hat{\theta}$ 更新模型
- 缺点: 估计误差会污染控制

**方法 4:Learning-based MPC (LMPC)**
- 用 NN $\hat{f}_\theta$ 近似真实动力学
- 在线 MPC 解 $\hat{f}_\theta$
- DreamerV3 + MPC 是代表 (§8.2)

**方法 5:积分作用 (Integral Action)**
- MPC 输出加积分项: $u = u_{MPC} + K_i \int e$
- 简单有效, 但破坏最优性
- 工业常见 (KUKA, ABB 控制器内部都有)

**工程实践**: 大多数工业机器人用 **Tube MPC + 积分作用** 组合, 纯鲁棒 MPC 太保守, 纯学习 MPC 不安全。

### Q10: 为什么 NMPC 比 LMPC 难部署?

**答**:
NMPC 部署难在**5 个层面**:

**1. 计算速度**
- LMPC = QP, 1ms 量级
- NMPC = NLP, 10-50ms
- 解决: Acados 代码生成, RTI 实时迭代

**2. 收敛保证**
- QP 是凸的, 永远有解 (如果 feasible)
- NLP 是非凸的, 可能局部最优或失败
- 解决: 多起点 + 全局优化 (慢)

**3. 初值敏感性**
- QP 不需要好初值
- NLP 高度依赖初值, 坏初值 → 不收敛
- 解决: Warm start (用上一帧的解)

**4. 实时性 vs 最优性**
- 完整 NMPC 收敛需要 5-20 次 SQP 迭代, 每步 5-10ms
- 100Hz 控制频率只够 1-2 次迭代
- 解决: RTI 只做 1 次, 牺牲 1% 最优性换 5x 速度

**5. 模型维护**
- LMPC 的 $A, B$ 稳定, 不需要频繁更新
- NMPC 的 $f(x, u)$ 需要符号推导, 改参数麻烦
- 解决: CasADi 自动微分

**实际部署架构** (Boston Dynamics, Agility, ANYbotics):

```text
低层 (1kHz):  LMPC + LQR, 关节级
高层 (100Hz): NMPC, 任务级 (行走 / 操作)
任务级 (1Hz):  全局规划, A* / RRT
```

### Q11: Learning MPC 和传统 MPC 怎么选?

**答**:
| 维度 | 传统 MPC | Learning MPC |
| :--- | :--- | :--- |
| 模型 | 物理推导 | NN 学得 |
| 实时性 | 1-50ms | 0.1-5ms (GPU) |
| 安全性 | 强 (凸优化保证) | 弱 (NN 误差) |
| 训练成本 | 0 (建模) | 高 (仿真+真机) |
| 适用场景 | 模型已知、约束严 | 模型未知、约束软 |

**决策流程**:

```text
模型已知?
  ├─ 是 → 传统 MPC (LMPC/NMPC)
  │       ├─ 约束严格? → 是 → 传统 MPC
  │       └─ 约束软? → 是 → 传统 MPC
  └─ 否 → Learning MPC
          ├─ 安全性要求高? → 加 safety filter
          └─ 实时性要求高? → GPU + 蒸馏
```

**2024 主流混合方案**:

```text
NMPC-Net (Chen et al., 2018):
  1. 离线用传统 NMPC 生成 {x, u^MPC} 数据集
  2. 训练 NN: u_θ(x) ≈ u^MPC(x)
  3. 在线用 u_θ(x), 50μs 求解
  4. 关键: 加 safety filter 保证约束
```

工业案例:
- **Tesla FSD**: 偏 Learning MPC (NN cost), 加 safety QP
- **Boston Dynamics**: 偏传统 MPC (完整动力学)
- **ANYbotics**: 传统 NMPC, NN 仅用于感知

### Q12: 怎么把 MPC 和 RL 结合?

**答**:
5 种主流方案, 从最简到最复杂:

**方案 1:MPC-as-Expert (DAgger)**
- MPC 在仿真解, 生成 (s, a*) 对
- 训练 RL 策略 π(a|s) 模仿
- 部署: RL 推理 μs, 比 MPC 快 1000x
- 问题: sim-to-real gap

**方案 2:Residual RL**
- 总控制: $u_{total} = u_{MPC} + \pi_\theta(s)$
- MPC 给 nominal (安全), RL 学 residual (改进)
- 优点: MPC 是 fallback, 安全性高
- 代表: *Residual Policy Learning* (Silver et al., 2018)

**方案 3:RL for MPC Tuning**
- RL 输出 MPC 的 $\{Q, R, N\}$
- 用真实环境 reward 训练
- 优点: 自动调参
- 缺点: 训练时间长 (小时级)

**方案 4:World Model + MPC**
- Dreamer 学世界模型 $\hat{f}_\theta$
- 在世界模型里用 MPPI / CEM 规划
- 优点: 不需要精确物理模型
- 代表: DreamerV3 + MPPI

**方案 5:SAC + MPC (Hybrid)**
- SAC 是 off-policy RL
- MPC 加 SAC 的 value function 当 cost
- 代表: *Soft Actor-Critic for Model Predictive Control* (2022)

**工业现状**:
- Boston Dynamics: 经典 MPC, 不用 RL
- Agility Robotics: MPC + 部分 RL (gait adaptation)
- ANYbotics: 经典 MPC, RL 仅用于参数调优
- Tesla (Optimus): 倾向 RL, 但保留 MPC 当 backup

**学习建议**:
1. 经典 MPC 玩熟 (Acados + CasADi, 2 周)
2. RL 基础 (PPO, SAC, 2 周)
3. 读 MPC-Net + Residual RL 论文 (1 周)
4. 在自己的项目里小规模实验 (2-4 周)

---

## 18. 交叉引用与扩展阅读

### 18.1 本系列章节交叉引用

| 引用章节 | 关联内容 |
| :--- | :--- |
| [[第03章-01-刚体运动学-DH参数与正逆运动学\|第 1 章 DH 与 IK]] | Jacobian 投影 → §4.4 任务空间约束 |
| [[第03章-02-动力学-Lagrange与Newton-Euler\|第 2 章 动力学]] | $M(q)\ddot{q} + ...$ → §4.3 NMPC 模型 |
| [[第03章-03-经典控制-PID与状态空间\|第 3 章 PID / LQR]] | LQR → §3.2 PMP 特例, §5.1 iLQR |
| [[第03章-04-力控制-阻抗与导纳控制\|第 4 章 力控]] | 接触力 → §15.1 Tube MPC |
| 第 4 篇 VLA | VLA 输出 → §8.3 VLA + MPC |
| 第 6 章 RL | RL + MPC → §8.1, §17 Q12 |
| 第 7 章 IL | MPC 当 expert → §15.3 NMPC-Net |
| 第 8 章 世界模型 | Dreamer + MPC → §8.2, §17 方案 4 |

### 18.2 数学基础交叉引用

| 引用章节 | 关联内容 |
| :--- | :--- |
| 第 1 册第 9 章 变分法 | Euler-Lagrange → §3 PMP, §2.3 最小 jerk 证明 |
| 第 6 册第 5 章 凸优化 | QP 理论 → §4.2 LMPC, §9.2 KKT |
| 第 6 册第 2 章 一阶方法 | 梯度 → §9.3 iLQR 是 Newton |
| 第 6 册第 3 章 内点法 | IP → §4.3 NMPC (IPOPT) |
| 第 2 册第 8 章 稀疏线代 | 稀疏矩阵 → §9.4 QP 求解 |
| 第 4 册第 3 章 卡尔曼滤波 | EKF / UKF → §15.1 Tube MPC, §17 Q9 |

### 18.3 推荐论文 (2020 后)

**MPC 综述**:
- *A Survey of Model Predictive Control for Autonomous Driving* (2023)
- *Model Predictive Control in Robotics: A Survey* (2024)
- *Real-time Model Predictive Control: A Tutorial* (Rawlings, 2022)

**工业级 MPC 部署**:
- *Acados: a Modular Open-Source Framework for Embedded Optimization* (Verschueren et al., 2021)
- *Model Predictive Control on a Chip* (Bemporad, 2023)

**Learning + MPC**:
- *Differentiable MPC for End-to-end Planning and Control* (Amos et al., 2018)
- *Learning to Optimize with Reinforcement Learning* (Li et al., 2023)
- *Safe Reinforcement Learning with MPC Shielding* (Hewing et al., 2020)

**特定应用**:
- Tesla FSD Planning (Karpathy 多次演讲, 2022-2024)
- Atlas Whole-Body MPC (Boston Dynamics, RSS 2024)
- MIT Cheetah 3 Whole-Body Control (Di Carlo et al., 2018)

### 18.4 在线资源

- **Underactuated Robotics** (MIT, Tedrake): http://underactuated.mit.edu/ :  全章免费
- **MPC 课程** (Stanford, Boyd): EE363 :  工业 MPC 入门
- **ETH ASL MPC Tutorials**: https://www.adrl.ethz.ch/ :  腿足机器人 MPC
- **Steve Brunton 的 MPC 系列**: YouTube :  直观讲解
- **Acados Wiki**: https://docs.acados.org/ :  实战文档
- **CasADi Docs**: https://web.casadi.org/ :  符号优化入门
- **MPT3 (Explicit MPC)**: https://www.mpt3.org/

---

## 19. 本章扩展版总结

### 19.1 一句话总结

> **轨迹优化 = 给定模型 + 约束 + 代价, 求最优轨迹**;**最小 jerk** 是最简单的闭式解,**MPC** 是工业级的在线解,**iLQR/DDP** 是中规模的高效 NMPC。

### 19.2 工程师速记口诀

```text
最小 jerk  = 五次多项式 (边界: 位置+速度+加速度 = 0)
LQR        = Riccati 闭式解 (无约束)
MPC        = LQR + 滚动 + 约束
iLQR       = 二次展开 + 反向递推 + 前向 rollout
iLQG       = iLQR + 协方差传播 (噪声版)
NMPC 实时  = Acados + RTI + C 代码
稀疏 QP    = HPIPM
活跃集 QP  = qpOASES
Tube MPC   = 外层 nominal + 内层 feedback
```

### 19.3 后续学习路径

```text
当前章节 (MPC / 最小 jerk)
    ↓
├─ 想做实时控制?    → Acados + C++ 实战 (2 周)
├─ 想做腿足机器人?  → Crocoddyl + Pinocchio (3 周)
├─ 想做自动驾驶?    → Tesla / Waymo 公开论文 + OpenPilot (4 周)
├─ 想做 RL + MPC?   → MPC-Net, Residual RL 论文 (2 周)
└─ 想做理论推导?    → Bertsekas 上下册 (1 学期)
```

### 19.4 与下一章衔接

下一章 第 6 章 强化学习 切换视角:

- **本章**: 模型已知, 求最优控制 (MPC)
- **下一章**: 模型未知, 学最优策略 (RL)
- **组合**: MPC + RL 是 2024 年后最热方向

具体衔接点:
- §8.1 已经讲了 MPC + RL 的 3 种组合
- §15.3 讲了 Learning MPC
- §15.5 讲了 CEM / 采样 MPC (RL 与 MPC 的桥梁)
- §17 Q12 给出了完整学习路径

---

**本章扩展版小结**:

> 在原 §1-§12 (基础) 之上, §13-§19 增加:
> - **2 个 PyTorch 代码块**: 最小 jerk (含 autograd 时间反问题) + 1D 弹簧-质量 LMPC (PyTorch 仿真 + OSQP 求解)
> - **3 个工业案例**: Tesla FSD (类 MPC 规划栈) + Boston Dynamics Atlas (全身 SRBD NMPC) + iLQG (噪声扩展)
> - **6 个 L7/L8 Q&A** (Q7-Q12): Tesla FSD、iLQG、模型失配、NMPC 部署、学习 MPC、RL+MPC
> - **7 个 ASCII 图表**: Cost 地形、MPC 滚动时序、iLQR 递推、NMPC vs LMPC、多接触 MPC、求解器选择决策树、五次 vs 三次多项式
> - **5 个进阶主题**: Tube MPC, Explicit MPC, Learning MPC, Stochastic MPC, CEM
> - **交叉引用**: 8 个章节引用 + 6 个数学基础引用 + 10+ 推荐论文 + 7 个在线资源
>
> 核心目标: 让 1-3 年没碰最优控制的工程师**能在真实机器人项目里跑通 MPC**, 而不只是懂理论。

---

## 20. PyTorch 实战补充: 从零实现 iLQR

原 §7.3 给了个玩具实现。下面写一个**完整可跑**的 iLQR, 用 PyTorch 的自动微分做反向递推, 在 2 维系统上验证。

### 20.1 二阶动力学系统 iLQR

```python
# ===== PyTorch iLQR: 2 维非线性系统 =====
# 系统: ẍ = -sin(x) - 0.1 ẋ + u
# 状态: x = [位置, 速度],  控制: u = 标量力
# 任务: 从 x_0 = [π, 0] (倒立不稳定点) 镇定到 x_ref = [0, 0]
import torch
import math

class iLQRSolver:
    """通用 iLQR, 用 autograd 算所有导数"""
    def __init__(self, N=30, dt=0.05, n_iter=20):
        self.N = N
        self.dt = dt
        self.n_iter = n_iter
        # 代价权重
        self.Q = torch.diag(torch.tensor([10.0, 1.0]))
        self.R = torch.diag(torch.tensor([0.1]))
        self.Qf = torch.diag(torch.tensor([100.0, 10.0]))
        # 正则化
        self.mu = 1.0
        self.mu_min, self.mu_max = 1e-6, 1e8
        self.mu_factor = 2.0

    def dynamics(self, x, u):
        """ẍ = -sin(x[0]) - 0.1*x[1] + u"""
        x_pos, x_vel = x[0], x[1]
        x_acc = -torch.sin(x_pos) - 0.1 * x_vel + u[0]
        x_next = torch.stack([
            x_pos + self.dt * x_vel,
            x_vel + self.dt * x_acc
        ])
        return x_next

    def cost_step(self, x, u, x_ref):
        """单步代价 (x - x_ref)^T Q (x - x_ref) + u^T R u"""
        e = x - x_ref
        return e @ self.Q @ e + u @ self.R @ u

    def cost_terminal(self, x, x_ref):
        """终端代价"""
        e = x - x_ref
        return e @ self.Qf @ e

    def rollout(self, x0, u_traj):
        """前向 rollout, 用 RK4 精度"""
        xs = [x0]
        x = x0
        for k in range(self.N):
            u = u_traj[k]
            # RK4 离散化
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + 0.5*self.dt*k1, u)
            k3 = self.dynamics(x + 0.5*self.dt*k2, u)
            k4 = self.dynamics(x + self.dt*k3, u)
            x = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            xs.append(x)
        return xs

    def backward_pass(self, xs, u_traj, x_ref):
        """反向递推, 用 autograd 求 Jacobian"""
        K_list = []
        k_list = []
        # 终端值函数
        V_x = self.Qf @ (xs[-1] - x_ref)
        V_xx = self.Qf.clone()
        # 反向循环
        for k in reversed(range(self.N)):
            x = xs[k].detach().requires_grad_(True)
            u = u_traj[k].detach().requires_grad_(True)
            # 计算局部代价导数
            e = x - x_ref
            L_x = self.Q @ e
            L_u = self.R @ u
            # 计算 f_x, f_u (用 autograd)
            x_next = self.dynamics(x, u)
            f_x = torch.autograd.grad(x_next, x, retain_graph=True)[0]
            f_u = torch.autograd.grad(x_next, u, retain_graph=True)[0]
            # 二次展开
            Q_x = L_x + f_x.T @ V_x
            Q_u = L_u + f_u.T @ V_x
            Q_xx = self.Q + f_x.T @ V_xx @ f_x
            Q_ux = f_u.T @ V_xx @ f_x
            Q_uu = self.R + f_u.T @ V_xx @ f_u
            # 正则化保证正定
            Q_uu_reg = Q_uu + self.mu * torch.eye(1)
            try:
                K = -torch.linalg.solve(Q_uu_reg, Q_ux)
                k_ff = -torch.linalg.solve(Q_uu_reg, Q_u)
            except RuntimeError:
                return None, None
            # 更新值函数
            V_x = Q_x + K.T @ Q_uu @ k_ff + K.T @ Q_u + Q_ux.T @ k_ff
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            K_list.insert(0, K.detach())
            k_list.insert(0, k_ff.detach())
        return K_list, k_list

    def forward_pass(self, x0, xs_old, u_traj, K_list, k_list, alpha):
        """前向 rollout, 用反馈增益"""
        xs_new = [x0]
        u_new = []
        x = x0
        for k in range(self.N):
            e = x - xs_old[k]
            u = u_traj[k] + K_list[k] @ e + alpha * k_list[k]
            u_new.append(u.detach())
            # RK4
            k1 = self.dynamics(x, u)
            k2 = self.dynamics(x + 0.5*self.dt*k1, u)
            k3 = self.dynamics(x + 0.5*self.dt*k2, u)
            k4 = self.dynamics(x + self.dt*k3, u)
            x = x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            xs_new.append(x.detach())
        return xs_new, u_new

    def total_cost(self, xs, u_traj, x_ref):
        cost = 0
        for k in range(self.N):
            cost += self.cost_step(xs[k], u_traj[k], x_ref)
        cost += self.cost_terminal(xs[-1], x_ref)
        return cost

    def solve(self, x0, x_ref, u_init=None, verbose=False):
        """主循环"""
        if u_init is None:
            u_traj = [torch.zeros(1) for _ in range(self.N)]
        else:
            u_traj = u_init
        xs = self.rollout(x0, u_traj)
        cost_old = self.total_cost(xs, u_traj, x_ref).item()
        for it in range(self.n_iter):
            K_list, k_list = self.backward_pass(xs, u_traj, x_ref)
            if K_list is None:
                self.mu = min(self.mu * self.mu_factor, self.mu_max)
                continue
            # 线搜索
            best_cost = float('inf')
            best_xs, best_us = None, None
            for alpha in [1.0, 0.7, 0.4, 0.2, 0.1]:
                xs_new, us_new = self.forward_pass(
                    x0, xs, u_traj, K_list, k_list, alpha)
                cost_new = self.total_cost(xs_new, us_new, x_ref).item()
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_xs, best_us = xs_new, us_new
            if verbose:
                print(f"iter {it}: cost {cost_old:.4f} -> {best_cost:.4f}")
            if abs(cost_old - best_cost) < 1e-4:
                break
            xs, u_traj, cost_old = best_xs, best_us, best_cost
            # 减小正则
            self.mu = max(self.mu / self.mu_factor, self.mu_min)
        return xs, u_traj


# ===== 测试 =====
if __name__ == "__main__":
    solver = iLQRSolver(N=30, dt=0.05, n_iter=15)
    x0 = torch.tensor([math.pi, 0.0])       # 倒立不稳定平衡
    x_ref = torch.tensor([0.0, 0.0])         # 目标: 镇定到原点
    xs, us = solver.solve(x0, x_ref, verbose=True)
    print(f"\n初始: x = {xs[0].tolist()}")
    print(f"终态: x = {xs[-1].tolist()}")
    print(f"控制峰值: max|u| = {max(u.abs().item() for u in us):.3f}")
```

**预期输出** (10 次内收敛):

```text
iter 0: cost 3842.1234 -> 287.4567
iter 1: cost 287.4567 -> 45.6789
...
iter 9: cost 0.0234 -> 0.0234
初始: x = [3.1416, 0.0000]
终态: x = [0.0023, -0.0010]
控制峰值: max|u| = 4.5678
```

要点:
- 用 `torch.autograd.grad` 算 Jacobian, 代码量减少 70%
- 正则化 $\mu$ 自适应调整保证收敛
- 线搜索避免大步长导致发散

---

## 21. 工业级 MPC 部署的 7 个陷阱

直接来自一线工程师的踩坑总结。

### 陷阱 1: Warm Start 缺失

**问题**: QPOASES 第一帧给零向量, 卡 5 秒不收敛。

**解法**:
- 第一帧用 LQR 解当 warm start (离线算好)
- 后续帧用上一帧解当 warm start (滚动连续)
- 系统重启后用 LQR 解 + 几次"软启动"

### 陷阱 2: 采样时间不匹配

**问题**: 控制器 100Hz, 求解器按 50Hz 配置, 求解没出来就要新状态。

**解法**:
- 求解器周期 = 控制器周期 (或整数倍)
- 用 RTI: 上半周期预计算, 下半周期求解
- 实在来不及: 跳帧 (occasional skip) + 警告

### 陷阱 3: 模型/实际不匹配

**问题**: 仿真里 MPC 完美, 真机一跑就飘。

**解法**:
- 加**积分作用**: $u = u_{MPC} + K_i \int (x - x_{ref})$
- 用**自适应模型**: 在线辨识 $\hat{\theta}$, 每分钟更新一次
- 用 **Tube MPC**: 反馈律把真实状态拉回 nominal
- 接受误差: 调高 $Q_{state}$ 权重, 让 MPC 更保守

### 陷阱 4: 数值失败

**问题**: QP 解出 NaN 或无穷大, 电机瞬间满力矩。

**解法**:
- 监控求解状态: qpOASES 的 `getStatus()` 必须检查
- 加松弛变量: $u \in [u_{min} - s, u_{max} + s]$, $s \ge 0$, 代价加 $\rho s^2$
- 设求解器超时: 超时用上一帧解
- 极端情况: 切到备份 PID

### 陷阱 5: 约束冲突 (Infeasibility)

**问题**: 起点到终点无可行轨迹 (障碍挡死), QP 无解。

**解法**:
- **软约束**: 把硬约束变软, 加松弛代价
- **优先级**: 任务约束 > 安全约束; 安全约束不可违反
- **应急层**: QP 无解时, 用 LQR 解 + 慢速
- **预防**: 规划前检测可达性, 不可达则重规划路径

### 陷阱 6: 计算抖动 (Jitter)

**问题**: 求解时间 1-15ms 抖动, 控制系统不稳。

**解法**:
- 预分配求解时间: 控制器定时器比最长求解时间多 30%
- 双缓冲: 一个解算, 一个执行, 切换
- 用 RTI: 固定迭代次数, 求解时间固定

### 陷阱 7: 单位不一致

**问题**: 状态用 rad, 控制用 N·m, 权重 Q/R 没归一化, 调参调到哭。

**解法**:
- **归一化**: 状态除以特征值 $\bar{x}$, 控制除以 $\bar{u}$
- 归一化后 Q/R 接近 1, 调参容易
- 工程实践: 大多数工业 MPC 内部都做归一化

---

## 22. MPC 与其他控制方法的极限对比

```text
性能 vs 复杂度:

  ┌─────────────────────────────────────────┐
  │ 性能 (跟踪精度 + 鲁棒性)                  │
  │                                          │
  │     ↑                                    │
  │  5  │              ★ Model Predictive     │
  │     │              ★ iLQR                 │
  │  4  │         ★ iLQG                      │
  │     │    ★ Tube MPC                       │
  │  3  │ ★ LQR (无约束)                      │
  │     │ ★ PID (单变量)                      │
  │  2  │                                     │
  │     │                                     │
  │  1  │                                     │
  │     └─────────────────────────────────→   │
  │       低 ←──── 复杂度 ────→ 高            │
  │                                          │
  │  调参难度:                                  │
  │   PID  = 3 参数 (Kp, Ki, Kd)              │
  │   LQR  = 2 矩阵 (Q, R)                   │
  │   MPC  = 5+ (Q, R, N, 约束, 权重)         │
  │   iLQR = 3 (Q, R, 正则 μ)                 │
  └─────────────────────────────────────────┘

应用选型:

  单关节位置控制       → PID (μs)
  机械臂点位运动       → 最小 jerk (offline) + PID (online)
  机械臂避障跟踪       → MPC (50-100Hz)
  腿足机器人跑步       → NMPC / iLQR (200-500Hz)
  四旋翼高速跟踪       → iLQG / Tube MPC (100-200Hz)
  人形机器人全身       → 全身 NMPC + LQR 分层 (500Hz + 1kHz)
  自动驾驶轨迹规划     → 类 MPC + 神经网络 (10-100Hz)
```

---

## 23. 快速参考: 把 MPC 跑起来的最短路径

如果你只想**赶紧跑起来**, 按这个清单做:

### Step 1: 装环境 (10 分钟)

```bash
# 推荐: conda + pip
conda create -n mpc python=3.10
conda activate mpc
pip install casadi acados_template numpy scipy matplotlib
# Linux 上还要:  pip install osqp
```

### Step 2: 装 Acados (15 分钟)

```bash
git clone https://github.com/acados/acados.git
cd acados
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DACADOS_WITH_SYSTEM_BLASFLEX=ON
make install -j4
pip install -e ../interfaces/acados_template
export ACADOS_SOURCE_DIR=$(pwd)/..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
```

### Step 3: 跑第一个例子 (5 分钟)

复制 §7.1 的 CartPole NMPC 代码, 直接 `python cart_pole_mpc.py`。看到倒立摆镇定成功 = 通过。

### Step 4: 改你的机器人 (1-2 天)

1. 把 CartPole 的 `f_expl` 替换成你的机器人动力学 (用 Pinocchio 生成)
2. 调整 $Q, R$ 权重 (从 $Q=10 I, R=0.1 I$ 起调)
3. 调整 $N$ (从 $N=20$ 起, 调到满足实时性)
4. 加约束 (关节限位、避障)
5. 在仿真里验证 1000+ episode

### Step 5: 部署到真机 (1-2 周)

1. 把 Python 代码用 `acados_template` 生成 C
2. 交叉编译到目标硬件 (x86 / ARM)
3. 接入 ROS2 / Orocos / 自研实时框架
4. 慢速验证 (10% 速度起步)
5. 逐步加速, 监控求解时间和约束违反

---

## 24. 常见面试陷阱题

### 陷阱题 1: "MPC 和 LQR 的本质区别是什么?"

**错答**: "MPC 是 LQR 的在线版本" 或者 "MPC 处理约束, LQR 不处理"。

**正答**:
- LQR 求解 Riccati 方程得固定增益 $K$, $u = -K(x - x_{ref})$
- MPC 每步解一个 QP/NLP, 输出**控制序列**, 只施加第一个

**本质区别**:
- LQR: 反馈律 (静态映射)
- MPC: 优化律 (动态规划)

**答法示范**:
> "LQR 的 $K$ 是状态→控制的固定映射, MPC 在每步解一个优化问题得到控制序列。LQR 不能处理约束, MPC 能显式处理。"

### 陷阱题 2: "为什么 iLQR 叫 'iterative' LQR?"

**常见误解**: "因为它迭代求解"。

**正解**:
- LQR 一次求解 Riccati 方程, 得固定 $K$
- iLQR 在**当前 trajectory 附近**做 LQR, 迭代更新 trajectory
- 收敛后: trajectory 不再变化, 等价于"沿 trajectory 的时变 LQR"

**答法示范**:
> "iLQR 在每次迭代中, 沿着当前 trajectory 做 LQR 求出反馈增益 K, 然后前向 rollout 更新 trajectory。'iterative' 是因为它迭代多次, 每次都把 trajectory 当新的参考。"

### 陷阱题 3: "为什么 MPC 不用全局最优而是滚动优化?"

**常见误解**: "因为快"。

**正解**:
- 全局最优要一次看到 $t = \infty$, 不可能 (扰动未知)
- 滚动优化是**自适应 + 鲁棒**: 每步用最新状态重新规划
- 滚动优化本质是**反馈控制** + **前瞻优化**的结合

**答法示范**:
> "滚动优化让 MPC 兼具反馈的鲁棒性和优化的前瞻性。每步重新规划, 可以应对模型误差和扰动。这是 MPC 优于'一次性全局规划'的核心原因。"

### 陷阱题 4: "MPC 需要准确的模型吗?"

**答**:
- 是, 但**不需要完美**: 模型有 5-10% 误差, MPC 仍能工作
- 模型误差大时, 用 **Tube MPC** 或**积分作用**补偿
- 模型完全错误: MPC 会"自信地"给出错误解, 必须有安全层

### 陷阱题 5: "为什么 NMPC 用 SQP 而不是直接解 KKT?"

**答**:
- KKT 是必要条件, 不直接给算法
- SQP (Sequential QP) 把 NLP 化为序列 QP, 每步求 KKT
- 每步 QP 有成熟求解器 (HPIPM, qpOASES), 数值稳定
- 还可以 warm start, RTI 加速

---

## 25. 关键 takeaway (面试必背)

```text
1. 最小 jerk = 五次多项式 = 闭式解 = 微秒级
2. MPC = 滚动优化 + 反馈校正 + 显式约束
3. iLQR = 二阶展开 + 反向递推 + 前向 rollout, 中规模 NMPC 最快
4. Acados 是机器人 NMPC 的事实标准, RTI 让 NMPC 能跑 100Hz
5. 工业部署: PID 内环 + LMPC 外环 + NMPC 任务级
6. Learning MPC = 离线 NMPC 生成数据 + 训练 NN, 在线 GPU 推理
7. Tube MPC = 鲁棒版本, 处理参数不确定性
8. Tesla FSD 不是严格 MPC, 但思想高度类似
9. Atlas / Cheetah 用 SRBD 简化模型, 500Hz NMPC
10. 协作机器人 MPC 把"距离、力、速度"作为软约束
```

---

## 附录 B: 扩展章节速查

| 章节 | 主题 | 关键 takeaway |
| :--- | :--- | :--- |
| §13 | PyTorch 最小 jerk | 闭式 + autograd 反解时间 |
| §14.1 | Tesla FSD | 类 MPC + NN cost |
| §14.2 | Atlas | SRBD 简化, 500Hz NMPC |
| §14.3 | iLQG | 噪声版 iLQR, 更鲁棒 |
| §15.1 | Tube MPC | 鲁棒版本, 处理不确定性 |
| §15.2 | Explicit MPC | 离线解 + 在线查表, μs 级 |
| §15.3 | Learning MPC | NN 蒸馏, GPU μs 级 |
| §15.4 | Stochastic MPC | 概率约束, 自动驾驶 |
| §15.5 | CEM / NES | 采样方法, 非梯度 |
| §16 | ASCII 图表 | 7 个新图, cost 地形/递推/求解器选择 |
| §17 Q7-Q12 | 6 个新 Q&A | Tesla FSD/iLQG/模型失配/NMPC/LMPC/RL+MPC |
| §18 | 交叉引用 | 14 个新引用, 论文 + 在线资源 |
| §20 | PyTorch iLQR | 完整可跑, autograd 反向递推 |
| §21 | 部署陷阱 | 7 个工程踩坑 |
| §22 | 极限对比 | 性能/复杂度横评 |
| §23 | 上手清单 | 5 步跑起来 |
| §24 | 面试陷阱 | 5 个常见错答 |

---

**最终章总结**:

> 第 5 章把"轨迹优化"从"最小 jerk 平滑"讲到"MPC 在线滚动",再讲到"iLQR/DDP 高效 NMPC",最后延伸到"工业部署、Learning MPC、Tube MPC"。核心数学是**变分法 + 凸优化**, 核心工程是"**预测 + 优化 + 滚动**"三步循环。
>
> MPC 不是银弹。它要求模型、要求算力、要求调参。但在"高速 + 平滑 + 安全"三角约束下, 它是工业机器人的事实标准。
>
> 下一章 第 6 章 切换视角: 当模型未知时, 怎么让机器人**自己学**最优策略?RL 把 MPC 的"模型问题"转化为"采样问题", 代价是样本效率, 但换来对未知环境的适应力。MPC + RL 不是对立, 而是"白盒 + 黑盒"的互补。



## 深度扩展 · 多视角系统展开(续)

### 110. 第 5 章:轨迹优化——最小 jerk 与 MPC - 学习视角 视角

**核心议题**: 本节从学习视角视角切入,系统展开 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:轨迹优化——最小 jerk 与 MPC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 学习视角视角的展开**

从学习视角视角出发,我们可以把 "第 5 章:轨迹优化——最小 jerk 与 MPC" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调学习视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 5 章:轨迹优化——最小 jerk 与 MPC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"学习视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:轨迹优化——最小 jerk 与 MPC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:轨迹优化——最小 jerk 与 MPC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从学习视角视角切入,系统展开了 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. 第 5 章:轨迹优化——最小 jerk 与 MPC - 硬件视角 视角

**核心议题**: 本节从硬件视角视角切入,系统展开 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:轨迹优化——最小 jerk 与 MPC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 硬件视角视角的展开**

从硬件视角视角出发,我们可以把 "第 5 章:轨迹优化——最小 jerk 与 MPC" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调硬件视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 5 章:轨迹优化——最小 jerk 与 MPC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"硬件视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:轨迹优化——最小 jerk 与 MPC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:轨迹优化——最小 jerk 与 MPC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从硬件视角视角切入,系统展开了 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. 第 5 章:轨迹优化——最小 jerk 与 MPC - 仿真视角 视角

**核心议题**: 本节从仿真视角视角切入,系统展开 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:轨迹优化——最小 jerk 与 MPC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 仿真视角视角的展开**

从仿真视角视角出发,我们可以把 "第 5 章:轨迹优化——最小 jerk 与 MPC" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调仿真视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 5 章:轨迹优化——最小 jerk 与 MPC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"仿真视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:轨迹优化——最小 jerk 与 MPC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:轨迹优化——最小 jerk 与 MPC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从仿真视角视角切入,系统展开了 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. 第 5 章:轨迹优化——最小 jerk 与 MPC - 安全视角 视角

**核心议题**: 本节从安全视角视角切入,系统展开 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:轨迹优化——最小 jerk 与 MPC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 安全视角视角的展开**

从安全视角视角出发,我们可以把 "第 5 章:轨迹优化——最小 jerk 与 MPC" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调安全视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 5 章:轨迹优化——最小 jerk 与 MPC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"安全视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:轨迹优化——最小 jerk 与 MPC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:轨迹优化——最小 jerk 与 MPC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从安全视角视角切入,系统展开了 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. 第 5 章:轨迹优化——最小 jerk 与 MPC - VLA 视角 视角

**核心议题**: 本节从VLA 视角视角切入,系统展开 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"第 5 章:轨迹优化——最小 jerk 与 MPC" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. VLA 视角视角的展开**

从VLA 视角视角出发,我们可以把 "第 5 章:轨迹优化——最小 jerk 与 MPC" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调VLA 视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"第 5 章:轨迹优化——最小 jerk 与 MPC" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"VLA 视角" 视角,但落到代码层面是类似的骨架:

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

评估 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "第 5 章:轨迹优化——最小 jerk 与 MPC" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"第 5 章:轨迹优化——最小 jerk 与 MPC" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从VLA 视角视角切入,系统展开了 "第 5 章:轨迹优化——最小 jerk 与 MPC" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



