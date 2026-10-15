---
title: "第07章-06-SAPIEN与DexArt-灵巧手仿真"
tags:
  - 具身智能
  - SAPIEN与DexArt
  - 灵巧手仿真
description: "本章要点：2023 年, 你读到 DexArt 论文 (Bao et al., RSS 2023), 作者声称 Shadow Hand 训了一个策略, 仿真成功率…"
chapter_id: 具身智能-6-07
last_updated: 2026-07-14
---

# 第07章-06-SAPIEN与DexArt-灵巧手仿真

过灵巧手仿真的工程师的话**:
>
> 你在 MuJoCo 里训过 HalfCheetah, 调过 Panda 的 7 自由度, 甚至写过 Shadow Hand 抓方块的脚本。但当你打开一篇 2023 年的 SAPIEN/DexArt 论文, 大概率会卡在三个地方:
>
> - **灵巧手到底有多少自由度?** Shadow Hand 24 关节听起来很多, 但"能做什么"和关节数没有线性关系。一个 5 指 20 关节的手在 6-DoF 末端运动下控制空间是 $6 + 20 = 26$ 维, 动作空间小则几万、复杂任务上百万, BC 根本扛不住。
> - **SAPIEN 的接触求解到底比 MuJoCo 强在哪?** 物理引擎之争在 2020 年之后基本定型: MuJoCo 强在数值稳定, SAPIEN(PhysX)强在大量并发接触 + 真实摩擦模型 + 大场景。你抓一个塑料瓶, SAPIEN 会算每个指尖-瓶身的接触点 + 法向力分布, MuJoCo 默认 LCP 只给总反力。
> - **DexArt 任务为什么 sim2real gap 小?** 因为 DexArt 的任务场景就是真实室内物体(马桶、键盘、抽屉、水龙头), 几何与动力学双重要求, 仿真里训的策略必须能在真实硬件上跑通。
>
> 这一章用"灵巧手仿真工程师"的视角拆开 SAPIEN + DexArt:
>
> - 从 **灵巧操作 vs 平行夹爪的本质差异** 出发, 搞清楚"为什么 5 指比 2 指难一个数量级"
> - 拆 **SAPIEN 的架构**(Scene / Actor / Joint / Drive / Contact / Renderer), 用一张类图讲清楚"装一台灵巧手需要调用哪些 API"
> - 拆 **PhysX 接触求解器**(PGS → TGS → GPU pipeline), 为什么灵巧手仿真必须用 GPU
> - 对比 **Shadow Hand / LEAP Hand / Allegro / Inspire** 四种主流灵巧手, 给出自由度 / 扭矩 / 成本对比
> - 拆 **DexArt 8 个任务**(buckle / mouse / keyboard / syringe / toilet / faucet / door / bucket), 解释为何这些任务可以学到真机能用的策略
> - 给出 **端到端可运行的 PyTorch 实验**, 在 SAPIEN 里加载 Shadow Hand, 跑一个简化的 reach 任务, 可视化接触点
> - 互引数学基础: 接触力学(LCP / 库仑锥)、控制论(冗余机械臂零空间)、凸优化(Contact Invariant Optimization)
>
> **学完这一章, 你能回答的问题**:
>
> - Shadow Hand 和 LEAP Hand 该怎么选? 一个 24 关节一个 16 关节, 对学习算法影响多大?
> - SAPIEN 的接触点数量瓶颈是多少? 为什么训练 4096 个并行环境时 GPU 显存比 CPU 慢?
> - DexArt 的 8 个任务里, 哪个 sim2real gap 最小? 为什么 faucet 这么难?
> - 灵巧手任务为什么需要 domain randomization? 接触点的微小偏移会让策略失效吗?
> - 用 SAPIEN 训练 Shadow Hand 抓方块需要多少样本? RL? IL?
>
> **与本文件夹其他章节的关系**:
> - [[第07章-01-MuJoCo-XML建模与控制]], SAPIEN 早期版本用 MuJoCo, 后改 PhysX。两者接触模型差异巨大(MuJoCo 的 LCP vs PhysX 的 PGS), 任务数值会差 30%。
> - [[第07章-02-Isaac-Sim-工业级GPU仿真]], Isaac Lab 用 PhysX 5 GPU, SAPIEN 用 PhysX 4 CPU, 工程差异要知道。
> - [[第07章-03-Habitat-室内导航仿真]], Habitat 解决导航, SAPIEN/DexArt 解决灵巧操作, 两者底层都是 UE4 渲染管线。
> - [[第07章-04-ManiSkill-操作任务基准]], ManiSkill v1/v2 物理后端是 SAPIEN, v3 转自家 ManiSkill Engine, 这条历史线要清楚。
> - [[第07章-05-RoboCasa与行为树]], RoboCasa 偏家庭操作 + 行为树, SAPIEN/DexArt 偏单任务 + 灵巧手 + Sim2Real。
> - **互引其他文件夹**:
>   - `../../../基础/数学基础/第06章-最优化/第06章-05-凸优化理论-强凸·光滑·Lipschitz·收敛率证明.md`, 接触不变优化(CIO)在灵巧手轨迹规划里的应用。
>   - `数学基础/8-控制论/3-李群与刚体运动`, 接触点法向量的 SO(3) 表达。
>   - `../第03章-机器人运动学与控制/0-总目录与阅读路径.md`, 接触事件的概率建模。
>   - `../第01章-具身智能概论/0-总目录与阅读路径.md`, 摩擦锥作为分布约束。

---

## 1. 直觉引入:为什么灵巧手仿真这么难?

### 1.1 一个会让你算力崩盘的故事

2023 年, 你读到 DexArt 论文 (Bao et al., RSS 2023), 作者声称 Shadow Hand 训了一个策略, 仿真成功率 78%, 真机搬到 Allegro 上 71%。你 pull 代码, 跑 `python train.py --task buckle --hand shadow`, 第 6 小时, 第 1.2 万 step, CUDA OOM。

排查一周, 发现三个你不会在 MuJoCo 遇到的问题:

1. **接触点太多, GPU 显存爆掉**。Shadow Hand 24 关节, 5 指, 每指 3 段, 每段 + 关节 + 周围场景, 光接触对(contact pair)就有 2000+。PhysX 默认把每个潜在接触对存在 GPU, 4096 个并行环境直接爆。
2. **摩擦锥违反**, 一抓就飞。SAPIEN 默认库仑摩擦系数是工业标准 0.5, 但你的 Shadow Hand 默认指尖橡胶是 1.2。一旦摩擦反了, 接触约束的方向改变, 看似同样的动作, 输出完全不同的抓取轨迹。
3. **仿真步长对结果敏感**。灵巧手任务 dt 一般要 ≤ 1/240 s, 否则指尖关节的接触瞬间被积分器吃掉。同样 50 步, MuJoCo 默认 1/100 s 跑 0.5 s, SAPIEN 1/240 s 跑 0.21 s, episode 长度变了, success 重新定义。

这些坑 MuJoCo 玩家不会碰到。但灵巧手仿真, 这三个问题是入场门票。

### 1.2 灵巧手仿真到底难在哪?

把"灵巧手仿真"四个字拆开, 难点有四层:

**层 1: 高自由度控制**

```text
机械手          指   关节/手  末端   控制维度   任务难度
──────────────────────────────────────────────────────
Panda(平行夹爪)  2   7        6      7          ⭐
Allegro         4   16       16     16         ⭐⭐⭐
Inspire         5   12       12     12         ⭐⭐⭐
LEAP Hand       5   16       16     16         ⭐⭐⭐
Shadow Hand     5   24       24     24         ⭐⭐⭐⭐⭐
Adroit(24 自由度,≈ Shadow) 24                ⭐⭐⭐⭐⭐
```

每个关节都是 1-DoF 旋转, 总状态空间 = 关节角 + 关节速 + 接触点法向 + 摩擦锥方向。BC 在 24 维空间里训, 100 演示根本不够, 一般要 1000-10000 演示。

**层 2: 接触动力学的强非线性**

"抓"这个动作看起来是连续的, 实际上是离散接触事件的组合。指尖接近物体表面 → 接触建立 → 法向力上升 → 切向摩擦支撑 → 抬升。一个灵巧操作任务的接触事件典型要 30-100 个, 每个都是 if-then 触发。

```text
指尖轨迹                  物理事件
────────────────────────────────────────────────────
接近 (10cm → 1cm)     无接触, 自由空间轨迹规划
接触建立 (1cm)        第一个 contact pair 出现, 法向力 ≈ 0
抓稳 (手掌内)        5+ 接触点支撑, 库仑锥闭合
抬起 (10cm)          物体受重力 + 接触反力, 平衡配置
移动 / 旋转           多个接触点滑移 / 重分布
释放 (放置)          接触点全部消失, 物体自由落体
```

SAPIEN / PhysX 这层做得好, 但代价是每步都要算所有接触对的法向冲量, 计算量 O(N²) 涨。

**层 3: 触觉与本体感知的高维耦合**

灵巧手操作不是纯视觉任务, 接触力是另一个观察通道。Shadow Hand 指尖有 92 个触觉传感器(taxel), 每个给出 3 轴力, 加上 5 指 × 4 段 × 1 关节角 = 20 个本体感受, 状态维度 92×3 + 20 = 296 维。MuJoCo 这层只仿真关节角, 没触觉。

**层 4: Sim2Real 的多源漂移**

- **几何差**: 仿真里指尖是理想刚体 + 橡胶, 真实 Shadow Hand 指尖有弹性变形, 接触面积比仿真大 5-15%。
- **物理参数差**: 仿真里库仑摩擦系数定值, 真实取决于表面湿度、磨损。
- **执行器差**: Shadow Hand 真实电机的电流环响应时间 5-20 ms, 仿真假设理想。
- **传感器差**: 真实触觉传感器有迟滞、有温度漂移、有非均匀分布。

这四层叠加, 灵巧手任务的 sim2real gap 通常比夹爪任务大 20-40%。

### 1.3 SAPIEN 的定位:不是 MuJoCo 的替代,是补集

SAPIEN(Scene-wise Articulated Physics-based Interactive ENvironment, Mu et al., 2019)不是一个独立物理引擎, 它是 **NVIDIA PhysX 的 Python 包装 + 强化的关节物体建模 + 高质量渲染层**。

```text
生态层级                  SAPIEN 提供
──────────────────────────────────────────────────
底层物理                NVIDIA PhysX 4(CPU, 不是 GPU)
关节物体                Articulation (joint tree), 强于 URDF
接触求解                PhysX TGS(Temporal Gauss-Seidel) + PGS
高级接触约束            库仑锥, 各向异性摩擦, 黏附
渲染                    UE4 / OpenGL 可选, RTX 光线追踪可选
任务层                  ADROIT / DexArt / ManiSkill v1 / others
资源层                  PartNet-Mobility / ShapeNet / YCB / SAPIEN Assets
许可                    Apache 2.0
语言                    Python (C++ 核心编译为 Python 绑定)
```

**和 MuJoCo 的对比**:

| 维度 | MuJoCo | SAPIEN (PhysX 4) |
| :--- | :--- | :--- |
| 接触求解 | LCP, LCP-ng | LCP + PGS + TGS |
| 多接触点 | 中(每对象几百) | 强(每对象上万) |
| 摩擦模型 | 库仑 + 黏附 | 库仑 + 各向异性 + 黏附 |
| 关节树 | 通用 MJCF | 强 URDF 风格 articulation |
| 软体 | 弱 | 中(PhysX soft body) |
| GPU 仿真 | MuJoCo XLA(2023) | 不支持原生 GPU(CPU 串行) |
| 灵巧手模型 | 内置 Shadow Hand, Allegro | 内置 Shadow, Allegro, LEAP, Inspire |
| 渲染 | 可选 GL + 可选光线追踪 | UE4 + 可选 RTX |
| 步长控制 | implicit Euler, 稳定 | semi-implicit Euler, 快但敏感 |
| 数值稳定性 | 高 | 中(接触多时偶发) |

**核心结论**: MuJoCo 强在数值稳定, SAPIEN 强在接触动力学 + 大场景 + 灵巧手支持。选哪个看任务: 灵巧手选 SAPIEN, RL 单臂抓取选 MuJoCo, 工业数字孪生选 Isaac Lab。

### 1.4 灵巧操作 vs 平行夹爪的本质区别

| 维度 | 平行夹爪 (Panda, Robotiq 2F-85) | 灵巧手 (Shadow, LEAP) |
| :--- | :--- | :--- |
| 末端执行 | 二指对夹, 2 接触点 | 多指包络, 5+ 接触点 |
| 控制维度 | 7-DoF | 16-24-DoF |
| 抓取策略 | pincer grasp(夹), 1 模式 | envelop / pinch / hook / tripod, 多种 |
| 适用物体 | 规则形状(方块, 圆柱) | 不规则 / 可变形 / 复杂几何 |
| 通用性 | 低(每个物体需新 grasp 设计) | 高(泛化抓取姿态学习) |
| 训练成本 | RL 100k 步成功 | RL 1-10M 步成功 |
| 学术热度 | 高(2015-2020 主线) | 极高(2020 后主线) |
| 工业落地 | 高(夹爪便宜稳定) | 低(造价 5-20 万美元) |

核心结论: **夹爪擅长"夹", 灵巧手擅长"操作"**。开瓶盖、插 U 盘、叠毛巾、用筷子, 夹爪要复杂末端工具, 灵巧手直接做。这是 SAPIEN/DexArt 这条线热的根本原因。

### 1.5 章节路线图

```text
§1 直觉引入                 ← 你在这里
§2 SAPIEN 基础              ← 本章核心,装一台灵巧手要调用什么 API
§3 DexArt 基准              ← 8 个任务怎么定义,为什么能 sim2real
§4 灵巧手操作               ← Shadow / LEAP / Allegro / Inspire 对比
§5 PyTorch 实验             ← 装 Shadow Hand,跑 reach 任务,可视化接触
§6 与其他章节的关系
§7 与数学基础的关系
§8 常见误区
§9 L7/L8 面试 Q&A
§10 参考文献
```

---

## 2. SAPIEN 基础:物理引擎、接触求解、渲染

### 2.1 安装与配置

**最小安装**(CPU, 单环境, 学习用):

```bash
# 推荐用 conda,避免系统库冲突
conda create -n sapien python=3.8 -y
conda activate sapien

# SAPIEN 核心(只 CPU,最轻)
pip install sapien

# 装好之后验证
python -c "import sapien; print(sapien.__version__)"
# 输出: 3.0.0 或更新
```

**完整安装**(含 GPU 渲染 / 关节物体 demo):

```bash
# 推荐安装方式 sapien官方发布版本
pip install sapien==3.0.0

# 如果要 ray-tracing 渲染(可选, 需要 RTX 卡)
pip install sapien[vulkan]

# 如果要 ManiSkill v2 环境(旧版后端)
pip install mani-skill==2.4.0
```

**常见安装问题**:

| 错误 | 原因 | 解法 |
| :--- | :--- | :--- |
| `libvulkan not found` | 没装 vulkan driver | `apt install libvulkan1` |
| `Cannot find Vulkan ICD` | 显卡驱动没装 | 重装 NVIDIA driver |
| `dlopen: cannot load PhysX` | GLIBC 太老 | 用 conda 装 libstdcxx |
| `No module named sapien.render` | 没装 vulkan 扩展 | `pip install sapien[vulkan]` |
| 仿真跑但无画面 | EGL 没配好 | 用 OSMesa 软渲染 |

**硬件推荐**:

```text
规模                CPU                GPU              适用
───────────────────────────────────────────────────────────
学习(单 env)        4 核               可选              跑 demo, 看代码
研究(8 env)         8 核               RTX 3080          BC 训 baseline
中等(64 env)        16 核              RTX 4090          灵巧手 RL
大规模(512+ env)    32+ 核             A100             大规模策略训练
```

注: SAPIEN **物理 step 只跑 CPU**, 不像 Isaac Lab 那样原生 GPU 仿真。所以"4096 env"在 SAPIEN 上只能 CPU 多进程, 而 ManiSkill3 4096 env 是真 GPU。计算效率和显存差异大。

### 2.2 SAPIEN 架构总览

SAPIEN 整体结构像一棵"场景树":

```text
Sapien Engine (核心命名空间)
├── Engine (仿真引擎, 持有 PhysX scene)
│   ├── Renderer (渲染管线)
│   ├── Scene (仿真场景)
│   │   ├── Actor (刚体 / 关节物体)
│   │   │   ├── Link (刚体部件)
│   │   │   └── Joint (关节)
│   │   ├── Light
│   │   └── Camera
│   ├── Contact (接触点信息)
│   └── Pose / Transform
├── Asset (URDF / MJCF / OBJ 资源加载)
└── utils / ros / viewer (辅助)
```

**Core 数据流**:

```text
┌──────────────────────────────────────────────────────────┐
│           Sapien Scene (CPU 主线程, 串行)               │
│                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  Actor 1 │    │  Actor 2 │    │  Shadow  │         │
│   │  (桌子)  │    │  (物体)  │    │   Hand   │         │
│   └─────┬────┘    └─────┬────┘    └─────┬────┘         │
│         │              │              │                  │
│         ▼              ▼              ▼                  │
│   ┌──────────────────────────────────────────┐         │
│   │     PhysX 4 接触求解 (PGS / TGS)         │         │
│   └──────────────────────────────────────────┘         │
│         │                                              │
│         ▼                                              │
│   ┌──────────────────────────────────────────┐         │
│   │   Step: integrate contacts + dynamics    │         │
│   └──────────────────────────────────────────┘         │
│         │                                              │
│         ▼                                              │
│   Renderer (CPU / GL / Vulkan / RTX)                    │
└──────────────────────────────────────────────────────────┘
```

### 2.3 物理引擎:PhysX 内核

SAPIEN 直接调用 **NVIDIA PhysX SDK 4.x** 作为物理引擎。这意味着 SAPIEN 没有自己的物理求解器, 所有动力学和接触都是 PhysX 算的, SAPIEN 只是包装。

**PhysX 的关键求解器**:

1. **PGS (Projected Gauss-Seidel)**:
   - 求解 LCP (Linear Complementarity Problem): $A x + b = x$, $x \ge 0$
   - 接触约束 + 摩擦约束一次性投影, 高斯-赛德尔迭代
   - 优点: 简单, 稳定
   - 缺点: 接触多时 O(N²), 慢

2. **TGS (Temporal Gauss-Seidel)**:
   - PGS 的时域增强版, 多帧松弛, 利用时间相关性
   - 收敛更快, 但实现复杂
   - PhysX 默认 solver mode

3. **岛 (Island) 处理**:
   - 把物理上不相关的 actor 分到不同岛
   - 一个岛内并行求解, 岛之间串行
   - 灵巧手 + 物体 + 桌面 算一个岛(全部接触耦合)

**动力学方程**(以刚体为例):

$$M(q) \ddot{q} + C(q, \dot{q}) + G(q) = \tau + J^T F_{\text{contact}}$$

其中 $M$ 是质量矩阵, $C$ 是科里奥利项, $G$ 是重力, $\tau$ 是关节力矩, $J^T F_{\text{contact}}$ 是接触反力对广义坐标的投影。PhysX 用半隐式欧拉积分:

$$\dot{q}_{t+1} = \dot{q}_t + h \cdot M^{-1} (\tau + J^T F_{\text{contact}} - C - G)$$
$$q_{t+1} = q_t + h \cdot \dot{q}_{t+1}$$

注意 **接触反力 $F_{\text{contact}}$ 依赖位置 $q_{t+1}$ 和速度 $\dot{q}_{t+1}$**, 因为接触点法向和摩擦方向都依赖当前姿态。这就是 LCP 求解的来源。

**接触反力求解**:

PhysX 把接触建模为 **N 个法向约束 + 切向摩擦约束**。每个约束是非穿透:

$$g_N(q) \ge 0$$
$$F_N \ge 0$$
$$g_N(q) \cdot F_N = 0 \quad (\text{互补松弛})$$

切向摩擦满足 **库仑锥**:

$$\|F_T\|_2 \le \mu F_N$$

每个接触点生成这些约束, 加到全局 LCP 一起解。Shadow Hand + 物体 + 桌面的系统, 接触约束典型 30-150 条。

### 2.4 接触求解:库仑锥与各向异性摩擦

SAPIEN 在 PhysX 默认摩擦之上做了两个增强:

**1. 各向异性摩擦** (SAPIEN 强项):

很多物体在不同方向摩擦系数不同(比如木纹 / 拉丝金属)。SAPIEN 支持给 link 配各向异性摩擦张量:

$$
\mu_{\text{aniso}} =
\begin{pmatrix}
\mu_x & 0 & 0 \\
0 & \mu_y & 0 \\
0 & 0 & \mu_z
\end{pmatrix}
$$

库仑锥相应变为椭球锥。这对手指抓塑料瓶(瓶身有纹路)、抓纸片(摩擦方向敏感)很重要。

**2. 接触点可视化**:

SAPIEN 提供 `get_contact_points()` API, 返回所有接触点对 `(actor_a, actor_b, position, normal, force, torque)`。这是研究接触动力学的关键工具, 不像 MuJoCo 默认只在 console print。

```python
# 取接触点
contacts = scene.get_contacts()  # List[Contact]

for c in contacts:
    actor0 = c.actor0.name
    actor1 = c.actor1.name
    point = c.position  # (3,) 世界坐标系
    normal = c.normal    # (3,) 接触法向(从 actor0 指向 actor1)
    force = c.force      # (3,) 接触反力
    print(f"{actor0} ↔ {actor1}: 接触点 {point}, 法向 {normal}, 力 {force}")
```

### 2.5 关节物体建模

灵巧手是关节物体的极致: 24 个关节, 5 个手指, 每个手指是 4 段 + 4 关节的树结构。SAPIEN 用 **Articulation (关节物体)** 表达:

```text
shadow_hand (Articulation)
├── arm (Link) ─── palm_mount (Joint, fixed)
│   ├── wrist (Link) ─── wrist_joint (Joint, revolute, axis=x)
│   ├── forearm (Link) ─── forearm_joint (Joint, revolute, axis=x)
│   └── ...
├── thumb (Link chain)
│   ├── thumb_base (Joint, revolute)
│   ├── thumb_middle (Joint, revolute)
│   └── ...
├── index_finger (Link chain)
│   ├── index_base (Joint, revolute)
│   ├── index_middle (Joint, revolute)
│   └── ...
├── middle_finger (similar)
├── ring_finger (similar)
└── pinky_finger (similar)
```

**关节的关键属性**:

```python
joint = articulation.get_joints()[0]  # 取第一个关节
joint.set_drive_target(0.5)        # 设目标位置 (rad or m)
joint.set_drive_velocity_target(1.0)  # 设目标速度 (rad/s)
joint.set_drive_stiffness(1000.0)  # 刚度 (Nm/rad)
joint.set_drive_damping(50.0)      # 阻尼 (Nm·s/rad)
joint.set_friction(0.5)            # 关节摩擦
```

**PD 控制**(默认行为):

每个关节内部是 PD 控制器:

$$\tau = K_p (\theta_{\text{target}} - \theta) - K_d \dot{\theta}$$

SAPIEN 默认 $K_p = 1000$, $K_d = 50$。这两个值对所有关节默认一致, 但对灵巧手而言 **不同关节需要不同 $K_p$**(手掌根部要硬, 指尖要软, 否则接触时 PID 注入额外能量把物体弹飞)。

### 2.6 渲染层:UE4 与 OpenGL

SAPIEN 提供多种渲染后端:

| 后端 | 特点 | 速度 | 视觉 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `VulkanRenderer` (默认) | Vulkan + OptiX 光追 | 快 | 高 | 主流, 支持 RGBD |
| `RayTracingRenderer` | RTX 光线追踪 | 中 | 极高 | 真实感 + 大场景 |
| `GLRenderer` | OpenGL 兼容 | 慢 | 低 | 旧硬件 fallback |

**典型渲染代码**:

```python
import sapien.core as sapien

engine = sapien.Engine()
renderer = sapien.VulkanRenderer(offscreen_only=True)  # 不开窗口
engine.set_renderer(renderer)

scene = engine.create_scene()
scene.set_timestep(1/240)  # 物理 dt

# 加载桌面 + Shadow Hand + 物体
# ... (后面 §5 详写)

# 单步仿真
scene.step()

# 渲染 RGB
camera = scene.add_camera(
    width=128, height=128, fov=1.0,
    near=0.01, far=100,
    mount=robot_link  # 装在手上或者别的 link
)
camera.set_pose(sapien.Pose([0.5, 0, 0.5], [0, 0, 0, 1]))  # 世界位姿
rgb = camera.get_color_rgba()  # (H, W, 4) uint8
depth = camera.get_depth()     # (H, W) float32
```

**渲染对灵巧操作的影响**:

仿真器层面的视觉差异是 sim2real gap 第三大来源(仅次于物理参数和执行器延迟)。SAPIEN 默认渲染是"近物理"但有偏差:

- **阴影**: 仿真阴影是纯几何 + Lambertian, 没有全局光照, 真实 Shadow Hand 的金属关节 + 橡胶指尖有次表面散射, 视觉差异 5-15%。
- **纹理**: SAPIEN 支持 PBR 材质但默认弱, 真实物体有划痕、指纹、脏污, 这是 sim2real 大坑。
- **运动模糊**: 仿真无运动模糊, 真实相机快门 1-10 ms, 灵巧手快速动作时模糊可见。

Mitigation: 加 **视觉域随机化**(VAE 改光照、AddNoise、加纹理), 在训练时让策略见过"漂移"的图。

### 2.7 物理参数调节

SAPIEN 物理参数全在 PhysXSceneDesc 里:

```python
scene = engine.create_scene(
    gravity=[0, 0, -9.81],
    timestep=1/240,
    contact_offset=0.005,        # 接触探测距离 (m)
    rest_offset=0.002,           # 静止距离 (m)
    solver_iterations=8,         # PGS 迭代次数
    friction_threshold=0.01,     # 静摩擦阈值
    enable_adaptive_force=True,  # 自适应接触力
    enable_tgs=True,             # 用 TGS solver
    bounce_threshold=0.5,        # 反弹阈值
    sleep_threshold=0.05         # sleep 阈值
)
```

**灵巧手任务的典型参数**:

```python
# 在 SAPIEN 默认配置基础上改两个:
scene.timestep = 1/240         # 灵巧手任务必须 ≤ 1/240, 不则指尖关节穿透物体
scene.enhanced_determinism = True  # 增强确定性(同一 seed 出同一轨迹)
```

`enhanced_determinism` 让 PhysX 用确定性的浮点累加顺序, 关掉之后有些 sort 会引入微小差异, 影响 RL 训练稳定性。

### 2.8 SAPIEN 与 ManiSkill 的关系(简短)

SAPIEN 的"任务层"早期由 ManiSkill v1/v2 提供(2021-2023), 但 ManiSkill v3(2024)脱离 SAPIEN, 自家 ManiSkill Engine(PhysX 5 GPU)。这是行业路线分歧: SAPIEN 走"仿真器开源", ManiSkill 走"任务生态开源"。

如果你只是要跑 DexArt 任务, 直接装 SAPIEN + DexArt 即可(2023 版)。但如果你要做大规模 RL 训练(>256 env), ManiSkill v3 + Isaac Lab 是更优解。

---

## 3. DexArt 基准:灵巧手任务定义

### 3.1 为什么需要 DexArt?

灵巧手任务在 2018-2022 年没有一个公认基准。各家训自己的版本: OpenAI 训 Rubik's Cube(单指机器人 + 触觉), Berkeley 训 Adroit(hand manipulation benchmark), Stanford 训 SHAP(Shadow Hand Activity Playbook)。每个都基于不同物理引擎, 跑不出对照数字。

DexArt (Bao et al., RSS 2023) 的赌注是: **用一个统一的灵巧手仿真任务集, 把 sim2real 的差距打到 5% 以内**。它有 8 个任务, 全部用真实室内物体几何, 全部可以搬到真 Allegro / Shadow Hand 跑。

### 3.2 DexArt 任务集合(8 个)

| 任务 ID | 物体 | 动作 | 难度 | 仿真成功率 | 真机成功率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `buckle` | 安全带扣 | 把安全带卡入卡扣 | ⭐⭐⭐ | 78% | 71% |
| `mouse` | 鼠标 | 按鼠标左键 | ⭐⭐⭐⭐ | 65% | 60% |
| `keyboard` | 键盘 | 按指定键 | ⭐⭐⭐⭐ | 52% | 48% |
| `syringe` | 注射器 | 推动推杆到指定刻度 | ⭐⭐⭐⭐ | 60% | 56% |
| `toilet` | 马桶 | 按下冲水按钮 | ⭐⭐⭐ | 70% | 65% |
| `faucet` | 水龙头 | 旋转打开 | ⭐⭐⭐⭐⭐ | 35% | 30% |
| `door` | 门把手 | 握住把手下压开锁 | ⭐⭐⭐⭐⭐ | 38% | 32% |
| `bucket` | 桶 | 握住桶手柄 | ⭐⭐⭐ | 75% | 68% |

**任务分级**:

- **简单类(⭐⭐⭐)**: toilet, bucket, buckle。几何简单, 接触面积大, 单指尖主导。成功率高, sim2real gap 小。
- **中等类(⭐⭐⭐⭐)**: mouse, keyboard, syringe。多指协同, 时序依赖, 但有空间容差。
- **困难类(⭐⭐⭐⭐⭐)**: faucet, door。需精确力 + 旋转, 容差 < 5°。

**为什么这些任务能 sim2real**:

1. **几何真实**: 每个物体都是从 PartNet-Mobility 或真实扫描建模, 几何精度 ±1mm。
2. **物理真实**: 摩擦系数、阻尼、刚度按真实测量, 不是随机设的。
3. **接触容差友好**: 任务的 success threshold 不是 0 cm, 而是 2-5 cm 的容差。这对 sim2real 友好, 因为真实控制不可能毫米级。
4. **动作空间一致**: 全部用 Shadow Hand 的 24 维关节角度空间(归一化 -1 到 1), 真机控制接口直接搬。
5. **评估协议统一**: 每个任务用 50 个不同初始位姿, 跑 3 个种子, 算 mean ± std。

### 3.3 DexArt 的核心创新:双 loss 设计

传统 IL 方法在灵巧手任务上效果差, 因为 24 维动作空间 + 接触突变, 演示数据稀疏。Bao et al. 提了两个 loss:

**1. 行为克隆 loss (BC Loss)**:

$$L_{BC} = \| a_t - \pi(o_t) \|^2$$

其中 $a_t$ 是专家动作(关节角度 delta), $o_t$ 是状态(RGB + 本体感觉 + 触觉)。

**2. 残差动作 loss**:

$$L_{R} = \| a_t - a_{t}^{\text{teacher}} \|^2$$

其中 $a^{\text{teacher}}$ 是教师网络(预训练)输出的"软"动作。这两个 loss 加权:

$$L = \lambda_1 L_{BC} + \lambda_2 L_{R}$$

默认 $\lambda_1 = 1, \lambda_2 = 0.5$。

**关键**:

- BC loss 让策略尽量拟合演示。
- 残差 loss 让策略不要偏离教师太远, 保证初期训练稳定。
- 两者一起, 在 8 个任务上平均涨 15-20%。

### 3.4 DexArt 的评估协议

**评估流程**:

```python
# env.reset() 取定初始状态
obs = env.reset(seed=42)

# 推策略
done = False
while not done:
    action = policy(obs)
    obs, reward, done, info = env.step(action)

# info["success"] 是布尔, 任务相关定义
# 例:buckle 任务: 安全带底端中心距卡扣中心 < 0.5 cm 且姿态差 < 5°
```

**指标**:

| 指标 | 定义 | 用途 |
| :--- | :--- | :--- |
| `success_rate` | 100 episode 成功比例 | 主指标 |
| `first_success_step` | 首次成功 episode 的步数 | 收敛速度 |
| `action_smooth` | 相邻动作差分的 L2 norm | 策略平滑度 |
| `contact_force_avg` | 平均接触力 (N) | 物理合理性 |
| `penetration_depth` | 接触穿透深度 (mm) | 仿真质量 |

**评估陷阱**:

1. **seed 选择**: DexArt 用 `seed ∈ {0, 1, ..., 99}` 报告成功率, **不能用 train seed**。否则过拟合。
2. **初始位姿分布**: 物体初始位姿来自 `dataset.json`, **不能改**, 否则不算 DexArt。
3. **任务级 max_episode_steps**: 不同任务不同(buckle 80, faucet 120), **不能改**, 否则容易截断成功 episode。

### 3.5 DexArt 与 Adroit / MetaWorld 的对比

| 维度 | Adroit (2020) | MetaWorld (2020) | DexArt (2023) |
| :--- | :--- | :--- | :--- |
| 仿真器 | MuJoCo | MuJoCo | SAPIEN (PhysX 4) |
| 灵巧手 | Adroit (24-DoF) | 无灵巧手 | Shadow Hand (24-DoF) |
| 任务数 | 5 | 50 | 8 |
| 物体类型 | 简单 (pen, hammer) | 工具 | 真实家具物件 |
| sim2real | 较低 | N/A | 较高 |
| 评估 | 50 demo | 测试 50 任务 | 100 seed × 8 任务 |
| 任务复杂度 | 中 | 中 | 高 |
| 真机支持 | 有 | 无 | 有 (Allegro + Shadow) |

**关键差异**: Adroit 的手指是 Adroit 自己设计的 (Stanford), DexArt 用 Shadow Hand 这个真实商业手。Adroit 的 sim2real 在 Stanford 实验里 ~30%, DexArt 在 Berkeley 实验里 ~85% (这是它主要的卖点)。

### 3.6 DexArt 训练数据来源

8 个任务的演示数据怎么来? DexArt 用了三种方式:

1. **人类遥操作 (Human Teleop)**: PhaseSpace + VR 头显, 操作员用控制器驱动 Shadow Hand, 20-30 分钟一个任务。
2. **脚本策略 (Scripted Policy)**: 用反向运动学 + 任务相关的轨迹生成器, 自动生成 100-1000 演示。
3. **强化学习教师 (RL Teacher)**: 用 PPO / SAC 在仿真里训到 90% 成功率, 然后录完整轨迹作为演示。

每个任务 100-500 演示。**够不够?** 对 BC 不够, 所以原论文用了残差 loss + 多任务迁移。对 IL + 残差, 够了。

### 3.7 DexArt 任务的代码骨架(简化)

```python
import sapien.core as sapien
from sapien.core import Pose
import numpy as np

# 创建仿真
engine = sapien.Engine()
renderer = sapien.VulkanRenderer(offscreen_only=True)
engine.set_renderer(renderer)

scene = engine.create_scene()
scene.set_timestep(1/240)

# 加载桌面 + 物体 + Shadow Hand
ground = scene.add_ground(altitude=0)
table = load_table(scene)
object = load_buckle_object(scene)
hand = load_shadow_hand(scene, position=[0.3, 0, 0.5])

# 相机
cam = scene.add_camera(width=128, height=128, fov=1.2, near=0.01, far=100)
cam.set_pose(Pose([0.8, 0, 0.8], look_at_object=[0.3, 0, 0.5]))

# RL 循环
obs = env.reset(seed=42)
for step in range(80):
    action = policy(obs)  # 24 维 Shadow Hand 关节 delta
    obs, reward, done, info = env.step(action)
    if done:
        break
```

---

## 4. 灵巧手操作:Shadow / LEAP / Allegro / Inspire

### 4.1 灵巧手自由度控制基础

灵巧手本质是一个冗余机械臂 + 多指末端。控制空间有 N 个关节, 任务是末端位姿(6-DoF) + 多指抓握姿态(k-DoF)。

**控制数学模型**:

设手有 $n$ 个关节(Shadow Hand $n=24$), 总状态 $\theta \in \mathbb{R}^n$。动力学:

$$M(\theta) \ddot{\theta} + C(\theta, \dot{\theta}) + G(\theta) = \tau_{\text{act}} + J_c^T F_c$$

其中 $F_c$ 是所有接触点的反力。冗余度 $n - 6 = 18$, 让手可以做"末端不动 + 手指调整"的事情。这对操作非常重要。

**零空间优化**:

经典控制里, 利用冗余度的零空间投影:

$$\dot{\theta} = J^\dagger \dot{x}_{\text{end}} + (I - J^\dagger J) z$$

其中 $J$ 是末端雅可比, $J^\dagger$ 伪逆, $z$ 是任意优化目标(比如最小化关节能量)。这在灵巧手操作里是基础, MuJoCo 的 IK 求解器就是这套思路。

### 4.2 Shadow Hand (Shadow Robot, 英国, 2005-)

**参数**:

```text
规格                     数值
──────────────────────────────────────
指                       5(拇指 + 4 指)
总关节数                 24
驱动器                   20 个气动肌肉 + 4 个电机
自由度(主动)             20
自由度(被动)             4(指间的虚关节)
最大抓握力               50 N (whole hand)
指尖力 (单指)            10 N
重量                     4.3 kg
造价                     ~$100k
通讯协议                 EtherCAT
主传感器                 关节角 + 指尖 6D 力矩
触觉传感器               BioTac(可选)
```

**优势**: 工业级精度, 气动 + 电机混合驱动响应快, BioTac 触觉好。

**劣势**: 造价高, 维护复杂, 软件 stack 老旧, 仿真模型精度受限(BioTac 触觉仿真成本巨大)。

### 4.3 LEAP Hand (Stanford / Meta, 2023)

LEAP = LEgged-Arm-Platform Hand, 是 Stanford 与 Meta AI 联合开发, **专为零成本/低成本研究** 设计。

**参数**:

```text
规格                     数值
──────────────────────────────────────
指                       5(拇指 + 4 指)
总关节数                 16
驱动器                   16 个微型伺服(全部电机)
自由度(主动)             16
自由度(被动)             0
最大抓握力               30 N(whole hand)
指尖力 (单指)            6 N
重量                     1.1 kg
造价                     ~$2k (DIY)
通讯协议                 USB / CAN / 串口
主传感器                 关节角 + 可选指尖力
触觉传感器               可选(可换 3rd party)
```

**优势**: 极低造价, 全电机驱动简单, 与 Allegro 软件兼容, URDF + MJCF 模型开源。

**劣势**: 集成度低, 力矩小, 触觉要外加, 真机 sim2real gap 比 Shadow Hand 略大(2024 数据约 5-10%)。

### 4.4 Allegro Hand (Wonik Robotics, 韩国, 2012-)

**参数**:

```text
规格                     数值
──────────────────────────────────────
指                       4(拇指 + 3 指, 没有小指)
总关节数                 16
驱动器                   16 个电机
自由度(主动)             16
最大抓握力               25 N
指尖力                   5 N
重量                     1.16 kg
造价                     ~$25k
通讯协议                 CAN / Ethernet
主传感器                 关节角 + 指尖电流估算
触觉传感器               可选
```

**优势**: 商业化最早, 4 指(比 5 指简单), 软件 stack 成熟, 价格适中。

**劣势**: 无小指(限制一些抓取姿态), 触觉标配差。

### 4.5 Inspire Hand (Inspire Robotics, 中国, 2021-)

**参数**:

```text
规格                     数值
──────────────────────────────────────
指                       5(拇指 + 4 指, 含小指)
总关节数                 12
驱动器                   12 个电机
自由度(主动)             12
最大抓握力               20 N
指尖力                   4 N
重量                     0.55 kg (业内最轻)
造价                     ~$8k
通讯协议                 EtherCAT
主传感器                 关节角 + 可选指尖力
触觉传感器               可选(三方)
```

**优势**: 5 指中重量最轻(0.55 kg), 价格最低, 含小指(全 5 指)。

**劣势**: 关节数少(12), 一些灵巧姿态做不到, 触觉标配弱。

### 4.6 四种灵巧手横向对比

| 维度 | Shadow | LEAP | Allegro | Inspire |
| :--- | :--- | :--- | :--- | :--- |
| 关节数 | 24 | 16 | 16 | 12 |
| 主动自由度 | 20 | 16 | 16 | 12 |
| 最大抓握力 (N) | 50 | 30 | 25 | 20 |
| 指尖力 (N) | 10 | 6 | 5 | 4 |
| 重量 (kg) | 4.3 | 1.1 | 1.16 | 0.55 |
| 造价 (美元, 量级) | 100k | 2k | 25k | 8k |
| 触觉 | BioTac | 第三方 | 可选 | 第三方 |
| 工业级 | 强 | 弱 | 中 | 弱 |
| 学术研究 | 中 | 强(2024 起) | 强 | 弱 |
| 仿真支持 | 优(SAPIEN / MuJoCo) | 优 | 优 | 中 |

**选型决策树**:

```text
任务?
├── sim2real 工业装配        → Shadow Hand
├── 大规模 RL / IL 研究      → LEAP Hand
├── 标准化商业研究           → Allegro
├── 轻量化服务机器人         → Inspire
└── 学习入门(便宜)           → LEAP Hand (DIY)
```

### 4.7 抓取 vs 操控

灵巧手可以分两大任务: **抓(grasp)** 和 **操作(manipulate)**。两者难度差一个量级:

**抓 (Grasp)**:

```text
目标: 把物体稳定抓住, 抵抗重力
时间尺度: 0.5-2 s
策略: 选择接触点 + 接近 + 闭合
评估: 是否握住 + 不掉
难度: ⭐⭐
```

**操作 (Manipulate)**:

```text
目标: 用手 + 物体完成某个子任务(开瓶、转门、写字)
时间尺度: 2-30 s
策略: 序列接触 + 在手内重定向
评估: 任务特定成功指标
难度: ⭐⭐⭐⭐⭐
```

DexArt 的 8 个任务全部是 manipulate, 没一个 simple grasp。原因是 dex2sim 的核心难题就是 manipulate。

**in-hand manipulation** (手内重定向):

最高难度的操作: 物体被抓住后, 通过指尖调整把它从 A 姿态变到 B 姿态(不松手):

```text
物体被 5 指握住 → 食指和中指轻推 → 物体旋转 90° → 抓稳继续推
```

这种任务的 RL 训练是 OpenAI 在 2018 (Rubik's Cube) 之后才真训通的, 而且 hand 是 Dactyl(自己设计, 24-DoF, $500k 造价)。

### 4.8 工业上的应用

灵巧手在工业上的真实落地不多(2024 数据), 主要在:

1. **特种装配**: 航天器内部狭小空间连接(Shadow Hand, 4k 美元一天租赁)。
2. **手术机器人**: Intuitive Surgical 的 da Vinci 系统用的不是灵巧手, 是多关节器械, 但 DexArt 的 syringe 任务直接对应外科注射。
3. **服务机器人**: Tesla Optimus(Gen 2) / Figure 01 / Unitree H1 都用了 5 指灵巧手, 但工业级稳定性还在追赶人类。
4. **物流分拣**: Amazon Picking Challenge (2015-2017) 用了夹爪, 灵巧手优势在异形物, 还没普及。

工业上大规模使用灵巧手的时间, 大概率不会早于 2027-2028。原因是 **sim2real 还不够 + 维护成本高 + 触觉欠成熟**。

### 4.9 灵巧手训练与评估的资源消耗

| 任务 | 算法 | 演示数 | 算力 | 时间 |
| :--- | :--- | :--- | :--- | :--- |
| Shadow 抓方块 | BC | 100 | 1 RTX 4090 | 4 hr |
| DexArt buckle | IL + 残差 | 200 | 1 RTX 4090 | 8 hr |
| DexArt faucet | IL + RL 微调 | 500 + 100k 步 | 8 × A100 | 30 hr |
| OpenAI Rubik's Cube | PPO + 域随机 | 0 演示 | 384 × CPU + 16 GPU | 48 hr |

工业上做灵巧手研究, **一台 A100 + 1-2 周** 是入门门槛。

---

## 5. PyTorch 实验:在 SAPIEN 里装 Shadow Hand 跑 Reach

### 5.1 环境准备

```python
# 文件:sapien_reach_demo.py
# 环境:conda activate sapien
# 运行:python sapien_reach_demo.py

import sapien.core as sapien
from sapien.core import Pose
import numpy as np
import matplotlib.pyplot as plt

# 初始化引擎与渲染
engine = sapien.Engine()
renderer = sapien.VulkanRenderer(offscreen_only=True, device="cuda:0")
engine.set_renderer(renderer)

scene = engine.create_scene(gravity=[0, 0, -9.81])
scene.set_timestep(1/240)

# 加灯光
scene.add_ground(altitude=0)
light = scene.add_directional_light(
    direction=[0, 0, -1],
    color=[1.0, 1.0, 1.0],
    shadow=True
)
```

### 5.2 加载 Shadow Hand 模型

```python
# 创建 Shadow Hand articulation(关节物体)
hand_loader = scene.create_urdf_loader()
hand_loader.fix_root_link = False
hand_loader.disable_self_collision = False  # 自碰撞检测
hand = hand_loader.load(
    "/path/to/shadow_hand.urdf",
    material=scene.create_physical_material(
        static_friction=1.5, dynamic_friction=1.2, restitution=0.0
    )
)
hand.set_root_pose(Pose([0, 0, 0.5], [0, 0, 0, 1]))  # 抬升 50 cm

# 查看所有关节
joints = hand.get_joints()
print(f"Shadow Hand 总关节数: {len(joints)}")
for i, j in enumerate(joints[:6]):
    print(f"关节 {i}: 名称={j.name}, 类型={j.type}, 限制={j.get_limits()}")
# 输出:
# 关节 0:名称 = wrist_joint, 类型 = revolute, 限制 = [(-1.5708, 1.5708)]
# ...
```

### 5.3 加载目标物体 + 随机化初始位姿

```python
# 加一个目标物体(红色方块)
box_actor = scene.add_box(
    half_size=[0.02, 0.02, 0.02],  # 4 cm 立方体
    material=scene.create_physical_material(
        static_friction=0.6, dynamic_friction=0.5, restitution=0.1
    ),
    density=500  # 轻木
)
box_actor.set_pose(Pose([0.3, 0, 0.02]))  # 桌面摆放

# 目标位姿(可视化用)
goal_pose = Pose([0.3, 0.15, 0.5], [0, 0, 0, 1])

# PD 控制器设置
for joint in hand.get_joints():
    joint.set_drive_stiffness(1000.0)
    joint.set_drive_damping(50.0)
```

### 5.4 Reach 任务的 PD 控制循环

```python
def forward_kinematics(hand):
    """计算各指尖的世界坐标位置"""
    finger_links = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    positions = {}
    for link in hand.get_links():
        if link.name in finger_links:
            pose = link.get_pose()  # 末端 link 的世界 pose
            positions[link.name] = pose.p  # (3,) np.ndarray
    return positions


def run_reach_episode(scene, hand, steps=200):
    """跑一个 reach episode, 让指尖到达目标"""
    initial_joint_angles = [j.get_drive_target() for j in hand.get_joints()]
    goal_pos = np.array([0.3, 0.15, 0.5])  # 目标指尖位置

    trajectory = []
    for step in range(steps):
        finger_tips = forward_kinematics(hand)
        middle_tip = finger_tips["middle_tip"]
        # 计算到目标的误差
        err = goal_pos - middle_tip
        # 用极简 PD 控制器:每个关节加一个小 delta 让指尖接近目标
        # 实际中用雅可比, 这里用简化近似
        for j in hand.get_joints():
            target = j.get_drive_target()
            # 简化为腕关节向上倾
            if "wrist" in j.name:
                j.set_drive_target(target + 0.02 * err[2] / max(np.linalg.norm(err), 0.01))
            elif "forearm" in j.name:
                j.set_drive_target(target + 0.01 * err[1] / max(np.linalg.norm(err), 0.01))
            else:
                j.set_drive_target(target)

        scene.step()
        trajectory.append(middle_tip.copy())

    return np.array(trajectory)


traj = run_reach_episode(scene, hand, steps=240)
print(f"指尖轨迹形状:{traj.shape}")  # (240, 3)
```

### 5.5 接触点可视化

```python
# 跑完后可视化接触点
def visualize_contacts(scene, step_count=200):
    """渲染并收集接触点"""
    contact_data = []
    contact_renderer = renderer

    for step in range(step_count):
        scene.step()
        contacts = scene.get_contacts()
        frame_contacts = []
        for c in contacts:
            frame_contacts.append({
                "point": c.position.copy(),
                "normal": c.normal.copy(),
                "force_mag": np.linalg.norm(c.force),
                "actor0": c.actor0.name,
                "actor1": c.actor1.name,
            })
        contact_data.append(frame_contacts)

    return contact_data


contacts = visualize_contacts(scene, step_count=120)

# 统计接触事件
total_pairs = sum(len(c) for c in contacts)
avg_pairs = total_pairs / len(contacts)
print(f"120 步总接触对:{total_pairs}")
print(f"平均每步接触对:{avg_pairs:.1f}")

# 找最大法向力
max_force = 0
for frame in contacts:
    for c in frame:
        if c["force_mag"] > max_force:
            max_force = c["force_mag"]
            print(f"step max force: {max_force:.3f} N, "
                  f"{c['actor0']} ↔ {c['actor1']}")
```

### 5.6 渲染图像采集

```python
# 加一台相机
cam = scene.add_camera(width=128, height=128, fov=1.2, near=0.01, far=100)
cam.set_local_pose(
    Pose(p=[0.8, 0, 0.8], q=[0, 0.707, 0, 0.707])
)
scene.step()

# 渲染 RGB
rgb = cam.get_color_rgba()  # (128, 128, 4)
print(f"RGB 形状:{rgb.shape}, dtype:{rgb.dtype}")

# 渲染 depth
depth = cam.get_depth()  # (128, 128)
print(f"Depth 范围:[{depth.min():.2f}, {depth.max():.2f}] m")

# 保存可视化
plt.imsave("shadow_hand_view.png", rgb[:, :, :3])
plt.imsave("shadow_hand_depth.png", depth, cmap="jet")
```

### 5.7 完整脚本

把上面的代码拼起来, ~150 行, 应该在任何一台有 RTX 卡的机器上 5 分钟内跑通。运行后会得到:

1. **指尖轨迹图**: 中指指尖从起点到目标的空间曲线。
2. **接触事件流**: 每个 step 的接触点数 + 最大力。
3. **RGB 图 + 深度图**: 从固定相机角度看 Shadow Hand + 物体的画面。

这个实验的目的不是训出 SOTA 策略, 而是让你感受:

- **SAPIEN API 的具体形态**(URDF loader, get_joints, get_pose)。
- **PD 控制的行为**(关节 target → step → 检查)。
- **接触求解器的开销**(每个 step O(N²) 接触检查)。

下一步可以接入 BC / RL 算法训一个真实 reach 策略。

---

## 6. 与其他章节的关系

### 6.1 章节内引用(7-仿真器与基准)

- **[[第07章-01-MuJoCo-XML建模与控制]]**: SAPIEN 早期版本底层是 MuJoCo, 后转 PhysX 4。两者接触模型差异是任务数值差异的主因。MuJoCo 的 LCP + 库仑锥 vs PhysX 的 TGS + 库仑锥, 在 30+ 接触点的灵巧操作下, 物理轨迹能差 5-20%。
- **[[第07章-02-Isaac-Sim-工业级GPU仿真]]**: Isaac Lab 用 PhysX 5 GPU pipeline, SAPIEN 是 PhysX 4 CPU。这是灵巧手仿真最大的工程分水岭。SAPIEN 受限于 CPU 串行, 同时跑 32 env 已经接近极限; Isaac Lab 在 A100 上跑 4096 env, 单 episode 耗时差 50 倍。
- **[[第07章-03-Habitat-室内导航仿真]]**: Habitat 解决室内导航, SAPIEN/DexArt 解决灵巧操作, 两者底层都和 UE4 渲染集成。任务思路相似("统一协议"), 但接触动力学要求差异巨大, Habitat 不需要物理模拟, SAPIEN 必须有。
- **[[第07章-04-ManiSkill-操作任务基准]]**: ManiSkill v1/v2 的物理后端是 SAPIEN, v3 改自家 ManiSkill Engine(PhysX 5 GPU)。这条历史线说明: 灵巧操作基准的工程瓶颈从"任务定义不一致"进化到"仿真器吞吐不够"。SAPIEN 在 2023 年达到了上限, ManiSkill3 / Isaac Lab 是回应。
- **[[第07章-05-RoboCasa与行为树]]**: RoboCasa 偏家庭场景 + 行为树生成, SAPIEN/DexArt 偏单任务 + 灵巧手 + Sim2Real。两者不竞争, 是不同切入点的灵巧操作研究。

### 6.2 章节间互引(其他文件夹)

- **`../../../基础/数学基础/第06章-最优化/第06章-05-凸优化理论-强凸·光滑·Lipschitz·收敛率证明.md`**: 接触不变优化 (CIO, Contact-Invariant Optimization) 是灵巧手轨迹规划的标准工具。给定任务的 6-DoF 末端轨迹, 在接触约束下求最优关节角度序列, 这是一个严格凸问题。
- **`数学基础/8-控制论/3-李群与刚体运动`**: Shadow Hand 的指尖位姿 + 力矩都是 SO(3) 上的元素。yaw / pitch / roll 加上 reach, 控制空间是 SE(3) × R^n。
- **`../第03章-机器人运动学与控制/0-总目录与阅读路径.md`**: 接触事件的概率建模: 物体表面哪些点会被指尖触及, 用高斯过程或接触图 (contact graph) 描述。
- **`../第01章-具身智能概论/0-总目录与阅读路径.md`**: 摩擦锥约束是一个 **狄拉克分布 + 凸约束**, 严格数学表达要靠测度论。

### 6.3 共同的工程问题

灵巧手仿真在四个工程问题上是共同的:

1. **大接触数**: 单纯 SAPIEN / MuJoCo 都吃不动, 都要 contact pair pruning。
2. **域随机化**: 物理参数 + 视觉的随机化是 sim2real 的入场券, 没别的办法。
3. **演示数据**: 24 维动作空间 BC 起码要 100-1000 演示, 不够就要行为树 / RL 教师。
4. **真机迁移**: sim2real gap 平均 5-15%, 没办法 "完美", 只能靠 domain randomization + 真实微调缩小。

---

## 7. 与数学基础的关系

### 7.1 接触动力学的数学化

灵巧手操作的核心数学是 **库仑摩擦锥 + LCP 求解**。具体:

对每个接触点 $i$, 设法向反力 $n_i = F_N^i \hat{n}_i$, 切向力 $t_i \in \mathbb{R}^2$, 摩擦约束:

$$\|t_i\|_2 \le \mu n_i \quad (n_i \ge 0)$$

所有接触点的约束形成一个 **凸锥**, 加上非穿透约束, 全局是一个 **LCP (Linear Complementarity Problem)**:

$$M q + N u + b \ge 0, \quad u \ge 0, \quad (M q + N u + b)^\top u = 0$$

PhysX 求解的就是这个 LCP 的最优解(投影到可行域)。SAPIEN 加的各向异性摩擦把上面的球锥换成椭球锥, 求解不变。

### 7.2 库仑锥的凸优化视角

库仑锥是一个二阶锥 (SOC):

$$\mathcal{C} = \{ (n, t) \in \mathbb{R} \times \mathbb{R}^2 \mid \|t\|_2 \le \mu n, n \ge 0 \}$$

凸优化框架下, 接触约束可以这样写:

$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \| q - q_{\text{ref}} \|^2 \\
\text{subject to} \quad & (n_i, t_i) \in \mathcal{C}, \quad \forall i \in \text{contacts} \\
& \text{contact kinematics}
\end{aligned}
$$

这是 Contact-Invariant Optimization (CIO) 的核心。在灵巧手规划里非常有用。

### 7.3 关节物体动力学的几何力学

灵巧手是关节树(tree articulation), 它的配置空间是 $SE(3)^{\text{palm}} \times \prod_{i} S^1$，即手掌的刚体运动 + 每个关节的圆环。这个空间是 **纤维丛**, 底部是末端位姿(6D), 纤维是关节角(冗余自由度)。

这个几何视角给灵巧手控制带来两个有用思路:

1. **空运动算子**(Null-space operator): 给定末端运动 + 关节零空间中的额外任务。
2. **几何控制**: 用 SO(3) 上的李群代数表达指尖力矩, 不用欧拉角。

### 7.4 动力学方程的辛积分

SAPIEN 用半隐式欧拉:

$$\dot{q}_{t+1} = \dot{q}_t + h M^{-1} (\tau - C - G + J^T F)$$
$$q_{t+1} = q_t + h \dot{q}_{t+1}$$

这保留了 Hamilton 系统的辛结构, 长期能量不会漂移。比显式欧拉好, 但比 RK4 简单。MuJoCo 用的是 RK4 (Runge-Kutta), SAPIEN 选择了半隐式, 是 **速度优先于位置** 的哲学。

### 7.5 蒙特卡洛接触估计

触觉策略学习中, 接触事件是稀疏采样。DexArt 用 IL, 但工业上也可以用 **蒙特卡洛树搜索 (MCTS)**:

```text
状态 = (joint angles, object pose, contact points)
动作 = 24-DoF joint delta
奖励 = task-specific
蒙特卡洛: 从每个状态采样 1000 个 rollout, 取最优
```

这是和 RL 的关键区别: MCTS 不依赖值函数近似。

---

## 8. 常见误区

### 8.1 误区一:"SAPIEN 仿真精度高于 MuJoCo, 所以任务应该选 SAPIEN"

**错的**。SAPIEN 在接触动力学上比 MuJoCo 强(更多接触点 + 各向异性摩擦), 但 MuJoCo 在 RL 单臂任务的数值稳定性更好(LCP 解唯一)。**任务性质决定选哪个**, 不是精度优先:

- 灵巧手操作(多接触 + 手指) → SAPIEN
- 单臂抓取(单接触 + 夹爪) → MuJoCo 或 PyBullet 都行
- 关节物体 + 装饰物 → SAPIEN 的 Articulation 表达更清晰

### 8.2 误区二:"24 维动作空间的灵巧手必须用 RL"

**错的**。BC + 好的演示数据在很多任务上能训出 70-80% 的策略。DexArt 论文里的策略全部是 IL(行为克隆 + 残差), 不是 RL。原因:

- RL 在 24 维空间需要 100k-1M 步采样, SAPIEN 跑这么慢不合算。
- 演示数据比纯 RL 收敛快 10 倍。
- 残差 loss 让 BC 更鲁棒, 等价于一种弱 RL。

**但 24 维空间用 RL 配大规模算力**也可以训出 SOTA, 例如 OpenAI 的 Rubik's Cube。这是 **算力 vs 数据** 的 trade-off。

### 8.3 误区三:"触觉传感器越多越好"

**错的**。Shadow Hand 配 BioTac 后传感器是 92 个, 但每个手指增加 18 个 taxel, **数据维数 + 噪声** 同时上升。下游策略网络要更大, 训练更难收敛。DexArt 的实验证明: **用 RGB + 关节角就够**, 触觉只是锦上添花。

工业上的结论: 除非你研究触觉本身, 否则 **RGB + 关节本体感觉** 就够训一个不错的策略。触觉是真的, 不是必需品。

### 8.4 误区四:"sim2real gap 通过加更多传感器的仿真就能解决"

**错的**。sim2real gap 来自物理参数 + 视觉 + 执行器 + 传感器四层叠加, 其中 **物理参数最重要**(接触点偏移 1mm 就让 sim2real 失败)。Sensors 仿真到位顶多解决一部分, 而且代价是仿真变慢。

**正确做法**: **domain randomization**(物理参数 + 视觉 + 执行器延迟全随机化), 让策略见过"漂移"世界, 而不是追求"完美仿真"。

### 8.5 误区五:"SAPIEN 训的策略, 仿真成功率 90% 真机就能用"

**错的**。DexArt 论文里, 仿真 90% 成功率 + Allegro 真机 30% 成功率的情况是真实的, 看起来反直觉。原因是:

- 仿真里物体几何完美对齐。
- 真机物体几何有 ±5mm 误差, 接触点变了。
- 真机控制有 5-20 ms 延迟, 灵巧手任务对延迟敏感。
- 真机执行器有死区, 电流环有偏差。

**正确做法**: sim2real 评估要做, 别只看仿真数字。仿真 90% 真机可能 30-70%, 这是工业常态。

### 8.6 误区六:"LEAP Hand 比 Shadow Hand 简单, 所以更好学"

**错的**。LEAP Hand 16 关节 vs Shadow Hand 24 关节, 看起来学 LEAP 简单, 但实际:

- LEAP Hand 关节少 → 表达姿态少 → 复杂任务完成度低。
- Shadow Hand 关节多 → 控制空间高维 → 但接触面大、稳定。
- 教学角度 LEAP 入门友好, 但要训灵巧操作策略 Shadow Hand 是更好的 baseline。

### 8.7 误区七:"PhysX 的接触力一定是真实的"

**错的**。PhysX 把接触建模成刚体 + 软接触近似, 接触点附近的弹性变形是简化的。这意味着:

- 刚体接触力(法向 + 切向)相对真实。
- 接触面积(接触斑, contact patch) 仿真给的是单点, 真实是面。
- 软接触(橡胶、指尖)仿真给的是穿透深度, 真实是变形量。

工业上, 灵巧手策略仿真能训, 但 **不能直接用仿真接触力作为真机反馈**, 必须真机触觉兜底。

---

## 9. L7/L8 面试 Q&A

### 9.1 Q1: SAPIEN 和 MuJoCo 的核心区别是什么? 灵巧手任务应该选哪个?

**考察点**: 仿真器选型判断, 加分点: 物理引擎层面的接触模型差异。

**回答模板**:

> 三大区别. 第一底层物理: SAPIEN 是 NVIDIA PhysX 4(PGS + TGS, CPU 串行), MuJoCo 是自家引擎(LCP + 自适应, 也 CPU). 第二接触建模: SAPIEN 支持各向异性摩擦(摩擦张量), MuJoCo 仅支持各向同性库仑摩擦. 第三关节物体建模: SAPIEN 用 Articulation(URDF 风格关节树), MuJoCo 用 MJCF(xml/通用). 灵巧手任务选 SAPIEN, 因为 Shadow Hand 的 24 关节 + 多接触点需求超出了 MuJoCo 的舒适区, 而且 BioTac 触觉也要更高级的接触表达. 但 RL 单臂抓取或 locomotion 类任务选 MuJoCo 因为数值稳定性更好, sim2real gap 更小.

**加分点**: 提到 PhysX 4 还在用 CPU, 不像 PhysX 5 GPU 或 MuJoCo XLA 那么快, 这是 SAPIEN 训练大规模 RL 算力瓶颈.

### 9.2 Q2: Shadow Hand / LEAP / Allegro / Inspire 该怎么选?

**考察点**: 灵巧手硬件选型能力.

**回答模板**:

> 看四个维度. **任务难度**: 复杂灵巧操作(装配、开门、转水龙头)选 Shadow(24 关节, 自由度最高); 标准研究选 Allegro 或 LEAP; 简单抓握选 Inspire(轻量 0.55 kg). **预算**: Shadow ~$100k, Allegro ~$25k, Inspire ~$8k, LEAP ~$2k(DIY). **sim2real 要求**: Shadow + Allegro + Inspire 都支持, LEAP 2024 起才进入工业 sim2real. **维护成本**: Shadow 是气动 + 电机, 维护难; LEAP / Allegro / Inspire 全电机, 维护简单. 我在工业项目里默认选 Allegro, 学术研究默认选 LEAP 或 Shadow.

**加分点**: 提到 DexArt 在真机部署用了 Allegro 而不是 Shadow, 因为 Allegro 软件 stack 更成熟.

### 9.3 Q3: DexArt 的 8 个任务为什么能 sim2real?

**考察点**: sim2real 工程理解.

**回答模板**:

> 三个原因. 第一 **几何真实**: 每个物体来自真实扫描, 不是简化的盒子. 第二 **物理参数真实**: 摩擦系数、阻尼按真实测量, 不是随便设的. 第三 **接触容差友好**: 任务的 success threshold 不是 0 cm, 是 2-5 cm 容差. 这给真实控制留了空间, 因为真机不可能毫米级. 加上 Allegro 真实机的硬件延迟小(全电机, 1-5 ms), 总 sim2real gap 平均 5-15%. 这是 DexArt 的核心卖点, 也是它 2023 后成为灵巧手基准的原因.

**加分点**: 提到 DexArt 的两个 loss(BC + 残差)对 sim2real 也很重要, 因为残差 loss 让策略初期更稳定.

### 9.4 Q4: 灵巧手任务为什么要 domain randomization? 怎么量化 sim2real gap?

**考察点**: Sim2Real 工程能力, 加分点: 能说出具体的 randomization 范围.

**回答模板**:

> 灵巧手任务对接触动力学高度敏感, 一个指尖位置的 ±5mm 偏移就可能让策略失效. domain randomization 在训练时随机化物理参数(摩擦 ±30%, 质量 ±20%, 接触点 ±2mm) + 视觉外观(光照、纹理、噪声) + 执行器(延迟 0-20 ms), 让策略见过"漂移"的世界. 量化的方法是在真机跑 N 个 episode, 算 success rate vs 仿真数字, gap = (sim_success - real_success)/sim_success, 一般灵巧手任务 5-15%. DexArt 8 任务平均 gap 约 8%, 这在灵巧手研究里是非常低的数字.

**加分点**: 提到 domain randomization 也有上限, 随机化范围太大反而让策略学不到本质特征, 一般摩擦 ±20%, 接触点 ±1mm 是甜蜜点.

### 9.5 Q5: SAPIEN 训练灵巧手策略需要多少样本? RL 还是 IL?

**考察点**: 算法选型 + 算力评估.

**回答模板**:

> 看任务. 简单抓(单接触)IL 100 演示够, BC 训到 80%. 中等灵巧操作(DexArt buckle / toilet)IL 200-500 演示 + 残差 loss 训到 70%. 困难任务(faucet / door)IL 1000-5000 演示 + RL finetune 100k 步. RL 从零训需要 1-10M 步, 单纯 SAPIEN 跑这么慢不现实, 至少 64 CPU core × 1 周. 工业上 IL + 残差是性价比最高的方案; 学术上 RL + 大算力可以做最难的 in-hand manipulation.

**加分点**: 提到 DiffDex (Diffusion Policy for Dexterous Manipulation, 2023) 用 500 演示训出 80% 成功率的策略, 比 BC 高 25%.

### 9.6 Q6: PhysX 4 vs PhysX 5 区别在哪? SAPIEN 不用 GPU 的本质原因?

**考察点**: 物理引擎版本演进, 加分点: 知道 GPU pipeline 收益.

**回答模板**:

> PhysX 4 是 2017 年发布, CPU 为主, GPU pipeline 仅支持 Apex Destruction 粒子. PhysX 5(2020)是 NVIDIA 重新架构, 大部分求解可走 GPU. SAPIEN 用 PhysX 4 是因为开发时间早(2019), GPU 化要重写接触求解层, 工程代价巨大. 本质上 PhysX 4 的 GPU pipeline 不支持 articulated body(关节树)的接触求解, 只支持单刚体, SAPIEN 又大量用关节物体(Skadow Hand 24 关节), 只能用 CPU. ManiSkill3 转 PhysX 5 GPU 是重新实现了关节物体的 GPU pipeline, 工程量巨大.

**加分点**: 提到 NVIDIA 在 2024 Isaac Lab 也是基于 PhysX 5, 但 SAPIEN 没跟进, 是因为 SAPIEN 团队核心人员去了 NVIDIA, SAPIEN 维护弱化了.

### 9.7 Q7: 接触点可视化怎么用来 debug 灵巧手策略?

**考察点**: 工程调试能力.

**回答模板**:

> 三个套路. 第一是 **看接触时机**: 在 episode 中标出接触点建立的时间戳, 看是否符合预期(如"接近物体 → 接触建立 → 抓稳"三阶段). 第二是 **看接触数量**: 训练时统计平均接触对数 vs 推理时, 如果差很多说明分布漂移(sim2real gap 信号). 第三是 **看接触力分布**: 画接触力的角度分布, 看是否符合库仑锥. SAPIEN 提供 `get_contacts()` 返回每个接触对的 position, normal, force, 这些可视化在灵巧手策略调试里至关重要. 经验上, 接触对数从仿真 50 对掉到真机 30 对说明 sim2real gap 偏大, 需要加强 domain randomization.

**加分点**: 提到 DexArt 论文里有一张接触点热力图, 显示不同策略学到的"接触重点"不同, 这是策略比较的可解释性工具.

### 9.8 Q8: 灵巧手任务的长期发展瓶颈是什么?

**考察点**: 前瞻性判断.

**回答模板**:

> 三个瓶颈. 第一是 **触觉仿真**: BioTac 这类真实触觉传感器在仿真里太简化, 严重限制真机迁移. 第二是 **多指协调**: 现有 IL / RL 都在"模仿" 演示 / 优化单一 reward, 多指协调缺乏先验知识, 训练数据需求大. 第三是 **能耗**: Shadow Hand 4.3 kg + 气动单元总重 30 kg, 装载移动机器人负担重. 短期突破点在 **触觉仿真**(2024 起 NVIDIA 开始把 BioTac 仿真化), 长期突破点在 **多指协调先验**(2025 起 Open X-Embodiment 等大规模数据集可能降低需要).

**加分点**: 提到 Tesla / Figure / Unitree 三家 2024-2025 都在做量产灵巧手, 工业级硬件快了, 算法反而是短板.

---

## 10. 参考文献

### 10.1 SAPIEN 核心论文

1. **Mu et al. (2019)**. *SAPIEN: A SimulAted Part-based Interactive Environment*. CVPR.
   - SAPIEN 第一版, USC 团队, 强 PartNet-Mobility 关节物体建模.

2. **Mu et al. (2021)**. *ManiSkill: Generalizable Manipulation Skill Benchmark via Large-Scale Manipulation*. CoRL.
   - ManiSkill 第一版, 基于 SAPIEN, 4 个 RGB 模仿学习任务.

3. **Gu et al. (2023)**. *ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills*. ICLR.
   - ManiSkill v2, 20+ 任务, 任务框架重构.

### 10.2 DexArt 核心论文

4. **Bao et al. (2023)**. *DexArt: Learning Generalizable Dexterous Manipulation from Diverse Simulated Objects*. RSS.
   - DexArt 第一版, 8 个任务, UC Berkeley.

5. **Wang et al. (2024)**. *DexArt 2.0: Learning Dexterous Manipulation from Single Pixel Visualization*. arXiv.
   - 进阶版, 视觉输入更接近真机.

### 10.3 灵巧手硬件

6. **Shadow Robot Company**. *Shadow Hand Technical Specification*. 2020.
   - Shadow Hand 24-DoF 工业规格.

7. **Shaw et al. (2023)**. *LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning*. RSS.
   - LEAP Hand 开源设计, Stanford + Meta AI.

8. **Wonik Robotics**. *Allegro Hand User Manual v3*. 2018.
   - Allegro Hand 商业规格.

9. **Inspire Robotics (2023)**. *Inspire Hand RH56 Technical Document*.
   - 中国 5 指灵巧手, 0.55 kg 业内最轻.

### 10.4 仿真器与物理引擎

10. **NVIDIA PhysX 4 SDK Documentation**. NVIDIA Developer, 2018.
    - SAPIEN 底层物理引擎.

11. **Tedrake (2023)**. *Drake: Model-Based Design for Robotic Systems*. MIT.
    - 对标 SAPIEN 的另一条 Python 路线, 但走刚体路线.

12. **Howard et al. (2022)**. *MuJoCo XLA: GPU-Accelerated RL*. DeepMind.
    - MuJoCo XLA 模式, 2023 起进入 GPU 仿真竞争.

13. **NVIDIA (2024)**. *Isaac Lab PhysX 5 GPU Pipeline*. NVIDIA Technical Report.
    - ManiSkill3 / Isaac Lab 的底层物理.

### 10.5 接触力学与控制论

14. **Caron et al. (2015)**. *Stability of Surface Contacts for Robotic Manipulation*. RSS.
    - 接触点稳定性的几何分析, 灵巧手规划理论.

15. **Mason (2001)**. *Mechanics of Robotic Manipulation*. MIT Press.
    - 经典教材, 接触力学与库仑锥.

16. **Todorov (2014)**. *A General, Time-Optimal Solution to the Contact-Transition Problem*. RSS.
    - 接触不变优化的理论基础.

17. **Murray, Li, Sastry (1994)**. *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
    - 接触力学的李群视角.

### 10.6 RL / IL 基线

18. **Schulman et al. (2017)**. *PPO*. arXiv.
    - DexArt 论文里 finetune 用的算法.

19. **Chi et al. (2023)**. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS.
    - 灵巧手策略的 SOTA baseline, DexArt 8 任务涨 15%.

20. **Mandlekar et al. (2021)**. *Robomimic: What Matters in Learning from Demonstrations for Robot Manipulation*. ICLR.
    - IL 算法的标准化 benchmark, 也覆盖 DexArt 任务.

### 10.7 Sim2Real 与域随机化

21. **Tobin et al. (2017)**. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. IROS.
    - 域随机化开山论文, 灵巧手 Sim2Real 标配.

22. **OpenAI et al. (2019)**. *Solving Rubik's Cube with a Robot Hand*. OpenAI Blog.
    - Dactyl 灵巧手 + 域随机化训 Rubik's Cube, 灵巧手训推的天花板.

23. **Andrychowicz et al. (2020)**. *Learning Dexterous In-Hand Manipulation*. IJRR.
    - OpenAI 灵巧手 in-hand manipulation 论文, 5 指 + 视觉.

### 10.8 资源与工具

24. **SAPIEN GitHub**. https://github.com/haosulab/SAPIEN
    - SAPIEN v3 开源仓库.

25. **DexArt GitHub**. https://github.com/hyperplane-lab/DexArt
    - DexArt 任务代码, 2023 起维护.

26. **PartNet-Mobility Dataset**. Stanford, 2018.
    - SAPIEN 用的关节物体数据集, 几何来自真实扫描.

27. **YCB Object and Model Set**. Berkeley, 2015.
    - 通用物体数据集, DexArt 部分任务从中取物.

28. **LEAP Hand GitHub**. https://github.com/leap-hand/LEAP_Hand
    - LEAP Hand 硬件 + URDF + 真机控制代码.

### 10.9 综述与发展

29. **Billard & Kragic (2019)**. *Trends and Challenges in Robot Manipulation*. Science.
    - 灵巧手仿真的综述, 涵盖 SAPIEN, MuJoCo, Isaac.

30. **Khatib et al. (2024)**. *Dexterous Manipulation: A Decade of Progress*. ICRA.
    - 灵巧操作 10 年回顾, SAPIEN/DexArt 是其中重要里程碑.

---

**章节结束**

> **学完这一章你应该能动手做的事**:
>
> 1. 在自己机器上装 SAPIEN, 加载 Shadow Hand URDF, 跑一个 reach episode。
> 2. 复现 DexArt 的 buckle 任务, 训一个 IL 策略到 50%+。
> 3. 用 `get_contacts()` 可视化灵巧手接触点, 对比仿真 vs 真机轨迹的接触对数。
> 4. 在面试里说出 SAPIEN vs MuJoCo 的核心差异, 并解释为什么灵巧手选 SAPIEN。
> 5. 评估一个 DexArt 任务的 sim2real gap, 给出量化的 domain randomization 方案。
>
> **下一章**: 第 7 章 Robocasa 仿真——家庭操作基准, 用行为树 + 大规模场景生成, 把 sim2real 推到家庭场景。

---

## 11. Articulation Tree 深度剖析: 灵巧手到底是棵什么树?

### 11.1 为什么 URDF 不够?

URDF (Unified Robot Description Format) 把机器人描述成 **link + joint 的树状结构**。Shadow Hand 的 URDF 有 25 个 link(手掌 + 5 指 × 4 段 + 4 个指间连接) + 24 个 revolute joint。每个 joint 是 1-DoF 旋转, 没有 roll/pitch 这种解耦设计。

MuJoCo 用 MJCF, Drake 用 SDFormat, SAPIEN 自己写了一个 `ArticulationBuilder`。这三家对"关节树"的理解有微妙差异:

```text
引擎        树表达              驱动模型             接触求解
────────────────────────────────────────────────────────────────────
MuJoCo      MJCF (xml)         actuator + gain      LCP 全局解
Drake       SDFormat (yaml)    PD + torque          半隐式 + 软约束
SAPIEN      Articulation       drive (pos/vel/torque)  PGS / TGS 子迭代
Isaac Lab   USD + PhysX 5     actuator + gain      GPU TGS
PyBullet    URDF               torque / position    LCP / NNC
```

工业上要混用 URDF, 经常在转换时丢驱动参数。SAPIEN 的 ArticulationBuilder 接受标准 URDF, 但会用默认 drive 替换 `<transmission>` 标签。

### 11.2 Shadow Hand 的关节树 ASCII 图

```text
                    palm_root (固定, base_link)
                         │
                  ┌──────┼──────┐
                  │      │      │
                thumb  index  middle   ring   pinky
                 │      │      │       │      │
              ┌──┤      J1     J1      J1     J1       (knuckle: 1-DoF, 内收/外展)
              │  J1      │      │       │      │
              │  │     ┌──┴──┐ ┌┴───┐ ┌─┴───┐ ┌┴────┐
              J2 │     J2    J2 J2   J2 J2    J2 J2    J2  (proximal: 1-DoF 弯曲)
              │  │     │     │ │    │ │     │ │     │
              J3 │     J3    J3 J3   J3 J3    J3 J3    J3  (middle: 1-DoF 弯曲)
              │  │     │     │ │    │ │     │ │     │
              J4 │     J4    J4 ─    ─ ─    ─ ─    ─     (distal: 1-DoF 弯曲, 末段)
              │  │
              J5 │                                              (thumb 中段)
              │  │
              J6                                                  (thumb 远段)

Legend: J1=knuckle(J0=abduction/adduction), J2-J4=finger joints, J5-J6=thumb extra
─────────────────────────────────────────────────────────────────────────────
总关节数:  thumb 5 + 其他 4 指 × 4 = 21 + 掌指关节(knuckle)覆盖 4 指共 4 + 手腕 2 = 24-DoF
```

**关键观察**:

- 拇指自由度是 5(J1 ab/ad + J2-J4 bend + J5 CMC), 比其他指多 1。这是人手生物学的体现: 拇指承担对掌(opposition)功能。
- 食指到小指的 knuckle 关节(J1)是 abduction/adduction(内收/外展), 不是弯曲。
- 末端关节(末段, distal)是 **被动的**: 在 URDF 里 J4 是 joint, 但实际上由 tendon 联动, SAPIEN 里当成独立 drive 也能训, 但运动学耦合。

### 11.3 SAPIEN ArticulationBuilder 怎么用?

`ArticulationBuilder` 是 SAPIEN 装一台机器人的工厂函数。用法:

```python
import sapien.core as sapien

# ponytail: scene 是全局容器, 一个进程一个
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene(gpu=True)  # gpu=True 走 PhysX GPU 路径(若 PhysX 4 支持)

# ponytail: URDF 路径直接给, 内部走 ArticulationBuilder
loader = scene.create_urdf_loader()
loader.fix_root_link = True  # palm 固定到世界
loader.scale = 1.0

# ponytail: 默认 drive 类型是 position + 默认 stiffness / damping, 灵巧手要改
loader.load_multiple_collisions_from_file = True

robot = loader.load("shadow_hand.urdf")
print(f"Loaded {robot.name}, links={robot.get_links().__len__()}, joints={robot.get_joints().__len__()}")

# ponytail: 关节树遍历是 qpos / qvel 的语义前提
for joint in robot.get_joints():
    print(f"  {joint.name}  type={joint.type}  drive={joint.get_drive_type()}")
```

**关键点**: SAPIEN 的 drive 有 5 种类型: `position`, `velocity`, `force`, `acceleration`, `none`。DexArt 默认是 `force` drive(给关节 torque), 这样 PD 控制器在 policy 网络外做。如果用 `position` drive, 就把 PD 烧死在 SAPIEN 里, 灵活度差。

### 11.4 LEAP Hand 的简化树

LEAP Hand 是 Stanford + Meta 2023 的开源设计, 16-DoF, 4 指:

```text
LEAP Hand (16-DoF)
──────────────────
palm_root (固定)
   │
   ├── index (3 DoF)   ── knuckle + proximal + distal
   ├── middle (3 DoF)  ── 同上
   ├── ring (3 DoF)    ── 同上
   └── thumb (4 DoF)   ── CMC + knuckle + proximal + distal

Joint 总数 = 3+3+3+4 = 13? 不, 还有 3 个隐式耦合 = 16 DoF
─────────────────────────────────────────────────────
```

LEAP 的设计哲学是 **少而精**: 16 DoF 够覆盖大部分灵巧任务, 但成本只有 Shadow 的 1/50。研究项目默认 LEAP。

### 11.5 Articulation 的零空间 (Null-space) 几何

冗余机械臂(7+ DoF)的经典概念是 **零空间 (Null-space)**: 在不改变末端位姿的前提下, 关节角可以在某个子空间内自由运动。灵巧手有同样的概念:

给定末端 6-DoF 速度 $\dot{x}_{\text{end}} \in \mathbb{R}^6$, 关节速度 $\dot{q} \in \mathbb{R}^{24}$ 满足雅可比:

$$\dot{x}_{\text{end}} = J(q) \dot{q}$$

零空间投影算子:

$$N(q) = I - J^\# J = I - J^\top (J J^\top)^{-1} J$$

$N(q)$ 是把任意关节速度投影到 "不引起末端运动" 的子空间。冗余自由度 $\dim N = 24 - 6 = 18$, 这 18 个 DoF 用来做 **自避碰**、**关节极限回避**、**力矩最小化** 等二级目标。

工程上, 灵巧手策略网络学到的隐式策略可以解释为: **主任务由末端 6-DoF 完成, 二级任务(避碰、关节限位)由零空间完成**。

### 11.6 关节动力学方程的工程简化

灵巧手是 **树状关节**, 它的动力学可以用 Featherstone 算法高效求解。SAPIEN 内部用的是简化版 Articulated Body Algorithm (ABA):

$$M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = \tau + J^\top F_{\text{contact}}$$

其中:

- $M(q)$: 24×24 质量矩阵(稀疏, 带宽 ≈ 树深)
- $C(q, \dot{q})$: 科氏力矩阵
- $G(q)$: 重力向量
- $J^\top F_{\text{contact}}$: 接触反力的关节投影

灵巧手与单臂的核心区别是 $J^\top F_{\text{contact}}$ 这一项: 单臂可能 1-3 个接触点, 灵巧手可达 20-50 个接触点。PhysX PGS 求解器要同时解出所有 $F_{\text{contact}}$, 这就是 GPU 化动机。

---

## 12. 进阶 PyTorch 实战代码

### 12.1 代码 1: SAPIEN Articulation 加载 + 灵巧手 reach 任务

```python
"""sapien_reach.py
加载 Shadow Hand, 跑一个 5-step reach episode, 可视化接触点。
依赖: sapien (pip install sapien), torch
"""

import sapien.core as sapien
import numpy as np
import torch
import time

# ponytail: scene 配置 GPU 路径, PhysX 4 实际仍走 CPU, 但 gpu=True 启用 PhysX GPU 子模块
engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=False)
engine.set_renderer(renderer)
scene = engine.create_scene(gpu=False)  # gpu=False 更稳, PhysX 4 GPU pipeline 不完整
scene.set_timestep(1 / 240)  # 灵巧手任务 dt 必须 ≤ 1/240

# ponytail: 桌面 + 目标物体, 简化的 reach setup
scene.add_ground(altitude=0)
table = scene.create_actor_builder().add_box_visual(
    half_size=[0.5, 0.5, 0.02]
).add_box_collision(
    half_size=[0.5, 0.5, 0.02]
).build_static(name="table")
table.set_pose(sapien.Pose([0, 0, 0.5]))

target = scene.create_actor_builder().add_sphere_visual(
    radius=0.02
).add_sphere_collision(
    radius=0.02
).build_dynamic(name="target")
target.set_pose(sapien.Pose([0.3, 0, 0.55]))

# 加载 Shadow Hand (URDF 路径按本地调整)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
loader.scale = 1.0
hand = loader.load("shadow_hand.urdf")
hand.set_pose(sapien.Pose([0, 0, 0.55]))

# ponytail: 把所有关节设成 torque drive, 让 PD 在外层做
for joint in hand.get_joints():
    joint.set_drive_properties(stiffness=0, damping=0)

# 简化 reach: 用指尖位置作为 reward 信号
tip_link_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

for step in range(5):
    # 仿真 step
    scene.step()
    scene.update_render()

    # 获取指尖位置
    tip_positions = []
    for name in tip_link_names:
        link = hand.get_link(name)
        pose = link.get_pose()
        tip_positions.append(pose.p)

    # 简单策略: 让指尖靠近目标球(用一个固定方向的关节速度)
    target_p = target.get_pose().p
    palm_p = hand.get_link("palm").get_pose().p
    direction = target_p - palm_p
    direction = direction / (np.linalg.norm(direction) + 1e-6)

    # 关节速度命令(简化: 5 指都向目标方向微动)
    qvel_cmd = np.random.randn(hand.get_dof()) * 0.05  # 占位
    hand.set_qvel(qvel_cmd)

    # ponytail: 接触点诊断, debug 用
    contacts = scene.get_contacts()
    n_contacts = sum(len(c) for c in contacts)
    print(f"step={step}  tips_mean={np.mean(tip_positions, axis=0)}  contacts={n_contacts}")

print("Reached", np.linalg.norm(tip_positions[1] - target_p))
```

**代码要点**:

1. `set_timestep(1/240)`: 灵巧手任务的核心参数, 改了仿真行为完全不一样。
2. `set_drive_properties(stiffness=0, damping=0)`: 关掉 SAPIEN 内置 PD, 让外层 policy 控制。
3. `scene.get_contacts()`: 每个 step 返回所有接触对的列表, 用于 sim2real 对比。
4. 这个脚本只是 reach demo, 真实训练时换成 PPO / BC policy 网络。

### 12.2 代码 2: 接触点 + 关节角 联合观测的 BC 策略

```python
"""dexart_bc.py
DexArt buckle 任务的简化 BC 训练: 关节角 + 接触点向量 = 策略输入。
依赖: torch, numpy, sapien (仅用于生成 demo, 这里用合成数据)
"""

import torch
import torch.nn as nn
import numpy as np

# ponytail: 输入维度 = 24(关节角) + 5(指尖位置)× 3 + 30(接触点 Top-10) × 3 = 87
# 真实 DexArt 用 point cloud, 这里简化成扁平向量
INPUT_DIM = 24 + 15 + 30  # qpos + fingertip_pos + contact_points(最多10)
ACTION_DIM = 24  # 24-DoF joint delta

class DexArtPolicy(nn.Module):
    """简化版 BC 策略: MLP, 4 层, 256 单元"""

    def __init__(self, obs_dim=INPUT_DIM, act_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),  # 输出 ±1, 实际执行时乘 0.05 作为关节速度增量
        )

    def forward(self, obs):
        return self.net(obs)


def extract_observation(scene, hand, target_actor):
    """从 SAPIEN scene 抽观测向量"""

    # 24 维关节角
    qpos = hand.get_qpos()  # numpy array, 长度 = dof

    # 5 指尖位置(各 3 维)
    fingertip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    fingertip_pos = np.concatenate([
        hand.get_link(n).get_pose().p for n in fingertip_names
    ])

    # 接触点(最多 10 个, 不够补 0)
    contacts = scene.get_contacts()
    flat_contacts = []
    for cs in contacts:
        for c in cs:
            flat_contacts.append(c.point)  # 3D point
            if len(flat_contacts) >= 10:
                break
        if len(flat_contacts) >= 10:
            break
    while len(flat_contacts) < 10:
        flat_contacts.append([0.0, 0.0, 0.0])
    contact_pts = np.array(flat_contacts[:10]).flatten()

    return np.concatenate([qpos, fingertip_pos, contact_pts])


def train_bc(demos, n_epochs=50, batch_size=256, lr=3e-4):
    """demos: list of (obs, action) pairs, 来自人类示教或 privileged policy"""

    obs_list = np.stack([d[0] for d in demos]).astype(np.float32)
    act_list = np.stack([d[1] for d in demos]).astype(np.float32)

    policy = DexArtPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    obs_t = torch.from_numpy(obs_list)
    act_t = torch.from_numpy(act_list)

    for epoch in range(n_epochs):
        idx = np.random.permutation(len(obs_t))
        total_loss = 0.0
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            obs_batch = obs_t[batch_idx]
            act_batch = act_t[batch_idx]

            pred = policy(obs_batch)
            loss = ((pred - act_batch) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"epoch={epoch}  loss={total_loss / (len(idx) / batch_size):.4f}")

    return policy


# ponytail: 用合成数据演示训练流程, 真实 DexArt demo 需要 200-500 个示教
def make_synthetic_demos(n=200):
    demos = []
    for _ in range(n):
        obs = np.random.randn(INPUT_DIM) * 0.1
        action = obs[:ACTION_DIM] * 0.5 + np.random.randn(ACTION_DIM) * 0.05
        demos.append((obs, action))
    return demos


if __name__ == "__main__":
    demos = make_synthetic_demos(500)
    policy = train_bc(demos, n_epochs=100)
    print("BC 训练完成, 参数量:", sum(p.numel() for p in policy.parameters()))
```

**代码要点**:

1. **观测设计**: `qpos + fingertip_pos + contact_points` 是 DexArt 实际使用的简化版, 真实论文用 point cloud(1024 点 × 3 维)。
2. **MLP + Tanh**: 4 层 256 单元是 2023 年的标准, 2024 起 Diffusion Policy 替换了它, 涨 15%。
3. **损失函数**: 纯 MSE, DexArt 论文加了一个残差 loss(辅助预测 next state), 在 contact 稀疏时有用。
4. **合成数据 demo**: 真实场景下 demo 来自 VR 示教(Human-in-the-loop), 或 privileged policy(用 ground truth object pose 训一个 teacher, 然后 student 学习)。

### 12.3 调试 SAPIEN 的三个黄金诊断

装好 SAPIEN 后, 99% 的"奇怪 bug"都来自三个坑: timestep (灵巧手必须 1/240, 否则接触瞬间消失); drive 参数 (IL 用 stiffness=200/damping=20, RL 改为 0); contact friction (库仑标准 0.5, 错设导致物体乱飞). **经验法则**: 改这三个参数之前别动策略网络, 90% 策略不收敛是仿真参数问题, 不是网络问题. 下面是最常见的诊断模板:

```python
scene.set_timestep(1 / 240)  # 灵巧手必须这个值
for joint in hand.get_joints():
    joint.set_drive_properties(stiffness=200, damping=20)
for link in hand.get_links():
    for cs in link.get_collision_shapes():
        cs.set_friction(0.5)
```

---

## 13. 工业案例深度剖析

### 13.1 案例一: SAPIEN vs Isaac Lab 选型——某机器人创业公司的复盘

2024 年, 某国内人形机器人公司(代号 R)做灵巧手操作研究, 起初全用 SAPIEN, 跑了 3 个月后切到 Isaac Lab。复盘:

```text
时间线         阶段                  仿真器        算力           关键发现
─────────────────────────────────────────────────────────────────────
第 1 个月      PoC + 算法验证         SAPIEN       64 CPU         buckle 任务 60% 成功率
第 2 个月      大规模 RL             SAPIEN       256 CPU        4096 并行 OOM, 退回 1024
第 3 个月      接触精度提升           Isaac Lab    8 × A100       PhysX 5 GPU 跑通, 2048 env
第 4 个月      Sim2Real              Isaac Lab    8 × A100       真机 55% 成功率
─────────────────────────────────────────────────────────────────────
复盘:  SAPIEN 适合算法快速迭代 + 小规模 RL, Isaac Lab 适合大规模 GPU RL + 真机迁移
```

**关键差异**: SAPIEN v3 用 PhysX 4 CPU (4096 env 约 12k step/s), Isaac Lab (PhysX 5 GPU) 约 50k step/s, 4 倍加速. 关节物体 GPU pipeline Isaac Lab 支持, SAPIEN 不支持. SAPIEN 社区 2024 起下滑, Isaac Lab NVIDIA 主推. 学术研究 + 小规模实验选 SAPIEN (API 简单 + URDF 友好), 工业级大规模训练切 Isaac Lab.

### 13.2 案例二: DexArt buckle 任务详解—— 一个最简灵巧操作

DexArt 8 个任务里, **buckle**(扣安全带)是最简单的, 也是 sim2real gap 最小的一个。原因: 任务几何固定(安全带扣), 容差大(2 cm 算成功), 接触事件少(3-5 个接触点)。

```text
任务流程 (buckle episode, 200 steps):
─────────────────────────────────────────────
t=0          安全带扣在桌面, 灵巧手悬空(20 cm 上方)
t=0-50       reach: 灵巧手移向扣具, 5 指张开
t=50-100     grasp: 拇指 + 食指夹住扣具一段, 中指扶稳
t=100-150    insert: 把一段插入另一段(对齐 + 推)
t=150-180    lock: 扣具卡入, 接触反力峰值
t=180-200    release: 张开手指, 检查插入是否成功
─────────────────────────────────────────────
总接触事件: 3-5 次接触建立 / 1 次卡入 peak
容差: 末端位置 ±2 cm 算成功
奖励: 0 / 1 二值
```

**buckle 的策略关键点**:

1. **reach 阶段**: 视觉 policy 够用, RGB → joint delta 直接 BC。
2. **grasp 阶段**: 接触点必须包含拇指 + 食指, 中指扶稳是技巧。
3. **insert 阶段**: 最难, 需要 **力反馈**(指尖接触力方向)。DexArt 用接触点位置代替力, 是简化版。
4. **lock 阶段**: 接触反力会突变, policy 必须对峰值鲁棒。

**buckle 论文数字**:
- 仿真成功率: 78% (BC + 残差 loss)
- Allegro 真机: 71% (2024 年新数字)
- Gap: 7%, 是 DexArt 8 任务中最低的

**为什么 sim2real gap 小?** 因为 buckle 几何容差 + 接触动力学简单, 物理参数误差影响有限。

### 13.3 案例三: Allegro Hand 真机部署——DexArt 团队的经验

DexArt 真机用 Allegro 不是 Shadow Hand, 选型对比:

```text
指标            Shadow Hand        Allegro Hand       LEAP Hand
──────────────────────────────────────────────────────────────
DoF             24                 16                 16
重量            4.3 kg             1.1 kg             0.5 kg
成本            ~$100k             ~$25k              ~$2k (DIY)
驱动器          气压 + 电机        全电机             全电机
软件栈          ROS + 自家 SDK     ROS + C++          ROS + Python
维护难度        高(气密)          中                  低
──────────────────────────────────────────────────────────────
选 Allegro:  工业成熟 + 软件栈完整 + 维护成本可控
```

**Allegro 部署细节**: ROS topic 100 Hz, 控制环 1 kHz, joint_trajectory_controller 桥接延迟 5-10 ms. 关节峰值扭矩 0.5 Nm / 持续 0.2 Nm, 仿真给超过真机的命令会被夹断, 策略必须学"温柔"动作. 指尖力传感器噪声 0.05 N, sim2real 要降权. 每批 Allegro 重新标定零位, 否则初始姿态偏差让 BC 策略跑偏. DexArt 2024 数字: Allegro 真机 bucket 68%, door 54%, 平均 62%, 低于仿真 78%, 但已是灵巧手 sim2real SOTA.

### 13.4 案例四: 灵巧手仿真成本对照

跑一次完整 DexArt 训练的资源消耗:

```text
任务                   演示数   GPU 小时    美元成本(AWS)
─────────────────────────────────────────────────────
buckle (BC)            200      8           ~$30
toilet (BC)            500      24          ~$90
faucet (BC + RL ft)    1000     96          ~$360
door (BC + RL ft)      2000     192         ~$720
RL finetune 100k 步    -        48          ~$180
真机部署(含硬件)      -        -           ~$5000
─────────────────────────────────────────────────────
全 8 任务复现           ~5000    ~500        ~$2500
```

**关键观察**: buckle 最便宜 ($30 + 8 GPU 小时), faucet 最贵 (接触事件复杂 + RL finetune). 真机部署占总成本 60-70%, 不是算力, 是硬件 + 维护. 评估灵巧手项目总成本别只看 GPU 钱, 还要算 URDF 维护 + 真机标定 + 域随机化调参的人力.

---

## 14. 性能优化与并行化

### 14.1 SAPIEN 的并行环境瓶颈

SAPIEN 跑 RL 时, 一个进程开多个 scene 是最直接的方式:

```python
# ponytail: 这种写法简单, 但 IPC 开销大, 256+ env 会卡
scenes = [engine.create_scene() for _ in range(256)]
for step in range(total_steps):
    for s in scenes:
        s.step()
```

更高效的做法是用 `SubprocEngine`(子进程), 把每个 scene 放到独立进程:

```python
# ponytail: 子进程并行, 显存不爆, 但通信开销
from sapien.utils import SubprocEngine
sub_engine = SubprocEngine(256)  # 256 个子进程
```

**性能对比** (4096 env, 1M step): 单进程 1 scene + batch 约 200 step/s (4 GB, 太慢); 单进程 256 scene 1500 step/s (64 GB, CPU bound); SubprocEngine 256 4000 step/s (80 GB, 通信有开销); ManiSkill3 (PhysX 5 GPU) 50k step/s (32 GB, 工业级); Isaac Lab (PhysX 5 GPU) 60k step/s (48 GB, NVIDIA 主推). GPU pipeline 在 env ≥ 4096 时才显著赢, 小规模反而被 kernel launch overhead 拖累.

### 14.2 GPU 内存的真正瓶颈

灵巧手仿真的 GPU 显存: 接触对存储 (每 env ~50 MB × 4096 = 200 GB, 瓶颈) + 碰撞体 BVH (10 MB × 4096 = 40 GB) + 渲染纹理 offscreen (30 MB × 4096 = 120 GB). 总和 ~360 GB, A100 80GB × 5 才能放下. 优化: 关闭渲染省 30% 显存; 降接触对 (PhysX 4 `max_contacts_per_pair` 默认 4 → 1) 省 50%; contact history 用 ring buffer 限 100 step 不全存.

### 14.3 接触点降采样的算法技巧

24-DoF 灵巧手单 step 接触点 50+, 策略网络只需 10-20 个最有信息量的. 用 PointNet 风格的 farthest point sampling (FPS):

```python
# ponytail: FPS 选 10 个代表接触点, 比随机采样覆盖更全
def farthest_point_sample(points, n_samples=10):
    N = points.shape[0]
    if N <= n_samples:
        return np.arange(N)
    sampled = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    for _ in range(n_samples - 1):
        last = sampled[-1]
        d = np.linalg.norm(points - points[last], axis=1)
        distances = np.minimum(distances, d)
        sampled.append(np.argmax(distances))
    return np.array(sampled)
```

为什么 FPS 而不是随机? FPS 保证选中的点空间上分散, 覆盖接触区域, 随机采样可能漏掉关键接触点 (指尖 + 物体边缘).

### 14.4 训练循环的 profiler 实战

`torch.profiler` 是调 RL 训练循环的必备工具:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=5, warmup=2, active=10),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/sapien_bc"),
) as prof:
    for epoch in range(n_epochs):
        # train one epoch
        prof.step()

# ponytail: profile 结果通常显示 PhysX step 占 70%, 网络 forward 15%, 数据传输 15%
# 优化顺序: 先 PhysX(降 env 数 / 简化接触), 再数据传输, 最后网络
```

**灵巧手训练常见 profile 信号**:

- PhysX step > 80% → 减少 env 数, 或者降 dt
- 网络 forward > 30% → 网络太大, 简化
- 数据传输 > 20% → GPU 没用到, 检查 device placement

---

## 15. L7/L8 面试 Q&A 第二轮

### 15.1 Q9: SAPIEN Articulation 和 Link 有什么区别? 为什么要分这两层?

**考察点**: SAPIEN 内部架构理解.

**回答模板**:

> Link 是刚体, 单 Actor 的物理单元; Articulation 是关节树的容器, 包含多个 Link + Joint + Drive. 灵巧手必须用 Articulation, 因为它把整只手作为一个整体配置(qpos / qvel 一致), 关节树上的力矩传递通过 Featherstone 算法自动计算. 如果用多个独立 Actor 模拟灵巧手, 每个手指是独立 Actor, 关节耦合就要在 policy 外手动算, 工程上不现实. SAPIEN 把 Articulation 设计成单独的类, 是为了让 Policy 直接调 `set_qvel(24维向量)` 而不用关心底层 link 拓扑.

**加分点**: 提到 PhysX 内部 `PxArticulationReducedCoordinate` 就是 Featherstone 的 GPU 实现.

### 15.2 Q10: 灵巧手仿真里, 为什么 GPU pipeline 比 CPU pipeline 慢的也有?

**考察点**: GPU 化的代价认知.

**回答模板**:

> GPU pipeline 在 **大规模并行** 时才赢. 如果只跑 64-256 env, GPU 的 kernel launch overhead + 数据传输(H2D / D2H)反而比 CPU 慢 1.5-2 倍. 灵巧手任务的 GPU 优势要 env ≥ 1024 才显现, env ≥ 4096 才显著. 此外 PhysX 4 的 GPU pipeline 不支持 articulated body, SAPIEN 跑关节物体只能 CPU; ManiSkill3 切到 PhysX 5 才解决. 所以早期 SAPIEN 的 GPU 模式其实是个 marketing 名字, 实际 CPU 求解.

**加分点**: 提到 contact pair 的 GPU reduction 在小规模时是负优化.

### 15.3 Q11: 灵巧手策略的 sim2real gap 主要来自哪? 怎么排序?

**考察点**: sim2real 工程理解.

**回答模板**:

> 四个来源按贡献排序: 1. **物理参数**(摩擦 / 阻尼 / 质量), 贡献 40-50%. 2. **几何误差**(URDF 关节偏移 / 物体扫描误差), 贡献 25-35%. 3. **执行器**(延迟 / 死区 / 电流环偏差), 贡献 15-25%. 4. **视觉**(光照 / 纹理 / 镜头畸变), 贡献 10-20%. 灵巧手任务前三项是关键, 视觉相对次要. 所以 domain randomization 的预算应该按这个比例分配, 物理参数 ±30%, 几何 ±5mm, 执行器延迟 ±10ms.

**加分点**: 提到 contact patch 误差(单点 vs 面)也是隐藏的几何误差源, 仿真里永远是单点.

### 15.4 Q12: DiffDex / Diffusion Policy 在灵巧手任务上为什么比 BC 高 15-25%?

**考察点**: 现代 IL 算法的工业落地.

**回答模板**:

> 三点原因. 第一 **多模态动作分布**: buckle 任务里 "先插左边还是右边" 是多峰的, BC 用 MSE 平均掉. Diffusion Policy 用 DDPM 显式建模多模态. 第二 **长 horizon 一致性**: 200 步任务里, BC 容易在第 50 步漂移, Diffusion Policy 通过去噪过程保持轨迹一致. 第三 **观测噪声鲁棒**: Diffusion Policy 的去噪过程天然抗噪声, 仿真 vs 真机的观测差异会被去噪吸收. 代价是推理慢 5-10×, Diffusion step = 100. 工业部署用 DDIM(10 step) + 模型蒸馏.

**加分点**: 提到 DexArt 论文里 Diffusion Policy 在 bucket / door 涨 18-22%, 但 buckle 只涨 5%, 因为 buckle 任务本身就单峰.

### 15.5 Q13: SAPIEN 里的接触点 (contact point) 和接触对 (contact pair) 有什么区别?

**考察点**: PhysX API 细节.

**回答模板**:

> 接触对是两个 collision shape 的配对(actor A 的 shape i + actor B 的 shape j), 是求解的最小单元. 一个接触对可能产生多个 contact point(一个 shape 上的多个凸包交点). SAPIEN `get_contacts()` 返回每个 contact 的 position/normal/force, 但内部用 contact pair 组织. 灵巧手策略关心的是 contact point, 因为策略网络输入是 30 维的 (x,y,z) 坐标; 物理求解关心的是 contact pair, 因为 PhysX 把 pair 作为约束单元. 这两者差 1-2 个数量级, 一个 Shadow Hand 抓握可能有 20 个 contact pair 但 60 个 contact point.

**加分点**: 提到 PhysX `PxContactStream` 是接触对的连续内存表示, GPU pipeline 里靠这个并行.

### 15.6 Q14: 灵巧手训练如何选 batch size 和 learning rate?

**考察点**: 训练超参工程.

**回答模板**:

> BC: batch_size=256-1024, lr=3e-4 (Adam), 200-500 epoch. RL (PPO): batch_size= rollouts=2048 env × 8 step, lr=3e-4 actor + 1e-3 critic, clip_ratio=0.2. 灵巧手任务比 locomotion 难, batch 偏小(256-512)反而更稳, 因为高维策略梯度方差大. Diffusion Policy: batch_size=128, lr=1e-4 (AdamW), 100 epoch. 关键不是 batch 大, 是 **梯度裁剪**(grad clip norm=1.0 必备) + **EMA**(0.99-0.999). 不做这两个, 灵巧手训练一定崩.

**加分点**: 提到 DexArt 论文里 lr 是 1e-3, 比标准 BC 高 3 倍, 因为 IL demo 噪声大, 大 lr 反而跳过 local minima.

---

## 16. 附录 A: SAPIEN API 速查表

### 16.1 核心类层次

Engine (单例) → Renderer (SapienRenderer / OptiXRenderer) + Scene (容器) → Actor (Static / Dynamic) + Articulation (关节树) + Light + Camera + Contact (跨 scene 接触对).

### 16.2 常用 API

| API | 用途 | 备注 |
| :--- | :--- | :--- |
| `engine.create_scene(gpu=True)` | 创建场景 | gpu=True 启用 PhysX GPU (PhysX 5 完整) |
| `scene.create_urdf_loader()` | 加载 URDF | 支持 fixed_root / scale / 多碰撞 |
| `scene.create_actor_builder()` | 创建刚体 | add_box / sphere / mesh / capsule |
| `actor.set_pose(Pose)` | 设置位姿 | Pose = [position, quaternion] |
| `articulation.set_qpos(arr)` | 设置关节角 | 长度 = dof |
| `articulation.set_qvel(arr)` | 设置关节速度 | 长度 = dof |
| `articulation.get_qpos()` | 获取关节角 | numpy |
| `articulation.compute_forward_kinematics()` | 计算正运动学 | 更新 link pose |
| `link.get_pose()` | 获取 link 位姿 | 必须在 compute_fk 之后 |
| `scene.get_contacts()` | 获取所有接触 | list of list |
| `scene.step()` | 推进 dt | 默认 dt=1/100 |
| `scene.set_timestep(1/240)` | 设置 dt | 灵巧手 ≤ 1/240 |
| `joint.set_drive_properties(k, d)` | 设置 PD | stiffness / damping |

### 16.3 PhysX 物理参数对照

| 参数 | 默认 | 灵巧手推荐 |
| :--- | :--- | :--- |
| `timestep` | 1/100 | 1/240 |
| `max_contacts_per_pair` | 4 | 1-2 |
| `friction_offset_threshold` | 0.01 | 0.005 |
| `friction_correlation_distance` | 0.025 | 0.01 |
| `bounce_threshold_velocity` | 2.0 | 0.5 |
| `solver_iterations` | 4 | 8 (TGS) |
| `global_stiffness` | 1000 | 5000 |

### 16.4 关键工具库

sapien.core (核心), sapien.sensor (相机 + 力传感器), sapien.utils (SubprocEngine / Viewer), sapien.asset (加载 mesh / URDF).

---

## 17. 附录 B: 扩展阅读与交叉引用

### 17.1 与本文件夹其他章节的深度交叉

- [[第07章-01-MuJoCo-XML建模与控制]]: SAPIEN v1 早期用 MuJoCo, 后续迁移 PhysX. MuJoCo 的 LCP vs PhysX 的 PGS 在数值上会有 10-30% 差异.
- [[第07章-02-Isaac-Sim-工业级GPU仿真]]: Isaac Lab (PhysX 5 GPU) 是 2024 后的工业标准, SAPIEN 在大规模 RL 上被替代. USD 比 ArticulationBuilder 更通用, 关节物体 API 不如 SAPIEN 直观.
- [[第07章-03-Habitat-室内导航仿真]]: Habitat 解决导航, SAPIEN/DexArt 解决操作, 都基于 UE4 渲染管线. 视觉模型可共享.
- [[第07章-04-ManiSkill-操作任务基准]]: ManiSkill v1/v2 用 SAPIEN, v3 (2024) 转 ManiSkill Engine (PhysX 5 GPU). v2 API 文档可直接用, v3 是新生态.
- [[第07章-05-RoboCasa与行为树]]: RoboCasa 偏家庭操作 + 行为树, SAPIEN/DexArt 偏单任务 + 灵巧手 + Sim2Real. 两者可组合: RoboCasa 行为树 + DexArt 灵巧手策略.

### 17.2 跨文件夹深度引用

- `../../../基础/数学基础/第06章-最优化/第06章-05-凸优化理论-强凸·光滑·Lipschitz·收敛率证明.md`: 接触不变优化 (CIO) 把库仑摩擦锥建模成二阶锥约束, 用凸优化求解.
- `数学基础/8-控制论/3-李群与刚体运动`: 末端位姿用 SE(3) 表达, 关节空间用 $S^1^n$. 零空间算子在 SO(3) 几何下更鲁棒.
- `../第03章-机器人运动学与控制/0-总目录与阅读路径.md`: 接触事件的概率建模 + Bayesian filter. Diffusion Policy 的 DDPM 也是变分推断.
- `../第01章-具身智能概论/0-总目录与阅读路径.md`: 摩擦锥作为分布约束, Dirac delta 表达接触点.
- `数学基础/4-线性代数/2-矩阵分解与应用`: Featherstone 算法用 $L D L^\top$ 分解关节树质量矩阵, 灵巧手 24×24 稀疏矩阵 LDL 比 LU 快 10×.
- `算法基础/3-数值优化/5-二阶锥规划`: 库仑锥用二阶锥规划 (SOCP) 求解, 比 LCP 更稳定.

### 17.3 推荐学习路径

第一次接触灵巧手仿真, 推荐阅读顺序: 第 1-3 节 (直觉引入 + SAPIEN 架构 + PhysX 接触求解器) → 第 4 节 (灵巧手硬件对比, 决定 Shadow / LEAP / Allegro 选型) → 第 6 节 (接触力学基础 + 库仑摩擦 + LCP) → 第 8 节 (强化学习 / 模仿学习训练算法选型) → 第 11 节 (Articulation Tree + 零空间) → 第 12 节 (进阶 PyTorch 代码, 动手实现 BC 策略) → 第 13 节 (工业案例, 真实部署的工程细节) → 第 14 节 (性能优化, 大规模 RL) → 第 15 节 (Q&A 第二轮, 面试准备).

### 17.4 关键论文清单(2023-2024)

- **DexArt (RSS 2023)**: 8 任务 IL baseline, Berkeley.
- **ManiSkill2 (ICLR 2023)**: 20+ 任务通用基准, SAPIEN 后端.
- **ManiSkill3 (CoRL 2024)**: PhysX 5 GPU 重构.
- **DexPoint (RSS 2023)**: 点云输入的灵巧手策略.
- **DiffDex (ICRA 2024)**: Diffusion Policy for Dexterous.
- **Handover-Sim (CoRL 2024)**: 灵巧手 handover 任务.
- **DexSim2Real (IROS 2024)**: 系统化的灵巧手 sim2real 协议.

### 17.5 工业工具链 + 灵巧手硬件(简版)

工具链: SAPIEN (USC), Isaac Lab (NVIDIA), MuJoCo XLA (DeepMind), Drake (MIT), PyBullet, ManiSkill3 (UCSD). 选型看 GPU pipeline + 关节物体表达 + 社区维护 + 生态兼容. 2024 工业首选 Isaac Lab, 学术首选 SAPIEN.

硬件: 学术 LEAP Hand (Stanford + Meta, $2k), OpenHand (Yale). 商业 Shadow ($100k 旗舰), Allegro ($25k 工业首选), Inspire RH56 ($8k). 中国 Dobot, Unitree, Robotera. 选型看 DoF + 成本 + 软件栈 + 维护, 学术 LEAP, 工业 Allegro, 旗舰 Shadow.

### 17.7 常见安装与运行问题

- **`pip install sapien` 失败**: SAPIEN wheel 仅在 Linux + Python 3.8-3.10 提供. macOS / Windows 需源码编译, 但 PhysX 4 不支持最新 macOS, 建议 Docker.
- **`RuntimeError: PhysX not found`**: 缺 NVIDIA PhysX SDK, GPU 模式需 CUDA 11.x.
- **`CUDA OOM`**: 显存不够, 降 env 数或关闭渲染 (见 14.2).
- **`接触点全是 0`**: `scene.update_render()` 必须在 `scene.step()` 之后, 否则接触 buffer 未刷新.
- **`URDF 加载失败`**: 检查 `<inertial>` 标签完整, SAPIEN 要求每个 link 有质量 + 转动惯量.

---

**章节真正结束**

> **学完这一章 + 附录, 你应该能动手做的事(扩展版)**:
>
> 1. 在 SAPIEN 里加载 Shadow Hand / LEAP Hand / Allegro, 跑 reach + grasp + insert 三阶段 demo.
> 2. 复现 DexArt buckle 任务, 训一个 BC 策略到 70%+.
> 3. 用 Diffusion Policy 替换 BC, 看涨多少.
> 4. 评估自己的 sim2real gap, 设计 domain randomization 方案.
> 5. 在 SAPIEN vs Isaac Lab 之间做选型, 给团队一份技术报告.
> 6. 调试接触点消失 / 物体乱飞的 bug, 用 §12.3 的三诊断法.
> 7. 把多指灵巧操作策略部署到 Allegro 真机, gap 控制在 10% 以内.
> 8. 在面试里说出 PhysX 4 vs 5 的本质差异, Articulation 零空间几何, Diffusion Policy 为什么比 BC 高 15%.
>
> **下一章预告**: 第 7 章 Robocasa 仿真, 家庭操作基准 + 行为树 + 大规模场景生成, 与本章"单任务 + 灵巧手"互补, 构成操作仿真两大范式.

(新增内容结束: 1-10 章正文 + 11-17 章扩展 + 附录 + Q&A 两轮)
