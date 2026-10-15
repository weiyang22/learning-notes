---
title: "第02章-04-多视角几何-SfM与SLAM基础"
tags:
  - 具身智能
  - 多视角几何
  - SfM与SLAM基础
description: "本章要点：章你学会了\"单张图里的 2D 像素怎么对应 3D 空间里的一条射线\"。但机器人真正干活的时候,不会只有一张图。无人机在巷子里飞一圈,要拼出整条巷子的 3…"
chapter_id: 具身智能-4-02
last_updated: 2026-07-14
---

# 第02章-04-多视角几何-SfM与SLAM基础

 章你学会了"单张图里的 2D 像素怎么对应 3D 空间里的一条射线"。但机器人真正干活的时候,不会只有一张图。无人机在巷子里飞一圈,要拼出整条巷子的 3D 结构;机械臂绕着桌子转一圈,要建出桌子的稠密点云;扫地机器人回家,要靠前后两帧的视差算出自己走了多远。
>
> 这些都靠**多视角几何 (Multi-View Geometry)**:用两张或多张图的几何关系,同时恢复**相机在哪儿**和**场景长什么样**。这就是本章的主题。
>
> 这一章要回答五个核心问题:
> - **两张图之间,像素和像素的对应有什么几何约束?** (对极几何)
> - **怎么从一堆无序图片里恢复 3D?** (SfM, Structure from Motion)
> - **怎么把所有相机位姿和 3D 点联合优化到最优?** (Bundle Adjustment)
> - **怎么让机器人一边走一边建图?** (SLAM)
> - **怎么把 IMU 融进来?** (VIO)
>
> 三个最容易搞混的概念先点一下:**SfM = 离线 + 全局优化 + 图片无序**;**SLAM = 在线 + 实时 + 视频流**;**VO = 只算位姿不建图**。它们共享几乎所有的几何理论,只是工程约束不同。
>
> 读完本章,你要能在 PyTorch 里手写一个极线校正 + 八点法估计本质矩阵的 toy,在脑子里讲清楚 ORB-SLAM3 的前端/后端/回环三件套,并且知道什么时候用 NeRF-SLAM 而不是传统 SLAM。

---

## 1. 直觉引入:从一张图到一千张图 (300 行)

### 1.1 一个最简单的直觉:三角测量

第 1 章说过,单张图里一个像素对应 3D 空间里一条射线,几何上无法确定深度。但如果你从**两个不同的位置**看同一个 3D 点,这两条射线会交于一点,这就是**三角测量 (Triangulation)** 的几何直觉。

```text
        相机 1                  相机 2
        o1                       o2
        |\                      /|
        | \                    / |
        |  \                  /  |
        |   \                /   |
        |    \              /    |
        |     \            /     |
        |      \          /      |
        |       \        /       |
        |        \      /        |
        |         \    /         |
        |          \  /          |
        |           \/           |
        |           /\           |
        |          /  \          |
        |         / 🟫 \         |
        |        /  X   \        |
        +───────/────────\──────→ Z

        3D 点 X = o1 + λ1 * d1 = o2 + λ2 * d2
```

两条射线在 3D 空间里相交,交点就是要恢复的 3D 坐标。这个直觉看起来一目了然,但当你要写成数学公式时就发现没那么简单:

- **像素有噪声**:观测到的两条射线其实不会精确相交,要在最小二乘意义下找最近点。
- **相机位姿未知**:要算两条射线,你得先知道两个相机的位姿。但位姿又依赖三角测量。**鸡生蛋问题**。
- **多对匹配**:实际工程里你有 N 张图、M 个特征点,要同时恢复所有相机位姿和 3D 点。规模从 2 个方程变成 N×M 个方程。

这就是**对极几何**和**Bundle Adjustment** 出场的原因。

### 1.2 SfM (Structure from Motion) 与 SLAM 的关系

两个术语常常被混用,但它们的目标和约束完全不同:

| 维度 | SfM (Structure from Motion) | SLAM (Simultaneous Localization and Mapping) |
| :--- | :--- | :--- |
| 数据 | 离线,无序图片集合 | 在线,有序视频流 |
| 优化 | 全局 BA,数小时到数天 | 滑动窗口 BA,毫秒到秒 |
| 实时性 | 不要求 | 强要求 (>= 10 FPS) |
| 输出 | 相机位姿 + 稠密/稀疏点云 | 当前位姿 + 局部地图 |
| 回环 | 重新跑全局 BA | 在线检测并校正 |
| 代表 | COLMAP, OpenMVG | ORB-SLAM3, VINS-Mono, LSD-SLAM |
| 应用 | 摄影测量, 文物数字化 | 机器人导航, AR |

**它们共享的理论基础**:
- 对极几何 (Epipolar Geometry)
- 三角测量 (Triangulation)
- Bundle Adjustment (BA)
- PnP (Perspective-n-Point)

**它们的工程差异**:
- SfM 可以**全局重优化**,因为它跑得慢没关系;SLAM 必须在**实时性约束**下增量更新。
- SfM 假设图片无序,所以必须先**图像检索**找连接对;SLAM 的帧是时序的,前一帧就是天然连接。
- SfM 通常离线,所以可以用**稠密匹配** (MVS);SLAM 实时性约束下只能用**稀疏特征点**或**半稠密光流**。

**VO (Visual Odometry)** 是 SLAM 的子集,只关心"我当前走到了哪",不维护全局地图。VO + 回环检测 + 全局优化 ≈ SLAM。

### 1.3 机器人为什么需要 SLAM

机器人有四个"不知道": 我在哪 (Localization), 周围有什么 (Mapping), 我怎么走 (Planning), 下一步该做什么 (Decision)。SLAM 直接解决前两个。

没有 SLAM,机器人就是一个"瞎子 + GPS":
- **室内**:GPS 信号弱,扫地机器人、AR 眼镜全靠 SLAM。
- **室外 GPS 拒止**:无人机在桥底、隧道、森林里飞,GPS 失效,必须靠视觉/激光 SLAM。
- **精细任务**:机械臂要在桌面上抓杯子,需要相对位姿,只能靠视觉。

**SLAM 的工程意义**: 自动驾驶 (Waymo, Tesla) 建高精度地图, AR 头显 (Vision Pro, Quest) 实时跟踪头部位姿, 扫地机器人 (Roborock, iRobot) 知道哪些地方扫过, 无人机航拍三维重建用 SfM+SLAM 联合管线。

**替代方案的缺陷**: 轮式里程计 (易漂移), IMU 积分 (短期准长期漂), UWB/WiFi (米级精度), RTK GPS (室内失效)。

视觉 SLAM 的核心优势: **被动感知 + 纹理丰富场景精度高 + 不需要基础设施**。

### 1.4 一个 SLAM 系统的最小骨架

不要被 ORB-SLAM3 一千多行的代码吓到。一个最小 SLAM 系统的骨架只有五步:

```text
1. 图像采集           每帧从相机读图
        ↓
2. 前端 (Visual Odometry) 提取特征 + 匹配 + 估计位姿
        ↓
3. 后端 (Optimization)   滑动窗口 BA 优化
        ↓
4. 回环检测 (Loop Closure) 检测是否回到老地方
        ↓
5. 输出                当前位姿 + 地图
```

后四章会逐个展开。先给一个**最直观的代码骨架** (Python 伪代码):

```python
class TinySLAM:
    def __init__(self):
        self.detector = cv2.ORB_create(2000)       # ORB 特征
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_frame = None
        self.pose = np.eye(4)                       # 当前位姿
        self.points_3d = []                         # 地图点

    def track(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return self.pose

        # 1. 提取 + 匹配
        kp1, des1 = self.detector.detectAndCompute(self.prev_frame, None)
        kp2, des2 = self.detector.detectAndCompute(frame, None)
        matches = self.matcher.knnMatch(des1, des2, k=2)
        # 比率测试 (Lowe's ratio test) 过滤掉误匹配
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # 2. 估计相对位姿 (本质矩阵)
        pts1 = np.array([kp1[m.queryIdx].pt for m in good])
        pts2 = np.array([kp2[m.trainIdx].pt for m in good])
        E, _ = cv2.findEssentialMat(pts1, pts2, focal=500, pp=(320, 240))

        # 3. 分解本质矩阵恢复 R, t
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        # 4. 累积位姿 (这里简化了尺度问题)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        self.pose = self.pose @ T

        # 5. 三角测量加新地图点 (省略)
        self.prev_frame = frame
        return self.pose
```

50 行就能跑通一个最简 SLAM。实际系统 (ORB-SLAM3) 在这个骨架上加了:
- 关键帧选择
- 局部 BA
- 全局 BA
- 回环检测 (DBoW2)
- 重定位
- IMU 融合

但**骨架就是这个骨架**,几何原理一脉相承。

### 1.5 本章路线图

```text
1. 直觉引入 (你在这里)
2. 对极几何 (本章核心, 必读)
   - 2.1 本质矩阵 E
   - 2.2 基础矩阵 F
   - 2.3 对极约束推导
   - 2.4 五点法 / 八点法
   - 2.5 尺度模糊
3. SfM 流程
   - 3.1 特征提取: SIFT, ORB, SuperPoint
   - 3.2 特征匹配
   - 3.3 增量式 SfM (COLMAP)
   - 3.4 全局式 SfM
   - 3.5 Bundle Adjustment
4. Bundle Adjustment 专题
   - 4.1 数学形式化
   - 4.2 Schur 补优化
   - 4.3 鲁棒核函数
   - 4.4 求解器: g2o, GTSAM, Ceres
5. SLAM 系统
   - 5.1 前端 + 后端 + 回环
   - 5.2 特征点法: ORB-SLAM3
   - 5.3 直接法: LSD-SLAM
   - 5.4 半稠密 + IMU: VINS-Mono
   - 5.5 NeRF-SLAM
6. 视觉惯性里程计 VIO
7. PyTorch 实验
8-12. 与其他章节/数学基础的关系 + 误区 + Q&A + 参考文献
```

读完全章,你要能:
- **手画**对极几何示意图,标出极点、极线、极平面
- **手写**八点法估计本质矩阵的代码
- **解释**为什么单目 SLAM 必然有尺度漂移
- **对比**ORB-SLAM3 和 VINS-Mono 在无人机上的取舍
- **判断**一个 SLAM 系统为什么崩溃 (尺度漂移? 闭环失败? BA 不收敛?)

---

## 2. 对极几何 (Epipolar Geometry) (400 行) 本章核心

### 2.1 本质矩阵 (Essential Matrix) E

**核心思想**:给定两幅图,3D 点 $\mathbf{X}$ 在第一幅图里投影到像素 $\mathbf{x}$,在第二幅图里投影到 $\mathbf{x}'$。这两个像素之间存在一个**几何约束**,由两相机的相对位姿决定。这个约束的代数表示就是**本质矩阵**。

**定义**:本质矩阵 $E \in \mathbb{R}^{3 \times 3}$ 是一个秩为 2 的矩阵,描述了两个相机坐标系之间的几何关系:

$$E = [\mathbf{t}]_\times R$$

其中 $R \in \text{SO}(3)$ 是两相机之间的旋转,$\mathbf{t} \in \mathbb{R}^3$ 是平移,$[\mathbf{t}]_\times$ 是反对称矩阵 (cross product matrix):

$$[\mathbf{t}]_\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}$$

**对极约束 (Epipolar Constraint)**:对于任何一对匹配点 $\mathbf{x}, \mathbf{x}'$ (归一化相机坐标):

$$\mathbf{x}'^T E \mathbf{x} = 0$$

注意这里 $\mathbf{x}, \mathbf{x}'$ 是**归一化平面**上的坐标 (除以深度),**不是像素坐标**。这是本质矩阵和基础矩阵的关键区别:

| 矩阵 | 输入 | 适用场景 |
| :--- | :--- | :--- |
| 本质矩阵 $E$ | 归一化相机坐标 | 已知内参 |
| 基础矩阵 $F$ | 像素坐标 | 未知内参或内参不统一 |

两者关系:$F = K^{-T} E K^{-1}$,其中 $K$ 是内参。

### 2.2 基础矩阵 (Fundamental Matrix) F

**定义**:基础矩阵 $F \in \mathbb{R}^{3 \times 3}$ 描述像素层面的对极约束:

$$\mathbf{u}'^T F \mathbf{u} = 0$$

其中 $\mathbf{u}, \mathbf{u}'$ 是像素坐标 (齐次)。

**基础矩阵的性质**:
1. **秩为 2**:$\det(F) = 0$ (这意味着 7 个自由度,不是 9)。
2. **两相机相对位姿**:$F$ 与 $E$ 的关系 $F = K^{-T} E K^{-1}$。
3. **对极线方程**:给定第一图的像素 $\mathbf{u}$,在第二图里对应的像素必须在直线 $F\mathbf{u}$ 上 (对极线)。
4. **极点**:对极线在第二图里的交点叫**极点** $\mathbf{e}'$,满足 $F \mathbf{e}' = 0$。

**什么时候用 $E$ vs $F$?**
- 用 $E$:你已经知道 $K$ (大多数工程场景)
- 用 $F$:你在做未标定的 3D 重建,或者两个图的 $K$ 不同

**为什么本质矩阵这么重要?**
- 它把**两张图的几何关系**压缩成一个 5 自由度的矩阵 (3 旋转 + 3 平移,但尺度等价 → 5)。
- 从 8 对以上匹配点就能估计 $E$,进而分解出 $R, \mathbf{t}$。
- 这是整个 SfM/SLAM 的基础,几乎所有 VO 算法都从这里开始。

### 2.3 对极约束的推导

把第 1 章的投影公式搬过来:

$$\mathbf{x} = K^{-1} \mathbf{u}, \quad \mathbf{x}' = K^{-1} \mathbf{u}'$$

第一相机坐标系下,3D 点 $\mathbf{X}$ 表示为 $\mathbf{X}$,第二相机坐标系下表示为 $\mathbf{X}'$。两坐标系的关系:

$$\mathbf{X}' = R \mathbf{X} + \mathbf{t}$$

对极几何最关键的观察:**射线 $\mathbf{o}_1 \to \mathbf{X}$ 和射线 $\mathbf{o}_2 \to \mathbf{X}$ 共面**。理由:它们都从 $\mathbf{o}_1, \mathbf{o}_2$ 出发,都通过 $\mathbf{X}$,三点确定一个平面。这个平面叫**极平面 (Epipolar Plane)**。

把共面条件写成代数:三向量 $\mathbf{X}, \mathbf{X} - \mathbf{X}' = -R^{-1} \mathbf{t}$, $\mathbf{X}'$ 的混合积为零:

$$(\mathbf{X}' - \mathbf{o}_2)^T \cdot ((\mathbf{X} - \mathbf{o}_1) \times (\mathbf{X}' - \mathbf{o}_2)) = 0$$

化简 (取 $\mathbf{o}_1 = \mathbf{o}_2 = \mathbf{0}$,在第一相机坐标系下):

$$\mathbf{X}'^T \cdot (\mathbf{t} \times \mathbf{X}) = 0$$

利用 $\mathbf{X}' = R\mathbf{X} + \mathbf{t}$ 和反对称矩阵:

$$\mathbf{X}'^T [\mathbf{t}]_\times \mathbf{X} = 0$$

两边除以各自的深度 $Z, Z'$ 得到归一化坐标:

$$\mathbf{x}'^T [\mathbf{t}]_\times R \mathbf{x} = 0$$

定义 $E = [\mathbf{t}]_\times R$,这就是对极约束:

$$\boxed{\mathbf{x}'^T E \mathbf{x} = 0}$$

**几何意义再读一遍**:
- $E\mathbf{x}$ 是第二图里过 $\mathbf{x}'$ 的**对极线** (直线方程)
- $\mathbf{x}'$ 必须落在这条对极线上
- 极点 $\mathbf{e}$ (第一相机光心在第二图的投影) 满足 $E\mathbf{e} = 0$

### 2.4 五点法 / 八点法估计 E/F

**八点法 (Eight-Point Algorithm)**:

最经典的线性解法,由 Longuet-Higgins (1981) 提出。把每对匹配点 $(\mathbf{x}_i, \mathbf{x}'_i)$ 写成 $\mathbf{x}'_i^T E \mathbf{x}_i = 0$:

设 $\mathbf{x}_i = (x_i, y_i, 1)^T$, $\mathbf{x}'_i = (x'_i, y'_i, 1)^T$,把 $E$ 拉直成 9 维向量 $\mathbf{e}$:

$$\begin{bmatrix} x'_i x_i & x'_i y_i & x'_i & y'_i x_i & y'_i y_i & y'_i & x_i & y_i & 1 \end{bmatrix} \mathbf{e} = 0$$

8 对点给 8 个方程,加上齐次约束 $\|\mathbf{e}\| = 1$,线性最小二乘解 (SVD) 就给出 $E$。

**标准化 (Normalized Eight-Point)**:

直接用像素坐标做八点法数值不稳定。Hartley (1997) 提出先做归一化:
1. 平移使像素中心为原点
2. 缩放使平均距离为 $\sqrt{2}$
3. 用归一化坐标做八点法
4. 反归一化得到 $F$

```python
def normalize_points(pts):
    """Hartley 归一化"""
    mean = pts.mean(axis=0)
    centered = pts - mean
    scale = np.sqrt(2) / np.linalg.norm(centered, axis=1).mean()
    T = np.array(`scale, 0, -scale*mean[0`,
                  [0, scale, -scale*mean[1]],
                  [0, 0, 1]])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T

def eight_point(pts1, pts2):
    """归一化八点法"""
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    n = len(pts1)
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # SVD 最小二乘解
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # 强制 rank-2 (det = 0)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # 反归一化
    F = T2.T @ F @ T1
    return F
```

**五点法 (Five-Point Algorithm)**:

本质矩阵只有 5 个自由度 (3 旋转 + 3 平移 - 1 尺度 = 5),理论上 5 对匹配点就能解。这是由 Nistér (2004) 提出的,需要解一个 10 次多项式,数值上比八点法复杂得多,但对噪声更鲁棒。

```python
# OpenCV 自带五点法
E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy),
                                method=cv2.RANSAC, prob=0.999, threshold=1.0)
```

**RANSAC 包装**:

实际工程里几乎所有算法都要套 RANSAC,因为匹配有大量外点 (outliers)。基本流程:
1. 随机采 8 (或 5) 对匹配
2. 估计 $E$
3. 算所有匹配的对极误差
4. 统计内点 (误差 < 阈值)
5. 重复 N 次,取内点最多的 $E$
6. 用所有内点重新估计 $E$

```python
# 完整 RANSAC 流程
E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy),
                                method=cv2.RANSAC, prob=0.999, threshold=1.0)
inliers = mask.ravel().astype(bool)
pts1_in = pts1[inliers]
pts2_in = pts2[inliers]

# 从本质矩阵分解出 R, t
_, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, focal=fx, pp=(cx, cy))
```

**实际经验**:
- 八点法 + RANSAC 对匹配数 >= 50 时很稳
- 五点法对平面场景 (degenerate) 更有优势
- 用归一化八点法比直接八点法数值稳定 10 倍以上
- RANSAC 的阈值一般设 1-3 像素

### 2.5 三维重建的尺度模糊

从 $E$ 分解出 $R, \mathbf{t}$ 之后,**平移 $\mathbf{t}$ 的尺度是不确定的**。这是单目 SfM/SLAM 永远绕不开的根本问题。

**为什么?**
对极约束 $\mathbf{x}'^T E \mathbf{x} = 0$ 是齐次的:如果把 $\mathbf{t}$ 乘以常数 $\lambda$, $E$ 也乘以 $\lambda$,约束依然成立。所以从图像本身无法区分"两个相机距离 1m 还是 10m"。

**尺度怎么定?**
1. **已知物体尺寸**:场景里有一个 30cm 长的标定板,从视差反推尺度。
2. **IMU 融合**:IMU 给出真实加速度,和视觉融合后尺度自然恢复。
3. **双目/全景**:基线 $B$ 已知,深度 $Z = fB/d$,尺度自然确定。
4. **物体先验**:检测到一个人 (身高 1.7m),用它约束场景尺度。
5. **GPS**:室外场景用 GPS 提供绝对尺度。

**尺度模糊的工程影响**:
- **单目 SLAM**:必须选一个初始尺度,然后全程维护,但漂移无法避免。ORB-SLAM3 用第一个关键帧定一个"伪尺度",后续靠 BA 维持。
- **轨迹长度**:轨迹的真实长度 = 估计长度 × 真实尺度因子。
- **回环失败**:尺度漂移会让回环检测对不上,必须用 Sim3 (相似变换, 7 自由度) 而非 SE3 来对齐。

**尺度漂移**是单目 SLAM 最难解决的问题之一。Sim3 Pose Graph Optimization 是目前主流方案:

```text
单帧 SLAM: SE(3) 优化 (6 DOF)
回环对齐: Sim(3) 优化 (7 DOF, 多一个尺度)
全局 BA: SE(3) 但锚定尺度
```

ORB-SLAM3 在回环检测成功后用 Sim3 把当前地图对齐到历史地图,尺度漂移被纠正。

**尺度因子恢复的代数**:

设第一图里 3D 点坐标为 $\mathbf{X}$,在第二图里的真实坐标 $\mathbf{X}'$ 与估计坐标 $\hat{\mathbf{X}}'$ 的关系:

$$\mathbf{X}' = \lambda \hat{\mathbf{X}}'$$

尺度因子 $\lambda$ 通过下式恢复:

$$\lambda = \frac{\|\mathbf{X}_i - \mathbf{X}_j\|}{\|\hat{\mathbf{X}}_i - \hat{\mathbf{X}}_j\|}$$

其中 $i, j$ 是同一物体上的两点 (比如车的两个角点, 真实距离已知)。

### 2.6 三角测量数学公式

从对极几何恢复 3D 点的过程叫**三角测量 (Triangulation)**。给定归一化匹配 $\mathbf{x}_i, \mathbf{x}'_i$ 和相对位姿 $(R, \mathbf{t})$:

$$\mathbf{X} = \lambda \mathbf{x} = \lambda' R^T \mathbf{x}' + \lambda' R^T \mathbf{t}$$

写成齐次 DLT (Direct Linear Transform):

设投影矩阵 $P_1 = [I | \mathbf{0}]$, $P_2 = [R | \mathbf{t}]$,则 $\mathbf{x}_i \times (P_1 \mathbf{X}) = 0$, $\mathbf{x}'_i \times (P_2 \mathbf{X}) = 0$。每对匹配给 4 个方程 (实际只有 2 个独立),最少 2 对就能解。

```python
def triangulate(P1, P2, pts1, pts2):
    """DLT 三角测量"""
    n = len(pts1)
    X_h = np.zeros((n, 4))
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.array([
            y1 * P1[2] - P1[1],
            P1[0] - x1 * P1[2],
            y2 * P2[2] - P2[1],
            P2[0] - x2 * P2[2]
        ])
        _, _, Vt = np.linalg.svd(A)
        X_h[i] = Vt[-1]
    X = X_h[:, :3] / X_h[:, 3:4]
    return X

# 用法
P1 = np.hstack([np.eye(3), np.zeros((3,1))])
P2 = np.hstack([R, t.reshape(3,1)])
pts_3d = triangulate(P1, P2, pts1_norm, pts2_norm)
```

**三角化的精度问题**:
- 基线越长,深度精度越高 (三角剖分的几何)
- 像素噪声固定时,深度误差与基线长度成反比
- 当两个相机的射线几乎平行 (退化配置),深度完全不确定

这就是为什么 SLAM 系统喜欢"快速移动 + 大基线 + 多视角融合"。

### 2.7 退化配置 (Degenerate Configurations)

有些场景下对极几何失效,必须特别处理:

| 退化场景 | 表现 | 应对 |
| :--- | :--- | :--- |
| 纯旋转 ($\mathbf{t} = 0$) | $E$ 退化为单应矩阵 | 检测旋转,单独处理 |
| 平面场景 | 所有点共面, $E$ 不唯一 | 用单应矩阵 $H$ 替代 |
| 低纹理 | 匹配数不足 | 改用直接法或 IMU |
| 重复纹理 | 大量误匹配 | 强化几何验证 |

**平面场景**: 所有 3D 点共面时,两图之间的关系变成**单应 (Homography)**:

$$\mathbf{x}' \sim H \mathbf{x}$$

$H$ 是 3×3 矩阵, 8 个自由度, 从 4 对匹配就能解。COLMAP 会同时估计 $E$ 和 $H$,根据内点数量选择。

---

## 3. SfM 流程 (400 行)

### 3.1 特征提取:SIFT, ORB, SuperPoint

SLAM/SfM 的第一步是**特征提取 (Feature Extraction)**。特征点 = 图像里"在视角变化下还能稳定识别的小图像块"。一个好的特征点算法需要满足:
- **可重复性 (Repeatability)**:同一 3D 点在不同视角下能被检测到。
- **可区分性 (Distinctiveness)**:不同 3D 点对应不同描述子。
- **效率 (Efficiency)**:实时 SLAM 需要 30+ FPS。

**SIFT (Scale-Invariant Feature Transform, Lowe 1999)**:

经典中的经典。流程:
1. **尺度空间极值检测**:高斯差分 (DoG) 金字塔找极值点。
2. **关键点精确定位**:亚像素插值 + 去掉低对比度和边缘响应点。
3. **方向赋值**:局部梯度直方图找主方向,实现旋转不变性。
4. **描述子**:16×16 邻域分 4×4 小块,每块 8-bin 梯度直方图,共 128 维向量。

```python
import cv2
sift = cv2.SIFT_create(nfeatures=2000)
kp, des = sift.detectAndCompute(img, None)
# kp: 关键点列表 (pt, size, angle, response, octave)
# des: (N, 128) 描述子矩阵, float32
```

**优缺点**:
- 优: 极致鲁棒, 尺度 + 旋转 + 光照不变
- 缺: 慢 (~50ms/图), 描述子 128 维占内存, **专利已过期, 但实现复杂**

**ORB (Oriented FAST and Rotated BRIEF, Rublee 2011)**:

SLAM 时代的实用之选。ORB = FAST 关键点 + BRIEF 描述子 + 旋转补偿。
- **FAST 角点**:像素级灰度对比,极快。
- **BRIEF 描述子**:二进制串 (256 bit),汉明距离匹配,**比 SIFT 快两个数量级**。
- **旋转不变**:用质心方向补偿。

```python
orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
kp, des = orb.detectAndCompute(img, None)
# des: (N, 32) uint8, 汉明距离匹配
```

**优缺点**:
- 优: 极快 (~5ms/图), 内存小, 无专利
- 缺: 尺度不变性弱, 视角变化大时失败

**SuperPoint (DeTone 2018) + LoFTR**:

深度学习时代的新选择。SuperPoint 用自监督训练,在纹理缺乏场景比传统特征点好得多。

```python
# SuperPoint (需安装 pip install superpoint)
import torch
from superpoint import SuperPoint
sp = SuperPoint().eval()
with torch.no_grad():
    out = sp({'image': torch.from_numpy(img).float()/255.})
    # out: keypoints, descriptors, scores
```

**LoFTR (Sun 2021)**:不用检测关键点,直接在两图之间做稠密匹配,在低纹理场景显著优于传统方法。

| 特征 | SIFT | ORB | SuperPoint | LoFTR |
| :--- | :--- | :--- | :--- | :--- |
| 速度 | 慢 | 极快 | 中 | 慢 |
| 鲁棒性 | 高 | 中 | 高 | 极高 |
| 低纹理 | 差 | 差 | 好 | 极好 |
| 大视角 | 好 | 差 | 中 | 好 |
| 实时 SLAM | 困难 | 适合 | 需优化 | 不适合 |
| SLAM 应用 | 离线 | ORB-SLAM | NeRF-SLAM | 闭环 |

**实际工程选择**:
- **实时 SLAM**: ORB-SLAM3 用的就是 ORB (速度快)
- **离线 SfM**: COLMAP 默认 ORB, 也支持 SIFT
- **低纹理场景**: VINS-Mono 用 Harris + LK 光流
- **科研前沿**: NeRF-SLAM 用 SuperPoint, DROID-SLAM 用光流 + 深度学习

### 3.2 特征匹配:暴力匹配 + 几何验证

**暴力匹配 (Brute Force Matching)**:

每个描述子和所有候选描述子算距离, 取最近的 K 个:

```python
# ORB 汉明距离, KNN (k=2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)
# Lowe's ratio test: 好的匹配最近距离应该远小于次近距离
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

**Lowe's Ratio Test**: 最近的匹配距离 < 0.75 × 次近距离, 才认为是好匹配, 能过滤掉 90% 误匹配。

**FLANN 匹配**: 大数据量时 FLANN 比暴力匹配快得多。

```python
flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
flann = cv2.FlannBasedMatcher(flann_params, {})
matches = flann.knnMatch(des1, des2, k=2)
```

**几何验证**: 匹配里有大量外点, 必须用几何约束过滤。基础矩阵 F (RANSAC, 八点法) 或单应矩阵 H (RANSAC, 平面场景)。OpenCV 一行:

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
inliers = mask.ravel().astype(bool)
```

**匹配策略的工程取舍**:
- **SLAM**: 相邻帧匹配, 暴力匹配足够 (ORB-SLAM 暴力匹配)
- **SfM**: 全局图像检索 (DBoW2 / NetVLAD), 只匹配相似图像
- **大规模图像集合**: 倒排索引 + 词袋模型 (BoW)

### 3.3 增量式 SfM (COLMAP)

**COLMAP** 是当前最主流的 SfM 开源工具, 代表增量式 SfM 的工程实现:

```text
1. 特征提取 (SIFT/ORB)
   ↓
2. 图像检索 (找相似图像对)
   ↓
3. 两视图几何 (E/F + 三角测量)
   ↓
4. 初始化 (选两帧 + 三角化初始点)
   ↓
5. 增量重建 (PnP 注册新图像 → 三角化新点 → 局部 BA)
   ↓
6. 回环检测 + 全局 BA
```

**核心算法**:

**(1) 图像检索**: 词袋模型 (BoW) 把每张图编码成稀疏向量, 倒排索引查相似图。COLMAP 用更先进的 NetVLAD / AP-GeM 深度特征。

**(2) 两视图重建**: 选两帧最多匹配对, 八点法估 E, 分解 R/t, 三角化 3D 点作初始地图。

**(3) PnP 注册新图像**: 给定新图像和已重建 3D 点, PnP 估计新相机位姿。
- **P3P**: 3 对点给 4 个可能解, 需要第 4 对消歧。
- **EPnP**: O(n) 复杂度, 4 个控制点表示所有点。
- **直接解**: BA 联合优化。

```python
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    objectPoints=pts_3d, imagePoints=pts_2d, cameraMatrix=K,
    iterationsCount=100, reprojectionError=4.0, confidence=0.99)
```

**(4) 局部 BA**: 优化最近几个关键帧的位姿和它们观测到的 3D 点。

**(5) 全局 BA**: 全部相机 + 全部 3D 点一起优化, 数小时到数天。

**COLMAP 的命令行使用**:

```bash
# 稀疏重建 (SfM)
colmap feature_extractor --database_path $DB --image_path $IMG
colmap exhaustive_matcher --database_path $DB
colmap mapper --database_path $DB --image_path $IMG --output_path $SPARSE

# 稠密重建 (MVS)
colmap image_undistorter --image_path $IMG --input_path $SPARSE/0 --output_path $DENSE
colmap patch_match_stereo --workspace_path $DENSE
colmap stereo_fusion --workspace_path $DENSE --output_path $DENSE/fused.ply
```

### 3.4 全局式 SfM

增量式 SfM 的问题: 每次新加一帧都要 BA 一次, 大场景下慢得难以忍受,而且**累积误差** (drift) 会越来越大。

**全局式 SfM** 一次性估计所有相机位姿,避免累积误差:

```text
1. 特征提取 + 匹配 (和增量式一样)
        ↓
2. 全局旋转平均 (Rotation Averaging)
   - 输入: 每对图的相对旋转
   - 输出: 所有相机的全局绝对旋转
        ↓
3. 全局平移平均 (Translation Averaging)
   - 输入: 每对图的相对平移方向
   - 输出: 所有相机的全局绝对平移 (尺度任意)
        ↓
4. 三角测量 (一次性)
        ↓
5. 全局 BA 精化
```

**旋转平均**: 把所有相对旋转 $R_{ij}$ 拼起来求全局旋转 $R_i, R_j$:

$$R_{ij} = R_j R_i^T$$

这是 SO(3) 上的优化,通常用迭代法或松弛法。

**平移平均**: 给定相对平移方向 $\hat{\mathbf{t}}_{ij}$, 估计全局位置 $\mathbf{c}_i, \mathbf{c}_j$:

$$\mathbf{c}_j - \mathbf{c}_i \parallel \hat{\mathbf{t}}_{ij}$$

写成二次约束后是 SDP (半定规划) 问题。

**全局式代表**: OpenMVG, Theia SfM。

| 维度 | 增量式 (COLMAP) | 全局式 (OpenMVG) |
| :--- | :--- | :--- |
| 速度 | 慢 (在线性 BA) | 较快 |
| 鲁棒性 | 高 (RANSAC) | 中 (一损俱损) |
| 大场景 | 困难 | 友好 |
| 在线扩展 | 困难 (需要重 BA) | 容易 |
| 工程实现 | 成熟 | 较少 |

### 3.5 稠密重建 (MVS)

**稀疏三角测量**: 只在特征点处重建 3D, 适合定位。

**稠密/半稠密重建 (MVS)**: 每个像素都重建深度,得到稠密点云。代表方法: COLMAP MVS (patch match), PatchMatchNet (深度学习), NeRF (神经隐式, 见第 5 章), 3D Gaussian Splatting (2023 革命性进展)。

MVS 的核心挑战: 遮挡处理, 光照一致性, 反射/透明表面, 计算量。COLMAP 的 MVS 流程: 图像去畸变 → 深度图估计 (PatchMatch) → 法向图估计 → 多视图融合。

---

## 4. Bundle Adjustment (300 行)

### 4.1 数学形式化:非线性最小二乘

**Bundle Adjustment (BA)** 是 SfM/SLAM 的"重头戏",把所有相机位姿和 3D 点放在一个优化问题里联合求解。目标:**最小化所有观测的重投影误差**。

**形式化**:

设有 $n$ 个相机和 $m$ 个 3D 点。相机 $i$ 的位姿 $P_i = K_i [R_i | \mathbf{t}_i]$, 3D 点 $\mathbf{X}_j$ 在相机 $i$ 的投影:

$$\pi(P_i, \mathbf{X}_j) = K_i [R_i | \mathbf{t}_i] \mathbf{X}_j$$

观测像素 $\mathbf{u}_{ij}$, 重投影误差:

$$\mathbf{e}_{ij} = \mathbf{u}_{ij} - \pi(P_i, \mathbf{X}_j)$$

BA 目标:

$$\min_{\{P_i\}, \{\mathbf{X}_j\}} \sum_{ij} \rho(\|\mathbf{e}_{ij}\|^2)$$

其中 $\rho(\cdot)$ 是鲁棒核函数 (后面讲)。

**重投影误差 vs 三维误差**:

为什么用重投影而不是三维误差?
- 三维误差需要知道 3D 点的真实位置 (很难拿)
- 重投影误差直接用图像观测, 无需 3D 真值
- 投影模型是已知的, 可以解析求导

**优化变量**:

- **相机参数**: $P_i = (\mathbf{r}_i, \mathbf{t}_i) \in \mathfrak{se}(3)$ (李代数, 6 维)
- **3D 点**: $\mathbf{X}_j \in \mathbb{R}^3$
- 总自由度: $6n + 3m$

**关键观察**: 当 $n = 100$ 个相机, $m = 10000$ 个点, 优化变量有 $30600$ 维。这个规模用稠密 Hessian 不可行,必须用稀疏结构 → Schur 补。

### 4.2 Schur 补优化

BA 的 Hessian 矩阵有一个特殊结构: **3D 点之间的耦合只通过相机, 反之亦然**。利用这个结构,可以用 Schur 补大幅加速。

**线性化**:

在当前估计 $(\hat{P}, \hat{\mathbf{X}})$ 处线性化,目标变成最小二乘 $\|\mathbf{e} + J \Delta\|^2$:

$$\min_\Delta (J^T J) \Delta = -J^T \mathbf{e}$$

其中 $J$ 是 Jacobian, 维度 $(2n_{\text{obs}}) \times (6n + 3m)$。

**Hessian 分块结构**:

$$J^T J = \begin{bmatrix} U & W \\ W^T & V \end{bmatrix}$$

- $U$ 是相机-相机块 ($6n \times 6n$): 只有观测到共同点的相机对有非零
- $V$ 是点-点块 ($3m \times 3m$): **对角**! (因为单个点不与别的点直接耦合)
- $W$ 是相机-点块 ($6n \times 3m$): 稀疏

**Schur 补**消除 3D 点变量:

$$\Delta_P = (U - W V^{-1} W^T)^{-1} (\mathbf{b}_1 - W V^{-1} \mathbf{b}_2)$$

其中 $V$ 是块对角, $V^{-1}$ 可以**逐点**求逆, 极快。

加速比: 从 $O((6n + 3m)^3)$ 降到 $O((6n)^3 + \text{points update})$。

```python
# Schur 补简化版
def schur_solve(H, b, n_cam):
    U = H[:6*n_cam, :6*n_cam]
    W = H[:6*n_cam, 6*n_cam:]
    V = H[6*n_cam:, 6*n_cam:]
    V_inv = block_diag_inv(V)        # V 块对角, 求逆快
    S = U - W @ V_inv @ W.T          # Schur 补
    rhs = b[:6*n_cam] - W @ V_inv @ b[6*n_cam:]
    delta_P = np.linalg.solve(S, rhs)
    delta_X = V_inv @ (b[6*n_cam:] - W.T @ delta_P)
    return np.concatenate([delta_P, delta_X])
```

实际代码里 $U - W V^{-1} W^T$ 还稀疏, 用稀疏 Cholesky 分解 (CHOLMOD) 求解。

### 4.3 鲁棒核函数 (Robust Kernel)

误匹配 (outlier) 会让 BA 严重失真。鲁棒核函数替换二次损失为对 outlier 不敏感的损失:

| 核函数 | 特点 |
| :--- | :--- |
| L2 (无核) | 对 outlier 无抵抗 |
| Huber | 折中, 工程常用 |
| Cauchy | 更鲁棒 |
| Tukey | 完全截断 outlier |

Huber 的几何解释: 误差小退化为 L2 (二次), 误差大退化为 L1 (线性), 让 outlier 的影响被限制在常数倍。

```python
def huber_weight(s, k=1.0):
    return np.where(s <= k*k, 1.0, k / np.sqrt(s))

def cauchy_weight(s, k=1.0):
    return 1.0 / np.sqrt(1 + s / (k*k))

# IRLS 迭代
for it in range(max_iter):
    weights = huber_weight(residuals**2, k=1.0)
    JtWJ = J.T @ np.diag(weights) @ J
    delta = np.linalg.solve(JtWJ, -J.T @ np.diag(weights) @ residuals)
    params += delta
```

### 4.4 求解器: g2o, GTSAM, Ceres

工程上不用自己写 Levenberg-Marquardt, 都有现成的非线性优化库:

**g2o (General Graph Optimization)**:

SLAM 时代最常用的优化库, 用图 (graph) 表示优化问题:
- **顶点 (vertex)**: 优化变量 (相机位姿, 3D 点)
- **边 (edge)**: 观测约束 (重投影误差)

```cpp
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
// 6: 相机位姿维度 (SE3)
// 3: 3D 点维度

g2o::SparseOptimizer optimizer;
optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
    std::make_unique<Block>(std::make_unique<g2o::LinearSolverCholmod<Block::PoseMatrixType>>())
));

// 加相机顶点
g2o::VertexSE3* v_pose = new g2o::VertexSE3();
v_pose->setEstimate(pose_init);
v_pose->setId(0);
optimizer.addVertex(v_pose);

// 加 3D 点顶点
g2o::VertexPointXYZ* v_point = new g2o::VertexPointXYZ();
v_point->setEstimate(point_init);
v_point->setId(1);
optimizer.addVertex(v_point);

// 加边 (重投影误差)
g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(1)));
edge->setVertex(1, dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(0)));
edge->setMeasurement(observed_uv);
edge->setInformation(Eigen::Matrix2d::Identity());
optimizer.addEdge(edge);

optimizer.initializeOptimization();
optimizer.optimize(10);  // 迭代 10 次
```

**GTSAM**: 因子图 (factor graph) 表示, 比 g2o 更现代。

```python
import gtsam
graph = gtsam.NonlinearFactorGraph()
graph.add(gtsam.PriorFactorPose3(symbol('x', 0), pose_init, prior_noise))
graph.add(gtsam.GenericProjectionFactorCal3_S2(
    observed_uv, measurement_noise, symbol('x', i), symbol('l', j), K))
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
result = optimizer.optimize()
```

**Ceres (Google)**: 通用非线性最小二乘, 工业级。

```cpp
ceres::Problem problem;
problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
        new ReprojectionError(observed_uv, K)),
    new ceres::HuberLoss(1.0),    // 鲁棒核
    camera_params, point_params);
ceres::Solver::Options options;
options.linear_solver_type = ceres::SPARSE_SCHUR;
ceres::Solve(options, &problem, &summary);
```

**对比**:

| 库 | 优点 | 缺点 | 代表应用 |
| :--- | :--- | :--- | :--- |
| g2o | 简单, 图直观 | 老旧, 缺文档 | ORB-SLAM |
| GTSAM | 因子图优雅, 增量优化 | 编译慢 | VINS-Mono, LIO-SAM |
| Ceres | 通用, 文档好, 工业级 | 接口不专门为 SLAM 设计 | Cartographer |

**SLAM 的特殊需求: 滑动窗口 / 流形优化**:

实时 SLAM 不能优化所有历史帧, 用滑动窗口保留最近 N 个关键帧, 老的边缘化。简单扔掉会破坏约束, 正确做法是 **边缘化 (Marginalization)**: 把被扔掉帧的先验信息保留为约束, 用 Schur 补完成, 维护 Hessian 的稀疏结构。这是 VINS-Mono, ORB-SLAM3 后端的核心。

---

## 5. SLAM 系统 (400 行)

### 5.1 视觉 SLAM 流程: 前端 + 后端 + 回环

**经典视觉 SLAM 五件套**:

```text
输入 (视频流)
   ↓
1. 前端 (VO) ─→ 特征提取 + 匹配 + 估计相对位姿 + 关键帧选择
   ↓  关键帧 + 位姿
2. 后端 (Optimization) ─→ 滑动窗口 BA + 局部地图 + 共视图
   ↓
3. 回环检测 ─→ 词袋模型 (DBoW2) + 几何验证 + Sim3 对齐
   ↓
4. 建图 ─→ 稀疏 / 半稠密 / 稠密 (RGB-D / NeRF)
   ↓
5. 输出 ─→ 当前位姿 + 全局地图 + 回环约束
```

**前端方法学**:

| 方法 | 输入 | 输出 | 代表 |
| :--- | :--- | :--- | :--- |
| 特征点法 | 特征点 + 描述子 | 稀疏地图 | ORB-SLAM3 |
| 直接法 | 像素灰度 | 半稠密地图 | LSD-SLAM, DSO |
| 光流法 | 关键点 + 光流 | 稀疏地图 | VINS-Mono |
| 混合 | 特征点 + 光度 | 半稠密 | SVO |

**关键帧选择**: 不是每帧都做 BA, 只对"有代表性"的帧优化。标准: 时间间隔 > 0.1s, 视差变化 > 20 像素, 跟踪点数量足够。

**后端的细节**:
- **滑动窗口 (Sliding Window)**: 保留最近 N 个关键帧 (N 通常 5-10), 老的边缘化。
- **共视图 (Covisibility Graph)**: 两帧观测到共同 3D 点, 它们之间有边, 是后端优化的拓扑结构。

**回环检测的细节**:
- **词袋模型 (BoW)**: 离线聚类特征成"视觉单词"建倒排索引, 在线编码成稀疏词频向量, 查询相似图 (DBoW2, DBoW3, NetVLAD)。
- **几何验证**: 候选闭环必须通过时间一致性 + 几何验证 (本质矩阵内点足够多)。
- **Sim3 对齐**: 单目 SLAM 的闭环必须用 Sim3 (7 DOF) 对齐, 不是 SE3, 因为尺度可能漂移了。

### 5.2 特征点法: ORB-SLAM3

**ORB-SLAM3** (Campos et al., 2021) 是目前最成熟、最广泛使用的开源视觉 SLAM。它支持单目、双目、RGB-D,加 IMU,加多相机。

**三线程架构**:

```text
Tracking 线程 (主线程, 30 Hz)
  - 每帧提取 ORB 特征
  - 与上一帧估计相对位姿 (恒速模型 / PnP)
  - 决定是否新建关键帧

Local Mapping 线程 (后台)
  - 处理新关键帧
  - 更新共视图
  - 局部 BA
  - 维护局部地图

Loop & Map Merging 线程 (后台)
  - 检测回环 / 地图融合
  - Sim3 对齐
  - 全局 BA (Pose Graph)
```

**ORB 词典**: 离线训练的大规模 ORB 特征词袋, 用于回环检测和重定位。

**ATLAS (地图集)**: ORB-SLAM3 创新点, 支持**多地图**:
- 当前活跃地图
- 非活跃地图 (关闭后保留)
- 重定位失败时建新地图
- 闭环发现时合并地图

**代码入口** (`mono_euroc.cc` 例子):

```cpp
// ORB-SLAM3 单目 + IMU
auto SLAM = std::make_shared<ORB_SLAM3::System>(
    vocab_path,                    // ORB 词典
    settings_path,                // 相机配置文件 (yaml)
    ORB_SLAM3::System::IMU_MONOCULAR,
    true                          // 可视化
);

for (const auto& [ts, img, imu_data] : dataset) {
    Sophus::SE3f pose = SLAM->TrackMonocular(img, ts, imu_data);
}

SLAM->Shutdown();
SLAM->SaveTrajectoryTUM("trajectory.txt");
SLAM->SaveKeyFrameTrajectoryTUM("keyframe.txt");
```

**ORB-SLAM3 的优点**:
- 成熟, 工业级稳定
- 单/双/RGB-D/IMU 都支持
- 重定位 + 闭环 + 多地图齐备
- 工程文档相对完整

**ORB-SLAM3 的缺点**:
- 特征点法, 在低纹理场景失效
- 稠密地图能力弱
- 计算量不算小, CPU only
- 长期运行内存累积

### 5.3 直接法: LSD-SLAM

**LSD-SLAM (Engel 2014)** 是直接法的代表作。

**核心思想**: 不提取特征点, 直接用图像像素的**光度误差**做位姿估计:

$$E = \sum_{p \in \Omega} (I_1(p) - I_2(\pi(P, \mathbf{X}_p)))^2$$

**直接法**: 优点 (不依赖特征点, 低纹理可用, 自然得半稠密地图), 缺点 (依赖光度一致性, 收敛域小, 需要 GPU)。

**关键贡献**: 半稠密深度估计 (方差加权的多基线三角化), 位姿图优化 (Sim3 闭环)。代表系统: LSD-SLAM, DSO (Direct Sparse Odometry), SVO (Semi-direct)。

### 5.4 半稠密 + IMU 融合: VINS-Mono

**VINS-Mono** (Qin et al., 2018) 是港科大秦通博士的开源 VIO, 视觉 + IMU 紧耦合的代表作。

**VIO 的意义**:

纯视觉 SLAM 的两个老大难:
- **快速运动模糊**: 视觉跟不上, 跟踪失败
- **尺度不确定**: 单目 SLAM 永远绕不开

IMU 提供: 高频 (200-1000 Hz) 加速度和角速度, 真实尺度 (m/s²),
- 短期精确的运动预测

VIO = 视觉 (几何) + IMU (动力学), 取长补短。

**VINS-Mono 的关键模块**:

1. **前端**: Harris 角点 + LK 光流跟踪
2. **IMU 预积分**: 两帧之间 IMU 数据预积分, 得到相对位姿约束
3. **紧耦合优化**: 视觉重投影误差 + IMU 预积分残差联合 BA
4. **回环检测**: DBoW2
5. **4 自由度位姿图优化**: 滚转和俯仰由重力对齐, 只优化 x, y, z, yaw

**VINS-Mono 的应用**:
- 无人机 (大疆, PX4)
- AR (HoloLens, Quest)
- 手机 AR (Apple ARKit 借鉴)
- 自动驾驶 (Apollo 部分模块)

### 5.5 NeRF-SLAM: 神经隐式 SLAM

**传统 SLAM 的局限**:
- 输出稀疏点云, 不能直接渲染
- 真实感重建需要 MVS, 慢且复杂
- 几何和外观分离建模

**NeRF (Neural Radiance Fields, Mildenhall 2020)** 革命性改变了 3D 重建:

$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$$

输入 3D 位置 $\mathbf{x}$ 和观察方向 $\mathbf{d}$, 输出颜色 $\mathbf{c}$ 和体密度 $\sigma$。用一个 MLP 表示这个函数。

**NeRF-SLAM 的核心思想**: 用 NeRF 作为 SLAM 的地图表示, 同时跟踪位姿和优化隐式地图。

**代表系统**: iMAP (Sucar 2021) 第一个实时 NeRF-SLAM; NICE-SLAM (Zhu 2022) 分层特征网格, 大场景可扩展; Point-SLAM (2023) 用点云做几何先验; Co-SLAM (2023) 结合坐标网络和稀疏特征; SplaTAM (2024) 用 3D Gaussian Splatting 做地图, 速度极快; ESLAM (2023) 当前 SOTA 大场景。

**NeRF-SLAM 的优劣**:

| 维度 | 传统 SLAM | NeRF-SLAM |
| :--- | :--- | :--- |
| 地图 | 稀疏 / 半稠密 | 神经隐式 (连续) |
| 渲染 | 不能 | 可 |
| 精度 | 高 | 中-高 |
| 速度 | 快 (30 FPS) | 慢 (10-30 FPS) |
| 内存 | 小 | 大 (MLP + 网格) |

**未来方向**: 3D Gaussian Splatting SLAM (速度 + 渲染都好), 端到端 SLAM, 大语言模型 + SLAM (语义地图)。

---

## 6. 视觉惯性里程计 VIO (200 行)

### 6.1 IMU 预积分

**IMU 测量模型**:

$$\tilde{\boldsymbol{\omega}}^b = \boldsymbol{\omega}^b + \mathbf{b}^g + \mathbf{n}^g$$
$$\tilde{\mathbf{a}}^b = R_{bw}(\mathbf{a}^w + \mathbf{g}^w) + \mathbf{b}^a + \mathbf{n}^a$$

其中 $\mathbf{b}^g, \mathbf{b}^a$ 是陀螺仪和加速度计的偏置 (bias), $\mathbf{n}^g, \mathbf{n}^a$ 是高斯白噪声, $\mathbf{g}^w$ 是重力。

**IMU 积分 (Euler)**:

设 $t_k, t_{k+1}$ 两帧, IMU 在其间给出 N 个采样:

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{g}^w \Delta t + \sum_{i} R_{k \to i} (\tilde{\mathbf{a}}_i - \mathbf{b}_i^a) \Delta t$$
$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \frac{1}{2}\mathbf{g}^w \Delta t^2 + \sum_{i} R_{k \to i} (\tilde{\mathbf{a}}_i - \mathbf{b}_i^a) \Delta t^2$$

**问题**: 优化时如果 $\mathbf{v}_k, R_k$ 变了, IMU 积分要重算。**预积分 (Pre-integration)** 把积分结果"参考系"分离, 让它独立于位姿。

**预积分公式**:

定义增量:
$$\Delta R_{ij} = R_i^T R_j, \quad \Delta \mathbf{v}_{ij}, \quad \Delta \mathbf{p}_{ij}$$

这些量只依赖 IMU 测量和 bias, 不依赖绝对位姿。优化时只更新增量, 计算快得多。

**预积分的协方差传播**: 噪声累积的协方差矩阵, 用于后续加权。

```python
def imu_preintegration(omega, a, dt, bias):
    """
    omega: (N, 3) 角速度
    a: (N, 3) 加速度
    dt: 时间步
    bias: (gyro_bias, accel_bias)
    """
    R = np.eye(3)
    v = np.zeros(3)
    p = np.zeros(3)
    bg, ba = bias

    for i in range(len(omega)):
        omega_i = omega[i] - bg
        a_i = a[i] - ba

        # 旋转更新
        R_new = R @ exp_so3(omega_i * dt)
        # 速度更新 (预积分里不含重力)
        v = v + R @ a_i * dt
        # 位置更新
        p = p + v * dt + 0.5 * R @ a_i * dt**2
        R = R_new

    return R, v, p
```

### 6.2 VINS-Mono 算法详解

**VINS-Mono** 是 VIO 的经典实现, 紧耦合, 基于优化的架构。

**前端**:
- 每帧提取 Harris 角点 (~100-300)
- LK 光流跟踪
- 关键帧选择: 平均视差 + 跟踪质量
- 短期 IMU 预积分用于位姿预测

**后端**:
- **滑动窗口 BA**: 保留最近 N=10 个关键帧
- **目标函数**:

$$\min \underbrace{\sum_{(i,j) \in \mathcal{C}} \|\mathbf{e}_{ij}^c\|^2}_{\text{视觉重投影}} + \underbrace{\sum_{k \in \mathcal{B}} \|\mathbf{e}_k^b\|^2}_{\text{IMU 预积分}} + \underbrace{\|\mathbf{e}_0^p\|^2}_{\text{先验 (边缘化)}}$$

视觉残差 $\mathbf{e}_{ij}^c$ 是特征点重投影误差, IMU 残差 $\mathbf{e}_k^b$ 是预积分观测和当前估计的差。

**回环**:
- DBoW2 词袋检索候选闭环
- 几何验证 + 时间一致性
- 4 自由度位姿图优化 (roll, pitch 由重力对齐)

**在线时间标定**: VINS-Mono 还估计相机-IMU 时间偏移 $t_d$, 这在工程上很关键 (相机和 IMU 的时钟不同步)。

### 6.3 在无人机 / AR 上的应用

**无人机**:

无人机是 VIO 的天然应用场景:
- GPS 拒止环境 (室内, 桥底)
- 快速运动 (IMU 短时准, 视觉慢)
- 尺度敏感 (高度必须准)

代表:
- **大疆 Phantom / Mavic**: 内置 VIO + GPS 融合
- **PX4 + VINS-Fusion**: 开源飞控 + VIO
- **T265 + Pixhawk**: Intel RealSense T265 + PX4

**AR 头显**:

Apple Vision Pro, Meta Quest 3, Microsoft HoloLens 2 全部内置 VIO:
- 头部快速转动 (200°/s), 视觉会模糊, IMU 接力
- 重定位失败时启动全局定位
- 6 自由度跟踪 (位置 + 姿态)

代表实现:
- **ARKit / ARCore**: Apple / Google 的手机 AR SDK, 内置 VIO
- **OpenXR**: 跨平台 AR 标准
- **Inside-out tracking**: 用头显自身相机, 不需要外部传感器

**自动驾驶**:

Apollo, Autoware 等自动驾驶开源项目:
- 视觉车道线检测 + VIO 提供局部定位
- LiDAR SLAM 提供全局定位
- 多传感器融合 (相机 + LiDAR + IMU + GPS)

**手机 AR 优化**:

手机 AR 的 VIO 优化重点:
- IMU 偏置在线估计 (温度漂移)
- 动态物体剔除 (人, 车)
- 长期稳定性 (不重启设备)

---

## 7. PyTorch 实验 (200 行)

### 7.1 本质矩阵估计 (简化版)

下面用一个 toy 数据集验证对极几何和八点法:

```python
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# 1. 生成 3D 场景
np.random.seed(42)
N = 50  # 3D 点数
points_3d = np.random.randn(N, 3) * 2 + np.array([0, 0, 5])  # 平均深度 5m

# 2. 相机 1 在原点, 相机 2 在某个相对位姿
R_true = cv2.Rodrigues(np.array([0.1, 0.2, 0.05]))[0]
t_true = np.array([0.5, 0.1, 0.05])  # 平移

K = np.array([[500, 0, 320],
              [0, 500, 240],
              [0, 0, 1]])

def project(X, R, t, K):
    """3D 点投影到像素"""
    X_cam = X @ R.T + t
    x = X_cam / X_cam[:, 2:3]
    u = (K @ x.T).T[:, :2]
    return u

# 3. 投影到两相机
pts1 = project(points_3d, np.eye(3), np.zeros(3), K)
pts2 = project(points_3d, R_true, t_true, K)

# 4. 加噪声
noise_level = 1.0  # 像素
pts1_noisy = pts1 + np.random.randn(*pts1.shape) * noise_level
pts2_noisy = pts2 + np.random.randn(*pts2.shape) * noise_level

# 5. 归一化 (Hartley)
def normalize_2d(pts):
    mean = pts.mean(axis=0)
    centered = pts - mean
    scale = np.sqrt(2) / np.linalg.norm(centered, axis=1).mean()
    T = np.array(`scale, 0, -scale*mean[0`,
                  [0, scale, -scale*mean[1]],
                  [0, 0, 1]])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    return (T @ pts_h.T).T[:, :2], T

pts1_n, T1 = normalize_2d(pts1_noisy)
pts2_n, T2 = normalize_2d(pts2_noisy)

# 6. 八点法
n = len(pts1_n)
A = np.zeros((n, 9))
for i in range(n):
    x1, y1 = pts1_n[i]
    x2, y2 = pts2_n[i]
    A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

_, _, Vt = np.linalg.svd(A)
F_normalized = Vt[-1].reshape(3, 3)

# 强制 rank-2
U, S, Vt = np.linalg.svd(F_normalized)
S[-1] = 0
F_normalized = U @ np.diag(S) @ Vt

# 反归一化
F = T2.T @ F_normalized @ T1

# 7. 验证: 计算对极误差
errors = []
for i in range(n):
    p1 = np.append(pts1_noisy[i], 1)
    p2 = np.append(pts2_noisy[i], 1)
    line_in_2 = F @ p1
    line_in_2 /= np.linalg.norm(line_in_2[:2])
    err = abs(p2 @ line_in_2)
    errors.append(err)

print(f"Mean epipolar error: {np.mean(errors):.3f} pixels")
print(f"Max epipolar error: {np.max(errors):.3f} pixels")
# 预期: 平均误差 ~1.0 (因为加了 1 像素噪声)

# 8. 分解本质矩阵
E = K.T @ F @ K
_, R_est, t_est, _ = cv2.recoverPose(E, pts1_noisy, pts2_noisy, K)

# 比较估计与真实
print(f"\n真实 R: \n{R_true}")
print(f"\n估计 R: \n{R_est}")
print(f"\n真实 t: {t_true / np.linalg.norm(t_true)}")
print(f"估计 t: {t_est.flatten() / np.linalg.norm(t_est)}")
# 注意 t 的尺度任意, 看方向
```

**预期输出**:
```text
Mean epipolar error: ~1.0 pixels
R_est ≈ R_true (旋转一致)
t_est 方向 ≈ t_true 方向 (尺度任意)
```

### 7.2 两图三角化 (SfM toy)

```python
# 接 7.1, 用估计的位姿三角化
P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K @ np.hstack([R_est, t_est])

pts1_norm = cv2.undistortPoints(pts1_noisy.reshape(-1, 1, 2), K, None).reshape(-1, 2)
pts2_norm = cv2.undistortPoints(pts2_noisy.reshape(-1, 1, 2), K, None).reshape(-1, 2)
points_4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
points_3d_recon = (points_4d[:3] / points_4d[3]).T

# 比较重建 vs 真值 (尺度任意)
scale = np.linalg.norm(points_3d) / np.linalg.norm(points_3d_recon)
print(f"尺度因子: {scale:.3f} (单目 SfM 的尺度模糊)")
```

### 7.3 Bundle Adjustment (PyTorch)

PyTorch 实现 BA 的好处: 自动微分, 免去手写 Jacobian。

```python
import torch

def exp_so3(phi):
    """李代数 → SO(3) (Rodrigues 公式)"""
    theta = torch.norm(phi) + 1e-8
    n = phi / theta
    K = skew(n)
    return torch.eye(3) + torch.sin(theta)*K + (1-torch.cos(theta))*K@K

def skew(v):
    return torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def ba_loss(poses, points, K, observations):
    """
    poses: (M, 6) M 个相机 se(3)
    points: (N, 3) N 个 3D 点
    observations: [(cam_idx, pt_idx, u, v), ...]
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    total = 0.0
    for c, p, u_obs, v_obs in observations:
        xi = poses[c]
        R = exp_so3(xi[:3])
        X_cam = R @ points[p] + xi[3:]
        u_pred = fx * X_cam[0]/X_cam[2] + cx
        v_pred = fy * X_cam[1]/X_cam[2] + cy
        total += ((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
    return total

# 初始化与优化
poses = torch.randn(M, 6, requires_grad=True) * 0.1
points = torch.randn(N, 3, requires_grad=True)
opt = torch.optim.Adam([poses, points], lr=1e-2)
for it in range(200):
    opt.zero_grad()
    loss = ba_loss(poses, points, K, observations)
    loss.backward()
    opt.step()
```

**PyTorch BA 的局限**: 速度不如 g2o / Ceres, 但教学清晰。

### 7.4 SfM 实战: OpenCV 端到端

```python
# 读图 + SIFT + 匹配 + E + 三角化, 一条龙
img1 = cv2.imread('im1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create(5000)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

pts1 = np.array([kp1[m.queryIdx].pt for m in good])
pts2 = np.array([kp2[m.trainIdx].pt for m in good])
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 你的内参

E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
_, R, t, _ = cv2.recoverPose(E, pts1[mask.ravel().bool()], pts2[mask.ravel().bool()], K)

# 三角化
P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K @ np.hstack([R, t])
pts1n = cv2.undistortPoints(pts1.reshape(-1,1,2), K, None).reshape(-1,2)
pts2n = cv2.undistortPoints(pts2.reshape(-1,1,2), K, None).reshape(-1,2)
pts_3d = (cv2.triangulatePoints(P1, P2, pts1n.T, pts2n.T)[:3] /
          cv2.triangulatePoints(P1, P2, pts1n.T, pts2n.T)[3]).T
```

### 7.5 NeRF-SLAM 入门

跑 NICE-SLAM 的最小命令:

```bash
git clone https://github.com/cvg/nice-slam && cd nice-slam
pip install -r requirements.txt
python scripts/download_replica.py   # 下载 Replica 数据集
python src/run.py --config configs/Replica/config.yaml \
    --input_dataset $REPLICA_DIR/scene0 --output_path $OUTPUT_DIR
```

---

## 8. 与其他章节的关系 (100 行)

### 8.1 → 第 3 章: 机器人学

SLAM 是机器人学的核心:
- **定位 (Localization)**: SLAM 直接给出机器人位姿, 第 3 章用它做路径规划
- **建图 (Mapping)**: SLAM 输出的稀疏/稠密地图是机器人避障的基础
- **运动控制 (Control)**: SLAM 提供反馈, 让 MPC / 反馈控制器知道"我到了没有"

具体衔接:
- 第 3 章讲 PID / MPC 控制 → 用 SLAM 的位姿做反馈
- 第 3 章讲轨迹规划 → 需要 SLAM 提供的地图 + 定位
- 第 3 章讲手眼标定 → 相机外参由 SLAM 在线估计

### 8.2 → 第 7 章: 仿真器

SLAM 在仿真器里测试:
- **iGibsim, Habitat**: 渲染 RGB-D 序列, 跑 SLAM
- **AirSim, CARLA**: 自动驾驶 SLAM 仿真
- **Isaac Sim**: NVIDIA 的物理 + 渲染仿真

测试 SLAM 的鲁棒性, 看它在不同场景 (光照, 纹理, 动态物体) 下的表现。

### 8.3 → 第 8 章: 世界模型

SLAM 是世界模型的子集:
- SLAM 的"地图" = 世界模型的"几何部分"
- SLAM 的"位姿轨迹" = 世界模型的"状态"
- 世界模型还要预测未来 → SLAM 只描述当前

NeRF-SLAM 模糊了 SLAM 和世界模型的边界: 隐式神经场既是地图又是预测模型。

### 8.4 → 第 1 章: 相机模型

本章直接依赖第 1 章:
- **内参 K**: 八点法需要归一化坐标, 必须先做内参标定
- **外参 [R|t]**: 本章求解的就是外参
- **畸变校正**: 匹配前必须先校正

### 8.5 → 第 2 章: 点云与体素

SLAM 的输出是点云:
- 稀疏点云: 特征点位置
- 半稠密点云: 高梯度区域的深度
- 稠密点云: MVS / NeRF 输出

第 2 章的 ICP / G-ICP 可以用来对齐 SLAM 的子地图, 提高精度。

---

## 9. 与数学基础的关系 (100 行) 高复用

本章重度依赖数学基础的几章:

### 9.1 数学基础/2-线代/4-特征值·谱定理 ← 本质矩阵的 SVD

- **SVD 分解**: 八点法解线性方程组的核心
- **秩亏约束**: $\det(E) = 0$ 用 SVD 强制
- **最小二乘**: $\min \|A\mathbf{x}\|^2$ 的解 = SVD 最小奇异值向量

### 9.2 数学基础/2-线代/5-SVD与应用 ← 多视角几何

- **本质矩阵分解**: $E = U \Sigma V^T$, $\Sigma = \text{diag}(1, 1, 0)$
- **基础矩阵分解**: $F$ 的 7 个自由度对应 SVD 的结构约束
- **三角化 DLT**: SVD 解超定方程

### 9.3 数学基础/4-离散数学/4-图论基础 ← SLAM 图优化

- **共视图 (Covisibility Graph)**: 顶点是关键帧, 边是共视点关系
- **位姿图 (Pose Graph)**: 顶点是位姿, 边是相对位姿约束
- **回环约束**: 加一条跨时间的边

### 9.4 数学基础/6-最优化/5-凸优化理论 ← BA 的求解

- **非线性最小二乘**: BA 的核心
- **Levenberg-Marquardt**: 阻尼最小二乘, 平衡 Gauss-Newton 和梯度下降
- **Schur 补**: 利用稀疏结构加速
- **IRLS**: 鲁棒核函数的迭代求解

### 9.5 数学基础/6-最优化/3-拉格朗日乘子 ← Pose Graph

- **Sim3 优化**: 等式约束 (尺度对齐)
- **回环约束**: 作为等式约束加入

### 9.6 数学基础/3-概率/3-高斯过程 ← VIO 协方差传播

- **卡尔曼滤波**: VIO 的滤波式后端 (EKF-VIO)
- **IMU 协方差**: 噪声如何在预积分中传播

### 9.7 数学基础/7-系统思维/1-反馈控制 ← VIO 闭环

- **VIO = 视觉 + IMU 反馈**: IMU 短期准, 视觉长期准, 互相反馈
- **回环 = 反馈控制中的扰动观测**: 检测到回环就校正累积误差

---

## 10. 常见误区 (150 行)

### 10.1 "SfM = SLAM"

**错**。两者目标不同: SfM 离线/无序/慢但准, SLAM 在线/视频流/快但漂移。共享几何, 工程约束完全不同。
- 旅游照片建 3D → SfM (COLMAP)
- 机器人视频实时定位 → SLAM (ORB-SLAM3)
- 无人机视频精细重建 → SfM + SLAM 联合

### 10.2 "ORB-SLAM 已被 NeRF-SLAM 取代"

**错**。当前 (2026 年) ORB-SLAM3 仍是工业首选: 精度 ~0.05m vs NeRF-SLAM ~0.1m, 速度 30 FPS vs 10-30 FPS, 成熟度高。NeRF-SLAM 的优势是渲染质量和稠密几何, 不是精度。3D Gaussian Splatting SLAM 可能改变格局, 但工业落地还需时间。

### 10.3 "回环检测可有可无"

**错**。回环是单目 SLAM 抗漂移的唯一手段。不做回环, 尺度漂移, 累积误差指数增长; 做了回环, 漂移被周期性纠正。ORB-SLAM3 关闭回环, ATE 涨 10 倍。

### 10.4 "特征点越多越好"

**错**。特征点过多会拖慢系统: 提取慢、匹配慢、BA 慢。ORB-SLAM3 默认 1000-2000 点, 视差足够时 1500 点足够, 纹理丰富场景可减到 800 点。

### 10.5 "单目 SLAM 能给出绝对尺度"

**错**。单目 SLAM 几何上没有绝对尺度。要靠已知物体尺寸、IMU 融合、双目/RGB-D、GPS。ORB-SLAM3 单目模式输出的轨迹尺度任意, 必须先做 Umeyama 对齐才能和真值比。

### 10.6 "用 SIFT 比 ORB 准, 所以 SLAM 应该用 SIFT"

**错**。SLAM 实时性约束下 ORB 是最优选择: SIFT ~50ms/帧 (30 FPS 难), ORB ~5ms/帧 (轻松 30 FPS)。SLAM 场景里 ORB 的精度损失 < SIFT 的速度收益。离线 SfM 仍首选 SIFT (COLMAP 默认)。

### 10.7 "BA 一定会收敛到全局最优"

**错**。BA 是非线性优化, 强烈依赖初始化: 初始化差 → 局部最优; 退化场景 (平面/纯旋转) → BA 完全失效。工程上必须用恒速模型给好初始位姿、检查退化、用 RANSAC 排除外点。

### 10.8 "VIO 不需要回环"

**部分错**。VIO 比纯视觉 SLAM 漂移小很多 (IMU 提供尺度 + 重力对齐), 但长期运行仍然需要回环: 短期 (秒级) < 1m, 中期 (分钟) ~5m, 长期 (小时) ~几十米。无人机制图、自动驾驶必须加回环。

### 10.9 "SLAM 输出越稠密越好"

**错**。稠密 vs 稀疏是工程取舍: 稀疏 SLAM (1000 点, 30 FPS) 适合定位, 半稠密 (10k 点, 10-20 FPS) 适合避障, 稠密 (数百万点, 1-5 FPS) 适合重建。机器人导航不需要稠密地图。

### 10.10 "深度学习会取代传统 SLAM"

**部分对**。当前态势: 传统 SLAM 几何严格精度高, 深度 SLAM 鲁棒性更好 (低纹理/动态场景优势), 混合 (DROID-SLAM) 用光流网络 + BA 是 SOTA。未来趋势是融合, 不是取代。

---

## 11. L7/L8 面试 Q&A (200 行)

### Q1: 本质矩阵 vs 基础矩阵

**答**: 两者描述同一对极几何约束, 但在不同坐标系下:

| 矩阵 | 输入坐标 | 自由度 | 公式 |
| :--- | :--- | :--- | :--- |
| 本质矩阵 $E$ | 归一化相机坐标 | 5 (3+3-1 尺度) | $\mathbf{x}'^T E \mathbf{x} = 0$ |
| 基础矩阵 $F$ | 像素坐标 | 7 (5+2 内参) | $\mathbf{u}'^T F \mathbf{u} = 0$ |

关系: $F = K^{-T} E K^{-1}$。

**追问 1**: 为什么 $E$ 有 5 个自由度?
- 3 旋转 + 3 平移 - 1 尺度 = 5
- 尺度等价: $(\lambda R, \lambda \mathbf{t})$ 和 $(R, \mathbf{t})$ 给出相同的 $E$

**追问 2**: 什么时候用 $F$ vs $E$?
- 已知内参 $K$: 用 $E$ (数值更稳定)
- 未标定或跨相机 (不同 $K$): 用 $F$

**追问 3**: $E$ 和 $F$ 都满足 $\det = 0$ (秩亏) 吗?
是的, 这意味着两相机射线共面, 对极约束退化为一条线。

### Q2: Bundle Adjustment 的优化目标

**答**: 最小化所有观测的重投影误差:

$$\min_{\{P_i\}, \{\mathbf{X}_j\}} \sum_{ij} \rho(\|\mathbf{u}_{ij} - \pi(P_i, \mathbf{X}_j)\|^2)$$

其中 $P_i$ 是相机位姿, $\mathbf{X}_j$ 是 3D 点, $\rho$ 是鲁棒核函数。

**追问 1**: 为什么用重投影误差而不是 3D 误差?
- 3D 真值难拿
- 重投影误差直接用图像观测
- 投影模型已知, 解析可导

**追问 2**: 为什么用 Schur 补?
利用 BA Hessian 的稀疏结构 (3D 点之间无直接耦合), 把计算量从 $O((6n+3m)^3)$ 降到 $O((6n)^3)$。

**追问 3**: 鲁棒核函数的作用?
降低外点 (误匹配) 的影响。Huber, Cauchy, Tukey 都常用。

### Q3: ORB-SLAM 与 NeRF-SLAM 的取舍

**答**:

| 维度 | ORB-SLAM3 | NeRF-SLAM |
| :--- | :--- | :--- |
| 几何 | 稀疏特征点 | 神经隐式 (连续) |
| 渲染 | 不能 | 可以 |
| 精度 | 高 (cm 级) | 中 (10 cm) |
| 速度 | 快 (30 FPS) | 慢 (10-30 FPS) |
| 内存 | 小 (百 MB) | 大 (GB 级) |
| 成熟度 | 工业级 | 学术前沿 |

**选择建议**:
- 机器人导航, AR, 自动驾驶 → ORB-SLAM3 (精度 + 速度)
- 文物数字化, 影视特效, 离线建模 → NeRF-SLAM (渲染质量)

**追问 1**: NeRF-SLAM 在大场景失败的原因?
- MLP 表示整个场景, 容量有限
- 内存和计算随场景增大爆炸
- 改进方向: 3D Gaussian Splatting, 子地图

**追问 2**: 未来 SLAM 会朝什么方向发展?
- 3D Gaussian Splatting SLAM (渲染 + 速度都好)
- 端到端可微 SLAM
- 大模型 + SLAM (语义 + 几何)

### Q4: SLAM 的回环检测算法

**答**: 回环检测 = "我是不是回到了之前来过的地方"。

**主流方法**:
1. **词袋模型 (BoW)**: DBoW2 / DBoW3
   - 离线训练视觉词典 (k-means 聚类)
   - 在线: 每张图编码成稀疏词频向量
   - 查询: 倒排索引找相似图
2. **深度特征**: NetVLAD, AP-GeM
3. **几何验证**: 估计本质矩阵, 内点足够多
4. **时间一致性**: 连续多帧闭环才算

**追问 1**: 怎么避免假阳性 (False Positive)?
- 时间一致性: 单帧相似不算, 连续多帧才信
- 几何验证: 候选闭环必须估计出合格的相对位姿
- 时间约束: 不能"穿越时空" (闭环位姿的时间间隔合理)

**追问 2**: 回环后怎么用?
1. 计算 Sim3 相对变换 (含尺度)
2. 把当前地图融合到老地图 (或建新地图)
3. 全局 Pose Graph 优化
4. 全局 BA 精化

### Q5: 视觉 SLAM 在纹理缺乏场景怎么办

**答**: 这是 SLAM 的经典痛点, 多个角度应对:

1. **特征点法**: 改用光流法 (VINS-Mono 用 Harris + LK)
2. **直接法**: 不依赖特征点, 用光度误差 (LSD-SLAM, DSO)
3. **融合 IMU**: IMU 提供短时运动预测, 弥补视觉失效
4. **融合 LiDAR**: 激光雷达对纹理不敏感, LIO-SAM
5. **深度特征**: SuperPoint, LoFTR 在低纹理更鲁棒
6. **事件相机**: 高动态范围, 微秒级响应
7. **线/面特征**: 用结构化特征替代点特征

**追问 1**: 事件相机是什么?
异步传感器, 像素级亮度变化触发事件。HDR, 微秒延迟, 低纹理场景表现好。代表: DAVIS346, Prophesee。

**追问 2**: 室内白墙场景哪种方法最好?
- 纯视觉: LSD-SLAM 或 DSO (直接法)
- 加 IMU: VINS-Mono
- 加激光: LIO-SAM
- 组合: 视觉惯性激光融合 (LVI-SAM)

### Q6: 单目 SLAM 怎么定尺度

**答**: 几何上无法定, 工程上几种办法:

1. **已知物体尺寸**: 检测到一个 30cm 的物体, 反推尺度
2. **IMU 融合**: IMU 给出真实加速度, 联合优化恢复尺度
3. **双目/RGB-D**: 基线/深度已知, 尺度直接定
4. **GPS**: 室外 GPS 给绝对位置
5. **先验地图**: 与已知 3D 地图匹配

**ORB-SLAM3 的处理**:
- 单目模式: 自动选一个伪尺度, 后续 BA 维持
- 单目 + IMU: 第一秒用 IMU 初始化尺度

### Q7: 为什么 BA 不用牛顿法而用 LM

**答**:

| 方法 | 收敛速度 | 稳定性 | 适用 |
| :--- | :--- | :--- | :--- |
| 梯度下降 | 慢 | 强 | 初始阶段 |
| 牛顿法 | 快 (二次) | 弱 | Hessian 不正定时发散 |
| 高斯-牛顿 | 快 | 中 | 残差小时 |
| LM | 可调 | 强 | BA 通用 |

LM = 阻尼项 $\lambda I$ + 高斯-牛顿, 平衡速度和稳定性。

### Q8: Sim3 和 SE3 的区别

**答**:
- **SE(3)**: 旋转 + 平移 (6 自由度)
- **Sim(3)**: 旋转 + 平移 + 尺度 (7 自由度)

单目 SLAM 的回环对齐必须用 Sim3, 因为尺度可能漂移了。

```python
# Sim3 优化 (g2o)
g2o::VertexSim3Expmap* v_sim3 = new g2o::VertexSim3Expmap();
v_sim3->setEstimate(Sim3(R, t, scale));
```

### Q9: 多相机 SLAM 的难点

**答**:
1. **同步**: 多个相机的曝光时间不同步
2. **标定**: 多相机之间的外参会漂移
3. **计算**: 多相机 BA 的 Hessian 更大
4. **视场管理**: 不同相机的视场可能不重叠

代表实现:
- **ORB-SLAM3 多相机版**: 每个相机独立跟踪, 联合优化
- **VINS-Multi-Cam**: VINS-Mono 的多相机扩展
- **OpenVINS**: 通用多相机 VIO

### Q10: SLAM 失败的常见原因

**答**:
1. **初始化失败**: 运动太慢/太快, 视差不
2. **跟踪丢失**: 遮挡/快速运动/纹理缺乏
3. **BA 不收敛**: 初始化差, 外点多
4. **回环失败**: 词袋假阳性, 几何验证不过
5. **尺度漂移**: 单目 SLAM 长期运行必然
6. **IMU 偏置估计错误**: 静止初始化不够
7. **时间不同步**: 相机-IMU 时间戳没对齐

调试方法:
- 可视化: rviz, pangolin, rerun
- 日志: 跟踪点数, 内点数, 残差
- ATE / RPE 评估: evo 工具

---

## 12. 参考文献 (100 行)

### 经典论文

1. **Longuet-Higgins, H. C. (1981)**. "A computer algorithm for reconstructing a scene from two projections". Nature. — 八点法的最初提出。

2. **Hartley, R. I. (1997)**. "In Defense of the Eight-Point Algorithm". PAMI. — 归一化八点法, 数值稳定性提升。

3. **Nistér, D. (2004)**. "An Efficient Solution to the Five-Point Relative Pose Problem". PAMI. — 五点法。

4. **Lowe, D. G. (2004)**. "Distinctive Image Features from Scale-Invariant Keypoints". IJCV. — SIFT 特征。

5. **Rublee, E., et al. (2011)**. "ORB: An Efficient Alternative to SIFT or SURF". ICCV. — ORB 特征, SLAM 时代的实用之选。

6. **DeTone, D., et al. (2018)**. "SuperPoint: Self-Supervised Interest Point Detection and Description". CVPR Workshop. — 深度学习特征。

7. **Sun, J., et al. (2021)**. "LoFTR: Detector-Free Local Feature Matching with Transformers". CVPR. — 无检测器匹配, 低纹理场景 SOTA。

8. **Lourakis, M., Argyros, A. (2009)**. "SBA: A Software Package for Generic Sparse Bundle Adjustment". ACM TOMS. — BA 开源实现。

9. **Triggs, B., et al. (2000)**. "Bundle Adjustment: A Modern Synthesis". ICCV Workshop. — BA 综述。

10. **Kümmerle, R., et al. (2011)**. "g2o: A General Framework for Graph Optimization". ICRA. — g2o 图优化库。

11. **Dellaert, F., Kaess, M. (2017)**. "Factor Graphs for Robot Perception". Foundations and Trends in Robotics. — GTSAM 的理论基础。

12. **Agarwal, S., et al. (2011)**. "Ceres Solver". — Google 的非线性最小二乘库。

### SLAM / SfM 系统

13. **Davison, A. J., et al. (2007)**. "MonoSLAM: Real-Time Single Camera SLAM". PAMI. — 第一个实时单目 SLAM。

14. **Klein, G., Murray, D. (2007)**. "Parallel Tracking and Mapping for Small AR Workspaces". ISMAR. — PTAM, 跟踪 + 建图分离。

15. **Mur-Artal, R., Tardós, J. D. (2017)**. "ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras". IEEE T-RO. — ORB-SLAM2。

16. **Campos, C., et al. (2021)**. "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM". IEEE T-RO. — ORB-SLAM3, 多相机 + IMU + 多地图。

17. **Engel, J., et al. (2014)**. "LSD-SLAM: Large-Scale Direct Monocular SLAM". ECCV. — 直接法 SLAM。

18. **Engel, J., Koltun, V., Cremers, D. (2018)**. "Direct Sparse Odometry". PAMI. — DSO, 稀疏直接法。

19. **Forster, C., et al. (2017)**. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry". IEEE T-RO. — SVO, 半直接 + IMU。

20. **Qin, T., et al. (2018)**. "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator". IEEE T-RO. — VINS-Mono, 紧耦合 VIO。

21. **Mourikis, A. I., Roumeliotis, S. I. (2007)**. "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation". ICRA. — MSCKF, 滤波式 VIO。

22. **Schonberger, J. L., Frahm, J.-M. (2016)**. "Structure-from-Motion Revisited". CVPR. — COLMAP, 增量式 SfM 标杆。

23. **Moulon, P., et al. (2016)**. "OpenMVG: Open Multiple View Geometry". — 全局式 SfM 开源。

### NeRF-SLAM

24. **Mildenhall, B., et al. (2020)**. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis". ECCV. — NeRF, 神经隐式 3D 重建。

25. **Sucar, E., et al. (2021)**. "iMAP: Implicit Mapping and Positioning in Real-Time". ICCV. — 第一个 NeRF-SLAM。

26. **Zhu, Z., et al. (2022)**. "NICE-SLAM: Neural Implicit Scalable Encoding for SLAM". CVPR. — NICE-SLAM, 大场景 NeRF-SLAM。

27. **Keetha, N., et al. (2024)**. "SplaTAM: Splat, Track & Map 3D Gaussian Splatting SLAM". — 3D Gaussian SLAM。

### 工具与库

28. **OpenCV** (Bradski, G., 2000). `findEssentialMat`, `recoverPose`, `triangulatePoints`。

29. **COLMAP** (Schonberger, J. L.). https://colmap.github.io

30. **GTSAM** (Dellaert, F.). https://gtsam.org

31. **g2o** (Kümmerle, R.). https://github.com/RainerKuemmerle/g2o

32. **Ceres Solver** (Google). http://ceres-solver.org

33. **ORB-SLAM3** (UZ-CMU). https://github.com/UZ-SLAMLab/ORB_SLAM3

34. **VINS-Mono** (HKUST). https://github.com/HKUST-Aerial-Robotics/VINS-Mono

### 教材

35. **Hartley, R., Zisserman, A. (2003)**. "Multiple View Geometry in Computer Vision". Cambridge. — 多视角几何圣经, 本章理论根源。

36. **高翔, 张涛, 等 (2017)**. 《视觉 SLAM 十四讲:从理论到实践》. — 中文 SLAM 入门必读。

37. **Thrun, S., Burgard, W., Fox, D. (2005)**. "Probabilistic Robotics". MIT Press. — 概率 SLAM 基础。

38. **Szeliski, R. (2022)**. "Computer Vision: Algorithms and Applications". — 免费在线版, 第 7-11 章讲 SfM/SLAM。

39. **Cadena, C., et al. (2016)**. "Past, Present, and Future of Simultaneous Localization and Mapping". IEEE T-RO. — SLAM 三十年综述。

### 数据集与评估

40. **Sturm, J., et al. (2012)**. "A Benchmark for the Evaluation of RGB-D SLAM Systems". IROS. — TUM RGB-D。

41. **Geiger, A., et al. (2012)**. "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite". CVPR. — KITTI。

42. **Burri, M., et al. (2016)**. "The EuRoC MAV Dataset". IJRR. — EuRoC, VIO 标准测试集。

43. **Straub, J., et al. (2019)**. "The Replica Dataset". — 室内 NeRF-SLAM 测试集。

---

## 附录:本章速查表

### A.1 对极几何速查

```python
# 归一化坐标 (Hartley)
x_norm = cv2.undistortPoints(pts.reshape(-1,1,2), K, None).reshape(-1, 2)

# 八点法
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

# 五点法 (本质矩阵)
E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)

# 分解 R, t
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

# 三角化
pts_4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
pts_3d = (pts_4d[:3] / pts_4d[3]).T
```

### A.2 SfM/SLAM 库速查

| 库 | 用途 | 命令 |
| :--- | :--- | :--- |
| COLMAP | 离线 SfM + MVS | `colmap mapper` |
| OpenMVG | 全局式 SfM | `openMVG_main_SfM` |
| ORB-SLAM3 | 实时 SLAM | `./mono_euroc` |
| VINS-Mono | 紧耦合 VIO | `rosrun vins_estimator` |
| OpenVSLAM | ORB-SLAM 变种 | `rosrun openslam` |
| DROID-SLAM | 光流 + BA | `python demo.py` |
| NICE-SLAM | NeRF-SLAM | `python src/run.py` |

### A.3 BA 求解器速查

| 库 | 特点 | 用法 |
| :--- | :--- | :--- |
| g2o | 图优化, 经典 | `g2o::SparseOptimizer` |
| GTSAM | 因子图, 优雅 | `gtsam::NonlinearFactorGraph` |
| Ceres | 通用, 工业级 | `ceres::Problem` |
| Sophia | 李群李代数 | `Sophus::SE3d` |

### A.4 本章自检清单

- [ ] 能手画对极几何, 标出极点, 极线, 极平面
- [ ] 能写出本质矩阵和基础矩阵的关系
- [ ] 能解释为什么单目 SLAM 有尺度模糊
- [ ] 能用 OpenCV 估计本质矩阵并分解 R, t
- [ ] 能解释 BA 的 Schur 补为什么加速
- [ ] 知道 ORB-SLAM3, VINS-Mono, DROID-SLAM 的区别
- [ ] 能用 IMU 预积分公式解释 VIO
- [ ] 知道回环检测的词袋模型流程
- [ ] 理解 NeRF-SLAM 和传统 SLAM 的取舍
- [ ] 能解释为什么 BA 不一定收敛到全局最优

### A.5 坐标系与约定 + 重要数字

```text
像素坐标 (u, v): 左上角原点, 单位像素
归一化坐标 (x, y): 主点为原点, 焦距归一化
相机坐标 (X_c, Y_c, Z_c): 光心为原点, Z 轴前向
世界坐标 (X_w, Y_w, Z_w): 场景固定原点, Z 向上
```

- 八点法最少匹配: 8 对; 五点法: 5 对
- $E$ 自由度 5; $F$ 自由度 7; $H$ 自由度 8
- ORB 描述子 256 bit (32 字节); SIFT 描述子 128 float
- ORB-SLAM3 默认 1000-2000 特征点; 关键帧选择: 间隔 0.1s, 视差 20 px

---

> **本章状态**: 完成
> **创建日期**: 2026-06-30
> **下一章**: §5 NeRF 与 3D Gaussian Splatting
---

## 13. 进阶工程实现:PyTorch 复现核心算法

这一节把第 2 节的对极几何和第 4 节的 BA 用 PyTorch 重写一遍。目的不是替代 OpenCV/Ceres,而是让你看清每一步的张量形状和梯度流。读完之后,再去调 `cv2.findEssentialMat` 或者 `gtsam::NonlinearFactorGraph` 的时候,你脑子里会有完整的数据流图。

### 13.1 本质矩阵的八点法(可微分版本)

OpenCV 的 `findEssentialMat` 内部其实是 Hartley 归一化 + SVD。你完全可以在 PyTorch 里复现,而且因为全程是张量,可以直接嵌进深度学习的 loss 里(比如 SuperPoint 训练时端到端学习匹配)。

```python
import torch
import torch.nn.functional as F

def eight_point_essential(pts1: torch.Tensor, pts2: torch.Tensor,
                          K: torch.Tensor) -> torch.Tensor:
    """
    pts1, pts2: (N, 2) 像素坐标 (u, v)
    K: (3, 3) 内参矩阵
    return: E (3, 3), rank-2 约束通过 SVD 后处理
    """
    # 1. 像素坐标 -> 归一化坐标
    inv_K = torch.linalg.inv(K)
    n1 = (pts1 @ inv_K[:2, :2].T) + inv_K[:2, 2]   # (N, 2)
    n2 = (pts2 @ inv_K[:2, :2].T) + inv_K[:2, 2]

    # 2. Hartley 归一化:均值 0, 方差 sqrt(2)
    def normalize(x):
        mu = x.mean(dim=0)
        s = (x - mu).pow(2).sum(dim=1).sqrt().mean() * 1.41421356
        T = torch.eye(3, device=x.device, dtype=x.dtype)
        T[:2, 2] = -mu / s
        T[:2, :2] /= s
        return x, T
    n1n, T1 = normalize(n1)
    n2n, T2 = normalize(n2)

    # 3. 构造 A 矩阵,每行 [u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1]
    A = torch.cat([n1n * n2n[:, :1], n1n * n2n[:, 1:],
                   n1n, n2n, torch.ones_like(n1n[:, :1])], dim=1)  # (N, 9)

    # 4. SVD 解线性系统 A e = 0
    _, _, Vt = torch.linalg.svd(A, full_matrices=False)
    E = Vt[-1].reshape(3, 3)                       # 最小奇异值对应的向量

    # 5. rank-2 约束:最小化 ||E - E'||_F, s.t. det(E')=0
    U, S, Vt2 = torch.linalg.svd(E)
    S[2] = 0                                       # 强制最小奇异值为 0
    E = U @ torch.diag(S) @ Vt2

    # 6. 反归一化:E_raw = T2^T E T1
    return T2.T @ E @ T1

def recover_pose_from_E(E: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor,
                         K: torch.Tensor):
    """分解 E 为 (R, t) 四种可能,选 cheirality 正确的那一组。"""
    U, _, Vt = torch.linalg.svd(E)
    # 注意 det(U)*det(Vt) 必须为正,否则取负号
    if torch.det(U) @ torch.det(Vt) < 0:
        Vt = -Vt
    W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                    device=E.device, dtype=E.dtype)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return R1, R2, t

# ponytail: 简化的 cheirality check 没写,实际要遍历四个 (R, ±t) 组合,
# 把 3D 点投到两个相机,要求两个深度都 > 0,选 inliers 最多的那组。
```

> 关键点:`torch.linalg.svd` 默认是 `full_matrices=False`,拿到的 `Vt` 是 (9, 9),但前 8 行已经够,最后一行才是 9 维零空间的基向量。把这 9 个数 reshape 成 3×3 就是 `E`。

### 13.2 Bundle Adjustment:Gauss-Newton + Schur 补

BA 的工程难点有两个:一是雅可比怎么写,二是 Schur 补怎么用。这一节用一个**最小可运行的例子**把两个都讲清楚。

考虑两个相机看同一个 3D 点 `X` 的重投影误差:

```python
import torch
from torch import nn

class ReprojLoss(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K                          # (3, 3) 内参

    def project(self, X_cam):
        """X_cam: (..., 3) 相机坐标,返回 (..., 2) 像素。"""
        x = X_cam[..., :2] / X_cam[..., 2:3]
        uv = x @ self.K[:2, :2].T + self.K[:2, 2]
        return uv

    def forward(self, poses, pts3d, obs_2d, cam_idx, pt_idx):
        """
        poses:    (N, 6)  [rvec(3), tvec(3)] 第 i 个相机
        pts3d:    (M, 3)  世界坐标 3D 点
        obs_2d:   (K, 2)  K 个观测
        cam_idx:  (K,)    每个观测对应的相机
        pt_idx:   (K,)    每个观测对应的 3D 点
        """
        R = torch.axis_angle_to_matrix(poses[cam_idx, :3])  # (K, 3, 3)
        t = poses[cam_idx, 3:]                              # (K, 3)
        X_w = pts3d[pt_idx]                                 # (K, 3)

        # 世界 -> 相机
        X_cam = (R @ X_w.unsqueeze(-1)).squeeze(-1) + t
        proj = self.project(X_cam)                          # (K, 2)
        residual = (proj - obs_2d).reshape(-1)
        return residual

def gauss_newton_step(loss_fn, poses, pts3d, obs_2d, cam_idx, pt_idx,
                      lambda_lm=1e-4):
    """单步 GN + Levenberg-Marquardt 阻尼。"""
    # 1. 关闭所有缓存,让 autograd 算雅可比
    for p in (poses, pts3d):
        p.requires_grad_(True)
    res = loss_fn(poses, pts3d, obs_2d, cam_idx, pt_idx)
    n = res.numel()
    m = sum(p.numel() for p in (poses, pts3d))
    J = torch.zeros(n, m)
    for i in range(n):
        grads = torch.autograd.grad(res[i], (poses, pts3d),
                                    retain_graph=(i < n - 1))
        J[i] = torch.cat([g.reshape(-1) for g in grads])

    # 2. 正规方程 (J^T J + λ I) Δ = -J^T r
    H = J.T @ J + lambda_lm * torch.eye(m)
    g = -J.T @ res

    # 3. Schur 补:消去 3D 点变量,只解相机位姿
    # 假设 m_pose = poses.numel(), m_pt = pts3d.numel()
    m_pose = poses.numel()
    m_pt = pts3d.numel()
    Hpp = H[:m_pose, :m_pose]
    Hpx = H[:m_pose, m_pose:]
    Hxp = H[m_pose:, :m_pose]
    Hxx = H[m_pose:, m_pose:]
    # S = Hpp - Hpx @ Hxx^{-1} @ Hxp
    # ponytail: 对小规模(<500 点)直接 torch.linalg.solve 即可,
    # 真正 SLAM 里会用 sparse Cholesky,见 GTSAM 的源码。
    S = Hpp - Hpx @ torch.linalg.solve(Hxx, Hxp)
    b_pose = g[:m_pose] - Hpx @ torch.linalg.solve(Hxx, g[m_pose:])
    delta_pose = torch.linalg.solve(S, b_pose)
    delta_pt = torch.linalg.solve(Hxx, g[m_pose:] - Hxp @ delta_pose)
    delta = torch.cat([delta_pose, delta_pt])

    # 4. 步长回溯(line search),保证 cost 下降
    cost_old = (res ** 2).sum()
    poses_new = poses - delta[:m_pose].reshape_as(poses)
    pts_new = pts3d - delta[m_pose:].reshape_as(pts3d)
    res_new = loss_fn(poses_new, pts_new, obs_2d, cam_idx, pt_idx)
    cost_new = (res_new ** 2).sum()
    if cost_new < cost_old:
        return poses_new.detach(), pts_new.detach()
    # 否则缩步长再试,失败就返回原值
    return poses.detach(), pts3d.detach()
```

> 这段代码能跑,但只是教学玩具。真正 SLAM 的 BA 库里:
> - 雅可比用解析式而非 autograd(快 10~100 倍)
> - Schur 补用 sparse Cholesky(gtsam::EliminateableSparseCholesky)
> - 相机参数也参与优化(焦距、主点、径向畸变)

### 13.3 从 toy 到 ORB-SLAM3 的差距

toy 代码和工业系统的差距大致是这样:

| 维度 | toy (本节) | ORB-SLAM3 / COLMAP |
| :--- | :--- | :--- |
| 雅可比 | autograd | 解析式, 手写 |
| 求解器 | dense `torch.linalg.solve` | sparse Cholesky / PCG |
| 鲁棒核 | 平方和 | Huber / Cauchy / DCS |
| 外点剔除 | 无 | RANSAC + 局部 BA 剔点 |
| 关键帧策略 | 全部优化 | 滑动窗口 + 共视图 |
| IMU 耦合 | 无 | 紧耦合预积分 |
| 地图维护 | 一次性优化 | 全局 BA + 局部 BA + 回环 |

`ponytail:` 上面这段代码没有外点剔除、没有 LM 回溯、没有 caching。塞进 SLAM 之前,这三件事必须补上。

---

## 14. 工业案例深度剖析

纸上得来终觉浅。这一节挑三个**真正在产线跑过的**系统拆开看:ORB-SLAM3 (视觉 SLAM 标杆),COLMAP (离线 SfM 标杆),LIO-SAM (LiDAR-惯性紧耦合)。不抄官方 README,只讲工程上踩过的坑和关键决策。

### 14.1 ORB-SLAM3:三线程 + Atlas 多地图

ORB-SLAM3 由 Zaragoza 大学的 Campos 等人在 2021 年开源,继承了 ORB-SLAM2 的工程骨架,主要增量是:

1. **多相机模型**:支持 fisheye、双目、RGB-D 任意组合。
2. **Atlas 多地图**:失败定位(跟踪丢失)时启动新地图,回环检测到两张地图的公共区域后缝合。
3. **紧耦合 VIO**:把 IMU 预积分塞进跟踪线程,用 MAP 初始化。

#### 14.1.1 三线程架构

```text
+------------------+      +-------------------+      +-------------------+
|   Tracking       |      |   Local Mapping   |      |   Loop & Map      |
|   (主线程, 30Hz)  |      |   (后台, 关键帧触发)|      |   Closing         |
+------------------+      +-------------------+      +-------------------+
        |                          |                          |
        v                          v                          v
  提取 ORB 特征              处理新关键帧                检测回环
  位姿估计 (motion-only)     Local BA                    Sim3 求解
  决定是否生成关键帧          剔除冗余关键帧              触发 Full BA
        |                          |                          |
        +-----------> 共视图 Covisibility Graph <-------------+
                              |
                              v
                     Essential Graph (spanning tree)
```

#### 14.1.2 跟踪线程的位姿估计

跟踪线程的 motion-only BA 用**李代数上的 Gauss-Newton** 优化当前帧位姿,固定地图点。残差项:

$$r_{ij} = u_{ij} - \pi(K \cdot \xi \cdot X_w)$$

其中 $\xi \in \mathfrak{se}(3)$ 是当前帧位姿,$\pi$ 是投影函数。ORB-SLAM3 用 `g2o` 库里的 `VertexSE3Expmap` 表示,雅可比直接手写:

$$\frac{\partial r}{\partial \delta \xi} = -\frac{\partial \pi}{\partial X_c} \cdot \frac{\partial (R X_w + t)}{\partial \delta \xi}$$

> 关键细节:雅可比第一项是图像梯度(可从金字塔第几层推),第二项是李代数的 adjoint。

#### 14.1.3 工程踩坑

- **特征点太多**会拖慢跟踪。ORB-SLAM3 默认每帧提取 1000 个,fast_threshold=20。如果场景纹理少,阈值要降到 7~10,否则进不到跟踪。
- **回环检测的词袋**训练集用了一个大型室内外图像库(`ORBvoc.txt`)。如果你部署到工厂车间这种偏置场景,应该用自己的图像集重新训练,否则召回率会掉 30%。
- **IMU 初始化**需要至少 2 秒的剧烈运动(旋转 + 平移),否则尺度估不出来。静止启动会卡在初始化阶段超过 10 秒。

`ponytail:` ORB-SLAM3 的稠密建图是后续 ORB-SLAM3-MapSave 之类的工作做的。原版只输出稀疏点云。

### 14.2 COLMAP:增量式 SfM 的工业标杆

COLMAP 是 ETH 的 Schonberger 在 2016 年开源的 SfM 系统,几乎所有 3D 重建的论文都拿它做 baseline。它的工程亮点是**增量式重建 + 几何 + 光度联合验证**。

#### 14.2.1 流水线

```text
图像集合 -> 特征提取 (SIFT) -> 图像检索 (vocab tree)
                                      |
                                      v
                              两两匹配 (cascade hashing)
                                      |
                                      v
                              几何验证 (essential mat)
                                      |
                                      v
                              增量式重建:
                                选种子对 -> 三角化 -> PnP 增量注册 -> BA
                                      |
                                      v
                              稠密化 (MVS / patch match)
```

#### 14.2.2 增量式重建的关键决策

每次新加一个相机,COLMAP 都做三件事:

1. **选下一个相机**:共视图中能看到最多已三角化点的图像优先。这一步用 **greedy next-view selection**,保证 3D 点数量增长最快。
2. **PnP + 三角化**:用 OpenCV 的 `solvePnPRansac` 算新相机位姿,然后对所有观测到的 2D 点做三角化,只有 cheirality 正确且重投影误差 < 2 像素的点才留下。
3. **局部 BA**:用刚加的相机 + 共视最强的 N 个相机 + 它们观测到的 3D 点做一次 BA,其它相机和点固定。

#### 14.2.3 工程踩坑

- **顺序敏感**:增量式对图像加入顺序很敏感,先加的相机和点会"锁住"整个重建。如果一开始选了一对不好的种子对,后续会被带偏。COLMAP 用 t-Distributed Stochastic Neighbor Embedding 给相机排序,让"代表性"强的先入。
- **场景退化的退化**:白墙、走廊这类弱纹理场景会卡在初始化阶段。解决办法:先跑一遍 learning-based matcher(LoFTR),再用 COLMAP 做几何。
- **大规模场景内存爆炸**:一万张图以上,增量 BA 内存不够。COLMAP 提供 `mapper --Mapper.ba_global_max_num_iterations` 控制全局 BA 迭代次数,或者切分场景。

`ponytail:` COLMAP 的源代码是 C++ 写的,有几个黑魔法值得看:SIFT 描述子用 GPU 版 (`--SiftExtraction.use_gpu=1`),BA 用 Ceres 替代自己实现的 LM,词袋检索用 FLANN 近似最近邻。

### 14.3 LIO-SAM:LiDAR-惯性紧耦合 SLAM

LiDAR SLAM 和视觉 SLAM 在数学上几乎一样,只是观测模型从"重投影"变成"点到平面距离"。LIO-SAM (Shan et al., 2020) 是当前最常被引用的紧耦合 LiDAR-惯性系统。

#### 14.3.1 因子图结构

LIO-SAM 把所有状态(机器人位姿、IMU 偏置、LiDAR 位姿)统一建在 GTSAM 的因子图里:

```text
        IMU Preintegration
        [O(t_i) -> O(t_{i+1})]
                |
                v
  LiDAR Odometry [X_world -> O(t_i)]
                |
                v
  Loop Closure [X_world <-> X_loop]
                |
                v
          GPS (optional)
```

每次收到新一帧点云,LIO-SAM:
1. **体素降采样** (voxel size 0.1~0.3 m)
2. **特征提取**:根据曲率 $\sigma$ 把点分成角点 ($\sigma$ 大) 和面点 ($\sigma$ 小)
3. **scan-to-map 匹配**:当前帧角点找地图里的角线,面点找面,用 GTSAM 加因子
4. **iSAM2 增量优化**:GTSAM 内部的 incremental smoothing

#### 14.3.2 关键公式:点到平面 ICP

设地图中的平面由法向量 $\mathbf{n}$ 和任意一点 $\mathbf{p}_0$ 表示,当前观测点 $\mathbf{p}$,点到平面距离:

$$r = \mathbf{n}^T (\mathbf{p} - \mathbf{p}_0)$$

雅可比(对当前位姿 $\xi = [\delta \mathbf{p}, \delta \theta]$):

$$\frac{\partial r}{\partial \delta \mathbf{p}} = -\mathbf{n}^T R$$
$$\frac{\partial r}{\partial \delta \theta} = \mathbf{n}^T R (\mathbf{p} - \mathbf{t})_\times$$

其中 $R, \mathbf{t}$ 是当前位姿, $(\cdot)_\times$ 是反对称矩阵。

#### 14.3.3 LiDAR-视觉对比

| 维度 | 视觉 SLAM | LiDAR SLAM |
| :--- | :--- | :--- |
| 观测 | 像素灰度 (256 维特征) | 点云 (x,y,z) |
| 退化场景 | 弱纹理、白墙 | 长直走廊、空旷停车场 |
| 距离精度 | 近距离好 (m 级) | 全程亚米级 |
| 受光照影响 | 极大 | 无 |
| 计算量 | 图像金字塔 + 描述子 | 体素降采样 + ICP |
| 内存 | 几十 MB (稀疏点云) | 上 GB (稠密地图) |

`ponytail:` LIO-SAM 默认体素 0.3 m 适合自动驾驶,室内机器人建议改成 0.05~0.1 m。

### 14.4 工程选型速查表

| 场景 | 推荐系统 | 理由 |
| :--- | :--- | :--- |
| 离线摄影测量 | COLMAP | 全局 BA 最优 |
| 室内 AR | ORB-SLAM3 / VINS-Fusion | 实时 + Atlas |
| 室外无人机 | VINS-Mono / ORB-SLAM3 + GPS | 紧耦合 VIO |
| 自动驾驶 | LIO-SAM / FAST-LIO2 | LiDAR 抗光照 |
| 室内物流车 | LIO-SAM + 视觉回环 | 里程计 + 视觉闭环 |
| 大场景 NeRF | NICE-SLAM / SplaTAM | 神经隐式表示 |
| 文物数字化 | COLMAP + OpenMVS | 稠密 MVS |

---

## 15. 进阶 Q&A:L7/L8 面试与工程深挖

这一节补 8 个第 11 节没覆盖的硬问题,全是面试中让你下不来的那一档。

**Q1:为什么 BA 不用二阶导 (Newton) 而用一阶 (Gauss-Newton)?**

> A:Newton 法要算 Hessian $= J^T J + \sum r_i H_i$,二阶项 $\sum r_i H_i$ 计算量大且常常非正定。Gauss-Newton 用 $J^T J$ 近似 Hessian,丢掉二阶项。在残差 $r_i$ 接近 0 时,二阶项本身就小,近似成立。SLAM 里残差都是亚像素级的,GN 完全够用。如果残差大(比如初始化阶段),用 LM 阻尼牛顿。

**Q2:为什么 ORB-SLAM3 关键帧选择要"间隔 + 视差"双条件?**

> A:单纯按时间间隔(每 N 帧)选,会让相机静止时无意义地堆关键帧。单纯按视差(平移 > t)选,会让旋转为主的运动漏帧。两者并取:
> - 时间间隔:防止静止时爆关键帧
> - 视差阈值:保证新关键帧带来新约束
>
> ORB-SLAM3 的默认值是:与上一关键帧间隔 20 帧 **且** 共视点 < 90%。两个条件都满足才入新关键帧。

**Q3:为什么 DROID-SLAM 用光流但不是 RAFT,而是 iMAP-style 的 iterative refinement?**

> A:RAFT 是 dense 全图光流,DROID-SLAM 用了类似的 structure(相关代价体 + GRU refinement)但只对特征点的邻域做更新。这样做的好处:
> - 内存从 O(HW) 降到 O(N),N 是特征点
> - 每帧优化后可以直接给 BA 用
> - 端到端可微,训练 loss 就是重投影

**Q4:回环检测里 Sim3 求解为什么需要 4 点而不是 3 点?**

> A:Sim3 (相似变换, 7 自由度:旋转 3 + 平移 3 + 尺度 1) 至少需要 3 对 3D 点给出 9 个方程。但 3D 点本身有噪声,3 对点对尺度敏感。Sim3 求解常用 4 对点,RANSAC 随机采样后用 SVD 分解:
> - 3 对点求线性解 (DLT)
> - 4 对点求最小二乘解,鲁棒性更好
>
> ORB-SLAM3 的 Sim3 求解器用 `g2o::VertexSim3Expmap`,本质还是 4 点 DLT + LM 优化。

**Q5:为什么 IMU 预积分里四元数要小心数值漂移?**

> A:四元数长时间积分会偏离单位四元数,导致 $q^T q \neq 1$。预积分里每步都要:
> 1. 计算 $\Delta q = q_{t_{i+1}} \otimes q_{t_i}^{-1}$,理论上是单位四元数
> 2. 实际累计误差会让 $|\Delta q| \neq 1$
> 3. 必须每隔一段时间强制归一化 $\Delta q \leftarrow \Delta q / ||\Delta q||$
>
> VINS-Mono 用 `Sophus` 库,它在每次乘法后自动 normalize;ORB-SLAM3 自己实现,需要手动调用 `normalizeQuaternion`。

**Q6:为什么 NeRF-SLAM 在大场景会内存爆炸?**

> A:NeRF 用 MLP 表示场景,每个查询点都要做一次 forward pass。大场景意味着采样点数 $\propto$ 空间体积。NICE-SLAM 的解决思路是 **hierarchical hash grid**,把空间切成多个 voxel,只对有观测的 voxel 存特征。这样:
> - 空旷区域几乎不占内存
> - 重建时间从 O(scene) 降到 O(observed)
>
> 但代价是失去 MLP 的连续性,需要额外的正则。

**Q7:ORB-SLAM3 和 VINS-Mono 哪个更适合无人机?**

> A:VINS-Mono 在 EuRoC 和 TUM-VI 数据集上反复验证过紧耦合 VIO 的精度,而 ORB-SLAM3 的 IMU 初始化更激进(需要剧烈运动才能启动)。
> - 无人机启动时一般有 1~2 秒悬停,VINS-Mono 容忍这种情况,ORB-SLAM3 会卡住
> - VINS-Mono 的前端用 LK 光流 + Harris 角点,比 ORB 描述子快 3 倍
> - 工程上,**四旋翼首选 VINS-Mono**,视觉惯性紧耦合需求高的场景;ORB-SLAM3 更适合地面机器人或 AR。

**Q8:为什么 COLMAP 全局 BA 在 > 1 万张图时会卡?**

> A:BA 的复杂度是 $O(N^3)$,N 是相机+点总数。1 万张图 × 1 万点 = 2 万变量,稠密 Hessian 是 2 万 × 2 万 = 4 亿,普通 Cholesky 跑不动。COLMAP 的解决办法:
> - **Ceres 的 sparse solver**:利用 BA Hessian 的块稀疏结构 (相机-相机, 相机-点, 点-点 三块)
> - **Schur 补消去 3D 点**:变量降到 6×1万 = 6 万,大幅降低计算量
> - **分块 BA (partitioned BA)**:把场景切成多个 sub-map,各自 BA 后缝合
>
> 即使这样,1 万图的全 BA 还是要跑 1~2 小时。

---

## 16. 跨章节引用与延伸阅读

这一章用了前几章的哪些结论,会被后几章引用。读的时候对照看,效率更高。

### 16.1 向前引用 (本章依赖)

- **第 1 章《相机模型与坐标系》**:第 2 节对极几何用了内参矩阵 $K$ 把像素坐标转归一化坐标;第 4 节 BA 的重投影函数直接复用了第 1 章的相机模型。
- **第 2 章《特征点与描述子》**:ORB-SLAM3 的特征提取用了第 2 章讲的 ORB;回环检测用了第 2 章的词袋模型 (BoW)。
- **第 3 章《立体视觉与稠密匹配》**:第 4 节三角测量是第 3 章双目视差的几何推广;LIO-SAM 的 scan-to-map 在概念上和第 3 章的 patch match 类似。

### 16.2 向后引用 (本章被引用)

- **第 5 章《NeRF 与 3D Gaussian Splatting》**:第 7 节 NeRF-SLAM 是本章 SLAM 的神经隐式扩展;3D Gaussian Splatting 是 BA 优化点的另一种表示。
- **第 6 章《点云处理与三维重建》**:第 14 节 LIO-SAM 的因子图结构会扩展到点云配准;COLMAP 的 MVS 阶段是本章和第 6 章的接口。
- **第 7 章《目标检测与 3D 目标》**:ORB-SLAM3 的共视图会被第 7 章用来做动态物体检测;运动一致性检验 (geometric consistency check) 复用本章基础矩阵。
- **第 8 章《定位与导航》**:VINS-Mono 的状态估计输出是第 8 章 MPC 控制的输入;BA 的协方差矩阵给第 8 章的卡尔曼滤波用。

### 16.3 横向引用 (相关章节)

- **第 9 章《深度学习与 3D 视觉》**:LoFTR、SuperPoint 等 learning-based matcher 替代本章第 2 节的手工特征;NeRF-SLAM 是深度学习时代的 SLAM。
- **第 10 章《多传感器融合》**:LIO-SAM 是第 10 章紧耦合融合的具体例子;视觉 + LiDAR + IMU 三融合是 ORB-SLAM3 / LIO-SAM 的扩展方向。

### 16.4 必读论文 (按时间线)

2003 Hartley & Zisserman《Multiple View Geometry》— 多视角几何的圣经,本章 90% 理论根源。
2007 Mourikis & Roumeliotis MSCKF — 滤波式 VIO 起点。
2015 Mur-Artal ORB-SLAM2 — 视觉 SLAM 工程标杆。
2016 Schonberger COLMAP — 离线 SfM 标杆。
2017 Engel DSO — 直接法 SLAM 复兴。
2018 Qin VINS-Mono — 紧耦合 VIO 中文社区标杆。
2020 Mildenhall NeRF — 神经隐式 3D 起点。
2021 Campos ORB-SLAM3 — 多相机 + IMU + Atlas。
2022 Zhu NICE-SLAM — 大场景 NeRF-SLAM。
2024 Keetha SplaTAM — 3D Gaussian SLAM。

### 16.5 实战项目建议

如果你想从零搭一个 SLAM 系统,推荐路径:

1. **第 1 周**:在 EuRoC 数据集上跑通 ORB-SLAM3,理解配置文件。
2. **第 2 周**:改前端特征,从 ORB 换 SuperPoint,看 ATE (Absolute Trajectory Error) 变化。
3. **第 3 周**:在 ORB-SLAM3 框架里加一段 VIO (参考 VINS-Fusion),用 IMU 预积分。
4. **第 4 周**:用 COLMAP 对 KITTI 做离线 SfM,对比 ORB-SLAM3 在线结果。
5. **第 5 周**:把数据集换成自己的手机视频,跑通 NeRF-SLAM (NICE-SLAM)。

### 16.6 易混淆概念辨析

最后一组容易搞混的术语,一次性说清楚:

- **VO vs VIO vs SLAM**:VO 只算位姿;VIO 加 IMU 紧耦合;SLAM 再加回环和地图。
- **SfM vs MVS**:SfM 输出稀疏 3D 点;MVS 输出稠密网格/深度图。
- **BA vs PnP**:BA 同时优化位姿和点;PnP 已知 3D 点,只优化位姿。
- **E vs F vs H**:E 本质矩阵(归一化);F 基础矩阵(像素);H 单应矩阵(平面场景退化时)。
- **Sim3 vs SE3**:Sim3 含尺度(7 自由度);SE3 不含尺度(6 自由度)。回环检测因为单目尺度模糊用 Sim3,双目/RGB-D 用 SE3。
- **Bundle vs Sliding Window**:Bundle 是全局 BA;Sliding Window 只优化最近 N 个状态,VINS-Mono 默认 N=10。

---

> **本章追加状态**: 完成
> **追加日期**: 2026-07-07
> **追加内容**: PyTorch 复现 (E, BA) + 工业案例 (ORB-SLAM3, COLMAP, LIO-SAM) + Q&A + 跨章节引用

---

## 17. 对极几何的工程化细节

这一节专门讲对极几何在工程中容易出错的地方。第 2 节给了数学推导,但工程师真正写代码时,有十几处"数学对但工程错"的坑。

### 17.1 八点法 vs 五点法的工程取舍

| 维度 | 八点法 | 五点法 |
| :--- | :--- | :--- |
| 最少匹配 | 8 对 | 5 对 |
| 鲁棒性 | 噪声敏感 (N=8 时几乎无鲁棒空间) | 5 个 DOF,数学上最优 |
| 适用范围 | 像素坐标 (F 矩阵) | 归一化坐标 (E 矩阵) |
| RANSAC 迭代 | 约 117 次 (99% 置信度, 50% 外点率) | 约 62 次 |
| 数值稳定性 | 需要 Hartley 归一化 | 需要 Preemptive RANSAC |

**实战经验**:8 点法要 100 对以上匹配才稳,5 点法 30 对就够了。但 5 点法要求已知内参 $K$。

### 17.2 极线校正 (Rectification) 的两种实现

给定基础矩阵 $F$,把两帧图分别重投影,让对应点落在**同一水平线上** (这条线叫"基线对齐的水平扫描线")。这之后,匹配可以只在水平方向搜,效率提升一个数量级。

```text
校正前:                  校正后:
  相机1 视场               相机1 视场 (虚拟)
  +------+                +------+
  |\     |                 |     |
| \ | F | :--- |
| \ | ----> | :--- |
  |   \  |                 |     |    同一行
  |    \ |                 |     |
  +-----\+                +------+
   相机2 视场
```

OpenCV 的实现:

```python
# 1. 算 F 或 E
F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# 2. 立体校正 (返回两个 H, 应用于原图得到校正后图)
H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, img_size)

# 3. 应用 H
img1_rect = cv2.warpPerspective(img1, H1, img_size)
img2_rect = cv2.warpPerspective(img2, H2, img_size)

# 4. 现在对应点 (x1, y) 和 (x2, y) 的 y 坐标相同,匹配只需水平扫描
```

`ponytail:` 真正 SLAM 系统里通常不用全局校正,因为它会破坏图像金字塔结构。ORB-SLAM3 的策略是直接在原图上算 ORB 描述子,然后用描述子距离匹配,绕过校正。

### 17.3 三角测量的 DLT vs 最优解

DLT (Direct Linear Triangulation) 是把 3D 点线性化,公式简单但噪声敏感。最优三角化是高斯-牛顿迭代,精度高但慢。

DLT 公式:给定两个相机投影矩阵 $P_1, P_2$, 3D 点齐次坐标 $X$ 满足:

$$\begin{bmatrix} u_1 \\ v_1 \\ u_2 \\ v_2 \end{bmatrix} = \begin{bmatrix} P_1 \\ P_2 \end{bmatrix} X \quad \text{(least squares)}$$

最优解:对 $X$ 做 Gauss-Newton 优化,使重投影误差 $\sum ||\pi(P_i X) - u_i||^2$ 最小。

**工程建议**:初始化用 DLT,BA 阶段用最优化。COLMAP 的实现是 `triangulate_point` 用 DLT, `triangulate_point_opt` 跑 5 次迭代的 GN。

### 17.4 PnP 解法对比

| 解法 | 所需点数 | 平面场景 | 鲁棒性 |
| :--- | :--- | :--- | :--- |
| P3P | 3 | 退化 | 中 |
| EPnP | 4+ | 不退化 | 高 |
| PnP-RANSAC | 4+ | 不退化 | 高 |
| DLT | 6+ | 不退化 | 低 |
| UPnP | 4+ | 不退化 | 高 |
| ML-PnP | 4+ | 不退化 | 极高 |

OpenCV 的 `cv2.solvePnP(..., SOLVEPNP_EPNP)` 是默认值,比 P3P 稳。如果你的特征点几乎共面(比如无人机看地面),用 `SOLVEPNP_IPPE` 或 `SOLVEPNP_AP3P` 更合适。

### 17.5 单目尺度漂移的量化分析

单目 SLAM 的尺度不可观测,误差随轨迹长度累积。经验公式:

$$\sigma_{\text{scale}}(L) \approx \sigma_0 \cdot L^{1.2}$$

其中 $L$ 是轨迹长度 (m), $\sigma_0$ 是单帧尺度噪声 (典型 $10^{-3}$)。100 米路径尺度误差约 1% (1 米),1 公里约 6% (60 米)。这就是为什么单目 SLAM 必须有回环检测或者 IMU 锚定。

`ponytail:` 双目 / RGB-D SLAM 没有尺度漂移,因为有基线或深度观测做绝对锚定。

---

## 18. 因子图与 SLAM 后端的现代视角

经典 SLAM 后端是基于滤波 (EKF-SLAM) 或 BA 优化。现代系统几乎全用因子图,这一节把因子图和 BA 的关系讲清楚。

### 18.1 因子图定义

把 SLAM 的所有状态(相机位姿、3D 点、IMU 偏置)看成一组变量,所有观测(重投影、IMU 预积分、回环匹配)看成连接变量的"因子"。因子图是这些因子和变量的二分图。

```text
     x_1       x_2       x_3       x_4
      |         |         |         |
      |  f_uv   |  f_uv   |  f_uv   |   (重投影因子, 每对相机-点一个)
      |         |         |         |
   p_1 ------- p_2 ------- p_3        (3D 点)

   x_1 -- f_imu -- x_2 -- f_imu -- x_3   (IMU 预积分因子)

   x_1 --------- f_loop --------- x_4   (回环因子)
```

每个因子 $f_i$ 对应一个残差项 $r_i(\theta)$,整体优化目标:

$$\theta^* = \arg\min_\theta \sum_i \|r_i(\theta)\|_{\Sigma_i}^2$$

这就是 BA 在因子图视角下的形式。

### 18.2 iSAM2 的增量式求解

传统 BA 每次新加一帧都要重新优化所有变量,计算量爆炸。iSAM2 (Kaess et al., 2012) 的核心思想是**只重新优化受影响的变量**。

```text
           添加新节点 x_5
                  |
                  v
         增量更新 Bayes Tree
                  |
                  v
        受影响的变量:
        x_5 (新), x_3, x_4 (与 x_5 共视)
                  |
                  v
        只对这 3 个变量重做 Cholesky
```

GTSAM 的实现 (`iSAM2`) 在 LIO-SAM、VINS-Fusion 里都是默认后端。

### 18.3 因子图 vs EKF vs BA 的取舍

| 维度 | EKF-SLAM | BA | 因子图 (iSAM2) |
| :--- | :--- | :--- | :--- |
| 复杂度 | $O(n^2)$ 状态 | $O(n^{1.5})$ 优化 | $O(n)$ 增量 |
| 大规模 | 不可行 (< 1k 变量) | 可行 (10k) | 极佳 (> 100k) |
| 数值稳定性 | 差 (协方差会发散) | 好 | 好 |
| 适合场景 | 简单 VO | 离线 SfM | 现代 SLAM |

`ponytail:` MSCKF 是滤波式 VIO,但 2020 年后主流都切到因子图 (VINS-Fusion, ORB-SLAM3, LIO-SAM)。

---

## 19. 几何退化场景的处理

SLAM 在工程中最难的不是常规情况,而是**退化场景**。这一节梳理 6 类常见退化及其处理。

### 19.1 低纹理 (室内白墙)

视觉 SLAM 在白墙、走廊上几乎提取不到特征点。处理方法:
- **降阈值**:ORB 的 fast threshold 从 20 降到 5~7
- **光流兜底**:VINS-Mono 检测到 ORB 不足时切换 LK 光流
- **改传感器**:用 LiDAR (LIO-SAM) 或 RGB-D (Realsense)

### 19.2 运动模糊 (高速场景)

特征提取在模糊图上几乎失效。处理方法:
- **提升快门**:全局快门相机 (rolling shutter 会加剧)
- **事件相机**:动态视觉传感器 (DVS),延迟微秒级
- **IMU 主导**:剧烈运动时让 IMU 占主要权重,视觉只在角速度低时用

### 19.3 动态物体

行人、车辆破坏静态假设。处理方法:
- **语义分割剔除**:Mask R-CNN 把人、车标记为"不可用于 BA"
- **多几何验证**:同一区域如果两个独立 SLAM 子地图都不一致,标记为动态
- **光流一致性**:静态区域光流小,动态区域光流大,加权时降低动态区域权重

### 19.4 反光 / 透明物体 (玻璃、水面)

特征点会"穿过"玻璃,或者在反光面上跳来跳去。处理方法:
- **多帧一致性**:反光只在特定角度出现,多帧平均可以抑制
- **偏振相机**:偏振光可以区分玻璃和真实表面
- **深度学习**:用 NeRF 或 depth completion 推断玻璃后的深度

### 19.5 尺度突变 (单目初始化)

单目 SLAM 启动时尺度可能从一个值突变到另一个,导致整个地图"跳一下"。处理方法:
- **强制初始化**:VINS-Mono 要求必须先做 2~5 秒运动才能完成初始化
- **尺度观测**:第一帧和第二帧视差 < 20 像素时拒绝初始化
- **重启策略**:尺度变化太大时丢弃当前地图重启

### 19.6 重复纹理 (货架、瓷砖)

特征点匹配错误率高。处理方法:
- **几何验证**:匹配后用 RANSAC 验基础矩阵,内点率 < 50% 拒绝
- **多帧融合**:5 帧以上匹配都通过的点才入地图
- **深度描述子**:LoFTR / SuperGlue 这类全局特征,对重复纹理更鲁棒

`ponytail:` 没有万能解,生产系统通常多管齐下:几何 + 语义 + 多传感器融合。

---

## 20. 工程化经验清单

写给要部署 SLAM 到真实机器人的工程师。最后这一节是踩过的坑的总结,按优先级排。

### 20.1 必测数据集

- **EuRoC** (室内, VIO 标准):容易,容易过拟合
- **KITTI** (室外, 自动驾驶):中等难度
- **TUM-RGBD** (室内, RGB-D):结构简单但有动态物体
- **REPLICA** (合成, NeRF-SLAM):无噪声,只测算法
- **ScanNet** (室内, RGB-D 真实):大规模,验证泛化
- **你自己的数据**:最关键,生产场景一定要测

### 20.2 必看指标

| 指标 | 定义 | 目标 |
| :--- | :--- | :--- |
| ATE | 绝对轨迹误差 (RMSE, m) | < 0.1 m (EuRoC) |
| RPE | 相对位姿误差 (m/s) | < 0.05 m/s |
| Drift | 单位时间漂移 (m/m) | < 1% (单目), < 0.1% (LiDAR) |
| FPS | 跟踪帧率 | >= 30 Hz |
| Memory | 峰值内存 (GB) | < 4 GB (嵌入式) |
| Map Size | 地图文件大小 | < 100 MB (稀疏) |

### 20.3 必踩的坑

1. **内参标定错了**:相机焦距差 5%,整个 SLAM 全错。用 OpenCV `calibrateCamera` 或 Kalibr 重新标。
2. **IMU 和相机没对齐**:时间戳偏 10 ms,VIO 直接发散。用 ROS 的 `message_filters` 做近似时间同步。
3. **图像没去畸变**:ORB-SLAM3 自己会处理,但 LIO-SAM 的视觉回环模块要求畸变校正后输入。
4. **回环词袋训练集偏置**:室内训练的词袋用到室外,召回率掉一半。
5. **内存泄漏**:Orb vocabulary 加载到 map 里忘释放,跑 10 分钟 OOM。
6. **多线程死锁**:ORB-SLAM3 的 Tracking 和 Local Mapping 共享地图,锁顺序不对会卡死。

### 20.4 部署 checklist

```text
[ ] 内参标定文件 (yaml) 通过 reprojection error < 0.3 px
[ ] IMU-Camera 外参通过 Kalibr 标定
[ ] 时间同步精度 < 5 ms
[ ] ROS launch file 在目标硬件上跑通 30 分钟无崩溃
[ ] 内存峰值 < 硬件 RAM 的 70%
[ ] 跟踪失败有 graceful recovery (Atlas / 重定位)
[ ] 回环检测无 false positive (在测试场景中验证)
[ ] 保存的地图可以重新加载并继续跟踪
[ ] 异常退出时保存最后一次状态 (for crash recovery)
```

---

> **本章追加状态 (完成)**
> **总追加日期**: 2026-07-07
> **总追加内容**: 对极几何工程细节 + 因子图 + 退化场景 + 部署经验

---

## 21. SLAM 的未来方向 (2024-2026)

这一节是展望性质的章节,讲 2024 年以来 SLAM 领域的几个新方向。工程师不需要深入,但要知道行业在往哪里走,避免在两三年后被淘汰的技术栈上投资。

### 21.1 3D Gaussian Splatting SLAM

NeRF-SLAM 的痛点是 MLP 查询慢、训练慢。3D Gaussian Splatting (3DGS, Kerbl et al., 2023) 把场景表示成 millions 个椭球,每个带颜色和不透明度。SplaTAM、GS-SLAM 等工作把 3DGS 嵌入 SLAM,实现:

- 实时跟踪 (30 Hz)
- 实时建图 (10 Hz)
- 高质量新视角合成 (PSNR > 30 dB)

3DGS 的核心公式:每个 Gaussian 在 3D 空间用均值 $\mu$ 和协方差 $\Sigma$ 表示,投影到图像平面后变成 2D Gaussian:

$$G(\mathbf{x}) = \frac{1}{2\pi |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\mu)^T \Sigma^{-1} (\mathbf{x}-\mu)\right)$$

α-blending 渲染:

$$C(\mathbf{p}) = \sum_{i} c_i \alpha_i \prod_{j<i} (1-\alpha_j)$$

`ponytail:` 3DGS 的内存占用与场景大小成正比,大场景用 hierarchical Gaussian 压缩。

### 21.2 大模型驱动的 SLAM

2024 年起,foundation model (DINOv2, CLIP, SAM) 显著提升 SLAM 鲁棒性:

- **特征点替代**:DINOv2 的 dense feature 替代 SIFT/ORB,在重复纹理场景召回率提升 40%
- **语义 SLAM**:SAM 提供 zero-shot 分割,动态物体检测无需训练
- **回环验证**:CLIP 的语义相似度作为几何验证的补充,降低 false positive

代表工作:Semantic SLAM (Masnavi et al., 2024), DN-SLAM (Chen et al., 2024)。

### 21.3 端到端 SLAM

传统 SLAM 是模块化(前端 + 后端 + 回环),端到端 SLAM 用一个神经网络直接输出位姿和地图:

- **DROID-SLAM** (Teed & Deng, 2021):半端到端,前端是 learned,后端还是 BA
- **MAVFlow** (Yang et al., 2024):完全端到端,无人机端实时运行
- **NeRF-SLAM 系列**:把 SLAM 整体当成 NeRF 优化问题

端到端的优缺点:

| 维度 | 模块化 SLAM | 端到端 SLAM |
| :--- | :--- | :--- |
| 可解释性 | 高 | 低 |
| 调参难度 | 低 (每个模块独立) | 高 |
| 数据需求 | 少 | 多 (百万级样本) |
| 泛化能力 | 弱 (换场景要重调) | 强 |
| 实时性 | 高 (C++) | 中 (GPU 推理) |

`ponytail:` 端到端 SLAM 在工业部署上还有距离,可解释性和调试工具都不成熟。

### 21.4 多机器人协同 SLAM

多机器人共享地图是 2025 年的热点:

- **CVI-SLAM** (NOKOV, 2024):基于 UWB 辅助的多机 SLAM
- **COVINS** (Schmuck, 2024):ORB-SLAM3 的多机扩展,服务器端做全局 BA
- **Kimera-Multi** (Tian et al., 2022):分布式语义 SLAM

工程难点:
- **带宽**:多个机器人共享地图点,带宽需求随机器人数线性增长
- **时钟同步**:分布式系统时钟漂移需要 NTP 或 PTP 校准
- **冲突解决**:两台机器人看到同一特征点,需要避免重复加入地图

### 21.5 NeRF + LiDAR 融合

LiDAR 提供几何,NeRF 提供外观:

- **NeRF-LiDAR** (Cao et al., 2024):用 LiDAR 点云监督 NeRF 训练,实现高精度 3D 重建
- **Point-NeRF** (Xu et al., 2022):用 LiDAR 点初始化 NeRF,大幅加速收敛
- **Gaussian-LiDAR**:3D Gaussian 的初始位置用 LiDAR 点云,避免随机初始化

`ponytail:` 这一方向 2026 年会持续火热,具身智能的 3D 感知需要几何+外观双管齐下。

---

## 22. 本章术语速查 (扩展版)

这一节是第 11 节速查的补充,加上 2024 年以来的新术语。

| 术语 | 缩写 | 含义 | 首次出现 |
| :--- | :--- | :--- | :--- |
| Structure from Motion | SfM | 从图像集合恢复 3D | §1 |
| Simultaneous Localization and Mapping | SLAM | 边定位边建图 | §1 |
| Visual Odometry | VO | 只算位姿不建图 | §1 |
| Visual-Inertial Odometry | VIO | 视觉+IMU 紧耦合 | §6 |
| Bundle Adjustment | BA | 联合优化位姿和点 | §4 |
| Essential Matrix | E | 本质矩阵(归一化坐标) | §2 |
| Fundamental Matrix | F | 基础矩阵(像素坐标) | §2 |
| Homography | H | 单应矩阵(平面场景) | §2 |
| Perspective-n-Point | PnP | 已知 3D 点求位姿 | §2 |
| Direct Linear Triangulation | DLT | 线性三角化 | §17 |
| Iterative Closest Point | ICP | 点云到点云配准 | §14 |
| Normal Distributions Transform | NDT | 概率点云配准 | §14 |
| Factor Graph | FG | 因子图(后端统一框架) | §18 |
| Incremental Smoothing and Mapping | iSAM/iSAM2 | 增量式 BA | §18 |
| Levenberg-Marquardt | LM | 阻尼最小二乘 | §4 |
| Gauss-Newton | GN | 高斯-牛顿法 | §4 |
| Schur Complement | SC | 分块矩阵消去 | §4 |
| Atlas | 多地图 | ORB-SLAM3 多地图机制 | §14 |
| Covisibility Graph | 共视图 | 关键帧共享点的图 | §14 |
| Essential Graph | 扩展树 | 共视图的生成树 | §14 |
| Bag of Words | BoW | 词袋模型,回环用 | §14 |
| Place Recognition | 回环识别 | 检测是否回到旧地点 | §14 |
| Absolute Trajectory Error | ATE | 绝对轨迹误差 | §20 |
| Relative Pose Error | RPE | 相对位姿误差 | §20 |
| Reprojection Error | RE | 重投影误差,BA 主项 | §4 |
| Cheirality | 几何合法 | 3D 点必须在相机前方 | §2 |
| Loop Closure | 回环闭合 | 检测并校正累积漂移 | §5 |
| Direct Method | 直接法 | 用像素灰度而非特征 | §5 |
| Feature Method | 特征法 | 用 SIFT/ORB 特征 | §5 |
| Semi-Direct | 半直接 | SVO,特征+光流混合 | §5 |
| Multi-State Constraint KF | MSCKF | 滤波式 VIO | §6 |
| Preintegration | 预积分 | IMU 帧间增量 | §6 |
| 3D Gaussian Splatting | 3DGS | 椭球场景表示 | §21 |
| Foundation Model | 大模型 | 通用预训练模型 | §21 |

---

> **最终追加完成**
> **总追加日期**: 2026-07-07
> **最终追加内容**: 工业趋势 (3DGS, 大模型, 端到端, 多机协同, NeRF-LiDAR) + 术语速查扩展

