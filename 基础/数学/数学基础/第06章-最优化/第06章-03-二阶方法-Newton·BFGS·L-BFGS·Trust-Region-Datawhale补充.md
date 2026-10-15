---
title: 第06章-03-二阶方法 · Datawhalechina 补充
tags:
  - 数学
  - 最优化
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/math-for-ai ch7 连续优化与 datawhalechina/key-book ch7 收敛率，补充二阶方法的严格数学证明与收敛率分析
companion_to: 第06章-03-二阶方法-Newton·BFGS·L-BFGS·Trust-Region.md
source: https://github.com/datawhalechina/math-for-ai ; https://github.com/datawhalechina/key-book
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 二阶方法 · Datawhalechina 对照补充

> 本文件是对同目录《第06章-03-二阶方法-Newton·BFGS·L-BFGS·Trust-Region.md》的补充，内容整理自 [datawhalechina/math-for-ai](https://github.com/datawhalechina/math-for-ai)（人工智能的数学基础，MML-Book 中译版）**第 7 章 连续优化**与 [datawhalechina/key-book](https://github.com/datawhalechina/key-book)（机器学习理论导引）**第 7 章 收敛率**。原文档侧重工程视角的直觉与对标；本补充聚焦**严格数学证明**这一原文档的薄弱面。

---

## 一、补充来源地图

| datawhale 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| math-for-ai ch7.1（一阶方法） | 梯度下降收敛率定理 | 第06章-02（一阶方法） | 一阶/二阶收敛率的对照基准 |
| math-for-ai ch7.2（二阶方法） | Newton 法、拟 Newton（BFGS/L-BFGS）严格推导 | 本文件对应章 | **核心补充**：收敛率证明、Hessian 正定性与下降方向 |
| math-for-ai ch7.3（优化器谱系） | 动量/自适应方法的统一分析 | 第06章-02 | Adam 可视为"对角近似的拟 Newton" |
| key-book ch7（收敛率） | 一阶/二阶方法的收敛速率定义与定理 | 本文件对应章 | O(1/k) vs O(1/k²) vs 二次收敛的严格刻画 |

---

## 二、核心补充内容

### 2.1 Newton 法的收敛率定理（math-for-ai ch7.2）

**定理（局部二次收敛）**：设 $f:\mathbb{R}^d \to \mathbb{R}$ 二阶连续可微，$x^\*$ 为满足 $\nabla f(x^\*)=0$、Hessian $\nabla^2 f(x^\*) \succeq mI$（强凸）、$\nabla^2 f$ 在 $x^\*$ 附近 $L$-Lipschitz 的局部极小点。则当初始点 $x_0$ 足够接近 $x^\*$ 时，Newton 迭代

$$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

满足二次收敛：

$$\|x_{k+1} - x^\*\| \le \frac{L}{2m} \|x_k - x^\*\|^2$$

**证明要点**（MML ch7 给出的完整推导骨架）：
1. 对 $\nabla f$ 在 $x^\*$ 处做 Taylor 展开：$\nabla f(x_k) = H_k(x_k - x^\*) + O(\|x_k-x^\*\|^2)$，其中 $H_k$ 为积分形式的平均 Hessian；
2. Newton 步相当于用当前点的 Hessian 替换平均 Hessian，误差由 Lipschitz 常数控制；
3. 结合强凸下界 $m$ 与 Lipschitz 上界 $L$ 得到压缩因子 $\frac{L}{2m}\|x_k - x^\*\|$。

**工程对应**：这就是原文档"为什么二阶方法在 DL 中不统治"的数学根源——二次收敛只在 $x_0$ 落入以 $x^\*$ 为中心的邻域（$\frac{L\|x_0-x^\*\|}{2m}<1$）时成立；深度学习的损失景观（鞍点、平坦区）不保证该邻域可达。

### 2.2 收敛率的三种刻度（key-book ch7）

key-book 第 7 章给出了统一的收敛率语言，可直接套在原文档的对比表上：

| 刻度 | 定义 | GD（凸） | GD（强凸） | Newton |
| :--- | :--- | :--- | :--- | :--- |
| 次线性 | $f(x_k)-f^\* \le O(1/k^p)$ | $p=1$ | $p=1$（带指数因子） | — |
| 线性（几何） | 误差乘 $\rho^k$，$\rho<1$ | — | $\rho = 1-\frac{m}{L}$ | — |
| 二次（超线性） | $\|e_{k+1}\| \le C\|e_k\|^2$ | — | — | ✅ |

**关键推论**（key-book 定理 7.x）：强凸光滑函数上 GD 的线性收敛率 $\rho = 1 - m/L$ 由**条件数** $\kappa = L/m$ 决定；$\kappa$ 越大收敛越慢。这解释了原文档"第02章-05 二次型·条件数——为什么 Adam 比 SGD 稳"的现象学观察：Adam 通过逐坐标缩放等效地改善了病态方向的曲率补偿。

### 2.3 拟 Newton：从 BFGS 到 L-BFGS 的数学（math-for-ai ch7.2.3）

**割线条件**：要求新的 Hessian 近似 $B_{k+1}$ 满足

$$B_{k+1}(x_{k+1}-x_k) = \nabla f(x_{k+1}) - \nabla f(x_k) \triangleq y_k$$

**BFGS 更新公式**（MML 给出完整推导）：

$$B_{k+1} = B_k + \frac{y_k y_k^\top}{y_k^\top s_k} - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k}, \quad s_k = x_{k+1}-x_k$$

- **正定性保持定理**：若 $B_k \succ 0$ 且 $y_k^\top s_k > 0$，则 $B_{k+1} \succ 0$。证明用 Sherman-Morrison 逆引理对 $B_{k+1}^{-1}$（即 Hessian 逆近似 $H_{k+1}$）直接验证二次型正性。
- **L-BFGS 的内存逻辑**：不显式存 $H_k$（$O(d^2)$），只保留最近 $m$ 组 $(s_i, y_i)$，用两次递归循环（two-loop recursion）隐式计算 $H_k \nabla f_k$，内存 $O(md)$。这就是原文档"为什么 L-BFGS 是 DL 时代唯一存活的二阶方法"的数学原因。

### 2.4 Trust Region 的收敛保证（math-forai ch7.2.4 补充）

原文档讲了 Trust Region 的机制；math-for-ai 补充了它相对线搜索的**理论优势**：

- **Cauchy 点下降引理**：即使在信赖域边界上只走最速下降方向（Cauchy 点），模型下降量也保证 $\ge \frac{1}{2}\|\nabla f_k\| \min(\Delta_k, \frac{\|\nabla f_k\|}{\beta_k})$，从而全局收敛到一阶驻点；
- **自适应半径定理**：当实际下降/预测下降比值 $\rho_k \ge \eta$（如 0.75）且触碰边界时扩大 $\Delta_{k+1} = 2\Delta_k$，失败时收缩——该策略保证 $\{\Delta_k\}$ 不会塌缩到 0，二阶收敛在 $\Delta_k \to \infty$ 后自动恢复。

---

## 三、可直接动手的实践

```python
# L-BFGS two-loop recursion 的最小实现（对照 math-for-ai ch7 伪代码）
import numpy as np

def two_loop_recursion(grad, S, Y, rhos):
    """S, Y: 最近 m 步的 (s_i, y_i) 列表; rhos_i = 1/(y_i^T s_i)"""
    q = grad.copy()
    alphas = []
    for i in reversed(range(len(S))):          # 后向循环
        alphas.append(rhos[i] * S[i] @ q)
        q -= alphas[-1] * Y[i]
    if len(Y):                                  # 初始 Hessian 缩放 γ_k = y^T s / y^T y
        gamma = Y[-1] @ S[-1] / (Y[-1] @ Y[-1])
        q *= gamma
    for i in range(len(S)):                     # 前向循环
        beta = rhos[i] * Y[i] @ q
        q += (alphas[::-1][i] - beta) * S[i]
    return -q                                   # 搜索方向
```

配套习题（选自 math-for-ai ch7 习题风格）：
1. 证明 BFGS 更新在 $y_k^\top s_k > 0$ 时保持正定（提示：对 $B_{k+1}$ 任意 $v \ne 0$ 展开 $v^\top B_{k+1} v$，分 $v^\top B_k s_k = 0$ 与 $\ne 0$ 两种情形）。
2. 对 $f(x) = \frac{1}{2}x^\top A x$（$A$ 对称正定）验证 Newton 法**一步收敛**，并说明拟 Newton 为何需要多步。
3. 用条件数 $\kappa$ 表示 GD 达到精度 $\epsilon$ 所需迭代数，并与 Newton 的 $O(\log\log(1/\epsilon))$ 对比。

---

## 四、延伸链接

- math-for-ai 第 7 章在线阅读：<https://datawhalechina.github.io/math-for-ai/#/ch07>
- key-book（收敛率章）：<https://github.com/datawhalechina/key-book>
- 本地相关章节：第06章-02（一阶方法）、第06章-05（凸优化收敛率证明）、第02章-05（条件数与 Adam）
