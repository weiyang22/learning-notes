---
title: 第10章-Model-Infra-性能剖析与算子优化 · Datawhalechina 补充
tags:
  - LLM
  - Infra
  - CUDA
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/llm-algo-leetcode CUDA 算子线与 zero-to-sglang 内核线，补充从算子手写到推理引擎的 Infra 学习路径
companion_to: 第10章-Model-Infra-性能剖析与算子优化.md
source: https://github.com/datawhalechina/llm-algo-leetcode ; https://github.com/datawhalechina/zero-to-sglang
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Model-Infra 性能剖析与算子优化 · Datawhalechina 对照补充

> 本文件是对同目录《第10章-Model-Infra-性能剖析与算子优化.md》的补充。与《工程能力/第07章-性能剖析补充》（profiler 方法论层）互补，本份聚焦**算子层**：内容整理自 [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)（大模型算法岗之 CUDA 算子：Triton/CUDA 专题）与 [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)（引擎内核）的衔接段——从"手写一个算子"到"算子如何被引擎调度"。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| llm-algo-leetcode 04 CUDA | 算子开发：向量加法→reduction→GEMM 优化阶梯 | 算子优化节 | **核心补充**：优化阶梯讲义 |
| llm-algo-leetcode 03 Triton | Triton 融合算子 | 算子优化节 | 融合视角（第07章补充已引，此处算子细节） |
| zero-to-sglang 内核章 | PagedAttention/RadixAttention 内核 | 引擎层 | **核心补充**：算子→引擎的桥 |
| 本地《工程能力/第07章补充》 | roofline/profiler 三步法 | （方法论层，不重复） | 互引 |

---

## 二、核心补充内容

### 2.1 CUDA GEMM 优化阶梯（llm-algo-leetcode 04）

CUDA 算子教学的经典路径，每级解决上一级的瓶颈（与 roofline 定位呼应）：

```text
L0 朴素 GEMM            —— 每元素全局读，带宽浪费严重
L1 全局内存合并访问      —— coalescing：相邻线程读相邻地址
L2 共享内存分块(tiling)  —— 块级复用，全局读取降 O(tile) 倍
L3 寄存器微tile          —— 每线程算 8×8 微块，共享内存读取再降
L4 双缓冲/异步拷贝       —— 拷贝与计算流水重叠，隐藏延迟
L5 Tensor Core/wmma     —— 半精度矩阵单元，算力跃迁
```

**面试用法**：能讲出"每级解决什么瓶颈 + 用什么指标验证"（内存带宽占比→共享内存 bank conflict→流水重叠率→TC 利用率）即可对标 Infra 岗手撕题。

### 2.2 从算子到引擎：Attention 内核的两个里程碑（zero-to-sglang 内核章）

原文档讲引擎全景；两个内核把"算子优化"与"引擎能力"连起来：

1. **FlashAttention**：IO 感知分块 + online softmax——不存 N×N 注意力矩阵，显存 O(N) 化；它是"算法-算子协同设计"的范式（数学等价变换换 IO）；
2. **PagedAttention**：KV cache 按 block 分页管理——解决显存碎片与并发调度，是 vLLM/SGLang 吞吐优势的根；RadixAttention 再加前缀树复用（与《0-K8s 补充》2.2 呼应）。

**学习路径建议**：手写 L1~L3 GEMM（懂硬件）→ 读 FlashAttention 论文与简化实现（懂算法-算子协同）→ 读 PagedAttention 数据结构（懂引擎）——三级递进恰好覆盖 Infra 面试的三个深度档。

### 2.3 算子优化的验证纪律

llm-algo-leetcode 的作业式训练给出的验证闭环，可直接并入原文档规范：

1. **正确性先行**：与 PyTorch 参考实现逐元素比对（容差按精度档设）；
2. **基准三件套**：时间（CUDA event 计时，排除拷贝）、带宽/算力占比（roofline 定位）、与 cuBLAS/官方内核的比值（"达到 cuBLAS 的 X%"是标准汇报口径）；
3. **多形状扫描**：性能要在形状矩阵上测（小/中/大、方/长条），单点最优常是假象。

---

## 三、可直接动手的实践

1. 按 2.1 阶梯实现到 L3（tiling + 微 tile），报告相对 cuBLAS 的性能百分比与每级的带宽占比变化；
2. 用 Triton 写 fused softmax（衔接《第07章补充》），并迁移到 fused LayerNorm+残差；
3. 读 zero-to-sglang 的 mini-attention 内核，画出"PagedAttention 的 block 表 → 物理显存"的映射图。

---

## 四、延伸链接

- llm-algo-leetcode：<https://github.com/datawhalechina/llm-algo-leetcode>
- zero-to-sglang：<https://github.com/datawhalechina/zero-to-sglang>
- 本地相关文件：工程能力/第07章-性能剖析与优化-Datawhale补充.md（方法论层）、第10章正文（Infra 全景）、工程与职业/0-K8s与ML模型部署-Datawhale补充.md（引擎部署侧）、手撕/第09章-推理优化
