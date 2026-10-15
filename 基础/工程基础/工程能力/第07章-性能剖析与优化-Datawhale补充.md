---
title: 第07章-性能剖析与优化 · Datawhalechina 补充
tags:
  - 工程
  - 性能优化
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/thorough-pytorch 第七章可视化与 datawhalechina/llm-algo-leetcode Triton/CUDA 优化，补充 profiler 工具链与 kernel 级优化实战
companion_to: 第07章-性能剖析与优化.md
source: https://github.com/datawhalechina/thorough-pytorch ; https://github.com/datawhalechina/llm-algo-leetcode
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 性能剖析与优化 · Datawhalechina 对照补充

> 本文件是对同目录《第07章-性能剖析与优化.md》的补充，内容整理自 [datawhalechina/thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)（深入浅出 PyTorch）**第七章 可视化**与 [datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)（大模型算法与系统教程）**03 Triton Kernels / 04 CUDA and System Optimization**。原文档已有完整的剖析方法论；本补充聚焦两点：①工具链的标准化用法，②从算子到 kernel 的底层优化实战——这是原文档相对薄弱的系统级部分。

---

## 一、补充来源地图

| datawhale 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| thorough-pytorch 第七章 | TensorBoard / wandb / SwanLab 三件套 | 第05章（训练可视化）+ 本章 | 可视化定位瓶颈的标准姿势 |
| llm-algo-leetcode 02 PyTorch_Algorithms | 算子复杂度分析、显存工程 | 第02、03章 | FLOPs/内存访问量化习题 |
| llm-algo-leetcode 03 Triton_Kernels | Triton 编写高性能 kernel | 本章（缺失部分） | **核心补充**：从 Python 算子到 GPU kernel |
| llm-algo-leetcode 04 CUDA_System_Optimization | CUDA 系统级优化、benchmark 方法 | 本章（缺失部分） | 带宽/占用率/warp 效率的度量 |
| llm-algo-leetcode 05 CUDA_Rust | Rust 加速与跨语言绑定 | 本章（缺失部分） | 替代加速路径 |

---

## 二、核心补充内容

### 2.1 Profiler 工具链的标准三步法（thorough-pytorch 第七章）

**第 1 步：torch.profiler 拿到算子级时间线**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")   # 可导入 chrome://tracing 看时间线
```

关注三列：`Self CUDA %`（算子耗时占比）、`CUDA Mem %`（显存峰值来源）、`Calls`（高频小算子 = kernel launch 开销嫌疑）。

**第 2 步：TensorBoard 定位时间线空洞**

`prof.export_chrome_trace` 的时间线中，GPU 空闲段 = host 侧瓶颈（数据加载/同步/Python 开销）。thorough-pytorch 第七章强调的检查顺序：**先看 GPU 利用率（nvidia-smi / DCGM），再看时间线空洞，最后才看单算子优化**。

**第 3 步：SwanLab/wandb 做跨 run 对比**

把每次优化的 step time / 显存峰值 / tokens per second 记录为标量曲线，用 wandb Sweep 或 SwanLab 对比视图做 A/B。避免"优化后感觉变快"的不可复现结论。

### 2.2 从算子到 kernel：Triton 优化路径（llm-algo-leetcode 03）

llm-algo-leetcode 的 Triton 系列给出了 DL 算子手写的标准工作流，以 **fused softmax** 为例（这是它 03 章的第一个完整案例）：

```python
import triton, triton.language as tl, torch

@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                    # 数值稳定（对应本地第08章 LogSumExp）
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + cols, e / s, mask=mask)
```

**该工作流的四个度量维度**（对应 04 章的 benchmark 方法论）：

| 维度 | 指标 | 工具 |
| :--- | :--- | :--- |
| 计算强度 | FLOPs / Byte（算术强度） | 手算 + `torch.cuda.Event` 计时 |
| 带宽利用率 | 实测 GB/s ÷ 峰值 GB/s | `triton.testing.do_bench` |
| 占用率 | 每 SM 常驻 warp 数 | `nsight-compute`（ncu） |
| 访存模式 | 全局内存合并度 / bank conflict | `ncu` 的 Memory Workload Analysis |

**经验法则**（03 章反复强调）：元素级/归约类算子是 **memory-bound**，优化目标是打满带宽（>80% 峰值）；GEMM 类是 **compute-bound**，优化目标是 Tensor Core 利用率。先判断 bound 类型，再决定优化方向——这一步判断比任何技巧都重要。

### 2.3 CUDA 系统级优化清单（llm-algo-leetcode 04）

04 章把系统级优化整理为可核对的清单，可直接并入原文档的优化 checklist：

1. **Kernel 融合**：把逐算子的多次 global memory 往返合并为一次（如 LayerNorm + Dropout + 残差加和融合）；
2. **消除同步点**：`.item()`、`print(tensor)`、`cpu()` 都隐式同步；训练循环内用 detach 的 GPU 侧指标累积；
3. **内存池与碎片**：`torch.cuda.empty_cache()` 只在评估阶段切换时调用；高频变长序列用 `expandable_segments:True`；
4. **通信与计算重叠**：DDP 的 bucket 化 AllReduce 与反向传播重叠（`bucket_cap_mb` 调参）；
5. **精度策略**：bf16 优先于 fp16（无 loss scale）；权重用 fp32 master copy（对应本地第08章数值稳定性）。

### 2.4 显存工程速查（llm-algo-leetcode 02 + 本地第03章联动）

- 激活重计算（gradient checkpointing）：用 ~30% 额外计算换 ~60% 激活显存，只对Transformer block 级别开启，不要逐层细粒度；
- ZeRO 分档的显存收益估算公式：`参数显存 × (1 - 分片度)` 的三个档位对照，详见 llm-algo-leetcode `02_PyTorch_Algorithms/` 的推导题；
- Embedding 巨表（搜推广场景）：`SparseAdam` + 分片 optimizer state 是显存大头，对应本地第01章的稀疏更新讨论。

---

## 三、可直接动手的实践

1. **复现 fused softmax benchmark**：用 `triton.testing.do_bench` 对比 `torch.softmax` 与手写 Triton kernel 在 `[4096, 4096]`、`[16384, 128]` 两种形状下的带宽利用率，验证 memory-bound 判断；
2. **profiler 三步法走一遍本地训练**：对本地任意训练脚本（如 PTH 教程的 DLRM 小例）跑 `torch.profiler`，找出 CUDA 时间占比 top-3 算子，判断是否值得融合；
3. **习题**（llm-algo-leetcode 02 章风格）：估算 `[B=32, T=2048, D=4096]` 的 Transformer 一层在前向/反向/重计算三种模式下的激活显存与 FLOPs。

---

## 四、延伸链接

- thorough-pytorch 在线阅读：<https://datawhalechina.github.io/thorough-pytorch/>
- llm-algo-leetcode 仓库：<https://github.com/datawhalechina/llm-algo-leetcode>
- 本地相关章节：第02章-时间与空间复杂度分析、第03章-参数量与显存预估、编程/第05章-训练可视化与监控
