---
title: 推理优化 · Datawhalechina 补充
tags:
  - 推理优化
  - 量化
  - 推测解码
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/awesome-compression（量化）与 zero-to-sglang/llm-deploy（引擎），补充量化谱系、推测解码族与缓存技术的组合使用
companion_to: 第09章-推理优化/
source: https://github.com/datawhalechina/awesome-compression ; https://github.com/datawhalechina/zero-to-sglang
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 推理优化 · Datawhalechina 对照补充

> 本文件是对同目录《第09章-推理优化》（量化/推测解码/剪枝/PagedAttention/Prefix-Caching 五节）的补充，内容整理自 [awesome-compression](https://github.com/datawhalechina/awesome-compression)（量化章节的谱系）与 [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)（引擎侧缓存与调度）。本地五节按技术罗列；本补充给**技术分层与组合使用**的视角——推理优化是一个"组合拳"系统，单项最优 ≠ 整体最优。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地节 | 补充价值 |
| :--- | :--- | :--- | :--- |
| awesome-compression 量化章 | PTQ/QAT、GPTQ/AWQ 谱系 | 09-01 量化 | **核心补充**：量化方法分类学 |
| zero-to-sglang | KV cache 管理/调度 | 09-04/05 | **核心补充**：缓存技术组合 |
| llm-deploy | 服务侧优化 | 09-02（关联） | 端到端视角 |
| 本地《第10章 Model-Infra 补充》 | 算子层 | （关联） | 层次不重复 |

---

## 二、核心补充内容

### 2.1 推理优化的分层与组合矩阵

五节技术按"优化哪一层"归位，组合时按层各取一招：

| 层 | 技术（本地节） | 典型收益 | 组合注意 |
| :--- | :--- | :--- | :--- |
| 权重层 | 量化 09-01、剪枝 09-03 | 显存/带宽减半级 | 量化优先（工程成熟），剪枝需稀疏硬件支持 |
| 计算层 | 算子融合/FlashAttention（《第10章补充》） | 时延 | 与量化正交，可叠加 |
| 生成过程 | **推测解码 09-02** | 时延 2~3x（同输出分布） | 需要 draft 模型；与量化叠加时 draft/target 都可量化 |
| 显存管理 | PagedAttention 09-04 | 吞吐（并发×2~4） | 引擎自带，选引擎即选它 |
| 请求层 | Prefix caching 09-05 | 多轮/同前缀场景 TTFT 大降 | 命中率依赖路由亲和（见《0-K8s 补充》2.2） |

**组合拳默认配方**（7B~70B 级）：W4A16 量化（GPTQ/AWQ）+ 引擎（PagedAttention + continuous batching）+ 前缀缓存开启；时延敏感再叠推测解码。

### 2.2 量化谱系的选型树（awesome-compression）

```text
需要训练吗？
├─ 不训练 → PTQ
│   ├─ 权重 only：GPTQ（逐层 Hessian 误差补偿）/ AWQ（激活感知缩放）
│   │   → 显存优先选 W4；质量敏感选 W8
│   └─ 权重+激活：W8A8（SmoothQuant 迁移激活异常值）
└─ 可训练 → QAT（训练中模拟量化）
    → 精度上限最高、成本最高；与蒸馏组合（见《第19章补充》2.2）
```

**易错点**：PTQ 的 per-channel vs per-group（g128）粒度差异在大模型上显著；KV cache 量化（FP8/INT8）单独评估——它影响长上下文吞吐，且与 Prefix caching 的精度交互需要实测。

### 2.3 推测解码的适用判据与变体

推测解码 = 小 draft 模型猜多步 + 大 target 模型并行验证（拒绝采样保证输出分布不变）：

- **适用**：时延敏感、batch 小（batch 大时 GPU 已满载，验证无免费算力，收益趋零）；
- **变体谱系**：独立 draft 模型 → 自投机（Medusa 多头/M-EAGLE）→ n-gram/prompt lookup（无模型、抽取式任务收益高）；
- **验收指标**：接受长度（平均每步被采纳的 draft token 数）与端到端加速比——接受长度 <2 时应关闭该特性。

---

## 三、可直接动手的实践

1. 同一模型分别跑 GPTQ-W4 与 AWQ-W4（vLLM），对比困惑度回退与吞吐，填一张量化选型表；
2. 开/关推测解码各测一次交互式负载（batch=1）与批量负载（batch=32），验证 2.3 的适用判据；
3. 构造同前缀多请求场景，测 Prefix caching 命中率与 TTFT 改善（衔接前缀亲和路由实验）。

---

## 四、延伸链接

- awesome-compression：<https://github.com/datawhalechina/awesome-compression>
- zero-to-sglang：<https://github.com/datawhalechina/zero-to-sglang>
- 本地相关文件：第09章五节正文、LLM发展/第10章-Model-Infra-Datawhale补充.md（算子层）、第19章-知识蒸馏-Datawhale补充.md（蒸馏×量化）、工程与职业/0-K8s…/2-生产工程补充（服务侧）
