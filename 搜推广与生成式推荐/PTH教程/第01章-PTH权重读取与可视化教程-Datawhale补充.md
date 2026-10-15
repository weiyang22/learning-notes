---
title: 第01章-PTH权重读取与可视化教程 · Datawhalechina 补充
tags:
  - PTH
  - PyTorch
  - 模型产物
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/thorough-pytorch（模型保存加载）与 torch-rechub（权重导出），补充 checkpoint 工程、安全加载与产物可视化
companion_to: 第01章-PTH权重读取与可视化教程.md
source: https://github.com/datawhalechina/thorough-pytorch ; https://github.com/datawhalechina/torch-rechub
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# PTH 权重读取与可视化 · Datawhalechina 对照补充

> 本文件是对同目录《第01章-PTH权重读取与可视化教程.md》的补充，内容整理自 [thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)（模型保存与加载章节）与 [torch-rechub](https://github.com/datawhalechina/torch-rechub)（推荐模型权重实践）。原文档讲 .pth 文件的读取与可视化；本补充聚焦**checkpoint 工程化**——搜推广场景里"权重怎么存、怎么安全加载、怎么检查"的完整闭环。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| thorough-pytorch | state_dict vs 整模型保存、断点续训 | 第01章正文 | **核心补充**：两种保存方式的取舍 |
| torch-rechub | 推荐模型（Embedding 巨表）的保存实践 | 第01章正文 | Embedding 分桶/稀疏权重的特殊处理 |
| torchcachegen/安全加载 | weights_only 反序列化安全 | 第01章正文（关联） | **核心补充**：加载 .pth 的安全纪律 |

---

## 二、核心补充内容

### 2.1 state_dict vs 整模型：取舍与规范（thorough-pytorch）

| 方式 | 内容 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| `torch.save(model.state_dict())` | 纯参数张量 | 与类定义解耦、可跨版本加载 | 加载需先有模型结构 |
| `torch.save(model)` 整模型 | 对象 pickle | 一行加载 | 绑定代码路径/版本，易碎 |

**规范**：训练产物一律存 state_dict + 配置快照（模型结构由配置重建），`torch.save(model)` 只用于临时调试。断点续训的完整包 = `{model, optimizer, scheduler, scaler, epoch, step, rng_state}` 七件套——缺 rng_state 会导致复现实验不可对齐。

### 2.2 安全加载纪律（PyTorch 2.x 的 weights_only）

`.pth` 本质是 pickle——**加载来源不明的 checkpoint 可以执行任意代码**。生产纪律：

```python
# 信任来源
ckpt = torch.load(path, weights_only=True)          # 只反序列化张量
# 必须加载自定义对象时（旧版兼容）
ckpt = torch.load(path, weights_only=False)         # 仅限内部可信产物 + 审计
```

外部下载的预训练权重（如 ModelScope/HF）走框架官方入口（`from_pretrained`），不走裸 `torch.load`。

### 2.3 Embedding 巨表的保存与检查（torch-rechub 视角）

搜推广模型的参数大头在 Embedding 表（对应本地《工程能力/第03章》），checkpoint 工程有特殊性：

1. **冷热分离**：model.state_dict() 里 Embedding 常占 >95% 体积——全量快照大且慢，可用"高频保存稠密部分 + 低频保存全量"的双频策略；
2. **稀疏更新一致性**：SparseAdam 的 optimizer state 与参数同构，保存/加载必须成对，否则动量丢失导致续训 loss 跳变；
3. **权重健康检查**：加载后必做的三项 sanity check——(a) 参数 shape 与结构匹配；(b) Embedding 行利用率（非零行占比，防码表/ID 映射错位）；(c) 抽查一行向量的范数量级（全零/异常大 = 加载错位信号）。

### 2.4 产物可视化清单

权重可视化的分层用途：

| 层 | 工具/方法 | 回答的问题 |
| :--- | :--- | :--- |
| 训练动力学 | checkpoint 序列的 loss/grad-norm 曲线（SwanLab/wandb） | 训练是否健康 |
| 参数分布 | 逐层 weight histogram / 范数条形图 | 哪层异常 |
| Embedding 空间 | 抽样降维散点（TensorBoard projector / umap） | 语义结构、簇质量（SID 码表检查） |
| 注意力/特征图 | hook 抓中间激活 | 模型在看什么 |

---

## 三、可直接动手的实践

1. 实现七件套断点续训，验证：中断重启后 loss 曲线与不间断训练重合；
2. 对一个含大 Embedding 的推荐模型（torch-rechub DeepFM）做双频保存，比较全量 vs 稠密部分的体积与耗时；
3. 用 projector 可视化任意 RQ-VAE 码表向量（衔接 SID 第03章），检查簇是否按类目聚拢。

---

## 四、延伸链接

- thorough-pytorch（模型保存加载章节）：<https://datawhalechina.github.io/thorough-pytorch/>
- torch-rechub：<https://github.com/datawhalechina/torch-rechub>
- 本地相关文件：基础/工程基础/编程/第05章-训练可视化与监控（可视化平台）、LLM发展/第08章-训练监控与权重续训（LLM 侧续训）、工程能力/第03章-参数量与显存预估（Embedding 体积账）
