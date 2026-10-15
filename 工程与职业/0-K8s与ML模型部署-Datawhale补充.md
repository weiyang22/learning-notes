---
title: 0-K8s与ML模型部署 · Datawhalechina 补充
tags:
  - 工程与职业
  - 模型部署
  - LLM推理
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/llm-deploy、handy-ollama、zero-to-sglang、docker-notes，补充大模型推理服务化与推理引擎的部署实战线
companion_to: 0-K8s与ML模型部署.md
source: https://github.com/datawhalechina/llm-deploy ; https://github.com/datawhalechina/zero-to-sglang ; https://github.com/datawhalechina/handy-ollama ; https://github.com/datawhalechina/docker-notes
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# K8s 与 ML 模型部署 · Datawhalechina 对照补充

> 本文件是对同目录《0-K8s与ML模型部署.md》的补充。原文档已把 K8s 侧（调度/GPU 分配/服务网格）讲得比较完整；本次对照发现的缺口在**大模型推理服务化**（vLLM/SGLang/Ollama 推理引擎、量化部署、推理性能工程）——datawhale 有四条现成教程线正好补上，同时《2-生产工程与模型服务化.md》也可直接引用本文件。

---

## 一、补充来源地图

| datawhale 仓库 | 定位 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| [llm-deploy](https://github.com/datawhalechina/llm-deploy) | LLM 推理和部署理论与实践 | 全章 | **核心补充**：推理系统理论 + 部署实务 |
| [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) | SGLang 官方课程：理解推理 → 手写 mini-sglang → 读源码 → 提 PR | 第 K8s 推理服务部分 | **核心补充**：推理引擎内核视角 |
| [handy-ollama](https://github.com/datawhalechina/handy-ollama) | CPU 玩转大模型部署 | 部署章节 | 轻量/边缘部署线 |
| [docker-notes](https://github.com/datawhalechina/docker-notes) | Docker 从零到掌握 | 基础容器部分（本地假设已知） | 补基础缺口 |
| [self-llm](https://github.com/datawhalechina/self-llm) | 开源大模型 Linux 微调部署食用指南 | —（关联 LLM发展/附录B） | 各模型部署手册 |

---

## 二、核心补充内容

### 2.1 推理服务的三层视角（llm-deploy 的组织逻辑）

原文档按 K8s 资源视角组织；llm-deploy 补充了 LLM 推理特有的三层，建议并入部署架构图：

| 层 | 关注点 | 关键指标 |
| :--- | :--- | :--- |
| 引擎层（vLLM/SGLang/TGI） | KV cache 管理、PagedAttention、continuous batching、投机解码 | TTFT / TPOT / 吞吐 tokens/s |
| 服务层 | 路由、多模型池、限流、优先级队列 | QPS、排队时延、SLO 达成率 |
| 资源层（K8s） | GPU 分配、拓扑感知调度、弹性伸缩 | GPU 利用率、单卡成本 |

**LLM 侧与原文档传统 ML 部署的最大差异**：负载是"有状态的生成过程"——KV cache 显存占用随上下文动态变化，因此 HPA 不能只看 CPU/QPS，要**以 KV cache 命中率与队列深度为主信号**（vLLM/SGLang 暴露的引擎指标）。

### 2.2 SGLang 内核视角：为什么理解引擎才能调好部署（zero-to-sglang）

zero-to-sglang 的课程设计（理解推理 → 手写 mini-sglang → 读真实源码 → 落地第一个 PR）给部署工程师的增量价值：

1. **RadixAttention 前缀缓存**：相同 system prompt/前缀的请求共享 KV cache——多租户场景下 TTFT 可降一个数量级；部署时的实际含义是"**路由要把同前缀请求黏到同一实例**"；
2. **continuous batching 的调度自由度**：引擎每步可以插入/退出请求，网关的超时与重试策略必须与之匹配（盲目重试会放大排队）；
3. **量化的部署含义**：W4A16/W8A8 不仅省显存，还改变算子 bound 类型（对应本地《工程能力/第07章》roofline 方法），吞吐-质量曲线要实测。

### 2.3 CPU/边缘部署线（handy-ollama）

不是所有部署都要 GPU 池：handy-ollama 完整覆盖 Ollama 在 CPU 上的量化部署（GGUF 量化档位选择、内存估算、API 网关化）。适用位：开发/测试环境、私有化小流量场景、CI 中的模型冒烟测试——与原文档的 K8s GPU 生产线形成"双轨"，按流量与 SLO 选轨。

### 2.4 部署清单合并建议

把 datawhale 四条线的检查项并入原文档的部署 checklist：

- [ ] 引擎指标暴露（TTFT/TPOT/KV 命中率/队列深度）接入监控（衔接《5-监控与可观测性》）；
- [ ] 前缀缓存路由策略（RadixAttention 亲和性）；
- [ ] 量化档位与吞吐-质量回归测试记录；
- [ ] GPU 拓扑感知调度（NVLink 亲和，多卡张量并行）；
- [ ] 弹性策略以 KV/队列为主信号；
- [ ] CPU 轨（Ollama）用于预发冒烟，与 GPU 生产轨配置同构。

---

## 三、可直接动手的实践

1. 跑 zero-to-sglang 的 mini-sglang 作业，在单卡上复现 continuous batching 的吞吐对比（batch=1 vs 动态 batch）；
2. 用 handy-ollama 把一个 7B 模型跑在 CPU 上，记录量化档位 × 内存 × tokens/s 的实测表；
3. 按三层视角画一次自己业务推理服务的架构图，标注每层的 SLO 与监控信号（对照《5-监控与可观测性》）。

---

## 四、延伸链接

- llm-deploy：<https://github.com/datawhalechina/llm-deploy>
- zero-to-sglang：<https://github.com/datawhalechina/zero-to-sglang>（中英双语）
- handy-ollama：<https://datawhalechina.github.io/handy-ollama/>
- docker-notes：<https://github.com/datawhalechina/docker-notes>
- 本地相关文件：2-生产工程与模型服务化.md（直接引用本补充）、5-监控与可观测性.md、11-算法工程师面试全攻略.md（推理系统面试题来源）、LLM发展/第10章-Model-Infra-性能剖析与算子优化（内核优化衔接）
