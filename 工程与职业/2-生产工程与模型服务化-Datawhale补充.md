---
title: 2-生产工程与模型服务化 · Datawhalechina 补充
tags:
  - 工程与职业
  - 模型服务化
  - 推理服务
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/llm-deploy 与 wow-fullstack，从"服务化"视角（SLA/灰度/降级/推理网关）补充 0-K8s 补充文件的姊妹篇
companion_to: 2-生产工程与模型服务化.md
source: https://github.com/datawhalechina/llm-deploy ; https://github.com/datawhalechina/wow-fullstack
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 生产工程与模型服务化 · Datawhalechina 对照补充

> 本文件是对同目录《2-生产工程与模型服务化.md》的补充，与《0-K8s与ML模型部署-Datawhale补充.md》为姊妹篇：那份讲**部署**（引擎/资源/上线路径），本份讲**服务化**——模型上线之后的流量治理、SLA 保障与演进。内容整理自 [llm-deploy](https://github.com/datawhalechina/llm-deploy) 的服务层章节与 [wow-fullstack](https://github.com/datawhalechina/wow-fullstack)（全栈开发教程）的后端部署篇。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| llm-deploy 服务层 | 推理网关、多模型路由、SLO 定义 | 服务化架构节 | **核心补充**：LLM 服务的流量形态 |
| wow-fullstack 后端篇 | API 设计、限流、灰度发布 | 服务化通用节 | 通用服务治理模式 |
| 本地《5-监控与可观测性》 | 监控指标 | （关联，不重复） | 指标采集侧 |

---

## 二、核心补充内容

### 2.1 推理服务的 SLO 分层定义（llm-deploy）

LLM 推理的 SLA 不能只写"延迟 <100ms"，llm-deploy 给出的分层口径建议并入原文档：

| SLO 层 | 指标 | 典型目标（交互式） |
| :--- | :--- | :--- |
| 首字延迟 TTFT | 请求发出 → 首 token | P95 < 2s（流式） |
| 逐 token 延迟 TPOT | 相邻 token 间隔 | P95 < 50ms |
| 完整延迟 E2E | 按 max_tokens 估计 | 与长度契约挂钩 |
| 吞吐 | tokens/s/卡 | 容量规划用 |
| 质量护栏 | 拒答率/幻觉抽检率 | 与延迟显式交换（预算制） |

**关键纪律**：SLO 按"任务长度契约"分组（短问答/长摘要/Agent 多轮），混在一起取平均会掩盖长尾。

### 2.2 灰度与回滚（wow-fullstack 后端篇 × LLM 特化）

- **双版本影子流量**：新模型版本先接影子流量（只记录不返回），对比离线评测集 + 线上抽样质量，再逐步放量（1% → 5% → 50% → 100%）；
- **按维度切流**：按用户/请求长度/任务类型切——LLM 版本在长上下文段的表现可能与整体均值相反；
- **回滚就绪**：旧版本模型文件与引擎配置保留在线（热备），回滚 = 路由权重切回，不做重新部署；
- **配置即回滚**：温度/top-p/系统提示的变更走独立配置版本——"提示词回滚"往往比模型回滚更常用。

### 2.3 降级与过载保护

LLM 服务特有的降级阶梯（llm-deploy + 通用模式的合并）：

```text
过载时按序降级：
1. 缩短 max_tokens / 压缩系统提示（减负载）
2. 切换小参数模型（同系蒸馏版）
3. 非核心流量排队/限流（按优先级队列）
4. 兜底静态回复 + 事后补偿
```

配套熔断信号：队列深度、KV cache 命中率骤降、TPOT 突增（与《0-K8s 补充》2.1 的引擎指标一致，监控接线见《5-监控与可观测性》）。

### 2.4 服务化的版本管理矩阵

| 对象 | 版本载体 | 回滚方式 |
| :--- | :--- | :--- |
| 模型权重 | 模型仓库（版本化 tag） | 路由切回 |
| 推理引擎 | 镜像 tag | 滚动回退 |
| 提示词/采样配置 | 配置中心 | 配置版本切换 |
| 特征/检索索引 | 索引版本（快照） | 快照切回（与 SID 补充文件 2.4 呼应） |

---

## 三、可直接动手的实践

1. 为自己的推理服务写一页 SLO 表（按任务长度分组），标注当前 P95 与目标的 gap；
2. 搭一条影子流量链路：新版本模型接 5% 只记不返，产出"影子 vs 线上"的质量对比报表；
3. 实现降级阶梯的第 1、2 级（压缩提示 + 切小模型），压测验证过载下的可用性。

---

## 四、延伸链接

- llm-deploy：<https://github.com/datawhalechina/llm-deploy>
- wow-fullstack：<https://github.com/datawhalechina/wow-fullstack>
- 本地相关文件：0-K8s与ML模型部署-Datawhale补充.md（姊妹篇：部署视角）、5-监控与可观测性.md（指标接线）、1-推荐系统端到端设计-Datawhale补充.md（推荐服务特化）
