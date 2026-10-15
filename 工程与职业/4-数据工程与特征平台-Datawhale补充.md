---
title: 4-数据工程与特征平台 · Datawhalechina 补充
tags:
  - 工程与职业
  - 数据工程
  - 特征平台
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/joyful-pandas、hands-on-data-analysis、juicy-bigdata，补充 pandas 高级操作、数据分析方法轮与大数据技术栈
companion_to: 4-数据工程与特征平台.md
source: https://github.com/datawhalechina/joyful-pandas ; https://github.com/datawhalechina/hands-on-data-analysis ; https://github.com/datawhalechina/juicy-bigdata
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 数据工程与特征平台 · Datawhalechina 对照补充

> 本文件是对同目录《4-数据工程与特征平台.md》的补充，内容整理自 [joyful-pandas](https://github.com/datawhalechina/joyful-pandas)（ pandas 中文教程）、[hands-on-data-analysis](https://github.com/datawhalechina/hands-on-data-analysis)（动手学数据分析）与 [juicy-bigdata](https://github.com/datawhalechina/juicy-bigdata)（大数据处理导论）。原文档讲特征平台的架构与搜推广特征体系；三份资料补的是**底层技能**：pandas 高级操作、分析方法论、Hadoop/Spark 技术栈——特征平台的实现层地基。

---

## 一、补充来源地图

| 来源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| joyful-pandas | 分组/变形/连接/缺失/文本/时序六大专题 | 特征加工实现层 | **核心补充**：特征工程的 pandas 习语 |
| hands-on-data-analysis | 项目制：数据加载→清洗→可视化→建模 | 分析方法论节 | **核心补充**：分析方法轮（Datawhale 风格） |
| juicy-bigdata | Hadoop/HDFS/MapReduce/Spark/Hive/Flink | 大数据栈节 | **核心补充**：平台基础设施层 |

---

## 二、核心补充内容

### 2.1 特征加工的 pandas 习语（joyful-pandas）

特征平台离线侧的日常是 pandas/SQL；joyful-pandas 的高级专题对应特征加工的高频操作：

| 特征操作 | pandas 习语 | 坑 |
| :--- | :--- | :--- |
| 聚合统计特征（用户近 N 天 CTR） | `groupby + rolling / expanding` | 时序泄漏：rolling 窗口必须 exclude 当前样本 |
| 类目交叉 | `merge` / `MultiIndex` / `pivot_table` | 内存爆炸：先做基数检查 |
| 序列特征（行为序列截断/聚合） | `groupby.agg(list)` + `explode` | list 列不支持向量化，量大时换 polars/Spark |
| 缺失与异常 | `fillna` 策略 / `clip` / 分位数盖帽 | 训练-推理的填充值必须同源（特征平台管版本） |
| 时序特征 | `dt` 访问器 / 周期编码（sin/cos） | 时区与对齐粒度 |

**与特征平台架构的衔接**：以上操作的"训练-推理一致性"风险正是特征平台存在的理由——计算逻辑进平台统一维护，别散落在 notebook。

### 2.2 数据分析方法轮（hands-on-data-analysis）

hands-on-data-analysis 以泰坦尼克等经典项目贯穿"提出问题 → 数据清洗 → EDA → 特征 → 建模 → 结论"的完整轮次，对原文档的价值：

- **EDA 清单化**：单变量分布 / 缺失画像 / 相关性热力图 / 目标关联——EDA 产出直接决定特征候选，这是特征平台上游的"需求来源"；
- **清洗的可追溯性**：每步清洗保留中间版本，对应特征平台的数据血缘（lineage）思想。

### 2.3 大数据技术栈地图（juicy-bigdata）

原文档的平台架构落到基础设施层时，juicy-bigdata 提供系统讲义：

```text
存储：HDFS（三副本/块结构）→ 对象存储（OSS/S3）演进
计算：MapReduce（原理课）→ Spark（内存 DAG / shuffle 机制）→ Flink（流式，对应在线特征）
数仓：Hive（表/分区/分区裁剪）→ 湖（Iceberg/Hudi，流批一体）
调度：DAG 编排（Airflow/DolphinScheduler）
```

**与搜推广特征的对应**：离线特征 = Spark/Hive 批产 → 写入在线存储；实时特征 = Flink 消费行为流 → 落 KV；本地原文档的"离线/近线/在线"三层正好挂在这张栈图上。

---

## 三、可直接动手的实践

1. 用 joyful-pandas 的 rolling 习语实现"用户近 7 天 CTR"特征，并写一个泄漏检测断言（特征值不包含 label 时间之后的信息）；
2. 按 hands-on 的 EDA 清单对一份曝光日志做完整 EDA，产出特征候选清单并排序；
3. 在本地/集群跑一个 Spark word-count 级作业，对照 juicy-bigdata 讲义观察 shuffle 行为，理解为什么宽依赖决定作业划分。

---

## 四、延伸链接

- joyful-pandas 在线阅读：<https://datawhalechina.github.io/joyful-pandas/>
- hands-on-data-analysis：<https://github.com/datawhalechina/hands-on-data-analysis>
- juicy-bigdata：<https://github.com/datawhalechina/juicy-bigdata>
- 本地相关文件：4-数据工程与特征平台.md 正文（架构层）、1-推荐系统端到端设计（数据流全景）、5-监控与可观测性（特征监控）、基础/工程基础/编程/第02章-搜推广场景Python工程底座
