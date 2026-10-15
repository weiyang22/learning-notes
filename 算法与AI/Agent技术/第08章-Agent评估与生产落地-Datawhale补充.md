---
title: 第08章-Agent评估与生产落地 · Datawhalechina 补充
tags:
  - Agent
  - 评估
  - 生产落地
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/hello-agents 第十二章（性能评估）与 Extra01/Extra09/Extra10，补充评估指标体系、踩坑经验与自进化闭环
companion_to: 第08章-Agent评估与生产落地.md
source: https://github.com/datawhalechina/hello-agents
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Agent 评估与生产落地 · Datawhalechina 对照补充

> 本文件是对同目录《第08章-Agent评估与生产落地.md》的补充，内容整理自 [hello-agents](https://github.com/datawhalechina/hello-agents) **第十二章 智能体性能评估**与三篇社区精选：**Extra01（Agent 面试题总结+参考答案）、Extra09（Agent 应用开发实践踩坑与经验分享）、Extra10（Agent 自进化四类闭环）**。原文档讲评估方法论；这三篇的增量是**题库 + 一线踩坑 + 自进化机制**三个实证维度。

---

## 一、补充来源地图

| hello-agents 资源 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| ch12 性能评估 | 核心指标、基准测试、评估框架 | 第08章正文 | 指标体系系统化 |
| Extra01 面试题 + 参考答案 | Agent 岗高频题库 | 面试笔记（关联） | **核心补充**：求职题库 |
| Extra09 踩坑经验 | Code Agent 应用开发实战坑 | 第08章正文 | **核心补充**：生产事故模式 |
| Extra10 自进化 | Agent 自进化四类闭环 | 第07章补充文件（关联） | 进化机制分类学 |

---

## 二、核心补充内容

### 2.1 评估的三层指标体系（hello-agents ch12）

ch12 把 Agent 评估组织为三层，比"只看任务成功率"精细：

| 层 | 指标 | 说明 |
| :--- | :--- | :--- |
| 最终结果层 | 任务成功率 / 完成度评分 | 端到端结果，最真实但方差大、反馈慢 |
| 过程行为层 | 步数效率、工具调用正确率、重复动作率、死循环率 | 定位"为什么失败"——很多失败不是模型差而是编排差 |
| 成本时延层 | tokens 消耗、时延分位（P50/P95）、每任务成本 | 上线门槛指标 |

**配套基准**：ch12 梳理了 GAIA（通用助手）、SWE-bench（代码）、WebArena/WebVoyager（网页操作）等基准的适用域；选基准先看**任务分布是否匹配自己的业务形态**。

**评估框架**：深拆了"LLM-as-judge"的偏差控制（位置偏差、长度偏差、自偏好）——判分 prompt 要做位置交换校验与锚定样例校准，否则 judge 分数不可信。

### 2.2 生产踩坑模式清单（Extra09）

Extra09 的踩坑经验可归纳为高频事故模式，建议作为原文档"生产落地"节的 checklist：

1. **上下文污染**：工具返回的大段 JSON 挤占上下文 → 中途"忘记"系统约束（对策：返回值裁剪，见第07章补充文件预算表）；
2. **循环重试风暴**：失败任务无限重试烧钱（对策：步数上限 + 退避 + 人工升级通道）；
3. **权限过大**：Agent 持有全量凭证 → 一次幻觉操作删库（对策：最小权限 + 危险操作二次确认 + 审计日志）；
4. **评测集过拟合**：反复在固定测试集上调 prompt → 线上分布一变就崩（对策：保留滚动隐藏集）；
5. **静默失败**：任务"完成"但产物为空/格式错（对策：产物 schema 校验作为验收闭环）。

### 2.3 Agent 面试题库（Extra01）

Extra01 提供了 Agent 岗位的高频面试题与参考答案（含框架选型、记忆设计、多智能体分工、评估方法等主题），与本地《面试笔记》的关系：

- 《面试笔记/0-LLM面试笔记-2026》可把 Extra01 作为 Agent 专题的补充题源；
- 题目按 hello-agents 章节组织，答错可回跳对应章节复习——是"评估自己 Agent 知识体系"的现成卷子。

### 2.4 自进化四类闭环（Extra10）

Extra10 把"Agent 自进化"分为四类闭环，给原文档的"持续改进"提供分类学：

| 闭环 | 进化对象 | 机制 |
| :--- | :--- | :--- |
| Prompt 进化 | 提示词 | 失败案例回灌改写 prompt |
| Skill 进化 | 技能库 | 成功轨迹沉淀为可复用 Skill |
| 工具进化 | 工具集 | 自动生成/组合新工具 |
| 模型进化 | 权重 | 轨迹→SFT/RL（衔接 LLM发展/第05章补充文件） |

判读：越靠前的闭环成本低、见效快、风险小；模型进化是最后手段。与第07章补充文件的"记忆复用率统计"形成联动——被高频复用的 Skill 是进化成功的信号。

---

## 三、可直接动手的实践

1. 用 ch12 的三层指标为自己的 Agent 建一次基线报表（成功率/死循环率/P95 时延/每任务成本）；
2. 对照 2.2 的五类事故模式审计现有系统，每类写出"是否发生过 + 现有防线"两栏；
3. 做一次 Extra01 的模拟面试（抽 10 题），错题回跳对应章节并记入本地错题本。

---

## 四、延伸链接

- hello-agents ch12：<https://github.com/datawhalechina/hello-agents/tree/main/docs/chapter12>
- Extra01 题目/答案：<https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra01-面试问题总结.md>
- Extra09/Extra10：同仓库 Extra-Chapter 目录
- 本地相关文件：第07章-记忆与上下文工程-Datawhale补充.md（预算表联动）、第02章-RAG检索增强生成-Datawhale补充.md（RAG 评估指标并入）、面试/面试笔记/0-LLM面试笔记-2026-Datawhale补充.md（题库合并）
