---
title: LeetCode 专题刷题法 · Datawhalechina 补充
tags:
  - 面试
  - LeetCode
  - 刷题
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/leetcode-notes（分类题解）与 huawei-od-python，补充按专题的刷题路线、Hot 100 打法与错题闭环
companion_to: 第10章-LeetCode实战刷题集/
source: https://github.com/datawhalechina/leetcode-notes ; https://github.com/datawhalechina/huawei-od-python
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# LeetCode 专题刷题法 · Datawhalechina 对照补充

> 本文件是对同目录《第10章-LeetCode实战刷题集》的补充，内容整理自 [leetcode-notes](https://github.com/datawhalechina/leetcode-notes)（Datawhale 算法题解笔记，15k stars：按**分类标签**组织的题解体系）与 [huawei-od-python](https://github.com/datawhalechina/huawei-od-python)。本地第10章是刷题集；本补充给**刷题法的元层**——顺序、节奏与闭环机制（题单机制与《第09章 算法题库扩展补充》2.1 同款）。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| leetcode-notes | 按标签分类题解（数组/双指针/哈希/栈队列/树/回溯/DP/图…） | 第10章各专题 | **核心补充**：分类题解库 |
| leetcode-notes 热题 | Hot 100 / 剑指 Offer 系列 | 第06/07章高频题 | **核心补充**：冲刺题单 |
| huawei-od-python | 场景化题 | 第10章（补充） | 与《第09章 补充》联动 |

---

## 二、核心补充内容

### 2.1 四阶段刷题路线（leetcode-notes 分类 × 本地章节）

| 阶段 | 目标 | 题源 | 对应本地章 |
| :--- | :--- | :--- | :--- |
| ① 按标签打地基 | 每标签 10~15 题，掌握模板 | leetcode-notes 标签目录 | 第01~05章 |
| ② Hot 100 过筛 | 高频题限时做（25~40 分钟/题） | leetcode-notes Hot 100 | 第06/07章 |
| ③ 弱项专项 | 错题本统计的薄弱标签二刷 | 自建错题单 | 第08章 |
| ④ 模拟实战 | 全真模拟（2 题/60 分钟） | OD 场景题 + 剑指 | 第09章及题库扩展 |

**节奏纪律**：每天 2~3 题 + 周末一次限时模拟；新题先想 20 分钟再看题解（看了就抄 = 无效刷题），看完题解必须关掉重写。

### 2.2 按标签的模板清单（leetcode-notes 组织方式的精华）

leetcode-notes 按标签组织；每个标签收一套"模板 + 变体识别口诀"：

| 标签 | 核心模板 | 变体识别口诀 |
| :--- | :--- | :--- |
| 双指针 | 对撞/快慢 | "有序数组找对" → 对撞；"环形/中点" → 快慢 |
| 滑动窗口 | 可变窗口 + 计数器 | "最长/最短子串满足…" |
| 二分 | 左闭右开统一模板 | "有序 + 找边界" → 想二分 |
| 回溯 | 路径+选择列表+撤销 | "所有组合/排列/子集" |
| DP | 状态定义→转移→初始化→遍历序 | "最值/可行/计数" 三问 |
| 树 | 递归三段式 / 层序队列 | "路径相关" 想前缀和+回溯 |
| 图 | BFS 层序 / 拓扑 / 并查集 | "依赖顺序" 想拓扑 |

**模板卡的用法**：每卡背面写 2 道代表题 + 自己的易错点；考前只翻卡不看题——这与《附录B-代码模板大全》的定位互补（那是代码库，这是识别口诀）。

### 2.3 与本地其他题库的分工

| 题库 | 定位 | 用在哪个阶段 |
| :--- | :--- | :--- |
| leetcode-notes | 分类学习 + 热题冲刺 | ①②③ |
| huawei-od-python | 场景化工程题 | ④ 模拟实战 |
| 本地第06/07章 高频题 | 面向目标公司 | ② 交叉使用 |
| 本地第08章 错题本 | 闭环枢纽 | 全程 |

---

## 三、可直接动手的实践

1. 建立四阶段进度表（标签 × 状态），本周先完成阶段①的两个弱标签；
2. 制作 10 张模板卡（2.2 表格展开），每张背面填代表题与易错点；
3. 本周末做一次阶段④模拟（2 题 60 分钟，用 OD 场景题），复盘入错题本。

---

## 四、延伸链接

- leetcode-notes：<https://github.com/datawhalechina/leetcode-notes>
- huawei-od-python：<https://github.com/datawhalechina/huawei-od-python>
- 本地相关文件：第06/07章高频题、第08章-所有习题解析、第09章…/算法题库扩展-Datawhale补充.md（题单闭环机制）、面试笔记/2-算法工程师通用面试-2026
