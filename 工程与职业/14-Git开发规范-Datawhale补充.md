---
title: 14-Git开发规范 · Datawhalechina 补充
tags:
  - 工程与职业
  - Git
  - 协作规范
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/faster-git lecture05/06/07，补充 Git 内部原理、GitFlow 工作流实战与提交规范的可操作细则
companion_to: 14-Git开发规范.md
source: https://github.com/datawhalechina/faster-git
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Git 开发规范 · Datawhalechina 对照补充

> 本文件是对同目录《14-Git开发规范.md》的补充，内容整理自 [faster-git](https://github.com/datawhalechina/faster-git)（Datawhale Git 教程）**lecture05 Git 内部原理、lecture06 GitFlow 工作流实战、lecture07 提交规范**。原文档（与《基础/工具与配置/第01章-Git完全教程》配套）覆盖命令与流程；faster-git 的差异化价值在**规范背后的"为什么"**：对象模型解释一切行为、工作流的适用边界、提交信息的信息论价值。

---

## 一、补充来源地图

| faster-git 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| lecture05 内部原理 | 对象模型（blob/tree/commit/tag）、引用、reflog | 第01章-Git完全教程（关联） | **核心补充**：用对象模型解释 rebase/reset 行为 |
| lecture06 GitFlow | master/develop/feature/release/hotfix 五分支 | 工作流节 | **核心补充**：分支模型与发布节奏的耦合 |
| lecture07 提交规范 | Conventional Commits、commit message 结构 | 提交规范节 | 类型/作用域/正文的标准化 |
| lecture08~09 | GitHub/Gitee、可视化工具 | 协作节 | 工具链 |
| lecture10 | 团队协作与 diff 工具 | Code Review 节 | diff 工具配置 |

---

## 二、核心补充内容

### 2.1 用对象模型解释"危险操作"（faster-git lecture05）

Git 的一切行为可从四类对象推导，这层理解让规范不再是死记：

```text
blob（文件内容快照）← tree（目录结构）← commit（指向 tree + 父提交）← tag/branch（指向 commit 的引用）
```

- **reset 三模式的对象论解释**：`--soft` 只动分支引用；`--mixed` 再重置 index；`--hard` 再重置工作区——三者动的是同一个 commit 链的"引用指针"，旧 commit 对象仍在，靠 `reflog` 找回（这是"reset 不可怕"的底气）；
- **rebase vs merge 的本质**：rebase 复制 commit 链（新对象、干净线性历史），merge 保留原链 + 合并提交——**已推送的共享分支禁 rebase** 的原因：他人基于旧对象的引用全部悬空；
- **squash 的适用判据**：PR 内的 WIP 碎提交合并为一；跨人共享的提交不合。

### 2.2 GitFlow 与简化工作流的适用边界（faster-git lecture06）

| 工作流 | 分支集 | 适用 | 不适用 |
| :--- | :--- | :--- | :--- |
| GitFlow | master + develop + feature + release + hotfix | 版本化发布（App/SDK/私有化交付） | Web 持续部署（过重） |
| GitHub Flow | main + feature/PR | 持续部署 SaaS（本库"推荐默认"） | 多版本并存维护 |
| Trunk-Based | main + 短命分支 + feature flag | 高频部署大团队 | 依赖功能分支评审的团队 |

**hotfix 纪律**（GitFlow 特有）：从 master 拉 hotfix 分支，修完同时合回 master 与 develop——漏合 develop 是 hotfix 最常见事故，应写进团队 checklist。

### 2.3 提交规范的可操作细则（faster-git lecture07）

Conventional Commits 的落地格式与本地规范的衔接：

```text
<type>(<scope>): <subject>          # 标题行 ≤50 字，祈使句，不加句号
                                    # 空行
<body>                              # 为什么改（why > what），≤72 字/行
<footer>                            # BREAKING CHANGE: / Closes #123
```

- **type 清单**：feat / fix / docs / style / refactor / perf / test / chore——type 决定 CHANGELOG 归类与版本号（feat→minor，fix→patch）；
- **squash 合并时的标题继承**：PR squash 后的 commit 标题取 PR 标题，因此 **PR 标题也按规范写**；
- **自动化**：`commitlint` + `husky` 提交钩子拦截不合规范信息，`semantic-release` 按 type 自动升版本——规范靠工具守而非自觉。

---

## 三、可直接动手的实践

1. 做一次 `reset --hard` 事故演习：写两笔提交 → hard reset → 用 reflog 找回，体验对象模型的可恢复性；
2. 给自己的仓库配 commitlint + husky，故意提交不规范信息验证拦截；
3. 画出自己团队的分支模型图（对照 2.2 表选型），标注 hotfix 的双向合并点。

---

## 四、延伸链接

- faster-git 在线阅读：<https://datawhalechina.github.io/faster-git/>
- 本地相关文件：基础/工具与配置/工具手册/第01章-Git完全教程.md（命令手册）、工程与职业/6-代码质量与工程实践.md（CI/评审衔接）、第03章-全栈测试Agent-Skills-MCP配置手册.md（GitHub MCP 自动化）
