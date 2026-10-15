---
title: 第02章-Vibe-Coding全栈配置手册 · Datawhalechina 补充
tags:
  - 工具
  - Vibe-Coding
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/easy-vibe stage-3（Claude Code 核心技能），补充 MCP/Skills/Agent Teams/长任务/Spec Coding 的系统化指南与用户研究方法论
companion_to: 第02章-Vibe-Coding全栈配置手册.md
source: https://github.com/datawhalechina/easy-vibe
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Vibe Coding · Datawhalechina 对照补充

> 本文件是对同目录《第02章-Vibe-Coding全栈配置手册.md》的补充，内容整理自 [datawhalechina/easy-vibe](https://github.com/datawhalechina/easy-vibe)（vibe coding 101，19k+ stars）**Stage 3 高级开发**与附录知识库。原文档以 OpenCode 为中心的配置手册已经很完整；easy-vibe 的差异化价值在三点：① **Claude Code 原生的 Agent Teams 与长任务方法论**；② **从产品原型到 SaaS 的端到端实战路径**；③ **用户研究与需求验证方法论**（这是纯配置手册不会覆盖的"上游能力"）。

---

## 一、补充来源地图

| easy-vibe 章节 | 内容 | 对应原文档位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| stage-3/core-skills/basics | Claude Code 安装/命令/工作原理 | 第一部分 | 工具对照（OpenCode vs Claude Code） |
| stage-3/core-skills/mcp | MCP 连接 GitHub/DB/API | 第三部分 MCP 体系 | 官方 MCP 服务器清单与配置样例 |
| stage-3/core-skills/skills | 把专长封装成可复用 Skill | 第二部分 Skills 体系 | Skill 写作范式 |
| stage-3/core-skills/long-running-tasks | 让 agent 持续工作到完成 | 第五部分工作流 | **核心补充**：长任务设计模式 |
| stage-3/core-skills/agent-teams | 多实例协作如真实开发团队 | —（本地缺） | **核心补充** |
| stage-3/core-skills/superpowers | TDD + 工程级代码 | —（本地缺） | 工程质量保障 |
| stage-3/core-skills/spec-coding | 从 vibe 到 spec 驱动开发 | —（本地缺） | **核心补充**：范式升级 |
| stage-1/appendix-*（用户研究） | Double Diamond / JTBD / Mom Test | —（本地缺） | 需求验证方法论 |

---

## 二、核心补充内容

### 2.1 长任务设计模式（long-running-tasks）

easy-vibe 把"让 Claude Code 持续工作直到完成"总结为四个机制，可直接并入原文档的 Agent 配置：

1. **任务清单外置**：把目标写进仓库内的 plan 文件（如 `PLAN.md`），每完成一项就更新——agent 的上下文会被压缩，但磁盘上的计划文件始终是"外部记忆"；
2. **验收标准可执行**：验收不是自然语言而是测试命令（`pnpm test && pnpm build`），agent 每轮自检；
3. **边界明确**：明确列出"不要动的文件/目录"，防止长任务中 agent 自行扩大改动面；
4. **检查点提交**：每个里程碑自动 commit，失败可回滚——对应原文档"工作流 2"的增强。

### 2.2 Agent Teams：多实例协作（agent-teams）

easy-vibe 的 Agent Teams 模式把一个任务拆给多个并行的 Claude Code 实例，分工模式：

| 角色 | 职责 | 交互方式 |
| :--- | :--- | :--- |
| Lead | 拆解任务、review、合并 | 读所有分支的 diff |
| Coder ×N | 各领一个独立子任务 | 各自分支上工作 |
| Reviewer | 对每个 diff 出意见 | 只读 + 评论 |

**适用判据**：子任务之间**文件级隔离**（不同模块/目录）才值得并行；强耦合改动并行只会增加合并冲突。这与原文档"5. 三者协同的完整工作流"互补——原文档讲单实例内的 Agent/Skills/MCP 协同，这里讲多实例横向扩展。

### 2.3 Spec Coding：从 vibe 到规范驱动（spec-coding）

这是 easy-vibe stage-3 提出的范式升级路径，与原文档"0. Vibe Coding 是什么"形成呼应：

```text
Vibe Coding（意图驱动，适合原型/小工具）
    ↓ 项目变大、多人协作、需要回归保障
Spec Coding（规范驱动）：
    1. requirements.md —— 需求与验收标准（人写/AI 辅助）
    2. design.md —— 技术方案与数据流（AI 起草、人评审）
    3. tasks.md —— 拆解为可独立验收的任务（AI 生成）
    4. 每个 task：AI 实现 → 跑测试 → 人抽查 → 合并
```

**判断标准**：当同一个功能的第三轮"vibe 修改"开始破坏第一轮的功能时，就该升级到 Spec Coding。配套的 superpowers 技能（TDD 优先）是这个范式的质量底座：先写失败测试 → 最小实现 → 重构，AI 负责每一步的代码，人负责验收标准。

### 2.4 Skills 写作范式（skills 章节）

easy-vibe 对"什么是好的 Skill"给出的结构模板：

```markdown
---
name: my-skill
description: 何时触发（触发条件写得越具体越好）
---
# 步骤化的操作流程（命令级）
# 输出格式约定
# 常见错误与自检清单
```

与原文档"9. 自定义 Skills 的方法"互补的要点：**description 是路由依据**——agent 靠它决定何时加载该 Skill，所以触发条件必须包含具体关键词与场景，而不是抽象的"提高质量"。

### 2.5 上游能力：用户研究与需求验证（stage-1 附录）

纯工程手册往往忽略"做什么"，easy-vibe stage-1 附录补齐了这一段，四个方法可直接用于自研工具/内部产品的立项：

- **Idea 来源三渠道**：自己的工作流痛点、社区高频抱怨、竞品 review 区；
- **Double Diamond（双钻模型）**：先发散理解问题（探索域），再收敛定义问题；先发散找方案，再收敛做方案——防止"拿到第一个想法就开写"；
- **Jobs to Be Done（JTBD）**：用户"雇佣"你的产品完成什么任务？从任务而非功能列表出发定义需求；
- **The Mom Test**：访谈时问具体行为（"你上次遇到 X 是什么时候、怎么处理的"），不问观点（"你会用吗"）——避免礼貌性肯定造成的假需求。

---

## 三、可直接动手的实践

1. **长任务改造**：把原文档"工作流 1（新项目）"改造成 PLAN.md 外置 + 测试验收 + 里程碑提交的三件套，观察 agent 在同一任务上的完成率差异；
2. **双实例实验**：用 Agent Teams 模式给一个前后端分离的小项目并行开发（后端 API 与前端页面文件隔离），对比单实例耗时；
3. **Spec 升级演练**：挑一个本地反复 vibe 失败的功能，按 requirements → design → tasks 三步重做一遍。

---

## 四、延伸链接

- easy-vibe 在线阅读（中文）：<https://datawhalechina.github.io/easy-vibe/zh-cn/>
- Stage 3 目录：<https://datawhalechina.github.io/easy-vibe/zh-cn/stage-3/>
- 姊妹教程 vibe-vibe：<https://github.com/datawhalechina/vibe-vibe>
- 本地相关文件：第03章-全栈测试Agent-Skills-MCP配置手册（测试向的 MCP/Skills 配套）、self-harness（Harness Engineering 代码向）
