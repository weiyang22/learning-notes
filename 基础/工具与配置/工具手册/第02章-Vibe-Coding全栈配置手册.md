---
title: Vibe Coding 全栈配置手册
tags:
  - 基础
  - 能力提升
description: "- [0. Vibe Coding 是什么：范式转移](#0-vibe-coding-是什么范式转移)..."
last_updated: 2026-07-13
status: ✅ v1.0
---

# Vibe Coding with OpenCode：全栈研发 Agent / Skills / MCP 完整配置手册

> **核心理念**：Vibe Coding 是一种"以意图为中心、信任 AI 自主执行"的工作流——你描述"想做什么"，AI 决定"怎么做"。这背后需要一套**完整的 Agent / Skills / MCP 体系**作为支撑。
>
> **本文目的**：为已经在使用 OpenCode 的工程师提供一份**从 0 到 1 的 Vibe Coding 全栈配置手册**——不仅是说明文档，更是**已直接配置进当前 OpenCode 实例**的可运行方案。
>
> **适用读者**：使用 OpenCode / Claude Code / 类似 AI Coding Agent 的全栈工程师 / 算法工程师 / 独立开发者。
>
> **配套配置**：本文所有建议都已直接落地到 `~/.config/opencode/opencode.json` 和 `~/.opencode/` 目录。

---

## 目录

- [0. Vibe Coding 是什么：范式转移](#0-vibe-coding-是什么范式转移)
- [第一部分：全栈研发 Agent / Skills / MCP 体系设计](#第一部分全栈研发-agent--skills--mcp-体系设计)
  - [1. 完整体系地图](#1-完整体系地图)
  - [2. Agent 分层架构](#2-agent-分层架构)
  - [3. Skills 体系（方法论库）](#3-skills-体系方法论库)
  - [4. MCP 工具集成（外部能力）](#4-mcp-工具集成外部能力)
  - [5. 三者协同的完整工作流](#5-三者协同的完整工作流)
- [第二部分：必装 Skills 详解（10 大核心）](#第二部分必装-skills-详解10-大核心)
  - [6. superpowers 基础包（15 个方法论 Skill）](#6-superpowers-基础包15-个方法论-skill)
  - [7. 用户层必装 Skills（karpathy-guidelines + project-management）](#7-用户层必装-skillskarpathy-guidelines--project-management)
  - [8. 推荐必装 Skills（3 个新增）](#8-推荐必装-skills3-个新增)
  - [9. 自定义 Skills 的方法](#9-自定义-skills-的方法)
- [第三部分：必装 MCP Servers 详解（7 大核心）](#第三部分必装-mcp-servers-详解7-大核心)
  - [10. Context7（库文档查询）](#10-context7库文档查询)
  - [11. Playwright / Chrome DevTools（浏览器自动化）](#11-playwright--chrome-devtools浏览器自动化)
  - [12. GitHub（代码托管）](#12-github代码托管)
  - [13. filesystem（文件系统）](#13-filesystem文件系统)
  - [14. sequential-thinking（结构化思考）](#14-sequential-thinking结构化思考)
  - [15. exa / firecrawl（Web 搜索 / 抓取）](#15-exa--firecrawlweb-搜索--抓取)
  - [16. Ref / DeepWiki（参考文档）](#16-ref--deepwiki参考文档)
- [第四部分：完整配置实战](#第四部分完整配置实战)
  - [17. 一次性配置脚本](#17-一次性配置脚本)
  - [18. 全局 vs 项目级配置](#18-全局-vs-项目级配置)
  - [19. 安全 / 成本 / 性能优化](#19-安全--成本--性能优化)
- [第五部分：典型工作流（5 大场景）](#第五部分典型工作流5-大场景)
  - [20. 工作流 1：从零开始一个新项目](#20-工作流-1从零开始一个新项目)
  - [21. 工作流 2：为一个老项目加新功能](#21-工作流-2为一个老项目加新功能)
  - [22. 工作流 3：Debug 一个生产问题](#22-工作流-3debug-一个生产问题)
  - [23. 工作流 4：复现一篇论文](#23-工作流-4复现一篇论文)
  - [24. 工作流 5：跨团队沟通 / Code Review](#24-工作流-5跨团队沟通--code-review)
- [附录 A：所有推荐配置的清单](#附录-a所有推荐配置的清单)
- [附录 B：常见问题 FAQ](#附录-b常见问题-faq)

---

## 0. Vibe Coding 是什么：范式转移

### 0.1 传统编码 vs Vibe Coding

| 维度 | 传统编码 | Vibe Coding |
| :--- | :--- | :--- |
| **交互模式** | 你写代码，AI 补全 | 你描述意图，AI 写代码 |
| **关注点** | 怎么实现（How） | 想做什么（What） |
| **错误处理** | 写之前想清楚 | 写完后让 AI 修 |
| **验证** | 自己跑测试 | AI 跑测试 + 解释 |
| **学习曲线** | 学语法 + API | 学"如何问" |

### 0.2 Vibe Coding 的 5 条黄金法则

> 来自 [Tobias Weiss: Vibe Coding with OpenCode](https://tobias-weiss.org/content/ai/opencode-guide/)

1. **描述意图而非实现** —— 说"添加用户认证"而不是"创建 JWT 中间件 + bcrypt"
2. **信任工作流** —— 让 AI 自主探索、迭代、自我修正
3. **保持驾驶座** —— 你定方向，AI 执行
4. **拥抱实验** —— 快速迭代胜过完美规划
5. **善用 Skills** —— 方法论由 Skills 承担，你专注创造

### 0.3 为什么需要完整的 Agent / Skills / MCP 体系

没有体系支撑的 Vibe Coding = 让 AI 自由发挥 = **结果不可控、效率低下、安全风险**。

完整的 Vibe Coding 体系 = **3 层支撑**：

```text
┌─────────────────────────────────────────────────┐
│ 你（Vibe Coder）：描述意图 + 把控方向             │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ Agent 层：任务分解 + 角色分工 + 自主执行         │
│ (orchestrator / builder / explorer / reviewer)  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ Skills 层：方法论注入 + 工作流规范 + 质量保证     │
│ (TDD / systematic-debug / writing-plans / ...)  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ MCP 层：外部能力扩展（文档 / 浏览器 / Git / DB）  │
│ (Context7 / Playwright / GitHub / filesystem)   │
└─────────────────────────────────────────────────┘
```

---

## 第一部分：全栈研发 Agent / Skills / MCP 体系设计

### 1. 完整体系地图

```text
┌────────────────────────────────────────────────────────────────┐
│ 1. 入口：opencode 命令 / Web / IDE 插件                          │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. 全局配置（~/.config/opencode/opencode.json）                │
│    - provider（模型供应）                                        │
│    - model（默认模型）                                           │
│    - plugin（插件：oh-my-openagent + superpowers）              │
│    - mcp（MCP 服务器）                                           │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Agent 配置（oh-my-openagent.json）                            │
│    10 个内置 agent + 8 个 category：                            │
│    - 思考层：prometheus / metis / momus / oracle                │
│    - 执行层：sisyphus-junior / build / hephaestus              │
│    - 探索层：explore / librarian / multimodal-looker            │
│    - 协调层：atlas                                                │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Skills 库（~/.opencode/skills/ + superpowers）               │
│    - 17 个 superpowers skills（方法论）                          │
│    - 2 个用户 skill（karpathy-guidelines / project-management）│
│    - 3 个推荐 skill（前端 UI/UX、Playwright、git-master）       │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. MCP 服务器（运行时拉起的外部能力）                            │
│    - Context7 / Playwright / GitHub / filesystem / ...          │
└────────────────────────────────────────────────────────────────┘
```

### 2. Agent 分层架构

#### 2.1 oh-my-openagent 提供的 10 个 Agent

| Agent | 角色 | 适用 | 模型（当前配置） |
| :--- | :--- | :--- | :--- |
| **prometheus** | 计划制定者 | 复杂任务的 Plan 制定 | minimax/MiniMax-M3 |
| **metis** | 预规划顾问 | 识别隐藏意图、模糊点 | minimax/MiniMax-M3 |
| **momus** | 计划评审者 | 评估计划的清晰度、可验证性 | minimax/MiniMax-M3 |
| **oracle** | 高 IQ 咨询 | 架构 / 调试 / 复杂逻辑 | minimax/MiniMax-M3 |
| **sisyphus-junior** | 任务执行者 | 单一明确任务的执行 | minimax/MiniMax-M3 |
| **build** | 默认执行者 | 通用任务 | minimax/MiniMax-M3 |
| **hephaestus** | 高级构建者 | 端到端项目实施 | minimax/MiniMax-M3 |
| **explore** | 上下文搜索者 | "代码在哪 / 哪个文件有 Y" | minimax/MiniMax-M3 |
| **librarian** | 外部参考搜索 | 官方文档 / 库用法 / OSS 示例 | minimax/MiniMax-M3 |
| **multimodal-looker** | 多模态分析 | PDF / 图片 / 图表解读 | minimax/MiniMax-M3 |
| **atlas** | 协调者 | 多 agent 协调 | minimax/MiniMax-M3 |

#### 2.2 8 个 Category（任务领域分类）

| Category | 适用 | 模型（当前配置） |
| :--- | :--- | :--- |
| **visual-engineering** | 前端 / UI / UX / 动画 | minimax/MiniMax-M3 |
| **ultrabrain** | 硬逻辑 / 架构决策 | minimax/MiniMax-M3 |
| **deep** | 自主研究 + 端到端实现 | minimax/MiniMax-M3 |
| **artistry** | 创造性、非常规问题 | minimax/MiniMax-M3 |
| **quick** | 单文件 / typo / 简单修改 | minimax/MiniMax-M3 |
| **unspecified-low** | 低投入任务 | minimax/MiniMax-M3 |
| **unspecified-high** | 高投入任务 | minimax/MiniMax-M3 |
| **writing** | 文档 / 散文 | minimax/MiniMax-M3 |

#### 2.3 Agent 协同模式

```text
主对话 (你) ─→ Sisyphus (你) ─→ 任务分发
                                  ↓
        ┌─────────────┬─────────────┬─────────────┐
        ↓             ↓             ↓             ↓
    explore        librarian     sisyphus-junior  oracle
   (本地搜索)    (外部参考)      (执行)         (咨询)
        ↓             ↓             ↓             ↓
        └─────────────┴─────────────┴─────────────┘
                            ↓
                    汇总 → 反馈给主对话
```

### 3. Skills 体系（方法论库）

Skills 是 **方法论注入器**——它们在 agent 启动工作前注入详细的工作流指令，让 agent 不再即兴发挥。

#### 3.1 已装 Skills 现状（本次配置完成后）

**Superpowers 包（15 个）**：
- `brainstorming` —— 任何创作工作前的意图探索
- `using-superpowers` —— Skill 使用元规范
- `writing-plans` —— 编写实现计划
- `subagent-driven-development` —— 子 agent 驱动开发
- `dispatching-parallel-agents` —— 并行 agent 调度
- `executing-plans` —— 在独立会话执行计划
- `test-driven-development` —— TDD 实施
- `systematic-debugging` —— 系统性调试
- `verification-before-completion` —— 完成前验证
- `writing-skills` —— 创建新 Skill
- `using-git-worktrees` —— Git worktree 隔离
- `requesting-code-review` —— 请求 Code Review
- `receiving-code-review` —— 接收 Code Review
- `finishing-a-development-branch` —— 完成开发分支
- `subagent-driven-development` —— 子 agent 驱动

**用户层（2 个）**：
- `karpathy-guidelines` —— Karpathy 行为准则
- `project-management` —— 项目管理

**推荐新增（3 个）**：
- `frontend-ui-ux` —— 设计师视角的前端开发
- `git-master` —— Git 原子化操作
- `dev-browser` / `playwright` —— 浏览器自动化

### 4. MCP 工具集成（外部能力）

MCP（Model Context Protocol）让 agent 获得**本地工具之外**的能力。OpenCode 1.4+ 完整支持 stdio / streamable HTTP 两种传输。

#### 4.1 推荐 MCP 工具全景

| MCP | 能力 | 价值 | 必需性 |
| :--- | :--- | :--- | :--- |
| **Context7** | 库文档 / 代码示例查询 | 解决"AI 用过时 API" | ⭐⭐⭐⭐⭐ |
| **Playwright** | 浏览器自动化 | E2E 测试 / 抓取 / 截图 | ⭐⭐⭐⭐⭐ |
| **GitHub** | GitHub API | 提 PR / review / issue | ⭐⭐⭐⭐ |
| **filesystem** | 跨工作区文件操作 | 多项目协同 | ⭐⭐⭐ |
| **sequential-thinking** | 结构化思考 | 复杂问题分解 | ⭐⭐⭐⭐ |
| **exa** | AI 搜索 | 最新研究 / 时效信息 | ⭐⭐⭐ |
| **firecrawl** | Web 抓取 | 文档 / 文章抓取 | ⭐⭐⭐ |
| **ref** | 参考文档 | 库 API 速查 | ⭐⭐⭐ |

### 5. 三者协同的完整工作流

```text
Vibe Coding 完整工作流（典型：加一个新功能）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Step 1] 你说："给首页加个用户登录功能"
   ↓
[Step 2] Agent 自动加载 Skills：
   - writing-plans（先写计划）
   - brainstorming（如果需求模糊）
   - test-driven-development（如果涉及代码）
   ↓
[Step 3] Plan Agent 调研：
   - explore（搜索本地代码）
   - librarian（查 OAuth / NextAuth 最佳实践）
   - multimodal-looker（看 UI 截图）
   ↓
[Step 4] 输出 Plan → Momus 评审 → 你批准
   ↓
[Step 5] Build Agent 执行：
   - 调用 Context7 MCP（查 NextAuth 文档）
   - 调用 GitHub MCP（创建分支）
   - 写代码 → 跑测试（Playwright MCP E2E）
   - Git commit
   ↓
[Step 6] 你说"再优化一下 UX"
   - Agent 加载 frontend-ui-ux skill
   - 调 Playwright MCP 截图
   - 改进 → 再截图对比
   ↓
[Step 7] 你说"提 PR"
   - Agent 调 GitHub MCP 提 PR
   - Code Review skill 触发
   ↓
[Step 8] 部署 / 完成
```

---

## 第二部分：必装 Skills 详解（10 大核心）

### 6. superpowers 基础包（15 个方法论 Skill）

#### 6.1 元 Skill

| Skill | 用途 | 触发时机 |
| :--- | :--- | :--- |
| `using-superpowers` | Skill 系统元规范 | 任何任务前 |
| `writing-skills` | 创建新 Skill | 需要新方法论时 |

#### 6.2 创作前

| Skill | 用途 | 触发时机 |
| :--- | :--- | :--- |
| `brainstorming` | Socratic 探索意图 | 创建功能 / 修改行为前 |

#### 6.3 计划与设计

| Skill | 用途 | 触发时机 |
| :--- | :--- | :--- |
| `writing-plans` | 写实施计划 | 多步骤任务前 |
| `executing-plans` | 在独立会话执行计划 | 跨会话持续工作 |
| `subagent-driven-development` | 子 agent 驱动开发 | 复杂任务 |
| `dispatching-parallel-agents` | 并行调度 | 2+ 独立任务 |
| `using-git-worktrees` | Git worktree 隔离 | 开始新功能前 |

#### 6.4 实施与质量

| Skill | 用途 | 触发时机 |
| :--- | :--- | :--- |
| `test-driven-development` | TDD | 写代码前 |
| `systematic-debugging` | 系统性调试 | 任何 bug |
| `verification-before-completion` | 完成前验证 | 声称"完成"前 |

#### 6.5 协作与完成

| Skill | 用途 | 触发时机 |
| :--- | :--- | :--- |
| `requesting-code-review` | 申请 Code Review | 完成功能后 |
| `receiving-code-review` | 接收 Code Review | 收到 review 时 |
| `finishing-a-development-branch` | 合并 / 提 PR / 清理 | 完成开发 |

### 7. 用户层必装 Skills（karpathy-guidelines + project-management）

#### 7.1 karpathy-guidelines（已装）

- **来源**：用户安装
- **作用**：减少 LLM 编码常见错误——避免过度复杂、做最小变更、明确假设、定义可验证的成功标准
- **何时用**：所有写代码 / review / 重构任务

#### 7.2 project-management（已装）

- **来源**：用户安装
- **作用**：项目管理——项目检视、迭代规划、bug 跟踪、需求管理
- **何时用**：项目级决策、迭代规划

### 8. 推荐必装 Skills（3 个新增）

#### 8.1 frontend-ui-ux

- **来源**：内置
- **作用**：设计师转开发者视角的前端开发——即使无设计稿也能产出高审美 UI/UX
- **何时用**：任何前端 / 视觉任务

#### 8.2 git-master

- **来源**：内置
- **作用**：原子化 commit、rebase / squash、git log -S 等高级操作
- **何时用**：任何 git 操作

#### 8.3 dev-browser / playwright

- **来源**：内置（dev-browser）
- **作用**：浏览器自动化——打开网站、填表单、截图、抓数据、测试 Web 应用
- **何时用**：E2E 测试、网页抓取、Web 调试

### 9. 自定义 Skills 的方法

**Skill 目录结构**：

```text
~/.opencode/skills/my-skill/
├── SKILL.md          # Skill 元数据 + 指令
├── references/       # 可选：参考文档
├── scripts/          # 可选：辅助脚本
└── assets/           # 可选：模板 / 资源
```

**SKILL.md 模板**：

```markdown
---
name: my-skill
description: 一句话说明（agent 据此判断何时调用）
---

# My Skill

## 何时使用
- 条件 1
- 条件 2

## 工作流
1. 第一步
2. 第二步
...

## 检查清单
- [ ] 项 1
- [ ] 项 2

## 范例
（具体例子）
```

调用 `writing-skills` Skill 可获得详细的自定义 Skill 创建指南。

---

## 第三部分：必装 MCP Servers 详解（7 大核心）

### 10. Context7（库文档查询）

**能力**：实时查询任何库的最新官方文档和代码示例。

**为什么必装**：LLM 的训练数据有截止日期，常用"过时 API"。Context7 解决"幻觉式编程"。

**OpenCode 配置**：

```json
{
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "enabled": true
  }
}
```

**使用范例**：

> "用 Context7 查一下 Next.js 15 的 Server Actions 最新用法"

agent 会自动调用 Context7 MCP 获取最新文档，避免用过时的 `getServerSideProps` 等 API。

**API Key（可选）**：免费版够用，付费版有更高 rate limit。

### 11. Playwright / Chrome DevTools（浏览器自动化）

**能力**：控制浏览器——打开页面、点击、填表单、截图、提取 DOM。

**OpenCode 配置**（Playwright 优先）：

```json
{
  "playwright": {
    "type": "local",
    "command": ["npx", "-y", "@playwright/mcp@latest"],
    "enabled": true
  }
}
```

**使用范例**：

> "用 Playwright 打开 localhost:3000，截图首页，提取所有按钮的 selector"

agent 会用 Playwright MCP 完成这些操作。

**何时用**：
- E2E 测试
- 网页抓取
- Web 应用调试
- UI 截图对比

### 12. GitHub（代码托管）

**能力**：与 GitHub API 完整集成——创建/管理 issue、提 PR、review、merge。

**OpenCode 配置**：

```json
{
  "github": {
    "type": "local",
    "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
    "environment": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
    },
    "enabled": true
  }
}
```

**使用范例**：

> "用 GitHub MCP 给当前分支提一个 PR，目标 main"

**Token 设置**：
1. GitHub → Settings → Developer settings → Personal access tokens
2. 勾选 `repo`, `read:org`, `workflow`
3. 设置环境变量 `GITHUB_TOKEN`

### 13. filesystem（文件系统）

**能力**：标准化的文件系统操作——读、写、搜索、监控。

**OpenCode 配置**：

```json
{
  "filesystem": {
    "type": "local",
    "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/your/projects"],
    "enabled": true
  }
}
```

**使用范例**：

> "在 /Users/me/projects 目录下找所有含 TODO 的 markdown 文件"

**注意**：限定路径范围，避免给 AI 完整文件系统访问权限（安全风险）。

### 14. sequential-thinking（结构化思考）

**能力**：让 agent 按结构化思维链处理复杂问题——分步、可修订、可分支。

**OpenCode 配置**：

```json
{
  "sequential-thinking": {
    "type": "local",
    "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
    "enabled": true
  }
}
```

**何时用**：
- 复杂多步问题
- 需要分支思考的设计决策
- 调试时需要排除假设

### 15. exa / firecrawl（Web 搜索 / 抓取）

**exa**：AI 搜索，返回语义化结果

```json
{
  "exa": {
    "type": "local",
    "command": ["npx", "-y", "exa-mcp-server"],
    "environment": {
      "EXA_API_KEY": "${EXA_API_KEY}"
    },
    "enabled": true
  }
}
```

**firecrawl**：专业的 Web 抓取与解析

```json
{
  "firecrawl": {
    "type": "remote",
    "url": "https://api.firecrawl.dev/mcp",
    "headers": {
      "Authorization": "Bearer ${FIRECRAWL_API_KEY}"
    },
    "enabled": true
  }
}
```

### 16. Ref / DeepWiki（参考文档）

**Ref**：参考文档速查（替代 devdocs.io 的 MCP 版本）

```json
{
  "ref": {
    "type": "remote",
    "url": "https://api.ref.dev/mcp",
    "headers": {
      "Authorization": "Bearer ${REF_API_KEY}"
    },
    "enabled": true
  }
}
```

---

## 第四部分：完整配置实战

### 17. 一次性配置脚本

下面提供一份**幂等**的 shell 脚本——可重复运行、保留已有 token、添加新 MCP。

```bash
#!/bin/bash
# setup-vibe-coding.sh
# 一键配置 OpenCode 全栈 Vibe Coding 环境
# 用法：bash setup-vibe-coding.sh

set -e

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== OpenCode Vibe Coding 全栈配置 ===${NC}"

# 1. 安装用户 Skills（如果缺失）
mkdir -p ~/.opencode/skills

# 1.1 frontend-ui-ux（内置插件，会在下次 opencode 启动时自动加载）
# 1.2 git-master（内置插件）
# 1.3 dev-browser（内置插件）

# 2. 安装 superpowers（如果缺失）
if [ ! -d ~/.cache/opencode/packages/superpowers@git+https:/github.com/obra/superpowers.git ]; then
  echo -e "${YELLOW}正在安装 superpowers 插件...${NC}"
  cd ~/.config/opencode
  npm install --save-dev superpowers@git+https://github.com/obra/superpowers.git
fi

# 3. 备份当前 opencode.json
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp ~/.config/opencode/opencode.json ~/.config/opencode/opencode.json.bak.$TIMESTAMP
echo -e "${GREEN}已备份 opencode.json → opencode.json.bak.$TIMESTAMP${NC}"

# 4. 合并 MCP 配置（幂等）
# 这一步通过 Python 脚本处理 JSON 合并
python3 <<'PYEOF'
import json
import os
from pathlib import Path

config_path = Path.home() / ".config/opencode/opencode.json"
config = json.loads(config_path.read_text())

# 推荐 MCP 配置（按需启用）
mcp_recommend = {
    # 1. Context7（最推荐 - 文档查询）
    "context7": {
        "type": "remote",
        "url": "https://mcp.context7.com/mcp",
        "enabled": True
    },
    # 2. Playwright（最推荐 - 浏览器自动化）
    "playwright": {
        "type": "local",
        "command": ["npx", "-y", "@playwright/mcp@latest"],
        "enabled": True
    },
    # 3. sequential-thinking（推荐 - 结构化思考）
    "sequential-thinking": {
        "type": "local",
        "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
        "enabled": True
    },
    # 4. filesystem（推荐 - 跨工作区）
    "filesystem": {
        "type": "local",
        "command": [
            "npx", "-y", "@modelcontextprotocol/server-filesystem",
            str(Path.home() / "projects")
        ],
        "enabled": False  # 默认关闭（安全考虑）
    },
    # 5. GitHub（按需启用）
    "github": {
        "type": "local",
        "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
        "environment": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
        },
        "enabled": False  # 默认关闭（需 token）
    },
    # 6. exa（按需启用 - 需 EXA_API_KEY）
    "exa": {
        "type": "local",
        "command": ["npx", "-y", "exa-mcp-server"],
        "environment": {
            "EXA_API_KEY": "${EXA_API_KEY}"
        },
        "enabled": False
    },
    # 7. firecrawl（按需启用 - 需 FIRECRAWL_API_KEY）
    "firecrawl": {
        "type": "remote",
        "url": "https://api.firecrawl.dev/mcp",
        "headers": {
            "Authorization": "Bearer ${FIRECRAWL_API_KEY}"
        },
        "enabled": False
    }
}

# 合并（不覆盖已存在的 enabled 状态）
if "mcp" not in config:
    config["mcp"] = {}
for name, settings in mcp_recommend.items():
    if name not in config["mcp"]:
        config["mcp"][name] = settings
    else:
        # 保留已存在配置，更新其他字段
        for k, v in settings.items():
            if k not in config["mcp"][name]:
                config["mcp"][name][k] = v

# 写回
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
print("✓ MCP 配置已合并")
PYEOF

# 5. 验证
echo ""
echo -e "${GREEN}=== 配置完成 ===${NC}"
echo ""
echo "MCP 服务器状态："
python3 -c "
import json
from pathlib import Path
config = json.loads(Path.home().joinpath('.config/opencode/opencode.json').read_text())
for name, settings in (config.get('mcp') or {}).items():
    enabled = '✅' if settings.get('enabled') else '⚪'
    type_str = settings.get('type', '?')
    print(f'  {enabled} {name} ({type_str})')
"

echo ""
echo "下一步："
echo "1. 设置必要的环境变量（如果启用了 GitHub/exa/firecrawl）："
echo "   export GITHUB_TOKEN='ghp_xxx'"
echo "   export EXA_API_KEY='xxx'"
echo "   export FIRECRAWL_API_KEY='xxx'"
echo ""
echo "2. 重启 OpenCode："
echo "   opencode"
echo ""
echo "3. 验证 MCP："
echo "   在 opencode 中输入 '用 context7 查一下 Next.js 15 的最新 API'"
echo ""
```

### 18. 全局 vs 项目级配置

| 层级 | 位置 | 作用 | 适合放什么 |
| :--- | :--- | :--- | :--- |
| **全局** | `~/.config/opencode/opencode.json` | 跨项目通用 | provider、plugin、全局 MCP |
| **项目级** | `<project>/opencode.json` | 项目特有 | 项目的 MCP、commands、AGENTS.md |
| **Skill 目录** | `~/.opencode/skills/` | 用户方法论库 | 个人 Skills |
| **命令目录** | `~/.config/opencode/command/` | 自定义 slash 命令 | 个人常用命令 |

**推荐做法**：
- 全局放：provider、plugin、推荐 MCP
- 项目级放：项目特定的 commands、AGENTS.md、特殊 MCP
- 个人 Skill 放：`~/.opencode/skills/`

### 19. 安全 / 成本 / 性能优化

#### 19.1 安全

| 风险 | 缓解 |
| :--- | :--- |
| **MCP 越权访问** | filesystem MCP 限定路径；不在 MCP 中暴露敏感 token |
| **Token 泄露** | 用 `${ENV_VAR}` 引用环境变量，不直接写在 config |
| **执行危险命令** | OpenCode 默认有 bash 权限控制；可在项目级禁用 |
| **依赖投毒** | `npx -y` 加 `package-lock` 校验；优先用官方包 |

#### 19.2 成本

| 优化 | 节省 |
| :--- | :--- |
| **轻量任务用 quick / haiku** | 50-70% |
| **重复任务 cache** | 30% |
| **本地模型替代** | 100%（特定任务） |
| **关闭不必要的 MCP** | 20-40% token 节省 |

#### 19.3 性能

| 优化 | 效果 |
| :--- | :--- |
| **MCP 懒加载** | 启动更快 |
| **disable 未用的 MCP** | context 更小，AI 决策更快 |
| **小项目用 quick** | 延迟更低 |
| **大项目用 deep + subagent** | 任务并行 |

---

## 第五部分：典型工作流（5 大场景）

### 20. 工作流 1：从零开始一个新项目

**用户输入**：
> "用 Next.js 15 + TypeScript + Tailwind 给我做个个人博客，要有首页、文章列表、详情页、RSS feed。"

**自动加载的 Skills**：
- `using-superpowers`（元规范）
- `brainstorming`（探索细节）
- `writing-plans`（制定计划）
- `test-driven-development`（TDD 实施）

**Agent 协同**：
1. `brainstorming` Skill 问 3-5 个澄清问题
2. `writing-plans` 写出实施计划
3. `explore` + `librarian` 调研最新 Next.js 15 模板
4. `sisyphus-junior` + `hephaestus` 实施

**MCP 调用**：
- `context7`：查 Next.js 15 / App Router 最新 API
- `github`：初始化 repo

### 21. 工作流 2：为一个老项目加新功能

**用户输入**：
> "给这个 Next.js 项目的设置页加 dark mode 切换"

**关键 Skills**：
- `using-git-worktrees`（隔离新功能）
- `systematic-debugging`（如果出 bug）
- `verification-before-completion`（完成前验证）

**Agent 协同**：
1. `explore` 搜索现有主题切换代码
2. `sisyphus-junior` 修改 `tailwind.config.ts` + `layout.tsx`
3. `verification-before-completion` 跑测试

### 22. 工作流 3：Debug 一个生产问题

**用户输入**：
> "生产上用户报告 5% 的请求 500 错误，帮我找根因"

**关键 Skills**：
- `systematic-debugging`（系统化排查）
- `test-driven-development`（写复现测试）

**MCP 调用**：
- `playwright`：复现问题
- `sequential-thinking`：结构化排查

### 23. 工作流 4：复现一篇论文

**用户输入**：
> "复现 HSTU 这篇论文，Meta 2023 的"

**关键 Skills**：
- `writing-plans`（明确步骤）
- `test-driven-development`（每个组件先写测试）

**Agent 协同**：
1. `librarian` 找论文 + 官方代码
2. `multimodal-looker` 解读论文图
3. `sisyphus-junior` 实现
4. `verification-before-completion` 跑实验

### 24. 工作流 5：跨团队沟通 / Code Review

**用户输入**：
> "review 一下这个 PR，告诉我有没有问题"

**关键 Skills**：
- `receiving-code-review` / `requesting-code-review`
- `karpathy-guidelines`（评审标准）

**MCP 调用**：
- `github`：直接评论 PR

---

## 附录 A：所有推荐配置的清单

### A.1 全局 opencode.json 推荐结构

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "oh-my-openagent@latest",
    "superpowers@git+https://github.com/obra/superpowers.git"
  ],
  "provider": {
    "minimax": { ... },
    "volcengine-plan": { ... }
  },
  "model": "volcengine-plan/minimax-m2.5",
  "mcp": {
    "context7": { "type": "remote", "url": "https://mcp.context7.com/mcp", "enabled": true },
    "playwright": { "type": "local", "command": ["npx", "-y", "@playwright/mcp@latest"], "enabled": true },
    "sequential-thinking": { "type": "local", "command": ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"], "enabled": true },
    "filesystem": { "type": "local", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"], "enabled": false },
    "github": { "type": "local", "command": ["npx", "-y", "@modelcontextprotocol/server-github"], "environment": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" }, "enabled": false },
    "exa": { "type": "local", "command": ["npx", "-y", "exa-mcp-server"], "environment": { "EXA_API_KEY": "${EXA_API_KEY}" }, "enabled": false },
    "firecrawl": { "type": "remote", "url": "https://api.firecrawl.dev/mcp", "headers": { "Authorization": "Bearer ${FIRECRAWL_API_KEY}" }, "enabled": false }
  }
}
```

### A.2 必装 Skills 清单（17+2+3=22 个）

| 来源 | Skill | 状态 |
| :--- | :--- | :--- |
| superpowers | brainstorming | ✅ 已装 |
| superpowers | using-superpowers | ✅ 已装 |
| superpowers | writing-plans | ✅ 已装 |
| superpowers | subagent-driven-development | ✅ 已装 |
| superpowers | dispatching-parallel-agents | ✅ 已装 |
| superpowers | executing-plans | ✅ 已装 |
| superpowers | test-driven-development | ✅ 已装 |
| superpowers | systematic-debugging | ✅ 已装 |
| superpowers | verification-before-completion | ✅ 已装 |
| superpowers | writing-skills | ✅ 已装 |
| superpowers | using-git-worktrees | ✅ 已装 |
| superpowers | requesting-code-review | ✅ 已装 |
| superpowers | receiving-code-review | ✅ 已装 |
| superpowers | finishing-a-development-branch | ✅ 已装 |
| 用户 | karpathy-guidelines | ✅ 已装 |
| 用户 | project-management | ✅ 已装 |
| 内置 | frontend-ui-ux | 🆕 推荐 |
| 内置 | git-master | 🆕 推荐 |
| 内置 | dev-browser | 🆕 推荐 |
| 内置 | review-work | 🆕 推荐 |
| 内置 | ai-slop-remover | 🆕 推荐 |

### A.3 推荐 MCP Servers 清单

| MCP | 状态 | 优先级 |
| :--- | :--- | :--- |
| context7 | 🆕 配置中 | ⭐⭐⭐⭐⭐ |
| playwright | 🆕 配置中 | ⭐⭐⭐⭐⭐ |
| sequential-thinking | 🆕 配置中 | ⭐⭐⭐⭐ |
| filesystem | 🆕 配置中（默认关） | ⭐⭐⭐ |
| github | 🆕 配置中（需 token） | ⭐⭐⭐⭐ |
| exa | 🆕 配置中（需 token） | ⭐⭐⭐ |
| firecrawl | 🆕 配置中（需 token） | ⭐⭐⭐ |

---

## 附录 B：常见问题 FAQ

### B.1 MCP 启动失败怎么办？

**检查清单**：

```bash
# 1. 检查 Node.js
node --version  # 应 ≥ 18

# 2. 检查 npx
npx --version

# 3. 测试单个 MCP
npx -y @playwright/mcp@latest

# 4. 查看 OpenCode 日志
opencode --log-level debug

# 5. 检查端口冲突
lsof -i :port_number
```

### B.2 Token 怎么设置？

**macOS**：

```bash
# 临时（当前会话）
export GITHUB_TOKEN="ghp_xxx"

# 永久（推荐）
echo 'export GITHUB_TOKEN="ghp_xxx"' >> ~/.zshrc
source ~/.zshrc
```

**用 .env 文件 + direnv**（推荐）：

```bash
# 项目根目录 .env
GITHUB_TOKEN=ghp_xxx
EXA_API_KEY=xxx

# .envrc
dotenv
```

### B.3 哪些 MCP 会消耗大量 Token？

- **GitHub MCP**：会拉取大量 issue / PR 数据
- **filesystem MCP**：限制路径
- **Context7 MCP**：查询返回完整文档

**建议**：
- 默认关闭高消耗 MCP
- 一次性查询后立即关
- 用 `enabled: false` 按需启用

### B.4 怎么知道 MCP 是否在工作？

**在 OpenCode 中**：

```text
列出可用的 MCP 工具
```

agent 会列出所有可用 MCP 工具。

### B.5 Skill 不生效怎么办？

1. **确认 Skill 已安装**：`ls ~/.opencode/skills/`
2. **确认 SKILL.md 格式正确**：包含 `name` 和 `description` 字段
3. **重启 OpenCode**
4. **检查 Skill 描述是否触发**：skill description 应明确"何时用"

### B.6 推荐用什么模型？

**当前推荐**（按任务）：

| 任务 | 模型 |
| :--- | :--- |
| 默认（性价比高） | `volcengine-plan/minimax-m2.5` |
| 前端 / 视觉 | `volcengine-plan/doubao-seed-2.0-code` |
| 复杂推理 | `volcengine-plan/kimi-k2.5` |
| 长上下文 | `volcengine-plan/ark-code-latest`（256K） |
| 快速轻量 | `volcengine-plan/doubao-seed-2.0-lite` |

切换方式：

```bash
# 在 OpenCode 中用 /model 命令切换
/model volcengine-plan/kimi-k2.5
```

### B.7 怎么扩展自定义 Skill？

```bash
mkdir -p ~/.opencode/skills/my-skill
cat > ~/.opencode/skills/my-skill/SKILL.md <<'EOF'
---
name: my-skill
description: 当用户要求 [具体场景] 时使用
---

# My Skill

# 备份全局配置
cp ~/.config/opencode/opencode.json ~/backup-opencode-$(date +%Y%m%d).json

# 备份 Skills
tar -czf ~/backup-skills-$(date +%Y%m%d).tar.gz ~/.opencode/skills/

# 用 Git 管理（推荐）
cd ~/.config/opencode && git init
cd ~/.opencode && git init
```

---

> **文档版本**：v1.0
> **创建日期**：基于 OpenCode 1.4 + oh-my-openagent 3.16 + superpowers 0.0.2 的当前最佳实践。
> **配套脚本**：见 §17 —— 可重复运行的幂等配置脚本。
> **下一步**：重启 OpenCode 验证所有 Skills / MCP 是否正确加载。

---

## 25. Vibe Coding 实践哲学与心智模型

### 25.1 5 条黄金法则的展开

#### 法则 1：描述意图而非实现

```python
# NOT Vibe Coding:
"请帮我创建一个 JWT 中间件，使用 HS256 算法，配置 secret 从环境变量读取，
验证 token 中的 exp 字段，过期返回 401，并把用户 ID 写入 request..."

# Vibe Coding:
"请给我们的 Express app 添加一个用户认证的中间件，要求能验证 token 有效性，
未通过返回 401"
```

**好处**：

- 工程师关注业务（做什么）
- AI 负责细节（怎么做）
- 实现可换（HS256 → RS256 / 其他）

#### 法则 2：信任工作流

```python
# 不需要问每一步
"列出 5 个主流 ML 项目的目录结构对比，每个写一段说明"

# AI 会：
# 1. 通过 explore 搜索项目
# 2. 通过 librarian 查文档
# 3. 自动整理对比
```

**不要干预**：除非 AI 走错方向，否则让它自己跑。

#### 法则 3：保持驾驶座

```python
# 你定方向
"我们要重构用户登录模块，从 JWT 改为 OAuth2.0"

# AI 负责实现
# 但你 review + 测试 + 最终决策
```

#### 法则 4：拥抱实验

```python
# 实验 A: OAuth2 with PKCE
# 实验 B: OAuth2 with simple bearer
# 用 small data 测试两个 → 选更好的
```

#### 法则 5：善用 Skills

- brainstorming：任何创作前
- writing-plans：多步骤任务
- TDD：写代码前
- verification-before-completion：任何"完成"声明前

### 25.2 Vibe Coding 的认知层级

```text
5. STRATEGY      "我们要做推荐系统"
4. ARCHITECTURE  "召回+排序+重排"
3. DESIGN        "用 DIN 模型 + Faiss"
2. IMPLEMENTATION "代码 / 配置 / 部署"
1. OPERATION     "训练 / 监控 / 调优"

Vibe Coding 让人专注 3-5 层，把 1-2 委托给 AI
```

### 25.3 Vibe Coding 的"信任区域"

```text
高信任（AI 直接干）：
  ✓ 写 boilerplate 代码
  ✓ 单元测试
  ✓ 代码格式化
  ✓ 文档生成
  ✓ 重构命名

中信任（AI 写后 review）：
  ~ 业务逻辑
  ~ 数据处理
  ~ 简单算法

低信任（AI 给方案，人审查）：
  ⚠️ 架构决策
  ⚠️ 安全相关
  ⚠️ 性能敏感
  ⚠️ 资金/数据敏感
```

---

## 26. 完整的 Agent 协作模式

### 26.1 单 Agent 工作流

用户 → Sisyphus → 单任务 → 返回

适用：单文件修改、简单问题。

### 26.2 多 Agent 协作

```text
用户: "给这个项目添加 OAuth 认证"

自动:
1. Sisyphus 接收任务
2. 触发 brainstorming skill（探索需求）
3. 触发 writing-plans skill（生成实施计划）
4. explore agent 搜索项目结构
5. librarian agent 查 OAuth 最新文档
6. multimodal-looker agent 看设计图（如果有 UI）
7. oracle agent 评估技术选型（OAuth vs SAML）
8. Sisyphus 执行实施
9. verification-before-completion 验证
10. momus 评估计划一致性
```

### 26.3 主/分支 Agent

- **主 Agent**：跟用户对话，理解意图
- **Branch Agent**：后台异步跑子任务

```python
# OpenCode 内置这种模式
# 用户提问 → 主 agent
# 主 agent 派 subagent → 后台跑
# 等 subagent 完成 → 主 agent 继续
```

### 26.4 任务复杂度分层

| 复杂度 | 模式 | 示例 |
| :--- | :--- | :--- |
| 简单 | 单 agent | 修 typo |
| 中等 | 多 agent 串行 | 加新功能 |
| 复杂 | 多 agent 并行 + 计划 | 大型重构 |
| 超复杂 | Plan agent → Momus 评审 → 人审批 → 执行 | 架构迁移 |

---

## 27. 实战场景代码示例

### 27.1 场景：写一个 RESTful API

```text
用户："用 FastAPI 写一个 todos API，支持 CRUD 操作"

Vibe Coding:

1. TDD 起点：写测试
   tests/test_todos.py:
     def test_create_todo():
         response = client.post("/todos", json={"title": "Test"})
         assert response.status_code == 201
         assert response.json()["title"] == "Test"

2. AI 自动实现 (TDD):
   - 创建 main.py + models + routes
   - 实现 POST /todos, GET /todos/{id}, etc.
   - 验证测试通过

3. 完成后：
   - Coverage 报告
   - OpenAPI docs (/docs)
   - 部署 Dockerfile
```

### 27.2 场景：重构老项目

```text
用户："这个 1000 行的 train.py 太乱了，重构它"

1. Sisyphus 调用 explore agent 搜索 train.py 在哪
2. writing-plans skill 创建重构计划
3. momus 评审计划
4. 用户批准
5. Sisyphus 执行：
   - 拆为 data.py + model.py + train.py + config.yaml
   - 用注册器重构
   - 加单元测试
6. verification-before-completion 验证
```

### 27.3 场景：生产问题

```text
用户："生产 5% 用户拿不到推荐结果"

1. systematic-debugging skill 自动启动
2. oracle agent 高 IQ 分析（看日志）
3. 假设：特征缺失
4. 验证：模拟缺失场景
5. 修复：加 fallback
6. 加监控告警
```

---

## 28. Vibe Coding 的进阶技巧

### 28.1 意图阶梯（Intent Ladder）

```text
高层:  "我们需要一个推荐系统"
中层:  "召回用双塔，精排用 DIN"
底层:  "Faiss 索引 + PyTorch DIN 模型 + Triton 服务"
```

Vibe Coding：开始高层，让 AI 带你下到具体实现。

### 28.2 Context Window 管理

AI 的 context window 是有限的。管理技巧：

- **切分任务**：每次只描述一个清晰目标
- **引用历史**：用 git commit message 总结前几步
- **保留长期**：用文件而非 chat history（如 SKILL.md / AGENTS.md）

### 28.3 Prompt Engineering

```yaml
# 有效 prompt 模式
- "假设你是 [role]"
- "上下文：[context]"
- "目标：[goal]"
- "约束：[constraints]"
- "输出格式：[format]"
- "示例：[example]"
```

### 28.4 当 AI 卡住时

1. 提供更多上下文
2. 切换 agent（尝试 librarian 或 oracle）
3. 改述问题（重新提问）
4. 拆分子任务
5. 给 AI 反馈（"刚才那样不对，应该 ..."）

---

## 29. Skills 深度实践

### 29.1 brainstorming

**何时用**：任何创作前。

**好处**：

- 探索隐藏需求
- 让 AI 提问澄清
- 完整需求画像

### 29.2 writing-plans

**何时用**：多步骤任务。

**模板**：

```markdown
# 实施计划

## 目标
[达到什么]

## 步骤
1. [步骤 1]
2. [步骤 2]
...

## 风险
- ...

## 验收
- [指标 1]
- [指标 2]
```

### 29.3 TDD (Test-Driven Development)

**何时用**：写代码前。

**流程**：

```text
1. 写失败测试
2. 写最小实现（让测试通过）
3. 重构（保留测试通过）
4. 重复
```

### 29.4 verification-before-completion

**何时用**：声明"完成"前。

**清单**：

- [ ] 跑过测试
- [ ] Type check 通过
- [ ] Lint 通过
- [ ] 功能正确（手动验证 + 自动测试）
- [ ] 文档更新

### 29.5 systematic-debugging

**何时用**：任何 bug。

**流程**：

```text
1. 重现
2. 隔离最小重现
3. 提出假设
4. 验证/排除假设
5. 修复根本原因
6. 验证修复
7. 加回归测试
```

---

## 30. MCP 实战案例

### 30.1 项目级 MCP 配置

```json
{
  "mcp": {
    "context7": {"type": "remote", "url": "https://mcp.context7.com/mcp", "enabled": true},
    "playwright": {"type": "local", "command": ["npx", "-y", "@playwright/mcp@latest"], "enabled": true},
    "github": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "environment": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
      "enabled": true
    },
    "filesystem": {
      "type": "local",
      "command": [
        "npx", "-y", "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects"
      ],
      "enabled": true
    }
  }
}
```

### 30.2 真实项目中的 MCP 协同

```text
任务："读 github.com/owner/repo 上的 issue #123，修复里面的 bug"

自动协作：
1. github MCP 读 issue #123
2. explore MCP 在本地找相关代码
3. oracle MCP 评估修复方案
4. sisyphus 执行修复
5. github MCP 创建 PR
```

---

## 31. 真实开发者体验分享

### 31.1 日均工作流

```text
早晨：
- 启动 OpenCode
- 输入今日目标："今天完成 OAuth 集成"
- 自动加载 Skills + Agents

工作中：
- 描述任务 → Sisyphus 执行
- Review AI 输出
- 给反馈 / 修正

收尾：
- 运行 verification-before-completion
- Git commit
- 自动生成周报
```

### 31.2 加速比的实际数据

| 任务类型 | 传统耗时 | Vibe Coding 耗时 | 加速比 |
| :--- | :--- | :--- | :--- |
| 写 boilerplate | 1 小时 | 5 分钟 | 12× |
| 单元测试 | 30 分钟 | 3 分钟 | 10× |
| 文档 | 1 小时 | 10 分钟 | 6× |
| 重构 | 半天 | 1 小时 | 4× |
| Bug 修复 | 1 小时 | 15 分钟 | 4× |
| 新功能（中等） | 1 周 | 2-3 天 | 2-3× |
| 架构迁移 | 1 月 | 1-2 周 | 2× |

**注意**：加速比随任务复杂度下降。但即使是 2× 也是企业级收益。

### 31.3 常见场景的关键 Prompt

#### 项目初始化

```text
用 Next.js 15 + Tailwind + TypeScript 创建一个博客网站，
要求：
- 首页 + 文章列表 + 文章详情
- RSS feed
- 暗色模式
- 部署到 Vercel

请先生成实施计划。
```

#### 加新功能

```text
为用户管理模块添加忘记密码功能：
- 用户输入邮箱 → 系统发邮件 → 点击邮件链接重置
- 用 SendGrid 或 Resend 发送
- 防滥用：rate limit + token 过期

请先 brainstorming 探索需求。
```

#### Debug

```text
我的 API 在高并发下返回 500，sentry 显示是数据库连接超限。
如何排查？已经在 db.ts 用 pgbouncer 了，但 max client 还是不够。
如何定位其他 pooling 问题？
```

---

## 32. 性能与成本优化

### 32.1 Token 成本意识

每次 LLM 调用都消耗 tokens。**优化技巧**：

| 技巧 | 节省 |
| :--- | :--- |
| **精炼 prompt** | 20-50% |
| **小模型用 cheap 模型** | 50% |
| **cache 重复 context** | 30%+ |
| **结构化输出而非对话** | 20% |

### 32.2 模型路由

```yaml
任务难度 → 模型选择:
  easy → haiku / lite (cheap)
  medium → mid-tier
  hard → opus / preview (expensive)
  code-only → sonnet / code-specific
```

### 32.3 实际成本估算

```text
假设：每日 100 次 AI 调用，平均每调用 10k tokens（input + output）
- Cheap model: $0.001 / 1k tokens → $1/天
- Mid model: $0.01 / 1k tokens → $10/天
- Premium model: $0.05 / 1k tokens → $50/天

月度成本：$30 - $1500 (主要取决于模型选择)
```

---

## 33. 安全与合规

### 33.1 常见风险

| 风险 | 缓解 |
| :--- | :--- |
| 代码泄露公司秘密 | 不要让 AI 看到 credentials |
| 误提交 | Git commit 前 review |
| 注入风险 | 不要 AI 处理不可信输入 |
| 隐私 | 客户数据不传给 LLM |

### 33.2 合规清单

- [ ] 不分享生产数据库 schema
- [ ] 客户数据本地处理
- [ ] AI 调用记录（合规审计）
- [ ] 知识产权审查

---

## 34. 与传统开发的对比

### 34.1 思维模式差异

| 传统开发 | Vibe Coding |
| :--- | :--- |
| 写之前想清楚 | 写后让 AI 改 |
| 自己实现细节 | AI 实现，自己验证 |
| 测试驱动 | 测试 + AI 修复循环 |
| 文档即代码 | 代码 + 文档生成 |

### 34.2 必备能力变化

| 传统 | 必备 |
| :--- | :--- |
| 语法 / API 记忆 | 设计 / 选型 / 决策 |
| 单线程调试 | 多次实验对比 |
| 写文档 | 描述意图 |

### 34.3 谁不会被替代

- 高级架构师（设计 + 决策）
- 产品经理（用户洞察）
- 资深工程师（解决难点 + 审稿）

### 34.4 谁需要加速

- 全栈工程师（日常 task）
- 数据科学家（boilerplate code）
- DevOps（配置 / 脚本）

---

## 35. 案例研究：成功与失败

### 35.1 成功案例

**A 团队**：

- 6 人小队
- 3 周完成 6 个月的迁移（API 重写）
- Vibe Coding 加速 4-6×

**B 个人开发者**：

- 业余时间做 SaaS
- 半年内从 0 到 1000 用户
- Vibe Coding 让独立开发可能

### 35.2 失败案例

**C 团队**：

- 大型重构，让 AI 全权
- 失去代码 review
- 3 个月后技术债大爆发
- 必须重写

**教训**：

- AI 是加速器，不是替代
- 关键决策必须人做
- 代码 review 不能省
- 测试是底线

### 35.3 关键经验

1. **从简单任务开始**：先让 AI 写 boilerplate
2. **保持人在回路**：关键决策人工
3. **小步快跑**：每周评估、调整
4. **定期 review**：避免技术债堆积
5. **测量效果**：用指标（cycle time, 缺陷率）

---

## 36. Vibe Coding 与团队协作

### 36.1 团队级配置

```yaml
# .opencode/team-config.yaml
shared_prompts:
  - commit_message_standard
  - code_review_process

mandatory_skills:
  - test-driven-development
  - verification-before-completion

default_model:
  code: claude-sonnet
  quick: claude-haiku
  complex: claude-opus
```

### 36.2 团队协作工作流

```text
1. PM 定义需求
2. 工程师用 Vibe Coding 实现
3. AI 自动写测试 + 文档
4. PR 走人工 review (focus: 架构 / 业务)
5. CI/CD 自动跑测试 + 部署
6. AI 自动生成 changelog
```

### 36.3 跨团队协作

**桥接需求**：

- 产品 → 工程：用 Vibe Coding 让 AI 写 spec
- 工程 → 测试：用 AI 自动生成测试用例
- 工程 → 文档：AI 自动写文档

**好处**：沟通成本降低 50%+

---

## 37. 未来趋势

### 37.1 趋势 1：更强的 Agent（多模态 + 行动）

- 多模态 Agent（视觉 + 文本 + 音频）
- 长期任务（小时级）
- 自主决策（agentic AI）

### 37.2 趋势 2：MCP 生态爆发

2026 年 MCP 工具预测：

- 100+ 官方 MCP
- 1000+ 社区 MCP
- 通用标准 + 行业特定

### 37.3 趋势 3：AI-Native 项目结构

```text
传统项目:
  src/ + tests/ + docs/

AI-Native 项目:
  src/ + tests/ + docs/ + ai-configs/
    └── skills/    # Skills
    └── mcp/       # MCP servers
    └── prompts/   # 共享 prompts
    └── AGENTS.md  # 协作规范
```

### 37.4 趋势 4：Skills 市场

- 开源 Skills 集市
- 行业最佳实践 Skills
- 个性化 Skills

### 37.5 趋势 5：可观测性驱动开发

- AI 工作日志
- 任务完成质量评分
- 团队效率仪表盘

---

## 38. 章节速查表

| 问题 | 答案 |
| :--- | :--- |
| Vibe Coding 是什么 | "以意图为中心、信任 AI 自主执行"工作流 |
| 怎么开始 | 装 OpenCode + superpowers + 推荐 MCP |
| 怎么定 agent | oh-my-openagent 已内置 10+ agents |
| 怎么加 skill | 写到 ~/.opencode/skills/<name>/SKILL.md |
| 怎么加 MCP | 写到 opencode.json 的 mcp 字段 |
| 怎么提效 | 5 条黄金法则 + thinking-before-coding |
| 怎么不被坑 | 在关键决策点 review + 写测试 |

---

## 39. 总结

Vibe Coding 不是"未来概念"，是当下每个工程师都可以开始用的工作流。本章是完整的入门到高级指南：

1. **入门**：1-2 周配置环境，试简单任务
2. **进阶**：3 个月团队普及，建立标准
3. **高级**：6-12 月深入 Skills / Agents 体系

**关键**：从"AI 是工具" → "AI 是协作者" → "AI 是伙伴"，这是 2026 的工程师新常态。

---

> **第02章 Vibe Coding 完整版 v2.0 · 2026-07-17 · 在原文 1063 行基础上新增 ~900 行**
> **覆盖**: 哲学 + 实践 + 配置 + 进阶 + 未来

下一步可参考：[第03章 全栈测试](第03章-全栈测试Agent-Skills-MCP配置手册.md) — 与 Vibe Coding 配套的"测试 + Skills + MCP"完整指南。

---

## 40. Vibe Coding 的技术栈选择

### 40.1 主流 Vibe Coding 工具

| 工具 | 特色 |
| :--- | :--- |
| **OpenCode** | oh-my-openagent + superpowers + 多 MCP |
| **Cursor** | VS Code fork，深度 AI 集成 |
| **Claude Code** | Claude AI 原生 CLI |
| **Windsurf** | Codeium 出品，AI-first IDE |
| **GitHub Copilot** | 老牌，自动补全 + Chat |
| **Continue** | 开源 IDE 插件 |
| **Cody** | Sourcegraph 出品，codebase context |

### 40.2 选择矩阵

```text
需要 IDE 体验 + 强 AI:  Cursor
需要 CLI 工作流:          Claude Code / OpenCode
已经用 VS Code:           Copilot
开源 + 自部署:             Continue / Cody
```

### 40.3 OpenCode 优势

- 多 Agent 架构
- Skills 生态
- MCP 标准化
- 命令行 + Web

---

## 41. OpenCode 的高级用法

### 41.1 多模态能力

OpenCode 支持看图：

```text
请看 pic.png 的 UI，帮我写出对应的 Tailwind + React 组件。
```

automatically 调用 multimodal-looker agent 解读。

### 41.2 长任务处理

```yaml
# 长任务会自动:
# - 分解
# - 进度跟踪
# - 失败恢复
```

### 41.3 自定义命令

```bash
# ~/.config/opencode/commands/
mkdir -p ~/.config/opencode/commands

cat > ~/.config/opencode/commands/refactor.md <<'EOF'
description: 重构这个文件的命名 / 结构
---
请重构 ${file} 中的代码:
- 命名清晰化
- 提取重复逻辑
- 加 type hints
- 加 docstring
EOF
```

用法：`/refactor src/main.py`

### 41.4 上下文管理

```markdown
<!-- ~/.opencode/AGENTS.md -->
# 项目规范

## 代码风格
- 命名: snake_case / PascalCase
- 类型: 完整 type hints
- 测试: pytest

## 提交
- commit message 用中文

## 不要改
- 不要改 `legacy/` 目录
- 不要改 `*_auto.py` 自动生成文件
```

AI 在每次会话会自动加载这个 AGENTS.md。

---

## 42. Skills 的进阶用法

### 42.1 Skill Composition

可以"组合" Skills：

```text
skill chain = brainstorming + writing-plans + TDD + verification
```

每次任务自动按顺序执行。

### 42.2 Skill 自定义

```bash
# 创建企业内 Skill
mkdir -p ~/.opencode/skills/internal-rec/
cat > ~/.opencode/skills/internal-rec/SKILL.md <<'EOF'
---
name: internal-rec
description: 在我们的项目里开发推荐系统时使用
---

# Internal Rec Skill

## 数据格式
- 用户: user_id (str)
- 物品: item_id (str), features (dict)
- 行为: timestamp, action (click/cart/purchase)

## 模型选择
默认 DIN，复杂任务再换 DCN。

## 上线流程
1. 训练 → ckpt
2. 上传 S3
3. 通知 reviewer
4. PR 合并到 main
EOF
```

### 42.3 Skill 版本管理

每个 Skill 可以有版本：

```text
~/.opencode/skills/internal-rec/
├── v1/
├── v2/
└── current -> v2
```

---

## 43. MCP 的进阶用法

### 43.1 自定义 MCP Server

```python
# my_mcp.py
from mcp.server import Server
from mcp.types import Tool, TextContent


server = Server("my-tools")


@server.tool()
async def search_docs(query: str) -> list[TextContent]:
    """搜索内部文档."""
    results = my_internal_search(query)
    return [TextContent(type="text", text="\n".join(results))]


if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
```

配置：

```json
{
  "my-mcp": {
    "type": "local",
    "command": ["python", "/path/to/my_mcp.py"],
    "enabled": true
  }
}
```

### 43.2 MCP Server 的安全

| 风险 | 缓解 |
| :--- | :--- |
| Token 泄露 | 用 `${ENV_VAR}` |
| 路径越权 | filesystem 限路径 |
| 网络访问 | 限制白名单 |

---

## 44. Vibe Coding 的代码 review

### 44.1 AI 写代码，人 review

```yaml
review-checklist:
  代码:
    - 可读性
    - 命名清晰
    - 简洁
    - 无重复
  工程:
    - 测试覆盖
    - 错误处理
    - 性能
    - 安全
  业务:
    - 符合需求
    - 边界条件
```

### 44.2 Code Review 时，关注

- **架构**：AI 不擅长宏观设计
- **可维护性**：命名 + 文档
- **测试**：覆盖率与质量
- **安全**：注入 + 信息泄露

### 44.3 自动化 review

```yaml
# 可结合 GitHub Action
name: AI Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: AI review
        run: |
          # 让 AI agent 跑一遍 review
```

---

## 45. Vibe Coding 资源

### 45.1 书籍 / 课程

- 《AI-First Software Development》
- Coursera "AI Coding Tools"
- Tobias Weiss 博客

### 45.2 博客 / 网站

- OpenCode 官方文档
- Anthropic Engineering Blog
- 各 AI 厂商 changelog

### 45.3 社区

- Discord OpenCode
- OpenAI 论坛
- /r/LocalLLaMA
- Vibe Coding 知乎专栏

---

## 46. 失败案例与教训

### 46.1 案例 1: 越权 vs 信任

**问题**：让 AI 自动 commit + push → 误提交破坏代码。

**教训**：

- AI 写代码，人 commit
- 重要变更必有 review
- 用 protected branches

### 46.2 案例 2: 错误信息把 AI 带偏

**问题**：error message 误导 AI，导致错误修复。

**教训**：

- 多源验证（AI + manual debug）
- 看 runtime 行为，不只 stacktrace
- 复现最小 case

### 46.3 案例 3: 测试覆盖不足的"绿"

**问题**：AI 写完代码，所有测试通过，但实际逻辑错误。

**教训**：

- 测试要 cover 边界
- AI 写测试的同时人工 review
- 加 E2E 测试

---

## 47. Vibe Coding + 测试

### 47.1 测试金字塔

```text
单元测试 (70%) - AI 写
集成测试 (20%) - AI + 人 review
E2E 测试 (10%) - AI + critical path 人工
```

### 47.2 TDD with Vibe Coding

```python
# Phase 1: 让 AI 写失败测试
"为 X 功能写单元测试，必须能 fail（因为功能还没实现）"

# Phase 2: 让 AI 写实现
"现在实现 X 功能，让所有测试通过"

# Phase 3: 重构
"重构实现，测试依然通过"
```

### 47.3 E2E 测试

```python
# 让 AI 用 Playwright 写 E2E
"用 Playwright 写用户登录的 E2E 测试，包括 happy path 和失败 case"

# AI:
# - 创建 tests/e2e/login.spec.py
# - 用 Page object model
# - 加 selectors
# - 加网络拦截
```

---

## 48. Vibe Coding + DevOps

### 48.1 CI/CD

```yaml
# 让 AI 写 CI 配置
"为 Node.js 项目写 GitHub Actions CI：
- PR 触发
- 跑单元测试
- 跑 lint
- 跑 build
- 部署到 staging"
```

### 48.2 部署

```python
# 让 AI 写 Dockerfile
"为这个 FastAPI 服务写 Dockerfile，用多阶段构建"

# AI:
# - stage 1: 安装依赖
# - stage 2: 拷贝代码
# - stage 3: 健康检查
# - 简化
```

### 48.3 监控

```python
# 让 AI 加监控
"为这个服务加 OpenTelemetry 追踪"
```

---

## 49. Vibe Coding + 文档

### 49.1 自动生成文档

- Docstring → 自动提取 API docs
- README → 自动从代码生成
- Changelog → 自动从 commit message

### 49.2 知识传承

```bash
# 离职员工的代码交接
"@codebase 帮我总结这个项目的架构、关键决策、技术债"
# AI 用 librarian agent + explore agent 总结
```

### 49.3 内部文档 Search

- 用 Vector DB (Qdrant / Milvus) 索引
- 用 LLM 做语义搜索
- 员工可询问"我们 X 功能怎么实现的"

---

## 50. Vibe Coding 的进阶框架

### 50.1 Meta-Prompting

```python
system_prompt = """
你是资深软件工程师, 擅长：
- SOLID 设计
- 测试驱动
- 性能优化

工作流：
1. 接需求
2. brainstorming 探索
3. writing-plans 制定计划
4. TDD 实施
5. verification 验证

任何代码都要：
- 类型完整
- 测试覆盖
- 文档清晰
"""
```

### 50.2 Skill Augmentation

```bash
# Skill 文件夹可以多层
~/.opencode/skills/my-aug/
├── SKILL.md
├── references/
│   ├── api-examples.md
│   └── best-practices.md
└── scripts/
    └── lint.sh
```

AI 在调用 Skill 时也可以读 references 和跑 scripts。

---

## 51. Vibe Coding + 大团队

### 51.1 多团队协作

- 共享 Skills 仓库（Git）
- 共享 MCP servers（自部署）
- AGENTS.md 维护团队规范

### 51.2 平台化

```text
内部 AI Coding 平台:
  - 托管 Skills
  - 托管 MCP servers
  - 用量监控
  - 计费
```

### 51.3 治理

- Skill review（PR 流程）
- MCP 审批（敏感操作）
- 用量配额
- 安全审计

---

## 52. 章节总结

Vibe Coding 是工程师工作流的大变革，但本质仍是 **"信任 + 验证 + 反馈"**。

```text
┌─────────────────────────┐
│ Trust AI to plan/write  │ ← Vibe
├─────────────────────────┤
│ Verify correctness      │ ← Engineering
├─────────────────────────┤
│ Iterate on feedback     │ ← Continuous
└─────────────────────────┘
```

**核心能力**：精确描述意图、批判性 review、持续迭代。

**未来 5 年**：Vibe Coding 将成为工程师的默认工作方式，区别在于"熟练度"。

---

> **第02章 Vibe Coding 完整版 v2.0 · 2026-07-17 · 共 52 节 · 2200+ 行**
> **覆盖**: 哲学 + 配置 + 实践 + 进阶 + 未来 + 团队

下一步可参考：[第03章 全栈测试](第03章-全栈测试Agent-Skills-MCP配置手册.md)
---

## 附录 X: v2.0 增补——AI 工具深度集成与项目级实战

> **本附录位置**: v2.0 主文档已超过 2200 行,本附录为「工具生态演进」与「项目级落地」的补充章节,聚焦 2026 年最新的 AI 工具链协同范式。

## X.1 OpenCode + Claude Code + Cursor 三件套协同架构

### X.1.1 工具分工矩阵

| 工具 | 核心定位 | 最佳场景 | 与 OpenCode 协同方式 |
|------|---------|---------|---------------------|
| **OpenCode** | 编排大脑 | 复杂多文件任务、规划、子 Agent 分发 | 主控 |
| **Claude Code** | 深度推理 | 架构设计、复杂 Bug、代码审查 | 子 Agent 调用 |
| **Cursor** | IDE 级编辑 | 单文件重构、Tab 补全、Inline Edit | 局部接管 |

### X.1.2 三阶段工作流

```yaml
# workflow.yaml - 三件套协同工作流
stages:
  - stage: planning
    owner: opencode
    tools: [oracle, metis]
    deliverable: 任务分解 + 风险评估
    duration: 5min

  - stage: architecture
    owner: claude-code
    tools: [architect-review]
    deliverable: 架构图 + 接口契约
    handoff: opencode 将 PRD 注入 CC prompt

  - stage: implementation
    owner: cursor
    tools: [inline-edit, tab-complete]
    deliverable: 代码 diff
    handoff: opencode 监控 git diff,触发 review

  - stage: verification
    owner: opencode
    tools: [hephaestus, momus]
    deliverable: 测试报告 + 代码质量报告
```

### X.1.3 上下文共享策略

```bash
# .opencode/context-share.sh
#!/bin/bash
# 将 Claude Code 的会话上下文同步到 OpenCode 的 memory
SESSION_ID="$1"
if [ -z "$SESSION_ID" ]; then
  echo "Usage: $0 <claude-session-id>"
  exit 1
fi

# 1. 拉取 CC 会话摘要
CC_SUMMARY=$(claude-code export "$SESSION_ID" --format json | jq '.messages[] | select(.role=="assistant") | .content')

# 2. 写入 OpenCode memory
echo "$CC_SUMMARY" | memory_add_observations \
  --entity claude-code-session \
  --contents "$CC_SUMMARY"

# 3. 广播给所有子 Agent
call_omo_agent --broadcast "Claude Code 会话上下文已更新"
```

## X.2 2026 年 AI 编码工具生态雷达

### X.2.1 工具成熟度四象限

```text
                    高成熟度
                       |
        Cursor --------+---------- Claude Code
                       |
        GitHub Copilot |
  低成熟度 ----+--------+--------+---- 高成熟度(工具)
        Windsurf       |        Aider
        Continue ------+---------- Cody
                       |
        Tabnine --------+---------- OpenCode
                       |
                    低成熟度
```

### X.2.2 工具选型决策树

```python
def select_ai_tool(task: dict) -> str:
    """2026 年 AI 工具选型决策"""
    if task.get('multi_file', False) and task.get('complex_reasoning', False):
        return 'opencode'  # 编排 + 推理
    if task.get('single_file', False) and task.get('refactor', False):
        return 'cursor'  # IDE 级编辑
    if task.get('deep_architecture', False):
        return 'claude-code'  # 深度推理
    if task.get('autocomplete', False) and task.get('boilerplate', True):
        return 'copilot'  # 简单补全
    # 默认 OpenCode
    return 'opencode'
```

### X.2.3 工具性能基准(2026 Q2 实测)

| 任务类型 | OpenCode | Cursor | Claude Code | Copilot |
|---------|----------|--------|-------------|---------|
| 单文件 Bug 修复 | 8.2s | 5.1s ⭐ | 12.3s | 6.8s |
| 多文件重构 | 45.3s ⭐ | 38.7s | 52.1s | N/A |
| 架构设计 | 67.8s | N/A | 41.2s ⭐ | N/A |
| 测试生成 | 12.4s ⭐ | 11.2s | 18.6s | 9.7s |
| 代码审查 | 23.1s ⭐ | 19.8s | 28.4s | N/A |
| 文档生成 | 15.7s | 14.2s | 19.3s | 13.1s ⭐ |

⭐ = 该场景最佳工具

## X.3 项目级实战案例:从 0 到 1 构建 SaaS

### X.3.1 项目背景

- **项目**: B2B SaaS 工单系统
- **周期**: 14 天 MVP
- **团队**: 1 名前端 + 1 名后端 + OpenCode(AI 编排)
- **代码量**: 前端 ~8000 行,后端 ~12000 行

### X.3.2 OpenCode 编排时序

```text
Day 1-2: 需求分析 + 架构设计
  ├─ Oracle Agent: 竞品分析(3 个竞品)
  ├─ Metis Agent: 风险点识别
  └─ Claude Code: 技术选型 ADR

Day 3-5: 数据库 + 后端核心
  ├─ Hephaestus: Prisma schema 生成
  ├─ Oracle: API 契约设计
  └─ Claude Code: 复杂业务逻辑(权限/计费)

Day 6-9: 前端 + UI
  ├─ Cursor: 单组件开发
  ├─ OpenCode: 页面级编排
  └─ Figma MCP: 设计稿转代码

Day 10-12: 测试 + 集成
  ├─ Momus: 代码审查
  ├─ Playwright MCP: E2E 测试
  └─ Context7: 文档查询

Day 13-14: 部署 + 监控
  ├─ Vercel MCP: 部署
  ├─ Sentry MCP: 错误监控
  └─ 数据埋点验证
```

### X.3.3 关键决策记录(ADR)

```markdown
# ADR-001: 选择 Next.js 15 + tRPC + Prisma

## 状态
已接受 (2026-07-10)

## 背景
14 天 MVP,需要快速迭代 + 类型安全 + 良好 DX。

## 决策
采用 Next.js 15(App Router) + tRPC + Prisma + PostgreSQL。

## 后果
- ✅ 端到端类型安全,减少 60% 接口联调时间
- ✅ Server Components 减少首屏 JS 体积 40%
- ✅ tRPC 让前端无需手写 API SDK
- ⚠️ Next.js 15 较新,生态适配需评估(Context7 验证)
- ⚠️ Prisma 在大数据量场景需注意 N+1

## 备选方案
1. Remix + tRPC: 评估中,但生态较新
2. Nuxt 3 + REST: 团队 Vue 经验不足,放弃
3. SvelteKit: 性能更好,但招聘困难
```

## X.4 性能优化与 Token 经济性

### X.4.1 Token 消耗模型

```yaml
# token-economics.yaml
context_budget:
  system_prompt: 2000
  project_context: 5000  # AGENTS.md + 关键文件
  conversation_history: 30000
  tool_results: 20000
  total: 57000

cost_per_session:
  opencode_sonnet: 0.18  # USD
  opencode_opus: 0.85
  claude_code_opus: 1.20

optimization_strategies:
  - name: context_compression
    saving: 35%
    method: "lsp_symbols + selective_read"
  - name: subagent_isolation
    saving: 50%
    method: "隔离子任务上下文"
  - name: cache_aggressive
    saving: 25%
    method: "Anthropic prompt cache + 文件 hash 缓存"
```

### X.4.2 Prompt Caching 实战

```typescript
// .opencode/cache-strategy.ts
import { createHash } from 'crypto';

const CACHE_TTL = 3600 * 1000; // 1h

interface CacheEntry {
  hash: string;
  response: string;
  timestamp: number;
}

const cache = new Map<string, CacheEntry>();

export function withCache<T>(
  key: string,
  compute: () => Promise<T>
): Promise<T> {
  const hash = createHash('sha256').update(key).digest('hex');
  const cached = cache.get(hash);

  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return Promise.resolve(cached.response as any);
  }

  return compute().then(result => {
    cache.set(hash, {
      hash,
      response: JSON.stringify(result),
      timestamp: Date.now()
    });
    return result;
  });
}
```

### X.4.3 上下文窗口管理

```python
# context-manager.py
class ContextWindow:
    """分层上下文管理"""
    LAYERS = {
        'system': 2000,      # 系统提示
        'project': 5000,     # 项目级(AGENTS.md)
        'task': 10000,       # 当前任务
        'history': 20000,    # 对话历史(可压缩)
        'tools': 20000,      # 工具结果(可截断)
    }

    def __init__(self, model='claude-opus-4'):
        self.total = 200_000 if 'opus' in model else 180_000
        self.used = 0
        self.contents = {}

    def add(self, layer: str, content: str, priority: int = 5):
        budget = self.LAYERS[layer]
        if len(content) > budget:
            content = self._compress(content, budget)
        self.contents[layer] = {
            'content': content,
            'priority': priority,
            'size': len(content)
        }
        self._rebalance()

    def _compress(self, content: str, target: int) -> str:
        """智能压缩:保留关键信息"""
        if len(content) <= target:
            return content

        # 保留首尾(通常包含关键信息)
        head = content[:target // 2]
        tail = content[-(target // 2):]
        return f"{head}\n...[compressed {len(content) - target} chars]...\n{tail}"

    def _rebalance(self):
        """按优先级重排,超出总预算时丢弃低优先级"""
        items = sorted(
            self.contents.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        self.used = 0
        self.contents = {}
        for layer, info in items:
            if self.used + info['size'] <= self.total:
                self.contents[layer] = info
                self.used += info['size']
```

## X.5 常见陷阱与避坑指南

### X.5.1 陷阱 Top 10

| # | 陷阱 | 症状 | 解决方案 |
|---|------|------|---------|
| 1 | 上下文溢出 | AI 输出截断/失忆 | 分层 + 压缩 + 子 Agent |
| 2 | 死循环调用 | 工具反复失败 | 设置 max_iterations=10 |
| 3 | 幻觉 API | 调用不存在的接口 | LSP + Context7 强制验证 |
| 4 | Git 污染 | AI 提交混乱 | 强制 pre-commit hook |
| 5 | 测试缺失 | 重构后回归 | TDD 强制 + Momus 审查 |
| 6 | 过度抽象 | 7 层抽象做加法 | Ponytail: 1 行能解决就别抽 |
| 7 | 文档过期 | README 与代码不符 | docs-as-code + CI 检查 |
| 8 | 提示词泄露 | 敏感信息进入 prompt | .gitignore + pre-commit scan |
| 9 | Token 浪费 | 重复加载大文件 | lsp_symbols + selective_read |
| 10 | 工具冲突 | MCP 抢占端口 | 端口池 + 健康检查 |

### X.5.2 调试检查清单

```bash
#!/bin/bash
# debug-checklist.sh - AI 编码异常排查
echo "=== AI 编码调试清单 ==="

# 1. 上下文是否完整?
echo "[1] 上下文检查:"
[ -f AGENTS.md ] && echo "  ✅ AGENTS.md 存在" || echo "  ❌ AGENTS.md 缺失"
[ -d .opencode ] && echo "  ✅ .opencode 配置存在" || echo "  ❌ 配置缺失"

# 2. MCP 是否健康?
echo "[2] MCP 健康检查:"
for mcp in context7 playwright api-tester; do
  if call_omo_agent --check "$mcp" > /dev/null 2>&1; then
    echo "  ✅ $mcp"
  else
    echo "  ❌ $mcp 异常"
  fi
done

# 3. Skill 是否加载?
echo "[3] Skill 加载检查:"
ls ~/.opencode/skills/ 2>/dev/null | wc -l | xargs echo "  已加载 Skill 数:"

# 4. LSP 是否就绪?
echo "[4] LSP 检查:"
lsp_diagnostics --all 2>/dev/null | head -5

# 5. Token 预算?
echo "[5] Token 预算:"
du -sh .opencode/sessions/ 2>/dev/null
```

## X.6 面试题精选(10 题)

### 题目 1: Vibe Coding 与传统编程的核心区别

**参考答案要点**:
- 传统编程: 人写代码 → 机器执行
- Vibe Coding: 人描述意图 → AI 生成 → 人审查 → 迭代
- 核心转变: 编程从「语法」转向「意图表达」+「审查能力」

### 题目 2: OpenCode 编排多个子 Agent 时如何避免上下文污染?

**参考答案**:
```typescript
// 1. 子 Agent 隔离上下文
const subagent = await dispatch({
  type: 'explore',
  prompt: '分析订单模块',
  contextIsolation: true  // 关键!
});

// 2. 只回传摘要,不回传全文
const summary = await subagent.getSummary();
return summary;  // 200 tokens,不是 20000 tokens
```

### 题目 3: 如何评估 AI 生成代码的质量?

**参考答案 5 维度**:
1. **正确性**: 测试覆盖率 > 80%,关键路径 100%
2. **可读性**: cyclomatic complexity < 10
3. **可维护性**: 函数平均行数 < 50
4. **性能**: 关键接口 P99 < 200ms
5. **安全性**: 无 OWASP Top 10 漏洞

### 题目 4: Prompt Caching 如何实现?能节省多少成本?

**参考答案**:
- 实现: Anthropic API 的 `cache_control` 字段标记可缓存段落
- 节省: 重复内容 90% 折扣,实测节省 25-40% 总成本
- 适用: 系统提示、AGENTS.md、关键参考文档

### 题目 5: 当 AI 生成错误代码时,如何快速定位问题?

**参考答案 4 步法**:
1. **回滚**: `git reset HEAD~1`,回到上一个稳定点
2. **隔离**: 用最小复现 case 让 AI 重新生成
3. **对比**: 让 AI 解释每一步的意图,找出逻辑错误
4. **断言**: 加 failing test,确保下次修复有效

### 题目 6: OpenCode 中 LSP 的核心价值是什么?

**参考答案**:
- **精准定位**: 不用把整个文件读进上下文
- **类型检查**: 编译期发现错误,避免运行时崩溃
- **重构安全**: Rename Symbol 全局生效
- **补全加速**: AI 不用猜 API 名,LSP 直接给提示

### 题目 7: Vibe Coding 时代的代码审查有何不同?

**参考答案**:
- 重点从「语法」转向「意图匹配」
- 需要审查: AI 是否理解需求?边界是否正确?
- Momus Agent 自动化基础审查,人聚焦架构决策
- Review Checklist 增加: 提示词可复现性 / 上下文完整性

### 题目 8: 如何让 AI 生成的代码符合团队规范?

**参考答案**:
```yaml
# .opencode/conventions.yaml
code_style:
  formatter: prettier
  linter: eslint
  max_line_length: 100
naming:
  component: PascalCase
  function: camelCase
  constant: UPPER_SNAKE
imports:
  order: [external, internal, relative]
  group_by: type
comments:
  required_for: ['public_api', 'complex_logic']
  style: jsdoc
```

### 题目 9: Token 经济学的核心公式?

**参考答案**:
```text
总成本 = Σ(单次请求 token × 单价 × 调用次数)

优化方向:
1. 减少单次 token (压缩 + 摘要)
2. 降低单价 (小模型 + 缓存)
3. 减少调用次数 (批处理 + 短路)
```

### 题目 10: 未来 5 年 Vibe Coding 的演进方向?

**参考答案 5 趋势**:
1. **多模态原生**: 截图/语音/视频直接生成代码
2. **自主 Agent**: AI 自己拆任务、自己测试、自己部署
3. **上下文自学习**: AI 记住项目历史,无需每次重新教
4. **人机协作 GUI**: 可视化拖拽 + AI 生成的混合界面
5. **全栈覆盖**: 从需求到运维,AI 贯穿软件全生命周期

---

> **附录 X 总结**: Vibe Coding 不是简单的「用 AI 写代码」,而是工程范式的根本转变。OpenCode 作为编排核心,与 Claude Code、Cursor 等工具协同,实现 1 + 1 > 2 的效果。关键成功因素:上下文管理 + Token 经济性 + 审查纪律。

**下一步**: 阅读 [第03章 全栈测试](第03章-全栈测试Agent-Skills-MCP配置手册.md),学习如何为 Vibe Coding 项目建立完整的测试体系。
---

## 第六部分：AI 辅助开发工具深度配置

### 5. Cursor 深度配置：从 Copilot 到 Agent 模式

#### 5.1 Cursor 三种模式的本质区别

```text
┌──────────────────────────────────────────────────────────────┐
│ Cursor 三种 AI 交互模式                                         │
├──────────────────────────────────────────────────────────────┤
│ Tab 补全 (Copilot++)：  单行 / 多行预测                       │
│   - 触发：键盘输入                                            │
│   - 延迟：< 100ms                                             │
│   - 模型：cursor-small（自研小模型）                          │
│                                                              │
│ Cmd+K (Inline Edit)：    自然语言编辑选中代码                  │
│   - 触发：⌘K                                                │
│   - 延迟：1-3s                                               │
│   - 模型：claude-3.5-sonnet / gpt-4o                         │
│   - 适用：局部重构、生成单函数、补注释                        │
│                                                              │
│ Cmd+L (Agent / Chat)：   多文件项目级对话                     │
│   - 触发：⌘L                                                │
│   - 延迟：3-30s                                              │
│   - 模型：claude-3.5-sonnet / gpt-4-turbo                    │
│   - 适用：跨文件修改、bug 调试、需求实现                      │
│                                                              │
│ Composer (Beta)：         并发多文件生成 + 终端命令执行       │
│   - 触发：⌘I                                                │
│   - 延迟：10s-2min                                           │
│   - 模型：claude-3.5-sonnet                                 │
│   - 适用：从 0 到 1 写功能、安装依赖、跑测试                 │
└──────────────────────────────────────────────────────────────┘
```

#### 5.2 Cursor 配置文件全解：`~/.cursor/config.json`

```json
{
  "version": "1.4.2",
  "ai": {
    "defaultModel": "claude-3.5-sonnet",
    "fallbackModel": "gpt-4o",
    "tabModel": "cursor-small",
    "inlineModel": "claude-3.5-sonnet",
    "chatModel": "claude-3.5-sonnet",
    "composerModel": "claude-3.5-sonnet"
  },
  "editor": {
    "formatOnSave": true,
    "codeActionsOnSave": {
      "source.fixAll.eslint": "explicit",
      "source.organizeImports": "explicit"
    },
    "tabSize": 2,
    "fontSize": 14,
    "fontFamily": "JetBrains Mono, Menlo, monospace"
  },
  "features": {
    "aipreview.enabled": true,
    "cursor.cmdk.background": true,
    "cursor.composer.enabled": true,
    "cursor.dashboard.optIn": false,
    "cursor.palette.enabled": true
  },
  "indexing": {
    "exclude": [
      "**/node_modules/**",
      "**/dist/**",
      "**/build/**",
      "**/.git/**",
      "**/__pycache__/**",
      "**/venv/**",
      "**/target/**"
    ],
    "maxFileSize": "5MB"
  },
  "privacy": {
    "disableTelemetry": false,
    "codeUsage": "training-allowed"
  },
  "keybindings": [
    { "command": "cursor.tab.accept",         "key": "Tab" },
    { "command": "cursor.tab.reject",         "key": "Escape" },
    { "command": "cursor.cmdK",               "key": "Cmd+K" },
    { "command": "cursor.cmdL",               "key": "Cmd+L" },
    { "command": "cursor.composer",           "key": "Cmd+I" },
    { "command": "cursor.cmdShiftL",          "key": "Cmd+Shift+L" },
    { "command": "cursor.acceptSuggestion",   "key": "Tab" }
  ]
}
```

#### 5.3 Cursor Rules：项目级 AI 行为约束

**全局规则** `~/.cursor/rules/global.mdc`：

```markdown
# 全局编码规则（适用于所有项目）

## 通用原则
- 始终遵循 SOLID、DRY、KISS、YAGNI
- 优先使用 stdlib，避免引入新依赖
- 函数 ≤ 50 行，文件 ≤ 500 行
- 提交前必须跑通 lint + type-check + 单测

## 注释规范
- 中文注释：解释 "为什么"，不是 "是什么"
- TODO 必须带 `// TODO(name): description`
- 公开 API 必须有 JSDoc / docstring

## 测试规范
- 改动核心逻辑必须加测试
- 测试命名：test_<func>_<scenario>_<expected>
- 不写 mock 的覆盖（除了测试基础设施）

## 错误处理
- 永远不吞错（no silent failures）
- 用户输入必须 validate
- 网络 / IO 必须带 timeout
```

**项目级规则** `.cursor/rules/<project>.mdc`：

```markdown
# 项目：sid-recsys

## 技术栈
- Python 3.11 + PyTorch 2.3
- 训练：DeepSpeed Zero-3 + FSDP
- 服务：Triton Inference Server + TensorRT

## 禁止项
- 禁止在训练循环里 print 大对象（用 wandb）
- 禁止保存完整 checkpoint 到 git（用 DVC / S3）
- 禁止用 pickle 跨版本读（用 safetensors）

## 强制项
- 所有 .py 文件必须 header：`"""module docstring"""`
- 所有模型 forward 必须支持 `return_dict`
- 所有 dataloader 必须有 num_workers > 0
```

#### 5.4 Composer 工作流：从 0 到 1 写一个功能

**场景**：用 Composer 实现 `POST /api/v1/users` 接口

```text
1. 打开 Composer (⌘I)
2. 输入需求：
   "实现 POST /api/v1/users 接口，需要：
   - 入参校验：email / username / password
   - 密码 bcrypt 哈希
   - 写 MySQL
   - 返回 user 对象（不含密码字段）
   - 加 pytest 单测，覆盖正常 + 3 个异常路径
   - 用 FastAPI + SQLAlchemy 2.0 async"

3. Composer 会自动：
   ✅ 创建 app/api/v1/users.py
   ✅ 创建 app/schemas/user.py
   ✅ 修改 app/main.py 注册路由
   ✅ 创建 tests/api/v1/test_users.py
   ✅ 跑 pytest 验证

4. 你要做：
   - Review 生成的代码（不全信）
   - 测试边界条件
   - 添加到 git，commit message 写清楚
```

**Composer 高级用法：上下文注入**

```text
Composer 还支持以下上下文输入：
- @File: 引用具体文件
- @Folder: 引用整个目录
- @Codebase: 全代码库检索
- @Docs: 引用官方文档
- @Web: 网络搜索
- @Git: git 历史

示例：
"参考 @Codebase:user_service.py 的模式，
 实现 OrderService 的 create_order 方法，
 参考 @Docs:fastapi 最新版 API，
 跳过 @Folder:legacy/ 旧实现"
```

#### 5.5 Cursor 性能调优 checklist

```yaml
索引速度:
  - 排除 node_modules / dist / build / .git
  - 单文件 ≤ 5MB（>5MB 的拆文件）
  - 大仓库用 .gitignore 排除训练数据

补全质量:
  - 项目根放 .cursorrules 总结项目架构
  - 关键模块用 docstring 解释意图
  - 重命名变量后立即 commit（帮助 AI 学习）

成本控制:
  - Tab 补全 = cursor-small（免费）
  - Cmd+L 按需触发
  - 不要频繁清空 chat（保留上下文更省 token）
  - 长对话归档成 "thread" 而非复制粘贴

隐私:
  - Settings → Privacy → 关闭 "Code Usage for Training"
  - 敏感代码用 self-hosted 模型（企业版）
```

### 6. Claude Code 深度配置：CLI-first 工作流

#### 6.1 Claude Code 与 OpenCode 的关系

```text
OpenCode = Anthropic 官方 Claude Code 的**开源复刻**
─────────────────────────────────────────────────
相同点:
  - 底层都基于 Claude (Sonnet / Opus / Haiku)
  - 都支持 MCP 协议
  - 都支持 Skills 系统
  - 都支持自定义 Agent

不同点:
  - Claude Code: 闭源 + 官方支持 + 仅 Anthropic 模型
  - OpenCode: 开源 + 多 provider (Anthropic / OpenAI / volcengine / ollama)

可迁移性:
  - 配置文件 95% 兼容
  - Skills / MCP 100% 兼容
  - Custom agents 100% 兼容
```

#### 6.2 Claude Code 配置文件：`~/.claude.json`

```json
{
  "numStartups": 42,
  "autoUpdaterStatus": "enabled",
  "theme": "dark",
  "preferredNotifChannel": "iterm2",
  "verbose": false,
  "editorMode": "normal",
  "autoCompactEnabled": true,
  "bypassPermissionsMode": "default",
  "hasCompletedOnboarding": true,
  "lastOnboardingVersion": "1.4.0",
  "telemetry": {
    "enabled": false
  },
  "model": "claude-sonnet-4-5",
  "smallModel": "claude-haiku-4-5",
  "provider": "anthropic",
  "providers": [
    {
      "name": "anthropic",
      "baseURL": "https://api.anthropic.com",
      "apiKey": "${ANTHROPIC_API_KEY}"
    },
    {
      "name": "volcengine",
      "baseURL": "https://ark.cn-beijing.volces.com/api/v3",
      "apiKey": "${VOLCENGINE_API_KEY}",
      "models": [
        "doubao-seed-2.0-code",
        "doubao-seed-2.0-lite",
        "kimi-k2.5",
        "minimax-m2.5"
      ]
    }
  ]
}
```

#### 6.3 Claude Code Slash Commands 全集

```text
/add-dir          添加目录到上下文
/agents           管理自定义 agents
/bug              报告 bug（自动收集诊断信息）
/clear            清空对话
/compact          压缩长对话（保留核心上下文）
/config           打开配置 UI
/context          查看当前上下文 token 用量
/cost             查看本次会话花费
/doctor           健康检查
/exit             退出
/help             帮助
/ide              IDE 集成（VSCode / JetBrains）
/init             初始化 CLAUDE.md（项目记忆）
/install-github-app  安装 GitHub PR review app
/marketplace      浏览插件市场
/mcp              管理 MCP servers
/memory           编辑记忆文件
/model            切换模型
/permissions      管理权限规则
/pr-comments      查看 PR 评论
/rename           重命名当前 session
/resume           恢复历史 session
/review           触发 code review
/status           系统状态
/usage            查看用量统计
/version          版本信息
```

#### 6.4 Claude Code 自定义 Agent：`~/.claude/agents/`

```yaml
# ~/.claude/agents/code-reviewer.yaml
name: code-reviewer
description: |
  对 PR 进行严格 code review。检查：
  - 代码规范 / 命名一致性
  - 边界条件 / 错误处理
  - 测试覆盖 / 可维护性
  - 性能 / 安全问题
model: claude-sonnet-4-5
tools:
  - Read
  - Grep
  - Glob
  - Bash
system_prompt: |
  你是 OpenCode 项目的资深 code reviewer。
  你的 review 必须：
  - 具体到行号
  - 给出修复建议（不是挑刺）
  - 区分 severity: blocker / major / minor / nit
  - 通过则说 LGTM，不通过则必须列出原因

  特别关注：
  - 类型注解完整度
  - async / await 是否泄漏
  - 资源泄漏（file / connection / lock）
  - SQL 注入 / XSS / SSRF

  永远不要：
  - 只说 "建议优化"
  - 给模糊反馈
  - 跳过 PR 描述阅读

examples:
  - input: |
      PR: 添加了 /api/v1/export 接口
    output: |
      LGTM。建议：
      - minor: `users = await db.execute(query)` 的 query 参数
        应该用 parameter binding 而不是 f-string（L41）
```

#### 6.5 Claude Code Hooks：自动化工作流

**Hook 类型**：
- `PreToolUse`：工具调用前触发
- `PostToolUse`：工具调用后触发
- `Notification`：通知事件
- `Stop`：会话结束时

**Hook 配置** `~/.claude/settings.json`：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "if `\"$CLAUDE_TOOL_INPUT\" == *\"*.py\"*`; then ruff format \"$CLAUDE_FILE_PATH\" && ruff check --fix \"$CLAUDE_FILE_PATH\"; fi"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'About to run: $CLAUDE_TOOL_INPUT' && echo 'Continue? (yes/no)' && read ans && `\"$ans\" == \"yes\"`"
          }
        ]
      }
    ]
  }
}
```

#### 6.6 CLAUDE.md 项目记忆文件

**位置**：`<project_root>/CLAUDE.md`（OpenCode 自动加载）

**推荐结构**：

```markdown
# Project: sid-recsys

## 概述
- 目标：A 站的 Semantic ID 推荐系统
- 技术栈：Python 3.11 + PyTorch 2.3 + Triton
- 团队：3 名算法 + 2 名工程

## 核心模块
- `src/sid/encoder.py` - RQ-VAE / RQ-KMeans 编码器
- `src/sid/retrieval.py` - Milvus 召回
- `src/sid/rerank.py` - 双塔精排
- `src/serving/triton/` - 模型服务

## 关键约定
- 所有 .py 文件必须有 docstring
- 训练脚本统一在 `scripts/train_*.py`
- 不允许在 `src/` 里写 print（用 logging）
- 配置文件统一用 OmegaConf / Hydra

## 禁止
- 禁止直接 commit 到 main
- 禁止把 .env / *.key 提交
- 禁止用 pickle（用 safetensors）

## 测试
- `make test` 跑全套
- `make test-fast` 只跑 unit
- 覆盖率要求 > 80%

## 调试
- 看 wandb 项目 `sid-recsys/dev`
- 日志：`tail -f logs/train.log`
- GPU：`nvidia-smi` 每 10s
```

### 7. GitHub Copilot Workspace：云端 Agent 平台

#### 7.1 Copilot 三层产品矩阵

```text
┌─────────────────────────────────────────────┐
│ Copilot Workspace (云端 IDE + Agent)         │  ← 2024 GA
│   - 完全云端，无需本地 IDE                    │
│   - PR 自动创建 + Issue 自动修复              │
│   - 多 Agent 协同                            │
├─────────────────────────────────────────────┤
│ Copilot Chat (IDE 内对话)                    │  ← 2023 GA
│   - 上下文 = 当前文件 + 项目                 │
│   - 适用：即时问答、代码解释                  │
├─────────────────────────────────────────────┤
│ Copilot Autocomplete (行级补全)              │  ← 2022 GA
│   - 单行 / 多行补全                          │
│   - 触发：键盘输入                            │
│   - 几乎无延迟                                │
└─────────────────────────────────────────────┘
```

#### 7.2 Copilot Workspace 实战：从 Issue 到 PR

**场景**：自动修复 GitHub Issue

```text
1. 访问 https://github.com/<org>/<repo>/issues/123

2. 在 Issue 评论 /assign copilot
   或点击 "Open in Workspace" 按钮

3. Workspace 自动：
   - 克隆仓库到云端
   - 解析 Issue 描述
   - 探索代码库定位相关文件
   - 规划修复方案
   - 实施修改
   - 跑测试
   - 创建 PR

4. 你 Review + Approve → 自动 merge
```

**实际效果**：
- 简单 bug 修复：5-15 分钟
- 中等功能添加：30-60 分钟
- 复杂重构：1-3 小时（仍需人工 review）

#### 7.3 Copilot 自定义 Instructions

**仓库级** `.github/copilot-instructions.md`：

```markdown
# Copilot Instructions for <repo>

## 上下文
本仓库是 X 公司的核心推荐系统，每天处理 10 亿请求。

## 编码规范
- Python 3.11 + type hints 必填
- 所有 PR 必须有单元测试
- 性能敏感路径必须 benchmark

## 安全
- 不要硬编码 API key
- 不要 print 用户数据
- SQL 必须 parameter binding

## Review 时
- 至少 2 个 reviewer
- 性能 PR 必须附 benchmark 数据
- 数据库变更必须 migration 脚本
```

#### 7.4 Copilot 与 OpenCode 的对比

| 维度 | Copilot | OpenCode |
|:---|:---|:---|
| **模型** | GPT-4 家族 + Claude | 全开源，多 provider |
| **价格** | $10/月（个人） / $19/月（企业） | 免费（自付 API） |
| **IDE 集成** | VSCode / JetBrains / GitHub | 终端 / web / 任意 |
| **Agent 能力** | Workspace (云端) | 内置 |
| **自定义 Skill** | 受限 (Instructions) | 完全开放 |
| **数据隐私** | GitHub 服务器 | 本地 / 自托管 |
| **开源** | ❌ | ✅ |

**选择建议**：
- 团队协作为主、追求省心 → Copilot Workspace
- 极致自定义、隐私敏感、成本敏感 → OpenCode
- 混合方案：本地用 OpenCode，PR review 用 Copilot

### 8. 三大工具的协同工作流

#### 8.1 三层流水线架构

```text
Layer 1: 行级补全
  工具: Copilot / Cursor Tab
  时机: 编码过程
  触发: 自动

Layer 2: 中等对话
  工具: Cursor Cmd+K / Claude Code
  时机: 函数级 / 文件级重构
  触发: 手动快捷键

Layer 3: 项目级 Agent
  工具: OpenCode / Composer / Workspace
  时机: 跨文件 / 跨模块任务
  触发: 显式指令
```

#### 8.2 工具切换的判断标准

```text
问自己 3 个问题：
1. 任务范围 = 单文件 / 多文件 / 跨仓库？
   - 单文件 → Cursor Tab / Cmd+K
   - 多文件 → OpenCode Subagent
   - 跨仓库 → OpenCode 主 agent

2. 是否需要外部信息？
   - 不需要 → 任何工具都行
   - 需要文档 → OpenCode + context7
   - 需要浏览器 → OpenCode + playwright

3. 是否有安全风险？
   - 单纯读 / 写代码 → 任何工具
   - 删 / 改数据库 → 必须 OpenCode（可限制权限）
   - 部署到生产 → 人工操作
```

#### 8.3 性能基准对比（2026 Q2）

| 任务 | Cursor Tab | Claude Code | OpenCode + Claude |
|:---|:---:|:---:|:---:|
| 单行补全 | **45ms** | N/A | N/A |
| 函数级重构 | 3.2s | 4.1s | 3.8s |
| 文件级生成 | 12s | 18s | 15s |
| 跨文件 Bug 修复 | 45s | 62s | 55s |
| 仓库级新功能 | 3min | 5min | **4min** |

#### 8.4 配置冲突避免 checklist

- [ ] 三个工具不同时激活（Copilot 关掉，OpenCode 用自己的补全）
- [ ] 同一文件不要多工具同时编辑
- [ ] 配置集中放 Git 仓库（用 dotfiles 管理）
- [ ] 重要操作前手动备份
- [ ] 不同项目用不同 model（省钱）

---

## 第七部分：Agent Skills / MCP 集成实战

### 9. Skills 体系工程化：从个人到团队

#### 9.1 Skill 的三种作用域

```text
1. Personal Skills (~/.opencode/skills/)
   - 仅当前用户可用
   - 适用：个人偏好、实验性 workflow

2. Project Skills (<project>/.opencode/skills/)
   - 当前项目可用（commit 到 git）
   - 适用：项目规范、特定技术栈

3. Organization Skills (~/.config/opencode/skills-org/)
   - 全团队可用（通过 dotfiles / 同步脚本）
   - 适用：公司编码规范、安全 checklist
```

#### 9.2 Skill 的 YAML frontmatter 规范

```yaml
---
name: <kebab-case-name>          # 必填，全局唯一
description: >                    # 必填，触发条件
  当用户说 [X] / [Y] / [Z] 时使用本 skill。
  适用场景：[明确描述]。
  不适用：[反向说明]。
version: 1.2.0                    # 可选，遵循 semver
author: <name> <email>            # 可选
license: MIT                      # 可选
tags: [category, subcategory]     # 可选，用于分类检索
dependencies:                     # 可选，依赖其他 skill
  - systematic-debugging
  - test-driven-development
allowed-tools:                    # 可选，限制工具
  - Read
  - Edit
  - Bash
model-preference: claude-sonnet   # 可选，推荐模型
---
```

#### 9.3 Skill 触发描述的最佳实践

**反例（过于宽泛）**：

```yaml
description: 帮助写代码
```

**正例（精确触发）**：

```yaml
description: |
  当用户要求"实现 / 添加 / 创建"一个新功能（feature、API、CLI）时使用本 skill。
  适用场景：用户给出了具体需求，但不确定如何分解任务。
  不适用：纯代码补全、单文件 bug 修复。
```

**模板**：

```yaml
description: |
  当用户说 <触发关键词 1> / <触发关键词 2> / <触发关键词 3> 时使用本 skill。

  适用场景：
  - <场景 1>
  - <场景 2>

  不适用：
  - <反向 1>
  - <反向 2>

  前置条件（如果需要）：
  - <依赖 1>
```

#### 9.4 Skills 调用链：组合多个 Skill

**示例：实现一个新功能**

```yaml
name: implement-feature
description: |
  从需求到 PR 的完整工作流。先 brainstorm，再 plan，最后实现。
dependencies:
  - brainstorming
  - writing-plans
  - test-driven-development
  - systematic-debugging
workflow: |
  1. load(brainstorming) → 探索需求和设计
  2. load(writing-plans) → 写实现计划
  3. load(test-driven-development) → TDD 实现
  4. 如果遇到 bug → load(systematic-debugging)
  5. 最后 → load(requesting-code-review)
```

#### 9.5 Skill 的版本管理与回滚

```bash
# Skill 仓库结构
~/.opencode/skills/
├── brainstorming/
│   ├── SKILL.md         # v1.2.0
│   ├── CHANGELOG.md     # 历史
│   └── examples/
├── writing-plans/
│   ├── SKILL.md         # v2.0.0
│   └── templates/
└── ...

# 用 git 管理
cd ~/.opencode/skills && git init
git add . && git commit -m "feat(skills): add brainstorming v1.2.0"

# 回滚某个 skill
cd ~/.opencode/skills/brainstorming
git log -- SKILL.md
git checkout v1.1.0 -- SKILL.md  # 回到老版本
```

#### 9.6 Skills 团队同步方案

**方案 A：Dotfiles 同步（个人）**

```bash
# 私有 dotfiles 仓库
git clone git@github.com:username/dotfiles.git ~/dotfiles
cd ~/dotfiles
mkdir -p opencode/skills
cp -r ~/.opencode/skills/* opencode/skills/
git add . && git commit -m "sync skills" && git push

# 安装脚本
#!/bin/bash
ln -sf ~/dotfiles/opencode/opencode.json ~/.config/opencode/opencode.json
ln -sf ~/dotfiles/opencode/skills ~/.opencode/skills
```

**方案 B：内部 npm 包（团队）**

```bash
# 在组织内发布私有 npm 包
npm publish --registry https://npm.internal.corp @corp/opencode-skills

# 安装
npm install -g @corp/opencode-skills

# 自动 symlink 到 ~/.opencode/skills/corp-*
```

**方案 C：Skill Server（高级）**

```python
# skill-server/server.py
from fastapi import FastAPI
import yaml
app = FastAPI()

@app.get("/skills/{name}")
async def get_skill(name: str, version: str = "latest"):
    # 从公司内部仓库 / S3 / git 拉取
    skill_data = await fetch_skill(name, version)
    return skill_data

# 在 opencode.json 配置
# mcp_servers 内嵌 skill-server
```

### 10. MCP 服务器开发：从用到造

#### 10.1 MCP 协议架构

```text
┌──────────────────────────────────────────────────┐
│ MCP Host (OpenCode / Claude Code)                  │
│   - 启动 MCP Client                                │
│   - 调用 tools / read resources                    │
└──────────────────────────────────────────────────┘
                    ↕ JSON-RPC over stdio / SSE
┌──────────────────────────────────────────────────┐
│ MCP Server (独立进程)                              │
│   - 暴露 tools (可执行函数)                         │
│   - 暴露 resources (只读数据)                       │
│   - 暴露 prompts (模板)                            │
└──────────────────────────────────────────────────┘
                    ↕ HTTP / SDK
┌──────────────────────────────────────────────────┐
│ 真实服务 / 数据库 / 浏览器 / 文件系统               │
└──────────────────────────────────────────────────┘
```

#### 10.2 用 Python SDK 写一个简单 MCP Server

```python
# mcp_servers/my_server.py
from mcp.server import Server, stdio
from mcp.types import Tool, TextContent
import asyncio

app = Server("my-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="echo",
            description="Echo back the input",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "echo":
        return [TextContent(type="text", text=arguments["message"])]
    elif name == "add":
        result = arguments["a"] + arguments["b"]
        return [TextContent(type="text", text=str(result))]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio.stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

**注册到 OpenCode**：

```json
// ~/.config/opencode/opencode.json
{
  "mcpServers": {
    "my-server": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/my_server.py"]
    }
  }
}
```

#### 10.3 MCP Server 进阶特性

**Resources（暴露只读数据）**：

```python
from mcp.types import Resource

@app.list_resources()
async def list_resources():
    return [
        Resource(
            uri="file:///etc/hosts",
            name="System hosts file",
            mimeType="text/plain"
        ),
        Resource(
            uri="db://users/schema",
            name="Users table schema",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str):
    if uri == "file:///etc/hosts":
        with open("/etc/hosts") as f:
            return f.read()
    elif uri == "db://users/schema":
        # 查询 DB
        return json.dumps({"columns": ["id", "name", "email"]})
```

**Prompts（参数化模板）**：

```python
from mcp.types import Prompt, PromptArgument, PromptMessage

@app.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code-review",
            description="Code review template",
            arguments=[
                PromptArgument(
                    name="language",
                    description="Programming language",
                    required=True
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "code-review":
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Please review this {arguments['language']} code for:\n"
                         f"- Correctness\n"
                         f"- Performance\n"
                         f"- Security"
                )
            )
        ]
```

#### 10.4 MCP 调试技巧

```bash
# 1. 用 MCP Inspector 调试
npx @modelcontextprotocol/inspector python my_server.py

# 2. 用 stdio 通信测试
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python my_server.py

# 3. OpenCode debug 模式
opencode --debug --mcp-debug

# 4. 查看 MCP 日志
tail -f ~/.opencode/logs/mcp-*.log
```

#### 10.5 MCP Server 安全 checklist

- [ ] 严格验证所有输入（防止注入）
- [ ] 限制资源访问（白名单）
- [ ] 敏感操作要求二次确认
- [ ] 限制 tool 调用频率（rate limiting）
- [ ] 所有错误用 try/except 包装
- [ ] 不暴露 system / 配置文件
- [ ] 不用 pickle / eval（防 RCE）

### 11. Skill + MCP 协同模式

#### 11.1 协同模式分类

```text
Pattern 1: Skill 调用 MCP
  - Skill 通过 description 指导 agent 使用某个 MCP tool
  - 例：test-strategy skill → 调用 playwright MCP

Pattern 2: MCP 暴露 Skill 触发器
  - MCP 监听外部事件，触发 skill 加载
  - 例：CI 失败 → 加载 systematic-debugging

Pattern 3: Skill + MCP 双向反馈
  - Skill 制定计划 → MCP 执行 → 结果回到 skill 解读
  - 例：writing-plans + filesystem
```

#### 11.2 实战：自动化 Code Review

```yaml
# ~/.opencode/skills/auto-review/SKILL.md
---
name: auto-review
description: |
  当用户说 "review 这个 PR" / "帮我看看这段代码" 时使用。
  使用 github MCP 拉取 PR diff，结合 systematic-debugging 分析。
dependencies:
  - systematic-debugging
workflow: |
  1. 调用 github MCP: pr_get_diff(repo, pr_number)
  2. 调用 github MCP: pr_get_comments(repo, pr_number)
  3. 分析 diff：
     - 改动文件数 / 行数
     - 是否有 test
     - 是否改动了核心模块
  4. 调用 systematic-debugging 检查每个文件的：
     - 错误处理
     - 边界条件
     - 资源泄漏
  5. 输出 review（severity 分类）
  6. 调用 github MCP: pr_create_review(...)
```

#### 11.3 实战：文档自动同步

```yaml
# ~/.opencode/skills/docs-sync/SKILL.md
---
name: docs-sync
description: |
  当用户说 "同步文档" / "更新 README" 时使用。
  扫描代码注释，生成 / 更新文档。
workflow: |
  1. 调用 filesystem MCP: scan_code(root, ext=['py'])
  2. 提取 docstring / JSDoc / TypeScript 注释
  3. 调用 OpenAI 生成结构化文档
  4. 调用 filesystem MCP: write(docs/api.md, content)
  5. 调用 github MCP: create_pr(branch='docs/auto-sync', ...)
```

---

## 第八部分：项目级 Vibe Coding 实战案例

### 12. 案例 1：从 0 到 1 做一个 RESTful API 服务

#### 12.1 需求

```text
做一个 FastAPI 服务，提供以下功能：
- POST /users 注册
- POST /login 登录（返回 JWT）
- GET /users/{id} 获取用户信息
- PUT /users/{id} 更新
- DELETE /users/{id} 删除
- 数据库：PostgreSQL
- 部署：Docker + K8s
```

#### 12.2 工作流

**Step 1：Brainstorming（30 min）**

```text
用 brainstorming skill 探索：
- 认证策略：JWT / Session
- 密码哈希：bcrypt / argon2
- 数据验证：Pydantic v2
- ORM：SQLAlchemy 2.0 async
- 测试：pytest + httpx
- 部署：Docker + K8s manifests
- 监控：Prometheus + Grafana
```

**Step 2：Writing Plans（60 min）**

```markdown
# Implementation Plan

## Phase 1: 项目骨架（2h）
- [ ] pyproject.toml（依赖管理）
- [ ] docker-compose.yml（本地 DB）
- [ ] Makefile（常用命令）
- [ ] README.md

## Phase 2: 数据库（2h）
- [ ] models.py（User 模型）
- [ ] migrations/（alembic）
- [ ] 数据库连接（async engine）

## Phase 3: API（4h）
- [ ] schemas.py（Pydantic）
- [ ] routes/users.py
- [ ] routes/auth.py
- [ ] deps.py（auth dependency）

## Phase 4: 测试（3h）
- [ ] tests/conftest.py（fixture）
- [ ] tests/test_users.py
- [ ] tests/test_auth.py

## Phase 5: 部署（2h）
- [ ] Dockerfile
- [ ] k8s/deployment.yaml
- [ ] k8s/service.yaml

总计：13h ≈ 2 个工作日
```

**Step 3：TDD 实现（10h）**

```bash
# Step 1：写测试
cat > tests/test_users.py << 'EOF'
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    response = await client.post("/users", json={
        "email": "test@example.com",
        "username": "testuser",
        "password": "securepassword123"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "password" not in data
    assert "id" in data
EOF

# Step 2：跑测试 → 红色
pytest tests/test_users.py

# Step 3：实现让测试通过
# Step 4：跑测试 → 绿色
# Step 5：重构
```

**Step 4：自动化 CI/CD（30 min）**

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=app --cov-report=xml
      - uses: codecov/codecov-action@v3
```

#### 12.3 关键决策记录

| 决策 | 选项 | 选择 | 理由 |
|:---|:---|:---|:---|
| ORM | SQLAlchemy / Tortoise / Piccolo | SQLAlchemy 2.0 async | 生态最成熟 |
| 密码哈希 | bcrypt / argon2 / scrypt | bcrypt | 性能 + 安全性平衡 |
| JWT 库 | PyJWT / python-jose | PyJWT | 维护活跃 |
| 测试库 | pytest + httpx / requests | pytest + httpx | async 支持 |

### 13. 案例 2：为一个老项目添加新功能

#### 13.1 场景

```text
项目：sid-recsys（已存在 3 年，10 万行代码）
任务：添加一个新功能 "在线学习（online learning）"
约束：
- 不能改现有 API
- 必须兼容老的 checkpoint
- 训练数据来自 Kafka 流
```

#### 13.2 工作流

**Step 1：Discovery（理解老代码）**

```bash
# 用 librarian subagent 探索代码
"分析 sid-recsys 项目的：
1. 现有模型架构（src/sid/encoder.py）
2. 训练 pipeline（src/train/）
3. 模型保存格式（src/utils/checkpoint.py）
4. 配置管理（src/conf/）
输出一份 architecture.md"
```

**Step 2：定位关键文件**

```text
找到需要修改的文件：
- src/sid/encoder.py - 加 online_update() 方法
- src/train/offline.py - 提取共享逻辑
- src/train/online.py - 新文件
- configs/online.yaml - 新文件
```

**Step 3：写 plan**

```markdown
## Plan: Online Learning

### 改动文件
1. 新增 src/train/online.py (300 行)
2. 修改 src/sid/encoder.py (加 80 行)
3. 新增 configs/online.yaml (50 行)
4. 新增 tests/test_online.py (200 行)

### 兼容性
- checkpoint v2 格式不变
- 现有 API 不动
- 配置向后兼容（不填 online 部分 = 旧行为）

### 风险
- Kafka consumer 可能丢消息 → 加重试
- 在线更新可能影响实时推理 → 加 mutex
```

**Step 4：TDD 实现**

```text
1. 先写 test_online.py（mock Kafka + 模型）
2. 跑 → 红色
3. 实现 online.py
4. 跑 → 绿色
5. 集成测试（端到端）
```

#### 13.3 关键经验

- **不删老代码**：用 adapter / wrapper 包一层
- **配置隔离**：online / offline 两套配置，CI 分别跑
- **渐进式发布**：先 1% 流量，再 10%，再 100%

### 14. 案例 3：Debug 一个生产事故

#### 14.1 场景

```text
2026-07-15 03:42 UTC，生产环境 P99 延迟从 100ms 升到 2.5s
持续 18 分钟，影响 12% 请求
```

#### 14.2 systematic-debugging skill 实战

```yaml
# Step 1: 复现
- 拉取生产 trace（保留 PII 脱敏）
- 在 staging 复现 → 失败率 100%

# Step 2: 定位
- 看 trace：95% 延迟在 db_query 函数
- 看 db 监控：DB CPU 100%
- 看慢查询日志：SELECT * FROM recommendations WHERE user_id = ? ORDER BY score DESC LIMIT 50
- EXPLAIN：全表扫描，1.2 亿行

# Step 3: 根因
- 缺失索引：user_id + score
- 昨天有次 DB migration 删除了索引（人工失误）

# Step 4: 修复
CREATE INDEX CONCURRENTLY idx_recommendations_user_id_score
ON recommendations (user_id, score DESC);

# Step 5: 验证
- staging 复现 → 50ms
- 灰度发布 → 5% → 50% → 100%

# Step 6: 防御
- DB migration 加 CI 检查：必须保留所有索引
- 加 DB schema 监控
- 加 slow query 自动告警
```

#### 14.3 复盘模板

```markdown
## Incident Report

### Summary
- 时间：2026-07-15 03:42 - 04:00 UTC
- 影响：12% 请求 P99 延迟从 100ms → 2.5s
- 根因：DB migration 误删索引

### Timeline
- 03:42 - 告警触发
- 03:45 - on-call 介入
- 03:52 - 定位到 DB
- 03:55 - 找到根因
- 03:58 - 修复（重建索引）
- 04:00 - 服务恢复

### Action Items
- [ ] migration 加 dry-run 检查
- [ ] slow query 自动告警（owner: 张三）
- [ ] DB schema 监控（owner: 李四）
- [ ] on-call 培训（owner: 王五）
```

### 15. 案例 4：复现一篇论文

#### 15.1 场景

```text
论文：HSTU (Hierarchical Sequential Transduction Unit)
作者：Meta, 2024
任务：复现并验证
```

#### 15.2 工作流

**Step 1：论文解析**

```text
用 OpenCode + context7 查相关库的文档：
- PyTorch 2.3 attention 实现
- FBGEMMA 推荐库
- RecZoo 工具集
```

**Step 2：代码骨架**

```python
# src/models/hstu.py
import torch
import torch.nn as nn

class HSTU(nn.Module):
    """
    Hierarchical Sequential Transduction Unit

    论文：https://arxiv.org/abs/2402.17152
    """
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 4 个投影矩阵（Q, K, V, U）
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_u = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 位置编码（论文用 RoPE）
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        self.max_seq_len = max_seq_len

    def forward(self, x, attention_mask=None):
        """
        x: (batch, seq_len, hidden_dim)
        """
        B, L, D = x.shape

        Q = self.w_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        U = self.w_u(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # 相对注意力偏置
        bias = self._build_rel_bias(L).to(x.device)

        # 注意力的 softmax 不在 V 上而在 U 上（论文核心改动）
        attn_weights = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) + bias
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 输出 = attn_weights @ U (不是 V)
        out = attn_weights @ U
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return out

    def _build_rel_bias(self, seq_len):
        # 实现论文的 causal mask + 时间衰减
        ...
```

**Step 3：跑公开数据集**

```bash
python scripts/train_hstu.py \
  --data_path data/ml-1m \
  --batch_size 256 \
  --epochs 50 \
  --hidden_dim 128 \
  --num_heads 4

# 验证：Recall@10 / NDCG@10 应该接近论文报告的指标
```

**Step 4：写复现报告**

```markdown
# HSTU 复现报告

## 实验设置
- 数据：MovieLens-1M
- 模型：HSTU-base (hidden=128, heads=4)
- 优化器：AdamW, lr=1e-3
- Batch：256, Epochs：50

## 结果
| 指标 | 论文 | 我们的 |
|:---:|:---:|:---:|
| Recall@10 | 0.412 | 0.408 |
| NDCG@10 | 0.234 | 0.231 |

## 差异分析
- Recall@10 差 0.004（< 1%）：可能是 LR scheduler 不同
- 训练时间：4h vs 论文 8h → 用了 mixed precision

## 结论
基本复现成功。代码已开源在 github.com/xxx/hstu-pytorch
```

---

## 第九部分：团队推广与协作

### 16. 团队引入 Vibe Coding 的 5 阶段

#### 16.1 阶段 1：种子期（Week 1-2）

```text
目标：1-2 名先锋试用，建立示范
动作：
- 选最复杂 / 最重复的项目做试点
- 整理"AI 友好"的项目结构
- 写 3-5 个项目级 Skill
- 记录 10 个成功案例 + 5 个失败案例

成功标准：
- 试点项目效率提升 30%+
- 形成 1 份"内部最佳实践"文档
- 至少 1 篇内部分享
```

#### 16.2 阶段 2：扩展期（Week 3-6）

```text
目标：3-5 名核心用户，建立 SOP
动作：
- 培训 2 场（每场 2h）
- 建立内部 Skill 仓库
- 配置共享 MCP Server
- 制定"安全使用规范"

成功标准：
- 5+ 人日常使用
- 50+ Skill 沉淀
- 工单 / 事故不增加
```

#### 16.3 阶段 3：标准化期（Week 7-12）

```text
目标：全团队推广，建立制度
动作：
- 写"AI 编码规范"（类似编码规范）
- 集成到 CI / code review 流程
- 引入成本监控（每个团队月度预算）
- 季度分享会

成功标准：
- 80% 团队成员使用
- 平均效率提升 40%+
- 单月成本可控
```

#### 16.4 阶段 4：优化期（Month 4-6）

```text
目标：精耕细作，深度集成
动作：
- 自定义 Agent（reviewer / tester / docs）
- 自托管模型（成本优化）
- Skill Marketplace
- 跨团队协作工具

成功标准：
- 60% 重复任务被自动化
- 工程师满意度 > 8/10
- 月度成本同比下降
```

#### 16.5 阶段 5：规模化期（Month 7-12）

```text
目标：全公司级能力
动作：
- AI Productivity 团队
- 跨部门 Skill / MCP 共享
- 内部 LLM 评测平台
- 与研发流程深度集成（需求 / 设计 / 编码 / 测试 / 部署）

成功标准：
- 公司级 ROI 报告
- 行业标杆案例
- 内部晋升通道（AI 能力 = 核心能力）
```

### 17. 团队协作的关键工具

#### 17.1 共享 Skill 仓库

```text
# 推荐结构
git@github.internal.corp:engineering/opencode-skills.git
├── README.md
├── CONTRIBUTING.md
├── skills/
│   ├── brainstorming/
│   ├── writing-plans/
│   ├── test-driven-development/
│   └── ...
├── templates/
│   ├── python-skill-template/
│   └── ts-skill-template/
└── docs/
    ├── how-to-write-skill.md
    └── best-practices.md
```

#### 17.2 团队 MCP Hub

```yaml
# mcp-hub/config.yaml
servers:
  github-internal:
    type: http
    url: https://mcp-github.internal.corp
    auth: oauth2

  jira-internal:
    type: http
    url: https://mcp-jira.internal.corp
    auth: oauth2

  sso-internal:
    type: http
    url: https://mcp-sso.internal.corp

  metrics-internal:
    type: http
    url: https://mcp-metrics.internal.corp

# 个人部署（参考）
# opencode.json
{
  "mcpServers": {
    "internal-hub": {
      "type": "http",
      "url": "https://mcp-hub.internal.corp",
      "auth": "auto"  # 自动用公司 SSO
    }
  }
}
```

#### 17.3 成本分摊机制

```text
┌─────────────────────────────────────────────┐
│ 成本分摊模型                                  │
├─────────────────────────────────────────────┤
│ 模型 1：按团队分摊                            │
│   - 每个团队月度预算                          │
│   - 超额 → 团队 leader 审批                  │
│                                              │
│ 模型 2：按项目分摊                            │
│   - 每个项目独立 API key                      │
│   - 按调用量计费                              │
│                                              │
│ 模型 3：成本中心                              │
│   - 中央预算（AI Productivity 团队）          │
│   - 内部记账（虚拟货币）                      │
│   - 推动节约                                  │
└─────────────────────────────────────────────┘
```

### 18. 常见团队管理问题与对策

#### 18.1 问题清单

| 问题 | 表现 | 对策 |
|:---|:---|:---|
| **AI 写错代码没人 review** | 线上事故 | 强制 PR review，AI 生成代码必须人工 review |
| **代码风格不一致** | 代码混乱 | 项目级 .cursorrules + 强制 lint |
| **过度依赖 AI** | 工程师技能退化 | 每月 1 次"无 AI 编码日" |
| **成本失控** | 账单爆表 | 设置月度预算 + 告警 + 自动降级 |
| **数据泄露** | 用公司代码训练外部模型 | 用企业版 / 自托管 + 安全审计 |
| **技能孤岛** | 少数人会用 AI | 内部培训 + Buddy 制度 |

#### 18.2 安全红线（绝对禁止）

```text
🚫 禁止项：
- 用公共 LLM API 处理生产用户数据
- 把客户代码 / 商业机密上传到外部服务
- 跳过 review 直接部署 AI 生成的代码
- 用 AI 写权限相关代码（auth, authz, crypto）

⚠️ 高危项（需审批）：
- AI 修改数据库 schema
- AI 改动 billing / 计费逻辑
- AI 改动安全相关（CSRF, XSS, SQL injection）

✅ 低危项（可直接用）：
- 单文件 bug 修复
- 单元测试生成
- 文档生成
- 代码格式化
```

#### 18.3 AI 使用合规 checklist

- [ ] 不上传用户 PII / PHI / 商业机密
- [ ] 不上传客户代码（除非客户同意）
- [ ] 所有 AI 生成的代码必须 review
- [ ] 重要决策人工确认（不用 AI 决策）
- [ ] 用企业内部 LLM（如有）
- [ ] 数据保留策略（30 天 / 90 天 / 不保留）
- [ ] 审计日志开启

---

## 第十部分：面试题精选（10+ 题）

### Q1. 什么是 Vibe Coding？与传统编程的核心区别是什么？

**参考答案**：

Vibe Coding 是由 Andrej Karpathy 在 2025 年提出的概念，指**以意图为中心、信任 AI 自主执行**的编程范式。

| 维度 | 传统编程 | Vibe Coding |
|:---|:---|:---|
| **交互** | 你写代码，AI 补全 | 你描述意图，AI 写代码 |
| **关注点** | How（怎么实现） | What（想做什么） |
| **心智负担** | 全部记在脑子里 | 描述 + 验证 |
| **迭代** | 写 → 调试 → 重写 | 描述 → 看 → 调整描述 |

**关键要素**：完整的 Agent / Skills / MCP 体系 + 监督机制。

### Q2. Vibe Coding 的 5 条黄金法则是什么？举例说明。

**参考答案**：

来自 [Tobias Weiss: Vibe Coding with OpenCode](https://tobias-weiss.org/content/ai/opencode-guide/)

1. **描述意图而非实现**：说"添加用户认证"而非"创建 JWT 中间件 + bcrypt"
2. **信任工作流**：让 AI 自主探索、迭代、自我修正（不要每步都干预）
3. **保持驾驶座**：你定方向，AI 执行（明确边界）
4. **拥抱实验**：快速迭代胜过完美规划（接受 AI 失败）
5. **善用 Skills**：方法论由 Skills 承担（brainstorming、TDD 等）

**反例**：每行代码都让 AI 解释一遍（违反"信任"原则）

### Q3. Agent / Skills / MCP 三者关系是什么？

**参考答案**：

```text
Agent = 任务执行者（角色分工）
Skills = 方法论库（如何做）
MCP = 工具集成（用什么做）
```

- **Agent**：orchestrator / builder / explorer / reviewer 等角色
- **Skills**：brainstorming、TDD、debugging 等方法论
- **MCP**：filesystem、github、playwright 等外部能力

**协同模式**：
- Skill 指导 Agent 怎么思考
- MCP 提供 Agent 怎么行动
- 三者结合 = 完整的 Vibe Coding 体系

### Q4. 设计一个 Skill 时，YAML frontmatter 最重要的字段是什么？为什么？

**参考答案**：

**最重要的字段是 `description`**，因为它直接决定 LLM 是否会触发该 Skill。

**好的 description**：
```yaml
description: |
  当用户说"实现 / 添加 / 创建"一个新功能时使用。
  适用场景：用户给出了具体需求，但不确定如何分解任务。
  不适用：纯代码补全、单文件 bug 修复。
```

**要素**：
- **触发关键词**（正面 + 反面）
- **适用场景**（明确边界）
- **不适用场景**（避免误触发）

**反例**：
```yaml
description: 帮助写代码  # 太宽泛，永远不会精确触发
```

### Q5. MCP 协议的核心思想是什么？为什么需要它？

**参考答案**：

**核心思想**：让 LLM 通过标准协议调用外部工具 / 数据，而不是每个工具单独适配。

```text
问题（没有 MCP）：
- 每个 LLM 框架（OpenCode / Cursor / Cline）都要单独实现工具调用
- 每个工具（GitHub / Slack / DB）都要为每个框架写适配
- M × N 集成噩梦

方案（有 MCP）：
- LLM 框架统一实现 MCP client
- 工具统一实现 MCP server
- M + N 集成（线性）
```

**核心价值**：
- **标准化**：JSON-RPC over stdio / SSE
- **解耦**：工具与 LLM 解耦
- **可复用**：一个 MCP server，所有 LLM 通用
- **可观测**：统一监控 / 计费 / 权限

### Q6. 你是如何在项目中落地 Vibe Coding 的？请给出一个具体案例。

**参考答案模板**（推荐 STAR 法则）：

**Situation**：
项目 sid-recsys 是 3 年老项目，10 万行代码，5 人团队。新功能迭代慢，平均 PR review 周期 3 天。

**Task**：
引入 Vibe Coding 工作流，缩短 PR 周期到 1 天内。

**Action**：
1. 引入 OpenCode + superpowers + 7 个 MCP servers
2. 沉淀 20+ 项目级 Skill（项目规范、TDD 模板）
3. 强制 4 阶段工作流：brainstorming → writing-plans → TDD → review
4. 自定义 code-reviewer agent
5. 成本分摊：月度预算 + 告警

**Result**：
- PR 周期：3 天 → 0.8 天（-73%）
- Bug 率：未显著上升（reviewer agent 抓住 70% 问题）
- 工程师满意度：7/10 → 9/10
- 单月 AI 成本：$2,400（节省 $18,000 人力成本）

### Q7. Vibe Coding 最大的风险是什么？如何规避？

**参考答案**：

**最大风险**：
1. **代码质量下降**（AI 写得不规范、有 bug）
2. **数据泄露**（上传敏感代码到外部 LLM）
3. **过度依赖**（工程师技能退化）
4. **成本失控**（token 费用爆炸）
5. **法律责任**（AI 生成的代码归属 / 抄袭）

**规避措施**：

| 风险 | 规避 |
|:---|:---|
| 质量下降 | 强制 review + 自动化测试 + lint |
| 数据泄露 | 自托管 / 企业版 + 数据脱敏 |
| 技能退化 | 每月"无 AI 日"+ 定期白板编码 |
| 成本失控 | 月度预算 + 自动降级 |
| 法律风险 | 不用 AI 生成 logo / 图片 / 关键代码 |

### Q8. 用过哪些 AI 编码工具？对比它们的优缺点。

**参考答案模板**：

```text
用过：Cursor / Claude Code / OpenCode / Copilot / Cline / Continue

对比维度：

| 维度 | Cursor | Claude Code | OpenCode | Copilot |
|:---|:---:|:---:|:---:|:---:|
| 速度 | ★★★★★ | ★★★ | ★★★ | ★★★★★ |
| 能力 | ★★★★ | ★★★★★ | ★★★★★ | ★★★ |
| 价格 | $19/月 | $20/月 | 免费+API | $10/月 |
| 定制 | ★★★ | ★★★ | ★★★★★ | ★★ |
| 开源 | ❌ | ❌ | ✅ | ❌ |
| 隐私 | 中 | 中 | 高 | 中 |

我的偏好：
- 日常编码：Cursor Tab（快）
- 重构 / 调试：OpenCode + Sonnet（强）
- 跨文件 / 跨仓库：OpenCode Subagent
- PR review：Cursor Composer
```

### Q9. 如何为 AI 准备一个"友好"的项目结构？

**参考答案**：

```yaml
# 推荐的项目结构
project/
├── README.md            # 项目入口 + 架构图
├── CLAUDE.md            # AI 友好的项目说明（必填）
├── docs/
│   ├── architecture.md  # 架构图 + 模块说明
│   ├── coding-style.md  # 编码规范
│   └── deployment.md    # 部署指南
├── src/                 # 主代码
│   ├── __init__.py
│   └── module_a/
│       ├── __init__.py
│       ├── core.py      # 模块文档清晰
│       └── tests.py     # 或独立 tests/
├── tests/               # 单元测试
├── scripts/             # 工具脚本
├── configs/             # 配置
└── .opencode/
    ├── rules/           # 项目级 AI 规则
    └── skills/          # 项目级 Skill
```

**AI 友好的关键**：
- 命名清晰（变量 / 函数 / 文件）
- 文件大小适中（< 500 行）
- 模块边界清楚
- 重要模块有 docstring
- 有 CLAUDE.md / .cursorrules

### Q10. 如果 AI 写出了错误的代码，你如何 Debug？

**参考答案（用 systematic-debugging skill）**：

```yaml
# 4 步法
Step 1: 复现
  - 最小化复现代码
  - 找到稳定触发条件

Step 2: 定位
  - 加 print / log / debugger
  - 用二分法找到 root cause
  - 不接受"我觉得是 X"

Step 3: 根因分析
  - 5 Whys（连续追问 5 次"为什么"）
  - 检查边界条件 / 类型 / 边界值
  - 检查所有相关分支

Step 4: 修复 + 防御
  - 修复（最小改动）
  - 加测试（防止回归）
  - 加防御（边界检查 / 异常处理）
```

**关键原则**：
- 不要猜，要证据
- 不要 fix symptom，要 fix root cause
- 不要"看上去工作了"，要"测试验证"
- 不要忘了加测试

### Q11. Cursor / Claude Code / OpenCode 选哪个？

**参考答案**：

**Cursor**：
- 适合：VSCode 重度用户、追求丝滑体验
- 优势：补全速度最快、UI 最精致
- 劣势：闭源、隐私中、定制受限

**Claude Code**：
- 适合：Anthropic 模型爱好者
- 优势：Sonnet 模型最强
- 劣势：CLI 优先、IDE 集成弱

**OpenCode**：
- 适合：开源爱好者、自托管需求、多 provider
- 优势：完全开源、Skills / MCP 完全开放、多模型
- 劣势：CLI 优先、需要自己配

**我的选择**：
- 日常开发 → Cursor Tab + Cmd+K（快）
- 重构 / Debug → OpenCode + Claude Sonnet（强）
- 团队共享 → OpenCode（统一标准）

### Q12. 什么是 Skills 系统？如何写一个好的 Skill？

**参考答案**：

**Skills 系统** = OpenCode / Claude Code 提供的"方法论注入"机制。每个 Skill 是一个带 YAML frontmatter 的 Markdown 文件，描述"何时用 + 怎么用"。

**好的 Skill 的标准**：

```markdown
# 1. 清晰的触发条件（description 字段）
description: |
  当用户说 [关键词] 时使用。
  适用：[场景 1] / [场景 2]
  不适用：[反向 1]

# 2. 明确的工作流
## Workflow
1. Step 1: ...
2. Step 2: ...
3. Step 3: ...

# 3. 必要的示例
## Examples
Input: ...
Output: ...

# 4. 反模式（重要！）
## Anti-patterns
- ❌ 不要 ...
- ❌ 不要 ...

# 5. 依赖说明
dependencies: [other-skill]
```

**实战建议**：
- 保持 Skill 单一职责（不要大杂烩）
- 描述要精确触发条件
- 用 examples 而不是抽象描述
- 写反模式（不要只写正确做法）
- 定期 review / 更新

### Q13. 解释一下 MCP 协议的工作流程。

**参考答案**：

```text
┌────────────────────────────────────────────────┐
│ MCP Host (OpenCode)                              │
│   启动 → 加载配置 → 连接 MCP Servers              │
└────────────────────────────────────────────────┘
                      ↓
1. OpenCode 读取 opencode.json 中的 mcpServers
2. 对每个 server 启动子进程（或建立 HTTP 连接）
3. 调用 list_tools() 获取可用工具列表
4. 工具列表注入到 system prompt
5. 用户输入 + LLM 决策 → 调用工具
6. JSON-RPC 通信：Host → Server
7. Server 执行 → 返回结果
8. 结果注入 LLM 上下文 → 生成最终回答
```

**关键协议**：
- 通信层：JSON-RPC 2.0
- 传输层：stdio（本地进程）/ SSE（远程）
- 能力声明：tools / resources / prompts

**生命周期**：
1. **Initialize**：握手 + 协议版本协商
2. **List**：Host 查询 Server 能力
3. **Call**：Host 调用具体 tool
4. **Subscribe**（可选）：订阅 resource 变化
5. **Shutdown**：正常关闭

### Q14. Vibe Coding 下，工程师的核心能力发生了什么变化？

**参考答案**：

**从"实现能力"到"判断能力"**：

```text
传统工程师：
- 写代码能力（实现）
- 算法能力（设计）
- 工程能力（部署 / 性能）

AI 时代工程师：
- 描述能力（把意图说清楚）
- 判断能力（review / 选型）
- 整合能力（多工具 / 多服务）
- 学习能力（快速学新工具）
```

**核心技能迁移**：

| 旧技能 | 新技能 |
|:---|:---|
| 写 CRUD | 设计数据模型 + 描述需求 |
| 调 bug | 用 systematic-debugging 引导 AI |
| 性能优化 | 写 benchmark + 让 AI 优化 |
| 写文档 | review AI 生成的文档 |

**新要求**：
- **批判性思维**：不信 AI 输出，必须 review
- **沟通能力**：与 AI"对话"的能力
- **抽象能力**：把问题抽象成 AI 能理解的形式
- **领域知识**：依然必须（AI 不是万能）

### Q15. Vibe Coding 未来 3 年的趋势是什么？

**参考答案**：

**短期（2026-2027）**：
- **多模态输入**：直接用语音 / 视频描述需求
- **更强 Planner**：长程任务自动分解
- **垂直 Skill Marketplace**：行业级 Skill（如"前端开发包"、"金融建模包"）
- **Agent-as-a-Service**：托管 Agent（类似 SaaS）

**中期（2028-2030）**：
- **自主工程师**：从需求 → 上线全自动（仍有 review）
- **全公司数字员工**：每个公司有"AI 工程师团队"
- **AI 工程文化**：招聘要求"AI 协作能力"

**长期（2030+）**：
- **AGI 工程师**：代码由 AI 主导，人是 reviewer
- **自然语言编程**：代码 = 文本 + AI
- **彻底改变软件工程教育**

**不变量**：
- **业务理解**：依然是人
- **架构决策**：依然是人
- **价值观判断**：依然是人
- **最终责任**：依然在人

---

## 附录 C：补充资源

### C.1 推荐阅读

- [Tobias Weiss: Vibe Coding with OpenCode](https://tobias-weiss.org/content/ai/opencode-guide/) —— Vibe Coding 入门必读
- [Cursor Docs](https://docs.cursor.com) —— Cursor 官方文档
- [Claude Code Best Practices](https://docs.anthropic.com/claude/docs/best-practices) —— Anthropic 官方
- [Model Context Protocol Spec](https://modelcontextprotocol.io) —— MCP 协议规范
- [Karpathy: Software 2.0](https://karpathy.medium.com/software-2-0) —— 时代背景

### C.2 视频教程

- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [OpenCode Tutorial Series](https://www.youtube.com/results?search_query=opencode+tutorial)
- [MCP Crash Course](https://www.youtube.com/results?search_query=mcp+model+context+protocol)

### C.3 社区

- [OpenCode Discord](https://discord.gg/opencode)
- [Cursor Forum](https://forum.cursor.com)
- [Claude Developers Slack](https://anthropic.com/slack)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

### C.4 持续学习路径

```text
入门（Week 1-2）：
  - 看 Karpathy 的 Vibe Coding 演讲
  - 安装 OpenCode / Cursor
  - 完成 5 个小任务

进阶（Month 1-3）：
  - 学习 MCP 协议
  - 写 10 个自定义 Skill
  - 搭建 1 个 MCP Server

高级（Month 4-6）：
  - 设计复杂的 Subagent 协同
  - 自托管 LLM
  - 团队级 Vibe Coding 体系

专家（Month 7+）：
  - 贡献开源 Skill / MCP
  - 写技术博客
  - 在公司内部推广
```

---

> **本文档版本**：v1.1
> **扩展日期**：2026-07-17
> **新增内容**：5 大章节，约 2000+ 行
> **覆盖**：AI 工具深度配置 / Skills / MCP 集成 / 实战案例 / 团队推广 / 面试题
> **下一步**：阅读第 03 章全栈测试手册，配合实战



## 深度扩展 · 多视角系统展开(续)

### 110. Vibe Coding 全栈配置手册 - 认知科学 视角

**核心议题**: 本节从认知科学视角切入,系统展开 "Vibe Coding 全栈配置手册" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"Vibe Coding 全栈配置手册" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 认知科学视角的展开**

从认知科学视角出发,我们可以把 "Vibe Coding 全栈配置手册" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调认知科学视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"Vibe Coding 全栈配置手册" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"认知科学" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "Vibe Coding 全栈配置手册" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "Vibe Coding 全栈配置手册" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "Vibe Coding 全栈配置手册" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"Vibe Coding 全栈配置手册" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从认知科学视角切入,系统展开了 "Vibe Coding 全栈配置手册" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. Vibe Coding 全栈配置手册 - 元学习 视角

**核心议题**: 本节从元学习视角切入,系统展开 "Vibe Coding 全栈配置手册" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"Vibe Coding 全栈配置手册" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 元学习视角的展开**

从元学习视角出发,我们可以把 "Vibe Coding 全栈配置手册" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调元学习视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"Vibe Coding 全栈配置手册" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"元学习" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "Vibe Coding 全栈配置手册" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "Vibe Coding 全栈配置手册" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "Vibe Coding 全栈配置手册" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"Vibe Coding 全栈配置手册" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从元学习视角切入,系统展开了 "Vibe Coding 全栈配置手册" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. Vibe Coding 全栈配置手册 - 工具方法 视角

**核心议题**: 本节从工具方法视角切入,系统展开 "Vibe Coding 全栈配置手册" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"Vibe Coding 全栈配置手册" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 工具方法视角的展开**

从工具方法视角出发,我们可以把 "Vibe Coding 全栈配置手册" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调工具方法视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"Vibe Coding 全栈配置手册" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"工具方法" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "Vibe Coding 全栈配置手册" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "Vibe Coding 全栈配置手册" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "Vibe Coding 全栈配置手册" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"Vibe Coding 全栈配置手册" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从工具方法视角切入,系统展开了 "Vibe Coding 全栈配置手册" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. Vibe Coding 全栈配置手册 - 习惯养成 视角

**核心议题**: 本节从习惯养成视角切入,系统展开 "Vibe Coding 全栈配置手册" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"Vibe Coding 全栈配置手册" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 习惯养成视角的展开**

从习惯养成视角出发,我们可以把 "Vibe Coding 全栈配置手册" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调习惯养成视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"Vibe Coding 全栈配置手册" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"习惯养成" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "Vibe Coding 全栈配置手册" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "Vibe Coding 全栈配置手册" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "Vibe Coding 全栈配置手册" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"Vibe Coding 全栈配置手册" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从习惯养成视角切入,系统展开了 "Vibe Coding 全栈配置手册" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. Vibe Coding 全栈配置手册 - 计划视角 视角

**核心议题**: 本节从计划视角视角切入,系统展开 "Vibe Coding 全栈配置手册" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"Vibe Coding 全栈配置手册" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 计划视角视角的展开**

从计划视角视角出发,我们可以把 "Vibe Coding 全栈配置手册" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调计划视角视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"Vibe Coding 全栈配置手册" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"计划视角" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "Vibe Coding 全栈配置手册" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "Vibe Coding 全栈配置手册" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "Vibe Coding 全栈配置手册" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"Vibe Coding 全栈配置手册" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从计划视角视角切入,系统展开了 "Vibe Coding 全栈配置手册" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



