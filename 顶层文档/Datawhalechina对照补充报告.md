---
title: Datawhalechina 对照补充报告
tags:
  - 顶层文档
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina 全部相关开源教程，给出与本学习库（除科目四外）各分类章节的映射、补充方案与薄弱点分析
last_updated: 2026-09-20
status: ✅ v1.0
---

# Datawhalechina 对照补充报告

> **本报告目的**：对照 [datawhalechina](https://github.com/datawhalechina) 组织下与本学习库重叠的开源教程，逐分类给出「我们的章节 ↔ datawhale 仓库/章节」的映射关系、datawhale 能补充什么、以及双向 Gap 分析。
>
> **覆盖范围**：除「科目四」外的全部 md 教程（共 579 个文件，按 6 大分类整理）。
>
> **配套产出**：各分类下放置了以 `-Datawhale补充.md` 结尾的伴生补充文件（实质内容补充），本报告是它们的总索引。

---

## 目录

- [一、补充来源总览](#一补充来源总览)
- [二、基础分类映射](#二基础分类映射)
- [三、算法与AI 分类映射](#三算法与ai-分类映射)
- [四、搜推广与生成式推荐分类映射](#四搜推广与生成式推荐分类映射)
- [五、工程与职业分类映射](#五工程与职业分类映射)
- [六、面试分类映射](#六面试分类映射)
- [七、顶层文档映射](#七顶层文档映射)
- [八、高价值资源排行](#八高价值资源排行)
- [九、薄弱章节与补充方案 Top 10](#九薄弱章节与补充方案-top-10)
- [十、双向 Gap 分析](#十双向-gap-分析)
- [十一、伴生补充文件索引](#十一伴生补充文件索引)

---

## 一、补充来源总览

本次对照涉及 datawhalechina 的 25+ 个仓库，按补充价值分档：

| 档位 | 仓库 | Star | 与本库的关系 |
| :--- | :--- | ---: | :--- |
| ⭐⭐⭐ | [fun-rec](https://github.com/datawhalechina/fun-rec)（深度推荐算法实践·小麦书） | 7.3k | **上篇级联架构 + 下篇生成式范式**，与「经典推荐/GR研究/SID技术」几乎逐章对应 |
| ⭐⭐⭐ | [hello-agents](https://github.com/datawhalechina/hello-agents)（从零开始构建智能体） | 79k | 16 章 + 13 篇 Extra，与「Agent技术」全部 8 章对应，补通信协议/Agentic-RL/评估/面试题 |
| ⭐⭐⭐ | [happy-llm](https://github.com/datawhalechina/happy-llm)（从零开始构建大模型） | 34k | 手搓 LLaMA2、训练全流程、GRPO/Agentic-RL，与「LLM发展」强对应 |
| ⭐⭐⭐ | [every-embodied](https://github.com/datawhalechina/every-embodied)（具身智能） | 3.7k | 19+ 篇 VLA 模型导读、9 篇具身世界模型、LIBERO/SimplerEnv 基准，与「具身智能」强对应 |
| ⭐⭐⭐ | [leetcode-notes](https://github.com/datawhalechina/leetcode-notes) | 1.2k | 与「面试」第 1~10 章几乎 1:1 对应的官方图解题库 |
| ⭐⭐ | [math-for-ai](https://github.com/datawhalechina/math-for-ai)（人工智能的数学基础，MML 中译） | 393 | 12 章严格数学推导，补「数学」全 8 章的证明与习题 |
| ⭐⭐ | [all-in-rag](https://github.com/datawhalechina/all-in-rag)（RAG 全栈指南） | 11k | 10 章 + GraphRAG 实战，补「Agent技术/第02章 RAG」 |
| ⭐⭐ | [pumpkin-book](https://github.com/datawhalechina/pumpkin-book)（南瓜书） | 26k | 西瓜书公式逐行推导，补「数学」概率/优化/信息论推导细节 |
| ⭐⭐ | [daily-interview](https://github.com/datawhalechina/daily-interview)（Datawhale 面经） | 3.8k | ML/CV/NLP/推荐/开发面经 + LLM 面试题库（48KB），补「工程与职业/面试笔记」 |
| ⭐⭐ | [so-large-lm](https://github.com/datawhalechina/so-large-lm)（大模型基础） | 7.7k | CS324 风格 14 章，补 LLM 训练/分布式/PEFT/Llama 家族 |
| ⭐⭐ | [thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)（深入浅出 PyTorch） | 3.8k | 张量底层/自动求导/训练技巧/10+ 经典模型源码，补「编程/工程能力/PTH教程」 |
| ⭐⭐ | [key-book](https://github.com/datawhalechina/key-book)（机器学习理论导引） | 1.7k | VC 维/泛化界/收敛率/遗憾界，补「数学」理论证明 |
| ⭐⭐ | [torch-rechub](https://github.com/datawhalechina/torch-rechub) | 1.2k | 13+ 排序模型（DeepFM/DCN-v2/DIN/DIEN/MMoE…）可运行实现 |
| ⭐ | [self-llm](https://github.com/datawhalechina/self-llm)、[llm-deploy](https://github.com/datawhalechina/llm-deploy)、[handy-ollama](https://github.com/datawhalechina/handy-ollama)、[zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) | — | 开源大模型微调/部署/推理引擎，补「工程与职业/0-K8s 与 ML 部署」「LLM发展/附录B」 |
| ⭐ | [easy-vibe](https://github.com/datawhalechina/easy-vibe)、[vibe-vibe](https://github.com/datawhalechina/vibe-vibe) | 19k/6k | Vibe Coding 三阶段（原型→全栈→Claude Code 高级技能），补「工具与配置」 |
| ⭐ | [self-harness](https://github.com/datawhalechina/self-harness) | 252 | Harness Engineering + miniMaster2.0 三层智能体代码，补「工程能力/第05章」「工具与配置」 |
| ⭐ | [learn-world-model](https://github.com/datawhalechina/learn-world-model) | 408 | 世界模型课程（架构深拆），补「具身智能/第08章」 |
| ⭐ | [huawei-od-python](https://github.com/datawhalechina/huawei-od-python) | 203 | 400+ 场景化算法题（choice100/questions200），补「面试/第09章」习题量 |
| ⭐ | [team-learning-program](https://github.com/datawhalechina/team-learning-program) | 850 | 数据结构与算法组队学习（树/图/线段树讲义），补「面试」高级章节理论地基 |
| ⭐ | [sweetalk-design-pattern](https://github.com/datawhalechina/sweetalk-design-pattern) | 142 | 23 种 GoF 模式 Python/C++/Java 实现，补「工程能力/第04章」 |
| ⭐ | [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) | 586 | LLM 算法系统级面试：Triton/CUDA/显存工程，补「工程能力/第02、03、07章」 |
| ⭐ | [learn-python-the-smart-way-v2](https://github.com/datawhalechina/learn-python-the-smart-way-v2) | 432 | slides/homework 全套交互材料，补「编程/第01章」 |
| ⭐ | [faster-git](https://github.com/datawhalechina/faster-git) | 183 | 10 讲 Git（内部原理/GitFlow/提交规范），补「工具手册/第01章」 |
| ⭐ | [dive-into-embodied-ai](https://github.com/datawhalechina/dive-into-embodied-ai)、[hello-robotics](https://github.com/datawhalechina/hello-robotics) | 458/70 | 仿真环境/控制器/运动规划链路，补「具身智能/第03、07章」 |
| ⭐ | [docker-notes](https://github.com/datawhalechina/docker-notes)、[easy-linux](https://github.com/datawhalechina/easy-linux)、[wonderful-sql](https://github.com/datawhalechina/wonderful-sql) | — | Docker/Linux/SQL 基础，补「工程与职业」工具链层 |
| ⭐ | [leegenai-tutorial](https://github.com/datawhalechina/leegenai-tutorial)（李宏毅生成式 AI） | 304 | Diffusion/CoT/RLHF 直观讲解，补「数学/第04章逻辑」「LLM发展」 |
| ⭐ | [leedl-tutorial](https://github.com/datawhalechina/leedl-tutorial)（李宏毅深度学习·苹果书） | 17k | 作业+课程笔记，补「深度学习手撕」全部 5 章 |
| ⭐ | [competition-baseline](https://github.com/datawhalechina/competition-baseline) | 4.8k | 数据挖掘/推荐竞赛 baseline，补「搜推广」实战 |
| ⭐ | [joyful-pandas](https://github.com/datawhalechina/joyful-pandas)、[hands-on-data-analysis](https://github.com/datawhalechina/hands-on-data-analysis) | — | pandas/数据分析，补「编程/第02章」「数学/第04章特征工程」 |
| ⭐ | [easy-rl](https://github.com/datawhalechina/easy-rl)（蘑菇书）、[joyrl-book](https://github.com/datawhalechina/joyrl-book) | 15k/313 | RL 经典教材，补「搜推广/RL」「具身智能/第06章」 |

---

## 二、基础分类映射

### 2.1 基础/工具与配置（3 个文件）

| 本地文件 | datawhale 仓库与路径 | 补充点 |
| :--- | :--- | :--- |
| 第01章-Git完全教程 | faster-git `docs/zh/lecture01~10` | Git 内部原理（对象模型/packfile）、GitFlow 工作流规范、提交规范（Conventional Commits）、团队协作与 diff 工具。本地教程已 3400+ 行，主要补「原理内幕 + 规范化」 |
| 第02章-Vibe-Coding全栈配置手册 | easy-vibe `docs/zh-cn/stage-1~3 + appendix`（80+ 交互条目）；vibe-vibe | Claude Code 核心技能（MCP/Skills/Agent Teams/长任务/Spec Coding/Agent SDK）、SaaS 端到端实战（Stripe 计费）、用户研究方法论（Double Diamond/JTBD/Mom Test） |
| 第03章-全栈测试Agent-Skills-MCP配置手册 | easy-vibe `stage-3/core-skills/*`；self-harness `docs/chapter1~6 + code/miniMaster2.0` | MCP/Skills/Agent Teams 全套指南；Harness Engineering（Prompt→Context→Harness 演进）与 Planner/Executor/Validator 三层智能体完整代码 |

### 2.2 基础/工程基础/工程能力（9 个文件）

| 本地文件 | datawhale 仓库与路径 | 补充点 |
| :--- | :--- | :--- |
| 第01章-模型架构设计方法论 | leedl-tutorial `Homework/`；thorough-pytorch `第十章`；so-large-lm `ch03/ch04` | ResNet/Swin/ViT/LSTM/YOLO 源码逐行解读；LLM 视角 Transformer/MoE 深度解析 |
| 第02章-时间与空间复杂度分析 | llm-algo-leetcode `02_PyTorch_Algorithms`；key-book `chapter3/chapter7` | 算子 FLOPs/内存访问复杂度量化习题；VC 维/Rademacher 复杂度理论视角 |
| 第03章-参数量与显存预估 | llm-algo-leetcode `01/03/04`；thorough-pytorch `第六章` | Triton/CUDA kernel 级显存控制、激活重计算、混合精度实战 |
| 第04章-代码抽象与设计模式 | sweetalk-design-pattern `src/design_patterns/{python,cpp,java}` | 23 种 GoF 模式三语言完整实现 + 《大话设计模式》案例 |
| 第05章-框架流程与工程架构设计 | self-harness `code/miniMaster2.0`（bootstrap/engine/memory/llm/tools）；easy-vibe `stage-3` | 三层 Agent 工程架构、动态工作记忆、ToolService、状态机控制；Superpowers TDD 工作流 |
| 第06章-超参数与配置设计 | thorough-pytorch `第六章` | LambdaLR/CosineAnnealing/Warmup 调度器代码、超参组织与 wandb/SwanLab 集成 |
| 第07章-性能剖析与优化 | thorough-pytorch `第七章`；llm-algo-leetcode `03_Triton_Kernels + 04_CUDA` | PyTorch profiler/TensorBoard 可视化定位瓶颈；Triton kernel 与 CUDA 系统级优化 benchmark |
| 第08章-评估指标大全 | thorough-pytorch `第零章`；so-large-lm `ch02/ch12` | 分类/回归/排序指标基础；LLM 评估体系（HELM/In-context 评估/碳排放） |
| macos深度学习小tips | llm-algo-leetcode `01_Hardware_Math_and_Systems` | Apple Silicon（MLX/MPS）训练环境配置与基准 |

### 2.3 基础/工程基础/编程（9 个文件）

| 本地文件 | datawhale 仓库与路径 | 补充点 |
| :--- | :--- | :--- |
| 第01章-Python与PyTorch基础 | learn-python-the-smart-way-v2 `docs/{Aims,Notes,Method,Question}`；thorough-pytorch `第零~二章` | 交互式 slides/homework；张量 stride/view/contiguous 底层机制 notebook |
| 第02章-搜推广场景Python工程底座 | thorough-pytorch `第三章` | 数据/模型/损失/优化器/评估/可视化六模块标准化模板 |
| 第03章-深度学习工程进阶 | thorough-pytorch `第五~六章` | 模型封装、timm/torchvision 微调、半精度/分布式训练 |
| 第04章-大模型训练全生命周期 | so-large-lm `ch06~ch08`；llm-algo-leetcode | 目标函数/PEFT(LoRA/QLoRA)/数据并行·模型并行·流水线并行全链路 |
| 第05章-训练可视化与监控 | thorough-pytorch `第七章` | TensorBoard/wandb/SwanLab 三套工具集成实战 |
| 第06章-搜推广能力重构 | so-large-lm `ch07/ch14` | PEFT 与 Llama 1-3 架构演进，可迁移到生成式重排 |
| 第07章-能力提升计划专项 | llm-algo-leetcode `00_Prerequisites + 02` | 从 PyTorch 基础到 CUDA/Rust 系统级编程的渐进路径 |
| 附录A-深度调研报告 | so-large-lm `ch09/ch10/ch14` | Llama 家族技术复盘、有害性分析 |
| 附录B-交流笔记 | easy-vibe `vibe-stories` | 4 个真实用户故事（乡村教师/大学生/IT 老师/货车司机） |

### 2.4 基础/数学（90 个文件，8 大章）

数学分类是 datawhale 补充密度最高的区域（约 62 条映射），三大支柱：

| datawhale 仓库 | 补充定位 | 覆盖的本地章节 |
| :--- | :--- | :--- |
| **math-for-ai**（MML 中译，12 章） | 严格数学推导 + 习题 | 第01章高数（ch5 向量微积分/ch7 连续优化）、第02章线代（ch2/ch4 分解/ch10 降维 PCA）、第03章概率（ch6/ch11 混合 Gauss）、第04章离散（ch1 导引）、第05章信息论（ch6/ch8）、第06章最优化（ch7） |
| **pumpkin-book**（南瓜书，16 章） | 西瓜书公式逐行推导 | 第01章-04 链式法则（ch6 SVM 对偶/KKT）、第01章-09 变分法（ch7 EM）、第03章-05/06 贝叶斯与 MLE/MAP（ch7）、第03章-09 采样（ch7/ch14 概率图）、第04章-04 图论（ch14）、第05章-04/05 KL 与交叉熵（ch7/ch3）、第06章-08 约束优化（ch6 KKT） |
| **key-book**（钥匙书，8 章+附录） | 可学性/VC 维/泛化界/收敛率/遗憾界 | 第01章-05 Taylor 展开与收敛率（ch7）、第01章-08 泛函/RKHS（ch3 PAC）、第02章复杂度（ch3）、第04章-10 P/NP 与学习理论（ch3）、第06章-05/06 凸与非凸收敛率（ch6/ch7） |

辅助补充：thorough-pytorch（第02章-07 张量 stride/view、第06章-02 优化器代码）、so-large-lm（第02章-09 LoRA 低秩、第03章-10 概率视角生成式 AI、第05章-07/08 信息论在 LLM、第06章-04/07 二阶近似与分布式）、leegenai-tutorial（第04章-02 CoT 逻辑、第04章-11 语言模型）、joyful-pandas（第04章-08 特征工程）、leedl-tutorial（第01章-07 Fourier/Swin 直觉）。

> 注：第07章系统思维（15 个工业案例）与第08章数值计算，datawhale 无一一对应仓库；第07章的 so-large-lm（RLHF/GRPO/Llama 案例）与 leegenai-tutorial（Diffusion 直观）可作散点补充。

### 2.5 基础/因果推断（6 个文件）

datawhale **没有专门的因果推断仓库**（见 [Gap 分析](#十双向-gap-分析)）。最接近的替代：

| 本地文件 | 最接近的 datawhale 资源 | 补充点 |
| :--- | :--- | :--- |
| 0~5 全部 6 章 | math-for-ai `ch6 概率与统计`；hands-on-data-analysis；3-AB测试（与「工程与职业/3-AB测试与因果推断」互为补充） | 假设检验/置信区间的数理基础；数据分析实战中的统计口径。Do-calculus/工具变量/Causal Forest 等 core 内容需依赖外部文献（Pearl/Imbens/Wager），datawhale 无覆盖 |

### 2.6 基础/学习方法、联邦学习、语言学习（9 个文件）

| 分类 | datawhale 资源 | 说明 |
| :--- | :--- | :--- |
| 学习方法 | easy-vibe `vibe-stories`（真实学习者故事）；ai-learning-roadmap；Agent-Learning-Hub（学习路线） | 仅风格/路线参考，无直接对应 |
| 联邦学习 | **无对应仓库**（datawhale 无 FL 教程） | Gap，需外部补充 |
| 语言学习 | **无对应仓库** | Gap |

---

## 三、算法与AI 分类映射

### 3.1 Agent技术（9 个文件）—— 主补充源：hello-agents（79k stars）

| 本地文件 | hello-agents 章节 | 补充点 |
| :--- | :--- | :--- |
| 第01章-Agent技术发展脉络 | 第一章 初识智能体 + 第二章 智能体发展史 | 从符号主义到 LLM 驱动的演进史、"软件工程类 vs AI 原生"两派划分 |
| 第02章-RAG检索增强生成 | **all-in-rag 全 10 章** | 数据分块策略、混合检索（稠密+稀疏）、查询构建/重构/Text2SQL、RAG 评估方法论与工具（RAGAS 等）、KG-RAG/GraphRAG 双实战项目 |
| 第03章-工具调用与Function-Calling | 第十章 智能体通信协议 | MCP/A2A/ANP 协议解析；Extra05 Agent Skills 与 MCP 对比 |
| 第04章-Agent范式-ReAct与规划 | 第四章 智能体经典范式构建 | 手把手实现 ReAct/Plan-and-Solve/Reflection 三范式的可运行代码 |
| 第05章-Agent工作流与编排 | 第五章 低代码平台 + 第六章 框架开发实践 | Coze/Dify/n8n 低代码对比；AutoGen/AgentScope/LangGraph 框架实战 |
| 第06章-多智能体系统 | 第六章 + 第十五章 构建赛博小镇 | 多智能体框架应用；"赛博小镇"社会模拟完整案例 |
| 第07章-记忆与上下文工程 | 第八章 记忆与检索 + 第九章 上下文工程 + Extra02 | 记忆系统分层设计、上下文工程方法论、社区补充知识 |
| 第08章-Agent评估与生产落地 | 第十二章 智能体性能评估 + Extra09/Extra10 | 核心指标/基准测试/评估框架；踩坑经验与 Agent 自进化四类闭环 |
| 附录E-从零构建智能体 | 第七章 构建你的Agent框架 | 基于 OpenAI 原生 API 从 0 构建自研框架（HelloAgents）；第十三章智能旅行助手、第十四章 DeepResearch Agent 复现 |
| —（本地缺） | 第十一章 Agentic-RL + Extra01/Extra11/Extra12 | SFT→GRPO 的 LLM 训练实战；Agent 面试题与参考答案；GUI/Web Agent 实战 |

### 3.2 LLM发展（16 个文件）—— 主补充源：happy-llm（34k stars）、so-large-lm、self-llm

| 本地文件 | datawhale 仓库与路径 | 补充点 |
| :--- | :--- | :--- |
| 第01章-LLM发展历程总览 | happy-llm ch1/ch4；so-large-lm ch01 | NLP 任务演进、LLM 涌现能力分析 |
| 第02章-架构演进 | happy-llm ch2/ch3；so-large-lm ch03；diy-llm | Encoder-only/Encoder-Decoder/Decoder-only 对比；手把手搭 Transformer |
| 第03章-位置编码与长上下文演进 | so-large-lm；leedl-tutorial | RoPE/ALiBi 直观推导 |
| 第04章-注意力与效率演进 | so-large-lm ch03；zero-to-sglang | 注意力变体与推理优化 |
| 第05章-对齐技术演进 | happy-llm ch8；so-large-lm | GRPO/OPD/Search-R1/ReTool（Agentic RL） |
| 第06章-推理能力演进 | leegenai-tutorial；unlock-deepseek | CoT/测试时扩展；DeepSeek 系列工作解读复现 |
| 第07章-LLM训练全流程实务 | happy-llm ch6；so-large-lm ch06 | Transformers 框架 Pretrain/SFT/PEFT 实战 |
| 第08章-训练监控与权重续训 | thorough-pytorch 第七章 | SwanLab/wandb 监控集成 |
| 第09章-SFT等后训练教程 | happy-llm ch6；self-llm；post-training-of-llms | LoRA/QLoRA 高效微调；各开源模型微调实操 |
| 第10章-Model-Infra-性能剖析与算子优化 | llm-algo-leetcode `03/04/05`；zero-to-sglang | Triton/CUDA kernel；mini-sglang 从零推理引擎 |
| 第11章-Scaling-up-规模化扩展 | so-large-lm ch08；diy-llm（MoE/缩放定律） | 数据/模型/流水线并行；Scaling Laws |
| 附录A-NLP基础 | happy-llm ch1；base-llm | NLP 基础与文本表示演进 |
| 附录B-开源大模型实战 | **self-llm**（32k stars）；handy-ollama；unlock-hf | 各开源模型（Qwen/GLM/LLaMA）Linux 微调部署"食用指南" |
| 附录C-Prompt工程与RAG实战 | llm-cookbook（吴恩达中译）；all-in-rag | 提示工程课程；RAG 全栈 |
| 附录D-从零构建LLM | **happy-llm ch5**；llms-from-scratch-cn；code-your-own-llm；tiny-universe | 亲手实现 LLaMA2 + 训练 Tokenizer + 215M 预训练模型（可下载复现）；GLM4/Llama3/RWKV6 逐步构建 |
| Mamba/0-Mamba与SSM详解 | diy-llm；llm-preview | SSM/MoE 等架构专题 |

### 3.3 具身智能（约 60 个文件）—— 主补充源：every-embodied（3.7k stars）、dive-into-embodied-ai、learn-world-model

| 本地章节 | datawhale 资源 | 补充点 |
| :--- | :--- | :--- |
| 第01章-具身智能概论 | every-embodied `01-具身智能概述`；dive-into-embodied-ai | 定义/发展历程/趋势；完整技能树（仿真→决策→控制→感知） |
| 第02章-3D视觉与多模态感知 | every-embodied 第5章（视觉） | 相机/点云工程化 |
| 第03章-机器人运动学与控制 | every-embodied `02-机器人基础`（DH 参数/动力学/坐标变换）；hello-robotics | 运动学/动力学讲义与仿真链路 |
| 第04章-VLA | **every-embodied `06-策略抓取/VLA`（19 篇模型导读）** | SmolVLA 训练部署、OpenVLA 复现、RT-1/RT-2/RT-X 论文+代码分析、DiT4DiT、EventVLA、WALL-OSS/WALL-X、3DVLA、PhysBrain、PRTS、Galaxea G0.5、Dexora、LWD 真机机群、VisualThink、Agentic-VLA、Dexbotic-RLinf、DM0.5、EVA-Client 真机部署 |
| 第05章-模仿学习 | every-embodied 第6章（LeRobot 遥操作数据采集） | 真机数据采集流程 |
| 第06章-强化学习与Sim2Real | every-embodied 第二阶段；easy-rl 蘑菇书 | RL 基础系统化；"识别-规划-抓取"闭环 |
| 第07章-仿真器与基准 | **every-embodied `09-具身智能数据及评估基准`**（LIBERO/SimplerEnv）；dive-into-embodied-ai | LIBERO/SimplerENV 基准实操；Isaac 链路 |
| 第08章-世界模型与规划 | **every-embodied `17-具身世界模型`（9 篇导读）+ learn-world-model 课程** | LeWM、RAW-Dream、WoG、WALL-WM、RoboDream、Gamma-World、BWM、τ0-WM、GE-Sim 2.0；世界模型架构深拆课程 |
| 第09章-前沿与工业实践 | every-embodied `13-其他前沿项目`（无人机多模态 LLM 导航）+ `12-导航与主动感知`（VLN/ETPNav/NBS） | VLN 导航复现；主动感知路径规划 |

### 3.4 深度学习手撕（17 个文件）—— 主补充源：leedl-tutorial（17k stars）

| 本地章节 | datawhale 资源 | 补充点 |
| :--- | :--- | :--- |
| 第01章-神经网络基础（激活/损失/反传） | leedl-tutorial；d2l-ai-solutions-manual | 课程笔记 + 《动手学深度学习》习题解答 |
| 第02章-优化算法 | thorough-pytorch 第五~六章 | 优化器/调度器代码 |
| 第03章-CNN | thorough-pytorch 第十章（ResNet 源码解读）；dive-into-cv-pytorch | ResNet 逐行解读；CV 完整教程 |
| 第04章-RNN | thorough-pytorch 第十章（LSTM 实战） | LSTM 实现 |
| 第05章-Transformer架构 | happy-llm ch2；leedl-tutorial（Self-attention 作业） | 手把手搭 Transformer；注意力作业 |
| —（本地缺） | unusual-deep-learning（水很深的深度学习）、learning-terrain（学习的地形）、hack-rnns、hand-bert | 学习动力学视角、从零搭 BERT/RNN |

---

## 四、搜推广与生成式推荐分类映射（215 个文件）

### 4.1 核心发现：fun-rec 已升级为《深度推荐算法实践（小麦书）》

fun-rec 现在的结构与本库「经典推荐 → GR研究 → SID技术」的演进线**高度重合**：

| fun-rec 章节 | 对应本地内容 | 补充点 |
| :--- | :--- | :--- |
| 上篇 ch1 推荐系统概述 | 经典推荐/第01章 | 概览对齐 |
| 上篇 ch2 快速候选召回（CF/矩阵分解/向量召回 I2I·U2I/序列召回/流式索引） | 经典推荐/第25章-召回与向量检索 | 物品CF/用户CF/矩阵分解讲义；**全量兴趣建模与流式索引**是本地薄弱点 |
| 上篇 ch3 精准偏好预测（记忆与泛化/二阶·高阶特征交叉/局部激活注意力/兴趣演化/行为到会话序列/多目标结构·任务依赖·损失融合/多塔·动态权重多场景） | 经典推荐/第24章-特征工程与特征交叉、第04章-多任务学习、第18章-多场景 | 系统化的多目标（ESMM/PLE 谱系）与多场景（Star topology/PEPNet 谱系）讲义 |
| 上篇 ch4 重排多样性建模（MMR/DPP/Transformer 重排/排列组合重排） | 经典推荐/第26章-粗排精排与重排 | **重排多样性专题是本地最薄的一环**，MMR/DPP 有完整推导 |
| 下篇 ch5 生成式推荐基础（判别式 vs 生成式本质区别/Transformer/Diffusion 基石/LLM 三阶段范式/**推荐中的 Tokenizer 技术：范式演进、端到端离散化、工业级方案、核心挑战**） | GR研究/第01章、SID技术 全 12 章 | 与 SID 技术逐章对应，**工业级 Tokenizer 方案与挑战**是关键补充 |
| 下篇 ch6 Scaling Law 架构探索（HSTU 演进/工程突破/混合范式/**硬件感知架构设计/统一序列与特征交叉建模**） | GR研究/第08章-01-HSTU | 硬件感知（Hardware-Aware）视角是本地缺的角度 |
| 下篇 ch7 端到端生成式建模（OneRec-V1/V2、OneSug、OneSearch、EGA、GPR） | GR研究/第09章-04 补充 GenRec 经典论文 | OneRec 全家族最新演进 |
| 下篇 ch8 会思考的推荐模型（物品索引学习/**PLUM 框架**/语义对齐训练、OneRec-Think：物品对齐·推理激活·推理增强·Think-Ahead、RecZero/RecOne） | GR研究/第08章、第09章 deep-dives/25-PLUM | 思考型推荐（Reasoning Rec）系统性框架 |
| 下篇 ch9 基于扩散的推荐（Diffusion 分类/前向加噪反向去噪/训练与采样/条件生成、DiffuASR/Diff-MSR/AsymDiffRec/DMSG） | 经典推荐/第20章-世界模型、数学/第07章-10 Diffusion 案例 | 扩散推荐完整专题，本地较薄 |
| 下篇 ch10 生产级推荐系统构建（系统架构/离线流水线/在线流程/部署运维） | 工程与职业/1-推荐系统端到端设计、2-生产工程与模型服务化 | 端到端生产项目（含前端与交互） |

### 4.2 其余子分类映射

| 本地子分类/章节 | datawhale 资源 | 补充点 |
| :--- | :--- | :--- |
| PTH教程 | torch-rechub；thorough-pytorch | 13+ 排序模型（DeepFM/Wide&Deep/DCN/DCN-v2/DIN/DIEN…）可运行实现与基准 |
| RL | easy-rl（蘑菇书 15 章）；joyrl-book；fun-marl | RL 经典教材系统化；多智能体 RL |
| 工业架构/论文分析 | fun-rec ch3/ch6（工业界架构谱系）；so-large-lm | 大厂方案谱系对齐 |
| 经典推荐/第09~11章 Python/数据/PyTorch | joyful-pandas；hands-on-data-analysis；scientific-computing；powerful-numpy | 数据处理工程化 |
| 经典推荐/第27章-在线学习与AB实验 | competition-baseline；hands-on-data-analysis | 竞赛中的实验设计 |
| 经典推荐/第19章-知识蒸馏 | awesome-compression | 模型压缩专题（量化/剪枝/蒸馏） |
| 经典推荐/附录E-面试与Q&A | daily-interview `04-ai-algorithms`（推荐部分） | 推荐算法面经题库 |
| deep-dives（50 篇论文深度解析） | datawhale 无对应（本身已超纲） | 本库独有优势，无需补充 |

---

## 五、工程与职业分类映射（16 个文件）

| 本地文件 | datawhale 仓库与路径 | 补充点 |
| :--- | :--- | :--- |
| 0-K8s与ML模型部署 | **llm-deploy**（LLM 推理和部署理论与实践）；handy-ollama；zero-to-sglang；docker-notes | vLLM/推理引擎/量化部署实战；Docker 基础 |
| 0-学习库结构梳理 / 0-工程师核心能力图谱 | （无直接对位仓库） | meta 章节，仅风格参考 |
| 1-推荐系统端到端设计 | fun-rec `ch10 生产级推荐系统构建` | 离线流水线/在线流程/前端交互/部署运维完整项目 |
| 1-社招1年经验简历模板与制作手册 | get-job；daily-interview `07-project-experience` | 求职资料与项目经验写法 |
| 10-沟通与跨团队协作 | daily-interview `08-behavioral-interview` | 行为面试中的沟通案例 |
| 11-算法工程师面试全攻略 | daily-interview `05-system-design + 06-development-tech/big-data` | 系统设计面 + 大数据开发面经 |
| 14-Git开发规范 | faster-git `lecture06 GitFlow + lecture07 提交规范` | 工作流与提交规范标准化 |
| 2-生产工程与模型服务化 | llm-deploy；wow-fullstack（部署篇） | 大模型在线推理服务化 |
| 3-AB测试与因果推断 | math-for-ai `ch6 假设检验`；fun-rec `ch4 多样性` | 统计基础；多样性对实验指标的影响 |
| 4-数据工程与特征平台 | joyful-pandas；hands-on-data-analysis；juicy-bigdata（大数据处理导论） | 数据分析/大数据技术栈 |
| 5-监控与可观测性 | （datawhale 覆盖弱）| Gap，需外部 SRE 资料 |
| 6-代码质量与工程实践 | sweetalk-design-pattern；easy-vibe `stage-3 superpowers（TDD）` | 设计模式与 TDD 工作流 |
| 7-生产调试与失败案例 | daily-interview `06-development-tech`（OS 章） | 仅部分 OS 案例可补；系统性事故案例 Gap |
| 8-职业发展与晋升路线 | daily-interview `08-behavioral`；get-job | 行为面与职级参考 |
| 9-项目展示与技术影响力 | daily-interview `07-project-experience`；vibe-blog | 项目包装与影响力案例 |

---

## 六、面试分类映射（88 个文件）

| 本地章节 | datawhale 仓库与路径 | 补充点 |
| :--- | :--- | :--- |
| 第01章~第05章（数组/链表/哈希/树/DP/贪心/回溯） | **leetcode-notes** `docs/ch01~ch05` | 官方图解、规范题解、复杂度标注——与本地大纲 1:1 对应，补图解与标准解法 |
| 第06章-高频题1-50 / 第07章-高频题51-100 | leetcode-notes（热题 100 全覆盖）；daily-interview | 每题的官方图解/多解法对比；datawhale 成员真实面经交叉验证 |
| 第08章-所有习题解析 | leetcode-notes `ch08 题解清单` | 分类清单（Categories-List 对齐） |
| 第09章-挑战程序设计竞赛精选专题（线段树/BIT/博弈论/网络流等） | team-learning-program `DataStructureAndAlgorithm/07 树.md + 08 图.md`；**huawei-od-python**（choice100/others100/questions200 共 400+ 题） | 树/图讲义做理论地基；华为 OD 场景化题库补习题量（本地最薄点） |
| 第10章-LeetCode实战刷题集 | leetcode-notes；huawei-od-python | 按专题刷题路线 |
| 第11章~第15章（DP 优化/字符串高级/数学/图论/计算几何） | team-learning-program `07 树 + 08 图`；huawei-od-python | 高级专题理论；**计算几何 datawhale 几乎无覆盖（Gap）** |
| 附录A~E | leetcode-notes 附录；hello-agents Extra01（Agent 面试题） | 复杂度速查补充；AI 岗新增面试题 |
| 面试笔记/0-LLM面试笔记-2026 | **daily-interview `04-ai-algorithms/llm/llm-interview-questions.md`（48KB 题库）** | 1:1 强对应，最高优先合并/校对对象 |
| 面试笔记/1-GenRec面试笔记-2026 | fun-rec `ch5~ch8`（GenRec 面试向）+ daily-interview 推荐部分 | HSTU/OneRec/RQ-VAE/语义索引题库 |
| 面试笔记/2-算法工程师通用面试 | daily-interview 全库 | ML/CV/NLP/推荐/开发五方向面经 |
| 面试笔记/3-公司面试真题与经验 | daily-interview `03-interview-experience` | 真题经验帖 |

---

## 七、顶层文档映射（1 个文件）

| 本地文件 | datawhale 资源 | 补充点 |
| :--- | :--- | :--- |
| 工作沟通方法论.md | daily-interview `08-behavioral-interview` | 仅行为面案例可参考；会议/RFC/跨时区模板 datawhale 无覆盖（Gap） |

---

## 八、高价值资源排行

综合全部 6 大分类的映射密度与内容差异度：

1. **fun-rec（小麦书）** —— 搜推广分类的"对照镜像教程"，上篇补级联架构、下篇补生成式范式，是本次对照中唯一在「生成式推荐」上有完整体系的 datawhale 仓库
2. **hello-agents** —— Agent 分类的系统性教材（16 章 + 13 Extra），补协议/评估/Agentic-RL/面试题四大缺口
3. **leetcode-notes + daily-interview + huawei-od-python** —— 面试分类三件套：图解题库 + 真实面经 + 场景化题库
4. **every-embodied + learn-world-model** —— 具身智能分类的独有补充（19 篇 VLA 导读 + 9 篇世界模型导读 + LIBERO/SimplerEnv 基准实操）
5. **math-for-ai + pumpkin-book + key-book** —— 数学分类三大理论支柱（严格推导 + 公式详解 + 学习理论）
6. **happy-llm + so-large-lm + self-llm** —— LLM 分类三件套（手搓 + 原理 + 实战部署）
7. **thorough-pytorch + torch-rechub + llm-algo-leetcode** —— 工程实现三件套（框架细节 + 推荐模型实现 + 系统级优化）

---

## 九、薄弱章节与补充方案 Top 10

| # | 本地薄弱章节 | 薄弱点 | 补充方案（datawhale 来源） |
| :-: | :--- | :--- | :--- |
| 1 | 经典推荐/第26章-粗排精排与重排 | 重排多样性（MMR/DPP）几乎空白 | fun-rec `ch4`（贪心 MMR、行列式点过程 DPP、Transformer 重排、排列组合重排完整推导）→ 已建伴生补充 |
| 2 | SID技术/第03章-RQ-VAE | 工业级 Tokenizer 方案与挑战视角 | fun-rec `ch5 推荐中的 Tokenizer 技术`（范式演进/端到端离散化/工业级方案）→ 已建伴生补充 |
| 3 | GR研究/第08章-01-HSTU | 缺硬件感知架构与 OneRec-Think"会思考的推荐" | fun-rec `ch6 Scaling Law + ch8 PLUM/OneRec-Think/RecZero` → 已建伴生补充 |
| 4 | 面试/第09章-挑战程序设计竞赛精选专题 | 习题量严重不足（讲多题少） | huawei-od-python（400+ 场景题）+ team-learning-program `07 树/08 图` 讲义 → 已建伴生补充 |
| 5 | 工程与职业/0-K8s与ML模型部署、2-生产工程与模型服务化 | 缺大模型推理服务化实战 | llm-deploy + handy-ollama + zero-to-sglang + docker-notes → 已建伴生补充 |
| 6 | Agent技术/第02章-RAG | 缺评估体系与高级检索（混合检索/Text2SQL/KG-RAG） | all-in-rag `ch4~ch9` → 已建伴生补充 |
| 7 | Agent技术/第07章-记忆与上下文工程 | 缺系统性上下文工程方法论 | hello-agents `ch8/ch9 + Extra02` → 已建伴生补充 |
| 8 | 具身智能/第04章 VLA、第08章世界模型 | 模型覆盖面（本地 6 篇 vs every-embodied 19 篇）、世界模型 9 篇新导读 | every-embodied `06 + 17`、learn-world-model → 已建伴生补充 |
| 9 | 数学/第06章-03 二阶方法 | 缺严格收敛率证明 | math-for-ai `ch7` + key-book `ch7` → 已建伴生补充 |
| 10 | 工程能力/第07章-性能剖析与优化 | 缺 kernel 级 profiling 与 Triton/CUDA 实战 | llm-algo-leetcode `03/04/05` + thorough-pytorch `第七章` → 已建伴生补充 |

次优先级（✅ 第二批已完成伴生文件）：经典推荐/第24章特征交叉、第25章召回与向量检索、第04章多任务学习；搜推广/RL 教程对照；LLM发展/第05章对齐、第11章 Scaling、附录B 开源模型实战；工程能力/第04章设计模式；Agent技术/第03章协议、第08章评估；面试笔记/0-LLM面试笔记；手撕/第05章 Transformer（共 12 个，见第十一节第二批索引）。

---

## 十、双向 Gap 分析

### 10.1 datawhale 有、而本库缺失或很弱（建议吸收）

| 主题 | datawhale 来源 | 说明 |
| :--- | :--- | :--- |
| 重排多样性建模（MMR/DPP） | fun-rec ch4 | 本地几乎空白 |
| 推荐中的 Tokenizer 工业级方案 | fun-rec ch5 | 本地 SID 偏原理、缺工业方案对比 |
| 硬件感知（Hardware-Aware）架构设计 | fun-rec ch6 | 本地 HSTU 章缺此视角 |
| 思考型推荐（OneRec-Think/RecZero/RecOne） | fun-rec ch8 | 本地 deep-dives 有 PLUM 单篇，缺体系 |
| 扩散模型做推荐（数据增强/多样性） | fun-rec ch9 | 本地仅有 Diffusion 数学案例 |
| RAG 评估体系（RAGAS 等）与混合检索 | all-in-rag ch4/ch6 | 本地 RAG 章缺评估闭环 |
| Agent 通信协议（MCP/A2A/ANP 对比） | hello-agents ch10 | 本地只有 Function Calling |
| Agentic-RL（SFT→GRPO 训练实战） | hello-agents ch11；happy-llm ch8 | 本地缺训练侧实操 |
| VLA 新模型跟踪（19 篇导读） | every-embodied 06 | 本地仅 6 篇经典模型 |
| 具身世界模型专题（9 篇） | every-embodied 17 | 本地第08章偏经典（Dreamer 系） |
| LIBERO/SimplerEnv 基准实操 | every-embodied 09 | 本地第07章偏仿真器介绍 |
| LLM 推理部署（vLLM/SGLang/Ollama） | llm-deploy/zero-to-sglang/handy-ollama | 本地 K8s 章偏 K8s 侧 |
| LLM 面试题库 | daily-interview llm 章 | 与本地笔记可互校 |
| 场景化算法题库（400+） | huawei-od-python | 本地竞赛章习题量不足 |
| Docker/Linux/SQL 基础 | docker-notes/easy-linux/wonderful-sql | 本地工具链层缺口 |
| Python 交互式作业体系 | learn-python-the-smart-way-v2 | 本地编程章缺配套练习 |
| ML 理论（VC 维/泛化界/遗憾界） | key-book | 本地数学章理论证明弱 |

### 10.2 本库有、而 datawhale 无（本库独有优势，反哺方向）

| 主题 | 说明 |
| :--- | :--- |
| SID 技术体系（12 章，RQ-VAE → 部署 → 评估 → 下游集成） | datawhale 仅 fun-rec ch5 一章 |
| GR 工业架构 deep-dives（50+ 篇大厂论文逐篇解析） | datawhale 无 |
| 因果推断体系（Do-calculus/工具变量/Causal Forest） | datawhale 无因果推断仓库 |
| K8s 与 ML 模型部署、监控可观测性、生产调试失败案例 | datawhale 缺 SRE/云原生向内容 |
| 搜推广科研方法论（读论文/发现问题/Scaling 实验/字节七剑案例） | datawhale 无 |
| 竞赛级算法专题（第9~15章：网络流/计算几何/字符串高级） | datawhale 覆盖浅 |
| 工作沟通方法论 | datawhale 无职场软技能仓库 |
| 数值计算与稳定性、泛函/NTK 等数学深水区 | math-for-ai 未覆盖 |

---

## 十一、伴生补充文件索引

本次已在对应章节目录下创建的伴生补充文件（每个文件含：来源地图、实质内容补充、可动手实践、延伸链接）：

| 分类 | 伴生文件 | 对照源 |
| :--- | :--- | :--- |
| 数学 | `基础/数学/数学基础/第06章-最优化/第06章-03-二阶方法-Newton·BFGS·L-BFGS·Trust-Region-Datawhale补充.md` | math-for-ai ch7 + key-book ch7 |
| 工程能力 | `基础/工程基础/工程能力/第07章-性能剖析与优化-Datawhale补充.md` | llm-algo-leetcode + thorough-pytorch |
| 工具 | `基础/工具与配置/工具手册/第02章-Vibe-Coding全栈配置手册-Datawhale补充.md` | easy-vibe stage-3 |
| Agent | `算法与AI/Agent技术/第02章-RAG检索增强生成-Datawhale补充.md` | all-in-rag |
| Agent | `算法与AI/Agent技术/第07章-记忆与上下文工程-Datawhale补充.md` | hello-agents ch8/9 + Extra02 |
| LLM | `算法与AI/LLM发展/附录D-从零构建LLM-Datawhale补充.md` | happy-llm ch5 + llms-from-scratch-cn |
| 具身 | `算法与AI/具身智能/第04章-视觉语言动作模型-VLA/VLA模型清单与跟踪-Datawhale补充.md` | every-embodied 06 |
| 具身 | `算法与AI/具身智能/第08章-世界模型与规划/具身世界模型专题-Datawhale补充.md` | every-embodied 17 + learn-world-model |
| GR | `搜推广与生成式推荐/GR研究/GR/第08章-前沿与LLM4Rec/第08章-01-HSTU与生成式推荐架构-Datawhale补充.md` | fun-rec ch6~8 |
| SID | `搜推广与生成式推荐/SID技术/第03章-RQ-VAE与残差量化/推荐Tokenizer技术全景-Datawhale补充.md` | fun-rec ch5 |
| 经典推荐 | `搜推广与生成式推荐/经典推荐/第26章-粗排精排与重排-Datawhale补充.md` | fun-rec ch4 + torch-rechub |
| 工程与职业 | `工程与职业/0-K8s与ML模型部署-Datawhale补充.md` | llm-deploy + handy-ollama + zero-to-sglang |
| 面试 | `面试/第09章-挑战程序设计竞赛精选专题/算法题库扩展-Datawhale补充.md` | huawei-od-python + team-learning-program |

**第二批（12 个）**：

| 分类 | 伴生文件 | 对照源 |
| :--- | :--- | :--- |
| 经典推荐 | `搜推广与生成式推荐/经典推荐/第24章-特征工程与特征交叉-Datawhale补充.md` | fun-rec ch3（记忆vs泛化/FM谱系/DCN-v2）+ torch-rechub |
| 经典推荐 | `搜推广与生成式推荐/经典推荐/第25章-召回与向量检索-Datawhale补充.md` | fun-rec ch2（CF/双塔/**全量兴趣建模与流式索引**） |
| 经典推荐 | `搜推广与生成式推荐/经典推荐/第04章-多任务学习-Datawhale补充.md` | fun-rec ch3 多目标（ESMM/PLE/损失融合配方） |
| RL | `搜推广与生成式推荐/RL/RL/RL教程对照-Datawhale补充.md` | easy-rl 蘑菇书（贝尔曼推导/值线策略线）+ joyrl-book |
| LLM | `算法与AI/LLM发展/第05章-对齐技术演进-Datawhale补充.md` | happy-llm ch8（GRPO 实战/Search-R1/ReTool）+ hello-agents ch11 |
| LLM | `算法与AI/LLM发展/第11章-Scaling-up-规模化扩展-Datawhale补充.md` | so-large-lm ch08（并行账本/3D 并行）+ diy-llm（Scaling 实证） |
| LLM | `算法与AI/LLM发展/附录B-开源大模型实战-Datawhale补充.md` | self-llm 食谱 + handy-ollama + unlock-hf |
| 工程能力 | `基础/工程基础/工程能力/第04章-代码抽象与设计模式-Datawhale补充.md` | sweetalk-design-pattern（23 GoF × 工程映射） |
| Agent | `算法与AI/Agent技术/第03章-工具调用与Function-Calling-Datawhale补充.md` | hello-agents ch10（MCP/A2A/ANP）+ Extra05 |
| Agent | `算法与AI/Agent技术/第08章-Agent评估与生产落地-Datawhale补充.md` | hello-agents ch12 + Extra01/09/10 |
| 面试 | `面试/面试笔记/0-LLM面试笔记-2026-Datawhale补充.md` | daily-interview LLM 题库 + hello-agents Extra01 |
| 手撕 | `算法与AI/深度学习手撕/第05章-Transformer架构/手撕Transformer-Datawhale补充.md` | happy-llm ch2 + thorough-pytorch 第十章 + leedl-tutorial |

> **第三批（6 个）**：

| 分类 | 伴生文件 | 对照源 |
| :--- | :--- | :--- |
| LLM | `算法与AI/LLM发展/附录C-Prompt工程与RAG实战-Datawhale补充.md` | llm-cookbook（吴恩达课程）+ all-in-rag |
| 数学 | `基础/数学/数学基础/第03章-概率论/第03章-08-…-Datawhale补充.md` | math-for-ai ch6（检验构造/功效/CI 对偶/贝叶斯 A-B） |
| 具身 | `算法与AI/具身智能/第07章-仿真器与基准/仿真器与基准实操-Datawhale补充.md` | every-embodied 09 章（LIBERO/SimplerENV） |
| 工程与职业 | `工程与职业/4-数据工程与特征平台-Datawhale补充.md` | joyful-pandas + hands-on-data-analysis + juicy-bigdata |
| 工程与职业 | `工程与职业/14-Git开发规范-Datawhale补充.md` | faster-git lecture05/06/07（对象模型/GitFlow/提交规范） |
| 面试 | `面试/第14章-高级图论/高级图论讲义与题源-Datawhale补充.md` | team-learning-program《08 图》+ huawei-od-python |

> **第四批（10 个）**：

| 分类 | 伴生文件 | 对照源 |
| :--- | :--- | :--- |
| 工程与职业 | `工程与职业/2-生产工程与模型服务化-Datawhale补充.md` | llm-deploy 服务层 + wow-fullstack（SLO/灰度/降级） |
| 工程与职业 | `工程与职业/1-推荐系统端到端设计-Datawhale补充.md` | fun-rec ch10 生产级推荐系统（总装章） |
| PTH | `搜推广与生成式推荐/PTH教程/第01章-…-Datawhale补充.md` | thorough-pytorch 保存加载 + torch-rechub（checkpoint 工程/安全加载） |
| 数学 | `基础/数学/数学基础/第05章-信息论/第05章-08-…-Datawhale补充.md` | so-large-lm（极大似然=最小KL 等价链/PPL 可比性） |
| 具身 | `算法与AI/具身智能/第05章-模仿学习/模仿学习数据与前沿-Datawhale补充.md` | every-embodied 03/05（LeRobot 数据流水线/格式） |
| Agent | `算法与AI/Agent技术/第04章-Agent范式-ReAct与规划-Datawhale补充.md` | hello-agents ch4（三范式代码骨架/组合形态） |
| Agent | `算法与AI/Agent技术/第06章-多智能体系统-Datawhale补充.md` | hello-agents（通信拓扑/赛博小镇案例） |
| Agent | `算法与AI/Agent技术/附录E-从零构建智能体-Datawhale补充.md` | hello-agents ch7/13/14（自研内核/案例阶梯） |
| LLM | `算法与AI/LLM发展/第07章-LLM训练全流程实务-Datawhale补充.md` | happy-llm ch6（数据打包/Trainer/收尾检查单） |
| 经典推荐 | `搜推广与生成式推荐/经典推荐/第19章-知识蒸馏-Datawhale补充.md` | awesome-compression（蒸馏三通道/压缩四件套组合） |

> **第五批（8 个）**：

| 分类 | 伴生文件 | 对照源 |
| :--- | :--- | :--- |
| LLM | `算法与AI/LLM发展/第02章-架构演进-Datawhale补充.md` | happy-llm ch2~3（组件 diff 清单/Decoder-only 论证） |
| LLM | `算法与AI/LLM发展/第09章-SFT等后训练教程-Datawhale补充.md` | post-training-of-llms + SWIFT/LLaMA-Factory（数据构造/框架选型/评估闭环） |
| LLM | `算法与AI/LLM发展/第10章-Model-Infra-性能剖析与算子优化-Datawhale补充.md` | llm-algo-leetcode CUDA（GEMM 五级阶梯）+ zero-to-sglang（FlashAttention/PagedAttention 桥） |
| Agent | `算法与AI/Agent技术/第01章-Agent技术发展脉络-Datawhale补充.md` | hello-agents ch1~3（定义严格版/史前三线/四条件论证） |
| Agent | `算法与AI/Agent技术/第05章-Agent工作流与编排-Datawhale补充.md` | hello-agents ch5（Anthropic 五模式/三层辨析/人机协同） |
| 手撕 | `算法与AI/深度学习手撕/手撕基础模块第01-04章-Datawhale补充.md` | leedl-tutorial 作业 + d2l 解答 + hand-bert/hack-rnns 裸写 |
| 经典推荐 | `搜推广与生成式推荐/经典推荐/第16章-生成式推荐与小型LLM设计模式-Datawhale补充.md` | happy-llm ch5（三阶段产线→GR 映射/产线检查点/训练-服务桥） |
| 工业架构 | `搜推广与生成式推荐/工业架构/工业架构与fun-rec工业谱系-Datawhale补充.md` | fun-rec 工业小节（四类降级改造/五层视图）+ torch-rechub 基线 |

> **第六批（9 个，收尾批）**：

| 分类 | 伴生文件 | 对照源 |
| :--- | :--- | :--- |
| LLM | `算法与AI/LLM发展/第03章-位置编码与长上下文演进-Datawhale补充.md` | happy-llm ch2 RoPE 推导 + so-large-lm（外推/插值分类） |
| LLM | `算法与AI/LLM发展/第06章-推理能力演进-Datawhale补充.md` | happy-llm ch8 RLVR + tiny-universe（两条正交线收拢） |
| LLM | `算法与AI/LLM发展/第01章-LLM发展历程总览-Datawhale补充.md` | so-large-lm ch1 + happy-llm ch1（三时代框架/开源谱系） |
| 具身 | `算法与AI/具身智能/第01章-具身智能概论/概论与学习路径-Datawhale补充.md` | every-embodied ch1/19（概念辨析/组队路径） |
| 具身 | `算法与AI/具身智能/第06章-强化学习与Sim2Real/RL与Sim2Real-Datawhale补充.md` | easy-rl（PPO/SAC 地基）+ every-embodied（GPU 并行/DR 工具箱） |
| 手撕 | `算法与AI/深度学习手撕/第09章-推理优化/推理优化-Datawhale补充.md` | awesome-compression 量化谱系 + zero-to-sglang（分层组合矩阵） |
| 手撕 | `算法与AI/深度学习手撕/第10章-经典模型架构/经典模型架构-Datawhale补充.md` | thorough-pytorch 第十章（ViT/Swin 源码）+ happy-llm（自检锚点） |
| 工具 | `基础/工具与配置/工具手册/Linux与容器环境-Datawhale补充.md` | easy-linux + docker-notes（环境 SOP/GPU 容器化） |
| 面试 | `面试/第10章-LeetCode实战刷题集/LeetCode专题刷题法-Datawhale补充.md` | leetcode-notes（四阶段路线/标签模板卡） |

**批次累计：13 + 12 + 6 + 10 + 8 + 9 = 58 个伴生文件。**

**体系导航层**（配套两份元文档，建议收藏）：

- 《[Datawhale补充速查索引](Datawhale补充速查索引.md)》— 58 个文件的一句话要点速查表（按分类组织，含主源标注）；
- 《[Datawhale补充阅读路线图](Datawhale补充阅读路线图.md)》— 依赖顺序阅读路线（主干六阶段 / 数学支撑线 / 面试线 / 四条角色化快捷路径 / 硬依赖速查表）。

> **体系饱和声明**：datawhalechina 高价值主线（fun-rec、hello-agents、happy-llm、every-embodied、all-in-rag、easy-rl、thorough-pytorch、torch-rechub、math-for-ai、so-large-lm、key-book、llm-algo-leetcode、daily-interview、huawei-od-python、team-learning-program、easy-vibe、llm-deploy、zero-to-sglang、post-training-of-llms、awesome-compression、faster-git、joyful-pandas、easy-linux、docker-notes、leetcode-notes 等 25+ 条）已全部落位伴生补充，不再规划后续批次。若需长尾深化（如第05/07章补充内的次级主题单列、新仓库跟进），按需单点添加即可。datawhale 无覆盖的领域（因果推断、联邦学习、计算几何、职场软技能等）见第十节 10.2 的本库独有优势清单，反哺方向而非补充方向。
