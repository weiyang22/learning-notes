---
title: Datawhale 补充速查索引（58 文件一句话版）
tags:
  - 速查
  - Datawhalechina
  - 索引
description: 全部 58 个"-Datawhale补充"伴生文件的一句话要点速查表，按分类组织，含主源标注
companion_to: Datawhalechina对照补充报告.md
source: 体系内 58 个伴生文件
last_updated: 2026-09-26
status: 📖 v1.0 (速查索引)
---

# Datawhale 补充速查索引

> 用法：想快速回忆某个补充文件讲了什么 → 查本表；要看完整映射与 Gap 分析 → 《[Datawhalechina对照补充报告](Datawhalechina对照补充报告.md)》；要按依赖顺序学 → 《[Datawhale补充阅读路线图](Datawhale补充阅读路线图.md)》。链接均相对于本目录（顶层文档/）。

## 一、基础（7）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [数学·第06章-03 二阶方法](../基础/数学/数学基础/第06章-最优化/第06章-03-二阶方法-Newton·BFGS·L-BFGS·Trust-Region-Datawhale补充.md) | Newton/BFGS/L-BFGS 推导与收敛率标度，two-loop 递归代码 | math-for-ai + key-book |
| [数学·第03章-08 假设检验](../基础/数学/数学基础/第03章-概率论/第03章-08-数理统计·假设检验·置信区间-A·B-实验的科学基础-Datawhale补充.md) | 检验的反证法构造、功效分析样本量公式、p 值误用表、CI 与检验对偶、贝叶斯 A/B | math-for-ai ch6 |
| [数学·第05章-08 信息论在LLM中](../基础/数学/数学基础/第05章-信息论/第05章-08-信息论在LLM中的角色-Datawhale补充.md) | 极大似然 ⟺ 最小交叉熵 ⟺ 最小 KL 的等价链，PPL 四个可比性前提 | so-large-lm |
| [工程能力·第04章 设计模式](../基础/工程基础/工程能力/第04章-代码抽象与设计模式-Datawhale补充.md) | 23 种 GoF 模式 × DL/搜推广工程场景映射表（含优先级）+ 重构叙事学习法 | sweetalk-design-pattern |
| [工程能力·第07章 性能剖析](../基础/工程基础/工程能力/第07章-性能剖析与优化-Datawhale补充.md) | roofline 定位 + profiler 三步法 + Triton fused softmax 实操 | llm-algo-leetcode + thorough-pytorch |
| [工具·第02章 Vibe-Coding](../基础/工具与配置/工具手册/第02章-Vibe-Coding全栈配置手册-Datawhale补充.md) | 长任务四机制、Agent Teams 角色、Spec Coding（requirements/design/tasks）、Mom Test | easy-vibe |
| [工具·Linux 与容器环境](../基础/工具与配置/工具手册/Linux与容器环境-Datawhale补充.md) | 环境配置 SOP 六步、GPU 容器化三关键件（镜像/透传/挂载纪律）、排错最短路径表 | easy-linux + docker-notes |

## 二、算法与AI · LLM 发展（12）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [第01章 发展历程总览](../算法与AI/LLM发展/第01章-LLM发展历程总览-Datawhale补充.md) | 三时代框架（每时代一个主导矛盾）、BERT/GPT 两路线"分→合→分流"、开源谱系坐标 | so-large-lm ch1 + happy-llm ch1 |
| [第02章 架构演进](../算法与AI/LLM发展/第02章-架构演进-Datawhale补充.md) | 原始 Transformer → LLaMA 系的组件 diff 清单（每处改动归因"训练稳/推理省"）+ Decoder-only 论证 | happy-llm ch2~3 |
| [第03章 位置编码与长上下文](../算法与AI/LLM发展/第03章-位置编码与长上下文演进-Datawhale补充.md) | RoPE 复数推导主线（旋转产生相对位置）+ 长上下文外推/插值二分法（NTK/YaRN） | happy-llm ch2 + so-large-lm |
| [第05章 对齐技术演进](../算法与AI/LLM发展/第05章-对齐技术演进-Datawhale补充.md) | GRPO 训练实战（组采样/难度筛选/KL 位置）+ Search-R1/ReTool/RLVR 新范式 + 三阶段 Agentic 管线 | happy-llm ch8 + hello-agents ch11 |
| [第06章 推理能力演进](../算法与AI/LLM发展/第06章-推理能力演进-Datawhale补充.md) | 推理时干预 vs 训练时干预两条正交线收拢、RLVR 最小配方（验证器是灵魂）与 reward hacking 复现 | happy-llm ch8 + tiny-universe |
| [第07章 训练全流程实务](../算法与AI/LLM发展/第07章-LLM训练全流程实务-Datawhale补充.md) | 数据打包细节（EOS/移位陷阱）、TrainingArguments 速查、训练后四步检查单 | happy-llm ch6 |
| [第09章 SFT 等后训练](../算法与AI/LLM发展/第09章-SFT等后训练教程-Datawhale补充.md) | SFT 数据构造四链条（种子→三道闸→配比→格式）、框架选型（SWIFT/Factory/裸 Trainer）、评估三层闭环 | post-training-of-llms + self-llm |
| [第10章 Model-Infra](../算法与AI/LLM发展/第10章-Model-Infra-性能剖析与算子优化-Datawhale补充.md) | GEMM 五级优化阶梯（每级解决什么瓶颈）、FlashAttention/PagedAttention 的"算子→引擎"桥、基准三件套 | llm-algo-leetcode + zero-to-sglang |
| [第11章 Scaling](../算法与AI/LLM发展/第11章-Scaling-up-规模化扩展-Datawhale补充.md) | 并行策略显存-通信账本、Megatron 切分与气泡率公式、3D 并行配比、Scaling Law 作业级实证 | so-large-lm ch08 + diy-llm |
| [附录B 开源大模型实战](../算法与AI/LLM发展/附录B-开源大模型实战-Datawhale补充.md) | self-llm"食谱"模板（环境→微调→部署→评测）、显存门槛速查、HF 工具链地图、CPU 双轨 | self-llm + unlock-hf |
| [附录C Prompt 与 RAG](../算法与AI/LLM发展/附录C-Prompt工程与RAG实战-Datawhale补充.md) | 吴恩达式迭代提示开发流程、分类路由与审核链模式、LangChain 取舍判断 | llm-cookbook + all-in-rag |
| [附录D 从零构建 LLM](../算法与AI/LLM发展/附录D-从零构建LLM-Datawhale补充.md) | LLaMA2 组件级复现（四件套→215M 全流程）、SentencePiece、预训练-SFT-LoRA 产线 | happy-llm ch5 |

## 三、算法与AI · Agent 技术（9）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [第01章 发展脉络](../算法与AI/Agent技术/第01章-Agent技术发展脉络-Datawhale补充.md) | Agent 定义的严格版+四模块解剖、史前三线（符号/RL/BDI）遗产表、"为什么是 2023"四条件论证 | hello-agents ch1~3 |
| [第02章 RAG](../算法与AI/Agent技术/第02章-RAG检索增强生成-Datawhale补充.md) | 混合检索与 RRF、HyDE/查询分解、RAGAS 评估指标（Hit Rate/MRR/Faithfulness）、GraphRAG | all-in-rag |
| [第03章 工具调用与协议](../算法与AI/Agent技术/第03章-工具调用与Function-Calling-Datawhale补充.md) | Function Calling→MCP 的 N+M 逻辑、MCP 三原语、A2A 双栈心智模型、Skill vs MCP 判据 | hello-agents ch10 + Extra05 |
| [第04章 Agent 范式](../算法与AI/Agent技术/第04章-Agent范式-ReAct与规划-Datawhale补充.md) | ReAct/Plan-and-Solve/Reflection 三范式代码骨架对照（决策时机差异）、组合形态与选型判据 | hello-agents ch4 |
| [第05章 工作流与编排](../算法与AI/Agent技术/第05章-Agent工作流与编排-Datawhale补充.md) | Anthropic 五模式判据表、工作流→自主 Agent→多智能体三层决策图、人机协同与检查点 | hello-agents ch5 + easy-vibe |
| [第06章 多智能体系统](../算法与AI/Agent技术/第06章-多智能体系统-Datawhale补充.md) | 通信拓扑选择矩阵（含成本纪律）、赛博小镇案例拆解（记忆/规划/涌现评测） | hello-agents |
| [第07章 记忆与上下文](../算法与AI/Agent技术/第07章-记忆与上下文工程-Datawhale补充.md) | 记忆三层分类、上下文四操作（写选压删）、上下文预算表、记忆复用率统计 | hello-agents ch8/9 + Extra02 |
| [第08章 评估与生产落地](../算法与AI/Agent技术/第08章-Agent评估与生产落地-Datawhale补充.md) | 评估三层指标（结果/行为/成本）、五类生产事故模式 checklist、Extra01 题库、自进化四类闭环 | hello-agents ch12 + Extra01/09/10 |
| [附录E 从零构建智能体](../算法与AI/Agent技术/附录E-从零构建智能体-Datawhale补充.md) | Agent 最小完备集（四组件 200 行内核）、案例能力阶梯、"从零到框架"迁移路径 | hello-agents ch7/13/14 |

## 四、算法与AI · 深度学习手撕（4）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [手撕基础模块 第01~04章](../算法与AI/深度学习手撕/手撕基础模块第01-04章-Datawhale补充.md) | leedl 作业×d2l 习题映射表、hand-bert/hack-rnns"只用线性层"裸写、进 Transformer 前自检清单 | leedl-tutorial + hand-bert + hack-rnns |
| [手撕 Transformer（第05章）](../算法与AI/深度学习手撕/第05章-Transformer架构/手撕Transformer-Datawhale补充.md) | 自底向上手撕路线图（每步数值自检锚点）、论文 vs 工业实现差异表、leedl 作业 | happy-llm ch2 + thorough-pytorch |
| [第09章 推理优化](../算法与AI/深度学习手撕/第09章-推理优化/推理优化-Datawhale补充.md) | 推理优化分层组合矩阵（权重/计算/生成/显存/请求五层）、量化选型树、推测解码适用判据 | awesome-compression + zero-to-sglang |
| [第10章 经典模型架构](../算法与AI/深度学习手撕/第10章-经典模型架构/经典模型架构-Datawhale补充.md) | ResNet→ViT→Swin→LLaMA"同一张图"三问读法、LLaMA 实现三个高频 bug 锚点、mini-GPT 演练 | thorough-pytorch 第十章 + happy-llm |

## 五、算法与AI · 具身智能（6）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [第01章 概论与学习路径](../算法与AI/具身智能/第01章-具身智能概论/概论与学习路径-Datawhale补充.md) | 三个概念辨析（具身图灵测试/Moravec 悖论）、任务分类映射、十周入门路线图 | every-embodied ch1/19 |
| [第04章 VLA 清单与跟踪](../算法与AI/具身智能/第04章-视觉语言动作模型-VLA/VLA模型清单与跟踪-Datawhale补充.md) | 19 篇 VLA 论文四演化轴（规模/动作生成/感知增强/系统）清单与复现顺序 | every-embodied ch10 |
| [第05章 模仿学习数据](../算法与AI/具身智能/第05章-模仿学习/模仿学习数据与前沿-Datawhale补充.md) | 示范数据采集-清洗-配比流水线、LeRobot 数据格式、数据缺陷→算法的因果映射表 | every-embodied ch03/05 |
| [第06章 RL 与 Sim2Real](../算法与AI/具身智能/第06章-强化学习与Sim2Real/RL与Sim2Real-Datawhale补充.md) | PPO/SAC 运动控制分工、Isaac GPU 大规模并行范式、Sim2Real"gap 来源→工具"对照表 | easy-rl + every-embodied |
| [第07章 仿真器与基准实操](../算法与AI/具身智能/第07章-仿真器与基准/仿真器与基准实操-Datawhale补充.md) | 仿真器 vs 基准分层、LIBERO 四任务族实操、SimplerENV 真机替代评测、选型决策表 | every-embodied ch09 |
| [第08章 具身世界模型专题](../算法与AI/具身智能/第08章-世界模型与规划/具身世界模型专题-Datawhale补充.md) | 9 篇具身世界模型导读、learn-world-model 路线、LIBERO/SimplerENV 评测关联 | every-embodied ch17 + learn-world-model |

## 六、搜推广与生成式推荐（11）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [经典推荐·第04章 多任务学习](../搜推广与生成式推荐/经典推荐/第04章-多任务学习-Datawhale补充.md) | 任务关系三假设→结构选型（MMoE/ESMM/PLE）、损失融合四层配方（σ 加权/PCGrad） | fun-rec ch3 + torch-rechub |
| [经典推荐·第16章 小型 LLM 设计模式](../搜推广与生成式推荐/经典推荐/第16章-生成式推荐与小型LLM设计模式-Datawhale补充.md) | 三阶段产线→生成式推荐映射（预训练=行为序列）、多机产线检查点、训练-服务桥 | happy-llm ch5 |
| [经典推荐·第19章 知识蒸馏](../搜推广与生成式推荐/经典推荐/第19章-知识蒸馏-Datawhale补充.md) | 蒸馏三通道类型学（输出/特征/关系）、压缩四件套组合纪律（蒸馏→量化→剪枝） | awesome-compression |
| [经典推荐·第24章 特征交叉](../搜推广与生成式推荐/经典推荐/第24章-特征工程与特征交叉-Datawhale补充.md) | 记忆 vs 泛化第一性框架、FM→DCN-v2 演进链、FM 的 O(nk) 化简推导 | fun-rec ch3 + torch-rechub |
| [经典推荐·第25章 召回与向量检索](../搜推广与生成式推荐/经典推荐/第25章-召回与向量检索-Datawhale补充.md) | 召回三范式分类框架、I2I/U2I 对齐问题、全量兴趣建模与流式索引（薄弱点补强） | fun-rec ch2 |
| [经典推荐·第26章 重排](../搜推广与生成式推荐/经典推荐/第26章-粗排精排与重排-Datawhale补充.md) | MMR/贪心 DPP 完整推导（行列式=体积直觉）、Transformer 重排、torch-rechub 对照 | fun-rec ch4 + torch-rechub |
| [SID·Tokenizer 全景](../搜推广与生成式推荐/SID技术/第03章-RQ-VAE与残差量化/推荐Tokenizer技术全景-Datawhale补充.md) | Tokenizer 三代范式、端到端离散化（STE/Gumbel-Softmax/码表崩溃）、工业方案对照 | fun-rec ch5 |
| [GR·HSTU 与生成式架构](../搜推广与生成式推荐/GR研究/GR/第08章-前沿与LLM4Rec/第08章-01-HSTU与生成式推荐架构-Datawhale补充.md) | fun-rec ch6~8：硬件感知 roofline、OneRec 全家族（V1→V2→OneSug/OneSearch）、PLUM/OneRec-Think/RecZero | fun-rec ch6~8 |
| [RL·教程对照](../搜推广与生成式推荐/RL/RL/RL教程对照-Datawhale补充.md) | 贝尔曼方程推导链、值线/策略线算法谱系叙事、PPO 裁剪→GRPO 的衔接 | easy-rl + joyrl-book |
| [PTH·权重读取与可视化](../搜推广与生成式推荐/PTH教程/第01章-PTH权重读取与可视化教程-Datawhale补充.md) | checkpoint 工程（七件套续训）、weights_only 安全加载、Embedding 巨表双频保存与健康检查 | thorough-pytorch + torch-rechub |
| [工业架构·fun-rec 工业谱系](../搜推广与生成式推荐/工业架构/工业架构与fun-rec工业谱系-Datawhale补充.md) | "论文→工业"四类降级改造模式、工业 deep-dive 的五层收编视图、torch-rechub 基线锚点 | fun-rec 工业小节 |

## 七、工程与职业（5）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [1-推荐系统端到端设计](../工程与职业/1-推荐系统端到端设计-Datawhale补充.md) | fun-rec ch10 总装：端到端时延预算表（<60ms 分解）、三层缓存、前端埋点即标签体系、最小脚手架 | fun-rec ch10 |
| [2-生产工程与模型服务化](../工程与职业/2-生产工程与模型服务化-Datawhale补充.md) | LLM 服务 SLO 分层（TTFT/TPOT）、影子流量灰度、降级阶梯、服务化版本管理矩阵 | llm-deploy + wow-fullstack |
| [4-数据工程与特征平台](../工程与职业/4-数据工程与特征平台-Datawhale补充.md) | 特征加工 pandas 习语（时序泄漏坑）、EDA 方法轮、大数据栈地图（Spark/Flink↔离线/实时特征） | joyful-pandas + juicy-bigdata |
| [14-Git 开发规范](../工程与职业/14-Git开发规范-Datawhale补充.md) | 对象模型解释 reset/rebase/squash、GitFlow 三工作流适用边界、Conventional Commits + commitlint | faster-git lecture05~07 |
| [0-K8s 与 ML 模型部署](../工程与职业/0-K8s与ML模型部署-Datawhale补充.md) | 推理服务三层视角（引擎/服务/资源）、SGLang 内核视角、CPU 双轨、部署 checklist 合并 | llm-deploy + zero-to-sglang + handy-ollama |

## 八、面试（4）

| 补充文件 | 一句话要点 | 主源 |
| :--- | :--- | :--- |
| [第09章 算法题库扩展](../面试/第09章-挑战程序设计竞赛精选专题/算法题库扩展-Datawhale补充.md) | 400+ OD 场景题按考点配题单、"讲义→热身→场景→错题→重做"闭环、网络流建模套路清单 | huawei-od-python + team-learning-program |
| [第10章 LeetCode 专题刷题法](../面试/第10章-LeetCode实战刷题集/LeetCode专题刷题法-Datawhale补充.md) | 四阶段刷题路线（标签→Hot100→弱项→模拟）、标签模板卡（识别口诀）、题库分工表 | leetcode-notes + huawei-od-python |
| [第14章 高级图论讲义](../面试/第14章-高级图论/高级图论讲义与题源-Datawhale补充.md) | Dijkstra 证明与分层图、Tarjan 统一 low 框架表、网络流建模三要素模板卡 | team-learning-program 08 图 |
| [面试笔记·LLM 面试](../面试/面试笔记/0-LLM面试笔记-2026-Datawhale补充.md) | daily-interview LLM 题库按概念块对练表（三栏闭环）、hello-agents Extra01 Agent 专题 | daily-interview + hello-agents Extra01 |

---

> 统计：基础 7 · LLM 12 · Agent 9 · 手撕 4 · 具身 6 · 搜推广 11 · 工程与职业 5 · 面试 4 = **58**。
