---
title: VLA模型清单与跟踪 · Datawhalechina 补充
tags:
  - 具身智能
  - VLA
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/every-embodied 第10章（19+ 篇 VLA 模型导读），补充本地第04章 VLA 章节之外的模型跟踪清单与复现入口
companion_to: 第04章-视觉语言动作模型-VLA/（整章）
source: https://github.com/datawhalechina/every-embodied
last_updated: 2026-09-20
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# VLA 模型清单与跟踪 · Datawhalechina 对照补充

> 本文件是对同目录《第04章-视觉语言动作模型-VLA》整章的补充。本地第04章已覆盖 6 个经典模型（VLM 基础 / RT-1·RT-2 / Octo / OpenVLA / π0·HiRT）；[datawhalechina/every-embodied](https://github.com/datawhalechina/every-embodied) 的**第 10 章 VLA 大模型**维护了 19+ 篇持续更新的模型导读（含代码复现），是本地章节最有价值的"模型跟踪雷达"。本文件把它整理成对照清单。

---

## 一、补充来源地图

| every-embodied 10 章条目 | 主题 | 与本地第04章的关系 |
| :--- | :--- | :--- |
| SmolVLA-LIBERO（训练和部署） | HuggingFace 轻量 VLA 全流程 | 补充：小参数量 VLA 的训练实战（本地缺） |
| OpenVLA 复现 | 7B 开源 VLA | 对应本地第04章-05，every-embodied 提供**复现步骤** |
| RT-1/RT-2/RT-X 论文解读与代码分析 | Google 系开创工作 | 对应本地第04章-03，补代码分析视角 |
| DiT4DiT-LIBERO（训练与评估） | 扩散策略 | 补充：diffusion policy 线（本地缺） |
| EventVLA 视觉证据记忆 | 事件相机 + VLA | 补充：新模态方向 |
| WALL-OSS / WALL-X 工程框架 | 开源 VLA 模型 + 工程框架 | 补充：工程化部署视角 |
| 3DVLA 三维空间实例增强 | 3D 感知增强 VLA | 补充：3D 方向（与本地第02章衔接） |
| PhysBrain 物理常识增强 | 物理世界知识 | 补充：认知增强方向 |
| PRTS 强化学习原生 VLA | RL-native VLA | 补充：RL 与 VLA 结合（与本地第06章衔接） |
| Galaxea G0.5 自回归 VLA | 星海图系 | 补充：国产工业系 |
| Dexora 高自由度双臂灵巧 | 灵巧手 | 补充：灵巧操作方向 |
| LWD 真机机群强化学习 | 多机 RL | 补充：规模部署方向 |
| VisualThink 视觉证据推理 | 推理增强 VLA | 补充：CoT-for-Action 方向 |
| Agentic-VLA 在线适应 | Agent 化 VLA | 补充：与本地 Agent 分类交叉 |
| Dexbotic-RLinf 工程化后训练 | VLA 后训练 infra | 补充：训练 infra 视角 |
| DM0.5 高性能推理与 OpenDM | 推理优化 | 补充：推理引擎视角 |
| EVA-Client 真机部署与评测 | 真机工程 | 补充：真机落地视角 |

---

## 二、核心补充内容

### 2.1 VLA 模型的四个演化轴（基于 every-embodied 清单归纳）

把 19+ 篇导读按技术轴归类，比逐篇阅读更高效：

1. **规模轴**：SmolVLA（<1B，消费级 GPU 可训）→ OpenVLA（7B）→ 闭源大模型（π0 后继）；
   本地第04章的 6 篇集中在"经典锚点"，every-embodied 补的是该轴上持续移动的前沿。
2. **动作生成轴**：自回归 token 解码（RT 系/OpenVLA）→ 扩散/流匹配动作头（DiT4DiT、π0 系、DM0.5）→ RL 原生策略（PRTS、LWD）。
   判断一个新 VLA 属于哪类，直接决定它的推理时延与控制频率适用范围。
3. **感知增强轴**：2D 图像（经典）→ 3D/点云（3DVLA）→ 事件流（EventVLA）→ 物理常识（PhysBrain）。
4. **系统轴**：模型本体 → 工程框架（WALL-X、Dexbotic-RLinf、EVA-Client）→ 真机机群（LWD）。**工程轴是本地最缺的一环**，every-embodied 的独有价值就在这里。

### 2.2 复现优先级建议（面向本地第04章读者）

以 LIBERO 基准（见具身世界模型补充文件的基准部分）为统一评测场，every-embodied 提供复现步骤的推荐顺序：

1. **SmolVLA-LIBERO**：全流程最短路径（训练→评估→部署），消费级硬件可跑，适合建立"VLA 训练长什么样"的体感；
2. **OpenVLA 复现**：7B 规模的微调与部署，理解 LoRA 在 VLA 上的用法（与 LLM发展/第09章 PEFT 方法同构）；
3. **DiT4DiT**：切换到扩散动作头，对比自回归与扩散策略在成功率/时延上的差异；
4. **WALL-X / EVA-Client**：进入工程化与真机视角。

### 2.3 跟踪方法

every-embodied 的第 10 章持续追加新模型导读（README 中可见 2025 年内从 SmolVLA 一路更新到 EVA-Client/DM0.5）。建议：把它当 VLA 方向的"周更论文雷达"使用——本地第04章保持经典锚点的稳定性，新模型先入本清单的跟踪表，成熟后再升格为本地正式章节。

---

## 三、可直接动手的实践

1. 跑通 SmolVLA-LIBERO 训练（every-embodied 给出分步命令），记录显存占用与 step time；
2. 在 LIBERO-Spatial 上对比 OpenVLA（zero-shot）与 SmolVLA（fine-tuned）的成功率，形成自己的对照数据；
3. 把 2.1 的四轴分类表扩展成自己的"VLA 模型登记册"，每读一篇新论文填一行。

---

## 四、延伸链接

- every-embodied 仓库：<https://github.com/datawhalechina/every-embodied>
- VLA 章节（06-策略抓取或抓取VLA/大模型控制、VLA、VLM/）：
  <https://github.com/datawhalechina/every-embodied/tree/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM>
- 本地相关文件：第04章-05-OpenVLA（复现对照）、第04章-附录-VLA训练pipeline完整复现（pipeline 对照）、第08章-世界模型与规划/具身世界模型专题-Datawhale补充.md（世界模型侧）
