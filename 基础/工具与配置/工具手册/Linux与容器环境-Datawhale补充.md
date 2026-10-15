---
title: Linux 与容器环境 · Datawhalechina 补充
tags:
  - 工具
  - Linux
  - Docker
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/easy-linux 与 docker-notes，补充 Linux 环境配置 SOP 与 GPU 容器化训练环境的实操手册
companion_to: 基础/工具与配置/工具手册/（Linux 环境向）
source: https://github.com/datawhalechina/easy-linux ; https://github.com/datawhalechina/docker-notes
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# Linux 与容器环境 · Datawhalechina 对照补充

> 本文件是对《基础/工具与配置/工具手册》Linux/容器方向的补充（本地工具手册已有 Git 与 Vibe-Coding 两册，环境向为空缺），内容整理自 [easy-linux](https://github.com/datawhalechina/easy-linux)（Linux 教程：从零开始）与 [docker-notes](https://github.com/datawhalechina/docker-notes)（Docker 从零到掌握）。定位：算法工程师的**环境搭建 SOP**——conda/CUDA/SSH/容器化的可复制配方。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地位置 | 补充价值 |
| :--- | :--- | :--- | :--- |
| easy-linux | 命令行/Shell/权限/软件管理 | （环境向空缺） | **核心补充** |
| docker-notes | 镜像/容器/网络/挂载/Compose | （环境向空缺） | **核心补充**：训练环境容器化 |
| self-llm 环境章 | CUDA/驱动矩阵 | （关联附录B 补充 2.1） | 互引 |

---

## 二、核心补充内容

### 2.1 算法工程师的环境 SOP（easy-linux 视角收编）

新机器/新服务器的标准配置流（每步一句要点）：

```text
1. SSH 免密 + tmux 常驻（断线不断训练）
2. conda 多环境隔离（每项目一 env，导出 environment.yml 入库）
3. 驱动/CUDA 检查：nvidia-smi 版本 ≥ CUDA Toolkit 需求（驱动向下兼容）
4. 磁盘规划：数据/缓存/代码分盘，HF 缓存重定向（HF_HOME）
5. 常用命令内化：find/grep/sed/awk、journalctl、df/du 排查
6. 权限纪律：少用 root、密钥不入库（衔接 .gitignore 卫生）
```

### 2.2 GPU 容器化训练环境（docker-notes × 训练场景）

Docker 用于训练环境的三个关键件（docker-notes 对应章节）：

1. **镜像选型**：用 NVIDIA 官方 PyTorch 镜像（`nvidia/cuda:x.x-cudnn-devel-ubuntu` 或 pytorch 官方 tag）打底，`pip install` 差量层放后（层缓存友好）；
2. **GPU 透传**：`docker run --gpus all`（依赖 nvidia-container-toolkit）；验证 `docker run --rm --gpus all nvidia/cuda:x.x nvidia-smi`；
3. **数据挂载**：代码挂载（`-v $(pwd):/workspace`）+ 数据集只读挂载（`:ro`）+ 输出目录独立——容器无状态、状态全在挂载卷，环境可复现的本质就在这条纪律。

**Compose 模板**：多服务（训练 + tensorboard + 数据服务）用 `docker-compose.yml` 声明，与《1-推荐系统端到端补充》2.3 的 deploy 目录同构——单机 Compose 是 K8s 的前站。

### 2.3 环境排错的最短路径表

| 症状 | 首查 | 命令 |
| :--- | :--- | :--- |
| CUDA OOM | 进程残留/批大小 | `nvidia-smi`、`fuser -v /dev/nvidia*` |
| 版本不匹配 | 驱动 vs Toolkit vs torch cu 版本 | `nvidia-smi`、`python -c "import torch; torch.version.cuda"` |
| 容器内看不到 GPU | toolkit 未装/--gpus 未加 | `nvidia-container-cli -k -d /dev/tty info` |
| 磁盘爆 | 缓存/镜像堆积 | `docker system df`、`du -sh ~/.cache/*` |
| 网络拉取慢 | 镜像源 | pip/conda/HF 三源配置 |

---

## 三、可直接动手的实践

1. 按 2.1 SOP 把一台新机器（或云实例）配置完并写成自己的 checklist；
2. 用 docker-notes 的知识做一个 PyTorch 训练镜像：Dockerfile 分层 + GPU 透传 + 三类挂载，跑通一次小训练；
3. 写一个 docker-compose.yml（训练 + tensorboard 两服务），并演练"删容器重建后训练状态可恢复"。

---

## 四、延伸链接

- easy-linux：<https://github.com/datawhalechina/easy-linux>
- docker-notes：<https://github.com/datawhalechina/docker-notes>
- 本地相关文件：工具手册/第01章-Git完全教程、第02章-Vibe-Coding全栈配置手册（本地开发环境）、第03章-全栈测试Agent-Skills-MCP配置手册、附录B-开源大模型实战-Datawhale补充.md（CUDA 矩阵）、工程与职业/0-K8s与ML模型部署-Datawhale补充.md（容器编排的下一站）
