---
title: 手撕基础模块（第01~04章） · Datawhalechina 补充
tags:
  - 深度学习
  - 手撕
  - 作业
  - 补充资料
  - Datawhalechina
description: 对照 datawhalechina/leedl-tutorial、d2l-ai-solutions-manual、hand-bert/hack-rnns，为神经网络基础~RNN 四章补充作业源与实现参照
companion_to: 深度学习手撕/第01~04章（基础模块）
source: https://github.com/datawhalechina/leedl-tutorial ; https://github.com/datawhalechina/d2l-ai-solutions-manual ; https://github.com/datawhalechina/hand-bert ; https://github.com/datawhalechina/hack-rnns
last_updated: 2026-09-26
status: 📖 v1.0 (Datawhalechina 对照补充)
---

# 手撕基础模块（第01~04章） · Datawhalechina 对照补充

> 本文件是对《深度学习手撕/第01章-神经网络基础 ~ 第04章-RNN循环神经网络》基础四章的合并补充，内容整理自 [leedl-tutorial](https://github.com/datawhalechina/leedl-tutorial)（李宏毅课程作业详解）、[d2l-ai-solutions-manual](https://github.com/datawhalechina/d2l-ai-solutions-manual)（《动手学深度学习》习题解答）与 [hand-bert](https://github.com/datawhalechina/hand-bert)、[hack-rnns](https://github.com/datawhalechina/hack-rnns)（两个"裸写"项目）。本地四章讲原理与实现；四份资料补**作业与裸写参照**——手撕能力的验证靠做题，不只靠读。

---

## 一、补充来源地图

| 来源 | 内容 | 对应本地章 | 补充价值 |
| :--- | :--- | :--- | :--- |
| leedl-tutorial | 李宏毅 HW 作业详解（回归/分类/CNN/RNN，含 ML2017/2021/2022 各版本） | 第01~04章 | **核心补充**：课程级作业链 |
| d2l-ai-solutions-manual | 《动手学》习题参考解答 | 第01~04章 | 习题自查 |
| hand-bert | 用线性层从零手搭 BERT | 第01章（关联第05章） | **核心补充**：裸写视角 |
| hack-rnns | 用线性层从零搭 NLP 模型 | 第04章 | **核心补充**：裸写视角 |

---

## 二、核心补充内容

### 2.1 四章的作业映射表（leedl 作业 × d2l 习题）

基础四章的"做题计划"，按章配作业：

| 本地章 | leedl 作业 | d2l 对应习题章 | 手撕自检目标 |
| :--- | :--- | :--- | :--- |
| 第01章 神经网络基础 | HW1 回归（PM2.5）、HW3 分类（图像） | ch3 线性神经网 / ch4 MLP | 反向传播手推 + `nn.Module` 裸写 MLP |
| 第02章 优化算法 | HW 中的 optimizer 消融任务 | ch11 优化算法 | SGD/Momentum/Adam 从零实现并对比收敛 |
| 第03章 CNN | HW3-CNN 结构改造 | ch6/7 卷积神经网络 | 卷积/池化裸写（禁用 nn.Conv2d 的教学版） |
| 第04章 RNN | HW2 语音分类（RNN 变体对比） | ch9 循环神经网络 | RNN/LSTM 前向裸写 + 梯度消失实验 |

**闭环纪律**（与《第09章 算法题库扩展补充》同款）：读讲义 → 做对应作业 → 错题与"卡壳点"记入本地错题本 → 周末限时重做。

### 2.2 "裸写"项目的独特价值（hand-bert / hack-rnns）

两个项目的共同约束：**只用线性层 + 基础张量操作**搭出完整模型（BERT/RNN 家族），把 nn.Module 的高级封装全部展开：

- hand-bert：embedding + 位置编码 + 多头自注意力 + FFN + 预训练任务全裸写——是《手撕Transformer 补充》的 BERT 侧对称物（那份面向 Decoder/LLM 主干，这份面向 Encoder/预训练任务）；
- hack-rnns：用线性层实现 RNN/LSTM/GRU 门控——把第04章"门控如何对抗梯度消失"从公式变成可调试的代码；
- 使用方式：**先自己写，再对照**（顺序反了就变成抄代码），卡壳处回本地对应节看推导。

### 2.3 基础四章 → 第05章的衔接检查单

进入 Transformer 手撕前的自检（不达标先补对应作业）：

- [ ] 能手推两层 MLP 的反向传播（第01章 HW1 级）；
- [ ] 能解释 Adam 的三个一阶/二阶矩修正各自解决什么（第02章）；
- [ ] 能裸写一个含 padding 的 conv 前向（第03章）；
- [ ] 能说清 LSTM 三门与梯度通路的关系（第04章）。

---

## 三、可直接动手的实践

1. 按 2.1 表跑完第01~02章的作业（HW1 回归 + optimizer 消融），产出对比报告；
2. 用 hack-rnns 的方式裸写一个单层 LSTM，与 `nn.LSTM` 数值对齐；
3. 完成 2.3 检查单，未达标项回对应章补课后进入第05章。

---

## 四、延伸链接

- leedl-tutorial：<https://github.com/datawhalechina/leedl-tutorial>
- d2l-ai-solutions-manual：<https://github.com/datawhalechina/d2l-ai-solutions-manual>
- hand-bert：<https://github.com/datawhalechina/hand-bert>；hack-rnns：<https://github.com/datawhalechina/hack-rnns>
- 本地相关文件：第01~04章正文、第05章-Transformer架构/手撕Transformer-Datawhale补充.md（下一站）、面试/第09章…/算法题库扩展-Datawhale补充.md（题单闭环机制同款）
