#!/usr/bin/env python3
# 全库文档优化脚本 · 2026-09-26
# 修复项：①2个frontmatter被污染文件重构 ②复盘文档补frontmatter ③12个多H1文档降级 ④30条失效链接修复
import os, re

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.chdir(root)
FIXED = []

def load(p): return open(p, encoding='utf-8').read()
def save(p, t): open(p, 'w', encoding='utf-8').write(t)

# ---------- 1. 修复两个 frontmatter 被污染的文件 ----------
f1 = "基础/数学/数学基础/第02章-线性代数/第02章-09-稀疏与低秩-Embedding压缩·LoRA·矩阵分解推荐.md"
f2 = "算法与AI/具身智能/第08章-世界模型与规划/第08章-02-DreamerV3-状态空间模型在机器人.md"

fm1 = '''---
title: 稀疏与低秩：Embedding 压缩 · LoRA · 矩阵分解推荐
tags:
  - 稀疏矩阵
  - 低秩分解
  - LoRA
  - Embedding 压缩
  - 矩阵分解
  - ALS
  - 线性代数
description: 稀疏与低秩两大压缩视角：CSR 稀疏格式、LoRA 低秩微调、Funk SVD/ALS 矩阵分解与工业 Embedding 压缩方法
chapter_id: 数学-2-09
last_updated: 2026-07-14
status: ✅ v1.0
---

# 第02章-09 稀疏与低秩：Embedding 压缩 · LoRA · 矩阵分解推荐

> 本章导读:'''

fm2 = '''---
title: DreamerV3：状态空间模型在机器人
tags:
  - DreamerV3
  - RSSM
  - 世界模型
  - POMDP
  - 变分推断
  - ELBO
  - 具身智能
description: 机器人工程师视角拆解 DreamerV3：POMDP→RSSM 三模块→变分推断与 ELBO 推导→PyTorch 端到端实现与想象 rollout
chapter_id: 具身-8-02
last_updated: 2026-07-14
status: ✅ v1.0
---

# 第08章-02 DreamerV3：状态空间模型在机器人

> 在机器人上直接训 RL，样本效率是出了名的低。'''

for path, fm, old_first in [
    (f1, fm1, "要把这件事"),
    (f2, fm2, "。你跑 1M 步"),
]:
    lines = load(path).split('\n')
    assert lines[14] == 'status: ✅ v1.0', (path, lines[14])
    rest = lines[15:]           # 从被截断的首行起
    assert rest[0].startswith(old_first), (path, rest[0][:30])
    new = fm + rest[0] + '\n' + '\n'.join(rest[1:])
    save(path, new)
    FIXED.append("重构frontmatter+H1: " + path)

# ---------- 2. 复盘文档：加 frontmatter（H1 降级在步骤 3 统一处理会保留首个） ----------
fk = "顶层文档/个人规划/2026-09职业状态复盘与3个月Reset计划.md"
fm_k = '''---
title: 2026 年 9 月职业状态复盘与 3 个月 Reset 计划
tags:
  - 职业规划
  - 复盘
  - 个人成长
description: 2026-09 职业状态复盘与以"随时能跳槽"为标准的 3 个月 Reset 计划
last_updated: 2026-09-19
status: ✅ v1.0
---

'''
save(fk, fm_k + load(fk))
FIXED.append("补frontmatter: " + fk)

# ---------- 3. 多 H1 文件降级（围栏代码块外，保留首个 H1） ----------
multi = '''基础/数学/数学基础/第01章-高等数学/第01章-07-级数·收敛·Fourier分析-Swin-Transformer的数学直觉.md
基础/数学/数学基础/第04章-离散数学/第04章-04-图论基础-顶点·边·路径·连通性·树·二分图.md
搜推广与生成式推荐/PTH教程/第01章-PTH权重读取与可视化教程.md
搜推广与生成式推荐/SID技术/第02章-向量量化基础/第02章-附录-VQ-VAE完整PyTorch实现.md
搜推广与生成式推荐/SID技术/第07章-在线SID生成与服务/第07章-附录-在线SID服务实现.md
搜推广与生成式推荐/SID技术/第08章-SID评估与监控/第08章-附录-评估脚本实现.md
算法与AI/LLM发展/第06章-推理能力演进.md
算法与AI/深度学习手撕/第08章-微调与对齐/第08章-05-SFT实战.md
面试/第01章-算法入门与数组篇/第01章-02-数组基础.md
面试/第02章-基础数据结构篇上-链表堆栈队列/第02章-01-链表.md
面试/第09章-挑战程序设计竞赛精选专题/09.04.1-线段树实战.md
顶层文档/个人规划/2026-09职业状态复盘与3个月Reset计划.md'''.split('\n')

def demote_h1(text):
    out, seen, fence = [], False, False
    for ln in text.split('\n'):
        if ln.startswith('```') or ln.startswith('~~~'): fence = not fence
        if not fence and re.match(r'^# ', ln):
            if seen: ln = '##' + ln[1:]
            else: seen = True
        out.append(ln)
    return '\n'.join(out)

for rel in multi:
    path = os.path.join(root, rel)
    save(path, demote_h1(load(path)))
    FIXED.append("H1降级: " + rel)

# ---------- 4. 失效链接修复 ----------
def fix_links(rel, pairs):
    path = os.path.join(root, rel)
    t = load(path); orig = t
    for old, new in pairs: t = t.replace(old, new)
    if t != orig: save(path, t); FIXED.append("链接修复: " + rel)

import glob
for p in glob.glob("基础/数学/数学基础/**/*.md", recursive=True):
    fix_links(os.path.relpath(p, root),
              [("](../../../深度学习手撕/", "](../../../../算法与AI/深度学习手撕/")])

fix_links("基础/工程基础/编程/附录A-深度调研报告.md",
          [("](../数学基础/第07章-系统思维/", "](../../数学/数学基础/第07章-系统思维/")])
fix_links("搜推广与生成式推荐/SID技术/第06章-HY-Token与混合Tokenization/第06章-附录-HY-Token实现.md",
          [("](../../sid_implementation/", "](../sid_implementation/")])
for rel in ["搜推广与生成式推荐/SID技术/第08章-SID评估与监控/第08章-02-碰撞率与覆盖率.md",
            "搜推广与生成式推荐/SID技术/第08章-SID评估与监控/第08章-04-码本利用率与信息损失.md",
            "搜推广与生成式推荐/SID技术/第12章-生成式检索与约束解码/第12章-03-SID生成模型的Scaling.md"]:
    fix_links(rel, [("](../../../基础/数学基础/", "](../../../基础/数学/数学基础/")])
fix_links("搜推广与生成式推荐/经典推荐/第25章-召回与向量检索.md",
          [("](../SID教程/", "](../SID技术/")])
fix_links("搜推广与生成式推荐/经典推荐/附录B.md",
          [("](../../基础/数学基础/", "](../../基础/数学/数学基础/"),
           ("](../../基础/工程能力/", "](../../基础/工程基础/工程能力/"),
           ("](../../深度学习手撕/", "](../../算法与AI/深度学习手撕/")])
fix_links("算法与AI/Agent技术/附录E-从零构建智能体.md",
          [("](算法与AI/Agent技术/", "](./")])

print("修复操作 %d 项:" % len(FIXED))
for f in FIXED: print("  OK", f)
