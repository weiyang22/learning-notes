#!/usr/bin/env python3
# 全库文档优化脚本 v3 · 2026-09-26
# 补丁：① 为 263 个文件回填派生 tags ② 删除 190 处残留 stray H1 ③ 修复 H1 后孤行引用标记
import os, re

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.chdir(root)

STRAY = "# 分布式训练 (DDP / FSDP / DeepSpeed ZeRO)"
GENUINE = "算法与AI/深度学习手撕/第07章-训练技术/第07章-02-分布式训练.md"

def category_tag(rel):
    parts = rel.split(os.sep)
    top = parts[0]
    if top == "算法与AI" and len(parts) > 2:
        return {"LLM发展": "LLM", "Agent技术": "Agent", "深度学习手撕": "深度学习",
                "具身智能": "具身智能"}.get(parts[1], "AI")
    if top == "搜推广与生成式推荐": return "搜推广"
    if top == "基础" and len(parts) > 2:
        return {"数学": "数学", "工程基础": "工程", "学习方法": "学习方法", "工具与配置": "工具",
                "因果推断": "因果推断", "联邦学习": "联邦学习", "语言学习": "语言学习"}.get(parts[1], "基础")
    return {"面试": "面试", "工程与职业": "工程", "顶层文档": "顶层"}.get(top, "学习")

def derive_tags(stem, rel):
    parts = re.split(r"[-·]+", stem)
    tags = []
    for p in parts:
        p = p.strip()
        if not p or re.fullmatch(r"第?\d+", p) or p in tags: continue
        if re.match(r"^第\d+", p): continue
        if len(p) >= 2: tags.append(p)
        if len(tags) >= 5: break
    cat = category_tag(rel)
    if cat not in tags: tags.insert(0, cat)
    return tags[:6]

n_tags, n_stray, n_orphan = 0, 0, 0

for dp, dn, fn in os.walk('.'):
    dn[:] = [d for d in dn if d not in ('科目四', '.rlab')]
    for f in fn:
        if not f.endswith('.md'): continue
        path = os.path.join(dp, f)
        rel = os.path.normpath(os.path.relpath(path, '.'))
        if rel == GENUINE: continue
        text = open(path, encoding='utf-8').read()
        lines = text.split('\n')
        orig = text

        # ① 回填空 tags
        for i, ln in enumerate(lines[:40]):
            if ln.strip() == 'tags:':
                nxt = lines[i+1] if i+1 < len(lines) else ''
                if not nxt.startswith('  - '):
                    stem = os.path.splitext(f)[0]
                    tag_lines = ["  - " + t for t in derive_tags(stem, rel)]
                    lines[i+1:i+1] = tag_lines
                    n_tags += 1
                break

        # ② 删除残留 stray H1（正文中）
        before = len(lines)
        lines = [ln for ln in lines if ln.strip() != STRAY]
        if len(lines) != before: n_stray += 1

        # ③ H1 后孤行引用修复：首内容行无前缀、下一非空行是引用 → 并回引用
        h1_idx = None
        for i, ln in enumerate(lines):
            if re.match(r'^# ', ln): h1_idx = i; break
        if h1_idx is not None:
            j = h1_idx + 1
            while j < len(lines) and not lines[j].strip(): j += 1
            if j < len(lines):
                L1 = lines[j]
                if not L1.startswith(('>', '#', '|', '-', '```', '~~~')) and not re.match(r'^\d+\.', L1):
                    k = j + 1
                    while k < len(lines) and not lines[k].strip(): k += 1
                    if k < len(lines) and lines[k].startswith('> '):
                        lines[j] = '> ' + L1.lstrip()
                        n_orphan += 1

        new = '\n'.join(lines)
        if new != orig:
            open(path, 'w', encoding='utf-8').write(new)

print(f"回填 tags: {n_tags}  删除 stray H1: {n_stray}  修复孤行引用: {n_orphan}")
