#!/usr/bin/env python3
# 全库文档优化脚本 v2 · 2026-09-26
# 修复 2026-07-14 批次生成缺陷：frontmatter 缺关闭行 / title·tags·description 模板污染 / 正文 H1 错误
import os, re, sys

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.chdir(root)

BAD_TITLE = "分布式训练 (DDP / FSDP / DeepSpeed ZeRO)"
STRAY = "# 分布式训练 (DDP / FSDP / DeepSpeed ZeRO)"
GENUINE = "算法与AI/深度学习手撕/第07章-训练技术/第07章-02-分布式训练.md"

CAT_TAG = {}  # 目录 → 类别标签，构建时填充

def category_tag(rel):
    top = rel.split(os.sep)[0]
    if top == "算法与AI":
        return {"LLM发展": "LLM", "Agent技术": "Agent", "深度学习手撕": "深度学习",
                "具身智能": "具身智能"}.get(rel.split(os.sep)[1] if len(rel.split(os.sep)) > 2 else "", "AI")
    if top == "搜推广与生成式推荐": return "搜推广"
    if top == "基础":
        return {"数学": "数学", "工程基础": "工程", "学习方法": "学习方法", "工具与配置": "工具",
                "因果推断": "因果推断", "联邦学习": "联邦学习", "语言学习": "语言学习"}.get(
                rel.split(os.sep)[1] if len(rel.split(os.sep)) > 2 else "", "基础")
    return {"面试": "面试", "工程与职业": "工程", "顶层文档": "顶层"}.get(top, "学习")

def derive_tags(stem, rel):
    stop = {"第", "章", "和", "与", "的", "在", "从", "到"}
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

def yaml_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

def derive_desc(rest_lines, title):
    for ln in rest_lines[:60]:
        t = ln.strip()
        if not t or t == '---' or t.startswith(('#', '>', '|', '```', '~~~')): continue
        t = re.sub(r"[*_`$\\]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) >= 15:
            return ("本章要点：" + (t[:76] + "…" if len(t) > 78 else t))
    return "「%s」—— 本库系统学习笔记章节。" % title

fixed, skipped = [], []
md_files = []
for dirpath, dirnames, filenames in os.walk('.'):
    dirnames[:] = [d for d in dirnames if d != '科目四' and d != '.rlab']
    for fn in filenames:
        if fn.endswith('.md'): md_files.append(os.path.normpath(os.path.join(dirpath, fn)))

for path in sorted(md_files):
    rel = os.path.normpath(os.path.relpath(path, '.'))
    if rel == GENUINE: continue
    text = open(path, encoding='utf-8').read()
    lines = text.split('\n')
    if not lines or lines[0].strip() != '---': continue  # 无 frontmatter 的跳过（已另行处理）

    # 判定污染
    i_status = None
    for i in range(1, min(45, len(lines))):
        if lines[i].startswith('status:'): i_status = i; break
    if i_status is None: continue
    head = lines[1:i_status]
    title_bad = any(l.startswith('title: 分布式训练') for l in head)
    tags_bad = any(l.strip() == '- DDP' for l in head)
    desc_bad = any(l.startswith('description: 分布式训练 3 大策略') for l in head)
    has_stray = STRAY in lines[i_status+1:i_status+400] or any(
        l.strip() == STRAY for l in lines[i_status+1:])
    if not (title_bad or tags_bad or desc_bad or has_stray): continue

    stem = os.path.splitext(os.path.basename(rel))[0]
    title = stem
    tags = derive_tags(stem, rel)

    # 定位 close（status 之后 3 行内的 ---）
    i_close = None
    for j in range(i_status+1, min(i_status+4, len(lines))):
        if lines[j].strip() == '---': i_close = j; break

    if i_close is None:
        rest = lines[i_status+1:]
        desc = derive_desc(rest, title)
        # 去掉紧随的 stray H1
        while rest and (rest[0].strip() == STRAY): rest = rest[1:]
        new_lines = (["---"] + head + ["---", "", "# " + title, ""] + rest)
    else:
        rest = lines[i_close+1:]
        desc = derive_desc(rest, title)
        while rest and not rest[0].strip(): rest = rest[1:]
        while rest and (rest[0].strip() == STRAY): rest = rest[1:]
        new_lines = (["---"] + head + ["---", "", "# " + title, ""] + rest)

    # 回填正确的 title/tags/description
    out = []
    mode = None
    for ln in new_lines:
        if ln.startswith('title:'): out.append("title: " + yaml_str(title)); continue
        if ln.startswith('tags:'): mode = 'tags'; out.append("tags:"); continue
        if mode == 'tags':
            if ln.startswith('  - '): continue
            mode = None
        if ln.startswith('description:'): out.append("description: " + yaml_str(desc)); continue
        out.append(ln)
    open(path, 'w', encoding='utf-8').write('\n'.join(out))
    fixed.append(rel)

print("修复 %d 个文件" % len(fixed))
for f in fixed[:12]: print("  OK", f)
if len(fixed) > 12: print("  ... 共 %d" % len(fixed))
