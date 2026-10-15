#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Docsify 站点所需的三个文件（在仓库根目录运行）:

  _sidebar.md                侧边栏目录树（自然排序，文件夹可折叠）
  _assets/wikimap.json       Obsidian 双链 [[文件名]] -> 实际路径 解析表
  _assets/search-index.json  轻量搜索索引（标题 + 各级标题 + 开头摘要）

用法: python3 tools/gen_sidebar.py
说明: 仓库的 pre-commit 钩子会在每次提交前自动重新生成并加入暂存区。
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {'_assets', 'tools', '.obsidian', '.git', '.github', 'node_modules', '__pycache__'}
SKIP_FILES = {'_sidebar.md', '_navbar.md', 'progress.md', 'scratch.md',
              'index.html', '网站使用指南.md', '.gitignore'}
# 隐私路径前缀：不进公开仓库（与 .gitignore 保持一致），线上站点完全不可见
PRIVACY_SKIP = ['顶层文档/个人规划', '工程与职业/13-社招求职']
TOP_ORDER = ['顶层文档', '算法与AI', '搜推广与生成式推荐', '基础',
             '工程与职业', '面试', '科目四']

FM_RE = re.compile(r'\A---\s*\n(.*?)\n---\s*\n', re.S)
TITLE_RE = re.compile(r'^title:\s*["\']?(.+?)["\']?\s*$', re.M)
FENCE_RE = re.compile(r'^(```|~~~)')

TOP_ITEMS = [
    ('🏠 主页', 'README.md'),
    ('📈 学习进度', 'progress.md'),
    ('📝 随手记', 'scratch.md'),
    ('📖 网站使用指南', '网站使用指南.md'),
]


def natural_key(s):
    return [(0, int(p)) if p.isdigit() else (1, p) for p in re.split(r'(\d+)', s)]


def fm_title(text):
    m = FM_RE.match(text)
    if not m:
        return None
    t = TITLE_RE.search(m.group(1))
    return t.group(1).strip() if t else None


def safe_label(s):
    """避免标题里的方括号破坏 markdown 链接"""
    return s.replace('[', '「').replace(']', '」')


def headings_and_intro(text):
    """扫描 md 文本（跳过代码块），返回 (h2/h3 标题列表, 开头摘要)"""
    heads, intro = [], []
    in_fence = False
    body = FM_RE.sub('', text, count=1) if FM_RE.match(text) else text
    for line in body.splitlines():
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        s = line.strip()
        if s.startswith('## ') and len(heads) < 40:
            heads.append(s[3:].lstrip('#').strip())
        elif s.startswith('### ') and len(heads) < 40:
            heads.append(s[4:].strip())
        elif len(intro) < 700:
            intro.append(s)
    return heads, clean_text(' '.join(intro))[:700]


def clean_text(t):
    t = re.sub(r'`[^`]*`', ' ', t)
    t = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', t)
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)
    t = re.sub(r'\[\[([^\]|]*)(?:\|([^\]]*))?\]\]',
               lambda m: (m.group(2) or m.group(1) or ''), t)
    t = re.sub(r'[#>*_~$\-]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def is_private(rel):
    p = rel.as_posix()
    return any(p == s or p.startswith(s + '/') for s in PRIVACY_SKIP)


def collect():
    """遍历全库，收集 md 文件（排除隐私路径）"""
    files = []
    for p in ROOT.rglob('*.md'):
        rel = p.relative_to(ROOT)
        parts = rel.parts
        if any(seg in SKIP_DIRS or seg.startswith('.') for seg in parts[:-1]):
            continue
        if p.name in SKIP_FILES or p.name.startswith('_') or p.name.startswith('.'):
            continue
        if is_private(rel):
            continue
        files.append(rel)
    return files


def build_sidebar(files):
    lines = ['- [{}]({})'.format(safe_label(n), quote(p)) for n, p in TOP_ITEMS]
    lines.append('')

    dirs = sorted({f.parts[0] for f in files if len(f.parts) > 1},
                  key=lambda d: (TOP_ORDER.index(d) if d in TOP_ORDER else 99,
                                 natural_key(d)))

    title_cache = {}
    def title_of(rel):
        if rel not in title_cache:
            try:
                title_cache[rel] = fm_title((ROOT / rel).read_text(encoding='utf-8')) or ''
            except Exception:
                title_cache[rel] = ''
        return title_cache[rel]

    def walk(dirpath, depth):
        base = ROOT / dirpath
        subs = sorted([d for d in base.iterdir()
                       if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')],
                      key=lambda d: natural_key(d.name))
        mds = sorted([f for f in base.iterdir()
                      if f.suffix == '.md' and f.name not in SKIP_FILES
                      and not f.name.startswith('_') and not f.name.startswith('.')],
                     key=lambda f: natural_key(f.name))
        for d in subs:
            if is_private(d.relative_to(ROOT)):
                continue
            lines.append('  ' * depth + '- **{}**'.format(safe_label(d.name)))
            walk(d, depth + 1)
        for f in mds:
            rel = f.relative_to(ROOT).as_posix()
            label = title_of(rel) or f.stem
            lines.append('  ' * depth + '- [{}]({})'.format(safe_label(label), quote(rel)))

    for d in dirs:
        lines.append('- **{}**'.format(safe_label(d)))
        walk(d, 1)
    return '\n'.join(lines) + '\n'


def build_wikimap(files):
    by_name, by_path, dups = {}, {}, []
    for rel in files:
        posix = rel.as_posix()
        by_path[posix[:-3]] = posix  # 不带 .md 的路径
        stem = rel.stem
        if stem in by_name and by_name[stem] != posix:
            dups.append(stem)
            continue  # 同名文件保留先出现的
        by_name[stem] = posix
    # 目录也纳入映射（Obsidian 双链可指向文件夹）：指向目录内 README.md 或自然排序第一个 md
    for d in sorted(ROOT.rglob('*')):
        if not d.is_dir() or d.name in SKIP_DIRS or d.name.startswith('.'):
            continue
        rel = d.relative_to(ROOT)
        if any(seg in SKIP_DIRS or seg.startswith('.') for seg in rel.parts):
            continue
        if is_private(rel):
            continue
        mds = sorted([f for f in d.iterdir()
                      if f.suffix == '.md' and f.name not in SKIP_FILES
                      and not f.name.startswith('_') and not f.name.startswith('.')],
                     key=lambda f: natural_key(f.name))
        readme = next((f for f in mds if f.stem == 'README'), mds[0] if mds else None)
        if readme:
            target = readme.relative_to(ROOT).as_posix()
            by_path.setdefault(rel.as_posix(), target)
            by_name.setdefault(d.name, target)
    return {'byName': by_name, 'byPath': by_path,
            'dups': sorted(set(dups))}, dups


def build_search_index(files):
    entries = []
    for rel in sorted(files, key=lambda r: (
            TOP_ORDER.index(r.parts[0]) if r.parts[0] in TOP_ORDER and len(r.parts) > 1 else 99,
            natural_key(r.as_posix()))):
        try:
            text = (ROOT / rel).read_text(encoding='utf-8')
        except Exception:
            continue
        heads, intro = headings_and_intro(text)
        entries.append({
            'p': rel.as_posix(),
            't': fm_title(text) or rel.stem,
            'm': rel.parts[0] if len(rel.parts) > 1 else '',
            'h': heads,
            'x': intro,
            'n': (ROOT / rel).stat().st_size,
        })
    return {'generated': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'),
            'count': len(entries), 'files': entries}


def main():
    files = collect()
    (ROOT / '_assets').mkdir(exist_ok=True)

    sidebar = build_sidebar(files)
    (ROOT / '_sidebar.md').write_text(sidebar, encoding='utf-8')

    wikimap, dups = build_wikimap(files)
    (ROOT / '_assets' / 'wikimap.json').write_text(
        json.dumps(wikimap, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

    index = build_search_index(files)
    (ROOT / '_assets' / 'search-index.json').write_text(
        json.dumps(index, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

    idx_size = (ROOT / '_assets' / 'search-index.json').stat().st_size / 1024
    print('md 文件: {} | sidebar: {:.0f}KB | wikimap 同名冲突: {} | 索引: {:.0f}KB'.format(
        len(files), len(sidebar) / 1024, len(set(dups)), idx_size))
    if dups:
        print('同名文件(双链解析取先出现者):', sorted(set(dups))[:10], file=sys.stderr)


if __name__ == '__main__':
    main()
