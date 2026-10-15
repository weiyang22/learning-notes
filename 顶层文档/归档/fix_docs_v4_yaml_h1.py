#!/usr/bin/env python3
# 全库文档优化脚本 v4 · 2026-09-26
# ① 56 个 YAML 不合法 frontmatter：给含冒号的 title/status/description 标量加引号
# ② 48 个多 H1 文件：围栏外多余 H1 降级为 H2（保留首个）
import os, re, yaml

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.chdir(root)

def q(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

n_yaml, n_h1 = 0, 0

for dp, dn, fn in os.walk('.'):
    dn[:] = [d for d in dn if d not in ('科目四', '.rlab')]
    for f in fn:
        if not f.endswith('.md'): continue
        path = os.path.join(dp, f)
        text = open(path, encoding='utf-8').read()
        m = re.match(r'^---\n(.*?)\n---\n', text, re.S)
        if not m: continue
        fm_text = m.group(1)

        # ① YAML 合法性修复
        try:
            yaml.safe_load(fm_text)
        except Exception:
            lines = fm_text.split('\n')
            out = []
            for ln in lines:
                mm = re.match(r'^(title|status|description|last_updated|chapter_id):\s?(.*)$', ln)
                if mm and mm.group(2) and not (mm.group(2).startswith('"') or mm.group(2).startswith('[')):
                    val = mm.group(2)
                    if ':' in val or val.startswith('|') or val.startswith('>'):
                        out.append("%s: %s" % (mm.group(1), q(val))); continue
                out.append(ln)
            new_fm = '\n'.join(out)
            try:
                yaml.safe_load(new_fm)
                text = '---\n' + new_fm + '\n---\n' + text[m.end():]
                n_yaml += 1
            except Exception:
                pass  # 无法自动修复的跳过

        # ② 多 H1 降级（行级围栏跟踪，保留首个）
        lines = text.split('\n')
        h1s = [i for i, ln in enumerate(lines) if ln.startswith('# ')]
        if len(h1s) > 1:
            fence = False
            seen = False
            changed = False
            for i, ln in enumerate(lines):
                if ln.startswith('```') or ln.startswith('~~~'): fence = not fence
                if fence: continue
                if ln.startswith('# '):
                    if seen: lines[i] = '##' + ln[1:]; changed = True
                    else: seen = True
            if changed:
                text = '\n'.join(lines)
                n_h1 += 1

        if text != open(path, encoding='utf-8').read():
            open(path, 'w', encoding='utf-8').write(text)

print(f"YAML 修复: {n_yaml}  多H1 降级: {n_h1}")
