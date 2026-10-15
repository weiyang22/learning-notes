#!/usr/bin/env python3
"""分析每个MD文件的文本占比"""
import os
import re
import glob

BASE_DIR = "/Users/weiyang/Desktop/个人资料/资料/学习/基础/数学基础"

def text_ratio(filepath):
    """计算文件的文本占比(非代码块字符 / 总字符)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    total = len(content)
    if total == 0:
        return 0, 0, 0, 0, 0

    # 去掉代码块
    no_code = re.sub(r'```[\s\S]*?```', '', content)
    # 去掉纯公式行(粗略)
    no_code = re.sub(r'^\s*\$\$.*?\$\$\s*$', '', no_code, flags=re.MULTILINE)
    no_code = re.sub(r'^\s*\$[^\$]+\$\s*$', '', no_code, flags=re.MULTILINE)

    # 去掉空白
    no_code_no_space = re.sub(r'\s', '', no_code)
    return total, len(no_code), len(no_code_no_space), len(re.sub(r'\s', '', content))

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(BASE_DIR, "**/*.md"), recursive=True))
    print(f"总文件数: {len(files)}")
    print()
    under_50 = []
    for f in files:
        total, code, text_no_space, total_no_space = text_ratio(f)
        ratio = text_no_space / total if total else 0
        rel = f.replace(BASE_DIR + "/", "")
        status = "✅" if ratio >= 0.5 else "❌"
        print(f"{status} {ratio*100:5.1f}% {total:6d}c  {rel}")
        if ratio < 0.5:
            under_50.append((f, ratio, total, text_no_space))
    print()
    print(f"不达标(<50%)文件数: {len(under_50)}")
