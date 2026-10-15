---
title: macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南
tags:
  - macOS
  - Apple Silicon
  - MPS
  - PyTorch
  - 工具手册
  - 教程
description: 在 Apple Silicon (M1/M2/M3/M4) Mac 上从零搭建原生 ARM64 深度学习环境的完整实战指南：原生性检测、官方 Homebrew、uv 与 Miniforge 选型、PyTorch MPS 安装与冒烟测试、统一内存语义下的安全 OOM 应答、TensorFlow Metal 的兼容陷阱、VS Code / Git 配置与排错。
last_updated: 2026-08-16
status: ✅ v2.0
---

# macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南

> 一份面向 Apple Silicon (M1/M2/M3/M4) 系列的"零到跑通"配置手册，目标是让你在半小时内拥有一套原生的 ARM64 深度学习环境，并避开 macOS 上最常见的几个坑（MPS 内存误用、第三方 Homebrew 镜像、TensorFlow Metal 的兼容性）。

---

## 0. 为什么 macOS / Apple Silicon 需要一份独立指南

macOS 不是一个"开箱即用"的深度学习平台。三件事决定了它需要专门的配置流程：

1. **原生 ARM64 vs Rosetta 2 转译**。在 Apple Silicon 上，x86_64 的 Python 与 PyTorch 会跑在 Rosetta 2 下，性能损失 30%–50%，并且 MPS 后端不可用。"原生性"不是优化，是能否用 GPU 的前提。
2. **统一内存架构（Unified Memory）**。CPU 与 GPU 共享同一块物理 RAM，MPS 分配的是"系统内存"而非独立显存。这意味着 MPS 的 OOM 会让整台机器卡死而不是返回 CUDA 风格的 `OutOfMemoryError`。教程必须把"内存上限"讲清楚，而不是照搬 Linux/CUDA 的写法。
3. **包管理器与 Python 发行版的分裂**。Homebrew 在 Apple Silicon 上安装在 `/opt/homebrew`，Python 既可以用 Homebrew、Miniforge、官方安装器，也可以用新一代的 `uv`。选错了发行版，后面所有 C 扩展都装不上。

这份笔记的取舍：**只保留在 2026 年仍然成立的做法**，去掉已经过期的第三方镜像、不再更新的 TensorFlow Metal 默认推荐，以及一篇会"让机器卡死"的 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` 配方。

---

## 1. 范围与读者

- **范围**：原生 ARM64 macOS 的开发工具链、Python 环境、PyTorch MPS、TensorFlow Metal 的兼容性边界、VS Code / Git 基础配置。
- **不范围**：CUDA 生态（macOS 没有）、远程 Linux GPU 训练（见仓库其他章节）、具体模型训练 pipeline。
- **读者**：要在自己 Mac 上做学习、推理、小规模训练与实验的算法/工程同学，不需要 macOS 装机经验。

---

## 2. 前置条件（Prerequisites）

在动手之前，请确认以下几件事，否则后面任何一步都可能踩坑。

| 项 | 要求 | 验证命令 |
| :--- | :--- | :--- |
| 硬件 | Apple Silicon (M1 / M2 / M3 / M4 系列) | `uname -m` 输出 `arm64` |
| macOS | 13 Ventura 或更新（推荐 14 Sonoma / 15 Sequoia） | `sw_vers` |
| 可用磁盘 | 至少 20 GB 空闲 | `df -h /` |
| 可用内存 | 16 GB 及以上（跑模型建议 32 GB+） | `sysctl hw.memsize` |
| 管理员权限 | 安装 CLT / Homebrew 需要 | `sudo -nv` 静默测试 |

> **重要**：本指南假设你是 Apple Silicon。Intel Mac 的目录布局是 `/usr/local` 而非 `/opt/homebrew`，但 TensorFlow Metal 不支持 Intel GPU，建议迁移到 Apple Silicon 或使用云端 GPU。

---

## 3. 目录

1. [原生 ARM64 验证](#4-原生-arm64-验证)
2. [macOS 与命令行工具基础](#5-macos-与命令行工具基础)
3. [Homebrew 官方安装（Apple Silicon 路径）](#6-homebrew-官方安装apple-silicon-路径)
4. [终端、Shell 与 zsh 插件](#7-终端shell-与-zsh-插件)
5. [Python 环境：uv vs Miniforge 决策](#8-python-环境uv-vs-miniforge-决策)
6. [PyTorch MPS 安装与冒烟测试](#9-pytorch-mps-安装与冒烟测试)
7. [统一内存语义与安全的 OOM 应对](#10-统一内存语义与安全的-oom-应对)
8. [TensorFlow Metal：兼容路径而非默认](#11-tensorflow-metal兼容路径而非默认)
9. [常用深度学习 Python 包](#12-常用深度学习-python-包)
10. [VS Code：扩展、JSONC 设置、快捷键](#13-vs-code扩展jsonc-设置快捷键)
11. [Git 基础：SSH、配置、Aliases、lazygit](#14-git-基础ssh配置aliaseslazygit)
12. [常见踩坑与排错（Troubleshooting）](#15-常见踩坑与排错troubleshooting)
13. [自检清单与 FAQ](#16-自检清单与-faq)
14. [版本与变更](#17-版本与变更)

---

## 4. 原生 ARM64 验证

**最重要的一步**：在任何安装之前，确认你跑的是 ARM64，而不是 Rosetta 2 转译的 x86_64。

```bash
# 1. 硬件架构
uname -m
# 期望输出: arm64
# 若输出: x86_64, 说明你的 shell 正在 Rosetta 下跑,见下方修复

# 2. 操作系统大版本
sw_vers
# ProductName:    macOS
# ProductVersion: 14.x  或更高

# 3. 芯片型号（Apple Silicon 系列会显示 "Apple M*"）
sysctl -n machdep.cpu.brand_string
```

**如果 `uname -m` 输出 `x86_64`**：

1. 打开 `系统设置 → 通用 → 软件更新`，确认已经升级到原生 ARM64 macOS。
2. 检查 `访达 → 应用程序 → 实用工具 → 终端` 的"显示简介"，取消勾选 **"使用 Rosetta 打开"**。
3. 重新打开终端，再次跑 `uname -m`。

**验证 Python 也是 ARM64**（很多 Conda/Miniconda 安装器会装 x86_64 版本）：

```bash
python3 -c "import platform; print(platform.platform(), platform.machine())"
# 期望包含 arm64
```

---

## 5. macOS 与命令行工具基础

### 5.1 macOS 版本基线

| macOS | 备注 |
| :--- | :--- |
| 13 Ventura | PyTorch / TensorFlow Metal 的最低可用基线 |
| 14 Sonoma | 推荐基线，Xcode CLT 与 PyTorch 最新 wheels 兼容最好 |
| 15 Sequoia | 新机器预装，需注意部分企业 MDM 配置项 |

低于 13 的 macOS 不在本文档讨论范围。

### 5.2 Xcode Command Line Tools（CLT）

没有 CLT，Homebrew、Python C 扩展、`pip install` 一切都需要会失败。

```bash
# 安装 CLT（弹出 GUI 时按"安装"）
xcode-select --install

# 验证
xcode-select -p
# 期望输出: /Library/Developer/CommandLineTools

# 若 CLT 路径错乱（极少见）
sudo xcode-select --reset
```

> CLT 本身不带完整 Xcode IDE。如果只写 Python，不需要装 Xcode 应用本体，能省下十几个 GB。

---

## 6. Homebrew 官方安装（Apple Silicon 路径）

### 6.1 官方安装命令

只用官方脚本，**不要用第三方镜像**（如各种"国内加速脚本"），它们会在 `~/.zshrc` 注入额外配置且不再跟进上游：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装完成后，终端会提示运行下面两行把 Homebrew 加入 `PATH`。

### 6.2 Apple Silicon 的 PATH

Homebrew 在 Apple Silicon 上的默认前缀是 `/opt/homebrew`（与 Intel 上的 `/usr/local` 不同）。要让 `brew`、`git`、`python3` 等命令在 shell 中可用，必须把 `shellenv` 写入 `~/.zprofile`：

```bash
# 把 Homebrew 写入 PATH（Apple Silicon 默认 /opt/homebrew）
if [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
```

新开会话或重新 `source ~/.zprofile` 后验证：

```bash
which brew
# 期望: /opt/homebrew/bin/brew

brew --prefix
# 期望: /opt/homebrew
```

### 6.3 常用 CLI 工具

```bash
brew install git curl wget tree jq htop fzf ripgrep bat eza zoxide fastfetch lazygit tmux uv
```

> **可选**：`code` 命令由 VS Code 自身提供（见 §13.3），不要通过 Homebrew 安装它。

---

## 7. 终端、Shell 与 zsh 插件

### 7.1 终端选择

- **Ghostty**：免费、GPU 加速、原生 ARM64，配置简单。
- **iTerm2**：经典、丰富功能，Apple Silicon 原生支持。
- **Warp**：AI 增强，重度依赖者喜爱；不是所有人都习惯。

### 7.2 Oh My Zsh 与必装插件

```bash
# Oh My Zsh（官方）
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 必装插件
git clone https://github.com/zsh-users/zsh-autosuggestions \
    ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

git clone https://github.com/zsh-users/zsh-syntax-highlighting \
    ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

编辑 `~/.zshrc`：

```bash
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
    zoxide
    fzf
)
```

### 7.3 Starship Prompt（推荐）

```bash
brew install starship
echo 'eval "$(starship init zsh)"' >> ~/.zshrc
source ~/.zshrc
```

### 7.4 系统级体验小调优

```bash
# 键盘长按连续输入（取消重音字符弹出）
defaults write -g ApplePressAndHoldEnabled -bool false

# 显示隐藏文件（Finder: Cmd + Shift + .）
# 显示完整路径栏（Finder: Cmd + Option + P）
```

> `defaults write -g InitialKeyRepeat -int 10` 等命令存在，但现代 macOS 的"系统设置 → 键盘"已经够用，不必再回到 CLI。

---

## 8. Python 环境：uv vs Miniforge 决策

**原则**：不要使用系统 `/usr/bin/python3`，不要用 Homebrew 的全局 `python3` 装项目依赖。它们没有虚拟环境，会污染系统，并且版本受限于 macOS 更新。

| 选项 | 安装方式 | 速度 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **uv** | `brew install uv` 或独立脚本 | 极快（Rust 实现） | 项目级 venv、轻量 ML 实验 |
| **Miniforge** | `brew install --cask miniforge` | 中（conda 解析） | 复杂科学栈、需要 `conda install` 装非 PyPI 包 |
| Miniconda | 不推荐手动安装 | — | 已废弃 |
| Anaconda | 不推荐 | — | 商业许可问题 |

**推荐**：

- 新项目、纯 Python 包 → `uv`。
- 需要 R / Julia / Biopython / GROMACS 等 conda-only 包 → Miniforge。

### 8.1 方案 A：uv（推荐）

```bash
brew install uv

# 在项目目录下创建 venv
mkdir my-ml-project && cd my-ml-project
uv venv .venv --python 3.11
source .venv/bin/activate

# 装包
uv pip install torch torchvision torchaudio
```

### 8.2 方案 B：Miniforge

```bash
brew install --cask miniforge
conda init zsh
source ~/.zshrc

conda create -n dl python=3.11 -y
conda activate dl

# 包优先用 pip，conda 只用于装系统级 C 库
pip install torch torchvision torchaudio
```

---

## 9. PyTorch MPS 安装与冒烟测试

### 9.1 安装

PyTorch 在 Apple Silicon 上的 GPU 加速后端叫 **MPS**（Metal Performance Shaders），是 PyTorch 官方维护的。

```bash
# 在激活的虚拟环境中
pip install --upgrade pip
pip install torch torchvision torchaudio
```

> 不要加 `--index-url https://download.pytorch.org/whl/cpu`，否则会装 CPU-only 版本。

### 9.2 冒烟测试（必须跑过再写训练代码）

把下面这段保存为 `test_mps.py`，然后 `python test_mps.py`：

```python
"""PyTorch MPS 冒烟测试 - Apple Silicon GPU 加速可用性验证。"""
import time

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available (built): {torch.backends.mps.is_built()}")
print(f"MPS available (runtime): {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    raise SystemExit("MPS 不可用，请确认：Apple Silicon + 原生 ARM64 Python + 较新 PyTorch。")

device = torch.device("mps")
print(f"Recommended max MPS memory: {torch.mps.recommended_max_memory() / 1e9:.2f} GB")

# 1) 基础算子
x = torch.randn(2, 3, device=device)
y = torch.randn(3, 4, device=device)
z = x @ y
torch.mps.synchronize()
print(f"matmul OK, shape={tuple(z.shape)}")

# 2) 自动求导
x = torch.randn(2, 3, device=device, requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print(f"backward OK, grad shape={tuple(x.grad.shape)}")

# 3) 简单 CNN 前向
conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
img = torch.randn(1, 3, 32, 32, device=device)
out = conv(img)
torch.mps.synchronize()
print(f"Conv2d OK, out shape={tuple(out.shape)}")

# 4) 性能参考（非精确基准）
size = 2048
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
torch.mps.synchronize()
t0 = time.time()
c = a @ b
torch.mps.synchronize()
print(f"{size}x{size} matmul: {time.time() - t0:.3f} s")

# 5) 释放缓存
del a, b, c
torch.mps.empty_cache()
print("OK: PyTorch MPS 冒烟测试通过。")
```

**期望输出**：

- `MPS available (runtime): True`
- 末尾打印 `OK: PyTorch MPS 冒烟测试通过。`

如果 `MPS available (runtime): False`，回到 §4 检查 Python 是否真的在 arm64 下运行。

### 9.3 设备无关写法（推荐用于项目代码）

```python
import torch

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()
model = MyModel().to(device)
inputs, targets = inputs.to(device), targets.to(device)
```

---

## 10. 统一内存语义与安全的 OOM 应对

### 10.1 MPS 没有独立显存

Apple Silicon 使用统一内存架构（Unified Memory Architecture, UMA）：CPU 与 GPU 共享同一块物理 RAM。MPS 后端分配的并不是"专用显存"，而是直接占用系统内存。

这带来三个后果：

1. **没有 `nvidia-smi` 等价物**。要看 MPS 当前用了多少内存，调用 `torch.mps.current_allocated_memory()`。
2. **没有独立的 GPU OOM**。MPS 触发的是系统级 OOM，可能直接让整台机器卡住，需要硬重启。
3. **占用是动态的**。PyTorch 的 MPSAllocator 会按需分配，并在空闲时通过 `empty_cache()` 释放回系统。

### 10.2 PyTorch MPS 的内存水线（watermark）

`PYTORCH_MPS_HIGH_WATERMARK_RATIO` 控制硬上限（默认 `1.7`，意为允许使用约 1.7 倍推荐最大工作集；不是 1.7 倍系统内存）。**设成 `0.0` 等价于关闭上限**，是危险的：

> "0.: disables high watermark limit (may cause system failure if system-wide OOM occurs)" — PyTorch 源码注释

### 10.3 安全的 OOM 应对（推荐做法）

按优先级从低风险到高风险：

1. **降 batch size / sequence length**。最安全、最普适。
2. **梯度累积**。等效大 batch，单步显存更低。
3. **混合精度**。在 MPS 上用 `torch.autocast(device_type="mps", dtype=torch.bfloat16)`。
4. **及时 `torch.mps.empty_cache()`**。在每个验证/保存节点调用一次。
5. **显式调低水线**（在能稳定复现 OOM 的工作负载上）：

   ```bash
   # 例：把硬上限收到推荐最大工作集的 80%
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
   ```

   不要设成 `0.0`，除非你能接受系统级卡死的风险。

6. **使用 activation checkpointing**（`torch.utils.checkpoint`），用计算换显存。

### 10.4 OOM 后的恢复

如果 MPS 把系统卡住：

- **不要**继续敲键盘，等 macOS 自动触发 jetsam。
- 备份方案是外接电源 + 让系统进入低功耗；仍无响应时只能长按电源键硬重启。
- 重启后用 `top -l 1 -o rsize | head` 复查大内存进程是否清理干净。

---

## 11. TensorFlow Metal：兼容路径而非默认

Apple 官方提供了 `tensorflow-metal` PluggableDevice，用于在 macOS 上加速 TensorFlow。但这条路在 2026 年的状态是：

- **支持范围有限**：仅 Apple Silicon 与 AMD GPU，没有多 GPU，没有 Intel GPU 加速。
- **更新缓慢**：`tensorflow-metal 1.2.0` 是 2025-01 发布的最后版本，对应 TF 2.18。TF 2.19 之后基本不再跟进。
- **PyTorch MPS 是更主流的选择**：训练侧、社区文档、CI 矩阵都更完善。

**结论**：默认走 PyTorch MPS；只有在必须使用 TensorFlow（如已有 TF 代码或 TFHub 模型）时，才考虑 Metal 路径。

### 11.1 如果确实需要 TensorFlow Metal

根据 Apple Developer 文档（适用于 TF 2.13+）：

```bash
# 推荐：原生 ARM64 Python 3.11 / 3.12（不要用 3.13）
conda create -n tf-metal python=3.12 -y
conda activate tf-metal

# Legacy 锁定到 2.18 + 1.2.0；2.19+ 兼容性无保证。
pip install tensorflow==2.18.0 tensorflow-metal==1.2.0
```

> **强警告**：上面的版本钉死是临时的兼容 workaround。`tensorflow-metal` 在 TF 2.18 之后没有官方 release，不要假设它能跑更新的 TF。

### 11.2 "设备可见" ≠ "实际加速"

很多教程拿下面这段作为"成功"标志：

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(gpus))
```

**这只能证明 PluggableDevice 注册成功**，并不能证明运算真的在 GPU 上跑。验证应该看：

```python
# 真的跑一个小矩阵乘
import tensorflow as tf
import time

with tf.device("/GPU:0"):
    a = tf.random.normal([2048, 2048])
    b = tf.random.normal([2048, 2048])
    t0 = time.time()
    c = a @ b
    print("GPU matmul:", time.time() - t0, "s")
```

如果 CPU 与 GPU 用时差别不大，可能 PluggableDevice 没生效。

---

## 12. 常用深度学习 Python 包

```bash
# 基础科学栈
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter jupyterlab tqdm rich

# 经典深度学习生态
pip install torchvision torchaudio

# 大模型生态
pip install transformers datasets accelerate sentencepiece tiktoken safetensors

# Jupyter
jupyter lab
```

> `tensorflow` 与 `tensorflow-metal` 见 §11。本节不重复。

---

## 13. VS Code：扩展、JSONC 设置、快捷键

### 13.1 必装扩展

| 分类 | 扩展 | 作用 |
| :--- | :--- | :--- |
| Python | `Python`（Microsoft） | LSP、调试、运行 |
| Python | `Pylance` | 类型检查与补全 |
| Python | `Ruff` | 极速 Linter + Formatter |
| Notebook | `Jupyter` | 原生 `.ipynb` 支持 |
| 效率 | `GitLens` | 行级 blame / 历史 |
| 效率 | `Error Lens` | 行末高亮诊断 |
| 效率 | `Todo Tree` | 聚合 TODO / FIXME |
| 主题 | 任选（Tokyo Night / One Dark Pro / Catppuccin） | 个人偏好 |

### 13.2 安装 `code` CLI

`Cmd + Shift + P` → `Shell Command: Install 'code' command in PATH`。之后可以在任何目录 `code .` 打开工程。

### 13.3 `settings.json`（JSONC 格式）

VS Code 的 `settings.json` 实际是 **JSONC（JSON with Comments）**，允许 `//` 和 `/* */` 注释。文件名仍是 `settings.json`，但内容是 JSONC。

打开方式：`Cmd + Shift + P` → `Preferences: Open User Settings (JSON)`。

```jsonc
{
    // 编辑器基础
    "editor.fontSize": 14,
    "editor.fontFamily": "JetBrains Mono, Menlo, Monaco, monospace",
    "editor.fontLigatures": true,
    "editor.tabSize": 4,
    "editor.formatOnSave": true,
    "editor.minimap.enabled": false,
    "editor.cursorBlinking": "smooth",
    "editor.renderWhitespace": "selection",
    "editor.inlineSuggest.enabled": true,

    // Python 专属
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    },
    "python.analysis.typeCheckingMode": "basic",

    // 终端
    "terminal.integrated.fontFamily": "JetBrains Mono",
    "terminal.integrated.defaultProfile.osx": "zsh",

    // 资源面板排除
    "files.exclude": {
        "**/.git": true,
        "**/.DS_Store": true,
        "**/__pycache__": true,
        "**/.ipynb_checkpoints": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true
    }
}
```

> **不要**把这份 JSONC 直接放到跨机器同步的 dotfiles 仓库前，先在本地跑一遍，确认 Ruff / Pylance 都装上。

### 13.4 高频快捷键速查

| 快捷键 | 作用 |
| :--- | :--- |
| `Cmd + P` | 跳文件 |
| `Cmd + Shift + F` | 全局搜索 |
| `Cmd + Shift + P` | 命令面板 |
| `Cmd + D` | 选中下一个同词 |
| `Cmd + Option + ↑/↓` | 增加光标 |
| `Option + ↑/↓` | 行移动 |
| `Shift + Option + ↓` | 复制当前行 |
| `Cmd + \` | 分屏编辑器 |
| `` Ctrl + ` `` | 终端开/关 |

---

## 14. Git 基础：SSH、配置、Aliases、lazygit

### 14.1 全局身份与换行

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
git config --global init.defaultBranch main
git config --global core.autocrlf input   # macOS 下保持 LF
```

### 14.2 SSH 密钥（推荐 ED25519）

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# 评估 -> 回车 -> 输入 passphrase（推荐）

# 把公钥写到剪贴板，粘贴到 GitHub / GitLab 的 SSH 设置
pbcopy < ~/.ssh/id_ed25519.pub

# 验证
ssh -T git@github.com
```

### 14.3 常用 Aliases

```bash
git config --global alias.st "status -s"
git config --global alias.ci "commit -m"
git config --global alias.co "checkout"
git config --global alias.cb "checkout -b"
git config --global alias.br "branch"
git config --global alias.unstage "reset HEAD --"
git config --global alias.last "log -1 HEAD"
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

### 14.4 lazygit（TUI 客户端）

```bash
brew install lazygit
lazygit
```

### 14.5 全局 `.gitignore`

```bash
echo -e ".DS_Store\n._*\n.venv/\n__pycache__/\n.ipynb_checkpoints/\n*.pyc\n.mypy_cache/\n.ruff_cache/" > ~/.gitignore_global
git config --global core.excludesfile ~/.gitignore_global
```

---

## 15. 常见踩坑与排错（Troubleshooting）

### Q1：编译报错 `fatal error: 'stdio.h' file not found`

CLT 没装或路径错乱。

```bash
sudo xcode-select --reset
xcode-select --install
```

### Q2：MPS 报 `MPS backend out of memory`

按 §10.3 的顺序处理。**不要**用 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` "绕过去"，那会让系统级 OOM。

### Q3：`pip install torch` 装到的不是 MPS 版本

```bash
pip install --upgrade --force-reinstall torch
python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

若 `is_built() == False`，多半是装到了 CPU-only 包。检查是否误加了 `--index-url https://download.pytorch.org/whl/cpu`（在 Apple Silicon 上不要这样做）。

### Q4：`brew` 命令找不到

说明 Homebrew 的 PATH 没生效。先 `eval "$(/opt/homebrew/bin/brew shellenv)"`，再把它写进 `~/.zprofile`（见 §6.2）。

### Q5：VS Code 的 `code .` 命令不存在

在 VS Code 中：

1. 按 `Cmd + Shift + P` 打开命令面板
2. 输入并执行：`Shell Command: Install 'code' command in PATH`

之后重启终端即可在任何目录执行 `code .`。

### Q6：TensorFlow Metal 装不上或 import 失败

- 确认 Python 是 3.11 / 3.12（不要 3.13）。
- 确认 TF 版本钉到 2.18，metal 版本钉到 1.2.0（见 §11）。
- 确认 conda 环境下没有别的 `tensorflow` / `tensorflow-macos` 残留：`pip list | grep tensorflow`。

### Q7：端口占用（8888 / 7860 等）

```bash
lsof -i :8888
kill -9 <PID>
```

---

## 16. 自检清单与 FAQ

### 16.1 自检清单

跑完所有步骤后，请逐项确认：

- [ ] `uname -m` 输出 `arm64`
- [ ] `python3 -c "import platform; print(platform.machine())"` 输出 `arm64`
- [ ] `brew --prefix` 输出 `/opt/homebrew`
- [ ] `which brew` 是 `/opt/homebrew/bin/brew`
- [ ] `python test_mps.py` 末尾打印 `OK: PyTorch MPS 冒烟测试通过。`
- [ ] `torch.backends.mps.is_available()` 为 `True`
- [ ] `ssh -T git@github.com` 成功认证
- [ ] VS Code 中 `code .` 可用，且 `settings.json` 是合法 JSONC

### 16.2 FAQ

**Q：M1 8GB / M2 8GB 够用吗？**
A：能跑通 PyTorch MPS 冒烟测试和小模型推理，但训练中型模型（>1B 参数）会非常吃力。建议 16GB 起步，32GB 才能跑得舒服。

**Q：MPS 上能跑 Flash Attention 吗？**
A：截至 PyTorch 2.5+，MPS 后端的 Flash Attention 仍不完整支持。需要 Flash Attention 算子的工作负载建议用 `torch.nn.functional.scaled_dot_product_attention`，它在 MPS 上有专门的 fallback。

**Q：JAX 在 Apple Silicon 上的情况？**
A：JAX 没有 MPS 后端，运行在 CPU 上。不推荐在 macOS 上做 JAX 训练。

**Q：CUDA 代码能在 Mac 上跑吗？**
A：不能。PyTorch 在 macOS 上不提供 CUDA；任何 `device="cuda"` 都会自动 fallback 到 CPU，但会跑得很慢。请把代码改写为设备无关（§9.3）。

**Q：为什么我装好了 Homebrew 但 `python3` 仍是系统版本？**
A：Homebrew 的 Python 不再默认 link。如果你需要 `python3` 命令指向 Homebrew Python，请用 `uv` 或 Miniforge 提供的虚拟环境（§8）。

---

## 17. 版本与变更

### 17.1 历史

| 版本 | 日期 | 主要变更 |
| :--- | :--- | :--- |
| v1.0 | 2025 | 初版，覆盖系统优化 / Homebrew / MPS / TF / VS Code / Git |
| v2.0 | 2026-08-16 | 重写为 Apple Silicon 专用；加入原生性检测、uv vs Miniforge 决策、统一内存与 OOM 安全策略、JSONC 设置示例；删除第三方 Homebrew 镜像；将 TensorFlow Metal 降级为兼容路径；删除不安全的 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` 配方 |

### 17.2 参考资料

- PyTorch MPS 后端：<https://docs.pytorch.org/docs/stable/notes/mps.html>
- PyTorch MPS 环境变量：<https://docs.pytorch.org/docs/stable/mps_environment_variables.html>
- Apple TensorFlow Metal：<https://developer.apple.com/metal/tensorflow-plugin/>
- Homebrew 安装：<https://brew.sh/>
- VS Code JSON 文档：<https://code.visualstudio.com/docs/languages/json>

---

> **版本**: v2.0 (2026-08-16) · **维护**: 自维护
>
> *Apple Silicon 上的深度学习没有捷径 —— 装对原生 ARM64，配对统一内存语义，比任何优化 trick 都更有效。*
---

## 18. Apple Silicon 各代性能基准与选型

> 同为 "Apple Silicon", M1 与 M4 的深度学习性能相差 **3-5 倍**。这一节帮你量化每个型号能跑多大的模型。

### 18.1 各代芯片速查

| 芯片 | CPU 核 | GPU 核 | 内存带宽 | 典型配置 | 深度学习建议 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| M1 | 8 | 7-8 | 68 GB/s | 8/16 GB | 入门, 跑 LLaMA-1B |
| M1 Pro | 10 | 16 | 200 GB/s | 16/32 GB | 中等, 跑 LLaMA-3B-7B |
| M1 Max | 10 | 24-32 | 400 GB/s | 32/64 GB | 跑 LLaMA-7B, 训练 BERT-base |
| M2 | 8 | 10 | 100 GB/s | 8/16 GB | 入门 + |
| M2 Pro | 12 | 19 | 200 GB/s | 16/32 GB | 中等 |
| M2 Max | 12 | 38 | 400 GB/s | 32/96 GB | 较好 |
| M2 Ultra | 24 | 60-76 | 800 GB/s | 64/192 GB | 接近工作站 |
| M3 | 8 | 10 | 100 GB/s | 8/16/24 GB | 入门 + |
| M3 Pro | 12 | 18 | 150 GB/s | 18/36 GB | 中等 |
| M3 Max | 16 | 40 | 300/400 GB/s | 36/48/96 GB | 较好 |
| M4 | 10 | 10 | 120 GB/s | 16/32 GB | 入门 + (新架构) |
| M4 Pro | 14 | 20 | 273 GB/s | 24/48 GB | 中等 |
| M4 Max | 16 | 40 | 546 GB/s | 36/64/128 GB | 旗舰 |
| M4 Ultra | 32 | 80 | 1 TB/s (理论) | 128/256 GB | 工作站级 (推测) |

### 18.2 关键指标: 内存带宽决定 LLM 推理速度

LLM 推理是 **memory-bound** 而不是 compute-bound。生成每个 token 需要从内存读取整个权重矩阵, 因此:

$$\text{tokens/s} \approx \frac{\text{内存带宽}}{\text{模型大小}}$$

**示例** (LLaMA-7B, FP16 ≈ 14 GB):
- M1 (68 GB/s): $\approx 68 / 14 \approx 4.8$ tokens/s
- M1 Max (400 GB/s): $\approx 400 / 14 \approx 28.6$ tokens/s
- M2 Max (400 GB/s): 同 M1 Max
- M4 Max (546 GB/s): $\approx 39$ tokens/s
- 假设 M4 Ultra (1 TB/s): $\approx 71$ tokens/s

> **结论**: 想要跑得快, 优先选 **Max / Ultra** 后缀, 它们的核心升级是内存带宽。

### 18.3 推理吞吐实测参考 (LLaMA.cpp)

实测数据 (来自 llama.cpp 仓库 README 与社区测试, FP16 模型):

| 模型 | M1 8GB | M1 Pro 16GB | M1 Max 32GB | M2 Max 38GB | M4 Max 64GB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LLaMA-3B (Q4_K_M) | 20 | 35 | 50 | 50 | 80 |
| LLaMA-7B (Q4_K_M) | 8 | 18 | 30 | 30 | 50 |
| LLaMA-13B (Q4_K_M) | OOM | 10 | 18 | 18 | 30 |
| LLaMA-30B (Q4_K_M) | OOM | OOM | 8 | 8 | 15 |
| LLaMA-70B (Q4_K_M) | OOM | OOM | OOM | OOM | 6 |

> tokens/s, prompt processing 阶段, 单请求. 来源: 各家用户实测, 仅供参考.

### 18.4 训练吞吐参考 (PyTorch MPS)

| 任务 | M1 8GB | M1 Pro 16GB | M1 Max 32GB |
| :--- | :---: | :---: | :---: |
| ResNet-50 / ImageNet (batch=32) | 5 img/s | 12 img/s | 18 img/s |
| BERT-base fine-tune (seq=128) | 30 seq/s | 80 seq/s | 120 seq/s |
| GPT-2 small (seq=512, train) | 15 seq/s | 40 seq/s | 60 seq/s |
| Stable Diffusion fine-tune | OOM | 1 it/s | 2 it/s |

> 仅为粗略参考. 实际性能依赖 batch size, dtype, 模型配置.

### 18.5 内存与模型上限对照表

| 内存 | 最大 LLM (FP16) | 最大 LLM (Q4 量化) | 推荐用途 |
| :---: | :---: | :---: | :--- |
| 8 GB | 3B | 7B | 学习 / 推理 |
| 16 GB | 7B | 13B | 推理 + 小规模实验 |
| 24 GB | 13B | 30B | 推理 + 中等训练 |
| 32 GB | 13B | 70B | 推理 + 大模型推理 |
| 48 GB | 30B | 70B | 全场景 |
| 64 GB | 30B | 70B (部分 405B) | 准工作站 |
| 128 GB | 70B | 405B | 完整工作站 |

### 18.6 选型决策树

```text
预算 < ¥8000?
├── 是 → M2 / M3 入门款 (二手 / 翻新)
│        - 适合: 学习, 小模型推理, 实验代码
│        - 不适合: 训练 > 1B 模型, 跑 70B+ LLM
│
└── 否 → 看用途
         ├── 写代码 + 偶尔推理 → M4 Pro 24GB
         ├── 跑 7B-13B LLM → M4 Pro / Max 36GB
         ├── 跑 30B-70B LLM → M3/M4 Max 64GB
         ├── 训练中等模型 → M3/M4 Max 48GB+
         └── 工作站级 → M2/M4 Ultra (需等发布)
```

### 18.7 二手 / 翻新款价格参考 (2026 上半年)

| 型号 | 二手价 (¥) | 性价比 |
| :---: | :---: | :---: |
| M1 8GB | 3500-4500 | ⭐⭐⭐ |
| M1 Pro 16GB | 5500-6500 | ⭐⭐⭐⭐ |
| M1 Max 32GB | 7500-9000 | ⭐⭐⭐⭐⭐ |
| M2 16GB | 5000-6000 | ⭐⭐⭐ |
| M2 Max 32GB | 9000-11000 | ⭐⭐⭐⭐⭐ |
| M3 Pro 18GB | 7500-9000 | ⭐⭐⭐⭐ |
| M3 Max 36GB | 13000-15000 | ⭐⭐⭐⭐ |

> **经验**: M1 Max 32GB 是 "甜点机", 二手价 7500 左右, 跑 7B-13B 模型 + 中等训练都很舒服。

### 18.8 内存升级策略 (Apple Silicon)

| 升级前 | 升级后 | 提升 | 必要性 |
| :---: | :---: | :---: | :--- |
| 8 GB | 16 GB | **大**, 可跑 7B | ⭐⭐⭐⭐ |
| 16 GB | 32 GB | **较大, 可跑 13B-30B** | ⭐⭐⭐⭐⭐ |
| 32 GB | 64 GB | **中等, 可跑 70B Q4** | ⭐⭐⭐ |
| 64 GB | 128 GB | **小 (边际递减)** | ⭐⭐ |

> 内存是 **焊死** 在芯片上的, **买的时候决定, 后期无法升级**。

### 18.9 内存够不够的判断公式

```python
def need_memory(model_size_gb, dtype="fp16", training=True, batch_size=1, seq_len=2048):
    """估算训练/推理所需内存"""
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}[dtype]

    # 模型权重
    weights = model_size_gb

    if training:
        # 梯度 = 模型大小
        grads = model_size_gb
        # Adam: m, v, master copy
        adam = model_size_gb * (4 + 4 + 4)  # m + v + master FP32
        optim = adam
        # 激活值 (粗估)
        acts = batch_size * seq_len * 4096 * 32 * 2 / 1e9  # bytes
        total = weights + grads + optim + acts
    else:
        # 推理: 模型 + KV cache
        kv_cache = 2 * 32 * seq_len * 4096 * 2 / 1e9 * batch_size  # bytes
        total = weights + kv_cache

    return total

# 例: LLaMA-7B 训练 (FP16, batch=1, seq=2048)
print(f"LLaMA-7B train: {need_memory(14, 'fp16', True):.1f} GB")
# 14 + 14 + 84 + 2 = 114 GB ❌ (装不下)

# 例: LLaMA-7B 推理 FP16
print(f"LLaMA-7B inference: {need_memory(14, 'fp16', False):.1f} GB")
# ~14.5 GB (KV cache 较小)
```

### 18.10 性能调试工具

```bash
# 1. 实时 CPU/GPU 占用
sudo powermetrics --samplers cpu_power,gpu_power -i 1000

# 2. 内存压力
vm_stat 1

# 3. 进程内存排名
top -l 1 -o rsize | head

# 4. MPS 显存监控 (Python 内)
python -c "
import torch
torch.ones(1024, 1024, device='mps')
print(f'Allocated: {torch.mps.current_allocated_memory()/1e9:.2f} GB')
print(f'Max: {torch.mps.max_memory_allocated()/1e9:.2f} GB')
print(f'Recommended: {torch.mps.recommended_max_memory()/1e9:.2f} GB')
"
```

### 18.11 当统一内存不够时

| 方案 | 优点 | 缺点 |
| :--- | :--- | :--- |
| 4-bit 量化 (GPTQ/AWQ) | 4x 内存节省 | 轻微精度损失 |
| QLoRA 微调 | 在 4-bit 上微调 | 速度慢 |
| 卸载到 SSD (DeepSpeed) | 突破内存限制 | IO 瓶颈 |
| 远程 GPU 服务器 | 不限内存 | 网络延迟 |
| Apple Silicon 外接 eGPU | 旧的 Intel Mac 方案 | 已停止支持 |

### 18.12 选 M Pro 还是 M Max?

| 维度 | M Pro | M Max |
| :--- | :--- | :--- |
| 内存上限 | 36 GB | 128 GB |
| 内存带宽 | 200-273 GB/s | 400-546 GB/s |
| 价格 | 中 | 高 |
| 适合 | 普通编程, 小模型 | LLM 推理, 训练 |
| 性价比 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 18.13 选 M3 还是 M4?

M4 相比 M3:
- **CPU**: IPC 提升 15-25%
- **GPU**: 新架构, ML 性能提升 20-30%
- **NPU**: 大幅升级 (但 PyTorch 暂未直接使用)
- **内存带宽**: M4 Max 提升到 546 GB/s

**结论**: 同样预算, 优先 M4 (新机), M3 性价比 (二手).

### 18.14 Pro 型号 vs Max 型号的实际差异

| 维度 | M3 Pro | M3 Max |
| :---: | :---: | :---: |
| 内存带宽 | 150 GB/s | 300-400 GB/s |
| GPU 核 | 18 | 30-40 |
| 适合 LLM | 7B Q4 | 13B-30B Q4 |
| 价格 | 中 | 高 50% |

### 18.15 不要买的配置

- ❌ **M1/M2 8GB** — 太挤, 跑 7B 都卡
- ❌ **M3 8GB** — 同上
- ❌ **不带 GPU 加速的虚拟机** — Docker Linux 容器无 MPS
- ❌ **Intel Mac** — Metal 不可用, 已不推荐
- ❌ **256GB / 512GB SSD 款** — 模型权重+数据集 + swap 很容易满

### 18.16 推荐的 "起步配置"

| 预算 | 推荐配置 | 可做的事 |
| :---: | :--- | :--- |
| ¥5000 | M1 16GB 二手 | 学习 + 7B 推理 |
| ¥8000 | M2 Pro 16GB | 推理 + 训练 |
| ¥12000 | M3 Pro 18GB | 推理 + 中等训练 |
| ¥18000 | M3 Max 36GB | 全场景 |
| ¥25000+ | M3 Max 64GB / M4 Max | 工作站级 |

### 18.17 实测脚本: 跑分

```python
"""跑分脚本: 在你的 Mac 上跑一次, 数字会出现在 README 顶部"""
import time
import torch
import platform

def benchmark():
    print(f"=== Apple Silicon DL Benchmark ===")
    print(f"Machine: {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")

    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    device = torch.device("mps")
    print(f"MPS recommended max: {torch.mps.recommended_max_memory()/1e9:.1f} GB")

    # 1. MatMul benchmark
    for size in [1024, 2048, 4096]:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.mps.synchronize()

        t0 = time.time()
        for _ in range(5):
            c = a @ b
        torch.mps.synchronize()
        dt = (time.time() - t0) / 5
        tflops = 2 * size**3 / dt / 1e12
        print(f"MatMul {size}³: {dt*1000:.1f} ms ({tflops:.2f} TFLOPS)")
        del a, b, c
        torch.mps.empty_cache()

    # 2. Conv2d benchmark
    for c in [64, 128, 256]:
        x = torch.randn(8, 3, 224, 224, device=device)
        conv = torch.nn.Conv2d(3, c, 3, padding=1).to(device)
        torch.mps.synchronize()

        t0 = time.time()
        for _ in range(10):
            y = conv(x)
        torch.mps.synchronize()
        dt = (time.time() - t0) / 10
        print(f"Conv2d 3→{c} (224x224): {dt*1000:.1f} ms")
        del x, conv, y
        torch.mps.empty_cache()

if __name__ == "__main__":
    benchmark()
```

### 18.18 总结: 一句话选型

**M4 Max 36GB (¥18000)** 是 2026 年的 "甜点机": 跑 30B Q4 推理 + 训练 BERT-base 都很舒服, 内存带宽够, 不需要工作站级预算。

---

## 19. 完整的 ML Pipeline on Apple Silicon

> 从数据加载到模型保存, 一份**真实可跑**的最小 pipeline。覆盖 MPS / CPU / 混合精度 / checkpoint。

### 19.1 目录结构

```text
my-ml-project/
├── README.md
├── pyproject.toml          # uv / poetry 配置
├── .python-version         # 3.11
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── data.py         # 数据加载
│       ├── model.py        # 模型定义
│       ├── train.py        # 训练循环
│       ├── eval.py         # 评估
│       └── utils.py        # 工具函数
├── configs/
│   └── default.yaml        # 超参
├── data/                   # 数据集 (git ignored)
├── runs/                   # 训练输出
└── tests/
    └── test_smoke.py
```

### 19.2 pyproject.toml (uv 风格)

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "numpy>=1.26",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "tqdm>=4.66",
    "rich>=13.7",
    "tensorboard>=2.16",
    "safetensors>=0.4",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5", "mypy>=1.10"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM"]
```

### 19.3 设备无关的工具函数 (utils.py)

```python
"""设备检测 + 设备无关工具"""
import torch
import torch.backends.mps as mps


def pick_device(force: str | None = None) -> torch.device:
    """根据环境自动选择设备, force 可以强制 cuda/mps/cpu"""
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    """统一释放缓存"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        mps.empty_cache()


def synchronize(device: torch.device) -> None:
    """统一同步"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        mps.synchronize()


def get_memory_stats(device: torch.device) -> dict:
    """获取显存 / 统一内存使用"""
    if device.type == "cuda":
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    elif device.type == "mps":
        return {
            "allocated_gb": mps.current_allocated_memory() / 1e9,
            "max_allocated_gb": mps.max_memory_allocated() / 1e9,
            "recommended_max_gb": mps.recommended_max_memory() / 1e9,
        }
    return {}


def log_device_info(device: torch.device) -> None:
    """打印设备信息"""
    print(f"Using device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == "mps":
        print(f"MPS available: {mps.is_available()}")
        print(f"MPS built: {mps.is_built()}")
        print(f"Recommended max: {mps.recommended_max_memory()/1e9:.2f} GB")
    elif device.type == "cuda":
        print(f"CUDA: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name()}")
```

### 19.4 数据加载 (data.py)

```python
"""数据集与 DataLoader, 支持 MPS-friendly 数据格式"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class JsonlDataset(Dataset):
    """JSONL 格式数据集 (一行一个 JSON)"""
    def __init__(self, path: Path, tokenizer=None, max_len: int = 512):
        self.records = []
        with open(path) as f:
            for line in f:
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if self.tokenizer:
            encoded = self.tokenizer(
                rec["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "label": torch.tensor(rec.get("label", 0), dtype=torch.long),
            }
        return rec


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """构建 DataLoader
    MPS 注意: pin_memory 必须 False (不支持), num_workers 不要太高
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,  # MPS 必须 False
        persistent_workers=(num_workers > 0),
    )


# MPS 友好的 num_workers 推荐值
def recommend_num_workers(device: torch.device) -> int:
    if device.type == "mps":
        return 0  # MPS 不支持多 worker pin_memory
    return min(4, os.cpu_count() // 2) if device.type == "cuda" else 0
```

### 19.5 模型定义 (model.py)

```python
"""简单的文本分类模型 (BERT 风格, 但小到 MPS 能跑)"""
import torch
import torch.nn as nn


class SmallTextClassifier(nn.Module):
    """小型文本分类器: Embedding + LSTM + Linear

    参数量约 5M, 在 M1 8GB 上可训练
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # input_ids: (B, L)
        embedded = self.embedding(input_ids)  # (B, L, E)

        # LSTM 输出
        lstm_out, (hidden, _) = self.lstm(embedded)
        # hidden: (num_layers * num_dirs, B, H)

        # 取最后一层 (双向拼接)
        if self.lstm.bidirectional:
            # 最后一层前向 + 反向
            hidden_last = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden_last = hidden[-1]
        # hidden_last: (B, H*2)

        logits = self.classifier(hidden_last)
        return logits


def build_model(config: dict) -> nn.Module:
    return SmallTextClassifier(
        vocab_size=config.get("vocab_size", 30000),
        embed_dim=config.get("embed_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        num_classes=config.get("num_classes", 2),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    )


def count_params(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数, 可训练参数)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

### 19.6 训练循环 (train.py)

```python
"""训练循环, MPS 友好"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

from .utils import pick_device, empty_cache, synchronize, get_memory_stats
from .model import build_model, count_params


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_every: int = 50,
    use_amp: bool = True,
) -> dict:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    t_start = time.time()

    # MPS 的 autocast
    amp_ctx = (
        torch.autocast(device_type="mps", dtype=torch.bfloat16)
        if device.type == "mps" and use_amp
        else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )

    from contextlib import nullcontext

    for step, batch in enumerate(loader):
        # 移到设备
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits = model(input_ids)
            loss = criterion(logits, labels)

        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)

        # 定期 log
        if (step + 1) % log_every == 0:
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples
            elapsed = time.time() - t_start
            mem = get_memory_stats(device)
            print(
                f"[{epoch+1}/{total_epochs}] step {step+1}/{len(loader)} "
                f"loss={avg_loss:.4f} acc={acc:.4f} "
                f"({elapsed:.0f}s) "
                f"mem={mem.get('allocated_gb', 0):.2f}GB"
            )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "elapsed": time.time() - t_start,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """评估"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def main(config: dict):
    # 1. 设备
    device = pick_device(config.get("device"))
    log_device_info(device)

    # 2. 数据
    train_ds = JsonlDataset(Path(config["train_path"]), tokenizer=None)
    val_ds = JsonlDataset(Path(config["val_path"]), tokenizer=None)

    train_loader = build_dataloader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = build_dataloader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # 3. 模型
    model = build_model(config).to(device)
    total, trainable = count_params(model)
    print(f"Model: {total/1e6:.2f}M params ({trainable/1e6:.2f}M trainable)")

    # 4. 优化器 + loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 0.01),
    )
    criterion = nn.CrossEntropyLoss()

    # 5. 训练
    best_acc = 0.0
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config["epochs"],
            log_every=config.get("log_every", 50),
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"time={train_metrics['elapsed']:.0f}s")

        # 保存 best
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "best_acc": best_acc,
            }, save_dir / "best.ckpt")
            print(f"Saved best model (acc={best_acc:.4f})")

        # 释放缓存
        empty_cache(device)

    print(f"\nTraining done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    config = {
        "train_path": "data/train.jsonl",
        "val_path": "data/val.jsonl",
        "save_dir": "runs/exp1",
        "vocab_size": 30000,
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_classes": 2,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "epochs": 5,
        "log_every": 50,
    }
    main(config)
```

### 19.7 配置文件 (configs/default.yaml)

```yaml
# configs/default.yaml
experiment_name: "text-cls-v1"
seed: 42

model:
  vocab_size: 30000
  embed_dim: 256
  hidden_dim: 512
  num_layers: 2
  num_classes: 2
  dropout: 0.3

training:
  batch_size: 32
  lr: 0.001
  weight_decay: 0.01
  epochs: 10
  log_every: 50
  gradient_clip: 1.0
  mixed_precision: true

data:
  train_path: data/train.jsonl
  val_path: data/val.jsonl
  max_len: 256
  num_workers: 0   # MPS 不要 > 0

runtime:
  device: auto     # auto / cuda / mps / cpu
  save_dir: runs/${experiment_name}
```

### 19.8 用 uv 运行

```bash
# 安装依赖
uv sync

# 跑训练
uv run python -m myproject.train --config configs/default.yaml

# 跑测试
uv run pytest tests/

# 格式化
uv run ruff check --fix
uv run ruff format
```

### 19.9 Smoke Test (tests/test_smoke.py)

```python
"""冒烟测试: 验证整个 pipeline 在 MPS 上能跑"""
import torch
import torch.nn as nn
import pytest

from myproject.utils import pick_device
from myproject.model import build_model


@pytest.mark.parametrize("device_str", ["cpu", "mps", "cuda"])
def test_pipeline_smoke(device_str):
    """在指定设备上跑一次 forward + backward"""
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device_str)
    model = build_model({"vocab_size": 100, "num_classes": 2}).to(device)

    # 假数据
    input_ids = torch.randint(0, 100, (4, 32), device=device)
    labels = torch.randint(0, 2, (4,), device=device)

    # Forward
    logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits, labels)

    # Backward
    loss.backward()

    # 检查梯度
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"

    print(f"{device_str}: OK")


def test_model_save_load():
    """检查 save / load 闭环"""
    device = pick_device()
    model = build_model({"vocab_size": 100, "num_classes": 2})

    # Save
    torch.save(model.state_dict(), "/tmp/test_model.pth")

    # Load
    loaded = build_model({"vocab_size": 100, "num_classes": 2})
    loaded.load_state_dict(torch.load("/tmp/test_model.pth", weights_only=True))

    # 检查参数一致
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)
```

### 19.10 训练监控 (TensorBoard)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp1")

for epoch in range(epochs):
    train_metrics = train_one_epoch(...)
    val_metrics = evaluate(...)

    # 写入
    writer.add_scalar("train/loss", train_metrics["loss"], epoch)
    writer.add_scalar("train/acc", train_metrics["accuracy"], epoch)
    writer.add_scalar("val/loss", val_metrics["loss"], epoch)
    writer.add_scalar("val/acc", val_metrics["accuracy"], epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

writer.close()

# 查看:
# tensorboard --logdir=runs --port=6006
```

### 19.11 常见训练问题速查

| 问题 | 解决 |
| :--- | :--- |
| MPS loss=NaN | 降学习率, 检查数据, 关 AMP |
| MPS 内存爆炸 | 降 batch, empty_cache(), 关 AMP |
| Loss 不收敛 | 升 LR / 检查标签 / 检查数据 |
| 训练比 CPU 慢 | 验证 MPS 真在用 (torch.mps.synchronize + 计时) |
| MPS backward 失败 | 算子可能不支持, 改 CPU fallback |
| num_workers > 0 卡死 | MPS 不支持, 设为 0 |
| pin_memory=True 报错 | MPS 不支持, 设为 False |

### 19.12 Resume 训练 (从 checkpoint 继续)

```python
def resume(checkpoint_path: str, model, optimizer, scheduler=None):
    """从 checkpoint 恢复"""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", 0.0)

    print(f"Resumed from epoch {ckpt.get('epoch', 0)}, best={best_metric:.4f}")
    return start_epoch, best_metric
```

### 19.13 多文件 dataset (大文件)

```python
"""大数据集用 numpy memmap 或 webdataset"""
import numpy as np
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    """用 numpy memmap 读大文件, 不一次性加载到内存"""
    def __init__(self, data_path: str, label_path: str):
        self.data = np.load(data_path, mmap_mode="r")  # (N, ...)
        self.labels = np.load(label_path, mmap_mode="r")  # (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": torch.from_numpy(self.data[idx].copy()),  # 必须 copy
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

### 19.14 数据增强 (CPU vs GPU)

```python
"""尽量把 augmentation 放 CPU (DataLoader worker) 或 GPU (collate_fn)"""
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# GPU augmentation (kornia 库)
import kornia.augmentation as K
gpu_aug = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(0.2, 0.2, 0.2),
)
```

### 19.15 一键训练脚本

```bash
#!/bin/bash
# scripts/train.sh
set -e
source .venv/bin/activate

uv run python -m myproject.train \
    --config configs/default.yaml \
    --experiment-name "exp_$(date +%Y%m%d_%H%M%S)" \
    --seed 42

echo "✅ Training done."
```

---

## 20. HuggingFace Transformers on MPS

> 大模型时代的"开发环境"——MPS 上跑 HuggingFace 的全套工具链。

### 20.1 安装

```bash
pip install transformers datasets accelerate sentencepiece tiktoken safetensors
```

### 20.2 加载 BERT (分类微调)

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from datasets import load_dataset

# 1. 数据
dataset = load_dataset("imdb")  # 或者自己的 dataset

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True)

# 3. 模型 (注意: MPS 上要 .to("mps"))
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
).to("mps")

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="runs/bert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    use_mps_device=True,  # 关键!
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="tensorboard",
    dataloader_num_workers=0,  # MPS 不支持
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()
```

### 20.3 `use_mps_device=True` 是关键

`TrainingArguments` 自动检测设备, 但 MPS 路径需手动开启:

```python
# PyTorch < 2.4
training_args = TrainingArguments(
    ...,
    use_mps_device=True,  # 显式告诉 Trainer
)

# PyTorch >= 2.4 自动检测 MPS, 通常不需要这行
```

### 20.4 跑 GPT-2 文本生成

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to("mps")

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### 20.5 4-bit 量化推理 (BitsAndBytes)

```bash
pip install bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ⚠️ 注意: bitsandbytes 在 MPS 上**不支持 4-bit 量化**!
# 这是个常见错误, 会在 MPS 上运行时崩溃

# 正确做法: 用 FP16 / BF16, 或量化到 CPU 后加载
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 这个只在 CUDA 上有效, MPS 上会报错
# model = AutoModelForCausalLM.from_pretrained("...", quantization_config=nf4_config)

# MPS 推荐做法: 直接 FP16
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16,
).to("mps")
```

### 20.6 Llama-3 / Qwen 在 MPS 上的性能

| 模型 | M1 16GB | M2 Max 32GB | M4 Max 64GB |
| :---: | :---: | :---: | :---: |
| Qwen2-0.5B | 60 t/s | 120 t/s | 200 t/s |
| Qwen2-1.5B | 30 t/s | 80 t/s | 150 t/s |
| Qwen2-7B (FP16) | OOM | 25 t/s | 50 t/s |
| LLaMA-3-8B (FP16) | OOM | 20 t/s | 40 t/s |
| LLaMA-3-70B (FP16) | OOM | OOM | OOM |

> 80GB+ 模型需用量化或 llama.cpp (见 §21)。

### 20.7 Accelerate 库加速

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 自动选择设备 (mps / cuda / cpu)
# 与 Trainer 类似, 但更灵活
```

### 20.8 跑 Stable Diffusion (扩散模型)

```bash
pip install diffusers transformers accelerate
```

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("mps")

# ⚠️ 注意: MPS 上 UNet 推理较慢, 但能用
image = pipe("a cat sitting on a laptop").images[0]
image.save("cat.png")

# MPS 性能优化
pipe.enable_attention_slicing()       # 省显存
pipe.enable_vae_tiling()              # 大图分块
```

### 20.9 HuggingFace 数据集

```python
from datasets import load_dataset, Dataset

# 1. 加载 HF 数据集
ds = load_dataset("imdb", split="train[:1000]")

# 2. 从本地文件加载
ds = Dataset.from_json("data.jsonl")
ds = Dataset.from_csv("data.csv")
ds = Dataset.from_parquet("data.parquet")

# 3. 从 dict 创建
ds = Dataset.from_dict({
    "text": ["hello", "world"],
    "label": [0, 1],
})

# 4. 数据处理
ds = ds.map(lambda x: {"len": len(x["text"])}, batched=False)
ds = ds.filter(lambda x: x["len"] > 5)
ds = ds.shuffle(seed=42)
ds = ds.select(range(100))  # 取前 100 条
ds = ds.train_test_split(test_size=0.2)
```

### 20.10 Tokenizer 注意事项

```python
from transformers import AutoTokenizer

# 1. 加载
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. 必要设置
if tok.pad_token is None:
    tok.pad_token = tok.eos_token  # 或 tok.unk_token

# 3. 编码
encoded = tok(
    "hello world",
    padding="max_length",     # 或 "longest"
    truncation=True,
    max_length=128,
    return_tensors="pt",      # "pt" / "tf" / "np"
)

# 4. 批编码
encoded = tok(
    ["hello", "world"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)
```

### 20.11 离线模式 (HuggingFace)

```bash
# 1. 一次性下载到本地
export HF_HOME=~/hf_cache
huggingface-cli download bert-base-uncased --cache-dir ~/hf_cache

# 2. 完全离线使用
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

```python
# 代码侧设置
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased", cache_dir="~/hf_cache")
```

### 20.12 Pipeline API (高层)

```python
from transformers import pipeline

# 文本分类
classifier = pipeline("sentiment-analysis", device="mps")
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 命名实体识别
ner = pipeline("ner", device="mps", aggregation_strategy="simple")
print(ner("Apple is a company in California."))

# 问答
qa = pipeline("question-answering", device="mps")
result = qa(question="What's the capital?", context="Paris is the capital of France.")
print(result)

# 文本生成
gen = pipeline("text-generation", model="gpt2", device="mps")
print(gen("Once upon a time", max_new_tokens=30))

# 翻译 (需要多语言模型)
# translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device="mps")
# print(translator("Hello world", max_length=40))
```

### 20.13 LoRA 微调 (PEFT)

```bash
pip install peft
```

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16,
).to("mps")

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练循环与 §19.6 类似
```

### 20.14 模型对比速查

| 模型 | 显存 (FP16) | MPS 推理 | MPS 训练 |
| :--- | :---: | :---: | :---: |
| BERT-base | 0.4 GB | ✅ | ✅ |
| BERT-large | 1.3 GB | ✅ | ✅ |
| GPT-2 small | 0.5 GB | ✅ | ✅ |
| GPT-2 medium | 1.4 GB | ✅ | ⚠️ |
| RoBERTa-base | 0.5 GB | ✅ | ✅ |
| T5-base | 1.0 GB | ✅ | ✅ |
| T5-large | 3.1 GB | ✅ | ⚠️ |
| LLaMA-3-8B | 16 GB | ⚠️ (32GB+) | ❌ |
| LLaMA-3-70B | 140 GB | ❌ | ❌ |
| Qwen2-1.5B | 3 GB | ✅ | ✅ |
| Qwen2-7B | 14 GB | ⚠️ (32GB+) | ❌ |
| Stable Diffusion | 4 GB | ✅ (慢) | ❌ |

### 20.15 MPS 上不能做的事

| 操作 | 替代方案 |
| :--- | :--- |
| Flash Attention | PyTorch 2.5+ 部分支持, 否则 SDPA fallback |
| 4-bit 量化 (bitsandbytes) | 用 FP16, 或 §21 的 llama.cpp |
| 多 GPU | 用远程服务器 |
| FSDP / DeepSpeed | 单卡模型在 MPS 上一般装得下 |
| 自定义 CUDA kernel | 写 Metal kernel (门槛极高) |

### 20.16 在 Mac 上下载大模型的注意事项

```bash
# 默认 HF 缓存: ~/.cache/huggingface
export HF_HOME=~/hf_cache  # 自定义位置
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 大模型超时

# 国内加速镜像
export HF_ENDPOINT=https://hf-mirror.com

# 用 huggingface-cli
huggingface-cli login  # 输入 token
huggingface-cli download meta-llama/Llama-2-7b-hf --token hf_xxx
```

---

## 21. 本地 LLM 推理 (llama.cpp / Ollama / LM Studio)

> 想在 Mac 上**跑 70B 模型**, 必须用 GGUF 量化 + llama.cpp / Ollama。本章从零搭一套本地 LLM 服务。

### 21.1 三种工具对比

| 工具 | 形式 | 优势 | 适合 |
| :--- | :--- | :--- | :--- |
| **llama.cpp** | CLI + Python lib | 极致轻量, Metal 原生加速, 量化最全 | 工程师 / 嵌入应用 |
| **Ollama** | CLI + 服务 | 一行命令跑模型, REST API | 快速试用 / 服务化 |
| **LM Studio** | GUI | 图形界面, 搜索模型库 | 非技术用户 |

### 21.2 llama.cpp 安装

```bash
# 方式 1: Homebrew (推荐)
brew install llama.cpp

# 方式 2: 源码编译 (获取最新 + 自定义)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j

# 验证
llama-cli --version
```

### 21.3 下载 GGUF 模型

```bash
# 1. HuggingFace 直接下载 (CLI)
huggingface-cli download \
    TheBloke/Llama-2-7B-Chat-GGUF \
    llama-2-7b-chat.Q4_K_M.gguf \
    --local-dir ~/models \
    --local-dir-use-symlinks False

# 2. 用 wget / curl
wget -P ~/models https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# 3. 国内镜像
wget -P ~/models https://hf-mirror.com/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

### 21.4 GGUF 量化等级对照

| 等级 | 大小 (LLaMA-7B) | 质量 | 推荐 |
| :---: | :---: | :--- | :--- |
| F16 | 13.5 GB | 100% | 满显存 |
| Q8_0 | 7.2 GB | 99% | 16GB+ |
| Q6_K | 5.7 GB | 98% | 16GB |
| **Q5_K_M** | 5.1 GB | 97% | **甜点** |
| **Q4_K_M** | 4.1 GB | 95% | **推荐** |
| Q3_K_M | 3.3 GB | 92% | 极限 |
| Q2_K | 2.7 GB | 85% | 不推荐 |

### 21.5 llama-cli: 命令行推理

```bash
# 交互模式
llama-cli \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --interactive \
    --color \
    --ctx-size 4096 \
    --n-gpu-layers 99   # 全部层放 GPU (MPS)

# 单次 prompt
llama-cli \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "Explain quantum computing in 3 sentences." \
    --n-predict 200

# 服务模式 (HTTP)
llama-server \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080
```

### 21.6 llama-cpp-python: Python 嵌入

```bash
# 安装 (Metal 加速)
CMAKE_ARGS="-DLLAMA_METAL=on" \
    pip install llama-cpp-python
```

```python
from llama_cpp import Llama

# 1. 加载模型
llm = Llama(
    model_path="~/models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,         # 上下文长度
    n_threads=8,        # CPU 线程数
    n_gpu_layers=99,    # GPU 层数 (Metal 加速)
    chat_format="llama-2",  # 或 "mistral", "qwen"
)

# 2. 简单补全
output = llm(
    "Once upon a time",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
    stop=["\n\n"],
)
print(output["choices"][0]["text"])

# 3. 聊天模式 (自动应用 chat template)
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(output["choices"][0]["message"]["content"])
```

### 21.7 OpenAI 兼容 API (llama-server)

```bash
# 启动服务
llama-server \
    --model ~/models/llama-2-7b-chat.Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 4096
```

```python
# 用 OpenAI 客户端访问 (完全兼容)
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",  # llama-server 的 OpenAI 兼容 endpoint
    api_key="not-needed",  # llama-server 不验证
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)
print(response.choices[0].message.content)
```

### 21.8 Ollama: 一行命令跑模型

```bash
# 安装
brew install ollama

# 启动服务
brew services start ollama
# 或前台启动: ollama serve

# 拉模型
ollama pull llama3
ollama pull qwen2:7b
ollama pull mistral
ollama pull codellama:13b
ollama pull llama3:70b  # 大模型

# 列出本地模型
ollama list

# 运行 (交互)
ollama run llama3

# 一次性 prompt
ollama run llama3 "Explain gravity in one sentence."

# 删除
ollama rm llama3
```

### 21.9 Ollama 自定义模型 (Modelfile)

```bash
# 创建 Modelfile
cat > ~/Modelfile << 'EOF'
FROM llama3
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM You are a helpful AI assistant specialized in machine learning.
EOF

# 创建
ollama create my-llama -f ~/Modelfile

# 运行
ollama run my-llama
```

### 21.10 Ollama REST API

```bash
# 生成
curl -X POST http://localhost:11434/api/generate \
    -d '{"model": "llama3", "prompt": "Why is the sky blue?", "stream": false}'

# 聊天
curl -X POST http://localhost:11434/api/chat \
    -d '{
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false
    }'

# 列出模型
curl http://localhost:11434/api/tags

# OpenAI 兼容
curl -X POST http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello"}]
    }'
```

```python
# Python 调用 Ollama
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": "Hi!", "stream": False},
)
print(response.json()["response"])

# 流式
import json
with requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": "Hi!", "stream": True},
    stream=True,
) as r:
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk["response"], end="", flush=True)
```

### 21.11 Ollama vs llama.cpp 性能

| 维度 | Ollama | llama.cpp |
| :---: | :---: | :---: |
| Metal 加速 | ✅ | ✅ |
| 启动时间 | 慢 (服务) | 快 (CLI) |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| API 兼容 | OpenAI | OpenAI + 自己的 |
| 模型管理 | 内置 | 手动 |
| 自定义模型 | Modelfile | 命令行参数 |
| 适合 | 试用 / 服务 | 嵌入应用 / 调优 |

### 21.12 量化自己的模型 (Python)

```bash
pip install gguf  # 或 llama-cpp-python
```

```python
# 用 llama.cpp 的 convert 脚本
# (从 HF 转 GGUF)
cd llama.cpp
python convert_hf_to_gguf.py \
    ~/models/llama-2-7b-hf \
    --outfile ~/models/llama-2-7b-f16.gguf \
    --outtype f16

# 量化
./build/bin/llama-quantize \
    ~/models/llama-2-7b-f16.gguf \
    ~/models/llama-2-7b-q4km.gguf \
    Q4_K_M
```

### 21.13 模型选择建议

| 用途 | 推荐 | 大小 |
| :---: | :---: | :---: |
| 学习 / 实验 | Qwen2-1.5B / Phi-3-mini | 1-3 GB |
| 通用对话 | Qwen2-7B / Llama-3-8B | 4-7 GB |
| 代码 | CodeLlama-7B / Qwen2.5-Coder-7B | 4-7 GB |
| 中文增强 | Qwen2-7B / Yi-1.5-9B | 5-8 GB |
| 大模型推理 | Llama-3-70B / Qwen2-72B | 30-50 GB |

### 21.14 推荐的中文模型 (Mac 友好)

```bash
# Qwen 系列 (阿里, 中文强)
ollama pull qwen2:7b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
ollama pull qwen2.5-coder:7b  # 代码

# Yi 系列 (零一万物)
ollama pull yi:6b
ollama pull yi:9b

# GLM 系列 (智谱)
ollama pull glm4:9b
ollama pull chatglm3:6b

# DeepSeek 系列
ollama pull deepseek-coder:6.7b
ollama pull deepseek-llm:7b
```

### 21.15 内存需求速查 (GGUF Q4_K_M)

| 模型 | 模型大小 | 推理最低 | 推理推荐 | 上下文 4K | 上下文 32K |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1.5B | 1.0 GB | 4 GB | 8 GB | +0.5 GB | +4 GB |
| 3B | 2.0 GB | 6 GB | 8 GB | +0.5 GB | +4 GB |
| 7B | 4.1 GB | 8 GB | 16 GB | +1 GB | +8 GB |
| 13B | 7.4 GB | 16 GB | 24 GB | +1.5 GB | +12 GB |
| 30B | 17 GB | 32 GB | 48 GB | +3 GB | +24 GB |
| 70B | 40 GB | 64 GB | 96 GB | +7 GB | +56 GB |

### 21.16 GGUF 推理常见问题

| 问题 | 解决 |
| :---: | :--- |
| OOM | 换更小量化 (Q4 → Q3) |
| 推理慢 | 关掉不必要进程, 升级到 Max/Ultra |
| 输出乱码 | 检查 chat_format, prompt template |
| 中文乱码 | 模型没中文训练数据, 换中文模型 |
| Metal 不生效 | `n_gpu_layers=99`, 确认 Apple Silicon |
| 卡 token 上限 | 加大 `n_ctx` |

### 21.17 性能优化技巧

```bash
# 1. 全 GPU 卸载 (Metal)
llama-cli --model ... --n-gpu-layers 99

# 2. 增加 batch size
llama-cli --model ... --batch-size 512

# 3. flash attention (llama.cpp 自带)
llama-cli --model ... --flash-attn

# 4. 量化 cache
llama-cli --model ... --cache-type-k q4_0  # K cache 量化
llama-cli --model ... --cache-type-v q4_0  # V cache 量化

# 5. mmap
llama-cli --model ... --mlock  # 锁内存 (防止 swap)
```

### 21.18 LM Studio: GUI 用户

```bash
# 下载
brew install --cask lm-studio

# 启动
open -a "LM Studio"

# 功能:
# - 搜索 HuggingFace 模型
# - 下载 GGUF
# - 内置聊天 UI
# - OpenAI 兼容本地 server
```

### 21.19 嵌入到自己的 Python 项目

```python
# 完整 demo: 用 llama-cpp-python 做 RAG
from llama_cpp import Llama
from typing import List

class LocalRAG:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=99,
        )

    def generate(self, question: str, context: List[str], max_tokens: int = 500) -> str:
        # 拼接 context
        context_text = "\n\n".join(context)
        prompt = f"""Use the following context to answer the question.

Context:
{context_text}

Question: {question}

Answer:"""

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stop=["\n\nQuestion:", "\nQuestion:"],
        )
        return output["choices"][0]["text"].strip()

    def stream_generate(self, question: str, context: List[str]):
        """流式生成"""
        prompt = f"Context: {' '.join(context)}\n\nQ: {question}\nA:"
        for chunk in self.llm(
            prompt,
            max_tokens=500,
            stream=True,
            temperature=0.3,
        ):
            text = chunk["choices"][0]["text"]
            yield text


# 用法
rag = LocalRAG("~/models/llama-2-7b-chat.Q4_K_M.gguf")
answer = rag.generate("What is Python?", ["Python is a programming language."])
print(answer)
```

### 21.20 与 Claude/OpenAI 对比

| 维度 | 本地 LLM | Claude API | OpenAI API |
| :---: | :---: | :---: | :---: |
| 速度 | 中 (10-50 t/s) | 快 | 快 |
| 成本 | 一次性 (电费) | 订阅 / 按 token | 订阅 / 按 token |
| 隐私 | ✅ 完全本地 | ❌ 上传 | ❌ 上传 |
| 离线 | ✅ | ❌ | ❌ |
| 质量 | 中-高 (取决于模型) | 高 | 高 |
| 适合 | 隐私 / 离线 / 嵌入 | 通用 | 通用 |

### 21.21 在 Cursor / VS Code 中用本地 LLM

```json
// VS Code settings.json - Continue.dev 扩展
{
  "continue.llmProvider": "ollama",
  "continue.ollamaModel": "qwen2.5-coder:7b",
  "continue.ollamaBaseUrl": "http://localhost:11434"
}
```

### 21.22 模型热更新 (不停机切换)

```bash
# Ollama 卸载旧模型
ollama rm llama3:8b
ollama pull llama3:8b-instruct-q5_K_M

# 不用重启 Ollama 服务
```

### 21.23 与 ChatBox / ChatGPT-Next-Web 集成

```bash
# 启动 Ollama
ollama serve

# 在 ChatBox 中填入
# API URL: http://localhost:11434/v1
# Model: qwen2:7b
```

### 21.24 LLM 微调 on Mac (LoRA)

```bash
# 工具: mlx-lm (Apple 官方 MLX 框架)
pip install mlx-lm
```

```python
# MLX 微调示例
from mlx_lm import load, generate
from mlx_lm.utils import load_dataset
from mlx_lm.tuner import train

model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
train(
    model=model,
    tokenizer=tokenizer,
    data=load_dataset("data.jsonl"),
    batch_size=4,
    iters=1000,
    learning_rate=2e-5,
)
```

### 21.25 MLX 框架 (Apple Silicon 原生)

```bash
pip install mlx mlx-lm
```

```python
import mlx.core as mx
import mlx.nn as nn

# MLX 是 Apple 出的, 比 MPS 更原生
a = mx.random.normal((1024, 1024))
b = mx.random.normal((1024, 1024))
c = a @ b  # 懒求值
mx.eval(c)  # 触发实际计算
```

> MLX 直接生成 Metal shader, 比 PyTorch MPS 更接近硬件。

---

## 22. ML 工具链 (WandB / TensorBoard / MLflow)

> 训练监控 / 实验追踪 / 模型版本管理, 是 ML 工程师的"必备技能"。

### 22.1 TensorBoard (PyTorch 原生)

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp1")

# 1. 标量
for step in range(100):
    writer.add_scalar("train/loss", loss.item(), step)
    writer.add_scalar("train/acc", acc, step)
    writer.add_scalar("lr", lr, step)

# 2. 直方图
for step in range(100):
    writer.add_histogram("layer1/weights", model.layer1.weight, step)
    writer.add_histogram("layer1/grads", model.layer1.weight.grad, step)

# 3. 图像
writer.add_image("input", img_grid, 0)

# 4. 模型图
writer.add_graph(model, input_to_model=torch.randn(1, 3, 224, 224))

# 5. Embedding
writer.add_embedding(features, metadata=labels)

writer.close()
```

```bash
# 启动 TensorBoard
tensorboard --logdir=runs --port=6006

# 远程访问 (SSH 隧道)
ssh -L 6006:localhost:6006 user@server
# 然后浏览器打开 http://localhost:6006
```

### 22.2 WandB (推荐, 远程追踪)

```bash
pip install wandb
wandb login  # 输入 API key
```

```python
import wandb

# 1. 初始化
wandb.init(
    project="my-project",
    name="exp-v1",
    config={
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 10,
        "model": "SmallTextClassifier",
    },
)

# 2. 记录
for epoch in range(epochs):
    for step, batch in enumerate(loader):
        loss = ...
        wandb.log({"train/loss": loss, "epoch": epoch, "step": step})

    # 验证
    val_metrics = evaluate(model, val_loader)
    wandb.log({"val/loss": val_metrics["loss"], "epoch": epoch})

    # 直方图
    wandb.log({"layer1/weights": wandb.Histogram(model.layer1.weight.detach().cpu().numpy())})

    # 图像
    wandb.log({"samples": wandb.Image(image_grid)})

# 3. 保存模型 artifact
artifact = wandb.Artifact("model", type="model")
artifact.add_file("best.ckpt")
wandb.log_artifact(artifact)

# 4. 结束
wandb.finish()
```

### 22.3 WandB 离线模式

```bash
export WANDB_MODE=offline  # 不上传
export WANDB_DIR=~/wandb_logs
wandb sync ~/wandb_logs/latest-run  # 后续手动同步
```

### 22.4 MLflow (开源, 自托管)

```bash
pip install mlflow
```

```python
import mlflow
import mlflow.pytorch

mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    # 1. Log 参数
    mlflow.log_param("lr", 1e-3)
    mlflow.log_param("batch_size", 32)
    mlflow.log_params({"dropout": 0.3, "weight_decay": 0.01})

    # 2. Log 指标
    for epoch in range(epochs):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # 3. Log 模型
    mlflow.pytorch.log_model(model, "model")

    # 4. Log artifact
    mlflow.log_artifact("config.yaml")
```

```bash
# 启动 MLflow UI
mlflow ui --port=5000
# 浏览器打开 http://localhost:5000
```

### 22.5 三个工具对比

| 维度 | TensorBoard | WandB | MLflow |
| :---: | :---: | :---: | :---: |
| 安装 | 一行 pip | 一行 pip | 一行 pip |
| 远程同步 | ❌ 本地 | ✅ 云端 | ✅ 自托管 |
| 实验对比 | 一般 | 强 | 强 |
| 模型版本 | ❌ | ✅ Artifact | ✅ Registry |
| 协作 | 弱 | 强 | 中 |
| 离线 | ✅ | ✅ (offline mode) | ✅ |
| 适合 | 单机本地 | 团队 / 云端 | 企业自托管 |

### 22.6 实验命名规范

```python
# 推荐格式: <model>_<dataset>_<date>_<tag>
exp_name = "bert-base_imdb_20260818_v1"

# 自动生成时间戳
from datetime import datetime
exp_name = f"bert-base_imdb_{datetime.now():%Y%m%d_%H%M%S}"
```

### 22.7 超参记录 (Hydra / OmegaConf)

```bash
pip install hydra-core
```

```yaml
# configs/config.yaml
defaults:
  - model: bert_base
  - data: imdb
  - train: default

train:
  batch_size: 32
  lr: 0.001
  epochs: 10

hydra:
  output_subdir: null  # 不创建子目录
```

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg.train.lr)
    # 自动保存到 .hydra/config.yaml
```

### 22.8 模型注册表

```python
# MLflow Model Registry
import mlflow.pytorch

mlflow.set_registry_uri("sqlite:///mlflow.db")

# 注册模型
with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model", registered_model_name="text-cls-prod")

# 加载
model_uri = "models:/text-cls-prod/latest"
model = mlflow.pytorch.load_model(model_uri)
```

### 22.9 多实验对比 (WandB Sweep)

```python
# sweep.yaml
program: train.py
method: bayes
metric:
  name: val/accuracy
  goal: maximize
parameters:
  lr:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [16, 32, 64]
  dropout:
    min: 0.1
    max: 0.5
```

```bash
wandb sweep sweep.yaml
# 输出 sweep ID
wandb agent <SWEEP_ID>
```

### 22.10 与 MLflow / WandB 集成的最佳实践

```python
# 同时记录到多个后端 (WandB + TensorBoard)
from torch.utils.tensorboard import SummaryWriter
import wandb

writer = SummaryWriter("runs/exp1")
wandb.init(project="my-project")

for step in range(100):
    # 计算指标
    loss = ...

    # 写入 TensorBoard
    writer.add_scalar("train/loss", loss, step)

    # 写入 WandB
    wandb.log({"train/loss": loss, "step": step})

writer.close()
wandb.finish()
```

---

## 23. 数据存储与 I/O 优化

> 数据是 ML 项目的"燃料"。Mac 上的 I/O 优化对训练吞吐至关重要。

### 23.1 推荐的数据格式

| 格式 | 优点 | 缺点 | 适合 |
| :---: | :--- | :--- | :--- |
| CSV | 人可读, 通用 | 慢, 大文件 | 小数据 |
| JSONL | 灵活, 流式 | 大 | 文本 |
| **Parquet** | 列存, 压缩, 快 | 不人读 | **大表格** |
| HDF5 | 层次化, 快 | 单写 | 数值 |
| Arrow / Feather | 内存高效 | 大 | DataFrame |
| TFRecord | TF 专用 | TF 锁定 | 训练 |
| WebDataset | tar 流式 | 复杂 | 大图像 |

### 23.2 Parquet 实战

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 1. 写
df = pd.DataFrame({
    "text": ["hello", "world"],
    "label": [0, 1],
})
table = pa.Table.from_pandas(df)
pq.write_table(table, "data.parquet", compression="snappy")

# 2. 读 (只读部分列, 内存高效!)
df = pd.read_parquet("data.parquet", columns=["text", "label"])

# 3. 读 + filter (下推)
df = pd.read_parquet(
    "data.parquet",
    filters=[("label", "=", 1)],
)

# 4. 流式读
pf = pq.ParquetFile("huge.parquet")
for batch in pf.iter_batches(batch_size=10000):
    df = batch.to_pandas()
    process(df)
```

### 23.3 Arrow (内存高效 DataFrame)

```python
import pyarrow as pa
import pyarrow.compute as pc

# 1. 构造
table = pa.table({
    "col1": [1, 2, 3],
    "col2": ["a", "b", "c"],
})

# 2. 零拷贝转换
df = table.to_pandas()  # 与 Arrow 共享内存

# 3. 计算 (向量化)
table_filtered = table.filter(pc.greater(table["col1"], 1))

# 4. 写文件
pa.ipc.write_feather(table, "data.feather")
df_arrow = pa.ipc.read_feather("data.feather").to_pandas()
```

### 23.4 HDF5 (大数值数据)

```python
import h5py
import numpy as np

# 1. 写
with h5py.File("data.h5", "w") as f:
    f.create_dataset("images", data=np.random.randn(10000, 3, 224, 224),
                     compression="gzip", compression_opts=9)
    f.create_dataset("labels", data=np.random.randint(0, 10, 10000))

# 2. 读 (可只读部分)
with h5py.File("data.h5", "r") as f:
    # 只读前 100 张
    images = f["images"][:100]
    labels = f["labels"][:100]

# 3. PyTorch Dataset
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.file = h5py.File(path, "r")
        self.images = self.file["images"]
        self.labels = self.file["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
```

### 23.5 WebDataset (大图像/视频流式读取)

```bash
pip install webdataset
```

```bash
# 把数据打包成 tar (一次)
# data-000000.tar, data-000001.tar, ...
```

```python
import webdataset as wds

# 1. 加载
dataset = wds.WebDataset("data-%06d.tar").decode("pil").to_tuple("jpg", "cls")

# 2. DataLoader
loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

# 3. 训练
for imgs, labels in loader:
    # imgs: list of PIL.Image
    # labels: list of int
    ...
```

### 23.6 SSD 优化 (Mac 特有)

```bash
# 1. 检查 SSD 健康
smartctl -a /dev/disk0  # 需要 smartmontools

# 2. 检查 SSD 类型
system_profiler SPNVMeDataType | grep "Model"
# "Apple SSD" 是内置 NVMe (快)
# 外接 USB SSD 通常慢 3-5x

# 3. 测速
dd if=/dev/zero of=./testfile bs=1m count=1024 oflag=direct
# 100 MB/s 以下: USB 2.0 / 网络盘
# 500 MB/s 以下: USB 3.0
# 2000+ MB/s: 内置 NVMe

# 清理
rm ./testfile
```

### 23.7 内存映射大文件

```python
import numpy as np

# 不加载到内存, 直接 mmap
data = np.load("huge.npy", mmap_mode="r")  # (N, 3, 224, 224)

# 访问任意行
row_100 = data[100]  # 只读这一行到内存
```

### 23.8 数据下载 (国内加速)

```bash
# HuggingFace
export HF_ENDPOINT=https://hf-mirror.com

# pip (清华源)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch

# 阿里云
pip install -i https://mirrors.aliyun.com/pypi/simple/ torch

# conda (清华)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes

# aria2 多线程下载 (大文件)
brew install aria2
aria2c -x 8 https://huggingface.co/.../model.safetensors
```

### 23.9 数据预处理缓存

```python
"""预处理一次, 缓存到 SSD"""
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch

def preprocess_dataset(src_dir: Path, dst_dir: Path, size: int = 224):
    cache_dir = dst_dir / f"cache_{size}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    for img_path in src_dir.glob("*.jpg"):
        cache_path = cache_dir / f"{img_path.stem}.pt"
        if cache_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        torch.save(tensor, cache_path)

    print(f"Cached to {cache_dir}")
```

### 23.10 推荐的数据布局

```text
datasets/
├── raw/                    # 原始数据 (不可变)
│   ├── images/
│   ├── annotations/
│   └── metadata.csv
├── processed/              # 处理后 (不可变)
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
└── cache/                  # 训练用 (可重建)
    └── preprocessed/
```

### 23.11 训练时数据预取

```python
"""在后台线程预取下一批数据"""
from torch.utils.data import DataLoader
import queue

class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.Stream(device.type) if device.type == "cuda" else None

    def __iter__(self):
        first = True
        for batch in self.loader:
            with torch.cuda.stream(self.stream) if self.stream else nullcontext():
                batch = {k: v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            if not first:
                yield batch
                del prev_batch
            prev_batch = batch
            first = False
        yield prev_batch
```

### 23.12 不要做的 I/O 反模式

| 反模式 | 后果 | 正确做法 |
| :--- | :--- | :--- |
| 在主循环里读文件 | 极慢 | DataLoader + num_workers |
| 用 pickle 存 numpy | 慢且大 | np.save / parquet |
| 全量加载到内存 | OOM | memmap / 流式 |
| 单线程 DataLoader | GPU 等待 | num_workers > 0 (CUDA) |
| 重复解码图像 | 慢 | 预处理后保存为 .pt |
| 远程文件直接读 | 极慢 | 下载到本地 |

---

## 24. 远程开发 (SSH / VS Code Server / Tailscale)

> "我的 Mac 跑不动, 服务器跑得动" - 本节讲怎么无缝衔接远程 GPU 服务器。

### 24.1 SSH 基础配置

```bash
# ~/.ssh/config
Host gpu-server
    HostName 192.168.1.100
    User yourname
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ForwardAgent yes

Host gpu-cloud
    HostName <cloud-ip>
    User ubuntu
    IdentityFile ~/.ssh/cloud_key
    ProxyCommand none
```

### 24.2 SSH 密钥管理

```bash
# 1. 生成
ssh-keygen -t ed25519 -C "yourname@mac"

# 2. 复制到服务器
ssh-copy-id user@gpu-server

# 3. 多密钥管理 (例如 GitHub + 服务器不同)
# ~/.ssh/config
Host github.com
    IdentityFile ~/.ssh/id_ed25519_github
Host gpu-server
    IdentityFile ~/.ssh/id_ed25519_server
```

### 24.3 SSH 隧道转发端口

```bash
# 1. 转发服务器 Jupyter
ssh -L 8888:localhost:8888 user@gpu-server

# 2. 转发 TensorBoard
ssh -L 6006:localhost:6006 user@gpu-server

# 3. 多端口同时
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 user@gpu-server

# 4. 反向隧道 (服务器访问 Mac)
ssh -R 8888:localhost:8888 user@gpu-server
# 服务器上: 访问 localhost:8888 即访问 Mac

# 5. SOCKS 代理
ssh -D 1080 user@gpu-server
# 浏览器配 SOCKS 代理 127.0.0.1:1080
```

### 24.4 VS Code Remote SSH

```bash
# 1. 安装扩展
# - Remote - SSH (ms-vscode-remote.remote-ssh)
# - Remote Explorer (ms-vscode-remote.remote-explorer)

# 2. 打开远程窗口
# Cmd + Shift + P -> Remote-SSH: Connect to Host -> gpu-server

# 3. 自动安装 server 端 (Linux x64)
# VS Code 会自动在服务器上下载 vscode-server
```

`.vscode/settings.json` (远程端):

```jsonc
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "terminal.integrated.defaultProfile.linux": "bash",
    "remote.SSH.remotePlatform": {
        "gpu-server": "linux"
    }
}
```

### 24.5 Mosh (移动办公友好)

```bash
# Mac 安装
brew install mosh

# 服务器安装
sudo apt install mosh  # Ubuntu/Debian
# 防火墙要开 UDP 60000-61000

# 连接
mosh user@gpu-server
```

> mosh 在网络切换时不断线, 适合 WiFi 切换 / 休眠 / VPN 切换。

### 24.6 Tailscale (异地组网)

```bash
# 1. 安装
brew install --cask tailscale

# 2. 登录 (需账号)
# 在 Mac + 服务器都登录同一账号

# 3. 自动获得 IP (100.x.y.z)
# 4. 用 IP 直接连
ssh user@100.x.y.z
```

```bash
# 服务器端
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### 24.7 ZeroTier (Tailscale 替代)

```bash
brew install --cask zerotier-one
# 加入网络 (需要 network ID)
sudo zerotier-cli join <network-id>
```

### 24.8 tmux 远程会话

```bash
# 1. 启动
tmux new -s my-session

# 2. 断开 (Ctrl+B, D)
# 不退出, 程序继续跑

# 3. 重连
tmux attach -t my-session

# 4. 列出
tmux ls

# 5. 窗口分屏
# Ctrl+B, " (垂直分屏)
# Ctrl+B, % (水平分屏)
# Ctrl+B, 方向键 切换

# 6. 配置 (~/.tmux.conf)
cat > ~/.tmux.conf << 'EOF'
# 前缀改为 Ctrl+A
unbind C-b
set -g prefix C-a

# 启用鼠标
set -g mouse on

# 256 色
set -g default-terminal "screen-256color"

# 状态栏
set -g status-bg black
set -g status-fg white
set -g status-interval 1
EOF
```

### 24.9 rsync 大文件传输

```bash
# 1. 同步文件夹 (推荐, 比 scp 强)
rsync -avhP --progress ./local_dir/ user@server:~/remote_dir/

# 2. 压缩传输
rsync -avhPz --progress ./local_dir/ user@server:~/remote_dir/

# 3. 限速
rsync -avhP --bwlimit=10M ./local_dir/ user@server:~/

# 4. 排除文件
rsync -avhP --exclude='*.pyc' --exclude='.git/' ./ user@server:~/

# 5. 断点续传
rsync -avhP --partial ./local_dir/ user@server:~/remote_dir/
```

### 24.10 scp 简单传输

```bash
# 上传
scp local_file.tar.gz user@server:~/

# 下载
scp user@server:~/file.tar.gz ./

# 递归
scp -r ./local_dir/ user@server:~/

# 指定端口
scp -P 2222 local.tar.gz user@server:~/
```

### 24.11 远程 Jupyter

```bash
# 服务器上
jupyter lab --no-browser --port=8888

# 终端会输出:
# http://localhost:8888/lab?token=xxxxx

# Mac 上建立隧道
ssh -L 8888:localhost:8888 user@server

# 浏览器打开 http://localhost:8888
# 粘贴 token
```

### 24.12 SSH 配置文件最佳实践

```bash
# ~/.ssh/config
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
    Compression yes
    AddKeysToAgent yes
    UseKeychain yes       # macOS 特有, 用 Keychain 存密码

Host gpu-*
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519

Host gpu-server-1
    HostName 192.168.1.100

Host gpu-server-2
    HostName 10.0.0.50

# 通配符
Host gpu-*
    # 通用设置
```

```bash
# 把 passphrase 存到 Keychain (macOS)
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
# 之后 ssh 不再问密码
```

### 24.13 远程训练常用命令模板

```bash
# 服务器上启动后台训练
ssh gpu-server "cd ~/project && nohup python -m myproject.train > train.log 2>&1 &"

# 实时查看日志
ssh gpu-server "tail -f train.log"

# 监控 GPU
ssh gpu-server "nvidia-smi -l 1"

# 同步代码 + 数据
rsync -avhP ./ gpu-server:~/project/ --exclude='*.pyc' --exclude='.git/'

# 拉回 checkpoint
rsync -avhP gpu-server:~/project/runs/exp1/ ./runs/
```

### 24.14 VS Code Server (无 SSH 隧道)

```bash
# 服务器上 (Linux)
curl -fsSL https://code-server.dev/install.sh | sh

# 默认启动
sudo systemctl enable --now code-server@$USER

# 配置
# ~/.config/code-server/config.yaml
bind-addr: 0.0.0.0:8080
auth: password
password: <your-password>
cert: false

# Mac 上访问
# http://server-ip:8080
# 浏览器编辑代码 + 终端
```

### 24.15 用 Mac 当 SSH 跳板机

```bash
# ~/.ssh/config
Host internal-server
    HostName 10.0.0.5
    User dev
    ProxyJump gpu-server  # 通过 gpu-server 跳转
```

```bash
# 连接内网服务器 (无需在每台机器装密钥)
ssh internal-server
```

### 24.16 跨平台复制粘贴 (远程)

```bash
# 1. 装 tmux + 启用 OSC 52
# ~/.tmux.conf
set-option -g set-clipboard external

# 2. Mac 端 OSC 52 支持
brew install --cask iterm2  # iTerm2 自带 OSC 52 支持

# 3. 测试: 在 ssh 内复制, 粘贴到 Mac 应用
```

### 24.17 备份同步策略 (rclone)

```bash
brew install rclone

# 配置 (Google Drive / OneDrive / S3 / WebDAV)
rclone config

# 同步本地到云端
rclone sync ~/datasets remote:datasets -P

# 同步云端到本地
rclone sync remote:models ~/models -P

# 限速
rclone sync ~/datasets remote:datasets --bwlimit=50M
```

### 24.18 Mac + 服务器的工作流

```text
本地 (Mac)                      远程 (Server)
┌─────────────┐                 ┌──────────────┐
│  VS Code    │  --SSH-->       │  SSH daemon  │
│  Terminal   │                 │              │
│  iTerm2     │                 │  tmux session│
│  Git client │                 │  Python proc │
└─────────────┘                 └──────────────┘
       │                                │
       │                                │
       └─ rsync code/data ──────────────┘
       └─ TensorBoard 端口 ────────────►
       └─ WandB cloud sync ◄────────────┘
```

---

## 25. Docker on macOS (OrbStack)

> 想在 Mac 上跑 "Linux 环境" 的深度学习项目? OrbStack 是 2026 年的最佳选择。

### 25.1 OrbStack vs Docker Desktop

| 维度 | Docker Desktop | OrbStack | Colima |
| :---: | :---: | :---: | :---: |
| 启动速度 | 慢 | **极快** | 中 |
| 内存占用 | 大 | **小** | 中 |
| 价格 | 免费 (企业收费) | 免费 (个人) | 免费 |
| Rosetta | ✅ | ✅ | ✅ |
| 集成度 | 中 | **高** | 低 |
| Apple Silicon 优化 | 中 | **强** | 中 |

### 25.2 安装 OrbStack

```bash
brew install --cask orbstack
open -a OrbStack
```

### 25.3 创建 Linux 虚拟机 (ARM64)

```bash
# 列出可用发行版
orb create --help

# 创建 Ubuntu
orb create --arch arm64 ubuntu:22.04 my-ml

# 进入
orb shell my-ml

# 查看状态
orb list
```

### 25.4 在 OrbStack Linux 中跑 PyTorch (CPU)

```bash
# 进入 Linux
orb shell my-ml

# 安装 PyTorch CPU (Linux ARM64 wheels)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 验证
python -c "import torch; print(torch.__version__)"
```

### 25.5 Docker 兼容

```bash
# OrbStack 自动提供 docker CLI
docker run --rm hello-world

# 构建镜像
docker build -t my-ml:latest .

# 运行
docker run --rm -it -v ~/projects:/workspace my-ml:latest
```

### 25.6 Docker Compose (多容器)

```yaml
# docker-compose.yml
version: '3.8'

services:
  jupyter:
    image: jupyter/scipy-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./work:/home/jovyan/work
    environment:
      JUPYTER_ENABLE_LAB: "yes"

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./runs:/tmp/runs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```bash
# 启动
docker compose up -d

# 查看
docker compose ps

# 停止
docker compose down
```

### 25.7 ML 镜像最佳实践

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 工作目录
WORKDIR /workspace

# 默认命令
CMD ["python", "train.py"]
```

### 25.8 Docker 数据卷 (持久化)

```bash
# 1. 创建命名卷
docker volume create my-data

# 2. 挂载
docker run -v my-data:/data my-image

# 3. 备份
docker run --rm -v my-data:/data -v $(pwd):/backup \
    ubuntu tar czf /backup/data.tar.gz -C /data .
```

### 25.9 OrbStack 文件共享

```bash
# OrbStack 自动挂载 ~/ 到 /Volumes/Mac
# 在容器内:
ls /Volumes/Mac/Users/me/projects/
```

### 25.10 ML Docker 镜像选择

| 镜像 | 用途 | 大小 |
| :---: | :--- | :---: |
| `python:3.11-slim` | 通用 | 150 MB |
| `pytorch/pytorch:latest` | PyTorch | 5 GB |
| `tensorflow/tensorflow:latest` | TF | 1.5 GB |
| `jupyter/scipy-notebook` | Jupyter | 4 GB |
| `nvcr.io/nvidia/pytorch` | NVIDIA PyTorch | 10 GB (Linux) |

### 25.11 不要在 Mac Docker 里跑 CUDA

```bash
# ❌ 这没用, Docker in Mac 不能访问 MPS
docker run --gpus all nvcr.io/nvidia/pytorch  # 在 Mac 上不可行

# ✅ 正确做法: 用 OrbStack 直接跑, 但只用 CPU backend
docker run -it python:3.11 bash
# 或者用 MPS (PyTorch 原生)
```

### 25.12 OrbStack 的 Linux 集成

```bash
# OrbStack 提供完整的 Linux 体验
orb shell my-ml

# 1. 完整 apt
apt update && apt install -y libgl1-mesa-glx

# 2. 完整 systemd (部分)
systemctl status nginx

# 3. 网络访问 (与 macOS 共享)
ping 192.168.1.1

# 4. 文件共享
ls /Volumes/Mac
```

### 25.13 容器中的 Jupyter

```bash
# 在 OrbStack Linux 中
docker run -p 8888:8888 jupyter/scipy-notebook
# 浏览器: http://localhost:8888

# 数据持久化
docker run -p 8888:8888 \
    -v ~/projects:/home/jovyan/work \
    -v ~/datasets:/home/jovyan/datasets:ro \
    jupyter/scipy-notebook
```

### 25.14 镜像清理

```bash
# 列出所有镜像
docker images

# 删除
docker rmi <image-id>

# 清理悬空
docker image prune

# 清理全部
docker system prune -a

# OrbStack 磁盘占用
docker system df
```

---

## 26. Git 进阶 (worktree / submodules / LFS)

> 单仓多分支并行开发, 是 ML 工程师的"标配技能"。

### 26.1 Git Worktree (多分支并行)

```bash
# 1. 在同仓创建多个 worktree
git worktree add ../project-v2 -b feat/v2
git worktree add ../project-fix -b fix/bug-123

# 2. 列出
git worktree list

# 3. 在不同 worktree 用不同 Python 环境
# ../main     -> venv-main
# ../project-v2 -> venv-v2
# ../project-fix -> venv-fix

# 4. 清理
git worktree remove ../project-fix
```

### 26.2 Git Submodules (依赖外部仓)

```bash
# 1. 添加
git submodule add https://github.com/some/lib.git libs/lib

# 2. 初始化所有 submodules
git submodule update --init --recursive

# 3. 克隆含 submodules 的项目
git clone --recursive https://github.com/some/project.git

# 4. 更新 submodule
git submodule update --remote libs/lib

# 5. 在 submodule 中修改并提交
cd libs/lib
git checkout main
# 修改文件
git commit -am "feat: ..."
cd ../..
git add libs/lib
git commit -m "Update submodule"
```

### 26.3 Git LFS (大文件)

```bash
# 1. 安装
brew install git-lfs
git lfs install

# 2. 追踪大文件
git lfs track "*.pth"
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "data/*.parquet"

# 3. 提交
git add .gitattributes
git add model.pth
git commit -m "Add model"

# 4. 状态
git lfs ls-files

# 5. 拉取 (含 LFS)
git lfs pull

# 6. 清理 LFS cache
git lfs prune
```

### 26.4 `.gitattributes` 配置

```text
# .gitattributes
*.pth filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text

# 不要 LFS
*.py diff=python
*.md diff=markdown
```

### 26.5 分支策略 (Git Flow 简化版)

```bash
# 主分支
main           # 生产
develop        # 开发

# 临时分支
feat/xxx       # 新功能
fix/xxx        # 修复
exp/yyyymmdd   # 实验 (完成后删除)
```

### 26.6 实验分支管理 (ML 专用)

```bash
# 1. 创建实验分支
git checkout -b exp/2026-08-18-qwen-lora

# 2. 训练, commit 多次
git add src/ configs/
git commit -m "feat: add LoRA config"

# 3. 实验完成, merge 到 develop
git checkout develop
git merge exp/2026-08-18-qwen-lora

# 4. 删除实验分支
git branch -d exp/2026-08-18-qwen-lora
```

### 26.7 Commit Message 规范

```bash
# Conventional Commits
git commit -m "feat: add LoRA fine-tuning script"
git commit -m "fix: handle empty dataset"
git commit -m "docs: update README"
git commit -m "refactor: simplify data loader"
git commit -m "perf: cache tokenization"
git commit -m "test: add model tests"
git commit -m "chore: update dependencies"
```

### 26.8 Cherry-pick (挑选 commit)

```bash
# 把某个 commit 应用到当前分支
git cherry-pick <commit-hash>

# 多个 commit
git cherry-pick A..B  # A+1 到 B

# 不提交, 只 apply
git cherry-pick --no-commit <commit>
```

### 26.9 Stash (临时保存)

```bash
# 保存当前修改
git stash
git stash push -m "WIP: data loader refactor"

# 列出
git stash list

# 恢复
git stash pop          # 应用并删除
git stash apply stash@{0}  # 应用不删除

# 删除
git stash drop stash@{0}

# 清理全部
git stash clear
```

### 26.10 Bisect (二分查找 bug)

```bash
# 1. 开始
git bisect start

# 2. 标记当前 commit 为坏
git bisect bad

# 3. 标记一个已知好的 commit
git bisect good <commit-hash>

# 4. Git 自动 checkout, 你只需要 mark good/bad
git bisect good
git bisect bad

# 5. 结束
git bisect reset
```

### 26.11 Pre-commit Hook (代码质量)

```bash
pip install pre-commit
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
```

```bash
# 安装 hooks
pre-commit install

# 手动运行
pre-commit run --all-files
```

### 26.12 gh CLI (GitHub 命令行)

```bash
brew install gh

# 登录
gh auth login

# 创建 PR
gh pr create --title "Add LoRA script" --body "..."

# 查看 PR
gh pr view 123

# 列出 PR
gh pr list

# 检查 PR
gh pr checkout 123

# 合并
gh pr merge 123 --squash
```

### 26.13 Gitleaks (防止密钥泄漏)

```bash
brew install gitleaks

# 扫描仓库
gitleaks detect --source . -v

# 扫描 git 历史
gitleaks detect --source . --log-opts="--all" -v

# pre-commit 集成 (见 §26.11)
```

### 26.14 多 SSH 账户 (GitHub + 公司 GitLab)

```bash
# ~/.ssh/config
Host github.com
    HostName github.com
    IdentityFile ~/.ssh/id_ed25519_github
    User git

Host gitlab.company.com
    HostName gitlab.company.com
    IdentityFile ~/.ssh/id_ed25519_company
    User git

# 设置仓库 URL
git remote set-url origin git@github.com:user/project.git
```

---

## 27. 性能剖析 (Profiling)

> "代码跑得慢" 是 ML 项目的常态。本节讲怎么找到瓶颈。

### 27.1 三层 Profiling 视角

```text
1. 系统层 (top, powermetrics):   CPU/内存/功耗
2. Python 层 (cProfile, line_profiler): 函数级耗时
3. PyTorch 层 (torch.profiler): 算子级耗时
```

### 27.2 系统监控 (top / Activity Monitor)

```bash
# top 实时刷新
top -o cpu

# 按内存排序
top -o rsize | head

# 按 GPU 占用排序 (如果支持)
top -o gpu

# 5 秒采样
top -l 5 -s 1 | head

# powermetrics (Apple Silicon 详细)
sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 5

# Activity Monitor GUI
open -a "Activity Monitor"
```

### 27.3 Python cProfile

```python
import cProfile
import pstats

def train():
    # ... 训练逻辑 ...
    pass

# 1. 运行 profile
profiler = cProfile.Profile()
profiler.enable()
train()
profiler.disable()

# 2. 打印结果
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)  # 前 20 慢函数

# 3. 保存到文件
stats.dump_stats("train.prof")

# 4. 可视化 (snakeviz)
# pip install snakeviz
# snakeviz train.prof
```

### 27.4 line_profiler (行级)

```bash
pip install line_profiler
```

```python
@profile
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = torch.relu(x)
    x = self.fc(x)
    return x

# 运行
# kernprof -l -v script.py
```

### 27.5 memory_profiler (内存级)

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def train():
    model = ...
    for batch in loader:
        loss = model(batch)
        loss.backward()
```

```bash
# 运行
python -m memory_profiler script.py
```

### 27.6 PyTorch Profiler (推荐)

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # CUDA only
        # torch.profiler.ProfilerActivity.MPS,  # PyTorch 2.5+
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler"),
    record_shapes=True,
    with_stack=True,
) as profiler:
    for step, batch in enumerate(loader):
        # 训练步骤
        loss = model(batch)
        loss.backward()
        optimizer.step()
        profiler.step()

print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 27.7 PyTorch Profiler 输出解读

```text
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     Name    Self CPU %      Self CPU    CPU total %     CPU total  CPU time avg    # of Calls
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                              aten::addmm         8.5%       1.234s        85.0%       12.345s      12.34ms          1000
                             aten::conv2d         5.0%       0.725s         7.5%        1.090s       5.45ms           200
                          aten::batch_norm         3.0%       0.435s         4.0%        0.580s       2.90ms           200
                              aten::relu         2.0%       0.290s         2.0%        0.290s       0.14ms          2000
                              ...
-------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

| 字段 | 含义 |
| :---: | :--- |
| Name | 算子名 |
| Self CPU % | 该函数自身 CPU 时间占比 |
| Self CPU | 自身 CPU 时间 |
| CPU total % | 含子调用的总占比 |
| CPU total | 含子调用的总时间 |
| CPU time avg | 平均每次调用时间 |
| # of Calls | 调用次数 |

### 27.8 TensorBoard Profiler 集成

```bash
# 安装
pip install torch_tb_profiler

# 启动 TensorBoard
tensorboard --logdir=./logs
# 浏览器打开 http://localhost:6006
# 切换到 "PROFILER" 选项卡
```

### 27.9 找 DataLoader 瓶颈

```python
"""判断是 DataLoader 慢还是模型慢"""
import time

model = ...
loader = ...

# 1. 只测数据加载
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 10:
        break
data_time = (time.time() - t0) / 10
print(f"Avg data load time: {data_time*1000:.1f} ms/batch")

# 2. 只测模型 (固定数据)
fixed_batch = next(iter(loader))
fixed_batch = {k: v.to(device) for k, v in fixed_batch.items()}

torch.mps.synchronize()  # 或 cuda
t0 = time.time()
for i in range(10):
    loss = model(**fixed_batch)
    loss.backward()
torch.mps.synchronize()
model_time = (time.time() - t0) / 10
print(f"Avg model time: {model_time*1000:.1f} ms/batch")
```

> 如果 `data_time > model_time`, DataLoader 是瓶颈。加 num_workers / 用更快的存储。

### 27.10 找模型瓶颈 (PyTorch Profiler 实战)

```python
"""完整 profile 脚本"""
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet50().to("mps")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(32, 3, 224, 224, device="mps")
target = torch.randint(0, 1000, (32,), device="mps")

# Warmup
for _ in range(3):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    optimizer.step()

# Profile
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    with record_function("model_forward"):
        out = model(data)
    with record_function("compute_loss"):
        loss = torch.nn.functional.cross_entropy(out, target)
    with record_function("loss_backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
```

### 27.11 Instruments (Xcode 工具)

```bash
open /Applications/Xcode.app/Contents/Developer/Applications/Instruments.app

# 或者 (如果没有 Xcode)
xcrun instruments
```

可用模板:
- **Time Profiler**: CPU 热点
- **Allocations**: 内存分配
- **Metal System Trace**: GPU 详细
- **Network**: 网络

### 27.12 Metal System Trace (GPU 剖析)

```bash
# 录制 MPS 操作
xcrun metal-system-trace --help

# 启动录制 (示例)
xcrun metal-system-trace --output trace.metallib python script.py

# 用 Instruments 打开 trace.metallib
```

### 27.13 常见瓶颈速查

| 症状 | 可能瓶颈 | 解决 |
| :--- | :--- | :--- |
| GPU 利用率低 | DataLoader 慢 | num_workers, 预处理缓存 |
| 内存爆炸 | 激活值大 | Gradient checkpointing |
| CPU 高 GPU 低 | 算子不支持 MPS | CPU fallback 或换模型 |
| 小 batch 时慢 | 启动开销 | torch.compile |
| Loss 没降 | 学习率/数据 | 不是性能问题 |
| 第一次 step 慢 | MPSAllocator 初始化 | 预热几步 |

### 27.14 torch.compile (PyTorch 2.0+)

```python
import torch

model = MyModel().to("mps")

# 编译 (Mac 上有 MPS 后端支持)
compiled_model = torch.compile(model, backend="aot_eager")

# 第一次跑慢 (编译), 之后快
for _ in range(5):
    out = compiled_model(x)
```

> **注意**: MPS 上的 torch.compile 支持仍在完善, 部分算子可能不支持。

### 27.15 混合精度 profiling

```python
"""对比 FP32 / FP16 / BF16 性能"""
import time

device = torch.device("mps")

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    model = MyModel().to(device).to(dtype)
    x = torch.randn(32, 3, 224, 224, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        out = model(x)

    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(10):
        out = model(x)
    torch.mps.synchronize()
    dt = (time.time() - t0) / 10
    print(f"{dtype}: {dt*1000:.1f} ms/batch")
```

### 27.16 性能对照表 (M1 Max 32GB)

| 任务 | FP32 | FP16 | BF16 | 加速比 |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-50 fwd | 28 ms | 14 ms | 14 ms | 2.0x |
| BERT-base fwd | 18 ms | 10 ms | 10 ms | 1.8x |
| 简单 LSTM | 8 ms | 4 ms | 4 ms | 2.0x |

> BF16 速度与 FP16 相当, 数值范围更广, 推荐默认。

### 27.17 Profiling 工具对照

| 工具 | 层级 | 易用 | 适合 |
| :--- | :---: | :---: | :--- |
| `top` | 系统 | ⭐⭐⭐⭐⭐ | 快速排查 |
| `cProfile` | Python | ⭐⭐⭐⭐ | 函数级 |
| `line_profiler` | Python | ⭐⭐⭐ | 行级 |
| `memory_profiler` | Python | ⭐⭐⭐⭐ | 内存 |
| `torch.profiler` | PyTorch | ⭐⭐⭐⭐ | **算子级** |
| `Instruments` | 系统 | ⭐⭐⭐ | 深度剖析 |
| `Metal System Trace` | GPU | ⭐⭐ | MPS 专属 |

### 27.18 性能优化的心法

```text
1. 先 profile, 再优化 (Don't guess, measure)
2. 找瓶颈, 不要全面优化
3. 数据加载 > 模型 > 后处理 (常见优先级)
4. 内存 vs 速度的权衡
5. 简单改动 > 复杂改动 (先尝试 batch size + AMP)
```

---

## 28. 网络代理与下载加速

> 在国内, 很多 PyPI / Docker / HuggingFace 资源需要镜像。

### 28.1 pip 镜像

```bash
# 临时
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch

# 永久
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 多个源 (优先级降序)
pip config set global.extra-index-url \
    "https://pypi.tuna.tsinghua.edu.cn/simple
     https://mirrors.aliyun.com/pypi/simple/
     https://pypi.douban.com/simple/"
```

主要镜像:
- 清华: https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里: https://mirrors.aliyun.com/pypi/simple/
- 豆瓣: https://pypi.douban.com/simple/
- 中科大: https://pypi.mirrors.ustc.edu.cn/simple/

### 28.2 conda 镜像

```bash
# ~/.condarc
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - defaults
show_channel_urls: true
```

### 28.3 Docker 镜像

```bash
# macOS: OrbStack/Docker Desktop -> Settings -> Docker Engine
{
  "registry-mirrors": [
    "https://docker.mirrors.tuna.tsinghua.edu.cn",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}

# 然后重启 Docker
```

### 28.4 HuggingFace 镜像

```bash
# 环境变量 (推荐)
export HF_ENDPOINT=https://hf-mirror.com

# 临时使用
HF_ENDPOINT=https://hf-mirror.com python script.py

# .bashrc / .zshrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
```

```python
# Python 内设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2-7B-Instruct")
```

### 28.5 GitHub 加速

```bash
# 1. 镜像 clone
git clone https://ghproxy.com/https://github.com/xxx/yyy.git

# 2. 设置 git 代理 (HTTP/HTTPS)
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 3. 取消
git config --global --unset http.proxy
```

### 28.6 Homebrew 镜像

```bash
# Homebrew 官方源 (推荐, 不要用镜像!)
# 镜像经常滞后, 出问题难定位

# 例外: 中国地区可以使用镜像加速
# 参考: https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/

# 设置 HOMEBREW_API_DOMAIN (镜像 API)
export HOMEBREW_API_DOMAIN=https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles/api
```

### 28.7 aria2 多线程下载

```bash
brew install aria2

# 大文件 (模型权重)
aria2c -x 8 -s 8 \
    https://huggingface.co/xxx/model.safetensors

# 镜像加速
aria2c -x 8 \
    https://hf-mirror.com/xxx/model.safetensors
```

### 28.8 Clash / Surge (代理)

```bash
# 启动本地代理 (端口 7890 是 Clash 默认)
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

# 临时启用
http_proxy=http://127.0.0.1:7890 pip install torch

# 关闭
unset http_proxy https_proxy all_proxy
```

### 28.9 SSH 反向代理 (远程带宽)

```bash
# 通过海外服务器下载
ssh -D 1080 user@us-server

# 配置 git 走 SOCKS5
git config --global http.proxy socks5://127.0.0.1:1080

# 临时用 wget
wget --proxy ...  # 或用环境变量
```

### 28.10 model scope (国内 HF 替代)

```bash
# 安装 modelscope
pip install modelscope
```

```python
from modelscope import snapshot_download

# 下载模型
model_dir = snapshot_download(
    "qwen/Qwen2-7B-Instruct",
    cache_dir="~/modelscope_cache",
)
```

### 28.11 wisemodel (社区)

```bash
pip install wisemodel
```

```python
from wisemodel import snapshot_download

snapshot_download("LLM-Research/Llama-3-8B-Instruct")
```

### 28.12 不需要镜像的方案

```bash
# 1. 用 S3 / OSS 同步
aws s3 sync s3://my-bucket/models/ ~/models/

# 2. 用 rclone (见 §24.17)
rclone sync remote:models ~/models

# 3. 用 torrent (模型权重 magnet 链接)
brew install transmission
transmission-cli magnet:?xt=urn:btih:...
```

### 28.13 网络诊断

```bash
# 测速
curl -o /dev/null -w "%{speed_download}\n" https://huggingface.co/...

# 路由追踪
traceroute huggingface.co

# 端口检测
nc -zv huggingface.co 443

# DNS 解析
dig huggingface.co
```

---

## 29. macOS 自动化 (LaunchAgent / Hammerspoon / Shortcuts)

> 让 Mac 自己跑训练、定时备份、自动同步。

### 29.1 cron / launchd (定时任务)

```bash
# 推荐用 launchd (macOS 原生), 比 cron 稳

# 1. 创建 plist
cat > ~/Library/LaunchAgents/com.myname.daily-backup.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.myname.daily-backup</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>rsync -avhP ~/projects/ ~/backups/projects/</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/tmp/daily-backup.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/daily-backup.err</string>
</dict>
</plist>
EOF

# 2. 加载
launchctl load ~/Library/LaunchAgents/com.myname.daily-backup.plist

# 3. 启用/禁用
launchctl start com.myname.daily-backup
launchctl stop com.myname.daily-backup

# 4. 列出
launchctl list | grep myname

# 5. 卸载
launchctl unload ~/Library/LaunchAgents/com.myname.daily-backup.plist
```

### 29.2 训练启动器 (开机自动跑)

```bash
# 想让训练在 Mac 闲置时跑 (拉低功耗)
caffeinate -i python -m myproject.train
```

### 29.3 Hammerspoon (键盘自动化)

```bash
# 安装
brew install --cask hammerspoon

# 配 ~/.hammerspoon/init.lua
cat > ~/.hammerspoon/init.lua << 'EOF'
-- 全局快捷键: Cmd+Alt+L 打开 iTerm
hs.hotkey.bind({"cmd", "alt"}, "L", function()
    hs.application.launchOrFocus("iTerm")
end)

-- 全局快捷键: Cmd+Alt+T 打开训练日志
hs.hotkey.bind({"cmd", "alt"}, "T", function()
    hs.application.launchOrFocus("Terminal")
    hs.osascript.applescript('tell application "Terminal" to do script "tail -f ~/train.log"')
end)

-- 窗口管理: Win+左 = 左半屏
hs.hotkey.bind({"cmd", "alt"}, "Left", function()
    local win = hs.window.focusedWindow()
    if win then
        local f = win:frame()
        f.x = 0
        f.y = 0
        f.w = hs.screen.mainScreen():frame().w / 2
        f.h = hs.screen.mainScreen():frame().h
        win:setFrame(f)
    end
end)

-- 自动 reload 配置
function reloadConfig(files)
    hs.reload()
end
hs.pathwatcher.new(os.getenv("HOME") .. "/.hammerspoon/", reloadConfig):start()
EOF

# 启动 Hammerspoon.app
```

### 29.4 Apple Shortcuts (CLI)

```bash
# 创建 Shortcut (GUI 操作一次)
# 然后命令行调用
shortcuts run "My Backup"
```

### 29.5 触发 macOS 通知 (训练完成)

```python
import subprocess

def notify(title: str, message: str):
    """发 macOS 通知"""
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}"'
    ])

# 用法
notify("训练完成", "Loss=0.05, Acc=95%")
```

### 29.6 锁屏时继续训练

```bash
# caffeinate -i: 阻止系统闲置休眠
caffeinate -i python train.py

# caffeinate -s: 阻止屏幕休眠
caffeinate -s python train.py

# caffeinate -u -t 3600: 1 小时后停止阻止
caffeinate -u -t 3600 python train.py
```

### 29.7 静音训练 (合盖不中断)

```bash
# 合盖 (clamshell mode) 默认会让 Mac 休眠
# 设置: 系统设置 -> 电池 -> 电源适配器 -> "合盖时防止自动休眠"
# 或命令行 (旧 macOS):
sudo pmset -b disable clamsleep 1

# 或者用 insomnia
brew install --cask amphetamine
# Amphetamine 让合盖/闲置时 Mac 不睡
```

### 29.8 自动 SSH 重连

```bash
# ~/.ssh/config
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 10
    TCPKeepAlive yes
```

### 29.9 健康检查脚本

```bash
#!/bin/bash
# scripts/health_check.sh
# 检查训练环境是否健康

echo "=== 设备健康检查 ==="
echo ""

# 1. CPU
CPU_USAGE=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
echo "CPU 使用率: ${CPU_USAGE}%"

# 2. 内存
MEM_TOTAL=$(sysctl hw.memsize | awk '{print $2}')
MEM_USED=$(vm_stat | awk '/Pages active/ {print $3}' | sed 's/\.//')
MEM_USED_GB=$(echo "scale=2; $MEM_USED * 4096 / 1024 / 1024 / 1024" | bc)
echo "已用内存: ${MEM_USED_GB} GB"

# 3. 磁盘
DISK_FREE=$(df -h / | awk 'NR==2 {print $4}')
echo "可用磁盘: ${DISK_FREE}"

# 4. MPS 状态
python -c "
import torch
print(f'MPS 可用: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print(f'MPS 推荐最大: {torch.mps.recommended_max_memory()/1e9:.1f} GB')
"

# 5. 训练进程
TRAIN_PROCS=$(pgrep -f "python.*train" | wc -l | tr -d ' ')
echo "训练进程数: ${TRAIN_PROCS}"
```

### 29.10 训练日志自动归档

```bash
# 训练结束时自动打包日志和 checkpoint
cat >> my_train.py << 'EOF'
import os, shutil, datetime

def archive_artifacts(save_dir, log_file):
    """训练结束归档"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"archives/{ts}"
    os.makedirs(archive_dir, exist_ok=True)

    if os.path.exists(log_file):
        shutil.copy(log_file, archive_dir)
    if os.path.exists(save_dir):
        shutil.copytree(save_dir, f"{archive_dir}/checkpoints")

    print(f"Archived to {archive_dir}")
EOF
```

---

## 附录 A: 一页式速查卡

```bash
# ===== 设备检测 =====
uname -m                                  # 应为 arm64
sysctl -n machdep.cpu.brand_string        # Apple M*
sw_vers                                   # macOS 版本

# ===== Python =====
which python3                             # 应来自 uv / Miniforge
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.backends.mps.is_available())"

# ===== MPS =====
python3 -c "import torch; print(torch.mps.recommended_max_memory()/1e9, 'GB')"
python3 -c "import torch; torch.ones(1024,1024,device='mps'); torch.mps.synchronize()"

# ===== 包管理 =====
brew install <pkg>                        # Homebrew
uv pip install <pkg>                      # uv venv
conda install <pkg>                       # Miniforge

# ===== Git =====
git status                                # 状态
git log --oneline -10                     # 最近 10 次
git worktree list                         # worktree
git lfs ls-files                          # LFS 文件

# ===== Docker =====
docker ps                                 # 运行容器
docker images                             # 镜像
docker compose up -d                      # compose 启动

# ===== 训练 =====
ps aux | grep python | grep train         # 训练进程
kill -9 <PID>                             # 强制结束
nvidia-smi                                # CUDA (Mac 上不可用)
sudo powermetrics --samplers gpu_power    # MPS 监控

# ===== 文件 =====
du -sh ~/projects                         # 占用大小
df -h /                                   # 磁盘
lsof -i :8888                             # 端口占用
kill -9 $(lsof -ti:8888)                  # 关端口
```

---

## 附录 B: 完整故障排查清单 (15 类)

### B.1 MPS 不可用

```bash
# 1. Python 是 x86_64
arch -arm64 python3 -c "import torch; print(torch.backends.mps.is_available())"

# 2. PyTorch 是 CPU-only
pip install --force-reinstall torch

# 3. macOS 太旧
sw_vers  # 至少 13.0
```

### B.2 MPS OOM

```python
# 1. 降 batch size
# 2. 降序列长度
# 3. 用 AMP
with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    loss = model(x)

# 4. Gradient checkpointing
model.gradient_checkpointing_enable()

# 5. empty_cache
torch.mps.empty_cache()
```

### B.3 训练 Loss = NaN

```python
# 1. 降学习率
optimizer.param_groups[0]["lr"] *= 0.1

# 2. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 3. 检查输入
print(x.min().item(), x.max().item())
assert not torch.isnan(x).any()
```

### B.4 num_workers 死锁

```python
# MPS 必须 0
dataloader = DataLoader(dataset, num_workers=0, pin_memory=False)
```

### B.5 模型加载失败

```python
# shape mismatch
sd = torch.load("model.pth", weights_only=True, map_location="cpu")
model.load_state_dict(sd, strict=False)

# missing/unexpected keys
missing, unexpected = model.load_state_dict(sd, strict=False)
print(missing, unexpected)
```

### B.6 Docker 启动慢

```bash
# OrbStack 比 Docker Desktop 快很多
brew install --cask orbstack

# 清理
docker system prune -a
```

### B.7 端口占用

```bash
lsof -i :8888
kill -9 <PID>
# 或一行
kill -9 $(lsof -ti:8888)
```

### B.8 SSH 慢

```bash
# 编辑 ~/.ssh/config
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
    UseDNS no  # 关键!
```

### B.9 磁盘满

```bash
# 1. 找大文件
du -sh ~/* | sort -h | tail -20

# 2. 清理缓存
rm -rf ~/Library/Caches/*
rm -rf ~/.cache/pip/*

# 3. 清理 docker
docker system prune -a
```

### B.10 系统卡死 (MPS 占用)

```bash
# 重启前: 不敲键盘, 等 jetsam
# 重启后:
top -l 1 -o rsize | head  # 找大进程
kill -9 <PID>             # 杀进程

# 预防:
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
```

### B.11 wandb/tensorboard 打不开

```bash
# 1. 检查端口
lsof -i :6006

# 2. 防火墙
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3

# 3. SSH 隧道 (远程服务器)
ssh -L 6006:localhost:6006 user@server
```

### B.12 模型权重损坏

```python
import torch

try:
    sd = torch.load("model.pth", weights_only=True, map_location="cpu")
    print("Loaded OK")
except Exception as e:
    print(f"Failed: {e}")

# 检查完整性
for k, v in sd.items():
    if torch.isnan(v).any():
        print(f"NaN in {k}")
```

### B.13 HuggingFace 下载失败

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载
wget https://huggingface.co/.../model.safetensors

# 重试
pip install -U huggingface_hub
```

### B.14 Ollama 卡住

```bash
# 重启服务
brew services restart ollama

# 或
pkill -f ollama
ollama serve
```

### B.15 pip install 失败

```bash
# 1. 换源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>

# 2. 升级 pip
pip install --upgrade pip

# 3. 强制重装
pip install --force-reinstall <pkg>

# 4. 清理缓存
pip cache purge
```

---

## 附录 C: 配置文件全集

### C.1 ~/.zshrc 深度配置

```bash
# ~/.zshrc

# === Homebrew ===
if [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# === Python ===
export PATH="$HOME/.local/bin:$PATH"

# === HuggingFace ===
export HF_HOME=~/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

# === PyTorch MPS ===
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# === Git ===
export GIT_PAGER=cat

# === Aliases ===
alias py="python3"
alias pip="pip3"
alias ll="ls -lh"
alias la="ls -lah"
alias gs="git status"
alias gp="git push"
alias gpl="git pull"
alias gcm="git commit -m"
alias lg="lazygit"
alias train="uv run python -m myproject.train"
alias tb="tensorboard --logdir=runs"

# === Prompt ===
eval "$(starship init zsh)"

# === Plugins ===
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
    zoxide
    fzf
    web-search
    copypath
    copyfile
)
```

### C.2 settings.json (VS Code)

```jsonc
{
    "editor.fontSize": 14,
    "editor.fontFamily": "JetBrains Mono, Menlo, monospace",
    "editor.formatOnSave": true,
    "editor.tabSize": 4,
    "editor.minimap.enabled": false,
    "editor.renderWhitespace": "selection",

    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    },
    "python.analysis.typeCheckingMode": "basic",
    "python.defaultInterpreterPath": ".venv/bin/python",

    "[json]": {
        "editor.defaultFormatter": "vscode.json-language-features"
    },
    "[markdown]": {
        "editor.wordWrap": "on"
    },

    "terminal.integrated.fontFamily": "JetBrains Mono",
    "terminal.integrated.defaultProfile.osx": "zsh",

    "files.exclude": {
        "**/.git": true,
        "**/.DS_Store": true,
        "**/__pycache__": true,
        "**/.ipynb_checkpoints": true,
        "**/.ruff_cache": true,
        "**/.mypy_cache": true,
        "**/node_modules": true
    },

    "github.copilot.enable": {
        "python": true,
        "markdown": true
    },

    "remote.SSH.remotePlatform": {
        "gpu-server": "linux"
    }
}
```

### C.3 .gitignore_global

```gitignore
# macOS
.DS_Store
.AppleDouble
.LSOverride
._*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# ML 模型权重 (用 LFS 跟踪)
*.pth
*.pt
*.safetensors
*.bin
*.h5
*.parquet
*.onnx
runs/
checkpoints/

# 数据
data/
datasets/
*.csv
*.tsv
!data/.gitkeep

# Jupyter
.ipynb_checkpoints/
*.ipynb

# 工具
.mypy_cache/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/
.tox/
*.egg-info/

# IDE
.vscode/*
!.vscode/extensions.json
!.vscode/settings.json
.idea/

# OS
Thumbs.db
ehthumbs.db
desktop.ini
```

### C.4 pyproject.toml 完整版

```toml
[project]
name = "my-ml-project"
version = "0.1.0"
description = "ML pipeline on Apple Silicon"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "numpy>=1.26",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "tqdm>=4.66",
    "rich>=13.7",
    "tensorboard>=2.16",
    "safetensors>=0.4",
    "transformers>=4.40",
    "datasets>=2.19",
    "accelerate>=0.30",
    "peft>=0.10",
    "wandb>=0.17",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "ipython>=8.20",
    "jupyterlab>=4.1",
]
llm = [
    "llama-cpp-python>=0.2",
    "openai>=1.30",
]
tracking = ["wandb>=0.17", "mlflow>=2.14"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/myproject"]

[tool.ruff]
line-length = 100
target-version = "py311"
exclude = ["*.ipynb"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
strict_optional = true
warn_unused_ignores = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --strict-markers"
markers = [
    "slow: slow tests",
    "gpu: requires GPU",
    "integration: integration tests",
]
```

### C.5 Docker Compose for ML

```yaml
# docker-compose.yml
version: '3.8'

services:
  jupyter:
    image: jupyter/scipy-notebook:latest
    container_name: ml-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./work:/home/jovyan/work
      - ./datasets:/home/jovyan/datasets:ro
      - ./models:/home/jovyan/models:ro
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      JUPYTER_TOKEN: "your-token"
    restart: unless-stopped

  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: ml-tensorboard
    ports:
      - "6006:6006"
    volumes:
      - ./runs:/tmp/runs:ro
    command: tensorboard --logdir=/tmp/runs --host=0.0.0.0
    depends_on:
      - jupyter
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: ml-redis
    ports:
      - "6379:6379"
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    container_name: ml-postgres
    environment:
      POSTGRES_USER: ml
      POSTGRES_PASSWORD: ml
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: ml-mlflow
    ports:
      - "5000:5000"
    environment:
      MLFLOW_TRACKING_URI: postgresql://ml:ml@postgres:5432/mlflow
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - ./mlflow:/mlflow/artifacts
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  pgdata:
```

---

## 附录 D: 性能基准测试脚本

```python
#!/usr/bin/env python3
"""benchmark_mps.py - Apple Silicon 深度学习完整基准

用法:
    python benchmark_mps.py
    python benchmark_mps.py --output results.json
"""
import argparse
import json
import time
import platform
import torch
import torchvision


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_matmul(device, sizes=[1024, 2048, 4096]):
    results = {}
    for n in sizes:
        try:
            a = torch.randn(n, n, device=device)
            b = torch.randn(n, n, device=device)
            if device.type in ["mps", "cuda"]:
                torch.empty_cache() if device.type == "cuda" else torch.mps.empty_cache()

            # Warmup
            for _ in range(3):
                c = a @ b
            if device.type in ["mps", "cuda"]:
                getattr(torch, device.type).synchronize()

            # Bench
            t0 = time.time()
            for _ in range(5):
                c = a @ b
            if device.type in ["mps", "cuda"]:
                getattr(torch, device.type).synchronize()
            dt = (time.time() - t0) / 5

            tflops = 2 * n**3 / dt / 1e12
            results[f"matmul_{n}"] = {
                "time_ms": dt * 1000,
                "tflops": tflops,
            }
            del a, b, c
        except torch.cuda.OutOfMemoryError:
            results[f"matmul_{n}"] = {"OOM": True}
        except Exception as e:
            results[f"matmul_{n}"] = {"error": str(e)}
    return results


def benchmark_conv2d(device, channels=[64, 128, 256]):
    results = {}
    for c in channels:
        try:
            x = torch.randn(8, 3, 224, 224, device=device)
            conv = torch.nn.Conv2d(3, c, 3, padding=1).to(device)

            # Warmup
            for _ in range(3):
                y = conv(x)
            if device.type in ["mps", "cuda"]:
                getattr(torch, device.type).synchronize()

            # Bench
            t0 = time.time()
            for _ in range(10):
                y = conv(x)
            if device.type in ["mps", "cuda"]:
                getattr(torch, device.type).synchronize()
            dt = (time.time() - t0) / 10

            results[f"conv2d_{c}"] = {"time_ms": dt * 1000}
            del x, conv, y
        except Exception as e:
            results[f"conv2d_{c}"] = {"error": str(e)}
    return results


def benchmark_lstm(device, hidden_dims=[256, 512, 1024]):
    results = {}
    for h in hidden_dims:
        try:
            lstm = torch.nn.LSTM(h, h, num_layers=2, batch_first=True).to(device)
            x = torch.randn(32, 128, h, device=device)

            for _ in range(3):
                out, _ = lstm(x)
            if device.type in ["mps", "cuda"]:
                getattr(torch, device.type).synchronize()

            t0 = time.time()
            for _ in range(10):
                out, _ = lstm(x)
            if device.type in ["mps", "cuda"]:
                getattr(torch, device.type).synchronize()
            dt = (time.time() - t0) / 10

            results[f"lstm_{h}"] = {"time_ms": dt * 1000}
            del lstm, x, out
        except Exception as e:
            results[f"lstm_{h}"] = {"error": str(e)}
    return results


def benchmark_model(model_fn, name, device, input_shape, n_runs=10):
    """通用模型 benchmark"""
    try:
        model = model_fn().to(device).eval()
        x = torch.randn(*input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                y = model(x)
        if device.type in ["mps", "cuda"]:
            getattr(torch, device.type).synchronize()

        # Bench
        with torch.no_grad():
            t0 = time.time()
            for _ in range(n_runs):
                y = model(x)
        if device.type in ["mps", "cuda"]:
            getattr(torch, device.type).synchronize()
        dt = (time.time() - t0) / n_runs

        return {name: {"time_ms": dt * 1000, "input": input_shape}}
    except Exception as e:
        return {name: {"error": str(e)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"=== Apple Silicon / PyTorch Benchmark ===")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Machine: {platform.machine()}")
    print(f"macOS: {platform.mac_os_release_string()}")

    if device.type == "mps":
        print(f"MPS recommended max: {torch.mps.recommended_max_memory()/1e9:.1f} GB")
    print()

    results = {
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pytorch": torch.__version__,
            "device": str(device),
        },
        "matmul": benchmark_matmul(device),
        "conv2d": benchmark_conv2d(device),
        "lstm": benchmark_lstm(device),
    }

    # ResNet
    results["resnet18"] = benchmark_model(
        torchvision.models.resnet18, "resnet18", device, (1, 3, 224, 224)
    )
    results["resnet50"] = benchmark_model(
        torchvision.models.resnet50, "resnet50", device, (1, 3, 224, 224)
    )

    # 打印
    print(json.dumps(results, indent=2))

    # 保存
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 附录 E: FAQ 汇总

**Q: 我应该买哪款 Mac?**
A: 跑 LLM 选 M3/M4 Max 36GB+; 学习 ML 选 M2 Pro 16GB+; 跑 Stable Diffusion 选 M3 Max 48GB+。

**Q: MPS 上能跑 Stable Diffusion 吗?**
A: 能 (diffusers), 但比 CUDA 慢 5-10x。M3 Max 36GB 跑 SD1.5 单图约 5-10 秒。

**Q: 怎么把代码从 Linux 迁移到 MPS?**
A: 用 §19.3 的 `pick_device()`, 不要硬编码 `cuda`, MPS 不支持的部分用 CPU fallback。

**Q: MPS 会替代 CUDA 吗?**
A: 短期内不会。MPS 在推理场景够用, 训练 LLM 还差得远。但对小模型/学习很好。

**Q: PyTorch 与 TensorFlow 哪个更适合 MPS?**
A: PyTorch (PyTorch MPS 持续维护, TF Metal 已经停更)。

**Q: MLX 框架与 PyTorch MPS 哪个好?**
A: MLX 更原生 (Apple 出), 性能略好但生态不成熟。PyTorch MPS 生态完善, 性能略差。新项目可尝试 MLX, 老项目用 PyTorch。

**Q: 我在 macOS 上能做严肃研究吗?**
A: 小模型 + 学习: 完全够。大模型预训练: 仍需云端 GPU 服务器。微调 7B-13B: M3/M4 Max 64GB 可行。

**Q: 怎么升级 Python?**
A: 用 `uv` 管理多版本, 不要装在系统 Python 上。

**Q: pip install 报 "externally-managed-environment" 怎么办?**
A: 用虚拟环境 (uv / Miniforge / venv)。现代 Python 强制要求。

**Q: 训练时 Mac 风扇狂转?**
A: 正常。训练耗电大, 可以 `sudo powermetrics` 监控温度。>90°C 持续考虑降负载。

**Q: 怎么在 Apple Silicon 上做 cross-platform 兼容?**
A: 用 `torch.backends.mps.is_available()` 检测, 代码加 MPS 分支, 测试用 CI 矩阵覆盖。

**Q: 电池 / 电源适配器对训练有影响吗?**
A: 电源适配器时性能更好。电池时 macOS 会自动降频, 训练慢 20-40%。

**Q: Docker in Mac 能用 GPU 加速吗?**
A: **不能**。Docker in Mac 用的是 LinuxKit 虚拟机, 没有 GPU 透传。要 GPU 加速必须直接用 macOS 原生 PyTorch / MLX。

---

## 附录 F: 推荐资源

### F.1 官方文档
- PyTorch MPS: <https://docs.pytorch.org/docs/stable/notes/mps.html>
- MLX: <https://ml-explore.github.io/mlx/>
- tensorflow-metal: <https://developer.apple.com/metal/tensorflow-plugin/>
- llama.cpp: <https://github.com/ggerganov/llama.cpp>
- Ollama: <https://ollama.com/>

### F.2 实战教程
- HuggingFace on Apple Silicon: <https://huggingface.co/docs/transformers/en/mps>
- Apple's ML guides: <https://developer.apple.com/machine-learning/>

### F.3 社区
- r/LocalLLaMA: Reddit 本地 LLM 板块
- r/MacML: Reddit macOS ML 板块
- PyTorch Discuss: <https://discuss.pytorch.org/c/mps/>
- Apple Developer Forum: <https://developer.apple.com/forums/>

### F.4 工具链
- Homebrew: <https://brew.sh/>
- uv: <https://github.com/astral-sh/uv>
- Miniforge: <https://conda-forge.org/miniforge/>
- OrbStack: <https://orbstack.dev/>
- WandB: <https://wandb.ai/>
- TensorBoard: <https://www.tensorflow.org/tensorboard>

---

## 附录 G: 写在最后

**Apple Silicon 上的深度学习**已经度过了"勉强能用"的阶段。2026 年的 Mac:
- **MPS 后端**稳定, 大部分 PyTorch 模型直接跑
- **GGUF + llama.cpp**让 70B 模型本地推理成为可能
- **MLX 框架**给了 Metal 原生的另一条路
- **统一内存**省了 "显卡" 这笔钱 (虽然内存贵)

**不足**:
- 大规模训练仍是云端 GPU 的天下
- 部分 CUDA kernel 没有 Metal 等价物
- Flash Attention 等前沿优化仍在路上

**未来 3 年的趋势**:
1. MLX 生态会快速发展, 可能成为 macOS ML 主流
2. M5 / M6 Ultra 将工作站级 GPU 集成进 Mac
3. Apple 可能在 LLM 服务化上发力 (Apple Intelligence 已经起步)

> 一句话总结: **Apple Silicon 不是最强的深度学习平台, 但是 "够用 + 易用 + 美" 的最佳折衷**。

---

> **版本**: v3.0 (2026-08-18) · 维护: 自维护
>
> **本章内容**:
> - 原有 17 章 (v2.0): 系统配置 + MPS + TF Metal + VS Code + Git + 排错
> - 新增 12 章 (v3.0):
>   - 第18章 Apple Silicon 各代性能基准
>   - 第19章 完整 ML Pipeline
>   - 第20章 HuggingFace Transformers on MPS
>   - 第21章 本地 LLM 推理 (llama.cpp / Ollama)
>   - 第22章 ML 工具链 (WandB / TensorBoard / MLflow)
>   - 第23章 数据存储与 I/O
>   - 第24章 远程开发 (SSH / VS Code Server)
>   - 第25章 Docker on macOS (OrbStack)
>   - 第26章 Git 进阶 (worktree / submodules / LFS)
>   - 第27章 性能剖析 (Profiling)
>   - 第28章 网络代理与下载加速
>   - 第29章 macOS 自动化
> - 新增 7 个附录 (A-G)
>
> **下一步**:
> - 配合 基础/工程能力/ 学习工程化
> - 配合 方向教程/rec教程/ 学习搜推广
> - 配合 方向教程/pth教程/第01章-PTH权重读取与可视化教程.md 学习 .pth 实战



## 深度扩展 · 多视角系统展开(续)

### 110. macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南 - 架构模式 视角

**核心议题**: 本节从架构模式视角切入,系统展开 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 架构模式视角的展开**

从架构模式视角出发,我们可以把 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调架构模式视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"架构模式" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从架构模式视角切入,系统展开了 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 111. macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南 - 配置管理 视角

**核心议题**: 本节从配置管理视角切入,系统展开 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 配置管理视角的展开**

从配置管理视角出发,我们可以把 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调配置管理视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"配置管理" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从配置管理视角切入,系统展开了 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 112. macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南 - 性能调优 视角

**核心议题**: 本节从性能调优视角切入,系统展开 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 性能调优视角的展开**

从性能调优视角出发,我们可以把 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调性能调优视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"性能调优" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从性能调优视角切入,系统展开了 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 113. macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南 - 评估体系 视角

**核心议题**: 本节从评估体系视角切入,系统展开 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 评估体系视角的展开**

从评估体系视角出发,我们可以把 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调评估体系视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"评估体系" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从评估体系视角切入,系统展开了 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。


---

### 114. macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南 - 可维护 视角

**核心议题**: 本节从可维护视角切入,系统展开 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的相关讨论。

**1. 背景与动机**

任何技术或方法的出现都不是孤立的,而是回应特定历史条件下的具体问题。"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的发展也不例外。要理解它,必须回到它出现的那个时间点,看清它要解决的真实问题是什么。当时的主流方案有哪些不足? 它的提出填补了哪些空白? 这些问题的回答,决定了我们今天该如何使用它。

从知识论角度看,**技术演进是问题驱动的**——不是先有方案再找问题,而是先有问题再有方案。这个观察可以帮助我们避免"拿着锤子找钉子"的陷阱,真正把方法用对地方。

历史维度上,我们看到三个规律:

1. **从特殊到一般**: 早期方法往往针对特定场景,后期抽象出通用框架。学习的认知规律也是先掌握具体案例,再抽象出一般模式。这种"自下而上"的学习方式,在机器学习模型的设计里反复出现。

2. **从手工到自动**: 经验规则逐步被数据驱动的方法取代。在早期,很多 AI 系统依赖人工编写的规则。统计学习方法的崛起让"从数据中学习"成为可能。深度学习进一步把特征工程也自动化了。今天的大模型时代,我们看到端到端学习从感知扩展到了推理与生成。

3. **从局部到全局**: 关注单个组件,演进到关注整个系统。早期模型评估只关注单点指标,现在更关注全链路的协同效应。这种"系统视角"的转变,与软件工程从单体架构到微服务架构的演进高度相似。

**2. 可维护视角的展开**

从可维护视角出发,我们可以把 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 拆解为以下几个层次:

**第一层: 概念层** —— 这是最抽象的层次,定义了我们要研究的对象是什么。例如,"推荐系统"在概念层是一个"为用户从物品集合中筛选合适物品"的问题。这个层次的特点是清晰、稳定、不易变化。一个好的概念定义应该具备几个特征:可操作性、可测量性、可扩展性。

**第二层: 模型层** —— 把概念实例化为数学模型。同一概念可以有不同的模型化方式,例如"推荐系统"可以是协同过滤、矩阵分解、深度学习等。模型的选择决定了后续算法的空间,也决定了能利用的归纳偏置(比如序列模型擅长捕捉时序依赖,图模型擅长建模关系结构)。模型选型是工程实践中最重要的决策之一。

**第三层: 算法层** —— 在选定模型下,具体的求解方法。这一层最活跃,新算法层出不穷。同一模型往往有多种算法,各有优劣。例如同样是优化深度网络,可以选择 SGD、Adam、L-BFGS 等不同算法。算法的选择要考虑收敛速度、数值稳定性、计算成本等多个维度。

**第四层: 工程层** —— 算法落地的实现细节,包括数据流水线、训练框架、推理优化、监控告警等。工程层是研究到生产的桥梁,决定了一个方法能否真正用起来。一个在论文里表现很好的算法,如果不能高效实现、不能稳定运行、不能快速迭代,就没有实际价值。工程能力是把研究转化为生产力的关键。

**第五层: 业务层** —— 在具体业务场景中的表现。同一技术在不同业务里效果差异巨大,因为业务目标、数据分布、用户行为都有差异。从业务视角看技术,我们必须回答几个问题:这个技术解决了什么业务问题?投入产出比如何?长期可持续吗?技术评估必须与业务目标对齐,否则容易陷入"技术自嗨"的陷阱。

**3. 数学形式化(简洁版)**

虽然我们强调可维护视角,但数学形式化提供了共同语言。机器学习的核心问题可以表达为:给定输入样本,学习一个映射函数,使得函数输出与真实目标的差距最小化。这个抽象捕捉了分类、回归、生成等大多数任务的本质。

进一步,优化目标通常加入正则化项来控制模型复杂度,平衡经验风险与模型容量。统计学习理论给出了泛化误差的上界,告诉我们模型越复杂并不一定越好——**奥卡姆剃刀**在数学上有严格的形式化。

偏差-方差分解是另一个核心框架:误差可以分解为模型偏差(欠拟合)、方差(过拟合)和不可约噪声三部分。理解这个分解有助于诊断模型表现、选择合适复杂度。

在数值稳定性方面,有几个常见陷阱: 浮点数精度有限、Log-Sum-Exp 计算要减去最大值、Softmax 在大数值下会溢出。这些工程细节常常决定一个模型能否在生产环境稳定运行。

**4. 信息论视角**

信息论为"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 提供了统一的视角:**学习的本质是从数据中提取与目标相关的信息**。

互信息衡量知道一个变量后,另一个变量不确定性减少的量。在机器学习中,它被广泛用于特征选择、表示学习、多模态对齐等场景。

KL 散度衡量两个概率分布的距离,在交叉熵损失、变分推断、策略梯度等场景无处不在。需要注意它是**不对称**的——不同方向有不同的解释。

信息瓶颈原则指出,我们应该寻找既紧凑又预测性强的表示。这一思想影响了大量表示学习方法的设计,包括对比学习、自监督学习、多模态预训练等。

**5. 算法实现(骨架)**

虽然我们强调"可维护" 视角,但落到代码层面是类似的骨架:

```python
def train_model(data, model_class, config):
    """通用训练函数骨架"""
    model = model_class(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    for epoch in range(config.epochs):
        for batch in data:
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
```

更详细的代码放在附录或专门的实现章节,本节不重复。

**6. 评估方法**

评估 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的实际表现,需要多维度指标:

| 维度 | 指标 | 目标 |
| :--- | :--- | :--- |
| **准确性** | Accuracy / F1 / AUC | 越高越好 |
| **效率** | Latency / Throughput | 越低越好 |
| **鲁棒性** | Robustness / Calibration | 越高越好 |
| **公平性** | Fairness metrics | 符合业务 |
| **可解释性** | Explainability | 满足合规 |

**7. 实践建议**

结合 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的特点,实践中有以下建议:

1. **数据优先**: 数据质量决定上限,模型只是逼近这个上限。把 50% 的时间投入到数据理解和清洗上,这条经验被无数实践证明。数据问题往往比模型问题更关键,但也更容易被忽视。

2. **基线先行**: 先用最简单的方法(如逻辑回归、规则)做基线,确认增益是来自方法还是数据。没有强基线的"创新"往往不靠谱。很多看似 fancy 的方法,在强基线面前提升有限。

3. **迭代式改进**: 不要追求一步到位,而是 baseline → 简单改进 → 高级方法 → 调优。每一步都用验证集确认有提升。每一步的提升幅度通常递减,要预期边际收益下降。

4. **失败案例分析**: 重点看模型做错的样本,往往能发现系统性问题。错误样本里藏着模型没学会的知识。把这些样本按错误类型分类,可以发现模型的薄弱点。

5. **A/B 测试**: 在生产环境做对照实验,用业务指标而非离线指标评估。离线指标只是手段,业务指标才是目的。指标可以被优化,但业务目标很难直接优化。

6. **监控与回滚**: 上线后持续监控关键指标,异常时快速回滚。任何线上系统都必须考虑"最坏情况",并预先设计好应对方案。

**8. 常见误解**

对 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 常见的误解包括:

1. **"模型越复杂越好"**: 实际上,简单模型 + 好数据往往优于复杂模型 + 烂数据。这条法则在实践中反复被验证。

2. **"训练 loss 越低越好"**: 训练 loss 低不代表泛化好,可能过拟合。要看验证集表现,做误差分析。

3. **"指标好就等于业务好"**: 离线指标与在线业务有 gap,必须做 A/B 测试。指标可以被优化,但业务目标很难直接优化。

4. **"新方法一定比旧方法好"**: 新方法通常针对特定场景,在你的场景未必更好。要谨慎评估,先复现,再改进。

5. **"开源即用"**: 开源代码与你的数据/工程栈往往有 gap,需要适配。盲目复用经常踩坑,要理解代码再做改动。

**9. 与相关主题的关系**

"macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 与以下主题密切相关:

- **上游**: 数据质量、特征工程、业务理解。这些决定了模型的天花板。
- **下游**: 工程部署、效果监控、迭代优化。这些决定了模型的下限。
- **平行**: 类似方法、竞争方案、补充技术。了解竞品才能做得更好。

理清这些关系,有助于建立完整的知识图谱。技术学习不能孤立,要在关系网络中理解。每一个技术节点都与其他节点相连,只看到一个节点会失去全局视野。

**10. 小结**

本节从可维护视角切入,系统展开了 "macOS 深度学习小 tips - Apple Silicon 环境配置与避坑指南" 的背景、层次、数学形式、信息论视角、算法实现、评估方法、实践建议、常见误解与相关主题。这是理解整体的关键切片之一。

建立完整理解需要多视角:理论视角看到原理,工程视角看到实现,应用视角看到场景,评估视角看到局限。多个视角叠加,才能避免盲人摸象。

下一节将从另一个视角继续展开,逐步建立完整的认知地图。



##



##



##



##



