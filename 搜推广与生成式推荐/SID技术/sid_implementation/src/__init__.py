"""sid_implementation: 从 LLM embedding 到在线 SID 服务的端到端实现。

本包是一个完整的 Semantic ID (SID) 工程化实现, 涵盖以下核心能力:

1. **LLM Embedding 提取** (`llm_embedding.py`)
   支持 Qwen / LLaMA / GLM 等文本 LLM 与 Qwen2-VL / InternVL2 等多模态 LLM,
   并提供基于特征哈希的回退实现, 保证无 transformers 环境下也能跑通。

2. **行为融合** (`behavior_fusion.py`, `bipartite_graph.py`)
   把用户的曝光 / 点击 / 下载 / 付费等行为信号注入到物品 / 查询 embedding 中,
   让 SID 同时编码内容语义与行为偏好。

3. **SID 编码器**
   - `rq_kmeans.py`: RQ-Kmeans (残差量化 + K-means, 无 encoder, TRM 论文方案)
   - `rq_vae.py`   : RQ-VAE (残差量化 + VAE, 带 encoder / decoder)
   - `build_candidate_sid.py`: 方案 1, 只对候选物品编码
   - `build_unified_sid.py`  : 方案 2, 查询与物品编码到统一语义空间

4. **辅助模块**
   - `hy_token.py`  : Hy Token (gen-token + mem-token 的 Wide & Deep 融合)
   - `quasid.py`    : QuaSID 系列增强 (rating 损失、norm 均衡、变长 SID)
   - `hamr_loss.py` : Hamming-guided Margin Repulsion 损失

5. **在线服务**
   - `faiss_sid.py`        : FAISS 加速的 O(1) 新物品 SID 推断
   - `online_sid_server.py`: 生产级服务, 支持冷启动、多路径、健康监控、增量更新

6. **评估体系**
   - `sid_metrics.py`   : 14 项单点指标 (QuaSID / DRQ / semantic-id lib)
   - `sid_evaluation.py`: 5 维度综合评估 + 自动重训决策

## 包级环境守卫

导入本包时会做一次环境守卫: 当 PyTorch 与 FAISS 同时被加载时, 两者各自静态
链接了一份 OpenMP 运行时 (libomp/libiomp), 在 macOS 上会触发 libomp 的
"already initialized" 检测并 SIGABRT (OMP Error #15), 使得依赖 faiss 的示例
(run_dlrm_production / run_online_sid_server) 直接崩溃。

设置 KMP_DUPLICATE_LIB_OK=TRUE 允许第二份运行时继续加载, 这是 faiss+torch
共存在 macOS 上的标准 workaround。用 setdefault 以尊重用户/CI 的显式覆盖;
必须在 `import faiss` 之前生效, 而本 __init__ 在任何 `from src.* import` 时最先执行。
"""

import os
import sys
import warnings

# ---------------------------------------------------------------------------
# OpenMP 双运行时兼容: PyTorch 与 faiss 各自静态链接 OpenMP, 在 macOS 上
# 同时加载会 SIGABRT。这里允许加载第二份运行时副本, 是社区公认 workaround。
# 必须放在任何 torch/faiss 导入之前, 而本 __init__ 是包内第一个被执行的文件。
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# macOS Accelerate BLAS (numpy 2.x 默认后端) 的已知问题: 即便对完全有限的单位向量
# 做 matmul, 也会置位 FPE 标志并抛出
#   "divide by zero / overflow / invalid value encountered in matmul"
# 这些是 Accelerate 的伪警告, 与本仓库的数值正确性无关 (可用 -W error 复现: 有限输入
# 依然触发)。真正的非有限输入由各函数内部的 nan_to_num 兜底。仅在 macOS 上按精确的
# 消息模式静默, 不影响其它平台上真实的溢出告警。
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    warnings.filterwarnings(
        "ignore",
        message=r".*encountered in matmul",
        category=RuntimeWarning,
    )
