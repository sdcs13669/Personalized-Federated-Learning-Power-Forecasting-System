#!/usr/bin/env bash
# ============================================================
# 一键创建 fl 环境（路线2）— Ubuntu Linux + NVIDIA GPU
# 用法：
#     bash setup_env.sh            # 默认 CUDA 12.8（torch 2.8.0+cu128，驱动 >= 570）
#     bash setup_env.sh cu126      # 驱动较旧：cu126 / cu124 / cu121
#     bash setup_env.sh cpu        # 无 GPU 的机器
#     conda activate fl
#
# 说明：
#   - 所有版本与本机 Windows fl 环境一致（torch 2.8.0 / flwr 1.30.0 等）
#   - 本脚本独立运行，不依赖 environment.yml
#   - 国内服务器可取消下方镜像注释加速下载
# ============================================================
set -euo pipefail

ENV_NAME="fl"
PY_VER="3.10"
CUDA="${1:-cu128}"

# --- 清华镜像（可选加速，取消注释即可） ---
# PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
# TORCH_INDEX_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels"
PIP_MIRROR="${PIP_MIRROR:-}"
TORCH_INDEX_MIRROR="${TORCH_INDEX_MIRROR:-https://download.pytorch.org/whl}"

case "$CUDA" in
  cu128) TORCH_INDEX="$TORCH_INDEX_MIRROR/cu128" ;;
  cu126) TORCH_INDEX="$TORCH_INDEX_MIRROR/cu126" ;;
  cu124) TORCH_INDEX="$TORCH_INDEX_MIRROR/cu124" ;;
  cu121) TORCH_INDEX="$TORCH_INDEX_MIRROR/cu121" ;;
  cpu)   TORCH_INDEX="$TORCH_INDEX_MIRROR/cpu" ;;
  *) echo "未知 CUDA 版本: $CUDA（可选: cu128 / cu126 / cu124 / cu121 / cpu）"; exit 1 ;;
esac

# 1. 检查 conda
if ! command -v conda >/dev/null 2>&1; then
  echo "[错误] 未找到 conda，请先安装 Miniconda："
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  echo "  bash Miniconda3-latest-Linux-x86_64.sh"
  exit 1
fi

# 2. 创建环境（tk 供 visualize_eval.py 图形界面使用）
echo "[1/5] 创建 conda 环境 $ENV_NAME (python=$PY_VER) ..."
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "  环境 $ENV_NAME 已存在，跳过创建"
else
  conda create -n "$ENV_NAME" python="$PY_VER" tk -c conda-forge -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. PyTorch（CUDA 版，先装，避免后续包把 CPU 版 torch 拖进来）
echo "[2/5] 安装 PyTorch 2.8.0 ($CUDA) ..."
pip install $PIP_MIRROR torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url "$TORCH_INDEX"

# 4. 科学计算栈
echo "[3/5] 安装科学计算库 ..."
pip install $PIP_MIRROR \
  numpy==2.2.6 pandas==2.3.3 matplotlib==3.10.9 scipy==1.15.3 \
  scikit-learn==1.7.2 statsmodels==0.14.6 PyYAML==6.0.3

# 5. 联邦学习 / 差分隐私 / 特征选择
echo "[4/5] 安装联邦学习与特征选择库 ..."
pip install $PIP_MIRROR \
  "flwr[simulation]==1.30.0" ray==2.57.0 opacus==1.6.0 \
  xgboost==2.1.4 shap==0.49.1

# 6. 验证
echo "[5/5] 验证安装 ..."
python - <<'EOF'
import numpy, pandas, matplotlib, yaml, sklearn, statsmodels
import xgboost, shap, flwr, ray, opacus, torch
print(f"  python OK | torch {torch.__version__} | "
      f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "============================================================"
echo "环境创建完成！激活方式："
echo "  conda activate $ENV_NAME"
echo ""
echo "快速自检（联邦训练冒烟测试）："
echo "  python -m fl_code.train_baseline --rounds 1 --clients steel_ind_0"
echo "============================================================"
