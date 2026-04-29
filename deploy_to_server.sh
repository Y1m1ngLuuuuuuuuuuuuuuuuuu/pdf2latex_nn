#!/usr/bin/env bash
# AutoDL environment verification helper.
# Run on the server after activating the intended conda environment:
#   source /root/miniconda3/etc/profile.d/conda.sh
#   conda activate pdf2latex
#   bash deploy_to_server.sh

set -euo pipefail

echo "=== 1. 检查环境 ==="
python --version
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc not found"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo -e "\n=== 2. 创建工作目录 ==="
PROJECT_DIR="${AUTODL_PROJECT:-/root/autodl-tmp/pdf2latex_nn}"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

echo -e "\n=== 3. 检查 Kaggle 凭证 ==="
if [ -f /root/.kaggle/access_token ]; then
  stat -c "%A %U %G %n" /root/.kaggle/access_token
else
  echo "missing /root/.kaggle/access_token"
fi

echo -e "\n=== 4. 验证 Python 依赖 ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}')"
python -c "import kaggle; print('Kaggle CLI module: ok')"

echo -e "\n=== 环境检查完成 ==="
