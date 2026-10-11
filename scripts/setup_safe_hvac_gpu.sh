#!/usr/bin/env bash
# GPU HOST ONLY. EnergyPlus must be installed separately using the runbook.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/setup_safe_hvac_gpu.sh https://download.pytorch.org/whl/<CUDA build>'
  echo 'Choose the build for this GPU/driver using https://pytorch.org/get-started/locally/'
  exit 2
fi
task_torch_index="$1"
case "$task_torch_index" in
  https://download.pytorch.org/whl/cu*) ;;
  *) echo 'Use an official CUDA PyTorch wheel index'; exit 2 ;;
esac
task_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_project_root"
python3.12 -m venv .venv-safe-hvac
source .venv-safe-hvac/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.7,<3' --index-url "$task_torch_index"
python -m pip install -r requirements-safe-hvac.txt
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
mkdir -p results/setup
python -m pip freeze > results/setup/python-lock.txt
echo 'Activate .venv-safe-hvac, set EPLUS_PATH/PYTHONPATH, then follow docs/safe_hvac/GPU_RUNBOOK.md'
