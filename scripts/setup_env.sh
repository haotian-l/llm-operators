#!/usr/bin/env bash
# 在 GPU 机器上建引擎环境：envs/vllm、envs/sglang。
#
#   scripts/setup_env.sh vllm|sglang|all
#
# 步骤（每个环境）：
#   1. 装引擎（requirements/<engine>.txt），torch / flashinfer 等由引擎自己 pin
#   2. pip freeze 成 constraints，保证后面装的工具不会改动引擎的版本
#   3. 在 constraints 下装共用工具（requirements/tools.txt）
#   4. --no-deps 装 FLA（requirements/nodeps.txt）和本仓库的 k3ops
#   5. 跑 scripts/check_env.py
#
# 可调环境变量：
#   PY=3.12                     Python 版本
#   VLLM_TORCH_BACKEND=auto     传给 uv 的 --torch-backend（vLLM 官方推荐 auto）
#   SGLANG_TORCH_BACKEND=       为空时按 PyPI 默认解析
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
  echo "需要先安装 uv：https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

setup() {
  local engine="$1"
  local env="$ROOT/envs/$engine"
  local py="$env/bin/python"
  local backend=""
  case "$engine" in
    vllm) backend="${VLLM_TORCH_BACKEND-auto}" ;;
    sglang) backend="${SGLANG_TORCH_BACKEND-}" ;;
    *) echo "未知引擎：$engine" >&2; exit 1 ;;
  esac
  # 空数组在 bash<4.4 + set -u 下会报 unbound，所以用 ${arr[@]+...} 展开
  local backend_args=()
  if [ -n "$backend" ]; then
    backend_args=(--torch-backend "$backend")
  fi

  echo "==> envs/$engine"
  uv venv --python "$PY" "$env"
  uv pip install --python "$py" ${backend_args[@]+"${backend_args[@]}"} \
    -r "$ROOT/requirements/$engine.txt"
  uv pip freeze --python "$py" > "$env/engine-constraints.txt"
  uv pip install --python "$py" ${backend_args[@]+"${backend_args[@]}"} \
    -c "$env/engine-constraints.txt" -r "$ROOT/requirements/tools.txt"
  uv pip install --python "$py" --no-deps -r "$ROOT/requirements/nodeps.txt"
  uv pip install --python "$py" --no-deps -e "$ROOT"
  uv pip freeze --python "$py" > "$env/freeze.txt"
  "$py" "$ROOT/scripts/check_env.py" --engine "$engine"
}

case "${1:-}" in
  vllm|sglang) setup "$1" ;;
  all) setup vllm; setup sglang ;;
  *) echo "用法：$0 vllm|sglang|all" >&2; exit 1 ;;
esac
