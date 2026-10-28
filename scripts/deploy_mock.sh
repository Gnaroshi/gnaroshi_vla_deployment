#!/usr/bin/env bash
# Generic skeleton example: fake policy and fake embodiment only.
set -euo pipefail
target_hz=60
max_steps=5000
python_bin="${DEPLOYMENT_PYTHON:-python3}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
output_root="${DEPLOYMENT_OUTPUT_ROOT:-$repo_root/.runtime/mock}"
export PYTHONPATH="$repo_root/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" -m gnaroshi_deployment.cli --output-root "$output_root" --target-hz "$target_hz" --max-steps "$max_steps" "$@"
