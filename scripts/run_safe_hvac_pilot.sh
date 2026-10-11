#!/usr/bin/env bash
# GPU HOST ONLY: six matched runs, sequential EnergyPlus processes.
set -euo pipefail
task_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_project_root"
python -m safe_hvac.doctor --config configs/safe_hvac/train_s01.yaml
python -m pytest -q tests/safe_hvac
task_batch_name="${1:-pilot-$(date -u +%Y%m%dT%H%M%S)}"
task_result_root="results/safe_hvac/$task_batch_name"
task_runs=()
for task_agent_seed in 2 3 4; do
  task_env_seed=$((task_agent_seed + 100))
  for task_algorithm in ppo barrier_ppo; do
    task_run_name="$task_algorithm-a$task_agent_seed-e$task_env_seed"
    python -m safe_hvac.train --config configs/safe_hvac/train_s01.yaml \
      --algorithm "$task_algorithm" --agent-seed "$task_agent_seed" --env-seed "$task_env_seed" \
      --output-root "$task_result_root" --run-name "$task_run_name"
    task_runs+=("$task_result_root/$task_run_name")
  done
done
python -m safe_hvac.report --runs "${task_runs[@]}" --output "$task_result_root/comparison"
