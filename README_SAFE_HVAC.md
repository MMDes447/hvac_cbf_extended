# Corrected safe HVAC PPO

This package prepares the released HVAC barrier method for controlled Sinergym experiments. It includes a plain SB3 PPO baseline, a corrected barrier PPO agent, bounded records/checkpoints, regression tests, and a GPU runbook. **No building simulation or experimental training was run on the preparation machine. Real Sinergym results remain to be measured on the GPU host.**

Start with [GPU_RUNBOOK.md](docs/safe_hvac/GPU_RUNBOOK.md). Give the other machine's Codex [CODEX_REMOTE.md](docs/safe_hvac/CODEX_REMOTE.md). Read [METHOD_AND_CHANGES.md](docs/safe_hvac/METHOD_AND_CHANGES.md) for equations, corrections, differences from the reference, and experimental scope.

## Entry points

Run from the project root after setting up the GPU environment:

```bash
python -m safe_hvac.doctor
python -m pytest -q tests/safe_hvac
python -m safe_hvac.preflight --output results/safe_hvac/preflight
python -m safe_hvac.train --config configs/safe_hvac/smoke_s01.yaml --run-name smoke-barrier
python -m safe_hvac.train --config configs/safe_hvac/smoke_s01.yaml --algorithm ppo --run-name smoke-ppo
```

The doctor and unit tests do not start EnergyPlus. Preflight starts it without learning. The two smoke commands actually train and evaluate, so run them **on the GPU machine**. Smoke results establish integration only.

## Implementation

| File | Role |
|---|---|
| `safe_hvac/algorithm.py` | PPO + barrier/dual training; reuses SB3's collection and GAE |
| `safe_hvac/buffer.py` | Stores the actual physical next state for each recorded action |
| `safe_hvac/safety.py` | Physical labels, fixed features, margins, bounded class replay, horizon masks |
| `safe_hvac/environment.py` | Sinergym adapter, action scaling, physical metrics, episode boundaries |
| `safe_hvac/train.py` | Exact interaction budget, records, checkpoints, final evaluation |
| `safe_hvac/evaluate.py` | Loads the saved agent and freezes training normalization |
| `safe_hvac/report.py` | Checks experiment comparability and writes CSV/JSON results |
| `safe_hvac/legacy.py` | Corrected discrete BOPTEST agent for the three old variants |

The primary preset is S01: 5Zone, 15-minute control, baseline comfort bounds, fixed New York reference weather with seeded perturbations. Sinergym's default 5Zone registration measures temperature in **SPACE5-1 only**. Reports describe that measured zone; they do not establish safety in all five zones. The pilot covers the first winter week. Use `annual_s01.yaml` for an annual task after the pilot succeeds.

## Reference repositories and publishing

`references/srlnbc` is an isolated clone of the original paper repository. `HVAC_SAFE_RL_` contains corrected local legacy entry points. Nested clones and experiment data are ignored in the parent repository; the tracked source lock and patch recreate them:

```bash
python scripts/bootstrap_references.py
```

Publish `safe_hvac/`, the new configs/tests/docs/scripts, `references/sources.json`, `patches/`, `requirements-safe-hvac.txt`, `pytest.ini`, this README, and the ignore file. You can instead extract `dist/safe-hvac-ready.tar.gz` into a fresh directory and publish that directory. The bundle excludes large historical experiments and nested Git repositories. Do not install `srlnbc`'s old TensorFlow/Ray requirements into the new environment.

For Sinergym, use `python -m safe_hvac.train`; the historical `src/train.py` and shell launchers are separate legacy paths and do not select the new agent.
