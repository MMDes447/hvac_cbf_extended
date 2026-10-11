# GPU runbook

## 1. Move the source

Push the prepared source files to your GitHub repository, then clone that repository on the GPU machine. Alternatively, extract `safe-hvac-ready.tar.gz`, initialize a repository in the extracted directory, and publish it. The preparation workspace has a read-only empty Git placeholder, so no parent commit or push was made here.

Work from the project root. Reference repositories are optional for Sinergym execution:

```bash
python scripts/bootstrap_references.py
```

This recreates the two pinned clones and applies the tracked legacy patch without resetting existing work. `--include-sinergym` also fetches the pinned simulator source for inspection. The original `srlnbc` environment is separate and is not needed for the new trainer.

## 2. Install the GPU environment

Target: **Ubuntu 24.04, Python 3.12, Sinergym 3.12.0, SB3 2.8.0, EnergyPlus 25.1.0**. The pinned Sinergym [Dockerfile](https://github.com/ugr-sail/sinergym/blob/v3.12.0/Dockerfile) uses the EnergyPlus `v25.1.0-WithDSOASpaceListFixes` installer. Use that release, rather than the old project README's 25.2 claim. Consult [Sinergym installation](https://ugr-sail.github.io/sinergym/compilation/v3.12.0/pages/installation.html) and [PyTorch's GPU installation selector](https://pytorch.org/get-started/locally/).

Check the GPU/driver with `nvidia-smi`. Choose a CUDA wheel index supported by the hardware and driver using the PyTorch selector. Then run, replacing the example index if necessary:

```bash
bash scripts/setup_safe_hvac_gpu.sh https://download.pytorch.org/whl/cu128
source .venv-safe-hvac/bin/activate
```

The script installs GPU PyTorch first, then the small pinned requirements file. It creates `results/setup/python-lock.txt` containing the actual resolved packages. It does not install EnergyPlus or modify system Python. Use that lock and the same GPU wheel source for subsequent replications. Do not install the historical full `requirements.txt` into this environment.

Install EnergyPlus separately. The exact Linux installer referenced by Sinergym 3.12 is:

```text
https://github.com/NREL/EnergyPlus/releases/download/v25.1.0-WithDSOASpaceListFixes/EnergyPlus-25.1.0-1c11a3d85f-Linux-Ubuntu24.04-x86_64.sh
```

Download it on the GPU host and follow its installation prompts. A user-owned installation directory is suitable if supported by the installer; inspect its help before unattended use. Set the variables to the actual installation directory, for example:

```bash
export EPLUS_PATH=/usr/local/EnergyPlus-25-1-0
export PYTHONPATH="$EPLUS_PATH${PYTHONPATH:+:$PYTHONPATH}"
```

Keep those exports in the shell running the commands. If the host OS differs, use a compatible container based on the pinned Sinergym source; verify NVIDIA container access before running experiments. The doctor rejects wrong simulator/package versions and unavailable CUDA.

## 3. Check software and numerical regressions

```bash
python -m safe_hvac.doctor --config configs/safe_hvac/smoke_s01.yaml
python -m pytest -q tests/safe_hvac
```

These commands do not train a building agent. Tests inject small numerical fixtures and execute isolated optimizer updates. They check safety weighting/gradients, GAE boundaries, positive margins, bounded replay, physical records/action mapping, pretraining, and complete checkpoint reload. They are not evidence that EnergyPlus runs or that a learned policy is safe.

## 4. Check real Sinergym transitions

```bash
python -m safe_hvac.preflight --config configs/safe_hvac/smoke_s01.yaml \
  --output results/safe_hvac/preflight-01 --steps 8
```

This launches EnergyPlus, applies midpoint setpoints, and saves eight actual transitions with zero optimizer steps. Check `status.json`, the asset/variable manifest, temperatures, clock advance, and native actions. An observation/info timeout, missing variable, stale simulator clock, or unsupported action stops the run. Fix that failure before training.

## 5. Train the two smoke agents

```bash
python -m safe_hvac.train --config configs/safe_hvac/smoke_s01.yaml \
  --algorithm barrier_ppo --run-name smoke-barrier-01
python -m safe_hvac.train --config configs/safe_hvac/smoke_s01.yaml \
  --algorithm ppo --run-name smoke-ppo-01
python -m safe_hvac.report \
  --runs results/safe_hvac/smoke-ppo-01 results/safe_hvac/smoke-barrier-01 \
  --output results/safe_hvac/smoke-comparison-01
```

Each smoke run collects exactly **256 training interactions** in four 64-step rollouts. It updates the policy, saves the final bundle, reloads it, and evaluates the single configured test seed for 64 steps with frozen training normalization. It crosses an episode boundary in the two-day simulation. This checks that the agent is actually updated and used in Sinergym. Its short results cannot establish superiority.

Accept the smoke run only when:

- `status.json` says `complete`, `added_interactions=256`, and `policy_updates_verified=true`.
- `train-updates.jsonl.gz` shows positive actor action-head parameter changes (separately from critic changes), finite losses/gradients, and positive barrier parameter changes for barrier PPO.
- The saved model loads successfully; final evaluation ran from that saved model.
- Every `eval-*.json` says `normalization_frozen=true` and reports the expected evaluation limit/physical duration.
- Raw physical clock increments match the control interval; the last transition is retained and no repeated end-of-simulation observation is learned from.
- Temperature bounds and the source mapping identify SPACE5-1; no all-zone claim appears in the report.

## 6. Run a matched winter pilot

Once both smoke runs pass:

```bash
bash scripts/run_safe_hvac_pilot.sh pilot-01
```

This runs plain PPO and corrected barrier PPO for three independently trained agent seeds. Each pair shares its training environment seed. Every run collects **32768 training interactions**, uses the same PPO architecture/reward/task and test seeds, and evaluates the final agent on all three seeds. The default task is a seven-day winter episode, with a fresh seeded weather realization each episode. No best-of-three episode is chosen.

The comparison is saved to `results/safe_hvac/pilot-01/comparison/per_seed.csv` and `summary.json`. Inspect score, kWh, mean power, violation fraction, degree-hours, maximum deviation, and barrier classification errors. Low barrier loss alone is not a certificate. Three training replications are a pilot; they do not implement the paper's 30-run reliability study.

Evaluate extensions as separate, named ablations after the primary pilot:

```bash
python -m safe_hvac.train --config configs/safe_hvac/multistep_s01.yaml --run-name multistep-a2-e1
python -m safe_hvac.train --config configs/safe_hvac/pretrain_s01.yaml --run-name pretrain-a2-e1
python -m safe_hvac.train --config configs/safe_hvac/occupancy_s01.yaml --run-name occupancy-a2-e1
```

Keep seeds and budgets matched. Occupancy changes both the comfort reward and safety bounds, so compare it against PPO under the same occupancy config. Barrier pretraining uses 2048 interactions **inside** the 32768 total budget, leaving 30720 for policy training. The multi-step variant uses 10 horizons; this is inspired by the original repository, not a novel contribution by itself.

The primary agent uses raw transition residuals and the unscaled actor loss. `reference_style_s01.yaml` additionally enables the original repository's residual standardization and actor scaling. Treat that as a separate combined variant; use single-switch ablations to attribute improvements.

For annual coverage use `configs/safe_hvac/annual_s01.yaml`. Do not mix annual and winter-pilot scores. The full LaTeX study's 48 situations, five baseline algorithms, HPO and 30 replications are a later study; this package currently compares PPO and barrier PPO on explicit continuous Sinergym tasks.

## 7. Results and storage

Each run creates an exclusive directory:

```text
config.json                     resolved settings, seeds, interaction budget
metadata.json                   versions, code hashes, physical variables, asset hashes
status.json                     completion/failure and actual interaction counts
train-steps.jsonl.gz             normalized input, raw physical observation, sampled/executed actions,
                                reward decomposition, energy and comfort diagnostics
train-episodes.jsonl.gz          completed physical episode summaries
train-last-episode.json          final episode/partial-episode summary
train-updates.jsonl.gz           policy/barrier losses, parameter changes, dual, class errors
checkpoints/step-*/              last two update checkpoints by default
final/model.zip                 policy, critic, barrier, dual, optimizers, bounded replay
final/vecnormalize.pkl           training observation statistics
final/rng.pt                    Python/NumPy/Torch/CUDA RNG states
final/config.json,metadata.json bundle provenance
eval-<seed>-steps.jsonl.gz       final evaluation transitions
eval-<seed>.json,evaluation.json per-seed metrics and aggregate; all seeds retained
simulator/                      last two raw EnergyPlus episode directories per workspace
```

All scientific records are streamed with gzip **outside** Sinergym's rotating episode directories. `max_ep_store=2` controls raw simulator directory retention; it has no model-selection role. Checkpoints keep the latest two plus the final bundle. A 10 GiB run budget is checked every 64 interactions and after updates/evaluation; it stops an expanding run, rather than silently deleting scientific records. It is a periodic guard, not a hard filesystem quota. Evaluation has one separate workspace per seed, so raw storage also grows with the number of seeds. After verifying and backing up a completed run, its entire `simulator/` directory is disposable; model evaluation requires EnergyPlus assets and the final bundle, not these raw outputs.

For the paper's detailed timestep analysis leave `record_every_steps=1`. A larger stride reduces log volume but does not retain every transition. CSV/JSON summaries and streamed step records are preferable to collecting all DataFrames in memory.

Optional checkpoint evaluation is disabled in the smoke/pilot to control cost. Set `evaluation.checkpoint_every_steps` to a multiple of the saved-checkpoint interval (e.g. 8192), **before the study**. This evaluates the same configured test seeds at each checkpoint, writes step-tagged records and removes only temporary raw checkpoint-evaluation workspaces. It preserves the training RNG and normalization. If these seeds inform model/hyperparameter selection, treat them as validation seeds and define a separate untouched final test set in a new frozen configuration.

## 8. Re-evaluate and continue

Re-evaluate a saved final model into a new directory:

```bash
python -m safe_hvac.evaluate --bundle results/safe_hvac/smoke-barrier-01/final \
  --output results/safe_hvac/re-eval-smoke-01
```

Resume a trusted checkpoint in a **new run directory**, using unchanged task/model/safety settings. `--steps` adds interactions and must be divisible by `n_steps`:

```bash
python -m safe_hvac.train --config configs/safe_hvac/train_s01.yaml \
  --resume results/safe_hvac/barrier-run/final --steps 32768 --run-name barrier-continued-01
```

Model, barrier, dual, optimizers, bounded label memory, normalization, and RNG states are restored. EnergyPlus starts a new episode; its internal physical state and weather RNG progress are not snapshotted. Continuation is **not** a bit-for-bit simulator resume or an independent replication. Metadata records the parent bundle and cumulative timestep count. The comparison tool rejects resumed runs unless you explicitly analyze cumulative budgets separately.

## 9. What to return from the GPU host

Return the setup lock, command/configuration used, `status.json`, `metadata.json`, final bundles, `evaluation.json`, and the comparison directory. Keep compressed training/evaluation records for analysis and error diagnosis. Report actual failures and measured performance; do not infer building results from the local unit tests.
