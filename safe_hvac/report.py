"""Compare complete matched runs and write a compact CSV/JSON result table."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import statistics
from .storage import atomic_json


def compare(runs, output):
    rows, signature = [], None
    if len({str(Path(p).resolve()) for p in runs}) != len(runs):
        raise ValueError("The same run was supplied more than once")
    for root in map(Path, runs):
        status = json.loads((root / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError(f"Run did not complete: {root}")
        cfg = json.loads((root / "config.json").read_text())
        metadata = json.loads((root / "metadata.json").read_text())
        evaluation = json.loads((root / "evaluation.json").read_text())
        method = "ppo"
        if cfg["algorithm"] == "barrier_ppo":
            method_config = dict(barrier=cfg["barrier"], pretrain=cfg["barrier_pretrain_steps"])
            fingerprint = hashlib.sha256(json.dumps(method_config, sort_keys=True).encode()).hexdigest()[:8]
            method = f"barrier_ppo-H{cfg['barrier']['horizons']}-P{cfg['barrier_pretrain_steps']}-{fingerprint}"
        comparable = dict(situation_id=cfg["situation_id"], environment=cfg["environment"], safety=cfg["safety"], evaluation=cfg["evaluation"],
                          evaluation_seeds=cfg["evaluation_seeds"], total_timesteps=cfg["total_timesteps"],
                          ppo=cfg["ppo"], assets=metadata.get("assets"), software=metadata["software"]["packages"],
                          energyplus=metadata["software"].get("energyplus"))
        if metadata.get("resume_from"):
            raise ValueError("Automatic comparison excludes resumed runs; explicitly account for cumulative budgets")
        if signature is None:
            signature = comparable
        elif signature != comparable:
            raise ValueError("Runs differ in task, safety scope, test set, budget, PPO settings, assets or software")
        for item in evaluation["per_seed"]:
            rows.append(dict(run=str(root.resolve()), run_name=root.name, method=method, algorithm=cfg["algorithm"], agent_seed=cfg["agent_seed"],
                             env_seed=cfg["env_seed"], evaluation_seed=item["seed"],
                             **{k: item[k] for k in ("score", "steps", "duration_hours", "energy_kwh",
                               "mean_power_kw", "violation_rate", "degree_hours", "occupied_degree_hours",
                               "max_deviation_c", "cumulative_cost")}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Average test seeds within a trained model first; uncertainty is across training runs.
    summaries = []
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        by_run = {name: [r for r in selected if r["run"] == name] for name in {r["run"] for r in selected}}
        result = dict(method=method, algorithm=selected[0]["algorithm"], training_runs=len(by_run))
        for metric in ("score", "energy_kwh", "mean_power_kw", "violation_rate", "degree_hours",
                       "max_deviation_c", "cumulative_cost"):
            averages = [statistics.mean(r[metric] for r in items) for items in by_run.values()]
            result[metric] = dict(mean=statistics.mean(averages),
                                  median=statistics.median(averages),
                                  std_across_training_runs=statistics.stdev(averages) if len(averages) > 1 else None)
        summaries.append(result)
    paired = []
    methods = sorted({r["method"] for r in rows if r["algorithm"] == "barrier_ppo"})
    baseline = [r for r in rows if r["algorithm"] == "ppo"]
    if baseline:
        for method in methods:
            selected = [r for r in rows if r["method"] == method]
            for env_seed in sorted({r["env_seed"] for r in selected}):
                first = [r for r in selected if r["env_seed"] == env_seed]
                other = [r for r in baseline if r["env_seed"] == env_seed]
                if not other:
                    raise ValueError(f"Missing matched PPO environment seed {env_seed}")
                if len({r["run"] for r in first}) != 1 or len({r["run"] for r in other}) != 1:
                    raise ValueError("Paired report requires one trained model per method/environment seed")
                paired.append(dict(method=method, env_seed=env_seed,
                                   **{metric: statistics.mean(r[metric] for r in first) - statistics.mean(r[metric] for r in other)
                                      for metric in ("score", "energy_kwh", "violation_rate", "degree_hours", "cumulative_cost")}))
    atomic_json(output / "paired_differences.json", dict(sign="barrier minus PPO; positive score is better, positive cost/violation is worse",
                                                        pairs=paired))
    atomic_json(output / "summary.json", dict(summaries=summaries, comparability=signature,
                                             note="Test seeds are repeated measures, not independent trained agents"))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
