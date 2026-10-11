"""Evaluate a saved final model with frozen training normalization."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from .algorithm import BarrierPPO, ReportingPPO
from .environment import make_vec_env
from .provenance import require_runtime, manifest
from .storage import RunStore, atomic_json


def model_class(config):
    return BarrierPPO if config["algorithm"] == "barrier_ppo" else ReportingPPO


def evaluate_bundle(bundle, store, config, device=None, checkpoint_step=None):
    bundle = Path(bundle)
    model = model_class(config).load(bundle / "model.zip", device=device or config["device"])
    results = []
    prefix = "" if checkpoint_step is None else f"checkpoint-{checkpoint_step}-"
    for seed in config["evaluation_seeds"]:
        vec = make_vec_env(config, seed, store.root / "simulator" / f"eval-{seed}",
                           f"eval-{seed}", bundle / "vecnormalize.pkl", training=False)
        try:
            before_mean = vec.obs_rms.mean.copy()
            before_var = vec.obs_rms.var.copy()
            before_count = float(vec.obs_rms.count)
            obs = vec.reset()
            done, steps, score = False, 0, 0.
            stop = config["evaluation"]["max_steps"]
            while not done and (stop is None or steps < stop):
                action, _ = model.predict(obs, deterministic=config["evaluation"]["deterministic"])
                next_obs, rewards, dones, infos = vec.step(action)
                record = dict(infos[0]["safe_hvac_record"], seed=seed, evaluation_step=steps + 1,
                              policy_observation=obs[0], predicted_normalized_action=action[0])
                store.write(f"{prefix}eval-{seed}-steps", record)
                score += float(rewards[0])
                steps += 1
                done = bool(dones[0])
                # DummyVecEnv resets automatically; use the summary from the completed episode.
                summary = infos[0].get("safe_hvac_episode")
                obs = next_obs
                if steps % 64 == 0:
                    check_storage(store.root, config["storage"]["max_run_gb"])
            if summary is None:
                summary = vec.venv.envs[0].summary()
            if not (np.array_equal(before_mean, vec.obs_rms.mean) and
                    np.array_equal(before_var, vec.obs_rms.var) and before_count == vec.obs_rms.count):
                raise AssertionError("Evaluation changed the training normalization statistics")
            result = dict(summary, seed=seed, score=score,
                          stopped_by_evaluation_limit=not done,
                          normalization_frozen=True)
            results.append(result)
            atomic_json(store.root / f"{prefix}eval-{seed}.json", result)
        finally:
            vec.close()
            # Checkpoint evaluation can occur many times. Retain compressed scientific
            # records and remove only this run's temporary checkpoint simulator workspace.
            if checkpoint_step is not None:
                shutil.rmtree(store.root / "simulator" / f"eval-{seed}")
    # Means describe independent seeded episodes. They do not select a model.
    aggregate = {key: {"mean": float(np.mean([r[key] for r in results])),
                       "std": float(np.std([r[key] for r in results], ddof=1)) if len(results) > 1 else None}
                 for key in ("score", "energy_kwh", "mean_power_kw", "violation_rate", "degree_hours",
                             "occupied_degree_hours", "duration_hours", "max_deviation_c", "cumulative_cost")}
    atomic_json(store.root / f"{prefix}evaluation.json", dict(per_seed=results, aggregate=aggregate,
                                                    model_selection="final" if checkpoint_step is None else "fixed checkpoint",
                                                    checkpoint_step=checkpoint_step, bundle=str(bundle.resolve())))
    check_storage(store.root, config["storage"]["max_run_gb"])
    return results


def check_storage(root, limit_gb):
    total = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
    if total > limit_gb * 1024 ** 3:
        raise RuntimeError(f"Run exceeded {limit_gb} GiB storage budget; stopped with records preserved")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True, help="New evaluation directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = json.loads((Path(args.bundle) / "config.json").read_text())
    runtime = require_runtime(args.device or config["device"])
    store = RunStore(args.output)
    try:
        atomic_json(store.root / "metadata.json", manifest(config, runtime=runtime))
        evaluate_bundle(args.bundle, store, config, args.device)
    finally:
        store.close()


if __name__ == "__main__":
    main()
