"""Run on the GPU machine: train, checkpoint, then evaluate the saved final agent."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .checkpoints import save_bundle, restore_rng, prune_checkpoints, preserve_rng
from .config import load_config, safe_run_name
from .environment import make_vec_env
from .evaluate import evaluate_bundle, model_class, check_storage
from .provenance import manifest, require_runtime
from .safety import SafetySpec
from .storage import RunStore, atomic_json


class RecordsCallback(BaseCallback):
    def __init__(self, store, config):
        super().__init__()
        self.store, self.config = store, config

    def _on_step(self):
        stride = self.config["storage"]["record_every_steps"]
        for env_index, info in enumerate(self.locals["infos"]):
            if self.num_timesteps % stride == 0:
                record = dict(info["safe_hvac_record"], timestep=self.num_timesteps, env_index=env_index,
                              sampled_policy_action=self.locals["actions"][env_index],
                              policy_observation=self.model._last_obs[env_index],
                              old_log_probability=float(self.locals["log_probs"][env_index].item()),
                              old_value=float(self.locals["values"][env_index].item()))
                self.store.write("train-steps", record)
            if "safe_hvac_episode" in info:
                self.store.write("train-episodes", dict(info["safe_hvac_episode"], timestep=self.num_timesteps))
        if self.num_timesteps % 64 == 0:
            check_storage(self.store.root, self.config["storage"]["max_run_gb"])
        return True


def build_model(config, vec):
    options = dict(config["ppo"], seed=config["agent_seed"], device=config["device"], verbose=1)
    if config["algorithm"] == "barrier_ppo":
        options.update(safety_config=config["barrier"], feature_dim=SafetySpec(**config["safety"]).feature_dim,
                       barrier_pretrain_steps=config["barrier_pretrain_steps"])
    return model_class(config)("MlpPolicy", vec, **options)


def run(config, output, resume=None):
    runtime = require_runtime(config["device"])
    old_config = None
    if resume:
        resume = Path(resume).resolve()
        old_config = json.loads((resume / "config.json").read_text())
        for key in ("algorithm", "ppo", "barrier", "safety", "environment", "agent_seed", "env_seed",
                    "barrier_pretrain_steps"):
            if old_config[key] != config[key]:
                raise ValueError(f"Resume changed {key}; use the saved configuration for continuation")
    store = RunStore(output)
    vec, model = None, None
    try:
        atomic_json(store.root / "config.json", config)
        vec = make_vec_env(config, config["env_seed"], store.root / "simulator" / "train", "train",
                           resume / "vecnormalize.pkl" if resume else None)
        metadata = manifest(config, vec, runtime)
        metadata["resume_from"] = str(resume) if resume else None
        atomic_json(store.root / "metadata.json", metadata)
        model = (model_class(config).load(resume / "model.zip", env=vec, device=config["device"])
                 if resume else build_model(config, vec))
        if resume:
            restore_rng(resume)
        start = model.num_timesteps if resume else 0
        target = start + config["total_timesteps"]
        updates = []

        def report(agent, metrics):
            store.write("train-updates", metrics)
            updates.append(metrics)
            # Bounded list is sufficient for acceptance checks; full history is streamed.
            if len(updates) > 4:
                del updates[0]
            if agent.num_timesteps % config["storage"]["checkpoint_every_steps"] == 0:
                save_bundle(agent, vec, store.root / "checkpoints" / f"step-{agent.num_timesteps:012d}", config, metadata)
                prune_checkpoints(store.root / "checkpoints", config["storage"]["keep_checkpoints"])
                interval = config["evaluation"]["checkpoint_every_steps"]
                if interval and agent.num_timesteps % interval == 0:
                    with preserve_rng():
                        evaluate_bundle(store.root / "checkpoints" / f"step-{agent.num_timesteps:012d}",
                                        store, config, checkpoint_step=agent.num_timesteps)
            check_storage(store.root, config["storage"]["max_run_gb"])

        model.reporter = report
        model.learn(total_timesteps=config["total_timesteps"], callback=RecordsCallback(store, config),
                    reset_num_timesteps=not bool(resume), progress_bar=False)
        if model.num_timesteps != target:
            raise AssertionError(f"Interaction budget mismatch: {model.num_timesteps} != {target}")
        if not updates or not any(m["train/actor_head_parameter_delta"] > 0 for m in updates):
            raise AssertionError("Final actor updates did not change its action head parameters")
        if config["algorithm"] == "barrier_ppo" and not any(m["safety/barrier_parameter_delta"] > 0 for m in updates):
            raise AssertionError("Barrier updates did not change parameters")
        final = save_bundle(model, vec, store.root / "final", config, metadata)
        atomic_json(store.root / "train-last-episode.json", vec.venv.envs[0].summary())
        # The simulator is closed before loading the saved agent for final evaluation.
        vec.close()
        vec = None
        results = evaluate_bundle(final, store, config)
        atomic_json(store.root / "status.json", dict(status="complete", start_timesteps=start,
                                                    final_timesteps=model.num_timesteps,
                                                    added_interactions=config["total_timesteps"],
                                                    policy_updates_verified=True,
                                                    evaluation_seeds=[r["seed"] for r in results],
                                                    size_bytes=check_storage(store.root, config["storage"]["max_run_gb"])))
        print(f"Completed {model.num_timesteps} cumulative steps. Results: {store.root}")
    except BaseException as error:
        atomic_json(store.root / "status.json", dict(status="failed", error=repr(error),
                                                    traceback=traceback.format_exc(),
                                                    timesteps=getattr(model, "num_timesteps", None)))
        raise
    finally:
        if vec is not None:
            vec.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="results/safe_hvac")
    parser.add_argument("--run-name")
    parser.add_argument("--algorithm", choices=["ppo", "barrier_ppo"])
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--agent-seed", type=int)
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--resume", help="Trusted bundle; total_timesteps adds interactions in a new simulator episode")
    args = parser.parse_args()
    cfg = load_config(args.config, dict(algorithm=args.algorithm, device=args.device, total_timesteps=args.steps,
                                       agent_seed=args.agent_seed, env_seed=args.env_seed))
    default_name = f"{cfg['algorithm']}-a{cfg['agent_seed']}-e{cfg['env_seed']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    run(cfg, Path(args.output_root) / safe_run_name(args.run_name or default_name), args.resume)


if __name__ == "__main__":
    main()
