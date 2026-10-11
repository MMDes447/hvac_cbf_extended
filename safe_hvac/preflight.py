"""GPU host only: verify real physical transitions; no optimizer/training steps."""
import argparse
from pathlib import Path
import numpy as np
from .config import load_config
from .environment import make_vec_env
from .provenance import manifest, require_runtime
from .storage import RunStore, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/safe_hvac/smoke_s01.yaml")
    parser.add_argument("--output", default="results/safe_hvac/preflight")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("Preflight steps must be positive")
    config = load_config(args.config)
    runtime = require_runtime(config["device"])
    store = RunStore(args.output)
    vec = None
    try:
        vec = make_vec_env(config, config["env_seed"], store.root / "simulator", "preflight")
        atomic_json(store.root / "metadata.json", manifest(config, vec, runtime))
        obs = vec.reset()
        for step in range(args.steps):
            # Midpoint of physical setpoint ranges; no learned policy involved.
            action = np.zeros((1, *vec.action_space.shape), np.float32)
            obs, reward, done, info = vec.step(action)
            record = info[0]["safe_hvac_record"]
            transition = info[0]["safe_hvac_transition"]
            assert transition["state"].shape == transition["next_state"].shape
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            store.write("steps", dict(record, preflight_step=step + 1))
        atomic_json(store.root / "status.json", dict(status="complete", physical_transitions=args.steps,
                                                     optimizer_steps=0))
        print(f"Physical preflight passed. Records: {store.root}")
    finally:
        if vec is not None:
            vec.close()
        store.close()


if __name__ == "__main__":
    main()
