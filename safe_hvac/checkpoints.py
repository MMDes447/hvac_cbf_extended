"""Atomic, bounded bundles: policy + barrier + optimizers + normalization + RNG."""
from pathlib import Path
from contextlib import contextmanager
import random
import shutil
import numpy as np
import torch

from .storage import atomic_json


@contextmanager
def preserve_rng():
    """Model construction/evaluation must not reseed the live training process."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def save_bundle(model, vec_env, path, config, metadata):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    temporary.mkdir(parents=True)
    model.save(temporary / "model.zip")
    vec_env.save(temporary / "vecnormalize.pkl")
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, temporary / "rng.pt")
    atomic_json(temporary / "config.json", config)
    atomic_json(temporary / "metadata.json", {**metadata, "timesteps": model.num_timesteps,
                                             "resume_semantics": "new simulator episode; no EnergyPlus state snapshot"})
    temporary.rename(path)
    return path


def restore_rng(path):
    # Load only your own trusted bundles; PyTorch/SB3/pickle checkpoints contain code.
    state = torch.load(Path(path) / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume RNG has a different number of CUDA devices")
        torch.cuda.set_rng_state_all(state["cuda"])


def prune_checkpoints(root, keep):
    checkpoints = sorted(Path(root).glob("step-*"))
    for path in checkpoints[:-keep]:
        if path.is_dir() and (path / "model.zip").is_file():
            shutil.rmtree(path)
