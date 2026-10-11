"""Small, strict experiment configuration without a Hydra dependency."""
from copy import deepcopy
from pathlib import Path
import re
import yaml

from .safety import SafetySpec


DEFAULTS = {
    "situation_id": "S01", "study_stage": "pilot",
    "algorithm": "barrier_ppo", "device": "cuda", "total_timesteps": 32768,
    "agent_seed": 2, "env_seed": 1, "evaluation_seeds": [6767, 690, 2845],
    "barrier_pretrain_steps": 0,
    "environment": {
        "id": "Eplus-5zone-mixed-continuous-stochastic-v1",
        "weather_file": "USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw",
        "max_ep_store": 2, "max_episode_steps": None,
        "building_config": {"timesteps_per_hour": 4, "runperiod": [1, 1, 1991, 7, 1, 1991]},
        "reward": {"energy_weight": 0.25, "lambda_energy": 0.00001, "lambda_temperature": 1.0},
    },
    "safety": {"temperature_variables": ["air_temperature"], "winter_bounds": [21., 24.],
               "summer_bounds": [23., 26.], "summer_start": [6, 1], "summer_end": [9, 30],
               "occupancy_variables": [], "occupancy_threshold": 0.5,
               "unoccupied_bounds": [15., 30.], "label_guard_band_c": 0.2,
               "extra_features": {"outdoor_temperature": [15., 20.],
                                  "heating_setpoint": [22., 10.], "cooling_setpoint": [26., 10.]},
               "energy_variable": "HVAC_electricity_demand_rate", "history_length": 1},
    "ppo": {"learning_rate": 0.0003, "n_steps": 512, "batch_size": 64, "n_epochs": 10,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
            "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": 0.03,
            "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}},
    "barrier": {"hidden_sizes": [64, 64], "learning_rate": 0.001, "epsilon": 0.01,
                "contraction": 0.1, "discount": 0.99, "horizons": 1,
                "updates_per_rollout": 10, "batch_size": 128, "replay_capacity": 8192,
                "anchor": 0.1, "anchor_weight": 0.01, "initial_dual": 1.0,
                "dual_learning_rate": 0.001, "max_dual": 10., "violation_budget": 0.,
                "standardize_residual": False, "scale_actor_loss": False},
    "storage": {"checkpoint_every_steps": 4096, "keep_checkpoints": 2,
                "max_run_gb": 10., "record_every_steps": 1},
    "evaluation": {"max_steps": None, "deterministic": True, "checkpoint_every_steps": 0},
}


def _merge(base, changes, path=""):
    if not isinstance(changes, dict):
        raise TypeError(f"{path or 'config'} must be a mapping")
    for key, value in changes.items():
        location = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown configuration key: {location}")
        if location in ("safety.extra_features", "ppo.policy_kwargs"):
            base[key] = deepcopy(value)
        elif isinstance(base[key], dict):
            _merge(base[key], value, location)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(path, overrides=None, _ancestors=()):
    path = Path(path).resolve()
    if path in _ancestors:
        raise ValueError("Cyclic configuration inheritance")
    with path.open() as stream:
        data = yaml.safe_load(stream) or {}
    parent = data.pop("extends", None)
    base = (load_config(path.parent / parent, _ancestors=(*_ancestors, path))
            if parent else deepcopy(DEFAULTS))
    config = _merge(base, data)
    if overrides:
        _merge(config, {k: v for k, v in overrides.items() if v is not None})
    validate(config)
    return config


def validate(cfg):
    import numpy as np
    SafetySpec(**cfg["safety"])
    if cfg["algorithm"] not in ("ppo", "barrier_ppo"):
        raise ValueError("algorithm must be ppo or barrier_ppo")
    if cfg["device"] not in ("cuda", "cpu", "auto") and not re.fullmatch(r"cuda:\d+", cfg["device"]):
        raise ValueError("Unsupported device")
    for key in ("agent_seed", "env_seed"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    seeds = cfg["evaluation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Evaluation seeds must be unique nonnegative integers")
    if cfg["env_seed"] in seeds:
        raise ValueError("Training and final evaluation environment seeds must differ")
    if cfg["env_seed"] == 0 or 0 in seeds:
        raise ValueError("Use positive environment seeds: Sinergym 3.12 treats constructor seed zero as unset")
    ppo, barrier, storage = cfg["ppo"], cfg["barrier"], cfg["storage"]
    for group, keys in ((ppo, ["n_steps", "batch_size", "n_epochs"]),
                        (barrier, ["horizons", "updates_per_rollout", "batch_size", "replay_capacity"]),
                        (storage, ["checkpoint_every_steps", "keep_checkpoints", "record_every_steps"])):
        for key in keys:
            if type(group[key]) is not int or group[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    n = ppo["n_steps"]  # Deliberately one EnergyPlus environment per process.
    if n < 2 or n % ppo["batch_size"] or ppo["batch_size"] < 2:
        raise ValueError("n_steps must be divisible by batch_size; both >= 2")
    total, pretrain = cfg["total_timesteps"], cfg["barrier_pretrain_steps"]
    if type(total) is not int or total <= 0 or total % n:
        raise ValueError("total_timesteps must be a positive multiple of n_steps (no silent overshoot)")
    if type(pretrain) is not int or pretrain < 0 or pretrain % n or pretrain >= total:
        raise ValueError("barrier_pretrain_steps must align with rollouts and leave policy training steps")
    if cfg["algorithm"] == "ppo" and pretrain:
        raise ValueError("PPO baseline does not have barrier pretraining")
    if storage["checkpoint_every_steps"] % n:
        raise ValueError("Checkpoints are saved after complete updates; align interval with n_steps")
    evaluation_interval = cfg["evaluation"]["checkpoint_every_steps"]
    if type(evaluation_interval) is not int or evaluation_interval < 0 or evaluation_interval % n:
        raise ValueError("Evaluation checkpoint interval must be zero or a multiple of n_steps")
    if evaluation_interval and evaluation_interval % storage["checkpoint_every_steps"]:
        raise ValueError("Evaluation interval must align with saved checkpoints")
    if not 0 < ppo["gamma"] <= 1 or not 0 <= ppo["gae_lambda"] <= 1:
        raise ValueError("Invalid PPO gamma/gae_lambda")
    if not 0 < barrier["contraction"] < 1 or not 0 < barrier["discount"] <= 1:
        raise ValueError("Invalid barrier contraction/discount")
    for group, keys in ((ppo, ["learning_rate", "clip_range", "max_grad_norm"]),
                        (barrier, ["learning_rate", "epsilon", "anchor", "dual_learning_rate", "max_dual"]),
                        (storage, ["max_run_gb"])):
        for key in keys:
            if not np.isfinite(group[key]) or group[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
    if barrier["anchor"] < barrier["epsilon"] or barrier["anchor_weight"] <= 0:
        raise ValueError("Anchor must exceed the margin and have a positive weight")
    if not 0 <= barrier["initial_dual"] <= barrier["max_dual"] or barrier["violation_budget"] < 0:
        raise ValueError("Invalid dual initialization/budget")
    env = cfg["environment"]
    if type(env["max_ep_store"]) is not int or not 1 <= env["max_ep_store"] <= 10:
        raise ValueError("max_ep_store must be between 1 and 10")
    for value in (env["max_episode_steps"], cfg["evaluation"]["max_steps"]):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Episode/evaluation limits must be positive integers or null")
    timestep = env["building_config"]["timesteps_per_hour"]
    if type(timestep) is not int or timestep < 1 or 60 % timestep:
        raise ValueError("EnergyPlus timestep must be a positive divisor of 60")
    from datetime import date
    period = env["building_config"]["runperiod"]
    if len(period) != 6:
        raise ValueError("runperiod is [start_day,start_month,start_year,end_day,end_month,end_year]")
    start, end = date(period[2], period[1], period[0]), date(period[5], period[4], period[3])
    if end < start:
        raise ValueError("Run period end precedes start")
    if not 0 <= env["reward"]["energy_weight"] <= 1:
        raise ValueError("Energy reward weight must be in [0,1]")


def safe_run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Run name must be a simple directory name of at most 100 characters")
    return value
