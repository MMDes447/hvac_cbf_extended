"""Sinergym adapter: physical safety records before policy normalization."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from pathlib import Path
import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .safety import SafetySpec
from .reward import PhysicalComfortReward


class SafetyRecordWrapper(gym.Wrapper):
    def __init__(self, env, spec: SafetySpec, steps_per_hour=4,
                 normalize_actions=True, max_episode_steps=None):
        super().__init__(env)
        self.safety_spec = spec
        self.names = list(env.unwrapped.observation_variables)
        missing = set(spec.required_variables()) - set(self.names)
        if missing:
            raise ValueError(f"Environment lacks safety variables: {sorted(missing)}")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("The Sinergym runner requires continuous Box actions")
        self.native_action_space = env.action_space
        if not np.isfinite(env.action_space.low).all() or not np.isfinite(env.action_space.high).all():
            raise ValueError("Action bounds must be finite")
        self.normalize_actions = normalize_actions
        if normalize_actions:
            self.action_space = gym.spaces.Box(-1., 1., env.action_space.shape, dtype=np.float32)
        self.steps_per_hour = steps_per_hour
        physical_horizon = getattr(getattr(env.unwrapped, "model", None), "timestep_per_episode", None)
        # Sinergym's reset already consumes the first observation. Avoid its
        # synthetic end-of-simulation step with a repeated observation/reward.
        limits = [int(x) for x in (max_episode_steps,
                                  physical_horizon - 1 if physical_horizon else None) if x is not None]
        self.limit = min(limits) if limits else None
        if self.limit is not None and self.limit < 1:
            raise ValueError("Episode must contain at least one physical transition")
        self.history = deque(maxlen=spec.history_length)
        self.episode = 0

    def _assessment(self, observation):
        if np.asarray(observation).shape != (len(self.names),):
            raise ValueError("Observation shape does not match physical variable names")
        raw = dict(zip(self.names, np.asarray(observation).tolist(), strict=True))
        return raw, self.safety_spec.assess(raw)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw, assessment = self._assessment(obs)
        self.history.clear()
        for _ in range(self.safety_spec.history_length):
            self.history.append(assessment["features"].copy())
        self.current_features = np.concatenate(self.history)
        self.current_label = assessment["label"]
        self.elapsed = info.get("time_elapsed(hours)")
        if hasattr(self.env.unwrapped, "model") and self.elapsed is None:
            raise RuntimeError("Sinergym reset returned no physical clock; EnergyPlus initialization may have failed")
        self.steps = 0
        self.episode += 1
        self.totals = dict(steps=0, duration_hours=0., energy_kwh=0.,
                           violating_steps=0, degree_hours=0.,
                           occupied_degree_hours=0., occupied_zone_hours=0.,
                           occupied_violating_zone_hours=0., cumulative_reward=0., cumulative_cost=0.,
                           cumulative_energy_term=0., cumulative_comfort_term=0.,
                           max_deviation_c=0., worst_step_cost=0.)
        return obs, info

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Invalid policy action")
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        native = (self.native_action_space.low + (clipped + 1) / 2 *
                  (self.native_action_space.high - self.native_action_space.low)
                  if self.normalize_actions else clipped)
        state, label = self.current_features.copy(), self.current_label
        obs, reward, terminated, truncated, info = self.env.step(native)
        info = dict(info)
        raw, assessment = self._assessment(obs)
        new_elapsed = info.get("time_elapsed(hours)")
        if hasattr(self.env.unwrapped, "model") and new_elapsed is None:
            raise RuntimeError("Sinergym step returned no physical clock")
        dt = ((float(new_elapsed) - float(self.elapsed))
              if new_elapsed is not None and self.elapsed is not None else 1 / self.steps_per_hour)
        if not np.isfinite(dt) or dt <= 0:
            raise RuntimeError("Sinergym returned no physical time advance; check run-period horizon")
        self.elapsed = new_elapsed
        self.steps += 1
        if self.limit is not None and self.steps >= self.limit and not terminated:
            truncated = True
            info["safe_hvac_horizon_truncation"] = True
        self.history.append(assessment["features"].copy())
        next_state = np.concatenate(self.history)
        self.current_features, self.current_label = next_state, assessment["label"]
        if not np.isfinite(reward):
            raise FloatingPointError("Nonfinite environment reward")
        deviation, occupied = assessment["deviation_c"], assessment["occupied"]
        record = dict(episode=self.episode, episode_step=self.steps, reward=float(reward),
                      duration_hours=dt, energy_kwh=assessment["power_w"] * dt / 1000,
                      violation=bool(np.any(deviation > 0)),
                      degree_hours=float(deviation.sum()) * dt,
                      occupied_degree_hours=float(deviation[occupied].sum()) * dt,
                      occupied_zone_hours=float(occupied.sum()) * dt,
                      occupied_violating_zone_hours=float(((deviation > 0) & occupied).sum()) * dt,
                      raw_observation=raw, requested_action=action, executed_action=native,
                      temperatures_c=assessment["temperatures_c"], bounds_c=assessment["bounds_c"],
                      max_deviation_c=float(deviation.max()),
                      reward_components={key: info[key] for key in ("energy_term", "comfort_term",
                          "energy_penalty", "comfort_penalty", "total_power_demand",
                          "total_temperature_violation", "reward_weight") if key in info},
                      terminated=bool(terminated), truncated=bool(truncated))
        for key in ("duration_hours", "energy_kwh", "degree_hours", "occupied_degree_hours",
                    "occupied_zone_hours", "occupied_violating_zone_hours"):
            self.totals[key] += record[key]
        self.totals["steps"] += 1
        self.totals["violating_steps"] += int(record["violation"])
        self.totals["cumulative_reward"] += float(reward)
        self.totals["cumulative_cost"] -= float(reward)
        self.totals["cumulative_energy_term"] += float(info.get("energy_term", 0))
        self.totals["cumulative_comfort_term"] += float(info.get("comfort_term", 0))
        self.totals["max_deviation_c"] = max(self.totals["max_deviation_c"], record["max_deviation_c"])
        self.totals["worst_step_cost"] = max(self.totals["worst_step_cost"], -float(reward))
        info["safe_hvac_record"] = record
        info["safe_hvac_transition"] = dict(state=state, next_state=next_state,
                                             label=label, next_label=self.current_label,
                                             episode_end=bool(terminated or truncated))
        if terminated or truncated:
            info["safe_hvac_episode"] = self.summary()
        return obs, reward, terminated, truncated, info

    def summary(self):
        result = dict(self.totals)
        result["episode"] = self.episode
        result["violation_rate"] = result["violating_steps"] / max(result["steps"], 1)
        result["mean_power_kw"] = result["energy_kwh"] / max(result["duration_hours"], 1e-12)
        return result


@contextmanager
def _construction_directory(path):
    """Sinergym chooses absolute workspace paths at construction."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def make_raw_env(config, seed, directory, name):
    import sinergym  # Registers the pinned environment IDs.
    spec = SafetySpec(**config["safety"])
    settings = config["environment"]
    kwargs = dict(env_name=name, seed=int(seed), max_ep_store=int(settings["max_ep_store"]),
                  building_config=settings["building_config"])
    if settings.get("weather_file"):
        kwargs["weather_files"] = settings["weather_file"]
    # Sinergym's reward reads physical observations; keep it below VecNormalize.
    kwargs["reward"] = PhysicalComfortReward
    kwargs["reward_kwargs"] = dict(safety=config["safety"], **settings["reward"])
    with _construction_directory(directory):
        env = gym.make(settings["id"], **kwargs)
    try:
        return SafetyRecordWrapper(env, spec,
                                   steps_per_hour=settings["building_config"]["timesteps_per_hour"],
                                   normalize_actions=True,
                                   max_episode_steps=settings.get("max_episode_steps"))
    except Exception:
        env.close()
        raise


def make_vec_env(config, seed, directory, name, normalization_path=None, training=True):
    raw = make_raw_env(config, seed, directory, name)
    try:
        vec = DummyVecEnv([lambda: raw])
        if normalization_path:
            vec = VecNormalize.load(str(normalization_path), vec)
        else:
            vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.)
        vec.training = training
        vec.norm_reward = False
        return vec
    except Exception:
        raw.close()
        raise
